"""Пол пула фото-кандидатов, НЕЗАВИСИМЫЙ от VISUAL_DIRECTOR_MODE.

Реальный дефект связности, найденный при разборе внешней критики (07.09):
good_needed (сколько кандидатов реально оценивается до выбора победителя)
поднимался до DIRECTOR_MIN_POOL=8 ТОЛЬКО если передан director_score_fn, а
он строится только при VISUAL_DIRECTOR_MODE в (shadow, assist). Дефолт
реестра — "off". То есть на дефолтной конфигурации канала цикл подбора фото
останавливался на ПЕРВОМ кандидате, прошедшем все гейты (дедуп, размер,
relevance+вето+домен-гвард, резкость) — aesthetic_score/luma физически не из
чего было выбирать, а самый сильный локальный судья (SigLIP2 sentence_
relevance) не вызывался вообще.

Это не то же самое, что DIRECTOR_MIN_POOL: тот расширяет пул под ДОРОГОЙ
ensemble-скоринг (SigLIP2+Jina, ~2.5-3с/кандидат). BASE_MIN_POOL расширяет
пул под УЖЕ И ТАК считающиеся для каждого кандидата дешёвые оси (aesthetic,
luma, relevance, negative-veto) — они каждый один CLIP-проход на CPU (доли
секунды), не ensemble.
"""
import os
import io
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402


def _valid_png_bytes():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (100, 120, 140)).save(buf, "PNG")
    return buf.getvalue()


def _stub_pool(monkeypatch, tmp_path, n=9):
    png = _valid_png_bytes()
    # Defense in depth: conftest.py уже гасит OPENVERSE_ENABLED автоматически,
    # но этот тест хочет проверить РОВНО пул Pexels-кандидатов — явное
    # отключение здесь не зависит от того, не подвинется ли когда-нибудь
    # общая изоляция conftest.
    monkeypatch.setenv("OPENVERSE_ENABLED", "0")
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(png))
    monkeypatch.setattr(ps, "_pexels_search_photos", lambda q: [
        {"id": i, "alt": "", "url": f"https://www.pexels.com/photo/knight-{i}/",
         "src": {"large2x": f"http://x/{i}.jpg"}} for i in range(1, n + 1)])
    monkeypatch.setattr(ps, "disambiguate_search_query", lambda q: q)
    monkeypatch.setattr(ps, "is_relevant_candidate", lambda *a, **k: True)
    monkeypatch.setattr(ps, "image_sharpness_score", lambda p: 999.0)
    monkeypatch.setattr(ps, "measure_luma", lambda p: 0.4)
    monkeypatch.setattr(ps, "estimate_shot_size", lambda p: "medium")


class TestPoolFloorWithoutDirector:
    """director_score_fn=None — режим по умолчанию канала (VISUAL_DIRECTOR_MODE=off)."""

    def test_more_than_one_candidate_is_evaluated(self, tmp_path, monkeypatch):
        _stub_pool(monkeypatch, tmp_path)
        processed = []
        monkeypatch.setattr(ps, "aesthetic_score",
                            lambda p: processed.append(p) or 5.0)
        ps.pexels_photo("medieval sword", 0, used_ids=set(), used_hashes=[],
                        target_luma=0.4)
        assert len(processed) == ps.BASE_MIN_POOL, (
            f"обработано {len(processed)} кандидатов, ожидалось {ps.BASE_MIN_POOL} "
            f"(пол пула не сработал без Director)")

    def test_aesthetic_ranking_has_a_real_alternative(self, tmp_path, monkeypatch):
        """Раньше при good_needed=1-2 побеждал первый прошедший гейты —
        эстетика не успевала сравнить кандидатов. Теперь она реально решает
        между несколькими реальными вариантами: кандидат #1 (первый в
        выдаче) сознательно сделан ХУЖЕ по эстетике, #3 — лучше, и без пола
        пула #1 победил бы просто по порядку, не будучи лучшим."""
        _stub_pool(monkeypatch, tmp_path)
        scores = {1: 1.0, 2: 2.0, 3: 9.0, 4: 3.0}
        monkeypatch.setattr(ps, "aesthetic_score",
                            lambda p: scores.get(_id_from_trial(p), 2.0))
        out = ps.pexels_photo("medieval sword", 0, used_ids=set(), used_hashes=[],
                              target_luma=0.4)
        assert out is not None
        meta = ps.read_media_sidecar(out)
        assert meta is not None and meta.get("pexels_id") == 3, (
            f"победил не самый эстетичный кандидат: sidecar={meta}")

    def test_fast_tail_gets_a_smaller_but_nonzero_floor(self, tmp_path, monkeypatch):
        """Слоты после FAST_MODE_START_INDEX держат меньший пол, но не 1."""
        _stub_pool(monkeypatch, tmp_path)
        processed = []
        monkeypatch.setattr(ps, "aesthetic_score",
                            lambda p: processed.append(p) or 5.0)
        ps.pexels_photo("medieval sword", ps.FAST_MODE_START_INDEX + 1,
                        used_ids=set(), used_hashes=[], target_luma=0.4)
        assert len(processed) == ps.FAST_BASE_MIN_POOL
        assert ps.FAST_BASE_MIN_POOL >= 2, "хвостовой пол не должен схлопываться до 1"


def _id_from_trial(path):
    import re
    m = re.search(r"trial_(\d+)\.jpg", path)
    return int(m.group(1)) if m else -1


class TestDirectorStillWins:
    """Ноль регрессии: если Director включён, его пул (обычно больше) побеждает."""

    def test_director_pool_is_not_shrunk_by_the_base_floor(self, tmp_path, monkeypatch):
        _stub_pool(monkeypatch, tmp_path)
        processed = []
        monkeypatch.setattr(ps, "aesthetic_score",
                            lambda p: processed.append(p) or 5.0)
        ps.pexels_photo("medieval sword", 0, used_ids=set(), used_hashes=[],
                        target_luma=0.4, director_score_fn=lambda *a, **k: 0.5)
        assert len(processed) == ps.DIRECTOR_MIN_POOL


class TestSelectionSignature:
    def test_new_constants_are_part_of_the_signature(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        start = src.index("def _selection_stack_signature")
        block = src[start:src.index("\ndef candidate_gate_signature", start)]
        assert "BASE_MIN_POOL" in block
        assert "FAST_BASE_MIN_POOL" in block
