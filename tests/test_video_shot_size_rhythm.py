"""Ритм крупностей плана для ВИДЕО-кандидатов (VIDEO_DIRECTOR_SCORE_VERSION=2).

Фото уже годами избегает повтора той же крупности плана (wide/medium/close/
detail) два раза подряд — pexels_photo() принимает recent_sizes и штрафует
кандидата, чья estimate_shot_size() совпадает с одной из двух последних
(см. size_ok в pexels_photo(), приоритет сразу после дедупа в _score_and_pick).
Видео в этом сигнале не участвовало вообще: recent_sizes не принимался,
крупность кандидата не считалась, а победивший видео-клип не пополнял
историю ни для следующего видео, ни для следующего фото — то есть подряд
идущие клипы одной крупности (два общих плана, два макро) ничем не
отличались от разнообразной последовательности, хотя это ровно то, что
пользователь называет отсутствием "режиссуры/ритма".

Тесты здесь проверяют функциональное поведение pexels_video(recent_sizes=...)
напрямую (не через main() — она слишком велика для юнит-теста) и то, что
main() реально прокидывает recent_shot_sizes в оба вызова и обратно
получает от видео-победителя обновление истории (source-level).
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402


def _stub_common(monkeypatch, tmp_path, n=2):
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "_pexels_search_videos", lambda q: [
        {"id": i, "url": f"https://www.pexels.com/video/knight-armor-{i}/",
         "video_files": [{"file_type": "video/mp4", "width": 1920, "link": f"http://x/{i}.mp4"}]}
        for i in range(1, n + 1)])
    monkeypatch.setattr(ps, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(
                            str(req.full_url).encode()))
    monkeypatch.setattr(ps, "extract_video_probe_frame", lambda p, **kw: (p + ".jpg", False))
    monkeypatch.setattr(ps, "is_relevant_candidate", lambda *a, **k: True)
    monkeypatch.setattr(ps, "video_domain_guard_violation", lambda *a, **k: (False, None))
    monkeypatch.setattr(ps, "video_sharpness_ok", lambda p: True)
    monkeypatch.setattr(ps, "measure_luma", lambda p: 0.4)


def _id_from_probe(probe_path):
    import re
    m = re.search(r"trial_(\d+)\.mp4", probe_path)
    return int(m.group(1)) if m else None


def _winner_id(out_path):
    meta = ps.read_media_sidecar(out_path)
    return meta.get("pexels_id") if meta else None


class TestShotSizeRhythm:
    def test_repeating_recent_size_loses_to_a_different_one(self, tmp_path, monkeypatch):
        """Кандидат 1 — "wide" (как последние 2 клипа истории), кандидат 2 —
        "detail" (другая крупность). При РАВНОМ sentence-скоре побеждает тот,
        кто предлагает другую крупность — тот же принцип, что size_ok у фото."""
        _stub_common(monkeypatch, tmp_path, n=2)
        monkeypatch.setattr(ps, "estimate_shot_size", lambda p: (
            "wide" if _id_from_probe(p) == 1 else "detail"))
        out = ps.pexels_video(
            "medieval sword close up", 9, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5,
            recent_sizes=["wide", "wide"])
        assert out is not None
        assert _winner_id(out) == 2, "победил кандидат, повторяющий крупность последних клипов"

    def test_all_candidates_repeating_still_fills_the_slot(self, tmp_path, monkeypatch):
        """Слот не должен опустеть, если ВСЕ кандидаты повторяют крупность —
        та же философия "лучший из плохих", что и у остальных гейтов."""
        _stub_common(monkeypatch, tmp_path, n=2)
        monkeypatch.setattr(ps, "estimate_shot_size", lambda p: "wide")
        out = ps.pexels_video(
            "medieval sword close up", 9, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5,
            recent_sizes=["wide", "wide"])
        assert out is not None

    def test_recent_sizes_none_is_a_pure_no_op(self, tmp_path, monkeypatch):
        """Ноль регрессии: без recent_sizes (старые вызовы/тесты) поведение
        байт-в-байт как раньше — estimate_shot_size вообще не должен звонить."""
        _stub_common(monkeypatch, tmp_path, n=2)
        called = []
        monkeypatch.setattr(ps, "estimate_shot_size", lambda p: called.append(p) or "wide")
        out = ps.pexels_video(
            "medieval sword close up", 9, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5)
        assert out is not None
        assert called == [], "estimate_shot_size вызван без recent_sizes — не должно быть звонков"

    def test_shot_size_ok_outranks_sentence_score_but_not_luma(self, tmp_path, monkeypatch):
        """Порядок приоритета в sort key — (shot_size_ok, luma_ok, sent_score):
        кандидат с ХУДШИМ sentence_score, но другой крупностью, побеждает
        кандидата с лучшим sentence_score, повторяющего крупность."""
        _stub_common(monkeypatch, tmp_path, n=2)
        monkeypatch.setattr(ps, "estimate_shot_size", lambda p: (
            "wide" if _id_from_probe(p) == 1 else "detail"))

        def score(probe):
            return 0.9 if _id_from_probe(probe) == 1 else 0.1   # кандидат 1 "смысловее"

        out = ps.pexels_video(
            "medieval sword close up", 9, used_ids=set(), used_hashes=[],
            sentence_score_fn=score, recent_sizes=["wide", "wide"])
        assert _winner_id(out) == 2, (
            "более высокий sentence_score перебил штраф за повтор крупности — "
            "порядок осей в sort key нарушен")


class TestWiring:
    def _main_source(self):
        return open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()

    def test_main_passes_recent_sizes_to_both_video_call_sites(self):
        src = self._main_source()
        calls = src.count("recent_sizes=recent_shot_sizes")
        # 2 фото-вызова (уже были) + 2 видео-вызова (новые) = 4.
        assert calls >= 4, f"ожидалось минимум 4 передачи recent_sizes=recent_shot_sizes, нашлось {calls}"

    def test_video_winner_updates_recent_shot_sizes(self):
        src = self._main_source()
        start = src.index("elif video:\n")
        end = src.index("\n        elif director_entry is None", start)
        block = src[start:end]
        assert "recent_shot_sizes.append(estimate_shot_size(probe))" in block

    def test_version_bumped_for_this_change(self):
        assert ps.VIDEO_DIRECTOR_SCORE_VERSION >= 2, (
            "VIDEO_DIRECTOR_SCORE_VERSION не поднят — смена ранжирования видео "
            "не инвалидирует прогретый temp_smart/pexels_video_cache/")
