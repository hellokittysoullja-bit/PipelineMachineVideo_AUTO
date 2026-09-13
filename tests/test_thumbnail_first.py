"""Кандидаты оцениваются по ПРЕВЬЮ, полный файл качается только у победителя.

Измеренная причина (A/B на 9 слотах эпизода 02, 13.09): при цене
полноразмерной скачки на каждого кандидата перебор останавливался после 4
прошедших гейт, и на «medieval plate armour museum» настоящий доспех
(relevance 0.325) стоял десятым — его не рассматривали. С превью вся
пробная выборка (20) стоит дешевле четырёх полных файлов.

Резкость — единственная ось, которую на превью мерить нельзя (уменьшение
«лечит» размытие): она проверяется на полноразмерном файле победителя, и
размытый победитель уступает следующему по ТОМУ ЖЕ ранжированию.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.mkdtemp(prefix="thumbfirst_")]
import pipeline_smart as ps  # noqa: E402


def _jpeg(path, colour, size=(640, 360)):
    from PIL import Image
    Image.new("RGB", size, colour).save(path, "JPEG")
    return path


def _cand(cid, full, thumb):
    return {"id": cid, "alt": cid, "url": "u",
            "src": {"large2x": "file://" + full, "medium": "file://" + thumb}}


class TestProbeUrl:
    def test_prefers_medium_then_falls_back_to_full(self):
        assert ps.candidate_probe_url({"src": {"medium": "m", "large2x": "L"}}) == "m"
        assert ps.candidate_probe_url({"src": {"large2x": "L"}}) == "L"
        assert ps.candidate_probe_url({"src": {"large": "l"}}) == "l"

    def test_museum_candidates_carry_a_preview(self):
        import museum_sources as ms
        c = ms._candidate("met:1", "t", "http://full", "http://page", {}, thumb_url="http://small")
        assert c["src"]["medium"] == "http://small" and c["src"]["large2x"] == "http://full"


@pytest.fixture
def selection_env(monkeypatch, tmp_path):
    """Отбор без сети и без моделей: гейты и оценки подменены детерминированными
    функциями по имени файла, кандидаты лежат на диске под file://."""
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "")
    monkeypatch.setattr(ps, "_openverse_search_photos", lambda q: [])
    monkeypatch.setattr(ps, "_pixabay_search_photos", lambda q: [])
    monkeypatch.setattr(ps, "_unsplash_search_photos", lambda q: [])
    monkeypatch.setattr(ps, "filter_alt_blocklist", lambda items: items)
    monkeypatch.setattr(ps, "estimate_shot_size", lambda p: "medium")
    monkeypatch.setattr(ps, "measure_luma", lambda p, **k: 0.4)
    monkeypatch.setattr(ps, "aesthetic_score", lambda p: 5.0)
    monkeypatch.setattr(ps, "PHOTO_SHARPNESS_REJECT", 10.0)
    ps._PEXELS_SEARCH_CACHE.clear()
    ps.reset_source_stats()
    return tmp_path


class TestWholeSliceIsEvaluated:
    def test_best_candidate_at_the_end_of_the_slice_still_wins(self, selection_env, monkeypatch):
        """Шесть кандидатов, все проходят гейт, лучший по relevance — ПОСЛЕДНИЙ.
        Раньше перебор останавливался на четвёртом прошедшем, и лучший не
        рассматривался вообще."""
        d = selection_env
        cands, rel = [], {}
        for i in range(6):
            full = _jpeg(str(d / f"full{i}.jpg"), (10 * i, 50, 50))
            thumb = _jpeg(str(d / f"thumb{i}.jpg"), (10 * i, 50, 50), size=(320, 180))
            cands.append(_cand(f"met:{i}", full, thumb))
            rel[os.path.basename(thumb)] = 0.20 + 0.02 * i     # лучший — met:5 (0.30)
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q: [dict(c) for c in cands])
        monkeypatch.setattr(ps, "clip_relevance",
                            lambda path, text: rel.get(os.path.basename(path).split("trial_")[-1].replace(".jpg", ""), None)
                            if False else rel.get(_which(path), 0.0))
        monkeypatch.setattr(ps, "is_relevant_candidate", lambda path, q, relevance=None: (relevance or 0) >= 0.19)
        monkeypatch.setattr(ps, "image_sharpness_score", lambda p: 100.0)

        def _which(path):
            # trial-файл именуется по id кандидата: cf.trial_<id>.jpg
            tid = path.rsplit(".trial_", 1)[-1].replace(".jpg", "")
            return f"thumb{tid.split(':')[-1]}.jpg"
        globals()["_which"] = _which

        out = ps.pexels_photo("medieval armour", 0, used_ids=set(), used_hashes=[], target_luma=0.4,
                              text_key="whole-slice")
        assert out is not None
        assert ps.SOURCE_STATS["met"]["considered"] == 6, ps.SOURCE_STATS
        # Победил met:5 — файл-победитель качался ПОЛНЫМ (640x360), не превью.
        from PIL import Image
        assert Image.open(out).size == (640, 360)
        sidecar = ps.read_media_sidecar(out)
        assert str(sidecar.get("pexels_id")) == "met:5"


class TestSharpnessOnFullSizeWinner:
    def test_blurry_full_size_winner_yields_to_next(self, selection_env, monkeypatch):
        d = selection_env
        cands = []
        for i in range(3):
            full = _jpeg(str(d / f"full{i}.jpg"), (100, 20 * i, 20))
            thumb = _jpeg(str(d / f"thumb{i}.jpg"), (100, 20 * i, 20), size=(320, 180))
            cands.append(_cand(f"met:{i}", full, thumb))
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q: [dict(c) for c in cands])
        # relevance: met:0 лучший, но его ПОЛНЫЙ файл размыт
        monkeypatch.setattr(ps, "clip_relevance",
                            lambda path, text: {"0": 0.35, "1": 0.30, "2": 0.25}[
                                path.rsplit(".trial_met:", 1)[-1][0]])
        monkeypatch.setattr(ps, "is_relevant_candidate", lambda path, q, relevance=None: True)
        monkeypatch.setattr(ps, "image_sharpness_score",
                            lambda p: 1.0 if os.path.basename(p).startswith("0000_") and _is_full0(p) else 100.0)
        seen_full = {}

        def _is_full0(p):
            # полноразмерный файл победителя лежит под именем кэша cf; отличаем по содержимому
            from PIL import Image
            return Image.open(p).getpixel((0, 0))[1] < 10   # met:0 -> зелёный канал 0
        globals()["_is_full0"] = _is_full0

        out = ps.pexels_photo("medieval armour", 0, used_ids=set(), used_hashes=[], target_luma=0.4,
                              text_key="sharp-repick")
        sidecar = ps.read_media_sidecar(out)
        assert str(sidecar.get("pexels_id")) == "met:1"
        assert "sharp_repick" in str(sidecar.get("chosen_by"))


class TestOpenverseProbeDoesNotBurnTheApiQuota:
    def test_wikimedia_thumb_follows_storage_convention(self):
        """Поле `thumbnail` у Openverse — URL через ЕГО API с лимитом 20/мин на
        каждую скачку: в A/B ab4 все кандидаты Openverse молча выпали из
        пробной выборки на скачке превью. Превью берётся по конвенции самого
        Wikimedia — без API и без квоты."""
        u = "https://upload.wikimedia.org/wikipedia/commons/2/22/Allington_Castle.jpg"
        assert ps.wikimedia_thumb_url(u) == (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Allington_Castle.jpg/640px-Allington_Castle.jpg")
        # tif/svg — другое имя превью у Wikimedia; честнее полный файл
        assert ps.wikimedia_thumb_url("https://upload.wikimedia.org/wikipedia/commons/2/22/Map.tif") is None
        assert ps.wikimedia_thumb_url("https://example.org/x.jpg") is None

    def test_failed_probe_falls_back_to_full_file_and_is_counted(self, selection_env, monkeypatch):
        d = selection_env
        full = _jpeg(str(d / "full.jpg"), (90, 90, 90))
        cand = {"id": "openverse:1", "alt": "x", "url": "u",
                "src": {"large2x": "file://" + full, "medium": "file:///nonexistent/thumb.jpg"}}
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q: [dict(cand)])
        monkeypatch.setattr(ps, "clip_relevance", lambda path, text: 0.30)
        monkeypatch.setattr(ps, "is_relevant_candidate", lambda path, q, relevance=None: True)
        monkeypatch.setattr(ps, "image_sharpness_score", lambda p: 100.0)
        out = ps.pexels_photo("medieval castle", 0, used_ids=set(), used_hashes=[], target_luma=0.4,
                              text_key="probe-fallback")
        assert out is not None
        st = ps.SOURCE_STATS["openverse"]
        assert st["considered"] == 1 and st["won"] == 1 and st.get("download_errors", 0) == 0

    def test_unreachable_candidate_is_counted_as_download_error(self, selection_env, monkeypatch):
        cand = {"id": "openverse:2", "alt": "x", "url": "u",
                "src": {"large2x": "file:///nonexistent/full.jpg", "medium": "file:///nonexistent/t.jpg"}}
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q: [dict(cand)])
        out = ps.pexels_photo("medieval castle", 0, used_ids=set(), used_hashes=[], target_luma=0.4,
                              text_key="probe-dead")
        assert ps.SOURCE_STATS["openverse"]["download_errors"] == 1
