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
# TEMP_FOLDER задаётся ФИКСТУРОЙ (см. selection_env), а не подменой
# sys.argv на импорте. Подмена работала ровно до тех пор, пока ЭТОТ файл
# первым импортировал pipeline_smart: VIDEO_FOLDER/TEMP_FOLDER считаются
# ОДИН РАЗ на импорте модуля, поэтому любой тест-файл, который в алфавите
# раньше и тоже импортирует pipeline_smart, забирал это право себе — и
# TEMP_FOLDER становился путём к .py-файлу самого того теста
# (`tests/test_brief_stock_query.py/temp_smart`, NotADirectoryError на
# первом же makedirs). Измерено 15.09: файл в одиночку зелёный, после
# соседнего — четыре падения; суита, результат которой зависит от порядка
# файлов, не является защитой ни от чего.
sys.argv = ["pipeline_smart.py", tempfile.mkdtemp(prefix="thumbfirst_")]
import pipeline_smart as ps  # noqa: E402
from _media_calls import pick_photo, pick_video  # noqa: E402


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
    # Рабочая папка кэша — своя на каждый тест и НЕ зависит от того, кто
    # первым импортировал модуль (см. комментарий у sys.argv выше).
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp_smart"))
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
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q, department=None: [dict(c) for c in cands])
        monkeypatch.setattr(ps, "clip_relevance",
                            lambda path, text: rel.get(os.path.basename(path).split("trial_")[-1].replace(".jpg", ""), None)
                            if False else rel.get(_which(path), 0.0))
        monkeypatch.setattr(ps, "is_relevant_candidate", lambda path, q, relevance=None: (relevance or 0) >= 0.19)
        monkeypatch.setattr(ps, "image_sharpness_score", lambda p: 100.0)

        # Обратное соответствие «имя trial-файла -> кандидат» строится ТОЙ
        # ЖЕ функцией, которой имя строит прод (`candidate_path_token`), а не
        # разбором написания. Первая версия резала id по ':' — и замолчала,
        # когда прод стал санитизировать id для имени файла (двоеточие
        # запрещено на Windows, слэш уводил путь в несуществующий каталог).
        # Тест при этом не нашёл ни одного дефекта: все кандидаты получили
        # relevance 0.0, победил первый, и падение указывало на ранжирование
        # вместо собственной подсказки.
        _tok = {ps.candidate_path_token(f"met:{i}"): f"thumb{i}.jpg" for i in range(6)}

        def _which(path):
            tid = path.rsplit(".trial_", 1)[-1].replace(".jpg", "")
            return _tok.get(tid, "")
        globals()["_which"] = _which

        out = pick_photo(ps, "medieval armour", 0, used_ids=set(), used_hashes=[], target_luma=0.4,
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
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q, department=None: [dict(c) for c in cands])
        # relevance: met:0 лучший, но его ПОЛНЫЙ файл размыт
        # Та же причина, что выше: соответствие строится функцией прода.
        _rel = {ps.candidate_path_token(f"met:{i}"): r
                for i, r in enumerate((0.35, 0.30, 0.25))}
        monkeypatch.setattr(ps, "clip_relevance",
                            lambda path, text: _rel[
                                path.rsplit(".trial_", 1)[-1].replace(".jpg", "")])
        monkeypatch.setattr(ps, "is_relevant_candidate", lambda path, q, relevance=None: True)
        monkeypatch.setattr(ps, "image_sharpness_score",
                            lambda p: 1.0 if os.path.basename(p).startswith("0000_") and _is_full0(p) else 100.0)
        seen_full = {}

        def _is_full0(p):
            # полноразмерный файл победителя лежит под именем кэша cf; отличаем по содержимому
            from PIL import Image
            return Image.open(p).getpixel((0, 0))[1] < 10   # met:0 -> зелёный канал 0
        globals()["_is_full0"] = _is_full0

        out = pick_photo(ps, "medieval armour", 0, used_ids=set(), used_hashes=[], target_luma=0.4,
                              text_key="sharp-repick")
        sidecar = ps.read_media_sidecar(out)
        assert str(sidecar.get("pexels_id")) == "met:1"
        assert "sharp_repick" in str(sidecar.get("chosen_by"))


class TestOpenverseProbeDoesNotBurnTheApiQuota:
    def test_wikimedia_files_are_requested_at_a_bounded_width(self):
        """Две измеренные причины, обе с живых прогонов: поле `thumbnail` у
        Openverse — URL через ЕГО API (лимит 20/мин, в прогоне отвечал 424),
        а оригиналы upload.wikimedia.org отвечают 429 с прямой просьбой
        пользоваться превью — это стоило 49 сорванных скачек и ПУСТОГО
        слота «medieval castle moat water». Special:FilePath?width= —
        документированный способ получить любую ширину для любого формата."""
        u = "https://upload.wikimedia.org/wikipedia/commons/2/22/Allington_Castle.jpg"
        assert ps.wikimedia_thumb_url(u, 640) == (
            "https://commons.wikimedia.org/wiki/Special:FilePath/Allington_Castle.jpg?width=640")
        # Формат значения не имеет — правило одно на все
        assert ps.wikimedia_thumb_url(
            "https://upload.wikimedia.org/wikipedia/commons/2/22/Map.tif", 2000).endswith(
            "Special:FilePath/Map.tif?width=2000")
        assert ps.wikimedia_thumb_url("https://example.org/x.jpg") is None

    def test_openverse_candidate_never_points_at_a_wikimedia_original(self, monkeypatch, tmp_path):
        import io as _io, json as _json
        import stock_fetch_multisource as ov

        class _Resp(_io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        payload = {"results": [{"id": "w", "title": "Castle", "license": "cc0", "source": "wikimedia",
                                "url": "https://upload.wikimedia.org/wikipedia/commons/2/22/Castle.jpg",
                                "foreign_landing_url": "http://page"}]}
        monkeypatch.setattr(ps, "OPENVERSE_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(ps, "OPENVERSE_ANON_MIN_INTERVAL_SEC", 0.0)
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda req, timeout=None: _Resp(_json.dumps(payload).encode()))
        c = ps._openverse_fetch_one("castle", ov)[0]
        assert "upload.wikimedia.org" not in c["src"]["large2x"]
        assert "width=2000" in c["src"]["large2x"] and "width=640" in c["src"]["medium"]

    def test_failed_probe_falls_back_to_full_file_and_is_counted(self, selection_env, monkeypatch):
        d = selection_env
        full = _jpeg(str(d / "full.jpg"), (90, 90, 90))
        cand = {"id": "openverse:1", "alt": "x", "url": "u",
                "src": {"large2x": "file://" + full, "medium": "file:///nonexistent/thumb.jpg"}}
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q, department=None: [dict(cand)])
        monkeypatch.setattr(ps, "clip_relevance", lambda path, text: 0.30)
        monkeypatch.setattr(ps, "is_relevant_candidate", lambda path, q, relevance=None: True)
        monkeypatch.setattr(ps, "image_sharpness_score", lambda p: 100.0)
        out = pick_photo(ps, "medieval armour", 0, used_ids=set(), used_hashes=[], target_luma=0.4,
                              text_key="probe-fallback")
        assert out is not None
        st = ps.SOURCE_STATS["openverse"]
        assert st["considered"] == 1 and st["won"] == 1 and st.get("download_errors", 0) == 0

    def test_unreachable_candidate_is_counted_as_download_error(self, selection_env, monkeypatch):
        cand = {"id": "openverse:2", "alt": "x", "url": "u",
                "src": {"large2x": "file:///nonexistent/full.jpg", "medium": "file:///nonexistent/t.jpg"}}
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q, department=None: [dict(cand)])
        out = pick_photo(ps, "medieval armour", 0, used_ids=set(), used_hashes=[], target_luma=0.4,
                              text_key="probe-dead")
        assert ps.SOURCE_STATS["openverse"]["download_errors"] == 1


class TestDownloadPoliteness:
    """Пробная выборка качает до 20 кандидатов параллельно, и Wikimedia
    отвечает на такой всплеск HTTP 429: в прогоне 14.09 так молча потерялись
    25 кандидатов из 145, предложенных Openverse. Интервал — на ХОСТ, чтобы
    чужой лимит не замедлял Pexels и музеи."""

    def test_only_listed_hosts_are_slowed_down(self, monkeypatch):
        monkeypatch.setattr(ps, "DOWNLOAD_HOST_MIN_INTERVAL", {"commons.wikimedia.org": 0.05})
        ps.source_health.reset_all()
        import time as _t
        t0 = _t.monotonic()
        for _ in range(3):
            ps._download_host_throttle("https://commons.wikimedia.org/wiki/Special:FilePath/X.jpg")
        slowed = _t.monotonic() - t0
        t0 = _t.monotonic()
        for _ in range(3):
            ps._download_host_throttle("https://images.pexels.com/photos/1.jpg")
        free = _t.monotonic() - t0
        assert slowed >= 0.09 and free < 0.02

    def test_429_is_retried_once_then_succeeds(self, monkeypatch, tmp_path):
        import io as _io
        import urllib.error
        calls = []

        class _Resp(_io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def flaky(req, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.HTTPError(req.full_url, 429, "slow down", {}, _io.BytesIO(b""))
            return _Resp(b"payload")

        monkeypatch.setattr(ps, "DOWNLOAD_RETRY_PAUSE_SEC", 0.0)
        monkeypatch.setattr(ps.urllib.request, "urlopen", flaky)
        dest = str(tmp_path / "x.jpg")
        ps.atomic_url_download(ps.urllib.request.Request("https://commons.wikimedia.org/x.jpg"), dest, 10)
        assert open(dest, "rb").read() == b"payload" and len(calls) == 2

    def test_404_is_not_retried(self, monkeypatch, tmp_path):
        import io as _io
        import urllib.error
        calls = []

        def gone(req, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 404, "no", {}, _io.BytesIO(b""))

        monkeypatch.setattr(ps.urllib.request, "urlopen", gone)
        with pytest.raises(urllib.error.HTTPError):
            ps.atomic_url_download(ps.urllib.request.Request("https://x/y.jpg"), str(tmp_path / "y"), 10)
        assert len(calls) == 1

    def test_wikimedia_gets_the_user_agent_their_policy_asks_for(self, monkeypatch, tmp_path):
        """Соответствие правилам источника, а НЕ измеренное улучшение:
        прямой замер 14.09 дал 0 успешных из 6 и с браузерной строкой, и с
        политикой — сегодняшние 429 стоят на адресе. Pexels при этом
        обязан остаться на браузероподобной строке (ЧАСТЬ 14)."""
        import io as _io
        seen = {}

        class _Resp(_io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def spy(req, timeout=None):
            seen[urllib.parse.urlsplit(req.full_url).hostname] = req.get_header("User-agent")
            return _Resp(b"x")

        import urllib.parse
        monkeypatch.setattr(ps.urllib.request, "urlopen", spy)
        for url in ("https://commons.wikimedia.org/a.jpg", "https://images.pexels.com/b.jpg"):
            r = ps.urllib.request.Request(url, headers={"User-Agent": ps.UA})
            ps.atomic_url_download(r, str(tmp_path / "f.jpg"), 10)
        assert "PipelineMachineVideo_AUTO" in seen["commons.wikimedia.org"]
        assert seen["images.pexels.com"] == ps.UA
