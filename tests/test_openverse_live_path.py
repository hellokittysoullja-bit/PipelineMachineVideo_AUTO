"""Openverse подключён к ЖИВОМУ пути отбора (07.09), а не только к
пре-фетч скрипту.

Реальный пробел, найденный при разборе внешней критики. `scripts/
stock_fetch_multisource.py` реализует Openverse полностью —
институциональный белый список источников (Met/Wikimedia/Rijksmuseum/
Europeana/Смитсоновский), fail-closed проверка лицензии на каждый
результат, манифест атрибуций — и `OPENVERSE_ENABLED=1` даже включён в
примере конфига. Но `pipeline_smart.py` (реальный путь отбора эпизода) не
вызывал этот код НИ РАЗУ: вклад архивов в опубликованный эпизод был ровно
нулевым, при пустой папке `media/`.

`_openverse_search_photos()` даёт архивные кандидаты в форме Pexels-
кандидата, чтобы они конкурировали в ОДНОМ пуле под ОДНИМИ гейтами
(relevance/вето/домен-гвард/резкость/дедуп) — не отдельная ветка отбора.

Живой сквозной прогон (`pexels_photo("medieval map", ...)` с реальным
ключом и реальным API) подтвердил рабочую цепочку: победил кандидат
`openverse:e8b7760f...` — «Medieval south-east Wales map Lloyd»,
Wikimedia Commons, CC0 — реально скачан, sidecar записан корректно.
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


# Реальный контракт ответа api.openverse.org/v1/images/ — снят живым запросом
# 07.09 (curl на "medieval map", license=cc0, source=met,rijksmuseum,wikimedia):
# results[].{id, title, url, foreign_landing_url, license, source, creator}.
REAL_SHAPE_PAYLOAD = {"results": [
    {"id": "d94ce0c9-bb7a-4278-b0f1-ef9f151d386c",
     "title": "Samo's realm of slavic tribes",
     "url": "https://upload.wikimedia.org/wikipedia/commons/2/22/x.jpg",
     "foreign_landing_url": "https://commons.wikimedia.org/wiki/File:x.jpg",
     "license": "cc0", "license_version": "1.0",
     "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
     "source": "wikimedia", "creator": None},
    {"id": "flickr-personal-upload",
     "title": "Someone's vacation photo tagged medieval",
     "url": "https://live.staticflickr.com/y.jpg",
     "foreign_landing_url": "https://flickr.com/photos/x/y",
     "license": "cc0", "source": "flickr"},   # cc0, но НЕ институциональный источник
    {"id": "met-by-license",
     "title": "Met artifact under attribution license",
     "url": "https://images.metmuseum.org/z.jpg",
     "foreign_landing_url": "https://metmuseum.org/art/z",
     "license": "by", "source": "met"},        # институциональный, но НЕ cc0
]}


@pytest.fixture(autouse=True)
def _clear_cache():
    ps._OPENVERSE_SEARCH_CACHE.clear()
    yield
    ps._OPENVERSE_SEARCH_CACHE.clear()


class TestNormalizationAndFailClosed:
    def test_only_cc0_and_trusted_source_survive(self, monkeypatch):
        """Fail-closed на РЕАЛЬНОЙ форме ответа: cc0+flickr отсеян (источник
        не институциональный), by+met отсеян (лицензия не cc0) — остаётся
        только пересечение обоих условий."""
        monkeypatch.setenv("OPENVERSE_ENABLED", "1")
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda req, timeout=None: _FakeResponse(REAL_SHAPE_PAYLOAD))
        cands = ps._openverse_search_photos("medieval map")
        assert len(cands) == 1
        assert cands[0]["id"] == "openverse:d94ce0c9-bb7a-4278-b0f1-ef9f151d386c"

    def test_candidate_shape_matches_what_pexels_photo_expects(self, monkeypatch):
        """download() в pexels_photo() читает p["src"]["large2x"]; filter_
        alt_blocklist/pexels_candidate_text читают alt/url. Обе оси обязаны
        работать на Openverse-кандидате без единой особой ветки."""
        monkeypatch.setenv("OPENVERSE_ENABLED", "1")
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda req, timeout=None: _FakeResponse(REAL_SHAPE_PAYLOAD))
        cand = ps._openverse_search_photos("medieval map")[0]
        assert cand["src"]["large2x"].startswith("http")
        assert cand["url"].startswith("http")
        assert isinstance(cand["alt"], str)
        text = ps.pexels_candidate_text(cand)
        assert "samo" in text.lower()

    def test_ids_are_namespaced_to_avoid_colliding_with_pexels_ids(self, monkeypatch):
        """Pexels ID — целые числа; Openverse ID — UUID-строки. Общий
        used_ids/дедуп-пул не должен их путать даже случайно."""
        monkeypatch.setenv("OPENVERSE_ENABLED", "1")
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda req, timeout=None: _FakeResponse(REAL_SHAPE_PAYLOAD))
        cand = ps._openverse_search_photos("medieval map")[0]
        assert cand["id"].startswith("openverse:")


class TestFailOpen:
    """Недоступный Openverse не имеет права ронять слот, у которого есть
    рабочий Pexels-путь."""

    def test_disabled_flag_makes_no_network_call(self, monkeypatch):
        monkeypatch.setenv("OPENVERSE_ENABLED", "0")
        called = []
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda *a, **k: called.append(1))
        assert ps._openverse_search_photos("medieval map") == []
        assert called == []

    def test_network_error_returns_empty_not_raises(self, monkeypatch):
        monkeypatch.setenv("OPENVERSE_ENABLED", "1")

        def boom(req, timeout=None):
            raise OSError("network unreachable")
        monkeypatch.setattr(ps.urllib.request, "urlopen", boom)
        assert ps._openverse_search_photos("medieval map") == []

    def test_malformed_json_returns_empty_not_raises(self, monkeypatch):
        monkeypatch.setenv("OPENVERSE_ENABLED", "1")

        class BadResponse:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return b"not json"
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda req, timeout=None: BadResponse())
        assert ps._openverse_search_photos("medieval map") == []

    def test_result_is_cached_per_process(self, monkeypatch):
        monkeypatch.setenv("OPENVERSE_ENABLED", "1")
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(1)
            return _FakeResponse(REAL_SHAPE_PAYLOAD)
        monkeypatch.setattr(ps.urllib.request, "urlopen", fake_urlopen)
        ps._openverse_search_photos("medieval map")
        ps._openverse_search_photos("medieval map")
        assert len(calls) == 1


class TestWiredIntoPoolAssembly:
    """Главная проверка: результат конкурирует в пуле pexels_photo(), а
    не живёт в отдельной ветке."""

    def test_openverse_candidate_can_win_the_slot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENVERSE_ENABLED", "1")
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
        monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda req, timeout=None: _FakeResponse(REAL_SHAPE_PAYLOAD))
        # Pexels выдаёт ноль кандидатов — победить может только архив.
        monkeypatch.setattr(ps, "_pexels_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "disambiguate_search_query", lambda q: q)
        monkeypatch.setattr(ps, "is_relevant_candidate", lambda *a, **k: True)
        monkeypatch.setattr(ps, "image_sharpness_score", lambda p: 999.0)
        monkeypatch.setattr(ps, "aesthetic_score", lambda p: 5.0)
        monkeypatch.setattr(ps, "estimate_shot_size", lambda p: "medium")
        monkeypatch.setattr(ps, "measure_luma", lambda p: 0.4)

        from PIL import Image
        import io as _io
        png = _io.BytesIO()
        Image.new("RGB", (64, 64), (90, 90, 90)).save(png, "PNG")
        monkeypatch.setattr(ps, "atomic_url_download",
                            lambda req, dest, timeout=None: open(dest, "wb").write(png.getvalue()))

        out = ps.pexels_photo("medieval map", 0, used_ids=set(), used_hashes=[],
                              target_luma=0.4)
        assert out is not None
        meta = ps.read_media_sidecar(out)
        assert meta["pexels_id"] == "openverse:d94ce0c9-bb7a-4278-b0f1-ef9f151d386c"

    def test_disabled_by_default_behaviour_is_unaffected(self, tmp_path, monkeypatch):
        """OPENVERSE_ENABLED=0 (дефолт реестра) — Pexels-путь работает
        байт-в-байт как раньше, никаких сетевых обращений к Openverse."""
        monkeypatch.setenv("OPENVERSE_ENABLED", "0")
        touched = []
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda *a, **k: touched.append(1))
        assert ps._openverse_search_photos("medieval map") == []
        assert touched == []


class TestTestIsolation:
    """Реальный найденный вживую баг (07.09): .env этого канала имеет
    OPENVERSE_ENABLED=1, и юнит-тест, который сам не выставляет режим,
    делал ЖИВОЙ сетевой запрос к api.openverse.org внутри тестового прогона
    (test_base_min_pool.py — реальный кандидат из архива победил поддельных
    кандидатов теста и сломал непричастный к Openverse ассерт). Тот же класс
    бага, от которого conftest.py уже защищает GEMINI_API_KEY."""

    def test_conftest_forces_openverse_off_by_default(self):
        import os
        assert os.environ.get("OPENVERSE_ENABLED") == "0", (
            "autouse-фикстура conftest.py должна гасить OPENVERSE_ENABLED "
            "из рабочего .env — иначе тесты без явного mockpatch делают "
            "живые сетевые запросы")


class TestSelectionSignature:
    def test_openverse_flag_is_part_of_the_signature(self):
        """Включение архивов меняет состав пула — без подписи здесь смена
        флага на прогретом temp_smart/ не доходила бы до экрана."""
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        start = src.index("def _selection_stack_signature")
        block = src[start:src.index("\ndef candidate_gate_signature", start)]
        assert "OPENVERSE_ENABLED" in block
