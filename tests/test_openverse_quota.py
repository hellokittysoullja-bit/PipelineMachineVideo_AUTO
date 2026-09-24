"""Openverse: квота измерена по заголовкам ответа (13.09) — анонимно 20
запросов в минуту и 200 в день. Каскад делает до 3 запросов на авторский
запрос, эпизод из 42 запросов упирается в лимит посреди КАЖДОГО рендера, а
fail-open превращал это в «источник дал ноль» без единой строки.

Здесь запирается: интервал между запросами (общий на процесс), дисковый кэш
(только непустые ответы, TTL), bearer-токен по ключу (анонимно — без него),
и что всё это НЕ трогает сеть в тестах.
"""
import io
import json
import os
import sys
import tempfile
import urllib.error

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.mkdtemp(prefix="ovquota_")]
import pipeline_smart as ps  # noqa: E402
import stock_fetch_multisource as ov  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(payload):
    def f(req, timeout=None):
        return _Resp(json.dumps(payload).encode("utf-8"))
    return f


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "OPENVERSE_CACHE_DIR", str(tmp_path / "ovcache"))
    monkeypatch.setattr(ps, "OPENVERSE_ANON_MIN_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(ps, "OPENVERSE_AUTH_MIN_INTERVAL_SEC", 0.0)
    monkeypatch.delenv("OPENVERSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("OPENVERSE_CLIENT_SECRET", raising=False)
    ps._OPENVERSE_TOKEN.update({"value": None, "expires_at": 0.0, "failed": False})
    ps._OPENVERSE_HOST.next_slot = 0.0
    for k in ("requests", "cache_hits", "cache_misses"):
        ps.OPENVERSE_STATS[k] = 0
    yield


_WIKI = {"results": [{"id": "w1", "title": "Allington Castle",
                      "url": "https://upload.wikimedia.org/wikipedia/commons/2/22/Allington_Castle.jpg",
                      "thumbnail": "https://api.openverse.org/v1/images/w1/thumb/", "license": "cc0",
                      "source": "wikimedia", "foreign_landing_url": "http://page"}]}
_RESULT = {"results": [{"id": "abc", "title": "Allington Castle", "url": "http://img/full.jpg",
                        "thumbnail": "http://img/thumb.jpg", "license": "cc0",
                        "source": "wikimedia", "foreign_landing_url": "http://page"}]}


class TestDiskCache:
    def test_second_call_does_not_touch_the_network(self, monkeypatch):
        calls = []

        def spy(req, timeout=None):
            calls.append(req.full_url)
            return _Resp(json.dumps(_RESULT).encode("utf-8"))

        monkeypatch.setattr(ps.urllib.request, "urlopen", spy)
        first = ps._openverse_fetch_one("medieval castle", ov)
        assert [c["id"] for c in first] == ["openverse:abc"]
        # Превью НЕ из поля thumbnail (это URL через API с квотой 20/мин на
        # каждую скачку), а по конвенции Wikimedia; не-Wikimedia URL -> полный.
        assert first[0]["src"]["medium"] == "http://img/full.jpg"
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("сеть тронута")))
        again = ps._openverse_fetch_one("medieval castle", ov)
        assert again == first
        assert ps.OPENVERSE_STATS["cache_hits"] == 1 and len(calls) == 1

    def test_empty_answer_is_not_frozen(self, monkeypatch):
        """Пустой ответ может быть следствием квоты, а не корпуса — его
        нельзя запомнить на месяц."""
        monkeypatch.setattr(ps.urllib.request, "urlopen", _fake_urlopen({"results": []}))
        assert ps._openverse_fetch_one("nothing here", ov) == []
        assert not os.path.exists(ps._openverse_cache_path("nothing here", ov))


class TestPreviewNeverGoesThroughTheApi:
    def test_wikimedia_file_is_requested_by_width_not_through_the_api(self, monkeypatch):
        monkeypatch.setattr(ps.urllib.request, "urlopen", _fake_urlopen(_WIKI))
        c = ps._openverse_fetch_one("allington castle", ov)[0]
        assert "api.openverse.org" not in c["src"]["medium"]
        # Special:FilePath?width= — и превью, и рабочий файл: оригиналы
        # upload.wikimedia.org отвечают 429 и просят брать превью.
        assert c["src"]["medium"].endswith("Special:FilePath/Allington_Castle.jpg?width=640")
        assert "upload.wikimedia.org" not in c["src"]["large2x"]


class TestAnonymousMode:
    def test_anonymous_request_has_no_bearer_and_page_size_20(self, monkeypatch):
        seen = {}

        def spy(req, timeout=None):
            seen["auth"] = req.headers.get("Authorization")
            seen["url"] = req.full_url
            return _Resp(json.dumps(_RESULT).encode("utf-8"))

        monkeypatch.setattr(ps.urllib.request, "urlopen", spy)
        ps._openverse_fetch_one("medieval sword", ov)
        assert seen["auth"] is None and "page_size=20" in seen["url"]

    def test_throttle_spaces_requests(self, monkeypatch):
        monkeypatch.setattr(ps, "OPENVERSE_ANON_MIN_INTERVAL_SEC", 0.05)
        import time as _t
        t0 = _t.monotonic()
        for _ in range(3):
            ps._openverse_throttle(False)
        assert _t.monotonic() - t0 >= 0.09
        assert ps.OPENVERSE_STATS["requests"] == 3


class TestAuthenticatedMode:
    def test_key_yields_bearer_and_page_100(self, monkeypatch):
        monkeypatch.setenv("OPENVERSE_CLIENT_ID", "id")
        monkeypatch.setenv("OPENVERSE_CLIENT_SECRET", "secret")
        seen = []

        def spy(req, timeout=None):
            seen.append((req.full_url, req.headers.get("Authorization")))
            if "auth_tokens/token" in req.full_url:
                return _Resp(json.dumps({"access_token": "TOK", "expires_in": 3600}).encode("utf-8"))
            return _Resp(json.dumps(_RESULT).encode("utf-8"))

        monkeypatch.setattr(ps.urllib.request, "urlopen", spy)
        ps._openverse_fetch_one("medieval sword", ov)
        urls = [u for u, _ in seen]
        assert any("auth_tokens/token" in u for u in urls)
        search = [(u, a) for u, a in seen if "/images/" in u][0]
        assert search[1] == "Bearer TOK" and "page_size=100" in search[0]
        assert ps.OPENVERSE_STATS["auth"] is True

    def test_rejected_key_falls_back_to_anonymous_once(self, monkeypatch, capsys):
        monkeypatch.setenv("OPENVERSE_CLIENT_ID", "id")
        monkeypatch.setenv("OPENVERSE_CLIENT_SECRET", "bad")

        def spy(req, timeout=None):
            if "auth_tokens/token" in req.full_url:
                raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b""))
            return _Resp(json.dumps(_RESULT).encode("utf-8"))

        monkeypatch.setattr(ps.urllib.request, "urlopen", spy)
        ps._openverse_fetch_one("medieval sword", ov)
        ps._openverse_fetch_one("medieval dagger", ov)
        out = capsys.readouterr().out
        assert out.count("ключ не принят") == 1
        assert ps._OPENVERSE_TOKEN["failed"] is True
