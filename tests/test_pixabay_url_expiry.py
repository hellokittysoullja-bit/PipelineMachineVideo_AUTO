"""Подписанные ссылки Pixabay на файлы протухают (замер 28.09: все /get/-ссылки
из ответов 24.09 и 26.09 отвечают 400, свежая по номеру кадра — 200), а
дисковый кэш поиска держал ответ 30 дней. При повторном рендере позже суток
ВСЕ кандидаты Pixabay выпадали из кучи — среди них упавший рыцарь фразы #4
эп.94."""
import io
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402
import stock_fetch_multisource as ms  # noqa: E402

OLD = "https://pixabay.com/get/gOLD_640.jpg"
NEW = "https://pixabay.com/get/gNEW_640.jpg"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_expired_pixabay_link_is_refreshed_by_id(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "PIXABAY_API_KEY", "k")
    ps._remember_pixabay_urls({"id": 321443, "webformatURL": OLD})
    seen = []

    def fake(req, timeout=None):
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        seen.append(url)
        if url == OLD:
            raise urllib.error.HTTPError(url, 400, "expired", {}, io.BytesIO(b""))
        if "pixabay.com/api/" in url:
            assert "id=321443" in url
            return _Resp(json.dumps({"hits": [{"id": 321443, "webformatURL": NEW}]}).encode())
        if url == NEW:
            return _Resp(b"JPEGDATA")
        raise AssertionError(url)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    dest = str(tmp_path / "p.jpg")
    assert ps.atomic_url_download(urllib.request.Request(OLD), dest, timeout=5)
    assert open(dest, "rb").read() == b"JPEGDATA"
    assert seen[0] == OLD and seen[-1] == NEW


def test_non_pixabay_400_is_still_a_failure(tmp_path, monkeypatch):
    def fake(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b""))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    try:
        ps.atomic_url_download(urllib.request.Request("https://example.org/x.jpg"),
                               str(tmp_path / "x.jpg"), timeout=5)
    except urllib.error.HTTPError as e:
        assert e.code == 400
    else:
        raise AssertionError("ожидался отказ")


def test_pixabay_search_cache_expires_before_its_links(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path))
    calls = []
    ps.cached_search_json("pixabay_photo", "q", lambda: calls.append(1) or {"hits": []},
                          ttl=ps.PIXABAY_SEARCH_CACHE_TTL_SEC)
    fp = next(p for p in os.listdir(tmp_path) if p.startswith("pixabay_photo_"))
    old = __import__("time").time() - ps.PIXABAY_SEARCH_CACHE_TTL_SEC - 60
    os.utime(tmp_path / fp, (old, old))
    ps.cached_search_json("pixabay_photo", "q", lambda: calls.append(1) or {"hits": []},
                          ttl=ps.PIXABAY_SEARCH_CACHE_TTL_SEC)
    assert len(calls) == 2
    assert ps.PIXABAY_SEARCH_CACHE_TTL_SEC <= 24 * 3600 < ps.SEARCH_DISK_CACHE_TTL_SEC


def test_both_pixabay_searches_use_the_short_ttl():
    import inspect
    for fn in (ps._pixabay_search_photos, ps._pixabay_search_videos):
        assert "ttl=PIXABAY_SEARCH_CACHE_TTL_SEC" in inspect.getsource(fn)
    assert "_remember_pixabay_urls(h)" in inspect.getsource(ps._pixabay_search_photos)
