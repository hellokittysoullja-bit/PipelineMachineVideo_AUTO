"""Wikimedia Commons как источник кадра: только то, что свободно без
указания автора, в общей форме кандидата.

Фикстура — НАСТОЯЩИЙ ответ API по запросу «Battle of Agincourt miniature»
(24.09): в нём сами собой есть и свободные миниатюры (pd, cc0), и файл
CC BY 4.0 с обязательной атрибуцией, и файл без метаданных лицензии —
обе отрицательные проверки на живых полях, а не на выдуманных."""
import json
import os
import sys
import urllib.error

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import commons_source as cs  # noqa: E402

FIXTURE = os.path.join(REPO, "tests", "fixtures", "commons", "agincourt_search.json")


def _pages():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _meta(**fields):
    return {k: {"value": v} for k, v in fields.items()}


def test_fixture_keeps_only_free_files_without_attribution(monkeypatch):
    monkeypatch.setattr(cs, "_fetch_pages", lambda q, limit: _pages())
    cs.reset_stats()
    got = cs.search("Battle of Agincourt miniature")
    titles = [c["alt"] for c in got]
    assert any("St. Alban's Chronicle" in t for t in titles), "свободная миниатюра обязана пройти"
    assert not any("Slag bij Azincourt" in t for t in titles), "CC BY 4.0 требует атрибуции — отказ"
    assert not any(t.startswith("Military and religious life") for t in titles), \
        "нет метаданных лицензии — нет кандидата"
    assert cs.STATS["rejected_license"] >= 2
    for c in got:
        lic = c["_commons_meta"]["license"].lower()
        assert "public domain" in lic or "cc0" in lic, lic


def test_candidate_has_the_common_shape_and_polite_headers(monkeypatch):
    monkeypatch.setattr(cs, "_fetch_pages", lambda q, limit: _pages())
    c = cs.search("x")[0]
    assert c["id"].startswith("commons:") and c["id"].split(":", 1)[1].isdigit()
    assert c["src"]["medium"].startswith(cs.FILEPATH) and c["src"]["medium"].endswith("?width=640")
    assert c["src"]["large2x"].endswith(f"?width={cs.WORK_WIDTH}")
    ua = c["_download_headers"]["User-Agent"]
    assert "github.com" in ua and "@" not in ua, "контакт — адрес проекта, почту владельца не отдаём"
    assert c["url"].startswith("https://commons.wikimedia.org/wiki/File:")
    assert c["_commons_meta"]["file_page"] == c["url"]


@pytest.mark.parametrize("fields,ok", [
    ({"License": "pd", "AttributionRequired": "false"}, True),
    ({"License": "cc0", "AttributionRequired": "false"}, True),
    ({"License": "cc-by-sa-4.0", "AttributionRequired": "true"}, False),
    ({"License": "pd", "AttributionRequired": "true"}, False),
    ({"License": "pd", "AttributionRequired": "false", "Restrictions": "personality"}, False),
    ({"License": "pd", "NonFree": "true"}, False),
    ({}, False),
])
def test_license_rule_is_fail_closed(fields, ok):
    assert cs.is_free_without_attribution(_meta(**fields)) is ok


def test_small_and_non_image_files_are_rejected():
    base = {"pageid": 1, "title": "File:X.jpg", "imageinfo": [{
        "mime": "image/jpeg", "width": 1200, "height": 900, "descriptionurl": "u",
        "extmetadata": _meta(License="pd", AttributionRequired="false")}]}
    assert cs.to_candidate(base) is not None
    small = json.loads(json.dumps(base))
    small["imageinfo"][0]["height"] = 300
    assert cs.to_candidate(small) is None
    pdf = json.loads(json.dumps(base))
    pdf["imageinfo"][0]["mime"] = "application/pdf"
    assert cs.to_candidate(pdf) is None


def test_html_is_stripped_from_metadata():
    assert cs.plain('<a href="x">Thomas&nbsp;Walsingham</a>  (1422)') == "Thomas Walsingham (1422)"


class _Host:
    def __init__(self):
        self.throttles = 0

    def cooling(self):
        return False

    def cooldown_left(self):
        return 0.0

    def wait(self, interval=None):
        pass

    def throttled(self, retry_after=None):
        self.throttles += 1
        return True

    def succeeded(self):
        pass


def _http_error(code):
    return urllib.error.HTTPError("u", code, "too many", {}, None)


def test_429_backs_off_and_then_succeeds(monkeypatch, tmp_path):
    host = _Host()
    monkeypatch.setattr(cs, "HOST", host)
    monkeypatch.setattr(cs, "CACHE_DIR", str(tmp_path))
    answers = [_http_error(429)]

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"query": {"pages": {"1": _pages()[0]}}}).encode()

    def fake_urlopen(req, timeout=None):
        if answers:
            raise answers.pop()
        return _Resp()
    monkeypatch.setattr(cs.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cs.json, "load", lambda r: json.loads(r.read()))
    pages = cs._fetch_pages("q", 5)
    assert host.throttles == 1 and len(pages) == 1


def test_repeated_429_raises_for_the_caller_to_fail_open(monkeypatch, tmp_path):
    monkeypatch.setattr(cs, "HOST", _Host())
    monkeypatch.setattr(cs, "CACHE_DIR", str(tmp_path))

    def always_429(req, timeout=None):
        raise _http_error(429)
    monkeypatch.setattr(cs.urllib.request, "urlopen", always_429)
    with pytest.raises(urllib.error.HTTPError):
        cs._fetch_pages("q", 5)


def test_search_cache_roundtrip_keeps_raw_pages(monkeypatch, tmp_path):
    monkeypatch.setattr(cs, "CACHE_DIR", str(tmp_path))
    cs._cache_put("q", 5, _pages())
    assert cs._cache_get("q", 5) == _pages()
    assert cs._cache_get("other", 5) is None


def test_pipeline_wiring(monkeypatch):
    import pipeline_smart as ps
    import shot_types
    assert ps.candidate_source({"id": "commons:123"}) == "commons"
    assert shot_types.SOURCE_CAPABILITIES["commons"]["supports"]
    assert ps.source_allowed_for("commons", "scene") and not ps.source_allowed_for("commons", "texture")
    monkeypatch.setattr(cs, "_fetch_pages", lambda q, limit: _pages())
    cand = cs.search("x")[0]
    prov = ps.candidate_provenance(cand)
    assert prov and prov["license"] and prov["file_page"] == cand["url"]
    assert "dated" in ps.candidate_caption(cand)


def test_flag_off_means_no_request(monkeypatch):
    import pipeline_smart as ps
    monkeypatch.setenv("COMMONS_ENABLED", "0")
    monkeypatch.setattr(cs, "search", lambda q: (_ for _ in ()).throw(AssertionError("запрос при выключенном флаге")))
    ps._COMMONS_SEARCH_CACHE.clear()
    assert ps._commons_search_photos("anything") == []


def test_source_failure_is_noted_not_raised(monkeypatch):
    import pipeline_smart as ps
    monkeypatch.setenv("COMMONS_ENABLED", "1")
    noted = []
    monkeypatch.setattr(ps, "_note_source_search_error", lambda src, e, q: noted.append(src))

    def boom(q):
        raise _http_error(429)
    monkeypatch.setattr(cs, "search", boom)
    ps._COMMONS_SEARCH_CACHE.clear()
    assert ps._commons_search_photos("q") == [] and noted == ["commons"]


def test_429_pause_is_the_one_the_service_asked_for(monkeypatch, tmp_path):
    seen = []

    class Host(_Host):
        def throttled(self, retry_after=None):
            seen.append(retry_after)
            return True
    monkeypatch.setattr(cs, "HOST", Host())
    monkeypatch.setattr(cs, "CACHE_DIR", str(tmp_path))

    def always_429(req, timeout=None):
        raise urllib.error.HTTPError("u", 429, "too many", {"Retry-After": "22"}, None)
    monkeypatch.setattr(cs.urllib.request, "urlopen", always_429)
    with pytest.raises(urllib.error.HTTPError):
        cs._fetch_pages("q", 5)
    assert seen == [22.0, 22.0]


@pytest.mark.parametrize("headers,want", [({"Retry-After": "22"}, 22.0), ({}, None),
                                          ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None)])
def test_retry_after_header_parsing(headers, want):
    assert cs.retry_after_sec(urllib.error.HTTPError("u", 429, "x", headers, None)) == want
