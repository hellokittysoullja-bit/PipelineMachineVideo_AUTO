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
    # Первая свободная страница фикстуры — 1500 px в ширину: превью 640
    # округляется до стандартных 960, рабочий файл 2000 просит 3840, что шире
    # оригинала, поэтому берётся самая большая стандартная не шире файла — 1280.
    assert c["src"]["medium"].startswith(cs.FILEPATH) and c["src"]["medium"].endswith("?width=960")
    assert c["src"]["large2x"].endswith("?width=1280")
    ua = c["_download_headers"]["User-Agent"]
    assert "github.com" in ua and "@" not in ua, "контакт — адрес проекта, почту владельца не отдаём"
    assert c["url"].startswith("https://commons.wikimedia.org/wiki/File:")
    assert c["_commons_meta"]["file_page"] == c["url"]


# Первые шесть строк — живой замер 28.09: файлы из реальной выдачи Commons,
# запрос width=2000 стоил четырём из них 429 с паузой 600 с, а стандартная
# ширина не больше родной прошла на всех шести. Остальное — границы правила.
@pytest.mark.parametrize("native,target,want", [
    (1666, 2000, 1280),    # Creci, Фруассар — 429 при width=2000
    (2947, 2000, 1920),    # Uccello — 429
    (2024, 2000, 1920),    # Uccello — 429
    (1066, 2000, 960),     # Uccello, Париж — 429
    (3118, 2000, 1920),    # Фиоре — прошёл и так
    (2800, 2000, 1920),    # Тальхоффер — прошёл и так
    (4320, 2000, 3840),    # оригинал шире 3840: просим ту самую ступень
    (4000, 640, 960),      # превью: ближайшая ступень вверх
    (960, 640, 960),       # ровно на границе — годится
    (959, 640, 500),       # на пиксель уже 960 — вниз
    (800, 640, 500),
    (489, 640, 330),
    (3840, 2000, 3840),
    (1920, 1920, 1920),    # желаемая ступень совпала с родной шириной
    (1919, 1920, 1280),
])
def test_thumb_width_is_standard_and_never_wider_than_the_original(native, target, want):
    assert cs.thumb_width(native, target) == want


@pytest.mark.parametrize("native", [None, 0, "", "не число", -5])
def test_unknown_native_width_keeps_the_requested_width(native):
    assert cs.thumb_width(native, 640) == 640
    assert cs.thumb_width(native, 2000) == 2000


def test_thumb_width_never_exceeds_the_original_on_any_size():
    """Свойство, а не примеры: какой бы ни была родная ширина, просимая не
    больше неё, и это стандартная ступень (кроме файлов уже самой малой)."""
    for native in range(20, 9000, 7):
        for target in (640, 1920, 2000):
            got = cs.thumb_width(native, target)
            assert got <= native, (native, target, got)
            assert got in cs.THUMB_STEPS, (native, target, got)
            # Меньше желаемой — только если желаемая (округлённая вверх) шире файла
            if got < target:
                up = next((s for s in cs.THUMB_STEPS if s >= target), None)
                assert up is None or up > native, (native, target, got)


def test_every_candidate_of_the_real_fixture_asks_for_no_more_than_its_width(monkeypatch):
    """Те же страницы реального ответа API: по РЕАЛЬНЫМ родным ширинам."""
    import re as _re
    monkeypatch.setattr(cs, "_fetch_pages", lambda q, limit: _pages())
    got = cs.search("x")
    assert len(got) >= 5
    for c in got:
        for key in ("medium", "large2x"):
            asked = int(_re.search(r"width=(\d+)$", c["src"][key]).group(1))
            assert asked <= int(c["width"]), (c["id"], key, asked, c["width"])
            assert asked in cs.THUMB_STEPS


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


# --- ослабление запроса при пустой выдаче -------------------------------------
#
# Живой случай 28.09: `talhoffer fechtbuch dagger armour 1467` дал ноль, а
# планшеты Тальхоффера с боем на кинжалах в Commons есть. Порядок ступеней
# выведен из кэша всех сессий (1438 запросов, см. relaxed_queries).

def _page(pid, lic="pd", width=1500):
    return {"pageid": pid, "title": f"File:P{pid}.jpg", "imageinfo": [{
        "mime": "image/jpeg", "width": width, "height": 1000, "descriptionurl": "u",
        "extmetadata": _meta(License=lic,
                             AttributionRequired="false" if lic in ("pd", "cc0") else "true")}]}


@pytest.mark.parametrize("query,want", [
    ("talhoffer fechtbuch dagger armour 1467",
     ["talhoffer fechtbuch dagger armour", "talhoffer fechtbuch dagger", "talhoffer fechtbuch"]),
    ("medieval rondel dagger close up", ["medieval rondel dagger", "medieval rondel"]),
    ("armoured knights dagger fight", ["armoured knights dagger", "armoured knights"]),
    ("knight plate armour", ["knight plate"]),      # одиночное слово не ищем: «knight» даёт медали
    ("dagger mud 1467", ["dagger mud"]),
    ("knight armour", []),                          # два слова — ослаблять нечего
    ("close up 1467 macro", []),                    # из чисел и слов кадра — предмета нет
    ("", []),
])
def test_relaxed_queries_steps(query, want):
    assert cs.relaxed_queries(query) == want


def test_relaxed_queries_only_shorten_and_never_reorder():
    queries = ["talhoffer fechtbuch dagger armour 1467", "armoured horsemen charging foot soldiers",
               "gauntlet gripping dagger close up", "a b c d e f g", "Knight ARMOUR mud Detail 1415",
               "medieval print workshop reenactment"]
    for q in queries:
        words = q.split()
        got = cs.relaxed_queries(q)
        assert len(got) <= cs.RELAX_MAX_VARIANTS
        assert len({g.lower() for g in got}) == len(got), "повторы"
        for g in got:
            gw = g.split()
            assert g.lower() != q.lower()
            assert len(gw) >= cs.RELAX_KEEP_WORDS
            it = iter(words)                        # gw — подпоследовательность слов запроса
            assert all(any(w == x for x in it) for w in gw), (q, g)


def _fake_fetch(monkeypatch, answers):
    calls = []

    def fake(q, limit):
        calls.append((q, limit))
        return answers.get(q, [])
    monkeypatch.setattr(cs, "_fetch_pages", fake)
    cs.reset_stats()
    return calls


def test_exact_hit_is_never_relaxed(monkeypatch):
    calls = _fake_fetch(monkeypatch, {"talhoffer fechtbuch dagger armour 1467": [_page(1)]})
    got = cs.search("talhoffer fechtbuch dagger armour 1467")
    assert [c["id"] for c in got] == ["commons:1"]
    assert [q for q, _ in calls] == ["talhoffer fechtbuch dagger armour 1467"]
    assert cs.STATS["relaxed_tries"] == 0


def test_empty_exact_falls_to_the_first_nonempty_relaxed_query(monkeypatch):
    calls = _fake_fetch(monkeypatch, {"talhoffer fechtbuch dagger": [_page(7)]})
    got = cs.search("talhoffer fechtbuch dagger armour 1467", limit=25)
    assert [c["id"] for c in got] == ["commons:7"]
    # Третья ступень не спрашивалась: первая непустая побеждает
    assert [q for q, _ in calls] == ["talhoffer fechtbuch dagger armour 1467",
                                     "talhoffer fechtbuch dagger armour",
                                     "talhoffer fechtbuch dagger"]
    assert {lim for _, lim in calls} == {25}
    assert cs.STATS["relaxed_tries"] == 2 and cs.STATS["relaxed_hits"] == 1


def test_pages_all_refused_by_license_also_count_as_empty(monkeypatch):
    _fake_fetch(monkeypatch, {"knight dagger fight": [_page(1, lic="cc-by-sa-4.0")],
                              "knight dagger": [_page(2)]})
    got = cs.search("knight dagger fight")
    assert [c["id"] for c in got] == ["commons:2"]
    assert cs.STATS["rejected_license"] == 1 and cs.STATS["relaxed_hits"] == 1


def test_nothing_anywhere_returns_empty_and_counts_every_try(monkeypatch):
    calls = _fake_fetch(monkeypatch, {})
    assert cs.search("talhoffer fechtbuch dagger armour 1467") == []
    assert len(calls) == 1 + cs.RELAX_MAX_VARIANTS
    assert cs.STATS["relaxed_tries"] == cs.RELAX_MAX_VARIANTS and cs.STATS["relaxed_hits"] == 0


def test_two_word_query_is_asked_once_and_not_relaxed(monkeypatch):
    calls = _fake_fetch(monkeypatch, {})
    assert cs.search("knight armour") == []
    assert len(calls) == 1 and cs.STATS["relaxed_tries"] == 0


def test_failure_on_a_relaxed_step_is_raised_not_swallowed(monkeypatch):
    """Ослабленный шаг упал — выдача неполная. Молча вернуть пустой список
    значило бы запомнить на прогон «в Commons ничего нет», которого не было."""
    seq = []

    def fake(q, limit):
        seq.append(q)
        if len(seq) == 2:
            raise _http_error(429)
        return []
    monkeypatch.setattr(cs, "_fetch_pages", fake)
    with pytest.raises(urllib.error.HTTPError):
        cs.search("talhoffer fechtbuch dagger armour 1467")


def test_pipeline_does_not_remember_a_search_that_failed_midway(monkeypatch):
    import pipeline_smart as ps
    monkeypatch.setenv("COMMONS_ENABLED", "1")
    noted = []
    monkeypatch.setattr(ps, "_note_source_search_error", lambda src, e, q: noted.append(src))
    seq = []

    def fake(q, limit):
        seq.append(q)
        if len(seq) == 2:
            raise _http_error(429)
        return []
    monkeypatch.setattr(cs, "_fetch_pages", fake)
    ps._COMMONS_SEARCH_CACHE.clear()
    assert ps._commons_search_photos("talhoffer fechtbuch dagger armour 1467") == []
    assert noted == ["commons"]
    assert "talhoffer fechtbuch dagger armour 1467" not in ps._COMMONS_SEARCH_CACHE


def test_pipeline_returns_relaxed_candidates_and_remembers_them(monkeypatch):
    import pipeline_smart as ps
    monkeypatch.setenv("COMMONS_ENABLED", "1")
    _fake_fetch(monkeypatch, {"talhoffer fechtbuch dagger": [_page(9)]})
    ps._COMMONS_SEARCH_CACHE.clear()
    got = ps._commons_search_photos("talhoffer fechtbuch dagger armour 1467")
    assert [c["id"] for c in got] == ["commons:9"]
    assert ps._COMMONS_SEARCH_CACHE["talhoffer fechtbuch dagger armour 1467"] == got

