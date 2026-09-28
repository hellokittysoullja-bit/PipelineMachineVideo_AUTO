#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Временный сбой источника — повтор, а не потеря; сбой не запоминается.

Живой прогон эпизода 94 (27.09, профиль каждого HTTP-запроса): Commons
отвечал 429 на превью (по два отказа подряд — пауза 2 с их не спасала), и
кандидаты молча выпадали; DeepSeek и источники изредка отвечали 5xx. А
ответ источника после разового сбоя запоминался на ВЕСЬ прогон пустым —
запрос оставался пустым во всех следующих слотах, которые его делят.
"""
import io
import os
import sys
import urllib.error

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import source_health as sh  # noqa: E402


class Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, headers=None):
    return urllib.error.HTTPError("u", code, "x", headers or {}, io.BytesIO(b""))


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(sh.time, "sleep", lambda s: slept.append(s))
    sh.reset_all()
    yield slept
    sh.reset_all()


def _opener(monkeypatch, answers):
    calls = []

    def fake(req, timeout=None):
        calls.append(req)
        a = answers[len(calls) - 1]
        if isinstance(a, Exception):
            raise a
        return Resp(a)
    monkeypatch.setattr(sh.urllib.request, "urlopen", fake) if hasattr(sh, "urllib") else None
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return calls


def test_transient_error_is_retried(monkeypatch, no_sleep):
    calls = _opener(monkeypatch, [http_error(503), b"ok"])
    with sh.urlopen_retry("http://x", 5, "search:t1") as r:
        assert r.read() == b"ok"
    assert len(calls) == 2


def test_broken_connection_is_retried(monkeypatch, no_sleep):
    calls = _opener(monkeypatch, [urllib.error.URLError("reset"), b"ok"])
    with sh.urlopen_retry("http://x", 5, "search:t2") as r:
        assert r.read() == b"ok"
    assert len(calls) == 2


def test_permanent_error_is_not_retried(monkeypatch, no_sleep):
    calls = _opener(monkeypatch, [http_error(404), b"ok"])
    with pytest.raises(urllib.error.HTTPError):
        sh.urlopen_retry("http://x", 5, "search:t3")
    assert len(calls) == 1


def test_long_retry_after_is_a_quota_not_a_burst(monkeypatch, no_sleep):
    """Pexels после исчерпания часовой квоты просит ждать до часа —
    ждать это внутри слота нельзя."""
    calls = _opener(monkeypatch, [http_error(429, {"Retry-After": "900"}), b"ok"])
    with pytest.raises(urllib.error.HTTPError):
        sh.urlopen_retry("http://x", 5, "search:t4")
    assert len(calls) == 1


def test_exhausted_pexels_quota_is_not_retried(monkeypatch, no_sleep):
    calls = _opener(monkeypatch, [http_error(429, {"X-Ratelimit-Remaining": "0"}), b"ok"])
    with pytest.raises(urllib.error.HTTPError):
        sh.urlopen_retry("http://x", 5, "search:t5")
    assert len(calls) == 1


def test_rate_limit_pauses_the_whole_host(monkeypatch, no_sleep):
    """429 с Retry-After — пауза всего хоста: соседние потоки не бьют в тот
    же лимит (host.cooling()), а сам вызов ждёт и повторяет."""
    calls = _opener(monkeypatch, [http_error(429, {"Retry-After": "7"}), b"ok"])
    with sh.urlopen_retry("http://x", 5, "search:t6") as r:
        assert r.read() == b"ok"
    assert len(calls) == 2
    assert sh.host("search:t6").stats["cooldowns"] == 1


# ---------------------------------------------------------------- pipeline_smart

@pytest.fixture
def ps(monkeypatch, tmp_path):
    pytest.importorskip("PIL")
    sys.argv = ["pipeline_smart.py", str(tmp_path)]
    import pipeline_smart
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path / "temp_smart"))
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "search_cache"))
    monkeypatch.setattr(pipeline_smart.time, "sleep", lambda s: None)
    sh.reset_all()
    yield pipeline_smart
    sh.reset_all()     # пауза хоста из теста не должна пережить тест


def test_pixabay_failure_is_not_remembered_for_the_run(ps, monkeypatch):
    import stock_fetch_multisource as ms
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    monkeypatch.setattr(ms, "PIXABAY_API_KEY", "k")
    ps._PIXABAY_PHOTO_CACHE.clear()
    answers = iter([http_error(503)] * 3 + [b'{"hits": [{"id": 5, "largeImageURL": "http://i/5.jpg"}]}'])
    calls = []

    def fake(req, timeout=None):
        calls.append(1)
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return Resp(a)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert ps._pixabay_search_photos("medieval dagger") == []
    got = ps._pixabay_search_photos("medieval dagger")
    assert [c["id"] for c in got] == ["pixabay:5"], "сбой не должен остаться пустым ответом на весь прогон"


def test_commons_and_openverse_failures_are_not_remembered(ps, monkeypatch):
    import commons_source
    monkeypatch.setenv("COMMONS_ENABLED", "1")
    ps._COMMONS_SEARCH_CACHE.clear()
    state = {"n": 0}

    def search(q):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("разрыв")
        return [{"id": "commons:1"}]
    monkeypatch.setattr(commons_source, "search", search)
    assert ps._commons_search_photos("battle") == []
    assert ps._commons_search_photos("battle") == [{"id": "commons:1"}]

    monkeypatch.setenv("OPENVERSE_ENABLED", "1")
    ps._OPENVERSE_SEARCH_CACHE.clear()
    ov = {"n": 0}

    def variants(label, q, fetch_one):
        ov["n"] += 1
        if ov["n"] == 1:
            raise OSError("503")
        return [{"id": "openverse:1"}]
    monkeypatch.setattr(ps, "_search_with_variants", variants)
    assert ps._openverse_search_photos("battle") == []
    assert ps._openverse_search_photos("battle") == [{"id": "openverse:1"}]


def test_incomplete_museum_answer_is_not_remembered(ps, monkeypatch):
    """Мет на паузе или упал музей — ответ неполный: ни модуль музеев, ни
    пайплайн не запоминают его на прогон."""
    import museum_sources as ms
    monkeypatch.setattr(ms.feature_flags, "enabled", lambda name, *a, **k: name != "MET_CATALOG")
    monkeypatch.setattr(ms, "MUSEUM_CACHE_DIR", str(ps.TEMP_FOLDER) + "_museum")
    ms._SEARCH_CACHE.clear()
    ps._MUSEUM_SEARCH_CACHE.clear()
    state = {"broken": True}

    def met(q, limit=None, department=None):
        return [{"id": "met:1"}]

    def chicago(q, limit=None):
        if state["broken"]:
            raise OSError("разрыв")
        return [{"id": "chicago:2"}]
    monkeypatch.setattr(ms, "_sources", lambda department=None: [
        ("met", met), ("chicago", chicago)])
    first = ps._museum_search_photos("medieval dagger")
    assert [c["id"] for c in first] == ["met:1"]
    state["broken"] = False
    second = ps._museum_search_photos("medieval dagger")
    assert [c["id"] for c in second] == ["met:1", "chicago:2"]


def test_download_retries_server_errors_and_broken_connections(ps, monkeypatch, tmp_path):
    answers = iter([http_error(502), urllib.error.URLError("reset"), b"img"])
    calls = []

    def fake(req, timeout=None):
        calls.append(1)
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return Resp(a)
    monkeypatch.setattr(ps.urllib.request, "urlopen", fake)
    dest = str(tmp_path / "x.jpg")
    ps.atomic_url_download(ps.urllib.request.Request("https://images.example/x.jpg"), dest, 10)
    assert open(dest, "rb").read() == b"img" and len(calls) == 3


def test_background_download_waits_for_the_pause_the_service_named(ps, monkeypatch, tmp_path):
    """Commons 27.09: «Retry-After: 22» — пауза всего хоста, затем повтор;
    раньше — повтор через 2 с и потеря кандидата. Так ждёт фоновая
    подготовка слотов: ей это не стоит времени слота."""
    import slot_prefetch
    sh.reset_all()
    answers = iter([http_error(429, {"Retry-After": "22"}), b"img"])
    slept = []
    monkeypatch.setattr(ps.time, "sleep", lambda s: slept.append(s))

    def fake(req, timeout=None):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return Resp(a)
    monkeypatch.setattr(ps.urllib.request, "urlopen", fake)
    dest = str(tmp_path / "y.jpg")
    token = slot_prefetch.TARGET.set(3)
    try:
        ps.atomic_url_download(ps.urllib.request.Request("https://commons.wikimedia.org/y.jpg"), dest, 10)
    finally:
        slot_prefetch.TARGET.reset(token)
    assert open(dest, "rb").read() == b"img"
    assert slept and max(slept) >= 21, slept


def test_foreground_download_does_not_stall_the_slot(ps, monkeypatch, tmp_path):
    """На глазах у слота пауза источника длиннее 8 с не ждётся: превью, не
    пришедшее вовремя, стоит в каскаде после оценённых, как и раньше."""
    sh.reset_all()
    answers = iter([http_error(429, {"Retry-After": "22"}), b"img"])
    slept = []
    monkeypatch.setattr(ps.time, "sleep", lambda s: slept.append(s))

    def fake(req, timeout=None):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return Resp(a)
    monkeypatch.setattr(ps.urllib.request, "urlopen", fake)
    with pytest.raises(urllib.error.HTTPError):
        ps.atomic_url_download(ps.urllib.request.Request("https://commons.wikimedia.org/w.jpg"),
                               str(tmp_path / "w.jpg"), 10)
    assert not slept or max(slept) <= ps.DOWNLOAD_FOREGROUND_MAX_WAIT_SEC


def test_download_gives_up_on_a_quota_sized_pause(ps, monkeypatch, tmp_path):
    answers = iter([http_error(429, {"Retry-After": "3600"}), b"img"])

    def fake(req, timeout=None):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return Resp(a)
    monkeypatch.setattr(ps.urllib.request, "urlopen", fake)
    with pytest.raises(urllib.error.HTTPError):
        ps.atomic_url_download(ps.urllib.request.Request("https://x.example/z.jpg"),
                               str(tmp_path / "z.jpg"), 10)


def test_chicago_and_cleveland_retry_a_transient_error(monkeypatch):
    import museum_sources as ms
    monkeypatch.setattr(ms.time, "sleep", lambda s: None)
    state = {"n": 0}

    def get(url):
        state["n"] += 1
        if state["n"] == 1:
            raise http_error(503)
        return {"data": []}
    monkeypatch.setattr(ms, "_get_json", get)
    assert ms._get_json_retry("https://api.artic.edu/x") == {"data": []}
    assert state["n"] == 2
