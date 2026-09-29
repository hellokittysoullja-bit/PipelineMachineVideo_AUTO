#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Темп источников без простоя (решение владельца 29.09: скорость важнее
полноты последних фраз).

Замер живого прогона judge14: Commons из контейнера получал 429 на 26 из 41
запроса, и каждый отказ стоил слоту до минуты ожидания паузы хоста;
Openverse без ключа — 3.1 с на запрос, то есть очередь из десятка
формулировок держала слот полминуты. Здесь запирается:
  * очередь или пауза хоста длиннее SOURCE_MAX_WAIT_SEC — запрос
    пропускается (source_health.Skipped), а не ждёт, и место в очереди не
    занимается;
  * пропуск — не ошибка источника: свой счёт, и пустота не пишется в кэш на
    прогон (следующий слот спросит снова);
  * Commons стартует с 0.3 с (лимит подписанного клиента WMF), после 429
    замедляется и ВОЗВРАЩАЕТ темп после серии успехов; не больше 3 запросов
    одновременно;
  * харнесс ждёт очередь всегда — иначе запись и воспроизведение разошлись
    бы по времени, а не по входу.
"""
import io
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import source_health as sh  # noqa: E402
import commons_source as cs  # noqa: E402
import museum_sources as ms  # noqa: E402

sys.argv = ["pipeline_smart.py", tempfile.mkdtemp(prefix="pacing_")]
import pipeline_smart as ps  # noqa: E402
import stock_fetch_multisource as ov  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, retry_after=None):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return urllib.error.HTTPError("u", code, "too many", headers, None)


# --- source_health ------------------------------------------------------------

def test_default_limit_is_five_seconds(monkeypatch):
    monkeypatch.delenv("SOURCE_MAX_WAIT_SEC", raising=False)
    assert sh.max_wait_sec() == 5.0
    monkeypatch.setenv("SOURCE_MAX_WAIT_SEC", "1e9")
    assert sh.max_wait_sec() == 1e9
    monkeypatch.setenv("SOURCE_MAX_WAIT_SEC", "мусор")
    assert sh.max_wait_sec() == 5.0


def test_long_queue_is_skipped_without_taking_a_place():
    h = sh.Host("q", interval=10.0)
    t0 = time.monotonic()
    h.wait(max_wait=1.0)                       # первый — сразу
    reserved = h.next_slot
    with pytest.raises(sh.Skipped):
        h.wait(max_wait=1.0)                   # второй ждал бы 10 с
    assert time.monotonic() - t0 < 0.5
    assert h.next_slot == reserved, "пропуск не должен сдвигать очередь"
    assert h.stats["skipped"] == 1 and h.stats["requests"] == 1


def test_pause_counts_as_waiting_only_with_a_limit():
    h = sh.Host("p", cooldown_sec=60.0)
    h.throttled(retry_after=30)
    with pytest.raises(sh.Skipped):
        h.wait(max_wait=5.0)
    # Без предела — прежнее поведение: паузу проверяет вызывающий сам.
    t0 = time.monotonic()
    h.wait()
    assert time.monotonic() - t0 < 0.5


def test_pace_recovers_after_a_run_of_successes():
    h = sh.Host("r", interval=0.3, max_interval=8.0, cooldown_sec=60, recover_after=3)
    h.throttled()
    h.cooldown_until = 0.0
    h.throttled()
    assert abs(h.interval - 1.2) < 1e-9
    for _ in range(3):
        h.succeeded()
    assert abs(h.interval - 0.6) < 1e-9
    for _ in range(30):
        h.succeeded()
    assert abs(h.interval - 0.3) < 1e-9, "не быстрее исходного темпа"


def test_without_recover_after_the_slowdown_stays():
    h = sh.Host("s", interval=0.3, max_interval=8.0, cooldown_sec=60)
    h.throttled()
    for _ in range(50):
        h.succeeded()
    assert abs(h.interval - 0.6) < 1e-9


# --- Commons ------------------------------------------------------------------

def test_commons_starts_at_the_wmf_limit_and_recovers():
    assert cs.MIN_INTERVAL_SEC == 0.3
    assert cs.HOST.base_interval == 0.3 and cs.HOST.recover_after > 0
    assert cs.MAX_INFLIGHT == 3


def test_commons_429_with_long_pause_is_skipped_not_slept(monkeypatch, tmp_path):
    monkeypatch.delenv("SOURCE_MAX_WAIT_SEC", raising=False)
    monkeypatch.setattr(cs, "HOST", sh.Host("c429", cooldown_sec=60.0))
    monkeypatch.setattr(cs, "CACHE_DIR", str(tmp_path))
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req.full_url)
        raise _http_error(429, retry_after=22)

    monkeypatch.setattr(cs.urllib.request, "urlopen", urlopen)
    t0 = time.monotonic()
    with pytest.raises(sh.Skipped):
        cs.search("battle of agincourt")
    assert time.monotonic() - t0 < 2.0, "пауза 22 с не должна проживаться"
    assert len(calls) == 1


def test_commons_never_has_more_than_three_requests_in_flight(monkeypatch, tmp_path):
    monkeypatch.setenv("SOURCE_MAX_WAIT_SEC", "1e9")
    monkeypatch.setattr(cs, "HOST", sh.Host("cflight"))
    monkeypatch.setattr(cs, "CACHE_DIR", str(tmp_path))
    lock = threading.Lock()
    now, peak = [0], [0]

    def urlopen(req, timeout=None):
        with lock:
            now[0] += 1
            peak[0] = max(peak[0], now[0])
        time.sleep(0.15)
        with lock:
            now[0] -= 1
        return _Resp(json.dumps({"query": {"pages": {}}}).encode("utf-8"))

    monkeypatch.setattr(cs.urllib.request, "urlopen", urlopen)
    threads = [threading.Thread(target=cs.search, args=(f"query {i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 3


def test_commons_skip_is_counted_apart_and_not_cached_for_the_run(monkeypatch):
    monkeypatch.setattr(ps.feature_flags, "enabled", lambda name, *a, **k: True)
    ps.reset_source_stats()
    calls = []

    def search(q):
        calls.append(q)
        raise sh.Skipped("очередь")

    monkeypatch.setattr(cs, "search", search)
    assert ps._commons_search_photos("medieval battle") == []
    assert ps._commons_search_photos("medieval battle") == []
    assert len(calls) == 2, "пропуск не должен застывать в кэше прогона"
    st = ps.SOURCE_STATS["commons"]
    assert st["skipped"] == 2 and st["search_errors"] == 0


# --- Openverse ----------------------------------------------------------------

@pytest.fixture
def _openverse(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "OPENVERSE_CACHE_DIR", str(tmp_path / "ov"))
    monkeypatch.delenv("OPENVERSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("OPENVERSE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("SOURCE_MAX_WAIT_SEC", raising=False)
    ps._OPENVERSE_TOKEN.update({"value": None, "expires_at": 0.0, "failed": False})
    ps._OPENVERSE_HOST.reset()
    ps._OPENVERSE_SEARCH_CACHE.clear()
    for k in ("requests", "cache_hits", "cache_misses"):
        ps.OPENVERSE_STATS[k] = 0
    yield
    ps._OPENVERSE_HOST.reset()


def test_openverse_long_anonymous_queue_is_skipped(_openverse, monkeypatch):
    monkeypatch.setattr(ps, "OPENVERSE_ANON_MIN_INTERVAL_SEC", 3.1)
    ps._openverse_throttle(False)            # сразу; следующее место — через 3.1 с
    ps._OPENVERSE_HOST.next_slot += 3.1      # ещё одно место занял соседний слот
    t0 = time.monotonic()
    with pytest.raises(sh.Skipped):
        ps._openverse_throttle(False)        # ждал бы 6.2 с
    assert time.monotonic() - t0 < 0.5
    assert ps.OPENVERSE_STATS["requests"] == 1, "пропущенный запрос не считается сделанным"


def test_openverse_429_pauses_the_host_so_the_rest_skip_at_once(_openverse, monkeypatch):
    monkeypatch.setattr(ps, "OPENVERSE_ANON_MIN_INTERVAL_SEC", 0.0)

    def urlopen(req, timeout=None):
        raise _http_error(429, retry_after=40)

    monkeypatch.setattr(ps.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        ps._openverse_fetch_one("medieval castle", ov)
    assert ps._OPENVERSE_HOST.cooling()
    with pytest.raises(sh.Skipped):
        ps._openverse_fetch_one("medieval sword", ov)


def test_openverse_skip_is_counted_apart_and_not_cached(_openverse, monkeypatch):
    monkeypatch.setattr(ps.feature_flags, "enabled", lambda name, *a, **k: True)
    ps.reset_source_stats()
    calls = []

    def fetch(variant, _ov):
        calls.append(variant)
        raise sh.Skipped("очередь")

    monkeypatch.setattr(ps, "_openverse_fetch_one", fetch)
    assert ps._openverse_search_photos("medieval knight armour") == []
    first = len(calls)
    assert ps._openverse_search_photos("medieval knight armour") == []
    assert len(calls) == 2 * first
    st = ps.SOURCE_STATS["openverse"]
    assert st["skipped"] == 2 * first and st["search_errors"] == 0


# --- Мет ----------------------------------------------------------------------

def test_met_long_queue_skips_and_the_answer_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.delenv("SOURCE_MAX_WAIT_SEC", raising=False)
    host = sh.Host("met-test", interval=0.0, cooldown_sec=60.0)
    host.next_slot = time.monotonic() + 100.0          # очередь на 100 с
    monkeypatch.setattr(ms, "MET_HOST", host)
    monkeypatch.setattr(ms, "MUSEUM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ms, "_get_json",
                        lambda u: (_ for _ in ()).throw(AssertionError("Мет спрошен в обход очереди")))
    monkeypatch.setattr(ms, "search_cleveland", lambda q, **k: [{"id": "cleveland:1"}])
    monkeypatch.setattr(ms, "search_chicago", lambda q, **k: [])
    monkeypatch.setattr(ms.feature_flags, "enabled", lambda name, *a, **k: name != "MET_CATALOG")
    ms.reset_fetch_stats()
    ms.MET_HOST = host                                  # reset_fetch_stats обнулил очередь
    host.next_slot = time.monotonic() + 100.0
    ms._SEARCH_CACHE.clear()
    t0 = time.monotonic()
    assert ms.search_museums("medieval helmet") == [{"id": "cleveland:1"}]
    assert time.monotonic() - t0 < 2.0
    assert ms.FETCH_STATS["met_skipped"] == 1
    assert ms.FETCH_STATS["met_search_failures"] == 0
    assert not ms._SEARCH_CACHE, "неполный ответ не кэшируется даже на прогон"
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".json")]


def test_met_mirror_tool_still_waits(monkeypatch):
    """Инструменты без спешки (mirror-met) предела не передают."""
    seen = []
    monkeypatch.setattr(ms, "MET_HOST", type("H", (), {
        "wait": lambda self, interval=None, max_wait=None: seen.append(max_wait),
        "cooling": lambda self: False})())
    monkeypatch.setattr(ms, "_get_json", lambda u: {"ok": 1})
    assert ms._met_get("https://x") == {"ok": 1}
    assert seen == [None]


# --- харнесс ------------------------------------------------------------------

def test_harness_always_waits_the_queue(monkeypatch, tmp_path):
    import selection_freeze as sf
    monkeypatch.delenv("SOURCE_MAX_WAIT_SEC", raising=False)
    monkeypatch.setattr(sf, "_dotenv_values", lambda: {})
    env = sf.child_env({}, str(tmp_path), 0)
    assert env["SOURCE_MAX_WAIT_SEC"] == "1e9"
    env = sf.child_env({"SOURCE_MAX_WAIT_SEC": "3"}, str(tmp_path), 0)
    assert env["SOURCE_MAX_WAIT_SEC"] == "3"
