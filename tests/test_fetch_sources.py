#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Источники кучи опрашиваются параллельно, каждый своей очередью: куча та
же до кандидата, каждый источник видит запросы в прежнем порядке, а время —
максимум по источникам, а не сумма. Сбой источника: очередь источника
останавливается на первой ошибке; с хуком адаптера on_source_failure сбой
стоит только этого источника, без хука уходит наверх, как раньше."""
import os
import sys
import threading
import time
import types
import urllib.error

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import selection_engine  # noqa: E402

SOURCES = ("museum", "archive", "stock")
DELAY = {"museum": 0.05, "archive": 0.1, "stock": 0.02}


class _Adapter:
    """Три источника; источник отвечает на запрос своими кандидатами, с
    повторами между источниками (их снимает unique_by_id)."""

    def __init__(self, fail=None):
        self.calls = {s: [] for s in SOURCES}
        self.lock = threading.Lock()
        self.fail = fail

    def brief_query(self, request):
        return None

    def _job(self, src, pq):
        def job():
            time.sleep(DELAY[src])
            with self.lock:
                self.calls[src].append(pq)
            if self.fail == (src, pq):
                raise ValueError(f"{src} упал на {pq}")
            return [{"id": f"{src}-{pq}-{k}"} for k in range(3)] + [{"id": f"shared-{pq}"}]
        return job

    def _jobs(self, request, pq):
        return [(s, self._job(s, pq)) for s in SOURCES]

    def source_jobs(self, request, pq):
        return self._jobs(request, pq)

    def sources(self, request, pq):
        return [job() for _n, job in self._jobs(request, pq)]

    def filter_pool(self, request, pool):
        return pool

    def note_offered(self, pool):
        pass


class _Sequential(_Adapter):
    """Тот же адаптер без source_jobs — прежний путь, по одному."""
    source_jobs = None


class _Absorbing(_Adapter):
    """Адаптер, у которого сбой источника стоит только этого источника."""

    def __init__(self, fail=None):
        super().__init__(fail)
        self.failures = []

    def on_source_failure(self, request, source_name, exc):
        self.failures.append((source_name, str(exc)))


def _request():
    return types.SimpleNamespace(query="q0", extra_queries=("q1", "q2", "q3"), shot_spec=None)


def test_pool_is_the_same_as_one_by_one():
    par, seq = _Adapter(), _Sequential()
    got = selection_engine.build_pool(_request(), par)
    want = selection_engine.build_pool(_request(), seq)
    assert [c["id"] for c in got] == [c["id"] for c in want]
    assert len(got) == 4 * 3 * 3 + 4


def test_each_source_sees_queries_in_the_old_order():
    a = _Adapter()
    selection_engine.build_pool(_request(), a)
    for src in SOURCES:
        assert a.calls[src] == ["q0", "q1", "q2", "q3"], src


def test_time_is_the_slowest_source_not_the_sum():
    t = time.time()
    selection_engine.build_pool(_request(), _Adapter())
    par = time.time() - t
    total = 4 * sum(DELAY.values())          # по одному: 0.68 с
    slowest = 4 * max(DELAY.values())        # параллельно: ~0.4 с
    assert par < (total + slowest) / 2, (par, total, slowest)


def test_source_failure_goes_up_as_before():
    a = _Adapter(fail=("archive", "q2"))
    with pytest.raises(ValueError, match="archive упал на q2"):
        selection_engine.build_pool(_request(), a)
    # Упавший источник в этом слоте больше не спрашивают — как и раньше,
    # когда первая ошибка прекращала поиск слота.
    assert a.calls["archive"] == ["q0", "q1", "q2"]


def test_source_failure_costs_only_that_source_with_the_hook():
    """Сбой одного источника не уносит кучу: остальные источники в ней
    целиком, упавший — до места сбоя, адаптеру сообщено один раз."""
    a = _Absorbing(fail=("archive", "q1"))
    got = [c["id"] for c in selection_engine.build_pool(_request(), a)]
    assert a.failures == [("archive", "archive упал на q1")]
    assert a.calls["archive"] == ["q0", "q1"]
    assert a.calls["museum"] == a.calls["stock"] == ["q0", "q1", "q2", "q3"]
    for q in ("q0", "q1", "q2", "q3"):
        assert f"museum-{q}-0" in got and f"stock-{q}-2" in got
    assert "archive-q0-0" in got
    assert not any(c.startswith(("archive-q1", "archive-q2", "archive-q3")) for c in got)
    # Без сбоя куча та же, что у адаптера без хука.
    assert [c["id"] for c in selection_engine.build_pool(_request(), _Absorbing())] == \
        [c["id"] for c in selection_engine.build_pool(_request(), _Adapter())]


def test_the_raised_failure_is_the_first_in_the_old_order():
    """Два источника падают: наверх уходит та ошибка, на которой остановился
    бы последовательный путь (первая по порядку запрос -> источник)."""
    class _TwoFail(_Adapter):
        def _job(self, src, pq):
            job = super()._job(src, pq)
            if (src, pq) == ("stock", "q1"):
                def failing():
                    job()
                    raise ValueError("stock упал на q1")
                return failing
            return job
    with pytest.raises(ValueError, match="stock упал на q1"):
        selection_engine.build_pool(_request(), _TwoFail(fail=("museum", "q2")))


def test_photo_adapter_jobs_match_its_sources(monkeypatch):
    """Прод-адаптер фото: sources() — те же задачи, по одному."""
    sys.argv = ["pipeline_smart.py", REPO]
    import pipeline_smart as ps
    for f in ("_shelf_search_photos", "_museum_search_photos", "_commons_search_photos",
              "_openverse_search_photos", "_pexels_search_photos",
              "_pixabay_search_photos", "_unsplash_search_photos"):
        monkeypatch.setattr(ps, f, lambda *a, _f=f, **k: [{"id": _f, "alt": "", "url": "",
                                                           "src": {"medium": "x"}}])
    monkeypatch.setattr(ps, "local_stock_candidate", lambda index: None)
    req = selection_engine.SlotRequest(
        index=0, query="medieval dagger", extra_queries=(), text_key=None,
        shot_brief=None, shot_spec=None, block_text="Вот кинжал.", arbiter_text=None,
        is_opening=False, slot_dur=4.0, action_qualifier=None, target_luma=None,
        director_score_fn=None, director_assist=False, director_report=None,
        video_score_fn=None, used_photo_ids=set(), used_video_ids=set(),
        used_hashes=[], recent_sizes=[])
    jobs = ps.PHOTO_ADAPTER.source_jobs(req, "medieval dagger")
    lists = selection_engine.fetch_sources(req, ps.PHOTO_ADAPTER, ["medieval dagger"])["medieval dagger"]
    assert lists == ps.PHOTO_ADAPTER.sources(req, "medieval dagger")
    assert [n for n, _j in jobs][0] in ("shelf", "museum", "commons")
    assert all(c["_origin_query"] == "medieval dagger" for lst in lists for c in lst)


def _slot_request(ps):
    return selection_engine.SlotRequest(
        index=0, query="medieval dagger", extra_queries=("medieval sword",), text_key=None,
        shot_brief=None, shot_spec=None, block_text="Вот кинжал.", arbiter_text=None,
        is_opening=False, slot_dur=None, action_qualifier=None, target_luma=None,
        director_score_fn=None, director_assist=False, director_report=None,
        video_score_fn=None, used_photo_ids=set(), used_video_ids=set(),
        used_hashes=[], recent_sizes=[])


def _pexels_500(calls):
    def boom(q):
        calls.append(q)
        raise urllib.error.HTTPError("https://api.pexels.com", 500, "Server Error", {}, None)
    return boom


def test_pexels_error_does_not_discard_the_other_sources(monkeypatch):
    """Живой дефект (25.09): ошибка Pexels 500 на одном запросе выбрасывала
    кучу фото целиком, вместе с музеями, Commons и архивами."""
    sys.argv = ["pipeline_smart.py", REPO]
    import pipeline_smart as ps
    others = ("_shelf_search_photos", "_museum_search_photos", "_commons_search_photos",
              "_openverse_search_photos", "_pixabay_search_photos", "_unsplash_search_photos")
    for f in others:
        monkeypatch.setattr(ps, f, lambda q, *a, _f=f, **k: [{"id": f"{_f}:{q}", "alt": "",
                                                              "url": "", "src": {"medium": "x"}}])
    calls = []
    monkeypatch.setattr(ps, "_pexels_search_photos", _pexels_500(calls))
    monkeypatch.setattr(ps, "local_stock_candidate", lambda index: None)
    monkeypatch.setattr(ps, "PEXELS_FAIL_STREAK", 0)
    monkeypatch.setattr(ps, "PEXELS_BROKEN", False)
    req = _slot_request(ps)
    names = [n for n, _j in ps.PHOTO_ADAPTER.source_jobs(req, "medieval dagger")]
    pool = selection_engine.build_pool(req, ps.PHOTO_ADAPTER)
    ids = {c["id"] for c in pool}
    for f in others:
        src = {"_shelf_search_photos": "shelf", "_museum_search_photos": "museum",
               "_commons_search_photos": "commons", "_openverse_search_photos": "openverse",
               "_pixabay_search_photos": "pixabay", "_unsplash_search_photos": "unsplash"}[f]
        if src in names:
            assert any(i.startswith(f + ":") for i in ids), f
    assert len(calls) == 1                     # упавший Pexels в слоте больше не спрашивают
    assert ps.PEXELS_FAIL_STREAK == 1 and ps.PEXELS_BROKEN is False


def test_pexels_video_error_keeps_pixabay(monkeypatch):
    sys.argv = ["pipeline_smart.py", REPO]
    import pipeline_smart as ps
    monkeypatch.setattr(ps, "_pixabay_search_videos",
                        lambda q: [{"id": f"pixabay:{q}", "alt": "", "url": ""}])
    calls = []
    monkeypatch.setattr(ps, "_pexels_search_videos", _pexels_500(calls))
    monkeypatch.setattr(ps, "PEXELS_FAIL_STREAK", 0)
    pool = selection_engine.build_pool(_slot_request(ps), ps.VIDEO_ADAPTER)
    assert {c["id"] for c in pool} == {"pixabay:" + ps.stock_api_query(_slot_request(ps), q, video=True)
                                       for q in ("medieval dagger", "medieval sword")}
    assert len(calls) == 1 and ps.PEXELS_FAIL_STREAK == 1
