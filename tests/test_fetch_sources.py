#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Источники кучи опрашиваются параллельно, каждый своей очередью: куча та
же до кандидата, каждый источник видит запросы в прежнем порядке, сбой
источника уходит наверх, как раньше, а время — максимум по источникам,
а не сумма."""
import os
import sys
import threading
import time
import types

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
    with pytest.raises(ValueError, match="archive упал на q2"):
        selection_engine.build_pool(_request(), _Adapter(fail=("archive", "q2")))


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
