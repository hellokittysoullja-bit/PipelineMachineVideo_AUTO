#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Упреждающий отбор слотов: песочница не меняет того, от чего зависят
решения эпизода, и отдаёт настоящему циклу только готовые кэши."""
import os
import sys
import threading
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import slot_speculation  # noqa: E402


def _ps():
    import pipeline_smart as ps
    return ps


class speculating_ctx:
    def __init__(self, ps):
        self.ps = ps

    def __enter__(self):
        self.t1 = self.ps._SPECULATING.set(True)
        self.t2 = self.ps._SPEC_LOGS.set({})
        return self

    def __exit__(self, *exc):
        self.ps._SPEC_LOGS.reset(self.t2)
        self.ps._SPECULATING.reset(self.t1)
        return False


def test_logs_of_speculation_are_its_own(monkeypatch):
    ps = _ps()
    monkeypatch.setattr(ps, "SHOT_JUDGE_LOG", [])
    with speculating_ctx(ps):
        ps._log_list("SHOT_JUDGE_LOG").append({"index": 3, "verify": {"why": "x"}})
        assert ps._log_list("SHOT_JUDGE_LOG") == [{"index": 3, "verify": {"why": "x"}}]
    assert ps.SHOT_JUDGE_LOG == [], "запись упреждения не попала в журнал прогона"


def test_decision_state_is_untouched_by_speculation(monkeypatch):
    ps = _ps()
    monkeypatch.setattr(ps, "SOURCE_STATS", {})
    monkeypatch.setattr(ps, "PEXELS_FAIL_STREAK", 0)
    monkeypatch.setattr(ps, "PEXELS_BROKEN", False)
    monkeypatch.setattr(ps, "PEXELS_LOW_PRIORITY_SKIPPED", 0)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", 3)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_RESERVE", 10)
    votes = {}
    monkeypatch.setattr(ps, "_world_votes", lambda: votes)
    with speculating_ctx(ps):
        ps._source_bump("pexels", "offered")
        ps._note_pexels_failure(IOError("x"), "t")
        ps._record_world_vote(5, 3, 3)
        assert ps.pexels_query_allowed("q", {}, low_priority=True) is False
    assert ps.SOURCE_STATS == {} and ps.PEXELS_FAIL_STREAK == 0 and not ps.PEXELS_BROKEN
    assert ps.PEXELS_LOW_PRIORITY_SKIPPED == 0, "пропуск квоты считает только настоящий цикл"
    assert votes == {}, "голос мира подаёт только настоящий цикл"


def test_speculation_decides_pexels_quota_by_its_own_slot(monkeypatch):
    ps = _ps()
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", 50)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_RESERVE", 100)     # запас текущего слота цикла
    with speculating_ctx(ps):
        t = ps._SPEC_QUOTA_RESERVE.set(20)                   # запас упреждаемого слота
        try:
            assert ps.pexels_query_allowed("q", {}, low_priority=True) is True
        finally:
            ps._SPEC_QUOTA_RESERVE.reset(t)


def test_speculative_attempts_have_their_own_ids(tmp_path, monkeypatch):
    ps = _ps()
    import selection_attempt
    selection_attempt.reset_attempt_ids()
    with speculating_ctx(ps):
        a = ps.new_attempt(4, "photo")
    b = ps.new_attempt(4, "photo")
    assert a.attempt_id.startswith("spec") and b.attempt_id == "4-photo-1", \
        "упреждение не сдвигает нумерацию настоящих попыток"
    a.discard()
    b.discard()


def test_download_made_by_speculation_is_handed_to_the_real_loop(tmp_path, monkeypatch):
    ps = _ps()
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    calls = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return Resp(b"IMAGEBYTES")
    monkeypatch.setattr(ps.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ps, "_download_host_throttle", lambda url: None)
    import urllib.request
    url = "https://cdn.example/x.jpg"
    with speculating_ctx(ps):
        ps.atomic_url_download(urllib.request.Request(url), str(tmp_path / "spec.jpg"), 5)
    ps.atomic_url_download(urllib.request.Request(url), str(tmp_path / "real.jpg"), 5)
    assert calls == [url], "настоящий цикл взял файл упреждения, а не сеть"
    assert (tmp_path / "real.jpg").read_bytes() == b"IMAGEBYTES"
    ps.atomic_url_download(urllib.request.Request(url), str(tmp_path / "again.jpg"), 5)
    assert len(calls) == 2, "копия отдаётся один раз, дальше — как без упреждения"
    ps.discard_speculated_downloads()
    assert not (tmp_path / "speculated_downloads").exists()


def test_speculator_runs_ahead_and_real_loop_waits():
    done = []
    gate = threading.Event()

    def job(j, snap):
        if j == 2:
            gate.wait(5)
        time.sleep(0.01)
        done.append((j, snap))
    s = slot_speculation.SlotSpeculator(5, job, depth=3, workers=2)
    s.advance(0, lambda: "snap0")
    assert s.stats["scheduled"] == 3
    gate.set()
    s.wait(2)
    assert (2, "snap0") in done, "цикл дождался упреждения слота 2"
    s.advance(1, lambda: "snap1")       # 2 и 3 уже поставлены, новый — только 4
    assert s.stats["scheduled"] == 4
    s.close()


def test_speculator_swallows_job_errors():
    def job(j, snap):
        raise RuntimeError("boom")
    s = slot_speculation.SlotSpeculator(3, job, depth=2, workers=1)
    s.advance(0, lambda: None)
    s.wait(1)
    s.wait(2)
    s.close()
    assert s.stats["failed"] == 2


def test_quiet_output_hides_only_speculation_prints(capsys):
    flag = threading.local()
    with slot_speculation.quiet_output(lambda: getattr(flag, "on", False)):
        print("видно")
        flag.on = True
        print("не видно")
        flag.on = False
    assert capsys.readouterr().out == "видно\n"


def test_ladder_and_request_are_one_code_for_both_paths():
    """Настоящий цикл и упреждение зовут ОДНИ функции: разойтись по коду
    выбора слота они не могут."""
    ps = _ps()
    import inspect
    main_src = inspect.getsource(ps.main)
    assert main_src.count("build_slot_selection(") == 2 and main_src.count("run_slot_ladder(") == 2
