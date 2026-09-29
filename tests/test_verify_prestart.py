"""Проверка первых по каскаду финалистов идёт одновременно с сеткой судьи
(аудит 29.09): те же вопросы, по одному разу, на один ход судьи меньше."""
import os
import sys
import tempfile
import threading
import time

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402
import shot_judge  # noqa: E402

ORIG_PRESTART = ps.prestart_verify


def _cands(tmp_path, n):
    out = []
    for k in range(n):
        p = tmp_path / f"c{k}.jpg"
        Image.new("RGB", (64, 48), (k * 20 % 255, 10, 10)).save(p)
        out.append({"p": {"id": f"c{k}"}, "path": str(p), "is_dup_free": 1, "is_readable": 1})
    return out


class _Log:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = []

    def add(self, what):
        with self.lock:
            self.events.append((time.perf_counter(), what))


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "SHOT_JUDGE_LOG", [])
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp_smart"))
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    monkeypatch.setattr(ps, "shot_judge_model", lambda: "m")
    monkeypatch.setattr(ps, "episode_world_card", lambda video_dir=None: None)
    monkeypatch.setattr(ps, "_rank_look_ties", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_judge_budget_forecast", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_record_world_vote", lambda *a, **k: None)
    monkeypatch.setattr(ps, "SHOT_JUDGE_PAID_SLOTS", 25)
    log = _Log()

    def fake_verify(gw, model, *, path, **kw):
        log.add(("verify_start", os.path.basename(path)))
        time.sleep(0.2)
        log.add(("verify_end", os.path.basename(path)))
        return {"claims": {}, "main_in_world": True, "background": "none"}, {}
    monkeypatch.setattr(shot_judge, "verify_claims", fake_verify)
    return log


def _grid(log, scores):
    def judge(gw, model, *, candidates, **kw):
        log.add(("grid_start", None))
        time.sleep(0.4)
        log.add(("grid_end", None))
        return None if scores is None else {cid: scores.get(cid, 1) for cid, _p in candidates}
    return judge


def _run(env, tmp_path, monkeypatch, scores, early_on):
    env.events.clear()
    monkeypatch.setattr(shot_judge, "judge", _grid(env, scores))
    monkeypatch.setattr(ps, "prestart_verify", ORIG_PRESTART if early_on else (lambda ex, judged, ask: {}))
    cands = _cands(tmp_path, 10)
    ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", cands, spec=None)
    return list(env.events), cands


@pytest.mark.parametrize("scores", [{"c9": 3, "c8": 3}, None])
def test_first_by_cascade_are_verified_during_the_grid_once_each(env, tmp_path, monkeypatch, scores):
    base, cb = _run(env, tmp_path, monkeypatch, scores, early_on=False)
    ev, ce = _run(env, tmp_path, monkeypatch, scores, early_on=True)
    asked = lambda e: [x for _t, (w, x) in e if w == "verify_start"]
    assert sorted(asked(ev)) == sorted(asked(base)), "те же вопросы, что без раннего запуска"
    assert len(asked(ev)) == len(set(asked(ev))), "каждый вопрос — один раз"
    grid_end = next(t for t, (w, _x) in ev if w == "grid_end")
    early = [x for t, (w, x) in ev if w == "verify_start" and t < grid_end]
    k = min(ps.VERIFY_FINALISTS, len(ce))
    assert sorted(early) == sorted(f"c{j}.jpg" for j in range(k)), "первые по каскаду — пока думает сетка"
    keys = ("judge", "verify", "verify_focus", "verify_nothing", "verify_perfect", "world_clear",
            "verify_skipped", "_asked")
    assert [{k: c.get(k) for k in keys} for c in ce] == [{k: c.get(k) for k in keys} for c in cb]
