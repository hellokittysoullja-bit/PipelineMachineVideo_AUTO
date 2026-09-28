#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка финалистов, которых проверяют при любой сетке, идёт ВМЕСТЕ с
сеткой, а не после неё (27.09): ответы и число вызовов те же, меньше
только ожидание."""
import os
import sys
import threading

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402


def _cands(tmp_path, n, blocklisted=()):
    out = []
    for k in range(n):
        p = tmp_path / f"{k}.jpg"
        Image.new("RGB", (32, 32), (20 * k, 0, 0)).save(p)
        pp = {"id": str(k)}
        if k in blocklisted:
            pp["_blocklisted"] = True
        out.append({"path": str(p), "p": pp, "is_dup_free": 1})
    return out


def _setup(monkeypatch, grid, verify):
    import shot_judge
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    monkeypatch.setattr(shot_judge, "judge", grid)
    monkeypatch.setattr(shot_judge, "verify_claims", verify)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    monkeypatch.setattr(ps, "episode_world_card", lambda video_dir=None: None)
    monkeypatch.setattr(ps, "_rank_look_ties", lambda *a, **k: None)
    ps.JUDGE_GAPS.clear()


def test_sure_finalists_are_verified_while_the_grid_is_asked(monkeypatch, tmp_path):
    cs = _cands(tmp_path, 8, blocklisted=(7,))
    started = threading.Event()
    asked = []
    lock = threading.Lock()

    def verify(gw, model, *, path, **k):
        with lock:
            asked.append(os.path.basename(path))
        started.set()
        return {"claims": {"c1": "yes"}, "medium": "photo"}, {"call": True}

    def grid(gw, model, *, candidates, report=None, **k):
        # Сетка «думает», пока не увидит, что проверка уже идёт; прежний
        # порядок (проверка после сетки) ждал бы здесь впустую.
        report["overlap"] = started.wait(5)
        return {cid: (3 if cid == "6" else 1) for cid, _p in candidates}
    _setup(monkeypatch, grid, verify)
    assert ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", cs)
    log = [e for e in ps.SHOT_JUDGE_LOG if "scores" in e][-1]
    assert log["overlap"] is True, "проверка не началась до ответа сетки"
    # Проверены: первые 5 по каскаду, помеченный словарём (7) и лучшие по
    # сетке (6 — оценка 3) — каждый ровно один раз.
    assert sorted(asked) == sorted(f"{k}.jpg" for k in (0, 1, 2, 3, 4, 6, 7))


def test_sure_finalists_match_the_grid_independent_part_of_the_choice(tmp_path):
    cs = _cands(tmp_path, 9, blocklisted=(8,))
    for c in cs:
        c["judge"] = 0
    sure = ps.sure_finalists(cs)
    chosen = ps.verify_finalists_of(cs)
    assert all(any(c is s for c in chosen) for s in sure)
    assert [c["p"]["id"] for c in sure] == ["0", "1", "2", "3", "4", "8"]


def test_answers_are_the_same_as_in_sequence(monkeypatch, tmp_path):
    """Тот же слот с ранней проверкой и без неё: одинаковые отметки
    кандидатов и одинаковые вызовы."""
    def run(early):
        cs = _cands(tmp_path, 7)
        calls = []

        def verify(gw, model, *, path, **k):
            calls.append(os.path.basename(path))
            yes = "yes" if path.endswith(("2.jpg", "5.jpg")) else "no"
            return {"claims": {"c1": yes}, "medium": "photo"}, {"call": True}

        def grid(gw, model, *, candidates, report=None, **k):
            return {cid: (3 if cid in ("5", "6") else 1) for cid, _p in candidates}
        # Свой контекст на прогон, а не monkeypatch.undo(): undo снял бы и
        # рабочую папку из conftest, и голоса мира второго прогона ушли бы в
        # корень репозитория (так и было: media_plan/world_breaker.json).
        with monkeypatch.context() as m:
            _setup(m, grid, verify)
            if not early:
                m.setattr(ps, "sure_finalists", lambda judged: [])
            ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", cs)
        marks = [(c["p"]["id"], c.get("verify"), c.get("judge"), bool(c.get("_asked"))) for c in cs]
        return marks, sorted(calls)
    assert run(True) == run(False)
