#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Судья кадров внутри отбора: где стоит его оценка и что он НЕ меняет."""
import os
import sys

import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import selection_attempt as sa  # noqa: E402


def _c(cid, *, judge=None, relevant=1, size_ok=1, aesthetic=5.0, dup_free=1, path="x"):
    return {"path": path, "p": {"id": cid}, "is_dup_free": dup_free, "size_ok": size_ok,
            "is_relevant": relevant, "sharp_ok": 1, "aesthetic_val": aesthetic,
            "luma_score": 0.0, "min_d": 10, "relevance": 0.1, "judge": judge}


def test_judge_score_outranks_embedding_gates():
    """Живой случай эпизода 94: эмбеддинг пропустил современный нож и
    отверг подлинный кинжал ловушкой. Судья сильнее — его оценка решает."""
    knife = _c("knife", judge=0, relevant=1, aesthetic=9.0)
    dagger = _c("dagger", judge=3, relevant=0, aesthetic=4.0)
    assert ps._score_and_pick([knife, dagger])[0] is dagger


def test_duplicate_stays_first_key():
    assert ps._score_and_pick([_c("dup", judge=3, dup_free=0), _c("ok", judge=1)])[0]["p"]["id"] == "ok"


def test_without_judge_the_order_is_exactly_as_before():
    """Судьи не было — у всех judge=None; победитель тот же, что без ключа."""
    a = [_c("a", relevant=0, aesthetic=9.0), _c("b", relevant=1, aesthetic=1.0)]
    b = [dict(c, judge=None) for c in a]
    for c in a:
        del c["judge"]
    assert ps._score_and_pick(a)[0]["p"]["id"] == ps._score_and_pick(b)[0]["p"]["id"] == "b"


def test_inactive_judge_leaves_no_trace_in_the_selection_signature(monkeypatch):
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    without_key = ps._selection_stack_signature()
    monkeypatch.setenv("SHOT_JUDGE", "0")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    flag_off = ps._selection_stack_signature()
    assert without_key == flag_off and "judge" not in without_key
    monkeypatch.setenv("SHOT_JUDGE", "1")
    active = ps._selection_stack_signature()
    assert active != without_key and "judge" in active
    monkeypatch.setenv("SHOT_JUDGE_MODEL", "other/model")
    assert ps._selection_stack_signature() != active, "смена модели меняет победителя"


def test_judge_candidates_scores_only_non_duplicates(tmp_path, monkeypatch):
    paths = []
    for k in range(3):
        p = tmp_path / f"{k}.jpg"
        Image.new("RGB", (32, 32), (k * 60, 0, 0)).save(p)
        paths.append(str(p))
    info = [_c("a", path=paths[0]), _c("b", path=paths[1]), _c("dup", path=paths[2], dup_free=0)]
    seen = {}

    def fake_judge(gw, model, *, phrase, brief, candidates, cache_dir=None, report=None):
        seen["ids"] = [cid for cid, _p in candidates]
        seen["brief"], seen["phrase"] = brief, phrase
        return {"a": 3, "b": 1}
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge", fake_judge)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    assert ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", info)
    assert seen["ids"] == ["a", "b"] and seen["brief"] == "a dagger"
    assert [c["judge"] for c in info] == [3, 1, None]


def test_failed_judge_clears_every_score(monkeypatch):
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    info = [_c("a", judge=3, path=__file__), _c("b", judge=0, path=__file__)]
    assert not ps.judge_candidates(0, "photo", "p", "b", info)
    assert all(c["judge"] is None for c in info), "без смешивания оценённых с неоценёнными"


def test_judge_rejection_is_the_strongest_known_bad_reason():
    assert ps.known_bad_reason([("smart_veto", {}), ("judge", {})]) == "shot_judge_rejected"
    assert ps.VERDICT_REPORT_LISTS["judge"] == "SHOT_JUDGE_MISSES"


def test_approved_frame_is_not_vetoed_by_the_weaker_check():
    assert ps.judge_approved(_c("a", judge=2)) and not ps.judge_approved(_c("a", judge=1))
    assert not ps.judge_approved(None) and not ps.judge_approved(_c("a", judge=None))
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    assert "judge_approved(winner) or not smart_relevance_veto(cf, query)" in src


def test_tests_never_reach_the_paid_gateway():
    """conftest гасит судью: дефолт реестра 1, ключ мог быть в окружении."""
    assert os.environ.get("SHOT_JUDGE") == "0" and not os.environ.get("LLM_GATEWAY_API_KEY")
    assert not ps.shot_judge_active()
