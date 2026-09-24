#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Одна власть над вердиктом кадра: судья старше порогов эмбеддинга.

Найдено по коду 25.09: вердикты relevance/stock писались победителю
независимо от судьи, а known_bad_reason() делала из них «брак» — кадр,
который судья посмотрел по описанию и одобрил, уходил на вторую страницу,
на спасение фотографией и в поглощение соседом."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402

from _video_world import QUERY, infra, video  # noqa: E402,F401
from test_repick_exclusion import _judge, _select_video  # noqa: E402


def test_judge_approval_outranks_the_embedding_thresholds(infra, monkeypatch):
    """Весь пул ниже порога CLIP, судья поставил победителю высшую оценку:
    кадр на экране, а не в поглощении."""
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = set()
    _judge(monkeypatch, {1: 3, 2: 1})
    res, att = _select_video()
    kinds = {k for k, _ in att.verdicts}
    assert res is not None
    assert {"relevance", "stock"} <= kinds, "слабые вердикты по-прежнему записаны в журнал"
    assert ps.JUDGE_APPROVED_VERDICT in kinds
    assert ps.known_bad_reason(att.verdicts) is None


def test_judge_rejection_is_still_the_strongest():
    v = [("stock", {}), ("judge", {})]
    assert ps.known_bad_reason(v) == "shot_judge_rejected"


def test_without_a_judge_the_thresholds_decide_as_before():
    assert ps.known_bad_reason([("stock", {}), ("relevance", {})]) == "stock_exhausted"


def test_approved_frame_leaves_no_miss_in_the_reports(monkeypatch):
    monkeypatch.setattr(ps, "STOCK_EXHAUSTED_MISSES", [])
    monkeypatch.setattr(ps, "RELEVANCE_GATE_MISSES", [])
    ps._project_verdicts([("stock", {"index": 0}), ("relevance", {"index": 0}),
                          (ps.JUDGE_APPROVED_VERDICT, {"index": 0})])
    assert ps.STOCK_EXHAUSTED_MISSES == [] and ps.RELEVANCE_GATE_MISSES == [], \
        "отчёт не называет браком кадр, который стоит на экране по решению судьи"


def test_both_media_kinds_record_the_approval():
    src = open(ps.__file__, encoding="utf-8").read()
    assert src.count("record_verdict(JUDGE_APPROVED_VERDICT") == 2


def test_cache_hit_brings_back_the_verdicts_of_the_attempt_that_made_the_frame(tmp_path):
    """Повторный рендер берёт кадр из кэша без отбора. Раньше вердикты
    терялись: кадр, признанный браком и поглощённый в первом прогоне, во
    втором вставал на экран."""
    cf = str(tmp_path / "0003_x.jpg")
    open(cf, "wb").write(b"x")
    first = ps.new_attempt(3, "photo")
    with ps.selection_attempt.activate(first):
        ps.selection_attempt.record_verdict("judge", {"index": 3, "score": 0})
        ps.write_media_sidecar(cf, pexels_id=1, query="q", kind="photo")
    first.discard()
    again = ps.new_attempt(3, "photo")
    with ps.selection_attempt.activate(again):
        ps._restore_cached_quality(cf)
    assert ps.known_bad_reason(again.verdicts) == "shot_judge_rejected"
    again.discard()


def test_old_sidecar_without_verdicts_restores_nothing(tmp_path):
    cf = str(tmp_path / "0004_x.jpg")
    open(cf, "wb").write(b"x")
    ps.write_media_sidecar(cf, pexels_id=1, query="q", kind="photo")
    att = ps.new_attempt(4, "photo")
    with ps.selection_attempt.activate(att):
        ps._restore_cached_quality(cf)
    assert att.verdicts == []
    att.discard()
