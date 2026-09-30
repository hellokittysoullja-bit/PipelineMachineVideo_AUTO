#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Читаемость кадра стоит выше оценки судьи. Живой случай эпизода 94:
почти чёрный кадр (93% пикселей почти чёрные) получил от судьи 3 за
«блик на острие» и встал на экран вместо кинжала."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures")


def test_the_live_dark_frame_is_unreadable():
    # Pexels 7033822 (Pexels License), уменьшен; победитель слота 3 прогона judge5.
    assert ps.frame_readable(os.path.join(FIX, "unreadable", "ep94_needle_dark.jpg")) == 0


def test_no_labelled_good_or_tolerable_frame_is_unreadable():
    """Порог не режет осознанно тёмный грейд канала: самый тёмный годный
    кадр золотого набора — 56% почти чёрных пикселей."""
    items = json.load(open(os.path.join(FIX, "golden_set", "manifest.json"), encoding="utf-8"))["items"]
    bad = [it["id"] for it in items if it["verdict"] in ("good", "tolerable")
           and not ps.frame_readable(os.path.join(FIX, "golden_set", "images",
                                                  os.path.basename(it["image"])))]
    assert bad == []


def _c(cid, judge, readable):
    return {"p": {"id": cid}, "path": cid, "is_dup_free": 1, "size_ok": 1, "is_relevant": 1,
            "sharp_ok": 1, "aesthetic_val": 5.0, "luma_score": 0.0, "min_d": 10,
            "relevance": 0.1, "judge": judge, "is_readable": readable}


def test_readable_beats_a_higher_judge_score():
    dark, dagger = _c("dark", 3, 0), _c("dagger", 2, 1)
    base, _d = ps._score_and_pick([dark, dagger])
    assert base is dagger


def test_unreadable_is_not_shown_to_the_judge(tmp_path, monkeypatch):
    from PIL import Image
    paths = []
    for k in range(2):
        p = tmp_path / f"{k}.jpg"
        Image.new("RGB", (8, 8), (100, 100, 100)).save(p)
        paths.append(str(p))
    info = [dict(_c("dark", None, 0), path=paths[0]), dict(_c("ok", None, 1), path=paths[1])]
    seen = {}

    def fake(gw, model, *, phrase, brief, candidates, cache_dir=None, report=None, kind, setting=None):
        seen["ids"] = [cid for cid, _p in candidates]
        return {cid: 2 for cid, _p in candidates}
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge", fake)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    ps.judge_candidates(0, "photo", "x", "y", info)
    assert seen["ids"] == ["ok"]
