# -*- coding: utf-8 -*-
"""Арифметика стенда «реранкер против судьи» (scripts/rerank_vs_judge.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import rerank_vs_judge as r  # noqa: E402


def test_metrics_on_a_known_slot():
    tiles = {"0:photo": [{"id": "a", "label": 0, "pos": 0}, {"id": "b", "label": 2, "pos": 1},
                         {"id": "c", "label": 1, "pos": 2}, {"id": "d", "label": 2, "pos": 5}]}
    judge = {"0:photo": {"pick": 1, "pick_id": "c"}}
    scores = {("0:photo", "a"): 0.9, ("0:photo", "b"): 0.8, ("0:photo", "c"): 0.1, ("0:photo", "d"): 0.5}
    ev = r.evaluate(tiles, judge, scores, handoff=3)
    # пары с разной меткой: (a,b) (a,c) (a,d) (b,c) (c,d); верно упорядочены (b,c) и (c,d)
    assert ev["pairs"] == 5 and ev["pairs_ok"] == 2.0
    # в первых 3 (a,b,c) реранкер ставит a — брак; лучший (2) — b
    assert (ev["rr_best"], ev["rr_brak"], ev["rr_label_sum"]) == (0, 1, 0)
    assert (ev["judge_best"], ev["judge_brak"]) == (0, 0)
    # выбор судьи c — третий по реранкеру, третий по каскаду
    assert ev["judge_pick_in_rr_top"][2] == 0 and ev["judge_pick_in_rr_top"][3] == 1
    assert ev["judge_pick_in_cascade_top"][3] == 1
    assert ev["best_in_rr_top"][1] == 0 and ev["best_in_rr_top"][2] == 1
    assert ev["best_in_cascade_top"][2] == 1
