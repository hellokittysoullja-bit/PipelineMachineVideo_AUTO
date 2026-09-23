#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Слепые листы разметки пулов: без сети, без чисел модели на листе."""
import json
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pool_sheet as ps  # noqa: E402


def _pools(tmp_path):
    recs = [
        {"index": 0, "query": "q", "shot_brief": "a dagger", "block_text": "Вот кинжал.",
         "extra_queries": [], "pool": [{"id": 1, "channel": "met", "via": "q", "text": "dagger",
                                        "probe_url": "file://x/1", "headers": {}},
                                       {"id": 2, "channel": "pexels", "via": "q", "text": "knife",
                                        "probe_url": "file://x/2", "headers": {}}]},
    ]
    recs.append(dict(recs[0]))                     # тот же пул того же слота — один лист
    recs.append(dict(recs[0], pool=recs[0]["pool"][:1]))   # другой пул того же слота — второй
    p = tmp_path / "pools.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    return str(p)


def test_same_pool_of_a_slot_is_one_sheet_different_pool_is_another(tmp_path):
    got = ps.read_pools(_pools(tmp_path))
    assert [(r["index"], r["call"]) for r in got] == [(0, 0), (0, 1)]


def test_build_and_mark_without_network(tmp_path, monkeypatch):
    def fake_fetch(url, headers, dst, timeout=30):
        Image.new("RGB", (40, 30), (100, 0, 0)).save(dst)
        return True
    monkeypatch.setattr(ps, "fetch", fake_fetch)
    out = tmp_path / "out"
    assert ps.main(["build", _pools(tmp_path), "--out", str(out), "--tag", "ep"]) == 0
    assert (out / "sheets" / "ep_000.jpg").exists() and (out / "sheets" / "ep_000_1.jpg").exists()
    doc = json.load(open(out / "labels" / "ep_000.json", encoding="utf-8"))
    assert [c["label"] for c in doc["candidates"]] == [None, None]
    ps.main(["mark", str(out), "ep_000", "--s3", "0", "--rest", "0"])
    doc = json.load(open(out / "labels" / "ep_000.json", encoding="utf-8"))
    assert [c["label"] for c in doc["candidates"]] == ["3", "0"]
    ps.main(["build", _pools(tmp_path), "--out", str(out), "--tag", "ep"])
    doc = json.load(open(out / "labels" / "ep_000.json", encoding="utf-8"))
    assert [c["label"] for c in doc["candidates"]] == ["3", "0"], "пересборка не стирает разметку"


def test_mark_refuses_double_labels_and_unknown_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "fetch", lambda *a, **k: False)
    out = tmp_path / "out"
    ps.main(["build", _pools(tmp_path), "--out", str(out), "--tag", "ep"])
    with pytest.raises(SystemExit):
        ps.main(["mark", str(out), "ep_000", "--s3", "0", "--s0", "0"])
    with pytest.raises(SystemExit):
        ps.main(["mark", str(out), "ep_000", "--s3", "7"])


def test_sheet_labels_use_the_judge_scale():
    import shot_judge
    assert set(ps.LABELS) == {str(k) for k in range(shot_judge.SCORE_MAX + 1)}
