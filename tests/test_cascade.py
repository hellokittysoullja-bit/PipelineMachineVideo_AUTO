#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Каскад: гейты и судья видят лучших по описанию кадра, а не первых по
порядку пула. Опыт эпизода 94: высшую оценку у первых 18 — 1/0/0 кадров,
у 18 лучших по локальной модели — 8/3/7."""
import os
import sys

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402


def _cands(n):
    return [{"id": f"c{k}", "src": {"large": f"http://x/{k}.jpg"}} for k in range(n)]


def _probe(fail=()):
    def probe(p, dest):
        if p["id"] in fail:
            raise OSError("нет превью")
        Image.new("RGB", (8, 8), (int(p["id"][1:]) * 20 % 255, 0, 0)).save(dest, "JPEG")
    return probe


def test_best_by_brief_go_first_failed_after_tail_last(tmp_path, monkeypatch):
    cands = _cands(5)
    rel = {"c0": 0.01, "c1": 0.05, "c2": None, "c3": 0.09, "c4": 0.02}
    monkeypatch.setattr(ps, "CASCADE_PREVIEW_N", 4)

    def fake_batch(paths, text):
        assert text == "a rondel dagger"
        return [rel[os.path.basename(p).split("casc_")[1].split(".")[0]] for p in paths]
    monkeypatch.setattr(ps, "clip_relevance_batch", fake_batch)
    order, paths = ps.cascade_reorder(cands, "a rondel dagger", str(tmp_path / "cf.jpg"),
                                      _probe(fail={"c2"}), 0)
    assert [p["id"] for p in order] == ["c3", "c1", "c0", "c2", "c4"]
    assert set(paths) == {id(cands[k]) for k in (0, 1, 3)}, "c2 не скачался — превью у него нет"
    assert all(os.path.exists(v) for v in paths.values())


def test_no_model_keeps_order_and_leaves_no_files(tmp_path, monkeypatch):
    cands = _cands(4)
    monkeypatch.setattr(ps, "clip_relevance_batch", lambda paths, text: [None] * len(paths))
    order, paths = ps.cascade_reorder(cands, "x", str(tmp_path / "cf.jpg"), _probe(), 0)
    assert order is cands and paths == {}
    assert not [f for f in os.listdir(tmp_path) if "casc_" in f]


def test_cascade_runs_only_with_an_active_judge():
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    i = src.index("candidates, casc_paths = cascade_reorder(")
    assert "if shot_judge_active():" in src[i - 120:i]
    assert "CASCADE_PREVIEW_N" in src[src.index("def shot_judge_signature"):][:1500]


def test_batch_matches_single_image_scores(tmp_path):
    ps_ok = ps.clip_relevance(str(_img(tmp_path, "a", (200, 30, 30))), "a red square")
    if ps_ok is None:
        import pytest
        pytest.skip("модель гейта недоступна")
    paths = [str(_img(tmp_path, n, c)) for n, c in (("a", (200, 30, 30)), ("b", (20, 20, 220)))]
    one = [ps.clip_relevance(p, "a red square") for p in paths]
    batch = ps.clip_relevance_batch(paths + [str(tmp_path / "missing.jpg")], "a red square")
    assert batch[2] is None
    assert all(abs(a - b) < 1e-4 for a, b in zip(one, batch[:2]))


def _img(tmp_path, name, color):
    p = tmp_path / f"{name}.jpg"
    Image.new("RGB", (64, 64), color).save(p)
    return p
