#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Каскад: гейты и судья видят лучших по описанию кадра из всего пула, а не
первых по порядку. Опыт эпизода 94 (9 фото-слотов): слотов с лучшим «3» —
3 у первых 18, 7 у лучших 18 всего пула; ни один не стал хуже."""
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402


def _cands(n):
    return [{"id": f"c{k}", "src": {"large": f"http://x/{k}.jpg"}} for k in range(n)]


def _world(tmp_path, monkeypatch, rel, fail=()):
    """rel: {id: сходство с описанием}; эмбеддинг картинки кодирует id цветом."""
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp"))
    monkeypatch.setattr(ps, "_CASCADE_EMB", {})
    calls = {"images": 0}
    ids = sorted(rel)

    def probe(p, dest):
        if p["id"] in fail:
            raise OSError("нет превью")
        Image.new("RGB", (8, 8), (ids.index(p["id"]) * 10, 0, 0)).save(dest, "JPEG")

    def embed(images=None, text=None):
        if text is not None:
            return np.ones((1, 1), dtype="float32")
        calls["images"] += len(images)
        return np.array([[rel[ids[round(im.getpixel((4, 4))[0] / 10)]]] for im in images], "float32")
    monkeypatch.setattr(ps, "_gate_embed", embed)
    return probe, calls


def test_best_by_brief_first_failed_after(tmp_path, monkeypatch):
    rel = {"c0": 0.01, "c1": 0.05, "c2": 0.9, "c3": 0.09, "c4": 0.02}
    probe, _ = _world(tmp_path, monkeypatch, rel, fail={"c2"})
    order = ps.cascade_reorder(_cands(5), "a rondel dagger", str(tmp_path / "cf.jpg"), probe, 0)
    assert [p["id"] for p in order] == ["c3", "c1", "c4", "c0", "c2"]
    assert not [f for f in os.listdir(tmp_path) if "casc_" in f], "превью не остаются на диске"


def test_window_limits_ranking_tail_keeps_order(tmp_path, monkeypatch):
    rel = {"c0": 0.01, "c1": 0.05, "c2": 0.9, "c3": 0.09}
    probe, _ = _world(tmp_path, monkeypatch, rel)
    monkeypatch.setenv("CASCADE_PREVIEW_N", "2")
    order = ps.cascade_reorder(_cands(4), "x", str(tmp_path / "cf.jpg"), probe, 0)
    assert [p["id"] for p in order] == ["c1", "c0", "c2", "c3"]


def test_embeddings_are_cached_across_slots(tmp_path, monkeypatch):
    rel = {"c0": 0.1, "c1": 0.2, "c2": 0.3}
    probe, calls = _world(tmp_path, monkeypatch, rel)
    ps.cascade_reorder(_cands(3), "x", str(tmp_path / "a.jpg"), probe, 0)
    assert calls["images"] == 3
    monkeypatch.setattr(ps, "_CASCADE_EMB", {})        # новый процесс — кэш на диске
    ps.cascade_reorder(_cands(3), "y", str(tmp_path / "b.jpg"), probe, 1)
    assert calls["images"] == 3, "картинка оценивается один раз, в любом слоте и прогоне"


def test_no_model_keeps_order(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_gate_embed", lambda images=None, text=None: None)
    cands = _cands(4)
    assert ps.cascade_reorder(cands, "x", str(tmp_path / "cf.jpg"), lambda p, d: None, 0) is cands


def test_several_claims_rank_by_the_worst_place(tmp_path, monkeypatch):
    """«Стрела скользит по нагруднику»: нагрудник без стрелы отлично
    совпадает с «нагрудником» и проваливает «стрелу» — он ниже картины, где
    есть и то, и другое, и ниже полёта стрелы (равное худшее место —
    решает главное, первое утверждение)."""
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp"))
    monkeypatch.setattr(ps, "_CASCADE_EMB", {})
    vec = {"breast": (0.1, 0.9), "painting": (0.8, 0.5), "flying": (0.9, 0.0)}
    ids = sorted(vec)

    def probe(p, dest):
        Image.new("RGB", (8, 8), (ids.index(p["id"]) * 10, 0, 0)).save(dest, "JPEG")

    def embed(images=None, text=None):
        if text is not None:
            return np.array([[1.0, 0.0]] if text == "arrow" else [[0.0, 1.0]], "float32")
        return np.array([vec[ids[round(im.getpixel((4, 4))[0] / 10)]] for im in images], "float32")
    monkeypatch.setattr(ps, "_gate_embed", embed)
    cands = [{"id": k, "src": {"large": f"http://x/{k}.jpg"}} for k in ("breast", "painting", "flying")]
    order = ps.cascade_reorder(cands, ["arrow", "breastplate"], str(tmp_path / "cf.jpg"), probe, 0)
    assert [p["id"] for p in order] == ["painting", "flying", "breast"]


def test_cascade_texts_come_from_the_must_claims():
    spec = {"claims": [{"id": "c1", "text": "an arrow", "tier": "must"},
                       {"id": "c2", "text": "a wall", "tier": "should"},
                       {"id": "c3", "text": "it bounces", "tier": "must", "motion": True}]}
    assert ps.cascade_texts(spec, "brief", "video") == ["an arrow", "it bounces"]
    assert ps.cascade_texts(spec, "brief", "photo") == ["an arrow"], "движение фото не ранжирует"
    assert ps.cascade_texts(None, "brief") == ["brief"]


def test_cascade_runs_only_with_an_active_judge():
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    i = src.index("candidates = cascade_reorder(")
    assert "if shot_judge_active(index):" in src[i - 80:i]
    assert "cascade_preview_n()" in src[src.index("def shot_judge_signature"):][:3000]


def test_separate_embeddings_match_the_gate_score(tmp_path):
    p = tmp_path / "r.jpg"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(p)
    single = ps.clip_relevance(str(p), "a red square")
    if single is None:
        import pytest
        pytest.skip("модель гейта недоступна")
    with Image.open(p) as im:
        img = ps._gate_embed(images=[im.convert("RGB")])
    txt = ps._gate_embed(text="a red square")
    assert abs(float(img[0] @ txt[0]) - single) < 1e-4


def test_second_page_is_scoped_and_skips_the_first_page():
    assert ps.CASCADE_PAGE.get() == 0
    with ps.cascade_page(1):
        assert ps.CASCADE_PAGE.get() == 1
    assert ps.CASCADE_PAGE.get() == 0
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    assert "skip = CASCADE_PAGE.get() * _photo_dedup_max_tries_for(index)" in src


def test_second_page_runs_only_for_a_missing_or_known_bad_frame():
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    body = src[src.index("\ndef main("):]
    i = body.index("with cascade_page(1):")
    head = body[i - 400:i]
    assert "known_bad_reason(cur_att.verdicts)" in head and "shot_judge_active(i)" in head
    tail = body[i:i + 500]
    assert "not known_bad_reason(page2_att.verdicts)" in tail, "брак второй страницы не заменяет кадр"
