#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WeMM-Embedding-9B как модель каскада (CASCADE_MODEL=wemm9b) и смешанный
ключ реранкера верха: формула та же, что в замере 29.09 на 72 размеченных
кучах (scratchpad/ens/analyze2.order_for, вариант "rr+ens")."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import wemm_embed  # noqa: E402


def _ps():
    import pipeline_smart
    return pipeline_smart


def reference_blend(rr, emb):
    """Формула замера дословно (numpy, население, nan — без оценки)."""
    s = np.array([np.nan if v is None else v for v in rr], float)
    ok = ~np.isnan(s)
    zs = np.zeros(len(s))
    if ok.sum() > 1:
        zs[ok] = (s[ok] - s[ok].mean()) / (s[ok].std() or 1)
    e = np.array(emb, float)
    key = zs + (e - e.mean()) / (e.std() or 1)
    return sorted(range(len(s)), key=lambda j: (-key[j], j))


def test_selected_reads_cascade_model(monkeypatch):
    monkeypatch.setenv("CASCADE_MODEL", "wemm9b")
    assert wemm_embed.selected() and _ps().cascade_model() == "wemm9b"
    monkeypatch.setenv("CASCADE_MODEL", "")
    assert not wemm_embed.selected() and _ps().cascade_model() == "qwen"


def test_signature_and_rerank_depth_follow_the_model(monkeypatch):
    ps = _ps()
    monkeypatch.delenv("CASCADE_RERANK_TOP", raising=False)
    monkeypatch.setenv("CASCADE_MODEL", "wemm9b")
    assert ps.cascade_model_signature() == wemm_embed.signature()
    assert wemm_embed.REVISION[:12] in wemm_embed.signature()
    assert ps.cascade_rerank_top() == 30
    monkeypatch.setenv("CASCADE_MODEL", "")
    import qwen_vl_embed
    assert ps.cascade_model_signature() == qwen_vl_embed.signature()
    assert ps.cascade_rerank_top() == ps.CASCADE_RERANK_TOP_DEFAULT


@pytest.mark.parametrize("rr,emb", [
    ([0.1, 0.9, 0.5, 0.2], [0.30, 0.10, 0.25, 0.40]),
    ([0.7, None, 0.2, 0.9, 0.1], [0.2, 0.9, 0.2, 0.1, 0.5]),
    ([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]),
])
def test_blend_matches_the_measured_formula(monkeypatch, tmp_path, rr, emb):
    ps = _ps()
    import qwen_vl_rerank
    monkeypatch.setenv("CASCADE_RERANK_TOP", str(len(rr)))
    top = [{"id": f"c{j}"} for j in range(len(rr))]
    tmp = {}
    for j, p in enumerate(top):
        f = tmp_path / f"p{j}.jpg"
        if rr[j] is not None:
            from PIL import Image
            Image.new("RGB", (8, 8)).save(f)
        tmp[id(p)] = str(f)

    def fake_score(text, paths, instruction=None):
        return [v for v in rr if v is not None]
    monkeypatch.setattr(qwen_vl_rerank, "score", fake_score)
    probe = lambda p, dest: (_ for _ in ()).throw(OSError("нет превью"))  # noqa: E731
    out = ps._rerank_top(top + [{"id": "tail"}], "a hand holds a dagger", tmp,
                         str(tmp_path / "cf"), probe, 0,
                         emb_best={id(p): v for p, v in zip(top, emb)})
    assert [p["id"] for p in out] == [f"c{j}" for j in reference_blend(rr, emb)] + ["tail"]


def test_without_emb_best_order_is_the_old_pure_reranker(monkeypatch, tmp_path):
    ps = _ps()
    import qwen_vl_rerank
    from PIL import Image
    monkeypatch.setenv("CASCADE_RERANK_TOP", "3")
    top = [{"id": f"c{j}"} for j in range(3)]
    tmp = {}
    for j, p in enumerate(top):
        f = tmp_path / f"p{j}.jpg"
        Image.new("RGB", (8, 8)).save(f)
        tmp[id(p)] = str(f)
    monkeypatch.setattr(qwen_vl_rerank, "score", lambda *a, **k: [0.1, 0.9, 0.5])
    out = ps._rerank_top(top, "x", tmp, str(tmp_path / "cf"), lambda p, d: None, 0)
    assert [p["id"] for p in out] == ["c1", "c2", "c0"]


def test_cascade_routes_to_wemm_and_passes_best_similarity(monkeypatch, tmp_path):
    ps = _ps()
    import qwen_vl_embed
    import qwen_vl_rerank
    monkeypatch.setenv("CASCADE_MODEL", "wemm9b")
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("CASCADE_RERANK_TOP", "3")
    monkeypatch.setattr(ps, "_CASCADE_EMB", {})
    monkeypatch.setattr(ps, "CLIP_BROKEN", False)
    monkeypatch.setattr(ps, "CLIP_ENABLED", True)
    calls = {"wemm": 0}

    def fake_images(imgs):
        calls["wemm"] += len(imgs)
        return np.stack([np.eye(4, dtype="float32")[k % 4] for k in range(len(imgs))])
    monkeypatch.setattr(wemm_embed, "embed_images", fake_images)
    monkeypatch.setattr(wemm_embed, "embed_text", lambda t: np.eye(4, dtype="float32")[1])
    monkeypatch.setattr(qwen_vl_embed, "embed_images",
                        lambda imgs: pytest.fail("каскад WeMM позвал Qwen-эмбеддинг"))
    seen = {}

    def fake_rerank(ranked, text, tmp, cf, probe_fn, index, emb_best=None):
        seen["emb_best"] = emb_best
        return ranked
    monkeypatch.setattr(ps, "_rerank_top", fake_rerank)
    cands = [{"id": f"c{j}", "n": j, "src": {"large": f"https://h/{j}.jpg"}} for j in range(4)]

    def probe(p, dest):
        from PIL import Image
        Image.new("RGB", (64, 48), (p["n"] * 40, 10, 10)).save(dest)
    out = ps.cascade_reorder(cands, ["a query"], str(tmp_path / "cf"), probe, index=0,
                             url_of=lambda p: p["src"]["large"], rerank_text="x")
    assert calls["wemm"] == 4 and out[0]["id"] == "c1"
    assert seen["emb_best"] is not None and len(seen["emb_best"]) == 4
    assert max(seen["emb_best"].values()) == pytest.approx(1.0)


def test_size_sorted_batches_keep_input_order(monkeypatch):
    """Пачки идут по размеру картинок, векторы возвращаются в порядке входа."""
    from PIL import Image
    import ml_device

    class Fake:
        def __init__(self):
            self.batches = []

        def encode_document(self, docs, **kw):
            self.batches.append([d["image"].size for d in docs])
            return np.array([[d["image"].size[0], d["image"].size[1]] for d in docs], float)
    fake = Fake()
    monkeypatch.setattr(wemm_embed, "_load", lambda: True)
    import queue
    free = queue.Queue()
    free.put("cuda:0")
    monkeypatch.setitem(wemm_embed._STATE, "models", {"cuda:0": fake})
    monkeypatch.setitem(wemm_embed._STATE, "free", free)
    monkeypatch.setattr(ml_device, "run", lambda fn, dev=None: fn())
    monkeypatch.setenv("WEMM_BATCH", "2")
    sizes = [(90, 60), (40, 40), (200, 100), (50, 40), (60, 60)]
    out = wemm_embed.embed_images([Image.new("RGB", s) for s in sizes])
    assert [tuple(r) for r in out.astype(int)] == sizes
    areas = [w * h for b in fake.batches for w, h in b]
    assert areas == sorted(areas)


def test_prepare_image_caps_pixels():
    from PIL import Image
    im = wemm_embed.prepare_image(Image.new("RGBA", (4000, 3000)))
    assert im.mode == "RGB" and im.size[0] * im.size[1] <= wemm_embed.MAX_PIXELS


def test_vram_check_counts_wemm_even_when_qwen_is_already_loaded(monkeypatch):
    """Qwen уже загружен, WeMM ещё нет — проверка обязана спросить память под
    WeMM (раньше embed=False выключал и её)."""
    import qwen_vl_embed
    import vision_model
    monkeypatch.setenv("CASCADE_MODEL", "wemm9b")
    monkeypatch.setattr(wemm_embed, "devices", lambda: ["cuda:0"])
    monkeypatch.setitem(wemm_embed._STATE, "models", {})
    monkeypatch.setitem(qwen_vl_embed._STATE, "model", object())
    seen = {}
    real = vision_model.vram_need_gib

    def spy(embed=True, rerank=True, wemm=None):
        out = real(embed, rerank, wemm)
        seen["need"] = out
        return None
    monkeypatch.setattr(vision_model, "vram_need_gib", spy)
    vision_model.vram_shortage(embed=True, rerank=False)
    assert seen["need"]["cuda:0"] >= vision_model.WEIGHTS_GIB[wemm_embed.MODEL_NAME]


def test_render_workers_reserve_ignores_wemm_when_not_selected(monkeypatch):
    """Вес WeMM в таблице не должен урезать процессы рендера на карте, если
    каскад считает другой моделью."""
    import torch
    ps = _ps()

    class P:
        total_memory = 48 * 2 ** 30
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: P)
    monkeypatch.delenv("GPU_RENDER_WORKERS", raising=False)
    monkeypatch.setenv("CASCADE_MODEL", "")
    without = ps.gpu_render_workers()
    monkeypatch.setenv("CASCADE_MODEL", "wemm9b")
    with_wemm = ps.gpu_render_workers()
    assert without > with_wemm >= 1
