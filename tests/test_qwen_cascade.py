#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-VL-Embedding как модель каскада: расчёт как в официальном коде,
переключение без следа на SigLIP2 по умолчанию."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import qwen_vl_embed as q  # noqa: E402


def reference_smart_resize(height, width, factor, min_pixels, max_pixels):
    """qwen_vl_utils.vision_process.smart_resize 0.0.14, дословно."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@pytest.mark.parametrize("h,w", [(300, 400), (1080, 1920), (40, 40), (4000, 3000), (97, 1203)])
def test_resize_matches_qwen_vl_utils(h, w):
    mp = q.max_pixels()
    assert q.smart_resize(h, w) == reference_smart_resize(h, w, 32, q.MIN_PIXELS, mp)
    rh, rw = q.smart_resize(h, w)
    assert rh % 32 == 0 and rw % 32 == 0 and rh * rw <= max(mp, 32 * 32 * 4)


def test_last_token_pooling_follows_the_mask():
    torch = pytest.importorskip("torch")
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    out = q.pool_last(hidden, mask)
    assert torch.equal(out[0], hidden[0, 2]) and torch.equal(out[1], hidden[1, 3])


def test_instruction_gets_final_punctuation_like_the_official_code():
    conv = q.conversation(text="x", instruction="Retrieve images")
    assert conv[0]["content"][0]["text"] == "Retrieve images."
    assert q.conversation(image="img")[0]["content"][0]["text"] == q.IMAGE_INSTRUCTION


def _ps():
    import pipeline_smart as ps
    return ps


def test_default_cascade_is_siglip2_and_signature_is_unchanged(monkeypatch):
    ps = _ps()
    monkeypatch.delenv("CASCADE_MODEL", raising=False)
    assert ps.cascade_model() == "siglip2"
    # Ключ кэша превью — прежняя формула (модель гейта), без изменений.
    import hashlib
    import ml_device
    url = "https://x/y.jpg"
    assert ps._cascade_key(url) == hashlib.md5(
        f"{ps.CLIP_GATE_MODEL_NAME}{ml_device.tag()}|{url}".encode()).hexdigest()


def test_qwen_cascade_routes_embeddings_and_keeps_gate_cache_clean(monkeypatch, tmp_path):
    ps = _ps()
    monkeypatch.setenv("CASCADE_MODEL", "qwen3vl")
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path / "cc"))
    calls = {"img": 0, "txt": 0}

    def fake_images(imgs):
        calls["img"] += len(imgs)
        return np.stack([np.eye(4, dtype="float32")[k % 4] for k in range(len(imgs))])

    def fake_text(t, instruction=q.QUERY_INSTRUCTION):
        calls["txt"] += 1
        return np.eye(4, dtype="float32")[1]
    monkeypatch.setattr(q, "available", lambda: True)
    monkeypatch.setattr(q, "embed_images", fake_images)
    monkeypatch.setattr(q, "embed_text", fake_text)
    monkeypatch.setattr(ps, "_gate_embed", lambda **k: pytest.fail("каскад позвал модель гейта"))
    ps._CASCADE_TEXT_CACHE.clear()
    assert ps.cascade_model() == "qwen3vl"

    from PIL import Image

    def probe(p, dest):
        Image.new("RGB", (64, 48), (p["n"] * 40, 10, 10)).save(dest)
    cands = [{"id": f"c{k}", "n": k, "src": {"large": f"https://h/{k}.jpg"}} for k in range(4)]
    before = dict(ps._GATE_IMG_EMB_CACHE)
    out = ps.cascade_reorder(cands, ["a query"], str(tmp_path / "cf"), probe, index=0,
                             url_of=lambda p: p["src"]["large"])
    assert calls["img"] == 4 and calls["txt"] == 1
    assert out[0]["id"] == "c1", "ближе всего к запросу — кадр с вектором запроса"
    assert ps._GATE_IMG_EMB_CACHE == before, "векторы Qwen не должны попасть в кэш гейта SigLIP2"


def test_qwen_unavailable_falls_back_to_siglip2(monkeypatch):
    ps = _ps()
    monkeypatch.setenv("CASCADE_MODEL", "qwen3vl")
    monkeypatch.setattr(q, "available", lambda: False)
    assert ps.cascade_model() == "siglip2"
