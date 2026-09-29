#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-VL-Embedding — модель каскада и гейтов (GPU-ветка, SigLIP2 удалена
29.09): расчёт как в официальном коде; Qwen3-VL-Reranker доранжирует верх
каскада."""
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


def test_cascade_key_names_the_qwen_embedding(monkeypatch):
    ps = _ps()
    import hashlib
    import ml_device
    url = "https://x/y.jpg"
    assert ps._cascade_key(url) == hashlib.md5(
        f"{q.signature()}{ml_device.tag()}|{url}".encode()).hexdigest()


def _fake_qwen(monkeypatch, ps, calls):
    def fake_images(imgs):
        calls["img"] += len(imgs)
        return np.stack([np.eye(4, dtype="float32")[k % 4] for k in range(len(imgs))])

    def fake_text(t, instruction=q.QUERY_INSTRUCTION):
        calls["txt"] += 1
        return np.eye(4, dtype="float32")[1]
    monkeypatch.setattr(q, "embed_images", fake_images)
    monkeypatch.setattr(q, "embed_text", fake_text)
    monkeypatch.setattr(ps, "CLIP_BROKEN", False)
    monkeypatch.setattr(ps, "CLIP_ENABLED", True)
    ps._CLIP_TEXT_EMB_CACHE.clear()


def _cands(k=4):
    return [{"id": f"c{j}", "n": j, "src": {"large": f"https://h/{j}.jpg"}} for j in range(k)]


def _probe(p, dest):
    from PIL import Image
    Image.new("RGB", (64, 48), (p["n"] * 40, 10, 10)).save(dest)


def test_cascade_ranks_with_qwen_and_shares_vectors_with_the_gate(monkeypatch, tmp_path):
    """Каскад и гейты — одна модель: вектор картинки, посчитанный каскадом,
    гейт берёт готовым (по содержимому файла), а не считает заново."""
    ps = _ps()
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("CASCADE_RERANK_TOP", "0")
    calls = {"img": 0, "txt": 0}
    _fake_qwen(monkeypatch, ps, calls)
    ps._GATE_IMG_EMB_CACHE.clear()
    out = ps.cascade_reorder(_cands(), ["a query"], str(tmp_path / "cf"), _probe, index=0,
                             url_of=lambda p: p["src"]["large"])
    assert calls["img"] == 4 and calls["txt"] == 1
    assert out[0]["id"] == "c1", "ближе всего к запросу — кадр с вектором запроса"
    assert len(ps._GATE_IMG_EMB_CACHE) == 4


def test_cascade_without_the_model_keeps_the_order(monkeypatch, tmp_path):
    """Модели нет — порядок прежний; отката на другую модель нет (рендер без
    Qwen до этого места не доходит: vision_model.require_ready)."""
    ps = _ps()
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path / "cc"))
    monkeypatch.setattr(q, "embed_images", lambda imgs: None)
    monkeypatch.setattr(q, "embed_text", lambda t, instruction=None: None)
    ps._CLIP_TEXT_EMB_CACHE.clear()
    cands = _cands()
    assert ps.cascade_reorder(cands, ["a query"], str(tmp_path / "cf"), _probe, index=0,
                              url_of=lambda p: p["src"]["large"]) == cands


def test_reranker_reorders_only_the_top(monkeypatch, tmp_path):
    ps = _ps()
    import qwen_vl_rerank
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("CASCADE_RERANK_TOP", "3")
    _fake_qwen(monkeypatch, ps, {"img": 0, "txt": 0})
    seen = {}

    def fake_score(text, paths, instruction=None):
        seen["text"], seen["n"] = text, len(paths)
        return [0.1, 0.9, 0.5]
    monkeypatch.setattr(qwen_vl_rerank, "score", fake_score)
    base = ps.cascade_reorder(_cands(5), ["a query"], str(tmp_path / "cf0"), _probe, index=0,
                              url_of=lambda p: p["src"]["large"])
    out = ps.cascade_reorder(_cands(5), ["a query"], str(tmp_path / "cf"), _probe, index=0,
                             url_of=lambda p: p["src"]["large"], rerank_text="a hand holds a dagger")
    b = [p["id"] for p in base]
    assert seen == {"text": "a hand holds a dagger", "n": 3}
    assert [p["id"] for p in out] == [b[1], b[2], b[0]] + b[3:]
    assert not [f for f in os.listdir(tmp_path) if ".rr_" in f or ".casc_" in f], \
        "превью реранкера не убраны"


def test_reranker_failure_keeps_the_cascade_order(monkeypatch, tmp_path):
    ps = _ps()
    import qwen_vl_rerank
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("CASCADE_RERANK_TOP", "3")
    _fake_qwen(monkeypatch, ps, {"img": 0, "txt": 0})
    monkeypatch.setattr(qwen_vl_rerank, "score", lambda *a, **k: None)
    base = ps.cascade_reorder(_cands(5), ["a query"], str(tmp_path / "cf0"), _probe, index=0,
                              url_of=lambda p: p["src"]["large"])
    out = ps.cascade_reorder(_cands(5), ["a query"], str(tmp_path / "cf"), _probe, index=0,
                             url_of=lambda p: p["src"]["large"], rerank_text="x")
    assert [p["id"] for p in out] == [p["id"] for p in base]


def test_rerank_signature_enters_the_judge_signature(monkeypatch):
    """Реранкер меняет, кого увидит судья: смена модели или верха — другая
    подпись, иначе прогретый кэш отдал бы выбор старого порядка."""
    ps = _ps()
    import qwen_vl_rerank
    monkeypatch.setattr(ps, "shot_judge_active", lambda index=None: True)
    monkeypatch.setattr(ps, "shot_judge_model", lambda: "m")
    a = ps.shot_judge_signature(0)
    monkeypatch.setattr(qwen_vl_rerank, "signature", lambda: "other")
    assert ps.shot_judge_signature(0) != a
    b = ps.shot_judge_signature(0)
    monkeypatch.setenv("CASCADE_RERANK_TOP", "7")
    assert ps.shot_judge_signature(0) != b


def test_reranker_conversation_matches_the_official_format():
    import qwen_vl_rerank as r
    conv = r.conversation("a dagger", "IMG")
    assert conv[0]["content"][0]["text"] == r.SYSTEM_PROMPT
    parts = conv[1]["content"]
    assert [x.get("text") for x in parts[:4]] == [
        "<Instruct>: " + r.DEFAULT_INSTRUCTION, "<Query>:", "a dagger", "\n<Document>:"]
    assert parts[4] == {"type": "image", "image": "IMG"}


def test_qwen_refuses_to_load_without_cuda(monkeypatch):
    """Аудит 28.09: без видеокарты 8B-модель грузилась на процессор в fp32
    (~32 ГБ памяти); режим обещает откат на SigLIP2."""
    import ml_device
    import qwen_vl_embed as q
    monkeypatch.setattr(ml_device, "device", lambda: "cpu")
    monkeypatch.setitem(q._STATE, "model", None)
    monkeypatch.setitem(q._STATE, "broken", None)
    assert q.available() is False
    assert "CUDA" in q._STATE["broken"]


def test_qwen_runtime_failure_turns_it_off_instead_of_raising(monkeypatch):
    import qwen_vl_embed as q
    monkeypatch.setitem(q._STATE, "model", object())
    monkeypatch.setitem(q._STATE, "broken", None)
    monkeypatch.setattr(q, "_load", lambda: True)

    def boom(*a, **k):
        raise ValueError("height:1 must be larger than factor:32")
    monkeypatch.setattr(q, "_encode", boom)
    monkeypatch.setattr(q, "prepare_image", lambda im: im)
    assert q.embed_images(["x"]) is None
    assert q._STATE["model"] is None and "factor" in q._STATE["broken"]


def test_encode_runs_outside_the_module_lock(monkeypatch):
    """Аудит 29.09: под замком модуля шли подготовка картинок и токенизация,
    и потоки ждали друг друга при свободной видеокарте. Замок — только на
    загрузку и отключение; прогон — через общий замок видеокарты."""
    import qwen_vl_embed as q
    monkeypatch.setattr(q, "_load", lambda: True)
    monkeypatch.setattr(q, "prepare_image", lambda im: im)
    held = []

    def enc(conv, images):
        held.append(q._LOCK.locked())
        import numpy as np
        return np.zeros((len(conv), 4), dtype="float32")
    monkeypatch.setattr(q, "_encode", enc)
    q.embed_images(["a", "b"])
    q.embed_text("x")
    assert held and not any(held)


def test_second_failure_keeps_the_first_reason(monkeypatch):
    import qwen_vl_embed as q
    monkeypatch.setattr(q, "_STATE", {"model": object(), "processor": None, "device": "cuda", "broken": None})
    q._fail(RuntimeError("первая"))
    q._fail(RuntimeError("вторая"))
    assert "первая" in q._STATE["broken"] and q._STATE["model"] is None
