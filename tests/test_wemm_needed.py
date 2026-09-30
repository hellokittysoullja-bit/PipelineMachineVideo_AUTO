"""WeMM-9B грузится и учитывается в видеопамяти только когда каскад будет вызван.

Прогон 29.09 без судьи держал её 17.6 ГиБ впустую (см. cascade_model_needed).
Выбор кадров это не меняет: cascade_reorder зовётся исключительно под
shot_judge_active() — инвариант держит статический тест ниже."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

PIPELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "pipeline_smart.py")


def _ps():
    import pipeline_smart
    return pipeline_smart


@pytest.mark.parametrize("cascade,judge,key,want", [
    ("wemm9b", "1", "k", True),
    ("wemm9b", "0", "k", False),
    ("wemm9b", "1", "", False),
    ("", "1", "k", False),
    ("qwen", "1", "k", False),
])
def test_cascade_model_needed_truth_table(monkeypatch, cascade, judge, key, want):
    monkeypatch.setenv("CASCADE_MODEL", cascade)
    monkeypatch.setenv("SHOT_JUDGE", judge)
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", key)
    assert _ps().cascade_model_needed() is want


def _warm(monkeypatch, judge):
    ps = _ps()
    monkeypatch.setenv("CASCADE_MODEL", "wemm9b")
    monkeypatch.setenv("SHOT_JUDGE", judge)
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "0")
    monkeypatch.setattr(ps, "CLIP_ENABLED", True)
    monkeypatch.setattr(ps, "AESTHETIC_ENABLED", False)
    monkeypatch.setattr(ps, "PARALLAX_ENABLED", False)
    return [n for n, _ in ps.model_warmup_jobs()]


def test_warmup_skips_wemm_without_the_judge(monkeypatch):
    assert "WeMM-Embedding-9B" not in _warm(monkeypatch, "0")


def test_warmup_loads_wemm_with_the_judge(monkeypatch):
    assert "WeMM-Embedding-9B" in _warm(monkeypatch, "1")


def test_vram_need_excludes_wemm_when_not_needed(monkeypatch):
    import ml_device
    import vision_model
    import wemm_embed
    monkeypatch.setenv("CASCADE_MODEL", "wemm9b")
    monkeypatch.setattr(ml_device, "device_for", lambda role=None: "cuda:0")
    monkeypatch.setattr(wemm_embed, "devices", lambda: ["cuda:0"])
    full = vision_model.vram_need_gib(True, True)["cuda:0"]
    lean = vision_model.vram_need_gib(True, True, wemm=False)["cuda:0"]
    assert full - lean == pytest.approx(
        vision_model.WEIGHTS_GIB[wemm_embed.MODEL_NAME] + vision_model.WEMM_HEADROOM_GIB)


def test_readiness_does_not_load_wemm_when_not_needed(monkeypatch):
    import ml_device
    import qwen_vl_embed
    import qwen_vl_rerank
    import vision_model
    import wemm_embed
    monkeypatch.setenv("CASCADE_MODEL", "wemm9b")
    monkeypatch.setattr(ml_device, "device", lambda: "cuda")
    monkeypatch.setattr(vision_model, "vram_shortage", lambda *a, **k: None)
    monkeypatch.setattr(qwen_vl_embed, "available", lambda: True)
    monkeypatch.setattr(qwen_vl_rerank, "available", lambda: True)
    monkeypatch.setattr(vision_model, "calibration", lambda: {"ok": 1})

    def boom():
        raise AssertionError("WeMM не должна грузиться")
    monkeypatch.setattr(wemm_embed, "available", boom)
    assert vision_model.readiness(True, True, wemm=False) == []


def test_main_passes_the_wemm_need_to_require_ready():
    src = open(PIPELINE, encoding="utf-8").read()
    assert "vision_model.require_ready(need_embed, need_rerank, wemm=cascade_model_needed())" in src


def test_every_cascade_call_is_under_the_judge():
    """Инвариант, на котором держится cascade_model_needed(): каскад не зовётся
    без судьи. Новый вызов вне shot_judge_active()/флага cascade сломает его."""
    src = open(PIPELINE, encoding="utf-8").read()
    sites = [m.start() for m in re.finditer(r"(?<![\w.])cascade_reorder\(", src)
             if not src[max(0, m.start() - 4):m.start()].endswith("def ")]
    assert len(sites) >= 3
    for pos in sites:
        window = src[max(0, pos - 1800):pos]
        assert "shot_judge_active(" in window or "if not cascade:" in window, src[pos - 120:pos + 60]
