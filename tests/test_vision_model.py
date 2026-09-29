#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Модели зрения GPU-ветки (Qwen3-VL): какие модели прогон требует, отказ
рендера без видеокарты или калибровки, пороги только из калибровки."""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import vision_model  # noqa: E402


@pytest.mark.parametrize("clip,veto,top,want", [
    (True, False, "24", (True, True)),     # гейты + реранкер верха каскада
    (True, False, "0", (True, False)),     # без реранкера верха
    (False, False, "24", (False, False)),  # каскад без эмбеддингов не работает
    (False, True, "24", (False, True)),    # только вторая проверка
])
def test_models_needed_follow_the_layers_that_call_them(monkeypatch, clip, veto, top, want):
    monkeypatch.setattr(ps, "CLIP_ENABLED", clip)
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "1" if veto else "0")
    monkeypatch.setenv("CASCADE_RERANK_TOP", top)
    assert ps.vision_models_needed() == want


def test_warmup_follows_the_same_rule(monkeypatch):
    monkeypatch.setattr(ps, "CLIP_ENABLED", False)
    monkeypatch.setattr(ps, "AESTHETIC_ENABLED", False)
    monkeypatch.setattr(ps, "PARALLAX_ENABLED", False)
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "1")
    assert [n for n, _ in ps.model_warmup_jobs()] == ["Qwen3-VL-Reranker"]


def test_nothing_needed_means_nothing_required(monkeypatch):
    import ml_device
    monkeypatch.setattr(ml_device, "device", lambda: pytest.fail("устройство не спрашивается"))
    assert vision_model.readiness(False, False) == []
    vision_model.require_ready(False, False)


def test_render_refuses_without_cuda(monkeypatch):
    import ml_device
    monkeypatch.setattr(ml_device, "device", lambda: "cpu")
    with pytest.raises(SystemExit) as e:
        vision_model.require_ready(True, True)
    assert "CUDA" in str(e.value) and vision_model.CALIBRATE_COMMAND in str(e.value)


def _ready_models(monkeypatch):
    import ml_device
    import qwen_vl_embed
    import qwen_vl_rerank
    monkeypatch.setattr(ml_device, "device", lambda: "cuda")
    monkeypatch.setattr(qwen_vl_embed, "available", lambda: True)
    monkeypatch.setattr(qwen_vl_rerank, "available", lambda: True)


def test_render_refuses_without_calibration(monkeypatch, tmp_path):
    _ready_models(monkeypatch)
    monkeypatch.setenv("VISION_CALIBRATION", str(tmp_path / "none.json"))
    problems = vision_model.readiness()
    assert len(problems) == 1 and "калибровки" in problems[0]
    with pytest.raises(vision_model.NotCalibrated):
        vision_model.threshold("relevance")


def _write_cal(path, signature, thresholds):
    path.write_text(json.dumps({"signature": signature, "thresholds": thresholds,
                                "scale": {"gate": {"mean": 0.3, "std": 0.1},
                                          "sentence": {"mean": 0.2, "std": 0.05}}}),
                    encoding="utf-8")


FULL = {"relevance": 0.2, "risky_margin": 0.03, "negative_veto_margin": -0.05,
        "smart_rerank": 0.4}


def test_calibration_of_other_models_is_refused(monkeypatch, tmp_path):
    _ready_models(monkeypatch)
    p = tmp_path / "cal.json"
    _write_cal(p, dict(vision_model.current_signature(), embed="other-model"), FULL)
    monkeypatch.setenv("VISION_CALIBRATION", str(p))
    problems = vision_model.readiness()
    assert problems and "других моделей" in problems[0]


def test_calibration_missing_a_threshold_is_refused(monkeypatch, tmp_path):
    _ready_models(monkeypatch)
    p = tmp_path / "cal.json"
    _write_cal(p, vision_model.current_signature(),
               {k: v for k, v in FULL.items() if k != "smart_rerank"})
    monkeypatch.setenv("VISION_CALIBRATION", str(p))
    assert "smart_rerank" in vision_model.readiness()[0]


def test_valid_calibration_gives_thresholds_and_scale_transfer(monkeypatch, tmp_path):
    _ready_models(monkeypatch)
    p = tmp_path / "cal.json"
    _write_cal(p, vision_model.current_signature(), FULL)
    monkeypatch.setenv("VISION_CALIBRATION", str(p))
    assert vision_model.readiness() == []
    assert vision_model.threshold("relevance") == 0.2
    assert vision_model.threshold("particle") is None      # слой без разметки выключен
    # Перенос разницы: отношение разбросов (0.1 / 0.05258).
    assert vision_model.legacy_margin(0.02, "gate") == pytest.approx(0.02 * 0.1 / 0.05258)
    # Обратный перенос уровня возвращает шкалу SigLIP2.
    assert vision_model.to_legacy_level(0.2, "sentence") == pytest.approx(
        vision_model.LEGACY_SENTENCE_SCALE["mean"])
    # reload переписывает глобальные пороги модуля — все под monkeypatch,
    # иначе тест оставил бы свои числа соседним тестам.
    for name in ("CLIP_RELEVANCE_THRESHOLD", "RISKY_QUERY_MARGIN", "NEGATIVE_VETO_MARGIN",
                 "SMART_RELEVANCE_THRESHOLD", "PARTICLE_SCORE_THRESHOLD", "VISUAL_DOMAIN_GUARDS",
                 "_CANDIDATE_GATE_SIG"):
        monkeypatch.setattr(ps, name, getattr(ps, name))
    ps.reload_vision_thresholds()
    assert ps.CLIP_RELEVANCE_THRESHOLD == 0.2 and ps.SMART_RELEVANCE_THRESHOLD == 0.4


def test_uncalibrated_gate_refuses_instead_of_passing(monkeypatch):
    """Модель ответила, а порога нет: громкий отказ, не пропуск кадра."""
    monkeypatch.setattr(ps, "CLIP_RELEVANCE_THRESHOLD", None)
    monkeypatch.setattr(ps, "clip_relevance", lambda *a, **k: 0.5)
    with pytest.raises(vision_model.NotCalibrated):
        ps.is_relevant_candidate("x.jpg", "some query")
