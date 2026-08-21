"""Юнит-тесты Reference-Guided Look Management (scripts/look_reference.py).
Без реального torch/CLIP — CLIP-вызов вынесен в _domain_scores() именно для
того, чтобы ранжирование/margin-gate/коррекция тестировались изолированно
(тот же принцип разделения, что test_visual_qc.py уже применяет к своим
scorer-функциям). Самый важный тест файла —
test_look_correction_filter_noop_on_real_empty_lookbook: он единственный
прямо доказывает, что система не влияет на сегодняшний рендер, пока в
assets/lookbook/lookbook.json реально нет ни одного эталона."""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["look_reference.py", tempfile.gettempdir()]
import look_reference as lr   # noqa: E402


# ---------- sRGB <-> Lab round-trip ----------

@pytest.mark.parametrize("rgb", [
    (0.5, 0.5, 0.5), (0.8, 0.2, 0.15), (0.1, 0.4, 0.9), (0.95, 0.95, 0.9), (0.05, 0.05, 0.08),
])
def test_lab_round_trip(rgb):
    lab = lr._srgb_to_lab(rgb)
    back = lr._lab_to_srgb(lab)
    for a, b in zip(rgb, back):
        assert abs(a - b) < 1e-3


def test_neutral_gray_has_near_zero_chroma():
    L, a, b = lr._srgb_to_lab((0.5, 0.5, 0.5))
    assert abs(a) < 0.5
    assert abs(b) < 0.5
    assert 40 < L < 70


# ---------- classify_domain (через monkeypatch _domain_scores) ----------

def test_classify_domain_none_when_clip_unavailable(monkeypatch):
    monkeypatch.setattr(lr, "_domain_scores", lambda path: None)
    domain, margin = lr.classify_domain("x.jpg")
    assert domain is None
    assert margin == 0.0


def test_classify_domain_picks_top_score_with_enough_margin(monkeypatch):
    monkeypatch.setattr(lr, "_domain_scores", lambda path: {
        "snow": 0.30, "night": 0.10, "museum_daylight": 0.05, "portrait": 0.02,
        "urban": 0.01, "archive_bw": 0.0, "ai_illustration": -0.01, "battle": -0.02,
    })
    domain, margin = lr.classify_domain("x.jpg")
    assert domain == "snow"
    assert margin == pytest.approx(0.20)


def test_classify_domain_none_when_margin_too_small(monkeypatch):
    monkeypatch.setattr(lr, "_domain_scores", lambda path: {
        "snow": 0.201, "night": 0.20, "museum_daylight": 0.05,
    })
    domain, margin = lr.classify_domain("x.jpg")
    assert domain is None
    assert margin == 0.0


# ---------- find_reference ----------

def _ref(id_, domain, lab_mean, brightness=0.5, contrast=0.4, temperature=0.0, max_delta=None):
    r = {"id": id_, "domain": domain, "lab_mean": list(lab_mean),
         "brightness": brightness, "contrast": contrast, "temperature": temperature}
    if max_delta is not None:
        r["max_correction_delta"] = list(max_delta)
    return r


def test_find_reference_none_when_domain_absent():
    lookbook = {"references": [_ref("a", "snow", (50, 0, 0))]}
    ref, conf = lr.find_reference("night", (50, 0, 0), 0.5, 0.4, 0.0, lookbook)
    assert ref is None
    assert conf == 0.0


def test_find_reference_picks_closest_of_same_domain():
    far = _ref("far", "snow", (80, 20, 20))
    near = _ref("near", "snow", (52, 1, -1))
    lookbook = {"references": [far, near]}
    ref, conf = lr.find_reference("snow", (50, 0, 0), 0.5, 0.4, 0.0, lookbook)
    assert ref["id"] == "near"
    assert conf > 0.0


def test_find_reference_none_when_too_far():
    lookbook = {"references": [_ref("a", "snow", (95, 40, 40))]}
    ref, conf = lr.find_reference("snow", (10, -40, -40), 0.9, 0.9, -1.0, lookbook)
    assert ref is None
    assert conf == 0.0


def test_find_reference_empty_lookbook():
    ref, conf = lr.find_reference("snow", (50, 0, 0), 0.5, 0.4, 0.0, {"references": []})
    assert ref is None
    assert conf == 0.0


# ---------- compute_correction ----------

def test_compute_correction_normal_case_returns_gains_near_one():
    ref = _ref("a", "snow", (52, 1, -1), max_delta=(10, 8, 8))
    gains, qc, delta = lr.compute_correction((0.5, 0.5, 0.5), (50, 0, 0), ref, confidence=0.8, prev_delta=None)
    assert gains is not None
    assert qc["decision"] == "ok"
    for g in gains:
        assert lr.GAIN_CLAMP[0] <= g <= lr.GAIN_CLAMP[1]


def test_compute_correction_rejects_extreme_target_as_clipping():
    # Тёмный исходник (L=5, нейтральный) + эталон, требующий большой сдвиг
    # по b (жёлто-синяя ось) — у тёмных тонов гамма sRGB узкая по хроме,
    # даже после клэмпа силы (MAX_STRENGTH=0.35) целевая точка уходит в
    # отрицательный линейный синий (проверено напрямую через
    # _lab_to_linear_rgb при разработке теста, не угадано).
    ref = _ref("a", "night", (5, 0, 115), max_delta=(200, 200, 200))
    gains, qc, delta = lr.compute_correction((0.03, 0.03, 0.03), (5, 0, 0), ref, confidence=1.0, prev_delta=None)
    assert gains is None
    assert qc["decision"] == "reject_clipping"


def test_compute_correction_rejects_oversaturation():
    ref = _ref("a", "portrait", (50, 60, 60), max_delta=(5, 60, 60))
    gains, qc, delta = lr.compute_correction((0.5, 0.5, 0.5), (50, 1, 1), ref, confidence=1.0, prev_delta=None)
    assert gains is None
    assert qc["decision"] == "reject_oversaturation"


def test_compute_correction_rejected_case_keeps_prev_delta_unchanged():
    # prev близко к цели, чтобы клэмп шага EMA (DELTA_STEP_CLAMP) не увёл
    # сглаженную дельту прочь от того же клиппинг-сценария, что и в тесте
    # без prev выше — это тест на "отклонённое НЕ фиксируется в состоянии",
    # не повторная проверка самого условия клиппинга.
    ref = _ref("a", "night", (5, 0, 115), max_delta=(200, 200, 200))
    prev = (0.0, 0.0, 37.0)
    gains, qc, delta = lr.compute_correction((0.03, 0.03, 0.03), (5, 0, 0), ref, confidence=1.0, prev_delta=prev)
    assert gains is None
    assert qc["decision"] == "reject_clipping"
    assert delta == prev


def test_compute_correction_ema_step_is_clamped():
    ref = _ref("a", "snow", (70, 20, -20), max_delta=(30, 30, 30))
    prev = (0.0, 0.0, 0.0)
    _, _, delta1 = lr.compute_correction((0.5, 0.5, 0.5), (50, 0, 0), ref, confidence=1.0, prev_delta=prev)
    # Шаг не может быть больше DELTA_STEP_CLAMP за один вызов.
    for d, cap in zip(delta1, lr.DELTA_STEP_CLAMP):
        assert abs(d - 0.0) <= cap + 1e-6


def test_compute_correction_never_upgrades_gain_beyond_clamp_even_unclamped_target():
    ref = _ref("a", "snow", (60, 5, -5), max_delta=(15, 10, 10))
    gains, qc, delta = lr.compute_correction((0.5, 0.5, 0.5), (50, 0, 0), ref, confidence=1.0, prev_delta=None)
    if gains is not None:
        for g in gains:
            assert lr.GAIN_CLAMP[0] <= g <= lr.GAIN_CLAMP[1]


# ---------- look_correction_filter (оркестрация) ----------

def test_look_correction_filter_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_ENABLED", False)
    filt, report, delta = lr.look_correction_filter("x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False)
    assert filt is None
    assert report["decision"] == "skipped_disabled"


def test_look_correction_filter_noop_on_real_empty_lookbook(monkeypatch):
    """Главный тест модуля: с РЕАЛЬНЫМ assets/lookbook/lookbook.json (пустым
    сегодня — канал не выпустил ни одного эпизода) система обязана быть
    полностью инертной даже при включённом флаге — это единственная
    гарантия, что коммит этой системы не меняет сегодняшний рендер."""
    assert os.path.exists(lr.LOOKBOOK_PATH), "assets/lookbook/lookbook.json должен существовать в репозитории"
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_ENABLED", True)
    filt, report, delta = lr.look_correction_filter("x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False)
    assert filt is None
    assert report["decision"] == "skipped_empty_lookbook"


def test_look_correction_filter_noop_when_face_detected(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [_ref("a", "portrait", (50, 0, 0))]})
    filt, report, delta = lr.look_correction_filter("x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=True)
    assert filt is None
    assert report["decision"] == "skipped_face_detected"


def test_look_correction_filter_noop_without_signal(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [_ref("a", "portrait", (50, 0, 0))]})
    filt, report, delta = lr.look_correction_filter("x.jpg", (None, None), None, has_face=False)
    assert filt is None
    assert report["decision"] == "skipped_no_signal"


def test_look_correction_filter_noop_when_domain_unmatched(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [_ref("a", "snow", (50, 0, 0))]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: (None, 0.0))
    filt, report, delta = lr.look_correction_filter("x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False)
    assert filt is None
    assert report["decision"] == "skipped_low_domain_confidence"


def test_look_correction_filter_noop_when_no_reference_of_matched_domain(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [_ref("a", "night", (50, 0, 0))]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("snow", 0.1))
    filt, report, delta = lr.look_correction_filter("x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False)
    assert filt is None
    assert report["decision"] == "skipped_no_reference_match"


def test_look_correction_filter_applies_when_everything_matches(monkeypatch):
    ref = _ref("snow_01", "snow", (55, 1, -2), brightness=0.5, contrast=0.4, temperature=0.0,
               max_delta=(10, 8, 8))
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [ref]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("snow", 0.1))
    filt, report, delta = lr.look_correction_filter("x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False)
    assert filt is not None
    assert filt.startswith("colorchannelmixer=rr=")
    assert report["decision"] == "applied"
    assert report["reference_id"] == "snow_01"
    assert delta is not None
