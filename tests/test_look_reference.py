"""Юнит-тесты Reference-Guided Look Management (scripts/look_reference.py).
Без реального torch/CLIP — CLIP-вызов вынесен в _domain_scores() именно для
того, чтобы ранжирование/margin-gate/коррекция тестировались изолированно
(тот же принцип разделения, что test_visual_qc.py уже применяет к своим
scorer-функциям). test_look_correction_filter_noop_on_empty_lookbook
доказывает инертность системы на СИНТЕТИЧЕСКИ пустом lookbook (не на
реальном assets/lookbook/lookbook.json — тот сегодня уже не пуст, 13
эталонов канала, добавлены вручную вне тестов, см. коммит)."""
import os
import shutil
import sys
import tempfile

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["look_reference.py", tempfile.gettempdir()]
import look_reference as lr   # noqa: E402

_no_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не найден в PATH")


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


# ---------- text_domain_hint (через monkeypatch _domain_scores_from_text) ----------
# Та же margin-gate логика, что classify_domain() выше, но текст-vs-текст —
# добавлена для Semantic Visual Director (scripts/visual_director.py), нужен
# домен ОЖИДАНИЯ по тексту фразы, для сравнения с доменом КАНДИДАТА.

def test_text_domain_hint_none_when_clip_unavailable(monkeypatch):
    monkeypatch.setattr(lr, "_domain_scores_from_text", lambda text: None)
    domain, margin = lr.text_domain_hint("какой-то текст блока")
    assert domain is None
    assert margin == 0.0


def test_text_domain_hint_picks_top_score_with_enough_margin(monkeypatch):
    monkeypatch.setattr(lr, "_domain_scores_from_text", lambda text: {
        "snow": 0.30, "night": 0.10, "museum_daylight": 0.05, "portrait": 0.02,
        "urban": 0.01, "archive_bw": 0.0, "ai_illustration": -0.01, "battle": -0.02,
    })
    domain, margin = lr.text_domain_hint("метель замела деревню")
    assert domain == "snow"
    assert margin == pytest.approx(0.20)


def test_text_domain_hint_none_when_margin_too_small(monkeypatch):
    monkeypatch.setattr(lr, "_domain_scores_from_text", lambda text: {
        "snow": 0.201, "night": 0.20, "museum_daylight": 0.05,
    })
    domain, margin = lr.text_domain_hint("неоднозначный текст")
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


def test_compute_correction_rejected_case_with_prev_none_stays_none():
    """Regression на баг, пойманный при разборе плана: hard-reject ПОСЛЕ
    EMA reset (prev_delta=None на входе) обязан вернуть delta=None, а не
    какое-либо старое значение — compute_correction() возвращает ровно то,
    что ей передали, без пересчёта."""
    ref = _ref("a", "night", (5, 0, 115), max_delta=(200, 200, 200))
    gains, qc, delta = lr.compute_correction((0.03, 0.03, 0.03), (5, 0, 0), ref, confidence=1.0, prev_delta=None)
    assert gains is None
    assert delta is None


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


# ---------- cache_signature() ----------

def test_cache_signature_off_for_off_and_shadow(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "off")
    assert lr.cache_signature() == "look:off"
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "shadow")
    assert lr.cache_signature() == "look:off"


def test_cache_signature_changes_with_lookbook_content(monkeypatch, tmp_path):
    lookbook_path = tmp_path / "lookbook.json"
    lookbook_path.write_text('{"references": []}', encoding="utf-8")
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(lookbook_path))
    sig1 = lr.cache_signature()
    # Тот же mtime/size класс не имеет значения — реальный digest содержимого.
    lookbook_path.write_text('{"references": [{"id": "x"}]}', encoding="utf-8")
    sig2 = lr.cache_signature()
    assert sig1 != sig2


def test_cache_signature_changes_with_threshold_constant(monkeypatch, tmp_path):
    lookbook_path = tmp_path / "lookbook.json"
    lookbook_path.write_text('{"references": []}', encoding="utf-8")
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(lookbook_path))
    sig1 = lr.cache_signature()
    monkeypatch.setattr(lr, "MAX_STRENGTH", lr.MAX_STRENGTH + 0.01)
    sig2 = lr.cache_signature()
    assert sig1 != sig2


def test_cache_signature_changes_with_channel_id(monkeypatch, tmp_path):
    lookbook_path = tmp_path / "lookbook.json"
    lookbook_path.write_text('{"references": []}', encoding="utf-8")
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(lookbook_path))
    monkeypatch.setattr(lr, "CHANNEL_ID", "channel_a")
    sig1 = lr.cache_signature()
    monkeypatch.setattr(lr, "CHANNEL_ID", "channel_b")
    sig2 = lr.cache_signature()
    assert sig1 != sig2


# ---------- load_lookbook (channel-scoping, fail-closed) ----------

def _write_lookbook(path, references, channel_id=None):
    import json
    data = {"references": references}
    if channel_id is not None:
        data["channel_id"] = channel_id
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_lookbook_no_channel_id_passes_through(monkeypatch, tmp_path):
    p = tmp_path / "lookbook.json"
    _write_lookbook(p, [_ref("a", "snow", (50, 0, 0))])   # без channel_id вообще
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(p))
    monkeypatch.setattr(lr, "CHANNEL_ID", "my_channel")
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "shadow")
    data = lr.load_lookbook()
    assert len(data["references"]) == 1
    assert "_reject_reason" not in data


def test_load_lookbook_matching_channel_id_passes_through(monkeypatch, tmp_path):
    p = tmp_path / "lookbook.json"
    _write_lookbook(p, [_ref("a", "snow", (50, 0, 0))], channel_id="my_channel")
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(p))
    monkeypatch.setattr(lr, "CHANNEL_ID", "my_channel")
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    data = lr.load_lookbook()
    assert len(data["references"]) == 1


def test_load_lookbook_mismatch_rejects_fail_closed(monkeypatch, tmp_path):
    p = tmp_path / "lookbook.json"
    _write_lookbook(p, [_ref("a", "snow", (50, 0, 0))], channel_id="channel_a")
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(p))
    monkeypatch.setattr(lr, "CHANNEL_ID", "channel_b")
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "shadow")
    data = lr.load_lookbook()
    assert data["references"] == []
    assert data["_reject_reason"] == "channel_mismatch"


def test_load_lookbook_default_not_configured_rejected_only_in_assist(monkeypatch, tmp_path):
    p = tmp_path / "lookbook.json"
    _write_lookbook(p, [_ref("a", "snow", (50, 0, 0))])   # без channel_id -> None
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(p))
    monkeypatch.setattr(lr, "CHANNEL_ID", "default")

    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    data_assist = lr.load_lookbook()
    assert data_assist["references"] == []
    assert data_assist["_reject_reason"] == "channel_not_configured"

    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "shadow")
    data_shadow = lr.load_lookbook()
    assert len(data_shadow["references"]) == 1
    assert "_reject_reason" not in data_shadow


def test_load_lookbook_empty_lookbook_ignores_channel_check(monkeypatch, tmp_path):
    p = tmp_path / "lookbook.json"
    _write_lookbook(p, [], channel_id="channel_a")
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(p))
    monkeypatch.setattr(lr, "CHANNEL_ID", "channel_b")
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    data = lr.load_lookbook()
    assert data["references"] == []
    assert "_reject_reason" not in data


# ---------- look_correction_filter (оркестрация) ----------

def test_look_correction_filter_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "off")
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False, scene_boundary=False)
    assert filt is None
    assert report["decision"] == "skipped_disabled"


def test_look_correction_filter_noop_on_empty_lookbook(monkeypatch):
    """С пустым lookbook (синтетический — см. модульный докстринг: реальный
    assets/lookbook/lookbook.json сегодня уже не пуст) система обязана быть
    полностью инертной даже при включённом режиме."""
    assert os.path.exists(lr.LOOKBOOK_PATH), "assets/lookbook/lookbook.json должен существовать в репозитории"
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": []})
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False, scene_boundary=False)
    assert filt is None
    assert report["decision"] == "skipped_empty_lookbook"


def test_look_correction_filter_noop_when_face_detected(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [_ref("a", "portrait", (50, 0, 0))]})
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=True, scene_boundary=False)
    assert filt is None
    assert report["decision"] == "skipped_face_detected"


def test_look_correction_filter_noop_without_signal(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [_ref("a", "portrait", (50, 0, 0))]})
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (None, None), None, has_face=False, scene_boundary=False)
    assert filt is None
    assert report["decision"] == "skipped_no_signal"


def test_look_correction_filter_noop_when_domain_unmatched(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [_ref("a", "snow", (50, 0, 0))]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: (None, 0.0))
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False, scene_boundary=False)
    assert filt is None
    assert report["decision"] == "skipped_low_domain_confidence"


def test_look_correction_filter_noop_when_no_reference_of_matched_domain(monkeypatch):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [_ref("a", "night", (50, 0, 0))]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("snow", 0.1))
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False, scene_boundary=False)
    assert filt is None
    assert report["decision"] == "skipped_no_reference_match"


def test_look_correction_filter_applies_when_everything_matches(monkeypatch):
    ref = _ref("snow_01", "snow", (55, 1, -2), brightness=0.5, contrast=0.4, temperature=0.0,
               max_delta=(10, 8, 8))
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [ref]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("snow", 0.1))
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False, scene_boundary=False)
    assert filt is not None
    assert filt.startswith("colorchannelmixer=rr=")
    assert report["decision"] == "applied"
    assert report["reference_id"] == "snow_01"
    assert state["delta"] is not None
    assert state["domain"] == "snow"
    assert state["reference_id"] == "snow_01"


def test_look_correction_filter_shadow_never_returns_filter_but_computes_everything(monkeypatch):
    ref = _ref("snow_01", "snow", (55, 1, -2), max_delta=(10, 8, 8))
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "shadow")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [ref]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("snow", 0.1))
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False, scene_boundary=False)
    assert filt is None   # НИКОГДА не просачивается в рендер
    assert report["decision"] == "shadow_would_apply"
    assert report["reference_id"] == "snow_01"
    assert state["delta"] is not None   # состояние обновляется как в assist — честное превью


# ---------- EMA reset при смене сцены/домена/эталона ----------

def _run_with_state(monkeypatch, ref, domain, scene_boundary, prev_state):
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [ref]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: (domain, 0.1))
    return lr.look_correction_filter(
        "x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False,
        scene_boundary=scene_boundary, prev_state=prev_state)


def test_ema_resets_on_scene_boundary(monkeypatch):
    ref = _ref("snow_01", "snow", (55, 1, -2), max_delta=(10, 8, 8))
    stale_state = {"delta": (9.0, 7.0, -7.0), "domain": "snow", "reference_id": "snow_01"}
    filt, report, state = _run_with_state(monkeypatch, ref, "snow", scene_boundary=True, prev_state=stale_state)
    assert report["ema_reset"] is True
    assert report["reset_reason"] == "scene_boundary"
    # Дельта посчитана заново (эффективный prev=None), не унаследована от stale_state.
    assert state["delta"] != stale_state["delta"]


def test_ema_resets_on_domain_change(monkeypatch):
    # lab_mean выбран близко к frame_lab у wb=(0.5,0.5,0.5) (~L=53), чтобы
    # пройти дистанционный гейт find_reference() (MAX_MATCH_DISTANCE=35) —
    # проверено численно при написании теста, не угадано.
    ref = _ref("night_01", "night", (45, -3, -8), max_delta=(10, 8, 8))
    stale_state = {"delta": (9.0, 7.0, -7.0), "domain": "snow", "reference_id": "snow_01"}
    filt, report, state = _run_with_state(monkeypatch, ref, "night", scene_boundary=False, prev_state=stale_state)
    assert report["ema_reset"] is True
    assert report["reset_reason"] == "domain_changed"
    assert state["delta"] != stale_state["delta"]


def test_ema_resets_on_reference_change_same_domain(monkeypatch):
    ref = _ref("snow_02", "snow", (60, 3, -4), max_delta=(10, 8, 8))
    stale_state = {"delta": (9.0, 7.0, -7.0), "domain": "snow", "reference_id": "snow_01"}
    filt, report, state = _run_with_state(monkeypatch, ref, "snow", scene_boundary=False, prev_state=stale_state)
    assert report["ema_reset"] is True
    assert report["reset_reason"] == "reference_changed"
    assert state["delta"] != stale_state["delta"]


def test_ema_does_not_reset_when_same_domain_and_reference(monkeypatch):
    ref = _ref("snow_01", "snow", (55, 1, -2), max_delta=(10, 8, 8))
    prev_state = {"delta": (1.0, 0.5, -0.5), "domain": "snow", "reference_id": "snow_01"}
    filt, report, state = _run_with_state(monkeypatch, ref, "snow", scene_boundary=False, prev_state=prev_state)
    assert report["ema_reset"] is False
    assert report["reset_reason"] is None


def test_hard_reject_immediately_after_reset_leaves_delta_none(monkeypatch):
    """Regression на баг, пойманный при разборе плана: reset (scene_boundary)
    сразу за которым следует hard-reject (клиппинг) НЕ должен воскресить
    дельту старой, несвязанной сцены — new_state["delta"] обязан быть None,
    не stale_state["delta"]. find_reference() замокан на заведомо клиппинг-
    сценарий (см. test_compute_correction_rejects_extreme_target_as_clipping)
    — иначе find_reference() сам отфильтровал бы этот эталон по
    MAX_MATCH_DISTANCE раньше, чем дело дойдёт до compute_correction, и
    тест проверял бы не то, что задумано."""
    ref = _ref("a", "night", (5, 0, 115), max_delta=(200, 200, 200))
    stale_state = {"delta": (9.0, 7.0, -7.0), "domain": "snow", "reference_id": "snow_01"}
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [ref]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("night", 0.1))
    monkeypatch.setattr(lr, "find_reference", lambda *a, **kw: (ref, 1.0))
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (0.02, 0.04), (0.03, 0.03, 0.03), has_face=False,
        scene_boundary=True, prev_state=stale_state)
    assert filt is None
    assert report["decision"] == "reject_clipping"
    assert report["ema_reset"] is True
    assert state["delta"] is None


# ---------- closed-loop проверка (P1-3 форензик-аудита) ----------
# compute_correction() считает gains по СЫРЫМ (негрейженным) измерениям, но
# реальный рендер приклеивает colorchannelmixer(gains) ПОСЛЕ полного
# film_look()-грейда — "ближе к эталону" на входе не гарантирует "ближе к
# эталону" на том, что реально увидит зритель. _closed_loop_improves
# рендерит настоящий film_look()-граф (через ffmpeg) и перемеряет.

def _solid_photo(path, rgb):
    Image.new("RGB", (64, 64), rgb).save(path)


@_no_ffmpeg
def test_render_graded_preview_produces_real_file(tmp_path):
    photo = tmp_path / "src.jpg"
    _solid_photo(str(photo), (120, 120, 120))
    out = lr._render_graded_preview(str(photo), "BODY", (0.3, 0.7), (0.47, 0.47, 0.47), None)
    try:
        assert out is not None
        assert os.path.exists(out)
    finally:
        if out and os.path.exists(out):
            os.remove(out)


def test_render_graded_preview_none_for_missing_file():
    assert lr._render_graded_preview("/no/such/file.jpg", "BODY", (0.3, 0.7),
                                      (0.47, 0.47, 0.47), None) is None


def test_closed_loop_improves_none_for_missing_file():
    ref = _ref("r", "battle", (60, 5, 5))
    assert lr._closed_loop_improves("/no/such/file.jpg", "BODY", (0.3, 0.7),
                                     (0.47, 0.47, 0.47), None, ref, (1.1, 1.0, 0.95)) is None


@_no_ffmpeg
def test_closed_loop_improves_true_for_correction_moving_toward_reference(tmp_path):
    # Тусклый серый источник, эталон заметно ЯРЧЕ и ТЕПЛЕЕ (высокий L,
    # положительный b) — реальные gains из compute_correction() двигают
    # именно в эту сторону, closed-loop обязан подтвердить улучшение.
    photo = tmp_path / "src.jpg"
    # (95,105,115) — холодный серый, НЕ skin-tone в YCrCb (проверено —
    # см. test_skin_tone_stats_none_for_non_skin_color), чтобы этот тест
    # проверял ИМЕННО closed-loop, не пересекаясь с skin corridor гейтом.
    _solid_photo(str(photo), (95, 105, 115))
    levels, wb = (0.3, 0.5), (95 / 255, 105 / 255, 115 / 255)
    frame_lab = lr._srgb_to_lab(wb)
    ref = _ref("bright_warm", "battle", (75, 2, 18), max_delta=(15, 10, 10))
    gains, qc, _delta = lr.compute_correction(wb, frame_lab, ref, confidence=1.0, prev_delta=None)
    assert gains is not None, f"ожидались реальные gains, получен reject: {qc}"
    result = lr._closed_loop_improves(str(photo), "BODY", levels, wb, None, ref, gains)
    assert result is True


@_no_ffmpeg
def test_closed_loop_improves_false_for_gains_moving_away_from_reference(tmp_path):
    # Те же фото/эталон, что и в тесте выше (эталон ЯРЧЕ/ТЕПЛЕЕ источника),
    # но gains намеренно НАПРАВЛЕНЫ В ОБРАТНУЮ СТОРОНУ (затемнить/охладить
    # ещё сильнее) — closed-loop обязан поймать это как ухудшение, а не
    # молча пропустить.
    photo = tmp_path / "src.jpg"
    # (95,105,115) — холодный серый, НЕ skin-tone в YCrCb (проверено —
    # см. test_skin_tone_stats_none_for_non_skin_color), чтобы этот тест
    # проверял ИМЕННО closed-loop, не пересекаясь с skin corridor гейтом.
    _solid_photo(str(photo), (95, 105, 115))
    levels, wb = (0.3, 0.5), (95 / 255, 105 / 255, 115 / 255)
    ref = _ref("bright_warm", "battle", (75, 2, 18), max_delta=(15, 10, 10))
    wrong_gains = (0.6, 0.6, 0.7)   # темнее и холоднее — прямо противоположно нужному
    result = lr._closed_loop_improves(str(photo), "BODY", levels, wb, None, ref, wrong_gains)
    assert result is False


@_no_ffmpeg
def test_look_correction_filter_rejects_when_closed_loop_disagrees(tmp_path, monkeypatch):
    # Интеграционный тест поверх юнитов выше: compute_correction() замокан
    # на заведомо "неправильные" (уводящие от эталона в графированном
    # пространстве) gains — look_correction_filter() обязан вернуть
    # reject_closed_loop_no_improvement, а не applied.
    photo = tmp_path / "src.jpg"
    _solid_photo(str(photo), (95, 105, 115))   # холодный серый, не skin-tone — см. коммент выше
    ref = _ref("bright_warm", "battle", (75, 2, 18), max_delta=(15, 10, 10))
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [ref]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("battle", 0.5))
    monkeypatch.setattr(lr, "find_reference", lambda *a, **kw: (ref, 1.0))
    monkeypatch.setattr(lr, "compute_correction",
                         lambda *a, **kw: ((0.6, 0.6, 0.7), {"decision": "ok", "notes": []}, (5.0, 1.0, 3.0)))
    filt, report, state = lr.look_correction_filter(
        str(photo), (0.3, 0.5), (95 / 255, 105 / 255, 115 / 255), has_face=False,
        scene_boundary=False, section="BODY")
    assert filt is None
    assert report["decision"] == "reject_closed_loop_no_improvement"
    assert report["closed_loop_improves"] is False


# ---------- skin tone corridor (пользователь: "как делает профессиональная
# цветокоррекция" — vectorscope skin tone line, см. константы SKIN_HUE_RANGE_DEG/
# SKIN_CHROMA_RANGE) ----------

_MEDIUM_SKIN_RGB_255 = (198, 134, 66)
_MEDIUM_SKIN_RGB_01 = tuple(c / 255 for c in _MEDIUM_SKIN_RGB_255)


def test_skin_tone_stats_none_for_non_skin_color(tmp_path):
    pytest.importorskip("cv2")
    import pipeline_smart as ps
    if not ps.PARALLAX_LIBS:
        pytest.skip("cv2 недоступен")
    photo = tmp_path / "gray.jpg"
    _solid_photo(str(photo), (95, 105, 115))
    frac, rgb = ps.skin_tone_stats(str(photo))
    assert rgb is None


def test_skin_tone_stats_detects_real_skin_color(tmp_path):
    pytest.importorskip("cv2")
    import pipeline_smart as ps
    if not ps.PARALLAX_LIBS:
        pytest.skip("cv2 недоступен")
    photo = tmp_path / "skin.jpg"
    _solid_photo(str(photo), _MEDIUM_SKIN_RGB_255)
    frac, rgb = ps.skin_tone_stats(str(photo))
    assert frac > 0.9   # сплошной кадр — почти вся площадь
    assert rgb is not None
    assert rgb[0] > rgb[2]   # красный канал заметно выше синего — тёплый тон кожи


def test_skin_gains_stay_in_corridor_identity_and_mild_correction():
    assert lr._skin_gains_stay_in_corridor(_MEDIUM_SKIN_RGB_01, (1.0, 1.0, 1.0)) is True
    assert lr._skin_gains_stay_in_corridor(_MEDIUM_SKIN_RGB_01, (0.9, 1.0, 1.3)) is True


def test_skin_gains_stay_in_corridor_false_for_hue_shifting_gains():
    # Ассиметричные gains, сдвигающие тон кожи к неестественному
    # зелёному/пурпурному — реальный случай, который коррекция под
    # "холодный доспех" вполне могла бы дать без этой защиты.
    assert lr._skin_gains_stay_in_corridor(_MEDIUM_SKIN_RGB_01, (1.4, 0.7, 1.5)) is False
    assert lr._skin_gains_stay_in_corridor(_MEDIUM_SKIN_RGB_01, (0.6, 1.3, 0.6)) is False


@_no_ffmpeg
def test_look_correction_filter_rejects_when_skin_present_and_gains_shift_hue(tmp_path, monkeypatch):
    # Интеграционный тест: реальное фото цвета кожи (рука держит меч, лица
    # каскад не нашёл — has_face=False), compute_correction() замокан на
    # gains, уводящие тон кожи из коридора — look_correction_filter()
    # обязан отклонить ДО closed-loop рендера (дешёвая проверка первой).
    import pipeline_smart as ps
    if not ps.PARALLAX_LIBS:
        pytest.skip("cv2 недоступен")
    photo = tmp_path / "skin.jpg"
    _solid_photo(str(photo), _MEDIUM_SKIN_RGB_255)
    ref = _ref("cool_steel", "battle", (40, -5, -10), max_delta=(15, 10, 10))
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [ref]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("battle", 0.5))
    monkeypatch.setattr(lr, "find_reference", lambda *a, **kw: (ref, 1.0))
    monkeypatch.setattr(lr, "compute_correction",
                         lambda *a, **kw: ((0.6, 1.3, 0.6), {"decision": "ok", "notes": []}, (5.0, 1.0, 3.0)))
    filt, report, state = lr.look_correction_filter(
        str(photo), (0.3, 0.7), _MEDIUM_SKIN_RGB_01, has_face=False,
        scene_boundary=False, section="BODY")
    assert filt is None
    assert report["decision"] == "reject_skin_tone_corridor"
    assert report["skin_fraction"] > 0.9


def test_look_correction_filter_reports_skin_fraction_even_when_zero(monkeypatch):
    # skin_fraction обязан быть в отчёте даже когда кожи в кадре нет (0.0),
    # для честного аудит-трейла (тот же принцип, что look_manifest.json
    # уже применяет к остальным полям)."""
    ref = _ref("snow_01", "snow", (55, 1, -2), brightness=0.5, contrast=0.4, temperature=0.0,
               max_delta=(10, 8, 8))
    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "assist")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [ref]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("snow", 0.1))
    filt, report, state = lr.look_correction_filter(
        "x.jpg", (0.1, 0.9), (0.5, 0.5, 0.5), has_face=False, scene_boundary=False)
    assert "skin_fraction" in report
