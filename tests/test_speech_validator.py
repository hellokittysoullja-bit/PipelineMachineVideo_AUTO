"""Юнит-тесты Cadence Validator (scripts/speech_validator.py): вердикты по
типам юнитов, и — самое важное — что пауза НИКОГДА не считается через
границу секции (alignment.csv разных секций может иметь разные нулевые
точки времени, реальный баг, пойманный интеграционным тестом при
разработке: разница между файлами давала отрицательную/бессмысленную
величину, окно вида [5.05, 0.0] — конец раньше начала)."""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["speech_validator.py", tempfile.gettempdir()]
import speech_validator as sv   # noqa: E402


def _unit(unit_id, section, kind, target, protected=True):
    return {"unit_id": unit_id, "section": section, "text": "x", "words": 1,
            "rhetorical_kind": kind, "target_range_sec": list(target), "protected": protected,
            "tag": None, "tag_suggestion": None, "stat": None, "stat_word_pos": None, "is_climax": False}


# ---------- _validate_unit ----------

def test_no_signal_when_observed_is_none():
    u = _unit("a#0", "HOOK", "reveal_hold", (0.9, 1.4))
    v = sv._validate_unit(u, None)
    assert v["status"] == "no_signal"
    assert v["action"] == "none"


def test_reveal_hold_ok_when_close_to_target():
    u = _unit("a#0", "BODY", "reveal_hold", (0.9, 1.4))
    v = sv._validate_unit(u, 1.0)
    assert v["status"] == "ok"


def test_reveal_hold_mismatch_short_when_far_below_target():
    u = _unit("a#0", "BODY", "reveal_hold", (0.9, 1.4))
    v = sv._validate_unit(u, 0.3)   # < 0.9 * 0.6 = 0.54
    assert v["status"] == "mismatch_short"
    assert v["action"] == "re-record phrase manually"


def test_reveal_hold_ok_when_overshoots_target():
    # Перебор цели — не провал (fix_pauses просто обрежет до target_hi).
    u = _unit("a#0", "BODY", "reveal_hold", (0.9, 1.4))
    v = sv._validate_unit(u, 3.0)
    assert v["status"] == "ok"


def test_punchy_mismatch_long_when_far_above_target():
    u = _unit("a#0", "HOOK", "punchy", (0.15, 0.35))
    v = sv._validate_unit(u, 1.0)   # > 0.35 * 2.0 = 0.70
    assert v["status"] == "mismatch_long"
    assert v["action"] == "re-record phrase manually"


def test_punchy_ok_when_within_reasonable_range():
    u = _unit("a#0", "HOOK", "punchy", (0.15, 0.35))
    v = sv._validate_unit(u, 0.5)
    assert v["status"] == "ok"


def test_connective_zero_range_is_ok_without_signal_requirement():
    u = _unit("a#0", "BODY", "connective", (0.0, 0.0))
    v = sv._validate_unit(u, 0.0)
    assert v["status"] == "ok"


# ---------- граница секции: НИКОГДА не считать разрыв между файлами ----------

def test_build_timeline_never_computes_gap_across_section_boundary(monkeypatch):
    units = [
        _unit("HOOK#0", "HOOK", "punchy", (0.15, 0.35)),
        _unit("BODY#0", "BODY", "connective", (0.55, 0.95)),
    ]
    # HOOK-файл заканчивается на t=5.0 (своя шкала), BODY-файл НАЧИНАЕТСЯ на
    # t=0.2 (своя шкала, с нуля) — наивная разность (0.2 - 5.0) была бы
    # отрицательной, что и произошло в реальном баге при разработке.
    fake_bounds = {
        "HOOK": [(4.0, 5.0)],
        "BODY": [(0.2, 1.0)],
    }
    monkeypatch.setattr(sv, "_load_section_segments", lambda section_order: {"HOOK": [1], "BODY": [1]})
    monkeypatch.setattr(sv, "_flat_segment_bounds",
                         lambda units, segs: [fake_bounds["HOOK"][0], fake_bounds["BODY"][0]])
    timeline = sv.build_timeline(units, audio_md5="deadbeef")
    assert timeline["units"][0]["validation"]["status"] == "no_signal"
    assert timeline["units"][0]["raw_window"] is None
    assert timeline["protected_windows"] == []   # ни одного окна не построено через границу


def test_build_timeline_computes_gap_within_same_section(monkeypatch):
    units = [
        _unit("HOOK#0", "HOOK", "reveal_hold", (0.9, 1.4)),
        _unit("HOOK#1", "HOOK", "connective", (0.55, 0.95)),
    ]
    monkeypatch.setattr(sv, "_load_section_segments", lambda section_order: {"HOOK": [1, 1]})
    monkeypatch.setattr(sv, "_flat_segment_bounds", lambda units, segs: [(1.0, 2.0), (3.1, 4.0)])
    timeline = sv.build_timeline(units, audio_md5="deadbeef")
    u0 = timeline["units"][0]
    assert u0["validation"]["observed_value"] == 1.1   # 3.1 - 2.0
    assert u0["raw_window"] == [2.0, 3.1]
    assert len(timeline["protected_windows"]) == 1
    assert timeline["protected_windows"][0][:2] == [2.0, 3.1]


def test_build_timeline_never_negative_even_within_section(monkeypatch):
    # Защитный клэмп: даже ВНУТРИ одной секции если данные почему-то дали
    # next_start < this_end (битый/рассинхроненный alignment) — не считаем
    # observed_raw_dur, не строим protected-окно.
    units = [
        _unit("HOOK#0", "HOOK", "reveal_hold", (0.9, 1.4)),
        _unit("HOOK#1", "HOOK", "connective", (0.55, 0.95)),
    ]
    monkeypatch.setattr(sv, "_load_section_segments", lambda section_order: {"HOOK": [1, 1]})
    monkeypatch.setattr(sv, "_flat_segment_bounds", lambda units, segs: [(2.0, 3.0), (2.5, 2.9)])
    timeline = sv.build_timeline(units, audio_md5="deadbeef")
    assert timeline["units"][0]["validation"]["status"] == "no_signal"
    assert timeline["protected_windows"] == []


def test_protected_windows_only_include_protected_units(monkeypatch):
    units = [
        _unit("HOOK#0", "HOOK", "connective", (0.25, 0.5), protected=False),
        _unit("HOOK#1", "HOOK", "connective", (0.55, 0.95)),
    ]
    monkeypatch.setattr(sv, "_load_section_segments", lambda section_order: {"HOOK": [1, 1]})
    monkeypatch.setattr(sv, "_flat_segment_bounds", lambda units, segs: [(1.0, 1.5), (2.0, 2.5)])
    timeline = sv.build_timeline(units, audio_md5="deadbeef")
    assert timeline["protected_windows"] == []   # оба unprotected (connective, protected=False)


def test_protected_window_target_never_exceeds_observed_plus_margin(monkeypatch):
    # kept = min(observed, target_hi) — никогда не "изобретаем" тишину сверх
    # того, что реально есть в аудио.
    units = [
        _unit("HOOK#0", "HOOK", "reveal_hold", (0.9, 1.4)),
        _unit("HOOK#1", "HOOK", "connective", (0.55, 0.95)),
    ]
    monkeypatch.setattr(sv, "_load_section_segments", lambda section_order: {"HOOK": [1, 1]})
    monkeypatch.setattr(sv, "_flat_segment_bounds", lambda units, segs: [(1.0, 2.0), (2.3, 3.0)])
    # observed = 0.3с — короче нижней границы 0.9, kept обязан остаться 0.3 (min(observed, hi)), не 0.9
    timeline = sv.build_timeline(units, audio_md5="deadbeef")
    kept = timeline["protected_windows"][0][2]
    assert abs(kept - 0.3) < 1e-6


# ---------- Pause Intelligence wiring (cooldown + аудит-трейл) ----------

def test_build_timeline_exposes_pause_decisions_key(monkeypatch):
    units = [
        _unit("HOOK#0", "HOOK", "reveal_hold", (0.9, 1.2)),
        _unit("HOOK#1", "HOOK", "connective", (0.55, 0.95)),
    ]
    monkeypatch.setattr(sv, "_load_section_segments", lambda section_order: {"HOOK": [1, 1]})
    monkeypatch.setattr(sv, "_flat_segment_bounds", lambda units, segs: [(1.0, 2.0), (3.0, 3.5)])
    timeline = sv.build_timeline(units, audio_md5="deadbeef")
    assert "pause_decisions" in timeline
    assert len(timeline["pause_decisions"]) == 1
    assert timeline["pause_decisions"][0]["unit_id"] == "HOOK#0"


def test_build_timeline_cooldown_shrinks_second_of_two_close_long_holds(monkeypatch):
    # Две протяжённые protected-паузы одной секции близко друг к другу
    # (1.2с речи между ними, меньше COOLDOWN_MIN_SPEECH_GAP_SEC=2.0) —
    # Pause Intelligence обязан понизить вторую до её lo,
    # protected_windows[1][2] должен отразить ИМЕННО это (а не сырую цель
    # плана) — иначе fix_pauses.py всё равно вырежет две длинные паузы
    # подряд, кулдаун существовал бы только на бумаге. Третий юнит нужен
    # только чтобы дать границу для паузы ПОСЛЕ второго юнита (см.
    # _flat_segment_bounds — окно юнита i строится из bounds[i]/bounds[i+1]).
    units = [
        _unit("BLOCK1#0", "BLOCK1", "reveal_hold", (0.9, 1.2)),
        _unit("BLOCK1#1", "BLOCK1", "closing_hold", (0.7, 1.1)),
        _unit("BLOCK1#2", "BLOCK1", "connective", (0.55, 0.95)),
    ]
    monkeypatch.setattr(sv, "_load_section_segments", lambda section_order: {"BLOCK1": [1, 1, 1]})
    # юнит0: речь 1.0-2.0, пауза 2.0-3.5 (1.5с сырых, min(1.5, hi=1.2) -> 1.2)
    # юнит1: речь 3.5-4.7 (сразу после паузы0 — 0с зазора внутри самой речи
    # не считается, гэп между паузами меряется по их собственным границам),
    # пауза 4.7-5.9 (1.2с сырых, min(1.2, hi=1.1) -> 1.1 ДО кулдауна)
    # юнит2: речь начинается 5.9 — просто закрывает окно паузы юнита1
    monkeypatch.setattr(sv, "_flat_segment_bounds",
                         lambda units, segs: [(1.0, 2.0), (3.5, 4.7), (5.9, 6.1)])
    timeline = sv.build_timeline(units, audio_md5="deadbeef")
    pw = timeline["protected_windows"]
    assert len(pw) == 2
    assert abs(pw[0][2] - 1.2) < 1e-6   # первая пауза — без изменений
    assert abs(pw[1][2] - 0.7) < 1e-6   # вторая — понижена кулдауном до lo=0.7
    assert timeline["pause_decisions"][1]["cooldown_applied"] is True


# ---------- main(): коды возврата — сбой (1) vs advisory (2) vs чисто (0) ----------
# Регрессия: раньше 1 означало и "реальный сбой" (нет входа/невалидный план),
# и "прогон прошёл успешно, но часть фраз стоит переслушать" — при цепочном
# запуске (`&&`) эти два исхода были неотличимы.

def _write_json(path, data):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_main_returns_1_when_speech_plan_missing(tmp_path):
    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = ["speech_validator.py", str(tmp_path)]
    try:
        assert sv.main() == 1
    finally:
        _sys.argv = old_argv


def test_main_returns_2_when_a_unit_needs_retake(tmp_path, monkeypatch):
    plan = {"units": [_unit("HOOK#0", "HOOK", "punchy", (0.15, 0.35))]}
    os.makedirs(tmp_path / "media_plan")
    _write_json(tmp_path / "media_plan" / "speech_plan.json", plan)
    (tmp_path / "audio.mp3").write_bytes(b"fake audio bytes")

    fake_timeline = {
        "units": [{"unit_id": "HOOK#0",
                   "validation": {"status": "mismatch_long", "action": "re-record phrase manually",
                                   "reason": "перебор темпа"}}],
        "protected_windows": [], "pause_decisions": [],
    }
    monkeypatch.setattr(sv, "build_timeline", lambda units, audio_md5: fake_timeline)

    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = ["speech_validator.py", str(tmp_path)]
    try:
        assert sv.main() == 2
    finally:
        _sys.argv = old_argv


def test_main_returns_0_when_all_units_ok(tmp_path, monkeypatch):
    plan = {"units": [_unit("HOOK#0", "HOOK", "punchy", (0.15, 0.35))]}
    os.makedirs(tmp_path / "media_plan")
    _write_json(tmp_path / "media_plan" / "speech_plan.json", plan)
    (tmp_path / "audio.mp3").write_bytes(b"fake audio bytes")

    fake_timeline = {
        "units": [{"unit_id": "HOOK#0",
                   "validation": {"status": "ok", "action": "none", "reason": "в норме"}}],
        "protected_windows": [], "pause_decisions": [],
    }
    monkeypatch.setattr(sv, "build_timeline", lambda units, audio_md5: fake_timeline)

    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = ["speech_validator.py", str(tmp_path)]
    try:
        assert sv.main() == 0
    finally:
        _sys.argv = old_argv
