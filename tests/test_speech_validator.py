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
