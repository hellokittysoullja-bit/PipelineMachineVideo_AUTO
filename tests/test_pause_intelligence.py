"""Юнит-тесты Pause Intelligence (scripts/pause_intelligence.py) — безопасное
подмножество внешнего ТЗ "Adaptive Pause Intelligence": cooldown между
соседними длинными protected-паузами одной секции, confidence-gate для
reveal_hold, music_action-классификация. Все проверки на синтетических
данных — модуль не трогает TTS/файлы."""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pause_intelligence.py", tempfile.gettempdir()]
import pause_intelligence as pi   # noqa: E402


def _entry(unit_id, section, kind, raw_window, kept, lo, hi):
    return {"unit_id": unit_id, "section": section, "rhetorical_kind": kind,
            "raw_window": list(raw_window), "kept": kept, "lo": lo, "hi": hi}


# ---------- confidence_for / apply_confidence_gate ----------

def test_confidence_for_known_kinds():
    assert pi.confidence_for("reveal_hold") == 1.0
    assert pi.confidence_for("evidence_beat") == 0.9
    assert pi.confidence_for("connective") == 0.5


def test_confidence_for_unknown_kind_defaults_neutral():
    assert pi.confidence_for("some_future_kind") == 0.5


def test_gate_is_noop_for_non_reveal_hold_kinds():
    hi, passed, note = pi.apply_confidence_gate("closing_hold", 0.1, 1.10)
    assert hi == 1.10
    assert passed is True
    assert note is None


def test_gate_passes_reveal_hold_at_real_confidence():
    # Сегодня confidence для reveal_hold всегда 1.0 (явный тег [climax]) —
    # ворота фактически no-op, но должны явно пропускать.
    hi, passed, note = pi.apply_confidence_gate("reveal_hold", pi.confidence_for("reveal_hold"), 1.20)
    assert hi == 1.20
    assert passed is True
    assert note is None


def test_gate_downgrades_reveal_hold_when_confidence_synthetically_low():
    # Синтетический случай — сегодня в пайплайне confidence для reveal_hold
    # не бывает ниже 1.0, но структура ворот обязана работать корректно,
    # если/когда climax начнёт определяться не только явным тегом.
    hi, passed, note = pi.apply_confidence_gate("reveal_hold", 0.3, 1.20)
    assert passed is False
    assert note is not None
    assert hi <= 1.20
    assert hi == min(1.20, pi.splanner.RHETORICAL_RANGES["evidence_beat"][1])


# ---------- apply_cooldown ----------

def test_cooldown_not_applied_to_single_long_pause():
    entries = [_entry("A#0", "BLOCK1", "reveal_hold", (10.0, 11.2), 1.2, 0.9, 1.2)]
    out = pi.apply_cooldown(entries)
    assert out[0]["cooldown_applied"] is False
    assert out[0]["kept"] == 1.2


def test_cooldown_downgrades_second_of_two_close_long_pauses_same_section():
    entries = [
        _entry("A#0", "BLOCK1", "reveal_hold", (10.0, 11.2), 1.2, 0.9, 1.2),
        _entry("A#1", "BLOCK1", "closing_hold", (11.5, 12.6), 1.1, 0.7, 1.1),   # 0.3с речи между ними
    ]
    out = pi.apply_cooldown(entries)
    assert out[0]["cooldown_applied"] is False
    assert out[0]["kept"] == 1.2
    assert out[1]["cooldown_applied"] is True
    assert out[1]["kept"] == 0.7   # понижен до lo своей же роли


def test_cooldown_not_applied_when_gap_wide_enough():
    entries = [
        _entry("A#0", "BLOCK1", "reveal_hold", (10.0, 11.2), 1.2, 0.9, 1.2),
        _entry("A#1", "BLOCK1", "closing_hold", (13.5, 14.6), 1.1, 0.7, 1.1),   # 2.3с речи между ними
    ]
    out = pi.apply_cooldown(entries)
    assert out[1]["cooldown_applied"] is False
    assert out[1]["kept"] == 1.1


def test_cooldown_never_compares_across_sections():
    entries = [
        _entry("A#0", "BLOCK1", "reveal_hold", (10.0, 11.2), 1.2, 0.9, 1.2),
        _entry("B#0", "BLOCK2", "closing_hold", (11.3, 12.4), 1.1, 0.7, 1.1),   # соседняя секция, встык
    ]
    out = pi.apply_cooldown(entries)
    assert out[1]["cooldown_applied"] is False
    assert out[1]["kept"] == 1.1


def test_cooldown_downgrade_is_transitive_through_intervening_short_pause():
    # Короткая (не длинная) пауза между двумя длинными не "сбрасывает"
    # кулдаун — правило .md прямо про "не ставить ДВЕ длинные подряд", не
    # про соседство по индексу.
    entries = [
        _entry("A#0", "BLOCK1", "reveal_hold", (10.0, 11.2), 1.2, 0.9, 1.2),
        _entry("A#1", "BLOCK1", "evidence_beat", (11.3, 11.7), 0.4, 0.35, 0.55),   # короткая, не long
        _entry("A#2", "BLOCK1", "closing_hold", (11.9, 13.0), 1.1, 0.7, 1.1),      # рядом с A#0, не с A#1
    ]
    out = pi.apply_cooldown(entries)
    assert out[1]["cooldown_applied"] is False   # сама короткая — не подпадает под порог
    assert out[2]["cooldown_applied"] is True
    assert out[2]["kept"] == 0.7


def test_cooldown_only_ever_downgrades_never_upgrades():
    entries = [
        _entry("A#0", "BLOCK1", "reveal_hold", (10.0, 11.2), 1.2, 0.9, 1.2),
        _entry("A#1", "BLOCK1", "closing_hold", (11.5, 12.6), 1.1, 0.7, 1.1),
        _entry("A#2", "BLOCK1", "evidence_beat", (20.0, 20.6), 0.5, 0.35, 0.55),
    ]
    out = pi.apply_cooldown(entries)
    for original, adjusted in zip(entries, out):
        assert adjusted["kept"] <= original["kept"]


# ---------- _music_action_for ----------

def test_music_action_reveal_hold_always_dip_only():
    assert pi._music_action_for("reveal_hold", 1.20) == "climax_dip_only_swell_suppressed"
    assert pi._music_action_for("reveal_hold", 0.05) == "climax_dip_only_swell_suppressed"


def test_music_action_swell_above_threshold_for_other_kinds():
    assert pi._music_action_for("closing_hold", pi.pipeline_smart.PAUSE_SWELL_MIN_KEEP_SEC) == "swell"


def test_music_action_none_below_threshold():
    assert pi._music_action_for("closing_hold", pi.pipeline_smart.PAUSE_SWELL_MIN_KEEP_SEC - 0.01) == "none"


# ---------- build_decisions (интеграция gate + cooldown + music_action) ----------

def test_build_decisions_end_to_end_shape():
    entries = [
        _entry("A#0", "BLOCK1", "reveal_hold", (10.0, 11.2), 1.2, 0.9, 1.2),
        _entry("A#1", "BLOCK1", "closing_hold", (11.5, 12.6), 1.1, 0.7, 1.1),
    ]
    decisions = pi.build_decisions(entries)
    assert len(decisions) == 2
    d0, d1 = decisions
    assert d0["unit_id"] == "A#0"
    assert d0["rhetorical_kind"] == "reveal_hold"
    assert d0["music_action"] == "climax_dip_only_swell_suppressed"
    assert d0["visual_action"] == pi.VISUAL_ACTION_EXISTING_CUT
    assert d0["subtitle_action"] == pi.SUBTITLE_ACTION_NOT_EVALUATED
    assert d0["confidence_gate_passed"] is True
    assert d1["cooldown_applied"] is True
    assert d1["final_target_sec"] == 0.7
    assert any("cooldown" in n for n in d1["notes"])


def test_build_decisions_preserves_order_and_count():
    entries = [
        _entry(f"A#{i}", "BLOCK1", "evidence_beat", (float(i), float(i) + 0.4), 0.4, 0.35, 0.55)
        for i in range(5)
    ]
    decisions = pi.build_decisions(entries)
    assert [d["unit_id"] for d in decisions] == [e["unit_id"] for e in entries]


def test_build_decisions_empty_input_is_noop():
    assert pi.build_decisions([]) == []
