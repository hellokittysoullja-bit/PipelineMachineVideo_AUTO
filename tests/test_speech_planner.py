"""Юнит-тесты Speech Planner (scripts/speech_planner.py): классификация
риторических единиц, диапазоны, валидация плана — чистая логика, без TTS."""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["speech_planner.py", tempfile.gettempdir()]
import speech_planner as sp   # noqa: E402
import pytest                 # noqa: E402


def _block(text, words=None, section="BODY", pause_after=0.8, stat=None,
           stat_word_pos=None, is_climax=False):
    return {"text": text, "words": words if words is not None else len(text.split()),
            "section": section, "pause_after": pause_after, "stat": stat,
            "stat_word_pos": stat_word_pos, "is_climax": is_climax}


# ---------- classify_unit ----------

def test_climax_beats_everything_else():
    b = _block("Короткая?", section="HOOK", is_climax=True)
    assert sp.classify_unit(b, 0, [b]) == "reveal_hold"


def test_stat_gives_evidence_beat():
    b = _block("Всего два кило.", stat="2 КГ")
    assert sp.classify_unit(b, 0, [b]) == "evidence_beat"


def test_last_final_block_is_closing_hold():
    blocks = [_block("Первая мысль в финале.", section="FINAL"),
              _block("Последняя мысль.", section="FINAL")]
    assert sp.classify_unit(blocks[1], 1, blocks) == "closing_hold"


def test_non_last_final_block_is_not_closing_hold():
    blocks = [_block("Первая мысль в финале.", section="FINAL"),
              _block("Последняя мысль.", section="FINAL")]
    assert sp.classify_unit(blocks[0], 0, blocks) != "closing_hold"


def test_question_mark_gives_question_rise():
    b = _block("Откуда такая цифра?")
    assert sp.classify_unit(b, 0, [b]) == "question_rise"


def test_dash_ending_gives_anticipation():
    b = _block("И вот тут возникает вопрос —")
    assert sp.classify_unit(b, 0, [b]) == "anticipation"


def test_colon_ending_gives_anticipation():
    b = _block("Ответ прост:")
    assert sp.classify_unit(b, 0, [b]) == "anticipation"


def test_short_hook_phrase_is_punchy():
    b = _block("Меч. Тяжёлый.", section="HOOK")
    assert sp.classify_unit(b, 0, [b]) == "punchy"


def test_long_hook_phrase_is_not_punchy():
    b = _block("Это довольно длинная фраза для короткого удара в хуке.", section="HOOK")
    assert sp.classify_unit(b, 0, [b]) != "punchy"


def test_default_is_connective():
    b = _block("Обычная связующая фраза без особых сигналов")
    assert sp.classify_unit(b, 0, [b]) == "connective"


# ---------- target_range_for / tag_suggestion ----------

def test_connective_range_follows_pause_tag():
    b = _block("x", pause_after=0.8)
    assert sp.target_range_for(b, "connective") == sp.TAG_BASE_RANGE["[pause]"]


def test_connective_range_follows_short_pause_tag():
    b = _block("x", pause_after=0.4)
    assert sp.target_range_for(b, "connective") == sp.TAG_BASE_RANGE["[short pause]"]


def test_reveal_hold_range_is_rhetorical_not_tag_based():
    b = _block("x", pause_after=0.4)   # тег слабый, но роль всё равно reveal_hold
    assert sp.target_range_for(b, "reveal_hold") == sp.RHETORICAL_RANGES["reveal_hold"]


def test_tag_suggestion_upgrades_weak_tag_for_reveal_hold():
    b = _block("x", pause_after=0.4)   # [short pause] — слабее рекомендованного [pause]
    assert sp.tag_suggestion(b, "reveal_hold") == "[pause]"


def test_tag_suggestion_none_when_already_strong_enough():
    b = _block("x", pause_after=0.8)   # уже [pause] — рекомендованный минимум для reveal_hold
    assert sp.tag_suggestion(b, "reveal_hold") is None


def test_tag_suggestion_none_for_connective():
    b = _block("x", pause_after=0.0)
    assert sp.tag_suggestion(b, "connective") is None


def test_tag_suggestion_never_recommends_disallowed_tag():
    # Регрессия: reveal_hold не должен рекомендовать [long pause] — тег,
    # который CLAUDE.md прямо запрещает как источник TTS-артефактов.
    for kind, min_tag in sp.RHETORICAL_MIN_TAG.items():
        assert min_tag in sp.ALLOWED_TAGS
        assert min_tag != "[long pause]"


# ---------- build_units / validate_plan ----------

def test_build_units_produces_unique_ids_with_safe_separator():
    blocks = [_block("Один.", section="BLOCK 1: Вес меча"),
              _block("Два.", section="BLOCK 1: Вес меча")]
    units = sp.build_units(blocks)
    ids = [u["unit_id"] for u in units]
    assert len(set(ids)) == 2
    assert ids[0] == "BLOCK 1: Вес меча#0"
    assert ids[1] == "BLOCK 1: Вес меча#1"


def test_validate_plan_accepts_well_formed_plan():
    blocks = [_block("Тест.", section="HOOK")]
    plan = {"units": sp.build_units(blocks)}
    assert sp.validate_plan(plan) is True


def test_validate_plan_rejects_missing_units_key():
    with pytest.raises(ValueError):
        sp.validate_plan({})


def test_validate_plan_rejects_out_of_range_target():
    plan = {"units": [{"unit_id": "a#0", "section": "HOOK", "text": "x",
                        "rhetorical_kind": "connective", "target_range_sec": [0.1, 99.0],
                        "protected": False}]}
    with pytest.raises(ValueError):
        sp.validate_plan(plan)


def test_validate_plan_rejects_unknown_rhetorical_kind():
    plan = {"units": [{"unit_id": "a#0", "section": "HOOK", "text": "x",
                        "rhetorical_kind": "mystery", "target_range_sec": [0.1, 0.5],
                        "protected": False}]}
    with pytest.raises(ValueError):
        sp.validate_plan(plan)


def test_validate_plan_rejects_disallowed_tag_suggestion():
    plan = {"units": [{"unit_id": "a#0", "section": "HOOK", "text": "x",
                        "rhetorical_kind": "connective", "target_range_sec": [0.1, 0.5],
                        "protected": False, "tag_suggestion": "[long pause]"}]}
    with pytest.raises(ValueError):
        sp.validate_plan(plan)


def test_validate_plan_rejects_duplicate_unit_id():
    plan = {"units": [
        {"unit_id": "a#0", "section": "HOOK", "text": "x", "rhetorical_kind": "connective",
         "target_range_sec": [0.1, 0.5], "protected": False},
        {"unit_id": "a#0", "section": "HOOK", "text": "y", "rhetorical_kind": "connective",
         "target_range_sec": [0.1, 0.5], "protected": False},
    ]}
    with pytest.raises(ValueError):
        sp.validate_plan(plan)


# ---------- end-to-end через реальный parse_blocks ----------

def test_end_to_end_script_produces_valid_plan(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text(
        "=== HOOK === Ты видел это сто раз.[short pause]И это неправда.\n"
        "=== BLOCK 1: Тест === Обычная фраза.[pause][climax]Вот правда.[pause]Всё просто.\n"
        "=== FINAL === Финальная мысль.\n",
        encoding="utf-8")
    blocks = sp.pipeline_smart.parse_blocks(str(script))
    units = sp.build_units(blocks)
    plan = {"units": units}
    assert sp.validate_plan(plan) is True
    kinds = {u["rhetorical_kind"] for u in units}
    assert "reveal_hold" in kinds
    assert "closing_hold" in kinds
