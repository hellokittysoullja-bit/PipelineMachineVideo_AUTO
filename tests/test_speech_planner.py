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

def test_unit_before_climax_gets_reveal_hold():
    # "Тишина как акцент ПЕРЕД разоблачением" (см. CLIMAX_DIP_LEAD_SEC в
    # pipeline_smart.py — дип начинается ДО climax-момента, не после) —
    # reveal_hold защищает ХВОСТОВУЮ паузу юнита ПЕРЕД climax-блоком, не
    # сам climax-блок (та пауза уже прозвучала ПОСЛЕ разоблачения).
    setup = _block("Обычная фраза перед разоблачением.")
    reveal = _block("Короткая?", section="HOOK", is_climax=True)
    blocks = [setup, reveal]
    assert sp.classify_unit(setup, 0, blocks) == "reveal_hold"


def test_climax_block_itself_is_not_reveal_hold():
    setup = _block("Обычная фраза перед разоблачением.")
    reveal = _block("Короткая?", section="HOOK", is_climax=True)
    blocks = [setup, reveal]
    assert sp.classify_unit(reveal, 1, blocks) != "reveal_hold"


def test_climax_as_last_block_does_not_crash():
    # idx+1 выходит за границы blocks, когда climax — последний блок
    # сценария (например, единственная строка в HOOK) — не должно падать,
    # просто некому получить reveal_hold (нет предыдущего блока с этим
    # is_climax соседом... сам по себе climax-блок без предыдущего юнита).
    b = _block("Короткая?", section="HOOK", is_climax=True)
    assert sp.classify_unit(b, 0, [b]) != "reveal_hold"


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


# ---------- assign_chapter_arcs (планирование по главам) ----------

def _make_units_from_script(tmp_path, text):
    script = tmp_path / "script.txt"
    script.write_text(text, encoding="utf-8")
    blocks = sp.pipeline_smart.parse_blocks(str(script))
    units = sp.build_units(blocks)
    sp.assign_chapter_arcs(units)
    return units


def test_assign_chapter_arcs_hook_gets_single_stage(tmp_path):
    units = _make_units_from_script(tmp_path, "=== HOOK === Раз.[pause]Два.[pause]Три.\n")
    hook_units = [u for u in units if u["section"] == "HOOK"]
    assert all(u["arc_stage"] == "hook" for u in hook_units)
    assert all(u["chapter_id"] == 0 for u in hook_units)


def test_assign_chapter_arcs_final_gets_single_stage(tmp_path):
    units = _make_units_from_script(tmp_path, "=== FINAL === Конец истории.\n")
    assert units[0]["arc_stage"] == "final"


def test_assign_chapter_arcs_block_covers_all_seven_stages_with_enough_units(tmp_path):
    # Границы стадий (см. _BLOCK_STAGE_BOUNDARIES) НЕ равномерны по замыслу
    # (CLAUDE.md отводит "доказательству" больше доли блока, чем "заходу-
    # якорю") — при 7 юнитах на 7 стадий поровну НЕ выйдет (проверено:
    # первый юнит уже попадает в "постановка", не в "заход-якорь", т.к.
    # 1/7 > 0.10). С МНОГИМИ юнитами (тут 40 — заведомо больше, чем полос)
    # каждая стадия обязана появиться хотя бы раз, и стадии обязаны идти
    # в документированном порядке без скачков назад.
    text = "=== BLOCK 1: Тест === " + "[pause]".join([f"Юнит {i}." for i in range(40)]) + "\n"
    units = _make_units_from_script(tmp_path, text)
    stages = [u["arc_stage"] for u in units]
    assert set(stages) == set(sp.CHAPTER_ARC_STAGES_BLOCK)
    seen_order = []
    for s in stages:
        if not seen_order or seen_order[-1] != s:
            seen_order.append(s)
    assert seen_order == list(sp.CHAPTER_ARC_STAGES_BLOCK)


def test_assign_chapter_arcs_first_unit_of_block_is_early_stage(tmp_path):
    text = "=== BLOCK 1: Тест === " + "[pause]".join([f"Юнит {i}." for i in range(40)]) + "\n"
    units = _make_units_from_script(tmp_path, text)
    assert units[0]["arc_stage"] in ("заход-якорь", "постановка")


def test_assign_chapter_arcs_last_unit_of_block_is_late_stage(tmp_path):
    text = "=== BLOCK 1: Тест === " + "[pause]".join([f"Юнит {i}." for i in range(40)]) + "\n"
    units = _make_units_from_script(tmp_path, text)
    assert units[-1]["arc_stage"] in ("перенос-на-зрителя", "мостик")


def test_assign_chapter_arcs_different_sections_get_different_chapter_ids(tmp_path):
    text = ("=== HOOK === Хук.\n"
            "=== BLOCK 1: А === Блок раз.\n"
            "=== BLOCK 2: Б === Блок два.\n"
            "=== FINAL === Финал.\n")
    units = _make_units_from_script(tmp_path, text)
    chapter_ids = [u["chapter_id"] for u in units]
    assert chapter_ids == [0, 1, 2, 3]


def test_assign_chapter_arcs_single_unit_block_gets_final_stage_of_arc():
    units = [{"section": "BLOCK 1", "text": "x", "words": 1, "unit_id": "a#0"}]
    sp.assign_chapter_arcs(units)
    # Один юнит -> frac = 1/1 = 1.0 -> первая стадия с right-bound >= 1.0 -> "мостик"
    assert units[0]["arc_stage"] == "мостик"
    assert units[0]["tempo_mult"] == sp.ARC_STAGE_PROFILE["мостик"]["tempo_mult"]
    assert units[0]["target_wpm"] == round(sp.BASE_WPM * sp.ARC_STAGE_PROFILE["мостик"]["tempo_mult"], 1)


def test_assign_chapter_arcs_mutates_and_returns_same_list():
    units = [{"section": "HOOK", "text": "x", "words": 1, "unit_id": "a#0"}]
    result = sp.assign_chapter_arcs(units)
    assert result is units
    assert "arc_stage" in units[0]


# ---------- validate_plan с новыми полями (chapter_id/arc_stage/tempo_mult) ----------

def test_validate_plan_accepts_units_without_arc_fields():
    # Обратная совместимость: план БЕЗ chapter/arc-полей всё ещё валиден
    # (старый план, записанный до этой правки).
    plan = {"units": [{"unit_id": "a#0", "section": "HOOK", "text": "x",
                       "rhetorical_kind": "connective", "target_range_sec": [0.1, 0.5],
                       "protected": False}]}
    assert sp.validate_plan(plan) is True


def test_validate_plan_rejects_unknown_arc_stage():
    plan = {"units": [{"unit_id": "a#0", "section": "HOOK", "text": "x",
                       "rhetorical_kind": "connective", "target_range_sec": [0.1, 0.5],
                       "protected": False, "arc_stage": "несуществующая-стадия"}]}
    with pytest.raises(ValueError):
        sp.validate_plan(plan)


def test_validate_plan_rejects_invalid_tempo_mult():
    plan = {"units": [{"unit_id": "a#0", "section": "HOOK", "text": "x",
                       "rhetorical_kind": "connective", "target_range_sec": [0.1, 0.5],
                       "protected": False, "tempo_mult": 99.0}]}
    with pytest.raises(ValueError):
        sp.validate_plan(plan)


def test_validate_plan_accepts_valid_arc_fields():
    plan = {"units": [{"unit_id": "a#0", "section": "HOOK", "text": "x",
                       "rhetorical_kind": "connective", "target_range_sec": [0.1, 0.5],
                       "protected": False, "arc_stage": "hook", "tempo_mult": 1.05}]}
    assert sp.validate_plan(plan) is True


def test_main_populates_chapter_arc_fields_end_to_end(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text(
        "=== HOOK === Раз.[pause]Два.\n=== FINAL === Конец.\n", encoding="utf-8")
    saved_argv = sys.argv
    sys.argv = ["speech_planner.py", str(tmp_path)]
    try:
        rc = sp.main()
    finally:
        sys.argv = saved_argv
    assert rc == 0
    import json
    with open(tmp_path / "media_plan" / "speech_plan.json", encoding="utf-8") as f:
        plan = json.load(f)
    assert all("arc_stage" in u for u in plan["units"])
    assert all("chapter_id" in u for u in plan["units"])


# ---------- homograph_hints_for_text (scripts/stress_placement.py надстройка,
# STRESS_HINTS_ENABLED=0 по умолчанию — см. докстринг в speech_planner.py) ----

def test_homograph_hints_off_by_default(monkeypatch):
    monkeypatch.setattr(sp, "STRESS_HINTS_ENABLED", False)
    assert sp.homograph_hints_for_text("Мука для хлеба стоила дорого.") == []


def test_homograph_hints_fail_open_on_import_error(monkeypatch):
    # Fail-open: если stress_placement/ruaccent недоступны или падают —
    # тихая пустая подсказка, планирование не должно падать из-за этого.
    monkeypatch.setattr(sp, "STRESS_HINTS_ENABLED", True)
    import builtins
    real_import = builtins.__import__

    def boom(name, *a, **kw):
        if name == "stress_placement":
            raise ImportError("simulated missing dependency")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", boom)
    assert sp.homograph_hints_for_text("Мука для хлеба стоила дорого.") == []


def test_homograph_hints_no_signal_returns_empty(monkeypatch):
    monkeypatch.setattr(sp, "STRESS_HINTS_ENABLED", True)
    # Нет известных омографов в тексте вообще.
    assert sp.homograph_hints_for_text("Солнце светило над полем весь день.") == []


class TestHomographHintsRealModel:
    """Реальный ruaccent (тяжёлая модель, ~30с загрузка один раз на класс,
    см. stress_placement.py) — та же категория тестов, что
    TestSentenceRelevanceSiglip2RealModel в test_visual_director.py."""

    def test_flour_sense_detected_with_context(self):
        sp.STRESS_HINTS_ENABLED = True
        try:
            hints = sp.homograph_hints_for_text(
                "Мешок муки стоял в углу амбара рядом с хлебом.")
        finally:
            sp.STRESS_HINTS_ENABLED = False
        assert any("мука" in h.lower() or "мешок" in h.lower() or "муки" in h for h in hints), hints
        assert any("flour" in h for h in hints), hints

    def test_suffering_sense_detected_with_context(self):
        sp.STRESS_HINTS_ENABLED = True
        try:
            hints = sp.homograph_hints_for_text(
                "Голод причинял людям страшные муки день за днём.")
        finally:
            sp.STRESS_HINTS_ENABLED = False
        assert any("suffering" in h for h in hints), hints

    def test_render_annotated_text_includes_homograph_hint(self):
        blocks = [_block("Мешок муки стоял в углу амбара рядом с хлебом.", section="BODY")]
        units = sp.build_units(blocks)
        sp.STRESS_HINTS_ENABLED = True
        try:
            text = sp.render_annotated_text(units)
        finally:
            sp.STRESS_HINTS_ENABLED = False
        assert "омограф" in text
