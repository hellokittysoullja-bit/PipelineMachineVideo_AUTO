#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Предмет фразы спасает замену (26.09).

Утверждения спецификации составные («рыцарь падает в грязь»), и годная
замена — рыцарь, который стоит, — не выполняет ни одного обязательного
пункта. Правило «ничего не выполнено — брак» опустошало слот, хотя
замена была среди первых кадров: на эп.94 в 3-4 слотах из 9. Предмет
спрашивается отдельно и просто («a knight is visible»); виден — кадр
замена, а не брак. Замер по экрану (4 прогона эп.94): брак на экране 8
-> 8, лучший кадр 12 -> 22, пустых слотов 13 -> 3.

Обратная половина («предмет не виден — брак») НЕ действует: замер
24-26.09 — она выбрасывает годные кадры другого вида (дага на «рондельный
кинжал»)."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import shot_judge as sj  # noqa: E402
import stock_query_planner as sqp  # noqa: E402

SPEC = {"focus": "a knight falls into the mud", "subject": "a knight",
        "claims": [{"id": "core", "text": "a knight in armour falls face down into mud", "tier": "must"},
                   {"id": "c1", "text": "mud splashes around him", "tier": "must"},
                   {"id": "c2", "text": "the shot is a close-up", "tier": "should"}]}


def _ans(core, c1, subject, c2="no"):
    return {"claims": {"core": core, "c1": c1, "c2": c2, "subject": subject}, "medium": "photo",
            "main_in_world": True, "background_foreign": False}


def test_subject_is_asked_last_and_simply():
    asked = sj.asked_claims(SPEC, "photo")
    assert asked[-1] == {"id": "subject", "text": "a knight is visible", "tier": "subject"}
    assert '"subject": "..."' in sj.claims_question("x", SPEC, kind="photo")


def test_standing_knight_is_a_substitute_not_brak():
    assert not sj.nothing_met(SPEC, _ans("no", "no", "yes"))
    assert not sj.shows_nothing(SPEC, _ans("no", "no", "yes"), grid=2)


def test_subject_not_seen_does_not_reject_by_itself():
    """Обратная половина снята замером: решают обязательные пункты."""
    assert not sj.nothing_met(SPEC, _ans("no", "yes", "no"))
    assert sj.nothing_met(SPEC, _ans("no", "no", "no"))
    assert sj.nothing_met(SPEC, _ans("no", "no", "unsure"))


def test_grid_zero_is_still_brak_even_with_the_subject_seen():
    assert sj.shows_nothing(SPEC, _ans("no", "no", "yes"), grid=0)


def test_subject_answer_does_not_enter_the_ranking_vector():
    assert sj.claims_vector(SPEC, _ans("yes", "yes", "yes")) == \
        sj.claims_vector(SPEC, _ans("yes", "yes", "unsure"))


def test_spec_without_subject_behaves_as_before():
    old = {k: v for k, v in SPEC.items() if k != "subject"}
    assert [c["id"] for c in sj.asked_claims(old, "photo")] == ["core", "c1", "c2"]
    assert sj.nothing_met(old, {"claims": {"core": "no", "c1": "no", "c2": "yes"}})


def test_planner_reads_and_keeps_the_subject(tmp_path):
    packet = {"units": [{"n": 1, "text": "Вот кинжал."}]}
    raw = json.dumps({"n": 1, "focus": "a medieval dagger", "subject": "a dagger",
                      "core": "a medieval dagger is visible",
                      "claims": [{"id": "c1", "text": "the blade is narrow", "tier": "must"}],
                      "queries": [{"q": "rondel dagger", "for": ["core"]}]})
    assert sqp.parse_spec(raw, packet)[1]["subject"] == "a dagger"
    assert "subject" not in sqp.parse_spec(raw.replace('"subject": "a dagger", ', '"subject": "", '),
                                           packet)[1]


def test_plan_on_disk_keeps_the_subject_for_the_render(tmp_path):
    (tmp_path / "media_plan").mkdir()
    (tmp_path / "media_plan" / sqp.PLAN_NAME).write_text(json.dumps({
        "version": sqp.PLAN_VERSION,
        "units": {"k": {"text": "Вот кинжал.", "focus": "a dagger", "subject": "a dagger",
                        "claims": [{"id": "core", "text": "a dagger is visible", "tier": "must"}]}}}),
        encoding="utf-8")
    assert sqp.load_specs(str(tmp_path))["k"]["subject"] == "a dagger"


def test_look_question_carries_the_film_look_for_photos_only():
    base = sj.look_question("photo", 3)
    styled = sj.look_question("photo", 3, "bright pictures for children")
    assert base == sj.LOOK_PROMPT.format(k=3)
    assert "shown as: bright pictures for children;" in styled and styled != base
    assert sj.look_question("video", 3, "bright pictures for children") == sj.LOOK_PROMPT_VIDEO.format(k=3)
