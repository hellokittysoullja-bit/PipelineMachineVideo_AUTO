#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Предмет фразы — отдельный простой вопрос проверки (М3, 25.09).

Живой случай judge12, слот 1 («Он весил меньше... грамм триста»):
утверждения спецификации составные («кинжал лежит на открытой ладони»,
«видна открытая ладонь»), и кадр с ПУСТОЙ ладонью выполнил второе —
прошёл «заменой» и встал на экран, хотя кинжала в нём нет. Симуляция
правила «брак, если не выполнено ни одно утверждение о предмете» на
составных утверждениях теряла 23-45 годных кадров из 215: кинжал без
доспеха не выполняет «кинжал в щели доспеха». Поэтому предмет спрашивается
отдельно и просто: «a dagger is visible»."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import shot_judge as sj  # noqa: E402
import stock_query_planner as sqp  # noqa: E402

SPEC = {"focus": "a small dagger in an open palm", "subject": "a small dagger",
        "claims": [{"id": "core", "text": "a small dagger lies flat in an open palm", "tier": "must"},
                   {"id": "c1", "text": "an open human palm is visible", "tier": "must"},
                   {"id": "c2", "text": "the shot is a close-up", "tier": "should"}]}


def _ans(core, c1, subject, c2="no"):
    return {"claims": {"core": core, "c1": c1, "c2": c2, "subject": subject}, "medium": "photo",
            "main_in_world": True, "background_foreign": False}


def test_subject_is_asked_last_and_simply():
    asked = sj.asked_claims(SPEC, "photo")
    assert asked[-1] == {"id": "subject", "text": "a small dagger is visible", "tier": "subject"}
    assert '"subject": "..."' in sj.claims_question("x", SPEC, kind="photo")


def test_empty_palm_is_rejected_although_the_palm_claim_is_met():
    assert sj.nothing_met(SPEC, _ans("no", "yes", "no"))


def test_dagger_without_the_palm_stays_a_substitute():
    assert not sj.nothing_met(SPEC, _ans("no", "no", "yes"))


def test_subject_answer_does_not_enter_the_ranking_vector():
    v1 = sj.claims_vector(SPEC, _ans("yes", "yes", "yes"))
    v2 = sj.claims_vector(SPEC, _ans("yes", "yes", "unsure"))
    assert v1 == v2


def test_spec_without_subject_asks_as_before():
    old = {k: v for k, v in SPEC.items() if k != "subject"}
    assert [c["id"] for c in sj.asked_claims(old, "photo")] == ["core", "c1", "c2"]
    assert not sj.nothing_met(old, {"claims": {"core": "no", "c1": "yes", "c2": "no"}})


def test_planner_reads_the_subject():
    packet = {"units": [{"n": 1, "text": "Вот кинжал."}]}
    raw = json.dumps({"n": 1, "focus": "a medieval dagger", "subject": "a dagger",
                      "core": "a medieval dagger is visible",
                      "claims": [{"id": "c1", "text": "the blade is narrow", "tier": "must"}],
                      "queries": [{"q": "rondel dagger", "for": ["core"]}]})
    assert sqp.parse_spec(raw, packet)[1]["subject"] == "a dagger"
    raw2 = raw.replace('"subject": "a dagger", ', '"subject": "", ')
    assert "subject" not in sqp.parse_spec(raw2, packet)[1]
