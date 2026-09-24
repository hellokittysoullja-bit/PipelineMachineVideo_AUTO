#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Гейт регрессий порядка пула: прод-код каскада на ВСЕХ снятых пулах.

Зачем: правка каскада 24.09 («лучшее сходство с запросами») прошла замер
на пулах одного прогона и дала живой регресс в трёх слотах judge12 — кадр
«ладонь с кинжалом» ушёл с 9-го места на 73-е, судья его не увидел. Этот
тест прогоняет формулу по пулам judge9/11/12 и падает, если в каком-то
пуле лучший размеченный кадр ушёл из первых 20 (их видит судья) или
годных там стало меньше в сумме. Метки — tests/fixtures/pool_regression/
labels.json, у каждой разметки имя оценщика; метки владельца старше.

Сменить формулу каскада можно, только переписав базовую линию
(pool_recall.py rankcheck --write-baseline) — то есть явно и с диффом
baseline.json в коммите."""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import pool_recall as pr  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "pool_regression")


def _load(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as f:
        return json.load(f)


def test_cascade_order_is_not_worse_on_any_stored_pool():
    if ps._gate_embed(text="a dagger") is None:
        pytest.skip("модель гейта недоступна")
    base = _load("baseline.json")
    now = pr.rank_metrics(pr.fixture_orders(FIX), _load("labels.json"), base["handoff"])
    assert pr.rankcheck_failures(now, base["metrics"]) == []


def test_best_frame_leaving_the_head_is_a_failure():
    base = {"r|1|photo": {"best": 2, "good": 3}}
    now = {"r|1|photo": {"best": 1, "good": 3}}
    assert pr.rankcheck_failures(now, base) == ["r|1|photo: лучший в первых — 1 вместо 2"]


def test_fewer_good_frames_in_total_is_a_failure():
    base = {"a": {"best": 1, "good": 3}, "b": {"best": 1, "good": 3}}
    now = {"a": {"best": 1, "good": 4}, "b": {"best": 1, "good": 1}}
    assert pr.rankcheck_failures(now, base) == ["годных в первых всего 5 вместо 6"]


def test_owner_label_outranks_claude():
    labels = {"claude": {"1|photo|7": 0}, "owner": {"1|photo|7": 2}}
    assert pr.merged_labels(labels)["1|photo|7"] == 2
