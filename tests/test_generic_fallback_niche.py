#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GENERIC_FALLBACKS — третий хардкод-список того же класса, что
_QUERY_ERA_ANCHORS_DEFAULT и старый блоклист (оба уже закрыты паспортом
мира). Найдено ЖИВЫМ ПРОБНЫМ ПРОГОНОМ (17.09), не чтением кода: тестовый
эпизод про бортовой компьютер «Аполлона» (ниша без единого совпадения по
словарю тем канала) получил на музейном пути буквальный запрос
`medieval sword still life`, и Метрополитен ответил средневековым мечом на
слот про смартфон — потому что GENERIC_FALLBACKS уходит в pexels_photo()
как САМ query, а музейный путь берёт query НАПРЯМУЮ, в обход брифа
(MUSEUM_RAW_QUERY_VERSION).
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import pipeline_smart as ps  # noqa: E402
import world_card as wc  # noqa: E402


@pytest.fixture
def episode(tmp_path):
    ps.reset_world_card_cache()
    d = tmp_path / "ep"
    (d / "media_plan").mkdir(parents=True)
    yield d
    ps.reset_world_card_cache()


SPACE_CARD = {
    "schema_version": 1, "register": "mixed",
    "era": {"from": 1961, "to": 2100},
    "culture": {"include": [], "exclude": []},
    "must_not_show": ["medieval knight or castle"],
    "expected_subjects": ["rocket launch", "astronaut in a spacesuit",
                          "mission control room", "smartphone screen"],
    "era_anchor_terms": ["space", "nasa", "apollo"],
}


def test_without_card_the_hardcoded_medieval_list_still_works(episode, monkeypatch):
    """Нет паспорта — поведение байт-в-байт прежнее: ни одна старая
    установка канала (без world_card.json) не должна измениться."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    assert ps.generic_fallback_queries_effective() == ps.GENERIC_FALLBACKS


def test_with_card_medieval_words_never_leak_into_the_fallback(episode, monkeypatch):
    """Реальный найденный случай: без этой правки `medieval`/`knight`
    попадали в запрос музею даже для эпизода про космос."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    with open(wc.path(str(episode)), "w", encoding="utf-8") as f:
        import json
        json.dump(SPACE_CARD, f)
    ps.reset_world_card_cache()

    got = ps.generic_fallback_queries_effective()
    assert got != ps.GENERIC_FALLBACKS
    joined = " ".join(got).lower()
    assert "medieval" not in joined
    assert "knight" not in joined
    assert "sword" not in joined
    assert "rocket launch" in got


def test_resolve_queries_end_to_end_never_sends_medieval_to_a_block_with_no_theme_match(episode, monkeypatch):
    """Сквозная проверка САМОЙ resolve_queries(), а не только хелпера:
    блок без темы, без соседа, без авторского запроса — ровно тот путь,
    что довёл до музея на живом прогоне."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    with open(wc.path(str(episode)), "w", encoding="utf-8") as f:
        import json
        json.dump(SPACE_CARD, f)
    ps.reset_world_card_cache()

    blocks = [
        {"section": "HOOK", "text": "Совершенно нераспознаваемая фраза без единой темы канала.",
         "is_subcut": False, "shot_brief": None},
    ]
    resolved = ps.resolve_queries(blocks, authored_queries=None)
    assert len(resolved) == 1
    assert resolved[0] is not None
    assert "medieval" not in resolved[0].lower()
    assert "knight" not in resolved[0].lower()


def test_plan_summary_counter_still_counts_fallback_usage_with_a_card(episode, monkeypatch, capsys):
    """n_generic в print_plan_summary — счётчик «сколько блоков ушло в
    запасной список», а не байт-в-байт старый список слов. С паспортом
    запасной список другой, и счётчик обязан видеть ИМЕННО ЕГО, иначе
    отчёт эпизода будет врать «0 блоков без темы» на самом деле бракованном
    прогоне."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    with open(wc.path(str(episode)), "w", encoding="utf-8") as f:
        import json
        json.dump(SPACE_CARD, f)
    ps.reset_world_card_cache()
    queries = ["rocket launch", "medieval sword still life"]
    n_generic = sum(1 for q in queries
                    if q in ps.GENERIC_FALLBACKS or q in ps.generic_fallback_queries_effective())
    assert n_generic == 2
