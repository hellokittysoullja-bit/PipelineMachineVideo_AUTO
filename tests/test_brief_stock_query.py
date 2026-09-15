# -*- coding: utf-8 -*-
"""Бриф -> короткий запрос для СТОКА.

Полка принимает описание целиком; у стоков текстовый API, где каждое лишнее
слово сужает выдачу. Перевод между этими двумя режимами — то место, где
брифы впервые начинают влиять на слоты, у которых нет музейного пути.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pipeline_smart as ps  # noqa: E402


def _q(brief, fallback="medieval knight"):
    return ps.brief_to_stock_query(brief, fallback=fallback)


def test_no_brief_is_byte_identical_to_today():
    """Нет брифа — слот обязан получить ровно то, что получал раньше."""
    assert _q(None) == "medieval knight"
    assert _q("") == "medieval knight"
    assert _q("   ") == "medieval knight"


def test_brief_of_pure_framing_words_falls_back():
    """Бриф из одних слов ракурса не несёт предмета — откат, а не мусор."""
    assert _q("close up, seen from the front, whole figure") == "medieval knight"


def test_era_anchor_survives_the_word_cap():
    """Реальный найденный промах: единственный якорь эпохи стоял шестым
    словом и обрезался, оставляя запрос без эпохи вообще. Одиночное
    `plate armour` первым результатом даёт танк — см. каскад."""
    q = _q("a manuscript illumination of a battle between armoured knights")
    era = {a.lower() for a in ps.OPENVERSE_ERA_ANCHORS}
    assert any(w in era for w in q.split()), q


def test_author_word_order_is_preserved():
    """Перестановка ломает термин: `rondel dagger` — название предмета,
    `dagger rondel` — нет. Измерено на реальных брифах эпизода."""
    q = _q("a rondel dagger with a narrow stiff blade and a disc guard")
    assert "rondel dagger" in q


def test_subject_of_a_scene_is_not_stripped():
    """Словарь ракурса НЕ ТОТ ЖЕ, что OPENVERSE_QUERY_MODIFIERS: тот режет
    `mud`/`field`/`battlefield`, потому что расширяет архивный запрос. Для
    брифа сцены это и есть предмет кадра."""
    q = _q("thick wet clay mud with deep boot prints")
    assert "mud" in q
    q2 = _q("an open muddy field seen low from the ground")
    assert "field" in q2 or "muddy" in q2


def test_the_two_vocabularies_are_intentionally_different():
    """Если списки когда-нибудь сольют в один, сцены начнут терять предмет."""
    framing = {w.lower() for w in ps.BRIEF_FRAMING_WORDS}
    modifiers = {w.lower() for w in ps.OPENVERSE_QUERY_MODIFIERS}
    assert "mud" in modifiers and "mud" not in framing
    assert "battlefield" in modifiers and "battlefield" not in framing


def test_query_stays_short_enough_for_a_text_api():
    q = _q("a complete articulated suit of medieval plate armour standing, "
           "whole figure, frontal, dark background, polished steel")
    assert ps.OPENVERSE_QUERY_MIN_WORDS <= len(q.split()) <= ps.BRIEF_STOCK_QUERY_MAX_WORDS


def test_brief_query_is_added_not_substituted():
    """Additive по построению: авторский запрос и запросы секции остаются
    в пуле, бриф только добавляет свой."""
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    start = src.index("pool_queries = [query]")
    block = src[start:start + 1200]
    assert "_bq = brief_to_stock_query(shot_brief" in block
    assert "pool_queries = [_bq] + pool_queries" in block


def test_brief_is_in_the_candidate_cache_key():
    """Бриф меняет состав пула — без него прогретый temp_smart/ отдал бы
    кандидата, выбранного до появления брифа."""
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    start = src.index("_brief_key = brief_to_stock_query")
    assert "[_brief_key] if _brief_key else []" in src[start:start + 600]


def test_change_is_in_the_selection_signature():
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    start = src.index("def _selection_stack_signature")
    block = src[start:src.index("\ndef candidate_gate_signature", start)]
    assert "BRIEF_STOCK_QUERY_VERSION" in block
    assert isinstance(ps.BRIEF_STOCK_QUERY_VERSION, int)


@pytest.mark.parametrize("brief,must", [
    ("an armoured foot in a steel sabaton standing on bare earth", "sabaton"),
    ("a war hammer head of blunt steel", "hammer"),
    ("a human skull from an archaeological excavation", "skull"),
    ("a medieval military camp of tents", "camp"),
])
def test_the_subject_of_the_shot_reaches_the_query(brief, must):
    assert must in _q(brief)
