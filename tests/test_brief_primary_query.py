"""Главный запрос слота выводится из ЕГО СОБСТВЕННОГО брифа, а не из
запроса соседней фразы той же секции.

Замер, ради которого правка сделана (кофейный эпизод, 50 слотов, метки
глазами по Шагу 7.5): 11 браков из 14 — слот, чей `[shot:]` написан верно, а
судил его запрос секции, доставшийся ему раздачей. Авторских запросов на
секцию четыре, фраз восемь — подходящей строки для половины фраз в списке
нет физически, и любая раздача обязана промахнуться.
"""
import os
import sys

import pytest

sys.argv = ["pipeline_smart.py", "/tmp"]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pipeline_smart as ps  # noqa: E402


# Настоящие фразы, брифы и авторские запросы секции BLOCK 2 кофейного
# эпизода — синтетика не воспроизвела бы главного: запросов вчетверо
# меньше, чем фраз, и именно поэтому раздача обязана промахнуться.
BLOCKS = [
    {"section": "BLOCK 2: НАПИТОК", "is_subcut": False,
     "text": "Османский султан Мурад Четвёртый запретил кофе под страхом смерти.",
     "shot_brief": "an ottoman sultan on a throne, historical illustration"},
    {"section": "BLOCK 2: НАПИТОК", "is_subcut": False,
     "text": "Позже римские священники требовали, чтобы Папа объявил кофе напитком дьявола.",
     "shot_brief": "a papal seal on an old document, close up"},
    {"section": "BLOCK 2: НАПИТОК", "is_subcut": False,
     "text": "то прямо запрещали, опасаясь, что люди перестанут пить пиво и вино.",
     "shot_brief": "a rainy vienna street with an old coffeehouse sign, historical"},
]
AUTHORED = {"BLOCK2": ["ottoman sultan throne illustration", "royal decree document",
                       "historical coffeehouse crowd", "vienna coffeehouse rain"]}


def _resolve(monkeypatch, enabled):
    monkeypatch.setenv("BRIEF_PRIMARY_QUERY", "1" if enabled else "0")
    return ps.resolve_queries([dict(b) for b in BLOCKS], authored_queries=AUTHORED)


def test_slot_query_comes_from_its_own_brief(monkeypatch):
    qs = _resolve(monkeypatch, True)
    # Каждый слот спрашивает про то, что написано в ЕГО брифе.
    assert "papal" in qs[1], qs[1]
    assert "vienna" in qs[2], qs[2]
    assert "sultan" in qs[0] or "ottoman" in qs[0], qs[0]


def test_disabled_flag_keeps_section_query(monkeypatch):
    """Откат обязан быть настоящим: с флагом 0 запрос слота снова из секции."""
    qs = _resolve(monkeypatch, False)
    assert all(q in AUTHORED["BLOCK2"] for q in qs), qs


def test_block_without_brief_is_untouched(monkeypatch):
    """Нет брифа — прежнее поведение байт-в-байт, слот берёт запрос секции."""
    blocks = [dict(b) for b in BLOCKS]
    blocks[1]["shot_brief"] = None
    monkeypatch.setenv("BRIEF_PRIMARY_QUERY", "1")
    qs = ps.resolve_queries(blocks, authored_queries=AUTHORED)
    assert qs[1] in AUTHORED["BLOCK2"], qs[1]
    assert "vienna" in qs[2], qs[2]


def test_query_is_short_like_a_section_query(monkeypatch):
    """Судья остаётся на коротком запросе: калибровка CLIP_RELEVANCE_THRESHOLD
    снята на запросах в несколько слов, и подмена их брифом ЦЕЛИКОМ сдвинула
    бы распределение скоров. В пул уходит стоковый перевод брифа, не бриф."""
    qs = _resolve(monkeypatch, True)
    for q in qs:
        assert len(q.split()) <= ps.BRIEF_STOCK_QUERY_MAX_WORDS + 1, q
