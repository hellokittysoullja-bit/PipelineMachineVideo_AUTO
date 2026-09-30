# -*- coding: utf-8 -*-
"""Уточнение запроса обязано срабатывать на РЕАЛЬНОМ словаре канала.

Первая линия защиты от анахронизмов (CLAUDE.md, ЧАСТЬ 14: «САМАЯ дешёвая и
эффективная линия») уточняет запрос к стоку ДО того, как контаминация
попадёт в пул. Замер 14.09 показал, что на словаре этого канала она почти
не срабатывала:

* `\\bsword\\b` не совпадает внутри «longsword»/«greatsword» — границы слова
  между частями составного нет;
* правило знало только «armor», а в 42 авторских запросах эпизода 02
  «armour» встречается 13 раз (второе по частоте слово после «medieval»),
  «armor» — ни одного.

Из 12 реальных запросов канала правило срабатывало на ОДНОМ. Цена измерена
на золотом наборе: 4 утечки из 9 (44%) дал запрос `greatsword warrior
fight`, который эта линия не трогала вообще, — и он приносил
`men-fighting-in-forest`, `traditional-martial-arts-duel`, то есть ровно тот
брак (non_european, modern_intrusion), против которого линия и заведена.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pipeline_smart as ps


@pytest.mark.parametrize("query", [
    "greatsword warrior fight",      # 4 утечки золотого набора
    "longsword duel",
    "broadsword blade",
    "swordsman stance",              # sword + s + man — суффиксы сочетаются
    "medieval knight armour fallen mud",
    "medieval armour museum display",
    "armoured knight horse",
    "knight armour plate",
])
def test_real_channel_queries_get_their_anchor(query):
    out = ps.disambiguate_search_query(query)
    assert out != query, f"запрос канала остался без якоря эпохи/культуры: {query}"
    assert "european" in out


@pytest.mark.parametrize("query", [
    "password manager screen",   # содержит «sword» хвостом составного
    "crossword puzzle",
    "shakespeare quill portrait",  # содержит «spear»
])
def test_traps_do_not_fire(query):
    """Составное сопоставление без списка ловушек однажды срабатывает не там —
    тот же приём, что уже держит словарь атмосферы от «зал» внутри «ЗАЛП»."""
    assert ps.disambiguate_search_query(query) == query


@pytest.mark.parametrize("query", [
    "katana sword",            # unless по культуре
    "japanese armor",
    "motorcycle helmet",
    "football helmet",
])
def test_unless_still_wins(query):
    """Расширение сопоставления не имеет права отменить исключения: уточнять
    «european» у запроса про катану — прямая ошибка."""
    assert ps.disambiguate_search_query(query) == query


def test_qualifier_never_duplicates_a_word_already_there():
    """Реальный баг первой версии функции: «european medieval medieval ...»."""
    out = ps.disambiguate_search_query("medieval armour close up")
    assert out.split().count("medieval") == 1
    assert out.startswith("european ")


def test_matching_logic_invalidates_the_candidate_cache():
    """Кэш-хит кандидата отдаёт файл БЕЗ повторной проверки гейтов, поэтому
    правка «как ищем термин» обязана менять подпись отбора — иначе на
    прогретом temp_smart/ она не дойдёт до экрана. Ровно этот класс уже
    стоил проекту анахронизма, пережившего свой собственный фикс."""
    ps._CANDIDATE_GATE_SIG = None
    before = ps.candidate_gate_signature()
    saved = ps.QUERY_TERM_TRAPS
    try:
        ps.QUERY_TERM_TRAPS = saved + ("longsword",)
        ps._CANDIDATE_GATE_SIG = None
        assert ps.candidate_gate_signature() != before
    finally:
        ps.QUERY_TERM_TRAPS = saved
        ps._CANDIDATE_GATE_SIG = None


def test_channel_vocabulary_is_actually_covered():
    """Словарь берётся из РЕАЛЬНОГО сценария канала, а не из головы: тест
    падает, если эпизод начнёт пользоваться словом, мимо которого правило
    молча проходит."""
    import re
    path = os.path.join(ROOT, "videos", "02_ne-mechom", "script.txt")
    if not os.path.exists(path):
        pytest.skip("сценарий эпизода недоступен")
    m = re.search(r"=== PEXELS QUERIES ===(.*?)(?:\n===|\Z)",
                  open(path, encoding="utf-8").read(), re.S)
    if not m:
        pytest.skip("в сценарии нет секции запросов")
    queries = []
    for line in m.group(1).strip().split("\n"):
        if ":" in line:
            queries += [q.strip().lower() for q in line.split(":", 1)[1].split(",") if q.strip()]
    assert queries
    # у запросов с доспехом/клинком якорь культуры обязан появиться
    armed = [q for q in queries if re.search(r"armou?r|sword|helmet|spear", q)]
    assert armed, "в эпизоде нет ни одного запроса с предметным термином"
    # Инвариант про РЕЗУЛЬТАТ, а не про факт правки: якорь культуры мог
    # написать сам автор («european longsword blade macro» в этом эпизоде),
    # и тогда функции верно нечего добавлять.
    missed = [q for q in armed
              if "european" not in ps.disambiguate_search_query(q)
              and not any(u in q for u in ("katana", "japanese", "asian", "chinese"))]
    assert not missed, f"запрос с предметным термином остался без якоря: {missed[:5]}"
