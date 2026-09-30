"""Слияние формулировок слота по рангам (RRF) — scripts/query_fusion.py.

Проверяется не «функция что-то возвращает», а те три свойства, ради которых
она заведена, и тот случай, который она обязана лечить: кандидат, найденный
НЕСКОЛЬКИМИ формулировками слота, поднимается над найденным одной, и при этом
одна широкая формулировка не может в одиночку перебить точную.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import query_fusion as qf   # noqa: E402


def ids(cands):
    return [c["id"] for c in cands]


def cand(i):
    return {"id": i}


def key(c):
    return c.get("id")


def test_single_list_preserves_its_order():
    """Одна формулировка -> слияние обязано быть тождественным. Иначе откат
    «как было» переставал бы существовать даже при одном варианте."""
    lst = [cand("a"), cand("b"), cand("c")]
    assert ids(qf.fuse([lst], key)) == ["a", "b", "c"]


def test_consensus_beats_single_list_lead():
    """Ровно тот случай, из-за которого модуль написан: кандидат, стоящий у
    точной формулировки НЕ первым, но найденный и второй формулировкой,
    обгоняет кандидата, которого нашла только точная (арбалетный ворот на
    'european longsword blade macro')."""
    exact = [cand("spanner"), cand("sword1"), cand("x"), cand("y"), cand("sword2")]
    broad = [cand("sword2"), cand("sword1"), cand("z")]
    out = ids(qf.fuse([exact, broad], key))
    assert out.index("sword1") < out.index("spanner")
    assert out.index("sword2") < out.index("spanner")


def test_exact_variant_wins_ties():
    """Точная формулировка сохраняет первенство ПРИ РАВНОМ РАНГЕ — за счёт
    порядка первого появления, а не за счёт убывающего веса.

    Вес был понижающим в первой версии и ЗАМЕР это опроверг (см. коммент к
    DECAY): пересечение списков двух формулировок на реальном слоте — один
    кандидат из 38, и при убывающем весе слияние вырождалось в прежний
    каскад. Первенство точной формулировки при этом обязано сохраниться, и
    проверяется здесь отдельно от веса."""
    exact = [cand("exact_top")]
    broad = [cand("broad_top")]
    assert ids(qf.fuse([exact, broad], key))[0] == "exact_top"


def test_broad_variant_reaches_the_probe_window():
    """Главное, ради чего слияние вообще делается: кандидаты более широкой
    формулировки обязаны попадать В НАЧАЛО общего списка, а не за хвост
    точной. Оценивается лишь первые BASE_MIN_POOL кандидатов — список,
    идущий подряд, физически вытесняет вторую формулировку из оценки.

    Замерено на слоте «european longsword blade macro»: выдача Мет (реальное
    оружие) при подряд-склейке не попадала в пул вообще, при слиянии встаёт
    на позиции 1, 4, 6, 8."""
    exact = [cand(f"junk{i}") for i in range(24)]
    broad = [cand(f"real{i}") for i in range(15)]
    out = ids(qf.fuse([exact, broad], key))
    assert out.index("real0") <= 2, out[:6]
    assert len([x for x in out[:8] if x.startswith("real")]) >= 3


def test_weights_are_equal_by_default_and_measured():
    """Равный вес — не «забыли настроить», а результат замера. Молчаливый
    возврат к понижающему весу снова превратил бы слияние в прежний каскад."""
    assert qf.DECAY == 1.0
    assert qf.variant_weights(3) == [1.0, 1.0, 1.0]


def test_ties_break_by_first_appearance_deterministically():
    """Детерминизм обязателен: этот порядок решает, какие кандидаты вообще
    дойдут до оценки, а ключ кэша кандидата от него не зависит — недетерминизм
    давал бы разный ролик на одном и том же входе."""
    a = [cand("p"), cand("q")]
    b = [cand("q"), cand("p")]
    first = ids(qf.fuse([a, b], key, decay=1.0))
    for _ in range(5):
        assert ids(qf.fuse([a, b], key, decay=1.0)) == first
    assert first == ["p", "q"]


def test_duplicates_across_variants_appear_once():
    a = [cand("x"), cand("y")]
    b = [cand("x"), cand("z")]
    out = ids(qf.fuse([a, b], key))
    assert sorted(out) == ["x", "y", "z"]


def test_empty_and_none_keys_are_skipped_not_crashing():
    """Fail-open той же дисциплины, что и у источников: кандидат без id не
    должен ронять сборку пула."""
    a = [{"id": None}, cand("ok")]
    assert ids(qf.fuse([a, []], key)) == ["ok"]
    assert qf.fuse([], key) == []


def test_no_variant_returns_nothing_new():
    """Слияние не выдумывает кандидатов: множество результатов равно
    объединению входов. Заявление «пул не растёт, меняется только порядок»
    должно быть проверяемым, а не декларацией в докстринге."""
    a = [cand("1"), cand("2")]
    b = [cand("2"), cand("3")]
    assert set(ids(qf.fuse([a, b], key))) == {"1", "2", "3"}


def test_max_variants_cap_is_small_enough_to_bound_api_cost():
    """Потолок формулировок — прямая цена в запросах к API (каскад её
    экономил). Его молчаливый рост означал бы, что слот тихо стал стоить
    вдвое больше запросов у источника с квотой."""
    assert qf.MAX_VARIANTS <= 3


@pytest.mark.parametrize("k", [qf.RRF_K])
def test_rrf_k_keeps_rank_meaningful_within_a_list(k):
    """k из статьи (60) рассчитан на выдачи в сотни позиций; здесь список
    источника — десятки кандидатов, и при слишком большом k разница между
    1-й и 20-й позицией исчезает, то есть ранг перестаёт что-либо значить."""
    assert (1.0 / (k + 1)) / (1.0 / (k + 20)) >= 1.5
