#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Паспорт мира эпизода: отсутствие — тишина, поломка — крик.

Оба инварианта проверены контрольным прогоном со снятой правкой (тест
падает), иначе они были бы зелёными по построению.
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import world_card as wc  # noqa: E402


def _historical():
    return {
        "schema_version": 1,
        "register": "historical",
        "era": {"from": 1380, "to": 1450},
        "culture": {"include": ["western european"], "exclude": ["japanese"]},
        "must_not_show": ["modern people", "cars"],
        "expected_subjects": ["plate armour"],
        "era_anchor_terms": ["medieval", "15th century"],
    }


def _psychology():
    return {
        "schema_version": 1,
        "register": "modern",
        "era": None,
        "culture": {"include": [], "exclude": []},
        "must_not_show": ["plate armour", "castle", "medieval manuscript"],
        "expected_subjects": ["untouched food", "phone face down"],
        "era_anchor_terms": ["contemporary", "everyday"],
    }


def _prehistoric():
    return {
        "schema_version": 1,
        "register": "historical",
        "era": {"from": -300000, "to": -10000},
        "culture": {"include": ["neanderthal", "paleolithic europe"],
                    "exclude": ["modern european"]},
        "must_not_show": ["metal tools", "woven cloth", "masonry", "pottery"],
        "expected_subjects": ["flint scraper", "cave wall"],
        "era_anchor_terms": ["paleolithic", "prehistoric", "stone age"],
    }


@pytest.mark.parametrize("card", [_historical(), _psychology(), _prehistoric()])
def test_three_different_niches_are_all_valid(card):
    """Паспорт не про средневековье: зашитый в код словарь этого канала
    (_QUERY_ERA_ANCHORS_DEFAULT в pipeline_smart) неверен для двух из трёх
    этих эпизодов, и ровно поэтому мир обязан приходить из эпизода."""
    assert wc.validate(card) == []


def test_missing_card_is_silence_not_error(tmp_path):
    """Нет файла — потребители работают как раньше, байт-в-байт. Эпизоды,
    написанные до этой правки, не должны ничего заметить."""
    assert wc.load(str(tmp_path)) is None


def test_broken_card_is_loud(tmp_path):
    """Файл есть и сломан — это НЕ «мира нет». Тихо продолжить значило бы
    молча выключить приёмку кадра: тот самый класс no-op, которым этот
    репозиторий горел шесть раз (CLAUDE.md)."""
    d = tmp_path / "media_plan"
    d.mkdir()
    (d / wc.CARD_NAME).write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(wc.WorldCardError):
        wc.load(str(tmp_path))


def test_unparsable_json_is_loud(tmp_path):
    d = tmp_path / "media_plan"
    d.mkdir()
    (d / wc.CARD_NAME).write_text("{не json", encoding="utf-8")
    with pytest.raises(wc.WorldCardError):
        wc.load(str(tmp_path))


def test_historical_without_era_is_rejected():
    c = _historical()
    c["era"] = None
    assert any("окно эпохи" in p for p in wc.validate(c))


def test_modern_without_era_is_fine():
    """У психологии окна эпохи нет, и требовать его значило бы заставлять
    автора выдумывать дату."""
    assert wc.validate(_psychology()) == []


def test_era_must_be_signed_integers_not_prose():
    c = _historical()
    c["era"] = {"from": "XV век", "to": "XV век"}
    probs = wc.validate(c)
    assert any("целое число" in p for p in probs)


def test_era_bounds_and_order():
    c = _historical()
    c["era"] = {"from": 1500, "to": 1400}
    assert any("больше" in p for p in wc.validate(c))
    c["era"] = {"from": 1400, "to": 9999}
    assert any("вне диапазона" in p for p in wc.validate(c))


def test_bce_window_survives_roundtrip(tmp_path):
    p = wc.save(str(tmp_path), _prehistoric(), derived_by="test")
    assert os.path.exists(p)
    got = wc.load(str(tmp_path))
    assert wc.era_window(got) == (-300000, -10000)
    assert "до н.э." in wc.describe(got)


def test_empty_anchors_rejected_for_historical():
    """Запрос без якоря мира — измеренная причина 4 браков из одного
    запроса (`greatsword warrior fight`, золотой набор эпизода 01)."""
    c = _historical()
    c["era_anchor_terms"] = []
    assert any("era_anchor_terms" in p for p in wc.validate(c))


def test_anchors_from_card_beat_channel_fallback():
    card = _psychology()
    got = wc.era_anchors(card, fallback=("medieval", "knight"))
    assert got == ("contemporary", "everyday")
    assert "medieval" not in got


def test_anchors_fall_back_when_no_card():
    assert wc.era_anchors(None, fallback=("Medieval", "Knight")) == ("medieval", "knight")


def test_modern_props_field_survives_roundtrip(tmp_path):
    """Слот «вспомни, сколько весит пакет молока» ТРЕБУЕТ современный пакет
    молока на средневековом канале — запрет современности на весь эпизод
    убил бы законный кадр (золотой набор, ep01_000/ep01_005). Поле в
    паспорте есть и переживает запись; ЧИТАТЕЛЯ у него пока нет
    сознательно — единственный законный читатель это приёмка кадра, и
    accessor появится вместе с ней, а не заранее.
    """
    card = dict(_historical(), modern_props_allowed=True)
    wc.save(str(tmp_path), card, derived_by="test")
    assert wc.load(str(tmp_path))["modern_props_allowed"] is True


def test_save_refuses_broken_card(tmp_path):
    """Пусть сломанный паспорт не существует вовсе, чем существует и роняет
    каждый следующий прогон."""
    with pytest.raises(wc.WorldCardError):
        wc.save(str(tmp_path), {"schema_version": 1}, derived_by="test")
    assert not os.path.exists(wc.path(str(tmp_path)))


def test_terms_are_normalised_and_deduped_keeping_author_order():
    c = _historical()
    c["era_anchor_terms"] = ["Medieval", "medieval", " 15th Century ", ""]
    assert wc.era_anchors(c) == ("medieval", "15th century")


def test_parse_answer_tolerates_fences_and_prose():
    card = _historical()
    wrapped = "Вот паспорт:\n```json\n" + json.dumps(card) + "\n```\nготово"
    assert wc.parse_answer(wrapped)["register"] == "historical"
    with pytest.raises(wc.WorldCardError):
        wc.parse_answer("никакого json здесь нет")


def test_module_holds_no_second_copy_of_term_matching():
    """Сопоставление термина с запросом живёт в одном месте
    (pipeline_smart.query_mentions_term: британские написания, составные
    слова, ловушки вроде `password` внутри `sword`). Вторая копия этого
    правила однажды уже стоила эпизоду PHRASE LOCK — здесь её нет, и тест
    сторожит именно это, а не наличие строк."""
    src = open(os.path.join(SCRIPTS_DIR, "world_card.py"), encoding="utf-8").read()
    for forbidden in ("QUERY_TERM_SPELLINGS", "QUERY_TERM_TRAPS",
                      "QUERY_TERM_SUFFIXES", "re.search", "\\b"):
        assert forbidden not in src, (
            f"в world_card.py появилось {forbidden!r} — сопоставление слов "
            f"должно остаться в pipeline_smart.query_mentions_term")


def test_prompt_carries_the_whole_script_and_asks_for_signed_years():
    """Вопрос мозгу обязан объяснить формат года: без этого модель отвечает
    «XV век», а строку нельзя сравнить с паспортом музейного предмета."""
    p = wc.prompt_for_script("=== HOOK === Текст эпизода.", niche="психология")
    assert "Текст эпизода." in p
    assert "-40000" in p
    assert "психология" in p


def test_describe_never_lies_about_absent_card():
    assert "не задан" in wc.describe(None)


def test_judge_setting_is_one_line_from_the_card_only():
    card = {"register": "historical", "era": {"from": -2600, "to": -30},
            "culture": {"include": ["egyptian"], "exclude": ["roman"]},
            "must_not_show": ["modern tourist"]}
    assert wc.judge_setting(card) == "historical, 2600 BC-30 BC, egyptian"
    assert wc.judge_setting({"register": "abstract"}) == "abstract"
    assert wc.judge_setting(None) is None and wc.judge_setting({}) is None
