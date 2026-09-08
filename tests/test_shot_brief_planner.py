"""Локальный режиссёр кадра (Контур A) — валидация и разбор ответа модели.

Тестируется НЕ качество модели (это меряет
scripts/shot_brief_planner.py --benchmark на реальных трудных случаях), а
БАРЬЕР между моделью и пайплайном: `validate_brief()` и `_extract_json()`.

Почему барьер важнее самой модели. Локальная 3B-модель ошибается — это
данность, а не дефект. Цена ошибки несимметрична: отсутствующий бриф
просто оставляет прежнее поведение (рендер идёт как раньше), а
ПРОПУЩЕННЫЙ битый бриф отправляет в сток мусорный запрос или подсовывает
в вето негатив, который забракует правильные кадры. Поэтому валидатор
обязан быть строгим до паранойи и отклонять при малейшем сомнении.

Все тесты — чистые функции, без модели и без сети.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import shot_brief_planner as sbp  # noqa: E402


def _ok_raw(**over):
    """Валидный ответ модели; отдельные поля переопределяются в тесте."""
    base = {
        "reading": "figurative",
        "why": "фраза про миф о весе, а не про предмет",
        "subject": "medieval longsword on a wooden table",
        "setting": "dim museum hall, side light",
        "era_from": 1000,
        "era_to": 1500,
        "queries_en": ["medieval longsword close up", "knight sword museum"],
        "must_not_contain": ["kitchen scales", "modern gym"],
        "confidence": 0.8,
    }
    base.update(over)
    return base


class TestValidBrief:
    def test_clean_brief_passes(self):
        brief, err = sbp.validate_brief(_ok_raw())
        assert err is None and brief is not None
        assert brief["reading"] == "figurative"
        assert brief["queries_en"] == ["medieval longsword close up", "knight sword museum"]
        assert brief["confidence"] == 0.8

    def test_era_reversed_is_swapped_not_rejected(self):
        """Модель путает местами from/to — это не повод терять слот целиком."""
        brief, err = sbp.validate_brief(_ok_raw(era_from=1500, era_to=1000))
        assert err is None
        assert brief["era_from"] == 1000 and brief["era_to"] == 1500

    def test_confidence_is_clamped(self):
        brief, _ = sbp.validate_brief(_ok_raw(confidence=42))
        assert brief["confidence"] == 1.0
        brief, _ = sbp.validate_brief(_ok_raw(confidence=-5))
        assert brief["confidence"] == 0.0

    def test_missing_confidence_is_none_not_a_failure(self):
        brief, err = sbp.validate_brief(_ok_raw(confidence="не число"))
        assert err is None and brief["confidence"] is None


class TestRejectsBadOutput:
    """Всё, что ниже, обязано ОТКЛОНЯТЬСЯ: слот тогда остаётся на прежнем
    поведении, а не получает поле, которому нельзя доверять."""

    def test_not_a_dict(self):
        assert sbp.validate_brief(["список"])[0] is None
        assert sbp.validate_brief(None)[0] is None
        assert sbp.validate_brief("строка")[0] is None

    def test_unknown_reading(self):
        assert sbp.validate_brief(_ok_raw(reading="metaphorical"))[0] is None
        assert sbp.validate_brief(_ok_raw(reading=None))[0] is None

    def test_queries_not_a_list(self):
        assert sbp.validate_brief(_ok_raw(queries_en="medieval sword"))[0] is None

    def test_all_queries_cyrillic_is_rejected(self):
        """Русский запрос стоковые архивы не понимают — молча дал бы пустую
        выдачу, то есть слот тихо остался бы без кандидатов."""
        brief, err = sbp.validate_brief(_ok_raw(queries_en=["меч крупным планом", "рыцарь"]))
        assert brief is None
        assert "английск" in err

    def test_cyrillic_query_is_dropped_but_english_ones_survive(self):
        brief, err = sbp.validate_brief(
            _ok_raw(queries_en=["меч крупным планом", "medieval sword macro"]))
        assert err is None
        assert brief["queries_en"] == ["medieval sword macro"]

    def test_subject_in_cyrillic_is_rejected(self):
        assert sbp.validate_brief(_ok_raw(subject="средневековый меч"))[0] is None

    def test_empty_subject_is_rejected(self):
        assert sbp.validate_brief(_ok_raw(subject="   "))[0] is None
        assert sbp.validate_brief(_ok_raw(subject=None))[0] is None

    def test_overlong_query_is_dropped(self):
        long_q = "a " * 100
        brief, err = sbp.validate_brief(_ok_raw(queries_en=[long_q, "knight armor"]))
        assert err is None and brief["queries_en"] == ["knight armor"]

    def test_no_valid_queries_left_is_rejected(self):
        assert sbp.validate_brief(_ok_raw(queries_en=[]))[0] is None
        assert sbp.validate_brief(_ok_raw(queries_en=["", "   "]))[0] is None


class TestSanitizing:
    def test_duplicate_queries_are_deduped_preserving_order(self):
        brief, _ = sbp.validate_brief(_ok_raw(queries_en=[
            "medieval sword", "MEDIEVAL SWORD", "knight armor", "medieval sword"]))
        assert brief["queries_en"] == ["medieval sword", "knight armor"]

    def test_queries_are_capped(self):
        many = [f"query number {i}" for i in range(20)]
        brief, _ = sbp.validate_brief(_ok_raw(queries_en=many))
        assert len(brief["queries_en"]) == sbp.MAX_QUERIES

    def test_negatives_are_capped_and_cleaned(self):
        many = [f"bad thing {i}" for i in range(30)] + ["плохая вещь", "", "ok thing"]
        brief, _ = sbp.validate_brief(_ok_raw(must_not_contain=many))
        assert len(brief["must_not_contain"]) <= sbp.MAX_NEGATIVES
        assert all(not any("а" <= ch <= "я" for ch in n.lower())
                    for n in brief["must_not_contain"])

    def test_absurd_era_becomes_none_not_a_rejection(self):
        """Год 99999 — галлюцинация, но остальной бриф может быть полезен."""
        brief, err = sbp.validate_brief(_ok_raw(era_from=99999, era_to=100000))
        assert err is None
        assert brief["era_from"] is None and brief["era_to"] is None

    def test_whitespace_is_normalized(self):
        brief, _ = sbp.validate_brief(_ok_raw(
            queries_en=["  medieval    sword\n  macro "], subject=" a  sword "))
        assert brief["queries_en"] == ["medieval sword macro"]
        assert brief["subject"] == "a sword"


class TestJsonExtraction:
    """Модели этого размера обрамляют ответ ```json ... ``` или добавляют
    фразу до/после — разбор обязан это переживать."""

    def test_plain_json(self):
        assert sbp._extract_json('{"a": 1}') == {"a": 1}

    def test_json_in_markdown_fence(self):
        assert sbp._extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_prose_around(self):
        raw = 'Sure! Here is the brief:\n{"a": 1}\nHope that helps.'
        assert sbp._extract_json(raw) == {"a": 1}

    def test_nested_braces_are_balanced_correctly(self):
        raw = '{"era": {"from": 1300, "to": 1500}, "x": 1}'
        assert sbp._extract_json(raw) == {"era": {"from": 1300, "to": 1500}, "x": 1}

    def test_braces_inside_strings_do_not_break_balance(self):
        """Регуляркой этот случай режется неверно — поэтому разбор по скобкам."""
        raw = '{"why": "фраза со скобкой } внутри", "n": 2}'
        assert sbp._extract_json(raw) == {"why": "фраза со скобкой } внутри", "n": 2}

    def test_escaped_quote_inside_string(self):
        raw = '{"why": "он сказал \\"нет\\"", "n": 1}'
        assert sbp._extract_json(raw) == {"why": 'он сказал "нет"', "n": 1}

    def test_no_json_returns_none(self):
        assert sbp._extract_json("совсем не json") is None
        assert sbp._extract_json("") is None
        assert sbp._extract_json(None) is None

    def test_truncated_json_returns_none(self):
        """Обрыв по max_tokens — частый случай, не должен падать."""
        assert sbp._extract_json('{"a": 1, "b": ') is None


class TestDeterminism:
    def test_cache_key_is_stable_for_same_input(self):
        a = sbp._cache_key("текст", "до", "после", "model.gguf")
        b = sbp._cache_key("текст", "до", "после", "model.gguf")
        assert a == b

    def test_cache_key_changes_with_every_input_component(self):
        base = sbp._cache_key("текст", "до", "после", "model.gguf")
        assert sbp._cache_key("другой", "до", "после", "model.gguf") != base
        assert sbp._cache_key("текст", "иное", "после", "model.gguf") != base
        assert sbp._cache_key("текст", "до", "иное", "model.gguf") != base
        assert sbp._cache_key("текст", "до", "после", "other.gguf") != base

    def test_prompt_version_is_part_of_the_key(self):
        """Переписал промпт -> старые вердикты не наследуются молча.
        Тот же урок, что уже усвоен у VLM-арбитра (_arbiter_prompt_signature)."""
        base = sbp._cache_key("текст", "", "", "model.gguf")
        saved = sbp.PROMPT_VERSION
        try:
            sbp.PROMPT_VERSION = saved + 1
            assert sbp._cache_key("текст", "", "", "model.gguf") != base
        finally:
            sbp.PROMPT_VERSION = saved
