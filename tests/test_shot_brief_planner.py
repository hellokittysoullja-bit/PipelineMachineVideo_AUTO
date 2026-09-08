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


class TestExampleQueryGuard:
    """Запрос, ДОСЛОВНО списанный из разобранного примера промпта, не несёт
    информации об этой фразе. Замер v2 по живому Pexels показал цену: один
    запрос примера ушёл в девять слотов, и один и тот же кадр победил в трёх
    разных местах ролика — ровно та болезнь, которую режиссёр лечит."""

    def test_copied_example_query_is_dropped(self):
        copied = "medieval sword blade macro detail"
        assert copied in sbp.EXAMPLE_QUERIES
        brief, err = sbp.validate_brief(
            _ok_raw(queries_en=[copied, "gauntlet gripping a crossguard"]))
        assert err is None
        assert brief["queries_en"] == ["gauntlet gripping a crossguard"]

    def test_case_and_spacing_do_not_smuggle_a_copy_through(self):
        brief, err = sbp.validate_brief(_ok_raw(queries_en=[
            "  Medieval   Sword Blade   MACRO Detail ", "knight in a winter field"]))
        assert err is None
        assert brief["queries_en"] == ["knight in a winter field"]

    def test_brief_made_only_of_copies_is_rejected(self):
        """Слот тогда остаётся на прежнем поведении пайплайна (авторский
        запрос / словарь тем) — строго не хуже, чем сегодня."""
        brief, err = sbp.validate_brief(_ok_raw(queries_en=[
            "medieval sword blade macro detail",
            "knight longsword in museum display case",
            "armourer holding longsword in forge"]))
        assert brief is None
        assert "английск" in err or "валидн" in err

    def test_guard_is_verbatim_only_and_does_not_eat_similar_work(self):
        """Намеренно узкий гвард: похожий по смыслу, но самостоятельно
        сформулированный запрос — законная работа модели, его резать нельзя."""
        brief, err = sbp.validate_brief(_ok_raw(queries_en=[
            "medieval sword blade macro", "sword blade detail medieval"]))
        assert err is None
        assert len(brief["queries_en"]) == 2

    def test_every_example_query_of_every_prompt_version_is_covered(self):
        """Канарейка против тихого расхождения: добавили версию промпта с
        новым примером — его запросы обязаны попасть в гвард, иначе
        списывание вернётся необнаруженным."""
        import re as _re
        for version, (system, _user) in sbp.PROMPTS.items():
            for raw in _re.findall(r'"queries_en"\s*:\s*\[(.*?)\]', system, _re.S):
                for q in _re.findall(r'"([^"]+)"', raw):
                    assert " ".join(q.split()).lower() in sbp.EXAMPLE_QUERIES, (
                        f"запрос примера промпта v{version} не в EXAMPLE_QUERIES: {q!r}")


class TestSelfContradictingBrief:
    """Бриф, запрещающий собственный subject, — не придирка к стилю, а
    измеренная болезнь: на прогоне v3 по 20 реальным слотам эпизода 13
    брифов из 20 внесли в must_not_contain слово из своего же subject
    («sword» на канале про мечи). Их негативы в роли анкеров вето
    забраковали бы ровно тот предмет, ради которого слот существует.

    Барьер проверен против НЕЗАВИСИМОГО замера по живому Pexels: он
    отклоняет 1 из 4 зафиксированных глазами регрессов и 0 из 7
    улучшений."""

    def test_banning_own_subject_is_rejected(self):
        brief, err = sbp.validate_brief(_ok_raw(
            subject="medieval longsword on a table",
            must_not_contain=["katana", "longsword"]))
        assert brief is None
        assert "longsword" in err

    def test_the_real_measured_case_is_caught(self):
        """Дословный бриф v3 на ep01_005 («вспомни, сколько весит пакет
        молока») — тот самый слот, чей кадр с современной кухней и
        хлопьями стоит в опубликованном эпизоде. Модель поняла фразу
        верно («milk is a comparison, not the actual weight»), запретила
        пакет молока — и поставила его же в subject."""
        brief, err = sbp.validate_brief(_ok_raw(
            reading="figurative",
            subject="a carton of milk",
            setting="a medieval kitchen with a woman holding a sword",
            queries_en=["woman holding a carton of milk macro",
                        "woman in a medieval kitchen holding a sword"],
            must_not_contain=["sword", "carton of milk", "milk", "kitchen"]))
        assert brief is None
        assert "milk" in err

    def test_function_words_do_not_trigger_a_false_rejection(self):
        """«of», «in», «close up» общие у половины строк — совпадение по
        ним не самозапрет, а грамматика."""
        brief, err = sbp.validate_brief(_ok_raw(
            subject="close up of a sword in a hall",
            must_not_contain=["view of a modern gym", "close up of a phone"]))
        assert err is None and brief is not None

    def test_unrelated_negatives_pass(self):
        brief, err = sbp.validate_brief(_ok_raw(
            subject="medieval longsword on a wooden table",
            must_not_contain=["kitchen scales", "modern gym", "katana"]))
        assert err is None and brief is not None

    def test_empty_negatives_are_not_a_contradiction(self):
        brief, err = sbp.validate_brief(_ok_raw(must_not_contain=[]))
        assert err is None and brief is not None


class TestChannelEraWindow:
    """Окно эпохи брифа, не пересекающееся с эпохой канала, — бриф не про
    материал этого канала. Правило проверено на 40 реальных брифах двух
    прогонов: срабатывает ровно один раз, ложных — ноль."""

    def test_the_real_measured_case_is_caught(self):
        """Дословный бриф v3 на ep01_133 («человек в доспехе — это человек
        в термосе»): era 1700-1800 и термос прямо в subject."""
        brief, err = sbp.validate_brief(_ok_raw(
            reading="figurative",
            subject="soldier in armor and a thermos",
            era_from=1700, era_to=1800,
            queries_en=["soldier in armor holding a thermos macro"],
            must_not_contain=["sword", "mace", "castle"]))
        assert brief is None
        assert "1700" in err and "эпох" in err

    def test_overlapping_window_passes(self):
        """Частичное пересечение — не повод терять слот: канал говорит и о
        предыстории, и о том, что было после."""
        brief, err = sbp.validate_brief(_ok_raw(era_from=1450, era_to=1650))
        assert err is None and brief is not None

    def test_missing_years_are_not_evidence(self):
        brief, err = sbp.validate_brief(_ok_raw(era_from=None, era_to=None))
        assert err is None and brief is not None

    def test_window_comes_from_channel_profile_when_declared(self, tmp_path, monkeypatch):
        """Те же ворота настройки под нишу, что у content_alt_blocklist:
        канал про античность не должен упираться в чужое окно."""
        prof = tmp_path / "channel_profile.json"
        prof.write_text('{"era_from": -800, "era_to": 400}', encoding="utf-8")
        monkeypatch.setattr(sbp, "REPO", str(tmp_path))
        assert sbp.channel_era_window() == (-800, 400)

    def test_broken_profile_falls_back_to_default(self, tmp_path, monkeypatch):
        (tmp_path / "channel_profile.json").write_text("{ битый", encoding="utf-8")
        monkeypatch.setattr(sbp, "REPO", str(tmp_path))
        assert sbp.channel_era_window() == sbp.CHANNEL_ERA_DEFAULT


class TestPromptVersions:
    def test_v3_example_is_off_topic_for_this_episode(self):
        """Причина копирования в v2 — пример был про меч, то есть про то же,
        что почти каждый слот эпизода. Пример v3 обязан быть из другой темы,
        иначе правка косметическая."""
        system, _ = sbp.PROMPTS[3]
        example = system.split("WORKED EXAMPLE", 1)[1]
        for word in ("sword", "longsword", "blade", "katana"):
            assert word not in example.lower(), f"пример v3 снова про {word}"

    def test_v3_names_the_two_phrases_that_failed_in_v2(self):
        """«потолок» и «пакет молока» — не абстрактный риск, а два
        измеренных промаха v2 на реальных слотах эпизода."""
        system, _ = sbp.PROMPTS[3]
        assert "потолок" in system
        assert "milk" in system.lower()


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
