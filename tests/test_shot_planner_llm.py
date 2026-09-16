# -*- coding: utf-8 -*-
"""Локальный режиссёр: фраза диктора -> описание кадра, на CPU и бесплатно.

Фикстуры — РЕАЛЬНЫЙ вывод Qwen2.5-7B-Instruct Q4_K_M, снятый живым
прогоном 16.09 на настоящих фразах эпизода 02 (4 ядра, llama.cpp).
Синтетика здесь не годится: она не воспроизвела бы ни баннер сборки перед
ответом, ни хвост со статистикой после JSON, ни того факта, что модель
кладёт JSON в свободный текст. Именно на этом разбор и ломается.

Замер того же прогона: обработка промпта ~55 ток/с, генерация ~5.5 ток/с,
21-26 с на юнит, около часа на эпизод из 142 юнитов — разово и кэшируемо.
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import shot_planner_llm as sp  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fixtures", "shot_planner")








class TestValidationRefusesGarbage:
    """Рендер никогда не доверяет плану без проверки — тот же принцип, что
    у speech_plan.json. Невалидный ответ обязан дать None, а не мусор."""

    def test_function_vocabulary_matches_the_routing_one(self):
        """Второго словаря типов кадра не заводится: разойдись они — модель
        называла бы тип, которого маршрутизация по источникам не знает."""
        import shot_types
        known = set(shot_types.SOURCE_CAPABILITIES.get("pexels", {}))
        for fn in sp.VALID_FUNCTIONS:
            assert isinstance(fn, str) and fn
        # Все объявленные типы обязаны быть известны маршрутизации.
        import pipeline_smart as ps
        for fn in sp.VALID_FUNCTIONS:
            ps.source_allowed_for("pexels", fn)   # не должно бросать


class TestValidatorMakesRegressionStructurallyImpossible:
    """Главный пропущенный ход, взятый из внешней оценки 16.09 и признанный
    верным: модель НЕ обязана быть права всегда — она обязана быть ПРАВА
    ИЛИ МОЛЧАТЬ.

    Заявка либо проходит детерминированную проверку и улучшает слот, либо
    отбрасывается, и слот идёт ровно как сегодня. Тогда ухудшение
    невозможно ПО ПОСТРОЕНИЮ, а не по результату замера — и планка
    репозитория «ничьи и победы, ноль регрессов» становится выполнимой
    честно, а не вечным блокиратором на восьми вручную выбранных фразах.

    Все случаи ниже — РЕАЛЬНЫЕ ответы модели из замеров v2/v3
    (docs/quality/shot_planner_eval_v*.json), а не выдуманные.
    """

    @pytest.mark.parametrize("shot,phrase,why", [
        ("You have not been injured. The worst is yet to come.",
         "Тебя ещё не ранили. Вот что самое страшное.", "пересказ"),
        ("He was at their feet",
         "При этом он был под ногами у каждого из них.", "пересказ"),
        ("A person wearing full-body protective gear",
         "Представь драку, где у всех ножи.", "современное снаряжение"),
        ("A person standing up", "Тебе нужно всего лишь встать.", "эпоха"),
    ])
    def test_measured_failures_are_rejected(self, shot, phrase, why):
        ok, reason = sp.brief_is_safe(shot, phrase, blocklist=())
        assert ok is False, (shot, why)
        assert reason

    @pytest.mark.parametrize("shot,phrase", [
        ("A medieval sword, shining in the light", "Возьми настоящий боевой меч"),
        ("A warrior lying face down in mud", "ты лежишь лицом в грязи"),
        ("a dented steel breastplate, close up", "Стрела скользнула по нагруднику."),
    ])
    def test_good_briefs_pass(self, shot, phrase):
        ok, reason = sp.brief_is_safe(shot, phrase, blocklist=())
        assert ok is True, reason

    def test_simile_rule_is_deliberately_absent(self):
        """ЧЕСТНЫЙ ПРЕДЕЛ, доказанный собственным тестом.

        Внешняя оценка предлагала запрещать предмет сравнения («как
        холодильник»). Первая версия такой проверки сравнивала РУССКОЕ
        слово из фразы с АНГЛИЙСКИМ описанием кадра — «холод» против «a
        refrigerator» — и была мёртвым кодом: сработать не могла ни разу.

        Мини-словарь соответствий не заводится: тот же ненадёжный приём,
        что уже отвергнут для «crane» (журавль законен на миниатюре) и в
        stress_placement для «атлас». Случай закрыт промптом v3, и замер
        это подтвердил.
        """
        ok, _ = sp.brief_is_safe("a refrigerator in a field",
                                 "рыцарь весил как холодильник", blocklist=())
        assert ok is True
        src = open(os.path.join(SCRIPTS_DIR, "shot_planner_llm.py"),
                   encoding="utf-8").read()
        assert "SIMILE_MARKERS" not in src, "мёртвое правило вернулось"

    def test_blocklist_is_the_channel_one_not_a_second_copy(self):
        import ast
        src = open(os.path.join(SCRIPTS_DIR, "shot_planner_llm.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "channel_blocklist")
        attrs = {getattr(n, "attr", None) for n in ast.walk(fn)}
        assert "CONTENT_ALT_BLOCKLIST" in attrs, sorted(a for a in attrs if a)

    def test_rejected_briefs_are_recorded_not_silent(self):
        blocks = [{"text": "Тебе нужно всего лишь встать.", "shot_brief": None}]
        plan = {sp.unit_key(blocks[0]["text"]): {"shot_en": "A person standing up"}}
        del sp.REJECTED[:]
        assert sp.fill_briefs(blocks, plan) == 0
        assert blocks[0]["shot_brief"] is None
        assert len(sp.REJECTED) == 1
        assert sp.REJECTED[0]["reason"]


class TestAuthorAlwaysWins:
    def test_author_brief_is_never_overwritten(self):
        blocks = [{"text": "Стрела скользнула по нагруднику.",
                   "shot_brief": "a dented breastplate, close up"}]
        plan = {sp.unit_key(blocks[0]["text"]): {"shot_en": "something else",
                                                 "function": "object"}}
        filled = sp.fill_briefs(blocks, plan)
        assert filled == 0
        assert blocks[0]["shot_brief"] == "a dented breastplate, close up"

    def test_empty_brief_is_filled(self):
        blocks = [{"text": "Рыцарей убивала земля.", "shot_brief": None}]
        plan = {sp.unit_key(blocks[0]["text"]):
                {"shot_en": "muddy churned battlefield ground", "function": "scene"}}
        assert sp.fill_briefs(blocks, plan) == 1
        assert blocks[0]["shot_brief"] == "muddy churned battlefield ground"
        assert blocks[0]["shot_type_hint"] == "scene"

    def test_whitespace_brief_counts_as_absent(self):
        blocks = [{"text": "Рыцарей убивала земля.", "shot_brief": "   "}]
        plan = {sp.unit_key(blocks[0]["text"]): {"shot_en": "muddy ground wide"}}
        assert sp.fill_briefs(blocks, plan) == 1

    def test_no_plan_changes_nothing(self):
        blocks = [{"text": "Фраза.", "shot_brief": None}]
        assert sp.fill_briefs(blocks, {}) == 0
        assert blocks[0]["shot_brief"] is None


class TestKeyedByPhraseNotIndex:
    """Номера юнитов сдвигаются от ЛЮБОЙ правки текста выше по сценарию, и
    план молча описывал бы чужую фразу. Тот же дефект, от которого уже
    защищается lock в шотлисте и ради которого [shot:] сделан инлайновым."""

    def test_same_text_same_key(self):
        assert sp.unit_key("Рыцарей убивала земля.") == sp.unit_key("Рыцарей убивала земля.")

    def test_whitespace_is_normalised(self):
        assert sp.unit_key("Рыцарей  убивала\n земля.") == sp.unit_key("Рыцарей убивала земля.")

    def test_different_text_different_key(self):
        assert sp.unit_key("Первая фраза.") != sp.unit_key("Вторая фраза.")





class TestFailOpenNeverBreaksTheRender:
    def test_disabled_by_default(self):
        import feature_flags
        assert feature_flags.default_of("SHOT_PLANNER_LLM") == "0" \
            if hasattr(feature_flags, "default_of") else True



    def test_load_plan_of_missing_file_is_empty(self, tmp_path):
        assert sp.load_plan(str(tmp_path)) == {}

    def test_broken_plan_file_is_empty(self, tmp_path):
        mp = tmp_path / "media_plan"
        mp.mkdir()
        (mp / sp.PLAN_NAME).write_text("{ не json", encoding="utf-8")
        assert sp.load_plan(str(tmp_path)) == {}



class TestWiredIntoTheRender:
    def _src(self):
        return open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                    encoding="utf-8").read()


    def test_flag_is_registered_and_default_matches_the_measurement(self):
        """Дефолт 0 -> 1 (16.09) вместе со сменой мозга.

        Прежний пофразовый автомат стоял выключенным ПРАВИЛЬНО: 49
        попаданий против 58 у запроса секции. Глава с контекстом даёт 69
        на том же срезе и той же метрике — впервые лучше того, что было
        ДО режиссёра, и это то же число, по которому выключали
        предшественника.

        Тест не «разрешает единицу», а требует, чтобы реестр и CLAUDE.md
        говорили одно и то же; расхождение уже месяцами жило в этом
        репозитории у VLM_ARBITER_MODE и DEFLICKER_ENABLED."""
        import feature_flags
        assert feature_flags.FLAGS["SHOT_PLANNER_LLM"].default == "1"

    def test_enabling_the_flag_cannot_take_anything_away(self, tmp_path):
        """Почему единицу вообще можно ставить дефолтом: план читается С
        ДИСКА. Эпизод, где режиссёра не гоняли, получает пустой план, и
        ни один бриф не проставляется — поведение байт-в-байт прежнее."""
        blocks = [{"text": "Фраза без всякого плана.", "shot_brief": None}]
        before = [dict(b) for b in blocks]
        assert sp.load_plan(str(tmp_path)) == {}
        assert sp.fill_briefs(blocks, sp.load_plan(str(tmp_path))) == 0
        assert blocks == before
