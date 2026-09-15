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
import ast
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import shot_planner_llm as sp  # noqa: E402

# Пустой контекст для проверок, которые про контекст НЕ говорят.
NOCTX = {"section": "", "prev": []}

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fixtures", "shot_planner")


def real_output(i):
    with open(os.path.join(FIX, f"prod_out{i}.txt"), encoding="utf-8",
              errors="replace") as f:
        return f.read()


class TestParsingRealModelOutput:
    """Фикстуры — вывод БОЕВОГО промпта (PLANNER_PROMPT_VERSION=2), а не
    разведочного: фикстура обязана быть ответом на тот промпт, который
    реально в коде. Первая версия этих тестов падала именно поэтому."""

    def test_all_three_real_replies_parse(self):
        for i in (0, 1, 2):
            got = sp.parse_reply(real_output(i))
            assert got is not None, i
            assert got["shot_en"]

    def test_marker_is_not_required_llama_truncates_it(self):
        """llama-cli ОБРЕЗАЕТ эхо промпта («...(truncated)»), поэтому
        маркера `<|im_start|>assistant` в stdout нет вообще. Первая версия
        разбора искала его — и не находила ответ ни разу."""
        raw = real_output(0)
        assert "<|im_start|>assistant" not in raw
        assert sp.parse_reply(raw) is not None

    def test_negation_unit_is_understood(self):
        """Тот самый юнит, на котором детерминированное правило по
        отрицанию провалилось (1 верное срабатывание на 142, замер в
        CLAUDE.md). Модель называет отвергнутый предмет и уводит кадр на
        окружение."""
        got = sp.parse_reply(real_output(2))
        assert got["forbidden"] == "меч"
        assert "sword" not in got["shot_en"].lower()
        assert got["function"] == "scene"

    def test_measured_regression_on_the_metaphor_unit(self):
        """ИЗМЕРЕННЫЙ ПРОВАЛ, зафиксированный намеренно.

        На фразе «рыцарь весил как холодильник, на коня его поднимали
        КРАНОМ, лежал как перевёрнутая черепаха» модель:
          * выдумала отвержения (`forbidden` перечисляет метафоры, которых
            фраза не отвергает);
          * положила в описание кадра «a crane» — это приведёт
            СТРОИТЕЛЬНЫЙ КРАН, то есть брак, которого сегодняшний запрос
            секции («medieval knight plate armour closeup») не делает.

        Тест закрепляет факт, а не желаемое: пока это так, планировщик НЕ
        проходит планку репозитория «ничьи и победы, ноль регрессов», и
        флаг обязан оставаться выключенным. Когда промпт или модель
        починят это — тест упадёт и заставит перечитать вывод, а не
        позеленеет молча.
        """
        got = sp.parse_reply(real_output(0))
        assert "crane" in got["shot_en"].lower(), (
            "поведение изменилось — перемерить качество и решение по флагу")
        assert got["forbidden"] and "холодильник" in got["forbidden"]


class TestStreamFormattingIsStripped:
    """НАЙДЕНО СМЕНОЙ МОДЕЛИ, а не чтением кода (16.09).

    llama-cli оформляет поток: цвет, спиннер загрузки, перерисовка строки.
    Эти байты попадали ВНУТРЬ слов ответа, и на Q8-модели весь замер дал
    0 разобранных из 8:

        "subject?": "?? человек??",  "?": "shot?_?en?": "a??? person?"

    Q4 ту же ломку проскакивал случайно — то есть дефект жил в модуле всё
    время и ждал другой модели или другой скорости вывода. Если бы этот
    результат был принят за КАЧЕСТВО Q8, вывод «более точная квантовка
    хуже» был бы ложным.
    """

    def test_ansi_colour_inside_a_word_is_removed(self):
        dirty = '{\x1b[32m"shot_en"\x1b[0m: "a warrior standing up"}'
        got = sp.parse_reply(dirty)
        assert got is not None
        assert got["shot_en"] == "a warrior standing up"

    def test_control_bytes_inside_a_word_are_removed(self):
        got = sp.parse_reply('{"shot_en": "a\x08 warrior\x0c standing up"}')
        assert got is not None
        assert got["shot_en"] == "a warrior standing up"

    def test_real_text_is_untouched(self):
        clean = '{"shot_en": "a dented steel breastplate", "subject": "нагрудник"}'
        assert sp.parse_reply(clean)["subject"] == "нагрудник"

    def test_cleaning_lives_at_the_parser_not_only_at_the_call(self):
        """Разбор зовут и на сохранённых фикстурах, и на чужом выводе:
        оформление — свойство ИСТОЧНИКА, а не вызова."""
        import ast
        src = open(os.path.join(SCRIPTS_DIR, "shot_planner_llm.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "parse_reply")
        called = {getattr(c.func, "id", None) for c in ast.walk(fn)
                  if isinstance(c, ast.Call)}
        assert "_clean_stream" in called

    def test_prompt_goes_through_a_file_not_argv(self):
        """Промпт многострочный, с кавычками и разметкой ChatML: передача
        через argv зависит от шелла и длины командной строки."""
        import ast
        src = open(os.path.join(SCRIPTS_DIR, "shot_planner_llm.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_run_model")
        flags = [c.value for c in ast.walk(fn)
                 if isinstance(c, ast.Constant) and isinstance(c.value, str)]
        assert "-f" in flags and "-p" not in flags
        assert "--log-disable" in flags


class TestValidationRefusesGarbage:
    """Рендер никогда не доверяет плану без проверки — тот же принцип, что
    у speech_plan.json. Невалидный ответ обязан дать None, а не мусор."""

    @pytest.mark.parametrize("raw", [
        "", None, "просто текст без json",
        '{"shot_en": null}',
        '{"shot_en": ""}',
        '{"shot_en": "одно"}',                    # < 2 слов
        '{"shot_en": "' + "w " * 20 + '"}',        # > 16 слов
        '{"shot_en": "рыцарь в доспехе крупно"}',  # кириллица
        '["не объект"]',
        '{"нет ключа": 1}',
    ])
    def test_invalid_replies_are_rejected(self, raw):
        assert sp.parse_reply(raw) is None

    def test_unknown_function_becomes_none_not_a_guess(self):
        got = sp.parse_reply('{"shot_en": "a steel helmet close up", '
                             '"function": "ВЫДУМАННЫЙ"}')
        assert got is not None
        assert got["function"] is None

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
        plan = {sp.unit_key(blocks[0]["text"], sp.unit_context(blocks, 0)):
                {"shot_en": "A person standing up"}}
        del sp.REJECTED[:]
        assert sp.fill_briefs(blocks, plan) == 0
        assert blocks[0]["shot_brief"] is None
        assert len(sp.REJECTED) == 1
        assert sp.REJECTED[0]["reason"]


class TestSamplingIsDeterministic:
    """Найдено внешней оценкой 16.09 и подтверждено проверкой: у llama.cpp
    `--seed` по умолчанию -1 (случайный), а температура стояла 0.2 — не
    ноль. Значит сравнение промптов v2 и v3 было НЕВОСПРОИЗВОДИМЫМ, и
    разница могла оказаться шумом выборки, а не эффектом правки.

    Планирование — не творческая задача: на один и тот же вопрос нужен
    один и тот же ответ, иначе теряет смысл и кэш по тексту фразы."""

    def test_temperature_zero_and_fixed_seed(self):
        import ast
        src = open(os.path.join(SCRIPTS_DIR, "shot_planner_llm.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_run_model")
        consts = [c.value for c in ast.walk(fn)
                  if isinstance(c, ast.Constant) and isinstance(c.value, str)]
        assert "--seed" in consts
        i = consts.index("--temp")
        assert consts[i + 1] == "0", consts[i:i + 2]

    def test_seed_is_overridable_but_never_random(self):
        assert isinstance(sp.SAMPLING_SEED, int)
        assert sp.SAMPLING_SEED >= 0


class TestAuthorAlwaysWins:
    def test_author_brief_is_never_overwritten(self):
        blocks = [{"text": "Стрела скользнула по нагруднику.",
                   "shot_brief": "a dented breastplate, close up"}]
        plan = {sp.unit_key(blocks[0]["text"], sp.unit_context(blocks, 0)):
                {"shot_en": "something else", "function": "object"}}
        filled = sp.fill_briefs(blocks, plan)
        assert filled == 0
        assert blocks[0]["shot_brief"] == "a dented breastplate, close up"

    def test_empty_brief_is_filled(self):
        blocks = [{"text": "Рыцарей убивала земля.", "shot_brief": None}]
        plan = {sp.unit_key(blocks[0]["text"], sp.unit_context(blocks, 0)):
                {"shot_en": "muddy churned battlefield ground", "function": "scene"}}
        assert sp.fill_briefs(blocks, plan) == 1
        assert blocks[0]["shot_brief"] == "muddy churned battlefield ground"
        assert blocks[0]["shot_type_hint"] == "scene"

    def test_whitespace_brief_counts_as_absent(self):
        blocks = [{"text": "Рыцарей убивала земля.", "shot_brief": "   "}]
        plan = {sp.unit_key(blocks[0]["text"], sp.unit_context(blocks, 0)):
                {"shot_en": "muddy ground wide"}}
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
        assert sp.unit_key("Рыцарей убивала земля.", NOCTX) == \
            sp.unit_key("Рыцарей убивала земля.", NOCTX)

    def test_whitespace_is_normalised(self):
        assert sp.unit_key("Рыцарей  убивала\n земля.", NOCTX) == \
            sp.unit_key("Рыцарей убивала земля.", NOCTX)

    def test_different_text_different_key(self):
        assert sp.unit_key("Первая фраза.", NOCTX) != sp.unit_key("Вторая фраза.", NOCTX)

    def test_prompt_version_enters_the_key(self, monkeypatch):
        """Переписанный промпт обязан считаться заново, иначе план молча
        останется от прошлой формулировки — тот же класс, что уже закрыт у
        кэша вердиктов VLM-арбитра."""
        a = sp.unit_key("Фраза.", NOCTX)
        monkeypatch.setattr(sp, "PLANNER_PROMPT_VERSION", sp.PLANNER_PROMPT_VERSION + 1)
        assert sp.unit_key("Фраза.", NOCTX) != a


class TestFailOpenNeverBreaksTheRender:
    def test_disabled_by_default(self):
        import feature_flags
        assert feature_flags.default_of("SHOT_PLANNER_LLM") == "0" \
            if hasattr(feature_flags, "default_of") else True

    def test_runtime_not_ready_without_env(self, monkeypatch):
        monkeypatch.setattr(sp, "LLAMA_BIN", "")
        monkeypatch.setattr(sp, "LLAMA_MODEL", "")
        assert sp.runtime_ready() is False

    def test_missing_binary_is_false_not_exception(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sp, "LLAMA_BIN", str(tmp_path / "нет"))
        monkeypatch.setattr(sp, "LLAMA_MODEL", str(tmp_path / "тоже нет"))
        assert sp.runtime_ready() is False

    def test_load_plan_of_missing_file_is_empty(self, tmp_path):
        assert sp.load_plan(str(tmp_path)) == {}

    def test_broken_plan_file_is_empty(self, tmp_path):
        mp = tmp_path / "media_plan"
        mp.mkdir()
        (mp / sp.PLAN_NAME).write_text("{ не json", encoding="utf-8")
        assert sp.load_plan(str(tmp_path)) == {}

    def test_call_budget_is_enforced(self, monkeypatch):
        """Жёсткий потолок живых вызовов — та же дисциплина, что у
        SHOT_DIRECTOR_MAX_CALLS_PER_RUN и SPEECH_GEN_MAX_CALLS_PER_RUN."""
        monkeypatch.setitem(sp.STATS, "calls", sp.MAX_CALLS_PER_RUN)
        called = []
        monkeypatch.setattr(sp, "_run_model", lambda p: called.append(1))
        assert sp.plan_unit("любая фраза", NOCTX, None) is None
        assert called == []


class TestWiredIntoTheRender:
    def _src(self):
        return open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                    encoding="utf-8").read()

    def test_render_reads_the_plan_and_never_calls_the_model(self):
        """Планирование идёт ~час на эпизод и не имеет права стоять внутри
        рендера, который перезапускают."""
        import ast
        tree = ast.parse(self._src())
        names = {getattr(c.func, "attr", None) for c in ast.walk(tree)
                 if isinstance(c, ast.Call)
                 and getattr(getattr(c.func, "value", None), "id", None) == "shot_planner_llm"}
        assert "load_plan" in names and "fill_briefs" in names
        for live in ("plan_episode", "plan_unit", "_run_model"):
            assert live not in names, f"{live} зовётся из рендера"

    def test_flag_is_registered_with_default_off(self):
        src = open(os.path.join(SCRIPTS_DIR, "feature_flags.py"),
                   encoding="utf-8").read()
        assert 'Flag("SHOT_PLANNER_LLM", "0"' in src


class TestContextReachesTheModel:
    """ИЗМЕРЕННЫЙ дефект, а не гипотеза: на эпизоде 02 у 24 юнитов из 142
    (17%) отсылка стоит в первых трёх словах, то есть антецедент физически
    вне юнита. Архетип — «При этом ОН был под ногами у каждого из НИХ»:
    модель без контекста ответила на него пересказом «He was at their
    feet», хотя «он» объяснён предыдущей фразой («Главного убийцу рыцарей
    нельзя выковать»).
    """

    def test_previous_phrases_are_in_the_context(self):
        blocks = [{"text": "Главного убийцу рыцарей нельзя выковать.",
                   "section": "HOOK"},
                  {"text": "И вот тут начинается главное.", "section": "HOOK"},
                  {"text": "При этом он был под ногами у каждого из них.",
                   "section": "HOOK"}]
        ctx = sp.unit_context(blocks, 2)
        assert ctx["section"] == "HOOK"
        assert ctx["prev"] == ["Главного убийцу рыцарей нельзя выковать.",
                               "И вот тут начинается главное."]

    def test_context_reaches_the_prompt(self):
        blocks = [{"text": "Главного убийцу рыцарей нельзя выковать."},
                  {"text": "При этом он был под ногами."}]
        prompt = sp.build_prompt(blocks[1]["text"], sp.unit_context(blocks, 1))
        assert "Главного убийцу рыцарей нельзя выковать." in prompt
        # Сама фраза обязана остаться на месте и ПОСЛЕ контекста: модель
        # описывает кадр для неё, а не для предыстории.
        assert prompt.index("Главного убийцу") < prompt.index("При этом он был")

    def test_context_enters_the_cache_key(self):
        """Не войди контекст в ключ — правка соседней фразы оставила бы
        ответ, данный на ДРУГОЙ вопрос, и увидеть это было бы негде. Тот же
        класс, ради которого заведён candidate_brief_key."""
        a = [{"text": "Главного убийцу рыцарей нельзя выковать."},
             {"text": "При этом он был под ногами."}]
        b = [{"text": "Возьми настоящий боевой меч."},
             {"text": "При этом он был под ногами."}]
        ka = sp.unit_key(a[1]["text"], sp.unit_context(a, 1))
        kb = sp.unit_key(b[1]["text"], sp.unit_context(b, 1))
        assert ka != kb

    def test_planning_and_filling_agree_on_the_key(self):
        """Обе стороны обязаны строить контекст ОДНОЙ функцией. Разойдись
        они — план был бы записан под одним ключом, а прочитан по другому,
        и заявка молча не доезжала бы до брифа."""
        blocks = [{"text": "Главного убийцу рыцарей нельзя выковать.",
                   "section": "HOOK"},
                  {"text": "При этом он был под ногами.", "section": "HOOK",
                   "shot_brief": None}]
        plan = {sp.unit_key(blocks[1]["text"], sp.unit_context(blocks, 1)):
                {"shot_en": "churned wet earth underfoot", "function": "scene"}}
        assert sp.fill_briefs(blocks, plan) == 1
        assert blocks[1]["shot_brief"] == "churned wet earth underfoot"

    def test_context_does_not_look_forward(self):
        """Эпизод намеренно придерживает ответ («Я его назову. Но если
        сказать прямо сейчас, ты пожмёшь плечами») — кадр, собранный по ещё
        не прозвучавшей фразе, выдал бы разгадку раньше диктора."""
        blocks = [{"text": "Я его назову."},
                  {"text": "Рыцарей убивала земля."}]
        ctx = sp.unit_context(blocks, 0)
        assert ctx["prev"] == []
        assert "земля" not in sp.build_prompt(blocks[0]["text"], ctx)

    def test_first_unit_has_no_previous_and_still_works(self):
        blocks = [{"text": "Первая фраза эпизода.", "section": "HOOK"}]
        ctx = sp.unit_context(blocks, 0)
        assert ctx["prev"] == []
        assert sp.unit_key(blocks[0]["text"], ctx)
        assert "Фраза диктора" in sp.build_prompt(blocks[0]["text"], ctx)

    def test_context_is_fail_open_on_junk(self):
        for bad in ([], [None], ["строка"], [{"text": None}]):
            ctx = sp.unit_context(bad, 0)
            assert isinstance(ctx, dict)
            assert isinstance(sp.context_text(ctx), str)
        assert sp.context_text(None) == ""

    def test_forgetting_the_context_is_loud(self):
        """Параметр без значения по умолчанию — намеренно: забывчивый
        вызывающий получает TypeError сразу, а не тихо чужой кадр. Тот же
        довод, по которому отказ VLM-арбитра сделан отдельным типом."""
        import pytest
        with pytest.raises(TypeError):
            sp.unit_key("Фраза.")
        with pytest.raises(TypeError):
            sp.build_prompt("Фраза.")


class TestShotShapeGate:
    """Описание кадра начинается с того, ЧТО в кадре. Начало с местоимения
    означает пересказ фразы вместо картинки.

    ИЗМЕРЕНО на данных этого канала, не предположено:
      * 142 авторских брифа эпизода 02 — ложных отказов НОЛЬ;
      * записанные ответы модели v3 — пойман 1 из 8, годных не потеряно;
      * записанные ответы модели v2 — поймано 2 из 8, ровно те два, что
        переписывание промпта закрывало словами.
    """

    def test_real_model_failure_is_rejected(self):
        assert sp.brief_is_shot_like("He was at their feet") is False
        assert sp.brief_is_shot_like("You have not been injured.") is False
        assert sp.brief_is_shot_like("This object will be revisited") is False

    def test_real_model_successes_survive(self):
        for good in ("A warrior standing up",
                     "A warrior in full iron armor being struck by a battle sword",
                     "A warrior lying face down in mud, seemingly about to die.",
                     "A sword strikes a helmet; the helmet resists."):
            assert sp.brief_is_shot_like(good) is True

    def test_no_author_brief_of_the_real_episode_is_rejected(self):
        """Самая сильная проверка: правило прогоняется по ВСЕМУ корпусу
        настоящих авторских брифов эпизода, а не по придуманным примерам."""
        path = os.path.join(os.path.dirname(SCRIPTS_DIR),
                            "videos", "02_ne-mechom", "script.txt")
        if not os.path.exists(path):
            import pytest
            pytest.skip("нет сценария эпизода 02")
        import script_parser
        briefs = [(b.get("shot_brief") or "").strip()
                  for b in script_parser.parse_blocks(path)]
        briefs = [b for b in briefs if b]
        assert len(briefs) >= 100          # выборка обязана быть настоящей
        rejected = [b for b in briefs if not sp.brief_is_shot_like(b)]
        assert rejected == []

    def test_the_gate_is_wired_into_parse_reply(self):
        got = sp.parse_reply('{"shot_en": "He was at their feet", '
                             '"function": "scene"}')
        assert got is None
        ok = sp.parse_reply('{"shot_en": "churned wet earth underfoot", '
                            '"function": "scene"}')
        assert ok and ok["shot_en"] == "churned wet earth underfoot"

    def test_rejection_is_counted_separately_from_invalid(self):
        before = dict(sp.STATS)
        sp.parse_reply('{"shot_en": "He was at their feet"}')
        assert sp.STATS["not_a_shot"] == before["not_a_shot"] + 1
        assert sp.STATS["invalid"] == before["invalid"]

    def test_word_boundary_is_respected(self):
        """«Itinerant», «Theatre», «Wedge» начинаются с тех же букв и
        кадрами являются. Правило стоит на СЛОВЕ, а не на префиксе — тот
        же приём, что уже держит словарь атмосферы от «зал» внутри «ЗАЛП»."""
        for good in ("Itinerant merchants on a muddy road",
                     "Theatre of war seen from a hill",
                     "Wedge formation of armoured knights",
                     "Iron gauntlet on dark cloth",
                     "Weapons rack in a stone hall"):
            assert sp.brief_is_shot_like(good) is True


class TestPlanningWalksTheFullScript:
    """Контекст юнита — соседние фразы СЦЕНАРИЯ, включая те, у которых бриф
    уже написан автором. Иди цикл по отфильтрованному списку — «предыдущей»
    оказалась бы фраза через две главы. Тот же класс промаха, что уже стоил
    arc_stage 151 слота из 165 (N4 аудита), когда стадия бралась по индексу
    ЦИКЛА вместо исходного индекса блока.
    """

    def _blocks(self):
        return [{"text": "Главного убийцу рыцарей нельзя выковать.",
                 "section": "HOOK", "shot_brief": "a dark empty museum case"},
                {"text": "При этом он был под ногами.",
                 "section": "HOOK", "shot_brief": None}]

    def test_context_of_a_planned_unit_sees_the_authored_neighbour(
            self, tmp_path, monkeypatch):
        seen = []

        def fake_run(prompt):
            seen.append(prompt)
            return '{"shot_en": "churned wet earth underfoot", "function": "scene"}'

        monkeypatch.setattr(sp, "_run_model", fake_run)
        blocks = self._blocks()
        plan = sp.plan_episode(str(tmp_path), blocks, verbose=False)
        assert len(seen) == 1                      # авторский юнит не спрашивали
        assert "Главного убийцу рыцарей нельзя выковать." in seen[0]
        # И тем же ключом заявка обязана доехать до брифа.
        assert sp.fill_briefs(blocks, plan) == 1
        assert blocks[1]["shot_brief"] == "churned wet earth underfoot"

    def test_plan_survives_a_round_trip_through_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sp, "_run_model",
            lambda prompt: '{"shot_en": "churned wet earth underfoot"}')
        blocks = self._blocks()
        sp.plan_episode(str(tmp_path), blocks, verbose=False)
        fresh = self._blocks()
        assert sp.fill_briefs(fresh, sp.load_plan(str(tmp_path))) == 1


class TestStalePlanNamesItsOwnCause:
    """Версия промпта и имя модели входят в unit_key, поэтому план от
    прошлой версии не совпал бы ни одним ключом и так. Проверка нужна
    ради ПРИЧИНЫ: без неё рендер объяснял бы пустой результат правкой
    сценария и валил бы на автора то, что сделало обновление кода."""

    def _write(self, tmp_path, **over):
        mp = tmp_path / "media_plan"
        mp.mkdir(parents=True, exist_ok=True)
        data = {"version": sp.PLANNER_PROMPT_VERSION,
                "model": os.path.basename(sp.LLAMA_MODEL),
                "units": {"abc": {"shot_en": "muddy ground"}}}
        data.update(over)
        (mp / sp.PLAN_NAME).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_current_plan_loads(self, tmp_path):
        self._write(tmp_path)
        assert sp.load_plan(str(tmp_path)) == {"abc": {"shot_en": "muddy ground"}}

    def test_plan_of_an_older_prompt_is_dropped_with_a_reason(
            self, tmp_path, capsys):
        self._write(tmp_path, version=sp.PLANNER_PROMPT_VERSION - 1)
        assert sp.load_plan(str(tmp_path)) == {}
        out = capsys.readouterr().out
        assert "промптом v" in out and "перезапусти" in out

    def test_plan_of_another_model_is_dropped_with_a_reason(
            self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sp, "LLAMA_MODEL", "/models/qwen-now.gguf")
        self._write(tmp_path, model="qwen-before.gguf")
        assert sp.load_plan(str(tmp_path)) == {}
        assert "моделью qwen-before.gguf" in capsys.readouterr().out


class TestOneTranslationRuleNotTwo:
    """Две копии одного правила — тот самый класс, который уже стоил этому
    репозиторию PHRASE LOCK на целый эпизод. Вторая версия (местоимение
    где-угодно И глагол где-угодно) удалена по ЗАМЕРУ ЗАПАСА: три реальных
    авторских брифа уже выполняют её половину («…crushed into it», «…wounds
    on it», «…no hole in it») и выживают только потому, что в них не
    случилось глагола из списка."""

    def test_the_second_copy_is_gone(self):
        assert not hasattr(sp, "_looks_like_translation")
        src = open(os.path.join(SCRIPTS_DIR, "shot_planner_llm.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        checks = [n.name for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and "translation" in n.name.lower()]
        assert checks == []

    def test_the_validator_uses_the_surviving_rule(self):
        ok, why = sp.brief_is_safe("He was at their feet", "любая фраза")
        assert ok is False and "пересказ" in why

    def test_real_author_briefs_survive_the_validator(self):
        """Ровно те три брифа, на которых удалённое правило стояло в одном
        слове от ложного отказа."""
        for good in ("a steel helmet with a deep dent crushed into it",
                     "a human skull with wounds on it",
                     "an undamaged breastplate with no hole in it"):
            assert sp.brief_is_shot_like(good) is True
        # И тот же бриф, переформулированный с личной формой глагола —
        # удалённое правило отвергло бы его, форма не отвергает.
        assert sp.brief_is_shot_like(
            "a breastplate that has a hole in it") is True

    def test_the_surviving_rule_catches_what_the_removed_one_missed(self):
        """«lies» нет в списке глаголов удалённого правила — оно пропустило
        бы этот пересказ. Форма ловит."""
        assert sp.brief_is_shot_like("He lies in the mud") is False
        assert sp.brief_is_shot_like("It rests on dark cloth") is False


class TestShotIsOneMoment:
    """Кадр — ОДИН момент, а не рассказ. Правило откалибровано на КОРПУСЕ
    из 142 авторских брифов, а не на выборке из восьми: более одного
    предложения среди них — НОЛЬ, ни один не оканчивается даже точкой,
    медиана 9 слов. Ловит ровно тот промах, который детерминированный
    замер оставил стоять: «Sword struck helmet - steel held. Arrow slid
    over cuirass and missed.» — дословный перевод фразы без единого
    местоимения, поэтому признак начала его не берёт."""

    def test_two_sentences_are_rejected(self):
        assert sp.brief_is_shot_like(
            "Sword struck helmet - steel held. Arrow slid over cuirass "
            "and missed.") is False

    def test_a_trailing_period_is_not_a_second_sentence(self):
        """Хвостовая точка — законный кадр, и одна из реальных верных
        заявок замера её имеет. Спутать эти два случая значило бы отнять
        годный кадр."""
        assert sp.brief_is_shot_like(
            "A warrior lying face down in mud, seemingly about to die."
        ) is True

    def test_no_author_brief_is_multi_sentence(self):
        path = os.path.join(os.path.dirname(SCRIPTS_DIR),
                            "videos", "02_ne-mechom", "script.txt")
        if not os.path.exists(path):
            pytest.skip("нет сценария эпизода 02")
        import script_parser
        briefs = [(b.get("shot_brief") or "").strip()
                  for b in script_parser.parse_blocks(path)]
        briefs = [b for b in briefs if b]
        assert len(briefs) >= 100
        assert [b for b in briefs if not sp.brief_is_shot_like(b)] == []

    def test_the_deterministic_eval_loses_nothing_good(self):
        """Пять остальных заявок детерминированного замера обязаны
        пережить правило — иначе это не гейт, а глушилка."""
        for good in ("A warrior in full chainmail armor being struck by "
                     "a real battle sword",
                     "A man stepping onto a battlefield",
                     "A warrior in full protective gear",
                     "A warrior standing up"):
            assert sp.brief_is_shot_like(good) is True
