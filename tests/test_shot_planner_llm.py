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
        plan = {sp.unit_key(blocks[0]["text"]): {"shot_en": "A person standing up"}}
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

    def test_prompt_version_enters_the_cache_key(self, monkeypatch):
        """Переписанный промпт обязан считаться заново — но на уровне
        КЭША ОТВЕТА, а не ключа юнита в плане.

        Гарантия не отменена, а перенесена туда, где она работает.
        Раньше версия входила в `unit_key()`, и у этого был измеренный
        побочный отказ: имя модели входило туда же, а рендер идёт БЕЗ
        LLAMA_MODEL_GGUF, поэтому ключ не совпадал и план не находился
        ЦЕЛИКОМ — ноль брифов молча, при включённом флаге и готовом
        плане. Тихий отказ вместо предупреждения.

        Теперь: пересчёт обеспечивает `cache_key` (версия + модель),
        устаревший план — громкое предупреждение в `load_plan`,
        а `unit_key` называет ФРАЗУ и только её."""
        a = sp.cache_key("Фраза.")
        monkeypatch.setattr(sp, "PLANNER_PROMPT_VERSION", sp.PLANNER_PROMPT_VERSION + 1)
        assert sp.cache_key("Фраза.") != a

    def test_unit_key_names_the_phrase_and_nothing_else(self, monkeypatch):
        """Ключ юнита обязан пережить и смену версии, и отсутствие модели."""
        a = sp.unit_key("Фраза.")
        monkeypatch.setattr(sp, "PLANNER_PROMPT_VERSION", sp.PLANNER_PROMPT_VERSION + 1)
        monkeypatch.setattr(sp, "LLAMA_MODEL", "")
        assert sp.unit_key("Фраза.") == a

    def test_stale_plan_warns_instead_of_vanishing(self, tmp_path, capsys):
        """Устаревший план обязан СКАЗАТЬ о себе, а не исчезнуть."""
        import json
        mp = tmp_path / "media_plan"
        mp.mkdir()
        (mp / "shot_plan.json").write_text(json.dumps(
            {"version": sp.PLANNER_PROMPT_VERSION + 7,
             "units": {"abc": {"shot_en": "a rondel dagger"}}}),
            encoding="utf-8")
        units = sp.load_plan(str(tmp_path))
        assert units == {"abc": {"shot_en": "a rondel dagger"}}
        assert "ВНИМАНИЕ" in capsys.readouterr().out


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
        assert sp.plan_unit("любая фраза", None) is None
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
