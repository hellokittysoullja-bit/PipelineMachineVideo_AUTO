"""Тип кадра решает, в какой источник и в каком виде уходит запрос.

ИЗМЕРЕННАЯ причина (A/B на девяти реальных слотах эпизода 02, 13-14.09).
Два регресса, которых не лечит ни фильтр, ни ранжирование:

* «medieval plate armour museum» — Мет по слову *plate* отдаёт керамические
  тарелки и настенные часы; они честно проходят паспорт (Европа, XV век) и
  проигрывают только настоящему доспеху, которого в выборке не было.
  Замер живого API: свободный текст 480 objectID (первые — «Plate with
  Water Bird», «Mirror clock»), `departmentId=4` — 277, первые шесть все
  до одного латы. На «medieval rondel dagger» отдел не теряет ничего:
  17 -> 15, те же кинжалы.
* «medieval castle moat water» — у музеев фотографии рва нет вообще: Мет
  отдаёт «Мадонну с младенцем» (relevance 0.14-0.21), и такой кандидат
  выигрывал слот только потому, что стоял в списке первым. С отделом
  оружия тот же запрос даёт 0 objectID — музей честно не про это.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.mkdtemp(prefix="shottypes_")]

import shot_types as st  # noqa: E402
import pipeline_smart as ps  # noqa: E402
import script_parser  # noqa: E402


class TestExplicitMarkup:
    def test_author_marks_the_type_in_the_query_line(self):
        assert st.parse_query_spec("medieval sword macro [object]") == ("medieval sword macro", "object")
        assert st.parse_query_spec("  medieval camp [scene] ") == ("medieval camp", "scene")

    def test_unknown_word_in_brackets_is_not_swallowed(self):
        """Опечатка `[objct]` не должна молча выключить маршрутизацию и
        уехать скобкой в поисковый запрос — она остаётся видимой."""
        text, kind = st.parse_query_spec("medieval sword [objct]")
        assert kind is None and text == "medieval sword [objct]"

    def test_no_brackets_is_no_type(self):
        assert st.parse_query_spec("medieval sword") == ("medieval sword", None)

    def test_explicit_beats_inference(self):
        assert st.shot_type_for("medieval sword macro", explicit="scene") == "scene"
        assert st.shot_type_for("medieval sword macro") == "object"

    def test_parser_strips_the_marker_from_the_query(self, tmp_path):
        p = tmp_path / "script.txt"
        p.write_text("=== HOOK ===\nтекст\n\n=== PEXELS QUERIES ===\n"
                     "HOOK: medieval sword macro [object], medieval camp tent [scene]\n",
                     encoding="utf-8")
        queries = script_parser.parse_pexels_queries(str(p))
        assert queries["HOOK"] == ["medieval sword macro", "medieval camp tent"]
        types = script_parser.parse_query_shot_types(str(p))
        assert types == {"medieval sword macro": "object", "medieval camp tent": "scene"}


class TestInferenceOrderIsMeasured:
    @pytest.mark.parametrize("query", [
        "medieval battlefield armour mud",
        "medieval knight armour fallen mud",
        "knight armour fallen ground",
        "medieval helmet lying dirt",
    ])
    def test_object_plus_place_is_a_scene(self, query):
        """Первая версия словаря ставила `object` выше `scene`, и эти четыре
        РЕАЛЬНЫХ запроса эпизода уезжали в музей из-за слова armour/helmet.
        В A/B ровно такой слот выиграла музейная шкатулка из слоновой кости.
        Есть и предмет, и место — это сцена с предметом внутри."""
        assert st.shot_type_for(query) == "scene"

    @pytest.mark.parametrize("query,expected", [
        ("medieval plate armour museum", "object"),
        ("medieval rondel dagger", "object"),
        ("medieval sword hilt pommel macro", "object"),
        ("medieval manuscript battle illustration", "illustration"),
        ("medieval tomb effigy knight", "illustration"),
        ("medieval castle moat water", "scene"),
        ("medieval camp tent knight", "scene"),
    ])
    def test_real_episode_queries(self, query, expected):
        assert st.shot_type_for(query) == expected

    def test_medium_beats_place_for_illustrations(self):
        """«manuscript battle illustration» — это рукопись, а не поле боя, и
        рукописи как раз лучшее, что отдают музеи и архивы."""
        assert st.shot_type_for("medieval manuscript knight battle") == "illustration"

    def test_no_signal_means_no_guess(self):
        assert st.infer_shot_type("something entirely unrelated") is None
        assert st.shot_type_for("something entirely unrelated") == st.ANY


class TestEnglishMorphology:
    @pytest.mark.parametrize("query", [
        "medieval sabaton armoured foot",     # armoured != armour по целому слову
        "medieval blades collection",         # blades
        "castle ruins overgrown",             # ruins
    ])
    def test_derived_forms_are_matched(self, query):
        assert st.shot_type_for(query) != st.ANY

    @pytest.mark.parametrize("word", ["coincidence", "mapping", "maple", "bowl"])
    def test_short_terms_do_not_match_by_prefix(self, word):
        """Тот же класс ловушки, что уже ловили в словаре атмосферы («зал»
        ловил ЗАЛП): короткие термины сравниваются только по целому слову."""
        assert st.infer_shot_type(f"medieval {word} thing") is None


class TestStructuredMuseumQuery:
    def test_object_goes_to_arms_and_armor(self):
        assert st.met_department_for("medieval plate armour museum", "object") == 4
        assert st.met_department_for("medieval rondel dagger", "object") == 4

    def test_illustration_prefers_medieval_art_even_with_armour_in_it(self):
        """«medieval archer armour manuscript» — иллюстрация; первая версия
        отдавала её отделу оружия, потому что слово armour встречалось
        раньше в списке."""
        assert st.met_department_for("medieval archer armour manuscript", "illustration") == 17

    def test_scene_and_unknown_never_get_a_department(self):
        """Сужать по отделу на догадке нельзя: у `any` сегодняшнее поведение
        (свободный текст) обязано остаться байт-в-байт."""
        assert st.met_department_for("medieval castle moat water", "scene") is None
        assert st.met_department_for("medieval plate armour museum", st.ANY) is None


class TestRouting:
    @pytest.mark.parametrize("shot_type,expected", [
        ("object", True), ("illustration", True), ("map", True),
        ("scene", False), ("texture", False), (st.ANY, True),
    ])
    def test_museum_takes_objects_not_scenes(self, shot_type, expected):
        assert st.source_supports("museum", shot_type) is expected

    @pytest.mark.parametrize("source", ["openverse", "pexels", "pixabay", "unsplash"])
    def test_other_sources_take_everything(self, source):
        for shot_type in st.SHOT_TYPES + (st.ANY,):
            assert st.source_supports(source, shot_type)

    def test_unknown_source_is_not_silenced(self):
        assert st.source_supports("some_future_source", "scene")


class TestWiredIntoSelection:
    def test_scene_query_never_reaches_the_museums(self, monkeypatch, tmp_path):
        """Поведенческая проверка, не чтение исходника: на сценическом
        запросе музейный поиск не должен быть вызван ни разу."""
        called = []
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "")
        monkeypatch.setattr(ps, "_museum_search_photos",
                            lambda q, department=None: called.append(q) or [])
        monkeypatch.setattr(ps, "_openverse_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_pixabay_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_unsplash_search_photos", lambda q: [])
        ps._PEXELS_SEARCH_CACHE.clear()
        ps.pexels_photo("medieval castle moat water", 0, used_ids=set(), used_hashes=[],
                        text_key="routing-scene")
        assert called == []

    def test_object_query_reaches_the_museums_with_a_department(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "")
        monkeypatch.setattr(ps, "_museum_search_photos",
                            lambda q, department=None: seen.update(q=q, dep=department) or [])
        monkeypatch.setattr(ps, "_openverse_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_pixabay_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_unsplash_search_photos", lambda q: [])
        ps._PEXELS_SEARCH_CACHE.clear()
        ps.pexels_photo("medieval plate armour museum", 0, used_ids=set(), used_hashes=[],
                        text_key="routing-object")
        assert seen.get("dep") == 4

    def test_routing_version_is_in_the_selection_signature(self):
        assert str(ps.SHOT_TYPE_ROUTING_VERSION) in ps._selection_stack_signature()

    def test_department_is_part_of_the_museum_cache_key(self):
        """Тот же запрос со структурным сужением и без него — разные списки;
        общий ключ молча отдавал бы чужой."""
        import museum_sources as ms
        assert ms._disk_cache_key("sword", 4) != ms._disk_cache_key("sword", None)


class TestCachesCoverTheirBuilders:
    """Реальный дефект, пойманный на A/B 14.09: дисковые кэши хранят ГОТОВЫХ
    кандидатов, а ключ покрывал только запрос. После правки «превью не через
    API Openverse, а по конвенции Wikimedia» кэш продолжал отдавать
    кандидатов, собранных СТАРЫМ кодом, и превью снова уходили на
    api.openverse.org, где отвечали HTTP 424 — то есть правка молча не
    доходила до экрана, ровно как когда-то правки гейтов до
    candidate_gate_signature."""

    def test_openverse_key_changes_with_the_builder_code(self, monkeypatch):
        import stock_fetch_multisource as ov
        before = ps._openverse_cache_path("q", ov)
        monkeypatch.setattr(ps, "_OPENVERSE_MAPPING_SIG", [None])
        monkeypatch.setattr(ps, "wikimedia_thumb_url", lambda url, width=640: "другой код")
        after = ps._openverse_cache_path("q", ov)
        assert before != after

    def test_museum_key_changes_with_the_builder_code(self, monkeypatch):
        import museum_sources as ms
        before = ms._disk_cache_key("q", 4)
        monkeypatch.setattr(ms, "_MAPPING_SIG", [None])
        monkeypatch.setattr(ms, "_candidate", lambda *a, **k: {})
        after = ms._disk_cache_key("q", 4)
        assert before != after


class TestInflectionTailOnly:
    """Хвост префиксного совпадения ограничен СЛОВОИЗМЕНЕНИЕМ (17.09).

    MIN_PREFIX_LEN не закрывал класс: все ловушки, запертые тестом выше,
    короче пяти букв (coin, map, bow), а пятибуквенные термины сравнивались
    по началу слова с ЛЮБЫМ хвостом. Замер на брифах чужой ниши поймал
    реальный промах: «a blister pack of pills on a nightstand» -> scene,
    потому что night внутри nightstand. Предметный кадр, помеченный сценой,
    исключает музей и полку из пула — для ниши, чей предмет лежит именно в
    музейном каталоге, это потеря лучшего источника по совпадению букв.
    """

    @pytest.mark.parametrize("query,trap", [
        ("a blister pack of pills on a nightstand", "night/nightstand"),
        ("initially the room was empty", "initial/initially"),
    ])
    def test_non_inflectional_tail_does_not_match(self, query, trap):
        assert st.shot_type_for(query) == st.ANY, trap

    @pytest.mark.parametrize("query", [
        "medieval sabaton armoured foot",      # armour + ed
        "medieval blades collection",          # blade + s
        "knights marching column",             # march + ing
        "castle ruins overgrown",              # ruins — целым словом
    ])
    def test_documented_forms_still_match(self, query):
        """Нулевая регрессия: все совпадения, ради которых префикс вообще
        заведён, — словоизменение, и они обязаны сохраниться."""
        assert st.shot_type_for(query) != st.ANY

    def test_suffix_list_is_inflectional_only(self):
        """Список окончаний — словоизменение, не произвольные хвосты:
        «stand», «ly», «box» в нём быть не должно, иначе класс ловушки
        вернётся тихо."""
        for tail in ("stand", "ly", "box", "r"):
            assert tail not in st.INFLECTION_SUFFIXES


class TestSlotTypeBeatsQueryType:
    """ТИП КАДРА СЛОТА НАЗЫВАЕТ МОЗГ, а не словарь английских слов.

    Формат ответа режиссёра — «номер | тип | описание», то есть тип у слота
    уже есть. Раньше `write_inline()` записывал в сценарий только описание,
    и маршрут восстанавливался словарём по УЖЕ ОБРЕЗАННОМУ до пяти слов
    стоковому переводу брифа. Замер на реальном эпизоде 02 (142 брифа):
    у 13 (9%) тип при переводе МЕНЯЕТСЯ — 4 сценических становятся
    предметными (музей получает структурный запрос по отделу оружия на
    кадр «сапог, вылезающий из грязи»), 5 предметных теряют тип совсем
    (структурный запрос к Мет не строится вовсе — тот самый, что давал 277
    настоящих лат вместо керамических тарелок).

    На ЧУЖОЙ нише словарь не просто молчит, а ошибается: из 15 брифов
    психологии/медицины/каменного века/техники 13 дали `any`, и оба
    сработавших сработали неверно.
    """

    def test_hint_wins_over_lexicon(self):
        assert ps.slot_shot_type("a boot pulling out of deep thick mud", "scene") == "scene"
        # тот же бриф без подсказки словарь читает как предмет (sabaton/steel
        # нет, но `mud` есть) — проверяем, что подсказка реально решает
        assert ps.slot_shot_type("an armoured foot in a steel sabaton pressed "
                                 "into soft ground", "scene") == "scene"

    def test_full_brief_is_used_when_the_brain_did_not_say(self):
        assert ps.slot_shot_type("a two-handed sword with a long blade", None) == "object"

    def test_unknown_stays_unknown_instead_of_guessing(self):
        """`any` от словаря — отсутствие сигнала, и оно не имеет права
        выглядеть решением: вызывающий обязан остаться на типе ЗАПРОСА,
        то есть на сегодняшнем поведении."""
        assert ps.slot_shot_type("a stethoscope resting on a hospital bed", None) is None
        assert ps.slot_shot_type(None, None) is None
        assert ps.slot_shot_type("whatever", "not-a-shot-type") is None

    def test_declared_object_reaches_the_museum_though_the_section_query_is_a_scene(
            self, monkeypatch):
        """Поведенческая проверка: запрос секции сценический (музей по нему
        не спрашивается), а бриф слота объявлен предметным — музей обязан
        получить ЗАПРОС ИЗ БРИФА, и со структурным отделом."""
        seen = []
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "")
        monkeypatch.setattr(ps, "_museum_search_photos",
                            lambda q, department=None: seen.append((q, department)) or [])
        monkeypatch.setattr(ps, "_openverse_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_pixabay_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_unsplash_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_shelf_search_photos", lambda q, **kw: [])
        ps._PEXELS_SEARCH_CACHE.clear()
        ps.pexels_photo("medieval castle moat water", 0, used_ids=set(), used_hashes=[],
                        text_key="slot-type-object",
                        shot_brief="a two-handed sword with a long fullered blade",
                        shot_type_hint="object")
        assert seen, "музей не спрошен ни разу — запрос из брифа не дошёл"
        assert any(dep == 4 for _, dep in seen), seen

    def test_declared_scene_keeps_the_museum_out(self, monkeypatch):
        """Обратная сторона: бриф объявлен сценой, а его стоковый перевод
        словарь читает как предмет (измеренный случай «armoured foot in a
        steel sabaton») — музей не должен получить структурный запрос по
        отделу оружия на кадр земли."""
        seen = []
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "")
        monkeypatch.setattr(ps, "_museum_search_photos",
                            lambda q, department=None: seen.append((q, department)) or [])
        monkeypatch.setattr(ps, "_openverse_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_pixabay_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_unsplash_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_shelf_search_photos", lambda q, **kw: [])
        ps._PEXELS_SEARCH_CACHE.clear()
        brief = "an armoured foot in a steel sabaton pressed into soft ground"
        ps.pexels_photo("medieval battlefield churned mud", 0, used_ids=set(),
                        used_hashes=[], text_key="slot-type-scene",
                        shot_brief=brief, shot_type_hint="scene")
        brief_q = ps.brief_to_stock_query(brief, fallback=None)
        assert all(q != brief_q for q, _ in seen), seen

    def test_section_queries_keep_their_own_type(self, monkeypatch):
        """Переопределять тип ЗАПРОСА СЕКЦИИ догадкой о соседнем слоте
        нельзя: запросы секции обслуживают 4-10 слотов, и их текст про то,
        что в них написано. Сценический запрос секции остаётся вне музея
        даже когда бриф слота объявлен предметным."""
        seen = []
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "")
        monkeypatch.setattr(ps, "_museum_search_photos",
                            lambda q, department=None: seen.append(q) or [])
        monkeypatch.setattr(ps, "_openverse_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_pixabay_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_unsplash_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_shelf_search_photos", lambda q, **kw: [])
        ps._PEXELS_SEARCH_CACHE.clear()
        ps.pexels_photo("medieval castle moat water", 0, used_ids=set(), used_hashes=[],
                        text_key="slot-type-section",
                        shot_brief="a two-handed sword with a long fullered blade",
                        shot_type_hint="object")
        assert "medieval castle moat water" not in seen, seen


class TestShotTypeTravelsWithTheBrief:
    def test_type_prefix_is_parsed_out(self):
        assert script_parser.split_shot_brief(
            "object|a dented steel breastplate, close up") == (
            "object", "a dented steel breastplate, close up")

    @pytest.mark.parametrize("raw", [
        "a dented steel breastplate, close up",   # как писали 142 брифа эпизода 02
        "nonsense|something",                     # слева не тип кадра
        "a wall with a|b pattern",                # труба внутри описания
        "object|",                                # пустое описание
        "",
    ])
    def test_backward_compatible(self, raw):
        """Ни один существующий бриф не должен прочитаться иначе."""
        hint, brief = script_parser.split_shot_brief(raw)
        assert hint is None
        assert brief == raw.strip()

    def test_valid_types_are_not_a_second_copy_of_the_list(self):
        """Список имён типов живёт в shot_types и больше нигде: вторая копия
        разошлась бы при добавлении типа, и префикс молча стал бы частью
        описания."""
        for t in st.SHOT_TYPES:
            assert script_parser.split_shot_brief(f"{t}|a thing")[0] == t

    def test_parser_puts_the_hint_on_the_block(self, tmp_path):
        p = tmp_path / "script.txt"
        p.write_text("=== HOOK ===\n[shot:object|a plain steel helmet]Шлем. [pause]\n"
                     "[shot:a muddy field]Поле.\n", encoding="utf-8")
        blocks = script_parser.parse_blocks(str(p))
        assert blocks[0]["shot_type_hint"] == "object"
        assert blocks[0]["shot_brief"] == "a plain steel helmet"
        assert blocks[1]["shot_type_hint"] is None
        assert blocks[1]["shot_brief"] == "a muddy field"

    def test_real_episode_briefs_parse_unchanged(self):
        """Нулевая регрессия на настоящем эпизоде: 142 брифа без префикса."""
        path = os.path.join(REPO_ROOT, "videos", "02_ne-mechom", "script.txt")
        if not os.path.exists(path):
            pytest.skip("эпизода нет в поставке")
        blocks = script_parser.parse_blocks(path)
        with_brief = [b for b in blocks if b.get("shot_brief")]
        assert len(with_brief) == 142
        assert all(b.get("shot_type_hint") is None for b in with_brief)


class TestCompoundArmsTermsKeepTheirDepartment:
    """Найдено СВОЕЙ ЖЕ проверкой нулевой регрессии, а не рассуждением.

    Ограничение хвоста словоизменением отняло совпадение «arrow» внутри
    «arrowhead»: два реальных брифа эпизода 02 («a bodkin arrowhead close up
    beside a steel plate», «a bodkin arrowhead, close up») потеряли
    структурный запрос по отделу оружия, то есть уходили бы в Мет свободным
    текстом — ровно то, что замер 14.09 показал как источник керамических
    тарелок вместо лат. Починено явными формами (так этот модуль уже
    перечисляет ходовые слова), а не возвратом к совпадению с любым хвостом.
    """

    @pytest.mark.parametrize("brief", [
        "a bodkin arrowhead close up beside a steel plate",
        "a bodkin arrowhead, close up",
        "a bodkin arrowheads group",      # словоизменение поверх формы
    ])
    def test_arrowhead_still_gets_arms_and_armor(self, brief):
        assert st.met_department_for(brief, "object") == 4

    def test_the_trap_class_did_not_come_back(self):
        assert st.shot_type_for("a blister pack of pills on a nightstand") == st.ANY
