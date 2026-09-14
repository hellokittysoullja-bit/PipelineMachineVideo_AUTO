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
