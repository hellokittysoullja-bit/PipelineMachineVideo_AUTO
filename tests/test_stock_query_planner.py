#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запросы к стокам на каждую фразу: план по главам, проверка ответа
модели, подключение к слоту и бюджет квоты Pexels. Без сети."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import stock_query_planner as sqp  # noqa: E402

SCRIPT = """=== METADATA ===
TITLE: Тест
=== HOOK ===
[shot:a medieval rondel dagger, studio shot]Вот кинжал.[pause]
[shot:an arrow glancing off a dented steel breastplate]Стрела скользит по нагруднику.[pause]
=== FINAL ===
Итог.[pause]
"""


CARD = {"schema_version": 1, "register": "historical", "era": {"from": 1300, "to": 1500}, "culture": {"include": [], "exclude": ["asian", "japanese", "chinese", "ottoman", "islamic"]}, "must_not_show": ["modern tactical knife", "modern military uniform", "napoleonic uniform", "firearm", "gunpowder era"], "expected_subjects": ["rondel dagger", "knight", "plate armour", "cavalry", "arrow", "muddy battlefield"], "era_anchor_terms": ["medieval", "knight", "14th century", "15th century"]}


class FakeGateway:
    """Отвечает на главу хука; на остальные главы молчит (пустой ответ)."""

    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def chat(self, model, content, max_tokens, est):
        text = content[0]["text"]
        self.prompts.append(text)
        return (self.answer if "Вот кинжал" in text else ""), {}, 1


def _episode(tmp_path, card=None):
    d = tmp_path / "ep"
    (d / "media_plan").mkdir(parents=True)
    (d / "script.txt").write_text(SCRIPT, encoding="utf-8")
    if card:
        (d / "media_plan" / "world_card.json").write_text(json.dumps(card), encoding="utf-8")
    import script_parser
    return str(d), script_parser.parse_blocks(str(d / "script.txt"))


def test_clean_query_keeps_only_short_latin_queries():
    assert sqp.clean_query(' "Medieval Battle Reenactment". ') == "medieval battle reenactment"
    assert sqp.clean_query("1. knight armour mud") == "knight armour mud"
    assert sqp.clean_query("рыцарь в грязи") is None
    # Запрос-знание называет работу архивным названием: до 7 слов, с годом.
    assert sqp.clean_query("Battle of Poitiers 1356 miniature") == "battle of poitiers 1356 miniature"
    assert sqp.clean_query("a b c d e f g") == "a b c d e f g"
    assert sqp.clean_query("a b c d e f g h") is None
    assert sqp.clean_query("") is None


def test_prompt_carries_no_niche_words_and_takes_the_world_from_the_card(tmp_path):
    card = CARD
    d, blocks = _episode(tmp_path, card=card)
    gw = FakeGateway("")
    sqp.plan_episode(d, blocks, gw, model="m", verbose=False)
    prompt = gw.prompts[0]
    assert "Setting: historical, 1300 AD-1500 AD" in prompt
    template = sqp.SPEC_PROMPT.lower()
    for word in ("medieval", "knight", "sword", "armour", "europe"):
        assert word not in template, f"слово ниши «{word}» в шаблоне вопроса"


DAGGER = ('{"n": 1, "focus": "a medieval rondel dagger", "core": "a rondel dagger is visible",'
          ' "claims": [{"id": "c2", "text": "a plain background", "tier": "should"}],'
          ' "queries": [{"q": "museum dagger", "for": ["c2"]}, {"q": "rondel dagger closeup", "for": ["core"]},'
          ' {"q": "medieval dagger", "for": ["core", "c2"]}]}\n')
ARROW = ('{"n": 2, "focus": "an arrow glancing off plate armour", "core": "an arrow is visible",'
         ' "claims": [{"id": "c1", "text": "the arrow glances off armour", "tier": "must", "motion": true}],'
         ' "queries": [{"q": "arrow hitting armor", "for": ["core", "c1"]}, {"q": "archer shooting", "for": ["core"]}]}\n')


def test_plan_roundtrip_attaches_queries_and_spec_by_phrase_text(tmp_path):
    d, blocks = _episode(tmp_path)
    gw = FakeGateway(DAGGER + ARROW)
    assert sqp.plan_episode(d, blocks, gw, model="m", verbose=False) == 2
    assert sqp.attach(blocks, sqp.load(d), sqp.load_specs(d)) == 2
    by_text = {b["text"]: b for b in blocks}
    assert by_text["Вот кинжал."]["phrase_queries"] == ["rondel dagger closeup", "medieval dagger",
                                                        "museum dagger"], "сначала запросы фокуса"
    arrow = by_text["Стрела скользит по нагруднику."]["shot_spec"]
    assert arrow["focus"] == "an arrow glancing off plate armour"
    assert sqp.has_motion(arrow) and not sqp.has_motion(by_text["Вот кинжал."]["shot_spec"])
    assert "phrase_queries" not in by_text["Итог."] and "shot_spec" not in by_text["Итог."], \
        "фраза без ответа идёт прежним путём"


def test_second_run_is_served_from_cache(tmp_path):
    d, blocks = _episode(tmp_path)
    gw = FakeGateway(DAGGER)
    sqp.plan_episode(d, blocks, gw, model="m", verbose=False)
    hook = [p for p in gw.prompts if "Вот кинжал" in p]
    sqp.plan_episode(d, blocks, gw, model="m", verbose=False)
    assert [p for p in gw.prompts if "Вот кинжал" in p] == hook, "отвеченная глава из кэша"
    assert len(gw.prompts) > len(hook) + 1, "пустой ответ не кэшируется — глава спрашивается снова"


def test_broken_plan_file_is_empty_plan_not_a_crash(tmp_path):
    d, _blocks = _episode(tmp_path)
    open(os.path.join(d, "media_plan", sqp.PLAN_NAME), "w").write("{not json")
    assert sqp.load(d) == {}


def test_slot_takes_its_phrase_query_first_and_keeps_the_section_pool():
    blocks = [{"text": "a", "phrase_queries": ["q1", "q2", "q3"]}, {"text": "b"}]
    assert ps.apply_phrase_queries(blocks, ["sec a", "sec b"]) == ["q1", "sec b"]
    assert ps.slot_extra_queries(blocks[0], ["s1", "q3"]) == ["q2", "q3", "s1"]
    assert ps.slot_extra_queries(blocks[1], ["s1"]) == ["s1"], "без плана — прежний пул секции"


def test_pexels_budget_spares_the_slot_own_queries(monkeypatch):
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", 10)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_RESERVE", 20)
    monkeypatch.setattr(ps, "PEXELS_LOW_PRIORITY_SKIPPED", 0)
    assert ps.pexels_query_allowed("own", {}, low_priority=False)
    assert ps.pexels_query_allowed("cached", {"cached": []}, low_priority=True), "кэш бесплатен"
    assert not ps.pexels_query_allowed("extra", {}, low_priority=True)
    assert ps.PEXELS_LOW_PRIORITY_SKIPPED == 1
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", 50)
    assert ps.pexels_query_allowed("extra", {}, low_priority=True)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", None)
    assert ps.pexels_query_allowed("extra", {}, low_priority=True), "остаток неизвестен — не режем"


def test_quota_is_read_from_the_response_header():
    class R:
        headers = {"X-Ratelimit-Remaining": "137"}
    ps.PEXELS_QUOTA_LEFT = None
    ps._note_pexels_quota(R())
    assert ps.PEXELS_QUOTA_LEFT == 137
    ps.PEXELS_QUOTA_LEFT = None


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _video_world import QUERY, infra, video  # noqa: E402,F401


def test_video_path_spends_pexels_quota_on_own_query_only(infra, monkeypatch):
    asked = []

    def search(q):
        asked.append(q)
        return [video(1)]
    monkeypatch.setattr(ps, "_pexels_search_videos", search)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", 5)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_RESERVE", 50)
    import dataclasses
    import selection_engine
    f = {x.name: None for x in dataclasses.fields(selection_engine.SlotRequest)}
    f.update(index=0, query=QUERY, extra_queries=("reenactment knight fall",), is_opening=False,
             director_assist=False)
    req = ps.build_slot_request(**f)
    ps.VIDEO_ADAPTER.sources(req, QUERY)
    ps.VIDEO_ADAPTER.sources(req, "reenactment knight fall")
    assert len(asked) == 1 and "reenactment" not in asked[0]


def test_spec_line_is_parsed_alone_and_bad_specs_drop():
    packet = {"units": [{"n": k} for k in range(1, 7)] + [{"n": 9}]}
    core = '"core": "a knight kneeling in armour"'
    ok_q = '"queries": [{"q": "knight", "for": ["core"]}]'
    raw = (
        '{"n": 1, "focus": "a knight kneeling on grass", ' + core + ', "claims": ['
        '{"id": "c1", "text": "grass field", "tier": "should"}, {"id": "c1", "text": "a duplicate id",'
        ' "tier": "must"}, {"id": "c3", "text": "sky above", "tier": "maybe"}],'
        ' "queries": [{"q": "knight kneeling", "for": ["core", "zz"]}, {"q": "Knight \\"1\\"", "for": ["core"]},'
        ' {"q": "field grass", "for": ["zz"]}]}\n'
        'garbage line\n'
        '{"n": 2, "focus": "рыцарь на коленях", ' + core + ', "claims": [], ' + ok_q + '}\n'
        '{"n": 3, "focus": "a knight kneeling", "claims": [{"id": "c1", "text": "grass field", "tier": "must"}], '
        + ok_q + '}\n'
        '{"n": 4, "focus": "a knight kneeling", ' + core + ', "claims": [{"id": "c2", "text": "grass field",'
        ' "tier": "should"}], "queries": [{"q": "green grass", "for": ["c2"]}]}\n'
        '{"n": 5, "focus": "a knight kneeling", ' + core + ', "claims": [{"id": "c2", "text": "he moves forward",'
        ' "tier": "must", "motion": true}, {"id": "c3", "text": "he falls down", "tier": "must", "motion": true}], '
        + ok_q + '}\n'
        '{"n": 9, "focus": "a knight kneeling", ' + core + ', "claims": [], ' + ok_q + '}\n')
    got = sqp.parse_spec(raw, packet)
    assert sorted(got) == [1, 5, 9], "кириллица, нет главного, ни одного запроса главного — фраза выпадает"
    assert [c.get("motion", False) for c in got[5]["claims"]] == [False, True, False], \
        "лишний флаг движения снимается, фраза остаётся"
    assert [c["id"] for c in got[1]["claims"]] == ["core", "c1"], "главное первым; повтор id и чужой tier отброшены"
    assert got[1]["claims"][0] == {"id": "core", "text": "a knight kneeling in armour", "tier": "must"}
    assert got[1]["queries"] == [{"q": "knight kneeling", "for": ["core"]}], "цель без утверждения — не цель"


def test_model_cannot_smuggle_its_own_core_claim():
    raw = ('{"n": 1, "focus": "a ball bouncing", "core": "a ball is visible", "claims": [{"id": "core",'
           ' "text": "a wall is visible", "tier": "must"}], "queries": [{"q": "ball", "for": ["core"]}]}')
    got = sqp.parse_spec(raw, {"units": [{"n": 1}]})
    assert got[1]["claims"] == [{"id": "core", "text": "a ball is visible", "tier": "must"}]


def test_query_target_given_as_a_string_is_accepted():
    got = sqp.parse_spec('{"n": 1, "focus": "a b c", "core": "a b", "claims": [],'
                         ' "queries": [{"q": "a b", "for": "core"}]}', {"units": [{"n": 1}]})
    assert got[1]["queries"] == [{"q": "a b", "for": ["core"]}]


def test_unknown_unit_number_is_ignored():
    got = sqp.parse_spec('{"n": 7, "focus": "a b c", "core": "a b", "claims": [],'
                         ' "queries": [{"q": "a b", "for": ["core"]}]}', {"units": [{"n": 1}]})
    assert got == {}


def test_queries_ordered_by_importance_of_what_they_look_for():
    spec = {"claims": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "queries": [{"q": "x", "for": ["c"]}, {"q": "y", "for": ["b", "c"]}, {"q": "z", "for": ["a"]}]}
    assert [x["q"] for x in sqp.order_queries(spec)] == ["z", "y", "x"]


def test_old_plan_version_gives_no_specs(tmp_path, capsys):
    d, _blocks = _episode(tmp_path)
    json.dump({"version": 2, "units": {"k": {"text": "t", "queries": ["a b"], "rungs": [], "focus": "f",
                                             "claims": [{"id": "c1"}]}}},
              open(os.path.join(d, "media_plan", sqp.PLAN_NAME), "w"))
    assert sqp.load_specs(d) == {}
    assert "версия 2" in capsys.readouterr().out
    assert sqp.load(d) == {"k": ["a b"]}, "запросы старого плана по-прежнему читаются"


def test_prompt_never_tells_the_model_to_drop_the_subject():
    t = sqp.SPEC_PROMPT.lower()
    assert "without the action" not in t and "substitute" not in t
    assert "never make words, captions" in t, "текст в кадре запрещён — правило терялось при переписывании"
