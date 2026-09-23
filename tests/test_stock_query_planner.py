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
    assert sqp.clean_query("a b c d e f g") is None
    assert sqp.clean_query("") is None


def test_answer_is_parsed_per_line_and_a_broken_line_costs_one_phrase():
    packet = {"units": [{"n": 1, "text": "a"}, {"n": 2, "text": "b"}]}
    raw = ("1 | rondel dagger closeup ; medieval dagger ; museum dagger ; dagger\n"
           "2 | ???\n")
    got = sqp.parse_answer(raw, packet)
    assert got == {1: ["rondel dagger closeup", "medieval dagger", "museum dagger", "dagger"]}


def test_prompt_carries_no_niche_words_and_takes_the_world_from_the_card(tmp_path):
    card = CARD
    d, blocks = _episode(tmp_path, card=card)
    gw = FakeGateway("1 | rondel dagger closeup ; medieval dagger\n2 | arrow hitting armor ; armour plate\n")
    sqp.plan_episode(d, blocks, gw, model="m", verbose=False)
    prompt = gw.prompts[0]
    assert "Setting: historical, 1300 AD-1500 AD" in prompt
    template = sqp.PROMPT.lower()
    for word in ("medieval", "knight", "sword", "armour", "europe"):
        assert word not in template, f"слово ниши «{word}» в шаблоне вопроса"


def test_plan_roundtrip_attaches_queries_by_phrase_text(tmp_path):
    d, blocks = _episode(tmp_path)
    gw = FakeGateway("1 | rondel dagger closeup ; medieval dagger\n2 | arrow hitting armor ; armour plate\n")
    assert sqp.plan_episode(d, blocks, gw, model="m", verbose=False) == 2
    plan = sqp.load(d)
    assert sqp.attach(blocks, plan) == 2
    by_text = {b["text"]: b.get("phrase_queries") for b in blocks}
    assert by_text["Вот кинжал."] == ["rondel dagger closeup", "medieval dagger"]
    assert by_text["Итог."] is None, "фраза без ответа идёт прежним путём"


def test_second_run_is_served_from_cache(tmp_path):
    d, blocks = _episode(tmp_path)
    gw = FakeGateway("1 | rondel dagger closeup\n2 | arrow hitting armor\n")
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
