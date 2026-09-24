#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Паспорт мира и спецификации кадров — сам рендер, без ручного запуска;
мир эпизода — в музейный фильтр и в запросы. Без сети."""
import json
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import museum_sources as ms  # noqa: E402
import pipeline_smart as ps  # noqa: E402
import stock_query_planner as sqp  # noqa: E402
import world_card as wc  # noqa: E402

CARD = {"schema_version": 1, "register": "historical", "era": {"from": 1300, "to": 1500},
        "culture": {"include": [], "exclude": ["japanese"]}, "must_not_show": ["firearm"],
        "expected_subjects": ["rondel dagger", "arrow"], "era_anchor_terms": ["medieval"]}
SCRIPT = "=== METADATA ===\nTITLE: T\n=== HOOK ===\nВот кинжал.[pause]\nСтрела летит.[pause]\n"


class GW:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def chat(self, model, content, max_tokens, est, **kw):
        self.calls += 1
        return self.text, {}, 1


def _ep(tmp_path, script=SCRIPT):
    d = tmp_path / "ep"
    (d / "media_plan").mkdir(parents=True)
    (d / "script.txt").write_text(script, encoding="utf-8")
    return str(d)


def test_passport_is_made_by_the_model_and_kept_until_the_script_changes(tmp_path):
    d = _ep(tmp_path)
    gw = GW(json.dumps(CARD))
    card, what = wc.generate(d, gw, model="m")
    assert what == "made" and card["era"] == {"from": 1300, "to": 1500}
    assert json.load(open(wc.path(d)))["derived_by"] == "auto:m"
    assert wc.generate(d, gw, model="m")[1] == "fresh" and gw.calls == 1
    open(os.path.join(d, "script.txt"), "a", encoding="utf-8").write("Новая фраза.[pause]\n")
    assert wc.generate(d, gw, model="m")[1] == "made" and gw.calls == 2


def test_manual_passport_is_never_overwritten(tmp_path):
    d = _ep(tmp_path)
    wc.save(d, CARD, derived_by="claude-session")
    gw = GW(json.dumps(dict(CARD, era={"from": 1, "to": 2})))
    card, what = wc.generate(d, gw)
    assert what == "manual" and gw.calls == 0 and card["era"]["from"] == 1300


def test_broken_model_answer_keeps_the_previous_state(tmp_path):
    d = _ep(tmp_path)
    card, what = wc.generate(d, GW("not json"))
    assert card is None and what.startswith("failed") and not os.path.exists(wc.path(d))


def test_world_questions_only_when_there_is_a_world():
    assert wc.world_to_check(CARD)
    assert wc.world_to_check({"register": "scientific", "era": None, "culture": {}}) is None
    assert wc.is_historical(CARD) and not wc.is_historical({"register": "scientific"})


def test_museum_filter_passport_then_profile_then_nothing(monkeypatch):
    try:
        monkeypatch.setattr(ms, "_profile", lambda: {})
        ms.set_episode_world(None)
        assert ms.era_window() is None and ms.foreign_culture_terms() == ()
        assert ms.era_overlaps(-3000, -2900), "нет ни паспорта, ни эпохи канала — не фильтровать"
        monkeypatch.setattr(ms, "_profile", lambda: {"era_from": 900, "era_to": 1600})
        assert ms.era_window() == (900, 1600) and "egypt" in ms.foreign_culture_terms()
        ms.set_episode_world({"era": {"from": -3100, "to": 2026},
                              "culture": {"include": ["egyptian"], "exclude": ["roman"]}})
        terms = ms.foreign_culture_terms()
        assert ms.era_window() == (-3100, 2026)
        assert "egypt" not in terms and "roman" in terms and "japan" in terms, \
            "свою культуру паспорт снимает, чужую добавляет, остальной список канала остаётся"
    finally:
        ms.set_episode_world(None)


def test_spec_queries_go_to_stock_without_dictionary_appendices(monkeypatch):
    spec = {"queries": [{"q": "bulls bears battle", "for": ["core"]}]}
    req = types.SimpleNamespace(shot_spec=spec, action_qualifier="slashing")
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"register": "modern"})
    assert ps.stock_api_query(req, "bulls bears battle", video=True) == "bulls bears battle"
    monkeypatch.setattr(ps, "episode_world_card", lambda: CARD)
    got = ps.stock_api_query(req, "bulls bears battle", video=True)
    assert "slashing" not in got, "слова движения к запросу спецификации не приписываются никогда"
    plain = types.SimpleNamespace(shot_spec=None, action_qualifier="slashing")
    assert "slashing" in ps.stock_api_query(plain, "knight", video=True), "без спецификации — как раньше"


def test_art_museums_only_for_episodes_about_the_past(monkeypatch):
    for reg, ok in (("historical", True), ("mixed", True), ("modern", False),
                    ("scientific", False), ("abstract", False)):
        monkeypatch.setattr(ps, "episode_world_card", lambda r=reg: {"register": r})
        assert ps.art_museums_fit_episode() is ok
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)
    assert ps.art_museums_fit_episode()


def test_unchanged_phrases_keep_their_spec_when_a_chapter_is_replanned(tmp_path):
    import script_parser
    d = _ep(tmp_path)
    blocks = script_parser.parse_blocks(os.path.join(d, "script.txt"))
    ans = ('{"n": 1, "focus": "a dagger", "core": "a dagger is visible", "claims": [],'
           ' "queries": [{"q": "dagger", "for": ["core"]}]}\n'
           '{"n": 2, "focus": "an arrow", "core": "an arrow is visible", "claims": [],'
           ' "queries": [{"q": "arrow", "for": ["core"]}]}\n')
    assert sqp.plan_episode(d, blocks, GW(ans), model="m", verbose=False) == 2
    assert not sqp.needs_planning(d, blocks, model="m")
    first = json.load(open(os.path.join(d, "media_plan", sqp.PLAN_NAME)))["units"]
    open(os.path.join(d, "script.txt"), "w", encoding="utf-8").write(
        SCRIPT.replace("Стрела летит.", "Стрела летит в щит."))
    blocks = script_parser.parse_blocks(os.path.join(d, "script.txt"))
    assert sqp.needs_planning(d, blocks, model="m")
    other = ans.replace("a dagger is visible", "a knife is visible")
    sqp.plan_episode(d, blocks, GW(other), model="m", verbose=False)
    units = json.load(open(os.path.join(d, "media_plan", sqp.PLAN_NAME)))["units"]
    dagger = [v for v in units.values() if v["text"] == "Вот кинжал."][0]
    assert dagger == [v for v in first.values() if v["text"] == "Вот кинжал."][0], \
        "фраза не менялась — спецификация прежняя, кадры не перепокупаются"


def test_render_without_key_says_so_and_does_not_call_the_model(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    ps.auto_plan_episode([], video_dir=_ep(tmp_path))
    assert "нет LLM_GATEWAY_API_KEY" in capsys.readouterr().out


def test_repeated_query_fallback_comes_from_the_passport(monkeypatch):
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"expected_subjects": ["coral reef"]})
    assert ps.generic_fallback_queries_effective() == ("coral reef",)
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    body = src[src.index("def _diversify_repeated_query_runs"):][:6000]
    assert "GENERIC_FALLBACKS[" not in body


def test_passport_survives_edits_outside_the_narration(tmp_path):
    """Правка названия, анализа конкурентов или проставленные [shot:] мир не
    меняют — паспорт не перепокупается и строка мира в подписи плана та же."""
    d = _ep(tmp_path)
    gw = GW(json.dumps(CARD))
    assert wc.generate(d, gw, model="m")[1] == "made"
    text = SCRIPT.replace("TITLE: T", "TITLE: Другое название").replace(
        "Вот кинжал.", "[shot:a rondel dagger, close up]Вот кинжал.")
    open(os.path.join(d, "script.txt"), "w", encoding="utf-8").write(
        text + "=== TITLE OPTIONS ===\nвариант\n")
    assert wc.generate(d, gw, model="m")[1] == "fresh" and gw.calls == 1


def test_old_whole_file_digest_is_upgraded_without_a_model_call(tmp_path):
    d = _ep(tmp_path)
    wc.save(d, dict(CARD, script_sha=wc._script_digest(SCRIPT)), derived_by="auto:m")
    gw = GW(json.dumps(CARD))
    assert wc.generate(d, gw, model="m")[1] == "fresh" and gw.calls == 0
    assert json.load(open(wc.path(d)))["script_sha"] == wc._narration_digest(os.path.join(d, "script.txt"))


def test_world_fields_reach_the_selection_signature(monkeypatch):
    """Правка культур паспорта при тех же ловушках меняет, кого пропустит
    музейный фильтр, — прогретый кэш не должен отдавать прежний выбор."""
    monkeypatch.setattr(ps, "episode_world_card", lambda: CARD)
    monkeypatch.setattr(ps, "_CANDIDATE_GATE_SIG", None)
    a = ps.candidate_gate_signature()
    monkeypatch.setattr(ps, "episode_world_card",
                        lambda: dict(CARD, culture={"include": [], "exclude": ["japanese", "roman"]}))
    monkeypatch.setattr(ps, "_CANDIDATE_GATE_SIG", None)
    assert ps.candidate_gate_signature() != a


def test_failed_chapter_keeps_its_previous_specs(tmp_path):
    import llm_gateway
    import script_parser
    d = _ep(tmp_path)
    blocks = script_parser.parse_blocks(os.path.join(d, "script.txt"))
    ans = ('{"n": 1, "focus": "a dagger", "core": "a dagger is visible", "claims": [],'
           ' "queries": [{"q": "dagger", "for": ["core"]}]}\n'
           '{"n": 2, "focus": "an arrow", "core": "an arrow is visible", "claims": [],'
           ' "queries": [{"q": "arrow", "for": ["core"]}]}\n')
    assert sqp.plan_episode(d, blocks, GW(ans), model="m", verbose=False) == 2

    class Broken(GW):
        def chat(self, *a, **k):
            raise llm_gateway.GatewayError("503")
    import shutil
    shutil.rmtree(os.path.join(d, "media_plan", sqp.CACHE_DIR_NAME))
    assert sqp.plan_episode(d, blocks, Broken(""), model="m", verbose=False) == 2, \
        "разовый сбой шлюза не стирает спецификации главы"


def test_catalog_answers_only_within_the_episode_world(monkeypatch):
    import met_catalog
    rows = [{"id": 1, "dept": "Arms and Armor", "name": "Dagger", "cls": "", "tags": "", "title": "",
             "culture": "French", "b": 1450, "e": 1500},
            {"id": 2, "dept": "Arms and Armor", "name": "Dagger", "cls": "", "tags": "", "title": "",
             "culture": "Egyptian", "b": -1300, "e": -1200}]
    monkeypatch.setattr(met_catalog, "_load", lambda: {"rows": rows})
    ms.set_episode_world({"register": "historical", "era": {"from": -1500, "to": -1000},
                          "culture": {"include": ["egyptian"], "exclude": ["medieval"]}})
    try:
        assert [r["id"] for r in met_catalog.search("dagger")] == [2]
    finally:
        ms.set_episode_world(None)
    assert [r["id"] for r in met_catalog.search("dagger")] == [1], "канал: как раньше"


def test_european_qualifier_is_not_added_against_the_episode_world(monkeypatch):
    egypt = {"register": "historical", "era": {"from": -2600, "to": -30},
             "culture": {"include": ["egyptian"], "exclude": ["medieval", "european"]},
             "era_anchor_terms": ["ancient egyptian"]}
    monkeypatch.setattr(ps, "episode_world_card", lambda: egypt)
    assert "european" not in ps.disambiguate_search_query("egyptian spear warrior")
    monkeypatch.setattr(ps, "episode_world_card", lambda: CARD)
    assert "european" in ps.disambiguate_search_query("spear warrior"), "мир канала — как раньше"


def test_own_culture_of_the_episode_is_never_blocklisted(monkeypatch):
    monkeypatch.setattr(ps, "CONTENT_ALT_BLOCKLIST", ("korean", "anime"))
    monkeypatch.setattr(ps, "episode_world_card", lambda: dict(
        CARD, culture={"include": ["korean", "joseon"], "exclude": []}))
    assert ps.content_blocklist_effective() == ("anime",)
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)
    assert ps.content_blocklist_effective() == ("korean", "anime")
