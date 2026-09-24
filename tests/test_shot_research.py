"""Второй круг поиска кадра: что видит мозг, что возвращает, что
сохраняется и что берётся с диска без нового вызова."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import shot_research as sr  # noqa: E402

SPEC = {"focus": "a knight collapsing into mud",
        "claims": [{"id": "core", "text": "a knight in armour is visible", "tier": "must"},
                   {"id": "c1", "text": "the knight falls into mud", "tier": "must"},
                   {"id": "c2", "text": "a muddy battlefield", "tier": "should"}],
        "queries": [{"q": "knight fallen mud", "for": ["core"]}, {"q": "medieval battle reenactment", "for": ["core"]}]}
PHRASE = "Рыцарь падает в грязь под тяжестью металла."


class FakeGW:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def chat(self, model, content, max_tokens, est):
        self.prompts.append(content[0]["text"])
        return self.answer, {}, 0


def test_prompt_carries_rejections_tried_queries_and_musts():
    p = sr.render_prompt(PHRASE, SPEC, "historical, 1300 AD-1500 AD",
                         ["knight fallen mud"], ["Roman soldiers standing, not a knight falling"])
    assert "Roman soldiers standing" in p and "knight fallen mud" in p
    assert "the knight falls into mud" in p and "a muddy battlefield" not in p, "should-пункты не обязательны"
    assert "historical, 1300 AD-1500 AD" in p


def test_parse_cleans_dedupes_and_drops_tried_queries():
    raw = ('Sure: {"queries": [{"q": "Battle of Agincourt St Albans Chronicle", "type": "illustration"},'
           ' {"q": "knight fallen mud", "type": "scene"}, {"q": "рыцарь", "type": "scene"},'
           ' {"q": "knight unhorsed miniature", "type": "weird"},'
           ' {"q": "battle of agincourt st albans chronicle"}, {"q": "a", "type": "scene"},'
           ' {"q": "b c"}, {"q": "d e"}]}')
    got = sr.parse(raw, tried=["knight fallen mud"])
    qs = [g["q"] for g in got]
    assert qs[0] == "battle of agincourt st albans chronicle" and got[0]["type"] == "illustration"
    assert "knight fallen mud" not in qs, "опробованный запрос не повторяется"
    assert "рыцарь" not in qs and qs.count("battle of agincourt st albans chronicle") == 1
    assert "type" not in got[1], "неизвестный тип кадра не сохраняется"
    assert len(got) == sr.MAX_NEW_QUERIES
    assert sr.parse("no json here") == [] and sr.parse('{"queries": "x"}') == []


def test_rejections_come_from_the_judge_log_of_this_slot_only():
    log = [{"index": 4, "verify": {"why": "Roman soldiers,  not a knight"}},
           {"index": 3, "verify": {"why": "other slot"}},
           {"index": 4, "verify": {"why": "Roman soldiers, not a knight"}},
           {"index": 4, "verify": None},
           {"index": 4, "verify": {"why": "a 3D character on red"}}]
    assert sr.rejections_from_log(log, 4) == ["Roman soldiers, not a knight", "a 3D character on red"]


def test_signature_ignores_judge_wording_but_not_what_was_searched():
    a = sr.signature("m", "s", PHRASE, SPEC, ["q1"])
    assert a == sr.signature("m", "s", PHRASE, SPEC, ["q1"])
    assert a != sr.signature("m", "s", PHRASE, SPEC, ["q1", "q2"])
    assert a != sr.signature("m2", "s", PHRASE, SPEC, ["q1"])


def test_second_run_takes_queries_from_disk_without_a_call(tmp_path):
    ans = '{"queries": [{"q": "froissart battle miniature", "type": "illustration"}, {"q": "tomb effigy knight", "type": "object"}]}'
    gw = FakeGW(ans)
    kw = dict(phrase=PHRASE, spec=SPEC, setting="s", tried=["knight fallen mud"], rejections=["r1"])
    items, origin = sr.new_queries(str(tmp_path), gw, "m", **kw)
    assert origin == "model" and [i["q"] for i in items] == ["froissart battle miniature", "tomb effigy knight"]
    saved = json.load(open(tmp_path / "media_plan" / sr.FILE_NAME, encoding="utf-8"))
    assert list(saved.values())[0]["rejections"] == ["r1"]
    gw2 = FakeGW("must not be asked")
    kw["rejections"] = ["другая формулировка судьи"]
    items2, origin2 = sr.new_queries(str(tmp_path), gw2, "m", **kw)
    assert origin2 == "disk" and items2 == items and gw2.prompts == []


def test_empty_answer_is_not_saved_so_next_run_asks_again(tmp_path):
    items, _ = sr.new_queries(str(tmp_path), FakeGW("nothing"), "m", phrase=PHRASE, spec=SPEC,
                              setting="s", tried=[], rejections=[])
    assert items == [] and not (tmp_path / "media_plan" / sr.FILE_NAME).exists()


def test_new_queries_become_spec_queries_that_look_for_the_core():
    assert sr.as_spec_queries([{"q": "a b", "type": "scene"}, {"q": "c d"}]) == [
        {"q": "a b", "for": ["core"], "type": "scene"}, {"q": "c d", "for": ["core"]}]


def test_research_request_uses_only_new_queries(monkeypatch, tmp_path):
    import pipeline_smart as ps
    import selection_engine
    monkeypatch.setenv("RESEARCH_ROUND", "1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-key")
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)
    gw = FakeGW('{"queries": [{"q": "froissart battle miniature", "type": "illustration"},'
                ' {"q": "tomb effigy knight", "type": "object"}]}')
    monkeypatch.setattr(ps, "_research_gateway", lambda: gw)
    ps.RESEARCH_ROUND_LOG.clear()
    ps.SHOT_JUDGE_LOG[:] = [{"index": 4, "verify": {"why": "Roman soldiers, not a knight"}}]
    req = ps.build_slot_request(
        index=4, query="knight fallen mud", extra_queries=["section query"], text_key="t",
        shot_brief=None, block_text=PHRASE, shot_spec=SPEC, arbiter_text=None, is_opening=False,
        slot_dur=3.0, action_qualifier=None, target_luma=None, director_score_fn=None,
        director_assist=False, director_report=None, video_score_fn=None, used_photo_ids=set(),
        used_video_ids=set(), used_hashes=[], recent_sizes=[])
    req2 = ps.research_round_request(4, {"text": PHRASE, "shot_spec": SPEC}, req)
    assert isinstance(req2, selection_engine.SlotRequest)
    assert req2.query == "froissart battle miniature" and req2.extra_queries == ("tomb effigy knight",)
    assert [q["q"] for q in req2.shot_spec["queries"]] == ["froissart battle miniature", "tomb effigy knight"]
    assert req2.used_photo_ids is req.used_photo_ids, "общее состояние эпизода — те же объекты"
    assert (req2.block_text, req2.shot_brief, req2.index) == (req.block_text, req.shot_brief, req.index), \
        "второй круг меняет только запросы: фраза и бриф — те же"
    assert "Roman soldiers, not a knight" in gw.prompts[0] and "section query" in gw.prompts[0]
    assert ps.RESEARCH_ROUND_LOG[-1]["queries"] == ["froissart battle miniature", "tomb effigy knight"]
    ps.SHOT_JUDGE_LOG.clear()


def test_research_request_is_inert_without_flag_key_or_spec(monkeypatch):
    import pipeline_smart as ps
    req = object()
    monkeypatch.setenv("RESEARCH_ROUND", "0")
    assert ps.research_round_request(1, {"shot_spec": SPEC}, req) is None
    monkeypatch.setenv("RESEARCH_ROUND", "1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "")
    assert ps.research_round_request(1, {"shot_spec": SPEC}, req) is None
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    assert ps.research_round_request(1, {"shot_spec": None}, req) is None


def test_gateway_failure_does_not_break_the_slot(monkeypatch, tmp_path):
    import pipeline_smart as ps
    monkeypatch.setenv("RESEARCH_ROUND", "1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("gateway down")
    monkeypatch.setattr(ps, "_research_gateway", lambda: Boom())
    ps.RESEARCH_ROUND_LOG.clear()
    req = ps.build_slot_request(
        index=0, query="q", extra_queries=[], text_key="t", shot_brief=None, block_text=PHRASE,
        shot_spec=SPEC, arbiter_text=None, is_opening=False, slot_dur=3.0, action_qualifier=None,
        target_luma=None, director_score_fn=None, director_assist=False, director_report=None,
        video_score_fn=None, used_photo_ids=set(), used_video_ids=set(), used_hashes=[], recent_sizes=[])
    assert ps.research_round_request(0, {"text": PHRASE, "shot_spec": SPEC}, req) is None
    assert "gateway down" in ps.RESEARCH_ROUND_LOG[-1]["error"]
