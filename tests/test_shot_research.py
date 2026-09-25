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

    def chat(self, model, content, max_tokens, est, reasoning=None):
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


# ---------- второй круг и для замены без главного фразы ----------

def test_weak_trigger_changes_the_question_and_the_signature():
    failed = sr.render_prompt(PHRASE, SPEC, None, ["q"], ["r"], trigger="failed")
    weak = sr.render_prompt(PHRASE, SPEC, None, ["q"], ["r"], trigger="weak")
    assert sr.VERDICTS["failed"] in failed and sr.VERDICTS["weak"] in weak
    assert sr.VERDICTS["failed"] not in weak
    assert sr.signature("m", None, PHRASE, SPEC, ["q"], "weak") != \
        sr.signature("m", None, PHRASE, SPEC, ["q"], "failed"), \
        "другой вопрос — другая запись на диске"


def test_weak_trigger_reaches_the_model_and_the_disk(tmp_path):
    gw = FakeGW('{"queries": [{"q": "dagger lying on open palm", "type": "object"}]}')
    items, origin = sr.new_queries(str(tmp_path), gw, "m", phrase=PHRASE, spec=SPEC, setting=None,
                                   tried=["q"], rejections=["a fist grip, not an open palm"],
                                   trigger="weak")
    assert origin == "model" and items[0]["q"] == "dagger lying on open palm"
    assert sr.VERDICTS["weak"] in gw.prompts[0]
    assert sr.load(str(tmp_path))[sr.unit_key(PHRASE)]["trigger"] == "weak"


def _att(tmp_path, *, media="x.jpg", verdicts=(), **notes):
    import selection_attempt
    a = selection_attempt.Attempt(7, "photo", str(tmp_path))
    a.media = media
    a.verdicts = list(verdicts)
    a.notes = dict(notes)
    return a


def test_research_trigger_failed_weak_or_none(tmp_path):
    import pipeline_smart as ps
    assert ps.research_trigger(None) == "failed"
    assert ps.research_trigger(_att(tmp_path, verdicts=[("judge", {"index": 7})])) == "failed"
    assert ps.research_trigger(_att(tmp_path, focus_met=False)) == "weak"
    assert ps.research_trigger(_att(tmp_path, focus_met=True)) is None
    assert ps.research_trigger(_att(tmp_path)) is None, "без судьи (нет focus_met) — не трогаем"


def test_weak_substitute_is_replaced_only_by_a_strictly_better_frame(tmp_path):
    import pipeline_smart as ps
    cur = _att(tmp_path, focus_met=False, quality=((1, 0, 1), 2, False))
    better = _att(tmp_path, media="y.jpg", quality=((1, 1, 0), 2, False))
    same = _att(tmp_path, media="z.jpg", quality=((1, 0, 1), 2, False))
    worse = _att(tmp_path, media="w.jpg", quality=((1, 0, 0), 3, False))
    bad = _att(tmp_path, media="v.jpg", quality=((1, 1, 1), 3, True), verdicts=[("judge", {})])
    assert ps.research_takes_over("weak", cur, better)
    assert not ps.research_takes_over("weak", cur, same), "ничья — остаётся прежний кадр"
    assert not ps.research_takes_over("weak", cur, worse)
    assert not ps.research_takes_over("weak", cur, bad), "брак не встаёт никогда"
    assert not ps.research_takes_over("weak", cur, _att(tmp_path, media=None, quality=None))
    # после провала первого круга годный кадр второго ставится всегда
    assert ps.research_takes_over("failed", None, same)
    assert not ps.research_takes_over("failed", None, bad)


def test_main_runs_the_second_round_through_the_trigger():
    import inspect
    import pipeline_smart as ps
    src = inspect.getsource(ps.main)
    assert "trigger = (research_trigger(cur_att)" in src
    assert "research_takes_over(trigger, cur_att, got_att)" in src


class ReasoningGW:
    """Двойник рассуждающей модели: если рассуждение не выключено явно, оно
    съедает любой запас выхода — пустой ответ, как у шлюза вживую (judge14:
    выход 1024 из 400; judge15: 4000 из 4000, всё рассуждение)."""

    def __init__(self, answer):
        self.answer, self.budgets, self.reasoning = answer, [], []

    def chat(self, model, content, max_tokens, est, reasoning=None, **kw):
        import llm_gateway
        self.budgets.append(max_tokens)
        self.reasoning.append(reasoning)
        if reasoning is not False:
            raise llm_gateway.EmptyAnswer(f"{model}: пустой ответ (finish_reason=length)")
        return self.answer, {}, 0


def test_reasoning_is_switched_off_so_the_answer_is_not_eaten(tmp_path):
    gw = ReasoningGW('{"queries": [{"q": "froissart battle miniature", "type": "illustration"}]}')
    items, origin = sr.new_queries(str(tmp_path), gw, "ds/deepseek-v4-flash", phrase=PHRASE, spec=SPEC,
                                   setting=None, tried=[], rejections=[])
    assert origin == "model" and items and items[0]["q"] == "froissart battle miniature"
    assert gw.reasoning == [False] and gw.budgets[0] >= 1000
