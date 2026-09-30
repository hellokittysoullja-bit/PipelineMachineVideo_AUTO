# -*- coding: utf-8 -*-
"""Цепочка запасных моделей для текстовых шагов (30.09). Живой прогон
videos/99_mify: DeepSeek на шлюзе отвечал 502, и один вызов висел 11-28
минут (повторы по 240 с, переспрос, паузы шлюза по 60 с), а замены модели не
было. Теперь 5xx, таймаут и пустой ответ — сразу следующая модель, без пауз."""
import io
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import llm_gateway as lg  # noqa: E402
import source_health  # noqa: E402

FLASH, PRO, QWEN = lg.TEXT_FALLBACK_CHAIN
CATALOG = {"data": [{"id": m, "billing": {"coefficient": {"input": 0.05, "output": 0.05}},
                     "capabilities": {"thinkingFormat": fmt}}
                    for m, fmt in ((FLASH, "deepseek"), (PRO, "deepseek"), (QWEN, "qwen"))]}


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b'{"error": {"code": "bad_gateway"}}'))


def ok(text):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class Opener:
    """Каталог на /models; на чат — ответ по модели из тела запроса."""

    def __init__(self, by_model):
        self.by_model = {m: list(v) for m, v in by_model.items()}
        self.chat = []

    def __call__(self, req, timeout):
        if req.full_url.endswith("/models"):
            return Resp(json.dumps(CATALOG).encode())
        body = json.loads(req.data.decode())
        self.chat.append((body["model"], timeout, body))
        a = self.by_model[body["model"]].pop(0)
        if isinstance(a, BaseException):
            raise a
        return Resp(json.dumps(a).encode())


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    slept = []
    monkeypatch.setattr(lg.time, "sleep", slept.append)
    monkeypatch.delenv("LLM_TEXT_FALLBACK", raising=False)
    source_health.reset_all()
    yield slept
    source_health.reset_all()


def _ask(gw, **kw):
    return lg.chat_fallback(gw, lg.text_chain(FLASH), [{"type": "text", "text": "q"}], 100, 50, **kw)


def test_chain_order_is_deepseek_first_then_other_provider():
    assert lg.text_chain(FLASH) == (FLASH, PRO, QWEN)
    assert lg.text_chain("x/other") == ("x/other", FLASH, PRO, QWEN)


@pytest.mark.parametrize("failure", [http_error(502), http_error(503), http_error(429),
                                     TimeoutError("timed out"),
                                     urllib.error.URLError(TimeoutError("timed out"))])
def test_primary_failure_goes_to_next_model_without_pauses(clock, failure):
    op = Opener({FLASH: [failure], PRO: [ok("ответ pro")]})
    gw = lg.Gateway(api_key="k", opener=op)
    got = _ask(gw, reasoning=False)
    assert got[0] == "ответ pro" and got.model == PRO and not got.from_cache
    assert [m for m, _t, _b in op.chat] == [FLASH, PRO], "одна попытка первичной, без повторов"
    assert sum(clock) == 0, "ни одной паузы: ни повтора, ни ожидания шлюза"
    assert all(t == lg.CHAIN_TIMEOUT_SEC for _m, t, _b in op.chat), "короткий таймаут шага"
    s = gw.summary()
    assert s["fallback_switches"] == 1 and s["fallback_answers"] == {PRO: 1}
    assert not gw.health().failed(), "отказ одной модели не ставит на паузу весь шлюз (судью)"


def test_reasoning_is_switched_off_in_each_models_own_format():
    op = Opener({FLASH: [http_error(502)], PRO: [http_error(502)], QWEN: [ok("q")]})
    gw = lg.Gateway(api_key="k", opener=op)
    assert _ask(gw, reasoning=False).model == QWEN
    bodies = {m: b for m, _t, b in op.chat}
    assert bodies[FLASH]["thinking"] == {"type": "disabled"}
    assert bodies[PRO]["thinking"] == {"type": "disabled"}
    assert bodies[QWEN]["reasoning"] == {"enabled": False}
    assert gw.summary()["fallback_switches"] == 2


def test_empty_answer_goes_to_next_model_without_reask():
    op = Opener({FLASH: [ok("")], PRO: [ok("ok")]})
    gw = lg.Gateway(api_key="k", opener=op)
    assert _ask(gw).model == PRO
    assert [m for m, _t, _b in op.chat] == [FLASH, PRO]


def test_connection_reset_is_retried_once_on_the_same_model():
    op = Opener({FLASH: [ConnectionResetError(), ok("ok")]})
    gw = lg.Gateway(api_key="k", opener=op)
    got = _ask(gw)
    assert got.model == FLASH and gw.summary()["fallback_switches"] == 0


def test_money_problems_do_not_walk_the_chain():
    op = Opener({FLASH: [urllib.error.HTTPError("u", 402, "x", {}, io.BytesIO(b"{}"))], PRO: [ok("x")]})
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.PaymentRequired):
        _ask(gw)
    assert [m for m, _t, _b in op.chat] == [FLASH]


def test_last_model_failure_is_raised():
    op = Opener({FLASH: [http_error(502)], PRO: [http_error(502)], QWEN: [http_error(502)]})
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.GatewayError):
        _ask(gw)


def test_primary_cache_first_fallback_cache_after_primary_fails():
    seen = []

    def cached(m):
        seen.append(m)
        return "из кэша pro" if m == PRO else None
    op = Opener({FLASH: [http_error(502)]})
    gw = lg.Gateway(api_key="k", opener=op)
    got = _ask(gw, cached=cached)
    assert got.from_cache and got.model == PRO and got[0] == "из кэша pro"
    assert seen == [FLASH, PRO], "кэш первичной смотрится первым, запасной — после её отказа"


def test_env_off_disables_the_chain(monkeypatch):
    monkeypatch.setenv("LLM_TEXT_FALLBACK", "off")
    assert lg.text_chain(FLASH) == (FLASH,)


# ------------------------------------------------------------------ вызывающие


def test_caption_screen_keeps_primary_cache_key_and_caches_fallback_under_its_name(tmp_path):
    import hashlib
    import caption_screen as cs
    text = "вопрос"
    old = hashlib.sha256((cs.MODEL + "\n" + text).encode("utf-8")).hexdigest()[:24] + ".json"
    assert os.path.basename(cs._cache_path(str(tmp_path), text)) == old, "кэш первичной в силе"
    op = Opener({FLASH: [http_error(502)], PRO: [ok("ответ pro")]})
    gw = lg.Gateway(api_key="k", opener=op)
    ans, _price, hit = cs._ask(gw, text, str(tmp_path))
    assert ans == "ответ pro" and not hit
    assert not os.path.exists(cs._cache_path(str(tmp_path), text)), "под первичной не записано"
    assert json.load(open(cs._cache_path(str(tmp_path), text, PRO)))["model"] == PRO
    op2 = Opener({FLASH: [http_error(502)]})
    ans2, _p, hit2 = cs._ask(lg.Gateway(api_key="k", opener=op2), text, str(tmp_path))
    assert ans2 == "ответ pro" and hit2, "повтор берёт ответ запасной из её кэша бесплатно"


def test_planner_answer_of_fallback_is_marked_and_primary_gets_a_new_chance(tmp_path):
    import stock_query_planner as sqp
    op = Opener({FLASH: [http_error(502)], PRO: [ok("ответ pro")]})
    text, hit, used = sqp.ask(lg.Gateway(api_key="k", opener=op), FLASH, "prompt", str(tmp_path))
    assert (text, hit, used) == ("ответ pro", False, PRO)
    assert op.chat[0][1] == sqp.CHAIN_TIMEOUT_SEC
    assert os.path.exists(sqp._cache_path(str(tmp_path), PRO, "prompt"))
    assert not os.path.exists(sqp._cache_path(str(tmp_path), FLASH, "prompt"))
    op2 = Opener({FLASH: [ok("ответ flash")]})
    text2, hit2, used2 = sqp.ask(lg.Gateway(api_key="k", opener=op2), FLASH, "prompt", str(tmp_path))
    assert (text2, used2) == ("ответ flash", FLASH), "первичная поднялась — ответ её, а не кэш запасной"


def test_research_round_uses_the_chain_and_signs_with_the_model_that_answered(tmp_path):
    import shot_research as sr
    spec = {"focus": "a knight falls", "claims": [{"id": "core", "text": "a knight falls", "tier": "must"}]}
    answer = '{"queries": [{"q": "knight unhorsed miniature", "type": "illustration"}]}'
    op = Opener({FLASH: [TimeoutError("timed out")], PRO: [ok(answer)]})
    items, origin = sr.new_queries(str(tmp_path), lg.Gateway(api_key="k", opener=op), FLASH,
                                   phrase="Рыцарь падает.", spec=spec, setting="s", tried=["x"],
                                   rejections=[])
    assert origin == "model" and items[0]["q"] == "knight unhorsed miniature"
    entry = next(iter(sr.load(str(tmp_path)).values()))
    assert entry["model"] == PRO and entry["sig"] == sr.signature(PRO, "s", "Рыцарь падает.", spec, ["x"])
    op2 = Opener({FLASH: [http_error(502)]})
    items2, origin2 = sr.new_queries(str(tmp_path), lg.Gateway(api_key="k", opener=op2), FLASH,
                                     phrase="Рыцарь падает.", spec=spec, setting="s", tried=["x"],
                                     rejections=[])
    assert origin2 == "disk" and items2 == items


def test_generation_brief_uses_the_chain(tmp_path):
    import shot_generator as sg
    spec = {"focus": "an hourglass", "claims": [{"id": "core", "text": "an hourglass", "tier": "must"}]}
    op = Opener({FLASH: [http_error(502)], PRO: [ok("a glass hourglass on a desk")]})
    desc, info = sg.describe(lg.Gateway(api_key="k", opener=op), FLASH, phrase="п", spec=spec,
                             brief=None, card=None, cache_dir=str(tmp_path))
    assert desc == "a glass hourglass on a desk" and info["model"] == PRO


def test_needs_planning_when_plan_holds_fallback_answers(tmp_path, monkeypatch):
    import stock_query_planner as sqp
    import shot_planner_llm
    import world_card
    monkeypatch.setattr(world_card, "load", lambda *a, **k: None)
    blocks = [{"text": "Фраза один."}]
    key = shot_planner_llm.unit_key(blocks[0]["text"])
    os.makedirs(tmp_path / "media_plan")

    def write(unit):
        with open(tmp_path / "media_plan" / sqp.PLAN_NAME, "w", encoding="utf-8") as f:
            json.dump({"version": sqp.PLAN_VERSION, "sig": sqp.plan_signature(FLASH, world_card.judge_setting(None)),
                       "units": {key: unit}}, f)
    write({"focus": "f", "claims": [], "model": FLASH})
    assert not sqp.needs_planning(str(tmp_path), blocks, FLASH)
    write({"focus": "f", "claims": []})
    assert not sqp.needs_planning(str(tmp_path), blocks, FLASH), "старый план без поля — первичная"
    write({"focus": "f", "claims": [], "model": PRO})
    assert sqp.needs_planning(str(tmp_path), blocks, FLASH)
