#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Спекулятивные вызовы шлюза (упреждающий отбор слотов): настоящий цикл
видит те же деньги, отказы и ответы, что без упреждения."""
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import llm_gateway as lg  # noqa: E402
from test_llm_gateway import EMPTY as EMPTY_ANSWER, Opener, chat_calls, no_sleep, ok  # noqa: E402,F401

Q = [{"type": "text", "text": "question"}]


def spec_chat(gw, content=Q, model="m/vision"):
    with lg.speculation():
        return gw.chat(model, content, 50, 1000)


def test_speculative_answer_is_charged_only_when_the_real_loop_asks():
    op = Opener([ok("A", pt=100, ct=10)])
    gw = lg.Gateway(api_key="k", opener=op)
    assert spec_chat(gw)[0] == "A"
    assert gw.spent == 0 and gw.calls == 0, "упреждение не списывает с бюджета прогона"
    text, _u, price = gw.chat("m/vision", Q, 50, 1000)
    assert text == "A" and gw.spent == price == 70 and gw.calls == 1
    assert chat_calls(op) == 1, "настоящий цикл получил ответ из хранилища, без второго вызова"
    s = gw.summary()
    assert s["speculative_used"] == 1 and s["speculative_wasted"] == 0


def test_real_loop_refuses_on_cap_exactly_like_a_live_call():
    """Потолок проверяется в момент настоящего вопроса по тому же резерву,
    что у живого вызова: без упреждения вызов был бы отвергнут — и с ним тоже."""
    op = Opener([ok("A"), ok("B")])
    gw = lg.Gateway(api_key="k", opener=op, spend_cap=10_000)
    spec_chat(gw)
    gw.spent = 9_500                 # резерв вызова 1000*0.5+50*2 = 600 -> 10100 > cap
    with pytest.raises(lg.BudgetExhausted):
        gw.chat("m/vision", Q, 50, 1000)
    assert gw.spent == 9_500, "отказ ничего не списал"
    gw.spent = 0
    assert gw.chat("m/vision", Q, 50, 1000)[0] == "A", "ответ вернулся в хранилище после отказа"


def test_same_question_twice_in_speculation_is_paid_once():
    op = Opener([ok("A")])
    gw = lg.Gateway(api_key="k", opener=op)
    assert spec_chat(gw)[0] == "A" and spec_chat(gw)[0] == "A"
    assert chat_calls(op) == 1 and gw.spec_calls == 1


def test_unused_speculation_is_reported_as_waste():
    op = Opener([ok("A", pt=100, ct=10)])
    gw = lg.Gateway(api_key="k", opener=op)
    spec_chat(gw)
    assert gw.summary()["speculative_wasted"] == 70 and gw.spent == 0


def test_empty_answer_is_consumed_like_the_live_one_and_reask_goes_live():
    op = Opener([EMPTY_ANSWER, ok("fine")])
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.EmptyAnswer):
        spec_chat(gw)
    # Настоящий цикл: первый ответ — тот же пустой (списан, счётчики те же),
    # переспрос — живой вызов, как без упреждения.
    assert gw.chat("m/vision", Q, 50, 1000)[0] == "fine"
    assert gw.empty_answers == 1 and gw.empty_answer_reasked == 1 and chat_calls(op) == 2


def test_real_question_waits_for_the_same_speculative_call_in_flight():
    release = threading.Event()
    entered = threading.Event()

    class SlowOpener(Opener):
        def __call__(self, req, timeout):
            if "chat" in req.full_url:
                entered.set()
                release.wait(5)
            return super().__call__(req, timeout)
    op = SlowOpener([ok("A")])
    gw = lg.Gateway(api_key="k", opener=op)
    gw.billing("m/vision")
    t = threading.Thread(target=spec_chat, args=(gw,))
    t.start()
    assert entered.wait(5)
    got = []
    r = threading.Thread(target=lambda: got.append(gw.chat("m/vision", Q, 50, 1000)[0]))
    r.start()
    release.set()
    t.join(5)
    r.join(5)
    assert got == ["A"] and chat_calls(op) == 1, "второй оплаты за тот же вопрос нет"


def test_speculation_respects_the_cap_together_with_unused_answers():
    op = Opener([ok("A", pt=100, ct=10), ok("B")])
    gw = lg.Gateway(api_key="k", opener=op, spend_cap=1000)
    spec_chat(gw)                                # 70 оплачено и не использовано
    gw.spent = 300                               # 300 + 70 + резерв 600 = 970 <= 1000
    spec_chat(gw, content=[{"type": "text", "text": "other"}])
    with pytest.raises(lg.BudgetExhausted):      # 300 + 140 + 600 > 1000
        spec_chat(gw, content=[{"type": "text", "text": "third"}])


def test_without_speculation_nothing_changes():
    op = Opener([ok("A"), ok("A")])
    gw = lg.Gateway(api_key="k", opener=op)
    gw.chat("m/vision", Q, 50, 1000)
    gw.chat("m/vision", Q, 50, 1000)
    assert chat_calls(op) == 2 and gw.spec_calls == 0


def test_speculative_failures_never_pause_or_kill_the_gateway(monkeypatch):
    """Аудит 29.09: сбои упреждения входили в общий счёт пауз, и три паузы
    из-за упреждения выключали судью настоящему циклу — решение, которого
    без упреждения не было бы."""
    from test_llm_gateway import Clock, http_error
    Clock(monkeypatch)
    op = Opener([http_error(502)] * 1000)
    gw = lg.Gateway(api_key="k", opener=op)
    for _ in range(20):
        with pytest.raises(lg.GatewayError):
            spec_chat(gw, model="m/free")
    assert not gw.dead and gw.pauses == 0
    assert gw.health().cooldown_left() == 0, "упреждение не начинает паузу настоящему циклу"
