"""Повторный запрос при зависании шлюза (прогон 30.09 на L40: короткие вопросы
судьи и отсева висели по 300-620 с при обычных 6-22 с)."""
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_gateway as lg  # noqa: E402
import source_health  # noqa: E402
from test_llm_gateway import CATALOG, Resp, http_error, ok  # noqa: E402

HEDGE = 0.2


class Opener:
    """Каталог на /models; вопросы чата — по очереди поведений: ответ, ошибка
    или ожидание события (зависший вызов)."""

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.chat_calls = 0
        self.lock = threading.Lock()

    def __call__(self, req, timeout):
        if req.full_url.endswith("/models"):
            return Resp(json.dumps(CATALOG).encode())
        with self.lock:
            self.chat_calls += 1
            b = self.behaviours.pop(0)
        if isinstance(b, tuple):          # ("hang", event, answer): ждать, потом ответить
            _tag, ev, then = b
            ev.wait(10)
            b = then
        if isinstance(b, Exception):
            raise b
        return Resp(json.dumps(b).encode())


@pytest.fixture(autouse=True)
def _fresh_health():
    source_health.reset_all()
    yield
    source_health.reset_all()


def ask(gw, hedge=HEDGE):
    return gw.chat("m/vision", [{"type": "text", "text": "q"}], 50, 1000, hedge_after=hedge)


def _wait(cond, sec=5):
    t = time.monotonic() + sec
    while not cond() and time.monotonic() < t:
        time.sleep(0.01)
    return cond()


def test_hung_call_is_answered_by_second_copy_and_loser_is_billed():
    release = threading.Event()
    op = Opener([("hang", release, ok("A", pt=100, ct=10)), ok("B", pt=100, ct=10)])
    gw = lg.Gateway(api_key="k", opener=op)
    t0 = time.monotonic()
    text, _u, price = ask(gw)
    assert text == "B" and time.monotonic() - t0 < 2, "ответ второй копии, без ожидания первой"
    assert gw.hedged == 1 and gw.hedge_won == 1
    release.set()                                   # зависшая копия всё-таки ответила
    assert _wait(lambda: gw.hedge_spent == price)
    assert gw.spent == 2 * price, "оплачены обе копии — потолок видит правду"
    assert _wait(lambda: gw.reserved == 0)


def test_fast_answer_sends_no_copy():
    op = Opener([ok("A")])
    gw = lg.Gateway(api_key="k", opener=op)
    assert ask(gw)[0] == "A"
    assert op.chat_calls == 1 and gw.hedged == 0 and gw.reserved == 0


def test_without_hedge_after_path_is_unchanged():
    release = threading.Event()
    op = Opener([("hang", release, ok("A"))])
    gw = lg.Gateway(api_key="k", opener=op)
    threading.Timer(0.5, release.set).start()
    assert ask(gw, hedge=None)[0] == "A"
    assert op.chat_calls == 1 and gw.hedged == 0


def test_no_copy_when_it_does_not_fit_the_spend_cap():
    release = threading.Event()
    op = Opener([("hang", release, ok("A", pt=100, ct=10))])
    reserve = 1000 * 0.5 + 50 * 2               # резерв одного вопроса
    gw = lg.Gateway(api_key="k", opener=op, spend_cap=int(reserve * 1.5))
    threading.Timer(0.6, release.set).start()
    assert ask(gw)[0] == "A"
    assert op.chat_calls == 1 and gw.hedged == 0, "вторая копия вышла бы за потолок — её нет"


def test_first_failing_after_copy_sent_returns_the_copy():
    release = threading.Event()
    op = Opener([("hang", release, http_error(400, {"error": {"code": "bad"}})), ok("B")])
    gw = lg.Gateway(api_key="k", opener=op)
    threading.Timer(0.5, release.set).start()
    assert ask(gw)[0] == "B"
    assert gw.hedge_won == 1
    assert _wait(lambda: gw.reserved == 0)


def test_cancelled_loser_does_not_pause_the_gateway(monkeypatch):
    monkeypatch.setattr(lg.time, "sleep", lambda s: None)
    release = threading.Event()
    boom = TimeoutError("hang")
    op = Opener([("hang", release, boom), ok("B")] + [boom] * 10)
    gw = lg.Gateway(api_key="k", opener=op)
    t0 = time.monotonic()
    assert ask(gw)[0] == "B"
    assert time.monotonic() - t0 < 2, "ответ второй копии, без ожидания зависшей"
    release.set()
    assert _wait(lambda: gw.reserved == 0)
    time.sleep(0.2)
    h = gw.health()
    assert h.fails == 0 and not h.cooling(), "сбой ненужной копии не ставит шлюз на паузу"
    assert op.chat_calls == 2, "отменённая копия новых попыток не делает"
