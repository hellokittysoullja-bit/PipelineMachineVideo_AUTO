#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Клиент шлюза моделей: деньги, повторы, отказы — без сети."""
import io
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import llm_gateway as lg  # noqa: E402
import source_health  # noqa: E402

CATALOG = {"data": [{"id": "m/vision", "billing": {"coefficient": {"input": 0.5, "output": 2}}},
                    {"id": "m/free", "billing": {"coefficient": {"input": 0, "output": 0}}},
                    {"id": "m/img", "billing": {"unit": "image", "base_tokens": 100000,
                                                "scales": {"quality": {"low": 0.25, "high": 4},
                                                           "size": {"1792x1024": 1.75}},
                                                "coefficient": {"input": 1.5, "output": 1.5}}},
                    {"id": "m/img-free", "billing": {"unit": "image", "base_tokens": 100000,
                                                     "coefficient": {"input": 0, "output": 0}}}]}


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body=None, headers=None):
    return urllib.error.HTTPError("u", code, "x", headers or {}, io.BytesIO(json.dumps(body or {}).encode()))


class Opener:
    """Отвечает каталогом на /models, дальше — очередью ответов чата."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def __call__(self, req, timeout):
        self.requests.append(req)
        if req.full_url.endswith("/models"):
            return Resp(json.dumps(CATALOG).encode())
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        if isinstance(a, Resp):
            return a
        return Resp(json.dumps(a).encode())


class Truncated(Resp):
    """Статус 200 получен, тело оборвалось на середине (живой случай)."""

    def read(self, *a):
        import http.client
        raise http.client.IncompleteRead(b'{"choi')


def ok(text="{}", pt=100, ct=10):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": pt, "completion_tokens": ct}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(lg.time, "sleep", lambda s: None)
    source_health.reset_all()   # регулятор шлюза общий на процесс
    yield
    source_health.reset_all()


class Clock:
    """Часы, которые идут только во сне: пауза шлюза проходит мгновенно,
    но проходит — иначе «переждать паузу» нечем проверить."""

    def __init__(self, monkeypatch):
        self.now = 1000.0
        self.slept = []
        monkeypatch.setattr(source_health.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(lg.time, "sleep", self.sleep)

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def chat_calls(op):
    return len([r for r in op.requests if "chat" in r.full_url])


def test_price_comes_from_the_catalog_and_actual_usage():
    op = Opener([ok(pt=100, ct=10)])
    gw = lg.Gateway(api_key="k", opener=op)
    text, _u, price = gw.chat("m/vision", [{"type": "text", "text": "q"}], 50, 1000)
    assert price == 100 * 0.5 + 10 * 2 and gw.spent == price
    assert op.requests[-1].get_header("User-agent") == lg.USER_AGENT, "Cloudflare режет стандартную подпись"
    assert op.requests[-1].get_header("Authorization") == "Bearer k"


def test_spend_cap_refuses_before_the_call():
    op = Opener([])
    gw = lg.Gateway(api_key="k", opener=op, spend_cap=100)
    with pytest.raises(lg.BudgetExhausted):
        gw.chat("m/vision", [], 50, 1000)       # резерв 1000*0.5+50*2 > 100
    assert not [r for r in op.requests if r.full_url.endswith("/chat/completions")]


def test_payment_required_kills_the_gateway_for_the_run():
    op = Opener([http_error(402, {"error": {"code": "payment_required", "request_id": "r1"}})])
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.PaymentRequired):
        gw.chat("m/free", [], 10, 10)
    with pytest.raises(lg.GatewayError):
        gw.chat("m/free", [], 10, 10)
    assert gw.dead and len([r for r in op.requests if "chat" in r.full_url]) == 1, "после 402 — ни одного вызова"


def test_rate_limit_and_server_errors_are_retried():
    op = Opener([http_error(429, headers={"Retry-After": "1"}), http_error(503), ok("fine")])
    gw = lg.Gateway(api_key="k", opener=op)
    assert gw.chat("m/free", [], 10, 10)[0] == "fine"


def test_bad_key_is_not_retried():
    op = Opener([http_error(401), ok()])
    with pytest.raises(lg.GatewayError):
        lg.Gateway(api_key="k", opener=op).chat("m/free", [], 10, 10)
    assert len([r for r in op.requests if "chat" in r.full_url]) == 1


def test_unknown_model_is_an_error_not_a_free_call():
    with pytest.raises(lg.GatewayError):
        lg.Gateway(api_key="k", opener=Opener([])).chat("m/unknown", [], 10, 10)


def test_no_key_means_not_configured(monkeypatch):
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    gw = lg.Gateway()
    assert not gw.configured
    with pytest.raises(lg.GatewayError):
        gw.chat("m/free", [], 10, 10)


def test_error_message_never_contains_the_key():
    op = Opener([http_error(400, {"error": {"code": "bad_request", "request_id": "r9"}})])
    with pytest.raises(lg.GatewayError) as e:
        lg.Gateway(api_key="sk-secret-value", opener=op).chat("m/free", [], 10, 10)
    assert "sk-secret-value" not in str(e.value) and "r9" in str(e.value)


def test_truncated_body_is_retried_and_counted_as_spent():
    """Живой прогон брифов упал целиком на http.client.IncompleteRead:
    исключения не было в списке повторяемых. Оборванный ответ сервис,
    скорее всего, уже списал — его резерв засчитывается в расход."""
    op = Opener([Truncated(), ok("fine", pt=100, ct=10)])
    gw = lg.Gateway(api_key="k", opener=op)
    text, _u, price = gw.chat("m/vision", [], 50, 1000)
    reserve = 1000 * 0.5 + 50 * 2
    assert text == "fine" and gw.lost_bodies == 1
    assert gw.spent == reserve + price and gw.reserved == 0


def test_truncated_bodies_exhaust_retries_with_a_gateway_error():
    op = Opener([Truncated() for _ in range(lg.MAX_ATTEMPTS)])
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.GatewayError, match="оборван"):
        gw.chat("m/free", [], 10, 10)
    assert gw.lost_bodies == lg.MAX_ATTEMPTS


def test_empty_answer_is_an_error_that_names_the_reason():
    """Рассуждающая модель израсходовала max_tokens на рассуждение: 13
    оплаченных вызовов, ноль ответов, и снаружи это выглядело как
    «модель промолчала». Теперь это ошибка с причиной, а деньги учтены."""
    empty = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                       "completion_tokens_details": {"reasoning_tokens": 50}}}
    gw = lg.Gateway(api_key="k", opener=Opener([empty]))
    with pytest.raises(lg.EmptyAnswer) as e:
        gw.chat("m/vision", [], 50, 1000)
    assert "length" in str(e.value) and "50" in str(e.value)
    assert gw.spent == 100 * 0.5 + 50 * 2 and gw.empty_answers == 1


def test_brief_brain_turns_a_chapter_failure_into_an_empty_chapter():
    import shot_brief_director as sbd

    class Gw:
        def __init__(self, exc):
            self.exc = exc

        def chat(self, *a, **k):
            raise self.exc
    assert sbd.GatewayBrain("m", gateway=Gw(lg.EmptyAnswer("пусто"))).ask("q", 3) == ""
    with pytest.raises(lg.PaymentRequired):
        sbd.GatewayBrain("m", gateway=Gw(lg.PaymentRequired("402"))).ask("q", 3)


# ------------------------------------------------------------------ картинки

def _img_answer(n=1):
    import base64
    return {"data": [{"b64_json": base64.b64encode(b"img%d" % k).decode()} for k in range(n)]}


def test_image_price_follows_the_catalog_ladder():
    gw = lg.Gateway(api_key="k", opener=Opener([]))
    assert gw.image_cost("m/img", "1792x1024") == 100000 * 1.75 * 1.5
    assert gw.image_cost("m/img", "1792x1024", quality="low") == 100000 * 0.25 * 1.75 * 1.5
    assert gw.image_cost("m/img", "999x999") == 150000, "нет в лестнице — множитель 1, как auto"
    assert gw.image_cost("m/img-free", "1792x1024") == 0
    with pytest.raises(lg.GatewayError):
        gw.image_cost("m/vision", "1024x1024")


def test_image_decodes_and_charges_what_arrived():
    op = Opener([_img_answer(1)])
    gw = lg.Gateway(api_key="k", opener=op)
    images, price = gw.image("m/img", "p", "1792x1024")
    assert images == [b"img0"] and price == gw.spent == 262500
    body = json.loads(op.requests[-1].data)
    assert body["response_format"] == "b64_json" and body["size"] == "1792x1024"


def test_image_spend_cap_refuses_before_the_call():
    op = Opener([])
    gw = lg.Gateway(api_key="k", opener=op, spend_cap=100000)
    with pytest.raises(lg.BudgetExhausted):
        gw.image("m/img", "p", "1792x1024")
    assert not [r for r in op.requests if "images" in r.full_url]


def test_image_answer_without_pictures_is_an_error_not_a_frame():
    gw = lg.Gateway(api_key="k", opener=Opener([{"data": []}]))
    with pytest.raises(lg.EmptyAnswer):
        gw.image("m/img-free", "p", "1024x1024")
    assert gw.spent == 0


def test_refused_prompt_is_not_retried():
    op = Opener([http_error(400, {"error": {"code": "content_policy", "request_id": "r"}}), _img_answer()])
    with pytest.raises(lg.GatewayError):
        lg.Gateway(api_key="k", opener=op).image("m/img-free", "p", "1024x1024")
    assert len([r for r in op.requests if "images" in r.full_url]) == 1


def test_hidden_per_call_overhead_cannot_blow_through_the_cap():
    """Живой случай 23.09: маршрут со скрытой надбавкой на вызов (оценка —
    сотни, факт — десятки тысяч) и параллельные сетки. Потолок проверялся по
    оценке, все вызовы проходили проверку разом, и прогон с потолком 80 тыс.
    потратил 647 тыс. Теперь первый вызов модели идёт один, его цена
    становится резервом следующих, и потолок держит."""
    import concurrent.futures
    op = Opener([ok(pt=50000, ct=10) for _ in range(8)])
    gw = lg.Gateway(api_key="k", opener=op, spend_cap=60000)
    content = [{"type": "text", "text": "q"}]

    def call(_):
        try:
            gw.chat("m/vision", content, 50, 1000)
            return "ok"
        except lg.BudgetExhausted:
            return "cap"
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(call, range(6)))
    assert gw.spent <= 60000, f"потрачено {gw.spent} при потолке 60000"
    assert res.count("ok") >= 1 and "cap" in res


def test_known_price_keeps_calls_parallel():
    """После первого вызова цена известна: дальше вызовы не сериализуются."""
    op = Opener([ok(pt=100, ct=10) for _ in range(3)])
    gw = lg.Gateway(api_key="k", opener=op)
    for _ in range(3):
        gw.chat("m/vision", [{"type": "text", "text": "q"}], 50, 1000)
    assert "m/vision" in gw._ratio and gw.spent == 3 * (100 * 0.5 + 10 * 2)


def test_calls_that_waited_for_the_first_price_then_run_in_parallel():
    """Первый вызов модели идёт один: его цена становится резервом следующих.
    Но потоки, что ждали его, дальше должны идти параллельно. Раньше каждый
    из них держал замок первого вызова весь свой вызов, и первая волна
    параллельных вызовов шла гуськом (замер 27.09: восемь проверок подписей
    DeepSeek-ом по одной за раз)."""
    import concurrent.futures
    import threading
    events, guard = [], threading.Lock()
    start = threading.Barrier(6)

    class SlowOpener(Opener):
        def __call__(self, req, timeout):
            if req.full_url.endswith("/models"):
                return super().__call__(req, timeout)
            with guard:
                events.append("start")
            threading.Event().wait(0.15)   # time.sleep заглушён фикстурой
            with guard:
                events.append("end")
            return super().__call__(req, timeout)

    gw = lg.Gateway(api_key="k", opener=SlowOpener([ok() for _ in range(6)]))

    def call(_):
        start.wait()
        return gw.chat("m/vision", [{"type": "text", "text": "q"}], 50, 1000)[0]

    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        assert list(ex.map(call, range(6))) == ["{}"] * 6
    assert events[:2] == ["start", "end"], "первый вызов модели обязан идти один"
    depth = peak = 0
    for e in events:
        depth += 1 if e == "start" else -1
        peak = max(peak, depth)
    assert peak >= 2, f"после первого вызова ждавшие шли гуськом: {events}"


def test_reasoning_switch_follows_the_models_thinking_format():
    """DeepSeek на длинном вопросе игнорирует reasoning.enabled=false и
    тратит весь выход на рассуждение (замер 24.09) — у него свой выключатель."""
    import llm_gateway
    assert llm_gateway.reasoning_switch("deepseek", False) == {"thinking": {"type": "disabled"}}
    assert llm_gateway.reasoning_switch("qwen", False) == {"reasoning": {"enabled": False}}
    assert llm_gateway.reasoning_switch(None, True) == {"reasoning": {"enabled": True}}


def test_a_call_during_a_pause_waits_it_out_instead_of_failing(monkeypatch):
    """judge14, слот 0: шлюз ушёл на паузу посреди проверки финалистов, и
    каждый вызов в паузе получал отказ — три настоящих кинжала остались без
    проверки, а меч, проверенный до сбоя, встал на экран. Вызывающие код
    чинили это сами, двумя разными копиями; теперь пауза пережидается в
    самом шлюзе, для всех вызовов."""
    clock = Clock(monkeypatch)
    op = Opener([ok("after pause")])
    gw = lg.Gateway(api_key="k", opener=op)
    gw.health("m/free").cooldown_until = clock.now + 42.0
    assert gw.chat("m/free", [], 10, 10)[0] == "after pause"
    assert clock.slept == [42.0] and gw.summary()["pause_wait_sec"] == 42.0


def test_exhausted_retries_are_asked_once_more():
    op = Opener([http_error(502)] * lg.MAX_ATTEMPTS + [ok("second try")])
    gw = lg.Gateway(api_key="k", opener=op)
    assert gw.chat("m/free", [], 10, 10)[0] == "second try"
    assert gw.reasked == 1 and chat_calls(op) == lg.MAX_ATTEMPTS + 1


def test_a_second_exhaustion_is_a_failure_not_an_endless_loop():
    op = Opener([http_error(502)] * (2 * lg.MAX_ATTEMPTS) + [ok("never")])
    with pytest.raises(lg.GatewayError, match="повторы исчерпаны"):
        lg.Gateway(api_key="k", opener=op).chat("m/free", [], 10, 10)
    assert chat_calls(op) == 2 * lg.MAX_ATTEMPTS


def test_a_lost_paid_body_is_not_asked_again():
    """Ответ оборвался после начала — сервис его, скорее всего, списал.
    Переспрос платил бы за тот же вопрос ещё до MAX_ATTEMPTS раз."""
    op = Opener([Truncated() for _ in range(lg.MAX_ATTEMPTS)] + [ok("never")])
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.GatewayError, match="оборван"):
        gw.chat("m/free", [], 10, 10)
    assert gw.reasked == 0 and chat_calls(op) == lg.MAX_ATTEMPTS


def test_a_dead_gateway_stops_costing_a_pause_per_call(monkeypatch):
    """Лежащий сервис: после GATEWAY_MAX_PAUSES пауз подряд без единого
    ответа шлюз засыпает, и следующий вызов до срока пробы отказывает
    сразу — без минуты ожидания и без запросов."""
    clock = Clock(monkeypatch)
    op = Opener([http_error(502)] * 1000)
    gw = lg.Gateway(api_key="k", opener=op)
    for _ in range(50):
        try:
            gw.chat("m/free", [], 10, 10)
        except lg.GatewayError:
            pass
        if gw.dead:
            break
    assert gw.dead and "пауз подряд" in gw.dead
    assert gw.pauses == lg.GATEWAY_MAX_PAUSES
    before, waited = chat_calls(op), len(clock.slept)
    with pytest.raises(lg.GatewayUnavailable, match="спит"):
        gw.chat("m/free", [], 10, 10)
    assert chat_calls(op) == before and len(clock.slept) == waited


def _asleep(monkeypatch, then):
    """Шлюз, уснувший после GATEWAY_MAX_PAUSES пауз; дальше отвечает then."""
    clock = Clock(monkeypatch)
    fails = [http_error(502)] * (4 * lg.GATEWAY_MAX_PAUSES * lg.GATEWAY_FAIL_THRESHOLD * lg.MAX_ATTEMPTS)
    op = Opener(fails)
    gw = lg.Gateway(api_key="k", opener=op)
    for _ in range(50):
        try:
            gw.chat("m/free", [], 10, 10)
        except lg.GatewayError:
            pass
        if gw.dead:
            break
    assert gw.dead
    op.answers[:] = list(then)
    return gw, op, clock


def test_an_asleep_gateway_wakes_after_one_probe(monkeypatch):
    """Живой случай 26.09: Qwen отвечал 503 три паузы подряд на втором слоте
    замера — дальше все слоты шли без судьи, хотя сервис вернулся. Теперь
    через GATEWAY_REVIVE_SEC один пробный вызов; ответил — работа дальше."""
    gw, op, clock = _asleep(monkeypatch, [ok("back"), ok("and again")])
    clock.sleep(lg.GATEWAY_REVIVE_SEC - 1)
    with pytest.raises(lg.GatewayUnavailable):
        gw.chat("m/free", [], 10, 10)
    clock.sleep(1)
    assert gw.chat("m/free", [], 10, 10)[0] == "back"
    assert not gw.dead and gw.revived == 1 and gw.summary()["revived"] == 1
    assert gw.chat("m/free", [], 10, 10)[0] == "and again"


def test_a_failed_probe_sleeps_again_and_costs_one_call(monkeypatch):
    gw, op, clock = _asleep(monkeypatch, [http_error(502)] * lg.MAX_ATTEMPTS + [ok("finally")])
    clock.sleep(lg.GATEWAY_REVIVE_SEC)
    before = chat_calls(op)
    with pytest.raises(lg.GatewayError):
        gw.chat("m/free", [], 10, 10)
    assert chat_calls(op) - before == lg.MAX_ATTEMPTS, "проба — один вызов со своими повторами, без переспроса"
    assert gw.dead and gw.revived == 0
    before = chat_calls(op)
    with pytest.raises(lg.GatewayUnavailable):
        gw.chat("m/free", [], 10, 10)
    assert chat_calls(op) == before, "после неудачной пробы — снова сон, без запросов"
    clock.sleep(lg.GATEWAY_REVIVE_SEC)
    assert gw.chat("m/free", [], 10, 10)[0] == "finally" and not gw.dead


def test_only_one_probe_at_a_time_and_no_money_never_wakes(monkeypatch):
    gw, _op, clock = _asleep(monkeypatch, [])
    clock.sleep(lg.GATEWAY_REVIVE_SEC)
    assert gw._admit("m/free") is True
    with pytest.raises(lg.GatewayUnavailable):
        gw._admit("m/free")
    gw._sleep_again("m/free")
    op = Opener([http_error(402)])
    broke = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.PaymentRequired):
        broke.chat("m/free", [], 10, 10)
    clock.sleep(10 * lg.GATEWAY_REVIVE_SEC)
    with pytest.raises(lg.GatewayError, match="до конца прогона"):
        broke.chat("m/free", [], 10, 10)
    assert chat_calls(op) == 1


def test_an_answer_resets_the_pause_count(monkeypatch):
    Clock(monkeypatch)
    fails = [http_error(502)] * (lg.GATEWAY_FAIL_THRESHOLD * 2 * lg.MAX_ATTEMPTS)
    op = Opener(fails + [ok("alive")] + fails + [ok("alive")])
    gw = lg.Gateway(api_key="k", opener=op)
    for _ in range(2):
        for _try in range(10):
            try:
                gw.chat("m/free", [], 10, 10)
                break
            except lg.GatewayError:
                assert not gw.dead
        else:
            raise AssertionError("шлюз так и не ответил")
    assert gw.pauses == 0 and not gw.dead


# ---------- дубль зависшего вызова (hedged request) ----------

class SlowFirst:
    """Первый вызов чата висит, пока тест его не отпустит; остальные отвечают
    сразу. Живой случай 27.09: 2 из 10 вызовов сетки по 110-120 с при
    остальных 8-12 с."""

    def __init__(self, first=None, rest=None):
        import threading
        self.release = threading.Event()
        self.first = first or ok("slow", pt=100, ct=10)
        self.rest = list(rest or [ok("fast", pt=100, ct=10)])
        self.chat_calls = 0
        self.requests = []
        self.first_done = threading.Event()

    def __call__(self, req, timeout):
        self.requests.append(req)
        if req.full_url.endswith("/models"):
            return Resp(json.dumps(CATALOG).encode())
        self.chat_calls += 1
        if self.chat_calls == 1:
            self.release.wait(10)
            try:
                if isinstance(self.first, Exception):
                    raise self.first
                return Resp(json.dumps(self.first).encode())
            finally:
                self.first_done.set()
        a = self.rest.pop(0)
        if isinstance(a, Exception):
            raise a
        return Resp(json.dumps(a).encode())


def test_stuck_call_is_hedged_and_the_first_answer_wins(monkeypatch):
    monkeypatch.setattr(lg, "HEDGE_SHORT_MAX_SEC", 0.05)
    op = SlowFirst()
    gw = lg.Gateway(api_key="k", opener=op)
    text, _u, price = gw.chat("m/vision", [], 50, 1000)
    assert text == "fast" and gw.hedged == 1 and gw.hedge_wins == 1
    # Опоздавший доходит сам и платится тогда, когда ответит: до ответа он
    # держит свой резерв, после — его цена в расходе.
    assert gw.reserved > 0
    op.release.set()
    op.first_done.wait(5)
    import threading
    for _ in range(200):
        if gw.reserved == 0:
            break
        threading.Event().wait(0.01)
    assert gw.reserved == 0 and gw.spent == 2 * price


def test_fast_call_is_not_hedged(monkeypatch):
    monkeypatch.setattr(lg, "HEDGE_SHORT_MAX_SEC", 5.0)
    op = Opener([ok("fine")])
    gw = lg.Gateway(api_key="k", opener=op)
    assert gw.chat("m/free", [], 10, 10)[0] == "fine"
    assert gw.hedged == 0 and chat_calls(op) == 1


def test_error_of_the_stuck_call_waits_for_the_hedge(monkeypatch):
    """Первый вызов упал уже после того, как ушёл дубль: ответ дубля, а не
    ошибка первого."""
    monkeypatch.setattr(lg, "HEDGE_SHORT_MAX_SEC", 0.05)
    import threading
    op = SlowFirst(first=http_error(400, {"error": {"code": "bad", "request_id": "r"}}))

    def later():
        # Настоящая пауза: time.sleep в этих тестах подменён (no_sleep).
        threading.Event().wait(0.2)
        op.release.set()
    import threading
    threading.Thread(target=later, daemon=True).start()
    gw = lg.Gateway(api_key="k", opener=op)
    assert gw.chat("m/vision", [], 50, 1000)[0] == "fast" and gw.hedged == 1


def test_both_copies_failing_raise_the_first_error(monkeypatch):
    monkeypatch.setattr(lg, "HEDGE_SHORT_MAX_SEC", 0.05)
    op = SlowFirst(first=http_error(400, {"error": {"code": "first", "request_id": "r1"}}),
                   rest=[http_error(400, {"error": {"code": "second", "request_id": "r2"}})])
    gw = lg.Gateway(api_key="k", opener=op)
    import threading

    def later():
        threading.Event().wait(0.3)
        op.release.set()
    import threading
    threading.Thread(target=later, daemon=True).start()
    with pytest.raises(lg.GatewayError, match="second"):
        gw.chat("m/vision", [], 50, 1000)


def test_long_answers_are_not_hedged_before_their_own_latency_is_known(monkeypatch):
    """Планировщик с рассуждением отвечает минутами: дубль по холодному
    порогу коротких вызовов стоил бы полной цены каждого его вызова."""
    monkeypatch.setattr(lg, "HEDGE_SHORT_MAX_SEC", 0.01)
    gw = lg.Gateway(api_key="k", opener=Opener([]))
    assert gw.hedge_after("m/vision", 16000) is None
    assert gw.hedge_after("m/vision", 500) == 0.01


def test_hedge_threshold_follows_the_median_of_the_same_kind():
    gw = lg.Gateway(api_key="k", opener=Opener([]))
    for s in (6.0, 7.0, 8.0, 9.0, 30.0):
        gw._note_latency("m/vision", 500, s)
    assert gw.hedge_after("m/vision", 500) == lg.HEDGE_FACTOR * 8.0
    # Другой род (длинный ответ) своих замеров не имеет — не заимствует.
    assert gw.hedge_after("m/vision", 16000) is None
    for s in (1.0,) * 5:
        gw._note_latency("m/vision", 100, s)
    assert gw.hedge_after("m/vision", 100) == lg.HEDGE_MIN_SEC


def test_short_answer_is_hedged_even_when_the_provider_is_slow_overall():
    """Медленный период (медиана 20 с) не должен отключать дубль: вызов
    короткого ответа без ответа дольше HEDGE_SHORT_MAX_SEC — завис (замер
    27.09: вызовы по 45 с при медиане 20 с держали круг проверки)."""
    gw = lg.Gateway(api_key="k", opener=Opener([]))
    for s in (18.0, 19.0, 20.0, 21.0, 22.0):
        gw._note_latency("m/vision", 500, s)
    assert gw.hedge_after("m/vision", 500) == lg.HEDGE_SHORT_MAX_SEC
    for s in (100.0, 110.0, 120.0, 130.0, 140.0):
        gw._note_latency("m/vision", 16000, s)
    assert gw.hedge_after("m/vision", 16000) == lg.HEDGE_FACTOR * 120.0, "длинный ответ — без потолка"


def test_hedging_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("GATEWAY_HEDGE", "0")
    gw = lg.Gateway(api_key="k", opener=Opener([]))
    assert gw.hedge_after("m/vision", 500) is None


def test_payment_required_is_not_hedged(monkeypatch):
    monkeypatch.setattr(lg, "HEDGE_SHORT_MAX_SEC", 5.0)
    op = Opener([http_error(402, {"error": {"code": "payment_required", "request_id": "r"}}), ok()])
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.PaymentRequired):
        gw.chat("m/free", [], 10, 10)
    assert gw.hedged == 0 and chat_calls(op) == 1


# ---------- сбой одной модели и шторм ошибок ----------

class ByModel:
    """Отвечает по модели запроса: сбой одной — не сбой другой."""

    def __init__(self, answers):
        self.answers = {m: list(a) for m, a in answers.items()}
        self.requests = []

    def __call__(self, req, timeout):
        self.requests.append(req)
        if req.full_url.endswith("/models"):
            return Resp(json.dumps(CATALOG).encode())
        model = json.loads(req.data.decode("utf-8"))["model"]
        a = self.answers[model].pop(0)
        if isinstance(a, Exception):
            raise a
        return Resp(json.dumps(a).encode())


def test_one_models_outage_does_not_pause_another(monkeypatch):
    """27.09: 502 от Qwen (судья) ставили на паузу весь адрес шлюза — вызовы
    DeepSeek (отсев по подписи, второй круг) ждали чужую паузу, и спящий
    из-за одной модели шлюз не пускал другую. Пауза и сон — у модели."""
    clock = Clock(monkeypatch)
    op = ByModel({"m/vision": [http_error(502)] * 1000, "m/free": [ok("free is fine")] * 3})
    gw = lg.Gateway(api_key="k", opener=op)
    for _ in range(50):
        try:
            gw.chat("m/vision", [], 10, 10)
        except lg.GatewayError:
            pass
        if gw.dead:
            break
    assert gw.dead and "m/vision" in gw.dead
    assert gw.health("m/vision").cooling() or gw._route("m/vision").dead
    slept = len(clock.slept)
    assert gw.chat("m/free", [], 10, 10)[0] == "free is fine"
    assert len(clock.slept) == slept, "здоровая модель не ждёт чужую паузу"
    with pytest.raises(lg.GatewayUnavailable):
        gw.chat("m/vision", [], 10, 10)


def test_attempt_outcomes_are_recorded_per_model():
    op = ByModel({"m/free": [http_error(502), ok("second")]})
    gw = lg.Gateway(api_key="k", opener=op)
    assert gw.chat("m/free", [], 10, 10)[0] == "second"
    assert list(gw._route("m/free").outcomes) == [1, 0]
    assert not gw._routes.get("m/vision")


def test_ordinary_error_rate_is_not_a_storm():
    """Обычная ночь 27.09: 8% попыток Qwen — 502. Это не шторм: дубль нужен
    ровно там, где 502 приходит через 41 с вместо ответа."""
    gw = lg.Gateway(api_key="k", opener=Opener([]))
    gw._route("m/vision").outcomes.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 2)
    assert not gw.storm("m/vision")
    assert gw.hedge_after("m/vision", 500) == lg.HEDGE_SHORT_MAX_SEC


def test_storm_is_detected_and_passes():
    gw = lg.Gateway(api_key="k", opener=Opener([]))
    r = gw._route("m/vision")
    r.outcomes.extend([1] * (lg.HEDGE_STORM_MIN - 1))
    assert not gw.storm("m/vision"), "пять ошибок подряд — ещё не статистика"
    r.outcomes.append(1)
    assert gw.storm("m/vision") and gw.hedge_after("m/vision", 500) is None
    r.outcomes.extend([0] * (lg.HEDGE_STORM_WINDOW // 2 + 1))
    assert not gw.storm("m/vision"), "ответы вернулись — дубль снова работает"
    assert not gw.storm("m/free"), "шторм — у своей модели"


def test_no_hedge_during_an_error_storm(monkeypatch):
    """Прерванный прогон 27.09: 78% попыток — 502/503, и дубли удваивали
    нагрузку на лежащего провайдера, ничего не ускоряя."""
    import threading
    monkeypatch.setattr(lg, "HEDGE_SHORT_MAX_SEC", 0.05)
    op = SlowFirst()
    gw = lg.Gateway(api_key="k", opener=op)
    gw._route("m/vision").outcomes.extend([1] * lg.HEDGE_STORM_WINDOW)

    def later():
        threading.Event().wait(0.3)
        op.release.set()
    threading.Thread(target=later, daemon=True).start()
    assert gw.chat("m/vision", [], 50, 1000)[0] == "slow"
    assert gw.hedged == 0 and op.chat_calls == 1 and gw.hedge_storm_skips >= 1


def test_a_storm_starting_while_waiting_cancels_the_backup(monkeypatch):
    """Порог дубля истёк, а модель тем временем ушла в шторм: первый вызов,
    скорее всего, не завис, а повторяет попытки — второй не уходит."""
    import threading
    monkeypatch.setattr(lg, "HEDGE_SHORT_MAX_SEC", 0.05)
    op = SlowFirst()
    gw = lg.Gateway(api_key="k", opener=op)
    seen = []

    def storm(model):
        seen.append(model)
        return len(seen) > 1          # до вызова — тихо, к порогу — шторм
    monkeypatch.setattr(gw, "storm", storm)

    def later():
        threading.Event().wait(0.3)
        op.release.set()
    threading.Thread(target=later, daemon=True).start()
    assert gw.chat("m/vision", [], 50, 1000)[0] == "slow"
    assert gw.hedged == 0 and op.chat_calls == 1 and gw.hedge_storm_skips == 1


# ---------------------------------------------------------------- картинка не дошла

IMAGE_REQUEST = [{"type": "text", "text": "Is there a dagger?"},
                 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]
# Дословно ответ Qwen из записи эп.94 (27.09): prompt_tokens 718 — вопрос без картинки.
DROPPED = "The image failed to upload. Please resend it."


def test_dropped_image_is_asked_again_and_both_calls_are_paid():
    op = Opener([ok(DROPPED, pt=718, ct=10), ok('{"main": "yes"}', pt=1500, ct=20)])
    gw = lg.Gateway(api_key="k", opener=op)
    text, _u, price = gw.chat("m/vision", IMAGE_REQUEST, 50, 1000)
    assert text == '{"main": "yes"}' and chat_calls(op) == 2
    assert gw.image_drops == 1 and gw.summary()["image_drops"] == 1
    assert gw.spent == (718 * 0.5 + 10 * 2) + (1500 * 0.5 + 20 * 2), "потерянный вызов оплачен — он в расходе"


def test_repeated_drops_end_as_a_gateway_failure_not_an_answer():
    """Повторы исчерпаны — это сбой шлюза (слот решится заново), а не ответ,
    который вызывающий разбирал бы как «неразобранный»."""
    op = Opener([ok(DROPPED, pt=718)] * (lg.IMAGE_DROP_RETRIES + 1))
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.ImageNotReceived) as e:
        gw.chat("m/vision", IMAGE_REQUEST, 50, 1000)
    assert isinstance(e.value, lg.GatewayError) and "не дошла" in str(e.value)
    assert chat_calls(op) == lg.IMAGE_DROP_RETRIES + 1 and gw.image_drops == lg.IMAGE_DROP_RETRIES + 1


def test_the_same_words_without_an_image_are_just_an_answer():
    op = Opener([ok(DROPPED)])
    gw = lg.Gateway(api_key="k", opener=op)
    assert gw.chat("m/vision", [{"type": "text", "text": "q"}], 50, 1000)[0] == DROPPED
    assert chat_calls(op) == 1 and gw.image_drops == 0


@pytest.mark.parametrize("answer", [
    '{"main_in_world": true, "background_foreign": false, "why": "I cannot see the image clearly, it is dark"}',
    "1: 3\n2: 0\n3: 1",
    "red",
    "The image shows a dagger lying on a table; no image of armour is visible.",
    "There is no image of a knight here, only a sword.",
    # Найдено собственным адверсариальным стресс-тестом (28.09, не живым
    # случаем) — компаунд-фраза с оговоркой о качестве, которая всё же
    # отвечает на вопрос: формально задевает "did not load" в старой версии
    # регэкспа, но модель продолжает описывать содержимое после "but" —
    # значит картинка дошла, и ложный повтор жёг бы вызов шлюза впустую.
    "The photo did not load properly for the camera due to motion blur, but the sword is visible.",
    "The image quality is poor but I can make out a blade shape.",
])
def test_ordinary_answers_about_the_picture_are_not_drops(answer):
    """Ответы судьи — JSON, строки оценок, описание кадра: ни одно не
    должно читаться как «картинка не дошла» (на 346 613 строках архива
    ответов и 1055 файлах кэша судьи правило не сработало ни разу)."""
    assert not lg.image_not_received(IMAGE_REQUEST, answer)


@pytest.mark.parametrize("answer", [
    DROPPED,
    "I'm sorry, but I can't see the image you attached.",
    "It seems no image was attached. Could you share it again?",
    "The picture didn't come through — please resend the image.",
    "There is no image attached to your message.",
    # Собственный адверсариальный стресс-тест 28.09 нашёл 7 реальных
    # промахов прежнего узкого регэкспа (7/15 фраз не ловились вовсе) —
    # добавлены как постоянные регрессионные случаи, не только разовая
    # проверка на бумаге.
    "I don't see any image attached to your request.",
    "The uploaded photo could not be loaded, could you try again?",
    "I wasn't able to load the photo you shared.",
    "Your image was not successfully uploaded on our end.",
    "Please re-upload the image as it was not received.",
    "It appears the image link is broken and nothing came through.",
    "Sorry, no attachment is visible in this conversation.",
])
def test_drop_phrasings_are_recognised(answer):
    assert lg.image_not_received(IMAGE_REQUEST, answer)


def test_vision_check_survives_a_dropped_canary():
    """Проверка зрения в начале прогона: потерянная картинка раньше читалась
    как «модель не видит картинок» — и судья выключался на весь прогон."""
    import shot_judge
    op = Opener([ok(DROPPED, pt=40, ct=10), ok("red", pt=90, ct=1), ok("blue", pt=90, ct=1)])
    gw = lg.Gateway(api_key="k", opener=op)
    assert shot_judge.vision_check(gw, "m/vision") == (True, "")


def test_vision_check_that_never_gets_the_image_is_a_gateway_failure():
    import shot_judge
    op = Opener([ok(DROPPED, pt=40)] * (lg.IMAGE_DROP_RETRIES + 1))
    gw = lg.Gateway(api_key="k", opener=op)
    ok_, why = shot_judge.vision_check(gw, "m/vision")
    assert not ok_ and shot_judge.vision_check_failed_by_gateway(why), why


# ---------------------------------------------------------------- рассуждение не выключилось

def runaway(tokens=2500, text=""):
    """Дословная форма сорвавшегося ответа из записи эп.94 (27.09): рассуждение
    в reasoning_content, выход весь израсходован, content пуст, служебных
    полей usage нет."""
    return {"choices": [{"message": {"role": "assistant", "content": text,
                                     "reasoning_content": "We need answer JSON only with marks. " * 50},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2644, "completion_tokens": tokens, "total_tokens": 2644 + tokens}}


def test_ignored_thinking_switch_is_asked_again():
    op = Opener([runaway(), ok('{"drop": [2, 5]}', pt=2584, ct=159)])
    gw = lg.Gateway(api_key="k", opener=op)
    text, _u, _p = gw.chat("m/vision", [{"type": "text", "text": "screen"}], 2048, 7000, reasoning=False)
    assert text == '{"drop": [2, 5]}' and chat_calls(op) == 2
    assert gw.thinking_slips == 1 and gw.summary()["thinking_slips"] == 1


def test_ignored_thinking_twice_is_a_failure_not_a_loop():
    op = Opener([runaway()] * (lg.THINKING_SLIP_RETRIES + 1))
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.ThinkingIgnored) as e:
        gw.chat("m/vision", [{"type": "text", "text": "screen"}], 2048, 7000, reasoning=False)
    assert isinstance(e.value, lg.EmptyAnswer), "вызывающие, что ловят пустой ответ, ловят и этот"
    assert "EmptyAnswer" not in f"{type(e.value).__name__}: {e.value}", \
        "причина в отчёте слота — сбой провайдера, не «пустой ответ» (тот слот не перерешается)"
    assert chat_calls(op) == lg.THINKING_SLIP_RETRIES + 1


def test_thinking_that_was_asked_for_is_not_retried():
    """Рассуждение включено вызывающим и съело лимит — это свойство вопроса:
    повтор дал бы то же (планировщик сам переспрашивает окнами меньше)."""
    op = Opener([runaway()])
    gw = lg.Gateway(api_key="k", opener=op)
    with pytest.raises(lg.EmptyAnswer) as e:
        gw.chat("m/vision", [{"type": "text", "text": "plan"}], 2500, 7000, reasoning=True)
    assert not isinstance(e.value, lg.ThinkingIgnored) and chat_calls(op) == 1


def test_caption_screen_is_a_short_call_hedged_from_the_first_slot(monkeypatch):
    """Отсев по подписи — короткий ответ: дубль через 30 с без ответа уходит
    с первого вызова, а не после пяти замеров."""
    import caption_screen
    monkeypatch.setenv("GATEWAY_HEDGE", "1")
    gw = lg.Gateway(api_key="k", opener=Opener([]))
    assert caption_screen.MAX_TOKENS <= lg.HEDGE_SHORT_MAX_TOKENS
    assert gw.hedge_after(caption_screen.MODEL, caption_screen.MAX_TOKENS) == lg.HEDGE_SHORT_MAX_SEC
