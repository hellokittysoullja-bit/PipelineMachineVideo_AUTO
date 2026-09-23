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
