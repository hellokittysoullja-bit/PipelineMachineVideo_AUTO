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
                    {"id": "m/free", "billing": {"coefficient": {"input": 0, "output": 0}}}]}


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
        return Resp(json.dumps(a).encode())


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
