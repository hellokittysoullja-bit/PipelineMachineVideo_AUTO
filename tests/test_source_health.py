#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Один регулятор здоровья на все внешние хосты (М5, 25.09)."""
import io
import json
import os
import sys
import urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import llm_gateway  # noqa: E402
import source_health as sh  # noqa: E402


def test_throttle_pauses_and_slows_down_to_the_floor():
    h = sh.Host("t1", interval=0.2, max_interval=1.0, cooldown_sec=60)
    assert h.throttled() and h.cooling() and abs(h.rate - 2.5) < 1e-9
    assert not h.throttled(), "пауза продолжается — второй раз не объявляется"
    for _ in range(5):
        h.cooldown_until = 0.0
        h.throttled()
    assert h.rate == 1.0


def test_consecutive_failures_trip_the_breaker_and_success_resets():
    h = sh.Host("t2", fail_threshold=3, cooldown_sec=60)
    assert not h.failed() and not h.failed()
    h.succeeded()
    assert not h.failed() and not h.failed()
    assert h.failed() and h.cooling()


def _gw_with(responses, monkeypatch):
    monkeypatch.setattr(llm_gateway, "BACKOFF_SEC", (0, 0, 0))
    calls = []

    def opener(req, timeout=None):
        calls.append(req.full_url)
        code = responses(len(calls))
        if code == 200:
            return io.BytesIO(json.dumps({"data": []}).encode())
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b"{}"))
    gw = llm_gateway.Gateway(api_key="k", base_url="http://gw-test-%d" % id(calls), opener=opener)
    return gw, calls


def test_lying_gateway_is_not_hammered_by_every_call(monkeypatch):
    gw, calls = _gw_with(lambda n: 502, monkeypatch)
    for _ in range(llm_gateway.GATEWAY_FAIL_THRESHOLD):
        try:
            gw._request("GET", "/models")
        except llm_gateway.GatewayError:
            pass
    spent = len(calls)
    try:
        gw._request("GET", "/models")
        raise AssertionError("вызов на паузе обязан отказать сразу")
    except llm_gateway.GatewayUnavailable:
        pass
    assert len(calls) == spent, "на паузе шлюз не спрашивается"


def test_a_single_answer_resets_the_failure_count(monkeypatch):
    seq = iter([502] * llm_gateway.MAX_ATTEMPTS + [200] + [502] * llm_gateway.MAX_ATTEMPTS)
    gw, _calls = _gw_with(lambda n: next(seq), monkeypatch)
    for _ in range(3):
        try:
            gw._request("GET", "/models")
        except llm_gateway.GatewayError:
            pass
    assert not gw.health().cooling()


def test_pause_follows_the_services_retry_after_within_bounds():
    """Викимедиа на 429 называет паузу (Retry-After: 22) — ждём её, а не
    свои 60 с; слишком короткая — не меньше 5 с, слишком длинная — не
    больше своей паузы хоста."""
    h = sh.Host("t_ra", cooldown_sec=60)
    assert h.throttled(retry_after=22) and 20 < h.cooldown_left() <= 22.01
    h.cooldown_until = 0.0
    assert h.throttled(retry_after=1) and 4.5 < h.cooldown_left() <= sh.MIN_RETRY_AFTER_SEC + 0.01
    h.cooldown_until = 0.0
    assert h.throttled(retry_after=600) and h.cooldown_left() <= 60.01
    h.cooldown_until = 0.0
    assert h.throttled() and h.cooldown_left() > 59, "без заголовка — своя пауза хоста"
