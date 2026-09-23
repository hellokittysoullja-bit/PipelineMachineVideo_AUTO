#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Клиент OpenAI-совместимого шлюза моделей (по умолчанию AnyModel).

Один ключ и один адрес для моделей разных вендоров. Здесь — только то, что
нужно пайплайну, но сделанное надёжно:

  * ключ и адрес — только из окружения (.env): LLM_GATEWAY_API_KEY,
    LLM_GATEWAY_BASE_URL (по умолчанию https://anymodel.org/v1). В коде и в
    логах ключа нет; в ошибках — только request_id сервиса;
  * цена — из каталога самого шлюза (GET /models, коэффициент модели), а не
    из вшитой таблицы: каталог меняется, таблица отстала бы молча;
  * потолок расходов на прогон (spend_cap): вызов, который его превысил бы,
    не делается вовсе — BudgetExhausted. Считается по фактическому usage
    ответа, а до ответа — резервом по max_tokens (иначе параллельные вызовы
    вместе проскочили бы потолок);
  * 429 — пауза по Retry-After; 5xx и обрыв связи — повтор с нарастающей
    паузой; 402 (кончились деньги) и 401/403 — сразу громкая остановка без
    повторов: повтор тут ничего не исправит, а 402 означает реальные деньги.

Сервис закрыт Cloudflare-правилом, отвергающим стандартную подпись
Python-клиента (ошибка 1010) — заголовок User-Agent обязателен.
"""
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://anymodel.org/v1"
USER_AGENT = "PipelineMachineVideo/1.0"
MAX_ATTEMPTS = 4
BACKOFF_SEC = (2, 6, 15)


class GatewayError(Exception):
    """Вызов не удался и повтор не поможет (или повторы исчерпаны)."""


class PaymentRequired(GatewayError):
    """402: баланс ключа кончился. Дальнейшие вызовы бессмысленны."""


class BudgetExhausted(GatewayError):
    """Вызов превысил бы потолок расходов прогона — не делается."""


def _env(name, default=None):
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


class Gateway:
    def __init__(self, api_key=None, base_url=None, spend_cap=None, opener=None):
        self.api_key = api_key if api_key is not None else _env("LLM_GATEWAY_API_KEY")
        self.base_url = (base_url or _env("LLM_GATEWAY_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.spend_cap = spend_cap
        self._open = opener or urllib.request.urlopen
        self._lock = threading.Lock()
        self._prices = None
        self.spent = 0          # фактически списано (по usage ответов)
        self.reserved = 0       # зарезервировано вызовами в полёте
        self.calls = 0
        self.failures = 0
        self.dead = None        # причина, по которой шлюз выключен до конца прогона

    @property
    def configured(self):
        return bool(self.api_key)

    # ---------------------------------------------------------------- транспорт

    def _request(self, method, path, body=None, timeout=120):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json",
            "User-Agent": USER_AGENT})
        last = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                with self._open(req, timeout=timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                info = self._error_info(e)
                if e.code == 402:
                    raise PaymentRequired(f"402 баланс ключа исчерпан ({info})")
                if e.code in (401, 403):
                    raise GatewayError(f"{e.code} ключ не принят ({info})")
                if e.code == 429 or e.code >= 500:
                    last = f"{e.code} ({info})"
                    time.sleep(self._retry_after(e, attempt))
                    continue
                raise GatewayError(f"{e.code} ({info})")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = type(e).__name__
                time.sleep(BACKOFF_SEC[min(attempt, len(BACKOFF_SEC) - 1)])
        raise GatewayError(f"повторы исчерпаны: {last}")

    @staticmethod
    def _error_info(e):
        try:
            err = json.loads(e.read().decode("utf-8")).get("error", {})
            return f"{err.get('code')}, request_id={err.get('request_id')}"
        except Exception:
            return "без тела ответа"

    @staticmethod
    def _retry_after(e, attempt):
        try:
            return min(60.0, float(e.headers.get("Retry-After")))
        except (TypeError, ValueError):
            return BACKOFF_SEC[min(attempt, len(BACKOFF_SEC) - 1)]

    # ---------------------------------------------------------------- цены

    def coefficient(self, model):
        """(вход, выход) токенов баланса за токен модели — из каталога шлюза."""
        with self._lock:
            if self._prices is None:
                cat = self._request("GET", "/models", timeout=60)
                self._prices = {m["id"]: m.get("billing", {}).get("coefficient") for m in cat.get("data", [])}
        c = self._prices.get(model)
        if not c:
            raise GatewayError(f"модели {model!r} нет в каталоге шлюза или у неё нет цены")
        return float(c["input"]), float(c["output"])

    def cost(self, model, prompt_tokens, completion_tokens):
        ci, co = self.coefficient(model)
        return math.ceil(prompt_tokens * ci + completion_tokens * co)

    # ---------------------------------------------------------------- чат

    def chat(self, model, content, max_tokens, estimate_prompt_tokens, temperature=0.0, timeout=180):
        """Один вызов чата. content — список частей OpenAI (text / image_url).
        Возвращает (текст ответа, usage, цена). Потолок проверяется ДО вызова
        по резерву (оценка входа + max_tokens выхода)."""
        if self.dead:
            raise GatewayError(f"шлюз выключен до конца прогона: {self.dead}")
        if not self.configured:
            raise GatewayError("нет LLM_GATEWAY_API_KEY")
        reserve = self.cost(model, estimate_prompt_tokens, max_tokens)
        with self._lock:
            if self.spend_cap is not None and self.spent + self.reserved + reserve > self.spend_cap:
                raise BudgetExhausted(f"потолок {self.spend_cap}: потрачено {self.spent}, "
                                      f"в полёте {self.reserved}, нужно ещё до {reserve}")
            self.reserved += reserve
        try:
            r = self._request("POST", "/chat/completions", {
                "model": model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}]}, timeout=timeout)
        except PaymentRequired as e:
            self.dead = str(e)
            raise
        except GatewayError:
            with self._lock:
                self.failures += 1
            raise
        finally:
            with self._lock:
                self.reserved -= reserve
        u = r.get("usage") or {}
        price = self.cost(model, u.get("prompt_tokens") or estimate_prompt_tokens,
                          u.get("completion_tokens") or max_tokens)
        with self._lock:
            self.spent += price
            self.calls += 1
        text = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return text, u, price

    def summary(self):
        return {"base_url": self.base_url, "calls": self.calls, "failures": self.failures,
                "spent": self.spent, "spend_cap": self.spend_cap, "dead": self.dead}
