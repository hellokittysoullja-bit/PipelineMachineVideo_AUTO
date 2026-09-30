#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Здоровье внешних источников — ОДИН механизм на все хосты.

Раньше каждый источник держал свой ограничитель: Мет (интервал + пауза на
403 + вдвое медленнее после неё), Openverse (интервал анонимно/с ключом),
Викимедиа (интервал по хосту), шлюз моделей (повторы без памяти о том, что
сервис лежит). Четыре копии одной механики — «следующий слот под замком,
пауза, замедление, счётчики» — и у каждой своё подмножество: у шлюза не
было паузы вовсе, и при лежащем сервисе КАЖДЫЙ вызов сам проходил все
повторы (разбор 25.09: +25-50 с на слот).

Здесь механика одна:
  wait()       — дождаться своего слота (интервал между запросами хоста);
  throttled()  — сервис попросил притормозить (403/429): пауза cooldown_sec
                 и интервал вдвое длиннее, не длиннее max_interval;
  failed()     — вызов не удался (повторы исчерпаны): после fail_threshold
                 подряд — пауза cooldown_sec без замедления;
  succeeded()  — сервис ответил, счётчик отказов подряд обнуляется;
  cooling()    — хост на паузе: вызывающий решает, ждать или пропустить.

Параметры хостов — замеренные раньше числа их модулей, а не новые."""
import contextvars
import threading
import time

# Запрос сделан фоновым потоком (упреждающий поиск, упреждающий отбор), а не
# настоящим циклом слота. Такой запрос НЕ имеет права менять состояние хоста,
# от которого зависит настоящий цикл: ни запускать паузу и замедление (403/429
# на фоновом запросе оставляли настоящему слоту неполную кучу — разбор кода
# 30.09), ни занимать очередь глубже одного интервала (настоящий запрос иначе
# ждал бы дольше max_wait и пропускался). Хост на паузе — фон его пропускает.
# Очередь и пауза проверяются только у вызовов с max_wait: они готовы к Skipped
# (источники кучи); шлюз моделей без max_wait Skipped не ждёт.
BACKGROUND = contextvars.ContextVar("SOURCE_HEALTH_BACKGROUND", default=False)

# Сервис не назвал паузу — ждать не меньше этого (правило Викимедиа для
# клиента без заголовка Retry-After: «at least five seconds»).
MIN_RETRY_AFTER_SEC = 5.0


class Skipped(Exception):
    """Запрос не сделан: своей очереди у хоста (интервал или пауза) пришлось
    бы ждать дольше предела (max_wait_sec). Вызывающий пропускает источник
    для этого запроса, как при сбое, — отбор идёт дальше без ожидания."""


def max_wait_sec():
    """Предел ожидания очереди хоста (SOURCE_MAX_WAIT_SEC, по умолчанию 5 с;
    решение владельца 29.09: скорость важнее полноты последних фраз).
    Большое число — прежнее поведение «ждать сколько нужно» (замеры на
    записи сети, где пропуск по времени сделал бы прогоны несравнимыми)."""
    import os
    raw = (os.environ.get("SOURCE_MAX_WAIT_SEC") or "").strip()
    try:
        return max(0.0, float(raw)) if raw else 5.0
    except ValueError:
        return 5.0


class Host:
    def __init__(self, name, interval=0.0, *, max_interval=None, cooldown_sec=0.0,
                 slow_factor=2.0, fail_threshold=0, recover_after=0):
        self.name = name
        self.base_interval = float(interval)
        self.max_interval = float(interval if max_interval is None else max_interval)
        self.cooldown_sec = float(cooldown_sec)
        self.slow_factor = float(slow_factor)
        self.fail_threshold = int(fail_threshold)
        # Возврат темпа: после recover_after успехов подряд интервал снова
        # делится на slow_factor, не ниже исходного (0 — не возвращать:
        # прежнее поведение, замедление до конца прогона).
        self.recover_after = int(recover_after)
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.interval = self.base_interval
        self.next_slot = 0.0
        self.cooldown_until = 0.0
        self.fails = 0
        self.streak = 0
        self.stats = {"requests": 0, "cooldowns": 0, "failures": 0, "skipped": 0,
                      "bg_skipped": 0, "bg_throttled": 0, "bg_failed": 0}

    @property
    def rate(self):
        """Запросов в секунду сейчас (бесконечность — без интервала)."""
        return 1.0 / self.interval if self.interval > 0 else float("inf")

    def cooling(self):
        return time.monotonic() < self.cooldown_until

    def cooldown_left(self):
        return max(0.0, self.cooldown_until - time.monotonic())

    def wait(self, interval=None, max_wait=None):
        """Дождаться своей очереди. max_wait — предел: ждать пришлось бы
        дольше (очередь или пауза хоста) — Skipped, место в очереди не
        занимается."""
        iv = self.interval if interval is None else float(interval)
        background = BACKGROUND.get()
        with self._lock:
            now = time.monotonic()
            if background and max_wait is not None and (
                    self.cooling() or max(now, self.next_slot) - now > iv):
                # Фон не встаёт в очередь глубже одного интервала и не идёт на
                # хост, который на паузе: место остаётся настоящему циклу.
                self.stats["bg_skipped"] += 1
                raise Skipped(f"{self.name}: фоновый запрос уступает очередь настоящему циклу")
            slot = max(now, self.next_slot, self.cooldown_until if max_wait is not None else 0.0)
            if max_wait is not None and slot - now > max_wait:
                self.stats["skipped"] += 1
                raise Skipped(f"{self.name}: очередь {slot - now:.0f} с дольше предела {max_wait:.0f} с")
            self.next_slot = slot + iv
            self.stats["requests"] += 1
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _enter_cooldown(self, seconds=None):
        sec = self.cooldown_sec
        if seconds is not None:
            sec = min(self.cooldown_sec, max(MIN_RETRY_AFTER_SEC, float(seconds)))
        self.cooldown_until = time.monotonic() + sec
        self.stats["cooldowns"] += 1

    def throttled(self, retry_after=None):
        """Сервис попросил притормозить. True — пауза началась сейчас (а не
        продолжается), то есть об этом стоит сказать один раз.

        retry_after — сколько секунд попросил подождать сам сервис
        (заголовок Retry-After): пауза — столько, но не меньше
        MIN_RETRY_AFTER_SEC и не больше cooldown_sec. Живой случай judge14
        (24.09): Викимедиа отвечала 429 с «Retry-After: 22», а хост ждал
        свои 60 с на каждую из трёх попыток — две минуты простоя на каждый
        отказавший запрос, четверть времени прогона."""
        if BACKGROUND.get():
            with self._lock:
                self.stats["bg_throttled"] += 1
            return False
        with self._lock:
            if self.cooling():
                return False
            self._enter_cooldown(retry_after)
            self.streak = 0
            if self.interval > 0:
                self.interval = min(self.max_interval, self.interval * self.slow_factor)
            return True

    def failed(self):
        """Вызов не удался. True — после этого отказа хост ушёл на паузу."""
        if BACKGROUND.get():
            with self._lock:
                self.stats["bg_failed"] += 1
            return False
        with self._lock:
            self.fails += 1
            self.stats["failures"] += 1
            if self.fail_threshold and self.fails >= self.fail_threshold and not self.cooling():
                self._enter_cooldown()
                self.fails = 0
                return True
            return False

    def succeeded(self):
        if BACKGROUND.get():
            return          # успех фона не возвращает темп и не обнуляет отказы настоящего цикла
        with self._lock:
            self.fails = 0
            if self.recover_after and self.interval > self.base_interval:
                self.streak += 1
                if self.streak >= self.recover_after:
                    self.interval = max(self.base_interval, self.interval / self.slow_factor)
                    self.streak = 0


_REGISTRY = {}
_REG_LOCK = threading.Lock()


def host(name, **params):
    """Хост по имени; параметры применяются при первом обращении."""
    with _REG_LOCK:
        h = _REGISTRY.get(name)
        if h is None:
            h = _REGISTRY[name] = Host(name, **params)
        return h


def reset_all():
    """Новый прогон — новый счёт (тесты, повторный main())."""
    with _REG_LOCK:
        for h in _REGISTRY.values():
            h.reset()


def snapshot():
    """{хост: счётчики} — для отчёта прогона."""
    with _REG_LOCK:
        return {name: dict(h.stats, rate=(None if h.interval <= 0 else round(h.rate, 3)))
                for name, h in _REGISTRY.items()}
