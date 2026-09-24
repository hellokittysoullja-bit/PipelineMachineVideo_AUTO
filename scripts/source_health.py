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
import threading
import time


class Host:
    def __init__(self, name, interval=0.0, *, max_interval=None, cooldown_sec=0.0,
                 slow_factor=2.0, fail_threshold=0):
        self.name = name
        self.base_interval = float(interval)
        self.max_interval = float(interval if max_interval is None else max_interval)
        self.cooldown_sec = float(cooldown_sec)
        self.slow_factor = float(slow_factor)
        self.fail_threshold = int(fail_threshold)
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.interval = self.base_interval
        self.next_slot = 0.0
        self.cooldown_until = 0.0
        self.fails = 0
        self.stats = {"requests": 0, "cooldowns": 0, "failures": 0}

    @property
    def rate(self):
        """Запросов в секунду сейчас (бесконечность — без интервала)."""
        return 1.0 / self.interval if self.interval > 0 else float("inf")

    def cooling(self):
        return time.monotonic() < self.cooldown_until

    def cooldown_left(self):
        return max(0.0, self.cooldown_until - time.monotonic())

    def wait(self, interval=None):
        iv = self.interval if interval is None else float(interval)
        with self._lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + iv
            self.stats["requests"] += 1
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _enter_cooldown(self):
        self.cooldown_until = time.monotonic() + self.cooldown_sec
        self.stats["cooldowns"] += 1

    def throttled(self):
        """Сервис попросил притормозить. True — пауза началась сейчас (а не
        продолжается), то есть об этом стоит сказать один раз."""
        with self._lock:
            if self.cooling():
                return False
            self._enter_cooldown()
            if self.interval > 0:
                self.interval = min(self.max_interval, self.interval * self.slow_factor)
            return True

    def failed(self):
        """Вызов не удался. True — после этого отказа хост ушёл на паузу."""
        with self._lock:
            self.fails += 1
            self.stats["failures"] += 1
            if self.fail_threshold and self.fails >= self.fail_threshold and not self.cooling():
                self._enter_cooldown()
                self.fails = 0
                return True
            return False

    def succeeded(self):
        with self._lock:
            self.fails = 0


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
