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

# Сервис не назвал паузу — ждать не меньше этого (правило Викимедиа для
# клиента без заголовка Retry-After: «at least five seconds»).
MIN_RETRY_AFTER_SEC = 5.0


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
        with self._lock:
            if self.cooling():
                return False
            self._enter_cooldown(retry_after)
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


# ВРЕМЕННЫЙ СБОЙ — ПОВТОР, А НЕ ПОТЕРЯ ИСТОЧНИКА. Живой прогон эпизода 94
# (27.09, профиль каждого HTTP-запроса): поиск и превью источников изредка
# отвечают 429/5xx или рвут соединение, и каждый такой ответ стоил слоту
# всего источника (поиск) или кандидата (превью) — молча, отбор просто
# видел кучу беднее. Повтор с паузой, которую назвал сам сервис
# (Retry-After), а при 429/503 — ещё и замедление всего хоста (throttled):
# следующие запросы других потоков не бьют в тот же лимит. Пауза длиннее
# RETRY_MAX_WAIT_SEC — это не всплеск, а исчерпанная квота (у Pexels —
# до часа), её ждать нельзя: ошибка уходит вызывающему сразу.
TRANSIENT_HTTP = (429, 500, 502, 503, 504)
RETRY_BACKOFF_SEC = (1.5, 4.0)
RETRY_MAX_WAIT_SEC = 20.0


def transient_error(exc):
    """Сбой, который повтор может вылечить: 429, 5xx, обрыв, таймаут. Одно
    правило для всех, кто решает «повторить / считать ответ неполным»:
    постоянный отказ (401/403/404, битый ответ) повтором не лечится."""
    import http.client
    import urllib.error
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)):
        return False
    # Остальное семейство OSError — сеть: обрыв, таймаут, «сеть недоступна»
    # (urllib заворачивает их в URLError, но не везде).
    return isinstance(exc, (OSError, http.client.HTTPException))


def retry_after_of(error):
    """Retry-After в секундах из HTTPError или None."""
    try:
        value = (error.headers or {}).get("Retry-After")
        return float(value) if value is not None and str(value).strip().isdigit() else None
    except (AttributeError, TypeError, ValueError):
        return None


def urlopen_retry(req, timeout, host_name=None, attempts=3):
    """urllib.request.urlopen с повтором временных сбоев. Ответ — как у
    urlopen (контекстный менеджер). Невременная ошибка (404, 401, 403) и
    последняя попытка — исключение наружу, как раньше."""
    import http.client
    import urllib.error
    import urllib.request
    h = host(host_name, cooldown_sec=RETRY_MAX_WAIT_SEC) if host_name else None
    for attempt in range(attempts):
        if h is not None and h.cooling():
            time.sleep(h.cooldown_left())
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            if h is not None:
                h.succeeded()
            return r
        except urllib.error.HTTPError as e:
            if e.code not in TRANSIENT_HTTP or attempt == attempts - 1:
                raise
            if e.code == 429 and str((e.headers or {}).get("X-Ratelimit-Remaining", "")).strip() == "0":
                raise    # квота исчерпана (Pexels: до часа) — не всплеск, ждать нечего
            wait = retry_after_of(e)
            if wait is not None and wait > RETRY_MAX_WAIT_SEC:
                raise
            if h is not None and e.code in (429, 503):
                h.throttled(retry_after=wait if wait is not None else MIN_RETRY_AFTER_SEC)
                continue
            time.sleep(wait if wait is not None else RETRY_BACKOFF_SEC[min(attempt, len(RETRY_BACKOFF_SEC) - 1)])
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            if attempt == attempts - 1:
                raise
            time.sleep(RETRY_BACKOFF_SEC[min(attempt, len(RETRY_BACKOFF_SEC) - 1)])
