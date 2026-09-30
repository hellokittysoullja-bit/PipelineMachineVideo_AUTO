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
    повторов: повтор тут ничего не исправит, а 402 означает реальные деньги;
  * обрыв ПОСЛЕ того, как сервис начал отвечать (IncompleteRead, обрезанный
    JSON), — тоже повтор, но такой ответ сервис, скорее всего, уже списал:
    его резерв засчитывается в расход как потраченный. Иначе потолок
    считал бы деньги, которых больше нет, как свободные. Найдено живым
    прогоном: один обрыв ронял весь прогон брифов исключением http.client,
    которого список повторяемых ошибок не знал;
  * пустой ответ — ошибка, а не пустая строка. Рассуждающая модель может
    израсходовать весь max_tokens на рассуждение и вернуть content="" с
    finish_reason="length": 13 оплаченных вызовов DeepSeek на прогоне
    брифов дали ноль ответов, и снаружи это выглядело как «модель молчит».
    Ошибка называет finish_reason и число токенов рассуждения.

Сервис закрыт Cloudflare-правилом, отвергающим стандартную подпись
Python-клиента (ошибка 1010) — заголовок User-Agent обязателен.
"""
import collections
import contextvars
import hashlib
import http.client
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import source_health

# СПЕКУЛЯТИВНЫЕ ВЫЗОВЫ (упреждающий отбор слотов, pipeline_smart). Пока
# настоящий цикл решает слот i, слоты впереди прогоняются заранее, чтобы их
# вызовы моделей уже были сделаны к моменту, когда цикл до них дойдёт.
# Решения при этом обязаны быть теми же, что без упреждения, — а потолок
# расходов и отказы по нему зависят от того, КОГДА и СКОЛЬКО списано. Поэтому
# спекулятивный ответ не списывается сразу: он кладётся в хранилище шлюза
# вместе с ценой и резервом, а списывается в тот момент, когда настоящий
# цикл задаёт тот же вопрос, — ровно с той проверкой потолка, какую прошёл
# бы живой вызов. Настоящий цикл видит те же spent/reserved/отказы, что без
# упреждения. Ответ, который настоящий цикл так и не спросил, — оплаченный
# впустую; его сумма в summary()["speculative_wasted"].
_SPECULATIVE = contextvars.ContextVar("llm_gateway_speculative", default=False)


# ЗАМЕР ВЫЗОВОВ (прогон 30.09, пункт B4): фаза судьи занимала слоту 0 больше
# тысячи секунд, а stage_timings не знал, сколько из них — ответ модели,
# сколько — очередь и повторы. Каждый вызов чата и генерации пишет строку
# gateway_call: кто спросил (функция вне шлюза), модель, секунды целиком
# (очередь, повторы, пауза — всё, что ждал вызывающий), токены, цена, исход.
# Ответ, взятый из хранилища упреждения, пишется с from_spec: его секунды —
# сколько настоящий цикл ждал готовый ответ. Выключено вместе с STAGE_TIMER;
# на ответы и выбор не влияет.
def _caller():
    import sys
    f = sys._getframe(1)
    here = __name__
    while f is not None:
        mod = f.f_globals.get("__name__", "")
        if mod not in (here, "threading", "contextvars", "concurrent.futures.thread"):
            return f"{mod}.{f.f_code.co_name}"
        f = f.f_back
    return "?"


_FROM_SPEC = contextvars.ContextVar("llm_gateway_from_spec", default=False)


def _record_call(kind, model, t0, ok, usage=None, price=None, err=None, **extra):
    try:
        import stage_timer
        if not stage_timer.STAGE_TIMER_ENABLED:
            return
        u = usage or {}
        rec = {"kind": kind, "model": model, "caller": _caller(), "ok": ok,
               "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
               "price": price}
        if err is not None:
            rec["error"] = f"{type(err).__name__}: {err}"[:300]
        rec.update(extra)
        stage_timer.record("gateway_call", time.monotonic() - t0, **rec)
    except Exception:  # noqa: BLE001 — телеметрия не роняет вызов
        pass


# Сколько вопросов упреждения идёт в шлюз одновременно. Лимит параллельности
# шлюза не опубликован, поэтому он подбирается на ходу: после серии ответов
# без отказа лимит растёт на один, на отказ «слишком много запросов» (429,
# от любого вызова этого адреса) — делится пополам. Вопросы настоящего
# цикла ограничитель не трогает: ответы и выбор те же, меняется только
# очерёдность заранее заданных вопросов.
SPEC_LIMIT_START = 4
SPEC_LIMIT_MAX = 32


class AdaptiveLimit:
    def __init__(self, start=None, maximum=None):
        env_max = (os.environ.get("SPEC_GATEWAY_MAX_CONCURRENT") or "").strip()
        self.maximum = max(1, int(env_max) if env_max.isdigit() else (maximum or SPEC_LIMIT_MAX))
        self.limit = min(self.maximum, start or SPEC_LIMIT_START)
        self.active = 0
        self._streak = 0
        self._cv = threading.Condition()
        self.stats = {"peak_limit": self.limit, "throttles": 0}

    def acquire(self):
        with self._cv:
            while self.active >= self.limit:
                self._cv.wait()
            self.active += 1

    def release(self, ok=True):
        with self._cv:
            self.active -= 1
            if ok:
                self._streak += 1
                if self._streak >= self.limit and self.limit < self.maximum:
                    self.limit += 1
                    self._streak = 0
                    self.stats["peak_limit"] = max(self.stats["peak_limit"], self.limit)
            self._cv.notify_all()

    def throttled(self):
        with self._cv:
            self.limit = max(1, self.limit // 2)
            self._streak = 0
            self.stats["throttles"] += 1


_SPEC_LIMITERS = {}
_SPEC_LIMITERS_LOCK = threading.Lock()


def speculative():
    """True, если текущий поток делает спекулятивный вызов."""
    return _SPECULATIVE.get()


class speculation:
    """Контекст спекулятивных вызовов (для упреждающего отбора)."""

    def __enter__(self):
        self._token = _SPECULATIVE.set(True)
        return self

    def __exit__(self, *exc):
        _SPECULATIVE.reset(self._token)
        return False


def _fingerprint(model, content, max_tokens, estimate_prompt_tokens, temperature, reasoning):
    """Всё, от чего зависит ответ и резерв вызова. Картинки входят целиком
    (base64 в content): другой кадр в сетке — другой вопрос."""
    h = hashlib.sha256()
    h.update(json.dumps([model, content, max_tokens, estimate_prompt_tokens, temperature,
                         reasoning], ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()

DEFAULT_BASE_URL = "https://anymodel.org/v1"
USER_AGENT = "PipelineMachineVideo/1.0"
MAX_ATTEMPTS = 4
BACKOFF_SEC = (2, 6, 15)


class GatewayError(Exception):
    """Вызов не удался и повтор не поможет (или повторы исчерпаны)."""


class PaymentRequired(GatewayError):
    """402: баланс ключа кончился. Дальнейшие вызовы бессмысленны."""


class GatewayUnavailable(GatewayError):
    """Шлюз не отвечает после нескольких пауз подряд — выключен до конца прогона."""


# Шлюз лежит — вызовы не делаются GATEWAY_COOLDOWN_SEC после
# GATEWAY_FAIL_THRESHOLD вызовов подряд с исчерпанными повторами. Без этого
# КАЖДЫЙ вызов сам проходил все повторы (разбор 25.09: слот с сеткой и
# пятью проверками при лежащем шлюзе терял минуты). Пауза — то же правило,
# что у Мет (source_health), и те же 60 с.
GATEWAY_FAIL_THRESHOLD = 3
GATEWAY_COOLDOWN_SEC = 60.0
# Политика ожидания — ОДНА, здесь, а не у каждого вызывающего. Раньше пауза
# сразу давала отказ, и вызывающие код обходили это сами: слот судьи ждал
# паузу в начале, проверка финалистов переспрашивала сорванные после неё
# (живой случай judge14: три настоящих кинжала остались без проверки, и
# меч, проверенный до сбоя, встал на экран). Две копии одной механики у
# двух вызывающих, остальные (планировщик, второй круг, паспорт мира) не
# имели и её. Теперь вызов сам пережидает паузу, после исчерпанных повторов
# переспрашивает один раз, а после GATEWAY_MAX_PAUSES пауз подряд без
# единого ответа шлюз выключается до конца прогона: лежащий сервис не
# должен стоить минуту ожидания каждому следующему вызову.
GATEWAY_MAX_PAUSES = 3


class BudgetExhausted(GatewayError):
    """Вызов превысил бы потолок расходов прогона — не делается."""


class EmptyAnswer(GatewayError):
    """Сервис ответил и списал деньги, но текста ответа нет."""


def _empty_answer(model, choice, usage, max_tokens, price):
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return EmptyAnswer(f"{model}: пустой ответ (finish_reason={choice.get('finish_reason')}, "
                       f"токенов рассуждения {reasoning}, выход {usage.get('completion_tokens')} "
                       f"из {max_tokens}); оплачено {price}")


def _env(name, default=None):
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def reasoning_switch(thinking_format, on):
    """Поле запроса, которое включает или выключает рассуждение модели, — по
    её формату из каталога шлюза (capabilities.thinkingFormat), а не одно на
    всех. Проверено живьём 24.09:
      * qwen: {"reasoning": {"enabled": false}} — 0 токенов рассуждения;
        reasoning_effort и enable_thinking шлюз игнорирует;
      * deepseek: на длинном вопросе {"reasoning": {"enabled": false}} НЕ
        выключает рассуждение (весь лимит выхода ушёл в него, ответа нет), а
        {"thinking": {"type": "disabled"}} выключает — 0 токенов, ответ
        целиком.
    Остальные форматы — как у qwen (поведение до этой правки)."""
    if thinking_format == "deepseek":
        return {"thinking": {"type": "enabled" if on else "disabled"}}
    return {"reasoning": {"enabled": bool(on)}}


def _is_stdlib_urlopen(fn):
    """urlopen — настоящая функция urllib, а не подмена. Проверка по самой
    функции, а не по снимку при импорте: харнесс эквивалентности
    (net_recorder) подменяет urlopen ДО импорта этого модуля, и снимок
    принимал подмену за оригинал — пул соединений шёл в живую сеть мимо
    записи (вызовы шлюза в прогонах A/B 28.09 были живыми)."""
    return getattr(fn, "__module__", None) == "urllib.request" and getattr(fn, "__name__", "") == "urlopen"


_SESSION = None
_SESSION_LOCK = threading.Lock()


class _PooledResponse:
    """Ответ пула соединений в форме ответа urlopen: read(), headers,
    контекстный менеджер. Обрыв тела — http.client.IncompleteRead, как у
    urllib (ветка «ответ оборван» в _request_with_retries)."""

    def __init__(self, resp):
        self._resp = resp
        self.headers = resp.headers
        self.status = resp.status_code

    def read(self):
        import requests
        try:
            return self._resp.content
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            raise http.client.IncompleteRead(b"", None) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(str(e)) from e

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._resp.close()
        return False


def pooled_urlopen(req, timeout=None):
    """urlopen с повторным использованием соединений (HTTP keep-alive).

    Каждый вызов шлюза раньше открывал новое TCP+TLS-соединение (и CONNECT
    через прокси, если он есть). Замер 26.09 на живом шлюзе, GET /models:
    новое соединение — 1.5-1.9 с, повторное — 0.75-0.82 с. У слота 6-10
    вызовов подряд на критическом пути (сетка, проверки, мир, «лучший как
    кадр фильма», рамка). Запрос, заголовки и тело — те же байты; ошибки —
    те же исключения urllib (HTTPError с кодом, заголовками и телом;
    URLError/TimeoutError на сеть), поэтому повторы, паузы и учёт
    оборванных ответов работают как раньше.

    Если urllib.request.urlopen подменён (запись и воспроизведение сети
    харнессом эквивалентности, net_recorder), вызов идёт через подмену: вся
    сеть процесса обязана оставаться видимой для записи."""
    if not _is_stdlib_urlopen(urllib.request.urlopen):
        return urllib.request.urlopen(req, timeout=timeout)
    try:
        import requests
    except ImportError:
        return urllib.request.urlopen(req, timeout=timeout)
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=32)
            _SESSION.mount("https://", adapter)
            _SESSION.mount("http://", adapter)
        session = _SESSION
    url = req.full_url
    try:
        resp = session.request(req.get_method(), url, data=req.data,
                               headers=dict(req.header_items()), timeout=timeout, stream=True)
    except requests.exceptions.Timeout as e:
        raise TimeoutError(str(e)) from e
    except requests.exceptions.RequestException as e:
        raise urllib.error.URLError(e) from e
    if resp.status_code >= 400:
        try:
            body = resp.content
        except Exception:  # noqa: BLE001 — тело ошибки не обязательно
            body = b""
        finally:
            resp.close()
        import io
        raise urllib.error.HTTPError(url, resp.status_code, resp.reason, resp.headers,
                                     io.BytesIO(body))
    return _PooledResponse(resp)


class Gateway:
    def __init__(self, api_key=None, base_url=None, spend_cap=None, opener=None):
        self.api_key = api_key if api_key is not None else _env("LLM_GATEWAY_API_KEY")
        self.base_url = (base_url or _env("LLM_GATEWAY_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.spend_cap = spend_cap
        self._open = opener or pooled_urlopen
        self._lock = threading.Lock()
        self._prices = None
        # Фактическая цена вызова против оценки, по модели. Замер 23.09:
        # маршрут cc/ (Claude через шлюз) брал ~150 тыс. токенов баланса за
        # ОДИН вызов сетки судьи при оценке в несколько тысяч — скрытая
        # надбавка на вызов. Потолок проверялся по оценке, параллельные
        # сетки все прошли проверку, и прогон с потолком 80 тыс. потратил
        # 647 тыс. Теперь резерв умножается на наблюдённое отношение, а
        # первый вызов незнакомой модели идёт один, без параллельных: его
        # цена успевает стать известной до следующего.
        self._ratio = {}
        self._spec_ratio = {}   # то же по спекулятивным вызовам (в настоящий учёт не идёт)
        self._first_call = {}
        self.spent = 0          # фактически списано (по usage ответов)
        self.reserved = 0       # зарезервировано вызовами в полёте
        self.calls = 0
        self.failures = 0
        self.lost_bodies = 0    # ответы, оборванные после начала: засчитаны резервом
        self.empty_answers = 0  # оплаченные ответы без текста
        self.dead = None        # причина, по которой шлюз выключен до конца прогона
        # Свой замок у счётчиков пауз: billing() держит self._lock, пока
        # читает каталог через _request, и общий замок тут был бы взаимной
        # блокировкой (поймано первым же тестом).
        self._pause_lock = threading.Lock()
        self.pauses = 0         # пауз подряд без единого ответа
        self.waited = 0.0       # секунд, прожданных на паузах
        self.reasked = 0        # вызовов, переспрошенных после исчерпанных повторов
        self.empty_answer_reasked = 0   # переспрошено после пустого ответа (200 OK, без текста)
        # Спекулятивное хранилище: отпечаток вопроса -> очередь ответов.
        self._spec = collections.defaultdict(collections.deque)
        self._spec_pending = {}       # отпечаток -> Event: спекулятивный вызов в полёте
        self.spec_reserved = 0        # резерв спекулятивных вызовов в полёте
        self.spec_outstanding = 0     # оплачено спекулятивно и ещё не использовано
        self.spec_calls = 0           # спекулятивных вызовов в сеть
        self.spec_used = 0            # из них использовано настоящим циклом
        self.spec_spent = 0           # оплачено спекулятивно всего
        # Секунды, проведённые в сетевых вызовах чата (настоящих и
        # спекулятивных), — для разреза времени прогона: сколько судья
        # занимает на самом деле, а не по оценке.
        self.net_seconds = 0.0
        self.spec_net_seconds = 0.0

    @property
    def configured(self):
        return bool(self.api_key)

    # ---------------------------------------------------------------- транспорт

    def spec_limiter(self):
        """Ограничитель одновременных вопросов упреждения (один на адрес)."""
        host = urllib.parse.urlsplit(self.base_url).hostname or self.base_url
        with _SPEC_LIMITERS_LOCK:
            lim = _SPEC_LIMITERS.get(host)
            if lim is None:
                lim = _SPEC_LIMITERS[host] = AdaptiveLimit()
            return lim

    def health(self):
        """Регулятор здоровья этого шлюза (один на адрес в процессе)."""
        host = urllib.parse.urlsplit(self.base_url).hostname or self.base_url
        return source_health.host("gateway:" + host, fail_threshold=GATEWAY_FAIL_THRESHOLD,
                                  cooldown_sec=GATEWAY_COOLDOWN_SEC)

    def _request(self, method, path, body=None, timeout=120, on_lost_body=None):
        """on_lost_body() зовётся на каждый ответ, оборвавшийся после того,
        как сервис начал его отдавать: такой вызов, скорее всего, оплачен.

        Пауза шлюза пережидается здесь же; исчерпанные повторы — один
        переспрос после паузы (если её никто не начал — сразу), кроме
        случая, когда ответ обрывался после начала (он оплачен). Ответ
        важнее скорости: каждый вызов шлюза в пайплайне — оплаченная
        работа (проверка кадра, спецификация главы), и отказ из-за чужой
        паузы означал бы кадр без проверки там, где проверка оплачена."""
        health = self.health()
        lost = []
        # Вызов упреждающего отбора не трогает «здоровье» шлюза и счётчик
        # пауз (аудит 29.09): его сбои раньше входили в общий счёт, и три
        # паузы из-за упреждения выключали судью настоящему циклу до конца
        # прогона — решение, которого без упреждения не было бы. Паузу,
        # начатую настоящим циклом, упреждение пережидает; при сбое
        # отказывает сразу, без переспроса: слот спросит сам.
        spec = speculative()

        def lost_body():
            lost.append(1)
            if on_lost_body is not None:
                on_lost_body()
        for attempt in range(2):
            self._wait_pause(health)
            try:
                out = self._request_with_retries(method, path, body, timeout, lost_body)
            except GatewayError as e:
                if "повторы исчерпаны" not in str(e) or spec:
                    raise
                if health.failed():
                    with self._pause_lock:
                        self.pauses += 1
                        if self.pauses >= GATEWAY_MAX_PAUSES and not self.dead:
                            self.dead = (f"не отвечает после {self.pauses} пауз подряд "
                                         f"по {GATEWAY_COOLDOWN_SEC:.0f} с")
                    # Причина — в строке: «не отвечает» без неё не отличает
                    # таймаут от 429 и 5xx, а от этого зависит, сколько ждать.
                    print(f"  шлюз не отвечает {GATEWAY_FAIL_THRESHOLD} вызова подряд ({e}) — пауза "
                          f"{GATEWAY_COOLDOWN_SEC:.0f} с" + (f"; {self.dead} — выключен до конца "
                                                          f"прогона" if self.dead else ""))
                # Оборванный после начала ответ, скорее всего, уже оплачен:
                # переспрос такого вызова платил бы ещё до MAX_ATTEMPTS раз.
                if attempt == 0 and not self.dead and not lost:
                    with self._pause_lock:
                        self.reasked += 1
                    continue
                raise
            if not spec:
                health.succeeded()
                with self._pause_lock:
                    self.pauses = 0
            return out

    def _wait_pause(self, health):
        if self.dead:
            raise GatewayUnavailable(f"шлюз выключен до конца прогона: {self.dead}")
        left = health.cooldown_left()
        if left > 0:
            time.sleep(left)
            with self._pause_lock:
                self.waited += left
        if self.dead:
            raise GatewayUnavailable(f"шлюз выключен до конца прогона: {self.dead}")

    def _request_with_retries(self, method, path, body, timeout, on_lost_body):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json",
            "User-Agent": USER_AGENT})
        last = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                with self._open(req, timeout=timeout) as r:
                    try:
                        return json.loads(r.read().decode("utf-8"))
                    except (http.client.HTTPException, ValueError, ConnectionError, TimeoutError) as e:
                        # Статус 200 уже получен — тело оборвалось или битое.
                        if on_lost_body is not None:
                            on_lost_body()
                        last = f"ответ оборван: {type(e).__name__}"
                        time.sleep(BACKOFF_SEC[min(attempt, len(BACKOFF_SEC) - 1)])
                        continue
            except urllib.error.HTTPError as e:
                info = self._error_info(e)
                if e.code == 402:
                    raise PaymentRequired(f"402 баланс ключа исчерпан ({info})")
                if e.code in (401, 403):
                    raise GatewayError(f"{e.code} ключ не принят ({info})")
                if e.code == 429 or e.code >= 500:
                    if e.code == 429:
                        self.spec_limiter().throttled()
                    last = f"{e.code} ({info})"
                    time.sleep(self._retry_after(e, attempt))
                    continue
                raise GatewayError(f"{e.code} ({info})")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
                    http.client.HTTPException) as e:
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

    def billing(self, model):
        """Запись цены модели из каталога шлюза (GET /models) как есть."""
        with self._lock:
            if self._prices is None:
                cat = self._request("GET", "/models", timeout=60)
                self._prices = {m["id"]: m.get("billing") or {} for m in cat.get("data", [])}
                self._thinking = {m["id"]: (m.get("capabilities") or {}).get("thinkingFormat")
                                  for m in cat.get("data", [])}
        b = self._prices.get(model)
        if not b or not b.get("coefficient"):
            raise GatewayError(f"модели {model!r} нет в каталоге шлюза или у неё нет цены")
        return b

    def coefficient(self, model):
        """(вход, выход) токенов баланса за токен модели — из каталога шлюза."""
        c = self.billing(model)["coefficient"]
        return float(c["input"]), float(c["output"])

    def image_cost(self, model, size, quality=None, n=1):
        """Цена картинок по формуле шлюза: base_tokens × множители качества и
        размера × коэффициент выхода × число картинок. Множители — из
        каталога модели (billing.scales), а не вшитой таблицей: у моделей
        по подписке своя лестница качества, и таблица отстала бы молча.
        Размера или качества нет в лестнице модели — множитель 1 (так
        шлюз считает auto); отсутствующий base_tokens — ошибка, а не ноль."""
        b = self.billing(model)
        if b.get("unit") != "image" or "base_tokens" not in b:
            raise GatewayError(f"{model!r} не модель картинок (unit={b.get('unit')})")
        scales = b.get("scales") or {}
        qm = (scales.get("quality") or {}).get(quality or "auto", 1)
        sm = (scales.get("size") or {}).get(size, 1)
        return math.ceil(round(b["base_tokens"] * qm * sm) * n * float(b["coefficient"]["output"]))

    def cost(self, model, prompt_tokens, completion_tokens):
        ci, co = self.coefficient(model)
        return math.ceil(prompt_tokens * ci + completion_tokens * co)

    # ---------------------------------------------------------------- чат

    def chat(self, model, content, max_tokens, estimate_prompt_tokens, temperature=0.0, timeout=180,
             reasoning=None):
        """Один вызов чата (с одним переспросом на пустой ответ — см.
        _chat_reasked). content — список частей OpenAI (text / image_url).
        Возвращает (текст ответа, usage, цена). Потолок проверяется ДО вызова
        по резерву (оценка входа + max_tokens выхода)."""
        if self.dead:
            raise GatewayError(f"шлюз выключен до конца прогона: {self.dead}")
        if not self.configured:
            raise GatewayError("нет LLM_GATEWAY_API_KEY")
        t0 = time.monotonic()
        tok = _FROM_SPEC.set(False)
        try:
            with self._lock:
                first = self._first_call.setdefault(model, threading.Lock())
            if model in self._ratio or model in self._spec_ratio:
                out = self._chat_reasked(model, content, max_tokens, estimate_prompt_tokens,
                                         temperature, timeout, reasoning)
            else:
                with first:
                    out = self._chat_reasked(model, content, max_tokens, estimate_prompt_tokens,
                                             temperature, timeout, reasoning)
        except Exception as e:
            _record_call("chat", model, t0, False, err=e, from_spec=_FROM_SPEC.get())
            raise
        finally:
            from_spec = _FROM_SPEC.get()
            _FROM_SPEC.reset(tok)
        _record_call("chat", model, t0, True, usage=out[1], price=out[2], from_spec=from_spec)
        return out

    def _chat_reasked(self, model, content, max_tokens, estimate_prompt_tokens, temperature, timeout,
                      reasoning):
        """Пустой ответ (200 OK, но весь max_tokens ушёл на рассуждение,
        несмотря на reasoning=False) — воспроизводимый сбой конкретно под
        ПАРАЛЛЕЛЬНОЙ нагрузкой на одну модель: замер 27.09, два одновременных
        вызова одному DeepSeek-эндпоинту (caption_screen.screen() шлёт свои
        два вопроса разом) — один из двух систематически возвращает пустое
        тело; тот же вызов в одиночку, без второго вызова рядом, не
        воспроизводится ни разу за несколько попыток. До этой правки
        EmptyAnswer нигде не переспрашивался (в отличие от сетевых ошибок и
        5xx у _request(), где переспрос уже есть) — единственный сорванный
        параллельный вызов молча ронял всю ступень (caption_screen на всём
        слоте, второй круг поиска, генерацию), хотя причина — не содержание
        запроса, а гонка на стороне провайдера.

        Один переспрос ничего не может ухудшить: провайдер уже списал деньги
        за первую попытку (self.spent растёт до проверки на пустоту), то есть
        при повторном сбое цена та же, что и раньше без переспроса, а при
        удачном повторе (типичный случай по замеру) ступень получает ответ,
        которого раньше не получала вовсе."""
        try:
            return self._chat(model, content, max_tokens, estimate_prompt_tokens, temperature, timeout,
                              reasoning)
        except EmptyAnswer:
            if not speculative():
                with self._lock:
                    self.empty_answer_reasked += 1
            return self._chat(model, content, max_tokens, estimate_prompt_tokens, temperature, timeout,
                              reasoning)

    def _chat(self, model, content, max_tokens, estimate_prompt_tokens, temperature, timeout,
              reasoning=None):
        base = self.cost(model, estimate_prompt_tokens, max_tokens)
        reserve = math.ceil(base * max(1.0, self._ratio.get(model, 1.0)))
        fp = _fingerprint(model, content, max_tokens, estimate_prompt_tokens, temperature, reasoning)
        if speculative():
            return self._chat_speculative(fp, model, content, max_tokens, estimate_prompt_tokens,
                                          temperature, timeout, reasoning, base)
        entry = self._spec_take(fp)
        if entry is not None:
            return self._spec_consume(fp, entry, model, base, reserve, max_tokens)
        with self._lock:
            if self.spend_cap is not None and self.spent + self.reserved + reserve > self.spend_cap:
                raise BudgetExhausted(f"потолок {self.spend_cap}: потрачено {self.spent}, "
                                      f"в полёте {self.reserved}, нужно ещё до {reserve}")
            self.reserved += reserve
        def lost_body():
            with self._lock:
                self.spent += reserve
                self.lost_bodies += 1
        try:
            body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": content}]}
            if reasoning is not None:
                body.update(reasoning_switch(getattr(self, "_thinking", {}).get(model), reasoning))
            t0 = time.monotonic()
            try:
                r = self._request("POST", "/chat/completions", body, timeout=timeout,
                                  on_lost_body=lost_body)
            finally:
                with self._lock:
                    self.net_seconds += time.monotonic() - t0
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
            if base > 0:
                self._ratio[model] = max(self._ratio.get(model, 0.0), price / base)
        choice = (r.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        if not text.strip():
            with self._lock:
                self.failures += 1
                self.empty_answers += 1
            raise _empty_answer(model, choice, u, max_tokens, price)
        return text, u, price

    # ---------------------------------------------------------------- упреждение

    def _spec_take(self, fp):
        """Спекулятивный ответ на этот вопрос (или None). Вызов того же
        вопроса в полёте — дождаться его: второй вызов заплатил бы дважды."""
        while True:
            with self._lock:
                dq = self._spec.get(fp)
                if dq:
                    return dq.popleft()
                ev = self._spec_pending.get(fp)
            if ev is None:
                return None
            ev.wait()

    def _spec_consume(self, fp, entry, model, base, reserve, max_tokens):
        """Настоящий цикл получает спекулятивный ответ — ровно с тем учётом,
        какой был бы у живого вызова в этот момент: та же проверка потолка
        по резерву, то же списание цены и обновление отношения цены к
        оценке, тот же отказ на пустом ответе."""
        with self._lock:
            if self.spend_cap is not None and self.spent + self.reserved + reserve > self.spend_cap:
                self._spec[fp].appendleft(entry)
                raise BudgetExhausted(f"потолок {self.spend_cap}: потрачено {self.spent}, "
                                      f"в полёте {self.reserved}, нужно ещё до {reserve}")
            price = entry["price"]
            self.spent += price
            self.calls += 1
            self.spec_outstanding -= price
            self.spec_used += 1
            if base > 0:
                self._ratio[model] = max(self._ratio.get(model, 0.0), price / base)
            if entry["empty"]:
                self.failures += 1
                self.empty_answers += 1
        _FROM_SPEC.set(True)
        if entry["empty"]:
            raise _empty_answer(model, entry["choice"], entry["usage"], max_tokens, price)
        return entry["text"], entry["usage"], price

    def _chat_speculative(self, fp, model, content, max_tokens, estimate_prompt_tokens, temperature,
                          timeout, reasoning, base):
        """Спекулятивный вызов: ответ в хранилище, деньги — в счёт хранилища.
        Тот же вопрос уже задан (или задаётся) другим упреждением — его ответ,
        без второй оплаты. Потолок: спекуляция не выходит за него вместе с
        уже потраченным и ещё не использованным."""
        limiter = self.spec_limiter()
        while True:
            with self._lock:
                dq = self._spec.get(fp)
                if dq:
                    entry = dq[0]
                    break
                ev = self._spec_pending.get(fp)
            if ev is not None:
                ev.wait()
                continue
            # Место в очереди упреждения берётся ДО того, как вопрос помечен
            # «в полёте»: настоящий цикл, спросивший то же, ждёт помеченный
            # вопрос — и не должен ждать его в очереди ограничителя.
            limiter.acquire()
            with self._lock:
                if self._spec.get(fp) or fp in self._spec_pending:
                    registered = False
                else:
                    self._spec_pending[fp] = threading.Event()
                    registered = True
            if registered:
                entry = None
                break
            limiter.release(ok=False)
        if entry is not None:
            if entry["empty"]:
                raise _empty_answer(model, entry["choice"], entry["usage"], max_tokens, entry["price"])
            return entry["text"], entry["usage"], entry["price"]
        ok = False
        ratio = max(self._ratio.get(model, 1.0), self._spec_ratio.get(model, 1.0))
        reserve = math.ceil(base * max(1.0, ratio))
        try:
            with self._lock:
                if self.spend_cap is not None and (self.spent + self.reserved + self.spec_reserved
                                                   + self.spec_outstanding + reserve
                                                   > self.spend_cap):
                    raise BudgetExhausted(f"упреждение: потолок {self.spend_cap}")
                self.spec_reserved += reserve

            def lost_body():
                with self._lock:
                    self.spec_spent += reserve
            try:
                body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": content}]}
                if reasoning is not None:
                    body.update(reasoning_switch(getattr(self, "_thinking", {}).get(model), reasoning))
                t0 = time.monotonic()
                try:
                    r = self._request("POST", "/chat/completions", body, timeout=timeout,
                                      on_lost_body=lost_body)
                finally:
                    with self._lock:
                        self.spec_net_seconds += time.monotonic() - t0
            except PaymentRequired as e:
                self.dead = str(e)
                raise
            finally:
                with self._lock:
                    self.spec_reserved -= reserve
            u = r.get("usage") or {}
            price = self.cost(model, u.get("prompt_tokens") or estimate_prompt_tokens,
                              u.get("completion_tokens") or max_tokens)
            choice = (r.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            entry = {"text": text, "usage": u, "price": price, "empty": not text.strip(),
                     "choice": {"finish_reason": choice.get("finish_reason")}}
            with self._lock:
                self._spec[fp].append(entry)
                self.spec_spent += price
                self.spec_outstanding += price
                self.spec_calls += 1
                if base > 0:
                    self._spec_ratio[model] = max(self._spec_ratio.get(model, 0.0), price / base)
            ok = True
        finally:
            with self._lock:
                self._spec_pending.pop(fp).set()
            limiter.release(ok=ok)
        if entry["empty"]:
            raise _empty_answer(model, entry["choice"], u, max_tokens, price)
        return text, u, price

    def image(self, model, prompt, size, quality=None, n=1, timeout=300):
        """Сгенерировать картинки. Возвращает (список байтов картинок, цена).

        Потолок расходов проверяется ДО вызова по полной цене заказа (шлюз
        берёт за доставленные картинки, поэтому резерв — верхняя граница).
        Отказ сервиса по содержанию запроса (400) — GatewayError без
        повторов: повтор того же промпта ответил бы тем же. Ответ без
        картинок — EmptyAnswer: «успех без картинки» шлюз не берёт в счёт,
        но для вызывающего это отказ, а не пустой кадр."""
        t0 = time.monotonic()
        try:
            images, price = self._image(model, prompt, size, quality, n, timeout)
        except Exception as e:
            _record_call("image", model, t0, False, err=e, n=n)
            raise
        _record_call("image", model, t0, True, price=price, n=n, got=len(images))
        return images, price

    def _image(self, model, prompt, size, quality, n, timeout):
        import base64
        if self.dead:
            raise GatewayError(f"шлюз выключен до конца прогона: {self.dead}")
        if not self.configured:
            raise GatewayError("нет LLM_GATEWAY_API_KEY")
        reserve = self.image_cost(model, size, quality, n)
        with self._lock:
            if self.spend_cap is not None and self.spent + self.reserved + reserve > self.spend_cap:
                raise BudgetExhausted(f"потолок {self.spend_cap}: потрачено {self.spent}, "
                                      f"в полёте {self.reserved}, нужно ещё до {reserve}")
            self.reserved += reserve
        body = {"model": model, "prompt": prompt, "n": n, "size": size, "response_format": "b64_json"}
        if quality:
            body["quality"] = quality

        def lost_body():
            with self._lock:
                self.spent += reserve
                self.lost_bodies += 1
        try:
            r = self._request("POST", "/images/generations", body, timeout=timeout,
                              on_lost_body=lost_body)
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
        images = [base64.b64decode(d["b64_json"]) for d in (r.get("data") or []) if d.get("b64_json")]
        price = self.image_cost(model, size, quality, len(images)) if images else 0
        with self._lock:
            self.spent += price
            self.calls += 1
        if not images:
            with self._lock:
                self.failures += 1
                self.empty_answers += 1
            raise EmptyAnswer(f"{model}: ответ без картинок")
        return images, price

    def summary(self):
        return {"base_url": self.base_url, "calls": self.calls, "failures": self.failures,
                "lost_bodies": self.lost_bodies, "empty_answers": self.empty_answers,
                "spent": self.spent, "spend_cap": self.spend_cap, "dead": self.dead,
                "pause_wait_sec": round(self.waited, 1), "reasked": self.reasked,
                "empty_answer_reasked": self.empty_answer_reasked,
                "speculative_calls": self.spec_calls, "speculative_used": self.spec_used,
                "speculative_spent": self.spec_spent,
                "speculative_wasted": self.spec_outstanding,
                "net_seconds": round(self.net_seconds, 1),
                "speculative_net_seconds": round(self.spec_net_seconds, 1),
                **({"speculative_limit_peak": self.spec_limiter().stats["peak_limit"],
                    "speculative_limit_now": self.spec_limiter().limit,
                    "throttles_429": self.spec_limiter().stats["throttles"]}
                   if self.spec_calls else {})}
