#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запись и воспроизведение сети на границе процесса.

Вся сеть отбора кадра идёт через `urllib.request.urlopen(...)` — проверено
по всем модулям пути отбора (pipeline_smart, museum_sources, shelf_index,
europeana_corpus, stock_fetch_multisource, shot_director): ни одного
`requests`, ни одного `from urllib.request import urlopen`, вызов всегда через
атрибут модуля. Поэтому одна подмена атрибута в процессе ловит 100% сетевых
вызовов, не трогая ни строки продакшн-кода.

ЗАПИСЬ: настоящий запрос выполняется, ответ (тело, код, заголовки) или отказ
(HTTPError, таймаут, обрыв) сохраняется и отдаётся вызывающему коду в
неизменном виде.

ВОСПРОИЗВЕДЕНИЕ: ни одного сетевого вызова. Ответ берётся из записи.

КЛЮЧ ЗАПРОСА — метод + полный URL + хэш тела. К одному ключу хранится
ПОСЛЕДОВАТЕЛЬНОСТЬ ответов в порядке обращений: повтор после 429 получает
второй записанный ответ, а не первый снова. Порядок между РАЗНЫМИ ключами не
фиксируется намеренно: превью кандидатов качаются параллельно
(PHOTO_PREFETCH_WORKERS), и глобальный порядок вызовов недетерминирован по
построению, тогда как исход отбора от него не зависит (обработка идёт
последовательно, см. prefetch в pexels_photo).

ГИБРИДНЫЙ РЕЖИМ (overlay): запрос, которого нет в записи, выполняется
живьём и записывается в ОТДЕЛЬНЫЙ слой, а в divergences остаётся с
пометкой served="live". Нужен, когда новый код законно выбирает другой кадр
и просит файл, которого старый не просил: чистое воспроизведение тут
отказало бы на скачивании и исказило бы исход, а живая перезапись всего
эпизода смешала бы правку кода с дрейфом выдачи стоков.

МЕТКА СЛОТА. tagger (если задан) возвращает номер слота, в котором идёт
обращение; сводка раскладывает обращения по слотам (slots_by_key), и
сравнение может требовать, чтобы сеть расходилась только в слотах с
названной причиной.

ЗАПРОС, КОТОРОГО НЕТ В ЗАПИСИ, при воспроизведении — не тишина. Он попадает
в `divergences` с ключом и адресом, а вызывающему коду уходит URLError:
продакшн-код обработает его как обычный сетевой отказ, и расхождение
проявится в исходе отбора, где его и поймает сравнение. Молча подставить
«что-нибудь похожее» было бы ровно той подделкой эквивалентности, против
которой харнесс и строится.

СЕКРЕТЫ. Ключи API бывают в самом URL (у Pixabay — параметр `key=`). В
индексе URL хранится с подстановкой `<NAME>` вместо значения каждой
переменной окружения, похожей на секрет (имя содержит KEY/TOKEN/SECRET/
PASSWORD/CLIENT_ID). Список берётся из окружения, а не из кода: новый ключ
нового источника закрывается сам. Ключ записи — хэш, из него URL не
восстанавливается. Заголовки запроса не хранятся вовсе; из заголовков ответа
выбрасывается Set-Cookie.
"""
import builtins
import hashlib
import http.client
import io
import json
import os
import re
import threading
import urllib.error
import urllib.request
import urllib.response

RECORD = "record"
REPLAY = "replay"

_SECRET_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CLIENT_ID)", re.I)
_DROPPED_RESPONSE_HEADERS = {"set-cookie"}


def _secrets_from_env():
    out = []
    for name, value in os.environ.items():
        if _SECRET_NAME_RE.search(name) and value and len(value) >= 8:
            out.append((name, value))
    # длинные первыми: значение, входящее в другое, не должно заменить его часть
    return sorted(out, key=lambda kv: -len(kv[1]))


def redact(text, secrets):
    for name, value in secrets:
        text = text.replace(value, f"<{name}>")
    return text


def _request_parts(url_or_req, data):
    if isinstance(url_or_req, urllib.request.Request):
        url = url_or_req.full_url
        method = url_or_req.get_method()
        body = url_or_req.data if url_or_req.data is not None else data
    else:
        url = str(url_or_req)
        method = "POST" if data is not None else "GET"
        body = data
    if isinstance(body, str):
        body = body.encode("utf-8")
    return method, url, body


def request_key(method, url, body):
    h = hashlib.sha256()
    h.update(method.encode())
    h.update(b"\n")
    h.update(url.encode("utf-8"))
    h.update(b"\n")
    h.update(hashlib.sha256(body or b"").digest())
    return h.hexdigest()


def _headers_message(pairs):
    raw = "".join(f"{k}: {v}\r\n" for k, v in pairs) + "\r\n"
    return http.client.parse_headers(io.BytesIO(raw.encode("latin-1", errors="replace")))


def _rebuild_exception(rec):
    """Тот же ТИП исключения, что был при записи: код, ловящий конкретный
    класс (например, таймаут отдельно от обрыва), обязан повести себя так
    же. Класс ищется по записанному модулю и имени; сконструировать не
    вышло — URLError с тем же текстом, и это единственный откат."""
    import importlib
    name, module, msg = rec.get("exc_type", ""), rec.get("exc_module", "builtins"), rec.get("exc_msg", "")
    cls = None
    try:
        cls = getattr(importlib.import_module(module), name, None)
    except Exception:
        cls = getattr(builtins, name, None)
    exc = None
    if isinstance(cls, type) and issubclass(cls, BaseException):
        try:
            exc = cls(msg)
        except Exception:
            exc = None
    return exc if exc is not None else urllib.error.URLError(msg)


class NetRecorder:
    """Подмена `urllib.request.urlopen` на время жизни процесса.

    root/index.jsonl  — по строке на ответ: ключ, номер в последовательности,
                        вид (response/http_error/exception), код, заголовки,
                        адрес с вырезанными секретами, хэш тела;
    root/bodies/xx/…  — тела, адресуемые хэшем содержимого: одна и та же
                        картинка в разных слотах хранится один раз.
    """

    def __init__(self, root, mode, overlay=None, tagger=None):
        if mode not in (RECORD, REPLAY):
            raise ValueError(f"неизвестный режим {mode!r}")
        if overlay is not None and mode != REPLAY:
            raise ValueError("слой живых запросов бывает только у воспроизведения")
        self.root = root
        self.mode = mode
        self.tagger = tagger
        self.slot_calls = {}     # ключ -> {метка слота: число обращений}
        self._overlay = NetRecorder(overlay, RECORD) if overlay is not None else None
        self.index_path = os.path.join(root, "index.jsonl")
        self.bodies = os.path.join(root, "bodies")
        self._lock = threading.Lock()
        self._seq = {}           # ключ -> сколько раз уже обращались
        self._recorded = {}      # ключ -> [запись, ...] по seq (только replay)
        self.divergences = []    # запросы, которых нет в записи (только replay)
        self.calls = {}          # ключ -> число обращений в ЭТОМ прогоне
        self.secrets = _secrets_from_env()
        self._real = None
        os.makedirs(self.bodies, exist_ok=True)
        if mode == REPLAY:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"нет записи сети: {self.index_path}")
            with open(self.index_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self._recorded.setdefault(rec["key"], []).append(rec)
            for recs in self._recorded.values():
                recs.sort(key=lambda r: r["seq"])

    # -- хранилище тел -------------------------------------------------------

    def _body_path(self, digest):
        return os.path.join(self.bodies, digest[:2], digest)

    def _store_body(self, body):
        digest = hashlib.sha256(body).hexdigest()
        path = self._body_path(digest)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, path)
        return digest

    def _load_body(self, digest):
        with open(self._body_path(digest), "rb") as f:
            return f.read()

    def _append(self, rec):
        line = json.dumps(rec, ensure_ascii=False)
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # -- подмена --------------------------------------------------------------

    def install(self):
        if self._real is not None:
            raise RuntimeError("транспорт уже установлен")
        self._real = urllib.request.urlopen
        if self._overlay is not None:
            self._overlay._real = self._real
        urllib.request.urlopen = self._urlopen
        return self

    def uninstall(self):
        if self._real is not None:
            urllib.request.urlopen = self._real
            self._real = None

    def _next_seq(self, key, tag=None):
        with self._lock:
            n = self._seq.get(key, 0)
            self._seq[key] = n + 1
            self.calls[key] = n + 1
            per = self.slot_calls.setdefault(key, {})
            label = "none" if tag is None else str(tag)
            per[label] = per.get(label, 0) + 1
            return n

    def _urlopen(self, url_or_req, data=None, timeout=None, *args, **kwargs):
        method, url, body = _request_parts(url_or_req, data)
        key = request_key(method, url, body)
        tag = self.tagger() if self.tagger is not None else None
        seq = self._next_seq(key, tag)
        shown = redact(url, self.secrets)
        if self.mode == RECORD:
            return self._record(key, seq, method, shown, url_or_req, data, timeout, args, kwargs)
        return self._replay(key, seq, method, shown, tag,
                            (url_or_req, data, timeout, args, kwargs))

    def _record(self, key, seq, method, shown, url_or_req, data, timeout, args, kwargs):
        base = {"key": key, "seq": seq, "method": method, "url": shown}
        call_kwargs = dict(kwargs)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        try:
            resp = self._real(url_or_req, data, *args, **call_kwargs)
        except urllib.error.HTTPError as e:
            payload = e.read() if e.fp is not None else b""
            pairs = [(k, v) for k, v in (e.headers.items() if e.headers else [])
                     if k.lower() not in _DROPPED_RESPONSE_HEADERS]
            with self._lock:
                self._append({**base, "kind": "http_error", "status": e.code,
                              "reason": str(e.reason), "headers": pairs,
                              "body": self._store_body(payload)})
            raise urllib.error.HTTPError(e.url or getattr(url_or_req, "full_url", str(url_or_req)),
                                         e.code, e.msg, _headers_message(pairs),
                                         io.BytesIO(payload))
        except Exception as e:
            with self._lock:
                self._append({**base, "kind": "exception", "exc_type": type(e).__name__,
                              "exc_module": type(e).__module__, "exc_msg": str(e)})
            raise
        with resp:
            payload = resp.read()
            status = getattr(resp, "status", None) or resp.getcode()
            pairs = [(k, v) for k, v in resp.headers.items()
                     if k.lower() not in _DROPPED_RESPONSE_HEADERS]
            final_url = resp.geturl()
        with self._lock:
            self._append({**base, "kind": "response", "status": status, "headers": pairs,
                          "final_url": redact(final_url or "", self.secrets),
                          "body": self._store_body(payload)})
        return urllib.response.addinfourl(io.BytesIO(payload), _headers_message(pairs),
                                          final_url, status)

    def _replay(self, key, seq, method, shown, tag=None, live=None):
        recs = self._recorded.get(key, [])
        if seq >= len(recs):
            served = "live" if self._overlay is not None else "refused"
            with self._lock:
                self.divergences.append({"key": key, "seq": seq, "method": method, "url": shown,
                                         "recorded": len(recs), "slot": tag, "served": served})
            if self._overlay is not None:
                url_or_req, data, timeout, args, kwargs = live
                oseq = self._overlay._next_seq(key, tag)
                return self._overlay._record(key, oseq, method, shown, url_or_req, data,
                                             timeout, args, kwargs)
            raise urllib.error.URLError(
                f"freeze: запроса нет в записи ({method} {shown}, обращение #{seq + 1}, "
                f"записано {len(recs)})")
        rec = recs[seq]
        kind = rec["kind"]
        if kind == "response":
            payload = self._load_body(rec["body"])
            return urllib.response.addinfourl(io.BytesIO(payload), _headers_message(rec["headers"]),
                                              rec.get("final_url") or shown, rec["status"])
        if kind == "http_error":
            payload = self._load_body(rec["body"])
            raise urllib.error.HTTPError(shown, rec["status"], rec.get("reason", ""),
                                         _headers_message(rec["headers"]), io.BytesIO(payload))
        raise _rebuild_exception(rec)

    # -- итог ------------------------------------------------------------------

    def summary(self):
        return {
            "mode": self.mode,
            "distinct_requests": len(self.calls),
            "total_calls": sum(self.calls.values()),
            "divergences": list(self.divergences),
            "calls_by_key": dict(sorted(self.calls.items())),
            "slots_by_key": {k: dict(sorted(v.items())) for k, v in sorted(self.slot_calls.items())},
        }
