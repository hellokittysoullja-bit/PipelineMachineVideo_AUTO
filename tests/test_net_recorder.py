#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запись/воспроизведение сети (scripts/net_recorder.py) — без сети.

«Настоящий» urlopen подменяется фальшивым с управляемыми ответами. Запись
идёт через него, воспроизведение — без него вовсе: если воспроизведение
хоть раз обратится к «сети», фальшивка это заметит.

Каждый тест проверяет то, от чего зависит эквивалентность отбора: тот же
код, те же заголовки, то же тело, ТОТ ЖЕ ТИП исключения (код, ловящий
таймаут отдельно от обрыва, обязан повести себя так же) и та же
ПОСЛЕДОВАТЕЛЬНОСТЬ ответов на повторы одного адреса (429, затем 200).
"""
import http.client
import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import urllib.response

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import net_recorder as nr  # noqa: E402


def _msg(pairs):
    return nr._headers_message(pairs)


class FakeNet:
    """Сценарий «сети»: адрес -> список исходов по порядку обращений."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.hits = []

    def __call__(self, url_or_req, data=None, *a, **kw):
        url = url_or_req.full_url if isinstance(url_or_req, urllib.request.Request) else url_or_req
        self.hits.append(url)
        outcome = self.script[url].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        status, headers, body = outcome
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "err", _msg(headers), io.BytesIO(body))
        return urllib.response.addinfourl(io.BytesIO(body), _msg(headers), url, status)


@pytest.fixture
def restore_urlopen():
    real = urllib.request.urlopen
    yield
    urllib.request.urlopen = real


def _record(tmp_path, fake, calls):
    urllib.request.urlopen = fake
    rec = nr.NetRecorder(str(tmp_path / "net"), nr.RECORD).install()
    out = []
    try:
        for c in calls:
            out.append(_observe(c))
    finally:
        rec.uninstall()
    return out


def _replay(tmp_path, calls):
    def no_network(*a, **kw):
        raise AssertionError("воспроизведение обратилось к сети")
    urllib.request.urlopen = no_network
    rep = nr.NetRecorder(str(tmp_path / "net"), nr.REPLAY).install()
    out = []
    try:
        for c in calls:
            out.append(_observe(c))
    finally:
        rep.uninstall()
    return out, rep


def _observe(req):
    """Всё, что вызывающий код может увидеть у ответа или отказа."""
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return ("ok", r.status, r.getcode(), r.headers.get("Content-Type"),
                    r.headers.get("Retry-After"), r.read())
    except urllib.error.HTTPError as e:
        return ("http", e.code, e.headers.get("Retry-After"), e.read())
    except Exception as e:  # noqa: BLE001
        return ("exc", type(e).__module__, type(e).__name__, str(e))


def test_response_replays_identically(tmp_path, restore_urlopen):
    fake = FakeNet({"https://a/1": [(200, [("Content-Type", "image/jpeg")], b"\xff\xd8JPEG")]})
    rec = _record(tmp_path, fake, ["https://a/1"])
    rep, r = _replay(tmp_path, ["https://a/1"])
    assert rec == rep
    assert rep[0][0] == "ok" and rep[0][5] == b"\xff\xd8JPEG"
    assert not r.divergences


def test_http_error_replays_with_code_headers_and_body(tmp_path, restore_urlopen):
    fake = FakeNet({"https://a/rl": [(429, [("Retry-After", "7")], b"slow down")]})
    rec = _record(tmp_path, fake, ["https://a/rl"])
    rep, _ = _replay(tmp_path, ["https://a/rl"])
    assert rec == rep == [("http", 429, "7", b"slow down")]


def test_retry_sequence_on_one_address_is_preserved(tmp_path, restore_urlopen):
    """Повтор после 429 обязан получить ВТОРОЙ записанный ответ."""
    fake = FakeNet({"https://a/x": [(429, [], b"no"), (200, [], b"yes")]})
    rec = _record(tmp_path, fake, ["https://a/x", "https://a/x"])
    rep, _ = _replay(tmp_path, ["https://a/x", "https://a/x"])
    assert rec == rep
    assert rep[0][0] == "http" and rep[1][0] == "ok" and rep[1][5] == b"yes"


@pytest.mark.parametrize("exc", [
    TimeoutError("timed out"),
    ConnectionResetError("reset by peer"),
    http.client.RemoteDisconnected("closed"),
    urllib.error.URLError("dns failure"),
])
def test_exception_type_is_preserved(tmp_path, restore_urlopen, exc):
    fake = FakeNet({"https://a/e": [exc]})
    rec = _record(tmp_path, fake, ["https://a/e"])
    rep, _ = _replay(tmp_path, ["https://a/e"])
    assert rec[0][:3] == rep[0][:3], f"тип исключения изменился: {rec[0]} -> {rep[0]}"


def test_unrecorded_request_is_a_named_divergence_not_silence(tmp_path, restore_urlopen):
    fake = FakeNet({"https://a/1": [(200, [], b"x")]})
    _record(tmp_path, fake, ["https://a/1"])
    rep, r = _replay(tmp_path, ["https://a/1", "https://a/NEW", "https://a/1"])
    assert rep[1][0] == "exc" and rep[1][2] == "URLError"
    assert rep[2][0] == "exc", "второе обращение к адресу, записанному один раз, — тоже расхождение"
    urls = [d["url"] for d in r.divergences]
    assert urls == ["https://a/NEW", "https://a/1"]


def test_post_body_is_part_of_the_key(tmp_path, restore_urlopen):
    fake = FakeNet({"https://a/q": [(200, [], b"one"), (200, [], b"two")]})
    r1 = urllib.request.Request("https://a/q", data=b'{"q":1}', method="POST")
    r2 = urllib.request.Request("https://a/q", data=b'{"q":2}', method="POST")
    _record(tmp_path, fake, [r1, r2])
    # в обратном порядке: ключ по телу, а не по порядку обращений
    rep, r = _replay(tmp_path, [
        urllib.request.Request("https://a/q", data=b'{"q":2}', method="POST"),
        urllib.request.Request("https://a/q", data=b'{"q":1}', method="POST")])
    assert [x[5] for x in rep] == [b"two", b"one"]
    assert not r.divergences


def test_secrets_never_reach_the_index(tmp_path, restore_urlopen, monkeypatch):
    secret = "sk_live_0123456789abcdef"
    monkeypatch.setenv("PIXABAY_API_KEY", secret)
    url = f"https://pixabay.com/api/?key={secret}&q=knight"
    fake = FakeNet({url: [(200, [("Set-Cookie", "session=zzz"), ("Content-Type", "application/json")],
                           b"{}")]})
    _record(tmp_path, fake, [url])
    raw = (tmp_path / "net" / "index.jsonl").read_text(encoding="utf-8")
    assert secret not in raw, "ключ API попал в запись"
    assert "<PIXABAY_API_KEY>" in raw
    assert "session=zzz" not in raw, "Set-Cookie попал в запись"


def test_identical_bodies_are_stored_once(tmp_path, restore_urlopen):
    fake = FakeNet({"https://a/1": [(200, [], b"same")], "https://b/2": [(200, [], b"same")]})
    _record(tmp_path, fake, ["https://a/1", "https://b/2"])
    files = [f for _d, _s, fs in os.walk(tmp_path / "net" / "bodies") for f in fs]
    assert len(files) == 1


def test_concurrent_recording_loses_nothing(tmp_path, restore_urlopen):
    """Превью качаются пулом потоков — запись не имеет права терять строки."""
    urls = [f"https://a/{i}" for i in range(64)]
    fake = FakeNet({u: [(200, [], u.encode())] for u in urls})
    urllib.request.urlopen = fake
    rec = nr.NetRecorder(str(tmp_path / "net"), nr.RECORD).install()
    try:
        threads = [threading.Thread(target=_observe, args=(u,)) for u in urls]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        rec.uninstall()
    lines = (tmp_path / "net" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(urls)
    assert all(json.loads(line)["kind"] == "response" for line in lines)
    rep, r = _replay(tmp_path, urls)
    assert [x[5] for x in rep] == [u.encode() for u in urls]
    assert not r.divergences


def test_overlay_serves_missing_requests_live_and_names_them(tmp_path, restore_urlopen):
    """Новый код законно просит файл, которого старый не просил: запрос
    выполняется живьём, пишется в отдельный слой и остаётся названным
    расхождением — не тихой подменой записи."""
    fake = FakeNet({"https://a/1": [(200, [], b"old")], "https://a/new": [(200, [], b"fresh")]})
    urllib.request.urlopen = fake
    rec = nr.NetRecorder(str(tmp_path / "net"), nr.RECORD).install()
    try:
        _observe("https://a/1")
    finally:
        rec.uninstall()
    slot = {"i": 3}
    rep = nr.NetRecorder(str(tmp_path / "net"), nr.REPLAY, overlay=str(tmp_path / "overlay"),
                         tagger=lambda: slot["i"]).install()
    try:
        got_old = _observe("https://a/1")
        got_new = _observe("https://a/new")
    finally:
        rep.uninstall()
    assert got_old[5] == b"old" and got_new[5] == b"fresh"
    assert fake.hits == ["https://a/1", "https://a/new"], "записанный запрос ушёл в сеть повторно"
    assert [(d["url"], d["served"], d["slot"]) for d in rep.divergences] == [("https://a/new", "live", 3)]
    overlay = (tmp_path / "overlay" / "index.jsonl").read_text(encoding="utf-8")
    assert "https://a/new" in overlay and "https://a/1" not in overlay


def test_calls_are_labelled_by_slot(tmp_path, restore_urlopen):
    fake = FakeNet({"https://a/x": [(200, [], b"1"), (200, [], b"2"), (200, [], b"3")]})
    urllib.request.urlopen = fake
    state = {"slot": None}
    rec = nr.NetRecorder(str(tmp_path / "net"), nr.RECORD, tagger=lambda: state["slot"]).install()
    try:
        _observe("https://a/x")
        state["slot"] = 2
        _observe("https://a/x")
        _observe("https://a/x")
    finally:
        rec.uninstall()
    (by_slot,) = rec.summary()["slots_by_key"].values()
    assert by_slot == {"2": 2, "none": 1}
