"""Очередь хоста: настоящий цикл не ждёт за фоновыми запросами (прогон 30.09
на L40: превью упреждения занимали очередь Викимедии на минуты вперёд)."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import source_health as sh  # noqa: E402

IV = 0.05


def _bg_call(host, stamps, lock):
    tok = sh.BACKGROUND.set(True)
    try:
        host.wait(IV)
        with lock:
            stamps.append(time.monotonic())
    finally:
        sh.BACKGROUND.reset(tok)


def test_foreground_does_not_queue_behind_background():
    host = sh.Host("t_prio", IV)
    stamps, lock = [], threading.Lock()
    bgs = [threading.Thread(target=_bg_call, args=(host, stamps, lock)) for _ in range(10)]
    for t in bgs:
        t.start()
    time.sleep(0.01)
    t0 = time.monotonic()
    host.wait(IV)                       # настоящий цикл
    fg_wait = time.monotonic() - t0
    with lock:
        stamps.append(time.monotonic())
    for t in bgs:
        t.join(5)
    assert fg_wait < 3 * IV, f"текущий слот ждал {fg_wait:.2f} с за фоном"
    s = sorted(stamps)
    assert len(s) == 11
    gaps = [b - a for a, b in zip(s, s[1:])]
    assert min(gaps) >= IV * 0.8, f"частота запросов к хосту выросла: {min(gaps):.3f}"


def test_background_alone_keeps_the_interval():
    host = sh.Host("t_bg", IV)
    stamps, lock = [], threading.Lock()
    ts = [threading.Thread(target=_bg_call, args=(host, stamps, lock)) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    s = sorted(stamps)
    assert len(s) == 6 and min(b - a for a, b in zip(s, s[1:])) >= IV * 0.8
    assert host.stats["requests"] == 6


def test_frozen_clock_does_not_hang_background(monkeypatch):
    host = sh.Host("t_frozen", 1.0)
    monkeypatch.setattr(sh.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(sh.time, "sleep", lambda s: None)
    host.wait()
    tok = sh.BACKGROUND.set(True)
    try:
        host.wait()                     # часы стоят: очередь по-старому, без зависания
    finally:
        sh.BACKGROUND.reset(tok)
    assert host.stats["requests"] == 2
