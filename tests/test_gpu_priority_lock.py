"""Замок карты: настоящий цикл слота идёт к карте раньше упреждения (B1,
прогон 30.09 на A40 — упреждение занимало карту, пока текущий слот ждал)."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import ml_device  # noqa: E402
import source_health  # noqa: E402


def _waiter(lock, background, order, name, started):
    def body():
        tok = source_health.BACKGROUND.set(background)
        try:
            started.set()
            with lock:
                order.append(name)
                time.sleep(0.01)
        finally:
            source_health.BACKGROUND.reset(tok)
    return threading.Thread(target=body)


def test_foreground_goes_before_background_that_waited_longer():
    for _ in range(20):
        lock = ml_device.PriorityLock()
        order = []
        lock.acquire(background=False)          # карту держит чужой прогон
        bgs = []
        for k in range(3):                       # упреждение встало в очередь первым
            ev = threading.Event()
            t = _waiter(lock, True, order, f"bg{k}", ev)
            t.start()
            ev.wait()
            bgs.append(t)
        time.sleep(0.05)
        ev = threading.Event()
        fg = _waiter(lock, False, order, "fg", ev)
        fg.start()
        ev.wait()
        deadline = time.time() + 2
        while lock._fg_waiting == 0 and time.time() < deadline:
            time.sleep(0.001)
        lock.release()
        for t in bgs + [fg]:
            t.join(5)
        assert order[0] == "fg", order
        assert sorted(order) == ["bg0", "bg1", "bg2", "fg"]


def test_background_runs_when_card_is_free():
    lock = ml_device.PriorityLock()
    tok = source_health.BACKGROUND.set(True)
    try:
        with lock:
            assert lock.locked()
    finally:
        source_health.BACKGROUND.reset(tok)
    assert not lock.locked()


def test_one_holder_at_a_time():
    lock = ml_device.PriorityLock()
    inside, peak = [0], [0]
    guard = threading.Lock()

    def body(bg):
        tok = source_health.BACKGROUND.set(bg)
        try:
            for _ in range(50):
                with lock:
                    with guard:
                        inside[0] += 1
                        peak[0] = max(peak[0], inside[0])
                    with guard:
                        inside[0] -= 1
        finally:
            source_health.BACKGROUND.reset(tok)

    ts = [threading.Thread(target=body, args=(k % 2 == 0,)) for k in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert peak[0] == 1


def test_release_of_free_lock_is_an_error():
    lock = ml_device.PriorityLock()
    try:
        lock.release()
    except RuntimeError:
        return
    raise AssertionError("release без acquire должен падать")


def test_device_locks_are_priority_locks():
    assert isinstance(ml_device._lock("cuda:0"), ml_device.PriorityLock)
    assert isinstance(ml_device._lock("cuda:1"), ml_device.PriorityLock)
