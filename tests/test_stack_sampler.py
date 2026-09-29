#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Семплер стеков: находит функцию, в которой поток проводит время."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import stack_sampler as ss  # noqa: E402


def _busy_here(stop):
    while not stop.is_set():
        sum(range(2000))


def test_sampler_names_the_hot_function_and_writes_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SAMPLE_SEC", 0.01)
    monkeypatch.setattr(ss, "FLUSH_SEC", 0.05)
    out = tmp_path / "profile_stacks.txt"
    halt = threading.Event()
    worker = threading.Thread(target=_busy_here, args=(halt,), name="worker")
    worker.start()
    stop = ss.start(str(out))
    time.sleep(0.4)
    stop()
    time.sleep(0.1)
    halt.set()
    worker.join()
    text = out.read_text(encoding="utf-8")
    assert "_busy_here" in text and "[worker]" in text
    assert text.startswith("# семплов")
