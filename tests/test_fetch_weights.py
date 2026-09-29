#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Параллельная закачка весов: все модели отбора, WeMM с закреплённой ревизией."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import fetch_weights as fw  # noqa: E402
import wemm_embed  # noqa: E402


def test_model_list_pins_wemm_revision_and_can_skip_it():
    names = dict(fw.models())
    assert names[wemm_embed.MODEL_NAME] == wemm_embed.REVISION
    assert wemm_embed.MODEL_NAME not in dict(fw.models(wemm=False))
    assert len(fw.models(wemm=False)) == 2


def test_models_download_at_the_same_time(monkeypatch):
    inside = {"n": 0, "peak": 0}
    lock = threading.Lock()

    def slow(name, revision):
        with lock:
            inside["n"] += 1
            inside["peak"] = max(inside["peak"], inside["n"])
        time.sleep(0.2)
        with lock:
            inside["n"] -= 1
        return name, 0.2
    monkeypatch.setattr(fw, "fetch", slow)
    assert fw.main([]) == 0
    assert inside["peak"] == 3, "модели качаются по очереди, а не вместе"
