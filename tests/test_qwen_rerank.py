#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Реранкер Qwen3-VL: входы пар готовятся параллельно на процессоре, проходы
по видеокарте — по одной паре (числа как в официальном коде), порядок
ответов — порядок картинок, кэш не пересчитывается."""
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import qwen_vl_rerank as rr  # noqa: E402


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    from PIL import Image
    monkeypatch.setenv("RERANK_CACHE_DIR", str(tmp_path / "rc"))
    monkeypatch.setattr(rr, "_MEMO", {})
    monkeypatch.setattr(rr, "_load", lambda: True)
    monkeypatch.setitem(rr._STATE, "model", object())
    monkeypatch.setitem(rr._STATE, "processor", object())
    calls = {"prep_threads": set(), "forward": [], "barrier": threading.Barrier(2, timeout=5),
             "n": [0], "guard": threading.Lock()}

    def inputs(query, image, instruction):
        calls["prep_threads"].add(threading.get_ident())
        with calls["guard"]:
            calls["n"][0] += 1
            first_two = calls["n"][0] <= 2
        if first_two:
            try:
                calls["barrier"].wait()   # две подготовки идут одновременно
            except threading.BrokenBarrierError:
                pass
        return {"red": image.getpixel((0, 0))[0]}

    def forward(inp):
        calls["forward"].append(inp["red"])
        return inp["red"] / 255.0
    monkeypatch.setattr(rr, "_inputs", inputs)
    monkeypatch.setattr(rr, "_forward", forward)
    imgs = [Image.new("RGB", (64, 64), (r, 0, 0)) for r in (10, 200, 10, 90)]
    return calls, imgs


def test_order_is_kept_duplicates_scored_once_and_prep_runs_in_parallel(fake_model):
    calls, imgs = fake_model
    got = rr.score("a dagger", imgs + [os.path.join("нет", "файла.jpg")])
    assert got[:4] == pytest.approx([10 / 255, 200 / 255, 10 / 255, 90 / 255])
    assert got[4] is None, "нечитаемый файл — None на его месте"
    assert sorted(calls["forward"]) == [10, 90, 200], "одинаковая пара — один проход"
    assert len(calls["prep_threads"]) >= 2, "подготовка входов не параллельна"


def test_second_call_is_served_from_cache(fake_model):
    calls, imgs = fake_model
    rr.score("a dagger", imgs)
    calls["forward"].clear()
    rr.score("a dagger", imgs)
    assert calls["forward"] == []
    rr.score("another query", imgs[:1])
    assert calls["forward"] == [10], "другой запрос — другой ключ"


def test_forward_failure_turns_the_reranker_off(fake_model, monkeypatch):
    calls, imgs = fake_model
    monkeypatch.setitem(rr._STATE, "broken", None)

    def boom(inp):
        raise RuntimeError("CUDA error")
    monkeypatch.setattr(rr, "_forward", boom)
    assert rr.score("q", imgs) is None
    assert rr._STATE["model"] is None and "CUDA error" in rr._STATE["broken"]
