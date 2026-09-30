#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Решения по часам (остывание Мет) записываются и воспроизводятся.

Фальшивый модуль повторяет устройство museum_sources: _met_get спрашивает
«в остывании?» на входе, при 403 зовёт _met_enter_cooldown, который
спрашивает то же самое под своей блокировкой. Часы подменены управляемыми.
"""
import os
import sys
import threading
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import time_decisions as td  # noqa: E402


def _fake_module(clock, answers):
    ms = types.ModuleType("fake_ms")
    ms.until = [0.0]
    ms.log = []

    def met_is_cooling_down():
        return clock[0] < ms.until[0]

    def _met_enter_cooldown():
        if not ms.met_is_cooling_down():
            ms.until[0] = clock[0] + 60

    def _met_get(url):
        if ms.met_is_cooling_down():
            ms.log.append(("skip", url))
            return None
        ms.log.append(("get", url))
        if answers.get(url) == 403:
            ms._met_enter_cooldown()
            return None
        return {"ok": url}
    ms.met_is_cooling_down = met_is_cooling_down
    ms._met_enter_cooldown = _met_enter_cooldown
    ms._met_get = _met_get
    return ms


def test_replay_reproduces_skips_regardless_of_the_clock(tmp_path):
    path = str(tmp_path / "time.jsonl")
    clock = [0.0]
    ms = _fake_module(clock, {"a": 403})
    rec = td.MetCooldownRecorder(path, td.RECORD).install(ms)
    ms._met_get("a")          # 403 -> остывание до 60
    clock[0] = 10
    ms._met_get("b")          # в окне -> пропуск
    clock[0] = 100
    ms._met_get("c")          # окно прошло
    recorded_log = list(ms.log)
    assert [k for k, _ in recorded_log] == ["get", "skip", "get"]
    assert rec.summary()["decisions"] == 4   # вход a, остывание a, вход b, вход c

    # Воспроизведение идёт БЫСТРЕЕ: часы почти стоят, «b» по часам уже вне
    # окна — а исход обязан быть тем же, что при записи.
    clock2 = [0.0]
    ms2 = _fake_module(clock2, {"a": 403})
    rep = td.MetCooldownRecorder(path, td.REPLAY).install(ms2)
    ms2._met_get("a")
    clock2[0] = 1000
    ms2._met_get("b")
    ms2._met_get("c")
    assert ms2.log == recorded_log
    assert rep.summary()["divergences"] == []


def test_unrecorded_decision_is_named(tmp_path):
    path = str(tmp_path / "time.jsonl")
    clock = [0.0]
    ms = _fake_module(clock, {})
    td.MetCooldownRecorder(path, td.RECORD).install(ms)
    ms._met_get("a")
    ms2 = _fake_module([0.0], {})
    rep = td.MetCooldownRecorder(path, td.REPLAY).install(ms2)
    ms2._met_get("a")
    ms2._met_get("new")
    assert rep.summary()["divergences"] == [{"key": "get|0|new", "slot": None}]


def test_unrecorded_decision_carries_its_slot(tmp_path):
    """Без слота харнесс не может отнести новое решение к слоту с названной
    причиной — законно новый запрос в изменившемся слоте был бы провалом."""
    path = str(tmp_path / "time.jsonl")
    td.MetCooldownRecorder(path, td.RECORD).install(_fake_module([0.0], {}))
    ms2 = _fake_module([0.0], {})
    rep = td.MetCooldownRecorder(path, td.REPLAY).install(ms2)
    slot = {"i": 7}
    rep.tagger = lambda: slot["i"]
    ms2._met_get("new")
    assert rep.summary()["divergences"] == [{"key": "get|0|new", "slot": 7}]


def test_keys_do_not_depend_on_thread_interleaving(tmp_path):
    """Карточки Мет тянутся параллельно: ключ решения — адрес и номер
    обращения к нему, а не глобальный порядок."""
    path = str(tmp_path / "time.jsonl")
    ms = _fake_module([0.0], {})
    td.MetCooldownRecorder(path, td.RECORD).install(ms)
    urls = [f"u{i}" for i in range(40)]
    ts = [threading.Thread(target=ms._met_get, args=(u,)) for u in urls]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ms2 = _fake_module([0.0], {})
    rep = td.MetCooldownRecorder(path, td.REPLAY).install(ms2)
    for u in reversed(urls):
        ms2._met_get(u)
    assert rep.summary()["divergences"] == []
