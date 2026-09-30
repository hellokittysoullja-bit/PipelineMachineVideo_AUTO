#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Надёжность запуска платных подов (аудит 30.09): цена и наличие по выбранному
облаку, отказ по видеопамяти как отказ типа карты, результат при исчерпанном
потолке, сигналы, код выхода подготовки. Ни одного настоящего пода."""
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import runpod_job as rj  # noqa: E402


@pytest.fixture(autouse=True)
def _own_smoke_mark(tmp_path, monkeypatch):
    monkeypatch.setattr(rj, "SMOKE_MARK", str(tmp_path / "smoke"), raising=False)


def test_plan_filters_both_clouds(monkeypatch):
    seen = []
    monkeypatch.setattr(rj, "gql", lambda q, key, v=None: seen.append(q) or {})
    rj.plan("k", ["X"], community=True)
    rj.plan("k", ["X"], community=False)
    rj.plan("k", None, community=False)
    assert "secureCloud:false" in seen[0]
    assert "secureCloud:true" in seen[1] and "secureCloud:false" not in seen[1]
    assert "secureCloud:true" in seen[2] and "secureCloud" in seen[2].split("lowestPrice")[0]


def _types():
    return [
        {"id": "A", "displayName": "A", "memoryInGb": 48, "communityCloud": True, "secureCloud": False,
         "lowestPrice": {"uninterruptablePrice": 0.5, "stockStatus": "Low"}},
        {"id": "B", "displayName": "B", "memoryInGb": 48, "communityCloud": False, "secureCloud": True,
         "lowestPrice": {"uninterruptablePrice": 0.6, "stockStatus": "Low"}},
    ]


def test_cheapest_gpus_follows_the_selected_cloud():
    assert rj.cheapest_gpus(_types(), 24, rj.DEFAULT_IMAGE, community=True) == ["A"]
    assert rj.cheapest_gpus(_types(), 24, rj.DEFAULT_IMAGE, community=False) == ["B"]
    assert rj.cheapest_gpus(_types(), 24, rj.DEFAULT_IMAGE) == ["A"]


def test_min_gb_follows_the_vram_threshold():
    assert rj.min_gb_for_vram(24, 0) == 24
    assert rj.min_gb_for_vram(24, 47000) == 46
    assert rj.min_gb_for_vram(60, 47000) == 60


def _args(**kw):
    base = dict(max_usd=1.0, max_hours=1.0, idle_min=4.0, image="i", disk_gb=10, cloud="COMMUNITY",
                cmd="c", fetch=[], dest=".", prepare=None, min_vram_mib=47000, watch=())
    base.update(kw)
    return SimpleNamespace(**base)


def _rent(monkeypatch, gpus, attempts=3):
    """_rent_attempts, где каждый под отказывает по видеопамяти."""
    pods = iter([{"id": f"p{i}", "gpuName": g, "gpuTypeId": g, "costPerHr": "0.5", "spent_usd": 0.01}
                 for i, g in enumerate(gpus)])
    monkeypatch.setattr(rj, "create_when_in_stock", lambda *a, **k: next(pods))

    def drive(*a, **k):
        raise rj.LowVram("мало памяти")
    monkeypatch.setattr(rj, "drive", drive)
    order = ["A", "B"]
    with pytest.raises(SystemExit) as e:
        rj._rent_attempts("k", _args(), order, {}, "t", {}, attempts, [], 0.0)
    return order, str(e.value)


def test_low_vram_removes_the_card_type_instead_of_retrying_it(monkeypatch):
    order, msg = _rent(monkeypatch, ["A", "B"])
    assert order == []
    assert "не вмещает" in msg


def test_ordinary_bad_host_still_moves_the_type_to_the_end(monkeypatch):
    pods = iter([{"id": "p0", "gpuName": "A", "gpuTypeId": "A", "costPerHr": "0.5", "spent_usd": 0.0},
                 {"id": "p1", "gpuName": "B", "gpuTypeId": "B", "costPerHr": "0.5", "spent_usd": 0.0}])
    monkeypatch.setattr(rj, "create_when_in_stock", lambda *a, **k: next(pods))
    calls = []

    def drive(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise rj.BadHost("не работает")
        return 0
    monkeypatch.setattr(rj, "drive", drive)
    order = ["A", "B"]
    assert rj._rent_attempts("k", _args(), order, {}, "t", {}, 3, [], 0.0) == 0
    assert order == ["B", "A"]


class _Runner:
    fetched = []

    def __init__(self, *a, **k):
        pass

    def wait_ready(self, *a, **k):
        return True

    def run(self, cmd, deadline=None):
        raise rj.BudgetExpired("потолок денег на запуск исчерпан — задача прервана, под удаляется")

    follow = start = call = lambda self, *a, **k: (_ for _ in ()).throw(rj.BudgetExpired("потолок"))

    def fetch(self, path, dest):
        _Runner.fetched.append(path)


def _drive(monkeypatch, events):
    _Runner.fetched = []
    monkeypatch.setattr(rj, "Runner", _Runner)
    monkeypatch.setattr(rj, "terminate", lambda key, pid: events.append(("terminate", pid)))
    monkeypatch.setattr(signal, "signal", lambda sig, h: events.append(("signal", sig, h)))
    return rj.drive("k", {"id": "p1", "costPerHr": "0.5"}, "t", 100, [], "cmd", ["a/b"], "dest")


def test_budget_expiry_still_fetches_results_and_removes_the_pod(monkeypatch, capsys):
    events = []
    code = _drive(monkeypatch, events)
    assert code == 124
    assert _Runner.fetched == ["a/b"], "результат оплаченного прогона потерян"
    assert ("terminate", "p1") in events


def test_both_signals_are_handled_and_ignored_while_the_pod_is_removed(monkeypatch, capsys):
    events = []
    _drive(monkeypatch, events)
    sigs = [e for e in events if e[0] == "signal"]
    installed = {s for _k, s, h in sigs if callable(h)}
    assert {signal.SIGINT, signal.SIGTERM} <= installed
    t = events.index(("terminate", "p1"))
    before = [e for e in events[:t] if e[0] == "signal"]
    assert {s for _k, s, h in before[-2:] if h == signal.SIG_IGN} == {signal.SIGINT, signal.SIGTERM}


def test_pod_prepare_exits_nonzero_when_install_or_weights_fail():
    path = os.path.join(REPO, "scripts", "pod_prepare.sh")
    src = open(path, encoding="utf-8").read()
    assert subprocess.run(["bash", "-n", path]).returncode == 0
    assert "|| { say \"ОШИБКА: установка пакетов не удалась\"; exit 1; }" in src.replace("\\\n", "").replace("    ||", "||")
    assert 'fetch_weights.py || { say "ОШИБКА: веса моделей не скачались"; exit 1; }' in src
