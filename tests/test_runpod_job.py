#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задача на Runpod: исполнитель пода (поднимается здесь локально) и
локальная сторона — загрузка кусками с докачкой, запуск, лог, возврат
результатов, самоудаление пода по простою и по потолку, удаление пода в
любом исходе. API Runpod подменяется — ни одного настоящего пода."""

import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import runpod_job as rj  # noqa: E402

TOKEN = "t0ken"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def runner(tmp_path):
    port = _free_port()
    env = dict(os.environ, RUNNER_TOKEN=TOKEN, RUNNER_WORK=str(tmp_path / "work"),
               RUNNER_PORT=str(port), RUNNER_IDLE_SEC="3", RUNNER_MAX_SEC="3600",
               RUNNER_POLL_SEC="0.3", RUNNER_DRY_TERMINATE="1", RUNPOD_POD_ID="pod123")
    proc = subprocess.Popen([sys.executable, os.path.join(REPO, "scripts", "runpod_runner.py")],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    r = rj.Runner(f"http://127.0.0.1:{port}", TOKEN)
    assert r.wait_ready(20)
    yield r, proc, tmp_path / "work"
    proc.kill()
    proc.wait(5)


def test_requests_without_the_token_are_refused(runner):
    r, _proc, _work = runner
    bad = rj.Runner(r.base, "wrong")
    with pytest.raises(urllib.error.HTTPError) as e:
        bad.call("POST", "/run", json.dumps({"cmd": "echo hacked"}).encode())
    assert e.value.code == 403


def test_upload_run_log_and_fetch_round_trip(runner, tmp_path, monkeypatch, capsys):
    r, _proc, work = runner
    src = tmp_path / "proj"
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "job.py").write_text(
        "import os\nos.makedirs('out/sub', exist_ok=True)\n"
        "open('out/sub/result.txt','w').write('42')\nprint('готово')\n", encoding="utf-8")
    (src / ".env").write_text("SECRET=1", encoding="utf-8")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_bytes(b"x")
    monkeypatch.setattr(rj, "REPO", str(src))
    monkeypatch.setattr(rj, "CHUNK", 64)          # несколько кусков даже на крошечном архиве
    r.upload_dir(str(src))
    assert (work / "scripts" / "job.py").exists()
    assert not (work / ".env").exists(), "секреты не должны уезжать в под"
    assert not (work / "__pycache__").exists()
    assert r.run(f"{sys.executable} scripts/job.py") == 0
    assert "готово" in capsys.readouterr().out
    dest = tmp_path / "back"
    r.fetch("out/sub", str(dest))
    assert (dest / "out" / "sub" / "result.txt").read_text() == "42", "путь сохраняется от /work"


def test_upload_resumes_from_the_offset_the_pod_reports(runner):
    r, _proc, work = runner
    first = r.call("PUT", "/upload?name=a.bin&offset=0", b"abc")
    assert first["size"] == 3
    with pytest.raises(urllib.error.HTTPError) as e:     # повтор куска после обрыва
        r.call("PUT", "/upload?name=a.bin&offset=0", b"abc")
    assert e.value.code == 409 and json.loads(e.value.read())["size"] == 3


def test_paths_outside_work_are_refused(runner):
    r, _proc, _work = runner
    with pytest.raises(urllib.error.HTTPError) as e:
        r.call("GET", "/download?path=../../etc", raw=True)
    assert e.value.code == 400
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"evil"
        ti = tarfile.TarInfo("../escape.txt")
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))
    r.call("PUT", "/upload?name=e.tar.gz&offset=0", buf.getvalue())
    with pytest.raises(urllib.error.HTTPError) as e:
        r.call("POST", "/extract?name=e.tar.gz", b"")
    assert e.value.code == 400


def test_pod_removes_itself_when_the_client_disappears(runner):
    """Нет задачи и обращений RUNNER_IDLE_SEC — под удаляет себя (сухой режим
    печатает команду вместо настоящего удаления)."""
    _r, proc, _work = runner
    time.sleep(5)
    proc.kill()
    out = proc.stdout.read()
    assert "удаляю под" in out and "remove pod pod123" in out


def test_running_job_keeps_the_pod_alive_past_the_idle_limit(runner):
    r, proc, _work = runner
    r.call("POST", "/run", json.dumps({"cmd": "sleep 6"}).encode())
    time.sleep(4.5)                                # дольше простоя, но задача идёт
    proc.kill()
    assert "удаляю под" not in proc.stdout.read()


class FakeApi:
    def __init__(self, fail_first=()):
        self.calls, self.pods, self.fail_first = [], set(), list(fail_first)

    def __call__(self, query, key, variables=None):
        self.calls.append((query.split("{")[0].strip(), variables))
        if "podFindAndDeployOnDemand" in query:
            gpu = variables["in"]["gpuTypeId"]
            if gpu in self.fail_first:
                raise RuntimeError("There are no longer any instances available")
            assert variables["in"]["volumeInGb"] == 0, "сетевой/постоянный диск стоит денег без пода"
            env = {e["key"]: e["value"] for e in variables["in"]["env"]}
            assert "RUNPOD_API_KEY" not in env, "ключ аккаунта не едет в под"
            self.pods.add("p1")
            return {"podFindAndDeployOnDemand": {"id": "p1", "costPerHr": 0.34,
                                                  "machine": {"gpuDisplayName": gpu}}}
        if "podTerminate" in query:
            self.pods.discard(variables["id"])
            return {"podTerminate": None}
        if "pod(input" in query:
            return {"pod": {"id": variables["id"]} if variables["id"] in self.pods else None}
        if "gpuTypes" in query:
            return {"gpuTypes": [
                {"id": "NVIDIA RTX A6000", "displayName": "RTX A6000", "memoryInGb": 48,
                 "communityCloud": True,
                 "lowestPrice": {"uninterruptablePrice": 0.33, "stockStatus": "Low"}}],
                "myself": {"clientBalance": 9.9, "spendLimit": 80, "currentSpendPerHr": 0}}
        raise AssertionError(query)


def test_next_gpu_is_tried_when_the_first_is_out_of_stock(monkeypatch):
    api = FakeApi(fail_first=("NVIDIA GeForce RTX 4090",))
    monkeypatch.setattr(rj, "gql", api)
    pod = rj.create_pod("k", rj.DEFAULT_GPUS, rj.DEFAULT_IMAGE, 80,
                        rj.runner_env("t", 10, 4, {}), "COMMUNITY")
    assert pod["machine"]["gpuDisplayName"] == "NVIDIA L40S"


def test_pod_is_removed_even_when_the_job_fails(monkeypatch):
    """Любой исход — удаление пода и проверка, что его больше нет."""
    api = FakeApi()
    monkeypatch.setattr(rj, "gql", api)
    monkeypatch.setattr(rj, "api_key", lambda: "k")
    monkeypatch.setattr(rj.Runner, "wait_ready", lambda self, *a, **k: True)

    def boom(self, cmd):
        raise RuntimeError("обрыв посреди задачи")
    monkeypatch.setattr(rj.Runner, "run", boom)
    with pytest.raises(RuntimeError):
        rj.main(["--cmd", "true"])
    assert not api.pods, "под остался жить после сбоя"
    assert any("podTerminate" in c[0] or "mutation" in c[0] for c in api.calls)


def test_plan_creates_nothing(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(rj, "gql", api)
    monkeypatch.setattr(rj, "api_key", lambda: "k")
    assert rj.main(["--plan"]) == 0
    assert not any("podFindAndDeployOnDemand" in str(c) for c in api.calls)


def test_only_named_secrets_go_to_the_pod(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("A=1\nB=2\nRUNPOD_API_KEY=zzz\n", encoding="utf-8")
    monkeypatch.setattr(rj, "REPO", str(tmp_path))
    assert rj.dotenv_subset(["A"]) == {"A": "1"}



def test_cheapest_fitting_community_gpu_first():
    types = [
        {"id": "a6000", "memoryInGb": 48, "communityCloud": True,
         "lowestPrice": {"uninterruptablePrice": 0.33, "stockStatus": "Low"}},
        {"id": "a5000", "memoryInGb": 24, "communityCloud": True,
         "lowestPrice": {"uninterruptablePrice": None, "stockStatus": None}},   # нет в наличии
        {"id": "t4", "memoryInGb": 16, "communityCloud": True,
         "lowestPrice": {"uninterruptablePrice": 0.1, "stockStatus": "High"}},  # мало памяти
        {"id": "secure_only", "memoryInGb": 48, "communityCloud": False,
         "lowestPrice": {"uninterruptablePrice": 0.2, "stockStatus": "High"}},
        {"id": "pro4500", "memoryInGb": 32, "communityCloud": True,
         "lowestPrice": {"uninterruptablePrice": 0.34, "stockStatus": "Low"}},
        {"id": "a40", "memoryInGb": 48, "communityCloud": True,
         "lowestPrice": {"uninterruptablePrice": 0.34, "stockStatus": "Low"}},
    ]
    types.append({"id": "NVIDIA GeForce RTX 5090", "displayName": "RTX 5090", "memoryInGb": 32,
                  "communityCloud": True,
                  "lowestPrice": {"uninterruptablePrice": 0.1, "stockStatus": "High"}})
    # при равной цене — больше памяти вперёд; Blackwell — только на образе с CUDA 12.8+
    old = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    assert rj.cheapest_gpus(types, image=old) == ["a6000", "a40", "pro4500"]
    assert rj.cheapest_gpus(types)[0] == "NVIDIA GeForce RTX 5090"
    assert rj.image_supports_blackwell("runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2204")
    assert not rj.image_supports_blackwell(old)


def test_wait_stops_early_when_the_container_runs_but_the_runner_is_silent(monkeypatch, capsys):
    """Деньги идут, а ждать нечего: контейнер работает дольше грации, а
    исполнитель не отвечает — стоп сразу, а не через полчаса."""
    r = rj.Runner("http://127.0.0.1:9", "t")
    monkeypatch.setattr(rj.time, "sleep", lambda s: None)
    calls = []

    def status():
        calls.append(1)
        return (None, "RUNNING (pulling)") if len(calls) < 2 else (400, "RUNNING (up)")
    t = [0.0]
    monkeypatch.setattr(rj.time, "time", lambda: (t.__setitem__(0, t[0] + 25), t[0])[1])
    assert r.wait_ready(10_000, status, run_grace_sec=300) is False
    out = capsys.readouterr().out
    assert "тянется образ" in out and "не отвечает" in out
