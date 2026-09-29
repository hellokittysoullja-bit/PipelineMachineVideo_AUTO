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

    def rest(self, method, path, key, body=None):
        self.calls.append((f"REST {method} {path}", body))
        if method == "POST" and path == "/pods":
            gpu = (body.get("gpuTypeIds") or ["CPU"])[0]
            if gpu in self.fail_first:
                raise RuntimeError("There are no longer any instances available")
            assert body["volumeInGb"] == 0, "постоянный диск стоит денег без пода"
            assert "RUNPOD_API_KEY" not in body["env"], "ключ аккаунта не едет в под"
            assert isinstance(body["dockerStartCmd"], list), "команда старта — списком"
            self.pods.add("p1")
            return {"id": "p1", "costPerHr": 0.34}
        if method == "DELETE":
            self.pods.discard(path.rsplit("/", 1)[1])
            return {}
        if method == "GET":
            pid = path.rsplit("/", 1)[1]
            return {"id": pid, "desiredStatus": "RUNNING"} if pid in self.pods else None
        raise AssertionError(path)


def test_next_gpu_is_tried_when_the_first_is_out_of_stock(monkeypatch):
    api = FakeApi(fail_first=("NVIDIA GeForce RTX 4090",))
    monkeypatch.setattr(rj, "gql", api)
    monkeypatch.setattr(rj, "rest", api.rest)
    pod = rj.create_pod("k", rj.DEFAULT_GPUS, rj.DEFAULT_IMAGE, 80,
                        rj.runner_env("t", 10, 4, {}), "COMMUNITY")
    assert pod["gpuName"] == "NVIDIA L40S"


def test_pod_is_removed_even_when_the_job_fails(monkeypatch):
    """Любой исход — удаление пода и проверка, что его больше нет."""
    api = FakeApi()
    monkeypatch.setattr(rj, "gql", api)
    monkeypatch.setattr(rj, "rest", api.rest)
    monkeypatch.setattr(rj, "api_key", lambda: "k")
    monkeypatch.setattr(rj.Runner, "wait_ready", lambda self, *a, **k: True)

    def boom(self, cmd, deadline=None):
        raise RuntimeError("обрыв посреди задачи")
    monkeypatch.setattr(rj.Runner, "run", boom)
    with pytest.raises(RuntimeError):
        rj.main(["--cmd", "true", "--no-smoke"])
    assert not api.pods, "под остался жить после сбоя"
    assert any(c[0].startswith("REST DELETE") for c in api.calls)


def test_plan_creates_nothing(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(rj, "gql", api)
    monkeypatch.setattr(rj, "rest", api.rest)
    monkeypatch.setattr(rj, "api_key", lambda: "k")
    assert rj.main(["--plan"]) == 0
    assert not any("podFindAndDeployOnDemand" in str(c) or c[0] == "REST POST /pods" for c in api.calls)


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


def test_money_cap_limits_pod_life_on_both_sides(monkeypatch):
    """--max-usd: под на своей стороне получает потолок жизни по цене своей
    карты, а локальная сторона прерывает задачу по тому же потолку."""
    assert rj.budget_seconds(1.0, 0.5, 4) == 1800        # $0.5 при $1/ч — 30 мин
    assert rj.budget_seconds(0.25, 5, 2) == 7200         # потолок времени меньше
    api = FakeApi()
    monkeypatch.setattr(rj, "gql", api)
    monkeypatch.setattr(rj, "rest", api.rest)
    monkeypatch.setattr(rj, "api_key", lambda: "k")
    monkeypatch.setattr(rj.Runner, "wait_ready", lambda self, *a, **k: True)
    seen = {}

    def run(self, cmd, deadline=None):
        seen["deadline"] = deadline
        return 0
    monkeypatch.setattr(rj.Runner, "run", run)
    t0 = rj.time.time()
    assert rj.main(["--cmd", "true", "--max-usd", "0.34", "--no-smoke"]) == 0
    assert 3600 - 5 <= seen["deadline"] - t0 <= 3600 + 5   # $0.34 при $0.34/ч — час
    created = [c for c in api.calls if c[0] == "REST POST /pods"]
    env = created[0][1]["env"]
    assert abs(int(env["RUNNER_MAX_SEC"]) - 3600 / 0.33 * 0.34) < 5


def test_run_stops_at_the_money_deadline(monkeypatch):
    r = rj.Runner("http://x", "t")
    monkeypatch.setattr(rj.Runner, "call", lambda self, *a, **k: {"offset": 0, "text": "",
                                                                  "running": True, "exit": None})
    monkeypatch.setattr(rj.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit):
        r.run("sleep 999", deadline=rj.time.time() - 1)


def test_failed_smoke_rents_no_gpu(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(rj, "gql", api)
    monkeypatch.setattr(rj, "rest", api.rest)
    monkeypatch.setattr(rj, "api_key", lambda: "k")
    monkeypatch.setattr(rj, "smoke", lambda key, a: 1)
    with pytest.raises(SystemExit):
        rj.main(["--cmd", "true"])
    assert not any(c[0] == "REST POST /pods" for c in api.calls), "видеокарта арендована после провала"


def test_transient_proxy_errors_are_retried_and_runner_errors_are_not(runner, monkeypatch):
    """Пустой 404/502 от прокси при живом исполнителе — повтор; ответ самого
    исполнителя (JSON с причиной) — сразу наружу."""
    r, _proc, _work = runner
    real = urllib.request.urlopen
    fails = {"n": 2}

    def flaky(req, *a, **k):
        if fails["n"] > 0:
            fails["n"] -= 1
            raise urllib.error.HTTPError(req.full_url, 404 if fails["n"] else 502, "proxy",
                                         {}, io.BytesIO(b""))
        return real(req, *a, **k)
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(rj.time, "sleep", lambda s: None)
    assert "offset" in r.call("GET", "/log?offset=0")
    assert fails["n"] == 0
    with pytest.raises(urllib.error.HTTPError) as e:        # ответ исполнителя
        r.call("GET", "/nope")
    assert e.value.code == 404 and json.loads(e.value.read())["error"]


def test_repeated_run_and_extract_do_not_happen_twice(runner, tmp_path):
    r, _proc, work = runner
    body = json.dumps({"cmd": "echo once >> count.txt", "id": "job1"}).encode()
    r.call("POST", "/run", body)
    time.sleep(1)
    again = r.call("POST", "/run", body)                     # повтор после «сбоя связи»
    assert again.get("again") is True
    time.sleep(1)
    assert (work / "count.txt").read_text().count("once") == 1
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"x"
        ti = tarfile.TarInfo("f.txt")
        ti.size = 1
        tar.addfile(ti, io.BytesIO(data))
    r.call("PUT", "/upload?name=u.tar.gz&offset=0", buf.getvalue())
    assert r.call("POST", "/extract?name=u.tar.gz", b"")["ok"]
    assert r.call("POST", "/extract?name=u.tar.gz", b"").get("again") is True


def test_container_restart_mid_job_is_detected(monkeypatch):
    r = rj.Runner("http://x", "t")
    seq = iter([{"ok": True}, {"offset": 0, "text": "", "running": True, "exit": None, "uptime": 100},
                {"offset": 0, "text": "", "running": False, "exit": None, "uptime": 3}])
    monkeypatch.setattr(rj.Runner, "call", lambda self, *a, **k: next(seq))
    monkeypatch.setattr(rj.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="перезапустился"):
        r.run("long job")


def test_sweep_deletes_only_our_expired_pods(monkeypatch):
    """Контейнер не стартовал и эта сторона пропала — удалить под некому,
    кроме следующего запуска: он убирает НАШИ поды с истёкшим сроком и не
    трогает ни чужие, ни живые."""
    now = 1_000_000
    pods = [{"id": "old", "name": rj.pod_name(now - 5), "desiredStatus": "RUNNING"},
            {"id": "live", "name": rj.pod_name(now + 600), "desiredStatus": "RUNNING"},
            {"id": "foreign", "name": "my-jupyter", "desiredStatus": "RUNNING"},
            {"id": "gone", "name": rj.pod_name(now - 5), "desiredStatus": "TERMINATED"}]
    killed = []
    monkeypatch.setattr(rj, "rest", lambda m, p, k, b=None: pods if (m, p) == ("GET", "/pods") else None)
    monkeypatch.setattr(rj, "terminate", lambda key, pid: killed.append(pid) or True)
    assert rj.sweep_expired("k", now=now) == ["old"] and killed == ["old"]


def test_pod_name_carries_a_deadline_beyond_its_own_cap(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(rj, "rest", api.rest)
    t = rj.time.time()
    rj.create_pod("k", ["g"], "img", 10, [], "COMMUNITY", life_sec=1800)
    body = [c for c in api.calls if c[0] == "REST POST /pods"][0][1]
    deadline = int(body["name"][len(rj.POD_PREFIX):])
    assert t + 1800 < deadline <= t + 1800 + 600 + 5


def test_prepare_runs_while_data_is_still_uploading(runner, tmp_path, monkeypatch):
    """Подготовка стартует сразу после кода, данные грузятся параллельно,
    команда ждёт и подготовку, и отметку «всё загружено»."""
    r, _proc, work = runner
    code_dir, data_dir = tmp_path / "code", tmp_path / "data"
    code_dir.mkdir()
    data_dir.mkdir()
    (code_dir / "a.txt").write_text("code")
    (data_dir / "d.txt").write_text("data")
    monkeypatch.setattr(rj, "REPO", str(code_dir))
    uploads = []
    real_upload = rj.Runner.upload_dir

    def upload(self, path):
        if path == str(data_dir):
            time.sleep(1)
            prep_started = (work / "prep.txt").exists()
            uploads.append(("data", prep_started))
        real_upload(self, path)
    monkeypatch.setattr(rj.Runner, "upload_dir", upload)
    cmd = rj.overlapped_cmd("echo prep > prep.txt", "cat data/d.txt prep.txt > out.txt")
    r.upload_dir(str(code_dir))
    r.start(cmd)
    r.upload_dir(str(data_dir))
    r.call("POST", f"/touch?name={rj.UPLOADS_DONE}", b"")
    assert r.follow() == 0
    assert uploads == [("data", True)], "подготовка не шла параллельно с загрузкой"
    assert (work / "out.txt").read_text() == "dataprep\n"   # d.txt без перевода строки


def test_failed_prepare_stops_the_command(runner):
    r, _proc, work = runner
    r.start(rj.overlapped_cmd("exit 3", "touch ran.txt"))
    r.call("POST", f"/touch?name={rj.UPLOADS_DONE}", b"")
    assert r.follow() == 97 and not (work / "ran.txt").exists()


def test_hosts_with_old_drivers_are_never_rented(monkeypatch):
    """Фильтр хостов по CUDA: драйвер обязан тянуть CUDA образа."""
    assert rj.allowed_cuda(rj.DEFAULT_IMAGE) == ["12.8", "12.9", "13.0"]
    assert rj.allowed_cuda("runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")[0] == "12.4"
    assert rj.allowed_cuda("python:3.11-slim") is None
    api = FakeApi()
    monkeypatch.setattr(rj, "rest", api.rest)
    rj.create_pod("k", ["NVIDIA RTX A6000"], rj.DEFAULT_IMAGE, 80, rj.runner_env("t", 10, 4, {}),
                  "COMMUNITY")
    rj.create_pod("k", [], rj.SMOKE_IMAGE, 20, rj.runner_env("t", 10, 4, {}), "SECURE", cpu=True)
    bodies = [c[1] for c in api.calls if c[0] == "REST POST /pods"]
    assert bodies[0]["allowedCudaVersions"] == ["12.8", "12.9", "13.0"]
    assert "allowedCudaVersions" not in bodies[1]


def test_semicolon_chain_no_longer_hides_a_failed_step(runner):
    """Живой прогон 29.09: «калибровка; замер» — калибровка упала, а код 0."""
    r, _proc, _work = runner
    assert r.run("false; echo дошло") != 0


def test_missing_result_is_reported_and_the_rest_is_still_fetched(runner, tmp_path, monkeypatch):
    r, _proc, work = runner
    monkeypatch.setattr(rj, "terminate", lambda key, pid: True)
    monkeypatch.setattr(rj.Runner, "wait_ready", lambda self, *a, **k: True)
    base = r.base
    monkeypatch.setattr(rj, "Runner", lambda url, token: rj.__dict__["_RealRunner"](base, token))
    monkeypatch.setattr(rj, "_RealRunner", type(r), raising=False)
    dest = tmp_path / "back"
    code = rj.drive("k", {"id": "p", "costPerHr": 0.3}, TOKEN, 600, [],
                    "mkdir -p res && echo 1 > res/a.txt", ["nope", "res"], str(dest))
    assert code == 98, "пропуск результата не должен выглядеть как успех"
    assert (dest / "res" / "a.txt").exists(), "остальные результаты забраны"


def test_preflight_output_is_not_reprinted_by_the_next_job(runner, capsys):
    r, _proc, _work = runner
    assert r.run("echo ПРОВЕРКА") == 0
    capsys.readouterr()
    assert r.run("echo ЗАДАЧА") == 0
    out = capsys.readouterr().out
    assert "ЗАДАЧА" in out and "ПРОВЕРКА" not in out


def _two_gpu_api(monkeypatch):
    api = FakeApi()
    real = api.rest

    def rest(method, path, key, body=None):
        got = real(method, path, key, body)
        if method == "POST":
            got = dict(got, costPerHr=0.5)
        return got
    monkeypatch.setattr(rj, "gql", api)
    monkeypatch.setattr(rj, "rest", rest)
    monkeypatch.setattr(rj, "api_key", lambda: "k")
    monkeypatch.setattr(rj, "sweep_expired", lambda key, now=None: [])
    monkeypatch.setattr(rj.Runner, "wait_ready", lambda self, *a, **k: True)
    return api


def test_broken_gpu_host_is_replaced_before_any_upload(monkeypatch):
    """Хост, где видеокарта не работает: под удаляется ДО загрузки данных,
    берётся другой; неисправная карта уходит в конец очереди."""
    api = _two_gpu_api(monkeypatch)
    seen = {"uploads": [], "cmds": []}

    def run(self, cmd, deadline=None):
        seen["cmds"].append(cmd)
        if cmd == rj.GPU_PREFLIGHT:
            gpus = [c[1]["gpuTypeIds"][0] for c in api.calls if c[0] == "REST POST /pods"]
            return 1 if gpus[-1] == "bad" else 0
        return 0
    monkeypatch.setattr(rj.Runner, "run", run)
    monkeypatch.setattr(rj.Runner, "upload_dir", lambda self, p: seen["uploads"].append(
        [c[1]["gpuTypeIds"][0] for c in api.calls if c[0] == "REST POST /pods"][-1]))
    assert rj.main(["--cmd", "job", "--upload", ".", "--gpu", "bad", "--gpu", "good",
                    "--no-smoke"]) == 0
    created = [c[1]["gpuTypeIds"][0] for c in api.calls if c[0] == "REST POST /pods"]
    assert created == ["bad", "good"]
    assert seen["uploads"] == ["good"], "данные не грузились на неисправный хост"
    assert not api.pods and sum(c[0].startswith("REST DELETE") for c in api.calls) == 2


def test_host_retries_share_one_money_cap(monkeypatch):
    """--max-usd — на все попытки вместе: каждая следующая живёт на остаток."""
    _two_gpu_api(monkeypatch)
    caps = []

    def drive(key, pod, token, cap_sec, *a, **k):
        caps.append(cap_sec)
        pod["spent_usd"] = 0.25
        raise rj.BadHost("карта не работает")
    monkeypatch.setattr(rj, "drive", drive)
    with pytest.raises(SystemExit) as e:
        rj.main(["--cmd", "job", "--gpu", "x", "--max-usd", "0.6", "--no-smoke"])
    assert "не заработала" in str(e.value)
    # $0.6 → $0.35 → $0.10 при $0.5/ч
    assert [round(c) for c in caps] == [4320, 2520, 720]
    caps.clear()
    with pytest.raises(SystemExit) as e:
        rj.main(["--cmd", "job", "--gpu", "x", "--max-usd", "0.3", "--no-smoke"])
    assert "потолок" in str(e.value) and len(caps) == 1   # на вторую попытку денег нет
