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


# --- неопределённый ответ создания пода -------------------------------------

class _CreateApi:
    """POST /pods отвечает ошибкой; под при этом создан или нет."""

    def __init__(self, error, created):
        self.error, self.created, self.pods, self.posts = error, created, [], 0

    def rest(self, method, path, key, body=None):
        if method == "POST":
            self.posts += 1
            if self.created:
                self.pods.append({"id": "p9", "name": body["name"], "desiredStatus": "RUNNING",
                                  "costPerHr": 0.49})
            raise self.error
        if method == "GET" and path == "/pods":
            return self.pods
        raise AssertionError((method, path))


def _create(monkeypatch, api, gpus=("A",)):
    monkeypatch.setattr(rj, "rest", api.rest)
    return rj.create_pod("k", list(gpus), "img", 10, [], "SECURE", life_sec=600)


def test_ambiguous_error_with_a_created_pod_adopts_it(monkeypatch, capsys):
    api = _CreateApi(RuntimeError("HTTP 500: Something went wrong. Please try again later"), created=True)
    pod = _create(monkeypatch, api)
    assert pod["id"] == "p9" and pod["gpuTypeId"] == "A"
    assert api.posts == 1, "второй под не создаётся"


def test_timeout_with_a_created_pod_adopts_it(monkeypatch, capsys):
    api = _CreateApi(TimeoutError("timed out"), created=True)
    assert _create(monkeypatch, api)["id"] == "p9"


def test_ambiguous_error_without_a_pod_is_no_stock(monkeypatch, capsys):
    api = _CreateApi(RuntimeError("HTTP 500: Something went wrong"), created=False)
    with pytest.raises(rj.NoStock):
        _create(monkeypatch, api)


def test_plain_no_stock_does_not_look_for_a_pod(monkeypatch, capsys):
    api = _CreateApi(RuntimeError("HTTP 500: There are no instances currently available"), created=True)
    with pytest.raises(rj.NoStock):
        _create(monkeypatch, api)
    assert api.posts == 1


@pytest.mark.parametrize("code", [401, 402, 403])
def test_hard_refusal_stops_at_once_instead_of_polling(monkeypatch, code):
    api = _CreateApi(RuntimeError(f"HTTP {code}: nope"), created=False)
    with pytest.raises(SystemExit) as e:
        _create(monkeypatch, api)
    assert not isinstance(e.value, rj.NoStock) and "отказал" in str(e.value)


def test_env_args_parse_and_refuse_secrets():
    assert rj.parse_env_args(["CASCADE_MODEL=wemm9b", "A=b=c"]) == {"CASCADE_MODEL": "wemm9b", "A": "b=c"}
    assert rj.parse_env_args([]) == {}
    for bad in ("NOEQUALS", "1BAD=x", "PEXELS_API_KEY=abc", "MY_TOKEN=1"):
        with pytest.raises(SystemExit):
            rj.parse_env_args([bad])


def test_fetch_weights_skips_wemm_unless_the_run_will_call_it(monkeypatch):
    import fetch_weights as fw
    monkeypatch.setenv("CASCADE_MODEL", "wemm9b")
    monkeypatch.setenv("SHOT_JUDGE", "0")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    assert not fw.wemm_needed() and len(fw.models(wemm=fw.wemm_needed())) == 2
    monkeypatch.setenv("SHOT_JUDGE", "1")
    assert fw.wemm_needed() and len(fw.models(wemm=fw.wemm_needed())) == 3
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "")
    assert not fw.wemm_needed()
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("CASCADE_MODEL", "")
    assert not fw.wemm_needed()


def test_pod_prepare_checks_the_ffmpeg_checksum(tmp_path):
    """Логика сверки, вырезанная из pod_prepare.sh, на подменённых файлах."""
    src = open(os.path.join(REPO, "scripts", "pod_prepare.sh"), encoding="utf-8").read()
    assert 'checksums.sha256' in src and 'sha256sum' in src and "контрольная сумма не совпала" in src
    # Поведение: тот же фрагмент в bash на заведомо верной и неверной сумме.
    start = src.index("if curl -fsL --retry 2 --max-time 60")
    end = src.index("if ! tar -xf")
    frag = src[start:end]
    frag = frag.replace('curl -fsL --retry 2 --max-time 60 "${u%/*}/checksums.sha256" -o $D/sums.txt 2>/dev/null',
                        'cp "$SUMS" $D/sums.txt')
    (tmp_path / "f.tar.xz").write_bytes(b"payload")
    import hashlib
    good = hashlib.sha256(b"payload").hexdigest()

    def run(sums_text):
        sums = tmp_path / "s.txt"
        sums.write_text(sums_text)
        script = f'say() {{ echo "$@"; }}\nD="{tmp_path}"\nu="https://h/x/latest/f.tar.xz"\nSUMS="{sums}"\n' \
                 f'for _ in 1; do\n{frag}\necho PASSED\ndone\n'
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout
    assert "совпала" in run(f"{good}  f.tar.xz\n") and "PASSED" in run(f"{good}  f.tar.xz\n")
    bad = run("0" * 64 + "  f.tar.xz\n")
    assert "не совпала" in bad and "PASSED" not in bad
    assert "принят без проверки" in run(f"{good}  other.tar.xz\n")


def test_pod_prepare_probes_nvenc_for_real(tmp_path):
    """nvenc_ok: без nvidia-smi — годна (проверять нечем); с ним — только если
    пробное кодирование прошло. Наличие hevc_nvenc в списке кодеков не считается."""
    src = open(os.path.join(REPO, "scripts", "pod_prepare.sh"), encoding="utf-8").read()
    start = src.index("  nvenc_ok() {")
    end = src.index("  nvenc_report() {")
    fn = src[start:end]
    assert "-c:v hevc_nvenc" in fn and "-f null" in fn
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ok_ff = tmp_path / "ff_ok"
    bad_ff = tmp_path / "ff_bad"
    ok_ff.write_text("#!/bin/sh\nexit 0\n")
    bad_ff.write_text("#!/bin/sh\necho 'Driver does not support the required nvenc API version' >&2\nexit 1\n")
    for f in (ok_ff, bad_ff):
        f.chmod(0o755)

    def run(ff, with_smi):
        if with_smi:
            smi = bin_dir / "nvidia-smi"
            smi.write_text("#!/bin/sh\nexit 0\n")
            smi.chmod(0o755)
        else:
            (bin_dir / "nvidia-smi").unlink(missing_ok=True)
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}
        return subprocess.run(["bash", "-c", fn + f'\nnvenc_ok "{ff}"'], env=env,
                              capture_output=True, text=True).returncode
    assert run(bad_ff, with_smi=False) == 0, "нет видеокарты — проверять нечем"
    assert run(ok_ff, with_smi=True) == 0
    assert run(bad_ff, with_smi=True) != 0


def test_pod_prepare_never_falls_back_to_apt_ffmpeg():
    """apt даёт 4.4 без переходов: ролик собрался бы не тем, что у CPU-ветки."""
    src = open(os.path.join(REPO, "scripts", "pod_prepare.sh"), encoding="utf-8").read()
    assert "apt-get install -y -qq ffmpeg" not in src
    assert "подходящего ffmpeg (5+, drawtext, переходы) не получено" in src
