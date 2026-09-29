#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запуск на поде без слепоты и без путаницы путей (живые прогоны 29.09:
13 минут вслепую из-за `| tail`, два запуска упали на `requirements.txt` из-за
того, что папка вне репозитория легла в /work/<имя>/)."""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import runpod_job as rj  # noqa: E402


def _args(**kw):
    a = argparse.Namespace(cmd="python x.py", prepare=None, allow_truncate=False, workdir="", upload=[])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.mark.parametrize("cmd", ["python x.py | tail -n 150", "python x.py 2>&1 |tail -5",
                                 "python x.py | head -3"])
def test_truncating_cmd_is_refused(cmd):
    with pytest.raises(SystemExit):
        rj.check_cmd_visible(_args(cmd=cmd))
    with pytest.raises(SystemExit):
        rj.check_cmd_visible(_args(prepare=cmd))


def test_truncation_can_be_allowed_and_plain_cmds_pass():
    rj.check_cmd_visible(_args(cmd="python x.py | tail -3", allow_truncate=True))
    rj.check_cmd_visible(_args(cmd="python -u x.py 2>&1 | grep --line-buffered ERROR"))


def test_workdir_prefixes_cmd_and_prepare(capsys):
    a = _args(cmd="python x.py", prepare="bash scripts/pod_prepare.sh", workdir="stage98",
              upload=["/tmp/stage98"])
    rj.apply_workdir(a)
    assert a.cmd == "cd stage98 && python x.py"
    assert a.prepare == "cd stage98 && bash scripts/pod_prepare.sh"
    assert "/work/stage98/" in capsys.readouterr().out, "не сказано, куда легла загрузка"


def test_no_workdir_leaves_commands_alone():
    a = _args(cmd="python x.py", prepare="p")
    rj.apply_workdir(a)
    assert (a.cmd, a.prepare) == ("python x.py", "p")


def test_log_lines_get_local_time_stamps():
    r = rj.Runner("https://x", "t")
    out = r._stamp("a\nb")
    assert out.count(":") >= 4 and out.splitlines()[0].endswith(" a")
    # строка, оборванная на середине, не получает вторую метку
    r2 = rj.Runner("https://x", "t")
    first = r2._stamp("par")
    second = r2._stamp("tial\n")
    assert second == "tial\n" and first.endswith("par")


def test_watch_prints_only_new_lines(monkeypatch, capsys):
    import io
    import tarfile
    r = rj.Runner("https://x", "t", watch=["media_plan/stage_timings.jsonl"])

    def fake_call(method, path, **kw):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            data = "\n".join(fake_call.lines).encode()
            info = tarfile.TarInfo("stage_timings.jsonl")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return buf.getvalue()
    monkeypatch.setattr(r, "call", fake_call)
    fake_call.lines = ["one", "two"]
    r._watch_once()
    assert capsys.readouterr().out.count("[stage_timings.jsonl]") == 2
    r._watch_at = 0.0
    fake_call.lines = ["one", "two", "three"]
    r._watch_once()
    out = capsys.readouterr().out
    assert "three" in out and "one" not in out


def test_pod_prepare_runs_three_parts_in_parallel():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "pod_prepare.sh"), encoding="utf-8").read()
    assert src.count(") &\n") == 3 and "wait $p" in src
    assert "fetch_weights.py" in src and "requirements-gpu.txt" in src
