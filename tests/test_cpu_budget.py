# -*- coding: utf-8 -*-
"""Ядра и память — выданные прогону, а не хозяина контейнера (A40 30.09: 9 vCPU,
процесс видел 96 и поднял 95 воркеров рендера)."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import cpu_budget as cb  # noqa: E402
import pipeline_smart as ps  # noqa: E402


@pytest.fixture
def host96(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 96)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(96)))


def _cg(tmp_path, cpu_max=None, mem_max=None, mem_cur=None):
    if cpu_max is not None:
        (tmp_path / "cpu.max").write_text(cpu_max)
    if mem_max is not None:
        (tmp_path / "memory.max").write_text(mem_max)
        (tmp_path / "memory.current").write_text(mem_cur)
    return str(tmp_path)


def test_cgroup_v2_quota_wins_over_host_cores(tmp_path, host96):
    assert cb.cores(_cg(tmp_path, "900000 100000"), {}) == 9
    assert cb.restricted(str(tmp_path), {})


def test_fractional_quota_rounds_up(tmp_path, host96):
    assert cb.cores(_cg(tmp_path, "250000 100000"), {}) == 3


def test_no_quota_means_host_cores(tmp_path, host96):
    assert cb.cores(_cg(tmp_path, "max 100000"), {}) == 96
    assert not cb.restricted(str(tmp_path), {})


def test_cgroup_v1_quota(tmp_path, host96):
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("400000")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert cb.cores(str(tmp_path), {}) == 4
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1")
    assert cb.cores(str(tmp_path), {}) == 96


def test_platform_variable_and_owner_override(tmp_path, host96):
    assert cb.cores(str(tmp_path), {"RUNPOD_CPU_COUNT": "9"}) == 9
    # минимум: платформа говорит больше, чем квота — верна квота
    assert cb.cores(_cg(tmp_path, "600000 100000"), {"RUNPOD_CPU_COUNT": "9"}) == 6
    # явное число владельца побеждает всё
    assert cb.cores(_cg(tmp_path, "600000 100000"), {"CPU_BUDGET": "12"}) == 12
    clean = tmp_path / "clean"
    clean.mkdir()
    assert cb.cores(str(clean), {"RUNPOD_CPU_COUNT": "junk", "CPU_BUDGET": "0"}) == 96


def test_affinity_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "cpu_count", lambda: 96)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2})
    assert cb.cores(str(tmp_path), {}) == 3


def test_memory_takes_the_smaller_of_host_and_cgroup(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal: 999\nMemAvailable:   67108864 kB\n")       # 65536 МБ
    root = _cg(tmp_path, mem_max=str(50 * 1024 ** 3), mem_cur=str(10 * 1024 ** 3))   # 40960 МБ
    assert cb.mem_available_mb(root, str(mi)) == pytest.approx(40960)
    root2 = _cg(tmp_path, mem_max="max", mem_cur="1")
    assert cb.mem_available_mb(root2, str(mi)) == pytest.approx(65536)
    assert cb.mem_available_mb(str(tmp_path / "нет"), str(tmp_path / "нет")) is None


def test_thread_env_only_when_restricted(monkeypatch, tmp_path):
    monkeypatch.setattr(cb, "restricted", lambda *a, **k: False)
    assert cb.thread_env_defaults() == {}
    monkeypatch.setattr(cb, "restricted", lambda *a, **k: True)
    monkeypatch.setattr(cb, "cores", lambda *a, **k: 9)
    env = cb.thread_env_defaults()
    assert env["OMP_NUM_THREADS"] == "9" and env["OPENBLAS_NUM_THREADS"] == "9"
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    done = cb.apply_thread_env()
    assert os.environ["OMP_NUM_THREADS"] == "2" and "OMP_NUM_THREADS" not in done
    assert done["OPENBLAS_NUM_THREADS"] == "9"


def test_ffmpeg_thread_args_limit_filters_only_in_a_restricted_container(monkeypatch):
    monkeypatch.setattr(cb, "restricted", lambda *a, **k: True)
    a = cb.ffmpeg_thread_args(1)
    assert a == ["-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1"]
    assert cb.ffmpeg_thread_args(0)[1] == "1"
    monkeypatch.setattr(cb, "restricted", lambda *a, **k: False)
    assert cb.ffmpeg_thread_args(1) == ["-threads", "1"], "ядра видны честно — команда прежняя"


def test_render_workers_follow_granted_cores_not_host(monkeypatch):
    monkeypatch.setattr(ps.cpu_budget, "cores", lambda *a, **k: 9)
    monkeypatch.setattr(ps.cpu_budget, "mem_available_mb", lambda *a, **k: 50 * 1024.0)
    assert ps._default_render_workers() == 8, "было 95 на поде с 9 vCPU"
    monkeypatch.setattr(ps.cpu_budget, "mem_available_mb", lambda *a, **k: 2000.0)
    assert ps._default_render_workers() == 2, "память тоже ограничивает"
    monkeypatch.setattr(ps.cpu_budget, "mem_available_mb", lambda *a, **k: None)
    assert ps._default_render_workers() == 8


def test_env_worker_count_cannot_exceed_granted_cores(monkeypatch):
    monkeypatch.setattr(ps.cpu_budget, "cores", lambda *a, **k: 9)
    monkeypatch.setenv("RENDER_WORKERS", "95")
    assert ps._render_workers_from_env() == 9


def test_parallax_and_final_chunks_use_granted_cores(monkeypatch):
    monkeypatch.setattr(ps.cpu_budget, "cores", lambda *a, **k: 9)
    monkeypatch.delenv("PARALLAX_WORKERS", raising=False)
    assert ps.parallax_workers() == 2
    monkeypatch.setattr(ps.cpu_budget, "mem_available_mb", lambda *a, **k: 500000.0)
    monkeypatch.delenv("FINAL_CHUNK_WORKERS", raising=False)
    assert ps.final_chunk_workers(50) <= max(1, 9 // ps.FINAL_CHUNK_CORES_PER_WORKER)


def test_parallax_ffmpeg_threads_only_when_restricted(monkeypatch):
    monkeypatch.delenv("PARALLAX_FFMPEG_THREADS", raising=False)
    monkeypatch.setattr(ps.cpu_budget, "restricted", lambda *a, **k: False)
    assert ps.parallax_thread_args() == [], "ядра видны честно — как было"
    monkeypatch.setattr(ps.cpu_budget, "restricted", lambda *a, **k: True)
    monkeypatch.setattr(ps.cpu_budget, "cores", lambda *a, **k: 9)
    args = ps.parallax_thread_args()
    assert args[:2] == ["-threads", "2"] and "-filter_threads" in args
    monkeypatch.setenv("PARALLAX_FFMPEG_THREADS", "3")
    assert ps.parallax_thread_args()[1] == "3"


def test_worker_ffmpeg_commands_limit_filter_threads():
    """Четыре места команд воркера ставят потоки кодека И фильтров, а не один
    -threads: три процессорных и кодер пути видеокарты (аудит 30.09 — x264 на
    карте без NVENC видел все ядра хозяина)."""
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    assert src.count("cpu_budget.ffmpeg_thread_args(ffmpeg_threads)") == 4
    assert '["-threads", str(ffmpeg_threads)]' not in src


def test_launcher_passes_pod_vcpu_to_the_command():
    import runpod_job as rj
    assert rj.with_cpu_budget("cd x && python a.py", {"vcpuCount": 9}) == "export CPU_BUDGET=9; cd x && python a.py"
    assert rj.with_cpu_budget("python a.py", {"vcpuCount": "9"}).startswith("export CPU_BUDGET=9;")
    assert rj.with_cpu_budget("python a.py", {}) == "python a.py"
    assert rj.with_cpu_budget("python a.py", {"vcpuCount": 0}) == "python a.py"
    own = "CPU_BUDGET=4 python a.py"
    assert rj.with_cpu_budget(own, {"vcpuCount": 9}) == own, "явный выбор владельца не перезаписывается"
