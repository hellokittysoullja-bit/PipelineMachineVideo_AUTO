#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сколько процессорных ядер и памяти прогону РЕАЛЬНО отдано.

В контейнере (Runpod, Docker, Kubernetes) os.cpu_count() и /proc/meminfo
показывают ХОЗЯИНА, а не выданную долю. Прогон A40 30.09: у пода 9 vCPU и
50 ГБ, а процесс видел 96 ядер — пул рендера поднялся на 95 процессов
(«остальное — процессор (95)»), у каждого ffmpeg фильтр-граф на 96 потоков,
параллакс в главном процессе — x264 на все 96. Клипы шли по 10-28 минут
(стоковое видео — медиана 674 с против 36 с на машине, где ядра видны честно),
и главный процесс отбора делил те же девять ядер с этой толпой.

Источники, от меньшего к большему доверию:
  * os.cpu_count() — ядра хозяина;
  * sched_getaffinity — то, что разрешено процессу (taskset, cpuset);
  * cgroup v2 cpu.max / cgroup v1 cfs_quota_us — квота по времени, именно её
    выдаёт Docker --cpus и облака;
  * RUNPOD_CPU_COUNT — число vCPU пода, если платформа его сообщила;
  * CPU_BUDGET в окружении — явное число владельца (побеждает всё).
Берётся МИНИМУМ: ни один источник не может отдать больше, чем разрешил другой.
Вне контейнера (нет квоты, нет переменных) результат равен os.cpu_count() —
поведение прежнее байт-в-байт.

Ничего не делает при импорте, кроме чтения нескольких файлов при вызове."""
import math
import os

CGROUP_ROOT = "/sys/fs/cgroup"


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _quota_cores(root=CGROUP_ROOT):
    """Квота cgroup в ядрах (дробная) или None, если её нет."""
    v2 = _read(os.path.join(root, "cpu.max"))
    if v2:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                q, p = float(parts[0]), float(parts[1])
                if q > 0 and p > 0:
                    return q / p
            except ValueError:
                pass
    q1 = _read(os.path.join(root, "cpu", "cpu.cfs_quota_us"))
    p1 = _read(os.path.join(root, "cpu", "cpu.cfs_period_us"))
    if q1 and p1:
        try:
            q, p = float(q1), float(p1)
            if q > 0 and p > 0:      # -1 — без квоты
                return q / p
        except ValueError:
            pass
    return None


def _positive_int(text):
    try:
        v = int(float(str(text).strip()))
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def cores(root=CGROUP_ROOT, environ=None):
    env = os.environ if environ is None else environ
    forced = _positive_int(env.get("CPU_BUDGET"))
    if forced:
        return forced
    seen = [os.cpu_count() or 1]
    try:
        seen.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    q = _quota_cores(root)
    if q is not None:
        seen.append(max(1, int(math.ceil(q - 1e-9))))
    rp = _positive_int(env.get("RUNPOD_CPU_COUNT"))
    if rp:
        seen.append(rp)
    return max(1, min(seen))


def restricted(root=CGROUP_ROOT, environ=None):
    """True, если выдано меньше, чем видит os.cpu_count()."""
    return cores(root, environ) < (os.cpu_count() or 1)


def mem_available_mb(root=CGROUP_ROOT, meminfo="/proc/meminfo"):
    """Свободная память в МБ: меньшее из MemAvailable хозяина и остатка лимита
    cgroup (лимит минус занято). None — определить нельзя."""
    host = None
    txt = _read(meminfo)
    if txt:
        for line in txt.splitlines():
            if line.startswith("MemAvailable:"):
                try:
                    host = int(line.split()[1]) / 1024.0
                except (ValueError, IndexError):
                    pass
                break
    cg = None
    for lim_p, use_p in ((os.path.join(root, "memory.max"), os.path.join(root, "memory.current")),
                         (os.path.join(root, "memory", "memory.limit_in_bytes"),
                          os.path.join(root, "memory", "memory.usage_in_bytes"))):
        lim, use = _read(lim_p), _read(use_p)
        if lim and lim != "max" and use:
            try:
                lim_b, use_b = int(lim), int(use)
            except ValueError:
                continue
            if 0 < lim_b < 1 << 60:      # v1 без лимита отдаёт огромное число
                cg = max(0.0, (lim_b - use_b) / 1048576.0)
                break
    vals = [v for v in (host, cg) if v is not None]
    return min(vals) if vals else None


def thread_env_defaults():
    """Переменные пулов потоков (OpenMP, BLAS) для процессов, которые их
    читают при старте: без них каждая библиотека берёт число ядер хозяина.
    Пусто, если ядра видны честно — ничего не меняется."""
    if not restricted():
        return {}
    n = str(cores())
    return {k: n for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def apply_thread_env():
    """Выставить thread_env_defaults, не перезаписывая то, что задал владелец.
    Вызывать до импорта torch/numpy-тяжёлых модулей. Возвращает выставленное."""
    done = {}
    for k, v in thread_env_defaults().items():
        if k not in os.environ:
            os.environ[k] = v
            done[k] = v
    return done


def ffmpeg_thread_args(n):
    """Потоки ffmpeg воркера. `-threads` (как было) ограничивает только
    энкодер; фильтры (scale, gblur, unsharp, blend) берут по потоку на каждое
    ВИДИМОЕ ядро. В контейнере, где видно ядра хозяина, а выдано меньше
    (A40 30.09: видно 96, выдано 9), фильтры ограничиваются тем же числом —
    результат кадра от этого не зависит (нарезка по строкам, проверено
    побайтно). Ядра видны честно — команда прежняя байт-в-байт: на обычной
    машине фильтры не лишаются свободных ядер, когда клипов в очереди мало."""
    n = str(max(1, int(n)))
    if not restricted():
        return ["-threads", n]
    return ["-threads", n, "-filter_threads", n, "-filter_complex_threads", n]
