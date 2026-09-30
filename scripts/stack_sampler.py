#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Семплер стеков процесса (PROFILE_SAMPLER=1): куда уходит время слота.

Без прав ptrace и без внешних пакетов (py-spy в контейнере часто не
работает): поток раз в SAMPLE_SEC берёт sys._current_frames() всех потоков и
считает, сколько раз каждый (поток, три верхних кадра) встретился. Файл
<video_dir>/media_plan/profile_stacks.txt переписывается каждые FLUSH_SEC —
его можно читать во время прогона (`--watch`). Слот ждёт сети, модели или
блокировки: по верхним кадрам это видно (urlopen/recv, ml_device.run/_lock,
torch, cv2).

Побочного эффекта на выбор нет: только чтение стеков; при выключенном флаге
модуль не импортируется."""
import collections
import os
import sys
import threading
import time

SAMPLE_SEC = 0.1
FLUSH_SEC = 20.0
DEPTH = 3
CPU_SEC = 2.0
GROUP_TOP = 25


def _label(frame):
    parts = []
    f, n = frame, 0
    while f is not None and n < DEPTH:
        co = f.f_code
        parts.append(f"{os.path.basename(co.co_filename)}:{co.co_name}:{f.f_lineno}")
        f, n = f.f_back, n + 1
    return " <- ".join(parts)


def _group(name):
    """Имя потока без номера: prefetch_3 и prefetch_0 — одна группа."""
    import re
    return re.sub(r"[_\-]?\d+$", "", name) or name


def _func_label(frame):
    """Три верхних кадра БЕЗ номеров строк: главный поток проходит по десяткам
    разных строк одной функции, и с номерами его время рассыпалось на сотни
    записей и не попадало в первые 400 (прогон A40 30.09 — главного потока в
    файле не было вовсе)."""
    parts = []
    f, n = frame, 0
    while f is not None and n < DEPTH:
        co = f.f_code
        parts.append(f"{os.path.basename(co.co_filename)}:{co.co_name}")
        f, n = f.f_back, n + 1
    return " <- ".join(parts)


def _cpu_times():
    """(занято, всего) тиков процессора по /proc/stat; None вне Linux."""
    try:
        with open("/proc/stat", encoding="utf-8") as fh:
            v = [int(x) for x in fh.readline().split()[1:9]]
    except (OSError, ValueError):
        return None
    idle = v[3] + v[4]
    return sum(v) - idle, sum(v), v[4]


def _count_procs(names=("ffmpeg", "python", "python3")):
    out = dict.fromkeys(names, 0)
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/comm", encoding="utf-8") as fh:
                    c = fh.read().strip()
            except OSError:
                continue
            if c in out:
                out[c] += 1
    except OSError:
        pass
    return out


def start_cpu_log(out_path):
    """cpu_usage.csv: занятость процессора, ожидание диска, средняя очередь,
    число ffmpeg/python — раз в CPU_SEC. Рядом с gpu_usage.csv: без этого
    «на карте 2%» не отвечает, что делает процессор (на A40 30.09 клипы шли
    по 10-28 минут, а метрик процессора не было вовсе)."""
    stop = threading.Event()

    def loop():
        prev = _cpu_times()
        try:
            fh = open(out_path, "w", encoding="utf-8")
        except OSError:
            return
        with fh:
            fh.write("timestamp,cpu_busy_pct,iowait_pct,load1,ffmpeg,python,cores\n")
            while not stop.wait(CPU_SEC):
                cur = _cpu_times()
                if cur is None or prev is None:
                    break
                dt = max(cur[1] - prev[1], 1)
                busy = 100.0 * (cur[0] - prev[0]) / dt
                iow = 100.0 * (cur[2] - prev[2]) / dt
                prev = cur
                try:
                    load1 = os.getloadavg()[0]
                except OSError:
                    load1 = -1
                pc = _count_procs()
                fh.write(f"{time.strftime('%H:%M:%S')},{busy:.1f},{iow:.1f},{load1:.2f},"
                         f"{pc['ffmpeg']},{pc['python'] + pc['python3']},{os.cpu_count()}\n")
                fh.flush()

    threading.Thread(target=loop, name="cpu_sampler", daemon=True).start()
    return stop.set


def start(out_path):
    """Запустить семплер (демон-поток); вернуть функцию остановки."""
    counts = collections.Counter()
    fcounts = collections.Counter()   # (группа потоков, функции без строк)
    stop = threading.Event()
    started = time.time()
    me = threading.get_ident()

    def flush():
        total = sum(counts.values()) or 1
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(f"# семплов {total}, шаг {SAMPLE_SEC} с, {time.time() - started:.0f} с работы\n")
            for (tname, stack), n in counts.most_common(400):
                fh.write(f"{n:6d} {100.0 * n / total:5.1f}%  [{tname}] {stack}\n")
            # По группам потоков: сколько времени группа вообще жива и на чём.
            groups = collections.defaultdict(collections.Counter)
            for (g, st), n in fcounts.items():
                groups[g][st] += n
            fh.write("\n# ПО ГРУППАМ ПОТОКОВ (функции без номеров строк)\n")
            for g, c in sorted(groups.items(), key=lambda kv: -sum(kv[1].values())):
                fh.write(f"## {g}: {sum(c.values())} семплов\n")
                for st, n in c.most_common(GROUP_TOP):
                    fh.write(f"{n:8d}  {st}\n")
        os.replace(tmp, out_path)

    def loop():
        last = time.time()
        while not stop.wait(SAMPLE_SEC):
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                if ident == me:
                    continue
                nm = names.get(ident, str(ident))
                counts[(nm, _label(frame))] += 1
                fcounts[(_group(nm), _func_label(frame))] += 1
            if time.time() - last > FLUSH_SEC:
                last = time.time()
                try:
                    flush()
                except OSError:
                    pass
        try:
            flush()
        except OSError:
            pass

    threading.Thread(target=loop, name="stack_sampler", daemon=True).start()
    return stop.set
