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


def _label(frame):
    parts = []
    f, n = frame, 0
    while f is not None and n < DEPTH:
        co = f.f_code
        parts.append(f"{os.path.basename(co.co_filename)}:{co.co_name}:{f.f_lineno}")
        f, n = f.f_back, n + 1
    return " <- ".join(parts)


def start(out_path):
    """Запустить семплер (демон-поток); вернуть функцию остановки."""
    counts = collections.Counter()
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
        os.replace(tmp, out_path)

    def loop():
        last = time.time()
        while not stop.wait(SAMPLE_SEC):
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                if ident == me:
                    continue
                counts[(names.get(ident, str(ident)), _label(frame))] += 1
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
