#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Упреждающий отбор: слоты впереди решаются заранее тем же кодом, что и в
настоящем цикле, а настоящий цикл потом берёт готовое из кэшей.

ЗАЧЕМ. Замер эпизода 98 (70 слотов): отбор шёл 298 минут, и слоты шли
строго по очереди — слот i+1 не начинал ни искать, ни спрашивать судью,
пока слот i не решён. Внутри слота всё, что можно, уже параллельно (сетки
судьи, финалисты, источники); длинная цепочка — между слотами. Упреждающий
поиск (slot_prefetch) снимал только поиск и каскад; вызовы судьи — 43%
времени — оставались строго последовательными.

КАК, и почему выбор кадра от этого не меняется.
  * Задание «упредить слот j» получает СНИМОК истории эпизода (анти-дубль,
    ритм крупностей, недавние виды — pipeline_smart.SlotContext.snapshot) и
    проходит ту же лестницу, что настоящий цикл (build_slot_selection,
    run_slot_ladder). Все его попытки выбрасываются: ни одного эффекта на
    эпизод.
  * Всё, что меняет решения эпизода, у упреждения своё или не трогается:
    голоса мира, прогноз бюджета судьи, серия сбоев и квота Pexels, потолок
    Unsplash, журналы и отчёты, нумерация попыток (pipeline_smart.
    speculating()/background()/_log_list).
  * Ответы моделей кладутся в хранилище шлюза и СПИСЫВАЮТСЯ, когда
    настоящий цикл задаёт тот же вопрос, — с той же проверкой потолка, что у
    живого вызова (llm_gateway.speculation). Дисковый кэш ответов упреждение
    не пишет, иначе ответ ушёл бы мимо учёта.
  * Настоящий цикл перед слотом j ждёт окончания его упреждения: иначе оба
    спросили бы одно и то же одновременно и заплатили бы дважды.
  * Настоящий цикл решает слот сам, своим кодом, со своей историей. Где
    история упреждения совпала с настоящей (почти всегда: разойтись она может
    только на победителях слотов между постановкой задания и настоящим
    решением), он берёт всё из кэшей; где нет — доделывает живьём.

ЧТО СТОИТ ДЕНЕГ, названо. Вопрос, который упреждение задало, а настоящий
цикл не задал (история разошлась), оплачен впустую; сумма — в сводке шлюза
судьи (speculative_wasted). Генерацию кадра упреждение не делает (картинки
хранилище шлюза не держит), арбитра Gemini — тоже (дневная квота).
"""
import builtins
import concurrent.futures
import threading
import time

# Сколько слотов вперёд и сколько одновременно. Глубина больше числа потоков
# ставит задания в очередь; сами потоки упираются в лимиты источников и в
# модели на процессоре/видеокарте. Переопределяются из .env
# (SLOT_SPECULATE_DEPTH, SLOT_SPECULATE_WORKERS).
DEPTH = 8
WORKERS = 4


class SlotSpeculator:
    """job(j, ctx_snapshot, extra) — упреждение слота j; его исключения
    глотаются: упреждение только ускоряет, слот всё сделает сам."""

    def __init__(self, n_slots, job, depth=DEPTH, workers=WORKERS):
        self.n_slots = n_slots
        self.job = job
        self.depth = max(1, int(depth))
        self._ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers)),
                                                          thread_name_prefix="speculate")
        self._futures = {}
        self._lock = threading.Lock()
        self.stats = {"scheduled": 0, "done": 0, "failed": 0, "waited_sec": 0.0}
        self._closed = False

    def advance(self, i, snapshot):
        """Слот i начинается: поставить упреждение слотов i+1..i+depth.
        snapshot() зовётся в потоке настоящего цикла — снимок истории на
        этот момент."""
        if self._closed:
            return
        todo = [j for j in range(i + 1, min(self.n_slots, i + 1 + self.depth))
                if j not in self._futures]
        if not todo:
            return
        snap = snapshot()
        for j in todo:
            with self._lock:
                self.stats["scheduled"] += 1
            self._futures[j] = self._ex.submit(self._run, j, snap)

    def _run(self, j, snap):
        try:
            self.job(j, snap)
            with self._lock:
                self.stats["done"] += 1
        except Exception:  # noqa: BLE001 — ускорение, а не решение
            with self._lock:
                self.stats["failed"] += 1

    def wait(self, i):
        """Перед слотом i: дождаться его упреждения (если оно было)."""
        f = self._futures.get(i)
        if f is None:
            return
        t0 = time.perf_counter()
        try:
            f.result()
        except Exception:  # noqa: BLE001
            pass
        self.stats["waited_sec"] += time.perf_counter() - t0

    def close(self):
        """Дождаться начатого и не брать нового (отмена ещё не начатых)."""
        self._closed = True
        self._ex.shutdown(wait=True, cancel_futures=True)


class quiet_output:
    """Печать упреждения не выводится: строки слота j печатает настоящий
    цикл, когда решает его. is_quiet() — признак потока упреждения."""

    def __init__(self, is_quiet):
        self.is_quiet = is_quiet
        self._orig = None

    def __enter__(self):
        self._orig = builtins.print
        orig, is_quiet = self._orig, self.is_quiet

        def _print(*args, **kwargs):
            if is_quiet():
                return
            orig(*args, **kwargs)
        builtins.print = _print
        return self

    def __exit__(self, *exc):
        builtins.print = self._orig
        return False
