#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Упреждающий поиск: пока слот i ждёт судью, готовятся кучи слотов впереди.

ЗАЧЕМ. Замер эпизода 98 (70 слотов, 15 минут видео, 5 часов отбора): около
37% времени — поиск по источникам и скачивание превью, около 43% — ответы
судьи, и всё это шло строго по очереди: слот i+1 не начинал искать, пока
судья думал над слотом i. Бесплатный слот (после 25-го) — это на 90% поиск.

ЧТО ДЕЛАЕТСЯ ЗАРАНЕЕ, и почему выбор кадра от этого не меняется. Только то,
что не зависит от предыдущих слотов и лежит в ДИСКОВЫХ кэшах, которые слот и
так прочтёт: выдача источников по запросам слота, превью и эмбеддинги
каскада. Анти-дубль, ритм крупностей, бюджет судьи, голоса мира — всё, что
делает выбор слота зависимым от соседей, — остаётся в самом слоте, по
очереди. Кэши на процесс и счётчики, влияющие на решения, упреждение не
трогает (pipeline_smart.prefetching()). Платных вызовов (судья, отсев по
подписи, планировщик) здесь нет вовсе.

ОСТАТОК, названный прямо. Упреждение делает живые запросы раньше, чем их
сделал бы слот. Остаток часовой квоты Pexels (заголовок ответа) слот видит
поэтому чуть меньшим — на решение это влияет, только когда квота почти
исчерпана, а там исход и без упреждения зависит от того, в какую минуту часа
идёт прогон. Счётчики запросов в media_plan/source_contribution.json
включают упреждающие запросы. Запросы слота, до которого прогон не дойдёт
(кэш клипа), упреждение пропускает по готовому файлу клипа.
"""
import concurrent.futures
import threading

# Сколько слотов вперёд и сколько слотов одновременно. Глубина больше —
# больше спрошенного впустую, если прогон оборвётся; одновременно больше —
# выше нагрузка на источники с общим лимитом (Мет, Commons), которые всё
# равно пропускают запросы по очереди через свой регулятор.
DEPTH = 3
WORKERS = 2
# Дальний проход (весь эпизод с первой секунды): сколько слотов одновременно.
# Внутри слота источники и так опрашиваются параллельно, у каждого свой
# регулятор темпа (Мет, Commons, Pixabay) — лишняя ширина здесь упёрлась бы
# в них же.
FAR_WORKERS = 4


def _pool(n, prefix):
    return concurrent.futures.ThreadPoolExecutor(max_workers=n, thread_name_prefix=prefix)


class SlotPrefetcher:
    """Два прохода упреждения.

    Ближний — job(j) для слотов i+1..i+depth, пока идёт слот i (как было).
    Дальний — far_job(j) для ВСЕХ слотов эпизода, поставленный при первом
    advance(): ближайшие первыми (очередь по порядку), своим пулом, чтобы
    ближний не ждал за семьюдесятью дальними. job=None — только дальний
    (ближний делает упреждающий отбор, slot_speculation).

    Ошибки заданий глотаются: упреждение только ускоряет, слот всё сделает
    сам."""

    def __init__(self, n_slots, job, depth=DEPTH, workers=WORKERS, far_job=None,
                 far_workers=FAR_WORKERS):
        self.n_slots = n_slots
        self.job = job
        self.far_job = far_job
        self.depth = depth
        self._ex = _pool(workers, "prefetch") if job is not None else None
        self._far_ex = _pool(far_workers, "prefetch_far") if far_job is not None else None
        self._far_started = False
        self._scheduled = set()
        self._lock = threading.Lock()
        self.stats = {"scheduled": 0, "done": 0, "failed": 0,
                      "far_scheduled": 0, "far_done": 0, "far_failed": 0}
        self._closed = False

    def advance(self, i):
        """Слот i начинается: поставить прогрев слотов i+1..i+depth, а при
        первом вызове — дальний проход по всем слотам после i."""
        if self._closed:
            return
        if self._far_ex is not None and not self._far_started:
            self._far_started = True
            for j in range(i + 1, self.n_slots):
                with self._lock:
                    self.stats["far_scheduled"] += 1
                self._far_ex.submit(self._run, j, self.far_job, "far_")
        if self._ex is None:
            return
        for j in range(i + 1, min(self.n_slots, i + 1 + self.depth)):
            with self._lock:
                if j in self._scheduled:
                    continue
                self._scheduled.add(j)
                self.stats["scheduled"] += 1
            self._ex.submit(self._run, j, self.job, "")

    def _run(self, j, job, prefix):
        if self._closed:
            return
        try:
            job(j)
            with self._lock:
                self.stats[prefix + "done"] += 1
        except Exception:  # noqa: BLE001 — ускорение, а не решение
            with self._lock:
                self.stats[prefix + "failed"] += 1

    def close(self):
        """Дождаться начатого и не брать нового (отмена ещё не начатых)."""
        self._closed = True
        for ex in (self._ex, self._far_ex):
            if ex is not None:
                ex.shutdown(wait=True, cancel_futures=True)
