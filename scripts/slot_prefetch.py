#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подготовка следующих слотов в фоне: поиск по источникам и эмбеддинги
каскада, пока текущий слот ждёт судью.

ЗАЧЕМ. Замер холодного прогона эпизода 94 (27.09, профиль каждой стадии):
слот платной зоны — 220-310 с, из них поиск по источникам 47-60 с и каскад
(превью всей кучи + модель гейта) 50-110 с на каждый вид медиа; судья,
которого слот на самом деле ждёт, — 30-45 с. Поиск и каскад слота не
зависят от того, чем кончились предыдущие слоты: куча собирается из запросов
фразы, а эмбеддинг превью — функция самой картинки. Значит их можно сделать
заранее, пока процессор простаивает в ожидании ответа шлюза.

ГЛАВНОЕ СВОЙСТВО: решений здесь нет. Подготовка только заполняет те же
кэши, которые отбор прочитал бы сам (дисковые кэши поиска, кэш эмбеддингов
каскада); отбор слота идёт прежним кодом и по прежним правилам. Поэтому
кадр тот же, что без подготовки, — меньше только ожидание. Что для этого
держится (и проверяется тестами):
  * эмбеддинг превью не зависит от того, КТО и В КАКОЙ пачке его посчитал:
    пачки всегда ровно по CASCADE_BATCH, неполная добивается копиями
    (pipeline_smart.gate_embed_batch);
  * подготовка не тратит квоты, которые решают состав кучи: Pexels и
    Unsplash она не спрашивает (берёт их выдачу, только если слот раньше
    уже спросил тот же запрос). Исключение — собственные запросы слота
    платной зоны к Pexels: их отбор задаёт при любом остатке квоты, и
    подготовка делает тот же вызов раньше (см. PREFETCH_QUOTA_SOURCES в
    pipeline_smart);
  * одинаковый запрос к источнику, идущий одновременно из отбора и из
    подготовки, делается ОДИН раз (singleflight): отбор ждёт ответа,
    который уже в пути, а не повторяет его;
  * сбой подготовки ничего не портит: ошибка источника не кэшируется, и
    отбор спросит сам, как спросил бы без подготовки;
  * процессор — отбору: пачка каскада в фоне начинается, только если отбор
    не считал моделью последние FOREGROUND_QUIET_SEC.

Модуль — только расписание (очереди, потоки, стоп); сами работы приходят
функциями из pipeline_smart.
"""
import contextlib
import contextvars
import threading
import time

# Слот, для которого сейчас идёт подготовка (None — работа самого отбора).
# Потоки подготовки ставят его на время работы; харнесс эквивалентности
# (selection_freeze.py) метит им сетевые обращения подготовки.
TARGET = contextvars.ContextVar("slot_prefetch_target", default=None)


class Prefetcher:
    """Два конвейера: поиск (сеть) и каскад (процессор). Слот сначала
    ищется, потом его куча оценивается; слоты идут строго по порядку.

    search(j)          — подготовить поиск слота j;
    cascade(j) -> bool — подготовить эмбеддинги кучи слота j (False — слоту
                         каскад не нужен: зона без судьи);
    search_ahead(i)    — на сколько слотов вперёд от текущего i искать;
    cascade_ahead(i)   — на сколько вперёд оценивать."""

    def __init__(self, n_slots, search, cascade, search_ahead, cascade_ahead, name="prefetch",
                 search_workers=2):
        self.n = int(n_slots)
        self._search = search
        self._cascade = cascade
        self._search_ahead = search_ahead
        self._cascade_ahead = cascade_ahead
        self._cond = threading.Condition()
        self._current = -1
        self._next_search = 0
        self._next_cascade = 0
        self._searched = set()
        self._stop = False
        self.stats = {"searched": 0, "cascaded": 0, "errors": 0}
        # Поиск слотов — в search_workers потоков: слот, упёршийся в паузу
        # одного источника (Commons из-за лимита отвечает минутами), не
        # держит поиск следующего; сам источник всё равно спрашивается по
        # своему регулятору, общему на процесс.
        self._threads = [threading.Thread(target=self._run_search, name=f"{name}-search{k}", daemon=True)
                         for k in range(max(1, int(search_workers)))]
        self._threads.append(threading.Thread(target=self._run_cascade, name=f"{name}-cascade",
                                              daemon=True))
        for t in self._threads:
            t.start()

    # ---------------------------------------------------------------- управление

    def advance(self, i):
        """Отбор начал слот i: подготовка может идти до i + окно."""
        with self._cond:
            self._current = max(self._current, int(i))
            self._cond.notify_all()

    def close(self, timeout=5.0):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout)

    @property
    def current(self):
        """Слот, который отбор сейчас решает (-1 — ещё ни одного)."""
        return self._current

    @property
    def stopped(self):
        return self._stop

    # ---------------------------------------------------------------- потоки

    def _wait_turn(self, attr, ahead_fn, ready=None):
        """Номер следующего слота этого конвейера, когда он попал в окно
        (и, если задано, готов по ready); None — стоп или слоты кончились."""
        with self._cond:
            while True:
                if self._stop:
                    return None
                j = getattr(self, attr)
                if j >= self.n:
                    return None
                cur = self._current
                if cur >= 0 and j < cur:
                    # Отбор уже прошёл этот слот — готовить его поздно.
                    setattr(self, attr, cur)
                    continue
                window = ahead_fn(max(cur, 0)) if cur >= 0 else 0
                if cur >= 0 and j <= cur + window and (ready is None or ready(j)):
                    setattr(self, attr, j + 1)
                    return j
                self._cond.wait(0.5)

    def _run_search(self):
        while True:
            j = self._wait_turn("_next_search", self._search_ahead)
            if j is None:
                return
            token = TARGET.set(j)
            ok = True
            try:
                self._search(j)
            except Exception:  # noqa: BLE001 — подготовка не имеет права уронить отбор
                ok = False
            finally:
                TARGET.reset(token)
            with self._cond:
                self.stats["searched" if ok else "errors"] += 1
                self._searched.add(j)
                self._cond.notify_all()

    def _run_cascade(self):
        while True:
            j = self._wait_turn("_next_cascade", self._cascade_ahead,
                                ready=lambda k: k in self._searched)
            if j is None:
                return
            token = TARGET.set(j)
            try:
                if self._cascade(j):
                    self.stats["cascaded"] += 1
            except Exception:  # noqa: BLE001
                self.stats["errors"] += 1
            finally:
                TARGET.reset(token)


class Eager:
    """Работы «сразу, как пришли данные» — один поток, по порядку поступления.

    Каскаду слота нужна куча целиком, а источники отвечают вразнобой: музеи и
    стоки — за секунды, Commons из-за лимита — минутами (замер 27.09, эп.94,
    слот 0: поиск 134 с, из них 120 с — ожидание Commons, и всё это время
    процессор стоял). Эмбеддинг превью от кучи не зависит — только от самой
    картинки, — поэтому кандидаты источника, который уже ответил, оцениваются
    сразу, пока ждут остальные. Решений нет и здесь: заполняется тот же кэш
    эмбеддингов, что прочитает каскад.

    submit(target, fn, *args) — target: слот, для которого работа (метит
    сетевые обращения, см. TARGET). Раньше — работа слота с меньшим номером:
    он ближе к отбору (текущий слот — первым), внутри слота — по порядку
    поступления."""

    def __init__(self, name="eager", horizon=None):
        import itertools
        import queue
        self._q = queue.PriorityQueue()
        self._seq = itertools.count()
        self._stop = False
        # horizon() — слот, который отбор сейчас решает: работа для слотов
        # ДО него уже никому не нужна (каскад слота посчитан отбором сам) и
        # не делается — иначе при очереди «по номеру слота» она шла бы
        # первой, впереди работы текущего слота.
        self._horizon = horizon
        self.stats = {"jobs": 0, "errors": 0}
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def stopped(self):
        return self._stop

    def submit(self, target, fn, *args):
        if not self._stop:
            self._q.put((target if target is not None else -1, next(self._seq), fn, args))

    def _run(self):
        while True:
            target, _seq, fn, args = self._q.get()
            if fn is None:
                return
            if self._stop:
                continue
            if self._horizon is not None and target >= 0 and target < self._horizon():
                self.stats["skipped"] = self.stats.get("skipped", 0) + 1
                continue
            token = TARGET.set(target)
            try:
                fn(*args)
                self.stats["jobs"] += 1
            except Exception:  # noqa: BLE001 — подготовка не имеет права уронить отбор
                self.stats["errors"] += 1
            finally:
                TARGET.reset(token)

    def close(self, timeout=5.0):
        self._stop = True
        self._q.put((-2, -1, None, ()))      # раньше любой работы: поток выходит сразу
        self._thread.join(timeout)


class SingleFlight:
    """Один запрос на ключ в полёте: второй вызывающий с тем же ключом ждёт
    ответа первого, а не повторяет запрос. Ошибка первого второму НЕ
    передаётся — он спрашивает сам (сбой подготовки не должен стать сбоем
    отбора)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._inflight = {}
        self.shared = 0

    def call(self, key, fn):
        with self._lock:
            ev = self._inflight.get(key)
            if ev is None:
                ev = self._inflight[key] = [threading.Event(), None, False]
                owner = True
            else:
                owner = False
        if not owner:
            ev[0].wait()
            if ev[2]:
                with self._lock:
                    self.shared += 1
                return ev[1]
            return fn()
        try:
            result = fn()
            ev[1], ev[2] = result, True
            return result
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            ev[0].set()


class ForegroundClock:
    """Одна модель на процесс — один вычислитель в каждый момент, отбор —
    первым.

    Замер 27.09 (модель гейта на 4 ядрах, две пачки по 16 кадров): по
    очереди 4.5 с, одновременно из двух потоков — 61 с. Потоки torch двух
    вызовов делят ядра и мешают друг другу в 14 раз; числа при этом те же
    до бита. Поэтому вызов модели — под общим замком: отбор ждёт не
    дольше одной фоновой пачки (~2 с), а фоновая пачка начинается, только
    если отбор молчал quiet_sec и никто из отбора не встал в очередь.

    model()           — контекст вызова модели отбором (в фоновом потоке —
                        пустой: фон уже держит замок через background_turn);
    background_turn() — контекст фоновой пачки: True — можно считать,
                        False — подготовку остановили."""

    def __init__(self, quiet_sec):
        self.quiet_sec = float(quiet_sec)
        self._last = 0.0
        self._local = threading.local()
        self._lock = threading.RLock()
        self._count = threading.Lock()
        self._waiting = 0

    def is_background(self):
        return bool(getattr(self._local, "bg", False))

    @contextlib.contextmanager
    def model(self):
        if self.is_background():
            yield
            return
        with self._count:
            self._waiting += 1
        self._lock.acquire()
        with self._count:
            self._waiting -= 1
        self._last = time.monotonic()
        try:
            yield
        finally:
            self._last = time.monotonic()
            self._lock.release()

    def wait_quiet(self, stopped=lambda: False, poll=0.2):
        while not stopped():
            idle = time.monotonic() - self._last
            if idle >= self.quiet_sec and not self._waiting:
                return True
            time.sleep(max(0.01, min(poll, self.quiet_sec - idle)))
        return False

    @contextlib.contextmanager
    def background_turn(self, stopped=lambda: False):
        while True:
            if not self.wait_quiet(stopped):
                yield False
                return
            self._lock.acquire()
            if not self._waiting and time.monotonic() - self._last >= self.quiet_sec:
                break
            self._lock.release()
        self._local.bg = True
        try:
            yield True
        finally:
            self._local.bg = False
            self._lock.release()
