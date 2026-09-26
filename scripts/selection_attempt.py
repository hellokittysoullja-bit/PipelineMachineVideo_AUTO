#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Попытка отбора кадра: всё, что добытчик узнал и хочет изменить, — данные,
а не мутации, пока слот не решён.

ПОЧЕМУ. Состояние эпизода (анти-дубль по id и aHash, ритм крупностей, счёт
побед источников, лицензионный манифест, кэш выбранных файлов) раньше
менялось ИЗНУТРИ добытчиков, в момент, когда кадр ещё только кандидат:
видео, которое потом заменили фотографией, успевало занять свой хэш; кадр,
который потом поглотили, успевал лечь в кэш и на следующем прогоне
воскресал кэш-хитом без единого вердикта; вердикт отвергнутого видео
оставался висеть на слоте и убивал найденную фотографию. Откатывать это
задним числом (_slot_miss_snapshot/_slot_miss_restore) — значит помнить
каждое место, куда могла утечь запись; одна забытая запись уже стоила
эпизоду слота.

КАК. Добытчик работает внутри попытки (Attempt). Всё, что он хочет
изменить, он ЗАПИСЫВАЕТ в неё:
  * вердикт            — attempt.verdict(kind, record);
  * эффект             — attempt.effect(kind, *args): резерв в анти-дубле,
                         крупность в ритм, победа источника, строка
                         лицензионного манифеста;
  * файл для кэша      — attempt.stage(final_path) даёт путь в ПРИВАТНОМ
                         каталоге попытки, куда и пишется файл.
Решение о слоте принимает вызывающий код. Только commit() применяет эффекты
(в записанном порядке) и переносит файлы попытки на их места в кэше;
discard() удаляет каталог попытки и не применяет ничего. Отката нет, потому
что до решения менять было нечего.

Вызов добытчика ВНЕ открытой попытки (тесты, скрипты-утилиты) — это
«выбрать и сразу принять»: см. run_attempt() — та же попытка, тот же
commit, без второй ветки кода.

Модуль не знает ничего о конкретных эффектах и вердиктах пайплайна: коммит
получает применитель эффектов от вызывающего кода. Так правило «менять
состояние может только коммит» проверяется по одному модулю, а не по
каждому месту, где эффекты применяются.
"""
import contextvars
import os
import shutil
import threading

OPEN, COMMITTED, DISCARDED = "open", "committed", "discarded"

_CURRENT = contextvars.ContextVar("selection_attempt", default=None)
_SEQ = {}                 # номер слота -> сколько попыток у него уже было
_SEQ_LOCK = threading.Lock()


class AttemptStateError(RuntimeError):
    """Запись в закрытую попытку, повторное закрытие, чужой поток."""


def _next_seq(index):
    with _SEQ_LOCK:
        _SEQ[index] = _SEQ.get(index, 0) + 1
        return _SEQ[index]


class Attempt:
    """Одна попытка добыть медиа для слота.

    attempt_id — «<слот>-<вид>-<номер попытки этого слота>»: детерминирован
    и МЕСТЕН для слота. Сквозной счётчик процесса сдвигал бы id во всех
    последующих слотах, стоило одному слоту сделать на попытку больше, — и
    журналы двух прогонов расходились бы там, где ничего не менялось."""

    def __init__(self, index, kind, staging_root):
        self.index = index
        self.kind = kind
        self.attempt_id = f"{index}-{kind}-{_next_seq(index)}"
        self.state = OPEN
        self.verdicts = []    # [(kind, record)] в порядке записи
        self.effects = []     # [(kind, args)] в порядке записи
        self.media = None     # что добытчик вернул (путь в стейджинге или вне его)
        # Сведения о победителе для решения СЛОТА (например, оценка судьи —
        # по ней слот выбирает между фото и видео). Не вердикт (в отчёты не
        # уходит) и не эффект (состояния эпизода не меняет).
        self.notes = {}
        self._owner = threading.get_ident()
        # Абсолютный путь обязателен: is_staged()/final_path() сравнивают его с
        # os.path.abspath(файла). При относительной папке эпизода (запуск
        # «pipeline_smart.py videos/NN») сравнение было ложным всегда, коммит
        # возвращал путь во временном каталоге, уже пустом после переноса, и
        # рендер падал на «нет файла» (живой случай 26.09, эп.95).
        self._root = os.path.abspath(os.path.join(staging_root, self.attempt_id))
        self._dirs = {}       # final_dir -> staged_dir

    # -- запись ----------------------------------------------------------------

    def _check_open(self):
        if self.state != OPEN:
            raise AttemptStateError(
                f"попытка {self.attempt_id} уже {self.state}: запись после решения "
                f"означала бы мутацию состояния в обход коммита")

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise AttemptStateError(
                f"попытка {self.attempt_id} закрывается не из своего потока — "
                f"порядок коммитов перестал бы быть порядком слотов")

    def verdict(self, kind, record):
        self._check_open()
        self.verdicts.append((kind, dict(record)))

    def effect(self, kind, *args):
        self._check_open()
        self.effects.append((kind, args))

    def stage(self, final_path):
        """Путь, по которому попытка пишет файл, предназначенный для
        final_path. Каталог попытки лежит рядом с кэшем (та же файловая
        система), поэтому перенос при коммите атомарен."""
        self._check_open()
        final_dir = os.path.dirname(os.path.abspath(final_path))
        staged_dir = self._dirs.get(final_dir)
        if staged_dir is None:
            staged_dir = os.path.join(self._root, f"d{len(self._dirs)}")
            os.makedirs(staged_dir, exist_ok=True)
            self._dirs[final_dir] = staged_dir
        return os.path.join(staged_dir, os.path.basename(final_path))

    def is_staged(self, path):
        return bool(path) and os.path.abspath(path).startswith(self._root + os.sep)

    def final_path(self, path):
        """Куда попадёт (или попал) файл попытки после коммита. Путь вне
        стейджинга возвращается как есть: кэш-хит, локальный файл."""
        if not self.is_staged(path):
            return path
        staged_dir = os.path.dirname(os.path.abspath(path))
        for final_dir, sd in self._dirs.items():
            if sd == staged_dir:
                return os.path.join(final_dir, os.path.basename(path))
        raise AttemptStateError(f"файл {path} лежит в стейджинге, но не в каталоге попытки")

    # -- закрытие --------------------------------------------------------------

    def commit(self, apply_effect):
        """Применить эффекты в записанном порядке и перенести файлы на их
        места. Возвращает итоговый путь медиа (или None)."""
        self._check_open()
        self._check_owner()
        for kind, args in self.effects:
            apply_effect(kind, *args)
        for final_dir, staged_dir in sorted(self._dirs.items()):
            if not os.path.isdir(staged_dir):
                continue
            os.makedirs(final_dir, exist_ok=True)
            for name in sorted(os.listdir(staged_dir)):
                os.replace(os.path.join(staged_dir, name), os.path.join(final_dir, name))
        media = self.final_path(self.media) if self.media else None
        self.state = COMMITTED
        self._drop_root()
        return media

    def discard(self):
        """Ничего не применяется; файлы попытки удаляются."""
        self._check_open()
        self._check_owner()
        self.state = DISCARDED
        self._drop_root()

    def _drop_root(self):
        shutil.rmtree(self._root, ignore_errors=True)


def reset_attempt_ids():
    """Новый прогон — нумерация попыток с начала (два прогона в одном
    процессе обязаны дать одинаковые id)."""
    with _SEQ_LOCK:
        _SEQ.clear()


def current():
    return _CURRENT.get()


def record_note(key, value):
    """Сведение о победителе попытки для решения слота (см. Attempt.notes)."""
    att = _CURRENT.get()
    if att is None:
        raise AttemptStateError("сведение вне попытки: добытчик обязан работать внутри run_attempt()")
    att._check_open()
    att.notes[key] = value


def record_verdict(kind, record):
    att = _CURRENT.get()
    if att is None:
        raise AttemptStateError(
            "вердикт вне попытки: добытчик обязан работать внутри run_attempt()")
    att.verdict(kind, record)


def record_effect(kind, *args):
    """Эффект вне попытки — ошибка, а не «применить сразу»: второй путь
    изменения состояния в обход коммита — ровно то, что модуль убирает.
    Кому нужно изменить состояние эпизода вне отбора (например, вернуть в
    анти-дубль кадр уже стоящего в ролике клипа), тот открывает попытку и
    сразу её принимает — тем же commit()."""
    att = _CURRENT.get()
    if att is None:
        raise AttemptStateError(
            "эффект вне попытки: вызывающий код обязан открыть попытку (run_attempt)")
    att.effect(kind, *args)


def stage_path(final_path):
    att = _CURRENT.get()
    if att is None:
        raise AttemptStateError("стейджинг вне попытки")
    return att.stage(final_path)


class activate:
    """Сделать попытку текущей на время блока (контекстный менеджер)."""

    def __init__(self, attempt):
        self.attempt = attempt
        self._token = None

    def __enter__(self):
        if _CURRENT.get() is not None:
            raise AttemptStateError(
                "вложенная попытка: у слота одна текущая попытка за раз")
        self._token = _CURRENT.set(self.attempt)
        return self.attempt

    def __exit__(self, *exc):
        _CURRENT.reset(self._token)
        return False


def run_attempt(attempt, fn, *args, **kwargs):
    """Выполнить добытчика внутри попытки и запомнить, что он вернул.
    Исключение закрывает попытку отказом — полуготовые записи не выживают."""
    with activate(attempt):
        try:
            attempt.media = fn(*args, **kwargs)
        except BaseException:
            if attempt.state == OPEN:
                attempt.discard()
            raise
    return attempt.media


def sweep_orphans(staging_root):
    """Каталоги попыток, оставшиеся от прерванного процесса. Живых попыток
    при старте прогона нет, поэтому всё, что лежит в корне стейджинга, —
    мусор, и в кэш оно не попадёт никогда."""
    if os.path.isdir(staging_root):
        shutil.rmtree(staging_root, ignore_errors=True)
