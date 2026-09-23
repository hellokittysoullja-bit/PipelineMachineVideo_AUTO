#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запись и воспроизведение решений отбора, зависящих от ЧАСОВ.

Сеть харнесс записывает на границе urlopen. Но часть решений отбора
зависит не от ответа сервера, а от того, СКОЛЬКО ВРЕМЕНИ прошло: после 403
Мет уходит в остывание на 60 секунд (museum_sources.met_is_cooling_down),
и запросы, попавшие в это окно, не делаются вовсе. Воспроизведение идёт
быстрее записи (сети нет), поэтому в окно попадают ДРУГИЕ запросы — и пул
кандидатов меняется при том же коде. Замер: запись эпизода 93 кодом этапа 0,
воспроизведение тем же кодом — один запрос к Мет пропал, другой появился,
met_requests 161 -> 160. Эквивалентность без этой записи недоказуема.

Решение записывается с привязкой к запросу, внутри которого оно
принималось (адрес + номер обращения к нему + место: вход в _met_get или
вход в остывание), а не к порядку во времени: карточки Мет тянутся
параллельно, и глобальный порядок решений между потоками не определён.
При воспроизведении ответ берётся из записи; решения, которого в записи
нет, — названное расхождение со слотом, в котором оно случилось (и живой
ответ часов, чтобы прогон дошёл до конца), а не тихая подстановка.
"""
import json
import os
import threading

RECORD, REPLAY = "record", "replay"
OUTSIDE = "<вне запроса>"


class MetCooldownRecorder:
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self._local = threading.local()
        self._lock = threading.Lock()
        self._seq = {}
        self.recorded = {}
        self.divergences = []
        self.decisions = 0
        self._real = {}
        # Слот, в котором принималось решение (харнесс ставит его так же, как
        # NetRecorder.tagger). Без привязки расхождение по часам нельзя
        # отнести к слоту с названной причиной, и законно новый запрос в
        # изменившемся слоте выглядел бы как необъяснённый сбой.
        self.tagger = None
        if mode == REPLAY:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        self.recorded[rec["key"]] = rec["value"]
        elif os.path.exists(path):
            os.remove(path)

    def _key(self, site):
        ctx = getattr(self._local, "url", None) or OUTSIDE
        with self._lock:
            k = (site, ctx)
            n = self._seq.get(k, 0)
            self._seq[k] = n + 1
        return f"{site}|{n}|{ctx}"

    def _cooling(self):
        key = self._key(getattr(self._local, "site", "outside"))
        with self._lock:
            self.decisions += 1
        if self.mode == RECORD:
            value = bool(self._real["cool"]())
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
            return value
        if key in self.recorded:
            return self.recorded[key]
        slot = self.tagger() if self.tagger is not None else None
        with self._lock:
            self.divergences.append({"key": key, "slot": slot})
        return bool(self._real["cool"]())

    def _within(self, site, fn, url=None):
        prev = (getattr(self._local, "url", None), getattr(self._local, "site", "outside"))
        if url is not None:
            self._local.url = url
        self._local.site = site
        try:
            return fn()
        finally:
            self._local.url, self._local.site = prev

    def install(self, ms):
        """ms — модуль museum_sources. Функции подменяются атрибутами модуля:
        внутри он зовёт их по глобальному имени, то есть через эти же
        атрибуты."""
        self._real = {"cool": ms.met_is_cooling_down, "get": ms._met_get,
                      "enter": ms._met_enter_cooldown}
        real = self._real
        ms.met_is_cooling_down = self._cooling
        ms._met_get = lambda url: self._within("get", lambda: real["get"](url), url=url)
        ms._met_enter_cooldown = lambda: self._within("enter", real["enter"])
        return self

    def summary(self):
        return {"mode": self.mode, "decisions": self.decisions,
                "divergences": list(self.divergences)}
