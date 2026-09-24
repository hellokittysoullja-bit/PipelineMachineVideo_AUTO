#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Профиль канала (channel_profile.json в корне репозитория) — ОДИН читатель.

Раньше файл читали три модуля по-своему: pipeline_smart (один раз при
импорте), shot_types и museum_sources (заново на каждый вызов). Разные
читатели одного файла — это разные ответы на один вопрос, как только один
из них научится чему-то, чего не умеет другой (кэш, обработка битого
файла). Здесь чтение одно: кэш по времени изменения файла, битый или
отсутствующий файл — пустой профиль (fail-open: у клона под новую нишу
профиля может не быть вовсе, см. CLAUDE.md ЧАСТЬ 24).

Мир и словарь канала (якоря эпохи, чужие культуры, типы кадра, отделы
музея, ловушки вето, запасные запросы) живут в профиле, а не в коде:
дефолты в коде пустые."""
import json
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "channel_profile.json")
_CACHE = {"key": None, "data": {}}


def load(path=None):
    path = path or PATH
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return {}
    if _CACHE["key"] != key:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Битый {path}, пропускаю: {e}")
            data = {}
        _CACHE["key"], _CACHE["data"] = key, data if isinstance(data, dict) else {}
    return _CACHE["data"]

