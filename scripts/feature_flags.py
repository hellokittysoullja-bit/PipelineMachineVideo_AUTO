#!/usr/bin/env python3
"""Единый реестр флагов режимов пайплайна (источник истины для дефолтов).

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. До него дефолт каждого режима был вписан ЛИТЕРАЛОМ
в КАЖДОЙ точке чтения — девять мест в пяти файлах (pipeline_smart.py x6,
shot_director.py x3, look_reference.py, speech_planner.py,
stock_fetch_multisource.py). Две РЕАЛЬНЫЕ, найденные вживую (03.09) поломки
ровно этого устройства — не гипотетические риски:

1. **VLM_ARBITER_MODE.** CLAUDE.md (ЧАСТЬ 14) объявляет «дефолт `on`,
   реализован и живьём проверен 29 августа», а все три точки чтения в коде
   подставляли `"off"` — и `config.example.env` тоже. То есть на любом
   рендере, где переменная не выставлена руками, VLM-арбитра не было вообще,
   при том что документация канала считала его работающим. Документация и код
   разошлись молча: ни один тест не сверял их между собой.

2. **DEFLICKER_ENABLED.** CLAUDE.md документирует этот флаг как «быстрый
   откат без редеплоя», а код читал переменную с ДРУГИМ именем —
   `DEFLICKER`. Человек, выставивший `DEFLICKER_ENABLED=0` ровно по
   документации, деффликер не отключал: флаг отката не работал по своему же
   документированному имени.

Поэтому: дефолт объявляется здесь ОДИН раз, точки чтения спрашивают реестр,
а `tests/test_feature_flags.py` сверяет реестр с тем, что обещает CLAUDE.md,
и с `config.example.env`. Расхождение теперь падает тестом, а не тихо живёт
в проде месяцами.

СЕМАНТИКА (сохранена байт-в-байт с тем, что было до реестра — модуль не
меняет поведение ни одного флага, только убирает дублирование дефолта):

* Значение читается из окружения при КАЖДОМ вызове, не кэшируется на импорте
  — иначе monkeypatch в тестах и любой код, выставляющий режим после импорта
  модуля, видели бы застывший снимок (та же причина, по которой
  shot_director.py уже проверяет `os.environ` внутри функций, а не
  модуль-константой, см. комментарий у direct_query()).
* Нормализация — `.strip().lower()`, как во всех прежних точках чтения.
* Значение ВНЕ списка допустимых → громкое предупреждение и откат на "off"
  (не на дефолт!). Именно так вело себя всё до реестра: сравнение вида
  `... == "on"` на мусорном значении давало «выключено», а look_reference.py
  делал этот откат явно. Опечатка не должна ВКЛЮЧАТЬ дорогой слой.
* Булевы флаги (`_ENABLED`/`_GATE`) — «включено, если значение не "0"», ровно
  как `os.environ.get(..., "1") != "0"` раньше. Список синонимов вроде
  "off"/"false" сюда СОЗНАТЕЛЬНО не добавлен: сегодня `RENDER_STRICT_GATE=off`
  означает «включено», и тихо перевернуть это значило бы ослабить жёсткий
  гейт сборки у того, кто уже так написал.
"""
import os
import sys


class Flag:
    """Описание одного флага. `aliases` — исторические имена переменной,
    которые продолжают работать (см. DEFLICKER ниже): читаются ПОСЛЕ
    основного имени, только если основное не выставлено."""

    def __init__(self, name, default, allowed=None, aliases=(), summary=""):
        self.name = name
        self.default = default
        self.allowed = tuple(allowed) if allowed else None
        self.aliases = tuple(aliases)
        self.summary = summary

    @property
    def is_boolean(self):
        return self.allowed is None


FLAGS = {f.name: f for f in (
    # --- режимы (строковые) ---
    Flag("VLM_ARBITER_MODE", "on", ("off", "on"),
         summary="VLM-арбитр шорт-листа хука (Gemini, только HOOK, fail-open без ключа)"),
    Flag("SHOT_DIRECTOR_MODE", "off", ("off", "on"),
         summary="LLM-режиссёр запросов для блоков без своего запроса и без словаря"),
    Flag("VISUAL_DIRECTOR_MODE", "off", ("off", "shadow", "assist"),
         summary="Semantic Visual Director — реранк пула кандидатов по смыслу фразы"),
    Flag("LOOK_MANAGEMENT_MODE", "off", ("off", "shadow", "assist"),
         summary="Reference-Guided Look Management — коррекция кадра к эталону канала"),
    Flag("GRAIN_BLEND_MODE", "softlight", ("softlight", "grainmerge", "expr"),
         summary="Наложение зерна: softlight (нативный, быстрый) / grainmerge / expr (прежняя формула, медленно)"),
    Flag("DELIVERY_PROFILE", "youtube", ("youtube", "archive", "hevc"),
         summary="Финальный проход: youtube (VBV-потолок 12 Мбит/с) / archive (без потолка) / hevc (libx265)"),
    # --- булевы ---
    Flag("RENDER_STRICT_GATE", "1", aliases=(),
         summary="Не собирать final.mp4, если хоть один клип не принят"),
    Flag("DEFLICKER_ENABLED", "1", aliases=("DEFLICKER",),
         summary="Деффликер стокового ВИДЕО перед творческим грейдом"),
    Flag("OPENVERSE_ENABLED", "0",
         summary="Openverse (CC0 + институциональные источники) в ротации фото-источников"),
    Flag("STRESS_HINTS_ENABLED", "0",
         summary="Подсказки по ударению омографов в speech_plan_annotated.txt"),
)}

_warned = set()


def _spec(name):
    try:
        return FLAGS[name]
    except KeyError:
        raise KeyError(
            f"{name} нет в реестре scripts/feature_flags.py. Новый флаг режима "
            f"добавляется СЮДА (и в CLAUDE.md, и в config.example.env — "
            f"tests/test_feature_flags.py сверяет все три)") from None


def raw(name):
    """Сырое значение из окружения (с учётом исторических имён) или None."""
    spec = _spec(name)
    for key in (spec.name,) + spec.aliases:
        val = os.environ.get(key)
        if val is not None and val.strip() != "":
            return val
    return None


def mode(name):
    """Нормализованное значение строкового режима. Мусор -> "off" (см.
    докстринг модуля) с однократным предупреждением."""
    spec = _spec(name)
    if spec.is_boolean:
        raise TypeError(f"{name} — булев флаг, используй enabled({name!r})")
    val = raw(name)
    val = spec.default if val is None else val.strip().lower()
    if val not in spec.allowed:
        if name not in _warned:
            _warned.add(name)
            print(f"  ВНИМАНИЕ: {name}={val!r} не входит в {spec.allowed} — "
                  f"откатываюсь на 'off'.", file=sys.stderr)
        return "off"
    return val


def enabled(name):
    """True/False для булева флага. Выключает ТОЛЬКО значение "0" — см.
    докстринг модуля про сознательно не добавленные синонимы."""
    spec = _spec(name)
    if not spec.is_boolean:
        raise TypeError(f"{name} — режим со списком значений, используй mode({name!r})")
    val = raw(name)
    return (spec.default if val is None else val.strip()) != "0"


def value(name):
    """Текущее значение как строка — для сводки/снимка, без ветвления по типу."""
    spec = _spec(name)
    return mode(name) if not spec.is_boolean else ("1" if enabled(name) else "0")


def is_default(name):
    return raw(name) is None


def snapshot():
    """Снимок всех флагов — что РЕАЛЬНО исполнялось в этом прогоне.
    Пишется рядом с рендером (media_plan/feature_flags.json), чтобы «какой
    пайплайн собрал этот ролик» был проверяемым фактом, а не памятью."""
    return {name: {"value": value(name),
                   "default": FLAGS[name].default,
                   "from_env": not is_default(name)}
            for name in FLAGS}


def write_snapshot(video_dir):
    """Кладёт снимок флагов в <video_dir>/media_plan/feature_flags.json.
    Нужен затем же, зачем render_manifest.json: через неделю вопрос «а этот
    ролик собран с арбитром или без» должен иметь ответ в файле, а не в
    памяти. Любая ошибка записи — не повод ронять рендер (fail-open)."""
    import json
    try:
        plan_dir = os.path.join(video_dir, "media_plan")
        os.makedirs(plan_dir, exist_ok=True)
        path = os.path.join(plan_dir, "feature_flags.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot(), fh, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None


def format_summary():
    """Человекочитаемая сводка активных слоёв. Печатается в начале рендера:
    самый дешёвый способ увидеть, что дорогой слой выключен, ДО того как
    потрачены часы CPU (та же дисциплина, что ЧАСТЬ 1 CLAUDE.md)."""
    on, off = [], []
    for name in FLAGS:
        val = value(name)
        mark = "" if is_default(name) else " (из .env)"
        (off if val in ("off", "0") else on).append(f"{name}={val}{mark}")
    lines = [f"  Активные слои: {', '.join(on) if on else '— (все выключены)'}"]
    if off:
        lines.append(f"  Выключено: {', '.join(off)}")
    return "\n".join(lines)


def print_summary():
    print(format_summary())


if __name__ == "__main__":
    print_summary()
