#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кадр по брифу фразы — генерацией, как ещё один кандидат пула.

ЗАЧЕМ. Поиск по стокам и музеям находит то, что кто-то когда-то снял. Для
кадров действия («клинок влетает в щель между пластинами», «рыцарь не
может подняться») такой съёмки может не существовать вовсе, и судья
честно выбирает лучшего из негодных. Замер на эпизоде 94 (9 фраз, тот же
судья, одна сетка на слот): генерация выиграла 6 слотов из 9, поиск — 3
(подлинные предметы); средняя оценка 1.44 у поиска, 2.11 у генерации,
2.44 у лучшего из двух (docs/refactor/NORTH_STAR.md).

КАК ВСТРОЕНО. Сгенерированный кадр — не замена поиску и не отдельная
ветка выбора, а КАНДИДАТ того же пула: судья сравнивает его с найденными
на одной сетке. Поэтому неудачная генерация (не та эпоха, лишняя рука,
современная вещь, которой требует фраза) просто проигрывает — выбор
лучшего из пула по построению не ухудшает слот.

БЕЗ НИШИ В КОДЕ. Промпт = бриф фразы + рамка из паспорта мира эпизода
(регистр и окно эпохи) + требования к кадру (фото, без текста). Ни одного
слова о конкретной нише здесь нет.

ЛИЦЕНЗИЯ — закрыто по умолчанию. Генерировать можно только моделью, чья
лицензия на выдачу проверена по первоисточнику и записана в LICENSES;
неизвестная модель — отказ, а не «наверное можно». Лицензия уезжает в
провенанс кадра.
"""
import hashlib
import json
import os

# Версия способа построения кадра: меняется промпт или разбор — меняется
# ключ кэша, и кэш не отдаёт картинку, сделанную по старому правилу.
GEN_VERSION = 1

DEFAULT_MODEL = "am/flux.2-klein-4b"
DEFAULT_SIZE = "1792x1024"

# Проверено по первоисточнику 23.09.2026:
#  - FLUX.2 [klein] 4B — Apache 2.0 (huggingface.co/black-forest-labs/FLUX.2-klein-4B);
#  - FLUX.1 [dev] — лицензия модели некоммерческая, но п. 2(d): выдачу
#    можно использовать в любых целях, включая коммерческие, кроме обучения
#    конкурирующих моделей (huggingface.co/black-forest-labs/FLUX.1-dev, LICENSE.md).
# Условия самого шлюза для выдачи не проверены — записано в NORTH_STAR.md.
LICENSES = {
    "am/flux.2-klein-4b": "Apache-2.0",
    "am/flux.1-dev": "FLUX.1-dev: outputs usable commercially (s. 2(d)); no training of competing models",
}

SHOT_REQUIREMENTS = ("documentary photograph, realistic, natural cinematic light, "
                     "no text, no captions, no watermark, no logos")


def _year(y):
    return f"{-y} BC" if y < 0 else f"{y} AD"


def prompt_for(brief, card=None):
    """Промпт кадра: бриф фразы + рамка мира эпизода + требования к кадру.

    Бриф — главное и стоит первым. Рамка мира берётся ТОЛЬКО из паспорта
    эпизода (world_card): регистр и окно эпохи. Паспорта нет или окна нет —
    рамки нет, а не чужая."""
    import world_card
    brief = (brief or "").strip().rstrip(".")
    if not brief:
        return None
    parts = [brief]
    window = world_card.era_window(card) if card else None
    if window:
        parts.append(f"set in {_year(window[0])}-{_year(window[1])}")
    register = (card or {}).get("register")
    parts.append(f"{register} {SHOT_REQUIREMENTS}" if register else SHOT_REQUIREMENTS)
    return ", ".join(parts)


def cache_key(model, size, prompt):
    return hashlib.sha256(f"{GEN_VERSION}|{model}|{size}|{prompt}".encode("utf-8")).hexdigest()[:20]


def generate(gateway, brief, card, cache_dir, model=DEFAULT_MODEL, size=DEFAULT_SIZE):
    """Сгенерировать кадр по брифу. Возвращает dict (path, model, prompt,
    license, cost, cached) или None с причиной в поле error — никогда не
    исключение: сбой генерации не имеет права уронить отбор кадра, у слота
    остаются найденные кандидаты."""
    license_ = LICENSES.get(model)
    if not license_:
        return {"error": f"лицензия выдачи {model!r} не проверена — генерация закрыта"}
    prompt = prompt_for(brief, card)
    if not prompt:
        return {"error": "нет брифа"}
    key = cache_key(model, size, prompt)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, key + ".jpg")
    meta_path = path + ".meta.json"
    if os.path.exists(path) and os.path.getsize(path) > 0 and os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        return dict(meta, path=path, cached=True)
    try:
        images, cost = gateway.image(model, prompt, size)
    except Exception as e:  # noqa: BLE001 — любой сбой генерации: кандидата нет
        return {"error": f"{type(e).__name__}: {e}"}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(images[0])
    os.replace(tmp, path)
    meta = {"model": model, "size": size, "prompt": prompt, "brief": brief,
            "license": license_, "cost": cost, "gen_version": GEN_VERSION}
    tmpm = meta_path + ".tmp"
    with open(tmpm, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    os.replace(tmpm, meta_path)
    return dict(meta, path=path, cached=False)
