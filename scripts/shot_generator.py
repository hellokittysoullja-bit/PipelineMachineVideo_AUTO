#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кадр по фразе — генерацией, как ещё один кандидат слота.

ЗАЧЕМ. Поиск по стокам и музеям находит то, что кто-то когда-то снял. Для
кадров действия («клинок влетает в щель между пластинами», «рыцарь не
может подняться») такой съёмки может не существовать вовсе, и судья
честно выбирает лучшего из негодных. Замер на эпизоде 94 (9 фраз, тот же
судья, одна сетка на слот): генерация выиграла 6 слотов из 9, поиск — 3
(подлинные предметы); средняя оценка 1.44 у поиска, 2.11 у генерации,
2.44 у лучшего из двух (docs/refactor/NORTH_STAR.md).

ПОДКЛЮЧЕНО 27.09 (решение владельца): ступень лестницы слота ПОСЛЕ второго
круга поиска — кадра в источниках нет или есть только замена без главного
фразы (pipeline_smart.generation_round).

КАК ВСТАЁТ. Сгенерированный кадр — не замена поиску и не отдельная ветка
выбора, а КАНДИДАТ слота: его судит тот же судья по тем же пунктам
спецификации, проверяет тот же вопрос о мире. Неудачная генерация (не тот
предмет, лишняя рука, псевдонадпись) просто проигрывает — на экран она не
встаёт и слот не ухудшает.

СТИЛЬ — РИСОВАННЫЙ, А НЕ ФОТО (решение владельца 27.09). Фотореализм
генератора читается как ИИ-слоп; цветной рисунок тушью и карандашом — как
авторская графика документального фильма. Подобран живыми пробами на
FLUX.2 Klein 4B по восьми фразам эпизода 97 (СДВГ):
  * стиль — КОРОТКИМ хвостом после предмета: длинное описание стиля у
    маленькой модели съедало предмет (таймер превращался в будильник);
  * чёрно-белый карандаш — не то, что нужно владельцу; цветной акцент
    словами «один акцент» модель игнорирует, палитра из трёх цветов
    держится;
  * «no text, no numbers» снимает псевдобуквы почти целиком (1 утечка из
    16 — на «тетради учёного», которая сама просит записей).
Стиль канала переопределяется в channel_profile.json → image_generation.style.

ОПИСАНИЕ КАДРА ПИШЕТ МОЗГ (describe), по правилам, выведенным из проб:
предмет первым; предмет, которого модель не знает («кухонный таймер»),
описывается внешним видом или заменяется узнаваемым с тем же смыслом
(песочные часы); действие — деталью крупным планом («рука прижимает
крышку ноутбука», а не «человек закрывает ноутбук»); люди одеты и
обыкновенны; никаких надписей, вывесок, экранов с текстом. Мозга нет —
берётся бриф фразы.

БЕЗ НИШИ В КОДЕ. Рамка мира — только из паспорта эпизода (регистр и окно
эпохи). Ни одного слова о конкретной нише здесь нет.

ЛИЦЕНЗИЯ — закрыто по умолчанию. Генерировать можно только моделью, чья
лицензия на выдачу проверена по первоисточнику и записана в LICENSES;
неизвестная модель — отказ, а не «наверное можно». Лицензия уезжает в
провенанс кадра.

ДЕНЬГИ. Klein 4B в каталоге шлюза стоит 0 (billing.coefficient 0, проверено
27.09). Шлюз генерации создаётся с потолком расходов 0
(IMAGE_GEN_MAX_SPEND), поэтому модель, ставшая платной, не спишет ни
токена: шлюз откажет ДО вызова по резерву.
"""
import hashlib
import json
import os
import pathlib

# Версия способа построения кадра: меняется промпт или разбор — меняется
# ключ кэша, и кэш не отдаёт картинку, сделанную по старому правилу.
GEN_VERSION = 2

DEFAULT_MODEL = "am/flux.2-klein-4b"
DEFAULT_SIZE = "1792x1024"
# Вариантов на слот: у маленькой модели одна попытка из двух бывает с
# браком (два слитых мозга, закрытые глаза, лишний предмет), а вторую
# судья сравнивает на той же сетке — бесплатно.
VARIANTS = 2

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

STYLE_DEFAULT = ("Ink line drawing with colored pencil shading on textured paper, confident sketchy lines, "
                 "limited palette of teal, ochre and warm red, dramatic cinematic light, "
                 "no text, no numbers, no signs, no labels")


def style_for(profile=None):
    """Стиль сгенерированных кадров канала: channel_profile.json →
    image_generation.style, иначе STYLE_DEFAULT."""
    if profile is None:
        import channel_profile
        profile = channel_profile.load()
    s = ((profile or {}).get("image_generation") or {}).get("style")
    return " ".join(str(s).split()) if isinstance(s, str) and s.strip() else STYLE_DEFAULT


def _year(y):
    return f"{-y} BC" if y < 0 else f"{y} AD"


def prompt_for(brief, card=None, style=None):
    """Промпт кадра: описание предмета первым + рамка мира эпизода + стиль.

    Рамка мира берётся ТОЛЬКО из паспорта эпизода (world_card): окно
    эпохи. Паспорта нет или окна нет — рамки нет, а не чужая."""
    import world_card
    brief = (brief or "").strip().rstrip(".")
    if not brief:
        return None
    parts = [brief]
    window = world_card.era_window(card) if card else None
    if window:
        parts.append(f"set in {_year(window[0])}-{_year(window[1])}")
    parts.append(style or STYLE_DEFAULT)
    return ", ".join(parts)


DESCRIBE_VERSION = 1
DESCRIBE_PROMPT = """You write the picture description for ONE shot of a documentary film. An image model will draw it by hand.
Narration line (the viewer hears it while seeing the picture): «{phrase}»
What the viewer must see: «{focus}»
Must be visible: {musts}{world}
Rules — the image model is small and literal:
1. Start with the main subject in plain words, then where it is. One or two sentences, at most 35 words.
2. If the subject is an object the model may not know by name, describe how it looks, or use a familiar object with the same meaning (a timer set for a short task -> a glass hourglass).
3. Show an action through a close-up detail (a hand pressing a laptop lid shut), not a whole person doing it.
4. People are ordinary and clothed. No famous people.
5. No writing anywhere: no text, letters, numbers, signs, labels, screens with words, documents, books with writing.
6. Do not describe the drawing style, only what is in the picture.
Reply with the description only."""


def _musts(spec):
    claims = (spec or {}).get("claims") or []
    got = [c.get("text") for c in claims if c.get("tier") == "must" and c.get("text")]
    return "; ".join(got) or "—"


def describe(gateway, model, *, phrase, spec, brief, card, cache_dir=None):
    """(описание кадра, info). Мозг пишет короткое описание по правилам
    выше; сбой, пустой ответ или нет шлюза — бриф фразы (или фокус
    спецификации), чтобы генерация не зависела от мозга целиком."""
    import world_card
    fallback = (brief or (spec or {}).get("focus") or "").strip()
    if gateway is None or not spec:
        return fallback, {"origin": "brief"}
    window = world_card.era_window(card) if card else None
    world = f"\nThe film's world: {world_card.judge_setting(card)}." if window else ""
    text = DESCRIBE_PROMPT.format(phrase=phrase or "", focus=spec.get("focus") or brief or "",
                                  musts=_musts(spec), world=world)
    key = hashlib.sha256(f"{DESCRIBE_VERSION}|{model}|{text}".encode("utf-8")).hexdigest()[:20]
    cp = os.path.join(cache_dir, "describe_" + key + ".json") if cache_dir else None
    if cp and os.path.exists(cp):
        try:
            return json.load(open(cp, encoding="utf-8"))["text"], {"origin": "model", "cache_hit": True}
        except Exception:  # noqa: BLE001 — битый кэш: спросить заново
            pass
    try:
        ans, _u, price = gateway.chat(model, [{"type": "text", "text": text}], 4000, 600)
    except Exception as e:  # noqa: BLE001 — сбой мозга: бриф вместо описания
        return fallback, {"origin": "brief", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    desc = " ".join((ans or "").replace('"', " ").split())
    if not desc:
        return fallback, {"origin": "brief", "error": "пустой ответ"}
    if cp:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"text": desc, "model": model}, f, ensure_ascii=False)
        os.replace(tmp, cp)
    return desc, {"origin": "model", "cost": price}


def cache_key(model, size, prompt, variant=0):
    return hashlib.sha256(f"{GEN_VERSION}|{model}|{size}|{variant}|{prompt}".encode("utf-8")).hexdigest()[:20]


def generate(gateway, brief, card, cache_dir, model=DEFAULT_MODEL, size=DEFAULT_SIZE, variant=0,
             style=None):
    """Сгенерировать кадр по описанию. Возвращает dict (path, model, prompt,
    license, cost, cached) или {"error": причина} — никогда не исключение:
    сбой генерации не имеет права уронить отбор кадра, у слота остаются
    найденные кандидаты."""
    license_ = LICENSES.get(model)
    if not license_:
        return {"error": f"лицензия выдачи {model!r} не проверена — генерация закрыта"}
    prompt = prompt_for(brief, card, style)
    if not prompt:
        return {"error": "нет брифа"}
    key = cache_key(model, size, prompt, variant)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, key + ".png")
    meta_path = path + ".meta.json"
    if os.path.exists(path) and os.path.getsize(path) > 0 and os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        return dict(meta, path=path, key=key, cached=True)
    try:
        images, cost = gateway.image(model, prompt, size)
    except Exception as e:  # noqa: BLE001 — любой сбой генерации: кандидата нет
        return {"error": f"{type(e).__name__}: {e}"}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(images[0])
    os.replace(tmp, path)
    meta = {"model": model, "size": size, "prompt": prompt, "brief": brief, "variant": variant,
            "license": license_, "cost": cost, "gen_version": GEN_VERSION}
    tmpm = meta_path + ".tmp"
    with open(tmpm, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    os.replace(tmpm, meta_path)
    return dict(meta, path=path, key=key, cached=False)


def candidate(meta):
    """Сгенерированный кадр в форме кандидата кучи фото (как файл Шага 4:
    ссылка file://, url и alt пустые — жанровый фильтр по тексту и подпись
    для судьи не должны реагировать на слова промпта). id — «gen:<ключ>»:
    по префиксу кандидат считается отдельным источником в отчётах."""
    uri = pathlib.Path(os.path.abspath(meta["path"])).as_uri()
    return {"id": "gen:" + meta["key"], "alt": "", "url": "",
            "_gen_meta": {k: meta.get(k) for k in ("model", "prompt", "brief", "variant", "license",
                                                    "gen_version")},
            "src": {"medium": uri, "large2x": uri, "large": uri}}
