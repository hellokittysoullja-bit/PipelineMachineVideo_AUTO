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
GEN_VERSION = 5

# FLUX.1 [dev] вместо Klein 4B — решение владельца 30.09. Замер на одном
# задании («рыцарь на коленях отдаёт перчатку»): у dev чище композиция и
# фигуры, но картинка гладкая, ближе к иллюстрации, чем к туши и карандашу,
# и в углу появилась нарисованная подпись (текст в кадре вопреки промпту);
# время 10.5 с против 4.7 с, разрешение то же 1344x768, цена 0. Качество на
# эпизоде НЕ замерено — поймает судья или контактный лист.
# Gemini 3.1 Flash Image вместо FLUX dev — решение владельца 01.10 после пробы
# (слот 2 эп.03): цветной карандаш с штриховкой и фактурой бумаги, сцена
# читается, ИИ-глянца нет. Шлюз НЕ принимает 1792x1024 (400), принимает
# 1536x1024; quality=low — 37 500 токенов за картинку (замер 01.10).
DEFAULT_MODEL = (os.environ.get("IMAGE_GEN_MODEL") or "").strip() or "ag/gemini-3.1-flash-image"
DEFAULT_SIZE = (os.environ.get("IMAGE_GEN_SIZE") or "").strip() or "1536x1024"
# Вариантов на слот: у маленькой модели попытка нередко выходит с браком
# (лишние руки, слитые предметы, закрытые глаза), а все варианты судья
# сравнивает на одной сетке (до 9 плиток) — деньгами бесплатно. Цена —
# время: ~35-40 с на вариант (эп.98), только на слотах, где сработала
# генерация. 4 — решение владельца 27.09.
QUALITY = (os.environ.get("IMAGE_GEN_QUALITY") or "").strip() or "low"
VARIANTS = int(os.environ.get("IMAGE_GEN_VARIANTS") or 1)

# Проверено по первоисточнику 23.09.2026:
#  - FLUX.2 [klein] 4B — Apache 2.0 (huggingface.co/black-forest-labs/FLUX.2-klein-4B);
#  - FLUX.1 [dev] — лицензия модели некоммерческая, но п. 2(d): выдачу
#    можно использовать в любых целях, включая коммерческие, кроме обучения
#    конкурирующих моделей (huggingface.co/black-forest-labs/FLUX.1-dev, LICENSE.md).
# Условия самого шлюза для выдачи не проверены — записано в NORTH_STAR.md.
LICENSES = {
    # Gemini 3.1 Flash Image («nano banana»): решение владельца 01.10. Условия
    # выдачи Google на шлюзе НЕ проверены по первоисточнику. ПЛАТНАЯ: 100 000
    # токенов базы x 1.75 за 1792x1024 = 175 000 за картинку.
    "ag/gemini-3.1-flash-image": "Google Gemini image: terms not verified (owner decision 2026-10-01)",
}

# Стиль подобран серией из ~300 пробных картинок FLUX dev (30.09, глаза Claude, не
# разметка владельца). Решение владельца: карандаш, не глянец, не «нейросетевое»,
# короткие простые слова (dev плохо понимает длинные промпты).
# Что замерено: «ink/colored pencil/палитра» даёт глянцевую цифровую графику и
# подписи в углу; «page/sketchbook» рисует пустой лист с фигуркой; «no blank white»
# тоже (читается буквально). Работает «уголь и карандаш на старой серо-коричневой
# бумаге, рисунок от края до края»: фон заполнен, подписей нет, эпоха держится,
# если описание кадра начинается словами «In the Middle Ages» (без эпохи модель
# рисовала пиджаки, телефонный столб и грузовик).
STYLE_DEFAULT = ("Hand-drawn colored pencil illustration on textured paper, visible pencil strokes and hatching, "
                 "soft warm earthy colors, no digital gloss, no text, no letters.")


def style_for(profile=None):
    """Стиль сгенерированных кадров канала: channel_profile.json →
    image_generation.style, иначе STYLE_DEFAULT."""
    if profile is None:
        import channel_profile
        profile = channel_profile.load()
    s = ((profile or {}).get("image_generation") or {}).get("style")
    return " ".join(str(s).split()) if isinstance(s, str) and s.strip() else STYLE_DEFAULT


def prompt_for(brief, card=None, style=None):
    """Промпт кадра: описание предмета первым + стиль.

    ГОДОВ В ПРОМПТЕ НЕТ — найдено живым прогоном эп.98 (27.09): рамка
    «set in 700 AD-2024 AD» рисовалась маленькой моделью как НАДПИСЬ на
    картинке — «700–2024 AD» стояло на трёх рисунках из четырёх, и судья это
    пропустил. Эпоху генератору передаёт описание кадра словами (describe
    получает мир эпизода и запрет писать даты). card оставлен в подписи:
    вызывающие его передают, а смысл — «мир эпизода известен описанию»."""
    brief = (brief or "").strip().rstrip(".")
    if not brief:
        return None
    # Gemini: стиль первым, затем сцена и композиция (так сделана удачная проба).
    return " ".join([(style or STYLE_DEFAULT).rstrip(), brief + ". Wide cinematic composition."])


DESCRIBE_VERSION = 3
DESCRIBE_PROMPT = """You write the picture description for ONE shot of a documentary film. An image model will draw it.
Narration line (the viewer hears it while seeing the picture): «{phrase}»
What the viewer must see: «{focus}»
Must be visible: {musts}{world}
Rules:
1. Two or three plain sentences, at most 50 words. Say who or what is in the picture, what they do, and where (ground, weather, sky, what is behind them).
2. Name the pose and the action exactly (lying on his back, hands open; a tool held near him but not touching him), so the picture shows this moment and not a generic scene.
3. If the subject is an object the model may not know by name, describe how it looks.
4. People are ordinary and clothed. No famous people.
5. No writing anywhere: no text, letters, numbers, dates, years, signs, labels, screens with words, documents, books with writing.
6. If the world is historical, START with the period in plain words (for example «In the Middle Ages, ...»), never as years or digits. Describe period clothing and tools by name.
7. Do not describe the drawing style, only what is in the picture.
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
    import llm_gateway

    def cache_path(m):
        # Ключ с моделью, которая ответила: у первичной он прежний, ответ
        # запасной (цепочка llm_gateway.text_chain) лежит под её именем.
        key = hashlib.sha256(f"{DESCRIBE_VERSION}|{m}|{text}".encode("utf-8")).hexdigest()[:20]
        return os.path.join(cache_dir, "describe_" + key + ".json") if cache_dir else None

    def cached(m):
        cp = cache_path(m)
        if cp and os.path.exists(cp):
            try:
                return json.load(open(cp, encoding="utf-8"))["text"]
            except Exception:  # noqa: BLE001 — битый кэш: спросить заново
                return None
        return None
    try:
        got = llm_gateway.chat_fallback(gateway, llm_gateway.text_chain(model),
                                        [{"type": "text", "text": text}], 4000, 600, cached=cached)
    except Exception as e:  # noqa: BLE001 — сбой мозга: бриф вместо описания
        return fallback, {"origin": "brief", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    if got.from_cache:
        return got[0], {"origin": "model", "cache_hit": True, "model": got.model}
    ans, _u, price = got
    desc = " ".join((ans or "").replace('"', " ").split())
    if not desc:
        return fallback, {"origin": "brief", "error": "пустой ответ"}
    cp = cache_path(got.model)
    if cp:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"text": desc, "model": got.model}, f, ensure_ascii=False)
        os.replace(tmp, cp)
    return desc, {"origin": "model", "cost": price, "model": got.model}


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
    key = cache_key(model, size + (f"|{QUALITY}" if QUALITY else ""), prompt, variant)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, key + ".png")
    meta_path = path + ".meta.json"
    if os.path.exists(path) and os.path.getsize(path) > 0 and os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        return dict(meta, path=path, key=key, cached=True)
    try:
        images, cost = gateway.image(model, prompt, size, quality=QUALITY)
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
