#!/usr/bin/env python3
"""ЗРЯЧИЙ ГЕЙТ КАДРА — единственный слой, который задаёт тот же вопрос, что и
владелец, глядя на контактный лист: «видно ли на этом кадре то, о чём говорит
фраза».

ЗАЧЕМ ОН ВООБЩЕ НУЖЕН, И ПОЧЕМУ ЭТОГО НЕ УМЕЕТ НИЧТО ИЗ УЖЕ ПОСТРОЕННОГО.
Все существующие гейты спрашивают «похожа ли картинка на пять слов запроса».
Это ДРУГОЙ вопрос, и 18.09 это измерено с пяти независимых сторон, каждый раз
с одинаковым результатом — распределения годных и брака перекрываются целиком:

  * абсолютный порог по сырому косинусу CLIP (два захода) — разделения нет;
  * сильная мультиязычная so400m судьёй (`docs/quality/so400m_as_judge.json`)
    — у слабой модели 7 браков из 17 выше медианы годных, у сильной ТЕ ЖЕ 7;
  * контрастивное вето по плиткам — на чужом корпусе 7 ложных отказов из 17;
  * близость ловушки к миру эпизода — у канала выше всего его же ловушки.

Вывод не про настройку, а про класс инструмента: эмбеддинг сравнивает
векторы и не отвечает на вопрос «есть ли в кадре коза». Отвечает на него
только модель, которая СМОТРИТ на кадр и ЧИТАЕТ фразу.

ЗАМЕР, НА КОТОРОМ ЭТОТ МОДУЛЬ ПОСТРОЕН (18.09, 8 реальных кадров: 6 промахов,
названных владельцем поимённо, плюс 2 годных кадра того же эпизода как
негативный контроль — гейт обязан ловить брак И не отклонять годное):

  ag/gemini-2.5-flash    6/8, поймала ВСЕ ШЕСТЬ названных браков
  qwen/qwen3.7-plus      6/8, тот же набор, в 2.4 раза дешевле
  glm/glm-4.6v           ответ не разбирается — её «6/8» артефакт подсчёта
  llama-3.2-11b (free)   4/8, и это худший исход: она НЕ ВИДИТ, а сочиняет
                         («на кратере вижу парашют», «на газетах человек
                         режет торт») — такой гейт уверенно пропускает брак

Дословные вердикты gemini на кадрах владельца:
  «коза не могла уснуть»      -> видит ветку с ягодами, не хватает КОЗЫ
  «Калди попробовал ягоды»    -> видит жареные зёрна, не хватает ЯГОД
  «остались мешки с зёрнами»  -> видит шатёр, не хватает МЕШКОВ
  «раскрывается парашют»      -> видит кратеры, не хватает ПАРАШЮТА

Оба «промаха» из 8 — кадры, которые размечающий пометил годными, а модель
отклонила; разбор показал, что в одном она права по существу (на гравюре
действительно нет ни кофе, ни запрета), а во втором расходится кадр-пробник
замера с кадром контактного листа, то есть это дефект замера, не модели.

ЦЕНА, ПОСЧИТАННАЯ, А НЕ ОЦЕНЁННАЯ. Один кадр 768 px = 1092 токена при
коэффициенте 0.6 => ~660 единиц баланса. На балансе ключа 2 358 933 единицы
это ~3500 проверок, то есть примерно 20-70 эпизодов в зависимости от числа
переподборов. Поэтому здесь есть жёсткий потолок вызовов за прогон и дисковый
кэш: повторный рендер того же эпизода не платит второй раз.

ЧЕГО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ. Он не ищет кадр и не ранжирует пул — он только
отвечает «да/нет» про УЖЕ ВЫБРАННОГО победителя. Решение, что делать с «нет»
(взять следующего кандидата, а при исчерпании — честная карточка), принимает
вызывающий код, и это сознательно: гейт, который сам лезет в отбор, нельзя
ни выключить, ни измерить отдельно.
"""
import base64
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.request

# Ключ читается из окружения (ANYMODEL_API_KEY, шапка CLAUDE.md), а .env
# грузит только pipeline_smart.py при СВОЁМ импорте — здесь этот модуль
# уже видит переменную, потому что импортирован ИЗ pipeline_smart.py.
# Собственный CLI-процесс файла (`python frame_verifier.py balance`) —
# отдельный процесс, которому .env никто не читал: без строки ниже ключ
# из файла не виден, и `balance`/ручная проверка вердикта молча отвечают
# «нет ключа» на машине, где он есть, просто в `.env`. Тот же паттерн,
# что уже стоит у speech_generate.py/lumean_tts.py/stock_fetch_
# multisource.py; override=False не перезатирает уже заданную переменную
# (CI, экспорт руками).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

try:                                   # тот же ленивый импорт, что у остальных слоёв
    import feature_flags
except Exception:                      # pragma: no cover - модуль всегда рядом
    feature_flags = None

# Версия промпта входит в ключ кэша: переписали вопрос — прежние вердикты
# отвечают уже не на него, и молча наследовать их значило бы повторить
# дефект, который в этом репозитории уже стоил кэшу вердиктов арбитра.
#
# 2 (19.09) — добавлен разрешённый бриф автора в сам вопрос. РЕАЛЬНЫЙ,
# живым прогоном найденный случай (не гипотеза, прямая жалоба владельца):
# фраза «Он весил меньше, чем ты думаешь — грамм триста» без разрешённого
# местоимения («он» = кинжал, названный двумя фразами раньше) — и гейт
# принял кандидата «гauntlet держит МЕЧ» вердиктом «да», хотя автор
# сценария давно разрешил это местоимение в инлайн-брифе `[shot:scene|a
# gauntleted hand holding THE DAGGER effortlessly]`. Живой контрольный
# вызов ТОЙ ЖЕ модели на ТОМ ЖЕ кадре: без брифа — «да, руки в латах
# держат меч»; с брифом — «нет, gauntleted hands holding sword hilt,
# missing dagger held by the tip». Вопрос не читал то, что уже разрешено
# в другом месте пайплайна (CLIP-запрос, `candidate_brief_keys()`,
# `brief_to_stock_query()`), и это и есть тот самый разрыв между «что мы
# искали» и «что мы проверяем», о котором был прямой запрос владельца
# «докопай глубже, без компромиссов».
PROMPT_VERSION = 2


# Выбрана замером на 8 реальных кадрах (6 названных владельцем браков + 2
# годных для контроля, шлюз anymodel.org): qwen/qwen3.7-plus и
# ag/gemini-2.5-flash дали одинаковый результат (6/8, все шесть браков
# пойманы), но qwen дешевле в 2.4 раза (0.25 против 0.6 за вызов) — на тот же
# баланс ключа это ~8000 проверок вместо ~3500. glm/glm-4.6v отклонена — ответ
# не разбирается парсером; бесплатная llama-3.2-11b — сочиняет несуществующее
# содержимое кадра (см. CLAUDE.md, раздел «ЗРЯЧИЙ ГЕЙТ КАДРА»). Честно: выборка
# в 8 кадров мала для железного вывода, но это те же данные, на которых дефолт
# уже был выбран, и при равной точности дешевле — разумный выбор.
DEFAULT_MODEL = "qwen/qwen3.7-plus"
DEFAULT_BASE_URL = "https://anymodel.org/v1"

# Cloudflare шлюза отдаёт 403 (error code 1010) на User-Agent питоновского
# urllib и пропускает curl. Изолировано перекрёстной проверкой, а не догадкой:
# ТЕКСТОВЫЙ вызов через urllib падал, КАРТИНКА через curl проходила — то есть
# дело в клиенте, а не в размере тела и не в картинке. Тот же класс, что уже
# записан в CLAUDE.md для Pexels/Cloudflare.
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# 768 px — замеренный компромисс: на нём модель уверенно читает содержимое
# кадра (все шесть браков владельца пойманы именно на этой стороне), а токенов
# уходит 1092. Поднимать сторону значит платить больше за тот же вердикт.
PROBE_MAX_SIDE = 768
JPEG_QUALITY = 85

MAX_CALLS_PER_RUN = int(os.environ.get("FRAME_VERIFIER_MAX_CALLS_PER_RUN", "200"))
TIMEOUT_SEC = int(os.environ.get("FRAME_VERIFIER_TIMEOUT", "90"))

STATS = {"calls": 0, "cache_hits": 0, "yes": 0, "no": 0,
         "errors": 0, "budget_stops": 0, "tokens": 0}

PROMPT = (
    "Ты монтажёр документального ролика.{world}Тебе дан КАДР и ФРАЗА "
    "диктора, которая звучит поверх этого кадра.\n\n"
    "ФРАЗА: «{phrase}»\n\n"
    "{intent}"
    "Вопрос ровно один: показывает ли этот кадр то, о чём говорит фраза"
    "{intent_ref}? "
    "Не «подходит ли по теме вообще», а видно ли на кадре именно то, что "
    "названо. Если назван КОНКРЕТНЫЙ предмет — он должен быть виден именно "
    "ЭТИМ предметом, а не похожим по силуэту другим (кинжал и меч, шлем и "
    "рыцарь целиком, современный нож и историческое оружие — разные "
    "предметы, силуэт не оправдание). Если во фразе назван предмет, "
    "существо или действие — оно должно быть видно. Если фраза — "
    "абстрактная мысль без предмета, засчитывается уместный образ.\n\n"
    "Ответь СТРОГО одной строкой JSON без пояснений:\n"
    '{{"verdict": "yes"|"no", "seen": "что реально на кадре, 3-6 слов", '
    '"missing": "чего не хватает, 3-6 слов или пусто"}}'
)

# Вставляется в {intent} промпта, ТОЛЬКО когда у юнита есть инлайн-бриф
# автора (`[shot:]`, см. shot_brief_director.py) — тот же текст, что уже
# отправлен в CLIP-поиск/полку/сток (candidate_brief_keys()), а не вторая,
# отдельно придуманная формулировка вопроса. Разрешает то, что сама фраза
# оставляет неоднозначным (местоимения, абстракции, пропущенный предмет) —
# автор уже сделал эту работу один раз при написании сценария (ЧАСТЬ 13,
# Шаг 3), и вопрос модели обязан пользоваться тем же решением, а не
# заново гадать по голому тексту диктора.
INTENT_CLAUSE = (
    "Уточнение автора сценария, что именно нужно показать на этом кадре "
    "(разрешает местоимения и абстракции во фразе выше): «{brief}»\n\n"
)
INTENT_REF = ", а точнее — то, что названо в уточнении автора выше"

# Мир кадра ЭТОГО канала, та же строка, что уже получает режиссёр брифов
# (shot_brief_director.domain_contract(), channel_profile.json -> shot_
# domain) — не вторая копия правила. Прямой ответ на запрос владельца
# «сделать облачный API умнее не в конкретных случаях, а везде»: без этого
# зрячий гейт сравнивает кадр ТОЛЬКО с буквальной фразой и не знает, что
# канал — например, «европейское Средневековье, 900-1600» — современный
# турист или техника в кадре на фразе без предметных слов формально
# «показывает то, что сказано», хотя очевидно чужероден миру ролика; тот
# же класс промаха, что уже ловит VISUAL_DOMAIN_GUARDS для формы клинка, но
# здесь — общим зрением, а не одним анкором. Пусто (мир не объявлен —
# психология, новый канал, SHOT_BRIEF_WORLD=off) — промпт байт-в-байт как
# раньше, третий параметр не добавляет ни одного лишнего токена.
WORLD_CLAUSE = (
    " Канал, для которого сделан ролик: {world} Если кадр явно чужероден "
    "этому миру (современная одежда, техника, интерьер там, где их не "
    "должно быть) — это брак, даже если формально по фразе всё на месте. "
)


def _world_context(video_dir=None):
    """domain_contract() того же режиссёра брифов, лениво — без цикла
    импорта (frame_verifier грузится РАНЬШЕ shot_brief_director внутри
    pipeline_smart.py, но эта функция читается только в момент вызова
    verify(), когда все модули уже загружены). Сбой любого рода -> "" —
    тот же fail-open, что у самого domain_contract().

    video_dir передаётся ДАЛЬШЕ, в domain_contract(video_dir) — не для
    красоты: без него та функция лезет за `import pipeline_smart`, а
    pipeline_smart.py — это модуль, который САМ импортирует frame_verifier
    (шапка pipeline_smart.py, `import frame_verifier`). Во время реального
    рендера это уже `__main__`, и `import pipeline_smart` изнутри
    domain_contract() заново выполнил бы файл целиком под вторым именем
    модуля. video_dir у verify() уже есть — платить за это незачем (см.
    докстринг domain_contract())."""
    try:
        import shot_brief_director
        return shot_brief_director.domain_contract(video_dir)
    except Exception:
        return ""


def reset_stats():
    for k in STATS:
        STATS[k] = 0


def _env(name, default=""):
    return (os.environ.get(name) or default).strip()


def model_name():
    return _env("FRAME_VERIFIER_MODEL", DEFAULT_MODEL)


def enabled():
    """Флаг реестра И наличие ключа. Ключ только из окружения (`.env`), в коде
    его нет и быть не может — см. шапку CLAUDE.md."""
    if feature_flags is not None:
        if not feature_flags.enabled("FRAME_VERIFIER"):
            return False
    elif _env("FRAME_VERIFIER", "1") == "0":
        return False
    return bool(_env("ANYMODEL_API_KEY"))


def _cache_path(video_dir, key):
    d = os.path.join(video_dir, "media_plan", "frame_verdicts")
    return os.path.join(d, key + ".json")


def _cache_key(image_path, phrase, world="", shot_brief=""):
    """Ключ по СОДЕРЖИМОМУ кадра, а не по его пути: один и тот же файл может
    лежать под разными именами кэша кандидата, а вердикт у него один. Плюс
    фраза, модель и версия промпта — вопрос изменился, значит и ответ другой.

    world входит ХЭШЕМ, не отдельным полем: если владелец поменяет мир
    канала в channel_profile.json (переезд на другую нишу, ЧАСТЬ 24) или
    включит content_world.json для нового эпизода, эффективный вопрос
    модели меняется, а прежний ключ (кадр+фраза+модель) остался бы тем же
    — вердикт «да, показывает то, что сказано» молча выжил бы после смены
    мира, хотя вопрос уже задан другой. Пустой world (SHOT_BRIEF_WORLD=off,
    новый канал без объявленной ниши) даёт тот же хэш, что и раньше этой
    правки — старый кэш таких эпизодов не протухает беспричинно.

    shot_brief — та же логика, тем же способом (19.09, PROMPT_VERSION 2):
    разрешённый бриф автора меняет ЭФФЕКТИВНЫЙ вопрос модели («меч» вместо
    «он»), значит вердикт «да» без брифа и вердикт с брифом отвечают на
    разные вопросы, даже если фраза и кадр те же самые. Пустой бриф (юнит
    без [shot:], старый эпизод без режиссёра брифов) — тот же хэш, что и
    без этого параметра: PROMPT_VERSION уже бампнут, поэтому старые записи
    и так не переживут эту правку, а не бампать хэш ЕЩЁ РАЗ пустой строкой
    незачем."""
    h = hashlib.md5()
    try:
        with open(image_path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return None
    h.update(b"|")
    h.update((phrase or "").strip().encode("utf-8"))
    h.update(b"|")
    h.update(model_name().encode("utf-8"))
    h.update(b"|v%d" % PROMPT_VERSION)
    h.update(b"|w")
    h.update((world or "").encode("utf-8"))
    h.update(b"|b")
    h.update((shot_brief or "").strip().encode("utf-8"))
    return h.hexdigest()


def _as_data_url(image_path):
    from PIL import Image
    im = Image.open(image_path).convert("RGB")
    im.thumbnail((PROBE_MAX_SIDE, PROBE_MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _parse(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    v = str(d.get("verdict", "")).strip().lower()
    if v not in ("yes", "no"):
        return None
    return {"verdict": v,
            "seen": str(d.get("seen") or "")[:120],
            "missing": str(d.get("missing") or "")[:120]}


def _ask(phrase, data_url, world="", shot_brief=""):
    key = _env("ANYMODEL_API_KEY")
    base = _env("ANYMODEL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    # world пуст (SHOT_BRIEF_WORLD=off, канал без объявленной ниши, сбой
    # domain_contract()) -> {world} рендерится ОДНИМ пробелом — байт-в-байт
    # тот же текст промпта, что был до этой правки ("ролика. Тебе дан...").
    world_clause = WORLD_CLAUSE.format(world=world) if world else " "
    # Бриф пуст (юнит без [shot:], SHOT_BRIEF_DIRECTOR не гонялся) -> оба
    # {intent}/{intent_ref} рендерятся пусто — байт-в-байт тот же вопрос,
    # что был до PROMPT_VERSION 2 (см. её докстринг у объявления).
    shot_brief = (shot_brief or "").strip()
    intent_clause = INTENT_CLAUSE.format(brief=shot_brief) if shot_brief else ""
    intent_ref = INTENT_REF if shot_brief else ""
    body = json.dumps({
        "model": model_name(),
        "max_tokens": 200,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(
                phrase=phrase, world=world_clause,
                intent=intent_clause, intent_ref=intent_ref)},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as r:
        d = json.load(r)
    usage = d.get("usage") or {}
    STATS["tokens"] += int(usage.get("total_tokens") or 0)
    return d["choices"][0]["message"]["content"]


def verify(image_path, phrase, video_dir=None, shot_brief=None):
    """(вердикт, детали) про ОДИН кадр и ОДНУ фразу.

    shot_brief (опционально) — уже разрешённый автором инлайн-бриф этого
    юнита (`[shot:]`, ЧАСТЬ 13, Шаг 3), ТА ЖЕ строка, что уже уходит в
    CLIP-поиск/полку/сток (`candidate_brief_keys()` в pipeline_smart.py).
    Без него голая фраза диктора может быть неоднозначной (местоимение
    без антецедента, абстракция без предмета) — модель тогда отвечает на
    более слабый вопрос, чем тот, которым кандидат вообще нашли, и может
    засчитать силуэтно похожий, но другой предмет (см. PROMPT_VERSION у
    её объявления — живой пример, не гипотеза). Нет брифа (юнит без
    [shot:], старый эпизод) -> прежний вопрос байт-в-байт.

    Возвращает dict с verdict/seen/missing, либо None — и None здесь значит
    ИМЕННО «мнения нет» (выключено, нет ключа, нет сети, исчерпан потолок,
    ответ не разобрался), а не «плохо». Вызывающий на None обязан оставить
    прежнее поведение байт-в-байт: молчаливый отказ, притворяющийся вердиктом,
    — тот самый класс дефекта, который в этом репозитории уже стоил отбору
    целого слоя (см. NO_CANDIDATE_FITS у VLM-арбитра).
    """
    if not enabled() or not phrase or not image_path:
        return None
    if not os.path.exists(image_path):
        return None

    world = _world_context(video_dir)
    key = _cache_key(image_path, phrase, world, shot_brief)
    cpath = _cache_path(video_dir, key) if (video_dir and key) else None
    if cpath and os.path.exists(cpath):
        try:
            with open(cpath, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("verdict") in ("yes", "no"):
                STATS["cache_hits"] += 1
                return cached
        except Exception:
            pass

    if STATS["calls"] >= MAX_CALLS_PER_RUN:
        STATS["budget_stops"] += 1
        return None

    try:
        data_url = _as_data_url(image_path)
    except Exception:
        STATS["errors"] += 1
        return None

    try:
        STATS["calls"] += 1
        txt = _ask(phrase, data_url, world, shot_brief)
    except Exception:
        # Fail-open на уровне слоя: недоступный шлюз не имеет права ни ронять
        # рендер, ни отклонять кадр. Считается и печатается вызывающим.
        STATS["errors"] += 1
        return None

    out = _parse(txt)
    if out is None:
        STATS["errors"] += 1
        return None
    STATS[out["verdict"]] += 1

    if cpath:
        try:
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            tmp = cpath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
            os.replace(tmp, cpath)
        except Exception:
            pass
    return out


def balance():
    """Остаток на ключе или None. Отдельной командой, не на каждом кадре:
    баланс — состояние счёта, а не свойство кадра."""
    key = _env("ANYMODEL_API_KEY")
    if not key:
        return None
    base = _env("ANYMODEL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    req = urllib.request.Request(base + "/balance", headers={
        "Authorization": "Bearer " + key, "User-Agent": BROWSER_UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def summary_line():
    s = STATS
    if not (s["calls"] or s["cache_hits"]):
        return ""
    return (f"  Зрячий гейт кадра ({model_name()}): вызовов {s['calls']}, "
            f"из кэша {s['cache_hits']}, годных {s['yes']}, отклонено {s['no']}, "
            f"ошибок {s['errors']}, токенов {s['tokens']}"
            + (f", УПЁРСЯ В ПОТОЛОК {MAX_CALLS_PER_RUN}" if s["budget_stops"] else ""))


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "balance":
        print(json.dumps(balance(), ensure_ascii=False, indent=1))
    elif len(sys.argv) >= 3:
        print(json.dumps(verify(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=1))
        print(summary_line())
    else:
        print("usage: frame_verifier.py balance | frame_verifier.py <image> <фраза>")
