#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Паспорт мира ЭПИЗОДА — из текста сценария, а не из кода и не из канала.

ЗАЧЕМ (измерено по золотому набору опубликованного эпизода 01, 17.09, не
гипотеза). Из 17 брака в наборе:

* **4 кадра дал ОДИН запрос** `greatsword warrior fight` — в нём нет ни
  эпохи, ни культуры, и поиск честно принёс современное спортивное
  фехтование и косплей. Ещё 2 дал `medieval knight sword battle`
  (реконструкторский фестиваль с толпой зрителей в футболках).
* **8 из 17 гейты забраковали САМИ**, и кадры всё равно ушли в ролик —
  потому что слот не имеет права остаться пустым, а альтернативы у него не
  было. Это отдельная задача (лестница фолбэков), но её решение тоже
  опирается на вопрос «а что в этом эпизоде вообще допустимо».

При этом «что допустимо» в системе было размазано по ПЯТИ местам, и главное
из них — литерал в коде:

    _QUERY_ERA_ANCHORS_DEFAULT = ("medieval", "knight", "armour", ... )

Сорок средневековых слов, зашитых в `pipeline_smart.py`; `channel_profile.
json` этого канала их даже не переопределяет. Для эпизода про психологию
или про неандертальцев этот список не «менее точен» — он просто про другое,
и ровно он подставлялся бы в каждый запрос к стоку. Тот же класс уже
измерен и записан в CLAUDE.md: бриф `a phone lying face down on a bedside
table at night` уезжал в сток как `medieval phone lying face down`.

ЧТО ЭТО ЗА ФАЙЛ. `media_plan/world_card.json` — один на эпизод, выводится
ИЗ ТЕКСТА этого эпизода: окно эпохи числами, культуры, регистр, что в этом
эпизоде запрещено показывать, каких предметов ждём, какими словами якорить
запрос к стоку. Дальше его читают все, кому нужно знать мир: гейт запросов,
маршрутизация по источникам, приёмка кадра.

МОЗГ СМЕНЯЕМЫЙ, ХАРНЕСС ОБЩИЙ — тот же приём, что уже закрыл разрыв у
брифов кадра (`shot_brief_director.py --brain packets/local/file`). Этот
модуль НЕ содержит ни одной эвристики по словам: он умеет собрать вопрос
(`prompt_for_script`), проверить ответ (`validate`) и отдать данные
потребителям. Кто отвечает — сессия Claude (она и так пишет сценарий,
ЧАСТЬ 13 Шаг 2-3), Gemini одним вызовом или человек руками — решает
вызывающий код. Эвристика по словам здесь была бы тем же хардкодом, только
переехавшим на этаж ниже.

ЭПОХА — ЗНАКОВЫЕ ГОДЫ, И ЭТО НЕ ПРИДИРКА К ФОРМАТУ. Требование владельца
прямое: ниша может быть любой, в том числе доисторической. Поэтому окно
хранится целыми числами со знаком (`-40000` = 40000 год до н.э.), а не
строкой «XV век»: строку нельзя сравнить с паспортом музейного предмета
(`objectBeginDate`/`objectEndDate` у Мет — тоже знаковые целые), а окно
эпохи — можно, теми же функциями, что уже работают в `museum_sources.
era_overlaps()`.

FAIL-OPEN НА ОТСУТСТВИЕ, LOUD НА ПОЛОМКУ. Нет файла — все потребители
работают как сегодня, байт-в-байт (эпизоды, написанные до этой правки, не
трогаются). Файл ЕСТЬ и сломан — `load()` бросает `WorldCardError` со
списком проблем, а не возвращает None. Причина не в строгости ради
строгости: на паспорте держится приёмка кадра, и «молча считать, что мира
нет» здесь означает «молча выключить защиту» — ровно тот класс тихого
no-op, которым этот репозиторий горел шесть раз.
"""
import json
import os
import time

SCHEMA_VERSION = 1
CARD_NAME = "world_card.json"

#: Регистр эпизода. Решает, какой вопрос вообще осмысленно задавать кадру.
#: `historical` — есть окно эпохи и культура, современность в кадре брак;
#: `modern` — действие сегодня, «анахронизм» не определён, зато брак —
#: исторические декорации (измеренный случай: медиевализмы в психологическом
#: сценарии); `abstract` — показывать нечего буквально, кадр заземляется
#: ситуацией или предметом-следом; `scientific` — схемы, приборы, натура;
#: `mixed` — эпизод честно смешанный, окно эпохи есть, но современные
#: аналогии законны (эпизод 01: «сколько весит пакет молока» — современный
#: пакет молока на средневековом канале, и это ПРАВИЛЬНЫЙ кадр).
REGISTERS = ("historical", "modern", "abstract", "scientific", "mixed")

#: Ключи, которые обязан назвать любой паспорт. Пустое значение допустимо
#: (кроме register/era_anchors — см. validate), отсутствие ключа — нет:
#: «забыли» и «в этом эпизоде нечего запрещать» должны различаться.
REQUIRED_KEYS = (
    "schema_version", "register", "era", "culture",
    "must_not_show", "expected_subjects", "era_anchor_terms",
)

_YEAR_MIN, _YEAR_MAX = -4000000, 2200


class WorldCardError(Exception):
    """Паспорт есть на диске и сломан. Не то же самое, что паспорта нет."""


def path(video_dir):
    return os.path.join(video_dir, "media_plan", CARD_NAME)


def _as_terms(value):
    """Список терминов -> кортеж нижнего регистра без пустых и дублей,
    порядок автора сохраняется (первый термин используется как основной
    якорь, см. era_anchors)."""
    out, seen = [], set()
    for x in value or ():
        t = str(x).strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return tuple(out)


def validate(card):
    """Список проблем паспорта. Пустой список — паспорт годен.

    Возвращаем СПИСОК, а не бросаем: тот же приём, что у валидации
    speech_plan — вызывающий код (или человек с черновиком паспорта в
    руках) должен увидеть ВСЕ проблемы сразу, а не первую.
    """
    bad = []
    if not isinstance(card, dict):
        return ["паспорт не объект JSON"]
    for k in REQUIRED_KEYS:
        if k not in card:
            bad.append(f"нет обязательного ключа {k!r}")
    if card.get("schema_version") != SCHEMA_VERSION:
        bad.append(f"schema_version={card.get('schema_version')!r}, "
                   f"ожидается {SCHEMA_VERSION}")
    reg = card.get("register")
    if reg not in REGISTERS:
        bad.append(f"register={reg!r} — допустимо только {REGISTERS}")

    era = card.get("era")
    if era is None:
        # Осознанный законный случай: у психологии и абстракции окна эпохи
        # нет, и требовать его значило бы заставлять автора выдумывать.
        if reg == "historical":
            bad.append("register=historical, но era=null — историческому "
                       "эпизоду окно эпохи обязательно")
    elif not isinstance(era, dict):
        bad.append("era должна быть объектом {from, to} или null")
    else:
        a, b = era.get("from"), era.get("to")
        for name, v in (("from", a), ("to", b)):
            if not isinstance(v, int) or isinstance(v, bool):
                bad.append(f"era.{name}={v!r} — нужно целое число со знаком "
                           f"(отрицательное = до н.э.)")
        if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
            if a > b:
                bad.append(f"era.from={a} больше era.to={b}")
            if not (_YEAR_MIN <= a <= _YEAR_MAX and _YEAR_MIN <= b <= _YEAR_MAX):
                bad.append(f"era вне диапазона [{_YEAR_MIN}, {_YEAR_MAX}]: {a}..{b}")

    cult = card.get("culture")
    if cult is not None and not isinstance(cult, dict):
        bad.append("culture должна быть объектом {include, exclude} или null")

    for k in ("must_not_show", "expected_subjects", "era_anchor_terms"):
        v = card.get(k)
        if v is not None and not isinstance(v, (list, tuple)):
            bad.append(f"{k} должен быть списком строк")

    # Якоря — единственное, что молча ломает подбор, если их нет: запрос
    # уходит в сток без привязки к миру, и это ровно те 4 брака из одного
    # запроса, ради которых паспорт и заведён.
    if not _as_terms(card.get("era_anchor_terms")):
        if reg in ("historical", "scientific", "mixed"):
            bad.append("era_anchor_terms пуст — запрос к стоку будет без "
                       "привязки к миру эпизода (измеренная причина 4 "
                       "браков из одного запроса, см. докстринг модуля)")
    return bad


def load(video_dir, strict=True):
    """Паспорт эпизода или None, если файла нет.

    `strict=True` (дефолт) — сломанный файл бросает WorldCardError. Снимать
    `strict` можно только в инструментах-отчётах, которые ничего не решают:
    в пути отбора сломанный паспорт обязан остановить работу, а не тихо
    выключить защиту.
    """
    p = path(video_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            card = json.load(f)
    except Exception as e:
        if strict:
            raise WorldCardError(f"{p}: не читается как JSON ({e})")
        return None
    problems = validate(card)
    if problems:
        msg = f"{p}: паспорт мира сломан:\n  - " + "\n  - ".join(problems)
        if strict:
            raise WorldCardError(msg)
        print("  ВНИМАНИЕ: " + msg)
        return None
    return card


def save(video_dir, card, derived_by="unknown"):
    """Записать паспорт. Валидируется ДО записи: пусть сломанный паспорт
    не существует на диске вовсе, чем существует и роняет каждый прогон."""
    problems = validate(card)
    if problems:
        raise WorldCardError("не записываю сломанный паспорт:\n  - "
                             + "\n  - ".join(problems))
    card = dict(card)
    card.setdefault("derived_by", derived_by)
    card.setdefault("derived_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    p = path(video_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


# --- потребители: только данные, никакой логики сопоставления слов --------
#
# Сопоставление термина с запросом живёт в pipeline_smart.query_mentions_term
# (британские написания, составные слова, список ловушек вроде `password`
# внутри `sword`). Второй копии этого правила здесь НЕТ СОЗНАТЕЛЬНО: в
# CLAUDE.md уже записано, чем стоила вторая копия словаря пайплайн-тегов —
# PHRASE LOCK на целый эпизод. Паспорт отдаёт термины, сопоставляет их тот,
# у кого живёт сопоставление.

def era_anchors(card, fallback=()):
    """Слова, которыми якорится запрос к стоку. Паспорт эпизода сильнее
    канала, канал сильнее зашитого дефолта."""
    if card:
        terms = _as_terms(card.get("era_anchor_terms"))
        if terms:
            return terms
    return _as_terms(fallback)


def forbidden_classes(card, fallback=()):
    """Что в кадре этого эпизода — брак. Уходит в вопрос приёмки кадра."""
    if card:
        terms = _as_terms(card.get("must_not_show"))
        if terms:
            return terms
    return _as_terms(fallback)


def culture_include(card):
    return _as_terms(((card or {}).get("culture") or {}).get("include"))


def culture_exclude(card, fallback=()):
    if card:
        terms = _as_terms(((card or {}).get("culture") or {}).get("exclude"))
        if terms:
            return terms
    return _as_terms(fallback)


def era_window(card):
    """(from, to) знаковыми годами или None. Совместимо с окном паспорта
    музейного предмета — сравнивается той же museum_sources.era_overlaps()."""
    era = (card or {}).get("era")
    if not isinstance(era, dict):
        return None
    a, b = era.get("from"), era.get("to")
    if isinstance(a, int) and isinstance(b, int):
        return (a, b)
    return None


def judge_setting(card):
    """Мир эпизода ОДНОЙ строкой для судьи кадров: регистр, окно эпохи и
    культуры «включить», только из паспорта. Нечего сказать — None (судья
    спрашивается как раньше). Списки «чужое» и «запрещено» сюда не входят
    сознательно: длинный список запретов по замеру сжимал оценки судьи к
    единице (см. shot_judge.question)."""
    if not card:
        return None
    parts = []
    reg = card.get("register")
    if isinstance(reg, str) and reg.strip():
        parts.append(reg.strip())
    w = era_window(card)
    if w:
        def y(v):
            return f"{-v} BC" if v < 0 else f"{v} AD"
        parts.append(f"{y(w[0])}-{y(w[1])}")
    parts.extend(culture_include(card))
    return ", ".join(parts) or None


def world_to_check(card):
    """Мир, который имеет смысл проверять НА КАДРЕ: окно эпохи или культуры.
    У научного, абстрактного, современного эпизода без них вопрос «мог ли
    предмет существовать в этом мире» бессмыслен и даёт случайные отказы —
    тогда None, и вопросы про мир не задаются."""
    if not card or not (era_window(card) or culture_include(card) or culture_exclude(card)):
        return None
    return judge_setting(card)


def claims_setting(card):
    """Мир для проверки кадра по утверждениям: world_to_check плюс короткий
    список культур «исключить». У паспорта эпизода 94 культура «включить»
    пустая, и строка мира была «historical, 1300 AD-1500 AD» — ни слова о
    том, ЧЬЯ культура; марокканская тбурида проходила вопрос «мог ли
    главный предмет существовать в этом мире». Сетке список не передаётся
    (judge_setting): там длинный список запретов по замеру сжимал оценки."""
    base = world_to_check(card)
    if base is None:
        return None
    exc = culture_exclude(card)
    return f"{base}; not: {', '.join(exc)}" if exc else base


def is_historical(card):
    """Эпизод про прошлое: 3D, мультфильм и инфографика там — брак. У
    научного или абстрактного эпизода рендер бывает единственным
    изображением предмета, и бракуют его только по смыслу."""
    return bool(card) and card.get("register") in ("historical", "mixed")


def expected_subjects(card):
    """Предметы и сцены, которых эпизод реально требует, короткими
    английскими фразами — читатель: generic_fallback_queries_effective()
    в pipeline_smart.py (см. её докстринг: живой пробный прогон 17.09
    поймал GENERIC_FALLBACKS, третий такой же хардкод, что и
    _QUERY_ERA_ANCHORS_DEFAULT — только в другом месте и с другим
    следствием: он уходит НАПРЯМУЮ в музейный поиск, в обход брифа)."""
    return _as_terms((card or {}).get("expected_subjects"))


# СОВРЕМЕННАЯ ВЕЩЬ В ИСТОРИЧЕСКОМ ЭПИЗОДЕ — ЗАКОННА, И ЭТО ИЗМЕРЕНО.
# В опубликованном эпизоде 01 слот на фразу «вспомни, сколько весит пакет
# молока» ТРЕБУЕТ современный пакет молока. Поэтому поле
# `modern_props_allowed` в паспорте есть (его спрашивает промпт и заполняет
# мозг), но ЧИТАТЕЛЯ у него пока нет: единственный законный читатель —
# приёмка кадра, которая ещё не написана. Заводить accessor заранее значило
# бы своими руками создать седьмой случай «слой есть, и его никто не зовёт»
# — тот самый класс, которым этот репозиторий горел шесть раз. Accessor
# появится вместе с приёмкой, в том же коммите.

def describe(card):
    """Одна строка для логов и отчётов: каким миром собран этот эпизод.
    Без неё ответ на вопрос «почему кадр приняли» жил бы только в голове."""
    if not card:
        return "паспорт мира не задан (поведение как раньше)"
    w = era_window(card)
    era = "—"
    if w:
        def y(v):
            return f"{abs(v)} {'до н.э.' if v < 0 else 'н.э.'}"
        era = f"{y(w[0])}..{y(w[1])}"
    inc = ", ".join(culture_include(card)) or "—"
    return (f"регистр {card.get('register')}, эпоха {era}, культуры {inc}, "
            f"якорей {len(era_anchors(card))}, запретов "
            f"{len(forbidden_classes(card))}")


# --- вопрос для мозга ------------------------------------------------------

PROMPT_VERSION = 1

_PROMPT = """Ты — редактор визуального ряда документального ролика. Прочитай сценарий и опиши МИР этого эпизода: что в кадре будет уместно, а что станет ошибкой.

Ответь ОДНИМ объектом JSON и ничем больше. Схема:

{{
  "schema_version": 1,
  "register": "historical" | "modern" | "abstract" | "scientific" | "mixed",
  "era": {{"from": <целое>, "to": <целое>}} | null,
  "culture": {{"include": [строки], "exclude": [строки]}},
  "must_not_show": [строки],
  "expected_subjects": [строки],
  "era_anchor_terms": [строки],
  "modern_props_allowed": true | false,
  "notes": строка
}}

Правила:
1. "era" — ГОДЫ ЦЕЛЫМИ ЧИСЛАМИ со знаком: -40000 значит 40000 год до н.э., 1415 значит 1415 год н.э. Окно берётся с запасом на весь материал эпизода, а не на одно событие. Если действие происходит сегодня или у эпизода нет привязки ко времени — null.
2. "culture.include" — откуда предметы и люди в кадре уместны; "culture.exclude" — какие культуры в этом эпизоде будут ошибкой, даже если предмет похож (например похожий по форме клинок другой культуры).
3. "must_not_show" — короткие английские названия того, что в кадре этого эпизода недопустимо. Пиши КЛАССЫ, которые видно на картинке: современные люди, современная одежда, автомобили, смартфоны, логотипы брендов, спортивный зал, витрина музея с посетителями, костюмированный фестиваль, реконструкция с публикой. Для эпизода про сегодняшний день сюда идёт обратное: исторические доспехи, замки, рукописи.
4. "expected_subjects" — короткие английские названия предметов и сцен, которых эпизод реально требует, по тексту. Это подсказка поиску, а не полный список.
5. "era_anchor_terms" — английские слова, которые ПРИВЯЗЫВАЮТ поисковый запрос к миру эпизода. Первое слово будет подставляться в запрос как основной якорь, поэтому оно должно быть самым надёжным. Для эпизода без эпохи (психология, современность) это слова регистра, а не эпохи.
6. "modern_props_allowed" — true, если по тексту эпизод сам использует современные вещи как сравнение или пример (например «весит как пакет молока»). Тогда современная вещь в кадре законна там, где её прямо просит описание кадра.
7. Ничего не выдумывай про мир, чего нет в тексте. Список пустой — так и оставь пустым.

Ниша канала (справка, текст сценария важнее): {niche}

СЦЕНАРИЙ:
{script}
"""


def prompt_for_script(script_text, niche="не указана"):
    """Готовый вопрос для любого мозга: сессии, Gemini, локальной модели.

    Мозг здесь не выбирается СОЗНАТЕЛЬНО — см. докстринг модуля: словарная
    эвристика вместо модели была бы тем же хардкодом, от которого паспорт и
    заводится.
    """
    return _PROMPT.format(niche=niche, script=script_text)


def parse_answer(text):
    """Вытащить объект JSON из ответа мозга. Модель почти всегда обрамляет
    JSON текстом или ```-блоком, и требовать чистый ответ значило бы терять
    годные ответы на форматировании (тот же урок, что у parse_answer в
    shot_brief_director: разбор обязан быть терпимым к обёртке)."""
    if not text:
        raise WorldCardError("пустой ответ мозга")
    s = text.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise WorldCardError("в ответе нет объекта JSON")
    try:
        return json.loads(s[i:j + 1])
    except Exception as e:
        raise WorldCardError(f"объект JSON не разбирается ({e})")


# АВТОПАСПОРТ (решение владельца 24.09): один вызов текстовой модели шлюза по
# всему сценарию. Паспорт решает эпоху и культуры для ВСЕГО отбора (музейный
# фильтр, проверка мира на кадре, якоря запросов), поэтому:
#   * существующий паспорт без пометки "auto:" считается ручным и не
#     перезаписывается никогда;
#   * свой (auto:) пересобирается, только когда сценарий изменился;
#   * ответ, не прошедший validate(), не записывается — остаётся прежнее
#     состояние, и об этом говорится вслух.
#
# Модель — по замеру 24.09 на шести эпизодах с ручными паспортами (02, 90-94):
# Gemini 3.7 Flash разобран 6/6 и назвал окна, которые ВЕРНЕЕ ручных там, где
# сценарий сам выходит в современность (Египет: -3100..2026, «mixed» — в тексте
# раскопки Картера 1922 и томограф; Аполлон: 1960..2026 — в тексте смартфон и
# дата-центр). DeepSeek v4 Flash с рассуждением срывался на длинных сценариях,
# без рассуждения дал Аполлону 1955..1980 — смартфон в кадре стал бы «чужим
# миром». Qwen 3.7 Max закрыл Египет 1922 годом (томограф за окном). Gemini 3.1
# Pro — те же ответы, что 3.7 Flash, втрое дороже.
#
# 24.09 вечером шлюз перестал обслуживать 3.7 Flash: на вызов он отвечает
# ТЕКСТОМ «Gemini 3.5 Flash is no longer available», и рендер печатал бы
# «паспорт не разобран» на каждом новом эпизоде. Перезамер тех же шести
# паспортов: Gemini 3.1 Pro разобран 6/6 (94: 1300..1500 ровно как ручной;
# Аполлон 1960..2024, Египет -3000..2024 — те же «сценарий выходит в
# современность»), 3.6 Flash — 5/6 (один 503), Kimi K3 и GLM-5.3 срываются
# в рассуждение на длинных сценариях, DeepSeek снова закрыл Аполлона 1975
# годом. Паспорт — один кэшируемый вызов на эпизод, поэтому цена Pro здесь
# не решает. Модели — по порядку: следующая спрашивается, только если
# предыдущая не дала годного паспорта.
AUTO_PREFIX = "auto:"
AUTO_MODELS = ("ag/gemini-3.1-pro-low", "ag/gemini-3.6-flash-high")
AUTO_MODEL = AUTO_MODELS[0]


def _script_digest(text):
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _narration_digest(script_path):
    """Отпечаток ТОЛЬКО озвучиваемого текста (секции HOOK/BLOCK/FINAL без
    служебных тегов). Паспорт описывает мир рассказа; правка анализа
    конкурентов, вариантов названия или проставленные режиссёром [shot:]
    мир не меняют, а перегенерация паспорта стоила бы вызова модели и —
    через строку мира в подписи плана — перепокупки всех спецификаций и
    переотбора платных слотов. None — сценарий не разбирается."""
    try:
        import script_parser
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            blocks = script_parser.parse_blocks(script_path)
    except Exception:  # noqa: BLE001 — не разобрался: откат на весь текст
        return None
    return _script_digest("\n".join(f"{b.get('section')}|{b.get('text')}" for b in blocks))


#: Поля паспорта, которые решают отбор. Их отпечаток входит в подпись
#: отбора: правка эпохи или культур меняет, кого пропустит фильтр музеев и
#: проверка мира, даже если строка мира для планировщика не изменилась.
WORLD_FIELDS = ("register", "era", "culture", "must_not_show",
                "expected_subjects", "era_anchor_terms")


def world_digest(card):
    """Отпечаток полей мира паспорта ('' — паспорта нет)."""
    if not card:
        return ""
    import hashlib
    body = json.dumps({k: card.get(k) for k in WORLD_FIELDS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _raw(video_dir):
    try:
        with open(path(video_dir), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 — нет или битый: решает load()
        return None


def is_manual(video_dir):
    """Паспорт на диске есть и написан не моделью (человек, сессия, файл)."""
    raw = _raw(video_dir)
    return bool(raw) and not str(raw.get("derived_by") or "").startswith(AUTO_PREFIX)


def generate(video_dir, gateway, model=None, niche="не указана"):
    """Паспорт эпизода от модели, если его нет или свой устарел. Возвращает
    (паспорт | None, что сделано): "manual" — ручной, не трогали;
    "fresh" — свой и сценарий не менялся; "made" — записан новый;
    "failed: ..." — модель не дала годного паспорта, на диске прежнее."""
    sp = os.path.join(video_dir, "script.txt")
    if not os.path.exists(sp):
        return None, "failed: нет script.txt"
    with open(sp, encoding="utf-8") as f:
        script = f.read()
    raw = _raw(video_dir)
    if raw is not None and is_manual(video_dir):
        return load(video_dir, strict=False), "manual"
    digest = _narration_digest(sp) or _script_digest(script)
    if raw is not None and not validate(raw):
        if raw.get("script_sha") == digest:
            return raw, "fresh"
        if raw.get("script_sha") == _script_digest(script):
            # Паспорт записан прежней схемой (отпечаток всего файла), и файл
            # с тех пор не менялся: мир тот же — переводим отпечаток на
            # озвучку без вызова модели.
            raw["script_sha"] = digest
            save(video_dir, raw, derived_by=raw.get("derived_by"))
            return raw, "fresh"
    errors = []
    for m in ((model,) if model else AUTO_MODELS):
        try:
            text, _u, _p = gateway.chat(m, [{"type": "text", "text": prompt_for_script(script, niche)}],
                                        6000, 2500 + len(script) // 2)
            card = parse_answer(text)
            card["schema_version"] = SCHEMA_VERSION
            card["script_sha"] = digest
            card.pop("derived_by", None)
            card.pop("derived_at", None)
            save(video_dir, card, derived_by=AUTO_PREFIX + m)
            return card, "made"
        except Exception as e:  # noqa: BLE001 — сбой модели/формата: следующая модель
            errors.append(f"{m}: {str(e)[:160]}")
    return (load(video_dir, strict=False) if raw else None), "failed: " + " | ".join(errors)


def main(argv=None):
    """CLI: собрать вопрос по сценарию (--prompt) или проверить готовый
    паспорт (--check). Записывает паспорт из файла-ответа (--answer) — тот
    же харнесс «вопрос -> ответ файлом -> запись», что у брифов кадра."""
    import argparse
    ap = argparse.ArgumentParser(description="Паспорт мира эпизода")
    ap.add_argument("video_dir")
    ap.add_argument("--prompt", action="store_true",
                    help="напечатать вопрос для мозга по script.txt")
    ap.add_argument("--answer", metavar="FILE",
                    help="файл с ответом мозга -> записать паспорт")
    ap.add_argument("--check", action="store_true",
                    help="проверить паспорт на диске")
    ap.add_argument("--niche", default="не указана")
    a = ap.parse_args(argv)

    if a.prompt:
        sp = os.path.join(a.video_dir, "script.txt")
        if not os.path.exists(sp):
            print(f"нет {sp}")
            return 1
        with open(sp, encoding="utf-8") as f:
            print(prompt_for_script(f.read(), niche=a.niche))
        return 0
    if a.answer:
        with open(a.answer, encoding="utf-8") as f:
            card = parse_answer(f.read())
        p = save(a.video_dir, card, derived_by="file")
        print(f"записан {p}\n  {describe(card)}")
        return 0
    if a.check:
        try:
            card = load(a.video_dir)
        except WorldCardError as e:
            print(f"ПАСПОРТ СЛОМАН\n{e}")
            return 2
        if card is None:
            print("паспорта нет — потребители работают как раньше")
            return 0
        print(f"паспорт годен\n  {describe(card)}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
