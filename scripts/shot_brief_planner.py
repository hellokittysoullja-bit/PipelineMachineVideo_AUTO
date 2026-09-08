# -*- coding: utf-8 -*-
"""Локальный режиссёр кадра (Контур A): фраза сценария -> структурированное ТЗ.

ЗАЧЕМ. Сегодня к стоку улетает КОРОТКИЙ АНГЛИЙСКИЙ КЛЮЧЕВОЙ ЗАПРОС —
либо авторский из `=== PEXELS QUERIES ===`, либо из словаря, либо
neighbor-inherit. Дальше весь подбор — это выбор лучшего из уже собранного
пула. Замер на опубликованном эпизоде 01_ves-mecha показал прямое
следствие: один запрос обслуживал по 3-4 блока сразу, и `greatsword
warrior fight` достался четырём слотам, где все четыре кадра оказались
браком. Никакое ранжирование внутри такого пула это не чинит — судья,
которому подали мусорный top-K, вынесет мусорное решение.

Второе, чего не может эмбеддинг ни при каком объёме фразы: отличить
БУКВАЛЬНОЕ от ФИГУРАЛЬНОГО. «Пятнадцать килограммов» — фраза про МИФ о
весе меча, объекта «пятнадцатикилограммовый меч» не существует;
буквальный поиск обязан промахнуться по построению (в реальном эпизоде
он привёл современную кухню с хлопьями). Это задокументированный режим
отказа contrastive-моделей (ARO/SugarCrepe/Winoground — модель ведёт себя
как «мешок слов»), а не недоработка порога.

АРХИТЕКТУРА — ОТДЕЛЬНЫЙ ПРОХОД, НЕ ИНЛАЙН. Скрипт запускается ДО рендера
(тот же паттерн, что уже работает у `speech_planner.py`), грузит модель,
пишет `media_plan/shot_briefs.json` и ВЫХОДИТ, освобождая всю память.
`pipeline_smart.py` только читает готовый JSON. Это принципиально:
прошлая попытка завести локальный VLM в этом проекте была отклонена
именно из-за резидентной памяти (пик рендера уже 9-10 ГБ из 15, а
холодный старт одной только SigLIP2-so400m — минуты). Отдельный проход
снимает конфликт полностью: во время рендера этой модели в памяти нет.

ДИСЦИПЛИНА БЕЗОПАСНОСТИ (требование «только upgrade, без минусов»):
  * Запросы ДОБАВЛЯЮТСЯ в пул секции, а не заменяют авторский. Новый
    механизм может добавить хорошего кандидата, но не может отнять того,
    кто нашёлся бы и без него.
  * `must_not_contain` генерируется и записывается, но в contrastive-вето
    НЕ подаётся, пока его вменяемость не измерена: галлюцинированный
    негатив (например «sword» на канале про мечи) забраковал бы все
    правильные кадры разом. Тот же shadow-принцип, что у
    VISUAL_DIRECTOR_MODE.
  * Строгая валидация КАЖДОГО поля. Любое нарушение -> слот честно
    помечается `"valid": false` и не влияет на подбор вообще.
  * Детерминизм: temperature=0, фиксированный seed, кэш по SHA-256
    (текст + контекст + версия промпта + имя модели). Повторный прогон
    не платит дважды и даёт тот же результат.
  * Fail-open на всё: нет llama_cpp, нет модели, битый JSON, таймаут ->
    файл просто не появляется/слот невалиден, рендер идёт как раньше.

Запуск:
    .venv/bin/python scripts/shot_brief_planner.py <video_dir> [--limit N]
    .venv/bin/python scripts/shot_brief_planner.py --benchmark
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Путь к модели: из .env (SHOT_BRIEF_MODEL_PATH) либо дефолт рядом с репо.
DEFAULT_MODEL_PATH = "/home/user/models/qwen2.5-3b-instruct-q4_k_m.gguf"

# Версия промпта входит в ключ кэша: переписал промпт -> старые вердикты
# не наследуются молча (тот же урок, что уже усвоен в _arbiter_prompt_
# signature() у VLM-арбитра). Переключается --prompt-version, чтобы
# сравнение версий шло на ОДНИХ И ТЕХ ЖЕ случаях, а не по памяти.
#
# Почему дефолт v3, хотя по ПРОПУСКНОЙ СПОСОБНОСТИ v2 лучше (16 брифов из 20
# против 6): у версий разный характер отказа. Болезнь v2 — списывание
# разобранного примера — даёт ВИДИМЫЙ зрителю дефект (один и тот же кадр в
# трёх разных местах ролика). Болезнь v3 — внутренне противоречивые брифы —
# ловится барьером, и слот просто остаётся на прежнем поведении пайплайна,
# то есть отказ безопасный. Выбран отказ в безопасную сторону.
# Полный разбор с числами — docs/quality/DIRECTOR_LOCAL_LLM.md.
PROMPT_VERSION = 3

# Потолки валидации. Всё, что за ними, — признак того, что модель поехала,
# и такой слот честнее пометить невалидным, чем пустить в подбор.
MAX_QUERIES = 5
MIN_QUERIES = 1
MAX_QUERY_CHARS = 70
MAX_NEGATIVES = 12
MAX_NEGATIVE_CHARS = 60
MAX_SUBJECT_CHARS = 200

VALID_READINGS = ("literal", "figurative")

# Окно эпохи канала. Промпт объявляет модели «roughly 1000-1500 AD»; бриф,
# чьё окно с этим НЕ ПЕРЕСЕКАЕТСЯ ВООБЩЕ, по построению не про материал
# этого канала. Границы взяты с запасом в столетие в обе стороны — задача
# не судить историческую точность модели, а поймать явный промах эпохи.
#
# Проверено на реальных данных, не введено на всякий случай: по 40 брифам
# двух прогонов правило срабатывает РОВНО ОДИН раз — на ep01_133 («человек
# в доспехе — это человек в термосе»), где модель выдала era 1700-1800 и
# subject «soldier in armor and a thermos», то есть поставила на экран
# риторический предмет метафоры. Ноль ложных срабатываний на остальных 39.
#
# Читается из channel_profile.json (`era_from`/`era_to`), если канал их
# объявил, — те же ворота настройки под нишу, что у content_alt_blocklist
# (ЧАСТЬ 24 CLAUDE.md). Дефолт — окно текущего канала.
CHANNEL_ERA_DEFAULT = (900, 1600)


def channel_era_window():
    try:
        with open(os.path.join(REPO, "channel_profile.json"), encoding="utf-8") as f:
            prof = json.load(f)
        a, b = int(prof["era_from"]), int(prof["era_to"])
        return (a, b) if a <= b else (b, a)
    except Exception:
        return CHANNEL_ERA_DEFAULT

SYSTEM_PROMPT = (
    "You are a shot director for a Russian-language documentary channel about "
    "EUROPEAN medieval history (roughly 1000-1500 AD). You are given one line of "
    "narration. You decide what appears on screen while that line is spoken.\n"
    "\n"
    "THE MOST IMPORTANT RULE - literal vs figurative:\n"
    "Some lines name a real filmable object ('swords lie in museums'). That is literal.\n"
    "Other lines use a number or an everyday image RHETORICALLY, and the named object "
    "is not the subject at all. Examples:\n"
    "  'Fifteen kilograms.' -> this line is about a MYTH about sword weight. "
    "A fifteen-kilogram sword does not exist. Do NOT search for scales or weights. "
    "Show a real medieval sword.\n"
    "  'remember how much a carton of milk weighs in your hand' -> milk is only a unit "
    "of comparison. Do NOT show milk, dairy, kitchens or babies. Show a sword being held.\n"
    "  'a man in armour is a man in a thermos' -> a metaphor about heat. "
    "Do NOT show a thermos. Show a knight in full plate, sweat, closed helmet.\n"
    "For figurative lines, choose what a documentary editor would actually cut to: "
    "the REAL subject of the sentence, never the words in it.\n"
    "\n"
    "Everything you output must be in ENGLISH (the stock archives are indexed in "
    "English), even though the narration is Russian.\n"
    "Never propose non-European material (no katana, samurai, kimono, Chinese jian, "
    "Korean or Japanese costume) - this channel is strictly European.\n"
    "Never propose modern intrusions (sport fencing, referees, scoreboards, "
    "spectators in modern clothes, cars, phones, gyms).\n"
)

USER_TEMPLATE = """NARRATION LINE (Russian):
{text}

CONTEXT (line before / line after, for meaning only - do not illustrate these):
before: {prev}
after: {next}

Answer with STRICT JSON only, no explanation outside the JSON, this exact schema:
{{
  "reading": "literal" or "figurative",
  "why": "<one short sentence in Russian: what this line is really about>",
  "subject": "<short English noun phrase: the main thing on screen>",
  "setting": "<short English phrase: where/lighting/atmosphere>",
  "era_from": <year, integer>,
  "era_to": <year, integer>,
  "queries_en": ["<stock search query>", "<another>", "<another>"],
  "must_not_contain": ["<thing that must not appear>", "<another>"],
  "confidence": <number between 0 and 1>
}}"""

# --- Промпт v2 ---
# Три системных сбоя v1, найденные прогоном по 20 реальным случаям, а не
# предположением:
#  1. Модель писала `subject` ПО-РУССКИ (3 отказа валидации из первых 7).
#     Вероятная причина — я сам просил `why` на русском, и язык подтягивался
#     во все остальные поля. В v2 ВЕСЬ вывод английский; читаемость `why`
#     для человека от этого не страдает.
#  2. На фигуральных фразах модель тянула риторический предмет в запрос
#     («milk carton weight» на фразе, где молоко — единица сравнения, а не
#     предмет). В v2 это отдельное явное правило, а не следствие общего.
#  3. Три запроса выходили синонимами одного («medieval sword weight»,
#     «sword weight comparison», «historical sword weight») — пул от такого
#     шире не становится. В v2 каждому запросу назначен СВОЙ ракурс.
# Плюс полноценный разобранный пример: модели этого размера следуют
# образцу заметно надёжнее, чем списку абстрактных требований.
SYSTEM_PROMPT_V2 = (
    "You are a shot director for a Russian-language documentary about EUROPEAN "
    "medieval history (roughly 1000-1500 AD). You get one line of narration and "
    "decide what appears on screen while it is spoken.\n"
    "\n"
    "RULE 1 - EVERY value you output is in ENGLISH. The narration is Russian, your "
    "answer is not. A Russian word anywhere in the JSON makes the answer unusable.\n"
    "\n"
    "RULE 2 - literal vs figurative. Decide what the line is REALLY about.\n"
    "A line is figurative when the object it names is not the subject: a number used "
    "to quote a myth, or an everyday thing used only as a unit of comparison, or an "
    "image used as a metaphor.\n"
    "When reading is figurative, the rhetorical object is FORBIDDEN in 'subject' and "
    "in every query. Put it in 'must_not_contain' instead.\n"
    "\n"
    "RULE 3 - the three queries must be three DIFFERENT camera angles on the same "
    "subject, not three wordings of one query:\n"
    "  query 1: a close detail (macro, texture, hands)\n"
    "  query 2: the whole object or person in its place (wider)\n"
    "  query 3: a related action or the setting itself\n"
    "Queries are what you type into a stock footage site: concrete visible things, "
    "no abstract nouns like 'weight', 'myth', 'comparison', 'history'.\n"
    "\n"
    "RULE 4 - this channel is strictly European. Never propose katana, samurai, "
    "kimono, Chinese jian, Korean or Japanese costume. Never propose modern "
    "intrusions: sport fencing, referees, scoreboards, spectators in modern clothes, "
    "cars, phones, gyms, kitchens.\n"
    "\n"
    "WORKED EXAMPLE.\n"
    "Narration: 'Пятнадцать килограммов.' (context: the next line says cinema and "
    "school books repeat this).\n"
    "Correct answer:\n"
    "{\n"
    '  "reading": "figurative",\n'
    '  "why": "the line quotes a myth about sword weight; no 15 kg sword exists",\n'
    '  "subject": "medieval european longsword",\n'
    '  "setting": "dark museum hall, single side light on the blade",\n'
    '  "era_from": 1100, "era_to": 1500,\n'
    '  "queries_en": ["medieval sword blade macro detail", '
    '"knight longsword in museum display case", "armourer holding longsword in forge"],\n'
    '  "must_not_contain": ["kitchen scales", "weighing scale", "dumbbell", '
    '"modern kitchen", "carton of milk", "printed numbers"],\n'
    '  "confidence": 0.9\n'
    "}\n"
    "Note how the number and the scales are absent from the queries and present in "
    "must_not_contain, and how the three queries are macro / wide / action.\n"
)

USER_TEMPLATE_V2 = """NARRATION LINE (Russian):
{text}

CONTEXT (only to understand the meaning - never illustrate these lines):
before: {prev}
after: {next}

Answer with STRICT JSON only, all values in English, this exact schema:
{{
  "reading": "literal" or "figurative",
  "why": "<one short English sentence: what this line is really about>",
  "subject": "<short English noun phrase: the main thing on screen>",
  "setting": "<short English phrase: place, light, atmosphere>",
  "era_from": <year, integer>,
  "era_to": <year, integer>,
  "queries_en": ["<close detail>", "<wider shot>", "<action or setting>"],
  "must_not_contain": ["<thing that must not appear>", "<another>"],
  "confidence": <number between 0 and 1>
}}"""

# --- Промпт v3 ---
# Два сбоя v2, найденные ЖИВЫМ замером по Pexels (20 реальных слотов
# опубликованного эпизода, docs/quality/director_impact_v2.json + 4
# контактных листа), а не рассуждением:
#
#  1. КОПИРОВАНИЕ ПРИМЕРА. Разобранный пример v2 был про меч — то есть про
#     тот же предмет, что и почти каждый слот этого эпизода. Модель этого
#     размера следует образцу буквально: запрос «medieval sword blade macro
#     detail» из примера ушёл в 9 слотов дословно, и в трёх разных слотах
#     победил один и тот же кадр. Это ровно та болезнь, которую Контур A
#     должен был лечить (один запрос на четыре слота в базовой линии), —
#     я внёс её обратно собственным примером.
#     Лечение: пример остаётся (без него v1 дал 20% валидного JSON), но
#     берётся из ДРУГОЙ темы того же канала — голод и цена хлеба. Списать
#     его в слот про меч физически бессмысленно, а структуру «фигуральное
#     чтение + три ракурса» он показывает ту же.
#  2. ФИГУРАЛЬНОЕ ЧТЕНИЕ ПРОВАЛИЛОСЬ НА ДВУХ РЕАЛЬНЫХ ФРАЗАХ. «Вспомни,
#     сколько весит пакет молока» -> запросы про пакеты молока; «цвайхендер
#     — это потолок» -> `dark room ceiling macro detail`. Правило v2 («в
#     фигуральном чтении риторический предмет запрещён») было СФОРМУЛИРОВАНО
#     верно, но требовало от модели самой сообразить, что предмет
#     риторический. Лечение — не ещё одно требование, а механическая
#     процедура из трёх вопросов, которую модель выполняет ПО ПОРЯДКУ, и
#     отдельная строка про слова-пределы («потолок», «вершина», «дно»),
#     на которых v2 промахнулась буквально.
SYSTEM_PROMPT_V3 = (
    "You are a shot director for a Russian-language documentary about EUROPEAN "
    "medieval history (roughly 1000-1500 AD). You get one line of narration and "
    "decide what appears on screen while it is spoken.\n"
    "\n"
    "RULE 1 - EVERY value you output is in ENGLISH. The narration is Russian, your "
    "answer is not. A Russian word anywhere in the JSON makes the answer unusable.\n"
    "\n"
    "RULE 2 - before anything else, run these three questions IN ORDER:\n"
    "  Q1. Which everyday object, number or place does the line NAME?\n"
    "  Q2. Is the line ABOUT that thing, or is it using that thing to explain "
    "something else?\n"
    "  Q3. If it explains something else, then that thing is the RHETORICAL OBJECT. "
    "The reading is 'figurative'. The rhetorical object is BANNED from 'subject' and "
    "from every query, and you must list it in 'must_not_contain'. What goes on "
    "screen is the thing being explained.\n"
    "Worked through: 'remember how much a carton of milk weighs' - Q1: a carton of "
    "milk. Q2: the line is about how light a sword is, milk is only the unit of "
    "comparison. Q3: milk, dairy, kitchens and shops are banned; on screen is a hand "
    "holding a sword.\n"
    "Words that name a LIMIT are always figurative: 'потолок' (ceiling), 'вершина' "
    "(peak), 'дно' (bottom), 'планка' (bar). They mean the extreme of a range. Show "
    "the object that reaches that extreme - never architecture, never a real ceiling.\n"
    "A number quoted as a myth ('fifteen kilograms') is figurative too: show the real "
    "object the myth is about, never scales, weights or printed digits.\n"
    "\n"
    "RULE 3 - the three queries must be three DIFFERENT camera angles on the same "
    "subject, not three wordings of one query:\n"
    "  query 1: a close detail (macro, texture, hands)\n"
    "  query 2: the whole object or person in its place (wider)\n"
    "  query 3: a related action or the setting itself\n"
    "Queries are what you type into a stock footage site: concrete visible things, "
    "no abstract nouns like 'weight', 'myth', 'comparison', 'history'.\n"
    "\n"
    "RULE 4 - the queries must fit THIS line and no other. If a query would suit any "
    "random line of this documentary equally well, it is too generic: make it name "
    "something specific to what is happening in this exact line.\n"
    "\n"
    "RULE 5 - this channel is strictly European. Never propose katana, samurai, "
    "kimono, Chinese jian, Korean or Japanese costume. Never propose modern "
    "intrusions: sport fencing, referees, scoreboards, spectators in modern clothes, "
    "cars, phones, gyms, kitchens.\n"
    "\n"
    "WORKED EXAMPLE (a different subject - do NOT reuse its words).\n"
    "Narration: 'Зимой хлеб стоил как конь.' (context: the next line is about the "
    "famine of 1315).\n"
    "Q1 names a horse. Q2: the line is about the price of bread in a famine, the "
    "horse is only a unit of price. Q3: the horse is the rhetorical object.\n"
    "Correct answer:\n"
    "{\n"
    '  "reading": "figurative",\n'
    '  "why": "the line is about famine bread prices; the horse is only a unit of '
    'comparison",\n'
    '  "subject": "starving townspeople and a bare bread stall",\n'
    '  "setting": "grey winter street of a medieval town, thin light",\n'
    '  "era_from": 1300, "era_to": 1350,\n'
    '  "queries_en": ["dark rye bread loaf on rough wood macro", '
    '"empty medieval market stall in winter", "poor peasants queuing in a medieval '
    'town street"],\n'
    '  "must_not_contain": ["horse", "horse market", "stable", "modern bakery", '
    '"supermarket shelf", "price tag"],\n'
    '  "confidence": 0.85\n'
    "}\n"
    "Note how the horse is absent from every query and present in must_not_contain, "
    "and how the three queries are macro / wide / people-in-action. Your line is "
    "about something else entirely - use its own words, not these.\n"
)

USER_TEMPLATE_V3 = USER_TEMPLATE_V2

PROMPTS = {
    1: (SYSTEM_PROMPT, USER_TEMPLATE),
    2: (SYSTEM_PROMPT_V2, USER_TEMPLATE_V2),
    3: (SYSTEM_PROMPT_V3, USER_TEMPLATE_V3),
}

# Запросы, дословно взятые из разобранного примера промпта. Ни один из них
# не может быть выводом ПРО КОНКРЕТНУЮ ФРАЗУ — это списанный образец, и
# замер v2 показал цену такого списывания: один запрос в девяти слотах и
# один и тот же кадр-победитель в трёх разных местах ролика.
#
# Гвард дословный и намеренно узкий: отбрасывается только запрос, ПОБУКВЕННО
# совпадающий с образцовым (после нормализации пробелов и регистра).
# Похожий по смыслу запрос, сформулированный моделью самостоятельно,
# остаётся — иначе гвард начал бы резать законную работу. Если после
# отброса не осталось ни одного запроса, бриф честно признаётся невалидным:
# слот тогда остаётся на прежнем поведении пайплайна (авторский запрос /
# словарь тем), то есть строго не хуже, чем сегодня.
EXAMPLE_QUERIES = frozenset(
    q.lower()
    for q in (
        # v2 — тот самый пример про меч, который и утёк в девять слотов.
        "medieval sword blade macro detail",
        "knight longsword in museum display case",
        "armourer holding longsword in forge",
        # v3 — пример про хлеб; блокируется заранее, до того как утечёт.
        "dark rye bread loaf on rough wood macro",
        "empty medieval market stall in winter",
        "poor peasants queuing in a medieval town street",
    )
)


def _cache_key(text, prev, next_, model_name):
    raw = json.dumps({"t": text, "p": prev, "n": next_, "v": PROMPT_VERSION,
                       "m": model_name}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _has_cyrillic(s):
    return bool(re.search(r"[а-яА-ЯёЁ]", s or ""))


# Служебные слова: их совпадение между subject и запретами ничего не значит.
# Список намеренно короткий — только грамматика и слова кадра. Чем он длиннее,
# тем больше настоящих самозапретов проскочит незамеченными.
_FUNCTION_WORDS = frozenset("""
a an the of in on at with and or for from to into is are be being
its his her their this that these those as by over under near
close up macro wide shot view detail scene
""".split())

_WORD_RE = re.compile(r"[a-z]+")


def _content_words(s):
    """Значимые слова строки — без грамматики и без слов про крупность."""
    return frozenset(_WORD_RE.findall((s or "").lower())) - _FUNCTION_WORDS


def validate_brief(raw):
    """(brief|None, причина_отказа|None) — строгая проверка КАЖДОГО поля.

    Возвращает None на любом нарушении: слот тогда честно считается
    невалидным и не влияет на подбор вообще. Это дешевле, чем пустить в
    пайплайн поле, которому нельзя доверять, — цена ошибки здесь
    несимметрична (плохой запрос портит кадр, отсутствующий — просто
    оставляет прежнее поведение)."""
    if not isinstance(raw, dict):
        return None, "не JSON-объект"

    reading = raw.get("reading")
    if reading not in VALID_READINGS:
        return None, f"reading={reading!r} не из {VALID_READINGS}"

    queries = raw.get("queries_en")
    if not isinstance(queries, list):
        return None, "queries_en не список"
    clean_q = []
    for q in queries:
        if not isinstance(q, str):
            continue
        q = " ".join(q.split()).strip()
        if not q or len(q) > MAX_QUERY_CHARS:
            continue
        # Кириллица в поисковом запросе — прямой признак того, что модель
        # не выполнила инструкцию про английский; стоковые архивы такой
        # запрос не поймут, и это молча дало бы пустую выдачу.
        if _has_cyrillic(q):
            continue
        # Списанный из промпта образец — не решение про эту фразу
        # (см. EXAMPLE_QUERIES: замер v2, девять слотов на один запрос).
        if q.lower() in EXAMPLE_QUERIES:
            continue
        clean_q.append(q)
    # Дедуп с сохранением порядка (модель любит повторять формулировки).
    seen = set()
    deduped = []
    for q in clean_q:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(q)
    if len(deduped) < MIN_QUERIES:
        return None, "не осталось валидных английских запросов"
    deduped = deduped[:MAX_QUERIES]

    negatives = []
    for n in (raw.get("must_not_contain") or []):
        if not isinstance(n, str):
            continue
        n = " ".join(n.split()).strip()
        if n and len(n) <= MAX_NEGATIVE_CHARS and not _has_cyrillic(n):
            negatives.append(n)
    negatives = negatives[:MAX_NEGATIVES]

    subject = raw.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        return None, "пустой subject"
    subject = " ".join(subject.split())[:MAX_SUBJECT_CHARS]
    if _has_cyrillic(subject):
        return None, "subject не на английском"

    # Самопротиворечие: модель запрещает то, что сама же поставила на экран.
    # Это не придирка к формулировке, а измеренная болезнь: на прогоне v3 по
    # 20 реальным слотам 13 брифов из 20 (65%) внесли в must_not_contain
    # слово из собственного subject — «sword» на канале про мечи. Пусти такой
    # бриф дальше, и его негативы в роли анкеров вето забраковали бы ровно
    # тот предмет, ради которого слот существует.
    # Отклоняется бриф ЦЕЛИКОМ, а не только поле негативов: на худших слотах
    # замера (ep01_005 «пакет молока», ep01_133 «человек в термосе»)
    # самозапрет шёл в паре с запросом, который сам же запрет и нарушал, —
    # то есть он маркирует слот, где модель не поняла задачу, а не отдельную
    # опечатку в одном поле. Слот тогда остаётся на прежнем поведении
    # пайплайна: гарантированный ноль регресса вместо ставки на догадку.
    subject_words = _content_words(subject)
    banned_words = _content_words(" ".join(negatives))
    self_banned = subject_words & banned_words
    if self_banned:
        return None, f"must_not_contain запрещает собственный subject: {sorted(self_banned)}"

    setting = raw.get("setting")
    setting = " ".join(setting.split())[:MAX_SUBJECT_CHARS] if isinstance(setting, str) else ""

    def _year(v):
        try:
            y = int(v)
        except (TypeError, ValueError):
            return None
        # Канал про Средневековье; год вне разумного диапазона — признак
        # галлюцинации, а не полезный сигнал.
        return y if -3000 <= y <= 2100 else None

    era_from, era_to = _year(raw.get("era_from")), _year(raw.get("era_to"))
    if era_from is not None and era_to is not None and era_from > era_to:
        era_from, era_to = era_to, era_from
    # Окно эпохи, не пересекающееся с окном канала (см. CHANNEL_ERA_DEFAULT),
    # — бриф не про материал этого канала. Отсутствующие годы не проверяются:
    # молчание модели здесь не улика.
    if era_from is not None and era_to is not None:
        ch_from, ch_to = channel_era_window()
        if era_to < ch_from or era_from > ch_to:
            return None, (f"эпоха брифа {era_from}-{era_to} не пересекается "
                          f"с эпохой канала {ch_from}-{ch_to}")

    try:
        conf = float(raw.get("confidence"))
    except (TypeError, ValueError):
        conf = None
    if conf is not None:
        conf = max(0.0, min(1.0, conf))

    why = raw.get("why")
    why = " ".join(why.split())[:300] if isinstance(why, str) else ""

    return {
        "reading": reading,
        "why": why,
        "subject": subject,
        "setting": setting,
        "era_from": era_from,
        "era_to": era_to,
        "queries_en": deduped,
        "must_not_contain": negatives,
        "confidence": conf,
    }, None


def _extract_json(text):
    """Первый сбалансированный JSON-объект из ответа модели.

    Модели этого размера любят обрамлять ответ ```json ... ``` или
    добавлять фразу до/после. Ищем по скобкам, а не регуляркой: вложенные
    объекты (era, кавычки внутри строк) регуляркой режутся неверно."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


class BriefGenerator:
    """Обёртка над локальной моделью. Модель грузится ОДИН раз на процесс."""

    def __init__(self, model_path=None, n_ctx=2048, n_threads=None, verbose=False,
                 prompt_version=None):
        self.model_path = model_path or os.environ.get(
            "SHOT_BRIEF_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.model_name = os.path.basename(self.model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or (os.cpu_count() or 4)
        self.verbose = verbose
        self.prompt_version = prompt_version or PROMPT_VERSION
        if self.prompt_version not in PROMPTS:
            raise ValueError(f"нет промпта версии {self.prompt_version}")
        self._llm = None
        self.load_seconds = None

    def load(self):
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama   # локальный импорт: fail-open у вызывающего
        t0 = time.time()
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            # seed фиксирован вместе с temperature=0 ниже: два прогона на
            # одном и том же входе обязаны давать один и тот же бриф,
            # иначе кэш и воспроизводимость эпизода теряют смысл.
            seed=0,
            verbose=self.verbose,
        )
        self.load_seconds = time.time() - t0
        return self._llm

    def generate(self, text, prev="", next_="", max_tokens=400):
        """(brief|None, meta) — meta всегда есть, даже когда brief нет."""
        llm = self.load()
        system_prompt, user_template = PROMPTS[self.prompt_version]
        user = user_template.format(text=text, prev=prev or "-", next=next_ or "-")
        t0 = time.time()
        out = llm.create_chat_completion(
            messages=[{"role": "system", "content": system_prompt},
                       {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t0
        content = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
        usage = out.get("usage") or {}
        meta = {
            "seconds": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "raw": content,
        }
        parsed = _extract_json(content)
        if parsed is None:
            meta["error"] = "ответ не содержит валидного JSON"
            return None, meta
        brief, why_bad = validate_brief(parsed)
        if brief is None:
            meta["error"] = f"валидация: {why_bad}"
            return None, meta
        return brief, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir", nargs="?", help="папка эпизода (videos/NN_...)")
    ap.add_argument("--model", default=None, help="путь к GGUF")
    ap.add_argument("--limit", type=int, default=None, help="только первые N блоков")
    ap.add_argument("--benchmark", action="store_true",
                    help="прогон по tests/fixtures/director_benchmark/cases.json")
    ap.add_argument("--out", default=None, help="куда писать результат замера")
    ap.add_argument("--prompt-version", type=int, default=None,
                    choices=sorted(PROMPTS), help="какую версию промпта мерить")
    args = ap.parse_args()

    gen = BriefGenerator(model_path=args.model, prompt_version=args.prompt_version)
    if not os.path.exists(gen.model_path):
        print(f"Модель не найдена: {gen.model_path}")
        return 1

    if args.benchmark:
        return _run_benchmark(gen, args.out)

    if not args.video_dir:
        ap.error("нужен video_dir либо --benchmark")
    return _run_episode(gen, args.video_dir, args.limit)


def _run_benchmark(gen, out_path):
    cases_path = os.path.join(REPO, "tests", "fixtures", "director_benchmark", "cases.json")
    with open(cases_path, encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload["cases"]
    out_path = out_path or os.path.join(REPO, "docs", "quality", "director_benchmark_run.json")

    print(f"Модель: {gen.model_name}, промпт v{gen.prompt_version}")
    print(f"Потоков: {gen.n_threads}, случаев: {len(cases)}")
    rows = []
    t_start = time.time()
    for i, c in enumerate(cases, 1):
        brief, meta = gen.generate(c["text"], c["context_prev"], c["context_next"])
        row = {
            "id": c["id"],
            "difficulty_class": c["difficulty_class"],
            "text": c["text"],
            "baseline_query": c["baseline_query"],
            "baseline_verdict": c["baseline_verdict"],
            "baseline_reject_reason": c.get("baseline_reject_reason"),
            "seconds": meta["seconds"],
            "completion_tokens": meta.get("completion_tokens"),
            "valid": brief is not None,
            "error": meta.get("error"),
            "brief": brief,
        }
        if brief is None:
            row["raw_head"] = (meta.get("raw") or "")[:300]
        rows.append(row)
        status = "OK " if brief else "FAIL"
        qs = ", ".join(brief["queries_en"]) if brief else meta.get("error", "")
        print(f"  [{i:2d}/{len(cases)}] {status} {c['id']} {meta['seconds']:5.1f}с "
              f"({c['difficulty_class']}) {qs[:90]}")
    total = time.time() - t_start

    valid = sum(1 for r in rows if r["valid"])
    result = {
        "model": gen.model_name,
        "model_load_seconds": round(gen.load_seconds or 0, 1),
        "n_threads": gen.n_threads,
        "prompt_version": gen.prompt_version,
        "total_seconds": round(total, 1),
        "avg_seconds_per_case": round(total / max(1, len(rows)), 2),
        "valid_json_rate": round(valid / max(1, len(rows)), 4),
        "cases": rows,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)

    print(f"\nЗагрузка модели: {result['model_load_seconds']}с")
    print(f"Всего: {total:.0f}с, в среднем {result['avg_seconds_per_case']}с на слот")
    print(f"Валидный JSON: {valid}/{len(rows)} ({100*valid/max(1,len(rows)):.0f}%)")
    print(f"Отчёт: {out_path}")
    return 0


def _run_episode(gen, video_dir, limit):
    import script_parser
    script_path = os.path.join(video_dir, "script.txt")
    if not os.path.exists(script_path):
        print(f"Нет сценария: {script_path}")
        return 1
    blocks = script_parser.parse_blocks(script_path)
    if limit:
        blocks = blocks[:limit]

    cache_dir = os.path.join(video_dir, "media_plan", "shot_brief_cache")
    os.makedirs(cache_dir, exist_ok=True)

    briefs = {}
    stats = {"cached": 0, "generated": 0, "invalid": 0}
    t_start = time.time()
    for i, b in enumerate(blocks):
        prev = blocks[i - 1]["text"] if i > 0 else ""
        nxt = blocks[i + 1]["text"] if i + 1 < len(blocks) else ""
        key = _cache_key(b["text"], prev, nxt, gen.model_name)
        cpath = os.path.join(cache_dir, f"{key}.json")
        if os.path.exists(cpath):
            try:
                with open(cpath, encoding="utf-8") as f:
                    briefs[str(i)] = json.load(f)
                stats["cached"] += 1
                continue
            except Exception:
                pass   # битый кэш — просто пересчитаем
        brief, meta = gen.generate(b["text"], prev, nxt)
        entry = {"index": i, "section": b.get("section"), "text": b["text"],
                  "valid": brief is not None, "brief": brief,
                  "error": meta.get("error"), "seconds": meta["seconds"]}
        briefs[str(i)] = entry
        if brief is None:
            stats["invalid"] += 1
        else:
            stats["generated"] += 1
        tmp = cpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=1)
        os.replace(tmp, cpath)
        print(f"  [{i + 1}/{len(blocks)}] {'OK ' if brief else 'FAIL'} "
              f"{meta['seconds']:5.1f}с {b['text'][:60]}")

    out = os.path.join(video_dir, "media_plan", "shot_briefs.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"model": gen.model_name, "prompt_version": gen.prompt_version,
                    "stats": stats, "briefs": briefs}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out)
    print(f"\nБлоков: {len(blocks)}, из кэша: {stats['cached']}, "
          f"посчитано: {stats['generated']}, невалидных: {stats['invalid']}")
    print(f"Время: {time.time() - t_start:.0f}с")
    print(f"Результат: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
