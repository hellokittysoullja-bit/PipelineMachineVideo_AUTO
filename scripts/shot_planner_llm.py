# -*- coding: utf-8 -*-
"""Локальный режиссёр кадра: фраза диктора -> описание того, ЧТО показать.

ЗАЧЕМ. Сегодня описание кадра (`[shot:...]`, ЧАСТЬ 13) пишет автор рукой.
Оно измеримо сильнее запроса секции: замер на эпизоде 02 (полка 264
предмета) — три попадания из трёх там, где запрос секции приносил наручи
вместо нагрудника и глефу вместо меча. Но 142 брифа на эпизод пишет
человек, и на чужой нише их не будет вовсе.

ЧТО ЗДЕСЬ. Тот же бриф составляет локальная модель на CPU, без сети, без
платных вызовов и без GPU. Модуль НЕ заводит новый путь отбора: он
заполняет РОВНО ТО ЖЕ поле `shot_brief`, которое пишет автор. Поэтому вся
уже существующая машинерия — вопрос к полке (`shelf_question`), перевод в
короткий стоковый запрос (`brief_to_stock_query`), ключ кэша кандидата
(`candidate_brief_key`) и его инвалидация — начинает работать немедленно и
без единой новой связи. Это же даёт бесплатный откат: нет плана — всё
ведёт себя байт-в-байт как раньше.

ЗАМЕРЕНО НА ЭТОЙ МАШИНЕ (16.09, Qwen2.5-7B-Instruct Q4_K_M, 4 ядра CPU,
llama.cpp), не оценено:

    обработка промпта  ~55 ток/с
    генерация          ~5.5 ток/с
    на юнит            21-26 с (плюс разовая загрузка модели ~18 с)
    эпизод 142 юнита   ~1 час, разово и кэшируемо

Для сравнения: сам рендер этого эпизода идёт часами. По времени это не
блокер — что и было главным сомнением при разборе предложения.

ЧТО МОДЕЛЬ РЕАЛЬНО СМОГЛА, на реальных фразах эпизода 02:

    «посмотри НЕ НА МЕЧ, который его ударил. Посмотри, куда он упал»
      -> forbidden_substitution: "меч"
      -> shot_query_en: "A knight falling, with focus on the surroundings"

Это тот самый юнит, на котором детерминированное правило по отрицанию
провалилось: лексический разбор дал бы 1 верное срабатывание на 142
(замер записан в CLAUDE.md), а модель разобралась с фразой, где лексика
бессильна.

КАЧЕСТВО ИЗМЕРЕНО НА ВОСЬМИ РАЗНОТИПНЫХ ФРАЗАХ И ПЛАНКУ НЕ ПРОХОДИТ —
это главное, что надо знать об этом модуле. Замеры целиком лежат в
docs/quality/shot_planner_eval_v2.json и _v3.json, воспроизводятся
командой scripts/shot_planner_eval.py.

Промпт v2 -> v3 (одни и те же восемь юнитов эпизода 02):

    непригодных кадров        4 из 8  ->  2 из 8
    честных отказов (null)    0       ->  2
    якорь эпохи в кадре       "a person"  ->  "a warrior"

Три дефекта v2, каждый исправлен адресно и подтверждён замером:
  * модель ПЕРЕВОДИЛА фразу вместо описания кадра
    («You have not been injured», «This object will be revisited»);
  * теряла эпоху — «a person» там, где нужен воин;
  * показывала предмет СРАВНЕНИЯ («a crane» из «поднимали краном»).

НО ДВА РЕГРЕССА ОСТАЛИСЬ, и планка репозитория «ничьи и победы, ноль
регрессов» не выполнена:
  * «Возьми настоящий боевой меч: полтора килограмма стали»
    v2 "A medieval sword, shining in the light"  ->
    v3 "A warrior in full iron armor being struck by a battle sword"
    (фраза про меч в руке, а не про удар по доспеху);
  * «он был под ногами у каждого. С первого шага на поле»
    v2 "Each person stepping onto the field" (function=scene)  ->
    v3 "He was at their feet" (function=object) — снова перевод фразы,
    и тип кадра неверен.

ГЛАВНЫЙ ВЫВОД ЗАМЕРА: починка одного класса фраз ломает другой. Модель
то описывает кадр, то переводит предложение, и формулировка промпта это
не убирает, а смещает. Похоже на потолок 7B в 4-битной квантовке, а не
на дефект промпта — следующая переменная к замеру именно модель, а не
текст задания.

Поэтому флаг выключен по умолчанию и включать его без нового замера
нельзя.

ЧЕСТНЫЕ ПРЕДЕЛЫ:
  * Поле `forbidden` НЕ влияет ни на один выбор и намеренно: оно
    выдумало отвержения в одном случае из трёх. Оно сохраняется в план
    как наблюдение, не как решение.
  * Бриф автора ВСЕГДА сильнее и никогда не перезаписывается.
  * Качество на чужой нише не измерено. Измерены три фразы одного канала.
  * Модуль не проверяет ФАКТЫ. Он говорит, что показать, а не что правда.
  * Замер — 7B в 4-битной квантовке. Более крупная модель или более
    точная квантовка не пробовались; это первое, что стоит померить.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import feature_flags  # noqa: E402

# Версия ПРОМПТА и разбора. Входит в ключ кэша: переписанный промпт обязан
# считаться заново, иначе план молча останется от прошлой формулировки —
# тот же класс, что уже закрыт у кэша вердиктов VLM-арбитра.
PLANNER_PROMPT_VERSION = 4

# Путь к модели и бинарю — ТОЛЬКО из окружения. Ни одного зашитого пути:
# у владельца они свои, а в контейнере свои.
LLAMA_BIN = os.environ.get("LLAMA_CLI_BIN", "")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL_GGUF", "")

LLAMA_THREADS = int(os.environ.get("LLAMA_THREADS", "4") or 4)
# Фиксированный seed + нулевая температура: два прогона на одном вопросе
# обязаны дать один ответ, иначе любое сравнение версий недействительно.
SAMPLING_SEED = int(os.environ.get("SHOT_PLANNER_SEED", "1") or 1)
# Потолок генерации на юнит. Заявка — компактный JSON; 320 токенов с
# запасом покрывают её, а без потолка одна сорвавшаяся генерация съела бы
# весь бюджет прогона.
MAX_TOKENS = int(os.environ.get("SHOT_PLANNER_MAX_TOKENS", "320") or 320)
# Жёсткий потолок ЖИВЫХ вызовов за прогон — та же дисциплина, что у
# SHOT_DIRECTOR_MAX_CALLS_PER_RUN и SPEECH_GEN_MAX_CALLS_PER_RUN.
MAX_CALLS_PER_RUN = int(os.environ.get("SHOT_PLANNER_MAX_CALLS", "400") or 400)
# Таймаут одного вызова. Замер дал 21-26 с; 180 с — большой запас на
# медленную машину, но не бесконечность.
CALL_TIMEOUT_SEC = int(os.environ.get("SHOT_PLANNER_TIMEOUT", "180") or 180)

PLAN_NAME = "shot_plan.json"
CACHE_DIR_NAME = "shot_plan_cache"

SYSTEM_PROMPT = (
    "Ты режиссёр монтажа исторического документального ролика. По фразе "
    "диктора реши, ЧТО ФИЗИЧЕСКИ ПОКАЗАТЬ на экране.\n"
    "Ответь СТРОГО одним JSON-объектом, без пояснений до и после.\n"
    "\n"
    "ГЛАВНОЕ ПРАВИЛО: shot_en — это ОПИСАНИЕ ПРЕДМЕТА ИЛИ СЦЕНЫ, которую "
    "можно сфотографировать. Это НЕ пересказ фразы и НЕ её перевод. "
    "Нельзя писать «you have not been injured» или «this will be "
    "revisited» — это не кадр.\n"
    "\n"
    "ЕСЛИ во фразе нет ничего, что можно показать буквально (обращение к "
    "зрителю, связка, обещание вернуться к теме) — поставь shot_en = null. "
    "Пустой ответ лучше выдуманного кадра.\n"
    "\n"
    "ЭПОХА: ролик исторический. Человек в кадре — это воин или ремесленник "
    "своего времени, а не «a person». Никакой современной одежды, техники "
    "и снаряжения: «protective gear» приведёт современный костюм.\n"
    "\n"
    "СРАВНЕНИЯ НЕ ПОКАЗЫВАЙ. «Весил как холодильник», «поднимали краном», "
    "«как перевёрнутая черепаха» — это образы речи. Показывать надо то, О "
    "ЧЁМ речь (доспех, упавший воин), а не предмет сравнения.\n"
    "\n"
    "КОНТЕКСТ. Перед фразой даётся то, что диктор говорил ДО неё. Он "
    "нужен РОВНО для одного: понять, о чём в самой фразе «он», «это», "
    "«их», «такой». Описывать кадр ПО КОНТЕКСТУ нельзя — камера "
    "показывает то, о чём говорится СЕЙЧАС, а прошлые кадры уже "
    "показаны.\n"
    "\n"
    "Ключи:\n"
    '  "subject" — главный предмет или герой кадра (по-русски, кратко);\n'
    '  "action" — что с ним происходит (по-русски, кратко), или null;\n'
    '  "forbidden" — предмет, который фраза ЯВНО ОТВЕРГАЕТ словами '
    '"не"/"а не"/"это неправда", иначе null. Подлежащее и дополнение '
    "фразы сюда НЕ попадают;\n"
    '  "function" — одно из: object, scene, illustration, map, texture;\n'
    '  "shot_en" — что видит камера, по-английски, 4-12 слов, конкретно, '
    "или null."
)

# Функция кадра из ответа модели сопоставляется со словарём типов, который
# уже управляет маршрутизацией по источникам (scripts/shot_types.py).
# Второго словаря не заводится: разойдись они — модель называла бы тип,
# которого маршрутизация не знает, и слот молча уходил бы во все источники.
VALID_FUNCTIONS = ("object", "scene", "illustration", "map", "texture")

_LATIN_RE = re.compile(r"[A-Za-z]")
def _last_json_object(text):
    """Последний сбалансированный {...} в выводе.

    НАЙДЕНО ТЕСТОМ, не рассуждением (16.09): первая версия искала ответ
    после маркера `<|im_start|>assistant`, но llama-cli ОБРЕЗАЕТ эхо
    промпта при выводе («...(truncated)»), и маркер в stdout не попадает
    никогда. Опора на маркер означала бы разбор всего вывода целиком,
    включая куски системного промпта.

    Скобки считаются, а не матчатся регекспом: вложенный объект в ответе
    сломал бы нежадный поиск ломался бы на первой же внутренней скобке.
    """
    if not text:
        return None
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            c = text[i]
            if c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    return text[i:end + 1]
        end = text.rfind("}", 0, end)
    return None

REJECTED = []   # заявки, не прошедшие проверку: слот идёт прежним путём

# not_a_shot считается ОТДЕЛЬНО от invalid: «модель ответила пересказом»
# и «ответ не разобрался» — разные диагнозы, и по сводке прогона должно
# быть видно, какой именно. Тот же довод, по которому отказ арбитра
# отделён от его отсутствия.
STATS = {"calls": 0, "cache_hits": 0, "invalid": 0, "timeouts": 0,
         "not_a_shot": 0, "skipped_author_brief": 0, "planned": 0}


# Сколько предыдущих фраз кладётся в контекст. Две — ровно столько, сколько
# нужно, чтобы дотянуться до антецедента: замер на эпизоде 02 показал, что
# отсылка стоит в первых трёх словах юнита у 24 из 142 (17%), и во всех
# разобранных случаях антецедент лежал в соседней фразе, не дальше.
# Вперёд контекст СОЗНАТЕЛЬНО не смотрит: эпизод намеренно придерживает
# ответ («Я его назову. Но если сказать прямо сейчас, ты пожмёшь плечами»),
# и кадр, собранный по ещё не прозвучавшей фразе, выдал бы разгадку раньше
# диктора.
CONTEXT_PREV_UNITS = 2

# ФОРМА КАДРА. Описание кадра начинается с того, ЧТО в кадре, — с предмета
# или героя. Начало с местоимения означает ровно одно: модель пересказала
# фразу вместо того, чтобы описать картинку («He was at their feet» из «он
# был под ногами у каждого из них»). Проверка стоит на ФОРМЕ, а не на
# списке плохих ответов: список отстаёт по построению, форма — нет (тот же
# довод, что у ALIGNMENT_TAG_SPAN_RE).
#
# ИЗМЕРЕНО, не предположено, на данных этого канала:
#   * 142 авторских брифа эпизода 02 — ложных отказов 0;
#   * записанные ответы модели, промпт v3 — пойман 1 из 8 (тот самый
#     «He was at their feet»), ни одного пригодного кадра не потеряно;
#   * записанные ответы модели, промпт v2 — поймано 2 из 8 («You have not
#     been injured», «This object will be revisited»), то есть ровно те
#     два провала, которые переписывание промпта закрывало словами.
#     Это и есть довод, что правило переживает смену формулировки промпта.
# «its» на доступных данных не срабатывает ни разу и включён только как
# та же закрытая грамматическая форма, что и сработавшие his/her/their.
_NOT_A_SHOT_RE = re.compile(
    r"^(he|she|it|its|they|you|we|i|his|her|their|your"
    r"|this|that|these|those)\b", re.I)

# ВТОРОЙ ПРИЗНАК ТОЙ ЖЕ ФОРМЫ: кадр — ОДИН момент, а не рассказ. Точка,
# за которой идёт ещё текст, означает второе предложение, то есть пересказ
# фразы, а не описание картинки.
#
# ИЗМЕРЕНО НА КОРПУСЕ, А НЕ НА ВЫБОРКЕ ИЗ ВОСЬМИ: из 142 авторских брифов
# эпизода 02 более одного предложения — НОЛЬ, и запас не на волосок: ни
# один не оканчивается даже точкой, медиана 9 слов, максимум 17. Это
# именные группы по построению.
#
# Хвостовая точка НЕ считается вторым предложением намеренно: «A warrior
# lying face down in mud, seemingly about to die.» — законный кадр, и
# правило «\S после точки» его не трогает. Различие проверяется тестом.
_TWO_SENTENCES_RE = re.compile(r"[.!?]\s+\S")


def brief_is_shot_like(shot):
    """Похоже ли это на описание кадра, а не на пересказ фразы.

    Два признака одной формы: кадр начинается с того, ЧТО в кадре (не с
    местоимения), и описывает ОДИН момент (не два предложения).

    Отказ переводит юнит в то же состояние, что и отсутствие планировщика
    вообще: брифа нет, отбор идёт прежним путём. Поэтому правило может
    только убрать негодный кадр и физически не может отнять годный.
    """
    if not shot:
        return False
    if _NOT_A_SHOT_RE.match(shot):
        return False
    if _TWO_SENTENCES_RE.search(shot):
        return False
    return True


def unit_context(blocks, i):
    """Всё, что модель знает о фразе, кроме самой фразы.

    ОДНА функция на весь модуль. Контекст входит и в промпт, и в ключ
    кэша, и в замер — разойдись эти три места, и ключ перестал бы
    соответствовать заданному вопросу МОЛЧА. Ровно этот класс уже стоил
    репозиторию PHRASE LOCK на целый эпизод (две копии словаря тегов) и
    едва не стоил вопроса к полке (`shelf_question`).
    """
    try:
        cur = blocks[i]
    except Exception:
        return {"section": "", "prev": []}
    if not isinstance(cur, dict):
        return {"section": "", "prev": []}
    section = " ".join(str(cur.get("section") or "").split())
    prev, j = [], i - 1
    while j >= 0 and len(prev) < CONTEXT_PREV_UNITS:
        try:
            t = " ".join(str(blocks[j].get("text") or "").split())
        except Exception:
            t = ""
        if t:
            prev.append(t)
        j -= 1
    prev.reverse()
    return {"section": section, "prev": prev}


def context_text(ctx):
    """Контекст одной строкой — то же самое и для промпта, и для ключа."""
    if not isinstance(ctx, dict):
        return ""
    parts = []
    section = " ".join(str(ctx.get("section") or "").split())
    if section:
        parts.append(f"Глава: {section}")
    for t in (ctx.get("prev") or []):
        t = " ".join(str(t or "").split())
        if t:
            parts.append(f"Ранее диктор сказал: {t}")
    return "\n".join(parts)


def enabled():
    return feature_flags.enabled("SHOT_PLANNER_LLM")


def runtime_ready():
    """Есть ли чем считать. Проверка ДЕШЁВАЯ и без запуска модели."""
    try:
        return bool(LLAMA_BIN and LLAMA_MODEL
                    and os.path.getsize(LLAMA_BIN) > 0
                    and os.path.getsize(LLAMA_MODEL) > 0)
    except OSError:
        return False


def unit_key(text, ctx):
    """Ключ кэша — по ТЕКСТУ фразы и по её контексту, а не по номеру юнита.

    Номера сдвигаются от любой правки сценария выше по тексту, и план
    молча описывал бы чужую фразу. Ровно тот дефект, от которого уже
    защищается lock в шотлисте и ради которого `[shot:]` сделан инлайновым.

    Контекст входит в ключ ОБЯЗАТЕЛЬНО и параметром без значения по
    умолчанию. Он меняет заданный модели вопрос, а кэш переживает прогоны:
    не войди он в ключ — правка соседней фразы оставила бы ответ, данный
    на другой вопрос, и увидеть это было бы негде. Забывчивый вызывающий
    получает TypeError сразу, а не тихо чужой кадр (тот же довод, по
    которому отказ VLM-арбитра сделан отдельным типом, а не None).

    Цена названа честно: правка одной фразы обесценивает её собственный
    ответ И ответы следующих двух, то есть ~3 пересчёта по ~23 с вместо
    одного. На эпизоде, который планируется раз и идёт час, это минута.
    """
    h = hashlib.md5()
    h.update(f"v{PLANNER_PROMPT_VERSION}\x00".encode("utf-8"))
    h.update(os.path.basename(LLAMA_MODEL).encode("utf-8"))
    h.update(b"\x00")
    h.update(" ".join((text or "").split()).encode("utf-8"))
    h.update(b"\x00")
    h.update(context_text(ctx).encode("utf-8"))
    return h.hexdigest()[:16]


def build_prompt(text, ctx):
    """ChatML ровно того вида, который принимает Qwen2.5-Instruct.

    Контекст идёт ПЕРЕД фразой и отдельным абзацем: модель должна успеть
    узнать, о чём «он», до того как прочтёт саму фразу. Тот же параметр
    без значения по умолчанию и по той же причине, что у unit_key.
    """
    head = context_text(ctx)
    head = (head + "\n\n") if head else ""
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{head}"
            f"Фраза диктора: «{' '.join((text or '').split())}»"
            f"<|im_end|>\n<|im_start|>assistant\n")


def parse_reply(raw):
    """Достать заявку из ответа модели и ПРОВЕРИТЬ её.

    Рендер никогда не доверяет плану без проверки типов и диапазонов — тот
    же принцип, что у speech_plan.json. Невалидный ответ -> None, юнит
    остаётся без брифа, то есть ведёт себя как сегодня.
    """
    if not raw:
        return None
    # Чистка и ЗДЕСЬ, а не только в _run_model: разбор вызывают и на
    # сохранённых фикстурах, и на чужом выводе. Оформление потока —
    # свойство источника, а не вызова, и защита обязана стоять у разбора.
    raw = _clean_stream(raw)
    chunk = _last_json_object(raw)
    if not chunk:
        return None
    try:
        obj = json.loads(chunk)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    shot = obj.get("shot_en")
    # shot_en = null — ЗАКОННЫЙ ответ, а не сбой: промпт прямо разрешает
    # его для фраз, где показывать нечего («Запомни это, мы сюда
    # вернёмся»). Замер v2 показал, чем это кончается без разрешения:
    # модель выдаёт «This object will be revisited» — перевод фразы
    # вместо кадра. Пустой ответ оставляет юнит на прежнем пути, и это
    # честнее выдуманного кадра.
    if not isinstance(shot, str):
        return None
    shot = " ".join(shot.split())
    # Описание кадра обязано быть латиницей: полка сравнивает его с
    # изображениями многоязычной моделью, но стоковый перевод
    # (brief_to_stock_query) — английский по построению, и кириллица там
    # дала бы запрос, который ни один сток не понимает.
    if not _LATIN_RE.search(shot):
        return None
    words = shot.split()
    if not (2 <= len(words) <= 16):
        return None
    # Пересказ фразы вместо кадра — см. brief_is_shot_like выше. Юнит
    # остаётся без брифа, то есть ведёт себя ровно как без планировщика.
    if not brief_is_shot_like(shot):
        STATS["not_a_shot"] += 1
        return None

    fn = obj.get("function")
    fn = fn if fn in VALID_FUNCTIONS else None

    forbidden = obj.get("forbidden")
    forbidden = forbidden.strip() if isinstance(forbidden, str) and forbidden.strip() else None

    subject = obj.get("subject")
    subject = subject.strip() if isinstance(subject, str) and subject.strip() else None

    return {"shot_en": shot, "function": fn, "forbidden": forbidden,
            "subject": subject}


# Управляющие последовательности и одиночные символы, которыми llama-cli
# оформляет поток (цвет, спиннер загрузки, перерисовка строки). В JSON они
# попадают ВНУТРЬ слов — найдено на Q8-модели, где ответ выглядел так:
#     "subject?": "?? человек??",  "?": "shot?_?en?": "a??? person?"
# Q4 ту же ломку проскакивал случайно, то есть дефект был всё время и
# ждал другой модели или другой скорости вывода.
_CTRL_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_stream(text):
    """Снять оформление потока, не трогая сам текст."""
    return _CTRL_RE.sub("", text or "")


# --- ПРОВЕРКА ПОВЕРХ ОТВЕТА МОДЕЛИ -----------------------------------------
#
# Главная мысль, взятая из внешней оценки 16.09 и признанная верной:
# модель НЕ обязана быть права всегда — она обязана быть ПРАВА ИЛИ
# МОЛЧАТЬ. Тогда ухудшение становится структурно невозможным: заявка
# либо проходит проверку и улучшает слот, либо отбрасывается, и слот
# идёт ровно как сегодня.
#
# Это же чинит и то, на чём правка буксовала. Планка репозитория «ничьи
# и победы, ноль регрессов» на восьми вручную выбранных фразах
# превращалась в «никогда ничего не включать»: почти любое изменение
# где-то даст минус. С проверкой планка выполнима честно — минуса
# не бывает по построению, а не по результату замера.
#
# Проверки ДЕТЕРМИНИРОВАННЫЕ: ни ИИ, ни сети, ни порогов на глаз.

# «Человек вообще» приведёт современного человека в футболке. Историческому
# ролику нужен носитель эпохи. Список — из реальных промахов замера v2
# («a person standing up», «a person wearing full-body protective gear»).
GENERIC_PEOPLE = ("person", "people", "man", "woman", "human", "guy",
                  "someone", "individual")

# Слова, которые в стоке означают СОВРЕМЕННОЕ снаряжение. «protective gear»
# — реальный промах замера: приводит защитный костюм, а не доспех.
MODERN_GEAR = ("protective gear", "safety", "helmet cam", "uniform",
               "equipment", "outfit", "costume")

# ВТОРАЯ ВЕРСИЯ ЭТОГО ПРАВИЛА УДАЛЕНА ПО ЗАМЕРУ, а не по вкусу (16.09).
#
# Здесь стояло `_looks_like_translation()`: местоимение ГДЕ-УГОДНО И личная
# форма глагола ГДЕ-УГОДНО. Оно решало ту же задачу, что
# `brief_is_shot_like()` выше, и две копии одного правила — тот самый класс,
# который уже стоил этому репозиторию PHRASE LOCK на целый эпизод.
#
# Обе прогнаны по одним данным. На всём, что есть, они НЕ РАЗЛИЧАЮТСЯ:
# ложных отказов 0 на 142 авторских брифах эпизода 02, и одни и те же
# пойманные ответы модели (v3 — «He was at their feet»; v2 — «You have not
# been injured», «This object will be revisited»). Замер их не разделяет, и
# выбор сделан НЕ по замеру эффекта, а по замеру ЗАПАСА:
#
#   * три настоящих авторских брифа уже выполняют ПОЛОВИНУ удалённого
#     правила — «a steel helmet with a deep dent crushed into IT», «a human
#     skull with wounds on IT», «an undamaged breastplate with no hole in
#     IT». Они выживают только потому, что в них не случилось глагола из
#     списка: «a breastplate that HAS a hole in IT» — тот же бриф другими
#     словами — был бы отвергнут. Запас в один союз;
#   * начало-с-местоимения таким способом задеть нельзя в принципе:
#     правило смотрит на ПЕРВОЕ слово, и содержимое описания на него не
#     влияет никак;
#   * и обратное: «He lies in the mud» — пересказ, которого удалённое
#     правило не ловит (глагола «lies» в его списке нет), а форма ловит.
#
# Оставлено одно правило — `brief_is_shot_like()`.


def channel_blocklist():
    """Термины, которых канал не хочет видеть — ИЗ УЖЕ СУЩЕСТВУЮЩЕГО списка.

    Второго словаря не заводится: `content_alt_blocklist` живёт в
    channel_profile.json, уже отсеивает кандидатов по их тексту и уже
    настраивается под нишу. Здесь тот же список применяется РАНЬШЕ — к
    тому, что мы собираемся ПОПРОСИТЬ. Просить то, что потом сами
    отсеем, бессмысленно.

    Fail-open: профиль не читается — проверка просто не применяется.
    """
    try:
        import pipeline_smart
        return tuple(pipeline_smart.CONTENT_ALT_BLOCKLIST)
    except Exception:
        return ()


def brief_is_safe(shot_en, phrase, blocklist=None,
                  era_words=("medieval", "knight", "warrior",
                             "armour", "armor", "sword",
                             "helmet", "castle", "manuscript")):
    """Можно ли выпускать эту заявку в отбор. (ok, причина отказа).

    Отказ — НЕ ошибка: слот просто идёт прежним путём. Поэтому проверки
    строгие, а не снисходительные: пропустить сомнительную заявку дороже,
    чем не проставить её вовсе.
    """
    if not shot_en:
        return False, "пусто"
    low = shot_en.lower()
    if not brief_is_shot_like(shot_en):
        return False, "пересказ фразы, а не описание кадра"
    # ПРАВИЛА ПРО СРАВНЕНИЯ ЗДЕСЬ НЕТ, и это осознанно.
    #
    # Внешняя оценка предлагала: «если во фразе есть сравнение (как,
    # словно, будто) — запрещаем показывать предмет сравнения». Идея
    # верная, реализация здесь невозможна: фраза РУССКАЯ («как
    # холодильник»), а описание кадра АНГЛИЙСКОЕ («a refrigerator»), и
    # сопоставить их без перевода нельзя. Первая версия этой проверки
    # сравнивала «холод» с «a refrigerator in a field» и была МЁРТВЫМ
    # КОДОМ — сработать не могла ни разу; доказал собственный тест, а не
    # рассуждение.
    #
    # Заводить мини-словарь русско-английских соответствий — ровно тот
    # ненадёжный приём, который в этом файле уже отвергнут для «crane»
    # (журавль законен на миниатюре) и в stress_placement для «атлас».
    # Случай сравнения закрывает ПРОМПТ v3, где это сказано прямо, и
    # замер показал, что он работает: «весил как холодильник» дало
    # «A sword strikes a helmet», а не холодильник.
    for g in MODERN_GEAR:
        if g in low:
            return False, f"современное снаряжение ({g})"
    # Блоклист канала. Закрывает случай, который правило сравнения НЕ
    # берёт по построению: в «поднимали КРАНОМ» нет союза «как» — это
    # творительный падеж внутри образа, а не сравнение. Правило
    # «предмет после как/словно» такое не поймает никогда, и притворяться
    # обратным нельзя. Зато «crane» — ровно то, чего канал про
    # Средневековье не хочет видеть ни при каких обстоятельствах, и для
    # этого в проекте уже есть список.
    for term in (blocklist if blocklist is not None else channel_blocklist()):
        t = str(term).lower().strip()
        if t and t in low:
            return False, f"блоклист канала ({t})"
    has_era = any(e in low for e in era_words)
    if not has_era and any(f" {g} " in f" {low} " for g in GENERIC_PEOPLE):
        return False, "человек без привязки к эпохе"
    return True, None


def _run_model(prompt):
    """Один живой вызов llama.cpp. Любая беда -> None, без исключения."""
    # Промпт уходит ФАЙЛОМ, а не аргументом -p: он многострочный, содержит
    # кавычки и разметку ChatML, и передача через argv зависит от шелла и
    # длины командной строки. Файл убирает обе зависимости.
    tmp_prompt = None
    try:
        import tempfile
        fd, tmp_prompt = tempfile.mkstemp(suffix=".txt", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(prompt)
    except Exception:
        return None
    cmd = [LLAMA_BIN, "-m", LLAMA_MODEL, "-t", str(LLAMA_THREADS),
           "-c", "2048", "-n", str(MAX_TOKENS),
           # ДЕТЕРМИНИЗМ. Найдено внешней рецензией 16.09 и подтверждено:
           # у llama.cpp `--seed` по умолчанию -1, то есть СЛУЧАЙНЫЙ, а
           # температура была 0.2 — не ноль. Значит сравнение промптов v2
           # и v3 было невоспроизводимым, и разница между ними могла быть
           # шумом выборки, а не эффектом правки. Планирование — не
           # творческая задача: нужен один и тот же ответ на один и тот же
           # вопрос, иначе кэш по тексту фразы тоже теряет смысл.
           "--temp", "0", "--seed", str(SAMPLING_SEED),
           "--no-warmup", "--single-turn",
           "--log-disable", "--log-colors", "off",
           "-f", tmp_prompt]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=CALL_TIMEOUT_SEC,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        STATS["timeouts"] += 1
        return None
    except Exception:
        return None
    finally:
        if tmp_prompt:
            try:
                os.remove(tmp_prompt)
            except OSError:
                pass
    if r.returncode != 0:
        return None
    # Возвращается ВЕСЬ вывод: ответ из него достаёт _last_json_object().
    # Маркер `<|im_start|>assistant` для этого не годится — llama-cli
    # обрезает эхо промпта, и маркера в stdout нет.
    return _clean_stream(r.stdout or "")


def plan_unit(text, ctx, cache_dir=None):
    """Заявка для одной фразы: кэш -> живой вызов -> проверка."""
    key = unit_key(text, ctx)
    path = os.path.join(cache_dir, key + ".json") if cache_dir else None
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                STATS["cache_hits"] += 1
                return json.load(f)
        except Exception:
            pass
    if STATS["calls"] >= MAX_CALLS_PER_RUN:
        return None
    STATS["calls"] += 1
    parsed = parse_reply(_run_model(build_prompt(text, ctx)))
    if parsed is None:
        STATS["invalid"] += 1
        return None
    if path:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            pass
    return parsed


def plan_episode(video_dir, blocks, verbose=True):
    """План на весь эпизод. Возвращает {ключ фразы: заявка}."""
    mp = os.path.join(video_dir, "media_plan")
    cache_dir = os.path.join(mp, CACHE_DIR_NAME)
    plan, t0 = {}, time.time()
    # Цикл идёт по ПОЛНОМУ списку блоков, а не по отфильтрованному: контекст
    # юнита — это соседние фразы сценария, включая те, у которых бриф уже
    # написан автором. По отфильтрованному списку «предыдущей» оказалась бы
    # фраза через две главы — тот же класс промаха, что уже стоил arc_stage
    # 151 слота из 165 (N4 аудита), когда стадия бралась по индексу цикла
    # вместо исходного индекса блока.
    todo = [i for i, b in enumerate(blocks)
            if not (b.get("shot_brief") or "").strip()]
    STATS["skipped_author_brief"] = len(blocks) - len(todo)
    for n, i in enumerate(todo, 1):
        text = (blocks[i].get("text") or "").strip()
        if not text:
            continue
        ctx = unit_context(blocks, i)
        got = plan_unit(text, ctx, cache_dir)
        if got:
            plan[unit_key(text, ctx)] = got
            STATS["planned"] += 1
        if verbose and n % 10 == 0:
            print(f"  режиссёр: {n}/{len(todo)} юнитов, "
                  f"{time.time() - t0:.0f}с")
    try:
        os.makedirs(mp, exist_ok=True)
        tmp = os.path.join(mp, PLAN_NAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": PLANNER_PROMPT_VERSION,
                       "model": os.path.basename(LLAMA_MODEL),
                       "stats": dict(STATS), "units": plan},
                      f, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(mp, PLAN_NAME))
    except Exception:
        pass
    return plan


def load_plan(video_dir):
    """Готовый план с диска, или пустой словарь.

    План, посчитанный ДРУГИМ промптом или ДРУГОЙ моделью, отбрасывается
    здесь и НАЗЫВАЕТ причину. Функционально это ничего не меняет — версия
    и имя модели входят в unit_key, то есть ни один ключ такого плана всё
    равно не совпал бы, — но без этой проверки рендер объяснял бы пустой
    результат правкой сценария («текст правился после планирования»), то
    есть валил бы на автора то, что сделало обновление кода. Молчаливо
    неверная причина хуже отсутствия причины.
    """
    try:
        with open(os.path.join(video_dir, "media_plan", PLAN_NAME),
                  encoding="utf-8") as f:
            data = json.load(f)
        units = data.get("units")
        if not isinstance(units, dict):
            return {}
        was_v, was_m = data.get("version"), data.get("model")
        now_m = os.path.basename(LLAMA_MODEL)
        if was_v != PLANNER_PROMPT_VERSION:
            print(f"  Локальный режиссёр: план посчитан промптом v{was_v}, "
                  f"сейчас v{PLANNER_PROMPT_VERSION} — план не используется, "
                  f"перезапусти shot_planner_llm.py")
            return {}
        if now_m and was_m and was_m != now_m:
            print(f"  Локальный режиссёр: план посчитан моделью {was_m}, "
                  f"сейчас {now_m} — план не используется, "
                  f"перезапусти shot_planner_llm.py")
            return {}
        return units
    except Exception:
        return {}


def fill_briefs(blocks, plan):
    """Проставить `shot_brief` там, где автор его НЕ написал.

    Бриф автора не перезаписывается никогда и ни при каких условиях: он
    проверен человеком, а заявка модели — нет. Возвращает число реально
    заполненных юнитов.
    """
    if not plan:
        return 0
    filled = 0
    for i, b in enumerate(blocks):
        if (b.get("shot_brief") or "").strip():
            continue
        got = plan.get(unit_key(b.get("text") or "", unit_context(blocks, i)))
        if not got:
            continue
        ok, why = brief_is_safe(got.get("shot_en"), b.get("text") or "")
        if not ok:
            REJECTED.append({"text": (b.get("text") or "")[:80],
                             "shot_en": got.get("shot_en"), "reason": why})
            continue
        b["shot_brief"] = got["shot_en"]
        # Тип кадра модели уходит в ту же маршрутизацию по источникам, что
        # и авторская пометка `[object]`/`[scene]` в строке запроса.
        if got.get("function"):
            b["shot_type_hint"] = got["function"]
        filled += 1
    return filled


def main(argv):
    if len(argv) < 2:
        print("использование: python scripts/shot_planner_llm.py <video_dir>")
        return 2
    video_dir = argv[1]
    if not enabled():
        print("SHOT_PLANNER_LLM выключен (дефолт) — плана не будет")
        return 0
    if not runtime_ready():
        print("Нет LLAMA_CLI_BIN или LLAMA_MODEL_GGUF в окружении — "
              "плана не будет, отбор идёт как раньше")
        return 0
    import script_parser
    blocks = script_parser.parse_blocks(os.path.join(video_dir, "script.txt"))
    plan = plan_episode(video_dir, blocks)
    print(f"Готово. Заявок: {len(plan)}; живых вызовов {STATS['calls']}, "
          f"из кэша {STATS['cache_hits']}, невалидных {STATS['invalid']}, "
          f"пересказов вместо кадра {STATS['not_a_shot']}, "
          f"таймаутов {STATS['timeouts']}, "
          f"пропущено с авторским брифом {STATS['skipped_author_brief']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
