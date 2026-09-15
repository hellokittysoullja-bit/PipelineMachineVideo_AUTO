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

КАЧЕСТВО ИЗМЕРЕНО И ПЛАНКУ ПОКА НЕ ПРОХОДИТ — это главное, что надо
знать об этом модуле. Прогон боевого промпта на трёх реальных фразах:

  «посмотри НЕ НА МЕЧ... посмотри, куда он упал»
      forbidden="меч", function="scene",
      shot_en="A fallen knight, his surroundings, people around him"   ВЕРНО

  «Рыцарей убивала земля»
      shot_en="Knights being killed by the earth"                      сносно
      function="object"                                                НЕВЕРНО (это сцена)

  «рыцарь весил как холодильник, на коня его поднимали КРАНОМ...»
      shot_en="A knight, a crane, a turned turtle"                     ПЛОХО
      forbidden="холодильник, коня, краном, черепаха"                  ВЫДУМАНО

Третий случай хуже сегодняшнего запроса секции: «a crane» приведёт
СТРОИТЕЛЬНЫЙ КРАН, то есть планировщик внесёт брак, которого сейчас нет.
Планка этого репозитория — «ничьи и победы, ноль регрессов» — не
выполнена, поэтому флаг выключен по умолчанию и включать его без
повторного замера нельзя. Факт закреплён тестом
(test_measured_regression_on_the_metaphor_unit), а не комментарием:
починится — тест упадёт и заставит перечитать вывод.

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
PLANNER_PROMPT_VERSION = 2

# Путь к модели и бинарю — ТОЛЬКО из окружения. Ни одного зашитого пути:
# у владельца они свои, а в контейнере свои.
LLAMA_BIN = os.environ.get("LLAMA_CLI_BIN", "")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL_GGUF", "")

LLAMA_THREADS = int(os.environ.get("LLAMA_THREADS", "4") or 4)
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
    "Ты режиссёр монтажа документального ролика. По фразе диктора реши, "
    "ЧТО показать на экране в этот момент.\n"
    "Ответь СТРОГО одним JSON-объектом, без пояснений до и после.\n"
    "Ключи:\n"
    '  "subject" — главный предмет или герой кадра (по-русски, кратко);\n'
    '  "action" — что с ним происходит (по-русски, кратко), или null;\n'
    '  "forbidden" — предмет, который фраза ЯВНО ОТВЕРГАЕТ словами '
    '"не"/"а не"/"это неправда". Если фраза просто рассказывает о предмете '
    '— null. Подлежащее и дополнение фразы сюда НЕ попадают;\n'
    '  "function" — одно из: object, scene, illustration, map, texture;\n'
    '  "shot_en" — ЧТО ВИДИТ КАМЕРА, по-английски, 4-12 слов, конкретно, '
    "без выдуманных дат и имён.\n"
    "Если чего-то нет в тексте — ставь null, не придумывай."
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
    сломал бы нежадный `\{.*?\}` на первой же внутренней скобке.
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

STATS = {"calls": 0, "cache_hits": 0, "invalid": 0, "timeouts": 0,
         "skipped_author_brief": 0, "planned": 0}


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


def unit_key(text):
    """Ключ кэша — по ТЕКСТУ фразы, а не по номеру юнита.

    Номера сдвигаются от любой правки сценария выше по тексту, и план
    молча описывал бы чужую фразу. Ровно тот дефект, от которого уже
    защищается lock в шотлисте и ради которого `[shot:]` сделан инлайновым.
    """
    h = hashlib.md5()
    h.update(f"v{PLANNER_PROMPT_VERSION}\x00".encode("utf-8"))
    h.update(os.path.basename(LLAMA_MODEL).encode("utf-8"))
    h.update(b"\x00")
    h.update(" ".join((text or "").split()).encode("utf-8"))
    return h.hexdigest()[:16]


def build_prompt(text):
    """ChatML ровно того вида, который принимает Qwen2.5-Instruct."""
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nФраза диктора: «{' '.join((text or '').split())}»"
            f"<|im_end|>\n<|im_start|>assistant\n")


def parse_reply(raw):
    """Достать заявку из ответа модели и ПРОВЕРИТЬ её.

    Рендер никогда не доверяет плану без проверки типов и диапазонов — тот
    же принцип, что у speech_plan.json. Невалидный ответ -> None, юнит
    остаётся без брифа, то есть ведёт себя как сегодня.
    """
    if not raw:
        return None
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

    fn = obj.get("function")
    fn = fn if fn in VALID_FUNCTIONS else None

    forbidden = obj.get("forbidden")
    forbidden = forbidden.strip() if isinstance(forbidden, str) and forbidden.strip() else None

    subject = obj.get("subject")
    subject = subject.strip() if isinstance(subject, str) and subject.strip() else None

    return {"shot_en": shot, "function": fn, "forbidden": forbidden,
            "subject": subject}


def _run_model(prompt):
    """Один живой вызов llama.cpp. Любая беда -> None, без исключения."""
    cmd = [LLAMA_BIN, "-m", LLAMA_MODEL, "-t", str(LLAMA_THREADS),
           "-c", "2048", "-n", str(MAX_TOKENS), "--temp", "0.2",
           "--no-warmup", "--single-turn", "-p", prompt]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=CALL_TIMEOUT_SEC,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        STATS["timeouts"] += 1
        return None
    except Exception:
        return None
    if r.returncode != 0:
        return None
    # Возвращается ВЕСЬ вывод: ответ из него достаёт _last_json_object().
    # Маркер `<|im_start|>assistant` для этого не годится — llama-cli
    # обрезает эхо промпта, и маркера в stdout нет.
    out = r.stdout or ""
    return out


def plan_unit(text, cache_dir=None):
    """Заявка для одной фразы: кэш -> живой вызов -> проверка."""
    key = unit_key(text)
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
    parsed = parse_reply(_run_model(build_prompt(text)))
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
    todo = [b for b in blocks if not (b.get("shot_brief") or "").strip()]
    STATS["skipped_author_brief"] = len(blocks) - len(todo)
    for n, b in enumerate(todo, 1):
        text = (b.get("text") or "").strip()
        if not text:
            continue
        got = plan_unit(text, cache_dir)
        if got:
            plan[unit_key(text)] = got
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
    """Готовый план с диска, или пустой словарь."""
    try:
        with open(os.path.join(video_dir, "media_plan", PLAN_NAME),
                  encoding="utf-8") as f:
            data = json.load(f)
        units = data.get("units")
        return units if isinstance(units, dict) else {}
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
    for b in blocks:
        if (b.get("shot_brief") or "").strip():
            continue
        got = plan.get(unit_key(b.get("text") or ""))
        if not got:
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
          f"таймаутов {STATS['timeouts']}, "
          f"пропущено с авторским брифом {STATS['skipped_author_brief']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
