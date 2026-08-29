"""LLM-режиссёр (опционально, SHOT_DIRECTOR_MODE=off/on, дефолт off) —
заполняет ТОЛЬКО остаток блоков, для которых нет ни авторского запроса
(=== PEXELS QUERIES === в script.txt), ни совпадения по channel_themes.json/
media_plan/themes.json (query_for() в pipeline_smart.py вернул None).
Именно в этом остатке проваливаются идиомы/метафоры ("взять быка за
рога" -> дословный запрос про рога быка мимо смысла) и абстрактные фразы
без предметных слов — подтверждено 113-позиционным независимым
бенчмарком трёх эмбеддинг-моделей (baseline/SigLIP2/Jina CLIP v2): часть
B (идиомы) у всех трёх 12-38% top-1, ни одна embedding-модель не
"понимает" метафору за счёт сравнения векторов — нужен LLM ДО поиска
(разворачивает идиому в конкретную сцену), реранк ПОСЛЕ (visual_director.py)
эту проблему в принципе не решает, он лишь сортирует уже неверно
подобранный пул.

НЕ вызывается на каждый слот эпизода — это было бы 200-400 вызовов на
ролик и противоречило бы уже принятому в этом пайплайне принципу не
заводить платный/квотированный API-вызов на каждую картинку (CLAUDE.md
ЧАСТЬ 13, Шаг 7.5 — тот же аргумент применён здесь). Раньше нераспознанный
остаток уходил в neighbor-inherit/GENERIC_FALLBACKS (pipeline_smart.
resolve_queries()) — тематически связанные с секцией, но не с КОНКРЕТНОЙ
фразой литералы. Этот модуль встаёт МЕЖДУ authored/theme-match и
neighbor-inherit, только для реального остатка (эмпирически — единицы
блоков на эпизод, не сотни).

Ключ — уже существующий GEMINI_API_KEY (.env, CLAUDE.md ЧАСТЬ 3/14), НЕ
новый сервис/ключ. thinkingConfig thinkingBudget=0 обязателен для
2.5-flash (ЧАСТЬ 14 — иначе обрыв MAX_TOKENS на длинных ответах; здесь
ответы короткие, но флаг ставится по той же документированной причине,
не наугад). Free-tier бюджет по факту ~20 вызовов/день НА МОДЕЛЬ —
эмпирика из ЧАСТЬ 14 этого канала, не рекламная цифра с сайта — поэтому:

- агрессивный кэш по хэшу текста (media_plan/shot_director_cache/,
  переживает повторные прогоны — тот же принцип, что speech_cache/ у
  speech_generate.py);
- жёсткий потолок вызовов ЗА ПРОГОН (SHOT_DIRECTOR_MAX_CALLS_PER_RUN,
  дефолт 15 — заведомо ниже дневного лимита, ENV может только СУЗИТЬ,
  тот же паттерн, что SPEECH_GEN_MAX_ATTEMPTS);
- fail-open на любую ошибку/таймаут/невалидный JSON/отсутствие ключа —
  никогда не роняет сборку, молча возвращает None, вызывающий код падает
  на старое поведение (neighbor-inherit/GENERIC_FALLBACKS) — ноль
  регрессии для эпизодов без ключа или с исчерпанным лимитом.

ОБНОВЛЕНО 27.08 — живой ключ подтверждён напрямую против реального API
Google (не мок, не гадание): этому конкретному ключу gemini-2.5-flash
недоступна ("This model models/gemini-2.5-flash is no longer available to
new users", HTTP 404) — Google сам подсказал в тексте ошибки
gemini-3.6-flash, она отвечает 200. thinkingConfig thinkingBudget=0 (был
обязателен для 2.5-flash, ЧАСТЬ 14 — иначе обрыв MAX_TOKENS) на 3.6-flash
даёт HTTP 400 "Request contains an invalid argument" — эта модель нулевой
бюджет мышления не принимает вообще (проверено изолированно: тот же
запрос без thinkingConfig -> HTTP 200, finishReason=STOP, валидный JSON;
thinkingBudget=-1 тоже 200). Модель и правило про thinkingConfig теперь
берутся динамически по имени модели (см. _thinking_config_for_model()) —
0 только для семейства 2.5, иначе не отправляется совсем. Прежний
GEMINI_API_KEY в .env был пуст — это больше не так, ключ вписан
пользователем и проверен реальным вызовом (см. коммит)."""
import os
import re
import json
import base64
import hashlib
import urllib.request
import urllib.error

SHOT_DIRECTOR_MAX_CALLS_PER_RUN = min(
    int(os.environ.get("SHOT_DIRECTOR_MAX_CALLS_PER_RUN", "15")), 15)
# gemini-2.5-flash недоступна ключу, проверенному 27.08 (см. докстринг
# модуля) — Google сам указал gemini-3.6-flash как замену. GEMINI_MODEL_
# OVERRIDE в .env позволяет вернуться на 2.5-flash (или любую другую) без
# правки кода, если ключ/аккаунт другой пользователь сменит.
SHOT_DIRECTOR_MODEL = os.environ.get("GEMINI_MODEL_OVERRIDE") or "gemini-3.6-flash"
SHOT_DIRECTOR_TIMEOUT_SEC = 20


def _thinking_config_for_model(model):
    """thinkingBudget=0 обязателен ТОЛЬКО для семейства 2.5 (см. ЧАСТЬ 14 —
    иначе обрыв MAX_TOKENS на длинных ответах). На gemini-3.6-flash тот же
    флаг ломает запрос целиком (HTTP 400 invalid argument — проверено
    вживую 27.08, не гипотеза). Для любой другой/будущей модели —
    thinkingConfig не отправляется вовсе: живой тест показал, что
    gemini-3.6-flash без него даёт finishReason=STOP на реальном промпте
    этого модуля (не обрублен), так что riskа MAX_TOKENS для неё нет —
    честная, проверенная деградация, не предположение на будущее."""
    if model.startswith("gemini-2.5"):
        return {"thinkingBudget": 0}
    return None

_PROMPT_TEMPLATE = (
    'Ты — постановщик кадра для закадрового видео на русском языке. '
    'Дана фраза из сценария военно-исторического YouTube-ролика. Она может '
    'быть идиомой/метафорой (её нельзя показывать буквально) или '
    'буквальным описанием сцены. Верни СТРОГО валидный JSON без пояснений '
    'и без markdown-разметки: {{"literal": true/false, "queries": '
    '["query1", "query2", "query3"]}} — 2-3 английских поисковых запроса '
    'для Pexels/Pixabay (2-5 слов каждый, конкретная визуальная сцена, без '
    'кириллицы, без указания текста на кадре). Если фраза — идиома или '
    'метафора, queries должны показывать смысл ДЕЙСТВИЯ/ЭМОЦИИ, которую '
    'она передаёт, а НЕ дословный перевод слов идиомы (пример: "взять '
    'быка за рога" -> "person taking decisive action", НЕ "bull horns").\n\n'
    'Фраза: "{text}"'
)

_calls_made = 0


def _cache_dir(video_dir):
    d = os.path.join(video_dir, "media_plan", "shot_director_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(video_dir, text):
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return os.path.join(_cache_dir(video_dir), h + ".json")


def _extract_queries(raw_text):
    """raw_text — сырой ответ модели (уже должен быть JSON-строкой из-за
    responseMimeType=application/json, но парсим защищённо — LLM иногда
    всё равно оборачивает в ```json несмотря на явный запрос)."""
    cleaned = raw_text.strip()
    m = re.search(r'\{.*\}', cleaned, re.S)
    if m:
        cleaned = m.group(0)
    parsed = json.loads(cleaned)
    queries = parsed.get("queries")
    if not isinstance(queries, list) or not queries:
        return None
    clean = []
    for q in queries:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if q and not re.search(r'[а-яА-ЯёЁ]', q):
            clean.append(q)
    return clean or None


def _call_gemini_prompt(prompt, api_key):
    """Общий HTTP-вызов Gemini — извлечено из _call_gemini(), чтобы
    enrich_atmospheric_queries() могла использовать тот же транспорт/парсинг
    с ДРУГИМ (не _PROMPT_TEMPLATE) промптом, не дублируя код запроса."""
    generation_config = {"responseMimeType": "application/json"}
    thinking_cfg = _thinking_config_for_model(SHOT_DIRECTOR_MODEL)
    if thinking_cfg is not None:
        generation_config["thinkingConfig"] = thinking_cfg
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }).encode("utf-8")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{SHOT_DIRECTOR_MODEL}:generateContent?key={api_key}")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=SHOT_DIRECTOR_TIMEOUT_SEC) as r:
        resp = json.loads(r.read().decode("utf-8"))
    raw = resp["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_queries(raw)


def _call_gemini(text, api_key):
    return _call_gemini_prompt(_PROMPT_TEMPLATE.format(text=text), api_key)


_ATMO_PROMPT_TEMPLATE = (
    'Ты — постановщик кадра для закадрового видео на русском языке. '
    'Визуальный мир этого ролика уже задан ДРУГИМИ кадрами того же раздела '
    'сценария (их авторские поисковые запросы): {context}.\n\n'
    'Дана ФРАЗА из сценария: "{text}". Её собственный поисковый запрос — '
    '"{own_query}" — технически релевантен фразе, но взят СЛИШКОМ БУКВАЛЬНО, '
    'в отрыве от темы всего ролика (см. другие запросы раздела выше) — так '
    'реальный монтажёр эту фразу НЕ поставил бы. Предложи 2-3 АЛЬТЕРНАТИВНЫХ '
    'английских поисковых запроса для Pexels/Pixabay (2-5 слов каждый, '
    'конкретная визуальная сцена, без кириллицы, без текста на кадре), '
    'которые: (1) остаются по смыслу верны именно ЭТОЙ фразе, (2) визуально '
    'принадлежат ТОМУ ЖЕ миру, что и остальные запросы раздела — то есть '
    'связывают содержание фразы с темой/предметами остальных кадров, а не '
    'берут первый напрашивающийся обобщённый предмет.\n\n'
    'Верни СТРОГО валидный JSON без пояснений и без markdown-разметки: '
    '{{"literal": true/false, "queries": ["query1", "query2", "query3"]}}.'
)


def _atmo_cache_path(video_dir, text, own_query, context_queries):
    h = hashlib.sha256(
        ("atmo|" + text + "|" + own_query + "|" + "|".join(sorted(context_queries))
         ).encode("utf-8")).hexdigest()[:24]
    return os.path.join(_cache_dir(video_dir), "atmo_" + h + ".json")


def reset_call_counter():
    """Тесты/повторный прогон в одном процессе — сбросить счётчик лимита."""
    global _calls_made
    _calls_made = 0


def enrich_atmospheric_queries(text, own_query, context_queries, video_dir):
    """Реальный, найденный вживую пробел (прямая жалоба пользователя на
    открывающий кадр хука: фраза "Пятнадцать килограммов" получила фото
    бытовых кухонных весов — технически релевантно ЧИСЛУ, но никак не
    связано с темой всего ролика про меч). direct_query() выше решает
    ДРУГУЮ задачу (совсем нет запроса — идиома/метафора без предметных
    слов); здесь запрос УЖЕ есть и технически подходит, проблема в том, что
    он взят СЛИШКОМ буквально, в отрыве от визуального мира остальных
    кадров раздела.

    own_query — уже назначенный авторский/смысловой запрос этого блока.
    context_queries — ОСТАЛЬНЫЕ запросы того же раздела (см. section_query_
    pool в pipeline_smart.py) — дают модели понять тему ролика без отдельного
    вызова за METADATA/TITLE (который на практике может быть плейсхолдером
    или кликбейтом, не описанием содержания — лишний источник ошибки).

    Возвращает список альтернативных запросов (str) или None (режим off /
    нет ключа / лимит исчерпан / ошибка) — вызывающий код ДОБАВЛЯЕТ их в
    пул кандидатов (extra_queries), никогда не заменяет own_query — тот же
    принцип "только плюс", что и у остального пайплайна: даже если LLM
    вернула что-то похуже, реальный отбор (relevance-гейт + Semantic Visual
    Director реранк по РЕАЛЬНОЙ фразе) их всё равно честно отсеет, слот не
    может стать ХУЖЕ, чем был без этого вызова — только получить более
    богатый пул кандидатов на выбор.

    Отдельный, ЗАМЕДЛЕННЫЙ кэш-неймспейс от direct_query() (префикс "atmo_"
    в имени файла в ТОМ ЖЕ media_plan/shot_director_cache/) — ключ кэша
    включает own_query И context_queries, поэтому смена состава запросов
    секции (человек отредактировал script.txt) сама инвалидирует кэш, не
    нужно чистить руками. Использует ТОТ ЖЕ счётчик лимита (_calls_made/
    SHOT_DIRECTOR_MAX_CALLS_PER_RUN), что и direct_query() — единый бюджет
    вызовов Gemini за прогон, не два независимых потолка."""
    global _calls_made
    if os.environ.get("SHOT_DIRECTOR_MODE", "off").strip().lower() != "on":
        return None
    context_queries = [q for q in (context_queries or []) if q and q != own_query]
    cache_file = _atmo_cache_path(video_dir, text, own_query, context_queries)
    if os.path.exists(cache_file):
        try:
            cached = json.load(open(cache_file, encoding="utf-8"))
            return cached.get("queries") or None
        except Exception:
            pass
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    if _calls_made >= SHOT_DIRECTOR_MAX_CALLS_PER_RUN:
        return None
    _calls_made += 1
    prompt = _ATMO_PROMPT_TEMPLATE.format(
        context=", ".join(f'"{q}"' for q in context_queries), text=text, own_query=own_query)
    try:
        queries = _call_gemini_prompt(prompt, api_key)
    except Exception as e:
        print(f"  [shot_director] Gemini (atmo) не ответил на \"{text[:40]}...\": {e}")
        return None
    if not queries:
        return None
    try:
        json.dump({"text": text, "own_query": own_query, "context_queries": context_queries,
                    "queries": queries}, open(cache_file, "w", encoding="utf-8"),
                   ensure_ascii=False, indent=2)
    except Exception:
        pass
    return queries


def direct_query(text, video_dir):
    """Возвращает лучший английский поисковый запрос (str) для фразы text,
    либо None (режим off / нет ключа / кэш-промах и лимит исчерпан /
    ошибка API/парсинга) — в любом случае вызывающий код обязан
    обработать None как "ничего не нашли", падая на прежнее поведение."""
    global _calls_made
    # Живой (не замороженный на момент импорта) читатель режима — pipeline_smart
    # уже гейтит САМ ИМПОРТ этого модуля тем же os.environ-чтением ДО вызова
    # (см. resolve_queries()), так что в проде это дублирующая защита, а не
    # единственная; но модуль-константа SHOT_DIRECTOR_MODE ниже вычисляется
    # ОДИН раз при импорте и не видит monkeypatch/переменные окружения,
    # выставленные после — прямой вызов direct_query() (как в тестах, или
    # любым будущим кодом, который импортирует модуль раньше, чем известен
    # режим) обязан проверять актуальное состояние, а не застывший снимок.
    if os.environ.get("SHOT_DIRECTOR_MODE", "off").strip().lower() != "on":
        return None
    cache_file = _cache_path(video_dir, text)
    if os.path.exists(cache_file):
        try:
            cached = json.load(open(cache_file, encoding="utf-8"))
            q = cached.get("queries")
            return q[0] if q else None
        except Exception:
            pass
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    if _calls_made >= SHOT_DIRECTOR_MAX_CALLS_PER_RUN:
        return None
    _calls_made += 1
    try:
        queries = _call_gemini(text, api_key)
    except Exception as e:
        print(f"  [shot_director] Gemini не ответил на \"{text[:40]}...\": {e}")
        return None
    if not queries:
        return None
    try:
        json.dump({"text": text, "queries": queries},
                   open(cache_file, "w", encoding="utf-8"),
                   ensure_ascii=False, indent=2)
    except Exception:
        pass
    return queries[0]


# ============================================================================
# VLM-АРБИТР (VLM_ARBITER_MODE=off/on, дефолт off) — по прямому запросу
# пользователя (29 августа, deep-audit): SAME_QUERY_BONUS в
# visual_director.compute_extra_score() (см. её докстринг) чинит СИМПТОМ —
# restорирует вес уже вложенного текст-текст решения semantic_query_
# assignment() поверх более слабого image-text CLIP-скоринга — но САМ
# скоринг остаётся эмбеддинг-косинусом, который в принципе не умеет понять
# композиционный/абстрактный смысл фразы ("меч весом 15 кг" — не "любой
# меч"), только сравнивать вектора. Пользователь прямо указал: это опять
# точечный фикс, а не поднятие потолка архитектуры. Единственный способ
# ДЕЙСТВИТЕЛЬНО поднять потолок без нового отдельного сервиса — заменить
# эмбеддинг-сравнение на РЕАЛЬНОЕ языковое понимание: показать модели
# (Gemini, тот же ключ, что уже подтверждён живым вызовом в этом модуле)
# саму фразу И картинки кандидатов, спросить "какая ДЕЙСТВИТЕЛЬНО подходит
# по смыслу" — то, что реально делает человек-монтажёр, а не formal
# similarity.
#
# СКОУП — ТОЛЬКО ХУК (осознанное, объявленное пользователю ограничение, не
# скрытый компромисс): free-tier бюджет Gemini ~20 вызовов/день НА МОДЕЛЬ
# (эмпирика ЧАСТЬ 14 CLAUDE.md) — арбитраж на КАЖДЫЙ слот целого эпизода
# (100+ кадров на реальный 60-минутный ролик) исчерпал бы дневной лимит на
# одном рендере. Хук — 5-8 слотов, самый решающий по удержанию участок
# ролика (ЧАСТЬ 9 CLAUDE.md) — уже даёт заметный эффект в разумном бюджете.
# Расширение на весь эпизод возможно, но требует ЛИБО платного тарифа Gemini
# (реальные деньги — ЧАСТЬ 1, не решается без пользователя), ЛИБО смирения
# с тем, что бюджет исчерпается на середине рендера — оба варианта требуют
# явного решения пользователя, не мой выбор по умолчанию.
#
# Использует ТОТ ЖЕ счётчик _calls_made/SHOT_DIRECTOR_MAX_CALLS_PER_RUN, что
# и direct_query()/enrich_atmospheric_queries() — единый бюджет вызовов
# Gemini за прогон (та же причина: одна и та же квота ключа/модели, два
# независимых потолка позволили бы двум фичам вместе тихо съесть вдвое
# больше дневного лимита).
#
# ЧЕСТНО: VLM-суждение — тоже не "истина", а более сильная, но НЕ
# безошибочная эвристика (языковая модель может ошибиться так же, как
# человек может ошибиться, глядя на кадр) — это подъём потолка, не
# гарантия. Живой ручной прогон на реальном ключе ОБЯЗАТЕЛЕН перед первым
# включением VLM_ARBITER_MODE=on (тот же принцип, что уже применён к
# speech_generate.py/этому же модулю выше) — контракт запроса/ответа ниже
# проверен только на моке до этого прогона.
VLM_ARBITER_TIMEOUT_SEC = 25   # тяжелее текстового промпта (несколько картинок в теле запроса)

_ARBITER_PROMPT_TEMPLATE = (
    'Ты — опытный монтажёр видео, подбираешь кадр под конкретную фразу '
    'закадрового текста для документального ролика на русском языке.\n\n'
    'ФРАЗА: "{text}"\n\n'
    'Ниже приложено {n} кандидатов-изображений по порядку (картинка 1, '
    'картинка 2, ...). Выбери НОМЕР картинки, которая РЕАЛЬНО, по смыслу '
    '(не по формальному сходству темы) лучше всего иллюстрирует именно эту '
    'фразу — так, как выбрал бы человек, прочитавший фразу и посмотревший '
    'на картинки, а не по совпадению ключевых слов с общей темой ролика. '
    'Если ни одна картинка реально не подходит — верни 0.\n\n'
    'Верни СТРОГО валидный JSON без пояснений и без markdown-разметки: '
    '{{"choice": <целое число 0..{n}>, "reason": "коротко, по-русски"}}.'
)


def _mime_type_for(path):
    return "image/png" if os.path.splitext(path)[1].lower() == ".png" else "image/jpeg"


def _extract_choice(raw_text, n):
    cleaned = raw_text.strip()
    m = re.search(r'\{.*\}', cleaned, re.S)
    if m:
        cleaned = m.group(0)
    parsed = json.loads(cleaned)
    choice = parsed.get("choice")
    if not isinstance(choice, (int, float)) or isinstance(choice, bool):
        return None
    choice = int(choice)
    if choice < 0 or choice > n:
        return None
    return choice   # 0 == "ни одна не подходит", 1..n == 1-based индекс


def _call_gemini_vision(prompt, image_paths, api_key):
    """Тот же транспорт/JSON-режим, что _call_gemini_prompt(), плюс
    inline_data-части картинок ПЕРЕД текстом — порядок картинок в частях
    запроса соответствует порядку "картинка N" в промпте, это и даёт модели
    однозначно сослаться на конкретного кандидата номером."""
    parts = []
    for p in image_paths:
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        parts.append({"inline_data": {"mime_type": _mime_type_for(p), "data": data}})
    parts.append({"text": prompt})
    generation_config = {"responseMimeType": "application/json"}
    thinking_cfg = _thinking_config_for_model(SHOT_DIRECTOR_MODEL)
    if thinking_cfg is not None:
        generation_config["thinkingConfig"] = thinking_cfg
    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
    }).encode("utf-8")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{SHOT_DIRECTOR_MODEL}:generateContent?key={api_key}")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=VLM_ARBITER_TIMEOUT_SEC) as r:
        resp = json.loads(r.read().decode("utf-8"))
    raw = resp["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_choice(raw, len(image_paths))


def _arbiter_cache_path(video_dir, text, candidate_ids):
    # candidate_ids в ПОРЯДКЕ (не sorted, в отличие от _atmo_cache_path) —
    # закэшированный choice это 1-based ИНДЕКС в этот порядок, менять
    # порядок и держать тот же кэш-файл значило бы отдать неверную картинку.
    h = hashlib.sha256(
        ("arbiter|" + text + "|" + "|".join(str(c) for c in candidate_ids)
         ).encode("utf-8")).hexdigest()[:24]
    return os.path.join(_cache_dir(video_dir), "arbiter_" + h + ".json")


def _resolve_choice(choice, candidate_paths):
    if not isinstance(choice, int) or choice <= 0 or choice > len(candidate_paths):
        return None
    return candidate_paths[choice - 1]


def arbitrate_hook_candidates(text, candidate_paths, candidate_ids, video_dir):
    """VLM-арбитр (см. блок-комментарий выше) — реальное языковое+визуальное
    суждение поверх уже готового шорт-листа (2-3 уже прошедших все гейты
    кандидата, см. вызывающий код в pipeline_smart.py — их СОБИРАЕТ, не
    скачивает заново), а не ещё один numeric-скоринг. text — РЕАЛЬНАЯ фраза
    блока (не дополненная соседями semantic_context_text — VLM, в отличие
    от CLIP, понимает короткую фразу саму по себе).

    candidate_ids — стабильный идентификатор каждого пути в candidate_paths
    (тот же порядок, Pexels photo/video id) — используется ТОЛЬКО для ключа
    кэша (temp-файлы кандидатов эфемерны и меняют имя между прогонами, id
    остаётся тем же).

    Возвращает путь ИЗ candidate_paths (не индекс) — победивший кандидат,
    либо None: режим off / нет ключа / лимит вызовов исчерпан / кандидатов
    меньше 2 (нечего арбитрировать) / ошибка сети/парсинга / модель
    вернула 0 ("ни один не подходит"). ЛЮБОЙ None — вызывающий код обязан
    остаться на уже вычисленном (director/base) победителе, не на пустом
    слоте — тот же fail-open принцип, что у direct_query()/
    enrich_atmospheric_queries()."""
    global _calls_made
    if os.environ.get("VLM_ARBITER_MODE", "off").strip().lower() != "on":
        return None
    if not candidate_paths or len(candidate_paths) < 2:
        return None
    if len(candidate_paths) != len(candidate_ids):
        return None
    cache_file = _arbiter_cache_path(video_dir, text, candidate_ids)
    if os.path.exists(cache_file):
        try:
            cached = json.load(open(cache_file, encoding="utf-8"))
            return _resolve_choice(cached.get("choice"), candidate_paths)
        except Exception:
            pass
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    if _calls_made >= SHOT_DIRECTOR_MAX_CALLS_PER_RUN:
        return None
    _calls_made += 1
    prompt = _ARBITER_PROMPT_TEMPLATE.format(text=text, n=len(candidate_paths))
    try:
        choice = _call_gemini_vision(prompt, candidate_paths, api_key)
    except Exception as e:
        print(f"  [shot_director] VLM-арбитр не ответил на \"{text[:40]}...\": {e}")
        return None
    if choice is None:
        return None
    try:
        json.dump({"text": text, "candidate_ids": candidate_ids, "choice": choice},
                   open(cache_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass
    return _resolve_choice(choice, candidate_paths)
