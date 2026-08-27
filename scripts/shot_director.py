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


def _call_gemini(text, api_key):
    generation_config = {"responseMimeType": "application/json"}
    thinking_cfg = _thinking_config_for_model(SHOT_DIRECTOR_MODEL)
    if thinking_cfg is not None:
        generation_config["thinkingConfig"] = thinking_cfg
    body = json.dumps({
        "contents": [{"parts": [{"text": _PROMPT_TEMPLATE.format(text=text)}]}],
        "generationConfig": generation_config,
    }).encode("utf-8")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{SHOT_DIRECTOR_MODEL}:generateContent?key={api_key}")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=SHOT_DIRECTOR_TIMEOUT_SEC) as r:
        resp = json.loads(r.read().decode("utf-8"))
    raw = resp["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_queries(raw)


def reset_call_counter():
    """Тесты/повторный прогон в одном процессе — сбросить счётчик лимита."""
    global _calls_made
    _calls_made = 0


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
