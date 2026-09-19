"""Викисклад (Wikimedia Commons) как ПРЯМОЙ источник кандидатов — знание о
содержимом для ЛЮБОЙ ниши, без ключа и без атрибуции.

ЗАЧЕМ ОТДЕЛЬНЫЙ ИСТОЧНИК, а не «и так есть через Openverse». Замер 18.09 на
живых API: Openverse предложил кофейному эпизоду 158 кандидатов и выиграл 3
слота из 50 (6%), потому что его индекс отдаёт ТОЛЬКО cc0-подмножество и
только через свой поиск. Прямой запрос к Викискладу на тех же нишах даёт
**59 кандидатов PD/CC0 с короткой стороной >=1080 из 172** (кофе, Марс,
глубоководные) — то есть треть выдачи проходит политику канала, и это
материал, которого в пуле сегодня нет вовсе.

ПОЧЕМУ ЭТО ВАЖНО ИМЕННО СЕЙЧАС. Замер того же дня (docs/quality/
so400m_as_judge.json) показал, что усиление модели-СУДЬИ не работает: у
брака и годных кадров распределения перекрываются целиком, а 11 браков из 17
— это анахронизм или чужая культура, то есть вопрос ЗНАНИЯ, а не сходства
пикселей. Единственный механизм этого репозитория, который такой брак реально
режет, — паспорт музейного предмета (дата и культура из каталога хранителя).
Но музеи отвечают только исторической нише и только при объявленном окне
эпохи. Викисклад — тот же принцип для ЛЮБОЙ темы: у файла есть
человекочитаемое название, категории и машиночитаемая лицензия, то есть
внешнее знание о содержимом кадра, а не догадка по пикселям.

ЛИЦЕНЗИЯ — FAIL-CLOSED, ДВАЖДЫ. Берётся только то, что сам Викисклад пометил
как общественное достояние или CC0: `LicenseShortName` обязан совпасть со
списком COMMONS_SAFE_LICENSES ТОЧНО (не «содержит»), плюс отдельно
проверяется, что поле вообще есть — 5 файлов из 172 в замере пришли без
лицензии вообще, и «не сказано» здесь означает «не берём». CC BY/BY-SA не
берутся ни в каком виде: у пайплайна нет механизма, который собирает
атрибуции в описание ролика, и это осознанное ограничение канала, а не
недосмотр (та же причина, по которой они не берутся из Openverse). Цена
названа числом: в замере на кофе 11 кандидатов из 33 были CC BY 4.0 и ушли
мимо.

СНИМКИ БЕРУТСЯ ТОЛЬКО ЧЕРЕЗ ОФИЦИАЛЬНЫЙ `Special:FilePath?width=`, и это
исправление СВОЕЙ ЖЕ первой версии, найденное живым прогоном. Первая версия
получала от API один `thumburl` шириной 2000 и собирала из него превью
регуляркой (`/2000px-` -> `/640px-`). Замер: из 10 кандидатов скачалась
ОДНА. Разбор по ответам — собранный URL отвечает HTTP 400, причём и с
query-строкой, и без неё (проверено обеими формами), то есть путь к
thumbnail на upload.wikimedia.org нельзя конструировать подстановкой: он
зависит от внутренней раскладки хранилища. Успевали скачаться ровно те
кандидаты, у которых оригинал уже был уже 2000, и подстановка не срабатывала
вовсе.

`Special:FilePath` — тот самый официальный путь, который в CLAUDE.md уже
записан как проверенный живьём после истории с 429 на оригиналах (там же
цена: 49 сорванных скачек). Замер после перехода: 6 запросов из 6, обе
ширины, оба размера разумные (640 -> 100-280 КБ, 2000 -> 1.5-3.7 МБ).
Оригинал не запрашивается никогда.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"

# Строка по политике WMF: инструмент и ссылка. Та же, что pipeline_smart
# использует при скачивании с хостов Викимедиа — вторая копия разошлась бы.
USER_AGENT = ("FacelessPipeline/1.0 "
              "(https://github.com/hellokittysoullja-bit/PipelineMachineVideo_AUTO)")

# Точные значения LicenseShortName, означающие отсутствие требований. Список
# закрытый и сравнение точное: «CC BY-SA 4.0» содержит «CC BY», и проверка
# «по вхождению» пропустила бы ровно то, что запрещено политикой канала.
COMMONS_SAFE_LICENSES = frozenset({
    "Public domain", "PD", "CC0", "No restrictions", "PDM-owner",
})

# Короткая сторона: кадр идёт в 1920x1080 и ещё проходит зум Ken Burns.
MIN_SHORT_SIDE = 1080
SEARCH_LIMIT = 40
PROBE_WIDTH = 640        # оценка гейтами — см. candidate_probe_url
RENDER_WIDTH = 2000      # рабочий файл победителя

# Версия ПРАВИЛ отбора этого источника. Входит в ключ дискового кэша и в
# подпись отбора: кэш хранит ГОТОВЫХ кандидатов, и без версии ужесточение
# лицензии или порога разрешения молча отдавалось бы из кэша по старым
# правилам (тот же класс, ради которого заведена candidate_gate_signature).
COMMONS_SOURCE_VERSION = 1

CACHE_DIR = os.environ.get("COMMONS_CACHE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_commons_cache")
CACHE_TTL_SEC = 30 * 24 * 3600

# Вежливость к хосту: у Викисклада реальный лимит, и 429 в замере 18.09
# прилетел уже на седьмом запросе подряд. Интервал — на процесс.
MIN_REQUEST_INTERVAL = 1.0
_RATE_LOCK = threading.Lock()
_NEXT_REQUEST_AT = [0.0]

FETCH_STATS = {"queries": 0, "http_errors": 0, "retries": 0, "cache_hits": 0,
               "rejected_license": 0, "rejected_small": 0, "kept": 0}


def reset_fetch_stats():
    for k in FETCH_STATS:
        FETCH_STATS[k] = 0


def _throttle():
    with _RATE_LOCK:
        now = time.monotonic()
        slot = max(now, _NEXT_REQUEST_AT[0])
        _NEXT_REQUEST_AT[0] = slot + MIN_REQUEST_INTERVAL
    delay = slot - time.monotonic()
    if delay > 0:
        time.sleep(delay)


RETRY_STATUSES = (429, 503)
RETRY_PAUSE_SEC = 5.0


def _api(params, timeout=45, _retries=1):
    """Один повтор на 429/503 — не оптимизм, а замер.

    Прямой прогон 18.09: три запроса подряд после моих же тестовых залпов
    дали два HTTP 429, и ровно те же два запроса прошли чисто через пять
    секунд. Без повтора это выглядело бы как «Викисклад не знает про кофе» —
    ровно тот молчаливый ноль, против которого в этом репозитории заведены
    счётчики вклада источников.
    """
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(_retries + 1):
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in RETRY_STATUSES and attempt < _retries:
                FETCH_STATS["retries"] += 1
                time.sleep(RETRY_PAUSE_SEC)
                continue
            raise


def license_is_safe(extmetadata):
    """Лицензия годна БЕЗ атрибуции — fail-closed.

    Поля нет, оно пустое, значение не из списка — нет. Именно точное
    сравнение: «CC BY-SA 4.0» содержит «CC BY» и «по вхождению» прошло бы.
    """
    if not isinstance(extmetadata, dict):
        return False
    raw = (extmetadata.get("LicenseShortName") or {}).get("value")
    if not raw or not isinstance(raw, str):
        return False
    name = re.sub(r"\s+", " ", raw).strip()
    if name not in COMMONS_SAFE_LICENSES:
        return False
    # Отдельная страховка: Викисклад отмечает несвободные оговорки
    # (товарный знак, право на изображение человека) отдельным полем.
    restrictions = (extmetadata.get("Restrictions") or {}).get("value")
    if restrictions and str(restrictions).strip():
        return False
    return True


def file_path_url(file_name, width):
    """Официальная точка выдачи снимка нужной ширины.

    Собирать путь к thumbnail подстановкой в URL НЕЛЬЗЯ — проверено живьём
    (см. докстринг модуля): такой URL отвечает 400. Здесь запрашивается ровно
    та ширина, которая нужна, и остальное решает сам источник.
    """
    return ("https://commons.wikimedia.org/wiki/Special:FilePath/"
            + urllib.parse.quote(file_name) + f"?width={int(width)}")

def _title_to_text(title):
    """«File:Coffee cherries on bush.jpg» -> «Coffee cherries on bush».

    Текст нужен не для красоты: filter_alt_blocklist/pexels_candidate_text
    судят кандидата ПО ТЕКСТУ, и у Викисклада имя файла — единственное
    человекочитаемое описание содержимого, которое приходит вместе с поиском.
    """
    t = re.sub(r"^File:", "", str(title or ""))
    t = re.sub(r"\.(jpe?g|png|tiff?|webp|gif)$", "", t, flags=re.I)
    return t.replace("_", " ").strip()


def _disk_cache_path(query):
    key = "%s|v%d|%d|%d" % (query.strip().lower(), COMMONS_SOURCE_VERSION,
                            MIN_SHORT_SIDE, SEARCH_LIMIT)
    import hashlib
    return os.path.join(CACHE_DIR, hashlib.md5(key.encode("utf-8")).hexdigest() + ".json")


def _cache_get(query):
    p = _disk_cache_path(query)
    try:
        if os.path.exists(p) and (time.time() - os.path.getmtime(p)) < CACHE_TTL_SEC:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _cache_put(query, results):
    # В кэш попадают только НЕПУСТЫЕ ответы: временный отказ источника иначе
    # заморозился бы на месяц (тот же урок, что у музейного кэша).
    if not results:
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_disk_cache_path(query), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
    except Exception:
        pass


def search_commons(query, limit=SEARCH_LIMIT):
    """Кандидаты Викисклада в ФОРМЕ PEXELS-КАНДИДАТА.

    Форма та же, что у музеев и Openverse, — чтобы конкурировать в ОДНОМ
    пуле под ОДНИМИ гейтами, а не жить отдельной веткой отбора со своими
    правилами.
    """
    query = (query or "").strip()
    if not query:
        return []
    cached = _cache_get(query)
    if cached is not None:
        FETCH_STATS["cache_hits"] += 1
        return cached
    FETCH_STATS["queries"] += 1
    try:
        data = _api({
            "action": "query", "format": "json", "formatversion": "2",
            "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": "6", "gsrlimit": str(int(limit)),
            # Категории приходят ТЕМ ЖЕ запросом — ни одного лишнего вызова.
            "prop": "imageinfo|categories",
            "cllimit": "max", "clshow": "!hidden",
            "iiprop": "url|size|extmetadata",
        })
    except Exception:
        FETCH_STATS["http_errors"] += 1
        return []

    out = []
    for page in ((data.get("query") or {}).get("pages") or []):
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        ii = infos[0]
        if not license_is_safe(ii.get("extmetadata")):
            FETCH_STATS["rejected_license"] += 1
            continue
        w, h = int(ii.get("width") or 0), int(ii.get("height") or 0)
        if min(w, h) < MIN_SHORT_SIDE:
            FETCH_STATS["rejected_small"] += 1
            continue
        title = page.get("title") or ""
        cats = [re.sub(r"^Category:", "", c.get("title") or "")
                for c in (page.get("categories") or [])]
        cats = [c for c in cats if c]
        file_name = re.sub(r"^File:", "", str(title)).replace(" ", "_")
        if not file_name:
            continue
        render_url = file_path_url(file_name, RENDER_WIDTH)
        probe_url = file_path_url(file_name, PROBE_WIDTH)
        out.append({
            "id": "commons:%s" % page.get("pageid"),
            # ТЕКСТ КАНДИДАТА = имя файла + КАТЕГОРИИ. Категории Викисклада
            # проставляет человек, и это единственное в этом источнике
            # описание СОДЕРЖИМОГО, а не набора букв в имени файла: «Coffea
            # arabica (fruit)», «Photos by the Spirit rover», «Mars (planet)».
            # Замер 18.09: категории есть у 15 файлов из 16, и приходят тем же
            # запросом — то есть канал бесплатный. Он важен не сам по себе:
            # filter_alt_blocklist()/pexels_candidate_text() судят кандидата ПО
            # ТЕКСТУ, и до этого им доставалось только имя файла.
            "alt": ", ".join([_title_to_text(title)] + cats),
            "url": ii.get("descriptionurl") or "",
            "width": w, "height": h,
            "src": {"large2x": render_url, "large": render_url,
                    "medium": probe_url, "small": probe_url},
            "_download_headers": {"User-Agent": USER_AGENT},
            "_commons_meta": {
                "title": title,
                "license": (ii.get("extmetadata", {}).get("LicenseShortName") or {}).get("value"),
                "page": ii.get("descriptionurl") or "",
                "width": w, "height": h,
                "categories": cats,
            },
        })
    FETCH_STATS["kept"] += len(out)
    _cache_put(query, out)
    return out


if __name__ == "__main__":
    import sys
    for q in sys.argv[1:] or ["coffee cherries harvest"]:
        res = search_commons(q)
        print(f"{q}: {len(res)} кандидатов")
        for r in res[:5]:
            m = r["_commons_meta"]
            print(f"   {r['id']:18s} {m['width']}x{m['height']:5d} {m['license']:15s} {r['alt'][:50]}")
    print("статистика:", FETCH_STATS)
