#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wikimedia Commons как источник кадра — только то, что свободно без
указания автора.

ЗАЧЕМ. Замер 24.09 на эпизоде 94 (docs/quality/RESEARCHER_PROTO_EP94.md):
на фразах про действие («рыцарь падает», «конница на пехоту», «стрела по
нагруднику») нужного кадра не было ни в первых 20, ни на местах 21-200
пула — его просто не приносили источники. Стоки дают реконструкции с
толпой, музеи — предметы на белом фоне. А изображения этих самых событий,
нарисованные современниками, лежат в Commons: Азенкур в Сент-Олбанской
хронике, Креси у Фруассара, фехтовальные трактаты Тальхоффера и
Валлерштайнского кодекса. До этого модуля пайплайн Commons напрямую не
спрашивал, а Openverse берётся только с меткой CC0 — старые миниатюры там
помечены как общественное достояние по возрасту и не проходили.

ЛИЦЕНЗИЯ — fail-closed, по метаданным САМОГО файла (extmetadata, их пишет
шаблон лицензии на странице файла):
  * License — ровно `pd` или `cc0`;
  * AttributionRequired — не `true`;
  * Restrictions — пусто (права на изображение личности, товарные знаки,
    ограничения страны — всё это отказ, даже при свободной лицензии).
Правило канала (решение владельца 10.09): в ролик идёт только то, что не
требует указывать автора в описании. PD и CC0 без атрибуции ему
соответствуют; CC BY/BY-SA — нет, и они отсекаются здесь же. Нет
метаданных — нет кандидата.

ВЕЖЛИВОСТЬ. Commons отвечает 429 на всплеск запросов (записано в CLAUDE.md
по живым прогонам). Поэтому: один регулятор хоста (source_health —
интервал, пауза и замедление на 429), дисковый кэш ответов поиска, превью
и рабочий файл — через Special:FilePath заданной ширины (оригиналы
upload.wikimedia.org прямо просят не качать), подпись клиента по политике
WMF с контактом.

Кандидат — в общей форме (как у Openverse/музеев): id `commons:<pageid>`
(числовой id страницы стабилен и не содержит двоеточий и пробелов, в
отличие от имени файла), alt — название и описание файла, url — страница
файла, src.medium — превью 640, src.large2x — рабочий файл 2000 px,
_commons_meta — лицензия, автор, дата, страница (след происхождения)."""
import hashlib
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import source_health

API = "https://commons.wikimedia.org/w/api.php"
FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
# Подпись клиента по политике WMF (https://meta.wikimedia.org/wiki/User-Agent_policy):
# название, адрес проекта, контакт. Без контакта WMF вправе отвечать отказом.
# Формат — как в примере самих правил: «Имя/версия (адрес; ...)». Контакт —
# адрес проекта, а не почта: почту владельца сторонним сервисам не отдаём.
# Лимит для подписанных клиентов — 200 запросов в минуту (без подписи —
# 10), не больше 3 одновременно; 429 — ждать Retry-After, иначе >= 5 с.
USER_AGENT = ("PipelineMachineVideo/1.0 "
              "(https://github.com/hellokittysoullja-bit/PipelineMachineVideo_AUTO; "
              "documentary b-roll research)")
PREVIEW_WIDTH = 640
WORK_WIDTH = 2000          # кадр 1920x1080 плюс запас на наезд камеры
SEARCH_LIMIT = 40          # страниц на запрос; свободных среди них — около половины
MIN_SHORT_SIDE = 400       # меньше — мыло даже после вписывания в кадр
CACHE_TTL_SEC = 30 * 24 * 3600
FREE_LICENSES = frozenset({"pd", "cc0"})
MIMES = frozenset({"image/jpeg", "image/png", "image/tiff", "image/webp"})
# Интервал и пауза — тот же порядок, что у остальных хостов Викимедиа в
# pipeline_smart (замер 13-14.09: всплески дают 429, редкие запросы — нет).
HOST = source_health.host("commons", interval=2.0, max_interval=8.0, cooldown_sec=60.0)

CACHE_DIR = os.environ.get("COMMONS_CACHE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_commons_cache")

_TAG_RE = re.compile(r"<[^>]+>")
STATS = {"requests": 0, "cache_hits": 0, "results": 0, "kept": 0,
         "rejected_license": 0, "rejected_size": 0, "rejected_mime": 0, "errors": 0}


def reset_stats():
    for k in STATS:
        STATS[k] = 0


def plain(value):
    """Текст поля extmetadata без HTML и лишних пробелов."""
    text = html.unescape(_TAG_RE.sub(" ", str(value or "")))
    return " ".join(text.split())


def _meta(m, key):
    return plain(((m or {}).get(key) or {}).get("value"))


def is_free_without_attribution(extmetadata):
    """Можно ли показать файл, не указывая автора: PD/CC0, атрибуция не
    обязательна, ограничений нет. Любая неясность — False."""
    m = extmetadata or {}
    lic = _meta(m, "License").lower()
    if lic not in FREE_LICENSES:
        return False
    if _meta(m, "AttributionRequired").lower() == "true":
        return False
    if _meta(m, "Restrictions"):
        return False
    if _meta(m, "NonFree").lower() == "true":
        return False
    return True


def file_url(title, width):
    """Ссылка на файл заданной ширины (Special:FilePath) по имени файла."""
    name = title[5:] if title.startswith("File:") else title
    return FILEPATH + urllib.parse.quote(name.replace(" ", "_")) + f"?width={int(width)}"


def to_candidate(page):
    """Страница файла из ответа API -> кандидат пула, или None (не прошла
    лицензию, размер, формат). Причина отказа считается в STATS."""
    ii = (page.get("imageinfo") or [{}])[0]
    m = ii.get("extmetadata") or {}
    if not is_free_without_attribution(m):
        STATS["rejected_license"] += 1
        return None
    if ii.get("mime") not in MIMES:
        STATS["rejected_mime"] += 1
        return None
    if min(int(ii.get("width") or 0), int(ii.get("height") or 0)) < MIN_SHORT_SIDE:
        STATS["rejected_size"] += 1
        return None
    title = page.get("title") or ""
    name = re.sub(r"\.(jpe?g|png|tiff?|webp)$", "", title[5:] if title.startswith("File:") else title,
                  flags=re.I)
    desc = _meta(m, "ImageDescription") or _meta(m, "ObjectName")
    alt = name if not desc else f"{name}. {desc[:300]}"
    return {
        "id": f"commons:{page.get('pageid')}",
        "alt": alt,
        "url": ii.get("descriptionurl") or ("https://commons.wikimedia.org/wiki/" +
                                            urllib.parse.quote(title.replace(" ", "_"))),
        "width": ii.get("width"), "height": ii.get("height"),
        "src": {"medium": file_url(title, PREVIEW_WIDTH), "large2x": file_url(title, WORK_WIDTH)},
        "_download_headers": {"User-Agent": USER_AGENT},
        "_commons_meta": {
            "source": "wikimedia_commons",
            "license": _meta(m, "LicenseShortName") or _meta(m, "License"),
            "artist": _meta(m, "Artist")[:200] or None,
            "date": _meta(m, "DateTimeOriginal")[:80] or None,
            "credit": _meta(m, "Credit")[:200] or None,
            "file_page": ii.get("descriptionurl"),
        },
    }


def _cache_path(query, limit):
    key = json.dumps(["commons", 1, query, int(limit)], ensure_ascii=False)
    return os.path.join(CACHE_DIR, hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")


def _cache_get(query, limit):
    path = _cache_path(query, limit)
    try:
        if time.time() - os.path.getmtime(path) > CACHE_TTL_SEC:
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["pages"] if data.get("query") == query else None
    except (OSError, ValueError, KeyError):
        return None


def _cache_put(query, limit, pages):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = _cache_path(query, limit)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"query": query, "pages": pages}, f, ensure_ascii=False)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


def retry_after_sec(error):
    """Пауза, которую назвал сам сервис (заголовок Retry-After в секундах),
    или None: заголовка нет или он датой, а не числом."""
    try:
        value = (error.headers or {}).get("Retry-After")
        return float(value) if value is not None and str(value).strip().isdigit() else None
    except (AttributeError, TypeError, ValueError):
        return None


def _fetch_pages(query, limit):
    """Сырые страницы файлов поиска (с метаданными) — из кэша или из API.
    В кэш кладутся СЫРЫЕ страницы, а не кандидаты: правило лицензии и
    форма кандидата могут поменяться, ответ поиска — нет."""
    cached = _cache_get(query, limit)
    if cached is not None:
        STATS["cache_hits"] += 1
        return cached
    params = {"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
              "gsrlimit": int(limit), "prop": "imageinfo",
              "iiprop": "url|size|mime|extmetadata", "iiextmetadatalanguage": "en"}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        if HOST.cooling():
            time.sleep(HOST.cooldown_left())
        HOST.wait()
        STATS["requests"] += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            HOST.succeeded()
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 2:
                HOST.throttled(retry_after=retry_after_sec(e))
                continue
            raise
    pages = sorted(((data.get("query") or {}).get("pages") or {}).values(),
                   key=lambda p: p.get("index", 0))
    _cache_put(query, limit, pages)
    return pages


def search(query, limit=SEARCH_LIMIT):
    """Кандидаты Commons по запросу, в порядке релевантности поиска.
    Исключения наружу: fail-open решает вызывающий (как у остальных
    источников пула)."""
    query = " ".join(str(query or "").split())
    if not query:
        return []
    pages = _fetch_pages(query, limit)
    out = []
    for page in pages:
        STATS["results"] += 1
        cand = to_candidate(page)
        if cand is not None:
            out.append(cand)
    STATS["kept"] += len(out)
    return out
