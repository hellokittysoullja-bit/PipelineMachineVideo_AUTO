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

ПУСТАЯ ВЫДАЧА — ЧАЩЕ ЛИШНЕЕ СЛОВО, ЧЕМ ОТСУТСТВИЕ ПРЕДМЕТА. Поиск Commons
ищет по И, и на четырёх-пяти словах половина запросов возвращает ноль (кэш
всех сессий: 3 слова — 31% пустых, 4 — 47%, 5 — 54%, 6 — 69%). Живой случай
28.09 (эпизод 94): запрос `talhoffer fechtbuch dagger armour 1467` дал ноль,
хотя планшеты Тальхоффера с боем на кинжалах в Commons есть. Поэтому при
пустом результате точного запроса `search()` пробует до трёх более общих
(relaxed_queries): без чисел и слов кадра, без последнего слова, без двух
последних; первая непустая побеждает, а найденное добавляется в пул как
всегда — под теми же лицензионными и смысловыми проверками.

Кандидат — в общей форме (как у Openverse/музеев): id `commons:<pageid>`
(числовой id страницы стабилен и не содержит двоеточий и пробелов, в
отличие от имени файла), alt — название и описание файла, url — страница
файла, src.medium — превью (желаемая ширина 640), src.large2x — рабочий
файл (желаемая 2000 px); обе ширины приводятся к стандартным и не шире
оригинала (thumb_width), _commons_meta — лицензия, автор, дата, страница
(след происхождения)."""
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
# Стандартные ширины миниатюр Викимедиа (mediawiki.org, Common thumbnail
# sizes; правило T414805): запрос через PHP (Special:FilePath, imageinfo)
# округляется ВВЕРХ до ближайшей из них, прямой — отклоняется, если ширины
# нет в списке.
THUMB_STEPS = (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)
SEARCH_LIMIT = 40          # страниц на запрос; свободных среди них — около половины
MIN_SHORT_SIDE = 400       # меньше — мыло даже после вписывания в кадр
CACHE_TTL_SEC = 30 * 24 * 3600
FREE_LICENSES = frozenset({"pd", "cc0"})
MIMES = frozenset({"image/jpeg", "image/png", "image/tiff", "image/webp"})
# Ослабление запроса, когда точный не дал ни одного годного кандидата. Поиск
# Commons — И по всем словам: чем длиннее запрос, тем чаще пустая выдача
# (кэш всех сессий, 1438 запросов: три слова — 31% пустых, четыре — 47%,
# пять — 54%, шесть — 69%).
RELAX_MIN_WORDS = 3        # короче — ослаблять нечего: два слова уже минимум
RELAX_KEEP_WORDS = 2       # короче двух слов не ослабляем: «knight» даёт медали
RELAX_MAX_VARIANTS = 3     # потолок лишних запросов к API на один запрос слота
# Слова кадра, а не предмета: ракурс, носитель. Убираются первыми — они
# режут выдачу в ноль, не неся ни эпохи, ни предмета. Только камера и
# носитель, ни одного слова какой-либо ниши.
RELAX_FRAMING_WORDS = frozenset({
    "closeup", "close-up", "close", "up", "macro", "detail", "shot", "view",
    "angle", "wide", "footage", "video", "animation",
})
# Интервал и пауза — тот же порядок, что у остальных хостов Викимедиа в
# pipeline_smart (замер 13-14.09: всплески дают 429, редкие запросы — нет).
HOST = source_health.host("commons", interval=2.0, max_interval=8.0, cooldown_sec=60.0)

CACHE_DIR = os.environ.get("COMMONS_CACHE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_commons_cache")

_TAG_RE = re.compile(r"<[^>]+>")
STATS = {"requests": 0, "cache_hits": 0, "results": 0, "kept": 0,
         "rejected_license": 0, "rejected_size": 0, "rejected_mime": 0, "errors": 0,
         "relaxed_tries": 0, "relaxed_hits": 0}


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


def thumb_width(native, target):
    """Ширина миниатюры, о которой просим сервер: стандартная и НЕ шире
    оригинала.

    Живой промах 28.09 (эпизод 94, слот про падающего рыцаря): рабочий файл
    просился шириной 2000, сервер округляет её вверх до 3840, а у файла
    ширина меньше. Замер по файлам из реальной выдачи (одна и та же сессия,
    вперемешку):

        родная ширина   width=2000              самая большая стандартная <= родной
        1666  Creci     429, Retry-After 600    1280: 200 OK, 318 КБ
        2947  Uccello   429, Retry-After 600    1920: 200 OK, 704 КБ
        2024  Uccello   429, Retry-After 600    1920: 200 OK, 892 КБ
        1066  Uccello   429, Retry-After 600    960:  200 OK, 241 КБ

    Отказ с паузой 600 с закреплён за адресом, а не за хостом, поэтому
    кандидат просто терялся при скачке (единственный подходящий кадр слота —
    миниатюра «Креси» — до экрана не дошёл). То же с превью: ширина 640
    округляется до 960, и у 17% кандидатов выдачи родная ширина меньше.

    Правило: берём ближайшую стандартную ширину не меньше желаемой, если она
    не выходит за оригинал; иначе — самую большую стандартную не больше
    оригинала. Родная ширина неизвестна — просим как просили (поведение
    вызовов без сведений об оригинале не меняется)."""
    try:
        target = int(target)
    except (TypeError, ValueError):
        return PREVIEW_WIDTH
    try:
        native = int(native or 0)
    except (TypeError, ValueError):
        native = 0
    if native <= 0:
        return target
    up = next((s for s in THUMB_STEPS if s >= target), None)
    if up is not None and up <= native:
        return up
    down = [s for s in THUMB_STEPS if s <= native]
    return down[-1] if down else native


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
    native_w = int(ii.get("width") or 0)
    if min(native_w, int(ii.get("height") or 0)) < MIN_SHORT_SIDE:
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
        # Ширина — стандартная и не шире оригинала (см. thumb_width): иначе
        # сервер округляет запрос за пределы файла и отвечает 429 на 600 с.
        "src": {"medium": file_url(title, thumb_width(native_w, PREVIEW_WIDTH)),
                "large2x": file_url(title, thumb_width(native_w, WORK_WIDTH))},
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
            if e.code in (500, 502, 504) and attempt < 2:
                time.sleep(2.0)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # Обрыв или таймаут — повтор, а не потеря запроса во всём слоте
            # (замер 27.09: сбой источника раньше стоил слоту всего Commons).
            if attempt < 2:
                time.sleep(2.0)
                continue
            raise
    pages = sorted(((data.get("query") or {}).get("pages") or {}).values(),
                   key=lambda p: p.get("index", 0))
    _cache_put(query, limit, pages)
    return pages


def _search_exact(query, limit):
    """Один запрос к поиску (или к его дисковому кэшу) -> годные кандидаты."""
    pages = _fetch_pages(query, limit)
    out = []
    for page in pages:
        STATS["results"] += 1
        cand = to_candidate(page)
        if cand is not None:
            out.append(cand)
    STATS["kept"] += len(out)
    return out


def relaxed_queries(query, max_variants=RELAX_MAX_VARIANTS):
    """Более общие формулировки запроса — от самой безопасной к самой
    широкой, не короче RELAX_KEEP_WORDS слов и без повторов.

    Поиск Commons — И по словам, и пустой ответ почти всегда значит «одно из
    слов лишнее», а не «предмета нет». Порядок ступеней — по измеренной доле
    возвращённых результатов на запросах из кэша (те самые запросы, где
    точная формулировка дала ноль, а её подмножество слов лежит в кэше):

        убрать числа и слова кадра     20 из 22   (91%)
        убрать последнее слово         49 из 69   (71%)
        убрать два последних слова     19 из 20   (95%)

    Последнее слово — чаще всего уточнение состояния («... mud», «... dark»),
    а предмет стоит первым, поэтому обрезка идёт с хвоста. Числа и слова
    кадра идут первыми: они не несут предмета вовсе. Честно про цифры: выборка
    мала и смещена (подзапросы лежат в кэше потому, что их писал сам план),
    поэтому это порядок ступеней, а не обещание охвата."""
    words = str(query or "").split()
    if len(words) < RELAX_MIN_WORDS:
        return []
    seen = {" ".join(words).lower()}
    out = []

    def has_content(ws):
        return any(not re.fullmatch(r"\d+", w) and w.lower() not in RELAX_FRAMING_WORDS
                   for w in ws)

    def add(ws):
        # Формулировка из одних чисел и слов кадра («close up») ничего не ищет
        if len(ws) < RELAX_KEEP_WORDS or len(out) >= max_variants or not has_content(ws):
            return
        q = " ".join(ws)
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)

    core = [w for w in words
            if not re.fullmatch(r"\d+", w) and w.lower() not in RELAX_FRAMING_WORDS]
    base = words
    if len(core) >= RELAX_KEEP_WORDS and len(core) < len(words):
        add(core)
        base = core
    add(base[:-1])
    add(base[:-2])
    return out


def search(query, limit=SEARCH_LIMIT):
    """Кандидаты Commons по запросу, в порядке релевантности поиска. Если
    точный запрос не дал ни одного годного, — по более общим формулировкам
    (relaxed_queries), первая непустая побеждает: ослабление добавляет
    кандидатов только там, где их не было вовсе.

    Исключения наружу: fail-open решает вызывающий (как у остальных
    источников пула). Сбой на ослабленном шаге — тоже исключение: слот тогда
    честно помечается неполным и решается заново, а не остаётся с пустой
    выдачей, которой на самом деле не было."""
    query = " ".join(str(query or "").split())
    if not query:
        return []
    out = _search_exact(query, limit)
    if out:
        return out
    for variant in relaxed_queries(query):
        STATS["relaxed_tries"] += 1
        got = _search_exact(variant, limit)
        if got:
            STATS["relaxed_hits"] += 1
            print(f"    Commons: {query!r} -> ничего, взят более общий запрос "
                  f"{variant!r} ({len(got)} канд.)")
            return got
    return out
