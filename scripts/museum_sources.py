"""Прямые открытые API музеев — источник кадров, где эпоха ИЗВЕСТНА, а не
угадывается по пикселям.

Зачем это существует (разбор 10.09, эпизод 02_ne-mechom). В опубликованный
ролик про Азенкур попали танк на кульминации «Рыцарей убивала земля»,
терракотовая армия, египетский саркофаг, космонавт и кавалерия XIX века.
Замер показал, что запросы слотов были ПРАВИЛЬНЫЕ («medieval rondel dagger»),
а пустым был пул: Pexels — библиотека современной lifestyle-фотографии,
средневековья там нет, и на запрос без совпадений он отдаёт ближайшее по
вектору. Гейты после этого спорят, приемлем ли космонавт.

Три попытки научить модель ОТЛИЧАТЬ эпоху по картинке провалились (две
задокументированы в CLAUDE.md, третья — «разбиение по эпохе» — не прошла
золотой набор: настоящее макро доспеха выглядит «не средневековым» сильнее
танка, а брак с современной улицей выглядит «средневековым», потому что в
кадре костюм рыцаря). Причина общая: правильность кадра живёт в паре
«фраза+картинка» и в контексте эпохи, а эмбеддинг-косинус меряет похожесть
пикселей на текст.

Здесь угадывать не нужно. Музей вместе со снимком отдаёт ПАСПОРТ предмета:
    Met      objectBeginDate=1300  objectEndDate=1375  culture='possibly Italian'
    Cleveland creation_date_earliest=1570 ... latest=1580  culture=['North Italy, 16th c.']
    Chicago  date_start=1500 date_end=1530  place_of_origin='Nuremberg'
Это данные хранителей коллекции, а не догадка по пикселям. Танк не может
пройти не потому, что мы его распознали, а потому что в музее нет предмета с
датой 1400 год, который был бы танком; терракотовая армия отсеивается по
культуре («Chinese»), саркофаг — по ней же и по дате.

Лицензии: берём ТОЛЬКО то, что каждый музей сам пометил как public domain
(Met isPublicDomain, Cleveland share_license_status == "CC0", Chicago
is_public_domain). Атрибуция НЕ требуется — сознательное условие: канал не
обязан вставлять список авторов в описание ролика (ровно поэтому сюда не
берутся by/by-sa источники, см. решение владельца от 10.09). Ключи API не
нужны ни одному из трёх.

Fail-open на каждом шаге: недоступный музей возвращает пустой список, пул
продолжает собираться из остальных источников — ни один слот не пустеет
из-за этого модуля.
"""
import concurrent.futures
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import feature_flags
except ImportError:  # модуль используется и как библиотека, и standalone
    feature_flags = None

UA = "Mozilla/5.0 (compatible; FacelessPipeline/1.0)"
HTTP_TIMEOUT = 20

# Окно эпохи канала. Тот же смысл, что era_from/era_to у validate_brief()
# локального режиссёра (docs/quality/DIRECTOR_LOCAL_LLM.md) — здесь оно
# применяется к ПАСПОРТУ предмета, а не к тексту брифа.
DEFAULT_ERA_FROM = 900
DEFAULT_ERA_TO = 1600

# Культуры/страны вне ниши канала. Проверяется по паспорту предмета, а не по
# картинке: "Iranian or Turkish, 1401-1600" проходит по дате и обязано
# отсекаться здесь — иначе восточный доспех XV века попадёт в ролик про
# Азенкур как «эпоха совпала».
DEFAULT_FOREIGN_CULTURE_TERMS = (
    "japan", "japanese", "china", "chinese", "tibet", "tibetan", "korea",
    "korean", "india", "indian", "indo", "iran", "iranian", "persia",
    "persian", "turk", "turkish", "ottoman", "egypt", "egyptian", "nubian",
    "islamic", "arab", "syria", "syrian", "mughal", "nepal", "nepalese",
    "thai", "burmese", "vietnam", "african", "mesoamerican", "aztec", "maya",
    "inca", "peru", "peruvian", "assyrian", "babylonian", "sumerian",
)

MET_API = "https://collectionapi.metmuseum.org/public/collection/v1"
CLEVELAND_API = "https://openaccess-api.clevelandart.org/api/artworks"
CHICAGO_API = "https://api.artic.edu/api/v1/artworks/search"

# Met отдаёт только objectID в поиске — карточка каждого предмета это
# ОТДЕЛЬНЫЙ запрос. Потолок держит цену слота вменяемой; кэш ниже делает
# повторные прогоны бесплатными.
#
# ГЛУБИНА ИЗМЕРЕНА, А НЕ ВЫБРАНА НА ГЛАЗ (13.09, 42 авторских запроса эпизода
# 02_ne-mechom, живые API без ключей). Потолок 12 был не ценой корпуса, а
# нашим собственным: у 27 запросов из 42 поиск Мет отдаёт БОЛЬШЕ 12 objectID
# (медиана 19.5, максимум 2992 на "medieval poleaxe weapon"). То есть слот
# голодал не потому, что в музее нет предметов, а потому что мы смотрели
# первые двенадцать. Выход, прошедший паспортные фильтры (public domain +
# era_overlaps + culture_is_foreign), на глубине 12 против 60:
#     medieval poleaxe weapon          7 -> 42
#     medieval sword museum display    6 -> 24
#     medieval plate armour museum     3 -> 20
#     medieval manuscript knight battle 10 -> 26
#     medieval rondel dagger           5 -> 12
# Цена по времени НЕ выросла: карточки тянутся параллельно
# (MET_DETAIL_WORKERS), 60 штук приходят за 1.7-6.5с — примерно столько же,
# сколько занимали 12 последовательных запросов. Порядок выдачи сохраняется
# (ex.map отдаёт результаты в порядке входа), поэтому релевантность поиска
# Мет по-прежнему определяет место кандидата в пуле.
MET_MAX_DETAIL_FETCHES = 60
MET_DETAIL_WORKERS = 12

# Кливленд и Чикаго отдают полную карточку прямо в поиске — там глубина стоит
# РОВНО ОДИН запрос, независимо от limit. Тем же замером: Чикаго на limit=20
# даёт 12-17 прошедших паспорт кандидатов, на limit=100 — 48-80 (четырёхкратно,
# бесплатно). Кливленд на средневековой европейской теме отдаёт 0-1 при любом
# limit (его коллекция про другое) — там выигрыша нет, но и цены тоже.
SEARCH_PAGE_SIZE = 100

_SEARCH_CACHE = {}


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return json.load(r)


# --- ВЕЖЛИВОСТЬ К МЕТ -------------------------------------------------------
# Не перестраховка, а реальный, пойманный вживую отказ (13.09): при замере
# глубины подряд ушло ~600 запросов карточек за пару минут, и Мет ответил
# HTTP 403 уже на ПОИСК — то есть источник выключился целиком, а не одна
# карточка. Через минуту доступ вернулся сам: это троттлинг, а не бан.
# Глубина 60 делает такой всплеск штатным (42 запроса эпизода x 60 карточек),
# поэтому у Мет теперь есть три вещи, которых не было:
#   1) общий на процесс ограничитель скорости (не на пул потоков одного
#      запроса — иначе каждый следующий слот начинал бы заново с полной
#      скорости);
#   2) один повтор с паузой на «временных» кодах, ПОСЛЕ которого карточка
#      честно считается потерянной;
#   3) счётчик потерь и пауза-остывание на весь источник: молча уменьшившийся
#      пул неотличим от бедного корпуса, а это ровно тот класс тихой
#      деградации, который в этом проекте ловили уже трижды.
MET_MAX_REQUESTS_PER_SEC = 10.0
MET_RETRY_STATUSES = (403, 429, 500, 502, 503, 504)
# На паузу источник уходит ТОЛЬКО по сигналам троттлинга/недоступности, а не
# по любой ошибке из списка повторов. Разница не косметическая: 500 на одной
# карточке предмета — это сломанная карточка, а не «Мет просит подождать», и
# уводить из-за неё весь источник на минуту значит терять все остальные
# карточки того же запроса. Ровно этот класс («один битый элемент кладёт
# весь слой») уже дважды чинился 13.09 в звуке: битый ассет ронял
# планировщик эффектов, а нечитаемый вход — весь вызов ffmpeg в сведении.
MET_COOLDOWN_STATUSES = (403, 429, 503)
MET_RETRY_PAUSE_SEC = 2.0
MET_COOLDOWN_SEC = 60.0

_MET_LOCK = threading.Lock()
_MET_NEXT_SLOT = [0.0]
_MET_COOLDOWN_UNTIL = [0.0]
# Видимость деградации: сколько карточек потеряно и сколько раз источник
# уходил в остывание за этот прогон. Читается вызывающим кодом/тестами.
FETCH_STATS = {"met_cards_lost": 0, "met_cooldowns": 0, "met_requests": 0}


def _met_throttle():
    """Общий на процесс интервал между запросами к Мет."""
    with _MET_LOCK:
        now = time.monotonic()
        slot = max(now, _MET_NEXT_SLOT[0])
        _MET_NEXT_SLOT[0] = slot + 1.0 / MET_MAX_REQUESTS_PER_SEC
        FETCH_STATS["met_requests"] += 1
    delay = slot - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def met_is_cooling_down():
    return time.monotonic() < _MET_COOLDOWN_UNTIL[0]


def _met_enter_cooldown():
    with _MET_LOCK:
        if not met_is_cooling_down():
            _MET_COOLDOWN_UNTIL[0] = time.monotonic() + MET_COOLDOWN_SEC
            FETCH_STATS["met_cooldowns"] += 1
            print(f"    Мет: троттлинг, источник на паузе {MET_COOLDOWN_SEC:.0f}с "
                  f"(остальные музеи работают)")


def _met_get(url):
    """Запрос к Мет через ограничитель, с одним повтором на временных кодах.

    Возвращает None вместо исключения — вызывающий код обязан отличать
    «карточка не пришла» от «карточка не подошла по паспорту»."""
    if met_is_cooling_down():
        return None
    for attempt in (0, 1):
        _met_throttle()
        try:
            return _get_json(url)
        except urllib.error.HTTPError as e:
            if e.code in MET_RETRY_STATUSES and attempt == 0:
                time.sleep(MET_RETRY_PAUSE_SEC)
                continue
            if e.code in MET_COOLDOWN_STATUSES:
                _met_enter_cooldown()
            else:
                FETCH_STATS["met_lost_cards"] = FETCH_STATS.get("met_lost_cards", 0) + 1
            return None
        except Exception:
            if attempt == 0:
                time.sleep(MET_RETRY_PAUSE_SEC)
                continue
            return None
    return None


def _profile():
    """channel_profile.json, если он есть рядом с репозиторием."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "channel_profile.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def era_window():
    p = _profile()
    return (int(p.get("era_from", DEFAULT_ERA_FROM)),
            int(p.get("era_to", DEFAULT_ERA_TO)))


def foreign_culture_terms():
    p = _profile()
    return tuple(p.get("foreign_culture_terms", DEFAULT_FOREIGN_CULTURE_TERMS))


def era_overlaps(begin, end):
    """Пересекается ли период предмета с окном эпохи канала.

    Пересечение, а не вхождение: у составных предметов музей ставит широкий
    диапазон ("Armor, 1375-1950" — итальянский доспех XIV века, part которого
    реставрирован в XX). Требовать полного вхождения значило бы выбрасывать
    подлинники. Японский доспех 1701-1800 при окне 900-1600 не пересекается
    и отсеивается — то, ради чего фильтр и нужен.
    """
    lo, hi = era_window()
    try:
        b, e = int(begin), int(end)
    except (TypeError, ValueError):
        return False
    if b == 0 and e == 0:
        return False
    return b <= hi and e >= lo


def culture_is_foreign(*fields):
    """Есть ли в паспорте предмета указание на культуру вне ниши канала."""
    terms = foreign_culture_terms()
    blob = " ".join(str(f).lower() for f in fields if f)
    return any(t in blob for t in terms)


# Институт искусств Чикаго отдаёт снимки только с этим заголовком: без него
# IIIF отвечает 403 на ЛЮБУЮ ширину (проверено на 843, 1920 и full/full).
# Требование их документации — назвать проект и контакт.
CHICAGO_IMAGE_HEADERS = {
    "AIC-User-Agent": "FacelessPipeline (github.com/faceless-pipeline)",
}


def _candidate(cid, title, image_url, page_url, meta, headers=None):
    """Кандидат в ФОРМЕ PEXELS — чтобы конкурировать в общем пуле под теми же
    гейтами, что Pexels и Openverse, а не жить отдельной веткой отбора (тот
    же приём, что _openverse_search_photos)."""
    out = {
        "id": cid,
        "alt": title or "",
        "url": page_url or image_url,
        "src": {"large2x": image_url},
        "_museum_meta": meta,
    }
    if headers:
        # Заголовки едут ВМЕСТЕ с кандидатом, а не ищутся по его источнику в
        # чужом модуле: скачивающий код не должен знать, у какого музея какие
        # требования.
        out["_download_headers"] = dict(headers)
    return out


def search_met(query, limit=MET_MAX_DETAIL_FETCHES):
    """Метрополитен: отдел Arms and Armor — лучшая в мире коллекция
    европейского доспеха, всё public domain, ключ не нужен."""
    out = []
    data = _met_get(f"{MET_API}/search?hasImages=true&q=" +
                    urllib.parse.quote(query))
    oids = ((data or {}).get("objectIDs") or [])[:limit]
    if not oids:
        return out

    def _detail(oid):
        # Fail-open ПОКАРТОЧНО, как и было в последовательной версии: упавший
        # запрос одной карточки не должен уносить остальные 59. Потеря
        # считается — иначе поредевший пул выглядел бы бедным корпусом.
        o = _met_get(f"{MET_API}/objects/{oid}")
        if o is None:
            with _MET_LOCK:
                FETCH_STATS["met_cards_lost"] += 1
        return o

    # ex.map сохраняет ПОРЯДОК входа — кандидаты остаются в порядке
    # релевантности поиска Мет, как при последовательном обходе. Это важно:
    # место кандидата в пуле определяет, кого гейты увидят первым (см.
    # чередование по запросам в pipeline_smart.pexels_photo).
    workers = max(1, min(MET_DETAIL_WORKERS, len(oids)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        objects = list(ex.map(_detail, oids))

    for oid, o in zip(oids, objects):
        if not o or not o.get("isPublicDomain"):
            continue
        # Полноразмерный снимок, а не web-версия: замер — 1.4-2 МБ против
        # 55-84 КБ (495x624). Кадр рендерится в 1920x1080 И проходит зум
        # Ken Burns, то есть кроп внутри исходника — на web-версии это
        # гарантированное мыло.
        img = o.get("primaryImage") or o.get("primaryImageSmall")
        if not img:
            continue
        if not era_overlaps(o.get("objectBeginDate"), o.get("objectEndDate")):
            continue
        if culture_is_foreign(o.get("culture"), o.get("country"),
                              o.get("title"), o.get("classification")):
            continue
        out.append(_candidate(
            f"met:{oid}", o.get("title"), img, o.get("objectURL"),
            {"source": "met", "begin": o.get("objectBeginDate"),
             "end": o.get("objectEndDate"), "culture": o.get("culture"),
             "country": o.get("country"), "department": o.get("department")}))
    return out


def search_cleveland(query, limit=SEARCH_PAGE_SIZE):
    """Кливлендский музей: полная карточка приходит прямо в поиске — один
    запрос на весь список, без похода за каждым предметом."""
    out = []
    data = _get_json(f"{CLEVELAND_API}/?q=" + urllib.parse.quote(query) +
                     f"&has_image=1&limit={int(limit)}")
    for a in data.get("data") or []:
        if (a.get("share_license_status") or "").upper() != "CC0":
            continue
        # print (≈2267x3400 jpeg) вместо web (600x900): тот же довод, что у
        # Мет. Вариант "full" сознательно не берётся — это TIFF на 60+ МБ.
        images = a.get("images") or {}
        img = ((images.get("print") or {}).get("url") or
               (images.get("web") or {}).get("url"))
        if not img:
            continue
        if not era_overlaps(a.get("creation_date_earliest"),
                            a.get("creation_date_latest")):
            continue
        culture = a.get("culture")
        if isinstance(culture, list):
            culture = "; ".join(str(c) for c in culture)
        if culture_is_foreign(culture, a.get("title")):
            continue
        out.append(_candidate(
            f"cleveland:{a.get('id')}", a.get("title"), img, a.get("url"),
            {"source": "cleveland", "begin": a.get("creation_date_earliest"),
             "end": a.get("creation_date_latest"), "culture": culture}))
    return out


def search_chicago(query, limit=SEARCH_PAGE_SIZE):
    """Институт искусств Чикаго: карточка тоже приходит в поиске, картинка
    собирается по IIIF-шаблону из ответа."""
    out = []
    fields = ("id,title,date_start,date_end,place_of_origin,"
              "is_public_domain,image_id")
    data = _get_json(f"{CHICAGO_API}?q=" + urllib.parse.quote(query) +
                     f"&limit={int(limit)}&fields={fields}")
    iiif = (data.get("config") or {}).get("iiif_url")
    if not iiif:
        return out
    for a in data.get("data") or []:
        if not a.get("is_public_domain") or not a.get("image_id"):
            continue
        if not era_overlaps(a.get("date_start"), a.get("date_end")):
            continue
        if culture_is_foreign(a.get("place_of_origin"), a.get("title")):
            continue
        out.append(_candidate(
            f"chicago:{a.get('id')}", a.get("title"),
            # IIIF отдаёт любую ширину: просим 1920 под финальный кадр, а не
            # дефолтные 843 из примеров документации.
            f"{iiif}/{a['image_id']}/full/1920,/0/default.jpg",
            f"https://www.artic.edu/artworks/{a.get('id')}",
            {"source": "chicago", "begin": a.get("date_start"),
             "end": a.get("date_end"), "place": a.get("place_of_origin")},
            headers=CHICAGO_IMAGE_HEADERS))
    return out


def _sources():
    """Источники, собираемые В МОМЕНТ ВЫЗОВА, а не при импорте.

    Раньше это был кортеж-константа со ССЫЛКАМИ на функции — и это не мелочь
    стиля: подменить источник (тестом, монкипатчем, будущей заменой
    реализации) было невозможно, вызов всё равно уходил в объект, захваченный
    при импорте. В тесте это выглядело особенно скверно: патч «пусть Мет
    падает» молча не применялся, и тест вместо изоляции уходил в ЖИВОЙ API
    (поймано 13.09). Тот же принцип, что у реестра флагов: значение читается
    в момент вызова.

    Имена функций стоят здесь ЯВНО, а не собираются через globals() по
    строке: строковый резолв работает точно так же, но делает источники
    невидимыми для статического анализа — первая версия этой правки ровно
    так и уронила tests/test_no_dead_layers.py, объявив search_chicago/
    search_cleveland мёртвыми. Позднее связывание не обязано стоить
    проверяемости.
    """
    return (("met", search_met), ("cleveland", search_cleveland),
            ("chicago", search_chicago))


def search_museums(query):
    """Кандидаты из всех трёх музеев, уже отфильтрованные по эпохе и культуре.

    Порядок источников фиксирован (Met первым — у него профильная коллекция
    оружия и доспеха), но победителя по-прежнему выбирают общие гейты и
    скоринг: этот модуль только приносит кандидатов в пул.
    """
    if feature_flags is not None and not feature_flags.enabled("MUSEUM_SOURCES_ENABLED"):
        return []
    if query in _SEARCH_CACHE:
        return _SEARCH_CACHE[query]
    out = []
    for name, fn in _sources():
        try:
            out.extend(fn(query))
        except Exception:
            # Fail-open ПОИСТОЧНИКОВО: упавший музей не должен уносить с
            # собой два оставшихся и уж тем более ронять слот.
            continue
    _SEARCH_CACHE[query] = out
    return out
