#!/usr/bin/env python3
"""Мультисточниковый сток: Pexels + Pixabay + Unsplash (ЧАСТЬ 14).
Читает media_plan/slots_master_index.txt (строки 'idx|текст').
Нечётные слоты -> фото, чётные -> видео. Внутри категории источники
round-robin с фоллбэком. Unsplash — demo 50/час (cap 45). Все источники
пишут в одно имя на слот. Safe re-run (пропуск готовых слотов).
Тематический словарь — media_plan/themes.json (рус.ключ -> англ.запрос).
Usage: python scripts/stock_fetch_multisource.py <video_dir>"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_API_KEY", "")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
UNSPLASH_HOURLY_CAP = 45
unsplash_used = 0
DEFAULT_QUERY = "cinematic atmospheric moody"


def _load_json_dict(p):
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return {}


def load_themes(base):
    """Два уровня, как в pipeline_smart.load_themes() — канальный словарь
    (channel_themes.json в корне репо: "меч"/"доспех"/"музе" и т.п., общие
    для ниши) + эпизодный (media_plan/themes.json конкретного видео, только
    специфичное этой теме). РЕАЛЬНЫЙ баг, пойманный при разборе: раньше
    здесь читался ТОЛЬКО эпизодный файл — на свежем эпизоде с типично
    коротким media_plan/themes.json (по замыслу CLAUDE.md — несколько строк
    добавки ПОВЕРХ базы, не полный словарь заново) канальная база вообще не
    подключалась, и build_query() почти на каждом слоте молча уходил в
    DEFAULT_QUERY вместо реального тематического запроса."""
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "channel_themes.json")
    episode_path = os.path.join(base, "media_plan", "themes.json")
    merged = _load_json_dict(base_path)
    merged.update(_load_json_dict(episode_path))
    return merged


def build_query(text, themes, keyword_counts=None):
    """keyword_counts (опционально, мутируется на месте) — значение themes
    может быть СПИСКОМ синонимичных запросов, не только строкой (см.
    channel_themes.json: "меч"/"музе" — самые частые слова в этой нише,
    один и тот же узкий запрос на весь эпизод быстро исчерпывает пул
    Pexels/Pixabay/Unsplash по нему). Без keyword_counts список просто
    ротируется по счётчику в 0 каждый вызов (первый вариант) — не падает,
    но и не разводит нагрузку между слотами; main() передаёт общий на весь
    прогон словарь, та же логика, что уже использует query_for() в
    pipeline_smart.py для того же channel_themes.json.

    Матч по границе слова (\\b), не по голой подстроке — ключи это корни
    словоформ ("музе" -> музей/музейные), \\w* в конце не нужен: re.search
    без $-якоря на конце уже допускает любое продолжение после найденного
    корня, важна только граница СЛЕВА (не разрешить попасть в середину
    случайно совпавшего произвольного слова)."""
    tl = text.lower()
    for kw, q in themes.items():
        if re.search(r'\b' + re.escape(kw), tl):
            if isinstance(q, list):
                idx = keyword_counts.get(kw, 0) if keyword_counts is not None else 0
                if keyword_counts is not None:
                    keyword_counts[kw] = idx + 1
                return q[idx % len(q)]
            return q
    return DEFAULT_QUERY


def _download(url, out, headers=None):
    """Атомарная запись: сначала в .part, потом os.replace.

    РЕАЛЬНЫЙ БАГ без этого: файл писался прямо в итоговое имя слота. Обрыв
    сети/Ctrl-C посреди скачивания оставлял в media/ ОБРЕЗАННЫЙ файл под
    правильным именем — а этот же скрипт при следующем запуске считает
    существующий файл готовым слотом и пропускает его ("Safe re-run"), а
    сборщик потом падает на битом кадре. Пустой ответ отвергается — тот же
    порог, что у pipeline_smart.atomic_url_download() (не выдуманный
    минимальный размер: он мог бы отбросить и настоящий маленький файл)."""
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
    tmp = out + ".part"
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if not data:
            raise IOError("скачан 0-байтный файл")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, out)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


USED_MEDIA_KEYS = set()   # что уже скачано в этом прогоне: "источник:id"


def _pick_unused(items, key_fn):
    """Первый кандидат, которого в этом прогоне ещё не брали.

    РЕАЛЬНЫЙ И САМЫЙ ЗАМЕТНЫЙ НА ГЛАЗ БАГ этого скрипта: все источники
    брали ЖЁСТКО первый результат выдачи (ph[0]/hits[0]/res[0]). Запрос же
    берётся из тематического словаря по корню слова — то есть десятки
    слотов эпизода получают ОДИН И ТОТ ЖЕ запрос ("доспех" -> "knight plate
    armor"), а значит и одну и ту же верхнюю картинку. Ролик собирался из
    нескольких фотографий, повторённых по многу раз, при том что в выдаче
    лежали десятки подходящих. Теперь ротация по выдаче с памятью на весь
    прогон; если пул исчерпан — берём первый (повтор лучше пустого слота),
    но уже не помечаем повторно."""
    for it in items:
        k = key_fn(it)
        if k not in USED_MEDIA_KEYS:
            USED_MEDIA_KEYS.add(k)
            return it
    return items[0] if items else None


def fetch_pexels_photo(q, out):
    if not PEXELS_API_KEY:
        return False
    # per_page=30, а не 3: тот же ОДИН запрос к API (квота не тратится
    # больше), но пул кандидатов, из которого _pick_unused() берёт ещё не
    # использованный кадр, а не один и тот же топ-1 на все слоты с этим
    # запросом.
    req = urllib.request.Request(
        f"https://api.pexels.com/v1/search?query={urllib.parse.quote(q)}&per_page=30&orientation=landscape",
        headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    ph = data.get("photos", [])
    if not ph:
        return False
    pick = _pick_unused(ph, lambda p: f"pexels_photo:{p.get('id')}")
    src = (pick or {}).get("src", {})
    # выбираем лучший доступный размер: раньше жёсткая ссылка на large2x
    # роняла источник с KeyError, если Pexels его не отдал
    url = src.get("large2x") or src.get("large") or src.get("original")
    if not url:
        return False
    _download(url, out)
    return True


def fetch_pexels_video(q, out):
    if not PEXELS_API_KEY:
        return False
    req = urllib.request.Request(
        f"https://api.pexels.com/videos/search?query={urllib.parse.quote(q)}&per_page=30&orientation=landscape",
        headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    vs = data.get("videos", [])
    if not vs:
        return False
    pick = _pick_unused(vs, lambda v: f"pexels_video:{v.get('id')}")
    files = sorted((pick or {}).get("video_files", []), key=lambda f: f.get("width", 0), reverse=True)
    hd = [f for f in files if 960 <= f.get("width", 0) <= 1920]
    chosen = hd[0] if hd else (files[-1] if files else None)
    if not chosen:
        return False
    _download(chosen["link"], out)
    return True


def fetch_pixabay_photo(q, out):
    if not PIXABAY_API_KEY:
        return False
    url = (f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={urllib.parse.quote(q)}"
           f"&image_type=photo&orientation=horizontal&per_page=30&safesearch=true")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
        data = json.load(r)
    hits = data.get("hits", [])
    if not hits:
        return False
    pick = _pick_unused(hits, lambda p: f"pixabay_photo:{p.get('id')}") or {}
    url = pick.get("largeImageURL") or pick.get("webformatURL")
    if not url:
        return False
    _download(url, out)
    return True


def fetch_pixabay_video(q, out):
    if not PIXABAY_API_KEY:
        return False
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={urllib.parse.quote(q)}&per_page=30&safesearch=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
        data = json.load(r)
    hits = data.get("hits", [])
    if not hits:
        return False
    v = (_pick_unused(hits, lambda p: f"pixabay_video:{p.get('id')}") or {}).get("videos", {})
    chosen = v.get("large") or v.get("medium") or v.get("small")
    if not chosen:
        return False
    _download(chosen["url"], out)
    return True


def fetch_unsplash_photo(q, out):
    global unsplash_used
    if not UNSPLASH_ACCESS_KEY or unsplash_used >= UNSPLASH_HOURLY_CAP:
        return False
    url = (f"https://api.unsplash.com/search/photos?query={urllib.parse.quote(q)}"
           f"&per_page=30&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
        data = json.load(r)
    unsplash_used += 1
    res = data.get("results", [])
    if not res:
        return False
    urls = (_pick_unused(res, lambda p: f"unsplash:{p.get('id')}") or {}).get("urls", {})
    url = urls.get("regular") or urls.get("full") or urls.get("small")
    if not url:
        return False
    _download(url, out)
    return True


# --- Openverse (опционально, OPENVERSE_ENABLED=0 по умолчанию — ВЫКЛЮЧЕН) ---
# По прямому запросу пользователя: атласные исторические карты (границы
# королевств/империй на конкретный год) — контент, которого физически НЕТ
# у Pexels/Pixabay/Unsplash (это фотосток современных сцен, не картография).
# Реально проверено вживую: запрос "holy roman empire political map" через
# https://api.openverse.org/v1/images/ вернул РЕАЛЬНУЮ карту того же типа,
# что в примере пользователя ("Europe 1200" с подписанными границами
# королевств, euratlas.com, лицензия CC BY 2.0) — не гипотеза, реальный файл
# скачан и просмотрен.
#
# ЧЕСТНО, по прямому требованию пользователя "супер осторожно, чтобы
# гарантированно не словить демонетизацию": ограничена ТОЛЬКО `license=cc0`
# — НИКАКОГО by/by-sa. Причина не «by тоже нельзя» (CC BY/BY-SA разрешают
# коммерческое использование, включая монетизацию) — причина в том, что у
# CC BY/BY-SA есть ОБЯЗАТЕЛЬНОЕ условие атрибуции (упоминание автора в
# описании ролика), а в этом пайплайне СЕГОДНЯ нет механизма, который
# собирает и вставляет такие атрибуции в описание YouTube — контент
# скачался бы легально, но без соблюдения условия лицензии на выходе, то
# есть реальный риск оказался бы не сразу в скачивании, а позже, в
# невыполненном требовании лицензии. by/by-sa — отдельная задача (нужен
# сборщик атрибуций в описание), сознательно не в этом заходе.
#
# PDM (Public Domain Mark) сознательно ИСКЛЮЧЕН, по второму заходу
# ("копни глубже", прямое требование пользователя). Разница с CC0 —
# не формальность, проверено на первоисточнике (creativecommons.org,
# официальный текст лицензии, не пересказ): CC0 — юридический документ,
# которым правообладатель ДЕЙСТВИТЕЛЬНО отказывается от прав ("dedication").
# PDM — просто пометка "This work has been **identified** as being free of
# known restrictions" — кто-то (не обязательно правообладатель) посчитал
# работу свободной, с явной оговоркой самого CC: "may not be free of known
# copyright restrictions in all jurisdictions", без каких-либо гарантий.
# Для канала это разница между "юридически оформленный отказ от прав" и
# "чьё-то мнение, что прав ни у кого нет" — при разметке демонетизацией
# вторая категория объективно рискованнее первой, даже если обе выглядят
# в UI Openverse одинаково безопасными. license=cc0 — единственный вариант,
# который не требует НИКАКОГО доверия к чужой оценке.
#
# Второй независимый слой (тоже по "копни глубже"): даже честный CC0-тег
# может быть проставлен ОШИБОЧНО — источник, где лицензию проставляет
# случайный пользователь без модерации (например, произвольный аккаунт на
# Flickr), физически может пометить чужую работу как CC0 по ошибке или
# незнанию. Openverse различает это через поле `source` (не `provider` —
# проверено вживую: NASA и Biodiversity Heritage Library технически идут
# через Flickr API и у обоих `provider=flickr`, но `source` сохраняет
# настоящую институциональную принадлежность: `source=nasa`/`source=
# bio_diversity`; спутать эти поля означало бы либо ошибочно доверять
# ВСЕМУ Flickr, либо ошибочно блокировать легитimate NASA/BHL). Поэтому
# автоматически берутся только результаты из `source`, принадлежащего
# заранее проверенному списку архивов/музеев/библиотек с собственной
# юридической проверкой прав ПЕРЕД публикацией (Wikimedia Commons —
# community-review; Смитсоновский институт, Метрополитен, Рейксмузеум,
# Кливлендский музей — официальные Open Access программы; Europeana,
# NASA, Biodiversity Heritage Library, Digitalt Museum, Wellcome
# Collection — институциональные агрегаторы/архивы). НЕ в списке —
# любые персональные/самотегируемые источники (обычный Flickr,
# iNaturalist, Rawpixel, StockSnap, WordPress и т.п.), там лицензию
# ставит сам загрузивший, без институциональной проверки.
#
# Дополнительная защита (как и раньше): license/source-фильтры в URL —
# это ЗАПРОШЕННОЕ условие поиска, не гарантия (мало ли баг на стороне
# API) — код ПОВТОРНО проверяет ОБА поля КАЖДОГО реального результата
# перед скачиванием (см. _is_safe_openverse_license/
# _is_trusted_openverse_source), fail closed на любом расхождении с
# ожиданием — не доверяет фильтру запроса вслепую.
OPENVERSE_ENABLED = os.environ.get("OPENVERSE_ENABLED", "0") != "0"
OPENVERSE_SAFE_LICENSES = {"cc0"}   # НЕ pdm — см. докстринг выше. НЕ трогать без сборщика атрибуций (by/by-sa)

# `source` (не `provider` — см. докстринг выше), институциональные архивы
# с собственной юридической проверкой прав перед публикацией. Проверено
# вживую на реальном API (не гипотеза): все перечисленные реально отдают
# license=cc0 контент под соответствующим source (кроме нескольких
# smithsonian_* веток и smk/sciencemuseum/finnish_heritage_agency/
# bib_gulbenkian/brooklynmuseum/finnish_satakunnan_museum — на момент
# проверки 0 cc0-результатов в индексе Openverse, оставлены в списке как
# легитимные институции на будущее, фильтр license=cc0 всё равно не даст
# им ничего пропустить, если результатов нет).
OPENVERSE_TRUSTED_SOURCES = {
    "wikimedia", "met", "rijksmuseum", "clevelandmuseum", "brooklynmuseum",
    "digitaltmuseum", "europeana", "nasa", "bio_diversity",
    "wellcome_collection", "sciencemuseum", "smk", "finnish_heritage_agency",
    "finnish_satakunnan_museum", "bib_gulbenkian",
    # Смитсоновские подразделения — Openverse хранит их отдельными
    # source-слагами, а параметр запроса source= не принимает префиксы
    # (проверено вживую), поэтому для реального URL-фильтра нужно
    # перечисление; _is_trusted_openverse_source() ниже дополнительно
    # проверяет ЛЮБОЙ smithsonian_* префиксом — на случай новых подразделений,
    # которые Openverse добавит после этого списка (они не будут покрыты
    # URL-фильтром, но безопасно пройдут пост-проверку, если когда-нибудь
    # окажутся среди результатов другого source= запроса).
    "smithsonian_national_museum_of_natural_history",
    "smithsonian_cooper_hewitt_museum", "smithsonian_american_history_museum",
    "smithsonian_portrait_gallery", "smithsonian_african_american_history_museum",
    "smithsonian_gardens", "smithsonian_american_art_museum",
    "smithsonian_postal_museum", "smithsonian_freer_gallery_of_art",
    "smithsonian_institution_archives", "smithsonian_air_and_space_museum",
}


def _is_safe_openverse_license(result):
    lic = (result.get("license") or "").strip().lower()
    return lic in OPENVERSE_SAFE_LICENSES


def _is_trusted_openverse_source(result):
    src = (result.get("source") or "").strip().lower()
    if src in OPENVERSE_TRUSTED_SOURCES:
        return True
    # Openverse хранит смитсоновские подразделения отдельными source-слагами
    # (smithsonian_national_museum_of_natural_history и т.д.) — префиксом,
    # чтобы не перечислять и не терять новые ветки, которые Openverse
    # добавит позже.
    return src.startswith("smithsonian_")


def _log_openverse_manifest(base, entry):
    """Юридический аудит-трейл (CLAUDE.md ЧАСТЬ 7-стиль: source/url/автор/
    лицензия/запрос) — append-only JSONL, ОДИН файл на эпизод, переживает
    повторные прогоны. Не блокирует скачивание при сбое записи (сам аудит-
    лог не должен ронять сток)."""
    try:
        manifest_dir = os.path.join(base, "media_plan")
        os.makedirs(manifest_dir, exist_ok=True)
        path = os.path.join(manifest_dir, "openverse_license_manifest.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def fetch_openverse_photo(q, out, base=None):
    """Та же сигнатура (q, out) -> bool, что и остальные fetch_*_photo, для
    совместимости с try_sources() — base (опционально, для манифеста)
    передаётся через замыкающую функцию при регистрации источника в main()
    (см. _openverse_source, не functools.partial — у partial-объекта нет
    __name__, try_sources() на нём упал бы при логировании ошибки), не
    через глобальный список PHOTO_SOURCES (тот вызывается с (q, out) без
    доп. аргументов)."""
    if not OPENVERSE_ENABLED:
        return False
    url = ("https://api.openverse.org/v1/images/?q=" + urllib.parse.quote(q) +
           "&license=" + ",".join(sorted(OPENVERSE_SAFE_LICENSES)) +
           "&source=" + ",".join(sorted(OPENVERSE_TRUSTED_SOURCES)) +
           "&page_size=5&mature=false")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    results = data.get("results", [])
    # Ротация как у остальных источников: сначала те, которых в этом прогоне
    # ещё не брали (лицензионные/источниковые проверки ниже не меняются).
    results = ([r for r in results if f"openverse:{r.get('id')}" not in USED_MEDIA_KEYS]
               + [r for r in results if f"openverse:{r.get('id')}" in USED_MEDIA_KEYS])
    for res in results:
        # Оба условия — запрошенные фильтры URL (license=/source=) ЭКОНОМЯТ
        # запросы, но не гарантия (баг/несогласованность на стороне API) —
        # ПОВТОРНАЯ проверка обоих полей каждого результата перед скачиванием,
        # fail closed. source= в URL не покрывает смитсоновские подветки
        # (Openverse не принимает префиксы в параметре запроса) — поэтому
        # они всё равно проходят через реальную проверку здесь, а не только
        # через URL-фильтр.
        if not _is_safe_openverse_license(res):
            continue   # fail closed — см. докстринг модуля выше
        if not _is_trusted_openverse_source(res):
            continue   # fail closed — см. докстринг модуля выше
        img_url = res.get("url")
        if not img_url:
            continue
        try:
            _download(img_url, out)
        except Exception:
            continue
        USED_MEDIA_KEYS.add(f"openverse:{res.get('id')}")
        if base:
            _log_openverse_manifest(base, {
                "query": q, "id": res.get("id"), "title": res.get("title"),
                "url": img_url, "creator": res.get("creator"),
                "license": res.get("license"), "license_version": res.get("license_version"),
                "license_url": res.get("license_url"), "provider": res.get("provider"),
                "source": res.get("source"),
                "foreign_landing_url": res.get("foreign_landing_url"),
            })
        return True
    return False


PHOTO_SOURCES = [fetch_pexels_photo, fetch_pixabay_photo, fetch_unsplash_photo]
VIDEO_SOURCES = [fetch_pexels_video, fetch_pixabay_video]


def try_sources(sources, start, q, out):
    n = len(sources)
    for off in range(n):
        fn = sources[(start + off) % n]
        try:
            if fn(q, out):
                return fn.__name__
        except Exception as e:
            # Причину печатаем: раньше любая ошибка глушилась молча и слот
            # просто оказывался пустым без единого намёка почему.
            detail = getattr(e, "code", None)
            print(f"    {fn.__name__}: {type(e).__name__}"
                  f"{f' {detail}' if detail else ''}: {e}", flush=True)
    return None


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    idx_path = os.path.join(base, "media_plan", "slots_master_index.txt")
    outdir = os.path.join(base, "media")
    if not os.path.exists(idx_path):
        print(f"Не найден {idx_path}")
        return 1
    os.makedirs(outdir, exist_ok=True)
    themes = load_themes(base)

    # OPENVERSE_ENABLED=0 (дефолт) -> photo_sources байт-в-байт PHOTO_SOURCES,
    # ноль влияния. Openverse ставится ПОСЛЕДНИМ (после Pexels/Pixabay/
    # Unsplash) в round-robin — уже проверенные фотостоки чаще дают
    # релевантный результат на обычный (не картографический) запрос,
    # Openverse — расширение покрытия, не замена.
    photo_sources = list(PHOTO_SOURCES)
    if OPENVERSE_ENABLED:
        # Обычная замыкающая функция, не functools.partial — try_sources()
        # печатает fn.__name__ при ошибке, у partial-объекта такого атрибута
        # нет (упало бы AttributeError вместо аккуратного лога).
        def _openverse_source(q, out):
            return fetch_openverse_photo(q, out, base=base)
        _openverse_source.__name__ = "fetch_openverse_photo"
        photo_sources.append(_openverse_source)

    rows = []
    bad_lines = []
    for n, line in enumerate(open(idx_path, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        # Раньше строка без '|' роняла весь скрипт голым ValueError на
        # unpack — одна опечатка при ручной правке индекса убивала весь
        # прогон. Теперь такая строка пропускается с понятным сообщением.
        if "|" not in line:
            bad_lines.append(n)
            continue
        i, text = line.split("|", 1)
        try:
            rows.append((int(i), text))
        except ValueError:
            bad_lines.append(n)
    if bad_lines:
        print(f"Пропущены битые строки (ожидался формат idx|текст): {bad_lines}")
    if not rows:
        print("Нет валидных строк в slots_master_index.txt")
        return 1

    ok = fail = skip = 0
    keyword_counts = {}   # общий на весь прогон — ротирует list-значения themes.json по слотам
    for idx, text in rows:
        q = build_query(text, themes, keyword_counts)
        is_video = (idx % 2 == 0)
        existing = None
        for suf in ("_stock_video.mp4", "_stock.jpg"):
            p = os.path.join(outdir, f"{idx:03d}{suf}")
            if os.path.exists(p):
                existing = p
                break
        if existing:
            skip += 1
            continue
        if is_video:
            out = os.path.join(outdir, f"{idx:03d}_stock_video.mp4")
            used = try_sources(VIDEO_SOURCES, idx, q, out)
            if not used:
                out = os.path.join(outdir, f"{idx:03d}_stock.jpg")
                used = try_sources(photo_sources, idx, q, out)
        else:
            out = os.path.join(outdir, f"{idx:03d}_stock.jpg")
            used = try_sources(photo_sources, idx, q, out)
        if used:
            ok += 1
            print(f"[{idx:03d}] OK ({used}): {q}", flush=True)
        else:
            fail += 1
            print(f"[{idx:03d}] NO RESULT: {q}", flush=True)
        time.sleep(0.3)
    print(f"\nИтого: OK={ok} FAIL={fail} SKIP={skip} | Unsplash {unsplash_used}", flush=True)
    # Отдельные промахи по слоту — штатный сценарий (лимиты источников,
    # временная недоступность), не повод считать прогон целиком неудачным.
    # Но если НИ ОДИН слот не заполнился — это реальный отказ, не частичность.
    if ok == 0 and (fail > 0 or skip == 0):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
