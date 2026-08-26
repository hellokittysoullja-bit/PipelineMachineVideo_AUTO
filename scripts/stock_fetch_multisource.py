#!/usr/bin/env python3
"""Мультисточниковый сток: Pexels + Pixabay + Unsplash (ЧАСТЬ 14).
Читает media_plan/slots_master_index.txt (строки 'idx|текст').
Нечётные слоты -> фото, чётные -> видео. Внутри категории источники
round-robin с фоллбэком. Unsplash — demo 50/час (cap 45). Все источники
пишут в одно имя на слот. Safe re-run (пропуск готовых слотов).
Тематический словарь: channel_themes.json (корень репо) + media_plan/themes.json.
Usage: python scripts/stock_fetch_multisource.py <video_dir>"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Уже использованные результаты, по источникам. Без этого каждый источник
# всегда брал hits[0]: все слоты с одинаковым тематическим запросом (а их в
# ролике десятки — словарь тем на то и словарь) получали ОДНУ И ТУ ЖЕ картинку.
# Теперь перебираем выдачу и берём первый ещё не использованный результат.
used_ids = {}


def _pick_unused(source, items, key):
    """Первый элемент, чей ключ ещё не встречался в этом прогоне. Если все уже
    были — берём первый (повтор лучше пустого слота) и не портим статистику."""
    seen = used_ids.setdefault(source, set())
    chosen = None
    for it in items:
        k = key(it)
        if k is None:
            continue
        if k not in seen:
            chosen = it
            seen.add(k)
            break
    if chosen is None:
        chosen = items[0]
        k = key(chosen)
        if k is not None:
            seen.add(k)
    return chosen


PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_API_KEY", "")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
UNSPLASH_HOURLY_CAP = 45
unsplash_used = 0
DEFAULT_QUERY = "cinematic atmospheric moody"


def _load_json_dict(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  Битый {path}, пропускаю: {e}")
    return {}


def load_themes(base):
    """Два уровня, ровно как в pipeline_smart.py: канальный словарь
    channel_themes.json в корне репозитория + эпизодный media_plan/themes.json
    поверх него.

    Канальный уровень тут отсутствовал: скрипт читал только эпизодный файл, и
    если его не завели (а его и не надо заводить — ради этого канальный словарь
    и делали), КАЖДЫЙ слот уходил в Pexels/Pixabay с одним и тем же
    DEFAULT_QUERY. То есть весь сток ролика скачивался по запросу
    "cinematic atmospheric moody" вместо тематических слов канала."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    merged = _load_json_dict(os.path.join(repo_root, "channel_themes.json"))
    merged.update(_load_json_dict(os.path.join(base, "media_plan", "themes.json")))
    return merged


def build_query(text, themes):
    tl = text.lower()
    for kw, q in themes.items():
        if kw in tl:
            return q
    return DEFAULT_QUERY


def _download(url, out, headers=None):
    """Скачать во временный файл и переименовать только после полной записи.

    Раньше писали прямо в целевое имя: обрыв связи или Ctrl-C посреди загрузки
    оставлял в media/ обрезанный файл, а следующий прогон считал слот готовым
    (проверка — только по существованию имени) и больше к нему не возвращался.
    Битый кадр доезжал до сборки и валил ffmpeg уже там."""
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
    tmp = f"{out}.part"
    try:
        with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
            data = r.read()
            if not data:
                raise ValueError("пустой ответ")
            f.write(data)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def fetch_pexels_photo(q, out):
    if not PEXELS_API_KEY:
        return False
    req = urllib.request.Request(
        f"https://api.pexels.com/v1/search?query={urllib.parse.quote(q)}&per_page=15&orientation=landscape",
        headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    ph = data.get("photos", [])
    if not ph:
        return False
    src = _pick_unused("pexels_photo", ph, lambda x: x.get("id")).get("src", {})
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
        f"https://api.pexels.com/videos/search?query={urllib.parse.quote(q)}&per_page=15&orientation=landscape",
        headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    vs = data.get("videos", [])
    if not vs:
        return False
    vid = _pick_unused("pexels_video", vs, lambda x: x.get("id"))
    files = sorted(vid.get("video_files", []), key=lambda f: f.get("width", 0), reverse=True)
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
           f"&image_type=photo&orientation=horizontal&per_page=15&safesearch=true")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
        data = json.load(r)
    hits = data.get("hits", [])
    if not hits:
        return False
    hit = _pick_unused("pixabay_photo", hits, lambda x: x.get("id"))
    url = hit.get("largeImageURL") or hit.get("webformatURL")
    if not url:
        return False
    _download(url, out)
    return True


def fetch_pixabay_video(q, out):
    if not PIXABAY_API_KEY:
        return False
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={urllib.parse.quote(q)}&per_page=15&safesearch=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
        data = json.load(r)
    hits = data.get("hits", [])
    if not hits:
        return False
    v = _pick_unused("pixabay_video", hits, lambda x: x.get("id")).get("videos", {})
    # Pixabay отдаёт ключ рендиции даже когда url в нём пустой — проверяем сам url,
    # а не наличие ключа, иначе уходим качать "" и теряем слот.
    link = next((v[k]["url"] for k in ("large", "medium", "small")
                 if isinstance(v.get(k), dict) and v[k].get("url")), None)
    if not link:
        return False
    _download(link, out)
    return True


def fetch_unsplash_photo(q, out):
    global unsplash_used
    if not UNSPLASH_ACCESS_KEY or unsplash_used >= UNSPLASH_HOURLY_CAP:
        return False
    url = (f"https://api.unsplash.com/search/photos?query={urllib.parse.quote(q)}"
           f"&per_page=15&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
        data = json.load(r)
    unsplash_used += 1
    res = data.get("results", [])
    if not res:
        return False
    urls = _pick_unused("unsplash_photo", res, lambda x: x.get("id")).get("urls", {})
    url = urls.get("regular") or urls.get("full") or urls.get("small")
    if not url:
        return False
    _download(url, out)
    return True


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
    for idx, text in rows:
        q = build_query(text, themes)
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
                used = try_sources(PHOTO_SOURCES, idx, q, out)
        else:
            out = os.path.join(outdir, f"{idx:03d}_stock.jpg")
            used = try_sources(PHOTO_SOURCES, idx, q, out)
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
