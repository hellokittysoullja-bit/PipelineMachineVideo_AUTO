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


def load_themes(base):
    p = os.path.join(base, "media_plan", "themes.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return {}


def build_query(text, themes):
    tl = text.lower()
    for kw, q in themes.items():
        if kw in tl:
            return q
    return DEFAULT_QUERY


def _download(url, out, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    open(out, "wb").write(data)


def fetch_pexels_photo(q, out):
    if not PEXELS_API_KEY:
        return False
    req = urllib.request.Request(
        f"https://api.pexels.com/v1/search?query={urllib.parse.quote(q)}&per_page=3&orientation=landscape",
        headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    ph = data.get("photos", [])
    if not ph:
        return False
    src = ph[0].get("src", {})
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
        f"https://api.pexels.com/videos/search?query={urllib.parse.quote(q)}&per_page=5&orientation=landscape",
        headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    vs = data.get("videos", [])
    if not vs:
        return False
    files = sorted(vs[0].get("video_files", []), key=lambda f: f.get("width", 0), reverse=True)
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
           f"&image_type=photo&orientation=horizontal&per_page=3&safesearch=true")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
        data = json.load(r)
    hits = data.get("hits", [])
    if not hits:
        return False
    url = hits[0].get("largeImageURL") or hits[0].get("webformatURL")
    if not url:
        return False
    _download(url, out)
    return True


def fetch_pixabay_video(q, out):
    if not PIXABAY_API_KEY:
        return False
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={urllib.parse.quote(q)}&per_page=3&safesearch=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
        data = json.load(r)
    hits = data.get("hits", [])
    if not hits:
        return False
    v = hits[0].get("videos", {})
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
           f"&per_page=3&orientation=landscape&client_id={UNSPLASH_ACCESS_KEY}")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
        data = json.load(r)
    unsplash_used += 1
    res = data.get("results", [])
    if not res:
        return False
    urls = res[0].get("urls", {})
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
    os.makedirs(outdir, exist_ok=True)
    themes = load_themes(base)

    rows = []
    for line in open(idx_path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        i, text = line.split("|", 1)
        rows.append((int(i), text))

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


if __name__ == "__main__":
    main()
