#!/usr/bin/env python3
"""Главная сборка по паузам (ЧАСТЬ 6, 9, 13-шаг7).
Разбивает script.txt по [pause] на смысловые блоки, считает тайминг
(слова+паузы, масштаб под реальную длину аудио), Ken Burns на всю длину
кадра, чередование in/out. Медиа: локальная папка media/ по порядку,
fallback — Pexels по тематическому запросу.
Usage: python scripts/pipeline_smart.py <video_dir>"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

VIDEO_FOLDER = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
SCRIPT_FILE = os.path.join(VIDEO_FOLDER, "script.txt")
MEDIA_FOLDER = os.path.join(VIDEO_FOLDER, "media")
OUTPUT_FILE = os.path.join(VIDEO_FOLDER, "final.mp4")
TEMP_FOLDER = os.path.join(VIDEO_FOLDER, "temp_smart")

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
FPS, WIDTH, HEIGHT = 25, 1920, 1080
ZOOM_START, ZOOM_END = 1.0, 1.12
MIN_CLIP, MAX_CLIP = 4.0, 20.0
PAUSE_DURATIONS = {"[pause]": 0.8, "[short pause]": 0.4,
                   "[slowly]": 0.0, "[emphasis]": 0.0, "[energetic]": 0.0}


def find_audio():
    for name in ("audio_fixed.mp3", "audio.mp3"):
        p = os.path.join(VIDEO_FOLDER, name)
        if os.path.exists(p):
            return p
    mp3s = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(".mp3")]
    return os.path.join(VIDEO_FOLDER, mp3s[0]) if mp3s else os.path.join(VIDEO_FOLDER, "audio.mp3")


AUDIO_FILE = find_audio()


def load_themes():
    p = os.path.join(VIDEO_FOLDER, "media_plan", "themes.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return {}


THEMES = load_themes()


def get_audio_duration():
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", AUDIO_FILE], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def parse_blocks(path):
    raw = open(path, encoding="utf-8").read()
    # Берём ТОЛЬКО озвучиваемые секции (HOOK / BLOCK* / FINAL). Всё служебное —
    # METADATA, PEXELS QUERIES, IMAGE PROMPTS, COMPETITOR ANALYSIS, TITLE/THUMBNAIL
    # OPTIONS — отбрасывается по имени секции, а не по списку известных заголовков:
    # раньше METADATA просачивалась в озвучку и съедала первый кадр.
    parts = re.split(r'===\s*(.*?)\s*===', raw)
    kept = []
    for i in range(1, len(parts), 2):
        name = parts[i].upper()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if name.startswith(("HOOK", "BLOCK", "FINAL")):
            kept.append(body)
    content = "\n".join(kept).strip()
    if not content:                      # сценарий без === заголовков — берём как есть
        content = re.sub(r'===.*?===', '', raw).strip()
    processed = content
    for tag in sorted(PAUSE_DURATIONS, key=len, reverse=True):
        processed = processed.replace(tag, f"__PAUSE_{PAUSE_DURATIONS[tag]}__")
    processed = re.sub(r'\[.*?\]', '', processed)
    parts = re.split(r'(__PAUSE_[\d.]+__)', processed)
    blocks, cur, pause = [], "", 0.0
    for part in parts:
        m = re.match(r'__PAUSE_([\d.]+)__', part)
        if m:
            pause += float(m.group(1))
        else:
            t = part.strip()
            if t:
                if cur:
                    blocks.append({"text": cur, "pause_after": pause, "words": len(cur.split())})
                cur, pause = t, 0.0
    if cur:
        blocks.append({"text": cur, "pause_after": pause, "words": len(cur.split())})
    print(f"Блоков: {len(blocks)}")
    return blocks


def block_durations(blocks, total):
    tw = sum(b["words"] for b in blocks)
    tp = sum(b["pause_after"] for b in blocks)
    wps = tw / max(total - tp, 1)
    d = [max(MIN_CLIP, min(MAX_CLIP, b["words"] / wps + b["pause_after"])) for b in blocks]
    scale = total / sum(d)
    return [x * scale for x in d]


def pexels_photo(query, index):
    cache = os.path.join(TEMP_FOLDER, "pexels_cache")
    os.makedirs(cache, exist_ok=True)
    cf = os.path.join(cache, f"{index:04d}.jpg")
    if os.path.exists(cf):
        return cf
    if not PEXELS_API_KEY:
        return None
    try:
        q = urllib.parse.quote(query)
        req = urllib.request.Request(
            f"https://api.pexels.com/v1/search?query={q}&per_page=5&orientation=landscape",
            headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        if not data.get("photos"):
            return None
        url = data["photos"][0]["src"].get("large2x") or data["photos"][0]["src"].get("large")
        img = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(img, timeout=20) as r:
            open(cf, "wb").write(r.read())
        return cf
    except Exception as e:
        print(f"  Pexels [{query}]: {e}")
        return None


def local_photo(index):
    photos = sorted([f for f in os.listdir(MEDIA_FOLDER)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0) \
        if os.path.isdir(MEDIA_FOLDER) else []
    if not photos:
        return None
    return os.path.join(MEDIA_FOLDER, photos[index % len(photos)])


def query_for(text):
    tl = text.lower()
    for kw, q in THEMES.items():
        if kw in tl:
            return q
    return "cinematic atmospheric moody"


def kenburns(photo, out, dur):
    frames = max(1, round(dur * FPS))
    zi = int(hashlib.md5(photo.encode()).hexdigest()[:8], 16) % 2 == 0
    z = (f"'1.0+{ZOOM_END - ZOOM_START}*on/{frames}'" if zi
         else f"'{ZOOM_END}-{ZOOM_END - ZOOM_START}*on/{frames}'")
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", photo, "-vf",
           (f"scale=8000:4500:force_original_aspect_ratio=decrease,"
            f"pad=8000:4500:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"zoompan=z={z}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"),
           "-t", str(dur), "-c:v", "libx264", "-preset", "fast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"Аудио не найдено: {AUDIO_FILE}")
        return
    total = get_audio_duration()
    print(f"Аудио: {total:.1f}с ({total/60:.1f} мин)")
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    blocks = parse_blocks(SCRIPT_FILE) if os.path.exists(SCRIPT_FILE) else []
    if not blocks:
        print("Сценарий не найден/пуст")
        return
    durs = block_durations(blocks, total)
    print(f"Средний кадр: {sum(durs)/len(durs):.1f}с")

    clips = []
    use_local = os.path.isdir(MEDIA_FOLDER) and bool(local_photo(0))
    use_pexels = bool(PEXELS_API_KEY)
    for i, (b, d) in enumerate(zip(blocks, durs)):
        out = os.path.join(TEMP_FOLDER, f"clip_{i:04d}.mp4")
        if os.path.exists(out):
            clips.append(out)
            continue
        photo = local_photo(i) if use_local else None
        if not photo and use_pexels:
            photo = pexels_photo(query_for(b["text"]), i)
            if not photo:
                use_pexels = False
        if not photo:
            photo = local_photo(i)
        if not photo:
            print(f"  [{i+1}] нет фото")
            continue
        if kenburns(photo, out, d):
            clips.append(out)
            if i % 20 == 0 or i < 3:
                print(f"  [{i+1}/{len(blocks)}] {d:.1f}с {b['words']} слов")
        if use_pexels and not use_local and i % 10 == 9:
            time.sleep(0.4)

    if not clips:
        print("Нет клипов")
        return
    concat = os.path.join(TEMP_FOLDER, "concat.txt")
    # Пути ТОЛЬКО абсолютные: concat-демуксер ffmpeg резолвит относительные
    # пути от папки самого concat.txt, а не от cwd — иначе сборка падает.
    open(concat, "w", encoding="utf-8").write(
        "".join(f"file '{os.path.abspath(c)}'\n" for c in clips))
    merged = os.path.join(TEMP_FOLDER, "merged.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat, "-c", "copy", merged], capture_output=True, text=True)
    if r.returncode != 0:
        print("Склейка:", r.stderr[-300:])
        return
    r = subprocess.run(["ffmpeg", "-y", "-i", merged, "-i", AUDIO_FILE,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", OUTPUT_FILE], capture_output=True, text=True)
    if r.returncode != 0:
        print("Аудио:", r.stderr[-300:])
        return
    mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nГОТОВО: {OUTPUT_FILE} ({mb:.0f} MB, {total/60:.1f} мин, {len(clips)} кадров)")


if __name__ == "__main__":
    main()
