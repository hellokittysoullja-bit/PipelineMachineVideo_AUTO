#!/usr/bin/env python3
"""Сборка по слот-схеме с приоритетом AI > сток и хук-таймингом (ЧАСТЬ 14).
Конфиг: <video_dir>/media_plan/assemble_config.json
  { "n_slots": int, "hook_slots": int, "hook_durations": {"1":5.0,...} }
Если конфига нет — n_slots выводится из media/, хук не выделяется (равномерно).
Слот: приоритет AI-фото (_fastgen/_grok/_flow/_ai .jpg) > сток-видео (_stock_video.mp4)
> сток-фото (_stock.jpg). Фото -> Ken Burns, видео -> кроп под слот.
Usage: python scripts/assemble.py <video_dir>"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

FPS, WIDTH, HEIGHT = 25, 1920, 1080
ZOOM_START, ZOOM_END = 1.0, 1.12
# Панорамирование Ken Burns: смещение центра растёт от 0 пропорционально
# (zoom-ZOOM_START), на полном кадре (zoom=1.0) offset строго 0. PAN_MAX_FRAC
# с запасом ниже безопасного предела (1-1/ZOOM_END)/2 ≈ 0.0536.
PAN_MAX_FRAC = 0.045
PAN_DIRECTIONS = [(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1)]


def find_audio(video_dir):
    for name in ("audio_fixed.mp3", "audio.mp3"):
        p = os.path.join(video_dir, name)
        if os.path.exists(p):
            return p
    mp3s = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp3")]
    return os.path.join(video_dir, mp3s[0]) if mp3s else None


def dur(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def load_config(video_dir, media_dir):
    p = os.path.join(video_dir, "media_plan", "assemble_config.json")
    if os.path.exists(p):
        c = json.load(open(p, encoding="utf-8"))
        hd = {int(k): float(v) for k, v in c.get("hook_durations", {}).items()}
        if "n_slots" not in c:      # раньше падало KeyError без единого пояснения
            print(f"В {p} нет обязательного поля n_slots — считаю слоты по media/")
        else:
            return int(c["n_slots"]), int(c.get("hook_slots", 0)), hd
    nums = set()
    for f in glob.glob(os.path.join(media_dir, "*")):
        m = re.match(r'(\d+)_', os.path.basename(f))
        if m:
            nums.add(int(m.group(1)))
    return (max(nums) if nums else 0), 0, {}


def resolve_slot(media_dir, n):
    for suf in ("_fastgen.jpg", "_grok.jpg", "_flow.jpg", "_ai.jpg"):
        p = os.path.join(media_dir, f"{n:03d}{suf}")
        if os.path.exists(p):
            return "photo", p
    v = os.path.join(media_dir, f"{n:03d}_stock_video.mp4")
    if os.path.exists(v):
        return "video", v
    p = os.path.join(media_dir, f"{n:03d}_stock.jpg")
    if os.path.exists(p):
        return "photo", p
    return None, None


def kenburns_clip(photo, out, d):
    frames = max(1, round(d * FPS))
    h = int(hashlib.md5(photo.encode()).hexdigest()[:8], 16)
    zi = h % 2 == 0
    dx, dy = PAN_DIRECTIONS[(h // 2) % len(PAN_DIRECTIONS)]
    z = (f"'1.0+{ZOOM_END - ZOOM_START}*on/{frames}'" if zi
         else f"'{ZOOM_END}-{ZOOM_END - ZOOM_START}*on/{frames}'")
    pan_scale = f"(zoom-{ZOOM_START})/{ZOOM_END - ZOOM_START}"
    x = (f"'iw/2-(iw/zoom/2)+{dx}*{PAN_MAX_FRAC}*iw*{pan_scale}'" if dx
         else "'iw/2-(iw/zoom/2)'")
    y = (f"'ih/2-(ih/zoom/2)+{dy}*{PAN_MAX_FRAC}*ih*{pan_scale}'" if dy
         else "'ih/2-(ih/zoom/2)'")
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", photo, "-vf",
           (f"scale=8000:4500:force_original_aspect_ratio=decrease,"
            f"pad=8000:4500:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"zoompan=z={z}:x={x}:y={y}:"
            f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"),
           "-t", str(d), "-c:v", "libx264", "-preset", "fast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def video_clip(vid, out, d):
    try:
        actual = dur(vid)
    except Exception:
        actual = d
    vf = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1"
    if actual < d - 0.05:
        vf += f",setpts={d/actual:.5f}*PTS"
    cmd = ["ffmpeg", "-y", "-i", vid, "-vf", vf, "-t", str(d), "-an",
           "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    media_dir = os.path.join(video_dir, "media")
    temp = os.path.join(video_dir, "temp_assemble")
    out_file = os.path.join(video_dir, "final.mp4")
    os.makedirs(temp, exist_ok=True)

    audio = find_audio(video_dir)
    if not audio:
        print("Аудио не найдено")
        return
    audio_dur = dur(audio)
    n_slots, hook_slots, hook_dur = load_config(video_dir, media_dir)
    if n_slots <= 0:
        print("Слотов не найдено (пустой media/ и нет конфига)")
        return
    hook_total = sum(hook_dur.values())
    body_slots = n_slots - hook_slots
    if hook_total >= audio_dur and body_slots > 0:
        # хук длиннее всей озвучки, а тело ещё есть -> телу доставалась
        # нулевая/отрицательная длительность и ffmpeg падал на каждом слоте.
        # Когда ВСЕ слоты хуковые (body_slots == 0), это не ошибка: длительности
        # заданы явно, лишний хвост подрежет -shortest.
        print(f"Хук ({hook_total:.1f}с) не короче всего аудио ({audio_dur:.1f}с) — "
              f"проверь hook_durations. Раскладываю равномерно.")
        hook_dur, hook_slots, hook_total, body_slots = {}, 0, 0.0, n_slots
    body_d = (audio_dur - hook_total) / max(body_slots, 1)

    # проверка хук-тайминга (ЧАСТЬ 14, ХУК-МЕДИА)
    if hook_slots and hook_total > 0:
        print(f"Хук: {hook_slots} слотов, сумма {hook_total:.1f}с | тело {body_slots} по {body_d:.2f}с")
    print(f"Аудио {audio_dur:.1f}с ({audio_dur/60:.1f} мин), слотов {n_slots}")

    clips, missing = [], []
    for slot in range(1, n_slots + 1):
        out = os.path.join(temp, f"clip_{slot:04d}.mp4")
        if os.path.exists(out):
            clips.append(out)
            continue
        kind, path = resolve_slot(media_dir, slot)
        if not path:
            missing.append(slot)
            continue
        d = hook_dur.get(slot, body_d) if slot <= hook_slots else body_d
        ok = video_clip(path, out, d) if kind == "video" else kenburns_clip(path, out, d)
        if ok:
            clips.append(out)
            if slot % 20 == 0 or slot <= 3:
                print(f"[{slot:03d}/{n_slots}] OK <- {os.path.basename(path)}", flush=True)
        else:
            missing.append(slot)
            print(f"[{slot:03d}] FFMPEG FAIL", flush=True)

    print(f"\nКлипов: {len(clips)}/{n_slots}, пропущено: {len(missing)}")
    if missing:
        print("Пропущены:", missing)
    if not clips:
        return
    concat = os.path.join(temp, "concat.txt")
    # Пути ТОЛЬКО абсолютные: concat-демуксер ffmpeg резолвит относительные
    # пути от папки самого concat.txt, а не от cwd — иначе сборка падает.
    open(concat, "w", encoding="utf-8").write(
        "".join(f"file '{os.path.abspath(c)}'\n" for c in clips))
    merged = os.path.join(temp, "merged.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat, "-c", "copy", merged], capture_output=True, text=True)
    if r.returncode != 0:
        print("Склейка:", r.stderr[-400:])
        return
    r = subprocess.run(["ffmpeg", "-y", "-i", merged, "-i", audio,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", out_file], capture_output=True, text=True)
    if r.returncode != 0:
        print("Аудио:", r.stderr[-400:])
        return
    mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\nГОТОВО: {out_file} ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
