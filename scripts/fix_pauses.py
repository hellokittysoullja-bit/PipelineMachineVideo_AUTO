#!/usr/bin/env python3
"""Подрезка длинных пауз в озвучке (ЧАСТЬ 9, п.7).
Находит тишину длиннее THRESH_SEC и укорачивает её до KEEP_SEC.
Usage: python scripts/fix_pauses.py <video_dir>
Вход: audio.mp3 (или первый *.mp3). Выход: audio_fixed.mp3."""
import os
import re
import subprocess
import sys

THRESH_SEC = 1.0     # тишина длиннее этого — подрезается
KEEP_SEC = 0.6       # до какой длины оставляем паузу
NOISE_DB = "-30dB"   # порог тишины


def find_audio(video_dir):
    for name in ("audio.mp3",):
        p = os.path.join(video_dir, name)
        if os.path.exists(p):
            return p
    mp3s = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp3")
            and "fixed" not in f.lower()]
    return os.path.join(video_dir, mp3s[0]) if mp3s else None


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def detect_silences(path):
    r = subprocess.run(["ffmpeg", "-i", path, "-af",
                        f"silencedetect=noise={NOISE_DB}:d={THRESH_SEC}",
                        "-f", "null", "-"], capture_output=True, text=True)
    log = r.stderr
    starts = [float(x) for x in re.findall(r'silence_start:\s*([\d.]+)', log)]
    ends = [float(x) for x in re.findall(r'silence_end:\s*([\d.]+)', log)]
    return list(zip(starts, ends))


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    src = find_audio(video_dir)
    if not src:
        print("Аудио не найдено (audio.mp3)")
        return
    out = os.path.join(video_dir, "audio_fixed.mp3")
    total = duration(src)
    sil = detect_silences(src)
    if not sil:
        print("Длинных пауз не найдено — копирую как есть.")
        subprocess.run(["ffmpeg", "-y", "-i", src, "-c", "copy", out], capture_output=True)
        print(f"Готово: {out}")
        return

    # Строим сегменты: речь целиком + каждая тишина обрезана до KEEP_SEC
    segments = []
    prev = 0.0
    for ss, se in sil:
        if ss > prev:
            segments.append((prev, ss))
        segments.append((ss, min(se, ss + KEEP_SEC)))
        prev = se
    if prev < total:
        segments.append((prev, total))

    parts, filt = [], ""
    for i, (a, b) in enumerate(segments):
        if b - a <= 0.02:
            continue
        filt += f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}];"
        parts.append(f"[a{i}]")
    filt += "".join(parts) + f"concat=n={len(parts)}:v=0:a=1[out]"

    cmd = ["ffmpeg", "-y", "-i", src, "-filter_complex", filt,
           "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "192k", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("Ошибка ffmpeg:", r.stderr[-400:])
        return
    print(f"Готово: {out} | подрезано пауз: {len(sil)} | было {total:.1f}с → стало {duration(out):.1f}с")


if __name__ == "__main__":
    main()
