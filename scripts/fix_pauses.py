#!/usr/bin/env python3
"""Подрезка длинных пауз в озвучке + нормализация громкости (ЧАСТЬ 9, п.7).
Находит тишину длиннее THRESH_SEC и укорачивает её до KEEP_SEC.
Двухпроходный loudnorm (EBU R128) поверх — гуляющая громкость между
блоками/эпизодами звучит непрофессионально и это бесплатно чинится.
Usage: python scripts/fix_pauses.py <video_dir>
Вход: audio.mp3 (или первый *.mp3). Выход: audio_fixed.mp3."""
import json
import os
import re
import subprocess
import sys

THRESH_SEC = 1.0     # тишина длиннее этого — подрезается
KEEP_SEC = 0.6       # до какой длины оставляем паузу
NOISE_DB = "-30dB"   # порог тишины
LOUDNORM_TARGET = "I=-16:TP=-1.5:LRA=11"   # стандарт для закадрового голоса


def measure_loudness(path):
    """Первый проход loudnorm: только измерение, ничего не меняет в файле."""
    r = subprocess.run(["ffmpeg", "-i", path, "-af",
                        f"loudnorm={LOUDNORM_TARGET}:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', r.stderr, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def loudnorm_filter(stats):
    """Второй проход: если есть замер с первого — линейная точная нормализация,
    иначе одиночный проход loudnorm (менее точно, но не падает)."""
    if not stats:
        return f"loudnorm={LOUDNORM_TARGET}"
    return (f"loudnorm={LOUDNORM_TARGET}:"
            f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
            f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
            f"offset={stats.get('target_offset', 0)}:linear=true:print_format=summary")


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
        return 1
    out = os.path.join(video_dir, "audio_fixed.mp3")
    total = duration(src)
    sil = detect_silences(src)
    loud = loudnorm_filter(measure_loudness(src))
    if not sil:
        print("Длинных пауз не найдено — нормализую громкость.")
        r = subprocess.run(["ffmpeg", "-y", "-i", src, "-af", loud,
                            "-c:a", "libmp3lame", "-b:a", "192k", out],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            print("Ошибка ffmpeg:", r.stderr[-400:])
            return 1
        print(f"Готово: {out}")
        return 0

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
    if not parts:
        # все сегменты оказались короче 0.02с — склеивать нечего, ffmpeg бы
        # упал на concat=n=0; отдаём исходник без изменений (кроме громкости)
        print("Нечего склеивать — нормализую громкость исходника.")
        r = subprocess.run(["ffmpeg", "-y", "-i", src, "-af", loud,
                            "-c:a", "libmp3lame", "-b:a", "192k", out],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            print("Ошибка ffmpeg:", r.stderr[-400:])
            return 1
        print(f"Готово: {out}")
        return 0
    filt += "".join(parts) + f"concat=n={len(parts)}:v=0:a=1[c];[c]{loud}[out]"

    cmd = ["ffmpeg", "-y", "-i", src, "-filter_complex", filt,
           "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "192k", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        print("Ошибка ffmpeg:", r.stderr[-400:])
        return 1
    print(f"Готово: {out} | подрезано пауз: {len(sil)} | было {total:.1f}с → стало {duration(out):.1f}с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
