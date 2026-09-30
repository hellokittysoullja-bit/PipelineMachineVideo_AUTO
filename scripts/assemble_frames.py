#!/usr/bin/env python3
"""Сборка final.mp4 из frames/NNN.png и озвучки.

1. Тайминг: frame_timing.frame_durations (alignment, иначе оценка).
2. Кадр вписывается в 1920x1080 ЦЕЛИКОМ (подписи у края не обрезаются):
   пустые поля — размытая копия самого кадра.
3. Каждый кадр — медленный наезд или отъезд (5%, по центру) на ВСЮ длину
   клипа: зум зависит от on/frames, а не от инкремента (ЧАСТЬ 6 исходного
   пайплайна — инкремент упирается в максимум и камера встаёт).
4. Клипы склеиваются жёсткими резами точно по кадровой сетке (ошибка
   округления переносится на следующий клип и не копится).
5. Звук: голос (audio_fixed.flac, если fix_pauses.py отработал) + музыка из
   assets/music (если есть) с уровнем от ЗАМЕРА громкости обеих дорожек и
   приглушением под голос; итог — loudnorm -14 LUFS.
6. Рядом пишется final.srt (субтитры по тем же стартам) для загрузки на YouTube.

Недостающий кадр (генерация не удалась) заменяется соседним и попадает в
отчёт — ролик не собирается с чёрной дырой.

Usage: python scripts/assemble_frames.py <video_dir>"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel  # noqa: E402
import frame_timing  # noqa: E402
import script_parser  # noqa: E402

FPS, W, H = 24, 1920, 1080
ZOOM = 0.05
FIT_SAFE = 0.96            # кадр чуть меньше холста: наезд 5% не срезает подписи у края
MUSIC_GAP_LU = 18.0        # музыка тише голоса на 18 LU (фон, не спорит со словами)
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"
CRF = "20"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {' '.join(cmd[:6])}...\n{r.stderr[-1500:]}")
    return r


def media_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    return float(r.stdout.strip())


def find_audio(video_dir):
    fixed = os.path.join(video_dir, "audio_fixed.flac")
    if os.path.exists(fixed):
        return fixed, True
    for name in ("audio.mp3", "audio.wav", "audio.flac", "audio.m4a"):
        p = os.path.join(video_dir, name)
        if os.path.exists(p):
            return p, False
    return None, False


def fit_canvas(src, dst):
    """Вписать кадр в 16:9 целиком; поля — размытая растянутая копия."""
    from PIL import Image, ImageFilter, ImageOps
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    bg = im.resize((W, H)).filter(ImageFilter.GaussianBlur(40))
    s = min(W / im.width, H / im.height) * FIT_SAFE
    fg = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
    bg.save(dst, "PNG")


def quantize(durs):
    out, carry = [], 0.0
    for d in durs:
        target = d + carry
        q = max(1, round(target * FPS)) / FPS
        carry = target - q
        out.append(q)
    return out


def clip(canvas, out, dur, zoom_in):
    frames = max(1, round(dur * FPS))
    z = f"1.0+{ZOOM}*on/{frames}" if zoom_in else f"{1 + ZOOM}-{ZOOM}*on/{frames}"
    vf = (f"scale={W * 2}:{H * 2},zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
          f":d={frames}:s={W}x{H}:fps={FPS},format=yuv420p")
    tmp = out + ".tmp.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-loop", "1", "-i", canvas,
         "-vf", vf, "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium", "-crf", CRF,
         "-r", str(FPS), tmp])
    os.replace(tmp, out)


def resolve_frames(video_dir, n):
    paths = [os.path.join(video_dir, "frames", f"{i + 1:03d}.png") for i in range(n)]
    have = [p if os.path.exists(p) and os.path.getsize(p) > 0 else None for p in paths]
    if not any(have):
        sys.exit("Нет ни одного кадра в frames/ — сначала frame_generator.py")
    missing = [i for i, p in enumerate(have) if p is None]
    for i in missing:                       # сосед: предыдущий, иначе следующий
        prev = next((have[j] for j in range(i - 1, -1, -1) if have[j]), None)
        have[i] = prev or next(have[j] for j in range(i + 1, n) if have[j])
    return have, missing


def loudness(path):
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.findall(r"I:\s*(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def music_file():
    files = sorted(glob.glob(os.path.join(channel.ROOT, "assets", "music", "*")))
    files = [f for f in files if f.lower().endswith((".mp3", ".flac", ".wav", ".ogg", ".m4a"))]
    return files[0] if files else None


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(path, blocks, starts, durs):
    with open(path, "w", encoding="utf-8") as f:
        for i, (b, s, d) in enumerate(zip(blocks, starts, durs), 1):
            f.write(f"{i}\n{srt_time(s)} --> {srt_time(s + d)}\n{b['text']}\n\n")


def main():
    video_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    blocks = script_parser.parse_blocks(os.path.join(video_dir, "script.txt"))
    audio, fixed = find_audio(video_dir)
    if not audio:
        sys.exit("Нет озвучки (audio.mp3) в папке ролика")
    total = media_duration(audio)
    starts, durs, treport = frame_timing.frame_durations(video_dir, blocks, total, fixed_audio=fixed)
    durs = quantize(durs)
    frames, missing = resolve_frames(video_dir, len(blocks))
    print(f"Кадров: {len(blocks)}, аудио {total:.1f} с, тайминг: {treport['source']} "
          f"(найдено в alignment {treport['found_in_alignment']}/{treport['blocks']})"
          + (f", заменены соседом: {[i + 1 for i in missing]}" if missing else ""))

    work = os.path.join(video_dir, "temp_render")
    os.makedirs(work, exist_ok=True)

    def render(i):
        src = frames[i]
        h = hashlib.md5(open(src, "rb").read()).hexdigest()[:12]
        key = hashlib.md5(f"{h}|{durs[i]:.5f}|{i % 2}|{ZOOM}|{FIT_SAFE}|{FPS}|{CRF}".encode()).hexdigest()[:16]
        out = os.path.join(work, f"clip_{i:04d}_{key}.mp4")
        if not os.path.exists(out):
            canvas = os.path.join(work, f"canvas_{h}.png")
            if not os.path.exists(canvas):
                # у заменённого кадра картинка общая с соседом: два потока пишут
                # один холст — пишем во временный файл потока и атомарно подменяем
                tmp = f"{canvas}.{threading.get_ident()}.png"
                fit_canvas(src, tmp)
                os.replace(tmp, canvas)
            clip(canvas, out, durs[i], zoom_in=(i % 2 == 0))
        return out

    workers = int(os.environ.get("RENDER_WORKERS", str(max(1, (os.cpu_count() or 2) - 1))))
    with ThreadPoolExecutor(workers) as ex:
        clips = list(ex.map(render, range(len(blocks))))

    lst = os.path.join(work, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.writelines(f"file '{c}'\n" for c in clips)
    video = os.path.join(work, "video.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video])

    music = music_file()
    final = os.path.join(video_dir, "final.mp4")
    if music:
        lv, lm = loudness(audio), loudness(music)
        gain = max(-40.0, min(6.0, (lv - MUSIC_GAP_LU - lm))) if lv is not None and lm is not None else -20.0
        af = (f"[2:a]aloop=loop=-1:size=2e9,atrim=0:{total:.3f},volume={gain:.2f}dB[m];"
              f"[1:a]asplit=2[v][sc];[m][sc]sidechaincompress=threshold=0.05:ratio=4:attack=20:release=400[md];"
              f"[v][md]amix=inputs=2:duration=first:normalize=0,{LOUDNORM}[a]")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-i", audio, "-i", music,
               "-filter_complex", af, "-map", "0:v", "-map", "[a]"]
        print(f"Музыка: {os.path.basename(music)}, уровень {gain:+.1f} дБ (замер: голос {lv}, музыка {lm} LUFS)")
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-i", audio, "-af", LOUDNORM,
               "-map", "0:v", "-map", "1:a"]
    run(cmd + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
               "-movflags", "+faststart", final])
    cum = [sum(durs[:i]) for i in range(len(durs))]     # старты по той же кадровой сетке, что и видео
    write_srt(os.path.join(video_dir, "final.srt"), blocks, cum, durs)
    real = media_duration(final)
    report = {"frames": len(blocks), "missing_replaced": [i + 1 for i in missing], "timing": treport,
              "audio_sec": round(total, 3), "video_sec": round(real, 3), "music": music}
    os.makedirs(os.path.join(video_dir, "media_plan"), exist_ok=True)
    with open(os.path.join(video_dir, "media_plan", "assemble_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"Готово. Файл: {final} ({real:.1f} с)")


if __name__ == "__main__":
    main()
