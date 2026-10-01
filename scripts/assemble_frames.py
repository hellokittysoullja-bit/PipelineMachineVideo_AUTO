#!/usr/bin/env python3
"""Сборка final.mp4 из frames/NNN.png и озвучки.

Картинка — нарочно простая (рисованной объяснялке сложная камера не нужна):
  * кадр вписывается в 16:9 ЦЕЛИКОМ, поля — размытая копия самого кадра
    (подписи у края не режутся);
  * медленный наезд или отъезд 4% по центру на ВСЮ длину клипа: зум от
    on/frames, а не инкрементом (ЧАСТЬ 6 старого CLAUDE.md — инкремент
    упирается в максимум и камера встаёт);
  * жёсткие резы по кадровой сетке, ошибка округления переносится на
    следующий клип и не копится; клип проверяется ffprobe (код 0 ffmpeg не
    гарантирует, что записана нужная длина).

Тайминг и звук — код старого генератора (render_core, дословно):
  * PHRASE LOCK: рез ровно на начале следующей фразы по alignment озвучки;
    не сошлось — оценка по реальной длине речи блоков и громко в логе;
  * голос: срез низов, EQ, де-эссер, компрессор; музыка (первый файл в
    assets/music, если есть) с уровнем по ЗАМЕРУ и приглушением под голос
    теми же параметрами; двухпроходный loudnorm -14 LUFS + лимитер;
    проверка звука входа и готового файла;
  * субтитры (2 строки, ~42 символа) и главы YouTube.

Кадр, который генератор пометил rejected/failed или которого нет, на экран
не идёт — его время получает предыдущий кадр («ни карточек, ни повторов»).
Пишет media_plan/phrase_timeline.json в формате старого рендера — по нему
verify_timing.py меряет резы в пикселях готового файла.

Коды: 0 — чисто, 2 — собран с замечаниями, 1 — не собран.
Usage: python scripts/assemble_frames.py <video_dir>"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_WARN = 0, 1, 2
FPS, W, H = 24, 1920, 1080
ZOOM = 0.04
FIT_SAFE = 0.96          # кадр чуть меньше холста: наезд 4% не срезает подписи у края
CRF = "18"
CLIP_TOLERANCE = 0.5 / FPS


def run(cmd, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}...: {r.stderr[-800:]}")
    return r


def probe_duration(path):
    try:
        return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], 60).stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def fit_canvas(src, dst):
    from PIL import Image, ImageFilter, ImageOps
    with Image.open(src) as im0:
        im = ImageOps.exif_transpose(im0).convert("RGB")
    bg = im.resize((W, H)).filter(ImageFilter.GaussianBlur(40))
    s = min(W / im.width, H / im.height) * FIT_SAFE
    fg = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
    tmp = f"{dst}.{threading.get_ident()}.png"
    bg.save(tmp, "PNG")
    os.replace(tmp, dst)


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
    for _attempt in range(2):
        try:
            run(["ffmpeg", "-y", "-v", "error", "-framerate", "1", "-loop", "1", "-i", canvas, "-vf", vf,
                 "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium", "-crf", CRF,
                 "-r", str(FPS), tmp])
            d = probe_duration(tmp)
            if d is not None and abs(d - frames / FPS) <= CLIP_TOLERANCE + 1e-6:
                os.replace(tmp, out)
                return True
        except Exception:  # noqa: BLE001 — одна повторная попытка
            pass
    return False


def load_statuses(video_dir):
    try:
        return {r["index"]: r.get("status") for r in json.load(
            open(os.path.join(video_dir, "media_plan", "frames_report.json"), encoding="utf-8"))["frames"]}
    except (OSError, ValueError, KeyError):
        return {}


def kept_frames(video_dir, n):
    statuses = load_statuses(video_dir)
    kept, absorbed = [], []
    for i in range(n):
        path = os.path.join(video_dir, "frames", f"{i + 1:03d}.png")
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            absorbed.append({"index": i, "reason": "no_frame"})
        elif statuses.get(i) in ("rejected", "failed"):
            absorbed.append({"index": i, "reason": statuses[i]})
        else:
            kept.append(i)
    return kept, absorbed


def find_audio(video_dir):
    for name in ("audio_fixed.flac", "audio.mp3", "audio.wav", "audio.flac", "audio.m4a"):
        p = os.path.join(video_dir, name)
        if os.path.exists(p):
            return p
    return None


def music_file():
    files = sorted(f for f in glob.glob(os.path.join(env.ROOT, "assets", "music", "*"))
                   if f.lower().endswith((".mp3", ".flac", ".wav", ".ogg", ".m4a")))
    return files[0] if files else None


def main(video_dir):
    env.load_env()
    import render_core as rc
    import script_parser

    video_dir = os.path.abspath(video_dir)
    rc.configure(video_dir)
    blocks = script_parser.parse_blocks(os.path.join(video_dir, "script.txt"))
    audio = find_audio(video_dir)
    if not audio:
        print("Нет озвучки (audio.mp3) в папке ролика")
        return EXIT_FAILED
    rc.audio_qc(audio)
    total = rc.get_media_duration(audio)

    kept, absorbed = kept_frames(video_dir, len(blocks))
    if not kept:
        print("Ни одного проверенного кадра — сборка остановлена (ролик из брака не собирается).")
        return EXIT_FAILED
    for a in absorbed:
        print(f"  [{a['index'] + 1}] кадра нет ({a['reason']}) — время отдано соседнему кадру")

    real_weights = rc.load_alignment_weights(blocks)
    onsets = rc.load_alignment_onsets(blocks)
    if onsets:
        starts, timing = [0.0] + list(onsets[1:]), "phrase_lock"
    else:
        f = rc.ALIGNMENT_ONSET_FAILURE or {"reason": "alignment не найден"}
        print(f"  ВНИМАНИЕ: PHRASE LOCK ВЫКЛЮЧЕН — {f['reason']}. Кадры по ОЦЕНКЕ длины речи.")
        words = [max(1, b["words"]) + 2 * b.get("pause_after", 0.0) for b in blocks]
        starts, acc = [], 0.0
        for w in words:
            starts.append(acc)
            acc += total * w / sum(words)
        timing = "estimate"
    base = [(starts[i + 1] if i + 1 < len(starts) else total) - starts[i] for i in range(len(starts))]

    kept_starts = [0.0] + [starts[i] for i in kept[1:]]
    durs = quantize([e - s for s, e in zip(kept_starts, kept_starts[1:] + [total])])
    print(f"Тайминг: {timing}; кадров на экране {len(kept)} из {len(blocks)}, средний {sum(durs) / len(durs):.1f} с")

    mp = os.path.join(video_dir, "media_plan")
    os.makedirs(mp, exist_ok=True)
    with open(os.path.join(mp, "phrase_timeline.json"), "w", encoding="utf-8") as fh:
        json.dump({"locked": timing == "phrase_lock", "fps": FPS, "audio_total_sec": total,
                   "blocks": [{"index": i, "section": blocks[i]["section"], "text": blocks[i]["text"][:120],
                               "speech_onset_sec": kept_starts[k] if timing == "phrase_lock" else None,
                               "duration_sec": durs[k]} for k, i in enumerate(kept)]},
                  fh, ensure_ascii=False, indent=2)
    rc.write_subtitles(video_dir, blocks, starts, base, real_weights=real_weights)
    rc.write_chapters(video_dir, blocks, starts)

    work = os.path.join(video_dir, "temp_render")
    os.makedirs(work, exist_ok=True)

    def render(k):
        i = kept[k]
        src = os.path.join(video_dir, "frames", f"{i + 1:03d}.png")
        h = hashlib.md5(open(src, "rb").read()).hexdigest()[:12]
        key = hashlib.md5(f"{h}|{durs[k]:.5f}|{k % 2}|{ZOOM}|{FIT_SAFE}|{FPS}|{CRF}".encode()).hexdigest()[:16]
        out = os.path.join(work, f"clip_{k:04d}_{key}.mp4")
        if os.path.exists(out) and probe_duration(out) is not None:
            return out
        canvas = os.path.join(work, f"canvas_{h}.png")
        if not os.path.exists(canvas):
            fit_canvas(src, canvas)
        return out if clip(canvas, out, durs[k], zoom_in=(k % 2 == 0)) else None

    workers = int(os.environ.get("RENDER_WORKERS", str(max(1, (os.cpu_count() or 2) - 1))))
    with ThreadPoolExecutor(workers) as ex:
        clips = list(ex.map(render, range(len(kept))))
    failed = [kept[k] + 1 for k, c in enumerate(clips) if not c]
    with open(os.path.join(mp, "render_manifest.json"), "w", encoding="utf-8") as fh:
        status = {i: {"index": i, "status": "ok" if clips[k] else "failed", "dur": round(durs[k], 4)}
                  for k, i in enumerate(kept)}
        status.update({a["index"]: {"index": a["index"], "status": "absorbed", "reason": a["reason"]}
                       for a in absorbed})
        json.dump({"total_blocks": len(blocks), "timing": timing, "clips": [status[i] for i in sorted(status)]},
                  fh, ensure_ascii=False, indent=1)
    if failed:
        print(f"СТОП: клипы не собраны {failed} — final.mp4 НЕ собирается (перезапуск достроит только их).")
        return EXIT_FAILED

    lst = os.path.join(work, "concat.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        fh.writelines(f"file '{c}'\n" for c in clips)
    video = os.path.join(work, "video.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video])

    # --- звук: цепочка старого генератора
    voice = rc.process_voice(audio, os.path.join(work, "voice_processed.wav"))
    music = music_file()
    premix = voice
    if music:
        gain, why = rc.music_bed_gain_db(voice, music)
        premix = os.path.join(work, "premix.wav")
        run(["ffmpeg", "-y", "-v", "error", "-i", voice, "-stream_loop", "-1", "-i", music, "-filter_complex",
             f"[1:a]atrim=0:{total:.3f},volume={gain:.2f}dB[m];[0:a]asplit=2[v][sc];"
             f"[m][sc]sidechaincompress=threshold={rc.MUSIC_DUCK_THRESHOLD}:ratio={rc.MUSIC_DUCK_RATIO}:"
             f"attack={rc.MUSIC_DUCK_ATTACK_MS}:release={rc.MUSIC_DUCK_RELEASE_MS}[md];"
             f"[v][md]amix=inputs=2:duration=first:normalize=0[a]", "-map", "[a]", "-ar", "48000", premix])
        print(f"  Музыка: {os.path.basename(music)}, {gain:+.1f} дБ ({why})")
    af = rc.build_master_af(rc.measure_loudnorm_stats(premix), max(0.0, total - 2.0), 0.05)
    final = os.path.join(video_dir, "final.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", premix, "-af", af, "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
         "-movflags", "+faststart", final + ".tmp.mp4"])
    os.replace(final + ".tmp.mp4", final)
    rc.audio_qc(final, label="Audio QC финала")
    print(f"Готово. Файл: {final}")
    return EXIT_WARN if absorbed or timing != "phrase_lock" else EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
