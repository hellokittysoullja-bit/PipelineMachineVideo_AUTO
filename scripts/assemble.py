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

try:
    import numpy as np
except ImportError:
    np = None   # аудио-ритм по громкости — опциональная фича, без numpy просто выключена

FPS, WIDTH, HEIGHT = 25, 1920, 1080
# ZOOM_FLOOR — минимальный зум держится ВЕСЬ клип (не 1.0). Раньше offset пана
# был обязан = 0 ровно в момент zoom=1.0 (иначе край вылезет за картинку), и на
# каждом втором клипе (zoom-out) кадр половину времени стоял мёртвым по центру.
# С постоянным полом запас под пан есть на любом кадре клипа, а не только к
# концу движения зума.
ZOOM_FLOOR = 1.04
# Скорость зума задаётся в долях/сек, а не фиксированным приростом на клип —
# иначе 4-секундный и 20-секундный кадр "дышат" с заметно разной скоростью.
ZOOM_RATE_BASE = 0.010
ZOOM_DELTA_MIN, ZOOM_DELTA_MAX = 0.05, 0.22
# Панорамирование считается от РЕАЛЬНОГО zoom в каждый момент кадра — (1-1/zoom)/2
# это точный геометрический запас смещения без вылета за картинку, безопасно по
# построению на любом кадре, поэтому PAN_SAFETY можно брать ближе к пределу, чем
# раньше (когда запас считался один раз заранее от конечного zoom клипа).
PAN_SAFETY = 0.9
PAN_JITTER_MIN, PAN_JITTER_MAX = 0.6, 1.0   # органический разброс силы пана по клипам
PAN_DIRECTIONS = [(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1)]
XFADE_DUR = 0.4        # диссолв на границе хук/тело и часть обычных склеек
XFADE_DUR_HARD = 0.06  # почти мгновенный переход — читается как жёсткий cut
XFADE_TRANSITIONS = ["fade", "dissolve", "smoothleft", "smoothright", "smoothup", "smoothdown"]


def film_look(source, photo_hash):
    # Один и тот же грейд на все слоты — тоже штамп, если приглядеться. Лёгкий
    # hash-джиттер контраста/сатурации/яркости на каждый клип убирает эту
    # одинаковость. Зерно — по источнику: сильнее на AI (маскирует "пластик"
    # генерации), почти незаметно на честном стоке (не раздувает битрейт зря).
    c = 1.03 + (photo_hash % 100) / 100 * 0.05          # 1.03-1.08
    s = 1.04 + ((photo_hash >> 7) % 100) / 100 * 0.09    # 1.04-1.13
    b = ((photo_hash >> 14) % 100) / 100 * 0.02          # 0.00-0.02
    grain = 6 if source == "ai" else 1
    return f"eq=contrast={c:.3f}:saturation={s:.3f}:brightness={b:.3f},vignette=PI/5,noise=alls={grain}:allf=t+u"


def pick_no_repeat(history, candidate, options, max_repeat):
    """Хэш даёт разнообразие "в среднем", но не мешает 3-4 одинаковым подряд
    случайным совпадением. Если последние max_repeat решений совпадают с
    кандидатом — детерминированно берём следующий вариант по кругу вместо
    него. history мутируется на месте (список последних решений)."""
    if len(history) >= max_repeat and all(x == candidate for x in history[-max_repeat:]):
        idx = options.index(candidate) if candidate in options else 0
        candidate = options[(idx + 1) % len(options)]
    history.append(candidate)
    del history[:-(max_repeat + 2)]
    return candidate


def audio_energy_curve(audio_path, window_sec=1.0):
    """RMS-громкость аудио по окнам — без Whisper/librosa, чистый PCM+numpy.
    Возвращает (rms_array, window_sec) или None, если numpy недоступен/что-то
    пошло не так (фича опциональная, пайплайн не должен падать без неё)."""
    if np is None:
        return None
    sr = 8000   # нужна только огибающая громкости, не звук — низкий sr достаточно и быстро
    try:
        r = subprocess.run(["ffmpeg", "-v", "quiet", "-i", audio_path, "-f", "s16le",
                            "-ac", "1", "-ar", str(sr), "-"], capture_output=True, timeout=120)
        if r.returncode != 0 or not r.stdout:
            return None
        raw = r.stdout[:len(r.stdout) - len(r.stdout) % 2]
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if len(samples) < sr:
            return None
        win = max(1, int(sr * window_sec))
        n_win = max(1, len(samples) // win)
        trimmed = samples[:n_win * win].reshape(n_win, win)
        rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
        return rms, window_sec
    except Exception as e:
        print(f"  Анализ громкости не удался (пропускаю): {e}")
        return None


def energy_pace_multipliers(curve, starts, durs, lo=0.8, hi=1.25):
    """Громче участок — короче кадры (множитель <1), тише — длиннее (>1).
    Множитель считается от отношения к МЕДИАНЕ громкости всей дорожки,
    чтобы тихий ролик целиком не растягивал все кадры одинаково."""
    if curve is None:
        return [1.0] * len(durs)
    rms, window_sec = curve
    med = float(np.median(rms))
    if med <= 0:
        return [1.0] * len(durs)
    mults = []
    for t0, d in zip(starts, durs):
        i0 = max(0, int(t0 / window_sec))
        i1 = min(len(rms), max(i0 + 1, int((t0 + d) / window_sec)))
        seg = rms[i0:i1] if i0 < len(rms) else rms[-1:]
        e = float(seg.mean()) if len(seg) else med
        ratio = max(0.5, min(2.0, e / med))
        mults.append(max(lo, min(hi, 1.0 / ratio)))
    return mults


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
            return "photo", p, "ai"
    v = os.path.join(media_dir, f"{n:03d}_stock_video.mp4")
    if os.path.exists(v):
        return "video", v, "stock"
    p = os.path.join(media_dir, f"{n:03d}_stock.jpg")
    if os.path.exists(p):
        return "photo", p, "stock"
    return None, None, None


def kb_hash_choices(photo):
    """Кандидаты зума/пана по хэшу файла — детерминированно, но БЕЗ памяти о
    соседних клипах (это даёт anti-repetition в main(), см. pick_no_repeat)."""
    h = int(hashlib.md5(photo.encode()).hexdigest()[:8], 16)
    zoom_in = bool(h & 1)
    pan_dir = PAN_DIRECTIONS[(h >> 1) % len(PAN_DIRECTIONS)]
    return h, zoom_in, pan_dir


def kenburns_clip(photo, out, d, source="stock", zoom_in=None, pan_dir=None):
    frames = max(1, round(d * FPS))
    h, zoom_in_default, pan_dir_default = kb_hash_choices(photo)
    if zoom_in is None:
        zoom_in = zoom_in_default
    dx, dy = pan_dir if pan_dir is not None else pan_dir_default
    rate_jit = ((h >> 5) % 1000) / 1000.0
    pan_jit = ((h >> 15) % 1000) / 1000.0
    delta = max(ZOOM_DELTA_MIN, min(ZOOM_DELTA_MAX,
                ZOOM_RATE_BASE * d * (0.75 + rate_jit * 0.5)))
    max_zoom = ZOOM_FLOOR + delta
    # PAN_SAFETY * jitter всегда < 1.0 (проверено численно) — офсет остаётся
    # строго внутри геометрического запаса (1-1/zoom)/2 на любом кадре клипа.
    pan_amt = PAN_SAFETY * (PAN_JITTER_MIN + pan_jit * (PAN_JITTER_MAX - PAN_JITTER_MIN))

    t = f"(on/{frames})"
    # smoothstep вместо линейного роста: старт/финиш без рывка — линейный зум
    # это самый узнаваемый штамп шаблонных слайд-шоу.
    eased = f"(3*pow({t},2)-2*pow({t},3))"
    z = (f"'{ZOOM_FLOOR}+{delta:.5f}*{eased}'" if zoom_in
         else f"'{max_zoom:.5f}-{delta:.5f}*{eased}'")
    # margin = (1-1/zoom)/2 считается от РЕАЛЬНОГО zoom в текущем кадре (переменная
    # zoompan даёт его сама) — офсет безопасен на любом кадре клипа по построению.
    x = (f"'iw/2-(iw/zoom/2){dx * pan_amt:+.5f}*(1-1/zoom)/2*iw'" if dx
         else "'iw/2-(iw/zoom/2)'")
    y = (f"'ih/2-(ih/zoom/2){dy * pan_amt:+.5f}*(1-1/zoom)/2*ih'" if dy
         else "'ih/2-(ih/zoom/2)'")
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", photo, "-vf",
           (f"scale=8000:4500:force_original_aspect_ratio=decrease,"
            f"pad=8000:4500:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"zoompan=z={z}:x={x}:y={y}:"
            f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
            f"{film_look(source, h)}"),
           "-t", str(d), "-c:v", "libx264", "-preset", "fast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def video_clip(vid, out, d, source="stock"):
    try:
        actual = dur(vid)
    except Exception:
        actual = d
    h = int(hashlib.md5(vid.encode()).hexdigest()[:8], 16)
    vf = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1"
    if actual < d - 0.05:
        vf += f",setpts={d/actual:.5f}*PTS"
    vf += f",{film_look(source, h)}"
    cmd = ["ffmpeg", "-y", "-i", vid, "-vf", vf, "-t", str(d), "-an",
           "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def jittered_body_durations(n, base_d, audio_path=None, start_offset=0.0):
    """Тело нарезано ровно на n одинаковых body_d — тоже штамп ("робот резал
    ровно"). Если есть numpy — множители берутся из РЕАЛЬНОЙ громкости
    аудио на месте каждого слота (громче -> короче), иначе fallback на
    псевдослучайный хэш-джиттер. Нормализовано так, что сумма не меняется."""
    if n <= 0:
        return []
    curve = audio_energy_curve(audio_path) if audio_path else None
    if curve:
        starts = [start_offset + i * base_d for i in range(n)]
        factors = energy_pace_multipliers(curve, starts, [base_d] * n)
    else:
        factors = []
        for i in range(n):
            h = int(hashlib.md5(f"body_slot:{i}".encode()).hexdigest()[:8], 16)
            factors.append(0.75 + (h % 1000) / 1000 * 0.5)   # 0.75-1.25
    avg = sum(factors) / len(factors)
    return [base_d * f / avg for f in factors]


def xfade_chain(clips, durs, is_hook, out, xfade_dur=XFADE_DUR):
    """Один проход filter_complex с цепочкой xfade между ВСЕМИ соседними
    кадрами — вместо жёсткой склейки. Тип перехода и длительность варьируются
    (разнообразие + иногда почти жёсткий cut), на границе хук/тело — заметный
    dissolve. Возвращает (True, итоговая_длительность) или (False, 0.0), если
    что-то пошло не так (тогда main() откатывается на обычный concat -c copy)."""
    n = len(clips)
    if n < 2:
        return False, 0.0
    parts, prev_label, cum = [], "0:v", durs[0]
    cut_hist, boundary_hist = [], []
    for i in range(1, n):
        h = int(hashlib.md5(f"{clips[i-1]}|{clips[i]}".encode()).hexdigest()[:8], 16)
        is_boundary = is_hook[i] != is_hook[i - 1]
        if is_boundary:
            # Граница хук/тело — заметный переход, не обычная склейка.
            candidate = "fadeblack" if (h % 2 == 0) else "dissolve"
            transition = pick_no_repeat(boundary_hist, candidate, ["dissolve", "fadeblack"], 1)
            this_dur = xfade_dur
        else:
            # Большинство склеек в реальном монтаже — жёсткий cut, не dissolve;
            # заметный переход — редкость, не норма. ~65% hard cut / ~35% вариация.
            candidate = "hardcut" if (h % 3 != 0) else XFADE_TRANSITIONS[(h >> 8) % len(XFADE_TRANSITIONS)]
            choice = pick_no_repeat(cut_hist, candidate, ["hardcut"] + XFADE_TRANSITIONS, max_repeat=3)
            this_dur = XFADE_DUR_HARD if choice == "hardcut" else xfade_dur
            transition = "fade" if choice == "hardcut" else choice
        offset = max(0.0, cum - this_dur)
        out_label = f"vx{i}" if i < n - 1 else "vout"
        parts.append(f"[{prev_label}][{i}:v]xfade=transition={transition}:"
                     f"duration={this_dur:.3f}:offset={offset:.3f}[{out_label}]")
        cum = cum + durs[i] - this_dur
        prev_label = out_label
    cmd = ["ffmpeg", "-y"]
    for c in clips:
        cmd += ["-i", c]
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  xfade-склейка не удалась, откат на concat:", r.stderr[-300:])
        return False, 0.0
    return True, max(cum, 0.1)


def pad_to_length(video, target, temp_dir):
    """Достраивает видео до нужной длины заморозкой последнего кадра — нужно
    после xfade_chain(), которая суммарно укорачивает ролик на (n-1)*XFADE_DUR
    относительно суммы длительностей кадров."""
    cur = dur(video)
    gap = target - cur
    if gap <= 0.05:
        return video
    lastframe = os.path.join(temp_dir, "_lastframe.jpg")
    padclip = os.path.join(temp_dir, "_pad.mp4")
    padded = os.path.join(temp_dir, "_padded.mp4")
    subprocess.run(["ffmpeg", "-y", "-sseof", "-0.3", "-i", video,
                    "-vframes", "1", lastframe], capture_output=True)
    r = subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", lastframe, "-t", f"{gap:.3f}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-r", str(FPS), padclip],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return video
    lst = os.path.join(temp_dir, "_pad_concat.txt")
    open(lst, "w", encoding="utf-8").write(
        f"file '{os.path.abspath(video)}'\nfile '{os.path.abspath(padclip)}'\n")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", lst, "-c", "copy", padded], capture_output=True, text=True)
    return padded if r.returncode == 0 else video


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
    # Кроссфейд между КАЖДОЙ парой слотов суммарно "съедает" (n-1)*XFADE_DUR —
    # закладываем это в тело заранее, хук-длительности заданы явно и их не трогаем.
    xfade_budget = max(0, n_slots - 1) * XFADE_DUR
    body_d_avg = (audio_dur + xfade_budget - hook_total) / max(body_slots, 1)
    body_durs = jittered_body_durations(body_slots, body_d_avg, audio_path=audio,
                                         start_offset=hook_total)

    # проверка хук-тайминга (ЧАСТЬ 14, ХУК-МЕДИА)
    if hook_slots and hook_total > 0:
        print(f"Хук: {hook_slots} слотов, сумма {hook_total:.1f}с | тело {body_slots} по ~{body_d_avg:.2f}с")
    print(f"Аудио {audio_dur:.1f}с ({audio_dur/60:.1f} мин), слотов {n_slots}")

    clips, clip_durs, clip_is_hook, missing = [], [], [], []
    zoom_hist, pan_hist = [], []
    for slot in range(1, n_slots + 1):
        out = os.path.join(temp, f"clip_{slot:04d}.mp4")
        is_hook_slot = slot <= hook_slots
        d = hook_dur.get(slot, body_d_avg) if is_hook_slot else body_durs[slot - hook_slots - 1]
        if os.path.exists(out):
            clips.append(out)
            clip_durs.append(d)
            clip_is_hook.append(is_hook_slot)
            continue
        kind, path, source = resolve_slot(media_dir, slot)
        if not path:
            missing.append(slot)
            continue
        if kind == "video":
            ok = video_clip(path, out, d, source)
        else:
            # anti-repetition: хэш сам по себе не мешает 3 зумам подряд случайно
            # совпасть — держим окно последних решений и форсируем смену при повторе.
            _, zi_cand, pd_cand = kb_hash_choices(path)
            zoom_in = pick_no_repeat(zoom_hist, zi_cand, [True, False], max_repeat=2)
            pan_dir = pick_no_repeat(pan_hist, pd_cand, PAN_DIRECTIONS, max_repeat=2)
            ok = kenburns_clip(path, out, d, source, zoom_in=zoom_in, pan_dir=pan_dir)
        if ok:
            clips.append(out)
            clip_durs.append(d)
            clip_is_hook.append(is_hook_slot)
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
    merged = os.path.join(temp, "merged.mp4")
    ok, xfade_total = xfade_chain(clips, clip_durs, clip_is_hook, merged)
    if not ok:
        concat = os.path.join(temp, "concat.txt")
        # Пути ТОЛЬКО абсолютные: concat-демуксер ffmpeg резолвит относительные
        # пути от папки самого concat.txt, а не от cwd — иначе сборка падает.
        open(concat, "w", encoding="utf-8").write(
            "".join(f"file '{os.path.abspath(c)}'\n" for c in clips))
        r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", concat, "-c", "copy", merged], capture_output=True, text=True)
        if r.returncode != 0:
            print("Склейка:", r.stderr[-400:])
            return
    merged = pad_to_length(merged, audio_dur, temp)
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
