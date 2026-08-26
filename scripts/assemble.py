#!/usr/bin/env python3
"""Сборка по слот-схеме с приоритетом AI > сток и хук-таймингом (ЧАСТЬ 14).
Конфиг: <video_dir>/media_plan/assemble_config.json
  { "n_slots": int, "hook_slots": int, "hook_durations": {"1":5.0,...} }
Если конфига нет — n_slots выводится из media/, хук не выделяется (равномерно).
Слот: приоритет AI-фото (_fastgen/_grok/_flow/_ai .jpg) > сток-видео (_stock_video.mp4)
> сток-фото (_stock.jpg). Фото -> Ken Burns, видео -> кроп под слот.
Usage: python scripts/assemble.py <video_dir>"""
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading

try:
    import numpy as np
except ImportError:
    np = None   # аудио-ритм по громкости — опциональная фича, без numpy просто выключена

# Пан по композиции и shot matching. Текста блоков у слот-сборщика нет
# (схема слотов, а не разбор по паузам), поэтому беаты сюда не применимы —
# но сама КАРТИНКА тут ровно та же, и решения по ней работают одинаково.
import director

FPS, WIDTH, HEIGHT = 25, 1920, 1080
# Каждый слот рендерится независимым процессом ffmpeg по уже полностью
# решённым параметрам (зум/пан/переход/медиа выбраны до рендера) — N
# параллельных процессов дают тот же результат, что и последовательный
# перебор, только быстрее. Та же формула, что в pipeline_smart.py.
def resolve_workers(env_value, cpu_count):
    if env_value:
        try:
            v = int(env_value)
            if v > 0:
                return v
        except ValueError:
            pass
    return max(1, (cpu_count or 4) - 1)


def resolve_ffmpeg_threads(workers, cpu_count):
    return max(1, (cpu_count or 4) // max(1, workers))


WORKERS = resolve_workers(os.environ.get("PIPELINE_WORKERS", ""), os.cpu_count())
FFMPEG_THREADS = resolve_ffmpeg_threads(WORKERS, os.cpu_count())

# CRF интермедиатов поднят с 23 (второе поколение потерь при склейке xfade
# поверх — см. pipeline_smart.py). Финальный проход может себе позволить
# -preset slow: он запускается один раз на весь ролик, а не N раз, как
# per-clip кодирование.
CLIP_CRF = "17"
FINAL_CRF = "18"
FINAL_PRESET = "slow"
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


def film_look(source, photo_hash, comp=None):
    # Один и тот же грейд на все слоты — тоже штамп, если приглядеться. Лёгкий
    # hash-джиттер контраста/сатурации/яркости на каждый клип убирает эту
    # одинаковость. Зерно — по источнику: сильнее на AI (маскирует "пластик"
    # генерации), почти незаметно на честном стоке (не раздувает битрейт зря).
    # comp — замер самого кадра: подтягивает тёмные и выбеленные планы друг к
    # другу (shot matching), иначе одна кривая eq ложится на всё подряд.
    c = 1.03 + (photo_hash % 100) / 100 * 0.05          # 1.03-1.08
    s = 1.04 + ((photo_hash >> 7) % 100) / 100 * 0.09    # 1.04-1.13
    b = ((photo_hash >> 14) % 100) / 100 * 0.02          # 0.00-0.02
    off = director.grade_offsets(comp)
    c = max(0.85, min(1.35, c + off["dc"]))
    s = max(0.60, min(1.25, s + off["ds"]))
    b = max(-0.06, min(0.09, b + off["db"]))
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
    # flac/wav — lossless-выход fix_pauses.py (см. pipeline_smart.py); mp3
    # остаётся резервом для роликов, сделанных до этого изменения.
    for name in ("audio_fixed.flac", "audio_fixed.wav", "audio_fixed.mp3", "audio.mp3"):
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


# .png тоже: Google Flow отдаёт скачанные картинки именно как png, и слот с
# NNN_flow.png раньше считался пустым, хотя AI-картинка для него уже лежала.
AI_SUFFIXES = tuple(f"_{src}.{ext}"
                    for src in ("fastgen", "grok", "flow", "ai")
                    for ext in ("jpg", "jpeg", "png"))


def resolve_slot(media_dir, n):
    for suf in AI_SUFFIXES:
        p = os.path.join(media_dir, f"{n:03d}{suf}")
        if os.path.exists(p):
            return "photo", p, "ai"
    v = os.path.join(media_dir, f"{n:03d}_stock_video.mp4")
    if os.path.exists(v):
        return "video", v, "stock"
    for ext in ("jpg", "jpeg", "png"):
        p = os.path.join(media_dir, f"{n:03d}_stock.{ext}")
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


def kenburns_clip(photo, out, d, source="stock", zoom_in=None, pan_dir=None, comp=None):
    frames = max(1, round(d * FPS))
    if comp is None:
        comp = director.frame_composition(photo)
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
    # increase+crop (не decrease+pad) — вписывание в рамку оставляло чёрные
    # поля по краям на фото, чей исходный кадр не ровно 16:9 (большинство
    # стока). Заливаем кадр целиком и обрезаем лишнее, как уже делает
    # video_clip() ниже для стокового видео.
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", photo, "-vf",
           (f"scale=8000:4500:force_original_aspect_ratio=increase,"
            f"crop=8000:4500,setsar=1,"
            f"zoompan=z={z}:x={x}:y={y}:"
            f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
            f"{film_look(source, h, comp)}"),
           "-t", str(d), "-c:v", "libx264", "-preset", "fast",
           "-crf", CLIP_CRF, "-threads", str(FFMPEG_THREADS),
           "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def video_clip(vid, out, d, source="stock", comp=None):
    try:
        actual = dur(vid)
    except Exception:
        actual = d
    h = int(hashlib.md5(vid.encode()).hexdigest()[:8], 16)
    vf = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1"
    if actual < d - 0.05:
        vf += f",setpts={d/actual:.5f}*PTS"
    vf += f",{film_look(source, h, comp)}"
    cmd = ["ffmpeg", "-y", "-i", vid, "-vf", vf, "-t", str(d), "-an",
           "-c:v", "libx264", "-preset", "fast", "-crf", CLIP_CRF,
           "-threads", str(FFMPEG_THREADS), "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def jittered_body_durations(n, base_d, audio_path=None, start_offset=0.0, start_step=None):
    """Тело нарезано ровно на n одинаковых body_d — тоже штамп ("робот резал
    ровно"). Если есть numpy — множители берутся из РЕАЛЬНОЙ громкости
    аудио на месте каждого слота (громче -> короче), иначе fallback на
    псевдослучайный хэш-джиттер. Нормализовано так, что сумма не меняется.
    start_step — шаг для стартовых точек САМПЛИРОВАНИЯ энергии; если base_d
    уже раздут под кроссфейд-бюджет, start_step должен быть РЕАЛЬНЫМ шагом
    по аудио, иначе поздние слоты сэмплят энергию не в том месте трека."""
    if n <= 0:
        return []
    curve = audio_energy_curve(audio_path) if audio_path else None
    if curve:
        step = start_step if start_step is not None else base_d
        starts = [start_offset + i * step for i in range(n)]
        factors = energy_pace_multipliers(curve, starts, [step] * n)
    else:
        factors = []
        for i in range(n):
            h = int(hashlib.md5(f"body_slot:{i}".encode()).hexdigest()[:8], 16)
            factors.append(0.75 + (h % 1000) / 1000 * 0.5)   # 0.75-1.25
    avg = sum(factors) / len(factors)
    return [base_d * f / avg for f in factors]


def plan_transitions(is_hook, xfade_dur=XFADE_DUR):
    """Заранее и детерминированно решить, КАКОЙ переход и КАКОЙ длины стоит на
    каждой склейке. Возвращает список (transition, duration) длиной n-1.

    Ключ хэша — номер стыка, а не имена файлов клипов: имена зависят от
    длительностей, длительности — от бюджета переходов, то есть от результата
    этой функции. Из-за этой завязки main() был вынужден закладывать XFADE_DUR
    на КАЖДУЮ склейку, хотя ~65% из них — hardcut на 0.06с; накопленная разница
    делала видео на десятки секунд длиннее аудио, и хвост срезался."""
    n = len(is_hook)
    plan = []
    cut_hist, boundary_hist = [], []
    for i in range(1, n):
        h = int(hashlib.md5(f"xfade:{i}|{is_hook[i-1]}|{is_hook[i]}".encode()).hexdigest()[:8], 16)
        if is_hook[i] != is_hook[i - 1]:
            # Граница хук/тело — заметный переход, не обычная склейка.
            candidate = "fadeblack" if (h % 2 == 0) else "dissolve"
            plan.append((pick_no_repeat(boundary_hist, candidate, ["dissolve", "fadeblack"], 1), xfade_dur))
        else:
            # Большинство склеек в реальном монтаже — жёсткий cut, не dissolve;
            # заметный переход — редкость, не норма. ~65% hard cut / ~35% вариация.
            candidate = "hardcut" if (h % 3 != 0) else XFADE_TRANSITIONS[(h >> 8) % len(XFADE_TRANSITIONS)]
            choice = pick_no_repeat(cut_hist, candidate, ["hardcut"] + XFADE_TRANSITIONS, max_repeat=3)
            plan.append(("fade" if choice == "hardcut" else choice,
                         XFADE_DUR_HARD if choice == "hardcut" else xfade_dur))
    return plan


def transitions_budget(is_hook, xfade_dur=XFADE_DUR):
    """Сколько секунд суммарно съедят склейки — точно, а не по верхней оценке."""
    return sum(d for _, d in plan_transitions(is_hook, xfade_dur))


def xfade_chain(clips, durs, is_hook, out, xfade_dur=XFADE_DUR, plan=None):
    """Один проход filter_complex с цепочкой xfade между ВСЕМИ соседними
    кадрами — вместо жёсткой склейки. Тип перехода и длительность варьируются
    (разнообразие + иногда почти жёсткий cut), на границе хук/тело — заметный
    dissolve. Возвращает (True, итоговая_длительность) или (False, 0.0), если
    что-то пошло не так (тогда main() откатывается на обычный concat -c copy)."""
    n = len(clips)
    if n < 2:
        return False, 0.0
    if plan is None:
        plan = plan_transitions(is_hook, xfade_dur)
    parts, prev_label, cum = [], "0:v", durs[0]
    for i in range(1, n):
        transition, this_dur = plan[i - 1]
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
            "-c:v", "libx264", "-preset", FINAL_PRESET, "-crf", FINAL_CRF,
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
                        "-c:v", "libx264", "-preset", "fast", "-crf", CLIP_CRF,
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
        return 1
    audio_dur = dur(audio)
    n_slots, hook_slots, hook_dur = load_config(video_dir, media_dir)
    if n_slots <= 0:
        print("Слотов не найдено (пустой media/ и нет конфига)")
        return 1
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
    # Кроссфейд между КАЖДОЙ парой слотов "съедает" время — закладываем это в
    # тело заранее (хук-длительности заданы явно и их не трогаем). Бюджет
    # считается ТОЧНО по заранее построенному плану переходов, а не по верхней
    # оценке (n-1)*XFADE_DUR: большинство склеек — hardcut на 0.06с, и завышенный
    # бюджет растягивал слоты так, что видео выходило на десятки секунд длиннее
    # аудио — хвост срезался, а картинка ползла от слов всё дальше к концу.
    slot_is_hook = [sl <= hook_slots for sl in range(1, n_slots + 1)]
    slot_plan = plan_transitions(slot_is_hook)
    xfade_budget = sum(d for _, d in slot_plan)
    body_d_avg = (audio_dur + xfade_budget - hook_total) / max(body_slots, 1)
    # body_d_avg раздут под кроссфейд-бюджет — для сэмплинга энергии по РЕАЛЬНОМУ
    # аудио стартовые точки должны идти с шагом от настоящей (не раздутой) длины,
    # иначе поздние слоты на ролике со множеством склеек съезжают по треку.
    body_d_real = (audio_dur - hook_total) / max(body_slots, 1)
    body_durs = jittered_body_durations(body_slots, body_d_avg, audio_path=audio,
                                         start_offset=hook_total, start_step=body_d_real)

    # проверка хук-тайминга (ЧАСТЬ 14, ХУК-МЕДИА)
    if hook_slots and hook_total > 0:
        print(f"Хук: {hook_slots} слотов, сумма {hook_total:.1f}с | тело {body_slots} по ~{body_d_avg:.2f}с")
    print(f"Аудио {audio_dur:.1f}с ({audio_dur/60:.1f} мин), слотов {n_slots}")

    # --- Фаза A: решения по порядку (медиа, зум/пан, кэш-хиты) — последовательно
    # (анти-повтор по истории должен видеть каждое предыдущее решение).
    missing = []
    zoom_hist, pan_hist = [], []
    jobs = [None] * n_slots
    for idx in range(n_slots):
        slot = idx + 1
        is_hook_slot = slot <= hook_slots
        d = hook_dur.get(slot, body_d_avg) if is_hook_slot else body_durs[slot - hook_slots - 1]
        kind, path, source = resolve_slot(media_dir, slot)
        if not path:
            missing.append(slot)
            continue
        # Хэш параметров рендера в имени клипа. Без него правка
        # assemble_config.json или новая озвучка другой длины молча переиспользовали
        # старые клипы со СТАРЫМИ длительностями — весь ролик уезжал по таймингу,
        # и никакого признака этого в логе не было. В pipeline_smart.py это уже
        # починено, сюда фикс не был перенесён.
        params_hash = hashlib.md5(
            f"{d:.3f}|{source}|{os.path.basename(path)}".encode()).hexdigest()[:8]
        out = os.path.join(temp, f"clip_{slot:04d}_{params_hash}.mp4")
        if os.path.exists(out):
            jobs[idx] = {"cached": True, "out": out, "d": d, "is_hook": is_hook_slot}
            continue
        if kind == "video":
            jobs[idx] = {"cached": False, "kind": "video", "path": path, "d": d,
                        "source": source, "is_hook": is_hook_slot, "out": out}
        else:
            # anti-repetition: хэш сам по себе не мешает 3 зумам подряд случайно
            # совпасть — держим окно последних решений и форсируем смену при повторе.
            _, zi_cand, pd_cand = kb_hash_choices(path)
            zoom_in = pick_no_repeat(zoom_hist, zi_cand, [True, False], max_repeat=2)
            # Пан ведём НА объект: если визуальная масса заметно смещена,
            # направление диктует картинка, а не хэш имени файла (иначе пан с
            # вероятностью 1/2 выдавливает объект за рамку). Отцентрованный
            # кадр — прежний путь через анти-повтор.
            comp = director.frame_composition(path)
            pan_dir = director.pan_for_composition(comp, None)
            if pan_dir is None:
                pan_dir = pick_no_repeat(pan_hist, pd_cand, PAN_DIRECTIONS, max_repeat=2)
            jobs[idx] = {"cached": False, "kind": "photo", "path": path, "d": d,
                        "source": source, "is_hook": is_hook_slot, "out": out,
                        "zoom_in": zoom_in, "pan_dir": pan_dir, "comp": comp}

    # --- Фаза B: параллельный рендер того, чего нет в кэше — каждый job уже
    # несёт все параметры, N процессов ffmpeg дают тот же результат быстрее.
    def render_job(job):
        if job["kind"] == "video":
            return video_clip(job["path"], job["out"], job["d"], job["source"])
        return kenburns_clip(job["path"], job["out"], job["d"], job["source"],
                             zoom_in=job["zoom_in"], pan_dir=job["pan_dir"],
                             comp=job.get("comp"))

    to_render = [(idx, job) for idx, job in enumerate(jobs) if job is not None and not job["cached"]]
    results = {}
    if to_render:
        print(f"Рендер {len(to_render)} слотов, {WORKERS} параллельно "
              f"({FFMPEG_THREADS} потока ffmpeg на слот)...")
        print_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(render_job, job): idx for idx, job in to_render}
            done_n = 0
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    ok = fut.result()
                except Exception as e:
                    print(f"  [{idx+1}] рендер упал: {type(e).__name__} {e}")
                    ok = False
                results[idx] = ok
                done_n += 1
                if not ok or done_n % 20 == 0 or done_n <= 3:
                    with print_lock:
                        print(f"[{done_n}/{len(to_render)}] слот {idx+1:03d}/{n_slots} "
                              f"{'OK' if ok else 'FFMPEG FAIL'}", flush=True)

    clips, clip_durs, clip_is_hook = [], [], []
    for idx, job in enumerate(jobs):
        if job is None:
            continue
        slot = idx + 1
        ok = True if job["cached"] else results.get(idx, False)
        if ok:
            clips.append(job["out"]); clip_durs.append(job["d"]); clip_is_hook.append(job["is_hook"])
        else:
            missing.append(slot)

    print(f"\nКлипов: {len(clips)}/{n_slots}, пропущено: {len(missing)}")
    if missing:
        print("Пропущены:", missing)
    if not clips:
        return 1
    merged = os.path.join(temp, "merged.mp4")
    # План должен совпадать с тем, по которому считался бюджет; если слот выпал —
    # честно пересчитываем по фактической цепочке.
    plan = slot_plan if len(clips) == n_slots else plan_transitions(clip_is_hook)
    ok, xfade_total = xfade_chain(clips, clip_durs, clip_is_hook, merged, plan=plan)
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
            return 1
    merged = pad_to_length(merged, audio_dur, temp)
    # -shortest САМ ПО СЕБЕ недостаточен с -c:v copy: копирование пакетов режет
    # только по границам GOP исходного клипа, а не по факту конца аудио — на
    # реальном длинном ролике это давало лишние секунды видео без звука сверху.
    # В pipeline_smart.py это уже починено (явный -t), сюда фикс не был перенесён.
    r = subprocess.run(["ffmpeg", "-y", "-i", merged, "-i", audio,
                        "-t", f"{audio_dur:.3f}",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart",
                        "-shortest", out_file], capture_output=True, text=True)
    if r.returncode != 0:
        print("Аудио:", r.stderr[-400:])
        return 1
    mb = os.path.getsize(out_file) / (1024 * 1024)
    # Раньше ГОТОВО печаталось одинаково что при полном покрытии слотов, что
    # при пропусках — успех был неотличим от частичной сборки, а код возврата
    # оставался 0 в обоих случаях. Теперь пропуски видны в итоговой строке и
    # код возврата честно ненулевой, если не все слоты доехали.
    status = f" | ПРОПУЩЕНО {len(missing)}/{n_slots} слотов: {missing}" if missing else ""
    print(f"\nГОТОВО: {out_file} ({mb:.0f} MB){status}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
