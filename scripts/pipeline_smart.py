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

try:
    import numpy as np
except ImportError:
    np = None   # аудио-ритм по громкости — опциональная фича, без numpy просто выключена

from PIL import Image as PILImage   # Pillow уже обязательная зависимость (requirements.txt)

try:
    import cv2
    PARALLAX_LIBS = True
except ImportError:
    PARALLAX_LIBS = False   # 2.5D-параллакс — опциональная фича (torch/transformers/opencv тяжёлые)

PARALLAX_ENABLED = os.environ.get("PARALLAX", "1") != "0" and PARALLAX_LIBS
PARALLAX_BROKEN = False   # взводится только на системном сбое (модель/сеть), не на одной плохой картинке

VIDEO_FOLDER = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
SCRIPT_FILE = os.path.join(VIDEO_FOLDER, "script.txt")
MEDIA_FOLDER = os.path.join(VIDEO_FOLDER, "media")
OUTPUT_FILE = os.path.join(VIDEO_FOLDER, "final.mp4")
TEMP_FOLDER = os.path.join(VIDEO_FOLDER, "temp_smart")

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
FPS, WIDTH, HEIGHT = 25, 1920, 1080

# ZOOM_FLOOR — минимальный зум держится ВЕСЬ клип (не 1.0). Раньше offset пана
# был обязан = 0 ровно в момент zoom=1.0 (иначе край вылезет за картинку), и на
# каждом втором клипе (zoom-out) кадр половину времени стоял мёртвым по центру.
# С постоянным полом запас под пан есть на любом кадре клипа, а не только к
# концу движения зума.
ZOOM_FLOOR = 1.04
# Скорость зума в долях/сек, а не фиксированным приростом на клип — иначе
# 4-секундный и 20-секундный кадр "дышат" с заметно разной скоростью.
ZOOM_RATE_BASE = 0.010
ZOOM_DELTA_MIN, ZOOM_DELTA_MAX = 0.05, 0.22
MIN_CLIP, MAX_CLIP = 4.0, 20.0
# Панорамирование считается от РЕАЛЬНОГО zoom в каждый момент кадра — (1-1/zoom)/2
# это точный геометрический запас смещения без вылета за картинку, безопасно по
# построению на любом кадре, поэтому PAN_SAFETY можно брать ближе к пределу, чем
# раньше (когда запас считался один раз заранее от конечного zoom клипа).
PAN_SAFETY = 0.9
PAN_JITTER_MIN, PAN_JITTER_MAX = 0.6, 1.0   # органический разброс силы пана по клипам
PAN_DIRECTIONS = [(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1)]
# Один и тот же грейд на все 200+ кадров — тоже штамп, если приглядеться.
# Лёгкий hash-джиттер контраста/сатурации/яркости на каждый клип убирает
# эту одинаковость, оставаясь в узком диапазоне (не превращается в разнобой).
# Базовый грейд — не нейтральный "как есть у стока": сатурация чуть прибрана,
# тени уводим в холод, света чуть тёплые (лёгкий teal-orange) — читается как
# военно-историческая документалка, а не разноцветный слайд-шоу из Pexels.
# MOOD_GRADE — тот же язык грейда, но с сдвигом под секцию: хук резче и
# холоднее (цепляет), тело — базовая "стальная" документалка, финал теплее
# и мягче (спад напряжения). LUT-переключение по секции без внешних .cube
# файлов — тот же eq/colorbalance, просто другие опорные точки.
MOOD_GRADE = {
    "HOOK":  {"c0": 1.10, "s0": 0.82, "vign": "PI/4.2", "bs": 0.13, "rh": 0.02},
    "FINAL": {"c0": 1.02, "s0": 0.90, "vign": "PI/6",   "bs": 0.06, "rh": 0.08},
    "BODY":  {"c0": 1.06, "s0": 0.86, "vign": "PI/5",   "bs": 0.10, "rh": 0.03},
}


def film_look(photo_hash, section=""):
    mood = MOOD_GRADE["HOOK"] if section.startswith("HOOK") else (
           MOOD_GRADE["FINAL"] if section.startswith("FINAL") else MOOD_GRADE["BODY"])
    c = mood["c0"] + (photo_hash % 100) / 100 * 0.05
    s = mood["s0"] + ((photo_hash >> 7) % 100) / 100 * 0.08
    b = ((photo_hash >> 14) % 100) / 100 * 0.02
    return (f"eq=contrast={c:.3f}:saturation={s:.3f}:brightness={b:.3f},"
            f"colorbalance=rs=-0.06:bs={mood['bs']:.3f}:rm=-0.02:bm=0.04:rh={mood['rh']:.3f}:bh=-0.02,"
            f"vignette={mood['vign']},noise=alls=2:allf=t+u")


XFADE_DUR = 0.4        # диссолв на границах секций и часть обычных склеек
XFADE_DUR_HARD = 0.06  # почти мгновенный переход — читается как жёсткий cut
# hblur (смаз в движении — читается как whip pan) и zoomin (панч-переход)
# добавлены в пул обычных склеек; fadewhite — вспышка светом на границах
# разделов вперемешку с dissolve/fadeblack, без кастомных текстур-ассетов.
XFADE_TRANSITIONS = ["fade", "dissolve", "smoothleft", "smoothright",
                      "smoothup", "smoothdown", "hblur", "zoomin"]
BOUNDARY_TRANSITIONS = ["dissolve", "fadeblack", "fadewhite"]
HOOK_MAX_CLIP = 5.0     # в хуке кадры короче и чаще — критично для удержания первых секунд
PAUSE_DURATIONS = {"[pause]": 0.8, "[short pause]": 0.4,
                   "[slowly]": 0.0, "[emphasis]": 0.0, "[energetic]": 0.0}

# Russo One — фирменный "рубленый" дисплейный шрифт (CHANNEL.md house
# style), не системный DejaVu. OFL, бесплатно (Google Fonts / google/fonts
# на GitHub). Anton (первый выбор) не подошёл — в нём НЕТ кириллицы вообще
# (проверено вживую: буквы рисуются квадратами-тофу). Russo One — с родной
# кириллицей от российской студии, того же плакатного характера.
# Держим DejaVu запасным на случай отсутствия assets/fonts/.
FONT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts", "RussoOne-Regular.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
FONT_IS_DISPLAY = bool(FONT_PATH) and "RussoOne" in FONT_PATH   # дисплейный шрифт уже капслочный и очень широкий по кеглю


def section_title(name):
    """BLOCK N: Название -> 'Название'. HOOK/FINAL/безымянные BLOCK — без титра."""
    m = re.match(r'BLOCK\s+\d+\s*:\s*(.+)', name, re.I)
    return m.group(1).strip() if m else None


def escape_drawtext(s):
    # % \u043d\u0435 \u044d\u043a\u0440\u0430\u043d\u0438\u0440\u043e\u0432\u0430\u043b\u0441\u044f \u2014 ffmpeg drawtext \u0442\u0440\u0430\u043a\u0442\u0443\u0435\u0442 \u0435\u0433\u043e \u043a\u0430\u043a \u043d\u0430\u0447\u0430\u043b\u043e strftime/
    # expansion-\u0442\u043e\u043a\u0435\u043d\u0430 \u0438 \u0440\u043e\u043d\u044f\u0435\u0442 "Stray %" \u043f\u0440\u0435\u0434\u0443\u043f\u0440\u0435\u0436\u0434\u0435\u043d\u0438\u0435 (\u043f\u0440\u043e\u0432\u0435\u0440\u0435\u043d\u043e \u0432\u0436\u0438\u0432\u0443\u044e).
    return (s.replace("\\", "\\\\").replace(":", "\\:")
             .replace("%", "%%").replace("'", "\u2019"))


def pick_no_repeat(history, candidate, options, max_repeat):
    """\u0425\u044d\u0448 \u0434\u0430\u0451\u0442 \u0440\u0430\u0437\u043d\u043e\u043e\u0431\u0440\u0430\u0437\u0438\u0435 "\u0432 \u0441\u0440\u0435\u0434\u043d\u0435\u043c", \u043d\u043e \u043d\u0435 \u043c\u0435\u0448\u0430\u0435\u0442 3-4 \u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u044b\u043c \u043f\u043e\u0434\u0440\u044f\u0434
    \u0441\u043b\u0443\u0447\u0430\u0439\u043d\u044b\u043c \u0441\u043e\u0432\u043f\u0430\u0434\u0435\u043d\u0438\u0435\u043c. \u0415\u0441\u043b\u0438 \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 max_repeat \u0440\u0435\u0448\u0435\u043d\u0438\u0439 \u0441\u043e\u0432\u043f\u0430\u0434\u0430\u044e\u0442 \u0441
    \u043a\u0430\u043d\u0434\u0438\u0434\u0430\u0442\u043e\u043c \u2014 \u0434\u0435\u0442\u0435\u0440\u043c\u0438\u043d\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u043e \u0431\u0435\u0440\u0451\u043c \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439 \u0432\u0430\u0440\u0438\u0430\u043d\u0442 \u043f\u043e \u043a\u0440\u0443\u0433\u0443 \u0432\u043c\u0435\u0441\u0442\u043e
    \u043d\u0435\u0433\u043e. history \u043c\u0443\u0442\u0438\u0440\u0443\u0435\u0442\u0441\u044f \u043d\u0430 \u043c\u0435\u0441\u0442\u0435 (\u0441\u043f\u0438\u0441\u043e\u043a \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0445 \u0440\u0435\u0448\u0435\u043d\u0438\u0439)."""
    if len(history) >= max_repeat and all(x == candidate for x in history[-max_repeat:]):
        idx = options.index(candidate) if candidate in options else 0
        candidate = options[(idx + 1) % len(options)]
    history.append(candidate)
    del history[:-(max_repeat + 2)]
    return candidate


def audio_energy_curve(audio_path, window_sec=1.0):
    """RMS-\u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u044c \u0430\u0443\u0434\u0438\u043e \u043f\u043e \u043e\u043a\u043d\u0430\u043c \u2014 \u0431\u0435\u0437 Whisper/librosa, \u0447\u0438\u0441\u0442\u044b\u0439 PCM+numpy.
    \u0412\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0435\u0442 (rms_array, window_sec) \u0438\u043b\u0438 None, \u0435\u0441\u043b\u0438 numpy \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d/\u0447\u0442\u043e-\u0442\u043e
    \u043f\u043e\u0448\u043b\u043e \u043d\u0435 \u0442\u0430\u043a (\u0444\u0438\u0447\u0430 \u043e\u043f\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u0430\u044f, \u043f\u0430\u0439\u043f\u043b\u0430\u0439\u043d \u043d\u0435 \u0434\u043e\u043b\u0436\u0435\u043d \u043f\u0430\u0434\u0430\u0442\u044c \u0431\u0435\u0437 \u043d\u0435\u0451)."""
    if np is None:
        return None
    sr = 8000   # \u043d\u0443\u0436\u043d\u0430 \u0442\u043e\u043b\u044c\u043a\u043e \u043e\u0433\u0438\u0431\u0430\u044e\u0449\u0430\u044f \u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u0438, \u043d\u0435 \u0437\u0432\u0443\u043a \u2014 \u043d\u0438\u0437\u043a\u0438\u0439 sr \u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0438 \u0431\u044b\u0441\u0442\u0440\u043e
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
        print(f"  \u0410\u043d\u0430\u043b\u0438\u0437 \u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u0438 \u043d\u0435 \u0443\u0434\u0430\u043b\u0441\u044f (\u043f\u0440\u043e\u043f\u0443\u0441\u043a\u0430\u044e): {e}")
        return None


def energy_pace_multipliers(curve, starts, durs, lo=0.8, hi=1.25):
    """\u0413\u0440\u043e\u043c\u0447\u0435 \u0443\u0447\u0430\u0441\u0442\u043e\u043a \u2014 \u043a\u043e\u0440\u043e\u0447\u0435 \u043a\u0430\u0434\u0440\u044b (\u043c\u043d\u043e\u0436\u0438\u0442\u0435\u043b\u044c <1), \u0442\u0438\u0448\u0435 \u2014 \u0434\u043b\u0438\u043d\u043d\u0435\u0435 (>1).
    \u041c\u043d\u043e\u0436\u0438\u0442\u0435\u043b\u044c \u0441\u0447\u0438\u0442\u0430\u0435\u0442\u0441\u044f \u043e\u0442 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u044f \u043a \u041c\u0415\u0414\u0418\u0410\u041d\u0415 \u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u0438 \u0432\u0441\u0435\u0439 \u0434\u043e\u0440\u043e\u0436\u043a\u0438,
    \u0447\u0442\u043e\u0431\u044b \u0442\u0438\u0445\u0438\u0439 \u0440\u043e\u043b\u0438\u043a \u0446\u0435\u043b\u0438\u043a\u043e\u043c \u043d\u0435 \u0440\u0430\u0441\u0442\u044f\u0433\u0438\u0432\u0430\u043b \u0432\u0441\u0435 \u043a\u0430\u0434\u0440\u044b \u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u043e."""
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


def find_audio():
    for name in ("audio_fixed.mp3", "audio.mp3"):
        p = os.path.join(VIDEO_FOLDER, name)
        if os.path.exists(p):
            return p
    mp3s = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(".mp3")]
    return os.path.join(VIDEO_FOLDER, mp3s[0]) if mp3s else os.path.join(VIDEO_FOLDER, "audio.mp3")


AUDIO_FILE = find_audio()


def load_json_dict(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  Битый {path}, пропускаю: {e}")
    return {}


def load_themes():
    """Два уровня: канальный словарь (channel_themes.json в корне репо —
    общие для ниши слова вроде "меч"/"доспех"/"музей", не меняются от
    ролика к ролику) + эпизодный (media_plan/themes.json конкретного
    видео — только то, что специфично именно этой теме). Раньше весь
    словарь собирался заново руками под каждый новый сценарий; теперь
    новому эпизоду нужно всего несколько строк добавки поверх базы."""
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "channel_themes.json")
    episode_path = os.path.join(VIDEO_FOLDER, "media_plan", "themes.json")
    merged = load_json_dict(base_path)
    merged.update(load_json_dict(episode_path))
    return merged


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
            kept.append((name, body))
    if kept:
        # Маркер секции — ещё один спецтокен в общем потоке (не просто "\n"-склейка
        # как раньше), чтобы граница === не рвала перенос паузы через границу блока,
        # а секция при этом была известна для каждого получившегося блока.
        # БЕЗ .strip("\x00") — иначе снимается ведущий \x00 у маркера самой первой
        # секции (обычно HOOK), split() перестаёт его узнавать, и "SECTION:HOOK"
        # утекает в текст как отдельный псевдо-блок из одного слова — под первый
        # кадр хука уходит мусорный клип с generic-запросом вместо реального текста.
        content = "".join(f"\x00SECTION:{name}\x00{body}" for name, body in kept)
    else:
        content = re.sub(r'===.*?===', '', raw).strip()   # сценарий без === — берём как есть
    # [stat:TEXT] — цифра-плашка на экран (ЧАСТЬ "Монтаж под удержание": цифра
    # без плашки не запоминается). Вытаскиваем ДО общего вырезания [...],
    # иначе текст плашки пропадает вместе со всеми остальными тегами.
    content = re.sub(r'\[stat:(.*?)\]', lambda m: f"\x01STAT:{m.group(1)}\x01", content)
    processed = content
    for tag in sorted(PAUSE_DURATIONS, key=len, reverse=True):
        processed = processed.replace(tag, f"__PAUSE_{PAUSE_DURATIONS[tag]}__")
    processed = re.sub(r'\[.*?\]', '', processed)
    parts = re.split(r'(__PAUSE_[\d.]+__|\x00SECTION:.*?\x00|\x01STAT:.*?\x01)', processed)
    blocks, cur, pause, stat = [], "", 0.0, None
    section = "BODY"

    def flush():
        nonlocal cur, pause, stat
        if cur:
            blocks.append({"text": cur, "pause_after": pause,
                           "words": len(cur.split()), "section": section, "stat": stat})
        cur, pause, stat = "", 0.0, None

    for part in parts:
        mp = re.match(r'__PAUSE_([\d.]+)__', part)
        ms = re.match(r'\x00SECTION:(.*?)\x00', part)
        mst = re.match(r'\x01STAT:(.*?)\x01', part)
        if mp:
            pause += float(mp.group(1))
        elif ms:
            flush()             # смена секции — всегда граница блока
            section = ms.group(1)
        elif mst:
            # Плашка НЕ режет монтаж — [stat:...] может стоять посреди фразы
            # без соседнего [pause], и это не повод обрывать клип на ровном
            # месте. Но если пауза уже открыта (граница блока ждёт следующий
            # текст), плашка относится к ЕЩЁ НЕ начатому следующему блоку —
            # без явного flush() она утекала бы в текущий cur (баг: плашка
            # после [pause] подписывалась под предыдущий, а не следующий кадр).
            if pause > 0 and cur:
                flush()
            stat = mst.group(1)
        else:
            t = part.strip()
            if t:
                if pause > 0 and cur:
                    flush()      # реальная пауза — вот это настоящая граница блока
                    cur = t
                else:
                    cur = f"{cur} {t}".strip()
    flush()
    print(f"Блоков: {len(blocks)}")
    return blocks


def block_durations(blocks, total, energy_mults=None):
    tw = sum(b["words"] for b in blocks)
    tp = sum(b["pause_after"] for b in blocks)
    wps = tw / max(total - tp, 1)
    raw = [b["words"] / wps + b["pause_after"] for b in blocks]
    if energy_mults:
        raw = [r * m for r, m in zip(raw, energy_mults)]
    d = []
    for b, r in zip(blocks, raw):
        # В хуке кадры короче и чаще — первые секунды решают, останется ли зритель.
        cap = HOOK_MAX_CLIP if b["section"].startswith("HOOK") else MAX_CLIP
        d.append(max(MIN_CLIP, min(cap, r)))
    scale = total / sum(d)
    return [x * scale for x in d]


PEXELS_BROKEN = False       # взводится только на реальном отказе API, не на пустой выдаче


def pexels_photo(query, index, used_ids=None):
    """used_ids — множество ID уже показанных в этом ролике фото (мутируется на
    месте). Разные блоки часто ловят один и тот же тематический запрос — без
    этого им всем доставался бы top-1 результат, то есть одна и та же картинка
    по нескольку раз за ролик. Перебираем выдачу (per_page=40) и берём первый
    ID, которого ещё не было; если все уже использованы — берём топ-1 всё равно
    (лучше повтор, чем сорванная сборка)."""
    global PEXELS_BROKEN
    cache = os.path.join(TEMP_FOLDER, "pexels_cache")
    os.makedirs(cache, exist_ok=True)
    # Хэш запроса в имени файла — иначе смена themes.json без чистки temp_smart/
    # молча оставляет картинку под старый запрос (кэш бил только по номеру блока).
    qhash = hashlib.md5(query.encode()).hexdigest()[:8]
    cf = os.path.join(cache, f"{index:04d}_{qhash}.jpg")
    if os.path.exists(cf):
        return cf
    if not PEXELS_API_KEY:
        return None
    try:
        q = urllib.parse.quote(query)
        req = urllib.request.Request(
            f"https://api.pexels.com/v1/search?query={q}&per_page=40&orientation=landscape",
            headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        photos = data.get("photos") or []
        if not photos:
            return None
        pick = photos[0]
        if used_ids is not None:
            for p in photos:
                if p.get("id") not in used_ids:
                    pick = p
                    break
        if used_ids is not None:
            used_ids.add(pick.get("id"))
        url = pick["src"].get("large2x") or pick["src"].get("large")
        img = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(img, timeout=20) as r:
            open(cf, "wb").write(r.read())
        return cf
    except Exception as e:
        PEXELS_BROKEN = True
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


GENERIC_FALLBACKS = [
    "medieval sword still life", "knight armor moody light",
    "medieval castle atmosphere", "old manuscript parchment history",
    "cinematic dark fantasy weapon",
]


def query_for(text):
    tl = text.lower()
    for kw, q in THEMES.items():
        if kw in tl:
            return q
    return None


def resolve_queries(blocks):
    """Прямой поиск по themes.json ловит не все блоки — короткие связки
    ("Не дрались. Несли.", "Береги себя.") и абстрактные куски без предметных
    слов улетают в generic-заглушку, которая никак не привязана к теме ролика.
    Вместо одной фиксированной строки на все такие блоки: сперва пробуем
    унаследовать запрос соседнего блока той же секции (тема раздела обычно
    не меняется от предложения к предложению), и только если во всей секции
    вообще ничего не нашлось — берём по кругу из GENERIC_FALLBACKS (не один
    и тот же текст на всё, иначе Pexels отдаёт одну и ту же жалкую пятёрку)."""
    raw = [query_for(b["text"]) for b in blocks]
    resolved = list(raw)
    for i, q in enumerate(resolved):
        if q is not None:
            continue
        for j in range(i - 1, -1, -1):
            if blocks[j]["section"] != blocks[i]["section"]:
                break
            if raw[j] is not None:
                resolved[i] = raw[j]
                break
        if resolved[i] is not None:
            continue
        for j in range(i + 1, len(blocks)):
            if blocks[j]["section"] != blocks[i]["section"]:
                break
            if raw[j] is not None:
                resolved[i] = raw[j]
                break
    fallback_n = 0
    for i, q in enumerate(resolved):
        if q is None:
            resolved[i] = GENERIC_FALLBACKS[fallback_n % len(GENERIC_FALLBACKS)]
            fallback_n += 1
    return resolved


def add_overlays(vf_base, dur, title=None, stat=None):
    """Титр секции (низ кадра, слайд+fade первые ~2.5с, фирменный дисплейный
    шрифт) + цифровая плашка (верх кадра, красный баннер — акцентный цвет
    CHANNEL.md house style, держится почти весь клип: цифру без плашки не
    запоминают, см. производственные пометки сценария). Общий код для фото
    и видео-стока. Обычный fade раньше был самым узнаваемым штампом
    автослайдшоу — слайд добавляет "кто-то это анимировал руками"."""
    vf = vf_base
    if title and FONT_PATH:
        hold = min(2.5, dur * 0.6)
        fin = max(0.3, hold * 0.3)
        text = title.upper() if FONT_IS_DISPLAY else title
        safe = escape_drawtext(text)
        fs = 46 if FONT_IS_DISPLAY else 54
        vf += (f",drawtext=fontfile='{FONT_PATH}':text='{safe}':"
               f"fontcolor=white:fontsize={fs}:borderw=3:bordercolor=black@0.8:"
               f"x=(w-text_w)/2:"
               f"y='h-180+(1-min(t/{fin:.2f}\\,1))*36':"
               f"alpha='if(lt(t\\,{fin:.2f})\\,t/{fin:.2f}\\,"
               f"if(lt(t\\,{hold:.2f})\\,1\\,max(0\\,1-(t-{hold:.2f})/0.4)))'")
    if stat and FONT_PATH:
        fin = 0.25
        hold = max(fin, dur - 0.5)
        text = stat.upper() if FONT_IS_DISPLAY else stat
        safe = escape_drawtext(text)
        fs = 58 if FONT_IS_DISPLAY else 64
        vf += (f",drawtext=fontfile='{FONT_PATH}':text='{safe}':"
               f"fontcolor=white:fontsize={fs}:"
               f"box=1:boxcolor=0xC8102E@0.92:boxborderw=18:"
               f"x=(w-text_w)/2:"
               f"y='110-(1-min(t/{fin:.2f}\\,1))*26':"
               f"alpha='if(lt(t\\,{fin:.2f})\\,t/{fin:.2f}\\,"
               f"if(lt(t\\,{hold:.2f})\\,1\\,max(0\\,1-(t-{hold:.2f})/0.4)))'")
    return vf


def kb_hash_choices(photo):
    """Кандидаты зума/пана по хэшу файла — детерминированно, но БЕЗ памяти о
    соседних клипах (это даёт anti-repetition в main(), см. pick_no_repeat)."""
    h = int(hashlib.md5(photo.encode()).hexdigest()[:8], 16)
    zoom_in = bool(h & 1)
    pan_dir = PAN_DIRECTIONS[(h >> 1) % len(PAN_DIRECTIONS)]
    return h, zoom_in, pan_dir


def estimate_busyness(photo_path):
    """Грубая мера "тесноты" кадра без depth-модели — просто плотность
    перепадов яркости на уменьшенной копии. Не настоящая детекция крупности
    плана, но дешёвый прокси: у тесного/плотного кадра (уже крупный план)
    перепадов много на каждый пиксель, у просторного — мало. Используется
    только чтобы НЕ зумить ещё сильнее то, что и так уже крупный план."""
    try:
        img = PILImage.open(photo_path).convert("L").resize((160, 90))
        arr = np.asarray(img, dtype=np.float32) if np is not None else None
        if arr is None:
            return 0.5
        gx = np.abs(np.diff(arr, axis=1)).mean()
        gy = np.abs(np.diff(arr, axis=0)).mean()
        return float(min(1.0, (gx + gy) / 40.0))
    except Exception:
        return 0.5


def kenburns(photo, out, dur, title=None, zoom_in=None, pan_dir=None, stat=None, section=""):
    frames = max(1, round(dur * FPS))
    h, zoom_in_default, pan_dir_default = kb_hash_choices(photo)
    if zoom_in is None:
        zoom_in = zoom_in_default
    dx, dy = pan_dir if pan_dir is not None else pan_dir_default
    rate_jit = ((h >> 5) % 1000) / 1000.0
    pan_jit = ((h >> 15) % 1000) / 1000.0
    delta = max(ZOOM_DELTA_MIN, min(ZOOM_DELTA_MAX,
                ZOOM_RATE_BASE * dur * (0.75 + rate_jit * 0.5)))
    # Уже тесный/плотный кадр (крупный план) не нуждается в таком же зуме,
    # как просторный — иначе на кадре и так крупного меча к концу клипа
    # остаётся одна текстура металла, некуда уже приближаться со смыслом.
    busy = estimate_busyness(photo)
    if busy > 0.6:
        delta *= 0.7
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
    # zoompan даёт его сама) — офсет безопасен на любом кадре клипа по построению,
    # а не только к концу движения, как раньше при завязке на delta.
    x = (f"'iw/2-(iw/zoom/2){dx * pan_amt:+.5f}*(1-1/zoom)/2*iw'" if dx
         else "'iw/2-(iw/zoom/2)'")
    y = (f"'ih/2-(ih/zoom/2){dy * pan_amt:+.5f}*(1-1/zoom)/2*ih'" if dy
         else "'ih/2-(ih/zoom/2)'")
    # increase+crop (не decrease+pad) — исходные фото редко ровно 16:9, "вписать
    # в рамку" оставляло чёрные поля по краям на большинстве кадров (проверено:
    # 3 из 8 тестовых кадров с полосами по обеим сторонам). Залить кадр целиком
    # и обрезать лишнее — тот же приём, что уже применялся к видео-стоку ниже.
    vf_base = (f"scale=8000:4500:force_original_aspect_ratio=increase,"
               f"crop=8000:4500,setsar=1,"
               f"zoompan=z={z}:x={x}:y={y}:"
               f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
               f"{film_look(h, section)}")
    vf_overlay = add_overlays(vf_base, dur, title, stat) if (title or stat) else None

    def render(vf):
        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", photo, "-vf", vf,
               "-t", str(dur), "-c:v", "libx264", "-preset", "fast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS), out]
        return subprocess.run(cmd, capture_output=True, text=True)

    if vf_overlay:
        r = render(vf_overlay)
        if r.returncode == 0:
            return True
        # Титр/плашка (drawtext/шрифт) — не повод терять весь кадр: некоторые
        # сборки ffmpeg собраны без drawtext вообще. Откат на версию без них.
        print(f"  титр/плашка не встали ({os.path.basename(out)}), рисую без них: {r.stderr[-200:]}")
    return render(vf_base).returncode == 0


# --- 2.5D-параллакс (опционально, нужны torch/transformers/opencv) ---
# Depth-Anything-V2-Small с Hugging Face: по одной фотке строит карту глубины
# (белое — близко, чёрное — далеко), дальше "ближние" пиксели двигаются
# сильнее "дальних" при панорамировании — настоящее ощущение объёма вместо
# плоского зума. GitHub codeload (откуда обычно тянут MiDaS) закрыт сетевой
# политикой (проверено: 403) — huggingface.co открыт, поэтому источник модели
# именно оттуда.
PARALLAX_MARGIN = 1.5     # рабочий холст на 50% больше кадра — запас под
                           # зум+пан+параллакс-смещение без чёрных дыр по краям
PARALLAX_PX_BASE = 55.0   # максимальный доп.разброс между ближним/дальним слоем, px
_depth_model = None


def get_depth_model():
    global _depth_model
    if _depth_model is None:
        from transformers import pipeline as hf_pipeline
        _depth_model = hf_pipeline(task="depth-estimation",
                                    model="depth-anything/Depth-Anything-V2-Small-hf")
    return _depth_model


def estimate_depth(canvas_bgr):
    """Карта глубины (float32, 0..1, 1=близко) — строится по УЖЕ обрезанному
    холсту (canvas_bgr из fill_crop_canvas), а не по исходному фото.
    Раньше модель считала глубину по несжатому оригиналу (его родное
    соотношение сторон), а карта потом растягивалась голым cv2.resize под
    размер холста — без кропа, который применялся к цветной картинке. При
    несовпадении соотношения сторон (портретное фото под альбомный холст)
    карта глубины съезжала и тянулась не по тому же кадру, что видит
    зритель: параллакс сдвигал пиксели по чужой геометрии — отсюда
    видимое искажение/"плавание" картинки. Считая глубину прямо по canvas,
    геометрия гарантированно совпадает."""
    model = get_depth_model()
    h, w = canvas_bgr.shape[:2]
    img = PILImage.fromarray(canvas_bgr[:, :, ::-1])  # BGR -> RGB
    out = model(img)
    depth = np.array(out["predicted_depth"], dtype=np.float32)
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        return np.full((h, w), 0.5, dtype=np.float32)
    depth = (depth - d_min) / (d_max - d_min)
    # Depth-Anything даёт РЕЗКИЙ край на границе объекта (перепад ~0..1 за
    # 1-2px — проверено: grad p99 ~0.08, максимум >2.0 на пиксель). При
    # remap с parallax_px до ~70px это превращает границу в артефакт
    # "двойного контура"/расслоения на резких силуэтах (рукоять меча на
    # фоне боке) — то, что видно глазом как "ИИ-искажение". Настоящей
    # физически верной параллакс-окклюзии тут всё равно нет (плоская
    # картинка, не многослойная сцена), поэтому лечим не резкость карты,
    # а её крутизну: широкое гауссово размытие растягивает переход через
    # много пикселей, и remap плывёт плавно вместо "разрыва" на 1-2px —
    # тот же приём, что использует Apple/Google в 3D-Ken-Burns эффектах.
    sigma = max(3.0, min(h, w) * 0.018)
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=sigma)
    return depth


def fill_crop_canvas(photo_path, cw, ch):
    """Тот же increase+crop, что и в ffmpeg-пути: заливаем холст целиком,
    обрезаем лишнее — без чёрных полос по краям (BGR для cv2)."""
    img = PILImage.open(photo_path).convert("RGB")
    iw, ih = img.size
    scale = max(cw / iw, ch / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    img = img.resize((nw, nh), PILImage.LANCZOS)
    x0, y0 = (nw - cw) // 2, (nh - ch) // 2
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    arr = np.array(img)[:, :, ::-1].copy()   # RGB -> BGR
    return arr


def parallax_kenburns(photo, out, dur, title=None, zoom_in=None, pan_dir=None, stat=None, section=""):
    """2.5D-версия kenburns(): собственный покадровый рендер (OpenCV remap)
    вместо ffmpeg zoompan — только так можно сделать смещение, зависящее от
    глубины пикселя. При любой накладке (модель не встала, ffmpeg-пайп упал)
    возвращает False — вызывающий код откатывается на обычный kenburns()."""
    global PARALLAX_BROKEN
    if PARALLAX_BROKEN:
        return False
    proc = None
    try:
        frames = max(1, round(dur * FPS))
        cw, ch = round(WIDTH * PARALLAX_MARGIN), round(HEIGHT * PARALLAX_MARGIN)
        canvas = fill_crop_canvas(photo, cw, ch)
        try:
            depth = estimate_depth(canvas)
        except Exception as e:
            # Сбой именно тут почти всегда системный (модель не встала /
            # сеть до HF Hub отвалилась), не проблема конкретной картинки —
            # дальше пытаться на каждом из оставшихся хайлайтов бессмысленно,
            # только тратим время на повторную (медленную) загрузку модели.
            print(f"  depth-модель недоступна, параллакс выключается на остаток ролика: "
                  f"{type(e).__name__} {e}")
            PARALLAX_BROKEN = True
            return False

        h, zoom_in_default, pan_dir_default = kb_hash_choices(photo)
        if zoom_in is None:
            zoom_in = zoom_in_default
        dx, dy = pan_dir if pan_dir is not None else pan_dir_default
        rate_jit = ((h >> 5) % 1000) / 1000.0
        pan_jit = ((h >> 15) % 1000) / 1000.0
        delta = max(ZOOM_DELTA_MIN, min(ZOOM_DELTA_MAX,
                    ZOOM_RATE_BASE * dur * (0.75 + rate_jit * 0.5)))
        max_zoom = ZOOM_FLOOR + delta
        pan_amt_frac = PAN_SAFETY * (PAN_JITTER_MIN + pan_jit * (PAN_JITTER_MAX - PAN_JITTER_MIN))
        parallax_px = PARALLAX_PX_BASE * (0.7 + rate_jit * 0.6)

        cx, cy = cw / 2.0, ch / 2.0
        ox_grid, oy_grid = np.meshgrid(np.arange(WIDTH, dtype=np.float32),
                                        np.arange(HEIGHT, dtype=np.float32))

        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
               "-frames:v", str(frames)]
        vf = film_look(h, section)
        vf = add_overlays(vf, dur, title, stat) if (title or stat) else vf
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-r", str(FPS), out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        for frame_i in range(frames):
            t = frame_i / frames
            eased = 3 * t ** 2 - 2 * t ** 3   # smoothstep — тот же профиль, что у zoompan-версии
            zoom = (ZOOM_FLOOR + delta * eased) if zoom_in else (max_zoom - delta * eased)
            pan_px = pan_amt_frac * eased * min(cw, ch) * 0.5 * ((1 - 1 / zoom))

            map_x0 = cx + (ox_grid - WIDTH / 2) / zoom + dx * pan_px
            map_y0 = cy + (oy_grid - HEIGHT / 2) / zoom + dy * pan_px
            map_x0 = np.clip(map_x0, 0, cw - 1)
            map_y0 = np.clip(map_y0, 0, ch - 1)

            d_here = cv2.remap(depth, map_x0, map_y0, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            extra = (d_here - 0.5) * parallax_px * eased   # центрировано: ближе/дальше среднего
            map_x = np.clip(map_x0 + dx * extra, 0, cw - 1)
            map_y = np.clip(map_y0 + dy * extra, 0, ch - 1)

            frame = cv2.remap(canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.wait(timeout=60)
        if proc.returncode != 0:
            print(f"  параллакс-рендер не встал ({os.path.basename(out)}): "
                  f"{proc.stderr.read().decode(errors='replace')[-200:]}")
            return False
        return True
    except Exception as e:
        print(f"  параллакс сорвался ({os.path.basename(out)}): {type(e).__name__} {e}")
        return False
    finally:
        # Раньше при сбое ПОСРЕДИ покадрового цикла (BrokenPipeError на
        # stdin.write, таймаут на wait) дочерний ffmpeg не убивался вообще —
        # завис бы висячим процессом до конца сессии. Гасим его в любом
        # случае, если он ещё жив к выходу из функции.
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass


# Не динамический ramp внутри клипа (это отдельная и намного более сложная
# задача — переменная PTS-кривая), а постоянная скорость по контексту клипа:
# хук чуть быстрее — энергичнее, финал чуть медленнее — снимает напряжение.
SPEED_BIAS = {"HOOK": 1.12, "FINAL": 0.92}


def video_render(vid, out, dur, title=None, stat=None, section=""):
    """Аналог kenburns(), но для стокового видео: без zoompan (движение уже
    есть в кадре), заливка кадра целиком + обрезка (не letterbox), растяжение
    по времени, если исходный ролик короче нужной длительности."""
    try:
        actual = get_media_duration(vid)
    except Exception:
        actual = dur
    h = int(hashlib.md5(vid.encode()).hexdigest()[:8], 16)
    vf_base = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1"
    bias = next((v for k, v in SPEED_BIAS.items() if section.startswith(k)), 1.0)
    setpts_factor = None
    if actual < dur - 0.05:
        setpts_factor = dur / max(actual, 0.1)   # уже растягиваем нехватку — bias не добавляем поверх
    elif bias != 1.0 and actual >= dur * bias:
        setpts_factor = 1.0 / bias
    if setpts_factor is not None:
        vf_base += f",setpts={setpts_factor:.5f}*PTS"
    vf_base += f",{film_look(h, section)}"
    vf_overlay = add_overlays(vf_base, dur, title, stat) if (title or stat) else None

    def render(vf):
        cmd = ["ffmpeg", "-y", "-i", vid, "-vf", vf, "-t", str(dur), "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-r", str(FPS), out]
        return subprocess.run(cmd, capture_output=True, text=True)

    if vf_overlay:
        r = render(vf_overlay)
        if r.returncode == 0:
            return True
        print(f"  титр/плашка не встали на видео ({os.path.basename(out)}), рисую без них: {r.stderr[-200:]}")
    return render(vf_base).returncode == 0


def pexels_video(query, index, used_ids=None):
    """Тот же принцип, что pexels_photo(): перебираем выдачу и берём первое
    ещё не показанное видео. Из доступных video_files берём ближайшее по
    ширине к целевому 1920 — не тянем 4K ради 1080p-выхода."""
    global PEXELS_BROKEN
    cache = os.path.join(TEMP_FOLDER, "pexels_video_cache")
    os.makedirs(cache, exist_ok=True)
    qhash = hashlib.md5(query.encode()).hexdigest()[:8]
    cf = os.path.join(cache, f"{index:04d}_{qhash}.mp4")
    if os.path.exists(cf):
        return cf
    if not PEXELS_API_KEY:
        return None
    try:
        q = urllib.parse.quote(query)
        req = urllib.request.Request(
            f"https://api.pexels.com/videos/search?query={q}&per_page=40&orientation=landscape",
            headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        videos = data.get("videos") or []
        if not videos:
            return None
        pick = videos[0]
        if used_ids is not None:
            for v in videos:
                if v.get("id") not in used_ids:
                    pick = v
                    break
        if used_ids is not None:
            used_ids.add(pick.get("id"))
        files = [f for f in (pick.get("video_files") or [])
                 if f.get("file_type") == "video/mp4" and f.get("width")]
        if not files:
            return None
        best = min(files, key=lambda f: abs(f["width"] - WIDTH))
        vid_req = urllib.request.Request(best["link"], headers={"User-Agent": UA})
        with urllib.request.urlopen(vid_req, timeout=40) as r:
            open(cf, "wb").write(r.read())
        return cf
    except Exception as e:
        PEXELS_BROKEN = True
        print(f"  Pexels video [{query}]: {e}")
        return None


def xfade_chain(clips, durs, sections, out, xfade_dur=XFADE_DUR):
    """Один проход filter_complex с цепочкой xfade между ВСЕМИ соседними
    кадрами — вместо жёсткой склейки. Тип перехода и длительность варьируются
    (разнообразие + иногда почти жёсткий cut), на границе секций — заметный
    dissolve. Возвращает (True, итоговая_длительность) или (False, 0.0), если
    что-то пошло не так (тогда main() откатывается на обычный concat -c copy)."""
    n = len(clips)
    if n < 2:
        return False, 0.0
    parts, prev_label, cum = [], "0:v", durs[0]
    cut_hist, boundary_hist = [], []
    for i in range(1, n):
        h = int(hashlib.md5(f"{clips[i-1]}|{clips[i]}".encode()).hexdigest()[:8], 16)
        is_boundary = sections[i] != sections[i - 1]
        if is_boundary:
            # Смена темы — заметный переход, не обычная склейка. dissolve/fadeblack/
            # fadewhite (вспышка светом — читается как "новая глава начинается ярко")
            # вперемешку.
            candidate = BOUNDARY_TRANSITIONS[h % len(BOUNDARY_TRANSITIONS)]
            transition = pick_no_repeat(boundary_hist, candidate, BOUNDARY_TRANSITIONS, 1)
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
    cur = get_media_duration(video)
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


def get_media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def ahash(photo_path, size=8):
    """Средний хэш (average hash) картинки — 64-битная строка, дешёвая
    замена imagehash-библиотеке (Pillow уже обязательная зависимость, лишний
    пакет не нужен). Похожие по контенту кадры дают близкий хэш даже если
    это разные файлы с разных источников."""
    img = PILImage.open(photo_path).convert("L").resize((size, size), PILImage.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p > avg else "0" for p in pixels)


def hamming(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def qc_report(media_log):
    """Не блокирует сборку — просто честно докладывает, если несколько
    кадров ролика визуально почти одинаковые (Pexels ID разные, а по факту
    похожий кроп/пересъёмка того же сюжета — то, что ловится глазом, но не
    ловится дедупом по ID). Порог 6 бит из 64 — эмпирически "заметно похоже".
    Возвращает список дублей — раньше это тонуло в середине лога, а ГОТОВО
    выглядело как чистый успех; теперь main() отражает находку в итоговой
    строке и коде возврата (не блокирует сборку — только честно не молчит:
    тот же принцип, что уже применён к пропущенным блокам)."""
    if len(media_log) < 2:
        return []
    hashes = []
    for i, path in media_log:
        try:
            hashes.append((i, ahash(path)))
        except Exception:
            pass
    dupes = []
    for a in range(len(hashes)):
        for bb in range(a + 1, len(hashes)):
            i1, h1 = hashes[a]
            i2, h2 = hashes[bb]
            d = hamming(h1, h2)
            if d <= 6:
                dupes.append((i1 + 1, i2 + 1, d))
    if dupes:
        print(f"QC: похожие кадры (проверь глазами) — {dupes}")
    else:
        print("QC: явных повторов по картинке не найдено")
    return dupes


def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"Аудио не найдено: {AUDIO_FILE}")
        return 1
    total = get_audio_duration()
    print(f"Аудио: {total:.1f}с ({total/60:.1f} мин)")
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    blocks = parse_blocks(SCRIPT_FILE) if os.path.exists(SCRIPT_FILE) else []
    if not blocks:
        print("Сценарий не найден/пуст")
        return 1
    # Кроссфейд между КАЖДОЙ парой кадров суммарно "съедает" (n-1)*XFADE_DUR —
    # закладываем это в целевую длительность заранее, чтобы после склейки общая
    # длина видео снова совпала с аудио (без этого хвост ролика проигрывался бы
    # без картинки под конец аудиодорожки).
    xfade_budget = max(0, len(blocks) - 1) * XFADE_DUR
    target = total + xfade_budget
    durs = block_durations(blocks, target)
    # Второй проход: ритм по громкости поверх word-count-базы (не вместо неё) —
    # громкие места режутся чаще, тихие держатся дольше. Опционально (нужен numpy).
    # Стартовые точки для сэмплинга энергии считаем от РЕАЛЬНОЙ длины аудио
    # (total), не от раздутой под кроссфейды target — иначе поздние блоки на
    # длинном ролике со множеством склеек сэмплили бы энергию не в том месте.
    curve = audio_energy_curve(AUDIO_FILE)
    if curve:
        baseline = block_durations(blocks, total)
        starts, acc = [], 0.0
        for d in baseline:
            starts.append(acc)
            acc += d
        mults = energy_pace_multipliers(curve, starts, baseline)
        durs = block_durations(blocks, target, energy_mults=mults)
        print("Ритм по громкости: включён")
    print(f"Средний кадр: {sum(durs)/len(durs):.1f}с")

    clips, clip_durs, clip_sections = [], [], []
    missing = []   # индексы блоков, для которых не нашлось ни фото, ни видео
    media_log = []   # (индекс, путь_к_фото) — для QC-проверки на похожие кадры в конце
    zoom_hist, pan_hist = [], []
    use_local = os.path.isdir(MEDIA_FOLDER) and bool(local_photo(0))
    use_pexels = bool(PEXELS_API_KEY)
    queries = resolve_queries(blocks)
    used_photo_ids = set()   # общий на весь ролик — не даём одной фотке всплыть дважды
    used_video_ids = set()   # то же самое, отдельно для видео (разные ID-пространства)
    for i, (b, d) in enumerate(zip(blocks, durs)):
        # Титр темы — только на ПЕРВОМ кадре новой секции (BLOCK N: Название).
        is_section_start = i == 0 or blocks[i]["section"] != blocks[i - 1]["section"]
        title = section_title(b["section"]) if is_section_start else None
        stat = b.get("stat")
        # Хэш параметров рендера в имени — иначе правка script.txt (текст,
        # тайминг, плашка) без ручной чистки temp_smart/ молча оставляла
        # старый клип под новые данные (тот же класс бага, что уже правили
        # для pexels_cache). Запрос (queries[i]) обязателен в хэше отдельно —
        # без него правка channel_themes.json/media_plan/themes.json при
        # совпавшей длительности тихо оставляла старый (уже нерелевантный)
        # клип под новым запросом — поймано QC-сверкой вживую: клип с новым
        # запросом "medieval sword close up" продолжал показывать старую
        # киноплёнку, потому что длительность/заголовок/плашка/секция не
        # изменились, а запрос в хэш не входил.
        params_hash = hashlib.md5(
            f"{d:.3f}|{title}|{stat}|{b['section']}|{queries[i]}".encode()).hexdigest()[:8]
        out = os.path.join(TEMP_FOLDER, f"clip_{i:04d}_{params_hash}.mp4")
        if os.path.exists(out):
            clips.append(out)
            clip_durs.append(d)
            clip_sections.append(b["section"])
            continue
        photo = local_photo(i) if use_local else None
        video = None
        if not photo and use_pexels:
            # Чередуем фото/видео через один — живое движение вперемешку со
            # статикой вместо чистого слайд-шоу (ЧАСТЬ 14: нечётные фото,
            # чётные видео). Слишком короткому кадру видео не заказываем —
            # не тянуть ролик ради 4 секунд. Если у предпочтённого типа для
            # этой темы пусто — откатываемся на другой тип, а не теряем кадр.
            prefer_video = (i % 2 == 1) and d >= MIN_CLIP + 1.0
            if prefer_video:
                video = pexels_video(queries[i], i, used_ids=used_video_ids)
                if not video:
                    photo = pexels_photo(queries[i], i, used_ids=used_photo_ids)
            else:
                photo = pexels_photo(queries[i], i, used_ids=used_photo_ids)
                if not photo and d >= MIN_CLIP + 1.0:
                    video = pexels_video(queries[i], i, used_ids=used_video_ids)
            # Раньше Pexels отключался навсегда после ЛЮБОГО промаха, включая
            # обычную пустую выдачу по одному неудачному запросу. Гасим источник
            # только если API реально отвалился.
            if not photo and not video and PEXELS_BROKEN:
                use_pexels = False
        if not photo and not video:
            photo = local_photo(i)
        if not photo and not video:
            print(f"  [{i+1}] нет медиа")
            missing.append(i + 1)
            continue
        if video:
            ok = video_render(video, out, d, title=title, stat=stat, section=b["section"])
        else:
            # anti-repetition: хэш сам по себе не мешает 3 зумам подряд случайно
            # совпасть — держим окно последних решений и форсируем смену при повторе.
            _, zi_cand, pd_cand = kb_hash_choices(photo)
            zoom_in = pick_no_repeat(zoom_hist, zi_cand, [True, False], max_repeat=2)
            pan_dir = pick_no_repeat(pan_hist, pd_cand, PAN_DIRECTIONS, max_repeat=2)
            # Параллакс — только на самые заметные точки ролика (хук целиком +
            # первый кадр каждого раздела), не на все фото: покадровый рендер
            # с depth-моделью в разы дороже по времени zoompan-версии, на
            # 40+ кадрах это лишние десятки минут ради эффекта, который
            # большую часть ролика зритель всё равно не разглядывает так
            # пристально, как хук и открывашки разделов.
            is_highlight = b["section"].startswith("HOOK") or is_section_start
            ok = False
            if PARALLAX_ENABLED and is_highlight:
                ok = parallax_kenburns(photo, out, d, title=title, zoom_in=zoom_in,
                                        pan_dir=pan_dir, stat=stat, section=b["section"])
            if not ok:
                ok = kenburns(photo, out, d, title=title, zoom_in=zoom_in, pan_dir=pan_dir,
                              stat=stat, section=b["section"])
        if ok:
            clips.append(out)
            clip_durs.append(d)
            clip_sections.append(b["section"])
            if not video and photo:
                media_log.append((i, photo))
            if i % 20 == 0 or i < 3:
                print(f"  [{i+1}/{len(blocks)}] {d:.1f}с {b['words']} слов ({'видео' if video else 'фото'})")
        else:
            missing.append(i + 1)
        if use_pexels and not use_local and i % 10 == 9:
            time.sleep(0.4)

    if not clips:
        print("Нет клипов")
        return 1
    merged = os.path.join(TEMP_FOLDER, "merged.mp4")
    ok, xfade_total = xfade_chain(clips, clip_durs, clip_sections, merged)
    if not ok:
        concat = os.path.join(TEMP_FOLDER, "concat.txt")
        # Пути ТОЛЬКО абсолютные: concat-демуксер ffmpeg резолвит относительные
        # пути от папки самого concat.txt, а не от cwd — иначе сборка падает.
        open(concat, "w", encoding="utf-8").write(
            "".join(f"file '{os.path.abspath(c)}'\n" for c in clips))
        r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", concat, "-c", "copy", merged], capture_output=True, text=True)
        if r.returncode != 0:
            print("Склейка:", r.stderr[-300:])
            return 1
    merged = pad_to_length(merged, total, TEMP_FOLDER)

    # -shortest САМ ПО СЕБЕ недостаточен с -c:v copy: копирование пакетов
    # режет только по границам GOP исходного клипа, а не по факту конца
    # аудио — на реальном 17-минутном ролике это давало +6с видео без
    # звука сверху (проверено вживую). Явный -t с точной длительностью
    # аудио режет ровно там, где надо, независимо от размера GOP.
    r = subprocess.run(["ffmpeg", "-y", "-i", merged, "-i", AUDIO_FILE,
                        "-t", f"{total:.3f}",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", OUTPUT_FILE], capture_output=True, text=True)
    if r.returncode != 0:
        print("Аудио:", r.stderr[-300:])
        return 1
    mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    # Пропущенные кадры и раньше не останавливали сборку (стратегия
    # "лучше меньше клипов, чем сорванный рендер") — но раньше это тонуло
    # в середине лога, а ГОТОВО выглядело как чистый успех. Теперь пропуски
    # видны в итоговой строке, и код возврата честно отражает, что не всё
    # доехало (не 0, если что-то пропущено).
    dupes = qc_report(media_log)
    status = f" | ПРОПУЩЕНО {len(missing)} блоков: {missing}" if missing else ""
    # Дубли по фото — та же логика: не блокируем сборку (видео уже готово
    # и в целом смотрибельно), но не даём находке потеряться в середине
    # лога и не даём коду возврата соврать, что всё чисто.
    status += f" | QC: {len(dupes)} похожих пар кадров — проверь глазами" if dupes else ""
    print(f"\nГОТОВО: {OUTPUT_FILE} ({mb:.0f} MB, {total/60:.1f} мин, {len(clips)} кадров){status}")
    return 1 if (missing or dupes) else 0


if __name__ == "__main__":
    sys.exit(main())
