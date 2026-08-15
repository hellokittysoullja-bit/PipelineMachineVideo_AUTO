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
def film_look(photo_hash):
    c = 1.03 + (photo_hash % 100) / 100 * 0.05          # 1.03-1.08
    s = 1.04 + ((photo_hash >> 7) % 100) / 100 * 0.09    # 1.04-1.13
    b = ((photo_hash >> 14) % 100) / 100 * 0.02          # 0.00-0.02
    return f"eq=contrast={c:.3f}:saturation={s:.3f}:brightness={b:.3f},vignette=PI/5,noise=alls=2:allf=t+u"


XFADE_DUR = 0.4        # диссолв на границах секций и часть обычных склеек
XFADE_DUR_HARD = 0.06  # почти мгновенный переход — читается как жёсткий cut
XFADE_TRANSITIONS = ["fade", "dissolve", "smoothleft", "smoothright", "smoothup", "smoothdown"]
HOOK_MAX_CLIP = 5.0     # в хуке кадры короче и чаще — критично для удержания первых секунд
PAUSE_DURATIONS = {"[pause]": 0.8, "[short pause]": 0.4,
                   "[slowly]": 0.0, "[emphasis]": 0.0, "[energetic]": 0.0}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)


def section_title(name):
    """BLOCK N: Название -> 'Название'. HOOK/FINAL/безымянные BLOCK — без титра."""
    m = re.match(r'BLOCK\s+\d+\s*:\s*(.+)', name, re.I)
    return m.group(1).strip() if m else None


def escape_drawtext(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")


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
    processed = content
    for tag in sorted(PAUSE_DURATIONS, key=len, reverse=True):
        processed = processed.replace(tag, f"__PAUSE_{PAUSE_DURATIONS[tag]}__")
    processed = re.sub(r'\[.*?\]', '', processed)
    parts = re.split(r'(__PAUSE_[\d.]+__|\x00SECTION:.*?\x00)', processed)
    blocks, cur, pause = [], "", 0.0
    section, pending_section = "BODY", "BODY"
    for part in parts:
        mp = re.match(r'__PAUSE_([\d.]+)__', part)
        ms = re.match(r'\x00SECTION:(.*?)\x00', part)
        if mp:
            pause += float(mp.group(1))
        elif ms:
            section = ms.group(1)
        else:
            t = part.strip()
            if t:
                if cur:
                    blocks.append({"text": cur, "pause_after": pause,
                                   "words": len(cur.split()), "section": pending_section})
                cur, pause = t, 0.0
                pending_section = section
    if cur:
        blocks.append({"text": cur, "pause_after": pause,
                       "words": len(cur.split()), "section": pending_section})
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
    по нескольку раз за ролик. Перебираем выдачу (per_page=10) и берём первый
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
            f"https://api.pexels.com/v1/search?query={q}&per_page=10&orientation=landscape",
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


def kb_hash_choices(photo):
    """Кандидаты зума/пана по хэшу файла — детерминированно, но БЕЗ памяти о
    соседних клипах (это даёт anti-repetition в main(), см. pick_no_repeat)."""
    h = int(hashlib.md5(photo.encode()).hexdigest()[:8], 16)
    zoom_in = bool(h & 1)
    pan_dir = PAN_DIRECTIONS[(h >> 1) % len(PAN_DIRECTIONS)]
    return h, zoom_in, pan_dir


def kenburns(photo, out, dur, title=None, zoom_in=None, pan_dir=None):
    frames = max(1, round(dur * FPS))
    h, zoom_in_default, pan_dir_default = kb_hash_choices(photo)
    if zoom_in is None:
        zoom_in = zoom_in_default
    dx, dy = pan_dir if pan_dir is not None else pan_dir_default
    rate_jit = ((h >> 5) % 1000) / 1000.0
    pan_jit = ((h >> 15) % 1000) / 1000.0
    delta = max(ZOOM_DELTA_MIN, min(ZOOM_DELTA_MAX,
                ZOOM_RATE_BASE * dur * (0.75 + rate_jit * 0.5)))
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
               f"{film_look(h)}")
    vf_title = None
    if title and FONT_PATH:
        # Титр новой темы держится первые ~2.5с клипа — плавный fade in/out,
        # это то самое "кто-то сел и отредактировал", а не бот нарезал слайды.
        hold = min(2.5, dur * 0.6)
        fin = max(0.3, hold * 0.3)
        safe = escape_drawtext(title)
        vf_title = vf_base + (
            f",drawtext=fontfile='{FONT_PATH}':text='{safe}':"
            f"fontcolor=white:fontsize=54:borderw=3:bordercolor=black@0.7:"
            f"x=(w-text_w)/2:y=h-180:"
            f"alpha='if(lt(t\\,{fin:.2f})\\,t/{fin:.2f}\\,"
            f"if(lt(t\\,{hold:.2f})\\,1\\,max(0\\,1-(t-{hold:.2f})/0.4)))'")

    def render(vf):
        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", photo, "-vf", vf,
               "-t", str(dur), "-c:v", "libx264", "-preset", "fast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS), out]
        return subprocess.run(cmd, capture_output=True, text=True)

    if vf_title:
        r = render(vf_title)
        if r.returncode == 0:
            return True
        # Титр (drawtext/шрифт) — не повод терять весь кадр: некоторые сборки
        # ffmpeg собраны без drawtext вообще. Откатываемся на версию без титра.
        print(f"  титр не встал ({os.path.basename(out)}), рисую без него: {r.stderr[-200:]}")
    return render(vf_base).returncode == 0


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
            # Смена темы — заметный переход, не обычная склейка. dissolve/fadeblack
            # вперемешку (fadeblack — через чёрный кадр, "конец главы").
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
    zoom_hist, pan_hist = [], []
    use_local = os.path.isdir(MEDIA_FOLDER) and bool(local_photo(0))
    use_pexels = bool(PEXELS_API_KEY)
    queries = resolve_queries(blocks)
    used_photo_ids = set()   # общий на весь ролик — не даём одной фотке всплыть дважды
    for i, (b, d) in enumerate(zip(blocks, durs)):
        out = os.path.join(TEMP_FOLDER, f"clip_{i:04d}.mp4")
        # Титр темы — только на ПЕРВОМ кадре новой секции (BLOCK N: Название).
        is_section_start = i == 0 or blocks[i]["section"] != blocks[i - 1]["section"]
        title = section_title(b["section"]) if is_section_start else None
        if os.path.exists(out):
            clips.append(out)
            clip_durs.append(d)
            clip_sections.append(b["section"])
            continue
        photo = local_photo(i) if use_local else None
        if not photo and use_pexels:
            photo = pexels_photo(queries[i], i, used_ids=used_photo_ids)
            # Раньше Pexels отключался навсегда после ЛЮБОГО промаха, включая
            # обычную пустую выдачу по одному неудачному запросу. Гасим источник
            # только если API реально отвалился.
            if not photo and PEXELS_BROKEN:
                use_pexels = False
        if not photo:
            photo = local_photo(i)
        if not photo:
            print(f"  [{i+1}] нет фото")
            continue
        # anti-repetition: хэш сам по себе не мешает 3 зумам подряд случайно
        # совпасть — держим окно последних решений и форсируем смену при повторе.
        _, zi_cand, pd_cand = kb_hash_choices(photo)
        zoom_in = pick_no_repeat(zoom_hist, zi_cand, [True, False], max_repeat=2)
        pan_dir = pick_no_repeat(pan_hist, pd_cand, PAN_DIRECTIONS, max_repeat=2)
        if kenburns(photo, out, d, title=title, zoom_in=zoom_in, pan_dir=pan_dir):
            clips.append(out)
            clip_durs.append(d)
            clip_sections.append(b["section"])
            if i % 20 == 0 or i < 3:
                print(f"  [{i+1}/{len(blocks)}] {d:.1f}с {b['words']} слов")
        if use_pexels and not use_local and i % 10 == 9:
            time.sleep(0.4)

    if not clips:
        print("Нет клипов")
        return
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
            return
    merged = pad_to_length(merged, total, TEMP_FOLDER)
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
