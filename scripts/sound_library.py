#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Библиотека НАСТОЯЩИХ звуков: полевые записи и эффекты под CC0, отобранные
двумя моделями и измерительными гейтами. Заменяет процедурный синтез там,
где он звучал как «гул вентилятора» (прямая жалоба владельца 13.09).

ОТКУДА. Openverse Audio API (без ключа) индексирует Freesound и Wikimedia
Commons; берётся ТОЛЬКО `license=cc0` — и в запросе, и повторно на каждом
результате (fail-closed, тот же принцип, что у картинок из Openverse).
CC0 — юридический отказ от прав, атрибуция не нужна; манифест всё равно
пишется (assets/library/manifest.json). При FREESOUND_API_KEY в .env
дополнительно идёт прямой поиск Freesound с фильтром лицензии — его индекс
полнее. Файлы — hq-превью Freesound (MP3 128 kbps): оригинальный WAV
Freesound отдаёт только по OAuth-авторизации пользователя, это не «ключ в
.env». Под закадром на -24..-40 LU ниже голоса разницы нет.

КАК ОТБИРАЕТСЯ — то же устройство, что у подбора картинок, перенесённое на
звук:
  1. Измерительные гейты (ffmpeg): длительность, доля тишины, диапазон
     громкости (LRA — «события» в фоне лезут из-под голоса), клиппинг,
     сетевой гул 50/60 Гц.
  2. CLAP (laion/larger_clap_general) — текст<->звук, как CLIP для картинок:
     положительный промпт против списка ловушек-негативов (речь, музыка,
     транспорт, гул, дисторшн). Считается на НЕСКОЛЬКИХ окнах записи
     (15/50/85%) — тот же урок, что video_negative_anchor_violation: брак
     в одном месте записи не виден в другом. Решает МАРЖА на одной записи,
     не абсолютный скор (сырой косинус между разными текстами несопоставим —
     см. историю негативного вето по картинкам).
  3. AST (MIT/ast-finetuned-audioset) — второй, независимый судья, как Jina
     рядом с SigLIP2: вероятности классов AudioSet «Speech»/«Music»/
     «Vehicle» — жёсткое вето на атмосферу с голосами, музыкой, машинами.

ЧТО НА ВЫХОДЕ. assets/library/ambience/<вид>/*.flac (48 кГц, стерео, пик
-12 dBFS, до AMBIENCE_MAX_SEC) и assets/library/sfx/<вид>/*.flac (пик -10).
Те же нормировки, что у синтезированных ассетов, — поэтому все усиления в
pipeline_smart.py остаются в силе. Сборка предпочитает библиотеку, синтез
остаётся запасным путём, если для вида ничего не прошло гейты — с явным
предупреждением, не молча.

Запуск:
    python scripts/sound_library.py build            # всё
    python scripts/sound_library.py build --kinds ambience:wind_open,sfx:chapter_turn
    python scripts/sound_library.py report
"""
import argparse
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

print = functools.partial(print, flush=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY_ROOT = os.path.join(ROOT, "assets", "library")
MANIFEST_PATH = os.path.join(LIBRARY_ROOT, "manifest.json")
CACHE_DIR = os.path.join(ROOT, "temp_library")
UA = "PipelineMachineVideo/1.0 (+sound library, CC0 only)"

OPENVERSE_AUDIO = "https://api.openverse.org/v1/audio/"
FREESOUND_SEARCH = "https://freesound.org/apiv2/search/text/"
SAFE_LICENSES = ("cc0",)

AMBIENCE_PEAK_DBFS = -12.0
SFX_PEAK_DBFS = -10.0
AMBIENCE_MAX_SEC = 180.0
CLAP_WINDOW_SEC = 10.0
CLAP_WINDOW_FRACS = (0.15, 0.5, 0.85)

# Порог маржи CLAP: положительный промпт обязан ПЕРЕБИВАТЬ худшую ловушку.
CLAP_MIN_MARGIN = 0.04
CLAP_MIN_POSITIVE = 0.08
# AST: вероятность класса выше — вето (на любом окне).
AST_VETO = {"Speech": 0.12, "Music": 0.25, "Vehicle": 0.30, "Singing": 0.12}
# Измерительные гейты атмосферы
AMB_MAX_SILENCE_SHARE = 0.25
AMB_MAX_LRA = 18.0
CLIP_SAMPLE_SHARE = 1e-4     # доля сэмплов на |1.0| — клиппинг
HUM_PROMINENCE_DB = 14.0     # узкая линия 50/60 Гц над соседями ±3..10 Гц

# Слова в НАЗВАНИИ записи, при которых кандидат отбрасывается до моделей.
# Тот же принцип, что filter_alt_blocklist() у Pexels: у CLAP ветер, дождь,
# град и прибой — акустические соседи (эксперимент 13.09: ловушка «ocean
# waves» роняла настоящий ветер с +0.075 до -0.089), а в названии автор
# пишет прямо: «Big waves breaking», «Hail Comes In», «Door and wind».
TITLE_BLOCK_COMMON = ("loop", "synth", "generated", "processed", "reverb test", "test ")
TITLE_BLOCK = {
    "wind_open": ("wave", "sea", "surf", "ocean", "beach", "rain", "hail", "thunder", "storm",
                  "door", "window", "indoor", "inside", "room", "car", "train", "city", "street"),
    "forest_birds": ("city", "street", "traffic", "zoo", "cage", "indoor", "room", "rain"),
    "night": ("city", "street", "traffic", "party", "club", "indoor"),
    "stone_hall": ("outdoor", "street", "traffic", "crowd", "concert", "organ", "choir"),
    "forge_fire": ("rain", "storm", "fireworks", "explosion", "gun"),
    "rain_mud": ("indoor", "inside", "window", "roof", "car", "tent", "umbrella", "thunder", "storm", "sea", "wave"),
    "crowd_market": ("stadium", "concert", "protest", "applause", "cheer", "traffic", "indoor", "restaurant", "cafe"),
    "river_stream": ("sea", "wave", "surf", "rain", "waterfall", "fountain", "tap", "sink", "toilet", "shower"),
    "chapter_turn": ("sword", "hit", "impact", "explosion", "punch"),
    "plate_tick": ("clock", "metronome", "loop"),
    "reveal_riser": (),
    "reveal_hit": ("cymbal", "drum kit", "snare", "gun", "explosion"),
    "typewriter": ("loop", "typing fast", "sequence"),
}


def title_blocked(name, title):
    t = (title or "").lower()
    return next((w for w in TITLE_BLOCK_COMMON + TITLE_BLOCK.get(name, ()) if w in t), None)


def title_relevance(spec, title):
    """Сколько ключевых слов запросов встречается в названии — дешёвый
    первичный порядок: кандидаты с «wind»/«field» в названии оцениваются
    моделями раньше, чем случайные соседи по выдаче."""
    words = {w for q in spec["queries"] for w in q.lower().split()
             if w not in ("ambience", "ambient", "sound", "sounds", "single", "short", "soft", "cinematic")}
    t = (title or "").lower()
    return sum(1 for w in words if w in t)


NEGATIVE_PROMPTS = (
    "people talking, human speech, voices",
    "music, melody, musical instruments, singing",
    "traffic, car engine, motor vehicle, airplane",
    "electrical hum, buzzing, mains noise, fan noise",
    "digital distortion, clipping, glitch, static",
)

# Что ищем. queries — поисковые запросы (И-логика у Freesound слабая, поэтому
# несколько коротких); prompt — положительное описание для CLAP; extra_neg —
# ловушки сверх общих. Длительности в секундах.
LIBRARY_SPEC = {
    "ambience": {
        "wind_open": dict(
            queries=["wind open field", "field ambience wind", "wind grass meadow", "moorland wind",
                     "steppe wind ambience", "windy plain"],
            prompt="steady wind blowing over an open field, outdoors, no people",
            min_sec=45, keep=3),
        "forest_birds": dict(
            queries=["forest birds ambience", "birds spring forest", "woodland birdsong ambience",
                     "forest ambience morning", "park birds ambience"],
            prompt="quiet forest ambience with birds singing softly in the distance",
            min_sec=45, keep=3),
        "night": dict(
            queries=["night ambience crickets", "night forest ambience", "night countryside ambience",
                     "owl night ambience"],
            prompt="calm night ambience outdoors with crickets and distant owls",
            min_sec=45, keep=3),
        "stone_hall": dict(
            queries=["church interior ambience", "cathedral ambience quiet", "room tone hall reverb",
                     "empty hall room tone", "castle interior ambience", "monastery ambience"],
            prompt="quiet interior room tone of a large stone hall, distant reverberant space",
            extra_neg=("footsteps walking",), min_sec=30, keep=3),
        "forge_fire": dict(
            queries=["fireplace crackling", "campfire crackling", "bonfire", "wood fire burning",
                     "blacksmith forge fire"],
            prompt="a wood fire burning and crackling, calm and steady",
            min_sec=30, keep=3),
        "rain_mud": dict(
            queries=["rain ambience", "light rain outdoors", "rain on grass field", "gentle rain nature",
                     "rain forest ambience"],
            prompt="light steady rain falling outdoors, natural, no people",
            min_sec=45, keep=3),
        "crowd_market": dict(
            queries=["market crowd ambience", "crowd murmur walla", "village market crowd",
                     "medieval fair crowd", "outdoor crowd ambience distant"],
            prompt="distant murmur of a crowd at an outdoor market, indistinct voices, no clear words",
            # речь тут допустима как далёкий гомон — ловушка «разборчивые слова»
            extra_neg=("a person speaking clearly into a microphone", "announcement, narration"),
            min_sec=30, keep=3, ast_veto={"Music": 0.25, "Vehicle": 0.30}),
        "river_stream": dict(
            queries=["stream water flowing", "river ambience", "brook water", "creek ambience"],
            prompt="a small stream of water flowing gently over stones",
            min_sec=45, keep=2),
    },
    "sfx": {
        "chapter_turn": dict(
            queries=["whoosh transition soft", "cinematic whoosh short", "air whoosh swoosh",
                     "subtle whoosh", "deep whoosh"],
            prompt="a short soft whoosh transition sound, smooth, cinematic",
            min_sec=0.25, max_sec=1.6, keep=8),
        "plate_tick": dict(
            queries=["ui tick", "soft click ui", "notification tick", "ui blip subtle", "menu click"],
            prompt="a very short soft click or tick, user interface sound",
            min_sec=0.04, max_sec=0.5, keep=4),
        "reveal_riser": dict(
            queries=["riser cinematic", "tension riser", "suspense rise", "cinematic build up short"],
            prompt="a cinematic riser building tension, rising sweep",
            min_sec=1.0, max_sec=4.0, keep=3),
        "reveal_hit": dict(
            queries=["low impact cinematic", "deep boom hit", "sub impact", "cinematic hit low"],
            prompt="a deep low cinematic impact boom, single hit, documentary",
            extra_neg=("drum kit, cymbal crash", "explosion with debris"),
            min_sec=0.4, max_sec=3.0, keep=3),
        "typewriter": dict(
            queries=["typewriter key single", "typewriter keystroke", "mechanical keyboard single key",
                     "keyboard key press single click"],
            prompt="a single typewriter key press, one click",
            min_sec=0.03, max_sec=0.6, keep=8),
    },
}


# --------------------------------------------------------------- источники
# Openverse без ключа: 20 запросов/мин и 200/день (заголовки x-ratelimit-*,
# замерено 13.09). Первая версия слала запросы подряд и получала HTTP 429 на
# ВСЕХ видах после первого — сборка «завершалась» за секунды с нулём
# результатов. Теперь: пауза между поисковыми запросами, повтор при 429 с
# ожиданием и дисковый кэш выдачи, чтобы перезапуск не тратил лимит заново.
SEARCH_MIN_INTERVAL_SEC = 3.3
_last_search = [0.0]


def _get_json(url, headers=None, timeout=40, retries=3):
    for attempt in range(retries + 1):
        wait = SEARCH_MIN_INTERVAL_SEC - (time.time() - _last_search[0])
        if wait > 0:
            time.sleep(wait)
        _last_search[0] = time.time()
        req = urllib.request.Request(url, headers=dict({"User-Agent": UA}, **(headers or {})))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(20.0 * (attempt + 1))
                continue
            raise


def _search_cached(cache_key, fetch):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "search_" + hashlib.sha1(cache_key.encode()).hexdigest()[:16] + ".json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    data = fetch()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def is_safe_license(item):
    """Fail-closed: строго cc0 по полю результата, не по фильтру запроса."""
    return str(item.get("license", "")).lower() in SAFE_LICENSES


def openverse_search(query, pages=2):
    """Анонимно Openverse отдаёт не больше 20 результатов на страницу
    (page_size>20 -> HTTP 401, поймано вживую), поэтому глубина — страницами:
    две на атмосферу, это ещё один запрос из дневных 200."""
    results = []
    for page in range(1, pages + 1):
        url = OPENVERSE_AUDIO + "?" + urllib.parse.urlencode(
            {"q": query, "license": "cc0", "page_size": 20, "page": page})
        try:
            data = _search_cached("openverse|" + url, lambda: _get_json(url))
        except Exception as e:
            print(f"    openverse: {type(e).__name__} на «{query}» (стр. {page})")
            break
        chunk = data.get("results") or []
        results.extend(chunk)
        if len(chunk) < 20:
            break
    out = []
    for it in results:
        if not is_safe_license(it):
            continue
        out.append({
            "source": f"openverse:{it.get('provider')}",
            "id": f"{it.get('provider')}:{it.get('id')}",
            "foreign_id": (it.get("foreign_landing_url") or "").rstrip("/").split("/")[-1],
            "title": it.get("title") or "",
            "creator": it.get("creator") or "",
            "license": it.get("license"),
            "license_url": it.get("license_url"),
            "url": it.get("url"),
            "landing": it.get("foreign_landing_url"),
            "duration": (it.get("duration") or 0) / 1000.0,
            "query": query,
        })
    return out


def freesound_search(query, key, page_size=30):
    """Прямой поиск Freesound — только при ключе, и только CC0.
    Отдаёт hq-превью (оригинал требует OAuth пользователя)."""
    if not key:
        return []
    url = FREESOUND_SEARCH + "?" + urllib.parse.urlencode({
        "query": query, "filter": 'license:"Creative Commons 0"', "page_size": page_size,
        "fields": "id,name,username,license,duration,previews,url", "token": key})
    out = []
    try:
        data = _search_cached("freesound|" + query, lambda: _get_json(url))
    except Exception as e:
        print(f"    freesound: {type(e).__name__} на «{query}»")
        return out
    for it in data.get("results") or []:
        lic = str(it.get("license", ""))
        if "publicdomain/zero" not in lic and "Creative Commons 0" not in lic:
            continue
        prev = (it.get("previews") or {}).get("preview-hq-mp3")
        if not prev:
            continue
        out.append({
            "source": "freesound", "id": f"freesound:{it['id']}", "foreign_id": str(it["id"]),
            "title": it.get("name") or "", "creator": it.get("username") or "",
            "license": "cc0", "license_url": lic, "url": prev, "landing": it.get("url"),
            "duration": float(it.get("duration") or 0), "query": query,
        })
    return out


def gather_candidates(spec):
    key = os.environ.get("FREESOUND_API_KEY", "").strip()
    seen, out = set(), []
    pages = 2 if spec.get("min_sec", 0) >= 30 else 1
    for q in spec["queries"]:
        for item in openverse_search(q, pages) + freesound_search(q, key):
            fid = item["foreign_id"] or item["id"]
            if fid in seen:
                continue
            seen.add(fid)
            out.append(item)
    return out


def preview_url(url, quality):
    """Freesound кладёт превью парами: ..._NNN-hq.mp3 (128 kbps) и -lq.mp3
    (64 kbps). Оценка идёт по lq — вдвое меньше трафика на 15-20 кандидатов,
    а решение «есть ли речь/музыка/гул и про то ли это» от битрейта не
    зависит; hq качается только для принятых."""
    if "freesound.org/previews/" in url:
        return re.sub(r"-(hq|lq)\.(mp3|ogg)$", f"-{quality}.\\2", url)
    return url


def download(item, quality="hq"):
    os.makedirs(CACHE_DIR, exist_ok=True)
    url = preview_url(item["url"], quality)
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"{h}.audio")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=180) as r, open(path + ".tmp", "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(path + ".tmp", path)
        return path
    except Exception as e:
        print(f"    скачивание: {type(e).__name__} {item['title'][:40]!r}")
        return None


# ------------------------------------------------------------ измерения
def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def probe_duration(path):
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    try:
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def decode_f32(path, start, dur, sr):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
                        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"], capture_output=True)
    import numpy as np
    return np.frombuffer(r.stdout, dtype=np.float32)


def measure_loudness(path):
    """(integrated LUFS, LRA, true peak dBTP) через ebur128."""
    r = _run(["ffmpeg", "-v", "info", "-t", "190", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"])
    txt = r.stderr
    def grab(pattern):
        m = re.findall(pattern, txt)
        return float(m[-1]) if m else None
    return grab(r"I:\s*(-?[\d.]+) LUFS"), grab(r"LRA:\s*([\d.]+) LU"), grab(r"Peak:\s*(-?[\d.]+) dBFS")


def silence_share(path, duration):
    r = _run(["ffmpeg", "-v", "info", "-t", "190", "-i", path, "-af", "silencedetect=noise=-45dB:d=0.5", "-f", "null", "-"])
    total = 0.0
    for m in re.finditer(r"silence_duration:\s*([\d.]+)", r.stderr):
        total += float(m.group(1))
    return total / min(duration, 190.0) if duration else 0.0


def clipping_share(samples):
    import numpy as np
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples) >= 0.999))


def hum_prominence_db(samples, sr):
    """Насколько УЗКАЯ линия на 50/60 Гц (и гармониках) торчит над своими
    ближайшими соседями по спектру (3..10 Гц в стороны).

    Именно над соседями, а не над медианой полосы: сетевой гул — линия
    шириной в доли герца, а ветер и огонь — широкий низкочастотный гул,
    у которого 50 Гц ничем не выделяются среди 45 и 55. Первая версия
    сравнивала с медианой 30..300 Гц и отбраковывала настоящий ветер как
    «гул» — поймано на первом же реальном кандидате.
    """
    import numpy as np
    n = min(samples.size, sr * 8)
    if n < sr:
        return 0.0
    x = samples[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    worst = 0.0
    for f0 in (50.0, 60.0, 100.0, 120.0, 150.0, 180.0):
        line = spec[(freqs >= f0 - 0.6) & (freqs <= f0 + 0.6)]
        side = spec[((freqs >= f0 - 10) & (freqs <= f0 - 3)) | ((freqs >= f0 + 3) & (freqs <= f0 + 10))]
        if line.size and side.size:
            worst = max(worst, 10 * math.log10((line.max() + 1e-12) / (side.mean() + 1e-12)))
    return worst


# ---------------------------------------------------------------- модели
_CLAP = {}
_AST = {}


def clap():
    if "model" not in _CLAP:
        import torch
        from transformers import ClapModel, ClapProcessor
        name = "laion/larger_clap_general"
        _CLAP["model"] = ClapModel.from_pretrained(name).eval()
        _CLAP["proc"] = ClapProcessor.from_pretrained(name)
        _CLAP["torch"] = torch
    return _CLAP


def ast():
    if "model" not in _AST:
        import torch
        from transformers import AutoFeatureExtractor, ASTForAudioClassification
        name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        _AST["fe"] = AutoFeatureExtractor.from_pretrained(name)
        _AST["model"] = ASTForAudioClassification.from_pretrained(name).eval()
        m = _AST["model"]
        _AST["label_idx"] = {lab: i for i, lab in m.config.id2label.items()}
        _AST["torch"] = torch
    return _AST


def clap_scores(windows, prompts):
    """[[скор по каждому промпту] для каждого окна]."""
    c = clap()
    torch = c["torch"]
    with torch.no_grad():
        ti = c["proc"](text=list(prompts), return_tensors="pt", padding=True)
        t = c["model"].get_text_features(**ti)
        t = getattr(t, "pooler_output", t)
        if not hasattr(t, "norm"):
            t = t[0]
        t = t / t.norm(dim=-1, keepdim=True)
        rows = []
        for w in windows:
            ai = c["proc"](audio=w, sampling_rate=48000, return_tensors="pt")
            a = c["model"].get_audio_features(**ai)
            a = getattr(a, "pooler_output", a)
            if not hasattr(a, "norm"):
                a = a[0]
            a = a / a.norm(dim=-1, keepdim=True)
            rows.append((a @ t.T)[0].tolist())
    return rows


def ast_probs(windows16k, labels):
    m = ast()
    torch = m["torch"]
    out = []
    with torch.no_grad():
        for w in windows16k:
            logits = m["model"](**m["fe"](w, sampling_rate=16000, return_tensors="pt")).logits[0]
            p = torch.sigmoid(logits)
            out.append({lab: float(p[m["label_idx"][lab]]) for lab in labels if lab in m["label_idx"]})
    return out


# ------------------------------------------------------------- оценка
def evaluate(item, path, kind, name, spec):
    """Полный разбор одного кандидата -> dict с вердиктом и причиной."""
    verdict = {"id": item["id"], "title": item["title"], "reasons": []}
    dur = probe_duration(path)
    if not dur:
        verdict["reasons"].append("undecodable")
        return verdict
    verdict["duration"] = round(dur, 3)
    lo, hi = spec.get("min_sec", 0.0), spec.get("max_sec", 1e9)
    if dur < lo or dur > hi:
        verdict["reasons"].append(f"duration_{'short' if dur < lo else 'long'}")
        return verdict

    is_amb = kind == "ambience"
    # окна анализа
    if dur <= CLAP_WINDOW_SEC + 1.0:
        starts = [0.0]
    else:
        starts = [max(0.0, min(dur - CLAP_WINDOW_SEC, f * dur - CLAP_WINDOW_SEC / 2)) for f in CLAP_WINDOW_FRACS]
    win48 = [decode_f32(path, s, min(CLAP_WINDOW_SEC, dur), 48000) for s in starts]
    win48 = [w for w in win48 if w.size > 4800]
    if not win48:
        verdict["reasons"].append("undecodable")
        return verdict

    # измерительные гейты
    clip = max(clipping_share(w) for w in win48)
    verdict["clipping_share"] = round(clip, 6)
    if clip > CLIP_SAMPLE_SHARE:
        verdict["reasons"].append("clipping")
    if is_amb:
        hum = max(hum_prominence_db(w, 48000) for w in win48)
        verdict["hum_db"] = round(hum, 1)
        if hum > HUM_PROMINENCE_DB:
            verdict["reasons"].append("mains_hum")
        sil = silence_share(path, dur)
        verdict["silence_share"] = round(sil, 3)
        if sil > AMB_MAX_SILENCE_SHARE:
            verdict["reasons"].append("too_much_silence")
        lufs, lra, tp = measure_loudness(path)
        verdict.update(lufs=lufs, lra=lra, true_peak=tp)
        if lra is not None and lra > AMB_MAX_LRA:
            verdict["reasons"].append("too_dynamic")
    if verdict["reasons"]:
        return verdict

    # CLAP: маржа положительного промпта над худшей ловушкой на КАЖДОМ окне
    negs = list(NEGATIVE_PROMPTS) + list(spec.get("extra_neg", ()))
    rows = clap_scores(win48, [spec["prompt"]] + negs)
    pos = [r[0] for r in rows]
    worst_neg = [max(r[1:]) for r in rows]
    worst_neg_name = [negs[max(range(len(negs)), key=lambda k: r[1 + k])] for r in rows]
    margin = min(p - n for p, n in zip(pos, worst_neg))
    verdict.update(clap_pos=round(sum(pos) / len(pos), 4), clap_margin=round(margin, 4),
                   clap_worst_neg=worst_neg_name[min(range(len(rows)), key=lambda k: pos[k] - worst_neg[k])])
    if sum(pos) / len(pos) < CLAP_MIN_POSITIVE:
        verdict["reasons"].append("clap_low_positive")
    if margin < CLAP_MIN_MARGIN:
        verdict["reasons"].append("clap_negative_wins")

    # AST: жёсткое вето по классам AudioSet (только атмосфера — у коротких
    # эффектов классы AudioSet малоинформативны)
    if is_amb:
        veto = dict(AST_VETO)
        veto.update(spec.get("ast_veto", {}))
        win16 = [decode_f32(path, s, min(CLAP_WINDOW_SEC, dur), 16000) for s in starts]
        probs = ast_probs([w for w in win16 if w.size > 1600], list(veto))
        worst = {lab: max(p.get(lab, 0.0) for p in probs) for lab in veto}
        verdict["ast"] = {k: round(v, 3) for k, v in worst.items()}
        for lab, thr in veto.items():
            if worst.get(lab, 0.0) > thr:
                verdict["reasons"].append(f"ast_{lab.lower()}")
    return verdict


def import_file(src, dst, kind, dur):
    """-> FLAC 48k stereo с объявленным пиком; атмосфера режется до
    AMBIENCE_MAX_SEC (начиная со 2-й секунды: в начале записей часто шорох
    рук/кнопки)."""
    peak_target = AMBIENCE_PEAK_DBFS if kind == "ambience" else SFX_PEAK_DBFS
    r = _run(["ffmpeg", "-v", "info", "-i", src, "-af", "volumedetect", "-f", "null", "-"])
    m = re.findall(r"max_volume:\s*(-?[\d.]+) dB", r.stderr)
    cur_peak = float(m[-1]) if m else 0.0
    gain = peak_target - cur_peak
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if kind == "ambience":
        cmd += ["-ss", "2" if dur > AMBIENCE_MAX_SEC + 4 else "0", "-t", f"{AMBIENCE_MAX_SEC:.1f}"]
    cmd += ["-i", src, "-af", f"volume={gain:.2f}dB", "-ar", "48000", "-ac", "2",
            "-c:a", "flac", "-compression_level", "8", dst]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = _run(cmd)
    return r.returncode == 0


def load_manifest():
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"items": {}}


def save_manifest(m):
    os.makedirs(LIBRARY_ROOT, exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    os.replace(tmp, MANIFEST_PATH)


def library_files(kind, name):
    d = os.path.join(LIBRARY_ROOT, kind, name)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".flac"))


def build_kind(kind, name, max_candidates=18, manifest=None, rejected_log=None):
    spec = LIBRARY_SPEC[kind][name]
    print(f"\n== {kind}/{name}: {spec['prompt']!r}")
    cands = gather_candidates(spec)
    # сначала подходящие по заявленной длительности — их дешевле проверять
    lo, hi = spec.get("min_sec", 0.0), spec.get("max_sec", 1e9)
    cands.sort(key=lambda c: (not (lo <= c["duration"] <= hi), -min(c["duration"], AMBIENCE_MAX_SEC)))
    cap = 420.0 if kind == "ambience" else hi * 1.25
    cands = [c for c in cands if lo * 0.8 <= c["duration"] <= min(cap, hi * 1.25)]
    blocked = [(c, title_blocked(name, c["title"])) for c in cands]
    for c, w in blocked:
        if w and rejected_log is not None:
            rejected_log.append({"id": c["id"], "title": c["title"], "kind": kind, "name": name,
                                 "reasons": [f"title:{w}"]})
    cands = [c for c, w in blocked if not w]
    cands.sort(key=lambda c: (-title_relevance(spec, c["title"]), -min(c["duration"], AMBIENCE_MAX_SEC)))
    cands = cands[:max_candidates]
    print(f"   кандидатов после фильтра длительности и названия: {len(cands)} "
          f"(отброшено по названию {sum(1 for _, w in blocked if w)})")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda c: download(c, "lq"), cands))
    scored = []
    for c, path in zip(cands, paths):
        if not path:
            continue
        v = evaluate(c, path, kind, name, spec)
        tag = "OK " if not v["reasons"] else "-- "
        print(f"   {tag}{c['title'][:44]:44s} {v.get('duration', 0):7.1f}с "
              f"margin={v.get('clap_margin', float('nan')):+.3f} "
              f"[{str(v.get('clap_worst_neg', ''))[:18]}] {','.join(v['reasons'])}")
        if rejected_log is not None and v["reasons"]:
            rejected_log.append(dict(v, kind=kind, name=name))
        if not v["reasons"]:
            scored.append((v["clap_margin"], c, path, v))
    scored.sort(key=lambda t: -t[0])
    kept = []
    for margin, c, path, v in scored[:spec.get("keep", 3)]:
        safe = re.sub(r"[^a-z0-9]+", "_", c["id"].lower()).strip("_")
        dst = os.path.join(LIBRARY_ROOT, kind, name, f"{safe}.flac")
        hq = download(c, "hq") or path
        if import_file(hq, dst, kind, v["duration"]):
            kept.append(dst)
            if manifest is not None:
                manifest["items"][os.path.relpath(dst, ROOT)] = {
                    "kind": kind, "name": name, "source": c["source"], "id": c["id"],
                    "title": c["title"], "creator": c["creator"], "license": c["license"],
                    "license_url": c["license_url"], "url": c["url"], "landing": c["landing"],
                    "query": c["query"], "scores": v, "imported_at": int(time.time()),
                }
    print(f"   принято {len(kept)} из {len(cands)}")
    return kept


def restore(manifest):
    """Воспроизвести библиотеку ПО МАНИФЕСТУ: скачать ровно те же файлы по
    тем же URL и импортировать с теми же нормировками — без поиска и без
    моделей. Нужно потому, что записи атмосферы (сотни МБ) в git не
    хранятся, а поисковая выдача со временем меняется: «собрать заново» дало
    бы другой набор, «восстановить» — тот же."""
    done = 0
    for rel, it in manifest.get("items", {}).items():
        dst = os.path.join(ROOT, rel)
        if os.path.exists(dst):
            continue
        item = {"url": it["url"], "title": it.get("title", "")}
        path = download(item, "hq")
        if path and import_file(path, dst, it["kind"], it["scores"].get("duration", 0.0)):
            done += 1
            print(f"   восстановлен {rel}")
    print(f"Восстановлено файлов: {done}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "report", "restore"])
    ap.add_argument("--kinds", default="", help="kind:name через запятую; пусто = всё")
    ap.add_argument("--max", type=int, default=30)
    args = ap.parse_args()
    manifest = load_manifest()
    if args.cmd == "restore":
        return restore(manifest)
    if args.cmd == "report":
        for rel, it in sorted(manifest["items"].items()):
            print(f"{rel:60s} {it['scores'].get('duration', 0):7.1f}с  margin {it['scores'].get('clap_margin'):+.3f}  {it['title'][:50]!r}")
        return 0
    wanted = [tuple(k.split(":", 1)) for k in args.kinds.split(",") if k.strip()] or \
             [(k, n) for k in LIBRARY_SPEC for n in LIBRARY_SPEC[k]]
    rejected = []
    for kind, name in wanted:
        build_kind(kind, name, args.max, manifest, rejected)
        save_manifest(manifest)
        with open(os.path.join(LIBRARY_ROOT, "rejected.json"), "w", encoding="utf-8") as f:
            json.dump(rejected, f, ensure_ascii=False, indent=1)
    print(f"\nГотово. Манифест: {MANIFEST_PATH}; отклонённые с причинами: assets/library/rejected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
