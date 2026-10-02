#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Библиотека НАСТОЯЩЕЙ музыки для любых ниш: Mixkit (лицензия Mixkit Stock
Music Free: коммерческое использование и YouTube разрешены, указывать
автора не нужно) + записи, уже отобранные в assets/library (music/medieval).

Повод — решение владельца 02.10: синтезированный гул «не профессиональный»;
нужна постоянная скачанная библиотека музыки без авторских прав, из которой
умный режиссёр выбирает трек под атмосферу ролика — в любой нише.

Как режиссёр без ушей понимает, что звучит: у каждого трека есть ТЕКСТОВАЯ
КАРТОЧКА из двух источников:
  * что сказал автор каталога: жанр и настроения (Mixkit: Film Score, mood/
    dark, mood/calm ...);
  * что слышит модель: AST (AudioSet) по трём окнам трека — инструменты
    (piano, strings, choir, drums, flute, bagpipes, synth...), характер
    (sad / tender / scary / exciting...), жанровые классы.
Карточка печатается режиссёру, он выбирает трек под главу. Модель слышит
трек сама, а не верит названию: трек с пением, речью, рэпом или поп/
танцевальным битом отбраковывается, даже если в каталоге он «cinematic».

Команды:
  python scripts/music_library.py build      # каталог -> отбор -> индекс
  python scripts/music_library.py restore    # скачать файлы по индексу
  python scripts/music_library.py cards      # напечатать карточки

Файлы (сотни МБ) в git не хранятся: assets/library/music_mixkit/ в
.gitignore, индекс assets/library/music_index.json — в git, restore качает
ровно те же файлы по нему (выдача каталога плывёт, «собрать заново» дало бы
другой набор).
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, "assets", "library")
MIXKIT_DIR = os.path.join(LIB, "music_mixkit")
INDEX_PATH = os.path.join(LIB, "music_index.json")
CACHE_DIR = os.path.join(ROOT, "temp_library", "mixkit")
UA = "Mozilla/5.0 (X11; Linux x86_64) PipelineMachineVideo music library"

MIXKIT_LICENSE = "Mixkit Stock Music Free License"
MIXKIT_LICENSE_URL = "https://mixkit.co/license/#musicFree"

# Разделы каталога: жанры фоновой музыки и настроения.
MIXKIT_CATEGORIES = (
    "ambient", "atmospheres", "cinematic", "film-score", "drone-music", "classical", "folk",
    "gregorian-chant", "medieval", "world", "piano", "meditation", "orchestral-hybrid",
    "traditional", "world-fusion", "electronica",
    "mood/dark", "mood/calm", "mood/epic", "mood/sad", "mood/mysterious", "mood/dramatic",
    "mood/emotional", "mood/hopeful", "mood/peaceful", "mood/melancholic", "mood/dreamy",
    "mood/uplifting", "mood/suspenseful",
)

# Жанры, годные под закадр документального фильма. Поп, рок, рэп, джаз,
# танцевальное, детское, кантри — нет: под голосом они спорят со словами.
BACKGROUND_GENRES = {
    "Film Score", "Ambient", "Atmospheres", "Drone Music", "Classical", "Folk", "Gregorian Chant",
    "Medieval", "Orchestral Hybrid", "Underscore", "New Age", "World Fusion", "Traditional",
    "Minimalism", "Film & Orchestral", "Percussion Trailer", "Trailer Music", "March & Military",
    "Religious", "Rhythmic Underscore", "Downtempo", "World", "Electronica", "Chillout",
}
# У широких жанров берётся только то, что каталог сам пометил фоновым
# настроением или традиционной музыкой: «World» — это и шаманский бубен, и
# латиноамериканский джаз.
LOOSE_GENRES = {"World", "Electronica", "Chillout"}
BACKGROUND_TAGS = {
    "mood/dark", "mood/calm", "mood/epic", "mood/sad", "mood/mysterious", "mood/dramatic",
    "mood/emotional", "mood/hopeful", "mood/peaceful", "mood/melancholic", "mood/dreamy",
    "mood/suspenseful", "traditional", "medieval", "meditation", "gregorian-chant", "drone-music",
    "atmospheres", "film-score", "cinematic", "ambient",
}
TITLE_BLOCK = ("christmas", "jazz", "latin", "salsa", "baila", "cuban", "party", "dance",
               "hip hop", "trap", "pop track", "kids", "children", "wedding", "birthday", "funky")
MIN_SEC, MAX_SEC = 60, 420

# Что слушает AST. Порядок — порядок в карточке.
INSTRUMENT_LABELS = ("Piano", "String section", "Bowed string instrument", "Violin, fiddle", "Cello",
                     "Orchestra", "Brass instrument", "Choir", "Flute", "Harp", "Bagpipes",
                     "Plucked string instrument", "Guitar", "Harpsichord", "Organ", "Synthesizer",
                     "Drum", "Percussion", "Drum kit", "Gong", "Bell", "Singing bowl")
CHARACTER_LABELS = ("Sad music", "Tender music", "Scary music", "Exciting music", "Angry music",
                    "Happy music", "Funny music")
STYLE_LABELS = ("Ambient music", "Soundtrack music", "Classical music", "Folk music",
                "Traditional music", "Middle Eastern music", "Music of Asia", "Music of Africa",
                "Christian music", "Electronic music", "New-age music", "Theme music")
VETO = {"Singing": 0.25, "Speech": 0.20, "Rapping": 0.10, "Pop music": 0.35,
        "Hip hop music": 0.30, "Rock music": 0.35, "Electronic dance music": 0.30,
        "House music": 0.30, "Dance music": 0.35, "Music for children": 0.30,
        "Christmas music": 0.25, "Funny music": 0.35, "Jingle (music)": 0.35,
        "Video game music": 0.40, "Music of Latin America": 0.35, "Swing music": 0.35}
AST_LABELS = ("Music",) + INSTRUMENT_LABELS + CHARACTER_LABELS + STYLE_LABELS + tuple(VETO)
MIN_MUSIC = 0.40
WINDOW_FRACS = (0.15, 0.5, 0.85)
WINDOW_SEC = 10.0


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _dur_sec(iso):
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def crawl_mixkit(categories=MIXKIT_CATEGORIES, max_pages=8, log=print):
    """{url: {name, genre, artist, sec, url, tags}} из JSON-LD страниц каталога."""
    items = {}
    for cat in categories:
        for page in range(1, max_pages + 1):
            url = f"https://mixkit.co/free-stock-music/{cat}/" + ("" if page == 1 else f"?page={page}")
            try:
                html = _get(url).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 — раздел без страницы, дальше
                break
            new = 0
            for blk in re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S):
                try:
                    data = json.loads(blk)
                except ValueError:
                    continue
                for g in data.get("@graph", []):
                    for it in g.get("itemListElement") or []:
                        if it.get("@type") != "MusicRecording" or not it.get("url"):
                            continue
                        if it.get("copyrightNotice") != MIXKIT_LICENSE:
                            continue  # чужая лицензия — не берём (fail-closed)
                        k = it["url"]
                        if k not in items:
                            items[k] = {"name": it.get("name", ""), "genre": it.get("genre", ""),
                                        "artist": it.get("byArtist", ""),
                                        "sec": _dur_sec(it.get("duration")), "url": k, "tags": []}
                            new += 1
                        if cat not in items[k]["tags"]:
                            items[k]["tags"].append(cat)
            if new == 0 and page > 1:
                break
        log(f"  каталог: {cat} -> всего {len(items)}")
    return items


def is_candidate(it):
    if it.get("genre") not in BACKGROUND_GENRES:
        return False
    if not (MIN_SEC <= int(it.get("sec") or 0) <= MAX_SEC):
        return False
    name = (it.get("name") or "").lower()
    if any(w in name for w in TITLE_BLOCK):
        return False
    if it["genre"] in LOOSE_GENRES and not (set(it.get("tags") or ()) & BACKGROUND_TAGS):
        return False
    return True


def track_id(it):
    m = re.search(r"/music/(\d+)/", it["url"])
    return f"mixkit_{m.group(1)}" if m else None


def _windows16k(path, dur):
    import numpy as np
    out = []
    for f in WINDOW_FRACS:
        start = max(0.0, dur * f - WINDOW_SEC / 2)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{WINDOW_SEC}",
                              "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                             capture_output=True).stdout
        a = np.frombuffer(raw, dtype=np.float32)
        if len(a) > 16000:
            out.append(a)
    return out


def media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                        path], capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def listen(path):
    """Что слышит AST: средние вероятности по окнам (вето — по максимуму)."""
    sys.path.insert(0, HERE)
    import sound_library
    dur = media_duration(path)
    if not dur:
        return None
    wins = _windows16k(path, dur)
    if not wins:
        return None
    rows = sound_library.ast_probs(wins, AST_LABELS)
    mean = {lab: sum(r.get(lab, 0.0) for r in rows) / len(rows) for lab in AST_LABELS}
    peak = {lab: max(r.get(lab, 0.0) for r in rows) for lab in AST_LABELS}
    return {"dur": round(dur, 2), "mean": mean, "peak": peak, "rhythm": rhythm(path, dur)}


def judge(heard):
    """None — годен; иначе причина отказа."""
    if heard is None:
        return "not_measured"
    if min(heard["mean"]["Music"], heard["peak"]["Music"]) < MIN_MUSIC:
        return "not_music"
    for lab, thr in VETO.items():
        if heard["peak"].get(lab, 0.0) > thr:
            return f"veto:{lab}"
    return None


def _top(mean, labels, thr, k):
    got = sorted(((mean.get(l, 0.0), l) for l in labels if mean.get(l, 0.0) >= thr), reverse=True)
    return [l.split(",")[0].replace(" music", "").lower() for _p, l in got[:k]]


# Стиль и культура — CLAP (текст<->звук) по описаниям ниже. Сырой косинус
# между разными текстами несопоставим (урок негативного вето по картинкам),
# поэтому в карточку идёт z-оценка по библиотеке: «этот трек похож на
# описание сильнее, чем остальные треки».
STYLE_PROMPTS = {
    "medieval European early music": "medieval European music played on lute, recorder, hurdy-gurdy and fiddle",
    "Celtic folk": "Celtic folk music with fiddle and tin whistle",
    "sacred choir or chant": "sacred church music, Gregorian chant, organ",
    "Middle Eastern": "Middle Eastern music with oud and darbuka",
    "Indian": "Indian classical music with sitar, tabla and bansuri flute",
    "East Asian": "East Asian music with guzheng, koto or erhu",
    "Andean": "Andean music with pan flute and charango",
    "African": "African music with djembe and kora",
    "primal tribal ritual": "primal tribal drums and bone flutes, ancient ritual",
    "epic orchestral": "epic orchestral film score with big drums and brass",
    "dark tension": "dark tense suspenseful film score",
    "horror": "horror movie music, creepy and scary",
    "sad piano or strings": "sad emotional music with piano and strings",
    "calm ambient pads": "calm ambient pads, relaxing atmosphere",
    "futuristic electronic": "futuristic electronic synthesizer music, science fiction",
    "upbeat corporate": "upbeat corporate background music, happy and positive",
    "romantic": "romantic love song music",
    "Christmas holiday": "Christmas holiday music with sleigh bells",
    "playful children's": "playful children's music, cartoon",
    "jazz or latin": "jazz or latin music",
    "country western": "country western music with acoustic guitar and harmonica",
}
STYLE_Z_MIN = 1.5
# Стили, которым не место под закадром документального фильма ни в какой нише.
STYLE_EXCLUDE = {"Christmas holiday": 1.5, "playful children's": 1.5, "upbeat corporate": 2.0,
                 "romantic": 2.0, "jazz or latin": 2.0}
TITLE_EXCLUDE = ("holiday", "seasonal", "santa", "happy", "party", "love", "baby")


def clap_style(path, dur):
    """{описание стиля: косинус} — среднее по трём окнам трека."""
    import numpy as np
    sys.path.insert(0, HERE)
    import sound_library
    wins = []
    for f in WINDOW_FRACS:
        start = max(0.0, dur * f - WINDOW_SEC / 2)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{WINDOW_SEC}",
                              "-i", path, "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                             capture_output=True).stdout
        a = np.frombuffer(raw, dtype=np.float32)
        if len(a) > 48000:
            wins.append(a)
    if not wins:
        return None
    names = list(STYLE_PROMPTS)
    rows = sound_library.clap_scores(wins, [STYLE_PROMPTS[n] for n in names])
    return {n: round(sum(r[k] for r in rows) / len(rows), 4) for k, n in enumerate(names)}


def style_ref(tracks):
    """{стиль: [среднее, разброс]} по библиотеке — шкала z, замороженная в индексе."""
    import numpy as np
    have = [t for t in tracks if t.get("clap")]
    if len(have) < 5:
        return {}
    ref = {}
    for n in STYLE_PROMPTS:
        v = np.array([t["clap"].get(n, 0.0) for t in have])
        ref[n] = [round(float(v.mean()), 5), round(float(v.std()) or 1.0, 5)]
    return ref


def style_z(tracks, ref):
    """{id: {стиль: z}} по замороженной шкале."""
    out = {}
    for t in tracks:
        if not t.get("clap") or not ref:
            continue
        out[t["id"]] = {n: round((t["clap"].get(n, 0.0) - mu) / sd, 2) for n, (mu, sd) in ref.items()}
    return out


PERC_DRUMS = 0.09     # доля ударной энергии (HPSS): барабаны 0.13-0.28, эмбиент/фортепиано 0.02-0.03
PULSE_STEADY = 0.40   # автокорреляция ударной огибающей: ровный ритм (марш, перебор)


def _median_filter(m, axis, size):
    import numpy as np
    pad = [(0, 0), (0, 0)]
    pad[axis] = (size // 2, size // 2)
    padded = np.pad(m, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, size, axis=axis), axis=-1)


def rhythm(path, dur):
    """(доля ударной энергии, сила ровного пульса) по 60 с из середины трека.

    AST на барабаны почти не реагирует (у «Epical Drums» Drum < 0.1), поэтому
    ритм меряется по сигналу: разделение на тональную и ударную части
    медианными фильтрами по времени и частоте (HPSS), затем автокорреляция
    ударной огибающей в окне 50-200 ударов в минуту."""
    import numpy as np
    sr, n, h = 11025, 1024, 256
    start = max(0.0, dur * 0.5 - 30.0)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.1f}", "-t", "60", "-i", path,
                          "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    if len(x) < n * 8:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::h] * np.hanning(n)
    m = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    harm = _median_filter(m, 0, 17)
    perc = _median_filter(m, 1, 17)
    mask = perc / (harm + perc + 1e-12) > 0.6
    share = float((m * mask).sum() / (m.sum() + 1e-12))
    env = (m * mask).sum(1)
    env = env - env.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-12)
    fps = sr / h
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 50)
    pulse = float(ac[lo:hi].max())
    return round(share, 3), round(pulse, 3)


def energy_words(rh):
    if rh is None:
        return "rhythm unknown"
    share, pulse = rh
    if share >= PERC_DRUMS:
        return "drums or percussion rhythm"
    if pulse >= PULSE_STEADY:
        return "steady rhythmic pattern"
    return "no beat"


def card(it, heard, z=None):
    """Текстовая карточка трека для режиссёра."""
    moods = [t.split("/", 1)[1] for t in it.get("tags", []) if t.startswith("mood/")]
    m = heard["mean"]
    instruments = _top(m, INSTRUMENT_LABELS, 0.08, 4)
    character = _top(m, CHARACTER_LABELS, 0.12, 2)
    style = _top(m, STYLE_LABELS, 0.15, 2)
    energy = energy_words(heard.get("rhythm"))
    parts = [f"\"{it.get('name', '')}\" {it.get('genre', '').lower()}"]
    if z:
        like = [n for n, v in sorted(z.items(), key=lambda kv: -kv[1]) if v >= STYLE_Z_MIN][:3]
        if like:
            parts.append("sounds like: " + ", ".join(like))
    if moods:
        parts.append("moods: " + ", ".join(moods[:4]))
    if instruments:
        parts.append("sounds: " + ", ".join(instruments))
    if character:
        parts.append("character: " + ", ".join(character))
    if style:
        parts.append("style: " + ", ".join(style))
    parts.append(energy)
    return "; ".join(parts)


def load_index():
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"tracks": []}


def save_index(index):
    os.makedirs(LIB, exist_ok=True)
    index["tracks"].sort(key=lambda t: t["id"])
    with open(INDEX_PATH + ".tmp", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    os.replace(INDEX_PATH + ".tmp", INDEX_PATH)


def track_path(t):
    return os.path.join(ROOT, t["path"])


def build(limit=0, log=print):
    """Каталог -> кандидаты -> скачивание -> AST -> карточки -> индекс.
    Уже принятые и уже отклонённые треки повторно не слушаются."""
    index = load_index()
    index_local(index, log=log)
    known = {t["id"] for t in index["tracks"]}
    rejected = index.setdefault("rejected", {})
    cat = crawl_mixkit(log=log)
    cands = [it for it in cat.values() if is_candidate(it)]
    log(f"каталог {len(cat)}, кандидатов {len(cands)}")
    os.makedirs(MIXKIT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    done = 0
    for it in cands:
        tid = track_id(it)
        if not tid or tid in known or tid in rejected:
            continue
        if limit and done >= limit:
            break
        done += 1
        tmp = os.path.join(CACHE_DIR, tid + ".mp3")
        try:
            if not os.path.exists(tmp):
                data = _get(it["url"], timeout=120)
                with open(tmp + ".part", "wb") as f:
                    f.write(data)
                os.replace(tmp + ".part", tmp)
        except Exception as e:  # noqa: BLE001
            log(f"  {tid} не скачался: {type(e).__name__}")
            continue
        heard = listen(tmp)
        why = judge(heard)
        if why:
            rejected[tid] = {"name": it["name"], "genre": it["genre"], "reason": why}
            log(f"  - {it['name'][:40]:40s} {why}")
            os.remove(tmp)
            continue
        dst = os.path.join(MIXKIT_DIR, tid + ".mp3")
        os.replace(tmp, dst)
        index["tracks"].append({
            "id": tid, "title": it["name"], "artist": it["artist"], "genre": it["genre"],
            "tags": it["tags"], "dur": heard["dur"], "card": card(it, heard),
            "ast": {k: round(v, 3) for k, v in heard["mean"].items() if v >= 0.05},
            "rhythm": heard["rhythm"], "source": "mixkit", "url": it["url"], "licence": MIXKIT_LICENSE,
            "licence_url": MIXKIT_LICENSE_URL,
            "path": os.path.relpath(dst, ROOT)})
        known.add(tid)
        log(f"  + {it['name'][:40]:40s} {index['tracks'][-1]['card'][:110]}")
        if done % 10 == 0:
            save_index(index)
    save_index(index)
    recard(log=log)      # стиль (CLAP) и карточки новых треков
    index = load_index()
    log(f"в библиотеке {len(index['tracks'])} треков, отклонено {len(index.get('rejected', {}))}")
    return index


def index_local(index, log=print):
    """Музыка, уже отобранная в assets/library/music/<вид>/ (Freesound CC0,
    Wikimedia PD): та же карточка из того, что слышит AST. Жанр — имя вида."""
    sys.path.insert(0, HERE)
    import sound_library
    manifest = sound_library.load_manifest().get("items", {})
    known = {t["id"] for t in index["tracks"]}
    root = os.path.join(LIB, "music")
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for path in sound_library.library_files("music", name):
            rel = os.path.relpath(path, ROOT)
            tid = "lib_" + os.path.splitext(os.path.basename(path))[0]
            if tid in known:
                continue
            meta = manifest.get(rel, {})
            heard = listen(path)
            if heard is None or judge(heard):
                continue
            it = {"name": meta.get("title", tid), "genre": name.capitalize(), "tags": [name]}
            index["tracks"].append({
                "id": tid, "title": it["name"], "artist": meta.get("creator", ""), "genre": it["genre"],
                "tags": it["tags"], "dur": heard["dur"], "card": card(it, heard) + f"; {name} period music",
                "ast": {k: round(v, 3) for k, v in heard["mean"].items() if v >= 0.05},
                "rhythm": heard["rhythm"], "source": meta.get("source", "library"), "url": meta.get("url", ""),
                "licence": meta.get("license", ""), "path": rel, "in_git": True})
            log(f"  + {it['name'][:40]:40s} {index['tracks'][-1]['card'][:110]}")


def recard(log=print):
    """Ритм, стиль (CLAP) и карточки уже принятых треков (без нового AST).
    Треки, чей стиль по замеру — рождественский, детский, корпоративный,
    романтический или джаз/латино (или такое название), уходят в rejected:
    под закадром документального фильма им не место ни в какой нише."""
    index = load_index()
    for t in index["tracks"]:
        p = track_path(t)
        if not os.path.exists(p):
            continue
        if t.get("rhythm") is None:
            t["rhythm"] = rhythm(p, t["dur"])
        if not t.get("clap") or set(t["clap"]) != set(STYLE_PROMPTS):
            t["clap"] = clap_style(p, t["dur"])
    ref = index.get("style_ref")
    if not ref:
        ref = index["style_ref"] = style_ref(index["tracks"])
    zs = style_z(index["tracks"], ref)
    keep = []
    rejected = index.setdefault("rejected", {})
    for t in index["tracks"]:
        z = zs.get(t["id"], {})
        bad = [n for n, thr in STYLE_EXCLUDE.items() if z.get(n, 0.0) >= thr]
        title = (t.get("title") or "").lower()
        bad += [f"title:{w}" for w in TITLE_EXCLUDE if re.search(rf"\b{w}", title)]
        # Отбор по стилю — один раз на трек: уже прошедший трек повторно не
        # судится (иначе каждое удаление сдвигало бы оценку остальных).
        if bad and not t.get("approved") and not t.get("style_checked"):
            rejected[t["id"]] = {"name": t["title"], "genre": t["genre"], "reason": "style:" + bad[0]}
            log(f"  - {t['title'][:40]:40s} {bad[0]}")
            p = track_path(t)
            if os.path.exists(p) and not t.get("in_git"):
                os.remove(p)
            continue
        heard = {"mean": t.get("ast", {}), "rhythm": t.get("rhythm")}
        it = {"name": t["title"], "genre": t["genre"], "tags": t.get("tags", [])}
        c = card(it, heard, z)
        if t.get("source") != "mixkit" and t.get("tags"):
            c += f"; {t['tags'][0]} period music"
        t["card"] = c
        t["style_z"] = z
        t["style_checked"] = True
        keep.append(t)
    index["tracks"] = keep
    save_index(index)
    log(f"карточки пересчитаны: {len(keep)}")


def restore(log=print):
    index = load_index()
    ok = 0
    for t in index["tracks"]:
        p = track_path(t)
        if os.path.exists(p) or t.get("in_git"):
            ok += 1
            continue
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            data = _get(t["url"], timeout=120)
            with open(p + ".part", "wb") as f:
                f.write(data)
            os.replace(p + ".part", p)
            ok += 1
        except Exception as e:  # noqa: BLE001
            log(f"  {t['id']} не скачался: {type(e).__name__}")
    log(f"на диске {ok} из {len(index['tracks'])}")


def available_tracks():
    """Треки индекса, файлы которых есть на диске."""
    return [t for t in load_index()["tracks"] if os.path.exists(track_path(t))]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "restore", "cards", "recard"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(limit=args.limit)
    elif args.cmd == "recard":
        recard()
    elif args.cmd == "restore":
        restore()
    else:
        for t in available_tracks():
            print(f"{t['id']:14s} {t['dur']:6.0f}s {t['title'][:30]:30s} {t['card']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
