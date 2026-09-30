#!/usr/bin/env python3
"""Когда на экране меняется кадр: старт каждой фразы в итоговом аудио.

Источник истины — посимвольный alignment озвучки (media_plan/alignment/NN.csv,
смещения секций в media_plan/section_offsets.json — их пишет lumean_tts.py).
Буквы фразы ищутся в потоке озвученных букв по порядку, время первой буквы —
старт кадра. Если fix_pauses.py резал тишину, время пересчитывается по его
карте media_plan/pause_cuts.json (вырезанное вычитается, вставленное
прибавляется).

Нет alignment — оценка по числу букв фразы и её паузе, растянутая на длину
аудио. Это ОЦЕНКА, и в отчёте она так и называется (source="estimate").

Фраза, которую не нашли в alignment, получает старт интерполяцией между
найденными соседями, а не нулём."""
import csv
import glob
import json
import os
import re

TAG_RE = re.compile(r"\[[^\]]*\]")


def norm_letters(s):
    s = TAG_RE.sub(" ", s.lower().replace("ё", "е"))
    return re.sub(r"[^0-9a-zа-я]", "", s)


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def raw_to_real(t, cuts, inserts):
    removed = 0.0
    for a, b in cuts:
        if a >= t:
            break
        removed += min(t, b) - a
    added = sum(float(sec) for pos, sec in inserts if pos < t)
    return t - removed + added


def letter_stream(video_dir):
    """[(буква, глобальное_сырое_время)] по всем секциям, или [] если нет alignment."""
    plan = os.path.join(video_dir, "media_plan")
    offsets = list(load_json(os.path.join(plan, "section_offsets.json"), {}).values())
    stream = []
    for path in sorted(glob.glob(os.path.join(plan, "alignment", "*.csv"))):
        idx = int(os.path.splitext(os.path.basename(path))[0])
        off = offsets[idx] if idx < len(offsets) else None
        if off is None:
            if idx > 0:
                return []    # без смещения секция после первой не имеет глобального времени
            off = 0.0
        with open(path, encoding="utf-8") as f:
            rows = [(r["char"], float(r["start"])) for r in csv.DictReader(f)]
        text = "".join(c for c, _ in rows)
        skip = set()
        for m in TAG_RE.finditer(text):
            skip.update(range(m.start(), m.end()))
        for j, (c, s) in enumerate(rows):
            if j in skip:
                continue
            n = norm_letters(c)
            if n:
                stream.append((n, s + off))
    return stream


def onsets_from_alignment(blocks, stream):
    letters = "".join(c for c, _ in stream)
    times = [t for _, t in stream]
    cursor, out = 0, []
    for b in blocks:
        want = norm_letters(b["text"])
        probe = want[:14]
        pos = letters.find(probe, cursor) if probe else -1
        if pos < 0 or pos - cursor > max(400, 3 * len(want)):
            out.append(None)         # не нашли рядом — лучше интерполяция, чем прыжок в чужую фразу
            continue
        out.append(times[pos])
        cursor = pos + len(want)
    return out


def estimate_onsets(blocks, total):
    weights = [len(norm_letters(b["text"])) + 14 * b.get("pause_after", 0.0) for b in blocks]
    scale = total / max(1e-6, sum(weights))
    out, t = [], 0.0
    for w in weights:
        out.append(t)
        t += w * scale
    return out


def fill_gaps(onsets, total):
    known = [(i, t) for i, t in enumerate(onsets) if t is not None]
    if not known:
        return None
    res = list(onsets)
    for i, t in enumerate(res):
        if t is not None:
            continue
        prev = max((k for k in known if k[0] < i), default=(-1, 0.0), key=lambda k: k[0])
        nxt = min((k for k in known if k[0] > i), default=(len(res), total), key=lambda k: k[0])
        res[i] = prev[1] + (nxt[1] - prev[1]) * (i - prev[0]) / (nxt[0] - prev[0])
    return res


def clamp_starts(starts, total):
    """Старты монотонны, каждый кадр не короче MIN_DUR, и сумма длительностей
    РОВНО длина аудио: проход вперёд раздвигает слипшиеся старты, проход
    назад не даёт хвосту вылезти за конец. Аудио короче n*MIN_DUR — поровну."""
    n = len(starts)
    if n * MIN_DUR >= total:
        return [total * i / n for i in range(n)]
    s = list(starts)
    s[0] = 0.0
    for i in range(1, n):
        s[i] = max(s[i], s[i - 1] + MIN_DUR)
    limit = total
    for i in range(n - 1, 0, -1):
        s[i] = min(s[i], limit - MIN_DUR)
        limit = s[i]
    return s


def frame_durations(video_dir, blocks, audio_total, fixed_audio=False):
    """(старты, длительности, отчёт). Первый кадр всегда с 0, последний — до
    конца аудио; длительность не короче MIN_DUR (слишком короткий кадр не
    читается, его время уходит соседу)."""
    stream = letter_stream(video_dir)
    source = "estimate"
    onsets = None
    found = 0
    if stream:
        raw = onsets_from_alignment(blocks, stream)
        found = sum(t is not None for t in raw)
        if fixed_audio:
            pc = load_json(os.path.join(video_dir, "media_plan", "pause_cuts.json"), {})
            cuts, inserts = pc.get("cuts") or [], pc.get("pause_inserts") or []
            raw = [None if t is None else raw_to_real(t, cuts, inserts) for t in raw]
        onsets = fill_gaps(raw, audio_total)
        if onsets is not None:
            source = "alignment"
    if onsets is None:
        onsets = estimate_onsets(blocks, audio_total)
    starts = clamp_starts([0.0] + [max(0.0, min(audio_total, t)) for t in onsets[1:]], audio_total)
    ends = starts[1:] + [audio_total]
    durs = [e - s for s, e in zip(starts, ends)]
    report = {"source": source, "found_in_alignment": found, "blocks": len(blocks)}
    return starts, durs, report


MIN_DUR = 0.8
