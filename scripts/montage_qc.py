#!/usr/bin/env python3
"""Приёмка монтажа по ПИКСЕЛЯМ готового файла — одна точка (docs/MONTAGE_SPEC.md, §0, §5.4).

  python scripts/montage_qc.py videos/NN_tema [final.mp4]

Источник истины — сам final.mp4, не план: план говорит, что задумано, файл — что
вышло (чанкование, потерянный клип, не та кривая — всё это план не увидит).
Резы берутся ТЕМ ЖЕ детектором, что в verify_timing (вторая копия разошлась бы).

Что меряется (каждая метрика — measured | no_signal; no_signal НИКОГДА не считается
успехом — класс ловушки, из-за которого у золотого набора стоит канарейка):
  * длины планов: максимум (≤ MAX_VIEW_SEC + кадр — структурный предохранитель §5.1),
    вариация (§2.2), зона хука (§5.2);
  * скорость камеры внутри плана: доля «замороженных» окон и max/mean (§1.1), знак
    направления внутри одной картинки (§1.3);
  * склейка: скачок средней яркости и уход центра внимания (§1.7);
  * резкость внутри плана: разброс (§1.6);
  * речь: онсет голоса в хуке (§5.2), резы в паузах (§2.1, из verify_timing).

Уровни (§5.4): block — ролик не публикуется (код 1); warn — публикуется, число в
отчёте (код 2); info — строка. Порог со статусом hypothesis блокировать не может
по построению: перевод в calibrated — правка ОДНОЙ строки в THRESHOLDS и запись
замера в docs/MONTAGE_SPEC.md.

Цена: один проход декодирования в 640x360 (скорость камеры нужна субпиксельная:
4% зума за 3 с — это 0.5 px на краю кадра за четверть секунды) плюс проход
verify_timing в 64x36. Время QC печатается в отчёт — если оно догонит рендер,
это видно, а не прячется."""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_timing as vt  # noqa: E402

SCAN_W, SCAN_H = 640, 360
ZOOM_STEP_FRAMES = 6            # окно оценки зума: 0.25 с при 24 к/с
ZOOM_CANDIDATES = np.exp(np.linspace(np.log(0.985), np.log(1.015), 31))   # ±1.5% за окно
HOOK_ZONE_SEC = 3.0
EXIT_OK, EXIT_BLOCK, EXIT_WARN = 0, 1, 2

# имя: (порог, сравнение, уровень, статус). Сравнение: "<=" — значение не выше порога.
THRESHOLDS = {
    "plan_max_sec":           (3.5 + 1/24, "<=", "block", "calibrated"),   # shots.MAX_VIEW_SEC + кадр
    "plan_len_cv":            (0.25, ">=", "warn", "hypothesis"),
    "plan_near_equal_share":  (0.30, "<=", "warn", "hypothesis"),
    "frozen_share":           (0.05, "<=", "warn", "hypothesis"),         # §1.1: ease_io давал 16% по пикселям, линейный ход — 0; запас на шум
    "speed_max_over_mean":    (1.25, "<=", "warn", "hypothesis"),       # на идеально линейном синтетическом материале оценка даёт 1.13 (шум)
    "direction_flips_excess": (0,    "<=", "warn", "hypothesis"),
    "cut_luma_jump":          (12.0, "<=", "warn", "hypothesis"),
    "cut_attention_shift":    (0.25, "<=", "warn", "hypothesis"),
    "sharpness_cv":           (0.10, "<=", "warn", "hypothesis"),
    "hook_first_cut_sec":     (2.0,  "<=", "warn", "hypothesis"),
    "hook_plan_len_sec":      ((1.2, 2.0), "range", "warn", "hypothesis"),
    "hook_voice_onset_sec":   (0.3,  "<=", "warn", "hypothesis"),
    "cuts_in_silence_share":  (vt.MIN_CUTS_IN_SILENCE, ">=", "block", "calibrated"),   # §2.1, порог verify_timing
}


def _passes(name, value):
    thr, op, _, _ = THRESHOLDS[name]
    if value is None:
        return None
    if op == "<=":
        return value <= thr
    if op == ">=":
        return value >= thr
    lo, hi = thr
    return lo <= value <= hi


def _check(name, value, extra=None):
    thr, op, level, status = THRESHOLDS[name]
    ok = _passes(name, value)
    rec = {"value": value, "threshold": thr, "op": op, "level": level, "status": status,
           "state": "no_signal" if value is None else ("ok" if ok else "violation")}
    if extra:
        rec.update(extra)
    return rec


# ----------------------------------------------------------------------------- декод
def scan(video_path, w=SCAN_W, h=SCAN_H):
    """Один проход: по каждому кадру — средняя яркость, центр масс чернил, резкость;
    по каждому окну ZOOM_STEP_FRAMES — оценка зума (перебор масштаба, параболическое
    уточнение). Кадры в памяти не копятся (кольцо на ZOOM_STEP_FRAMES+1)."""
    import cv2
    fps = vt.video_fps(video_path) or 24.0
    cmd = [vt.FFMPEG, "-v", "error", "-i", video_path, "-an",
           "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    nbytes = w * h
    ring, luma, cx, sharp, zoom = [], [], [], [], []
    cy0, cx0 = h / 2, w / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            f = np.frombuffer(buf, np.uint8).reshape(h, w)
            luma.append(float(f.mean()))
            paper = float(np.percentile(f, 90))
            ink = (f < paper - 40)
            cx.append(float(xx[ink].mean() / w) if ink.sum() > 50 else None)
            sharp.append(float(cv2.Laplacian(f, cv2.CV_32F).var()))
            ring.append(f)
            if len(ring) == ZOOM_STEP_FRAMES + 1:          # окна без перекрытия: zoom[j] — кадры [j·k, (j+1)·k]
                zoom.append(_zoom_between(ring[0], ring[-1], cx0, cy0))
                ring = [f]
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait(timeout=vt._timeout_for(video_path))
    return dict(fps=fps, n=len(luma), luma=luma, cx=cx, sharp=sharp, zoom=zoom)


def _zoom_between(a, b, cx0, cy0):
    """Масштаб b относительно a (перебор кандидатов по центральной части кадра,
    уточнение параболой по трём лучшим). None — кадры не похожи (склейка внутри окна)."""
    import cv2
    h, w = a.shape
    m = int(h * 0.1)
    crop = (slice(m, h - m), slice(int(w * 0.1), w - int(w * 0.1)))
    af = a.astype(np.float32)
    errs = []
    for s in ZOOM_CANDIDATES:
        M = np.float32([[s, 0, cx0 * (1 - s)], [0, s, cy0 * (1 - s)]])
        bs = cv2.warpAffine(af, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        errs.append(float(np.abs(bs[crop] - b[crop].astype(np.float32)).mean()))
    i = int(np.argmin(errs))
    base = float(np.abs(af[crop] - b[crop].astype(np.float32)).mean())
    if errs[i] > 0.6 * base and base > 8:        # лучший масштаб почти не объясняет разницу — это не зум
        return None
    if 0 < i < len(errs) - 1:
        y0, y1, y2 = errs[i - 1], errs[i], errs[i + 1]
        d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2)) if (y0 - 2 * y1 + y2) != 0 else 0.0
        lz = np.log(ZOOM_CANDIDATES[i]) + d * (np.log(ZOOM_CANDIDATES[1]) - np.log(ZOOM_CANDIDATES[0]))
        return float(np.exp(lz))
    return float(ZOOM_CANDIDATES[i])


# ----------------------------------------------------------------------------- метрики
def plan_bounds(cuts, total, picture_starts):
    """Границы планов: резы по пикселям + начала картинок (смена картинки — тоже склейка)."""
    pts = sorted(set([0.0, total] + [round(c, 3) for c in cuts] + [round(p, 3) for p in picture_starts if 0 < p < total]))
    return [(a, b) for a, b in zip(pts, pts[1:]) if b - a > 1 / 48]


def speed_metrics(sc, plans):
    """Скорость камеры по окнам внутри каждого плана (окна, задевающие склейку, выброшены)."""
    fps, k = sc["fps"], ZOOM_STEP_FRAMES
    frozen, ratio, per_plan = [], [], []
    for (a, b) in plans:
        j0, j1 = int(np.ceil(a * fps / k)), int(np.floor(b * fps / k))   # окна целиком внутри плана
        v = [abs(np.log(z)) for z in sc["zoom"][j0:j1] if z is not None] if j1 > j0 else []
        if len(v) < 4:
            per_plan.append(None)
            continue
        v = np.array(v); m = v.mean()
        if m < 1e-5:                              # план без движения вообще — отдельный сигнал
            per_plan.append(dict(frozen=1.0, ratio=None, static=True))
            frozen.append(1.0)
            continue
        fr = float((v < 0.2 * m).mean()); rt = float(v.max() / m)
        per_plan.append(dict(frozen=fr, ratio=rt, static=False)); frozen.append(fr); ratio.append(rt)
    sig = [p for p in per_plan if p]
    return dict(frozen_share=float(np.mean(frozen)) if frozen else None,
                speed_max_over_mean=float(np.median(ratio)) if ratio else None,
                plans_measured=len(sig), plans_total=len(plans), per_plan=per_plan)


def direction_metrics(sc, plans, picture_starts, total):
    """Смена знака зума внутри ОДНОЙ картинки — нарушение §1.3."""
    fps, k = sc["fps"], ZOOM_STEP_FRAMES
    if not picture_starts:                        # без карты картинок смена знака между картинками законна — сигнала нет
        return dict(direction_flips_excess=None, pictures_measured=0)
    pics = sorted(set([0.0] + [p for p in picture_starts if 0 < p < total])) + [total]
    flips = 0; measured = 0
    for pa, pb in zip(pics, pics[1:]):
        signs = []
        for (a, b) in plans:
            if a < pa or b > pb + 1e-6:
                continue
            j0, j1 = int(np.ceil(a * fps / k)), int(np.floor(b * fps / k))
            zs = [np.log(z) for z in sc["zoom"][j0:j1] if z is not None]
            if len(zs) >= 4 and abs(np.sum(zs)) > 1e-4:
                signs.append(np.sign(np.sum(zs)))
        if len(signs) >= 2:
            measured += 1
            flips += int(np.sum(np.array(signs[1:]) != np.array(signs[:-1])))
    return dict(direction_flips_excess=flips if measured else None, pictures_measured=measured)


def cut_metrics(sc, cuts):
    fps = sc["fps"]
    lj, sh = [], []
    for c in cuts:
        i = int(round(c * fps))
        if 1 <= i < sc["n"]:
            lj.append(abs(sc["luma"][i] - sc["luma"][i - 1]))
            a, b = sc["cx"][i - 1], sc["cx"][i]
            if a is not None and b is not None:
                sh.append(abs(a - b))
    return dict(cut_luma_jump=float(max(lj)) if lj else None, cut_luma_jump_median=float(np.median(lj)) if lj else None,
                cut_attention_shift=float(max(sh)) if sh else None, cuts=len(cuts))


def sharpness_metrics(sc, plans):
    fps = sc["fps"]; cvs = []
    for (a, b) in plans:
        s = np.array(sc["sharp"][int(np.ceil(a * fps)):int(np.floor(b * fps))])
        if len(s) >= 6 and s.mean() > 1e-6:
            cvs.append(float(s.std() / s.mean()))
    return dict(sharpness_cv=float(np.median(cvs)) if cvs else None)


def length_metrics(plans):
    L = np.array([b - a for a, b in plans])
    if len(L) == 0:
        return dict(plan_max_sec=None, plan_len_cv=None, plan_near_equal_share=None)
    near = float(np.mean(np.abs(np.diff(L)) < 0.2)) if len(L) > 1 else None
    return dict(plan_max_sec=float(L.max()), plan_len_cv=float(L.std() / L.mean()) if len(L) > 2 else None,
                plan_near_equal_share=near, plans=len(L), plan_mean_sec=float(L.mean()))


def hook_metrics(plans, cuts, silences):
    first_cut = min(cuts) if cuts else None
    hook_plans = [b - a for a, b in plans if a < HOOK_ZONE_SEC and b - a > 0.2]
    onset = None
    if silences is not None:
        onset = 0.0
        for s0, s1 in sorted(silences):
            if s0 <= 0.05:
                onset = s1
            break
    worst = None
    if hook_plans:
        lo, hi = THRESHOLDS["hook_plan_len_sec"][0]
        worst = max(hook_plans, key=lambda x: max(lo - x, x - hi, 0))
    return dict(hook_first_cut_sec=first_cut, hook_plan_len_sec=worst, hook_voice_onset_sec=onset,
                hook_plans=len(hook_plans))


# ----------------------------------------------------------------------------- отчёт
def build(video_dir, video_path=None):
    t0 = time.time()
    video_path = video_path or os.path.join(video_dir, "final.mp4")
    rep = {"schema_version": 1, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "video": os.path.basename(video_path), "spec": "docs/MONTAGE_SPEC.md",
           "limits": ["скорость камеры — по окнам 0.25 с, перебор масштаба ±1.5%: диссолвы и окна со склейкой выброшены",
                      "no_signal — отсутствие измерения, не успех",
                      "пороги со статусом hypothesis не блокируют до калибровки на живых роликах"]}
    if not os.path.exists(video_path):
        rep["verdict"] = "no_video"; rep["checks"] = {}
        return rep, EXIT_BLOCK
    total = vt.media_duration(video_path) or 0.0
    # склейки — по доле изменившихся пикселей, не по медиане разницы: на рисунке на бумаге медиана
    # равна нулю и на настоящей склейке (эп.01: 5 из 11 склеек пропущены) — см. frame_change_share_curve
    fps_d, diffs = vt.frame_change_share_curve(video_path)
    cuts, _ = vt.cuts_from_curve(fps_d, diffs, min_abs=vt.SHARE_MIN_ABS)
    silences = vt.detect_silences(video_path)
    pic = []
    pt = os.path.join(video_dir, "media_plan", "phrase_timeline.json")
    if os.path.exists(pt):
        try:
            d = json.load(open(pt, encoding="utf-8"))
            if d.get("locked"):
                pic = [b["speech_onset_sec"] for b in d.get("blocks", []) if b.get("speech_onset_sec") is not None]
        except (OSError, ValueError):
            pic = []
    sc = scan(video_path)
    plans = plan_bounds(cuts, total, pic)
    m = {}
    m.update(length_metrics(plans)); m.update(speed_metrics(sc, plans)); m.update(direction_metrics(sc, plans, pic, total))
    m.update(cut_metrics(sc, cuts)); m.update(sharpness_metrics(sc, plans)); m.update(hook_metrics(plans, cuts, silences))
    if cuts and silences is not None:
        m["cuts_in_silence_share"] = float(np.mean([vt.inside_silence(c, silences, pad=0.05) for c in cuts]))
    else:
        m["cuts_in_silence_share"] = None
    checks = {k: _check(k, m.get(k)) for k in THRESHOLDS}
    blocks = [k for k, c in checks.items() if c["state"] == "violation" and c["level"] == "block" and c["status"] == "calibrated"]
    warns = [k for k, c in checks.items() if c["state"] == "violation" and k not in blocks]
    nosig = [k for k, c in checks.items() if c["state"] == "no_signal"]
    rep.update(checks=checks, blocks=blocks, warnings=warns, no_signal=nosig,
               measured={k: v for k, v in m.items() if k not in ("per_plan",)}, per_plan=m.get("per_plan"),
               plans=[[round(a, 3), round(b, 3)] for a, b in plans], cuts=[round(c, 3) for c in cuts],
               qc_seconds=round(time.time() - t0, 1), video_seconds=round(total, 1))
    rep["verdict"] = "blocked" if blocks else ("warnings" if warns else "ok")
    code = EXIT_BLOCK if blocks else (EXIT_WARN if warns else EXIT_OK)
    return rep, code


def summarize(rep):
    lines = [f"Montage QC: {rep.get('verdict')}  ({rep.get('qc_seconds')} с на {rep.get('video_seconds')} с видео)"]
    for k, c in rep.get("checks", {}).items():
        mark = {"ok": "✓", "violation": "✗", "no_signal": "·"}[c["state"]]
        val = c["value"]
        val = f"{val:.3f}" if isinstance(val, float) else val
        lines.append(f"  {mark} {k:24s} {val!s:>10}  порог {c['op']} {c['threshold']}  [{c['level']}, {c['status']}]")
    if rep.get("no_signal"):
        lines.append(f"  без сигнала: {len(rep['no_signal'])} — это не «прошло»")
    return "\n".join(lines)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__); return EXIT_BLOCK
    video_dir = os.path.abspath(argv[0])
    rep, code = build(video_dir, argv[1] if len(argv) > 1 else None)
    out = os.path.join(video_dir, "media_plan", "montage_qc.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(summarize(rep))
    return code


if __name__ == "__main__":
    sys.exit(main())
