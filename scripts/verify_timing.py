#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Измеренная проверка тайминга ПО ГОТОВОМУ ФАЙЛУ, а не по модели монтажа.

ЗАЧЕМ ЭТО СУЩЕСТВУЕТ. У эпизода уже есть media_plan/phrase_timeline.json —
там по каждому блоку записан дрейф между стартом кадра и началом фразы. Но
`visual_start_sec` там СЧИТАЕТ hook_visual_starts(), то есть модель того, как
xfade_chain() сожмёт таймлайн. Дрейф в этом файле — разница между двумя
величинами, посчитанными ОДНИМ И ТЕМ ЖЕ кодом из одних и тех же данных.
Собственный протокол канала (CLAUDE.md, «Протокол проверки перед словом
работает», п.1) запрещает ровно это: «проверка одного оценочного вывода
другим оценочным выводом того же механизма ничего не доказывает, только
создаёт ложную уверенность».

Здесь источник истины другой — сам файл. Резы ищутся в ПИКСЕЛЯХ final.mp4
(scene-детектор ffmpeg), тишина — в его ЗВУКОВОЙ дорожке (silencedetect).
Если между расчётом и рендером что-то разъедется (чанкование, обрезка
муксом, потерянный клип, переход не той длины), phrase_timeline.json об этом
не узнает по устройству, а эта проверка увидит.

ДВЕ НЕЗАВИСИМЫЕ ОСИ — вторая не использует alignment вообще:

1. ДРЕЙФ РЕЗА ОТ ФРАЗЫ. Найденные в кадре резы сопоставляются с
   `speech_onset_sec` из phrase_timeline.json (реальные онсеты речи из
   посимвольного alignment). Отдельно считается ТРЕНД дрейфа (мс в минуту) —
   именно накопительный уход, а не разовая ошибка, был реальным симптомом
   всех трёх исторических поломок тайминга в этом проекте.

2. РЕЗ В ТИШИНЕ, А НЕ ПОПЕРЁК СЛОВА. Доля найденных резов, попавших внутрь
   паузы в звуке готового ролика. Эта ось не знает ни про alignment, ни про
   phrase_timeline.json — она сверяет картинку со звуком напрямую, поэтому
   переживает любую ошибку в самой карте фраз.

ЧЕСТНЫЕ ПРЕДЕЛЫ (записаны в отчёт, а не только здесь):
* Детектор находит РЕЗКИЕ смены кадра. 15% переходов эпизода — короткий
  диссолв (plan_transitions), у него нет одного момента смены, и он может не
  дать пика вовсе. Поэтому покрытие («сколько ожидаемых резов найдено»)
  печатается отдельно, и НИЗКОЕ покрытие никогда не выдаётся за «дрейфа
  нет»: не найденный рез — это отсутствие измерения, а не доказательство.
* Медленный Ken Burns по контрастному кадру изредка даёт ложный пик. Такие
  резы попадают в `unmatched_detected` и на вердикт по дрейфу не влияют.
* Квантование детектора — один кадр (41.7мс при 24 fps). Поэтому бюджет
  медианы (MAX_MEDIAN_DRIFT_SEC) заведомо шире, чем ±полкадра, которые
  обещает сам расчёт: измерение не может быть точнее своей сетки.
"""
import json
import os
import re
import subprocess
import sys
import time

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# Разрешение кривой покадровой разницы. 64x36 хватает, чтобы смена кадра была
# видна, и на порядок дешевле полного декода в 1920x1080.
DIFF_W, DIFF_H = 64, 36

# Разница считается по ЦВЕТУ (rgb24), а не по яркости — и это не вкусовщина,
# а исправленный при калибровке промах. Готовый `select='gt(scene,...)'`
# ffmpeg смотрит только на яркость и на синтетическом монтаже с ИЗВЕСТНЫМИ
# резами не нашёл переход красный -> зелёный даже с порогом 0.01, найдя все
# остальные три. Причина оказалась не в пороге: у ffmpeg «красный» (255,0,0)
# и «зелёный» (0,128,0) дают ОДНУ И ТУ ЖЕ яркость Y=76 — в яркостной проекции
# этого реза физически нет. Первая версия этой функции считала по серому и
# повторила тот же промах один в один. На тёмном грейде канала, где соседние
# клипы часто близки по яркости, это был бы систематический слепой участок,
# выглядящий как «резов нет» — то есть ровно ложная уверенность, против
# которой эта проверка и написана.
#
# Порог при этом выбирается ПО САМОМУ МАТЕРИАЛУ (медиана + k*MAD), а не
# константой: у тёмного грейда и у яркого стока разный уровень фона.
# Разница кадра — МЕДИАНА модуля разности по пикселям, не среднее. Замер на
# синтетике с известными резами (та же zoompan-формула, что в рендере):
#              фон p95   пик реза   отношение
#   среднее      2.09      18.4        8.8
#   медиана      1.00      16.0       16.0
# Причина в природе помехи: дыхание зума сдвигает ЧАСТЬ пикселей сильно
# (ресемплинг по краям контраста), а смена кадра меняет ВСЕ. Среднее чувствует
# первое, медиана — только второе.
#
# ЧЕСТНЫЙ ПРЕДЕЛ, установленный тем же замером и записанный в отчёт: диссолв
# 0.5с между двумя шумными кадрами близкой яркости даёт отношение ~1.0, то
# есть НЕ отличим от дыхания зума ничем — ни другим порогом, ни другим
# разрешением (проверено 64x36 и 128x72, среднее и медиана). Это предел
# метода, а не настройки. В эпизоде диссолвом идут ~15% переходов тела и
# ноль переходов хука (plan_transitions), поэтому дрейф меряется по жёстким
# резам, а ненайденные переходы честно уходят в покрытие — и низкое покрытие
# никогда не выдаётся за «всё хорошо».
#
# Событие определяется ОТНОСИТЕЛЬНО СВОЕГО СОСЕДСТВА, а не общего уровня
# файла: в одном ролике есть и почти неподвижные фото под Ken Burns, и живое
# видео со стока, и общий порог на оба не годится.
DIFF_PEAK_RATIO = 4.0
DIFF_LOCAL_WINDOW_SEC = 2.0
DIFF_MIN_ABS = 4.0        # выше p95 дыхания зума (1.0-3.0) на всех проверенных разрешениях
DIFF_MIN_SEPARATION_SEC = 0.30

# Окно сопоставления «найденный рез <-> ожидаемый онсет». Шире половины
# самого длинного перехода (XFADE_DUR) и заметно уже самого короткого клипа,
# иначе рез сопоставился бы с чужой фразой.
MATCH_WINDOW_SEC = 0.60

# Бюджеты вердикта.
MAX_MEDIAN_DRIFT_SEC = 0.10      # ~2.4 кадра при 24 fps
MAX_DRIFT_TREND_SEC_PER_MIN = 0.05   # накопительный уход — главный симптом
MIN_COVERAGE = 0.50              # ниже этого измерения просто нет
MIN_CUTS_IN_SILENCE = 0.70       # рез поперёк слова — слышимая ошибка

SILENCE_NOISE_DB = -35.0
SILENCE_MIN_DUR = 0.20


def _run(cmd, timeout=None):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout).stdout.decode("utf-8", "replace")


def media_duration(path):
    try:
        out = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nw=1:nk=1", path], timeout=120).strip()
        return float(out.splitlines()[-1])
    except Exception:
        return None


def _timeout_for(path, per_minute=90, floor=300):
    """Полный декод — на часовом эпизоде фиксированный таймаут не годится
    (тот же класс бага, что аудит 04.09 чинил у measure_loudnorm_stats)."""
    dur = media_duration(path) or 0.0
    return max(floor, int(dur / 60.0 * per_minute))


def video_fps(path):
    try:
        out = _run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate",
                    "-of", "default=nw=1:nk=1", path], timeout=120).strip().splitlines()[-1]
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except Exception:
        return None


def frame_diff_curve(video_path, w=DIFF_W, h=DIFF_H):
    """Медианная абсолютная разница между соседними кадрами, по всему файлу.

    Один проход декодирования в 64x36 RGB — дальше вся арифметика локальная.
    Возвращает (fps, [разница_кадра_i_с_предыдущим, ...]).

    Почему своя кривая, а не готовый scene-детектор ffmpeg: тот сравнивает
    только яркость и на калибровке молча пропустил реальный рез (см. коммент
    у DIFF_PEAK_K). Кривая даёт ещё и форму события — резкий пик у склейки
    против широкого плато у диссолва."""
    try:
        import numpy as np
    except Exception:
        return None, []
    fps = video_fps(video_path) or 24.0
    cmd = [FFMPEG, "-v", "error", "-i", video_path, "-an",
           "-vf", f"scale={w}:{h},format=rgb24", "-f", "rawvideo", "-"]
    frame_bytes = w * h * 3
    diffs = []
    prev = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            cur = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
            if prev is not None:
                diffs.append(float(np.median(np.abs(cur - prev))))
            prev = cur
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait(timeout=_timeout_for(video_path))
    return fps, diffs


def cuts_from_curve(fps, diffs, ratio=DIFF_PEAK_RATIO):
    """Моменты смены кадра — пики кривой разницы над СВОИМ локальным фоном.

    Почему локальный фон, а не один порог на файл: у неподвижного фото под
    Ken Burns фон разницы около 0.5, у живого видео со стока — в разы выше.
    Один общий порог либо пропустил бы диссолв на спокойном участке, либо
    засыпал ложными пиками участок с движением. Возвращает (времена, порог_
    медианный) — второй только для отчёта.
    """
    if not diffs:
        return [], None
    half = max(1, int(round(DIFF_LOCAL_WINDOW_SEC * fps / 2.0)))
    n = len(diffs)
    peaks = []
    for i, d in enumerate(diffs):
        if d < DIFF_MIN_ABS:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        local = _median(diffs[lo:hi]) or 0.0
        if d >= max(DIFF_MIN_ABS, ratio * local):
            peaks.append(i)
    # Одно событие (особенно диссолв) даёт несколько соседних кадров над
    # порогом — берём кадр с максимальной разницей внутри группы.
    min_sep = max(1, int(round(DIFF_MIN_SEPARATION_SEC * fps)))
    times, group = [], []
    for i in peaks:
        if group and i - group[-1] > min_sep:
            best = max(group, key=lambda j: diffs[j])
            times.append((best + 1) / fps)
            group = []
        group.append(i)
    if group:
        best = max(group, key=lambda j: diffs[j])
        times.append((best + 1) / fps)
    return times, (_median(diffs) or 0.0)


def detect_cuts(video_path, ratio=DIFF_PEAK_RATIO):
    """Моменты смены картинки в готовом файле (секунды).

    Индекс i кривой — разница между кадром i+1 и i, то есть новая картинка
    видна НА КАДРЕ i+1: время берётся по нему, иначе рез систематически
    оказывался бы на один кадр раньше, чем он есть.
    """
    fps, diffs = frame_diff_curve(video_path)
    if not diffs:
        return []
    times, _ = cuts_from_curve(fps, diffs, ratio)
    return times


def detect_silences(video_path):
    """Интервалы тишины в звуке готового файла — [(start, end), ...]."""
    out = _run([FFMPEG, "-hide_banner", "-nostats", "-i", video_path, "-vn",
                "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_DUR}",
                "-f", "null", "-"], timeout=_timeout_for(video_path))
    starts = [float(x) for x in re.findall(r"silence_start: (-?[0-9.]+)", out)]
    ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", out)]
    spans = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        if e is not None and e > s:
            spans.append((s, e))
    return spans


def _median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _percentile(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[k]


def _linear_trend(points):
    """Наклон дрейфа по времени (сек дрейфа на сек ролика), метод наименьших
    квадратов. Разовая ошибка даёт наклон около нуля, накопительный уход —
    ненулевой; именно он и был симптомом всех трёх поломок тайминга."""
    if len(points) < 3:
        return None
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    den = sum((p[0] - mx) ** 2 for p in points)
    if den <= 1e-9:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in points) / den


def match_cuts(expected, detected, window=MATCH_WINDOW_SEC):
    """Ближайший найденный рез к каждому ожидаемому онсету, без повторного
    использования одного и того же реза двумя онсетами."""
    used = set()
    pairs, unmatched = [], []
    for e in expected:
        best, best_d = None, None
        for i, d in enumerate(detected):
            if i in used:
                continue
            dist = abs(d - e)
            if dist <= window and (best_d is None or dist < best_d):
                best, best_d, best_i = d, dist, i
        if best is None:
            unmatched.append(e)
        else:
            used.add(best_i)
            pairs.append((e, best))
    extra = [d for i, d in enumerate(detected) if i not in used]
    return pairs, unmatched, extra


def inside_silence(t, spans, pad=0.0):
    for s, e in spans:
        if s - pad <= t <= e + pad:
            return True
    return False


def verify(video_dir, video_path=None, threshold=DIFF_PEAK_RATIO):
    video_path = video_path or os.path.join(video_dir, "final.mp4")
    report = {"schema_version": 1, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "video": os.path.basename(video_path), "peak_ratio": threshold,
              "match_window_sec": MATCH_WINDOW_SEC,
              "limits": ["диссолв может не дать пика — низкое покрытие это ОТСУТСТВИЕ "
                         "измерения, а не доказательство точности",
                         "квантование детектора — один кадр",
                         "порог пика выведен из локального фона самого материала",
                         "ложный пик на контрастном Ken Burns попадает в unmatched_detected"]}
    if not os.path.exists(video_path):
        report["verdict"] = "no_video"
        return report, 1

    pt_path = os.path.join(video_dir, "media_plan", "phrase_timeline.json")
    onsets, locked = [], False
    if os.path.exists(pt_path):
        try:
            with open(pt_path, encoding="utf-8") as f:
                pt = json.load(f)
            locked = bool(pt.get("locked"))
            onsets = [b.get("speech_onset_sec") for b in pt.get("blocks", [])
                      if b.get("speech_onset_sec") is not None]
        except Exception:
            onsets = []
    report["phrase_lock"] = locked

    detected = detect_cuts(video_path, threshold)
    silences = detect_silences(video_path)
    report["detected_cuts"] = len(detected)
    report["silence_spans"] = len(silences)

    # --- Ось 2: рез в тишине (не зависит от alignment вообще) ---
    if detected:
        in_sil = sum(1 for t in detected if inside_silence(t, silences))
        report["cuts_in_silence_share"] = round(in_sil / len(detected), 4)
    else:
        report["cuts_in_silence_share"] = None

    # --- Ось 1: дрейф реза от фразы ---
    expected = [t for t in onsets if t and t > 0.0]
    report["expected_cuts"] = len(expected)
    if not expected:
        report["verdict"] = "no_reference"
        report["note"] = ("нет phrase_timeline.json с онсетами речи — дрейф измерить "
                          "нечем; ось «рез в тишине» выше всё равно посчитана")
        return report, (0 if report["cuts_in_silence_share"] is None
                        or report["cuts_in_silence_share"] >= MIN_CUTS_IN_SILENCE else 2)

    pairs, unmatched, extra = match_cuts(expected, detected)
    coverage = len(pairs) / len(expected)
    drifts = [d - e for e, d in pairs]
    report["matched"] = len(pairs)
    report["coverage"] = round(coverage, 4)
    report["unmatched_expected_sec"] = [round(t, 3) for t in unmatched[:40]]
    report["unmatched_detected_sec"] = [round(t, 3) for t in extra[:40]]
    report["drift"] = {
        "median_ms": round((_median([abs(x) for x in drifts]) or 0) * 1000, 1),
        "p90_ms": round((_percentile([abs(x) for x in drifts], 0.9) or 0) * 1000, 1),
        "max_ms": round((max([abs(x) for x in drifts]) if drifts else 0) * 1000, 1),
        "signed_mean_ms": round((sum(drifts) / len(drifts) if drifts else 0) * 1000, 1),
    }
    trend = _linear_trend([(e, d - e) for e, d in pairs])
    report["drift"]["trend_ms_per_min"] = (round(trend * 60 * 1000, 1)
                                           if trend is not None else None)

    verdict, code = "ok", 0
    if coverage < MIN_COVERAGE:
        verdict, code = "low_coverage", 2
    elif report["drift"]["median_ms"] > MAX_MEDIAN_DRIFT_SEC * 1000:
        verdict, code = "drift_median", 2
    elif (report["drift"]["trend_ms_per_min"] is not None
          and abs(report["drift"]["trend_ms_per_min"]) > MAX_DRIFT_TREND_SEC_PER_MIN * 1000):
        verdict, code = "drift_trend", 2
    elif (report["cuts_in_silence_share"] is not None
          and report["cuts_in_silence_share"] < MIN_CUTS_IN_SILENCE):
        verdict, code = "cuts_across_speech", 2
    report["verdict"] = verdict
    return report, code


def save_report(video_dir, report):
    path = os.path.join(video_dir, "media_plan", "timing_verification.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    return path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        print("Использование: python scripts/verify_timing.py <video_dir> [--video path]")
        return 1
    video_dir = argv[0]
    video_path = None
    if "--video" in argv:
        video_path = argv[argv.index("--video") + 1]
    report, code = verify(video_dir, video_path)
    path = save_report(video_dir, report)
    v = report.get("verdict")
    if v == "no_video":
        print(f"  Тайминг: файла нет — {video_path or 'final.mp4'}")
        return code
    d = report.get("drift") or {}
    print(f"  Тайминг ИЗМЕРЕН по готовому файлу: вердикт={v}")
    print(f"    резов найдено {report['detected_cuts']}, ожидалось {report.get('expected_cuts')}, "
          f"сопоставлено {report.get('matched')} (покрытие {report.get('coverage')})")
    if d:
        print(f"    дрейф: медиана {d.get('median_ms')}мс, p90 {d.get('p90_ms')}мс, "
              f"макс {d.get('max_ms')}мс, тренд {d.get('trend_ms_per_min')}мс/мин")
    print(f"    резов в тишине: {report.get('cuts_in_silence_share')}")
    print(f"    отчёт: {path}")
    if v == "low_coverage":
        print("    ВНИМАНИЕ: измерения по дрейфу НЕТ (мало найденных резов) — "
              "это не «всё хорошо», это «нечем проверить»")
    return code


if __name__ == "__main__":
    sys.exit(main())
