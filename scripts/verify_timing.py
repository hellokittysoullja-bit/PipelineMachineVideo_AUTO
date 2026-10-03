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
(свой детектор покадровой разницы), слова голоса — из посимвольного alignment.
Если между расчётом и рендером что-то разъедется (чанкование, обрезка
муксом, потерянный клип, переход не той длины), phrase_timeline.json об этом
не узнает по устройству, а эта проверка увидит.

ДВЕ ОСИ — обе сверяют пиксели готового файла с речью, а не с моделью монтажа:

1. ДРЕЙФ РЕЗА ОТ ФРАЗЫ. Найденные в кадре резы сопоставляются с
   `speech_onset_sec` из phrase_timeline.json (реальные онсеты речи из
   посимвольного alignment). Отдельно считается ТРЕНД дрейфа (мс в минуту) —
   именно накопительный уход, а не разовая ошибка, был реальным симптомом
   всех трёх исторических поломок тайминга в этом проекте.

2. РЕЗ НЕ ПОПЕРЁК СЛОВА. Доля жёстких резов, найденных в пикселях
   final.mp4 и попавших ВНУТРЬ произнесённого слова (дальше допуска от его
   краёв). Слова — интервалы речи по посимвольному alignment
   (`speech_words` в phrase_timeline.json, та же шкала, что онсеты).

   До 02.10 эта ось искала ТИШИНУ в звуке final.mp4 — и была неверна дважды.
   Под голосом в готовом ролике лежат музыка и атмосфера: на 03_plen
   silencedetect не нашёл ни одного интервала тишины, доля «резов в тишине»
   вышла 0.0, и --strict-production остановил бы любой нормальный эпизод. А
   на чистом голосе ось противоречила самому PHRASE LOCK: рез ставится на
   онсет речи, то есть ровно туда, где тишина КОНЧАЕТСЯ, а рез внутри
   длинной фразы (нарезка) стоит между словами без всякой паузы — 8.6% резов
   «в тишине» на audio_fixed.flac при совершенно правильном монтаже.

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
# Рез поперёк слова — видимая ошибка: картинка меняется посреди слова.
# Допуск от края слова: кадр детектора (рез виден на кадре ПОСЛЕ смены) плюс
# полкадра сетки, на которую PHRASE LOCK кладёт границу. Порог доли — не
# измеренный оптимум, а потолок: у правильного PHRASE LOCK таких резов ноль,
# и единичные попадания — ложные пики детектора на живом видео со стока.
CUT_INSIDE_WORD_TOL_FRAMES = 1.5
MAX_CUTS_ACROSS_WORDS = 0.10
# Ось меряется только у жёстких резов: у диссолва нет одного момента смены,
# и детектор ставит его в середину смешивания (+90…+190 мс на 03_plen).
HARD_CUT_MAX_FRAMES = 1.5


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


def inside_word(t, words, tol):
    """Рез t лежит внутри слова дальше допуска tol от обоих его краёв."""
    for s, e in words:
        if s + tol < t < e - tol:
            return True
    return False


def cuts_across_words(pairs, hard_onsets, words, fps):
    """Ось 2: {checked, across, share, tolerance_ms, examples} по жёстким
    резам (pairs — (ожидаемый онсет, найденный рез)). None — слов нет."""
    if not words:
        return None
    tol = CUT_INSIDE_WORD_TOL_FRAMES / fps
    checked = [(e, d) for e, d in pairs if e in hard_onsets]
    across = [(e, d) for e, d in checked if inside_word(d, words, tol)]
    return {"checked": len(checked), "across": len(across),
            "share": round(len(across) / len(checked), 4) if checked else None,
            "tolerance_ms": round(tol * 1000, 1),
            "examples_sec": [[round(e, 3), round(d, 3)] for e, d in across[:20]]}


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
    onsets, locked, words, hard_onsets = [], False, [], set()
    fps = None
    if os.path.exists(pt_path):
        try:
            with open(pt_path, encoding="utf-8") as f:
                pt = json.load(f)
            locked = bool(pt.get("locked"))
            fps = float(pt.get("fps") or 0) or None
            blocks = [b for b in pt.get("blocks", []) if b.get("speech_onset_sec") is not None]
            onsets = [b.get("speech_onset_sec") for b in blocks]
            words = [(float(a), float(e)) for a, e in (pt.get("speech_words") or [])]
            fps_ = fps or 24.0
            hard_onsets = {b["speech_onset_sec"] for b in blocks
                           if float(b.get("transition_in_sec") if b.get("transition_in_sec")
                                    is not None else 1.0 / fps_) <= HARD_CUT_MAX_FRAMES / fps_ + 1e-9}
        except Exception:
            onsets, words, hard_onsets = [], [], set()
    report["phrase_lock"] = locked

    detected = detect_cuts(video_path, threshold)
    report["detected_cuts"] = len(detected)

    # --- Ось 1: дрейф реза от фразы ---
    expected = [t for t in onsets if t and t > 0.0]
    report["expected_cuts"] = len(expected)
    if not expected:
        report["cuts_across_words"] = None
        report["verdict"] = "no_reference"
        report["note"] = ("нет phrase_timeline.json с онсетами речи — дрейф и «рез "
                          "поперёк слова» измерить нечем")
        return report, 0

    pairs, unmatched, extra = match_cuts(expected, detected)
    report["cuts_across_words"] = cuts_across_words(
        pairs, hard_onsets, words, fps or video_fps(video_path) or 24.0)
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
    elif ((report["cuts_across_words"] or {}).get("share") is not None
          and report["cuts_across_words"]["share"] > MAX_CUTS_ACROSS_WORDS):
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
    caw = report.get("cuts_across_words")
    if caw:
        print(f"    жёстких резов поперёк слова: {caw.get('across')} из {caw.get('checked')} "
              f"(доля {caw.get('share')}, допуск {caw.get('tolerance_ms')}мс)")
    else:
        print("    рез поперёк слова: не измерено (в phrase_timeline.json нет слов голоса)")
    print(f"    отчёт: {path}")
    if v == "low_coverage":
        print("    ВНИМАНИЕ: измерения по дрейфу НЕТ (мало найденных резов) — "
              "это не «всё хорошо», это «нечем проверить»")
    return code


if __name__ == "__main__":
    sys.exit(main())
