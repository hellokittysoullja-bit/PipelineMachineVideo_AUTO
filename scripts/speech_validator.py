#!/usr/bin/env python3
"""Cadence Validator — вторая половина Speech Director (новая ЧАСТЬ 13, шаг
после озвучки, ДО fix_pauses.py). Сверяет speech_plan.json (риторическое
намерение, см. speech_planner.py) с тем, что РЕАЛЬНО получилось в
audio.mp3 — не "тихо ли тут" (silencedetect), а "сработал ли план".

Источник наблюдения — ПОСИМВОЛЬНЫЙ alignment.csv (media_plan/alignment/*.csv,
та же раскладка по секциям, что уже читает pipeline_smart.load_alignment_weights),
не акустическая тишина: разрыв между концом речи одного юнита и началом
следующего виден в alignment ТОЧНО и независимо от порога silencedetect
(THRESH_SEC=1.0 в fix_pauses.py) — короткая, но реальная пауза 0.3с
акустически может не набрать порог тишины вообще, а в alignment она видна
всегда. Это и есть разница между "восстанавливать ритм из silencedetect"
(старый fix_pauses.py) и "измерять реальное речевое событие" (что просит
Cadence Validator).

Результат — media_plan/speech_timeline.json:
  - "units": по каждому юниту плана — наблюдение (observed_raw_dur) и
    вердикт (ok / mismatch_short / mismatch_long / no_signal) с reason/
    confidence/observed_value/target_range/action, тот же принцип
    типизированных флагов, что уже применяет qc_verdict() в visual_qc.py.
  - "protected_windows": плоский список [raw_start, raw_end, target_kept_sec,
    unit_id] — ТОЛЬКО для юнитов с реальным наблюдением и protected=true из
    плана. fix_pauses.py читает именно этот список, чтобы НЕ трогать (или
    трогать по цели плана, не по гладкой кривой) паузы, которые Speech
    Planner поставил осознанно.
  - "pause_decisions": то же самое, но с объяснением (Pause Intelligence,
    см. scripts/pause_intelligence.py) — confidence, применён ли cooldown
    между двумя соседними длинными паузами одной секции, music_action и
    т.д. Дублируется отдельным файлом media_plan/pause_decisions.json
    (честный аудит-трейл, не только для fix_pauses.py, но и для человека).

Действие при mismatch — НЕ автоматический повторный вызов TTS (это стоило
бы реальных кредитов ElevenLabs при каждом ложном срабатывании и убрало бы
человека из уже существующего ручного шага озвучки — сознательно оставлено
человеку, см. action="re-record phrase manually" в отчёте). Валидатор
только измеряет и объясняет, что и почему разошлось с планом.

Usage: python scripts/speech_validator.py <video_dir>
Вход: media_plan/speech_plan.json + media_plan/alignment/*.csv (опционально —
нет alignment -> все юниты "no_signal", ничего не защищено, fix_pauses.py
работает как раньше, ноль регресса) + audio.mp3 (только для фингерпринта).
Выход: media_plan/speech_timeline.json.
Код возврата: 0 — все юниты в норме; 2 — прогон прошёл успешно, но есть
фразы, не достигшие задуманного ритма (advisory — "re-record phrase
manually", не сбой); 1 — реальный сбой (нет speech_plan.json, план не
прошёл валидацию, нет audio.mp3)."""
import csv
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

VIDEO_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
_saved_argv = sys.argv
sys.argv = ["pipeline_smart.py", VIDEO_DIR]
import pipeline_smart  # noqa: E402  (переиспользуем ALIGNMENT_TAG_RE/_real_speech_bounds — тот же парсинг alignment, что и рендер)
sys.argv = _saved_argv

import speech_planner as splanner  # noqa: E402  (validate_plan — план не доверяем без проверки)
import pause_intelligence as pause_intel  # noqa: E402  (cooldown + аудит-трейл поверх protected-окон)


# Допуск ("насколько короче/длиннее цели ещё не считаем провалом") —
# реальная TTS-начитка никогда не попадает в план миллисекунда-в-миллисекунду,
# и это нормально: живая речь и не должна. Разумная стартовая точка (как и
# пороги в visual_qc.py) — не откалибрована на реальных эпизодах этого
# канала, первый живой прогон может потребовать подстройки.
SHORT_HOLD_KINDS = frozenset({"reveal_hold", "closing_hold", "anticipation", "evidence_beat", "connective"})
FAST_KINDS = frozenset({"punchy", "question_rise"})
SHORT_MISMATCH_FACTOR = 0.6   # observed < target_lo * 0.6 -> явно не дотянули до намерения
LONG_MISMATCH_FACTOR = 2.0    # observed > target_hi * 2.0 -> явно сломали темп (для punchy/question_rise)
MIN_PROTECTED_SEC = 0.15      # тот же пол, что уже использует fix_pauses._keep_sec_for


def _audio_fingerprint(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_section_segments(section_order):
    """{имя_секции: [сегмент_символов, ...]} — точная копия того, как
    pipeline_smart.load_alignment_weights() режет alignment.csv по тегам
    [pause]/[short pause] (ALIGNMENT_TAG_RE) на сегменты 1:1 с под-блоками
    секции. Не переиспользуем саму load_alignment_weights() напрямую — та
    считает ВЕСА относительно уже урезанного (после fix_pauses) времени
    (raw_to_real_time), а нам здесь нужны СЫРЫЕ границы (fix_pauses ещё не
    отработал на этом шаге пайплайна)."""
    section_segments = {}
    if not os.path.isdir(pipeline_smart.ALIGNMENT_DIR):
        return section_segments
    for i, name in enumerate(section_order):
        path = os.path.join(pipeline_smart.ALIGNMENT_DIR, f"{i:02d}.csv")
        if not os.path.exists(path):
            continue
        try:
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            chars = [(r["char"], float(r["start"]), float(r["end"])) for r in rows]
        except Exception as e:
            print(f"  alignment {path} битый, пропускаю: {e}")
            continue
        text = "".join(c for c, s, e in chars)
        segs, pos = [], 0
        for m in pipeline_smart.ALIGNMENT_TAG_RE.finditer(text):
            segs.append(chars[pos:m.start()])
            pos = m.end()
        segs.append(chars[pos:])
        section_segments[name] = segs
    return section_segments


def _flat_segment_bounds(units, section_segments):
    """(raw_start, raw_end) | None — по одному на КАЖДЫЙ юнит плана, в том же
    порядке, что units. None — юнит без валидных данных (файл секции
    отсутствует, сегментов не хватило — текст правили после записи, тот же
    класс проблемы, что уже обрабатывает load_alignment_weights()).

    ВАЖНО: alignment.csv нарезан ПО СЕКЦИЯМ (00.csv, 01.csv, ...) — pipeline_smart.py
    прямо документирует, что каждый файл приходит "вместе с КАЖДЫМ TTS-заказом",
    то есть, по всей вероятности, отдельным TTS-запросом на секцию со своей
    нулевой точкой времени, а не общей сквозной шкалой на весь audio.mp3.
    load_alignment_weights() это ничем не нарушает — она считает только
    РАЗНОСТИ (длительность) ВНУТРИ одного файла, где абсолютный ноль не
    имеет значения. Здесь я делаю то же самое: возвращаю границы как есть
    (в собственной шкале файла секции), а build_timeline() ниже НИКОГДА не
    вычисляет разрыв между юнитами из РАЗНЫХ файлов — иначе получится
    случайная (и подтверждённая тестом — отрицательная/бессмысленная)
    величина вместо реальной паузы."""
    seg_cursor = {}
    bounds = []
    for u in units:
        segs = section_segments.get(u["section"])
        if segs is None:
            bounds.append(None)
            continue
        k = seg_cursor.get(u["section"], 0)
        seg_cursor[u["section"]] = k + 1
        if k >= len(segs):
            bounds.append(None)
            continue
        bounds.append(pipeline_smart._real_speech_bounds(segs[k]))
    return bounds


def _validate_unit(u, observed_raw_dur):
    lo, hi = u["target_range_sec"]
    if observed_raw_dur is None:
        return {"status": "no_signal", "reason": "нет alignment-данных для этого юнита",
                "confidence": 0.0, "observed_value": None, "target_range": [lo, hi], "action": "none"}
    kind = u["rhetorical_kind"]
    if kind in FAST_KINDS:
        if observed_raw_dur > hi * LONG_MISMATCH_FACTOR:
            return {"status": "mismatch_long",
                    "reason": f"пауза длиннее задуманного темпа ({observed_raw_dur:.2f}с "
                              f"против цели {lo:.2f}-{hi:.2f}с) — ломает momentum '{kind}'",
                    "confidence": 0.6, "observed_value": round(observed_raw_dur, 3),
                    "target_range": [lo, hi], "action": "re-record phrase manually"}
        return {"status": "ok", "reason": "в пределах ожидаемого темпа", "confidence": 0.7,
                "observed_value": round(observed_raw_dur, 3), "target_range": [lo, hi], "action": "accept"}
    if kind in SHORT_HOLD_KINDS and lo > 0:
        if observed_raw_dur < lo * SHORT_MISMATCH_FACTOR:
            return {"status": "mismatch_short",
                    "reason": f"TTS не дал нужного hold'а ({observed_raw_dur:.2f}с "
                              f"против цели {lo:.2f}-{hi:.2f}с для '{kind}')",
                    "confidence": 0.6, "observed_value": round(observed_raw_dur, 3),
                    "target_range": [lo, hi], "action": "re-record phrase manually"}
        return {"status": "ok", "reason": "намерение достигнуто (или обрежется до цели фиксом пауз)",
                "confidence": 0.7, "observed_value": round(observed_raw_dur, 3),
                "target_range": [lo, hi], "action": "accept"}
    return {"status": "ok", "reason": "нет строгой цели (connective/none) — без проверки",
            "confidence": 0.5, "observed_value": round(observed_raw_dur, 3),
            "target_range": [lo, hi], "action": "accept"}


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build_timeline(units, audio_md5):
    section_order = []
    for u in units:
        if not section_order or section_order[-1] != u["section"]:
            section_order.append(u["section"])
    section_segments = _load_section_segments(section_order)
    bounds = _flat_segment_bounds(units, section_segments)

    out_units, protected_entries = [], []
    for i, u in enumerate(units):
        this_b = bounds[i]
        next_u = units[i + 1] if i + 1 < len(units) else None
        # Разрыв меряется ТОЛЬКО внутри одной секции (см. докстринг
        # _flat_segment_bounds) — сосед из другой секции означает другой
        # alignment-файл со своей нулевой точкой, разность была бы
        # случайным числом, не реальной паузой.
        next_b = bounds[i + 1] if (next_u is not None and next_u["section"] == u["section"]) else None
        observed_raw_dur = None
        raw_window = None
        if this_b is not None and next_b is not None and next_b[0] >= this_b[1]:
            observed_raw_dur = next_b[0] - this_b[1]
            raw_window = [round(this_b[1], 6), round(next_b[0], 6)]
        validation = _validate_unit(u, observed_raw_dur)
        entry = dict(u)
        entry["raw_window"] = raw_window
        entry["validation"] = validation
        out_units.append(entry)

        if u["protected"] and raw_window is not None and observed_raw_dur is not None:
            lo, hi = u["target_range_sec"]
            kept = max(MIN_PROTECTED_SEC, min(observed_raw_dur, hi if hi > 0 else observed_raw_dur))
            protected_entries.append({
                "unit_id": u["unit_id"], "section": u["section"],
                "rhetorical_kind": u["rhetorical_kind"], "raw_window": raw_window,
                "kept": kept, "lo": lo, "hi": hi,
            })

    # Pause Intelligence (см. scripts/pause_intelligence.py): cooldown между
    # соседними длинными protected-паузами одной секции + аудит-трейл решения
    # (почему именно эта длительность и что она означает для музыки/визуала/
    # субтитров) — 1:1 с protected_entries, только опускает финальный kept.
    pause_decisions = pause_intel.build_decisions(protected_entries)
    protected_windows = [
        [d["raw_window"][0], d["raw_window"][1], d["final_target_sec"], d["unit_id"]]
        for d in pause_decisions
    ]

    return {"source_audio_md5": audio_md5, "units": out_units,
            "protected_windows": protected_windows, "pause_decisions": pause_decisions}


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    plan_path = os.path.join(video_dir, "media_plan", "speech_plan.json")
    if not os.path.exists(plan_path):
        print(f"speech_plan.json не найден: {plan_path} — запусти сначала scripts/speech_planner.py")
        return 1
    plan = json.load(open(plan_path, encoding="utf-8"))
    try:
        splanner.validate_plan(plan)
    except ValueError as e:
        print(f"speech_plan.json не прошёл валидацию — {e}")
        return 1

    audio_path = os.path.join(video_dir, "audio.mp3")
    audio_md5 = _audio_fingerprint(audio_path) if os.path.exists(audio_path) else None
    if audio_md5 is None:
        print("audio.mp3 не найден — Cadence Validator работает после озвучки, до fix_pauses.py")
        return 1

    if not os.path.isdir(pipeline_smart.ALIGNMENT_DIR):
        print("media_plan/alignment/ не найден — валидация невозможна без посимвольного "
              "alignment (ElevenLabs/Lumean отдают его вместе с TTS-заказом). "
              "speech_timeline.json будет пустым по protected_windows — fix_pauses.py "
              "отработает как обычно, без защиты плановых пауз (ноль регресса).")

    timeline = build_timeline(plan["units"], audio_md5)
    plan_dir = os.path.join(video_dir, "media_plan")
    os.makedirs(plan_dir, exist_ok=True)
    atomic_write_json(os.path.join(plan_dir, "speech_timeline.json"), timeline)
    atomic_write_json(os.path.join(plan_dir, "pause_decisions.json"),
                       {"source_audio_md5": audio_md5, "decisions": timeline["pause_decisions"]})

    cooldowns = sum(1 for d in timeline["pause_decisions"] if d["cooldown_applied"])
    if cooldowns:
        print(f"  Pause Intelligence: cooldown понизил {cooldowns} паузу(з) рядом с другой "
              f"длинной паузой той же секции — см. media_plan/pause_decisions.json")

    counts = {}
    needs_retake = []
    for u in timeline["units"]:
        st = u["validation"]["status"]
        counts[st] = counts.get(st, 0) + 1
        if u["validation"]["action"] == "re-record phrase manually":
            needs_retake.append((u["unit_id"], u["validation"]["reason"]))

    print(f"Cadence Validator: {len(timeline['units'])} юнитов — " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"  protected_windows для fix_pauses.py: {len(timeline['protected_windows'])}")
    if needs_retake:
        print(f"\n  {len(needs_retake)} фраз(а) не достигли задуманного ритма — ручная пересъёмка:")
        for unit_id, reason in needs_retake:
            print(f"    [{unit_id}] {reason}")
    print(f"\nГотово: {os.path.join(plan_dir, 'speech_timeline.json')}")
    # 0 — чисто; 2 — ПРОГОН ПРОШЁЛ УСПЕШНО, но есть фразы на ручную пересъёмку
    # (advisory, не сбой); 1 зарезервирован за реальными сбоями выше (нет
    # speech_plan.json, план не прошёл валидацию, нет audio.mp3). Раньше 1
    # означало и то, и другое — при цепочном запуске (`&&`, как подразумевает
    # ЧАСТЬ 13 CLAUDE.md) advisory-результат был неотличим от настоящего краша.
    return 2 if needs_retake else 0


if __name__ == "__main__":
    sys.exit(main())
