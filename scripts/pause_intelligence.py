#!/usr/bin/env python3
"""Pause Intelligence — минимальный, детерминированный слой поверх Cadence
Validator (см. speech_validator.py). Безопасное подмножество внешнего ТЗ
"Adaptive Pause Intelligence": только то, что можно построить БЕЗ живой
калибровки на реальных эпизодах этого канала. Контекстная коррекция цели по
темпу/плотности речи конкретного диктора и протокол shadow-калибровки из ТЗ
сознательно НЕ реализованы здесь — это не "то же самое", а осознанно урезанное
подмножество (см. обсуждение при вводе в эксплуатацию).

Что реально построено:

  1. Cooldown между двумя "длинными" (>= LONG_PAUSE_THRESHOLD_SEC — см.
     .md-таблицу "Рабочие диапазоны пауз", правило 4: "не ставить две паузы
     свыше 1.05с подряд") protected-паузами ВНУТРИ одной секции: если между
     ними меньше COOLDOWN_MIN_SPEECH_GAP_SEC живой речи, вторая пауза
     понижается до НИЖНЕЙ границы своего собственного диапазона. Кулдаун
     МОЖЕТ только понижать финальную длительность, никогда не повышать —
     безопасно включать сразу, без риска регресса на уже проверенных
     эпизодах (см. apply_cooldown()).
  2. Явный аудит-трейл решений (media_plan/pause_decisions.json, пишет
     speech_validator.py) — обобщение того, что раньше было зашито только в
     pipeline_smart._exclude_climax_overlapping_windows(): человекочитаемое
     "почему" для каждой protected-паузы (confidence, cooldown, music_action).
  3. Confidence + жёсткий порог для reveal_hold (HARD_CONFIDENCE_GATE_*) —
     сегодня фактически no-op предохранитель: единственный источник climax
     в этом пайплайне — явный тег [climax] в сценарии, confidence для
     reveal_hold всегда 1.0 (см. CONFIDENCE_BY_KIND). Но структура готова на
     будущее (вероятностное определение climax) без переписывания вызывающего
     кода — см. apply_confidence_gate().

ВАЖНО про music_action: климакс-провал (_climax_dip_window) в
pipeline_smart.py считается в РЕАЛЬНОМ (после fix_pauses.py) времени, а этот
модуль работает на этапе Cadence Validator — ДО fix_pauses.py, только в
сыром (raw) времени секции. Пересчитывать overlap по абсолютным меткам здесь
было бы реальной ошибкой (секции ещё не обрезаны и не сдвинуты в общую
шкалу — см. section_offsets.json). Поэтому music_action ниже — СТРУКТУРНЫЙ
вывод (по rhetorical_kind: reveal_hold по построению classify_unit() всегда
пауза непосредственно перед climax-блоком), не пересчёт по времени.
Фактическое гашение swell на пересечении с dip остаётся ЕДИНСТВЕННО в уже
протестированной pipeline_smart._exclude_climax_overlapping_windows() —
этот модуль её не дублирует и не заменяет, только объясняет решение заранее.

visual_action/subtitle_action — поля контракта присутствуют (по духу
ТЗ: music/visual/subtitle раздельно), но у пайплайна сегодня НЕТ источника
сигнала для содержательного решения по ним (плотность субтитров нигде не
считается; Ken Burns и так уже даёт каждому юниту свой видео-слот независимо
от роли паузы — не новое решение этого модуля). Оба поля честно
информационные (см. VISUAL_ACTION_EXISTING_CUT / SUBTITLE_ACTION_NOT_EVALUATED),
не выдаются за принятые решения.

Не самостоятельный CLI-скрипт — вызывается из speech_validator.py
(apply_cooldown()/build_decisions() внутри build_timeline())."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

VIDEO_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
_saved_argv = sys.argv
sys.argv = ["pipeline_smart.py", VIDEO_DIR]
import pipeline_smart  # noqa: E402  (PAUSE_SWELL_MIN_KEEP_SEC — единственный источник этого порога)
import speech_planner as splanner  # noqa: E402  (RHETORICAL_RANGES — безопасный откат для gate)
sys.argv = _saved_argv


LONG_PAUSE_THRESHOLD_SEC = 1.05    # .md "Рабочие диапазоны пауз", правило 2/4
COOLDOWN_MIN_SPEECH_GAP_SEC = 2.0  # НЕ откалибровано на реальных эпизодах (см. докстринг
                                     # модуля) — разумный дефолт: явно ~2с непрерывной речи
                                     # между двумя длинными паузами одной секции

HARD_CONFIDENCE_GATE_KIND = "reveal_hold"
HARD_CONFIDENCE_GATE_MIN = 0.70

# Детерминированная confidence по риторической роли — от самого прямого
# сигнала сценария (climax-тег) к самому косвенному (пунктуация). Не
# вероятностная модель, просто честная маркировка "насколько прямой сигнал
# сценария стоит за этой ролью" — та же дисциплина, что и в classify_unit()
# у speech_planner.py.
CONFIDENCE_BY_KIND = {
    "reveal_hold":   1.0,   # явный тег [climax] в блоке ПОСЛЕ этого юнита
    "evidence_beat": 0.9,   # явный тег [stat:...]
    "closing_hold":  0.8,   # структурная позиция (последний юнит секции FINAL)
    "punchy":        0.8,   # структурная позиция (HOOK, <=6 слов)
    "anticipation":  0.6,   # пунктуация (двоеточие/тире) — косвенный сигнал
    "question_rise": 0.6,   # пунктуация (?) — косвенный сигнал
    "connective":    0.5,   # нейтральный случай, план ничего не утверждает
}

VISUAL_ACTION_EXISTING_CUT = "existing_cut_boundary"
SUBTITLE_ACTION_NOT_EVALUATED = "not_evaluated"


def confidence_for(kind):
    return CONFIDENCE_BY_KIND.get(kind, 0.5)


def apply_confidence_gate(kind, confidence, hi):
    """Возвращает (final_hi, passed, note). Не трогает ничего, кроме
    HARD_CONFIDENCE_GATE_KIND — остальные роли сегодня без жёсткого порога
    (нет отдельного вероятностного источника сигнала, который стоило бы
    перепроверять). Сегодня для reveal_hold confidence всегда 1.0
    (CONFIDENCE_BY_KIND) — эта ветка не-no-op только когда/если climax
    начнёт определяться не только явным тегом сценария."""
    if kind != HARD_CONFIDENCE_GATE_KIND or confidence >= HARD_CONFIDENCE_GATE_MIN:
        return hi, True, None
    safe_hi = splanner.RHETORICAL_RANGES["evidence_beat"][1]
    note = (f"confidence {confidence:.2f} < {HARD_CONFIDENCE_GATE_MIN:.2f} для '{kind}' — "
            f"цель понижена до безопасной верхней границы evidence_beat ({safe_hi:.2f}с) "
            f"вместо полного reveal_hold")
    return min(hi, safe_hi), False, note


def apply_cooldown(entries):
    """entries: [{"unit_id","section","rhetorical_kind","raw_window":[s,e],
    "kept","lo"}, ...] в порядке появления в сценарии (тот же порядок, что
    speech_validator.build_timeline() строит protected-юниты). Кулдаун
    сравнивается ТОЛЬКО внутри одной секции — граница между секциями не
    несёт смысла в сыром времени (та же конвенция, что и у
    _flat_segment_bounds() в speech_validator.py). Возвращает НОВЫЙ список
    (та же длина и порядок), kept скорректирован, добавлен ключ
    'cooldown_applied'. Downgrade-only: kept может только уменьшиться (lo
    самой длинной из наших ролей всегда < LONG_PAUSE_THRESHOLD_SEC), поэтому
    безопасно включать сразу, без риска ухудшить уже проверенное поведение."""
    out = []
    last_long_end_by_section = {}
    for e in entries:
        e = dict(e)
        section = e["section"]
        kept = e["kept"]
        is_long = kept >= LONG_PAUSE_THRESHOLD_SEC
        applied = False
        if is_long:
            prev_end = last_long_end_by_section.get(section)
            if prev_end is not None:
                gap = e["raw_window"][0] - prev_end
                if gap < COOLDOWN_MIN_SPEECH_GAP_SEC:
                    kept = e["lo"]
                    applied = True
                    is_long = kept >= LONG_PAUSE_THRESHOLD_SEC
        if is_long:
            last_long_end_by_section[section] = e["raw_window"][1]
        e["kept"] = kept
        e["cooldown_applied"] = applied
        out.append(e)
    return out


def _music_action_for(kind, final_target_sec):
    if kind == "reveal_hold":
        return "climax_dip_only_swell_suppressed"
    if final_target_sec >= pipeline_smart.PAUSE_SWELL_MIN_KEEP_SEC:
        return "swell"
    return "none"


def build_decisions(protected_entries):
    """protected_entries: [{"unit_id","section","rhetorical_kind",
    "raw_window":[s,e],"kept","lo","hi"}, ...] в порядке появления в
    сценарии — см. speech_validator.build_timeline(). Возвращает список
    dict, 1:1 с входом (тот же порядок, ничего не добавляет/не убирает):
    unit_id/section/rhetorical_kind/confidence/confidence_gate_passed/
    base_target_sec/final_target_sec/cooldown_applied/music_action/
    visual_action/subtitle_action/notes/raw_window. final_target_sec — это
    и есть скорректированный kept, который speech_validator.py кладёт в
    protected_windows для fix_pauses.py."""
    gated = []
    for e in protected_entries:
        e = dict(e)
        kind = e["rhetorical_kind"]
        confidence = confidence_for(kind)
        gate_hi, gate_passed, gate_note = apply_confidence_gate(kind, confidence, e["hi"])
        e["kept"] = min(e["kept"], gate_hi)
        e["confidence"] = confidence
        e["confidence_gate_passed"] = gate_passed
        e["gate_note"] = gate_note
        gated.append(e)

    cooled = apply_cooldown(gated)

    decisions = []
    for e in cooled:
        notes = []
        if e.get("gate_note"):
            notes.append(e["gate_note"])
        if e["cooldown_applied"]:
            notes.append(
                f"cooldown: соседняя длинная пауза той же секции ближе "
                f"{COOLDOWN_MIN_SPEECH_GAP_SEC:.1f}с речи — цель понижена до нижней "
                f"границы {e['lo']:.2f}с (не ставить две паузы свыше "
                f"{LONG_PAUSE_THRESHOLD_SEC:.2f}с подряд)")
        music_action = _music_action_for(e["rhetorical_kind"], e["kept"])
        decisions.append({
            "unit_id": e["unit_id"],
            "section": e["section"],
            "rhetorical_kind": e["rhetorical_kind"],
            "confidence": e["confidence"],
            "confidence_gate_passed": e["confidence_gate_passed"],
            "base_target_sec": round(e["hi"], 6),
            "final_target_sec": round(e["kept"], 6),
            "cooldown_applied": e["cooldown_applied"],
            "music_action": music_action,
            "visual_action": VISUAL_ACTION_EXISTING_CUT,
            "subtitle_action": SUBTITLE_ACTION_NOT_EVALUATED,
            "notes": notes,
            "raw_window": e["raw_window"],
        })
    return decisions
