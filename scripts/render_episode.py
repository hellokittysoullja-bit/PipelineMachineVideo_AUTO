#!/usr/bin/env python3
"""render_episode.py — Production Orchestrator (P0-2 форензик-аудита:
"нет единого timeline authority: синхронизация зависит от ручных шагов и
имеет silent fallback").

ЧТО ЭТО. Раньше pipeline_smart.py сам по себе НЕ вызывал ни section_sync.py,
ни fix_pauses.py — каждый шаг протокола (ЧАСТЬ 13 CLAUDE.md) запускался
отдельной командой, по порядку. Пропущенный шаг не был ошибкой — просто
degraded timing (тот же безопасный откат, что уже описан в каждой из этих
функций по отдельности: "нет offset-файла — работает как раньше",
"нет speech_timeline.json — защита пауз не включается" и т.д.). Это разумно
для экспериментов, но для РЕАЛЬНОГО эпизода означает, что "всё
синхронизировано" — это лишь предположение, если человек/Claude Code сам
не проверил на глаз, что все нужные шаги действительно прошли по порядку.

Этот скрипт НЕ переписывает архитектуру — section_sync.py/fix_pauses.py/
pipeline_smart.py по-прежнему можно запускать и вручную, отдельно, как
раньше (ничего не убрано, ничего не сломано у существующих вызовов). Он
даёт ОДНУ команду, которая проходит бесплатные локальные шаги по порядку
и, в --strict-production, ОТКАЗЫВАЕТСЯ рендерить, если какой-то из них
пропущен или не дал уверенного результата, вместо того чтобы молча
деградировать.

НЕ тратит платные API за один запуск: speech_planner.py/speech_generate.py/
speech_validator.py сюда НЕ входят — предполагается, что audio.mp3 уже
существует (голос уже записан вручную ИЛИ через Stage B, Шаг 6 CLAUDE.md).
Этот оркестратор занимается только БЕСПЛАТНЫМИ локальными шагами ПОСЛЕ
этого (Шаги 6.7-7 протокола): section_sync.py (если нужен), fix_pauses.py,
затем сама сборка pipeline_smart.py.

Usage: python scripts/render_episode.py <video_dir> [--strict-production]
       [--legacy-allow-degraded-timing] [--legacy-allow-unreviewed-media]
       [--legacy-allow-unreviewed-render]

Без всех флагов — то же самое, что --legacy-allow-degraded-timing и
--legacy-allow-unreviewed-media вместе (полная обратная совместимость:
best-effort, ни на чём не падает, кроме явных ошибок ffmpeg/отсутствующего
audio.mp3) — этот скрипт остаётся строго ДОПОЛНЕНИЕМ к прежнему рабочему
процессу, не заменой по умолчанию.

--strict-production (рекомендуется для реального, не тестового эпизода):
    останавливается с точной ошибкой ДО запуска pipeline_smart.py, если:
    1) audio.mp3 отсутствует;
    2) Шаг 5.5 (scripts/visual_qc.py) ни разу не запускался ИЛИ его отчёт
       (media_plan/visual_qc_report.json) показывает нерешённые слоты
       (missing/reject/accepted_below_threshold — та же формула, что даёт
       return 2 у самого visual_qc.py) — реальный, ранее не закрытый пробел
       независимого архитектурного разбора этой сессии: strict-режим уже
       проверял тайминг/синхронизацию, но НИЧЕГО не знал про релевантность/
       качество самого медиа, эпизод с половиной отклонённых QC слотов
       спокойно проходил гейт;
    3) эпизод многосекционный (>=2 секций) и НИ Stage B, НИ section_sync.py
       не дали уверенного section_offsets.json на ВСЕ секции после первой;
    4) fix_pauses.py завершился с ошибкой (ffmpeg упал);
    5) ПОСЛЕ успешного рендера — media_plan/relevance_gate_report.json,
       media_plan/director_relevance_report.json или media_plan/
       render_qc_report.json (пишет сам pipeline_smart.py, см. их докстринги
       там) содержат непустые списки промахов — слот получил кандидата ниже
       порога relevance/семантики, либо готовый клип заметно размытее
       своего источника. final.mp4 при этом НЕ удаляется, просто сборка не
       считается чисто пройденной без ручной проверки (Шаг 7.5).
--legacy-allow-degraded-timing: явно разрешает продолжить рендер БЕЗ
    гарантий синхронизации (пункты 3-4 выше), даже если запрошен
    --strict-production — чтобы сознательный компромисс был явным флагом в
    команде, а не тихой деградацией по умолчанию.
--legacy-allow-unreviewed-media: тот же принцип, отдельным флагом — явно
    разрешает продолжить БЕЗ гарантии, что media\\ прошла Visual QC
    (пункт 2 выше). Отдельный от --legacy-allow-degraded-timing флаг
    (не одно и то же: тайминг и качество медиа — независимые гарантии,
    молчаливое объединение под одним именем было бы вводящим в заблуждение).
--legacy-allow-unreviewed-render: тот же принцип, третьим отдельным флагом —
    явно разрешает считать сборку пройденной, даже если пост-рендер отчёты
    (пункт 5 выше) нашли непройденные слоты. Отдельная гарантия от
    --legacy-allow-unreviewed-media: та проверяет media\\ ДО рендера (сырой
    кандидат), эта — САМ ГОТОВЫЙ КЛИП ПОСЛЕ рендера (может поймать баг,
    которого не было в исходнике, см. render_qc_report.json).

Пишет media_plan/timeline_manifest.json — сводка, что реально прогналось
и с каким результатом (тот же принцип честного аудит-трейла, что уже
применяют pause_decisions.json/visual_director_report.json/
section_sync_report.json)."""
import argparse
import json
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
# Явная гарантия импортируемости script_parser (см. _section_count): при
# запуске файлом sys.path[0] и так равен scripts/, но при любом другом
# способе вызова (импорт из другого каталога, -m, обёртка) этого нет —
# остальные модули пакета такую же страховку уже ставят.
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def _run(script_name, video_dir, extra_env=None):
    """Запускает scripts/<script_name> <video_dir> как подпроцесс (не
    import — section_sync.py/fix_pauses.py/pipeline_smart.py каждый сам
    парсит sys.argv[1] на импорте, см. их же докстринги про этот приём).
    Вывод дочернего процесса НЕ перехватывается (никакого capture_output) —
    идёт напрямую в унаследованные stdout/stderr в реальном времени. Раньше
    capture_output=True буферизовал ВЕСЬ вывод и печатал его только после
    завершения процесса — для рендер-шага (pipeline_smart.py), который может
    идти часами, это означало, что пользователь не видел ВООБЩЕ НИЧЕГО в
    терминале до конца или до падения, хотя сам pipeline_smart.py прилежно
    печатает прогресс каждые 10-20 блоков. PYTHONUNBUFFERED гарантирует
    реальное время и тогда, когда родительский stdout сам перенаправлен
    (например, в лог-файл через tee), а не только в интерактивном терминале."""
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, script_name), video_dir]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_env:
        env.update(extra_env)
    print(f"  -> {script_name} {video_dir}")
    r = subprocess.run(cmd, env=env)
    return r.returncode


def _section_count(video_dir):
    """Число секций (HOOK/BLOCK*/FINAL) в script.txt — та же логика, что
    section_sync.py использует для собственного порога "меньше двух секций
    -> детектор не нужен". parse_blocks() импортируется напрямую из
    script_parser.py (лёгкий модуль без побочных эффектов) — раньше это
    требовало временной подмены sys.argv и импорта ВСЕГО pipeline_smart.py
    (5900+ строк, тяжёлые torch/cv2-зависимости) только ради одной функции;
    section_sync.py вынужден так делать по другой причине (использует ещё
    несколько символов из pipeline_smart.py), render_episode.py — нет."""
    import script_parser
    script_path = os.path.join(video_dir, "script.txt")
    if not os.path.exists(script_path):
        return 0
    blocks = script_parser.parse_blocks(script_path)
    section_order = []
    for b in blocks:
        if not section_order or section_order[-1] != b["section"]:
            section_order.append(b["section"])
    return len(section_order)


def _section_offsets_cover_all(video_dir, expected_count):
    """True, если media_plan/section_offsets.json существует и покрывает
    ВСЕ секции эпизода (не только часть — частичный файл означает, что
    section_sync.py честно отказался от одной или нескольких секций по
    низкой уверенности, см. его докстринг про безопасный откат)."""
    path = os.path.join(video_dir, "media_plan", "section_offsets.json")
    if not os.path.exists(path):
        return False
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return False
    return len(data) >= expected_count


_POST_RENDER_REPORTS = (
    # (имя_файла, ключ_в_JSON_со_списком, человекочитаемая_причина)
    ("relevance_gate_report.json", "misses",
     "весь просмотренный пул кандидата провалил relevance-гейт (query)"),
    ("director_relevance_report.json", "misses",
     "выбранный кадр семантически слаб по РЕАЛЬНОМУ тексту блока (Директор)"),
    ("render_qc_report.json", "flagged",
     "готовый рендер заметно размытее своего источника (DOF/параллакс/грейд)"),
    # STOCK_EXHAUSTED_MISSES (см. её докстринг в pipeline_smart.py) — строго
    # более сильный сигнал, чем relevance_gate_report.json выше: не
    # "победитель не идеален", а "ни один кандидат из ВСЕГО просмотренного
    # пула не прошёл ни relevance, ни резкость" (независимо от дедупа).
    # Практический вывод для этих слотов другой — не "сверить глазами на
    # Шаге 7.5", а "сгенерировать AI-картинку/видео на Шаге 5 вместо стока".
    ("stock_exhausted_report.json", "misses",
     "сток не дал НИ ОДНОГО кандидата, прошедшего и relevance, и резкость — "
     "нужна AI-картинка/видео (Шаг 5) вместо стока, не ручная сверка"),
)


def _post_render_status(video_dir):
    """(status, total_count, details) — status "ok"/"unresolved"/"no_reports".
    Читает ТОЛЬКО отчёты, которые пишет сам pipeline_smart.py ПОСЛЕ рендера
    (см. RELEVANCE_GATE_MISSES/DIRECTOR_RELEVANCE_MISSES/RENDER_QC_REPORT в
    pipeline_smart.py) — в отличие от visual_qc_report.json (Шаг 5.5, ДО
    рендера, отдельный обязательный запуск), эти три существуют только
    ПОСЛЕ того, как pipeline_smart.py уже отработал, поэтому проверяются
    здесь ПОСТ-рендерно, не как preflight. "no_reports" — pipeline_smart.py
    ещё старой версии (или упал раньше, чем успел их записать) — то же
    отношение, что "missing_report" у visual_qc: не считается "ok" молча."""
    details = []
    any_report = False
    for fname, key, reason in _POST_RENDER_REPORTS:
        path = os.path.join(video_dir, "media_plan", fname)
        if not os.path.exists(path):
            continue
        any_report = True
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        items = data.get(key) or []
        if items:
            details.append({"report": fname, "count": len(items), "reason": reason})
    if not any_report:
        return "no_reports", 0, details
    total = sum(d["count"] for d in details)
    return ("unresolved", total, details) if total else ("ok", 0, details)


_VISUAL_QC_UNRESOLVED_VERDICTS = ("missing", "reject", "accepted_below_threshold")


def _visual_qc_status(video_dir):
    """(status, unresolved_count) — status один из "missing_report" (Шаг 5.5
    не запускался вообще), "unresolved" (запускался, но остались слоты,
    которые visual_qc.py сам просит проверить глазами — verdict в
    _VISUAL_QC_UNRESOLVED_VERDICTS, см. его main()/return 2), "ok".

    РЕАЛЬНЫЙ, ранее НЕ доведённый до конца пробел (см. независимый
    архитектурный разбор этой же сессии: "Visual QC... не является
    обязательным final gate текущего orchestrator") — --strict-production
    уже проверяет audio.mp3/section_offsets/fix_pauses, но НИЧЕГО не знает
    про релевантность/качество самого медиа: эпизод с половиной слотов,
    явно отклонённых QC (или вообще без единого прогона Шага 5.5), раньше
    спокойно проходил strict-гейт и собирался в final.mp4. Отчёт ТОЛЬКО
    читается (не запускает visual_qc.py заново — тот платно дёргает Pexels
    при подборе замены, это НЕ бесплатный локальный шаг, вне скоупа этого
    оркестратора, см. его докстринг)."""
    path = os.path.join(video_dir, "media_plan", "visual_qc_report.json")
    if not os.path.exists(path):
        return "missing_report", 0
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return "missing_report", 0
    # {"counts": {...}, "slots": [...]} — та же структура, что visual_qc.py
    # реально пишет (main()) и та же формула, что даёт его return 2.
    counts = data.get("counts", {})
    unresolved = sum(counts.get(v, 0) for v in _VISUAL_QC_UNRESOLVED_VERDICTS)
    return ("unresolved", unresolved) if unresolved else ("ok", 0)


def preflight_and_run(video_dir, strict, legacy_allow_degraded, legacy_allow_unreviewed_media=False,
                       legacy_allow_unreviewed_render=False):
    manifest = {"video_dir": video_dir, "strict_production": strict,
                "legacy_allow_degraded_timing": legacy_allow_degraded, "stages": {}}

    audio_path = os.path.join(video_dir, "audio.mp3")
    if not os.path.exists(audio_path):
        manifest["stages"]["audio"] = {"status": "missing"}
        _write_manifest(video_dir, manifest)
        print("  СТОП: audio.mp3 не найден — озвучка ещё не готова (Шаг 6 протокола).")
        return 1
    manifest["stages"]["audio"] = {"status": "ok"}

    vqc_status, vqc_unresolved = _visual_qc_status(video_dir)
    manifest["stages"]["visual_qc"] = {"status": vqc_status, "unresolved": vqc_unresolved}
    if vqc_status != "ok" and strict and not legacy_allow_unreviewed_media:
        _write_manifest(video_dir, manifest)
        if vqc_status == "missing_report":
            print("  СТОП (--strict-production): scripts/visual_qc.py ни разу не запускался "
                  "(Шаг 5.5 протокола, media_plan/visual_qc_report.json отсутствует) — "
                  "релевантность/качество media\\ не проверены. Запусти visual_qc.py <video_dir> "
                  "перед сборкой, или --legacy-allow-unreviewed-media для явного пропуска.")
        else:
            print(f"  СТОП (--strict-production): visual_qc_report.json — {vqc_unresolved} слот(ов) "
                  "требуют проверки глазами (missing/reject/accepted_below_threshold), Шаг 7.5 не "
                  "выполнен. Пересмотри media_plan/visual_qc_report.json, или "
                  "--legacy-allow-unreviewed-media для явного пропуска.")
        return 1

    n_sections = _section_count(video_dir)
    manifest["section_count"] = n_sections

    offsets_path = os.path.join(video_dir, "media_plan", "section_offsets.json")
    need_sync = n_sections >= 2 and not (os.path.exists(offsets_path)
                                          and _section_offsets_cover_all(video_dir, n_sections))
    if need_sync:
        rc = _run("section_sync.py", video_dir)
        covered = _section_offsets_cover_all(video_dir, n_sections)
        manifest["stages"]["section_sync"] = {"status": "ok" if covered else "partial",
                                               "exit_code": rc, "covers_all_sections": covered}
        if not covered and strict and not legacy_allow_degraded:
            _write_manifest(video_dir, manifest)
            print("  СТОП (--strict-production): section_sync.py не дал уверенного "
                  "смещения на все секции — синхронизация BLOCK1+ не гарантирована. "
                  "Перезапусти с --legacy-allow-degraded-timing, если это осознанный "
                  "компромисс, или пересмотри реальный audio.mp3/alignment.")
            return 1
    else:
        manifest["stages"]["section_sync"] = {"status": "skipped",
                                               "reason": ("single_section" if n_sections < 2
                                                          else "already_present")}

    rc = _run("fix_pauses.py", video_dir)
    manifest["stages"]["fix_pauses"] = {"status": "ok" if rc == 0 else "failed", "exit_code": rc}
    if rc != 0 and strict and not legacy_allow_degraded:
        _write_manifest(video_dir, manifest)
        print("  СТОП (--strict-production): fix_pauses.py завершился с ошибкой — "
              "аудио не обработано, дальше рендерить нечего.")
        return 1

    rc = _run("pipeline_smart.py", video_dir)
    manifest["stages"]["pipeline_smart"] = {"status": "ok" if rc == 0 else "failed", "exit_code": rc}
    if rc != 0:
        _write_manifest(video_dir, manifest)
        return rc

    # Пост-рендер отчёты (_POST_RENDER_REPORTS выше) — существуют только
    # ПОСЛЕ успешного pipeline_smart.py, поэтому эта проверка идёт здесь, а
    # не в preflight-блоке выше вместе с visual_qc (тот — ДО рендера,
    # отдельный обязательный шаг). final.mp4 при этом НЕ удаляется (тот же
    # принцип, что и у visual_qc "ниже порога" — честно пометить, не стереть
    # уже посчитанную работу), просто СБОРКА не считается чисто пройденной.
    pr_status, pr_total, pr_details = _post_render_status(video_dir)
    manifest["stages"]["post_render_qc"] = {"status": pr_status, "unresolved": pr_total,
                                             "details": pr_details}
    if pr_status == "unresolved" and strict and not legacy_allow_unreviewed_render:
        _write_manifest(video_dir, manifest)
        reasons = "; ".join(f"{d['report']}: {d['count']} ({d['reason']})" for d in pr_details)
        print(f"  СТОП (--strict-production): {pr_total} слот(ов) в пост-рендер отчётах требуют "
              f"внимания (конкретный следующий шаг — в скобках у каждого отчёта ниже) — "
              f"{reasons}. final.mp4 собран, но НЕ считается чисто "
              f"пройденным. Пересмотри отчёты в media_plan/, или --legacy-allow-unreviewed-render "
              f"для явного пропуска.")
        return 1

    _write_manifest(video_dir, manifest)
    return rc


def _write_manifest(video_dir, manifest):
    plan_dir = os.path.join(video_dir, "media_plan")
    os.makedirs(plan_dir, exist_ok=True)
    path = os.path.join(plan_dir, "timeline_manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video_dir")
    parser.add_argument("--strict-production", action="store_true")
    parser.add_argument("--legacy-allow-degraded-timing", action="store_true")
    parser.add_argument("--legacy-allow-unreviewed-media", action="store_true")
    parser.add_argument("--legacy-allow-unreviewed-render", action="store_true")
    args = parser.parse_args()
    return preflight_and_run(args.video_dir, args.strict_production, args.legacy_allow_degraded_timing,
                              legacy_allow_unreviewed_media=args.legacy_allow_unreviewed_media,
                              legacy_allow_unreviewed_render=args.legacy_allow_unreviewed_render)


if __name__ == "__main__":
    sys.exit(main())
