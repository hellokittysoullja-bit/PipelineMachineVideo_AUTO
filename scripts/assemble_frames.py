#!/usr/bin/env python3
"""Сборка final.mp4 — на рендере старого генератора (render_core), а не на
своём упрощённом.

Что делает старая цепочка (код render_core, дословно из pipeline_smart.py):
  * тайминг — PHRASE LOCK: рез ровно на начале следующей фразы по
    посимвольному alignment (load_alignment_onsets, с картой вырезанных пауз
    fix_pauses.py и смещениями секций); не сошлось — оценка по реальной
    длине речи блоков (block_durations) и громко в логе;
  * камера — choose_motion_mode (статика, дрифт, отъезд, наезд, панорама) по
    стадии рассказа (camera_language) и защита от повтора пары
    (режим, направление) — pick_direction;
  * клип — kenburns() (вписывание 16:9, zoompan на всю длину, проверка клипа
    ffprobe и повторы рендера), переходы — xfade_chain_chunked, хвост —
    pad_to_length;
  * звук — process_voice (срез низов, EQ, де-эссер, компрессор), музыка с
    уровнем по замеру и приглушением под голос, атмосфера, SFX-режиссёр,
    двухпроходный loudnorm + лимитер, audio_qc готового файла;
  * субтитры (2 строки, ~42 символа) и главы YouTube.

Что своё (природа рисованного кадра):
  * грейд film_look выключен (FILM_LOOK=0, см. render_core) — он тёмный
    киношный и портит белый лист;
  * схема и кадр с подписью всегда идут спокойным движением (micro_drift):
    подписи должны читаться, а наезд уводит их за край;
  * кадр, который генератор пометил rejected/failed, на экран не идёт — его
    время получает предыдущий проверенный кадр (NEVER_SHOW_KNOWN_BAD
    старого генератора: «ни карточек, ни повторов»).

Коды возврата как у старого рендера: 0 — чисто, 2 — собран с замечаниями,
1 — не собран.

Usage: python scripts/assemble_frames.py <video_dir>"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_WARN = 0, 1, 2
CALM_KINDS = ("diagram", "caption")
CALM_MODE = "micro_drift"


def load_plan(video_dir):
    try:
        return {f["index"]: f for f in json.load(open(os.path.join(video_dir, "media_plan", "frame_plan.json"),
                                                        encoding="utf-8"))["frames"]}
    except (OSError, ValueError, KeyError):
        return {}


def load_statuses(video_dir):
    try:
        return {r["index"]: r.get("status") for r in json.load(
            open(os.path.join(video_dir, "media_plan", "frames_report.json"), encoding="utf-8"))["frames"]}
    except (OSError, ValueError, KeyError):
        return {}


def kept_frames(video_dir, n):
    """Индексы блоков, чей кадр идёт на экран, и отчёт о поглощённых."""
    statuses = load_statuses(video_dir)
    kept, absorbed = [], []
    for i in range(n):
        path = os.path.join(video_dir, "frames", f"{i + 1:03d}.png")
        st = statuses.get(i)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            absorbed.append({"index": i, "reason": "no_frame"})
        elif st in ("rejected", "failed"):
            absorbed.append({"index": i, "reason": st})
        else:
            kept.append(i)
    return kept, absorbed


def main(video_dir):
    channel.load_env()
    import camera_language
    import feature_flags
    import render_core as rc
    import script_parser
    import shot_brief_director as sbd

    video_dir = os.path.abspath(video_dir)
    rc.configure(video_dir)
    blocks = script_parser.parse_blocks(os.path.join(video_dir, "script.txt"))
    if not rc.AUDIO_FILE or not os.path.exists(rc.AUDIO_FILE):
        print("Нет озвучки (audio.mp3) в папке ролика")
        return EXIT_FAILED
    rc.check_ffmpeg_filters()
    rc.audio_qc(rc.AUDIO_FILE)
    total = rc.get_media_duration(rc.AUDIO_FILE)

    kept, absorbed = kept_frames(video_dir, len(blocks))
    if not kept:
        print("Ни одного проверенного кадра — сборка остановлена (ролик из брака не собирается).")
        return EXIT_FAILED
    for a in absorbed:
        print(f"  [{a['index'] + 1}] кадра нет ({a['reason']}) — время отдано соседнему кадру")

    # --- тайминг: онсеты фраз по alignment (PHRASE LOCK), иначе оценка
    real_weights = rc.load_alignment_weights(blocks)
    onsets = rc.load_alignment_onsets(blocks)
    if onsets:
        sub_starts = [0.0] + list(onsets[1:])
        timing = "phrase_lock"
    else:
        f = rc.ALIGNMENT_ONSET_FAILURE or {"reason": "alignment не найден"}
        print(f"  ВНИМАНИЕ: PHRASE LOCK ВЫКЛЮЧЕН — {f['reason']}. Кадры по ОЦЕНКЕ длины речи.")
        base = rc.block_durations(blocks, total, real_weights=real_weights)
        sub_starts, acc = [], 0.0
        for d in base:
            sub_starts.append(acc)
            acc += d
        timing = "estimate"
    sub_baseline = [(sub_starts[i + 1] if i + 1 < len(sub_starts) else total) - sub_starts[i]
                    for i in range(len(sub_starts))]

    kb = [blocks[i] for i in kept]
    sections = [b["section"] for b in kb]
    kept_onsets = [0.0] + [sub_starts[i] for i in kept[1:]]
    plan_tr = rc.effective_transition_plan(rc.plan_transitions(sections, kb), sections)
    durs = rc.phrase_locked_durations(kept_onsets, total, plan_tr)
    if not durs:
        # старты кадров не дали монотонной шкалы — длительности по стартам, по кадровой сетке
        ends = kept_onsets[1:] + [total]
        durs = rc.quantize_durations_to_frames([max(1.0 / rc.FPS, e - s) for s, e in zip(kept_onsets, ends)])
    print(f"Тайминг: {timing}; кадров на экране {len(kept)} из {len(blocks)}, "
          f"средний {sum(durs) / len(durs):.1f} с")

    # Карта фраз — тот же формат, что пишет старый рендер: по ней
    # verify_timing.py меряет резы в пикселях готового файла против онсетов
    # речи. Только кадры на экране: у поглощённого кадра реза нет.
    timeline = {"locked": timing == "phrase_lock", "fps": rc.FPS, "audio_total_sec": total,
                "blocks": [{"index": i, "section": blocks[i]["section"], "text": blocks[i]["text"][:120],
                            "speech_onset_sec": (kept_onsets[k] if timing == "phrase_lock" else None),
                            "duration_sec": durs[k]} for k, i in enumerate(kept)]}
    os.makedirs(os.path.join(video_dir, "media_plan"), exist_ok=True)
    with open(os.path.join(video_dir, "media_plan", "phrase_timeline.json"), "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    rc.write_subtitles(video_dir, blocks, sub_starts, sub_baseline, real_weights=real_weights)
    rc.write_chapters(video_dir, blocks, sub_starts)

    # --- камера и клипы
    plan = load_plan(video_dir)
    stages = sbd.arc_stages(video_dir)
    zoom_hist, zoom_pair_hist, pan_hist, jobs = [], [], [], []
    for k, i in enumerate(kept):
        b, photo = blocks[i], os.path.join(video_dir, "frames", f"{i + 1:03d}.png")
        is_section_start = k == 0 or kb[k - 1]["section"] != b["section"]
        photo_hash, zi_cand, pd_cand = rc.kb_hash_choices(photo)
        stage = stages.get(" ".join(b["text"].split()))
        kind = (plan.get(i) or {}).get("kind")
        mode = CALM_MODE if kind in CALM_KINDS else rc.choose_motion_mode(
            b, is_section_start, photo_hash, arc_stage=stage)
        stage_zi = camera_language.stage_zoom_in(stage) if stage and feature_flags.enabled("CAMERA_LANGUAGE") else None
        want = zi_cand if stage_zi is None else stage_zi
        if feature_flags.enabled("CAMERA_LANGUAGE"):
            zoom_in = camera_language.pick_direction(zoom_pair_hist, mode, want, max_repeat=2)
        else:
            zoom_in = rc.pick_no_repeat(zoom_hist, want, [True, False], max_repeat=2)
        pan_dir = rc.pick_no_repeat(pan_hist, pd_cand, rc.PAN_DIRECTIONS, max_repeat=2)
        out = os.path.join(rc.TEMP_FOLDER, f"clip_{k:04d}.mp4")
        jobs.append((k, i, photo, out, durs[k], b["section"], mode, zoom_in, pan_dir))

    def render(job):
        k, i, photo, out, d, section, mode, zoom_in, pan_dir = job
        try:
            return rc.kenburns(photo, out, d, zoom_in=zoom_in, pan_dir=pan_dir, section=section, motion_mode=mode)
        except Exception as e:  # noqa: BLE001 — сбой одного клипа фиксируется, сборка решает ниже
            print(f"  [{i + 1}] сбой рендера: {type(e).__name__}: {e}")
            return False

    workers = int(os.environ.get("RENDER_WORKERS", str(max(1, (os.cpu_count() or 2) - 1))))
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(render, jobs))
    failed = [jobs[n][1] + 1 for n, ok in enumerate(results) if not ok]
    # Формат старого рендера (index/status по каждому блоку): его читают
    # segment_report.py и прочие отчёты. Первая версия писала свой формат, и
    # отчёт по готовому файлу засчитал все клипы упавшими.
    clips_m = {j[1]: {"index": j[1], "status": "ok" if r else "failed", "path": j[3],
                      "dur": round(j[4], 4), "motion_mode": j[6]} for j, r in zip(jobs, results)}
    for a in absorbed:
        clips_m[a["index"]] = {"index": a["index"], "status": "absorbed", "reason": a["reason"]}
    manifest = {"total_blocks": len(blocks), "ok": sum(1 for r in results if r), "timing": timing,
                "missing": [a["index"] for a in absorbed if a["reason"] == "no_frame"],
                "clips": [clips_m[i] for i in sorted(clips_m)]}
    os.makedirs(os.path.join(video_dir, "media_plan"), exist_ok=True)
    with open(os.path.join(video_dir, "media_plan", "render_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    if failed:
        print(f"СТОП: клипы не собраны {failed} — final.mp4 НЕ собирается "
              f"(перезапуск достроит только их, остальное из кэша).")
        return EXIT_FAILED

    clips = [j[3] for j in jobs]
    merged = os.path.join(rc.TEMP_FOLDER, "merged.mp4")
    ok, _xt = rc.xfade_chain_chunked(clips, durs, sections, merged, rc.TEMP_FOLDER, blocks=kb)
    if not ok:
        print("Склейка переходами не удалась — финал не собран.")
        return EXIT_FAILED
    merged, _pad_gap = rc.pad_to_length(merged, total, rc.TEMP_FOLDER)

    # --- звук: та же цепочка, что в старом рендере
    hook_end, final_start = rc.section_audio_bounds(blocks, sub_starts, total)
    voice = rc.process_voice(rc.AUDIO_FILE, os.path.join(rc.TEMP_FOLDER, "voice_processed.wav"))
    premix = rc.build_episode_audio_layers(voice, video_dir, rc.TEMP_FOLDER, blocks, sub_starts, real_weights,
                                           total, hook_end, final_start, phrase_locked=bool(onsets))
    stats = rc.measure_loudnorm_stats(premix)
    af = rc.build_master_af(stats, max(0.0, total - 2.0), 0.05)
    tmp = rc.render_tmp_path(rc.OUTPUT_FILE)
    frames_target = min(round(total * rc.FPS), round(rc.get_media_duration(merged) * rc.FPS))
    r = subprocess.run(["ffmpeg", "-y", "-i", merged, "-i", premix, "-t", f"{total:.3f}", "-af", af,
                        "-c:v", "copy", "-vframes", str(frames_target), "-c:a", "aac", "-b:a", "192k",
                        "-ar", "48000", "-movflags", "+faststart", tmp],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=max(180, total))
    if r.returncode != 0:
        print("Финальный мукс:", r.stderr[-400:])
        rc.finalize_render(tmp, rc.OUTPUT_FILE, False)
        return EXIT_FAILED
    rc.finalize_render(tmp, rc.OUTPUT_FILE, True)
    rc.audio_qc(rc.OUTPUT_FILE, label="Audio QC финала")
    rc.write_audio_master_report(video_dir)
    print(f"Готово. Файл: {rc.OUTPUT_FILE}")
    return EXIT_WARN if absorbed or timing != "phrase_lock" else EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
