#!/usr/bin/env python3
"""Сборка final.mp4 из кадров генератора и озвучки.

Картинка (решения владельца и критиков 04.10, подробности — frame_clip.py,
shots.py, camera.py):
  * фон — кремовая бумага, 16:9 дополняется той же бумагой (рисунок на
    бумаге) или обрезается по центру рисунка (нарисовано место целиком);
    лёгкая фактура бумаги едет вместе с рисунком;
  * рисунок увеличивается нейросетью (upscale.py) — крупные планы резкие;
  * один рисунок держится 6-10 с, а план меняется каждые 2-4 с: подписи
    схемы появляются, когда голос их называет; быстрый наезд на предмет на
    его слове (не чаще раза в 15 с по ролику); длинный план — склейкой на
    средний план главного предмета; внутри плана движение не больше 4%;
  * главная мысль пишется карандашом (writeon.py) со звуком (pencil_sound.py),
    не чаще раза в 10 с; подписи — обычный жирный шрифт;
  * конец ролика — плавно в крем, не в чёрное.
Резы между кадрами — жёсткие, по кадровой сетке, ошибка округления
переносится на следующий клип; клип проверяется ffprobe. Отчёт по планам —
media_plan/shots_report.json.

Тайминг и звук — код старого генератора, логика без изменений
(speech_timing, audio_master, subtitles):
  * PHRASE LOCK: рез ровно на начале следующей фразы по alignment озвучки;
    не сошлось — оценка по реальной длине речи блоков и громко в логе;
  * голос: срез низов, EQ, де-эссер, компрессор; музыка (первый файл в
    assets/music, если есть) с уровнем по ЗАМЕРУ и приглушением под голос
    теми же параметрами; карандаш — на PENCIL_GAP_LU ниже голоса по замеру;
    двухпроходный loudnorm -14 LUFS + лимитер;
    проверка звука входа и готового файла;
  * субтитры (2 строки, ~42 символа) и главы YouTube.

Кадр, который генератор пометил rejected/failed или которого нет, на экран
не идёт — его время получает предыдущий кадр («ни карточек, ни повторов»).
Пишет media_plan/phrase_timeline.json в формате старого рендера — по нему
verify_timing.py меряет резы в пикселях готового файла.

Коды: 0 — чисто, 2 — собран с замечаниями, 1 — не собран.
Usage: python scripts/assemble_frames.py <video_dir>"""
import glob
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_WARN = 0, 1, 2
FPS, W, H = 24, 1920, 1080
CRF = "18"
CLIP_TOLERANCE = 0.5 / FPS


def run(cmd, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}...: {r.stderr[-800:]}")
    return r


def probe_duration(path):
    try:
        return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], 60).stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def quantize(durs):
    out, carry = [], 0.0
    for d in durs:
        target = d + carry
        q = max(1, round(target * FPS)) / FPS
        carry = target - q
        out.append(q)
    return out


KEY_GAP_SEC = 10.0     # главная мысль карандашом — не чаще раза в 10 с (решение владельца: 1 на 10-30 с)
WRITE_WORDS_MAX = 3    # брендбук: 1-3 рукописных ключевых слова за WRITE_WINDOW_SEC — мысль и акцент вместе
WRITE_WINDOW_SEC = 30.0


def frame_source(video_dir, i, rec):
    """Исходник кадра для сборки: чистая картинка генератора + подписи отдельно
    (появятся по словам), иначе готовый кадр с уже впечёнными подписями."""
    placed = rec.get("labels_placed") or []
    chosen = rec.get("chosen")
    raw = os.path.join(video_dir, "media_plan", "image_cache", chosen) if chosen else None
    if raw and os.path.exists(raw) and not any(r.get("fallback") for r in placed):
        return raw, placed
    return os.path.join(video_dir, "frames", f"{i + 1:03d}.png"), []


def load_plan_frames(video_dir):
    try:
        return {f["index"]: f for f in json.load(open(os.path.join(video_dir, "media_plan", "frame_plan.json"),
                                                      encoding="utf-8"))["frames"]}
    except (OSError, ValueError, KeyError):
        return {}


# ---- работа воркеров (процессы: письмо карандашом — чистый Python)
_TEX = {}


def _tex(path):
    if path not in _TEX:
        from PIL import Image
        _TEX[path] = Image.open(path).convert("L")
    return _TEX[path]


def writing_budget(written, T0, key, accent, reserved=0):
    """Брендбук: 1-3 рукописных ключевых слова за WRITE_WINDOW_SEC — главная мысль и
    акцент вместе, мысль важнее. written — [(глобальное время, число слов)] уже
    написанного; reserved — слова главных мыслей следующих кадров в пределах окна:
    акцент их не вытесняет (живой прогон 05.10: «пять минут» в начале съели место
    «только открыть»). Возвращает (key, accent, замечания)."""
    recent = sum(n for t, n in written if t > T0 - WRITE_WINDOW_SEC)
    notes = []
    if key and recent + len(key.split()) > WRITE_WORDS_MAX:
        notes.append(f"главная мысль «{key}» пропущена: за {WRITE_WINDOW_SEC:.0f} с на экране уже {recent} "
                     "рукописных слов")
        key = None
    if accent and recent + (len(key.split()) if key else 0) + reserved + len(accent.split()) > WRITE_WORDS_MAX:
        notes.append(f"акцент «{accent}» пропущен: лимит рукописных слов")
        accent = None
    return key, accent, notes


def _prepare(job):
    import frame_clip
    return frame_clip.prepare(job["src"], job["recs"], job["objects"], job["work"], seed=job["seed"],
                              tex=_tex(job["tex"]))


def _warm(job):
    _prepare(job)
    return True


def _render(job):
    import frame_clip
    fr = _prepare(job)
    tmp = job["out"]
    cues = frame_clip.render(fr, job["plan"], job["dur"], tmp, fps=FPS, end_fade=job["end_fade"], crf=CRF)
    d = probe_duration(tmp)
    if d is None or abs(d - round(job["dur"]*FPS)/FPS) > CLIP_TOLERANCE + 1e-6:
        os.remove(tmp)
        return None, []
    with open(tmp + ".cues.json", "w") as fh:
        json.dump(cues, fh)
    return tmp, cues


def load_report(video_dir):
    try:
        return {r["index"]: r for r in json.load(
            open(os.path.join(video_dir, "media_plan", "frames_report.json"), encoding="utf-8"))["frames"]}
    except (OSError, ValueError, KeyError):
        return {}


def kept_frames(video_dir, blocks):
    """Кадр идёт на экран, только если он нарисован под ЭТУ фразу: ключ фразы
    в отчёте совпал с текстом блока. Вставили фразу в сценарий — кадры не
    съезжают под чужие фразы, а честно отсутствуют до перерисовки."""
    from frame_planner import unit_key
    report = load_report(video_dir)
    kept, absorbed = [], []
    for i, b in enumerate(blocks):
        path = os.path.join(video_dir, "frames", f"{i + 1:03d}.png")
        rec = report.get(i) or {}
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            absorbed.append({"index": i, "reason": "no_frame"})
        elif rec.get("key") != unit_key(b["text"]):
            absorbed.append({"index": i, "reason": "frame_for_another_line"})
        elif rec.get("status") in ("rejected", "failed"):
            absorbed.append({"index": i, "reason": rec["status"]})
        else:
            kept.append(i)
    return kept, absorbed


def find_audio(video_dir):
    for name in ("audio_fixed.flac", "audio.mp3", "audio.wav", "audio.flac", "audio.m4a"):
        p = os.path.join(video_dir, name)
        if os.path.exists(p):
            return p
    return None


def music_file():
    files = sorted(f for f in glob.glob(os.path.join(env.ROOT, "assets", "music", "*"))
                   if f.lower().endswith((".mp3", ".flac", ".wav", ".ogg", ".m4a")))
    return files[0] if files else None


def main(video_dir):
    env.load_env()
    import audio_master as am
    import script_parser
    import shots
    import subtitles
    from speech_timing import SpeechTiming

    video_dir = os.path.abspath(video_dir)
    speech = SpeechTiming(video_dir)
    blocks = script_parser.parse_blocks(os.path.join(video_dir, "script.txt"))
    audio = find_audio(video_dir)
    if not audio:
        print("Нет озвучки (audio.mp3) в папке ролика")
        return EXIT_FAILED
    am.audio_qc(audio)
    total = am.get_media_duration(audio)

    kept, absorbed = kept_frames(video_dir, blocks)
    if not kept:
        print("Ни одного проверенного кадра — сборка остановлена (ролик из брака не собирается).")
        return EXIT_FAILED
    for a in absorbed:
        print(f"  [{a['index'] + 1}] кадра нет ({a['reason']}) — время отдано соседнему кадру")

    real_weights = speech.weights(blocks)
    onsets = speech.onsets(blocks)
    if onsets:
        starts, timing = [0.0] + list(onsets[1:]), "phrase_lock"
    else:
        f = speech.failure or {"reason": "alignment не найден"}
        print(f"  ВНИМАНИЕ: PHRASE LOCK ВЫКЛЮЧЕН — {f['reason']}. Кадры по ОЦЕНКЕ длины речи.")
        words = [max(1, b["words"]) + 2 * b.get("pause_after", 0.0) for b in blocks]
        starts, acc = [], 0.0
        for w in words:
            starts.append(acc)
            acc += total * w / sum(words)
        timing = "estimate"
    base = [(starts[i + 1] if i + 1 < len(starts) else total) - starts[i] for i in range(len(starts))]

    kept_starts = [0.0] + [starts[i] for i in kept[1:]]
    durs = quantize([e - s for s, e in zip(kept_starts, kept_starts[1:] + [total])])
    print(f"Тайминг: {timing}; кадров на экране {len(kept)} из {len(blocks)}, средний {sum(durs) / len(durs):.1f} с")

    mp = os.path.join(video_dir, "media_plan")
    os.makedirs(mp, exist_ok=True)
    with open(os.path.join(mp, "phrase_timeline.json"), "w", encoding="utf-8") as fh:
        json.dump({"locked": timing == "phrase_lock", "fps": FPS, "audio_total_sec": total,
                   "blocks": [{"index": i, "section": blocks[i]["section"], "text": blocks[i]["text"][:120],
                               "speech_onset_sec": kept_starts[k] if timing == "phrase_lock" else None,
                               "duration_sec": durs[k]} for k, i in enumerate(kept)]},
                  fh, ensure_ascii=False, indent=2)
    subtitles.write_subtitles(video_dir, blocks, starts, base, real_weights=real_weights)
    subtitles.write_chapters(video_dir, blocks, starts)

    work = os.path.join(video_dir, "temp_render")
    os.makedirs(work, exist_ok=True)
    import canvas
    import frame_clip
    tex_path = os.path.join(work, f"paper_texture_{canvas.TEX_W}.png")
    if not os.path.exists(tex_path):
        canvas.paper_texture().save(tex_path + ".tmp.png")
        os.replace(tex_path + ".tmp.png", tex_path)

    report = load_report(video_dir)
    plan_frames = load_plan_frames(video_dir)
    word_times = speech.word_times if timing == "phrase_lock" else []
    jobs = []
    for k, i in enumerate(kept):
        rec = report.get(i) or {}
        src, recs = frame_source(video_dir, i, rec)
        nxt = kept[k + 1] if k + 1 < len(kept) else len(blocks)
        words = [{"word": w["word"], "start": w["start"] - kept_starts[k], "end": w["end"] - kept_starts[k]}
                 for b in range(i, nxt) if b < len(word_times) for w in word_times[b]]
        pf = plan_frames.get(i) or {}
        jobs.append(dict(src=src, recs=recs, objects=rec.get("objects") or [], work=work, tex=tex_path,
                         seed=k, key=(pf.get("key_thought") or "").strip() or None, words=words,
                         accent=(pf.get("accent") or "").strip() or None,
                         dur=durs[k], T0=kept_starts[k], end_fade=(k == len(kept) - 1)))

    workers = int(os.environ.get("RENDER_WORKERS", str(max(1, min(4, (os.cpu_count() or 2) - 1)))))
    with ProcessPoolExecutor(workers) as ex:              # увеличение рисунков — дорого, параллельно и с кэшем
        list(ex.map(_warm, jobs))
    last_punch, last_key, shot_log, written = -1e9, -1e9, [], []
    for k, job in enumerate(jobs):                         # план — по порядку: наезды и мысли не чаще порога
        fr = _prepare(job)
        key = job["key"] if job["key"] and job["T0"] - last_key >= KEY_GAP_SEC else None
        if job["key"] and not key:
            shot_log.append({"index": kept[k], "note": f"главная мысль «{job['key']}» пропущена: прошлая "
                                                      f"{job['T0'] - last_key:.1f} с назад"})
        reserved = sum(len(j["key"].split()) for j in jobs[k + 1:]
                       if j.get("key") and j["T0"] < job["T0"] + WRITE_WINDOW_SEC)
        key, accent, why = writing_budget(written, job["T0"], key, job.get("accent"), reserved)
        shot_log += [{"index": kept[k], "note": n} for n in why]
        p = frame_clip.plan_clip(fr, job["dur"], job["words"], key=key, last_punch=last_punch,
                                 T0=job["T0"], zoom_in=(k % 2 == 0), fps=FPS, accent=accent)
        if p["punch_at"] is not None:
            last_punch = p["punch_at"]
        if p.get("key_time") is not None and p.get("key"):
            last_key = job["T0"] + p["key_time"]
            written.append((last_key, len(p["key"]["text"].split())))
        else:
            p["key_time"] = None
        if p.get("accent_time") is not None and p.get("accent"):
            written.append((job["T0"] + p["accent_time"], len(p["accent"]["text"].split())))
        job["plan"] = p
        h = hashlib.sha256(open(job["src"], "rb").read()).hexdigest()[:12]
        sig = hashlib.sha256(json.dumps([h, job["recs"], job["objects"], frame_clip.plan_record(p), job["dur"],
                                         job["end_fade"], job["seed"], FPS, CRF, frame_clip.RENDER_VERSION,
                                         canvas.TEXTURE, frame_clip.UPSCALE],
                                        sort_keys=True, default=str).encode()).hexdigest()[:16]
        job["out"] = os.path.join(work, f"clip_{k:04d}_{sig}.mp4")
        shot_log.append({"index": kept[k], "duration": round(job["dur"], 3),
                         "views": [[round(s_["t0"], 2), "lean" if s_.get("lean") else ("assemble" if s_.get("pushes")
                                                                                     else s_["kind"])]
                                   for s_ in p["segments"]],
                         "labels_at": [round(t, 2) for t in p["label_times"]],
                         "punch": p.get("punch_name"), "key": (p.get("key") or {}).get("text") if p.get("key_time") is not None else None,
                         "key_at": None if p.get("key_time") is None else round(p["key_time"], 2),
                         "accent": (p.get("accent") or {}).get("text") if p.get("accent_time") is not None else None,
                         "notes": p["notes"]})

    def cached(job):
        c = job["out"] + ".cues.json"
        if os.path.exists(job["out"]) and os.path.exists(c) and probe_duration(job["out"]) is not None:
            return job["out"], json.load(open(c))
        return None

    results = [cached(j) for j in jobs]
    todo = [k for k, r in enumerate(results) if r is None]
    with ProcessPoolExecutor(workers) as ex:
        for k, r in zip(todo, ex.map(_render, [jobs[k] for k in todo])):
            results[k] = r
    clips = [r[0] if r else None for r in results]
    cues = [(jobs[k]["T0"] + a, jobs[k]["T0"] + b, kind, 0.0) for k, r in enumerate(results) if r
            for a, b, kind in r[1]]
    with open(os.path.join(mp, "shots_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"punch_gap_sec": shots.PUNCH_GAP_SEC, "key_gap_sec": KEY_GAP_SEC, "clips": shot_log},
                  fh, ensure_ascii=False, indent=1)
    views = sum(len(c.get("views", [])) for c in shot_log)
    print(f"Планов на экране: {views} на {len(kept)} кадров (в среднем {sum(durs)/max(1, views):.1f} с), "
          f"наездов {sum(1 for c in shot_log if c.get('punch'))}, мыслей карандашом "
          f"{sum(1 for c in shot_log if c.get('key'))}")
    failed = [kept[k] + 1 for k, c in enumerate(clips) if not c]
    with open(os.path.join(mp, "render_manifest.json"), "w", encoding="utf-8") as fh:
        status = {i: {"index": i, "status": "ok" if clips[k] else "failed", "dur": round(durs[k], 4)}
                  for k, i in enumerate(kept)}
        status.update({a["index"]: {"index": a["index"], "status": "absorbed", "reason": a["reason"]}
                       for a in absorbed})
        json.dump({"total_blocks": len(blocks), "timing": timing, "clips": [status[i] for i in sorted(status)]},
                  fh, ensure_ascii=False, indent=1)
    if failed:
        print(f"СТОП: клипы не собраны {failed} — final.mp4 НЕ собирается (перезапуск достроит только их).")
        return EXIT_FAILED

    lst = os.path.join(work, "concat.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        fh.writelines(f"file '{c}'\n" for c in clips)
    video = os.path.join(work, "video.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video])

    # --- звук: цепочка старого генератора
    voice = am.process_voice(audio, os.path.join(work, "voice_processed.wav"))
    music = music_file()
    premix = voice
    if music:
        gain, why = am.music_bed_gain_db(voice, music)
        premix = os.path.join(work, "premix.wav")
        run(["ffmpeg", "-y", "-v", "error", "-i", voice, "-stream_loop", "-1", "-i", music, "-filter_complex",
             f"[1:a]atrim=0:{total:.3f},volume={gain:.2f}dB[m];[0:a]asplit=2[v][sc];"
             f"[m][sc]sidechaincompress=threshold={am.MUSIC_DUCK_THRESHOLD}:ratio={am.MUSIC_DUCK_RATIO}:"
             f"attack={am.MUSIC_DUCK_ATTACK_MS}:release={am.MUSIC_DUCK_RELEASE_MS}[md];"
             f"[v][md]amix=inputs=2:duration=first:normalize=0[a]", "-map", "[a]", "-ar", "48000", premix])
        print(f"  Музыка: {os.path.basename(music)}, {gain:+.1f} дБ ({why})")
    if cues:
        import pencil_sound
        trk = pencil_sound.track(cues, total)
        pencil = os.path.join(work, "pencil.wav")
        pencil_sound.write_wav(trk, pencil)
        pg, why = pencil_sound.gain_for(voice, pencil)
        if pg is None:
            print(f"  ВНИМАНИЕ: звук карандаша не сведён — {why}")
        else:
            mixed = os.path.join(work, "premix_pencil.wav")
            run(["ffmpeg", "-y", "-v", "error", "-i", premix, "-i", pencil, "-filter_complex",
                 f"[1:a]volume={pg:.2f}dB,aformat=channel_layouts=mono[p];[0:a][p]amix=inputs=2:duration=first:"
                 f"normalize=0[a]", "-map", "[a]", "-ar", "48000", mixed])
            premix = mixed
            print(f"  Карандаш: {len(cues)} штрихов, {pg:+.1f} дБ ({why})")
    af = am.build_master_af(am.measure_loudnorm_stats(premix), max(0.0, total - 2.0), 0.05)
    final = os.path.join(video_dir, "final.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", premix, "-af", af, "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
         "-movflags", "+faststart", final + ".tmp.mp4"])
    os.replace(final + ".tmp.mp4", final)
    am.audio_qc(final, label="Audio QC финала")
    print(f"Готово. Файл: {final}")
    return EXIT_WARN if absorbed or timing != "phrase_lock" else EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
