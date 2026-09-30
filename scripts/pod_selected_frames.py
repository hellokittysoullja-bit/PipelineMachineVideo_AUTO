#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Выбранные кадры — по ходу прогона, а не в конце.

Читает media_plan/run_journal.live.jsonl (пайплайн дописывает его в момент
решения о слоте) и на каждую принятую попытку кладёт уменьшенную копию
кадра в media_plan/selected/: фото — картинка 960 px, видео — три кадра
(20/50/80 %) в ряд. Рядом index.json: слот, вид, файл-источник, вердикты.

Зачем: живой прогон 30.09 потерял под посреди отбора, и кадры уже
выбранных слотов пропали вместе с ним — контактный лист собирается только
после всех слотов. media_plan забирается с пода по ходу (runpod_job --sync),
так что кадры принятых слотов остаются у нас при любом обрыве.

Отбор не трогает: только читает журнал и файлы, пишет в свою папку.
Запуск на поде рядом с пайплайном:
    python scripts/pod_selected_frames.py videos/NN_название &
Завершается сам, когда в media_plan появляется shotlist.json новее старта,
или по --once (один проход — для тестов)."""
import json
import os
import subprocess
import sys
import time

THUMB_W = 960
VIDEO_FRACS = (0.2, 0.5, 0.8)
POLL_SEC = 10


def _video_duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", path], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def thumb_photo(src, dst):
    from PIL import Image, ImageOps
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((THUMB_W, THUMB_W * 2))
        im.save(dst, "JPEG", quality=85)


def thumb_video(src, dst):
    from PIL import Image
    dur = _video_duration(src)
    if not dur:
        raise RuntimeError("ffprobe не прочитал длительность")
    tiles = []
    for k, frac in enumerate(VIDEO_FRACS):
        tmp = f"{dst}.{k}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{dur * frac:.3f}", "-i", src,
                        "-frames:v", "1", "-vf", f"scale={THUMB_W // 3}:-2", tmp],
                       check=True, capture_output=True, timeout=60)
        tiles.append(Image.open(tmp).convert("RGB"))
    h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (sum(t.width for t in tiles), h), (0, 0, 0))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width
    sheet.save(dst, "JPEG", quality=85)
    for k in range(len(tiles)):
        try:
            os.remove(f"{dst}.{k}.jpg")
        except OSError:
            pass


def committed(journal_path):
    """Принятые попытки журнала: [(index, kind, media, attempt_id, verdicts)]."""
    out = []
    try:
        with open(journal_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue      # строка дописывается прямо сейчас
                if rec.get("record") == "attempt" and rec.get("media"):
                    out.append((rec["index"], rec.get("kind"), rec["media"], rec.get("attempt_id"),
                                rec.get("verdicts") or []))
    except OSError:
        pass
    return out


def one_pass(video_dir, done):
    """Сделать копии для новых принятых попыток. done — множество attempt_id."""
    plan = os.path.join(video_dir, "media_plan")
    out_dir = os.path.join(plan, "selected")
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.json")
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        index = []
    made = 0
    for idx, kind, media, att_id, verdicts in committed(os.path.join(plan, "run_journal.live.jsonl")):
        if att_id in done:
            continue
        src = media if os.path.isabs(media) else os.path.join(video_dir, media)
        dst = os.path.join(out_dir, f"{int(idx):02d}_{kind or 'media'}.jpg")
        entry = {"index": idx, "kind": kind, "media": media, "attempt_id": att_id,
                 "thumb": os.path.basename(dst), "verdicts": verdicts}
        try:
            (thumb_video if kind == "video" else thumb_photo)(src, dst)
        except Exception as e:  # noqa: BLE001 — один кадр не останавливает остальные
            entry["thumb"] = None
            entry["error"] = f"{type(e).__name__}: {e}"[:300]
        index = [e for e in index if e.get("attempt_id") != att_id] + [entry]
        done.add(att_id)
        made += 1
    if made:
        tmp = index_path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sorted(index, key=lambda e: e["index"]), f, ensure_ascii=False, indent=1)
        os.replace(tmp, index_path)
    return made


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    once = "--once" in argv
    argv = [a for a in argv if a != "--once"]
    video_dir = argv[0]
    t0 = time.time()
    done = set()
    shotlist = os.path.join(video_dir, "media_plan", "shotlist.json")
    while True:
        one_pass(video_dir, done)
        if once:
            return 0
        try:
            if os.path.getmtime(shotlist) > t0:
                one_pass(video_dir, done)      # последние слоты после конца отбора
                return 0
        except OSError:
            pass
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())
