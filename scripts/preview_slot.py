"""Быстрое превью ОДНОГО слота эпизода — без полной пересборки final.mp4.

Проблема, которую решает: правка одного клипа (другая картинка/motion_mode)
требует полной пересборки xfade-цепочки всех 165 клипов (~25-30 минут),
даже если проверить нужно ровно один кадр. На реальной сессии (01_ves-mecha,
слот 0) это стоило три прогона подряд — каждый раз ради взгляда на 1.8
секунды видео.

Что делает этот скрипт: рендерит указанный слот через РЕАЛЬНЫЕ kenburns()/
video_render() (тот же код, что и main(), тот же грейд/зерно/деффликер —
не переизобретает конвейер), плюс — если у соседних слотов уже есть
готовые кэшированные клипы — короткую xfade-склейку слот-1/слот/слот+1,
чтобы увидеть переход в контексте. Секунды, не 25 минут.

НЕ заменяет полную сборку — только визуальная проверка перед тем, как
тратить время на настоящий рендер. Не трогает video_dir/temp_smart кроме
записи предпросмотра в отдельную папку.

Использование:
    python scripts/preview_slot.py <video_dir> <index> [--photo PATH] [--motion MODE]

Пример (проверить слот 0 с новым фото и static_hold ДО настоящей пересборки):
    python scripts/preview_slot.py videos/01_ves-mecha 0 \\
        --photo temp_smart/pexels_cache/0000_manual_witcher_wide.jpg \\
        --motion static_hold
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir")
    ap.add_argument("index", type=int)
    ap.add_argument("--photo", help="переопределить источник фото для этого слота")
    ap.add_argument("--motion", help="переопределить motion_mode (classic_kb/static_hold/snap_push/slow_pull/horizontal_pan/micro_drift)")
    ap.add_argument("--with-neighbors", action="store_true",
                     help="дополнительно собрать короткий xfade slot-1/slot/slot+1, если соседи уже отрендерены")
    args = ap.parse_args()

    sys.argv = ["pipeline_smart.py", args.video_dir]
    import pipeline_smart as ps

    manifest_path = os.path.join(args.video_dir, "media_plan", "render_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    clips_by_index = {c["index"]: c for c in manifest["clips"]}
    entry = clips_by_index.get(args.index)
    if entry is None:
        print(f"Слот {args.index} не найден в {manifest_path}")
        return 1
    dur = entry["duration"]
    section = entry.get("section", "")

    with open(os.path.join(args.video_dir, "media_plan", "shot_manifest.json"), encoding="utf-8") as f:
        shots = {s["index"]: s for s in json.load(f)}
    text = shots.get(args.index, {}).get("text_preview", "")

    out_dir = os.path.join(args.video_dir, "temp_smart", "_slot_preview")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"slot_{args.index:04d}_preview.mp4")

    photo = args.photo
    if photo and not os.path.isabs(photo):
        photo = os.path.join(args.video_dir, photo)

    motion = args.motion or "classic_kb"
    if photo:
        ok = ps.kenburns(photo, out_path, dur, section=section, motion_mode=motion)
    else:
        # Без --photo — переиспользовать уже отрендеренный клип как есть
        # (просто копия для единообразного пути превью).
        existing = os.path.join(args.video_dir, entry["path"]) if not os.path.isabs(entry["path"]) else entry["path"]
        if not os.path.exists(existing):
            print(f"Нет ни --photo, ни готового клипа {existing}")
            return 1
        import shutil
        shutil.copyfile(existing, out_path)
        ok = True

    if not ok:
        print("Рендер превью не удался")
        return 1

    print(f"Превью слота {args.index} ({dur:.2f}с, motion={motion}): {out_path}")
    print(f"Текст: {text!r}")

    if args.with_neighbors:
        neighbor_clips = []
        for j in (args.index - 1, args.index, args.index + 1):
            if j == args.index:
                neighbor_clips.append((out_path, dur))
                continue
            nentry = clips_by_index.get(j)
            if nentry is None:
                continue
            npath = os.path.join(args.video_dir, nentry["path"]) if not os.path.isabs(nentry["path"]) else nentry["path"]
            if os.path.exists(npath):
                neighbor_clips.append((npath, nentry["duration"]))
        if len(neighbor_clips) >= 2:
            ctx_out = os.path.join(out_dir, f"slot_{args.index:04d}_context.mp4")
            paths = [c[0] for c in neighbor_clips]
            durs = [c[1] for c in neighbor_clips]
            secs = [section] * len(paths)
            ok2, _ = ps.xfade_chain(paths, durs, secs, ctx_out, xfade_dur=ps.XFADE_DUR)
            if ok2:
                print(f"Контекст (соседние клипы + переходы): {ctx_out}")
            else:
                print("Не удалось собрать контекст с соседями (xfade не встал)")
        else:
            print("Соседние клипы ещё не отрендерены — контекст пропущен")

    return 0


if __name__ == "__main__":
    sys.exit(main())
