"""Контактный лист шотлиста — сетка «кадр + фраза» по media_plan/shotlist.json.

Зачем: Шаг 7.5 протокола (CLAUDE.md) требует пройтись глазами по
релевантности КАЖДОГО кадра, но до этого скрипта единственный способ увидеть
выбор пайплайна — досмотреть готовый final.mp4 или вручную резать кадры
ffmpeg'ом. Здесь — одна картинка на 24 слота: номер, секция, фраза блока и
сам кадр (для видео — кадр-пробник с 0.5с). Плохой кадр -> в shotlist.json
у этого слота `"lock": true` + свой `"file"` -> следующий прогон берёт его
как есть (см. shotlist_locked_media в pipeline_smart.py).

Использование:
    python scripts/shotlist_contact.py <video_dir> [--cols 4] [--per-page 24]

Результат: <video_dir>/media_plan/shotlist_contact_01.jpg, _02.jpg, ...
Не трогает ни shotlist.json, ни кэш, ни рендер — только читает.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

THUMB_W, THUMB_H = 480, 270
CAPTION_H = 74
PAD = 10
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v")

# Кириллица обязательна (фразы сценария) — Benzin-Bold из assets/fonts её не
# гарантирует, поэтому сначала системные шрифты с полным покрытием.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:   # Pillow < 10.1
        return ImageFont.load_default()


def resolve_file(file_value, video_dir):
    if not file_value:
        return None
    return file_value if os.path.isabs(file_value) else os.path.normpath(os.path.join(video_dir, file_value))


def thumbnail_for(path):
    """PIL-картинка THUMB_W x THUMB_H (вписана, чёрные поля) или None."""
    if not path or not os.path.exists(path):
        return None
    src = None
    tmp = None
    try:
        if path.lower().endswith(VIDEO_EXTS):
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "0.5", "-i", path,
                                "-frames:v", "1", "-q:v", "4", tmp],
                               capture_output=True, timeout=30)
            if r.returncode != 0 or not os.path.getsize(tmp):
                return None
            src = Image.open(tmp).convert("RGB")
        else:
            src = Image.open(path).convert("RGB")
        src.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
        canvas = Image.new("RGB", (THUMB_W, THUMB_H), (16, 16, 16))
        canvas.paste(src, ((THUMB_W - src.width) // 2, (THUMB_H - src.height) // 2))
        return canvas
    except Exception:
        return None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def wrap_text(text, font, max_w, draw, max_lines=3):
    words = (text or "").split()
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if draw.textlength(cand, font=font) <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) == max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(words) and len(lines) == max_lines and " ".join(lines) != " ".join(words):
        lines[-1] = lines[-1][:max(0, len(lines[-1]) - 1)] + "…"
    return lines


def render_page(shots, video_dir, cols, out_path):
    rows = (len(shots) + cols - 1) // cols
    cell_w, cell_h = THUMB_W + PAD, THUMB_H + CAPTION_H + PAD
    page = Image.new("RGB", (cols * cell_w + PAD, rows * cell_h + PAD), (28, 28, 28))
    draw = ImageDraw.Draw(page)
    font_head = load_font(17)
    font_text = load_font(14)
    for k, shot in enumerate(shots):
        x0 = PAD + (k % cols) * cell_w
        y0 = PAD + (k // cols) * cell_h
        thumb = thumbnail_for(resolve_file(shot.get("file"), video_dir))
        if thumb is None:
            thumb = Image.new("RGB", (THUMB_W, THUMB_H), (70, 20, 20))
            ImageDraw.Draw(thumb).text((14, 14), "НЕТ ФАЙЛА / не прочитан", fill=(255, 200, 200), font=font_head)
        page.paste(thumb, (x0, y0))
        lock = " 🔒" if shot.get("lock") else ""
        head = f"#{shot.get('index', 0) + 1}  {shot.get('section', '')}  [{shot.get('kind') or '—'}/{shot.get('source', '')}]{lock}"
        color = (255, 220, 120) if shot.get("lock") else (220, 220, 220)
        draw.text((x0 + 2, y0 + THUMB_H + 4), head[:70], fill=color, font=font_head)
        for j, line in enumerate(wrap_text(shot.get("text", ""), font_text, THUMB_W - 4, draw)):
            draw.text((x0 + 2, y0 + THUMB_H + 26 + j * 16), line, fill=(180, 180, 180), font=font_text)
    page.save(out_path, quality=85)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--per-page", type=int, default=24)
    args = ap.parse_args(argv)
    path = os.path.join(args.video_dir, "media_plan", "shotlist.json")
    if not os.path.exists(path):
        print(f"Нет {path} — сначала прогон pipeline_smart.py (он пишет шотлист после отбора кадров)")
        return 1
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    shots = sorted([s for s in data.get("shots", []) if isinstance(s, dict)], key=lambda s: s.get("index", 0))
    if not shots:
        print("Шотлист пуст")
        return 1
    gates = data.get("gates", {})
    off = [k for k, v in gates.items() if v in (False, "off", 0)]
    print(f"Гейты этого прогона: {json.dumps(gates, ensure_ascii=False)}")
    if off:
        print(f"  выключено/не сработало: {', '.join(off)}")
    outputs = []
    for p in range(0, len(shots), args.per_page):
        page_no = p // args.per_page + 1
        out = os.path.join(args.video_dir, "media_plan", f"shotlist_contact_{page_no:02d}.jpg")
        outputs.append(render_page(shots[p:p + args.per_page], args.video_dir, args.cols, out))
    missing = [s["index"] + 1 for s in shots if not s.get("file")]
    print(f"Готово: {len(outputs)} страниц(ы) — {', '.join(outputs)}")
    if missing:
        print(f"  слоты без файла: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
