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


def resolve_thumb_source(shot, video_dir):
    """Путь к превью для этого слота: сперва исходный кандидат (`file`,
    обычно temp_smart/pexels_*_cache/...) — он честнее показывает, ЧТО
    выбрал отбор, до грейда/оверлеев. Но эта папка — рабочий кэш, который
    безопасно чистят между прогонами (не входит в поставку, не нужен после
    сборки) — если её уже нет, откат на `clip` (temp_smart/clip_NNNN_*.mp4,
    финальный рендер ЭТОГО слота) — тот же принцип "честно показать, что
    реально ушло в ролик", просто после грейда/титров вместо до. Оба поля
    пишет pipeline_smart.py в один и тот же shotlist.json, ни один не
    гарантирован живым дольше следующей чистки диска."""
    direct = resolve_file(shot.get("file"), video_dir)
    if direct and os.path.exists(direct):
        return direct
    clip = shot.get("clip")
    if clip:
        return os.path.normpath(os.path.join(video_dir, "temp_smart", clip))
    return direct


def build_absorption_cover_map(all_shots):
    """Слот без своего кадра (`source == "absorbed"`) не остаётся на экране
    пустым — `pipeline_smart.py` переносит его длительность на СОСЕДНИЙ
    проверенный слот (см. `_absorb_known_bad_slot`/`NEVER_SHOW_KNOWN_BAD`
    в CLAUDE.md, 21.09): обычно ВПЕРЁД, на ближайший следующий слот с
    настоящим кадром (перенос длительности копится по цепочке, пока не
    дойдёт до первого проверенного); а если поглощён ХВОСТ эпизода (нести
    вперёд некуда — слотов больше нет) — назад, заморозкой последнего
    проверенного кадра (см. "ХВОСТОВОЕ ПОГЛОЩЕНИЕ" в pipeline_smart.py).

    До этой функции контактный лист рисовал ОДИНАКОВЫЙ тёмно-красный
    прямоугольник «НЕТ ФАЙЛА» и для честного отсутствия кандидата, и для
    поглощённого слота — а поглощённый слот в реальном final.mp4 пустым
    не бывает НИКОГДА, там стоит картинка соседа. Лист без этой поправки
    выглядит так, будто половина ролика — чёрные дыры, хотя на самом деле
    там просто меньше уникальных кадров, чем реплик.

    Считается ПО ПОЛНОМУ списку (все страницы разом, по индексу слота, не
    по позиции в срезе страницы) — иначе сосед на следующей странице был бы
    не виден, а первая версия этой функции как раз на этом бы и ловилась.
    Возвращает {index поглощённого слота: shot-запись соседа}; слот без
    доступного соседа (во всём эпизоде нет ни одного проверенного слота) в
    словарь не попадает — для него лист честно останется красным."""
    ordered = sorted(all_shots, key=lambda s: s.get("index", 0))
    n = len(ordered)
    cover = {}
    for k, shot in enumerate(ordered):
        if shot.get("source") != "absorbed":
            continue
        neighbour = None
        for j in range(k + 1, n):
            if ordered[j].get("source") != "absorbed":
                neighbour = ordered[j]
                break
        if neighbour is None:
            for j in range(k - 1, -1, -1):
                if ordered[j].get("source") != "absorbed":
                    neighbour = ordered[j]
                    break
        if neighbour is not None:
            cover[shot.get("index", k)] = neighbour
    return cover


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


def fit_one_line(text, font, max_w, draw):
    """Обрезать строку по РЕАЛЬНОЙ ширине, а не по числу символов.

    Заголовок плитки резался как `head[:70]`, и на секции с длинным именем
    («BLOCK 1: ЛОЖЬ ПЕРВАЯ — "ДОСПЕХ БЫЛ ГРОБОМ"») подпись уезжала за
    границу плитки и налезала на соседнюю — две подписи сливались в одну
    нечитаемую строку. 70 символов кириллицы шире 70 символов латиницы, и
    никакое одно число символов не бывает верным для пропорционального
    шрифта. Лист — материал для разметки глазами, нечитаемая подпись
    обесценивает саму разметку.
    """
    text = text or ""
    if draw.textlength(text, font=font) <= max_w:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if draw.textlength(text[:mid] + "…", font=font) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…"


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


def render_page(shots, video_dir, cols, out_path, cover_map=None):
    cover_map = cover_map or {}
    rows = (len(shots) + cols - 1) // cols
    cell_w, cell_h = THUMB_W + PAD, THUMB_H + CAPTION_H + PAD
    page = Image.new("RGB", (cols * cell_w + PAD, rows * cell_h + PAD), (28, 28, 28))
    draw = ImageDraw.Draw(page)
    font_head = load_font(17)
    font_text = load_font(14)
    for k, shot in enumerate(shots):
        x0 = PAD + (k % cols) * cell_w
        y0 = PAD + (k // cols) * cell_h
        absorbed = shot.get("source") == "absorbed"
        neighbour = cover_map.get(shot.get("index")) if absorbed else None
        thumb_source = resolve_thumb_source(neighbour, video_dir) if neighbour else resolve_thumb_source(shot, video_dir)
        thumb = thumbnail_for(thumb_source)
        if thumb is None:
            thumb = Image.new("RGB", (THUMB_W, THUMB_H), (70, 20, 20))
            ImageDraw.Draw(thumb).text((14, 14), "НЕТ ФАЙЛА / не прочитан", fill=(255, 200, 200), font=font_head)
        elif absorbed and neighbour is not None:
            # Честно, но не тревожно: слот поглощён, картинка — не его
            # собственная. В final.mp4 здесь НЕТ пустого места (см.
            # build_absorption_cover_map) — экран занимает сосед, просто
            # дольше. Полупрозрачная плашка поверх кадра соседа отличает
            # «одолжено» от «своё», не пряча при этом сам кадр целиком —
            # именно то, что реально увидит зритель.
            overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).rectangle([0, 0, THUMB_W, 24], fill=(30, 20, 60, 190))
            thumb = Image.alpha_composite(thumb.convert("RGBA"), overlay).convert("RGB")
            ImageDraw.Draw(thumb).text(
                (6, 3), f"ПОГЛОЩЁН → экран слота #{neighbour.get('index', 0) + 1}",
                fill=(210, 190, 255), font=font_text)
        page.paste(thumb, (x0, y0))
        lock = " 🔒" if shot.get("lock") else ""
        # Провенанс и релевантность идут в подпись плитки, потому что именно
        # этот лист человек и размечает глазами: без «откуда кадр» разметка
        # не отвечает на вопрос, какой источник даёт годное, а какой брак.
        # Отсутствующие поля (кадр скачан до появления sidecar) просто не
        # печатаются — плитка не должна врать прочерком там, где нет данных.
        who = shot.get("provider") or shot.get("source", "")
        rel = shot.get("relevance")
        rel_s = f" rel {rel:.2f}" if isinstance(rel, (int, float)) else ""
        head = (f"#{shot.get('index', 0) + 1}  [{shot.get('kind') or '—'}/{who}]{rel_s}{lock}"
                f"  {shot.get('section', '')}")
        color = (255, 220, 120) if shot.get("lock") else (220, 220, 220)
        draw.text((x0 + 2, y0 + THUMB_H + 4),
                  fit_one_line(head, font_head, THUMB_W - 4, draw),
                  fill=color, font=font_head)
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
    cover_map = build_absorption_cover_map(shots)
    outputs = []
    for p in range(0, len(shots), args.per_page):
        page_no = p // args.per_page + 1
        out = os.path.join(args.video_dir, "media_plan", f"shotlist_contact_{page_no:02d}.jpg")
        outputs.append(render_page(shots[p:p + args.per_page], args.video_dir, args.cols, out, cover_map))
    absorbed = [s["index"] + 1 for s in shots if s.get("source") == "absorbed"]
    missing = [s["index"] + 1 for s in shots if not s.get("file") and s.get("source") != "absorbed"]
    print(f"Готово: {len(outputs)} страниц(ы) — {', '.join(outputs)}")
    if absorbed:
        print(f"  поглощены соседом (в final.mp4 — не пустые, экран соседа дольше): {absorbed}")
    if missing:
        print(f"  слоты без файла: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
