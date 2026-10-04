#!/usr/bin/env python3
"""Где на готовом рисунке предметы, нужные камере: главный предмет кадра (на
нём средний план) и предмет наезда (на него камера «бросается» на его слове).

Рамки даёт модель со зрением (тот же судья, что проверяет кадры): один
короткий вопрос на кадр, кэш по картинке и списку. Рамка проверяется по
пикселям (внутри должен быть рисунок, а не пустая бумага) и по размеру:
во весь кадр — не предмет, точкой — не найден. Не нашлось — камера просто
не наезжает (сборка пишет почему), кадр не бракуется.

Отдельный запуск дозаполняет отчёт уже нарисованного ролика (без перерисовки):
python scripts/objects.py <video_dir>"""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: E402

VERSION = 1
MIN_AREA, MAX_AREA = 0.002, 0.85
MIN_INK = 0.03
PROMPT = """This is a drawing from an explainer film. Find each listed thing in it.
{items}
For each thing return its tight bounding box [x1, y1, x2, y2] in coordinates 0-1000 relative to the image width and height, or null if it is not clearly drawn.
Reply with JSON only: {{"boxes": {{"1": [x1, y1, x2, y2] or null, ...}}}}"""


def parse(answer, n):
    m = re.search(r"\{.*\}", answer or "", re.S)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0)).get("boxes") or {}
    except ValueError:
        return None
    out = {}
    for k in range(1, n + 1):
        b = raw.get(str(k))
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            x1, y1, x2, y2 = (max(0.0, min(1000.0, float(v))) for v in b)
            out[k] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)) if x2 != x1 and y2 != y1 else None
        else:
            out[k] = None
    return out


def ink_share(img, box):
    import numpy as np
    a = np.asarray(img.convert("RGB").crop(tuple(int(round(v)) for v in box)), dtype=np.int16)
    if a.size == 0:
        return 0.0
    med = np.median(np.asarray(img.convert("RGB"), dtype=np.int16).reshape(-1, 3), axis=0)
    return float((np.abs(a - med).sum(axis=2) > 90).mean())


def check(img, box):
    """None — годная рамка, иначе причина."""
    w, h = img.size
    area = (box[2] - box[0])*(box[3] - box[1])/(w*h)
    if area < MIN_AREA:
        return "точка, а не предмет"
    if area > MAX_AREA:
        return "во весь кадр"
    if ink_share(img, box) < MIN_INK:
        return "внутри пустая бумага"
    return None


def locate(gateway, model, img_path, names, cache_dir=None):
    """{имя: рамка в пикселях или None}, плюс {имя: причина отказа}."""
    from PIL import Image
    from labels import _image_part
    img = Image.open(img_path).convert("RGB")
    digest = hashlib.sha256(open(img_path, "rb").read()).hexdigest()
    key = hashlib.sha256(f"{VERSION}|{model}|{names}|{digest}".encode()).hexdigest()[:20]
    cp = os.path.join(cache_dir, f"objects_{key}.json") if cache_dir else None
    raw = None
    if cp and os.path.exists(cp):
        try:
            raw = {int(k): (tuple(v) if v else None) for k, v in json.load(open(cp, encoding="utf-8")).items()}
        except (OSError, ValueError):
            raw = None
    if raw is None:
        items = "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
        answer, _u, _p = gateway.chat(model, [{"type": "text", "text": PROMPT.format(items=items)}, _image_part(img)],
                                      300, 1500, reasoning=False)
        raw = parse(answer, len(names))
        if raw is None:
            return {n: None for n in names}, {n: "модель не вернула рамки" for n in names}
        if cp:
            os.makedirs(cache_dir, exist_ok=True)
            json.dump({k: v for k, v in raw.items()}, open(cp, "w", encoding="utf-8"))
    w, h = img.size
    boxes, why = {}, {}
    for i, n in enumerate(names, 1):
        b = raw.get(i)
        if b is None:
            boxes[n], why[n] = None, "не найден"
            continue
        px = (b[0]*w/1000, b[1]*h/1000, b[2]*w/1000, b[3]*h/1000)
        bad = check(img, px)
        boxes[n], why[n] = (None, bad) if bad else ([round(v, 1) for v in px], None)
    return boxes, why


def wanted(frame, hero_text):
    """[(имя, роль, слово)] — что искать на кадре плана. Для схемы — ещё часть,
    которую описывает каждая подпись (роль label:N): сборка проявляет её на
    слове подписи."""
    out = []
    if frame.get("kind") == "diagram":
        for k, lab in enumerate(frame.get("labels") or []):
            out.append((f"the drawn part that the label «{lab}» would describe (what its little arrow points to)",
                        f"label:{k}", lab))
    z = frame.get("zoom") or {}
    if z.get("object"):
        out.append((z["object"], "zoom", z.get("word")))
    subj = hero_text if frame.get("hero") and hero_text else (frame.get("spec") or {}).get("subject")
    if subj and subj != z.get("object"):
        out.append((subj, "subject", None))
    return out


def objects_for(gateway, model, img_path, frame, hero_text, cache_dir=None):
    """Записи для отчёта кадра: [{name, role, word, box | null, why}]."""
    want = wanted(frame, hero_text)
    if not want or gateway is None:
        return []
    boxes, why = locate(gateway, model, img_path, [n for n, _r, _w in want], cache_dir)
    return [{"name": n, "role": r, "word": w, "box": boxes.get(n), "why": why.get(n)} for n, r, w in want]


def main():
    env.load_env()
    import llm_gateway
    import look
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    a = ap.parse_args()
    mp = os.path.join(a.video_dir, "media_plan")
    plan = {f["index"]: f for f in json.load(open(os.path.join(mp, "frame_plan.json"), encoding="utf-8"))["frames"]}
    rp = os.path.join(mp, "frames_report.json")
    rep = json.load(open(rp, encoding="utf-8"))
    gw = llm_gateway.Gateway(spend_cap=int(os.environ.get("OBJECTS_MAX_SPEND", "20000")))
    model = rep.get("judge_model") or os.environ.get("SHOT_JUDGE_MODEL") or "qwen/qwen3.7-plus"
    hero_text = look.load().hero_text
    n = 0
    for r in rep["frames"]:
        f = plan.get(r["index"])
        if not f or not r.get("chosen"):
            continue
        r["objects"] = objects_for(gw, model, os.path.join(mp, "image_cache", r["chosen"]), f, hero_text,
                                   os.path.join(mp, "judge_cache"))
        n += sum(1 for o in r["objects"] if o["box"])
    json.dump(rep, open(rp + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(rp + ".tmp", rp)
    print(f"Предметов найдено: {n}; потрачено {gw.spent}")


if __name__ == "__main__":
    main()
