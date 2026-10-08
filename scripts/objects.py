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


DETAILS_PROMPT = """This is a drawing from an explainer film. The editor will cut from the whole picture to 2-3 close shots of its parts while the narration plays. Name up to 3 small, clearly drawn parts of this picture worth their own close shot: each a separate visible thing or part (a hand holding something, a flame, a chain link, a stain, a face), a quarter of the picture or less, never the whole picture or the whole main object. English, 2-6 words each, named so they can be found again.
Reply with JSON only: {"parts": ["...", "..."]}"""
MAX_AUTO_DETAILS = 3


def suggest_details(gateway, model, img_path, cache_dir=None):
    """Имена 1-3 частей рисунка под крупные планы — от модели со зрением. Живой эп.01: планировщик
    (DeepSeek v4 flash) поле details не пишет вообще, и у камеры не было ни одной склейки внутри фразы.
    Кэш по картинке."""
    from PIL import Image
    from labels import _image_part
    digest = hashlib.sha256(open(img_path, "rb").read()).hexdigest()
    key = hashlib.sha256(f"details|{VERSION}|{model}|{digest}".encode()).hexdigest()[:20]
    cp = os.path.join(cache_dir, f"details_{key}.json") if cache_dir else None
    if cp and os.path.exists(cp):
        try:
            return json.load(open(cp, encoding="utf-8"))
        except (OSError, ValueError):
            pass
    img = Image.open(img_path).convert("RGB")
    try:
        answer, _u, _p = gateway.chat(model, [{"type": "text", "text": DETAILS_PROMPT}, _image_part(img)],
                                      200, 1500, reasoning=False)
        m = re.search(r"\{.*\}", answer or "", re.S)
        parts = json.loads(m.group(0)).get("parts") if m else None
    except Exception:  # noqa: BLE001 — без подсказки камера просто без деталей, как раньше
        parts = None
    out = []
    for x in (parts or [])[:MAX_AUTO_DETAILS]:
        x = " ".join(str(x or "").split())
        if x and 1 <= len(x.split()) <= 8 and not re.search(r"[а-яА-ЯёЁ]", x) and x.lower() not in {o.lower() for o in out}:
            out.append(x)
    if cp:                                   # и пустой ответ кэшируется: не переспрашивать платно каждый прогон
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(out, open(cp, "w", encoding="utf-8"))
    return out


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
    near = (frame.get("key_near") or "").strip()
    if frame.get("key_thought") and near:
        out.append((near, "key_anchor", None))
    # главный предмет — то, о чём фраза (spec.subject), а не всегда герой: на «мозг весит тонну»
    # средний план по коту срезал мозг с гирей (критик 04.10). Человек фразы — это герой.
    subj = (frame.get("spec") or {}).get("subject")
    hero = frame.get("hero") and hero_text
    if hero and (not subj or re.search(r"\bperson|people|man|woman|someone\b", subj, re.I)):
        subj = None                          # «человек» фразы — это и есть герой
    if subj and subj != z.get("object"):
        out.append((subj, "subject", None))
    # детали для крупных планов (камера режет на них каждые 2-3 с); у героя с состоянием огонька —
    # всегда его хвост: огонёк и есть то, что он сейчас чувствует (критик 05.10: «крупно дымящийся хвост»)
    det = list(frame.get("details") or [])
    if hero and frame.get("hero_state"):
        det.append(f"the flame on the tip of the tail of {hero_text}")
    taken = {n for n, _r, _w in out}
    for d in det[:4]:
        if d not in taken:
            out.append((d, "detail", None))
            taken.add(d)
    if hero and det:
        # крупный план детали может задеть край героя, но не разрезать ему голову
        out.append((f"the head of {hero_text}", "hero_head", None))
    if hero:
        # средний план держит и предмет фразы, и героя (критик: план по коту срезал мозг с гирей,
        # а план по одному письму был бы конвертом без кота)
        out.append((hero_text, "hero", None))
    return out


def objects_for(gateway, model, img_path, frame, hero_text, cache_dir=None):
    """Записи для отчёта кадра: [{name, role, word, box | null, why}]. Сцена без деталей от
    планировщика получает их от судьи (suggest_details) — иначе камере не на что резать."""
    if gateway is None:
        return []
    if frame.get("kind") != "diagram" and not frame.get("details"):
        auto = suggest_details(gateway, model, img_path, cache_dir)
        if auto:
            frame = dict(frame, details=auto, details_by="judge")
    want = wanted(frame, hero_text)
    if not want:
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
