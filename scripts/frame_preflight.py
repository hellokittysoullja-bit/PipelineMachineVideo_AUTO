#!/usr/bin/env python3
"""Предпроверка описания кадра ДО платной генерации — чтобы на кадр уходила
одна попытка рисования, а не несколько.

Разбор браков 04.10 показал: все отбракованные кадры были заказаны таким
описанием. Модель картинок рисует состояние, а не глагол («lifts its flap» —
конверт закрыт), сажает фигуру на предмет, рядом с которым её поставили
(«walks around an envelope lying on a desk» — кот на столе), пишет буквами то,
что названо словами («labeled… as shame»), и рисует чужого кота там, где
героя сняли с кадра, а слово «cat» в описании осталось.

Два слоя:
  1. issues(frame) — правила кода, бесплатно: сдвоенный артикль, персонаж в
     кадре без героя, предмет наезда или место мысли, которых нет в описании,
     два требования размера, длина, слова, вызывающие буквы.
  2. Текстовая модель (та же, что у планировщика: одна глава — один вопрос,
     ~2 тыс. токенов баланса) проходит по чек-листу и переписывает только
     описания с ошибкой. Переписанное принимается, только если прошло ту же
     проверку формы, что и ответ планировщика (validate_frame), сохранило
     предмет наезда и место мысли и не нарушает правил кода — иначе остаётся
     прежнее описание с пометкой.

Ответы модели кэшируются по содержимому вопроса (media_plan/frame_preflight_cache).
Итог по кадру — поле "preflight" в плане: {"issues", "rewritten", "original"}."""
import hashlib
import json
import os
import re

PREFLIGHT_VERSION = 2
CACHE_DIR_NAME = "frame_preflight_cache"
MAX_TOKENS = 12000      # модель рассуждает до ответа, сам ответ короткий
EST_PROMPT_TOKENS = 2000
MIN_WORDS, MAX_WORDS = 20, 90

BIG_RE = re.compile(r"\b(huge|giant|enormous|massive|large|big|oversized|fills? (?:the|most of the) (?:whole )?frame)\b",
                    re.I)
DUP_ARTICLE_RE = re.compile(r"\b(the|a|an)\s+(?:the|a|an)\b", re.I)
CHARACTER_RE = re.compile(r"\b(?:the|a|an) (?:main )?character\b", re.I)


def hero_noun(hero_text):
    """Существительное героя («a black cartoon cat» -> «cat»): по нему видно,
    что кадр без референса всё равно просит того же зверя словами."""
    w = re.findall(r"[A-Za-z]+", hero_text or "")
    return w[-1].lower() if w else None


def tidy(picture):
    return " ".join(DUP_ARTICLE_RE.sub(r"\1", picture).split())


def issues(frame, hero_text=None):
    """Список ошибок задания, которые видны без модели."""
    import frame_planner
    pic = frame.get("picture") or ""
    out = []
    if DUP_ARTICLE_RE.search(pic):
        out.append("duplicate_article")
    n = len(pic.split())
    if n < MIN_WORDS:
        out.append(f"too_short:{n}")
    elif n > MAX_WORDS:
        out.append(f"too_long:{n}")
    if not frame.get("hero"):
        noun = hero_noun(hero_text)
        if CHARACTER_RE.search(pic) or (noun and re.search(rf"\b{noun}s?\b", pic, re.I)):
            out.append("character_without_reference")
    zoom = (frame.get("zoom") or {}).get("object")
    if zoom and _name_core(zoom) not in pic.lower():
        out.append("zoom_object_not_in_picture")
    near = frame.get("key_near")
    if near and _name_core(near) not in pic.lower():
        out.append("key_near_not_in_picture")
    if len(BIG_RE.findall(pic)) >= 2:
        out.append("several_size_demands")
    fixed, why = frame_planner.validate_frame({"kind": frame.get("kind"), "labels": frame.get("labels"),
                                                "hero": frame.get("hero"), "picture": pic})
    if why and why.startswith("writing_in_picture"):
        out.append(why)
    return out


def _name_core(name):
    """«the blank wall calendar» -> «blank wall calendar»: артикль модель
    свободно меняет, предмет — нет."""
    return re.sub(r"^(?:the|a|an)\s+", "", " ".join(name.lower().split()))


CHECKLIST = """You check picture descriptions written for an image model before it draws them. The image model draws each description ONCE and the picture goes straight into a film, so a fault in the words becomes a fault in the picture. Faults to look for:
1. A motion or a process instead of a visible moment: the image model draws a state, not a verb. Rewrite into the end state you can see ("lifts the flap" -> "the flap is folded open, the letter half out"; "walks around the envelope" -> "stands on the floor beside the envelope").
2. A figure without ground, or placed on an object it should stand beside: say what it stands, sits or lies on.
3. Something repeated or going on for long (circling, pacing, waiting for days) drawn as motion or as several copies of the same figure: show it by the traces it left (a worn circle of paw prints around the desk, dust on the lid). Never drop it: when the motion is what the line is about, the traces carry it.
4. Anything that invites letters, numbers or signs (titles, labels, words, documents with text, apps on screens, clocks with numerals, calendars with dates, starting lines, crossed-off days, tally marks, check marks): replace with a drawn object or make it blank ("a calendar page with no marks").
5. A feeling shown by symbols floating in the air (hearts, question marks, lightning, icons): show it by pose, face and objects.
6. More than one thing demanded to be big: keep only the one named in KEEP BIG.
7. Two things that must both be seen clearly (the main character and the KEEP BIG object) touching or overlapping: put them side by side with a clear gap of plain background.
8. NO CHARACTER frames: any recurring character ("the main character", {hero}) — remove it entirely and show the same idea through objects, hands of an unnamed person, or traces.
9. A vaguer word for an object named before ("a device" after "the phone"), or a pronoun with no clear owner.
Keep everything else unchanged: the meaning of the line, the framing, the objects listed in KEEP, 30-80 words, plain English, no words about drawing style. Never add anything to draw that the line does not need.

For EVERY numbered frame answer one JSON object on its own line:
{{"n": 1, "ok": true}}  — nothing to fix;
{{"n": 2, "ok": false, "faults": [1, 7], "picture": "<the full rewritten description>"}}.
No other text.

Frames:
{frames}"""


def chapter_prompt(frames, hero_text=None):
    rows = []
    for n, f in enumerate(frames, 1):
        zoom = (f.get("zoom") or {}).get("object")
        keep = [x for x in (zoom, f.get("key_near")) if x]
        lines = [f"{n}. Line (Russian narration): {f.get('text', '')}",
                 f"   Kind: {f.get('kind')}" + (f", labels added by code later: {len(f.get('labels') or [])}"
                                               if f.get("labels") else ""),
                 "   Main character: " + ("YES — call it \"the main character\", describe only pose, action, "
                                         "expression and props" if f.get("hero") else "NO CHARACTER")]
        if zoom:
            lines.append(f"   KEEP BIG: {zoom}")
        if keep:
            lines.append("   KEEP (name exactly as written): " + "; ".join(keep))
        known = issues(f, hero_text)
        if known:
            lines.append("   Already found: " + ", ".join(known))
        lines.append(f"   Description: {f.get('picture', '')}")
        rows.append("\n".join(lines))
    hero = f"the {hero_noun(hero_text)}" if hero_noun(hero_text) else "any named animal or person of the film"
    return CHECKLIST.format(hero=hero, frames="\n\n".join(rows))


def parse(raw, count):
    """{номер: (ok, faults, picture)} из строк JSON; мусор пропускается."""
    import frame_planner
    out = {}
    for obj in frame_planner.json_objects(raw or ""):
        try:
            n = int(obj.get("n"))
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= count or n in out:
            continue
        ok = obj.get("ok") is True
        faults = [x for x in (obj.get("faults") or []) if isinstance(x, (int, str))][:9]
        pic = " ".join(str(obj.get("picture") or "").split())
        out[n] = (ok, faults, pic)
    return out


def accept(frame, picture, hero_text=None):
    """Переписанное описание годится? (описание, None) или (None, причина)."""
    import frame_planner
    picture = tidy(picture)
    fixed, why = frame_planner.validate_frame({"kind": frame.get("kind"), "labels": frame.get("labels"),
                                                "hero": frame.get("hero"), "picture": picture})
    if why:
        return None, why
    trial = dict(frame, picture=picture)
    # Счёт слов размера — грубый (в «the only large object» он видит требование),
    # поэтому он не запрещает правку, а лишь не даёт ей добавить требований.
    left = [i for i in issues(trial, hero_text) if not i.startswith("too_") and i != "several_size_demands"]
    if len(BIG_RE.findall(picture)) > max(1, len(BIG_RE.findall(frame.get("picture") or ""))):
        left.append("more_size_demands")
    if left:
        return None, "still:" + ",".join(left)
    if not MIN_WORDS <= len(picture.split()) <= MAX_WORDS:
        return None, "length"
    if frame.get("hero") and not CHARACTER_RE.search(picture):
        return None, "hero_lost"
    return picture, None


def ask(gateway, model, prompt, cache_dir):
    key = hashlib.sha256(f"{PREFLIGHT_VERSION}|{model}|{prompt}".encode("utf-8")).hexdigest()[:24]
    cp = os.path.join(cache_dir, key + ".txt")
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            return f.read(), True
    text, _u, _p = gateway.chat(model, [{"type": "text", "text": prompt}], MAX_TOKENS, EST_PROMPT_TOKENS)
    if text.strip():
        os.makedirs(cache_dir, exist_ok=True)
        with open(cp + ".part", "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(cp + ".part", cp)
    return text, False


def check_chapter(frames, gateway, model, cache_dir, hero_text=None):
    """Проверить и при нужде переписать описания одной главы (на месте).
    Возвращает сводку {"checked", "rewritten", "rejected", "cached", "error"}."""
    todo = [f for f in frames if not f.get("fallback")]
    stats = {"checked": len(todo), "rewritten": 0, "rejected": 0, "cached": False, "error": None}
    for f in todo:
        f["picture"] = tidy(f["picture"])          # бесплатно и всегда
        f["preflight"] = {"issues": issues(f, hero_text), "rewritten": False}
    if not todo or gateway is None:
        return stats
    try:
        raw, stats["cached"] = ask(gateway, model, chapter_prompt(todo, hero_text), cache_dir)
    except Exception as e:  # noqa: BLE001 — без проверки кадр остаётся как спланирован
        stats["error"] = f"{type(e).__name__}: {e}"[:200]
        return stats
    for n, (ok, faults, pic) in parse(raw, len(todo)).items():
        f = todo[n - 1]
        pf = f["preflight"]
        if ok or not pic:
            continue
        pf["faults"] = faults
        new, why = accept(f, pic, hero_text)
        if new is None:
            pf["rejected"] = why
            stats["rejected"] += 1
            continue
        pf.update(rewritten=True, original=f["picture"], issues=issues(dict(f, picture=new), hero_text))
        f["picture"] = new
        stats["rewritten"] += 1
    return stats


def run(frames, gateway, model, cache_dir, hero_text=None, workers=4):
    """Все главы плана параллельно. Сводка по эпизоду."""
    import concurrent.futures
    chapters = {}
    for f in frames:
        chapters.setdefault(f.get("section"), []).append(f)
    total = {"checked": 0, "rewritten": 0, "rejected": 0, "cached_chapters": 0, "errors": {}}
    with concurrent.futures.ThreadPoolExecutor(max(1, min(workers, len(chapters) or 1))) as ex:
        futs = {ex.submit(check_chapter, fs, gateway, model, cache_dir, hero_text): sec
                for sec, fs in chapters.items()}
        for fut, sec in futs.items():
            s = fut.result()
            for k in ("checked", "rewritten", "rejected"):
                total[k] += s[k]
            total["cached_chapters"] += int(bool(s["cached"]))
            if s["error"]:
                total["errors"][sec] = s["error"]
    return total


def main():
    """Проверить готовый план без перепланирования:
    python scripts/frame_preflight.py <video_dir>"""
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import env  # noqa: F401
    import frame_planner
    import llm_gateway
    import look as look_mod
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--model", default=frame_planner.DEFAULT_MODEL)
    a = ap.parse_args()
    env.load_env()
    path = os.path.join(a.video_dir, "media_plan", frame_planner.PLAN_NAME)
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    gw = llm_gateway.Gateway(spend_cap=int(os.environ.get("PREFLIGHT_MAX_SPEND", "30000")))
    hero_text = look_mod.load().hero_text if plan.get("has_hero") else None
    s = run(plan["frames"], gw, a.model, os.path.join(a.video_dir, "media_plan", CACHE_DIR_NAME), hero_text)
    plan["preflight"] = s
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    print(f"Предпроверка: {s['checked']} кадров, переписано {s['rewritten']}, отклонено переписанных "
          f"{s['rejected']}, глав из кэша {s['cached_chapters']}, ошибок {len(s['errors'])}")
    for fr in plan["frames"]:
        pf = fr.get("preflight") or {}
        if pf.get("issues") or pf.get("rewritten") or pf.get("rejected"):
            print(f"  [{fr['index']}] {pf}")


if __name__ == "__main__":
    main()
