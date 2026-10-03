#!/usr/bin/env python3
"""План рисованного кадра на каждую фразу.

Основа — планировщик старого генератора (stock_query_planner v3): правила
спецификации кадра (focus / core / subject / claims must-should) и функции
разбора ответа перенесены сюда ДОСЛОВНО (блок «из stock_query_planner v3»
ниже), остальной стоковый планировщик не нужен. Спецификацию потом проверяет
судья shot_judge.verify_claims — тот же контракт, что в старом.

Поверх — описание рисунка (своё, у старого кадр искался, а не рисовался):
  kind     — scene | caption | diagram;
  labels   — русские подписи дословно (генератор нарисует ровно их, проверка
             букв сверит их точно);
  picture  — английское описание рисунка, главное первым;
  hero     — появляется ли главный герой (референс look/hero.*): только где
             фраза о зрителе или обычном человеке, примерно на каждом
             третьем-четвёртом кадре; не три подряд и не больше трети
             эпизода — правило кода (limit_hero).
Стиль в описании не пишется никогда: его задают образцы look/style/.

Глава — один вопрос (фразы по порядку, хвост прошлой главы, бриф автора
[shot:]); сорванная строка теряет себя, а не главу; при сбитом формате глава
спрашивается второй раз. Кэш по содержимому вопроса — перезапуск не платит.
Фраза без годного ответа получает запасной кадр-сцену (слот не пустеет).

Выход: media_plan/frame_plan.json.
Usage: python scripts/frame_planner.py <video_dir> [--model M] [--force]"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: E402
import script_parser  # noqa: E402

PLAN_NAME = "frame_plan.json"
CACHE_DIR_NAME = "frame_plan_cache"
PLAN_VERSION = 8
# Модель выбрана замером старого генератора 24.09 (58 фраз трёх ниш):
# DeepSeek v4 Flash — 58/58, ~2 тыс. токенов баланса; Gemini 3.7 Flash по
# смыслу наравне, но ~35 тыс.; Qwen 3.8 Max — 46/58.
DEFAULT_MODEL = os.environ.get("PLANNER_MODEL") or "ds/deepseek-v4-flash"
MAX_TOKENS = 8000
EST_PROMPT_TOKENS = 2500
KINDS = ("scene", "caption", "diagram")
MAX_HERO_RUN = 2       # героя не бывает на трёх кадрах подряд
MAX_HERO_SHARE = 0.35  # и не больше трети кадров эпизода — для героя-гостя


def hero_limits():
    """(подряд, доля) из .env: HERO_MAX_RUN, HERO_MAX_SHARE. Герой-гость
    (стикмен) — треть кадров; герой-маскот, лицо канала, — чаще (решение
    владельца 03.10: кот бренда в 3 кадрах из 5, а не в одном)."""
    run = int(os.environ.get("HERO_MAX_RUN", "").strip() or MAX_HERO_RUN)
    share = float(os.environ.get("HERO_MAX_SHARE", "").strip() or MAX_HERO_SHARE)
    return max(1, run), min(1.0, max(0.0, share))
MAX_LABELS = {"scene": 0, "caption": 1, "diagram": 6}
MAX_LABEL_WORDS = 5

# ---------------------------------------------------------------- из stock_query_planner v3 (дословно)
TIERS = ("must", "should")
CORE_ID = "core"
MAX_CLAIMS = 5
RETRY_NOTE = "\n\n(Answer again: one JSON object per numbered line, every line, nothing else.)"
SPEC_RULES = """focus — the new thing this line says, understood in the context of the chapter (resolve pronouns and references from the lines around it). 3 to 12 English words.

core — WHO or WHAT must be visible: the single thing (an object, a person, an animal, a place) that, even alone in a picture, still makes the viewer think of this line — with the state that defines it, if any ("an exhausted person", "a burnt letter"). Name the thing, not an event: what it does goes into the claims. Ask yourself: if the picture could show only one thing, which one? When the line is about something happening to, on or around something else, the core is what the line is about — usually the thing that moves, acts or changes — not the surface, place or object it happens on. When the line is abstract (a feeling, an idea, a process, an argument), the core is a concrete situation, a bodily sign or an object left behind that can be drawn and that a viewer reads as this idea — never a bare "a person is visible" or an invisible thing like "a memory" or "a brain decision": say what makes the picture show THIS line ("a person slumped over an untouched plate", "a crumpled paper covered in red corrections"). Never make words, captions, labels, signs or logos in the picture part of the core or of a claim — the viewer hears the words, the picture shows things — unless the line is about that very document, chart, headline, sign or screen. Write it as a statement: "a ball is visible".

subject — the thing the line is ABOUT, as a bare noun phrase of 1 to 4 English words ("a dagger", "an arrow", "a tired person"): the thing that must be in the picture for the picture to be about this line at all. Name its GENERAL kind, the word anyone would use — not its type, model, material or part: "a guitar", not "a flamenco guitar", "a guitar neck" or "a wooden guitar"; "a dog", not "a sleeping dog". A picture of another type of the same thing still shows the subject; the exact type belongs in the claims. Only the thing — no action, no place, no other object. For an abstract line, the subject is the thing in the core ("a crumpled paper"). If nothing concrete can be named, give an empty string.

claims — 1 to {c1} more statements checkable by looking at the picture, most important first. Each checks ONE thing (an object, an action, a place, a detail) and does not repeat the core. "tier": "must" if without it the picture does not show this line, "should" if it only makes the picture better."""


def _clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def clean_text(text, lo=2, hi=16):
    """Английское описание (фокус, утверждение) или None: lo..hi слов
    латиницей, без кириллицы и кавычек."""
    text = _clean(text).strip(" \"'«».;:")
    words = text.split()
    if not lo <= len(words) <= hi or not re.search(r"[a-zA-Z]", text):
        return None
    if re.search(r"[а-яА-ЯёЁ]", text):
        return None
    return text


def _parse_claims(raw_claims):
    """Утверждения по порядку важности, или None, если спецификация негодна:
    первое утверждение обязано быть must (это фокус), id уникальны, движение
    требует не больше одно утверждение."""
    claims, ids = [], set()
    moving = False
    for c in (raw_claims or []):
        if len(claims) >= MAX_CLAIMS:
            break
        if not isinstance(c, dict):
            continue
        cid = _clean(str(c.get("id") or "")).lower()
        text = clean_text(c.get("text"))
        tier = c.get("tier") if c.get("tier") in TIERS else None
        if not cid or cid in ids or not text or not tier:
            continue
        ids.add(cid)
        claim = {"id": cid, "text": text, "tier": tier}
        # Движение — одно на фразу: лишний флаг снимается, утверждение и
        # фраза остаются (раньше из-за него выпадала вся фраза).
        if c.get("motion") is True and not moving and cid != CORE_ID:
            claim["motion"] = True
            moving = True
        claims.append(claim)
    if len(claims) < 1 or claims[0]["tier"] != "must" or claims[0]["id"] != CORE_ID:
        return None
    return claims


def json_objects(raw):
    """Все JSON-объекты ответа по порядку: по строке на объект, массивом, в
    блоке ```json или объектом на несколько строк. Сорванный объект теряет
    только себя — разбор продолжается со следующей скобки."""
    text = raw or ""
    dec = json.JSONDecoder()
    i, out = 0, []
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and "n" in obj:
            out.append(obj)
            i = end
        else:
            i += 1

# ---------------------------------------------------------------- конец блока из v3

FRAME_RULES = """frame — ONE hand-drawn picture per line. It is generated once and goes straight into the film, so describe it completely: the image model sees only your "picture" text and the reference images, nothing else. The picture must show the core and every must claim.
  "kind" — by what the line does:
    "scene" — a drawn moment: people, objects, places, actions. The default.
    "caption" — the line is a punchline, a verdict or an emotional beat that lands harder written: one drawn moment plus ONE Russian caption of 1-4 words (like «ЖИВ. ПОЛНОСТЬЮ.»).
    "diagram" — the line explains a structure, a comparison, a sequence, a list or a cause: a simple hand-drawn diagram with 2-6 short Russian labels — a pyramid, a ladder, arrows from cause to effect, before and after, a list on a board, a timeline, a path of footprints, a crowd shrinking to one figure.
  "labels" — Russian, UPPERCASE, at most {max_words} words each, taken from or clearly implied by the line, correctly spelled; empty for "scene". Code writes them on the finished picture.
  "hero" — {hero_rule}
  "picture" — English, 30-80 words, the full instruction for the image model. How to choose WHAT to draw:
    - show what is NEW in this line — the word it was written for, not its subject that the previous picture already showed ("he knows the job, but today his HANDS shake" -> the shaking hands, close);
    - a comparison that only flashes by ("heavy as a fridge") is not the picture — draw what the line is about; draw the comparison only when it fills the whole line and the narration unfolds it;
    - a contrast of two things ("ten years of practice against two weeks") is ONE picture with both side by side — half of a pair loses the thought;
    - an abstract line (a feeling, an idea, a process) becomes a concrete situation, a bodily sign or an object left behind (hunched shoulders, an untouched plate). A visual metaphor is welcome when it explains the mechanism in a fresh, specific way (a brain lighting up like a slot machine at each notification); stock symbols are not (a stone of burden, a broken chain, an hourglass for "time");
    - resolve "he", "it", "this" from the neighbouring lines and name the thing; no pronouns without a clear owner in the picture;
    - keep the mood of the chapter: a heavy chapter is not drawn with sunny cheerful frames.
  How to WRITE it:
    - in order of importance: the main subject with its pose, gesture and facial expression; the action; at most two supporting props; the place in a few words;
    - an object the image model may not know by name is described by its look, or replaced by a familiar object with the same meaning; a small action (pressing, pouring, signing) is shown close up, through the hands;
    - an object that appears twice in one picture is named the same way both times ("the phone ... the same phone"), never by a vaguer word ("a screen", "a device"): the image model draws a vaguer word as a different object;
    - exact counts for everything countable ("three children", "one phone"); every person has two arms and two legs and holds things in clearly drawn hands;
    - the framing (close-up, medium or wide shot) and where the main subject sits in the frame, with calm empty background around it; neighbouring pictures differ in subject and framing unless the lines continue one moment in the same place;
    - the background: plain and light for diagrams and simple statements, the place itself for scenes set somewhere;
    - "caption": the bottom fifth of the frame is plain empty background. "diagram": the diagram fills the middle, next to each labelled part there is a wide empty patch of plain background (room for a word in big letters), away from the frame edges, with a short hand-drawn arrow from it to the part — no boxes, frames or lines around the empty patches; every label needs its own patch, so name as many patches as there are labels;
    - nothing may carry writing: no letters, numbers, digits, dates, symbols, logos or signs anywhere. Avoid objects that come with writing (apps on screens, book covers, slot-machine reels, price tags, clock numerals); when one is needed, make it blank ("a phone with a blank glowing screen", "a clock face without numerals"). A period is named in words, never as years;
    - people of the past wear the clothes and use the objects of their time;
    - never describe the drawing style, line work or palette: the style comes from the reference images."""

HERO_RULE = """the film has one recurring main character, shown to the image model as a reference picture. true only when the line speaks to the viewer ("you") or shows what an ordinary person feels, does or reacts to — the character then plays that person. Never for objects, places, maps, statistics, diagrams of facts or named historical people. The character appears in at most {hero_share}% of the pictures and never on {hero_run_plus} lines in a row. When true, call the character "the main character" in the picture and describe only pose, action, expression and props, never looks or clothes: the reference picture defines them."""
NO_HERO_RULE = """always false: this film has no recurring main character."""


def hero_rule_text():
    run, share = hero_limits()
    return HERO_RULE.format(hero_share=round(share * 100), hero_run_plus=run + 1)

PROMPT = """You are the director and storyboard artist of a hand-drawn explainer film.
Film: «{title}».
Below are the narration lines of one chapter, in order{prev}. A line may come with the shot the author wants — keep its meaning.

For EVERY numbered line decide what the viewer must SEE while hearing it, then describe that picture.

{spec_rules}

{frame_rules}

Three examples from another film. «The ball bounced off the wall and rolled away» — the core is the ball, not the wall:
{{"n": 2, "focus": "a ball bouncing off a wall", "subject": "a ball", "core": "a ball is visible", "claims": [{{"id": "c1", "text": "the ball bounces off a wall", "tier": "must"}}], "frame": {{"kind": "scene", "labels": [], "hero": false, "picture": "close-up: a red rubber ball in mid-air just after hitting a brick wall, small curved motion lines behind it, a little dust puff at the wall; the ball sits in the right third of the frame, plain light background on the left"}}}}
«First you need food and safety — only then friends, and only then dreams»:
{{"n": 3, "focus": "needs built from the bottom up", "subject": "a pyramid", "core": "a pyramid of needs is visible", "claims": [{{"id": "c1", "text": "the pyramid has three tiers", "tier": "must"}}], "frame": {{"kind": "diagram", "labels": ["ЕДА И БЕЗОПАСНОСТЬ", "ДРУЗЬЯ", "МЕЧТЫ"], "hero": false, "picture": "a large hand-drawn pyramid with three tiers in the middle of the frame: a bowl and a little house in the wide bottom tier, two stick figures holding hands in the middle tier, a small star in the top tier; to the right of each tier an empty patch of plain background with a short arrow pointing at that tier; plain light background"}}}}
«And you just lie there, scrolling, while the evening is gone» (a film with a main character):
{{"n": 3, "focus": "a person lost in a phone while the evening passes", "subject": "a person with a phone", "core": "a person lying with a phone is visible", "claims": [{{"id": "c1", "text": "the person stares at the phone", "tier": "must"}}, {{"id": "c2", "text": "a dark window shows night has fallen", "tier": "should"}}], "frame": {{"kind": "scene", "labels": [], "hero": true, "picture": "medium shot: the main character lies on a sofa on their back, holding one phone with a blank glowing screen above their face with both hands, eyes wide and tired; a window behind shows a dark night sky with a crescent moon; a cold cup of tea on the floor; the character sits in the left half of the frame"}}}}

Answer with one JSON object per narration line, one per line, and nothing else — no explanations, no reasoning, no markdown.

{lines}"""


def unit_key(text):
    """Ключ фразы — её текст (номера сдвигаются от правок сценария)."""
    return hashlib.sha1(" ".join((text or "").split()).encode("utf-8")).hexdigest()[:16]


def episode_title(script_path):
    try:
        m = re.search(r"TITLE\s*:\s*(.+)", open(script_path, encoding="utf-8").read())
        return m.group(1).strip() if m else ""
    except OSError:
        return ""


def packets(blocks, title):
    """Главы как вопросы: фразы по порядку, бриф автора, хвост прошлой главы."""
    out, order = {}, []
    for i, b in enumerate(blocks):
        sec = b.get("section") or "—"
        if sec not in out:
            out[sec] = []
            order.append(sec)
        out[sec].append((i, b))
    res, prev_tail = [], ""
    for sec in order:
        units = [{"n": n, "block_index": i, "text": _clean(b["text"]),
                  "author_brief": _clean(b.get("shot_brief")) or None}
                 for n, (i, b) in enumerate(out[sec], 1)]
        res.append({"section": sec, "episode_title": title, "prev_tail": prev_tail, "units": units})
        prev_tail = _clean(out[sec][-1][1]["text"])[:180]
    return res


def render_prompt(packet, has_hero):
    lines = []
    for u in packet["units"]:
        brief = u.get("author_brief")
        lines.append(f"{u['n']}. «{u['text']}»" + (f" — shot: {brief}" if brief else ""))
    prev = f" (the previous chapter ended with: «{packet['prev_tail']}»)" if packet.get("prev_tail") else ""
    return PROMPT.format(
        title=packet.get("episode_title") or "—", prev=prev,
        spec_rules=SPEC_RULES.format(c1=MAX_CLAIMS - 1),
        frame_rules=FRAME_RULES.format(max_words=MAX_LABEL_WORDS, hero_rule=hero_rule_text() if has_hero else NO_HERO_RULE),
        lines="\n".join(lines))


def _clean_label(s):
    return " ".join(str(s).split()).strip().upper()


def validate_frame(obj):
    """Форма описания кадра: (frame, None) или (None, причина)."""
    if not isinstance(obj, dict):
        return None, "no_frame"
    kind = obj.get("kind")
    if kind not in KINDS:
        return None, f"kind={kind!r}"
    picture = " ".join(str(obj.get("picture") or "").split())
    if len(picture.split()) < 4 or re.search(r"[а-яА-ЯёЁ]", picture):
        return None, "bad_picture"
    labels = [] if kind == "scene" else [_clean_label(x) for x in (obj.get("labels") or []) if str(x).strip()]
    if kind != "scene" and not labels:
        return None, "no_labels"
    labels = labels[:MAX_LABELS[kind]]
    for lab in labels:
        if re.search(r"[A-Za-z]", lab):
            return None, f"latin_in_label:{lab}"
        if not re.search(r"[А-ЯЁ]", lab):
            return None, f"no_cyrillic:{lab}"
        if len(lab.split()) > MAX_LABEL_WORDS:
            return None, f"label_too_long:{lab}"
    return {"kind": kind, "hero": obj.get("hero") is True, "labels": labels, "picture": picture}, None


def parse_answer(raw, packet):
    """{номер юнита: {"spec", "frame"}} + ошибки. Спецификация — тем же
    разбором, что в v3 (без поисковых запросов: здесь их нет). Движение из
    утверждений снимается: рисунок статичен."""
    known = {u["n"] for u in packet["units"]}
    out, errors = {}, []
    for obj in json_objects(raw):
        n = obj.get("n")
        if not isinstance(n, int) or n not in known or n in out:
            continue
        focus = clean_text(obj.get("focus"), lo=2)
        core = clean_text(obj.get("core"), lo=2)
        rest = [c for c in (obj.get("claims") or []) if isinstance(c, dict)
                and _clean(str(c.get("id") or "")).lower() != CORE_ID]
        claims = _parse_claims([{"id": CORE_ID, "text": core, "tier": "must"}] + rest) if core else None
        if not focus or not claims:
            errors.append(f"{n}:bad_spec")
            continue
        for c in claims:
            c.pop("motion", None)
        spec = {"focus": focus, "claims": claims}
        subject = clean_text(obj.get("subject"), lo=1, hi=5)
        if subject:
            spec["subject"] = subject
        frame, err = validate_frame(obj.get("frame"))
        if err:
            errors.append(f"{n}:{err}")
            continue
        out[n] = {"spec": spec, "frame": frame}
    return out, errors


def ask(gateway, model, prompt, cache_dir):
    """Ответ модели на главу; кэш по содержимому вопроса (как в v3).
    Пустой ответ в кэш не пишется."""
    key = hashlib.sha256(f"{PLAN_VERSION}|{model}|{prompt}".encode("utf-8")).hexdigest()[:24]
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


def ask_chapter(gateway, model, packet, has_hero, cache_dir):
    import llm_gateway
    prompt = render_prompt(packet, has_hero)
    raw, hit = ask(gateway, model, prompt, cache_dir)
    got, errors = parse_answer(raw, packet)
    if len(got) < len(packet["units"]):
        try:
            raw2, _h = ask(gateway, model, prompt + RETRY_NOTE, cache_dir)
            more, _e = parse_answer(raw2, packet)
            for n, v in more.items():
                got.setdefault(n, v)
        except llm_gateway.PaymentRequired:
            raise
        except llm_gateway.GatewayError as e:
            errors.append(f"retry_failed:{e}")
    return got, errors, hit


def fallback(block):
    """Спецификации нет — кадр всё равно нужен: сцена без текста по брифу
    автора или по фразе; судья проверит её по одному must-утверждению."""
    text = block.get("shot_brief") or block["text"]
    return {"spec": {"focus": text, "claims": [{"id": CORE_ID, "text": text, "tier": "must"}]},
            "frame": {"kind": "scene", "hero": False, "labels": [],
                      "picture": block.get("shot_brief") or f"a simple drawn scene illustrating: {block['text']}"},
            "fallback": True}


def _drop_hero(f):
    """Снять героя с кадра. В описании «the main character» становится «a person»:
    иначе модель без референса нарисует другого «главного героя»."""
    f["hero"] = False
    f["picture"] = re.sub(r"\b[Tt]he main character\b", "a person", f["picture"])


def limit_hero(frames, max_run=None, max_share=None):
    """Герой — гость, а не ведущий; правило кода, а не просьба к модели:
    не больше max_run кадров подряд и не больше max_share кадров эпизода.
    Лишнее снимается там, где герой стоит теснее всего (рядом с другими
    кадрами героя), — так он остаётся разбросанным по ролику. Сколько снято."""
    run, share = hero_limits()
    max_run = run if max_run is None else max_run
    max_share = share if max_share is None else max_share
    trimmed = 0
    while True:
        idx = [i for i, f in enumerate(frames) if f.get("hero")]
        runs = [i for i in idx if all(i - k in idx for k in range(1, max_run + 1))]
        over = len(idx) > max(1, int(max_share * len(frames)))
        if not runs and not over:
            return trimmed
        if runs:
            victim = runs[0]
        else:
            def crowd(i):
                gaps = [abs(i - j) for j in idx if j != i]
                return (min(gaps) if gaps else len(frames), -i)
            victim = min(idx, key=crowd)
        _drop_hero(frames[victim])
        trimmed += 1


def plan_episode(video_dir, gateway, model=DEFAULT_MODEL, force=False, workers=4, verbose=True, has_hero=False):
    script = os.path.join(video_dir, "script.txt")
    blocks = script_parser.parse_blocks(script)
    cache_dir = os.path.join(video_dir, "media_plan", CACHE_DIR_NAME)
    if force and os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))
    pk = packets(blocks, episode_title(script))

    def one(packet):
        try:
            return ask_chapter(gateway, model, packet, has_hero, cache_dir)
        except Exception as e:  # noqa: BLE001 — глава без ответа получит запасные кадры
            return {}, [f"call_failed:{type(e).__name__}: {e}"[:200]], False

    with concurrent.futures.ThreadPoolExecutor(max(1, min(workers, len(pk) or 1))) as ex:
        results = list(ex.map(one, pk))
    by_index, stats, errs = {}, {"planned": 0, "fallback": 0, "cached_chapters": 0}, {}
    for packet, (got, errors, hit) in zip(pk, results):
        stats["cached_chapters"] += int(bool(hit))
        if errors:
            errs[packet["section"]] = errors
        for u in packet["units"]:
            if u["n"] in got:
                by_index[u["block_index"]] = got[u["n"]]
    frames = []
    for i, b in enumerate(blocks):
        entry = by_index.get(i) or fallback(b)
        stats["fallback" if entry.get("fallback") else "planned"] += 1
        frames.append({"index": i, "section": b["section"], "text": b["text"], "key": unit_key(b["text"]),
                       "spec": entry["spec"], **entry["frame"], "fallback": bool(entry.get("fallback"))})
    if not has_hero:
        for f in frames:
            if f.get("hero"):
                _drop_hero(f)
    stats["hero_trimmed"] = limit_hero(frames)
    plan = {"version": PLAN_VERSION, "model": model, "has_hero": has_hero, "frames": frames,
            "stats": stats, "errors": errs}
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    if verbose:
        kinds = {k: sum(1 for f in frames if f["kind"] == k) for k in KINDS}
        print(f"План: {len(frames)} кадров {kinds}, с героем {sum(f['hero'] for f in frames)} "
              f"(снято правилом кода: {stats['hero_trimmed']}), "
              f"запасных {stats['fallback']}, глав из кэша {stats['cached_chapters']}")
        for sec, e in errs.items():
            print(f"  {sec}: {len(e)} замечаний: {', '.join(e[:4])}")
    return plan


def main():
    env.load_env()
    import llm_gateway
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    gw = llm_gateway.Gateway(spend_cap=int(os.environ.get("PLANNER_MAX_SPEND", "200000")))
    if not gw.configured:
        sys.exit("Нет LLM_GATEWAY_API_KEY в .env")
    import look
    try:
        has_hero = look.load().hero is not None
    except look.LookError as e:
        sys.exit(str(e))
    plan_episode(a.video_dir, gw, model=a.model, force=a.force, has_hero=has_hero)


if __name__ == "__main__":
    main()
