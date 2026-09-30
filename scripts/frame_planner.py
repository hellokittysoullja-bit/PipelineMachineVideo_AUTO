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
  mascot   — сквозной герой канала; backdrop — white | paper | painted.

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
import channel  # noqa: E402
import script_parser  # noqa: E402

PLAN_NAME = "frame_plan.json"
CACHE_DIR_NAME = "frame_plan_cache"
PLAN_VERSION = 4
# Модель выбрана замером старого генератора 24.09 (58 фраз трёх ниш):
# DeepSeek v4 Flash — 58/58, ~2 тыс. токенов баланса; Gemini 3.7 Flash по
# смыслу наравне, но ~35 тыс.; Qwen 3.8 Max — 46/58.
DEFAULT_MODEL = os.environ.get("PLANNER_MODEL") or "ds/deepseek-v4-flash"
MAX_TOKENS = 8000
EST_PROMPT_TOKENS = 2500
KINDS = ("scene", "caption", "diagram")
BACKDROPS = ("white", "paper", "painted")
MAX_LABELS = {"scene": 0, "caption": 1, "diagram": 6}
MAX_LABEL_WORDS = 5

# ---------------------------------------------------------------- из stock_query_planner v3 (дословно)
TIERS = ("must", "should")
CORE_ID = "core"
MAX_CLAIMS = 5
RETRY_NOTE = "\n\n(Answer again: one JSON object per numbered line, every line, nothing else.)"
SPEC_RULES = """focus — the new thing this line says, understood in the context of the chapter (resolve pronouns and references from the lines around it). 3 to 12 English words.

core — WHO or WHAT must be visible: the single thing (an object, a person, an animal, a place) that, even alone in a picture, still makes the viewer think of this line — with the state that defines it, if any ("an exhausted person", "a burnt letter"). Name the thing, not an event: what it does goes into the claims. Ask yourself: if the picture could show only one thing, which one? When the line is about something happening to, on or around something else, the core is what the line is about — usually the thing that moves, acts or changes — not the surface, place or object it happens on. When the line is abstract (a feeling, an idea, a process, an argument), the core is a concrete situation, a bodily sign or an object left behind that a camera can photograph and a viewer reads as this idea — never a bare "a person is visible" or an invisible thing like "a memory" or "a brain decision": say what makes the picture show THIS line ("a person slumped over an untouched plate", "a crumpled paper covered in red corrections"). Never make words, captions, labels, signs or logos in the picture part of the core or of a claim — the viewer hears the words, the picture shows things — unless the line is about that very document, chart, headline, sign or screen. Write it as a statement: "a ball is visible".

subject — the thing the line is ABOUT, as a bare noun phrase of 1 to 4 English words ("a dagger", "an arrow", "a tired person"): the thing that must be in the picture for the picture to be about this line at all. Name its GENERAL kind, the word anyone would use — not its type, model, material or part: "a guitar", not "a flamenco guitar", "a guitar neck" or "a wooden guitar"; "a dog", not "a sleeping dog". A picture of another type of the same thing still shows the subject; the exact type belongs in the claims. Only the thing — no action, no place, no other object. For an abstract line, the subject is the thing in the core ("a crumpled paper"). If nothing concrete can be named, give an empty string.

claims — 1 to {c1} more statements checkable by looking at the picture, most important first. Each checks ONE thing (an object, an action, a place, a detail) and does not repeat the core. "tier": "must" if without it the picture does not show this line, "should" if it only makes the picture better. If the line is about a movement that only footage can show, one claim has "motion": true and describes this movement; lines about objects, places or states have no motion claim."""


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

FRAME_RULES = """frame — how to DRAW this shot for a hand-drawn explainer video (doodle style: stick figures, simple drawn objects and places, diagrams, arrows, short hand-lettered Russian labels). The drawing must show the core and the must claims.
  "kind": "scene" — a drawn situation with no text at all (actions, places, everyday life);
          "caption" — a drawn situation plus ONE big short Russian caption of 1-4 words that states the punchline of the line (like «ЖИВ. ПОЛНОСТЬЮ.»); use it for punchlines and emotional beats;
          "diagram" — a hand-drawn diagram (pyramid, arrows, before/after, a list on a board, a comparison, a timeline, footprints) with 2-6 short Russian labels; use it when the line explains a structure, a comparison, a list or a cause.
  Mix kinds across the chapter: roughly 45% scene, 25% caption, 30% diagram; never three identical kinds in a row.
  "labels": Russian, UPPERCASE, max {max_words} words each, taken from or clearly implied by the line, correct spelling, no English; empty for "scene".
  "picture": English, 15-45 words; start with the core, then the action, then the place. The image model draws NO text at all — the labels are added later by code — so never ask for words, letters or numbers in the picture; for a diagram say where the empty space for each label is and where its arrow points (empty space for label 1 to the right of the top tier, an arrow from it to the top tier). Do not describe the drawing style.
  "mascot": true only when the line is about "you", a child, a typical person or an emotional reaction and the channel's recurring character fits: {mascot}.
  "backdrop": "white" for diagrams, "paper" for calm explanations, "painted" for scenes set in a place (cave, field, sea shore, village)."""

PROMPT = """You direct the visuals of a hand-drawn explainer video.
Episode: «{title}». Channel: {niche}.
Below are the narration lines of one chapter, in order{prev}. A line may come with the shot the author wants — keep its meaning.

For EVERY numbered line decide what the viewer must SEE while hearing it.

{spec_rules}

{frame_rules}

Example from another film, «The ball bounced off the wall and rolled away» — the core is the ball, not the wall:
{{"n": 3, "focus": "a ball bouncing off a wall", "subject": "a ball", "core": "a ball is visible", "claims": [{{"id": "c1", "text": "the ball bounces off a wall", "tier": "must"}}, {{"id": "c2", "text": "a wall", "tier": "should"}}], "frame": {{"kind": "caption", "labels": ["ОТСКОК!"], "picture": "a round ball bouncing off a brick wall with small motion lines, the caption big at the bottom", "mascot": false, "backdrop": "paper"}}}}

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


def render_prompt(packet, profile):
    lines = []
    for u in packet["units"]:
        brief = u.get("author_brief")
        lines.append(f"{u['n']}. «{u['text']}»" + (f" — shot: {brief}" if brief else ""))
    prev = f" (the previous chapter ended with: «{packet['prev_tail']}»)" if packet.get("prev_tail") else ""
    return PROMPT.format(
        title=packet.get("episode_title") or "—", niche=profile["niche"], prev=prev,
        spec_rules=SPEC_RULES.format(c1=MAX_CLAIMS - 1),
        frame_rules=FRAME_RULES.format(max_words=MAX_LABEL_WORDS, mascot=profile["mascot"]["description"]),
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
    backdrop = obj.get("backdrop") if obj.get("backdrop") in BACKDROPS else (
        "white" if kind == "diagram" else "paper")
    return {"kind": kind, "mascot": bool(obj.get("mascot")), "backdrop": backdrop,
            "labels": labels, "picture": picture}, None


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


def ask_chapter(gateway, model, packet, profile, cache_dir):
    import llm_gateway
    prompt = render_prompt(packet, profile)
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
            "frame": {"kind": "scene", "mascot": False, "backdrop": "paper", "labels": [],
                      "picture": block.get("shot_brief") or f"a simple drawn scene illustrating: {block['text']}"},
            "fallback": True}


def plan_episode(video_dir, gateway, model=DEFAULT_MODEL, force=False, workers=4, verbose=True):
    profile = channel.load_profile()
    script = os.path.join(video_dir, "script.txt")
    blocks = script_parser.parse_blocks(script)
    cache_dir = os.path.join(video_dir, "media_plan", CACHE_DIR_NAME)
    if force and os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))
    pk = packets(blocks, episode_title(script))

    def one(packet):
        try:
            return ask_chapter(gateway, model, packet, profile, cache_dir)
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
    plan = {"version": PLAN_VERSION, "model": model, "frames": frames, "stats": stats, "errors": errs}
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    if verbose:
        kinds = {k: sum(1 for f in frames if f["kind"] == k) for k in KINDS}
        print(f"План: {len(frames)} кадров {kinds}, с героем {sum(f['mascot'] for f in frames)}, "
              f"запасных {stats['fallback']}, глав из кэша {stats['cached_chapters']}")
        for sec, e in errs.items():
            print(f"  {sec}: {len(e)} замечаний: {', '.join(e[:4])}")
    return plan


def main():
    channel.load_env()
    import llm_gateway
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    gw = llm_gateway.Gateway(spend_cap=int(os.environ.get("PLANNER_MAX_SPEND", "200000")))
    if not gw.configured:
        sys.exit("Нет LLM_GATEWAY_API_KEY в .env")
    plan_episode(a.video_dir, gw, model=a.model, force=a.force)


if __name__ == "__main__":
    main()
