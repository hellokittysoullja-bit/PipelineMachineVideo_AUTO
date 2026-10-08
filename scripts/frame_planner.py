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
PLAN_VERSION = 15
# Модель выбрана замером старого генератора 24.09 (58 фраз трёх ниш):
# DeepSeek v4 Flash — 58/58, ~2 тыс. токенов баланса; Gemini 3.7 Flash по
# смыслу наравне, но ~35 тыс.; Qwen 3.8 Max — 46/58.
DEFAULT_MODEL = os.environ.get("PLANNER_MODEL") or "ds/deepseek-v4-flash"
MAX_TOKENS = 16000     # 04.10: DeepSeek рассуждал все 8000 и не успевал ответить
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
  "zoom" — optional. When the line names ONE concrete object that deserves a close look at the moment it is said (the letter, the timer, the open door): {{"object": "<English name of that object exactly as in your picture>", "word": "<the word of the line at which the camera rushes onto it, copied exactly as written in the line>"}}. The camera then fills the screen with that object for a second or two. null when nothing is worth it; at most one line in three.
  "details" — REQUIRED on every frame (the editor cannot cut without it): 1-3 English names of drawn things in your picture worth their own close shot while the line plays (the chain and the boulder; the smoking tail flame; the sticky blobs on the envelope), each named exactly as in your picture and each a small separate visible part — a quarter of the picture or less, not the whole picture or its main object ("the biggest dripping blob", not "the blobs on the envelope"); [] for a single simple object. The editor cuts between the whole picture and these close shots every 2-3 seconds, so a picture held for a long line needs them.
  "key" — optional: the chapter's main thought, written by hand on the picture as it is said — 1-3 Russian words copied word for word from the line ("только открыть"). Only for the one or two lines of a chapter that carry its main idea; null for all others. With a key, "key_near" — the English name of the drawn thing (from your picture) the words belong next to ("the blank wall calendar"), or null.
  "accent" — optional: 1-3 Russian words copied word for word from the line that pop up on screen in bold as they are said — a number with its unit ("пять минут", "две минуты") or one short punchy word the line hits ("тонну"). Never on a line with a "key"; about one line in three; null otherwise.
  "labels" — Russian, UPPERCASE, at most {max_words} words each, taken from or clearly implied by the line, correctly spelled; empty for "scene". Code writes them on the finished picture.
  "hero" — {hero_rule}
  "picture" — English, 30-80 words, the full instruction for the image model. How to choose WHAT to draw:
    - show what is NEW in this line — the word it was written for, not its subject that the previous picture already showed ("he knows the job, but today his HANDS shake" -> the shaking hands, close);
    - a comparison that only flashes by ("heavy as a fridge") is not the picture — draw what the line is about; draw the comparison only when it fills the whole line and the narration unfolds it;
    - a contrast of two things ("ten years of practice against two weeks") is ONE picture with both side by side — half of a pair loses the thought;
    - an abstract line (a feeling, an idea, a process) becomes a concrete situation, a bodily sign or an object left behind (hunched shoulders, an untouched plate). A visual metaphor is welcome when it explains the mechanism in a fresh, specific way (a brain lighting up like a slot machine at each notification); stock symbols are not (a stone of burden, a broken chain, an hourglass for "time");
    - ONE image idea per picture: never stack a second metaphor on the first (a heavy brain chained to a boulder does not also get a burning fuse); every prop either is the line's thought or supports it;
    - resolve "he", "it", "this" from the neighbouring lines and name the thing; no pronouns without a clear owner in the picture;
    - keep the mood of the chapter: a heavy chapter is not drawn with sunny cheerful frames.
  How to WRITE it:
    - in order of importance: the main subject with its pose, gesture and facial expression; the action; at most two supporting props; the place in a few words;
    - an object the image model may not know by name is described by its look, or replaced by a familiar object with the same meaning; a small action (pressing, pouring, signing) is shown close up, through the hands;
    - an object that appears twice in one picture is named the same way both times ("the phone ... the same phone"), never by a vaguer word ("a screen", "a device"): the image model draws a vaguer word as a different object;
    - describe the moment you SEE, never a motion or a process: the image model draws a state, not a verb ("lifts the flap" -> "the flap is folded open, the letter half out"; "walks around the envelope" -> "stands on the floor beside the envelope");
    - every figure stands, sits or lies on something named (the floor, a chair, the grass); a figure next to an object is beside it on the ground, never on top of it unless the line says so;
    - something that happens again and again, or has gone on for long, is shown by the traces it left (a worn path, a pile of unopened envelopes, dust on the lid), not by several copies of the same figure;
    - a feeling is shown only by the pose, the face and the objects around, never by symbols floating in the air (hearts, question marks, lightning, sweat drops as icons);
    - only ONE thing is drawn big: the "zoom" object when there is one, otherwise the main subject; never ask for two big things;
    - when the main character and the "zoom" object are both in the picture, they sit side by side with a clear gap of plain background between them, neither touching nor covering the other: the camera cuts to each of them in turn;
    - with "hero" false the picture has no recurring character at all — no "main character", no animal of the film: show the idea through objects, hands of an unnamed person, or traces;
    - exact counts for everything countable ("three children", "one phone"); every person has two arms and two legs and holds things in clearly drawn hands;
    - the framing (close-up, medium or wide shot) and where the main subject sits in the frame, with calm empty background around it; neighbouring pictures differ in subject and framing unless the lines continue one moment in the same place;
    - the background: plain and light for diagrams and simple statements, the place itself for scenes set somewhere;
    - the "zoom" object is drawn large and clear, never tiny, with plain background around it; a line with a "key" keeps a calm area of plain background (about a third of the frame) where the words will be written by hand;
    - "caption": the bottom fifth of the frame is plain empty background. "diagram": the diagram fills the middle, next to each labelled part there is a wide empty patch of plain background (room for a word in big letters), away from the frame edges, with a short hand-drawn arrow from it to the part — no boxes, frames or lines around the empty patches; every label needs its own patch, so name as many patches as there are labels;
    - nothing may carry writing: no letters, numbers, digits, dates, symbols, logos or signs anywhere. Never ask for things that are shown by writing: nothing "labeled", "named" or "marked as", no starting or finish lines, no crossed-off days, ticks or tally marks — show the idea by a drawn object or the character instead (shame — the character hiding its face; days passing — dust and cobwebs on the envelope). Avoid objects that come with writing (apps on screens, book covers, slot-machine reels, price tags, clock numerals); when one is needed, make it blank ("a phone with a blank glowing screen", "a clock face without numerals"). A period is named in words, never as years;
    - people of the past wear the clothes and use the objects of their time;
    - never describe the drawing style, line work or palette: the style comes from the reference images."""

HERO_RULE = """the film has one recurring main character ({hero_text}), shown to the image model as a reference picture. true only when the line speaks to the viewer ("you") or shows what an ordinary person feels, does or reacts to — the character then plays that person. Never for objects, places, maps, statistics, diagrams of facts or named historical people. The character appears in at most {hero_share}% of the pictures and never on {hero_run_plus} lines in a row. When true, call the character "the main character" in the picture and describe only pose, action, expression and props, never looks, clothes or ears ("ears up", "ears back" redraw the character's fixed ears — show the feeling by the face, the tail and the pose): the reference picture defines them. Traces it leaves are prints of its own feet (an animal leaves paw prints, not shoe prints).{states}"""
MASCOT_RULE = """the film has one recurring main character ({hero_text}) — the face of the channel, shown to the image model as a reference picture. true for most lines: the character ACTS OUT the line — what it says to the viewer, what a person feels or does, and even an abstract idea is shown through what the character does with an object or how it reacts. false only for named historical people, maps, statistics and pure diagrams of facts. The character appears in at most {hero_share}% of the pictures and never on {hero_run_plus} lines in a row. When true, call the character "the main character" in the picture and describe only pose, action, expression and props, never looks, clothes or ears ("ears up", "ears back" redraw the character's fixed ears — show the feeling by the face, the tail and the pose): the reference picture defines them. Traces it leaves are prints of its own feet (an animal leaves paw prints, not shoe prints).{states}"""
MASCOT_SHARE = 0.5      # доля героя от этой — маскот: действует почти в каждом кадре, а не гость
NO_HERO_RULE = """always false: this film has no recurring main character."""


LIVE_ACTIONS = ("look", "paw_on_chest")      # что умеет живая кукла без жестов лап (владелец 07.10 отложил
                                              # повороты и жесты лап; остались глаза, голова, ухо, хвост, огонёк,
                                              # дыхание и поза «лапа у груди»)
LIVE_RULE = ('\n  "hero_action" — only with hero true: what the character physically does in your picture, one of '
             '"look" (sits or stands on the ground beside the thing of the line and looks at it, nothing in its '
             'paws), "paw_on_chest" (sits with one paw pressed to its chest — a feeling, a confession, nothing else '
             'in its paws), "other" (holds, carries, lies, climbs, hides its face, touches or interacts with anything '
             'in any other way). Choose "other" whenever in doubt.')


LIVE_NOFIG_ID = "nofig"
LIVE_NOFIG_TEXT = "no character, creature, animal, person, hand or paw is anywhere in the picture"


def live_hero_enabled():
    """MASCOT_LIVE_PLAN=1: герой на простых кадрах (сидит и смотрит, лапа у груди) не рисуется моделью,
    а ставится в сборке живой куклой (mascot_live) рядом с предметом. По умолчанию 0: меняет задание
    модели и состав картинок — включать после разметки владельцем пилота."""
    return os.environ.get("MASCOT_LIVE_PLAN", "0").strip() == "1"


def hero_rule_text(hero_text=None, states=None):
    run, share = hero_limits()
    st = ""
    if states:
        opts = "; ".join(f'"{k}" when {v["when"]}' for k, v in states.items())
        st = f'\n  "hero_state" — only with hero true: {opts}; null for a neutral moment.'
    rule = MASCOT_RULE if share >= MASCOT_SHARE else HERO_RULE
    if live_hero_enabled():
        st += LIVE_RULE
    return rule.format(hero_share=round(share * 100), hero_run_plus=run + 1,
                       hero_text=hero_text or "the main character", states=st)

PROMPT = """You are the director and storyboard artist of a hand-drawn explainer film.
Film: «{title}».
Below are the narration lines of one chapter, in order{prev}. A line may come with the shot the author wants — keep its meaning.

For EVERY numbered line decide what the viewer must SEE while hearing it, then describe that picture.

{spec_rules}

{frame_rules}

Three examples from another film. «The ball bounced off the wall and rolled away» — the core is the ball, not the wall:
{{"n": 2, "focus": "a ball bouncing off a wall", "subject": "a ball", "core": "a ball is visible", "claims": [{{"id": "c1", "text": "the ball bounces off a wall", "tier": "must"}}], "frame": {{"kind": "scene", "labels": [], "hero": false, "details": ["the dust puff at the wall"], "accent": null, "picture": "close-up: a red rubber ball in mid-air just after hitting a brick wall, small curved motion lines behind it, a little dust puff at the wall; the ball sits in the right third of the frame, plain light background on the left"}}}}
«First you need food and safety — only then friends, and only then dreams»:
{{"n": 3, "focus": "needs built from the bottom up", "subject": "a pyramid", "core": "a pyramid of needs is visible", "claims": [{{"id": "c1", "text": "the pyramid has three tiers", "tier": "must"}}], "frame": {{"kind": "diagram", "labels": ["ЕДА И БЕЗОПАСНОСТЬ", "ДРУЗЬЯ", "МЕЧТЫ"], "hero": false, "picture": "a large hand-drawn pyramid with three tiers in the middle of the frame: a bowl and a little house in the wide bottom tier, two stick figures holding hands in the middle tier, a small star in the top tier; to the right of each tier an empty patch of plain background with a short arrow pointing at that tier; plain light background"}}}}
«And you just lie there, scrolling, while the evening is gone» (a film with a main character):
{{"n": 3, "focus": "a person lost in a phone while the evening passes", "subject": "a person with a phone", "core": "a person lying with a phone is visible", "claims": [{{"id": "c1", "text": "the person stares at the phone", "tier": "must"}}, {{"id": "c2", "text": "a dark window shows night has fallen", "tier": "should"}}], "frame": {{"kind": "scene", "labels": [], "hero": true, "details": ["the phone with a blank glowing screen", "the crescent moon in the window", "the cold cup of tea"], "accent": null, "picture": "medium shot: the main character lies on a sofa on their back, holding one phone with a blank glowing screen above their face with both hands, eyes wide and tired; a window behind shows a dark night sky with a crescent moon; a cold cup of tea on the floor; the character sits in the left half of the frame"}}}}

A Russian line with a camera rush and a handwritten key thought («Поставь таймер на десять минут — и всё, больше ничего не нужно», a film with a main character):
{{"n": 4, "focus": "a ten-minute timer as the whole task", "subject": "a kitchen timer", "core": "a kitchen timer is visible", "claims": [{{"id": "c1", "text": "the timer is being set", "tier": "must"}}], "frame": {{"kind": "scene", "labels": [], "hero": true, "zoom": {{"object": "the kitchen timer", "word": "таймер"}}, "details": ["the dial of the kitchen timer"], "key": "десять минут", "key_near": "the kitchen timer", "accent": null, "picture": "medium shot: the main character turns the dial of one big round kitchen timer with a blank face on an empty table, ears up, calm focused look; the timer is large in the right half of the frame with plain light background around it; calm empty background above the timer"}}}}

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


def render_prompt(packet, has_hero, hero=None):
    lines = []
    for u in packet["units"]:
        brief = u.get("author_brief")
        lines.append(f"{u['n']}. «{u['text']}»" + (f" — shot: {brief}" if brief else ""))
    prev = f" (the previous chapter ended with: «{packet['prev_tail']}»)" if packet.get("prev_tail") else ""
    return PROMPT.format(
        title=packet.get("episode_title") or "—", prev=prev,
        spec_rules=SPEC_RULES.format(c1=MAX_CLAIMS - 1),
        frame_rules=FRAME_RULES.format(max_words=MAX_LABEL_WORDS, hero_rule=hero_rule_text(
            (hero or {}).get("text"), (hero or {}).get("states")) if has_hero else NO_HERO_RULE),
        lines="\n".join(lines))


def _clean_label(s):
    return " ".join(str(s).split()).strip().upper()


# Слова, которые модель картинок рисует БУКВАМИ (живые кадры 04.10: «labeled by shape as shame and
# anxiety» -> на кляксах «shame»/«anxiety», «a starting line» -> надпись STARTING LINE, «crossed-off
# days» -> крестики, прочитанные судьёй как текст). Описание с ними — брак задания, а не кадра.
WRITING_RE = re.compile(r"\b(label(?:l)?ed|labels?|written|writing|inscri\w*|says|reads|titled|named|marked as|"
                        r"word|words|lettering|captions?|signs?|starting line|finish line|"
                        r"cross(?:ed)?[- ]off|tally|check ?marks?|ticks?)\b", re.I)


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
    m = WRITING_RE.search(re.sub(r"\bno (?:text|letters|words|writing|signs)\b|without (?:text|letters|numerals)|"
                                 r"blank[^,;]*", "", picture, flags=re.I))
    if m and kind != "scene" and m.group(0).lower().startswith(("label", "caption")):
        m = None        # у схемы и кадра-подписи места под подписи законны: подписи кладёт код
    if m:
        return None, f"writing_in_picture:{m.group(0)}"
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


def extras(obj, text, states=()):
    """Необязательные поля кадра: наезд, главная мысль, состояние героя.
    Слова наезда и мысли обязаны быть в самой фразе (тем же правилом, что
    сборка ищет их в речи), иначе поле отбрасывается — не ошибка кадра."""
    import words
    out, notes = {}, []
    z = obj.get("zoom")
    if isinstance(z, dict):
        o = " ".join(str(z.get("object") or "").split())
        w = " ".join(str(z.get("word") or "").split())
        if o and 1 <= len(o.split()) <= 6 and not re.search(r"[а-яА-ЯёЁ]", o) and w and words.in_text(w, text):
            out["zoom"] = {"object": o, "word": w}
        else:
            notes.append("zoom_dropped")
    k = obj.get("key")
    if isinstance(k, str) and k.strip():
        k = " ".join(k.split()).lower()
        if 1 <= len(k.split()) <= 3 and re.search(r"[а-яё]", k) and not re.search(r"[a-z]", k) and words.in_text(k, text):
            out["key_thought"] = k
            near = " ".join(str(obj.get("key_near") or "").split())
            if near and 1 <= len(near.split()) <= 6 and not re.search(r"[а-яА-ЯёЁ]", near):
                out["key_near"] = near
        else:
            notes.append("key_dropped")
    a = obj.get("accent")
    if isinstance(a, str) and a.strip() and "key_thought" not in out:
        a = " ".join(a.split()).lower()
        if 1 <= len(a.split()) <= 3 and re.search(r"[а-яё0-9]", a) and not re.search(r"[a-z]", a) and words.in_text(a, text):
            out["accent"] = a
        else:
            notes.append("accent_dropped")
    det = obj.get("details")
    if isinstance(det, list):
        keep = []
        for d in det[:3]:
            d = " ".join(str(d or "").split())
            if d and 1 <= len(d.split()) <= 8 and not re.search(r"[а-яА-ЯёЁ]", d) and d not in keep:
                keep.append(d)
        if keep:
            out["details"] = keep
    st = obj.get("hero_state")
    if isinstance(st, str) and st in states and obj.get("hero") is True:
        out["hero_state"] = st
    act = obj.get("hero_action")
    if live_hero_enabled() and isinstance(act, str) and act in LIVE_ACTIONS and obj.get("hero") is True:
        out["hero_action"] = act                 # "other" и всё незнакомое — герой рисуется моделью, как раньше
    return out, notes


def parse_answer(raw, packet, states=()):
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
        text = next(u["text"] for u in packet["units"] if u["n"] == n)
        more, notes = extras(obj.get("frame"), text, states)
        frame.update(more)
        errors += [f"{n}:{x}" for x in notes]
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


def ask_chapter(gateway, model, packet, has_hero, cache_dir, hero=None):
    import llm_gateway
    prompt = render_prompt(packet, has_hero, hero)
    states = tuple(((hero or {}).get("states") or {}).keys()) if has_hero else ()
    raw, hit = ask(gateway, model, prompt, cache_dir)
    got, errors = parse_answer(raw, packet, states)
    if len(got) < len(packet["units"]):
        try:
            raw2, _h = ask(gateway, model, prompt + RETRY_NOTE, cache_dir)
            more, _e = parse_answer(raw2, packet, states)
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


def _drop_hero(f, replacement="a person"):
    """Снять героя с кадра. В описании «the main character» становится «a person»:
    иначе модель без референса нарисует другого «главного героя»."""
    f["hero"] = False
    f.pop("hero_state", None)
    f.pop("hero_action", None)      # действие куклы без героя — мусор в плане (живой прогон 07.10)
    f["picture"] = re.sub(r"\b[Tt]he main character\b", replacement, f["picture"])


NUMERAL_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?|одн[аоу]|один|одного|одной|две|два|двух|три|трёх|трех|четыре|пять|пяти|шесть|семь|восемь|девять|"
    r"десять|десяти|одиннадцать|двенадцать|пятнадцать|двадцать|тридцать|сорок|пятьдесят|сто|двести|триста|пятьсот|"
    r"тысяч[аиу]?|миллион[аов]?|полчаса|полтора|полторы)\b", re.I)


UNIT_RE = re.compile(r"^(секунд|минут|час|дн[яей]|день|недел|месяц|лет|год|раз|процент|рубл|доллар|евро|кг|килограмм|грамм|"
                     r"тонн|метр|км|километр|сантиметр|литр|шаг|слов|страниц|пис[её]м|человек|людей|штук|попыт|лет)")


def auto_accent(text):
    """Число с единицей из фразы («пять минут», «две минуты», «3 дня») — акцент по правилу кода.
    Планировщик (DeepSeek v4 flash) поле accent не пишет вообще (живой эп.01: ни одного на пять фраз,
    при трёх числах в тексте), а число на экране — самый дешёвый якорь внимания. Берётся первое число
    фразы и следующее за ним слово, если это не предлог/союз; «полчаса» — само по себе."""
    toks = re.findall(r"[А-Яа-яЁё0-9.,]+", text or "")
    for i, t in enumerate(toks):
        if not NUMERAL_RE.fullmatch(t.strip(".,")):
            continue
        if t.lower().startswith("полчаса"):
            return t.strip(".,").lower()
        if i + 1 < len(toks):
            nxt = toks[i + 1].strip(".,").lower()
            if nxt and UNIT_RE.match(nxt):          # число + мера («пять минут»), а не «одно письмо»
                return f"{t.strip(',.').lower()} {nxt}"
    return None


def accent_pass(frames):
    """Акцент по правилу там, где модель его не дала и нет главной мысли (акцент и мысль — не вместе)."""
    import words
    n = 0
    for f in frames:
        if f.get("accent") or f.get("key_thought") or f.get("kind") == "caption":
            continue
        a = auto_accent(f.get("text", ""))
        if a and words.in_text(a, f.get("text", "")):
            f["accent"] = a
            f["accent_by"] = "rule"
            n += 1
    return n


DRAWN_FIELDS = ("kind", "labels", "hero", "hero_live", "hero_action", "hero_state", "picture", "zoom", "details",
                "key_thought", "key_near", "accent", "spec", "preflight")


def keep_drawn_frames(frames, video_dir):
    """Кадр, который уже нарисован и принят судьёй (frames_report: status ok, тот же ключ фразы),
    НЕ перепланируется: его поля берутся из прежнего плана. Живой прогон 08.10: правка задания
    планировщика переписала все пять кадров эпизода — кадр куклы снова стал «кот нарисован», описания
    разошлись с оплаченными картинками, и следующий генератор перерисовал бы четыре кадра (150 000).
    Новое поле, которого в старом плане не было (например accent), остаётся от нового ответа.
    Сколько кадров сохранено."""
    mp = os.path.join(video_dir, "media_plan")
    try:
        old = {f.get("key"): f for f in json.load(open(os.path.join(mp, PLAN_NAME), encoding="utf-8"))["frames"]
               if isinstance(f, dict)}
        rep = json.load(open(os.path.join(mp, "frames_report.json"), encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return 0
    drawn = {r.get("key") for r in rep.get("frames", []) if r.get("status") == "ok" and r.get("path")}
    n = 0
    for f in frames:
        o = old.get(f.get("key"))
        if not o or f.get("key") not in drawn:
            continue
        for k in DRAWN_FIELDS:
            if k in o:
                f[k] = o[k]
            elif k in f and k not in ("accent", "details"):
                f.pop(k)
        f["kept_drawn"] = True
        n += 1
    return n


def live_hero_pass(frames):
    """Кадры героя с простым действием (hero_action из LIVE_ACTIONS) отдаются живой кукле: картинка
    генерируется БЕЗ героя (hero False — без референса), а в описании он пока остаётся, чтобы
    предпроверка переписала его в «предметы и следы» (правило 8). Идёт ПОСЛЕ limit_hero: доля и
    серии героя считаются по всем его кадрам, живым и нарисованным. Сколько кадров отдано."""
    n = 0
    for f in frames:
        if f.get("hero") and f.get("hero_action") in LIVE_ACTIONS:
            f["hero"] = False
            f["hero_live"] = True
            # судья обязан отклонить кадр с любой фигурой: живой прогон 07.10 — модель дорисовала существо за
            # конвертом, судья по утверждениям «конверт виден» его принял, и кукла встала бы рядом с чужаком
            cl = list((f.get("spec") or {}).get("claims") or [])
            if not any(c.get("id") == LIVE_NOFIG_ID for c in cl):
                cl.append({"id": LIVE_NOFIG_ID, "text": LIVE_NOFIG_TEXT, "tier": "must"})
                f.setdefault("spec", {})["claims"] = cl
            n += 1
    return n


def live_hero_revert(frames, hero_text=None):
    """Живой кадр, в описании которого персонаж остался (предпроверки не было или она отклонила
    переписанное), возвращается нарисованному герою: иначе модель без референса нарисовала бы
    чужого кота рядом с куклой. Сколько возвращено."""
    import frame_preflight
    n = 0
    for f in frames:
        if f.get("hero_live") and "character_without_reference" in frame_preflight.issues(f, hero_text):
            f["hero"] = True
            f["hero_live"] = False
            n += 1
    return n


def limit_hero(frames, max_run=None, max_share=None, replacement="a person"):
    """Герой — гость, а не ведущий; правило кода, а не просьба к модели:
    не больше max_run кадров подряд и не больше max_share кадров эпизода.
    Лишнее снимается там, где герой стоит теснее всего (рядом с другими
    кадрами героя), — так он остаётся разбросанным по ролику. Сколько снято."""
    run, share = hero_limits()
    max_run = run if max_run is None else max_run
    max_share = share if max_share is None else max_share
    trimmed = 0

    def weight(i):
        # кадр, где герой несёт смысл (его состояние, наезд, главная мысль), снимается последним:
        # 04.10 правило сняло кота ровно с кульминации «только открыть письмо»
        f = frames[i]
        return int(bool(f.get("hero_state"))) + int(bool(f.get("zoom"))) + int(bool(f.get("key_thought")))
    while True:
        idx = [i for i, f in enumerate(frames) if f.get("hero")]
        runs = [i for i in idx if all(i - k in idx for k in range(1, max_run + 1))]
        over = len(idx) > max(1, int(max_share * len(frames)))      # «не больше доли» — вниз

        def crowd(i):
            gaps = [abs(i - j) for j in idx if j != i]
            return (weight(i), min(gaps) if gaps else len(frames), -i)
        if not runs and not over:
            return trimmed
        if runs:
            victim = min(range(runs[0] - max_run, runs[0] + 1), key=crowd)   # из самой длинной серии — наименее важный
        else:
            victim = min(idx, key=crowd)
        _drop_hero(frames[victim], replacement)
        trimmed += 1


def plan_episode(video_dir, gateway, model=DEFAULT_MODEL, force=False, workers=4, verbose=True, has_hero=False,
                 hero=None, preflight=None):
    """hero — {"text": кто герой, "states": состояния из look/hero_states.json} или None.
    preflight — проверить описания до генерации (frame_preflight); None — по
    FRAME_PREFLIGHT в .env (по умолчанию да)."""
    script = os.path.join(video_dir, "script.txt")
    blocks = script_parser.parse_blocks(script)
    cache_dir = os.path.join(video_dir, "media_plan", CACHE_DIR_NAME)
    if force and os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))
    pk = packets(blocks, episode_title(script))

    def one(packet):
        try:
            return ask_chapter(gateway, model, packet, has_hero, cache_dir, hero)
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
    n_kept = keep_drawn_frames(frames, video_dir)
    if n_kept:
        stats["kept_drawn"] = n_kept
    if not has_hero:
        for f in frames:
            if f.get("hero"):
                _drop_hero(f)
    # у маскота снятый с кадра герой — тот же зверь словами (без референса), а не «человек с ушами»
    _run, _share = hero_limits()
    repl = (hero or {}).get("text") if (hero or {}).get("text") and _share >= MASCOT_SHARE else "a person"
    free = [f for f in frames if not f.get("kept_drawn")]
    stats["hero_trimmed"] = limit_hero(free, replacement=repl) if free else 0
    n_acc = accent_pass(frames)
    if n_acc:
        stats["accent_by_rule"] = n_acc
    if has_hero and live_hero_enabled():          # без флага — stats байт в байт прежние
        stats["hero_live"] = live_hero_pass(free) + sum(1 for f in frames if f.get("kept_drawn") and f.get("hero_live"))
    plan = {"version": PLAN_VERSION, "model": model, "has_hero": has_hero, "frames": frames,
            "stats": stats, "errors": errs}
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    failed = sorted({f["section"] for f in frames if f.get("fallback")})
    if failed and os.path.exists(path) and not force:
        # Глава без ответа модели получила бы запасные кадры «a simple drawn scene illustrating: <фраза>»,
        # и генератор потратил бы деньги на них (живой случай 04.10: 429 шлюза). Прежний план не трогаем.
        print(f"План НЕ перезаписан: нет ответа модели по главам {', '.join(failed)} (запасные кадры вместо "
              f"задания) — повторите запуск позже.")
        plan["not_written"] = failed
        return plan
    if preflight is None:
        preflight = os.environ.get("FRAME_PREFLIGHT", "1").strip() != "0"
    if preflight:
        # Описание с ошибкой — это брак кадра, оплаченный генерацией: чинится здесь, словами, за копейки.
        import frame_preflight
        plan["preflight"] = frame_preflight.run(
            frames, gateway, model, os.path.join(video_dir, "media_plan", frame_preflight.CACHE_DIR_NAME),
            (hero or {}).get("text") if has_hero else None, workers)
    if stats.get("hero_live"):
        stats["hero_live_reverted"] = live_hero_revert(frames, (hero or {}).get("text"))
        stats["hero_live"] -= stats["hero_live_reverted"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    if verbose:
        kinds = {k: sum(1 for f in frames if f["kind"] == k) for k in KINDS}
        print(f"План: {len(frames)} кадров {kinds}, нарисованных сохранено {stats.get('kept_drawn', 0)}, "
              f"с героем {sum(f['hero'] for f in frames)} "
              f"(снято правилом кода: {stats['hero_trimmed']}"
              + (f", живой куклой: {stats['hero_live']}, возвращено рисунку: {stats.get('hero_live_reverted', 0)}"
                 if stats.get("hero_live") or stats.get("hero_live_reverted") else "") + "), "
              f"запасных {stats['fallback']}, глав из кэша {stats['cached_chapters']}")
        for sec, e in errs.items():
            print(f"  {sec}: {len(e)} замечаний: {', '.join(e[:4])}")
        pf = plan.get("preflight")
        if pf:
            print(f"Предпроверка описаний: переписано {pf['rewritten']} из {pf['checked']}, отклонено "
                  f"переписанных {pf['rejected']}, ошибок {len(pf['errors'])}")
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
        lk = look.load()
    except look.LookError as e:
        sys.exit(str(e))
    has_hero = lk.hero is not None
    plan = plan_episode(a.video_dir, gw, model=a.model, force=a.force, has_hero=has_hero,
                        hero={"text": lk.hero_text, "states": lk.hero_states} if has_hero else None)
    if plan.get("not_written"):
        sys.exit(1)


if __name__ == "__main__":
    main()
