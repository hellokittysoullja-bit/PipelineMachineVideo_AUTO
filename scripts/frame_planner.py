#!/usr/bin/env python3
"""План рисованного кадра на каждую фразу — на базе планировщика старого
генератора (stock_query_planner v3), а не с нуля.

Что взято из старого и НЕ переписано:
  * главы как вопросы (shot_brief_director.packets): фразы по порядку, хвост
    прошлой главы, стадия рассказа из speech_plan.json, бриф автора [shot:];
  * мир эпизода — паспорт world_card (модель пишет его по сценарию: эпоха,
    культуры, что нельзя показывать — телефона в сцене из древности);
  * правила спецификации кадра — ДОСЛОВНО абзацы focus/core/subject/claims из
    stock_query_planner.SPEC_PROMPT (вырезаются из него в момент вызова,
    второй копии текста нет): «главное — то, что само по себе напомнит
    фразу», абстракция через ситуацию/телесный признак/предмет-след,
    утверждения must/should, проверяемые взглядом. Их потом проверяет
    судья (shot_judge.verify_claims) — тот же контракт, что в старом;
  * разбор ответа — те же функции (json_objects, clean_text, _parse_claims),
    кэш по содержимому вопроса, второй вопрос при сбитом формате.

Что новое (другая природа кадра): вместо запросов к стокам модель пишет,
КАК кадр нарисовать:
  kind     — scene | caption | diagram;
  labels   — русские подписи дословно (генератор нарисует ровно их, проверка
             букв сверит их точно);
  picture  — английское описание рисунка, главное первым;
  mascot   — есть ли сквозной герой канала; backdrop — white | paper | painted.

Выход: media_plan/frame_plan.json (кадры по порядку блоков, у каждого spec
для судьи и frame для генератора). Фраза без годной спецификации получает
запасной кадр-сцену и помечается fallback — слот не пустеет.

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
import shot_brief_director as sbd  # noqa: E402
import shot_planner_llm  # noqa: E402
import stock_query_planner as sqp  # noqa: E402
import world_card  # noqa: E402

PLAN_NAME = "frame_plan.json"
CACHE_DIR_NAME = "frame_plan_cache"
PLAN_VERSION = 2
DEFAULT_MODEL = os.environ.get("PLANNER_MODEL") or sqp.DEFAULT_MODEL
KINDS = ("scene", "caption", "diagram")
BACKDROPS = ("white", "paper", "painted")
MAX_LABELS = {"scene": 0, "caption": 1, "diagram": 6}
MAX_LABEL_WORDS = 5


def spec_rules():
    """Абзацы focus/core/subject/claims старого SPEC_PROMPT, дословно.
    Граница — начало абзаца queries: он про стоки и здесь не нужен."""
    t = sqp.SPEC_PROMPT
    a, b = t.index("focus —"), t.index("queries —")
    return t[a:b].rstrip()


FRAME_RULES = """frame — how to DRAW this shot for a hand-drawn explainer video (doodle style: stick figures, simple drawn objects and places, diagrams, arrows, short hand-lettered Russian labels). The drawing must show the core and the must claims.
  "kind": "scene" — a drawn situation with no text at all (actions, places, everyday life);
          "caption" — a drawn situation plus ONE big short Russian caption of 1-4 words that states the punchline of the line (like «ЖИВ. ПОЛНОСТЬЮ.»); use it for punchlines and emotional beats;
          "diagram" — a hand-drawn diagram (pyramid, arrows, before/after, a list on a board, a comparison, a timeline, footprints) with 2-6 short Russian labels; use it when the line explains a structure, a comparison, a list or a cause.
  Mix kinds across the chapter: roughly 45% scene, 25% caption, 30% diagram; never three identical kinds in a row.
  "labels": Russian, UPPERCASE, max {max_words} words each, taken from or clearly implied by the line, correct spelling, no English; empty for "scene".
  "picture": English, 15-45 words; start with the core, then the action, then the place; for diagrams say where each label goes by its number (label 1, label 2). Do not describe the drawing style.
  "mascot": true only when the line is about "you", a child, a typical person or an emotional reaction and the channel's recurring character fits: {mascot}.
  "backdrop": "white" for diagrams, "paper" for calm explanations, "painted" for scenes set in a place (cave, field, sea shore, village)."""

PROMPT = """You direct the visuals of a hand-drawn explainer video.
Episode: «{title}». Channel: {niche}. Setting: {setting}.
Below are the narration lines of one chapter, in order{prev}. A line may come with the shot the author wants — keep its meaning.

For EVERY numbered line decide what the viewer must SEE while hearing it.

{spec_rules}

{frame_rules}

Example from another film, «The ball bounced off the wall and rolled away» — the core is the ball, not the wall:
{{"n": 3, "focus": "a ball bouncing off a wall", "subject": "a ball", "core": "a ball is visible", "claims": [{{"id": "c1", "text": "the ball bounces off a wall", "tier": "must"}}, {{"id": "c2", "text": "a wall", "tier": "should"}}], "frame": {{"kind": "caption", "labels": ["ОТСКОК!"], "picture": "a round ball bouncing off a brick wall with small motion lines, the caption big at the bottom", "mascot": false, "backdrop": "paper"}}}}

Answer with one JSON object per narration line, one per line, and nothing else — no explanations, no reasoning, no markdown.

{lines}"""


def render_prompt(packet, setting, profile):
    lines = []
    for u in packet["units"]:
        brief = u.get("author_brief")
        lines.append(f"{u['n']}. «{u['text']}»" + (f" — shot: {brief}" if brief else ""))
    prev = f" (the previous chapter ended with: «{packet['prev_tail']}»)" if packet.get("prev_tail") else ""
    return PROMPT.format(
        title=packet.get("episode_title") or "—", niche=profile["niche"], setting=setting or "not specified",
        prev=prev, spec_rules=spec_rules().format(c1=sqp.MAX_CLAIMS - 1),
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
    """{номер юнита: {"spec", "frame"}} + ошибки. Спецификация разбирается
    теми же функциями, что в старом планировщике (без требования запросов:
    здесь их нет)."""
    known = {u["n"] for u in packet["units"]}
    out, errors = {}, []
    for obj in sqp.json_objects(raw):
        n = obj.get("n")
        if not isinstance(n, int) or n not in known or n in out:
            continue
        focus = sqp.clean_text(obj.get("focus"), lo=2)
        core = sqp.clean_text(obj.get("core"), lo=2)
        rest = [c for c in (obj.get("claims") or []) if isinstance(c, dict)
                and sqp._clean(str(c.get("id") or "")).lower() != sqp.CORE_ID]
        claims = sqp._parse_claims([{"id": sqp.CORE_ID, "text": core, "tier": "must"}] + rest) if core else None
        if not focus or not claims:
            errors.append(f"{n}:bad_spec")
            continue
        # Движение здесь не выполнит ни один кадр (рисунок статичен): утверждение
        # остаётся, но без флага motion — судья спросит его как обычное.
        for c in claims:
            c.pop("motion", None)
        spec = {"focus": focus, "claims": claims, "queries": []}
        subject = sqp.clean_text(obj.get("subject"), lo=1, hi=5)
        if subject:
            spec["subject"] = subject
        frame, err = validate_frame(obj.get("frame"))
        if err:
            errors.append(f"{n}:{err}")
            continue
        out[n] = {"spec": spec, "frame": frame}
    return out, errors


def ask_chapter(gateway, model, packet, setting, profile, cache_dir):
    """Как sqp.ask_chapter: при сбитом формате глава спрашивается второй раз,
    из второго ответа берутся только недостающие фразы."""
    import llm_gateway
    prompt = render_prompt(packet, setting, profile)
    raw, hit = sqp.ask(gateway, model, prompt, cache_dir)
    got, errors = parse_answer(raw, packet)
    if len(got) < len(packet["units"]):
        try:
            raw2, _h = sqp.ask(gateway, model, prompt + sqp.RETRY_NOTE, cache_dir)
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
    автора или по фразе; судья проверит её по брифу (spec_from_brief)."""
    import shot_judge
    brief = block.get("shot_brief")
    picture = brief or f"a simple drawn scene illustrating: {block['text']}"
    return {"spec": shot_judge.spec_from_brief(block["text"], brief),
            "frame": {"kind": "scene", "mascot": False, "backdrop": "paper", "labels": [],
                      "picture": picture}, "fallback": True}


def ensure_world(video_dir, gateway, verbose=True):
    """Паспорт мира эпизода — тот же world_card.generate, что в старом
    генераторе (ручной не трогается, свой пересоздаётся при правке озвучки)."""
    prof = channel.load_profile()
    card, what = world_card.generate(video_dir, gateway, niche=prof["niche"])
    if verbose:
        print(f"Паспорт мира: {what}; {world_card.describe(card)}")
    return card


def plan_episode(video_dir, gateway, model=DEFAULT_MODEL, force=False, workers=4, verbose=True):
    profile = channel.load_profile()
    blocks = script_parser.parse_blocks(os.path.join(video_dir, "script.txt"))
    card = ensure_world(video_dir, gateway, verbose)
    setting = world_card.judge_setting(card)
    cache_dir = os.path.join(video_dir, "media_plan", CACHE_DIR_NAME)
    if force and os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))
    packets = list(sbd.packets(video_dir, blocks))

    def one(packet):
        try:
            return ask_chapter(gateway, model, packet, setting, profile, cache_dir)
        except Exception as e:  # noqa: BLE001 — глава без ответа получит запасные кадры
            return {}, [f"call_failed:{type(e).__name__}: {e}"[:200]], False

    with concurrent.futures.ThreadPoolExecutor(max(1, min(workers, len(packets) or 1))) as ex:
        results = list(ex.map(one, packets))

    by_index, stats, errs = {}, {"planned": 0, "fallback": 0, "cached_chapters": 0}, {}
    for packet, (got, errors, hit) in zip(packets, results):
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
        frames.append({"index": i, "section": b["section"], "text": b["text"],
                       "key": shot_planner_llm.unit_key(b["text"]), "spec": entry["spec"],
                       **entry["frame"], "fallback": bool(entry.get("fallback"))})
    plan = {"version": PLAN_VERSION, "model": model, "setting": setting,
            "world_card": world_card.describe(card),
            "signature": hashlib.sha256(f"{PLAN_VERSION}|{model}|{PROMPT}|{setting}".encode()).hexdigest()[:16],
            "frames": frames, "stats": stats, "errors": errs}
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
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
