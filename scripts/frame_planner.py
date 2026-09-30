#!/usr/bin/env python3
"""План кадра на каждую фразу сценария.

Фраза = блок между [pause] (script_parser.parse_blocks): один блок — один
кадр, тот же принцип, что в исходном пайплайне.

Для каждого блока модель (DeepSeek через шлюз, одна глава — один вызов)
решает:
  kind     — scene   : рисунок без текста;
             caption : рисунок + одна крупная подпись (до 4 слов);
             diagram : схема (пирамида, стрелки, список, доска) с 2-6
                       подписями;
  mascot   — есть ли в кадре сквозной герой канала;
  backdrop — white | paper | painted;
  labels   — русские подписи ДОСЛОВНО (генератор нарисует ровно их);
  picture  — английское описание кадра для генератора картинок.

Бриф автора ([shot:...] в сценарии) главнее модели: если он есть, модель
получает его как готовое описание картинки и дописывает только тип и
подписи.

Выход: <video_dir>/media_plan/frame_plan.json. Повторный запуск берёт главы
из кэша, если их фразы не менялись (ключ — текст фраз + версия вопроса +
модель), то есть правка одной главы не перепланирует ролик целиком.

Usage: python scripts/frame_planner.py <video_dir> [--model M] [--force]"""
import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel  # noqa: E402
import script_parser  # noqa: E402

DEFAULT_MODEL = os.environ.get("PLANNER_MODEL") or "ds/deepseek-v4-flash"
PROMPT_VERSION = 1
KINDS = ("scene", "caption", "diagram")
BACKDROPS = ("white", "paper", "painted")
MAX_LABELS = {"scene": 0, "caption": 1, "diagram": 6}
MAX_LABEL_WORDS = 5

PROMPT = """You are the art director of a Russian YouTube explainer channel drawn in a simple doodle style: stick figures, hand-drawn diagrams, arrows and short hand-lettered Russian labels.
Channel niche: {niche}
Recurring main character (use when the narration is about "you", a child, a typical person, or an emotional reaction): {mascot}
Episode title: {title}
Chapter: {chapter}
Previous chapter ended with: {prev_tail}

For EACH numbered narration line below, design ONE frame that a viewer understands in one second.

Frame kinds:
- "scene": a drawn situation, no text at all. Use for actions, places, everyday life.
- "caption": a drawn situation plus ONE big short Russian caption (1-4 words) that states the punchline of the line, like "ЖИВ. ПОЛНОСТЬЮ." Use for punchlines and emotional beats.
- "diagram": a hand-drawn diagram (pyramid, arrows, before/after, list on a board, comparison, timeline, footprints) with 2-6 short Russian labels. Use when the line explains a structure, a comparison, a list or a cause.

Rules:
1. Show the idea of the line concretely: a situation, a body reaction or an object, never an abstract word.
2. Mix kinds across the chapter: roughly 45% scene, 25% caption, 30% diagram. Never three identical kinds in a row.
3. Russian labels: UPPERCASE, max {max_words} words each, taken from or clearly implied by the line, correct spelling. No English in labels.
4. "picture" is in English, 15-45 words: what is drawn and where; for diagrams, say where each label goes by its number (label 1, label 2). Do not describe the drawing style.
5. backdrop: "white" for diagrams, "paper" for calm explanations, "painted" for scenes set in a place (cave, field, sea shore, village).
6. mascot true only when the recurring character fits; otherwise ordinary stick figures.
7. If a line has "author brief", keep that picture and only choose kind and labels.

Answer with one JSON object per line, one per narration line, nothing else:
{{"n": 1, "kind": "diagram", "mascot": false, "backdrop": "white", "labels": ["РЕДКОЕ.", "ЕДА, БЕЗОПАСНОСТЬ"], "picture": "..."}}

Narration lines:
{lines}"""


def unit_key(text):
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def chapters(blocks):
    """Блоки -> [(секция, [(номер_в_ролике, блок), ...]), ...] по порядку."""
    out = []
    for i, b in enumerate(blocks):
        if not out or out[-1][0] != b["section"]:
            out.append((b["section"], []))
        out[-1][1].append((i, b))
    return out


def build_prompt(section, units, *, title, niche, mascot, prev_tail):
    lines = []
    for k, (_i, b) in enumerate(units, 1):
        line = f"{k}. {b['text']}"
        if b.get("shot_brief"):
            line += f"  [author brief: {b['shot_brief']}]"
        lines.append(line)
    return PROMPT.format(niche=niche, mascot=mascot, title=title or "-", chapter=section,
                         prev_tail=prev_tail or "-", max_words=MAX_LABEL_WORDS,
                         lines="\n".join(lines))


def _clean_label(s):
    s = " ".join(str(s).split()).strip()
    return s.upper()


def validate(obj):
    """Проверка одной строки ответа. Возвращает (frame, None) или (None, причина).
    Проверяется ФОРМА, а не вкус: неверный тип, подписи не на кириллице,
    слишком длинные подписи, подписи у сцены, пустое описание."""
    if not isinstance(obj, dict):
        return None, "not_object"
    kind = obj.get("kind")
    if kind not in KINDS:
        return None, f"kind={kind!r}"
    picture = " ".join(str(obj.get("picture") or "").split())
    if len(picture.split()) < 4:
        return None, "picture_too_short"
    labels = [_clean_label(x) for x in (obj.get("labels") or []) if str(x).strip()]
    if kind == "scene":
        labels = []
    if kind in ("caption", "diagram") and not labels:
        return None, "no_labels"
    if len(labels) > MAX_LABELS[kind]:
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


def parse_answer(text, n_units):
    """Ответ модели -> {номер_в_главе: frame}. Сорванная строка теряет
    только себя, не главу."""
    frames, errors = {}, []
    for line in (text or "").splitlines():
        line = line.strip().strip(",")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            errors.append("bad_json")
            continue
        n = obj.get("n")
        if not isinstance(n, int) or not 1 <= n <= n_units:
            errors.append(f"bad_n:{n!r}")
            continue
        frame, err = validate(obj)
        if err:
            errors.append(f"{n}:{err}")
            continue
        frames[n] = frame
    return frames, errors


def fallback_frame(block):
    """Модель промолчала или ответ не прошёл проверку — кадр всё равно
    нужен. Сцена без текста по брифу автора или по самой фразе: генератор
    получит фразу как описание (переведёт он её сам хуже, но слот не пуст)."""
    picture = block.get("shot_brief") or f"a simple drawn scene illustrating: {block['text']}"
    return {"kind": "scene", "mascot": False, "backdrop": "paper", "labels": [],
            "picture": picture, "fallback": True}


def plan_chapter(gateway, model, section, units, *, title, niche, mascot, prev_tail, cache):
    key = hashlib.sha1("|".join([str(PROMPT_VERSION), model, section, title or "", niche, mascot,
                                 prev_tail or ""] + [u[1]["text"] + "#" + (u[1].get("shot_brief") or "")
                                                     for u in units]).encode("utf-8")).hexdigest()
    if key in cache:
        return key, cache[key], True
    prompt = build_prompt(section, units, title=title, niche=niche, mascot=mascot, prev_tail=prev_tail)
    text, _usage, price = gateway.chat(model, [{"type": "text", "text": prompt}],
                                       max_tokens=400 * len(units) + 400,
                                       estimate_prompt_tokens=len(prompt) // 3, reasoning=False)
    frames, errors = parse_answer(text, len(units))
    entry = {"frames": {str(k): v for k, v in frames.items()}, "errors": errors, "cost": price}
    return key, entry, False


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def episode_title(script_path):
    raw = open(script_path, encoding="utf-8").read()
    m = re.search(r"TITLE\s*:\s*(.+)", raw)
    return m.group(1).strip() if m else ""


def plan_episode(video_dir, gateway, model=DEFAULT_MODEL, force=False, workers=4, verbose=True):
    script_path = os.path.join(video_dir, "script.txt")
    blocks = script_parser.parse_blocks(script_path)
    prof = channel.load_profile()
    niche, mascot = prof["niche"], prof["mascot"]["description"]
    title = episode_title(script_path)
    cache_path = os.path.join(video_dir, "media_plan", "frame_plan_cache.json")
    cache = {} if force else load_json(cache_path, {})
    chs = chapters(blocks)
    jobs = []
    for ci, (section, units) in enumerate(chs):
        prev_tail = chs[ci - 1][1][-1][1]["text"] if ci else ""
        jobs.append((section, units, prev_tail))

    def run(job):
        section, units, prev_tail = job
        try:
            return plan_chapter(gateway, model, section, units, title=title, niche=niche,
                                mascot=mascot, prev_tail=prev_tail, cache=cache)
        except Exception as e:   # сбой главы не роняет план: её кадры получат запасной вариант
            return None, {"frames": {}, "errors": [f"call_failed:{e}"], "cost": 0}, False

    with ThreadPoolExecutor(max(1, workers)) as ex:
        results = list(ex.map(run, jobs))

    frames, total_cost, stats = [], 0, {"planned": 0, "fallback": 0, "cached_chapters": 0}
    new_cache = {}
    for (section, units, _t), (key, entry, cached) in zip(jobs, results):
        if key:      # key None = вызов главы сорвался: в кэш не пишем, следующий прогон спросит снова
            new_cache[key] = entry
        stats["cached_chapters"] += int(cached)
        total_cost += 0 if cached else entry.get("cost", 0)
        for k, (i, b) in enumerate(units, 1):
            fr = entry["frames"].get(str(k))
            if fr is None:
                fr = fallback_frame(b)
                stats["fallback"] += 1
            else:
                stats["planned"] += 1
            frames.append(dict(fr, index=i, section=section, text=b["text"], key=unit_key(b["text"])))
        if verbose and entry["errors"]:
            print(f"  {section}: {len(entry['errors'])} строк(и) не прошли проверку: "
                  f"{', '.join(entry['errors'][:4])}")
    write_json(cache_path, new_cache)
    plan = {"version": PROMPT_VERSION, "model": model, "title": title, "frames": frames,
            "stats": stats, "cost": total_cost}
    write_json(os.path.join(video_dir, "media_plan", "frame_plan.json"), plan)
    if verbose:
        kinds = {k: sum(1 for f in frames if f["kind"] == k) for k in KINDS}
        print(f"План: {len(frames)} кадров ({kinds}), героя в кадре: "
              f"{sum(f['mascot'] for f in frames)}, запасных: {stats['fallback']}, "
              f"глав из кэша: {stats['cached_chapters']}, цена: {total_cost}")
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
