#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Звуковой режиссёр: где в ролике звучит атмосфера и средневековая музыка.

Повод — прямая жалоба владельца 02.10 после прослушивания 03_plen: «атмосферы
нет, вайба нет». Прежний слой (ambience_plan.py) клал ОДИН фон под целую главу
по словарю слов и молчал, если не был уверен: на эпизоде про пленных он дал
один ветер на 48% ролика на -24 LU под голосом, то есть его почти не слышно.

Теперь атмосфера — СОБЫТИЯ, а не фон: короткий звук (~20 с) ровно там, где
рассказ ставит зрителя в сцену, которую слышно — битва, море, конница,
колокола, костёр, вороны над полем. Решает это не словарь, а модель, которая
читает главу целиком, как звукорежиссёр читает монтажный лист: «цена рыцаря»
— это не рынок, а «рыцари пошли в атаку» — конница, даже если слова «конь»
во фразе нет.

ТРИ ИСТОЧНИКА РЕШЕНИЯ, по старшинству:
  1. Теги автора в сценарии: `[amb:sea_waves]`, `[music:medieval]` — перед
     фразой. Автор знает лучше всех; при их наличии модель не спрашивается.
  2. План модели (`media_plan/sound_plan.json`) — по главам, с кэшем, только
     при ключе шлюза (ключ = согласие владельца на платные вызовы). Ключ
     события в плане — ТЕКСТ фразы, а не номер: правка текста выше по
     сценарию не сдвигает звук на чужую фразу.
  3. Ни того, ни другого — прежний фон по словарю (ambience_plan.py).

Средневековая музыка звучит в начале ролика ВСЕГДА (решение владельца:
«чтобы звучала средневековая музыка в начале»), в финале — под последние
секунды, и под началом глав — где модель решила, что глава открывается
сменой сцены (не чаще MUSIC_MAX_CHAPTER_CUES за ролик).

Платность и отказ: один короткий вопрос на главу дешёвой моделью
(DeepSeek v4 Flash, как у планировщика кадров), кэш по содержимому вопроса,
запасные модели — llm_gateway.text_chain. Любой сбой — глава без событий,
рендер не падает.
"""
import hashlib
import json
import os
import re

PLAN_VERSION = 2
DEFAULT_MODEL = "ds/deepseek-v4-flash"
MAX_TOKENS = 4000              # DeepSeek рассуждает до ответа (см. shot_research)
EST_PROMPT_TOKENS = 2500
CHAIN_TIMEOUT_SEC = 120

# --- атмосфера-события
AMB_EVENT_SEC = 20.0           # длина события: решение владельца «секунд 20»
AMB_PREROLL_SEC = 1.5          # звук входит чуть раньше фразы: сначала слышишь, потом понимаешь
AMB_MIN_EVENT_SEC = 7.0        # короче — не событие, а щелчок; выбрасывается
AMB_MIN_START_GAP_SEC = 12.0   # два старта ближе — второй лишний (модель просят 25+)
AMB_CROSSFADE_SEC = 2.5        # следующее событие наплывает на хвост предыдущего

# --- средневековая музыка
MUSIC_INTRO_SEC = 40.0
MUSIC_CHAPTER_SEC = 16.0
MUSIC_OUTRO_SEC = 40.0
MUSIC_MAX_CHAPTER_CUES = 3
MUSIC_CHAPTER_MIN_SPACING_SEC = 150.0

# Что звучит у каждого вида — для модели. Только место или событие, никогда
# предмет: урок ambience_plan.py («меч в кадре не говорит, где человек»).
KIND_DESCRIPTIONS = {
    "wind_open": "wind over open country: a field, hills, a plain, mountains, an army on the march",
    "forest_birds": "a forest with birds: woods, morning, a quiet countryside",
    "night": "night outdoors: crickets, owls, a camp at night",
    "stone_hall": "inside stone walls: a castle hall, a church, a monastery, a crypt, a prison cell",
    "forge_fire": "a fire: a campfire, a hearth, a burning town, a forge",
    "rain_mud": "rain and mud: bad weather, a wet field, autumn",
    "river_stream": "a river: a stream, a ford, a bridge over water",
    "crowd_market": "a crowd of people: a town square, a market, a fair, a gathered crowd",
    "sea_waves": "the sea: a shore, a sea crossing, ships, a coast, a port",
    "battle_distant": "a battle in progress: clashing swords, shouting soldiers, a melee",
    "cavalry_horses": "horses: a cavalry charge, galloping riders, mounted knights",
    "church_bells": "church bells over a town or a village: alarm, funeral, celebration",
    "crows_field": "crows over an empty field: the aftermath of a battle, the dead, desolation",
    "war_drums": "war drums: an army marching to battle, getting ready to attack",
}

PROMPT = """You are the sound designer of a narrated history documentary (Russian voice-over).
Episode: {title}
Chapter: {chapter}
{prev}
The chapter's lines (number, start time, text):
{lines}

Ambience sounds you may use (name: what it is):
{kinds}

Task: decide where the viewer should HEAR the place or the event the narration puts them in. A cue starts at a line and plays about 20 seconds quietly under the voice.

Rules:
1. Put a cue where the narration places the viewer INSIDE a scene that has a sound: a battle, a sea crossing, a cavalry charge, a town with bells, a camp by the fire, rain on a field, crows over the dead. Do not put a cue where a line explains money, law, numbers or ideas, unless that line is set in such a scene.
2. Choose the sound of the scene, not of a word: "the price of a knight" is not a market; "the knights charged" is cavalry even without the word "horse".
3. Vary the sounds: do not use the same sound for two cues in a row unless the same scene continues; prefer the sound most specific to the scene.
4. The film should feel alive: aim for a cue roughly every 30-45 seconds of the chapter (see the times), never closer than 25 seconds. Even an analytical chapter is usually set somewhere: a castle hall where a ransom is counted, a town square, a camp, a field after the battle; give it that place's sound. Leave a stretch silent only when the narration is truly placeless.
5. If the chapter opens a new story, a new battle or a new era, put "music" on line 1: a short medieval music sting under the chapter title.

Answer with one line per cue and nothing else:
number | sound name
If this chapter should have no cue, answer: none"""


def _fmt_time(sec):
    sec = max(0, int(round(float(sec))))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def chapter_units(blocks, sub_starts):
    """[(глава, [(индекс_блока, время, текст), ...]), ...] в порядке ролика."""
    out = []
    for i, b in enumerate(blocks or []):
        sec = str(b.get("section", ""))
        t = float(sub_starts[i]) if i < len(sub_starts) else 0.0
        if not out or out[-1][0] != sec:
            out.append((sec, []))
        out[-1][1].append((i, t, str(b.get("text", "")).strip()))
    return out


def render_prompt(title, chapter, units, prev_tail, kinds):
    lines = "\n".join(f"{n} [{_fmt_time(t)}] {text}" for n, (_i, t, text) in enumerate(units, 1))
    prev = f"The previous chapter ended with: {prev_tail}\n" if prev_tail else ""
    kind_lines = "\n".join(f"- {k}: {KIND_DESCRIPTIONS.get(k, k.replace('_', ' '))}" for k in kinds)
    return PROMPT.format(title=title or "(untitled)", chapter=chapter, prev=prev,
                         lines=lines, kinds=kind_lines)


_LINE_RE = re.compile(r"^\s*\**\s*(\d+)\s*[|:\-–]\s*([a-z_]+)", re.I)


def parse_answer(raw, n_units, kinds):
    """{номер_строки: вид} — только известные виды и «music». Чужое молча
    пропускается: недоверие к формату, а не к главе целиком."""
    out = {}
    allowed = set(kinds) | {"music"}
    for line in (raw or "").splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        n, name = int(m.group(1)), m.group(2).lower()
        if 1 <= n <= n_units and name in allowed and n not in out:
            out[n] = name
    return out


def _cache_path(cache_dir, model, prompt):
    key = hashlib.sha256(f"{PLAN_VERSION}|{model}|{prompt}".encode("utf-8")).hexdigest()[:24]
    return os.path.join(cache_dir, key + ".txt")


def ask(gateway, model, prompt, cache_dir):
    """Ответ на главу через цепочку моделей; кэш по содержимому вопроса."""
    import llm_gateway

    def cached(m):
        cp = _cache_path(cache_dir, m, prompt)
        if os.path.exists(cp):
            with open(cp, encoding="utf-8") as f:
                return f.read()
        return None
    got = llm_gateway.chat_fallback(gateway, llm_gateway.text_chain(model),
                                    [{"type": "text", "text": prompt}], MAX_TOKENS,
                                    EST_PROMPT_TOKENS, cached=cached, timeout=CHAIN_TIMEOUT_SEC)
    text = got[0]
    if not got.from_cache and text.strip():
        cp = _cache_path(cache_dir, got.model, prompt)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cp + ".part", "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(cp + ".part", cp)
    return text, got.from_cache, got.model


def episode_title(video_dir):
    try:
        with open(os.path.join(video_dir, "script.txt"), encoding="utf-8") as f:
            for line in f:
                if line.strip().upper().startswith("TITLE"):
                    return line.split(":", 1)[-1].strip()
    except OSError:
        pass
    return os.path.basename(os.path.normpath(video_dir))


def plan_path(video_dir):
    return os.path.join(video_dir, "media_plan", "sound_plan.json")


def plan_episode(video_dir, blocks, sub_starts, gateway, kinds, model=DEFAULT_MODEL, workers=4):
    """Спросить модель по главам (параллельно) -> план на диск.

    План: {"version", "model", "cues": [{"section", "text", "name"}]} —
    событие привязано к ТЕКСТУ фразы."""
    from concurrent.futures import ThreadPoolExecutor
    import llm_gateway
    title = episode_title(video_dir)
    cache_dir = os.path.join(video_dir, "media_plan", "sound_director_cache")
    chapters = chapter_units(blocks, sub_starts)
    jobs = []
    for k, (sec, units) in enumerate(chapters):
        prev_tail = chapters[k - 1][1][-1][2] if k else ""
        jobs.append((sec, units, render_prompt(title, sec, units, prev_tail, kinds)))

    def one(job):
        sec, units, prompt = job
        try:
            raw, _hit, used = ask(gateway, model, prompt, cache_dir)
        except llm_gateway.PaymentRequired:
            raise
        except Exception as e:  # noqa: BLE001 — глава без событий, не отказ рендера
            print(f"  звуковой режиссёр: глава «{sec[:40]}» без ответа ({type(e).__name__})")
            return sec, units, {}, None
        return sec, units, parse_answer(raw, len(units), kinds), used

    cues, models = [], set()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for sec, units, picks, used in pool.map(one, jobs):
            if used:
                models.add(used)
            for n, name in sorted(picks.items()):
                _i, _t, text = units[n - 1]
                cues.append({"section": sec, "text": text, "name": name})
    plan = {"version": PLAN_VERSION, "model": sorted(models) or [model], "cues": cues}
    path = plan_path(video_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return plan


def load_plan(video_dir):
    try:
        with open(plan_path(video_dir), encoding="utf-8") as f:
            plan = json.load(f)
        return plan if plan.get("version") == PLAN_VERSION else None
    except (OSError, ValueError):
        return None


def inline_cues(blocks):
    """[(индекс_блока, тип, вид)] из тегов автора [amb:]/[music:]."""
    out = []
    for i, b in enumerate(blocks or []):
        for c in b.get("sound_cues") or []:
            typ = "music" if c.get("type") == "music" else "amb"
            out.append((i, typ, str(c.get("name", "")).strip().lower()))
    return out


def plan_cues(plan, blocks):
    """[(индекс_блока, тип, вид)] из плана модели — по тексту фразы в своей
    главе. Фраза, которой больше нет в сценарии, молча выпадает."""
    out = []
    used = set()
    for c in (plan or {}).get("cues") or []:
        for i, b in enumerate(blocks):
            if i in used or str(b.get("section", "")) != c.get("section"):
                continue
            if str(b.get("text", "")).strip() == c.get("text"):
                used.add(i)
                name = c.get("name", "")
                out.append((i, "music", "medieval") if name == "music" else (i, "amb", name))
                break
    return out


def _stable_seed(*parts):
    h = 0
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def ambience_events(cues, sub_starts, total_dur, available=None,
                    event_sec=AMB_EVENT_SEC, preroll=AMB_PREROLL_SEC):
    """События атмосферы на шкале аудио: [{start, end, kind, seed, block}].

    Правила, все от слуха, а не от удобства кода:
      * старт чуть раньше фразы (preroll) — звук наплывает, потом слово;
      * два старта ближе AMB_MIN_START_GAP_SEC — второй лишний;
      * тот же вид подряд, пока первый ещё звучит, — продление, а не
        повторный вход (иначе «выключилось и включилось»);
      * следующее событие наплывает на хвост предыдущего на AMB_CROSSFADE_SEC;
      * короче AMB_MIN_EVENT_SEC — выбрасывается.
    seed — номер вхождения вида: два события одного вида берут РАЗНЫЕ
    записи библиотеки, а не одну и ту же.
    """
    raw = []
    for i, typ, name in cues:
        if typ != "amb" or not name or (available is not None and name not in available):
            continue
        if i >= len(sub_starts):
            continue
        raw.append((max(0.0, float(sub_starts[i]) - preroll), name, i))
    raw.sort()
    events = []
    for start, name, i in raw:
        if events and start - events[-1]["start"] < AMB_MIN_START_GAP_SEC:
            continue
        if events and events[-1]["kind"] == name and start < events[-1]["end"]:
            events[-1]["end"] = min(float(total_dur), start + event_sec)
            continue
        events.append({"start": start, "end": min(float(total_dur), start + event_sec),
                       "kind": name, "block": i})
    for a, b in zip(events, events[1:]):
        a["end"] = min(a["end"], b["start"] + AMB_CROSSFADE_SEC)
    events = [e for e in events if e["end"] - e["start"] >= AMB_MIN_EVENT_SEC]
    seen = {}
    for e in events:
        seen[e["kind"]] = seen.get(e["kind"], -1) + 1
        e["seed"] = seen[e["kind"]] * 7919 + _stable_seed(e["kind"]) % 97
        e["start"], e["end"] = round(e["start"], 3), round(e["end"], 3)
    return events


def music_cues(cues, sub_starts, total_dur, hook_end=None, final_start=None, enabled=True):
    """Средневековая музыка: [{start, end, role, seed}].

    Вступление — всегда с нуля; финал — под последние MUSIC_OUTRO_SEC;
    под главы — по тегам/плану, не чаще MUSIC_MAX_CHAPTER_CUES и не ближе
    MUSIC_CHAPTER_MIN_SPACING_SEC друг к другу и к вступлению/финалу."""
    if not enabled or total_dur <= 0:
        return []
    total = float(total_dur)
    out = [{"start": 0.0, "end": min(total, MUSIC_INTRO_SEC), "role": "intro"}]
    outro_start = max(out[0]["end"] + 5.0, total - MUSIC_OUTRO_SEC)
    has_outro = outro_start < total - 8.0
    picked = []
    for i, typ, _name in sorted(cues, key=lambda c: c[0]):
        if typ != "music" or i >= len(sub_starts):
            continue
        t = max(0.0, float(sub_starts[i]) - 1.0)
        if t < out[0]["end"] + MUSIC_CHAPTER_MIN_SPACING_SEC / 2:
            continue
        if has_outro and t > outro_start - MUSIC_CHAPTER_MIN_SPACING_SEC / 2:
            continue
        if picked and t - picked[-1] < MUSIC_CHAPTER_MIN_SPACING_SEC:
            continue
        if len(picked) >= MUSIC_MAX_CHAPTER_CUES:
            break
        picked.append(t)
        out.append({"start": round(t, 3), "end": round(min(total, t + MUSIC_CHAPTER_SEC), 3),
                    "role": "chapter"})
    if has_outro:
        out.append({"start": round(outro_start, 3), "end": round(total, 3), "role": "outro"})
    for k, c in enumerate(out):
        c["seed"] = k
    return out


def episode_sound_cues(video_dir, blocks, sub_starts, gateway=None, kinds=None, verbose=True):
    """(cues, источник): теги автора -> план модели (свежий или с диска).

    Модель спрашивается, только если у автора тегов нет, есть шлюз и план
    на диске не покрывает текущие фразы (кэш вопросов делает повтор
    бесплатным; новый вопрос — только для изменённых глав)."""
    own = inline_cues(blocks)
    if own:
        return own, "author"
    plan = None
    if gateway is not None and kinds:
        try:
            plan = plan_episode(video_dir, blocks, sub_starts, gateway, kinds)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ВНИМАНИЕ: звуковой режиссёр не отработал ({type(e).__name__}: "
                      f"{str(e)[:160]}) — берётся план с диска, если есть")
    if plan is None:
        plan = load_plan(video_dir)
    if plan is None:
        return [], "none"
    return plan_cues(plan, blocks), "director"


def write_inline(script_path, plan):
    """Проставить план в сценарий тегами [amb:]/[music:] перед своей фразой
    (.bak перед записью). Только там, где текст фразы встречается в файле
    ровно один раз; теги автора не трогаются."""
    with open(script_path, encoding="utf-8") as f:
        src = f.read()
    done, skipped = 0, []
    for c in (plan or {}).get("cues") or []:
        text = c.get("text", "")
        tag = "[music:medieval]" if c.get("name") == "music" else f"[amb:{c.get('name')}]"
        if not text or src.count(text) != 1:
            skipped.append(text[:60])
            continue
        pos = src.index(text)
        if re.search(r"\[(?:amb|music):[^\]]*\]\s*$", src[max(0, pos - 40):pos]):
            continue
        src = src[:pos] + tag + src[pos:]
        done += 1
    if done:
        with open(script_path + ".bak", "w", encoding="utf-8") as f:
            with open(script_path, encoding="utf-8") as g:
                f.write(g.read())
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(src)
    return done, skipped


def main(argv=None):
    """python scripts/sound_director.py <video_dir> --write-inline

    Проставить сохранённый план модели (media_plan/sound_plan.json) в
    сценарий тегами [amb:]/[music:], чтобы автор мог их поправить руками.
    Сам план составляет рендер (pipeline_smart.py) при ключе шлюза."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("video_dir")
    ap.add_argument("--write-inline", action="store_true")
    args = ap.parse_args(argv)
    plan = load_plan(args.video_dir)
    if not plan:
        print("Плана нет: его составляет рендер (pipeline_smart.py) при LLM_GATEWAY_API_KEY.")
        return 1
    for c in plan["cues"]:
        print(f"  {c['section'][:30]:30s} {c['name']:16s} {c['text'][:70]}")
    if args.write_inline:
        done, skipped = write_inline(os.path.join(args.video_dir, "script.txt"), plan)
        print(f"Проставлено тегов: {done}; пропущено (текст не найден один раз): {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
