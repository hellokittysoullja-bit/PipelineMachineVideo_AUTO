#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Звуковой режиссёр: где в ролике звучит атмосфера и какая музыка.

Повод — жалоба владельца 02.10 после прослушивания 03_plen: «атмосферы нет,
вайба нет». Затем — требование: в ЛЮБОЙ нише звук к месту, без лишнего.

АТМОСФЕРА — события, а не фон: короткий звук (~20 с) там, где рассказ
ставит зрителя внутрь сцены, которую слышно. Решает модель по главе
целиком, как звукорежиссёр по монтажному листу. Устройство выбрано
замером на эталоне (scripts/sound_director_eval.py, метки — Claude):
  * задание строгое: звук только на сцене, на объяснениях, цифрах, деньгах,
    мнениях и упоминаниях вскользь — тишина (прежнее правило «звук каждые
    30-45 с» давало ~15% неверных и ~12% сомнительных звуков);
  * ДВА черновика разных моделей объединяются, ПРОВЕРКА только вычёркивает
    (добавить или заменить не может); проверка не ответила — глава без
    звуков: лишний звук хуже тишины;
  * мир фильма берётся из паспорта эпизода, а не из кода: ниша не зашита.

МУЗЫКА — треки из библиотеки (scripts/music_library.py) по их текстовым
карточкам: вступление, финал, до трёх врезок под главы и тихая подложка
под каждую главу. Тот же приём: два черновика, проверка выбирает или
отказывается. Длины не зашиты: врезка кончается на границе фразы рядом с
целевой длиной.

Старшинство: теги автора в сценарии ([amb:вид] перед фразой; [music:id]
— врезка этим треком под главой) > план модели > ничего. Платно только при
ключе шлюза; кэш по содержимому вопроса делает повтор бесплатным.
"""
import hashlib
import json
import os
import re

PLAN_VERSION = 7
# Модели выбраны замером 02.10 на эталоне 03_plen (scripts/sound_director_eval.py):
# DeepSeek Flash — черновик с 8 неверными из 17; Gemini 3.7 Flash и Qwen 3.8 Max —
# ноль неверных, но каждый находит треть сцен; Sonnet 5 — лучший, но ~143 тыс.
# баланса за ролик. Два точных черновика объединяются, проверка вычёркивает.
DRAFT_MODELS = ("ag/gemini-3.7-flash-low", "qwen/qwen3.8-max")
CRITIC_MODELS = ("qwen/qwen3.8-max", "ag/gemini-3.7-flash-low")
DEFAULT_MODEL = DRAFT_MODELS[0]
CRITIC_MODEL = CRITIC_MODELS[0]
MAX_TOKENS = 8000              # модели рассуждают до ответа: на 4000 Qwen 3.8 Max отдавал пустой ответ
EST_PROMPT_TOKENS = 2500
CHAIN_TIMEOUT_SEC = 120

# --- атмосфера-события
AMB_EVENT_SEC = 20.0           # длина события: решение владельца «секунд 20»
AMB_PREROLL_SEC = 1.5          # звук входит чуть раньше фразы: сначала слышишь, потом понимаешь
AMB_MIN_EVENT_SEC = 7.0        # короче — не событие, а щелчок; выбрасывается
AMB_MIN_START_GAP_SEC = 12.0   # два старта ближе — второй лишний (модель просят 25+)
AMB_CROSSFADE_SEC = 2.5        # следующее событие наплывает на хвост предыдущего

# --- музыкальные врезки под главы (выбор трека — музыкальный режиссёр ниже)
MUSIC_MAX_CHAPTER_CUES = 3
MUSIC_CHAPTER_MIN_SPACING_SEC = 150.0

# Что звучит у каждого вида — для модели. Только место или событие, никогда
# предмет: урок ambience_plan.py («меч в кадре не говорит, где человек»).
#
# Ниша — в профиле канала, а не в коде (аудит 03_plen 02.10: «mounted
# knights», «castle hall» жили литералами, и любой клон получал их как свои).
# channel_profile.json -> sound.ambience_kinds — описания и порядок видов
# этого канала. Профиля нет — описания берутся из каталога библиотеки
# (sound_library.LIBRARY_SPEC: что записано в файлах, без сцен ниши).
def _sound_profile():
    try:
        import channel_profile
        prof = channel_profile.load()
    except Exception:  # noqa: BLE001 — профиль необязателен
        return {}
    snd = prof.get("sound") if isinstance(prof, dict) else None
    return snd if isinstance(snd, dict) else {}


def kind_descriptions():
    """{вид: что слышно} в порядке, в котором виды показываются модели."""
    own = _sound_profile().get("ambience_kinds")
    if isinstance(own, dict) and own:
        return {str(k): str(v) for k, v in own.items()}
    try:
        import sound_library
        return {k: str(v.get("prompt", k.replace("_", " ")))
                for k, v in sound_library.LIBRARY_SPEC.get("ambience", {}).items()}
    except Exception:  # noqa: BLE001
        return {}


# Примеры «не из мира фильма» в заданиях музыкального режиссёра. Тот же
# принцип: примеры этого канала — в профиле, дефолт без ниши.
MUSIC_WORLD_EXAMPLES_DEFAULT = (
    "no futuristic synthesizers in a film about the distant past, no historical instruments "
    "in a film about modern science, no music of a culture that is not the film's own")
MUSIC_CRITIC_WORLD_EXAMPLES_DEFAULT = (
    "futuristic synthesizer in a film about the past, historical instruments in a film about "
    "modern science, music of a culture that is not the film's own")


def music_world_examples():
    v = _sound_profile().get("music_world_examples")
    return str(v) if v else MUSIC_WORLD_EXAMPLES_DEFAULT


def music_critic_world_examples():
    v = _sound_profile().get("music_critic_world_examples")
    return str(v) if v else MUSIC_CRITIC_WORLD_EXAMPLES_DEFAULT


PROMPT = """You are the sound designer of a narrated documentary film (the voice-over is in Russian).
Film: {title}
World of the film: {world}
Chapter: {chapter}
{prev}
The chapter's lines (number, start time, text):
{lines}

Ambience recordings you may use (name: what you hear):
{kinds}

Your job: mark the few moments where the viewer should HEAR the place the story is happening in. A cue starts at a line and plays about 20 seconds, quietly under the voice. A professional film uses ambience sparingly: a wrong or pointless sound is worse than silence.

Put a cue ONLY when all three are true:
a) The line takes the viewer INTO a concrete scene of the story at this moment. Typical forms: "Imagine you are lying in the mud of a battlefield..."; "<date>, <place>." opening an episode of the story; "Now the battle itself..."; an action unfolding right there (they attack, the cavalry gets stuck, the city is looted, the house is set on fire, the ship goes down). Such lines are scenes even if they also give a date or a number.
b) That scene really sounds like one of the recordings, literally: crows over the dead after a battle, the sea when ships cross, a fire when a house burns, a battle when men fight.
c) It is the first line of that scene, not the second or third line of the same scene.

Never put a cue on: explanations; numbers, money, prices, laws, rules, customs; opinions and conclusions; the narrator talking to the viewer; what a chronicler or historian wrote or said; modern research; a battle, a place or an event mentioned only in passing, as an example or a comparison; a person's biography; a line that sums up.

Choose the sound of the scene, not of a word: "the price of a horse" is not cavalry. If no recording fits the scene exactly, leave it silent. Do not use the same recording twice in one chapter unless it is a new scene. A chapter that tells a story gets one cue for every scene it takes the viewer into; a chapter of pure explanation gets none.
Answer with one line per cue and nothing else:
number | sound name
If this chapter should have no cue, answer: none"""

CRITIC_PROMPT = """You are the supervising sound editor of a narrated documentary film (the voice-over is in Russian). An assistant proposed ambience cues for one chapter. Remove every cue that is WRONG; keep the ones that are right. You can only keep or drop; you cannot add or change a cue.

Film: {title}
World of the film: {world}
Chapter: {chapter}
The chapter's lines (number, text):
{lines}

Recordings (name: what you hear):
{kinds}

Proposed cues (cue id: line number | recording):
{cues}

A cue is RIGHT when its line takes the viewer into a concrete scene of the story: "imagine you are lying in the mud of a battlefield", a date and a place opening an episode, an action unfolding there (they attack, the cavalry gets stuck, the city is looted, prisoners are locked in a house and it is set on fire), even when the line also gives a number, a date, or says that someone later recalled or wrote it; and the recording sounds like that scene.

Drop a cue only when:
a) its line is not a scene: an explanation, money, a price, a rule, an opinion, a conclusion, modern research, the narrator addressing the viewer, or an event mentioned only in passing as an example or comparison; or
b) the recording would sound wrong for that scene (a market crowd for a monastery, birds for a battle); or
c) an earlier cue in this chapter already covers the very same scene, or another cue on the same line fits better.

Answer with one line per proposed cue id and nothing else:
cue id | keep
or
cue id | drop"""


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


def world_summary(video_dir):
    """Одна строка о мире фильма из паспорта эпизода (media_plan/world_card.json):
    режиссёр не должен предполагать нишу. Нет паспорта — «not specified»."""
    try:
        with open(os.path.join(video_dir, "media_plan", "world_card.json"), encoding="utf-8") as f:
            wc = json.load(f)
    except (OSError, ValueError):
        return "not specified"
    parts = []
    if wc.get("register"):
        parts.append(str(wc["register"]))
    era = wc.get("era") or {}
    if era.get("from") is not None and era.get("to") is not None:
        parts.append(f"years {era['from']}-{era['to']}")
    inc = (wc.get("culture") or {}).get("include") or []
    if inc:
        parts.append(", ".join(str(x) for x in inc[:5]))
    style = (wc.get("look") or {}).get("style")
    if style:
        parts.append(str(style)[:160])
    return "; ".join(parts) or "not specified"


def _kind_lines(kinds):
    desc = kind_descriptions()
    return "\n".join(f"- {k}: {desc.get(k, k.replace('_', ' '))}" for k in kinds)


def render_prompt(title, chapter, units, prev_tail, kinds, world="not specified"):
    lines = "\n".join(f"{n} [{_fmt_time(t)}] {text}" for n, (_i, t, text) in enumerate(units, 1))
    prev = f"The previous chapter ended with: {prev_tail}\n" if prev_tail else ""
    return PROMPT.format(title=title or "(untitled)", world=world, chapter=chapter, prev=prev,
                         lines=lines, kinds=_kind_lines(kinds))


def render_critic(title, chapter, units, kinds, candidates, world="not specified"):
    """candidates — [(номер строки, вид)] в порядке номеров; id предложения = позиция+1."""
    lines = "\n".join(f"{n} {text}" for n, (_i, _t, text) in enumerate(units, 1))
    cues = "\n".join(f"{k}: {n} | {name}" for k, (n, name) in enumerate(candidates, 1))
    return CRITIC_PROMPT.format(title=title or "(untitled)", world=world, chapter=chapter,
                                lines=lines, kinds=_kind_lines(kinds), cues=cues)


_VERDICT_RE = re.compile(r"^\s*\**\s*(\d+)\s*[|:.\-–][^\n]*?\b(keep|drop)\b", re.I)


def apply_critic(candidates, raw):
    """{номер строки: вид} — только то, что проверка явно оставила. Строки без
    вердикта вычеркнуты: лишний звук хуже тишины. Добавить проверка не может;
    два оставленных вида на одной строке — первый."""
    keep = set()
    for line in _strip_think(raw).splitlines():
        m = _VERDICT_RE.match(line)
        if m and m.group(2).lower() == "keep":
            keep.add(int(m.group(1)))
    out = {}
    for k, (n, name) in enumerate(candidates, 1):
        if k in keep and n not in out:
            out[n] = name
    return out


def merge_drafts(drafts):
    """Объединение черновиков: [(номер строки, вид)] без повторов, по номерам."""
    seen, out = set(), []
    for d in drafts:
        for n, name in d.items():
            if (n, name) not in seen:
                seen.add((n, name))
                out.append((n, name))
    return sorted(out, key=lambda c: c[0])


def thin_repeats(picks):
    """Один вид звука — один раз на главу: повтор того же звука внутри главы
    слышится как штамп. Остаётся первое вхождение."""
    seen, out = set(), {}
    for n, name in sorted(picks.items()):
        if name != "music" and name in seen:
            continue
        seen.add(name)
        out[n] = name
    return out


_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)


def _strip_think(raw):
    """Рассуждение модели в ответе не разбирается: строки вида «2 | keep»
    внутри него — черновые мысли, а не ответ."""
    return _THINK_RE.sub("", raw or "")


_LINE_RE = re.compile(r"^\s*\**\s*(\d+)\s*[|:\-–]\s*([a-z_]+)", re.I)


def parse_answer(raw, n_units, kinds, music=False):
    """{номер_строки: вид} — только известные виды и «music». Чужое молча
    пропускается: недоверие к формату, а не к главе целиком."""
    out = {}
    allowed = set(kinds) | {"music"} if music else set(kinds)
    for line in _strip_think(raw).splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        n, name = int(m.group(1)), m.group(2).lower()
        if 1 <= n <= n_units and name in allowed and n not in out:
            out[n] = name
    return out


def _signature(*parts):
    """Отпечаток входов плана: тот же сценарий, библиотека, мир и модели —
    тот же план с диска. Без него каждый рендер спрашивал бы заново, и сбой
    шлюза или иной ответ модели менял бы звук готового эпизода."""
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, default=list).encode("utf-8")).hexdigest()[:20]


def _cache_path(cache_dir, model, prompt):
    key = hashlib.sha256(f"{PLAN_VERSION}|{model}|{prompt}".encode("utf-8")).hexdigest()[:24]
    return os.path.join(cache_dir, key + ".txt")


def ask(gateway, model, prompt, cache_dir, est=EST_PROMPT_TOKENS, max_tokens=MAX_TOKENS):
    """Ответ на главу через цепочку моделей; кэш по содержимому вопроса.
    model — имя модели (цепочка llm_gateway.text_chain) или кортеж моделей."""
    import llm_gateway

    def cached(m):
        cp = _cache_path(cache_dir, m, prompt)
        if os.path.exists(cp):
            with open(cp, encoding="utf-8") as f:
                return f.read()
        return None
    chain = tuple(model) if isinstance(model, (tuple, list)) else llm_gateway.text_chain(model)
    got = llm_gateway.chat_fallback(gateway, chain,
                                    [{"type": "text", "text": prompt}], max_tokens,
                                    est, cached=cached, timeout=CHAIN_TIMEOUT_SEC)
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
                # ровно «TITLE:» — «TITLE_CARD:» (текст заставки названия) не название
                if re.match(r"TITLE\s*:", line.strip(), re.I):
                    return line.split(":", 1)[-1].strip()
    except OSError:
        pass
    return os.path.basename(os.path.normpath(video_dir))


def plan_path(video_dir):
    return os.path.join(video_dir, "media_plan", "sound_plan.json")


def plan_episode(video_dir, blocks, sub_starts, gateway, kinds, draft_models=DRAFT_MODELS,
                 critic_models=CRITIC_MODELS, workers=4):
    """Черновики нескольких моделей по главам (параллельно) -> объединение ->
    проверка вычёркивает -> план на диск.

    План: {"version", "models", "cues": [{"section", "text", "name"}]} —
    событие привязано к ТЕКСТУ фразы. Проверка не ответила — глава без
    звуков (лишний звук хуже тишины), с предупреждением."""
    from concurrent.futures import ThreadPoolExecutor
    import llm_gateway
    title = episode_title(video_dir)
    world = world_summary(video_dir)
    cache_dir = os.path.join(video_dir, "media_plan", "sound_director_cache")
    chapters = chapter_units(blocks, sub_starts)
    sig = plan_signature(video_dir, blocks, sub_starts, kinds, draft_models, critic_models)
    old = load_plan(video_dir)
    if old and old.get("signature") == sig and not (old.get("stats") or {}).get("critic_failed") \
            and not (old.get("stats") or {}).get("draft_failed"):
        return old          # тот же вопрос, план без сбоев — тот же звук, без вызовов
    jobs = []
    for k, (sec, units) in enumerate(chapters):
        prev_tail = chapters[k - 1][1][-1][2] if k else ""
        jobs.append((sec, units, render_prompt(title, sec, units, prev_tail, kinds, world)))
    stats = {"draft": 0, "kept": 0, "critic_failed": 0, "draft_failed": 0}

    def one(job):
        sec, units, prompt = job
        drafts, failed_drafts, used = [], 0, set()
        for m in draft_models:
            try:
                raw, _hit, who = ask(gateway, (m, m), prompt, cache_dir)
            except llm_gateway.PaymentRequired:
                raise
            except Exception as e:  # noqa: BLE001
                failed_drafts += 1
                print(f"  звуковой режиссёр: черновик {m} главы «{sec[:40]}» не получен ({type(e).__name__})")
                continue
            used.add(who)
            drafts.append(parse_answer(raw, len(units), kinds))
        candidates = merge_drafts(drafts)
        picks, critic_failed = {}, False
        if candidates:
            try:
                verdict, _h, who = ask(gateway, tuple(critic_models),
                                       render_critic(title, sec, units, kinds, candidates, world),
                                       cache_dir)
                used.add(who)
                picks = apply_critic(candidates, verdict)
            except llm_gateway.PaymentRequired:
                raise
            except Exception as e:  # noqa: BLE001
                critic_failed = True
                print(f"  звуковой режиссёр: проверка главы «{sec[:40]}» не ответила "
                      f"({type(e).__name__}) — глава без звуков")
        proposed = [{"section": sec, "text": units[n - 1][2], "name": name,
                     "kept": picks.get(n) == name} for n, name in candidates]
        return (sec, units, thin_repeats(picks), used, len(candidates), critic_failed, failed_drafts,
                proposed)

    cues, models, proposed_all = [], set(), []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for sec, units, picks, used, n_draft, cfail, dfail, proposed in pool.map(one, jobs):
            proposed_all += proposed
            models |= used
            stats["draft"] += n_draft
            stats["kept"] += len(picks)
            stats["critic_failed"] += int(cfail)
            stats["draft_failed"] += dfail
            for n, name in sorted(picks.items()):
                _i, _t, text = units[n - 1]
                cues.append({"section": sec, "text": text, "name": name})
    plan = {"version": PLAN_VERSION, "signature": sig, "models": sorted(models), "stats": stats,
            "cues": cues, "proposed": proposed_all}
    path = plan_path(video_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    print(f"  звуковой режиссёр: предложено {stats['draft']} звуков, после проверки {stats['kept']}"
          + (f"; проверка не ответила в {stats['critic_failed']} главах" if stats["critic_failed"] else ""))
    return plan


def plan_signature(video_dir, blocks, sub_starts, kinds, draft_models=DRAFT_MODELS,
                   critic_models=CRITIC_MODELS):
    """Отпечаток входов плана атмосферы — ОДНА формула и для составления
    плана, и для проверки плана с диска без ключа шлюза."""
    chapters = chapter_units(blocks, sub_starts)
    return _signature(PLAN_VERSION, draft_models, critic_models, sorted(kinds),
                      world_summary(video_dir), episode_title(video_dir),
                      [(sec, [u[2] for u in units]) for sec, units in chapters])


def stale_plan_note(plan, matched, expected_sig=None):
    """Строка-предупреждение, если план с диска не про этот сценарий, или None.

    Аудит 03_plen (02.10): без ключа шлюза план брался с диска без проверки,
    и после правки сценария события молча выпадали (сверка по точному тексту
    фразы). Звук от этого не меняется — меняется то, что человек это видит."""
    if not plan:
        return None
    n_plan = len(plan.get("cues") or [])
    lost = n_plan - len(matched)
    notes = []
    if expected_sig is not None and plan.get("signature") and plan["signature"] != expected_sig:
        notes.append("план составлен для другой версии сценария/библиотеки/паспорта")
    if lost > 0:
        notes.append(f"{lost} из {n_plan} событий не нашли свою фразу в сценарии и выпали")
    return "; ".join(notes) or None


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
                out.append((i, "amb", name))
                break
    return out


def _stable_seed(*parts):
    h = 0
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def ambience_events(cues, sub_starts, total_dur, available=None,
                    event_sec=AMB_EVENT_SEC, preroll=AMB_PREROLL_SEC, dropped=None):
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

    dropped — список, куда записывается КАЖДОЕ решение «не звучит» с
    причиной (аудит 03_plen 02.10: rain_mud на «Азенкур» пропал по правилу
    12 с без единой строки). Звук от него не меняется.
    """
    def drop(i, name, t, reason):
        if dropped is not None:
            dropped.append({"block": i, "kind": name, "start": round(t, 3), "reason": reason})
    raw = []
    for i, typ, name in cues:
        if typ != "amb" or not name:
            continue
        if available is not None and name not in available:
            drop(i, name, 0.0, "no_recording")
            continue
        if i >= len(sub_starts):
            drop(i, name, 0.0, "no_phrase")
            continue
        raw.append((max(0.0, float(sub_starts[i]) - preroll), name, i))
    raw.sort()
    events = []
    for start, name, i in raw:
        if events and start - events[-1]["start"] < AMB_MIN_START_GAP_SEC:
            drop(i, name, start, f"start_gap<{AMB_MIN_START_GAP_SEC:g}s:{events[-1]['kind']}")
            continue
        if events and events[-1]["kind"] == name and start < events[-1]["end"]:
            events[-1]["end"] = min(float(total_dur), start + event_sec)
            continue
        events.append({"start": start, "end": min(float(total_dur), start + event_sec),
                       "kind": name, "block": i})
    for a, b in zip(events, events[1:]):
        a["end"] = min(a["end"], b["start"] + AMB_CROSSFADE_SEC)
    for e in events:
        if e["end"] - e["start"] < AMB_MIN_EVENT_SEC:
            drop(e["block"], e["kind"], e["start"], f"shorter_than_{AMB_MIN_EVENT_SEC:g}s")
    events = [e for e in events if e["end"] - e["start"] >= AMB_MIN_EVENT_SEC]
    seen = {}
    for e in events:
        seen[e["kind"]] = seen.get(e["kind"], -1) + 1
        e["occ"] = seen[e["kind"]]          # n-е событие вида: запись и отрезок (ambience_event_pick)
        e["seed"] = seen[e["kind"]] * 7919 + _stable_seed(e["kind"]) % 97
        e["start"], e["end"] = round(e["start"], 3), round(e["end"], 3)
    return events


def episode_sound_cues(video_dir, blocks, sub_starts, gateway=None, kinds=None, verbose=True):
    """(cues, источник): теги автора -> план модели (свежий или с диска).

    Модель спрашивается, только если у автора тегов нет, есть шлюз и план
    на диске не покрывает текущие фразы (кэш вопросов делает повтор
    бесплатным; новый вопрос — только для изменённых глав)."""
    own = inline_cues(blocks)
    own_music = [c for c in own if c[1] == "music"]
    if any(c[1] == "amb" for c in own):
        return own, "author"
    plan = None
    if gateway is not None and kinds:
        try:
            plan = plan_episode(video_dir, blocks, sub_starts, gateway, kinds)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ВНИМАНИЕ: звуковой режиссёр не отработал ({type(e).__name__}: "
                      f"{str(e)[:160]}) — берётся план с диска, если есть")
    from_disk = plan is None
    if plan is None:
        plan = load_plan(video_dir)
    if plan is None:
        return own_music, "none"
    matched = plan_cues(plan, blocks)
    note = stale_plan_note(plan, matched,
                           plan_signature(video_dir, blocks, sub_starts, kinds)
                           if from_disk and kinds else None)
    if note and verbose:
        print(f"  ВНИМАНИЕ: план атмосферы с диска — {note}. "
              f"Пересоставит рендер с LLM_GATEWAY_API_KEY.")
    return matched + own_music, "director"


def write_inline(script_path, plan):
    """Проставить план в сценарий тегами [amb:]/[music:] перед своей фразой
    (.bak перед записью). Только там, где текст фразы встречается в файле
    ровно один раз; теги автора не трогаются."""
    with open(script_path, encoding="utf-8") as f:
        src = f.read()
    done, skipped = 0, []
    for c in (plan or {}).get("cues") or []:
        text = c.get("text", "")
        tag = f"[amb:{c.get('name')}]"
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


# ======================================================================
# МУЗЫКАЛЬНЫЙ РЕЖИССЁР: какой трек из библиотеки и где (любая ниша).
#
# Библиотека — scripts/music_library.py: у каждого трека текстовая карточка
# (жанр и настроения каталога + что слышит AST: инструменты, характер,
# есть ли бит). Режиссёр без ушей выбирает по карточкам, как музыкальный
# редактор по описаниям в библиотеке; два черновика разных моделей,
# проверка выбирает из них по каждому месту или отказывается от обоих.
# Длины не зашиты: врезка кончается на границе фразы рядом с целевой
# длиной, а не посреди слова.
# ======================================================================
MUSIC_PLAN_VERSION = 4
MUSIC_DRAFT_MODELS = ("qwen/qwen3.8-max", "ag/gemini-3.7-flash-low")
MUSIC_CRITIC_MODELS = ("ag/gemini-3.7-flash-low", "qwen/qwen3.8-max")
MUSIC_INTRO_TARGET_SEC = 36.0          # владелец 02.10: «чуть короче, на пару секунд» (было 40)
MUSIC_INTRO_RANGE = (30.0, 42.0)
MUSIC_STING_TARGET_SEC = 14.0          # было 16
MUSIC_STING_RANGE = (11.0, 18.0)
MUSIC_OUTRO_TARGET_SEC = 38.0
MUSIC_OUTRO_RANGE = (32.0, 46.0)
MUSIC_MAX_CARDS = 260
# Вопрос с сотней карточек Qwen 3.8 Max обдумывает дольше 8000 токенов и
# отдаёт пустой ответ (живой прогон 03_plen 02.10). Платится только сказанное.
MUSIC_MAX_TOKENS = 20000

MUSIC_PROMPT = """You are the music supervisor of a narrated documentary film (the voice-over is in Russian).
Film: {title}
World of the film: {world}

Chapters (number, start time, title, how it begins ... how it ends):
{chapters}

Music library (id | length | description):
{cards}

Choose music from the library:
- intro: plays under the first ~35 seconds, under the opening lines of chapter 1. It must fit the mood of those opening lines (read them: a grim or tense opening needs dark, calm, slow or melancholic music, never a lively dance tune) and the world of the film (prefer instruments and style of that world).
- outro: plays under the last ~40 seconds and closes the story.
- sting: at most 3 chapters that open a new story in a concrete place and time; a short (~14 s) music entrance under the start of that chapter. Leave out chapters that open with an explanation.
- bed: for EVERY chapter, a background track that plays very quietly under the whole chapter. It must have no beat and no busy melody, and fit the chapter's mood. Neighbouring chapters may share a bed when the mood continues.

Rules:
1. The music must belong to the film's world: {world_examples}.
2. The mood must match the moment ("owner hears:" in a description is a human verdict by ear and outranks every other word of it): nothing lively, cheerful or dance-like under danger, death, cruelty or loss; nothing scary under a calm explanation.
3. Use different tracks for the intro, the outro and the stings.
4. Use everything in the description, the title included: a title like "Happy ...", "Bangkok ..." or "Indian ..." tells the mood or the culture.

Answer with lines only, nothing else:
intro | track id
outro | track id
sting | chapter number | track id
bed | chapter number | track id"""

MUSIC_CRITIC_PROMPT = """You are the supervising music editor of a narrated documentary film (the voice-over is in Russian). Two assistants chose music from the library for each slot. For every slot pick the better option, or reject both if neither fits.

Film: {title}
World of the film: {world}

Chapters (number, start time, title, how it begins ... how it ends):
{chapters}

Slots and options (slot | option letter: track id | description):
{options}

A choice is wrong if its mood ("owner hears:" is a human verdict by ear and outranks the rest of the description) contradicts the lines it plays under (a lively dance tune under a battle, a fall or a death), or if the track does not belong to the film's world ({world_examples}; the title counts too), if its mood contradicts the moment (cheerful or light under death or cruelty, scary under a calm explanation), or, for a bed, if it has a beat or a busy melody. If both options are fine, pick the one that fits the world and the moment more exactly.

Answer with one line per slot and nothing else:
slot | A
slot | B
slot | none"""


def music_prompt_text(**kw):
    return MUSIC_PROMPT.format(world_examples=music_world_examples(), **kw)


def music_critic_prompt_text(**kw):
    return MUSIC_CRITIC_PROMPT.format(world_examples=music_critic_world_examples(), **kw)


def _chapter_lines(chapters):
    rows = []
    for k, (sec, units) in enumerate(chapters, 1):
        first = units[0][2][:160] if units else ""
        last = units[-1][2][:110] if len(units) > 1 else ""
        t = units[0][1] if units else 0.0
        rows.append(f"{k} [{_fmt_time(t)}] {sec}: {first} ... {last}")
    return "\n".join(rows)


def _card_lines(tracks):
    return "\n".join(f"{t['id']} | {int(t['dur'])}s | {t['card']}" for t in tracks[:MUSIC_MAX_CARDS])


_MUSIC_LINE_RE = re.compile(
    r"^\s*\**\s*(intro|outro|sting|bed)\s*\|\s*(?:(\d+)\s*\|\s*)?([A-Za-z0-9_]+)\s*\**\s*$", re.I)


def parse_music_answer(raw, n_chapters, track_ids):
    """{слот: id}; слот — "intro", "outro", ("sting", n), ("bed", n)."""
    out = {}
    for line in _strip_think(raw).splitlines():
        m = _MUSIC_LINE_RE.match(line)
        if not m:
            continue
        role, num, tid = m.group(1).lower(), m.group(2), m.group(3)
        if tid not in track_ids:
            continue
        if role in ("intro", "outro"):
            out.setdefault(role, tid)
        elif num and 1 <= int(num) <= n_chapters:
            out.setdefault((role, int(num)), tid)
    return out


def _slot_name(slot):
    return slot if isinstance(slot, str) else f"{slot[0]} {slot[1]}"


_MUSIC_VERDICT_RE = re.compile(r"^\s*\**\s*(intro|outro|sting \d+|bed \d+)\s*\|\s*(A|B|none)\b", re.I)


def music_options(drafts):
    """[(слот, [id A, id B])] — варианты черновиков без повторов, по порядку слотов."""
    slots = {}
    for d in drafts:
        for slot, tid in d.items():
            opts = slots.setdefault(slot, [])
            if tid not in opts:
                opts.append(tid)

    def order(slot):
        if slot == "intro":
            return (0, 0)
        if slot == "outro":
            return (3, 0)
        return (1 if slot[0] == "sting" else 2, slot[1])
    return sorted(slots.items(), key=lambda kv: order(kv[0]))


def apply_music_critic(options, raw):
    choice = {}
    for line in _strip_think(raw).splitlines():
        m = _MUSIC_VERDICT_RE.match(line)
        if m:
            choice[m.group(1).lower()] = m.group(2).upper()
    out = {}
    for slot, opts in options:
        c = choice.get(_slot_name(slot))
        if c in ("A", "B") and "AB".index(c) < len(opts):
            out[slot] = opts["AB".index(c)]
    return out


def enforce_music_rules(picks, tracks_by_id):
    """Правила, которые не доверяются модели: подложка без бита; вступление,
    финал и врезки — разные треки; не больше MUSIC_MAX_CHAPTER_CUES врезок."""
    out, used = {}, set()
    for slot in ("intro", "outro"):
        tid = picks.get(slot)
        if tid and tid not in used:
            out[slot] = tid
            used.add(tid)
    stings = sorted((s for s in picks if isinstance(s, tuple) and s[0] == "sting"), key=lambda s: s[1])
    for s in stings[:MUSIC_MAX_CHAPTER_CUES]:
        if picks[s] not in used:
            out[s] = picks[s]
            used.add(picks[s])
    for s, tid in picks.items():
        if isinstance(s, tuple) and s[0] == "bed":
            if tracks_by_id.get(tid, {}).get("card", "").endswith("no beat"):
                out[s] = tid
    return out


def plan_music(video_dir, blocks, sub_starts, gateway, tracks, draft_models=MUSIC_DRAFT_MODELS,
               critic_models=MUSIC_CRITIC_MODELS):
    """Музыкальный план эпизода на диск (media_plan/music_plan.json):
    {"version", "chapters": [названия], "intro", "outro", "stings": [{section, id}],
     "beds": [{section, id}]}."""
    import llm_gateway
    title = episode_title(video_dir)
    world = world_summary(video_dir)
    cache_dir = os.path.join(video_dir, "media_plan", "sound_director_cache")
    chapters = chapter_units(blocks, sub_starts)
    ids = {t["id"] for t in tracks}
    by_id = {t["id"]: t for t in tracks}
    sig = music_plan_signature(video_dir, blocks, sub_starts, tracks, draft_models, critic_models)
    old = load_music_plan(video_dir)
    if old and old.get("signature") == sig and old.get("complete"):
        return old
    prompt = music_prompt_text(title=title or "(untitled)", world=world,
                                 chapters=_chapter_lines(chapters), cards=_card_lines(tracks))
    drafts = []
    est = max(EST_PROMPT_TOKENS, len(prompt) // 3)
    for m in draft_models:
        try:
            raw, _h, _w = ask(gateway, (m, m), prompt, cache_dir, est=est,
                              max_tokens=MUSIC_MAX_TOKENS)
        except llm_gateway.PaymentRequired:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"  музыкальный режиссёр: черновик {m} не получен ({type(e).__name__})")
            continue
        drafts.append(parse_music_answer(raw, len(chapters), ids))
    options = music_options(drafts)
    picks = {}
    complete = len(drafts) == len(draft_models)
    if options:
        opt_lines = "\n".join(
            f"{_slot_name(slot)} | " + " ; ".join(f"{'AB'[k]}: {tid} | {by_id[tid]['card']}"
                                                 for k, tid in enumerate(opts[:2]))
            for slot, opts in options)
        try:
            verdict, _h, _w = ask(gateway, tuple(critic_models),
                                  music_critic_prompt_text(title=title or "(untitled)", world=world,
                                                             chapters=_chapter_lines(chapters),
                                                             options=opt_lines), cache_dir)
            picks = apply_music_critic([(s, o[:2]) for s, o in options], verdict)
        except llm_gateway.PaymentRequired:
            raise
        except Exception as e:  # noqa: BLE001
            complete = False
            print(f"  музыкальный режиссёр: проверка не ответила ({type(e).__name__}) — без музыки")
    picks = enforce_music_rules(picks, by_id)
    secs = [sec for sec, _u in chapters]
    plan = {"version": MUSIC_PLAN_VERSION, "signature": sig, "complete": complete, "chapters": secs,
            "intro": picks.get("intro"), "outro": picks.get("outro"),
            "stings": [{"section": secs[s[1] - 1], "id": tid} for s, tid in sorted(
                ((s, t) for s, t in picks.items() if isinstance(s, tuple) and s[0] == "sting"))],
            "beds": [{"section": secs[s[1] - 1], "id": tid} for s, tid in sorted(
                ((s, t) for s, t in picks.items() if isinstance(s, tuple) and s[0] == "bed"))],
            "options": [{"slot": _slot_name(s), "ids": o} for s, o in options]}
    path = os.path.join(video_dir, "media_plan", "music_plan.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    print(f"  музыкальный режиссёр: вступление {plan['intro']}, финал {plan['outro']}, "
          f"врезок {len(plan['stings'])}, подложек {len(plan['beds'])} из {len(secs)} глав")
    return plan


def music_plan_signature(video_dir, blocks, sub_starts, tracks, draft_models=MUSIC_DRAFT_MODELS,
                         critic_models=MUSIC_CRITIC_MODELS):
    """Отпечаток входов музыкального плана — одна формула для составления и
    для проверки плана с диска."""
    chapters = chapter_units(blocks, sub_starts)
    return _signature(MUSIC_PLAN_VERSION, draft_models, critic_models, world_summary(video_dir),
                      episode_title(video_dir), sorted(t["id"] for t in tracks),
                      [(sec, [u[2] for u in units]) for sec, units in chapters])


def stale_music_note(mplan, blocks, expected_sig=None):
    """Предупреждение, если музыкальный план с диска не про этот сценарий."""
    if not mplan:
        return None
    secs = {str(b.get("section", "")) for b in blocks}
    lost = [x["section"] for x in (mplan.get("stings") or []) + (mplan.get("beds") or [])
            if x.get("section") not in secs]
    notes = []
    if expected_sig is not None and mplan.get("signature") and mplan["signature"] != expected_sig:
        notes.append("план составлен для другой версии сценария/библиотеки/паспорта")
    if lost:
        notes.append(f"{len(lost)} врезок/подложек привязаны к главам, которых в сценарии нет")
    return "; ".join(notes) or None


def _snap_end(sub_starts, start, target, lo, hi, total):
    """Конец врезки — на начале фразы (то есть в паузе перед ней), ближайшем
    к start+target в окне [start+lo, start+hi]; нет такой — start+target."""
    want = start + target
    best = None
    for t in sub_starts:
        t = float(t)
        if start + lo <= t <= start + hi and (best is None or abs(t - want) < abs(best - want)):
            best = t
    return min(float(total), best if best is not None else want)


def music_plan_cues(mplan, blocks, sub_starts, total, tracks_by_id, dropped=None):
    """Врезки с путями: [{start, end, role, path, id}] и подложки
    [{start, end, path, id, section}] на шкале аудио.

    dropped — куда записать каждую врезку, снятую расписанием, с причиной
    (аудит 03_plen 02.10: врезка BLOCK 7 срезана правилом 150 с молча)."""
    def drop(role, tid, section, reason):
        if dropped is not None:
            dropped.append({"role": role, "id": tid, "section": section, "reason": reason})
    if not mplan:
        return [], []
    total = float(total)
    first = {}
    for i, b in enumerate(blocks):
        first.setdefault(str(b.get("section", "")), i)
    cues = []

    def path_of(tid):
        t = tracks_by_id.get(tid)
        if not t:
            return None
        import music_library
        p = music_library.track_path(t)
        return p if os.path.exists(p) else None
    intro_end = 0.0
    if mplan.get("intro") and path_of(mplan["intro"]):
        intro_end = _snap_end(sub_starts, 0.0, MUSIC_INTRO_TARGET_SEC, *MUSIC_INTRO_RANGE, total)
        cues.append({"start": 0.0, "end": round(intro_end, 3), "role": "intro",
                     "id": mplan["intro"], "path": path_of(mplan["intro"])})
    outro_start = None
    if mplan.get("outro") and path_of(mplan["outro"]):
        want = total - MUSIC_OUTRO_TARGET_SEC
        cands = [float(t) for t in sub_starts
                 if total - MUSIC_OUTRO_RANGE[1] <= float(t) <= total - MUSIC_OUTRO_RANGE[0]]
        outro_start = min(cands, key=lambda t: abs(t - want)) if cands else max(0.0, want)
        if outro_start > intro_end + 5.0:
            cues.append({"start": round(outro_start, 3), "end": round(total, 3), "role": "outro",
                         "id": mplan["outro"], "path": path_of(mplan["outro"])})
        else:
            drop("outro", mplan["outro"], None, "overlaps_intro")
            outro_start = None
    last = None
    for s in mplan.get("stings") or []:
        i = first.get(s.get("section"))
        p = path_of(s.get("id"))
        if i is None or p is None or i >= len(sub_starts):
            drop("chapter", s.get("id"), s.get("section"),
                 "no_section" if i is None or i >= len(sub_starts) else "no_file")
            continue
        t = max(0.0, float(sub_starts[i]) - 1.0)
        half = MUSIC_CHAPTER_MIN_SPACING_SEC / 2
        if t < intro_end + half:
            drop("chapter", s["id"], s.get("section"), f"within_{half:g}s_of_intro")
            continue
        if outro_start is not None and t > outro_start - half:
            drop("chapter", s["id"], s.get("section"), f"within_{half:g}s_of_outro")
            continue
        if last is not None and t - last < MUSIC_CHAPTER_MIN_SPACING_SEC:
            drop("chapter", s["id"], s.get("section"),
                 f"within_{MUSIC_CHAPTER_MIN_SPACING_SEC:g}s_of_previous_sting")
            continue
        end = _snap_end(sub_starts, t, MUSIC_STING_TARGET_SEC, *MUSIC_STING_RANGE, total)
        cues.append({"start": round(t, 3), "end": round(end, 3), "role": "chapter",
                     "id": s["id"], "path": p})
        last = t
    cues.sort(key=lambda c: c["start"])
    for k, c in enumerate(cues):
        c["seed"] = k
    beds = []
    secs = list(first.items())
    by_sec = {b["section"]: b["id"] for b in mplan.get("beds") or []}
    for k, (sec, i) in enumerate(secs):
        start = float(sub_starts[i]) if i < len(sub_starts) else 0.0
        end = float(sub_starts[secs[k + 1][1]]) if k + 1 < len(secs) else total
        tid = by_sec.get(sec)
        p = path_of(tid) if tid else None
        if tid and p is None:
            drop("bed", tid, sec, "no_file")
        if p is None or end - start < 1.0:
            continue
        if beds and beds[-1]["id"] == tid and abs(beds[-1]["end"] - start) < 0.01:
            beds[-1]["end"] = round(end, 3)      # та же подложка дальше — без стыка
            continue
        beds.append({"start": round(start, 3), "end": round(end, 3), "id": tid, "path": p,
                     "section": sec})
    if beds:
        beds[0]["start"] = 0.0
        beds[-1]["end"] = round(total, 3)
    return cues, beds


def load_music_plan(video_dir):
    try:
        with open(os.path.join(video_dir, "media_plan", "music_plan.json"), encoding="utf-8") as f:
            p = json.load(f)
        return p if p.get("version") == MUSIC_PLAN_VERSION else None
    except (OSError, ValueError):
        return None
