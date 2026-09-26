#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запросы к стокам на КАЖДУЮ фразу — пишет текстовая модель, по главам.

ЗАЧЕМ. Слот спрашивает источники запросами, и до сих пор их было два вида,
оба слабые:
  * авторские запросы секции (=== PEXELS QUERIES ===) раздаются фразам
    сопоставлением текста с текстом, а у эмбеддинг-моделей оно вырождается.
    Живой случай 23.09, хук эпизода 94: «medieval armour joint gap» достался
    фразе про исход поединка, а фраза «Клинок влетает в узкую щель между
    пластинами доспеха» получила «medieval steel dagger blade»;
  * перевод брифа (brief_to_stock_query) обрезает описание до пяти слов и
    теряет предмет: «an arrow glancing off a dented steel breastplate» ->
    «medieval arrow glancing off dented» (нагрудник пропал). На 142 брифах
    эпизода 02 последнее значимое слово теряется в 86 случаях.
И главное — ни один из них не знает, что РЕАЛЬНО лежит в стоках. Зонд 23.09:
на «рыцарь падает в грязь» запрос «knight face down mud» даёт современных
людей в грязи, а «medieval battle reenactment» — рыцаря, ползущего по земле,
и бой реконструкторов. Такое понимание есть только у модели.

КАК. Глава уходит одним вопросом (фразы по порядку, бриф каждой, строка
мира из паспорта эпизода), ответ — строка на фразу: 4 запроса от точного к
общему. План — media_plan/stock_queries.json, ключ юнита — ТЕКСТ фразы
(shot_planner_llm.unit_key, та же дисциплина: номера сдвигаются от правок).
Рендер плана не спрашивает модель, только читает файл.

БЕЗ НИШИ В КОДЕ. В вопросе нет ни слова о теме канала: мир приходит строкой
из паспорта эпизода (world_card.judge_setting), а что лежит в стоках для
этого мира, модель решает сама.

Нет плана или фраза в нём не найдена — слот идёт прежним путём, байт-в-байт.

ВЕРСИЯ 3 — СПЕЦИФИКАЦИЯ КАДРА (24.09). Версия 2 писала «лестницу замен» и
прямо велела модели запасной ступенью брать «предмет без действия». На фразе
«Стрела скользит по нагруднику» это дало ступень «помятый нагрудник в
галерее»: главное (стрела) выброшено по инструкции, проверка засчитала
нагрудник «близкой заменой», и он встал в ролик. Решать, чем пожертвовать,
нельзя ни словарём, ни порядком в коде — это смысл конкретной фразы.

Теперь модель на фразу пишет:
  * focus — что новое зритель должен увидеть (понятое в контексте главы:
    местоимения разрешены; фокус — агент, объект, место или состояние);
  * claims — 2-5 утверждений, которые проверяются взглядом на картинку, по
    убыванию важности; первое — сам фокус; must/should; одно утверждение
    может требовать движения (motion) — его выполняет только видео;
  * queries — запросы, у каждого помечено, какие утверждения он ищет.
Замена больше не пишется заранее: побеждает кадр, выполнивший больше важных
утверждений (shot_judge.claims_vector), и нагрудник без стрелы проигрывает
любой картине со стрелой. Код смысл не решает — он сравнивает векторы в
порядке, который задала спецификация.
"""
import argparse
import hashlib
import json
import os
import re
import sys

PLAN_NAME = "stock_queries.json"
CACHE_DIR_NAME = "stock_query_cache"
PLAN_VERSION = 4
TIERS = ("must", "should")
# Тип кадра запроса — тот же словарь, что у маршрутизации источников
# (shot_types.SHOT_TYPES): «any» модель не пишет, его значит отсутствие поля.
SHOT_KINDS = ("object", "scene", "illustration", "map", "texture")
# Главное утверждение фразы — отдельное обязательное поле ответа, а не
# «первое в списке»: замер 24.09 (эп.94) — при правиле «первое утверждение —
# главное» Gemini Flash и Qwen Max на фразе «Стрела скользит по нагруднику»
# ставили первым «виден нагрудник». Прямой вопрос «что одно на картинке
# напомнит эту фразу» модель решает отдельно, а не порядком.
CORE_ID = "core"
# Размеры ответа — цена и внимание модели, а не смысл: утверждений больше
# пяти человек у кадра не проверяет, запросов больше шести — это уже
# расход квоты стоков на одну фразу.
MAX_CLAIMS = 5
MAX_QUERIES = 6
QUERY_MAX_WORDS = 7
# Модель выбрана замером 24.09 (58 фраз трёх ниш — кинжал, психология,
# глубоководье — без авторских описаний кадра, одна инструкция на всех):
# DeepSeek v4 Flash разобрал 58/58 во всех прогонах, главное конкретное и
# снимаемое, ~2 тыс. токенов баланса на 58 фраз; Gemini 3.7 Flash по смыслу
# наравне, но ~35 тыс.; Qwen 3.8 Max (стоял здесь раньше) — 46/58 и «виден
# человек» на половине абстрактных фраз. Качество оценено глазами Claude,
# не разметкой владельца. Сменить — --model.
DEFAULT_MODEL = "ds/deepseek-v4-flash"
# Запас выхода с рассуждением: DeepSeek v4 Flash рассуждает до ответа, и
# версия 4 пишет на фразу больше полей (смысл, тип чтения, ловушки). При
# 8000 глава из 13 фраз эп.95 обрывалась на шестой (26.09).
MAX_TOKENS = 16000
EST_PROMPT_TOKENS = 2500

SPEC_PROMPT = """You direct the visuals of a documentary video.
Episode: «{title}». Setting: {setting}.
{direction}Below are the narration lines of one chapter, in order{prev}. A line may come with a shot the author suggests: use it when it shows what the line means; if it only illustrates one word of a figurative or abstract line, or goes against the film direction, show the meaning instead.

For EVERY numbered line decide what the viewer must SEE while hearing it.

meaning — what the line says in this film, in plain English, with every pronoun and reference resolved from the lines around it ("it" becomes "the dagger"). 4 to 16 words.

reading — "literal" if what the line says can be filmed as it is said; "figurative" if it speaks through a metaphor, idiom or image that must not be shown word for word; "abstract" if it states an idea, feeling, number, process or argument. For "figurative" and "abstract" lines the picture shows the meaning, never the words.

focus — the new thing this line says, understood in the context of the chapter (resolve pronouns and references from the lines around it). 3 to 12 English words.

core — WHO or WHAT must be visible: the single thing (an object, a person, an animal, a place) that, even alone in a picture, still makes the viewer think of this line — with the state that defines it, if any ("an exhausted person", "a burnt letter"). Name the thing, not an event: what it does goes into the claims. Ask yourself: if the picture could show only one thing, which one? When the line is about something happening to, on or around something else, the core is what the line is about — usually the thing that moves, acts or changes — not the surface, place or object it happens on. When the line is abstract (a feeling, an idea, a process, an argument), the core is a concrete situation, a bodily sign or an object left behind that a camera can photograph and a viewer reads as this idea — never a bare "a person is visible" or an invisible thing like "a memory" or "a brain decision": say what makes the picture show THIS line ("a person slumped over an untouched plate", "a crumpled paper covered in red corrections"). Never make words, captions, labels, signs or logos in the picture part of the core or of a claim — the viewer hears the words, the picture shows things — unless the line is about that very document, chart, headline, sign or screen. Write it as a statement: "a ball is visible". Follow the film direction: when the film should look cinematic, prefer real people, places and moments to diagrams, icons and toy models, unless the line is about that model, diagram or image itself. Two neighbouring lines never get the same core unless the second line is still about the very same thing.

claims — 1 to {c1} more statements checkable by looking at the picture, most important first. Each checks ONE thing (an object, an action, a place, a detail) and does not repeat the core. "tier": "must" if without it the picture does not show this line, "should" if it only makes the picture better. If the line is about a movement that only footage can show, one claim has "motion": true and describes this movement; lines about objects, places or states have no motion claim.

traps — 1 to 3 pictures that a search for this line would likely return and that look related but are WRONG for it: a generic look-alike, the literal word of a metaphor, a toy, cartoon or diagram version of a real thing, the wrong moment (a clock showing another time), another era, culture or genre. Each 2 to 10 English words, specific enough to recognise in a picture.

queries — 3 to {q} different search queries for free stock sites (photos and videos) and museum or archive search, each 2 to 4 English words. Write queries for what really exists in such libraries for this setting: things photographed or filmed today (people, staged scenes, re-enactments, museum objects, places, nature, close-ups) and, where the setting is historical, old artworks (paintings, engravings, manuscript miniatures). When the setting is historical and you know an old artwork that shows this very event, moment or thing — a chronicle or manuscript illustration, a drawing from a period treatise, a known painting — add one query that names it the way an archive titles it (the event or the work, the chronicle, manuscript or artist; up to 7 words, a year is allowed) with "type": "illustration". Name only works you know exist; if you know none, add none. Every word of a query must mean only what you want: a word with another common meaning that a search engine would match (fall — autumn, bank — money, crane — bird) needs a word that fixes its meaning. "for" lists the ids of what the query can find ("core" or claim ids). "type" says what kind of picture the query finds: "object" (one thing on its own, a museum object), "scene" (people, a place, an event), "illustration" (a painting, engraving or manuscript), "map" or "texture". Most queries look for the core; try different ways to find it (another kind of picture, another wording), not the same words with an extra word.

Example from another film, «The ball bounced off the wall and rolled away» — the core is the ball, not the wall:
{{"n": 3, "meaning": "the ball bounced off the wall and rolled away", "reading": "literal", "focus": "a ball bouncing off a wall", "core": "a ball is visible", "traps": ["a ball lying still on a shelf", "a cartoon ball"], "claims": [{{"id": "c1", "text": "the ball bounces off a wall", "tier": "must", "motion": true}}, {{"id": "c2", "text": "a wall", "tier": "should"}}], "queries": [{{"q": "ball bouncing wall", "for": ["core", "c1", "c2"], "type": "scene"}}, {{"q": "ball rolling", "for": ["core"], "type": "scene"}}, {{"q": "ball close up", "for": ["core"], "type": "object"}}]}}

Answer with one JSON object per narration line, one per line, and nothing else — no explanations, no reasoning, no markdown.

{lines}"""

RETRY_NOTE = "\n\n(Answer again: one JSON object per numbered line, every line, nothing else.)"

# Цифры разрешены: запрос-знание называет работу архивным названием, и год
# или век в нём — часть названия («battle of poitiers 1356 miniature»).
_QUERY_RE = re.compile(r"^[a-z0-9][a-z0-9'\- ]*[a-z0-9]$")


def _clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def clean_query(q):
    """Запрос в той форме, что принимает сток, или None. Модель иногда
    ставит кавычки, нумерацию, точку; всё, что не 1..7 латинских слов, —
    не запрос. Семь, а не пять: запрос, называющий известное изображение
    так, как его называет архив (событие, хроника, трактат, автор), длиннее
    стокового — прототип эп.94 находил нужные кадры именно такими."""
    q = _clean(q).strip(" \"'«».,;:-*").lower()
    q = re.sub(r"^\d+[.)]\s*", "", q)
    if not q or not _QUERY_RE.match(q):
        return None
    if not 1 <= len(q.split()) <= QUERY_MAX_WORDS:
        return None
    return q


# РЕЖИССЁРСКОЕ ЗАДАНИЕ РОЛИКА (версия 4, 26.09). Один вопрос на эпизод,
# по ВСЕМУ сценарию: модель решает, что это за фильм и как он должен
# выглядеть, до того как писать кадр на каждую фразу. Причина — эп.95:
# без общего задания каждая глава решала заново, и абстрактная фраза
# («дофамин — вещество ожидания») получала буквальную картинку (шарики на
# палочках), а «мозг» — светящийся абстрактный шар. Нужна не заплатка
# словарём под тему, а понимание фильма целиком: ниша, насколько он
# кинематографичен, как в НЁМ показывать абстракции и термины и что в
# нём чужое. Всё это модель выводит из текста сценария сама — в коде нет
# ни слова о теме канала.
DIRECTION_VERSION = 1
DIRECTION_NAME = "direction.json"
# 4000 не хватило: рассуждение съело весь запас, ответ пустой (эп.95, 26.09).
DIRECTION_MAX_TOKENS = 16000
DIRECTION_PROMPT = """You are the director of a documentary-style video. Read the whole narration and write the visual direction that the shot planner will follow for every line.
Episode: «{title}». World of the film: {setting}.

Narration, sections in order:
{script}

Reply with ONE JSON object and nothing else:
{{"topic": "...", "genre": "...", "viewer": "...", "look": "...", "literal": "...", "motifs": ["..."], "terms": [{{"term": "...", "show": "...", "avoid": "..."}}], "never": ["..."]}}

topic — one sentence: what the film is about.
genre — the kind of film and its niche (history documentary, popular psychology, science explainer, true crime, and so on).
viewer — who watches and what they should feel.
look — how the film should look: how cinematic (real people and places, light, mood, faces and hands) versus illustrative (diagrams, museum objects, archival art), and which kinds of pictures carry this film.
literal — how to picture figurative and abstract lines in THIS film: when to show the literal thing, when a human situation the viewer recognises, when a visual metaphor; which kinds of lines here are metaphors or rhetoric.
motifs — 0 to 5 recurring visual threads for continuity (a recurring person, place or object), only if the narration has them.
terms — every technical term, abstract concept or named thing the narration keeps returning to: how to SHOW it on screen in this film, and which lazy pictures to AVOID (a generic stock cliche, a toy or cartoon model, a wrong era).
never — 3 to 8 kinds of pictures that must never appear in this film because they belong to a far-away world (another era, culture, genre, fiction) or break its tone. Only clearly foreign things: never ban the film's own subjects.
Short English phrases."""


def _direction_script(video_dir):
    """Весь озвучиваемый текст по секциям — то, что модель читает как сценарий."""
    import lumean_tts
    try:
        secs = lumean_tts.extract_section_texts(os.path.join(video_dir, "script.txt"))
    except OSError:
        return ""
    tag = re.compile(r"\[[^\]]*\]")
    return "\n".join("[" + name + "] " + tag.sub(" ", text) for name, text in secs)


def _str_list(xs, lo=1, hi=12, limit=8):
    """Список коротких английских фраз (ловушки фразы)."""
    out = []
    for x in xs if isinstance(xs, list) else []:
        t = clean_text(x, lo=lo, hi=hi) if isinstance(x, str) else None
        if t and t not in out:
            out.append(t)
    return out[:limit]


def _txt(x, limit):
    """Строка задания: пробелы свёрнуты, длина ограничена (обрез по концу
    предложения, если он есть в пределах лимита). Не строка — пусто."""
    if not isinstance(x, str):
        return ""
    t = _clean(x)
    if len(t) <= limit:
        return t
    cut = t[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0]).strip()


def parse_direction(raw):
    """Режиссёрское задание из ответа модели, или None: без темы и облика
    фильма задание не годится (лучше без него, чем с обрывком). Поля —
    свободный текст модели с ограничением длины: задание читает модель же,
    и проверять его словарём значит выбрасывать хорошие ответы (первая
    версия разбора отвергла развёрнутый ответ из-за длины и русских
    терминов)."""
    for obj in _all_json(raw):
        topic, look = _txt(obj.get("topic"), 300), _txt(obj.get("look"), 900)
        if not topic or not look:
            continue
        terms = []
        for t in obj.get("terms") or []:
            if isinstance(t, dict) and _txt(t.get("term"), 60) and _txt(t.get("show"), 300):
                terms.append({"term": _txt(t.get("term"), 60), "show": _txt(t.get("show"), 300),
                              "avoid": _txt(t.get("avoid"), 250)})
        listed = lambda xs, n, lim: [y for y in (_txt(x, lim) for x in (xs if isinstance(xs, list) else [])) if y][:n]
        return {"topic": topic, "look": look, "genre": _txt(obj.get("genre"), 250),
                "viewer": _txt(obj.get("viewer"), 400), "literal": _txt(obj.get("literal"), 1200),
                "motifs": listed(obj.get("motifs"), 5, 250), "terms": terms[:12],
                "never": listed(obj.get("never"), 8, 150)}
    return None


def _all_json(raw):
    """Все JSON-объекты ответа по порядку (без требования поля n)."""
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
        if isinstance(obj, dict):
            out.append(obj)
            i = end
        else:
            i += 1


def direction_block(direction):
    """Режиссёрское задание строками для вопроса по главе. Нет задания —
    пустая строка, вопрос как раньше."""
    if not direction:
        return ""
    lines = ["Film direction (follow it for every line):",
             f"- about: {direction['topic']}"]
    if direction.get("genre"):
        lines.append(f"- genre: {direction['genre']}")
    if direction.get("viewer"):
        lines.append(f"- viewer: {direction['viewer']}")
    lines.append(f"- look: {direction['look']}")
    if direction.get("literal"):
        lines.append(f"- figurative and abstract lines: {direction['literal']}")
    if direction.get("motifs"):
        lines.append("- recurring motifs: " + "; ".join(direction["motifs"]))
    for t in direction.get("terms") or []:
        lines.append(f"- «{t['term']}»: show {t['show']}" + (f"; avoid {t['avoid']}" if t.get("avoid") else ""))
    if direction.get("never"):
        lines.append("- never show: " + "; ".join(direction["never"]))
    return "\n".join(lines) + "\n\n"


def direction_digest(direction):
    return hashlib.sha256(json.dumps(direction or {}, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()[:12]


def make_direction(video_dir, gateway, model=DEFAULT_MODEL, setting=None, title=""):
    """Режиссёрское задание эпизода: из файла, если сценарий (озвучиваемый
    текст), вопрос и модель те же, иначе — один вопрос модели. Сбой — None
    и прежний путь (задание по главам без общего задания)."""
    script = _direction_script(video_dir)
    if not script.strip():
        return None
    prompt = DIRECTION_PROMPT.format(title=title or "—", setting=setting or "not specified", script=script)
    sig = hashlib.sha256(f"{DIRECTION_VERSION}|{model}|{prompt}".encode("utf-8")).hexdigest()[:16]
    path = os.path.join(video_dir, "media_plan", DIRECTION_NAME)
    old = _read_plan(path)
    if old.get("sig") == sig and old.get("direction"):
        return old["direction"]
    import llm_gateway
    try:
        raw, _u, _p = gateway.chat(model, [{"type": "text", "text": prompt}], DIRECTION_MAX_TOKENS,
                                   max(EST_PROMPT_TOKENS, len(prompt) // 3))
    except llm_gateway.PaymentRequired:
        raise
    except llm_gateway.GatewayError as e:
        print(f"  режиссёрское задание не получено ({str(e)[:160]}) — главы без него")
        return old.get("direction") if old.get("direction") else None
    direction = parse_direction(raw)
    if not direction:
        print("  режиссёрское задание: ответ не разобран — главы без него")
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": DIRECTION_VERSION, "model": model, "sig": sig, "direction": direction},
                  f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return direction


READINGS = ("literal", "figurative", "abstract")


def render_spec_prompt(packet, setting):
    lines = []
    for u in packet["units"]:
        brief = u.get("author_brief")
        lines.append(f"{u['n']}. «{u['text']}»" + (f" — shot: {brief}" if brief else ""))
    prev = f" (the previous chapter ended with: «{packet['prev_tail']}»)" if packet.get("prev_tail") else ""
    return SPEC_PROMPT.format(title=packet.get("episode_title") or "—", setting=setting or "not specified",
                              direction=direction_block(packet.get("direction")),
                              prev=prev, c1=MAX_CLAIMS - 1, q=MAX_QUERIES, lines="\n".join(lines))


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


def _parse_queries(raw_queries, claim_ids):
    """Запросы с целями; запрос без годной цели не нужен — неизвестно, что
    он ищет."""
    out, seen = [], set()
    for x in (raw_queries or []):
        if not isinstance(x, dict):
            continue
        q = clean_query(x.get("q")) if isinstance(x.get("q"), str) else None
        raw_for = x.get("for")
        raw_for = [raw_for] if isinstance(raw_for, str) else (raw_for or [])
        targets = [t for t in (_clean(str(t)).lower() for t in raw_for) if t in claim_ids]
        if not q or q in seen or not targets:
            continue
        seen.add(q)
        item = {"q": q, "for": list(dict.fromkeys(targets))}
        kind = _clean(str(x.get("type") or "")).lower()
        if kind in SHOT_KINDS:
            item["type"] = kind
        out.append(item)
    return out[:MAX_QUERIES]


def order_queries(spec):
    """Запросы по важности того, что они ищут: сначала те, что ищут первое
    утверждение (фокус), дальше по самому важному утверждению цели. Порядок
    — из спецификации, а не из кода."""
    rank = {c["id"]: i for i, c in enumerate(spec["claims"])}
    return sorted(spec["queries"], key=lambda x: min(rank[t] for t in x["for"]))


def focus_query_count(spec):
    first = spec["claims"][0]["id"]
    return sum(1 for x in spec["queries"] if first in x["for"])


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


def parse_spec(raw, packet):
    """{номер юнита: спецификация}. Каждая строка разбирается отдельно:
    сорванная строка теряет одну фразу, а не главу. Фраза без фокуса, без
    годных утверждений или без ЕДИНОГО запроса, ищущего фокус, выпадает —
    юнит идёт прежним путём, а не планом, который фокус не ищет."""
    known = {u["n"] for u in packet["units"]}
    out = {}
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
            continue
        spec = {"focus": focus, "claims": claims,
                "queries": _parse_queries(obj.get("queries"), {c["id"] for c in claims})}
        if not focus_query_count(spec):
            continue
        spec["queries"] = order_queries(spec)
        # Поля версии 4 необязательны: сорванное поле не отнимает у фразы
        # её кадр, а только подсказку.
        meaning = clean_text(obj.get("meaning"), lo=2, hi=24)
        if meaning:
            spec["meaning"] = meaning
        reading = _clean(str(obj.get("reading") or "")).lower()
        if reading in READINGS:
            spec["reading"] = reading
        traps = _str_list(obj.get("traps"), lo=2, hi=12, limit=3)
        if traps:
            spec["traps"] = traps
        out[n] = spec
    return out


def flat_queries(spec):
    """Строки запросов по порядку — поле queries плана."""
    return [x["q"] for x in spec["queries"]]


def _cache_path(cache_dir, model, prompt):
    key = hashlib.sha256(f"{PLAN_VERSION}|{model}|{prompt}".encode("utf-8")).hexdigest()[:24]
    return os.path.join(cache_dir, key + ".txt")


def ask(gateway, model, prompt, cache_dir):
    """Ответ модели на главу; кэш по содержимому вопроса — повторный прогон
    не платит. Пустой ответ в кэш не пишется."""
    cp = _cache_path(cache_dir, model, prompt)
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            return f.read(), True
    text, _u, _p = gateway.chat(model, [{"type": "text", "text": prompt}], MAX_TOKENS, EST_PROMPT_TOKENS)
    if text.strip():
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cp + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, cp)
    return text, False


def ask_chapter(gateway, model, packet, setting, cache_dir):
    """Спецификации фраз одной главы: ({номер: спецификация}, из кэша ли).
    Модель иногда сбивается с формата (рассуждение вместо JSON, обрыв) —
    тогда глава спрашивается ещё раз, отдельным ключом кэша, и из второго
    ответа берутся только недостающие фразы. Сбой второго вопроса не
    отменяет первый ответ."""
    import llm_gateway
    prompt = render_spec_prompt(packet, setting)
    raw, hit = ask(gateway, model, prompt, cache_dir)
    got = parse_spec(raw, packet)
    if len(got) < len(packet["units"]):
        try:
            raw2, _hit2 = ask(gateway, model, prompt + RETRY_NOTE, cache_dir)
            for n, spec in parse_spec(raw2, packet).items():
                got.setdefault(n, spec)
        except llm_gateway.PaymentRequired:
            raise
        except llm_gateway.GatewayError:
            pass
    return got, hit


def plan_signature(model, setting, direction=None):
    """Что делает спецификации сопоставимыми между прогонами: версия,
    модель, текст инструкции, мир эпизода и режиссёрское задание. Совпадает
    — спецификацию фразы с неизменным текстом можно взять из прежнего плана."""
    return hashlib.sha256(f"{PLAN_VERSION}|{model}|{SPEC_PROMPT}|{setting}|{direction_digest(direction)}"
                          .encode("utf-8")).hexdigest()[:16]


def load_direction(video_dir):
    """Режиссёрское задание с диска или None (без сети)."""
    return _read_plan(os.path.join(video_dir, "media_plan", DIRECTION_NAME)).get("direction")


def _read_plan(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 — нет или битый: плана нет
        return {}


def plan_episode(video_dir, blocks, gateway, model=DEFAULT_MODEL, verbose=True, workers=None):
    """Спросить модель по главам и записать план. Возвращает число фраз с
    запросами. Главы спрашиваются параллельно (вопросы независимы). Сбой
    одной главы не рвёт прогон: глава пропускается с причиной; нехватка
    денег (PaymentRequired) поднимается — решает вызывающий.

    Правка одной фразы меняет вопрос всей главы, и модель отвечает заново
    на каждую её фразу. Спецификация фразы с НЕИЗМЕННЫМ текстом при этом
    берётся из прежнего плана, если подпись плана (модель, инструкция, мир)
    та же: иначе правка одного слова перепокупала бы кадры всей главы
    (спецификация входит в ключ кэша кандидата)."""
    import concurrent.futures
    import llm_gateway
    import shot_brief_director as sbd
    import shot_planner_llm
    import world_card
    setting = world_card.judge_setting(world_card.load(video_dir, strict=False))
    cache_dir = os.path.join(video_dir, "media_plan", CACHE_DIR_NAME)
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    packets = list(sbd.packets(video_dir, blocks))
    direction = make_direction(video_dir, gateway, model=model, setting=setting,
                               title=(packets[0].get("episode_title") if packets else ""))
    for packet in packets:
        packet["direction"] = direction
    sig = plan_signature(model, setting, direction)
    old = _read_plan(path)
    old_units = (old.get("units") or {}) if old.get("sig") == sig else {}

    def one(packet):
        try:
            return ask_chapter(gateway, model, packet, setting, cache_dir), None
        except llm_gateway.PaymentRequired:
            raise
        except llm_gateway.GatewayError as e:
            return None, e

    units, kept = {}, 0
    with concurrent.futures.ThreadPoolExecutor(max(1, min(workers or len(packets), 8))) as ex:
        results = list(ex.map(one, packets))
    for no, (packet, (res, err)) in enumerate(zip(packets, results), 1):
        if res is None:
            # Разовый сбой шлюза не стирает спецификации главы: прежние
            # (той же подписи) остаются, иначе ключи кэша кандидатов главы
            # сменились бы и слоты перевыбирались бы из-за сбоя сети.
            for u in packet["units"]:
                key = shot_planner_llm.unit_key(u["text"])
                if key in old_units:
                    units[key] = old_units[key]
                    kept += 1
            print(f"  глава {no}: модель не ответила — {err}")
            continue
        got, hit = res
        for u in packet["units"]:
            key = shot_planner_llm.unit_key(u["text"])
            if not hit and key in old_units:
                units[key] = old_units[key]
                kept += 1
                continue
            spec = got.get(u["n"])
            if spec:
                units[key] = dict(spec, text=u["text"], queries_for=spec["queries"],
                                  queries=flat_queries(spec))
        if verbose:
            print(f"  глава {no} «{_clean(packet['section'])[:40]}»: запросы на {len(got)} "
                  f"из {len(packet['units'])} фраз{' (кэш)' if hit else ''}")
    if kept and verbose:
        print(f"  спецификаций сохранено из прежнего плана (текст фразы не менялся): {kept}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": PLAN_VERSION, "model": model, "setting": setting, "sig": sig,
                   "units": units}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return len(units)


def needs_planning(video_dir, blocks, model=DEFAULT_MODEL):
    """Есть ли в сценарии фразы, которых нет в плане текущей версии и
    подписи (или плана нет вовсе). Без сети."""
    import shot_planner_llm
    import world_card
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    plan = _read_plan(path)
    if plan.get("version") != PLAN_VERSION:
        return True
    setting = world_card.judge_setting(world_card.load(video_dir, strict=False))
    if plan.get("sig") != plan_signature(model, setting, load_direction(video_dir)):
        return True
    have = plan.get("units") or {}
    return any(shot_planner_llm.unit_key(b.get("text") or "") not in have
               for b in blocks if (b.get("text") or "").strip())


def load(video_dir):
    """{ключ юнита: [запросы]} из плана на диске, или {}. Битый файл —
    пустой план с названной причиной, а не падение рендера."""
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        units = data.get("units") or {}
        return {k: [q for q in (v.get("queries") or []) if clean_query(q)]
                for k, v in units.items() if isinstance(v, dict)}
    except Exception as e:  # noqa: BLE001
        print(f"  {PLAN_NAME} не читается ({type(e).__name__}) — запросы фраз не используются")
        return {}


def load_specs(video_dir):
    """{ключ юнита: {"focus", "claims", "queries"}} из плана версии 3, или
    {}. План старой версии спецификаций не даёт: его ступени — ровно то,
    от чего версия 3 уходит, и молча смешивать их с новыми нельзя."""
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001 — причину уже назвал load()
        return {}
    if data.get("version") != PLAN_VERSION:
        print(f"  {PLAN_NAME}: версия {data.get('version')}, нужна {PLAN_VERSION} — "
              f"спецификации кадров не используются; перепланировать: "
              f"python scripts/stock_query_planner.py <эпизод>")
        return {}
    out = {}
    for k, v in (data.get("units") or {}).items():
        if isinstance(v, dict) and v.get("claims") and v.get("focus"):
            out[k] = {"focus": v["focus"], "claims": v["claims"],
                      "queries": v.get("queries_for") or []}
            for extra in ("meaning", "reading", "traps"):
                if v.get(extra):
                    out[k][extra] = v[extra]
    return out


def has_motion(spec, must=False):
    """Есть ли у фразы утверждение движения (must=True — обязательное)."""
    return any(c.get("motion") and (not must or c.get("tier") == "must")
               for c in (spec or {}).get("claims") or [])


def attach(blocks, plan, specs=None):
    """Проставить блокам b["phrase_queries"] по тексту фразы, а по плану
    версии 3 ещё b["shot_spec"]. Возвращает, скольким блокам нашлись
    запросы. Вызывается по ФИНАЛЬНЫМ блокам (после нарезки и слияния в
    pipeline_smart.main): у каждого подкадра свой текст и своё задание.
    Раньше привязка шла до нарезки, и подкадры наследовали задание всей
    фразы — два подкадра подряд искали одно и то же."""
    if not plan:
        return 0
    import shot_planner_llm
    n = 0
    for b in blocks:
        key = shot_planner_llm.unit_key(b.get("text") or "")
        qs = plan.get(key)
        if qs:
            b["phrase_queries"] = list(qs)
            n += 1
        spec = (specs or {}).get(key)
        if spec:
            b["shot_spec"] = spec
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video_dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cap", type=int, default=30000, help="потолок расходов шлюза на прогон")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import llm_gateway
    import script_parser
    blocks = script_parser.parse_blocks(os.path.join(args.video_dir, "script.txt"))
    gw = llm_gateway.Gateway(spend_cap=args.cap)
    n = plan_episode(args.video_dir, blocks, gw, model=args.model)
    print(f"Готово: запросы на {n} фраз из {len(blocks)}. {gw.summary()}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
