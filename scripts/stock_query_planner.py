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
PLAN_VERSION = 3
TIERS = ("must", "should")
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
# Модель выбрана замером 24.09 (58 фраз трёх ниш — кинжал, психология,
# глубоководье — без авторских описаний кадра, одна инструкция на всех):
# DeepSeek v4 Flash разобрал 58/58 во всех прогонах, главное конкретное и
# снимаемое, ~2 тыс. токенов баланса на 58 фраз; Gemini 3.7 Flash по смыслу
# наравне, но ~35 тыс.; Qwen 3.8 Max (стоял здесь раньше) — 46/58 и «виден
# человек» на половине абстрактных фраз. Качество оценено глазами Claude,
# не разметкой владельца. Сменить — --model.
DEFAULT_MODEL = "ds/deepseek-v4-flash"
MAX_TOKENS = 8000
EST_PROMPT_TOKENS = 2500

SPEC_PROMPT = """You direct the visuals of a documentary video.
Episode: «{title}». Setting: {setting}.
Below are the narration lines of one chapter, in order{prev}. A line may come with the shot the author wants — keep its meaning.

For EVERY numbered line decide what the viewer must SEE while hearing it.

focus — the new thing this line says, understood in the context of the chapter (resolve pronouns and references from the lines around it). 3 to 12 English words.

core — WHO or WHAT must be visible: the single thing (an object, a person, an animal, a place) that, even alone in a picture, still makes the viewer think of this line — with the state that defines it, if any ("an exhausted person", "a burnt letter"). Name the thing, not an event: what it does goes into the claims. Ask yourself: if the picture could show only one thing, which one? When the line is about something happening to, on or around something else, the core is what the line is about — usually the thing that moves, acts or changes — not the surface, place or object it happens on. When the line is abstract (a feeling, an idea, a process, an argument), the core is a concrete situation, a bodily sign or an object left behind that a camera can photograph and a viewer reads as this idea — never a bare "a person is visible" or an invisible thing like "a memory" or "a brain decision": say what makes the picture show THIS line ("a person slumped over an untouched plate", "a crumpled paper covered in red corrections"). Never make words, captions, labels, signs or logos in the picture part of the core or of a claim — the viewer hears the words, the picture shows things — unless the line is about that very document, chart, headline, sign or screen. Write it as a statement: "a ball is visible".

claims — 1 to {c1} more statements checkable by looking at the picture, most important first. Each checks ONE thing (an object, an action, a place, a detail) and does not repeat the core. "tier": "must" if without it the picture does not show this line, "should" if it only makes the picture better. If the line is about a movement that only footage can show, one claim has "motion": true and describes this movement; lines about objects, places or states have no motion claim.

queries — 3 to {q} different search queries, each 2 to 4 English words, for free stock sites (photos and videos) and museum or archive search. Write queries for what really exists in such libraries for this setting: things photographed or filmed today (people, staged scenes, re-enactments, museum objects, places, nature, close-ups) and, where the setting is historical, old artworks (paintings, engravings, manuscript miniatures). "for" lists the ids of what the query can find ("core" or claim ids). Most queries look for the core; try different ways to find it (another kind of picture, another wording), not the same words with an extra word.

Example from another film, «The ball bounced off the wall and rolled away» — the core is the ball, not the wall:
{{"n": 3, "focus": "a ball bouncing off a wall", "core": "a ball is visible", "claims": [{{"id": "c1", "text": "the ball bounces off a wall", "tier": "must", "motion": true}}, {{"id": "c2", "text": "a wall", "tier": "should"}}], "queries": [{{"q": "ball bouncing wall", "for": ["core", "c1", "c2"]}}, {{"q": "ball rolling", "for": ["core"]}}, {{"q": "ball close up", "for": ["core"]}}]}}

Answer with one JSON object per narration line, one per line, and nothing else — no explanations, no reasoning, no markdown.

{lines}"""

RETRY_NOTE = "\n\n(Answer again: one JSON object per numbered line, every line, nothing else.)"

_QUERY_RE = re.compile(r"^[a-z][a-z'\- ]*[a-z]$")


def _clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def clean_query(q):
    """Запрос в той форме, что принимает сток, или None. Модель иногда
    ставит кавычки, нумерацию, точку; всё, что не 1..5 латинских слов, —
    не запрос."""
    q = _clean(q).strip(" \"'«».,;:-*").lower()
    q = re.sub(r"^\d+[.)]\s*", "", q)
    if not q or not _QUERY_RE.match(q):
        return None
    if not 1 <= len(q.split()) <= 5:
        return None
    return q


def render_spec_prompt(packet, setting):
    lines = []
    for u in packet["units"]:
        brief = u.get("author_brief")
        lines.append(f"{u['n']}. «{u['text']}»" + (f" — shot: {brief}" if brief else ""))
    prev = f" (the previous chapter ended with: «{packet['prev_tail']}»)" if packet.get("prev_tail") else ""
    return SPEC_PROMPT.format(title=packet.get("episode_title") or "—", setting=setting or "not specified",
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
        out.append({"q": q, "for": list(dict.fromkeys(targets))})
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


def plan_signature(model, setting):
    """Что делает спецификации сопоставимыми между прогонами: версия,
    модель, текст инструкции и мир эпизода. Совпадает — спецификацию фразы с
    неизменным текстом можно взять из прежнего плана."""
    return hashlib.sha256(f"{PLAN_VERSION}|{model}|{SPEC_PROMPT}|{setting}".encode("utf-8")).hexdigest()[:16]


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
    sig = plan_signature(model, setting)
    old = _read_plan(path)
    old_units = (old.get("units") or {}) if old.get("sig") == sig else {}
    packets = list(sbd.packets(video_dir, blocks))

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
    if plan.get("sig") != plan_signature(model, setting):
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
    return out


def has_motion(spec, must=False):
    """Есть ли у фразы утверждение движения (must=True — обязательное)."""
    return any(c.get("motion") and (not must or c.get("tier") == "must")
               for c in (spec or {}).get("claims") or [])


def attach(blocks, plan, specs=None):
    """Проставить блокам b["phrase_queries"] по тексту фразы, а по плану
    версии 3 ещё b["shot_spec"]. Возвращает, скольким блокам нашлись
    запросы. Под-кадры наследуют поля при нарезке (dict(b) в
    split_long_blocks), поэтому проставляется ДО неё."""
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
