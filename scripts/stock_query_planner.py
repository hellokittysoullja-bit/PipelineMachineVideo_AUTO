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

ВЕРСИЯ 2 — ЛЕСТНИЦА ЗАМЕН (план 24.09). Точного кадра фразы в бесплатных
источниках часто нет вовсе: замер эп.94 — точный кадр есть в пуле у 5
фото-слотов из 9, у видео ни у одного; живые запросы к видеостоку по
«рыцарь падает», «стрела бьёт в доспех» приносят римлян, наполеонику и
спортивную стрельбу. Поэтому на фразу модель пишет не четыре запроса одного
кадра, а ЛЕСТНИЦУ: ступень 1 — самый точный реальный кадр, ступени 2-3 —
замены, несущие ту же мысль (предмет без действия, смежная сцена, картина
или миниатюра события), у каждой свои запросы; плюс предпочтение вида
(фото/видео). Поле queries плана — запросы ступеней по порядку, поэтому
прежний путь чтения плана работает как раньше; ступени и предпочтение —
отдельные поля (load_specs).
"""
import argparse
import hashlib
import json
import os
import re
import sys

PLAN_NAME = "stock_queries.json"
CACHE_DIR_NAME = "stock_query_cache"
PLAN_VERSION = 2
MAX_RUNGS = 3
QUERIES_PER_RUNG = 3
MAX_PLAN_QUERIES = 8
PREFER = ("photo", "video", "either")
DEFAULT_MODEL = "qwen/qwen3.8-max"
MAX_TOKENS = 2500
EST_PROMPT_TOKENS = 2500

SPEC_PROMPT = """You plan shots for a documentary video.
Episode: «{title}». Setting: {setting}.
Below are the narration lines of one chapter, in order{prev}. A line may come with the shot the author wants.

Free stock sites (Pexels, Pixabay) and museum or archive search rarely have the exact shot a line describes. For EVERY numbered line plan a ladder of shots that really exist in such libraries for this setting: things photographed or filmed today (people, staged scenes, re-enactments and tournaments, museum objects, places, nature, close-ups of objects) and, where the setting is historical, old artworks (paintings, engravings, manuscript miniatures).
- rung 1: the most exact real shot of the line; keep its action if real footage of it can exist;
- rungs 2 and 3: substitutes that still carry the same idea when rung 1 is not found — the object without the action, a related scene, an artwork of the event;
- each shot: 4 to 12 English words, something a camera can see; never text, captions or logos;
- each rung: {q} search queries of 2 to 4 English words, no punctuation, the first the most exact;
- prefer: "video" if the line is about motion that footage shows better, "photo" if it is about an object or a still state, otherwise "either".

Answer with one JSON object per narration line, one per line, and nothing else:
{{"n": 1, "prefer": "photo", "rungs": [{{"shot": "...", "queries": ["...", "..."]}}, {{"shot": "...", "queries": ["..."]}}]}}

{lines}"""

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
                              prev=prev, q=QUERIES_PER_RUNG, lines="\n".join(lines))


def clean_shot(shot):
    """Описание кадра ступени или None: 3..16 слов латиницей, без кавычек."""
    shot = _clean(shot).strip(" \"'«».;:")
    words = shot.split()
    if not 3 <= len(words) <= 16 or not re.search(r"[a-zA-Z]", shot):
        return None
    if re.search(r"[а-яА-ЯёЁ]", shot):
        return None
    return shot


def parse_spec(raw, packet):
    """{номер юнита: {"rungs": [{"shot", "queries"}], "prefer"}}. Каждая
    строка разбирается отдельно: сорванная строка теряет одну фразу, а не
    главу. Ступень без годного описания или без годного запроса выпадает;
    фраза без ступеней — тоже."""
    known = {u["n"] for u in packet["units"]}
    out = {}
    for line in (raw or "").splitlines():
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        n = obj.get("n")
        if not isinstance(n, int) or n not in known or n in out:
            continue
        rungs = []
        for r in (obj.get("rungs") or [])[:MAX_RUNGS]:
            if not isinstance(r, dict):
                continue
            shot = clean_shot(r.get("shot"))
            qs = []
            for q in (r.get("queries") or []):
                q = clean_query(q) if isinstance(q, str) else None
                if q and q not in qs:
                    qs.append(q)
            if shot and qs:
                rungs.append({"shot": shot, "queries": qs[:QUERIES_PER_RUNG]})
        if not rungs:
            continue
        prefer = obj.get("prefer") if obj.get("prefer") in PREFER else "either"
        out[n] = {"rungs": rungs, "prefer": prefer}
    return out


def flat_queries(rungs):
    """Запросы ступеней по порядку, без повторов — поле queries плана."""
    out = []
    for r in rungs:
        for q in r["queries"]:
            if q not in out:
                out.append(q)
    return out[:MAX_PLAN_QUERIES]


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


def plan_episode(video_dir, blocks, gateway, model=DEFAULT_MODEL, verbose=True):
    """Спросить модель по главам и записать план. Возвращает число фраз с
    запросами. Сбой одной главы не рвёт прогон: глава пропускается с причиной."""
    import llm_gateway
    import shot_brief_director as sbd
    import shot_planner_llm
    import world_card
    setting = world_card.judge_setting(world_card.load(video_dir, strict=False))
    cache_dir = os.path.join(video_dir, "media_plan", CACHE_DIR_NAME)
    units = {}
    for no, packet in enumerate(sbd.packets(video_dir, blocks), 1):
        prompt = render_spec_prompt(packet, setting)
        try:
            raw, hit = ask(gateway, model, prompt, cache_dir)
        except llm_gateway.PaymentRequired:
            raise
        except llm_gateway.GatewayError as e:
            print(f"  глава {no}: модель не ответила — {e}")
            continue
        got = parse_spec(raw, packet)
        for u in packet["units"]:
            spec = got.get(u["n"])
            if spec:
                units[shot_planner_llm.unit_key(u["text"])] = {
                    "text": u["text"], "queries": flat_queries(spec["rungs"]),
                    "rungs": spec["rungs"], "prefer": spec["prefer"]}
        if verbose:
            print(f"  глава {no} «{_clean(packet['section'])[:40]}»: запросы на {len(got)} "
                  f"из {len(packet['units'])} фраз{' (кэш)' if hit else ''}")
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": PLAN_VERSION, "model": model, "setting": setting, "units": units},
                  f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return len(units)


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
    """{ключ юнита: {"rungs", "prefer"}} из плана версии 2, или {}."""
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            units = (json.load(f).get("units") or {})
    except Exception:  # noqa: BLE001 — причину уже назвал load()
        return {}
    return {k: {"rungs": v["rungs"], "prefer": v.get("prefer", "either")}
            for k, v in units.items() if isinstance(v, dict) and v.get("rungs")}


def attach(blocks, plan, specs=None):
    """Проставить блокам b["phrase_queries"] по тексту фразы, а по плану
    версии 2 ещё b["shot_rungs"] (описания ступеней замены, без первой) и
    b["kind_pref"]. Возвращает, скольким блокам нашлись запросы. Под-кадры
    наследуют поля при нарезке (dict(b) в split_long_blocks), поэтому
    проставляется ДО неё."""
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
            b["shot_rungs"] = [r["shot"] for r in spec["rungs"][1:]]
            b["kind_pref"] = spec.get("prefer", "either")
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
