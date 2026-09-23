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
"""
import argparse
import hashlib
import json
import os
import re
import sys

PLAN_NAME = "stock_queries.json"
CACHE_DIR_NAME = "stock_query_cache"
PLAN_VERSION = 1
QUERIES_PER_PHRASE = 4
DEFAULT_MODEL = "qwen/qwen3.8-max"
MAX_TOKENS = 2500
EST_PROMPT_TOKENS = 2500

PROMPT = """You pick stock footage for a documentary video.
Episode: «{title}». Setting: {setting}.
Below are the narration lines of one chapter, in order{prev}. Each line comes with the shot it needs, if known.

For EVERY numbered line write {k} search queries for stock photo and video sites (Pexels, Pixabay) and museum or archive search:
- 2 to 4 English words each, no quotes, no punctuation inside a query;
- the first query is the most exact; each next one is more general but still shows the same idea;
- search for what really exists in such libraries for this setting: things photographed or filmed today (people, staged scenes, re-enactments, museum objects, places, nature, close-ups of objects), not what only a feature film or a painting could show;
- keep the action of the line where real footage of it can exist; never ask for text, captions or logos.

Answer with one line per narration line and nothing else:
n | query 1 ; query 2 ; query 3 ; query 4

{lines}"""

_ROW_RE = re.compile(r"^\s*\**\s*(\d{1,3})\s*[|.)]\s*(.*)$")
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


def render_prompt(packet, setting):
    lines = []
    for u in packet["units"]:
        brief = u.get("author_brief")
        lines.append(f"{u['n']}. «{u['text']}»" + (f" — shot: {brief}" if brief else ""))
    prev = f" (the previous chapter ended with: «{packet['prev_tail']}»)" if packet.get("prev_tail") else ""
    return PROMPT.format(title=packet.get("episode_title") or "—", setting=setting or "not specified",
                         prev=prev, k=QUERIES_PER_PHRASE, lines="\n".join(lines))


def parse_answer(raw, packet):
    """{номер юнита: [запросы]}; строки, не прошедшие проверку, пропускаются
    по одной — сорванная строка теряет одну фразу, а не главу."""
    known = {u["n"] for u in packet["units"]}
    out = {}
    for line in (raw or "").splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if n not in known or n in out:
            continue
        qs = []
        for part in m.group(2).split(";"):
            q = clean_query(part)
            if q and q not in qs:
                qs.append(q)
        if qs:
            out[n] = qs[:QUERIES_PER_PHRASE]
    return out


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
        prompt = render_prompt(packet, setting)
        try:
            raw, hit = ask(gateway, model, prompt, cache_dir)
        except llm_gateway.PaymentRequired:
            raise
        except llm_gateway.GatewayError as e:
            print(f"  глава {no}: модель не ответила — {e}")
            continue
        got = parse_answer(raw, packet)
        for u in packet["units"]:
            qs = got.get(u["n"])
            if qs:
                units[shot_planner_llm.unit_key(u["text"])] = {"text": u["text"], "queries": qs}
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


def attach(blocks, plan):
    """Проставить блокам b["phrase_queries"] по тексту фразы. Возвращает,
    скольким блокам нашлись запросы. Под-кадры наследуют поле при нарезке
    (dict(b) в split_long_blocks), поэтому проставляется ДО неё."""
    if not plan:
        return 0
    import shot_planner_llm
    n = 0
    for b in blocks:
        qs = plan.get(shot_planner_llm.unit_key(b.get("text") or ""))
        if qs:
            b["phrase_queries"] = list(qs)
            n += 1
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
