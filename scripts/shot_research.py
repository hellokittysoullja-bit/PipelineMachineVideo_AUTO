#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Второй круг поиска кадра: ничего не прошло проверку (или лучший кадр —
замена без главного фразы) — мозг видит, что нашлось и почему отклонено, и
пишет НОВЫЕ запросы.

ЗАЧЕМ. Замер глубины пула эпизода 94 (docs/quality/RESEARCHER_PROTO_EP94.md):
на фразах про действие нужного кадра не было ни в первых 20 кандидатах, ни
на местах 21-200 — его не приносили запросы. Копать глубже бесполезно,
судье нечего было показать. А прототип «исследователя вручную» нашёл кадр
по смыслу на всех девяти фразах — другими запросами, в других местах
(хроники, фехтовальные трактаты, надгробия). Живой исследователь так и
работает: ищет, смотрит, что вышло, и переформулирует. Раньше запросы
писались один раз, вслепую, и провал на них был окончательным.

ЧТО ВИДИТ МОЗГ: фразу, что должен увидеть зритель, обязательные пункты,
опробованные запросы и — главное — причины, по которым проверка отклонила
каждого финалиста («на кадре римские легионеры, а не рыцарь; никто не
падает»). Это ровно то знание, которого не было у первого круга.

ЧТО ДЕЛАЕТ РЕНДЕР (pipeline_smart, после второй страницы каскада): если у
фразы в платной зоне нет годного кадра, куча второго круга собирается ТОЛЬКО
из новых запросов (иначе сортировка снова поставила бы перед судьёй тот же
мусор) и проходит того же судью. Не нашлось — прежний путь (поглощение
соседним кадром).

ДЕТЕРМИНИЗМ. Новые запросы сохраняются в media_plan/research_queries.json
по ключу фразы; подпись — версия вопроса, модель и то, ЧТО искали (текст
фразы, фокус, опробованные запросы), но не причины отказов: те зависят от
ответов судьи, и их дрожь не должна заставлять платить за новый вопрос при
каждом перезапуске. Повторный рендер берёт запросы с диска."""
import hashlib
import json
import os
import re

PROMPT_VERSION = 2
# Почему нужен второй круг: всё найденное отклонено, или лучший кадр —
# ближайшая замена без главного фразы. Модели это разные задачи: во втором
# случае найденное годится, не хватает именно главного.
VERDICTS = {
    "failed": "A reviewer looked at the best pictures they found and rejected every one:",
    "weak": ("A reviewer looked at the best pictures they found: none shows the main thing "
             "of the line, the best one is only a substitute:"),
}
MAX_NEW_QUERIES = 4
MAX_REJECTIONS = 8
FILE_NAME = "research_queries.json"
# Мозг второго круга — та же модель, что у планировщика (DeepSeek v4 Flash),
# С ВЫКЛЮЧЕННЫМ рассуждением. По умолчанию модель рассуждает, и рассуждение
# не ограничено ничем, кроме запаса выхода: judge14 (24.09) — 400 токенов,
# пустой ответ; judge15 (25.09) — 4000 токенов, снова пустой ответ, всё ушло
# в рассуждение. Замер на шести реальных слотах эп.94 (фразы и отказы
# judge14): без рассуждения — ответ на всех шести, 3-52 с и 47-240 токенов
# баланса на вызов; с рассуждением и запасом 16 000 — тоже на всех шести, но
# 25-131 с, до 445 токенов и до 7 957 токенов рассуждения (вплотную к
# обрыву). Запросы по смыслу равноценны: оба варианта называют трактаты
# (Codex Wallerstein, Flos Duellatorum / Fiore dei Liberi), хроники
# (Froissart, St Albans) и реконструкцию (глазами Claude, не разметкой).
# Ответ — четыре коротких запроса: запаса 2000 с лихвой.
MAX_TOKENS = 2000
EST_PROMPT_TOKENS = 900
REASONING = False

PROMPT = """You find pictures for a documentary video. Setting: {setting}.
Narration line: «{phrase}»
What the viewer must see: {focus}
Required in the picture: {musts}

These searches were tried: {tried}
{verdict}
{rejections}

Write {n} NEW search queries that could find a picture that does show this line. Think where such a picture really exists: stock photo and video sites (people, staged scenes, re-enactments, places, close-ups), museum collections (objects), and — when the setting is historical — public archives of old artworks (chronicle and manuscript illustrations, drawings from period treatises, paintings, engravings). You may name a specific artwork, manuscript or series the way an archive titles it — only one you know exists. If no picture can show this exact moment, search for the closest picture a viewer still reads as this line: the moment just before or after it, a close-up detail, an object that stands for it. Do not repeat a tried query. Every word must mean only what you want: a word with another common meaning (fall — autumn, bank — money) needs a word that fixes it. Each query 2 to 7 English words.

Answer with JSON only: {{"queries": [{{"q": "...", "type": "object|scene|illustration|map|texture"}}]}}"""


def unit_key(text):
    import shot_planner_llm
    return shot_planner_llm.unit_key(text or "")


def tried_queries(spec, request_queries=()):
    """Запросы первого круга: спецификации фразы и запроса слота, по
    порядку, без повторов."""
    out = []
    for item in (spec or {}).get("queries") or []:
        q = item.get("q") if isinstance(item, dict) else item
        if q and q not in out:
            out.append(q)
    for q in request_queries:
        if q and q not in out:
            out.append(q)
    return out


def rejections_from_log(log, index, limit=MAX_REJECTIONS):
    """Причины отказа финалистов слота — из журнала проверки судьи, по
    порядку, без повторов."""
    out = []
    for e in log or []:
        if e.get("index") != index:
            continue
        why = " ".join(str(((e.get("verify") or {}).get("why")) or "").split())
        if why and why not in out:
            out.append(why[:300])
        if len(out) >= limit:
            break
    return out


def signature(model, setting, phrase, spec, tried, trigger="failed"):
    focus = (spec or {}).get("focus") or ""
    payload = json.dumps([PROMPT_VERSION, PROMPT, VERDICTS.get(trigger, ""), model, setting or "",
                          phrase or "", focus, list(tried)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render_prompt(phrase, spec, setting, tried, rejections, trigger="failed"):
    musts = [c["text"] for c in (spec or {}).get("claims") or [] if c.get("tier") == "must"]
    return PROMPT.format(
        setting=setting or "not specified", phrase=phrase or "—",
        focus=(spec or {}).get("focus") or "—", musts="; ".join(musts) or "—",
        tried="; ".join(tried) or "—", verdict=VERDICTS.get(trigger, VERDICTS["failed"]),
        rejections="\n".join(f"- {r}" for r in rejections) or "- (no picture passed the check)",
        n=MAX_NEW_QUERIES)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse(raw, tried=()):
    """Новые запросы из ответа модели: [{"q", "type"}], уже очищенные тем
    же правилом, что у планировщика, без опробованных и повторов."""
    import stock_query_planner as sqp
    m = _JSON_RE.search(raw or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    tried_norm = {sqp.clean_query(t) or t for t in tried}
    out, seen = [], set()
    for x in (data.get("queries") or []) if isinstance(data, dict) else []:
        if not isinstance(x, dict) or not isinstance(x.get("q"), str):
            continue
        q = sqp.clean_query(x["q"])
        if not q or q in seen or q in tried_norm:
            continue
        seen.add(q)
        kind = str(x.get("type") or "").strip().lower()
        item = {"q": q}
        if kind in sqp.SHOT_KINDS:
            item["type"] = kind
        out.append(item)
        if len(out) >= MAX_NEW_QUERIES:
            break
    return out


def _path(video_dir):
    return os.path.join(video_dir, "media_plan", FILE_NAME)


def load(video_dir):
    try:
        with open(_path(video_dir), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(video_dir, data):
    path = _path(video_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def new_queries(video_dir, gateway, model, *, phrase, spec, setting, tried, rejections,
                trigger="failed"):
    """(запросы, откуда): с диска, если эта фраза с тем же поиском уже
    исследовалась, иначе — вопрос модели и запись на диск. Пустой ответ на
    диск не пишется: следующий прогон спросит снова."""
    key = unit_key(phrase)
    sig = signature(model, setting, phrase, spec, tried, trigger)
    data = load(video_dir)
    entry = data.get(key)
    if isinstance(entry, dict) and entry.get("sig") == sig and entry.get("queries"):
        return entry["queries"], "disk"
    prompt = render_prompt(phrase, spec, setting, tried, rejections, trigger)
    raw, _usage, _price = gateway.chat(model, [{"type": "text", "text": prompt}], MAX_TOKENS,
                                       EST_PROMPT_TOKENS, reasoning=REASONING)
    items = parse(raw, tried)
    if items:
        data = load(video_dir)
        data[key] = {"sig": sig, "model": model, "phrase": phrase, "queries": items,
                     "rejections": list(rejections), "trigger": trigger}
        save(video_dir, data)
    return items, "model"


def as_spec_queries(items):
    """Новые запросы в форме запросов спецификации: каждый ищет главное."""
    out = []
    for it in items:
        q = {"q": it["q"], "for": ["core"]}
        if it.get("type"):
            q["type"] = it["type"]
        out.append(q)
    return out
