#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Отсев кандидатов чужого мира по подписи, до судьи кадров.

ЗАЧЕМ. Судья кадров смотрит на картинку и регулярно принимает кадр чужого
мира за свой: марокканскую «фантазию» (конный праздник), турецкую
реконструкцию в Конье, гладиаторов, терракотовых воинов — на фразу про
средневековую конницу. При этом подпись такого кадра прямо называет, откуда
он («moroccan fantasia horsemen in rabat»). Две дешёвые текстовые проверки
DeepSeek по подписи ловят именно этот брак до судьи.

КАК. Два РАЗНЫХ вопроса с учётом фразы и паспорта мира, одновременно:
  X — подпись уверенно говорит, что кадр из культуры или эпохи, которую
      паспорт исключает (или это современное шоу/фестиваль такой культуры);
  B — подпись уверенно говорит, что кадр вне мира фильма или совсем о другом.
Кадр выбрасывается, только если СОГЛАСНЫ ОБА. Каждый вопрос по отдельности
теряет годные кадры; согласие — нет.

ЗАМЕР (26.09, 72 размеченные кучи, два прогона X):
  эп.94 (средневековье, в паспорте чужие культуры): выкинуто 106 бракованных
  из 2053, среди них 31 кадр, который судья ошибочно одобрял; годных
  потеряно 0 из 638, лучший кадр кучи не потерян ни в одной из 49 куч.
  эп.93 (глубоководье, чужих культур в паспорте нет): вопрос X не задаётся,
  отсев выключен сам — потерь 0.
Отдельный вопрос «явно не про эту фразу» (J) проверен тем же замером и НЕ
используется: терял 53-57 годных на эп.94 и 29-38 на эп.93 (удильщик,
осьминог, мокрица на фразе про гигантизм — сравнение, которое по подписи
выглядит посторонним). Вопрос X задаётся в той формулировке, в которой
замерен (вместе с правилом J); отметки J игнорируются.

ГРАНИЦЫ. Работает только там, где работает судья (платная зона, ключ
шлюза), только в историческом эпизоде с чужими культурами в паспорте и только на первых
TOP_N кандидатах каскада. Любой сбой — кандидаты остаются как есть.
Решения кэшируются на диске по тексту вопроса: повторный рендер не платит и
получает тот же ответ."""
import concurrent.futures
import hashlib
import json
import os
import re

import world_card

SCREEN_VERSION = 2
MODEL = "ds/deepseek-v4-flash"
TOP_N = 100
MAX_TOKENS = 2500
EST_PROMPT_TOKENS = 7000

URL_JUNK = set("""www com org edu net https http photo photos video videos pexels pixabay metmuseum art collection search
artic artworks commons wikimedia wiki file index php curid clevelandart openverse images image jpg jpeg png tif
upload org uk free stock hd""".split())

PROMPT_X = """You pre-screen stock search results for ONE shot of a documentary film, by their TEXT only (caption, title, tags or URL words). The pictures are not shown; a vision judge looks at the survivors later.
Narration line (Russian): «{phrase}»
The shot should show: {focus}
{world}
Mark a result ONLY when its text makes you SURE:
{xrule}J = it has nothing to do with this line or with the film's subject: none of the line's objects, creatures, people, places, materials, actions or ideas, and nothing a documentary editor could use as a comparison, detail or metaphor for this line (for example food, a pet, a car, an office, jewelry or a keychain in a film about something else).
Never mark results that are vague, only generic tags, only a URL, or could plausibly fit. Related creatures, objects, materials, comparisons, museum pieces and historical art are NOT J. When in doubt, do not mark.
Reply with JSON only: {{"mark": {{"<number>": "<code>", ...}}}}  (empty object if nothing)
Results:
{rows}"""

XRULE = ("X = it clearly comes from a culture or era the film's world excludes ({exclude}), or from a modern "
         "show, festival, parade, reenactment or tourist event presenting such a culture.\n")

PROMPT_B = """You screen image search results for one shot of a documentary film, by their TEXT only (caption, title, tags or URL words). The images themselves are not shown.
Narration line (Russian): «{phrase}»
The shot must show: {focus}
{world}
Below are numbered results: "number | source | text". Mark a result DROP only when its text CLEARLY shows one of these:
- it is outside the film's world: {outside};
- it is about a completely different topic that has nothing to do with the line's subject area.
KEEP every result from the film's world that shows the subject, a part of it, a related object, a person or a scene of that world — even if the action, pose or composition is different from the line. Keep results whose text is vague, only tags, or not enough to judge. When in doubt, keep.
Reply with JSON only: {{"drop": [numbers]}}
Results:
{rows}"""


def prompts_digest():
    """Тексты вопросов — в подпись отбора: их правка меняет, кто дойдёт до судьи."""
    return hashlib.sha256((PROMPT_X + XRULE + PROMPT_B).encode("utf-8")).hexdigest()[:12]


def active_for(card):
    """Отсев работает только там, где он замерен: исторический эпизод
    (world_card.is_historical), у паспорта которого есть чужие культуры.
    Без чужих культур вопрос X не задаётся и выбрасывать нечего. «Ноль
    потерь годных» снят только на исторических кучах; в научном эп.95 отсев
    выбросил песочные часы с тегом «ancient» на фразе про слепоту ко времени
    (26.09) — вне замеренной области он не включается."""
    return world_card.is_historical(card) and bool(world_card.culture_exclude(card))


def clean(text, channel):
    """Подпись кандидата в том виде, в котором её видела модель на замере:
    свой текст плюс значимые слова адреса, если текста мало."""
    t = (text or "").strip()
    head, _, url = t.partition("https:")
    head = re.sub(r"\s+", " ", head).strip(" .,")
    host = ""
    m = re.search(r"www\.([a-z0-9-]+)\.|([a-z0-9-]+)\.(?:org|com|edu)", url)
    if m:
        host = (m.group(1) or m.group(2) or "")
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ]+", url) if w.lower() not in URL_JUNK and len(w) > 2]
    if len(head) < 25 and words:
        head = (head + " " + " ".join(words)).strip()
    if channel == "openverse" and host and host not in ("pexels", "pixabay"):
        head += f" [{host}]"
    return head[:150] or "(no text)"


def _world_x(card):
    cul = (card or {}).get("culture") or {}
    parts = []
    reg = (card or {}).get("register")
    era = (card or {}).get("era")
    if reg:
        tail = f", {era.get('from')}-{era.get('to')} AD" if isinstance(era, dict) and era else ""
        parts.append(f"The film's world: {reg}{tail}.")
    if cul.get("include"):
        parts.append("Cultures of the film: " + ", ".join(cul["include"]) + ".")
    exc = list(world_card.culture_exclude(card))
    if exc:
        parts.append("Excluded cultures: " + ", ".join(exc) + ".")
    return "\n".join(parts), exc


def _world_b(card):
    setting = world_card.judge_setting(card) or ""
    forbidden = list(world_card.forbidden_classes(card))
    historical = world_card.is_historical(card)
    lines = []
    if setting:
        lines.append(f"The film's world: {setting}.")
    if forbidden:
        lines.append("Never in the film: " + ", ".join(forbidden) + ".")
    if historical:
        outside = ("a modern item, modern people or clothes, a modern setting or event, fantasy, cosplay, a toy, "
                   "a costume party, another culture or another era than the film's world")
    else:
        outside = "fantasy, fiction, a cartoon, a toy, cosplay" + (", " + ", ".join(forbidden) if forbidden else "")
    return "\n".join(lines), outside


def build_prompts(phrase, focus, card, rows):
    """rows: [(id, channel, text)] -> (вопрос X, вопрос B)."""
    world_x, exc = _world_x(card)
    listing_x = "\n".join(f"{n} | {ch} | {clean(tx, ch)}" for n, (_i, ch, tx) in enumerate(rows, 1))
    px = PROMPT_X.format(phrase=phrase, focus=focus, world=world_x,
                         xrule=XRULE.format(exclude=", ".join(exc)) if exc else "", rows=listing_x)
    world_b, outside = _world_b(card)
    pb = PROMPT_B.format(phrase=phrase, focus=focus, world=world_b, outside=outside, rows=listing_x)
    return px, pb


def _json_obj(ans):
    m = re.search(r"\{.*\}", ans or "", re.S)
    if not m:
        raise ValueError("в ответе нет JSON")
    return json.loads(m.group(0), strict=False)


def parse_x(ans, n):
    """Номера результатов с отметкой X (J игнорируется, см. докстринг модуля)."""
    marks = _json_obj(ans).get("mark") or {}
    if not isinstance(marks, dict):
        return set()
    return {int(k) for k, v in marks.items()
            if str(k).strip().isdigit() and 1 <= int(k) <= n and str(v)[:1].upper() == "X"}


def parse_b(ans, n):
    drop = _json_obj(ans).get("drop") or []
    if not isinstance(drop, list):
        return set()
    return {int(k) for k in drop if str(k).strip().isdigit() and 1 <= int(k) <= n}


def _cache_path(cache_dir, text):
    key = hashlib.sha256((MODEL + "\n" + text).encode("utf-8")).hexdigest()[:24]
    return os.path.join(cache_dir, key + ".json")


def _ask(gw, text, cache_dir):
    """(ответ, цена, из кэша ли). Кэш — по полному тексту вопроса и модели."""
    path = _cache_path(cache_dir, text) if cache_dir else None
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)["answer"], 0, True
        except (OSError, ValueError, KeyError):
            pass
    ans, _usage, price = gw.chat(MODEL, [{"type": "text", "text": text}], MAX_TOKENS, EST_PROMPT_TOKENS,
                                 reasoning=False, timeout=240)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"answer": ans}, f, ensure_ascii=False)
        os.replace(tmp, path)
    return ans, price, False


def screen(gw, phrase, focus, card, rows, cache_dir=None):
    """Какие id выбросить: {id} и сводка. Сбой — пустое множество (fail-open)."""
    info = {"asked": len(rows), "dropped": [], "price": 0, "cached": 0, "error": None}
    if not rows or not active_for(card):
        return set(), info
    px, pb = build_prompts(phrase, focus, card, rows)
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            fx, fb = ex.submit(_ask, gw, px, cache_dir), ex.submit(_ask, gw, pb, cache_dir)
            (ax, cx, hx), (ab, cb, hb) = fx.result(), fb.result()
        both = parse_x(ax, len(rows)) & parse_b(ab, len(rows))
    except Exception as e:  # noqa: BLE001 — отсев необязателен: сбой оставляет кандидатов как есть
        info["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return set(), info
    info["price"] = cx + cb
    info["cached"] = int(hx) + int(hb)
    ids = {rows[n - 1][0] for n in both}
    info["dropped"] = [{"id": rows[n - 1][0], "channel": rows[n - 1][1],
                        "text": clean(rows[n - 1][2], rows[n - 1][1])} for n in sorted(both)]
    return ids, info
