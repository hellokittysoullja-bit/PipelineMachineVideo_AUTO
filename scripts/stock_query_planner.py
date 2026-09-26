#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Оркестратор кадров: понимание фильма и задание на КАЖДЫЙ кадр. Пишет
текстовая модель (DeepSeek) один раз на эпизод; рендер читает план с диска.

ЗАЧЕМ ОРКЕСТРАТОР, А НЕ «ЗАПРОСЫ К СТОКАМ». Девять кадров из десяти (всё
после платной зоны судьи) выбираются из первых ответов первых запросов
этого плана, без модели «зрение+язык» (docs/quality/PREJUDGE_LINK_2509.md).
То есть то, как ЭТОТ модуль понял фразу, и есть кадр почти всего ролика, а
дешевле его в контуре нет ничего: DeepSeek v4 Flash — коэффициент 0.05,
весь эпизод стоит меньше одного слота судьи. Поэтому понимание собрано
здесь, а остальные читают его, а не толкуют фразу заново.

ДВА УРОВНЯ.
1. Библия фильма (один вопрос по ВСЕМУ сценарию, media_plan/film_bible.json):
   о чём фильм и для кого; насколько он кинематографичен (0-3) и какой
   материал его несёт; кто такой «ты» в кадрах этого фильма; сквозные
   предметы и люди, на которые сценарий потом ссылается местоимениями;
   что сценарий прячет и раскрывает позже («главный убийца» = земля); как
   здесь показывать метафоры и абстракции; ключевые понятия — как показать
   и чего избегать; далёкие миры, которых в фильме не бывает никогда.
   Без неё глава видела только себя, название и хвост предыдущей главы:
   что сценарий прячет и назовёт лишь потом, кто такой «ты», какой
   материал несёт фильм и насколько он кинематографичен — знать ей было
   неоткуда.
2. Задание на кадр (главами, окнами по WINDOW кадров): смысл с
   разрешёнными ссылками; тип чтения (буквально / образ / абстракция);
   ОБРАЗ метафоры, которого показывать нельзя; что видно на картинке;
   главное; проверяемые утверждения; ловушки; запросы.

ЧТО ЗАКРЫТО УСТРОЙСТВОМ, А НЕ ТЕСТОМ (всё — живые промахи прежних версий):
  * «что фраза ГОВОРИТ» отдавалось судье как «что ДОЛЖНО БЫТЬ ВИДНО». В
    v3 поле focus определялось как «новое, что говорит фраза», а читали его
    сетка судьи («Required shot»), проверка утверждений, рамка детали и
    отсев по подписи. На фразе «доспех держит человека, как капкан» focus
    был «armour holds the fallen man down like a trap» — капкан ехал судье
    требованием. Теперь смысл (meaning) и кадр (shot) — разные поля, и
    focus = shot;
  * образ метафоры не может попасть ни в кадр, ни в главное, ни в
    обязательные утверждения, ни в запросы: модель сама называет его
    английскими словами (vehicle), код ищет их там. Раньше это правило было
    невозможно — фраза русская, кадр английский (см. shot_planner_llm.
    brief_is_safe: «сопоставить их без перевода нельзя»); теперь перевод
    делает тот, кто понимает фразу;
  * под-кадры одной фразы и слитые блоки получают СВОИ задания: план
    строится по финальным блокам рендера (после split/merge), части одной
    фразы помечены как части. Раньше под-кадры наследовали задание всей
    фразы, и два кадра подряд искали одно и то же;
  * обрыв ответа не теряет фразы молча: окно — не больше WINDOW кадров,
    причина конца ответа читается (finish_reason), недостающие и
    отклонённые кадры переспрашиваются только они, с причиной отказа, до
    REASK_ROUNDS раз; что не получилось — записано в план поимённо и
    напечатано. Раньше глава из 13 фраз обрывалась на шестой (8000 токенов
    с рассуждением), а переспрос повторял ту же длину и обрывался там же;
  * «[shot:] автора — закон» снят: предложение автора — подсказка; если
    модель от неё отказалась, в плане записано почему.

БЕЗ НИШИ В КОДЕ. В шаблонах вопросов нет слов темы канала (тест держит):
мир приходит из паспорта эпизода, ниша — из сценария и CHANNEL.md
справкой, а что показывать — решает модель.

НЕТ ПЛАНА — ПРЕЖНИЙ ПУТЬ. Фраза без задания отбирается как раньше, байт в
байт; план старой версии заданий не даёт (молча смешивать нельзя).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading

PLAN_NAME = "stock_queries.json"
CACHE_DIR_NAME = "stock_query_cache"
BIBLE_NAME = "film_bible.json"
READABLE_NAME = "shot_plan.txt"
# 4 — откатанная версия 26.09 (коммит 3f52f7c); номер не переиспользуется,
# чтобы план той версии на диске владельца не читался как этот.
PLAN_VERSION = 5
BIBLE_VERSION = 1
TIERS = ("must", "should")
READINGS = ("literal", "figurative", "abstract")
# Тип кадра запроса — тот же словарь, что у маршрутизации источников
# (shot_types.SHOT_TYPES): «any» модель не пишет, его значит отсутствие поля.
SHOT_KINDS = ("object", "scene", "illustration", "map", "texture")
# Главное — отдельное поле ответа, а не «первое в списке»: замер 24.09
# (эп.94) — при правиле «первое утверждение — главное» Gemini Flash и Qwen
# Max на фразе «Стрела скользит по нагруднику» ставили первым «виден
# нагрудник».
CORE_ID = "core"
# Размеры ответа — цена и внимание модели, а не смысл: утверждений больше
# пяти человек у кадра не проверяет, запросов больше шести — это уже расход
# квоты стоков на одну фразу.
MAX_CLAIMS = 5
MAX_QUERIES = 6
QUERY_MAX_WORDS = 7
MAX_TRAPS = 3
# Обязательных утверждений сверх главного — не больше двух: судья сравнивает
# кадры по ним лексикографически, и каждое лишнее «must» (руки, грязь, фон)
# отдаёт выбор кадру, у которого есть обстановка, но нет предмета фразы.
# Замер 26.09 (судья на снимке эп.94): «клинок в грязной латной перчатке» и
# «брошенный меч в грязи» как must — музейный кинжал с меткой «точно»
# проиграл музейному же кинжалу с меткой «брак».
MAX_MUST_EXTRA = 2
MAX_VEHICLE = 4
# Кадров в одном вопросе. Выход на кадр — ~250-350 токенов JSON, то есть
# окно ограничивает ответ ~2 тыс. токенов при ЛЮБОЙ длине главы: обрыв по
# лимиту становится исключением, а не свойством длинных глав.
WINDOW = 6
REASK_ROUNDS = 2
DEFAULT_MODEL = "ds/deepseek-v4-flash"
# Рассуждение модели — явно, а не по умолчанию провайдера (у DeepSeek оно
# включено, и в v3 его никто не выбирал). Запас выхода — под выбранный режим.
REASONING = True
MAX_TOKENS = {True: 16000, False: 6000}
BIBLE_MAX_TOKENS = {True: 16000, False: 6000}
BIBLE_ATTEMPTS = 2
EST_PROMPT_TOKENS = 4000
CALL_TIMEOUT = 420

# Кадр, собранный монтажом, найти нельзя: такого снимка нет ни в одной
# библиотеке, а запрос за ним («thin circle graphic overlay») приносит мусор.
# Замер 26.09: библия фильма про прокрастинацию предложила «анимированное
# кольцо поверх съёмки», и кадр «Круг замкнулся» стал этим кольцом.
_COMPOSITE_WORDS = ("overlay", "superimposed", "split screen", "split-screen", "picture-in-picture")

_PRONOUN_START = frozenset(("he", "she", "it", "they", "we", "you", "i", "this", "that",
                            "these", "those", "his", "her", "its", "their", "our", "your", "my"))


# --- общее ---------------------------------------------------------------------

def _clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


# Цифры разрешены: запрос-знание называет работу архивным названием, и год
# или век в нём — часть названия («battle of poitiers 1356 miniature»).
_QUERY_RE = re.compile(r"^[a-z0-9][a-z0-9'\- ]*[a-z0-9]$")


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


def clean_text(text, lo=2, hi=16):
    """Английское описание или None: lo..hi слов латиницей, без кириллицы."""
    text = _clean(text).strip(" \"'«».;:")
    words = text.split()
    if not lo <= len(words) <= hi or not re.search(r"[a-zA-Z]", text):
        return None
    if re.search(r"[а-яА-ЯёЁ]", text):
        return None
    return text


def _txt(x, limit):
    """Строка библии: пробелы свёрнуты, длина ограничена по концу
    предложения, если он есть в пределах лимита. Не строка — пусто."""
    if not isinstance(x, str):
        return ""
    t = _clean(x)
    if len(t) <= limit:
        return t
    cut = t[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0]).strip()


def _all_json(raw):
    """Все JSON-объекты ответа по порядку. Сорванный объект теряет только
    себя — разбор продолжается со следующей скобки."""
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


def json_objects(raw):
    """JSON-объекты ответа с полем n (строки заданий)."""
    return [o for o in _all_json(raw) if "n" in o]


def _word_re(word):
    """Слово или словосочетание целиком, с окончанием множественного числа."""
    w = re.escape(_clean(word).lower())
    return re.compile(r"\b" + w + r"(?:s|es)?\b", re.I)


def mentions(text, words):
    """Первое из words, встреченное в text целым словом, или None."""
    for w in words or ():
        if w and _word_re(w).search(text or ""):
            return w
    return None


def _starts_with_pronoun(text):
    first = (_clean(text).lower().split() or [""])[0].strip(",.:;\"'")
    return first in _PRONOUN_START


def _str_list(xs, lo=1, hi=12, limit=8):
    out = []
    for x in xs if isinstance(xs, list) else []:
        t = clean_text(x, lo=lo, hi=hi) if isinstance(x, str) else None
        if t and t.lower() not in (o.lower() for o in out):
            out.append(t)
    return out[:limit]


_ARTICLES = frozenset(("a", "an", "the", "this", "that", "his", "her", "its", "their", "our", "your", "my"))


def head_noun(phrase):
    """Главное слово английской именной группы: последнее слово, а у группы
    с «of» — последнее слово до «of» («a pile of coins» -> "pile»; «the
    rondel dagger» -> "dagger"). Пусто — пустая строка."""
    words = [w for w in re.findall(r"[a-z][a-z'\-]*", _clean(phrase).lower())]
    if "of" in words:
        words = words[:words.index("of")]
    words = [w for w in words if w not in _ARTICLES]
    return words[-1] if words else ""


def noun_forms(noun):
    """Слово и его единственное число (arrows -> arrow, boxes -> box): чтобы
    «the arrows» в поле about совпало с «an arrow» в главном."""
    forms = [noun] if noun else []
    if noun.endswith("es") and len(noun) > 4:
        forms.append(noun[:-2])
    if noun.endswith("s") and not noun.endswith("ss") and len(noun) > 3:
        forms.append(noun[:-1])
    return forms


_FUNCTION_WORDS = _ARTICLES | frozenset(("of", "and", "or", "with", "in", "on", "at", "to", "from", "for",
                                          "by", "into", "onto", "over", "under", "its", "own"))
# Местоимения в поле about («you» — зритель, как его показывает библия):
# словом их в главном не найти, сверять там нечего.
_PRONOUNS = frozenset(("you", "yourself", "we", "us", "he", "him", "she", "they", "them", "it",
                       "me", "i", "one"))


def _proper_name(words, i):
    """Слово — часть имени собственного: с заглавной и не первое, либо первое,
    за которым идёт ещё одно с заглавной («Johann Gutenberg»). Одиночное
    первое слово с заглавной («Macrophages») — просто начало фразы."""
    w = words[i]
    if not w[:1].isupper():
        return False
    return i > 0 or (len(words) > 1 and words[1][:1].isupper())


def shares_a_word(text, about):
    """Есть ли в text хоть одно значимое слово из about — целиком, в
    единственном числе или внутри составного слова («longsword» содержит
    «sword»). Первая версия сверяла только главное слово целиком и на эп.02
    отклонила 49 заданий из 142: «the sword» против «a longsword blade»,
    «the reversed sword strike» против «a sword pommel hitting a helmet» —
    переспросы стоили +44% времени, две фразы остались без задания."""
    words = re.findall(r"[a-z][a-z'\-]*", _clean(text).lower())
    # Имя собственное («Johann Gutenberg») сверке не подлежит: судья не
    # узнаёт человека по лицу, и требовать имя в главном значило бы
    # толкать туда то, что проверить нельзя (замер 26.09: с правилом «о чём —
    # подлежащее» имя вставало в главное в 4 прогонах из 4).
    raw = re.findall(r"[A-Za-z][A-Za-z'\-]*", _clean(about))
    content = [w.lower() for i, w in enumerate(raw)
               if not _proper_name(raw, i) and w.lower() not in _FUNCTION_WORDS
               and w.lower() not in _PRONOUNS and len(w) >= 3]
    if not content:
        return True
    for w in content:
        for form in noun_forms(w):
            if len(form) >= 3 and any(form in x for x in words):
                return True
    return False


def _vehicle_words(raw):
    """Образ метафоры: 1..MAX_VEHICLE английских слов или сочетаний до трёх
    слов. Строка «a, b» тоже принимается — модель иногда так пишет."""
    if isinstance(raw, str):
        raw = [p for p in re.split(r"[,;/]", raw)]
    out = []
    for x in raw if isinstance(raw, list) else []:
        t = _clean(x).strip(" \"'«».;:").lower()
        t = re.sub(r"^(a|an|the)\s+", "", t)
        if t and 1 <= len(t.split()) <= 3 and re.fullmatch(r"[a-z][a-z'\- ]*", t) and t not in out:
            out.append(t)
    return out[:MAX_VEHICLE]


# --- библия фильма -------------------------------------------------------------

BIBLE_PROMPT = """You are the visual director of a narrated video: a voice reads the narration while the screen shows found pictures — stock photos and footage, museum objects, archival artworks. There is no presenter and nothing is generated. Read the WHOLE narration and write the visual bible that every shot of THIS film will follow. Shots are planned one chapter at a time, so everything the planner must know about the film as a whole goes here.

Title: «{title}»
Channel niche (a hint; the narration decides): {niche}
World passport of this episode (fixed — do not contradict it):
{world}

Narration by sections, in order:
{script}

Reply with ONE JSON object and nothing else:
{{"topic": "...", "genre": "...", "audience": "...", "cinematic": 0, "look": "...", "viewer": "...", "cast": [{{"name": "...", "refs": ["..."], "show": "..."}}], "reveals": [{{"hidden": "...", "is": "..."}}], "figurative": "...", "terms": [{{"term": "...", "show": "...", "avoid": "..."}}], "never": ["..."]}}

topic — one sentence: what this film is about.
genre — the kind of film and its niche.
audience — who watches and what they should feel.
cinematic — how cinematic the pictures must be for this niche and audience, 0 to 3: 3 = like a feature film (real people, faces, hands, action, light, mood); 2 = mostly real photos and footage with some archival art or objects; 1 = mostly archival art, documents and museum objects; 0 = mostly diagrams and simple illustrations.
look — which kinds of pictures carry this film and which kinds would break it, in one or two sentences.
viewer — the narration speaks to "you" (Russian «ты»/«тебя»/«тебе»): who "you" is in the pictures of this film — who, which era, clothes, place. If "you" is the audience today, say how to show them.
cast — people, creatures, objects and places the narration keeps coming back to, especially the ones it later calls only by a pronoun: name (English), refs (the Russian words and pronouns the narration uses for it), show (how it looks in the found pictures of this film — never an overlay, animation or effect made for it).
reveals — things the narration keeps hidden and names only later (a mystery in the hook): hidden (how it is referred to before), is (what it turns out to be). [] if none.
figurative — how to picture THIS film's figurative and abstract lines: which metaphors, comparisons and idioms appear, and the rule that a line is shown by its meaning, never by the image in its words. A comparison that describes something — as if, like, the size of, the weight of — is never shown, not even as a size comparison; only a real, familiar thing that the line itself asks the viewer to recall, hold or try is shown.
terms — key concepts and named things the narration keeps returning to: how to SHOW each in this film (show) and which lazy picture to AVOID (avoid).
never — 4 to 10 kinds of pictures from a far-away world that must never appear in this film: another era, another culture, another genre, fiction, fantasy, games, cartoons for a grown-up film, staged stock clichés. Only clearly foreign things: never list this film's own subjects, its comparisons or its evidence.
All values in English, short."""


def bible_digest(bible):
    return hashlib.sha256(json.dumps(bible or {}, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()[:12]


def parse_bible(raw):
    """Библия из ответа модели или None. Без темы и облика фильма она не
    годится: лучше без неё (прежний вопрос по главам), чем с обрывком."""
    for obj in _all_json(raw):
        topic, look = _txt(obj.get("topic"), 300), _txt(obj.get("look"), 600)
        if not topic or not look:
            continue
        cin = obj.get("cinematic")
        cin = int(cin) if isinstance(cin, (int, float)) and not isinstance(cin, bool) \
            and 0 <= cin <= 3 else None
        cast = []
        for c in obj.get("cast") or []:
            if isinstance(c, dict) and _txt(c.get("name"), 80) and _txt(c.get("show"), 360):
                refs = [_txt(r, 40) for r in (c.get("refs") or []) if _txt(r, 40)][:8]
                cast.append({"name": _txt(c["name"], 80), "refs": refs, "show": _txt(c["show"], 360)})
        reveals = [{"hidden": _txt(r.get("hidden"), 120), "is": _txt(r.get("is"), 160)}
                   for r in obj.get("reveals") or []
                   if isinstance(r, dict) and _txt(r.get("hidden"), 120) and _txt(r.get("is"), 160)]
        terms = [{"term": _txt(t.get("term"), 80), "show": _txt(t.get("show"), 360),
                  "avoid": _txt(t.get("avoid"), 240)}
                 for t in obj.get("terms") or []
                 if isinstance(t, dict) and _txt(t.get("term"), 80) and _txt(t.get("show"), 360)]
        never = [_txt(x, 200) for x in obj.get("never") or [] if _txt(x, 200)]
        return {"topic": topic, "genre": _txt(obj.get("genre"), 200),
                "audience": _txt(obj.get("audience"), 300), "cinematic": cin, "look": look,
                "viewer": _txt(obj.get("viewer"), 300), "cast": cast[:10], "reveals": reveals[:6],
                "figurative": _txt(obj.get("figurative"), 1200), "terms": terms[:12],
                "never": never[:10]}
    return None


def bible_block(bible):
    """Библия строками для вопроса по главе. Нет библии — пустая строка."""
    if not bible:
        return ""
    lines = ["FILM BIBLE — the whole film; follow it for every shot:",
             f"- film: {bible['topic']}" + (f" ({bible['genre']})" if bible.get("genre") else "")
             + (f"; for {bible['audience']}" if bible.get("audience") else "")]
    cin = bible.get("cinematic")
    lines.append("- look: " + (f"cinematic {cin}/3 — " if isinstance(cin, int) else "") + bible["look"])
    if bible.get("viewer"):
        lines.append(f"- \"you\" (ты) in the pictures: {bible['viewer']}")
    for c in bible.get("cast") or []:
        refs = f" (called: {', '.join(c['refs'])})" if c.get("refs") else ""
        lines.append(f"- recurring: {c['name']}{refs} — {c['show']}")
    for r in bible.get("reveals") or []:
        lines.append(f"- hidden, revealed later: \"{r['hidden']}\" is {r['is']}")
    if bible.get("figurative"):
        lines.append(f"- figurative and abstract lines: {bible['figurative']}")
    for t in bible.get("terms") or []:
        lines.append(f"- «{t['term']}»: show {t['show']}" + (f"; avoid {t['avoid']}" if t.get("avoid") else ""))
    # Далёкое — указание модели, а не фильтр по словам: в списке «никогда»
    # живут и свои слова в чужом контексте («magic glowing brains» в фильме
    # про мозг), и фильтр по слову выбросил бы настоящий мозг.
    if bible.get("never"):
        lines.append("- never in this film (far-away worlds; do not plan or search for them): "
                     + "; ".join(bible["never"]))
    return "\n".join(lines) + "\n\n"


def _narration(video_dir, blocks=None):
    """Озвучиваемый текст по секциям — то, что модель читает как сценарий.
    Блоки рендера (после нарезки) дают тот же текст, что и сценарий."""
    if blocks is None:
        import script_parser
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            blocks = script_parser.parse_blocks(os.path.join(video_dir, "script.txt"))
    sections, order = {}, []
    for b in blocks:
        sec = b.get("section") or "—"
        if sec not in sections:
            sections[sec] = []
            order.append(sec)
        sections[sec].append(_clean(b.get("text")))
    return "\n".join(f"[{s}] " + " ".join(t for t in sections[s] if t) for s in order)


def world_lines(card):
    """Паспорт мира для текстовой модели — ЦЕЛИКОМ, строками. Сетке судьи
    длинный список запретов по замеру вредил (shot_judge, «МИР ЭПИЗОДА»);
    планировщику он нужен весь: он пишет запросы и должен знать, чего в
    этом фильме не бывает."""
    if not card:
        return "not specified"
    import world_card
    out = [f"register: {card.get('register') or '—'}"]
    w = world_card.era_window(card)
    if w:
        def y(v):
            return f"{-v} BC" if v < 0 else f"{v} AD"
        out.append(f"era: {y(w[0])} to {y(w[1])}")
    inc, exc = world_card.culture_include(card), world_card.culture_exclude(card)
    if inc:
        out.append("cultures shown: " + ", ".join(inc))
    if exc:
        out.append("cultures that would be wrong here: " + ", ".join(exc))
    look = card.get("look") if isinstance(card.get("look"), dict) else {}
    if look.get("style"):
        out.append(f"look: {look['style']}")
    out.append("computer graphics (3D, cartoons, infographics): "
               + ("allowed" if world_card.renders_allowed(card) else "never"))
    if card.get("modern_props_allowed") is not None:
        out.append("modern everyday things: " + ("allowed where a line compares with them or asks for them"
                                                 if card.get("modern_props_allowed") else "never"))
    forb = world_card.forbidden_classes(card)
    if forb:
        out.append("never show: " + "; ".join(forb))
    return "\n".join(out)


def bible_prompt(video_dir, card, blocks=None):
    import shot_brief_director as sbd
    ctx = sbd.episode_context(video_dir)
    return BIBLE_PROMPT.format(title=ctx.get("title") or "—", niche=ctx.get("niche") or "not given",
                               world=world_lines(card), script=_narration(video_dir, blocks))


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 — нет или битый: читать нечего
        return {}


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_bible(video_dir):
    """Библия с диска или None (без сети)."""
    return _read_json(os.path.join(video_dir, "media_plan", BIBLE_NAME)).get("bible")


def make_bible(video_dir, gateway, card, model=DEFAULT_MODEL, reasoning=REASONING, blocks=None,
               verbose=True):
    """(библия | None, что сделано): «disk» — сценарий, вопрос и модель те
    же; «made» — новый ответ; «failed: ...» — ответа нет, на диске прежнее
    (прежняя библия лучше никакой: сценарий правился, фильм тот же)."""
    import llm_gateway
    prompt = bible_prompt(video_dir, card, blocks)
    sig = hashlib.sha256(f"{BIBLE_VERSION}|{model}|{reasoning}|{prompt}".encode("utf-8")).hexdigest()[:16]
    path = os.path.join(video_dir, "media_plan", BIBLE_NAME)
    old = _read_json(path)
    if old.get("sig") == sig and old.get("bible"):
        return old["bible"], "disk"
    # Сорванный ответ (битый JSON, обрыв лимитом, пустой) спрашивается
    # ещё раз: библия одна на эпизод, и без неё ВСЕ главы идут без облика
    # фильма, его «ты» и сквозных предметов. Живой случай 26.09: у эпизода
    # про крах 1929 ответ не разобрался, и эпизод спланирован без библии.
    why = ""
    for _attempt in range(BIBLE_ATTEMPTS):
        try:
            raw, usage, _p = gateway.chat(model, [{"type": "text", "text": prompt}],
                                          BIBLE_MAX_TOKENS[bool(reasoning)],
                                          max(EST_PROMPT_TOKENS, len(prompt) // 3),
                                          reasoning=reasoning, timeout=CALL_TIMEOUT)
        except llm_gateway.PaymentRequired:
            raise
        except llm_gateway.EmptyAnswer as e:
            why = f"пустой ответ: {str(e)[:160]}"
            continue
        except llm_gateway.GatewayError as e:
            return old.get("bible"), f"failed: {str(e)[:200]}"
        bible = parse_bible(raw)
        if bible:
            break
        why = "ответ оборван лимитом" if (usage or {}).get("finish_reason") == "length" else "ответ не разобран"
    else:
        return old.get("bible"), f"failed: {why}"
    _write_json(path, {"version": BIBLE_VERSION, "model": model, "reasoning": reasoning, "sig": sig,
                       "bible": bible})
    return bible, "made"


# --- задание на кадр -----------------------------------------------------------

SPEC_PROMPT = """You plan the pictures of a narrated video. A voice reads the lines; while each numbered shot is heard, the screen shows ONE found picture: a stock photo or video, a museum object or an archival artwork. There is no presenter and nothing is generated.

{bible}WORLD of this episode:
{world}

CHAPTER «{section}». Read all of it first{around}:
{lines}

Plan shots {first} to {last}{done}. For each of them write one JSON object on its own line, in order, and nothing else — no explanations, no markdown:
{{"n": <number>, "meaning": "...", "about": "...", "reading": "...", "vehicle": [...], "shot": "...", "core": "...", "claims": [...], "traps": [...], "queries": [...]{brief_key}}}

meaning — what these words say in THIS film, in plain English, with every pronoun, every "you" and every not-yet-named thing resolved from the chapter and the film bible («он» → "the old lighthouse", «ты» → "you, a night-shift nurse"). 4 to 20 words.

about — who or what the line says something about, its pronouns resolved: usually the subject of its sentence. In «Но именно он спас экспедицию, когда карта уже не помогала» (it was he who saved the expedition when the map no longer helped), where «он» is the old compass named two lines before, the about is "the old compass" — not "the expedition", not "the map". An event or an action is never the about: the one who acts is. A real person is written as a picture shows them — «a 1920s woman chemist», not «Marie Curie». 1 to 6 English words, when it is a thing, creature, person or place that a found picture can show; "" when the line is about an idea, a feeling, a number, a process or a rule that no picture shows by itself.

reading — "literal" if what the words describe can be shown as it is; "figurative" if they speak through an image that is not what happens (a metaphor, a comparison, an idiom, a personification): the picture shows the meaning and never that image; "abstract" if they state an idea, a feeling, a number, a rule or an argument: the picture shows a concrete situation, a bodily sign or a trace that a viewer reads as it. A line that itself asks the viewer to recall, hold or try a real, familiar thing («вспомни, сколько весит пакет молока» — remember how heavy a milk carton is) is "literal": that thing is shown. A comparison that only describes — «будто», «как», «словно», «размером с», «весом с» (as if, like, the size of) — is "figurative": show the real thing it describes, not the thing it is compared with.

vehicle — English nouns (0 to 4) naming the images in the words that are NOT what happens and must NOT be shown: the image of a metaphor, a comparison or an idiom («цены растут, как на дрожжах» → "yeast"; «держит, как тиски» → "vise"; «камень размером с дом» → "house"), in a line of any reading; [] if the words have none. Only a real thing the line itself asks the viewer to recall, hold or try is not a vehicle.

shot — the picture the viewer sees: 6 to 18 English words naming concrete things a camera or a painter could show, in this film's world and look. It shows the meaning: a figurative line never shows its vehicle, an abstract line shows its concrete stand-in. Start with the thing itself, never with a pronoun. No words, captions or signs in the picture unless the line is about that very document. The picture must already exist somewhere to be found: never an overlay, an animation, a split screen or an effect added in editing. Neighbouring shots show different pictures — another angle, moment, setting or kind of picture — unless a line continues the very same moment, but never swap what the line is about for another thing just to be different; the parts of one sentence show its parts.

core — WHO or WHAT the line is about, once its pronouns and references are resolved (for a line about an idea — its concrete stand-in); when "about" is not empty, the core is that very thing, even if the shot before showed it too. Name the thing, not an event or a scene, in general words that EVERY correct picture of it satisfies: its kind, and its era or world when the film has one ("an 18th-century sailing ship", not "a three-masted frigate"). A subtype, a feature, a material and anything it does are claims ("the ship has three masts", "the ship heels in the wind"). At most one state that defines it ("an exhausted person", "a burnt letter", "a capsized boat") and nothing about what holds it, where it lies or what surrounds it — those go into the claims too. A person is named by what a camera sees ("a grey-haired woman in a 1920s laboratory coat"): nobody can tell who a person is by looking, so a name belongs in the core only when the line is about the known portrait of that person. A plain photo or museum object of that thing must satisfy the core: the judge rejects every picture where the core is not true. Written as a statement "... is visible".

claims — 1 to {c1} more statements checkable by looking at the picture, most important first. Each checks ONE thing (an object, an action, a place, a detail) and does not repeat the core. "tier": "must" only when the line is about exactly that — its action (the ball bounces off the wall, the door slams) or the one detail that carries the meaning — usually one, never more than two; hands, ground, light, setting and the other things in the shot are "should": a picture that shows the core and misses them is still right. If the shot needs a movement that only footage can show, one claim has "motion": true.

traps — 1 to 3 pictures that a search for this shot would likely return and that look related but are WRONG for it: the vehicle, a toy, cartoon, model or diagram version of a real thing, the wrong moment, a staged stock cliché, another era or culture. 2 to 10 English words each.

queries — 3 to {q} different searches for free stock photo and video sites and museum or archive search, 2 to 4 English words each. The FIRST query is often used alone: make it the one most likely to return the right picture as its top result. Write queries for what really exists in such libraries for this world: things photographed or filmed today (people, staged scenes, re-enactments, museum objects, places, nature, close-ups) and, where the world is historical, old artworks (paintings, engravings, manuscript miniatures). When the world is historical and you know an old artwork that shows this very event, moment or thing — a chronicle or manuscript illustration, a drawing from a period treatise, a known painting — add one query that names it the way an archive titles it (up to 7 words, a year is allowed) with "type": "illustration". Name only works you know exist. Every word of a query must mean only what you want: a word with another common meaning that a search engine would match (fall — autumn, bank — money, crane — bird) needs a word that fixes its meaning. Never search for the vehicle. "for" lists the ids of what the query can find ("core" or claim ids); "type" is "object" (one thing on its own, a museum object), "scene" (people, a place, an event), "illustration" (a painting, engraving or manuscript), "map" or "texture". Most queries look for the core, each in a different way.
{brief_rule}
Examples from another film:
«Мяч отскочил от стены и укатился» (the ball bounced off the wall and rolled away) — the core is the ball, not the wall:
{{"n": 3, "meaning": "the ball bounced off the wall and rolled away", "about": "the ball", "reading": "literal", "vehicle": [], "shot": "a ball bouncing off a brick wall and rolling across the yard", "core": "a ball is visible", "claims": [{{"id": "c1", "text": "the ball bounces off a wall", "tier": "must", "motion": true}}, {{"id": "c2", "text": "a brick wall", "tier": "should"}}], "traps": ["a ball lying still on a shelf", "a cartoon ball"], "queries": [{{"q": "ball bouncing wall", "for": ["core", "c1", "c2"], "type": "scene"}}, {{"q": "ball rolling yard", "for": ["core"], "type": "scene"}}, {{"q": "ball close up", "for": ["core"], "type": "object"}}]}}
«Город заснул» (the city fell asleep) — a personification: the city does not sleep, its streets go quiet:
{{"n": 4, "meaning": "night came and the city went quiet", "about": "the city", "reading": "figurative", "vehicle": ["sleep", "bed"], "shot": "an empty city street at night under orange street lamps, dark windows above", "core": "an empty city street at night is visible", "claims": [{{"id": "c1", "text": "nobody is on the street", "tier": "must"}}, {{"id": "c2", "text": "most windows are dark", "tier": "should"}}], "traps": ["a person sleeping in bed", "a sleeping cat"], "queries": [{{"q": "empty street night", "for": ["core", "c1"], "type": "scene"}}, {{"q": "quiet city night lamps", "for": ["core", "c2"], "type": "scene"}}, {{"q": "deserted road night", "for": ["core"], "type": "scene"}}]}}"""

BRIEF_RULE = ("\nA line may come with a suggested shot. Follow it when it shows the meaning; when it only "
              "illustrates one word of a figurative or abstract line, or goes against the film bible or the world, "
              "show the meaning instead. Say which in \"brief\": \"kept\" or \"changed: <short reason>\".\n")
REASK_NOTE = ("\n\nYour previous answers for these shots were missing or rejected — answer them again, "
              "every field:\n{reasons}")
_STAT_NOTE = " (a big number is shown on screen over this shot — choose a calm, uncluttered picture)"


def _group_key(b):
    """Части одной фразы (под-кадры split_long_blocks) делят orig_index; у
    блока без него — своя группа."""
    oi = b.get("orig_index")
    return ("orig", oi) if oi is not None else ("id", id(b))


def chapter_packets(video_dir, blocks):
    """Главы как самостоятельные вопросы: фразы по порядку, части одной
    фразы помечены, у главы — хвост предыдущей и начало следующей."""
    import shot_brief_director as sbd
    ctx = sbd.episode_context(video_dir)
    groups, order = {}, []
    for i, b in enumerate(blocks):
        sec = b.get("section") or "—"
        if sec not in groups:
            groups[sec] = []
            order.append(sec)
        if _clean(b.get("text")):
            groups[sec].append(b)
    packets = []
    for k, sec in enumerate(order):
        items = groups[sec]
        if not items:
            continue
        units, part = [], {}
        for n, b in enumerate(items, 1):
            g = _group_key(b)
            part.setdefault(g, []).append(n)
            units.append({"n": n, "text": _clean(b.get("text")), "group": g,
                          "author_brief": _clean(b.get("shot_brief")) or None,
                          "stat": bool(b.get("stat"))})
        for u in units:
            members = part[u["group"]]
            u["parts"] = (members[0], members[-1]) if len(members) > 1 else None
        prev = groups[order[k - 1]] if k > 0 else []
        nxt = groups[order[k + 1]] if k + 1 < len(order) else []
        packets.append({"section": sec, "episode_title": ctx.get("title") or "",
                        "units": units,
                        "prev_tail": [_clean(b.get("text"))[:220] for b in prev[-2:]],
                        "next_head": _clean(nxt[0].get("text"))[:220] if nxt else ""})
    return packets


def windows_of(units, size=WINDOW):
    """Окна по size кадров; части одной фразы в одном окне (не рвутся)."""
    out, cur = [], []
    i = 0
    while i < len(units):
        j = i
        while j + 1 < len(units) and units[j + 1]["group"] == units[i]["group"]:
            j += 1
        grp = units[i:j + 1]
        if cur and len(cur) + len(grp) > size:
            out.append(cur)
            cur = []
        cur.extend(grp)
        i = j + 1
    if cur:
        out.append(cur)
    return out


def render_window_prompt(packet, window, bible, world, done=None, reasons=None):
    """Вопрос по окну главы. done — {n: кадр} уже спланированных кадров этой
    главы (для разнообразия и связности); reasons — {n: почему прежний ответ
    отклонён} при переспросе."""
    lines = []
    for u in packet["units"]:
        brief = f" — suggested shot: {u['author_brief']}" if u.get("author_brief") else ""
        stat = _STAT_NOTE if u.get("stat") else ""
        part = ""
        if u.get("parts") and u["n"] == u["parts"][0]:
            part = f"  [shots {u['parts'][0]}-{u['parts'][1]} are one sentence shown as separate pictures]"
        lines.append(f"{u['n']}. «{u['text']}»{brief}{stat}{part}")
    around = []
    if packet.get("prev_tail"):
        around.append("the previous chapter ended with: " + " / ".join(f"«{t}»" for t in packet["prev_tail"]))
    if packet.get("next_head"):
        around.append(f"the next chapter begins: «{packet['next_head']}»")
    around = (" (" + "; ".join(around) + ")") if around else ""
    ns = [u["n"] for u in window]
    done_txt = ""
    if done:
        done_txt = ("; these shots of the chapter are already planned — do not repeat their pictures: "
                    + "; ".join(f"{n}: {s}" for n, s in sorted(done.items())))
    has_brief = any(u.get("author_brief") for u in window)
    text = SPEC_PROMPT.format(bible=bible_block(bible), world=world, section=_clean(packet["section"]),
                              around=around, lines="\n".join(lines), first=min(ns), last=max(ns),
                              done=done_txt, brief_key=', "brief": "..."' if has_brief else "",
                              brief_rule=BRIEF_RULE if has_brief else "", c1=MAX_CLAIMS - 1, q=MAX_QUERIES)
    if ns != list(range(min(ns), max(ns) + 1)):
        text += f"\n\n(Plan only shots {', '.join(str(n) for n in ns)}.)"
    if reasons:
        text += REASK_NOTE.format(reasons="\n".join(f"{n}: {r}" for n, r in sorted(reasons.items())))
    return text


def _parse_claims(raw_claims):
    """Утверждения по порядку важности, или None: первое обязано быть must
    (это главное), id уникальны, движение требует не больше одно."""
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
        # фраза остаются.
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
    """Запросы по важности того, что они ищут: сначала ищущие главное, дальше
    по самому важному утверждению цели; внутри — порядок модели."""
    rank = {c["id"]: i for i, c in enumerate(spec["claims"])}
    return sorted(spec["queries"], key=lambda x: min(rank[t] for t in x["for"]))


def focus_query_count(spec):
    first = spec["claims"][0]["id"]
    return sum(1 for x in spec["queries"] if first in x["for"])


def parse_shot(obj, unit=None, reasked=False):
    """(задание | None, почему отклонено) из одного объекта ответа.

    Отклонение — не ошибка: кадр переспрашивается с причиной, а после
    REASK_ROUNDS идёт прежним путём. Поэтому проверки строгие по смыслу:
    образ метафоры в кадре, в главном или в обязательном утверждении и кадр,
    начатый местоимением («he is lying...» — пересказ фразы, а не то, что
    видно), до отбора не доходят."""
    # Длина — не смысл: пределы вдвое шире просимых. Замер 26.09 без
    # рассуждения: 10 фраз эп.02 из 142 отклонялись за смысл в 27-28 слов
    # на длинных фразах, и переспрос не помогал — годная работа терялась
    # из-за счёта слов.
    meaning = clean_text(obj.get("meaning"), lo=3, hi=40)
    shot = clean_text(obj.get("shot"), lo=4, hi=32)
    core = clean_text(obj.get("core"), lo=2, hi=20)
    if not meaning:
        return None, "no meaning in English"
    if not shot:
        return None, "no shot of 4-32 English words"
    if not core:
        return None, "no core"
    reading = _clean(obj.get("reading")).lower()
    # Образ бывает и внутри буквальной фразы («берёшь его обратным хватом,
    # как ледоруб» — действие буквальное, ледоруб — сравнение): правило
    # «образ не показывать» действует при ЛЮБОМ чтении. Замер 26.09: фраза,
    # записанная буквальной, получила запрос «ice axe reverse grip climbing».
    vehicle = _vehicle_words(obj.get("vehicle"))
    if reading not in READINGS:
        reading = "figurative" if vehicle else "literal"
    for name, text in (("shot", shot), ("core", core)):
        if _starts_with_pronoun(text):
            return None, f"the {name} starts with a pronoun — name what is visible instead"
        made = mentions(text, _COMPOSITE_WORDS)
        if made:
            return None, (f"the {name} is a composite made in editing («{made}») — describe a picture "
                          f"that already exists")
        leak = mentions(text, vehicle)
        if leak:
            return None, f"the {name} shows the vehicle «{leak}» — show the meaning, not the image in the words"
    # О чём фраза (ссылки разрешены) — это и показывает главное. Замер 26.09,
    # эп.94: «Но именно он решал исход поединка, когда меч уже бесполезен» —
    # смысл разрешён верно («the dagger, not the sword»), а главным дважды
    # вставал меч: соседние кадры уже показали кинжал, и модель искала
    # разнообразие в другом предмете. Модель сама называет предмет фразы, код
    # сверяет с ним главное; идея без предмета — пустое поле, проверки нет.
    # Сверка по словам не знает синонимов («the ground» и «churned clay»), на
    # кэше эп.02 почти все её расхождения мнимые (9 из 136), поэтому
    # расхождение переспрашивается ОДИН раз, с причиной, а ответ переспроса
    # принимается: подмену модель исправляет, синоним оставляет.
    about = clean_text(obj.get("about"), lo=1, hi=8) if isinstance(obj.get("about"), str) else None
    about_off = bool(about) and not shares_a_word(core, about)
    if about_off and not reasked:
        return None, (f"the core must show what the line is about — «{about}»: name it in the core "
                      f"and show it, not something else instead")
    rest = [c for c in (obj.get("claims") or []) if isinstance(c, dict)
            and _clean(str(c.get("id") or "")).lower() != CORE_ID]
    musts = 0
    for c in rest:
        if c.get("tier") == "must":
            musts += 1
            if musts > MAX_MUST_EXTRA:
                c["tier"] = "should"
    claims = _parse_claims([{"id": CORE_ID, "text": core, "tier": "must"}] + rest)
    if not claims:
        return None, "no valid claims"
    for c in claims[1:]:
        leak = mentions(c["text"], vehicle)
        if leak and c["tier"] == "must":
            return None, f"a must claim asks for the vehicle «{leak}»"
    # Утверждение «should» с образом метафоры не нужно, но и кадр из-за него
    # не отклоняется: оно лишь украшало бы картинку.
    claims = [c for c in claims if c["tier"] == "must" or not mentions(c["text"], vehicle)]
    # В запросах запрещён и главный предмет образа из нескольких слов:
    # образ «iron rain» пропускал запрос «arrow rain» (замер 26.09, эп.02,
    # «И ещё они сжимали строй»).
    banned_q = vehicle + [h for h in (head_noun(v) for v in vehicle if " " in v) if h]
    queries = [q for q in _parse_queries(obj.get("queries"), {c["id"] for c in claims})
               if not mentions(q["q"], banned_q)]
    spec = {"focus": shot, "meaning": meaning, "reading": reading, "vehicle": vehicle,
            "claims": claims, "queries": queries}
    if about:
        spec["about"] = about
    if about_off:
        spec["about_off"] = True
    if not focus_query_count(spec):
        return None, "no query searches for the core (queries with the vehicle are dropped)"
    spec["queries"] = order_queries(spec)
    traps = _str_list(obj.get("traps"), lo=2, hi=12, limit=MAX_TRAPS)
    for v in vehicle:
        if not any(mentions(t, [v]) for t in traps):
            traps.append(v)
    spec["traps"] = traps[:MAX_TRAPS + MAX_VEHICLE]
    brief = _clean(obj.get("brief"))
    if unit is not None and unit.get("author_brief") and brief:
        spec["brief"] = brief[:160]
    return spec, None


def parse_window(raw, window, reasked=False):
    """({n: задание}, {n: почему нет}) по окну. Номер вне окна, повтор или
    сорванная строка теряют только себя."""
    by_n = {u["n"]: u for u in window}
    got, why = {}, {}
    for obj in json_objects(raw):
        n = obj.get("n")
        if isinstance(n, str) and n.strip().isdigit():
            n = int(n.strip())
        if not isinstance(n, int) or n not in by_n or n in got:
            continue
        spec, reason = parse_shot(obj, by_n[n], reasked=reasked)
        if spec:
            got[n] = spec
            why.pop(n, None)
        else:
            why[n] = reason
    for n in by_n:
        if n not in got and n not in why:
            why[n] = "missing from the answer"
    return got, why


def _cache_path(cache_dir, model, reasoning, prompt):
    key = hashlib.sha256(f"{PLAN_VERSION}|{model}|{reasoning}|{prompt}".encode("utf-8")).hexdigest()[:24]
    return os.path.join(cache_dir, key + ".txt")


def ask(gateway, model, prompt, cache_dir, reasoning=REASONING):
    """(ответ, из кэша ли, оборван ли лимитом). Кэш — по содержимому
    вопроса: повторный прогон не платит. Пустой и оборванный ответы в кэш
    не пишутся (их части переспросятся)."""
    cp = _cache_path(cache_dir, model, reasoning, prompt)
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            return f.read(), True, False
    text, usage, _p = gateway.chat(model, [{"type": "text", "text": prompt}], MAX_TOKENS[bool(reasoning)],
                                   max(EST_PROMPT_TOKENS, len(prompt) // 3), reasoning=reasoning,
                                   timeout=CALL_TIMEOUT)
    cut = (usage or {}).get("finish_reason") == "length"
    if text.strip() and not cut:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cp + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, cp)
    return text, False, cut


_REASON_KINDS = (("missing from the answer", "missing"), ("answer cut", "cut"),
                 ("what the line is about", "about"), ("starts with a pronoun", "pronoun"),
                 ("composite made in editing", "composite"), ("no query searches", "no_core_query"),
                 ("vehicle", "vehicle"))


_STATS_LOCK = threading.Lock()


def _bump(stats, key, n=1, kind=None):
    """Счётчик сводки плана. Главы спрашиваются параллельно в одну сводку:
    прибавление под замком, иначе одновременные записи теряются."""
    with _STATS_LOCK:
        if kind is None:
            stats[key] = stats.get(key, 0) + n
        else:
            sub = stats.setdefault(key, {})
            sub[kind] = sub.get(kind, 0) + n


def reason_kind(reason):
    """Вид отказа для сводки плана (stats.rejected): по сводке видно, ЧТО
    стоит переспросов — без неё число переспросов (32 на 142 фразы эп.02,
    26.09) нечем было объяснить."""
    r = str(reason)
    for needle, kind in _REASON_KINDS:
        if needle in r:
            return kind
    return "form"


def plan_chapter(gateway, model, packet, bible, world, cache_dir, need=None, keep=None,
                 reasoning=REASONING, stats=None):
    """({n: задание}, {n: почему нет}) по главе. need — номера, которые надо
    спланировать (None — все); keep — {n: задание} из прежнего плана: их
    кадры видны модели как уже спланированные. Окна спрашиваются по порядку
    (следующее окно знает кадры предыдущих), недостающие и отклонённые —
    переспрашиваются только они, окнами вдвое меньше, с причиной отказа."""
    import llm_gateway
    stats = stats if stats is not None else {}
    keep = dict(keep or {})
    todo = [u for u in packet["units"] if need is None or u["n"] in need]
    got, why = {}, {}

    def run(windows, reasons=None):
        for window in windows:
            done = {n: s["focus"] for n, s in sorted({**keep, **got}.items())
                    if n not in {u["n"] for u in window}}
            prompt = render_window_prompt(packet, window, bible, world, done=done,
                                          reasons={u["n"]: reasons[u["n"]] for u in window
                                                   if reasons and u["n"] in reasons} or None)
            try:
                raw, hit, cut = ask(gateway, model, prompt, cache_dir, reasoning)
            except llm_gateway.PaymentRequired:
                raise
            except llm_gateway.EmptyAnswer as e:
                # Оплаченный ответ без текста — не лежащий шлюз, а сорванное
                # окно (чаще всего рассуждение съело весь лимит выхода):
                # переспрашивается здесь же окнами вдвое меньше. Замер 26.09:
                # 9 пустых ответов из 22 вызовов главы эп.02, и 45 фраз из 142
                # остались без задания, потому что такие окна не переспрашивались.
                for u in window:
                    why[u["n"]] = f"empty answer: {str(e)[:160]}"
                _bump(stats, "empty")
                # length — рассуждение съело весь лимит выхода; stop — сервис
                # ответил пустым сам: разные лечения, поэтому разные счётчики.
                m = re.search(r"finish_reason=(\w+)", str(e))
                _bump(stats, "empty_finish", kind=m.group(1) if m else "?")
                continue
            except llm_gateway.GatewayError as e:
                for u in window:
                    why[u["n"]] = f"gateway: {str(e)[:160]}"
                _bump(stats, "gateway_errors")
                continue
            _bump(stats, "calls", 0 if hit else 1)
            _bump(stats, "cache_hits", 1 if hit else 0)
            _bump(stats, "cut", 1 if cut else 0)
            g, w = parse_window(raw, window, reasked=bool(reasons))
            got.update(g)
            for n in g:
                why.pop(n, None)
            for n, r in w.items():
                why[n] = ("answer cut by the output limit" if cut and r == "missing from the answer" else r)
                _bump(stats, "rejected", kind=reason_kind(why[n]))

    run(windows_of(todo))
    size = WINDOW
    for _round in range(REASK_ROUNDS):
        left = [u for u in todo if u["n"] not in got and not _outage(why.get(u["n"], ""))]
        if not left:
            break
        size = max(1, size // 2)
        _bump(stats, "reasked", len(left))
        run(windows_of(left, size), reasons=dict(why))
    return got, {n: r for n, r in why.items() if n not in got}


def world_digest_of(card):
    import world_card
    return world_card.world_digest(card) if card else ""


def plan_signature(model, card, reasoning=REASONING):
    """Что делает задания сопоставимыми между прогонами: версия, модель,
    режим рассуждения, текст инструкции и мир эпизода. Совпадает — задание
    фразы с неизменным текстом берётся из прежнего плана.

    Библии здесь нет СОЗНАТЕЛЬНО. Библия читает весь сценарий и меняется от
    правки любой строки; будь она в подписи, правка одного слова
    перепокупала бы задания всех фраз, а задание входит в ключ кэша
    кандидата — то есть переотбор и перепроверку судьёй всего эпизода.
    Задание фразы следует её тексту; главы с новыми фразами спрашиваются с
    новой библией."""
    return hashlib.sha256(f"{PLAN_VERSION}|{model}|{reasoning}|{SPEC_PROMPT}|{BRIEF_RULE}|"
                          f"{world_digest_of(card)}".encode("utf-8")).hexdigest()[:16]


def _outage(reason):
    """Шлюз или код не работают: переспрос в этом же прогоне бесполезен."""
    return str(reason).startswith(("gateway", "error"))


def _transient(reason):
    """Причина — сбой (связи, кода или пустой оплаченный ответ), а не ответ
    модели о фразе: такая фраза не записывается в «не удалось» и
    спрашивается на следующем рендере."""
    return _outage(reason) or str(reason).startswith("empty answer")


def _unit_key(text):
    import shot_planner_llm
    return shot_planner_llm.unit_key(text or "")


def _spec_of(unit):
    """Задание в рабочей форме (запросы — объектами) из записи плана, где
    queries — строки, а объекты лежат в queries_for."""
    spec = {k: v for k, v in unit.items() if k not in ("text", "queries", "queries_for")}
    spec["queries"] = list(unit.get("queries_for") or [])
    return spec


def plan_episode(video_dir, blocks, gateway, model=DEFAULT_MODEL, verbose=True, workers=None,
                 reasoning=REASONING, card=None, bible="auto"):
    """Спланировать эпизод и записать план. Возвращает число фраз с
    заданием. blocks — финальные блоки рендера (после нарезки и слияния)
    или блоки сценария (CLI). Главы спрашиваются параллельно, окна главы —
    по порядку. Сбой главы не рвёт прогон; нехватка денег поднимается.

    bible="auto" — библия с диска или новым вопросом; None — без библии
    (замер); dict — готовая."""
    import world_card
    if card is None:
        card = world_card.load(video_dir, strict=False)
    what = "given" if bible else "none"
    if bible == "auto":
        bible, what = make_bible(video_dir, gateway, card, model=model, reasoning=reasoning)
        if verbose or what.startswith("failed"):
            print(f"  Библия фильма: {what}" + (f" — {bible['topic'][:120]}" if bible else ""))
    world = world_lines(card)
    cache_dir = os.path.join(video_dir, "media_plan", CACHE_DIR_NAME)
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    sig = plan_signature(model, card, reasoning)
    old = _read_json(path)
    old = old if old.get("version") == PLAN_VERSION else {}
    old_units = (old.get("units") or {}) if old.get("sig") == sig else {}
    old_failed = (old.get("failed") or {}) if old.get("sig") == sig else {}
    packets = chapter_packets(video_dir, blocks)
    stats = {"bible": what}

    def one(packet):
        keep = {u["n"]: _spec_of(old_units[_unit_key(u["text"])]) for u in packet["units"]
                if _unit_key(u["text"]) in old_units}
        need = {u["n"] for u in packet["units"] if u["n"] not in keep
                and _unit_key(u["text"]) not in old_failed}
        if not need:
            return keep, {}, 0
        try:
            got, why = plan_chapter(gateway, model, packet, bible, world, cache_dir, need=need, keep=keep,
                                    reasoning=reasoning, stats=stats)
        except Exception as e:  # noqa: BLE001 — PaymentRequired поднимается ниже
            import llm_gateway
            if isinstance(e, llm_gateway.PaymentRequired):
                raise
            # Не ответ о фразе, а сбой: в «не удалось» не записывается,
            # следующий рендер спросит снова.
            return keep, {n: f"error: {type(e).__name__}: {str(e)[:160]}" for n in need}, len(need)
        return {**keep, **got}, why, len(need)

    units, failed, retry, kept = {}, {}, {}, 0
    with concurrent.futures.ThreadPoolExecutor(max(1, min(workers or len(packets) or 1, 8))) as ex:
        results = list(ex.map(one, packets))
    for packet, (specs, why, asked) in zip(packets, results):
        for u in packet["units"]:
            key = _unit_key(u["text"])
            if u["n"] in specs:
                spec = specs[u["n"]]
                units[key] = dict(spec, text=u["text"], queries_for=spec["queries"],
                                  queries=[x["q"] for x in spec["queries"]])
                kept += 1 if key in old_units else 0
            elif key in old_failed:
                failed[key] = old_failed[key]
            elif u["n"] in why and not _transient(why[u["n"]]):
                failed[key] = {"text": u["text"], "why": why[u["n"]]}
            elif u["n"] in why:
                # Сбой связи или кода: следующий рендер спросит снова, но
                # причина видна сейчас, а не только счётчиком.
                retry[key] = {"text": u["text"], "why": why[u["n"]]}
        if verbose and asked:
            print(f"  глава «{_clean(packet['section'])[:40]}»: заданий {sum(1 for u in packet['units'] if u['n'] in specs)}"
                  f" из {len(packet['units'])}")
    lost = [u["text"] for p in packets for u in p["units"] if _unit_key(u["text"]) not in units]
    if lost:
        print(f"  ВНИМАНИЕ: без задания {len(lost)} фраз из {sum(len(p['units']) for p in packets)} — "
              f"они идут прежним путём: " + " | ".join(t[:50] for t in lost[:6])
              + (" ..." if len(lost) > 6 else ""))
    _write_json(path, {"version": PLAN_VERSION, "model": model, "reasoning": reasoning, "sig": sig,
                       "bible_digest": bible_digest(bible), "world": world, "units": units,
                       "failed": failed, "retry": retry, "stats": stats})
    write_readable(video_dir, packets, units, {**failed, **retry}, bible)
    return len(units)


def write_readable(video_dir, packets, units, failed, bible):
    """media_plan/shot_plan.txt — понимание фильма и каждого кадра для
    человека: «почему этот кадр» читается до рендера, а не угадывается."""
    out = []
    if bible:
        out.append("БИБЛИЯ ФИЛЬМА\n" + bible_block(bible))
    for p in packets:
        out.append(f"=== {p['section']}")
        for u in p["units"]:
            spec = units.get(_unit_key(u["text"]))
            out.append(f"{u['n']}. {u['text']}")
            if not spec:
                why = (failed.get(_unit_key(u["text"])) or {}).get("why", "нет задания")
                out.append(f"   — задания нет ({why}); кадр прежним путём")
                continue
            out.append(f"   смысл: {spec.get('meaning')}  [{spec.get('reading')}]")
            if spec.get("about"):
                out.append(f"   о чём: {spec['about']}" + ("  (в главном другими словами — проверить глазами)"
                                                        if spec.get("about_off") else ""))
            if spec.get("vehicle"):
                out.append(f"   не показывать образ: {', '.join(spec['vehicle'])}")
            out.append(f"   кадр: {spec['focus']}")
            out.append(f"   главное: {spec['claims'][0]['text']}")
            musts = [c["text"] for c in spec["claims"][1:] if c["tier"] == "must"]
            if musts:
                out.append("   обязательно: " + "; ".join(musts))
            if spec.get("traps"):
                out.append("   ловушки: " + "; ".join(spec["traps"]))
            out.append("   запросы: " + " | ".join(spec["queries"]))
            if spec.get("brief"):
                out.append(f"   предложение автора: {spec['brief']}")
    path = os.path.join(video_dir, "media_plan", READABLE_NAME)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    except OSError:
        pass


def needs_planning(video_dir, blocks, model=DEFAULT_MODEL, reasoning=REASONING):
    """Есть ли фразы без задания текущей версии и подписи (или плана нет).
    Фраза, для которой модель уже не смогла дать годного задания при этой
    подписи, не спрашивается заново на каждом рендере. Без сети."""
    import world_card
    plan = _read_json(os.path.join(video_dir, "media_plan", PLAN_NAME))
    if plan.get("version") != PLAN_VERSION:
        return True
    card = world_card.load(video_dir, strict=False)
    if plan.get("sig") != plan_signature(model, card, reasoning):
        return True
    have = set(plan.get("units") or {}) | set(plan.get("failed") or {})
    return any(_unit_key(b.get("text") or "") not in have for b in blocks if _clean(b.get("text")))


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


SPEC_FIELDS = ("meaning", "about", "reading", "vehicle", "traps")


def load_specs(video_dir):
    """{ключ юнита: задание} из плана текущей версии, или {}. План другой
    версии заданий не даёт: молча смешивать их с этими нельзя."""
    path = os.path.join(video_dir, "media_plan", PLAN_NAME)
    if not os.path.exists(path):
        return {}
    data = _read_json(path)
    if data.get("version") != PLAN_VERSION:
        print(f"  {PLAN_NAME}: версия {data.get('version')}, нужна {PLAN_VERSION} — "
              f"задания кадров не используются; перепланировать: "
              f"python scripts/stock_query_planner.py <эпизод>")
        return {}
    out = {}
    for k, v in (data.get("units") or {}).items():
        if isinstance(v, dict) and v.get("claims") and v.get("focus"):
            out[k] = {"focus": v["focus"], "claims": v["claims"], "queries": v.get("queries_for") or []}
            for extra in SPEC_FIELDS:
                if v.get(extra):
                    out[k][extra] = v[extra]
    return out


def has_motion(spec, must=False):
    """Есть ли у фразы утверждение движения (must=True — обязательное)."""
    return any(c.get("motion") and (not must or c.get("tier") == "must")
               for c in (spec or {}).get("claims") or [])


def attach(blocks, plan, specs=None):
    """Проставить блокам b["phrase_queries"] и b["shot_spec"] по тексту
    фразы. Вызывается по ФИНАЛЬНЫМ блокам рендера (после нарезки и
    слияния): у каждого под-кадра свой текст и своё задание."""
    if not plan:
        return 0
    n = 0
    for b in blocks:
        key = _unit_key(b.get("text") or "")
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
    print(f"Готово: задания на {n} фраз из {len(blocks)} (читаемо: media_plan/{READABLE_NAME}). "
          f"{gw.summary()}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
