#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Судья кадров: модель «зрение + язык» смотрит на кандидатов слота и
оценивает каждого по описанию кадра (брифу), фразе диктора и миру эпизода.

ЗАЧЕМ. Прежний судья — косинус эмбеддингов картинки и текста с порогами —
не рассуждает: «средневековый рондельный кинжал» и современный нож в
ножнах для него почти одно и то же, действие («стрела СКОЛЬЗИТ по
нагруднику») он не видит, и каждую нишу приходилось подпирать своими
ловушками и блоклистами. Замер 23.09 на кадрах эпизода 94 с известным
ответом (4 слота, 28 кадров, правильный ответ проставлен глазами):

  судья                       пары верно   брак принят за годный
  Qwen 3.7 Plus, сеткой       54/64        0
  Gemini 3.7 Flash, сеткой    51/64        0
  Llama 3.2 11B Vision        13/64        8
  GLM-4.6V                    13/44        6

СЕТКОЙ, А НЕ ПО ОДНОМУ. Кандидаты слота уходят одной картинкой-сеткой 3x3
с номерами: бриф и инструкция передаются один раз, а модель сравнивает
кандидатов между собой, как монтажёр. На том же замере это в 3.5-5 раз
дешевле вызовов по одному и не хуже по точности (Qwen стал точнее: 46 -> 54
верных пар).

МИР ЭПИЗОДА — ОДНОЙ СТРОКОЙ, А НЕ ПАСПОРТОМ (оба замера на кадрах эп. 94).
Паспорт целиком (годы, чужие культуры, «запрещено в кадре») ухудшил обе
модели (верных пар из 61: Qwen 53 -> 45, Gemini 57 -> 42): длинный список
запретов сжимает оценки к единице. Одна строка «регистр, годы» из того же
паспорта (world_card.judge_setting) — улучшает ПОРЯДОК оценок, но брак
почти не снижает. Повторные прогоны без кэша (36 кадров, два слота): со
строкой верных пар 94/101/91 из 201, одобренного брака 7/8/6; без строки —
86/82 и 7/8. Первый замер («брак 10 -> 4, пары 93 -> 116») НЕ
воспроизвёлся: модель шумит от прогона к прогону сильнее, чем сдвигала
правка, и один прогон на 36 кадрах ничего не доказывает. Одни и те же
кадры чужой эпохи судья принимает во всех прогонах — предел зрения
модели, а не формулировки: ужесточённая шкала («мультфильм, 3D, игрушка —
это 1») на том же стенде дала пары 80/75/87 — хуже, и отклонена.

ШКАЛА (та же, что замерена):
  3 — показан требуемый предмет и действие, в нужной эпохе и культуре;
  2 — предмет тот, но действие или композиция другие, или близкая замена;
  1 — связано, но не тот предмет, эпоха или культура;
  0 — не по теме, современное там, где эпоха историческая, или негодно.

НАДЁЖНОСТЬ:
  * кэш по СОДЕРЖИМОМУ: ключ — модель, версия вопроса, текст вопроса и
    отпечатки байтов картинок в порядке сетки. Повторный прогон не платит;
    другая картинка под тем же id — другой ключ;
  * ответ разбирается строго: каждый номер сетки обязан получить целое 0..3;
    ответ без полной сетки — не оценка (None), а не угаданные нули;
  * любой сбой шлюза — None для всего вызова: вызывающий код обязан
    остаться на прежнем ранжировании (fail-open), а не смешивать оценённых
    кандидатов с неоценёнными.
"""
import base64
import concurrent.futures
import hashlib
import io
import json
import os
import threading

PROMPT_VERSION = 2
GRID_COLS = 3
GRID_MAX = 9
TILE = (400, 300)
SCORE_MAX = 3
# Оценка вызова: сетка 1200x900 + текст — порядка 1500 токенов входа у
# замеренных моделей; ответ — JSON из девяти чисел. Резерв для потолка
# расходов шлюза, а не цена: списывается фактический usage.
EST_PROMPT_TOKENS = 2500
MAX_TOKENS = 1200

PROMPT = """You check shots for a documentary video.
Narration line: «{phrase}»
Required shot: «{brief}»
The picture is a grid of {n} numbered candidate images (numbers in the top-left corner of each tile).
Rate EACH candidate for that shot:
3 = shows the required subject and action, in the right era and culture
2 = right subject, but the action or composition differs, or a close substitute
1 = related, but wrong subject, era or culture
0 = unrelated, modern where the era is historical, or unusable
Reply with JSON only: {{"scores": {{"1": <0-3>, "2": <0-3>, ...}}}}"""


# Видео: плитка — лента из трёх кадров ОДНОГО ролика (15/50/85% длины —
# превью источника, без скачивания ролика). Лента в 5 раз шире своей высоты,
# поэтому сетка в одну колонку: в три колонки кадры стали бы нечитаемыми.
PROMPT_VIDEO = """You check shots for a documentary video.
Narration line: «{phrase}»
Required shot: «{brief}»
The picture is a stack of {n} numbered video clips (numbers in the top-left corner of each row).
Each row shows three frames of ONE clip (beginning, middle, end); judge the clip as a whole, including what happens in it.
Rate EACH clip for that shot:
3 = shows the required subject and action, in the right era and culture
2 = right subject, but the action or composition differs, or a close substitute
1 = related, but wrong subject, era or culture
0 = unrelated, modern where the era is historical, or unusable
Reply with JSON only: {{"scores": {{"1": <0-3>, "2": <0-3>, ...}}}}"""

# Раскладка по виду медиа: (колонок, плитка, кандидатов в сетке, вопрос).
# Фото — ровно замеренная постановка (3x3, 400x300); её числа не трогать
# без нового замера.
LAYOUTS = {"photo": (GRID_COLS, TILE, GRID_MAX, PROMPT),
           "video": (1, (1200, 240), 6, PROMPT_VIDEO)}


def question(phrase, brief, n, kind="photo", setting=None):
    """Вопрос судье. setting — мир эпизода одной строкой (world_card.
    judge_setting): без него судья не знает эпохи, если её нет в самом
    описании кадра. Что строка даёт и чего не даёт — числами в докстринге
    модуля («МИР ЭПИЗОДА»): порядок оценок лучше, брак почти тот же.
    Полный паспорт (годы, чужие культуры, список запретов) по прежнему
    замеру УХУДШАЛ судью — поэтому одна строка, а не паспорт."""
    brief = brief or phrase or "—"
    if setting:
        brief = f"{brief} — setting: {setting}"
    return LAYOUTS[kind][3].format(phrase=phrase or "—", brief=brief, n=n)


def flat_rgb(im):
    """RGB без мусора прозрачности. У PNG с прозрачным фоном (предмет
    «isolated» со стока) цвет прозрачных пикселей произволен, и простое
    convert("RGB") показывает его полосами — судья видел испорченный кадр
    (живой случай: кинжалы Pixabay в сетке слота «Вот кинжал»). Прозрачное
    кладётся на светлый фон, как такие снимки и задуманы."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        from PIL import Image
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (235, 235, 235))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return im.convert("RGB")


def _grid_bytes(paths, kind="photo"):
    from PIL import Image, ImageDraw, ImageFont
    cols, tile, _max, _q = LAYOUTS[kind]
    rows = (len(paths) + cols - 1) // cols
    g = Image.new("RGB", (cols * tile[0], rows * tile[1]), (0, 0, 0))
    d = ImageDraw.Draw(g)
    font = None
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "assets", "fonts", "Montserrat-Bold.ttf")):
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 40)
            break
    for k, p in enumerate(paths):
        im = flat_rgb(Image.open(p))
        im.thumbnail((tile[0] - 8, tile[1] - 8))
        x, y = (k % cols) * tile[0], (k // cols) * tile[1]
        g.paste(im, (x + (tile[0] - im.width) // 2, y + (tile[1] - im.height) // 2))
        d.rectangle([x + 4, y + 4, x + 58, y + 54], fill=(255, 230, 0))
        d.text((x + 16, y + 6), str(k + 1), fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    g.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def parse_scores(text, n):
    """{номер 1..n: целое 0..3} или None, если хоть один номер без оценки."""
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        return None
    scores = obj.get("scores") if isinstance(obj, dict) else None
    if not isinstance(scores, dict):
        return None
    out = {}
    for k in range(1, n + 1):
        v = scores.get(str(k), scores.get(k))
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) \
                or not 0 <= v <= SCORE_MAX:
            return None
        out[k] = int(v)
    return out


def _file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(model, text, paths):
    h = hashlib.sha256()
    for part in (model, str(PROMPT_VERSION), text, *(_file_digest(p) for p in paths)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


# ПРОВЕРКА ЗРЕНИЯ. Не каждая модель шлюза видит картинки, и не каждая об
# этом говорит: qwen3.8-max на замере 23.09 молча поставила ВСЕМ кадрам
# сетки 0 — формально верный ответ, который браковал бы каждый слот.
# Поэтому перед судейством модель один раз на прогон называет цвет
# однотонной картинки; ошиблась — судья выключается громко.
VISION_CANARY_COLORS = (("red", (220, 20, 20)), ("blue", (20, 40, 220)))


def vision_check(gateway, model):
    """(True, "") если модель видит картинку, иначе (False, причина).
    Два цвета, чтобы угадывание одного слова не сходило за зрение."""
    for word, rgb in VISION_CANARY_COLORS:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), rgb).save(buf, "JPEG", quality=90)
        content = [{"type": "text", "text": "What single colour fills this image? Answer with one word."},
                   {"type": "image_url", "image_url": {
                       "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]
        try:
            answer, _u, _p = gateway.chat(model, content, 20, 400)
        except Exception as e:  # noqa: BLE001
            return False, f"проверка зрения не состоялась: {type(e).__name__}: {e}"[:300]
        if word not in (answer or "").lower():
            return False, f"модель {model} не видит картинок (на {word} ответила {answer[:80]!r})"
    return True, ""


def _judge_chunk(gateway, model, text, chunk, cache_dir, kind="photo"):
    """Одна сетка: (оценки {номер: 0..3} | None, запись для отчёта)."""
    paths = [p for _cid, p in chunk]
    key = cache_key(model, text, paths)
    cp = os.path.join(cache_dir, key + ".json") if cache_dir else None
    if cp and os.path.exists(cp):
        try:
            cached = json.load(open(cp, encoding="utf-8"))
            return {int(k): v for k, v in cached["scores"].items()}, {"cache_hit": True}
        except Exception:
            pass
    if gateway is None:
        return None, {"refused": "нет шлюза"}
    content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(_grid_bytes(paths, kind)).decode()}}]
    try:
        answer, _usage, price = gateway.chat(model, content, MAX_TOKENS, EST_PROMPT_TOKENS)
    except Exception as e:  # noqa: BLE001 — любой сбой шлюза: судьи нет
        return None, {"refused": f"{type(e).__name__}: {e}"[:300]}
    scores = parse_scores(answer, len(chunk))
    if scores is None:
        return None, {"refused": "неполный ответ: " + answer[-200:], "cost": price, "call": True}
    if cp:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cp}.{os.getpid()}.{threading.get_ident()}.part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"model": model, "version": PROMPT_VERSION, "scores": scores}, f)
        os.replace(tmp, cp)
    return scores, {"cost": price, "call": True}


def judge(gateway, model, *, phrase, brief, candidates, cache_dir=None, report=None, kind="photo",
          setting=None):
    """candidates — [(id, путь к картинке)] в порядке прежнего ранжирования.
    Возвращает {id: оценка 0..3} для ВСЕХ кандидатов или None (судьи не было:
    нет ключа, сбой, неполный ответ, потолок расходов). Частичных ответов нет.
    Сетки одного слота независимы и спрашиваются параллельно: время слота —
    самая медленная сетка, а не их сумма. report (dict) пополняется:
    вызовы, попадания в кэш, цена, причина отказа."""
    if report is None:
        report = {}
    report.setdefault("calls", 0)
    report.setdefault("cache_hits", 0)
    report.setdefault("cost", 0)
    if not candidates:
        return {}
    per_grid = LAYOUTS[kind][2]
    chunks = [candidates[i:i + per_grid] for i in range(0, len(candidates), per_grid)]
    with concurrent.futures.ThreadPoolExecutor(len(chunks)) as ex:
        answers = list(ex.map(lambda ch: _judge_chunk(gateway, model,
                                                      question(phrase, brief, len(ch), kind, setting),
                                                      ch, cache_dir, kind),
                              chunks))
    refused = None
    for _scores, info in answers:
        report["calls"] += 1 if info.get("call") else 0
        report["cache_hits"] += 1 if info.get("cache_hit") else 0
        report["cost"] += info.get("cost", 0)
        refused = refused or info.get("refused")
    if refused or any(sc is None for sc, _info in answers):
        report["refused"] = refused or "сетка без ответа"
        return None
    result = {}
    for chunk, (scores, _info) in zip(chunks, answers):
        for k, (cid, _p) in enumerate(chunk, start=1):
            result[cid] = scores[k]
    return result


# ПРОВЕРКА МИРА ПОБЕДИТЕЛЯ — узкий вопрос по ОДНОМУ кадру. Шкала в сетке
# одобряет кадры чужой эпохи и с современными вещами стабильно, во всех
# прогонах (марокканское конное шоу, реконструкция Гражданской войны США,
# видео со стрелами в табличках «HOMEWORK»). Тот же вопрос по одному
# кадру — «может ли то, что на нём, существовать в мире эпизода» — модель
# видит: называет тбуриду с баннерами Coca-Cola, мундиры XIX-XX века.
# Замер 23.09 (36 размеченных кадров эпизода 94, два прогона без кэша,
# совпадение ответов 32 из 36): брак (0) проходит 1-2 из 13, «не тот
# предмет/эпоха/культура» (1) — 5 из 16, годные (2) — 3-4 из 7 (два из
# отклонённых годных — рыцарский турнир с современными зрителями). Из
# семи кадров, которые сетка одобряла во всех прогонах, отклонены шесть.
# Цена — ~200 токенов баланса на кадр. Вопрос — ровно замеренный (второй
# пункт про предмет в нём остаётся ради тождества замеру, но не читается:
# он требует точной композиции и отклоняет годные).
WORLD_CHECK_VERSION = 1
WORLD_PROMPT = """You check one shot for a documentary video.
Narration line: «{phrase}»
Required shot: «{brief}»
The episode's world: {setting}.
Look at the picture carefully and answer two questions:
1. world: does it show anything that could NOT exist in that world — modern people, modern clothing or haircuts, modern objects or vehicles, printed text or signs, spectators of a modern show, or a different era or culture?
2. subject: is the main subject the kind of thing the required shot asks for?
Reply with JSON only: {{"world_ok": true/false, "subject_ok": true/false, "why": "<short>"}}"""
WORLD_VIDEO_NOTE = "\nThe picture shows three frames (beginning, middle, end) of ONE video clip."


def parse_world(text):
    """(True/False, почему) или (None, None) — ответ не разобран."""
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, None
    try:
        j = json.loads(m.group(0))
    except ValueError:
        return None, None
    ok = j.get("world_ok")
    return (ok, str(j.get("why") or "")[:300]) if isinstance(ok, bool) else (None, None)


def world_check(gateway, model, *, phrase, brief, setting, path, kind="photo", cache_dir=None):
    """(True — мир не нарушен / False — нарушен / None — проверки не было,
    почему, {"cost", "call", "cache_hit"}). Без строки мира не спрашивает:
    «чужая эпоха» без мира не определена."""
    if not setting or gateway is None or not path or not os.path.exists(path):
        return None, None, {}
    text = WORLD_PROMPT.format(phrase=phrase or "—", brief=brief or phrase or "—", setting=setting)
    if kind == "video":
        text += WORLD_VIDEO_NOTE
    h = hashlib.sha256()
    for part in ("world", str(WORLD_CHECK_VERSION), model, text, _file_digest(path)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    cp = os.path.join(cache_dir, "world_" + h.hexdigest() + ".json") if cache_dir else None
    if cp and os.path.exists(cp):
        try:
            c = json.load(open(cp, encoding="utf-8"))
            return c["ok"], c["why"], {"cache_hit": True}
        except Exception:
            pass
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = flat_rgb(im)
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    except Exception:
        return None, None, {}
    content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]
    try:
        answer, _u, price = gateway.chat(model, content, 400, 1200)
    except Exception as e:  # noqa: BLE001 — сбой шлюза: проверки не было
        return None, None, {"refused": f"{type(e).__name__}: {e}"[:200]}
    ok, why = parse_world(answer)
    if ok is not None and cp:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cp}.{os.getpid()}.{threading.get_ident()}.part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"ok": ok, "why": why, "model": model}, f, ensure_ascii=False)
        os.replace(tmp, cp)
    return ok, why, {"cost": price, "call": True}


# Общие части вопроса проверки кадра (verify_claims ниже). Прежняя проверка
# по пунктам «тот ли предмет: yes/close/no» + замены из плана (VERIFY_VERSION
# 3) удалена: «close» засчитывал замену без главного — нагрудник на фразе
# «Стрела скользит по нагруднику».
VERIFY_WORLD = "\nThe episode's world: {setting}."
VERIFY_WORLD_KEYS = ', "main_in_world": true/false, "background_foreign": true/false'
# Подпись источника — свидетельство рядом с картинкой. Замер 24.09 (эп.94):
# все три промаха проверки по одной картинке названы в подписи прямо —
# «moroccan horsemen perform a tbourida», «fish shaped metal keychain»,
# «drone shot of a man lying on dry soil». Подпись бывает неполной и
# неверной, поэтому она — довод, а не приговор.
VERIFY_CAPTION = ("\nThe source's own caption for this picture (may be incomplete or wrong; "
                  "use it as evidence, the picture decides): «{caption}»")
VERIFY_CAPTION_MAX = 240
MEDIUMS = ("photo", "artwork", "object", "cg")
# Сторона картинки для проверки. Цена вызова почти целиком — картинка:
# на 1024 px замер 24.09 дал ~540 токенов баланса за кадр.
VERIFY_MAX_SIDE = 512


# ПРОВЕРКА ПО УТВЕРЖДЕНИЯМ СПЕЦИФИКАЦИИ (план 24.09, «одна спецификация кадра
# на фразу»). Прежняя проверка спрашивала «тот ли предмет: yes/close/no», а
# «close» включал замены из плана — и на фразе «Стрела скользит по
# нагруднику» нагрудник без стрелы получил «close», то есть «годен». Решать,
# что в кадре главное, проверка не должна: это смысл фразы, и его уже
# записал планировщик (stock_query_planner v3) — фокус и утверждения по
# убыванию важности. Здесь модель только отвечает, выполнено ли каждое
# утверждение на ЭТОЙ картинке; сравнивает кадры код, по вектору в порядке
# спецификации (claims_vector). Замена получается сама: картина со стрелами
# выполняет «видна стрела», нагрудник — нет, и проигрывает ей.
#
# Утверждение с движением (motion) фото выполнить не может физически — его не
# спрашивают и ставят «нет»; у видео спрашивают по кадрам ленты.
CLAIMS_VERSION = 1
CLAIM_ANSWERS = {"yes": 2, "unsure": 1, "no": 0}
CLAIMS_PROMPT = """You check one shot for a documentary video.
Narration line: «{phrase}»
What the viewer must see: «{focus}»{world}{caption}{video}
Look at the picture carefully. For each statement answer "yes", "no" or "unsure" — about THIS picture only:
{claims}
Then:
- medium: "photo", "artwork" (painting, drawing, engraving, manuscript), "object" (museum object on a plain background) or "cg" (3D render, cartoon, toy, video game){world_q}
Reply with JSON only: {{"claims": {{{keys}}}, "medium": "..."{world_keys}, "why": "<short>"}}"""
CLAIMS_VIDEO_NOTE = ("\nThe picture shows three frames (beginning, middle, end) of ONE video clip; "
                     "judge the clip, a movement counts if the frames show it happening.")
CLAIMS_WORLD_Q = """
- main_in_world: could the MAIN subject exist in that world (era, culture)? true/false
- background_foreign: is there anything ELSE in the picture (background, edges, people around) that could not exist in that world — modern people, clothing, objects, vehicles, signs, spectators? true/false"""


def spec_from_brief(phrase, brief):
    """Спецификация без плана: одно must-утверждение — бриф (или сама фраза).
    Путь один и тот же со спецификацией и без неё."""
    text = (brief or phrase or "").strip() or "—"
    return {"focus": text, "claims": [{"id": "c1", "text": text, "tier": "must"}]}


def asked_claims(spec, kind):
    """Утверждения, которые спрашиваются у кадра этого вида: движение у фото
    не спрашивают — фото его показать не может."""
    return [c for c in spec["claims"] if kind == "video" or not c.get("motion")]


def claims_question(phrase, spec, setting=None, kind="photo", caption=None):
    asked = asked_claims(spec, kind)
    caption = " ".join(str(caption or "").split())[:VERIFY_CAPTION_MAX]
    return CLAIMS_PROMPT.format(
        phrase=phrase or "—", focus=spec.get("focus") or "—",
        world=VERIFY_WORLD.format(setting=setting) if setting else "",
        caption=VERIFY_CAPTION.format(caption=caption) if caption else "",
        video=CLAIMS_VIDEO_NOTE if kind == "video" else "",
        claims="\n".join(f"{c['id']}: {c['text']}" for c in asked),
        keys=", ".join(f'"{c["id"]}": "..."' for c in asked),
        world_q=CLAIMS_WORLD_Q if setting else "",
        world_keys=VERIFY_WORLD_KEYS if setting else "")


def parse_claims_answer(text, ids, with_world):
    """{"claims": {id: yes|no|unsure}, "medium", [мир], "why"} или None —
    хоть один пункт не разобран (угадывать ответ за модель нельзя)."""
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except ValueError:
        return None
    got = j.get("claims")
    if not isinstance(got, dict):
        return None
    claims = {}
    for cid in ids:
        v = str(got.get(cid, "")).strip().lower()
        if v not in CLAIM_ANSWERS:
            return None
        claims[cid] = v
    medium = str(j.get("medium", "")).strip().lower()
    if medium not in MEDIUMS:
        return None
    out = {"claims": claims, "medium": medium}
    if with_world:
        for key in ("main_in_world", "background_foreign"):
            if not isinstance(j.get(key), bool):
                return None
            out[key] = j[key]
    out["why"] = str(j.get("why") or "")[:300]
    return out


def claim_values(spec, votes, kind):
    """{id: 0..2} — среднее по голосам; утверждение, которое этот вид кадра
    не может выполнить (движение у фото), — 0."""
    out = {}
    for c in spec["claims"]:
        vals = [CLAIM_ANSWERS[v["claims"][c["id"]]] for v in votes if c["id"] in v["claims"]]
        out[c["id"]] = sum(vals) / len(vals) if vals else 0.0
    return out


def claims_vector(spec, votes, kind):
    """Ключ сравнения кадров по спецификации; каждый элемент 0..1, больше —
    лучше, первый — фокус (первое утверждение спецификации всегда must).
    None — отказ
    (главный предмет не из мира или кадр — 3D/мультфильм хоть в одном
    голосе). Порядок: must-утверждения в порядке спецификации, затем чистота
    фона (чужое только на фоне — штраф, не отказ), затем should. Порядок
    утверждений задаёт спецификация, код его не меняет."""
    if not votes:
        return None
    if any(v.get("medium") == "cg" or v.get("main_in_world") is False for v in votes):
        return None
    top = CLAIM_ANSWERS["yes"]
    vals = claim_values(spec, votes, kind)
    musts = [vals[c["id"]] / top for c in spec["claims"] if c["tier"] == "must"]
    shoulds = [vals[c["id"]] / top for c in spec["claims"] if c["tier"] != "must"]
    clean = sum(0 if v.get("background_foreign") else 1 for v in votes) / len(votes)
    return tuple(musts) + (clean,) + tuple(shoulds)


def focus_met(spec, votes, kind):
    """Кадр показывает фокус: первое утверждение выполнено единогласно."""
    if not votes:
        return False
    return claim_values(spec, votes, kind)[spec["claims"][0]["id"]] >= CLAIM_ANSWERS["yes"]


def nothing_met(spec, votes, kind):
    """Кадр не показывает из спецификации НИЧЕГО обязательного: каждое
    must-утверждение — единогласное «нет». Это брак; кадр, который не
    показал главное, но показал обязательную деталь фразы, — замена,
    а не брак (он проигрывает любому кадру с главным, но лучше соседнего
    кадра, растянутого на чужую фразу)."""
    if not votes:
        return False
    vals = claim_values(spec, votes, kind)
    return all(vals[c["id"]] == 0 for c in spec["claims"] if c["tier"] == "must")


def all_met(spec, votes, kind):
    """Кадр выполняет ВСЁ, что спросила спецификация, и фон чистый — лучше
    искать незачем."""
    if claims_vector(spec, votes, kind) is None:
        return False
    return (all(v >= CLAIM_ANSWERS["yes"] for v in claim_values(spec, votes, kind).values())
            and not any(v.get("background_foreign") for v in votes))


def needs_second_vote(spec, answers, kind):
    """Второй голос нужен, когда исход решает сомнение: хоть одно
    must-утверждение этого вида кадра — «unsure»."""
    if answers is None:
        return False
    return any(c["tier"] == "must" and answers["claims"].get(c["id"]) == "unsure"
               for c in asked_claims(spec, kind))


def _image_content(path, max_side):
    from PIL import Image
    with Image.open(path) as im:
        im = flat_rgb(im)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def verify_claims(gateway, model, *, phrase, spec, setting, path, kind="photo", cache_dir=None,
                  max_side=VERIFY_MAX_SIDE, reasoning=None, caption=None, vote=1):
    """(ответы | None, {"cost", "call", "cache_hit", "refused"}). None —
    проверки не было (нет шлюза, сбой, неразобранный ответ). vote — номер
    голоса: у второго голоса свой ключ кэша, иначе он был бы копией первого."""
    if gateway is None or not path or not os.path.exists(path):
        return None, {}
    asked = asked_claims(spec, kind)
    if not asked:
        return None, {}
    text = claims_question(phrase, spec, setting, kind, caption)
    h = hashlib.sha256()
    for part in ("claims", str(CLAIMS_VERSION), model, text, str(max_side), repr(reasoning),
                 str(vote), _file_digest(path)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    cp = os.path.join(cache_dir, "claims_" + h.hexdigest() + ".json") if cache_dir else None
    if cp and os.path.exists(cp):
        try:
            return json.load(open(cp, encoding="utf-8"))["answers"], {"cache_hit": True}
        except Exception:
            pass
    try:
        image = _image_content(path, max_side)
    except Exception:
        return None, {}
    try:
        answer, _u, price = gateway.chat(model, [{"type": "text", "text": text}, image], 500, 1200,
                                         reasoning=reasoning)
    except Exception as e:  # noqa: BLE001 — сбой шлюза: проверки не было
        return None, {"refused": f"{type(e).__name__}: {e}"[:200]}
    answers = parse_claims_answer(answer, [c["id"] for c in asked], bool(setting))
    if answers is None:
        return None, {"refused": "неразобранный ответ: " + (answer or "")[-200:], "cost": price,
                      "call": True}
    if cp:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cp}.{os.getpid()}.{threading.get_ident()}.part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"answers": answers, "model": model}, f, ensure_ascii=False)
        os.replace(tmp, cp)
    return answers, {"cost": price, "call": True}
