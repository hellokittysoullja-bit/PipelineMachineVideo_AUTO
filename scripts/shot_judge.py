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
паспорта (world_card.judge_setting) — наоборот: брак чужой эпохи и
культуры, одобренный Qwen, 10 -> 4 кадров из 36, верных пар 93 -> 116 из
201 (см. question()). Без строки судья не знает эпохи, если описание кадра
её не называет («a narrow rondel dagger blade» — современный нож проходил).

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
    описании кадра. Замер 23.09 (эпизод 94, 36 кадров, разметка автора
    правки): Qwen 3.7 Plus одобрял брак чужой эпохи и культуры в 10 кадрах
    из 36 (марокканское конное шоу — 3, современный нож на «кинжал» — 3),
    со строкой мира — в 4; верных попарных сравнений 93 -> 116 из 201.
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
