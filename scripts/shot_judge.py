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
# Рассуждение модели в сетке и в проверке зрения ВЫКЛЮЧЕНО явно. Оценки
# сетки (57/61 пар) сняты, когда Qwen 3.7 Plus на шлюзе по умолчанию не
# рассуждал; 24.09 провайдер включил рассуждение по умолчанию, и проверка
# зрения (20 токенов выхода) стала отдавать пустой ответ — судья молча
# выключился на весь прогон judge10. Явный выключатель возвращает ту
# конфигурацию, на которой всё замерено, и не зависит от чужого дефолта.
GRID_REASONING = False

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
    кладётся на светлый фон, как такие снимки и задуманы.

    И в той ориентации, в какой кадр увидит зритель: ffmpeg 6.x поворачивает
    JPEG по метке EXIF (проверено 24.09: 400x200 с Orientation=6 выходит
    200x400), а PIL отдаёт сырые пиксели — судья смотрел бы на кадр боком, и
    рамка детали (locate_box) легла бы не туда при вырезке."""
    try:
        from PIL import ImageOps
        im = ImageOps.exif_transpose(im)
    except Exception:  # noqa: BLE001 — битые EXIF: кадр как есть
        pass
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        from PIL import Image
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (235, 235, 235))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return im.convert("RGB")


def _grid_bytes(paths, kind="photo", cols=None, tile=None):
    from PIL import Image, ImageDraw, ImageFont
    base_cols, base_tile, _max, _q = LAYOUTS[kind]
    cols, tile = cols or base_cols, tile or base_tile
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
    """{номер 1..n: целое 0..3} или None, если хоть один номер без оценки.

    Оценка строкой («"2"») — та же оценка: живой случай judge14 (24.09),
    сетка слота 2 ответила {"1": "2", "2": "2"}, и полный ответ
    считался неполным — слот терял оценки сетки целиком."""
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
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
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


def _cache_write(cp, payload, readable=False):
    """Запись в кэш — по возможности. Не записалось (кончилось место на
    диске, 24.09: замер упал с OSError посреди прогона) — ответ модели уже
    получен и оплачен, работа продолжается, повторный прогон просто
    спросит заново. Недописанный файл удаляется: битый кэш хуже пустого."""
    tmp = f"{cp}.{os.getpid()}.{threading.get_ident()}.part"
    try:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=not readable)
        os.replace(tmp, cp)
        return True
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False

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
            answer, _u, _p = gateway.chat(model, content, 20, 400, reasoning=False)
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
        answer, _usage, price = gateway.chat(model, content, MAX_TOKENS, EST_PROMPT_TOKENS,
                                             reasoning=GRID_REASONING)
    except Exception as e:  # noqa: BLE001 — любой сбой шлюза: судьи нет
        return None, {"refused": f"{type(e).__name__}: {e}"[:300]}
    scores = parse_scores(answer, len(chunk))
    if scores is None:
        return None, {"refused": "неполный ответ: " + answer[-200:], "cost": price, "call": True}
    if cp:
        _cache_write(cp, {"model": model, "version": PROMPT_VERSION, "scores": scores}, readable=False)
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
# Утверждение с движением (motion) может выполнить только ролик, показанный
# хотя бы двумя кадрами; у фото и у ролика с одним превью (Pixabay) его не
# спрашивают и ставят «нет».
#
# Второго голоса нет сознательно: тот же вопрос, та же картинка, температура
# 0 — платная копия первого ответа. «Сомневаюсь» остаётся половиной балла.
CLAIMS_VERSION = 2
CLAIM_ANSWERS = {"yes": 2, "unsure": 1, "no": 0}
CLAIMS_PROMPT = """You check one shot for a documentary video.
Narration line: «{phrase}»
What the viewer must see: «{focus}»{world}{caption}{video}
Look at the picture carefully. For each statement answer "yes", "no" or "unsure" — about THIS picture only:
{claims}
Then:
- medium: "photo", "artwork" (painting, drawing, engraving, manuscript), "object" (museum object on a plain background) or "cg" (3D render, cartoon, toy, video game, infographic){world_q}
Reply with JSON only: {{"claims": {{{keys}}}, "medium": "..."{world_keys}, "why": "<short>"}}"""
CLAIMS_VIDEO_NOTE = ("\nThe picture shows {n} frames of ONE video clip in time order; judge the clip, "
                     "a movement counts if the frames show it happening.")
CLAIMS_WORLD_Q = """
- main_in_world: could the MAIN subject exist in that world (era, culture)? true/false
- background_foreign: is there anything ELSE in the picture (background, edges, people around) that could not exist in that world — modern people, clothing, objects, vehicles, signs, spectators? true/false"""


# МИР — ОДИН РАЗ НА КАРТИНКУ, БЕЗ ФРАЗЫ (вариант замера 24.09). Вопрос про
# мир внутри проверки по утверждениям зависит от фразы слота, и одна и та же
# картинка в соседних слотах получала разные ответы: из 39 кадров, стоящих
# в пулах двух и более слотов эпизода 94, у 6 ответ о мире расходился
# (pixabay:321443 — отказ в слоте 4, «свой» в слоте 6). Здесь вопрос о мире
# не знает фразы, поэтому кэшируется по картинке и один на все слоты.
WORLD_ONLY_VERSION = 1
WORLD_ONLY_PROMPT = """You check one picture for a documentary video.
The episode's world: {setting}.{caption}{video}
Look at the picture carefully and answer:
- main_in_world: could the MAIN subject of the picture exist in that world (era, culture)? true/false
- background_foreign: is there anything ELSE in the picture (background, edges, people around) that could not exist in that world — modern people, clothing, objects, vehicles, signs, spectators? true/false
Reply with JSON only: {{"main_in_world": true/false, "background_foreign": true/false, "why": "<short>"}}"""


def world_only_question(setting, kind="photo", caption=None, frames=None):
    caption = " ".join(str(caption or "").split())[:VERIFY_CAPTION_MAX]
    return WORLD_ONLY_PROMPT.format(
        setting=setting, caption=VERIFY_CAPTION.format(caption=caption) if caption else "",
        video=CLAIMS_VIDEO_NOTE.format(n=frames or 3) if shows_motion(kind, frames) else "")


def world_of_image(gateway, model, *, setting, path, kind="photo", cache_dir=None,
                   max_side=VERIFY_MAX_SIDE, reasoning=None, caption=None, frames=None):
    """({"main_in_world", "background_foreign"} | None, info). Кэш — по
    картинке, миру и подписи: фраза в вопрос не входит."""
    if gateway is None or not setting or not path or not os.path.exists(path):
        return None, {}
    text = world_only_question(setting, kind, caption, frames)
    h = hashlib.sha256()
    for part in ("world_only", str(WORLD_ONLY_VERSION), model, text, str(max_side), repr(reasoning),
                 _file_digest(path)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    cp = os.path.join(cache_dir, "worldonly_" + h.hexdigest() + ".json") if cache_dir else None
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
        answer, _u, price = gateway.chat(model, [{"type": "text", "text": text}, image], 300, 900,
                                         reasoning=reasoning)
    except Exception as e:  # noqa: BLE001 — сбой шлюза: проверки не было
        return None, {"refused": f"{type(e).__name__}: {e}"[:200]}
    import re
    m = re.search(r"\{.*\}", answer or "", re.S)
    try:
        j = json.loads(m.group(0)) if m else None
    except ValueError:
        j = None
    if not isinstance(j, dict) or not all(isinstance(j.get(k), bool)
                                          for k in ("main_in_world", "background_foreign")):
        return None, {"refused": "неразобранный ответ: " + (answer or "")[-200:], "cost": price,
                      "call": True}
    answers = {"main_in_world": j["main_in_world"], "background_foreign": j["background_foreign"]}
    if cp:
        _cache_write(cp, {"answers": answers, "model": model}, readable=True)
    return answers, {"cost": price, "call": True}



# ДЕФЕКТЫ РИСУНКА — ПО ПОЛНОМУ КАДРУ, С РЕФЕРЕНСОМ ГЕРОЯ (09.10). Сетка
# (400x300 на плитку) сравнивает варианты и не видит мелкого: записанный
# промах — «лишняя чёрная палочка под рукой — судья не поймал». Проверка по
# утверждениям отвечает «есть ли предмет», а не «цел ли рисунок». Арт-разбор
# эп.01 (docs/quality/compare_0810/report_art.md) назвал главный класс брака
# рисованного канала: герой не на модели (оба уха торчком в К2, вместо кота —
# серое существо в двух вариантах), и этот класс не ловил ни один вопрос.
# Здесь модель смотрит кадр в полный размер (DEFECTS_MAX_SIDE) рядом с
# референсом героя и отвечает про ошибки рисунка и про соответствие герою.
# Решение принимает код (defect_verdict): «reject» только за то, что зритель
# прочтёт как сломанную картинку или чужого персонажа; остальное — «minor»,
# пишется в отчёт и опускает вариант в ранжировании, кадр не теряется.
DEFECTS_VERSION = 1
DEFECTS_MAX_SIDE = 1264
DEFECTS_REF_SIDE = 640
DEFECTS_PROMPT = """You inspect ONE hand-drawn illustration for a children's explainer video before it goes on screen.
{reference}Look at the illustration carefully at full size and answer:
1. defects: drawing errors a viewer would notice — extra or missing limbs, paws, ears, eyes or tails; two heads or two of the main character; body parts merged into objects or into each other; an object fused with another; a stray stroke, stick, blob or mark that belongs to nothing; a half-drawn object; a floating detached part. Describe each briefly with where it is. Empty list if none. Do not list style choices, simplifications or texture hatching.
2. severe: true if at least one defect would make a viewer think the picture is broken (wrong number of body parts, merged bodies, duplicated character, detached floating part); false if all listed defects are small marks.
3. hero_present: is the main character in the picture at all? true/false{hero_q}
6. text: true if any letters, digits or words are drawn anywhere in the picture, false otherwise.
Reply with JSON only: {{"defects": [...], "severe": true/false, "hero_present": true/false, "off_model": [...], "wrong_character": true/false, "text": true/false, "why": "<short>"}}"""
DEFECTS_REFERENCE = ("The FIRST image is the reference of the main character: {hero}. {marks}\n"
                     "The SECOND image is the illustration to inspect.\n")
DEFECTS_NO_REFERENCE = "The image is the illustration to inspect; the main character is {hero}.\n"
DEFECTS_HERO_Q = """
4. off_model: if the main character is present, every way it differs from the reference that a viewer would notice: different species or colour, wrong ears (see the reference), wrong eye colour, missing tail flame or ember, a different character drawn instead. Empty list if it matches.
5. wrong_character: true if the figure in the picture is a different character than the reference (another animal, a ghost, a human, a different colour), false otherwise or if no character."""
DEFECTS_HERO_Q_NO_REF = """
4. off_model: empty list.
5. wrong_character: false."""


def defects_question(hero_text=None, hero_marks=None, with_reference=True):
    hero = hero_text or "the main character"
    marks = " ".join(str(hero_marks or "").split())
    if with_reference:
        ref = DEFECTS_REFERENCE.format(hero=hero, marks=marks).replace(". \n", ".\n")
        return DEFECTS_PROMPT.format(reference=ref, hero_q=DEFECTS_HERO_Q)
    return DEFECTS_PROMPT.format(reference=DEFECTS_NO_REFERENCE.format(hero=hero), hero_q=DEFECTS_HERO_Q_NO_REF)


def parse_defects_answer(text):
    """Словарь ответа с проверенными типами, иначе None."""
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        j = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    if not isinstance(j, dict) or not isinstance(j.get("severe"), bool):
        return None

    def strs(v):
        return [" ".join(str(x).split())[:200] for x in v if str(x).strip()] if isinstance(v, list) else []
    return {"defects": strs(j.get("defects")), "severe": j["severe"],
            "hero_present": bool(j.get("hero_present")), "off_model": strs(j.get("off_model")),
            "wrong_character": bool(j.get("wrong_character")), "text": bool(j.get("text")),
            "why": " ".join(str(j.get("why") or "").split())[:200]}


def drawing_defects(gateway, model, *, path, hero_ref=None, hero_text=None, hero_marks=None,
                    cache_dir=None, max_side=DEFECTS_MAX_SIDE, reasoning=False):
    """(ответ parse_defects_answer | None, info). None — проверки не было
    (нет шлюза, сбой, неразобранный ответ): это не брак. Кэш — по кадру,
    референсу и тексту вопроса."""
    if gateway is None or not path or not os.path.exists(path):
        return None, {}
    ref_ok = bool(hero_ref) and os.path.exists(hero_ref)
    text = defects_question(hero_text, hero_marks, with_reference=ref_ok)
    h = hashlib.sha256()
    for part in ("defects", str(DEFECTS_VERSION), model, text, str(max_side), repr(reasoning),
                 _file_digest(path), _file_digest(hero_ref) if ref_ok else ""):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    cp = os.path.join(cache_dir, "defects_" + h.hexdigest() + ".json") if cache_dir else None
    if cp and os.path.exists(cp):
        try:
            return json.load(open(cp, encoding="utf-8"))["answers"], {"cache_hit": True}
        except Exception:
            pass
    try:
        content = [{"type": "text", "text": text}]
        if ref_ok:
            content.append(_image_content(hero_ref, DEFECTS_REF_SIDE))
        content.append(_image_content(path, max_side))
    except Exception:
        return None, {}
    try:
        answer, _u, price = gateway.chat(model, content, 600, 3000, reasoning=reasoning)
    except Exception as e:  # noqa: BLE001 — сбой шлюза: проверки не было
        return None, {"refused": f"{type(e).__name__}: {e}"[:200]}
    answers = parse_defects_answer(answer)
    if answers is None:
        return None, {"refused": "неразобранный ответ: " + (answer or "")[-200:], "cost": price,
                      "call": True}
    if cp:
        _cache_write(cp, {"answers": answers, "model": model}, readable=True)
    return answers, {"cost": price, "call": True}


def defect_verdict(answers, expect_hero=False):
    """"reject" — вместо героя нарисован чужой персонаж; "minor" — есть
    замечания (палочка, слитые предметы, герой не на модели): кадр годен,
    но проигрывает чистому и печатается для глаз; "clean" — замечаний нет;
    None — проверки не было. expect_hero: кадр по плану с героем — его
    отсутствие не брак здесь (это решает проверка по утверждениям).
    «severe» модели — НЕ отказ: живой замер 09.10 (эп.01, вариант «кот держит
    мозг с камнем») дал severe=true за «цепь сливается с лапой» — спорное
    место, не сломанный рисунок; при одном варианте на кадр ложный отказ
    стоил бы кадра целиком. Отказ только за то, что зритель читает
    однозначно и что модель называла верно: другой персонаж вместо героя."""
    if answers is None:
        return None
    if answers["wrong_character"]:
        return "reject"
    if answers["severe"] or answers["defects"] or (expect_hero and answers["off_model"]):
        return "minor"
    return "clean"


def shows_motion(kind, frames=None):
    """Может ли кадр показать движение: ролик минимум из двух кадров."""
    return kind == "video" and (frames is None or frames >= 2)


SUBJECT_ID = "subject"


def subject_claim(spec):
    """Вопрос «виден ли сам предмет фразы» — простой, без действия и места,
    или None (у спецификации нет предмета). Его ответ не входит в вектор
    сравнения: он решает только «кадр про эту фразу вообще или нет»."""
    subject = (spec or {}).get("subject")
    if not subject:
        return None
    return {"id": SUBJECT_ID, "text": f"{subject} is visible", "tier": "subject"}


def asked_claims(spec, kind, frames=None):
    """Утверждения, которые спрашиваются у кадра: движение — только у ролика,
    показанного хотя бы двумя кадрами; вопрос о предмете фразы — последним,
    если у спецификации он есть."""
    moving = shows_motion(kind, frames)
    asked = [c for c in spec["claims"] if moving or not c.get("motion")]
    sc = subject_claim(spec)
    return asked + ([sc] if sc else [])


def claims_question(phrase, spec, setting=None, kind="photo", caption=None, frames=None):
    asked = asked_claims(spec, kind, frames)
    caption = " ".join(str(caption or "").split())[:VERIFY_CAPTION_MAX]
    return CLAIMS_PROMPT.format(
        phrase=phrase or "—", focus=spec.get("focus") or "—",
        world=VERIFY_WORLD.format(setting=setting) if setting else "",
        caption=VERIFY_CAPTION.format(caption=caption) if caption else "",
        video=CLAIMS_VIDEO_NOTE.format(n=frames or 3) if shows_motion(kind, frames) else "",
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


def claim_values(spec, answers):
    """{id: 0..1}: да — 1, сомневаюсь — 0.5, нет или не спрашивали (движение
    у кадра, который его показать не может) — 0."""
    top = CLAIM_ANSWERS["yes"]
    got = (answers or {}).get("claims") or {}
    return {c["id"]: CLAIM_ANSWERS[got[c["id"]]] / top if c["id"] in got else 0.0
            for c in spec["claims"]}


def claims_vector(spec, answers, *, world_veto=True, cg_veto=True):
    """Ключ сравнения кадров по спецификации; больше — лучше, каждый элемент
    0..1. None — отказ.

    Порядок: мир (главный предмет из мира фразы), must-утверждения в порядке
    спецификации (первое — главное), чистота фона, should-утверждения. Порядок
    утверждений задаёт спецификация, код его не меняет.

    world_veto — главный предмет чужого мира это отказ; выключается
    предохранителем прогона (pipeline_smart.world_veto_active), тогда это
    штраф: первый элемент 0. cg_veto — 3D/мультфильм/инфографика это отказ;
    только для исторического мира (у научной ниши рендер бывает
    единственным изображением).

    Чужое на фоне (зрители в футболках, бетонная стена, человек в куртке)
    в историческом мире — тоже отказ, а не штраф (25.09): штрафом такой
    кадр побеждал там, где у остальных не выполнено утверждение, — так в
    judge12 встала марокканская тбурида. Прецедент канала тот же: кадр #001
    золотого набора (реконструкторы на фоне современной толпы) — брак
    modern_intrusion. На сохранённых ответах эп.94 все кадры с этой
    пометкой, просмотренные глазами, современное содержат. Под
    предохранителем мира — снова штраф: при неверном паспорте «чужое»
    может означать не современность, а другую эпоху."""
    if answers is None:
        return None
    if cg_veto and answers.get("medium") == "cg":
        return None
    foreign = answers.get("main_in_world") is False
    if foreign and world_veto:
        return None
    if cg_veto and world_veto and answers.get("background_foreign"):
        return None
    vals = claim_values(spec, answers)
    musts = [vals[c["id"]] for c in spec["claims"] if c["tier"] == "must"]
    shoulds = [vals[c["id"]] for c in spec["claims"] if c["tier"] != "must"]
    clean = 0.0 if answers.get("background_foreign") else 1.0
    return (0.0 if foreign else 1.0,) + tuple(musts) + (clean,) + tuple(shoulds)


def must_failed(spec, answers, skip=("nofig",)):
    """Обязательные утверждения спецификации с ЯВНЫМ «нет» (не «сомневаюсь»). Для СГЕНЕРИРОВАННОГО
    кадра это брак раунда: модель можно попросить ещё раз, и цена ошибки — закрытый конверт на фразе
    «только открыть» (эп.01 08.10). Для стокового кадра то же правило выбрасывало годные замены
    (см. nothing_met) — там оно не применяется. skip — утверждения с собственной веткой (nofig)."""
    got = (answers or {}).get("claims") or {}
    return [c["id"] for c in spec["claims"] if c.get("tier") == "must" and c.get("id") not in skip
            and got.get(c["id"]) == "no"]


def nothing_met(spec, answers):
    """Кадр не показывает из спецификации НИЧЕГО обязательного: каждое
    must-утверждение — «нет». Это брак — кроме случая, когда на прямой
    вопрос «виден ли предмет фразы» ответ «да»: тогда кадр — замена.

    Утверждения составные («рыцарь падает в грязь»), и годная замена —
    рыцарь, который стоит, — не выполняет ни одного; без предмета такой
    кадр считался браком и слот пустел, хотя замена была среди первых
    кадров (эп.94: 3-4 слота из 9). Замер по экрану 26.09, эп.94 (4
    прогона): брак на экране 8 -> 8, лучший кадр 12 -> 22, пустых слотов
    13 -> 3; эп.93 — ничья, минусов нет.

    Обратная половина («предмет не виден — брак») НЕ действует: замер
    24-26.09 — она выбрасывает годные кадры другого вида (дага на
    «рондельный кинжал»). «Сомневаюсь» — решают утверждения."""
    vals = claim_values(spec, answers)
    if subject_claim(spec):
        seen = ((answers or {}).get("claims") or {}).get(SUBJECT_ID)
        if seen == "yes":
            # Предмет фразы в кадре — кадр про эту фразу, даже если её
            # действие и место не показаны: замена, а не брак (кинжал без
            # ладони на фразу про вес кинжала).
            return False
    return all(vals[c["id"]] == 0 for c in spec["claims"] if c["tier"] == "must")


def shows_nothing(spec, answers, grid=None):
    """Проверка по пунктам считает кадр браком: не выполнено ничего
    обязательного (nothing_met), либо сетка того же судьи поставила 0
    («не по теме») — тогда одобрению по пунктам верить нельзя (эп.95: часы
    на 10:07 прошли как «часы у полуночи»). Одно правило для рендера и для
    бенча: две копии уже расходились — бенч не знал про сетку 0."""
    return nothing_met(spec, answers) or grid == 0


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
                  max_side=VERIFY_MAX_SIDE, reasoning=None, caption=None, frames=None,
                  world_separate=False):
    """(ответы | None, {"cost", "call", "cache_hit", "refused"}). None —
    проверки не было (нет шлюза, сбой, неразобранный ответ).

    world_separate — мир спрашивается отдельным вопросом без фразы
    (world_of_image, один на картинку), а не внутри вопроса по утверждениям."""
    if gateway is None or not path or not os.path.exists(path):
        return None, {}
    if world_separate and setting:
        # Два вопроса не зависят друг от друга (мир — без фразы, пункты — без
        # мира), поэтому идут ОДНОВРЕМЕННО: раньше пункты ждали ответа о мире,
        # и каждая порция проверки стоила два хода к модели подряд вместо
        # одного. Вопросы, картинки и кэши те же, ответы те же. Цена: если
        # вопрос о мире не удался, ответ по пунктам уже оплачен — он ложится
        # в кэш и берётся при следующем вопросе о том же кадре.
        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            wf = ex.submit(world_of_image, gateway, model, setting=setting, path=path, kind=kind,
                           cache_dir=cache_dir, max_side=max_side, reasoning=reasoning,
                           caption=caption, frames=frames)
            cf = ex.submit(verify_claims, gateway, model, phrase=phrase, spec=spec, setting=None,
                           path=path, kind=kind, cache_dir=cache_dir,
                           max_side=max_side, reasoning=reasoning, caption=caption,
                           frames=frames)
            world, winfo = wf.result()
            answers, info = cf.result()
        if world is None:
            winfo = dict(winfo, cost=(winfo.get("cost") or 0) + (info.get("cost") or 0),
                         call=bool(winfo.get("call") or info.get("call")))
            return None, winfo
        cost = (info.get("cost") or 0) + (winfo.get("cost") or 0)
        info = dict(info, cost=cost, call=bool(info.get("call") or winfo.get("call")))
        return (None if answers is None else dict(answers, **world)), info
    asked = asked_claims(spec, kind, frames)
    if not asked:
        return None, {}
    text = claims_question(phrase, spec, setting, kind, caption, frames)
    h = hashlib.sha256()
    for part in ("claims", str(CLAIMS_VERSION), model, text, str(max_side), repr(reasoning),
                 _file_digest(path)):
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
        _cache_write(cp, {"answers": answers, "model": model}, readable=True)
    return answers, {"cost": price, "call": True}
