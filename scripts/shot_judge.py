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
        _cache_write(cp, {"ok": ok, "why": why, "model": model}, readable=True)
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


# РАМКА СМЫСЛОВОЙ ДЕТАЛИ — где на кадре то, о чём фраза. Нужна рисункам и
# страницам рукописей: миниатюра битвы — это страница с текстом и полями, а
# фраза — про одну сцену на ней. Документалисты показывают деталь медленным
# наездом, а не страницу целиком; рендер (pipeline_smart.focus_crop) вырезает
# область вокруг рамки (focus_frame.crop_region). Спрашивается ОДИН раз у
# одобренного победителя, и вырезку судья потом смотрит сам (confirm_crop):
# на странице бывает несколько сцен, и рамка может лечь не на ту.
#
# Формат — шкала 0-1000, родная для моделей Qwen: замер 24.09 на восьми
# случаях с известной рамкой (страницы Азенкура, Фиоре, Тальхоффера,
# миниатюра Азенкура) — при вопросе «доли кадра» модель половину ответов всё
# равно дала в тысячных. Поле «what» (что в рамке, до самой рамки) — замер
# того же дня: без него 7 из 8 рамок легли на нужную сцену (на странице
# Фиоре — нижний рисунок вместо верхнего), с ним — 8 из 8.
FOCUS_BOX_VERSION = 2
FOCUS_BOX_PROMPT = """Look at this picture. Find the part of it that shows: «{focus}».
Reply with JSON only: {{"what": "what is inside the box, in a few words", "box": [x0, y0, x1, y1]}} — coordinates on a 0-1000 scale (0 = left or top edge of the picture, 1000 = right or bottom edge), the tightest box that still contains all of it.
Reply {{"what": "", "box": null}} if it fills most of the picture or is not in the picture."""
FOCUS_CONFIRM_PROMPT = """Does this picture clearly show: «{focus}»?
Reply with JSON only: {{"shows": "yes"}} or {{"shows": "no"}}."""
FOCUS_BOX_MIN_AREA = 0.002     # меньше — точка, а не деталь: скорее ошибка разметки
FOCUS_BOX_MAX_AREA = 0.70      # больше — и так почти весь кадр, вырезать незачем


def parse_box(answer):
    """[x0, y0, x1, y1] долями кадра или None: без рамки, неразборчиво,
    рамка вне кадра, точка или почти весь кадр. Шкала 0-1000 (о ней
    просит вопрос) и доли (так модель тоже иногда отвечает) — обе."""
    import re
    m = re.search(r"\{.*\}", answer or "", re.S)
    try:
        j = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    box = j.get("box") if isinstance(j, dict) else None
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if max(vals) > 1.0:
        if max(vals) > 1000.0:
            return None
        vals = [v / 1000.0 for v in vals]
    x0, y0, x1, y1 = vals
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return None
    area = (x1 - x0) * (y1 - y0)
    if not FOCUS_BOX_MIN_AREA <= area <= FOCUS_BOX_MAX_AREA:
        return None
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def _ask_image(gateway, model, *, kind, text, path, cache_dir, max_side, reasoning, max_tokens):
    """(ответ | None, info): вопрос по картинке с кэшем по тексту вопроса,
    картинке и модели. Сбой шлюза — None."""
    h = hashlib.sha256()
    for part in (kind, model, text, str(max_side), repr(reasoning), _file_digest(path)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    cp = os.path.join(cache_dir, f"{kind}_{h.hexdigest()}.json") if cache_dir else None
    if cp and os.path.exists(cp):
        try:
            return json.load(open(cp, encoding="utf-8"))["answer"], {"cache_hit": True}
        except Exception:
            pass
    try:
        image = _image_content(path, max_side)
        answer, _u, price = gateway.chat(model, [{"type": "text", "text": text}, image],
                                         max_tokens, 900, reasoning=reasoning)
    except Exception as e:  # noqa: BLE001 — ответа нет: кадр идёт как раньше
        return None, {"refused": f"{type(e).__name__}: {e}"[:200]}
    if cp and answer:
        _cache_write(cp, {"answer": answer, "model": model}, readable=True)
    return answer, {"cost": price, "call": True}


def locate_box(gateway, model, *, path, focus, cache_dir=None, max_side=1024, reasoning=False):
    """(рамка | None, info). info["what"] — что модель увидела в рамке."""
    if gateway is None or not path or not os.path.exists(path) or not focus:
        return None, {}
    text = FOCUS_BOX_PROMPT.format(focus=" ".join(str(focus).split())[:200])
    answer, info = _ask_image(gateway, model, kind=f"focusbox{FOCUS_BOX_VERSION}", text=text,
                              path=path, cache_dir=cache_dir, max_side=max_side,
                              reasoning=reasoning, max_tokens=160)
    if answer is None:
        return None, info
    try:
        what = json.loads(answer[answer.index("{"): answer.rindex("}") + 1]).get("what")
    except (ValueError, AttributeError):
        what = None
    return parse_box(answer), dict(info, what=(str(what)[:120] if what else None))


def parse_shows(answer):
    """True / False / None (неразборчиво) из ответа на FOCUS_CONFIRM_PROMPT."""
    try:
        j = json.loads(answer[answer.index("{"): answer.rindex("}") + 1])
    except (ValueError, AttributeError, TypeError):
        return None
    v = str((j or {}).get("shows") or "").strip().lower() if isinstance(j, dict) else ""
    return True if v == "yes" else False if v == "no" else None


def confirm_crop(gateway, model, *, path, focus, cache_dir=None, max_side=1024, reasoning=False):
    """(видно ли на вырезке то, о чём фраза: True/False/None, info). None —
    вопроса не было или ответ неразборчив: вызывающий вырезку НЕ ставит."""
    if gateway is None or not path or not os.path.exists(path) or not focus:
        return None, {}
    text = FOCUS_CONFIRM_PROMPT.format(focus=" ".join(str(focus).split())[:200])
    answer, info = _ask_image(gateway, model, kind=f"focusok{FOCUS_BOX_VERSION}", text=text,
                              path=path, cache_dir=cache_dir, max_side=max_side,
                              reasoning=reasoning, max_tokens=40)
    return (None if answer is None else parse_shows(answer)), info


# ВЫБОР СРЕДИ РАВНЫХ ПО СМЫСЛУ. Проверка по утверждениям и сетка судьи
# часто оставляют ничью наверху: у judge12/13 эп.94 в 6-8 из 16-17 пар
# «слот, вид» лучший вектор утверждений был у нескольких кадров сразу, а у
# 2 из 7 фото-слотов judge13 — и при равной оценке сетки. Ничью дальше
# решали релевантность эмбеддинга, ритм крупностей и эстетика LAION —
# и LAION на кинематографичность не отвечает вообще: замер 24.09 на 133
# финалистах judge12/13, размеченных глазами Claude (сильный / простой /
# слабый кадр), — 0.510 верных пар, уровень монетки. Тот же судья, которому
# показали равных СЕТКОЙ и попросили упорядочить как кадры фильма, — 0.846
# верных пар (188 пар, 10 групп); та же разметка по одному кадру за раз —
# 0.68-0.74. Сравнение соседей надёжнее абсолютной оценки, поэтому вопрос —
# порядок, а не баллы. Спрашивается только при ничьей наверху, один раз на
# слот и вид.
LOOK_VERSION = 1
LOOK_MAX = 9
LOOK_TILE = (480, 320)
LOOK_PROMPT = """These {k} numbered pictures are candidates for the same shot of a documentary film; all of them show the right subject. Rank them from the best to the worst AS A FILM SHOT — light, composition, a clear subject, nothing distracting in the frame (onlookers, cars, signs, clutter), not an amateur snapshot. Do not judge what they show.
Reply with JSON only: {{"order": [numbers from best to worst]}}"""
LOOK_PROMPT_VIDEO = """These {k} numbered rows are candidate video clips for the same shot of a documentary film (each row shows three frames of one clip); all of them show the right subject. Rank the clips from the best to the worst AS A FILM SHOT — light, composition, a clear subject, nothing distracting in the frame (onlookers, cars, signs, clutter), not an amateur video. Do not judge what they show.
Reply with JSON only: {{"order": [numbers from best to worst]}}"""


def parse_order(answer, k):
    """Порядок 0..k-1 (лучший первым) из ответа судьи или None. Номера вне
    сетки и повторы отбрасываются; пропущенные встают в конец в исходном
    порядке — судья мог не назвать самые слабые. Ни одного номера — None."""
    import re
    m = re.search(r"\[[^\]]*\]", answer or "")
    if not m:
        return None
    out = []
    for tok in re.findall(r"\d+", m.group(0)):
        n = int(tok) - 1
        if 0 <= n < k and n not in out:
            out.append(n)
    if not out:
        return None
    return out + [n for n in range(k) if n not in out]


def rank_look(gateway, model, *, paths, kind="photo", cache_dir=None):
    """(порядок 0..k-1, info) — кадры-равные по смыслу, упорядоченные как
    кадры фильма; None — вопроса не было или ответ неразборчив."""
    paths = [p for p in paths if p and os.path.exists(p)][:LOOK_MAX]
    if gateway is None or len(paths) < 2:
        return None, {}
    text = (LOOK_PROMPT_VIDEO if kind == "video" else LOOK_PROMPT).format(k=len(paths))
    # Фото — плитки крупнее, чем у сетки смысла: вид кадра судится по свету
    # и мелочам в кадре. Замер на той же разметке: сетка смысла 3x3 400x300 —
    # 0.814 верных пар, плитки 480x320 в две-три колонки — 0.846-0.862.
    cols, tile = (None, None) if kind == "video" else (2 if len(paths) <= 4 else 3, LOOK_TILE)
    key = cache_key(model, "look" + str(LOOK_VERSION) + text + repr((cols, tile)), paths)
    cp = os.path.join(cache_dir, "look_" + key + ".json") if cache_dir else None
    if cp and os.path.exists(cp):
        try:
            return json.load(open(cp, encoding="utf-8"))["order"], {"cache_hit": True}
        except Exception:
            pass
    content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(
            _grid_bytes(paths, kind, cols=cols, tile=tile)).decode()}}]
    try:
        answer, _u, price = gateway.chat(model, content, 120, EST_PROMPT_TOKENS, reasoning=GRID_REASONING)
    except Exception as e:  # noqa: BLE001 — нет ответа: ничью решают прежние ключи
        return None, {"refused": f"{type(e).__name__}: {e}"[:200]}
    order = parse_order(answer, len(paths))
    if order is not None and cp:
        _cache_write(cp, {"order": order, "model": model}, readable=True)
    return order, {"cost": price, "call": True}


def spec_from_brief(phrase, brief):
    """Спецификация без плана: одно must-утверждение — бриф (или сама фраза).
    Путь один и тот же со спецификацией и без неё."""
    text = (brief or phrase or "").strip() or "—"
    return {"focus": text, "claims": [{"id": "c1", "text": text, "tier": "must"}]}


def shows_motion(kind, frames=None):
    """Может ли кадр показать движение: ролик минимум из двух кадров."""
    return kind == "video" and (frames is None or frames >= 2)


def asked_claims(spec, kind, frames=None):
    """Утверждения, которые спрашиваются у кадра: движение — только у ролика,
    показанного хотя бы двумя кадрами."""
    moving = shows_motion(kind, frames)
    return [c for c in spec["claims"] if moving or not c.get("motion")]


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


def focus_met(spec, answers):
    """Кадр показывает главное: первое утверждение — «да»."""
    return claim_values(spec, answers)[spec["claims"][0]["id"]] >= 1.0


def nothing_met(spec, answers):
    """Кадр не показывает из спецификации НИЧЕГО обязательного: каждое
    must-утверждение — «нет». Это брак; кадр, который не показал главное, но
    показал обязательную деталь фразы, — замена, а не брак (он проигрывает
    любому кадру с главным, но лучше соседнего кадра на чужой фразе).

    Отдельный вопрос «виден ли предмет фразы» проверен замером 25.09 (эп.94,
    два прогона на каждый вариант) и снят: узкий предмет («a rondel
    dagger») ловил +8 брака, но выбрасывал 3 годных кинжала другого вида;
    общий («a dagger») годных не терял, но и брак не ловил (37 принято
    против 35 без вопроса, лучший выбран 7/9 против 8/9)."""
    vals = claim_values(spec, answers)
    return all(vals[c["id"]] == 0 for c in spec["claims"] if c["tier"] == "must")


def shows_nothing(spec, answers, grid=None):
    """Проверка по пунктам считает кадр браком: не выполнено ничего
    обязательного (nothing_met), либо сетка того же судьи поставила 0
    («не по теме») — тогда одобрению по пунктам верить нельзя (эп.95: часы
    на 10:07 прошли как «часы у полуночи»). Одно правило для рендера и для
    бенча: две копии уже расходились — бенч не знал про сетку 0."""
    return nothing_met(spec, answers) or grid == 0


def musts_met_clean(spec, answers):
    """Все must-утверждения — «да», мир свой и фон чистый: лучше этот кадр по
    смыслу не станет, второй вид медиа искать незачем."""
    vals = claim_values(spec, answers)
    return (all(vals[c["id"]] >= 1.0 for c in spec["claims"] if c["tier"] == "must")
            and answers.get("main_in_world") is not False
            and not answers.get("background_foreign"))


def world_clear(answers):
    """Проверка мира состоялась и ничего чужого на кадре нет."""
    return (answers or {}).get("main_in_world") is True and answers.get("background_foreign") is False


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
        world, winfo = world_of_image(gateway, model, setting=setting, path=path, kind=kind,
                                      cache_dir=cache_dir, max_side=max_side, reasoning=reasoning,
                                      caption=caption, frames=frames)
        if world is None:
            return None, winfo
        answers, info = verify_claims(gateway, model, phrase=phrase, spec=spec, setting=None,
                                      path=path, kind=kind, cache_dir=cache_dir,
                                      max_side=max_side, reasoning=reasoning, caption=caption,
                                      frames=frames)
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
