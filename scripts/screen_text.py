#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Экранный текст поверх клипа: место и год печатной машинкой, карточка цитаты.

Оба приёма утверждены владельцем по ручному демо 03.10 (demo_glava/
2_mesto_i_god.mp4, 3_citata.mp4) с правками: подпись места крупнее
(+15-20%), кавычки в карточке цитаты белые.

Главное правило обоих — НИЧЕГО НЕ ДОДУМЫВАТЬ:
* подпись места и года — только там, где во фразе диктора явно названы и
  место, и год (place_year.place_and_year); не уверены — подписи нет;
* карточка цитаты — только для дословной цитаты в «ёлочках» с явно
  названным автором (имя рядом с глаголом речи в той же или соседней
  фразе). Пересказ («пишет, что…», «по словам…» без кавычек) — никогда: на
  экране это выглядело бы дословной цитатой, которой не было.

Модуль чистый: детектор, расписание и построение фильтров; рендер и
сведение — у вызывающего (pipeline_smart).
"""
import re

import feature_flags
import place_year

# --- место и год ---------------------------------------------------------------
PLACE_MIN_GAP_SEC = 40.0      # не чаще одной подписи на столько секунд
PLACE_LEAD_SEC = 0.55         # печать начинается через столько после начала фразы
PLACE_CPS = 0.075             # секунд на символ
PLACE_LINE_PAUSE = 0.18       # между местом и годом
PLACE_HOLD_SEC = 3.8          # от начала печати до начала исчезания
PLACE_FADE_OUT = 0.45
PLACE_MIN_SHOW_SEC = 2.0      # меньше — подписи нет (клип короче)
PLACE_X = 118
PLACE_SIZE, YEAR_SIZE = 56, 40       # +17% к демо (48/34) — правка владельца
PLACE_Y, YEAR_Y = 764, 832           # низ года ~872 — над зоной кнопок YouTube (880)
PLACE_BAR_W = 4
PLACE_BAR_X_GAP = 22
YEAR_COLOR = "0xE8E2D6"
CURSOR_BLINK = (0.5, 0.3)     # период, сколько горит
CURSOR_SEC = 1.2
KEY_GAP_LU = 20.0             # щелчок клавиши: макс. мгновенная на 20 LU ниже голоса (демо: -34 при -14)

# --- цитата --------------------------------------------------------------------
QUOTE_FADE = 0.5
QUOTE_TAIL_SEC = 0.25         # карточка гаснет до конца фразы
QUOTE_MIN_SHOW_SEC = 2.5
QUOTE_SIZE, QUOTE_MARK_SIZE, QUOTE_AUTHOR_SIZE = 66, 150, 32
QUOTE_MAX_WIDTH = 1500
QUOTE_MAX_LINES = 3
QUOTE_BLUR = "boxblur=26:2"
QUOTE_EQ = "eq=brightness=-0.22:saturation=0.5"
QUOTE_VIGNETTE = "vignette=PI/4"
QUOTE_AUTHOR_COLOR = "0xD8D2C6"
SPEECH_VERBS = ("писал", "пишет", "написал", "записал", "сказал", "говорил", "говорит",
                "вспоминал", "вспоминает", "ответил", "заявил", "восклицал", "кричал",
                "приказал", "произнёс", "произнес", "сообщал", "сообщает", "признавался",
                "признался", "диктовал")


def place_enabled():
    return feature_flags.enabled("PLACE_CAPTION") and feature_flags.enabled("ON_SCREEN_TEXT")


def quote_enabled():
    return feature_flags.enabled("QUOTE_CARD") and feature_flags.enabled("ON_SCREEN_TEXT")


def _year_word_pos(text):
    """Доля слов фразы до упоминания года (0..1) или None."""
    ys = place_year.years_in(text or "")
    if not ys:
        return None
    words_before = len((text or "")[:ys[0][1]].split())
    return words_before / max(1, len((text or "").split()))


def plan_place_captions(blocks, starts, ends, busy=(), confirm=None):
    """{индекс_слота: {"place", "year", "start"}} — start на шкале голоса.

    Фраза — исходная фраза сценария (parent_text): место и год должны быть
    названы в ОДНОЙ фразе. Подпись встаёт на первый её слот, где звучит год.
    busy — слоты, где уже есть плашка или заставка. starts/ends — начало и
    конец показа слота (сек)."""
    out, last = {}, None
    seen_parents = set()
    for i, b in enumerate(blocks or []):
        parent = b.get("parent_text") or b.get("text") or ""
        key = (b.get("orig_index"), parent)
        if key in seen_parents:
            continue
        got = place_year.place_and_year(parent, confirm)
        if not got:
            continue
        group = [k for k in range(i, len(blocks))
                 if (blocks[k].get("orig_index"), blocks[k].get("parent_text") or blocks[k].get("text")) == key]
        seen_parents.add(key)
        # Слот, где звучит год, первым; занят (плашка/заставка) или короток —
        # следующий слот той же фразы.
        order = sorted(group, key=lambda k: (not place_year.years_in(blocks[k].get("text") or ""), k))
        slot = next((k for k in order
                     if k not in busy and not blocks[k].get("stat") and not blocks[k].get("chapter_card")
                     and float(ends[k]) - (float(starts[k]) + PLACE_LEAD_SEC) >= PLACE_MIN_SHOW_SEC), None)
        if slot is None:
            continue
        start = float(starts[slot]) + PLACE_LEAD_SEC
        if last is not None and start - last < PLACE_MIN_GAP_SEC:
            continue
        out[slot] = {"place": got[0], "year": str(got[1]), "start": round(start, 6)}
        last = start
    return out


def place_char_times(place, year):
    """Моменты появления символов от начала печати (сек) и конец печати."""
    times = [k * PLACE_CPS for k in range(len(place))]
    base = len(place) * PLACE_CPS + PLACE_LINE_PAUSE
    times += [base + k * PLACE_CPS for k in range(len(year))]
    return times, base + len(year) * PLACE_CPS


def _w(text, size, font_path):
    try:
        from PIL import ImageFont
        return float(ImageFont.truetype(font_path, size).getlength(text))
    except Exception:
        return 0.6 * size * len(text)


def place_caption_chain(cap, local_start, dur, font_place, font_year, escape, year_font_path,
                        accent="0xC8102E"):
    """Цепочка drawtext/drawbox (через запятую) — подпись места и года в
    локальном времени клипа. fontfile-значения уже подготовлены."""
    c0 = float(local_start)
    times, typed = place_char_times(cap["place"], cap["year"])
    hold_end = min(c0 + PLACE_HOLD_SEC, float(dur) - PLACE_FADE_OUT - 0.05)
    alpha = f"if(lt(t\\,{hold_end:.3f})\\,1\\,max(0\\,1-(t-{hold_end:.3f})/{PLACE_FADE_OUT:.3f}))"
    end = hold_end + PLACE_FADE_OUT
    bar_h = (YEAR_Y + YEAR_SIZE) - (PLACE_Y - 2)
    parts = [f"drawbox=x={PLACE_X - PLACE_BAR_X_GAP}:y={PLACE_Y - 2}:w={PLACE_BAR_W}:h={bar_h}:"
             f"color={accent}@1:t=fill:enable='between(t,{c0 - 0.05:.3f},{end:.3f})'"]
    lines = ((cap["place"], PLACE_Y, PLACE_SIZE, font_place, "white", 0),
             (cap["year"], YEAR_Y, YEAR_SIZE, font_year, YEAR_COLOR, len(cap["place"])))
    for txt, y, fs, font, col, off in lines:
        for n in range(1, len(txt) + 1):
            a = c0 + times[off + n - 1]
            b = (c0 + times[off + n]) if n < len(txt) else end
            parts.append(f"drawtext=fontfile='{font}':text='{escape(txt[:n])}':fontsize={fs}:"
                         f"fontcolor={col}:shadowcolor=black@0.75:shadowx=0:shadowy=2:"
                         f"x={PLACE_X}:y={y}:alpha='{alpha}':enable='between(t,{a:.3f},{b - 0.001:.3f})'")
    t_typed = c0 + typed
    cx = PLACE_X + int(_w(cap["year"], YEAR_SIZE, year_font_path)) + 8
    per, on = CURSOR_BLINK
    parts.append(f"drawbox=x={cx}:y={YEAR_Y + 4}:w=3:h={int(YEAR_SIZE * 0.88)}:color={accent}@1:t=fill:"
                 f"enable='between(t,{t_typed:.3f},{t_typed + CURSOR_SEC:.3f})*lt(mod(t-{t_typed:.3f}\\,{per})\\,{on})'")
    return ",".join(parts)


# --- цитата --------------------------------------------------------------------
_QUOTE_RE = re.compile(r"«([^«»]{3,240})»")
_NAME = r"[А-ЯЁ][а-яё]+(?:[ -](?:[А-ЯЁ][а-яё]+|де|ле|фон|ван|да|ди))*"


# Слова, которые бывают с заглавной в начале предложения и автором не
# являются никогда: местоимения, наречия, союзы, частицы, адресаты. Нужны в
# двух местах: снять их с начала многословного «имени» («Потом Юниус пишет»
# -> Юниус) и не принять одиночное такое слово за автора, даже если оно
# где-то в сценарии стоит с заглавной.
_NOT_AUTHOR = {
    "он", "она", "оно", "они", "я", "ты", "мы", "вы", "его", "её", "ее", "их", "ему", "ей",
    "им", "это", "этот", "эта", "эти", "тот", "та", "те", "то", "там", "тут", "здесь",
    "тогда", "потом", "затем", "позже", "позднее", "после", "сначала", "вскоре", "ещё", "еще",
    "уже", "и", "а", "но", "да", "или", "так", "вот", "даже", "только", "лишь", "же", "ведь",
    "вдруг", "теперь", "сейчас", "кто", "что", "где", "когда", "как", "почему", "зачем",
    "сам", "сама", "сами", "само", "каждый", "один", "одна", "никто", "все", "всё", "многие",
    "итак", "впрочем", "кстати", "например", "наверное", "однажды", "снова", "опять",
    "дочери", "дочь", "сыну", "сын", "жене", "жена", "мужу", "муж", "матери", "мать", "отцу",
    "отец", "брату", "брат", "сестре", "сестра", "другу", "друг", "королю", "король",
    "историк", "историки", "летописец", "хронист", "монах", "автор", "свидетель", "судья",
    "в", "во", "на", "при", "по", "из", "у", "к", "с", "со", "о", "об", "за", "до", "от",
}

_SENT_END = ".!?…"


def _at_sentence_start(text, pos):
    """Позиция — начало предложения: начало текста или перед ней (через
    пробелы и закрывающие кавычки) конец предложения."""
    k = pos - 1
    while k >= 0 and text[k] in " \t\n\x00»\"'":
        k -= 1
    return k < 0 or text[k] in _SENT_END


_MID_NAME_RE = re.compile(r"(?<=[а-яёa-z0-9,;:)\-—] )([А-ЯЁ][а-яё]{2,})")


def script_names(text):
    """Слова с заглавной НЕ в начале предложения во всём сценарии — тот же
    приём подтверждения, что place_year.script_words: имя, которое где-то
    ещё стоит с заглавной посреди предложения, — имя собственное, а не
    первое слово фразы."""
    return set(_MID_NAME_RE.findall(text or ""))


def _confirmed_name(name, confirm):
    """Одиночное слово подтверждено сценарием: та же форма или та же основа
    с другим падежным окончанием («Юниус» <- «Иоганнеса Юниуса»)."""
    if not confirm or name.lower() in _NOT_AUTHOR:
        return False
    stem = name[:-1] if name[-1] in "аяйьоеиыу" else name
    if len(stem) < 3:
        return False
    return any(w == name or (w.startswith(stem) and len(w) - len(stem) <= 3) for w in confirm)


def _author_near(text, confirm=None):
    """Имя рядом с глаголом речи: «пишет Фруассар», «Фруассар писал». None — нет.

    Имя в НАЧАЛЕ предложения («Юниус пишет дочери: …») раньше отбрасывалось
    целиком: заглавная там бывает у любого слова. Теперь оно принимается,
    если это не служебное слово (_NOT_AUTHOR) и выполнено одно из:
    * перед ним стояли служебные слова («Потом Юниус пишет») — снимаются;
    * имя из двух слов с заглавной («Иоганнес Юниус пишет»);
    * одиночное слово подтверждено сценарием (confirm = script_names()):
      где-то ещё оно (или его падежная форма) стоит с заглавной посреди
      предложения. Без подтверждения — None: ложный автор хуже пропуска."""
    verbs = "|".join(SPEECH_VERBS)
    m = re.search(rf"\b(?:{verbs})\s+(?:[а-яё]+\s+)?({_NAME})", text)
    if m:
        return m.group(1)
    for m in re.finditer(rf"({_NAME})\s+(?:[а-яё]+\s+)?(?:{verbs})\b", text):
        name, start = m.group(1), m.start(1)
        if not _at_sentence_start(text, start):
            return name
        toks = name.split()
        lead = 0
        while lead < len(toks) and toks[lead].lower() in _NOT_AUTHOR:
            lead += 1
        rest = [t for t in toks[lead:]]
        if not rest or rest[0].lower() in _NOT_AUTHOR or not rest[0][:1].isupper():
            continue
        if lead or len(rest) >= 2 or _confirmed_name(rest[0], confirm):
            return " ".join(rest)
    return None


QUOTE_MIN_WORDS = 3            # «Демонологию», «Молот ведьм» — название, не речь


def _quote_is_speech(text, m):
    """Ёлочки — прямая речь, а не название/термин:
    * перед ними двоеточие («Юниус пишет дочери: «…»»);
    * перед ними запятая СРАЗУ после имени или глагола речи («Как писал
      Фруассар, «…»»). «…пишет целую книгу, «Демонологию»» — запятая после
      дополнения, это название;
    * после них «, —» («…», — вспоминал Маршал);
    и в самих ёлочках не меньше QUOTE_MIN_WORDS слов."""
    if len(m.group(1).split()) < QUOTE_MIN_WORDS:
        return False
    before = text[:m.start()].rstrip()
    after = text[m.end():].lstrip()
    if not before or before.endswith(":"):
        return True
    if after.startswith((", —", ",—", "—", ", -")):
        return True
    if before.endswith(","):
        prev = re.findall(r"[А-Яа-яЁё]+", before)
        return bool(prev) and (prev[-1][:1].isupper() or prev[-1].lower() in SPEECH_VERBS)
    return False


def find_quote(text, prev_text="", next_text="", confirm=None):
    """(цитата, автор) — только дословная цитата в «ёлочках», оформленная
    как прямая речь (_quote_is_speech), с автором рядом с глаголом речи в
    той же фразе (или соседней). Иначе None. confirm — script_names()."""
    m = _QUOTE_RE.search(text or "")
    if not m or not _quote_is_speech(text, m):
        return None
    quote = " ".join(m.group(1).split()).strip(" ,")
    rest = (text[:m.start()] + " " + text[m.end():])
    author = (_author_near(rest, confirm) or _author_near(prev_text or "", confirm)
              or _author_near(next_text or "", confirm))
    if not author:
        return None
    return quote, author


def plan_quote_cards(blocks, starts, ends, busy=(), confirm=None):
    """{индекс_слота: {"quote", "author", "q0", "q1"}} — q0/q1 на шкале голоса.

    Карточка держится над всеми слотами исходной фразы с цитатой: от момента
    начала цитаты (доля слов) до конца фразы минус QUOTE_TAIL_SEC."""
    out = {}
    parents = []
    for i, b in enumerate(blocks or []):
        key = (b.get("orig_index"), b.get("parent_text") or b.get("text"))
        if not parents or parents[-1][0] != key:
            parents.append((key, []))
        parents[-1][1].append(i)
    for n, (key, slots) in enumerate(parents):
        text = key[1] or ""
        prev_t = parents[n - 1][0][1] if n else ""
        next_t = parents[n + 1][0][1] if n + 1 < len(parents) else ""
        got = find_quote(text, prev_t, next_t, confirm)
        if not got or any(k in busy or blocks[k].get("chapter_card") for k in slots):
            continue
        p0, p1 = float(starts[slots[0]]), float(ends[slots[-1]])
        pos = text.find("«")
        frac = len(text[:pos].split()) / max(1, len(text.split()))
        q0 = p0 + frac * (p1 - p0)
        q1 = p1 - QUOTE_TAIL_SEC
        if q1 - q0 < QUOTE_MIN_SHOW_SEC:
            continue
        for k in slots:
            out[k] = {"quote": got[0], "author": got[1], "q0": round(q0, 6), "q1": round(q1, 6)}
    return out


def quote_layout(quote, font_path):
    """(кегль, строки) — по словам, не шире QUOTE_MAX_WIDTH, до QUOTE_MAX_LINES."""
    words = quote.split()
    for size in (QUOTE_SIZE, 58, 52):
        lines, cur = [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if cur and _w(cand, size, font_path) > QUOTE_MAX_WIDTH:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            lines.append(cur)
        if len(lines) <= QUOTE_MAX_LINES and all(_w(x, size, font_path) <= QUOTE_MAX_WIDTH for x in lines):
            return size, lines
    return None


def quote_graph(card, local_q0, local_q1, fonts, font_paths, escape, in_label, out_label):
    """Граф: in_label -> out_label с карточкой цитаты в локальном времени клипа.
    Кавычки белые (правка владельца)."""
    lay = quote_layout(card["quote"], font_paths["quote"])
    if not lay:
        return None
    size, lines = lay
    step = int(size * 1.27)
    top = -80 - (len(lines) - 2) * step // 2 if len(lines) >= 2 else -40

    def alpha(d):
        t0 = local_q0 + d
        return (f"if(lt(t\\,{t0:.3f})\\,0\\,if(lt(t\\,{t0 + 0.6:.3f})\\,(t-{t0:.3f})/0.6\\,"
                f"if(lt(t\\,{local_q1:.3f})\\,1\\,max(0\\,1-(t-{local_q1:.3f})/0.45))))")
    txt = [f"drawtext=fontfile='{fonts['mark']}':text='«':fontsize={QUOTE_MARK_SIZE}:fontcolor=white:"
           f"x=(w-text_w)/2:y=h/2+{top - 170}:alpha='{alpha(0.0)}'"]
    for k, line in enumerate(lines):
        txt.append(f"drawtext=fontfile='{fonts['quote']}':text='{escape(line)}':fontsize={size}:"
                   f"fontcolor=white:shadowcolor=black@0.6:shadowy=3:x=(w-text_w)/2:"
                   f"y=h/2+{top + k * step}:alpha='{alpha(0.15 + 0.2 * k)}'")
    ya = top + len(lines) * step + 30
    txt.append(f"drawtext=fontfile='{fonts['author']}':text='{escape('— ' + card['author'])}':"
               f"fontsize={QUOTE_AUTHOR_SIZE}:fontcolor={QUOTE_AUTHOR_COLOR}:x=(w-text_w)/2:"
               f"y=h/2+{ya}:alpha='{alpha(0.7)}'")
    fade_in = (f",fade=t=in:st={local_q0:.3f}:d={QUOTE_FADE:.3f}:alpha=1" if local_q0 > 0 else "")
    on = f"between(t,{max(0.0, local_q0):.3f},{local_q1 + QUOTE_FADE:.3f})"
    return (f"[{in_label}]split=2[qb][qc];[qc]{QUOTE_BLUR},{QUOTE_EQ},{QUOTE_VIGNETTE},format=yuva420p,"
            + ",".join(txt) + fade_in
            + f",fade=t=out:st={local_q1:.3f}:d={QUOTE_FADE:.3f}:alpha=1[qcard];"
            + f"[qb][qcard]overlay=format=yuv420p10:enable='{on}'[{out_label}]")
