#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Заставка главы: пауза диктора, размытый первый кадр с названием, удар.

Утверждено владельцем по ручному демо 03.10 (эпизод 03_plen, смена главы 4).
Отличие от демо одно и тоже по слову владельца: надписи «ГЛАВА N» нет —
только название главы по центру.

Что происходит на границе каждой BLOCK-секции с названием (`section_title()`
не None; HOOK и FINAL заставки не получают, как и раньше не получали титра):

1. ЗВУК — пауза диктора между главами ~CARD_PAUSE_SEC. Ставит её
   fix_pauses.py (вставкой тишины в саму паузу). Причина, по которой это
   понадобилось, измерена на 03_plen: lumean_tts.py склеивает секции с паузой
   SECTION_GAP_TARGET_SEC=0.95, вместе с хвостами она выходит ~1.0с, то есть
   ровно за порогом THRESH_SEC=1.0 у fix_pauses — и кривая подрезки сводит
   её к KEEP_MIN_SEC=0.42. Пауза на смене главы оказывалась КОРОЧЕ обычных
   пауз внутри глав (0.39–0.48с против медианы ~0.5–0.7).
2. ВИДЕО — первый клип новой главы начинается не на онсете речи, а сразу
   после конца речи прошлой главы (+CARD_LEAD_SEC). Переход в него — fade
   CARD_FADE_IN_SEC. Первые секунды клипа — заставка: тот же клип, размытый,
   притемнённый, с виньеткой и медленным дрейфом, по центру название и
   красная линия над ним. Название держится до CARD_TITLE_HOLD_AFTER_ONSET
   после того, как диктор начал новую главу, гаснет, и заставка растворяется
   в обычный чёткий клип с его Ken Burns. Нижний титр главы (add_overlays)
   на этом клипе не рисуется — двух титров быть не должно.
3. ЗВУК — на старте заставки удар (CC0, Freesound), атака в тишине паузы,
   хвост уходит под голос новой главы (осознанно, в отличие от правила
   «звук перехода только целиком в тишине»). Уровень выводится из громкости
   голоса: максимальная мгновенная громкость удара на CARD_HIT_GAP_LU ниже
   интегральной громкости голоса, пик удара не выше пика голоса.

Модуль чистый (без ffmpeg-вызовов, кроме чтения флагов): все числа и
построение фильтра здесь, сведение и рендер — у вызывающих.
"""
import os
import re

import feature_flags

# --- общее -------------------------------------------------------------------
CARD_LEAD_SEC = 0.07          # заставка видна через столько после конца речи
CARD_MIN_SHARP_TAIL = 0.3     # минимум чёткого клипа после растворения
CARD_TITLE_MAX_WIDTH = 1700
CARD_SHADOW = "shadowcolor=black@0.6:shadowx=0:shadowy=3"
CARD_LINE_W, CARD_LINE_H = 140, 4
CARD_LINE_GAP = 34            # красная линия над верхом названия
CARD_HIT_GAIN_MIN_DB, CARD_HIT_GAIN_MAX_DB = -40.0, 6.0

# Вид и тайминг — по стилю. «chapter» — заставка главы (демо chapter_1_udar),
# «title» — название ролика после хука (демо 1_nazvanie_posle_huka, без линии
# над названием по правке владельца). Числа взяты из утверждённых демо.
STYLES = {
    "chapter": {
        "pause": 1.8, "fade_in": 0.35,
        "blur": "boxblur=24:2", "eq": "eq=brightness=-0.16:saturation=0.55",
        "vignette": "vignette=PI/4", "scale_w": 2112, "drift_x": 8, "drift_y": 0,
        "font_size": 64, "min_font": 48, "split": "balanced", "top_single": -28,
        "spacing": 0.22, "t0": 0.05, "title_delay": 0.20, "line_stagger": 0.0,
        "text_fade_in": 0.45, "rise_px": 22, "rise_sec": 0.6,
        "hold_after_onset": 0.4, "text_fade_out": 0.45,
        "dissolve_delay": 0.30, "dissolve": 0.5, "accent_line": True,
        "hit_kind": "chapter_hit", "hit_gap_lu": 3.0, "hit_trim": None, "hit_fade_out": None,
    },
    "title": {
        "pause": 2.4, "fade_in": 0.4,
        "blur": "boxblur=30:2", "eq": "eq=brightness=-0.24:saturation=0.45",
        "vignette": "vignette=PI/3.5", "scale_w": 2200, "drift_x": 0, "drift_y": -6,
        "font_size": 72, "min_font": 72, "split": "greedy", "top_single": -36,
        "spacing": 0.31, "t0": 0.15, "title_delay": 0.10, "line_stagger": 0.25,
        "text_fade_in": 0.6, "rise_px": 26, "rise_sec": 0.8,
        "hold_after_onset": 0.25, "text_fade_out": 0.5,
        "dissolve_delay": 0.35, "dissolve": 0.6, "accent_line": False,
        "hit_kind": "title_hit", "hit_gap_lu": 1.5, "hit_trim": 9.0, "hit_fade_out": 3.0,
    },
}
CHAPTER, TITLE = STYLES["chapter"], STYLES["title"]

# Обратная совместимость имён (тесты, fix_pauses, pipeline_smart).
CARD_PAUSE_SEC = CHAPTER["pause"]
TITLE_PAUSE_SEC = TITLE["pause"]
CARD_FADE_IN_SEC = CHAPTER["fade_in"]
CARD_HIT_KIND = CHAPTER["hit_kind"]
CARD_HIT_GAP_LU = CHAPTER["hit_gap_lu"]

_TITLE_RE = re.compile(r'BLOCK\s+\d+\s*:\s*(.+)', re.I)


def style_of(card):
    return STYLES.get((card or {}).get("style") or "chapter", CHAPTER)


def enabled():
    """Заставка включена: свой флаг И экранный текст. ON_SCREEN_TEXT=0 —
    прежнее поведение без текста целиком (без заставки, паузы и удара)."""
    return feature_flags.enabled("CHAPTER_CARD") and feature_flags.enabled("ON_SCREEN_TEXT")


def title_drop_enabled():
    """Название ролика после хука: свой флаг И экранный текст."""
    return feature_flags.enabled("TITLE_DROP") and feature_flags.enabled("ON_SCREEN_TEXT")


def section_title(name):
    """BLOCK N: Название -> 'Название'. HOOK/FINAL/безымянные — None.
    Тот же разбор, что pipeline_smart.section_title() (тест держит их равными)."""
    m = _TITLE_RE.match(str(name or ""))
    return m.group(1).strip() if m else None


def card_boundaries(blocks):
    """Индексы блоков, с которых начинается секция С НАЗВАНИЕМ (кроме нулевого)."""
    out = []
    for i in range(1, len(blocks or [])):
        sec = blocks[i].get("section")
        if sec != blocks[i - 1].get("section") and section_title(sec):
            out.append(i)
    return out


def card_sections(section_names):
    """Имена секций (в порядке сценария), перед которыми стоит заставка."""
    out = []
    for k, name in enumerate(section_names or []):
        if k > 0 and section_title(name):
            out.append(name)
    return out


def card_visual_start(prev_speech_end):
    """Момент, когда заставка ПОЯВЛЯЕТСЯ (шкала реального аудио)."""
    return float(prev_speech_end) + CARD_LEAD_SEC


# --- название ролика ---------------------------------------------------------

def _metadata(script_path):
    out = {}
    try:
        with open(script_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return out
    m = re.search(r"===\s*METADATA\s*===(.*?)(?:\n===|\Z)", text, re.S)
    for line in (m.group(1) if m else "").splitlines():
        k, sep, v = line.partition(":")
        if sep and k.strip():
            out[k.strip().upper()] = v.strip()
    return out


def film_title(script_path, font_path):
    """Текст заставки названия ролика или None.

    METADATA `TITLE_CARD:` если есть, иначе `TITLE:` без скобок; не влезает в
    две строки шириной <= CARD_TITLE_MAX_WIDTH — часть до « — »; не влезает и
    так — None (заставки названия нет, у главы 1 обычная заставка)."""
    meta = _metadata(script_path)
    text = meta.get("TITLE_CARD") or re.sub(r"\([^)]*\)", "", meta.get("TITLE", ""))
    text = " ".join(text.split())
    if not text:
        return None
    for cand in (text, text.split(" — ")[0].strip()):
        if cand and title_layout(cand.upper(), font_path, TITLE)[2]:
            return cand
    return None


# --- вёрстка -------------------------------------------------------------------

def _text_width(text, size, font_path):
    try:
        from PIL import ImageFont
        return float(ImageFont.truetype(font_path, size).getlength(text))
    except Exception:
        # без PIL/шрифта — грубая оценка по ширине широкого гротеска
        return 0.92 * size * len(text)


def _split_two(words, size, font_path, how):
    if len(words) < 2:
        return None
    if how == "greedy":
        for k in range(len(words) - 1, 0, -1):
            a = " ".join(words[:k])
            if _text_width(a, size, font_path) <= CARD_TITLE_MAX_WIDTH:
                return [a, " ".join(words[k:])]
        return None
    best = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        w = max(_text_width(a, size, font_path), _text_width(b, size, font_path))
        if best is None or w < best[0]:
            best = (w, [a, b])
    return best[1]


def title_layout(title, font_path, style=None):
    """(кегль, [строки], влезло) — каждая строка не шире CARD_TITLE_MAX_WIDTH.

    Глава: одна строка с уменьшением до 56, потом две строки (баланс) с
    уменьшением до min_font. Название ролика: две строки (первая — сколько
    влезает, как в демо), без уменьшения ниже min_font."""
    st = style or CHAPTER
    text = " ".join(str(title or "").split())
    if not text:
        return st["font_size"], [], False
    size0 = st["font_size"]

    def fits(lines, size):
        return max(_text_width(x, size, font_path) for x in lines) <= CARD_TITLE_MAX_WIDTH

    if st["split"] == "balanced":
        for size in range(size0, 55, -2):
            if fits([text], size):
                return size, [text], True
    words = text.split(" ")
    if st["split"] == "greedy" and fits([text], size0) and len(words) < 3:
        return size0, [text], True
    size = size0
    while True:
        lines = _split_two(words, size, font_path, st["split"]) or [text]
        if fits(lines, size):
            return size, lines, True
        if size - 2 < st["min_font"]:
            return size, lines, False
        size -= 2


def card_timing(lead, dur, style=None):
    """Моменты внутри клипа-заставки (локальное время клипа, 0 = начало
    перехода в заставку). lead — где в клипе начинается речь новой главы.

    Клип короче задуманного — растворение подтягивается так, чтобы после
    него оставалось CARD_MIN_SHARP_TAIL чёткого клипа (но не раньше
    начала речи)."""
    st = style or CHAPTER
    lead = max(0.0, float(lead))
    dur = float(dur)
    t0 = st["t0"]
    title_out = lead + st["hold_after_onset"]
    dissolve = title_out + st["dissolve_delay"]
    latest = dur - st["dissolve"] - CARD_MIN_SHARP_TAIL
    if dissolve > latest:
        shift = dissolve - max(latest, lead)
        dissolve -= shift
        title_out -= shift
    title_in = t0 + st["title_delay"]
    title_out = max(title_out, title_in + st["text_fade_in"])
    return {"t0": t0, "line_on": t0 + 0.15, "line_off": title_out + 0.15,
            "title_in": title_in, "title_out": title_out,
            "dissolve": dissolve, "card_end": dissolve + st["dissolve"]}


def _alpha_expr(t_in, fade_in, t_out, fade_out):
    return (f"if(lt(t\\,{t_in:.3f})\\,0\\,if(lt(t\\,{t_in + fade_in:.3f})\\,"
            f"(t-{t_in:.3f})/{fade_in:.3f}\\,if(lt(t\\,{t_out:.3f})\\,1\\,"
            f"max(0\\,1-(t-{t_out:.3f})/{fade_out:.3f}))))")


def card_filter(title, lead, dur, font_path, fontfile, escape, accent="0xC8102E", fps=24,
                upper=True, style=None):
    """filter_complex: [0:v] (готовый клип) -> [vout] (клип с заставкой).

    Заставка — тот же клип (его Ken Burns даёт дрейф «живым»), размытый и
    притемнённый, плюс дрейф кадрирования; сверху — название (и у главы —
    красная линия над ним). Растворяется альфой поверх чёткого клипа, после
    конца растворения клип идёт как есть. fontfile — уже подготовленное
    значение опции (см. pipeline_smart.ffmpeg_filter_path), font_path — тот же
    шрифт как файл (для замера ширины строки), escape — экранирование текста."""
    st = style or CHAPTER
    tm = card_timing(lead, dur, st)
    text = (title or "").upper() if upper else (title or "")
    size, lines, _ok = title_layout(text, font_path, st)
    n = max(1, len(lines))
    spacing = int(round(size * st["spacing"]))
    if n == 1:
        top = st["top_single"]
    else:
        block_h = n * size + (n - 1) * spacing
        top = -int(round(block_h / 2.0))
    card_len = tm["card_end"]
    crop_x = f"(iw-1920)/2+t*{st['drift_x']}" if st["drift_x"] else "(iw-1920)/2"
    crop_y = f"(ih-1080)/2+t*{st['drift_y']}" if st["drift_y"] else "(ih-1080)/2"
    chain = (f"[c]trim=duration={card_len + 1.0 / fps:.3f},setpts=PTS-STARTPTS,"
             f"scale={st['scale_w']}:-2,crop=1920:1080:'{crop_x}':'{crop_y}',"
             f"{st['blur']},{st['eq']},{st['vignette']},format=yuva420p")
    if st["accent_line"]:
        chain += (f",drawbox=x=(iw-{CARD_LINE_W})/2:y=ih/2+{top - CARD_LINE_GAP}:"
                  f"w={CARD_LINE_W}:h={CARD_LINE_H}:color={accent}@1:t=fill:"
                  f"enable='between(t,{tm['line_on']:.3f},{tm['line_off']:.3f})'")
    for k, line in enumerate(lines):
        y = top + k * (size + spacing)
        t_in = tm["title_in"] + k * st["line_stagger"]
        alpha = _alpha_expr(t_in, st["text_fade_in"], tm["title_out"], st["text_fade_out"])
        rise = (f"(1-min(max(t-{t_in:.3f}\\,0)/{st['rise_sec']:.3f}\\,1))*{st['rise_px']}")
        chain += (f",drawtext=fontfile='{fontfile}':text='{escape(line)}':"
                  f"fontsize={size}:fontcolor=white:{CARD_SHADOW}:"
                  f"x=(w-text_w)/2:y='h/2+{y}+{rise}':alpha='{alpha}'")
    chain += f",fade=t=out:st={tm['dissolve']:.3f}:d={st['dissolve']:.3f}:alpha=1[card]"
    return ("[0:v]split=2[base][c];" + chain +
            ";[base][card]overlay=format=yuv420p10:eof_action=pass:shortest=0[vout]")


def card_signature_source():
    """Исходник рецепта заставки — для подписи клипа с заставкой (правка вида
    перерендерит клипы с заставкой, а ключи кэша остальных не трогает)."""
    import inspect
    import sys
    return inspect.getsource(sys.modules[__name__])


def hit_gain_db(voice_lufs, voice_peak_dbfs, hit_max_momentary, hit_peak_dbfs, gap_lu=CARD_HIT_GAP_LU):
    """(усиление_дБ, источник). Макс. мгновенная громкость удара на gap_lu
    ниже интегральной громкости голоса; пик удара не выше пика голоса. Нет
    замеров — None (вызывающий не ставит удар вслепую)."""
    if voice_lufs is None or hit_max_momentary is None:
        return None, "not_measured"
    gain = float(voice_lufs) - float(gap_lu) - float(hit_max_momentary)
    src = "measured"
    if voice_peak_dbfs is not None and hit_peak_dbfs is not None:
        cap = float(voice_peak_dbfs) - float(hit_peak_dbfs)
        if gain > cap:
            gain, src = cap, "measured_peak_capped"
    clamped = max(CARD_HIT_GAIN_MIN_DB, min(CARD_HIT_GAIN_MAX_DB, gain))
    if abs(clamped - gain) > 0.05:
        src = "measured_clamped"
    return round(clamped, 2), src


def hook_title_section(section_names):
    """Секция, перед которой встаёт название ролика: первая BLOCK-секция с
    названием сразу после хука. None — нет такой."""
    for k in range(1, len(section_names or [])):
        if str(section_names[k - 1]).upper().startswith("HOOK") and section_title(section_names[k]):
            return section_names[k]
    return None


def default_font():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "fonts", "Benzin-ExtraBold.ttf")
