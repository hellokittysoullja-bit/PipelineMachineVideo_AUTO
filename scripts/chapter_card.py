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
import re

import feature_flags

# --- тайминг (секунды) -------------------------------------------------------
CARD_PAUSE_SEC = 1.8          # пауза диктора на смене главы (тишина, -30 dB)
CARD_LEAD_SEC = 0.07          # заставка видна через столько после конца речи
CARD_FADE_IN_SEC = 0.35       # переход прошлый клип -> заставка
CARD_TITLE_T0 = 0.05          # опорный момент надписей от начала заставки
CARD_LINE_DELAY = 0.15        # линия появляется через столько от T0
CARD_TITLE_DELAY = 0.20       # название начинает всплывать через столько от T0
CARD_TITLE_FADE_IN = 0.45
CARD_TITLE_RISE_SEC = 0.6
CARD_TITLE_RISE_PX = 22
CARD_TITLE_HOLD_AFTER_ONSET = 0.4   # название держится после начала речи
CARD_TITLE_FADE_OUT = 0.45
CARD_DISSOLVE_DELAY = 0.30    # растворение начинается через столько после начала гашения
CARD_DISSOLVE_SEC = 0.5
CARD_MIN_SHARP_TAIL = 0.3     # минимум чёткого клипа после растворения

# --- вид заставки ------------------------------------------------------------
CARD_BLUR = "boxblur=24:2"
CARD_EQ = "eq=brightness=-0.16:saturation=0.55"
CARD_VIGNETTE = "vignette=PI/4"
CARD_DRIFT_PX_PER_SEC = 8
CARD_SCALE_W = 2112           # запас по ширине под дрейф (1.1 x 1920)
CARD_TITLE_FONTSIZE = 64
CARD_TITLE_MIN_FONTSIZE = 48
CARD_TITLE_MAX_WIDTH = 1700
CARD_TITLE_Y = -28            # верх строки относительно центра кадра (демо)
CARD_LINE_W, CARD_LINE_H = 140, 4
CARD_LINE_GAP = 34            # линия над верхом названия
CARD_LINE_SPACING = 0.22      # межстрочный зазор в долях кегля (2 строки)
CARD_SHADOW = "shadowcolor=black@0.6:shadowx=0:shadowy=3"

# --- удар --------------------------------------------------------------------
CARD_HIT_KIND = "chapter_hit"
CARD_HIT_GAP_LU = 3.0         # макс. мгновенная громкость удара ниже голоса
CARD_HIT_GAIN_MIN_DB, CARD_HIT_GAIN_MAX_DB = -40.0, 6.0

_TITLE_RE = re.compile(r'BLOCK\s+\d+\s*:\s*(.+)', re.I)


def enabled():
    """Заставка включена: свой флаг И экранный текст. ON_SCREEN_TEXT=0 —
    прежнее поведение без текста целиком (без заставки, паузы и удара)."""
    return feature_flags.enabled("CHAPTER_CARD") and feature_flags.enabled("ON_SCREEN_TEXT")


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


def card_timing(lead, dur):
    """Моменты внутри клипа-заставки (локальное время клипа, 0 = начало
    перехода в заставку). lead — где в клипе начинается речь новой главы.

    Клип короче задуманного — растворение подтягивается так, чтобы после
    него оставалось CARD_MIN_SHARP_TAIL чёткого клипа (но не раньше
    начала речи)."""
    lead = max(0.0, float(lead))
    dur = float(dur)
    t0 = CARD_TITLE_T0
    title_out = lead + CARD_TITLE_HOLD_AFTER_ONSET
    dissolve = title_out + CARD_DISSOLVE_DELAY
    latest = dur - CARD_DISSOLVE_SEC - CARD_MIN_SHARP_TAIL
    if dissolve > latest:
        shift = dissolve - max(latest, lead)
        dissolve -= shift
        title_out -= shift
    title_in = t0 + CARD_TITLE_DELAY
    title_out = max(title_out, title_in + CARD_TITLE_FADE_IN)
    return {"t0": t0, "line_on": t0 + CARD_LINE_DELAY,
            "line_off": title_out + CARD_LINE_DELAY,
            "title_in": title_in, "title_out": title_out,
            "dissolve": dissolve, "card_end": dissolve + CARD_DISSOLVE_SEC}


def _text_width(text, size, font_path):
    try:
        from PIL import ImageFont
        return float(ImageFont.truetype(font_path, size).getlength(text))
    except Exception:
        # без PIL/шрифта — грубая оценка по ширине широкого гротеска
        return 0.92 * size * len(text)


def title_layout(title, font_path):
    """(кегль, [строки]) так, чтобы каждая строка была уже CARD_TITLE_MAX_WIDTH.

    Сначала одна строка с уменьшением кегля до 56 (на 64 влезает большинство
    названий), потом две строки с переносом по словам (разрез, при котором
    длиннейшая строка короче всего), с уменьшением до CARD_TITLE_MIN_FONTSIZE."""
    text = " ".join(str(title or "").split())
    if not text:
        return CARD_TITLE_FONTSIZE, []
    for size in range(CARD_TITLE_FONTSIZE, 55, -2):
        if _text_width(text, size, font_path) <= CARD_TITLE_MAX_WIDTH:
            return size, [text]
    words = text.split(" ")
    best = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        w = max(_text_width(a, CARD_TITLE_FONTSIZE, font_path),
                _text_width(b, CARD_TITLE_FONTSIZE, font_path))
        if best is None or w < best[0]:
            best = (w, [a, b])
    if best is None:   # одно длинное слово — только уменьшение
        size = CARD_TITLE_FONTSIZE
        while size > CARD_TITLE_MIN_FONTSIZE and _text_width(text, size, font_path) > CARD_TITLE_MAX_WIDTH:
            size -= 2
        return size, [text]
    lines = best[1]
    size = CARD_TITLE_FONTSIZE
    while size > CARD_TITLE_MIN_FONTSIZE and max(
            _text_width(x, size, font_path) for x in lines) > CARD_TITLE_MAX_WIDTH:
        size -= 2
    return size, lines


def _alpha_expr(t_in, fade_in, t_out, fade_out):
    return (f"if(lt(t\\,{t_in:.3f})\\,0\\,if(lt(t\\,{t_in + fade_in:.3f})\\,"
            f"(t-{t_in:.3f})/{fade_in:.3f}\\,if(lt(t\\,{t_out:.3f})\\,1\\,"
            f"max(0\\,1-(t-{t_out:.3f})/{fade_out:.3f}))))")


def card_filter(title, lead, dur, font_path, fontfile, escape, accent="0xC8102E", fps=24, upper=True):
    """filter_complex: [0:v] (готовый клип главы) -> [vout] (клип с заставкой).

    Заставка — тот же клип (его Ken Burns даёт дрейф «живым»), размытый и
    притемнённый, плюс дрейф кадрирования; сверху — линия и название.
    Растворяется альфой поверх чёткого клипа, после конца растворения клип
    идёт как есть. fontfile — уже подготовленное значение опции (см.
    pipeline_smart.ffmpeg_filter_path), font_path — тот же шрифт как файл (для
    замера ширины строки), escape — экранирование текста drawtext."""
    tm = card_timing(lead, dur)
    text = (title or "").upper() if upper else (title or "")
    size, lines = title_layout(text, font_path)
    n = max(1, len(lines))
    spacing = int(round(size * CARD_LINE_SPACING))
    if n == 1:
        top = CARD_TITLE_Y
    else:
        block_h = n * size + (n - 1) * spacing
        top = -int(round(block_h / 2.0))
    alpha = _alpha_expr(tm["title_in"], CARD_TITLE_FADE_IN, tm["title_out"], CARD_TITLE_FADE_OUT)
    rise = (f"(1-min(max(t-{tm['title_in']:.3f}\\,0)/{CARD_TITLE_RISE_SEC:.3f}\\,1))"
            f"*{CARD_TITLE_RISE_PX}")
    card_len = tm["card_end"]
    chain = (f"[c]trim=duration={card_len + 1.0 / fps:.3f},setpts=PTS-STARTPTS,"
             f"scale={CARD_SCALE_W}:-2,"
             f"crop=1920:1080:'(iw-1920)/2+t*{CARD_DRIFT_PX_PER_SEC}':'(ih-1080)/2',"
             f"{CARD_BLUR},{CARD_EQ},{CARD_VIGNETTE},format=yuva420p,"
             f"drawbox=x=(iw-{CARD_LINE_W})/2:y=ih/2+{top - CARD_LINE_GAP}:"
             f"w={CARD_LINE_W}:h={CARD_LINE_H}:color={accent}@1:t=fill:"
             f"enable='between(t,{tm['line_on']:.3f},{tm['line_off']:.3f})'")
    for k, line in enumerate(lines):
        y = top + k * (size + spacing)
        chain += (f",drawtext=fontfile='{fontfile}':text='{escape(line)}':"
                  f"fontsize={size}:fontcolor=white:{CARD_SHADOW}:"
                  f"x=(w-text_w)/2:y='h/2+{y}+{rise}':alpha='{alpha}'")
    chain += (f",fade=t=out:st={tm['dissolve']:.3f}:d={CARD_DISSOLVE_SEC:.3f}:alpha=1[card]")
    return ("[0:v]split=2[base][c];" + chain +
            ";[base][card]overlay=format=yuv420p10:eof_action=pass:shortest=0[vout]")


def card_signature_source():
    """Исходник рецепта заставки — для render_recipe_signature (правка вида
    перерендерит клипы с заставкой, а ключи кэша остальных не трогает)."""
    import inspect
    import sys
    mod = sys.modules[__name__]
    return inspect.getsource(mod)


def hit_gain_db(voice_lufs, voice_peak_dbfs, hit_max_momentary, hit_peak_dbfs):
    """(усиление_дБ, источник). Макс. мгновенная громкость удара на
    CARD_HIT_GAP_LU ниже интегральной громкости голоса; пик удара не выше
    пика голоса. Нет замеров — None (вызывающий не ставит удар вслепую)."""
    if voice_lufs is None or hit_max_momentary is None:
        return None, "not_measured"
    gain = float(voice_lufs) - CARD_HIT_GAP_LU - float(hit_max_momentary)
    src = "measured"
    if voice_peak_dbfs is not None and hit_peak_dbfs is not None:
        cap = float(voice_peak_dbfs) - float(hit_peak_dbfs)
        if gain > cap:
            gain, src = cap, "measured_peak_capped"
    clamped = max(CARD_HIT_GAIN_MIN_DB, min(CARD_HIT_GAIN_MAX_DB, gain))
    if abs(clamped - gain) > 0.05:
        src = "measured_clamped"
    return round(clamped, 2), src
