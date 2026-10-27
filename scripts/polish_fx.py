"""Монтаж второго канала (POLISH_FX=1): акцентный наезд на слове, свуш на склейках
хука, подъём перед заставкой главы, сердцебиение под хуком, уход в чёрное
перед кульминацией.

Все приёмы — поверх уже принятых решений движка, ни одного нового решения
об отборе кадров. Флаг выключен — ни одна функция модуля не вызывается, и
ролик собирается байт-в-байт как раньше (первый канал).

Кадр: всё рисуется пост-проходом по готовому клипу (тот же apply_clip_post,
что у заставок и подписей), поэтому кэш клипа учитывает приём через ключ
наложения, а клипы без приёма не перерендериваются.
"""
import feature_flags

FPS_DEFAULT = 24
W, H = 1920, 1080

# Акцентный наезд: кадр за PUNCH_ATTACK_SEC приближается на PUNCH_ZOOM и так
# и остаётся до конца клипа (классический punch-in, не «пружина» назад).
PUNCH_ZOOM = 0.08
PUNCH_ATTACK_SEC = 0.12
# Наезд на слове попадает на первый слог: якорь — оценка позиции слова
# внутри фразы, поэтому наезд берётся с небольшим опережением.
PUNCH_LEAD_SEC = 0.06

# Звук: подъём к удару заставки — тише самого удара; удар под наезд — тише
# удара заставки; сердцебиение — фон под хуком, далеко под голосом.
# Уровни сверены с готовым миксом 27.10: при -6/-9 подъём выходил около
# -42 LUFS, удар около -45 (голос около -16) — на слух их не было вовсе.
# Отсчёт — измеренное усиление уровня «chapter» (sfx_cue_gain_db).
RISER_GAIN_OFFSET_DB = 4.0
PUNCH_HIT_GAIN_OFFSET_DB = 3.0
# Свуш на смене кадра в хуке (рилс-подача): пик свуша приходится чуть раньше
# первого слова новой фразы, звук тихий — это ритм, а не эффект.
HOOK_CUT_PEAK_BEFORE_SEC = 0.12
HOOK_CUT_GAIN_OFFSET_DB = -5.0
HEARTBEAT_GAIN_OFFSET_DB = -8.0
HEARTBEAT_FADE_IN_SEC = 1.0
HEARTBEAT_FADE_OUT_SEC = 1.5

# Уход в чёрное перед кульминацией вместо обычной склейки.
CLIMAX_DIP_FRAMES = 12


def enabled():
    return feature_flags.enabled("POLISH_FX")


def _clip_expr(t0, dur):
    """0 до t0, линейно до 1 за dur, дальше 1 — выражение ffmpeg по t."""
    return f"clip((t-{t0:.3f})/{max(dur, 1e-3):.3f},0,1)"


def punch_filter(in_label, out_label, times, w=W, h=H):
    """Наезд на моментах times (сек, локально в клипе): scale по кадру + crop
    обратно в w x h. Несколько наездов в клипе складываются (потолок 2x)."""
    times = sorted(float(t) for t in times if t is not None and float(t) >= 0)
    if not times:
        return None
    z = "+".join(f"{PUNCH_ZOOM}*{_clip_expr(max(0.0, t - PUNCH_LEAD_SEC), PUNCH_ATTACK_SEC)}"
                 for t in times[:2])
    sw = f"trunc({w}*(1+{z})/2)*2"
    sh = f"trunc({h}*(1+{z})/2)*2"
    return (f"[{in_label}]scale=w='{sw}':h='{sh}':eval=frame,"
            f"crop={w}:{h}:(iw-{w})/2:(ih-{h})/2,setsar=1[{out_label}]")


def punch_word_positions(block):
    """Позиции слов с тегом [punch] в блоке (после нарезки — свои у куска)."""
    return [int(c.get("word_pos", 0)) for c in (block or {}).get("sound_cues") or []
            if c.get("type") == "punch"]


def punch_times(block, speech_start, speech_dur):
    """Моменты наездов блока на шкале голоса: линейная оценка позиции слова
    (та же, что sfx_plan.word_anchor_time у предметных звуков)."""
    import sfx_plan
    return [sfx_plan.word_anchor_time(block, pos, speech_start, speech_dur)
            for pos in punch_word_positions(block)]


def riser_cues(hit_cues, riser_assets, durations, gain_for):
    """Подъём, который ЗАКАНЧИВАЕТСЯ на ударе заставки. riser_assets — пути,
    durations — их длительности, gain_for(path) -> (gain_db, source)."""
    out = []
    assets = [(p, d) for p, d in zip(riser_assets, durations) if d and d > 0.3]
    if not assets:
        return out
    for k, c in enumerate(hit_cues):
        if c.get("gain_db") is None:
            continue
        path, dur = assets[k % len(assets)]
        gain, src = gain_for(path)
        if gain is None:
            continue
        out.append({"kind": "card_riser", "time": max(0.0, float(c["time"]) - float(dur)),
                    "block": c.get("block"), "section": c.get("section"), "asset": path,
                    "gain_db": round(gain + RISER_GAIN_OFFSET_DB, 2), "gain_source": src})
    return out


def heartbeat_bed(asset, length, out_path):
    """Сердцебиение на length секунд: запись по кругу. Путь или None."""
    import subprocess
    if not asset or length <= 1.0:
        return None
    cmd = ["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", asset,
           "-t", f"{float(length):.3f}", "-ac", "2", "-ar", "48000", out_path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return out_path if r.returncode == 0 else None


def heartbeat_cue(asset, hook_end, gain_db, gain_source, out_path):
    """Кюй сердцебиения от начала ролика до входа в название (hook_end)."""
    if gain_db is None or not hook_end or hook_end <= 2.0:
        return None
    path = heartbeat_bed(asset, hook_end, out_path)
    if not path:
        return None
    return {"kind": "hook_heartbeat", "time": 0.0, "asset": path,
            "trim_sec": round(float(hook_end), 3), "fade_in_sec": HEARTBEAT_FADE_IN_SEC,
            "fade_out_sec": HEARTBEAT_FADE_OUT_SEC,
            "gain_db": round(float(gain_db) + HEARTBEAT_GAIN_OFFSET_DB, 2),
            "gain_source": gain_source}


def punch_hit_cues(blocks, hit_assets, gain_for):
    """Удар под каждый наезд (b["punch_times"] — шкала голоса)."""
    out, k = [], 0
    if not hit_assets:
        return out
    for i, b in enumerate(blocks or []):
        for t in b.get("punch_times") or []:
            path = hit_assets[k % len(hit_assets)]
            k += 1
            gain, src = gain_for(path)
            if gain is None:
                continue
            out.append({"kind": "punch_hit", "time": max(0.0, float(t) - PUNCH_LEAD_SEC),
                        "block": i, "section": b.get("section"), "asset": path,
                        "gain_db": round(gain + PUNCH_HIT_GAIN_OFFSET_DB, 2), "gain_source": src})
    return out


def mark_blocks(blocks, sub_starts, real_weights):
    """Метки приёмов в блоках: моменты наездов. Без флага — метки снимаются
    (блоки как раньше). Чёрные полосы кинорамки убраны по слову владельца
    27.10: на экране они читались как дефект."""
    for b in blocks or []:
        b.pop("letterbox", None)
        b.pop("punch_times", None)
    if not enabled():
        return
    for i, b in enumerate(blocks or []):
        if real_weights and i < len(sub_starts) and i < len(real_weights) and real_weights[i]:
            times = punch_times(b, float(sub_starts[i]), float(real_weights[i]))
            if times:
                b["punch_times"] = [round(t, 3) for t in times]


def hook_cut_swooshes(blocks, sub_starts, assets, peaks, gain_for):
    """Свуш на каждой смене ФРАЗЫ в хуке (кроме первой): ритм хука как в
    рилс. assets/peaks — файлы и момент пика в каждом (чередуются). Кадры
    внутри фразы (под-кадры) звука не получают: иначе свуши шли бы чаще,
    чем фразы, и читались бы как шум."""
    out, k = [], 0
    if not assets:
        return out
    for i, b in enumerate(blocks or []):
        if i == 0 or i >= len(sub_starts) or b.get("is_subcut"):
            continue
        if not str(b.get("section") or "").upper().startswith("HOOK"):
            continue
        path, peak = assets[k % len(assets)], peaks[k % len(assets)]
        k += 1
        gain, src = gain_for(path)
        if gain is None:
            continue
        out.append({"kind": "hook_cut", "time": max(0.0, float(sub_starts[i]) - HOOK_CUT_PEAK_BEFORE_SEC - peak),
                    "block": i, "section": b.get("section"), "asset": path,
                    "gain_db": round(gain + HOOK_CUT_GAIN_OFFSET_DB, 2), "gain_source": src})
    return out
