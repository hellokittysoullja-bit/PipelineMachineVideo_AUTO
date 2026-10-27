"""Монтаж второго канала (POLISH_FX=1): кинорамка хука, акцентный наезд на
слове, подъём перед заставкой главы, сердцебиение под хуком, уход в чёрное
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

# Кинорамка 2.39:1 в хуке: видимая высота 1920/2.39 = 803 px, полосы по 138.
LETTERBOX_BAR_PX = 138
# Полосы раздвигаются на заставке названия ролика: начинают, когда название
# уже проступило, и уходят за LETTERBOX_OPEN_SEC.
LETTERBOX_OPEN_AT_SEC = 0.35
LETTERBOX_OPEN_SEC = 1.2

# Акцентный наезд: кадр за PUNCH_ATTACK_SEC приближается на PUNCH_ZOOM и так
# и остаётся до конца клипа (классический punch-in, не «пружина» назад).
PUNCH_ZOOM = 0.08
PUNCH_ATTACK_SEC = 0.12
# Наезд на слове попадает на первый слог: якорь — оценка позиции слова
# внутри фразы, поэтому наезд берётся с небольшим опережением.
PUNCH_LEAD_SEC = 0.06

# Звук: подъём к удару заставки — тише самого удара; удар под наезд — тише
# удара заставки; сердцебиение — фон под хуком, далеко под голосом.
RISER_GAIN_OFFSET_DB = -6.0
PUNCH_HIT_GAIN_OFFSET_DB = -9.0
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


def letterbox_filter(in_label, out_label, dur, mode, fps=FPS_DEFAULT, w=W, h=H):
    """Чёрные полосы сверху и снизу. mode='full' — весь клип; mode='open' —
    полосы уходят за край кадра начиная с LETTERBOX_OPEN_AT_SEC."""
    bar = LETTERBOX_BAR_PX
    p = _clip_expr(LETTERBOX_OPEN_AT_SEC, LETTERBOX_OPEN_SEC) if mode == "open" else "0"
    d = f"{max(float(dur), 0.1):.3f}"
    return (f"color=black:s={w}x{bar}:r={fps}:d={d},split[lbt_{out_label}][lbb_{out_label}];"
            f"[{in_label}][lbt_{out_label}]overlay=x=0:y='-{bar}*{p}':eval=frame:shortest=1[lb1_{out_label}];"
            f"[lb1_{out_label}][lbb_{out_label}]overlay=x=0:y='{h - bar}+{bar}*{p}':eval=frame:shortest=1"
            f"[{out_label}]")


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
    """Метки приёмов в блоках: кинорамка хука, моменты наездов. Без флага —
    метки снимаются (блоки как раньше)."""
    for b in blocks or []:
        b.pop("letterbox", None)
        b.pop("punch_times", None)
    if not enabled():
        return
    for i, b in enumerate(blocks or []):
        if str(b.get("section") or "").upper().startswith("HOOK"):
            b["letterbox"] = "full"
        if real_weights and i < len(sub_starts) and i < len(real_weights) and real_weights[i]:
            times = punch_times(b, float(sub_starts[i]), float(real_weights[i]))
            if times:
                b["punch_times"] = [round(t, 3) for t in times]
