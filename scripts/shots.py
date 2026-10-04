#!/usr/bin/env python3
"""План камеры и событий одного кадра: что и когда видно на экране.

Один рисунок держится 6-10 с (новая картинка дорогая — решение владельца
04.10), а взгляд зрителя ждёт смены каждые 2-4 с. Поэтому внутри кадра:
  * подписи схемы появляются в момент, когда голос их произносит (схема
    собирается по частям вместе с голосом, а не висит готовой);
  * быстрый наезд на предмет в момент его слова (camera.punch_window):
    0.33 с с торможением, держим, назад — склейкой; не чаще PUNCH_GAP_SEC
    по всему ролику, иначе приём перестаёт работать;
  * длинный план (дольше MAX_VIEW_SEC) делится склейкой на другой план того
    же рисунка: общий <-> средний на главном предмете. Склейка — по началу
    слова, не короче MIN_VIEW_SEC и не «скачком» (camera.is_jump);
  * главная мысль пишется карандашом (writeon.py) в момент своего слова; пока
    она пишется — ни склеек, ни наездов (рука не прыгает вместе с кадром).
План — чистые данные (окна в координатах холста и время от начала кадра),
рисует его frame_clip.py."""
import numpy as np
from scipy import ndimage

import camera
import words as wordsmod

PUNCH_GAP_SEC = 15.0
PUNCH_HOLD_SEC = 1.2
MAX_VIEW_SEC = 4.0
MIN_VIEW_SEC = 1.5
LABEL_FADE_SEC = 0.15
CUT_SNAP_SEC = 0.6
MEDIUM_Z = (1.65, 1.95)        # от общего плана не меньше camera.CUT_MIN_RATIO с запасом на дрейф
WRITE_TAIL_SEC = 0.6
KEY_MAX_LEAD_SEC = 1.5   # мысль может начать писаться раньше своего слова не больше чем на 1.5 с
KEY_HOLD_SEC = 1.5       # дописанная мысль стоит на экране не меньше (критики 04.10: стояла 0.75 с)
PUSH_SEC = 0.6           # наклон камеры к части схемы, названной голосом
PUSH_STEP = 0.025        # каждый наклон — на 2.5% крупнее,
PUSH_MAX = 0.08          # всего не больше 8% (внутри плана — без «большого зума»)
PUSH_PULL = 0.3          # центр сдвигается к части на эту долю пути


def word_time(phrase, words, after=0.0):
    """Время (от начала кадра) первого произнесённого слова фразы phrase не раньше after.
    words — [{"word", "start"}]. Ищется первое значимое (3+ букв) слово фразы."""
    for t in wordsmod.significant(phrase):
        for w in words:
            if w["start"] >= after - 1e-6 and wordsmod.same(t, wordsmod.norm(w["word"])):
                return w["start"]
    return None


def subject_box(busy, objects):
    """Главный предмет: объект с ролью subject, иначе самый большой нарисованный участок."""
    for o in objects or []:
        if o.get("role") == "subject" and o.get("box"):
            return tuple(o["box"])
    lab, n = ndimage.label(busy > 0.3)
    if not n:
        return None
    sizes = ndimage.sum(np.ones_like(busy), lab, range(1, n + 1))
    sl = ndimage.find_objects(lab)[int(np.argmax(sizes))]
    return (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)


def plan(D, busy, words, labels=(), objects=(), key=None, key_dur=0.0, last_punch=-1e9, T0=0.0,
         zoom_in=True, parts=None):
    """Сегменты камеры и события кадра.

    D — длительность кадра; busy — карта занятости холста; words — слова речи
    кадра со временем от его начала; labels — [{"text", ...}]; objects —
    [{"name", "box", "word", "role"}] в координатах холста; key — текст главной
    мысли, key_dur — сколько он пишется; last_punch — глобальное время прошлого
    наезда; T0 — глобальное время начала кадра; parts — рамка (в координатах
    холста) части схемы, которую называет каждая подпись: на её слове камера
    чуть наклоняется к ней.

    Возвращает dict: segments [{t0, t1, kind: drift|punch, win, win_to, zoom_in}],
    label_times [t], key_time (или None), punch_at (глобальное время или None), notes."""
    SH, SW = busy.shape
    wide = camera.window(SW/2, SH/2, SW, SW, SH)
    notes = []

    # подписи — в момент слова; не нашлось слова — по очереди в первой половине кадра
    n = len(labels)
    label_times, prev = [], 0.25
    for i, lb in enumerate(labels):
        t = word_time(lb.get("text", ""), words, after=0.0)
        if t is None or t > D - 0.8:
            t = min(D - 0.8, 0.35 + i*min(0.9, 0.5*D/max(1, n)))
            notes.append(f"подпись «{lb.get('text', '')}» без своего слова — по очереди")
        t = max(t, prev)
        label_times.append(float(max(0.0, t)))
        prev = t + 0.25
    labels_done = (max(label_times) + 1.0) if label_times else 0.0

    # главная мысль
    key_time, busy_win = None, []
    if key:
        kt = word_time(key, words) if words else None
        said = kt
        kt = 0.3 if kt is None else kt
        if kt + key_dur + KEY_HOLD_SEC > D - 0.2:
            kt = max(0.2, D - 0.2 - KEY_HOLD_SEC - key_dur)
        if said is not None and said - kt > KEY_MAX_LEAD_SEC:
            notes.append("главная мысль не успевает дописаться и постоять к концу кадра — не пишется")
        elif kt + key_dur + KEY_HOLD_SEC <= D - 0.2 + 1e-6:
            key_time = kt
            busy_win.append((kt - 0.2, kt + key_dur + WRITE_TAIL_SEC))
        else:
            notes.append("главная мысль не успевает дописаться — не пишется")

    def free(t0, t1):
        return all(t1 <= a or t0 >= b for a, b in busy_win)

    # наезд на предмет
    punch = None
    for o in objects or []:
        if not o.get("word") or not o.get("box") or o.get("role") not in (None, "zoom"):
            continue
        t = word_time(o["word"], words)
        if t is None:
            notes.append(f"наезд на «{o.get('name')}»: слово «{o['word']}» не прозвучало")
            continue
        if T0 + t - last_punch < PUNCH_GAP_SEC:
            notes.append(f"наезд на «{o.get('name')}» пропущен: прошлый был {T0 + t - last_punch:.1f} с назад")
            continue
        if t < 0.3 or t + camera.PUNCH_SEC + PUNCH_HOLD_SEC > D:
            notes.append(f"наезд на «{o.get('name')}» не помещается в кадр")
            continue
        if any(lt > t for lt in label_times):
            notes.append(f"наезд на «{o.get('name')}» до появления всех подписей — подписи ушли бы за кадр")
            continue
        if not free(t, t + camera.PUNCH_SEC + PUNCH_HOLD_SEC):
            notes.append(f"наезд на «{o.get('name')}» во время письма — пропущен")
            continue
        pw = camera.punch_window(busy, tuple(o["box"]))
        if pw is None:
            notes.append(f"наезд на «{o.get('name')}» не делается: {getattr(camera.punch_window, 'why', '')}")
            continue
        punch = (t, pw, o.get("name"))
        break

    starts = [w["start"] for w in words if 0 < w["start"] < D]

    def snap(t, lo, hi):
        """Склейка — на начале ближайшего слова в пределах [lo, hi]."""
        cand = [s for s in starts if lo <= s <= hi and abs(s - t) <= CUT_SNAP_SEC and free(s - 0.05, s + 0.05)]
        return min(cand, key=lambda s: abs(s - t)) if cand else (t if free(t - 0.05, t + 0.05) else None)

    subj = subject_box(busy, objects)
    # средний план, режущий соседей, хуже, чем никакого; поле вокруг предмета даёт ореол карты
    # занятости (placement.busy_map), отдельный отступ margin его только удваивал и отсекал годные планы
    medium = (camera.frame_for(busy, subj, MEDIUM_Z, margin=0.0, spread=0.35, max_cross=camera.SEPARATE_MAX_CROSS,
                                grid=17, cross_map=camera.others(busy, subj))
              if subj else None)
    if medium is None and subj is not None and not any(o.get("role") == "subject" and o.get("box")
                                                       for o in objects or []):
        # главный предмет не размечен, а рисунок — одна сплошная группа (обычный случай: кот, стол и
        # следы слиты), и средний план «целиком на группу» не помещается. Тогда — на центр тяжести
        # рисунка: там почти всегда главное, край своей же группы обрезать можно.
        ys, xs = np.nonzero(busy > 0.3)
        if len(xs):
            wgt = busy[ys, xs]
            cx, cy = float((xs*wgt).sum()/wgt.sum()), float((ys*wgt).sum()/wgt.sum())
            bw, bh = 0.32*SW, 0.32*SH
            core = (cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2)
            medium = camera.frame_for(busy, core, MEDIUM_Z, margin=0.0, spread=0.2, max_cross=camera.SEPARATE_MAX_CROSS,
                                      grid=13, cross_map=camera.others(busy, subj))
    if medium is not None and camera.is_jump(medium, wide, SW, SH):
        medium = None

    # опорные склейки: наезд и возврат
    cuts = []                                     # (t, kind, win)
    if punch:
        t, pw, name = punch
        cuts.append((t, "punch", pw))
        ret = t + camera.PUNCH_SEC + PUNCH_HOLD_SEC
        if D - ret >= MIN_VIEW_SEC:
            r = snap(ret + 0.2, ret, D - MIN_VIEW_SEC)
            if r is not None:
                cuts.append((r, "cut", wide))

    # длинные планы — склейкой на другой план (не раньше, чем все подписи появились)
    def split(t0, t1, win):
        """Равные куски не длиннее MAX_VIEW_SEC, склейки на началах слов."""
        out, cur = [], win
        if medium is None:
            return out
        while t1 - t0 > MAX_VIEW_SEC:
            k = int(np.ceil((t1 - t0)/MAX_VIEW_SEC))
            t = t0 + (t1 - t0)/k
            lo = max(t0 + MIN_VIEW_SEC, labels_done)
            if lo > t1 - MIN_VIEW_SEC:
                break
            s_ = snap(max(t, lo), lo, t1 - MIN_VIEW_SEC)
            if s_ is None:
                break
            cur = medium if cur == wide else wide
            out.append((s_, "cut", cur))
            t0 = s_
        return out

    bounds = [(0.0, "start", wide)] + sorted(cuts) + [(D, "end", None)]
    extra = []
    for (a, ka, wa), (b, kb, _) in zip(bounds, bounds[1:]):
        if ka in ("start", "cut"):
            extra += split(a, b, wa)
    bounds = sorted(bounds[:-1] + extra) + [(D, "end", None)]

    segs, cur, zi = [], wide, zoom_in
    for (a, ka, wa), (b, _, _) in zip(bounds, bounds[1:]):
        if b - a <= 1e-6:
            continue
        if ka == "punch":
            segs.append(dict(t0=a, t1=a + camera.PUNCH_SEC, kind="punch", win=cur, win_to=wa))
            segs.append(dict(t0=a + camera.PUNCH_SEC, t1=b, kind="hold", win=wa))
            cur = wa
        else:
            if ka == "cut" and camera.is_jump(cur, wa, SW, SH):
                notes.append(f"склейка {a:.2f} с отменена: скачок")
                if segs:
                    segs[-1]["t1"] = b
                continue
            cur = wa
            segs.append(dict(t0=a, t1=b, kind="drift", win=wa, zoom_in=zi))
            zi = not zi
    # схема собирается по голосу: на каждой названной части камера чуть наклоняется к ней
    for sg in segs:
        if sg["kind"] != "drift" or sg["win"] != wide:
            continue
        pushes, k = [], 0
        for i, lt in enumerate(label_times):
            if not (sg["t0"] <= lt < sg["t1"] - PUSH_SEC):
                continue
            box = parts[i] if parts and i < len(parts) else None
            if not box:
                continue
            z = 1 + min(PUSH_MAX, PUSH_STEP*(k + 1))
            cx = SW/2 + ((box[0] + box[2])/2 - SW/2)*PUSH_PULL
            cy = SH/2 + ((box[1] + box[3])/2 - SH/2)*PUSH_PULL
            win = camera.window(cx, cy, (wide[2] - wide[0])/z, SW, SH)
            if camera.edge_cross(busy, win) > camera.PUNCH_MAX_CROSS:
                continue                         # наклон резал бы рисунок — эта часть без него
            pushes.append(dict(t=lt, win=win)); k += 1
        if pushes:
            sg["pushes"] = pushes
    for sg in segs:
        if sg["t1"] - sg["t0"] > MAX_VIEW_SEC + CUT_SNAP_SEC and sg["kind"] == "drift" and not sg.get("pushes"):
            notes.append(f"план {sg['t0']:.1f}-{sg['t1']:.1f} с без смены: другого плана без разреза рисунка нет")
    return dict(segments=segs, label_times=label_times, key_time=key_time,
                punch_at=(T0 + punch[0]) if punch else None, punch_name=punch[2] if punch else None,
                notes=notes)


def window_at(p, t, SW, SH):
    """Окно камеры в момент t (от начала кадра), всегда внутри холста."""
    w = _window_at(p, t, SW, SH)
    return camera.window((w[0] + w[2])/2, (w[1] + w[3])/2, w[2] - w[0], SW, SH)


def _window_at(p, t, SW, SH):
    segs = p["segments"]
    s = next((s for s in segs if s["t0"] <= t < s["t1"]), segs[-1])
    u = (t - s["t0"])/max(1e-6, s["t1"] - s["t0"])
    if s["kind"] == "punch":
        return camera.lerp(s["win"], s["win_to"], camera.ease_out(u))
    if s["kind"] == "hold":
        cx, cy = (s["win"][0] + s["win"][2])/2, (s["win"][1] + s["win"][3])/2
        z = 1 + 0.5*camera.DRIFT*camera.ease_io(u)        # после наезда — еле заметно дальше
        return camera.window(cx, cy, (s["win"][2] - s["win"][0])/z, SW, SH)
    if s.get("pushes"):
        cur = s["win"]
        for k, ps in enumerate(s["pushes"]):
            if t < ps["t"]:
                break
            a = camera.ease_io((t - ps["t"])/PUSH_SEC)
            cur = camera.lerp(cur, ps["win"], a) if a < 1 else ps["win"]
        last = s["pushes"][-1]
        rest_t0 = last["t"] + PUSH_SEC
        if t > rest_t0 and s["t1"] - rest_t0 > 0.3:
            # после сборки — медленный выдох обратно к общему плану, без остановки камеры
            u2 = camera.ease_io((t - rest_t0)/(s["t1"] - rest_t0))
            cur = camera.lerp(last["win"], s["win"], 0.6*u2)
        return cur
    return camera.drift(s["win"], u, s.get("zoom_in", True), SW, SH)
