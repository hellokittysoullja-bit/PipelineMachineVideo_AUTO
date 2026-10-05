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
MAX_VIEW_SEC = 3.5      # брендбук: план 1,2-3,5 с (критик 05.10: планы по 7-8 с, зритель уходит)
MIN_VIEW_SEC = 1.5
HEAD_SLIVER = 0.05      # крупный план детали может захватить краешек головы героя (кончик уха) — до этой доли её рамки (0.15 захватил полглаза, живой прогон 05.10)
DETAIL_FILL = 0.3       # деталь в крупном плане — не меньше этой доли кадра по своей тесной стороне (мельче — это уже пятнышко)
LABEL_FADE_SEC = 0.15
CUT_SNAP_SEC = 0.6
MEDIUM_STEP = (1.55, 2.1)      # средний план крупнее общего во столько раз (не меньше camera.CUT_MIN_RATIO + дрейф)
MEDIUM_MIN_FILL = 0.3        # предмет среднего плана — не меньше трети кадра по одной из сторон
MEDIUM_MARGIN = 0.06           # поле вокруг предмета в среднем плане (критик: уши кота срезаны; больше — соседний предмет не даёт плана вовсе)
WIDE_MAX_Z = 1.45              # общий план — плотно по рисунку, а не весь лист (критик: кот на 10% кадра)
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
    """Главное в кадре: предмет фразы вместе с героем (их общая рамка), иначе самый
    большой нарисованный участок."""
    boxes = [o["box"] for o in objects or [] if o.get("role") in ("subject", "hero") and o.get("box")]
    if boxes:
        return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    lab, n = ndimage.label(busy > 0.3)
    if not n:
        return None
    sizes = ndimage.sum(np.ones_like(busy), lab, range(1, n + 1))
    sl = ndimage.find_objects(lab)[int(np.argmax(sizes))]
    return (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)


def plan(D, busy, words, labels=(), objects=(), key=None, key_dur=0.0, last_punch=-1e9, T0=0.0,
         zoom_in=True, parts=None, accent=None, accent_dur=0.0):
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
    ys_, xs_ = np.nonzero(busy > 0.3)
    if len(xs_) and not (key or accent):      # мысли и акценту нужна пустая бумага — общий план целиком
        # общий план — весь рисунок целиком, но без пустой бумаги вокруг
        tight = camera.frame_for(busy, (xs_.min(), ys_.min(), xs_.max(), ys_.max()), (1.0, WIDE_MAX_Z),
                                 margin=0.02, spread=0.1)
        if tight is not None:
            wide = tight
    wz = camera.zoom_of(wide, SW, SH)
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

    # акцент — 1-3 слова фразы, тоже от руки карандашом (брендбук: на экране только рукописные
    # ключевые слова, со звуком карандаша) на своём слове; не поверх главной мысли; дописанный
    # стоит не меньше KEY_HOLD_SEC; пока пишется — ни склеек, ни наездов
    accent_time = None
    if accent:
        at = word_time(accent, words)
        if at is None:
            notes.append(f"акцент «{accent}»: слово не прозвучало")
        elif at + accent_dur + KEY_HOLD_SEC > D - 0.2 + 1e-6:
            notes.append(f"акцент «{accent}» не успевает дописаться и постоять к концу кадра")
        elif not free(at - 0.5, at + accent_dur + WRITE_TAIL_SEC):
            notes.append(f"акцент «{accent}» пропущен: пишется главная мысль")
        else:
            accent_time = at
            busy_win.append((at - 0.2, at + accent_dur + WRITE_TAIL_SEC))

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
        # сначала — наезд, не задевающий героя даже усами; нет такого — обычный (наезд важнее
        # кончика уса у края: решение владельца — быстрый наезд на предмет держим)
        heroes = [tuple(h["box"]) for h in objects or [] if h.get("role") == "hero" and h.get("box")]
        pw = (camera.punch_window(busy, tuple(o["box"]), keep=heroes) if heroes else None) \
            or camera.punch_window(busy, tuple(o["box"]))
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

    gaps = {w["start"]: w["start"] - p_["end"] for p_, w in zip(words, words[1:]) if 0 < w["start"] < D}

    def phrase_cut(lo, hi):
        """Начало слова после самой длинной паузы в [lo, hi]; при равных — ближе к середине окна."""
        cand = [s for s in starts if lo <= s <= hi and free(s - 0.05, s + 0.05)]
        if not cand:
            return None
        mid = (lo + hi)/2
        return max(cand, key=lambda s: (round(gaps.get(s, 0.0), 2), -abs(s - mid)))

    subj = subject_box(busy, objects)
    # средний план, режущий соседей, хуже, чем никакого; поле вокруг предмета даёт ореол карты
    # занятости (placement.busy_map), отдельный отступ margin его только удваивал и отсекал годные планы
    mz = (wz*MEDIUM_STEP[0], min(camera.PUNCH_MAX_ZOOM, wz*MEDIUM_STEP[1]))
    marked = any(o.get("role") in ("subject", "hero") and o.get("box") for o in objects or [])
    # у рамки от судьи поля нет — нужен отступ; у пятна карты занятости поле уже есть (ореол).
    # Предмет фразы с героем не влезает в средний план — тогда сам предмет фразы, потом герой.
    tries = [subj] + [tuple(o["box"]) for r in ("subject", "hero") for o in objects or []
                      if o.get("role") == r and o.get("box")] if subj else []
    medium = None
    keep = [tuple(o["box"]) for o in objects or [] if o.get("role") in ("subject", "hero") and o.get("box")]
    for tgt in tries:
        cm = camera.others(busy, tgt)
        for kb in keep:
            # герой и предмет фразы — целиком в кадре или целиком за ним, даже если слиты с группой:
            # средний план по мозгу срезал коту морду (живой прогон 04.10)
            if not (kb[0] >= tgt[0] - 1 and kb[1] >= tgt[1] - 1 and kb[2] <= tgt[2] + 1 and kb[3] <= tgt[3] + 1):
                ys_k, xs_k = slice(int(kb[1]), int(kb[3])), slice(int(kb[0]), int(kb[2]))
                cm[ys_k, xs_k] = np.maximum(cm[ys_k, xs_k], busy[ys_k, xs_k])
        kw = dict(margin=MEDIUM_MARGIN if marked else 0.0, spread=0.35, max_cross=camera.SEPARATE_MAX_CROSS,
                  grid=17, cross_map=cm)
        medium = camera.frame_for(busy, tgt, mz, away=((wide[0] + wide[2])/2, 0, 0.1*(wide[2] - wide[0])), **kw) \
            or camera.frame_for(busy, tgt, mz, **kw)
        if medium is not None and max((tgt[2] - tgt[0])/(medium[2] - medium[0]),
                                      (tgt[3] - tgt[1])/(medium[3] - medium[1])) < MEDIUM_MIN_FILL:
            medium = None       # мелкий предмет на пустом листе — не средний план (это работа быстрого наезда)
        if medium is not None:
            break
    if medium is None and subj is not None and not marked:
        # главный предмет не размечен, а рисунок — одна сплошная группа (обычный случай: кот, стол и
        # следы слиты), и средний план «целиком на группу» не помещается. Тогда — на центр тяжести
        # рисунка: там почти всегда главное, край своей же группы обрезать можно.
        ys, xs = np.nonzero(busy > 0.3)
        if len(xs):
            wgt = busy[ys, xs]
            cx, cy = float((xs*wgt).sum()/wgt.sum()), float((ys*wgt).sum()/wgt.sum())
            bw, bh = 0.32*SW, 0.32*SH
            # верх группы в кадре: у персонажа там голова и уши (критик: срезанные уши на 35 с)
            band = ys[(np.abs(xs - cx) < bw/2) & (busy[ys, xs] > 0.3)]
            top = float(band.min()) if len(band) else cy - bh/2
            top = max(min(top, cy - bh/2), cy - 0.45*SH/wz)     # голова целиком, но не весь рост сцены
            core = (cx - bw/2, top, cx + bw/2, cy + bh/2)
            medium = camera.frame_for(busy, core, mz, margin=0.02, spread=0.2, max_cross=camera.SEPARATE_MAX_CROSS,
                                      grid=13, cross_map=camera.others(busy, subj))
    if medium is not None and camera.is_jump(medium, wide, SW, SH):
        medium = None

    # крупные планы деталей рисунка (цепь с глыбой, дымящийся хвост): то, что монтажёр
    # снял бы отдельными планами. Деталь — часть общего рисунка, поэтому край кадра может
    # идти по рисунку (край соседней части в крупном плане детали — норма); голова героя —
    # целиком в кадре, целиком за ним или только краешком (до HEAD_SLIVER её рамки: на живых
    # кадрах огонёк и глыба нарисованы вплотную к уху). Головы нет в разметке (старый
    # кадр) — так же держится весь герой.
    heads = [tuple(o["box"]) for o in objects or [] if o.get("role") == "hero_head" and o.get("box")]
    guard = heads or [tuple(o["box"]) for o in objects or [] if o.get("role") == "hero" and o.get("box")]

    def head_ok(win, db):
        for kb in guard:
            if db[0] >= kb[0] - 1 and db[1] >= kb[1] - 1 and db[2] <= kb[2] + 1 and db[3] <= kb[3] + 1:
                continue                                  # деталь — часть головы: резать вокруг неё можно
            ix = max(0.0, min(win[2], kb[2]) - max(win[0], kb[0]))
            iy = max(0.0, min(win[3], kb[3]) - max(win[1], kb[1]))
            area = max(1.0, (kb[2] - kb[0])*(kb[3] - kb[1]))
            whole = win[0] <= kb[0] and win[1] <= kb[1] and win[2] >= kb[2] and win[3] >= kb[3]
            if not whole and ix*iy > HEAD_SLIVER*area:
                return False
        return True

    details = []
    for o in objects or []:
        if o.get("role") != "detail" or not o.get("box"):
            continue
        db = tuple(o["box"])
        # обычный крупный план — от 1.5x общего; крупная деталь, которая в него не влезает, — менее
        # крупно, но только со сдвигом центра (иначе склейка читается как «скачок», camera.is_jump)
        kw = dict(margin=MEDIUM_MARGIN, spread=0.25, grid=13)
        dz = (wz*camera.CUT_MIN_RATIO*1.02, camera.PUNCH_MAX_ZOOM)
        dz_lo = (wz*1.15, wz*camera.CUT_MIN_RATIO*1.02)
        dw = camera.frame_for(busy, db, dz, accept=lambda w, db=db: head_ok(w, db), **kw) or \
            camera.frame_for(busy, db, dz_lo, accept=lambda w, db=db: head_ok(w, db) and not camera.is_jump(w, wide, SW, SH),
                             **kw)
        if dw is None:
            any_ = camera.frame_for(busy, db, dz, **kw) or camera.frame_for(
                busy, db, dz_lo, accept=lambda w: not camera.is_jump(w, wide, SW, SH), **kw)
            why = "разрезал бы голову героя" if any_ else "деталь не помещается в план"
            notes.append(f"крупный план «{o.get('name')}» не делается: {why}")
            continue
        if max((db[2] - db[0])/(dw[2] - dw[0]), (db[3] - db[1])/(dw[3] - dw[1])) < DETAIL_FILL:
            notes.append(f"крупный план «{o.get('name')}» не делается: деталь мелкая даже на пределе зума")
            continue
        if camera.is_jump(dw, wide, SW, SH) or any(camera.is_jump(dw, v, SW, SH) for v in details + [medium] if v):
            notes.append(f"крупный план «{o.get('name')}» не делается: почти совпадает с другим планом")
            continue
        details.append(dw)

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
    # порядок планов как у монтажёра: общий -> средний -> детали подряд -> другой общий; повтор
    # плана — только когда новых не осталось (критик 05.10: «ровно тот же общий план читается
    # как повтор»). Другой общий — чуть крупнее и со сдвигом к главному, голова героя целиком.
    views = [v for v in [medium] + details if v is not None]
    wide_alt = None
    if views and subj is not None:
        fx, fy = (subj[0] + subj[2])/2, (subj[1] + subj[3])/2
        wcx, wcy = (wide[0] + wide[2])/2, (wide[1] + wide[3])/2
        for zf in (1.25, 1.18, 1.12, 0.9, 0.84):      # крупнее или чуть общее общего плана
            for k in (0.5, 0.3, 0.0, -0.3):
                w = camera.window(wcx + (fx - wcx)*k, wcy + (fy - wcy)*k, (wide[2] - wide[0])/zf, SW, SH)
                # в него можно склеиться из крупных планов без «скачка»
                if head_ok(w, (0, 0, 0, 0)) and not any(camera.is_jump(w, v, SW, SH) for v in views):
                    wide_alt = w
                    break
            if wide_alt is not None:
                break
    cycle = [wide] + views + ([wide_alt] if wide_alt is not None else [])
    used = {0}
    order = {"i": 0}

    def next_view(cur):
        n_ = len(cycle)
        for fresh in (True, False):
            for step in range(1, n_ + 1):
                j = (order["i"] + step) % n_
                v = cycle[j]
                if (fresh and j in used) or v == cur or camera.is_jump(cur, v, SW, SH):
                    continue
                order["i"] = j
                used.add(j)
                return v
        return wide

    def split(t0, t1, win):
        """Равные куски не длиннее MAX_VIEW_SEC, склейки на началах слов."""
        out, cur = [], win
        if not views:
            return out
        while t1 - t0 > MAX_VIEW_SEC:
            lo = max(t0 + MIN_VIEW_SEC, labels_done)
            hi = min(t0 + MAX_VIEW_SEC, t1 - MIN_VIEW_SEC)
            if lo > t1 - MIN_VIEW_SEC:
                break
            # склейка — на паузе речи (самый длинный промежуток между словами в допустимом окне):
            # длина плана идёт от фразы, а не ровными кусками (брендбук: «не ровно 2,0 с —
            # ровный ритм усыпляет»; критик 05.10: все планы по 2-3 с)
            # пауза ищется около «ровной» точки: кусков столько, сколько нужно, а не больше —
            # лишний кусок повторял бы план, когда новых у картинки уже нет (живой прогон 05.10)
            k = int(np.ceil((t1 - t0)/MAX_VIEW_SEC))
            ideal = t0 + (t1 - t0)/k
            s_ = phrase_cut(max(lo, ideal - 0.8), min(hi, ideal + 0.8)) if hi >= lo else None
            if s_ is None:
                k = int(np.ceil((t1 - t0)/MAX_VIEW_SEC))
                s_ = snap(max(t0 + (t1 - t0)/k, lo), lo, t1 - MIN_VIEW_SEC)
            if s_ is None:
                break
            nxt = next_view(cur)
            if nxt != wide and accent_time is not None and accent_time - MAX_VIEW_SEC - CUT_SNAP_SEC <= s_ <= accent_time + 1.5:
                break           # акцент встаёт на общем плане: на среднем ему нет места (живой прогон)
            cur = nxt
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
            wcx, wcy = (wide[0] + wide[2])/2, (wide[1] + wide[3])/2
            cx = wcx + ((box[0] + box[2])/2 - wcx)*PUSH_PULL
            cy = wcy + ((box[1] + box[3])/2 - wcy)*PUSH_PULL
            win = camera.window(cx, cy, (wide[2] - wide[0])/z, SW, SH)
            if camera.edge_cross(busy, win) > camera.PUNCH_MAX_CROSS:
                continue                         # наклон резал бы рисунок — эта часть без него
            pushes.append(dict(t=lt, win=win)); k += 1
        if pushes:
            sg["pushes"] = pushes
    # длинный план без смены и без сборки — медленный наезд к главному на всю длину (до 8%, без разреза);
    # возврат к общему плану после среднего — тоже в движении, а не тот же стоп-кадр (критик: «jump-back»)
    focus = subj
    for k_, sg in enumerate(segs):
        back = k_ > 0 and sg["win"] == wide and segs[k_ - 1]["win"] != wide   # возврат к общему после среднего
        if (sg["kind"] == "drift" and not sg.get("pushes") and focus is not None
                and (sg["t1"] - sg["t0"] > MAX_VIEW_SEC + CUT_SNAP_SEC or back)):
            w0 = sg["win"]
            wcx, wcy = (w0[0] + w0[2])/2, (w0[1] + w0[3])/2
            fcx, fcy = (focus[0] + focus[2])/2, (focus[1] + focus[3])/2
            lean = camera.window(wcx + (fcx - wcx)*0.4, wcy + (fcy - wcy)*0.4, (w0[2] - w0[0])/(1 + PUSH_MAX), SW, SH)
            if camera.edge_cross(camera.others(busy, focus), lean, level=0.15) <= camera.SEPARATE_MAX_CROSS:
                sg["lean"] = lean
    for sg in segs:
        if sg["t1"] - sg["t0"] > MAX_VIEW_SEC + CUT_SNAP_SEC and sg["kind"] == "drift" and not sg.get("pushes") \
                and not sg.get("lean"):
            notes.append(f"план {sg['t0']:.1f}-{sg['t1']:.1f} с без смены: другого плана без разреза рисунка нет")
    return dict(segments=segs, label_times=label_times, key_time=key_time, accent_time=accent_time,
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
    if s.get("lean"):
        return camera.lerp(s["win"], s["lean"], camera.ease_io(u))
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
