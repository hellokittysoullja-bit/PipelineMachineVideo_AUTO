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
import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import camera
import words as wordsmod

PUNCH_GAP_SEC = 15.0
PUNCH_HOLD_SEC = 1.2
MAX_VIEW_SEC = 3.5      # брендбук: план 1,2-3,5 с (критик 05.10: планы по 7-8 с, зритель уходит)
HOOK_FIRST_VIEW_SEC = 2.0   # планы в зоне хука — не дольше (брендбук: хук 1,2–2,0 с; живой эп.01: 8,25 с до первой склейки)
HOOK_ZONE_SEC = 3.0         # зона хука — первые секунды ролика (та же, что у приёмки montage_qc)
MIN_VIEW_SEC = 1.5
HEAD_SLIVER = 0.05      # крупный план детали может захватить краешек головы героя (кончик уха) — до этой доли её рамки
                        # (0.15 захватил полглаза, живой прогон 05.10)
HEAD_MAX_BOTTOM = 0.82  # голова героя в плане — целиком выше зоны плеера телефона (нижние 18% кадра): сравнение с эп.01 08.10 —
                        # подбородок на 0.98 и 1.03 высоты окна в среднем плане, лицо под плеером
BODY_CROSS_MAX = 0.05   # окно захватывает больше этой доли ЧЕРНИЛ героя без головы в кадре — «обрубок» (эп.01: туловище без
                        # головы в правом верхнем углу плана конверта, 2.5 с)
DETAIL_TIGHT_FILL = 0.5   # деталь внутри головы режет голову вокруг себя, только если сама занимает не меньше половины кадра —
                          # осознанный макро-план (глаза), а не лицо со срезанным подбородком
WRITE_LEAD_SEC = 0.3      # склейка перед письмом — не позже чем за столько до первого штриха
JITTER_SEC = 1.0          # разброс длин планов без таймингов речи: ±0.5 с от ровной точки
VIEW_EDGE_MARGIN_SEC = 2/24  # план без события короче нормы на два кадра: ровно 3.50 с на границе блока приёмки (3.54)
TEXT_EDGE_OVERLAP = 0.02  # надпись в плане целиком или вне его: больше этой доли её рамки на краю — обрезанная надпись
VIEW_MIN_INK = 0.10     # план не общий: доля чернил в окне не меньше этой
VIEW_MIN_INK_RATIO = 0.8  # и не меньше этой доли плотности общего плана — крупный план на пустой бумаге это не крупный план
                          # (эп.01: план хвоста 3.2 с, чернил 0.47 от общего; годные планы того же эпизода 1.3–2.7)
DETAIL_FILL = 0.3       # деталь в крупном плане — не меньше этой доли кадра по своей тесной стороне (мельче — это уже пятнышко)
LABEL_FADE_SEC = 0.15
CUT_SNAP_SEC = 0.6
MEDIUM_STEP = (1.55, 2.1)      # средний план крупнее общего во столько раз (не меньше camera.CUT_MIN_RATIO + дрейф)
MEDIUM_MIN_FILL = 0.3        # предмет среднего плана — не меньше трети кадра по одной из сторон
MEDIUM_MARGIN = 0.06           # поле вокруг предмета в среднем плане (критик: уши кота срезаны;
                               # больше — соседний предмет не даёт плана вовсе)
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


def make_view_ok(busy, objects, wide, text_boxes=()):
    """Проверка плана камеры (не общего) по правилам монтажа с персонажем — одна и та же для среднего
    плана, крупных планов деталей и второго общего (и для тестов). Возвращает функцию
    view_ok(win, db=None, ink=True); у неё атрибут wide_ink — доля чернил общего плана.
    text_boxes — рамки рукописных надписей кадра (мысль, акцент): надпись в плане либо целиком, либо
    её нет вовсе — обрезанная до «только от» надпись висела 2,3 с (эп.01 08.10, второй прогон)."""
    SH, SW = busy.shape
    ink_map = (busy > 0.3).astype(np.float64)
    ink_int = np.zeros((SH + 1, SW + 1)); ink_int[1:, 1:] = ink_map.cumsum(0).cumsum(1)

    def ink_share(win):
        x0, y0, x1, y1 = [int(round(v)) for v in win]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(SW, x1), min(SH, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(ink_int[y1, x1] - ink_int[y0, x1] - ink_int[y1, x0] + ink_int[y0, x0])/((x1 - x0)*(y1 - y0))
    wide_ink = ink_share(wide)
    hero_boxes = [tuple(o["box"]) for o in objects or [] if o.get("role") == "hero" and o.get("box")]

    def ink_count(x0, y0, x1, y1):
        x0, y0, x1, y1 = max(0, int(round(x0))), max(0, int(round(y0))), min(SW, int(round(x1))), min(SH, int(round(y1)))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(ink_int[y1, x1] - ink_int[y0, x1] - ink_int[y1, x0] + ink_int[y0, x0])
    hero_ink = {hb: max(1.0, ink_count(*hb)) for hb in hero_boxes}

    def hero_ink_share(win, hb):
        """Доля ЧЕРНИЛ героя, попавшая в окно (аудит 08.10: по площади рамки крупный план конверта
        «задевал тело» пустой бумагой между телом и хвостом, и камера повторяла планы)."""
        return ink_count(max(win[0], hb[0]), max(win[1], hb[1]), min(win[2], hb[2]), min(win[3], hb[3]))/hero_ink[hb]
    head_boxes = [tuple(o["box"]) for o in objects or [] if o.get("role") == "hero_head" and o.get("box")]

    def view_ok(win, db=None, ink=True):
        """План (не общий) годен: голова героя целиком в кадре и выше зоны плеера, либо целиком за
        кадром (краешек — до HEAD_SLIVER); тело героя не торчит обрубком без головы; чернил в окне
        не меньше, чем на общем плане. db — рамка детали, ради которой план: деталь внутри головы
        (глаза) или внутри тела (огонёк хвоста) режет вокруг себя своего хозяина законно."""
        for kb in head_boxes:
            if db is not None and camera.inside(db, kb):
                # деталь внутри головы (глаза): либо настоящий макро-план (деталь ≥ DETAIL_TIGHT_FILL кадра),
                # либо голова целиком и выше зоны плеера — «лицо без подбородка» (эп.01, план глаз 4.4–6.8 с) не план
                if camera.fill(db, win) >= DETAIL_TIGHT_FILL:
                    # макро режет голову вокруг детали законно; но голова, попавшая в кадр ЦЕЛИКОМ, всё равно
                    # не ниже зоны плеера (эп.01 кадр 3: макро глаз с головой целиком и подбородком на 0.83)
                    if camera.inside(kb, win) and (kb[3] - win[1])/max(1.0, win[3] - win[1]) > HEAD_MAX_BOTTOM:
                        return False
                    continue
            whole = camera.inside(kb, win)
            if whole:
                if (kb[3] - win[1])/max(1.0, win[3] - win[1]) > HEAD_MAX_BOTTOM:
                    return False
            elif camera.overlap_share(win, kb) > HEAD_SLIVER:
                return False
        for hb in hero_boxes:
            if db is not None and camera.inside(db, hb):
                continue
            if camera.inside(hb, win) or hero_ink_share(win, hb) <= BODY_CROSS_MAX:
                continue
            if not any(camera.inside(kb, win) for kb in head_boxes if camera.inside(kb, hb) or camera.overlap_share(hb, kb) > 0.5):
                return False
        for tb in text_boxes:
            if not camera.inside(tb, win) and camera.overlap_share(win, tb) > TEXT_EDGE_OVERLAP:
                return False
        if ink and ink_share(win) < max(VIEW_MIN_INK, VIEW_MIN_INK_RATIO*wide_ink):
            return False
        return True

    return view_ok


def _stable01(*key):
    """Детерминированное число в [0, 1) от чисел ключа: hash() у Python солится от запуска к запуску,
    repr(np.float64) отличается между версиями numpy — поэтому форматируются обычные float."""
    txt = "|".join(f"{float(v):.4f}" for v in key)
    return int(hashlib.sha256(txt.encode()).hexdigest()[:8], 16)/0x100000000


@dataclass
class Writing:
    """Рукописные события кадра: когда пишутся мысль и акцент, когда стоят на экране.
    busy_win — окна «пишется» (склейки и наезды запрещены), holds — окна «написано и стоит»
    (план обязан держать надпись целиком)."""
    key_time: float | None = None
    accent_time: float | None = None
    busy_win: list = field(default_factory=list)        # [(t0, t1)]
    holds: list = field(default_factory=list)           # [(kind, t_from, t_to, box)]

    def free(self, t0, t1):
        return all(t1 <= a or t0 >= b for a, b in self.busy_win)

    def held_boxes(self, t):
        return [bx for _, a, b, bx in self.holds if a - 1e-6 <= t < b - 1e-6]

    def hold_end(self, t):
        return max([b for _, a, b, _ in self.holds if a - 1e-6 <= t < b - 1e-6] or [t])

    def hold_starts(self):
        return [a for _, a, _, _ in self.holds]

    @staticmethod
    def shows(win, boxes):
        return all(camera.inside(b, win) for b in boxes)

    def record(self):
        return [dict(kind=k, t0=a, t1=b, box=list(bx)) for k, a, b, bx in self.holds]


class Speech:
    """Начала слов кадра — куда ложатся склейки."""

    def __init__(self, words, D, writing):
        self.starts = [w["start"] for w in words if 0 < w["start"] < D]
        self.gaps = {w["start"]: w["start"] - p_["end"] for p_, w in zip(words, words[1:]) if 0 < w["start"] < D}
        self.free = writing.free

    def snap(self, t, lo, hi):
        """Склейка — на начале ближайшего слова в пределах [lo, hi]; без слов — в самой точке, если она свободна."""
        cand = [s for s in self.starts if lo <= s <= hi and abs(s - t) <= CUT_SNAP_SEC and self.free(s - 0.05, s + 0.05)]
        return min(cand, key=lambda s: abs(s - t)) if cand else (t if self.free(t - 0.05, t + 0.05) else None)

    def phrase_cut(self, lo, hi):
        """Начало слова после самой длинной паузы в [lo, hi]; при равных — ближе к середине окна."""
        cand = [s for s in self.starts if lo <= s <= hi and self.free(s - 0.05, s + 0.05)]
        if not cand:
            return None
        mid = (lo + hi)/2
        return max(cand, key=lambda s: (round(self.gaps.get(s, 0.0), 2), -abs(s - mid)))


class ViewCycle:
    """Порядок планов как у монтажёра: общий -> средний -> детали подряд -> другой общий; повтор
    плана — только когда новых не осталось (критик 05.10: «ровно тот же общий план читается как повтор»)."""

    def __init__(self, wide, views, wide_alt, SW, SH):
        self.cycle = [wide] + views + ([wide_alt] if wide_alt is not None else [])
        self.wide, self.SW, self.SH = wide, SW, SH
        self.used, self.i = {0}, 0

    def next(self, cur, must=()):
        """Следующий план; must — рамки, которые обязаны быть в нём целиком (удерживаемая надпись).
        Нет ни одного подходящего — None."""
        n = len(self.cycle)
        for fresh in (True, False):
            for step in range(1, n + 1):
                j = (self.i + step) % n
                v = self.cycle[j]
                if (fresh and j in self.used) or v == cur or camera.is_jump(cur, v, self.SW, self.SH) \
                        or not Writing.shows(v, must):
                    continue
                self.i = j
                self.used.add(j)
                return v
        return self.wide if (self.wide != cur and Writing.shows(self.wide, must)) else None


def wide_view(busy, has_text):
    """Общий план: весь рисунок целиком, но без пустой бумаги вокруг; мысли и акценту нужна
    пустая бумага — тогда весь холст."""
    SH, SW = busy.shape
    wide = camera.full_window(SW, SH)
    ys, xs = np.nonzero(busy > 0.3)
    if len(xs) and not has_text:
        tight = camera.frame_for(busy, (xs.min(), ys.min(), xs.max(), ys.max()), (1.0, WIDE_MAX_Z), margin=0.02, spread=0.1)
        if tight is not None:
            wide = tight
    return wide


def schedule_labels(labels, words, D):
    """Подписи — в момент слова; не нашлось слова — по очереди в первой половине кадра. (времена, заметки)."""
    n = len(labels)
    times, prev, notes = [], 0.25, []
    for i, lb in enumerate(labels):
        t = word_time(lb.get("text", ""), words, after=0.0)
        if t is None or t > D - 0.8:
            t = min(D - 0.8, 0.35 + i*min(0.9, 0.5*D/max(1, n)))
            notes.append(f"подпись «{lb.get('text', '')}» без своего слова — по очереди")
        t = max(t, prev)
        times.append(float(max(0.0, t)))
        prev = t + 0.25
    return times, notes


def schedule_writing(D, words, key, key_dur, key_box, accent, accent_dur, accent_box):
    """Когда пишутся мысль и акцент. Мысль — на своём слове (без слов — с 0.3 с), не позже чем успеет
    дописаться и постоять; акцент — на своём слове, не поверх мысли. (Writing, заметки)."""
    wr, notes = Writing(), []
    if key:
        kt = word_time(key, words) if words else None
        said = kt
        kt = 0.3 if kt is None else kt
        if kt + key_dur + KEY_HOLD_SEC > D - 0.2:
            kt = max(0.2, D - 0.2 - KEY_HOLD_SEC - key_dur)
        if said is not None and said - kt > KEY_MAX_LEAD_SEC:
            notes.append("главная мысль не успевает дописаться и постоять к концу кадра — не пишется")
        elif kt + key_dur + KEY_HOLD_SEC <= D - 0.2 + 1e-6:
            wr.key_time = kt
            wr.busy_win.append((kt - 0.2, kt + key_dur + WRITE_TAIL_SEC))
        else:
            notes.append("главная мысль не успевает дописаться — не пишется")
    # акцент — 1-3 слова фразы, тоже от руки карандашом (брендбук: на экране только рукописные ключевые
    # слова, со звуком карандаша) на своём слове; не поверх главной мысли; дописанный стоит не меньше
    # KEY_HOLD_SEC; пока пишется — ни склеек, ни наездов
    if accent:
        at = word_time(accent, words)
        if at is None:
            notes.append(f"акцент «{accent}»: слово не прозвучало")
        elif at + accent_dur + KEY_HOLD_SEC > D - 0.2 + 1e-6:
            notes.append(f"акцент «{accent}» не успевает дописаться и постоять к концу кадра")
        elif not wr.free(at - 0.5, at + accent_dur + WRITE_TAIL_SEC):
            notes.append(f"акцент «{accent}» пропущен: пишется главная мысль")
        else:
            wr.accent_time = at
            wr.busy_win.append((at - 0.2, at + accent_dur + WRITE_TAIL_SEC))
    # удержание дописанного: пока мысль/акцент стоит на экране, план обязан держать надпись в кадре
    if wr.key_time is not None and key_box is not None:
        wr.holds.append(("key", wr.key_time, wr.key_time + key_dur + KEY_HOLD_SEC, tuple(key_box)))
    if wr.accent_time is not None and accent_box is not None:
        wr.holds.append(("accent", wr.accent_time, wr.accent_time + accent_dur + KEY_HOLD_SEC, tuple(accent_box)))
    return wr, notes


def choose_punch(objects, words, busy, D, T0, last_punch, label_times, writing):
    """Быстрый наезд на предмет фразы в момент его слова. ((t, окно, имя) | None, заметки)."""
    notes = []
    heroes = [tuple(h["box"]) for h in objects or [] if h.get("role") == "hero" and h.get("box")]
    for o in objects or []:
        if not o.get("word") or not o.get("box") or o.get("role") not in (None, "zoom"):
            continue
        name = o.get("name")
        t = word_time(o["word"], words)
        if t is None:
            notes.append(f"наезд на «{name}»: слово «{o['word']}» не прозвучало")
            continue
        if T0 + t - last_punch < PUNCH_GAP_SEC:
            notes.append(f"наезд на «{name}» пропущен: прошлый был {T0 + t - last_punch:.1f} с назад")
            continue
        end = t + camera.PUNCH_SEC + PUNCH_HOLD_SEC
        if t < 0.3 or end > D:
            notes.append(f"наезд на «{name}» не помещается в кадр")
            continue
        if any(lt > t for lt in label_times):
            notes.append(f"наезд на «{name}» до появления всех подписей — подписи ушли бы за кадр")
            continue
        if not writing.free(t, end):
            notes.append(f"наезд на «{name}» во время письма — пропущен")
            continue
        # сначала — наезд, не задевающий героя даже усами; нет такого — обычный (наезд важнее
        # кончика уса у края: решение владельца — быстрый наезд на предмет держим)
        pw, why = camera.punch_window(busy, tuple(o["box"]), keep=heroes) if heroes else (None, None)
        if pw is None:
            pw, why = camera.punch_window(busy, tuple(o["box"]))
        if pw is None:
            notes.append(f"наезд на «{name}» не делается: {why}")
            continue
        if any(not Writing.shows(pw, writing.held_boxes(tt)) for tt in np.arange(t, end, 0.1)):
            notes.append(f"наезд на «{name}» увёл бы дописанную надпись за кадр — пропущен")
            continue
        return (t, pw, name), notes
    return None, notes


def medium_view(busy, objects, subj, wide, wz, view_ok):
    """Средний план: предмет фразы с героем целиком (их общая рамка), иначе предмет, иначе герой; у
    неразмеченного рисунка — на центр тяжести. None — не нашлось без разреза соседей."""
    SH, SW = busy.shape
    # средний план, режущий соседей, хуже, чем никакого; поле вокруг предмета даёт ореол карты
    # занятости (placement.busy_map), отдельный отступ margin его только удваивал и отсекал годные планы
    mz = (wz*MEDIUM_STEP[0], min(camera.PUNCH_MAX_ZOOM, wz*MEDIUM_STEP[1]))
    marked = any(o.get("role") in ("subject", "hero") and o.get("box") for o in objects or [])
    # у рамки от судьи поля нет — нужен отступ; у пятна карты занятости поле уже есть (ореол).
    # Предмет фразы с героем не влезает в средний план — тогда сам предмет фразы, потом герой.
    tries = [subj] + [tuple(o["box"]) for r in ("subject", "hero") for o in objects or []
                      if o.get("role") == r and o.get("box")] if subj else []
    keep = [tuple(o["box"]) for o in objects or [] if o.get("role") in ("subject", "hero") and o.get("box")]
    medium = None
    for tgt in tries:
        cm = camera.others(busy, tgt)
        for kb in keep:
            # герой и предмет фразы — целиком в кадре или целиком за ним, даже если слиты с группой:
            # средний план по мозгу срезал коту морду (живой прогон 04.10)
            if not camera.inside(kb, tgt):
                ys_k, xs_k = slice(int(kb[1]), int(kb[3])), slice(int(kb[0]), int(kb[2]))
                cm[ys_k, xs_k] = np.maximum(cm[ys_k, xs_k], busy[ys_k, xs_k])
        kw = dict(margin=MEDIUM_MARGIN if marked else 0.0, spread=0.35, max_cross=camera.SEPARATE_MAX_CROSS,
                  grid=17, cross_map=cm, accept=view_ok)
        medium = camera.frame_for(busy, tgt, mz, away=(camera.center(wide)[0], 0.1*(wide[2] - wide[0])), **kw) \
            or camera.frame_for(busy, tgt, mz, **kw)
        if medium is not None and camera.fill(tgt, medium) < MEDIUM_MIN_FILL:
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
                                      grid=13, cross_map=camera.others(busy, subj), accept=view_ok)
    if medium is not None and camera.is_jump(medium, wide, SW, SH):
        medium = None
    return medium


def detail_views(busy, objects, wide, wz, medium, view_ok):
    """Крупные планы деталей рисунка (цепь с глыбой, дымящийся хвост) — то, что монтажёр снял бы
    отдельными планами; голова героя — тоже крупный план, последним в очереди (самая естественная
    склейка в анимации с персонажем). Край кадра может идти по рисунку (край соседней части в крупном
    плане — норма); герой — по правилам make_view_ok. (планы, заметки)."""
    SH, SW = busy.shape
    details, notes = [], []
    det_objs = [o for o in objects or [] if o.get("role") == "detail" and o.get("box")]
    det_objs += [dict(o, role="detail", name=o.get("name") or "the head of the hero")
                 for o in objects or [] if o.get("role") == "hero_head" and o.get("box")]

    def no_jump(w):
        return not camera.is_jump(w, wide, SW, SH)
    for o in det_objs:
        db = tuple(o["box"])
        # обычный крупный план — от 1.5x общего; крупная деталь, которая в него не влезает, — менее
        # крупно, но только со сдвигом центра (иначе склейка читается как «скачок», camera.is_jump)
        kw = dict(margin=MEDIUM_MARGIN, spread=0.25, grid=13)
        dz = (wz*camera.CUT_MIN_RATIO*1.02, camera.PUNCH_MAX_ZOOM)
        dz_lo = (wz*1.15, wz*camera.CUT_MIN_RATIO*1.02)
        dw = camera.frame_for(busy, db, dz, accept=lambda w, db=db: view_ok(w, db), **kw) or \
            camera.frame_for(busy, db, dz_lo, accept=lambda w, db=db: view_ok(w, db) and no_jump(w), **kw)
        if dw is None:
            any_ = camera.frame_for(busy, db, dz, **kw) or camera.frame_for(busy, db, dz_lo, accept=no_jump, **kw)
            if any_ is None:
                why = "деталь не помещается в план"
            elif not view_ok(any_, db, ink=False):
                why = "разрезал бы голову или тело героя"
            else:
                why = "вокруг детали пустая бумага"
            notes.append(f"крупный план «{o.get('name')}» не делается: {why}")
            continue
        if camera.fill(db, dw) < DETAIL_FILL:
            notes.append(f"крупный план «{o.get('name')}» не делается: деталь мелкая даже на пределе зума")
            continue
        if not no_jump(dw) or any(camera.is_jump(dw, v, SW, SH) for v in details + [medium] if v):
            notes.append(f"крупный план «{o.get('name')}» не делается: почти совпадает с другим планом")
            continue
        details.append(dw)
    return details, notes


def alt_wide(wide, subj, views, view_ok, SW, SH):
    """Другой общий план — чуть крупнее (или чуть общее) общего и со сдвигом к главному, в который
    можно склеиться из крупных планов без «скачка». None — нет."""
    if not views or subj is None:
        return None
    fx, fy = camera.center(subj)
    wcx, wcy = camera.center(wide)
    for zf in (1.25, 1.18, 1.12, 0.9):   # 0.84 не проходит порог чернил: 0.84² < VIEW_MIN_INK_RATIO
        for k in (0.5, 0.3, 0.0, -0.3):
            w = camera.window(wcx + (fx - wcx)*k, wcy + (fy - wcy)*k, (wide[2] - wide[0])/zf, SW, SH)
            if view_ok(w) and not any(camera.is_jump(w, v, SW, SH) for v in views):
                return w
    return None


def cut_timeline(D, cuts, wide, views, vc, writing, speech, labels_done, first_of_film):
    """Опорные склейки (наезд и возврат) плюс разрезка длинных планов: куски не длиннее MAX_VIEW_SEC
    (в зоне хука первого кадра — HOOK_FIRST_VIEW_SEC), склейки на началах слов, не раньше, чем все
    подписи появились, не во время письма, с надписью целиком в кадре, пока она стоит. (границы, заметки)."""
    notes = []

    def vmax_at(t):          # явный признак от сборщика, не T0==0: у вызовов без T0 он тоже 0
        return HOOK_FIRST_VIEW_SEC if (first_of_film and t < HOOK_ZONE_SEC) else MAX_VIEW_SEC

    def split(t0, t1, win):
        out, cur = [], win
        if not views:
            return out
        while t1 - t0 > vmax_at(t0):
            vmax = vmax_at(t0)
            lo = max(t0 + min(MIN_VIEW_SEC, vmax - 0.3), labels_done)
            hi = min(t0 + vmax - VIEW_EDGE_MARGIN_SEC, t1 - MIN_VIEW_SEC)
            if lo > t1 - MIN_VIEW_SEC:
                break
            # склейка — на паузе речи (самый длинный промежуток между словами в допустимом окне): длина плана
            # идёт от фразы, а не ровными кусками (брендбук: «не ровно 2,0 с — ровный ритм усыпляет»).
            # Пауза ищется около «ровной» точки: кусков столько, сколько нужно, а не больше — лишний кусок
            # повторял бы план, когда новых у картинки уже нет (живой прогон 05.10)
            k = int(np.ceil((t1 - t0)/vmax))
            ideal = t0 + (t1 - t0)/k
            s_ = speech.phrase_cut(max(lo, ideal - 0.8), min(hi, ideal + 0.8)) if hi >= lo else None
            if s_ is None:
                # без таймингов речи куски вышли бы ровными до миллисекунды (эп.01 08.10: семь планов 3.0–3.6 с
                # подряд): детерминированный сдвиг от ровной точки; ключ — от самого кадра (длина, точка, кусок),
                # не от T0, иначе правка ранней фразы перерендеривала бы все клипы превью после неё
                jit = 0.0 if speech.starts else (_stable01(round(D, 2), round(t0, 2), k) - 0.5)*JITTER_SEC
                s_ = speech.snap(min(max(ideal + jit, lo), hi) if hi >= lo else max(ideal, lo), lo, t1 - MIN_VIEW_SEC)
            if s_ is None:
                # окно целиком занято письмом (мысль, акцент): склейка — сразу после того, как дописано и
                # постояло, если до конца кадра ещё есть план (живой эп.01: кадр с мыслью шёл 6,8 с одним планом)
                for _, b_ in sorted(writing.busy_win):
                    if t0 < b_ <= t1 - MIN_VIEW_SEC and writing.free(b_ + 0.05, b_ + 0.15):
                        s_ = speech.snap(b_ + 0.1, b_ + 0.05, t1 - MIN_VIEW_SEC)
                        if s_ is not None:
                            break
            if s_ is None:
                break
            nxt = vc.next(cur, writing.held_boxes(s_))
            if nxt is None:
                # ни один план не держит дописанную надпись целиком: сначала — склейка ДО начала письма
                # (мысль пишется уже в новом плане; аудит 08.10: иначе план тянулся 7 с), иначе — после удержания
                w0 = min((a for a in writing.hold_starts() if t0 + MIN_VIEW_SEC <= a - WRITE_LEAD_SEC and a < s_), default=None)
                pre = None
                if w0 is not None:
                    edge = min(hi, w0 - WRITE_LEAD_SEC)
                    pre = speech.phrase_cut(lo, edge) or speech.snap(edge, lo, w0 - WRITE_LEAD_SEC)
                if pre is not None:
                    s_, nxt = pre, vc.next(cur, writing.held_boxes(pre))
                else:
                    he = writing.hold_end(s_)
                    s2 = speech.snap(he, he, t1 - MIN_VIEW_SEC) if he <= t1 - MIN_VIEW_SEC + 1e-6 else None
                    nxt = vc.next(cur, writing.held_boxes(s2)) if s2 is not None else None
                    s_ = s2 if s2 is not None else s_
                if nxt is None:
                    notes.append(f"склейка {s_:.2f} с не делается: надпись должна достоять на экране")
                    break
            at = writing.accent_time
            if nxt != wide and at is not None and at - MAX_VIEW_SEC - CUT_SNAP_SEC <= s_ <= at + 1.5:
                break           # акцент встаёт на общем плане: на среднем ему нет места (живой прогон)
            cur = nxt
            out.append((s_, "cut", cur))
            t0 = s_
        return out

    bounds = [(0.0, "start", wide)] + sorted(cuts) + [(D, "end", None)]
    extra = []
    for (a, ka, wa), (b, _, _) in zip(bounds, bounds[1:]):
        if ka in ("start", "cut"):
            extra += split(a, b, wa)
    return sorted(bounds[:-1] + extra) + [(D, "end", None)], notes


def build_segments(bounds, wide, zoom_in, SW, SH):
    """Границы -> сегменты камеры (drift / punch+hold); склейка-«скачок» отменяется. (сегменты, заметки)."""
    segs, cur, notes = [], wide, []
    for (a, ka, wa), (b, _, _) in zip(bounds, bounds[1:]):
        if b - a <= 1e-6:
            continue
        if ka == "punch":
            segs.append(dict(t0=a, t1=a + camera.PUNCH_SEC, kind="punch", win=cur, win_to=wa))
            segs.append(dict(t0=a + camera.PUNCH_SEC, t1=b, kind="hold", win=wa))
            cur = wa
            continue
        if ka == "cut" and camera.is_jump(cur, wa, SW, SH):
            notes.append(f"склейка {a:.2f} с отменена: скачок")
            if segs:
                segs[-1]["t1"] = b
            continue
        cur = wa
        segs.append(dict(t0=a, t1=b, kind="drift", win=wa, zoom_in=zoom_in))
    return segs, notes


def orient_drift(segs, writing):
    """Направление дрейфа — одно на всю картинку (§1.3: смена знака внутри картинки читается как
    дёрганье, приёмка это меряет); картинка с рукописной мыслью идёт наездом — к мысли, а не от неё
    (сравнение с прежним роликом 08.10: лицо на «только открыть» шло отъездом); иначе — чередование кадров."""
    if writing.key_time is None:
        return
    for sg in segs:
        if sg["kind"] == "drift":
            sg["zoom_in"] = True


def add_pushes(segs, wide, label_times, parts, busy):
    """Схема собирается по голосу: на каждой названной части камера чуть наклоняется к ней."""
    SH, SW = busy.shape
    wcx, wcy = camera.center(wide)
    for sg in segs:
        if sg["kind"] != "drift" or sg["win"] != wide:
            continue
        pushes = []
        for i, lt in enumerate(label_times):
            box = parts[i] if parts and i < len(parts) else None
            if not box or not (sg["t0"] <= lt < sg["t1"] - PUSH_SEC):
                continue
            z = 1 + min(PUSH_MAX, PUSH_STEP*(len(pushes) + 1))
            bcx, bcy = camera.center(box)
            win = camera.window(wcx + (bcx - wcx)*PUSH_PULL, wcy + (bcy - wcy)*PUSH_PULL, (wide[2] - wide[0])/z, SW, SH)
            if camera.edge_cross(busy, win) > camera.PUNCH_MAX_CROSS:
                continue                         # наклон резал бы рисунок — эта часть без него
            pushes.append(dict(t=lt, win=win))
        if pushes:
            sg["pushes"] = pushes


def add_lean(segs, wide, focus, busy):
    """Длинный план без смены и без сборки — медленный наезд к главному на всю длину (до 8%, без разреза);
    возврат к общему плану после среднего — тоже в движении, а не тот же стоп-кадр (критик: «jump-back»)."""
    if focus is None:
        return
    SH, SW = busy.shape
    sep = camera.others(busy, focus)
    fcx, fcy = camera.center(focus)
    for k, sg in enumerate(segs):
        back = k > 0 and sg["win"] == wide and segs[k - 1]["win"] != wide
        if sg["kind"] != "drift" or sg.get("pushes") or not (sg["t1"] - sg["t0"] > MAX_VIEW_SEC + CUT_SNAP_SEC or back):
            continue
        w0 = sg["win"]
        wcx, wcy = camera.center(w0)
        lean = camera.window(wcx + (fcx - wcx)*0.4, wcy + (fcy - wcy)*0.4, (w0[2] - w0[0])/(1 + PUSH_MAX), SW, SH)
        if camera.edge_cross(sep, lean, level=0.15) <= camera.SEPARATE_MAX_CROSS:
            sg["lean"] = lean


def plan(D, busy, words, labels=(), objects=(), key=None, key_dur=0.0, last_punch=-1e9, T0=0.0,
         zoom_in=True, parts=None, accent=None, accent_dur=0.0, first_of_film=False, key_box=None, accent_box=None):
    """Сегменты камеры и события кадра.

    D — длительность кадра; busy — карта занятости холста; words — слова речи
    кадра со временем от его начала; labels — [{"text", ...}]; objects —
    [{"name", "box", "word", "role"}] в координатах холста; key — текст главной
    мысли, key_dur — сколько он пишется; last_punch — глобальное время прошлого
    наезда; T0 — глобальное время начала кадра; zoom_in — направление дрейфа
    картинки (чередование кадров сборщиком; с мыслью — всегда наезд); parts — рамка
    (в координатах холста) части схемы, которую называет каждая подпись: на её слове
    камера чуть наклоняется к ней; first_of_film — первый кадр ролика: его первый план
    не дольше HOOK_FIRST_VIEW_SEC; key_box/accent_box — рамка (в координатах холста)
    уже написанной мысли/акцента: пока дописанное стоит (KEY_HOLD_SEC), склейка идёт
    только в план, где надпись целиком, иначе ждёт конца удержания
    (эп.01 08.10: «только открыть» дописана и через 0.7 с срезана крупным планом до «толь»).

    Возвращает dict: segments [{t0, t1, kind: drift|punch|hold, win, win_to, zoom_in, pushes?, lean?}],
    label_times [t], key_time и accent_time (или None), holds [{kind, t0, t1, box}] — окна, пока
    дописанная надпись стоит на экране, punch_at (глобальное время или None), punch_name, notes,
    views {wide, medium, details, wide_alt} — планы, из которых собран монтаж."""
    SH, SW = busy.shape
    wide = wide_view(busy, bool(key or accent))
    wz = camera.zoom_of(wide, SW, SH)
    view_ok = make_view_ok(busy, objects, wide, [tuple(b) for b in (key_box, accent_box) if b])

    label_times, notes = schedule_labels(labels, words, D)
    labels_done = (max(label_times) + 1.0) if label_times else 0.0
    writing, n_ = schedule_writing(D, words, key, key_dur, key_box, accent, accent_dur, accent_box); notes += n_
    punch, n_ = choose_punch(objects, words, busy, D, T0, last_punch, label_times, writing); notes += n_
    speech = Speech(words, D, writing)

    subj = subject_box(busy, objects)
    medium = medium_view(busy, objects, subj, wide, wz, view_ok)
    details, n_ = detail_views(busy, objects, wide, wz, medium, view_ok); notes += n_
    views = [v for v in [medium] + details if v is not None]
    wide_alt = alt_wide(wide, subj, views, view_ok, SW, SH)

    cuts = []                                     # опорные склейки: наезд и возврат
    if punch:
        t, pw, _ = punch
        cuts.append((t, "punch", pw))
        ret = t + camera.PUNCH_SEC + PUNCH_HOLD_SEC
        if D - ret >= MIN_VIEW_SEC:
            r = speech.snap(ret + 0.2, ret, D - MIN_VIEW_SEC)
            if r is not None:
                cuts.append((r, "cut", wide))
    bounds, n_ = cut_timeline(D, cuts, wide, views, ViewCycle(wide, views, wide_alt, SW, SH), writing, speech,
                              labels_done, first_of_film); notes += n_
    segs, n_ = build_segments(bounds, wide, zoom_in, SW, SH); notes += n_
    orient_drift(segs, writing)
    add_pushes(segs, wide, label_times, parts, busy)
    add_lean(segs, wide, subj, busy)
    for sg in segs:
        if sg["t1"] - sg["t0"] > MAX_VIEW_SEC + CUT_SNAP_SEC and sg["kind"] == "drift" and not sg.get("pushes") \
                and not sg.get("lean"):
            notes.append(f"план {sg['t0']:.1f}-{sg['t1']:.1f} с без смены: другого плана без разреза рисунка нет")
    return dict(segments=segs, label_times=label_times, key_time=writing.key_time, accent_time=writing.accent_time,
                punch_at=(T0 + punch[0]) if punch else None, punch_name=punch[2] if punch else None,
                notes=notes, views=dict(wide=wide, medium=medium, details=details, wide_alt=wide_alt),
                holds=writing.record())


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
        z = 1 + 0.5*camera.DRIFT*camera.through(u, ramp=0.25)   # после наезда — из покоя разогнаться и идти до склейки
        return camera.window(cx, cy, (s["win"][2] - s["win"][0])/z, SW, SH)
    if s.get("lean"):
        return camera.lerp(s["win"], s["lean"], camera.through(u))   # план начинается и кончается склейкой — без стопов
    if s.get("pushes"):
        cur = s["win"]
        for ps in s["pushes"]:
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
