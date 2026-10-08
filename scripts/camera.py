#!/usr/bin/env python3
"""Виртуальная камера по одному рисунку: окна 16:9, их подбор и движение.

Решения владельца и критиков (04.10):
  * внутри плана движение не больше ~4% (DRIFT) — медленный большой наезд
    читался как «слайд-шоу»; смена плана — склейкой, а не долгим зумом;
  * быстрый наезд на предмет, о котором говорят (PUNCH_SEC=0.33 с, торможение
    в конце): предмет в центре и почти на весь кадр, соседние предметы либо
    целиком в кадре, либо целиком за ним — край кадра не режет рисунок;
    не больше PUNCH_MAX_ZOOM; возврат — склейкой, не обратным зумом;
  * план выбирается по карте занятости (placement.busy_map): по краям рамки
    должно быть пусто (frame_for), снизу — строже (там полоса плеера)."""
import math

import numpy as np

ASPECT = 16/9
DRIFT = 0.04             # наезд/отъезд внутри плана, доля
PUNCH_SEC = 0.33
PUNCH_MAX_ZOOM = 2.5
PUNCH_FILL = 0.72        # предмет занимает такую долю кадра по своей «тесной» стороне
SEPARATE_MIN_AREA = 0.015   # отдельный предмет — от 1.5% холста (календарь ~5%; огонёк, искры, следы меньше)
SEPARATE_MAX_CROSS = 0.0    # отдельный предмет край кадра не задевает совсем (живой ролик: уголок календаря у края)
PUNCH_MAX_CROSS = 0.01   # край кадра наезда почти нигде не идёт по рисунку (соседи целиком или вне кадра)
PUNCH_MIN_FILL = 0.45    # мельче даже на PUNCH_MAX_ZOOM — не «на весь экран»: наезда нет (нужен крупный план)
CUT_MIN_RATIO, CUT_MIN_SHIFT = 1.5, 0.4   # соседние планы одного рисунка: иначе «скачок»


def ease_io(u):
    u = np.clip(u, 0, 1)
    return u*u*u*(u*(6*u - 15) + 10)


def ease_out(u):
    u = np.clip(u, 0, 1)
    return 1 - (1 - u)**3


def window(cx, cy, w, SW, SH):
    """Окно 16:9 шириной w с центром (cx, cy), сдвинутое внутрь холста SWxSH."""
    w = min(w, SW, SH*ASPECT)
    h = w/ASPECT
    cx = float(np.clip(cx, w/2, SW - w/2))
    cy = float(np.clip(cy, h/2, SH - h/2))
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


def lerp(a, b, u):
    """Промежуточное окно: центр линейно, ширина — по логарифму (равномерный зум на глаз)."""
    ca = ((a[0] + a[2])/2, (a[1] + a[3])/2); cb = ((b[0] + b[2])/2, (b[1] + b[3])/2)
    wa, wb = a[2] - a[0], b[2] - b[0]
    w = float(np.exp(np.log(wa) + (np.log(wb) - np.log(wa))*u))
    k = (wa - w)/(wa - wb) if abs(wa - wb) > 1e-6 else u        # центр едет вместе с зумом
    cx = ca[0] + (cb[0] - ca[0])*k; cy = ca[1] + (cb[1] - ca[1])*k
    h = w/ASPECT
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


def through(u, ramp=0.0):
    """Ход «сквозь склейку»: постоянная скорость, без разгона в начале и торможения в конце.
    План начинается и кончается склейкой, и ease_io на его концах давал стоп перед каждой
    склейкой: замер на плане 3 с — первые и последние 0.5 с камера шла на 21-26% средней
    скорости, 17% кадров плана практически стояли (то самое «движение останавливается
    раньше времени», которое запрещено правилом Кен Бёрнса). ramp>0 — короткий разгон из
    покоя для хода, который продолжает остановившееся движение (после наезда)."""
    u = float(np.clip(u, 0, 1))
    if ramp <= 0:
        return u
    a = ramp
    s = (u*u/(2*a)) if u < a else (u - a/2)
    return s/(1 - a/2)


def drift(win, u, zoom_in, SW, SH):
    """Окно плана в момент u∈[0,1]: наезд или отъезд на DRIFT вокруг центра плана."""
    k = through(u)
    z = 1 + DRIFT*(k if zoom_in else 1 - k)
    cx, cy = (win[0] + win[2])/2, (win[1] + win[3])/2
    return window(cx, cy, (win[2] - win[0])/z, SW, SH)


def _edge_busy(busy, win, bottom_w):
    SH, SW = busy.shape
    x0, y0, x1, y1 = win
    xs = np.clip(np.linspace(x0, x1 - 1, 60).astype(int), 0, SW - 1)
    ys = np.clip(np.linspace(y0, y1 - 1, 40).astype(int), 0, SH - 1)
    return (busy[int(np.clip(y0, 0, SH - 1)), xs].mean() + bottom_w*busy[int(np.clip(y1 - 1, 0, SH - 1)), xs].mean()
            + busy[ys, int(np.clip(x0, 0, SW - 1))].mean() + busy[ys, int(np.clip(x1 - 1, 0, SW - 1))].mean())/(3 + bottom_w)


def edge_cross(busy, win, level=0.3):
    """Доля края рамки (по худшей из четырёх сторон), которая идёт по рисунку:
    0 — рамка ничего не режет. Среднее по краю тут не годится — одна
    перерезанная стрелка тонет в пустой бумаге остального края."""
    SH, SW = busy.shape
    x0, y0, x1, y1 = win
    xs = np.clip(np.linspace(x0, x1 - 1, 120).astype(int), 0, SW - 1)
    ys = np.clip(np.linspace(y0, y1 - 1, 70).astype(int), 0, SH - 1)
    sides = [busy[int(np.clip(y0, 0, SH - 1)), xs], busy[int(np.clip(y1 - 1, 0, SH - 1)), xs],
             busy[ys, int(np.clip(x0, 0, SW - 1))], busy[ys, int(np.clip(x1 - 1, 0, SW - 1))]]
    # край холста — не разрез: там рисунок кончается сам
    keep = [y0 > 1, y1 < SH - 1, x0 > 1, x1 < SW - 1]
    return max([float((s_ > level).mean()) for s_, k in zip(sides, keep) if k] or [0.0])


def others(busy, box, level=0.3):
    """Карта занятости без той группы рисунка, к которой принадлежит box (связные
    участки, задевающие рамку). Крупный план вправе обрезать край своей же сцены —
    так снимают всегда; нельзя резать ОТДЕЛЬНЫЙ предмет рядом (живой ролик 04.10:
    кот держит конверт, и правило «не резать ничего» отменяло любой наезд и любой
    средний план — ролик стоял одним планом)."""
    from scipy import ndimage
    lab, n = ndimage.label(busy > level)
    if not n:
        return busy
    SH, SW = busy.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    inside = np.bincount(lab[max(0, y0):min(SH, y1), max(0, x0):min(SW, x1)].ravel(), minlength=n + 1)
    total = np.bincount(lab.ravel(), minlength=n + 1)
    # своя — группа, которая в основном внутри рамки; рамка большой группы может краем задеть
    # соседний предмет (живой кадр: календарь над столом), и он от этого своим не становится
    # …и группа, которой в рамке больше всего (предмет в лапах кота — часть группы «кот»)
    ids = set((np.nonzero(inside[1:] > 0.5*total[1:])[0] + 1).tolist())
    if inside[1:].max() > 0:
        ids.add(int(np.argmax(inside[1:])) + 1)
    ids = np.array(sorted(ids), int)
    out = busy.copy()
    mine = np.isin(lab, ids)
    # мелкие отдельные кусочки (огонёк хвоста, искры, следы) — не «соседний предмет»: обрезать можно
    sl = ndimage.find_objects(lab)
    tiny = [k + 1 for k, q in enumerate(sl) if q is not None
            and (q[0].stop - q[0].start)*(q[1].stop - q[1].start) < SEPARATE_MIN_AREA*SH*SW]   # по габариту
    mine |= np.isin(lab, tiny)
    rest = ndimage.binary_dilation((lab > 0) & ~mine, iterations=12)        # чужие предметы с их краями
    own = ndimage.binary_dilation(mine, iterations=30) & ~rest              # своя группа с бледным ореолом
    out[own] = 0
    return out


def _lin_rows(start, stop, num):
    """np.linspace по строкам (start, stop — векторы), бит-в-бит как numpy: arange*step + start, последний = stop."""
    step = (stop - start)/(num - 1)
    y = np.arange(num, dtype=np.float64)[None, :]*step[:, None] + start[:, None]
    y[:, -1] = stop
    return y


def _edge_cross_rows(cm, x0, y0, x1, y1, level):
    """edge_cross() сразу для многих окон (строки x0..y1)."""
    SH, SW = cm.shape
    xs = np.clip(_lin_rows(x0, x1 - 1, 120).astype(int), 0, SW - 1)
    ys = np.clip(_lin_rows(y0, y1 - 1, 70).astype(int), 0, SH - 1)
    ry0 = np.clip(y0, 0, SH - 1).astype(int)[:, None]; ry1 = np.clip(y1 - 1, 0, SH - 1).astype(int)[:, None]
    rx0 = np.clip(x0, 0, SW - 1).astype(int)[:, None]; rx1 = np.clip(x1 - 1, 0, SW - 1).astype(int)[:, None]
    sides = np.stack([(cm[ry0, xs] > level).mean(1), (cm[ry1, xs] > level).mean(1),
                      (cm[ys, rx0] > level).mean(1), (cm[ys, rx1] > level).mean(1)], 1)
    keep = np.stack([y0 > 1, y1 < SH - 1, x0 > 1, x1 < SW - 1], 1)      # край холста — не разрез
    out = np.where(keep, sides, -np.inf).max(1)
    out[~keep.any(1)] = 0.0
    return out


def _edge_busy_rows(busy, x0, y0, x1, y1, bottom_w):
    """_edge_busy() сразу для многих окон."""
    SH, SW = busy.shape
    xs = np.clip(_lin_rows(x0, x1 - 1, 60).astype(int), 0, SW - 1)
    ys = np.clip(_lin_rows(y0, y1 - 1, 40).astype(int), 0, SH - 1)
    ry0 = np.clip(y0, 0, SH - 1).astype(int)[:, None]; ry1 = np.clip(y1 - 1, 0, SH - 1).astype(int)[:, None]
    rx0 = np.clip(x0, 0, SW - 1).astype(int)[:, None]; rx1 = np.clip(x1 - 1, 0, SW - 1).astype(int)[:, None]
    return (busy[ry0, xs].mean(1) + bottom_w*busy[ry1, xs].mean(1) + busy[ys, rx0].mean(1)
            + busy[ys, rx1].mean(1))/(3 + bottom_w)


def frame_for(busy, target, z_range, margin=0.12, bottom_w=2.0, spread=0.2, edge_k=6.0, max_cross=None, grid=13,
              cross_map=None, away=None, accept=None):
    """Окно, в котором цель целиком с полями, а края рамки идут по пустому.
    z — крупность относительно всего холста. max_cross — рамки, режущие рисунок
    сильнее (edge_cross по cross_map, по умолчанию — по всей карте), не
    рассматриваются; away=(cx, _, доля) — центр нового плана не ближе доли ширины к cx
    (склейка в ту же точку крупнее читается как «цифровой зум»); accept(win) — внешняя
    проверка, зовётся по возрастанию стоимости до первого принятого. None — не нашлось.

    Перебор (11 крупностей × grid² сдвигов) считается векторно на каждую крупность:
    тот же порядок кандидатов и тот же выбор, что у построчного перебора (_frame_for_scan,
    хранится ради теста эквивалентности), в 16 раз быстрее (ревью 08.10: frame_for —
    85% времени shots.plan, стоимость считалась для каждого кандидата по одному)."""
    cm = busy if cross_map is None else cross_map
    level = 0.3 if cross_map is None else 0.15
    SH, SW = busy.shape
    base = min(SW, SH*ASPECT)
    tx0, ty0, tx1, ty1 = target
    tcx, tcy = (tx0 + tx1)/2, (ty0 + ty1)/2
    offs = np.linspace(-spread, spread, grid)
    ox, oy = [a.ravel() for a in np.meshgrid(offs, offs, indexing="ij")]   # ox — внешний цикл, oy — внутренний
    rows, order0 = [], 0                                                  # (стоимость, порядок, окно)
    for z in np.linspace(z_range[0], z_range[1], 11):
        w = base/z; h = w/ASPECT
        if (tx1 - tx0) > w*(1 - 2*margin) or (ty1 - ty0) > h*(1 - 2*margin):
            continue
        w_ = min(w, SW, SH*ASPECT); h_ = w_/ASPECT                         # window(): клэмп ширины и центра
        cx = np.clip(tcx + ox*w, w_/2, SW - w_/2); cy = np.clip(tcy + oy*h, h_/2, SH - h_/2)
        x0, y0, x1, y1 = cx - w_/2, cy - h_/2, cx + w_/2, cy + h_/2
        ok = ~((tx0 < x0 + margin*w) | (tx1 > x1 - margin*w) | (ty0 < y0 + margin*h) | (ty1 > y1 - margin*h))
        if away is not None:
            ok &= ~(np.abs((x0 + x1)/2 - away[0]) < away[2])
        idx = np.nonzero(ok)[0]
        if len(idx) and max_cross is not None:
            idx = idx[~(_edge_cross_rows(cm, x0[idx], y0[idx], x1[idx], y1[idx], level) > max_cross)]
        if len(idx):
            eb = _edge_busy_rows(busy, x0[idx], y0[idx], x1[idx], y1[idx], bottom_w)
            off = np.hypot((x0[idx] + x1[idx])/2 - tcx, (y0[idx] + y1[idx])/2 - tcy)/w
            cost = eb*edge_k + off*0.8 - 0.05*z
            rows += [(float(c), order0 + int(i), (float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i])))
                     for c, i in zip(cost, idx)]
        order0 += len(ox)
    rows.sort(key=lambda r: (r[0], r[1]))
    for _, _, win in rows:
        if accept is None or accept(win):
            return win
    return None


def _frame_for_scan(busy, target, z_range, margin=0.12, bottom_w=2.0, spread=0.2, edge_k=6.0, max_cross=None,
                    grid=13, cross_map=None, away=None, accept=None):
    """Построчный перебор — эталон для теста эквивалентности frame_for(); в проде не вызывается."""
    cm = busy if cross_map is None else cross_map
    SH, SW = busy.shape
    base = min(SW, SH*ASPECT)
    tx0, ty0, tx1, ty1 = target
    tcx, tcy = (tx0 + tx1)/2, (ty0 + ty1)/2
    best = None
    for z in np.linspace(z_range[0], z_range[1], 11):
        w = base/z; h = w/ASPECT
        if (tx1 - tx0) > w*(1 - 2*margin) or (ty1 - ty0) > h*(1 - 2*margin):
            continue
        for ox in np.linspace(-spread, spread, grid):
            for oy in np.linspace(-spread, spread, grid):
                win = window(tcx + ox*w, tcy + oy*h, w, SW, SH)
                x0, y0, x1, y1 = win
                if tx0 < x0 + margin*w or tx1 > x1 - margin*w or ty0 < y0 + margin*h or ty1 > y1 - margin*h:
                    continue
                if max_cross is not None and edge_cross(cm, win, level=0.3 if cross_map is None else 0.15) > max_cross:
                    continue
                if away is not None and abs((x0 + x1)/2 - away[0]) < away[2]:
                    continue
                if accept is not None and not accept(win):
                    continue
                off = np.hypot((x0 + x1)/2 - tcx, (y0 + y1)/2 - tcy)/w
                cost = _edge_busy(busy, win, bottom_w)*edge_k + off*0.8 - 0.05*z
                if best is None or cost < best[0]:
                    best = (cost, win)
    return best[1] if best else None


def punch_window(busy, obj, keep=()):
    """Окно быстрого наезда: предмет ~PUNCH_FILL кадра, не крупнее PUNCH_MAX_ZOOM,
    края по пустому. None — предмет слишком велик (наезд меньше CUT_MIN_RATIO —
    не наезд, и возврат склейкой был бы «скачком»)
    или слишком мал (даже на PUNCH_MAX_ZOOM меньше PUNCH_MIN_FILL кадра)."""
    SH, SW = busy.shape
    base = min(SW, SH*ASPECT)
    ow, oh = obj[2] - obj[0], obj[3] - obj[1]
    z_fit = min(base*PUNCH_FILL/max(ow, 1), base/ASPECT*PUNCH_FILL/max(oh, 1))
    z_hi = min(PUNCH_MAX_ZOOM, z_fit)
    if z_hi < CUT_MIN_RATIO:
        punch_window.why = "предмет и так крупный"
        return None
    if z_hi < z_fit*PUNCH_MIN_FILL/PUNCH_FILL:
        punch_window.why = f"предмет мелкий: даже на {PUNCH_MAX_ZOOM}x меньше {PUNCH_MIN_FILL:.0%} кадра"
        return None
    m = (1 - PUNCH_FILL)/2*0.6
    sep = others(busy, obj)
    # герой рядом (keep — его рамка): целиком в кадре или вне его, даже мелкими частями —
    # усы и кончик хвоста у края кадра читались как случайные чёрточки (живой прогон 05.10)
    for kb in keep:
        if not (obj[0] >= kb[0] - 1 and obj[1] >= kb[1] - 1 and obj[2] <= kb[2] + 1 and obj[3] <= kb[3] + 1):
            ys_k, xs_k = slice(max(0, int(kb[1])), int(kb[3])), slice(max(0, int(kb[0])), int(kb[2]))
            sep = sep.copy() if sep is busy else sep
            sep[ys_k, xs_k] = np.maximum(sep[ys_k, xs_k], busy[ys_k, xs_k])
    # от самого крупного к более общему: первый, у которого края не режут рисунок
    for z in np.linspace(z_hi, max(CUT_MIN_RATIO, z_fit*PUNCH_MIN_FILL/PUNCH_FILL), 8):
        w = frame_for(busy, obj, (z, z), margin=m, spread=0.15, edge_k=10.0, max_cross=SEPARATE_MAX_CROSS,
                      cross_map=sep)
        if w is not None:
            punch_window.why = None
            return w
    punch_window.why = "любой кадр наезда режет соседний отдельный предмет"
    return None


def zoom_of(win, SW, SH):
    return min(SW, SH*ASPECT)/(win[2] - win[0])


def is_jump(a, b, SW, SH):
    """Склейка между планами одного рисунка — «скачок», если крупность отличается
    меньше чем в CUT_MIN_RATIO и центр сдвинут меньше CUT_MIN_SHIFT ширины."""
    za, zb = zoom_of(a, SW, SH), zoom_of(b, SW, SH)
    ratio = max(za, zb)/min(za, zb)
    shift = math.hypot((a[0] + a[2] - b[0] - b[2])/2, (a[1] + a[3] - b[1] - b[3])/2)/max(a[2] - a[0], b[2] - b[0])
    return ratio < CUT_MIN_RATIO and shift < CUT_MIN_SHIFT
