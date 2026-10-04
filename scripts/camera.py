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


def drift(win, u, zoom_in, SW, SH):
    """Окно плана в момент u∈[0,1]: наезд или отъезд на DRIFT вокруг центра плана."""
    z = 1 + DRIFT*(ease_io(u) if zoom_in else 1 - ease_io(u))
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


def frame_for(busy, target, z_range, margin=0.12, bottom_w=2.0, spread=0.2, edge_k=6.0, max_cross=None, grid=13,
              cross_map=None):
    """Окно, в котором цель целиком с полями, а края рамки идут по пустому.
    z — крупность относительно всего холста. max_cross — рамки, режущие рисунок
    сильнее (edge_cross по cross_map, по умолчанию — по всей карте), не
    рассматриваются. None — не нашлось ни одного."""
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
                # у отдельных предметов ловим и бледные края (листы календаря, ореол рисунка)
                if max_cross is not None and edge_cross(cm, win, level=0.3 if cross_map is None else 0.15) > max_cross:
                    continue
                off = np.hypot((x0 + x1)/2 - tcx, (y0 + y1)/2 - tcy)/w
                cost = _edge_busy(busy, win, bottom_w)*edge_k + off*0.8 - 0.05*z
                if best is None or cost < best[0]:
                    best = (cost, win)
    return best[1] if best else None


def punch_window(busy, obj):
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
    shift = np.hypot((a[0] + a[2] - b[0] - b[2])/2, (a[1] + a[3] - b[1] - b[3])/2)/max(a[2] - a[0], b[2] - b[0])
    return ratio < CUT_MIN_RATIO and shift < CUT_MIN_SHIFT
