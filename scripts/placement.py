#!/usr/bin/env python3
"""Где поставить надпись и как провести стрелку — по самой картинке, без зашитых координат.

1. Карта занятости: тёмные линии, цветные заливки, края — всё, что нарисовано.
2. Кандидаты — окна размера надписи (с полями) по сетке; окно годно, если почти пустое,
   внутри безопасной зоны кадра и не над самим предметом.
3. Оценка: «рядом, но не вплотную» к предмету, выше/сбоку лучше, чем снизу, путь стрелки
   по пустому месту, ближе к линиям третей.
4. Стрелка: от края надписи, обращённого к предмету, с зазором; кончик не касается
   предмета; изгиб — в более пустую сторону.
Порог «пусто» берётся по самому кадру (40-й перцентиль занятости + 0.04): на
увеличенном кадре шум бумаги выше фиксированного порога, и место «не находилось»
(демо 04.10 — надпись легла на календарь). Места нет — None, а не надпись
поверх рисунка: вызывающий уменьшает надпись (choose_fit) или не пишет её."""
import numpy as np
from scipy import ndimage

SAFE = (0.05, 0.06, 0.95, 0.86)
ARROW_MAX_BUSY = 0.12      # поля кадра: снизу больше — там полоса плеера YouTube

def busy_map(rgb, margin=26):
    f = rgb.astype(np.float32)
    lum = f.mean(2); sat = f.max(2) - f.min(2)
    paper = np.median(lum[lum >= np.percentile(lum, 60)])     # >=: на ровной бумаге > даёт пустой срез и NaN
    ink = np.clip((paper - lum)/60, 0, 1) + np.clip((sat - 18)/50, 0, 1)
    edge = np.hypot(ndimage.sobel(lum, 0), ndimage.sobel(lum, 1))/400
    b = np.clip(ink + edge, 0, 1)
    b = ndimage.gaussian_filter(b, 2)
    floor = np.percentile(b, 25)                               # шум бумаги и сжатия — это не рисунок
    b = np.clip((b - floor - 0.04)/(1 - floor), 0, 1)
    b = ndimage.grey_dilation(b, size=(margin, margin))       # поле вокруг всего нарисованного
    return ndimage.gaussian_filter(b, 6)

def _integral(a):
    return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _boxsum(integ, x0, y0, x1, y1):
    return integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]

def _line_busy(b, p0, p1, n=40):
    t = np.linspace(0.15, 0.85, n)
    xs = np.clip((p0[0] + (p1[0]-p0[0])*t).astype(int), 0, b.shape[1]-1)
    ys = np.clip((p0[1] + (p1[1]-p0[1])*t).astype(int), 0, b.shape[0]-1)
    return float(b[ys, xs].mean())

def choose(rgb, obj, text_wh, step=12, b=None):
    """obj — (x0, y0, x1, y1) предмета в координатах кадра; text_wh — размер надписи.
    Возвращает центр надписи и точки стрелки (или None, если места нет)."""
    H, W = rgb.shape[:2]
    if b is None:
        b = busy_map(rgb)
    integ = _integral(b)
    thr = max(0.06, float(np.percentile(b, 40)) + 0.04)      # порог «пусто» — по самому кадру, а не числом
    tw, th = text_wh; pw, ph = int(tw + 0.5*th), int(th*1.6)          # окно с полями
    free = obj is None                    # надпись без предмета: ни «рядом», ни стрелки
    if free:
        obj = (-100, -100, -99, -99)
    ox0, oy0, ox1, oy1 = obj
    od = max(ox1-ox0, oy1-oy0)
    best = None
    for cy in range(int(SAFE[1]*H + ph/2), int(SAFE[3]*H - ph/2), step):
        for cx in range(int(SAFE[0]*W + pw/2), int(SAFE[2]*W - pw/2), step):
            x0, y0, x1, y1 = int(cx-pw/2), int(cy-ph/2), int(cx+pw/2), int(cy+ph/2)
            fill = _boxsum(integ, x0, y0, x1, y1)/(pw*ph)
            if fill > thr: continue                                     # не на рисунке
            if x1 > ox0 - 10 and x0 < ox1 + 10 and y1 > oy0 - 10 and y0 < oy1 + 10: continue
            # расстояние от окна до предмета (по краям)
            dx = max(ox0 - x1, x0 - ox1, 0); dy = max(oy0 - y1, y0 - oy1, 0)
            gap = np.hypot(dx, dy)
            ideal = np.clip(0.9*od, 0.06*W, 0.16*W)
            s_gap = -abs(gap - ideal)/ideal                                # рядом, но не вплотную
            below = 1.0 if y0 > oy1 else 0.0                               # под предметом хуже
            s_pos = -0.6*below + 0.25*(1 - cy/H)                           # чуть выше — лучше
            third = min(abs(cy - H/3), abs(cy - 2*H/3))/H + min(abs(cx - W/3), abs(cx - 2*W/3))/W
            s_third = -0.8*third
            s_fill = -6*fill
            # путь стрелки: от ближнего края окна к предмету — должен идти по пустому
            tb = (cx - tw/2, cy - th/2, cx + tw/2, cy + th/2)
            s_path = 0.0
            if not free:
                p0, p1, _ = arrow_points(tb, obj, th, None)
                s_path = -9*_line_busy(b, p0, p1)              # стрелка не должна резать рисунок
            # не «висеть» над чужим рисунком: что прямо под надписью
            by0, by1 = min(H-1, y1), min(H-1, y1 + int(0.14*H))
            under = _boxsum(integ, x0, by0, x1, by1)/max(1, (x1-x0)*(by1-by0))
            s_under = -2.5*under
            if free:
                s_gap = s_path = 0.0
            s = s_gap + s_pos + s_third + s_fill + s_path + s_under
            if best is None or s > best[0]: best = (s, cx, cy, (x0, y0, x1, y1))
    if best is None:
        return None
    _, cx, cy, win = best
    tb = (cx - tw/2, cy - th/2, cx + tw/2, cy + th/2)
    arr = None if free else arrow_points(tb, obj, th, b)
    # стрелка поперёк рисунка хуже, чем её отсутствие: тогда только надпись рядом
    if arr and _line_busy(b, arr[0], arr[1]) > ARROW_MAX_BUSY:
        arr = None
    return dict(center=(cx, cy), arrow=arr)

def arrow_points(text_box, obj, th, b):
    tx0, ty0, tx1, ty1 = text_box; ox0, oy0, ox1, oy1 = obj
    tc = np.array([(tx0+tx1)/2, (ty0+ty1)/2]); oc = np.array([(ox0+ox1)/2, (oy0+oy1)/2])
    d = oc - tc; d /= np.linalg.norm(d) + 1e-9
    # старт: точка на краю надписи в сторону предмета + зазор 0.45 высоты строки
    def exit_point(box, c, dirv, pad):
        x0, y0, x1, y1 = box; ts = []
        for v, lo, hi, cc in ((dirv[0], x0, x1, c[0]), (dirv[1], y0, y1, c[1])):
            if abs(v) > 1e-6: ts.append(((hi if v > 0 else lo) - cc)/v)
        t = min(t for t in ts if t > 0)
        return c + dirv*(t + pad)
    p0 = exit_point(text_box, tc, d, 0.45*th)
    od = min(ox1-ox0, oy1-oy0)
    p1 = exit_point(obj, oc, -d, max(18, 0.10*od))          # кончик не касается предмета
    if b is None: return (tuple(p0), tuple(p1), 0.0)
    # изгиб в более пустую сторону
    n = np.array([-d[1], d[0]]); L = np.linalg.norm(p1 - p0)
    mid = (p0 + p1)/2
    sides = []
    for sgn in (1, -1):
        c = mid + n*sgn*0.18*L
        sides.append(_line_busy(b, p0, c) + _line_busy(b, c, p1))
    bend = 0.18 if sides[0] <= sides[1] else -0.18
    return (tuple(p0), tuple(p1), bend)


def choose_fit(rgb, obj, make_wh, sizes):
    """Ищет место, уменьшая надпись, если не влезает. make_wh(size)->(w,h). Возвращает (size, result)."""
    b = busy_map(rgb)
    for s in sizes:
        r = choose(rgb, obj, make_wh(s), b=b)
        if r: return s, r
    return None, None
