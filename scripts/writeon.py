#!/usr/bin/env python3
"""Главная мысль «пишется карандашом»: буква открывается вдоль своей средней линии
в порядке письма, со звуком (звук — pencil_sound.py).

Метод аниматоров (write-on по средней линии): форма буквы — точно шрифт (Caveat с
живыми вариантами букв через raqm/calt), внутри формы идёт «кисть» по скелету буквы.
Решения, каждое — из замера или из критики демо (04.10):
  * постоянная скорость пера в px/с, а не подгонка под время слова: подгонка давала
    штрихи по 3 кадра и пустые кадры между ними — «вспышки», а не рука;
  * свежий отрезок кадра рисуется полупрозрачнее и уплотняется в следующем кадре —
    кончик пера движется плавно даже при ~50 px за кадр;
  * начало штриха сужено, без круглой «точки» на кончике; по окончании штриха
    форма закрывается целиком (без дыр на кончиках);
  * порядок: точки и кратки последними; у р к н ж ф — стойки первыми; у д я а — овал;
    у «ю» — стойка, перекладина, потом овал (как в прописи); «д» (в Caveat — один
    штрих «∂») начинается со стыка у овала, а не с верхнего завитка;
  * мелкие части (точки ё, !, ?, :) рисуются отдельным касанием;
  * длинный текст уменьшается до 0.7 кегля, дальше переносится на две строки;
  * графит: слои штриха накапливаются (перекрытия темнее), зерно, тон ниже чёрного.
Карандаш — только для главной мысли (одна на 10-30 с); подписи схем — обычным
текстом (labels.py). Решение владельца 04.10."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.morphology import skeletonize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(ROOT, "assets", "fonts", "Caveat-Variable.ttf")
WEIGHT = 480
SPEED_PER_SIZE = 7.0          # px/с на 1 px кегля: ~1.1 с на слово (решение владельца 04.10)
STEMS_FIRST = set("ркнжфРКНЖФ")
LOOP_FIRST = set("дяаДЯА")
BOWL_START = set("д")         # «∂»: овал против часовой от стыка, потом вверх к завитку
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _font(size, weight=WEIGHT):
    f = ImageFont.truetype(FONT, int(size))
    try:
        f.set_variation_by_axes([weight])
    except Exception:  # noqa: BLE001 — шрифт без осей: обычное начертание
        pass
    return f


def fit_lines(text, size, max_w, min_frac=0.7):
    s = size
    while s >= size*min_frac:
        if _font(s).getlength(text) <= max_w:
            return int(s), [text]
        s -= 4
    words = text.split()
    if len(words) < 2:
        s = size
        while _font(s).getlength(text) > max_w and s > 16:
            s -= 4
        return int(s), [text]
    f = _font(size)
    best = min((max(f.getlength(" ".join(words[:i])), f.getlength(" ".join(words[i:]))), i)
               for i in range(1, len(words)))
    lines = [" ".join(words[:best[1]]), " ".join(words[best[1]:])]
    s = size
    while max(_font(s).getlength(l) for l in lines) > max_w and s > 16:
        s -= 4
    return int(s), lines


def measure(text, size, max_w):
    """(ширина, высота) надписи после подгонки — для поиска места."""
    s, lines = fit_lines(text, size, max_w)
    f = _font(s); asc, desc = f.getmetrics()
    return max(f.getlength(l) for l in lines), (asc + desc)*1.05*len(lines)


def layout(text, size, W, H, cx, cy, seed=1, jitter=1.0, max_w=None):
    """Буквы надписи: мягкая маска каждой буквы (с живой рукой: поворот/сдвиг/масштаб
    на БУКВУ целиком — дужка «й» и половинки «ы» едут вместе) и её мелкие части."""
    if not text or not text.strip():
        return []
    size, lines = fit_lines(text, size, max_w or 0.86*W)
    f = _font(size); asc, desc = f.getmetrics(); lh = (asc + desc)*1.05
    rng = np.random.default_rng(seed)
    out, word0 = [], 0
    for li, line in enumerate(lines):
        ly = cy - lh*len(lines)/2 + lh*li + lh/2
        x0 = cx - f.getlength(line)/2; y0 = ly - (asc + desc)/2
        im = Image.new("L", (W, H), 0)
        ImageDraw.Draw(im).text((x0, y0), line, font=f, fill=255)
        soft = np.asarray(im, np.float32)/255
        lab, n = ndimage.label(soft > 0.25, structure=np.ones((3, 3)))
        if n == 0:
            continue
        objs = ndimage.find_objects(lab)
        areas = ndimage.sum(np.ones_like(soft), lab, range(1, n + 1))
        bounds = [x0 + f.getlength(line[:i]) for i in range(len(line) + 1)]
        letters = [[] for _ in line]
        for k, sl in enumerate(objs):
            a, b = sl[1].start, sl[1].stop
            ov = [max(0, min(b, bounds[i + 1]) - max(a, bounds[i])) if line[i].strip() else -1
                  for i in range(len(line))]
            letters[int(np.argmax(ov))].append(k + 1)
        med = np.median(areas)
        for i, ks in enumerate(letters):
            if not ks:
                continue
            big = [k for k in ks if areas[k - 1] >= 0.15*med]
            small = [k for k in ks if areas[k - 1] < 0.15*med]
            ys, xs = np.nonzero(np.isin(lab, ks))
            y0b, y1b, x0b, x1b = ys.min() - 4, ys.max() + 5, xs.min() - 4, xs.max() + 5
            ang = rng.uniform(-2.0, 2.0)*jitter; sc = 1 + rng.uniform(-0.02, 0.02)*jitter
            dx = rng.uniform(-0.01, 0.01)*size*jitter; dy = rng.uniform(-0.025, 0.025)*size*jitter

            def move(mm):
                p = Image.fromarray((mm[max(0, y0b):y1b, max(0, x0b):x1b]*255).astype(np.uint8))
                p = p.rotate(ang, resample=Image.BICUBIC, expand=True)
                p = p.resize((max(1, round(p.width*sc)), max(1, round(p.height*sc))), Image.BICUBIC)
                pa = np.asarray(p, np.float32)/255
                full = np.zeros((H, W), np.float32)
                oy = int(round((y0b + y1b)/2 + dy - pa.shape[0]/2)); ox = int(round((x0b + x1b)/2 + dx - pa.shape[1]/2))
                ys_, xs_ = slice(max(0, oy), min(H, oy + pa.shape[0])), slice(max(0, ox), min(W, ox + pa.shape[1]))
                full[ys_, xs_] = pa[ys_.start - oy:ys_.stop - oy, xs_.start - ox:xs_.stop - ox]
                return full
            out.append(dict(ch=line[i], size=size, word=word0 + len(line[:i].split()),
                            soft=move(np.isin(lab, big)*soft) if big else np.zeros((H, W), np.float32),
                            dots=[move((lab == k)*soft) for k in small]))
        word0 += len(line.split())
    return out


# ---------------- штрихи
def _raw_paths(binm):
    pts = set(zip(*np.nonzero(skeletonize(binm))))

    def nb(p):
        return [(p[0] + a, p[1] + b) for a, b in NB if (p[0] + a, p[1] + b) in pts]
    out = []
    while pts:
        ends = [p for p in pts if len(nb(p)) == 1]
        cur = min(ends or list(pts), key=lambda p: p[1] + 0.55*p[0])
        path = [cur]; pts.discard(cur)
        while True:
            n = nb(cur)
            if not n:
                break
            if len(path) > 3:
                d = np.subtract(cur, path[-4]); n.sort(key=lambda q: -np.dot(np.subtract(q, cur), d))
            cur = n[0]; path.append(cur); pts.discard(cur)
        out.append(np.array(path, float))
    return out


def _smooth(p, k=2):
    q = p.copy()
    for i in range(len(p)):
        q[i] = p[max(0, i - k):i + k + 1].mean(0)
    return q


def _length(p):
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


def _closed(p):
    return np.linalg.norm(p[0] - p[-1]) < 4 and len(p) > 20


def _orient(p):
    """Стойки сверху вниз, петли против часовой с верхней правой точки, прочее слева направо."""
    if _closed(p):
        y, x = p[:, 0], p[:, 1]
        if 0.5*np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y) > 0:
            p = p[::-1]
        k = int(np.argmin(p[:, 0] - p[:, 1]*0.6))
        return np.r_[p[k:], p[:k]]
    if abs(p[-1, 0] - p[0, 0]) > 1.8*abs(p[-1, 1] - p[0, 1]):
        return p if p[0, 0] <= p[-1, 0] else p[::-1]
    return p if (p[0, 1] + 0.55*p[0, 0]) <= (p[-1, 1] + 0.55*p[-1, 0]) else p[::-1]


def _merge(paths):
    """Склеить продолжения: конец A у начала B и направление почти то же (рука не отрывается)."""
    changed = True
    while changed:
        changed = False
        for i in range(len(paths)):
            for j in range(len(paths)):
                A, B = paths[i], paths[j]
                if i != j and len(A) > 4 and len(B) > 4 and np.linalg.norm(A[-1] - B[0]) < 4:
                    da, db = A[-1] - A[-5], B[4] - B[0]
                    if np.dot(da, db)/(np.linalg.norm(da)*np.linalg.norm(db) + 1e-9) > 0.77:
                        paths[i] = np.r_[A, B[1:]]; del paths[j]; changed = True; break
            if changed:
                break
    return paths


def _extend(p, mask, rmax):
    """Дотянуть концы по касательной до края формы — кончики стоек открываются полностью."""
    H, W = mask.shape
    if len(p) < 6:
        return p

    def ext(end, prev):
        d = end - prev; d = d/(np.linalg.norm(d) + 1e-9); pts = []
        for s in range(1, int(rmax*1.6)):
            q = end + d*s; y, x = int(round(q[0])), int(round(q[1]))
            if not (0 <= y < H and 0 <= x < W) or mask[y, x] < 0.25:
                break
            pts.append(q)
        return pts
    head = ext(p[0], p[4])[::-1]; tail = ext(p[-1], p[-5])
    return np.r_[np.array(head).reshape(-1, 2), p, np.array(tail).reshape(-1, 2)]


def letter_strokes(L):
    soft = L["soft"]; binm = soft > 0.45; out = []
    if binm.any():
        dist = ndimage.distance_transform_edt(binm); rmed = float(np.median(dist[binm]))
        paths = [_smooth(p) for p in _raw_paths(binm) if len(p) >= 2]
        paths = [p for p in paths if _length(p) >= max(6.0, 0.6*rmed)]          # шпоры скелета — не штрихи
        paths = _merge([_orient(p) for p in paths])
        paths = [_extend(p, soft, dist.max() + 2) for p in paths]
        ch = L["ch"]

        def key(p):
            kind = "loop" if _closed(p) else ("stem" if abs(p[-1, 0] - p[0, 0]) > 1.8*abs(p[-1, 1] - p[0, 1]) else "other")
            pri = (0 if kind == "stem" else 1) if ch in STEMS_FIRST else ((0 if kind == "loop" else 1) if ch in LOOP_FIRST else 0)
            return (pri, round(p[0, 1]/max(8, 2*rmed)), p[0, 0])
        paths = sorted(paths, key=key)
        if ch in BOWL_START and paths:
            k = max(range(len(paths)), key=lambda i: _length(paths[i]))
            if paths[k][0, 0] < paths[k][-1, 0]:
                paths[k] = paths[k][::-1]
        for p in paths:
            r = dist[np.clip(p[:, 0].astype(int), 0, soft.shape[0] - 1), np.clip(p[:, 1].astype(int), 0, soft.shape[1] - 1)]
            out.append(dict(path=p, r=np.maximum(r, rmed*0.8), kind="stroke"))
    for d in L["dots"]:
        ys, xs = np.nonzero(d > 0.3)
        if len(ys) == 0:
            continue
        rr = float(ndimage.distance_transform_edt(d > 0.3).max())
        order = np.argsort(xs + ys*0.3)
        path = np.c_[ys, xs][order][::max(1, len(order)//12)].astype(float)
        if len(path) < 2:
            path = np.array([[ys.mean(), xs.mean()], [ys.mean(), xs.mean() + 1]])
        out.append(dict(path=path, r=np.full(len(path), max(rr*1.3, 2.0)), kind="dot"))
    return out


def plan(letters, fps, t0=0.0, speed=None, seed=1, factor=1.0):
    """Штрихи со временем. Постоянная скорость пера; отрыв ~0.02 с внутри буквы и между
    буквами, ~0.12 с между словами; короткий штрих — не короче кадра (без растягивания
    до двух: это давало пустые кадры и «пульс»). Замер 04.10: 0.9–1.4 с на слово, пустых
    кадров 7–12% против 16–24%. Возвращает (события, длительность)."""
    rng = np.random.default_rng(seed)
    ev, t, prev_word = [], 0.0, None
    for L in letters:
        sp = (speed or SPEED_PER_SIZE*L["size"])*factor
        # штрихи буквы — чистая функция её раскладки (не зависят от fps/t0/скорости): считаются один раз на
        # букву, план берёт копии (ревью 08.10: четыре пересчёта на клип с мыслью — 57% времени планирования)
        if "_strokes" not in L:
            L["_strokes"] = letter_strokes(L)
        sts = [dict(st) for st in L["_strokes"]]
        for j, s in enumerate(sts):
            if j == 0:
                gap = 0.0 if prev_word is None else (rng.uniform(0.11, 0.14) if L["word"] != prev_word else rng.uniform(0.012, 0.024))
            elif s["kind"] == "dot" or np.linalg.norm(s["path"][0] - sts[j - 1]["path"][-1]) > 4:
                gap = rng.uniform(0.012, 0.024)
            else:
                gap = 0.0
            ln = _length(s["path"])
            dur = max(1/fps, ln/(sp*rng.uniform(0.9, 1.1))) if s["kind"] == "stroke" else 1.5/fps
            t += gap
            s.update(t0=t0 + t, t1=t0 + t + dur, len=ln, pen=(gap > 0 or j == 0))
            t += dur
            ev.append(s)
        prev_word = L["word"]
    return ev, t


def ease(u):
    """Скорость руки: среднее между ровной и «минимальным рывком» — пик ~1.4 средней."""
    u = np.clip(u, 0, 1)
    return 0.5*u + 0.5*u**3*(10 - 15*u + 6*u*u)


def pressure(u):
    a = np.clip(u/0.10, 0, 1); b = np.clip((1 - u)/0.18, 0, 1)
    return (0.55 + 0.45*np.sqrt(a))*(0.62 + 0.38*np.sqrt(b))


class Ink:
    """Альфа надписи во времени. Рисует только новые отпечатки (цена кадра не растёт со
    словами); отпечатки текущего кадра — отдельным слоем с затуханием к кончику пера."""
    def __init__(self, letters, events, H, W, mask=None, tone=0.86, seed=5):
        self.H, self.W, self.ev, self.tone = H, W, events, tone
        if mask is None:
            mask = np.zeros((H, W), np.float32)
            for L in letters:
                np.maximum(mask, L["soft"], out=mask)
                for d in L["dots"]:
                    np.maximum(mask, d, out=mask)
        self.mask = mask.astype(np.float32)
        ys, xs = np.nonzero(self.mask > 0.01)
        self.box = ((max(0, ys.min() - 24), min(H, ys.max() + 25), max(0, xs.min() - 24), min(W, xs.max() + 25))
                    if len(ys) else (0, 1, 0, 1))
        y0, y1, x0, x1 = self.box
        self.cov = np.zeros((y1 - y0, x1 - x0), np.float32)
        self.front = np.zeros_like(self.cov)
        self.done = [0]*len(events); self.closed = [False]*len(events)
        rng = np.random.default_rng(seed)
        g = ndimage.gaussian_filter(rng.normal(0, 1, self.cov.shape).astype(np.float32), 0.55)
        st = ndimage.gaussian_filter(rng.normal(0, 1, self.cov.shape).astype(np.float32), (0.7, 2.2))
        self.grain = np.clip(0.93 + 0.05*g/(g.std() + 1e-9) + 0.04*st/(st.std() + 1e-9), 0.72, 1.0)
        self.t_end = max((e["t1"] for e in events), default=0.0)

    def _stamp(self, layer, x, y, r, a):
        y0, _, x0, _ = self.box; x -= x0; y -= y0; h, w = layer.shape
        a0, a1, b0, b1 = max(0, int(y - r - 2)), min(h, int(y + r + 3)), max(0, int(x - r - 2)), min(w, int(x + r + 3))
        if a1 <= a0 or b1 <= b0:
            return
        yy, xx = np.mgrid[a0:a1, b0:b1]
        v = np.clip(r + 0.7 - np.sqrt((xx - x)**2 + (yy - y)**2), 0, 1)*a
        sub = layer[a0:a1, b0:b1]
        sub += v*(1 - sub)

    def _radius(self, e, j, N):
        u = j/max(1, N - 1)
        start = 0.55 + 0.45*min(1.0, u/0.06)                  # сужение на входе — без «точки»
        rk = e.get("rk")
        return e["r"][j]*(rk*(0.55 + 0.45*pressure(u)) if rk else 1.12)*start

    def advance(self, t):
        self.cov += self.front*(1 - self.cov)                 # прошлый кончик уплотняется
        self.front[:] = 0
        for i, e in enumerate(self.ev):
            N = len(e["path"])
            if t <= e["t0"]:
                continue
            u = 1.0 if t >= e["t1"] else ease((t - e["t0"])/(e["t1"] - e["t0"]))
            n = max(1, int(round(u*N)))
            if n > self.done[i]:
                k0 = self.done[i]
                for j in range(k0, n):
                    y, x = e["path"][j]
                    fade = 1.0 - 0.45*((j - k0)/max(1, n - k0))     # кончик пера светлее
                    self._stamp(self.front, x, y, self._radius(e, j, N), (0.42 + 0.22*pressure(j/max(1, N - 1)))*fade)
                self.done[i] = n
            if t >= e["t1"] and not self.closed[i]:                  # штрих готов: форма закрыта целиком
                for j in range(0, N, 2):
                    y, x = e["path"][j]
                    self._stamp(self.cov, x, y, e["r"][j]*1.12, 0.3)
                self.closed[i] = True

    def alpha(self):
        y0, y1, x0, x1 = self.box
        a = np.zeros((self.H, self.W), np.float32)
        c = self.cov + self.front*(1 - self.cov)
        a[y0:y1, x0:x1] = np.clip(c*1.15, 0, 1)*self.mask[y0:y1, x0:x1]*self.grain*self.tone
        return a
