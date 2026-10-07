"""Кукла Чирка: риг из одного рисунка + автоматика движений. CPU, без нейросетей в рантайме.

Все движения — изгибы ОДНОЙ картинки (сетка с весами) и подмена частей глаза,
поэтому за движущейся частью никогда не открывается недорисованное место:
- глаза — отдельные части (мех без глаз, радужка, зрачок), моргание сжимает глаз к веку;
- голова / ухо / хвост / лапы — поворот или подъём с плавным весом, тело остаётся на месте;
- под поднятой лапой только бумага (лапа на полу, низ — край силуэта).
Действия задаются списком (сценарий), фон жизни (моргание, ухо, взгляд, хвост,
дыхание, огонёк) генерируется сам по зерну — повторяемо.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, subprocess, sys, time, numpy as np, cv2
cv2.setNumThreads(1)
from multiprocessing import Pool
from PIL import Image

def remap(img, X, Y, border=cv2.BORDER_CONSTANT):
    return cv2.remap(np.ascontiguousarray(img, np.float32), np.asarray(X, np.float32), np.asarray(Y, np.float32), cv2.INTER_LINEAR, borderMode=border)

def smooth(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

class Rig:
    def __init__(self, path="rig.json"):
        r = json.load(open(path)); self.r = r
        self.src = np.asarray(Image.open(r["image"]).convert("RGB")).astype(np.float32)
        self.alpha = np.asarray(Image.open(r["mask"])).astype(np.float32)
        self.fur = np.asarray(Image.open(r["fur"])).astype(np.float32)
        self.iris = np.asarray(Image.open(r["iris"])).astype(np.float32)
        self.E = np.load(r["eyes"]); self.M = np.load(r["eye_masks"])
        H, W = self.alpha.shape; self.H, self.W = H, W
        m = self.alpha > 128
        self.paper = np.median(self.src[~m], axis=0)
        ys, xs = np.nonzero(self.alpha > 1)
        pad = 60
        self.bb = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + pad), min(H, ys.max() + pad))
        x0, y0, x1, y1 = self.bb
        YY, XX = np.mgrid[y0:y1, x0:x1].astype(np.float32); self.XX, self.YY = XX, YY
        self.w_head = smooth((r["head_below_y"] - YY) / 55) * (XX < r["tail_from_x"] + 50)
        eb, et = np.array(r["ear_base"], np.float32), np.array(r["ear_tip"], np.float32); ev = et - eb
        self.w_ear = smooth((((XX - eb[0]) * ev[0] + (YY - eb[1]) * ev[1]) / (ev @ ev)) * 1.3) * \
            smooth((100 - np.hypot(XX - r["ear_center"][0], YY - r["ear_center"][1])) / 25)
        self.w_tail = smooth((XX - r["tail_from_x"]) / 140) * (YY > 260)
        self.w_paw = {}
        for k, p in r["paws"].items():
            sh, ft, hw = np.array(p["shoulder"], np.float32), np.array(p["foot"], np.float32), p["half_width"]
            along = smooth((YY - sh[1]) / (ft[1] - sh[1]))                 # 0 у плеча, 1 у ступни
            across = smooth((hw - np.abs(XX - ft[0])) / 18)
            self.w_paw[k] = along * across
        def box(w):
            ys, xs = np.nonzero(w > 1e-4)
            return (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)) if len(ys) else None
        self.sl = {"head": box(self.w_head), "ear": box(self.w_ear), "tail": box(self.w_tail)}
        self.sl.update({"paw_" + k: box(w) for k, w in self.w_paw.items()})
        # поза «машет»: тело с поднятой лапой (платная правка эталона + убранная лишняя ступня)
        self.wave = None
        if r.get("wave"):
            wv = r["wave"]
            rs = np.asarray(Image.open(wv["image"]).convert("RGB")).astype(np.float32)
            ra = np.asarray(Image.open(wv["mask"])).astype(np.float32)
            Hm, Wm = np.mgrid[0:H, 0:W]
            from scipy import ndimage
            arm = (ra > 128) & ~ndimage.binary_dilation(self.alpha > 128, iterations=2) & (Wm < wv["arm_max_x"]) & (Hm < wv["arm_max_y"])
            lab, n = ndimage.label(arm); keep = np.argmax(ndimage.sum(arm, lab, range(1, n + 1))) + 1
            arm = ndimage.binary_fill_holes(ndimage.binary_dilation(lab == keep, iterations=4)) & (ra > 8)
            arm_a = (ndimage.gaussian_filter(arm.astype(np.float32), 0.8) * ra).astype(np.float32)
            outside = ~ndimage.binary_dilation(self.alpha > 128, iterations=1)   # вне силуэта эталона — только лапа
            body_a = np.where(outside & ndimage.binary_dilation(arm, iterations=3), 0, ra).astype(np.float32)
            fur_r = rs.copy()                                  # морда та же, что в эталоне: глаза — из системы глаз
            for (mi, mw, mp) in self.M:
                ww = ndimage.binary_dilation(mw, iterations=8); fur_r[ww] = self.fur[ww]
            pv = np.array(wv["pivot"], np.float32); tip = np.array(wv["tip"], np.float32); v = tip - pv
            wgt = smooth((((Wm - pv[0]) * v[0] + (Hm - pv[1]) * v[1]) / (v @ v)) * 1.6).astype(np.float32)
            self.wave = dict(fur=fur_r, body_a=body_a, arm=np.dstack([rs, arm_a]), w=wgt, pivot=pv, down=wv["down_deg"])
        # части глаза — только в рамке глаз
        self.eye_bb = []
        for (cx, cy, rx, ry) in self.E:
            self.eye_bb.append((int(cx - rx - 14), int(cy - ry - 14), int(cx + rx + 15), int(cy + ry + 15)))

    # ------------------------------------------------------------- глаза
    def face(self, look, lid, base=None):
        img = (self.fur if base is None else base).copy()
        for (cx, cy, rx, ry), (mi, mw, mp), (bx0, by0, bx1, by1) in zip(self.E, self.M, self.eye_bb):
            sl = (slice(by0, by1), slice(bx0, bx1))
            yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
            dx, dy = look[0] * rx * .3, look[1] * ry * .3
            src4 = np.dstack([self.src[sl], mp[sl].astype(np.float32) * 255])
            pup = remap(src4, xx - bx0 - dx, yy - by0 - dy)
            pa = (pup[..., 3:4] / 255) * mi[sl][..., None]
            eye = self.iris[sl] * (1 - pa) + pup[..., :3] * pa
            sy = max(1 - lid, 1e-3); l0 = cy + ry * .24
            e4 = np.dstack([eye, mw[sl].astype(np.float32) * 255])
            e2 = remap(e4, xx - bx0, (l0 + (yy - l0) / sy) - by0)
            ea = e2[..., 3:4] / 255
            img[sl] = img[sl] * (1 - ea) + e2[..., :3] * ea
            if lid > .85:
                c = img[sl].copy()
                cv2.ellipse(c, (int(cx - bx0), int(l0 - ry * .15 - by0)), (int(rx * .85), int(ry * .4)), 0, 15, 165,
                            (24, 22, 24), 7, cv2.LINE_AA)
                img[sl] = c
        return img

    # ------------------------------------------------------------- кадр
    def frame(self, st, t):
        r = self.r; x0, y0, x1, y1 = self.bb
        wave = st.get("wave")                       # None или (подъём 0..1, угол маха)
        if wave and self.wave and wave[0] >= 0.35:
            wv = self.wave
            cat = np.dstack([self.face(st["look"], st["lid"], wv["fur"]), wv["body_a"]])
            k = (wave[0] - .35) / .65
            settle = wv["down"] * np.exp(-4 * k) * np.cos(7 * k)               # доворот с отскоком
            ang = np.radians(settle + wave[1])
            Hh, Ww = self.H, self.W; Ym, Xm = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
            a2 = -ang * wv["w"]; c, s_ = np.cos(a2), np.sin(a2); dx, dy = Xm - wv["pivot"][0], Ym - wv["pivot"][1]
            armw = remap(wv["arm"], wv["pivot"][0] + c * dx - s_ * dy, wv["pivot"][1] + s_ * dx + c * dy)
            aa = armw[..., 3:4] / 255
            cat = np.dstack([cat[..., :3] * (1 - aa) + armw[..., :3] * aa, np.maximum(cat[..., 3], armw[..., 3])])
        else:
            cat = np.dstack([self.face(st["look"], st["lid"]), self.alpha])
        fx0, fy0, fx1, fy1 = r["flame_box"]; hh = fy1 - fy0                 # огонёк — в своих координатах
        yy, xx = np.mgrid[0:hh, 0:fx1 - fx0].astype(np.float32); ku = (1 - yy / hh) ** 1.5
        dx = ku * (4 * np.sin(2 * np.pi * (yy / 45 - t * 2.3)) + 2.5 * np.sin(2 * np.pi * (t * 3.7 + .3)))
        dy = ku * 3 * np.sin(2 * np.pi * t * 4.1)
        cat[fy0:fy1, fx0:fx1] = remap(cat[fy0:fy1, fx0:fx1], xx - dx, yy + dy, cv2.BORDER_REPLICATE)
        X, Y = self.XX.copy(), self.YY.copy()
        def rot(X, Y, p, deg, w, sl):
            """Поворот с весом только там, где вес ненулевой (in place)."""
            if sl is None or abs(deg) < 1e-3: return
            ang = np.float32(-np.radians(deg)) * w[sl]
            c, s = np.cos(ang), np.sin(ang); dx, dy = X[sl] - p[0], Y[sl] - p[1]
            X[sl] = p[0] + c * dx - s * dy; Y[sl] = p[1] + s * dx + c * dy
        b = st["breath"]; fl = r["floor_y"]
        Y[:] = fl - (fl - Y) / b
        for k, (lift, swing) in st["paws"].items():
            sl = self.sl["paw_" + k]; w = self.w_paw[k]
            rot(X, Y, r["paws"][k]["shoulder"], swing, w, sl)
            if abs(lift) > 1e-3: Y[sl] += np.float32(lift) * w[sl]       # ступня вверх, плечо на месте
        rot(X, Y, r["tail_base"], st["tail"], self.w_tail, self.sl["tail"])
        rot(X, Y, r["ear_base"], st["ear"], self.w_ear, self.sl["ear"])
        rot(X, Y, r["neck"], st["head"], self.w_head, self.sl["head"])
        c = remap(cat, X, Y)
        out = np.empty((self.H, self.W, 3), np.float32); out[:] = self.paper
        a = c[..., 3:4] / 255
        out[y0:y1, x0:x1] = self.paper * (1 - a) + c[..., :3] * a
        return np.clip(out, 0, 255).astype(np.uint8)

# ----------------------------------------------------------------- движения
def key(t, pts, default=0.0):
    if not pts: return default
    if t <= pts[0][0]: return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1: return v0 + (v1 - v0) * float(smooth((t - t0) / max(t1 - t0, 1e-6)))
    return pts[-1][1]

def plan(dur, actions, seed=7):
    """Сценарий -> ключи. Фон жизни генерируется сам, действия из списка его перекрывают."""
    rng = np.random.default_rng(seed)
    blinks, t = [], rng.uniform(.8, 2.0)
    while t < dur - .3:
        blinks.append(t)
        if rng.random() < .15 and t + .35 < dur: blinks.append(t + .32)     # иногда двойное
        t += rng.uniform(2.5, 5.5)
    ears, t = [], rng.uniform(3, 6)
    while t < dur - .4: ears.append(t); t += rng.uniform(6, 11)
    look_x, look_y, head = [(0, 0.)], [(0, 0.)], [(0, 0.)]
    t = rng.uniform(1.5, 3)
    while t < dur - 1:                                                       # «оглядывание» сам по себе
        g = float(rng.choice([-.6, -.3, .3, .6])); hold = rng.uniform(1.0, 2.2)
        look_x += [(t, look_x[-1][1]), (t + .25, g), (t + .25 + hold, g), (t + .5 + hold, 0.)]
        t += hold + rng.uniform(2.5, 5)
    paws = {"left": ([], []), "right": ([], [])}
    WV = []
    for a in actions:
        t0, kind = a["t"], a["do"]
        if kind == "look":                      # смотреть на точку: x,y в [-1..1]
            d = a.get("dur", 1.5)
            look_x = [p for p in look_x if not (t0 - .3 <= p[0] <= t0 + d + .3)]
            look_y = [p for p in look_y if not (t0 - .3 <= p[0] <= t0 + d + .3)]
            look_x += [(t0, key(t0, look_x)), (t0 + .22, a["x"]), (t0 + d, a["x"]), (t0 + d + .3, 0.)]
            look_y += [(t0, 0.), (t0 + .22, a.get("y", 0.)), (t0 + d, a.get("y", 0.)), (t0 + d + .3, 0.)]
            look_x.sort(); look_y.sort()
        elif kind == "tilt":                    # наклон головы, градусы
            d = a.get("dur", 1.5)
            head += [(t0, 0.), (t0 + .5, a["deg"]), (t0 + .5 + d, a["deg"]), (t0 + 1.1 + d, 0.)]
        elif kind == "tap":                     # постучать лапой n раз
            L = paws[a.get("paw", "right")][0]
            for i in range(a.get("n", 2)):
                s = t0 + i * .42; L += [(s, 0.), (s + .14, 18.), (s + .3, 0.)]
        elif kind == "point":                   # показать лапой в сторону (± градусы)
            S, L = paws[a.get("paw", "right")][1], paws[a.get("paw", "right")][0]
            d = a.get("dur", 1.4); deg = a.get("deg", -12)
            S += [(t0, 0.), (t0 + .35, deg), (t0 + .35 + d, deg), (t0 + .8 + d, 0.)]
            L += [(t0, 0.), (t0 + .35, 12.), (t0 + .35 + d, 12.), (t0 + .8 + d, 0.)]
        elif kind == "wave":                    # помахать: подъём 0.2 с, n взмахов, опускание
            n = a.get("n", 2); up = .2; per = .42
            WV.append((t0, t0 + up, t0 + up + n * per, t0 + up + n * per + .22, per))
        elif kind == "blink":
            blinks.append(t0)
    blinks.sort(); head.sort()
    def state(t):
        lid = 0.
        for b in blinks:
            d = t - b
            if 0 <= d < .08: lid = max(lid, d / .08)
            elif .08 <= d < .14: lid = 1.
            elif .14 <= d < .26: lid = max(lid, 1 - (d - .14) / .12)
        ear = 0.
        for e in ears:
            for t1 in (e, e + .25):
                d = (t - t1) / .11
                if 0 <= d < 2: ear += -10 * np.sin(np.pi * d / 2) ** 2
        wave = None
        for (a0, a1, a2_, a3, per) in WV:
            if a0 <= t <= a3:
                lift = smooth((t - a0) / (a1 - a0)) if t < a1 else (1 - smooth((t - a2_) / (a3 - a2_)) if t > a2_ else 1.)
                sw = 12 * np.sin(2 * np.pi * (t - a1) / per) if a1 <= t <= a2_ else 0.
                wave = (float(lift), float(sw))
        return dict(wave=wave, look=(key(t, look_x), key(t, look_y)), lid=lid, head=key(t, head), ear=ear,
                    tail=4 * np.sin(2 * np.pi * t / 3.1), breath=1 + .012 * np.sin(2 * np.pi * t / 2.6),
                    paws={k: (key(t, sorted(v[0])), key(t, sorted(v[1]))) for k, v in paws.items()})
    return state

_RIG = None
_ST = None
def _work(args):
    global _RIG
    if _RIG is None: _RIG = Rig()
    i, fps, dur, actions = args
    global _ST
    if _ST is None: _ST = plan(dur, actions)
    return _RIG.frame(_ST(i / fps), i / fps).tobytes()

def render(out, dur, actions, fps=30, workers=4):
    rig = Rig(); H, W = rig.H, rig.W
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                          "-vf", "scale=1920:-2:flags=lanczos,crop=1920:1080", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
                          "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    t = time.time(); n = int(fps * dur)
    with Pool(workers) as pool:
        for b in pool.imap(_work, [(i, fps, dur, actions) for i in range(n)], chunksize=8): p.stdin.write(b)
    p.stdin.close(); p.wait()
    return time.time() - t

if __name__ == "__main__":
    ACT = [{"t": 1.2, "do": "look", "x": .9, "y": .3, "dur": 1.6},       # смотрит на письмо справа внизу
           {"t": 1.6, "do": "tilt", "deg": 5, "dur": 1.2},
           {"t": 3.6, "do": "point", "paw": "right", "deg": -12, "dur": 1.3},
           {"t": 6.0, "do": "tap", "paw": "left", "n": 3},
           {"t": 7.6, "do": "look", "x": 0, "y": 0, "dur": .5}]
    sec = render("doll_auto.mp4", 9.0, ACT)
    print(f"рендер 9 с: {sec:.1f} с, {sec / 9:.2f} с на секунду ролика")
