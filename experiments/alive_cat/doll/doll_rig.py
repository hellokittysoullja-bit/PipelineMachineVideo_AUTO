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
import json
import subprocess
import time
import numpy as np
import cv2
from scipy import ndimage
cv2.setNumThreads(1)
from multiprocessing import Pool
from PIL import Image


FLAME_NOISE = os.environ.get("FLAME_NOISE", "1") == "1"
TAIL_NOISE = os.environ.get("TAIL_NOISE", "1") == "1"
HIGHLIGHT_FIXED = os.environ.get("HIGHLIGHT_FIXED", "1") == "1"  # блик в глазу стоит на месте, зрачок ходит под ним
LID_ANCHOR = float(os.environ.get("LID_ANCHOR", "0.24"))   # куда сходится глаз при закрытии, в долях радиуса от центра вниз
LID_COVER = float(os.environ.get("LID_COVER", "0.84"))      # с какого lid штрих века накрывает остаток щёлки (к 0.95 — целиком)      # хвост: fBm вместо чистой синусоиды с периодом 3.1 с
BLINK_ON_GAZE = os.environ.get("BLINK_ON_GAZE", "1") == "1"  # большой перевод взгляда сопровождается морганием

def _vnoise(x, seed):
    """1D value-шум: случайные значения в целых точках, гладкая (smoothstep) интерполяция, диапазон -1..1."""
    i = np.floor(x); f = x - i; i = i.astype(np.int64)
    def h(n):
        n = (n * 374761393 + seed * 668265263) & 0xFFFFFFFF
        n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
        return (n & 0xFFFF) / 32767.5 - 1.0
    u = f * f * (3 - 2 * f)
    return (h(i) * (1 - u) + h(i + 1) * u).astype(np.float32)

def fbm(x, seed, octaves=3):
    """Сумма октав, нормирована к амплитуде синуса (~-1..1)."""
    tot = np.zeros_like(np.asarray(x, np.float32)); a = 1.0; fr = 1.0; norm = 0.0
    for o in range(octaves):
        tot = tot + a * _vnoise(np.asarray(x, np.float32) * fr, seed + 101 * o); norm += a; a *= .5; fr *= 2.0
    return tot / norm * 1.9

# Бикубика вместо билинейной: билинейная выборка при сдвиге на долю пикселя мылит рисунок, и при
# дыхании резкость меха пульсировала на 27% (резко ровно при масштабе 1, мягко в остальные моменты).
# Бикубика — 8%. В покое (сдвиг 0) обе дают исходник один в один.
REMAP_INTERP = cv2.INTER_CUBIC if os.environ.get("REMAP_CUBIC", "1") == "1" else cv2.INTER_LINEAR

def remap(img, X, Y, border=cv2.BORDER_CONSTANT):
    return cv2.remap(np.ascontiguousarray(img, np.float32), np.asarray(X, np.float32), np.asarray(Y, np.float32), REMAP_INTERP, borderMode=border)


def _lama_fill(model, rgb, al, mask, crop=128):
    """Дорисовать под маской LaMa (ONNX, 512x512): цвет и прозрачность отдельно. Кроп crop px
    вокруг маски увеличивается до 512, пиксели под маской обнуляются до прогона."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model)
    except Exception:
        return None
    ys, xs = np.nonzero(mask); cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    H, W = mask.shape; y0 = int(np.clip(cy - crop // 2, 0, H - crop)); x0 = int(np.clip(cx - crop // 2, 0, W - crop))
    sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
    m = cv2.resize(mask[sl].astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    def run(img3):
        im = cv2.resize(np.clip(img3, 0, 255), (512, 512), interpolation=cv2.INTER_CUBIC) / 255 * (1 - m[..., None])
        o = sess.run(None, {"image": im.transpose(2, 0, 1)[None].astype(np.float32), "mask": m[None, None]})[0][0].transpose(1, 2, 0)
        if o.max() <= 1.5: o = o * 255
        return cv2.resize(o.astype(np.float32), (crop, crop), interpolation=cv2.INTER_AREA)
    B = rgb.copy(); Ba = al.copy()
    B[sl][mask[sl]] = run(rgb[sl])[mask[sl]]
    Ba[sl][mask[sl]] = run(np.repeat(al[sl][..., None], 3, 2))[..., 0][mask[sl]]
    return B, np.clip(Ba, 0, 255)

# состояние огонька: (масштаб по высоте, по ширине, множитель цвета RGB) — look/hero_states.json
FLAME_STATES = {"ember": (0.55, 0.8, (0.72, 0.5, 0.42)), "golden": (1.15, 1.05, (1.0, 1.22, 0.8))}


def _load_npy(path):
    """npy, а при его отсутствии — npz с тем же именем (в репозитории маски глаз лежат сжатыми: 6.4 МБ -> десятки КБ)."""
    if os.path.exists(path): return np.load(path)
    alt = os.path.splitext(path)[0] + ".npz"
    if os.path.exists(alt): return np.load(alt)["arr"]
    raise FileNotFoundError(path)

def smooth(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

class Rig:
    def __init__(self, path="rig.json"):
        r = json.load(open(path)); self.r = r
        base = os.path.dirname(os.path.abspath(path))      # ассеты — рядом с rig.json, а не от cwd
        P = lambda k: r[k] if os.path.isabs(r[k]) else os.path.join(base, r[k])
        self.src = np.asarray(Image.open(P("image")).convert("RGB")).astype(np.float32)
        self.alpha = np.asarray(Image.open(P("mask"))).astype(np.float32)
        self.fur = np.asarray(Image.open(P("fur"))).astype(np.float32)
        self.base4 = np.dstack([self.fur, self.alpha]).astype(np.float32)
        self.iris = np.asarray(Image.open(P("iris"))).astype(np.float32)
        self.E = _load_npy(P("eyes")); self.M = _load_npy(P("eye_masks"))
        if r.get("lama") and not os.path.isabs(r["lama"]): r["lama"] = os.path.join(base, r["lama"])
        for pz in list((r.get("poses") or {}).values()) + ([r["wave"]] if isinstance(r.get("wave"), dict) else []):
            for k in ("image", "mask"):                      # пути поз и взмаха тоже от папки рига
                if pz.get(k) and not os.path.isabs(pz[k]): pz[k] = os.path.join(base, pz[k])
        H, W = self.alpha.shape; self.H, self.W = H, W
        m = self.alpha > 128
        self.paper = np.median(self.src[~m], axis=0)
        self.paper_src = self.paper.copy()               # бумага исходника: с ней смешаны края меха
        self.paper_u8 = np.empty((H, W, 3), np.uint8); self.paper_u8[:] = np.clip(self.paper, 0, 255).astype(np.uint8)
        ys, xs = np.nonzero(self.alpha > 1)
        pad = 60
        self.bb = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + pad), min(H, ys.max() + pad))
        x0, y0, x1, y1 = self.bb
        YY, XX = np.mgrid[y0:y1, x0:x1].astype(np.float32); self.XX, self.YY = XX, YY
        self.paper_bb = np.empty((y1 - y0, x1 - x0, 3), np.float32); self.paper_bb[:] = self.paper
        import hashlib
        key = hashlib.sha1(self.alpha.tobytes() + json.dumps([r["head_below_y"], r["neck"], r["flame_box"], r["fur"]]).encode()
                           + __import__("inspect").getsource(Rig._head_layer).encode()).hexdigest()[:16]
        cache = os.path.join(os.path.dirname(os.path.abspath(path)), f".head_layer_{key}.npz")
        if os.path.exists(cache):
            z = np.load(cache); hm = z["hm"]
            self.tip_patch = None if "sy" not in z else dict(sy=slice(*z["sy"]), sx=slice(*z["sx"]), mask=z["mask"], body=z["body"], head=z["head"])
        else:
            hm = self._head_layer(r); tp = self.tip_patch
            extra = {} if tp is None else dict(sy=[tp["sy"].start, tp["sy"].stop], sx=[tp["sx"].start, tp["sx"].stop], mask=tp["mask"], body=tp["body"], head=tp["head"])
            np.savez_compressed(cache, hm=hm, **extra)
        self._hm = hm                                              # слой головы; остальное — тело с хвостом

        self.w_head = smooth((r["head_below_y"] - YY) / 55)          # голова целиком (по ширине не режется)
        eb, et = np.array(r["ear_base"], np.float32), np.array(r["ear_tip"], np.float32); ev = et - eb
        self.w_ear = smooth((((XX - eb[0]) * ev[0] + (YY - eb[1]) * ev[1]) / (ev @ ev)) * 1.3) * \
            smooth((100 - np.hypot(XX - r["ear_center"][0], YY - r["ear_center"][1])) / 25)
        self.w_tail = smooth((XX - r["tail_from_x"]) / 140) * (YY > 260)
        self.head_rows = int(np.nonzero((self.w_head > 0).any(axis=1))[0].max()) + 1   # ниже — вес головы 0
        # прямоугольник стыка: где оба веса ненулевые, с запасом на ход (40 px) — по нему слои
        both = (self.w_head > 0) & (self.w_tail > 0); ys_, xs_ = np.nonzero(both); mg = 40
        self.conflict = (slice(max(0, ys_.min() - mg), min(YY.shape[0], ys_.max() + mg)),
                         slice(max(0, xs_.min() - mg), YY.shape[1]))
        cy, cx = self.conflict; ms = 120                           # окно источника: ход до 120 px
        self.conflict_src = (slice(max(0, y0 + cy.start - ms), min(self.H, y0 + cy.stop + ms)),
                             slice(max(0, x0 + cx.start - ms), min(self.W, x0 + cx.stop + ms)))
        self.m_head_src = np.repeat(self._hm[self.conflict_src].astype(np.float32)[..., None], 4, axis=2)
        fx0, fy0, fx1, fy1 = r["flame_box"]; hh = fy1 - fy0; mg = 30
        X0b, Y0b, X1b, Y1b = max(0, fx0 - mg), max(0, fy0 - mg), min(W, fx1 + mg), fy1
        yb, xb = np.mgrid[Y0b:Y1b, X0b:X1b].astype(np.float32)
        yy = yb - fy0; xx = xb - fx0                                        # координаты прежней рамки
        ku = np.clip(1 - yy / hh, 0, 1) ** 1.5
        edge = np.minimum.reduce([xb - X0b, (X1b - 1) - xb, yb - Y0b]) / 20.0
        from scipy import ndimage as _nd
        not_head = 1 - _nd.gaussian_filter(self._hm[Y0b:Y1b, X0b:X1b].astype(np.float32), 1.5)   # усы с пламенем не дрожат
        self.flame_geo = dict(box=(X0b, Y0b, X1b, Y1b), yy=yy, xx=xx, ku=ku.astype(np.float32),
                              w=(smooth(edge) * not_head).astype(np.float32))
        # состояние огонька (look/hero_states.json): маска самого пламени — тёплые насыщенные пиксели в рамке;
        # основание — нижний ряд пламени (где оно сидит на кончике хвоста), выше него пламя масштабируется
        fr = np.clip(self.fur[Y0b:Y1b, X0b:X1b], 0, 255).astype(np.uint8); fa = self.alpha[Y0b:Y1b, X0b:X1b]
        hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
        # ядро пламени — яркое насыщенное тёплое (S>110, V>150): оранжевое свечение на тёмном мехе хвоста
        # темнее (V медиана 182 против 244) и в ядро не попадает — первая версия красила его в бурый; берётся самая крупная связная область
        core = (hsv[..., 1] > 110) & (hsv[..., 2] > 210) & (hsv[..., 0] < 30) & (fa > 100)   # замер: пламя V≈244, свечение на хвосте V≈182
        lab, n = _nd.label(core)
        if n:
            core = lab == (np.argmax(_nd.sum(core, lab, range(1, n + 1))) + 1)
            fys, fxs = np.nonzero(core)
            base_y = float(yb[fys.max(), 0]) + 2.0
            # ядро + тушь контура: всё нарисованное над основанием в полосе пламени (расширение от ядра
            # не дотягивалось до туши у кончика, где пламя сужается — оставался призрак контура); усы — не пламя
            hw = (fxs.max() - fxs.min()) / 2 + 22
            outline = (fa > 3) & (yb < base_y + 1) & (np.abs(xb - xb[0, fxs].mean()) < hw) & \
                (self._hm[Y0b:Y1b, X0b:X1b] == 0)
            outline = _nd.binary_dilation(outline, iterations=4) & (yb < base_y + 1)   # край размытия — на бумаге, не на туши; хвост под основанием не трогаем
            self.flame_state = dict(mask=_nd.gaussian_filter(outline.astype(np.float32), 1.0).astype(np.float32),
                                    core=_nd.gaussian_filter(core.astype(np.float32), 1.0).astype(np.float32),
                                    base=base_y, top=float(yb[fys.min(), 0]), cx=float(xb[0, fxs].mean()), yb=yb, xb=xb)
        else:
            self.flame_state = None
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
        self.poses = {}
        for name, pz in (r.get("poses") or {}).items():
            rs = np.asarray(Image.open(pz["image"]).convert("RGB")).astype(np.float32)
            ra = np.asarray(Image.open(pz["mask"])).astype(np.float32)
            fr = rs.copy()
            for (mi, mw, mp) in self.M:
                ww = ndimage.binary_dilation(mw, iterations=8); fr[ww] = self.fur[ww]
            self.poses[name] = dict(fur=fr, alpha=ra, base4=np.dstack([fr, ra]).astype(np.float32))
        # части глаза — только в рамке глаз
        self.eye_bb = []
        for (cx, cy, rx, ry) in self.E:
            self.eye_bb.append((int(cx - rx - 14), int(cy - ry - 14), int(cx + rx + 15), int(cy + ry + 15)))
        # блик — отражение света на роговице: стоит на месте, когда зрачок уходит в сторону.
        # Раньше он был частью слоя зрачка и ездил вместе с ним. Вынимаем его из зрачка один раз.
        self.pupil_src = []; self.hl = []
        from scipy import ndimage as _nd
        for (cx, cy, rx, ry), (mi, mw, mp), (bx0, by0, bx1, by1) in zip(self.E, self.M, self.eye_bb):
            sl = (slice(by0, by1), slice(bx0, bx1))
            src = self.src[sl].copy(); lum = src @ np.float32([.299, .587, .114])
            sat = src.max(2) - src.min(2)
            white = (lum > 150) & (sat < 60) & _nd.binary_dilation(mp[sl], iterations=2)   # белое, не зелёная радужка
            lab_, n_ = _nd.label(white); core = np.zeros_like(white)
            if n_:
                sizes = _nd.sum(white, lab_, range(1, n_ + 1)); k_ = int(np.argmax(sizes)) + 1
                if sizes[k_ - 1] >= 20: core = lab_ == k_                            # одно пятно блика, не крапинки края
            if HIGHLIGHT_FIXED and core.any():
                reg = _nd.binary_dilation(core, iterations=2)
                a = np.clip((lum - 60) / 110, 0, 1) * np.clip((90 - sat) / 40, 0, 1) * reg   # мягкий край блика, без зелени радужки
                pm = mp[sl].astype(np.uint8)
                dark = np.median(src[(pm > 0) & ~reg], axis=0)                       # цвет зрачка под бликом
                fill = src.copy(); fill[reg] = dark
                # форма зрачка под бликом неизвестна (маска рисовалась с бликом) — достроить эллипсом по остальному контуру
                cnts, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
                keep = ~_nd.binary_dilation(reg, iterations=1)[cnt[:, 1], cnt[:, 0]]
                pm2 = (pm > 0) & ~reg
                if keep.sum() >= 5:
                    ell = cv2.fitEllipse(cnt[keep].astype(np.float32)); em = np.zeros_like(pm)
                    cv2.ellipse(em, ell, 1, -1); pm2 |= (em > 0) & reg                 # эллипс только там, где был блик
                self.pupil_src.append(np.dstack([fill, pm2.astype(np.float32) * 255]))
                self.hl.append((src * a[..., None], a[..., None]))                   # премультиплицированный блик
            else:
                self.pupil_src.append(np.dstack([src, mp[sl].astype(np.float32) * 255])); self.hl.append(None)


    def _head_layer(self, r):
        """Маска слоя ГОЛОВЫ (1) — всё нарисованное над шеей, что принадлежит голове, включая усы.
        Голова и хвост — отдельные слои с отдельными жёсткими поворотами (а не одна деформация
        с общими весами): общая деформация резала щеку вертикалью, а с плавной границей —
        растягивала и гнула усы. Плотная голова и плотный хвост разделены бумагой (эрозия 2 px);
        каждый штрих достаётся той форме, до которой ближе ВДОЛЬ рисунка; тонкие штрихи, которые
        дорастают до хвоста (кончики усов), — голове целиком."""
        from scipy import ndimage
        hb = r["head_below_y"]
        S = self.alpha > 128; S[hb:] = False
        er = ndimage.binary_erosion(S, iterations=2)
        lab, n = ndimage.label(er)
        nx, ny = r["neck"]; fx0, fy0, fx1, fy1 = r["flame_box"]
        hl = lab[min(ny, hb - 3), nx]
        fx = (fx0 + fx1) // 2                                  # хвост — та форма, что под огоньком
        tail_pts = [(y, fx) for y in range(fy1, hb) if lab[y, fx] and lab[y, fx] != hl]
        tl = lab[tail_pts[0]] if tail_pts else 0
        if not hl or not tl or hl == tl:
            raise RuntimeError("голова и хвост не разделились по маске — проверь neck/flame_box в rig.json")
        ink = self.alpha > 1; ink[hb:] = False
        Hm, T = lab == hl, lab == tl
        for _ in range(400):
            free = ink & ~Hm & ~T
            if not free.any(): break
            gh = ndimage.binary_dilation(Hm) & free; gt = ndimage.binary_dilation(T) & free
            if not gh.any() and not gt.any(): break
            Hm |= gh & ~gt; T |= gt & ~gh; Hm |= gh & gt       # ничья — голове
        # полоса у хвоста (10 px от плотного хвоста, с кончиками пучков меха): всё в ней — хвосту (его мех и крапинки)
        band = ndimage.binary_dilation(lab == tl, iterations=10)
        Hm |= T & ~band; Hm &= ~band; Hm[hb:] = False
        # кусочки, оторванные от головы (кончики пучков меха хвоста, крапинки), — ближайшей плотной части
        pl, pn = ndimage.label(Hm, structure=np.ones((3, 3)))
        main = set(np.unique(pl[(lab == hl) & Hm])) - {0}
        dH = ndimage.distance_transform_edt(~(lab == hl)); dT = ndimage.distance_transform_edt(~(lab == tl))
        for k in range(1, pn + 1):
            if k in main: continue
            piece = pl == k
            if dT[piece].min() < dH[piece].min(): Hm &= ~piece
        # кончики усов лежат поверх меха хвоста: что под ними — неизвестно. Один раз при подготовке:
        # мех под кончиком дорисовывается (inpaint), а сам кончик вынимается разностью с этим фоном
        # (I = a·F + (1-a)·B) и уходит в слой головы — в покое кадр собирается как исходник.
        rgb, al = self.fur, self.alpha
        patch = np.zeros((self.H, self.W), bool)
        whisk = np.zeros((self.H, self.W), np.float32); fcol = []; geo = []
        gap = ink & ~ndimage.binary_dilation(lab == hl, iterations=4) & ~band
        wl, wn = ndimage.label(gap, structure=np.ones((3, 3)))
        dtail = ndimage.distance_transform_edt(~(lab == tl))
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        for i in range(1, wn + 1):
            py, px = np.nonzero(wl == i)
            if len(py) < 40 or dtail[py, px].min() > 13: continue               # ус, упирающийся в хвост
            lum_ = rgb[py, px] @ np.float32([.299, .587, .114])
            ok = (dtail[py, px] > 16) & (dtail[py, px] < 45) & (lum_ < 140)       # последний чистый участок уса (касательная)
            if ok.sum() < 20: continue
            P = np.stack([px[ok], py[ok]], 1).astype(np.float64); c = P.mean(0)
            _, sv, vt = np.linalg.svd(P - c, full_matrices=False); d = vt[0]
            if sv[1] > .15 * sv[0]: continue                                    # не прямая линия
            t = (P - c) @ d
            e_hi, e_lo = c + d * t.max(), c + d * t.min()
            if dtail[int(round(e_hi[1])), int(round(e_hi[0]))] > dtail[int(round(e_lo[1])), int(round(e_lo[0]))]:
                d = -d; t = -t                                                  # ось — к хвосту
            tend = t.max(); dend = dtail[int(round((c + d * tend)[1])), int(round((c + d * tend)[0]))]
            perp_all = np.abs(-(P[:, 0] - c[0]) * d[1] + (P[:, 1] - c[1]) * d[0])
            hw = float(np.clip(np.percentile(perp_all, 95) + .7, 1.2, 2.5))
            qx, qy = xx - c[0], yy - c[1]
            along = qx * d[0] + qy * d[1]; perp = np.abs(-qx * d[1] + qy * d[0])
            reg = (perp <= hw) & (along > tend - 2) & (along < tend + dend + 12) & ~(lab == hl)
            patch |= reg
            core = (wl == i) & (perp <= .8) & (dtail > 16)                      # цвет туши уса
            fcol.append(np.median(rgb[core], axis=0) if core.any() else np.array([30, 30, 30], np.float32))
            whisk[reg] = len(fcol); geo.append((c, d, hw))
        if patch.any():
            pr = ndimage.binary_dilation(patch, iterations=1)
            ys_, xs_ = np.nonzero(pr); sy = slice(ys_.min() - 12, ys_.max() + 13); sx = slice(xs_.min() - 12, xs_.max() + 13)
            # фон под усом: линейно поперёк уса, между точками по обе стороны (ус тонкий, поэтому
            # контур хвоста, пересекающий ус под углом, остаётся непрерывным, а не размывается)
            B = rgb[sy, sx].copy(); Ba = al[sy, sx].copy(); sdiff = np.full(B.shape[:2], 1e9, np.float32)
            idx0 = whisk[sy, sx].astype(int)
            gy, gx = np.mgrid[sy, sx].astype(np.float32)
            for j, (c, d, hw) in enumerate(geo, start=1):
                sel = (idx0 == j) | ((idx0 == 0) & pr[sy, sx] & (np.abs(-(gx - c[0]) * d[1] + (gy - c[1]) * d[0]) <= hw + 1.2))
                if not sel.any(): continue
                n_ = np.array([-d[1], d[0]]); qx, qy = gx[sel], gy[sel]
                sp = -(qx - c[0]) * d[1] + (qy - c[1]) * d[0]                # со знаком, поперёк
                off = hw + 1.5
                ax, ay = qx - n_[0] * (sp + off), qy - n_[1] * (sp + off)    # точки за краями уса
                bx, by = qx - n_[0] * (sp - off), qy - n_[1] * (sp - off)
                w = np.clip((sp + off) / (2 * off), 0, 1)[:, None]
                samp = lambda img, X, Y: cv2.remap(img, X.reshape(-1, 1).astype(np.float32), Y.reshape(-1, 1).astype(np.float32), cv2.INTER_LINEAR).reshape(len(X), -1)
                rgb4 = np.dstack([rgb, al]).astype(np.float32)
                va, vb = samp(rgb4, ax, ay), samp(rgb4, bx, by)
                v = va * (1 - w) + vb * w
                B[sel] = v[:, :3]; Ba[sel] = v[:, 3]
                sdiff[sel] = np.abs(va - vb)[:, :3].max(axis=1)
            # LaMa продолжает линии, входящие в дырку: закрыть от неё ус целиком, не только кончик
            hide = pr | (ndimage.binary_dilation(Hm & ndimage.binary_dilation(whisk > 0, iterations=60), iterations=2) & ~(lab == hl))
            lama = r.get("lama") and os.path.exists(r["lama"]) and _lama_fill(r["lama"], rgb, al, hide)
            if lama:                                                           # LaMa — где ус пересекает контур
                struct = sdiff > 30                                            # по сторонам уса разное — структура
                B[struct] = lama[0][sy, sx][struct]; Ba[struct] = lama[1][sy, sx][struct]
            I = rgb[sy, sx]; lum = lambda v: v @ np.float32([.299, .587, .114])
            idx = whisk[sy, sx].astype(int)
            gy, gx = np.mgrid[sy, sx].astype(np.float32)
            a0 = al[sy, sx][..., None] / 255
            tb = Ba / 255; pmB = np.dstack([B * tb[..., None], tb])                 # хвост без кончика
            pmW = np.zeros_like(pmB)                                               # кончик уса
            orig_pm = np.dstack([I * a0, a0]).astype(np.float32)
            F = np.zeros_like(I)
            for j in range(1, len(fcol) + 1): F[idx == j] = fcol[j - 1]
            sel_all = idx > 0
            # над бумагой (под усом хвоста нет) — сам ус, пиксель в пиксель
            tail_ink = ndimage.binary_dilation(lab == tl, iterations=3)[sy, sx]      # хвост с контуром
            paper_zone = sel_all & (tb < .5) & ~tail_ink
            pmW[paper_zone] = orig_pm[paper_zone]; pmB[paper_zone] = 0
            # над мехом хвоста — ус вынимается разностью с дорисованным мехом: I = a·F + (1-a)·B
            fur_zone = sel_all & ~paper_zone
            aw = np.clip((lum(B) - lum(I)) / np.maximum(lum(B) - lum(F), 20), 0, 1)
            aw[~fur_zone] = 0
            # только то, что продолжает ус от бумаги (связная тёмная нить), без отдельных крапинок
            lab_w, nw = ndimage.label((aw > .25) | paper_zone, structure=np.ones((3, 3)))
            keep_ids = set(np.unique(lab_w[paper_zone])) - {0}
            thread = np.isin(lab_w, list(keep_ids)) & fur_zone
            aw[~thread] = 0
            pmW[fur_zone] = np.dstack([F * aw[..., None], aw])[fur_zone]
            oncore = sel_all
            # проверка покоя: ус поверх хвоста обязан дать исходный пиксель. Штрихи меха вне линии
            # уса, которых модель не объясняет, остаются хвосту как есть; на самой линии — ус
            pap = self.paper
            bodyc = pmB[..., :3] + (1 - pmB[..., 3:4]) * pap
            rec = pmW[..., :3] + (1 - pmW[..., 3:4]) * bodyc
            orig = I * a0 + (1 - a0) * pap
            bad = np.abs(rec - orig).max(axis=2) > 10
            mk = pr[sy, sx] & ~Hm[sy, sx]                                         # только пиксели хвоста
            keep = mk & bad & ~(pmW[..., 3] > .05) & (tb > .5)                   # на хвосте: что не рисует ус — как было
            pmB[keep] = np.dstack([I * a0, a0])[keep]; pmW[keep] = 0
            pmW[~mk] = 0
            self.tip_stats = dict(pixels=int(mk.sum()), kept_original=int(keep.sum()),
                                  rest_err_on_line=float(np.abs(rec - orig).max(axis=2)[mk & oncore].max()) if (mk & oncore).any() else 0.0)
            self.tip_patch = dict(sy=sy, sx=sx, mask=mk, body=pmB.astype(np.float32), head=pmW.astype(np.float32))
        else:
            self.tip_patch = None
        return Hm

    # ------------------------------------------------------------- глаза
    def set_paper(self, rgb):
        """Бумага, на которую кладётся кукла (сборщик роликов — canvas.CREAM, не белая).
        Полупрозрачные края меха в исходнике уже смешаны с ЕГО бумагой; чтобы на другой
        бумаге не было светлого ореола, примесь снимается обратным смешиванием:
        цвет = (наблюдаемый − (1−a)·бумага_исходника) / a. На бумаге исходника кадр
        остаётся байт в байт прежним (прямое смешивание восстанавливает исходник)."""
        rgb = np.asarray(rgb, np.float32)
        if np.allclose(rgb, self.paper_src):
            return
        self.defringe()
        self.paper = rgb
        self.paper_u8[:] = np.clip(rgb, 0, 255).astype(np.uint8)
        self.paper_bb[:] = rgb

    def defringe(self):
        """Снять с полупрозрачных краёв меха примесь бумаги исходника (один раз). Нужно и для RGBA-кадра:
        там кота кладут на чужую бумагу тем же обратным смешиванием."""
        if not getattr(self, "_defringed", False):
            a = self.alpha[..., None] / 255
            edge = (a > 0.02) & (a < 0.98)
            for arr in (self.fur, self.base4[..., :3]):
                fixed = (arr - (1 - a) * self.paper_src) / np.maximum(a, 0.02)
                arr[edge[..., 0]] = np.clip(fixed, 0, 255)[edge[..., 0]]
            for P in self.poses.values():
                pa = P["alpha"][..., None] / 255; pe = (pa > 0.02) & (pa < 0.98)
                for arr in (P["fur"], P["base4"][..., :3]):
                    fixed = (arr - (1 - pa) * self.paper_src) / np.maximum(pa, 0.02)
                    arr[pe[..., 0]] = np.clip(fixed, 0, 255)[pe[..., 0]]
            self._defringed = True

    def face4(self, look, lid, base4):
        """То же, что face(), но сразу на 4-канальной основе (мех + прозрачность), без склейки каждый кадр."""
        img4 = base4.copy(); img = img4[..., :3]
        self._eyes(img, look, lid)
        return img4

    def face(self, look, lid, base=None):
        img = (self.fur if base is None else base).copy()
        self._eyes(img, look, lid)
        return img

    def _eyes(self, img, look, lid):
        for k, ((cx, cy, rx, ry), (mi, mw, mp), (bx0, by0, bx1, by1)) in enumerate(zip(self.E, self.M, self.eye_bb)):
            sl = (slice(by0, by1), slice(bx0, bx1))
            yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
            dx, dy = look[0] * rx * .3, look[1] * ry * .3
            pup = remap(self.pupil_src[k], xx - bx0 - dx, yy - by0 - dy)
            pa = (pup[..., 3:4] / 255) * mi[sl][..., None]
            eye = self.iris[sl] * (1 - pa) + pup[..., :3] * pa
            if self.hl[k] is not None:                                             # блик на месте, поверх зрачка и радужки
                hc, ha = self.hl[k]; eye = eye * (1 - ha) + hc
            sy = max(1 - lid, 1e-3); l0 = cy + ry * LID_ANCHOR
            e4 = np.dstack([eye, mw[sl].astype(np.float32) * 255])
            # глаз сплющивается не к прямой, а к ДУГЕ закрытого века (той же, что рисуется штрихом):
            # остаток щёлки у самого закрытия лежит точно под штрихом, а не пересекает его прямой линией
            # одна гладкая кривая на всю ширину глаза: и линия сжатия, и штрих века (без углов на концах)
            u = np.clip((xx - cx) / (rx * .98), -1, 1); bump = (1 - u * u) ** 0.55
            curve = (l0 - ry * .12) + ry * .37 * bump
            lcurve = l0 + (curve - l0) * float(smooth(lid / .6))                # при открытом глазе — прежняя прямая
            e2 = remap(e4, xx - bx0, (lcurve + (yy - lcurve) / sy) - by0)
            ea = e2[..., 3:4] / 255
            img[sl] = img[sl] * (1 - ea) + e2[..., :3] * ea
            # Штрих закрытого века раньше ВКЛЮЧАЛСЯ разом при lid > 0.85 (между соседними кадрами
            # 0.84 -> 0.86 появлялась линия в 7 px) — измеренный скачок кадра. Теперь он проявляется
            # плавно от 0.7 к 1.0, вместе с тем как сплющенный глаз сходит в линию.
            ka = float(smooth((lid - .7) / .3))
            if ka > 0:
                c = img[sl].copy()
                us = np.linspace(-.92, .92, 48); pts = np.stack([cx - bx0 + us * rx * .98,
                                                                (l0 - ry * .12) + ry * .37 * (1 - us * us) ** 0.55 - by0], 1)
                cv2.polylines(c, [np.round(pts * 16).astype(np.int32)], False, (24, 22, 24), 7, cv2.LINE_AA, shift=4)
                cover = float(smooth((lid - LID_COVER) / .15))                 # у закрытия штрих накрывает остаток щёлки под собой
                w = ka * ((1 - ea) + ea * cover)
                img[sl] = img[sl] * (1 - w) + c * w

    # ------------------------------------------------------------- кадр
    def frame(self, st, t):
        cat = self.cat_layer(st, t)
        return self._warp_cpu(cat, st)

    def cat_layer(self, st, t):
        r = self.r
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
            cat = self.face4(st["look"], st["lid"], self.base4)
        pz = st.get("pose")                          # (имя позы, доля 0..1) — смена позы за 2 кадра
        if pz and pz[1] > 0 and pz[0] in self.poses:
            P = self.poses[pz[0]]
            other = self.face4(st["look"], st["lid"], P["base4"])
            k = np.float32(pz[1]); cat = cat * (1 - k) + other * k
        fx0, fy0, fx1, fy1 = r["flame_box"]                                 # огонёк — в своих координатах
        # рамка с запасом на бумагу: в старой рамке кончик пламени (3 px от верха) при сдвиге вверх
        # упирался в её край и срезался плоско (7% кадров), а мех хвоста, пересекающий её бок,
        # сдвигался внутри и стоял снаружи — ступенька на краю. Сдвиг к краям запаса плавно гаснет.
        g = self.flame_geo
        X0b, Y0b, X1b, Y1b = g["box"]; yy, xx = g["yy"], g["xx"]
        ku = g["ku"]                                                         # = прежний (1 - y/h)^1.5, выше рамки — 1
        if FLAME_NOISE:                                                      # fBm: без периода, языки не повторяются
            dx = ku * (4 * fbm(yy / 45 - t * 2.3, 11) + 2.5 * fbm(np.float32(t * 3.7), 23))
            dy = ku * 3 * fbm(np.float32(t * 4.1), 37)
        else:
            dx = ku * (4 * np.sin(2 * np.pi * (yy / 45 - t * 2.3)) + 2.5 * np.sin(2 * np.pi * (t * 3.7 + .3)))
            dy = ku * 3 * np.sin(2 * np.pi * t * 4.1)
        dx = dx * g["w"]; dy = dy * g["w"]
        cat[Y0b:Y1b, X0b:X1b] = remap(cat[Y0b:Y1b, X0b:X1b], xx - X0b + fx0 - dx, (yy + fy0) - Y0b + dy, cv2.BORDER_REPLICATE)
        fs = st.get("flame")
        if fs in FLAME_STATES and self.flame_state is not None:
            cat[Y0b:Y1b, X0b:X1b] = self._flame_state(cat[Y0b:Y1b, X0b:X1b], fs)
        return cat

    def _flame_state(self, reg, fs):
        """Огонёк по состоянию героя: ember — вдвое ниже и притушен (серо-оранжевый), golden — выше и
        жёлтый. Масштаб — от основания пламени (кончик хвоста стоит на месте), цвет — только в маске
        пламени. bright и отсутствие состояния — пламя как нарисовано, байт в байт."""
        F = self.flame_state; sy, sx, col = FLAME_STATES[fs]
        X0b, Y0b, _, _ = self.flame_geo["box"]
        yb, xb = F["yb"], F["xb"]
        Yq = F["base"] + (yb - F["base"]) / sy; Xq = F["cx"] + (xb - F["cx"]) / sx
        src = remap(reg, Xq - X0b, Yq - Y0b, cv2.BORDER_REPLICATE)
        m = F["mask"]
        w = np.maximum(m, remap(m, Xq - X0b, Yq - Y0b, cv2.BORDER_REPLICATE))   # старый след и новое пламя
        wm = remap(F["core"], Xq - X0b, Yq - Y0b, cv2.BORDER_REPLICATE)[..., None]   # цвет — только ядру, не туши и не меху
        src = src.copy(); src[..., :3] = src[..., :3] * (1 - wm) + np.clip(src[..., :3] * np.float32(col), 0, 255) * wm
        return reg * (1 - w[..., None]) + src * w[..., None]


    def frame_hd(self, st, t):
        """Кадр сразу 1920x1080: изгиб и увеличение одним пересчётом (INTER_CUBIC) — края резче,
        чем remap + отдельный Lanczos-апскейл."""
        cat = self.cat_layer(st, t); r = self.r
        if not hasattr(self, "_hd"):
            sc = 1920 / self.W; off = (self.H * sc - 1080) / 2
            Oy, Ox = np.mgrid[0:1080, 0:1920].astype(np.float32)
            Px = (Ox + .5) / sc - .5; Py = (Oy + off + .5) / sc - .5
            x0, y0, x1, y1 = self.bb
            ix = np.clip(np.round(Px).astype(int) - x0, 0, x1 - x0 - 1); iy = np.clip(np.round(Py).astype(int) - y0, 0, y1 - y0 - 1)
            inside = (Px >= x0) & (Px < x1) & (Py >= y0) & (Py < y1)
            self._hd = dict(Px=Px, Py=Py, inside=inside, wh=self.w_head[iy, ix] * inside, we=self.w_ear[iy, ix] * inside, wt=self.w_tail[iy, ix] * inside)
        h = self._hd; X, Y = h["Px"].copy(), h["Py"].copy()
        def rot(X, Y, p, deg, w):
            if abs(deg) < 1e-3: return X, Y
            a = np.float32(-np.radians(deg)) * w; c, s = np.cos(a), np.sin(a); dx, dy = X - p[0], Y - p[1]
            return p[0] + c * dx - s * dy, p[1] + s * dx + c * dy
        b = st["breath"] * (1 - .02 * st.get("squash", 0)); fl = r["floor_y"]
        Y = fl - (fl - Y) / b
        X, Y = rot(X, Y, r["tail_base"], st["tail"], h["wt"]); X, Y = rot(X, Y, r["ear_base"], st["ear"], h["we"]); X, Y = rot(X, Y, r["neck"], st["head"], h["wh"])
        c = cv2.remap(np.ascontiguousarray(cat, np.float32), X.astype(np.float32), Y.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        a = np.clip(c[..., 3:4], 0, 255) / 255 * h["inside"][..., None]
        return np.clip(self.paper * (1 - a) + np.clip(c[..., :3], 0, 255) * a, 0, 255).astype(np.uint8)

    def _warp_core(self, cat, st):
        """Одна общая деформация на весь кадр (как раньше) и два раздельных слоя — голова и тело
        с хвостом — только в прямоугольнике, где голова встречается с хвостом (self.conflict).
        На краях прямоугольника двигается только одна из частей, поэтому общая деформация и
        слои там дают одно и то же — шва нет."""
        r = self.r; x0, y0, x1, y1 = self.bb
        def rot(X, Y, p, deg, w, sl):
            """Поворот с весом только там, где вес ненулевой (in place)."""
            if sl is None or abs(deg) < 1e-3: return
            ang = np.float32(-np.radians(deg)) * w[sl]
            c, s = np.cos(ang), np.sin(ang); dx, dy = X[sl] - p[0], Y[sl] - p[1]
            X[sl] = p[0] + c * dx - s * dy; Y[sl] = p[1] + s * dx + c * dy
        b = st["breath"] * (1 - .02 * st.get("squash", 0)); fl = r["floor_y"]
        X0 = self.XX; Y0 = fl - (fl - self.YY) / b
        XR, YR = X0.copy(), Y0.copy()                                          # тело: лапы и хвост
        for k, (lift, swing) in st["paws"].items():
            sl = self.sl["paw_" + k]; w = self.w_paw[k]
            rot(XR, YR, r["paws"][k]["shoulder"], swing, w, sl)
            if abs(lift) > 1e-3: YR[sl] += np.float32(lift) * w[sl]       # ступня вверх, плечо на месте
        rot(XR, YR, r["tail_base"], st["tail"], self.w_tail, self.sl["tail"])
        cy, cx = self.conflict
        XRc, YRc = XR[cy, cx].copy(), YR[cy, cx].copy()
        hr = self.head_rows; rmax = max(hr, cy.stop)                           # голова живёт только выше rmax
        XH, YH = X0[:rmax].copy(), Y0[:rmax].copy()                            # голова: ухо и голова
        rot(XH, YH, r["ear_base"], st["ear"], self.w_ear, self.sl["ear"])
        rot(XH, YH, r["neck"], st["head"], self.w_head, self.sl["head"])
        self._YH = YH
        XC, YC = XR, YR                                                        # общая: выше шеи — ход головы
        XC[:hr] = XH[:hr]; YC[:hr] = YH[:hr]
        c = remap(cat, XC, YC)                                                 # как раньше, один в один
        return c, self._conflict_layers(cat, XH, XRc, YRc, cy, cx)

    def _conflict_layers(self, cat, XH, XRc, YRc, cy, cx):
        """Стык головы и хвоста: два премультиплицированных слоя, сумма с приоритетом головы."""
        sy, sx = self.conflict_src                                             # окно источника (абс.)
        pm = cat[sy, sx].copy(); a = pm[..., 3] * np.float32(1 / 255); pm[..., 3] = a
        for ch in range(3): pm[..., ch] *= a
        Hin = cv2.multiply(pm, self.m_head_src); Rin = cv2.subtract(pm, Hin)
        tp = self.tip_patch
        if tp is not None:                                                     # кончики усов: хвосту — мех, голове — ус
            py_, px_ = slice(tp["sy"].start - sy.start, tp["sy"].stop - sy.start), slice(tp["sx"].start - sx.start, tp["sx"].stop - sx.start)
            mk = tp["mask"]
            Rin[py_, px_][mk] = tp["body"][mk]; Hin[py_, px_][mk] = tp["head"][mk]
        Hc = remap(Hin, XH[cy, cx] - sx.start, self._YH[cy, cx] - sy.start)
        Rc = remap(Rin, XRc - sx.start, YRc - sy.start)
        Pc = Hc + Rc
        over = Pc[..., 3] > 1
        if over.any():
            aH, aR = Hc[over][:, 3:4], Rc[over][:, 3:4]
            Pc[over] = Hc[over] + Rc[over] * np.clip((1 - aH) / np.maximum(aR, 1e-6), 0, 1)
        return Pc

    def _warp_cpu(self, cat, st):
        """Кадр на бумаге рига (как было, байт в байт)."""
        x0, y0, x1, y1 = self.bb
        c, Pc = self._warp_core(cat, st); cy, cx = self.conflict
        a = c[..., 3] * np.float32(1 / 255); a3 = cv2.merge([a, a, a])
        reg = cv2.add(cv2.multiply(cv2.subtract(cv2.cvtColor(c, cv2.COLOR_RGBA2RGB), self.paper_bb), a3), self.paper_bb)   # = бумага·(1-A) + цвет·A
        reg[cy, cx] = self.paper * (1 - Pc[..., 3:4]) + Pc[..., :3]
        out = self.paper_u8.copy()
        out[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
        return out

    def frame_rgba(self, st, t):
        """Кот без бумаги: ПРЕМУЛЬТИПЛИЦИРОВАННЫЙ RGBA float32 (H, W, 4), цвет 0..255, альфа 0..1 — для
        сборщика роликов, который кладёт его на свою бумагу. Примесь белой бумаги исходника с краёв
        снята (defringe), иначе на кремовом холсте был бы светлый ореол (замер: 48% краевых пикселей)."""
        self.defringe()
        x0, y0, x1, y1 = self.bb
        c, Pc = self._warp_core(self.cat_layer(st, t), st); cy, cx = self.conflict
        a = c[..., 3:4] * np.float32(1 / 255)
        reg = np.concatenate([c[..., :3] * a, a], axis=2)
        reg[cy, cx] = Pc                                                       # там уже премультиплицировано
        out = np.zeros((self.H, self.W, 4), np.float32)
        out[y0:y1, x0:x1] = reg
        return out

# ----------------------------------------------------------------- движения
def key(t, pts, default=0.0):
    if not pts: return default
    if t <= pts[0][0]: return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1: return v0 + (v1 - v0) * float(smooth((t - t0) / max(t1 - t0, 1e-6)))
    return pts[-1][1]

BLINK_MIN_GAP = 0.3     # между морганиями не меньше (двойное — 0.32 с, остаётся)
END_QUIET_SEC = 0.5     # за столько до конца клипа новых движений век нет


def thin_blinks(blinks, dur, keep=()):
    """Моргания не чаще BLINK_MIN_GAP и не в последние END_QUIET_SEC клипа (веко закрывается 0.08 с и
    открывается 0.17 с — должно успеть). keep — моргания, которые несут функцию (взгляд уходит и
    возвращается под веком): при тесноте уступает случайное фоновое, а не они."""
    out = [b for b in sorted(keep) if b <= dur - END_QUIET_SEC]
    for b in sorted(blinks):
        if b > dur - END_QUIET_SEC or any(abs(b - k) < BLINK_MIN_GAP for k in out):
            continue
        out.append(b)
    return sorted(out)


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
    POSE = []
    gaze_blinks = []
    for a in actions:
        t0, kind = a["t"], a["do"]
        if kind == "look":                      # смотреть на точку: x,y в [-1..1]
            d = a.get("dur", 1.5)
            if BLINK_ON_GAZE and abs(a["x"] - key(t0, look_x)) >= .6 and not any(abs(b - t0) < .5 for b in blinks):
                gaze_blinks.append(t0 - .03)    # взгляд уходит под веком: так переводят глаза люди и кошки
            if BLINK_ON_GAZE and abs(a["x"]) >= .6 and not any(abs(b - (t0 + d)) < .5 for b in blinks):
                gaze_blinks.append(t0 + d - .03)  # и возвращается тоже под веком
            look_x = [p for p in look_x if not (t0 - .3 <= p[0] <= t0 + d + .3)]
            look_y = [p for p in look_y if not (t0 - .3 <= p[0] <= t0 + d + .3)]
            look_x += [(t0, key(t0, look_x)), (t0 + .22, a["x"]), (t0 + d, a["x"]), (t0 + d + .3, 0.)]
            look_y += [(t0, key(t0, look_y)), (t0 + .22, a.get("y", 0.)), (t0 + d, a.get("y", 0.)), (t0 + d + .3, 0.)]
            look_x.sort(); look_y.sort()
        elif kind == "tilt":                    # наклон головы, градусы
            d = a.get("dur", 1.5)
            head.sort(); cur = key(t0, head)    # с ТЕКУЩЕГО угла, не с нуля: два наклона внахлёст давали рывок 3.4°/кадр
            head = [p for p in head if not (t0 <= p[0] <= t0 + 1.1 + d)]
            head += [(t0, cur), (t0 + .5, a["deg"]), (t0 + .5 + d, a["deg"]), (t0 + 1.1 + d, 0.)]
        elif kind == "tap":                     # постучать лапой n раз
            L = paws[a.get("paw", "right")][0]
            for i in range(a.get("n", 2)):
                s = t0 + i * .42; L += [(s, 0.), (s + .14, 18.), (s + .3, 0.)]
        elif kind == "point":                   # показать лапой в сторону (± градусы)
            S, L = paws[a.get("paw", "right")][1], paws[a.get("paw", "right")][0]
            d = a.get("dur", 1.4); deg = a.get("deg", -12)
            S += [(t0, 0.), (t0 + .35, deg), (t0 + .35 + d, deg), (t0 + .8 + d, 0.)]
            L += [(t0, 0.), (t0 + .35, 12.), (t0 + .35 + d, 12.), (t0 + .8 + d, 0.)]
        elif kind == "paw":                     # лапа к груди (поза), держит dur секунд
            POSE.append((t0, t0 + a.get("dur", 2.0), "paw_up"))
        elif kind == "pose":                    # любая поза из rig.json, поверх текущей
            POSE.append((t0, t0 + a.get("dur", 1.5), a["name"]))
        elif kind == "wave":                    # помахать: подъём 0.2 с, n взмахов, опускание
            n = a.get("n", 2); up = .2; per = .42
            WV.append((t0, t0 + up, t0 + up + n * per, t0 + up + n * per + .22, per))
        elif kind == "blink":
            blinks.append(t0)
    # моргания не пачками и не у самой склейки (сравнение 08.10: три смыкания век за 0.62 с перед резом,
    # интервалы 0.08 с; интервал ≥ BLINK_MIN_GAP, двойное (0.32 с) остаётся, последнее — не позже
    # END_QUIET_SEC до конца клипа: движение, срезанное склейкой, читается как сбой)
    blinks = thin_blinks(blinks, dur, keep=gaze_blinks)
    head.sort()
    def state(t):
        lid = 0.
        for b in blinks:                       # веко: быстро вниз (80 мс), пауза, медленнее вверх (170 мс),
            d = t - b                          # оба хода с плавным разгоном/торможением, а не линейно
            if 0 <= d < .08: lid = max(lid, float(smooth(d / .08)))
            elif .08 <= d < .13: lid = 1.
            elif .13 <= d < .30: lid = max(lid, float(1 - smooth((d - .13) / .17)))
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
        pose = None; squash = 0.
        best = None
        for (p0, p1, name) in POSE:
            if name == "paw_up" and p0 - .15 <= t < p0:
                squash = float(np.sin(np.pi * (t - (p0 - .15)) / .15))                  # присед-замах перед подъёмом
            if p0 <= t < p1 and (best is None or p0 >= best[0]):  # мгновенная смена позы, последняя начатая главнее
                best = (p0, name)
        if best: pose = (best[1], 1.0)
        return dict(pose=pose, squash=squash, wave=wave, look=(key(t, look_x), key(t, look_y)), lid=lid, head=key(t, head), ear=ear,
                    tail=float(4.3 * np.tanh(fbm(np.float32(t / 3.1), 53))) if TAIL_NOISE else 4 * np.sin(2 * np.pi * t / 3.1),
                    breath=1 + .012 * np.sin(2 * np.pi * t / 2.6),
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

def render(out, dur, actions, fps=24, workers=4):   # 24 — как у сборщика роликов (assemble_frames.FPS); план в секундах, от fps не зависит
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
