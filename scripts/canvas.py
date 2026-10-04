#!/usr/bin/env python3
"""Холст кадра: кремовая бумага, 16:9 без размытых полей, фактура бумаги.

Решения владельца 04.10:
  * фон — кремовый #FAF7F0, а не белый: модель рисует на «белой» бумаге,
    которая на деле уже тёплая (замер: 248-254), поэтому кадр ДЕЛИТСЯ на свой
    цвет бумаги и умножается на крем — тушь остаётся тушью, бумага у всех
    кадров одного цвета;
  * в 16:9 рисунок на бумаге дополняется той же бумагой с растушёванным швом
    (размытая копия кадра по бокам читалась как «вставили картинку»);
    рисунок без бумаги по краям (нарисовано место целиком) — обрезается до
    16:9 по центру масс рисунка, а не обрамляется кремом;
  * лёгкая фактура бумаги «приклеена» к миру кадра, а не к экрану: при
    движении камеры фактура едет вместе с рисунком (на экране она стояла бы,
    как грязь на объективе). Поэтому она впекается в увеличенный холст один
    раз (bake_texture), а не накладывается на каждый кадр (замер: 98 мс на
    кадр 1080p против нуля)."""
import numpy as np
from PIL import Image, ImageFilter

CREAM = np.array([250, 247, 240], np.float32)   # #FAF7F0
ASPECT = 16 / 9
PAPER_TOL = 18          # отличие от цвета бумаги (макс. по каналу), до которого пиксель — бумага
PAPER_MIN_SHARE = 0.15  # меньше бумаги по краю — кадр не на бумаге, цвет не трогаем
PAD_MIN_SHARE = 0.6     # столько бумаги в полосе у шва — дополняем бумагой, иначе режем до 16:9
FEATHER = 0.05          # ширина растушёвки шва, доля высоты кадра
TEXTURE = 0.018         # сила фактуры (на светлом; на туши почти нет)
TEX_W, TEX_H = 4608, 2592


def border(a, k=0.03):
    h, w = a.shape[:2]
    m = max(2, int(round(k*min(h, w))))
    return np.concatenate([a[:m].reshape(-1, 3), a[-m:].reshape(-1, 3),
                           a[:, :m].reshape(-1, 3), a[:, -m:].reshape(-1, 3)]).astype(np.float32)


def paper_color(a):
    """(цвет бумаги, доля бумаги по краю). Цвет — медиана светлой половины
    краевых пикселей; доля — сколько краевых пикселей с ним совпадает."""
    b = border(a)
    lum = b.mean(1)
    col = np.median(b[lum >= np.percentile(lum, 50)], axis=0)
    share = float((np.abs(b - col).max(1) <= PAPER_TOL).mean())
    return col, share


PAPER_BLOCK = 48        # шаг карты освещённости бумаги, px исходника


def paper_map(a, col):
    """Плавная карта цвета бумаги по кадру. Модель кладёт бумагу неровно (замер
    04.10 на схеме: середина кадра темнее краёв на 6-9 единиц) — после деления
    на ОДИН цвет у рисунка проступал прямоугольник, а шов с полем 16:9 читался
    рамкой. Карта строится только по пикселям бумаги (близким к её цвету), где
    их нет — продолжается от соседей; рисунок в неё не попадает."""
    from scipy import ndimage
    h, w = a.shape[:2]
    gy, gx = (h + PAPER_BLOCK - 1)//PAPER_BLOCK, (w + PAPER_BLOCK - 1)//PAPER_BLOCK
    near = np.abs(a - col).max(2) <= PAPER_TOL + 6
    grid = np.full((gy, gx, 3), np.nan, np.float32)
    for i in range(gy):
        for j in range(gx):
            sl = (slice(i*PAPER_BLOCK, (i + 1)*PAPER_BLOCK), slice(j*PAPER_BLOCK, (j + 1)*PAPER_BLOCK))
            m = near[sl]
            if m.mean() >= 0.2:
                grid[i, j] = np.median(a[sl][m], axis=0)
    miss = np.isnan(grid[..., 0])
    if miss.all():
        return np.broadcast_to(col, a.shape).astype(np.float32)
    if miss.any():
        idx = ndimage.distance_transform_edt(miss, return_distances=False, return_indices=True)
        grid = grid[idx[0], idx[1]]
    grid = ndimage.uniform_filter(grid, size=(3, 3, 1), mode="nearest")
    out = np.stack([np.asarray(Image.fromarray(grid[..., c]).resize((w, h), Image.BICUBIC), np.float32)
                    for c in range(3)], axis=2)
    return ndimage.gaussian_filter(out, (PAPER_BLOCK/2, PAPER_BLOCK/2, 0))


def retint(a, col):
    """Бумага кадра -> ровный крем (по карте бумаги, а не по одному цвету);
    рисунок масштабируется тем же местным множителем — тушь остаётся тушью."""
    a = a.astype(np.float32)
    return np.clip(a/np.maximum(paper_map(a, col), 1.0)*CREAM, 0, 255)


def _ramp(n, fe):
    x = np.arange(n, dtype=np.float32)
    return (np.clip(x/fe, 0, 1)*np.clip((n - 1 - x)/fe, 0, 1))**1.5


def to_169(a, paper=None):
    """(холст 16:9 float32, (dx, dy) — сдвиг координат исходника на холсте).
    paper=None — бумаги нет, кадр режется до 16:9."""
    h, w = a.shape[:2]
    a = a.astype(np.float32)
    if abs(w/h - ASPECT) < 1e-3:
        return a, (0, 0)
    tall = w/h < ASPECT
    if paper is not None:
        W2, H2 = (int(round(h*ASPECT)), h) if tall else (w, int(round(w/ASPECT)))
        side = np.r_[a[:, :12].reshape(-1, 3), a[:, -12:].reshape(-1, 3)] if tall else \
            np.r_[a[:12].reshape(-1, 3), a[-12:].reshape(-1, 3)]
        if (np.abs(side - paper).max(1) <= PAPER_TOL).mean() >= PAD_MIN_SHARE:
            out = np.empty((H2, W2, 3), np.float32)
            out[:] = paper
            dx, dy = (W2 - w)//2, (H2 - h)//2
            fe = FEATHER*min(h, w)
            m = (_ramp(w, fe)[None, :] if tall else _ramp(h, fe)[:, None])[..., None]
            out[dy:dy + h, dx:dx + w] = a*m + paper*(1 - m)
            return out, (dx, dy)
    # режем: окно 16:9 по центру масс рисунка (тёмного), а не по центру кадра
    ink = 255 - a.mean(2)
    if tall:
        prof = ink.sum(1); c = float((prof*np.arange(h)).sum()/(prof.sum() + 1e-9))
        hh = int(round(w/ASPECT))
        y0 = int(np.clip(round(c - hh/2), 0, h - hh))
        return a[y0:y0 + hh], (0, -y0)
    prof = ink.sum(0); c = float((prof*np.arange(w)).sum()/(prof.sum() + 1e-9))
    ww = int(round(h*ASPECT))
    x0 = int(np.clip(round(c - ww/2), 0, w - ww))
    return a[:, x0:x0 + ww], (-x0, 0)


def prepare(img):
    """Картинка генератора -> (холст 16:9 uint8, сдвиг (dx, dy), на_бумаге)."""
    a = np.asarray(img.convert("RGB"), np.float32)
    col, share = paper_color(a)
    if share >= PAPER_MIN_SHARE:
        a = retint(a, col)
        canvas, off = to_169(a, CREAM)
        on_paper = True
    else:
        canvas, off = to_169(a, None)
        on_paper = False
    return np.clip(canvas, 0, 255).round().astype(np.uint8), off, on_paper


def paper_texture(seed=7, w=TEX_W, h=TEX_H):
    """Фактура ~1.0 ± несколько %: мягкие пятна плотности + зерно + горизонтальные
    волокна. Одна на ролик, uint8 (12 МБ), к кадру прикладывается по окну камеры."""
    rng = np.random.default_rng(seed)

    def noise(scale, blur):
        n = rng.normal(0, 1, (h//scale + 2, w//scale + 2)).astype(np.float32)
        im = Image.fromarray(((n - n.min())/(np.ptp(n) + 1e-9)*255).astype(np.uint8))
        im = im.resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(blur))
        x = np.asarray(im, np.float32)/255
        return (x - x.mean())/(x.std() + 1e-9)
    blot = noise(64, 20)
    grain = noise(1, 0.6)
    fib = Image.fromarray(((grain - grain.min())/np.ptp(grain)*255).astype(np.uint8))
    fib = fib.resize((w//3, h), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    fib = np.asarray(fib, np.float32)/255
    fib = (fib - fib.mean())/(fib.std() + 1e-9)
    t = 0.12*blot + 0.45*grain + 0.55*fib
    t /= t.std()
    return Image.fromarray(np.clip((t + 4)/8*255, 0, 255).astype(np.uint8))


def bake_texture(world, tex, seed=0):
    """Впечь фактуру в холст (uint8 HxWx3, любой масштаб). Сильнее на светлом,
    на туши почти нет. seed сдвигает участок фактуры: у соседних кадров разная
    бумага, а не одна и та же «грязь» на одном месте."""
    h, w = world.shape[:2]
    rng = np.random.default_rng(seed)
    sw = 0.8*min(tex.width, tex.height*w/h)          # участок 80%: есть куда сдвигать
    x0 = rng.uniform(0, tex.width - sw); y0 = rng.uniform(0, tex.height - sw*h/w)
    tt = np.asarray(tex.resize((w, h), Image.BILINEAR, box=(x0, y0, x0 + sw, y0 + sw*h/w)), np.float32)/255*8 - 4
    f = world.astype(np.float32)
    lum = f.mean(2, keepdims=True)/255
    return np.clip(f*(1 + TEXTURE*tt[..., None]*(0.2 + 0.8*lum**2)), 0, 255).round().astype(np.uint8)
