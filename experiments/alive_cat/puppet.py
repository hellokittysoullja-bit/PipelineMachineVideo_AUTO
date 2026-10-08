"""Оживление вырезанного кота кодом: моргание, огонёк, дыхание. CPU, без моделей."""
import sys
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
k = sys.argv[1]; C = "../cutout"
src = Image.open(f"{C}/src/{k}.png").convert("RGB")
alpha = Image.open(f"{C}/birefnet-general/{k}_mask.png")
bg = Image.open(f"{C}/lama/{k}_clean.png").convert("RGB")
W, H = src.size
cat = src.copy(); cat.putalpha(alpha)
a = np.asarray(src).astype(int); m = np.asarray(alpha) > 128
hsv = np.asarray(src.convert("HSV")).astype(int); h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]

# глаза: жёлто-зелёные пятна + зрачок внутри (заполнение дыр) + контур
eye = m & (h_ > 35) & (h_ < 70) & (s_ > 90) & (v_ > 120)
lab, n = ndimage.label(eye); sz = ndimage.sum(eye, lab, range(1, n + 1))
eyes = []
for i in np.argsort(sz)[::-1][:2]:
    if sz[i] < 80: continue
    from scipy.spatial import ConvexHull
    pts = np.argwhere(lab == i + 1)[:, ::-1]
    hull = pts[ConvexHull(pts).vertices]
    hi = Image.new("L", (W, H), 0); ImageDraw.Draw(hi).polygon([tuple(p) for p in hull], fill=255)
    # зрачок может выходить за радужку: добираем тёмные пиксели рядом и снова берём оболочку
    r = np.asarray(hi) > 0
    near = ndimage.binary_dilation(r, iterations=18) & m & (v_ < 60)
    pts = np.argwhere(r | (near & ndimage.binary_dilation(r, iterations=18)))[:, ::-1]
    lab2, _ = ndimage.label(r | near); keep = lab2 == lab2[tuple(np.argwhere(r)[0])]
    pts = np.argwhere(keep)[:, ::-1]; hull = pts[ConvexHull(pts).vertices]
    hi = Image.new("L", (W, H), 0); ImageDraw.Draw(hi).polygon([tuple(p) for p in hull], fill=255)
    r = ndimage.binary_dilation(np.asarray(hi) > 0, iterations=6) & m
    ys, xs = np.nonzero(r); eyes.append((r, ys.min(), ys.max(), xs.min(), xs.max()))
# цвет меха вокруг глаз
ring = np.zeros_like(m)
for r, *_ in eyes: ring |= ndimage.binary_dilation(r, iterations=10) & ~r
fur = np.median(a[ring & m], axis=0).astype(int) if ring.any() else np.array([50, 48, 50])

# закрытые глаза: мех со лба этого же кота поверх глаза + тонкая ресница, как тушью
closed = np.asarray(cat).copy()
img = Image.fromarray(closed)
for r, ey0, ey1, ex0, ex1 in eyes:
    h = ey1 - ey0 + 1; w = ex1 - ex0 + 1
    # донор: мех над глазом (лоб), на высоту глаза выше; если не влезает — под глазом (щека)
    dy = -int(h * 1.15) if ey0 - int(h * 1.15) > 0 else int(h * 1.1)
    donor = Image.fromarray(np.asarray(cat)[ey0 + dy:ey1 + dy + 1, ex0:ex1 + 1].copy())
    msk = Image.fromarray((r[ey0:ey1 + 1, ex0:ex1 + 1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.5))
    img.paste(donor, (ex0, ey0), msk)
# ресница: несколько тонких штрихов разной толщины -> «живая» линия туши
d = ImageDraw.Draw(img)
for r, ey0, ey1, ex0, ex1 in eyes:
    hw = ex1 - ex0; cyl = ey0 + (ey1 - ey0) * 0.58
    bb = [ex0 + hw * 0.14, cyl - hw * 0.22, ex1 - hw * 0.14, cyl + hw * 0.16]
    d.arc(bb, 22, 158, fill=(28, 24, 26, 255), width=5)
    d.arc([bb[0] + 2, bb[1] + 1, bb[2] - 2, bb[3] + 1], 35, 145, fill=(28, 24, 26, 255), width=3)
closed = np.asarray(img.filter(ImageFilter.GaussianBlur(0.5))).copy()
closed[..., 3] = np.asarray(cat)[..., 3]

# огонёк
fl = m & ((h_ < 30) | (h_ > 245)) & (s_ > 120) & (v_ > 150)
lab, n = ndimage.label(fl); sz = ndimage.sum(fl, lab, range(1, n + 1))
fl = lab == np.argmax(sz) + 1 if n else None
if fl is not None:
    ys, xs = np.nonzero(fl); fb = [xs.min() - 12, ys.min() - 25, xs.max() + 12, ys.max() + 4]
feet = np.nonzero(m)[0].max()

FPS, DUR = 30, 5.0
BLINKS = [1.6, 3.9]          # моменты моргания, с
def lid(t):                  # 0 открыт .. 1 закрыт
    for b in BLINKS:
        d = t - b
        if 0 <= d < 0.07: return d / 0.07
        if 0.07 <= d < 0.12: return 1
        if 0.12 <= d < 0.24: return 1 - (d - 0.12) / 0.12
    return 0

def frame(t):
    c = np.asarray(cat).copy()
    f = lid(t)
    if f > 0:
        yy = np.arange(H)[:, None]
        for r, y0, y1, x0, x1 in eyes:
            cut = y0 + f * (y1 - y0 + 2)
            sel = r & (yy < cut)
            c[sel] = closed[sel]
            if f < 0.999:   # мягкий край века
                img = Image.fromarray(c); d = ImageDraw.Draw(img)
                d.line([x0 + 6, int(cut), x1 - 6, int(cut)], fill=(30, 26, 28, 255), width=3)
                c = np.asarray(img).copy()
    if fl is not None:   # колыхание пламени: смещение растёт к верху
        x0, y0, x1, y1 = fb; reg = c[y0:y1, x0:x1].astype(float)
        hh = y1 - y0; yy, xx = np.mgrid[0:hh, 0:x1 - x0]
        k_up = (1 - yy / hh) ** 1.5
        dx = k_up * (3.5 * np.sin(2 * np.pi * (yy / 38 - t * 2.3)) + 2.0 * np.sin(2 * np.pi * (t * 3.7 + 0.3)))
        dy = k_up * 2.5 * np.sin(2 * np.pi * t * 4.1)
        out = np.stack([ndimage.map_coordinates(reg[..., ch], [yy + dy, xx - dx], order=1, mode='nearest') for ch in range(4)], -1)
        c[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    layer = Image.fromarray(c)
    # дыхание: растяжение по вертикали от линии лап
    b = 1 + 0.012 * np.sin(2 * np.pi * t / 2.6)
    nh = int(H * b); layer = layer.resize((W, nh), Image.BICUBIC)
    out = bg.copy(); out.paste(layer, (0, feet - int(feet * b)), layer)
    return out

p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-vf", "scale=1920:-2:flags=lanczos,crop=1920:1080", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", f"puppet_{k}.mp4"], stdin=subprocess.PIPE)
for i in range(int(FPS * DUR)):
    p.stdin.write(np.asarray(frame(i / FPS)).astype(np.uint8).tobytes())
p.stdin.close(); p.wait(); print(k, "eyes", len(eyes), "flame", fl is not None, "fur", fur)
