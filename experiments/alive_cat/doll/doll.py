"""Кукла Чирка из эталона: взгляд (зрачки), моргание (сжатие глаза к веку), наклон головы, ухо, хвост, огонёк, дыхание."""
import subprocess, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
src = np.asarray(Image.open("hero.png").convert("RGB")).astype(np.float32); H, W = src.shape[:2]
alpha = np.asarray(Image.open("mask.png")).astype(np.float32)
fur = np.asarray(Image.open("part_fur.png")).astype(np.float32)
iris = np.asarray(Image.open("part_iris.png")).astype(np.float32)
E = np.load("eyes.npy"); M = np.load("masks.npy")          # [eye][iris, whole, pupil]
m = alpha > 128
paper = np.median(src[~m], axis=0)
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
def smooth(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
def sample(img, X, Y): return np.stack([ndimage.map_coordinates(img[..., c], [Y, X], order=1, mode="nearest") for c in range(img.shape[2])], -1)
def rot(X, Y, p, ang): c, s = np.cos(ang), np.sin(ang); dx, dy = X - p[0], Y - p[1]; return p[0] + c * dx - s * dy, p[1] + s * dx + c * dy

NECK = (618., 575.); EAR_B, EAR_T = (742., 222.), (774., 111.); TAIL_B = (822., 705.)
FB = (920, 270, 1050, 480)                                   # огонёк
w_head = smooth((585 - YY) / 55) * (XX < 880)
ex, ey = EAR_T[0] - EAR_B[0], EAR_T[1] - EAR_B[1]
w_ear = smooth((((XX - EAR_B[0]) * ex + (YY - EAR_B[1]) * ey) / (ex * ex + ey * ey)) * 1.3) * smooth((100 - np.hypot(XX - 758, YY - 165)) / 25)
w_tail = smooth((XX - 830) / 140) * (YY > 260)

# ------------------------------------------------ сценарий (секунды)
FPS, DUR = 30, 8.0
def key(t, pts):   # кусочно-гладкая интерполяция по ключам [(t, value)]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1: return v0 + (v1 - v0) * smooth((t - t0) / (t1 - t0))
    return pts[-1][1] if t > pts[-1][0] else pts[0][1]
LOOK_X = [(0, 0), (1.0, 0), (1.35, -1), (2.6, -1), (2.95, 1), (4.6, 1), (4.95, 0), (8, 0)]
LOOK_Y = [(0, 0), (2.6, 0), (2.95, .25), (4.6, .25), (4.95, 0), (8, 0)]
HEAD = [(0, 0), (2.7, 0), (3.3, 5), (4.6, 5), (5.2, -2), (6.4, -2), (7.0, 0), (8, 0)]   # градусы, + к письму
BLINKS = [2.05, 5.3, 7.4]
def lid(t):
    for b in BLINKS:
        d = t - b
        if 0 <= d < .08: return d / .08
        if .08 <= d < .14: return 1.
        if .14 <= d < .26: return 1 - (d - .14) / .12
    return 0.
def ear(t):
    a = 0
    for t0 in (3.8, 4.05):
        d = (t - t0) / .11
        if 0 <= d < 2: a += -10 * np.sin(np.pi * d / 2) ** 2
    return np.radians(a)

def eyes_layer(t):
    """Морда без глаз + глаза: радужка (сжимается к веку при моргании) + зрачок со сдвигом взгляда."""
    img = fur.copy(); f = lid(t); lx, ly = key(t, LOOK_X), key(t, LOOK_Y)
    for (cx, cy, rx, ry), (mi, mw, mp) in zip(E, M):
        eye = iris.copy()
        # зрачок-спрайт: исходные пиксели зрачка, сдвинутые взглядом; внутри радужки
        dx, dy = lx * rx * .3, ly * ry * .3
        pup = sample(np.dstack([src, mp.astype(np.float32) * 255]), XX - dx, YY - dy)
        pa = (pup[..., 3:4] / 255) * mi[..., None]
        eye = eye * (1 - pa) + pup[..., :3] * pa
        # моргание: глаз сжимается по вертикали к линии века (62% высоты)
        sy = max(1 - f, 0.001); ly0 = cy + ry * .24
        Ys = ly0 + (YY - ly0) / sy
        e2 = sample(np.dstack([eye, mw.astype(np.float32) * 255]), XX, Ys)
        ea = e2[..., 3:4] / 255
        img = img * (1 - ea) + e2[..., :3] * ea
        if f > .85:   # закрытый глаз: дуга ресницы, как тушью
            pil = Image.fromarray(img.astype(np.uint8)); d = ImageDraw.Draw(pil)
            d.arc([cx - rx * .85, ly0 - ry * .55, cx + rx * .85, ly0 + ry * .25], 15, 165, fill=(24, 22, 24), width=7)
            img = np.asarray(pil).astype(np.float32)
    return img

def frame(t):
    cat = np.dstack([eyes_layer(t), alpha])
    # огонёк (до движения хвоста — колышется в своих координатах)
    x0, y0, x1, y1 = FB; reg = cat[y0:y1, x0:x1]; hh = y1 - y0
    yy, xx = np.mgrid[0:hh, 0:x1 - x0].astype(np.float32); ku = (1 - yy / hh) ** 1.5
    dx = ku * (4 * np.sin(2 * np.pi * (yy / 45 - t * 2.3)) + 2.5 * np.sin(2 * np.pi * (t * 3.7 + .3)))
    cat[y0:y1, x0:x1] = sample(reg, xx - dx, yy + ku * 3 * np.sin(2 * np.pi * t * 4.1))
    X, Y = XX, YY
    X, Y = rot(X, Y, NECK, -np.radians(key(t, HEAD)) * w_head)
    X, Y = rot(X, Y, EAR_B, -ear(t) * w_ear)
    X, Y = rot(X, Y, TAIL_B, -np.radians(4 * np.sin(2 * np.pi * t / 3.1)) * w_tail)
    b = 1 + .012 * np.sin(2 * np.pi * t / 2.6)                     # дыхание от лап
    c = sample(cat, X, 770 - (770 - Y) / b)
    a = c[..., 3:4] / 255
    return np.clip(paper * (1 - a) + c[..., :3] * a, 0, 255).astype(np.uint8)

p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-vf", "scale=1920:-2:flags=lanczos,crop=1920:1080", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "doll.mp4"], stdin=subprocess.PIPE)
for i in range(int(FPS * DUR)): p.stdin.write(frame(i / FPS).tobytes())
p.stdin.close(); p.wait(); print("ok")
