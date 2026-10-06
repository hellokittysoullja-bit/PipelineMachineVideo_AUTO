"""Живой кот кодом, CPU: параллакс по карте глубины + покачивание головы + дёрганье уха + огонёк + дыхание."""
import subprocess, numpy as np, onnxruntime as ort
from PIL import Image
from scipy import ndimage
C = "../cutout"; k = "10"
src = Image.open(f"{C}/src/{k}.png").convert("RGB"); W, H = src.size
alpha = np.asarray(Image.open(f"{C}/birefnet-general/{k}_mask.png")).astype(np.float32)
bg = np.asarray(Image.open(f"{C}/lama/{k}_clean.png").convert("RGB")).astype(np.float32)
cat = np.dstack([np.asarray(src).astype(np.float32), alpha])
m = alpha > 128

def depth(img):  # Depth-Anything-V2-Small (Apache-2.0), CPU ~1 c
    s = ort.InferenceSession("../depth/model.onnx")
    w, h = 518 * W // H // 14 * 14, 518
    x = np.asarray(Image.fromarray(img.astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32) / 255
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32)
    d = s.run(None, {"pixel_values": x.transpose(2, 0, 1)[None]})[0][0]
    d = (d - d.min()) / (d.max() - d.min())
    return np.asarray(Image.fromarray((d * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
dbg = ndimage.gaussian_filter(depth(bg), 6)               # глубина чистого фона
dcat = float(np.median(depth(np.asarray(src))[m]))        # глубина кота — точка фокуса

# огонёк
hsv = np.asarray(src.convert("HSV")).astype(int); h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
fl = m & ((h_ < 30) | (h_ > 245)) & (s_ > 120) & (v_ > 150)
lab, n = ndimage.label(fl); fl = lab == np.argmax(ndimage.sum(fl, lab, range(1, n + 1))) + 1
ys, xs = np.nonzero(fl); FB = (xs.min() - 12, ys.min() - 25, xs.max() + 12, ys.max() + 4)
feet = np.nonzero(m)[0].max()

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
NECK = (348.0, 480.0); EAR_TIP = (448.0, 197.0); EAR_BASE = (420.0, 268.0)
def smooth(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
w_head = smooth((500 - YY) / 60)                                  # голова и выше шеи — 1, тело — 0
ex, ey = EAR_TIP[0] - EAR_BASE[0], EAR_TIP[1] - EAR_BASE[1]; L2 = ex * ex + ey * ey
proj = ((XX - EAR_BASE[0]) * ex + (YY - EAR_BASE[1]) * ey) / L2   # 0 у основания уха, 1 на кончике
w_ear = smooth(proj * 1.4) * smooth((95 - np.hypot(XX - (EAR_BASE[0] + ex / 2), YY - (EAR_BASE[1] + ey / 2))) / 25)

def rot_map(X, Y, piv, ang):
    c, s = np.cos(ang), np.sin(ang); dx, dy = X - piv[0], Y - piv[1]
    return piv[0] + c * dx - s * dy, piv[1] + s * dx + c * dy

def sample(img, X, Y):
    return np.stack([ndimage.map_coordinates(img[..., ch], [Y, X], order=1, mode="nearest") for ch in range(img.shape[2])], -1)

FPS, DUR = 30, 6.0
def head_angle(t):   # медленный наклон к письму + лёгкое покачивание
    lean = 3.0 * smooth((t - 1.0) / 1.2) - 3.0 * smooth((t - 4.2) / 1.0)
    return np.radians(lean + 0.8 * np.sin(2 * np.pi * t / 2.9))
def ear_angle(t):    # двойное подёргивание уха на 2.6 с
    a = 0
    for t0 in (2.6, 2.85):
        d = (t - t0) / 0.12
        if 0 <= d < 2: a += -9 * np.sin(np.pi * d / 2) ** 2
    return np.radians(a)

def frame(t):
    # 1) кот: голова и ухо — обратное отображение (поворот, ослабленный весом)
    X, Y = XX, YY
    X, Y = rot_map(X, Y, NECK, -head_angle(t) * w_head)
    X, Y = rot_map(X, Y, EAR_BASE, -ear_angle(t) * w_ear)
    c = sample(cat, X, Y)
    # 2) огонёк (после поворота хвост не двигается — головы он не касается)
    x0, y0, x1, y1 = FB; reg = c[y0:y1, x0:x1]; hh = y1 - y0
    yy, xx = np.mgrid[0:hh, 0:x1 - x0].astype(np.float32); ku = (1 - yy / hh) ** 1.5
    dx = ku * (3.5 * np.sin(2 * np.pi * (yy / 38 - t * 2.3)) + 2.0 * np.sin(2 * np.pi * (t * 3.7 + .3)))
    dy = ku * 2.5 * np.sin(2 * np.pi * t * 4.1)
    c[y0:y1, x0:x1] = sample(reg, xx - dx, yy + dy)
    # 3) дыхание: растяжение по вертикали от лап
    b = 1 + 0.012 * np.sin(2 * np.pi * t / 2.6)
    c = sample(c, XX, feet - (feet - YY) / b)
    # 4) камера: облёт вокруг кота (кот — точка фокуса), фон смещается по глубине; плюс медленный наезд
    cam = 1.0 * np.sin(2 * np.pi * (t / DUR) - np.pi / 2) * 0.5 + 0.5      # 0..1..0
    cam = (cam - 0.5) * 2                                                   # -1..1
    shift = cam * 46 * (dbg - dcat)                                         # дальние и ближние едут в разные стороны
    b2 = sample(bg, XX - shift, YY)
    a = c[..., 3:4] / 255
    out = b2 * (1 - a) + c[..., :3] * a
    z = 1 + 0.05 * smooth(t / DUR)
    zx, zy = W / 2 + (XX - W / 2) / z, H * 0.55 + (YY - H * 0.55) / z
    return np.clip(sample(out, zx, zy), 0, 255).astype(np.uint8)

p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-vf", "scale=1920:-2:flags=lanczos,crop=1920:1080", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "alive_10.mp4"], stdin=subprocess.PIPE)
for i in range(int(FPS * DUR)): p.stdin.write(frame(i / FPS).tobytes())
p.stdin.close(); p.wait(); print("ok, cat depth", round(dcat, 3))
