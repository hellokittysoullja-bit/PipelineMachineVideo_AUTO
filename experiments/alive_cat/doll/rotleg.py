"""Лапа того же размера поворачивается от груди: вырезаем настоящую лапу эталона, поворачиваем как жёсткую часть."""
import numpy as np, cv2, onnxruntime as ort, sys
from PIL import Image, ImageDraw
from scipy import ndimage
H0 = "/home/user/DoodleExplainer_AUTO/look/hero.png"
src = Image.open(H0).convert("RGB"); W, H = src.size; A = np.asarray(src).astype(np.float32)
alpha = np.asarray(Image.open("../doll/mask.png")).astype(np.float32)
paper = np.median(A[:60], axis=(0, 1))
LEG = [(528, 640), (600, 648), (623, 680), (623, 745), (617, 771), (560, 773), (547, 752), (551, 736), (537, 700), (527, 662)]
PIVOT = (576.0, 650.0)
leg = Image.new("L", (W, H), 0); ImageDraw.Draw(leg).polygon(LEG, fill=255); leg = np.asarray(leg) > 0
# 1) основа без лапы: мех LaMa + контур низа живота тушью, ниже — бумага
box = (420, 518, 750, 848); sess = ort.InferenceSession("../cutout/lama_fp32.onnx")
ci = np.asarray(src.crop(box).resize((512, 512), Image.LANCZOS)).astype(np.float32) / 255
R = ndimage.binary_dilation(leg, iterations=4)
cm = np.asarray(Image.fromarray((R * 255).astype(np.uint8)).crop(box).resize((512, 512), Image.NEAREST)) > 127
ci = (ci * (1 - cm[..., None])).astype(np.float32)
o = sess.run(None, {"image": ci.transpose(2, 0, 1)[None], "mask": cm.astype(np.float32)[None, None]})[0][0].transpose(1, 2, 0)
patch = np.asarray(Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS)).astype(np.float32)
base = A.copy(); sub = R[box[1]:box[3], box[0]:box[2]]; base[box[1]:box[3], box[0]:box[2]][sub] = patch[sub]
hsv = np.asarray(Image.fromarray(base.clip(0, 255).astype(np.uint8)).convert("HSV")).astype(int)
warm = R & (hsv[..., 1] > 45) & ((hsv[..., 0] < 35) | (hsv[..., 0] > 240))
if warm.any():
    fill = np.median(base[R & ~warm], axis=0); base[ndimage.binary_dilation(warm, iterations=3) & R] = fill
P0, P1, P2 = np.array([556., 752.]), np.array([583., 714.]), np.array([624., 745.])
ts = np.linspace(0, 1, 200)[:, None]; rng = np.random.default_rng(5)
curve = (1 - ts) ** 2 * P0 + 2 * (1 - ts) * ts * P1 + ts ** 2 * P2
curve[:, 1] += ndimage.gaussian_filter1d(rng.normal(0, 1.3, 200), 6) * np.sin(np.pi * ts[:, 0])
YY, XX = np.mgrid[0:H, 0:W]
yc = np.interp(XX, curve[:, 0], curve[:, 1], left=1e9, right=1e9)
below = R & (YY > yc)
base[below] = paper
base_a = alpha.copy(); base_a[below] = 0
c = base.clip(0, 255).astype(np.uint8).copy()
for i in range(len(curve) - 1):
    w = max(2, int(round(3.2 + 2.0 * np.sin(np.pi * i / len(curve)) + .8 * np.sin(i / 7.0))))
    cv2.line(c, tuple(np.round(curve[i]).astype(int)), tuple(np.round(curve[i + 1]).astype(int)), (26, 23, 25), w, cv2.LINE_AA)
base = c.astype(np.float32)
Image.fromarray(c).save("base_noleg.png"); Image.fromarray(base_a.astype(np.uint8)).save("base_noleg_a.png")
# 2) слой лапы: настоящие пиксели эталона, верх растушёван в мех груди
la = ndimage.gaussian_filter(leg.astype(np.float32), 1.2)
la *= np.clip((YY - 640) / 22.0, 0, 1)                       # плавный вход в грудь
np.save("leg_rgba.npy", np.dstack([A, la * alpha]).astype(np.float32))
def pose(deg):
    M = cv2.getRotationMatrix2D(PIVOT, deg, 1.0)           # + = против часовой (ступня влево)
    L = cv2.warpAffine(np.load("leg_rgba.npy"), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = L[..., 3:4] / 255
    rgb = base * (1 - a) + L[..., :3] * a
    al = np.maximum(base_a, L[..., 3])
    return rgb, al
if __name__ == "__main__":
    for deg in (0, 38):
        rgb, al = pose(deg)
        a = al[..., None] / 255; out = paper * (1 - a) + rgb * a
        Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(f"rot_{deg}.png")
    h = Image.open(H0).convert("RGB").crop((420, 600, 720, 800))
    s = Image.new("RGB", (900, 200)); s.paste(h, (0, 0)); s.paste(Image.open("rot_0.png").crop((420, 600, 720, 800)), (300, 0)); s.paste(Image.open("rot_38.png").crop((420, 600, 720, 800)), (600, 0))
    s.resize((1800, 400)).save("rot_check.jpg")
