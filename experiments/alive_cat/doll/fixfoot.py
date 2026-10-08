"""Убрать лишнюю переднюю ступню внизу (лапа теперь поднята): мех — LaMa, контур живота — штрих туши кодом."""
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage
im = Image.open("raised.png").convert("RGB"); W, H = im.size; a = np.asarray(im).astype(np.float32)
paper = np.median(a[:60, :], axis=(0, 1))
R = Image.new("L", (W, H), 0); ImageDraw.Draw(R).polygon([(547, 668), (619, 668), (619, 776), (547, 776)], fill=255)
R = np.asarray(R) > 0
# 1) мех на месте лапы
box = (420, 518, 750, 848)
sess = ort.InferenceSession("../cutout/lama_fp32.onnx")
ci = np.asarray(im.crop(box).resize((512, 512), Image.LANCZOS)).astype(np.float32) / 255
cm = np.asarray(Image.fromarray((R * 255).astype(np.uint8)).crop(box).resize((512, 512), Image.NEAREST)) > 127
ci = (ci * (1 - cm[..., None])).astype(np.float32)
o = sess.run(None, {"image": ci.transpose(2, 0, 1)[None], "mask": cm.astype(np.float32)[None, None]})[0][0].transpose(1, 2, 0)
patch = np.asarray(Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).resize((330, 330), Image.LANCZOS)).astype(np.float32)
out = a.copy(); sub = R[box[1]:box[3], box[0]:box[2]]; out[box[1]:box[3], box[0]:box[2]][sub] = patch[sub]
# 2) контур живота: квадратичная кривая между внутренним краем задней лапы и левым краем правой передней
P0, P1, P2 = np.array([546., 744.]), np.array([583., 708.]), np.array([620., 744.])
ts = np.linspace(0, 1, 200)[:, None]
rng = np.random.default_rng(3)
curve = (1 - ts) ** 2 * P0 + 2 * (1 - ts) * ts * P1 + ts ** 2 * P2
wob = ndimage.gaussian_filter1d(rng.normal(0, 1.4, 200), 6)
curve[:, 1] += wob * np.sin(np.pi * ts[:, 0])
YY, XX = np.mgrid[0:H, 0:W]
hsv = np.asarray(Image.fromarray(np.clip(out,0,255).astype(np.uint8)).convert("HSV")).astype(int)
warm = R & (hsv[..., 1] > 45) & ((hsv[..., 0] < 35) | (hsv[..., 0] > 240))
warm = ndimage.binary_dilation(warm, iterations=4) & R
if warm.any():
    blur = np.stack([ndimage.gaussian_filter(np.where(R & ~warm, out[..., k], np.nan_to_num(np.median(out[R & ~warm][:, k]))), 6) for k in range(3)], -1)
    out[warm] = blur[warm]
print("рыжих пикселей убрано:", int(warm.sum()))
ycurve = np.interp(XX, curve[:, 0], curve[:, 1], left=1e9, right=1e9)
below = R & (YY > ycurve)
out[below] = paper
img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
# штрих туши: толщина плавно меняется, как у пера
c = np.asarray(img).copy()
for i in range(len(curve) - 1):
    w = max(2, int(round(3.2 + 2.2 * np.sin(np.pi * i / len(curve)) + 0.9 * np.sin(i / 7.0))))
    p, q = tuple(np.round(curve[i]).astype(int)), tuple(np.round(curve[i + 1]).astype(int))
    cv2.line(c, p, q, (26, 23, 25), w, cv2.LINE_AA)
Image.fromarray(c).save("raised_fixed.png")
np.save("below_belly.npy", below)
s = Image.new("RGB", (880, 600)); s.paste(Image.open("raised.png").crop((500, 640, 720, 790)).resize((440, 300)), (0, 0))
s.paste(Image.fromarray(c).crop((500, 640, 720, 790)).resize((440, 300)), (440, 0))
s.paste(Image.fromarray(c).crop((330, 380, 900, 790)).resize((440, 300)), (0, 300)); s.paste(Image.fromarray(c).resize((440, 300)), (440, 300))
s.save("fix_check.jpg")
