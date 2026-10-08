"""Нарезка эталонного кота на части: мех без глаз (LaMa), радужка без зрачка, зрачки-спрайты."""
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy import ndimage
src = Image.open("hero.png").convert("RGB"); W, H = src.size
a = np.asarray(src).astype(np.float32); m = np.asarray(Image.open("mask.png")) > 128
hsv = np.asarray(src.convert("HSV")).astype(int); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
EYES = [(444, 341, 568, 469), (658, 341, 784, 469)]
eyes = []
for (x0, y0, x1, y1) in EYES:
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    YY, XX = np.mgrid[0:H, 0:W]
    ell = ((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2
    iris = ell <= 1.0                                  # радужка со зрачком
    whole = ell <= ((rx + 9) / rx) ** 2               # с чёрной обводкой
    pupil = iris & ((v < 120) | ((s < 60) & (v > 200)))  # зрачок + блик
    pupil = ndimage.binary_closing(pupil, iterations=2)
    # только зрачок и блик: связные куски, касающиеся внутренней части радужки (без обводки на краю)
    core = (((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2) <= 0.55 ** 2
    lab, n = ndimage.label(pupil)
    keep = [i for i in range(1, n + 1) if (core & (lab == i)).any()]
    pupil = np.isin(lab, keep)
    eyes.append(dict(c=(cx, cy), r=(rx, ry), iris=iris, whole=whole, pupil=pupil))
# 1) мех без глаз
allw = np.zeros_like(m)
for e in eyes: allw |= ndimage.binary_dilation(e["whole"], iterations=6)
box = (330, 180, 330 + 570, 180 + 570)
sess = ort.InferenceSession("../cutout/lama_fp32.onnx")
ci = np.asarray(src.crop(box).resize((512, 512), Image.LANCZOS)).astype(np.float32) / 255
cm = np.asarray(Image.fromarray((allw * 255).astype(np.uint8)).crop(box).resize((512, 512), Image.NEAREST)) > 127
cm = ndimage.binary_dilation(cm, iterations=3)
ci = (ci * (1 - cm[..., None])).astype(np.float32)
o = sess.run(None, {"image": ci.transpose(2, 0, 1)[None], "mask": cm.astype(np.float32)[None, None]})[0][0].transpose(1, 2, 0)
patch = np.asarray(Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).resize((box[2] - box[0],) * 2, Image.LANCZOS)).astype(np.float32)
fur = a.copy(); sub = allw[box[1]:box[3], box[0]:box[2]]
fur[box[1]:box[3], box[0]:box[2]][sub] = patch[sub]
Image.fromarray(fur.astype(np.uint8)).save("part_fur.png")
# 2) радужка без зрачка: зрачок закрашиваем средним зелёным с лёгкой акварельной неровностью
iris_img = a.copy()
rng = np.random.default_rng(1)
for e in eyes:
    g = np.median(a[e["iris"] & ~e["pupil"]], axis=0)
    noise = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), 6)[..., None] * 25
    fill = np.clip(g + noise, 0, 255)
    pm = ndimage.binary_dilation(e["pupil"], iterations=6) & e["iris"]
    iris_img[pm] = fill[pm]
Image.fromarray(iris_img.astype(np.uint8)).save("part_iris.png")
np.save("eyes.npy", np.array([[*e["c"], *e["r"]] for e in eyes]))
np.save("masks.npy", np.stack([np.stack([e["iris"], e["whole"], e["pupil"]]) for e in eyes]))
# лист для проверки глазами
sheet = Image.new("RGB", (1500, 500), "white")
for i, im in enumerate([src, Image.open("part_fur.png"), Image.open("part_iris.png")]):
    sheet.paste(im.crop((330, 180, 900, 750)).resize((500, 500)), (i * 500, 0))
sheet.save("parts_sheet.jpg")
