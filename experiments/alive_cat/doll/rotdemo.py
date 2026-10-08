import numpy as np
import cv2
import subprocess
from PIL import Image
exec(open("rotleg.py").read().split('# 1) основа')[0])
base = np.asarray(Image.open("base_noleg.png")).astype(np.float32); base_a = np.asarray(Image.open("base_noleg_a.png")).astype(np.float32)
L0 = np.load("leg_rgba.npy")
EDGE = np.array([[623, 688], [623, 745], [618, 770]], np.float32)       # правый край лапы — линия туши
env = Image.open("../paw/envelope.png"); env = env.resize((int(env.size[0] * .6), int(env.size[1] * .6)), Image.LANCZOS)
def frame(deg):
    M = cv2.getRotationMatrix2D(PIVOT, deg, 1.0)
    L = cv2.warpAffine(L0, M, (W, H), flags=cv2.INTER_LINEAR)
    a = L[..., 3:4] / 255; rgb = base * (1 - a) + L[..., :3] * a; al = np.maximum(base_a, L[..., 3])
    a2 = al[..., None] / 255; out = (paper * (1 - a2) + rgb * a2).clip(0, 255).astype(np.uint8).copy()
    k = 0.0  # линия края отключена: уходила за лапу
    if k > 0:
        pts = (np.c_[EDGE, np.ones(3)] @ M.T).astype(np.float32)
        ov = out.copy()
        cv2.polylines(ov, [np.round(pts).astype(np.int32)], False, (26, 23, 25), 3, cv2.LINE_AA)
        out = (out * (1 - k) + ov * k).astype(np.uint8)
    im = Image.fromarray(out); im.paste(env, (250, 778 - env.size[1]), env)
    return np.asarray(im)
def ease(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)
FPS, DUR = 30, 4.0
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-vf", "scale=1920:-2:flags=lanczos,crop=1920:1080", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "reach_rot.mp4"], stdin=subprocess.PIPE)
for i in range(int(FPS * DUR)):
    t = i / FPS
    deg = -38 * (ease((t - .6) / .5) - ease((t - 2.6) / .5))
    p.stdin.write(frame(deg).tobytes())
p.stdin.close(); p.wait()
Image.fromarray(frame(-38)).save("reach_rot_final.png")
Image.fromarray(frame(-38)).crop((440, 620, 700, 790)).resize((780, 510), Image.LANCZOS).save("m38_zoom.jpg")
