"""Сцена пилота куклы в сборщике (07.10): кремовая бумага + конверт (ассет куклы), без кота.
Запуск: python pilot_run.py baseline|nomascot|mascot|samecam (из этой папки). Результат — pilot_out/."""
import json, os, sys
import numpy as np
from PIL import Image
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import canvas
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_out")
os.makedirs(P, exist_ok=True)
def build():
    out = os.path.join(P, "scene.png")
    if not os.path.exists(out):
        env = Image.open(os.path.join(os.path.dirname(P), "assets", "envelope.png")).convert("RGBA")
        env = env.resize((int(env.width*2.6), int(env.height*2.6)), Image.LANCZOS)
        bg = Image.new("RGB", (1920, 1080), tuple(int(c) for c in canvas.CREAM))
        bg.paste(env, (1010, 600), env)
        bg.save(out)
    box = [1010, 600, 1010 + int(304*2.6), 600 + int(107*2.6)]
    return out, [dict(name="envelope", box=box, word="письмо", role="subject")]
TEXT = "Ответить на одно письмо — это пять минут. А ты третий день ходишь вокруг него кругами."
D = 6.5
def words():
    ws = TEXT.split(); n = len(ws); step = (D - 0.8)/n
    return [dict(word=w, start=0.3 + i*step, end=0.3 + (i+0.9)*step) for i, w in enumerate(ws)]
