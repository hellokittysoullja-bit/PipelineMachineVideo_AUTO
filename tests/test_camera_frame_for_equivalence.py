"""camera.frame_for считается векторно; эталон — построчный перебор _frame_for_scan. Один и тот же выбор на
реальных картах занятости тестовых сцен, со всеми режимами (max_cross, cross_map, away, accept)."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import camera  # noqa: E402
import canvas  # noqa: E402
import placement  # noqa: E402


def _busy(draw):
    im = Image.new("RGB", (1264, 848), (251, 251, 246))
    draw(ImageDraw.Draw(im))
    cv, off, _ = canvas.prepare(im)
    return placement.busy_map(cv.astype(np.float32), margin=8), off[0]


def _scenes():
    def cat_and_letter(d):
        d.ellipse((150, 120, 450, 400), fill=(40, 40, 40)); d.rectangle((200, 380, 420, 700), fill=(30, 30, 30))
        d.rectangle((700, 560, 1100, 760), fill=(60, 60, 60))

    def one_blob(d):
        d.ellipse((380, 120, 720, 420), fill=(60, 60, 60)); d.ellipse((680, 160, 960, 420), fill=(40, 40, 40))
        d.rectangle((470, 380, 640, 720), fill=(30, 30, 30)); d.line((640, 700, 900, 640), fill=(30, 30, 30), width=18)
        d.ellipse((860, 560, 960, 690), fill=(30, 30, 30))

    def two_objects(d):
        d.rectangle((100, 500, 500, 780), fill=(50, 50, 50)); d.ellipse((800, 100, 1150, 450), fill=(70, 70, 70))
    return [cat_and_letter, one_blob, two_objects]


def test_vectorized_frame_for_matches_the_scan_on_every_mode():
    rng = np.random.default_rng(3)
    n = 0
    for draw in _scenes():
        busy, ox = _busy(draw)
        SH, SW = busy.shape
        targets = [(700 + ox, 560, 1100 + ox, 760), (150 + ox, 120, 450 + ox, 400), (380 + ox, 120, 960 + ox, 420),
                   (860 + ox, 560, 960 + ox, 690), (100 + ox, 500, 500 + ox, 780)]
        for tgt in targets:
            for kw in (dict(), dict(margin=0.06, spread=0.35, grid=17, max_cross=0.0, cross_map=camera.others(busy, tgt)),
                       dict(margin=0.02, spread=0.2, max_cross=0.0, away=(SW/2, 0, 0.1*SW)),
                       dict(margin=0.06, spread=0.25, grid=13, accept=lambda w: (w[3] - w[1]) < 0.7*SH),
                       dict(margin=0.12, bottom_w=1.0, edge_k=10.0, spread=0.15, grid=9)):
                for z in ((1.0, 1.45), (1.55, 2.1), (1.53, 2.5), (1.15, 1.53)):
                    a = camera.frame_for(busy, tgt, z, **kw)
                    b = camera._frame_for_scan(busy, tgt, z, **kw)
                    assert a == b, (draw.__name__, tgt, z, kw.keys(), a, b)
                    n += 1
    assert n >= 100
