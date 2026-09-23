#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Лист «до/после»: строка на слот, пропавший кадр виден, а не пропущен."""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import runs_sheet  # noqa: E402


def _ep(tmp_path, name, files):
    ep = tmp_path / name
    (ep / "media_plan").mkdir(parents=True)
    shots = []
    for i, f in enumerate(files):
        if f:
            Image.new("RGB", (64, 36), (200, 50, 50)).save(ep / f)
        shots.append({"index": i, "text": f"фраза {i}", "file": f, "kind": "photo", "provider": "met"})
    (ep / "media_plan" / "shotlist.json").write_text(json.dumps({"shots": shots}), encoding="utf-8")
    return str(ep)


def test_one_row_per_slot_and_missing_frame_is_marked(tmp_path):
    a = _ep(tmp_path, "a", ["0.jpg", "1.jpg"])
    b = _ep(tmp_path, "b", ["0.jpg", None])
    out = str(tmp_path / "s.jpg")
    w, h = runs_sheet.build(out, [("до", a), ("после", b)])
    assert h == 40 + 2 * (runs_sheet.TH + 30)
    assert w == 330 + 2 * (runs_sheet.TW + 10)
    im = Image.open(out).convert("RGB")
    x = 330 + (runs_sheet.TW + 10) + 200
    y = 40 + (runs_sheet.TH + 30) + 120
    r, g, _b = im.getpixel((x, y))
    assert r > g, "пропавший кадр второго прогона помечен красным полем"
