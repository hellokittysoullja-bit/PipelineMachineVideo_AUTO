#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Один прогон картинки на все гейты: релевантность, домен-гвард и вето
читают один эмбеддинг файла, тексты считаются по разу. Замер 24.09: 72%
процессорного времени отбора уходило на повторный пересчёт того же."""
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402


def test_image_is_embedded_once_for_all_gates_and_texts(tmp_path, monkeypatch):
    calls = {"image": 0, "text": 0}

    def fake(images=None, text=None):
        if text is not None:
            calls["text"] += 1
            v = np.zeros((1, 4), "float32")
            v[0, len(text) % 4] = 1.0
            return v
        calls["image"] += 1
        return np.array([[1.0, 0.0, 0.0, 0.0]], "float32")

    monkeypatch.setattr(ps, "_gate_embed", fake)
    monkeypatch.setattr(ps, "CLIP_ENABLED", True)
    monkeypatch.setattr(ps, "CLIP_BROKEN", False)
    p = str(tmp_path / "a.jpg")
    Image.new("RGB", (16, 16), (200, 10, 10)).save(p)
    ps.clip_relevance_multi(p, ["abcd", "abcde", "abcdef"])
    ps.clip_relevance_multi(p, ["abcd", "xyz"])
    ps.clip_relevance(p, "q")
    assert calls["image"] == 1, "картинка один раз на все вызовы"
    assert calls["text"] == 5, "каждый текст один раз"
    # Тот же кадр, скачанный заново в другой файл, — тот же эмбеддинг;
    # другой кадр в том же имени — новый.
    q = str(tmp_path / "b.jpg")
    open(q, "wb").write(open(p, "rb").read())
    ps.clip_relevance(q, "q")
    assert calls["image"] == 1
    Image.new("RGB", (16, 16), (10, 200, 10)).save(p)
    ps.clip_relevance_multi(p, ["abcd"])
    assert calls["image"] == 2
