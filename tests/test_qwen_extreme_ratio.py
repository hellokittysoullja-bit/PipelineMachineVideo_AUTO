#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Картинка шире 200:1 (полоса-превью, свиток из Commons) не роняет модели
Qwen: раньше smart_resize бросал ValueError, пачка выключала модель до конца
прогона, и рендер вставал на этом кадре при каждом перезапуске."""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import qwen_vl_embed  # noqa: E402
import qwen_vl_rerank  # noqa: E402


def test_wide_and_tall_strips_are_prepared_instead_of_raising():
    for size in ((20000, 40), (40, 20000), (12001, 60)):
        im = Image.new("RGB", size, (120, 80, 40))
        cw, ch = qwen_vl_embed.within_ratio(im).size
        assert max(cw, ch) / min(cw, ch) <= qwen_vl_embed.MAX_RATIO
        for prep in (qwen_vl_embed.prepare_image, qwen_vl_rerank.prepare_image):
            prep(im)          # раньше — ValueError, и модель выключалась до конца прогона


def test_ordinary_pictures_are_not_touched():
    im = Image.new("RGB", (1920, 1080), (1, 2, 3))
    assert qwen_vl_embed.within_ratio(im) is im
    exact = Image.new("RGB", (8000, 40))           # ровно 200:1 — как было
    assert qwen_vl_embed.within_ratio(exact) is exact
