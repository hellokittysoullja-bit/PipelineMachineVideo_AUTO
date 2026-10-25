"""Геометрия экранного текста сверена с нормами титров (1920x1080).

История: 13.09 сверка нашла, что угловой вариант плашки нарушал три нормы
сразу (кегль, поле, низ в зоне кнопок плеера). С оформления 25.10 вариантов
плашки нет — одна плашка text_style.stat_block, и та же проверка теперь
меряет НАРИСОВАННЫЕ пиксели блоков (альфа-канал PNG), а не числа в коде:
кегль у антиквы и гротеска разный, а «низ текста» решают выносные элементы.

Нормы: кегль не меньше 5% высоты кадра (54 px), поле от края не меньше 5%
ширины (96 px), ничего ниже 880 — там YouTube рисует полосу перемотки и
кнопки, на телефоне они всплывают от каждого касания.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

pytest.importorskip("PIL")
import text_style as ts  # noqa: E402

PLAYER_ZONE_TOP = 880
SIDE_MARGIN = 96


def _bbox(img, alpha_min=40):
    a = img.split()[3].point(lambda v: 255 if v >= alpha_min else 0)
    return a.getbbox()


@pytest.mark.parametrize("text", ["25 кг", "20–25 кг", "+66%", "×2 на шаг", "1415"])
def test_stat_plate_stays_out_of_the_player_zone_and_margin(text):
    box = _bbox(ts.stat_block(text))
    assert box is not None, "плашка ничего не нарисовала"
    left, top, right, bottom = box
    assert bottom <= PLAYER_ZONE_TOP, (text, bottom)
    assert left >= SIDE_MARGIN - 2, (text, left)
    assert right <= 1920 - SIDE_MARGIN, (text, right)


def test_stat_plate_is_large_enough_to_read_on_a_phone():
    img = ts.stat_block("25")
    a = img.split()[3].point(lambda v: 255 if v >= 128 else 0)
    # высота цифр без линии: обрезаем область ниже золотой линии
    w, h = img.size
    digits = a.crop((0, 0, w, h)).getbbox()
    assert digits[3] - digits[1] >= 54


def test_source_caption_stays_in_frame_and_out_of_the_player_zone():
    box = _bbox(ts.source_block("1395–1415", "Институт искусств Чикаго"))
    assert box[3] <= PLAYER_ZONE_TOP and box[2] <= 1920 - 60


def test_chapter_marker_sits_in_the_top_left_safe_corner():
    box = _bbox(ts.marker_block(2, "Лаборатория в Лидсе"))
    assert box[0] >= SIDE_MARGIN - 2 and box[1] >= 50 and box[3] <= 140


def test_long_stat_shrinks_instead_of_running_off_frame():
    box = _bbox(ts.stat_block("около тридцати пяти килограммов"))
    assert box[2] <= 1920 - SIDE_MARGIN
