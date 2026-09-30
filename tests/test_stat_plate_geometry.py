"""Геометрия экранных плашек сверена с отраслевыми нормами титров.

РЕАЛЬНАЯ находка (13.09, сверка всех четырёх вариантов разом). Варианты
0, 1, 2 и машинка нормам соответствовали. Вариант 3 (подпись в нижнем
правом углу) нарушал ТРИ нормы одновременно:

    кегль            46-52  при норме от 54 (5% высоты кадра)
    правое поле      70     при норме от 96 (5% ширины кадра)
    низ текста       940    при пределе 880
    низ подчёркивания 992   при пределе 880

Последнее — самое важное: нижние ~18% кадра это зона, где YouTube рисует
полосу перемотки и кнопки. На телефоне они всплывают от каждого касания и
перекрывают подпись. Причём в истории правок рядом стоит жалоба
пользователя "слишком мелкая и незаметная", по которой кегль уже поднимали
с 34-38 до 46-52 — то есть проблему знали, но до нормы не довели, а
позицию и отступ не проверял никто.

Вариант 3 выбирается для ПЛОТНЫХ кадров (busy > 0.55) — на канале про
макросъёмку клинков и доспехов он выпадает часто, это не редкий случай.
"""
import os
import re
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402

MIN_FONT = 54          # 5% высоты кадра — читаемо на телефоне
PLAYER_ZONE_TOP = 880  # ниже этого YouTube рисует полосу перемотки и кнопки


def _overlay(variant, text="25 КГ"):
    """Граф оверлея для конкретного варианта плашки."""
    return ps.add_overlays("null", 8.0, stat=text, stat_variant=variant, stat_delay=1.0)


@pytest.mark.parametrize("variant", [0, 1, 2, 3, 4])
def test_font_size_is_readable_on_a_phone(variant):
    vf = _overlay(variant)
    sizes = [int(m) for m in re.findall(r"fontsize=(\d+)", vf)]
    assert sizes, vf[:200]
    assert min(sizes) >= MIN_FONT, f"вариант {variant}: кегль {min(sizes)} < {MIN_FONT}"


@pytest.mark.parametrize("variant", [0, 1, 2, 3, 4])
def test_nothing_is_drawn_in_the_player_controls_zone(variant):
    """Текст и декор не имеют права заезжать туда, где всплывают кнопки
    плеера — иначе на телефоне подпись просто перекрывается."""
    vf = _overlay(variant)
    bottoms = [ps.HEIGHT - int(m) for m in re.findall(r"y=(?:')?(?:h|ih)-(\d+)", vf)]
    for y in bottoms:
        assert y <= PLAYER_ZONE_TOP, (
            f"вариант {variant}: элемент на y={y}, ниже предела {PLAYER_ZONE_TOP}")


def test_corner_variant_respects_the_side_margin():
    """Правое поле — не меньше 5% ширины кадра."""
    vf = _overlay(3)
    margins = [int(m) for m in re.findall(r"x=w-text_w-(\d+)", vf)]
    assert margins, vf[:200]
    assert min(margins) >= ps.STAT_SAFE_MARGIN_PX >= 96


def test_margin_constants_match_the_standard():
    assert ps.STAT_SAFE_MARGIN_PX >= round(ps.WIDTH * 0.05)
    assert ps.HEIGHT - ps.STAT_CORNER_RULE_BOTTOM_PX <= PLAYER_ZONE_TOP
    assert ps.HEIGHT - ps.STAT_CORNER_TEXT_BOTTOM_PX <= PLAYER_ZONE_TOP


def test_text_sits_above_its_own_underline():
    """Подчёркивание — декор ПОД подписью, не поверх неё."""
    assert ps.STAT_CORNER_TEXT_BOTTOM_PX > ps.STAT_CORNER_RULE_BOTTOM_PX


@pytest.mark.parametrize("variant", [0, 1, 2, 3, 4])
def test_plate_still_renders_its_text(variant):
    """Правка геометрии не должна потерять сам текст плашки."""
    assert "drawtext" in _overlay(variant)
