import json

import pytest
from PIL import Image, ImageDraw

import labels


def blank(tmp_path, name="raw.png", size=(1920, 1080), color=(250, 250, 246)):
    p = tmp_path / name
    Image.new("RGB", size, color).save(p)
    return str(p)


def test_fonts_cover_russian_and_fallback_for_missing_glyphs():
    assert labels.font_for("ЁЛКИ-ПАЛКИ, 12:00!").endswith("ShantellSans-ExtraBold.ttf")
    with pytest.raises(ValueError):
        labels.font_for("☃")


def test_fit_wraps_long_label_inside_box():
    size, lines = labels.fit("ЕДА, БЕЗОПАСНОСТЬ, ПРИНАДЛЕЖНОСТЬ (ОЧЕНЬ МНОГО ЛЮДЕЙ!)", 520, 260, labels.FONT_PRIMARY)
    assert len(lines) >= 2 and size >= 30


def test_caption_goes_to_empty_bottom_band_without_outline(tmp_path):
    ok, info = labels.compose(blank(tmp_path), str(tmp_path / "out.png"),
                              {"kind": "caption", "labels": ["ЖИВ. ПОЛНОСТЬЮ."]})
    assert ok and info[0]["color"] == "dark" and info[0]["size"] > 80
    out = Image.open(tmp_path / "out.png").convert("RGB")
    # нет обводки: в кадре только фон и цвет текста (плюс сглаживание между ними)
    band = out.crop(info[0]["box"])
    assert not any(px == (255, 255, 255) for px in band.getdata())


def test_light_text_on_dark_background(tmp_path):
    ok, info = labels.compose(blank(tmp_path, color=(40, 30, 20)), str(tmp_path / "o.png"),
                              {"kind": "caption", "labels": ["ТЬМА"]})
    assert ok and info[0]["color"] == "light"


def test_busy_bottom_asks_the_vision_model_and_refuses_drawing_under_text(tmp_path):
    raw = tmp_path / "busy.png"
    im = Image.new("RGB", (1920, 1080), (250, 250, 246))
    d = ImageDraw.Draw(im)
    for x in range(0, 1920, 40):
        d.line([(x, 850), (x + 30, 1070)], fill=(0, 0, 0), width=6)     # рисунок внизу
    im.save(raw)

    class VLM:
        def __init__(self, box):
            self.box = box

        def chat(self, *a, **k):
            return json.dumps({"boxes": {"1": self.box}}), {}, 0

    ok, info = labels.compose(str(raw), str(tmp_path / "a.png"), {"kind": "caption", "labels": ["ЖИВ"]},
                              VLM([100, 100, 600, 400]), "m")
    assert ok and info[0]["box"][1] < 500                      # легла в пустое место сверху
    ok, why = labels.compose(str(raw), str(tmp_path / "b.png"), {"kind": "caption", "labels": ["ЖИВ"]},
                             VLM([0, 800, 1000, 1000]), "m")
    assert not ok and "не пустое" in why                        # на рисунок — не ставим


def test_no_vision_model_and_no_empty_place_is_an_honest_refusal(tmp_path):
    ok, why = labels.compose(blank(tmp_path), str(tmp_path / "o.png"),
                             {"kind": "diagram", "labels": ["А", "Б"]})
    assert not ok and "нет модели" in why


def test_parse_boxes_rejects_garbage():
    assert labels.parse_boxes('{"boxes": {"1": [1, 2, 3]}}', 1) is None
    assert labels.parse_boxes("нет json", 1) is None
    assert labels.parse_boxes('{"boxes": {"1": [100, 100, 400, 300]}}', 1)[1] == (100, 100, 400, 300)


def test_model_patch_under_a_label_is_flattened_but_the_rest_of_the_drawing_is_not():
    """Живой кадр 01.10: светлые прямоугольники под подписями (на 3-9 светлее
    фона) — убираются; светлый экран телефона вдали от подписей — нет."""
    import numpy as np
    from PIL import Image, ImageDraw
    import labels
    img = Image.new("RGB", (1264, 848), (240, 239, 234))
    d = ImageDraw.Draw(img)
    d.rectangle((900, 60, 1200, 140), fill=(246, 246, 241))     # пятно модели под подпись
    d.rectangle((300, 400, 400, 600), fill=(250, 250, 248))     # экран телефона, далеко от подписи
    d.line((100, 100, 300, 100), fill=(30, 30, 30), width=4)    # линия рисунка
    out = np.asarray(labels.flatten_paper(img, [(920, 70, 1180, 130)]), dtype=int)
    bg = np.array(labels.background_color(img))
    assert np.abs(out[100, 1000] - bg).max() <= 1               # пятна нет
    assert tuple(out[500, 350]) == (250, 250, 248)              # экран не тронут
    assert tuple(out[100, 200]) == (30, 30, 30)                 # линия не тронута
