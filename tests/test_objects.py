"""Рамки предметов для камеры: разбор ответа, проверка по пикселям, что искать."""
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import objects  # noqa: E402


def test_parse_keeps_order_and_marks_missing():
    got = objects.parse('{"boxes": {"1": [100, 200, 300, 400], "2": null}}', 3)
    assert got == {1: (100.0, 200.0, 300.0, 400.0), 2: None, 3: None}
    assert objects.parse("nothing", 1) is None


def test_check_rejects_empty_paper_dots_and_whole_frame():
    im = Image.new("RGB", (1000, 600), (250, 250, 245))
    ImageDraw.Draw(im).ellipse((400, 200, 600, 400), fill=(20, 20, 20))
    assert objects.check(im, (380, 180, 620, 420)) is None
    assert objects.check(im, (10, 10, 200, 150)) == "внутри пустая бумага"
    assert objects.check(im, (500, 300, 505, 305)) == "точка, а не предмет"
    assert objects.check(im, (0, 0, 1000, 600)) == "во весь кадр"


def test_wanted_zoom_object_and_subject():
    f = {"hero": True, "zoom": {"object": "the envelope", "word": "письмо"}, "spec": {"subject": "a letter"}}
    assert objects.wanted(f, "a black cartoon cat") == [("the envelope", "zoom", "письмо"),
                                                        ("a black cartoon cat", "subject", None)]
    assert objects.wanted({"spec": {"subject": "a timer"}}, None) == [("a timer", "subject", None)]
    assert objects.wanted({}, None) == []


class FakeGW:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def chat(self, model, content, max_out, est, reasoning=False):
        self.calls += 1
        return self.answer, None, 0


def test_locate_returns_pixels_and_caches(tmp_path):
    p = tmp_path / "f.png"
    im = Image.new("RGB", (1000, 500), (250, 250, 245))
    ImageDraw.Draw(im).rectangle((100, 100, 300, 300), fill=(30, 30, 30))
    im.save(p)
    gw = FakeGW('{"boxes": {"1": [90, 180, 310, 620], "2": [600, 600, 700, 700]}}')
    recs = objects.objects_for(gw, "m", str(p), {"zoom": {"object": "box", "word": "ящик"},
                                                 "spec": {"subject": "sky"}}, None, str(tmp_path))
    assert recs[0]["role"] == "zoom" and recs[0]["box"] == [90.0, 90.0, 310.0, 310.0]
    assert recs[1]["box"] is None and recs[1]["why"] == "внутри пустая бумага"
    objects.objects_for(gw, "m", str(p), {"zoom": {"object": "box", "word": "ящик"},
                                          "spec": {"subject": "sky"}}, None, str(tmp_path))
    assert gw.calls == 1


def test_key_anchor_is_searched_only_with_a_key():
    f = {"key_thought": "третий день", "key_near": "the wall calendar", "spec": {"subject": "a desk"}}
    assert ("the wall calendar", "key_anchor", None) in objects.wanted(f, None)
    assert all(r != "key_anchor" for _n, r, _w in objects.wanted({"key_near": "x", "spec": {}}, None))
