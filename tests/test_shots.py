"""Холст, камера и план кадра: бумага, 16:9 без полей-размытий, наезд на предмет,
подписи по словам, смена плана каждые 2-4 с."""
import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import camera  # noqa: E402
import canvas  # noqa: E402
import placement  # noqa: E402
import shots  # noqa: E402


def _drawing(w=1264, h=848, paper=(251, 251, 246)):
    im = Image.new("RGB", (w, h), paper)
    d = ImageDraw.Draw(im)
    d.ellipse((300, 250, 560, 600), outline=(20, 20, 20), width=8, fill=(40, 40, 40))      # «кот»
    d.rectangle((740, 360, 1000, 520), outline=(20, 20, 20), width=6)                        # «письмо»
    d.rectangle((1150, 700, 1175, 715), outline=(20, 20, 20), width=3)                       # «марка»: мелочь
    d.rectangle((150, 60, 300, 220), outline=(20, 20, 20), width=5)                          # «календарь»
    return im


def test_paper_becomes_cream_and_ink_stays_ink():
    cv, off, on_paper = canvas.prepare(_drawing())
    assert on_paper
    assert cv.shape[:2] == (848, 1508) and off == (122, 0)
    corner = cv[10, 10].astype(int)
    assert np.abs(corner - canvas.CREAM).max() <= 2
    assert cv[425, 122 + 430].max() < 60                     # тушь не посветлела


def test_padding_is_paper_without_seam_or_blur():
    cv, off, _ = canvas.prepare(_drawing())
    row = cv[790].astype(int)                                # строка без рисунка
    assert np.abs(row - canvas.CREAM).max() <= 2             # поле и шов — тот же крем
    assert np.abs(np.diff(row, axis=0)).max() <= 2


def test_full_bleed_scene_is_cropped_not_framed():
    a = (np.random.default_rng(1).random((848, 1264, 3))*120).astype(np.uint8)   # место нарисовано целиком
    cv, off, on_paper = canvas.prepare(Image.fromarray(a))
    assert not on_paper
    assert cv.shape[:2] == (711, 1264) and off[0] == 0 and off[1] <= 0


def test_texture_moves_with_the_world_and_is_subtle():
    tex = canvas.paper_texture(w=640, h=360)
    w = np.full((360, 640, 3), 250, np.uint8)
    t = canvas.bake_texture(w, tex, seed=1).astype(float)
    assert 0.5 < t.std() < 6
    assert not np.array_equal(t, canvas.bake_texture(w, tex, seed=2))


def test_window_stays_inside_and_keeps_16x9():
    for cx, cy, wd in [(0, 0, 500), (1508, 848, 900), (700, 400, 5000)]:
        x0, y0, x1, y1 = camera.window(cx, cy, wd, 1508, 848)
        assert x0 >= -1e-6 and y0 >= -1e-6 and x1 <= 1508 + 1e-6 and y1 <= 848 + 1e-6
        assert (x1 - x0)/(y1 - y0) == pytest.approx(16/9)


def test_drift_is_small():
    win = camera.window(754, 424, 1508, 1508, 848)
    for zi in (True, False):
        ws = [camera.drift(win, u, zi, 1508, 848)[2] - camera.drift(win, u, zi, 1508, 848)[0] for u in (0, 0.5, 1)]
        assert max(ws)/min(ws) <= 1 + camera.DRIFT + 1e-6


def test_punch_window_fills_frame_with_object_and_does_not_slice_neighbours():
    cv, off, _ = canvas.prepare(_drawing())
    busy = placement.busy_map(cv.astype(np.float32), margin=8)
    letter = (740 + off[0], 360, 1000 + off[0], 520)
    pw = camera.punch_window(busy, letter)
    assert pw is not None
    z = camera.zoom_of(pw, 1508, 848)
    assert 1.3 <= z <= camera.PUNCH_MAX_ZOOM + 1e-6
    assert pw[0] <= letter[0] and pw[2] >= letter[2] and pw[1] <= letter[1] and pw[3] >= letter[3]
    ww = pw[2] - pw[0]
    assert (letter[2] - letter[0])/ww > camera.PUNCH_MIN_FILL*0.9   # предмет крупно, а не точкой
    # края окна идут по пустому: соседей (кот, календарь) не режет
    edge = camera._edge_busy(busy, pw, 1.0)
    assert edge < 0.15


def test_tiny_object_gets_no_punch():
    cv, off, _ = canvas.prepare(_drawing())
    busy = placement.busy_map(cv.astype(np.float32), margin=8)
    assert camera.punch_window(busy, (1150 + off[0], 700, 1175 + off[0], 715)) is None


def test_jump_cut_detection():
    wide = camera.window(754, 424, 1508, 1508, 848)
    near = camera.window(760, 430, 1508/1.2, 1508, 848)
    far = camera.window(754, 424, 1508/1.7, 1508, 848)
    assert camera.is_jump(wide, near, 1508, 848)
    assert not camera.is_jump(wide, far, 1508, 848)


def _words(text, t0=0.2, step=0.45):
    return [{"word": w, "start": t0 + i*step, "end": t0 + i*step + 0.35} for i, w in enumerate(text.split())]


def _busy():
    cv, off, _ = canvas.prepare(_drawing())
    return placement.busy_map(cv.astype(np.float32), margin=8), off


def test_labels_appear_when_the_voice_says_them():
    busy, _ = _busy()
    words = _words("Сначала телефон потом дофамин и хочется ещё")
    p = shots.plan(8.0, busy, words, labels=[{"text": "Телефон"}, {"text": "Дофамин"}, {"text": "Хочется ещё"}])
    assert p["label_times"] == pytest.approx([words[1]["start"], words[3]["start"], words[5]["start"]])


def test_label_forms_match_and_missing_words_are_staggered():
    busy, _ = _busy()
    words = _words("Это всё про телефоны")
    p = shots.plan(6.0, busy, words, labels=[{"text": "телефон"}, {"text": "петля"}])
    assert p["label_times"][0] == pytest.approx(words[3]["start"])
    assert p["label_times"][1] > p["label_times"][0]
    assert any("петля" in n for n in p["notes"])


def test_punch_at_word_returns_by_cut_and_respects_global_gap():
    busy, off = _busy()
    letter = {"name": "письмо", "box": (740 + off[0], 360, 1000 + off[0], 520), "word": "письмо"}
    words = _words("Ответить на одно письмо это пять минут а ты ходишь кругами")
    p = shots.plan(8.0, busy, words, objects=[letter], T0=100.0)
    kinds = [s["kind"] for s in p["segments"]]
    assert "punch" in kinds and p["punch_at"] == pytest.approx(100.0 + words[3]["start"])
    i = kinds.index("punch")
    assert p["segments"][i]["t1"] - p["segments"][i]["t0"] == pytest.approx(camera.PUNCH_SEC)
    assert kinds[i + 1] == "hold"
    if i + 2 < len(kinds):
        assert kinds[i + 2] == "drift"                       # назад — склейкой, не обратным зумом
    p2 = shots.plan(8.0, busy, words, objects=[letter], T0=100.0, last_punch=95.0)
    assert "punch" not in [s["kind"] for s in p2["segments"]]


def test_long_shot_gets_a_second_view_and_no_jump_cuts():
    busy, _ = _busy()
    words = _words(" ".join(["слово"]*20), step=0.45)
    p = shots.plan(9.5, busy, words)
    segs = p["segments"]
    assert len(segs) >= 2
    assert max(s["t1"] - s["t0"] for s in segs) <= shots.MAX_VIEW_SEC + shots.CUT_SNAP_SEC + 1e-6
    for a, b in zip(segs, segs[1:]):
        assert b["t0"] - a["t0"] >= shots.MIN_VIEW_SEC - 1e-6
        if b["kind"] == "drift":
            assert not camera.is_jump(a["win"], b["win"], 1508, 848)
    starts = {round(w["start"], 6) for w in words}
    assert all(round(s["t0"], 6) in starts for s in segs[1:])     # склейки — на начале слова


def test_no_cut_or_punch_while_key_thought_is_written():
    busy, off = _busy()
    letter = {"name": "письмо", "box": (740 + off[0], 360, 1000 + off[0], 520), "word": "открыть"}
    words = _words("Договорись с собой только открыть письмо и всё на сегодня хватит правда")
    p = shots.plan(9.0, busy, words, objects=[letter], key="только открыть", key_dur=2.5)
    kt = p["key_time"]
    assert kt == pytest.approx(words[3]["start"])
    for s in p["segments"][1:]:
        assert not (kt - 0.2 < s["t0"] < kt + 2.5 + shots.WRITE_TAIL_SEC)


def test_segments_cover_the_whole_frame():
    busy, _ = _busy()
    for D in (1.0, 3.0, 7.5, 12.0):
        p = shots.plan(D, busy, _words(" ".join(["раз"]*30), step=0.4))
        segs = p["segments"]
        assert segs[0]["t0"] == 0 and segs[-1]["t1"] == pytest.approx(D)
        assert all(a["t1"] == pytest.approx(b["t0"]) for a, b in zip(segs, segs[1:]))
        for t in np.linspace(0, D - 1e-3, 7):
            w = shots.window_at(p, t, 1508, 848)
            assert w[0] >= -1e-6 and w[2] <= 1508 + 1e-6


def test_key_thought_finishes_and_then_stays_on_screen():
    busy, _ = _busy()
    words = _words("а ты третий день ходишь вокруг него кругами")
    p = shots.plan(5.0, busy, words, key="третий день", key_dur=2.3)
    kt = p["key_time"]
    assert kt is not None and kt + 2.3 + shots.KEY_HOLD_SEC <= 5.0 - 0.2 + 1e-6
    assert words[2]["start"] - kt <= shots.KEY_MAX_LEAD_SEC + 1e-6
    p2 = shots.plan(3.0, busy, words, key="третий день", key_dur=2.3)     # не успевает постоять — не пишется
    assert p2["key_time"] is None


def test_camera_leans_to_each_named_part_within_8_percent():
    busy, off = _busy()
    words = _words("сначала кот потом письмо и календарь в конце")
    labels = [{"text": "кот"}, {"text": "письмо"}, {"text": "календарь"}]
    parts = [(414, 242, 690, 608), (862, 360, 1122, 520), (272, 60, 422, 220)]
    p = shots.plan(8.0, busy, words, labels=labels, parts=parts)
    seg = p["segments"][0]
    assert seg.get("pushes") and all(any(abs(ps["t"] - lt) < 1e-9 for lt in p["label_times"]) for ps in seg["pushes"])
    wide = seg["win"][2] - seg["win"][0]
    for t in np.linspace(seg["t0"], seg["t1"] - 1e-3, 40):
        w = shots.window_at(p, t, 1508, 848)
        assert wide/(w[2] - w[0]) <= 1 + shots.PUSH_MAX + 1e-6
    for ps in seg["pushes"]:                     # куда камера наклоняется — без разреза рисунка
        assert camera.edge_cross(busy, ps["win"]) <= camera.PUNCH_MAX_CROSS + 1e-9
    # после наклона камера не стоит: выдох обратно к общему плану
    last = seg["pushes"][-1]["t"] + shots.PUSH_SEC
    a, b = shots.window_at(p, last + 0.05, 1508, 848), shots.window_at(p, seg["t1"] - 0.01, 1508, 848)
    assert (b[2] - b[0]) > (a[2] - a[0])


def test_punch_may_crop_its_own_group_but_never_a_separate_object():
    im = Image.new("RGB", (1264, 848), (251, 251, 246))
    d = ImageDraw.Draw(im)
    d.ellipse((380, 250, 640, 700), fill=(40, 40, 40))            # «кот»
    d.rectangle((600, 380, 900, 560), fill=(230, 230, 230), outline=(20, 20, 20), width=6)   # конверт в лапах
    d.rectangle((1050, 80, 1200, 200), outline=(20, 20, 20), width=6)                          # отдельный календарь
    cv, off, _ = canvas.prepare(im)
    busy = placement.busy_map(cv.astype(np.float32), margin=8)
    env = (600 + off[0], 380, 900 + off[0], 560)
    pw = camera.punch_window(busy, env)
    assert pw is not None                                         # раньше: «режет кота» -> наезда нет
    cal = (1050 + off[0], 80, 1200 + off[0], 200)
    sep = camera.others(busy, env)
    assert sep[int(cal[1]) + 5:int(cal[3]) - 5, int(cal[0]) + 5:int(cal[2]) - 5].max() > 0.3   # календарь — отдельный
    assert camera.edge_cross(sep, pw) <= camera.PUNCH_MAX_CROSS


def test_long_shot_without_a_second_view_leans_slowly_to_the_subject():
    busy, off = _busy()
    words = _words(" ".join(["слово"]*20), step=0.45)
    objs = [{"role": "subject", "box": (414, 242, 690, 608)}, {"role": "hero", "box": (300, 60, 1150, 600)}]
    p = shots.plan(9.5, busy, words, objects=objs)
    seg = p["segments"][0]
    if len(p["segments"]) == 1:                       # средний план невозможен — медленный наезд
        assert seg.get("lean")
        a, b = shots.window_at(p, 0.0, 1508, 848), shots.window_at(p, 9.4, 1508, 848)
        z = (a[2] - a[0])/(b[2] - b[0])
        assert 1.04 < z <= 1 + shots.PUSH_MAX + 1e-6


def _one_blob():
    """Кот, мозг и глыба слиты в одну фигуру (живой кадр 05.10): среднего плана без
    разреза нет, а крупные планы деталей есть."""
    im = Image.new("RGB", (1264, 848), (251, 251, 246))
    d = ImageDraw.Draw(im)
    d.ellipse((380, 120, 720, 420), fill=(60, 60, 60))          # «мозг»
    d.ellipse((680, 160, 960, 420), fill=(40, 40, 40))          # «глыба», касается мозга
    d.rectangle((470, 380, 640, 720), fill=(30, 30, 30))        # «кот» под мозгом
    d.line((640, 700, 900, 640), fill=(30, 30, 30), width=18)   # «хвост»
    d.ellipse((860, 560, 960, 690), fill=(30, 30, 30))          # «огонёк с дымком» на конце хвоста
    cv, off, _ = canvas.prepare(im)
    return placement.busy_map(cv.astype(np.float32), margin=8), off


def test_a_long_shot_cuts_to_close_ups_of_named_details_every_few_seconds():
    busy, off = _one_blob()
    ox = off[0]
    hero = {"role": "hero", "box": (470 + ox - 10, 380, 940 + ox, 720)}
    objs = [{"role": "subject", "box": (380 + ox, 120, 960 + ox, 420)}, hero,
            {"role": "hero_head", "box": (470 + ox, 380, 640 + ox, 520)},
            {"role": "detail", "name": "the boulder", "box": (680 + ox, 160, 960 + ox, 420)},
            {"role": "detail", "name": "the tail flame", "box": (855 + ox, 555, 965 + ox, 695)}]
    words = _words(" ".join(["слово"]*20), step=0.45)
    p = shots.plan(9.0, busy, words, objects=objs)
    segs = p["segments"]
    assert len(segs) >= 3                                         # критик: 8.4 с одним планом
    assert max(s["t1"] - s["t0"] for s in segs) <= shots.MAX_VIEW_SEC + shots.CUT_SNAP_SEC + 1e-6
    wide = segs[0]["win"]
    closes = [s["win"] for s in segs if s["win"] != wide]
    assert closes
    for c in closes:                                              # каждый крупный план — на детали целиком
        inside = [o for o in objs[3:] if c[0] <= o["box"][0] and c[1] <= o["box"][1]
                  and c[2] >= o["box"][2] and c[3] >= o["box"][3]]
        assert inside
    assert len({s["zoom_in"] for s in segs if s["kind"] == "drift"}) == 1   # одно направление на мысль


def test_no_detail_close_up_that_slices_the_hero_head():
    busy, off = _one_blob()
    ox = off[0]
    head = {"role": "hero_head", "box": (470 + ox, 380, 640 + ox, 520)}
    objs = [{"role": "hero", "box": (470 + ox, 380, 940 + ox, 720)}, head,
            {"role": "detail", "name": "the boulder", "box": (680 + ox, 160, 960 + ox, 420)},
            {"role": "detail", "name": "a speck by the head", "box": (600 + ox, 430, 660 + ox, 480)}]
    p = shots.plan(9.0, busy, _words(" ".join(["слово"]*20)), objects=objs)
    b = head["box"]
    for s in p["segments"]:
        w = s["win"]
        overlap = not (w[2] <= b[0] or w[0] >= b[2] or w[3] <= b[1] or w[1] >= b[3])
        whole = w[0] <= b[0] and w[1] <= b[1] and w[2] >= b[2] and w[3] >= b[3]
        assert whole or not overlap                       # голова целиком или вне кадра
