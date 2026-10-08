"""Живая кукла героя в кадре сборщика (mascot_live): без неё — байт в байт прежний план; с ней —
стоит рядом с предметом, карты занятости её видят, кладётся через альфу, в плане только при риге."""
import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import mascot_live as M  # noqa: E402

needs_rig = pytest.mark.skipif(not M.available(), reason="рига куклы нет на диске")


@pytest.fixture(autouse=True)
def _no_neural(monkeypatch):
    monkeypatch.setenv("UPSCALE_NEURAL", "0")


def _fr(tmp_path):
    import canvas
    import frame_clip
    src = tmp_path / "raw.png"
    im = Image.new("RGB", (1264, 848), (251, 251, 246))
    ImageDraw.Draw(im).rectangle((860, 500, 1180, 640), outline=(20, 20, 20), width=8)   # предмет справа
    im.save(src)
    objs = [dict(name="envelope", box=[860, 500, 1180, 640], word="письмо", role="subject")]
    return frame_clip.prepare(str(src), [], objs, str(tmp_path), tex=canvas.paper_texture(w=1152, h=648))


def _words():
    return [{"word": w, "start": 0.3 + 0.45*i, "end": 0.6 + 0.45*i}
            for i, w in enumerate("ответить на письмо это пять минут а ты ходишь кругами?".split())]


def test_plan_without_mascot_has_no_trace(tmp_path):
    import frame_clip
    p = frame_clip.plan_clip(_fr(tmp_path), 5.0, _words(), fps=24)
    assert "mascot" not in p and not any("кукл" in n for n in p["notes"])


def test_no_rig_on_disk_means_frame_without_doll_and_a_note(tmp_path, monkeypatch):
    import frame_clip
    monkeypatch.setattr(M, "RIG_DIR", str(tmp_path / "nowhere"))
    p = frame_clip.plan_clip(_fr(tmp_path), 5.0, _words(), fps=24, mascot=dict(text="x", seed=0))
    assert "mascot" not in p and any("рига нет" in n for n in p["notes"])


def test_flag_off_disables_the_layer(monkeypatch):
    monkeypatch.setenv("MASCOT_LIVE", "0")
    assert not M.available()


def test_plan_actions_from_the_phrase():
    acts = M.plan_actions(5.0, _words(), "", [1.0, 0.2])
    kinds = [a["do"] for a in acts]
    assert kinds == ["look", "tilt"] and acts[1]["deg"] == 5          # вопрос — наклон головы
    assert M.plan_actions(1.0, [], "", [0, 0]) == []                  # короткий кадр — ничего
    acts = M.plan_actions(4.0, _words(), "", [0, 0], action="paw_on_chest")
    assert acts[0]["do"] == "paw" and acts[0]["dur"] <= 2.4
    # жестов лап (point/tap/wave) план не ставит: владелец их отложил
    assert not {a["do"] for a in acts} & {"point", "tap", "wave"}


@needs_rig
def test_place_puts_the_doll_beside_the_subject_and_marks_occupancy(tmp_path):
    fr = _fr(tmp_path)
    busy0 = fr["busy"].copy(); final0 = fr["final_rgb"].copy()
    ms = M.place(fr, dict(text="x"))
    assert ms and ms["side"] == "left"
    x0, y0, x1, y1 = ms["box"]
    assert x1 < fr["objects"][0]["box"][0]                          # слева от предмета, без касания
    assert y1 <= 0.86 * fr["SH"] + 1 and y0 >= 0.06 * fr["SH"] - 1  # внутри безопасной зоны
    assert ms["gaze"][0] > 0.5                                       # смотрит вправо, на предмет
    assert (fr["busy"][int(y0):int(y1), int(x0):int(x1)] >= 1).all() and busy0.max() <= 1
    assert [o["role"] for o in fr["objects"][-2:]] == ["hero", "hero_head"]
    hb = fr["objects"][-1]["box"]; assert hb[3] < y1 and hb[1] == y0        # голова — верх силуэта
    changed = np.abs(fr["final_rgb"].astype(int) - final0.astype(int)).max(2) > 8
    ys, xs = np.nonzero(changed)                                     # кот в покое впечатан в источник занятости
    assert changed.sum() > 10000 and xs.min() >= x0 - 2 and xs.max() <= x1 + 2 and ys.min() >= y0 - 2


@needs_rig
def test_key_thought_avoids_the_doll(tmp_path):
    import frame_clip
    p = frame_clip.plan_clip(_fr(tmp_path), 6.0, _words(), key="пять минут", fps=24, mascot=dict(text="x", seed=0))
    assert p.get("mascot") and p.get("key_time") is not None
    cx, cy = p["key"]["center"]; x0, y0, x1, y1 = p["mascot"]["box"]
    # центр мысли — в экранных координатах окна записи, кукла — в холсте: переводим
    win = p["key"]["win"]; s = 1920 / (win[2] - win[0])
    kx, ky = win[0] + cx / s, win[1] + cy / s
    assert not (x0 <= kx <= x1 and y0 <= ky <= y1)
    assert "sig" in p["mascot"] and p["mascot"]["actions"]


@needs_rig
def test_composite_is_alpha_over_paper_and_cached():
    import canvas
    R = M.rig(); ms = dict(origin=[0.0, 0.0], scale=1.0, actions=[], seed=0)
    L = M.Layer(ms, 3.0)
    W, H = 1920, 1080
    f = np.empty((H, W, 3), np.float32); f[:] = canvas.CREAM
    win = (0, 0, W, H)
    g = L.composite(f.copy(), 0.5, win, W, H)
    a = L._warped(0.5, win, W, H)[..., 3]
    assert np.allclose(g[a == 0], canvas.CREAM)                     # вне кота бумага нетронута
    assert g[a > 0.98].mean() < 120                                  # мех тёмный
    # сведение через альфу совпадает с собственным кадром рига на той же бумаге (тот путь уже
    # проверен на ореол: doll/paper_cmp.jpg) — то есть ореола нет и здесь
    R.set_paper(canvas.CREAM); ref = R.frame(L.state(0.5), 0.5).astype(int)
    assert np.abs(g[:R.H, :R.W].round().astype(int) - ref).max() <= 2
    g2 = L.composite(f.copy(), 0.5, win, W, H)                       # повторный вызов (кадр пересчитан) — из кэша
    assert np.array_equal(g, g2)


@needs_rig
def test_rgba_matches_frame_on_cream_paper():
    import canvas
    R = M.rig(); st = M.rest_state(); st["look"] = (0.4, -0.2); st["head"] = 4.0; st["tail"] = 3.0
    rg = R.frame_rgba(st, 0.5)
    comp = np.clip(rg[..., :3] + canvas.CREAM * (1 - rg[..., 3:4]), 0, 255).round().astype(int)
    R.set_paper(canvas.CREAM); ref = R.frame(st, 0.5).astype(int)
    assert np.abs(comp - ref).max() <= 1


@needs_rig
def test_flame_state_scales_only_the_flame():
    """Состояние огонька из плана: bright/None — кадр как нарисован; ember — ниже, golden — выше;
    хвост под основанием пламени не меняется ни на пиксель."""
    import cv2
    R = M.rig(); st = M.rest_state(); X0, Y0, X1, Y1 = R.flame_geo["box"]; F = R.flame_state
    base = R.cat_layer(dict(st), 0.5)
    assert np.array_equal(R.cat_layer(dict(st, flame="bright"), 0.5), base)
    assert np.array_equal(R.cat_layer(dict(st, flame="nonsense"), 0.5), base)

    def top_row(c):
        reg = c[Y0:Y1, X0:X1]; hsv = cv2.cvtColor(np.clip(reg[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        core = (hsv[..., 1] > 110) & (hsv[..., 2] > 210) & (reg[..., 3] > 100)
        return int(np.nonzero(core.any(1))[0].min())
    tops = {fs: top_row(R.cat_layer(dict(st, flame=fs), 0.5)) for fs in (None, "ember", "golden")}
    assert tops["golden"] < tops[None] < tops["ember"]
    yb = F["yb"][:, 0]
    for fs in ("ember", "golden"):
        d = np.abs(R.cat_layer(dict(st, flame=fs), 0.5) - base).max(2)
        assert (d[Y0:Y1, X0:X1][yb >= F["base"] + 4] <= 1).all()          # хвост не тронут
        assert (d[:Y0] <= 0).all() and (d[Y1:] <= 0).all()                  # вне рамки — ничего
    ms = dict(origin=[0.0, 0.0], scale=1.0, actions=[], seed=0, state="ember")
    assert M.Layer(ms, 2.0).flame == "ember" and M.Layer(dict(ms, state="bright"), 2.0).flame is None


@needs_rig
def test_doll_shrinks_beside_a_wide_subject_instead_of_vanishing(tmp_path):
    """Живой кадр 3 эп.01: конверт в треть ширины холста — на 52% кукле не хватало пикселей,
    и кадр шёл без неё. Теперь уменьшается ступенями, место слева или справа находится."""
    import canvas
    import frame_clip
    src = tmp_path / "raw.png"
    im = Image.new("RGB", (1264, 848), (251, 251, 246))
    ImageDraw.Draw(im).rectangle((190, 301, 708, 695), outline=(20, 20, 20), width=8)
    im.save(src)
    objs = [dict(name="a letter", box=[190, 301, 708, 695], word=None, role="subject")]
    fr = frame_clip.prepare(str(src), [], objs, str(tmp_path), tex=canvas.paper_texture(w=1152, h=648))
    ms = M.place(fr, dict(text="x"))
    assert ms and ms["share"] < M.HEIGHT_SHARE and ms["share"] in M.SHRINK_STEPS


@needs_rig
def test_look_match_grain_and_shadow():
    """Кукла в кадре: насыщенность подгоняется к нарисованным котам (только вниз), зерно бумаги кадра
    ложится на куклу, под лапами — мягкая тень в мире кадра, вне тени мир не тронут."""
    import canvas
    import cv2
    W, H = 1920, 1080
    ms = dict(origin=[0.0, 0.0], scale=1.0, actions=[], seed=0, box=[266, 109, 1141, 770])
    f = np.empty((H, W, 3), np.float32); f[:] = canvas.CREAM
    L1 = M.Layer(dict(ms, look={"sat": 1.0}), 2.0); L2 = M.Layer(dict(ms, look={"sat": 0.6}), 2.0)
    g1 = L1.composite(f.copy(), 0.5, (0, 0, W, H), W, H); g2 = L2.composite(f.copy(), 0.5, (0, 0, W, H), W, H)
    a = L1._warped(0.5, (0, 0, W, H), W, H)[..., 3] > 0.98
    s1 = cv2.cvtColor(g1.astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 1][a].mean()
    s2 = cv2.cvtColor(g2.astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 1][a].mean()
    assert 0.5 < s2 / s1 < 0.75                                     # насыщенность снижена примерно в 0.6
    # зерно: тёмная бумага под куклой (0.95) темнит и куклу, в пределах GRAIN_CLIP
    f2 = f * 0.95; g3 = L1.composite(f2.copy(), 0.5, (0, 0, W, H), W, H)
    assert 0.93 < (g3[a].mean() / g1[a].mean()) < 0.97
    f3 = f * 0.5; g4 = L1.composite(f3.copy(), 0.5, (0, 0, W, H), W, H)   # линия под куклой — не сильнее зерна
    assert g4[a].mean() / g1[a].mean() >= M.GRAIN_CLIP[0] - 0.01   # полупрозрачные края добавляют ~0.5%
    world = np.full((H, W, 3), 250, np.uint8)
    out = M.bake_shadow(world, ms, W, H, 1)
    fx, fy = int((ms["box"][0] + ms["box"][2]) / 2), int(ms["box"][3]) - 5
    assert out[fy, fx].mean() < 225 and np.array_equal(out[:50], world[:50]) and out.dtype == np.uint8


def test_episode_look_without_drawn_cats_is_identity(tmp_path):
    assert M.episode_look(str(tmp_path), [{"status": "ok", "path": "x.png", "hero": False, "objects": []}])["sat"] == 1.0
