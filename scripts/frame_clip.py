#!/usr/bin/env python3
"""Клип одного кадра: рисунок на кремовой бумаге, подписи по словам голоса,
быстрый наезд на предмет, главная мысль карандашом, камера по плану shots.py.

Мир кадра — холст 16:9, увеличенный в UPSCALE раз (upscale.py) с впечённой
фактурой бумаги. Подписи схемы и дописанная главная мысль впекаются в мир в
момент появления — дальше двигаются вместе с рисунком, как нарисованные.
Экранный кадр — окно камеры, вырезанное из мира одним Lanczos-ресемплом.
Пока главная мысль пишется, её чернила живут в экранных координатах того
окна, где начали писать, и переносятся в текущее окно аффинным переносом
(камера в это время только дрейфует — план запрещает склейки и наезды).

prepare() — дорогая часть, общая для плана и рендера (холст, увеличение,
карта занятости); plan_clip() — план с учётом наезда в прошлом кадре
(последовательно по ролику); render() — клип в mp4 (параллельно)."""
import hashlib
import json
import os
import subprocess

import numpy as np
from PIL import Image

import camera
import canvas
import labels as labels_mod
import placement
import shots
import upscale
import writeon

W, H = 1920, 1080
UPSCALE = 2
KEY_SIZES = (200, 175, 150, 125, 110)   # крупно, если есть место (критики: «мелкая, прижата к краю»)
KEY_MAX_W = 0.62
INK = np.array([52, 50, 56], np.float32)       # тёплый графит (вариант A владельца)
END_FADE_SEC = 0.6
DIM = 0.3                # схема до сборки — 30% силы, остальное бумага (критик монтажа 04.10)
REVEAL_SEC = 0.35        # часть схемы проявляется за столько
REVEAL_MAX_SHARE = 0.25  # общий прямоугольник «подпись + часть» не больше четверти холста
RENDER_VERSION = 1


def _hash(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def prepare(src_path, recs, objects, work, seed=0, tex=None):
    """Холст и мир кадра. recs — подписи генератора (labels_placed) в координатах
    src_path; пусто — подписей нет или они уже на картинке. objects — предметы
    [{name, box, word, role}] в координатах src_path. Возвращает dict."""
    img = Image.open(src_path).convert("RGB")
    if recs:
        img = labels_mod.flatten_paper(img, [r["box"] for r in recs])
    cv, off, on_paper = canvas.prepare(img)
    SH, SW = cv.shape[:2]
    ox, oy = off

    def shift(b):
        x0, y0, x1, y1 = b
        return (float(np.clip(x0 + ox, 0, SW)), float(np.clip(y0 + oy, 0, SH)),
                float(np.clip(x1 + ox, 0, SW)), float(np.clip(y1 + oy, 0, SH)))
    objs = []
    for o in objects or []:
        if o.get("box"):
            b = shift(o["box"])
            if b[2] - b[0] > 4 and b[3] - b[1] > 4:
                objs.append({**o, "box": b})
    world = upscale.upscale(cv, UPSCALE, cache_dir=os.path.join(work, "upscale"))
    if tex is None:
        tex = canvas.paper_texture()
    world = canvas.bake_texture(world, tex, seed=seed)
    final = Image.fromarray(cv)                       # холст со всеми подписями — для карты занятости
    for r in recs:
        labels_mod.draw_rec(final, r, 1.0, off)
    busy = placement.busy_map(np.asarray(final, np.float32), margin=8)
    # части схемы по подписям: рамка части от судьи, иначе — сама подпись с её стрелкой-указателем
    parts, have = [], 0
    for k, r in enumerate(recs):
        lb = shift(r["box"])
        pb = next((o["box"] for o in objs if o.get("role") == f"label:{k}"), None)
        have += pb is not None
        parts.append(dict(label=lb, part=pb))
    assemble = len(recs) >= 2 and have*2 >= len(recs)
    return dict(world=world, canvas=cv, off=off, on_paper=on_paper, SW=SW, SH=SH, recs=recs,
                objects=objs, busy=busy, final_rgb=np.asarray(final), parts=parts, assemble=assemble)


def reveal_mask(fr, i, scale):
    """Мягкая маска (в мире) того, что проявляется на слове подписи i: подпись со
    стрелкой-указателем и часть, которую она описывает."""
    from scipy import ndimage
    SW, SH = fr["SW"], fr["SH"]
    p = fr["parts"][i]
    lb = p["label"]
    lh = lb[3] - lb[1]
    rects = [(lb[0] - 0.8*lh, lb[1] - 0.8*lh, lb[2] + 0.8*lh, lb[3] + 0.8*lh)]
    if p["part"]:
        pb = p["part"]
        mx, my = 0.1*(pb[2] - pb[0]), 0.1*(pb[3] - pb[1])
        rects.append((pb[0] - mx, pb[1] - my, pb[2] + mx, pb[3] + my))
        u = (min(lb[0], pb[0]), min(lb[1], pb[1]), max(lb[2], pb[2]), max(lb[3], pb[3]))
        if (u[2] - u[0])*(u[3] - u[1]) <= REVEAL_MAX_SHARE*SW*SH:
            rects.append(u)
    m = np.zeros((SH*scale, SW*scale), np.float32)
    for x0, y0, x1, y1 in rects:
        m[max(0, int(y0*scale)):max(0, int(y1*scale)), max(0, int(x0*scale)):max(0, int(x1*scale))] = 1
    return np.clip(ndimage.gaussian_filter(m, 10*scale), 0, 1)[..., None]


def _screen(world, win, scale):
    x0, y0, x1, y1 = win
    return np.asarray(Image.fromarray(world).resize((W, H), Image.LANCZOS,
                                                    box=(x0*scale, y0*scale, x1*scale, y1*scale)), np.float32)


def key_layout(fr, text, win, seed=9):
    """Где и каким кеглем писать главную мысль в окне win. (letters, (cx, cy), кегль) или None."""
    scr = _screen(np.asarray(Image.fromarray(fr["final_rgb"]).resize(
        (fr["SW"]*UPSCALE, fr["SH"]*UPSCALE), Image.BILINEAR)), win, UPSCALE)

    def wh(sz):
        return writeon.measure(text, sz, KEY_MAX_W*W)
    # рядом со своим предметом (критики 04.10: надпись «висела в пустоте» отдельно от сцены)
    anchor = next((o["box"] for o in fr.get("objects", []) if o.get("role") == "key_anchor"), None)
    obj = None
    if anchor:
        s_ = W/(win[2] - win[0])
        obj = ((anchor[0] - win[0])*s_, (anchor[1] - win[1])*s_, (anchor[2] - win[0])*s_, (anchor[3] - win[1])*s_)
        if obj[2] < 0 or obj[3] < 0 or obj[0] > W or obj[1] > H:
            obj = None
    size, pl = placement.choose_fit(scr, obj, wh, KEY_SIZES)
    if not pl and obj is not None:
        size, pl = placement.choose_fit(scr, None, wh, KEY_SIZES)
    if not pl:
        return None
    cx, cy = pl["center"]
    return writeon.layout(text, size, W, H, cx, cy, seed=seed, max_w=KEY_MAX_W*W), (cx, cy), size


KEY_SPEEDUPS = (1.0, 1.25, 1.5)   # не успевает дописаться и постоять — карандаш быстрее, но не больше чем в 1.5 раза


def plan_clip(fr, D, words, key=None, last_punch=-1e9, T0=0.0, zoom_in=True, fps=24):
    """План кадра (shots.plan) + раскладка главной мысли. Главная мысль, которой
    нет места, не пишется — с записью в notes. Не успевает — пишется быстрее
    (KEY_SPEEDUPS), а не пропадает: живой ролик 04.10 потерял «только открыть»,
    потому что фраза стоит в конце кадра."""
    key_l = None
    if key:
        wide = camera.window(fr["SW"]/2, fr["SH"]/2, fr["SW"], fr["SW"], fr["SH"])
        key_l = key_layout(fr, key, wide)
    parts = [pp["part"] or pp["label"] for pp in fr.get("parts", [])]
    p, speed = None, 1.0
    for speed in (KEY_SPEEDUPS if key_l else (1.0,)):
        key_dur = writeon.plan(key_l[0], fps, factor=speed)[1] if key_l else 0.0
        p = shots.plan(D, fr["busy"], words, fr["recs"], fr["objects"], key if key_l else None, key_dur,
                       last_punch=last_punch, T0=T0, zoom_in=zoom_in, parts=parts)
        if not key_l or p["key_time"] is not None:
            break
    p["assemble"] = bool(fr.get("assemble"))
    if key and not key_l:
        p["notes"].append(f"главной мысли «{key}» нет места на кадре — не пишется")
    if key_l and p["key_time"] is not None:
        if speed > 1.0:
            p["notes"] = [n for n in p["notes"] if "главная мысль" not in n]
            p["notes"].append(f"главная мысль пишется в {speed:.2f} раза быстрее, чтобы успеть и постоять")
        # писать в окне, где камера будет в этот момент (план держит его без склеек)
        win = shots.window_at(p, p["key_time"], fr["SW"], fr["SH"])
        kl = key_layout(fr, key, win)
        if kl and writeon.plan(kl[0], fps, factor=speed)[1] <= key_dur*1.05 + 0.1:
            p["key"] = dict(text=key, win=win, center=kl[1], size=kl[2], speed=speed)
        else:
            p["notes"].append(f"главной мысли «{key}» нет места в плане момента — не пишется")
            p["key_time"] = None
    return p


def _warp(alpha, src_win, dst_win):
    """Альфа в экранных координатах окна src_win -> экранные координаты окна dst_win."""
    s_src = W/(src_win[2] - src_win[0]); s_dst = W/(dst_win[2] - dst_win[0])
    # экранная точка dst -> мир -> экранная точка src
    a = s_src/s_dst
    c = (dst_win[0] - src_win[0])*s_src
    f = (dst_win[1] - src_win[1])*s_src
    im = Image.fromarray(np.clip(alpha*255, 0, 255).astype(np.uint8))
    return np.asarray(im.transform((W, H), Image.AFFINE, (a, 0, c, 0, a, f), Image.BILINEAR), np.float32)/255


def _bake_alpha(world, alpha, win, scale, color):
    """Впечь экранную альфу окна win в мир (масштаб scale)."""
    x0, y0, x1, y1 = [v*scale for v in win]
    bw, bh = int(round(x1 - x0)), int(round(y1 - y0))
    a = np.asarray(Image.fromarray(np.clip(alpha*255, 0, 255).astype(np.uint8)).resize((bw, bh), Image.LANCZOS),
                   np.float32)/255
    X0, Y0 = int(round(x0)), int(round(y0))
    ys = slice(max(0, Y0), min(world.shape[0], Y0 + bh)); xs = slice(max(0, X0), min(world.shape[1], X0 + bw))
    aa = a[ys.start - Y0:ys.stop - Y0, xs.start - X0:xs.stop - X0][..., None]
    sub = world[ys, xs].astype(np.float32)
    world[ys, xs] = np.clip(sub*(1 - aa) + color*aa, 0, 255).round().astype(np.uint8)


def render(fr, p, D, out, fps=24, end_fade=False, crf="18"):
    """Клип в out. Возвращает [(t0, t1, kind)] штрихов карандаша (от начала клипа)."""
    SW, SH = fr["SW"], fr["SH"]
    full = fr["world"].copy()                  # рисунок целиком (и подписи, когда появятся)
    assemble = p.get("assemble") and len(fr["recs"]) >= 2
    if assemble:
        world = (full.astype(np.float32)*DIM + canvas.CREAM*(1 - DIM)).round().astype(np.uint8)
    else:
        world = full
    n = max(1, int(round(D*fps)))
    lab_t = list(p["label_times"])
    shown = [False]*len(fr["recs"])
    ink, key, cues, baked = None, p.get("key"), [], False
    if key and p.get("key_time") is not None:
        L = writeon.layout(key["text"], key["size"], W, H, key["center"][0], key["center"][1], seed=9,
                           max_w=KEY_MAX_W*W)
        ev, T = writeon.plan(L, fps, t0=p["key_time"], factor=key.get("speed", 1.0))
        ink = writeon.Ink(L, ev, H, W)
        cues = [(e["t0"], e["t1"], e["kind"]) for e in ev]
        key_end = p["key_time"] + T
    prev_world, fade_from, fade_len = None, None, shots.LABEL_FADE_SEC
    tmp = out + ".tmp.mp4"
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(fps), "-i", "-", "-frames:v", str(n), "-c:v", "libx264", "-preset", "medium",
                             "-crf", crf, "-pix_fmt", "yuv420p", "-r", str(fps), tmp], stdin=subprocess.PIPE)
    try:
        for k in range(n):
            t = k/fps
            for i, lt in enumerate(lab_t):
                if not shown[i] and t >= lt:
                    prev_world, fade_from = world.copy(), lt
                    off2 = (fr["off"][0]*UPSCALE, fr["off"][1]*UPSCALE)
                    if assemble:
                        big = Image.fromarray(full)
                        labels_mod.draw_rec(big, fr["recs"][i], UPSCALE, off2)
                        full = np.asarray(big).copy()
                        last = all(shown[j] or j == i for j in range(len(shown)))
                        m = 1.0 if last else reveal_mask(fr, i, UPSCALE)   # последняя — схема целиком
                        world = (world*(1 - m) + full*m).round().astype(np.uint8) if not last else full.copy()
                        fade_len = REVEAL_SEC
                    else:
                        big = Image.fromarray(world)
                        labels_mod.draw_rec(big, fr["recs"][i], UPSCALE, off2)
                        world = np.asarray(big).copy()
                        fade_len = shots.LABEL_FADE_SEC
                    shown[i] = True
            win = shots.window_at(p, t, SW, SH)
            f = _screen(world, win, UPSCALE)
            if prev_world is not None and t - fade_from < fade_len:
                a = camera.ease_io((t - fade_from)/fade_len)
                f = _screen(prev_world, win, UPSCALE)*(1 - a) + f*a
            elif prev_world is not None:
                prev_world = None
            if ink is not None and not baked:
                ink.advance(t)
                al = ink.alpha()
                if t >= key_end + 1.0/fps:
                    _bake_alpha(world, al, key["win"], UPSCALE, INK)
                    if assemble and full is not world:
                        _bake_alpha(full, al, key["win"], UPSCALE, INK)   # иначе следующее проявление её сотрёт
                    baked = True
                    f = _screen(world, win, UPSCALE)
                else:
                    if al.any():
                        al = _warp(al, key["win"], win)
                        f = f*(1 - al[..., None]) + INK*al[..., None]
            if end_fade and t > D - END_FADE_SEC:
                a = camera.ease_io((t - (D - END_FADE_SEC))/END_FADE_SEC)
                f = f*(1 - a) + canvas.CREAM*a
            proc.stdin.write(np.clip(f, 0, 255).round().astype(np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg не записал клип")
    except Exception:
        proc.kill()
        raise
    os.replace(tmp, out)
    return cues


def plan_record(p):
    """План в JSON-виде для отчёта."""
    return json.loads(json.dumps({k: v for k, v in p.items()}, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
