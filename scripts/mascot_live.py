#!/usr/bin/env python3
"""Живая кукла героя в кадре сборщика (MASCOT_LIVE=0/1, дефолт 1; работает, только если на диске
есть собранный риг — `experiments/alive_cat/doll` или MASCOT_RIG_DIR).

Кукла — это сам референс героя (look/hero.png), разобранный на слои (doll_rig): моргание, взгляд,
наклон головы, ухо, хвост, огонёк, дыхание. В кадр она ставится как РИСУНОК НА БУМАГЕ холста:
живёт в координатах холста, и камера (наезды, дрейф) двигает её вместе с остальным рисунком.
Кладётся через премультиплицированную альфу на кремовую бумагу (без ореола — замер в doll/README).

Куда идёт кукла: только в кадр, где нарисованного героя НЕТ (сборщик даёт кадру куклу, когда
план кадра просит героя живьём, а картинка сгенерирована без него). Кадр с нарисованным героем
не трогается — у него кот уже вписан в сцену, и вторая копия рядом была бы хуже прежнего
(правило владельца 02.10: всё добавленное только лучше).

Если куклу некуда поставить (места рядом с предметом нет), кадр остаётся как есть — честный
откат, не кукла поверх рисунка. Нет рига на диске — слой не оставляет следа: ни в плане, ни в
ключе кэша клипа."""
import hashlib
import inspect
import json
import os
import sys

import numpy as np

import placement

HERE = os.path.dirname(os.path.abspath(__file__))
RIG_DIR = os.environ.get("MASCOT_RIG_DIR") or os.path.join(os.path.dirname(HERE), "experiments", "alive_cat", "doll")
HEIGHT_SHARE = 0.52      # рост кота по плотному силуэту — доля высоты холста (решение владельца 07.10)
SHRINK_STEPS = (0.46, 0.40)   # тесно рядом с крупным предметом — кукла меньше, а не никакой (живой кадр 3 эп.01:
                              # конверт в треть ширины, коту 52% не хватило 3 px до безопасной зоны)
GAP_SHARE = 0.03         # зазор между котом и предметом, доля ширины холста
MAX_BUSY = 0.12          # средняя занятость под силуэтом, выше — кукле тут не место
BUSY_MARGIN = 26         # поле вокруг куклы в карте занятости (как у busy_map)
GRAIN_CLIP = (0.93, 1.03)   # зерно бумаги кадра поверх куклы: 10–90-й перцентили фактуры 0.946–1.018 (замер эп.01)
SHADOW_K = 0.13             # тень под лапами: бумага 250 -> ~217 в центре; пол в кадре эп.01 — 233
SAT_MIN = 0.5               # насыщенность куклы подгоняется к нарисованным котам эпизода, но не ниже половины
VERSION = 2

_RIG = None
_MOD = None


def enabled():
    return os.environ.get("MASCOT_LIVE", "1") != "0"


def _asset_paths():
    rj = os.path.join(RIG_DIR, "rig.json")
    if not os.path.exists(rj):
        return None
    r = json.load(open(rj))
    paths = [rj]
    for k in ("image", "mask", "fur", "iris", "eyes", "eye_masks"):
        p = r[k] if os.path.isabs(r[k]) else os.path.join(RIG_DIR, r[k])
        if not os.path.exists(p):
            alt = os.path.splitext(p)[0] + ".npz"
            if not os.path.exists(alt):
                return None
            p = alt
        paths.append(p)
    return paths


def available():
    """Риг на диске целиком и зависимости импортируются."""
    if not enabled() or _asset_paths() is None:
        return False
    try:
        import cv2  # noqa: F401
        import scipy  # noqa: F401
    except Exception:
        return False
    return True


def signature():
    """Подпись рига и кода куклы — в план кадра, чтобы смена рига или движений перерендерила клип."""
    h = hashlib.sha256()
    for p in _asset_paths() or []:
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}:{st.st_size}".encode())
        if p.endswith(".json"):
            h.update(open(p, "rb").read())
    h.update(open(os.path.join(RIG_DIR, "doll_rig.py"), "rb").read())
    h.update(inspect.getsource(sys.modules[__name__]).encode())
    return h.hexdigest()[:12]


def _mod():
    global _MOD
    if _MOD is None:
        if RIG_DIR not in sys.path:
            sys.path.insert(0, RIG_DIR)
        import doll_rig
        _MOD = doll_rig
    return _MOD


def rig():
    global _RIG
    if _RIG is None:
        R = _mod().Rig(os.path.join(RIG_DIR, "rig.json"))
        R.defringe()
        ys, xs = np.nonzero(R.alpha > 8)
        R.tight = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        R.eye_center = (float(R.E[:, 0].mean()), float(R.E[:, 1].mean()))
        _RIG = R
    return _RIG


def place(fr, spec):
    """Поставить куклу на холст рядом с предметом фразы. Правит fr: карта занятости (подписи и
    мысль обходят куклу) и objects (камера держит героя в кадре, как нарисованного).
    Возвращает геометрию {origin, scale, box, gaze} или None — места нет."""
    import shots
    R = rig(); SW, SH = fr["SW"], fr["SH"]
    tx0, ty0, tx1, ty1 = R.tight
    subj = shots.subject_box(fr["busy"], fr["objects"])
    sx0, sy0, sx1, sy1 = subj if subj else (SW * 0.45, SH * 0.4, SW * 0.55, SH * 0.6)
    L, T, Rr, B = placement.SAFE
    gap = GAP_SHARE * SW
    for share in (HEIGHT_SHARE,) + tuple(SHRINK_STEPS):
        s = share * SH / (ty1 - ty0)
        cw, ch = (tx1 - tx0) * s, (ty1 - ty0) * s
        floor = float(np.clip(max(sy1 + 0.02 * SH, 0.62 * SH), ch + T * SH, B * SH))   # пол: лапы не ниже безопасной зоны
        y0 = floor - ch
        cands = []
        for side in ("left", "right"):
            x0 = sx0 - gap - cw if side == "left" else sx1 + gap
            x0 = float(np.clip(x0, L * SW, Rr * SW - cw))
            x1 = x0 + cw
            if (side == "left" and x1 > sx0 - gap * 0.5) or (side == "right" and x0 < sx1 + gap * 0.5):
                continue                                               # не поместился сбоку от предмета
            b = fr["busy"][int(y0):int(y0 + ch), int(x0):int(x1)]
            cands.append((float(b.mean()) if b.size else 1.0, side, x0))
        cands.sort()
        if cands and cands[0][0] <= MAX_BUSY:
            break
    else:
        return None
    busy_v, side, x0 = cands[0]
    ox, oy = x0 - tx0 * s, y0 - ty0 * s                                # холст = origin + s * риг
    box = [x0, y0, x0 + cw, y0 + ch]
    ex, ey = ox + R.eye_center[0] * s, oy + R.eye_center[1] * s
    cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    gaze = [float(np.clip((cx - ex) / (0.45 * SW), -1, 1)), float(np.clip((cy - ey) / (0.45 * SH), -1, 1))]
    m = BUSY_MARGIN
    fr["busy"][max(0, int(y0) - m):int(y0 + ch) + m, max(0, int(x0) - m):int(x0 + cw) + m] = 1.0
    hy1 = oy + R.r["head_below_y"] * s                                   # голова куклы — крупный план для монтажа
    head = (box[0], box[1], box[2], float(min(box[3], hy1)))
    fr["objects"] = list(fr["objects"]) + [dict(name="hero", box=tuple(box), word=None, role="hero"),
                                           dict(name="the head of the hero", box=head, word=None, role="hero_head")]
    # раскладка мысли и акцента считает занятость САМА — по final_rgb (placement.choose_fit), а не по
    # fr["busy"]; без кота в final_rgb мысль легла бы ему на голову (пилот 07.10: центр мысли (635,360)
    # внутри рамки кота). Кот в покое впечатывается в final_rgb на своё место.
    fr["final_rgb"] = paint_rest(fr["final_rgb"], R, (ox, oy), s)
    return dict(origin=[ox, oy], scale=s, box=box, gaze=gaze, side=side, busy=round(busy_v, 4), share=share)


def rest_state():
    return dict(look=(0.0, 0.0), lid=0.0, head=0.0, ear=0.0, tail=0.0, breath=1.0, paws={})


def paint_rest(rgb, R, origin, s):
    """Кот в покое поверх холста rgb (uint8) в координатах холста: origin + s * риг. Для карт занятости."""
    import cv2
    rgba = R.frame_rgba(rest_state(), 0.0)
    x0, y0, x1, y1 = R.bb
    sub = np.ascontiguousarray(rgba[y0:y1, x0:x1])
    nw, nh = max(1, int(round(sub.shape[1] * s))), max(1, int(round(sub.shape[0] * s)))
    small = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    H, W = rgb.shape[:2]
    px, py = int(round(origin[0] + s * x0)), int(round(origin[1] + s * y0))
    cx0, cy0 = max(0, px), max(0, py); cx1, cy1 = min(W, px + nw), min(H, py + nh)
    if cx1 <= cx0 or cy1 <= cy0:
        return rgb
    part = small[cy0 - py:cy1 - py, cx0 - px:cx1 - px]
    a = part[..., 3:4]
    out = np.array(rgb, np.float32, copy=True)
    out[cy0:cy1, cx0:cx1] = out[cy0:cy1, cx0:cx1] * (1 - a) + part[..., :3]
    return np.clip(out, 0, 255).round().astype(np.uint8)


def doll_saturation():
    """Средняя насыщенность (HSV S) плотных пикселей куклы в покое — эталон для подгонки к рисунку."""
    import cv2
    R = rig()
    if getattr(R, "_sat", None) is None:
        rg = R.frame_rgba(rest_state(), 0.0); a = rg[..., 3]
        rgb = np.clip(rg[..., :3] / np.maximum(a[..., None], 1e-3), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        R._sat = float(hsv[..., 1][a > 0.98].mean())
    return R._sat


def episode_look(video_dir, report_frames):
    """Подгонка облика куклы под котов, НАРИСОВАННЫХ моделью в этом же эпизоде: коэффициент
    насыщенности = их средняя S / S куклы (только вниз, не ниже SAT_MIN). Замер эп.01: кукла 52,
    нарисованные коты 27/32/45 — кукла читалась наклейкой. Нет нарисованных котов — 1.0 (как есть)."""
    import cv2
    vals = []
    for f in report_frames or []:
        if f.get("status") != "ok" or not f.get("path"):
            continue
        boxes = [o for o in (f.get("objects") or []) if o.get("role") in ("hero", "subject") and o.get("box")
                 and f.get("hero")]
        if not boxes:
            continue
        im = cv2.imread(os.path.join(video_dir, f["path"]))
        if im is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in boxes[0]["box"]]
        c = im[max(0, y0):y1, max(0, x0):x1]
        if c.size == 0:
            continue
        h = cv2.cvtColor(c, cv2.COLOR_BGR2HSV); m = h[..., 2] < 200          # без бумаги
        if m.sum() > 500:
            vals.append(float(h[..., 1][m].mean()))
    if not vals or not available():
        return {"sat": 1.0, "drawn_sat": vals}
    ds = doll_saturation()
    return {"sat": float(np.clip(np.mean(vals) / max(ds, 1e-3), SAT_MIN, 1.0)), "drawn_sat": vals, "doll_sat": ds}


def shadow_mask(ms, SW, SH, scale=1):
    """Мягкая тень-пятно под лапами (как у предметов в кадрах эпизода) в координатах холста × scale:
    float32 0..1, эллипс у низа силуэта, размыт."""
    import cv2
    x0, y0, x1, y1 = ms["box"]; cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
    H, W = int(SH * scale), int(SW * scale)
    m = np.zeros((H, W), np.float32)
    cx, cy = int((x0 + x1) / 2 * scale), int(y1 * scale - 0.015 * ch)
    cv2.ellipse(m, (cx, cy), (int(0.44 * cw), int(0.07 * ch)), 0, 0, 360, 1.0, -1)
    k = max(3, int(0.05 * cw) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def bake_shadow(world, ms, SW, SH, scale=1):
    """Тень под куклой впекается в мир кадра один раз (кукла двигается на месте, тень стоит)."""
    m = shadow_mask(ms, SW, SH, scale)[..., None]
    out = world.astype(np.float32) * (1 - SHADOW_K * m)
    return np.clip(out, 0, 255).round().astype(world.dtype)


def plan_actions(D, words, text, gaze, seed=0, action=None):
    """Движения из самой фразы: смотрит на предмет почти весь кадр (в конце — снова на зрителя),
    вопрос — наклон головы на слове с «?», вторая мысль во фразе — лёгкий наклон в другую
    сторону. action из плана кадра: "paw_on_chest" — поза «лапа у груди» на первые секунды
    (единственная поза в работе; жесты лап владелец отложил 07.10). Фон (моргание, оглядывание,
    ухо, хвост, дыхание) даёт doll_rig.plan сам."""
    acts = []
    if action == "paw_on_chest" and D >= 1.2:
        acts.append(dict(t=0.35, do="paw", dur=float(min(2.4, D - 0.8))))
    if D >= 1.6:
        hold = max(0.6, D - 1.1)
        acts.append(dict(t=0.25, do="look", x=gaze[0], y=gaze[1], dur=hold))
    q = [w for w in (words or []) if w["word"].rstrip(".,!…»)").endswith("?")]
    if q:
        acts.append(dict(t=max(0.0, q[0]["start"] - 0.1), do="tilt", deg=5, dur=1.0))
    elif words and len(words) >= 4:
        for i, w in enumerate(words[:-2]):
            if w["word"].rstrip("»)").endswith((".", "!", "…")) and words[i + 1]["start"] > 0.8:
                acts.append(dict(t=words[i + 1]["start"] - 0.05, do="tilt", deg=-3, dur=0.9))
                break
    return acts


class Layer:
    """Кукла в клипе: состояние по времени + сведение в экранный кадр через окно камеры."""

    def __init__(self, ms, D):
        import cv2
        self.cv2 = cv2
        self.R = rig()
        self.state = _mod().plan(D, ms["actions"], seed=int(ms.get("seed", 0)))
        self.ox, self.oy = ms["origin"]; self.s = float(ms["scale"])
        self.flame = ms.get("state") if ms.get("state") in getattr(_mod(), "FLAME_STATES", {}) else None
        self.sat = float((ms.get("look") or {}).get("sat", 1.0))
        import canvas
        self.cream = canvas.CREAM

    def composite(self, f, t, win, W, H):
        key = (round(t, 6), tuple(round(v, 4) for v in win), W, H)
        if getattr(self, "_last", None) and self._last[0] == key:
            warped = self._last[1]
        else:
            warped = self._warped(t, win, W, H)
            self._last = (key, warped)
        a = warped[..., 3:4]
        # зерно бумаги кадра ложится и на куклу (иначе гладкая кукла на зернистом рисунке — наклейка):
        # множитель = кадр / чистая бумага, в пределах амплитуды зерна, чтобы линии рисунка под куклой
        # не просвечивали сильнее зерна
        grain = np.clip((f / self.cream).mean(axis=2, keepdims=True), *GRAIN_CLIP)
        # бикубическая выборка рига даёт альфу чуть вне 0..1 (замер: −0.09..1.09) — на бумаге рига это
        # режется при записи кадра; здесь тоже режем сразу, чтобы карандаш и затемнения выше по цепочке
        # не получали значения вне 0..255 (без клипа край отличался от кадра рига до 26 уровней)
        return np.clip(f * (1 - a) + warped[..., :3] * grain, 0, 255)

    def _warped(self, t, win, W, H):
        cv2 = self.cv2; R = self.R
        st = self.state(t)
        if self.flame:
            st["flame"] = self.flame                     # ember/golden из плана кадра; bright — как нарисовано
        rgba = R.frame_rgba(st, t)
        x0, y0, x1, y1 = R.bb
        sub = np.ascontiguousarray(rgba[y0:y1, x0:x1])
        if self.sat < 0.999:                                             # к насыщенности нарисованных котов
            a4 = sub[..., 3:4]; a1 = np.maximum(a4, 1e-3)
            rgb = np.clip(sub[..., :3] / a1, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32); hsv[..., 1] *= self.sat
            rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            sub = np.concatenate([rgb * a4, a4], axis=2)
        k = self.s * W / (win[2] - win[0])                               # экранных px на px рига
        if k < 1:                                                        # уменьшение — площадью, без алиасинга
            nw, nh = max(1, int(round(sub.shape[1] * k))), max(1, int(round(sub.shape[0] * k)))
            small = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA)
            kx, ky = k * sub.shape[1] / nw, k * sub.shape[0] / nh         # остаточный масштаб ≈ 1
        else:
            small, kx, ky = sub, k, k
        sx = (self.ox + self.s * x0 - win[0]) * W / (win[2] - win[0])
        sy = (self.oy + self.s * y0 - win[1]) * W / (win[2] - win[0])
        M = np.array([[kx, 0, sx], [0, ky, sy]], np.float32)
        return cv2.warpAffine(small, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
