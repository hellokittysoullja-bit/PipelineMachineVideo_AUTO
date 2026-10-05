#!/usr/bin/env python3
"""Облик ролика — картинки, а не текст в коде.

  look/style/   образцы стиля: 1-3 кадра, нарисованных так, как должен
                выглядеть ролик. Уходят в модель картинок на КАЖДОМ кадре.
  look/hero.*   главный герой (png/jpg/webp), необязательно. Уходит в модель
                только на кадрах, где планировщик поставил героя.

Папка меняется переменной LOOK_DIR (своя на канал). Значений по умолчанию
нет: без образцов стиля генерация не начинается — иначе модель рисует свой
стиль, и ролик выходит не тем."""
import base64
import glob
import hashlib
import json
import io
import os

import env

EXTS = (".png", ".jpg", ".jpeg", ".webp")
MAX_STYLE_REFS = 3      # больше — модель хуже держит героя и сам кадр
REF_SIDE = 1024         # длинная сторона референса: стиль и лицо читаются, запрос лёгкий


class LookError(Exception):
    pass


def _images(pattern):
    return sorted(p for p in glob.glob(pattern) if p.lower().endswith(EXTS))


def data_url(path, side=REF_SIDE):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        im = im.convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Look:
    def __init__(self, style, hero, hero_text=None, hero_states=None, hero_marks=None, palette=None):
        self.style, self.hero = style, hero
        # look/palette.txt — палитра бренда в задании каждого кадра (критик 05.10: крафт-конверт и
        # деревянная комната спорили с правилом брендбука «тёплое — только кот и огонёк»)
        self.palette = palette
        # look/hero_marks.txt — приметы героя, которые модель путает при перерисовке
        # (критик 05.10: свёрнутое ушко прыгало с одной стороны на другую). Сторона
        # пишется от зрителя: «левое ухо» читается двояко. Уходит в задание каждого кадра с героем.
        self.hero_marks = hero_marks
        # look/hero_states.json — состояния героя, которые план выбирает по смыслу
        # фразы, а генератор дописывает в задание: {"имя": {"when", "draw"}}.
        # У кота — огонёк на хвосте: тлеет, когда «застрял», горит, когда сделал шаг.
        self.hero_states = hero_states or {}
        # look/hero.txt — кто герой, коротко («a black cartoon cat»): длинное
        # описание судья проверял бы по приметам (огонёк хвоста не виден → «нет»). Нужно судье: план пишет
        # пункты проверки про «a person», и рисунок героя-не-человека (кот)
        # иначе отклоняется как «не человек».
        self.hero_text = hero_text

    def refs(self, with_hero):
        """Референсы кадра по порядку: сначала образцы стиля, герой последним
        (промпт ссылается на них в этом порядке)."""
        paths = list(self.style) + ([self.hero] if with_hero and self.hero else [])
        return [data_url(p) for p in paths]

    def signature(self, with_hero):
        """Отпечаток того, что уходит в модель: образцы стиля, герой — только
        для кадров с героем (смена героя не перерисовывает кадры без него)."""
        h = hashlib.sha256()
        for p in list(self.style) + ([self.hero] if with_hero and self.hero else []):
            h.update(os.path.basename(p).encode())
            with open(p, "rb") as f:
                h.update(f.read())
        return h.hexdigest()[:16]


def load(look_dir=None):
    d = look_dir or os.environ.get("LOOK_DIR") or os.path.join(env.ROOT, "look")
    style = _images(os.path.join(d, "style", "*"))
    if not style:
        raise LookError(f"Нет образцов стиля в {os.path.join(d, 'style')}: положите 1-{MAX_STYLE_REFS} "
                        "кадра в нужном стиле (png/jpg/webp).")
    if len(style) > MAX_STYLE_REFS:
        raise LookError(f"В {os.path.join(d, 'style')} {len(style)} образцов, нужно не больше "
                        f"{MAX_STYLE_REFS}: оставьте самые характерные.")
    heroes = _images(os.path.join(d, "hero.*"))
    if len(heroes) > 1:
        raise LookError(f"Героев несколько ({', '.join(map(os.path.basename, heroes))}): оставьте один hero.*")
    hero = heroes[0] if heroes else None
    hero_text = None
    tp = os.path.join(d, "hero.txt")
    if hero and os.path.exists(tp):
        hero_text = " ".join(open(tp, encoding="utf-8").read().split()) or None
    states = {}
    sp = os.path.join(d, "hero_states.json")
    if hero and os.path.exists(sp):
        try:
            raw = json.load(open(sp, encoding="utf-8"))
        except ValueError as e:
            raise LookError(f"{sp}: не JSON ({e})")
        for k, v in raw.items():
            if not (isinstance(v, dict) and str(v.get("when", "")).strip() and str(v.get("draw", "")).strip()):
                raise LookError(f"{sp}: у состояния «{k}» нужны непустые when и draw")
            states[str(k)] = {"when": " ".join(str(v["when"]).split()), "draw": " ".join(str(v["draw"]).split())}
    marks = None
    mp = os.path.join(d, "hero_marks.txt")
    if hero and os.path.exists(mp):
        marks = " ".join(open(mp, encoding="utf-8").read().split()) or None
    palette = None
    pp = os.path.join(d, "palette.txt")
    if os.path.exists(pp):
        palette = " ".join(open(pp, encoding="utf-8").read().split()) or None
    return Look(style, hero, hero_text, states, marks, palette)
