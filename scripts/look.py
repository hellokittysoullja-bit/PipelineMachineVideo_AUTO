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
    def __init__(self, style, hero):
        self.style, self.hero = style, hero

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
    return Look(style, heroes[0] if heroes else None)
