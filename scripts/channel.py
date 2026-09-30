#!/usr/bin/env python3
"""Профиль канала (channel_profile.json в корне репо) и загрузка .env.

Всё «про канал» — ниша, герой, стиль рисунка, модель картинок — живёт в
профиле, а не в коде: клон под другую нишу правит один файл."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH = os.path.join(ROOT, "channel_profile.json")

DEFAULTS = {
    "niche": "psychology and everyday history explained simply: how people lived, "
             "what children did in ancient times, why our brain works the way it does",
    "mascot": {
        "name": "Герой",
        "description": "a cheerful stick figure with a big round white head, simple dot eyes, "
                       "a wide smile and a small tuft of three hairs on top, thin black stick "
                       "arms and legs, wearing a long mustard-yellow scarf",
        "reference_image": "assets/mascot/mascot.png",
    },
    "style": {
        "base": "simple hand-drawn doodle illustration, stick figures with round heads and "
                "thin black marker lines, flat colors, a few red accent arrows, clean composition, "
                "16:9 frame",
        "backdrops": {
            "white": "plain white background",
            "paper": "warm beige parchment paper background",
            "painted": "loosely painted earthy background in ochre, olive and brown tones",
        },
        "lettering": "hand-lettered black uppercase marker letters",
    },
    "image": {"size": "1536x1024", "quality": None},
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_profile(path=PROFILE_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return _merge(DEFAULTS, json.load(f))
    except FileNotFoundError:
        return json.loads(json.dumps(DEFAULTS))


def load_env(path=None):
    """.env -> os.environ (без перезаписи уже заданного). python-dotenv, если
    есть; иначе простой разбор KEY=VALUE."""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
