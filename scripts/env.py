#!/usr/bin/env python3
"""Корень репозитория и загрузка .env. Больше ничего: стиль и герой —
картинки в look/ (look.py), а не значения в коде."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
