#!/usr/bin/env python3
"""Сопоставление русских слов в разных формах — одно правило на план и сборку.

План проверяет, что слово наезда и слова главной мысли действительно есть в
фразе; сборка ищет момент, когда голос их произносит. Правило одно: если бы
план и сборка сравнивали слова по-разному, план пропускал бы слово, которого
сборка потом «не слышит», и наезд молча пропадал."""
import re


def norm(w):
    return re.sub(r"[^\wё-]", "", (w or "").lower()).replace("ё", "е").strip("-")


def same(a, b):
    """Одно слово в разных формах: общее начало не короче 4 букв и не короче длины без окончания."""
    if not a or not b:
        return False
    if a == b:
        return True
    k = max(4, min(len(a), len(b)) - 2)
    return len(a) >= 4 and len(b) >= 4 and a[:k] == b[:k]


def significant(phrase):
    toks = [norm(t) for t in (phrase or "").split()]
    return [t for t in toks if len(t) >= 3] or [t for t in toks if t]


def in_text(phrase, text):
    """Все значимые слова phrase встречаются в text (в любой форме)."""
    have = [norm(t) for t in (text or "").split()]
    want = significant(phrase)
    return bool(want) and all(any(same(w, h) for h in have) for w in want)
