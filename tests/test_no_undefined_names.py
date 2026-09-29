# -*- coding: utf-8 -*-
"""Ни одного неопределённого имени в scripts/.

Зачем. 29.09 вызов GPU-рендера по ошибке встал и в video_render(), где нет
ни photo, ни z/x/y, ни геометрии холста. Строка стояла за условием «есть
видеокарта», поэтому ни один тест в контейнере без карты её не выполнил, и
на арендованном поде каждый видео-клип упал бы с NameError. Такие строки
находятся без выполнения — статическим разбором; этот тест и есть он.
"""
import glob
import os

import pytest

pyflakes_api = pytest.importorskip("pyflakes.api")
from pyflakes import reporter as _reporter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Collect(_reporter.Reporter):
    def __init__(self):
        self.found = []

    def unexpectedError(self, filename, msg):  # noqa: N802 — API pyflakes
        self.found.append(f"{filename}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text):  # noqa: N802
        self.found.append(f"{filename}:{lineno}: синтаксис: {msg}")

    def flake(self, message):
        if "undefined name" in str(message):
            self.found.append(str(message))


def test_scripts_have_no_undefined_names():
    rep = _Collect()
    for path in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py"))):
        pyflakes_api.checkPath(path, rep)
    assert not rep.found, "неопределённые имена:\n" + "\n".join(rep.found)
