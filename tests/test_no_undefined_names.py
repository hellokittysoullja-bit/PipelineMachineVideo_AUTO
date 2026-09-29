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


def test_every_flag_read_in_code_is_registered():
    """29.09: gpu_render_workers() читала флаг CASCADE_MODEL, которого в
    реестре GPU-ветки нет: KeyError при создании пула карты, на поде. Строка
    стояла за условием «есть видеокарта», тесты в контейнере её не
    выполняли. Имя флага в вызове реестра проверяется без выполнения."""
    import re
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import feature_flags
    known = set(feature_flags.FLAGS)
    bad = []
    for path in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py"))):
        src = open(path, encoding="utf-8").read()
        for m in re.finditer(r"feature_flags\.(?:enabled|mode|value)\(\s*[\"']([A-Z0-9_]+)[\"']", src):
            if m.group(1) not in known:
                bad.append(f"{os.path.basename(path)}: {m.group(1)}")
    assert not bad, "флаги вне реестра:\n" + "\n".join(bad)
