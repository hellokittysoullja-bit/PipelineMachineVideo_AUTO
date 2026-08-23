"""Юнит-тесты scripts/lookbook_remeasure.py — миграция/ре-калибровка
graded_lab_mean существующих lookbook-записей. Каждый тест monkeypatch'ит
look_reference.LOOKBOOK_PATH на tmp_path, ни один не должен коснуться
реального assets/lookbook/lookbook.json."""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["lookbook_remeasure.py", tempfile.gettempdir()]
import lookbook_remeasure as lrm   # noqa: E402
import look_reference as lr        # noqa: E402


def _make_photo(path, rgb=(120, 130, 140)):
    w, h = 64, 64
    arr = (np.ones((h, w, 3), dtype=np.float32) * np.array(rgb, dtype=np.float32)).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


@pytest.fixture(autouse=True)
def _isolated_lookbook(tmp_path, monkeypatch):
    lookbook_path = tmp_path / "lookbook" / "lookbook.json"
    lookbook_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(lookbook_path))
    return lookbook_path


def _write_lookbook(path, references):
    path.write_text(json.dumps({"references": references, "channel_id": lr.CHANNEL_ID}), encoding="utf-8")


def test_pregraded_entry_never_touched(tmp_path, monkeypatch, _isolated_lookbook):
    photo = tmp_path / "p.png"
    _make_photo(str(photo))
    ref = {
        "id": "pg", "domain": "snow",
        "image": os.path.relpath(str(photo), REPO_ROOT),
        "lab_mean": [50.0, 1.0, -2.0],
        "graded_lab_mean": {"HOOK": [50.0, 1.0, -2.0], "BODY": [50.0, 1.0, -2.0], "FINAL": [50.0, 1.0, -2.0]},
        "graded_recipe_fingerprint": "stale_on_purpose",
        "pregraded": True,
    }
    _write_lookbook(_isolated_lookbook, [ref])
    monkeypatch.setattr(lrm, "_real_argv", ["lookbook_remeasure.py", "--force"])
    assert lrm.main() == 0
    data = json.loads(_isolated_lookbook.read_text(encoding="utf-8"))
    stored = data["references"][0]
    # --force игнорируется для pregraded — fingerprint/graded_lab_mean не тронуты,
    # даже несмотря на явно устаревший fingerprint и --force флаг.
    assert stored["graded_recipe_fingerprint"] == "stale_on_purpose"
    assert stored["graded_lab_mean"] == ref["graded_lab_mean"]


def test_non_pregraded_entry_still_gets_remeasured(tmp_path, monkeypatch, _isolated_lookbook):
    photo = tmp_path / "p.png"
    _make_photo(str(photo))
    ref = {
        "id": "normal", "domain": "snow",
        "image": os.path.relpath(str(photo), REPO_ROOT),
        "lab_mean": [50.0, 1.0, -2.0],
        "graded_recipe_fingerprint": "stale_on_purpose",
        "pregraded": False,
    }
    _write_lookbook(_isolated_lookbook, [ref])
    monkeypatch.setattr(lrm, "_real_argv", ["lookbook_remeasure.py"])
    assert lrm.main() == 0
    data = json.loads(_isolated_lookbook.read_text(encoding="utf-8"))
    stored = data["references"][0]
    assert stored["graded_recipe_fingerprint"] == lr._grade_recipe_fingerprint()
    assert set(stored["graded_lab_mean"].keys()) == set(lr.GRADE_REFERENCE_SECTIONS)


def test_empty_lookbook_returns_zero(tmp_path, monkeypatch, _isolated_lookbook):
    _write_lookbook(_isolated_lookbook, [])
    monkeypatch.setattr(lrm, "_real_argv", ["lookbook_remeasure.py"])
    assert lrm.main() == 0
