"""Юнит-тесты scripts/lookbook_add.py — единственный способ пополнить
assets/lookbook/lookbook.json. КАЖДЫЙ тест monkeypatch'ит look_reference.
LOOKBOOK_PATH на tmp_path — ни один тест не должен коснуться реального
assets/lookbook/lookbook.json (см. test_real_repo_lookbook_untouched_by_
this_test_file — сверяет со снимком, снятым до тестов, файл сегодня уже
не пуст — 13 эталонов канала, добавлены вручную вне тестов)."""
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

# Снимок реального lookbook.json ДО того, как хоть один тест этого файла
# успел запуститься (модульный код выполняется один раз при импорте, до
# всех autouse-фикстур/тестов) — раньше test_real_repo_lookbook_untouched_
# by_this_test_file() сверялся с константой {"references": []}, что было
# верно, только пока канал не выпустил ни одного эпизода. Теперь реальный
# lookbook.json намеренно НЕ пуст (13 эталонов, добавлены вручную через
# lookbook_add.py CLI вне тестов, см. коммит) — тест должен доказывать
# "этот файл тестов его не трогал", а не "он пуст", иначе он сломан самим
# фактом легитимной курации, а не регрессией.
_REAL_LOOKBOOK_PATH = os.path.join(REPO_ROOT, "assets", "lookbook", "lookbook.json")
with open(_REAL_LOOKBOOK_PATH, encoding="utf-8") as _f:
    _REAL_LOOKBOOK_SNAPSHOT = _f.read()

sys.argv = ["lookbook_add.py", tempfile.gettempdir()]
import lookbook_add as la   # noqa: E402
import look_reference as lr  # noqa: E402


def _make_photo(path):
    w, h = 64, 64
    t = np.linspace(0.3, 0.7, w, dtype=np.float32)
    arr = (np.tile(t, (h, 1))[..., None] * np.ones(3, dtype=np.float32) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


@pytest.fixture(autouse=True)
def _isolated_lookbook(tmp_path, monkeypatch):
    lookbook_path = tmp_path / "lookbook" / "lookbook.json"
    lookbook_path.parent.mkdir(parents=True, exist_ok=True)
    lookbook_path.write_text('{"references": []}', encoding="utf-8")
    monkeypatch.setattr(lr, "LOOKBOOK_PATH", str(lookbook_path))
    return lookbook_path


def test_rejects_unknown_domain(tmp_path, monkeypatch):
    photo = str(tmp_path / "p.png")
    _make_photo(photo)
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "not_a_domain"])
    assert la.main() == 1


def test_rejects_missing_file(monkeypatch):
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", "/no/such/file.jpg", "--domain", "snow"])
    assert la.main() == 1


def test_adds_valid_reference(tmp_path, monkeypatch, _isolated_lookbook):
    photo = str(tmp_path / "p.png")
    _make_photo(photo)
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "snow", "--id", "snow_test"])
    assert la.main() == 0

    data = json.loads(_isolated_lookbook.read_text(encoding="utf-8"))
    assert len(data["references"]) == 1
    ref = data["references"][0]
    assert ref["id"] == "snow_test"
    assert ref["domain"] == "snow"
    assert len(ref["lab_mean"]) == 3
    assert "brightness" in ref and "contrast" in ref and "temperature" in ref
    assert os.path.exists(os.path.join(REPO_ROOT, ref["image"]))
    # Color Director: эталон измерен ТАКЖЕ в грейженом пространстве (см.
    # look_reference.graded_reference_lab), не только сыром lab_mean —
    # иначе closed-loop сравнивает разные пространства (см. коммит).
    assert set(ref["graded_lab_mean"].keys()) == set(lr.GRADE_REFERENCE_SECTIONS)
    assert ref["graded_recipe_fingerprint"] == lr._grade_recipe_fingerprint()


def test_adding_same_id_twice_replaces_not_duplicates(tmp_path, monkeypatch, _isolated_lookbook):
    photo = str(tmp_path / "p.png")
    _make_photo(photo)
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "snow", "--id", "dup"])
    assert la.main() == 0
    assert la.main() == 0
    data = json.loads(_isolated_lookbook.read_text(encoding="utf-8"))
    assert len(data["references"]) == 1


def test_adding_second_domain_keeps_first(tmp_path, monkeypatch, _isolated_lookbook):
    photo = str(tmp_path / "p.png")
    _make_photo(photo)
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "snow", "--id", "a"])
    assert la.main() == 0
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "night", "--id", "b"])
    assert la.main() == 0
    data = json.loads(_isolated_lookbook.read_text(encoding="utf-8"))
    ids = {r["id"] for r in data["references"]}
    assert ids == {"a", "b"}


def test_atomic_write_leaves_no_tmp_file(tmp_path, monkeypatch, _isolated_lookbook):
    photo = str(tmp_path / "p.png")
    _make_photo(photo)
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "snow"])
    assert la.main() == 0
    assert not os.path.exists(str(_isolated_lookbook) + ".tmp")


def test_no_args_prints_usage_and_fails(monkeypatch):
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py"])
    assert la.main() == 1


def test_first_write_stamps_channel_id(tmp_path, monkeypatch, _isolated_lookbook):
    photo = str(tmp_path / "p.png")
    _make_photo(photo)
    monkeypatch.setattr(lr, "CHANNEL_ID", "my_channel")
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "snow", "--id", "a"])
    assert la.main() == 0
    data = json.loads(_isolated_lookbook.read_text(encoding="utf-8"))
    assert data["channel_id"] == "my_channel"


def test_conflicting_channel_id_without_force_refuses(tmp_path, monkeypatch, _isolated_lookbook):
    photo = str(tmp_path / "p.png")
    _make_photo(photo)
    monkeypatch.setattr(lr, "CHANNEL_ID", "channel_a")
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "snow", "--id", "a"])
    assert la.main() == 0

    monkeypatch.setattr(lr, "CHANNEL_ID", "channel_b")
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "snow", "--id", "b"])
    assert la.main() == 1
    data = json.loads(_isolated_lookbook.read_text(encoding="utf-8"))
    assert data["channel_id"] == "channel_a"
    assert [r["id"] for r in data["references"]] == ["a"]   # ничего не дописано


def test_conflicting_channel_id_with_force_overwrites(tmp_path, monkeypatch, _isolated_lookbook):
    photo = str(tmp_path / "p.png")
    _make_photo(photo)
    monkeypatch.setattr(lr, "CHANNEL_ID", "channel_a")
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "snow", "--id", "a"])
    assert la.main() == 0

    monkeypatch.setattr(lr, "CHANNEL_ID", "channel_b")
    monkeypatch.setattr(la, "_real_argv", ["lookbook_add.py", photo, "--domain", "snow", "--id", "b", "--force"])
    assert la.main() == 0
    data = json.loads(_isolated_lookbook.read_text(encoding="utf-8"))
    assert data["channel_id"] == "channel_b"
    assert {r["id"] for r in data["references"]} == {"a", "b"}   # старая запись СОХРАНЕНА, не стёрта


def test_real_repo_lookbook_untouched_by_this_test_file():
    """Контрольная проверка: этот тестовый файл не должен был ни разу
    коснуться реального assets/lookbook/lookbook.json (все остальные тесты
    monkeypatch'или LOOKBOOK_PATH) — сверяем побайтово со снимком, снятым
    ДО первого теста (см. _REAL_LOOKBOOK_SNAPSHOT), а не с константной
    "пустотой" — реальный lookbook сегодня намеренно не пуст (13 эталонов
    канала, добавлены вне тестов)."""
    with open(_REAL_LOOKBOOK_PATH, encoding="utf-8") as f:
        current = f.read()
    assert current == _REAL_LOOKBOOK_SNAPSHOT
