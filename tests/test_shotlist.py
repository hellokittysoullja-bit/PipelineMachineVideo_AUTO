"""Шотлист (media_plan/shotlist.json) — проверяемый и правимый список кадров,
см. блок-комментарий у SHOTLIST_VERSION в pipeline_smart.py.

Проверяем контракт, на который опирается пользователь при ручной правке:
lock берётся ТОЛЬКО при совпадении фразы и существующем файле; видео/фото
различаются по расширению; ручные lock-флаги переживают перезапись
шотлиста следующим прогоном; ключ кэша клипа меняется при смене файла."""
import json
import os
import sys
import tempfile

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402
import shotlist_contact   # noqa: E402


def _shotlist(tmp_path, shots, locked=False):
    d = tmp_path / "media_plan"
    d.mkdir(exist_ok=True)
    (d / "shotlist.json").write_text(
        json.dumps({"version": 1, "locked": locked, "shots": shots}, ensure_ascii=False), encoding="utf-8")
    return ps.load_shotlist(str(tmp_path))


def _photo(tmp_path, name="media/007_mine.jpg"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), (120, 80, 40)).save(p)
    return p


def test_load_missing_returns_none(tmp_path):
    assert ps.load_shotlist(str(tmp_path)) is None


def test_load_broken_json_is_fail_open(tmp_path, capsys):
    d = tmp_path / "media_plan"
    d.mkdir()
    (d / "shotlist.json").write_text("{not json", encoding="utf-8")
    assert ps.load_shotlist(str(tmp_path)) is None
    assert "не читается" in capsys.readouterr().out


def test_locked_photo_resolved_relative_to_video_dir(tmp_path):
    _photo(tmp_path)
    sl = _shotlist(tmp_path, [{"index": 6, "text": "Меч весил полтора кило.", "file": "media/007_mine.jpg", "lock": True}])
    photo, video = ps.shotlist_locked_media(sl, 6, "Меч  весил полтора кило.", str(tmp_path))
    assert photo == os.path.normpath(str(tmp_path / "media/007_mine.jpg"))
    assert video is None


def test_locked_video_detected_by_extension(tmp_path):
    v = tmp_path / "temp_smart" / "clip.mp4"
    v.parent.mkdir()
    v.write_bytes(b"\x00" * 16)
    sl = _shotlist(tmp_path, [{"index": 0, "text": "x", "file": "temp_smart/clip.mp4", "lock": True}])
    photo, video = ps.shotlist_locked_media(sl, 0, "x", str(tmp_path))
    assert photo is None and video == os.path.normpath(str(v))


def test_unlocked_slot_is_ignored(tmp_path):
    _photo(tmp_path)
    sl = _shotlist(tmp_path, [{"index": 6, "text": "x", "file": "media/007_mine.jpg", "lock": False}])
    assert ps.shotlist_locked_media(sl, 6, "x", str(tmp_path)) == (None, None)


def test_top_level_locked_applies_to_all_with_file(tmp_path):
    _photo(tmp_path)
    sl = _shotlist(tmp_path, [{"index": 6, "text": "x", "file": "media/007_mine.jpg"}], locked=True)
    photo, _ = ps.shotlist_locked_media(sl, 6, "x", str(tmp_path))
    assert photo is not None


def test_text_mismatch_ignores_lock_with_warning(tmp_path, capsys):
    _photo(tmp_path)
    sl = _shotlist(tmp_path, [{"index": 6, "text": "старая фраза", "file": "media/007_mine.jpg", "lock": True}])
    assert ps.shotlist_locked_media(sl, 6, "новая фраза", str(tmp_path)) == (None, None)
    assert "фраза блока изменилась" in capsys.readouterr().out


def test_missing_file_ignores_lock_with_warning(tmp_path, capsys):
    sl = _shotlist(tmp_path, [{"index": 1, "text": "x", "file": "media/nope.jpg", "lock": True}])
    assert ps.shotlist_locked_media(sl, 1, "x", str(tmp_path)) == (None, None)
    assert "не найден" in capsys.readouterr().out


def test_lock_key_changes_with_file_content(tmp_path):
    p = _photo(tmp_path)
    k1 = ps.shotlist_lock_key(str(p), None)
    assert k1.startswith("lock:")
    Image.new("RGB", (128, 72), (10, 10, 10)).save(p)   # другой размер файла
    k2 = ps.shotlist_lock_key(str(p), None)
    assert k1 != k2
    assert ps.shotlist_lock_key(None, None) == ""


def test_source_classification(tmp_path):
    vd = str(tmp_path)
    assert ps.shotlist_source_for(os.path.join(vd, "media", "001.jpg"), vd) == "local"
    assert ps.shotlist_source_for(os.path.join(vd, "temp_smart", "pexels_cache", "a.jpg"), vd) == "pexels"
    assert ps.shotlist_source_for(os.path.join(vd, "media", "001.jpg"), vd, locked=True) == "shotlist_lock"
    assert ps.shotlist_source_for(None, vd) == "missing"


def test_relative_file_inside_and_outside_video_dir(tmp_path):
    vd = str(tmp_path / "ep")
    os.makedirs(vd)
    assert ps.shotlist_relative_file(os.path.join(vd, "media", "a.jpg"), vd) == "media/a.jpg"
    outside = str(tmp_path / "elsewhere.jpg")
    assert os.path.isabs(ps.shotlist_relative_file(outside, vd))


def test_write_preserves_manual_lock_and_file(tmp_path):
    vd = str(tmp_path)
    prev = {"locked": False, "shots": [
        {"index": 0, "text": "фраза ноль", "file": "media/manual.jpg", "kind": "photo", "lock": True},
        {"index": 1, "text": "фраза один", "file": "old.jpg", "lock": False},
    ]}
    shots = {
        0: {"index": 0, "section": "HOOK", "text": "фраза ноль", "query": "q", "kind": None, "file": None,
            "source": "cache_hit_unknown_file", "clip": "c0.mp4"},
        1: {"index": 1, "section": "HOOK", "text": "фраза один", "query": "q", "kind": "photo",
            "file": "temp_smart/pexels_cache/new.jpg", "source": "pexels", "clip": "c1.mp4"},
        2: {"index": 2, "section": "HOOK", "text": "фраза два", "query": "q", "kind": "photo",
            "file": "media/x.jpg", "source": "local", "clip": "c2.mp4"},
    }
    path = ps.write_shotlist(vd, shots, {"pexels_api_key": False}, prev=prev)
    data = json.load(open(path, encoding="utf-8"))
    by = {s["index"]: s for s in data["shots"]}
    assert by[0]["lock"] is True and by[0]["file"] == "media/manual.jpg" and by[0]["kind"] == "photo"
    assert by[1]["lock"] is False and by[1]["file"] == "temp_smart/pexels_cache/new.jpg"
    assert by[2]["lock"] is False
    assert data["gates"] == {"pexels_api_key": False}
    assert data["version"] == ps.SHOTLIST_VERSION


def test_write_drops_lock_when_text_changed(tmp_path):
    prev = {"shots": [{"index": 0, "text": "было", "file": "media/manual.jpg", "lock": True}]}
    shots = {0: {"index": 0, "section": "HOOK", "text": "стало", "query": "q", "kind": "photo",
                 "file": "media/other.jpg", "source": "local", "clip": "c.mp4"}}
    data = json.load(open(ps.write_shotlist(str(tmp_path), shots, {}, prev=prev), encoding="utf-8"))
    assert data["shots"][0]["lock"] is False and data["shots"][0]["file"] == "media/other.jpg"


def test_contact_sheet_renders_pages(tmp_path):
    vd = tmp_path
    _photo(vd, "media/001.jpg")
    shots = [{"index": i, "section": "BLOCK 1", "text": f"Фраза номер {i} про меч и доспех", "query": "q",
              "kind": "photo", "file": "media/001.jpg", "source": "local", "lock": i == 2} for i in range(5)]
    shots.append({"index": 5, "section": "FINAL", "text": "нет файла", "kind": None, "file": None, "source": "missing"})
    _shotlist(vd, shots)
    rc = shotlist_contact.main([str(vd), "--cols", "3", "--per-page", "4"])
    assert rc == 0
    p1 = vd / "media_plan" / "shotlist_contact_01.jpg"
    p2 = vd / "media_plan" / "shotlist_contact_02.jpg"
    assert p1.exists() and p2.exists()
    img = Image.open(p1)
    assert img.width == 3 * (shotlist_contact.THUMB_W + shotlist_contact.PAD) + shotlist_contact.PAD


def test_contact_sheet_without_shotlist(tmp_path):
    assert shotlist_contact.main([str(tmp_path)]) == 1


@pytest.mark.parametrize("mode,needle", [
    ("softlight", "blend=all_mode=softlight:all_opacity=0.2000"),
    ("grainmerge", "blend=all_mode=grainmerge:all_opacity=0.1000"),
    ("expr", "blend=all_expr="),
])
def test_grain_blend_modes(monkeypatch, mode, needle):
    monkeypatch.setattr(ps, "GRAIN_OPACITY", 0.10)
    monkeypatch.setattr(ps, "GRAIN_BLEND_MODE", mode)
    fc = ps.grain_blend_complex("in", 1, "out")
    assert fc.startswith("[1:v]scale=1920:1080")
    assert needle in fc and fc.endswith("[out]")


def test_grain_scale_multiplies_softlight_opacity(monkeypatch):
    monkeypatch.setattr(ps, "GRAIN_OPACITY", 0.10)
    monkeypatch.setattr(ps, "GRAIN_BLEND_MODE", "softlight")
    assert "all_opacity=0.1000" in ps.grain_blend_complex("in", 1, "out", opacity_scale=0.5)


def test_unknown_grain_mode_falls_back_to_softlight(monkeypatch):
    monkeypatch.setattr(ps, "GRAIN_BLEND_MODE", "typo")
    assert "all_mode=softlight" in ps.grain_blend_complex("in", 1, "out")
