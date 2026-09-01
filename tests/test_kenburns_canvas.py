"""KENBURNS_ADAPTIVE_CANVAS (по умолчанию выключен) — см. докстринг
константы и _kenburns_canvas_size() в pipeline_smart.py. kenburns() сейчас
апскейлит КАЖДОЕ фото до фиксированных 8000x4500 перед однопоточным
zoompan — реальный измеренный (01_ves-mecha) главный вклад в тайминг
клипа, не связанный ни с одной ML-моделью. Флаг переключает на холст,
выведенный из уже существующих в kenburns() констант зума (не с потолка),
но остаётся ВЫКЛЮЧЕННЫМ по умолчанию — это тесты за флаг, не переключение
поведения; применение к реальному эпизоду — отдельное, явное решение
пользователя (см. план на ночь 31 авг/1 сен)."""
import os
import subprocess
import sys
import tempfile

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402


def test_canvas_default_is_unchanged_8000x4500():
    assert ps.KENBURNS_ADAPTIVE_CANVAS is False
    assert ps._kenburns_canvas_size() == (8000, 4500)


def test_canvas_adaptive_derived_from_width_height_margin(monkeypatch):
    monkeypatch.setattr(ps, "KENBURNS_ADAPTIVE_CANVAS", True)
    cw, ch = ps._kenburns_canvas_size()
    assert cw == round(ps.WIDTH * ps.KENBURNS_CANVAS_MARGIN)
    assert ch == round(ps.HEIGHT * ps.KENBURNS_CANVAS_MARGIN)
    # Много меньше старого фиксированного холста — это и есть весь смысл фичи.
    assert cw < 8000 and ch < 4500


def test_canvas_covers_max_possible_zoom_with_margin():
    # Расчёт, обосновывающий KENBURNS_CANVAS_MARGIN (см. докстринг у
    # объявления констант): максимальный zoom = ZOOM_FLOOR + ZOOM_DELTA_MAX.
    # Холст ДОЛЖЕН покрывать WIDTH*max_zoom по каждой стороне без запаса
    # снизу — иначе на пике зума кроп-окно апскейлилось бы при финальном
    # ресайзе в WIDTHxHEIGHT (реальная потеря резкости, не гипотеза).
    max_zoom = ps.ZOOM_FLOOR + ps.ZOOM_DELTA_MAX
    min_required_w = ps.WIDTH * max_zoom
    min_required_h = ps.HEIGHT * max_zoom
    adaptive_w = ps.WIDTH * ps.KENBURNS_CANVAS_MARGIN
    adaptive_h = ps.HEIGHT * ps.KENBURNS_CANVAS_MARGIN
    assert adaptive_w >= min_required_w
    assert adaptive_h >= min_required_h


def _capture_kenburns_cmd(monkeypatch, tmp_path):
    photo = str(tmp_path / "p.jpg")
    Image.new("RGB", (1880, 1253), (100, 110, 120)).save(photo)   # реальный типичный размер Pexels large2x
    out = str(tmp_path / "clip_0000.mp4")

    captured = []

    def fake_retry(build_cmd, tmp_out, expected_dur, label=""):
        build_cmd()   # реально строит cmd через render(vf) -> subprocess.run (замокан ниже)
        return False, "test short-circuit"

    def fake_run(cmd, **kw):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fake failure")

    monkeypatch.setattr(ps, "run_ffmpeg_with_retry", fake_retry)
    monkeypatch.setattr(ps.subprocess, "run", fake_run)
    ok = ps.kenburns(photo, out, 2.0, section="HOOK")
    assert ok is False   # fake_retry всегда "проваливает" — это ожидаемо, нас интересует только cmd
    assert captured, "subprocess.run ни разу не вызван — build_cmd() не сработал"
    return " ".join(str(x) for x in captured[0])


def test_flag_off_produces_byte_identical_8000x4500_command(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "KENBURNS_ADAPTIVE_CANVAS", False)
    cmd_str = _capture_kenburns_cmd(monkeypatch, tmp_path)
    assert "crop=8000:4500" in cmd_str


def test_flag_on_produces_smaller_canvas_command(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "KENBURNS_ADAPTIVE_CANVAS", True)
    cmd_str = _capture_kenburns_cmd(monkeypatch, tmp_path)
    cw, ch = round(ps.WIDTH * ps.KENBURNS_CANVAS_MARGIN), round(ps.HEIGHT * ps.KENBURNS_CANVAS_MARGIN)
    assert f"crop={cw}:{ch}" in cmd_str
    assert "crop=8000:4500" not in cmd_str


def test_wobble_amplitude_scales_with_canvas_not_absolute(tmp_path, monkeypatch):
    # WOBBLE_AMP_CANVAS_PX калиброван под холст 8000px (см. докстринг) —
    # при флаге off множитель обязан остаться ровно 1.0 (byte-for-byte),
    # при flag on — пропорционально меньше, иначе "дрожание плёнки"
    # визуально усилится на меньшем холсте (реальный, не гипотетический
    # побочный эффект, найденный агентом-разведчиком при аудите константы).
    monkeypatch.setattr(ps, "KENBURNS_ADAPTIVE_CANVAS", False)
    cmd_str_off = _capture_kenburns_cmd(monkeypatch, tmp_path)
    # На выключенном флаге амплитуда wobble = WOBBLE_AMP_CANVAS_PX (6.00) буквально.
    assert f"{ps.WOBBLE_AMP_CANVAS_PX:.2f}*sin" in cmd_str_off

    monkeypatch.setattr(ps, "KENBURNS_ADAPTIVE_CANVAS", True)
    cmd_str_on = _capture_kenburns_cmd(monkeypatch, tmp_path)
    cw, _ = ps._kenburns_canvas_size()
    expected_amp = ps.WOBBLE_AMP_CANVAS_PX * (cw / 8000.0)
    assert f"{expected_amp:.2f}*sin" in cmd_str_on
    assert expected_amp < ps.WOBBLE_AMP_CANVAS_PX   # меньше холст -> меньше абсолютных пикселей дрожания
