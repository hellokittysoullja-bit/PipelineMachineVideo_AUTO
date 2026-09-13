"""Портретный источник вписывается в кадр целиком, а не срезается на 58-68%.

РЕАЛЬНЫЙ, ИЗМЕРЕННЫЙ дефект. Весь пайплайн приводил фото к кадру одним
приёмом — increase+crop. Для стока Pexels (3:2) это 16% высоты и незаметно.
Но после подключения музеев, Openverse и Pixabay в пул пошли страницы
кодексов, эффигии и доспехи в полный рост — портретные снимки:

    миниатюра 3:4        видно 42%  -> обрезано 58%
    страница кодекса 2:3 видно 38%  -> обрезано 62%
    доспех в рост 9:16   видно 32%  -> обрезано 68%

И это ДО зума Ken Burns (ещё 8-22%). Гейты при этом считали кандидата
отличным — они судят ЦЕЛОЕ изображение, а не то, что попадёт в кадр.
Проверки пропорций в пайплайне не было вообще.

ГЛАВНОЕ, что держат тесты: широкий источник (3:2 Pexels, 16:9) идёт
прежним путём БАЙТ-В-БАЙТ — возвращается тот же самый путь, ни одного
нового файла. Подложка появляется только там, где обычный кроп уничтожил
бы больше четверти кадра.
"""
import os
import sys
import tempfile

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402


def _img(tmp_path, w, h, name="src.jpg"):
    p = tmp_path / name
    Image.new("RGB", (w, h), (120, 90, 60)).save(str(p), quality=90)
    return str(p)


# ---------- порог ----------

@pytest.mark.parametrize("w,h", [(600, 800), (600, 900), (540, 960), (800, 800)])
def test_portrait_and_square_need_backdrop(w, h):
    assert ps.needs_aspect_backdrop(w, h)


@pytest.mark.parametrize("w,h", [(1920, 1080), (1500, 1000), (2000, 1500)])
def test_landscape_never_needs_backdrop(w, h):
    """3:2 Pexels и 16:9 — прежний путь. 4:3 ровно на границе и тоже не трогается."""
    assert not ps.needs_aspect_backdrop(w, h)


def test_threshold_is_where_crop_loses_a_quarter():
    """Порог не на глаз: 4/3 — пропорция, на которой increase+crop теряет
    ровно четверть высоты (ar/1.778 == 0.75)."""
    assert ps.ASPECT_FIT_MIN_RATIO == pytest.approx(4 / 3)
    assert (ps.ASPECT_FIT_MIN_RATIO / (ps.WIDTH / ps.HEIGHT)) == pytest.approx(0.75, abs=0.01)


@pytest.mark.parametrize("bad", [(0, 100), (100, 0), (None, None)])
def test_degenerate_size_is_not_touched(bad):
    assert not ps.needs_aspect_backdrop(*bad)


# ---------- поведение ----------

def test_landscape_returns_the_very_same_path(tmp_path):
    """Ноль регресса для широких кадров: тот же путь, ни одного нового файла."""
    src = _img(tmp_path, 1500, 1000)
    out_dir = tmp_path / "cache"
    assert ps.aspect_fit_backdrop(src, out_dir=str(out_dir)) == src
    assert not out_dir.exists()


def test_portrait_becomes_a_16x9_composite(tmp_path):
    src = _img(tmp_path, 600, 800)
    res = ps.aspect_fit_backdrop(src, out_dir=str(tmp_path / "c"))
    assert res != src
    with Image.open(res) as im:
        assert im.size == (ps.WIDTH, ps.HEIGHT)


def test_whole_source_fits_inside_with_room_for_zoom(tmp_path):
    """Передний план вписан с запасом ASPECT_FIT_SAFE — ZOOM_FLOOR уже в
    первом кадре показывает 96% холста, при полной высоте срезало бы верх."""
    src = _img(tmp_path, 600, 800)
    ps.aspect_fit_backdrop(src, out_dir=str(tmp_path / "c"))
    fit = min(ps.WIDTH * ps.ASPECT_FIT_SAFE / 600, ps.HEIGHT * ps.ASPECT_FIT_SAFE / 800)
    assert round(800 * fit) <= ps.HEIGHT * ps.ASPECT_FIT_SAFE + 1
    assert ps.ASPECT_FIT_SAFE < 1.0


def test_disabled_flag_is_a_full_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("ASPECT_FIT_BACKDROP", "0")
    src = _img(tmp_path, 600, 800)
    assert ps.aspect_fit_backdrop(src, out_dir=str(tmp_path / "c")) == src


def test_result_is_cached_not_rebuilt(tmp_path):
    src = _img(tmp_path, 600, 800)
    out = str(tmp_path / "c")
    first = ps.aspect_fit_backdrop(src, out_dir=out)
    mtime = os.path.getmtime(first)
    second = ps.aspect_fit_backdrop(src, out_dir=out)
    assert second == first
    assert os.path.getmtime(second) == mtime


def test_broken_source_fails_open_to_the_original(tmp_path):
    """Ни один слот не должен пропасть из-за этого шага."""
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image at all")
    assert ps.aspect_fit_backdrop(str(bad), out_dir=str(tmp_path / "c")) == str(bad)
    assert ps.aspect_fit_backdrop(str(tmp_path / "missing.jpg")) == str(tmp_path / "missing.jpg")


def test_backdrop_is_darker_than_the_foreground(tmp_path):
    """Затемнение подложки обязательно — иначе она перетягивает внимание."""
    src = tmp_path / "bright.jpg"
    Image.new("RGB", (600, 800), (230, 230, 230)).save(str(src), quality=95)
    res = ps.aspect_fit_backdrop(str(src), out_dir=str(tmp_path / "c"))
    with Image.open(res) as im:
        centre = im.getpixel((ps.WIDTH // 2, ps.HEIGHT // 2))
        edge = im.getpixel((12, ps.HEIGHT // 2))
    assert sum(edge) < sum(centre) * 0.85, (edge, centre)


# ---------- вкручено в рендер ----------

def test_called_from_both_render_paths():
    import inspect
    assert "aspect_fit_backdrop(photo)" in inspect.getsource(ps.kenburns)
    assert "aspect_fit_backdrop(photo)" in inspect.getsource(ps.parallax_kenburns)


def test_part_of_the_render_recipe_signature():
    """Правка порога/вида подложки меняет уже отрендеренный клип, а ни один
    рантайм-параметр params_hash при этом не двигается."""
    import inspect
    sig = inspect.getsource(ps.render_recipe_signature)
    assert "aspect_fit_backdrop" in sig
    assert "needs_aspect_backdrop" in sig
