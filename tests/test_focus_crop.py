"""Наезд на смысловую деталь рисунка: рамка от судьи, вырезка 16:9, проверка
вырезки самим судьёй.

Что держат тесты:
  * геометрия (focus_frame): вырезка 16:9 содержит рамку целиком, не уже
    порога, в пределах кадра; высокий предмет и «почти весь кадр» — без
    вырезки;
  * разбор ответа судьи: шкала 0-1000 и доли кадра; фикстуры — НАСТОЯЩИЕ
    ответы Qwen 3.7 Plus из замера 24.09;
  * путь в рендере: вырезка только с рамкой в метаданных кадра и при
    включённом флаге; любой сбой — кадр целиком;
  * отбор: рамка ставится только после «да» судьи на саму вырезку;
  * судья видит кадр с учётом поворота по EXIF, как ffmpeg;
  * подписи кэша меняются от флага и от геометрии."""
import json
import os
import sys
import tempfile
import types

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import focus_frame  # noqa: E402
import pipeline_smart as ps  # noqa: E402
import runs_sheet  # noqa: E402
import shot_judge  # noqa: E402

# Страница Азенкура (Français 5054, fol. 11) в рабочем размере Commons:
# миниатюра битвы и упавший рыцарь — рамки, найденные судьёй 24.09.
PAGE = (2000, 3673)
MINIATURE = [0.05, 0.095, 0.88, 0.385]
FALLEN_KNIGHT = [0.66, 0.31, 0.86, 0.375]


def _contains(region, box, iw, ih):
    x0, y0, x1, y1 = region
    return (x0 <= box[0] * iw + 0.5 and box[2] * iw <= x1 + 0.5
            and y0 <= box[1] * ih + 0.5 and box[3] * ih <= y1 + 0.5)


# ---------- геометрия ----------

@pytest.mark.parametrize("size,box", [
    (PAGE, MINIATURE), (PAGE, FALLEN_KNIGHT),
    ((2000, 2961), [0.085, 0.135, 0.518, 0.428]),     # щит на странице Тальхоффера
    ((2000, 1333), [0.44, 0.03, 0.67, 0.14]),         # замок вдали на миниатюре Азенкура
])
def test_region_is_16_9_contains_the_detail_and_stays_inside(size, box):
    iw, ih = size
    r = focus_frame.crop_region(iw, ih, box)
    assert r is not None
    x0, y0, x1, y1 = r
    assert 0 <= x0 < x1 <= iw and 0 <= y0 < y1 <= ih
    assert (x1 - x0) / (y1 - y0) == pytest.approx(16 / 9, abs=0.01)
    assert _contains(r, box, iw, ih)
    assert x1 - x0 >= focus_frame.MIN_WIDTH_PX


def test_small_detail_keeps_context_not_a_postage_stamp():
    """Упавший рыцарь — 1% страницы. Вырезка не уже порога: деталь в
    контексте сцены, а не растянутый на экран клочок."""
    iw, ih = PAGE
    x0, _y0, x1, _y1 = focus_frame.crop_region(iw, ih, FALLEN_KNIGHT)
    assert x1 - x0 >= max(focus_frame.MIN_WIDTH_PX, focus_frame.MIN_SHARE * iw) - 1


def test_tall_object_is_not_cut_into_a_strip():
    """Кинжал во всю страницу трактата в полосу 16:9 не входит — вырезки
    нет, страница идёт целиком (обрубок клинка хуже страницы)."""
    assert focus_frame.crop_region(2000, 2961, [0.587, 0.148, 0.787, 0.895]) is None


def test_whole_frame_box_needs_no_crop():
    assert focus_frame.crop_region(1920, 1080, [0.0, 0.0, 1.0, 1.0]) is None


def test_crop_file_is_cached_and_16_9(tmp_path):
    src = tmp_path / "page.jpg"
    Image.new("RGB", PAGE, (200, 180, 140)).save(src, quality=90)
    out = focus_frame.crop_file(str(src), MINIATURE, str(tmp_path / "crops"))
    with Image.open(out) as im:
        assert im.width / im.height == pytest.approx(16 / 9, abs=0.01)
    assert focus_frame.crop_file(str(src), MINIATURE, str(tmp_path / "crops")) == out


# ---------- разбор ответа судьи ----------

@pytest.mark.parametrize("raw,want", [
    # настоящие ответы Qwen 3.7 Plus, замер 24.09
    ('{"what": "fallen knight", "box": [670, 310, 870, 370]}', [0.67, 0.31, 0.87, 0.37]),
    ('```json\n{"what": "castle on a hill", "box": [438, 35, 662, 135]}\n```', [0.438, 0.035, 0.662, 0.135]),
    ('{"box": [0.05, 0.1, 0.88, 0.38]}', [0.05, 0.1, 0.88, 0.38]),
])
def test_parse_box_reads_both_scales(raw, want):
    assert shot_judge.parse_box(raw) == pytest.approx(want)


@pytest.mark.parametrize("raw", [
    '{"what": "", "box": null}',
    'не знаю',
    '{"box": [900, 100, 100, 900]}',          # перевёрнута
    '{"box": [0, 0, 1000, 1000]}',            # весь кадр
    '{"box": [500, 500, 510, 510]}',          # точка
    '{"box": [10, 10, 2000, 900]}',           # вне шкалы
    '{"box": [0.1, 0.2, 0.3]}',
])
def test_parse_box_rejects_garbage(raw):
    assert shot_judge.parse_box(raw) is None


@pytest.mark.parametrize("raw,want", [
    ('{"shows": "yes"}', True), ('```json\n{"shows": "no"}\n```', False),
    ('{"shows": "maybe"}', None), ("yes", None), (None, None)])
def test_parse_shows(raw, want):
    assert shot_judge.parse_shows(raw) is want


class _Gateway:
    """Шлюз-двойник: отвечает по очереди, считает вызовы."""

    def __init__(self, *answers, fail=False):
        self.answers = list(answers)
        self.calls = []
        self.fail = fail

    def chat(self, model, content, max_tokens, est, **kw):
        self.calls.append(content[0]["text"])
        if self.fail:
            raise RuntimeError("шлюз лежит")
        return self.answers.pop(0), {}, 200


def _page(tmp_path, name="page.jpg"):
    p = tmp_path / name
    Image.new("RGB", PAGE, (200, 180, 140)).save(p, quality=90)
    return str(p)


def test_locate_box_is_cached_by_picture(tmp_path):
    path = _page(tmp_path)
    gw = _Gateway('{"what": "battle", "box": [50, 95, 880, 385]}')
    box, info = shot_judge.locate_box(gw, "m", path=path, focus="knights fighting",
                                      cache_dir=str(tmp_path / "c"))
    assert box == pytest.approx(MINIATURE) and info["what"] == "battle"
    again, info2 = shot_judge.locate_box(gw, "m", path=path, focus="knights fighting",
                                         cache_dir=str(tmp_path / "c"))
    assert again == box and info2.get("cache_hit") and len(gw.calls) == 1


def test_gateway_failure_gives_no_box(tmp_path):
    box, info = shot_judge.locate_box(_Gateway(fail=True), "m", path=_page(tmp_path), focus="x")
    assert box is None and "refused" in info


def test_judge_sees_the_picture_upright_like_ffmpeg(tmp_path):
    """JPEG 400x200 с меткой поворота 6: ffmpeg 6.x отдаёт 200x400 (замер
    24.09) — судья обязан видеть то же, иначе рамка ляжет не туда."""
    p = tmp_path / "exif6.jpg"
    im = Image.new("RGB", (400, 200), (200, 50, 50))
    ex = im.getexif()
    ex[0x0112] = 6
    im.save(p, exif=ex.tobytes())
    with Image.open(p) as raw:
        assert shot_judge.flat_rgb(raw).size == (200, 400)


# ---------- рендер ----------

def _with_box(tmp_path, box, name="page.jpg"):
    path = _page(tmp_path, name)
    ps.write_media_sidecar(path, pexels_id="commons:1", kind="photo", focus_box=box)
    return path


def test_render_uses_the_crop_when_the_frame_has_a_box(tmp_path, monkeypatch):
    monkeypatch.setenv("FOCUS_CROP", "1")
    path = _with_box(tmp_path, MINIATURE)
    out = ps.focus_crop(path)
    assert out != path
    with Image.open(out) as im:
        assert im.width / im.height == pytest.approx(16 / 9, abs=0.01)
    # вырезка уже 16:9 — подложка её не трогает, путь тот же
    assert ps.aspect_fit_backdrop(out) == out
    # повторный вызов по вырезке — та же вырезка, без второй вырезки
    assert ps.focus_crop(out) == out


def test_render_is_unchanged_without_box_or_with_flag_off(tmp_path, monkeypatch):
    plain = _page(tmp_path, "plain.jpg")
    monkeypatch.setenv("FOCUS_CROP", "1")
    assert ps.focus_crop(plain) == plain
    boxed = _with_box(tmp_path, MINIATURE, "boxed.jpg")
    monkeypatch.setenv("FOCUS_CROP", "0")
    assert ps.focus_crop(boxed) == boxed


def test_render_survives_a_broken_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("FOCUS_CROP", "1")
    path = _page(tmp_path)
    with open(ps.media_sidecar_path(path), "w", encoding="utf-8") as f:
        f.write('{"focus_box": [1, 2')
    assert ps.focus_crop(path) == path


# ---------- отбор ----------

def _request(focus="a knight lying fallen on the ground"):
    return types.SimpleNamespace(shot_spec={"focus": focus}, shot_brief=None, block_text="фраза")


@pytest.fixture
def judge_state(monkeypatch, tmp_path):
    monkeypatch.setenv("FOCUS_CROP", "1")
    monkeypatch.setattr(ps, "SHOT_JUDGE_LOG", [])
    # Свой кэш ответов судьи: страницы тестов одинаковы по байтам, и общий
    # на сессию кэш отдал бы ответ соседнего теста.
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp_smart"))

    def install(gw):
        monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "gateway", gw)
        return gw
    return install


def test_box_is_kept_only_after_the_judge_confirms_the_crop(tmp_path, judge_state):
    gw = judge_state(_Gateway('{"what": "fallen knight", "box": [660, 310, 860, 375]}',
                              '{"shows": "yes"}'))
    box = ps.locate_focus_box(3, _page(tmp_path), {"verify_medium": "artwork"}, _request())
    assert box == pytest.approx(FALLEN_KNIGHT)
    assert len(gw.calls) == 2 and "Does this picture clearly show" in gw.calls[1]
    assert ps.SHOT_JUDGE_LOG[-1]["focus_box_used"] == box


def test_crop_the_judge_rejects_is_not_used(tmp_path, judge_state):
    judge_state(_Gateway('{"what": "two riders", "box": [118, 595, 472, 760]}', '{"shows": "no"}'))
    box = ps.locate_focus_box(3, _page(tmp_path), {"verify_medium": "artwork"}, _request())
    assert box is None and ps.SHOT_JUDGE_LOG[-1]["focus_dropped"] == "confirm"


def test_tall_detail_is_dropped_before_asking_again(tmp_path, judge_state):
    """Рамка, которая в вырезку 16:9 не входит, — без второго вопроса."""
    gw = judge_state(_Gateway('{"what": "dagger", "box": [100, 50, 300, 950]}'))
    box = ps.locate_focus_box(3, _page(tmp_path), {"verify_medium": "artwork"}, _request())
    assert box is None and len(gw.calls) == 1
    assert ps.SHOT_JUDGE_LOG[-1]["focus_dropped"] == "region"


def test_ordinary_landscape_photo_is_not_asked(tmp_path, judge_state):
    gw = judge_state(_Gateway())
    p = tmp_path / "wide.jpg"
    Image.new("RGB", (1920, 1080)).save(p)
    assert ps.locate_focus_box(3, str(p), {"verify_medium": "photo"}, _request()) is None
    assert gw.calls == []


def test_no_judge_or_flag_off_means_no_question(tmp_path, judge_state, monkeypatch):
    judge_state(None)
    assert ps.locate_focus_box(3, _page(tmp_path), {"verify_medium": "artwork"}, _request()) is None
    gw = judge_state(_Gateway())
    monkeypatch.setenv("FOCUS_CROP", "0")
    assert ps.locate_focus_box(3, _page(tmp_path), {"verify_medium": "artwork"}, _request()) is None
    assert gw.calls == []


# ---------- контактный лист ----------

def test_contact_sheet_shows_the_same_crop(tmp_path, monkeypatch):
    monkeypatch.setenv("FOCUS_CROP", "1")
    path = _with_box(tmp_path, MINIATURE)
    assert focus_frame.box_of(path) == MINIATURE
    tile, cropped = runs_sheet.thumb(path, "photo", MINIATURE)
    assert cropped and tile.size == (runs_sheet.TW, runs_sheet.TH)
    _tile, plain = runs_sheet.thumb(path, "photo", None)
    assert not plain


def test_shotlist_contact_tile_shows_the_crop_not_the_page(tmp_path):
    """Лист Шага 7.5 — то, что человек размечает глазами: на нём кадр,
    каким его увидит зритель. Страница целиком (узкая плитка с полями) и
    вырезка 16:9 (плитка во всю ширину) различимы по заполнению."""
    import shotlist_contact as sc
    path = _with_box(tmp_path, MINIATURE)
    full = sc.thumbnail_for(_page(tmp_path, "plain.jpg"))
    crop = sc.thumbnail_for(path)
    # у страницы по бокам чёрные поля, у вырезки — нет
    assert full.getpixel((2, sc.THUMB_H // 2)) == (16, 16, 16)
    assert crop.getpixel((2, sc.THUMB_H // 2)) != (16, 16, 16)


def test_sidecar_suffix_matches_the_pipeline():
    assert ps.media_sidecar_path("x.jpg") == "x.jpg" + focus_frame.SIDECAR_SUFFIX


# ---------- подписи кэша ----------

def test_judge_signature_follows_flag_and_geometry(monkeypatch):
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("FOCUS_CROP", "1")
    on = ps.shot_judge_signature(0)
    monkeypatch.setenv("FOCUS_CROP", "0")
    off = ps.shot_judge_signature(0)
    assert on and off and on != off
    monkeypatch.setenv("FOCUS_CROP", "1")
    monkeypatch.setattr(focus_frame, "MARGIN", focus_frame.MARGIN + 0.05)
    assert ps.shot_judge_signature(0) != on


def test_render_recipe_follows_geometry(monkeypatch):
    before = ps.render_recipe_signature()
    monkeypatch.setattr(focus_frame, "MIN_WIDTH_PX", focus_frame.MIN_WIDTH_PX + 100)
    assert ps.render_recipe_signature() != before


def test_focus_box_is_judge_code_not_free_zone_code():
    """Рамка работает только при судье — её правка не должна перекачивать
    слоты бесплатной зоны."""
    assert ps.locate_focus_box in ps._judge_code_entries()
    assert "focus_frame" in ps.SELECTION_CODE_MODULES


def test_main_measures_and_renders_the_crop():
    """Яркость, уровни, лицо и рендер слота считаются по вырезке — по тому,
    что увидит зритель, а не по странице целиком."""
    import inspect
    src = inspect.getsource(ps.main)
    assert src.index("photo = focus_crop(photo)") < src.index("luma = measure_luma(photo")


def test_verify_answer_names_the_medium_the_crop_relies_on():
    """Рамку спрашивают у рисунков — признак берётся из ответа проверки."""
    assert '"medium"' in shot_judge.CLAIMS_PROMPT and "artwork" in shot_judge.CLAIMS_PROMPT


def test_both_photo_render_paths_crop_before_the_backdrop():
    import inspect
    for fn in (ps.kenburns, ps.parallax_kenburns):
        src = inspect.getsource(fn)
        assert src.index("photo = focus_crop(photo)") < src.index("aspect_fit_backdrop(photo)")
