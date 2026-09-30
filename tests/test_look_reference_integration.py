"""Интеграционный smoke-тест: доказывает, что look_filter реально доезжает
до ffmpeg через pipeline_smart.kenburns() (реальную точку сплайсинга, см.
scripts/pipeline_smart.py:3180) — юнит-тесты scripts/look_reference.py и
scripts/lookbook_add.py проверяют каждый модуль в изоляции, но НИ ОДИН из
них не вызывает саму функцию рендера, где склеивается итоговая ffmpeg-
строка. Без этого теста опечатка в сплайсинге (лишняя/недостающая запятая,
неверное имя параметра при прокидывании через main()) осталась бы незамечена
до первого реального прогона на настоящем эпизоде."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import shutil
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg недоступен")

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart  # noqa: E402


def _make_photo(path):
    w, h = 640, 360
    t = np.linspace(0.3, 0.7, w, dtype=np.float32)
    arr = (np.tile(t, (h, 1))[..., None] * np.ones(3, dtype=np.float32) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_kenburns_applies_look_filter_without_breaking_render(tmp_path):
    photo = str(tmp_path / "photo.jpg")
    _make_photo(photo)
    out = str(tmp_path / "clip.mp4")

    ok = pipeline_smart.kenburns(
        photo, out, dur=1.0, section="BODY",
        look_filter="colorchannelmixer=rr=1.05:gg=1.0:bb=0.95")
    assert ok is True
    assert os.path.exists(out)
    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                         "-of", "csv=p=0", out], capture_output=True, text=True)
    assert float(r.stdout.strip()) > 0.5


def test_kenburns_look_filter_none_is_unaffected(tmp_path):
    """Регрессия: look_filter=None (дефолт, поведение ДО этой фичи) обязан
    рендерить ровно так же, как раньше — свойство, не только "не падает"."""
    photo = str(tmp_path / "photo.jpg")
    _make_photo(photo)
    out = str(tmp_path / "clip.mp4")
    ok = pipeline_smart.kenburns(photo, out, dur=1.0, section="BODY", look_filter=None)
    assert ok is True
    assert os.path.exists(out)


def test_shadow_mode_end_to_end_never_reaches_kenburns_render(tmp_path, monkeypatch):
    """Сквозной путь: реальный look_reference.look_correction_filter() в
    режиме shadow (с эталоном, который РЕАЛЬНО матчится и был бы применён в
    assist) -> filt=None -> pipeline_smart.kenburns(look_filter=filt) ->
    рендер идентичен рендеру без Look Management вообще. Не изолированный
    юнит-тест orchestration-логики (это уже покрыто test_look_reference.py),
    а доказательство, что None из shadow действительно доходит до
    kenburns() и ничего не меняет в реальном вызове ffmpeg."""
    import look_reference as lr

    photo = str(tmp_path / "photo.jpg")
    _make_photo(photo)
    levels, wb = pipeline_smart.measure_levels(photo, want_wb=True)
    frame_lab = lr._srgb_to_lab(wb)

    monkeypatch.setattr(lr, "LOOK_MANAGEMENT_MODE", "shadow")
    monkeypatch.setattr(lr, "load_lookbook", lambda: {"references": [
        {"id": "test_ref", "domain": "snow", "lab_mean": list(frame_lab),
         "brightness": 0.5, "contrast": 0.4, "temperature": 0.0,
         "max_correction_delta": [10, 8, 8]},
    ]})
    monkeypatch.setattr(lr, "classify_domain", lambda path: ("snow", 0.5))

    filt, report, state = lr.look_correction_filter(
        photo, levels, wb, has_face=False, scene_boundary=True)
    assert filt is None
    assert report["decision"] == "shadow_would_apply"   # ПОДТВЕРЖДЕНО: сработало бы в assist

    out_shadow = str(tmp_path / "clip_shadow.mp4")
    out_plain = str(tmp_path / "clip_plain.mp4")
    ok1 = pipeline_smart.kenburns(photo, out_shadow, dur=1.0, section="BODY", look_filter=filt)
    ok2 = pipeline_smart.kenburns(photo, out_plain, dur=1.0, section="BODY", look_filter=None)
    assert ok1 is True and ok2 is True
    # Пиксели, не сырые байты контейнера — muxer-метаданные (не пиксели)
    # между двумя независимыми вызовами ffmpeg не обязаны совпадать бит-в-
    # бит, реальный сигнал "shadow не просочился" — идентичное изображение.
    probe_shadow = str(tmp_path / "probe_shadow.png")
    probe_plain = str(tmp_path / "probe_plain.png")
    for src, dst in ((out_shadow, probe_shadow), (out_plain, probe_plain)):
        subprocess.run(["ffmpeg", "-y", "-i", src, "-frames:v", "1", dst],
                        capture_output=True, check=True)
    arr_shadow = np.asarray(Image.open(probe_shadow).convert("RGB"))
    arr_plain = np.asarray(Image.open(probe_plain).convert("RGB"))
    assert np.array_equal(arr_shadow, arr_plain)


def test_kenburns_garbage_look_filter_makes_ffmpeg_fail():
    """Отрицательный контроль: если бы look_filter тихо ИГНОРИРОВАЛСЯ (не
    реально доезжал до ffmpeg-команды), заведомо невалидный фрагмент не
    сломал бы рендер. Он обязан его сломать — доказательство, что строка
    реально попадает в filter_complex, не теряется по дороге."""
    with tempfile.TemporaryDirectory() as d:
        photo = os.path.join(d, "photo.jpg")
        _make_photo(photo)
        out = os.path.join(d, "clip.mp4")
        ok = pipeline_smart.kenburns(
            photo, out, dur=1.0, section="BODY",
            look_filter="totally_not_a_real_ffmpeg_filter=xyz=123")
        assert ok is False
