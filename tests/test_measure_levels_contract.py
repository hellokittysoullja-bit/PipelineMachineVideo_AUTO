# -*- coding: utf-8 -*-
"""measure_levels(want_wb=True) ВСЕГДА возвращает пару: вызывающий код в main()
распаковывает её без проверки. Раньше видео без извлечённых кадров давало голое
None, и весь рендер падал TypeError посреди эпизода (живой случай 26.09, эп.95)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _ps():
    sys.argv = ["pipeline_smart.py", "/tmp"]
    import pipeline_smart
    return pipeline_smart


def test_video_without_frames_returns_pair(tmp_path):
    ps = _ps()
    missing = str(tmp_path / "нет_такого.mp4")
    assert ps.measure_levels(missing, is_video=True, want_wb=True) == (None, None)
    assert ps.measure_levels(missing, is_video=True) is None


def test_photo_without_numpy_returns_pair(tmp_path, monkeypatch):
    ps = _ps()
    monkeypatch.setattr(ps, "np", None)
    assert ps.measure_levels(str(tmp_path / "x.jpg"), want_wb=True) == (None, None)
