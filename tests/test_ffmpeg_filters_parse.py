"""Разбор `ffmpeg -filters`: старый формат (3 флага) и новый (2 флага)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

OLD = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
 ... abench            A->A       Benchmark part of a filtergraph.
 T.C scale             V->V       Scale the input video size.
 T.. drawtext          V->V       Draw text on top of video frames.
 ..C xfade             VV->V      Cross fade one video with another video.
 ... amovie            |->N       Read audio from a movie source.
"""
NEW = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
  ------
 TS aap               AA->A      Apply Affine Projection algorithm to first audio stream.
 .. abench            A->A       Benchmark part of a filtergraph.
 T. scale             V->V       Scale the input video size.
 .S zoompan           V->V       Apply Zoom & Pan effect.
 .. xfade             VV->V      Cross fade one video with another video.
"""


def test_parses_the_old_three_flag_format():
    import pipeline_smart as ps
    assert {"abench", "scale", "drawtext", "xfade", "amovie"} <= ps.parse_ffmpeg_filters(OLD)


def test_parses_the_new_two_flag_format():
    import pipeline_smart as ps
    have = ps.parse_ffmpeg_filters(NEW)
    assert {"aap", "abench", "scale", "zoompan", "xfade"} <= have


def test_legend_lines_are_not_filters():
    import pipeline_smart as ps
    have = ps.parse_ffmpeg_filters(NEW + OLD)
    assert not {"=", "A", "V", "N", "|", "------", "Timeline"} & have


def test_nvenc_failure_reason_is_kept_and_printed(monkeypatch, capsys):
    import subprocess
    import pipeline_smart as ps

    class R:
        returncode = 1
        stderr = "[hevc_nvenc @ 0x1] Driver does not support the required nvenc API version.\nRequired: 13.0 Found: 12.2\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    monkeypatch.setenv("CLIP_ENCODER", "auto")
    assert ps.nvenc_works() is False
    assert "Required: 13.0" in ps.nvenc_failure_reason()
    assert ps.resolve_clip_encoder() == "x264"
    out = capsys.readouterr().out
    assert "причина:" in out and "Required: 13.0" in out


def test_nvenc_success_clears_the_reason(monkeypatch):
    import subprocess
    import pipeline_smart as ps

    class R:
        returncode = 0
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    ps._NVENC_REASON[0] = "старая"
    assert ps.nvenc_works() is True and ps.nvenc_failure_reason() is None


def test_depth_to_numpy_matches_the_old_conversion_without_warnings():
    import warnings

    import numpy as np
    import torch
    import pipeline_smart as ps
    t = torch.tensor([[[0.25, 1.5], [3.0, -2.0]]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = np.array(t, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = ps.depth_to_numpy(t)
    assert new.dtype == np.float32 and new.shape == old.shape
    assert np.array_equal(new, old)
    assert ps.depth_to_numpy([[1, 2]]).dtype == np.float32
