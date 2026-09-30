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
