# -*- coding: utf-8 -*-
"""Цвет и диапазон клипа не зависят от версии ffmpeg.

Найдено 30.09 при разборе самопроверки GPU-рендера на поде (сборка ffmpeg
N-126965): JPEG (yuvj, 0-255) при выходе в yuv420p не пересчитывался в
ограниченный диапазон (контраст +16%), `-colorspace bt709` как выходная
опция включал перевод матрицы (-1.5 уровня яркости), а `-color_primaries` и
`-color_trc` до потока не доходили. Тесты меряют результат самого ffmpeg,
который стоит в окружении, — на новой сборке они падали до правки."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="нет ffmpeg/ffprobe")


def _ff(args, inp=None):
    return subprocess.run(["ffmpeg", "-v", "error", "-y"] + args, input=inp, capture_output=True,
                          check=True).stdout


def _tags(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                          "stream=color_range,color_space,color_transfer,color_primaries",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return out.split(",")


def _photo(tmp_path, pix, ext):
    w, h = 200, 120
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.clip(np.stack([128 + 60 * np.sin(xx / 17 + c) * np.cos(yy / 23 - c) for c in (0, 1, 2)], -1)
                  + rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)
    p = tmp_path / f"p.{ext}"
    _ff(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-", "-pix_fmt", pix,
         "-q:v", "2", str(p)], img.tobytes())
    return p


def _encode(photo, out, first_filter):
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "10", "-threads", "1"] + ps.CLIP_PIX_ARGS
    enc += ps.color_meta_args("libx264")
    _ff(["-framerate", "1", "-loop", "1", "-i", str(photo), "-vf", first_filter + ",setsar=1",
         "-frames:v", "2"] + enc + [str(out)])


@needs_ffmpeg
def test_jpeg_full_range_is_converted_to_limited(tmp_path):
    photo = _photo(tmp_path, "yuvj420p", "jpg")
    full = np.frombuffer(_ff(["-i", str(photo), "-f", "rawvideo", "-pix_fmt", "yuvj420p",
                              "-frames:v", "1", "-"]), np.uint8)[:200 * 120].astype(float)
    ideal = (16 + 219 / 255 * full).mean()
    out = tmp_path / "o.mp4"
    _encode(photo, out, f"scale=200:120{ps.SOURCE_RANGE_OPT}")
    y = np.frombuffer(_ff(["-i", str(out), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-frames:v", "1", "-"]),
                      np.uint8)[:200 * 120].astype(float)
    assert abs(y.mean() - ideal) < 0.4, (y.mean(), ideal)
    # без явного перевода на новых сборках было +2: тест держит и сам приём
    assert y.min() >= 14 and y.max() <= 236


@needs_ffmpeg
def test_stream_carries_rec709_limited_tags(tmp_path):
    photo = _photo(tmp_path, "yuvj420p", "jpg")
    out = tmp_path / "o.mp4"
    _encode(photo, out, f"scale=200:120{ps.SOURCE_RANGE_OPT}")
    assert _tags(out) == ["tv", "bt709", "bt709", "bt709"]


@needs_ffmpeg
def test_range_option_is_a_noop_for_rgb_sources(tmp_path):
    photo = _photo(tmp_path, "rgb24", "png")
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _encode(photo, a, "scale=200:120")
    _encode(photo, b, f"scale=200:120{ps.SOURCE_RANGE_OPT}")

    def raw(p):
        return _ff(["-i", str(p), "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-frames:v", "2", "-"])
    assert raw(a) == raw(b)


def test_clip_and_delivery_args_carry_the_matching_bitstream_filter(monkeypatch):
    assert "h264_metadata" in " ".join(ps.clip_codec_args())
    monkeypatch.setattr(ps, "_NVENC_BROKEN", [False])
    monkeypatch.setattr(ps, "clip_encoder", lambda: "nvenc")
    nv = ps.clip_codec_args()
    assert "hevc_nvenc" in nv and "hevc_metadata" in " ".join(nv) and "h264_metadata" not in " ".join(nv)
    monkeypatch.setattr(ps, "DELIVERY_PROFILE", "hevc")
    assert "hevc_metadata" in " ".join(ps.final_pass_encode_args())
    monkeypatch.setattr(ps, "DELIVERY_PROFILE", "youtube")
    assert "h264_metadata" in " ".join(ps.final_pass_encode_args())


def test_chain_starts_convert_range_explicitly():
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    assert src.count("{SOURCE_RANGE_OPT}") == 3, "kenburns и оба варианта video_render"
    gpu = open(os.path.join(REPO, "scripts", "gpu_render.py"), encoding="utf-8").read()
    assert "scale={nw}:{nh}:out_range=tv" in gpu, "холст карты начинается так же, как процессорная цепочка"
    assert "-colorspace" not in " ".join(ps.clip_codec_args())
