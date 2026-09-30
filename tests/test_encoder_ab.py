"""Скрипт замера кодеров: разбор метрик и вердикт (без видеокарты — ffmpeg подменён)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import encoder_ab as ab  # noqa: E402


class R:
    def __init__(self, err="", code=0):
        self.stderr, self.returncode, self.stdout = err, code, ""


def test_metric_parses_ssim_and_psnr(monkeypatch):
    monkeypatch.setattr(ab, "_run", lambda cmd, timeout=900: R(
        "[Parsed_ssim_0 @ 0x1] SSIM Y:0.99 U:0.99 V:0.99 All:0.987654 (19.1)\n"
        "[Parsed_psnr_0 @ 0x2] PSNR y:48.1 u:50 v:50 average:48.765432 min:40 max:60\n"))
    assert ab.metric("a", "b", "ssim") == 0.987654
    assert ab.metric("a", "b", "psnr") == 48.765432


def _fake(monkeypatch, x, n):
    """x, n — (bytes, ssim, sec, ok) для x264 и nvenc."""
    rows = {"x264": x, "nvenc": n}

    def encode(label, args, source, out):
        b, ssim, sec, ok = rows[label]
        if ok:
            open(out, "wb").write(b"0" * b)
        return {"label": label, "ok": ok, "sec": sec, "bytes": b if ok else None,
                "error": None if ok else "boom"}
    monkeypatch.setattr(ab, "encode", encode)
    monkeypatch.setattr(ab, "synth_source", lambda p, s: open(p, "wb").write(b"x"))
    ssim = {"x264": x[1], "nvenc": n[1]}
    monkeypatch.setattr(ab, "metric", lambda ref, dist, name:
                        ssim["x264" if "x264" in dist else "nvenc"] if name == "ssim" else 40.0)


def test_verdict_nvenc_not_worse(monkeypatch):
    _fake(monkeypatch, (1000, 0.990, 10.0, True), (1050, 0.989, 2.0, True))
    v = ab.run()["verdict"]
    assert v["nvenc_not_worse"] is True and v["speedup"] == 5.0


def test_verdict_nvenc_worse_by_quality_or_size(monkeypatch):
    _fake(monkeypatch, (1000, 0.990, 10.0, True), (1000, 0.970, 2.0, True))
    assert ab.run()["verdict"]["nvenc_not_worse"] is False
    _fake(monkeypatch, (1000, 0.990, 10.0, True), (1400, 0.990, 2.0, True))
    assert ab.run()["verdict"]["nvenc_not_worse"] is False


def test_verdict_when_nvenc_fails(monkeypatch):
    _fake(monkeypatch, (1000, 0.990, 10.0, True), (0, 0.0, 0.0, False))
    v = ab.run()["verdict"]
    assert v["nvenc_not_worse"] is None and "NVENC не закодировал" in v["reason"]
