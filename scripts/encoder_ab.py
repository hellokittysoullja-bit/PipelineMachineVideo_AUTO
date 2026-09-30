#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Качество кодера клипа: NVENC (HEVC Main10, CQ NVENC_CQ) против x264 (CRF
RENDER_CRF, high10) на ОДНОМ и том же исходнике, по рецептам самого пайплайна.

Зачем: «NVENC CQ 16 с запасом к x264 CRF 17» в ветке GPU записано как оценка
без замера. Этот скрипт снимает замер там, где есть видеокарта:

    python scripts/encoder_ab.py [--source клип.mp4] [--seconds 6] [--out отчёт.json]

Без --source исходник синтезируется (движение + зернистый шум, самый тяжёлый
для кодера случай). Лучше передать настоящий клип из temp_smart.

Меряется: размер файла, время кодирования, SSIM и PSNR относительно исходника
(фильтры ffmpeg ssim/psnr, оба кодера сравниваются с ОДНИМ несжатым кадром).
Вердикт печатается по правилу: NVENC не хуже x264, если SSIM не ниже на
SSIM_TOL и размер не больше на SIZE_TOL. Порог — не оптимум, а граница
«незаметно», её решает владелец; отчёт хранит сами числа."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SSIM_TOL = 0.002
SIZE_TOL = 0.15


def _run(cmd, timeout=900):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


def synth_source(path, seconds):
    """Исходник без потерь: движущийся узор + шум (зерно), 1920x1080, 24 к/с."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
           f"testsrc2=s=1920x1080:r=24:d={seconds},noise=alls=12:allf=t",
           "-c:v", "ffv1", "-pix_fmt", "yuv420p10le", path]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"не удалось синтезировать исходник: {r.stderr[-300:]}")


def metric(ref, dist, name):
    """SSIM или PSNR (среднее по кадрам) dist относительно ref."""
    r = _run(["ffmpeg", "-hide_banner", "-i", dist, "-i", ref, "-lavfi",
              f"[0:v][1:v]{name}", "-f", "null", "-"])
    text = r.stderr
    if name == "ssim":
        m = re.search(r"SSIM .*All:([0-9.]+)", text)
    else:
        m = re.search(r"PSNR .*average:([0-9.inf]+)", text)
    return float(m.group(1)) if m and m.group(1) not in ("inf", ".") else None


def encode(label, args, source, out):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", source] + args + [out]
    t = time.time()
    r = _run(cmd)
    return {"label": label, "ok": r.returncode == 0, "sec": round(time.time() - t, 2),
            "bytes": os.path.getsize(out) if r.returncode == 0 and os.path.exists(out) else None,
            "error": None if r.returncode == 0 else (r.stderr or "")[-400:]}


def run(source=None, seconds=6):
    import pipeline_smart as ps
    report = {"source": source or "synthetic", "encoders": [], "verdict": None}
    with tempfile.TemporaryDirectory() as d:
        ref = source
        if ref is None:
            ref = os.path.join(d, "ref.mkv")
            synth_source(ref, seconds)
        variants = [("x264", ["-c:v", "libx264", "-preset", ps.RENDER_PRESET, "-crf", ps.RENDER_CRF]
                     + ps.CLIP_PIX_ARGS),
                    ("nvenc", list(ps.NVENC_CLIP_ARGS))]
        for label, args in variants:
            out = os.path.join(d, f"{label}.mp4")
            row = encode(label, args, ref, out)
            if row["ok"]:
                row["ssim"] = metric(ref, out, "ssim")
                row["psnr"] = metric(ref, out, "psnr")
            report["encoders"].append(row)
        x, n = report["encoders"]
        if x["ok"] and n["ok"] and x.get("ssim") and n.get("ssim"):
            better_q = n["ssim"] >= x["ssim"] - SSIM_TOL
            fair_size = n["bytes"] <= x["bytes"] * (1 + SIZE_TOL)
            report["verdict"] = {"nvenc_not_worse": bool(better_q and fair_size),
                                 "ssim_delta": round(n["ssim"] - x["ssim"], 5),
                                 "size_ratio": round(n["bytes"] / x["bytes"], 3),
                                 "speedup": round(x["sec"] / n["sec"], 2) if n["sec"] else None}
        elif not n["ok"]:
            report["verdict"] = {"nvenc_not_worse": None, "reason": "NVENC не закодировал: " + (n["error"] or "")}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source")
    ap.add_argument("--seconds", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = run(a.source, a.seconds)
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
