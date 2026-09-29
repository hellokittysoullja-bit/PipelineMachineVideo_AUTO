#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер финальной склейки (xfade_chain_chunked) на ролике заданной длины.

    python scripts/final_pass_bench.py --clips <папка .mp4> --minutes 20 [--workers N]

Клипы папки повторяются по кругу до нужной длины, секции меняются каждые
~8 клипов (как у эпизода), склейка — тем же кодом и тем же кодером
поставки, что рендер. Печатает время, секунд видео в секунду и число
кусков."""
import argparse
import glob
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--workers", default="")
    a = ap.parse_args(argv)
    if a.workers:
        os.environ["FINAL_CHUNK_WORKERS"] = a.workers
    sys.argv = ["pipeline_smart.py", tempfile.mkdtemp(prefix="fpb_")]
    import pipeline_smart as ps
    src = sorted(glob.glob(os.path.join(a.clips, "*.mp4")))
    durs0 = {c: ps.quantize_durations_to_frames([ps.get_media_duration(c)])[0] for c in src}
    clips, total = [], 0.0
    while total < a.minutes * 60:
        c = src[len(clips) % len(src)]
        clips.append(c)
        total += durs0[c]
    durs = [durs0[c] for c in clips]
    sections = [f"B{k // 8}" for k in range(len(clips))]
    blocks = [{"section": s_, "text": f"фраза {k}", "words": 3, "is_subcut": False}
              for k, s_ in enumerate(sections)]
    out = os.path.join(sys.argv[1], "merged.mp4")
    print(f"клипов {len(clips)}, видео {total / 60:.1f} мин, ядер {os.cpu_count()}, "
          f"кусков одновременно {ps.final_chunk_workers(max(2, len(clips) // 2))}")
    t = time.time()
    ok, dur = ps.xfade_chain_chunked(clips, durs, sections, out, sys.argv[1], blocks=blocks)
    dt = time.time() - t
    print(f"склейка: {'готово' if ok else 'СБОЙ'} за {dt:.1f} с, {dur / max(dt, 1e-9):.1f} с видео/с, "
          f"длительность {dur:.1f} с")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
