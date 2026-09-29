#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер: фото-клипы Ken Burns на видеокарте (GPU_RENDER) против нынешнего
процессорного пути, на одних и тех же фото, тем же kenburns().

    python scripts/gpu_render_bench.py --photos <папка> [--n 20] [--dur 4]

Процессорный путь — пулом процессов на все ядра (как RENDER_WORKERS в
рендере), GPU-путь — потоками (подготовка на процессоре идёт внахлёст с
расчётом на карте). Печатает время, секунд видео в секунду и совпадение
готовых клипов (PSNR по яркости и цвету после кодирования)."""
import argparse
import concurrent.futures
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _render(args):
    photo, out, dur, gpu = args
    os.environ["GPU_RENDER"] = "1" if gpu else "0"
    import pipeline_smart as ps
    import gpu_render
    t = time.time()
    ok = ps.kenburns(photo, out, dur, section="BLOCK_1", motion_mode="classic_kb")
    return ok, time.time() - t, os.getpid(), dict(gpu_render.PROFILE)


def _cpu_init():
    import pipeline_smart as ps
    ps._render_worker_init(None, "cpu")


def _gpu_init():
    import pipeline_smart as ps
    ps._render_worker_init(None, "gpu")


def _psnr_pair(a, b):
    import numpy as np

    def dec(f):
        return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-f", "rawvideo",
                                             "-pix_fmt", "yuv420p", "-"], capture_output=True,
                                            check=True).stdout, np.uint8)
    x, y = dec(a), dec(b)
    n = min(x.size, y.size)
    fs = 1920 * 1080 * 3 // 2
    ny = 1920 * 1080
    fr = n // fs
    x, y = x[:fr * fs].reshape(fr, fs).astype(float), y[:fr * fs].reshape(fr, fs).astype(float)

    def p(d):
        return float(10 * np.log10(255 ** 2 / max((d ** 2).mean(), 1e-9)))
    return p(x[:, :ny] - y[:, :ny]), p(x[:, ny:] - y[:, ny:]), float((x[:, :ny] - y[:, :ny]).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--cpu-workers", type=int, default=os.cpu_count())
    ap.add_argument("--gpu-workers", type=int, default=0, help="0 — как в рендере (gpu_render_workers)")
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--profile", action="store_true",
                    help="время по этапам (синхронизирует карту после каждого этапа — общее время хуже)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    photos = sorted(glob.glob(os.path.join(a.photos, "*.jpg")) + glob.glob(os.path.join(a.photos, "*.png")))[:a.n]
    os.makedirs(os.path.join(a.out, "cpu"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "gpu"), exist_ok=True)
    sys.argv = ["pipeline_smart.py", a.out]
    os.environ["GPU_RENDER"] = "1"
    if a.profile:
        os.environ["GPU_RENDER_PROFILE"] = "1"
    import pipeline_smart as ps
    import torch
    # Кодер клипов решается так же, как в main() рендера (NVENC, если этот
    # ffmpeg им кодирует): без этого замер шёл на x264 даже на поде с NVENC.
    ps.resolve_clip_encoder()
    print(f"фото: {len(photos)}, клип {a.dur} с; процессор: {os.cpu_count()} ядер; "
          f"карта: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'нет'}; "
          f"кодер клипов: {ps.clip_codec_args()[1]}; GPU-путь активен: {ps.gpu_render_active()}; "
          f"ffmpeg: {subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split(chr(10))[0]}")
    report = {"photos": len(photos), "dur": a.dur, "cpu_cores": os.cpu_count()}
    video_sec = len(photos) * a.dur

    # Карта — пулом процессов с ролью «gpu», как в рендере (ClipRenderPools):
    # каждый процесс — свой контекст CUDA, самопроверка на первом клипе.
    import multiprocessing
    gw = a.gpu_workers or ps.gpu_render_workers()
    ex = concurrent.futures.ProcessPoolExecutor(gw, mp_context=multiprocessing.get_context("spawn"),
                                                initializer=_gpu_init)
    # Прогрев (запуск процессов, контекст CUDA, таблицы, самопроверка) — не
    # входит в замер, но печатается: это цена первого клипа каждого процесса.
    t = time.time()
    list(ex.map(_render, [(photos[k % len(photos)], os.path.join(a.out, f"warm{k}.mp4"), 1.0, True)
                          for k in range(gw)]))
    print(f"прогрев {gw} процесс(ов) карты: {time.time() - t:.1f} с")
    t = time.time()
    res = list(ex.map(_render, [(p, os.path.join(a.out, "gpu", f"{k:03d}.mp4"), a.dur, True)
                                for k, p in enumerate(photos)]))
    tg = time.time() - t
    ex.shutdown()
    prof = {}
    last = {}
    for r in res:
        last[r[2]] = r[3]
    for d in last.values():
        for k_, v in d.items():
            prof[k_] = prof.get(k_, 0.0) + v
    report["gpu"] = {"wall_sec": round(tg, 1), "ok": sum(1 for r in res if r[0] is True),
                     "workers": gw, "video_sec_per_sec": round(video_sec / tg, 2),
                     "profile_sec": {k_: round(v, 2) for k_, v in prof.items()}}
    print(f"ВИДЕОКАРТА ({gw} процесс(а)): {tg:.1f} с на {video_sec:.0f} с видео "
          f"({video_sec / tg:.2f} с видео/с), успешно {report['gpu']['ok']}/{len(photos)}")
    if prof:
        tot = sum(prof.values())
        print("  этапы (сумма по процессам, с синхронизацией карты): " +
              ", ".join(f"{k_} {v:.1f} с ({100 * v / tot:.0f}%)" for k_, v in
                        sorted(prof.items(), key=lambda kv: -kv[1])))

    t = time.time()
    with concurrent.futures.ProcessPoolExecutor(a.cpu_workers, mp_context=multiprocessing.get_context("spawn"),
                                                initializer=_cpu_init) as ex:
        res = list(ex.map(_render, [(p, os.path.join(a.out, "cpu", f"{k:03d}.mp4"), a.dur, False)
                                    for k, p in enumerate(photos)]))
    tc = time.time() - t
    report["cpu"] = {"wall_sec": round(tc, 1), "ok": sum(1 for r in res if r[0] is True),
                     "video_sec_per_sec": round(video_sec / tc, 2), "workers": a.cpu_workers}
    print(f"ПРОЦЕССОР ({a.cpu_workers} процессов): {tc:.1f} с ({video_sec / tc:.2f} с видео/с), "
          f"успешно {report['cpu']['ok']}/{len(photos)}")
    print(f"ускорение: {tc / tg:.1f}x")
    report["speedup"] = round(tc / tg, 2)

    rows = []
    for k in range(len(photos)):
        g, c = os.path.join(a.out, "gpu", f"{k:03d}.mp4"), os.path.join(a.out, "cpu", f"{k:03d}.mp4")
        if os.path.exists(g) and os.path.exists(c):
            rows.append(_psnr_pair(c, g))
            print(f"  {os.path.basename(photos[k])}: яркость {rows[-1][0]:.1f} дБ, цвет {rows[-1][1]:.1f}, сдвиг {rows[-1][2]:+.2f}")
    if rows:
        import numpy as np
        ys, uvs, bias = zip(*rows)
        report["parity"] = {"y_min": round(min(ys), 1), "y_median": round(float(np.median(ys)), 1),
                            "uv_min": round(min(uvs), 1), "bias_max": round(max(abs(b) for b in bias), 2)}
        print(f"совпадение (после кодирования): яркость мин {min(ys):.1f} / медиана {np.median(ys):.1f} дБ, "
              f"цвет мин {min(uvs):.1f} дБ, наибольший сдвиг {max(abs(b) for b in bias):.2f}")
    if a.json:
        json.dump(report, open(a.json, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
