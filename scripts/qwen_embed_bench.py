#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер скорости Qwen3-VL-Embedding на видеокарте: куда уходит время и
сколько картинок в секунду реально выходит при разных пачках.

    python scripts/qwen_embed_bench.py --images freeze94 [--n 1024]

ЗАЧЕМ. Первый живой прогон (29.09, H100 NVL) дал 38.8 картинки/с при пачке 16
— это примерно четверть того, что карта может посчитать. Прежде чем что-то
менять, надо знать, где время: подготовка картинки на процессоре (смена
размера), процессор модели (токены и нарезка картинки на патчи), перенос на
карту, сам расчёт. Замер отвечает числами на три вопроса:

  1. по этапам: сколько секунд на пачку уходит на каждый шаг;
  2. скорость при пачках 16/32/64/128 — последовательно и с подготовкой
     следующей пачки на процессоре, пока карта считает текущую;
  3. меняет ли пачка сам вектор: косинус с вектором той же картинки,
     посчитанным по одной (так считают гейты, на этом откалиброваны пороги).

Ничего не пишет, кроме отчёта в stdout (и --json).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def collect_images(root, n):
    """Первые n картинок из папки (по имени — детерминированно)."""
    from PIL import Image
    paths = []
    for dp, _dn, names in sorted(os.walk(root)):
        for name in sorted(names):
            paths.append(os.path.join(dp, name))
    out = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.load()
                out.append(im.copy())
        except Exception:  # noqa: BLE001 — в записи есть и не-картинки (JSON ответов)
            continue
        if len(out) >= n:
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batches", default="16,32,64,128")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    import qwen_vl_embed as qe

    t = time.time()
    if not qe.available():
        raise SystemExit(f"модель не загрузилась: {qe._STATE['broken']}")
    model, processor, dev = qe._STATE["model"], qe._STATE["processor"], qe._STATE["device"]
    report = {"load_sec": round(time.time() - t, 1), "gpu": torch.cuda.get_device_name(0)}
    print(f"модель загружена за {report['load_sec']} с на {report['gpu']}")

    imgs = collect_images(a.images, a.n)
    print(f"картинок: {len(imgs)}")

    def prep(part):
        prepared = [qe.prepare_image(im) for im in part]
        conv = [qe.conversation(image=im) for im in prepared]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=prepared, truncation=True, max_length=qe.MAX_LENGTH,
                           padding=True, do_resize=False, return_tensors="pt")
        return {k: v.pin_memory() if hasattr(v, "pin_memory") else v for k, v in inputs.items()}

    def forward(inputs):
        inputs = {k: v.to(dev, non_blocking=True) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs)
            emb = qe.pool_last(out.last_hidden_state, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
        return emb.cpu().numpy()

    # Прогрев: первые вызовы включают выбор ядер CUDA.
    forward(prep(imgs[:4]))
    torch.cuda.synchronize()

    # 1. По этапам на пачке 16 — как считает рендер сегодня.
    part = imgs[:16]
    t0 = time.time(); prepared = [qe.prepare_image(im) for im in part]; t1 = time.time()
    conv = [qe.conversation(image=im) for im in prepared]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=prepared, truncation=True, max_length=qe.MAX_LENGTH,
                       padding=True, do_resize=False, return_tensors="pt")
    t2 = time.time()
    g = {k: v.to(dev) for k, v in inputs.items()}
    torch.cuda.synchronize(); t3 = time.time()
    with torch.inference_mode():
        out = model(**g)
    torch.cuda.synchronize(); t4 = time.time()
    report["stages_bs16_sec"] = {"resize": round(t1 - t0, 3), "processor": round(t2 - t1, 3),
                                 "to_gpu": round(t3 - t2, 3), "model": round(t4 - t3, 3)}
    report["tokens_per_image"] = int(inputs["attention_mask"].sum() / len(part))
    print("этапы пачки 16 (с):", report["stages_bs16_sec"], "токенов на картинку:",
          report["tokens_per_image"])

    # 3. Эталон — по одной картинке (как гейты).
    ref_n = min(64, len(imgs))
    ref = np.concatenate([forward(prep([im])) for im in imgs[:ref_n]])

    report["speed"] = {}
    for bs in [int(x) for x in a.batches.split(",")]:
        parts = [imgs[k:k + bs] for k in range(0, len(imgs), bs)]
        try:
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            vecs = [forward(prep(p)) for p in parts]
            torch.cuda.synchronize()
            seq = len(imgs) / (time.time() - t)
            # Подготовка следующей пачки на процессоре, пока карта считает текущую.
            t = time.time()
            with ThreadPoolExecutor(2) as ex:
                fut = ex.submit(prep, parts[0])
                for k in range(len(parts)):
                    ready = fut.result()
                    if k + 1 < len(parts):
                        fut = ex.submit(prep, parts[k + 1])
                    forward(ready)
            torch.cuda.synchronize()
            ovl = len(imgs) / (time.time() - t)
        except torch.cuda.OutOfMemoryError:
            print(f"  пачка {bs}: не помещается в память")
            torch.cuda.empty_cache()
            continue
        v = np.concatenate(vecs)[:ref_n]
        cos = (v * ref).sum(1)
        row = {"seq_img_s": round(seq, 1), "overlap_img_s": round(ovl, 1),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "min_cos_vs_single": float(cos.min()), "max_abs_vs_single": float(abs(v - ref).max())}
        report["speed"][bs] = row
        print(f"  пачка {bs:4d}: {row['seq_img_s']:6.1f} к/с подряд, {row['overlap_img_s']:6.1f} к/с "
              f"с подготовкой внахлёст; память {row['peak_gib']} ГиБ; отличие от «по одной»: "
              f"мин. косинус {row['min_cos_vs_single']:.6f}, макс. {row['max_abs_vs_single']:.2e}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
