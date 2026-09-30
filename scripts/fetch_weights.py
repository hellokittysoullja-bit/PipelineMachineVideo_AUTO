#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Веса моделей отбора — все сразу, параллельно.

Раньше подготовка пода качала модели одну за другой: замер на поде 29.09
(RTX 6000 Ada) — Qwen3-VL-Embedding-8B 2 мин 21 с, каждая из остальных
~38 с, всего сумма. Сети хватает на все потоки сразу (hf_transfer), поэтому
качаем одновременно, и время равно самой долгой модели.

    python scripts/fetch_weights.py            # Qwen-эмбеддинг, реранкер; WeMM — только если
                                               # прогон её позовёт (wemm_needed)
    python scripts/fetch_weights.py --wemm     # WeMM принудительно
    python scripts/fetch_weights.py --no-wemm  # без WeMM

HF_TOKEN из окружения (если задан) снимает ограничение скорости для
неавторизованных запросов."""
import argparse
import concurrent.futures
import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORKERS_PER_MODEL = 8


def wemm_needed():
    """Те же условия, что у pipeline_smart.cascade_model_needed(): каскад на
    WeMM выбран (CASCADE_MODEL=wemm9b) И судья работает (SHOT_JUDGE и ключ
    шлюза), потому что cascade_reorder зовётся только под судьёй. Без судьи
    17.6 ГБ скачивались зря (~1 минута подготовки пода 30.09); понадобится —
    модель докачается при первом вызове."""
    import feature_flags
    import wemm_embed
    return (wemm_embed.selected() and feature_flags.enabled("SHOT_JUDGE")
            and bool((os.environ.get("LLM_GATEWAY_API_KEY") or "").strip()))


def models(wemm=True):
    import qwen_vl_embed
    import qwen_vl_rerank
    import wemm_embed
    out = [(qwen_vl_embed.MODEL_NAME, qwen_vl_embed.REVISION), (qwen_vl_rerank.MODEL_NAME, qwen_vl_rerank.REVISION)]
    if wemm:
        out.append((wemm_embed.MODEL_NAME, wemm_embed.REVISION))
    return out


def fetch(name, revision):
    from huggingface_hub import snapshot_download
    t = time.time()
    snapshot_download(name, revision=revision, max_workers=WORKERS_PER_MODEL)
    return name, time.time() - t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wemm", action="store_true")
    ap.add_argument("--wemm", action="store_true", help="качать WeMM независимо от условий")
    a = ap.parse_args(argv)
    todo = models(wemm=False if a.no_wemm else (True if a.wemm else wemm_needed()))
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(len(todo)) as ex:
        futs = [ex.submit(fetch, n, r) for n, r in todo]
        for f in concurrent.futures.as_completed(futs):
            name, sec = f.result()
            print(f"  {name}: {sec:.0f} с", flush=True)
    print(f"все веса: {time.time() - t0:.0f} с", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
