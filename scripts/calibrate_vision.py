#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Калибровка порогов моделей зрения Qwen3-VL (GPU-ветка, vision_model.py).

    python scripts/calibrate_vision.py            # на видеокарте, 5-15 минут
    python scripts/calibrate_vision.py --dry-run  # посчитать и показать, не записывать

ЧТО ДЕЛАЕТ. Все пороги гейтов были откалиброваны на числах SigLIP2; у Qwen
шкала сходства другая. Скрипт прогоняет ТЕ ЖЕ функции модели, что решают в
рендере (pipeline_smart._gate_image_vec/_gate_text_vec — один вектор картинки
на все гейты; qwen_vl_rerank.score — вторая проверка), по размеченному
золотому набору (tests/fixtures/golden_set: 40 кадров опубликованного эп.01,
16 годных / 7 терпимых / 17 брака) и фикстурам мечей
(tests/fixtures/golden_media), и выставляет каждый порог по правилу,
которое записано у него в pipeline_smart:

  * НОЛЬ ПОТЕРЬ годных И терпимых кадров — порог ставится за самым слабым
    из них, с запасом BUFFER_STD_FRAC разброса сходства модели на наборе
    (у SigLIP2 запас был 0.0026-0.005 при разбросе 0.053 — те же 5-10%);
  * что при этом ловится из брака — печатается по каждой оси, число, а не
    обещание.

Ось без размеченных кадров под неё (например, ни одного годного кадра с
«музейным» запросом) получает порог ПЕРЕНОСОМ со шкалы SigLIP2 (z-перенос
по той же матрице набора) и помечается в отчёте как перенос, а не
калибровка. Частицы в кадре не размечены нигде — слой выключен (null).

Шкала сходства (среднее и разброс по матрице «все кадры × все запросы»)
пишется в файл: через неё переносятся некалиброванные разметкой константы
(порог домена у лука, веса режиссёра). Порог режиссёра — по его парам
tests/fixtures/director_calibration (та же методика, что была).

Файл калибровки — assets/calibration/vision_qwen3vl.json (или
VISION_CALIBRATION). Он привязан к моделям и протоколу вопросов
(vision_model.current_signature): другая модель, другое разрешение картинки
— рендер откажет и попросит перекалибровать.

ЧЕСТНЫЕ ПРЕДЕЛЫ. 40 кадров — малая выборка (метки: глаза, см. manifest
«labelled_by»); правило «ноль потерь годных» на ней держит пороги
консервативными, то есть брак ловится меньше, чем мог бы. Кадры набора —
один эпизод и одна ниша. Качество порядка кандидатов (каскад) этим скриптом
не меряется — это cascade_model_eval.py на записи эпизода 94.
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
GOLDEN = os.path.join(REPO, "tests", "fixtures", "golden_set")
MEDIA = os.path.join(REPO, "tests", "fixtures", "golden_media")
BUFFER_STD_FRAC = 0.05
KEEP = ("good", "tolerable")
# Мечи из фикстур для гварда культуры клинка (форма клинка не зависит от
# запроса: euro_prompt против asian_prompt на самой картинке).
GUARD_EUROPEAN = ("sword.jpg", "euro_sword_2.jpg")
GUARD_FOREIGN = ("katana.jpg",)
# AUC (годное выше брака) прежних моделей на этом наборе — из истории
# калибровок в pipeline_smart (замеры 17.09): база для сравнения.
LEGACY_AUC = {"CLIP ViT-B/32": 0.566, "SigLIP2-base-256": 0.658, "SigLIP2-so400m+Jina": 0.610}


def auc(pos, neg):
    """Доля пар (годный, брак), где годный оценён выше (ничья — половина)."""
    if not pos or not neg:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else (0.5 if a == b else 0.0)
    return s / (len(pos) * len(neg))


def mean_std(values):
    n = len(values)
    m = sum(values) / n
    return m, (sum((v - m) ** 2 for v in values) / n) ** 0.5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sys.argv = ["pipeline_smart.py", REPO]
    import pipeline_smart as ps
    import qwen_vl_embed
    import qwen_vl_rerank
    import vision_model
    import visual_director as vd

    import ml_device
    if ml_device.device() != "cuda":
        raise SystemExit(f"нужна видеокарта CUDA (устройство моделей: {ml_device.device()})")
    if not qwen_vl_embed.available():
        raise SystemExit(f"Qwen3-VL-Embedding не загрузилась: {qwen_vl_embed._STATE['broken']}")
    if not qwen_vl_rerank.available():
        raise SystemExit(f"Qwen3-VL-Reranker не загрузился: {qwen_vl_rerank.broken_reason()}")

    items = json.load(open(os.path.join(GOLDEN, "manifest.json"), encoding="utf-8"))["items"]
    img = [os.path.join(GOLDEN, it["image"]) for it in items]

    def iv(path):
        v = ps._gate_image_vec(path)
        if v is None:
            raise SystemExit(f"модель не ответила на {path}")
        return v

    def tv(text):
        v = ps._gate_text_vec(text)
        if v is None:
            raise SystemExit(f"модель не ответила на текст {text!r}")
        return v

    print(f"Золотой набор: {len(items)} кадров")
    I = [iv(p) for p in img]
    Q = [tv(it["query"]) for it in items]
    R = [tv(it["text"]) for it in items]
    gate_matrix = [float(x @ q) for x in I for q in Q]
    sent_matrix = [float(x @ r) for x in I for r in R]
    g_mean, g_std = mean_std(gate_matrix)
    s_mean, s_std = mean_std(sent_matrix)
    buf = BUFFER_STD_FRAC * g_std
    print(f"шкала гейтов: среднее {g_mean:.4f}, разброс {g_std:.4f} (запас порогов {buf:.4f})")

    keep = [k for k, it in enumerate(items) if it["verdict"] in KEEP]
    rej = [k for k, it in enumerate(items) if it["verdict"] == "reject"]
    good = [k for k, it in enumerate(items) if it["verdict"] == "good"]
    rel = [float(I[k] @ Q[k]) for k in range(len(items))]

    report, thresholds, transferred = {}, {}, []

    # Сохраняем шкалу сразу: переносы ниже её читают.
    cal = {"signature": vision_model.current_signature(),
           "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "scale": {"gate": {"mean": g_mean, "std": g_std},
                     "sentence": {"mean": s_mean, "std": s_std}},
           "thresholds": {}}

    def transfer_margin(value):
        return value * g_std / vision_model.LEGACY_GATE_SCALE["std"]

    # 1. Релевантность кадра запросу.
    thr = min(rel[k] for k in keep) - buf
    thresholds["relevance"] = thr
    report["relevance"] = {"caught_reject": sum(1 for k in rej if rel[k] < thr), "n_reject": len(rej),
                           "auc_good_vs_reject": auc([rel[k] for k in good], [rel[k] for k in rej])}

    # 2. Запас «собирательного» запроса (музей, витрина...).
    anchor = tv(ps.NEGATIVE_ANCHOR_PROMPT)
    risky = [k for k in range(len(items)) if ps.is_risky_query(items[k]["query"])]
    rmargin = {k: rel[k] - float(I[k] @ anchor) for k in risky}
    rkeep = [rmargin[k] for k in risky if k in keep]
    if rkeep:
        thr = min(rkeep) - buf
        report["risky_margin"] = {"caught_reject": sum(1 for k in risky if k in rej and rmargin[k] < thr),
                                  "n_reject": sum(1 for k in risky if k in rej), "n_keep": len(rkeep)}
    else:
        thr = transfer_margin(0.045)
        transferred.append("risky_margin")
        report["risky_margin"] = {"transferred_from_siglip2": 0.045}
    thresholds["risky_margin"] = thr

    # 3. Контрастивное вето по ловушкам канала.
    traps = [tv(t) for t in ps.CONTENT_NEGATIVE_ANCHORS]
    if traps:
        vmargin = [rel[k] - max(float(I[k] @ t) for t in traps) for k in range(len(items))]
        thr = min(vmargin[k] for k in keep) - buf
        report["negative_veto_margin"] = {
            "caught_reject": sum(1 for k in rej if vmargin[k] < thr), "n_reject": len(rej),
            "worst_keep": min(vmargin[k] for k in keep)}
    else:
        thr = transfer_margin(-0.06)
        transferred.append("negative_veto_margin")
    thresholds["negative_veto_margin"] = thr

    # 4. Гварды домена (форма клинка): годные европейские клинки набора, если
    # их запрос включает гвард, плюс фикстуры мечей.
    guards = {}
    for g in ps._PROFILE_DOMAIN_GUARDS:
        e, s_ = tv(g["euro_prompt"]), tv(g["asian_prompt"])

        def margin(v):
            return float(v @ e) - float(v @ s_)
        trig = [k for k in range(len(items))
                if any(t in items[k]["query"].lower() for t in g["trigger_terms"])]
        keep_m = [margin(I[k]) for k in trig if k in keep]
        keep_m += [margin(iv(os.path.join(MEDIA, f))) for f in GUARD_EUROPEAN
                   if os.path.exists(os.path.join(MEDIA, f))]
        foreign_m = [margin(iv(os.path.join(MEDIA, f))) for f in GUARD_FOREIGN
                     if os.path.exists(os.path.join(MEDIA, f))]
        foreign_m += [margin(I[k]) for k in trig if items[k].get("reject_reason") == "non_european"]
        if keep_m:
            thr = min(keep_m) - buf
            guards[g["name"]] = thr
            report[f"domain_guard:{g['name']}"] = {
                "n_keep": len(keep_m), "caught_foreign": sum(1 for m in foreign_m if m < thr),
                "n_foreign": len(foreign_m)}
        else:
            guards[g["name"]] = None
            report[f"domain_guard:{g['name']}"] = {"disabled": "нет годных кадров под этот гвард"}
    thresholds["domain_guard"] = guards

    # 5. Вторая проверка победителя — реранкер.
    scores = []
    for k, it in enumerate(items):
        got = qwen_vl_rerank.score(it["query"], [img[k]])
        if not got or got[0] is None:
            raise SystemExit(f"реранкер не ответил на {img[k]}")
        scores.append(got[0])
    r_std = mean_std(scores)[1]
    thr = min(scores[k] for k in keep) - BUFFER_STD_FRAC * r_std
    thresholds["smart_rerank"] = thr
    report["smart_rerank"] = {"caught_reject": sum(1 for k in rej if scores[k] < thr), "n_reject": len(rej),
                              "auc_good_vs_reject": auc([scores[k] for k in good], [scores[k] for k in rej])}

    # 6. Частицы — разметки нет нигде.
    thresholds["particle"] = None

    # 7. Порог режиссёра — по его парам, на шкале прежнего ансамбля.

    def sentence_legacy(image_path, text):
        raw = float(iv(image_path) @ tv(text))
        legacy = vision_model.LEGACY_SENTENCE_SCALE
        return (raw - s_mean) / s_std * legacy["std"] + legacy["mean"]
    floor = vd.calibrate_relevance_floor(score_fn=sentence_legacy)
    if floor and floor["margin"] >= vd.DIRECTOR_RELEVANCE_MIN_MARGIN:
        thresholds["director_floor"] = floor["floor"]
    else:
        thresholds["director_floor"] = None
    report["director_floor"] = floor and {k: floor[k] for k in ("floor", "margin", "n_pairs")}
    cal["thresholds"] = thresholds
    cal["report"] = dict(report, transferred=transferred, legacy_auc=LEGACY_AUC,
                         buffer_std_frac=BUFFER_STD_FRAC, n_items=len(items))

    print(json.dumps({"thresholds": thresholds, "report": cal["report"]}, ensure_ascii=False,
                     indent=1))
    if a.dry_run:
        print("--dry-run: файл не записан")
        return 0
    out = a.out or vision_model.calibration_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=1)
    os.replace(out + ".tmp", out)
    print(f"Калибровка записана: {out}")

    # Проверка тем же замером, что держит золотой набор в тестах: гейты
    # рендера с новыми порогами. Сводка ложится рядом с прежней базовой
    # линией SigLIP2 — для сравнения, не вместо неё.
    if a.out:
        os.environ["VISION_CALIBRATION"] = out
    ps.reload_vision_thresholds()
    import golden_set_eval as gse
    rows, _canary = gse.evaluate(items)
    summary = gse.summarize(rows)
    path = os.path.join(REPO, "docs", "quality", "golden_set_baseline_qwen3vl.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "model": vision_model.current_signature(),
                   "summary": summary}, f, ensure_ascii=False, indent=1)
    print(f"Золотой набор на Qwen: пропуск брака {summary['reject_leak']}, анахронизмов "
          f"{summary['anachronism_leak']}, ложный отказ годным {summary['good_false_reject']} "
          f"(SigLIP2: 0.7059 / 0.6364 / 0.0) — {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
