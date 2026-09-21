#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Папка-отчёт по ГОТОВОМУ ролику — одна команда, чтобы смотреть на результат,
а не спорить о нём словами.

ЗАЧЕМ. Три сессии подряд каждый разбор подбора кадров заканчивался словами
«нужен прогон на машине владельца»: у обеих сторон разбора не было способа
посмотреть на результат, и обе по разу ошиблись именно из-за этого — одна
считала сырой result_count выдачи вместо отфильтрованного, другая описывала
механизмы, которых в коде нет, и не видела тех, что есть. Отчёты эпизода при
этом уже существовали, но лежали по восьми файлам, и ни один не отвечал на
вопросы «сколько карточек», «как скачет тон между планами», «какой длины
планы» ОДНИМ числом.

ЧТО ЗДЕСЬ И ОТКУДА (стадия у каждой оси названа явно — по правилу
стадийности из CLAUDE.md, чтобы «работает механизм» и «механизм есть и не
справляется» не смешивались):

* ДЛИТЕЛЬНОСТИ ПЛАНОВ — по резам, найденным В ПИКСЕЛЯХ final.mp4
  (verify_timing.cuts_from_curve, тот же откалиброванный детектор). Не по
  плану монтажа: план — модель, файл — факт. Честный предел тот же, что у
  verify_timing: диссолв может не дать пика, поэтому покрытие (найдено
  планов / планов в шотлисте) печатается рядом, и низкое покрытие — это
  ОТСУТСТВИЕ измерения, а не «планы длинные».
* СКАЧКИ ТОНА МЕЖДУ СОСЕДНИМИ ПЛАНАМИ — средняя яркость (Rec.709 Y) и
  средняя насыщенность кадра из СЕРЕДИНЫ каждого найденного плана, на том же
  файле. Это ОСТАТОЧНЫЙ скачок после всей цепочки (грейд, EMA, Look
  Management, зерно) — то, что видит зритель. Замер 51→16 в CLAUDE.md был
  СИМУЛЯЦИЕЙ цепочки на исходниках; здесь — рендер.
* ДОЛЯ КАРТОЧЕК — media_plan/fallback_cards_report.json, по причинам
  (карточка за брак и карточка «нет медиа» — разные гарантии и разные
  счётчики, см. FALLBACK_NO_MEDIA_REASON в pipeline_smart.py).
* ЛОГ ОТКАЗОВ — relevance_gate / stock_exhausted / arbiter_rejected /
  director_relevance / render_manifest / render_qc, сведённые по причине и
  виду. Стадия: отбор и рендер, не готовый файл.
* ИСТОЧНИКИ И ВИДЫ — media_plan/shotlist.json (source/kind) плюс шапка
  gates: какие модели и ключи РЕАЛЬНО работали в этом прогоне.
* КОНТАКТНЫЙ ЛИСТ — те же страницы, что scripts/shotlist_contact.py,
  положенные в папку отчёта.

Один проход декодирования в 64x36 даёт и кривую разниц (резы), и яркость, и
насыщенность каждого кадра — второй проход не нужен.

ДЕТЕРМИНИЗМ. В pipeline_smart.py нет ни одного вызова random (проверено
grep'ом 14.09): при тех же входах и том же кэше рендер побайтово тот же,
«фиксированные семена» ему не нужны. Единственный seed — _stable_seed(имя
секции) у атмосферного слоя, он и так фиксирован. Что НЕ фиксировано —
живая выдача источников и версии ML-стека; обе пишутся в отчёт, чтобы два
отчёта можно было сравнивать честно.

Использование:
    python scripts/segment_report.py <video_dir> [--video final.mp4] [--no-contact]
Результат: <video_dir>/media_plan/segment_report/report.json + contact_NN.jpg.
render_episode.py вызывает это последним шагом сам.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCAN_W, SCAN_H = 64, 36
# Планы короче этого читаются как дёрганье, длиннее — как слайд (обе границы
# — ориентиры для чтения отчёта, не гейты; менять их — не менять ролик).
SHORT_PLAN_SEC = 1.5
LONG_PLAN_SEC = 12.0
# Ниже этого покрытия распределение длительностей НЕ считается измеренным —
# тот же порог, что у verify_timing.MIN_COVERAGE.
MIN_CUT_COVERAGE = 0.5
# Покрытие ВЫШЕ этого — резов найдено заметно больше, чем слотов в шотлисте.
# Замер 14.09 (videos/_test60s): 10 планов на 9 слотов, покрытие 1.111.
# Причина не в монтаже — у стокового ВИДЕО бывает своя внутренняя склейка,
# и детектор честно видит её как рез. Следствие важное: «план» в этом отчёте
# перестаёт быть синонимом слота, а «скачок тона между СОСЕДНИМИ планами»
# частично меряет перепад ВНУТРИ одного клипа. Молчать об этом нельзя — на
# том же прогоне два «плана» оказались одним клипом на 6.9с, и прочитать это
# по отчёту было невозможно.
MAX_CUT_COVERAGE = 1.05

REPORT_FILES = ("relevance_gate_report.json", "stock_exhausted_report.json",
                "arbiter_rejected_report.json", "director_relevance_report.json")


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _dist(xs):
    if not xs:
        return {"n": 0}
    mean = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return {"n": len(xs), "min": round(min(xs), 3), "p10": round(_pct(xs, 0.10), 3),
            "median": round(statistics.median(xs), 3), "p90": round(_pct(xs, 0.90), 3),
            "max": round(max(xs), 3), "mean": round(mean, 3),
            "cv": round(sd / mean, 3) if mean else None}


def scan_frames(video_path):
    """Один проход: (fps, [медианная разница с предыдущим кадром],
    [средняя яркость Y], [средняя насыщенность]) по каждому кадру.

    Разница считается ПО ЦВЕТУ и медианой — ровно как verify_timing.frame_
    diff_curve (тот же калиброванный пример: красный→зелёный при равной Y).
    """
    import numpy as np
    import verify_timing as vt
    fps = vt.video_fps(video_path) or 24.0
    cmd = [vt.FFMPEG, "-v", "error", "-i", video_path, "-an",
           "-vf", f"scale={SCAN_W}:{SCAN_H},format=rgb24", "-f", "rawvideo", "-"]
    nbytes = SCAN_W * SCAN_H * 3
    diffs, lumas, sats = [], [], []
    prev = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            cur = np.frombuffer(buf, dtype=np.uint8).astype(np.int16).reshape(-1, 3)
            if prev is not None:
                diffs.append(float(np.median(np.abs(cur - prev))))
            prev = cur
            f = cur.astype(np.float32)
            lumas.append(float((0.2126 * f[:, 0] + 0.7152 * f[:, 1] + 0.0722 * f[:, 2]).mean()))
            mx, mn = f.max(axis=1), f.min(axis=1)
            sats.append(float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0).mean()))
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait(timeout=vt._timeout_for(video_path))
    return fps, diffs, lumas, sats


def analyze_video(video_path, expected_plans=None):
    """Длительности планов и скачки тона — только из пикселей файла."""
    import verify_timing as vt
    fps, diffs, lumas, sats = scan_frames(video_path)
    n_frames = len(lumas)
    if n_frames == 0:
        return {"status": "no_frames"}
    cuts, _ = vt.cuts_from_curve(fps, diffs)
    total = n_frames / fps
    bounds = [0.0] + [c for c in cuts if 0.0 < c < total] + [total]
    spans = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    durs = [b - a for a, b in spans]
    mid_luma, mid_sat = [], []
    for a, b in spans:
        k = min(n_frames - 1, max(0, int(((a + b) / 2.0) * fps)))
        mid_luma.append(lumas[k])
        mid_sat.append(sats[k] * 255.0)   # в тех же единицах 0..255, что замер 51→16
    luma_jumps = [abs(x - y) for x, y in zip(mid_luma, mid_luma[1:])]
    sat_jumps = [abs(x - y) for x, y in zip(mid_sat, mid_sat[1:])]
    coverage = None
    if expected_plans:
        coverage = round(len(spans) / float(expected_plans), 3)
    if coverage is None:
        status = "measured"
    elif coverage < MIN_CUT_COVERAGE:
        status = "low_coverage"
    elif coverage > MAX_CUT_COVERAGE:
        status = "over_coverage"
    else:
        status = "measured"
    return {
        "status": status,
        "stage": "пиксели final.mp4 — остаток после всей цепочки рендера",
        "fps": fps, "duration_sec": round(total, 3),
        "plans_found": len(spans), "plans_expected": expected_plans, "cut_coverage": coverage,
        "plan_duration_sec": _dist(durs),
        "short_plans_lt_%.1fs" % SHORT_PLAN_SEC: sum(1 for d in durs if d < SHORT_PLAN_SEC),
        "long_plans_gt_%.0fs" % LONG_PLAN_SEC: sum(1 for d in durs if d > LONG_PLAN_SEC),
        "luma_jump_0_255": _dist(luma_jumps),
        "saturation_jump_0_255": _dist(sat_jumps),
        "plans": [{"start": round(a, 3), "end": round(b, 3), "luma": round(l, 1), "sat": round(s, 1)}
                  for (a, b), l, s in zip(spans, mid_luma, mid_sat)],
        "limits": ["диссолв может не дать пика: низкое покрытие — отсутствие измерения, "
                   "а не длинные планы",
                   "покрытие > %.2f: резов больше, чем слотов — у стокового видео своя "
                   "внутренняя склейка, и «план» тут НЕ равен слоту" % MAX_CUT_COVERAGE,
                   "тон — один кадр из середины плана, не среднее по плану",
                   "квантование — один кадр"],
    }


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_artifacts(video_dir):
    """Карточки, отказы, источники — из уже написанных отчётов эпизода."""
    mp = os.path.join(video_dir, "media_plan")
    out = {"stage": "отбор и рендер (media_plan/*), не готовый файл"}

    shotlist = _load(os.path.join(mp, "shotlist.json")) or {}
    shots = [s for s in shotlist.get("shots", []) if isinstance(s, dict)]
    n = len(shots)
    out["n_shots"] = n
    out["gates"] = shotlist.get("gates", {})
    by_source, by_kind, by_provider = {}, {}, {}
    rel_known = 0
    for s in shots:
        by_source[s.get("source") or "unknown"] = by_source.get(s.get("source") or "unknown", 0) + 1
        by_kind[s.get("kind") or "unknown"] = by_kind.get(s.get("kind") or "unknown", 0) + 1
        # provider — КТО принёс кадр (met/cleveland/chicago/openverse/pixabay/
        # unsplash/pexels), source — КАК слот разрешён (picked/local/lock/
        # cache_hit/missing). Раньше здесь была только вторая ось, и в ней всё
        # подобранное называлось «pexels»: на videos/_test60s это давало
        # «pexels: 10» при пяти реально победивших источниках.
        p = s.get("provider")
        if p:
            by_provider[p] = by_provider.get(p, 0) + 1
        if s.get("relevance") is not None:
            rel_known += 1
    out["by_source"] = by_source
    out["by_kind"] = by_kind
    out["by_provider"] = by_provider
    # Слоты без провенанса — не ноль и не ошибка: кадр мог быть скачан до
    # появления sidecar. Называем их числом, а не молчанием.
    out["provenance_unknown"] = n - sum(by_provider.values())
    out["relevance_known"] = rel_known

    cards = _load(os.path.join(mp, "fallback_cards_report.json")) or {}
    misses = cards.get("misses") or []
    by_reason = {}
    for m in misses:
        by_reason[m.get("reason") or "unknown"] = by_reason.get(m.get("reason") or "unknown", 0) + 1
    out["cards"] = {"n": len(misses), "share": round(len(misses) / n, 4) if n else None,
                    "by_reason": by_reason, "indices": sorted(m.get("index") for m in misses)}
    rescue = _load(os.path.join(mp, "video_photo_rescue_report.json")) or {}
    out["video_rescued_by_photo"] = len(rescue.get("misses") or [])

    rejections = {}
    for name in REPORT_FILES:
        rep = _load(os.path.join(mp, name)) or {}
        for m in rep.get("misses") or []:
            key = f"{name[:-5]}:{m.get('kind', '?')}"
            rejections[key] = rejections.get(key, 0) + 1
    rm = _load(os.path.join(mp, "render_manifest.json")) or {}
    # render_manifest несёт ЧЕТЫРЕ статуса (ok/failed/absorbed/
    # skipped_selection_only), не два — прежний `!= "ok"` считал
    # ПОГЛОЩЁННЫЙ слот (NEVER_SHOW_KNOWN_BAD, 21.09, намеренный и
    # успешный исход) и слот SELECTION_ONLY-прогона (рендера вообще не
    # было по замыслу режима) за ту же "render_manifest:failed", что и
    # настоящий сорванный рендер — тот самый класс, из-за которого этот
    # репозиторий уже когда-то завёл отдельные коды возврата EXIT_OK/
    # EXIT_NOT_BUILT/EXIT_BUILT_WITH_WARNINGS (см. комментарий у них в
    # pipeline_smart.py: "render_episode.py писал status='failed' для
    # совершенно нормального рендера с парой похожих кадров"). Поглощение
    # уже честно посчитано выше через by_source["absorbed"] — дважды, да
    # ещё и под именем "failed", его считать не нужно.
    failed = [c for c in rm.get("clips") or []
              if c.get("status") not in ("ok", "absorbed", "skipped_selection_only")]
    if failed:
        rejections["render_manifest:failed"] = len(failed)
    qc = _load(os.path.join(mp, "render_qc_report.json")) or {}
    for k in ("duplicates", "dupes", "issues"):
        if isinstance(qc.get(k), list) and qc[k]:
            rejections[f"render_qc:{k}"] = len(qc[k])
    out["rejections"] = rejections
    out["known_bad_slots"] = sorted({m.get("index") for name in REPORT_FILES
                                     for m in (_load(os.path.join(mp, name)) or {}).get("misses") or []
                                     if m.get("index") is not None})
    return out


def _stack_versions():
    v = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "numpy", "PIL"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception:
            v[mod] = None
    return v


def contact_sheets(video_dir, out_dir, cols=4, per_page=24):
    """Те же страницы, что shotlist_contact.py, в папку отчёта.

    ВТОРОЙ, независимый вызов render_page() — тот же класс расхождения,
    которым этот репозиторий уже горел не раз (filter_alt_blocklist на
    видео-пути, director_score_fn, бриф стокам): правка в главной точке
    входа (shotlist_contact.main()) не дошла бы сюда сама по себе. Без
    cover_map поглощённые слоты (NEVER_SHOW_KNOWN_BAD, 21.09) на ЭТОМ
    контактном листе снова рисовались бы красной заливкой «НЕТ ФАЙЛА»,
    хотя в final.mp4 они никогда не пустуют — ровно тот баг, который
    21.09 нашли и закрыли в shotlist_contact.main(), но не здесь."""
    import shotlist_contact as sc
    data = _load(os.path.join(video_dir, "media_plan", "shotlist.json")) or {}
    shots = sorted([s for s in data.get("shots", []) if isinstance(s, dict)],
                   key=lambda s: s.get("index", 0))
    cover_map = sc.build_absorption_cover_map(shots)
    pages = []
    for p in range(0, len(shots), per_page):
        out = os.path.join(out_dir, f"contact_{p // per_page + 1:02d}.jpg")
        pages.append(sc.render_page(shots[p:p + per_page], video_dir, cols, out, cover_map))
    return pages


def build(video_dir, video_path=None, with_contact=True):
    video_path = video_path or os.path.join(video_dir, "final.mp4")
    out_dir = os.path.join(video_dir, "media_plan", "segment_report")
    os.makedirs(out_dir, exist_ok=True)
    report = {"schema_version": 1, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "video": os.path.basename(video_path), "stack": _stack_versions(),
              "determinism": "в pipeline_smart.py нет random — при тех же входах и кэше рендер "
                             "побайтово тот же; не фиксированы живая выдача источников и версии стека"}
    report["artifacts"] = collect_artifacts(video_dir)
    if os.path.exists(video_path):
        try:
            report["video_measured"] = analyze_video(
                video_path, expected_plans=report["artifacts"].get("n_shots") or None)
        except Exception as e:
            report["video_measured"] = {"status": "not_measured",
                                        "error": f"{type(e).__name__}: {e}"}
    else:
        report["video_measured"] = {"status": "no_video"}
    if with_contact:
        try:
            report["contact_pages"] = contact_sheets(video_dir, out_dir)
        except Exception as e:
            report["contact_pages"] = []
            report["contact_error"] = f"{type(e).__name__}: {e}"
    path = os.path.join(out_dir, "report.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return report, path


def summarize(report):
    a, v = report.get("artifacts", {}), report.get("video_measured", {})
    lines = [f"Слотов: {a.get('n_shots')}  источники: {a.get('by_source')}  виды: {a.get('by_kind')}",
             f"Карточек: {a.get('cards', {}).get('n')} ({a.get('cards', {}).get('share')}) "
             f"по причинам {a.get('cards', {}).get('by_reason')}",
             f"Отказы: {a.get('rejections')}"]
    if v.get("status") in ("measured", "low_coverage", "over_coverage"):
        d = v["plan_duration_sec"]
        lines.append(f"Планов в пикселях: {v['plans_found']} (покрытие {v['cut_coverage']}, "
                     f"{v['status']}); длительность медиана {d.get('median')}с "
                     f"[{d.get('p10')}..{d.get('p90')}], cv {d.get('cv')}")
        lines.append(f"Скачок тона между соседними планами (0..255): яркость медиана "
                     f"{v['luma_jump_0_255'].get('median')} p90 {v['luma_jump_0_255'].get('p90')}; "
                     f"насыщенность медиана {v['saturation_jump_0_255'].get('median')} "
                     f"p90 {v['saturation_jump_0_255'].get('p90')}")
    else:
        lines.append(f"Видео не измерено: {v.get('status')} {v.get('error', '')}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video_dir")
    ap.add_argument("--video", default=None, help="по умолчанию <video_dir>/final.mp4")
    ap.add_argument("--no-contact", action="store_true")
    args = ap.parse_args(argv)
    report, path = build(args.video_dir, args.video, with_contact=not args.no_contact)
    print(summarize(report))
    print(f"Отчёт: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
