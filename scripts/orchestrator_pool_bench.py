#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Бесплатная зона: какие кадры приносят запросы оркестратора — живой пул, прод-код.

ЗАЧЕМ. После 25-го слота судьи нет (SHOT_JUDGE_PAID_SLOTS). Отбор берёт
первые FAST_PHOTO_DEDUP_MAX_TRIES (5) кандидатов кучи в порядке её сборки —
первые ответы первых источников по запросам задания фразы — и выбирает
среди них гейтами. Каскада там нет. То есть на ~90% кадров ролика решают
ЗАПРОСЫ задания и их типы (маршрут источников), а не судья. Текстовый
замер (orchestrator_bench.py) говорит, понял ли оркестратор фразу; этот —
что от этого приходит в кучу и что встаёт на экран.

КАК. Для фраз с ожиданиями (gold.json) тех же ниш ТЕ ЖЕ прод-функции —
selection_engine.build_pool и select_media — получают задание одной или
другой руки (план v3 или v5 с диска). Отличается только задание: запрос
слота, запросы пула и типы запросов. Индекс слота — в бесплатной зоне.
У каждой руки свой анти-дубль. Запросы секции сценария в кучу не
подмешиваются: меряется задание, а не авторские запросы главы.

КОМАНДЫ
  run    EP_DIR --gold GOLD --ep EP --arm v3=PLAN.json --arm v5=PLAN.json --out OUT [--index N]
                          --index меньше SHOT_JUDGE_PAID_SLOTS — платная зона: каскад и
                          судья работают, как в рендере (платно, по ключу шлюза)
  sheet  OUT              слепые листы: кандидаты обеих рук по фразе вперемешку,
                          без имени руки; номер плитки -> id в OUT/sheets/index.json
  score  OUT LABELS       LABELS — {"<фраза #>": {"<плитка>": 0|1|2}}; итог по рукам

ЧЕСТНЫЕ ПРЕДЕЛЫ. Только фото. Метки — глаза Claude, а не владельца. Живая
выдача источников плывёт во времени — руки гоняются в одном процессе
подряд по каждой фразе, чтобы сравнивать одну и ту же выдачу, порядок рук
чередуется по фразам. Руку, прогнанную отдельным процессом позже, с
остальными сравнивать нельзя: квота Pexels и троттлинг Мет и Commons
меняются за минуты (первый прогон так и сломался — у поздней руки Мет дал
треть кандидатов). Перед рукой прогон ждёт здоровых источников, сбои за
руку пишутся в source_trouble, итог считается и по «чистым» фразам.
Победитель слота здесь — то, что выбрал прод-отбор из пятёрки; ролика не
собирается.
"""
import argparse
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
FREE_INDEX = 100
TW, TH = 360, 240


def plan_specs(path):
    """{текст фразы: (задание, запросы)} в той форме, в какой их получает
    рендер (load_specs/load): запросы — строками, задание — с объектами."""
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    out = {}
    for u in (plan.get("units") or {}).values():
        if not isinstance(u, dict) or not u.get("claims") or not u.get("focus"):
            continue
        spec = {"focus": u["focus"], "claims": u["claims"], "queries": u.get("queries_for") or []}
        for k in ("meaning", "about", "reading", "vehicle", "traps"):
            if u.get(k):
                spec[k] = u[k]
        qs = [q for q in (u.get("queries") or []) if isinstance(q, str) and q.strip()]
        out[u["text"]] = (spec, qs)
    return out


def cand_row(ps, p):
    return {"id": str(p.get("id")), "channel": ps.candidate_channel(p),
            "probe_url": ps.candidate_probe_url(p), "headers": p.get("_download_headers") or {},
            "alt": (p.get("alt") or "")[:200], "origin_query": p.get("_origin_query"),
            "shot_type": p.get("_shot_type")}


def _replay_used(st, entry):
    """Победитель уже сделанной фразы — в анти-дубль руки, как если бы её
    прогнали в этом процессе (хэши файлов не восстанавливаются: дубль по
    содержимому среди разных id — редкость, и он одинаково бьёт обе руки)."""
    ids, _hashes = st
    if entry.get("winner"):
        ids.add(entry["winner"])
        if entry["winner"].isdigit():
            ids.add(int(entry["winner"]))


def _source_health(ps):
    """Снимок счётчиков отказа источников: ошибки поиска по каналам, остывания
    и потери карточек Мет, пропуски Pexels по квоте. Разница снимков до и
    после руки говорит, собиралась ли её куча при здоровых источниках.

    Зачем. Первый прогон этого замера гонял третью руку отдельным процессом,
    позже и параллельно с другими прогонами: у неё Мет отдал 235 кандидатов
    против 341 у первой руки, Commons — 221 против 380, а у одной фразы Мет и
    Commons дали по ОДНОМУ кандидату против 121 и 53. Разница пятёрок была бы
    записана заданию, хотя её сделала квота и троттлинг источников."""
    import museum_sources
    st = {}
    for src, v in ps.SOURCE_STATS.items():
        st[f"{src}.search_errors"] = v.get("search_errors", 0)
        st[f"{src}.download_errors"] = v.get("download_errors", 0)
    st["met.cooldowns"] = museum_sources.FETCH_STATS.get("met_cooldowns", 0)
    st["met.cards_lost"] = (museum_sources.FETCH_STATS.get("met_cards_lost", 0)
                            + museum_sources.FETCH_STATS.get("met_lost_cards", 0))
    st["pexels.quota_skipped"] = ps.PEXELS_LOW_PRIORITY_SKIPPED
    st["pexels.broken"] = int(bool(ps.PEXELS_BROKEN))
    return st


def _health_delta(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)
            if after.get(k, 0) - before.get(k, 0)}


def _wait_for_healthy_sources(ps, max_wait=3600):
    """Перед рукой — дождаться, пока Мет выйдет из остывания и у Pexels
    останется квота: куча руки не должна собираться на полуживых источниках."""
    import time
    import museum_sources
    waited = 0
    while waited < max_wait:
        # Остаток квоты Pexels обновляется только ответом Pexels: после паузы
        # он снова «неизвестен», и следующий вызов покажет настоящий.
        low = ps.PEXELS_QUOTA_LEFT is not None and ps.PEXELS_QUOTA_LEFT < 15
        cooling = museum_sources.met_is_cooling_down()
        if not (low or cooling):
            return waited
        print(f"    источники: ждём ({'квота Pexels ' + str(ps.PEXELS_QUOTA_LEFT) if low else ''}"
              f"{' остывание Мет' if cooling else ''})", flush=True)
        pause = 600 if low else 30
        time.sleep(pause)
        waited += pause
        if low:
            ps.PEXELS_QUOTA_LEFT = None
    return waited


def _judge_counts(ps):
    st = ps._SHOT_JUDGE_STATE
    gw = st.get("gateway")
    return {"calls": getattr(gw, "calls", 0), "failures": getattr(gw, "failures", 0),
            "spent": getattr(gw, "spent", 0), "dead": getattr(gw, "dead", None),
            "refused": st.get("refused")}


def cmd_run(a):
    ep_dir = os.path.abspath(a.ep_dir)
    sys.argv = ["pipeline_smart.py", ep_dir]
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    import pipeline_smart as ps
    import selection_engine
    with open(a.gold, encoding="utf-8") as f:
        gold = json.load(f)
    arms = []
    for spec in a.arm:
        name, path = spec.split("=", 1)
        arms.append((name, plan_specs(path)))
    items = [it for it in gold["items"] if it["ep"] == a.ep][:a.limit or None]
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, "runs.json")
    done = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    state = {name: (set(), []) for name, _ in arms}
    for k, it in enumerate(items):
        key = f"{a.ep}#{k}"
        if key in done and all(n in done[key]["arms"] for n, _ in arms):
            for name, _ in arms:
                _replay_used(state[name], done[key]["arms"][name])
            continue
        row = done.get(key) or {"ep": a.ep, "text": it["text"], "note": it.get("note"), "arms": {}}
        # Порядок рук чередуется по фразам: вторая рука подряд чаще попадает
        # на остывание Мет, вызванное первой, и без чередования это был бы
        # систематический сдвиг против одной и той же руки.
        for name, specs in (arms if k % 2 == 0 else arms[::-1]):
            if name in row["arms"] and "error" not in row["arms"][name]:
                # Рука уже прогнана по этой фразе (дозапуск другой руки):
                # не повторять, но её анти-дубль обязан знать победителя.
                _replay_used(state[name], row["arms"][name])
                continue
            hit = next(((t, v) for t, v in specs.items() if t.startswith(it["text"])), None)
            if not hit:
                row["arms"][name] = {"error": "нет задания в плане"}
                continue
            text, (spec, qs) = hit
            ids, hashes = state[name]
            request = ps.build_slot_request(
                index=a.index, query=qs[0], extra_queries=qs[1:], text_key=text, shot_brief=None,
                shot_spec=spec, block_text=text, arbiter_text=None, is_opening=False, slot_dur=4.0,
                action_qualifier=None, target_luma=None, director_score_fn=None, director_assist=False,
                director_report=None, video_score_fn=None, used_photo_ids=ids, used_video_ids=set(),
                used_hashes=hashes, recent_sizes=[])
            entry = {"text": text, "focus": spec["focus"], "queries": qs,
                     "types": [x.get("type") for x in spec["queries"]]}
            entry["waited_sec"] = _wait_for_healthy_sources(ps)
            health0 = _source_health(ps)
            gw0 = _judge_counts(ps)
            try:
                pool = selection_engine.build_pool(request, ps.PhotoAdapter())
                entry["pool_size"] = len(pool)
                # Пятёрка — как её видит отбор: без кадров, уже показанных
                # этой рукой раньше (анти-дубль по id общий на эпизод).
                fresh = [p for p in pool if p.get("id") not in ids] or pool
                entry["head"] = [cand_row(ps, p) for p in fresh[:ps.FAST_PHOTO_DEDUP_MAX_TRIES]]
                from collections import Counter
                entry["pool_channels"] = dict(Counter(ps.candidate_channel(p) for p in pool))
                got = ps.select_media(request, "photo")
                side = ps.read_media_sidecar(got) if got else {}
                entry["winner"] = str(side.get("pexels_id")) if side.get("pexels_id") else None
                entry["winner_file"] = got
                entry["chosen_by"] = side.get("chosen_by")
                entry["relevance"] = side.get("relevance")
                entry["quality"] = side.get("quality")
                entry["verdicts"] = side.get("verdicts")
            except Exception as e:  # noqa: BLE001 — замер не падает от одной фразы
                entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            entry["source_trouble"] = _health_delta(health0, _source_health(ps))
            if a.index < ps.SHOT_JUDGE_PAID_SLOTS:
                # Платная зона: руки сравнимы, только если судья отработал у
                # обеих. Отказы шлюза за фразу пишутся рядом; выключенный
                # судья — ошибка фразы (дозапуск спросит её снова), и прогон
                # останавливается, а не копит несравнимые строки.
                gw1 = _judge_counts(ps)
                entry["judge"] = {k: gw1[k] - gw0.get(k, 0) for k in ("calls", "failures", "spent")}
                if gw1["dead"] or gw1["refused"]:
                    entry["error"] = f"судья выключен: {gw1['dead'] or gw1['refused']}"
                    row["arms"][name] = entry
                    done[key] = row
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(done, f, ensure_ascii=False, indent=1)
                    print(f"  [{key}] {name}: {entry['error']} — остановка", flush=True)
                    return 3
            row["arms"][name] = entry
            print(f"  [{key}] {name}: куча {entry.get('pool_size')}, пятёрка "
                  f"{[h['channel'] for h in entry.get('head') or []]}, победитель {entry.get('winner')}"
                  + (f", сбои источников {entry['source_trouble']}" if entry["source_trouble"] else ""),
                  flush=True)
        done[key] = row
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(done, f, ensure_ascii=False, indent=1)
    return 0


def cmd_sheet(a):
    import pool_recall as pr
    import runs_sheet
    from PIL import Image, ImageDraw
    pr.IMG_STORE = os.path.join(a.out, "img")
    runs = json.load(open(os.path.join(a.out, "runs.json"), encoding="utf-8"))
    sdir = os.path.join(a.out, "sheets")
    os.makedirs(sdir, exist_ok=True)
    ipath = os.path.join(sdir, "index.json")
    # Лист уже размеченной фразы не перерисовывается: номера плиток обязаны
    # остаться теми, по которым стоят метки. Порядок плиток — своё
    # детерминированное зерно у каждой фразы, а не общий поток.
    index = json.load(open(ipath, encoding="utf-8")) if os.path.exists(ipath) else {}
    supp = index.setdefault("_supplement", {})
    for key in sorted((k for k in runs if "#" in k), key=lambda k: (k.split("#")[0], int(k.split("#")[1]))):
        row = runs[key]
        if len(row.get("arms") or {}) < 2:
            continue
        cands = {}
        for name, e in row["arms"].items():
            for h in e.get("head") or []:
                cands.setdefault(h["id"], h)
            # Победитель вне пятёрки (платная зона: каскад и судья смотрят
            # глубже) — плиткой из его файла.
            if e.get("winner") and e["winner"] not in cands and e.get("winner_file"):
                cands[e["winner"]] = {"id": e["winner"], "file": e["winner_file"]}
        seen = set((index.get(key) or {}).values())
        new = [c for c in cands if c not in seen]
        if not new:
            continue
        # Лист уже размеченной фразы не перерисовывается; кадры руки,
        # догнанной позже, идут ДОПОЛНИТЕЛЬНЫМ листом с продолжением
        # нумерации. Слепота такого листа частичная: на нём только новые
        # кадры — это записано в index.json (_supplement) и в отчёте.
        rnd = random.Random(f"20260926|{key}|{len(seen)}")
        order = new
        rnd.shuffle(order)
        start = len(seen)
        part = "" if not seen else f"_b{start}"
        if seen:
            supp[key] = sorted(set(supp.get(key, [])) | {str(start + n) for n in range(1, len(order) + 1)},
                               key=int)
        cols = 5
        n_rows = max(1, (len(order) + cols - 1) // cols)
        sheet = Image.new("RGB", (cols * TW, 60 + n_rows * (TH + 22)), (0, 0, 0))
        d = ImageDraw.Draw(sheet)
        d.text((6, 4), f"{key}: {row['text']}"[:150], fill=(255, 255, 255), font=runs_sheet._font(18))
        d.text((6, 32), f"смысл (ожидание до прогонов): {row.get('note') or '—'}"[:170], fill=(180, 180, 180),
               font=runs_sheet._font(13))
        index.setdefault(key, {})
        for n, cid in enumerate(order, start + 1):
            h = cands[cid]
            path = None
            if h.get("file") and os.path.exists(h["file"]):
                path = os.path.join(sdir, "_tmp_tile.jpg")
                shutil.copyfile(h["file"], path)
            else:
                path = pr.fetch(h.get("probe_url"), h.get("headers"), cid)
            tile = Image.new("RGB", (TW, TH + 22), (16, 16, 16))
            if path:
                try:
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((TW, TH))
                        tile.paste(im, ((TW - im.width) // 2, (TH - im.height) // 2))
                except Exception:  # noqa: BLE001
                    pass
                os.remove(path)
            ImageDraw.Draw(tile).text((4, TH + 2), f"#{n}", fill=(255, 220, 0), font=runs_sheet._font(15))
            m = n - start
            sheet.paste(tile, ((m - 1) % cols * TW, 60 + (m - 1) // cols * (TH + 22)))
            index[key][str(n)] = cid
        sheet.save(os.path.join(sdir, key.replace("#", "_") + part + ".jpg"), quality=85)
        print(f"  лист {key}{part}: {len(order)} плиток")
    with open(ipath, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"листов: {len(index) - 1} -> {sdir}")
    return 0


def _arm_stats(rows_arms):
    s = {"phrases": 0, "winner_sum": 0, "winner_known": 0, "winner_bad": 0,
         "no_winner": 0, "head_best_sum": 0, "head_good": 0, "head_n": 0}
    for e, head, win in rows_arms:
        known = [x for x in head if x is not None]
        s["phrases"] += 1
        if e.get("winner") is None:
            s["no_winner"] += 1
        elif win is not None:
            s["winner_known"] += 1
            s["winner_sum"] += win
            s["winner_bad"] += 1 if win == 0 else 0
        s["head_best_sum"] += max(known) if known else 0
        s["head_good"] += sum(1 for x in known if x >= 1)
        s["head_n"] += len(known)
    return s


def _searched_badly(trouble):
    """Куча собиралась при отказе поиска: ошибка поиска источника, остывание
    Мет, пропуск Pexels по квоте. Сбой скачивания превью — нет: он бывает и в
    рендере, и бьёт по выбору из готовой кучи, а не по её составу."""
    return any(k.endswith(".search_errors") or k in ("met.cooldowns", "pexels.quota_skipped",
                                                     "pexels.broken")
               for k in (trouble or {}))


def cmd_score(a):
    """Итог по рукам — по всем фразам и отдельно по «чистым»: где ни у одной
    руки куча не собиралась при сбоях источников (source_trouble пуст)."""
    runs = json.load(open(os.path.join(a.out, "runs.json"), encoding="utf-8"))
    index = json.load(open(os.path.join(a.out, "sheets", "index.json"), encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    by_arm, clean_by_arm, rows = {}, {}, []
    for key, row in runs.items():
        lab = {index[key][t]: v for t, v in (labels.get(key) or {}).items() if t in index.get(key, {})}
        unlabeled = [t for t in index.get(key, {}) if t not in (labels.get(key) or {})]
        if unlabeled:
            print(f"  {key}: без метки плитки {unlabeled}")
        clean = not any(_searched_badly(e.get("source_trouble")) for e in row["arms"].values())
        line = {"key": key, "text": row["text"], "clean": clean}
        for name, e in row["arms"].items():
            head = [lab.get(h["id"]) for h in e.get("head") or []]
            win = lab.get(e.get("winner")) if e.get("winner") else None
            by_arm.setdefault(name, []).append((e, head, win))
            if clean:
                clean_by_arm.setdefault(name, []).append((e, head, win))
            line[name] = {"winner": win, "head": head, "source_trouble": e.get("source_trouble")}
        rows.append(line)
    per = {name: _arm_stats(v) for name, v in by_arm.items()}
    per_clean = {name: _arm_stats(v) for name, v in clean_by_arm.items()}
    print(json.dumps({"all": per, "clean": per_clean}, ensure_ascii=False, indent=1))
    with open(os.path.join(a.out, "score.json"), "w", encoding="utf-8") as f:
        json.dump({"per_arm": per, "per_arm_clean": per_clean, "rows": rows}, f,
                  ensure_ascii=False, indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("ep_dir")
    r.add_argument("--gold", required=True)
    r.add_argument("--ep", required=True)
    r.add_argument("--arm", action="append", required=True, help="имя=путь к плану")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0, help="только первые N фраз (проверка харнесса)")
    r.add_argument("--index", type=int, default=FREE_INDEX,
                   help="индекс слота: по умолчанию бесплатная зона; меньше SHOT_JUDGE_PAID_SLOTS — платная (судья)")
    s = sub.add_parser("sheet")
    s.add_argument("out")
    c = sub.add_parser("score")
    c.add_argument("out")
    c.add_argument("labels")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "sheet": cmd_sheet, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
