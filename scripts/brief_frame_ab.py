# -*- coding: utf-8 -*-
"""A/B ГОТОВЫХ КАДРОВ: бриф Claude против брифа локальной модели.

ЗАЧЕМ. Все замеры режиссёра до сих пор отвечали на вопрос «чем СПРАШИВАЮТ
у источников» (`subject_hit` против эталонных брифов). Ни один не отвечал
на вопрос, который единственный имеет значение для ролика: **меняется ли
от этого КАДР НА ЭКРАНЕ, и в какую сторону**. 65 против 85 в запросах
может стоить три заметных кадра на эпизод, а может тридцать — это не
измерено ни разу, и без этого числа нельзя решить, стоит ли вообще
бороться за разрыв.

КАК. Берутся слоты, где два брифа РАЗЛИЧАЮТСЯ, и для каждого дважды
вызывается ТОТ ЖЕ `pexels_photo()`, которым собирается настоящий ролик —
с одинаковым запросом, одинаковыми extra_queries, одинаковым индексом.
Отличается РОВНО ОДИН аргумент: `shot_brief`. Всё остальное, включая
гейты, дедуп и ранжирование, — общий прод-путь, не копия.

У каждой руки СВОИ `used_ids`/`used_hashes`: общий дедуп между руками
означал бы, что вторая рука наказана за выбор первой.

ЧЕСТНЫЕ ПРЕДЕЛЫ, названные до чисел:
* Это НЕ полный рендер: сравниваются победители слотов, а не смонтированный
  ролик (грейд, движение камеры, склейка кадр не переназначают).
* Вердикт «лучше/хуже» по КАЖДОЙ паре выносит человек глазами по
  контактному листу — скрипт только собирает пары и считает, у скольких
  слотов кадр вообще СМЕНИЛСЯ. Доля сменившихся — объективное число;
  «стало лучше» — нет, и выдавать одно за другое нельзя.
* Полка (`shelf_index`) в этом окружении не собрана, поэтому влияние
  брифа идёт единственным живым каналом — `brief_to_stock_query()`. Это
  же верно и на машине владельца, пока полка не собрана.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def local_briefs(path):
    """Брифы локальной модели из замороженного прогона замера."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)["arms"]["C"]["rows"]
    return {int(k): v["shot_en"] for k, v in rows.items() if v.get("shot_en")}


def pick_slots(blocks, local, limit, stride):
    """Слоты, где брифы РЕАЛЬНО различаются. Берём с шагом по эпизоду, а
    не первые подряд: первые главы — это хук, у него свой характер, и
    выборка из одного хука не описывала бы эпизод."""
    import shot_brief_director as d
    out = []
    for i, b in enumerate(blocks):
        mine = d._clean(b.get("shot_brief"))
        theirs = d._clean(local.get(i))
        if not mine or not theirs or mine.lower() == theirs.lower():
            continue
        out.append((i, mine, theirs))
    return out[::stride][:limit]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--local-run", required=True,
                    help="JSON замера локальной модели (docs/quality/...)")
    ap.add_argument("--limit", type=int, default=12,
                    help="сколько слотов сравнить (квота Pexels 200/час)")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv[1:])

    import script_parser
    import pipeline_smart as ps

    blocks = script_parser.parse_blocks(os.path.join(a.video_dir, "script.txt"))
    local = local_briefs(a.local_run)
    slots = pick_slots(blocks, local, a.limit, a.stride)
    print(f"слотов с РАЗНЫМИ брифами взято: {len(slots)}\n")

    # Ровно то же, что делает main(): авторские запросы из
    # `=== PEXELS QUERIES ===` кормят и resolve_queries, и пул секции.
    # Своей копии этой сборки не заводится — иначе замер мерил бы другой
    # вход, чем настоящий рендер.
    authored = ps.parse_pexels_queries(os.path.join(a.video_dir, "script.txt"))
    queries = ps.resolve_queries(blocks, authored_queries=authored)
    pool = {}
    if authored:
        for b in blocks:
            key = ps._normalize_section_key(b["section"])
            if key and key in authored:
                pool[b["section"]] = authored[key]

    # Своё состояние дедупа на каждую руку — иначе вторая рука
    # наказывается за то, что выбрала первая.
    state = {"claude": (set(), []), "local": (set(), [])}
    results = []
    for i, mine, theirs in slots:
        b = blocks[i]
        row = {"index": i, "text": b.get("text", "")[:160],
               "section": b.get("section"), "query": queries[i],
               "brief_claude": mine, "brief_local": theirs}
        for arm, brief in (("claude", mine), ("local", theirs)):
            ids, hashes = state[arm]
            try:
                got = ps.pexels_photo(
                    queries[i], i, used_ids=ids, used_hashes=hashes,
                    extra_queries=pool.get(b.get("section")),
                    shot_brief=brief, block_text=b.get("text"))
            except Exception as e:
                got = None
                row[arm + "_error"] = f"{type(e).__name__}: {e}"
            row[arm + "_file"] = got
            print(f"  [{i:3d}] {arm:<6} -> {os.path.basename(got) if got else 'НЕТ'}")
        row["frame_changed"] = bool(
            row.get("claude_file") and row.get("local_file")
            and row["claude_file"] != row["local_file"])
        results.append(row)

    changed = sum(1 for r in results if r["frame_changed"])
    both = sum(1 for r in results if r.get("claude_file") and r.get("local_file"))
    summary = {"slots_compared": len(results), "both_arms_got_a_frame": both,
               "frame_changed": changed,
               "frame_changed_share": round(changed / both, 3) if both else None}
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "slots": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nJSON: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
