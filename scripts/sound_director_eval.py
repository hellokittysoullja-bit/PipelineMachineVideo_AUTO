#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер звукового режиссёра по эталонной разметке.

    python scripts/sound_director_eval.py --model ds/deepseek-v4-flash --critic ds/deepseek-v4-flash

Эталон — tests/fixtures/sound_director/gold_*.json: у каждой фразы статус
req (сцена, звук нужен), opt (звук допустим) или no (звук лишний) и список
подходящих видов. Метки — Claude, не владелец.

Числа:
  wrong  — звук на фразе «no» или вид не из подходящих (главный брак);
  right  — звук подходящего вида на фразе req/opt;
  recall — доля сцен req, где подходящий звук стоит на этой фразе или на
           одной из двух предыдущих или на следующей фразе той же главы
           (событие длится ~20 с).
Платно: вызовы шлюза (ключ = согласие владельца). Кэш вопросов — в
--work, повторный прогон тех же моделей бесплатен."""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sound_director as sd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures", "sound_director")
WPM = 125.0


def estimated_starts(blocks):
    t, out = 0.0, []
    for b in blocks:
        out.append(t)
        t += len(str(b.get("text", "")).split()) / WPM * 60.0
    return out


def score(gold_units, cues):
    """cues: [(индекс, тип, вид)] -> словарь чисел."""
    by_i = {}
    for i, typ, name in cues:
        if typ == "amb":
            by_i[i] = name
    right, wrong, wrong_list = 0, 0, []
    for i, name in by_i.items():
        g = gold_units[i]
        if g["status"] != "no" and name in g["ok_kinds"]:
            right += 1
        else:
            wrong += 1
            wrong_list.append((i, name, g["text"][:70]))
    req = [g for g in gold_units if g["status"] == "req"]
    covered = 0
    for g in req:
        i = g["i"]
        for j in (i, i - 1, i - 2, i + 1):
            if j >= 0 and gold_units[j]["section"] == g["section"] and by_i.get(j) in g["ok_kinds"]:
                covered += 1
                break
    missed = [(g["i"], g["text"][:70]) for g in req
              if not any(j >= 0 and gold_units[j]["section"] == g["section"]
                         and by_i.get(j) in g["ok_kinds"] for j in (g["i"], g["i"] - 1, g["i"] - 2))]
    return {"cues": len(by_i), "right": right, "wrong": wrong, "req": len(req),
            "covered": covered, "wrong_list": wrong_list, "missed": missed,
            "kinds": sorted({n for n in by_i.values()})}


def run(gold_path, drafts, critics, work, kinds):
    import llm_gateway
    import script_parser
    gold = json.load(open(gold_path, encoding="utf-8"))
    src = gold["script"]
    vdir = os.path.join(work, os.path.basename(gold_path).replace(".json", ""))
    os.makedirs(os.path.join(vdir, "media_plan"), exist_ok=True)
    shutil.copy(os.path.join(HERE, "..", src), os.path.join(vdir, "script.txt"))
    wc = gold.get("world_card")
    if wc:
        shutil.copy(os.path.join(HERE, "..", wc), os.path.join(vdir, "media_plan", "world_card.json"))
    blocks = script_parser.parse_blocks(os.path.join(vdir, "script.txt"))
    assert [b["text"] for b in blocks] == [u["text"] for u in gold["units"]], "эталон устарел"
    gw = llm_gateway.Gateway(spend_cap=200000)
    plan = sd.plan_episode(vdir, blocks, estimated_starts(blocks), gw, kinds,
                           draft_models=drafts, critic_models=critics)
    res = score(gold["units"], sd.plan_cues(plan, blocks))
    res["spent"] = gw.spent
    res["stats"] = plan.get("stats")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", default=",".join(sd.DRAFT_MODELS))
    ap.add_argument("--critics", default=",".join(sd.CRITIC_MODELS))
    ap.add_argument("--work", default=os.path.join(HERE, "..", "temp_sound_eval"))
    ap.add_argument("--repeat", default="", help="метка повтора: отдельный кэш, свежие ответы")
    args = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    import pipeline_smart
    kinds = pipeline_smart.available_ambience_kinds()
    drafts = tuple(m for m in args.drafts.split(",") if m)
    critics = tuple(m for m in args.critics.split(",") if m)
    work = os.path.join(args.work, f"v{sd.PLAN_VERSION}_{'+'.join(drafts)}__{'+'.join(critics)}".replace("/", "_") + (f"_r{args.repeat}" if args.repeat else ""))
    for name in sorted(os.listdir(FIX)):
        if name.startswith("gold_") and name.endswith(".json"):
            r = run(os.path.join(FIX, name), drafts, critics, work, kinds)
            print(f"{name}: звуков {r['cues']} верных {r['right']} НЕВЕРНЫХ {r['wrong']} "
                  f"сцен покрыто {r['covered']}/{r['req']} потрачено {r['spent']} {r['stats']}")
            for w in r["wrong_list"]:
                print("   неверно:", w)
            for w in r["missed"]:
                print("   пропущено:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
