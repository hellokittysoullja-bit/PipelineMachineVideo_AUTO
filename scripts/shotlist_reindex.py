#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Переносит выбранные кадры в шотлист под НОВУЮ нарезку слотов.

Зачем. Пересчёт нарезки хука (потолок слота, первый слот ≤3 с) сдвигает номера
всех слотов после хука, а кэш отбора и клипов привязан к номеру слота и к
тексту блока. Без переноса пришлось бы заново подбирать кадры для всех слотов.

Что делает. Строит новую нарезку блоков теми же функциями, что main()
(parse_blocks -> split_long_blocks -> merge_short_phrase_locked_blocks), и для
каждого нового слота ищет в СТАРОМ шотлисте кадр:
  * текст блока совпал дословно -> кадр берётся и слот ЛОЧИТСЯ (следующий
    прогон поставит файл как есть, без поиска и судьи);
  * новый блок — первый кусок разрезанного старого блока -> первому куску
    достаётся кадр старого блока (залочен);
  * иначе слот остаётся без кадра и подбирается обычным путём.
Старый шотлист сохраняется рядом (shotlist.before_reindex.json).

Использование: python scripts/shotlist_reindex.py <video_dir> [--dry-run]
Клипы после переноса всё равно перерендериваются (файл входит в ключ кэша
клипа), но отбор — самая дорогая часть — не повторяется.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile


def _norm(t):
    return " ".join(str(t or "").split())


def build_new_blocks(video_dir):
    sys.argv = ["shotlist_reindex", video_dir]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pipeline_smart as ps
    blocks = ps.parse_blocks(ps.SCRIPT_FILE)
    for i, b in enumerate(blocks):
        b["orig_index"] = i
    weights = ps.load_alignment_weights(blocks)
    blocks, weights = ps.split_long_blocks(blocks, weights)
    total = ps.get_audio_duration()
    blocks, weights = ps.merge_short_phrase_locked_blocks(blocks, weights, total)
    return blocks


def reindex(video_dir, dry_run=False):
    path = os.path.join(video_dir, "media_plan", "shotlist.json")
    old = json.load(open(path, encoding="utf-8"))
    old_shots = [s for s in old.get("shots", []) if s.get("file")]
    new_blocks = build_new_blocks(video_dir)
    used = set()
    out, stats = [], {"exact": 0, "first_piece": 0, "fresh": 0}
    for j, b in enumerate(new_blocks):
        text = _norm(b["text"])
        pick, how = None, None
        for k, s in enumerate(old_shots):
            if k in used or s.get("section") != b["section"]:
                continue
            if _norm(s.get("text")) == text:
                pick, how = k, "exact"
                break
        if pick is None and not b.get("is_subcut"):
            words = text.split()
            for k, s in enumerate(old_shots):
                if k in used or s.get("section") != b["section"]:
                    continue
                ow = _norm(s.get("text")).split()
                if len(ow) > len(words) and ow[:len(words)] == words:
                    pick, how = k, "first_piece"
                    break
        shot = {"index": j, "section": b["section"], "text": b["text"], "lock": False, "file": None}
        if pick is not None:
            used.add(pick)
            src = old_shots[pick]
            shot.update({k: src[k] for k in ("kind", "file", "provider", "candidate_id",
                                             "relevance", "chosen_by", "provenance") if k in src})
            shot["lock"] = True
            shot["source"] = "shotlist_lock"
            stats[how] += 1
        else:
            stats["fresh"] += 1
        out.append(shot)
    print(f"Слотов было (с кадром): {len(old_shots)}, стало: {len(out)}")
    print(f"  перенесено дословно: {stats['exact']}, первым куском разрезанного блока: "
          f"{stats['first_piece']}, подбирается заново: {stats['fresh']}")
    hook_new = [s for s in out if s["section"].startswith("HOOK")]
    print(f"  хук: {len(hook_new)} слотов, из них с готовым кадром {sum(1 for s in hook_new if s['lock'])}")
    if dry_run:
        return out
    shutil.copy2(path, os.path.join(video_dir, "media_plan", "shotlist.before_reindex.json"))
    new = dict(old)
    new["shots"] = out
    new.pop("locked", None)
    json.dump(new, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Записано:", path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    reindex(os.path.abspath(a.video_dir), a.dry_run)
