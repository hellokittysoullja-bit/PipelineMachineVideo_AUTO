# -*- coding: utf-8 -*-
"""A/B трёх режиссёров на ОДНОМ движке и ОДНОЙ модели.

ЗАЧЕМ ОТДЕЛЬНЫЙ ЗАМЕР. Предыдущее измерение планировщика стояло на ВОСЬМИ
фразах, и вердикт по каждой выносился глазами автора правки. Восемь точек
— это подгонка под выборку, что записано и в самом CLAUDE.md. Здесь
измеряется весь эпизод, и у него есть ЭТАЛОН: автор эпизода 02 написал
`[shot:]` ко всем 142 юнитам.

КТО ИХ НАПИСАЛ — ГЛАВНАЯ ОГОВОРКА. Не человек: их написал Claude в
предыдущей сессии этого же проекта (коммит 6ddd7b7). Поэтому рука
«Claude» мерится против текста той же модели и её уровень завышен
самосогласованностью на неизвестную величину. Сравнение между руками,
не родственными эталону, оговорка не трогает.

ТРИ РУКИ, ОДНА ПЕРЕМЕННАЯ:
  A. запрос секции      — то, чем слот обходился до всякого режиссёра;
  B. пофразовый промпт  — сегодняшний shot_planner_llm (v3), фраза одна;
  C. глава с контекстом — shot_brief_director, та же модель, тот же движок.

B и C идут через ОДИН объект LocalBrain: одна модель, одна квантовка, один
seed, temperature 0. Различается ровно одно — что модель видит на входе.
Иначе сравнение мерило бы разницу движков, а не разницу постановки.

ЧЕСТНАЯ ГРАНИЦА ГЛАВНОЙ МЕТРИКИ, названная до чисел. `subject_hit` —
это совпадение ПРЕДМЕТА с эталонным брифом. Совпало — кадр почти наверняка
годный. НЕ совпало — это НЕ доказательство брака: другой, тоже верный
кадр даст промах по этой метрике. То есть число снизу, а не приговор;
верхнюю границу может дать только разметка владельца глазами.

Второй предел, столь же честный: эталонные брифы написаны уже зная, что
корпус музейный (записано в CLAUDE.md), — эталон смещён в сторону
предметов, и на другой нише доля будет иной.
"""
import argparse
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import script_parser          # noqa: E402
import shot_planner_llm       # noqa: E402
import shot_brief_director as director  # noqa: E402

# Словарь ТОЛЬКО измерительный, в проде не участвует. Из сравнения
# предметов убираются служебные слова английского и рамка кадра (ракурс,
# крупность, фон) — совпадение по слову «close» ничего не говорит о том,
# тот ли предмет назван. Список ракурсных слов намеренно совпадает по
# смыслу с BRIEF_FRAMING_WORDS в pipeline_smart: там он режет запрос к
# стокам, здесь — шум при сравнении.
STOP = set("""a an the of in on at with and or to from for its his her their
this that these those is are was were be being been by as into over under
seen showing shown close up macro closeup detail view shot frame background
foreground front side above below whole full entire together beside near
against dark light bright plain single one two lying standing resting
catching taken apart""".split())

# Слова эпохи и материала: они общие почти у всех брифов канала и сами по
# себе НЕ доказывают, что назван тот же предмет. Из пересечения убираются,
# но наличие хотя бы одного проверяется отдельной осью (era_ok).
ERA_WORDS = set("""medieval steel iron european knight knights knightly
armour armor armoured armored warrior warriors""".split())

_WORD_RE = re.compile(r"[a-z]+")


def content_words(text):
    ws = [w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 2]
    return {w for w in ws if w not in STOP}


def subject_words(text):
    return content_words(text) - ERA_WORDS


def era_ok(text):
    return bool(content_words(text) & ERA_WORDS)


def subject_hit(candidate, reference):
    """Назван ли ТОТ ЖЕ предмет, что у автора.

    Совпадение считается по основе слова, а не по точной форме: у автора
    «gauntlet», у модели «gauntlets» — это один предмет. Порог — одно
    общее значимое слово: брифы короткие, и требовать двух значило бы
    мерить формулировку, а не предмет.
    """
    a, b = subject_words(candidate), subject_words(reference)
    if not a or not b:
        return False, 0.0
    hit = False
    for x in a:
        for y in b:
            if x == y or (len(x) > 4 and y.startswith(x[:5])) \
                      or (len(y) > 4 and x.startswith(y[:5])):
                hit = True
                break
        if hit:
            break
    inter = len(a & b)
    return hit, round(inter / max(1, len(a | b)), 3)


def score_row(shot_en, author_brief, phrase):
    hit, jac = subject_hit(shot_en, author_brief)
    ok, why = shot_planner_llm.brief_is_safe(shot_en or "", phrase)
    return {"shot_en": shot_en, "subject_hit": hit, "jaccard": jac,
            "era_ok": era_ok(shot_en or ""), "validator_ok": ok,
            "validator_reason": why,
            "translation_shape": shot_planner_llm._looks_like_translation(
                shot_en or "", phrase)}


# --- РУКА A: запрос секции --------------------------------------------------

def arm_section_query(video_dir, blocks):
    """Тот же запрос на всю секцию — состояние до любого режиссёра.

    Запросов у секции несколько; слоту достаётся один из них по
    semantic_query_assignment. Здесь берётся ЛУЧШИЙ из них по нашей же
    метрике — то есть рука A измеряется в свою пользу. Иначе сравнение
    было бы нечестным в другую сторону.
    """
    qmap = director.section_queries(video_dir)
    out = {}
    for i, b in enumerate(blocks):
        key = director._section_key(b.get("section") or "")
        raw = qmap.get(key, "")
        if not raw:
            continue
        variants = [re.sub(r"\[[^\]]*\]", "", q).strip()
                    for q in raw.split(",")]
        ref = director._clean(b.get("shot_brief"))
        best, best_row = None, None
        for q in variants:
            if not q:
                continue
            row = score_row(q, ref, b.get("text") or "")
            if best_row is None or (row["subject_hit"], row["jaccard"]) > \
                                   (best_row["subject_hit"], best_row["jaccard"]):
                best, best_row = q, row
        if best_row:
            out[i] = best_row
    return out


# --- РУКА B: пофразовый промпт v3 через тот же движок -----------------------

def arm_per_phrase(brain, blocks, limit=None, verbose=True):
    """Сегодняшний shot_planner_llm: одна фраза, ноль контекста.

    Промпт берётся ИЗ САМОГО МОДУЛЯ (SYSTEM_PROMPT), не переписывается
    здесь: переписанная копия мерила бы не то, что стоит в проде.
    """
    out, t0 = {}, time.time()
    todo = list(enumerate(blocks))[:limit] if limit else list(enumerate(blocks))
    for n, (i, b) in enumerate(todo, 1):
        text = director._clean(b.get("text"))
        if not text:
            continue
        raw = brain.ask_system(shot_planner_llm.SYSTEM_PROMPT,
                               f"Фраза диктора: «{text}»", max_tokens=320)
        parsed = shot_planner_llm.parse_reply(raw)
        shot = parsed["shot_en"] if parsed else None
        out[i] = dict(score_row(shot, director._clean(b.get("shot_brief")), text),
                      parsed=bool(parsed))
        if verbose and n % 10 == 0:
            print(f"    B: {n}/{len(todo)}  {time.time() - t0:.0f}с", flush=True)
    return out


# --- РУКА C: глава с контекстом ---------------------------------------------

def arm_chapter(brain, video_dir, blocks, cache_dir=None, verbose=True,
                use_vocabulary=False):
    found = director.run(video_dir, blocks, brain, cache_dir=cache_dir,
                         verbose=verbose, use_vocabulary=use_vocabulary)
    out = {}
    for i, b in enumerate(blocks):
        got = found.get(i)
        shot = got["shot_en"] if got else None
        out[i] = dict(score_row(shot, director._clean(b.get("shot_brief")),
                                b.get("text") or ""), parsed=bool(got))
    return out


def summarise(name, rows, total):
    said = [r for r in rows.values() if r.get("shot_en")]
    hits = [r for r in said if r["subject_hit"]]
    return {
        "arm": name,
        "units_total": total,
        "answered": len(said),
        "answered_share": round(len(said) / max(1, total), 3),
        "subject_hit": len(hits),
        "subject_hit_of_answered": round(len(hits) / max(1, len(said)), 3),
        "subject_hit_of_all": round(len(hits) / max(1, total), 3),
        "era_ok_of_answered": round(
            sum(1 for r in said if r["era_ok"]) / max(1, len(said)), 3),
        "validator_rejects": sum(1 for r in said if not r["validator_ok"]),
        "translation_shape": sum(1 for r in said if r["translation_shape"]),
        "distinct_briefs": len({r["shot_en"] for r in said}),
        "mean_jaccard_of_answered": round(
            sum(r["jaccard"] for r in said) / max(1, len(said)), 3),
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--model", default=os.environ.get("LLAMA_MODEL_GGUF", ""))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--arms", default="ABC")
    ap.add_argument("--answers", default=None,
                    help="папка ответов для руки D (мозг, который нельзя "
                         "запустить подпроцессом: человек или Claude)")
    ap.add_argument("--limit-b", type=int, default=None,
                    help="сколько юнитов прогнать рукой B (она дороже всех)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--vocabulary", action="store_true")
    a = ap.parse_args(argv[1:])

    blocks = script_parser.parse_blocks(os.path.join(a.video_dir, "script.txt"))
    total = len(blocks)
    ref_present = sum(1 for b in blocks if director._clean(b.get("shot_brief")))
    print(f"Юнитов {total}, эталонных брифов {ref_present}")
    if ref_present < total:
        print("ВНИМАНИЕ: эталон неполон — метрика считается только по тем, "
              "у кого бриф автора есть")

    res, brain = {}, None
    if "A" in a.arms:
        rows = arm_section_query(a.video_dir, blocks)
        res["A"] = {"summary": summarise("A: запрос секции", rows, total),
                    "rows": {str(k): v for k, v in rows.items()}}
        print(json.dumps(res["A"]["summary"], ensure_ascii=False, indent=2))

    if "D" in a.arms:
        if not a.answers:
            print("Рука D требует --answers")
            return 2
        print("\nРука D: главы с контекстом, мозг — из файлов")
        director.STATS.update({k: 0 for k in director.STATS})
        del director.REJECTED[:]
        rows = arm_chapter(director.FileBrain(a.answers), a.video_dir, blocks,
                           cache_dir=None, verbose=False)
        res["D"] = {"summary": summarise("D: глава, внешний мозг", rows, total),
                    "rows": {str(k): v for k, v in rows.items()},
                    "director_stats": dict(director.STATS),
                    "rejected": list(director.REJECTED)}
        print(json.dumps(res["D"]["summary"], ensure_ascii=False, indent=2))

    if "B" in a.arms or "C" in a.arms:
        if not a.model or not os.path.exists(a.model):
            print("Нет модели — руки B/C не считаются")
            return 2
        brain = EvalBrain(a.model, a.threads)

    if "C" in a.arms:
        print("\nРука C: главы с контекстом")
        rows = arm_chapter(brain, a.video_dir, blocks, cache_dir=a.cache,
                           use_vocabulary=a.vocabulary)
        res["C"] = {"summary": summarise("C: глава с контекстом", rows, total),
                    "rows": {str(k): v for k, v in rows.items()},
                    "director_stats": dict(director.STATS),
                    "rejected": director.REJECTED}
        print(json.dumps(res["C"]["summary"], ensure_ascii=False, indent=2))

    if "B" in a.arms:
        print("\nРука B: пофразовый промпт v3")
        rows = arm_per_phrase(brain, blocks, limit=a.limit_b)
        res["B"] = {"summary": summarise("B: пофразовый v3", rows,
                                         a.limit_b or total),
                    "rows": {str(k): v for k, v in rows.items()}}
        print(json.dumps(res["B"]["summary"], ensure_ascii=False, indent=2))

    payload = {"model": os.path.basename(a.model) if a.model else None,
               "episode": os.path.basename(a.video_dir.rstrip("/")),
               "packet_version": director.PACKET_VERSION,
               "prompt_version": shot_planner_llm.PLANNER_PROMPT_VERSION,
               "arms": res}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {a.out}")
    return 0


class EvalBrain(director.LocalBrain):
    """Тот же локальный мозг, плюс вызов с системным промптом — он нужен
    руке B, потому что пофразовый режим устроен именно так."""

    def ask_system(self, system, user, max_tokens=320):
        r = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0, seed=self.seed, max_tokens=max_tokens)
        return r["choices"][0]["message"]["content"] or ""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
