# -*- coding: utf-8 -*-
"""Сводка по рукам режиссёра: одна таблица, один набор осей, один срез глав.

ЗАЧЕМ ОТДЕЛЬНО ОТ ЗАМЕРА. Руки считаются разными запусками и разной ценой
(одна бесплатна, другая час на CPU), но сравнивать их нужно НА ОДНИХ И ТЕХ
ЖЕ юнитах. Здесь это делается пост-обработкой сохранённых строк, а не
перепрогоном: иначе честный срез стоил бы ещё одного часа модели.

ОСЬ КАТАЛОГА — единственная здесь, которая НЕ опирается на эталон автора.
Она спрашивает каталог Мет тем же `met_catalog.search()`, что и путь
отбора, и смотрит, назван ли в ответе предмет, о котором просил бриф.
Это и есть проверка СОДЕРЖАНИЯ вместо проверки формы: «poleaxe» и
«rondel» грамматически безупречны, а в каталоге музея их нет (у него
`Halberd` и `Roundel dagger`).

ЧЕСТНЫЙ ПРЕДЕЛ ОСИ КАТАЛОГА. Поиск каталога словарный, и у него есть
задокументированная ловушка: `neck` совпадает с `Necklace`. То есть ось
меряет «спросили ли словом, которое у музея есть», а не «хороший ли
кадр». Она дополняет эталон, а не заменяет его.
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import script_parser              # noqa: E402
import shot_brief_director as d   # noqa: E402
import shot_brief_eval as ev      # noqa: E402

_W = re.compile(r"[a-z]+")

# Крупность плана по словам самого брифа. Ось отвечает на прямой вопрос
# «не выглядит ли это шаблонно»: система, у которой все кадры одной
# крупности, читается как автомат независимо от того, верно ли назван
# предмет. Замер 15.09: у эталона доля самой частой крупности 65%, у
# пофразового режима 93% и НИ ОДНОГО крупного плана за эпизод.
_CLOSE = ("close up", "close-up", "macro", "detail", "tip of", "point of")
_WIDE = ("whole figure", "whole", "field", "landscape", "from above",
         "battlefield", "interior", "panorama")


def shot_scale(brief):
    low = (brief or "").lower()
    if any(k in low for k in _CLOSE):
        return "крупный"
    if any(k in low for k in _WIDE):
        return "общий"
    return "средний"


def scale_profile(briefs):
    """Доля самой частой крупности. Чем ближе к 1, тем однообразнее."""
    if not briefs:
        return None
    counts = {}
    for b in briefs:
        k = shot_scale(b)
        counts[k] = counts.get(k, 0) + 1
    return round(max(counts.values()) / len(briefs), 3)


def corpus_agrees(brief, limit=5):
    """Достаёт ли бриф из каталога предмет, который он и просил.

    Сравнивается со ЗНАЧИМЫМИ словами брифа, из которых убраны служебные
    и рамочные (те же, что в замере): совпадение по слову «close» ничего
    не сказало бы о предмете.
    """
    try:
        import met_catalog as mc
        rows = mc.search(brief, limit=limit)
    except Exception:
        return None
    if not rows:
        return False
    want = ev.subject_words(brief)
    for r in rows:
        got = {w for w in _W.findall((r.get("name") or "").lower()) if len(w) > 3}
        for a in want:
            for b in got:
                if a == b or (len(a) > 4 and b.startswith(a[:5])) \
                          or (len(b) > 4 and a.startswith(b[:5])):
                    return True
    return False


def load_arm(path, key):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    arm = data.get("arms", {}).get(key)
    return {int(k): v for k, v in arm["rows"].items()} if arm else {}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--arm", action="append", default=[],
                    metavar="ИМЯ=ФАЙЛ:КЛЮЧ", help="например C=ab.json:C")
    ap.add_argument("--sections", default=None,
                    help="через запятую: только эти главы (срез для честного "
                         "сравнения рук, посчитанных на разных наборах)")
    ap.add_argument("--corpus", action="store_true",
                    help="считать ось каталога (медленно: поиск по 30k строк)")
    ap.add_argument("--fallback", default=None, metavar="ФАЙЛ:КЛЮЧ",
                    help="чем закрывается молчащий юнит в проде (обычно "
                         "рука A: запрос секции). Без этого руки, которая "
                         "честно молчит, сравниваются несправедливо — "
                         "в проде слот не пустеет, он идёт прежним путём")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv[1:])

    blocks = script_parser.parse_blocks(os.path.join(a.video_dir, "script.txt"))
    keep = None
    if a.sections:
        want = {s.strip().upper() for s in a.sections.split(",")}
        keep = {i for i, b in enumerate(blocks)
                if d._section_key(b.get("section") or "") in want}

    ref = {i: d._clean(b.get("shot_brief")) for i, b in enumerate(blocks)}
    universe = keep if keep is not None else set(ref)

    fb = {}
    if a.fallback:
        fpath, fkey = a.fallback.rsplit(":", 1)
        fb = load_arm(fpath, fkey)

    table, detail = [], {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        path, key = rest.rsplit(":", 1)
        rows = load_arm(path, key)
        said = {i: r for i, r in rows.items()
                if i in universe and r.get("shot_en")}
        hits = [i for i, r in said.items() if r["subject_hit"]]
        row = {"рука": name, "юнитов": len(universe), "ответов": len(said),
               "предмет совпал": len(hits),
               "доля от ответов": round(len(hits) / max(1, len(said)), 3),
               "доля от всех": round(len(hits) / max(1, len(universe)), 3),
               "разных описаний": len({r["shot_en"] for r in said.values()}),
               "отклонено проверкой": sum(1 for r in said.values()
                                          if not r["validator_ok"]),
               "пересказ фразы": sum(1 for r in said.values()
                                     if r["translation_shape"])}
        row["однообразие крупностей"] = scale_profile(
            [r["shot_en"] for r in said.values()])
        if a.corpus:
            vals = [corpus_agrees(r["shot_en"]) for r in said.values()]
            ok = [v for v in vals if v is not None]
            row["каталог отвечает"] = (round(sum(ok) / max(1, len(ok)), 3)
                                       if ok else None)
        if fb:
            # ЭФФЕКТИВНАЯ рука: бриф там, где он есть, иначе прежний путь.
            # Это и есть то, что реально увидит эпизод: режиссёр ничего не
            # отнимает у молчащего слота, он только добавляет там, где
            # сказал. Сравнивать «долю от всех» без этого значит штрафовать
            # честное молчание, которого в проде не существует.
            eff_hits, eff_said = 0, 0
            for i in universe:
                r = said.get(i) or fb.get(i)
                if not r or not r.get("shot_en"):
                    continue
                eff_said += 1
                eff_hits += bool(r["subject_hit"])
            row["с откатом: ответов"] = eff_said
            row["с откатом: совпало"] = eff_hits
            row["с откатом: доля"] = round(eff_hits / max(1, len(universe)), 3)
        table.append(row)
        detail[name] = {str(i): said[i]["shot_en"] for i in sorted(said)}

    # Эталон как контрольная строка: он обязан стоять в таблице, иначе
    # неизвестно, на что вообще похожа «сотня процентов» по каждой оси.
    refs = {i: ref[i] for i in universe if ref.get(i)}
    ctrl = {"рука": "ЭТАЛОН (автор)", "юнитов": len(universe),
            "ответов": len(refs), "предмет совпал": len(refs),
            "доля от ответов": 1.0, "доля от всех":
                round(len(refs) / max(1, len(universe)), 3),
            "разных описаний": len(set(refs.values())),
            "отклонено проверкой": 0, "пересказ фразы": 0}
    ctrl["однообразие крупностей"] = scale_profile(list(refs.values()))
    if a.corpus:
        vals = [corpus_agrees(v) for v in refs.values()]
        ok = [v for v in vals if v is not None]
        ctrl["каталог отвечает"] = round(sum(ok) / max(1, len(ok)), 3) if ok else None
    table.append(ctrl)

    cols = list(table[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in table)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in table:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"sections": a.sections, "table": table,
                       "briefs": detail}, f, ensure_ascii=False, indent=2)
        print(f"\nJSON: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
