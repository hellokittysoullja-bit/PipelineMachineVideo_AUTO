# -*- coding: utf-8 -*-
"""Слепой лист разметки брифов: единственный способ узнать, кто лучше.

ЗАЧЕМ. Все числа замера режиссёра (docs/quality/SHOT_BRIEF_DIRECTOR.md)
считаны против ЭТАЛОНА, который написал Claude в предыдущей сессии этого
же проекта. Значит рука «Claude» мерится против текста той же модели, и её
уровень завышен самосогласованностью на неизвестную величину. Человеческой
разметки БРИФОВ в репозитории нет вообще: золотой набор размечает 40
КАДРОВ опубликованного эпизода, а не брифы.

Пока её нет, «этот мозг лучше» — мнение, а не измерение. Этот лист её
собирает.

ЧТО ДЕЛАЕТ ЛИСТ СЛЕПЫМ, а не просто удобным:
  * брифы разных систем перемешаны и обезличены (А/Б/В/Г) — владелец не
    знает, чей какой, и не может подыграть ни одной;
  * порядок букв СВОЙ у каждого юнита и выведен из хэша самой фразы:
    иначе «А» систематически оказывалась бы одной и той же системой, и
    привычка руки заменила бы суждение;
  * ключ расшифровки пишется ОТДЕЛЬНЫМ файлом, которого в листе нет;
  * выборка стратифицирована по секциям, а не «первые N подряд» — начало
    эпизода почти всегда предметные кадры и скрыло бы ровно те случаи,
    ради которых разметка нужна.

ЧЕГО ЛИСТ НЕ ДЕЛАЕТ. Он про БРИФ («то ли просят»), а не про КАДР («хорош
ли результат»). Кадр решают гейты и ранжирование среди всех источников;
для него есть свой контактный лист (scripts/shotlist_contact.py) и
Шаг 7.5.
"""
import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import script_parser              # noqa: E402
import shot_brief_director as d   # noqa: E402

LETTERS = "АБВГДЕ"


def _order(text, n):
    """Свой порядок букв на каждый юнит, детерминированный по фразе."""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    idx = list(range(n))
    out = []
    while idx:
        h, k = divmod(h, len(idx))
        out.append(idx.pop(k))
    return out


def pick_units(blocks, take, sections=None):
    """Стратифицированная выборка: по очереди из каждой секции."""
    buckets = {}
    for i, b in enumerate(blocks):
        key = d._section_key(b.get("section") or "")
        if sections and key not in sections:
            continue
        buckets.setdefault(key, []).append(i)
    picked, round_no = [], 0
    while len(picked) < take:
        added = False
        for key in buckets:
            if round_no < len(buckets[key]):
                picked.append(buckets[key][round_no])
                added = True
                if len(picked) >= take:
                    break
        if not added:
            break
        round_no += 1
    return sorted(picked)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--arm", action="append", default=[],
                    metavar="ИМЯ=ФАЙЛ:КЛЮЧ")
    ap.add_argument("--reference", action="store_true",
                    help="включить эталонный инлайновый бриф как ещё одну "
                         "обезличенную колонку (он тоже чей-то, и слепая "
                         "разметка обязана судить его наравне)")
    ap.add_argument("--take", type=int, default=40)
    ap.add_argument("--sections", default=None)
    ap.add_argument("--out", default="brief_marking_sheet.md")
    ap.add_argument("--key-out", default=None)
    a = ap.parse_args(argv[1:])

    blocks = script_parser.parse_blocks(os.path.join(a.video_dir, "script.txt"))
    sections = ({s.strip().upper() for s in a.sections.split(",")}
                if a.sections else None)

    arms = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        path, key = rest.rsplit(":", 1)
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)["arms"][key]["rows"]
        arms[name] = {int(k): v.get("shot_en") for k, v in rows.items()}
    if a.reference:
        arms["ЭТАЛОН"] = {i: d._clean(b.get("shot_brief"))
                          for i, b in enumerate(blocks)}
    if not arms:
        print("нужна хотя бы одна --arm")
        return 2

    names = list(arms)
    units = pick_units(blocks, a.take, sections)
    lines = [
        "# Слепая разметка брифов",
        "",
        f"Эпизод: `{os.path.basename(a.video_dir.rstrip('/'))}`, "
        f"юнитов в листе: {len(units)}, вариантов на юнит: {len(names)}.",
        "",
        "На каждую фразу — несколько описаний кадра от разных систем, "
        "перемешанных и обезличенных. Чей какой — не знаю ни я в момент "
        "чтения листа, ни вы: ключ лежит отдельным файлом.",
        "",
        "**Что отметить.** В строке «лучший» поставить букву — какое "
        "описание вы бы отправили в подбор. Если ни одно не годится, "
        "написать «нет». Если несколько равны — перечислить через запятую.",
        "",
        "---",
        "",
    ]
    key_rows = []
    for i in units:
        text = d._clean(blocks[i].get("text"))
        order = _order(text, len(names))
        lines.append(f"### {i}. {blocks[i].get('section')}")
        lines.append("")
        lines.append(f"> {text}")
        lines.append("")
        mapping = {}
        for pos, ai in enumerate(order):
            nm = names[ai]
            brief = arms[nm].get(i) or "— (система промолчала)"
            lines.append(f"- **{LETTERS[pos]}.** {brief}")
            mapping[LETTERS[pos]] = nm
        lines.append("")
        lines.append("лучший: ____    почему (одно слово, необязательно): ____")
        lines.append("")
        key_rows.append({"unit": i, "phrase": text[:70], "map": mapping})

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    key_path = a.key_out or (os.path.splitext(a.out)[0] + "_KEY.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump({"arms": names, "units": key_rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"Лист: {a.out}\nКлюч (НЕ открывать до разметки): {key_path}")
    print(f"Систем: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
