# -*- coding: utf-8 -*-
"""Сверка брифов с полкой ДО рендера: автор видит, что реально принесёт.

ЗАЧЕМ. Главная причина брака подбора в этом проекте никогда не была в
гейтах — она была в том, что ВХОД писался вслепую. Самый дорогой пример
задокументирован: в опубликованном эпизоде фраза «вспомни, сколько весит
пакет молока» получила запрос `milk bottle hand`, и в ролик ушла
современная кухня с хлопьями — при том что на полке в этот момент лежали
настоящие двуручные мечи и латные перчатки. Автор не мог этого знать: между
«написал строку» и «увидел кадр» стоял целый рендер.

Этот скрипт убирает именно этот разрыв. Он ничего не чинит и ничего не
блокирует — он ПОКАЗЫВАЕТ: по каждому юниту сценария печатает фразу, её
бриф (`[shot:...]`) и первые кандидаты, которые этот бриф реально достаёт с
полки. Дальше решение за автором: переписать бриф, оставить как есть или
признать, что нужного предмета на полке нет вообще (и тогда честно уйти в
сток или в карточку, а не отправлять слот в лотерею).

ПОЧЕМУ ЭТО НЕ ДУБЛИРУЕТ `lint_authored_queries()`. Тот линт смотрит на
ТЕКСТ запроса (противоречие блоклисту, голодающий пул) и физически не может
сказать, что запрос принесёт: у него нет полки. Здесь наоборот — вопрос
ровно один, «что придёт», и ответ берётся из того же индекса, который будет
отвечать на рендере, той же функцией.

Стоит ноль живых вызовов и нисколько денег: полка локальная, модель нужна
только чтобы посчитать вектор самого брифа.

Запуск:
    python scripts/shot_brief_review.py videos/02_ne-mechom
    python scripts/shot_brief_review.py videos/02_ne-mechom --missing-only
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pipeline_smart  # noqa: E402  (резолвер вопроса к полке — общий с рендером)


def review(video_dir, top=3, missing_only=False, limit=None):
    import script_parser
    import shelf_index

    script_path = os.path.join(video_dir, "script.txt")
    if not os.path.exists(script_path):
        print(f"Нет {script_path}")
        return 1
    blocks = script_parser.parse_blocks(script_path)
    if limit:
        blocks = blocks[: int(limit)]

    have = sum(1 for b in blocks if b.get("shot_brief"))
    shelf_ok = shelf_index.available()
    st = shelf_index.stats()
    print(f"Юнитов: {len(blocks)}   с брифом: {have}   без брифа: {len(blocks) - have}")
    if shelf_ok:
        print(f"Полка: {st['items']} предметов, модель {st['model']}")
    else:
        # Честно и громко: без полки этот отчёт показывает только покрытие
        # брифами, а не то, что они принесут. Молча напечатать половину
        # проверки и выглядеть как проверка — ровно тот класс ложной
        # уверенности, который протокол этого репозитория запрещает.
        print("Полка НЕ собрана — показываю только покрытие брифами, "
              "не то, что они принесут. Собрать: python scripts/shelf_index.py build")

    section = None
    for i, b in enumerate(blocks):
        brief = b.get("shot_brief")
        if missing_only and brief:
            continue
        if b.get("section") != section:
            section = b.get("section")
            print(f"\n=== {section}")
        phrase = (b.get("text") or "").strip()
        print(f"\n[{i:3}] {phrase[:96]}")
        # ЧЕМ полку спросят НА САМОМ ДЕЛЕ — тем же резолвером, что и
        # продакшен. Найденный дефект (16.09): здесь печаталось «слот
        # пойдёт на авторский запрос секции, как раньше», и это перестало
        # быть правдой в тот момент, когда pexels_photo начал спрашивать
        # полку фразой блока. Инструмент, существующий ровно для того,
        # чтобы показать автору ответ полки ДО рендера, показывал не то,
        # что сделает рендер. Вторая копия правила прожила меньше суток —
        # ровно тот класс, ради которого shelf_question() и заведена.
        question = pipeline_smart.shelf_question(brief, phrase)
        if brief:
            print(f"      бриф: {brief}")
        elif question:
            print(f"      бриф: — спрашиваем ФРАЗОЙ блока (полка сравнивает "
                  f"описание с изображениями, ей запрос секции не нужен)")
        else:
            print("      бриф: — и фразы нет: слот пойдёт на авторский запрос секции")
            continue
        if not shelf_ok:
            continue
        res = shelf_index.search(question, limit=top)
        agr = shelf_index.name_agreement(question)
        if agr is not None:
            hits, seen = agr
            mark = "" if hits else "   <-- полка отвечает НЕ О ТОМ, что просил бриф"
            print(f"      имя предмета совпало с брифом: {hits}/{seen}{mark}")
        if not res:
            print("      полка: НИЧЕГО — переписать бриф либо признать, "
                  "что предмета на полке нет")
            continue
        for r in res:
            print(f"        {r['score']:+.3f}  {r.get('name')} — {r.get('title')} "
                  f"[{r.get('culture')} {r.get('b')}-{r.get('e')}]")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_dir")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="разобрать только первые N юнитов")
    ap.add_argument("--missing-only", action="store_true",
                    help="показать только юниты БЕЗ брифа")
    a = ap.parse_args()
    raise SystemExit(review(a.video_dir, top=a.top, missing_only=a.missing_only,
                            limit=a.limit))


if __name__ == "__main__":
    main()
