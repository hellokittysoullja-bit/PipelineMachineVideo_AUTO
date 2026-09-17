#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Атмосфера по СМЫСЛУ главы, а не по словарю слов (`ambience_plan.score_text`).

РЕЖИМ ИЗМЕРЕНИЯ, НЕ РЕНДЕРА. Этот скрипт ничего не пишет в script.txt и
не подключён к сборке звука — он только считает, что сказала бы модель, и
кладёт это рядом с тем, что уже даёт словарный скоринг, чтобы решение
«стоит ли вообще заменять/дополнять словарь» принималось по числам, а не
на глаз. Тот же принцип, каким в этом репозитории уже проверялся
`shot_brief_director.py` до его включения в прод (десять прогонов, замер
`subject_hit`, и только потом дефолт `SHOT_PLANNER_LLM=0 -> 1`).

ПОЧЕМУ МОДЕЛЬ ЛУЧШЕ СЛОВАРЯ В ПРИНЦИПЕ (гипотеза, которую этот скрипт и
проверяет). `score_text()` видит только буквальные слова из
`AMBIENCE_VOCAB` — если глава описывает открытое всеми ветрами поле без
слова «ветер» ни разу, счёт будет нулевым и атмосфера не встанет вообще,
хотя сцена ровно про то самое место. Модель читает главу целиком (тот же
приём, что уже поднял точность подбора кадра с 58 до 85 из 116 — см.
CLAUDE.md, «Пофразовый автомат удалён») и может назвать место, которого в
словаре нет вовсе, — тогда её ответ заодно подсказывает, чем расширять
`AMBIENCE_VOCAB` или библиотеку.

Мозг переиспользуется у `shot_brief_director.py` целиком: LocalBrain
(llama.cpp, тот же принцип temperature=0 + фиксированный seed),
FileBrain (человек/Claude отвечает файлами), find_model() (тот же порядок
поиска весов). Заводить вторую копию загрузки модели незачем — задача
та же по форме: глава с контекстом -> текстовый ответ.

ЗАПУСК: python scripts/ambience_director.py <video_dir>
Печатает построчное сравнение (словарь / модель) по каждой главе и
пишет media_plan/ambience_director_report.json — только отчёт, решение
о включении в рендер за владельцем, отдельным шагом, после того как
числа что-то скажут.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import script_parser  # noqa: E402
import ambience_plan  # noqa: E402
from shot_brief_director import (  # noqa: E402
    episode_context, LocalBrain, FileBrain, find_model, _clean,
)

PROMPT_VERSION = 1
MODELS_DIR_NAME = "models"


def _clean_text(s):
    return _clean(s)


def chapter_packets(video_dir, blocks):
    """Главы эпизода целиком — секция, полный текст, тема/ниша.

    Группировка по `section` — та же, что уже использует `packets()` в
    shot_brief_director.py, но без разбивки на пофразовые юниты: здесь
    решение одно на главу, а не одно на фразу.
    """
    ctx = episode_context(video_dir)
    groups, order = {}, []
    for b in blocks:
        sec = b.get("section") or "—"
        if sec not in groups:
            groups[sec] = []
            order.append(sec)
        t = _clean_text(b.get("text"))
        if t:
            groups[sec].append(t)
    out = []
    for sec in order:
        text = " ".join(groups[sec])
        if not text:
            continue
        out.append({"section": sec, "text": text,
                    "episode_title": ctx["title"], "niche": ctx["niche"]})
    return out


def render_prompt(packet):
    lines = [
        "Ты звукорежиссёр документального YouTube-ролика. Ниже — текст ОДНОЙ "
        "главы дословно, целиком.",
        f"Тема эпизода: {packet['episode_title'] or '(не указана)'}.",
        f"Ниша канала: {packet['niche'] or '(не указана)'}." if packet["niche"] else "",
        "",
        "Реши: нужен ли под этой главой тихий непрерывный фоновый звук МЕСТА "
        "(атмосфера) — например ветер в открытом поле, костёр, шум леса, гул "
        "большого каменного зала, дождь, гомон рынка. Атмосфера уместна, "
        "когда глава явно происходит в конкретном физическом месте на "
        "протяжении хотя бы нескольких предложений подряд. НЕ уместна, если "
        "глава — это прямое обращение к зрителю, отвлечённое рассуждение, "
        "перечисление фактов без сцены, или место в главе меняется почти "
        "на каждой фразе.",
        "",
        "Ответь РОВНО ОДНОЙ строкой:",
        "NONE — если атмосфера не нужна;",
        "SCENE: <3-6 слов по-английски> — если нужна, опиши место так, как "
        "искал бы настоящую полевую звукозапись (пример: "
        "\"wind blowing over open snowy field\", "
        "\"wood fire crackling in a stone hall\").",
        "Без пояснений, без второй строки, без markdown.",
        "", "ТЕКСТ ГЛАВЫ:", packet["text"],
    ]
    return "\n".join(x for x in lines if x != "")


_SCENE_RE = re.compile(r"SCENE\s*:\s*(.+)", re.I)


def parse_answer(raw):
    """(нужна_ли_атмосфера, сцена_или_None). Мусор от модели -> (False, None)
    — тот же принцип fail-closed, что и у shot_brief_director: не угадывать
    смысл невнятного ответа, честно считать его отказом."""
    if not raw:
        return False, None
    first_line = _clean_text(raw.splitlines()[0]) if raw.strip() else ""
    if not first_line:
        # иногда модель кладёт пустую первую строку — берём первую непустую
        for ln in raw.splitlines():
            if _clean_text(ln):
                first_line = _clean_text(ln)
                break
    if re.match(r"^none\b", first_line, re.I):
        return False, None
    m = _SCENE_RE.search(first_line) or _SCENE_RE.search(raw)
    if m:
        scene = _clean_text(m.group(1)).strip(" .\"'")
        return bool(scene), (scene or None)
    return False, None


def _cache_key_text(text, brain_name):
    h = hashlib.md5()
    h.update(f"a{PROMPT_VERSION}\x00{brain_name}\x00".encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _ask_cached(brain, prompt_text, chapter_no, cache_dir, verbose, label):
    key = _cache_key_text(prompt_text, brain.name)
    path = os.path.join(cache_dir, key + ".txt") if cache_dir else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read(), True
    t0 = time.time()
    raw = brain.ask(prompt_text, chapter_no)
    if verbose:
        print(f"  {label:<48} {time.time() - t0:5.0f}с")
    if path and raw:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(raw)
        os.replace(tmp, path)
    return raw, False


def dictionary_verdict(text):
    """То, что сегодня реально решает в проде (`ambience_plan.score_text`),
    с теми же порогами, что `plan_ambience()` использует для одной главы —
    но БЕЗ учёта длины/соседей (это сравнение решения по тексту, не полного
    плана по таймлайну)."""
    scores = ambience_plan.score_text(text)
    if not scores:
        return None
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    top_name, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0
    if top_score < ambience_plan.AMBIENCE_MIN_SCORE:
        return None
    if top_score - second_score < ambience_plan.AMBIENCE_AMBIGUITY_MARGIN:
        return None
    return top_name


def run(video_dir, blocks, brain, cache_dir=None, verbose=True):
    rows = []
    for chapter_no, packet in enumerate(chapter_packets(video_dir, blocks), 1):
        label = _clean_text(packet["section"])[:48]
        raw, from_cache = _ask_cached(brain, render_prompt(packet), chapter_no,
                                      cache_dir, verbose, label)
        wants, scene = parse_answer(raw)
        dict_kind = dictionary_verdict(packet["text"])
        rows.append({
            "section": packet["section"],
            "dictionary_kind": dict_kind,
            "model_wants_ambience": wants,
            "model_scene": scene,
            "agree": bool(dict_kind) == wants,
            "from_cache": from_cache,
            "raw": raw.strip()[:200],
        })
    return rows


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_dir")
    ap.add_argument("--brain", choices=("local", "file"), default="local")
    ap.add_argument("--answers", help="папка с ответами для --brain file")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv[1:])

    blocks = script_parser.parse_blocks(os.path.join(a.video_dir, "script.txt"))

    if a.brain == "local":
        model = find_model(a.model)
        if not model:
            print("Модели нет. python scripts/setup_local_director.py")
            return 2
        print(f"Мозг: {os.path.basename(model)}")
        brain = LocalBrain(model, n_threads=a.threads)
    else:
        if not a.answers:
            print("Нужна --answers <папка>")
            return 2
        brain = FileBrain(a.answers)

    cache = None if a.no_cache else os.path.join(
        a.video_dir, "media_plan", "ambience_director_cache")
    rows = run(a.video_dir, blocks, brain, cache_dir=cache)

    print("\n{:<28} {:<14} {:<9} {}".format("ГЛАВА", "СЛОВАРЬ", "МОДЕЛЬ", "СЦЕНА (модель)"))
    disagreements = 0
    for r in rows:
        mark = "" if r["agree"] else "  <-- РАСХОЖДЕНИЕ"
        if not r["agree"]:
            disagreements += 1
        print("{:<28} {:<14} {:<9} {}{}".format(
            r["section"][:28], r["dictionary_kind"] or "—",
            "нужна" if r["model_wants_ambience"] else "нет",
            r["model_scene"] or "", mark))

    report_path = os.path.join(a.video_dir, "media_plan", "ambience_director_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"prompt_version": PROMPT_VERSION, "brain": brain.name,
                   "chapters": len(rows), "disagreements": disagreements,
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nГлав {len(rows)}, расхождений словарь/модель: {disagreements}")
    print(f"Отчёт: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
