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

# Тот же довод, что у shot_brief_director.py рядом: AMBIENCE_VETO_BRAIN=
# cloud читает ANYMODEL_API_KEY из окружения, а собственный CLI-процесс
# этого файла .env никто не грузил — без строки ниже ключ из .env не
# виден. override=False — уже заданную переменную (CI, экспорт руками)
# не перезатирает.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
except ImportError:
    pass

import script_parser  # noqa: E402
import ambience_plan  # noqa: E402
from shot_brief_director import (  # noqa: E402
    episode_context, LocalBrain, FileBrain, CloudBrain, find_model, _clean,
)

PROMPT_VERSION = 2
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
    # v2 (17.09) — два правила добавлены по РЕАЛЬНЫМ промахам первого замера
    # на эпизоде 02, а не заранее: (1) модель путала анализ/сравнение
    # («убери грязь — рыцарь бы встал», «при Куртре...») с показом сцены —
    # BLOCK 9 получил атмосферу за упоминание грязи/воды В СРАВНЕНИИ, хотя
    # чтения текста достаточно, чтобы увидеть — это рассуждение, а не сцена;
    # (2) модель предлагала «heavy footfalls in muddy field» — ритмичный,
    # единичный звук, а не текстуру, которую можно зациклить на минуты
    # (BLOCK 4, BLOCK 5). Оба правила — прямой запрет ИМЕННО того, что
    # реально случилось, а не общие пожелания.
    lines = [
        "Ты звукорежиссёр документального YouTube-ролика. Ниже — текст ОДНОЙ "
        "главы дословно, целиком.",
        f"Тема эпизода: {packet['episode_title'] or '(не указана)'}.",
        f"Ниша канала: {packet['niche'] or '(не указана)'}." if packet["niche"] else "",
        "",
        "Реши: нужен ли под этой главой тихий непрерывный фоновый звук МЕСТА "
        "(атмосфера) — например ветер в открытом поле, костёр, шум леса, гул "
        "большого каменного зала, дождь, гомон рынка.",
        "",
        "Атмосфера уместна, ТОЛЬКО если верны ОБА условия:",
        "1) глава показывает КОНКРЕТНУЮ ДЛЯЩУЮСЯ СЦЕНУ в определённом "
        "месте — а не рассуждение, историческое сравнение примеров или "
        "разбор причин. Если место упомянуто только для сравнения "
        "(«как было при таком-то сражении», «убери это условие — и...») "
        "или просто как один из фактов в перечислении — это НЕ показ "
        "сцены, даже если слово «поле»/«грязь»/«вода» встречается несколько раз.",
        "2) звук — НЕПРЕРЫВНАЯ ФОНОВАЯ ТЕКСТУРА, которую можно зациклить "
        "на несколько минут БЕЗ ощущения повтора одного и того же события "
        "(ветер, огонь, дождь, гул помещения, шум леса, гомон толпы) — а "
        "НЕ единичное, редкое или ритмичное действие (шаги, один удар, "
        "крик, выстрел, разговор). Если из сцены слышны только отдельные "
        "события, а не ровный фон, — атмосфера не нужна, даже если сцена "
        "реальная.",
        "",
        "Ответ — РОВНО ОДНА строка:",
        "NONE — если атмосфера не нужна;",
        "SCENE: <3-6 слов по-английски> — непрерывная текстура места, "
        "именно то, что звучит НЕПРЕРЫВНО ВОКРУГ сцены (пример: "
        "\"wind blowing over open snowy field\", "
        "\"wood fire crackling in a stone hall\"). НЕ называй единичное "
        "действие вроде шагов или удара — это не текстура.",
        "Без пояснений, без второй строки, без markdown.",
        "", "ТЕКСТ ГЛАВЫ:", packet["text"],
    ]
    return "\n".join(x for x in lines if x != "")


_SCENE_RE = re.compile(r"SCENE\s*:\s*(.+)", re.I)

# ПРОВЕРКА ФОРМЫ ПОВЕРХ СУЖДЕНИЯ МОДЕЛИ — не полагаемся на то, что она сама
# соблюдает правило «текстура, не разовое действие» (промпт v2 просит это
# прямо, но живой замер на эпизоде 02 показал: BLOCK 4 послушался («wind
# blowing over muddy battlefield»), BLOCK 5 — нет («steady footfall on muddy
# ground»), НА ПОЧТИ ОДИНАКОВОМ ТЕКСТЕ. Значит просьбой в промпте это не
# закрывается надёжно — нужна отдельная, детерминированная проверка ФОРМЫ
# ответа, тот же приём, что `brief_is_shot_like` уже применяет к брифам
# кадра: слова, а не суждение, решают форму. Список слов взят из РЕАЛЬНО
# увиденных ответов модели (footfall/footfalls), плюс очевидные соседи того
# же класса (единичные удары/крики/выстрелы) — не с потолка.
DISCRETE_EVENT_RE = re.compile(
    r"\b(footfall|footfalls|footstep|footsteps|walking|stomp|stomping|"
    r"thud|impact|strike|striking|clash|clashing|knock|knocking|bang|"
    r"shout|shouting|scream|screaming|gunshot|explosion|single|one[- ]time)\b",
    re.I)


def parse_answer(raw):
    """(нужна_ли_атмосфера, сцена_или_None, причина_отказа_или_None).

    Мусор от модели -> (False, None, "empty"/"none") — тот же принцип
    fail-closed, что и у shot_brief_director: не угадывать смысл невнятного
    ответа, честно считать его отказом. Сцена, описывающая РАЗОВОЕ действие
    (см. DISCRETE_EVENT_RE) отклоняется здесь же, а не оставляется на
    совесть модели — она уже доказала, что не следует этому правилу
    надёжно на все 100%.
    """
    if not raw:
        return False, None, "empty"
    first_line = _clean_text(raw.splitlines()[0]) if raw.strip() else ""
    if not first_line:
        # иногда модель кладёт пустую первую строку — берём первую непустую
        for ln in raw.splitlines():
            if _clean_text(ln):
                first_line = _clean_text(ln)
                break
    if re.match(r"^none\b", first_line, re.I):
        return False, None, "model_said_none"
    m = _SCENE_RE.search(first_line) or _SCENE_RE.search(raw)
    if not m:
        return False, None, "unparsed"
    scene = _clean_text(m.group(1)).strip(" .\"'")
    if not scene:
        return False, None, "empty_scene"
    if DISCRETE_EVENT_RE.search(scene):
        return False, None, f"rejected_discrete_event:{scene}"
    return True, scene, None


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
        wants, scene, reject_reason = parse_answer(raw)
        dict_kind = dictionary_verdict(packet["text"])
        rows.append({
            "section": packet["section"],
            "dictionary_kind": dict_kind,
            "model_wants_ambience": wants,
            "model_scene": scene,
            "reject_reason": reject_reason,
            "agree": bool(dict_kind) == wants,
            "from_cache": from_cache,
            "raw": raw.strip()[:200],
        })
    return rows


def build_veto_fn(video_dir, model_path=None, threads=4, cache_dir=None,
                   brain=None):
    """(текст главы) -> True/False — согласна ли модель со словарём, что
    атмосфера здесь нужна. Нет модели -> None, вызывающий код обязан
    оставить решение словаря как есть, а не трактовать отсутствие модели
    как «нет атмосферы».

    ТОЛЬКО ВЕТО, НЕ ЗАМЕНА: `ambience_plan.plan_ambience()` зовёт этот
    резолвер лишь тогда, когда словарь УЖЕ решил, что атмосфера нужна, и
    может по его ответу только убрать её, никогда не добавить то, что
    словарь не предложил сам. Схема доказуемо не может стать хуже словаря:
    живой прогон на эпизоде 02 (13 глав) дал те же 10 верных из 13, что и
    полная замена словаря моделью, но БЕЗ её единственной ошибки (глава
    «ДЕЛО НЕ В ГРЯЗИ» — историческое сравнение битв, которое модель дважды
    подряд путала со сценой, хотя то же правило верно сработало на очень
    похожих главах 10/11). AMBIENCE_LLM_VETO=0/1, дефолт `0`.

    МОЗГ — тот же ПЕРЕИСПОЛЬЗУЕМЫЙ харнесс, что у `shot_brief_director.py`
    (не вторая копия): `brain` можно передать явно (любой объект с
    `.ask(prompt, chapter_no)`), а без него — свобода переключения ОДНОЙ
    переменной `.env`, не правкой кода. `AMBIENCE_VETO_BRAIN=cloud`
    поднимает `CloudBrain` (`AMBIENCE_VETO_CLOUD_MODEL`, по умолчанию —
    та же измеренная дешёвая модель, что у режиссёра брифов); без неё или
    при `local` — прежнее поведение байт-в-байт, локальная модель как раньше.
    """
    if brain is None:
        if (os.environ.get("AMBIENCE_VETO_BRAIN") or "local").strip().lower() == "cloud":
            if not os.environ.get("ANYMODEL_API_KEY"):
                print("  ВНИМАНИЕ: AMBIENCE_VETO_BRAIN=cloud, но "
                      "ANYMODEL_API_KEY не задан — вето пропускается, "
                      "решение словаря остаётся как есть.")
                return None
            brain = CloudBrain(os.environ.get("AMBIENCE_VETO_CLOUD_MODEL") or None)
        else:
            model = find_model(model_path)
            if not model:
                print("  ВНИМАНИЕ: AMBIENCE_LLM_VETO включён, но локальной "
                      "модели нет (python scripts/setup_local_director.py) "
                      "— вето пропускается, решение словаря остаётся как есть.")
                return None
            brain = LocalBrain(model, n_threads=threads)
    ctx = episode_context(video_dir)
    cache = cache_dir or os.path.join(video_dir, "media_plan", "ambience_veto_cache")

    def veto(text):
        packet = {"text": text, "episode_title": ctx["title"], "niche": ctx["niche"]}
        label = _clean_text(text)[:48]
        raw, _ = _ask_cached(brain, render_prompt(packet), 0, cache, True, f"вето: {label}")
        keep, why = veto_decision(raw)
        if not keep:
            return False
        if why:
            print(f"    вето без вердикта ({why}) — решение словаря остаётся: {label}")
        return True

    return veto


def veto_decision(raw):
    """(оставить_ли_атмосферу, почему_нет_вердикта_или_None) по сырому ответу.

    Вынесено из замыкания отдельной функцией ровно затем, чтобы правило
    можно было проверить тестом без живой модели на 2.3 ГБ.
    """
    wants, _, reason = parse_answer(raw)
    if wants:
        return True, None
    # Атмосфера снимается ТОЛЬКО по явному «none» модели. Всё остальное —
    # не вердикт, а ОТСУТСТВИЕ вердикта, и решение словаря остаётся.
    #
    # Два РЕАЛЬНЫХ дефекта, закрытых этой развилкой (найдены разбором 17.09,
    # оба подтверждены исходниками, а не догадкой):
    #  1. Заявленный fail-open не покрывал главный режим отказа.
    #     `LocalBrain.ask()` при сбое возвращает ПУСТУЮ строку, а не бросает
    #     исключение (это задокументировано у неё же) — значит `except
    #     Exception` в plan_ambience не срабатывал, `parse_answer("")` давал
    #     `empty`, и упавшая модель МОЛЧА снимала атмосферу у КАЖДОЙ главы.
    #     Fail-open был на бумаге, fail-closed на деле.
    #  2. `DISCRETE_EVENT_RE` отклоняет описание РАЗОВОГО действия — правило
    #     писалось для режима, где модель ВЫБИРАЕТ сцену. В режиме вето сцена
    #     выбрасывается, и «SCENE: wind over a field with distant clashing»
    #     снимало бы законный `wind_open` из-за одного слова в строке,
    #     которая на решение всё равно не влияет.
    if reason == "model_said_none":
        return False, None
    return True, reason


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_dir")
    ap.add_argument("--brain", choices=("local", "cloud", "file", "packets"),
                    default="local")
    ap.add_argument("--answers", help="папка с ответами для --brain file")
    ap.add_argument("--out-packets", help="куда выложить промпты глав для --brain packets")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cloud-model", default=None,
                    help=f"модель шлюза для --brain cloud (по умолчанию "
                         f"{CloudBrain.DEFAULT_MODEL})")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv[1:])

    blocks = script_parser.parse_blocks(os.path.join(a.video_dir, "script.txt"))

    if a.brain == "packets":
        dest = a.out_packets or os.path.join(a.video_dir, "media_plan", "ambience_packets")
        os.makedirs(dest, exist_ok=True)
        packets = chapter_packets(a.video_dir, blocks)
        for i, p in enumerate(packets, 1):
            with open(os.path.join(dest, f"{i:02d}.txt"), "w", encoding="utf-8") as f:
                f.write(render_prompt(p))
        print(f"Пакеты глав: {dest} ({len(packets)} глав). Ответы положить рядом "
              f"под теми же номерами и запустить --brain file --answers <папка>")
        return 0

    if a.brain == "cloud":
        if not os.environ.get("ANYMODEL_API_KEY"):
            print("ANYMODEL_API_KEY не задан в .env — --brain cloud "
                  "работать не может.")
            return 2
        brain = CloudBrain(a.cloud_model)
        print(f"Мозг: облако {brain.model}")
    elif a.brain == "local":
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
        scene_col = r["model_scene"] or ""
        if r["reject_reason"] and r["reject_reason"].startswith("rejected_discrete_event:"):
            scene_col = f"(отклонено фильтром: {r['reject_reason'].split(':', 1)[1]})"
        print("{:<28} {:<14} {:<9} {}{}".format(
            r["section"][:28], r["dictionary_kind"] or "—",
            "нужна" if r["model_wants_ambience"] else "нет",
            scene_col, mark))

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
