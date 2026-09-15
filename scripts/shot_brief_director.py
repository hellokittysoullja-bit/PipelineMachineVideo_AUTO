# -*- coding: utf-8 -*-
"""Режиссёрская разработка ГЛАВЫ: контекст вместо одиночной фразы.

ЗАЧЕМ. `shot_planner_llm.plan_unit(text)` спрашивает модель об ОДНОЙ фразе,
вырванной из эпизода. Пайплайн при этом уже считает всё, чего этой фразе не
хватает — соседей (`semantic_context_text`), заголовок главы (`b["section"]`),
драматургическую стадию (`media_plan/speech_plan.json`), тему эпизода
(`=== METADATA ===`) — и до режиссёра не доносит ни одного сигнала.

Цена измерена на собственном замере планировщика, а не предположена. Юнит
[13] эпизода 02: «При этом ОН был под ногами у каждого из них». Модель
ответила «A man stepping onto a battlefield», и вердикт замера — промах.
Иначе она ответить не могла: «он» — это земля, названная двумя фразами
РАНЬШЕ, и в запросе этих фраз нет. Модель любого размера промахнётся здесь
одинаково; это не потолок 7B, это отсутствующий вход.

ЧТО ЗДЕСЬ. Глава целиком уходит одним вопросом: все её фразы по порядку,
заголовок главы, хвост предыдущей, тема эпизода, ниша канала. Ответ — по
строке на фразу. Отсюда четыре вещи, которых у пофразового режима нет
по построению:

  * местоимение разрешается по соседям, а не угадывается;
  * тип кадра выбирается, зная, что показывали рядом;
  * соседние кадры перестают быть одним и тем же предметом N раз подряд;
  * вызовов 13 вместо 142 — то есть тем же временем оплачивается модель
    в несколько раз крупнее.

МОЗГ СМЕНЯЕМ, ХАРНЕСС ОБЩИЙ. Модуль не привязан к llama.cpp: он собирает
пакеты, разбирает ответ и проверяет заявки. Кто отвечает — локальная
модель (`--brain local`), человек или Claude, работающий над сценарием
(`--brain file`, `--brain packets`) — на проверку и на формат плана не
влияет. План пишется в тот же `media_plan/shot_plan.json`, который уже
читает `pipeline_smart`, поэтому нового пути отбора не заводится.

ФОРМАТ ОТВЕТА — СТРОКИ, А НЕ ОДИН JSON, и это не вкусовщина. Заявок в
главе 8-16; один JSON-объект на всю главу теряется ЦЕЛИКОМ от одной
сорванной запятой, а построчный разбор теряет ровно ту строку, которая
сломалась. Для 7B на 4 ядрах это разница между «глава есть» и «главы нет».

ЧЕГО МОДУЛЬ НЕ ДЕЛАЕТ. Он не проверяет факты и не решает, какой кадр
победит, — только называет, что просить. Бриф автора не перезаписывается
никогда: он проверен человеком, заявка модели — нет.
"""
import argparse
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import script_parser        # noqa: E402
import shot_planner_llm     # noqa: E402

# Версия ПАКЕТА и разбора. Входит в ключ кэша главы: переписанный пакет
# обязан считаться заново, иначе план молча останется от прошлой
# формулировки — тот же класс, что уже закрыт у кэша вердиктов арбитра.
PACKET_VERSION = 1

PLAN_NAME = shot_planner_llm.PLAN_NAME
CACHE_DIR_NAME = "shot_brief_cache"

# Пример в промпте — НА ЧУЖОЙ ТЕМЕ (Рим, дороги) осознанно. Пример из
# этого же эпизода подсказал бы модели готовые ответы ровно на тех
# фразах, на которых её потом меряют, и замер перестал бы что-либо
# значить.
FEWSHOT = """ПРИМЕР (другая тема, чтобы было видно формат):
фразы:
 1. Рим держал границу не стенами, а дорогами.
 2. По ним легион проходил тридцать километров в день.
 3. Запомни эту цифру. Она пригодится.
ответ:
 1 | scene | a straight roman stone road running across empty hills
 2 | object | a hobnailed roman military sandal on stone paving, close up
 3 | - | -"""

RULES = """ПРАВИЛА
1. Кадр — это ПРЕДМЕТ ИЛИ СЦЕНА, которую можно сфотографировать. Не пересказ
   фразы и не её перевод. Нельзя писать «you have not been injured» или
   «this will be revisited» — это не кадр.
2. Ты видишь ВСЮ главу сразу. Если во фразе «он», «это», «тот», «она» —
   найди в соседних фразах, О ЧЁМ речь, и назови сам предмет. Местоимений
   в описании кадра быть не должно.
3. ЭПОХА обязательна. Человек в кадре — knight, warrior, archer, armourer
   своего времени, никогда не «a person» и не «a man». Никакой современной
   одежды, техники и снаряжения: «protective gear» приведёт современный
   костюм.
4. СРАВНЕНИЯ НЕ ПОКАЗЫВАЙ. «Весил как холодильник», «поднимали краном»,
   «как перевёрнутая черепаха» — образы речи. Показывать надо то, О ЧЁМ
   речь (доспех, упавший воин), а не предмет сравнения.
5. СОСЕДНИЕ КАДРЫ РАЗНЫЕ. Глава — последовательность разных кадров, а не
   один предмет подряд. Меняй и предмет, и крупность.
6. Показывать нечего — поставь «-» вместо описания. Обращение к зрителю,
   связка, обещание вернуться к теме: пустой кадр честнее выдуманного."""


def _clean(s):
    return " ".join((s or "").split())


def episode_context(video_dir):
    """Тема эпизода и ниша канала — одной строкой каждая.

    Fail-open: файлов нет — режиссёр работает без них, просто хуже.
    """
    title = ""
    try:
        with open(os.path.join(video_dir, "script.txt"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("TITLE:"):
                    title = _clean(line.split(":", 1)[1])
                    break
    except OSError:
        pass
    niche = ""
    try:
        with open(os.path.join(REPO, "CHANNEL.md"), encoding="utf-8") as f:
            body = f.read()
        m = re.search(r"##\s*1\.\s*Ниша\s*\n(.+)", body)
        if m:
            niche = _clean(m.group(1))[:220]
    except OSError:
        pass
    return {"title": title, "niche": niche}


def arc_stages(video_dir):
    """`arc_stage` по тексту юнита из speech_plan.json, если он есть.

    Ключ — ТЕКСТ, а не номер юнита: номера сдвигаются от любой правки
    сценария выше по тексту, и стадия молча описывала бы чужую фразу.
    Ровно тот дефект, от которого уже защищается lock в шотлисте.
    """
    path = os.path.join(video_dir, "media_plan", "speech_plan.json")
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for u in data.get("units", []):
            t = _clean(u.get("text"))
            if t and u.get("arc_stage"):
                out[t] = u["arc_stage"]
    except Exception:
        return {}
    return out


def section_queries(video_dir):
    """Запрос секции — то, чем слот обходился ДО режиссёра.

    Он уходит в пакет не как образец для подражания, а как известная
    рамка темы: модель видит, про что глава, словами, которые уже
    доказали, что приносят хоть что-то.
    """
    out = {}
    try:
        with open(os.path.join(video_dir, "script.txt"), encoding="utf-8") as f:
            body = f.read()
    except OSError:
        return out
    m = re.search(r"===\s*PEXELS QUERIES\s*===(.*?)(?:\n===|\Z)", body, re.S)
    if not m:
        return out
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[_clean(key).upper()] = _clean(val)
    return out


def _section_key(section):
    """"BLOCK 1: Ложь первая" -> "BLOCK_1" — так секции названы в
    PEXELS QUERIES. Второго разбора заголовка не заводится."""
    head = _clean(section).split(":", 1)[0]
    return head.replace(" ", "_").upper()


def packets(video_dir, blocks, max_units=None, use_vocabulary=False):
    """Главы эпизода как самостоятельные вопросы к режиссёру."""
    ctx = episode_context(video_dir)
    stages = arc_stages(video_dir)
    queries = section_queries(video_dir)

    groups, order = {}, []
    for i, b in enumerate(blocks):
        sec = b.get("section") or "—"
        if sec not in groups:
            groups[sec] = []
            order.append(sec)
        groups[sec].append((i, b))

    out, prev_tail = [], ""
    for sec in order:
        items = groups[sec]
        units = enumerate_units(items, stages, max_units)
        if not units:
            continue
        out.append({
            "section": sec,
            "use_vocabulary": use_vocabulary,
            "section_query": queries.get(_section_key(sec), ""),
            "episode_title": ctx["title"],
            "niche": ctx["niche"],
            "prev_tail": prev_tail,
            "units": units,
        })
        prev_tail = _clean(items[-1][1].get("text"))[:180]
    return out


def enumerate_units(items, stages, max_units):
    """Юниты главы в том виде, в каком их видит режиссёр."""
    res = []
    for n, (idx, b) in enumerate(items, 1):
        text = _clean(b.get("text"))
        if not text:
            continue
        res.append({
            "n": n,
            "block_index": idx,
            "text": text,
            "arc_stage": stages.get(text),
            "stat": b.get("stat"),
            "is_climax": bool(b.get("is_climax")),
            "author_brief": _clean(b.get("shot_brief")) or None,
        })
        if max_units and len(res) >= max_units:
            break
    return res


def corpus_vocabulary(packet, limit=36):
    """Настоящие имена предметов из каталога музея по теме этой главы.

    ЗАЧЕМ. CLAUDE.md держит этот шаг открытым прямым текстом: «словарь
    автора и словарь музея — разные», «шаг слово автора -> слово каталога
    ПОКА НЕ СДЕЛАН». Измеряется это мгновенно: `poleaxe` в каталоге Мет
    встречается НОЛЬ раз (предмет лежит под `Halberd` и `Partisan`),
    `longsword` — тоже ноль, а «rondel dagger» музей пишет как «Roundel
    dagger». Бриф, написанный словом, которого у корпуса нет, обречён ещё
    до всякого скоринга.

    Здесь список реальных имён предметов подаётся режиссёру ДО того, как
    он напишет бриф. Это превращает часть задачи из свободной генерации
    (где модель ошибается) в выбор из названного (где ошибается сильно
    реже) — тот же вывод, к которому пришла работа «From Shots to
    Stories» про LLM в монтаже: сходящиеся задачи модель решает заметно
    лучше расходящихся.

    Fail-open: индекса нет — список пуст, промпт возвращается к прежнему
    виду БАЙТ-В-БАЙТ.
    """
    try:
        import met_catalog
        if not met_catalog.available():
            return []
    except Exception:
        return []
    seed = " ".join([re.sub(r"\[[^\]]*\]", " ", packet.get("section_query") or ""),
                     _clean(packet.get("section"))])
    names, seen = [], set()
    try:
        for row in met_catalog.search(seed, limit=400):
            nm = _clean(row.get("name"))
            if not nm or len(nm) > 40:
                continue
            key = nm.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(nm)
            if len(names) >= limit:
                break
    except Exception:
        return []
    return names


def render_prompt(packet):
    """Один вопрос на главу. Всё, что знает пайплайн, — здесь."""
    head = ["Ты — визуальный редактор исторического документального ролика.",
            "Тебе дают ГЛАВУ ЦЕЛИКОМ: все фразы диктора по порядку.",
            "На каждую фразу назови ОДИН кадр — что физически показать на экране.",
            ""]
    if packet.get("niche"):
        head.append(f"КАНАЛ: {packet['niche']}")
    if packet.get("episode_title"):
        head.append(f"ЭПИЗОД: {packet['episode_title']}")
    head.append(f"ГЛАВА: {_clean(packet['section'])}")
    if packet.get("section_query"):
        head.append(f"О ЧЁМ ГЛАВА (рамка темы): {packet['section_query']}")
    if packet.get("prev_tail"):
        head.append(f"ЧЕМ КОНЧИЛАСЬ ПРЕДЫДУЩАЯ ГЛАВА: «{packet['prev_tail']}»")

    lines = [
        "\n".join(head), "", RULES, "",
        "ТИП КАДРА — одно из: object | scene | illustration | map | texture",
        "",
        "ФОРМАТ ОТВЕТА: ровно по одной строке на фразу, ничего до и после.",
        "номер | тип | описание кадра по-английски, 4-12 слов",
        "", FEWSHOT, "", "ФРАЗЫ ГЛАВЫ:",
    ]
    for u in packet["units"]:
        mark = []
        if u.get("stat"):
            mark.append(f"на экране цифра: {u['stat']}")
        if u.get("is_climax"):
            mark.append("кульминация главы")
        if u.get("arc_stage"):
            mark.append(f"стадия: {u['arc_stage']}")
        tail = f"   ({'; '.join(mark)})" if mark else ""
        lines.append(f" {u['n']}. {u['text']}{tail}")
    vocab = corpus_vocabulary(packet) if packet.get("use_vocabulary") else []
    if vocab:
        lines.append("")
        lines.append("СЛОВАРЬ МУЗЕЯ. Эти предметы в корпусе РЕАЛЬНО есть под "
                     "этими именами. Если кадр про один из них — называй его "
                     "ИМЕННО так:")
        lines.append("  " + ", ".join(vocab))
    lines.append("")
    lines.append(f"Ответь {len(packet['units'])} строками — по одной на каждую фразу.")
    return "\n".join(lines)


_ROW_RE = re.compile(r"^\s*\**\s*(\d{1,3})\s*[|.)]\s*(.*)$")


def parse_answer(raw, packet):
    """Ответ модели -> {номер фразы: заявка}. Битая строка теряет себя одну."""
    got = {}
    valid_n = {u["n"] for u in packet["units"]}
    for line in shot_planner_llm._clean_stream(raw or "").splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if n not in valid_n or n in got:
            continue
        rest = m.group(2)
        parts = [p.strip() for p in rest.split("|")]
        if len(parts) >= 2:
            fn, shot = parts[0].lower(), "|".join(parts[1:]).strip()
        else:
            fn, shot = "", parts[0]
        shot = _clean(shot.strip(" *`"))
        if not shot or shot in {"-", "—", "null", "none"}:
            continue
        if fn not in shot_planner_llm.VALID_FUNCTIONS:
            fn = None
        if not shot_planner_llm._LATIN_RE.search(shot):
            continue
        if not (2 <= len(shot.split()) <= 16):
            continue
        got[n] = {"shot_en": shot, "function": fn, "forbidden": None,
                  "subject": None}
    return got


# --- МОЗГИ ------------------------------------------------------------------
#
# Харнесс не знает, кто отвечает. Локальная модель, человек или Claude,
# пишущий сценарий, — на сборку пакета, разбор, проверку и формат плана
# это не влияет. Поэтому «взять модель получше» перестаёт быть правкой
# кода и становится сменой одного аргумента.

class LocalBrain:
    """llama.cpp на CPU. Загружается ОДИН раз на прогон, а не на вызов.

    Пофразовый режим платил загрузку модели заново на каждый вызов через
    подпроцесс `llama-cli` (замер: ~18 с). На 142 юнита это 40 минут
    ровно ни за что. Здесь модель живёт весь прогон, а вызовов и так 13.
    """

    def __init__(self, model_path, n_threads=4, n_ctx=8192, seed=1,
                 max_tokens=900):
        from llama_cpp import Llama
        self.name = os.path.basename(model_path)
        self.seed = seed
        self.max_tokens = max_tokens
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_threads=n_threads, seed=seed, verbose=False)

    def ask(self, prompt, chapter_no):
        # temperature=0 + фиксированный seed: два прогона на одном вопросе
        # обязаны дать один ответ, иначе сравнение версий недействительно
        # (урок замера 16.09, где temp стояла 0.2 при случайном seed).
        r = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, seed=self.seed, max_tokens=self.max_tokens)
        return r["choices"][0]["message"]["content"] or ""


class FileBrain:
    """Ответы лежат файлом: по файлу на главу, имя — номер главы.

    Это путь для мозга, которого нельзя запустить подпроцессом, — человека
    или Claude, который и так пишет сценарий этого эпизода. Проверка,
    формат плана и запись в script.txt у него РОВНО ТЕ ЖЕ, что у локальной
    модели: качество мозга меняется, дисциплина — нет.
    """

    def __init__(self, answers_dir):
        self.name = "file:" + os.path.basename(answers_dir.rstrip("/"))
        self.dir = answers_dir

    def ask(self, prompt, chapter_no):
        # Номер главы приходит СНАРУЖИ, а не считается внутренним счётчиком.
        # Со счётчиком любой пропуск главы (фильтр по секциям, кэш-хит,
        # пустой ответ) молча сдвигал бы все последующие ответы на одну
        # главу — то есть кадры встали бы под чужие фразы. Ровно тот
        # дефект, ради которого `[shot:]` сделан инлайновым.
        path = os.path.join(self.dir, f"{chapter_no:02d}.txt")
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""


# --- ПРОГОН -----------------------------------------------------------------

STATS = {"chapters": 0, "asked": 0, "cache_hits": 0, "rows": 0,
         "rejected": 0, "empty_chapters": 0}
REJECTED = []


def _cache_key(packet, brain_name):
    import hashlib
    h = hashlib.md5()
    h.update(f"p{PACKET_VERSION}\x00{brain_name}\x00".encode("utf-8"))
    h.update(render_prompt(packet).encode("utf-8"))
    return h.hexdigest()[:16]


def run(video_dir, blocks, brain, cache_dir=None, verbose=True,
        max_units=None, only_sections=None, use_vocabulary=False):
    """Пройти эпизод главами. Возвращает {индекс блока: заявка}."""
    out = {}
    for chapter_no, packet in enumerate(
            packets(video_dir, blocks, max_units=max_units,
                    use_vocabulary=use_vocabulary), 1):
        if only_sections and _section_key(packet["section"]) not in only_sections:
            continue
        STATS["chapters"] += 1
        raw, key = None, _cache_key(packet, brain.name)
        path = os.path.join(cache_dir, key + ".txt") if cache_dir else None
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    raw = f.read()
                STATS["cache_hits"] += 1
            except OSError:
                raw = None
        if raw is None:
            t0 = time.time()
            raw = brain.ask(render_prompt(packet), chapter_no)
            STATS["asked"] += 1
            if verbose:
                print(f"  {_clean(packet['section'])[:52]:<52} "
                      f"{len(packet['units']):2d} фраз  {time.time() - t0:5.0f}с")
            if path and raw:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        f.write(raw)
                    os.replace(tmp, path)
                except OSError:
                    pass
        rows = parse_answer(raw, packet)
        if not rows:
            STATS["empty_chapters"] += 1
        by_n = {u["n"]: u for u in packet["units"]}
        for n, got in rows.items():
            unit = by_n[n]
            # Та же проверка, что у пофразового режиссёра. Второго свода
            # правил не заводится: разойдись они — одна и та же заявка
            # проходила бы в одном режиме и отклонялась в другом.
            ok, why = shot_planner_llm.brief_is_safe(got["shot_en"], unit["text"])
            if not ok:
                STATS["rejected"] += 1
                REJECTED.append({"section": packet["section"],
                                 "text": unit["text"][:80],
                                 "shot_en": got["shot_en"], "reason": why})
                continue
            out[unit["block_index"]] = dict(got, text=unit["text"])
            STATS["rows"] += 1
    return out


def write_plan(video_dir, blocks, found, brain_name):
    """План в том же файле и формате, который уже читает pipeline_smart."""
    units = {}
    for idx, got in found.items():
        text = blocks[idx].get("text") or ""
        units[shot_planner_llm.unit_key(text)] = {
            "shot_en": got["shot_en"], "function": got.get("function"),
            "forbidden": None, "subject": None}
    mp = os.path.join(video_dir, "media_plan")
    os.makedirs(mp, exist_ok=True)
    tmp = os.path.join(mp, PLAN_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": PACKET_VERSION, "planner": "shot_brief_director",
                   "model": brain_name, "stats": dict(STATS),
                   "rejected": REJECTED, "units": units},
                  f, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(mp, PLAN_NAME))
    return os.path.join(mp, PLAN_NAME)


def main(argv):
    ap = argparse.ArgumentParser(
        description="Режиссёрская разработка главы: контекст вместо фразы")
    ap.add_argument("video_dir")
    ap.add_argument("--brain", choices=("local", "file", "packets"),
                    default="packets")
    ap.add_argument("--answers", help="папка с ответами для --brain file")
    ap.add_argument("--model", default=os.environ.get("LLAMA_MODEL_GGUF", ""))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-packets", help="куда выложить промпты глав")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--vocabulary", action="store_true",
                    help="подать режиссёру реальные имена предметов из "
                         "каталога Мет (шаг «слово автора -> слово каталога»)")
    a = ap.parse_args(argv[1:])

    blocks = script_parser.parse_blocks(os.path.join(a.video_dir, "script.txt"))

    if a.brain == "packets":
        dest = a.out_packets or os.path.join(a.video_dir, "media_plan",
                                             "shot_packets")
        os.makedirs(dest, exist_ok=True)
        for i, p in enumerate(packets(a.video_dir, blocks), 1):
            with open(os.path.join(dest, f"{i:02d}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(render_prompt(p))
        print(f"Пакеты глав: {dest} ({i} глав). Ответы положить рядом "
              f"под теми же номерами и запустить --brain file --answers <папка>")
        return 0

    if a.brain == "local":
        if not a.model or not os.path.exists(a.model):
            print("Нет модели: --model <файл.gguf> или LLAMA_MODEL_GGUF")
            return 2
        brain = LocalBrain(a.model, n_threads=a.threads)
    else:
        if not a.answers:
            print("Нужна --answers <папка с ответами>")
            return 2
        brain = FileBrain(a.answers)

    cache = None if a.no_cache else os.path.join(a.video_dir, "media_plan",
                                                 CACHE_DIR_NAME)
    found = run(a.video_dir, blocks, brain, cache_dir=cache,
                use_vocabulary=a.vocabulary)
    path = write_plan(a.video_dir, blocks, found, brain.name)
    print(f"\nГлав {STATS['chapters']}, вопросов {STATS['asked']}, "
          f"из кэша {STATS['cache_hits']}, заявок {STATS['rows']}, "
          f"отклонено проверкой {STATS['rejected']}, "
          f"глав без единого разбора {STATS['empty_chapters']}")
    print(f"План: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
