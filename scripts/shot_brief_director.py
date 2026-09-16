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

import feature_flags         # noqa: E402
import script_parser        # noqa: E402
import shot_planner_llm     # noqa: E402

# Версия ПАКЕТА и разбора. Входит в ключ кэша главы: переписанный пакет
# обязан считаться заново, иначе план молча останется от прошлой
# формулировки — тот же класс, что уже закрыт у кэша вердиктов арбитра.
PACKET_VERSION = 6

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
MOOD | 0 | 1 | спокойное объяснение
 1 | scene | a straight roman stone road running across empty hills
 2 | object | a hobnailed roman military sandal on stone paving, close up
 3 | - | -"""

# Ядро правил НИШИ НЕ ЗНАЕТ. До 15.09 третье правило требовало рыцарей
# буквально, и клон репозитория под психологию или игры получал бы чужой
# мир кадра. Теперь мир объявляет канал (`channel_profile.json` ->
# `shot_domain`), и если не объявил — доменного правила просто нет.
RULES_CORE = """ПРАВИЛА
1. Кадр — это ПРЕДМЕТ ИЛИ СЦЕНА, которую можно сфотографировать. Не пересказ
   фразы и не её перевод. Нельзя писать «you have not been injured» или
   «this will be revisited» — это не кадр.
2. Ты видишь ВСЮ главу сразу. Если во фразе «он», «это», «тот», «она» —
   найди в соседних фразах, О ЧЁМ речь, и назови сам предмет. Местоимений
   в описании кадра быть не должно.
3. МИМОЛЁТНОЕ СРАВНЕНИЕ НЕ ПОКАЗЫВАЙ. «Весил как холодильник»,
   «поднимали краном», «как перевёрнутая черепаха» — брошенный образ,
   речь дальше идёт о другом. Показывать надо то, О ЧЁМ речь, а не
   предмет сравнения.
   НО: если сравнение занимает ФРАЗУ ЦЕЛИКОМ и рассказ его разворачивает
   («срабатывает то же, что у человека, который обжёгся о плиту: рука
   отдёргивается раньше мысли») — оно и есть кадр этой фразы. Разница не
   в наличии «как», а в том, на чём держится внимание: на секунду или всю
   фразу.
4. СОСЕДНИЕ КАДРЫ РАЗНЫЕ. Глава — последовательность разных кадров, а не
   один предмет подряд. Меняй и предмет, и крупность.
5. Показывать нечего — поставь «-» вместо описания. Обращение к зрителю,
   связка, обещание вернуться к теме: пустой кадр честнее выдуманного.
6. АБСТРАКЦИЮ ПОКАЗЫВАЮТ СИТУАЦИЕЙ, А НЕ ПОНЯТИЕМ. Если фраза про чувство,
   идею или процесс — «тревога», «выгорание», «доверие рушится», «он
   потерял смысл» — понятие сфотографировать нельзя. Назови одно из трёх,
   в чём оно ВИДНО:
     СИТУАЦИЯ, в которой это происходит (пустая приёмная перед дверью врача);
     ТЕЛЕСНЫЙ признак (сжатые на столе руки, ссутуленная спина, взгляд в пол);
     ПРЕДМЕТ-СЛЕД (нетронутая еда, телефон экраном вниз, две чашки — одна полная).
   Определение показать нельзя, ситуацию — можно.
   И НИКАКИХ СИМВОЛОВ. Камень, «символизирующий груз», клубок ниток,
   песочные часы, разорванная цепь — это не кадр про человека, это
   штамп из фотобанка, и зритель считывает его как заглушку. Нужна
   вещь, которая РЕАЛЬНО стоит в этой комнате у этого человека.
7. ПОКАЗЫВАЙ НОВОЕ, А НЕ ГЛАВНОЕ. В каждой фразе есть слово, ради
   которого она написана, — то, чего не было в предыдущей. Его и
   показывай. «Он умеет всё, что нужно для этой работы. Но сегодня у
   него дрожат РУКИ» — кадр про руки, а не про человека: человека уже
   показали фразой раньше. Самая частая ошибка — поставить на фразу её
   подлежащее, которое зритель и так видел.
8. ПРОТИВОПОСТАВЛЕНИЕ — ОДИН КАДР, А НЕ ДВА. Если фраза сталкивает две
   вещи («у одного за спиной десять лет, у другого две недели»; «сделать
   это со стоящим — и с тем, кто уже упал»), поставь обе рядом В ОДНОМ
   кадре: толстая стопка тетрадей и один тонкий листок; то же самое
   стоящим и лежащим. Показать половину пары — потерять всю мысль.
9. НАСТРОЕНИЕ. Первой строкой ответа оцени главу по двум осям:
     MOOD | тон от -2 (тяжёлый) до +2 (светлый) | напряжение от 0 (покой) до 3 (предел) | 2-4 слова
   Дальше держи кадры в этом настроении: тяжёлую главу не иллюстрируют
   солнечным кадром, главу-покой — рваным движением и толпой."""


def _clean(s):
    return " ".join((s or "").split())


def domain_contract():
    """Мир кадра ЭТОГО канала — из `channel_profile.json`, не из кода.

    Пусто — доменного правила в промпте нет вовсе. Это и есть условие
    работы «в любой нише»: канал про психологию не должен получать
    требование показывать рыцарей только потому, что репозиторий
    начинался как исторический.
    """
    # Мир канала выключается для эпизода из ЧУЖОЙ ниши. Нужно потому, что
    # он не только фильтрует, но и РУЛИТ: живой прогон психологического
    # сценария в этом репозитории (профиль объявляет Средневековье) выдал
    # «a knight in armor standing beside a closed chest» и «a monk's hand
    # touching a cracked mirror» — на текст про пустой файл в ноутбуке.
    # Модель послушалась объявленного мира, отклонять было нечего, и
    # предохранитель по доле отказов промолчал.
    if os.environ.get("SHOT_BRIEF_WORLD", "") == "off":
        return ""
    try:
        import pipeline_smart
        d = pipeline_smart.CHANNEL_PROFILE.get("shot_domain") or {}
    except Exception:
        return ""
    parts = []
    if d.get("world"):
        parts.append(f"МИР КАДРА: {d['world']}.")
    if d.get("people_in_frame"):
        parts.append(f"Человек в кадре — {d['people_in_frame']}.")
    if d.get("forbidden"):
        parts.append(f"В кадре не должно быть: {d['forbidden']}.")
    return " ".join(parts)


# Правила, ДОБАВЛЕННЫЕ поверх ядра по разбору и по литературе: заземление
# абстракции, настроение, «показывай новое», противопоставление, запрет
# символа. Измерены на Qwen3-30B-A3B и ОКАЗАЛИСЬ НЕ БЕСПЛАТНЫМИ: точность
# на сказанном та же (58.9% -> 59.1%), но ответов стало 88 вместо 107 —
# от длинного свода модель осторожничает и чаще ставит прочерк.
#
# Выключатель нужен, чтобы этот обмен был ВЫБОРОМ, а не побочным
# эффектом: у исторического канала с богатым предметным рядом дороже
# покрытие, у канала про психологию — заземление абстракции, без него
# кадра не будет вовсе.
EXTRA_RULES = os.environ.get("SHOT_BRIEF_EXTRA_RULES", "1") != "0"

# Номера правил ядра, которые добавлены поверх базовых шести.
_EXTRA_RULE_HEADS = ("6. АБСТРАКЦИЮ", "7. ПОКАЗЫВАЙ НОВОЕ",
                     "8. ПРОТИВОПОСТАВЛЕНИЕ", "9. НАСТРОЕНИЕ")


def core_rules():
    """Ядро правил; без EXTRA_RULES — только базовые шесть."""
    if EXTRA_RULES:
        return RULES_CORE
    out, skip = [], False
    for line in RULES_CORE.splitlines():
        head = line.strip()
        if any(head.startswith(h) for h in _EXTRA_RULE_HEADS):
            skip = True
        elif head[:2].rstrip(".").isdigit() and ". " in head[:5]:
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out)


def rules_for(contract):
    """Ядро правил плюс доменное правило, если канал его объявил."""
    base = core_rules()
    if not contract:
        return base
    return base + (
        "\n10. МИР КАДРА — обязателен. " + contract +
        " Общее слово вроде «a person» или «a man» приведёт случайного "
        "современного человека.")


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
    head = ["Ты — визуальный редактор видеоролика.",
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
        "\n".join(head), "", rules_for(domain_contract()), "",
        "ТИП КАДРА — одно из: object | scene | illustration | map | texture",
        "",
        "ФОРМАТ ОТВЕТА: сначала ОДНА строка MOOD, дальше ровно по одной "
        "строке на фразу, ничего лишнего.",
        "MOOD | тон | напряжение | 2-4 слова",
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
    banned = shot_planner_llm.channel_blocklist()
    if banned:
        lines.append("")
        lines.append("ЭТИХ СЛОВ В ОТВЕТЕ БЫТЬ НЕ ДОЛЖНО — канал их не берёт, "
                     "и заявка с ними будет отклонена целиком:")
        lines.append("  " + ", ".join(sorted(banned)))
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
# Настроение главы двумя осями. Валентность/возбуждение — стандартная
# непрерывная запись настроения в работах по эмоционально-осознанному
# подбору и редактированию изображений; прилагательные («мрачно»,
# «тревожно») сравнивать между главами нечем, а два числа — можно.
# Строка НЕОБЯЗАТЕЛЬНА: её нет — глава просто идёт без пометки настроения,
# ни один кадр из-за этого не теряется.
_MOOD_RE = re.compile(
    r"^\s*\**\s*MOOD\s*\|\s*([+-]?\d+(?:[.,]\d+)?)\s*\|"
    r"\s*([+-]?\d+(?:[.,]\d+)?)\s*\|\s*(.*)$", re.I)


def parse_mood(raw):
    """Тон и напряжение главы, если модель их назвала. Иначе None.

    Значения ЗАЖИМАЮТСЯ в объявленные промптом диапазоны, а не
    отбрасываются: модель, ответившая «тон -5», имела в виду «очень
    тяжело», и терять из-за этого всю главу было бы дороже, чем
    привести число к границе.
    """
    for line in (raw or "").splitlines():
        m = _MOOD_RE.match(line)
        if not m:
            continue
        try:
            tone = float(m.group(1).replace(",", "."))
            tension = float(m.group(2).replace(",", "."))
        except ValueError:
            return None
        return {"tone": max(-2.0, min(2.0, tone)),
                "tension": max(0.0, min(3.0, tension)),
                "words": _clean(m.group(3))[:60] or None}
    return None

# Слова из описаний кадра в FEWSHOT. Пример стоит в промпте НАРОЧНО на
# чужой теме (Рим, дороги): бриф из этого же эпизода подсказал бы модели
# готовые ответы ровно на тех фразах, на которых её потом меряют. Побочный
# эффект измерен: 1 заявка из ~80 у 7B копирует пример дословно — «a
# battlefield with a straight roman stone road in the background» и «a
# roman stone road in a hilly landscape, 1461 AD». Второе ушло бы в сток
# как есть и принесло римскую дорогу в эпизод про Войну Роз; ни один гейт
# этого не ловит — слова эпохи там формально нет, а `roman` не в блоклисте.
#
# Проверка НЕ про Рим, а про КОПИРОВАНИЕ ПРИМЕРА: она останется верной,
# если пример когда-нибудь заменят на другой. Заодно это довод в пользу
# заведомо чужой темы примера — утечка из неё ВИДНА, а утечка из
# средневекового примера выглядела бы правдоподобно и прошла бы мимо.
_FEWSHOT_SHOTS = [ln.split("|")[-1].strip().lower()
                  for ln in FEWSHOT.splitlines()
                  if "|" in ln and not ln.strip().upper().startswith("MOOD")]
# Строка-прочерк («3 | - | -») значимых слов не даёт вовсе, а набор короче
# порога совпасть с ним не может. Пустой набор в списке сделал бы гвард
# тихим no-op на этой строке и заодно ронял собственный тест — поймано
# тестом до коммита, а не рассуждением.
_FEWSHOT_WORDS = [w for w in (set(re.findall(r"[a-z]{4,}", shot))
                              for shot in _FEWSHOT_SHOTS) if len(w) >= 3]


def copies_the_example(shot_en, min_shared=3):
    """Заявка пересказывает пример из промпта, а не отвечает на фразу."""
    words = set(re.findall(r"[a-z]{4,}", (shot_en or "").lower()))
    return any(len(words & ex) >= min_shared for ex in _FEWSHOT_WORDS if ex)


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
        # РАЗДЕЛИТЕЛЬ ОБЯЗАТЕЛЕН. Без него нумерованной строкой оказывается
        # любой список в тексте модели — и это не теория: «думающая»
        # Qwen3.6-35B-A3B рассуждает вслух нумерованными пунктами, и в план
        # ушли семнадцать «заявок» вида «1. Analyze User Input» и
        # «2. Resolve pronouns to actual subjects» — пересказ моих же
        # правил. Ни один гейт их не ловил: латиница, длина в норме,
        # местоимений нет, снаряжение не современное.
        #
        # Промпт просит ровно формат «номер | тип | описание», и требовать
        # его — единственная проверка, которую нельзя обойти случайно.
        # Проверено на всех уже снятых замерах (7B, 7B+словарь, 30B,
        # Qwen3-4B, 30B+словарь): строк без разделителя там НОЛЬ, то есть
        # ужесточение не отменяет ни одного прежнего числа.
        if "|" not in rest:
            continue
        parts = [p.strip() for p in rest.split("|")]
        fn, shot = parts[0].lower(), "|".join(parts[1:]).strip()
        shot = _clean(shot.strip(" *`"))
        if not shot or shot in {"-", "—", "null", "none"}:
            continue
        if fn not in shot_planner_llm.VALID_FUNCTIONS:
            fn = None
        if not shot_planner_llm._LATIN_RE.search(shot):
            continue
        if not (2 <= len(shot.split()) <= 16):
            continue
        if copies_the_example(shot):
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

    # Потолок генерации на главу. 900 токенов с запасом покрывают 16 строк
    # заявок, НО модель с внутренним рассуждением может потратить их на
    # рассуждение и не дойти до ответа. Поэтому потолок управляем снаружи:
    # у «думающих» моделей его надо поднимать, и молча обрезанная глава —
    # худший исход (ответ есть, но его не видно).
    DEFAULT_MAX_TOKENS = int(os.environ.get("SHOT_BRIEF_MAX_TOKENS", "900") or 900)

    def __init__(self, model_path, n_threads=4, n_ctx=8192, seed=1,
                 max_tokens=None):
        from llama_cpp import Llama
        self.name = os.path.basename(model_path)
        self.seed = seed
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_threads=n_threads, seed=seed, verbose=False)

    # «Думающие» модели сначала рассуждают вслух и только потом отвечают.
    # Измерено на Qwen3.6-35B-A3B: одна глава — 390 секунд, и к пределу в
    # 2400 токенов модель дошла только до 11-й фразы из 16, ТАК И НЕ ВЫДАВ
    # ответа. Глава при этом не падает с ошибкой, она просто оказывается
    # пустой — то есть час работы даёт ноль, и снаружи это неотличимо от
    # «модель ничего не нашла».
    #
    # Повышать лимит бессмысленно: длина рассуждения растёт вместе с
    # главой. У семейства Qwen режим выключается служебной строкой в конце
    # запроса.
    NO_THINK = os.environ.get("SHOT_BRIEF_NO_THINK", "") == "1"
    # Служебная строка «/no_think» ПРОВЕРЕНА И НЕ СРАБОТАЛА: модель
    # рассуждала по-прежнему (замер на короткой главе). Работает другое —
    # затравка ответа УЖЕ ЗАКРЫТЫМ пустым блоком размышления: модели
    # нечего продолжать, и она сразу пишет ответ.
    THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

    def _ask_without_thinking(self, prompt):
        """Чат-разметка вручную, с закрытым блоком размышления в затравке."""
        text = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{self.THINK_OPEN}\n\n{self.THINK_CLOSE}\n\n")
        r = self.llm.create_completion(
            prompt=text, temperature=0.0, seed=self.seed,
            max_tokens=self.max_tokens, stop=["<|im_end|>"])
        return r["choices"][0]["text"] or ""

    def ask(self, prompt, chapter_no):
        if self.NO_THINK:
            return self._ask_without_thinking(prompt)
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
         "rejected": 0, "empty_chapters": 0,
         "critique_asked": 0, "critique_cache_hits": 0, "critique_rewrites": 0,
         "critique_shift_caught": 0}
REJECTED = []
# Настроение по главам: аудит-трейл, а не решение. Сегодня оно только
# пишется в план и печатается; кто им воспользуется (грейд, музыка, язык
# камеры) — отдельный вопрос, и заявлять эффект до замера нельзя.
MOODS = {}


def _cache_key_text(text, brain_name):
    """Общий ключ по СОДЕРЖИМОМУ запроса, а не по номеру версии — так
    его используют оба вызова модели за главу (черновик и самопроверка),
    и у них разный текст промпта, значит разный ключ без всякой ручной
    метки. Второй копии этой функции для критики заводить незачем."""
    import hashlib
    h = hashlib.md5()
    h.update(f"p{PACKET_VERSION}\x00{brain_name}\x00".encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_key(packet, brain_name):
    return _cache_key_text(render_prompt(packet), brain_name)


def render_critique_prompt(packet, draft_raw):
    """Второй проход: та же глава, СВОЙ уже написанный черновик, проверка
    двух КОНКРЕТНЫХ измеренных классов ошибок — не общее «сделай лучше»
    (расплывчатая инструкция моделям этого размера ничего не чинит, тот
    же урок, что уже записан про молчание модели: без позитивного сигнала
    трогать заявку вредно).

    Оба класса взяты не из головы, а из докстринга shot_brief_eval /
    CLAUDE.md, как задокументированные остаточные промахи режиссёра главы:
    1) местоимение/отсылка разрешена НЕ на реальный антецедент (юнит [13]
       эпизода 02 — «он» стал случайным мужчиной, хотя это земля, названная
       двумя фразами раньше);
    2) кадр — грамматически чистый, но не про тот предмет («A warrior...
       struck by a real battle sword» на «возьми настоящий боевой меч» —
       меч должен быть В РУКЕ, а не бить воина).

    SHOT_BRIEF_CRITIQUE=0/1, дефолт `0` — эффект НЕ измерен ни разу, это
    второй звонок модели ценой ~того же времени, что первый (~35 с на
    главу), и включать его без замера значило бы удвоить время эпизода
    вслепую.
    """
    lines = [
        "Ты только что описал кадры к этой же главе. Вот твой черновик "
        "построчно:", "", draft_raw.strip(), "",
        "Перепроверь ТОЛЬКО две вещи по каждой строке, глядя на фразы "
        "главы ниже:",
        "1) местоимение или отсылка («он», «это», «та сцена») разрешены "
        "на РЕАЛЬНЫЙ предмет ИЗ ЭТИХ ФРАЗ, а не угаданы;",
        "2) названный предмет — то, о чём физически говорит фраза, а не "
        "грамматически похожая, но другая по смыслу картинка.",
        "",
        "Строка верна — повтори её БЕЗ ИЗМЕНЕНИЙ. Строка неверна — "
        "перепиши ТОЛЬКО описание кадра, номер и тип оставь. Отвечай ТЕМ "
        "ЖЕ ФОРМАТОМ (номер | тип | описание по-английски), ровно по "
        "одной строке на фразу, без MOOD и без пояснений от себя.",
        "", "ФРАЗЫ ГЛАВЫ:",
    ]
    for u in packet["units"]:
        lines.append(f" {u['n']}. {u['text']}")
    return "\n".join(lines)


# Насколько далеко от своей позиции искать совпадение с чужой строкой
# черновика — сигнатуру сдвига (см. merge_critique). Замер 16.09 дал
# сдвиг РОВНО на одну позицию (юниты 5-8 главы «ДВЕСТИ МЕТРОВ» эпизода
# 02, живой прогон, не гипотеза): критика пропустила строку 5 и заново
# пронумеровала хвост, поэтому каждая следующая строка стала ответом на
# ЧУЖОЙ, соседний юнит, оставаясь при этом формально безупречной — ни
# один гейт формы (`brief_is_safe`) такое не ловит, ошибка позиционная,
# не текстовая. Окно 2 — с запасом на двойной сдвиг, не только на
# измеренный одинарный.
CRITIQUE_SHIFT_WINDOW = 2


def _looks_like_neighbor_shift(n, shot_en, draft_rows):
    """Заявка критики слово в слово совпадает с ЧУЖИМ соседним юнитом
    черновика — почти наверняка не исправление, а тот самый сдвиг
    нумерации, не совпадение по смыслу."""
    for k in range(-CRITIQUE_SHIFT_WINDOW, CRITIQUE_SHIFT_WINDOW + 1):
        if k == 0:
            continue
        neighbor = draft_rows.get(n + k)
        if neighbor and neighbor["shot_en"] == shot_en:
            return True
    return False


def merge_critique(draft_rows, critique_rows):
    """Критика может ИСПРАВИТЬ существующую заявку и не может добавить
    новую там, где черновик промолчал — молчание черновика уже прошло
    свою собственную проверку (brief_is_safe у ПЕРВОГО прохода), и вторая
    модель того же размера, отвечающая на юнит, которого не видела первой
    попыткой, не более надёжна, чем сам первый проход на нём. Additive
    только в одну сторону: заменить, никогда не породить с нуля.

    ВТОРОЙ ГВАРД — против сдвига нумерации, найденного живым прогоном
    16.09 (см. CRITIQUE_SHIFT_WINDOW): если предложенная замена слово в
    слово повторяет ЧУЖОЙ соседний юнит черновика, это не исправление,
    заменять нечем — оставляем черновик. `STATS['critique_shift_caught']`
    считает, сколько раз гвард сработал, а не молчит."""
    merged = dict(draft_rows)
    for n, row in critique_rows.items():
        if n not in merged:
            continue
        if _looks_like_neighbor_shift(n, row["shot_en"], draft_rows):
            STATS["critique_shift_caught"] += 1
            continue
        merged[n] = row
    return merged


def _ask_cached(brain, prompt_text, chapter_no, cache_dir, verbose, label,
                 stat_asked, stat_hit):
    """Один вызов мозга с диск-кэшем по содержимому промпта. Черновик и
    критика зовут эту же функцию с РАЗНЫМ текстом — ключи расходятся сами,
    второй копии кэш-логики заводить незачем (см. `_cache_key_text`)."""
    key = _cache_key_text(prompt_text, brain.name)
    path = os.path.join(cache_dir, key + ".txt") if cache_dir else None
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            STATS[stat_hit] += 1
            return raw
        except OSError:
            pass
    t0 = time.time()
    raw = brain.ask(prompt_text, chapter_no)
    STATS[stat_asked] += 1
    if verbose:
        print(f"  {label:<52}         {time.time() - t0:5.0f}с")
    if path and raw:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(raw)
            os.replace(tmp, path)
        except OSError:
            pass
    return raw


def run(video_dir, blocks, brain, cache_dir=None, verbose=True,
        max_units=None, only_sections=None, use_vocabulary=False):
    """Пройти эпизод главами. Возвращает {индекс блока: заявка}."""
    contract = domain_contract()
    if verbose:
        if contract:
            print(f"  МИР КАДРА ЭТОГО КАНАЛА: {contract}")
            print("  Если эпизод НЕ про этот мир — он будет диктовать кадры. "
                  "Выключить: SHOT_BRIEF_WORLD=off")
        else:
            print("  Мир кадра не объявлен — доменных правил нет "
                  "(channel_profile.json -> shot_domain)")
    critique_on = feature_flags.enabled("SHOT_BRIEF_CRITIQUE")
    out = {}
    for chapter_no, packet in enumerate(
            packets(video_dir, blocks, max_units=max_units,
                    use_vocabulary=use_vocabulary), 1):
        if only_sections and _section_key(packet["section"]) not in only_sections:
            continue
        STATS["chapters"] += 1
        label = _clean(packet["section"])[:52]
        raw = _ask_cached(brain, render_prompt(packet), chapter_no,
                          cache_dir, verbose, f"{label} ({len(packet['units'])} фраз)",
                          "asked", "cache_hits")
        rows = parse_answer(raw, packet)
        mood = parse_mood(raw)
        if mood:
            MOODS[_clean(packet["section"])] = mood
        # Самопроверка — ВТОРОЙ вызов ТОЙ ЖЕ модели по ЧЕРНОВИКУ, а не
        # эвристика: единственный класс промахов, который она способна
        # закрыть (неверно разрешённая отсылка, грамматически чистый, но
        # не тот предмет), не ловится правилами формы (`brief_is_safe`).
        # Выключено по умолчанию — эффект не измерен ни разу, см.
        # докстринг render_critique_prompt.
        if critique_on and rows:
            crit_raw = _ask_cached(
                brain, render_critique_prompt(packet, raw), chapter_no,
                cache_dir, verbose, f"{label} (критика)",
                "critique_asked", "critique_cache_hits")
            crit_rows = parse_answer(crit_raw, packet)
            merged = merge_critique(rows, crit_rows)
            STATS["critique_rewrites"] += sum(
                1 for n, r in merged.items()
                if n in rows and r["shot_en"] != rows[n]["shot_en"])
            rows = merged
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
    _warn_if_world_rule_eats_everything()
    return out


# Доля отклонений правилом мира канала, после которой это перестаёт быть
# нормой и становится подозрением на чужой профиль.
FOREIGN_PROFILE_SHARE = 0.25


def _warn_if_world_rule_eats_everything():
    """Сказать ГРОМКО, если мир канала отклоняет слишком много заявок.

    Измерено живым прогоном: психологический сценарий, прогнанный в
    репозитории с объявленным СРЕДНЕВЕКОВЫМ миром, потерял четыре
    заявки из двенадцати — и все четыре были правильными кадрами («a
    person slumped over a desk, head in hands», «a human hand recoiling
    sharply from a hot stove burner»). Отказ при этом молчит: заявки
    просто не появляются, и снаружи это неотличимо от «модель не
    справилась».

    ЧАСТЬ 24 CLAUDE.md предупреждает не копировать чужой
    channel_profile.json, но предупреждение в документе не срабатывает в
    момент ошибки. Эта строка — срабатывает.
    """
    world = sum(1 for r in REJECTED if "мир" in (r.get("reason") or ""))
    total = STATS["rows"] + STATS["rejected"]
    if total and world / total >= FOREIGN_PROFILE_SHARE:
        print(f"  ВНИМАНИЕ: правило мира канала отклонило {world} заявок из "
              f"{total}. Если канал не про этот мир — в channel_profile.json "
              f"остался shot_domain от другого канала (ЧАСТЬ 24). Убрать "
              f"ключ или заменить своим.")


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
                   "moods": MOODS, "rejected": REJECTED, "units": units},
                  f, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(mp, PLAN_NAME))
    return os.path.join(mp, PLAN_NAME)


# Минимальная длина куска фразы, по которому ищется место в файле.
# Короче — и совпадение перестаёт быть однозначным («И ты не встаёшь.»
# короткое, но встречается один раз; «Смотри.» встретится где угодно).
ANCHOR_MIN_CHARS = 28


def _unique_anchor(body, text):
    """Где в сыром файле начинается эта фраза. None — если непонятно.

    Полный текст юнита искать нельзя: парсер СКЛЕИВАЕТ куски вокруг
    внутренних тегов, и фраза с `[stat:ГЕНРИХ V, 1422]` внутри в файле
    буквально не встречается ни разу. Живой прогон дал ровно это: три
    юнита из 107 пропущены с «встречается 0 раз».

    Поэтому ищется самый ДЛИННЫЙ префикс фразы, который встречается в
    файле ровно один раз, и не короче ANCHOR_MIN_CHARS. Длинный префикс
    предпочтительнее короткого: чем он длиннее, тем меньше шанс, что
    однозначность случайна.
    """
    text = (text or "").strip()
    if len(text) < ANCHOR_MIN_CHARS:
        return body.index(text) if body.count(text) == 1 else None
    for cut in range(len(text), ANCHOR_MIN_CHARS - 1, -1):
        piece = text[:cut]
        if body.count(piece) == 1:
            return body.index(piece)
    return None


def write_inline(video_dir, blocks, found, dry_run=False):
    """Проставить `[shot:...]` прямо в script.txt перед своей фразой.

    ЗАЧЕМ ИМЕННО ТУДА. CLAUDE.md объясняет выбор инлайнового тега: бриф,
    лежащий отдельной секцией, пришлось бы ключевать по НОМЕРУ юнита, а
    номера сдвигаются от любой правки текста выше по сценарию — и бриф
    молча описывал бы чужую фразу. Инлайн ездит ВМЕСТЕ со своей фразой и
    разъехаться физически не может. Плюс `script_parser` читает `[shot:]`
    штатно, то есть после этой записи план и флаг больше не нужны вовсе.

    ОСТОРОЖНО И НАМЕРЕННО УЗКО. Правится ЧУЖОЙ исходник — сценарий,
    который человек писал руками. Поэтому:
      * тег ставится только там, где текст фразы встречается в файле
        РОВНО ОДИН раз (иначе непонятно, к какой из копий);
      * фраза, у которой уже есть `[shot:`, не трогается никогда —
        бриф автора сильнее заявки модели, это правило всего модуля;
      * перед записью делается `.bak` (тот же урок, что с копией .env:
        без копии откат невозможен);
      * что не удалось проставить — печатается поимённо, а не молчит.
    """
    path = os.path.join(video_dir, "script.txt")
    with open(path, encoding="utf-8") as f:
        body = f.read()

    placed, skipped = 0, []
    for idx in sorted(found):
        text = _clean(blocks[idx].get("text"))
        brief = found[idx]["shot_en"]
        if (blocks[idx].get("shot_brief") or "").strip():
            skipped.append((text, "у автора уже есть бриф"))
            continue
        at = _unique_anchor(body, text)
        if at is None:
            skipped.append((text, "нет однозначного места в файле"))
            continue
        # Второй проверки «тег уже стоит» не заводится: её уже сделал
        # парсер — если тег есть, он лежит в blocks[idx]["shot_brief"], и
        # юнит отсеян строкой выше. Собственная эвристика по тексту файла
        # была бы вторым ответом на тот же вопрос и рано или поздно
        # разошлась бы с первым.
        body = body[:at] + f"[shot:{brief}]" + body[at:]
        placed += 1

    if not dry_run and placed:
        with open(path + ".bak", "w", encoding="utf-8") as f:
            f.write(open(path, encoding="utf-8").read())
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, path)
    return placed, skipped


# Куда setup_local_director.py кладёт модель и что он туда кладёт. Имена
# продублированы СОЗНАТЕЛЬНО не копией словаря: импортировать оттуда
# значило бы тянуть в рендер argparse-скрипт установки. Порядок — это
# ПРЕДПОЧТЕНИЕ по замеру (69 против 63 на срезе 116 юнитов), а не список
# разрешённых: любой другой .gguf в папке тоже берётся, если этих нет.
MODELS_DIR_NAME = "models"
PREFERRED_MODELS = (
    "Qwen3-30B-A3B-Instruct-2507-Q3_K_S.gguf",
    "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
)


def find_model(explicit=None):
    """Файл модели без единого флага в обычном случае.

    Порядок: явный --model -> LLAMA_MODEL_GGUF -> models/ рядом с репо.
    Внутри папки сначала замеренные предпочтения, потом ЛЮБОЙ .gguf в
    алфавитном порядке — детерминированно, а не «первый попавшийся от
    файловой системы»: два прогона на одной машине обязаны взять один
    файл, иначе план молча собран другим мозгом.

    Возвращает путь или None. Решение «что делать без модели» принимает
    вызывающий: у замера и у рендера оно разное.
    """
    for cand in (explicit, os.environ.get("LLAMA_MODEL_GGUF")):
        if cand and os.path.exists(cand):
            return cand
    d = os.path.join(REPO, MODELS_DIR_NAME)
    if not os.path.isdir(d):
        return None
    have = {f for f in os.listdir(d) if f.lower().endswith(".gguf")}
    for name in PREFERRED_MODELS:
        if name in have:
            return os.path.join(d, name)
    rest = sorted(have)
    return os.path.join(d, rest[0]) if rest else None



def main(argv):
    ap = argparse.ArgumentParser(
        description="Режиссёрская разработка главы: контекст вместо фразы")
    ap.add_argument("video_dir")
    ap.add_argument("--brain", choices=("local", "file", "packets"),
                    default="local",
                    help="по умолчанию local: модель ищется сама "
                         "(--model, LLAMA_MODEL_GGUF, models/*.gguf)")
    ap.add_argument("--answers", help="папка с ответами для --brain file")
    ap.add_argument("--model", default=None,
                    help="файл .gguf; по умолчанию ищется сам")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-packets", help="куда выложить промпты глав")
    ap.add_argument("--no-cache", action="store_true")
    # Запись включена ПО УМОЛЧАНИЮ: план на диске, который никто не
    # применил, — это ровно тот класс «слой есть, и его никто не зовёт»,
    # которым репозиторий горел шесть раз. Безопасность даёт не флаг, а
    # устройство: .bak перед записью, бриф автора не трогается никогда,
    # заявка, не прошедшая brief_is_safe(), не пишется вовсе.
    ap.add_argument("--no-inline", dest="write_inline", action="store_false",
                    help="только план, НЕ трогать script.txt")
    ap.add_argument("--write-inline", dest="write_inline",
                    action="store_true",
                    help=argparse.SUPPRESS)   # совместимость со старой строкой
    ap.set_defaults(write_inline=True)
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
        model = find_model(a.model)
        if not model:
            print("Модели нет. Поставить одной командой:\n"
                  "    python scripts/setup_local_director.py\n"
                  "Она сама выберет размер под память машины и положит файл "
                  f"в {MODELS_DIR_NAME}/. После этого команда выше работает "
                  "без единого флага.")
            return 2
        print(f"Мозг: {os.path.basename(model)}")
        brain = LocalBrain(model, n_threads=a.threads)
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
    if a.write_inline:
        placed, skipped = write_inline(a.video_dir, blocks, found)
        print(f"В script.txt проставлено [shot:] — {placed}; "
              f"не проставлено {len(skipped)}")
        for text, why in skipped[:20]:
            print(f"  [{why}] {text[:70]}")
    print(f"\nГлав {STATS['chapters']}, вопросов {STATS['asked']}, "
          f"из кэша {STATS['cache_hits']}, заявок {STATS['rows']}, "
          f"отклонено проверкой {STATS['rejected']}, "
          f"глав без единого разбора {STATS['empty_chapters']}")
    if feature_flags.enabled("SHOT_BRIEF_CRITIQUE"):
        print(f"Критика: {STATS['critique_asked']} вопросов, "
              f"{STATS['critique_cache_hits']} из кэша, "
              f"переписано строк {STATS['critique_rewrites']}, "
              f"отловлено сдвигов {STATS['critique_shift_caught']}")
    print(f"План: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
