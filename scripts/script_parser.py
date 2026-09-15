"""Разбор script.txt (=== HOOK/BLOCK*/FINAL === секции, теги [pause]/
[short pause]/[stat:...]/[climax] и т.д.) в список озвучиваемых блоков —
вынесено ИЗ pipeline_smart.py в отдельный лёгкий модуль БЕЗ побочных
эффектов импорта: только `import re`, ничего не читает из sys.argv, не
трогает диск на импорте.

pipeline_smart.py на импорте читает VIDEO_FOLDER из sys.argv[1] и запускает
find_audio() по всей папке (см. его докстринг/find_audio) — реальный,
пойманный вживую баг: три модуля (section_sync.py, render_episode.py,
pause_intelligence.py), которым нужна была ТОЛЬКО parse_blocks(), были
вынуждены временно подменять sys.argv, импортировать ВЕСЬ 5900+-строчный
pipeline_smart.py (плюс его тяжёлые torch/cv2-зависимости) и восстанавливать
sys.argv обратно — хрупкий обходной приём, повторённый в трёх местах.
section_sync.py и render_episode.py теперь импортируют parse_blocks отсюда
напрямую, без подмены sys.argv и без импорта pipeline_smart.py целиком.

pipeline_smart.py импортирует PAUSE_DURATIONS/parse_blocks ОТСЮДА и
реэкспортирует их под теми же именами (`pipeline_smart.parse_blocks` и
`pipeline_smart.PAUSE_DURATIONS` продолжают работать как раньше — ничего не
сломано у существующих вызовов/тестов, которые патчат их по имени модуля
pipeline_smart)."""
import re

PAUSE_DURATIONS = {"[pause]": 0.8, "[short pause]": 0.4,
                   "[slowly]": 0.0, "[emphasis]": 0.0, "[energetic]": 0.0}

# ПАЙПЛАЙН-ONLY МАРКЕРЫ — теги, которые TTS не должен увидеть НИКОГДА.
# Отличаются от PAUSE_DURATIONS принципиально: те — теги движка озвучки
# (ЧАСТЬ 10 CLAUDE.md), они уезжают в TTS буква-в-букву и там что-то значат.
# Эти — записка автора сценария самому пайплайну (плашка на экран, сигнал
# музыке, звук, защищённая тишина), и в тексте заказа им делать нечего.
#
# Список объявлен здесь, потому что именно здесь он и растёт: [stat:] и
# [climax] были с самого начала, [sfx:] и [hush] добавлены позже (см. их
# разбор в parse_blocks ниже). Реальная цена второй копии — замер 14.09:
# lumean_tts.py держал свои два регекспа и не узнал про [sfx:], поэтому
# литеральный "[sfx:armour_clank]" уехал в заказ. Вслух его не прочитали
# (0.0115 с/символ против 0.049 у речи — замерено по alignment), но в
# посимвольный alignment он попал, сходство текста блока с озвученным упало
# 1.000 -> 0.889 при пороге 0.9, и PHRASE LOCK выключился на ВЕСЬ эпизод:
# кадры поехали по оценочным длительностям вместо реальных онсетов речи.
PIPELINE_ONLY_TAG_RE = re.compile(r'\[(?:stat:[^\]]*|climax|sfx:[^\]]*|hush)\]',
                                  re.IGNORECASE)


def strip_pipeline_only_tags(text, repl=" "):
    """Убрать из текста маркеры, которые пишутся для пайплайна, а не для TTS.

    Единственное место, где этот словарь применяется к тексту заказа —
    звать отсюда, не копировать регекспы к себе."""
    return PIPELINE_ONLY_TAG_RE.sub(repl, text or "")


# Любой [...] в потоке символов alignment — РАЗМЕТКА, а не речь.
#
# Живёт здесь, рядом с PIPELINE_ONLY_TAG_RE, по той же причине: это словарь
# тегов, и его копии уже стоили эпизоду PHRASE LOCK. Потребители —
# pipeline_smart (сравнение блока с озвученным, слова хук-подписи),
# section_sync (поиск натуральных разрывов) и lumean_tts (граница секций).
ALIGNMENT_TAG_SPAN_RE = re.compile(r'\[[^\]]*\]')


def speech_bounds_from_alignment(chars):
    """(старт первой речи, конец последней) по посимвольному alignment.

    chars — [(символ, старт, конец), ...]. Теги и пробелы речью НЕ считаются:
    хвостовой [pause] формально занимает время, но это тишина, а не голос.
    None, если реальных символов нет вообще."""
    text = "".join(c for c, _s, _e in chars)
    excluded = set()
    for m in ALIGNMENT_TAG_SPAN_RE.finditer(text):
        excluded.update(range(m.start(), m.end()))
    real = [(s, e) for j, (c, s, e) in enumerate(chars)
            if j not in excluded and (c or "").strip()]
    if not real:
        return None
    return real[0][0], real[-1][1]


def parse_blocks(path):
    raw = open(path, encoding="utf-8").read()
    # Берём ТОЛЬКО озвучиваемые секции (HOOK / BLOCK* / FINAL). Всё служебное —
    # METADATA, PEXELS QUERIES, IMAGE PROMPTS, COMPETITOR ANALYSIS, TITLE/THUMBNAIL
    # OPTIONS — отбрасывается по имени секции, а не по списку известных заголовков:
    # раньше METADATA просачивалась в озвучку и съедала первый кадр.
    parts = re.split(r'===\s*(.*?)\s*===', raw)
    kept = []
    for i in range(1, len(parts), 2):
        name = parts[i].upper()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if name.startswith(("HOOK", "BLOCK", "FINAL")):
            kept.append((name, body))
    if kept:
        # Маркер секции — ещё один спецтокен в общем потоке (не просто "\n"-склейка
        # как раньше), чтобы граница === не рвала перенос паузы через границу блока,
        # а секция при этом была известна для каждого получившегося блока.
        # БЕЗ .strip("\x00") — иначе снимается ведущий \x00 у маркера самой первой
        # секции (обычно HOOK), split() перестаёт его узнавать, и "SECTION:HOOK"
        # утекает в текст как отдельный псевдо-блок из одного слова — под первый
        # кадр хука уходит мусорный клип с generic-запросом вместо реального текста.
        content = "".join(f"\x00SECTION:{name}\x00{body}" for name, body in kept)
    else:
        content = re.sub(r'===.*?===', '', raw).strip()   # сценарий без === — берём как есть
    # [stat:TEXT] — цифра-плашка на экран (ЧАСТЬ "Монтаж под удержание": цифра
    # без плашки не запоминается). Вытаскиваем ДО общего вырезания [...],
    # иначе текст плашки пропадает вместе со всеми остальными тегами.
    content = re.sub(r'\[stat:(.*?)\]', lambda m: f"\x01STAT:{m.group(1)}\x01", content)
    # [climax] — 2.8 (второй продакшн-документ, "тишина как акцент перед
    # разоблачением"): ставится сценаристом ЯВНО перед фразой-разоблачением
    # (не угадывается автоматически — риск ложного срабатывания на обычном
    # блоке того же типа, что уже отвели для CV-принудительной вариативности
    # и акцентных слов при выборе, что НЕ делать). Пипелайн-only маркер,
    # как [stat:...] — не входит в текст, который читает TTS (тот же
    # принцип: пользователь копирует ЧИСТЫЙ текст в ElevenLabs, [stat:...]
    # туда тоже никогда не попадал).
    content = content.replace("[climax]", "\x02CLIMAX\x02")
    # [sfx:концепт] и [hush] — намерение по ЗВУКУ, которое ставит автор
    # сценария в тот момент, когда он этот костёр в текст и вписывает.
    #
    # Почему так, а не разбором готового текста: сценарий не найден в
    # природе, он пишется — значит момент, когда система знает про костёр,
    # наступает на шаг РАНЬШЕ любого анализа. Выводить звук из написанного
    # текста значит угадывать то, что было известно точно. Прямая проверка
    # обеих автоматических схем это подтвердила: русский текст главы в
    # текстовую башню CLAP дал ОДИН И ТОТ ЖЕ вид на всех шести главах, а
    # мост «многоязычный эмбеддинг -> английская метка понятия» — 7 верных
    # из 18 на одиночных словах (случайный выбор дал бы 1 из 20).
    #
    # [hush] — не «здесь нет звука», а «здесь тишина НУЖНА». Разница
    # смысловая: пустое место планировщик имеет право заполнить, помеченную
    # тишину — нет. Тишина в звуковом монтаже такой же элемент плана, как
    # удар, и защищать её надо явно.
    #
    # Оба — пайплайн-only маркеры, как [stat:...] и [climax]: в текст для
    # TTS не попадают (см. ЧАСТЬ 10 CLAUDE.md).
    content = re.sub(r'\[sfx:(.*?)\]', lambda m: f"\x03SFX:{m.group(1)}\x03", content)
    content = content.replace("[hush]", "\x04HUSH\x04")
    processed = content
    for tag in sorted(PAUSE_DURATIONS, key=len, reverse=True):
        processed = processed.replace(tag, f"__PAUSE_{PAUSE_DURATIONS[tag]}__")
    # Всё, что дошло сюда как [...] — НЕ входит в PAUSE_DURATIONS (известные
    # теги уже заменены строчкой выше). Раньше re.sub ниже молча вырезал такие
    # теги без единого предупреждения и БЕЗ вставки паузы — если сценарист по
    # аналогии с [pause]/[short pause] напишет запрещённый [long pause]
    # (ЧАСТЬ 10 CLAUDE.md — ломает TTS-артефактами) или любой другой
    # незнакомый тег, граница блока/пауза в этом месте тихо пропадала.
    unknown_tags = sorted(set(re.findall(r'\[.*?\]', processed)))
    if unknown_tags:
        print(f"  ВНИМАНИЕ: неизвестные теги в script.txt — вырезаны БЕЗ вставки "
              f"паузы: {', '.join(unknown_tags)}. Разрешены только: "
              f"{', '.join(sorted(PAUSE_DURATIONS))} (ЧАСТЬ 10 CLAUDE.md). "
              f"[long pause] запрещён явно (ломает TTS-артефактами) — проверь script.txt.")
    processed = re.sub(r'\[.*?\]', '', processed)
    parts = re.split(r'(__PAUSE_[\d.]+__|\x00SECTION:.*?\x00|\x01STAT:.*?\x01|\x02CLIMAX\x02'
                     r'|\x03SFX:.*?\x03|\x04HUSH\x04)', processed)
    blocks, cur, pause, stat, stat_word_pos, pending_climax = [], "", 0.0, None, None, False
    sfx, hush = [], False
    # [sfx:...] стоит ВНУТРИ фразы и монтаж не режет. Но он разбивает строку
    # на части, и если к этому моменту висит несъеденная пауза (она осталась
    # от [pause] перед блоком), следующий же огрызок — хоть одна точка —
    # уходил в отдельный блок со своим слотом под картинку. Флаг говорит
    # разборщику: этот кусок — продолжение той же фразы, не новый блок.
    merge_next = False
    section = "BODY"

    def flush():
        nonlocal cur, pause, stat, stat_word_pos, pending_climax, sfx, hush
        if cur:
            blocks.append({"text": cur, "pause_after": pause,
                           "words": len(cur.split()), "section": section, "stat": stat,
                           "stat_word_pos": stat_word_pos, "is_climax": pending_climax,
                           "sfx": list(sfx), "hush": hush})
        cur, pause, stat, stat_word_pos, pending_climax = "", 0.0, None, None, False
        sfx, hush = [], False

    for part in parts:
        mp = re.match(r'__PAUSE_([\d.]+)__', part)
        ms = re.match(r'\x00SECTION:(.*?)\x00', part)
        mst = re.match(r'\x01STAT:(.*?)\x01', part)
        mc = part == "\x02CLIMAX\x02"
        if mp:
            pause += float(mp.group(1))
        elif ms:
            flush()             # смена секции — всегда граница блока
            section = ms.group(1)
        elif mc:
            # Клаймакс относится к СЛЕДУЮЩЕМУ (ещё не начатому) блоку — тот
            # же принцип, что у [stat:...] ниже: если что-то уже накоплено
            # в cur, это ЧУЖОЙ, предыдущий блок, флашим его, не помечаем.
            if cur:
                flush()
            pending_climax = True
        elif mst:
            # Плашка НЕ режет монтаж — [stat:...] может стоять посреди фразы
            # без соседнего [pause], и это не повод обрывать клип на ровном
            # месте. Но если пауза уже открыта (граница блока ждёт следующий
            # текст), плашка относится к ЕЩЁ НЕ начатому следующему блоку —
            # без явного flush() она утекала бы в текущий cur (баг: плашка
            # после [pause] подписывалась под предыдущий, а не следующий кадр).
            if pause > 0 and cur:
                flush()
            stat = mst.group(1)
            # Тег стоит ПОСЛЕ фразы с числом ("...до полутора.[stat:1,0-1,5 КГ]
            # Среднее...") — то есть к моменту тега число уже произнесено.
            # Запоминаем, сколько слов УЖЕ накоплено в cur — это точка, к
            # которой плашка должна появиться, а не начало клипа (раньше
            # позиция тега нигде не сохранялась, плашка стартовала от t=0
            # независимо от того, где в фразе реально звучит цифра — цифра
            # на экране опережала озвучку на полклипа и больше)."""
            stat_word_pos = len(cur.split())
        elif part.startswith("\x03SFX:"):
            # Позиция в СЛОВАХ, как у [stat:...] — по ней потом берётся
            # реальное время из посимвольного alignment. Сам звук ставится
            # НЕ на это слово, а с опережением (pre-lap): звук ровно на
            # слове — буквальная иллюстрация речи, узнаваемый признак
            # любителя; профессионально сначала слышишь, потом понимаешь.
            # Здесь хранится только якорь, решение о времени — у планировщика.
            # Пауза уже открыта — значит тег стоит В НАЧАЛЕ следующей фразы,
            # и граница блока тут НАСТОЯЩАЯ. Ровно тот же случай и то же
            # лекарство, что у [stat:...] двумя ветками выше.
            #
            # Почему это не косметика (замерено 14.09 на videos/_test60s):
            # без flush() тег в начале фразы отдавал merge_next=True, текст
            # следующей фразы приклеивался к ПРЕДЫДУЩЕМУ блоку (три блока
            # превращались в два, монтажный рез исчезал), а сам звук получал
            # word_pos по длине ЧУЖОЙ фразы. То есть позиция, которую
            # CLAUDE.md прямо предписывает автору («тег работает, когда стоит
            # В НАЧАЛЕ фразы»), ломала разбор. Вторая позиция — середина
            # фразы — разбор не ломает, но там нет тишины, и планировщик
            # отбрасывает кюй с причиной no_silence_for_object. Рабочей
            # позиции не существовало вообще: на реальном эпизоде звук
            # доспеха не прозвучал ни разу из двух.
            if pause > 0 and cur:
                flush()
            name = part[len("\x03SFX:"):-1].strip()
            if name:
                sfx.append({"name": name, "word_pos": len(cur.split())})
            merge_next = bool(cur)
        elif part == "\x04HUSH\x04":
            hush = True
        else:
            t = part.strip()
            if t:
                if merge_next:
                    cur = f"{cur} {t}".strip()
                    merge_next = False
                elif pause > 0 and cur:
                    flush()      # реальная пауза — вот это настоящая граница блока
                    cur = t
                else:
                    cur = f"{cur} {t}".strip()
    flush()
    print(f"Блоков: {len(blocks)}")
    return blocks


def _normalize_section_key(name):
    """"BLOCK 1: Постановка проблемы" (b["section"], полный заголовок) и
    "BLOCK_1" (строка в === PEXELS QUERIES ===, см. parse_pexels_queries) —
    два РАЗНЫХ написания одной и той же секции (пробел+заголовок против
    подчёркивания). Нормализует оба к общему ключу ("BLOCK1"/"HOOK"/
    "FINAL") — без этого сопоставление по имени секции никогда бы не
    совпало. None, если строка не начинается с известного имени секции."""
    m = re.match(r'\s*(HOOK|FINAL|BLOCK[\s_]*\d+)', name.strip().upper())
    if not m:
        return None
    return re.sub(r'[\s_]+', '', m.group(1))


def parse_pexels_queries(path):
    """Разбор === PEXELS QUERIES === (см. CLAUDE.md ЧАСТЬ 9/13, Шаг 3) —
    {normalized_section_key: [query, ...]}. Это ЗАПРОСЫ, НАПИСАННЫЕ ВРУЧНУЮ
    (человеком/LLM) в процессе написания сценария, С ПОЛНЫМ КОНТЕКСТОМ
    момента ("вспомни, сколько весит пакет молока у тебя в руке" ->
    "milk bottle hand") — то, что чисто алгоритмический query_for()/THEMES
    (см. pipeline_smart.py) физически не может воспроизвести: тот сопостав-
    ляет только отдельные существительные-корни из статического словаря,
    без понимания сцены целиком. Секция реально существует в script.txt
    production-эпизода (videos/01_ves-mecha) — просто ДО этого коммита ни
    разу не читалась пайплайном, несмотря на то что протокол явно требует
    её писать (реальный, найденный по прямому запросу пользователя пробел:
    "написанное для этого назначения не используется").

    Нет файла / нет секции / секция пуста -> {} (честный откат — вызывающий
    код в pipeline_smart.resolve_queries() просто не находит authored-
    запрос для секции и работает как раньше, query_for()/THEMES)."""
    return {key: [_strip_shot_type(q) for q in queries]
            for key, queries in (_parse_query_section(path) or {}).items()}


def _parse_query_section(path):
    """СЫРЫЕ строки запросов секции — вместе с пометкой типа кадра.

    Один разбор на двоих: parse_pexels_queries() снимает пометку (в API
    должен уйти чистый запрос), parse_query_shot_types() её читает. Пока
    это были два независимых прохода, второй получал уже очищенный текст
    и не видел ни одной пометки — поймано тестом, а не на проде."""
    try:
        raw = open(path, encoding="utf-8").read()
    except Exception:
        return {}
    parts = re.split(r'===\s*(.*?)\s*===', raw)
    body = None
    for i in range(1, len(parts), 2):
        if parts[i].strip().upper().startswith("PEXELS QUERIES"):
            body = parts[i + 1] if i + 1 < len(parts) else ""
            break
    if not body:
        return {}
    result = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        label, _, rest = line.partition(":")
        key = _normalize_section_key(label)
        if not key:
            continue
        queries = [q.strip() for q in rest.split(",") if q.strip()]
        if queries:
            result[key] = queries
    return result


def _strip_shot_type(raw):
    """Снять пометку типа кадра `[object]` с запроса. Модуль опционален —
    без него запрос возвращается как есть."""
    try:
        import shot_types
        return shot_types.parse_query_spec(raw)[0]
    except Exception:
        return raw


def parse_query_shot_types(path):
    """{текст запроса: тип кадра} из === PEXELS QUERIES ===.

    Автор помечает тип прямо в строке запроса:

        HOOK: medieval helmet visor slit [object], medieval camp tent [scene]

    Тип решает, В КАКОЙ ИСТОЧНИК уйдёт запрос и в каком виде (см.
    scripts/shot_types.py): музей — каталог предметов с паспортом, и
    сценический запрос туда отправлять бессмысленно, а предметный нужно
    отправлять структурно, по отделу коллекции. Разметки нет — тип
    выводится из слов запроса, а без сигнала остаётся `any`, то есть
    прежний маршрут во все источники.

    Ключ — ОЧИЩЕННЫЙ текст запроса (без скобки), тот же, что вернёт
    parse_pexels_queries() и что реально уйдёт в API."""
    try:
        import shot_types
    except Exception:
        return {}
    out = {}
    for queries in (_parse_query_section(path) or {}).values():
        for raw in queries:
            text, shot_type = shot_types.parse_query_spec(raw)
            if shot_type:
                out[text] = shot_type
    return out
