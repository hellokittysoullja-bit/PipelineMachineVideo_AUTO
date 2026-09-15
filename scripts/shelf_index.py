# -*- coding: utf-8 -*-
"""Визуальная полка: слот спрашивает картинки, а не слова.

ЗАЧЕМ. Весь вход подбора до сих пор был ТЕКСТОВЫМ ЗАПРОСОМ. Автор писал
короткую английскую строку, она уходила в поиск музея/архива/стока, и
дальше пайплайн выбирал лучшее из того, что ответил ПОИСК ПО СЛОВАМ. Это
ломается тремя независимыми способами, и все три измерены на реальных
данных этого канала, а не предположены:

1. СЛОВА МУЗЕЯ И СЛОВА АВТОРА — РАЗНЫЕ. `poleaxe` в словаре Мет не
   существует вообще; нужный предмет лежит под именем `Halberd` (96 штук)
   и `Partisan` (35). `longsword` — тоже ноль: у Мет это `Sword`,
   `Two-hand sword`. Автор пишет вслепую, и промах не виден ниоткуда.

2. ПОБЕЖДАЕТ СЛУЧАЙНОЕ СЛОВО, А НЕ ПРЕДМЕТ. `met_catalog.search()` считает
   совпадения по полям (Object Name +10, Classification +4, Tags +2,
   Title +1). Прямой прогон 14.09 по запросам эпизода 02:
     `medieval castle moat water`   -> Holy-water font, Water jar  (слово «water»)
     `european longsword blade macro` -> Blade, Axe blade          (слово «blade»)
     `medieval knight armour fallen mud` -> Plaque, Plaque, Plaque
   Ни один гейт ниже по конвейеру этого не лечит: они проверяют «хорош ли
   кандидат для СВОЕГО запроса», а в пул с самого начала попало не то.

3. ОПИСАНИЕ КАДРА В ПОИСК ПО СЛОВАМ НЕ ВЛЕЗАЕТ В ПРИНЦИПЕ. У архивов
   И-логика: замер на живом Europeana (окно 1000-1600, только картинки,
   свободные лицензии) — пятисловные запросы эпизода дают РОВНО НОЛЬ на
   всех девяти проверенных, а односложные — сотни (`battle` 187,
   `soldiers` 102, `siege` 36, и в выдаче настоящие датированные
   миниатюры KB Нидерландов 1332-1500 годов). То есть чем точнее описан
   кадр, тем гарантированнее пустой ответ. Тот же эффект давно записан
   про Openverse и музейный поиск (`_openverse_query_cascade`).

ЧТО ДЕЛАЕТ ЭТОТ МОДУЛЬ. Один раз оффлайн считает эмбеддинг КАЖДОЙ картинки
полки (предметы Мет, уже прошедшие паспорт эпохи и культуры в
`met_catalog.build`) и кладёт рядом с их метаданными. Дальше слот
спрашивает полку ОПИСАНИЕМ КАДРА обычным языком, и полка отвечает
сравнением этого описания с самими изображениями — той же моделью
SigLIP2, на которой уже построен `sentence_relevance()` и которая на
собственном 113-позиционном бенчмарке этого репозитория даёт top-1 = 90%.

ПОЧЕМУ ИМЕННО КАРТИНКИ, А НЕ КАРТОЧКИ (проверено, гипотеза отклонена).
Дешёвый вариант — индексировать ТЕКСТ карточки музея («Halberd, Shafted
Weapons, German, Steel, 1598») — стоит 80 минут вместо 20 часов и был
построен первым. Прямое сравнение на ОДНИХ И ТЕХ ЖЕ 70 предметах, где
рядом намеренно положены ловушки, на которых ошибается словарный поиск:

  бриф                                      по карточкам      по картинкам
  «древковое оружие с топором на древке»    Наручи            Halberd/Halberd/Halberd
  «полный латный доспех в рост»             Наручи            Field armor/Armor
  «рукописная миниатюра с битвой»           Евангелиарий      Beatus, лист рукописи
  «двуручный меч, весь клинок»              Плакетка навершия Early sword/Practice Sword
  «шлем с узкой смотровой щелью»            Забрало           Helmet/Helmet/Close-helmet
  «узкий клинок рондельного кинжала»        Рапира с дагой    Dagger

Шесть брифов из шести — за картинками. Причина не в удаче: текстовая
башня SigLIP2 обучена на пару текст-КАРТИНКА, и её текст-текст сходство
вырождается (тот же диагноз, что этот репозиторий уже записал про CLAP:
«все шесть глав выбрали один и тот же вид»). Индекс по карточкам оставлен
в истории как ОТРИЦАТЕЛЬНЫЙ результат, а не как запасной путь.

ЧЕСТНАЯ ЦЕНА. Сборка индекса — разовый оффлайн-прогон: замер на этой
машине 2.28 с на картинку (SigLIP2-so400m, 4 ядра CPU), то есть ~5 часов
на 7787 предметов Arms and Armor + Medieval Art + The Cloisters и ~20
часов на все 30 957 предметов каталога. Прогон РЕЗЮМИРУЕМЫЙ (падение,
Ctrl-C, обрыв сети продолжаются с того же места) и повторяется только при
обновлении дампа Мет. Во время рендера модель для полки не грузится
вообще: поиск — это одно матричное умножение над уже готовой матрицей.

ЧЕГО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ, названо прямо:
  * Не решает, какой кадр победит. Он только ДОБАВЛЯЕТ кандидатов в общий
    пул, где их судят те же гейты, что и всех (relevance, контрастивное
    вето, домен-гвард, резкость, дедуп) и то же ранжирование. Отнять
    кандидата, который нашёлся бы и без него, он не может по устройству.
  * Не заменяет стоки. У Pexels/Pixabay/Unsplash нет локальной полки, туда
    по-прежнему уходит текстовый запрос — это остаточное ограничение, а не
    решённая задача: их API принимают только слова.
  * Нет индекса на диске — модуль возвращает пустой список, и путь отбора
    БАЙТ-В-БАЙТ прежний. Флаг `SHELF_INDEX` даёт тот же откат явно.

Запуск:
    python scripts/shelf_index.py build [--limit N] [--departments "A,B"]
    python scripts/shelf_index.py search "a two-handed European sword"
    python scripts/shelf_index.py stats
"""
import argparse
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Каталог индекса — продукт, а не исходник (как temp_met_catalog): в git не
# хранится, пересобирается командой build.
INDEX_DIR = os.environ.get("SHELF_INDEX_DIR", os.path.join(REPO, "temp_shelf_index"))
ITEMS_PATH = os.path.join(INDEX_DIR, "items.jsonl")
VECTORS_PATH = os.path.join(INDEX_DIR, "vectors.f32")
IMAGES_DIR = os.path.join(INDEX_DIR, "images")

# Версия индекса входит в подпись отбора: смена способа построения обязана
# инвалидировать уже отрендеренные клипы, иначе правка не дойдёт до экрана
# на прогретом temp_smart/ (тот же урок, что уже усвоен с
# candidate_gate_signature и MUSEUM_SOURCES_VERSION).
SHELF_INDEX_VERSION = 1

# Модель полки ОБЯЗАНА совпадать с моделью, которой считается запрос — иначе
# сравниваются векторы из разных пространств, и результат будет выглядеть
# работающим (числа посчитаются), оставаясь шумом. Имя пишется в манифест и
# сверяется при загрузке.
SHELF_MODEL = "siglip2-so400m-patch14-384"


def stack_signature():
    """Версии библиотек, которыми СЧИТАЛИСЬ векторы полки.

    Имени модели недостаточно, и это не теория: дрейф ML-стека в этом
    репозитории уже задокументирован и уже сдвигал кадр золотого набора
    (`ep01_032`, 0.2006 при пороге 0.19) через порог БЕЗ единой правки кода.
    Полка — самый уязвимый к этому объект проекта: она собирается часами,
    лежит месяцами и переживает любое `pip install -U`. Имя модели при этом
    совпадает, гвард молчит, а векторы уже из слегка другого пространства.

    Не ошибка, а предупреждение — намеренно. Отказ обнулил бы 20 часов
    сборки из-за обновления библиотеки; молчание скрыло бы причину странного
    подбора. Правильное поведение — назвать расхождение и работать дальше.

    Сбой импорта -> None, тогда сверять нечего и никто ничего не печатает.
    """
    try:
        import torch, transformers
        return f"torch{torch.__version__}/tf{transformers.__version__}"
    except Exception:
        return None


def stack_drift(items, now):
    """Стеки, на которых собраны записи, но которые НЕ равны текущему.

    Отдельная чистая функция, а не три строки внутри load(): проверить её
    можно без собранного индекса и без 4 ГБ моделей, а внутри load() она
    была бы доступна только через постройку фикстуры из JSONL и сырой
    матрицы — то есть проверялась бы разглядыванием исходника.

    Пустое множество означает «сверять нечего ИЛИ всё совпало» — оба случая
    ведут к молчанию, и это намеренно: записи без поля `stack` (индексы,
    собранные до этой правки) не должны печатать предупреждение вечно.
    """
    if not now:
        return set()
    return {it.get("stack") for it in (items or []) if it.get("stack")} - {now}

# Отделы Мет, которые реально про этот канал. Полный каталог (30 957) тоже
# допустим (--departments all), это вопрос только времени сборки.
DEFAULT_DEPARTMENTS = ("Arms and Armor", "Medieval Art", "The Cloisters",
                       "European Sculpture and Decorative Arts", "Drawings and Prints",
                       "European Paintings", "Robert Lehman Collection")

_CACHE = {"loaded": False, "vectors": None, "items": None, "dim": None}


def _read_items():
    if not os.path.exists(ITEMS_PATH):
        return []
    out = []
    with open(ITEMS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # Обрыв на последней строке при падении процесса — обычное
                # дело для append-only файла. Битый хвост честнее выбросить,
                # чем уронить загрузку: вектор для него всё равно не дописан.
                break
    return out


def load():
    """Матрица векторов + метаданные, или None. Кэшируется в процессе."""
    if _CACHE["loaded"]:
        return (_CACHE["vectors"], _CACHE["items"])
    _CACHE["loaded"] = True
    try:
        import numpy as np
    except Exception:
        return (None, None)
    items = _read_items()
    if not items or not os.path.exists(VECTORS_PATH):
        return (None, None)
    # ЖЁСТКИЙ отказ при чужой модели. Векторы разных моделей — это разные
    # пространства, и косинус между ними посчитается без единой ошибки,
    # оставаясь шумом: отчёт выглядел бы рабочим, а подбор был бы случайным.
    # Это тот самый класс «молчаливого отказа», который в этом репозитории
    # уже один раз стоил незамеченного no-op у контрастивного вето, поэтому
    # здесь именно отказ, а не попытка продолжить.
    stale = {it.get("model") for it in items} - {SHELF_MODEL}
    if stale:
        print(f"  ВНИМАНИЕ: полка собрана другой моделью ({', '.join(sorted(map(str, stale)))}), "
              f"ожидается {SHELF_MODEL}. Полка НЕ используется — пересобрать: "
              f"python scripts/shelf_index.py build")
        return (None, None)
    # Стек — ПРЕДУПРЕЖДЕНИЕ, а не отказ (см. stack_signature): имя модели
    # совпадает, а версии библиотек могли уехать, и тогда векторы полки и
    # вектор запроса считаны слегка разными реализациями одной модели.
    # Строки без "stack" — индексы, собранные до этой правки; молчим про них,
    # иначе предупреждение печаталось бы на каждом старом индексе без повода.
    now = stack_signature()
    drifted = stack_drift(items, now)
    if drifted:
        print(f"  ВНИМАНИЕ: полка собрана на другом ML-стеке "
              f"({', '.join(sorted(map(str, drifted)))}), сейчас {now}. "
              f"Полка используется, но скоры могли поехать — при странном "
              f"подборе пересобрать: python scripts/shelf_index.py build")
    dim = items[0].get("dim")
    if not dim:
        return (None, None)
    raw = np.fromfile(VECTORS_PATH, dtype="float32")
    n = raw.size // dim
    if n < len(items):
        # Вектор дописывается ПОСЛЕ строки метаданных, поэтому хвост
        # метаданных может оказаться длиннее матрицы. Берём общий префикс —
        # это ровно то, что реально доведено до конца.
        items = items[:n]
    raw = raw[: len(items) * dim].reshape(len(items), dim)
    _CACHE["vectors"] = raw
    _CACHE["items"] = items
    _CACHE["dim"] = dim
    return (raw, items)


def available():
    vecs, items = load()
    return vecs is not None and items is not None and len(items) > 0


def stats():
    vecs, items = load()
    if vecs is None:
        return {"available": False, "items": 0}
    depts = {}
    for it in items:
        depts[it.get("dept") or "?"] = depts.get(it.get("dept") or "?", 0) + 1
    return {"available": True, "items": len(items), "dim": int(vecs.shape[1]),
            "model": items[0].get("model"), "departments": depts,
            "version": SHELF_INDEX_VERSION}


def _brief_vector(brief):
    """Эмбеддинг ОПИСАНИЯ КАДРА той же моделью, что и полка.

    Считается через visual_director._siglip2_text_emb — не второй копией
    вызова: там уже есть дисковый кэш и честный отчёт об обрезке текста по
    лимиту токенов (SIGLIP2_MAX_TEXT_LENGTH = 64, жёсткий предел текстовой
    башни). Вторая копия рано или поздно разошлась бы с первой по
    нормировке или по длине — и расхождение было бы невидимым."""
    import numpy as np
    import visual_director as vd
    emb = vd._siglip2_text_emb(brief)
    arr = emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
    arr = arr.reshape(-1).astype("float32")
    norm = float((arr * arr).sum()) ** 0.5
    return arr / norm if norm else arr


# Служебные слова имени и брифа: они есть почти у каждого предмета и
# совпадают ни о чём («pair of», «elements of», «close up», «dark background»).
_AGREEMENT_STOP = {
    "of", "the", "for", "with", "and", "from", "left", "right", "pair",
    "elements", "element", "portions", "part", "parts", "style", "century",
    "close", "seen", "front", "whole", "figure", "dark", "background",
    "steel", "plain", "other", "possibly", "probably", "complete",
}


def _content_words(text):
    import re
    return [w for w in re.findall(r"[a-z]+", (text or "").lower())
            if w not in _AGREEMENT_STOP and len(w) > 3]


def name_agreement(brief, limit=5):
    """Сколько из первых `limit` ответов полки названы ТЕМ, что просил бриф.

    Зачем нужна отдельная проверка ОТВЕТА. Полка всегда возвращает
    ближайших соседей — даже когда нужного предмета на ней нет вообще, и
    тогда она молча отдаёт свой центроид. Автору это неотличимо от успеха:
    список выглядит одинаково уверенно.

    ПОЧЕМУ НЕ ЧИСЛО ПО СКОРАМ (проверено и отклонено, 15.09). Первой была
    маржа top-1 над медианой полки, потом её нормировка на MAD, потом
    отрыв top1 от top10. Замер на 9 брифах трёх классов (предмет на полке
    есть / предмета нет / заведомо чужое) показал ПОЛНОЕ перекрытие всех
    трёх статистик: «припаркованная машина» давала маржу +0.087 против
    +0.071 у законного «латного поножа», а по z(MAD) чужое доходило до
    6.33 при 3.37 у настоящего нагрудника. Разделения нет; число, которое
    выглядит измерением и не разделяет, хуже отсутствия числа. Это тот же
    отрицательный результат, что уже дважды записан про попытки поставить
    абсолютный порог на сырой косинус CLIP.

    ЧТО РАБОТАЕТ. Полка отвечает не картинкой вообще, а КОНКРЕТНЫМ
    предметом, у которого есть собственное имя в каталоге музея. Сверка
    этого имени с брифом — не поиск, а проверка ответа, и словам здесь
    самое место. Сопоставление идёт через уже существующую
    `pipeline_smart.query_mentions_term()` В ОБЕ СТОРОНЫ: она знает и
    британское написание (`armour` против `armor` — иначе «Armor»
    не засчиталось бы к «plate armour»), и составной хвост (`visor`
    против `Visored sallet`). Вторую копию этих правил заводить нельзя —
    этот файл уже дважды документирует цену такой копии.

    ЭТО ПОДСКАЗКА, А НЕ РАЗДЕЛИТЕЛЬ — и вот измеренная причина
    формулировки. На полке из 342 предметов проверка разделяла чисто: все
    шесть присутствующих предметов давали не меньше 1 из 5, все шесть
    отсутствующих и заведомо чужих — ровно 0 из 5. На той же проверке при
    530 предметах разделение ПРОПАЛО: «рукописная миниатюра с битвой»
    получила 1 из 5, а «латный поножень» и «целый шлем без пробоины» — 0,
    хотя оба предмета на полке есть. Причина не в коде: полка растёт, top-5
    меняется, и результат, снятый на одном её размере, на другом не
    воспроизводится. Первая версия этого докстринга объявляла «перекрытия
    нет» — утверждение снято как верное только для снимка.

    Промахи имеют понятную природу и обе попытки их закрыть ИЗМЕРЕНЫ и
    отклонены: музей называет шлем своим термином (`Bascinet`, `Sallet`), а
    поножи — `Pair of greaves (Lower leg defenses)`, где значимое слово
    короче четырёх букв. Добавление классификации музея (`Helmets`,
    `Shields` — готовый словарь синонимов от хранителей) и снижение порога
    длины слова до трёх проверены по отдельности и вместе на одном наборе:
    по отдельности не меняют ничего, вместе ДОБАВЛЯЮТ ложное срабатывание
    («современная кухня с молоком» получает 1 из 5). Поэтому не внесены.

    Практический смысл остаётся: ноль совпадений — повод посмотреть на
    список глазами, а не приговор. Величина НИЧЕГО не блокирует и ни одного
    кандидата не отклоняет, только печатается человеку. И отвечает она на
    вопрос «полка ответила о том, что просили», а не «кадр хороший» —
    второе решают гейты и глаз на Шаге 7.5."""
    brief = (brief or "").strip()
    if not brief:
        return None
    res = search(brief, limit=limit)
    if not res:
        return None
    try:
        import pipeline_smart as ps
    except Exception:
        return None
    bw = _content_words(brief)
    if not bw:
        return None
    hits = 0
    for r in res:
        name = r.get("name") or ""
        nw = _content_words(name)
        if not nw:
            continue
        if (any(ps.query_mentions_term(brief, w) for w in nw)
                or any(ps.query_mentions_term(name, w) for w in bw)):
            hits += 1
    return (hits, len(res))


def search(brief, limit=40, min_score=None):
    """Кандидаты полки под ОПИСАНИЕ КАДРА, в порядке визуального сходства.

    Возвращает список словарей метаданных (не форму Pexels-кандидата —
    её собирает вызывающий код в pipeline_smart, чтобы этот модуль
    оставался независимым от формата пула)."""
    vecs, items = load()
    # .strip() здесь, а не только у вызывающего: пустой по сути бриф («   »)
    # проходил проверку на истинность и уходил в модель — лишний вызов и
    # заведомо бессмысленный вектор. Поймано тестом, не рассуждением.
    brief = (brief or "").strip()
    if vecs is None or not brief:
        return []
    try:
        import numpy as np
        q = _brief_vector(brief)
        if q.shape[0] != vecs.shape[1]:
            return []
        scores = vecs @ q
        order = np.argsort(-scores)[: max(1, int(limit))]
        out = []
        for i in order:
            s = float(scores[i])
            if min_score is not None and s < min_score:
                break
            rec = dict(items[int(i)])
            rec["score"] = s
            out.append(rec)
        return out
    except Exception:
        # Fail-open: полка — надстройка, её отказ не имеет права уронить
        # подбор слота, у которого есть все прежние источники.
        return []


# ---------------------------------------------------------------- сборка

def _iter_catalog_rows(departments, limit=None):
    import met_catalog
    idx = met_catalog._load()
    if not idx:
        raise SystemExit("Нет каталога Мет. Сначала: python scripts/met_catalog.py build")
    rows = idx.get("rows") or []
    if departments and departments != ("all",):
        # Список отделов — это ПРИОРИТЕТ, а не только фильтр. Сборка идёт
        # часами и переживает обрывы, поэтому порядок решает, что успеет
        # попасть в индекс раньше. Цена ошибки измерена 15.09: первые 700
        # предметов ушли в Arms and Armor, а молчание полки на брифах
        # эпизода оказалось сосредоточено в рукописях (Medieval Art, The
        # Cloisters) — тот же бюджет времени покрыл бы заметно больше
        # эпизода при другом порядке. Первая версия принимала список и
        # МОЛЧА игнорировала его порядок, сохраняя порядок каталога: флаг
        # выглядел как приоритет и им не был (проверено перебором очереди —
        # 298 предметов оружия в первых 300). Внутри отдела порядок
        # каталога сохраняется, поэтому сборка остаётся детерминированной.
        priority = {d: i for i, d in enumerate(departments)}
        rows = [r for r in rows if (r.get("dept") or "") in priority]
        rows.sort(key=lambda r: priority.get(r.get("dept") or "", len(priority)))
    if limit:
        rows = rows[: int(limit)]
    return rows


def _iter_europeana_rows():
    """Строки второго корпуса полки, ЛЕНИВО. Ходит в живой API Europeana —
    локального дампа, как у Мет, там нет, поэтому выгрузка идёт страницами
    по ходу сборки.

    Генератор, а не список, по двум причинам, и обе измеримы. Первая:
    `list(harvest(limit=N))` выкачивал N строк ДО того, как станет
    известно, сколько из них уже посчитано, — то есть резюмируемая сборка
    платила полный проход по API за строки, которые тут же выбрасывались.
    Вторая: список терял ВСЁ при сбое на последней странице, потому что
    исключение уносило уже собранное; генератор отдаёт собранное до сбоя.

    Отказ источника — конец выдачи, а не падение сборки: полка на
    предметах Мет остаётся ровно такой, какой была."""
    try:
        import europeana_corpus as ec
    except Exception as exc:
        print(f"Europeana недоступна ({type(exc).__name__}: {exc}) — "
              f"собираю только корпус Мет")
        return
    try:
        for row in ec.harvest():
            yield row
    except Exception as exc:
        print(f"Europeana оборвалась ({type(exc).__name__}: {exc}) — "
              f"собранное до этого места остаётся")


def _corpus_rows(corpus, departments):
    """Строки всех затребованных корпусов В ПОРЯДКЕ ПРИОРИТЕТА, лениво.

    Корпуса идут подряд, а не вперемешку: сборка резюмируемая и её можно
    оборвать на любой минуте, поэтому первым обязан лежать тот материал,
    который закрывает измеренное молчание полки. У Мет это отделы
    (`--departments`), у Europeana — коллекции (`COLLECTION_PRIORITY`).

    Имена корпусов проверяются СРАЗУ, до первой строки: у генератора тело
    не выполняется до первого `next()`, и неизвестное имя иначе всплыло бы
    не в момент запуска, а посреди сборки — или не всплыло бы вовсе, если
    выдачу никто не дочитал."""
    names = [c.strip().lower() for c in corpus if c.strip()]
    if "all" in names:
        names = ["europeana", "met"]
    for name in names:
        if name not in ("met", "europeana"):
            raise SystemExit(f"Неизвестный корпус: {name}")

    def _gen():
        for name in names:
            if name == "met":
                for row in _iter_catalog_rows(departments):
                    yield row
            else:
                for row in _iter_europeana_rows():
                    yield row
    return _gen()


def _image_url_for(object_id):
    """Ссылка на снимок предмета. Ходит в API Мет ЧЕРЕЗ museum_sources._met_get —
    там уже стоит адаптивный лимитер (старт 5 запр/с, каждый 403 режет
    скорость вдвое) и остывание источника. Своего лимитера здесь нет
    сознательно: два независимых счётчика к одному API вместе съели бы
    вдвое больше, чем разрешено, — тот же класс ошибки, что уже был
    закрыт для бюджета вызовов Gemini."""
    import museum_sources as ms
    o = ms._met_get(f"{ms.MET_API}/objects/{object_id}")
    if not o:
        return None, None, None
    if not o.get("isPublicDomain"):
        return None, None, None
    small = o.get("primaryImageSmall") or o.get("primaryImage")
    full = o.get("primaryImage") or o.get("primaryImageSmall")
    return small, full, o


def _row_id(row):
    """Единый ID предмета на всю систему.

    У строки Europeana он уже готов (`euro:/9200122/...`), у строки
    каталога Мет собирается из номера. Оба обязаны совпадать с тем, что
    отдаёт живой музейный путь (`met:<objectID>`): общий `used_ids` и
    дедуп должны видеть один предмет как один, каким бы путём он ни
    нашёлся."""
    rid = row.get("id")
    return rid if isinstance(rid, str) and ":" in rid else f"met:{rid}"


_PATH_TOKEN_FN = [None]   # см. _path_token(): разрешение запоминается один раз


def _path_token(rid):
    """ID предмета в виде, пригодном для ИМЕНИ ФАЙЛА.

    Правило одно на всю систему и живёт в одном месте
    (`pipeline_smart.candidate_path_token`): id со слэшами уже стоил
    проекту КАЖДОГО кандидата Europeana, потерянного до гейтов, а
    двоеточие в имени файла запрещено на Windows — то есть вторая,
    ослабленная копия этого правила здесь означала бы, что сборка полки
    наступает на те же грабли отдельно от отбора. Сегодня обе формы дают
    один и тот же результат побайтово (проверено тестом), поэтому уже
    скачанные превью не осиротеют; расходиться им запрещено впредь.

    Запасной путь на случай, если тяжёлый модуль отбора не импортируется
    (сборка полки не обязана тянуть PIL/numpy ради одной строки). Результат
    разрешения запоминается: НЕУДАЧНЫЙ импорт Python в `sys.modules` не
    кэширует, то есть без памятки сломанный или отсутствующий
    `pipeline_smart` заново пытался бы подняться на КАЖДОЙ строке каталога
    — десятки тысяч раз за одну сборку."""
    fn = _PATH_TOKEN_FN[0]
    if fn is None:
        try:
            import pipeline_smart as ps
            fn = ps.candidate_path_token
        except Exception:
            def fn(x):
                return re.sub(r"[^A-Za-z0-9._-]", "_", str(x)) or "noid"
        _PATH_TOKEN_FN[0] = fn
    return fn(rid)


def build(departments=DEFAULT_DEPARTMENTS, limit=None, keep_images=False,
          corpus=("met",)):
    """Разовая сборка индекса. Резюмируемая: пропускает уже посчитанные id."""
    import numpy as np
    import urllib.request
    import visual_director as vd

    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    done = {it.get("id") for it in _read_items()}
    # `--limit` — сколько НОВЫХ предметов добавить этим прогоном, а не
    # сколько строк корпуса посмотреть. Разница поймана на своей же
    # сборке: при старом счёте повторный запуск той же команды
    # (`--limit 6000` на полке, где 6000 уже есть) честно печатал
    # «осталось 0» и останавливался, хотя в корпусе оставалось 32 тысячи
    # записей, — выглядело как «полка собрана». Сейчас лимит считает
    # работу, а не просмотр, и выдача корпусов ленивая: лишних страниц у
    # чужого API не запрашивается вообще.
    #
    # `seen` начинается с уже посчитанного и пополняется ПО ХОДУ: одна и
    # та же запись может прийти дважды (две коллекции приоритета, два
    # корпуса), а `done` снимается один раз до цикла и такого дубля не
    # видит — он стоил бы второго вектора и второго кандидата одного и
    # того же предмета в пуле одного слота.
    seen = set(done)
    todo = []
    for r in _corpus_rows(corpus, departments):
        rid = _row_id(r)
        if rid in seen:
            continue
        seen.add(rid)
        todo.append(r)
        # Очередь у Europeana собирается ПО СЕТИ, страницами: без лимита
        # это минуты полного молчания перед первой строкой прогресса, и
        # отличить «идёт» от «повисло» человеку нечем. Строка раз в 500
        # предметов стоит ноль и отвечает на этот вопрос.
        if len(todo) % 500 == 0:
            print(f"  очередь: {len(todo)}...", flush=True)
        if limit and len(todo) >= int(limit):
            break
    print(f"Полка: уже посчитано {len(done)}, к сборке {len(todo)}"
          f"{' (лимит прогона)' if limit and len(todo) >= int(limit) else ''}")
    if not todo:
        return 0

    t0 = time.time()
    added = 0
    # Считается ОДИН раз на сборку, а не на запись: значение постоянно внутри
    # прогона, а вызов тянет импорт torch/transformers.
    _BUILD_STACK = stack_signature()
    ua = {"User-Agent": "Mozilla/5.0 (faceless-pipeline shelf index)"}
    with open(ITEMS_PATH, "a", encoding="utf-8") as items_f, \
            open(VECTORS_PATH, "ab") as vec_f:
        for n, r in enumerate(todo, 1):
            oid = r["id"]
            rid = _row_id(r)
            try:
                if r.get("image") or r.get("thumb"):
                    # Строка корпуса принесла ссылки с собой (Europeana) —
                    # лишний запрос к чужому API за тем, что уже известно,
                    # это не осторожность, а трата чужой квоты.
                    small = r.get("thumb") or r.get("image")
                    full = r.get("image") or r.get("thumb")
                    o = None
                else:
                    small, full, o = _image_url_for(oid)
                if not small:
                    continue
                path = os.path.join(IMAGES_DIR, _path_token(rid) + ".jpg")
                if not (os.path.exists(path) and os.path.getsize(path) > 2000):
                    req = urllib.request.Request(small, headers=ua)
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = resp.read()
                    if len(data) < 2000:
                        continue
                    with open(path, "wb") as fh:
                        fh.write(data)
                emb = vd._siglip2_image_emb(path)
                arr = emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
                arr = arr.reshape(-1).astype("float32")
                norm = float((arr * arr).sum()) ** 0.5
                if not norm:
                    continue
                arr = arr / norm
                rec = {"id": rid, "dim": int(arr.shape[0]),
                       "model": SHELF_MODEL, "version": SHELF_INDEX_VERSION,
                       "stack": _BUILD_STACK,
                       "name": r.get("name"), "title": r.get("title"),
                       "culture": r.get("culture"), "b": r.get("b"), "e": r.get("e"),
                       "dept": r.get("dept"),
                       "image": full or small, "thumb": small,
                       "page": r.get("page") or (o or {}).get("objectURL"),
                       # Право на публикацию хранится в самой строке
                       # индекса, а не во втором файле-манифесте: две
                       # копии одних и тех же данных рано или поздно
                       # разойдутся, и тогда непонятно, какой верить.
                       "source": r.get("source") or "met",
                       "rights": r.get("rights") or "isPublicDomain",
                       "provider": r.get("provider") or r.get("dept"),
                       # Разрешение источника: по готовому ролику иначе не
                       # ответить, был ли кадр мягким из-за самого снимка.
                       # У Мет поля нет — там медиана короткой стороны 2857
                       # по замеру, вопрос не стоит.
                       "image_size": r.get("image_size")}
                # Порядок важен: сперва метаданные, потом вектор. Обрыв между
                # ними даёт метаданные без вектора — это ловит load() по
                # длине матрицы. Обратный порядок дал бы вектор без подписи,
                # то есть молча сдвинутый индекс.
                items_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                items_f.flush()
                vec_f.write(arr.tobytes())
                vec_f.flush()
                added += 1
                if not keep_images:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            except KeyboardInterrupt:
                print("\nПрервано. Прогон резюмируемый — запустить build снова.")
                break
            except Exception as e:
                # Покарточный fail-open: один недоступный снимок не должен
                # уносить остальные тысячи.
                if n <= 5 or n % 200 == 0:
                    print(f"  [{oid}] пропущен: {type(e).__name__}: {e}")
                continue
            if n % 25 == 0:
                dt = time.time() - t0
                left = (len(todo) - n) * (dt / max(1, n))
                print(f"  {n}/{len(todo)}  {dt:.0f}с  ~{left/60:.0f} мин осталось", flush=True)
    print(f"Готово: добавлено {added} за {(time.time()-t0)/60:.1f} мин")
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--departments", default=",".join(DEFAULT_DEPARTMENTS),
                   help='список отделов через запятую, либо "all"')
    b.add_argument("--keep-images", action="store_true",
                   help="не удалять скачанные превью (для отладки/контактных листов)")
    b.add_argument("--corpus", default="met",
                   help='какие корпуса собирать: "met", "europeana", '
                        '"met,europeana" или "all" (Europeana первой — её '
                        'материал закрывает измеренное молчание полки)')
    s = sub.add_parser("search")
    s.add_argument("brief")
    s.add_argument("--limit", type=int, default=10)
    sub.add_parser("stats")
    a = ap.parse_args()
    if a.cmd == "build":
        deps = tuple(x.strip() for x in a.departments.split(",") if x.strip())
        corpus = tuple(x.strip() for x in a.corpus.split(",") if x.strip())
        build(departments=deps, limit=a.limit, keep_images=a.keep_images,
              corpus=corpus)
    elif a.cmd == "search":
        res = search(a.brief, limit=a.limit)
        if not res:
            print("Полка пуста или не собрана: python scripts/shelf_index.py build")
        for r in res:
            print(f"  {r['score']:+.4f}  {r['id']:>12}  {r.get('name')}  "
                  f"| {r.get('title')}  | {r.get('culture')} {r.get('b')}-{r.get('e')}")
    else:
        print(json.dumps(stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
