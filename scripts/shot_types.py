#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тип кадра слота и маршрутизация запроса по источникам.

ЗАЧЕМ (измеренная причина, 13-14.09). A/B отбора на девяти реальных слотах
эпизода 02 показал два регресса, которых не лечит ни один фильтр и ни одно
ранжирование:

* «medieval plate armour museum» — поиск Мет свободным текстом по слову
  *plate* отдаёт КЕРАМИЧЕСКИЕ ТАРЕЛКИ («Plate with Water Bird», «Plate with
  Wife Beating Husband») и настенные часы. Они честно проходят паспорт
  (Европа, XV век) и честно проигрывают только настоящему доспеху, которого
  в пробной выборке не было.
* «medieval castle moat water» — у музеев вообще нет фотографии рва: Мет
  отдаёт «Мадонну с младенцем» и «Брак в Кане», relevance 0.14-0.21. Они
  вытесняли фотографию замка из Openverse просто потому, что стояли раньше
  в списке.

Общее у обоих: запрос уходит НЕ В ТОТ ИСТОЧНИК и НЕ В ТОМ ВИДЕ. Музей — это
каталог ПРЕДМЕТОВ с паспортом, а не фотобанк сцен; спрашивать у него «ров в
тумане» бессмысленно, а спрашивать «доспех» нужно по полю отдела, а не
свободным текстом.

ЧТО ИЗМЕРЕНО ЖИВЬЁМ (Met API, 14.09):
    запрос                          без отдела          departmentId=4
    medieval plate armour museum    480 (тарелки,       277 (доспехи,
                                     часы, Ислам)        первые шесть — латы)
    medieval rondel dagger           17                  15 (те же кинжалы)
    medieval castle moat water       10 (Мадонна,         0
                                     Брак в Кане)
То есть структурный запрос убирает мусор предметного слота, ничего не теряя
на настоящем предмете, и честно даёт пусто там, где музей не про это.

УСТРОЙСТВО. Тип кадра ставится АВТОРОМ в самой строке запроса сценария:

    HOOK: medieval knight plate armour closeup [object], medieval battlefield mud [scene]

Не поставлен — выводится из слов запроса консервативным словарём, и если
сигнала нет, тип остаётся `any`, а маршрут — сегодняшний, все источники.
Ноль регрессии там, где мы не знаем: гадать нельзя, это то же правило, что
у омографов и у атмосферного слоя.

Источники объявляют себя ДАННЫМИ (`SOURCE_CAPABILITIES`), а не развилкой в
коде — новый источник добавляется строкой таблицы.
"""
import re

SHOT_TYPES = ("object", "scene", "illustration", "texture", "map")
ANY = "any"

# Какие типы кадра источник реально умеет. Список исключений, а не
# разрешений: всё, чего здесь нет, источнику НЕ отправляется.
#
# Музеи исключены из `scene`/`texture` ПО ЗАМЕРУ, а не по принципу: на
# сценических запросах их кандидаты — живопись и рукописи с relevance
# 0.14-0.21, и они выигрывали слот только потому, что стояли первыми в
# списке. Стоки остаются во ВСЕХ типах последними: они соревнуются под теми
# же гейтами, и лучше слабый кандидат в пуле, чем пустой слот (ЧАСТЬ 13).
SOURCE_CAPABILITIES = {
    "museum":    {"supports": ("object", "illustration", "map", ANY), "structured": True},
    # Полка (scripts/shelf_index.py) — тот же корпус Мет, что у "museum", но
    # спрашивается НЕ словами, а сравнением описания кадра с самими
    # изображениями. Типы кадра пока объявлены ТЕ ЖЕ, что у музея: на
    # `scene` музейный корпус измеренно слаб, и хотя визуальный поиск
    # находит там рукописные миниатюры с битвами (замер 15.09), отдельного
    # A/B по победителям сценических слотов ещё не было — расширять список
    # без замера значило бы повторить ровно ту ошибку, которую замер
    # маршрутизации 14.09 уже исправил.
    "shelf":     {"supports": ("object", "illustration", "map", ANY), "structured": True},
    # Wikimedia Commons: хроники, трактаты, картины и снимки предметов —
    # сцены в том числе (миниатюра битвы — это сцена, нарисованная
    # современником; замер эп.94, docs/quality/RESEARCHER_PROTO_EP94.md).
    # Фактур там почти нет — не спрашиваем.
    "commons":   {"supports": ("object", "scene", "illustration", "map", ANY)},
    "openverse": {"supports": ("object", "scene", "illustration", "texture", "map", ANY)},
    "pexels":    {"supports": ("object", "scene", "illustration", "texture", "map", ANY)},
    "pixabay":   {"supports": ("object", "scene", "illustration", "texture", "map", ANY)},
    "unsplash":  {"supports": ("object", "scene", "illustration", "texture", "map", ANY)},
}

# Словарь вывода типа по словам запроса. Порядок проверки — сверху вниз,
# первый сработавший выигрывает, и порядок здесь ИЗМЕРЕННЫЙ, не
# произвольный. Первая версия ставила `object` выше `scene`, и прогон по 42
# реальным запросам эпизода 02 показал систематический промах: «medieval
# battlefield armour mud», «medieval knight armour fallen mud», «knight
# armour fallen ground», «medieval helmet lying dirt» уезжали в музей как
# предметные — из-за слова armour/helmet, — хотя это сцены, и в A/B ровно
# такой слот выиграла музейная шкатулка из слоновой кости. Правило,
# подтверждённое данными: если в запросе есть И предмет, И место — это
# СЦЕНА с предметом внутри, а у музея её нет.
#
# `illustration`/`map` стоят выше сцены, потому что там решает НОСИТЕЛЬ
# изображения: «medieval manuscript battle illustration» — это рукопись, а
# не поле боя, и рукописи как раз лучшее, что отдают музеи и архивы.
# Словарь канала — в channel_profile.json (`shot_type_lexicon`); в коде
# пусто. Нет словаря — тип `any`: все источники, как до маршрутизации.
_TYPE_LEXICON = ()

# Отдел Мет для СТРУКТУРНОГО запроса предметного слота. Один отдел на
# запрос: параметр `departmentId` у Мет принимает одно значение, а
# несколько отделов означали бы несколько запросов и лишнюю квоту.
# Творческая настройка канала — переопределяется в channel_profile.json
# (`museum_departments`), как блоклист и палитра.
# (слова, id отдела, для каких типов кадра) — первая группа, подходящая
# и по типу, и по словам, выигрывает. Тип проверяется ПЕРВЫМ: «medieval
# archer armour manuscript» — иллюстрация, и отдел ей нужен «Средневековое
# искусство» (17), а не «Оружие и доспехи» (4), хотя слово armour там есть
# (реальный промах первой версии на 42 запросах эпизода).
# Отделы канала — в channel_profile.json (`museum_departments`); в коде
# пусто. Нет отделов — музей спрашивается свободным текстом.
_MET_DEPARTMENTS_DEFAULT = ()

_WORD_RE = re.compile(r"[a-z]+")

# Английская морфология: «coins», «armoured», «blades», «ruins» — это те же
# слова, и совпадение по ЦЕЛОМУ слову их не ловит (реальный промах первой
# версии: «medieval gold coins hoard» и «medieval sabaton armoured foot»
# остались без типа). Совпадение по НАЧАЛУ слова ловит их все, но создаёт
# обратный риск — ровно тот, что этот проект уже проходил на словаре
# атмосферы («зал» ловил ЗАЛП). Поэтому у префикса есть минимальная длина,
# а известные ловушки заперты отдельным тестом.
MIN_PREFIX_LEN = 5


def _term_matches(term, words, low):
    """Слово запроса против термина словаря: многословный термин — по
    подстроке, короткий — по целому слову, длинный — по началу слова."""
    if " " in term:
        return term in low
    if term in words:
        return True
    if len(term) < MIN_PREFIX_LEN:
        return False
    return any(w.startswith(term) for w in words)


def _profile():
    import channel_profile
    return channel_profile.load()


def _lexicon():
    """Словарь типов кадра; канал может переопределить его целиком."""
    custom = _profile().get("shot_type_lexicon")
    if not custom:
        return _TYPE_LEXICON
    try:
        return tuple((t, tuple(words)) for t, words in custom)
    except Exception:
        return _TYPE_LEXICON


def _met_departments():
    custom = _profile().get("museum_departments")
    if not custom:
        return _MET_DEPARTMENTS_DEFAULT
    try:
        return tuple((tuple(words), int(dep), tuple(types)) for words, dep, types in custom)
    except Exception:
        return _MET_DEPARTMENTS_DEFAULT


_SPEC_RE = re.compile(r"^(.*?)\s*\[\s*([a-zA-Z_]+)\s*\]\s*$")


def parse_query_spec(raw):
    """'medieval sword macro [object]' -> ('medieval sword macro', 'object').

    Скобка с НЕИЗВЕСТНЫМ словом не съедается молча: она остаётся частью
    запроса, и автор видит её в логе линта — иначе опечатка `[objct]`
    тихо выключила бы маршрутизацию, а запрос поехал бы вместе со скобкой
    в API. Нет скобки -> (запрос, None)."""
    text = (raw or "").strip()
    m = _SPEC_RE.match(text)
    if not m:
        return text, None
    body, word = m.group(1).strip(), m.group(2).strip().lower()
    if word in SHOT_TYPES:
        return body, word
    return text, None


def infer_shot_type(query):
    """Тип кадра по словам запроса. None — сигнала нет, гадать не будем."""
    words = set(_WORD_RE.findall((query or "").lower()))
    low = (query or "").lower()
    for shot_type, terms in _lexicon():
        for term in terms:
            if _term_matches(term, words, low):
                return shot_type
    return None


def shot_type_for(query, explicit=None):
    """Явный тип из сценария важнее выведенного; нет ни того ни другого —
    `any` (все источники, сегодняшнее поведение)."""
    if explicit in SHOT_TYPES:
        return explicit
    return infer_shot_type(query) or ANY


def source_supports(source, shot_type):
    caps = SOURCE_CAPABILITIES.get(source)
    if not caps:
        return True     # неизвестный источник не глушим — fail-open
    return (shot_type or ANY) in caps["supports"]


def met_department_for(query, shot_type=None):
    """Отдел Мет для структурного запроса. None — спрашиваем свободным
    текстом, как раньше.

    Только для предметных и иллюстративных слотов: на сценическом запросе
    отдел не сузил бы мусор, а обнулил бы выдачу (замер: `medieval castle
    moat water` + отдел 4 -> 0 objectID), и слот пошёл бы в фолбэк вместо
    честной попытки архивов."""
    resolved = shot_type or ANY
    # Только явно предметный или иллюстративный слот. `any` — это «тип не
    # известен», и сужать по отделу на догадке нельзя: сегодняшнее
    # поведение (свободный текст) остаётся байт-в-байт.
    if resolved not in ("object", "illustration"):
        return None
    words = set(_WORD_RE.findall((query or "").lower()))
    low = (query or "").lower()
    for terms, dep, types in _met_departments():
        if resolved not in types:
            continue
        for term in terms:
            if _term_matches(term, words, low):
                return dep
    return None
