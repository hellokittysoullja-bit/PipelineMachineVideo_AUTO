# -*- coding: utf-8 -*-
"""Europeana как ВТОРОЙ корпус визуальной полки — там, где музей молчит.

ЗАЧЕМ. Первый живой прогон сверки брифов по всему эпизоду 02 (142 брифа,
полка 530 предметов) дал число и, что важнее, назвал МЕСТО молчания: полка
отвечала по теме на 79 брифов, а оставшиеся 63 сосредоточились ровно в
BLOCK 4 «Ответ» (земля, грязь), BLOCK 5 «Двести метров» (поход по полю),
BLOCK 8 (геральдика, деньги), BLOCK 10 (лагерь, болезнь, надгробия),
BLOCK 11 (кости, раскопки). Это не случайный хвост: Метрополитен — каталог
ПРЕДМЕТОВ, и сцены он не отдаёт вообще, а рукописная миниатюра — ровно тот
носитель, где средневековая СЦЕНА нарисована современником события.

Europeana закрывает и то, и другое, и это измерено, а не предположено.
ЛЮБОЕ число ниже названо вместе с датой, и это не формальность: корпус
ЖИВОЙ, учреждения докладывают записи, и за несколько дней одного замера
он вырос на 6% (все свободные записи окна 209 142 -> 222 576). Тот же
урок уже записан про размер полки: замер по растущему корпусу без
названного размера не воспроизводится. Перемерить — одна дешёвая команда
`python scripts/europeana_corpus.py probe`, она задаёт ровно тот же
вопрос, что задаёт сборка.

    замер 15.09, окно эпохи 900-1600 (из channel_profile), живой API,
    ключ не нужен, только изображения, reusability=open:

    всего свободных                                       222 576
    из них CC0                                              4 266
    из них Public Domain Mark                              34 063
    CC0 + PDM, то есть корпус этого модуля                 38 329

(Первая версия этого блока называла три числа сразу — 38 025 в одной
строке и 38 262 в двух других, — потому что снимки делались в разные
часы и ни один не был датирован. Число, за которым не стоит дата, в
живом корпусе неверно по построению.)

Для сравнения: весь прошедший паспорт корпус Мет — 30 957 предметов. То
есть это не добавка к полке, а второй корпус того же порядка, причём
другой природы: KB Нидерландов 4 361 (рукописи), Альбертина 6 866
(рисунки и гравюры), Рейксмузеум 14 596.

ПОЧЕМУ PDM ЗДЕСЬ БЕРЁТСЯ, А У OPENVERSE — НЕТ. Это выглядит как
послабление и им не является. Отказ от PDM в `stock_fetch_multisource`
обоснован не самой меткой, а тем, КТО её ставит: там источник без
модерации (случайный аккаунт на Flickr) может пометить чужую работу как
свободную. У Europeana загрузить запись физически некому, кроме
учреждения-агрегатора: это институциональная сеть по построению. И главное
— полка УЖЕ стоит на институциональном утверждении: у Мет берётся
`isPublicDomain`, то есть собственное заявление музея, а не оформленный
отказ от прав. Требовать от Рейксмузеума больше, чем от Метрополитена,
было бы не строгостью, а непоследовательностью. CC-BY и BY-SA НЕ берутся
ни в каком виде — там условие лицензии (атрибуция в описании ролика), а
сборщика атрибуций у пайплайна нет; это то же правило, что и везде, и
стоит оно дорого: 166 949 записей The Portable Antiquities Scheme под
CC BY 3.0 в корпус не входят.

Белый список учреждений при этом НЕ церемония. В фасете 94 провайдера, и
все, кроме одного, — музеи, национальные и университетские библиотеки,
архивы. Исключение ровно одно и оно настоящее: `IMSLP/Petrucci Music
Library` — community-сайт, куда ноты заливают сами пользователи, то есть
единственный самотегирующийся источник в списке. Ради него список и
существует.

ПАСПОРТ — честно о том, что здесь СЛАБЕЕ, чем у Мет. Эпоха такая же
надёжная: Europeana нормализует `year`, и запись без года отбрасывается
(fail-closed), а не додумывается. Культура — слабее: поля `Culture` у
Europeana нет вообще, есть страна ПРОВАЙДЕРА, а это не культура предмета
(у Рейксмузеума лежат японские гравюры). Поэтому чужая культура ищется
теми же терминами `museum_sources.culture_is_foreign()`, но по ТЕКСТУ
записи (заголовок, тип, автор, место, тематические метки). Это текстовая
эвристика, а не паспорт хранителя, и называется здесь именно так.

Ключ API. `EUROPEANA_API_KEY` из `.env`, если он есть. Если нет —
публичный демонстрационный ключ самой Europeana, задокументированный ею
для проб; проверено живьём 15.09, работает. Он общий и может быть
ограничен или отозван в любой момент — свой ключ бесплатен и снимает этот
риск, но регистрация это решение владельца, а не условие работы модуля.

Аудит-трейл прав отдельным файлом не заводится СОЗНАТЕЛЬНО: каждая строка
индекса полки и так несёт `rights`, `provider`, `page` и `id` записи —
второй файл с теми же данными разошёлся бы с первым.

Запуск:
    python scripts/europeana_corpus.py probe
    python scripts/europeana_corpus.py harvest --limit 2000 --out corpus.jsonl
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

SEARCH_URL = "https://api.europeana.eu/record/v2/search.json"
THUMB_URL = "https://api.europeana.eu/thumbnail/v2/url.json"

#: Версия способа сборки корпуса. Входит в подпись отбора через
#: shelf_index: другой состав корпуса — другой победитель слота.
EUROPEANA_CORPUS_VERSION = 1

#: Публичный демо-ключ Europeana. Не секрет и не чужой ключ: он
#: опубликован самой Europeana для проб. Свой ключ — EUROPEANA_API_KEY.
DEMO_KEY = "api2demo"

CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
PDM = "http://creativecommons.org/publicdomain/mark/1.0/"
#: Только эти два. См. докстринг: BY/BY-SA требуют атрибуции в описании
#: ролика, а собирать её пайплайну нечем.
SAFE_RIGHTS = (CC0, PDM)

#: Учреждения, дающие 99% корпуса окна эпохи (замер 15.09 по фасету
#: DATA_PROVIDER). Список fail-closed: незнакомый провайдер не берётся,
#: добавить — одна строка. IMSLP намеренно НЕ включён (см. докстринг).
TRUSTED_PROVIDERS = (
    "Rijksmuseum",
    "The Albertina Museum",
    "KB, National Library of the Netherlands",
    "Tallinn City Museum",
    "National Gallery of Denmark",
    "Catholic University of Leuven",
    "Loevestein Castle",
    "Valkhof Museum",
    "Castle Huis Bergh",
    "Harju County Museum",
    "National Library of Israel",
    "Collective Catalog of the Library Network of the State Archives",
    "Cartographic and Geological Institute of Catalonia",
    "National Library of Portugal",
    "City Museum Zutphen",
    "Museum Rotterdam",
    "Geldersch Landschap & Kasteelen",
    "Leiden University Libraries",
    "Malmö Museum",
    "Regional Archive Nijmegen",
    "Kulturen",
    "Sörmland Museum",
    "The British Library",
    "Deutsche Fotothek",
    "Saaremaa Museum",
    "VU University Amsterdam Library",
    "Leipzig University Library",
    "Järvamaa Museum",
    "TIB - Leibniz Information Centre for Science and Technology and University Library",
    "Digital Memory of Catalonia",
    "Nationalmuseum Sweden",
    "Narva Museum",
    "Digital Library Real Academia de la Historia",
    "Estonian History Museum",
    "Library of the Wroclaw University",
    "National Archives of the Netherlands",
    "Jagiellonian Library",
)

#: Порядок сборки — ПО КОЛЛЕКЦИИ, а не по словам. Тот же урок, что уже
#: оплачен на отделах Мет дважды: сборка идёт часами и переживает обрывы,
#: поэтому список — это ПРИОРИТЕТ, а не только фильтр; и спрашивать
#: источник надо ПОЛЕМ, а не совпадением в описании.
#:
#: Почему НЕ тематическим запросом — замерено 15.09 и это главный
#: аргумент за всю визуальную полку: метаданные этого корпуса написаны
#: по-нидерландски и по-немецки. Английские тематические слова по нему
#: почти не работают: `knight` 16 записей, `armour` 5, `archer` 2,
#: `crossbow` 5, `heraldry` 5 — при 38 329 записях в окне (15.09). Работают
#: каталожные термины (`manuscript` 4 303, `miniature` 3 213), то есть
#: ровно те, что ничего не говорят о СОДЕРЖАНИИ кадра. Собирать корпус
#: словами здесь физически нечем — и именно поэтому полка спрашивает
#: картинки, а не текст: нидерландски описанная миниатюра отвечает
#: английскому брифу без единого общего слова.
#:
#: Первой идёт коллекция, закрывающая измеренное молчание полки, — 4 269
#: средневековых иллюминованных рукописей KB (BYVANCK). Проверено
#: выборкой: первые записи это миниатюры 1372 года.
COLLECTION_PRIORITY = (
    "9200122_Ag_EU_TEL_a0031_KB",          # 4 269 рукописных миниатюр
    "15508_Ag_AT_Kulturpool_albertina",    # 6 866 рисунков и гравюр
    "90402_M_NL_Rijksmuseum",              # 14 559
    "2021648_DigitaleCollectie_CollectieGelderland",
    "401_Muuseumid",
    "2020903_Museu_NationalGalleryOfDenmark",
    "318_Ag_JHN_NationalLibraryIsrael",
    "816_Leiden_University_Libraries_Bibliotheca_Thysiana",
)

#: Разрешение снимка — СТРУКТУРНОЕ поле Europeana (`IMAGE_SIZE`), а не
#: догадка: тот же принцип, что `departmentId` у Мет. Спрашивать им
#: обязательно, и это найдено собственной ошибкой, а не рассуждением.
#:
#: Замер 15.09, по 24 реально скачанным снимкам полки: у КАЖДОГО из них
#: короткая сторона 750 px при кадре 1080, медиана Мет для сравнения —
#: 2857. Разбор по фасету объяснил, почему так вышло: коллекция, которую я
#: поставил ПЕРВОЙ в приоритет, состоит из `medium` (2905) и `small`
#: (1317) и не содержит НИ ОДНОГО снимка выше — то есть по разрешению я
#: сам выбрал худшее, что есть в корпусе. У Альбертины 3114 `large` +
#: 2956 `extra_large`, у Рейксмузеума 13 000 `extra_large`.
#:
#: Почему `small` отброшен, а `medium` оставлен — это РАЗНЫЕ случаи, и
#: разницу видно на скачанных файлах. `medium` у KB это 750x1094,
#: портретный: `aspect_fit_backdrop()` вписывает такой кадр ПО ВЫСОТЕ в
#: 0.92*1080 = 994, то есть работает почти один к одному, без растяжения.
#: `small` у той же коллекции — 750x500, альбомный: до 1920 по ширине это
#: 2.56x растяжения ещё ДО зума Ken Burns. Первое терпимо, второе нет.
#:
#: Порядок именно приоритет, а не фильтр: сборка резюмируемая и её можно
#: оборвать на любой минуте, поэтому крупные снимки обязаны попасть в
#: индекс раньше мелких — внутри КАЖДОЙ коллекции.
IMAGE_SIZE_PRIORITY = ("extra_large", "large", "medium")

#: `small` (2 138 записей окна) намеренно не входит в приоритет выше.
IMAGE_SIZE_EXCLUDED = ("small",)

#: Ширина превью для эмбеддинга. Модель полки работает на 384 px, поэтому
#: 400 достаточно, а снимок идёт с CDN самой Europeana — это и вежливее к
#: учреждениям, и однороднее: не нужно знать про 429 у одного хранилища и
#: обязательный заголовок у другого. Полноразмерный `edmIsShownBy`
#: сохраняется в записи и берётся на рендере.
THUMB_WIDTH = "w400"

_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (faceless-pipeline shelf index)"}

#: Пауза между страницами поиска. Демо-ключ общий — вежливость не
#: опциональна.
PAGE_PAUSE_SEC = 0.35

#: Попытки на страницу поиска. Сборка корпуса идёт часами, и разовый
#: сетевой сбой не имеет права стоить остатка коллекции.
SEARCH_ATTEMPTS = 2
SEARCH_RETRY_PAUSE_SEC = 2.0


def api_key():
    return (os.environ.get("EUROPEANA_API_KEY") or "").strip() or DEMO_KEY


def _norm(value):
    return " ".join(str(value or "").split()).strip().lower()


_TRUSTED_NORM = frozenset(_norm(p) for p in TRUSTED_PROVIDERS)


def is_safe_rights(values):
    """Fail-closed: права обязаны быть ИМЕННО из списка.

    Сравнение по началу строки, потому что Europeana отдаёт ссылку и с
    завершающим слэшем, и без, а иногда с `https`. Ничего не распознали —
    False, а не «наверное можно»."""
    if isinstance(values, str):
        values = [values]
    for v in values or []:
        u = _norm(v).replace("https://", "http://").rstrip("/")
        for safe in SAFE_RIGHTS:
            if u == safe.rstrip("/"):
                return True
    return False


def is_trusted_provider(values):
    """Fail-closed белый список учреждений."""
    if isinstance(values, str):
        values = [values]
    return any(_norm(v) in _TRUSTED_NORM for v in (values or []))


def record_years(rec):
    """Годы записи из нормализованного Europeana поля `year`.

    Нет года — None, и запись не берётся. Додумывать эпоху по заголовку
    здесь нельзя: весь смысл корпуса в том, что дата известна."""
    years = []
    for v in rec.get("year") or []:
        s = str(v).strip()
        # Три ИЛИ четыре цифры. Три — не послабление, а следствие
        # собственного замера: `year_clause()` намеренно добавляет ветку
        # `YEAR:[900 TO 999]`, потому что часть записей хранит трёхзначный
        # год без дополнения нулями (77 записей на окне этого канала).
        # Принимать только `len == 4` значило запрашивать эти записи
        # страницами и выбрасывать КАЖДУЮ как «нет года» — ветка запроса
        # была чистым no-op, и молча: в отчёте они уходили в `no_year`
        # вместе с настоящими записями без даты. Проверено прогоном:
        # record_years({"year": ["950"]}) давало None.
        if s.isdigit() and 3 <= len(s) <= 4:
            years.append(int(s))
    if not years:
        return None
    return min(years), max(years)


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _lang_aware_en(block):
    """Значения из *LangAware-словаря: английские, затем безъязыковые."""
    out = []
    if isinstance(block, dict):
        for key in ("en", "def"):
            for v in block.get(key) or []:
                if isinstance(v, str) and not v.startswith("http"):
                    out.append(v)
    return out


def record_text(rec):
    """Весь текст записи, по которому ищется чужая культура.

    Поля `Culture` у Europeana нет — это текстовая эвристика, а не
    паспорт хранителя, и так она и называется в докстринге модуля."""
    parts = []
    for key in ("title", "dcCreator", "dataProvider", "country", "dcDescription"):
        for v in (rec.get(key) or []) if isinstance(rec.get(key), list) else [rec.get(key)]:
            if isinstance(v, str) and not v.startswith("http"):
                parts.append(v)
    parts += _lang_aware_en(rec.get("dcTypeLangAware"))
    parts += _lang_aware_en(rec.get("dcSubjectLangAware"))
    parts += _lang_aware_en(rec.get("dctermsSpatial"))
    if isinstance(rec.get("edmConceptPrefLabelLangAware"), dict):
        parts += _lang_aware_en(rec["edmConceptPrefLabelLangAware"])
    return " ".join(parts)


def object_name(rec):
    """Имя вида предмета — то, чем его называет учреждение.

    Именно оно сверяется с брифом в `shelf_index.name_agreement()`, и
    именно поэтому берётся `dcType` («Manuscript», «historiated initial»,
    «miniature»), а не заголовок конкретной работы."""
    skip = ("still image", "image", "text", "object", "photograph")
    names = [v for v in _lang_aware_en(rec.get("dcTypeLangAware"))
             if _norm(v) not in skip]
    if not names and isinstance(rec.get("edmConceptPrefLabelLangAware"), dict):
        # Запасной источник имени вида: у части учреждений (Альбертина)
        # dcType пуст, а тематический концепт Europeana заполнен. Без
        # запасного пути такие записи пришли бы на полку безымянными, и
        # `name_agreement()` не смог бы сказать о них вообще ничего.
        names = [v for v in _lang_aware_en(rec["edmConceptPrefLabelLangAware"])
                 if _norm(v) not in skip]
    return names[0] if names else None


def thumb_url(image_url, width=THUMB_WIDTH):
    if not image_url:
        return None
    q = urllib.parse.urlencode({"uri": image_url, "type": "IMAGE", "size": width})
    return f"{THUMB_URL}?{q}"


def _search(params, timeout=60):
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _rights_clause():
    return "(" + " OR ".join(f'RIGHTS:"{r}"' for r in SAFE_RIGHTS) + ")"


def year_clause(lo, hi):
    """Окно эпохи в том виде, в каком его понимает Europeana.

    РЕАЛЬНАЯ ловушка, найденная живым замером 15.09, а не предположенная:
    `YEAR` у Europeana — СТРОКОВОЕ поле, и диапазон сравнивается
    лексически. Окно этого канала (900-1600) в наивной записи
    `YEAR:[900 TO 1600]` даёт РОВНО НОЛЬ записей — потому что «1372»
    лексически меньше «900», то есть верхняя граница оказывается ниже
    нижней. Дополненное нулями `YEAR:[0900 TO 1600]` на том же запросе
    даёт 38 329 (замер 15.09). Ошибка была бы немой: пустая выдача у
    молча остался бы выключенным, — ровно тот класс «слой есть, и он
    ничего не даёт», которым этот репозиторий уже горел пять раз.

    Второй измеренный факт: годы лежат в обоих написаниях. `[0900 TO
    0999]` даёт 0, а `[900 TO 999]` — 77 записей, то есть у части записей
    трёхзначный год хранится без дополнения. Поэтому при нижней границе
    меньше 1000 к окну добавляется вторая, неполная ветка — иначе эти
    записи не нашлись бы никогда.

    Сам фильтр при этом влияет только на ПОЛНОТУ выдачи: год каждой
    записи всё равно перепроверяется локально (`record_years` +
    `museum_sources.era_overlaps`), поэтому ошибка здесь не может
    пропустить предмет чужой эпохи, только потерять свой."""
    lo, hi = int(lo), int(hi)
    parts = [f"YEAR:[{max(lo, 0):04d} TO {hi:04d}]"]
    if lo < 1000:
        parts.append(f"YEAR:[{max(lo, 1)} TO {min(hi, 999)}]")
    return "(" + " OR ".join(parts) + ")" if len(parts) > 1 else parts[0]


def _row(rec):
    """Запись Europeana в ФОРМЕ строки корпуса полки — той же, что даёт
    каталог Мет. Не прошедшая паспорт запись возвращает None вместе с
    причиной, чтобы отчёт сборки не был немым."""
    import museum_sources as ms

    rights = rec.get("rights")
    if not is_safe_rights(rights):
        return None, "rights"
    provider = _first(rec.get("dataProvider"))
    if not is_trusted_provider(rec.get("dataProvider")):
        return None, "provider"
    years = record_years(rec)
    if not years:
        return None, "no_year"
    b, e = years
    if not ms.era_overlaps(b, e):
        return None, "era"
    if ms.culture_is_foreign(record_text(rec)):
        return None, "culture"
    image = _first(rec.get("edmIsShownBy"))
    if not image:
        return None, "no_image"
    rid = rec.get("id")
    if not rid:
        return None, "no_id"
    return {
        "id": f"euro:{rid}",
        "dept": provider,
        "name": object_name(rec),
        "title": _first(rec.get("title")),
        # Поля культуры у Europeana нет, и подставлять сюда страну
        # провайдера значило бы выдать местонахождение за происхождение.
        "culture": None,
        "b": b, "e": e,
        "image": image,
        "thumb": thumb_url(image) or _first(rec.get("edmPreview")),
        "page": f"https://www.europeana.eu/item{rid}",
        "rights": _first(rights),
        "provider": provider,
        "source": "europeana",
    }, None


def harvest(limit=None, collections=None, page_size=100,
            era=None, sizes=None):
    """Строки корпуса по приоритету коллекций и разрешения снимка.

    Генератор — корпус большой, а сборка индекса всё равно идёт по одной
    картинке. Внутри каждой коллекции сначала идут крупные снимки: сборка
    резюмируемая, и оборвать её можно на любой минуте.

    `collections=None` — один сплошной проход по всему окну эпохи без
    разбиения по коллекциям (полнее, но порядок произвольный).
    `sizes=None` — без разбиения по разрешению, включая `small`."""
    import museum_sources as ms

    # Списки читаются В МОМЕНТ ВЫЗОВА, а не связываются значением по
    # умолчанию при объявлении функции. Разница не стилистическая:
    # `collections=COLLECTION_PRIORITY` в сигнатуре запоминает СПИСОК на
    # момент импорта, и подмена `ec.COLLECTION_PRIORITY` снаружи (замер,
    # эксперимент, правка порядка сборки из другого модуля) молча не
    # действовала — сборка шла по старому порядку и выглядела рабочей.
    # Поймано собственным замером 15.09: опыт «собрать только Альбертину»
    # вернул рукописи KB и отчитался «осталось 0». Тот же принцип «читать
    # в момент вызова», которым в этом репозитории уже закрыт реестр
    # флагов.
    if collections is None:
        collections = COLLECTION_PRIORITY
    if sizes is None:
        sizes = IMAGE_SIZE_PRIORITY
    if era is None and ms.era_window_declared() is None:
        # Та же причина, что у met_catalog.build(): корпус собирается ЧАСАМИ
        # под окно эпохи, и окно из константы модуля дало бы средневековый
        # корпус каналу любой другой ниши.
        print("  ВНИМАНИЕ: окно эпохи не объявлено ни в channel_profile.json "
              "(era_from/era_to), ни авто-нишей — корпус будет собран под "
              "окно из константы модуля.")
    lo, hi = era or ms.era_window()
    stats = {"seen": 0, "kept": 0, "pages": 0, "errors": 0,
             "rejected": {}, "by_provider": {}, "by_size": {}}
    # Счётчики публикуются СРАЗУ, живым словарём, а не в конце генератора.
    # Разница не косметическая: `harvest.stats = stats` в самом конце
    # выполняется только при ПОЛНОМ исчерпании, и любой потребитель,
    # оборвавший чтение (islice, ранний break, свой лимит), читал бы
    # пустой словарь из прошлой жизни — проверено прогоном:
    # `harvest.stats['kept']` после трёх элементов роняло KeyError.
    # Словарь мутируется на месте, поэтому ранняя привязка ещё и делает
    # числа видимыми ПО ХОДУ сборки, а не только после неё.
    harvest.stats = stats
    taken = 0
    seen_ids = set()
    buckets = [(c, z) for c in (list(collections) if collections else [None])
               for z in (list(sizes) if sizes else [None])]
    for coll, size in buckets:
        if limit and taken >= limit:
            break
        cursor = "*"
        while cursor:
            if limit and taken >= limit:
                break
            qf = ["TYPE:IMAGE", year_clause(lo, hi), _rights_clause()]
            if coll:
                qf.append(f'europeana_collectionName:"{coll}"')
            if size:
                qf.append(f"IMAGE_SIZE:{size}")
            params = {"wskey": api_key(), "query": "*:*", "rows": int(page_size),
                      "profile": "rich", "reusability": "open", "cursor": cursor,
                      "qf": qf}
            data = None
            for attempt in range(SEARCH_ATTEMPTS):
                try:
                    data = _search(params)
                    break
                except Exception as exc:
                    # Один повтор с паузой ПЕРЕД тем, как признать
                    # учреждение упавшим: сборка идёт часами, и разовый
                    # сетевой сбой не должен стоить остатка коллекции —
                    # он обрывал бы её молча, на середине, а `cursor`
                    # восстановить уже нечем. Не помогло — fail-open
                    # поисточниково, как у музеев: упавшее учреждение не
                    # уносит с собой остальные.
                    if attempt + 1 < SEARCH_ATTEMPTS:
                        time.sleep(SEARCH_RETRY_PAUSE_SEC * (attempt + 1))
                        continue
                    stats["errors"] += 1
                    print(f"  [{coll}/{size}] поиск упал: "
                          f"{type(exc).__name__}: {exc}")
            if data is None:
                break
            stats["pages"] += 1
            items = data.get("items") or []
            if not items:
                break
            for rec in items:
                stats["seen"] += 1
                row, why = _row(rec)
                if row is None:
                    stats["rejected"][why] = stats["rejected"].get(why, 0) + 1
                    continue
                # Одна запись может лежать в ДВУХ коллекциях приоритета —
                # тогда она приходит дважды и попадает на полку дважды:
                # второй вектор, вторые 2.28с эмбеддинга и один и тот же
                # предмет двумя кандидатами в пуле одного слота. Проверено
                # прогоном на двух корзинах: id повторялся буквально.
                if row["id"] in seen_ids:
                    stats["rejected"]["duplicate"] = \
                        stats["rejected"].get("duplicate", 0) + 1
                    continue
                seen_ids.add(row["id"])
                row["image_size"] = size
                stats["by_size"][size] = stats["by_size"].get(size, 0) + 1
                stats["kept"] += 1
                stats["by_provider"][row["dept"]] = \
                    stats["by_provider"].get(row["dept"], 0) + 1
                taken += 1
                yield row
                if limit and taken >= limit:
                    break
            nxt = data.get("nextCursor")
            # Курсор, равный текущему, — это бесконечный цикл, а не
            # следующая страница. Своей выдачей Europeana такого не давала,
            # но лимит без потолка пишется один раз и живёт годами.
            cursor = None if (not nxt or nxt == cursor) else nxt
            time.sleep(PAGE_PAUSE_SEC)


harvest.stats = {}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Корпус Europeana для визуальной полки")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="сколько записей даёт корпус прямо сейчас")
    p.add_argument("--sample", type=int, default=20)
    h = sub.add_parser("harvest", help="выгрузить строки корпуса в jsonl")
    h.add_argument("--limit", type=int, default=None)
    h.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.cmd == "probe":
        import museum_sources as ms
        if ms.era_window_declared() is None:
            print("  ВНИМАНИЕ: окно эпохи не объявлено ни в "
                  "channel_profile.json (era_from/era_to), ни авто-нишей — "
                  "цифры ниже посчитаны под окно из константы модуля.")
        lo, hi = ms.era_window()
        data = _search({"wskey": api_key(), "query": "*:*", "rows": 0,
                        "profile": "minimal", "reusability": "open",
                        "qf": ["TYPE:IMAGE", year_clause(lo, hi),
                               _rights_clause()]})
        print(f"Ключ: {'свой' if api_key() != DEMO_KEY else 'демонстрационный'}")
        print(f"Окно эпохи {lo}-{hi}, свободные права CC0/PDM, только "
              f"изображения: {data.get('totalResults')} записей")
        rows = list(harvest(limit=a.sample))
        st = harvest.stats
        print(f"Проба {a.sample}: взято {st['kept']} из {st['seen']}, "
              f"отклонено {st['rejected']}")
        for r in rows[:8]:
            print(f"  {r['id']}\n     {r['name']} — {r['title']} "
                  f"[{r['dept']} {r['b']}-{r['e']}]")
        return 0

    n = 0
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    try:
        for row in harvest(limit=a.limit):
            n += 1
            if out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if out:
            out.close()
    st = harvest.stats
    print(f"Взято {n} строк; просмотрено {st['seen']}, страниц {st['pages']}, "
          f"отклонено {st['rejected']}")
    for prov, cnt in sorted(st["by_provider"].items(), key=lambda x: -x[1]):
        print(f"  {cnt:>6}  {prov}")
    for size, cnt in sorted(st["by_size"].items(), key=lambda x: -x[1]):
        print(f"  {cnt:>6}  разрешение: {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
