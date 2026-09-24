"""Прямые открытые API музеев — источник кадров, где эпоха ИЗВЕСТНА, а не
угадывается по пикселям.

Зачем это существует (разбор 10.09, эпизод 02_ne-mechom). В опубликованный
ролик про Азенкур попали танк на кульминации «Рыцарей убивала земля»,
терракотовая армия, египетский саркофаг, космонавт и кавалерия XIX века.
Замер показал, что запросы слотов были ПРАВИЛЬНЫЕ («medieval rondel dagger»),
а пустым был пул: Pexels — библиотека современной lifestyle-фотографии,
средневековья там нет, и на запрос без совпадений он отдаёт ближайшее по
вектору. Гейты после этого спорят, приемлем ли космонавт.

Три попытки научить модель ОТЛИЧАТЬ эпоху по картинке провалились (две
задокументированы в CLAUDE.md, третья — «разбиение по эпохе» — не прошла
золотой набор: настоящее макро доспеха выглядит «не средневековым» сильнее
танка, а брак с современной улицей выглядит «средневековым», потому что в
кадре костюм рыцаря). Причина общая: правильность кадра живёт в паре
«фраза+картинка» и в контексте эпохи, а эмбеддинг-косинус меряет похожесть
пикселей на текст.

Здесь угадывать не нужно. Музей вместе со снимком отдаёт ПАСПОРТ предмета:
    Met      objectBeginDate=1300  objectEndDate=1375  culture='possibly Italian'
    Cleveland creation_date_earliest=1570 ... latest=1580  culture=['North Italy, 16th c.']
    Chicago  date_start=1500 date_end=1530  place_of_origin='Nuremberg'
Это данные хранителей коллекции, а не догадка по пикселям. Танк не может
пройти не потому, что мы его распознали, а потому что в музее нет предмета с
датой 1400 год, который был бы танком; терракотовая армия отсеивается по
культуре («Chinese»), саркофаг — по ней же и по дате.

Лицензии: берём ТОЛЬКО то, что каждый музей сам пометил как public domain
(Met isPublicDomain, Cleveland share_license_status == "CC0", Chicago
is_public_domain). Атрибуция НЕ требуется — сознательное условие: канал не
обязан вставлять список авторов в описание ролика (ровно поэтому сюда не
берутся by/by-sa источники, см. решение владельца от 10.09). Ключи API не
нужны ни одному из трёх.

Fail-open на каждом шаге: недоступный музей возвращает пустой список, пул
продолжает собираться из остальных источников — ни один слот не пустеет
из-за этого модуля.
"""
import concurrent.futures
import hashlib
import itertools
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import feature_flags
except ImportError:  # модуль используется и как библиотека, и standalone
    feature_flags = None

UA = "Mozilla/5.0 (compatible; FacelessPipeline/1.0)"
HTTP_TIMEOUT = 20

# Окно эпохи канала. Тот же смысл, что era_from/era_to у validate_brief()
# локального режиссёра (docs/quality/DIRECTOR_LOCAL_LLM.md) — здесь оно
# применяется к ПАСПОРТУ предмета, а не к тексту брифа.

# Культуры/страны вне ниши канала. Проверяется по паспорту предмета, а не по
# картинке: "Iranian or Turkish, 1401-1600" проходит по дате и обязано
# отсекаться здесь — иначе восточный доспех XV века попадёт в ролик про
# Азенкур как «эпоха совпала».
# СПИСОК ПЕРЕЕХАЛ В channel_profile.json (`foreign_culture_terms`, 25.09).
# Раньше он применялся ко ВСЕМУ, у чего профиль объявил эпоху, — скрытая
# связь «есть окно эпохи -> весь неевропейский мир чужой»: клон под нишу
# «Древний Египет» с окном эпохи отсекал бы собственный Египет. История
# замеров, по которым список дополнялся, сохранена ниже.
    # Дополнено 14.09 ПО ЗАМЕРУ, а не по интуиции: локальный каталог Мет
    # (scripts/met_catalog.py) впервые показал ВЕСЬ корпус, прошедший
    # паспорт, — 31 182 предмета, — и в нём нашлись культуры, которых в
    # списке не было. На узком запросе это било в полную силу: у
    # «european longsword blade macro» выдача состояла ИЗ ДВУХ предметов, и
    # оба были из этой дыры («Knife blade, Afghan», «Blade (Kudi
    # tranchang), Javanese»). Дефект не каталога — он всё это время
    # действовал и на живом API-пути, просто там его нечем было увидеть.
    # Счёт в индексе: javanese 54, mongol 5, afghan 4.
    # Дополнено 16.09, тем же методом и по той же причине: пересборка
    # каталога показала, что намерение списка («доколумбова Америка —
    # чужая», отсюда mesoamerican/aztec/maya/inca/peru выше) НЕ ВЫПОЛНЯЛОСЬ,
    # потому что Мет каталогизирует эти предметы ДРУГИМИ именами. Найдено
    # негативным контролем на живом брифе: «a steel gorget and bevor
    # covering the throat and the neck» первыми четырьмя ответами дал
    # НЕОЖЕРЕЛЬЯ доколумбовой Америки. Счёт по ИСХОДНОМУ дампу (не по
    # индексу — он не хранит колонку Country, и первый замер по нему дал
    # заниженные 350): colombia 188, costa rica 166, panama 87, tairona 68,
    # ecuador 59, veracruz 31, mixtec 23, olmec 11, indigenous american 6,
    # pre-columbian 5, toltec 5 — ВСЕГО 649 предметов, индекс 30 957 ->
    # 30 308. Каждый проверен поимённо: совпадение у 150 в Culture, у 299 в
    # Country, у 200 в обоих, и РОВНО НОЛЬ совпало только в Title — то есть
    # класса ложного удаления по случайному слову в названии здесь нет.
    #
    # НЕ добавлены по замеру, а не по забывчивости:
    #   honduras — ЛОВУШКА, поймана негативным контролем: единственное
    #     совпадение в корпусе это «Casket, Italian, Venice» (термин попал
    #     не в culture), то есть правило удалило бы подлинный венецианский
    #     ларец. Тот же класс, что «зал» внутри «ЗАЛП».
    #   precolumbian, guatemala, nazca, moche, zapotec — в корпусе НОЛЬ
    #     совпадений. Термин, который сегодня ничего не отсекает, завтра
    #     отсечёт неизвестно что; список дополняется по данным.
    # Дополнено 15.09, ТРЕТИЙ случай того же класса и тем же методом.
    # Намерение «индонезийский архипелаг — чужой» уже было записано в
    # список 14.09 словами java/javanese — и выполнялось только для Явы.
    # Найдено словарём каталога: список реальных имён предметов по теме
    # главы про кинжалы выдал «Dagger (Golok or pedang) with sheath» и
    # «Dagger (Bade-bade) with sheath» рядом с настоящим Roundel dagger.
    # Счёт по ИСХОДНОМУ ДАМПУ — 101 предмет, индекс 30 308 -> 30 208.
    # Счёт по индексу дал бы 86 и был бы занижен ровно по той же причине,
    # что и 16.09: индекс не хранит колонку Country, а паспорт её смотрит.
    # Совпадение: у 77 в Culture, у 23 в Country, у 1 в обоих и РОВНО НОЛЬ
    # только в Title — то есть ложного удаления по случайному слову в
    # названии здесь нет. Проверено поимённо: балийские и суматранские
    # крисы, копья с ножнами, клеванги, филиппинские чётки и ожерелья из
    # отдела Asian Art. Ни одного европейского предмета.
    #
    # НЕ добавлены, проверено негативным контролем:
    #   bali — ЛОВУШКА: совпадает внутри «Kabbalism» на европейской
    #     гравюре. Берётся только «balinese».
    #   moro — ЛОВУШКА: «Morose», «Nemorosus», «Amorosi» — шесть из девяти
    #     совпадений европейские, и ни одного в поле культуры.
    #   aceh, borneo, bornean, malay, sundanese, celebes, sulawesi,
    #     filipino, siamese — в корпусе НОЛЬ совпадений; термин, который
    #     сегодня ничего не отсекает, завтра отсечёт неизвестно что.
DEFAULT_FOREIGN_CULTURE_TERMS = ()

# СПОРНЫЕ культуры — решение творческое, а не техническое, и поэтому за
# владельцем, а не за кодом. Все четыре паспорт сегодня пропускает:
#   Byzantine, Coptic — христианские, средневековые по датам, но не
#     западноевропейские; византийский доспех в ролике про Азенкур это
#     вопрос вкуса, а не анахронизм в том же смысле, что яванский крис.
#   Armenian (4 предмета), Georgian (1) — Кавказ, та же развилка.
#   Mexican (485 предметов, 16.09) — САМЫЙ крупный спорный случай, и
#     разбор выборки показал, что он смешанный, а не однородный: рядом
#     лежат «Figure, Mexico, 1467-1533» (доколумбова) и «Rosary pendant,
#     Mexican, 1500-1599» / «Triptych, possibly Mexican, 1500-1599» —
#     колониальные христианские предметы XVI века, по форме европейские.
#     Запретить термин целиком значило бы выбросить розарии и триптихи,
#     разрешить — оставить доколумбовые фигуры. Разделить их кодом нечем:
#     поле culture у обеих групп одинаковое. Это вкус канала, не анахронизм.
# Добавить их — одна строка в channel_profile.json ("foreign_culture_terms"),
# и список выше не трогается. Молча решать это за канал я не стал.

# departmentId у API и название отдела в дампе — одно и то же, но выражены
# по-разному. Таблица нужна, чтобы локальный поиск фильтровал ровно тот же
# отдел, что и структурный запрос к API (см. scripts/shot_types.py).
MET_DEPARTMENT_NAMES = {
    4: "Arms and Armor",
    7: "The Cloisters",
    11: "European Paintings",
    17: "Medieval Art",
}

MET_API = "https://collectionapi.metmuseum.org/public/collection/v1"
CLEVELAND_API = "https://openaccess-api.clevelandart.org/api/artworks"
CHICAGO_API = "https://api.artic.edu/api/v1/artworks/search"

# Met отдаёт только objectID в поиске — карточка каждого предмета это
# ОТДЕЛЬНЫЙ запрос. Потолок держит цену слота вменяемой; кэш ниже делает
# повторные прогоны бесплатными.
#
# ГЛУБИНА ИЗМЕРЕНА, А НЕ ВЫБРАНА НА ГЛАЗ (13.09, 42 авторских запроса эпизода
# 02_ne-mechom, живые API без ключей). Потолок 12 был не ценой корпуса, а
# нашим собственным: у 27 запросов из 42 поиск Мет отдаёт БОЛЬШЕ 12 objectID
# (медиана 19.5, максимум 2992 на "medieval poleaxe weapon"). То есть слот
# голодал не потому, что в музее нет предметов, а потому что мы смотрели
# первые двенадцать. Выход, прошедший паспортные фильтры (public domain +
# era_overlaps + culture_is_foreign), на глубине 12 против 60:
#     medieval poleaxe weapon          7 -> 42
#     medieval sword museum display    6 -> 24
#     medieval plate armour museum     3 -> 20
#     medieval manuscript knight battle 10 -> 26
#     medieval rondel dagger           5 -> 12
# Цена по времени НЕ выросла: карточки тянутся параллельно
# (MET_DETAIL_WORKERS), 60 штук приходят за 1.7-6.5с — примерно столько же,
# сколько занимали 12 последовательных запросов. Порядок выдачи сохраняется
# (ex.map отдаёт результаты в порядке входа), поэтому релевантность поиска
# Мет по-прежнему определяет место кандидата в пуле.
MET_MAX_DETAIL_FETCHES = 60
MET_DETAIL_WORKERS = 12

# Кливленд и Чикаго отдают полную карточку прямо в поиске — там глубина стоит
# РОВНО ОДИН запрос, независимо от limit. Тем же замером: Чикаго на limit=20
# даёт 12-17 прошедших паспорт кандидатов, на limit=100 — 48-80 (четырёхкратно,
# бесплатно). Кливленд на средневековой европейской теме отдаёт 0-1 при любом
# limit (его коллекция про другое) — там выигрыша нет, но и цены тоже.
SEARCH_PAGE_SIZE = 100

# Глубина выдачи для ДОПОЛНИТЕЛЬНОЙ (более широкой) формулировки слота при
# слиянии рангов — см. scripts/query_fusion.py. Широкая формулировка нужна
# ради СОГЛАСИЯ с точной, а вес её позиций и так убывает: кандидат на 20-й
# позиции формулировки с весом 0.5 приносит 0.5/(10+21) = 0.016, тогда как
# первое место точной формулировки стоит 1/11 = 0.091.
#
# ЧЕСТНО: это граница ЦЕНЫ, а не доказательство бесполезности хвоста —
# бонус 0.016 всё ещё способен поднять кандидата примерно на четыре позиции
# в точном списке, и более глубокая широкая выдача что-то бы добавляла.
# Без границы слот стоил бы 3x60=180 карточек Мет вместо 60 (на эпизоде из
# 42 запросов это 7560 запросов к API, гарантированный 403 и получасовой
# прогон); с границей — 60+20+20=100, то есть 1.67x, и это названо прямо.
VARIANT_DETAIL_FETCHES = 20

_SEARCH_CACHE = {}


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return json.load(r)


# --- ВЕЖЛИВОСТЬ К МЕТ -------------------------------------------------------
# Не перестраховка, а реальный, пойманный вживую отказ (13.09): при замере
# глубины подряд ушло ~600 запросов карточек за пару минут, и Мет ответил
# HTTP 403 уже на ПОИСК — то есть источник выключился целиком, а не одна
# карточка. Через минуту доступ вернулся сам: это троттлинг, а не бан.
# Глубина 60 делает такой всплеск штатным (42 запроса эпизода x 60 карточек),
# поэтому у Мет теперь есть три вещи, которых не было:
#   1) общий на процесс ограничитель скорости (не на пул потоков одного
#      запроса — иначе каждый следующий слот начинал бы заново с полной
#      скорости);
#   2) один повтор с паузой на «временных» кодах, ПОСЛЕ которого карточка
#      честно считается потерянной;
#   3) счётчик потерь и пауза-остывание на весь источник: молча уменьшившийся
#      пул неотличим от бедного корпуса, а это ровно тот класс тихой
#      деградации, который в этом проекте ловили уже трижды.
# Скорость — АДАПТИВНАЯ, и стартовое значение получено отказом, а не выбрано.
# Второй замер (13.09, все 42 запроса эпизода подряд): при 10 запросах/с
# Мет ответил 403 на 289-м запросе, через ~39 секунд — то есть ~7.5/с в
# устойчивом режиме для него уже много, при том что 199 запросов за 23с
# прошли чисто. Порог документирован Метом как «80/с», на практике
# срабатывает раньше и, судя по всему, считается по окну в минуту.
# Поэтому: старт 5/с, и КАЖДЫЙ 403 вдвое снижает скорость до конца прогона
# (пол — 1/с) вдобавок к паузе-остыванию. Итоговая скорость пишется в
# FETCH_STATS — по артефактам видно, до чего дошло.
MET_MAX_REQUESTS_PER_SEC = 5.0
MET_MIN_REQUESTS_PER_SEC = 1.0
MET_RETRY_STATUSES = (403, 429, 500, 502, 503, 504)
# На паузу источник уходит ТОЛЬКО по сигналам троттлинга/недоступности, а не
# по любой ошибке из списка повторов. Разница не косметическая: 500 на одной
# карточке предмета — это сломанная карточка, а не «Мет просит подождать», и
# уводить из-за неё весь источник на минуту значит терять все остальные
# карточки того же запроса. Ровно этот класс («один битый элемент кладёт
# весь слой») уже дважды чинился 13.09 в звуке: битый ассет ронял
# планировщик эффектов, а нечитаемый вход — весь вызов ffmpeg в сведении.
MET_COOLDOWN_STATUSES = (403, 429, 503)
MET_RETRY_PAUSE_SEC = 2.0
MET_COOLDOWN_SEC = 60.0

_MET_LOCK = threading.Lock()
_MET_NEXT_SLOT = [0.0]
_MET_COOLDOWN_UNTIL = [0.0]
_MET_RATE = [MET_MAX_REQUESTS_PER_SEC]
# Видимость деградации: сколько карточек потеряно, сколько раз источник
# уходил в остывание и на какой скорости закончил. Читается вызывающим
# кодом/тестами и уезжает в media_plan/source_contribution.json.
FETCH_STATS = {"met_cards_lost": 0, "met_cooldowns": 0, "met_requests": 0,
               "met_catalog_hits": 0,
               "met_rate_final": MET_MAX_REQUESTS_PER_SEC,
               "search_cache_hits": 0, "search_cache_misses": 0}


def reset_fetch_stats():
    """Один прогон — один счёт. Нужен тестам и повторным вызовам в процессе."""
    for k in ("met_cards_lost", "met_cooldowns", "met_requests",
              "met_catalog_hits",
              "search_cache_hits", "search_cache_misses"):
        FETCH_STATS[k] = 0
    _MET_RATE[0] = MET_MAX_REQUESTS_PER_SEC
    FETCH_STATS["met_rate_final"] = MET_MAX_REQUESTS_PER_SEC
    _MET_COOLDOWN_UNTIL[0] = 0.0
    _MET_NEXT_SLOT[0] = 0.0


def _met_throttle():
    """Общий на процесс интервал между запросами к Мет."""
    with _MET_LOCK:
        now = time.monotonic()
        slot = max(now, _MET_NEXT_SLOT[0])
        _MET_NEXT_SLOT[0] = slot + 1.0 / _MET_RATE[0]
        FETCH_STATS["met_requests"] += 1
    delay = slot - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def met_is_cooling_down():
    return time.monotonic() < _MET_COOLDOWN_UNTIL[0]


def _met_enter_cooldown():
    with _MET_LOCK:
        if not met_is_cooling_down():
            _MET_COOLDOWN_UNTIL[0] = time.monotonic() + MET_COOLDOWN_SEC
            FETCH_STATS["met_cooldowns"] += 1
            _MET_RATE[0] = max(MET_MIN_REQUESTS_PER_SEC, _MET_RATE[0] / 2.0)
            FETCH_STATS["met_rate_final"] = _MET_RATE[0]
            print(f"    Мет: троттлинг, источник на паузе {MET_COOLDOWN_SEC:.0f}с, "
                  f"дальше {_MET_RATE[0]:.1f} запр/с (остальные музеи работают)")


def _met_get(url):
    """Запрос к Мет через ограничитель, с одним повтором на временных кодах.

    Возвращает None вместо исключения — вызывающий код обязан отличать
    «карточка не пришла» от «карточка не подошла по паспорту»."""
    if met_is_cooling_down():
        return None
    for attempt in (0, 1):
        _met_throttle()
        try:
            return _get_json(url)
        except urllib.error.HTTPError as e:
            if e.code in MET_RETRY_STATUSES and attempt == 0:
                time.sleep(MET_RETRY_PAUSE_SEC)
                continue
            if e.code in MET_COOLDOWN_STATUSES:
                _met_enter_cooldown()
            else:
                FETCH_STATS["met_lost_cards"] = FETCH_STATS.get("met_lost_cards", 0) + 1
            return None
        except Exception:
            if attempt == 0:
                time.sleep(MET_RETRY_PAUSE_SEC)
                continue
            return None
    return None


def _profile():
    """channel_profile.json — через общий читатель (channel_profile.py)."""
    import channel_profile
    return channel_profile.load()


# МИР ЭПИЗОДА СИЛЬНЕЕ КАНАЛА, КАНАЛ СИЛЬНЕЕ ПУСТОТЫ (план 24.09). Окно
# эпохи и чужие культуры решают, какой музейный предмет вообще попадёт в пул,
# — до этого они всегда брались из средневековых дефолтов кода, и эпизод про
# Египет или космос на этом канале получал ноль музейных кадров, а клон
# репозитория под другую нишу — средневековый фильтр по умолчанию.
# Теперь:
#   * паспорт эпизода (set_episode_world) — окно эпохи из паспорта; чужие
#     культуры — список канала плюс исключения паспорта, МИНУС всё, что
#     паспорт называет своим (culture.include: эпизод про Египет снимает
#     «egypt» из списка средневекового канала);
#   * нет паспорта — профиль канала (channel_profile.json: era_from/era_to и
#     foreign_culture_terms); список DEFAULT_FOREIGN_CULTURE_TERMS выше — это
#     выверенные по каталогу Мет имена, и применяется он только каналом,
#     который объявил эпоху в профиле;
#   * нет ни того, ни другого — фильтра эпохи и культуры нет вовсе.
_EPISODE_WORLD = {}


def set_episode_world(card):
    """Мир эпизода из паспорта (world_card) или None — сбросить."""
    import world_card
    _EPISODE_WORLD.clear()
    if not card:
        return
    _EPISODE_WORLD["era"] = world_card.era_window(card)
    _EPISODE_WORLD["card"] = card
    _EPISODE_WORLD["set"] = True


def _channel_era():
    p = _profile()
    if "era_from" in p and "era_to" in p:
        return int(p["era_from"]), int(p["era_to"])
    return None


def era_window():
    """(от, до) или None — эпоху не фильтровать."""
    if _EPISODE_WORLD.get("set"):
        return _EPISODE_WORLD.get("era")
    return _channel_era()


def foreign_culture_terms():
    p = _profile()
    channel = ()
    if "foreign_culture_terms" in p:
        channel = tuple(p["foreign_culture_terms"])
    if not _EPISODE_WORLD.get("set"):
        return channel
    import world_card
    return world_card.apply_culture(channel, _EPISODE_WORLD["card"])


def era_overlaps(begin, end):
    """Пересекается ли период предмета с окном эпохи канала.

    Пересечение, а не вхождение: у составных предметов музей ставит широкий
    диапазон ("Armor, 1375-1950" — итальянский доспех XIV века, part которого
    реставрирован в XX). Требовать полного вхождения значило бы выбрасывать
    подлинники. Японский доспех 1701-1800 при окне 900-1600 не пересекается
    и отсеивается — то, ради чего фильтр и нужен.
    """
    win = era_window()
    if win is None:
        return True
    lo, hi = win
    try:
        b, e = int(begin), int(end)
    except (TypeError, ValueError):
        return False
    if b == 0 and e == 0:
        return False
    return b <= hi and e >= lo


def culture_is_foreign(*fields):
    """Есть ли в паспорте предмета указание на культуру вне ниши канала."""
    terms = foreign_culture_terms()
    blob = " ".join(str(f).lower() for f in fields if f)
    return any(t in blob for t in terms)


# Институт искусств Чикаго отдаёт снимки только с этим заголовком: без него
# IIIF отвечает 403 на ЛЮБУЮ ширину (проверено на 843, 1920 и full/full).
# Требование их документации — назвать проект и контакт.
CHICAGO_IMAGE_HEADERS = {
    "AIC-User-Agent": "FacelessPipeline (github.com/faceless-pipeline)",
}


def _candidate(cid, title, image_url, page_url, meta, headers=None, thumb_url=None):
    """Кандидат в ФОРМЕ PEXELS — чтобы конкурировать в общем пуле под теми же
    гейтами, что Pexels и Openverse, а не жить отдельной веткой отбора (тот
    же приём, что _openverse_search_photos).

    thumb_url -> src["medium"]: уменьшенная версия для ОЦЕНКИ кандидата
    (CLIP/эстетика/дедуп работают на 224-384 px), полноразмерная качается
    только у победителя — см. candidate_probe_url() в pipeline_smart."""
    out = {
        "id": cid,
        "alt": title or "",
        "url": page_url or image_url,
        "src": {"large2x": image_url},
        "_museum_meta": meta,
    }
    if thumb_url:
        out["src"]["medium"] = thumb_url
    if headers:
        # Заголовки едут ВМЕСТЕ с кандидатом, а не ищутся по его источнику в
        # чужом модуле: скачивающий код не должен знать, у какого музея какие
        # требования.
        out["_download_headers"] = dict(headers)
    return out


# КАРТОЧКИ ПРЕДМЕТОВ МЕТ КЭШИРУЮТСЯ ПО objectID (замер 24.09, прогон
# эпизода 94: 863 запроса карточек на 437 разных предметов — половина
# запросов повторяла уже полученную карточку, и каждый стоял в очереди
# ограничителя Мет). Карточка предмета от запроса не зависит; решение
# «подходит ли предмет» по-прежнему принимается заново на каждом запросе
# (паспорт эпизода мог смениться). Кэшируется только полученная карточка —
# отказ не замораживается. Срок и папка — те же, что у кэша поиска.
_MET_CARD_CACHE = {}


def _met_card_path(oid):
    return os.path.join(MUSEUM_CACHE_DIR, "met_objects", f"{int(oid)}.json")


def _met_card_cached(oid):
    o = _MET_CARD_CACHE.get(oid)
    if o is not None:
        return o
    try:
        p = _met_card_path(oid)
        if time.time() - os.path.getmtime(p) < MUSEUM_CACHE_TTL_SEC:
            with open(p, encoding="utf-8") as f:
                o = json.load(f)
            _MET_CARD_CACHE[oid] = o
            FETCH_STATS["met_card_cache_hits"] = FETCH_STATS.get("met_card_cache_hits", 0) + 1
            return o
    except (OSError, ValueError, TypeError):
        pass
    return None


def _met_card_store(oid, o):
    _MET_CARD_CACHE[oid] = o
    try:
        p = _met_card_path(oid)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(o, f, ensure_ascii=False)
        os.replace(tmp, p)
    except (OSError, TypeError, ValueError):
        pass


def search_met(query, limit=MET_MAX_DETAIL_FETCHES, department=None):
    """Метрополитен: отдел Arms and Armor — лучшая в мире коллекция
    европейского доспеха, всё public domain, ключ не нужен.

    department — СТРУКТУРНЫЙ запрос по полю API вместо свободного текста
    (см. scripts/shot_types.py). Замер 14.09: «medieval plate armour
    museum» свободным текстом даёт 480 objectID, среди первых — настенные
    часы и керамические тарелки («Plate with Water Bird»), потому что Мет
    ищет слово *plate* по описаниям; с `departmentId=4` — 277, и первые
    шесть все до одного латы. На предметном запросе, где отдел и так
    совпадал («medieval rondel dagger»), выдача не потеряла ничего: 17 -> 15,
    те же кинжалы."""
    out = []
    # Фильтры НА СТОРОНЕ ПОИСКА: public domain и окно эпохи канала. Мет их
    # поддерживает, и это меняет цену глубины: без них 867 objectID по
    # «medieval sword museum display», с ними — 450, и каждая из 60 карточек,
    # которые мы тянем, уже не тратится на предмет XIX века или на закрытый
    # правами снимок (замер 13.09). Паспортная проверка НИЖЕ остаётся —
    # серверный фильтр экономит запросы, а не заменяет доказательство.
    window = era_window()

    # ЛОКАЛЬНЫЙ КАТАЛОГ ИДЁТ ПЕРВЫМ, И БЮДЖЕТ КАРТОЧЕК ОТ ЭТОГО НЕ РАСТЁТ.
    #
    # Поиск Мет по `q=` отвечает совпадением по ОПИСАНИЯМ, поэтому из 60
    # вытянутых карточек паспорт проходит около трети — сорок запросов
    # тратятся на предметы, которые будут отброшены. ID из каталога паспорт
    # уже прошли ЛОКАЛЬНО (те же era_overlaps/culture_is_foreign, см.
    # met_catalog.build), то есть выживают все до одного.
    #
    # Отсюда правило: общий потолок `limit` остаётся прежним, каталог просто
    # занимает его начало, а выдача API дозаполняет остаток. Цена в запросах
    # та же, доля полезных карточек выше. Каталога нет или он ничего не нашёл
    # (например, «longsword» — слова нет в словаре Мет, там `Sword` и
    # `Two-hand sword») — путь БАЙТ-В-БАЙТ прежний, ноль регрессии.
    cat_ids = []
    if feature_flags is None or feature_flags.enabled("MET_CATALOG"):
        try:
            import met_catalog
            if met_catalog.available():
                dept_name = MET_DEPARTMENT_NAMES.get(department)
                cat_ids = [int(r["id"]) for r in met_catalog.search(
                    query, department_name=dept_name, limit=limit)
                    if str(r.get("id") or "").isdigit()]
                FETCH_STATS["met_catalog_hits"] += len(cat_ids)
        except Exception:
            cat_ids = []   # fail-open: каталог не обязан существовать

    data = _met_get(f"{MET_API}/search?hasImages=true&isPublicDomain=true"
                    + (f"&dateBegin={int(window[0])}&dateEnd={int(window[1])}" if window else "")
                    + (f"&departmentId={int(department)}" if department else "")
                    + "&q=" + urllib.parse.quote(query))
    api_ids = (data or {}).get("objectIDs") or []
    seen = set(cat_ids)
    oids = cat_ids + [i for i in api_ids if i not in seen]
    oids = oids[:limit]
    if not oids:
        return out

    def _detail(oid):
        # Fail-open ПОКАРТОЧНО, как и было в последовательной версии: упавший
        # запрос одной карточки не должен уносить остальные 59. Потеря
        # считается — иначе поредевший пул выглядел бы бедным корпусом.
        o = _met_card_cached(oid)
        if o is not None:
            return o
        o = _met_get(f"{MET_API}/objects/{oid}")
        if o is None:
            with _MET_LOCK:
                FETCH_STATS["met_cards_lost"] += 1
        else:
            _met_card_store(oid, o)
        return o

    # ex.map сохраняет ПОРЯДОК входа — кандидаты остаются в порядке
    # релевантности поиска Мет, как при последовательном обходе. Это важно:
    # место кандидата в пуле определяет, кого гейты увидят первым (см.
    # чередование по запросам в pipeline_smart.pexels_photo).
    workers = max(1, min(MET_DETAIL_WORKERS, len(oids)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        objects = list(ex.map(_detail, oids))

    for oid, o in zip(oids, objects):
        if not o or not o.get("isPublicDomain"):
            continue
        # Полноразмерный снимок, а не web-версия: замер — 1.4-2 МБ против
        # 55-84 КБ (495x624). Кадр рендерится в 1920x1080 И проходит зум
        # Ken Burns, то есть кроп внутри исходника — на web-версии это
        # гарантированное мыло.
        img = o.get("primaryImage") or o.get("primaryImageSmall")
        if not img:
            continue
        if not era_overlaps(o.get("objectBeginDate"), o.get("objectEndDate")):
            continue
        if culture_is_foreign(o.get("culture"), o.get("country"),
                              o.get("title"), o.get("classification")):
            continue
        out.append(_candidate(
            f"met:{oid}", o.get("title"), img, o.get("objectURL"),
            {"source": "met", "begin": o.get("objectBeginDate"),
             "end": o.get("objectEndDate"), "culture": o.get("culture"),
             "country": o.get("country"), "department": o.get("department"),
             # Признак, ПО КОТОРОМУ предмет сюда попал (isPublicDomain /
             # share_license_status=="CC0" / is_public_domain). Журнал
             # лицензий без самой лицензии наполовину бесполезен.
             "license": "public_domain", "license_field": "isPublicDomain"},
            thumb_url=o.get("primaryImageSmall")))
    return out


def search_cleveland(query, limit=SEARCH_PAGE_SIZE):
    """Кливлендский музей: полная карточка приходит прямо в поиске — один
    запрос на весь список, без похода за каждым предметом."""
    out = []
    data = _get_json(f"{CLEVELAND_API}/?q=" + urllib.parse.quote(query) +
                     f"&has_image=1&limit={int(limit)}")
    for a in data.get("data") or []:
        if (a.get("share_license_status") or "").upper() != "CC0":
            continue
        # print (≈2267x3400 jpeg) вместо web (600x900): тот же довод, что у
        # Мет. Вариант "full" сознательно не берётся — это TIFF на 60+ МБ.
        images = a.get("images") or {}
        img = ((images.get("print") or {}).get("url") or
               (images.get("web") or {}).get("url"))
        thumb = (images.get("web") or {}).get("url")
        if not img:
            continue
        if not era_overlaps(a.get("creation_date_earliest"),
                            a.get("creation_date_latest")):
            continue
        culture = a.get("culture")
        if isinstance(culture, list):
            culture = "; ".join(str(c) for c in culture)
        if culture_is_foreign(culture, a.get("title")):
            continue
        out.append(_candidate(
            f"cleveland:{a.get('id')}", a.get("title"), img, a.get("url"),
            {"source": "cleveland", "begin": a.get("creation_date_earliest"),
             "end": a.get("creation_date_latest"), "culture": culture,
             "license": "cc0", "license_field": "share_license_status"},
            thumb_url=thumb))
    return out


def search_chicago(query, limit=SEARCH_PAGE_SIZE):
    """Институт искусств Чикаго: карточка тоже приходит в поиске, картинка
    собирается по IIIF-шаблону из ответа."""
    out = []
    fields = ("id,title,date_start,date_end,place_of_origin,"
              "is_public_domain,image_id")
    data = _get_json(f"{CHICAGO_API}?q=" + urllib.parse.quote(query) +
                     f"&limit={int(limit)}&fields={fields}")
    iiif = (data.get("config") or {}).get("iiif_url")
    if not iiif:
        return out
    for a in data.get("data") or []:
        if not a.get("is_public_domain") or not a.get("image_id"):
            continue
        if not era_overlaps(a.get("date_start"), a.get("date_end")):
            continue
        if culture_is_foreign(a.get("place_of_origin"), a.get("title")):
            continue
        out.append(_candidate(
            f"chicago:{a.get('id')}", a.get("title"),
            # IIIF отдаёт любую ширину: просим 1920 под финальный кадр, а не
            # дефолтные 843 из примеров документации.
            f"{iiif}/{a['image_id']}/full/1920,/0/default.jpg",
            f"https://www.artic.edu/artworks/{a.get('id')}",
            {"source": "chicago", "begin": a.get("date_start"),
             "end": a.get("date_end"), "place": a.get("place_of_origin"),
             "license": "public_domain", "license_field": "is_public_domain"},
            headers=CHICAGO_IMAGE_HEADERS,
            # Превью для оценки — IIIF отдаёт любую ширину; 400 px хватает
            # CLIP/эстетике, полный 1920 качается только у победителя.
            thumb_url=f"{iiif}/{a['image_id']}/full/400,/0/default.jpg"))
    return out


def _sources(department=None):
    """Источники, собираемые В МОМЕНТ ВЫЗОВА, а не при импорте.

    Раньше это был кортеж-константа со ССЫЛКАМИ на функции — и это не мелочь
    стиля: подменить источник (тестом, монкипатчем, будущей заменой
    реализации) было невозможно, вызов всё равно уходил в объект, захваченный
    при импорте. В тесте это выглядело особенно скверно: патч «пусть Мет
    падает» молча не применялся, и тест вместо изоляции уходил в ЖИВОЙ API
    (поймано 13.09). Тот же принцип, что у реестра флагов: значение читается
    в момент вызова.

    Имена функций стоят здесь ЯВНО, а не собираются через globals() по
    строке: строковый резолв работает точно так же, но делает источники
    невидимыми для статического анализа — первая версия этой правки ровно
    так и уронила tests/test_no_dead_layers.py, объявив search_chicago/
    search_cleveland мёртвыми. Позднее связывание не обязано стоить
    проверяемости.
    """
    met = (lambda q, **kw: search_met(q, department=department, **kw)) if department else search_met
    return (("met", met), ("cleveland", search_cleveland),
            ("chicago", search_chicago))


# --- ДИСКОВЫЙ КЭШ ПОИСКА ------------------------------------------------------
# Музейный поиск — самая дорогая по запросам часть пула (до 60 карточек Мет на
# запрос), а его результат не зависит от эпизода: «medieval rondel dagger»
# в Мете один и тот же для всех роликов канала. Кэш на процесс (выше) держал
# цену в пределах одного прогона; дисковый — переносит её между прогонами и
# между эпизодами. Ключ включает всё, от чего зависит ответ: запрос, глубину,
# окно эпохи и список чужих культур из профиля — смена любого из них даёт
# другой файл, а не устаревший ответ под старым именем (тот же принцип, что у
# candidate_gate_signature()). TTL — коллекции меняются медленно, месяц.
#
# В кэш попадают ТОЛЬКО полные ответы: если в ходе запроса Мет ушёл в
# остывание или какой-то музей упал, результат неполный, и заморозить его на
# месяц значило бы превратить временный отказ в постоянную дыру.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSEUM_CACHE_DIR = os.environ.get("MUSEUM_CACHE_DIR") or os.path.join(_REPO_ROOT, "temp_museum_cache")
MUSEUM_CACHE_TTL_SEC = 30 * 86400
MUSEUM_CACHE_SCHEMA = 1


_MAPPING_SIG = [None]


def _mapping_signature():
    """Подпись кода, который СОБИРАЕТ кандидата (см. тот же дефект у
    Openverse-кэша в pipeline_smart._openverse_mapping_signature): кэш
    хранит готовых кандидатов, и правка полей — превью, разрешения,
    заголовков — обязана менять ключ, иначе месяц отдаётся старая форма."""
    if _MAPPING_SIG[0] is None:
        try:
            import inspect
            src = "".join(inspect.getsource(fn) for fn in
                          (_candidate, search_met, search_cleveland, search_chicago))
        except Exception:
            src = "unavailable"
        _MAPPING_SIG[0] = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
    return _MAPPING_SIG[0]


def _disk_cache_key(query, department=None, limit=None):
    # Отдел ОБЯЗАН входить в ключ: тот же запрос со структурным сужением и
    # без него даёт разные списки, и общий ключ молча отдавал бы чужой.
    # Глубина ОБЯЗАНА входить в ключ: укороченная выдача широкой формулировки
    # и полная выдача точной — разные списки, общий ключ молча отдал бы чужой.
    payload = json.dumps([MUSEUM_CACHE_SCHEMA, query,
                          MET_MAX_DETAIL_FETCHES if limit is None else limit,
                          SEARCH_PAGE_SIZE if limit is None else limit,
                          list(era_window() or ()), sorted(foreign_culture_terms()),
                          [n for n, _ in _sources()], department, _mapping_signature()],
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _disk_cache_path(query, department=None, limit=None):
    return os.path.join(MUSEUM_CACHE_DIR, _disk_cache_key(query, department, limit) + ".json")


def _disk_cache_get(query, department=None, limit=None):
    try:
        path = _disk_cache_path(query, department, limit)
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > MUSEUM_CACHE_TTL_SEC:
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("query") != query or not isinstance(data.get("results"), list):
            return None
        return data["results"]
    except Exception:
        return None   # битый файл — как будто его нет


def _disk_cache_put(query, results, department=None, limit=None):
    try:
        os.makedirs(MUSEUM_CACHE_DIR, exist_ok=True)
        path = _disk_cache_path(query, department, limit)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"query": query, "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "results": results}, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass   # кэш — ускорение, не условие корректности


def search_museums(query, department=None, limit=None):
    """Кандидаты из всех трёх музеев, уже отфильтрованные по эпохе и культуре.

    Порядок источников фиксирован (Met первым — у него профильная коллекция
    оружия и доспеха), но победителя по-прежнему выбирают общие гейты и
    скоринг: этот модуль только приносит кандидатов в пул.
    """
    if feature_flags is not None and not feature_flags.enabled("MUSEUM_SOURCES_ENABLED"):
        return []
    mem_key = (query, department, limit)
    if mem_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[mem_key]
    cached = _disk_cache_get(query, department, limit)
    if cached is not None:
        FETCH_STATS["search_cache_hits"] += 1
        _SEARCH_CACHE[mem_key] = cached
        return cached
    FETCH_STATS["search_cache_misses"] += 1
    per_museum = []
    errors = 0
    cooldowns_before = FETCH_STATS["met_cooldowns"]
    was_cooling = met_is_cooling_down()
    for name, fn in _sources(department):
        try:
            per_museum.append(list(fn(query) if limit is None else fn(query, limit=limit)))
        except Exception:
            # Fail-open ПОИСТОЧНИКОВО: упавший музей не должен уносить с
            # собой два оставшихся и уж тем более ронять слот.
            errors += 1
            continue
    # ЧЕРЕДОВАНИЕ музеев, а не «весь Мет, потом Кливленд, потом Чикаго».
    # Измеренная причина (A/B, 13.09): с глубиной Мет 60 кливлендский
    # «Tilting Suit» (relevance 0.325, лучший кандидат слота «medieval plate
    # armour museum») оказывался 61-м в списке и не попадал в пробную
    # выборку из 20 — побеждала керамическая тарелка Мет (0.24). Порядок
    # ВНУТРИ музея сохранён (его собственная релевантность поиска).
    out = []
    for row in itertools.zip_longest(*per_museum):
        for c in row:
            if c is not None:
                out.append(c)
    _SEARCH_CACHE[mem_key] = out
    complete = (errors == 0 and not was_cooling
                and FETCH_STATS["met_cooldowns"] == cooldowns_before)
    if out and complete:
        _disk_cache_put(query, out, department, limit)
    return out
