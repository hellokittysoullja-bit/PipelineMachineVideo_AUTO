#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Автоматическое определение ниши/мира кадра ПО ТЕКСТУ сценария — один
раз за эпизод, ДО подбора медиа, без человека.

ЗАЧЕМ ЭТОТ МОДУЛЬ, А НЕ ПРАВКА channel_profile.json. shot_domain,
content_alt_blocklist, query_era_anchors, era_window — всё это в
channel_profile.json авторское, задаётся ЧЕЛОВЕКОМ один раз на канал
(CLAUDE.md ЧАСТЬ 24: "клонирование НЕ должно молча тащить чужую нишу").
Для канала с одной зафиксированной нишей (этот репозиторий — военная
история) это ровно то, что нужно, и трогать не надо.

Но задача "тем же генератором сделать идеальный ролик на ЛЮБУЮ, в том
числе никогда не виденную тему" человеком-конфигуратором не решается —
он должен заранее знать нишу и прописать её руками, а по условию задачи
темы не существовало никогда. У пайплайна уже есть всё нужное для этого
БЕЗ человека: полный текст сценария существует раньше первого запроса к
стоку, и мозг, который его прочтёт, — тот же самый, что уже читает главы
целиком в shot_brief_director.py (сессия Claude Code, которая и
пишет/утверждает сценарий, или локальная модель по тому же харнессу для
безсессионного/офлайн прогона).

ОДИН ВЫЗОВ НА ВЕСЬ ЭПИЗОД, НЕ СУДЬЯ НА КАЖДОЕ ФОТО. Прямое требование
задачи — не тратить время: не "модель долго смотрит каждую картинку", а
один недорогой проход по ВСЕМУ тексту сценария, результат которого
меняет, ЧТО запрашивается и ЧТО отбраковывается, — а дешёвые, уже
существующие механизмы (блоклист, era-anchors, маршрутизация источников,
"человек в кадре своей эпохи") получают точный словарь вместо чужого или
пустого. Тот же принцип разделения, что уже работает у
ambience_director.py (одно вето на главу, не судья на каждый звук) и у
музейного паспорта (знание из каталога заранее, не угадывание по
пикселю постфактум).

АДДИТИВНО И FAIL-OPEN — тот же инвариант, что у любого автоматического
слоя в этом репозитории (Openverse/Pixabay/каталог Мет/визуальная полка):
  * media_plan/content_world.json нет, битый или confidence ниже порога —
    эффективный профиль БАЙТ-В-БАЙТ равен channel_profile.json, как до
    этого модуля. Не уверен — молчи целиком, а не частично (тот же
    принцип, что у confidence-gate reveal_hold в pause_intelligence.py).
  * Списки (блоклист, era anchors, openverse-словари) только
    ДОПОЛНЯЮТСЯ union'ом, никогда не заменяют то, что задал человек на
    уровне канала.
  * shot_domain/era_window/use_museum_sources — канал побеждает, если уже
    задал их явно (устоявшаяся ниша — осознанный выбор человека); если
    канал их не задавал вовсе, используется авто-инференс ЭТОГО эпизода.

ЕДИНАЯ ТОЧКА ВХОДА. pipeline_smart.py, shot_types.py и museum_sources.py
до этого модуля читали channel_profile.json каждый СВОИМ независимым
`_profile()` — три копии одной и той же логики "прочитать JSON, вернуть
{} при ошибке". Заводить в каждой из трёх ЕЩЁ и свою копию слияния с
content_world.json значило бы повторить тот класс бага, что уже дважды
подтверждён в этом репозитории (CONTENT_ALT_BLOCKLIST/
_CONTENT_ALT_BLOCKLIST_DEFAULT молчаливо расходились; дубль
sound_library.library_files/pipeline_smart.library_sounds). Поэтому
`effective_profile()` здесь — ОДНА функция, которую зовут все три места.
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import script_parser

# Как pipeline_smart.py сообщает остальным (не связанным с ним напрямую)
# модулям, какой эпизод сейчас в работе — shot_types.py/museum_sources.py
# не получают video_dir параметром (они библиотечные, без понятия "текущий
# эпизод"), а VIDEO_FOLDER у pipeline_smart.py уже известен на момент
# импорта. Тот же класс сигналинга через окружение, что и у остальных
# межмодульных договорённостей этого пайплайна (CHANNEL_ID, TTS_MODEL и т.п.).
ENV_VIDEO_DIR = "PIPELINE_VIDEO_DIR"

CONTENT_WORLD_RELPATH = os.path.join("media_plan", "content_world.json")

# Ниже этой уверенности профиль игнорируется ЦЕЛИКОМ в merge_content_world()
# — см. docstring модуля. Не измеренный оптимум (нет живого корпуса ответов
# модели на разные ниши), консервативная граница: 0.5 значит "модель сама
# не уверена, что тема ей понятна", и в этом случае канальный дефолт
# безопаснее догадки.
MIN_CONFIDENCE = 0.5

# ЖИВОЙ ПРОГОН НА tests/fixtures/other_niche/script_psychology.txt (реальный
# психологический сценарий) НАШЁЛ дефект в первой версии этой логики: у
# ЭТОГО репозитория (военно-историческый канал) shot_domain уже задан, и
# правило "канал побеждает, если задал явно" душило автоматику именно там,
# где задача прямо требует автономии — на теме, которую канал никогда не
# настраивал. Живая проверка (см. коммит): domain_contract() для
# психологического эпизода печатал "МИР КАДРА: европейское Средневековье"
# несмотря на confidence=0.92 у авто-профиля.
#
# Решение — не "кто угодно побеждает", а РАЗНЫЙ порог доверия для
# ДОПОЛНЕНИЯ и для ПЕРЕЗАПИСИ: дополнить пустое (нет ни одного шанса
# отнять уже настроенное) достаточно MIN_CONFIDENCE, а переписать то, что
# человек ЯВНО задал на уровне канала, — только когда модель уверена
# гораздо сильнее, что тема другая (не шум на пограничном сценарии).
# Тот же принцип, что у cooldown/confidence-gate в pause_intelligence.py:
# высокий порог — не тормоз автономии, а защита уже принятого решения от
# случайной догадки.
WORLD_OVERRIDE_MIN_CONFIDENCE = 0.75

# Суффикс, под которым merge_content_world() кладёт списочные добавки —
# ОТДЕЛЬНЫМ ключом, не в тот же ключ, что использует channel_profile.json
# (см. docstring merge_content_world() и merged_list() ниже — реальная
# найденная дыра, не перестраховка).
ADDITIONS_SUFFIX = "_additions"

# Поля профиля, которые ДОПОЛНЯЮТ (не заменяют) списки channel_profile.json.
# (ключ ответа модели -> ключ channel_profile.json)
_LIST_FIELDS_ADDITIVE = (
    ("blocklist_additions", "content_alt_blocklist"),
    ("query_era_anchors_additions", "query_era_anchors"),
    ("openverse_era_anchors_additions", "openverse_era_anchors"),
    ("openverse_domain_nouns_additions", "openverse_domain_nouns"),
    # НЕ словарь-по-подстроке, как остальные пять строк выше. Ловушки
    # negative_anchor_violation() сравниваются с кандидатом ЭМБЕДДИНГОМ
    # (CLIP image-vs-text margin, pipeline_smart.negative_anchor_violation),
    # то есть судят СМЫСЛ картинки, а не совпадение слова в alt/слаге —
    # ровно тот механизм, который уже в репозитории отвечает за «современное
    # вторжение в историческую сцену» без единого ключевого слова. Для новой
    # ниши это самое важное поле из всех: словарный блоклист выше — дешёвая
    # подстраховка ДО скачивания, а этот список — единственное место, где
    # авто-профиль реально дотягивается до СЕМАНТИЧЕСКОГО судьи, а не до
    # текстового фильтра.
    ("negative_anchor_additions", "content_negative_anchors"),
)

# Поля, которые ПОБЕЖДАЮТ только если канал их не задал вовсе (см.
# docstring — устоявшаяся ниша канала не переписывается догадкой по
# одному эпизоду). "is_historical" — не литеральный ключ channel_profile.json
# (канал никогда не объявляет его руками, в отличие от shot_domain/
# era_from) — это чистый вывод content_world.py, поэтому для него
# достаточно MIN_CONFIDENCE, а не более строгого WORLD_OVERRIDE_MIN_
# CONFIDENCE: здесь нет "явного решения человека", которое можно было бы
# тихо переписать — см. его использование в historical_default() ниже.
_SCALAR_FIELDS_IF_ABSENT = ("era_from", "era_to", "use_museum_sources", "is_historical")

# СПИСКИ, КОТОРЫЕ АВТО-НИША ЗАПОЛНЯЕТ ТОЛЬКО ПУСТЫМИ (17.09). Не добавки к
# канальным, а именно заполнение молчания — и это не осторожность, а
# измеренная необходимость с обеих сторон.
#
# ЗАЧЕМ. `brief_to_stock_query()` ставит якорь эпохи в КАЖДЫЙ запрос без
# своего (CLAUDE.md: «0 запросов из 142 без якоря эпохи»), а берёт якоря
# из `openverse_era_anchors`. После того как дефолты кода обнулены
# (17.09, ни один список ниши больше не дефолт), у канала НОВОЙ ниши этот
# список пуст, и парсер ответа модели его не заполнял: `ANCHOR_WORDS`
# уходили только в `shot_domain`. Итог для эпизода про каменный век —
# «a flint hand axe held in a palm» уходит в сток без единого слова об
# эпохе, а это ровно тот случай, который в этом файле уже записан числом:
# одиночное `plate armour` первым результатом даёт «MkIV-Tank-Plate».
#
# ПОЧЕМУ НЕ ДОБАВКОЙ. Канал, который якоря объявил, откалиброван на них:
# лишние слова означают, что ЧАСТЬ запросов сочтётся «уже с якорем» и
# перестанет его получать — то есть добавление ослабило бы гарантию, а не
# усилило. Поэтому заполняется только пустое; у этого канала (объявлены
# openverse_era_anchors/openverse_domain_nouns/query_era_anchors) поведение
# БАЙТ-В-БАЙТ прежнее.
#
# ANCHOR_WORDS — уже существующее поле промпта («6-12 английских
# предметных/сценовых слов, которые ДЕЙСТВИТЕЛЬНО про эту тему»), то есть
# ровно то, чем этот канал и заполнил три своих списка вручную. Второго
# поля под то же самое не заводится.
_LIST_FIELDS_IF_ABSENT = (
    ("anchor_words", "openverse_era_anchors"),
    ("anchor_words", "openverse_domain_nouns"),
    ("anchor_words", "query_era_anchors"),
)


def content_world_path(video_dir):
    return os.path.join(video_dir, CONTENT_WORLD_RELPATH)


def full_script_text(video_dir, max_chars=None):
    """Весь читаемый текст сценария одной строкой — то, что услышит
    зритель, без служебных секций (=== PEXELS QUERIES ===, === IMAGE
    PROMPTS === и т.п.) и без пайплайн-тегов ([pause]/[shot:...]/[stat:...]).

    Не своя разборка тегов, а `script_parser.parse_blocks()` — тот же
    источник истины, что уже использует весь остальной пайплайн для "что
    реально услышит зритель". Своя копия регекспов по тегам здесь была бы
    ровно тем классом дефекта, что уже дважды стоил эпизоду реальных
    поломок (14.09, [sfx:]/[hush] потерялись во ВТОРОЙ, не синхронной
    копии словаря тегов) — script_parser уже отдаёт чистый text каждого
    блока, отдельно от shot_brief/stat/sfx.

    Ниша определяется по ВСЕМУ тексту, не по фрагменту: половина сценария
    несёт те же риски, что укороченный контекст у пофразового режиссёра
    (shot_brief_director.py, находка 15.09 — "модель любого размера
    промахнётся одинаково, это не потолок модели, это отсутствующий
    вход")."""
    path = os.path.join(video_dir, "script.txt")
    try:
        blocks = script_parser.parse_blocks(path)
    except OSError:
        return ""
    text = " ".join((b.get("text") or "").strip() for b in blocks if b.get("text"))
    text = " ".join(text.split())
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


PROMPT_TEMPLATE = """Ты — режиссёр, который видит сценарий ролика ПЕРЕД тем, как для него подбирают видеоряд. Определи нишу и мир кадра один раз для всего эпизода — дальше подбор картинок/видео не должен тащить чужую тему и не должен тратить время на источники, которые этой теме не подходят.

СЦЕНАРИЙ (полный текст, без пауз и служебных тегов):
\"\"\"
{script}
\"\"\"

Ответь СТРОГО построчно, каждое поле на своей строке, формат "КЛЮЧ: значение". Поле, в котором не уверен, — просто не пиши строку (пустой ответ по полю честнее угаданного).

NICHE: короткая метка ниши (2-5 слов, например "военная история", "психология/самопомощь", "медицина", "true crime", "наука/техника")
IS_HISTORICAL: yes или no — реальный физический предмет из музея/архива определённой эпохи вообще имеет смысл для этой темы?
ERA_FROM: год начала эпохи (число), только если IS_HISTORICAL=yes и эпоха определена сценарием
ERA_TO: год конца эпохи (число), только если IS_HISTORICAL=yes
WORLD: мир кадра одной фразой — где и когда происходит то, о чём говорит сценарий (например: "современный город, повседневная жизнь" или "европейское Средневековье, 900-1600 годы")
PEOPLE_IN_FRAME: как показывать человека в кадре, если он есть (например: "обычный человек, современная одежда, лицо не обязательно" или "рыцарь, воин своего времени, не «a person»")
FORBIDDEN: что визуально НЕЛЬЗЯ показывать, чтобы кадр не выглядел чужой темой/эпохой/жанром (конкретно, через запятую)
ANCHOR_WORDS: 6-12 английских предметных/сценовых слов, которые ДЕЙСТВИТЕЛЬНО про эту тему (через запятую, для англоязычного поиска по стокам)
BLOCKLIST: английские слова/фразы, которые если появятся в подписи к кандидату — почти наверняка означают ЧУЖУЮ тему/эпоху/жанр (через запятую)
NEGATIVE_ANCHORS: 3-6 английских ОПИСАНИЙ СЦЕНЫ (не отдельных слов, а того, что реально было бы видно на неподходящем кадре), которые СРАВНИВАЮТСЯ С КАРТИНКОЙ моделью — как выглядела бы явно чужая эпоха/жанр/культура рядом с этой темой (например, для средневековой темы: "modern city street with cars and asphalt"; для темы о психологии/офисе — эта строка обычно НЕ нужна вообще, современность здесь и есть тема; через точку с запятой)
USE_MUSEUM_SOURCES: yes или no — есть ли смысл искать в музейных каталогах (Метрополитен и т.п.) физические исторические предметы для этой темы?
MOOD_TONE: число от -2 (тяжёлый) до 2 (светлый)
MOOD_TENSION: число от 0 (покой) до 3 (предел)
CONFIDENCE: число от 0 до 1 — насколько ты уверен в этой нише по тексту (низкая уверенность на смешанном/абстрактном сценарии — нормальный, честный ответ, не недостаток)
"""


def build_prompt(video_dir, max_chars=6000):
    return PROMPT_TEMPLATE.format(script=full_script_text(video_dir, max_chars=max_chars))


_LINE_RE = re.compile(r"^\s*([A-Z_]+)\s*:\s*(.*)$")


def _split_list(value):
    return [x.strip() for x in re.split(r"[,;]", value or "") if x.strip()]


def _split_list_semicolon(value):
    """Для полей, где элемент сам — предложение и может содержать запятую
    (NEGATIVE_ANCHORS: "modern city street with cars, asphalt and signage").
    Разделитель ТОЛЬКО ";" — запятая внутри описания не режет список."""
    return [x.strip() for x in (value or "").split(";") if x.strip()]


def _to_float(value):
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _to_bool(value):
    v = (value or "").strip().lower()
    if v in ("yes", "true", "1", "да"):
        return True
    if v in ("no", "false", "0", "нет"):
        return False
    return None


def parse_answer(text):
    """Разбор построчного ответа модели в профиль (dict). Каждая строка —
    свой ключ, ОТДЕЛЬНО от остальных: сорванная/непонятая строка теряет
    ровно одно поле, не весь профиль — тот же принцип, что у построчного
    плана shot_brief_director (никогда один JSON на весь ответ; см. его же
    докстринг про "заявок в главе 8-16, одна сорванная запятая не должна
    ронять главу целиком")."""
    fields = {}
    for line in (text or "").splitlines():
        m = _LINE_RE.match(line)
        if m:
            fields[m.group(1)] = m.group(2).strip()

    out = {}
    if fields.get("NICHE"):
        out["niche"] = fields["NICHE"]

    is_hist = _to_bool(fields.get("IS_HISTORICAL"))
    if is_hist is not None:
        out["is_historical"] = is_hist

    for key_in, key_out in (("ERA_FROM", "era_from"), ("ERA_TO", "era_to")):
        v = _to_float(fields.get(key_in))
        if v is not None:
            out[key_out] = int(v)

    for key_in, key_out in (("WORLD", "world"), ("PEOPLE_IN_FRAME", "people_in_frame"),
                             ("FORBIDDEN", "forbidden")):
        if fields.get(key_in):
            out[key_out] = fields[key_in]

    if fields.get("ANCHOR_WORDS"):
        out["anchor_words"] = _split_list(fields["ANCHOR_WORDS"])
    if fields.get("BLOCKLIST"):
        out["blocklist_additions"] = _split_list(fields["BLOCKLIST"])
    if fields.get("NEGATIVE_ANCHORS"):
        out["negative_anchor_additions"] = _split_list_semicolon(fields["NEGATIVE_ANCHORS"])

    use_museum = _to_bool(fields.get("USE_MUSEUM_SOURCES"))
    if use_museum is not None:
        out["use_museum_sources"] = use_museum

    tone = _to_float(fields.get("MOOD_TONE"))
    if tone is not None:
        out["mood_tone"] = max(-2.0, min(2.0, tone))
    tension = _to_float(fields.get("MOOD_TENSION"))
    if tension is not None:
        out["mood_tension"] = max(0.0, min(3.0, tension))

    conf = _to_float(fields.get("CONFIDENCE"))
    out["confidence"] = max(0.0, min(1.0, conf)) if conf is not None else 0.0
    return out


def _shot_domain_from_profile(profile):
    d = {}
    if profile.get("world"):
        d["world"] = profile["world"]
    if profile.get("people_in_frame"):
        d["people_in_frame"] = profile["people_in_frame"]
    if profile.get("forbidden"):
        d["forbidden"] = profile["forbidden"]
    if profile.get("anchor_words"):
        d["anchor_words"] = list(profile["anchor_words"])
    return d


def write_content_world(video_dir, profile, source):
    """Пишет media_plan/content_world.json. `profile` — результат
    parse_answer() (или собранный вручную сессией/человеком словарь тех же
    полей — формат один и тот же, харнесс не знает, кто отвечал).
    `source` — кто решил ("claude-session" / "local:<model>" /
    "file:<путь>") — аудит-трейл, тот же принцип, что brain.name у
    shot_brief_director: по артефакту эпизода видно, чей это был вывод."""
    path = content_world_path(video_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(profile)
    payload["source"] = source
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


# Потолок длины строки, которая уйдёт в текстовую башню CLIP/SigLIP как
# ловушка вето. Не «на глаз»: у SigLIP2 лимит 64 токена
# (SIGLIP2_MAX_TEXT_LENGTH), у CLIP ViT-B/32 — 77, и обрезка по лимиту
# МОЛЧАЛИВАЯ — этот репозиторий уже измерял, как она тихо съедала 25% фраз
# (см. TEXT_TRUNCATION_REPORT в visual_director.py). Ловушка, обрезанная на
# середине, сравнивается с кадром НЕ тем текстом, который написан, — и
# решает при этом судьбу кандидата. 200 символов — консервативный
# эквивалент ~40-50 слов, заведомо внутри обоих лимитов.
MAX_ANCHOR_CHARS = 200

# Латиница обязательна для всего, что уходит В МОДЕЛЬ или В ТЕКСТОВЫЙ
# ПОИСК: текстовые башни CLIP/SigLIP английские, а alt/слаг стоков —
# латиница. Диагноз не гипотетический, он уже записан в этом репозитории
# дословно для CLAP: «текстовая башня английская, и сырой русский текст
# даёт шум», 7 попаданий из 8 на английском против разброса 0.31 на
# русском. Шум в ЛОВУШКЕ ВЕТО — это не «слой не сработал», это случайные
# отказы законным кандидатам.
_LATIN_RE = re.compile(r"[A-Za-z]")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def _is_model_ready_text(s):
    """Строка годится для английской текстовой башни / текстового поиска."""
    s = (s or "").strip()
    if not s or len(s) > MAX_ANCHOR_CHARS:
        return False
    return bool(_LATIN_RE.search(s)) and not _CYRILLIC_RE.search(s)


def validate_profile(profile):
    """(очищенный_профиль, список_отклонённого) — проверка СОДЕРЖИМОГО
    профиля на границе чтения, до того как оно начнёт решать судьбу
    кандидатов.

    ЗАЧЕМ ИМЕННО ЗДЕСЬ, а не у писателя. Профиль может написать кто
    угодно — CLI с локальной моделью, сессия Claude, человек руками,
    будущий скрипт. Проверка у ОДНОГО из писателей защищает только его;
    проверка на границе чтения защищает пайплайн от всех сразу. Тот же
    урок уже оплачен в этом репозитории: `brief_is_safe()` стоял в
    `fill_briefs()`, а измерительный харнесс звал модель мимо него — и всё,
    что когда-либо было измерено про режиссёра, описывало СЫРУЮ модель.

    ПОЧЕМУ ЭТО НЕ ПРИДИРКИ. Ловушки `negative_anchor_additions` уходят в
    контрастивное вето (`negative_anchor_violation`), а вето — механизм
    ОТКАЗА: каждая добавленная ловушка может только УЖЕСТОЧИТЬ отбор.
    Ловушка-мусор (кириллица в английской башне, обрезанная по лимиту
    токенов строка) не «не сработает» — она даст ШУМ, то есть случайные
    отказы кадрам, которые теме подходят. Это прямой источник брака,
    созданный самим слоем, который от брака защищает.

    ЧЕГО ЗДЕСЬ СОЗНАТЕЛЬНО НЕТ — проверки «ловушка не противоречит самой
    нише». Она напрашивается (запретить кадр, который сам объявлен
    предметом ролика) и НЕ реализуема лексически без ложных отказов:
    у исторической ниши с anchor_words=[armour] законная ловушка
    «modern military body armour» делит слово с предметом ролика, а
    незаконная «medieval armour» — то же самое слово. Разделить их можно
    только по смыслу квалификатора, то есть тем же эмбеддингом, что стоит
    на отборе. Строковая проверка ошибалась бы в обе стороны и сама стала
    бы источником отказов — ровно то, что она должна предотвращать.
    Структурно этот класс уже закрыт с другой стороны: `historical_default()`
    не даёт списку ловушек ЧУЖОЙ ниши применяться вообще."""
    rejected = []
    clean = dict(profile)

    for key in ("negative_anchor_additions", "blocklist_additions",
                "query_era_anchors_additions", "openverse_era_anchors_additions",
                "openverse_domain_nouns_additions", "anchor_words"):
        raw = clean.get(key)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)):
            rejected.append({"field": key, "value": repr(raw)[:80], "reason": "not_a_list"})
            clean.pop(key, None)
            continue
        kept = []
        for item in raw:
            if not isinstance(item, str):
                rejected.append({"field": key, "value": repr(item)[:80], "reason": "not_a_string"})
            elif not _is_model_ready_text(item):
                rejected.append({"field": key, "value": item[:80],
                                  "reason": "not_latin_or_too_long"})
            else:
                kept.append(item.strip())
        if kept:
            clean[key] = kept
        else:
            clean.pop(key, None)

    era_from, era_to = clean.get("era_from"), clean.get("era_to")
    if era_from is not None and era_to is not None:
        try:
            if int(era_from) > int(era_to):
                rejected.append({"field": "era_window", "value": f"{era_from}..{era_to}",
                                  "reason": "from_after_to"})
                clean.pop("era_from", None)
                clean.pop("era_to", None)
        except (TypeError, ValueError):
            rejected.append({"field": "era_window", "value": f"{era_from!r}..{era_to!r}",
                              "reason": "not_numeric"})
            clean.pop("era_from", None)
            clean.pop("era_to", None)

    return clean, rejected


def load_content_world(video_dir):
    """{} если файла нет, битый, не словарь, или confidence ниже
    MIN_CONFIDENCE — fail-open ЦЕЛИКОМ файлом, не по полю (см. docstring
    модуля). Содержимое прошедшего профиля проверяется validate_profile()
    — непригодные элементы отбрасываются ПОИМЁННО (`_rejected` в
    результате), а не молча уезжают в вето."""
    if not video_dir:
        return {}
    try:
        with open(content_world_path(video_dir), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    conf = _to_float(data.get("confidence"))
    if conf is None or conf < MIN_CONFIDENCE:
        return {}
    clean, rejected = validate_profile(data)
    if rejected:
        clean["_rejected"] = rejected
    return clean


def merge_content_world(base_profile, video_dir):
    """base_profile — уже прочитанный channel_profile.json (человеком, на
    уровне канала). Возвращает НОВЫЙ словарь; base_profile не мутируется.

    Нет файла / битый / confidence ниже MIN_CONFIDENCE -> возвращается ТОТ
    ЖЕ объект base_profile без изменений — байт-в-байт прежнее поведение
    (проверено тестом на identity, не только на равенство).

    ДВА порога, не один (см. WORLD_OVERRIDE_MIN_CONFIDENCE выше — найдено
    живым прогоном на психологическом сценарии): дополнить ПУСТОЕ
    (списки — всегда, shot_domain/era/use_museum_sources — когда канал их
    не задавал) достаточно MIN_CONFIDENCE; ПЕРЕЗАПИСАТЬ то, что канал
    задал ЯВНО (устоявшаяся ниша, осознанный выбор человека), — только при
    WORLD_OVERRIDE_MIN_CONFIDENCE.

    СПИСКИ ПИШУТСЯ ПОД ОТДЕЛЬНЫМ ИМЕНЕМ `<key>_additions`, А НЕ В ТОТ ЖЕ
    КЛЮЧ, ЧТО У КАНАЛА — найдено живой проверкой 17.09 на РЕАЛЬНОЙ дыре, не
    предположено. `content_alt_blocklist`/`query_era_anchors`/
    `content_negative_anchors` в pipeline_smart.py читаются как
    `CHANNEL_PROFILE.get(key, КОД_ДЕФОЛТ)` — если бы добавки content_world
    легли прямо в `merged[key]`, при ОТСУТСТВУЮЩЕМ у канала ключе (у этого
    самого репозитория `content_negative_anchors` в channel_profile.json
    просто нет) `.get()` увидел бы "ключ ЕСТЬ" и потерял бы КОД_ДЕФОЛТ
    ЦЕЛИКОМ (8 стандартных ловушек современного вторжения — молча
    заменились бы двумя авто-добавками для нового эпизода). Ровно тот
    класс дефекта, что уже дважды подтверждён в этом репозитории
    (CONTENT_ALT_BLOCKLIST/_CONTENT_ALT_BLOCKLIST_DEFAULT), только в
    собственном коде этого модуля, а не в чужом. Финальное объединение
    "канал-или-код-дефолт + добавки" делает `merged_list()` ниже, в точке
    потребления, которая одна знает свой код-дефолт."""
    cw = load_content_world(video_dir)
    if not cw:
        return base_profile

    merged = dict(base_profile)
    confident_enough_to_override = cw.get("confidence", 0.0) >= WORLD_OVERRIDE_MIN_CONFIDENCE

    if not merged.get("shot_domain") or confident_enough_to_override:
        sd = _shot_domain_from_profile(cw)
        if sd:
            merged["shot_domain"] = sd
            # ЧЕЙ это мир — не косметика: мир ЭТОГО эпизода уместен всегда,
            # а мир КАНАЛА на эпизоде из другой ниши диктует чужую тему
            # (см. shot_domain_for_prompt — воспроизведено живьём).
            merged["shot_domain_source"] = "content_world"

    for cw_key, profile_key in _LIST_FIELDS_ADDITIVE:
        additions = cw.get(cw_key)
        if additions:
            merged[profile_key + ADDITIONS_SUFFIX] = list(additions)

    for key in _SCALAR_FIELDS_IF_ABSENT:
        if key in cw and (key not in merged or confident_enough_to_override):
            merged[key] = cw[key]

    # Пустое заполняется всегда (MIN_CONFIDENCE); объявленное каналом
    # ЗАМЕНЯЕТСЯ — не дополняется — только при высокой уверенности
    # (WORLD_OVERRIDE_MIN_CONFIDENCE), той же, что у shot_domain выше.
    #
    # ПЕРВАЯ ВЕРСИЯ ЭТОГО ПРАВИЛА («канал побеждает всегда, даже при
    # высокой уверенности») ПРОВЕРЕНА ЖИВЫМ ПРОГОНОМ НА НАСТОЯЩЕМ,
    # УЖЕ НАСТРОЕННОМ КАНАЛЕ 17.09 И ПРОВАЛИЛАСЬ — тот же класс промаха,
    # что до появления WORLD_OVERRIDE_MIN_CONFIDENCE ловился на
    # shot_domain. У этого (военно-исторического) репозитория
    # `openverse_era_anchors` уже объявлены (9 средневековых слов), а
    # `brief_to_stock_query()` ставит ОДИН из них В КАЖДЫЙ стоковый
    # запрос без своего якоря. Живой тестовый эпизод про историю пиццы
    # (content_world confidence=0.92, свой мир «Неаполь 18-20 века»,
    # `use_museum_sources=False` — все ЭТИ поля корректно переписались)
    # получил дословно:
    #     'a ripe red tomato on a rustic table' -> 'medieval ripe red tomato rustic'
    #     'a wood-fired brick oven...'          -> 'medieval wood-fired brick oven glowing'
    # То есть три соседних поля override сработали правильно, а этот
    # список — нет, и результат ушёл бы в РЕАЛЬНЫЙ вызов Pexels/Openverse
    # с якорем чужой эпохи. Прежнее обоснование («добавка к
    # откалиброванному списку ослабляет гарантию») относилось к
    # ДОБАВЛЕНИЮ эпизодных слов В список канала — оно верно и никуда не
    # делось (см. `_LIST_FIELDS_ADDITIVE` выше, те списки по-прежнему
    # только дополняются). Здесь другой случай: список решает, каким
    # ЯКОРЕМ ЭПОХИ подписывать запросы ЭТОГО эпизода, а не что канал
    # считает своей нишей вообще, — при высокой уверенности мир эпизода
    # ПОЛНОСТЬЮ ЗАМЕНЯЕТ канальный (та же логика, что у shot_domain),
    # а не смешивается с ним: смешение вернуло бы ровно ту порчу запроса,
    # ради которой список остаётся невставляемым в `_LIST_FIELDS_ADDITIVE`.
    for cw_key, profile_key in _LIST_FIELDS_IF_ABSENT:
        anchors = cw.get(cw_key)
        if not anchors:
            continue
        if not merged.get(profile_key) or confident_enough_to_override:
            merged[profile_key] = list(anchors)

    return merged


def merged_list(profile, key, code_default=()):
    """channel-явно-заданный-список-ИЛИ-код-дефолт + аддитивные добавки
    content_world (см. docstring merge_content_world выше про то, зачем
    это отдельная функция, а не прямая запись в `merged[key]`). Без
    дублей, регистронезависимо. Единая точка для ВСЕХ мест пайплайна, что
    сегодня читают список через `CHANNEL_PROFILE.get(key, КОД_ДЕФОЛТ)` —
    вторая копия этого объединения в другом файле рано или поздно
    разойдётся (тот же класс, что уже дважды ловился в этом репозитории)."""
    base = list(profile.get(key, code_default) or [])
    additions = profile.get(key + ADDITIONS_SUFFIX) or []
    if not additions:
        return tuple(base)
    seen = {str(x).lower() for x in base}
    return tuple(base) + tuple(x for x in additions if str(x).lower() not in seen)


def channel_declares_history(profile):
    """Канал ОБЪЯВИЛ СЕБЯ историческим — только по однозначным признакам
    конфига: окно эпохи или якоря эпохи.

    Наличие `shot_domain` само по себе признаком НЕ является, и это не
    придирка: `shot_domain` объявляет и канал про психологию — просто мир
    там современный. Считать «мир задан» за «мир исторический» значило бы
    записать в историки любой настроенный канал, то есть вернуть ту же
    подмену, от которой уходим."""
    return bool(profile.get("era_from") or profile.get("era_to")
                or profile.get("query_era_anchors")
                or profile.get("openverse_era_anchors"))


def resolve_is_historical(profile):
    """True/False/None. Порядок: явный вывод content_world ЭТОГО эпизода
    (см. `is_historical` в `_SCALAR_FIELDS_IF_ABSENT` выше) -> канал объявил
    себя историческим (channel_declares_history) -> None, если сигнала нет
    вообще."""
    v = profile.get("is_historical")
    if v is not None:
        return bool(v)
    return True if channel_declares_history(profile) else None


def shot_domain_for_prompt(profile):
    """МИР КАДРА, который можно показывать мозгу, пишущему брифы, — или {}.

    РЕАЛЬНО ВОСПРОИЗВЕДЁННЫЙ СЛУЧАЙ (17.09, живой прогон на
    tests/fixtures/other_niche/script_psychology.txt в ЭТОМ репозитории):
    авто-ниша уверенно определила «психология / самопомощь»,
    is_historical=False, музеи выключены, ловушки вето подменены на
    корректные — а `shot_brief_director.domain_contract()` всё равно выдал
    мозгу «МИР КАДРА: европейское Средневековье… В кадре не должно быть:
    современная одежда, техника, снаряжение и интерьеры». То есть сценарию
    про человека за ноутбуком ПРЯМЫМ ТЕКСТОМ запрещалось показывать
    ноутбук. Это не «слой не помог» — это слой, который диктует чужую нишу,
    ровно тот класс, который CLAUDE.md называет самой опасной находкой
    («ЧУЖОЙ МИР КАНАЛА НЕ ФИЛЬТРУЕТ, А ДИКТУЕТ»), и до этой функции он
    закрывался только тем, что человек вспомнит выставить
    SHOT_BRIEF_WORLD=off руками.

    Правило: мир, пришедший ОТ САМОГО ЭПИЗОДА (`shot_domain_source ==
    "content_world"`), уместен всегда. Мир КАНАЛА подавляется, только
    когда он заведомо про другое: эпизод определён как нехисторический, а
    канал объявил себя историческим. Канал про современную нишу свой
    собственный современный мир при этом сохраняет — именно поэтому
    признаком историчности канала служит окно/якоря эпохи, а не сам факт
    наличия мира (см. channel_declares_history)."""
    sd = profile.get("shot_domain") or {}
    if not sd:
        return {}
    if profile.get("shot_domain_source") == "content_world":
        return sd
    if profile.get("is_historical") is False and channel_declares_history(profile):
        return {}
    return sd


def historical_default(profile, historical_value, other_value=()):
    """ДЕФОЛТ-ЗНАЧЕНИЕ, ЗАВИСЯЩЕЕ ОТ НИШИ, А НЕ УНИВЕРСАЛЬНЫЙ КОД-ДЕФОЛТ.

    Найдено 17.09 прямой проверкой: `_CONTENT_NEGATIVE_ANCHORS_DEFAULT` в
    pipeline_smart.py — восемь ловушек "современного вторжения" (толпа
    зрителей, современная кухня, городская улица), — при поверхностном
    взгляде выглядят как безопасный универсальный дефолт, а на деле
    ЦЕЛИКОМ кодируют допущение "этот канал исторический, современность —
    анахронизм". Живая проверка на психологическом сценарии в этом же
    репозитории: кандидат для фразы про пустой холодильник и заброшенный
    завтрак получает ВЫСОКОЕ сходство с ловушкой "modern domestic interior,
    kitchen, plastic and household objects" — и контрастивное вето
    (`negative_anchor_violation`) может ОТКЛОНИТЬ ровно тот кадр, который
    этой теме и нужен, потому что список ловушек калиброван для чужой
    ниши. Это не "дефолт, который можно расширить руками" — использование
    списка САМО ПО СЕБЕ неверно для темы, где современность не анахронизм.

    Поэтому такой список — не КОД_ДЕФОЛТ в обычном смысле (`_OPENVERSE_
    ERA_ANCHORS_DEFAULT`/`_OPENVERSE_DOMAIN_NOUNS_DEFAULT` уже пусты по
    коду ровно по этой причине, см. их докстринг), а ЗНАЧЕНИЕ ДЛЯ
    ИСТОРИЧЕСКОЙ НИШИ, применяемое только когда есть основание думать, что
    ниша историческая (см. resolve_is_historical). Нет сигнала вообще
    (совсем новый канал, содержательного профиля ни у канала, ни у
    content_world нет) -> исторический вариант остаётся ПРЕЖНИМ дефолтом
    (байт-в-байт поведение для всех уже настроенных каналов, включая этот
    репозиторий, где shot_domain задан явно) — но НЕ ПОТОМУ, что решили
    рискнуть, а потому что у существующих каналов сигнал `shot_domain` уже
    есть и однозначно указывает на историчность."""
    is_hist = resolve_is_historical(profile)
    if is_hist is False:
        return other_value
    return historical_value


def _base_channel_profile():
    path = os.path.join(REPO, "channel_profile.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def effective_profile(video_dir=None):
    """channel_profile.json канала + авто-профиль ЭТОГО эпизода (если есть
    и достаточно уверенный). Единая точка входа для ВСЕХ мест, которые
    сегодня читают channel_profile.json (см. docstring модуля).

    video_dir=None -> берётся из окружения (ENV_VIDEO_DIR) — так его видят
    shot_types.py/museum_sources.py, у которых нет параметра "текущий
    эпизод" в собственных функциях."""
    video_dir = video_dir or os.environ.get(ENV_VIDEO_DIR)
    return merge_content_world(_base_channel_profile(), video_dir)


def run_local(video_dir, model_path=None, threads=4):
    """Один вызов локальной модели (тот же харнесс, что у
    shot_brief_director.LocalBrain — не второй ML-харнесс, а импорт
    существующего). Для офлайн/безсессионного прогона; внутри живой
    сессии Claude Code нет смысла звать подпроцесс — сессия сама читает
    сценарий и пишет ответ (см. `--brain packets`/`--brain file`)."""
    import shot_brief_director as sbd
    model = sbd.find_model(model_path)
    if not model:
        return None, "no_model"
    brain = sbd.LocalBrain(model, n_threads=threads, max_tokens=600)
    answer = brain.ask(build_prompt(video_dir), 0)
    return parse_answer(answer), "local:" + brain.name


def main(argv):
    ap = argparse.ArgumentParser(
        description="Авто-определение ниши/мира кадра эпизода по тексту "
                     "сценария — один раз, до подбора медиа")
    ap.add_argument("video_dir")
    ap.add_argument("--brain", choices=("local", "packets", "file"), default="local",
                    help="local — модель ищется сама (--model, LLAMA_MODEL_GGUF, "
                         "models/*.gguf); packets/file — тот же поток, что у "
                         "shot_brief_director: сначала --brain packets печатает "
                         "промпт, ответ кладётся файлом, затем --brain file --answer <файл>")
    ap.add_argument("--answer", help="файл с готовым ответом (--brain file)")
    ap.add_argument("--out-packet", help="куда положить промпт (--brain packets)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv[1:])

    if a.brain == "packets":
        dest = a.out_packet or os.path.join(a.video_dir, "media_plan", "content_world_packet.txt")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(build_prompt(a.video_dir))
        print(f"Пакет: {dest}\nОтвет положить файлом и запустить "
              f"--brain file --answer <файл>")
        return 0

    if a.brain == "file":
        if not a.answer:
            print("Нужен --answer <файл с ответом>")
            return 2
        with open(a.answer, encoding="utf-8") as f:
            text = f.read()
        profile = parse_answer(text)
        source = "file:" + os.path.basename(a.answer)
    else:
        profile, source = run_local(a.video_dir, a.model, a.threads)
        if profile is None:
            print("Модели нет. Поставить: python scripts/setup_local_director.py\n"
                  "Или --brain packets — заполнить ответ прямо в сессии Claude Code.")
            return 2

    path = write_content_world(a.video_dir, profile, source)
    conf = profile.get("confidence", 0.0)
    if conf < MIN_CONFIDENCE:
        print(f"Уверенность {conf:.2f} ниже порога {MIN_CONFIDENCE} — профиль "
              f"записан, но эффективным не станет (merge_content_world "
              f"проигнорирует его целиком, канал останется прежним).")
    print(f"Ниша: {profile.get('niche', '?')} "
          f"(историчность: {profile.get('is_historical')}, "
          f"музеи: {profile.get('use_museum_sources')}, confidence: {conf:.2f})")
    print(f"Записано: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
