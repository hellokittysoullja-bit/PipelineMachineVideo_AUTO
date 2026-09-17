#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Библиотека НАСТОЯЩИХ звуков: полевые записи и эффекты под CC0, отобранные
двумя моделями и измерительными гейтами. Заменяет процедурный синтез там,
где он звучал как «гул вентилятора» (прямая жалоба владельца 13.09).

ОТКУДА. Openverse Audio API (без ключа) индексирует Freesound и Wikimedia
Commons; берётся ТОЛЬКО `license=cc0` — и в запросе, и повторно на каждом
результате (fail-closed, тот же принцип, что у картинок из Openverse).
CC0 — юридический отказ от прав, атрибуция не нужна; манифест всё равно
пишется (assets/library/manifest.json). При FREESOUND_API_KEY в .env
дополнительно идёт прямой поиск Freesound с фильтром лицензии — его индекс
полнее. Файлы — hq-превью Freesound (MP3 128 kbps): оригинальный WAV
Freesound отдаёт только по OAuth-авторизации пользователя, это не «ключ в
.env». Под закадром на -24..-40 LU ниже голоса разницы нет.

КАК ОТБИРАЕТСЯ — то же устройство, что у подбора картинок, перенесённое на
звук:
  1. Измерительные гейты (ffmpeg): длительность, доля тишины, диапазон
     громкости (LRA — «события» в фоне лезут из-под голоса), клиппинг,
     сетевой гул 50/60 Гц.
  2. CLAP (laion/larger_clap_general) — текст<->звук, как CLIP для картинок:
     положительный промпт против списка ловушек-негативов (речь, музыка,
     транспорт, гул, дисторшн). Считается на НЕСКОЛЬКИХ окнах записи
     (15/50/85%) — тот же урок, что video_negative_anchor_violation: брак
     в одном месте записи не виден в другом. Решает МАРЖА на одной записи,
     не абсолютный скор (сырой косинус между разными текстами несопоставим —
     см. историю негативного вето по картинкам).
  3. AST (MIT/ast-finetuned-audioset) — второй, независимый судья, как Jina
     рядом с SigLIP2: вероятности классов AudioSet «Speech»/«Music»/
     «Vehicle» — жёсткое вето на атмосферу с голосами, музыкой, машинами.

ЧТО НА ВЫХОДЕ. assets/library/ambience/<вид>/*.flac (48 кГц, стерео, пик
-12 dBFS, до AMBIENCE_MAX_SEC) и assets/library/sfx/<вид>/*.flac (пик -10).
Те же нормировки, что у синтезированных ассетов, — поэтому все усиления в
pipeline_smart.py остаются в силе. Сборка предпочитает библиотеку, синтез
остаётся запасным путём, если для вида ничего не прошло гейты — с явным
предупреждением, не молча.

Запуск:
    python scripts/sound_library.py build            # всё
    python scripts/sound_library.py build --kinds ambience:wind_open,sfx:chapter_turn
    python scripts/sound_library.py report
"""
import argparse
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

print = functools.partial(print, flush=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY_ROOT = os.path.join(ROOT, "assets", "library")
MANIFEST_PATH = os.path.join(LIBRARY_ROOT, "manifest.json")
CACHE_DIR = os.path.join(ROOT, "temp_library")
# Браузерный User-Agent — тот же урок, что у Pexels в pipeline_smart.py
# («urllib с браузерным User-Agent, иначе Cloudflare 403»): Openverse за
# Cloudflare периодически отвечал на «PipelineMachineVideo/1.0» страницей
# «Just a moment...» (HTTP 403), по одному запросу на вид.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

OPENVERSE_AUDIO = "https://api.openverse.org/v1/audio/"
FREESOUND_SEARCH = "https://freesound.org/apiv2/search/text/"
SAFE_LICENSES = ("cc0",)

AMBIENCE_PEAK_DBFS = -12.0
SFX_PEAK_DBFS = -10.0
AMBIENCE_MAX_SEC = 180.0
# Обработка атмосферы при импорте — по трём дефектам, найденным замером на
# 29 принятых записях (13.09), а не по общим соображениям:
#  * ШОВ ЗАЦИКЛИВАНИЯ. Запись крутится под главой в 2-11 минут через
#    -stream_loop, а её конец и начало не совпадают: разброс от +25.1 дБ
#    (конец громче начала) до -137.9 дБ (файл кончается тишиной от
#    авторского fade-out). Каждый круг — слышимый рез. У синтезированных
#    ассетов это решал _seamless() в генераторе, у записей не решал никто.
#  * МОНО, РАСТИРАЖИРОВАННОЕ В ДВА КАНАЛА. У трёх принятых записей
#    корреляция каналов ровно 1.000, у одной вообще NaN (второй канал
#    пустой) — то есть ровно тот дефект, за который я сам критиковал свою
#    первую версию синтеза: фон схлопывается в точку и садится в центр, где
#    идёт голос.
#  * НЧ-МУСОР. Доля энергии ниже 40 Гц доходит до 0.906 (ночь), 0.862 (лес),
#    0.523-0.584 (ветер, огонь) — микрофонный ветровой «бум» и рокот. Под
#    закадром это мутная подложка, а не атмосфера.
AMBIENCE_LOOP_XFADE_SEC = 3.0     # шов: хвост подмешивается в начало
AMBIENCE_HIGHPASS_HZ = 70.0       # срез рокота; голос начинается выше
AMBIENCE_WIDEN_DELAY_SEC = 2.3    # моно -> два канала с разным сдвигом
MONO_CORRELATION = 0.98           # выше этого запись считается моно
LF_SHARE_WARN = 0.35              # доля энергии <40 Гц ДО обработки, в отчёт
CLAP_WINDOW_SEC = 10.0
CLAP_WINDOW_FRACS = (0.15, 0.5, 0.85)

# Порог маржи CLAP: положительный промпт обязан ПЕРЕБИВАТЬ худшую ловушку.
CLAP_MIN_MARGIN = 0.04
# АБСОЛЮТНЫЙ порог положительного скора СНЯТ как гейт (13.09) и остался
# только числом в отчёте. Это была моя же ошибка, ровно та, которую эта
# кодовая база уже документировала про CLIP: сырой косинус несопоставим
# между РАЗНЫМИ текстами. Громкие текстуры (огонь, дождь, ветер) дают
# 0.2-0.5, а цель «тихий гул пустого каменного зала» — это почти тишина, и
# сопоставлять там нечему: все пещерные и подземельные записи падали с
# 0.04-0.05 при пороге 0.08, то есть порог отсекал вид целиком, а не брак.
# Гейтами остаются ДВА относительных сигнала, оба на одной записи: маржа
# над худшей ловушкой и конкуренция видов (kind_competition).
CLAP_MIN_POSITIVE_REPORT_ONLY = 0.08
# AST: вероятность класса выше — вето (на любом окне).
AST_VETO = {"Speech": 0.12, "Music": 0.25, "Vehicle": 0.30, "Singing": 0.12}
# Измерительные гейты атмосферы
AMB_MAX_SILENCE_SHARE = 0.25
AMB_MAX_LRA = 20.0     # 18.2 отсекало настоящее летнее поле с птицами; 26 (ветер без ветрозащиты) остаётся вне
CLIP_SAMPLE_SHARE = 1e-4     # доля сэмплов на |1.0| — клиппинг
# Библиотека отдаётся в 48 кГц. Исходник ниже 44.1 кГц — это апсемплинг:
# новой информации в нём нет, а потолок слышен. Найдено замером: один
# принятый переход главы (UI_Menu_Whosh) пришёл с 16 кГц, то есть с
# потолком 8 кГц, и стоял рядом с семью вариантами на 44/48 — на слух это
# заметно более глухой звук, и ни один существующий гейт про это не
# спрашивал.
MIN_SOURCE_SAMPLE_RATE = 44100
HUM_PROMINENCE_DB = 14.0     # узкая линия 50/60 Гц над соседями ±3..10 Гц

# Слова в НАЗВАНИИ записи, при которых кандидат отбрасывается до моделей.
# Тот же принцип, что filter_alt_blocklist() у Pexels: у CLAP ветер, дождь,
# град и прибой — акустические соседи (эксперимент 13.09: ловушка «ocean
# waves» роняла настоящий ветер с +0.075 до -0.089), а в названии автор
# пишет прямо: «Big waves breaking», «Hail Comes In», «Door and wind».
TITLE_BLOCK_COMMON = ("loop", "synth", "generated", "processed", "reverb test", "test ")
# Музыкальные инструменты — общая ловушка ПРЕДМЕТНОГО слоя, и ловится она
# только по названию. Замер 14.09 на пакете Sonniss GDC-2019: запись
# «INSTRU STRING Double Bass, Bowed, Harmonic» прошла CLAP с маржой +0.087 и
# была принята как ОБНАЖЕНИЕ МЕЧА. Проверено, что вето AST её не поймало бы:
# у неё Music 0.019 при пороге AST_VETO 0.25, и это НИЖЕ, чем у двух законных
# записей sword_draw с Freesound (0.017 и 0.020) — по всем 25 записям
# объектного слоя Music лежит в 0.002..0.035, то есть класс «Music» на
# коротком смычковом флажолете не срабатывает в принципе. Расширять AST на
# object (что код честно называл отдельным решением со своей калибровкой)
# эту дыру НЕ закрывает — замер это опроверг. Закрывает список слов.
TITLE_BLOCK_INSTRUMENT = ("instru", "guitar", "piano", "violin", "cello", "double bass",
                          "bowed", "harmonic", "drum", "cymbal", "flute", "trumpet",
                          "orchestr", "chord", "melod", "bass ")
TITLE_BLOCK = {
    "wind_open": ("wave", "sea", "surf", "ocean", "beach", "rain", "hail", "thunder", "storm",
                  "door", "window", "indoor", "inside", "room", "car", "train", "city", "street",
                  # «WindChimes» прошёл CLAP с маржой +0.112 и AST Music 0.059 —
                  # колокольчики ловятся только по названию
                  "chime", "bell", "whistl", "flute", "pipe"),
    # «EXT Park urban morning» прошёл все гейты (AST Vehicle 0.005) — по звуку
    # чисто, но городской парк под средневековый лес класть незачем, когда
    # есть пять честных лесных записей; слово «urban» решает это до моделей
    "forest_birds": ("city", "street", "traffic", "urban", "zoo", "cage", "indoor", "room", "rain"),
    "night": ("city", "street", "traffic", "urban", "party", "club", "indoor"),
    "stone_hall": ("outdoor", "street", "traffic", "crowd", "concert", "organ", "choir"),
    "forge_fire": ("rain", "storm", "fireworks", "explosion", "gun"),
    "rain_mud": ("indoor", "inside", "window", "roof", "car", "tent", "umbrella", "thunder", "storm", "sea", "wave"),
    "crowd_market": ("stadium", "concert", "protest", "applause", "cheer", "traffic", "indoor", "restaurant", "cafe",
                     # реальный прогон: продавцы в громкоговоритель, бинго-зал, студенты,
                     # офисный холл — всё «толпа», но не рынок под открытым небом
                     "seller", "announc", "loudspeaker", "speech", "talk", "scream", "calling",
                     "bingo", "student", "headquarters", "office", "int ", "interior", "binaural"),
    "river_stream": ("sea", "wave", "surf", "rain", "waterfall", "fountain", "tap", "sink", "toilet", "shower"),
    "chapter_turn": ("sword", "hit", "impact", "explosion", "punch"),
    "plate_tick": ("clock", "metronome", "loop"),
    "reveal_riser": (),
    "reveal_hit": ("cymbal", "drum kit", "snare", "gun", "explosion", "kick", "bell", "808"),
    "typewriter": ("loop", "typing fast", "sequence"),
    # Предметный слой до 14.09 не имел блоклиста ни у одного вида — при том
    # что у всех тринадцати остальных он есть. Слова ниже взяты НЕ из головы,
    # а из того, что реально доехало до гейтов на живом пакете Sonniss.
    "sword_draw": TITLE_BLOCK_INSTRUMENT + ("saw", "circular", "door", "tank", "barrel", "factory"),
    "armour_clank": TITLE_BLOCK_INSTRUMENT + ("bag", "wallet", "backpack", "ikea", "zipper",
                                              "keys", "coins", "door"),
    "arrow_shot": TITLE_BLOCK_INSTRUMENT + ("jet", "train", "car", "engine", "compressor",
                                            "boat", "ferry", "motorcycle", "helicopter",
                                            "fair ride", "race"),
    "hammer_anvil": TITLE_BLOCK_INSTRUMENT + ("concrete", "wall", "construction", "jackhammer",
                                              "howitzer", "gun", "glass"),
    "footsteps_mud": TITLE_BLOCK_INSTRUMENT + ("helicopter", "car", "honda", "bmw", "industrial",
                                               "rocks", "shooting", "gallery", "ambience"),
}


def title_blocked(name, title):
    t = (title or "").lower()
    return next((w for w in TITLE_BLOCK_COMMON + TITLE_BLOCK.get(name, ()) if w in t), None)


def title_relevance(spec, title):
    """Сколько ключевых слов запросов встречается в названии — дешёвый
    первичный порядок: кандидаты с «wind»/«field» в названии оцениваются
    моделями раньше, чем случайные соседи по выдаче."""
    words = {w for q in spec["queries"] for w in q.lower().split()
             if w not in ("ambience", "ambient", "sound", "sounds", "single", "short", "soft", "cinematic")}
    t = (title or "").lower()
    return sum(1 for w in words if w in t)


def negatives_for(spec):
    """Ловушки вида: свой полный список (`neg`) ИЛИ общий плюс `extra_neg`.
    Полная замена нужна там, где общая ловушка совпадает с целью: у толпы
    цель — далёкие голоса, и «people talking» как ловушка перебивала
    положительный промпт на всех 30 кандидатах."""
    if "neg" in spec:
        return list(spec["neg"])
    return list(NEGATIVE_PROMPTS) + list(spec.get("extra_neg", ()))


# Группы, внутри которых виды конкурируют между собой. sfx сюда НЕ входит:
# его виды (переход главы, тик плашки, нарастание, удар) — роли в монтаже,
# а не разные места или предметы, и «нарастание против удара» акустически
# осмысленного победителя не имеет.
# Ловушки тихого фоли, общие для ВСЕХ предметных концептов. Выведены не из
# интуиции, а из измеренных ложных приёмов на реальном пакете (Kenney RPG
# Audio, 51 файл, прогон 14.09): «выхват меча» выигрывали скрип двери и
# застёжка ремня, «выстрел из лука» — три шага и кожаный ремешок, «молот по
# наковальне» — книга, положенная на стол. Общие NEGATIVE_PROMPTS тут не
# работают по устройству: речь/музыка/транспорт/гул/дисторшн любой сухой
# фоли-удар обходит даром, то есть маржа была положительной ни за что.
# Ловушка «шаг» сознательно НЕ добавлена в footsteps_mud — там это цель.
OBJECT_FOLEY_DECOYS = (
    "clothing rustle, fabric or leather being handled",
    "paper pages, a book placed on a table",
)

KIND_COMPETITION_KINDS = ("ambience", "object")

KIND_DECOYS = {
    "surf": "ocean waves breaking on a beach, sea surf",
    "traffic_city": "city street with traffic and cars",
}


NEGATIVE_PROMPTS = (
    "people talking, human speech, voices",
    "music, melody, musical instruments, singing",
    "traffic, car engine, motor vehicle, airplane",
    "electrical hum, buzzing, mains noise, fan noise",
    "digital distortion, clipping, glitch, static",
)

# Что ищем. queries — поисковые запросы (И-логика у Freesound слабая, поэтому
# несколько коротких); prompt — положительное описание для CLAP; extra_neg —
# ловушки сверх общих. Длительности в секундах.
LIBRARY_SPEC = {
    "ambience": {
        "wind_open": dict(
            queries=["wind open field", "field ambience wind", "wind grass meadow", "moorland wind",
                     "steppe wind ambience", "windy plain"],
            prompt="steady wind blowing over an open field, outdoors, no people",
            min_sec=15, keep=5),
        "forest_birds": dict(
            queries=["forest birds ambience", "birds spring forest", "woodland birdsong ambience",
                     "forest ambience morning", "park birds ambience"],
            prompt="quiet forest ambience with birds singing softly in the distance",
            min_sec=15, keep=5),
        "night": dict(
            queries=["night ambience crickets", "night forest ambience", "night countryside ambience",
                     "owl night ambience"],
            prompt="calm night ambience outdoors with crickets and distant owls",
            min_sec=15, keep=5),
        "stone_hall": dict(
            # Первый прогон: 0 из 11 — пул был пустой (туристы, буддийские
            # храмы, вентиляция), а не гейты слишком строгие. Запросы шире и
            # ближе к тому, как такие записи реально называют.
            queries=["church interior ambience", "cathedral ambience quiet", "room tone hall reverb",
                     "empty hall room tone", "castle interior ambience", "monastery ambience",
                     "cathedral room tone", "empty church ambience", "crypt ambience",
                     "cave ambience drips", "dungeon ambience", "cellar ambience",
                     "large empty room tone", "stone room ambience"],
            prompt="quiet interior room tone of a large stone hall, distant reverberant space",
            # «footsteps walking» как ловушка была ошибкой: в реальной записи
            # большого зала шаги неизбежны и они часть сцены — эта ловушка
            # перебила все пещерные и подземельные записи (1 из 30).
            # Music-вето тоже подняли: реверберирующие капли в пещере AST
            # уверенно читает как музыку (0.25-0.4), хотя это ровно то, что
            # нужно под склеп.
            neg=("one person speaking clearly, conversation",
                 "music with melody and instruments, singing",
                 "traffic, car engine, motor vehicle, airplane",
                 "digital distortion, clipping, glitch, static"),
            ast_veto={"Speech": 0.15, "Vehicle": 0.30},
            min_sec=30, keep=5),
        "forge_fire": dict(
            queries=["fireplace crackling", "campfire crackling", "bonfire", "wood fire burning",
                     "blacksmith forge fire"],
            prompt="a wood fire burning and crackling, calm and steady",
            min_sec=30, keep=5),
        "rain_mud": dict(
            queries=["rain ambience", "light rain outdoors", "rain on grass field", "gentle rain nature",
                     "rain forest ambience"],
            prompt="light steady rain falling outdoors, natural, no people",
            min_sec=15, keep=5),
        "crowd_market": dict(
            queries=["market crowd ambience", "crowd murmur walla", "village market crowd",
                     "medieval fair crowd", "outdoor crowd ambience distant", "crowd walla outdoor",
                     "distant crowd murmur", "people murmur background", "busy street market ambience"],
            # Промпт описывает то, ЧЕМ запись является, а не чего в ней нет:
            # формулировка «no clear words» давала положительный скор 0.03-0.09
            # на настоящих рыночных записях, то есть ниже порога уверенности.
            prompt="many people talking at once outdoors, busy crowd, hubbub of voices",
            # Цель — голоса, поэтому общая ловушка «people talking» здесь
            # неприменима (первый прогон: 0 из 30, все — ей). Свой список:
            # разборчивая речь одного человека, громкоговоритель, музыка,
            # транспорт, дисторшн.
            neg=("one person speaking clearly, announcement through a loudspeaker",
                 "music, melody, musical instruments, singing",
                 "traffic, car engine, motor vehicle, airplane",
                 "digital distortion, clipping, glitch, static"),
            min_sec=30, keep=5, ast_veto={"Music": 0.30, "Vehicle": 0.30}),
        "river_stream": dict(
            queries=["stream water flowing", "river ambience", "brook water", "creek ambience"],
            prompt="a small stream of water flowing gently over stones",
            min_sec=15, keep=4),
    },
    # ОБЪЕКТНЫЙ СЛОЙ — предметные разовые звуки под конкретным словом
    # сценария (разметка `[sfx:концепт]`). Именно то, чего синтез не умеет:
    # «плохо синтезированный колокол слышен как подделка мгновенно, в отличие
    # от полосы шума» (CLAUDE.md, ЧАСТЬ 13) — поэтому здесь только настоящие
    # записи, запасного синтетического пути у этого вида нет.
    #
    # Словарь выведен из ЧАСТОТ реального сценария канала, а не придуман: на
    # `videos/02_ne-mechom` доспех встречается 23 раза, меч/клинок 28, стрела/
    # лук 13, шаги/грязь 11, молот 5. Тот же урок, что уже стоил атмосфере
    # промаха с кузницей: словарь, собранный «по принципу», описывает не тот
    # ролик.
    #
    # `max_sec` у всех концептов НЕ БОЛЬШЕ OBJECT_POINT_MAX_SEC = 2.5 —
    # и это не вкус, а согласование с классификатором: длиннее этого
    # `object_asset_for()` считает запись протяжённой и вешает на неё
    # обрезку с фейдами, то есть удар молота поехал бы как подзвучник.
    "object": {
        "sword_draw": dict(
            queries=["sword unsheathe", "sword draw scabbard", "metal blade slide",
                     "sword sheath metal", "blade unsheathing"],
            prompt="a single steel sword being drawn from a scabbard, one metallic slide",
            extra_neg=["orchestral music sting", "person talking",
                       *OBJECT_FOLEY_DECOYS,
                       "a wooden door hinge creaking"],
            min_sec=0.25, max_sec=2.5, keep=4),
        "armour_clank": dict(
            # Первый прогон: 0 кандидатов из 0 — пул был ПУСТ, а не гейты
            # строги. Замер по каждому запросу отдельно: «chainmail movement»
            # 0, «armour clank metal» 0, «metal armour foley» 0 — все пять
            # многословных дали ровно ноль, при том что одиночное «chainmail»
            # даёт 21, «armor foley» 2 («HEAVY ARMOUR.wav»), «knight armor» 7
            # («knight-walking-on-hard-ground.wav»), «chain rattle» 97.
            # Тот же корень, что уже документирован для картинок: поиск
            # Openverse работает по И-логике, и лишнее слово сужает выдачу в
            # ноль. Запросы подобраны по РЕАЛЬНОЙ выдаче, а не по тому, как
            # звук хочется назвать.
            queries=["chainmail", "armor foley", "knight armor",
                     "chain rattle", "metal clank"],
            prompt="metal armour and chainmail clanking as someone moves, foley recording",
            extra_neg=["keys jingling in a pocket", "coins in a jar", "person talking",
                       *OBJECT_FOLEY_DECOYS,
                       "a wooden door closing, door latch"],
            min_sec=0.3, max_sec=2.5, keep=4),
        "arrow_shot": dict(
            queries=["arrow whoosh", "bow release arrow", "arrow flyby",
                     "archery bow shot", "arrow swoosh past"],
            prompt="a single arrow released from a bow and whooshing past",
            extra_neg=["gunshot", "orchestral music sting",
                       *OBJECT_FOLEY_DECOYS,
                       "a single footstep on the ground",
                       "clothing rustle, fabric movement"],
            min_sec=0.2, max_sec=2.0, keep=4),
        "hammer_anvil": dict(
            queries=["blacksmith hammer anvil", "hammer strike metal anvil",
                     "forge hammer hit", "anvil strike single"],
            prompt="a blacksmith hammer striking hot steel on an anvil, single strike",
            extra_neg=["construction site machinery", "church bell", "person talking",
                       *OBJECT_FOLEY_DECOYS,
                       "a wooden door closing, door latch",
                       "a metal pot or pan in a kitchen"],
            min_sec=0.2, max_sec=2.5, keep=4),
        "footsteps_mud": dict(
            queries=["footsteps mud", "walking in mud squelch", "boots mud steps",
                     "footsteps wet ground", "squelching mud footsteps"],
            prompt="heavy boots stepping in thick wet mud, squelching footsteps",
            extra_neg=["footsteps on a wooden floor indoors", "person talking",
                       *OBJECT_FOLEY_DECOYS],
            min_sec=0.3, max_sec=2.5, keep=4),
    },
    "sfx": {
        "chapter_turn": dict(
            queries=["whoosh transition soft", "cinematic whoosh short", "air whoosh swoosh",
                     "subtle whoosh", "deep whoosh"],
            prompt="a short soft whoosh transition sound, smooth, cinematic",
            min_sec=0.25, max_sec=1.6, keep=8),
        "plate_tick": dict(
            queries=["ui tick", "soft click ui", "notification tick", "ui blip subtle", "menu click"],
            prompt="a very short soft click or tick, user interface sound",
            min_sec=0.04, max_sec=0.5, keep=4),
        "reveal_riser": dict(
            queries=["riser cinematic", "tension riser", "suspense rise", "cinematic build up short"],
            prompt="a cinematic riser building tension, rising sweep",
            # нарастание ЗАКАНЧИВАЕТСЯ на моменте кульминации, 5с под предыдущей
            # фразой — обычная практика; первый прогон отсёк 4 из 10 за 4.1-5.0с
            min_sec=1.0, max_sec=5.2, keep=4),
        "reveal_hit": dict(
            # первый прогон: 0 из 9 — пул из 9 и отсечка по длине 3.0с на
            # «Deep hit» 3.3с и «FX Cinematic Impact» 3.5с: у удара длинный хвост,
            # это норма; сам удар всё равно НАЧИНАЕТСЯ на моменте
            queries=["low impact cinematic", "deep boom hit", "sub impact", "cinematic hit low",
                     "cinematic boom", "impact boom low", "bass drop impact", "trailer hit deep",
                     "deep impact reverb", "cinematic drum hit low"],
            prompt="a deep low cinematic impact boom, single hit, documentary",
            extra_neg=("drum kit, cymbal crash", "explosion with debris"),
            min_sec=0.4, max_sec=4.5, keep=4),
        "typewriter": dict(
            queries=["typewriter key single", "typewriter keystroke", "mechanical keyboard single key",
                     "keyboard key press single click"],
            prompt="a single typewriter key press, one click",
            min_sec=0.03, max_sec=0.6, keep=8),
    },
}


# --------------------------------------------------------------- источники
# Openverse без ключа: 20 запросов/мин и 200/день (заголовки x-ratelimit-*,
# замерено 13.09). Первая версия слала запросы подряд и получала HTTP 429 на
# ВСЕХ видах после первого — сборка «завершалась» за секунды с нулём
# результатов. Теперь: пауза между поисковыми запросами, повтор при 429 с
# ожиданием и дисковый кэш выдачи, чтобы перезапуск не тратил лимит заново.
SEARCH_MIN_INTERVAL_SEC = 3.3
_last_search = [0.0]


def _get_json(url, headers=None, timeout=40, retries=3):
    for attempt in range(retries + 1):
        wait = SEARCH_MIN_INTERVAL_SEC - (time.time() - _last_search[0])
        if wait > 0:
            time.sleep(wait)
        _last_search[0] = time.time()
        req = urllib.request.Request(url, headers=dict({"User-Agent": UA}, **(headers or {})))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 403, 502, 503) and attempt < retries:
                time.sleep(20.0 * (attempt + 1))
                continue
            raise


def _search_cached(cache_key, fetch):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "search_" + hashlib.sha1(cache_key.encode()).hexdigest()[:16] + ".json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    data = fetch()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def is_safe_license(item):
    """Fail-closed: строго cc0 по полю результата, не по фильтру запроса."""
    return str(item.get("license", "")).lower() in SAFE_LICENSES


def openverse_search(query, pages=2):
    """Анонимно Openverse отдаёт не больше 20 результатов на страницу
    (page_size>20 -> HTTP 401, поймано вживую), поэтому глубина — страницами:
    две на атмосферу, это ещё один запрос из дневных 200."""
    results = []
    for page in range(1, pages + 1):
        url = OPENVERSE_AUDIO + "?" + urllib.parse.urlencode(
            {"q": query, "license": "cc0", "page_size": 20, "page": page})
        try:
            data = _search_cached("openverse|" + url, lambda: _get_json(url))
        except urllib.error.HTTPError as e:
            print(f"    openverse: HTTP {e.code} на «{query}» (стр. {page}): {e.read()[:120]!r}")
            break
        except Exception as e:
            print(f"    openverse: {type(e).__name__} на «{query}» (стр. {page}): {e}")
            break
        chunk = data.get("results") or []
        results.extend(chunk)
        if len(chunk) < 20:
            break
    out = []
    for it in results:
        if not is_safe_license(it):
            continue
        out.append({
            "source": f"openverse:{it.get('provider')}",
            "id": f"{it.get('provider')}:{it.get('id')}",
            "foreign_id": (it.get("foreign_landing_url") or "").rstrip("/").split("/")[-1],
            "title": it.get("title") or "",
            "creator": it.get("creator") or "",
            "license": it.get("license"),
            "license_url": it.get("license_url"),
            "url": it.get("url"),
            "landing": it.get("foreign_landing_url"),
            "duration": (it.get("duration") or 0) / 1000.0,
            "query": query,
        })
    return out


def freesound_search(query, key, page_size=30):
    """Прямой поиск Freesound — только при ключе, и только CC0.
    Отдаёт hq-превью (оригинал требует OAuth пользователя)."""
    if not key:
        return []
    url = FREESOUND_SEARCH + "?" + urllib.parse.urlencode({
        "query": query, "filter": 'license:"Creative Commons 0"', "page_size": page_size,
        "fields": "id,name,username,license,duration,previews,url", "token": key})
    out = []
    try:
        data = _search_cached("freesound|" + query, lambda: _get_json(url))
    except Exception as e:
        print(f"    freesound: {type(e).__name__} на «{query}»")
        return out
    for it in data.get("results") or []:
        lic = str(it.get("license", ""))
        if "publicdomain/zero" not in lic and "Creative Commons 0" not in lic:
            continue
        prev = (it.get("previews") or {}).get("preview-hq-mp3")
        if not prev:
            continue
        out.append({
            "source": "freesound", "id": f"freesound:{it['id']}", "foreign_id": str(it["id"]),
            "title": it.get("name") or "", "creator": it.get("username") or "",
            "license": "cc0", "license_url": lic, "url": prev, "landing": it.get("url"),
            "duration": float(it.get("duration") or 0), "query": query,
        })
    return out


def gather_candidates(spec):
    key = os.environ.get("FREESOUND_API_KEY", "").strip()
    seen, out = set(), []
    pages = 2 if spec.get("min_sec", 0) >= 15 else 1
    for q in spec["queries"]:
        for item in openverse_search(q, pages) + freesound_search(q, key):
            fid = item["foreign_id"] or item["id"]
            if fid in seen:
                continue
            seen.add(fid)
            out.append(item)
    return out


def preview_url(url, quality):
    """Freesound кладёт превью парами: ..._NNN-hq.mp3 (128 kbps) и -lq.mp3
    (64 kbps). Оценка идёт по lq — вдвое меньше трафика на 15-20 кандидатов,
    а решение «есть ли речь/музыка/гул и про то ли это» от битрейта не
    зависит; hq качается только для принятых."""
    if "freesound.org/previews/" in url:
        return re.sub(r"-(hq|lq)\.(mp3|ogg)$", f"-{quality}.\\2", url)
    return url


def download(item, quality="hq"):
    os.makedirs(CACHE_DIR, exist_ok=True)
    url = preview_url(item["url"], quality)
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"{h}.audio")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=180) as r, open(path + ".tmp", "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(path + ".tmp", path)
        return path
    except Exception as e:
        print(f"    скачивание: {type(e).__name__} {item['title'][:40]!r}")
        return None


# ------------------------------------------------------------ измерения
def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def probe_sample_rate(path):
    """Частота дискретизации первой аудиодорожки, 0 — не определилась.

    Берётся ПЕРВАЯ группа цифр, а не `int()` от всей строки: `-of csv=p=0`
    на одном поле всё равно печатает висячую запятую («16000,»), и `int()`
    на ней падает. Первая версия ловила это широким `except` и возвращала
    0, то есть гейт частоты был молчаливым no-op — он не отбраковал даже
    тот файл на 16 кГц, ради которого писался. Ровно тот же класс отказа,
    что уже дважды ловили у пустого шва: код возврата нулевой, ошибок нет,
    результата тоже нет.
    """
    r = _run(["ffprobe", "-v", "error", "-select_streams", "a:0",
              "-show_entries", "stream=sample_rate", "-of", "csv=p=0", path])
    m = re.search(r"\d+", r.stdout or "")
    return int(m.group(0)) if m else 0


def probe_duration(path):
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    try:
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def decode_f32(path, start, dur, sr):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
                        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"], capture_output=True)
    import numpy as np
    return np.frombuffer(r.stdout, dtype=np.float32)


def measure_loudness(path):
    """(integrated LUFS, LRA, true peak dBTP) через ebur128."""
    r = _run(["ffmpeg", "-v", "info", "-t", "190", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"])
    txt = r.stderr
    def grab(pattern):
        m = re.findall(pattern, txt)
        return float(m[-1]) if m else None
    return grab(r"I:\s*(-?[\d.]+) LUFS"), grab(r"LRA:\s*([\d.]+) LU"), grab(r"Peak:\s*(-?[\d.]+) dBFS")


def silence_share(path, duration, lufs=None):
    """Доля «провалов» относительно громкости САМОЙ записи (I − 25 LU), а не
    абсолютных −45 dB: тихий пустой зал целиком лежал бы ниже абсолютного
    порога и отбраковывался как «тишина» (Monastery ruins_2: 0.966), хотя
    после нормировки пика на импорте это ровно то, что нужно. Абсолютный
    порог остаётся полом −70 dB."""
    noise = max(-70.0, (lufs - 25.0)) if lufs is not None else -45.0
    r = _run(["ffmpeg", "-v", "info", "-t", "190", "-i", path, "-af",
              f"silencedetect=noise={noise:.0f}dB:d=0.5", "-f", "null", "-"])
    total = 0.0
    for m in re.finditer(r"silence_duration:\s*([\d.]+)", r.stderr):
        total += float(m.group(1))
    return total / min(duration, 190.0) if duration else 0.0


def channel_correlation(path, seconds=8.0):
    """Корреляция левого и правого канала. ~1.0 — моно, растиражированное в
    стерео; NaN (пустой канал) считается моно."""
    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-t", f"{seconds}", "-i", path,
                        "-f", "f32le", "-ac", "2", "-ar", "48000", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.float32)
    if a.size < 9600:
        return 1.0
    a = a.reshape(-1, 2).T
    if a[0].std() < 1e-9 or a[1].std() < 1e-9:
        return 1.0
    c = float(np.corrcoef(a[0], a[1])[0, 1])
    return 1.0 if c != c else c


def lf_share(samples, sr=48000, cutoff=40.0):
    """Доля энергии ниже cutoff — микрофонный рокот у полевых записей."""
    import numpy as np
    n = min(samples.size, sr * 8)
    if n < sr:
        return 0.0
    X = np.abs(np.fft.rfft(samples[:n] * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return float(X[f < cutoff].sum() / (X[f < 12000].sum() + 1e-12))


def clipping_share(samples):
    import numpy as np
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples) >= 0.999))


def hum_prominence_db(samples, sr):
    """Насколько УЗКАЯ линия на 50/60 Гц (и гармониках) торчит над своими
    ближайшими соседями по спектру (3..10 Гц в стороны).

    Именно над соседями, а не над медианой полосы: сетевой гул — линия
    шириной в доли герца, а ветер и огонь — широкий низкочастотный гул,
    у которого 50 Гц ничем не выделяются среди 45 и 55. Первая версия
    сравнивала с медианой 30..300 Гц и отбраковывала настоящий ветер как
    «гул» — поймано на первом же реальном кандидате.
    """
    import numpy as np
    n = min(samples.size, sr * 8)
    if n < sr:
        return 0.0
    x = samples[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    worst = 0.0
    for f0 in (50.0, 60.0, 100.0, 120.0, 150.0, 180.0):
        line = spec[(freqs >= f0 - 0.6) & (freqs <= f0 + 0.6)]
        side = spec[((freqs >= f0 - 10) & (freqs <= f0 - 3)) | ((freqs >= f0 + 3) & (freqs <= f0 + 10))]
        if line.size and side.size:
            worst = max(worst, 10 * math.log10((line.max() + 1e-12) / (side.mean() + 1e-12)))
    return worst


# ---------------------------------------------------------------- модели
_CLAP = {}
_AST = {}


def clap():
    if "model" not in _CLAP:
        import torch
        from transformers import ClapModel, ClapProcessor
        name = "laion/larger_clap_general"
        _CLAP["model"] = ClapModel.from_pretrained(name).eval()
        _CLAP["proc"] = ClapProcessor.from_pretrained(name)
        _CLAP["torch"] = torch
    return _CLAP


def ast():
    if "model" not in _AST:
        import torch
        from transformers import AutoFeatureExtractor, ASTForAudioClassification
        name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        _AST["fe"] = AutoFeatureExtractor.from_pretrained(name)
        _AST["model"] = ASTForAudioClassification.from_pretrained(name).eval()
        m = _AST["model"]
        _AST["label_idx"] = {lab: i for i, lab in m.config.id2label.items()}
        _AST["torch"] = torch
    return _AST


def clap_scores(windows, prompts):
    """[[скор по каждому промпту] для каждого окна]."""
    c = clap()
    torch = c["torch"]
    with torch.no_grad():
        ti = c["proc"](text=list(prompts), return_tensors="pt", padding=True)
        t = c["model"].get_text_features(**ti)
        t = getattr(t, "pooler_output", t)
        if not hasattr(t, "norm"):
            t = t[0]
        t = t / t.norm(dim=-1, keepdim=True)
        rows = []
        for w in windows:
            ai = c["proc"](audio=w, sampling_rate=48000, return_tensors="pt")
            a = c["model"].get_audio_features(**ai)
            a = getattr(a, "pooler_output", a)
            if not hasattr(a, "norm"):
                a = a[0]
            a = a / a.norm(dim=-1, keepdim=True)
            rows.append((a @ t.T)[0].tolist())
    return rows


def ast_probs(windows16k, labels):
    m = ast()
    torch = m["torch"]
    out = []
    with torch.no_grad():
        for w in windows16k:
            logits = m["model"](**m["fe"](w, sampling_rate=16000, return_tensors="pt")).logits[0]
            p = torch.sigmoid(logits)
            out.append({lab: float(p[m["label_idx"][lab]]) for lab in labels if lab in m["label_idx"]})
    return out


# ------------------------------------------------------------- оценка
def _measure_key(path, spec):
    negs = negatives_for(spec)
    raw = "|".join([os.path.basename(path), str(os.path.getsize(path)), spec["prompt"], *negs,
                    "clap:laion/larger_clap_general", "ast:MIT/ast-finetuned-audioset-10-10-0.4593", "v7"])
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def measure(path, kind, spec):
    """Сырые измерения кандидата — БЕЗ порогов, с кэшем на диске.

    Пороги живут в judge(): перенастройка порога после разбора результатов
    не должна заново гонять две модели по всем кандидатам (замер 13.09:
    ~10с на кандидата, 30 кандидатов на вид, 13 видов).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "measure_" + _measure_key(path, spec) + ".json")
    if os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    m = {}
    dur = probe_duration(path)
    if not dur:
        m["undecodable"] = True
        return m
    m["duration"] = round(dur, 3)
    m["source_sample_rate"] = probe_sample_rate(path)
    if dur <= CLAP_WINDOW_SEC + 1.0:
        starts = [0.0]
    else:
        starts = [max(0.0, min(dur - CLAP_WINDOW_SEC, f * dur - CLAP_WINDOW_SEC / 2)) for f in CLAP_WINDOW_FRACS]
    win48 = [decode_f32(path, st, min(CLAP_WINDOW_SEC, dur), 48000) for st in starts]
    win48 = [w for w in win48 if w.size > 4800]
    if not win48:
        m["undecodable"] = True
        return m
    m["clipping_share"] = round(max(clipping_share(w) for w in win48), 6)
    if kind == "ambience":
        m["hum_db"] = round(max(hum_prominence_db(w, 48000) for w in win48), 1)
        m["lf_share"] = round(max(lf_share(w) for w in win48), 3)
        m["channel_corr"] = round(channel_correlation(path), 3)
        lufs, lra, tp = measure_loudness(path)
        m.update(lufs=lufs, lra=lra, true_peak=tp)
        m["silence_share"] = round(silence_share(path, dur, lufs), 3)
    negs = negatives_for(spec)
    rows = clap_scores(win48, [spec["prompt"]] + negs)
    m["clap_rows"] = [[round(x, 4) for x in r] for r in rows]
    m["neg_names"] = negs
    if kind in KIND_COMPETITION_KINDS:
        # Конкуренция видов — ВНУТРИ своей группы (LIBRARY_SPEC[kind]).
        # Раньше здесь стояло kind == "ambience", и это была не осторожность,
        # а дыра: у object конкуренции не было вообще, то есть кандидату
        # хватало обойти пять общих приманок (речь/музыка/транспорт/гул/
        # дисторшн), которые любой сухой фоли-удар обходит даром. Ничто не
        # спрашивало «это скорее выхват меча или шаг?».
        by_name = {n: sp for n, sp in LIBRARY_SPEC[kind].items()}
        names = list(by_name) + list(KIND_DECOYS)
        kp = [by_name[n]["prompt"] for n in by_name] + list(KIND_DECOYS.values())
        krows = clap_scores(win48, kp)
        kavg = [sum(r[i] for r in krows) / len(krows) for i in range(len(names))]
        korder = sorted(range(len(names)), key=lambda i: -kavg[i])
        m["kind_winner"] = names[korder[0]]
        m["kind_gap"] = round(kavg[korder[0]] - kavg[korder[1]], 4)
        m["kind_scores"] = {n: round(x, 4) for n, x in zip(names, kavg)}
    if kind == "ambience":
        # AST-вето откалибровано на длинных полевых записях и сознательно
        # НЕ распространяется на object этим заходом — это отдельное решение
        # со своей калибровкой, а не довесок к починке конкуренции видов.
        labels = sorted(set(AST_VETO) | set(spec.get("ast_veto", {}) or {}))
        win16 = [decode_f32(path, st, min(CLAP_WINDOW_SEC, dur), 16000) for st in starts]
        probs = ast_probs([w for w in win16 if w.size > 1600], labels)
        m["ast"] = {lab: round(max(p.get(lab, 0.0) for p in probs), 3) for lab in labels}
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(m, f)
    return m


def judge(m, kind, spec):
    """Пороги поверх сырых измерений -> вердикт с причинами."""
    v = {"reasons": []}
    if m.get("undecodable"):
        v["reasons"].append("undecodable")
        return v
    v["duration"] = m["duration"]
    lo, hi = spec.get("min_sec", 0.0), spec.get("max_sec", 1e9)
    if m["duration"] < lo or m["duration"] > hi:
        v["reasons"].append(f"duration_{'short' if m['duration'] < lo else 'long'}")
        return v
    v["clipping_share"] = m["clipping_share"]
    # Для полевой записи сэмплы на полной шкале — клиппинг. Для СДЕЛАННОГО
    # эффекта (riser/hit из пака) это норма мастеринга: пик приведён к
    # 0 dBFS лимитером, не перегруз. Первый прогон нарастаний отсёк 4 из
    # 10 ровно за это. Порог у эффектов в 100 раз мягче.
    sr = m.get("source_sample_rate") or 0
    v["source_sample_rate"] = sr
    # 0 = не удалось определить; это не повод отказывать (fail-open, как и
    # везде, где измерение не получилось), отказываем только по измеренному
    if sr and sr < MIN_SOURCE_SAMPLE_RATE:
        v["reasons"].append("low_sample_rate")
    clip_limit = CLIP_SAMPLE_SHARE if kind == "ambience" else CLIP_SAMPLE_SHARE * 100
    if m["clipping_share"] > clip_limit:
        v["reasons"].append("clipping")
    if kind == "ambience":
        v.update(hum_db=m.get("hum_db"), silence_share=m.get("silence_share"),
                 lufs=m.get("lufs"), lra=m.get("lra"), true_peak=m.get("true_peak"),
                 lf_share=m.get("lf_share"), channel_corr=m.get("channel_corr"),
                 mono=bool((m.get("channel_corr") or 0) >= MONO_CORRELATION))
        if (m.get("hum_db") or 0) > HUM_PROMINENCE_DB:
            v["reasons"].append("mains_hum")
        if (m.get("silence_share") or 0) > AMB_MAX_SILENCE_SHARE:
            v["reasons"].append("too_much_silence")
        if m.get("lra") is not None and m["lra"] > AMB_MAX_LRA:
            v["reasons"].append("too_dynamic")
    if v["reasons"]:
        return v
    rows, negs = m["clap_rows"], m["neg_names"]
    pos = [r[0] for r in rows]
    worst = [max(r[1:]) for r in rows]
    margin = min(p - n for p, n in zip(pos, worst))
    k = min(range(len(rows)), key=lambda i: pos[i] - worst[i])
    v.update(clap_pos=round(sum(pos) / len(pos), 4), clap_margin=round(margin, 4),
             clap_worst_neg=negs[max(range(len(negs)), key=lambda j: rows[k][1 + j])])
    if margin < CLAP_MIN_MARGIN:
        v["reasons"].append("clap_negative_wins")
    if m.get("kind_scores"):
        # Вид обязан выиграть у всех остальных видов и приманок. Это замена
        # акустической части словесного блоклиста — см. kind_competition().
        v["kind_winner"] = m.get("kind_winner")
        v["kind_gap"] = m.get("kind_gap")
        if m.get("kind_winner") != spec.get("_name"):
            v["reasons"].append(f"kind_lost_to_{m.get('kind_winner')}")
    if kind == "ambience" and m.get("ast"):
        # Переопределение вида ЗАМЕНЯЕТ общий набор целиком. Первая версия
        # делала update(): у толпы Speech-вето оставалось от общего набора и
        # срезало все 30 кандидатов — гомон толпы для AST и есть «Speech».
        veto = dict(spec["ast_veto"]) if "ast_veto" in spec else dict(AST_VETO)
        v["ast"] = m["ast"]
        for lab, thr in veto.items():
            if m["ast"].get(lab, 0.0) > thr:
                v["reasons"].append(f"ast_{lab.lower()}")
    return v


def confirmed_sample_rate(item, measured):
    """Частота ИСХОДНИКА: lq — только подозрение, доказательство — hq.

    РЕАЛЬНЫЙ БАГ, найден живым прогоном 13.09, а не чтением кода. Отбор
    идёт по lq-превью (64 kbps — вдвое меньше трафика, см. preview_url), и
    гейт частоты мерил ИМЕННО ЕГО. Замер на живых парах превью одного и
    того же звука:

        источник 48000 -> hq 48000, lq 24000
        источник 44100 -> hq 44100, lq 24000
        другая запись  -> hq 44100, lq 44100

    То есть частота lq — свойство ТРАНСКОДА Freesound, а не исходника, и
    меняется от загрузки к загрузке. Отбраковка при этом была окончательной:
    на первом же живом прогоне вида `object:hammer_anvil` 9 кандидатов из 16
    ушли с причиной `low_sample_rate` — ровно те записи, которые и нужны
    («Anvil - Hammer on 6mm steel 1 time short», «Blacksmith Hammer»).
    Существующая библиотека уцелела случайно: её измерения закэшированы ДО
    появления гейта, у всех 60 записей манифеста `source_sample_rate` пуст,
    и условие `if sr and ...` на них не срабатывает вообще.

    hq качается ТОЛЬКО для подозреваемых (на hammer_anvil это 9 файлов по
    ~30 КБ), иначе экономия трафика на lq теряет смысл.

    Чинится ИЗМЕРЕНИЕ, а не вердикт, и это важно: `judge()` возвращается
    сразу, как только появилась первая причина отказа, ДО расчёта CLAP-маржи
    (у отклонённых кандидатов в логе стоит `margin=+nan` — она не считалась).
    Снять причину постфактум значило бы отдать кандидата с пустым списком
    причин, ни разу не проверенного на «про то ли это вообще».

    Спектральный срез вместо частоты рассматривался и ОТКЛОНЁН: этот же
    проект уже измерил, что у части легальных эффектов верха нет по самой их
    природе («низкий удар обрезан на 0.4-7 кГц — там просто нет верха»), то
    есть гейт по срезу выбрасывал бы настоящие глубокие удары.
    """
    if not measured or measured >= MIN_SOURCE_SAMPLE_RATE:
        return measured
    hq = download(item, "hq")
    sr = probe_sample_rate(hq) if hq else 0
    # Не скачалось/не измерилось — остаётся измеренное по lq, то есть отказ.
    # Fail-closed здесь безопасен: кандидат просто не попадает в библиотеку.
    return sr or measured


def evaluate(item, path, kind, name, spec):
    """Полный разбор одного кандидата -> dict с вердиктом и причиной."""
    m = measure(path, kind, spec)
    if m.get("source_sample_rate"):
        m["source_sample_rate"] = confirmed_sample_rate(item, m["source_sample_rate"])
    v = judge(m, kind, spec)
    v.update(id=item["id"], title=item["title"])
    return v


def _trim_bounds(src, lufs):
    """(старт, конец) без ведущей и хвостовой тишины — относительно громкости
    самой записи. Авторский fade-out в конце файла при зацикливании даёт
    провал в тишину и резкий вход обратно."""
    noise = max(-70.0, (lufs - 30.0)) if lufs is not None else -50.0
    r = _run(["ffmpeg", "-v", "info", "-i", src, "-af",
              f"silencedetect=noise={noise:.0f}dB:d=0.4", "-f", "null", "-"])
    dur = probe_duration(src) or 0.0
    start, end = 0.0, dur
    for m in re.finditer(r"silence_start:\s*(-?[\d.]+)[\s\S]*?silence_end:\s*([\d.]+)", r.stderr):
        a, b = float(m.group(1)), float(m.group(2))
        if a <= 0.15:
            start = max(start, b)
        if b >= dur - 0.15:
            end = min(end, a)
    if end - start < 10.0:
        return 0.0, dur
    return start, end


def _seamless_loop(stage, dst, body, xf):
    """Собрать бесшовную петлю: хвост длиной xf подмешивается в начало с
    обратной кривой, тело идёт следом. Конец итога переходит в его же начало
    непрерывно, потому что стык — это одно и то же место исходника.

    Кроссфейд считается ДВУМЯ ОТДЕЛЬНЫМИ ФАЙЛАМИ, а не тремя ветками одного
    `asplit` в одном filter_complex — и это исправление реального,
    измеренного бага, а не стилистика. Прежняя версия делала
    `[0:a]atrim.. -> [tail][head]acrossfade -> concat` в один проход, и
    acrossfade в такой схеме отдавал ПУСТОЙ поток: на выходе оставалось
    только тело, то есть «бесшовная петля» не собиралась НИ РАЗУ, ни у
    одной записи. Видно это было только по длительности (174с вместо 177с)
    и по замеру щелчка на стыке — код возврата ffmpeg был нулевой.
    """
    # Длина берётся у РЕАЛЬНОГО файла, а не из запрошенной обрезки: ffmpeg
    # отдаёт чуть меньше, чем просили (-t 77.662 -> 77.632), и хвост,
    # отсчитанный от запрошенной длины, выходил короче xf. acrossfade с
    # входом короче d молча не собирался, и запись оставалась без петли —
    # на замере это было видно как стык 11.2 при норме около 1.
    body = probe_duration(stage) or body
    xf = min(xf, body / 4.0)
    if xf <= 0.1:
        return None
    h = hashlib.sha1(dst.encode()).hexdigest()[:12]
    head = os.path.join(CACHE_DIR, f"lh_{h}.wav")
    tail = os.path.join(CACHE_DIR, f"lt_{h}.wav")
    mid = os.path.join(CACHE_DIR, f"lb_{h}.wav")
    seam = os.path.join(CACHE_DIR, f"ls_{h}.wav")
    out = os.path.join(CACHE_DIR, f"loop_{h}.wav")
    steps = [
        ["ffmpeg", "-y", "-v", "error", "-i", stage, "-t", f"{xf:.3f}",
         "-ar", "48000", "-ac", "2", head],
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{body - xf:.3f}", "-i", stage,
         "-t", f"{xf:.3f}", "-ar", "48000", "-ac", "2", tail],
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{xf:.3f}", "-i", stage,
         "-t", f"{body - 2 * xf:.3f}", "-ar", "48000", "-ac", "2", mid],
    ]
    for cmd in steps[:3]:
        if _run(cmd).returncode != 0:
            return None
    # Длительность кроссфейда берётся у РЕАЛЬНО извлечённых кусков. ffmpeg
    # отдаёт 2.999583 там, где просили 3.000, и acrossfade с d БОЛЬШЕ входа
    # молча отдаёт пустой поток — тот же класс бага, что и выше, просто на
    # третий знак. Округление вниз до миллисекунды, чтобы d гарантированно
    # не превысил вход.
    d = min(probe_duration(head) or 0.0, probe_duration(tail) or 0.0)
    d = math.floor(d * 1000) / 1000.0
    if d <= 0.1:
        return None
    if _run(["ffmpeg", "-y", "-v", "error", "-i", tail, "-i", head, "-filter_complex",
             f"[0:a][1:a]acrossfade=d={d:.3f}:c1=tri:c2=tri[o]", "-map", "[o]",
             "-ar", "48000", "-ac", "2", seam]).returncode != 0:
        return None
    # шов обязан быть непустым: именно молчаливая пустота и была багом
    if not probe_duration(seam):
        return None
    lst = os.path.join(CACHE_DIR, f"lc_{h}.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.write("file '%s'\nfile '%s'\n" % (os.path.abspath(seam), os.path.abspath(mid)))
    if _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
             "-ar", "48000", "-ac", "2", out]).returncode != 0:
        return None
    return out if probe_duration(out) else None


def _widen_mono(stage, dst):
    """Моно -> два канала КРУГОВЫМ сдвигом одной записи, после сборки петли.

    Смысл тот же, что был: для стационарной текстуры (ветер, дождь, гул)
    сдвиг на секунды даёт настоящую декорреляцию каналов и ощущение
    пространства, а не точки в центре; задержка секундная, не миллисекундная,
    поэтому гребёнчатой окраски, как у Хааса, не возникает.

    Почему именно КРУГОВОЙ сдвиг и именно ПОСЛЕ петли — это исправление
    реального, измеренного бага. Первая версия ставила `adelay` ДО сборки
    петли: правый канал получал 2.3 с цифровой тишины в начале, файл
    становился на 2.3 с длиннее, а петля резалась по исходной длине — то
    есть на каждом обороте правый канал проваливался в дыру. Замер щелчка
    на стыке (скачок между последним и первым семплом против типичного
    межсемплового скачка): 18.7 и 30.6 против 1.0 у нормального стыка.
    Круговой сдвиг длину не меняет и дыры не создаёт: повёрнутая
    бесшовная петля остаётся бесшовной.
    """
    dur = probe_duration(stage)
    shift = AMBIENCE_WIDEN_DELAY_SEC
    if dur <= 3 * shift:
        # слишком коротко, чтобы поворот дал независимый материал
        return stage
    out = os.path.join(CACHE_DIR, "wide_" + hashlib.sha1(dst.encode()).hexdigest()[:12] + ".wav")
    fc = (f"[0:a]aformat=channel_layouts=mono,asplit=3[l][r1][r2];"
          f"[r1]atrim=start={shift:.3f},asetpts=N/SR/TB[ra];"
          f"[r2]atrim=0:{shift:.3f},asetpts=N/SR/TB[rb];"
          f"[ra][rb]concat=n=2:v=0:a=1[r];"
          f"[l][r]join=inputs=2:channel_layout=stereo[out]")
    if _run(["ffmpeg", "-y", "-v", "error", "-i", stage, "-filter_complex", fc,
             "-map", "[out]", "-ar", "48000", "-ac", "2", out]).returncode == 0:
        return out
    return stage


def import_file(src, dst, kind, dur):
    """-> FLAC 48k stereo с объявленным пиком.

    Атмосфера дополнительно приводится к состоянию, в котором её можно
    крутить под главой (см. константы AMBIENCE_* выше): срез рокота,
    расширение моно, обрезка тишины по краям и подмешивание хвоста в начало
    для бесшовной петли. Эффекты не трогаются — они звучат один раз.
    """
    peak_target = AMBIENCE_PEAK_DBFS if kind == "ambience" else SFX_PEAK_DBFS
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if kind != "ambience":
        # Пик меряется ПОСЛЕ приведения к 48к/стерео, а не у исходника.
        # Реальный найденный перекос: ffmpeg при mono -> stereo применяет
        # -3 дБ на канал (сохраняет суммарную мощность), и эффект из моно
        # выходил на 3 дБ тише объявленного пика. На слух это значило, что
        # варианты ОДНОГО эффекта звучат на 3 дБ врозь случайным образом
        # (замер: стерео-исходники ровно -10.0, моно -13.0..-13.2).
        conv = os.path.join(CACHE_DIR, "sfx_" + hashlib.sha1(dst.encode()).hexdigest()[:12] + ".wav")
        if _run(["ffmpeg", "-y", "-v", "error", "-i", src,
                 "-ar", "48000", "-ac", "2", conv]).returncode != 0:
            return False
        r = _run(["ffmpeg", "-v", "info", "-i", conv, "-af", "volumedetect", "-f", "null", "-"])
        m = re.findall(r"max_volume:\s*(-?[\d.]+) dB", r.stderr)
        gain = peak_target - (float(m[-1]) if m else 0.0)
        return _run(["ffmpeg", "-y", "-v", "error", "-i", conv, "-af", f"volume={gain:.2f}dB",
                     "-ar", "48000", "-ac", "2", "-c:a", "flac", "-compression_level", "8",
                     dst]).returncode == 0

    lufs, _, _ = measure_loudness(src)
    t0, t1 = _trim_bounds(src, lufs)
    t1 = min(t1, t0 + AMBIENCE_MAX_SEC)
    body = t1 - t0
    xf = AMBIENCE_LOOP_XFADE_SEC if body > 4 * AMBIENCE_LOOP_XFADE_SEC else 0.0
    mono = channel_correlation(src) >= MONO_CORRELATION

    stage = os.path.join(CACHE_DIR, "stage_" + hashlib.sha1(dst.encode()).hexdigest()[:12] + ".wav")
    r = _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{body:.3f}", "-i", src,
              "-af", f"highpass=f={AMBIENCE_HIGHPASS_HZ:.0f}", "-ar", "48000", "-ac", "2", stage])
    if r.returncode != 0:
        return False

    if xf > 0:
        looped = _seamless_loop(stage, dst, body, xf)
        if looped:
            stage = looped

    if mono:
        stage = _widen_mono(stage, dst)

    r = _run(["ffmpeg", "-v", "info", "-i", stage, "-af", "volumedetect", "-f", "null", "-"])
    m = re.findall(r"max_volume:\s*(-?[\d.]+) dB", r.stderr)
    gain = peak_target - (float(m[-1]) if m else 0.0)
    return _run(["ffmpeg", "-y", "-v", "error", "-i", stage, "-af", f"volume={gain:.2f}dB",
                 "-ar", "48000", "-ac", "2", "-c:a", "flac", "-compression_level", "8",
                 dst]).returncode == 0


def load_manifest():
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"items": {}}


def save_manifest(m):
    os.makedirs(LIBRARY_ROOT, exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    os.replace(tmp, MANIFEST_PATH)


def library_files(kind, name):
    d = os.path.join(LIBRARY_ROOT, kind, name)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".flac"))


def build_kind(kind, name, max_candidates=18, manifest=None, rejected_log=None):
    spec = dict(LIBRARY_SPEC[kind][name], _name=name)
    print(f"\n== {kind}/{name}: {spec['prompt']!r}")
    cands = gather_candidates(spec)
    # сначала подходящие по заявленной длительности — их дешевле проверять
    lo, hi = spec.get("min_sec", 0.0), spec.get("max_sec", 1e9)
    cands.sort(key=lambda c: (not (lo <= c["duration"] <= hi), -min(c["duration"], AMBIENCE_MAX_SEC)))
    cap = 420.0 if kind == "ambience" else hi * 1.25
    cands = [c for c in cands if lo * 0.8 <= c["duration"] <= min(cap, hi * 1.25)]
    # Отклонённое ухом не предлагается заново: без этого пересборка вида
    # находит тот же файл по тому же запросу, и отказ человека живёт ровно
    # до следующего `build` — то есть не живёт.
    by_ear = rejected_ids_for(manifest, kind, name)
    if by_ear:
        before = len(cands)
        cands = [c for c in cands if c["id"] not in by_ear]
        if before != len(cands):
            print(f"   отклонённых ухом пропущено: {before - len(cands)}")
    blocked = [(c, title_blocked(name, c["title"])) for c in cands]
    for c, w in blocked:
        if w and rejected_log is not None:
            rejected_log.append({"id": c["id"], "title": c["title"], "kind": kind, "name": name,
                                 "url": c["url"], "creator": c.get("creator"),
                                 "landing": c.get("landing"), "reasons": [f"title:{w}"]})
    cands = [c for c, w in blocked if not w]
    cands.sort(key=lambda c: (-title_relevance(spec, c["title"]), -min(c["duration"], AMBIENCE_MAX_SEC)))
    cands = cands[:max_candidates]
    print(f"   кандидатов после фильтра длительности и названия: {len(cands)} "
          f"(отброшено по названию {sum(1 for _, w in blocked if w)})")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda c: download(c, "lq"), cands))
    scored = []
    for c, path in zip(cands, paths):
        if not path:
            continue
        v = evaluate(c, path, kind, name, spec)
        tag = "OK " if not v["reasons"] else "-- "
        print(f"   {tag}{c['title'][:44]:44s} {v.get('duration', 0):7.1f}с "
              f"margin={v.get('clap_margin', float('nan')):+.3f} "
              f"[{str(v.get('clap_worst_neg', ''))[:18]}] {','.join(v['reasons'])}")
        if rejected_log is not None and v["reasons"]:
            # url обязателен: без него отклонённого кандидата физически нечем
            # переслушать, а решение «гейт неправ» принимается только ушами.
            rejected_log.append(dict(v, kind=kind, name=name, url=c["url"],
                                     creator=c.get("creator"), landing=c.get("landing")))
        if not v["reasons"]:
            scored.append((v["clap_margin"], c, path, v))
    scored.sort(key=lambda t: -t[0])
    kept = []
    for margin, c, path, v in scored[:spec.get("keep", 3)]:
        safe = re.sub(r"[^a-z0-9]+", "_", c["id"].lower()).strip("_")
        dst = os.path.join(LIBRARY_ROOT, kind, name, f"{safe}.flac")
        hq = download(c, "hq") or path
        if import_file(hq, dst, kind, v["duration"]):
            kept.append(dst)
            if manifest is not None:
                manifest["items"][os.path.relpath(dst, ROOT)] = {
                    "kind": kind, "name": name, "source": c["source"], "id": c["id"],
                    "title": c["title"], "creator": c["creator"], "license": c["license"],
                    "license_url": c["license_url"], "url": c["url"], "landing": c["landing"],
                    "query": c["query"], "scores": v, "imported_at": int(time.time()),
                }
    print(f"   принято {len(kept)} из {len(cands)}")
    return kept


def kind_competition(path, want, spec_by_name):
    """Побеждает ли ЗАЯВЛЕННЫЙ вид среди всех остальных видов атмосферы плюс
    приманок (прибой, город)?

    Замена акустической части блоклиста по названиям. Признаковый разделитель
    (периодичность огибающей у прибоя, плотность ВЧ-транзиентов у дождя,
    НЧ-скос у ветра) проверен на этом же корпусе и НЕ работает: у прибоя
    периодичность 0.081 — НИЖЕ, чем у ветра (0.097-0.268) и леса
    (0.076-0.424); плотность транзиентов у дождя 0.007-0.018 — тот же
    диапазон, что у ветра, леса и реки. Конкуренция видов внутри CLAP на тех
    же файлах даёт 29 из 30 (единственный промах — моно-запись ветра,
    которую перебил прибой).

    Возвращает (победитель, отрыв_от_второго, {вид: скор}).
    """
    dur = probe_duration(path) or 0.0
    starts = ([max(0.0, min(dur - CLAP_WINDOW_SEC, f * dur - CLAP_WINDOW_SEC / 2))
               for f in CLAP_WINDOW_FRACS] if dur > CLAP_WINDOW_SEC + 1 else [0.0])
    wins = [decode_f32(path, st, min(CLAP_WINDOW_SEC, dur), 48000) for st in starts]
    wins = [w for w in wins if w.size > 4800]
    if not wins:
        return None, 0.0, {}
    names = list(spec_by_name) + list(KIND_DECOYS)
    prompts = [spec_by_name[n]["prompt"] for n in spec_by_name] + list(KIND_DECOYS.values())
    rows = clap_scores(wins, prompts)
    avg = [sum(r[i] for r in rows) / len(rows) for i in range(len(names))]
    order = sorted(range(len(names)), key=lambda i: -avg[i])
    winner = names[order[0]]
    gap = avg[order[0]] - avg[order[1]]
    return winner, round(gap, 4), {n: round(v, 4) for n, v in zip(names, avg)}


def _log_rejection(entry):
    """Дописать отказ в общий журнал, не затирая чужие записи."""
    path = os.path.join(LIBRARY_ROOT, "rejected.json")
    rows = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                prev = json.load(f)
            rows = prev["items"] if isinstance(prev, dict) else prev
        except Exception:
            rows = []
    rows = [v for v in rows if not (v.get("id") == entry.get("id")
                                    and v.get("name") == entry.get("name"))]
    rows.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


def verify(manifest):
    """Проверка библиотеки конкуренцией видов: запись, у которой выигрывает
    ЧУЖОЙ вид, удаляется. Отдельной командой, а не внутри build: конкуренция
    сравнивает вид со всеми остальными, то есть имеет смысл только когда
    словарь видов уже собран целиком."""
    spec_by_name = LIBRARY_SPEC["ambience"]
    dropped = 0
    for rel, it in list(manifest["items"].items()):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        # Частота исходника — гейт для ВСЕХ видов, и проверяется здесь же,
        # чтобы он подействовал на уже принятое, а не только на будущий
        # отбор: библиотека уже несла переход главы с исходника 16 кГц.
        if it.get("approved"):
            # Одобренное ухом не пересуживается НИКОГДА. Ровно этот случай и
            # стоил золотому набору кадра: человек посмотрел и сказал «годен»,
            # а дрейф версий модели через месяц передумал за него. Модель —
            # инструмент отбора кандидатов, вердикт человека выше её.
            print(f"  ++ {it['name']:14s} одобрено ухом, не пересуживается  "
                  f"{it['title'][:34]!r}")
            continue
        cached = download({"url": it["url"], "title": it.get("title", "")}, "hq")
        sr = probe_sample_rate(cached) if cached else 0
        if sr and sr < MIN_SOURCE_SAMPLE_RATE:
            print(f"  -- {it['name']:14s} исходник {sr} Гц  {it['title'][:38]!r}")
            _log_rejection(dict(it.get("scores", {}), kind=it["kind"], name=it["name"],
                                id=it["id"], title=it["title"], url=it["url"],
                                creator=it.get("creator"), landing=it.get("landing"),
                                source_sample_rate=sr, reasons=["low_sample_rate"]))
            os.remove(path)
            manifest["items"].pop(rel, None)
            dropped += 1
            continue
        if it.get("kind") != "ambience":
            continue
        winner, gap, scores = kind_competition(path, it["name"], spec_by_name)
        it["scores"]["kind_winner"] = winner
        it["scores"]["kind_gap"] = gap
        ok = winner == it["name"]
        print(f"  {'OK ' if ok else '-- '}{it['name']:14s} -> {str(winner):14s} отрыв {gap:+.3f}  {it['title'][:38]!r}")
        if not ok:
            # Удалять молча нельзя: это СПОРНЫЙ отказ (ветер по высокой траве
            # и правда похож на прибой), а спорное решается ушами. Запись
            # уходит в тот же журнал отказов, откуда её достаёт `audition`.
            _log_rejection(dict(
                it["scores"], kind=it["kind"], name=it["name"], id=it["id"],
                title=it["title"], url=it["url"], creator=it.get("creator"),
                landing=it.get("landing"), reasons=[f"kind_lost_to_{winner}"]))
            os.remove(path)
            manifest["items"].pop(rel, None)
            dropped += 1
    save_manifest(manifest)
    print(f"Удалено записей (чужой вид или низкая частота исходника): {dropped}")
    return 0


def reimport(manifest):
    """Переимпортировать библиотеку из кэша скачанного — без поиска и без
    моделей. Нужно, когда меняется ОБРАБОТКА при импорте (срез рокота,
    расширение моно, бесшовная петля), а отбор остаётся прежним."""
    done = 0
    for rel, it in manifest.get("items", {}).items():
        dst = os.path.join(ROOT, rel)
        src = download({"url": it["url"], "title": it.get("title", "")}, "hq")
        if src and import_file(src, dst, it["kind"], it["scores"].get("duration", 0.0)):
            done += 1
    print(f"Переимпортировано: {done}")
    return 0


def restore(manifest):
    """Воспроизвести библиотеку ПО МАНИФЕСТУ: скачать ровно те же файлы по
    тем же URL и импортировать с теми же нормировками — без поиска и без
    моделей. Нужно потому, что записи атмосферы (сотни МБ) в git не
    хранятся, а поисковая выдача со временем меняется: «собрать заново» дало
    бы другой набор, «восстановить» — тот же."""
    done = 0
    skipped = 0
    for rel, it in manifest.get("items", {}).items():
        dst = os.path.join(ROOT, rel)
        if os.path.exists(dst):
            continue
        if it.get("rejected"):
            # Отказ человека сильнее манифеста: запись физически отсутствует
            # именно потому, что её убрали ушами, и скачать её обратно по
            # сохранённому url значило бы отменить это решение молча.
            skipped += 1
            continue
        item = {"url": it["url"], "title": it.get("title", "")}
        path = download(item, "hq")
        if path and import_file(path, dst, it["kind"], it["scores"].get("duration", 0.0)):
            done += 1
            print(f"   восстановлен {rel}")
    print(f"Восстановлено файлов: {done}"
          + (f"; пропущено отклонённых ухом: {skipped}" if skipped else ""))
    return 0


# Причины отказа, которые НЕ обсуждаются ушами: это измеренные дефекты
# самой записи (сетевой гул, клиппинг, провалы в тишину, чужой источник по
# названию, речь/музыка поверх сцены). Спорны только два ОТНОСИТЕЛЬНЫХ
# сигнала CLAP — они про «про то ли это», а на этот вопрос честнее отвечает
# слух, чем ещё один подобранный порог.
AUDITION_DEBATABLE = ("clap_negative_wins",)
AUDITION_DEBATABLE_PREFIX = ("kind_lost_to_",)


def _debatable(reason):
    return reason in AUDITION_DEBATABLE or reason.startswith(AUDITION_DEBATABLE_PREFIX)


def audition(kind, name, limit=8):
    """Сложить отклонённых «почти прошедших» в temp_library/audition, чтобы
    их можно было ПОСЛУШАТЬ, а не двигать порог вслепую.

    Берутся только кандидаты, у которых все причины отказа — спорные
    (см. AUDITION_DEBATABLE). Запись с сетевым гулом сюда не попадает: её
    дефект измерен, а не предположен, и слушать там нечего.

    Обработка — ТА ЖЕ import_file, что у принятых: слушается то, что реально
    ушло бы в ролик (срез рокота, расширение моно, бесшовная петля), а не
    исходник со стока.
    """
    path = os.path.join(LIBRARY_ROOT, "rejected.json")
    if not os.path.exists(path):
        print("нет assets/library/rejected.json — сначала build")
        return 1
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    rows = rows["items"] if isinstance(rows, dict) else rows
    near = [v for v in rows
            if v.get("name") == name and v.get("url")
            and v.get("reasons") and all(_debatable(r) for r in v["reasons"])]
    # самые близкие к порогу — первыми: если неправ порог, ошибка именно здесь
    near.sort(key=lambda v: -(v.get("clap_margin") or -9.0))
    out = os.path.join(CACHE_DIR, "audition", kind, name)
    os.makedirs(out, exist_ok=True)
    done = 0
    for i, v in enumerate(near[:limit], 1):
        src = download({"url": v["url"], "title": v.get("title", "")}, "hq")
        if not src:
            continue
        safe = re.sub(r"[^a-z0-9]+", "_", str(v.get("id", i)).lower()).strip("_")
        dst = os.path.join(out, f"{i:02d}_{safe}.flac")
        if import_file(src, dst, kind, v.get("duration", 0.0)):
            done += 1
            print(f"   {i:02d} margin={v.get('clap_margin'):+.4f} "
                  f"проиграл={str(v.get('clap_worst_neg',''))[:26]!r} {v.get('title','?')[:44]}")
    print(f"На прослушивание отложено {done} из {len(near)} спорных: {out}")
    return 0


LOCAL_AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a")


def ingest_dir(manifest, src_dir, kind, name, licence, licence_url, limit=0,
               credit=None, allow_ai_embeddings=True):
    """Проиндексировать ЛОКАЛЬНУЮ папку теми же гейтами, что и сток.

    Написано под профессиональные пакеты, которые раздаются целиком и без
    API — Sonniss GDC, 99Sounds, Kenney, секция эффектов YouTube Audio
    Library. Там настоящие WAV вместо превью 128 kbps, но скачивание
    ручное, поэтому путь к файлам даёт человек.

    Гейты те же самые, не облегчённые: замеры (гул, клиппинг, тишина,
    динамика, частота исходника), CLAP-маржа против ловушек, вето AST,
    конкуренция видов. Пакет от профессиональной студии не освобождает от
    проверки «про то ли это»: в библиотеке ветра лежит и прибой, и он так
    же не подойдёт главе про поле, как и любительская запись.

    Лицензия НЕ угадывается по файлу и не берётся из имени папки — её
    называет человек флагом, и она попадает в манифест на каждую запись.
    Молча проставить «cc0» чему угодно локальному было бы ровно тем
    молчаливым допущением, от которого fail-closed проверка защищает сток.
    """
    files = []
    for root, _, fs in os.walk(src_dir):
        for f in sorted(fs):
            if f.lower().endswith(LOCAL_AUDIO_EXT):
                files.append(os.path.join(root, f))
    if not files:
        print(f"в {src_dir} не найдено аудио ({', '.join(LOCAL_AUDIO_EXT)})")
        return 1
    spec = dict(LIBRARY_SPEC[kind][name], _name=name)
    lo, hi = spec.get("min_sec", 0.0), spec.get("max_sec", 1e9)
    print(f"\n== {kind}/{name} из {src_dir}: файлов {len(files)}, лицензия {licence}")
    scored = []
    for i, f in enumerate(files, 1):
        d = probe_duration(f) or 0.0
        if not (lo * 0.8 <= d <= min(420.0 if kind == "ambience" else hi * 1.25, hi * 1.25)):
            continue
        w = title_blocked(name, os.path.basename(f))
        if w:
            print(f"   -- {os.path.basename(f)[:44]:46s} название: {w}")
            continue
        v = judge(measure(f, kind, spec), kind, spec)
        tag = "OK " if not v["reasons"] else "-- "
        print(f"   {tag}{os.path.basename(f)[:44]:46s} {v.get('duration',0):7.1f}с "
              f"margin={v.get('clap_margin', float('nan')):+.3f} {','.join(v['reasons'])}")
        if not v["reasons"]:
            scored.append((v["clap_margin"], f, v))
        if limit and len(scored) >= limit:
            break
    scored.sort(key=lambda t: -t[0])
    kept = 0
    for margin, f, v in scored[:spec.get("keep", 3)]:
        safe = re.sub(r"[^a-z0-9]+", "_", os.path.splitext(os.path.basename(f))[0].lower()).strip("_")
        dst = os.path.join(LIBRARY_ROOT, kind, name, f"local_{safe[:40]}.flac")
        if not import_file(f, dst, kind, v["duration"]):
            continue
        manifest["items"][os.path.relpath(dst, ROOT)] = {
            "kind": kind, "name": name, "source": "local",
            "id": "local:" + hashlib.sha1(f.encode()).hexdigest()[:16],
            "title": os.path.basename(f), "creator": credit,
            "license": licence, "license_url": licence_url,
            "url": None, "landing": None, "query": os.path.basename(src_dir),
            "source_path": f, "scores": v, "imported_at": int(time.time()),
            # Некоторые пакеты (Sonniss) прямо запрещают ОБУЧЕНИЕ на своих
            # звуках. Инференс ради отбора под запрет не попадает, но флаг
            # едет с записью, чтобы вопрос «можно ли публиковать её
            # эмбеддинги» имел ответ в данных, а не в чьей-то памяти.
            "allow_ai_embeddings": bool(allow_ai_embeddings),
        }
        kept += 1
    print(f"   принято {kept} из {len(files)}")
    save_manifest(manifest)
    return 0


def _match_items(manifest, patterns):
    """Записи манифеста по куску пути/id/названия. Пусто — ничего не нашлось,
    и это не молчаливый успех: вызывающий код обязан это сказать."""
    out = []
    for rel, it in manifest["items"].items():
        hay = " ".join([rel, str(it.get("id", "")), str(it.get("title", ""))]).lower()
        if any(pat.lower() in hay for pat in patterns):
            out.append((rel, it))
    return out


def set_approved(manifest, patterns, value=True):
    """Проставить вердикт человека. Это единственное место, где он ставится:
    автоматика подбирает кандидатов, человек подтверждает, дальше запись не
    пересуживается ни пересборкой вида, ни verify, ни дрейфом версий моделей.
    """
    hits = _match_items(manifest, patterns)
    if not hits:
        print("ничего не найдено по: " + ", ".join(patterns))
        return 1
    for rel, it in hits:
        it["approved"] = bool(value)
        it["approved_at"] = int(time.time()) if value else None
        print(f"   {'одобрено' if value else 'снято одобрение'}: "
              f"{it['kind']}/{it['name']}  {it.get('title','?')[:44]}")
    save_manifest(manifest)
    print(f"Записей затронуто: {len(hits)}")
    return 0


def set_rejected(manifest, patterns, value=True):
    """Вердикт человека «НЕ брать» — зеркало set_approved и такой же по силе:
    запись с `rejected` не возвращается ни `restore` (не скачивается обратно
    по своему же url), ни пересборкой вида (`build_kind` пропускает кандидата
    с тем же id), и переживает пересборку в манифесте.

    РЕАЛЬНЫЙ ДЕФЕКТ, ради которого это заведено — найден живой проверкой
    17.09, не чтением кода. Владелец послушал кандидатов и попросил убрать
    часть звуков; отказ был выражен ПЕРЕМЕЩЕНИЕМ файла в подпапку, то есть в
    данных не записан нигде. `restore()` пропускает только СУЩЕСТВУЮЩИЕ по
    своему пути файлы, а у всех девяти перемещённых записей в манифесте
    остался url — то есть одна штатная команда восстановления молча вернула
    бы отклонённые звуки в ротацию. Для атмосферы это не гипотетика: она не
    хранится в git и на машине владельца восстанавливается именно `restore`.

    Асимметрия была ровно в том, что «человек сказал ДА» жило в данных
    (`approved`), а «человек сказал НЕТ» — только в расположении файла на
    диске, которого ни один из механизмов восстановления не видит.
    """
    hits = _match_items(manifest, patterns)
    if not hits:
        print("ничего не найдено по: " + ", ".join(patterns))
        return 1
    for rel, it in hits:
        it["rejected"] = bool(value)
        it["rejected_at"] = int(time.time()) if value else None
        print(f"   {'отклонено ухом' if value else 'снят отказ'}: "
              f"{it['kind']}/{it['name']}  {it.get('title','?')[:44]}")
    save_manifest(manifest)
    print(f"Записей затронуто: {len(hits)}")
    return 0


def rejected_ids_for(manifest, kind, name):
    """id записей, которые человек отклонил для ЭТОГО вида. Нужен build_kind:
    без него пересборка находит тот же файл по тому же запросу и возвращает
    его, то есть отказ живёт ровно до следующего `build`."""
    return {it.get("id") for it in (manifest or {}).get("items", {}).values()
            if it.get("rejected") and it.get("kind") == kind and it.get("name") == name
            and it.get("id")}


def promote(manifest, kind, name, numbers):
    """Перевести спорную запись из temp_library/audition в библиотеку — по
    номеру, который печатает `audition` и который слышен на демо-ленте.

    Запись попадает СРАЗУ одобренной: она пришла сюда именно потому, что
    гейт её не пропустил, и единственное основание взять её — что человек
    послушал. Без `approved` следующий же verify выкинул бы её обратно тем
    же гейтом, и круг замкнулся бы.
    """
    src_dir = os.path.join(CACHE_DIR, "audition", kind, name)
    if not os.path.isdir(src_dir):
        print(f"нет отложенных на прослушивание: {src_dir}")
        return 1
    path = os.path.join(LIBRARY_ROOT, "rejected.json")
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
        rows = prev["items"] if isinstance(prev, dict) else prev
    near = [v for v in rows if v.get("name") == name and v.get("url")
            and v.get("reasons") and all(_debatable(r) for r in v["reasons"])]
    near.sort(key=lambda v: -(v.get("clap_margin") or -9.0))
    done = 0
    for n in numbers:
        if n < 1 or n > len(near):
            print(f"   номера {n} нет (всего спорных {len(near)})")
            continue
        v = near[n - 1]
        src = download({"url": v["url"], "title": v.get("title", "")}, "hq")
        if not src:
            print(f"   {n}: не скачалось")
            continue
        safe = re.sub(r"[^a-z0-9]+", "_", str(v.get("id", n)).lower()).strip("_")
        dst = os.path.join(LIBRARY_ROOT, kind, name, f"{safe}.flac")
        if not import_file(src, dst, kind, v.get("duration", 0.0)):
            print(f"   {n}: не импортировалось")
            continue
        manifest["items"][os.path.relpath(dst, ROOT)] = {
            "kind": kind, "name": name, "source": "audition",
            "id": v.get("id"), "title": v.get("title", "?"),
            "creator": v.get("creator"), "license": "cc0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "url": v["url"], "landing": v.get("landing"), "query": v.get("query"),
            "scores": {k: v[k] for k in v if k not in ("kind", "name", "url")},
            "imported_at": int(time.time()),
            "approved": True, "approved_at": int(time.time()),
            "approved_note": "переведено из audition: гейт отклонил, человек послушал",
        }
        done += 1
        print(f"   {n}: взято — {v.get('title','?')[:46]}")
    save_manifest(manifest)
    print(f"Переведено в библиотеку: {done}")
    return 0 if done else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "report", "restore", "reimport", "verify",
                                    "audition", "approve", "unapprove", "reject", "unreject",
                                    "promote", "ingest"])
    ap.add_argument("--dir", default="", help="ingest: папка с локальными файлами")
    ap.add_argument("--license", default="", help="ingest: лицензия пакета, называет человек")
    ap.add_argument("--license-url", default="", help="ingest: ссылка на текст лицензии")
    ap.add_argument("--credit", default="", help="ingest: правообладатель пакета")
    ap.add_argument("--no-ai-embeddings", action="store_true",
                    help="ingest: пакет запрещает обучение ИИ — пометить записи")
    ap.add_argument("--match", default="", help="approve/unapprove: куски пути/id/названия через запятую")
    ap.add_argument("--take", default="", help="promote: номера из audition через запятую")
    ap.add_argument("--kinds", default="", help="kind:name через запятую; пусто = всё")
    ap.add_argument("--max", type=int, default=30)
    args = ap.parse_args()
    manifest = load_manifest()
    if args.cmd == "restore":
        return restore(manifest)
    if args.cmd == "reimport":
        return reimport(manifest)
    if args.cmd == "verify":
        return verify(manifest)
    if args.cmd in ("approve", "unapprove", "reject", "unreject"):
        pats = [x.strip() for x in args.match.split(",") if x.strip()]
        if not pats:
            print("нужен --match")
            return 1
        if args.cmd in ("reject", "unreject"):
            return set_rejected(manifest, pats, args.cmd == "reject")
        return set_approved(manifest, pats, args.cmd == "approve")
    if args.cmd == "ingest":
        k = [x.strip() for x in args.kinds.split(",") if x.strip()]
        if not args.dir or len(k) != 1 or not args.license:
            print("нужно: --dir <папка> --kinds <kind>:<вид> --license <лицензия> "
                  "[--license-url ...] [--credit ...] [--no-ai-embeddings]")
            return 1
        kind, _, name = k[0].partition(":")
        return ingest_dir(manifest, args.dir, kind or "ambience", name,
                          args.license, args.license_url or None, args.max,
                          args.credit or None, not args.no_ai_embeddings)
    if args.cmd == "promote":
        k = [x.strip() for x in args.kinds.split(",") if x.strip()]
        nums = [int(x) for x in args.take.split(",") if x.strip().isdigit()]
        if len(k) != 1 or not nums:
            print("нужен --kinds ambience:<вид> и --take 1,3,5")
            return 1
        kind, _, name = k[0].partition(":")
        return promote(manifest, kind or "ambience", name, nums)
    if args.cmd == "audition":
        rc = 0
        for k in (args.kinds.split(",") if args.kinds.strip() else []):
            kind, _, name = k.partition(":")
            rc |= audition(kind or "ambience", name, args.max)
        if not args.kinds.strip():
            print("audition требует --kinds ambience:<вид>")
            return 1
        return rc
    if args.cmd == "report":
        for rel, it in sorted(manifest["items"].items()):
            print(f"{rel:60s} {it['scores'].get('duration', 0):7.1f}с  margin {it['scores'].get('clap_margin'):+.3f}  {it['title'][:50]!r}")
        return 0
    wanted = [tuple(k.split(":", 1)) for k in args.kinds.split(",") if k.strip()] or \
             [(k, n) for k in LIBRARY_SPEC for n in LIBRARY_SPEC[k]]
    # Отклонённые НАКАПЛИВАЮТСЯ между прогонами, как и манифест. Первая
    # версия заводила пустой список на каждый запуск, и пересборка одного
    # вида стирала причины отказа у всех остальных — то есть на вопрос
    # «почему эта запись не прошла» ответа не оставалось нигде.
    rejected_path = os.path.join(LIBRARY_ROOT, "rejected.json")
    rejected = []
    if os.path.exists(rejected_path):
        try:
            with open(rejected_path, encoding="utf-8") as f:
                prev = json.load(f)
            rejected = prev["items"] if isinstance(prev, dict) else prev
        except Exception:
            rejected = []
    drop = {(k, n) for k, n in wanted}
    rejected = [v for v in rejected if (v.get("kind"), v.get("name")) not in drop]
    for kind, name in wanted:
        # Пересборка вида — с чистого листа, НО одобренное ухом не трогается.
        # Всё остальное уходит, иначе «urban»-парк остался бы в лесу рядом с
        # новыми, а манифест хранил бы записи об удалённых файлах.
        keep = {rel for rel, it in manifest["items"].items()
                if it.get("kind") == kind and it.get("name") == name and it.get("approved")}
        # Отказ ухом обязан пережить пересборку ТАК ЖЕ, как одобрение: иначе
        # запись выпадает из манифеста, вместе с ней исчезает сам факт
        # отказа, и следующий же поиск предлагает этот файл заново.
        keep_verdict = {rel for rel, it in manifest["items"].items()
                        if it.get("kind") == kind and it.get("name") == name
                        and (it.get("approved") or it.get("rejected"))}
        keep_names = {os.path.basename(r) for r in keep}
        d = os.path.join(LIBRARY_ROOT, kind, name)
        for f in (os.listdir(d) if os.path.isdir(d) else []):
            p = os.path.join(d, f)
            # Пересборка удаляет ФАЙЛЫ вида, а не что попало: подпапка рядом
            # с записями (например отложенное на переслушивание) роняла весь
            # прогон IsADirectoryError — воспроизведено, не предположено.
            if f not in keep_names and os.path.isfile(p):
                os.remove(p)
        manifest["items"] = {rel: it for rel, it in manifest["items"].items()
                             if rel in keep_verdict or not (it.get("kind") == kind and it.get("name") == name)}
        if keep:
            print(f"   одобренных сохранено: {len(keep)}")
        if len(keep_verdict) > len(keep):
            print(f"   отказов ухом сохранено: {len(keep_verdict) - len(keep)}")
        build_kind(kind, name, args.max, manifest, rejected)
        save_manifest(manifest)
        with open(rejected_path, "w", encoding="utf-8") as f:
            json.dump(rejected, f, ensure_ascii=False, indent=1)
    print(f"\nГотово. Манифест: {MANIFEST_PATH}; отклонённые с причинами: assets/library/rejected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
