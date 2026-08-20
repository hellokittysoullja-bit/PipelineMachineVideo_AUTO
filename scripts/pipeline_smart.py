#!/usr/bin/env python3
"""Главная сборка по паузам (ЧАСТЬ 6, 9, 13-шаг7).
Разбивает script.txt по [pause] на смысловые блоки, считает тайминг
(слова+паузы, масштаб под реальную длину аудио), Ken Burns на всю длину
кадра, чередование in/out. Медиа: локальная папка media/ по порядку,
fallback — Pexels по тематическому запросу.
Usage: python scripts/pipeline_smart.py <video_dir>"""
import csv
import difflib
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import numpy as np
except ImportError:
    np = None   # аудио-ритм по громкости — опциональная фича, без numpy просто выключена

from PIL import Image as PILImage   # Pillow уже обязательная зависимость (requirements.txt)

try:
    import cv2
    PARALLAX_LIBS = True
except ImportError:
    PARALLAX_LIBS = False   # 2.5D-параллакс — опциональная фича (torch/transformers/opencv тяжёлые)

PARALLAX_ENABLED = os.environ.get("PARALLAX", "1") != "0" and PARALLAX_LIBS
PARALLAX_BROKEN = False   # взводится только на системном сбое (модель/сеть), не на одной плохой картинке

VIDEO_FOLDER = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
SCRIPT_FILE = os.path.join(VIDEO_FOLDER, "script.txt")
MEDIA_FOLDER = os.path.join(VIDEO_FOLDER, "media")
OUTPUT_FILE = os.path.join(VIDEO_FOLDER, "final.mp4")
TEMP_FOLDER = os.path.join(VIDEO_FOLDER, "temp_smart")

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
FPS, WIDTH, HEIGHT = 25, 1920, 1080
# Явные цветовые метаданные — без них YouTube на части устройств домысливает
# цветовое пространство сам и уходит в грязь/перенасыщение (особенно заметно
# на тёмном грейде — а у нас он тёмный). Добавляется на КАЖДЫЙ libx264-выход
# (промежуточные клипы и оба финальных прохода) — дешёвая гигиена, не
# косметика: без неё цвет непредсказуем именно там, где мы больше всего
# вложились в грейд/зерно/halation.
COLOR_META_ARGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
# 10-бит ТОЛЬКО на промежуточных клипах (kenburns/video_render/
# parallax_kenburns) — у нас две генерации перекодирования (клип -> общий
# xfade-проход), и 8-бит на КАЖДОЙ из них копит собственную ошибку
# квантования (видно на тёмных градиентах — deband лечит следствие, не
# причину). xfade_chain() и pad_to_length() ОСТАЮТСЯ на yuv420p (8-бит) —
# это финальный формат поставки (-c:v copy на финальном муксе копирует
# ИМЕННО их выход без изменений), 10-бит H.264 не универсально безопасен
# для доставки на YouTube. ffmpeg сам понижает точность при декодировании
# 10-битных клипов в 8-битный xfade-выход — обычный смешанный pipeline,
# ничего специального делать не нужно на стороне master-прохода.
CLIP_PIX_ARGS = ["-pix_fmt", "yuv420p10le", "-profile:v", "high10"]

# ZOOM_FLOOR — минимальный зум держится ВЕСЬ клип (не 1.0). Раньше offset пана
# был обязан = 0 ровно в момент zoom=1.0 (иначе край вылезет за картинку), и на
# каждом втором клипе (zoom-out) кадр половину времени стоял мёртвым по центру.
# С постоянным полом запас под пан есть на любом кадре клипа, а не только к
# концу движения зума.
ZOOM_FLOOR = 1.04
# Скорость зума в долях/сек, а не фиксированным приростом на клип — иначе
# 4-секундный и 20-секундный кадр "дышат" с заметно разной скоростью.
ZOOM_RATE_BASE = 0.010
ZOOM_DELTA_MIN, ZOOM_DELTA_MAX = 0.05, 0.22
MIN_CLIP, MAX_CLIP = 3.0, 20.0
# Раньше MIN_CLIP=4.0 был ЕДИНЫМ полом на весь ролик, включая хук — на
# реальном эпизоде это дало 0% клипов короче 3с и хук со средней длиной
# кадра 4.65с, почти не отличающейся от тела (6.8с). У каналов с высоким
# удержанием первые 15-30с режутся заметно чаще именно потому, что это
# самый крутой обрыв на кривой удержания YouTube — отдельный, более
# низкий пол специально для хука.
HOOK_MIN_CLIP = 1.8
# Топовые ролики первые 2-3 реза хука делают ЕЩЁ короче, чем весь остальной
# хук (0.8-1.2с) — самый первый удар по вниманию, а не просто "быстрее тела
# ролика". Общий HOOK_MIN_CLIP держал даже открывающие кадры вровень с
# остальным хуком.
HOOK_OPENING_MIN_CLIP = 1.0
HOOK_OPENING_CUTS = 3
# Панорамирование считается от РЕАЛЬНОГО zoom в каждый момент кадра — (1-1/zoom)/2
# это точный геометрический запас смещения без вылета за картинку, безопасно по
# построению на любом кадре, поэтому PAN_SAFETY можно брать ближе к пределу, чем
# раньше (когда запас считался один раз заранее от конечного zoom клипа).
PAN_SAFETY = 0.9
PAN_JITTER_MIN, PAN_JITTER_MAX = 0.6, 1.0   # органический разброс силы пана по клипам
PAN_DIRECTIONS = [(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1)]
# Один и тот же грейд на все 200+ кадров — тоже штамп, если приглядеться.
# Лёгкий hash-джиттер контраста/сатурации/яркости на каждый клип убирает
# эту одинаковость, оставаясь в узком диапазоне (не превращается в разнобой).
# Базовый грейд — не нейтральный "как есть у стока": сатурация чуть прибрана,
# тени уводим в холод, света чуть тёплые (лёгкий teal-orange) — читается как
# военно-историческая документалка, а не разноцветный слайд-шоу из Pexels.
# MOOD_GRADE — тот же язык грейда, но с сдвигом под секцию: хук резче и
# холоднее (цепляет), тело — базовая "стальная" документалка, финал теплее
# и мягче (спад напряжения). LUT-переключение по секции без внешних .cube
# файлов — тот же eq/colorbalance, просто другие опорные точки.
MOOD_GRADE = {
    "HOOK":  {"c0": 1.10, "s0": 0.82, "vign": "PI/4.2", "bs": 0.13, "rh": 0.02},
    "FINAL": {"c0": 1.02, "s0": 0.90, "vign": "PI/6",   "bs": 0.06, "rh": 0.08},
    "BODY":  {"c0": 1.06, "s0": 0.86, "vign": "PI/5",   "bs": 0.10, "rh": 0.03},
}


def film_look(photo_hash, section="", brightness_bias=0.0, energy_bias=0.0, levels=None):
    """brightness_bias — сглаживание скачка экспозиции со СОСЕДНИМ кадром
    (см. measure_luma()/main(): сток из разных источников иначе скачет по
    яркости кадр-к-кадру — частый "любительский" tell в компиляциях).
    energy_bias — лёгкая пульсация контраста/сатурации в такт громкости
    голоса в ЭТОМ отрезке (energy_levels()) — на эмоциональных пиках кадр
    чуть "подбирается", на тихих местах чуть мягче, как это интуитивно
    делает монтажёр руками, а не по одной и той же формуле весь ролик.

    levels — (чёрная,белая) точка ЭТОГО кадра по перцентилям гистограммы
    (см. measure_levels()) для адаптивной ТЕХНИЧЕСКОЙ нормализации
    экспозиции — отдельный, самый первый шаг цепочки (auto_levels_params(),
    клэмп силы — не полное выравнивание), решает другую задачу, чем
    brightness_bias (тот сглаживает скачок МЕЖДУ соседними кадрами, этот
    поднимает разброс экспозиции ВНУТРИ одного кадра к общей базе). Сам
    творческий грейд ниже (eq/curves/selectivecolor/...) НЕ адаптируется и
    не меняется от кадра к кадру — только то, что в него попадает."""
    mood = MOOD_GRADE["HOOK"] if section.startswith("HOOK") else (
           MOOD_GRADE["FINAL"] if section.startswith("FINAL") else MOOD_GRADE["BODY"])
    eb = max(-0.5, min(0.5, energy_bias))
    c = mood["c0"] + (photo_hash % 100) / 100 * 0.05 + eb * 0.06
    s = mood["s0"] + ((photo_hash >> 7) % 100) / 100 * 0.08 + eb * 0.05
    b = ((photo_hash >> 14) % 100) / 100 * 0.02 + brightness_bias
    al_c, al_b = auto_levels_params(levels)
    norm_stage = f"eq=contrast={al_c:.4f}:brightness={al_b:.4f}," if (al_c != 1.0 or al_b != 0.0) else ""
    return (
        norm_stage +
        f"eq=contrast={c:.3f}:saturation={s:.3f}:brightness={b:.3f},"
        # Тон-кривая: мягкий roll-off светов и лёгкий подъём теней вместо
        # жёсткого клиппинга голым eq — "мёртвый" digital-белый (255) и
        # проваленный чёрный (0) читаются дёшево, плёнка так не работает.
        # Общая для всех мудов — это базовая физика тона, не настроение
        # (то несут eq/colorbalance/vignette).
        f"curves=master='0/0.015 0.25/0.22 0.5/0.5 0.75/0.78 1/0.965',"
        # Избирательный цвет: зелень (фон/трава на реконструкциях) чуть
        # приглушена, жёлто-охристые тона (дерево/кожа/бронза) чуть теплее —
        # конкретные диапазоны, не "поднять сатурацию на всём".
        f"selectivecolor=greens=0.02 0.04 -0.02 0.02:yellows=-0.03 0 0.02 -0.02,"
        f"colorbalance=rs=-0.06:bs={mood['bs']:.3f}:rm=-0.02:bm=0.04:rh={mood['rh']:.3f}:bh=-0.02,"
        f"vignette={mood['vign']},"
        # Halation — ТОЛЬКО вокруг настоящих ярких источников (внутренняя
        # curves обнуляет всё темнее ~0.6, остаются только пересветы), тёплый
        # красно-оранжевый оттенок самого свечения (настоящий halation
        # цветной, не белый). opacity=0.09 — на грани восприятия: сильнее
        # уже читается как "наложенный эффект", не как деталь съёмки
        # (проверено вживую: mean diff ~3% на реальном фото, не доминирует
        # над общим грейдом). split/blend без хвостового лейбла — остаётся
        # простой implicit-цепочкой, unsharp/noise ниже продолжают её как
        # обычно, совместимо с embedding внутри больших filter_complex
        # (video_render ramp-путь, parallax rawvideo pipe).
        f"split[fl_hi_a][fl_hi_b];"
        f"[fl_hi_a]curves=all='0/0 0.6/0 0.8/0.22 1/0.6',"
        f"colorbalance=rs=0.12:bs=-0.12,gblur=sigma=14[fl_hi_glow];"
        f"[fl_hi_b][fl_hi_glow]blend=all_mode=screen:all_opacity=0.09,"
        # deband — тёмный тёплый грейд + vignette + halation усиливают риск
        # полос на гладких градиентах (небо, размытый фон, тень от vignette) —
        # то, что 8-бит легче всего выдаёт как "цифровое". Мягкие пороги
        # (0.04, не дефолтные 0.02) — заметно сглаживает тёмные плавные
        # участки (проверено вживую: ~5/255 средняя разница в затемнённом
        # vignette-углу через реальный libx264 CRF18 энкод), не размывает
        # текстуру/детали.
        f"deband=range=22:1thr=0.04:2thr=0.04:3thr=0.04:4thr=0.04,"
        # unsharp — резкость после даунскейла с 8000px-холста в 1920 (тот
        # сам по себе мягкий) — ощущение "живого" объектива. Зерно (grain)
        # СЮДА больше не входит — оно теперь настоящий процедурный видео-луп
        # (assets/grain/grain_loop.mp4, см. generate_grain_asset.py и
        # apply_grain()), подмешивается ОТДЕЛЬНЫМ шагом в каждой рендер-
        # функции (нужен второй ffmpeg-вход, filter_complex этого не
        # позволяет сделать внутри одной строки-чейна). Голый noise=alls=3
        # был плоским per-pixel шумом — у настоящего зерна структура
        # сгустками и посекундная смена, это и даёт grain_loop.
        f"unsharp=5:5:0.45:5:5:0.0"
    )


# Процедурный grain-луп (см. scripts/generate_grain_asset.py) — статический
# ассет в репозитории, тот же принцип, что assets/fonts. Нет файла (например,
# ещё не сгенерирован) -> GRAIN_ENABLED=False, весь рендер тихо откатывается
# на путь без зерна (безопасный откат, тот же принцип, что PARALLAX_LIBS).
GRAIN_LOOP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "assets", "grain", "grain_loop.mp4")
GRAIN_ENABLED = os.environ.get("GRAIN", "1") != "0" and os.path.exists(GRAIN_LOOP_PATH)
GRAIN_OPACITY = 0.10   # калибровано вживую — см. коммит с проверкой

# Процедурная атмосферная подложка (см. scripts/generate_music_asset.py) —
# тот же принцип, что и с grain_loop: статический ассет в репозитории, нет
# файла -> MUSIC_ENABLED=False, безопасный откат на прежнее поведение
# (только голос, без подложки).
_MUSIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "music")
# A2: три варианта настроения — та же логика секций, что MOOD_GRADE у
# визуального грейда (HOOK холоднее/острее, BODY нейтральная документалка,
# FINAL теплее/мягче). BODY — обязательный файл (используется как основной
# индикатор MUSIC_ENABLED и как fallback для HOOK/FINAL, если их отдельные
# файлы почему-то не сгенерированы).
MUSIC_BED_PATH = os.path.join(_MUSIC_DIR, "ambient_bed.flac")
MUSIC_BED_PATHS = {
    "HOOK": os.path.join(_MUSIC_DIR, "ambient_bed_hook.flac"),
    "BODY": MUSIC_BED_PATH,
    "FINAL": os.path.join(_MUSIC_DIR, "ambient_bed_final.flac"),
}
MUSIC_ENABLED = os.environ.get("MUSIC_BED", "1") != "0" and os.path.exists(MUSIC_BED_PATH)
# J-cut: смена настроения подложки опережает смену секции сценария на
# столько секунд — звук "приходит первым", смысловой сдвиг подложки уже
# ощущается ДО того, как текст/картинка формально сменили раздел (тот же
# приём, что в проф. монтаже: звук режется раньше картинки).
MUSIC_JCUT_LEAD = 2.5
MUSIC_MOOD_XFADE = 4.0   # кроссфейд между настроениями — медленный, подложка не должна "щёлкать" сменой
# Доп. затухание ПОВЕРХ собственных -12dBFS ассета — фон должен быть заметно
# тише голоса даже ДО сайдчейн-дакинга (дакинг — страховка на явных пиках
# речи, не единственная линия обороны против "музыка спорит с закадром").
MUSIC_BED_GAIN_DB = -13.0
# Порог в ЛИНЕЙНОЙ амплитуде (ffmpeg sidechaincompress, не дБ): реальный
# голос в проекте — mean -16dB/max -1.7dB (см. volumedetect на audio_fixed.mp3),
# 0.06 ~= -24dB — уверенно ловит речь, не ловит фоновую тишину/дыхание между
# фразами (там подложка может чуть подняться — естественное "дыхание" мастера,
# как в живом монтаже, а не плоская постоянная громкость).
MUSIC_DUCK_THRESHOLD = 0.06
MUSIC_DUCK_RATIO = 9.0
MUSIC_DUCK_ATTACK_MS = 15.0    # быстро прижать к началу фразы
MUSIC_DUCK_RELEASE_MS = 450.0  # плавно отпустить после — без эффекта "накачки"

# A6: цепочка обработки самого голоса (сырой TTS -> вещательного качества),
# ДО подмешивания музыки/дакинга — то, что обычно делает звукорежиссёр с
# дорожкой диктора: срез суб-баса (в голосе его быть не должно, только гул),
# лёгкая тональная коррекция (тепло внизу, разборчивость наверху — TTS
# нейтрален "из коробки", не спроектирован под смесь с музыкой), деэссер
# (TTS-движки нередко дают резкие "с/ш" на определённых голосах/скоростях),
# мягкая компрессия РОВНО голоса (не путать с loudnorm — тот выравнивает
# ГРОМКОСТЬ ролика целиком, это — микро-динамика внутри фраз, чтобы тихие
# слова не тонули под подложкой ДО дакинга).
VOICE_PROCESS_ENABLED = os.environ.get("VOICE_PROCESS", "1") != "0"
VOICE_HIGHPASS_HZ = 80
VOICE_EQ_WARMTH_HZ, VOICE_EQ_WARMTH_GAIN = 200, 1.5
VOICE_EQ_PRESENCE_HZ, VOICE_EQ_PRESENCE_GAIN = 3000, 2.0
VOICE_DEESS_INTENSITY = 0.3
VOICE_COMPRESS_THRESHOLD = 0.15
VOICE_COMPRESS_RATIO = 2.5


def grain_blend_complex(label_in, grain_input_idx, label_out):
    """Фрагмент filter_complex: масштабирует зацикленный (-stream_loop -1 на
    входе grain_input_idx) grain-луп под WIDTH/HEIGHT и подмешивает его к
    label_in через grainmerge (штатный blend-режим ffmpeg именно под
    добавление зерна: result = base + overlay - 128 — естественно слабее у
    самых крайних светов/теней за счёт клиппинга, без отдельной яркостной
    маски)."""
    return (f"[{grain_input_idx}:v]scale={WIDTH}:{HEIGHT}:flags=bicubic,setsar=1[gr_scaled];"
            f"[{label_in}][gr_scaled]blend=all_mode=grainmerge:all_opacity={GRAIN_OPACITY}[{label_out}]")


XFADE_DUR = 0.4        # диссолв на границах секций и часть обычных склеек
XFADE_DUR_HARD = 0.06  # почти мгновенный переход — читается как жёсткий cut
# hblur/hlwind/hrwind (смаз в движении — читается как whip pan, разные
# направления — не один и тот же смаз на каждой склейке) и zoomin
# (панч-переход) — одна мотивированная категория "движение камеры", не
# декоративные геометрические wipe (проверено вживую рендером — pixelize/
# circleopen/squeeze и т.п. читаются как шаблон видеоредактора, не приём
# оператора, поэтому НЕ в пуле). fadewhite/fadegrays — вспышка светом и
# тональный сдвиг на границах разделов вперемешку с dissolve/fadeblack.
XFADE_TRANSITIONS = ["fade", "dissolve", "smoothleft", "smoothright",
                      "smoothup", "smoothdown", "hblur", "hlwind", "hrwind", "zoomin"]
BOUNDARY_TRANSITIONS = ["dissolve", "fadeblack", "fadewhite", "fadegrays"]
HOOK_MAX_CLIP = 3.6     # в хуке кадры короче и чаще — критично для удержания первых секунд.
                        # Было 5.0 — на практике держало хук почти вровень с телом ролика
                        # (4.65с против 6.8с), а не заметно быстрее, как задумано.
PAUSE_DURATIONS = {"[pause]": 0.8, "[short pause]": 0.4,
                   "[slowly]": 0.0, "[emphasis]": 0.0, "[energetic]": 0.0}

# Russo One — фирменный "рубленый" дисплейный шрифт (CHANNEL.md house
# style), не системный DejaVu. OFL, бесплатно (Google Fonts / google/fonts
# на GitHub). Anton (первый выбор) не подошёл — в нём НЕТ кириллицы вообще
# (проверено вживую: буквы рисуются квадратами-тофу). Russo One — с родной
# кириллицей от российской студии, того же плакатного характера.
FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
# Типографическая иерархия (не один шрифт на всё): Benzin ExtraBold — жирный
# плакатный display для "геройского" текста (хук-титры, цифры-доказательства
# в stat-плашках), Montserrat ExtraBold — второй уровень для подписи раздела
# (менее кричащий, но всё ещё жирный гротеск) вместо того же display-шрифта
# в меньшем кегле. Оба протестированы вживую на полную кириллицу (65/65
# букв) — в отличие от Anton, который в сессии раньше вообще не имел
# кириллицы. RussoOne остаётся дальше по цепочке как safe fallback, если
# файлов Benzin/Montserrat вдруг нет на диске (не ломает сборку).
FONT_CANDIDATES = [
    os.path.join(FONTS_DIR, "Benzin-ExtraBold.ttf"),
    os.path.join(FONTS_DIR, "RussoOne-Regular.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
# И Benzin, и RussoOne — плотные капслочные display-гарнитуры (тот же режим
# кегля/аптейка), поэтому оба считаются "дисплейным" случаем.
FONT_IS_DISPLAY = bool(FONT_PATH) and any(name in FONT_PATH for name in ("Benzin", "RussoOne"))

FONT_SECONDARY_CANDIDATES = [
    os.path.join(FONTS_DIR, "Montserrat-ExtraBold.ttf"),
]
# Секондари не находится -> используем тот же FONT_PATH, что и раньше (одна
# гарнитура на всё) — деградация к старому поведению, не падение.
FONT_SECONDARY_PATH = next((p for p in FONT_SECONDARY_CANDIDATES if os.path.exists(p)), FONT_PATH)


def section_title(name):
    """BLOCK N: Название -> 'Название'. HOOK/FINAL/безымянные BLOCK — без титра."""
    m = re.match(r'BLOCK\s+\d+\s*:\s*(.+)', name, re.I)
    return m.group(1).strip() if m else None


def escape_drawtext(s):
    # % \u043d\u0435 \u044d\u043a\u0440\u0430\u043d\u0438\u0440\u043e\u0432\u0430\u043b\u0441\u044f \u2014 ffmpeg drawtext \u0442\u0440\u0430\u043a\u0442\u0443\u0435\u0442 \u0435\u0433\u043e \u043a\u0430\u043a \u043d\u0430\u0447\u0430\u043b\u043e strftime/
    # expansion-\u0442\u043e\u043a\u0435\u043d\u0430 \u0438 \u0440\u043e\u043d\u044f\u0435\u0442 "Stray %" \u043f\u0440\u0435\u0434\u0443\u043f\u0440\u0435\u0436\u0434\u0435\u043d\u0438\u0435 (\u043f\u0440\u043e\u0432\u0435\u0440\u0435\u043d\u043e \u0432\u0436\u0438\u0432\u0443\u044e).
    return (s.replace("\\", "\\\\").replace(":", "\\:")
             .replace("%", "%%").replace("'", "\u2019"))


def pick_no_repeat(history, candidate, options, max_repeat):
    """\u0425\u044d\u0448 \u0434\u0430\u0451\u0442 \u0440\u0430\u0437\u043d\u043e\u043e\u0431\u0440\u0430\u0437\u0438\u0435 "\u0432 \u0441\u0440\u0435\u0434\u043d\u0435\u043c", \u043d\u043e \u043d\u0435 \u043c\u0435\u0448\u0430\u0435\u0442 3-4 \u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u044b\u043c \u043f\u043e\u0434\u0440\u044f\u0434
    \u0441\u043b\u0443\u0447\u0430\u0439\u043d\u044b\u043c \u0441\u043e\u0432\u043f\u0430\u0434\u0435\u043d\u0438\u0435\u043c. \u0415\u0441\u043b\u0438 \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 max_repeat \u0440\u0435\u0448\u0435\u043d\u0438\u0439 \u0441\u043e\u0432\u043f\u0430\u0434\u0430\u044e\u0442 \u0441
    \u043a\u0430\u043d\u0434\u0438\u0434\u0430\u0442\u043e\u043c \u2014 \u0434\u0435\u0442\u0435\u0440\u043c\u0438\u043d\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u043e \u0431\u0435\u0440\u0451\u043c \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439 \u0432\u0430\u0440\u0438\u0430\u043d\u0442 \u043f\u043e \u043a\u0440\u0443\u0433\u0443 \u0432\u043c\u0435\u0441\u0442\u043e
    \u043d\u0435\u0433\u043e. history \u043c\u0443\u0442\u0438\u0440\u0443\u0435\u0442\u0441\u044f \u043d\u0430 \u043c\u0435\u0441\u0442\u0435 (\u0441\u043f\u0438\u0441\u043e\u043a \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0445 \u0440\u0435\u0448\u0435\u043d\u0438\u0439)."""
    if len(history) >= max_repeat and all(x == candidate for x in history[-max_repeat:]):
        idx = options.index(candidate) if candidate in options else 0
        candidate = options[(idx + 1) % len(options)]
    history.append(candidate)
    del history[:-(max_repeat + 2)]
    return candidate


def audio_energy_curve(audio_path, window_sec=1.0):
    """RMS-\u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u044c \u0430\u0443\u0434\u0438\u043e \u043f\u043e \u043e\u043a\u043d\u0430\u043c \u2014 \u0431\u0435\u0437 Whisper/librosa, \u0447\u0438\u0441\u0442\u044b\u0439 PCM+numpy.
    \u0412\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0435\u0442 (rms_array, window_sec) \u0438\u043b\u0438 None, \u0435\u0441\u043b\u0438 numpy \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d/\u0447\u0442\u043e-\u0442\u043e
    \u043f\u043e\u0448\u043b\u043e \u043d\u0435 \u0442\u0430\u043a (\u0444\u0438\u0447\u0430 \u043e\u043f\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u0430\u044f, \u043f\u0430\u0439\u043f\u043b\u0430\u0439\u043d \u043d\u0435 \u0434\u043e\u043b\u0436\u0435\u043d \u043f\u0430\u0434\u0430\u0442\u044c \u0431\u0435\u0437 \u043d\u0435\u0451)."""
    if np is None:
        return None
    sr = 8000   # \u043d\u0443\u0436\u043d\u0430 \u0442\u043e\u043b\u044c\u043a\u043e \u043e\u0433\u0438\u0431\u0430\u044e\u0449\u0430\u044f \u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u0438, \u043d\u0435 \u0437\u0432\u0443\u043a \u2014 \u043d\u0438\u0437\u043a\u0438\u0439 sr \u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0438 \u0431\u044b\u0441\u0442\u0440\u043e
    try:
        r = subprocess.run(["ffmpeg", "-v", "quiet", "-i", audio_path, "-f", "s16le",
                            "-ac", "1", "-ar", str(sr), "-"], capture_output=True, timeout=120)
        if r.returncode != 0 or not r.stdout:
            return None
        raw = r.stdout[:len(r.stdout) - len(r.stdout) % 2]
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if len(samples) < sr:
            return None
        win = max(1, int(sr * window_sec))
        n_win = max(1, len(samples) // win)
        trimmed = samples[:n_win * win].reshape(n_win, win)
        rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
        return rms, window_sec
    except Exception as e:
        print(f"  \u0410\u043d\u0430\u043b\u0438\u0437 \u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u0438 \u043d\u0435 \u0443\u0434\u0430\u043b\u0441\u044f (\u043f\u0440\u043e\u043f\u0443\u0441\u043a\u0430\u044e): {e}")
        return None


def energy_pace_multipliers(curve, starts, durs, lo=0.8, hi=1.25):
    """\u0413\u0440\u043e\u043c\u0447\u0435 \u0443\u0447\u0430\u0441\u0442\u043e\u043a \u2014 \u043a\u043e\u0440\u043e\u0447\u0435 \u043a\u0430\u0434\u0440\u044b (\u043c\u043d\u043e\u0436\u0438\u0442\u0435\u043b\u044c <1), \u0442\u0438\u0448\u0435 \u2014 \u0434\u043b\u0438\u043d\u043d\u0435\u0435 (>1).
    \u041c\u043d\u043e\u0436\u0438\u0442\u0435\u043b\u044c \u0441\u0447\u0438\u0442\u0430\u0435\u0442\u0441\u044f \u043e\u0442 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u044f \u043a \u041c\u0415\u0414\u0418\u0410\u041d\u0415 \u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u0438 \u0432\u0441\u0435\u0439 \u0434\u043e\u0440\u043e\u0436\u043a\u0438,
    \u0447\u0442\u043e\u0431\u044b \u0442\u0438\u0445\u0438\u0439 \u0440\u043e\u043b\u0438\u043a \u0446\u0435\u043b\u0438\u043a\u043e\u043c \u043d\u0435 \u0440\u0430\u0441\u0442\u044f\u0433\u0438\u0432\u0430\u043b \u0432\u0441\u0435 \u043a\u0430\u0434\u0440\u044b \u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u043e."""
    if curve is None:
        return [1.0] * len(durs)
    rms, window_sec = curve
    med = float(np.median(rms))
    if med <= 0:
        return [1.0] * len(durs)
    mults = []
    for t0, d in zip(starts, durs):
        i0 = max(0, int(t0 / window_sec))
        i1 = min(len(rms), max(i0 + 1, int((t0 + d) / window_sec)))
        seg = rms[i0:i1] if i0 < len(rms) else rms[-1:]
        e = float(seg.mean()) if len(seg) else med
        ratio = max(0.5, min(2.0, e / med))
        mults.append(max(lo, min(hi, 1.0 / ratio)))
    return mults


def energy_levels(curve, starts, durs):
    """То же окно сэмплинга, что energy_pace_multipliers(), но возвращает
    СЫРОЕ отношение к медиане (0.5..2.0, 1.0 = обычная громкость), не
    инвертированный множитель темпа. Используется для лёгкой визуальной
    пульсации грейда в такт эмоциональным пикам речи (см. main()) —
    отдельная кривая от таймингов, чтобы не путать две разные вещи под
    одной переменной."""
    if curve is None:
        return [1.0] * len(durs)
    rms, window_sec = curve
    med = float(np.median(rms))
    if med <= 0:
        return [1.0] * len(durs)
    levels = []
    for t0, d in zip(starts, durs):
        i0 = max(0, int(t0 / window_sec))
        i1 = min(len(rms), max(i0 + 1, int((t0 + d) / window_sec)))
        seg = rms[i0:i1] if i0 < len(rms) else rms[-1:]
        e = float(seg.mean()) if len(seg) else med
        levels.append(max(0.5, min(2.0, e / med)))
    return levels


HOOK_ENERGY_SNAP_WINDOW = 0.35   # A8: макс. сдвиг границы реза в поиске локального пика, сек


def snap_hook_cuts_to_energy(blocks, durs, real_starts, audio_path):
    """A8: резы ВНУТРИ хука слегка "прилипают" к локальным пикам энергии
    ГОЛОСА (ударный слог/акцент) — тот же эффект "монтаж режется в такт",
    что в трейлерах режут под бит музыки, но синхронизировано с речью, а
    не с музыкой: наша подложка (generate_music_asset.py) намеренно без
    ритмического бита — дрон/пэд, чтобы не спорить с закадром документалки
    (см. A1/A2) — вводить бит в музыку ради этого сломало бы уже
    проверенное решение, поэтому синхронизация идёт по тому, что реально
    отбивает акценты в этой подаче: по голосу.

    Только границы МЕЖДУ соседними HOOK-клипами (не самая первая точка
    хука и не граница хук->тело — те определяются структурой сценария, не
    тайминг-эстетикой). Сдвиг ищется в мелком (80мс) окне энергии вокруг
    текущей границы, симметрично перекладывается между двумя соседними
    клипами — суммарная длительность хука не меняется НИ НА МИЛЛИСЕКУНДУ
    (бюджет xfade/точное совпадение с аудио не трогается), только реальные
    клипы i/j локально короче/длиннее друг друга. Каждый сдвиг ищется от
    ОРИГИНАЛЬНОЙ (досдвиговой) позиции границы — соседние сдвиги не
    накапливают дрейф друг от друга. Пол безопасности — hook_floor_for(),
    та же per-позиция функция, что уже держит открывающие резы хука ещё
    короче остального хука (не плоский порог — плоский тихо утаскивал бы
    уже нарочно короткие открывающие кадры обратно к общему полу, тот же
    класс бага, что hook_floor_for() уже один раз чинила для jitter/сдвига
    границы секции)."""
    curve = audio_energy_curve(audio_path, window_sec=0.08)
    if curve is None:
        return durs
    rms, win_sec = curve
    hook_idx = [i for i, b in enumerate(blocks) if b["section"].startswith("HOOK")]
    if len(hook_idx) < 3:
        return durs
    new_durs = list(durs)
    for k in range(len(hook_idx) - 1):
        i, j = hook_idx[k], hook_idx[k + 1]
        boundary_t = real_starts[j]
        lo = max(0, int((boundary_t - HOOK_ENERGY_SNAP_WINDOW) / win_sec))
        hi = min(len(rms), int((boundary_t + HOOK_ENERGY_SNAP_WINDOW) / win_sec) + 1)
        if hi <= lo:
            continue
        peak_i = lo + int(np.argmax(rms[lo:hi]))
        delta = max(-HOOK_ENERGY_SNAP_WINDOW, min(HOOK_ENERGY_SNAP_WINDOW, peak_i * win_sec - boundary_t))
        if new_durs[i] + delta < hook_floor_for(blocks, i) or new_durs[j] - delta < hook_floor_for(blocks, j):
            continue
        new_durs[i] += delta
        new_durs[j] -= delta
    return new_durs


def measure_loudnorm_stats(audio_path, target_i=-16, target_tp=-1.5, target_lra=11):
    """Первый проход двух-проходного loudnorm — только измерение, звук не
    трогаем. Однопроходный (динамический) режим подстраивает усиление на
    лету по ходу файла и даёт неровную громкость внутри ролика и неточный
    итоговый integrated — это ffmpeg-документация прямо называет compromise-
    режимом, не тем, что стоит использовать для финального мастеринга.
    None при любой ошибке — main() откатывается на старый однопроходный af."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-af",
             f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
             "-f", "null", "-"], capture_output=True, text=True, timeout=60)
        start, end = r.stderr.rindex("{"), r.stderr.rindex("}") + 1
        return json.loads(r.stderr[start:end])
    except Exception:
        return None


def process_voice(voice_path, out_path):
    """A6: обработка сырой TTS-дорожки перед миксом с музыкой — срез
    суб-баса, лёгкая тональная коррекция, деэссер, мягкая компрессия
    микро-динамики фраз. VOICE_PROCESS_ENABLED=False -> просто копия
    исходника (безопасный откат, тот же принцип, что MUSIC_ENABLED)."""
    if not VOICE_PROCESS_ENABLED:
        r = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-ar", "44100", "-ac", "2", out_path],
                            capture_output=True, text=True)
        return out_path if r.returncode == 0 else voice_path
    af = (
        f"highpass=f={VOICE_HIGHPASS_HZ},"
        f"equalizer=f={VOICE_EQ_WARMTH_HZ}:width_type=o:width=1.5:g={VOICE_EQ_WARMTH_GAIN},"
        f"equalizer=f={VOICE_EQ_PRESENCE_HZ}:width_type=o:width=1.2:g={VOICE_EQ_PRESENCE_GAIN},"
        f"deesser=i={VOICE_DEESS_INTENSITY}:m=0.4,"
        f"acompressor=threshold={VOICE_COMPRESS_THRESHOLD}:ratio={VOICE_COMPRESS_RATIO}:"
        f"attack=8:release=120:makeup=1.15"
    )
    r = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-af", af, "-ar", "44100", "-ac", "2", out_path],
                        capture_output=True, text=True)
    if r.returncode != 0:
        r2 = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-ar", "44100", "-ac", "2", out_path],
                             capture_output=True, text=True)
        return out_path if r2.returncode == 0 else voice_path
    return out_path


def add_typewriter_clicks(mix_path, click_times, total_dur, out_path):
    """D4: звук печатной машинки — щелчок на КАЖДЫЙ символ, ровно в момент,
    когда он появляется на экране (тот же TYPEWRITER_CHAR_DUR, что и в
    add_overlays() — визуал и звук считаются от одной и той же ставки).
    click_times — абсолютные секунды в ФИНАЛЬНОМ таймлайне ролика (см.
    main(): sub_starts + delay + k*char_dur).

    KEYBOARD_CLICK_PATHS — библиотека из НЕСКОЛЬКИХ разных реальных ударов
    (не один сэмпл клонштампом на каждый символ — по прямой обратной связи
    пользователя один и тот же щелчок на каждый символ звучал механически,
    "не как в живой записи"; round-robin по библиотеке даёт естественный
    разброс тембра/громкости между соседними символами, как при живой
    печати). Каждый входной файл asplit'ится РОВНО на то число раз, сколько
    он реально используется в этом ролике — не больше (лишние неиспользуемые
    выходы asplit — трата, ffmpeg отдельно предупреждает про unconnected
    output)."""
    if not click_times or not TYPEWRITER_CLICK_ENABLED:
        return mix_path
    n_files = len(KEYBOARD_CLICK_PATHS)
    assigned = [i % n_files for i in range(len(click_times))]
    usage = [assigned.count(f) for f in range(n_files)]
    cmd = ["ffmpeg", "-y", "-i", mix_path]
    for p in KEYBOARD_CLICK_PATHS:
        cmd += ["-i", p]
    parts = []
    for f in range(n_files):
        if usage[f] > 0:
            parts.append(f"[{f + 1}:a]asplit={usage[f]}" + "".join(f"[fs{f}_{k}]" for k in range(usage[f])))
    cursor = [0] * n_files
    mix_inputs = ["[0:a]"]
    for i, (t, f_idx) in enumerate(zip(click_times, assigned)):
        ms = max(0, int(t * 1000))
        k = cursor[f_idx]
        cursor[f_idx] += 1
        parts.append(f"[fs{f_idx}_{k}]adelay={ms}|{ms},volume={TYPEWRITER_CLICK_GAIN_DB}dB[cd{i}]")
        mix_inputs.append(f"[cd{i}]")
    parts.append("".join(mix_inputs) + f"amix=inputs={len(click_times) + 1}:duration=first:normalize=0[out]")
    fc = ";".join(parts)
    cmd += ["-filter_complex", fc, "-map", "[out]",
            "-t", f"{total_dur:.3f}", "-ar", "44100", "-ac", "2", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return out_path if r.returncode == 0 else mix_path


def build_mood_timeline(hook_end, final_start, total_dur, out_path):
    """A2: единая дорожка подложки на весь ролик, склеенная из 3 настроений
    (HOOK/BODY/FINAL, см. MUSIC_BED_PATHS) через медленный кроссфейд
    (acrossfade), а не один и тот же луп на все 18 минут. Переходные точки
    сдвинуты РАНЬШЕ формальной границы секции на MUSIC_JCUT_LEAD секунд —
    звуковой J-cut: настроение подложки уже слегка меняется до того, как
    текст/картинка формально перешли в новый раздел.

    Математика длин сегментов для acrossfade с двумя переходами t1 (HOOK->
    BODY) и t2 (BODY->FINAL), d = MUSIC_MOOD_XFADE:
    L1 = t1+d, L2 = (t2-t1)+d, L3 = total_dur-t2 — тогда итоговая длина
    после двух acrossfade равна ровно total_dur (проверено расчётом и живым
    ffprobe на реальных цифрах), с чистыми "полными" участками настроения
    длиной t1 / (t2-t1-d) / (total_dur-t2-d) между кроссфейдами.

    Слишком короткий ролик/секция (t2-t1 меньше 2 кроссфейдов) -> тихий
    откат на одну BODY-петлю на всю длину, без попытки втиснуть переходы
    туда, где для них физически нет места."""
    d = MUSIC_MOOD_XFADE
    t1 = max(2.0, hook_end - MUSIC_JCUT_LEAD)
    t2 = min(total_dur - 2.0, max(t1 + 2 * d, final_start - MUSIC_JCUT_LEAD))
    if t2 - t1 < 2 * d or t1 >= total_dur - 2 * d:
        r = subprocess.run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", MUSIC_BED_PATHS["BODY"],
                             "-t", f"{total_dur:.3f}", "-ar", "44100", "-ac", "2", out_path],
                            capture_output=True, text=True)
        return out_path if r.returncode == 0 else None
    L1, L2, L3 = t1 + d, (t2 - t1) + d, total_dur - t2
    fc = (
        f"[0:a]atrim=0:{L1:.3f}[s1];"
        f"[1:a]atrim=0:{L2:.3f}[s2];"
        f"[2:a]atrim=0:{L3:.3f}[s3];"
        f"[s1][s2]acrossfade=d={d}:c1=tri:c2=tri[r1];"
        f"[r1][s3]acrossfade=d={d}:c1=tri:c2=tri[mood_mix]"
    )
    r = subprocess.run(
        ["ffmpeg", "-y",
         "-stream_loop", "-1", "-i", MUSIC_BED_PATHS["HOOK"],
         "-stream_loop", "-1", "-i", MUSIC_BED_PATHS["BODY"],
         "-stream_loop", "-1", "-i", MUSIC_BED_PATHS["FINAL"],
         "-filter_complex", fc, "-map", "[mood_mix]",
         "-t", f"{total_dur:.3f}", "-ar", "44100", "-ac", "2", out_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        r2 = subprocess.run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", MUSIC_BED_PATHS["BODY"],
                              "-t", f"{total_dur:.3f}", "-ar", "44100", "-ac", "2", out_path],
                             capture_output=True, text=True)
        return out_path if r2.returncode == 0 else None
    return out_path


def build_music_mix(voice_path, total_dur, out_path, hook_end=0.0, final_start=None):
    """A1: атмосферная подложка под голосом, с сайдчейн-дакингом — громкость
    музыки автоматически прижимается, когда звучит речь, и отпускает в
    паузах (тот же приём, что в любом проф. документальном монтаже: подложка
    ощущается, но никогда не спорит со словами). A2: подложка меняет
    настроение по секциям (build_mood_timeline), не один луп на весь ролик.
    MUSIC_ENABLED=False (нет ассета или явно выключено) -> просто копируем
    голос как есть, без регрессии.

    Микс делается ОТДЕЛЬНЫМ шагом (не встроен прямо в финальный мультиплекс)
    — так итоговый loudnorm (A7) считается на ГОТОВОМ миксе голос+музыка,
    а не только на голосе, как было раньше: иначе музыка добавляла бы
    громкость ПОСЛЕ калибровки и сводила её к нулю."""
    if not MUSIC_ENABLED:
        r = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-t", f"{total_dur:.3f}",
                             "-ar", "44100", "-ac", "2", out_path], capture_output=True, text=True)
        return out_path if r.returncode == 0 else voice_path
    if final_start is None:
        final_start = total_dur
    timeline_path = out_path + ".mood_timeline.wav"
    timeline = build_mood_timeline(hook_end, final_start, total_dur, timeline_path)
    if timeline is None:
        r = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-t", f"{total_dur:.3f}",
                             "-ar", "44100", "-ac", "2", out_path], capture_output=True, text=True)
        return out_path if r.returncode == 0 else voice_path
    filter_complex = (
        f"[1:a]volume={MUSIC_BED_GAIN_DB}dB[music_raw];"
        f"[0:a]asplit=2[voice_main][voice_sc];"
        f"[music_raw][voice_sc]sidechaincompress=threshold={MUSIC_DUCK_THRESHOLD}:"
        f"ratio={MUSIC_DUCK_RATIO}:attack={MUSIC_DUCK_ATTACK_MS}:release={MUSIC_DUCK_RELEASE_MS}[music_ducked];"
        f"[voice_main][music_ducked]amix=inputs=2:duration=first:weights=1 1:normalize=0[mix]"
    )
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", voice_path, "-i", timeline,
         "-filter_complex", filter_complex, "-map", "[mix]",
         "-t", f"{total_dur:.3f}", "-ar", "44100", "-ac", "2", out_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        # Микс не собрался — откат на голос без подложки, не срываем сборку
        # ролика ради необязательного улучшения.
        r2 = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-t", f"{total_dur:.3f}",
                              "-ar", "44100", "-ac", "2", out_path], capture_output=True, text=True)
        return out_path if r2.returncode == 0 else voice_path
    return out_path


def measure_levels(path, is_video=False, lo_pct=2, hi_pct=98, video_samples=4, video_span=3.5):
    """Чёрная/белая точка кадра (0..1) по перцентилям гистограммы — для
    адаптивной ТЕХНИЧЕСКОЙ нормализации экспозиции ДО фиксированного
    творческого грейда (см. auto_levels_params()/film_look()). Перцентили,
    не голый min/max — иначе один яркий блик/пересвеченный пиксель срывает
    белую точку всего кадра.

    Для видео — НЕСКОЛЬКО кадров-пробников (video_samples, разнесённых по
    первым video_span секундам, тот же принцип, что уже использует
    measure_motion() для спид-рампа), не один: экспозиция может плыть
    внутри клипа (панорама между светлым/тёмным), один кадр — нерепрезен-
    тативная выборка. Перцентили считаются по ОБЪЕДИНЁННЫМ пикселям всех
    сэмплов — честная оценка разброса по всему просматриваемому диапазону,
    не по случайному одному моменту."""
    try:
        if is_video:
            arrs = []
            for i in range(video_samples):
                # 0.5 старт, не 0 — реальный сток иногда начинается с чёрного
                # лидер-кадра/fade-in (поймано вживую: у одного клипа эпизода
                # кадр 0 был буквально чистым чёрным, min=max=0 — без сдвига
                # это заваливало бы коррекцию в вырожденный случай на КАЖДОМ
                # таком клипе).
                t = 0.5 + video_span * i / max(1, video_samples - 1)
                tmp = path + f"._levels_probe_{i}.jpg"
                r = subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", path,
                                     "-frames:v", "1", "-q:v", "5", tmp],
                                    capture_output=True, timeout=20)
                if r.returncode == 0 and os.path.exists(tmp):
                    if np is not None:
                        arrs.append(np.asarray(PILImage.open(tmp).convert("L"), dtype=np.float32) / 255.0)
                    os.remove(tmp)
            if not arrs:
                return None
            arr = np.concatenate([a.ravel() for a in arrs])
        else:
            if np is None:
                return None
            arr = np.asarray(PILImage.open(path).convert("L"), dtype=np.float32).ravel() / 255.0
        lo = float(np.percentile(arr, lo_pct))
        hi = float(np.percentile(arr, hi_pct))
        return lo, hi
    except Exception:
        return None


AUTO_LEVELS_TARGET_BLACK = 0.03
AUTO_LEVELS_TARGET_WHITE = 0.97
# Клэмп силы коррекции — НЕ полное выравнивание к цели (иначе кадр с
# осознанно низким контрастом/затемнением — музейный экспонат в полумраке —
# перекорректируется до нейтрального "мыла", тот же принцип, что уже держит
# в узде brightness_bias). 0.5 = на полпути между "как есть" и целью.
AUTO_LEVELS_MAX_STRENGTH = 0.5
AUTO_LEVELS_MIN_RANGE = 0.05   # чёрная/белая точка ближе этого — вырожденная гистограмма, не трогаем


def auto_levels_params(levels):
    """(contrast, brightness) для eq() — линейная растяжка чёрной/белой
    точки К цели (не ДО цели — см. AUTO_LEVELS_MAX_STRENGTH), формула eq
    подтверждена вживую на синтетическом градиенте: output=(input-0.5)*
    contrast+0.5+brightness. None/вырожденная гистограмма -> (1.0, 0.0) —
    нейтральный проход, эффекта нет."""
    if levels is None:
        return 1.0, 0.0
    black_in, white_in = levels
    if white_in - black_in < AUTO_LEVELS_MIN_RANGE:
        return 1.0, 0.0
    s = AUTO_LEVELS_MAX_STRENGTH
    # Блендим ЛИНЕЙНО две affine-трансформации — identity (as-is) и полную
    # коррекцию (black_in->TARGET_BLACK, white_in->TARGET_WHITE) — блендинг
    # двух affine-преобразований сам по себе affine, поэтому это по-прежнему
    # чистая (contrast, brightness) пара, не приближение.
    scale_full = (AUTO_LEVELS_TARGET_WHITE - AUTO_LEVELS_TARGET_BLACK) / (white_in - black_in)
    contrast_eff = (1 - s) + s * scale_full
    brightness_offset = s * (AUTO_LEVELS_TARGET_BLACK - black_in * scale_full)
    brightness = brightness_offset - 0.5 + 0.5 * contrast_eff
    return contrast_eff, brightness


def measure_luma(path, is_video=False):
    """Средняя яркость кадра (0..1) — для сглаживания скачков экспозиции
    между соседними склейками (см. main()): сток из разных источников
    скачет по яркости кадр-к-кадру, это частый "любительский" tell в
    компиляциях. Для видео берём один кадр в начале — грубой оценки
    достаточно, не нужен точный анализ всего клипа."""
    try:
        if is_video:
            tmp = path + "._luma_probe.jpg"
            # -ss 0.5, не кадр 0 — см. тот же фикс в measure_levels(): реальный
            # сток иногда начинается с чёрного лидер-кадра/fade-in, кадр 0
            # тогда даёт ложный "снято в темноте" на весь клип.
            r = subprocess.run(["ffmpeg", "-y", "-ss", "0.5", "-i", path, "-frames:v", "1", "-q:v", "5", tmp],
                                capture_output=True, timeout=20)
            if r.returncode != 0 or not os.path.exists(tmp):
                return None
            img = PILImage.open(tmp).convert("L").resize((64, 36))
            os.remove(tmp)
        else:
            img = PILImage.open(path).convert("L").resize((64, 36))
        arr = list(img.getdata())
        return sum(arr) / len(arr) / 255.0
    except Exception:
        return None


def find_audio():
    for name in ("audio_fixed.mp3", "audio.mp3"):
        p = os.path.join(VIDEO_FOLDER, name)
        if os.path.exists(p):
            return p
    mp3s = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(".mp3")]
    return os.path.join(VIDEO_FOLDER, mp3s[0]) if mp3s else os.path.join(VIDEO_FOLDER, "audio.mp3")


AUDIO_FILE = find_audio()


def load_json_dict(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  Битый {path}, пропускаю: {e}")
    return {}


def load_themes():
    """Два уровня: канальный словарь (channel_themes.json в корне репо —
    общие для ниши слова вроде "меч"/"доспех"/"музей", не меняются от
    ролика к ролику) + эпизодный (media_plan/themes.json конкретного
    видео — только то, что специфично именно этой теме). Раньше весь
    словарь собирался заново руками под каждый новый сценарий; теперь
    новому эпизоду нужно всего несколько строк добавки поверх базы."""
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "channel_themes.json")
    episode_path = os.path.join(VIDEO_FOLDER, "media_plan", "themes.json")
    merged = load_json_dict(base_path)
    merged.update(load_json_dict(episode_path))
    return merged


THEMES = load_themes()

# Слова, по которым блок вероятно описывает физическое действие/движение —
# видео туда осмысленно (бой на стоке реально показывает бой), не просто
# механическая чётность i%2 (см. main() — та не различала "плита лежит на
# столе" от "штурм крепости", зритель подсознательно считывает периодичность
# чередования). Общая лексика военно-исторической ниши, не привязана к
# конкретному эпизоду — как THEMES, расширяется по мере надобности.
ACTION_WORDS = {
    "бой", "битва", "сражение", "штурм", "атака", "атакуют", "нападение",
    "поход", "марш", "маршируют", "наступление", "отступление", "погоня",
    "бегут", "бежит", "мчатся", "скачут", "удар", "удары", "рубит", "рубят",
    "рубился", "сечи", "сеча", "фехтуют", "фехтование", "поединок", "дуэль",
    "разит", "разили", "пожар", "горит", "горят", "взрыв", "обстрел",
    "осада", "осаждают", "прорыв", "рукопашная", "схватка", "резня",
}


def has_action_word(text):
    return any(re.sub(r'[^\w]', '', w.lower()) in ACTION_WORDS for w in text.split())


def get_audio_duration():
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", AUDIO_FILE], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


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
    processed = content
    for tag in sorted(PAUSE_DURATIONS, key=len, reverse=True):
        processed = processed.replace(tag, f"__PAUSE_{PAUSE_DURATIONS[tag]}__")
    processed = re.sub(r'\[.*?\]', '', processed)
    parts = re.split(r'(__PAUSE_[\d.]+__|\x00SECTION:.*?\x00|\x01STAT:.*?\x01)', processed)
    blocks, cur, pause, stat, stat_word_pos = [], "", 0.0, None, None
    section = "BODY"

    def flush():
        nonlocal cur, pause, stat, stat_word_pos
        if cur:
            blocks.append({"text": cur, "pause_after": pause,
                           "words": len(cur.split()), "section": section, "stat": stat,
                           "stat_word_pos": stat_word_pos})
        cur, pause, stat, stat_word_pos = "", 0.0, None, None

    for part in parts:
        mp = re.match(r'__PAUSE_([\d.]+)__', part)
        ms = re.match(r'\x00SECTION:(.*?)\x00', part)
        mst = re.match(r'\x01STAT:(.*?)\x01', part)
        if mp:
            pause += float(mp.group(1))
        elif ms:
            flush()             # смена секции — всегда граница блока
            section = ms.group(1)
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
        else:
            t = part.strip()
            if t:
                if pause > 0 and cur:
                    flush()      # реальная пауза — вот это настоящая граница блока
                    cur = t
                else:
                    cur = f"{cur} {t}".strip()
    flush()
    print(f"Блоков: {len(blocks)}")
    return blocks


ALIGNMENT_DIR = os.path.join(VIDEO_FOLDER, "media_plan", "alignment")
ALIGNMENT_TAG_RE = re.compile(r'\[short pause\]|\[pause\]')
ALIGNMENT_STRIP_TAGS = ("[energetic]", "[slowly]", "[emphasis]")


def _real_speech_span(segment):
    """Реальная длительность озвученного текста в сегменте символов
    (index,char,start,end) — по первому и последнему буквенно-цифровому
    символу, за вычетом служебных тегов вроде [energetic]/[slowly],
    которые могут затесаться внутрь (они не читаются TTS вслух, но
    формально попадают в поток символов из alignment.csv)."""
    text = "".join(c for c, s, e in segment)
    excluded = set()
    for tag in ALIGNMENT_STRIP_TAGS:
        idx = text.find(tag)
        while idx != -1:
            for j in range(idx, idx + len(tag)):
                excluded.add(j)
            idx = text.find(tag, idx + 1)
    clean = [(c, s, e) for j, (c, s, e) in enumerate(segment)
             if j not in excluded and re.match(r'[^\s\[\]]', c or "")]
    if not clean:
        return 0.0
    return clean[-1][2] - clean[0][1]


def load_alignment_weights(blocks):
    """Реальные веса длительности блоков из посимвольного alignment.csv
    (ElevenLabs/Lumean отдают его бесплатно вместе с каждым TTS-заказом —
    раньше просто не забирался). Раньше длительность блока считалась ЧИСТО
    по числу слов и общей скорости слов/сек — то есть каждое слово "весило"
    одинаково. На деле темп TTS живой: короткая ударная фраза читается
    быстрее длинной со сложными словами, [slowly]-блок — заметно медленнее.
    Реальный посимвольный тайминг ловит эту вариацию, а не усредняет её.

    Файлы лежат в media_plan/alignment/00.csv, 01.csv... — по одному на
    КАЖДУЮ секцию (HOOK, BLOCK 1, ..., FINAL) в порядке их первого появления
    в script.txt, разрезаны по буквальным вхождениям [pause]/[short pause]
    на сегменты 1:1 с под-блоками этой секции в том же порядке.

    Возвращает список той же длины, что blocks, — вес (сек) реальной речи
    на блок или None для блоков без данных (эпизод без сохранённого
    alignment, или новый блок, добавленный после записи — сборка тогда
    просто откатывается на word-count оценку для этих блоков, как раньше)."""
    if not os.path.isdir(ALIGNMENT_DIR):
        return None
    section_order = []
    for b in blocks:
        if not section_order or section_order[-1] != b["section"]:
            section_order.append(b["section"])
    section_segments = {}
    for i, name in enumerate(section_order):
        path = os.path.join(ALIGNMENT_DIR, f"{i:02d}.csv")
        if not os.path.exists(path):
            continue
        try:
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            chars = [(r["char"], float(r["start"]), float(r["end"])) for r in rows]
        except Exception as e:
            print(f"  alignment {path} битый, пропускаю: {e}")
            continue
        text = "".join(c for c, s, e in chars)
        segs, pos = [], 0
        for m in ALIGNMENT_TAG_RE.finditer(text):
            segs.append(chars[pos:m.start()])
            pos = m.end()
        segs.append(chars[pos:])
        section_segments[name] = segs

    weights = []
    seg_cursor = {}
    stale = 0
    for b in blocks:
        segs = section_segments.get(b["section"])
        if segs is None:
            weights.append(None)
            continue
        k = seg_cursor.get(b["section"], 0)
        if k >= len(segs):
            # Число под-блоков в текущем script.txt разошлось с числом
            # сегментов в сохранённом alignment (текст правили после
            # записи) — честно откатываемся на word-count для остатка
            # секции, а не подставляем чужой сегмент не по месту.
            weights.append(None)
        else:
            # Число сегментов может совпасть, а ТЕКСТ — разойтись (слово-два
            # поправили в script.txt после записи озвучки, не тронув счётчик
            # [pause]) — тогда старый код молча применял чужой тайминг к
            # новому тексту. Fuzzy-сверка (difflib, без новых зависимостей)
            # блока против реально озвученного сегмента: разошлись сильнее
            # порога — честный откат на word-count вместо тихой подмены.
            seg_text = re.sub(r'\s+', ' ', "".join(c for c, s, e in segs[k])).strip().lower()
            block_text = re.sub(r'\s+', ' ', b["text"]).strip().lower()
            ratio = difflib.SequenceMatcher(None, block_text, seg_text).ratio() if (block_text and seg_text) else 0.0
            if ratio < 0.55:
                weights.append(None)
                stale += 1
            else:
                span = _real_speech_span(segs[k])
                weights.append(span if span > 0.05 else None)
        seg_cursor[b["section"]] = k + 1
    if stale:
        print(f"  Alignment: {stale} блок(ов) текстом разошлись с сохранённым таймингом "
              f"(script.txt правили после записи?) — откат на word-count для них")
    return weights


def load_hook_word_timings():
    """D2: посимвольный alignment.csv хука (00.csv — хук всегда первая
    секция эпизода, см. load_alignment_weights) собран в СЛОВА — тайминг
    в ХУК-локальной шкале (t=0 = начало хука), та же шкала, что sub_starts
    для HOOK-блоков в main(), поэтому напрямую сравнима без пересчёта.
    Служебные теги ([energetic]/[pause]/[short pause]/...) вычищаются —
    та же логика, что _real_speech_span, но разбирает на слова, а не на
    общий диапазон. Нет alignment.csv -> [] (кинетика хука тихо не
    показывается, не регрессия — как и все alignment-based фичи в пайплайне)."""
    path = os.path.join(ALIGNMENT_DIR, "00.csv")
    if not os.path.exists(path):
        return []
    try:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        chars = [(r["char"], float(r["start"]), float(r["end"])) for r in rows]
    except Exception:
        return []
    text = "".join(c for c, s, e in chars)
    excluded = set()
    for tag in ALIGNMENT_STRIP_TAGS + ("[pause]", "[short pause]"):
        idx = text.find(tag)
        while idx != -1:
            for j in range(idx, idx + len(tag)):
                excluded.add(j)
            idx = text.find(tag, idx + 1)
    # Вырезанный тег ТОЖЕ обязан рвать слово, не только пробел — в сыром
    # тексте тег часто стоит БЕЗ пробела ("...убеждают.[short pause]Так
    # говорят" — точка сразу перед тегом, слово сразу после), простое
    # исключение символов тега без разрыва склеивало соседние слова в одно
    # ("килограммов.Так") — поймано вживую на реальном 00.csv эпизода.
    words, cur = [], []
    for j, (c, s, e) in enumerate(chars):
        if j in excluded or c.isspace() or c in "[]":
            if cur:
                words.append(("".join(ch for ch, _, _ in cur), cur[0][1], cur[-1][2]))
                cur = []
        else:
            cur.append((c, s, e))
    if cur:
        words.append(("".join(ch for ch, _, _ in cur), cur[0][1], cur[-1][2]))
    return [w for w in words if w[0]]


HOOK_CAPTION_MIN_DUR = 0.15   # короткие служебные слова из alignment иначе мелькают 1-2 кадра, нечитаемо


def hook_captions_for_block(hook_words, block_start, block_dur):
    """Слова из load_hook_word_timings(), чьё время пересекается с окном
    [block_start, block_start+block_dur) конкретного HOOK-клипа — переведены
    в ЛОКАЛЬНОЕ время клипа (0 = начало клипа) и обрезаны по его границам,
    чтобы drawtext enable=between() не ссылался на моменты вне клипа.

    Минимальная длительность показа (HOOK_CAPTION_MIN_DUR) — короткие слова
    (особенно сразу после вырезанного тега паузы, см. load_hook_word_timings)
    иногда получают из alignment почти нулевую длительность и мелькают
    незаметно; растягиваем ВПЕРЁД, но не дальше начала следующего слова
    (иначе два слова читались бы на экране одновременно)."""
    caps = []
    for k, (word, s, e) in enumerate(hook_words):
        if e <= block_start or s >= block_start + block_dur:
            continue
        local_s = max(0.0, s - block_start)
        local_e = min(block_dur, e - block_start)
        if local_e - local_s < HOOK_CAPTION_MIN_DUR:
            next_start = (hook_words[k + 1][1] - block_start) if k + 1 < len(hook_words) else block_dur
            local_e = min(block_dur, next_start, local_s + HOOK_CAPTION_MIN_DUR)
        if local_e > local_s:
            caps.append((word, local_s, local_e))
    return caps


def hook_floor_for(blocks, i):
    """Пол длительности для блока i — с учётом того, что первые
    HOOK_OPENING_CUTS резов ХУКА получают ещё более низкий пол
    (HOOK_OPENING_MIN_CLIP), чем остальной хук (HOOK_MIN_CLIP). Единая
    функция для всех мест, что клэмпят длительность по этому полу
    (block_durations/apply_section_boundary_shift/apply_human_jitter) —
    иначе они разошлись бы между собой, и джиттер/сдвиг границы секции
    молча утаскивали бы уже нарочно короткие открывающие кадры обратно
    к общему HOOK_MIN_CLIP, сводя на нет весь смысл более быстрого пола."""
    if not blocks[i]["section"].startswith("HOOK"):
        return MIN_CLIP
    hook_idx = sum(1 for k in range(i + 1) if blocks[k]["section"].startswith("HOOK"))
    return HOOK_OPENING_MIN_CLIP if hook_idx <= HOOK_OPENING_CUTS else HOOK_MIN_CLIP


def block_durations(blocks, total, energy_mults=None, real_weights=None):
    tw = sum(b["words"] for b in blocks)
    tp = sum(b["pause_after"] for b in blocks)
    wps = tw / max(total - tp, 1)
    raw = []
    for i, b in enumerate(blocks):
        rw = real_weights[i] if real_weights else None
        base = rw if rw is not None else b["words"] / wps
        raw.append(base + b["pause_after"])
    if energy_mults:
        raw = [r * m for r, m in zip(raw, energy_mults)]
    d = []
    for i, (b, r) in enumerate(zip(blocks, raw)):
        # В хуке кадры короче и чаще — первые секунды решают, останется ли зритель.
        # Раньше пол был ОДИН на весь ролик (MIN_CLIP) — реальная короткая
        # фраза в хуке всё равно раздувалась до общего пола, и хук не
        # отличался по темпу от тела ролика. Свой, более низкий пол (ещё
        # ниже — для самых первых резов, см. hook_floor_for) — хук реально
        # может резать чаще, а не только "теоретически может".
        is_hook = b["section"].startswith("HOOK")
        cap = HOOK_MAX_CLIP if is_hook else MAX_CLIP
        floor = hook_floor_for(blocks, i)
        d.append(max(floor, min(cap, r)))
    scale = total / sum(d)
    return [x * scale for x in d]


PEXELS_BROKEN = False       # взводится только на реальном отказе API, не на пустой выдаче


PHOTO_DEDUP_HAMMING = 6   # тот же порог, что в qc_report() — "заметно похоже"
PHOTO_DEDUP_MAX_TRIES = 6 # сколько кандидатов реально СКАЧАТЬ и захэшировать, прежде чем сдаться


def pexels_photo(query, index, used_ids=None, used_hashes=None, recent_sizes=None, target_luma=None):
    """used_ids — множество ID уже показанных в этом ролике фото (мутируется на
    месте). Разные блоки часто ловят один и тот же тематический запрос — без
    этого им всем доставался бы top-1 результат, то есть одна и та же картинка
    по нескольку раз за ролик. Перебираем выдачу (per_page=80 — Pexels-максимум,
    проверено вживую: возвращает 79; 40 всё ещё не хватало на узкой теме
    вроде "meч" при 150 sub-cut блоках на одну лексическую тему — used_ids
    общий на весь ролик, а не на запрос, так что разные запросы про мечи
    делят один и тот же небольшой пул различимых стоковых фото).

    used_hashes (опционально) — список aHash уже отобранных в ролик фото.
    Дедуп только по ID ловит лишь буквально тот же файл повторно — не ловит
    РАЗНЫЕ ID из одной фотосессии (тот же реквизит/ракурс, другой кадр серии),
    а именно так выглядели реальные дубли, найденные постфактум в qc_report()
    на готовом ролике. Теперь при наличии used_hashes каждый кандидат перед
    принятием СКАЧИВАЕТСЯ и хэшируется — если он визуально слишком похож
    (hamming <= 6) на уже выбранную картинку, пробуем следующего кандидата
    из уже полученного списка (без нового сетевого запроса на список, только
    докачка кандидата) — до PHOTO_DEDUP_MAX_TRIES попыток, чтобы не жечь
    время/трафик на исчерпание всей выдачи. Не нашли непохожего — берём
    наименее похожего из просмотренных (лучше редкий повтор, чем сорванная
    сборка).

    target_luma (опционально) — текущий luma_ema соседних кадров (см. main()).
    Самый младший критерий выбора (после anti-dup и size-variety): среди уже
    и так скачанных в рамках анти-дубль перебора кандидатов предпочитаем того,
    чья ЕСТЕСТВЕННАЯ яркость ближе к контексту — меньше нагрузки на синтетическую
    коррекцию film_look(brightness_bias), та зажата в ±0.035 и не всегда
    вытягивает разницу целиком. Не тратит дополнительные скачивания сверх уже
    идущего анти-дубль перебора."""
    global PEXELS_BROKEN
    cache = os.path.join(TEMP_FOLDER, "pexels_cache")
    os.makedirs(cache, exist_ok=True)
    # Хэш запроса в имени файла — иначе смена themes.json без чистки temp_smart/
    # молча оставляет картинку под старый запрос (кэш бил только по номеру блока).
    qhash = hashlib.md5(query.encode()).hexdigest()[:8]
    cf = os.path.join(cache, f"{index:04d}_{qhash}.jpg")
    if os.path.exists(cf):
        if used_hashes is None:
            if recent_sizes is not None:
                try:
                    recent_sizes.append(estimate_shot_size(cf))
                except Exception:
                    pass
            return cf
        # Реальный баг, пойманный вживую: кэш-хит раньше отдавал файл СРАЗУ,
        # даже не проверяя его против used_hashes — анти-дубль код (ниже)
        # тогда никогда не успевал сработать на файлах, скачанных ЕЩЁ ДО
        # его появления (или на прошлом прогоне без анти-дубля). На готовом
        # ролике это дало 4 пары дублей — оба файла кэша существовали
        # заранее, каждый кэш-хит молча их одобрял. Теперь кэш-хит тоже
        # проверяется на похожесть — коллизия найдена, файл НЕ считается
        # валидным кэшем, код падает ниже в обычный путь скачивания и
        # перезаписывает cf нормальным анти-дубль перебором кандидатов.
        try:
            h = ahash(cf)
            if min((hamming(h, uh) for uh in used_hashes), default=99) > PHOTO_DEDUP_HAMMING:
                used_hashes.append(h)
                if recent_sizes is not None:
                    try:
                        recent_sizes.append(estimate_shot_size(cf))
                    except Exception:
                        pass
                return cf
        except Exception:
            return cf
    if not PEXELS_API_KEY:
        return None
    try:
        q = urllib.parse.quote(query)
        req = urllib.request.Request(
            f"https://api.pexels.com/v1/search?query={q}&per_page=80&orientation=landscape",
            headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        photos = data.get("photos") or []
        if not photos:
            return None
        candidates = [p for p in photos if used_ids is None or p.get("id") not in used_ids] or photos

        def download(p, dest):
            url = p["src"].get("large2x") or p["src"].get("large")
            req_img = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req_img, timeout=20) as r:
                open(dest, "wb").write(r.read())

        if used_hashes is None:
            pick = candidates[0]
            download(pick, cf)
        else:
            # score = (не дубль?, разнообразит масштаб плана?, релевантна
            # запросу по CLIP?, эстетическая оценка, ближе по яркости к
            # соседям?, дистанция по хэшу) — сравниваются как кортежи,
            # лексикографически: "не дубль" важнее "разнообразия", то
            # важнее релевантности, та важнее эстетики, та важнее яркостной
            # близости, та важнее точной дистанции (самый младший критерий —
            # просто добивает выбор среди и так уже равных по первым осям).
            # recent_sizes/luma/CLIP/эстетика — те же уже скачанные
            # кандидаты анти-дубля, никакого нового сетевого трафика ради
            # этой доп. проверки — переиспользуем то, что и так уже качаем.
            best_path, best_score, good_seen = None, (-1, -1, -1, -100.0, -1.0, -1), 0
            # Без target_luma брейк на первом же "хорошем" кандидате, как раньше
            # (0 доп. скачиваний). С target_luma нужно сравнить МИНИМУМ 2 таких
            # кандидата, иначе яркостный критерий никогда не встретится с
            # реальной альтернативой и молча ничего не решает — брейк на первом
            # же успехе просто не даёт ему сработать. +1 скачивание в типичном
            # случае (не x6) — единственный способ сделать tie-breaker не
            # фиктивным, не разгоняя стоимость до полного перебора. CLIP-
            # релевантность НЕ добавляет к good_needed — она не tie-breaker,
            # а гейт (см. is_relevant ниже, часть условия break) — кандидат
            # без неё не считается "хорошим", даже если дубль-фри/размер ок.
            good_needed = 2 if target_luma is not None else 1
            for p in candidates[:PHOTO_DEDUP_MAX_TRIES]:
                trial = cf + f".trial_{p.get('id')}.jpg"
                try:
                    download(p, trial)
                    h = ahash(trial)
                except Exception:
                    continue
                min_d = min((hamming(h, uh) for uh in used_hashes), default=99)
                is_dup_free = 1 if min_d > PHOTO_DEDUP_HAMMING else 0
                size_ok = 1
                if recent_sizes is not None:
                    try:
                        size_ok = 0 if estimate_shot_size(trial) in recent_sizes[-2:] else 1
                    except Exception:
                        size_ok = 1
                relevance = clip_relevance(trial, query)
                is_relevant = 1 if (relevance is None or relevance >= CLIP_RELEVANCE_THRESHOLD) else 0
                # Эстетика — доп. измерение, НЕ повышает good_needed само по
                # себе (в отличие от target_luma) — иначе каждый выбор фото
                # тратил бы вдвое больше скачиваний/Pexels-трафика по
                # умолчанию, а не только когда это уже оправдано другой
                # причиной. В реальных вызовах из main() target_luma и так
                # почти всегда задан (сравниваются 2+ кандидата), так что
                # эстетика получает шанс сработать бесплатно; если нет —
                # просто тихо не участвует в выборе (как раньше).
                aesthetic = aesthetic_score(trial)
                aesthetic_val = aesthetic if aesthetic is not None else 0.0
                luma_score = 0.0
                if target_luma is not None:
                    try:
                        cand_luma = measure_luma(trial)
                        if cand_luma is not None:
                            luma_score = -abs(cand_luma - target_luma)
                    except Exception:
                        pass
                score = (is_dup_free, size_ok, is_relevant, aesthetic_val, luma_score, min_d)
                if score > best_score:
                    if best_path and best_path != trial:
                        try:
                            os.remove(best_path)
                        except OSError:
                            pass
                    best_path, best_score, pick = trial, score, p
                elif trial != best_path:
                    try:
                        os.remove(trial)
                    except OSError:
                        pass
                if is_dup_free and size_ok and is_relevant:
                    good_seen += 1
                    if good_seen >= good_needed:
                        break   # набрали, сколько нужно для честного сравнения — не жжём оставшиеся попытки
            if best_path is None:
                pick = candidates[0]
                download(pick, cf)
            else:
                os.replace(best_path, cf)
        if used_ids is not None:
            used_ids.add(pick.get("id"))
        if used_hashes is not None:
            try:
                used_hashes.append(ahash(cf))
            except Exception:
                pass
        if recent_sizes is not None:
            try:
                recent_sizes.append(estimate_shot_size(cf))
            except Exception:
                pass
        return cf
    except Exception as e:
        PEXELS_BROKEN = True
        print(f"  Pexels [{query}]: {e}")
        return None


def local_photo(index):
    photos = sorted([f for f in os.listdir(MEDIA_FOLDER)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0) \
        if os.path.isdir(MEDIA_FOLDER) else []
    if not photos:
        return None
    return os.path.join(MEDIA_FOLDER, photos[index % len(photos)])


GENERIC_FALLBACKS = [
    "medieval sword still life", "knight armor moody light",
    "medieval castle atmosphere", "old manuscript parchment history",
    "cinematic dark fantasy weapon",
]


def query_for(text, keyword_counts=None):
    """keyword_counts (опционально, мутируется на месте) — значение THEMES
    может быть СПИСКОМ синонимичных запросов, не только строкой. Найден
    вживую реальный источник дублей на готовом ролике: "меч" — самое частое
    слово в видео буквально про вес меча, matches keyword напрямую (не
    через generic-заглушку с её собственной ротацией) — один и тот же
    запрос "medieval sword close up" уходил в Pexels 35 раз на 150 кадров
    эпизода. У Pexels физически нет 35 визуально различимых фото на один
    узкий запрос — сколько ни доверяй анти-дублю (pexels_photo), пул
    исчерпывается. Ротация по списку синонимов делит нагрузку на несколько
    независимых пулов вместо одного."""
    tl = text.lower()
    for kw, q in THEMES.items():
        if kw in tl:
            if isinstance(q, list):
                idx = keyword_counts.get(kw, 0) if keyword_counts is not None else 0
                if keyword_counts is not None:
                    keyword_counts[kw] = idx + 1
                return q[idx % len(q)]
            return q
    return None


def resolve_queries(blocks):
    """Прямой поиск по themes.json ловит не все блоки — короткие связки
    ("Не дрались. Несли.", "Береги себя.") и абстрактные куски без предметных
    слов улетают в generic-заглушку, которая никак не привязана к теме ролика.
    Вместо одной фиксированной строки на все такие блоки: сперва пробуем
    унаследовать запрос соседнего блока той же секции (тема раздела обычно
    не меняется от предложения к предложению), и только если во всей секции
    вообще ничего не нашлось — берём по кругу из GENERIC_FALLBACKS (не один
    и тот же текст на всё, иначе Pexels отдаёт одну и ту же жалкую пятёрку)."""
    keyword_counts = {}
    raw = [query_for(b["text"], keyword_counts) for b in blocks]
    resolved = list(raw)
    for i, q in enumerate(resolved):
        if q is not None:
            continue
        for j in range(i - 1, -1, -1):
            if blocks[j]["section"] != blocks[i]["section"]:
                break
            if raw[j] is not None:
                resolved[i] = raw[j]
                break
        if resolved[i] is not None:
            continue
        for j in range(i + 1, len(blocks)):
            if blocks[j]["section"] != blocks[i]["section"]:
                break
            if raw[j] is not None:
                resolved[i] = raw[j]
                break
    fallback_n = 0
    for i, q in enumerate(resolved):
        if q is None:
            resolved[i] = GENERIC_FALLBACKS[fallback_n % len(GENERIC_FALLBACKS)]
            fallback_n += 1
    return resolved


STAT_ACCENT = "0xC8102E"   # акцентный красный, CHANNEL.md house style
# D4: печатная машинка на 5-м варианте плашки (variant 4 из 5) — round-robin
# stat_variant % 5 уже даёт "не постоянно всегда" бесплатно (1 из 5 появлений
# цифры/даты в ролике), без отдельной классификации "это дата или нет" —
# [stat:...] и так по смыслу ВСЕГДА число/дата (см. ЧАСТЬ 9 script.txt),
# отдельный детектор не нужен. Тайминг посимвольного появления — константа,
# используется И в drawtext (визуал), И в main() при сборе таймкодов для
# звука печатной машинки (add_typewriter_clicks) — ОДНА и та же величина,
# иначе визуал и звук разъедутся по скорости печати.
TYPEWRITER_CHAR_DUR = 0.075
# Библиотека из 8 РЕАЛЬНЫХ отдельных ударов клавиш (не один клип, повторённый
# клонштампом на каждый символ — на реальной записи разные удары звучат чуть
# по-разному, один и тот же сэмпл на каждый символ читался механически,
# "не как в живой записи", по прямой обратной связи пользователя). Извлечены
# из присланной пользователем записи (scripts/generate — нет, это ручное
# разовое извлечение, см. коммит) — каждый нормализован к своему пику
# отдельно (не общий loudnorm), сохраняя естественный разброс громкости
# между разными ударами.
KEYBOARD_CLICKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "assets", "sfx", "keyboard_clicks")
KEYBOARD_CLICK_PATHS = sorted(glob.glob(os.path.join(KEYBOARD_CLICKS_DIR, "click_*.flac")))
TYPEWRITER_CLICK_ENABLED = os.environ.get("TYPEWRITER_CLICKS", "1") != "0" and bool(KEYBOARD_CLICK_PATHS)
TYPEWRITER_CLICK_GAIN_DB = -6.0


def typewriter_reveal_timing(n_chars, delay, dur, fin_dur=0.25, hold_margin=0.5):
    """Общая математика тайминга посимвольного появления — используется И в
    add_overlays() (визуал, локальное время клипа), И в main() при сборе
    таймкодов звука (реальное абсолютное время ролика) — ОДНА функция, чтобы
    визуал и звук печатной машинки не могли разъехаться по скорости из-за
    двух независимо подправленных копий одной и той же формулы клэмпа."""
    hold = max(delay + fin_dur, dur - hold_margin)
    cd = TYPEWRITER_CHAR_DUR
    if n_chars > 0 and delay + n_chars * cd > hold:
        cd = max(0.02, (hold - delay) / n_chars)
    return hold, cd


def pick_stat_variant(stat_variant, media_path=None):
    """D5: среди "статичных" оформлений плашки (0/1/2/3 — баннер-бокс/
    крупный центр без бокса/боковой акцент/мелкий угол) выбор не чисто по
    кругу, а с поправкой на РЕАЛЬНЫЙ контраст/плотность кадра под ней:
    - плотный кадр (busy>0.55, много локальных перепадов — макро/деталь) ->
      мелкий угловой (3), крупная плашка перекрыла бы сюжет;
    - малоконтрастный фон (узкий диапазон белая-чёрная точка) -> крупный
      центральный текст без бокса (1) читается безопасно, бокс тут не нужен;
    - просторный И контрастный кадр -> боковой акцент (2), есть настоящее
      негативное пространство под полосу+текст;
    - смешанный случай -> классический баннер с боксом (0), самый
      универсально безопасный (бокс сам создаёт контраст, не полагается на
      то, что уже есть в кадре).
    media_path=None (видео-клип или PIL/np недоступны) -> откат на прежний
    чистый round-robin (%4) — не регрессия, а прежнее поведение."""
    if media_path is None:
        return stat_variant % 4
    try:
        busy = estimate_busyness(media_path)
        levels = measure_levels(media_path)
        contrast = (levels[1] - levels[0]) if levels else 0.5
    except Exception:
        return stat_variant % 4
    if busy > 0.55:
        return 3
    if contrast < 0.35:
        return 1
    if busy < 0.25 and contrast > 0.55:
        return 2
    return 0


def add_overlays(vf_base, dur, title=None, stat=None, stat_variant=0, stat_delay=0.0, media_path=None):
    """Титр секции (низ кадра, слайд+fade первые ~2.5с, фирменный дисплейный
    шрифт) + цифровая плашка. Общий код для фото и видео-стока. Обычный
    fade раньше был самым узнаваемым штампом автослайдшоу — слайд добавляет
    "кто-то это анимировал руками".

    Плашка (2.2): раньше ОДИН и тот же красный баннер на каждую цифру за
    ролик — после 3-4 повторов за 16 минут читается как "шаблон программы",
    не дизайнерское решение. 4 варианта оформления, чередуются по позиции
    цифры в ролике (main() считает running-индекс): классический баннер /
    крупное число по центру без плашки / боковой акцент с полосой / мелкий
    текст в углу с подчёркиванием."""
    vf = vf_base
    # Титр раздела — вторичный уровень иерархии (Montserrat): это подпись,
    # не "геройский" факт-баннер — жирный, но не такой кричащий, как Benzin
    # на цифрах-доказательствах (stat ниже). Один и тот же шрифт на оба
    # уровня раньше читался как "программа так рисует", не решение монтажёра.
    if title and FONT_SECONDARY_PATH:
        hold = min(2.5, dur * 0.6)
        fin = max(0.3, hold * 0.3)
        text = title.upper() if FONT_IS_DISPLAY else title
        safe = escape_drawtext(text)
        fs = 46 if FONT_IS_DISPLAY else 54
        vf += (f",drawtext=fontfile='{FONT_SECONDARY_PATH}':text='{safe}':"
               f"fontcolor=white:fontsize={fs}:borderw=3:bordercolor=black@0.8:"
               f"x=(w-text_w)/2:"
               f"y='h-180+(1-min(t/{fin:.2f}\\,1))*36':"
               f"alpha='if(lt(t\\,{fin:.2f})\\,t/{fin:.2f}\\,"
               f"if(lt(t\\,{hold:.2f})\\,1\\,max(0\\,1-(t-{hold:.2f})/0.4)))'")
    if stat and FONT_PATH:
        fin_dur = 0.25
        # Плашка раньше ВСЕГДА стартовала от t=0 клипа — независимо от того,
        # где внутри фразы реально произносится число (см. stat_word_pos в
        # parse_blocks/split_long_blocks). Цифра появлялась на экране
        # заметно раньше, чем её произносили. delay — момент внутри клипа,
        # когда число уже озвучено; clamp оставляет минимум ~1.2с на сам
        # fade-in/hold/fade-out, даже если delay пришёлся почти на конец
        # клипа (не даём плашке исчезнуть, не успев появиться).
        delay = max(0.0, min(stat_delay, dur - 1.2))
        fin = delay + fin_dur
        hold = max(fin, dur - 0.5)
        text = stat.upper() if FONT_IS_DISPLAY else stat
        safe = escape_drawtext(text)
        alpha = (f"if(lt(t\\,{delay:.2f})\\,0\\,"
                 f"if(lt(t\\,{fin:.2f})\\,(t-{delay:.2f})/{fin_dur:.2f}\\,"
                 f"if(lt(t\\,{hold:.2f})\\,1\\,max(0\\,1-(t-{hold:.2f})/0.4))))")
        # D4: печатная машинка — фиксированный round-robin слот "не всегда"
        # (1 из 5 появлений цифры/даты), независимо от D5 ниже. D5: остальные
        # 4/5 появлений выбирают оформление по контрасту кадра, не по кругу.
        variant = 4 if stat_variant % 5 == 4 else pick_stat_variant(stat_variant, media_path)
        if variant == 0:
            # Классический баннер — верх кадра, слайд сверху.
            fs = 58 if FONT_IS_DISPLAY else 64
            vf += (f",drawtext=fontfile='{FONT_PATH}':text='{safe}':"
                   f"fontcolor=white:fontsize={fs}:"
                   f"box=1:boxcolor={STAT_ACCENT}@0.92:boxborderw=18:"
                   f"x=(w-text_w)/2:"
                   f"y='110-(1-max(0\\,min((t-{delay:.2f})/{fin_dur:.2f}\\,1)))*26':"
                   f"alpha='{alpha}'")
        elif variant == 1:
            # Крупное число по центру кадра — без плашки, тень вместо box.
            fs = 96 if FONT_IS_DISPLAY else 108
            vf += (f",drawtext=fontfile='{FONT_PATH}':text='{safe}':"
                   f"fontcolor=white:fontsize={fs}:"
                   f"shadowcolor=black@0.85:shadowx=4:shadowy=4:"
                   f"borderw=2:bordercolor=black@0.6:"
                   f"x=(w-text_w)/2:y=(h-text_h)/2:"
                   f"alpha='{alpha}'")
        elif variant == 2:
            # Боковой акцент — число слева, вертикальная полоса рядом.
            fs = 62 if FONT_IS_DISPLAY else 68
            # drawbox: w/h в x/y-выражениях ссылаются на СОБСТВЕННЫЕ параметры
            # бокса (w=8, h=160), не на кадр — иначе бы (h-160)/2 = 0
            # (самоссылка h=160-160=0), поймано на реальном рендере (полоса
            # уезжала в самый верх кадра). ih/iw — явно "кадр", не бокс.
            vf += (f",drawbox=x=100:y=(ih-160)/2:w=8:h=160:"
                   f"color={STAT_ACCENT}@0.92:t=fill:"
                   f"enable='between(t\\,{delay:.2f}\\,{hold:.2f})'"
                   f",drawtext=fontfile='{FONT_PATH}':text='{safe}':"
                   f"fontcolor=white:fontsize={fs}:"
                   f"shadowcolor=black@0.8:shadowx=3:shadowy=3:"
                   f"x=130:y=(h-text_h)/2:"
                   f"alpha='{alpha}'")
        elif variant == 3:
            # Минимальный — мелкий текст в нижнем правом углу, подчёркивание.
            # По смыслу это подпись, не геройская цифра — вторичный шрифт.
            fs = 34 if FONT_IS_DISPLAY else 38
            vf += (f",drawtext=fontfile='{FONT_SECONDARY_PATH}':text='{safe}':"
                   f"fontcolor=white:fontsize={fs}:"
                   f"borderw=2:bordercolor=black@0.7:"
                   f"x=w-text_w-60:y=h-110:"
                   f"alpha='{alpha}'"
                   f",drawbox=x=iw-260:y=ih-72:w=200:h=3:"
                   f"color={STAT_ACCENT}@0.92:t=fill:"
                   f"enable='between(t\\,{delay:.2f}\\,{hold:.2f})'")
        else:
            # D4: печатная машинка — сверено с референсом пользователя
            # (архивная аэросъёмка, дата "12.09.2020"): НЕ крупный "геройский"
            # центр кадра (как было раньше), а скромная подпись в нижней
            # трети — субтильный вес, тонкая тень вместо жирного борда, ближе
            # к "архивной датировке кадра", чем к рекламному баннеру. По X
            # оставлен центр (не копируем точный сдвиг референса влево — тот
            # кадр 9:16 с другим соотношением сторон, а текст variant 4 может
            # быть разной длины/языка, центр — единственное универсально
            # безопасное положение по горизонтали).
            # Символы появляются один за другим (та же ставка
            # TYPEWRITER_CHAR_DUR, что задаёт таймкоды звука в
            # main()/add_typewriter_clicks), после полного текста — обычная
            # задержка и fade-out, как у остальных вариантов. Цепочка
            # drawtext: filter k показывает text[:k+1] ТОЛЬКО в своём окне
            # [delay+k*cd, delay+(k+1)*cd) — сменяется следующим, длиннее на
            # один символ; последний остаётся до конца hold+fade-out.
            n_chars = len(text)
            _, cd = typewriter_reveal_timing(n_chars, delay, dur)
            fs = 56 if FONT_IS_DISPLAY else 62
            for k in range(n_chars):
                sub = escape_drawtext(text[:k + 1])
                seg_start = delay + k * cd
                seg_end = hold if k == n_chars - 1 else delay + (k + 1) * cd
                seg_alpha = alpha if k == n_chars - 1 else "1"
                vf += (f",drawtext=fontfile='{FONT_PATH}':text='{sub}':"
                       f"fontcolor=white:fontsize={fs}:"
                       f"shadowcolor=black@0.7:shadowx=1:shadowy=1:"
                       f"x=(w-text_w)/2:y=h*0.78-text_h/2:"
                       f"enable='between(t\\,{seg_start:.3f}\\,{seg_end:.3f})':"
                       f"alpha='{seg_alpha}'")
    return vf


CAPTION_FONT_SIZE = 76   # D2: крупные кинетические слова-акценты хука, Benzin-ExtraBold


def add_kinetic_captions(vf, captions):
    """D2: кинетические подписи хука — по одному слову крупным фирменным
    шрифтом (FONT_PATH), синхронно с РЕАЛЬНЫМ посимвольным таймингом
    произношения (см. load_hook_word_timings/hook_captions_for_block — тот
    же alignment.csv, что и везде в пайплайне, не оценка по words/wps).
    Только ХУК (main() зовёт это только для HOOK-блоков) — приём крупных
    слов-акцентов для самых первых секунд удержания, не постоянный субтитр
    на весь ролик (тот — write_subtitles(), полноразмерные описательные
    субтитры, другая задача).

    Позиция — нижняя треть, НЕ дно кадра (там уже фирменный титр раздела/
    полноразмерные субтитры) и не центр (там уже плашка-цифра, если есть) —
    все три уровня текста не должны драться за одно и то же место."""
    if not captions or not FONT_PATH:
        return vf
    for word, start, end in captions:
        text = word.upper() if FONT_IS_DISPLAY else word
        safe = escape_drawtext(text)
        if not safe:
            continue
        vf += (f",drawtext=fontfile='{FONT_PATH}':text='{safe}':"
               f"fontcolor=white:fontsize={CAPTION_FONT_SIZE}:"
               f"shadowcolor=black@0.85:shadowx=3:shadowy=3:"
               f"borderw=2:bordercolor=black@0.6:"
               f"x=(w-text_w)/2:y=h*0.66-text_h/2:"
               f"enable='between(t\\,{start:.3f}\\,{end:.3f})'")
    return vf


def kb_hash_choices(photo):
    """Кандидаты зума/пана по хэшу файла — детерминированно, но БЕЗ памяти о
    соседних клипах (это даёт anti-repetition в main(), см. pick_no_repeat)."""
    h = int(hashlib.md5(photo.encode()).hexdigest()[:8], 16)
    zoom_in = bool(h & 1)
    pan_dir = PAN_DIRECTIONS[(h >> 1) % len(PAN_DIRECTIONS)]
    return h, zoom_in, pan_dir


def estimate_busyness(photo_path):
    """Грубая мера "тесноты" кадра без depth-модели — просто плотность
    перепадов яркости на уменьшенной копии. Не настоящая детекция крупности
    плана, но дешёвый прокси: у тесного/плотного кадра (уже крупный план)
    перепадов много на каждый пиксель, у просторного — мало. Используется
    только чтобы НЕ зумить ещё сильнее то, что и так уже крупный план."""
    try:
        img = PILImage.open(photo_path).convert("L").resize((160, 90))
        arr = np.asarray(img, dtype=np.float32) if np is not None else None
        if arr is None:
            return 0.5
        gx = np.abs(np.diff(arr, axis=1)).mean()
        gy = np.abs(np.diff(arr, axis=0)).mean()
        return float(min(1.0, (gx + gy) / 40.0))
    except Exception:
        return 0.5


_FACE_CASCADE = None


def detect_face_anchor(photo_path):
    """Anchor-aware crop: раньше кроп под кадр 16:9 всегда бил ровно по
    центру исходного фото (increase+crop без смещения) — на портретных или
    непропорциональных стоковых кадрах это регулярно подрезало лицо/голову
    сверху или сбоку. Лёгкий локальный Haar-детектор (OpenCV, бесплатно,
    уже используется для параллакса — новых зависимостей и денег не стоит)
    ищет самое крупное лицо и возвращает точку привязки (fraction 0..1 по
    ширине/высоте ОРИГИНАЛА, чуть выше центра лица — глаза, не подбородок).
    Лицо не найдено/cv2 недоступен/фото не про людей (оружие, пейзаж) —
    возвращает None, и вызывающий код падает обратно на старый центр-кроп
    (то же поведение, что было всегда) — чистое дополнение без регрессии."""
    if not PARALLAX_LIBS:
        return None
    global _FACE_CASCADE
    try:
        if _FACE_CASCADE is None:
            _FACE_CASCADE = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        img = cv2.imread(photo_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        min_side = int(min(w, h) * 0.08)
        faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                minSize=(min_side, min_side))
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return ((fx + fw / 2) / w, (fy + fh * 0.35) / h)
    except Exception:
        return None


def spectral_saliency_anchor(photo_path, confidence_z=2.2):
    """Anchor-кроп по лицу (detect_face_anchor) покрывает ~10% кадров этого
    канала — остальное оружие/доспехи/предметы без единого лица в кадре, для
    них anchor всегда был None (центр-кроп). Saliency (spectral residual,
    Hou & Zhang 2007) — общий детектор "главного объекта" без лиц.

    Взял именно spectral residual, а не сырую плотность градиента/контраста
    (как estimate_busyness) — по конкретной причине: у сырого градиента
    текстурный ФОН (шершавый камень, кольчуга) часто даёт БОЛЬШЕ локального
    контраста, чем гладкий клинок на переднем плане — anchor утянуло бы к
    фону, кроп стал бы ХУЖЕ центрального, не лучше. Spectral residual вместо
    этого ищет то, что СТАТИСТИЧЕСКИ необычно относительно частотного
    профиля всего кадра — повторяющаяся текстура (камень, кольчуга) даёт
    низкий остаток именно потому, что она регулярна, один явный объект на
    таком фоне даёт пик. Чистый numpy (FFT) — cv2.saliency живёт только в
    opencv-contrib, которого нет в запиненном пакете (тот же класс промаха,
    что уже был с CascadeClassifier в этой сессии — не повторяем).

    ПОРОГ УВЕРЕННОСТИ (confidence_z): возвращаем anchor, только если пик
    saliency-карты заметно (z-score >= confidence_z) выше среднего уровня
    шума карты — на плоских/симметричных кадрах (широкий пейзаж, ткань)
    saliency не выражена вообще, и НАВЯЗЫВАТЬ смещение кропа туда, где нет
    реального сигнала, опаснее, чем оставить центр (см. критика раньше в
    этой же сессии). Слабый/отсутствующий пик -> None, откат на центр —
    то же safety-поведение, что и у detect_face_anchor."""
    if np is None:
        return None
    try:
        H, W = 36, 64
        img = PILImage.open(photo_path).convert("L").resize((W, H), PILImage.LANCZOS)
        gray = np.asarray(img, dtype=np.float64)
        # FFT считает кадр периодическим "по кругу" — резкий разрыв между
        # правым/левым и верхним/нижним краем сам создаёт ложную высокочастотную
        # "границу", которую spectral residual принимает за объект. Поймано
        # вживую на реальных фото канала: пик СИСТЕМАТИЧЕСКИ садился в первую
        # строку кадра (0/36) почти на каждом кадре — не совпадение, артефакт.
        # Окно Ханна плавно гасит яркость к краям перед FFT — убирает разрыв.
        win = np.outer(np.hanning(H), np.hanning(W))
        fft = np.fft.fft2(gray * win)
        amp = np.abs(fft)
        log_amp = np.log(amp + 1e-8)
        phase = np.angle(fft)
        kernel = np.ones((3, 3)) / 9.0
        avg_log_amp = _convolve2d_same(log_amp, kernel)
        residual = log_amp - avg_log_amp
        sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
        sal = _convolve2d_same(sal, np.ones((3, 3)) / 9.0)   # лёгкое сглаживание карты
        # Даже с окном Ханна самый край карты остаётся наименее надёжным —
        # исключаем внешнюю кайму из поиска пика (реальный объект в кадре
        # у стокового фотографа почти никогда не приклеен к самому краю).
        my, mx = max(1, H // 8), max(1, W // 8)
        interior = sal[my:H - my, mx:W - mx]
        if interior.size == 0:
            return None
        mean, std = float(sal.mean()), float(sal.std())
        # Верхний потолок на mean — поймано вживую: на ПОЧТИ однородном кадре
        # (плоский цвет/градиент без реальной текстуры) окно Ханна само по
        # себе создаёт плавный "холм" яркости, который residual-математика
        # принимает за огромный ложный пик (mean взлетает на 2-3 порядка
        # выше любого реального фото — проверено на 12 реальных кадрах
        # канала: 0.0005-0.00065, и на синтетическом плоском/градиентном
        # тесте: 0.33-0.74). Порог 0.02 — с большим запасом выше реальных
        # фото и с большим запасом ниже вырожденного случая.
        if std < 1e-9 or mean > 0.02:
            return None
        peak_idx = np.unravel_index(np.argmax(interior), interior.shape)
        z = (interior[peak_idx] - mean) / std
        if z < confidence_z:
            return None
        py, px = peak_idx[0] + my, peak_idx[1] + mx
        return ((px + 0.5) / W, (py + 0.5) / H)
    except Exception:
        return None


def _convolve2d_same(arr, kernel):
    """Мини-свёртка без scipy (не тянуть отдельную зависимость ради одной
    функции) — 'same'-паддинг через edge-repeat, реализация в лоб на
    маленькой (64x36) карте, скорость тут не критична."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(arr)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + arr.shape[0], j:j + arr.shape[1]]
    return out


def resolve_crop_anchor(photo_path):
    """Единая точка входа для anchor-кропа: лицо приоритетнее (выше точность,
    когда сработало), saliency — общий фоллбэк для кадров без лиц (оружие/
    доспехи/предметы, большинство кадров этого канала). Ни один сигнал не
    найден -> None, старый центр-кроп (ноль регрессии по построению)."""
    anchor = detect_face_anchor(photo_path)
    if anchor is not None:
        return anchor
    return spectral_saliency_anchor(photo_path)


def compute_crop_offset(iw, ih, cw, ch, anchor=None):
    """increase+crop геометрия (та же, что раньше делал ffmpeg force_original_
    aspect_ratio=increase сам, без смещения) — теперь считается в Python, чтобы
    можно было подставить anchor вместо жёсткого центра. anchor=None -> тот же
    центр-кроп, что был всегда (0 изменений в дефолтном поведении)."""
    scale = max(cw / iw, ch / ih)
    nw, nh = max(cw, round(iw * scale)), max(ch, round(ih * scale))
    if anchor:
        ax, ay = anchor
        cx, cy = ax * nw, ay * nh
        x0 = int(min(max(cx - cw / 2, 0), nw - cw))
        y0 = int(min(max(cy - ch / 2, 0), nh - ch))
    else:
        x0, y0 = (nw - cw) // 2, (nh - ch) // 2
    return nw, nh, x0, y0


SHOT_SIZES = ("wide", "medium", "close", "detail")


def estimate_shot_size(photo_path):
    """Четвёртая ось грамматики кадра (после функции блока) — масштаб
    плана: wide/medium/close/detail. Профессиональный монтаж постоянно
    МЕНЯЕТ масштаб визуальной информации между соседними кадрами; без
    этого сигнала пайплайн может случайно поставить подряд несколько
    общих или несколько крупных планов — по отдельности каждый кадр
    подобран хорошо, но подряд это читается как монотонность.

    Сигналы — все уже дешёвые и частично считаются в пайплайне для
    других целей, новой модели не тянем:
    1. Лицо и его относительный размер (Haar, уже есть для anchor-кропа)
       — самый надёжный сигнал, когда сработал.
    2. Плотность локального контраста (estimate_busyness) — тесный/
       текстурный кадр (макро металла, гравировка) даёт много перепадов
       на пиксель, просторный — мало.
    3. Уверенность spectral saliency — резкий узкий пик на спокойном
       фоне типичен для предметной съёмки крупным/средним планом
       (студийный сток), отсутствие пика — открытая сцена."""
    if PARALLAX_LIBS:
        try:
            global _FACE_CASCADE
            if _FACE_CASCADE is None:
                _FACE_CASCADE = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            img = cv2.imread(photo_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape[:2]
                min_side = int(min(w, h) * 0.08)
                faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                        minSize=(min_side, min_side))
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    face_frac = (fw * fh) / (w * h)
                    if face_frac > 0.06:
                        return "close"
                    if face_frac > 0.015:
                        return "medium"
                    return "wide"
        except Exception:
            pass
    busy = estimate_busyness(photo_path)
    if busy > 0.62:
        return "detail"
    if spectral_saliency_anchor(photo_path) is not None:
        return "medium"
    return "wide"


MOTION_MODES = ("classic_kb", "static_hold", "snap_push", "slow_pull",
                 "horizontal_pan", "micro_drift")


def choose_motion_mode(b, is_section_start, photo_hash, shot_size=None):
    """Все кадры "дышат" одним и тем же классом движения (smoothstep
    zoom+pan) — фирменный почерк автослайдшоу, даже с anti-repeat на
    выборе направления. Профессиональный монтаж чередует статику, быстрый
    push-in, медленный pull-out, чистую панораму, едва заметный дрифт.
    Выбор — по ФУНКЦИИ блока в первую очередь: цифра должна читаться (не
    ехать), reveal-кадр раздела — медленный отъезд, sub-cut деталь —
    почти неподвижна (не тянет внимание с главного кадра фразы).

    shot_size (см. estimate_shot_size) — вторая, более тонкая поправка
    ПОВЕРХ выбора по функции, не вместо него: широкий план (простор,
    открытая сцена) естественнее смотрится с панорамой/медленным отъездом,
    крупный/деталь — с едва заметным дрифтом или статикой (быстрый зум на
    и без того тесном кадре съедает сам объект). Дешёвая эвристика вместо
    отдельного классификатора "портрет/архитектура/пейзаж" — используем
    уже посчитанный сигнал, а не тянем новую модель ради категорий."""
    if b.get("stat"):
        return "static_hold"
    if is_section_start:
        return "slow_pull"
    if b["section"].startswith("HOOK"):
        return "snap_push" if (photo_hash & 2) else "classic_kb"
    if b.get("is_subcut"):
        if shot_size in ("close", "detail"):
            return "micro_drift"
        return "micro_drift" if (photo_hash & 4) else "static_hold"
    if shot_size == "wide":
        return "horizontal_pan" if (photo_hash & 1) else "slow_pull"
    if shot_size in ("close", "detail"):
        return "micro_drift" if (photo_hash & 1) else "static_hold"
    return "horizontal_pan" if (photo_hash % 7 == 0) else "classic_kb"


def classify_shot_function(b, is_section_start):
    """Грубая, но дешёвая категоризация "зачем этот кадр" — evidence (цифра-
    доказательство), context (reveal раздела), detail (sub-cut деталь одной
    фразы), hook (энергия хука), narrative (обычное повествование)."""
    if b.get("stat"):
        return "evidence"
    if b.get("is_subcut"):
        return "detail"
    if is_section_start:
        return "context"
    if b["section"].startswith("HOOK"):
        return "hook"
    return "narrative"


def _srt_timestamp(t):
    t = max(0.0, t)
    ms_total = int(round(t * 1000))
    h, rem = divmod(ms_total, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_subtitles(video_dir, blocks, starts, durs):
    """SRT — бесплатный побочный продукт уже посчитанного тайминга: реальный
    посимвольный alignment.csv (см. load_alignment_weights) уже участвует в
    расчёте durs/starts, отдельно парсить CSV второй раз не нужно. starts —
    РЕАЛЬНОЕ аудио-время (block_durations по total, не по раздутой под xfade
    target — та же модель, что уже использует energy_pace_multipliers для
    сэмплинга кривой громкости, см. main()), иначе субтитры разъехались бы
    с озвучкой на сумму xfade-нахлёстов к концу ролика.

    Один блок = один субтитр-кадр: текст уже чистый (теги вырезаны в
    parse_blocks), деления и так на границах пауз/фраз — не нужно заново
    резать по 5-7 слов, разбивка уже смысловая."""
    path = os.path.join(video_dir, "subtitles.srt")
    lines = []
    n = 0
    for b, s, d in zip(blocks, starts, durs):
        text = b["text"].strip()
        if not text:
            continue
        n += 1
        lines.append(str(n))
        lines.append(f"{_srt_timestamp(s)} --> {_srt_timestamp(s + max(d, 0.3))}")
        lines.append(text)
        lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return path


def write_chapters(video_dir, blocks, starts):
    """Список глав для описания YouTube — из уже известных границ секций
    (section_title()) и их реального старта в аудио (starts, см.
    write_subtitles). YouTube требует первую главу строго с 00:00 и минимум
    3 главы от 10с каждая, иначе не активирует таймлайн — это проверяется
    руками при вставке в описание, здесь только честный расчёт таймингов."""
    path = os.path.join(video_dir, "chapters.txt")
    seen, lines = set(), []
    for b, s in zip(blocks, starts):
        if b["section"] in seen:
            continue
        seen.add(b["section"])
        label = section_title(b["section"])
        if label is None:
            label = "Хук" if b["section"].startswith("HOOK") else (
                "Итоги" if b["section"].startswith("FINAL") else b["section"])
        # Заголовки блоков в script.txt пишутся КАПСОМ для видимости внутри
        # сценария (не для показа зрителю) — в реальном описании YouTube это
        # читается как крик. Трогаем только то, что ЦЕЛИКОМ капс — намеренно
        # смешанный регистр не портим.
        if label.isupper():
            label = label[0] + label[1:].lower()
        t = max(0.0, s)
        h, rem = divmod(int(t), 3600)
        m, sec = divmod(rem, 60)
        ts = f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
        lines.append(f"{ts} {label}")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return path


def write_shot_manifest(video_dir, blocks, durs, queries):
    """3.4: JSON-манифест по клипу — что это за кадр (function/motion_mode/
    запрос), не только "какая картинка досталась по номеру". Раньше связь
    блок->медиа была неявной, проверить "не стоят ли 5 context-кадров
    подряд" можно было только глазами по всему готовому рендеру. Манифест —
    читаемая опора для Шага 7.5 визуальной QC-проверки (CLAUDE.md), не
    отдельная фича ради фичи — сразу печатает подсказку о длинных пробегах
    одной функции подряд, чтобы не читать весь JSON руками, если паттерн
    уже виден невооружённым глазом.

    motion_mode_estimate тут ПРИБЛИЗИТЕЛЬНЫЙ — по хэшу текста блока, не
    реального файла фото (тот известен только во время рендера, здесь мы
    ещё даже не знаем, какая картинка достанется). Для обзорной QC-опоры
    точность до бита не нужна — расхождение возможно только на редких
    hash-tie-break внутри HOOK/обычных блоков (snap_push/classic_kb,
    horizontal_pan/classic_kb)."""
    entries = []
    for i, (b, d) in enumerate(zip(blocks, durs)):
        is_section_start = i == 0 or blocks[i]["section"] != blocks[i - 1]["section"]
        h = int(hashlib.md5(b["text"][:40].encode()).hexdigest()[:8], 16)
        mode = choose_motion_mode(b, is_section_start, h)
        section_slug = re.sub(r'[^A-Za-z0-9]+', '_', b["section"])[:24].strip('_')
        entries.append({
            "shot_id": f"{i:03d}_{section_slug}",
            "index": i,
            "section": b["section"],
            "duration_sec": round(d, 2),
            "function": classify_shot_function(b, is_section_start),
            "motion_mode_estimate": mode,
            "is_subcut": bool(b.get("is_subcut")),
            "has_stat": bool(b.get("stat")),
            "query": queries[i],
            "text_preview": b["text"][:60],
        })
    path = os.path.join(video_dir, "media_plan", "shot_manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    run_func, run_len, warn = None, 0, []
    for e in entries:
        if e["function"] == run_func:
            run_len += 1
        else:
            run_func, run_len = e["function"], 1
        if run_len == 5:
            warn.append(f"{run_func}@{e['index']-4}-{e['index']}")
    if warn:
        print(f"  Shot manifest: пробеги одной функции подряд (>=5) — {warn}")
    return path


WOBBLE_AMP_CANVAS_PX = 6.0   # ~1.4px в итоговом 1920-кадре (канва 8000px) — см. kenburns()
FREEZE_HOLD_DUR = 0.35       # секунд остановки движения перед stat-плашкой — см. kenburns()


def piecewise_ease_expr(t_expr, points):
    """Составная (multi-phase) траектория вместо одной гладкой кривой на
    весь клип — "push → micro-pan → settle" или "pull → reveal → hold"
    вместо равномерного smoothstep от начала до конца. points — контрольные
    точки [(0.0,0.0), ..., (1.0,1.0)] по возрастанию t; между соседними
    точками — НЕЗАВИСИМЫЙ smoothstep (гладкая скорость на стыке фаз, не
    рывок при переходе между ними, а не кусочно-линейная ломаная).
    Возвращает строку ffmpeg-eval выражения, t_expr — переменная времени
    0..1 (обычно on/frames)."""
    expr = f"{points[-1][1]:.4f}"
    for i in range(len(points) - 1, 0, -1):
        t0, v0 = points[i - 1]
        t1, v1 = points[i]
        local = f"(({t_expr}-{t0:.4f})/{(t1 - t0):.4f})"
        smooth = f"(3*pow({local},2)-2*pow({local},3))"
        seg_val = f"({v0:.4f}+{(v1 - v0):.4f}*{smooth})"
        expr = f"if(lt({t_expr}\\,{t1:.4f})\\,{seg_val}\\,{expr})"
    return expr


def kenburns(photo, out, dur, title=None, zoom_in=None, pan_dir=None, stat=None,
             section="", motion_mode="classic_kb", stat_variant=0,
             brightness_bias=0.0, energy_bias=0.0, stat_delay=0.0, levels=None, captions=None):
    frames = max(1, round(dur * FPS))
    h, zoom_in_default, pan_dir_default = kb_hash_choices(photo)
    if zoom_in is None:
        zoom_in = zoom_in_default
    dx, dy = pan_dir if pan_dir is not None else pan_dir_default
    rate_jit = ((h >> 5) % 1000) / 1000.0
    pan_jit = ((h >> 15) % 1000) / 1000.0
    # C6: скорость зума в такт энергии голоса — на эмоциональных пиках речи
    # камера "подбирается" быстрее, на спокойных местах медленнее, тот же
    # energy_bias, что уже модулирует контраст/сатурацию в film_look().
    # Клэмп ZOOM_DELTA_MIN/MAX всё равно применяется ПОСЛЕ — энергия сдвигает
    # в пределах уже безопасного диапазона, не расширяет его.
    energy_zoom_mult = 1.0 + max(-0.5, min(0.5, energy_bias)) * 0.2
    delta = max(ZOOM_DELTA_MIN, min(ZOOM_DELTA_MAX,
                ZOOM_RATE_BASE * dur * (0.75 + rate_jit * 0.5) * energy_zoom_mult))
    # Уже тесный/плотный кадр (крупный план) не нуждается в таком же зуме,
    # как просторный — иначе на кадре и так крупного меча к концу клипа
    # остаётся одна текстура металла, некуда уже приближаться со смыслом.
    busy = estimate_busyness(photo)
    if busy > 0.6:
        delta *= 0.7
    # PAN_SAFETY * jitter всегда < 1.0 (проверено численно) — офсет остаётся
    # строго внутри геометрического запаса (1-1/zoom)/2 на любом кадре клипа.
    pan_amt = PAN_SAFETY * (PAN_JITTER_MIN + pan_jit * (PAN_JITTER_MAX - PAN_JITTER_MIN))
    on_expr = "on"
    # Freeze-hold: короткая остановка движения ПРЯМО ПЕРЕД тем, как
    # появляется stat-плашка (stat_delay — см. main()/add_overlays,
    # тот же момент, синхронизированный с реальной озвучкой числа).
    # Контраст "движение → пауза → цифра" — приём, отличающий монтаж
    # с намерением от бесшовной анимации без акцентов. Замораживаем
    # именно ПЕРЕМЕННУЮ ВРЕМЕНИ (виртуальный "on"), а не сами zoom/pan —
    # эффект автоматически подхватывается ЛЮБЫМ motion_mode без
    # дублирования логики в каждой ветке. Движение после паузы продолжает
    # с того же места, откуда остановилось (не теряет фазу), а не рестартует.
    if stat and stat_delay > 0.5:
        freeze_start_f = max(0, round((stat_delay - FREEZE_HOLD_DUR) * FPS))
        freeze_end_f = round(stat_delay * FPS)
        if freeze_end_f > freeze_start_f:
            on_expr = (f"if(lt(on\\,{freeze_start_f})\\,on\\,"
                       f"if(lt(on\\,{freeze_end_f})\\,{freeze_start_f}\\,"
                       f"on-{freeze_end_f - freeze_start_f}))")
    t = f"({on_expr}/{frames})"
    # smoothstep вместо линейного роста: старт/финиш без рывка — линейный зум
    # это самый узнаваемый штамп шаблонных слайд-шоу.
    eased = f"(3*pow({t},2)-2*pow({t},3))"

    if motion_mode == "static_hold":
        # Цифра должна читаться — кадр не едет вообще. Лёгкий "дышащий" зум
        # (не 1.0 буквально — совсем мёртвый стоп-кадр сам по себе штамп)
        # без пана.
        delta = min(delta, 0.02)
        pan_amt = 0.0
    elif motion_mode == "classic_kb":
        # Multi-phase вместо одной ровной кривой на весь клип: "push →
        # micro-pan → settle" — быстрый ход в первой половине, спокойное
        # продолжение, гладкая остановка перед резом. Одна гладкая кривая
        # от начала до конца — сама по себе узнаваемый штамп "программа
        # анимировала", реальная камера редко держит идеально постоянную
        # скорость до самого конца движения.
        eased = piecewise_ease_expr(t, [(0.0, 0.0), (0.5, 0.72), (0.85, 0.95), (1.0, 1.0)])
    elif motion_mode == "snap_push":
        # Быстрый punch-in за ~0.3с, дальше — hold на максимуме. eased
        # клампится к 1.0 рано (по доле кадров, не по общей длительности —
        # короткий hook-кадр и так короткий).
        snap_frac = min(0.95, max(0.05, 8 / frames))
        eased = f"min(1,{t}/{snap_frac})"
        eased = f"(3*pow({eased},2)-2*pow({eased},3))"
        zoom_in = True
        delta = max(delta, ZOOM_DELTA_MIN * 1.4)
        pan_amt *= 0.4   # пан почти не нужен — внимание на резкость punch-in
    elif motion_mode == "slow_pull":
        # Reveal кадр раздела: стартуем зумленными, спокойно отъезжаем.
        # Без пана — это медленный отъезд, а не комбинация приёмов.
        # Multi-phase: "pull → reveal → hold" — основной отъезд в первой
        # половине-двух третях клипа, дальше движение гасится к спокойному
        # hold перед резом (реальный оператор не продолжает тянуть камеру
        # рывками ровно до конца кадра).
        zoom_in = False
        delta = max(delta, ZOOM_DELTA_MIN * 1.6)
        pan_amt = 0.0
        eased = piecewise_ease_expr(t, [(0.0, 0.0), (0.55, 0.68), (0.85, 0.92), (1.0, 1.0)])
    elif motion_mode == "horizontal_pan":
        # Почти чистая панорама — для длинных объектов (клинки, карты,
        # гравюры). ВАЖНО: offset пана ниже завязан на (1-1/zoom) — растёт
        # ВМЕСТЕ с изменением зума, а не по отдельной шкале времени. При
        # delta=0 (зум строго константа) этот множитель тоже был бы
        # константой — пан физически не двигался бы кадр от кадра (поймано
        # при проверке формулы, не вживую). Держим delta малой, но не
        # нулевой — зум еле заметен, а пан (усиленный) реально едет.
        delta = min(delta, 0.06)
        pan_amt = PAN_SAFETY * 0.95
    elif motion_mode == "micro_drift":
        # Едва заметное движение — не тянет внимание с главного кадра
        # sub-cut-фразы.
        delta = min(delta, 0.03)
        pan_amt = 0.0
    max_zoom = ZOOM_FLOOR + delta
    z = (f"'{ZOOM_FLOOR}+{delta:.5f}*{eased}'" if zoom_in
         else f"'{max_zoom:.5f}-{delta:.5f}*{eased}'")
    # Film weave: математически идеальная smoothstep-кривая зума/пана — сама
    # по себе штамп "отрендерено алгоритмом", у настоящей камеры/плёнки
    # всегда есть микро-дрожь. Низкочастотный синус (не шум per-frame — это
    # выглядело бы как тряска, не имитация плёнки), амплитуда и фаза/период —
    # детерминированно по хэшу фото. Меньше цикла за весь клип (0.6-1.4) —
    # чтобы не читалось как заметное "покачивание туда-сюда".
    wob_amp = WOBBLE_AMP_CANVAS_PX * (0.5 if motion_mode == "static_hold" else 1.0)
    wob_px = f"{wob_amp:.2f}*sin(2*PI*(on/{frames})*{0.6 + ((h >> 19) % 100) / 125.0:.3f}+{((h >> 3) % 628) / 100.0:.3f})"
    wob_py = f"{wob_amp:.2f}*sin(2*PI*(on/{frames})*{0.6 + ((h >> 23) % 100) / 125.0:.3f}+{((h >> 11) % 628) / 100.0:.3f})"
    # margin = (1-1/zoom)/2 считается от РЕАЛЬНОГО zoom в текущем кадре (переменная
    # zoompan даёт его сама) — офсет безопасен на любом кадре клипа по построению,
    # а не только к концу движения, как раньше при завязке на delta.
    x = (f"'iw/2-(iw/zoom/2){dx * pan_amt:+.5f}*(1-1/zoom)/2*iw+{wob_px}'" if dx
         else f"'iw/2-(iw/zoom/2)+{wob_px}'")
    y = (f"'ih/2-(ih/zoom/2){dy * pan_amt:+.5f}*(1-1/zoom)/2*ih+{wob_py}'" if dy
         else f"'ih/2-(ih/zoom/2)+{wob_py}'")
    # increase+crop (не decrease+pad) — исходные фото редко ровно 16:9, "вписать
    # в рамку" оставляло чёрные поля по краям на большинстве кадров (проверено:
    # 3 из 8 тестовых кадров с полосами по обеим сторонам). Залить кадр целиком
    # и обрезать лишнее — тот же приём, что уже применялся к видео-стоку ниже.
    # Anchor-aware: кроп бьёт по центру, только если ни лицо, ни saliency
    # не нашли уверенного сигнала (см. resolve_crop_anchor) — иначе
    # смещается так, чтобы не резать голову/главный объект.
    try:
        iw, ih = PILImage.open(photo).size
    except Exception:
        iw, ih = 8000, 4500
    anchor = resolve_crop_anchor(photo)
    nw, nh, cx0, cy0 = compute_crop_offset(iw, ih, 8000, 4500, anchor)
    vf_base = (f"scale={nw}:{nh},"
               f"crop=8000:4500:{cx0}:{cy0},setsar=1,"
               f"zoompan=z={z}:x={x}:y={y}:"
               f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
               f"{film_look(h, section, brightness_bias, energy_bias, levels)}")
    vf_overlay = None
    if title or stat or captions:
        vf_overlay = add_overlays(vf_base, dur, title, stat, stat_variant, stat_delay, media_path=photo)
        vf_overlay = add_kinetic_captions(vf_overlay, captions)

    def render(vf):
        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", photo]
        if GRAIN_ENABLED:
            # Зерно — отдельный вход (зацикленный grain_loop.mp4), не строка
            # фильтров — filter_complex вместо -vf, тот же приём, что уже
            # используется для speed ramp/halation embedding.
            cmd += ["-stream_loop", "-1", "-i", GRAIN_LOOP_PATH]
            fc = f"[0:v]{vf}[gr_in];{grain_blend_complex('gr_in', 1, 'vout')}"
            cmd += ["-filter_complex", fc, "-map", "[vout]"]
        else:
            cmd += ["-vf", vf]
        cmd += ["-t", str(dur), "-c:v", "libx264", "-preset", "fast",
               "-crf", "18", "-r", str(FPS)] + CLIP_PIX_ARGS + COLOR_META_ARGS + [out]
        return subprocess.run(cmd, capture_output=True, text=True)

    if vf_overlay:
        r = render(vf_overlay)
        if r.returncode == 0:
            return True
        # Титр/плашка (drawtext/шрифт) — не повод терять весь кадр: некоторые
        # сборки ffmpeg собраны без drawtext вообще. Откат на версию без них.
        print(f"  титр/плашка не встали ({os.path.basename(out)}), рисую без них: {r.stderr[-200:]}")
    return render(vf_base).returncode == 0


# --- 2.5D-параллакс (опционально, нужны torch/transformers/opencv) ---
# Depth-Anything-V2-Small с Hugging Face: по одной фотке строит карту глубины
# (белое — близко, чёрное — далеко), дальше "ближние" пиксели двигаются
# сильнее "дальних" при панорамировании — настоящее ощущение объёма вместо
# плоского зума. GitHub codeload (откуда обычно тянут MiDaS) закрыт сетевой
# политикой (проверено: 403) — huggingface.co открыт, поэтому источник модели
# именно оттуда.
PARALLAX_MARGIN = 1.5     # рабочий холст на 50% больше кадра — запас под
                           # зум+пан+параллакс-смещение без чёрных дыр по краям
PARALLAX_PX_BASE = 55.0   # максимальный доп.разброс между ближним/дальним слоем, px
# --- CLIP-релевантность медиа (опционально, нужны torch/transformers) ---
# Главная реально подтверждённая на QC этого канала проблема: текстовый
# поиск Pexels иногда возвращает кандидата, который ПОХОЖ по ключевым
# словам, но по смыслу мимо (найдено вживую — клипы кино/танцевального боя
# проходили через запросы про мечи). CLIP умеет реально сравнить картинку
# с текстом ЗАПРОСА (не с русским текстом блока — CLIP англоязычный, а
# query уже на английском, см. THEMES/query_for) и дать число, а не
# доверять слепо тому, что поиск вернул top-N по ключевым словам.
CLIP_ENABLED = os.environ.get("CLIP_RELEVANCE", "1") != "0"
CLIP_BROKEN = False   # взводится только на системном сбое (модель/сеть), не на одной картинке
# Калибровано вживую на 01_ves-mecha: 20 верных пар (картинка, её реальный
# запрос) дали score 0.217-0.303 (среднее 0.262); 15 пар с заведомо
# посторонним запросом ("pizza restaurant", "office meeting" и т.п. — не
# просто другой вариант той же темы, а другая тема целиком) дали 0.118-0.201
# (среднее 0.164). Порог 0.19 — выше кластера настоящих непопаданий, ниже
# минимума верных пар с запасом, не режет близкие по смыслу варианты одной
# темы (те тоже пересекаются с верными по диапазону — это ОК, задача ловить
# явный промах, не выбирать идеальный вариант из синонимов).
CLIP_RELEVANCE_THRESHOLD = 0.19
_clip_model = None
_clip_processor = None


def get_clip_model():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(name).eval()
        _clip_processor = CLIPProcessor.from_pretrained(name)
    return _clip_model, _clip_processor


def clip_relevance(image_path, text):
    """Косинусная близость картинки и текста ЗАПРОСА (0..1, реалистичный
    диапазон на наших фото ~0.1-0.3, не 0..1 в бытовом смысле "процент
    похожести" — это сырой косинус эмбеддингов CLIP). None при отключённой
    фиче/сбое модели — вызывающий код тогда просто не гейтит по релевантности
    (безопасный откат, тот же принцип, что PARALLAX_LIBS/PARALLAX_BROKEN)."""
    global CLIP_BROKEN
    if not CLIP_ENABLED or CLIP_BROKEN:
        return None
    try:
        import torch
        model, processor = get_clip_model()
        img = PILImage.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_e = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((img_e @ txt_e.T)[0][0])
    except ImportError:
        CLIP_BROKEN = True
        return None
    except Exception:
        return None


# --- Эстетическая оценка кадра (LAION aesthetic predictor v1, линейная
# голова поверх CLIP ViT-B/32) — та же модель, что уже загружена для
# clip_relevance(), никакого нового тяжёлого скачивания. Голова — всего
# 513 чисел (512-мерный вес + bias), извлечена ОДИН РАЗ из официальных
# весов (shunk031/aesthetics-predictor-v1-vit-base-patch32 на HF — это
# репаковка оригинальных LAION-AI/aesthetic-predictor sa_0_4_vit_b_32_
# linear.pth в формат HF, GitHub-источник в этой сессии недоступен по
# сетевой политике, HF открыт) и сохранена как assets/aesthetic/*.npz —
# не .pth (pickle, может исполнять код при загрузке), голый numpy-массив,
# без риска.
AESTHETIC_ENABLED = os.environ.get("AESTHETIC_SCORE", "1") != "0"
_aesthetic_head = None


def get_aesthetic_head():
    global _aesthetic_head
    if _aesthetic_head is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "assets", "aesthetic", "laion_aesthetic_vit_b_32_head.npz")
        data = np.load(path)
        _aesthetic_head = (data["weight"], float(data["bias"]))
    return _aesthetic_head


def aesthetic_score(image_path):
    """Эстетическая оценка кадра — линейная регрессия поверх L2-нормали-
    зованного CLIP image embedding (формула подтверждена по исходному коду
    модели: image_embeds /= norm(...); prediction = predictor(image_embeds)).
    Диапазон на реальных фото ~3-7 (проверено вживую на 25 кадрах эпизода —
    топ-3 по оценке визуально заметно сильнее по композиции/свету, чем
    низ; самый низкий скор корректно поймал случайно затесавшийся в кэш
    нерелевантный кадр — рюкзак/ноутбук вместо меча). None при отключённой
    фиче/сбое — вызывающий код просто не использует критерий (безопасный
    откат, тот же принцип, что CLIP_BROKEN/PARALLAX_BROKEN)."""
    global CLIP_BROKEN
    if not AESTHETIC_ENABLED or not CLIP_ENABLED or CLIP_BROKEN:
        return None
    try:
        import torch
        model, processor = get_clip_model()
        img = PILImage.open(image_path).convert("RGB")
        inputs = processor(images=[img], return_tensors="pt")
        with torch.no_grad():
            vis_out = model.vision_model(pixel_values=inputs["pixel_values"])
            feat = model.visual_projection(vis_out.pooler_output)
        e = (feat / feat.norm(dim=-1, keepdim=True))[0].numpy()
        w, b = get_aesthetic_head()
        return float(e @ w + b)
    except ImportError:
        CLIP_BROKEN = True
        return None
    except Exception:
        return None


_depth_model = None


def get_depth_model():
    global _depth_model
    if _depth_model is None:
        from transformers import pipeline as hf_pipeline
        _depth_model = hf_pipeline(task="depth-estimation",
                                    model="depth-anything/Depth-Anything-V2-Small-hf")
    return _depth_model


def estimate_depth(canvas_bgr):
    """Карта глубины (float32, 0..1, 1=близко) — строится по УЖЕ обрезанному
    холсту (canvas_bgr из fill_crop_canvas), а не по исходному фото.
    Раньше модель считала глубину по несжатому оригиналу (его родное
    соотношение сторон), а карта потом растягивалась голым cv2.resize под
    размер холста — без кропа, который применялся к цветной картинке. При
    несовпадении соотношения сторон (портретное фото под альбомный холст)
    карта глубины съезжала и тянулась не по тому же кадру, что видит
    зритель: параллакс сдвигал пиксели по чужой геометрии — отсюда
    видимое искажение/"плавание" картинки. Считая глубину прямо по canvas,
    геометрия гарантированно совпадает."""
    model = get_depth_model()
    h, w = canvas_bgr.shape[:2]
    img = PILImage.fromarray(canvas_bgr[:, :, ::-1])  # BGR -> RGB
    out = model(img)
    depth = np.array(out["predicted_depth"], dtype=np.float32)
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        return np.full((h, w), 0.5, dtype=np.float32)
    depth = (depth - d_min) / (d_max - d_min)
    # Depth-Anything даёт РЕЗКИЙ край на границе объекта (перепад ~0..1 за
    # 1-2px — проверено: grad p99 ~0.08, максимум >2.0 на пиксель). При
    # remap с parallax_px до ~70px это превращает границу в артефакт
    # "двойного контура"/расслоения на резких силуэтах (рукоять меча на
    # фоне боке) — то, что видно глазом как "ИИ-искажение". Настоящей
    # физически верной параллакс-окклюзии тут всё равно нет (плоская
    # картинка, не многослойная сцена), поэтому лечим не резкость карты,
    # а её крутизну: широкое гауссово размытие растягивает переход через
    # много пикселей, и remap плывёт плавно вместо "разрыва" на 1-2px —
    # тот же приём, что использует Apple/Google в 3D-Ken-Burns эффектах.
    sigma = max(3.0, min(h, w) * 0.018)
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=sigma)
    return depth


def fill_crop_canvas(photo_path, cw, ch, anchor=None):
    """Тот же increase+crop, что и в ffmpeg-пути: заливаем холст целиком,
    обрезаем лишнее — без чёрных полос по краям (BGR для cv2). anchor —
    та же логика, что в compute_crop_offset (см. detect_face_anchor):
    None -> центр-кроп, как было всегда."""
    img = PILImage.open(photo_path).convert("RGB")
    iw, ih = img.size
    nw, nh, x0, y0 = compute_crop_offset(iw, ih, cw, ch, anchor)
    img = img.resize((nw, nh), PILImage.LANCZOS)
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    arr = np.array(img)[:, :, ::-1].copy()   # RGB -> BGR
    return arr


def parallax_kenburns(photo, out, dur, title=None, zoom_in=None, pan_dir=None, stat=None,
                       section="", stat_variant=0, brightness_bias=0.0, energy_bias=0.0,
                       stat_delay=0.0, levels=None, captions=None):
    """2.5D-версия kenburns(): собственный покадровый рендер (OpenCV remap)
    вместо ffmpeg zoompan — только так можно сделать смещение, зависящее от
    глубины пикселя. При любой накладке (модель не встала, ffmpeg-пайп упал)
    возвращает False — вызывающий код откатывается на обычный kenburns()."""
    global PARALLAX_BROKEN
    if PARALLAX_BROKEN:
        return False
    proc = None
    try:
        frames = max(1, round(dur * FPS))
        cw, ch = round(WIDTH * PARALLAX_MARGIN), round(HEIGHT * PARALLAX_MARGIN)
        canvas = fill_crop_canvas(photo, cw, ch, anchor=resolve_crop_anchor(photo))
        try:
            depth = estimate_depth(canvas)
        except Exception as e:
            # Сбой именно тут почти всегда системный (модель не встала /
            # сеть до HF Hub отвалилась), не проблема конкретной картинки —
            # дальше пытаться на каждом из оставшихся хайлайтов бессмысленно,
            # только тратим время на повторную (медленную) загрузку модели.
            print(f"  depth-модель недоступна, параллакс выключается на остаток ролика: "
                  f"{type(e).__name__} {e}")
            PARALLAX_BROKEN = True
            return False

        h, zoom_in_default, pan_dir_default = kb_hash_choices(photo)
        if zoom_in is None:
            zoom_in = zoom_in_default
        dx, dy = pan_dir if pan_dir is not None else pan_dir_default
        rate_jit = ((h >> 5) % 1000) / 1000.0
        pan_jit = ((h >> 15) % 1000) / 1000.0
        # C6: см. kenburns() — та же логика скорости зума в такт энергии голоса.
        energy_zoom_mult = 1.0 + max(-0.5, min(0.5, energy_bias)) * 0.2
        delta = max(ZOOM_DELTA_MIN, min(ZOOM_DELTA_MAX,
                    ZOOM_RATE_BASE * dur * (0.75 + rate_jit * 0.5) * energy_zoom_mult))
        max_zoom = ZOOM_FLOOR + delta
        pan_amt_frac = PAN_SAFETY * (PAN_JITTER_MIN + pan_jit * (PAN_JITTER_MAX - PAN_JITTER_MIN))
        # Content-aware сила эффекта: раньше parallax_px был один и тот же
        # для любой картинки — эффект работает эффектно только там, где в
        # карте глубины ЕСТЬ реальный контраст (чёткое разделение объект/
        # фон); на плоских/текстурных кадрах (макро металла, ткань, ровный
        # задник) та же сила выглядит как случайное дрожание, не параллакс,
        # потому что двигать там физически нечего. depth.std() — дешёвая
        # мера контраста уже посчитанной карты (без доп. модели). Референс
        # 0.22 — медиана std по реальным фото канала (проверено: 0.18-0.31
        # на 8 кадрах), не гадание. Границы [0.4, 1.4] — не гасим эффект
        # до нуля даже на самом плоском кадре (тогда параллакс-путь вообще
        # не имел бы смысла выбирать), но и не разгоняем сверх разумного.
        depth_contrast = float(depth.std())
        strength_gain = max(0.4, min(1.4, depth_contrast / 0.22))
        parallax_px = PARALLAX_PX_BASE * (0.7 + rate_jit * 0.6) * strength_gain
        # DEPTH_ZOOM_STRENGTH — насколько сильно ближние/дальние слои
        # расходятся по СКОРОСТИ ЗУМА при push/pull (не только по офсету
        # при пане, как раньше). Настоящая перспектива: у камеры, которая
        # реально приближается к сцене, ближние объекты растут в кадре
        # быстрее дальних — старая версия зумила все пиксели с одной и той
        # же скоростью независимо от глубины (это и есть "2D remap", не
        # честная 3D-репроекция). depth_zoom_strength тоже масштабируется
        # content-aware — той же логикой, что и parallax_px выше.
        depth_zoom_strength = 0.12 * strength_gain
        # C2: лёгкий 3D-наклон камеры (roll) — до TILT_MAX_DEG, направление
        # по хэшу фото. Независимая от зума/пана ось движения, добавляет
        # ощущение "камера не на рельсах, а в руках", не дублирует то, что
        # уже даёт depth-параллакс (тот двигает БЛИЖНИЕ/ДАЛЬНИЕ слои с разной
        # скоростью, это вращает кадр целиком).
        TILT_MAX_DEG = 0.6
        tilt_dir = 1 if (h & 0x100) else -1
        # Не крутим кадр под цифрой-плашкой — читаемость важнее, тот же
        # принцип, что static_hold у kenburns() (параллакс своей "статичной"
        # ветки не имеет, гасим только этот один эффект).
        tilt_theta_max = 0.0 if stat else math.radians(TILT_MAX_DEG * tilt_dir)

        cx, cy = cw / 2.0, ch / 2.0
        ox_grid, oy_grid = np.meshgrid(np.arange(WIDTH, dtype=np.float32),
                                        np.arange(HEIGHT, dtype=np.float32))

        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-"]
        # Зерно — второй вход, ДО -frames:v (тот должен остаться однозначно
        # выходной опцией — если между ним и следующим -i ничего не стоит,
        # ffmpeg привязал бы его ко входу grain, а не к выходу).
        if GRAIN_ENABLED:
            cmd += ["-stream_loop", "-1", "-i", GRAIN_LOOP_PATH]
        vf = film_look(h, section, brightness_bias, energy_bias, levels)
        if title or stat or captions:
            vf = add_overlays(vf, dur, title, stat, stat_variant, stat_delay, media_path=photo)
            vf = add_kinetic_captions(vf, captions)
        if GRAIN_ENABLED:
            fc = f"[0:v]{vf}[gr_in];{grain_blend_complex('gr_in', 1, 'vout')}"
            cmd += ["-filter_complex", fc, "-map", "[vout]"]
        else:
            cmd += ["-vf", vf]
        cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-r", str(FPS)] + CLIP_PIX_ARGS + COLOR_META_ARGS + [out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        for frame_i in range(frames):
            t = frame_i / frames
            eased = 3 * t ** 2 - 2 * t ** 3   # smoothstep — тот же профиль, что у zoompan-версии
            zoom = (ZOOM_FLOOR + delta * eased) if zoom_in else (max_zoom - delta * eased)
            pan_px = pan_amt_frac * eased * min(cw, ch) * 0.5 * ((1 - 1 / zoom))

            map_x0 = cx + (ox_grid - WIDTH / 2) / zoom + dx * pan_px
            map_y0 = cy + (oy_grid - HEIGHT / 2) / zoom + dy * pan_px
            map_x0 = np.clip(map_x0, 0, cw - 1)
            map_y0 = np.clip(map_y0, 0, ch - 1)

            d_here = cv2.remap(depth, map_x0, map_y0, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            extra = (d_here - 0.5) * parallax_px * eased   # центрировано: ближе/дальше среднего
            # 3D-репроекция (не только офсет при пане, но и СКОРОСТЬ ЗУМА
            # по глубине): d_here уже известен на "базовой" (единый zoom)
            # позиции — используем его как приближение реальной глубины
            # источника (двухпроходная аппроксимация, честный mesh-warp
            # избыточен для этой задачи). zoom_gain > 1 для близких
            # пикселей (d_here -> 1) — под push-in они растут БЫСТРЕЕ
            # среднего, зеркально для дальних (d_here -> 0) — это и есть
            # разница между "камера едет" (старая версия, все пиксели
            # растут с одной скоростью) и "камера едет СКВОЗЬ сцену с
            # глубиной" (ближние слои обгоняют дальние).
            zoom_gain = 1.0 + (d_here - 0.5) * depth_zoom_strength * eased
            zoom_gain = np.clip(zoom_gain, 0.88, 1.15)
            map_x = cx + (ox_grid - WIDTH / 2) / (zoom * zoom_gain) + dx * pan_px + dx * extra
            map_y = cy + (oy_grid - HEIGHT / 2) / (zoom * zoom_gain) + dy * pan_px + dy * extra
            # Лёгкий поворот кадра (roll) — независимая от зума/пана степень
            # свободы: камера не строго на рельсах, а слегка меняет угол.
            # Амплитуда мала (умещается в запас холста PARALLAX_MARGIN=1.5,
            # проверено вживую на отсутствие смазанных краёв), направление —
            # детерминировано по хэшу фото.
            if tilt_theta_max != 0.0:
                theta = tilt_theta_max * eased
                cos_t, sin_t = np.float32(math.cos(theta)), np.float32(math.sin(theta))
                rel_x, rel_y = map_x - cx, map_y - cy
                map_x = cx + rel_x * cos_t - rel_y * sin_t
                map_y = cy + rel_y * cos_t + rel_x * sin_t
            map_x = np.clip(map_x, 0, cw - 1)
            map_y = np.clip(map_y, 0, ch - 1)

            frame = cv2.remap(canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.wait(timeout=60)
        if proc.returncode != 0:
            print(f"  параллакс-рендер не встал ({os.path.basename(out)}): "
                  f"{proc.stderr.read().decode(errors='replace')[-200:]}")
            return False
        return True
    except Exception as e:
        print(f"  параллакс сорвался ({os.path.basename(out)}): {type(e).__name__} {e}")
        return False
    finally:
        # Раньше при сбое ПОСРЕДИ покадрового цикла (BrokenPipeError на
        # stdin.write, таймаут на wait) дочерний ffmpeg не убивался вообще —
        # завис бы висячим процессом до конца сессии. Гасим его в любом
        # случае, если он ещё жив к выходу из функции.
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass


# Не динамический ramp внутри клипа (это отдельная и намного более сложная
# задача — переменная PTS-кривая), а постоянная скорость по контексту клипа:
# хук чуть быстрее — энергичнее, финал чуть медленнее — снимает напряжение.
SPEED_BIAS = {"HOOK": 1.12, "FINAL": 0.92}


def measure_motion(vid, start, span, samples=4):
    """Дешёвая оценка реального движения в клипе (frame-diff на N мелких
    170x90-кадрах, тот же приём, что measure_luma) — speed ramp (см. ниже)
    имеет смысл только там, где в кадре РЕАЛЬНО что-то происходит (взмах,
    шаг, панорама съёмщика), а не когда объект просто лежит/стоит в кадре —
    там ускорение ничем не мотивировано и выглядит как рывок, не приём.
    0.0 при любой ошибке/отсутствии numpy — ramp тогда просто не применяется
    (безопасный откат на старое поведение, тот же принцип, что PARALLAX_LIBS)."""
    if np is None:
        return 0.0
    try:
        frames = []
        for i in range(samples):
            t = start + span * (i + 0.5) / samples
            tmp = vid + f"._motion_probe_{i}.jpg"
            r = subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", vid, "-frames:v", "1",
                 "-vf", "scale=160:90", tmp],
                capture_output=True, timeout=15)
            if r.returncode == 0 and os.path.exists(tmp):
                img = PILImage.open(tmp).convert("L")
                frames.append(np.asarray(img, dtype=np.float32))
                os.remove(tmp)
        if len(frames) < 2:
            return 0.0
        diffs = [float(np.abs(frames[i] - frames[i - 1]).mean()) for i in range(1, len(frames))]
        return (sum(diffs) / len(diffs)) / 255.0
    except Exception:
        return 0.0


# Slow -> fast -> normal внутри ОДНОГО клипа — подчёркивает момент реального
# действия (замах, шаг, столкновение), а не плоская постоянная скорость на
# весь план (то, что уже делает SPEED_BIAS выше на уровне целого клипа).
# Тройка (доля_длительности_на_выходе, множитель_скорости).
SPEED_RAMP_SEGMENTS = [(0.30, 0.85), (0.45, 1.20), (0.25, 1.00)]
SPEED_RAMP_MOTION_THRESHOLD = 0.02   # ниже — план статичен, ramp не гейтится сюда


def build_speed_ramp_filter(skip, dur, actual, label_in="base", label_out="ramped"):
    """[label_in] -> trim/setpts по SPEED_RAMP_SEGMENTS -> concat -> [label_out],
    как отдельный кусок filter_complex-графа (не обычный -vf, т.к. нужен
    concat нескольких setpts-кусков одного и того же входа). None, если
    исходника не хватает (safety margin) — вызывающий код тогда просто не
    применяет ramp, как есть."""
    needed = dur * sum(frac * mult for frac, mult in SPEED_RAMP_SEGMENTS)
    if actual - skip < needed + 0.05:
        return None
    n = len(SPEED_RAMP_SEGMENTS)
    # Один и тот же [label_in] нельзя подать входом сразу в 3 независимых
    # trim (ffmpeg не фанаутит именованный pad автоматически) — явный split.
    split_labels = [f"{label_out}_s{i}" for i in range(n)]
    parts = [f"[{label_in}]split={n}{''.join(f'[{s}]' for s in split_labels)}"]
    labels, pos = [], skip
    for i, (frac, mult) in enumerate(SPEED_RAMP_SEGMENTS):
        out_d = dur * frac
        src_d = out_d * mult
        lbl = f"{label_out}{i}"
        # Motion blur ТОЛЬКО на реально ускоренном (mult>1) сегменте — без
        # смаза ускорение через голый setpts выглядит цифровым рывком
        # (кадры просто чаще пропускаются), со смазом читается как
        # операторский приём. Замедленный/нормальный сегмент не трогаем —
        # там лишнее размытие только портит резкость, которую специально
        # вернули unsharp'ом (см. film_look). tmix не меняет число кадров/
        # длительность — безопасно вставлять здесь без пересчёта таймингов.
        blur = ",tmix=frames=3:weights='1 2 1'" if mult > 1.0 else ""
        parts.append(
            f"[{split_labels[i]}]trim=start={pos:.3f}:duration={src_d:.3f},"
            f"setpts=PTS-STARTPTS,setpts={1 / mult:.5f}*PTS{blur}[{lbl}]")
        labels.append(f"[{lbl}]")
        pos += src_d
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[{label_out}]")
    return ";".join(parts)


def detect_scene_change_offset(vid, max_skip, scene_threshold=0.12):
    """Сцен-анализ вместо чисто произвольного хэш-пропуска (см. video_render):
    ищем РЕАЛЬНЫЙ момент смены кадра/начала движения в допустимом окне
    [0, max_skip] штатным ffmpeg-детектором сцен (select=gt(scene,THRESH) —
    честный frame-diff, не полноценный optical flow, но быстро и без новых
    зависимостей). Берём САМУЮ ПОЗДНЮЮ найденную смену в пределах окна —
    максимально используем допустимый запас, но приземляемся точно на
    реальное событие в кадре, а не на случайное место.

    Порог уверенности: сцен не найдено (статичный до самого края план,
    либо анализ не удался) -> None, вызывающий код откатывается на старый
    детерминированный хэш-пропуск — то же safety-поведение, что у
    anchor-кропа/saliency (см. resolve_crop_anchor)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", vid, "-t", f"{max_skip:.2f}", "-vf",
             f"select='gt(scene\\,{scene_threshold})',showinfo", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30)
        times = [float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr)]
        candidates = [t for t in times if 0.1 <= t <= max_skip]
        return max(candidates) if candidates else None
    except Exception:
        return None


def video_render(vid, out, dur, title=None, stat=None, section="", stat_variant=0,
                  brightness_bias=0.0, energy_bias=0.0, stat_delay=0.0, levels=None,
                  handheld=False, captions=None):
    """Аналог kenburns(), но для стокового видео: без zoompan (движение уже
    есть в кадре), заливка кадра целиком + обрезка (не letterbox), растяжение
    по времени, если исходный ролик короче нужной длительности.

    handheld — лёгкая ручная тряска (C3): у фото уже есть film weave
    (WOBBLE_AMP_CANVAS_PX), у видео-стока движения камеры не было вообще —
    боевые сцены на стоке выглядели подозрительно гладко (будто со штатива).
    Заказывается по контенту (ACTION_WORDS в тексте блока, см. main()), не
    на каждый видео-кадр — на спокойных сюжетах тряска ничем не мотивирована.

    Speed ramp (slow->fast->normal, SPEED_RAMP_SEGMENTS) — отдельный путь
    через -filter_complex (нужен concat нескольких setpts-кусков ОДНОГО
    входа, обычный -vf так не умеет). Применяется только когда есть и запас
    исходника, и реальное движение в кадре (measure_motion) — иначе тихо
    падаем обратно на старый -vf путь ниже."""
    try:
        actual = get_media_duration(vid)
    except Exception:
        actual = dur
    h = int(hashlib.md5(vid.encode()).hexdigest()[:8], 16)
    if handheld:
        # Небольшой запас (3.5%) сверх целевого кадра под тряску — тот же
        # принцип, что PARALLAX_MARGIN у фото, просто меньше (тут не нужно
        # места под пан/зум, только под дрожание в несколько пикселей).
        ov_w, ov_h = round(WIDTH * 1.035), round(HEIGHT * 1.035)
        amp = 6.0
        ph1, ph2 = ((h >> 3) % 628) / 100.0, ((h >> 11) % 628) / 100.0
        shake_x = f"{amp:.1f}*sin(2*PI*t*0.9+{ph1:.2f})+{amp*0.6:.1f}*sin(2*PI*t*2.3+{ph2:.2f})"
        shake_y = f"{amp:.1f}*sin(2*PI*t*1.1+{ph2:.2f})+{amp*0.6:.1f}*sin(2*PI*t*2.7+{ph1:.2f})"
        scale_crop = (f"scale={ov_w}:{ov_h}:force_original_aspect_ratio=increase,"
                      f"crop={WIDTH}:{HEIGHT}:x='({ov_w}-{WIDTH})/2+{shake_x}':"
                      f"y='({ov_h}-{HEIGHT})/2+{shake_y}',setsar=1")
    else:
        scale_crop = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1"
    bias = next((v for k, v in SPEED_BIAS.items() if section.startswith(k)), 1.0)
    # Стоковое видео часто начинается со статичного/вялого кадра до того, как
    # начнётся реальное движение (панорама разгоняется, руки только подносят
    # предмет к камере) — первая секунда почти всегда самая скучная в клипе.
    # Пропуск возможен только если есть запас (actual заметно больше dur) —
    # иначе откусили бы от нужной длительности и пришлось бы растягивать
    # (setpts) остаток, что и так уже отдельная ветка ниже. Детерминированный
    # джиттер по хэшу файла — не одна и та же доля пропуска на каждом клипе.
    skip = 0.0
    if actual - dur > 0.6:
        max_skip = min(1.6, (actual - dur) * 0.5)
        scene_skip = detect_scene_change_offset(vid, max_skip)
        skip = scene_skip if scene_skip is not None else (
            max_skip * (0.5 + ((h >> 20) % 1000) / 1000.0 * 0.5))

    ramp_filter = None
    margin_needed = dur * sum(frac * mult for frac, mult in SPEED_RAMP_SEGMENTS)
    if actual - skip >= margin_needed + 0.05:
        motion = measure_motion(vid, skip, min(dur, actual - skip))
        if motion >= SPEED_RAMP_MOTION_THRESHOLD:
            ramp_filter = build_speed_ramp_filter(skip, dur, actual)

    if ramp_filter:
        tail = film_look(h, section, brightness_bias, energy_bias, levels)
        full_tail = tail
        if title or stat or captions:
            full_tail = add_overlays(tail, dur, title, stat, stat_variant, stat_delay)
            full_tail = add_kinetic_captions(full_tail, captions)
        cmd = ["ffmpeg", "-y", "-i", vid]
        if GRAIN_ENABLED:
            cmd += ["-stream_loop", "-1", "-i", GRAIN_LOOP_PATH]
            filter_complex = (f"[0:v]{scale_crop}[base];{ramp_filter};"
                               f"[ramped]{full_tail}[gr_in];{grain_blend_complex('gr_in', 1, 'vout')}")
        else:
            filter_complex = f"[0:v]{scale_crop}[base];{ramp_filter};[ramped]{full_tail}[vout]"
        cmd += ["-filter_complex", filter_complex,
               "-map", "[vout]", "-t", str(dur), "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-r", str(FPS)] + CLIP_PIX_ARGS + COLOR_META_ARGS + [out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return True
        print(f"  speed ramp не встал на видео ({os.path.basename(out)}), рисую без ramp: {r.stderr[-200:]}")

    vf_base = scale_crop
    setpts_factor = None
    if actual < dur - 0.05:
        setpts_factor = dur / max(actual, 0.1)   # уже растягиваем нехватку — bias не добавляем поверх
    elif bias != 1.0 and actual >= dur * bias:
        setpts_factor = 1.0 / bias
    if setpts_factor is not None:
        vf_base += f",setpts={setpts_factor:.5f}*PTS"
    vf_base += f",{film_look(h, section, brightness_bias, energy_bias, levels)}"
    vf_overlay = None
    if title or stat or captions:
        vf_overlay = add_overlays(vf_base, dur, title, stat, stat_variant, stat_delay)
        vf_overlay = add_kinetic_captions(vf_overlay, captions)

    def render(vf):
        cmd = ["ffmpeg", "-y"]
        if skip > 0.05:
            cmd += ["-ss", f"{skip:.2f}"]
        cmd += ["-i", vid]
        if GRAIN_ENABLED:
            cmd += ["-stream_loop", "-1", "-i", GRAIN_LOOP_PATH]
            fc = f"[0:v]{vf}[gr_in];{grain_blend_complex('gr_in', 1, 'vout')}"
            cmd += ["-filter_complex", fc, "-map", "[vout]"]
        else:
            cmd += ["-vf", vf]
        cmd += ["-t", str(dur), "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-r", str(FPS)] + CLIP_PIX_ARGS + COLOR_META_ARGS + [out]
        return subprocess.run(cmd, capture_output=True, text=True)

    if vf_overlay:
        r = render(vf_overlay)
        if r.returncode == 0:
            return True
        print(f"  титр/плашка не встали на видео ({os.path.basename(out)}), рисую без них: {r.stderr[-200:]}")
    return render(vf_base).returncode == 0


def pexels_video(query, index, used_ids=None):
    """Тот же принцип, что pexels_photo(): перебираем выдачу и берём первое
    ещё не показанное видео. Из доступных video_files берём ближайшее по
    ширине к целевому 1920 — не тянем 4K ради 1080p-выхода."""
    global PEXELS_BROKEN
    cache = os.path.join(TEMP_FOLDER, "pexels_video_cache")
    os.makedirs(cache, exist_ok=True)
    qhash = hashlib.md5(query.encode()).hexdigest()[:8]
    cf = os.path.join(cache, f"{index:04d}_{qhash}.mp4")
    if os.path.exists(cf):
        return cf
    if not PEXELS_API_KEY:
        return None
    try:
        q = urllib.parse.quote(query)
        req = urllib.request.Request(
            f"https://api.pexels.com/videos/search?query={q}&per_page=80&orientation=landscape",
            headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        videos = data.get("videos") or []
        if not videos:
            return None
        pick = videos[0]
        if used_ids is not None:
            for v in videos:
                if v.get("id") not in used_ids:
                    pick = v
                    break
        if used_ids is not None:
            used_ids.add(pick.get("id"))
        files = [f for f in (pick.get("video_files") or [])
                 if f.get("file_type") == "video/mp4" and f.get("width")]
        if not files:
            return None
        best = min(files, key=lambda f: abs(f["width"] - WIDTH))
        vid_req = urllib.request.Request(best["link"], headers={"User-Agent": UA})
        with urllib.request.urlopen(vid_req, timeout=40) as r:
            open(cf, "wb").write(r.read())
        return cf
    except Exception as e:
        PEXELS_BROKEN = True
        print(f"  Pexels video [{query}]: {e}")
        return None


SNAP_CUT_DUR = 0.03   # мгновенный рез после stat-плашки — короче обычного hardcut


def xfade_chain(clips, durs, sections, out, xfade_dur=XFADE_DUR, blocks=None):
    """Один проход filter_complex с цепочкой xfade между ВСЕМИ соседними
    кадрами — вместо жёсткой склейки. Переход = сигнал драматургии, не
    случайность: раньше тип перехода выбирался чисто хэшем путей файлов,
    без связи со сменой темы/функции блока — dissolve мог случайно попасть
    внутрь одной фразы, hardcut — между разделами. Теперь мотивированно
    (нужен параллельный список blocks — метаданные клипов, is_subcut/stat/
    section; без него откат на старое хэш-разнообразие для обратной
    совместимости):
    - граница секции → dissolve/fadeblack/fadewhite (смена темы)
    - sub-cut внутри одного исходного блока → только hardcut (это один кадр
      мысли, разрезанный на фазы — не повод анонсировать переход)
    - сразу после клипа со stat-плашкой → snap cut (мгновенный, короче
      обычного hardcut — цифра "впечаталась" и тут же сменилась)
    - весь хук → только hardcut (энергия, скорость, никаких dissolve)
    - остальное — 85% hardcut / 15% короткий dissolve (не полный xfade_dur)
    Возвращает (True, итоговая_длительность) или (False, 0.0), если
    что-то пошло не так (тогда main() откатывается на обычный concat -c copy)."""
    n = len(clips)
    if n < 2:
        return False, 0.0
    parts, prev_label, cum = [], "0:v", durs[0]
    cut_hist, boundary_hist = [], []
    for i in range(1, n):
        h = int(hashlib.md5(f"{clips[i-1]}|{clips[i]}".encode()).hexdigest()[:8], 16)
        is_boundary = sections[i] != sections[i - 1]
        b_prev = blocks[i - 1] if blocks else None
        b_cur = blocks[i] if blocks else None
        if is_boundary:
            # Смена темы — заметный переход, не обычная склейка. dissolve/fadeblack/
            # fadewhite (вспышка светом — читается как "новая глава начинается ярко")
            # вперемешку.
            candidate = BOUNDARY_TRANSITIONS[h % len(BOUNDARY_TRANSITIONS)]
            transition = pick_no_repeat(boundary_hist, candidate, BOUNDARY_TRANSITIONS, 1)
            this_dur = xfade_dur
        elif b_cur is not None and b_cur.get("is_subcut"):
            transition, this_dur = "fade", XFADE_DUR_HARD
        elif b_prev is not None and b_prev.get("stat"):
            transition, this_dur = "fade", SNAP_CUT_DUR
        elif b_cur is not None and sections[i].startswith("HOOK"):
            transition, this_dur = "fade", XFADE_DUR_HARD
        else:
            # Большинство склеек в реальном монтаже — жёсткий cut, не dissolve;
            # заметный переход — редкость, не норма. ~85% hard cut / ~15% короткий dissolve.
            candidate = "hardcut" if (h % 20 >= 3) else XFADE_TRANSITIONS[(h >> 8) % len(XFADE_TRANSITIONS)]
            choice = pick_no_repeat(cut_hist, candidate, ["hardcut"] + XFADE_TRANSITIONS, max_repeat=3)
            this_dur = XFADE_DUR_HARD if choice == "hardcut" else xfade_dur * 0.4
            transition = "fade" if choice == "hardcut" else choice
        offset = max(0.0, cum - this_dur)
        out_label = f"vx{i}" if i < n - 1 else "vout"
        parts.append(f"[{prev_label}][{i}:v]xfade=transition={transition}:"
                     f"duration={this_dur:.3f}:offset={offset:.3f}[{out_label}]")
        cum = cum + durs[i] - this_dur
        prev_label = out_label
    cmd = ["ffmpeg", "-y"]
    for c in clips:
        cmd += ["-i", c]
    # Это единственный проход, который видит ВСЮ склейку сразу — то, что
    # уйдёт на YouTube (финальный мукс делает -c:v copy, второго прохода
    # уже не будет). Каждый отдельный клип и так уже CRF 18 (см. kenburns/
    # video_render/parallax_kenburns) — если тут снова ужать до 23, это
    # второе поколение потерь поверх первого, заметное именно на тёмном
    # грейде (градиенты/дым/зерно). medium+CRF16 — разовая цена, не на
    # каждый маленький клип.
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16",
            "-pix_fmt", "yuv420p", "-r", str(FPS)] + COLOR_META_ARGS + [out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  xfade-склейка не удалась, откат на concat:", r.stderr[-300:])
        return False, 0.0
    # Пойман вживую на 150 клипах (после sub-cuts): ffmpeg возвращает 0
    # (успех), но при очень длинной цепочке последовательных xfade в ОДНОМ
    # filter_complex молча "роняет" кадры (dup=0 drop=20000+ в stderr) и
    # застревает на застывшем кадре с середины ролика — итоговый файл
    # правильной длины (после -shortest/-t на муксе), но 800+ секунд из
    # них заморожены. Return-код тут не индикатор, реальная длительность —
    # индикатор. На ~90 клипах (до sub-cuts) с той же схемой такого не
    # было — это масштабная проблема ffmpeg с очень длинной цепочкой в
    # одном графе, не баг в математике offset/cum (та проверена отдельно).
    try:
        real_dur = get_media_duration(out)
    except Exception:
        real_dur = 0.0
    expected = max(cum, 0.1)
    if real_dur < expected * 0.9 - 2.0:
        print(f"  xfade-склейка вернула 0, но реальная длительность {real_dur:.1f}с "
              f"против ожидаемых {expected:.1f}с (похоже на застревание ffmpeg на "
              f"длинной цепочке xfade) — откат на concat.")
        return False, 0.0
    return True, expected


XFADE_CHUNK_SIZE = 35   # порог чанкования — см. xfade_chain_chunked


def _chunk_bounds(n, sections, chunk_size):
    """(start,end) полуинтервалы индексов клипов на чанки ~chunk_size —
    резать ТОЛЬКО на границах section (там и так планировался заметный
    dissolve/fadeblack/fadewhite, см. xfade_chain — на стыке чанков он
    станет обычным concat-cut, это читается как ещё один hardcut, не как
    потеря приёма, в отличие от разрыва ПОСЕРЕДИНЕ фразы). Защита от
    вырожденного случая (вся секция длиннее 2×chunk_size — например,
    гигантский BODY без внутренних BLOCK) — режем принудительно, не гоняясь
    за границей до бесконечности: маленькая потеря одного перехода лучше
    риска снова упереться в ту же ffmpeg-багу на длинной цепочке."""
    if n <= chunk_size:
        return [(0, n)]
    bounds, pos = [0], 0
    while pos < n:
        target = min(pos + chunk_size, n)
        if target >= n:
            bounds.append(n)
            break
        j = target
        hard_cap = min(pos + chunk_size * 2, n)
        while j < hard_cap and sections[j] == sections[j - 1]:
            j += 1
        bounds.append(j)
        pos = j
    bounds = sorted(set(bounds))
    chunks = list(zip(bounds[:-1], bounds[1:]))
    # xfade_chain() требует минимум 2 клипа (n<2 -> (False, 0.0) сразу) —
    # де-фрагментируем случайный "хвостик" в 1 клип (получается, если
    # граница секции легла ровно на предпоследний индекс) слиянием с
    # соседним чанком, а не оставляем orphan-чанк, гарантированно
    # роняющий всю сборку в откат на concat.
    fixed = []
    for a, b in chunks:
        if b - a < 2 and fixed:
            pa, _ = fixed[-1]
            fixed[-1] = (pa, b)
        else:
            fixed.append((a, b))
    if len(fixed) > 1 and fixed[0][1] - fixed[0][0] < 2:
        (a0, _), (_, b1) = fixed[0], fixed[1]
        fixed[0] = (a0, b1)
        del fixed[1]
    return fixed


def xfade_chain_chunked(clips, durs, sections, out, temp_dir, xfade_dur=XFADE_DUR, blocks=None,
                         chunk_size=XFADE_CHUNK_SIZE):
    """Обёртка над xfade_chain(): на длинной цепочке (150+ клипов после
    sub-cuts) один filter_complex со всеми xfade сразу ловит документированный
    в xfade_chain() баг ffmpeg — молча роняет кадры и застревает на
    застывшем кадре с середины ролика, при том что return-код 0. Раньше
    единственным ответом был полный откат на голый concat — ролик собирался,
    но терял ВСЕ переходы разом. Вместо этого режем на чанки по chunk_size
    (см. _chunk_bounds — только по границам section), каждый чанк — свой
    независимый xfade_chain() (короткая цепочка, баг не всплывает), чанки
    склеиваются -c copy (без потерь, все чанки — один и тот же кодек/
    параметры). Если ХОТЯ БЫ один чанк не собрался — честный откат на
    concat всего ролика, как раньше (лучше без переходов, чем сорванная
    сборка)."""
    n = len(clips)
    bounds = _chunk_bounds(n, sections, chunk_size)
    if len(bounds) <= 1:
        return xfade_chain(clips, durs, sections, out, xfade_dur=xfade_dur, blocks=blocks)
    chunk_files, chunk_total = [], 0.0
    for ci, (a, b) in enumerate(bounds):
        cblocks = blocks[a:b] if blocks else None
        cout = os.path.join(temp_dir, f"_xchunk_{ci:03d}.mp4")
        ok, cdur = xfade_chain(clips[a:b], durs[a:b], sections[a:b], cout,
                                xfade_dur=xfade_dur, blocks=cblocks)
        if not ok:
            print(f"  чанк {ci} ({b - a} клипов) xfade не собрался — вся склейка откатывается на concat")
            return False, 0.0
        chunk_files.append(cout)
        chunk_total += cdur
    concat_list = os.path.join(temp_dir, "_xchunk_concat.txt")
    open(concat_list, "w", encoding="utf-8").write(
        "".join(f"file '{os.path.abspath(c)}'\n" for c in chunk_files))
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_list, "-c", "copy", out], capture_output=True, text=True)
    if r.returncode != 0:
        print("  склейка чанков xfade не удалась, откат на concat:", r.stderr[-300:])
        return False, 0.0
    return True, chunk_total


def estimate_xfade_budget(blocks):
    """Сколько xfade_chain суммарно "съест" при склейке — точная оценка ПО
    КАТЕГОРИЯМ перехода (см. 2.1: xfade_chain), не плоская (n-1)*XFADE_DUR
    (та была верна для старой единой логики, но разошлась с реальностью,
    когда переходы стали разной длины — 0.03-0.4с по функции блока).
    Не можем предсказать хэш-выбор hardcut/variety внутри "обычных" пар
    (зависит от итоговых путей файлов клипов — а те зависят от длительностей,
    которые мы как раз считаем, циклическая зависимость) — берём верхнюю
    границу этой категории. Небольшой перебор безопасен: финальный мукс
    обрезает merged видео точно по аудио (-t), лишнее просто уходит.
    Недобор — то, чего мы тут избегаем: он означает freeze-frame padding
    в конце ролика."""
    total = 0.0
    for i in range(1, len(blocks)):
        b_prev, b_cur = blocks[i - 1], blocks[i]
        if b_cur["section"] != b_prev["section"]:
            total += XFADE_DUR
        elif b_cur.get("is_subcut") or b_prev.get("stat") or b_cur["section"].startswith("HOOK"):
            total += XFADE_DUR_HARD
        else:
            total += max(XFADE_DUR_HARD, XFADE_DUR * 0.4)   # верхняя граница "обычной" пары
    return total


def pad_to_length(video, target, temp_dir):
    """Достраивает видео до нужной длины заморозкой последнего кадра — нужно
    после xfade_chain(), если реальный итог всё же короче target (несмотря
    на estimate_xfade_budget с запасом сверху — редкий остаточный случай,
    не обычный путь). Возвращает (путь, gap) — gap > 0 означает, что
    заморозка реально сработала, main() отражает это в QC."""
    cur = get_media_duration(video)
    gap = target - cur
    if gap <= 0.05:
        return video, 0.0
    lastframe = os.path.join(temp_dir, "_lastframe.jpg")
    padclip = os.path.join(temp_dir, "_pad.mp4")
    padded = os.path.join(temp_dir, "_padded.mp4")
    subprocess.run(["ffmpeg", "-y", "-sseof", "-0.3", "-i", video,
                    "-vframes", "1", lastframe], capture_output=True)
    r = subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", lastframe, "-t", f"{gap:.3f}",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                        "-pix_fmt", "yuv420p", "-r", str(FPS)] + COLOR_META_ARGS + [padclip],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return video, gap
    lst = os.path.join(temp_dir, "_pad_concat.txt")
    open(lst, "w", encoding="utf-8").write(
        f"file '{os.path.abspath(video)}'\nfile '{os.path.abspath(padclip)}'\n")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", lst, "-c", "copy", padded], capture_output=True, text=True)
    return (padded, gap) if r.returncode == 0 else (video, gap)


def get_media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def audio_qc(path):
    """Технический QC дорожки ДО сборки — тот же принцип, что qc_report() для
    картинок: не блокирует, только честно докладывает брак, который иначе
    заметили бы только на слух постфактум (клиппинг, слишком тихо/громко,
    длинные участки мёртвой тишины помимо самих [pause] — забытый обрыв
    записи). Бесплатно — штатные ffmpeg-фильтры astats/silencedetect,
    никакого нового API."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostats", "-i", path, "-af",
             "astats=reset=0,silencedetect=noise=-40dB:d=1.5", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
        err = r.stderr
    except Exception as e:
        print(f"  Audio QC: не удалось проверить ({e})")
        return
    warns = []
    m = re.search(r"Peak level dB:\s*(-?[\d.]+)", err)
    if m and float(m.group(1)) > -0.1:
        warns.append(f"пик {m.group(1)} dBFS — на грани клиппинга")
    m = re.search(r"RMS level dB:\s*(-?[\d.]+)", err)
    rms = float(m.group(1)) if m else None
    if rms is not None and rms < -35:
        warns.append(f"RMS {rms:.1f} dBFS — очень тихо")
    elif rms is not None and rms > -12:
        warns.append(f"RMS {rms:.1f} dBFS — очень громко/пережато")
    silences = [float(x) for x in re.findall(r"silence_duration:\s*([\d.]+)", err)]
    if silences:
        try:
            dur = get_media_duration(path)
            ratio = sum(silences) / dur
            if ratio > 0.15:
                warns.append(f"{ratio*100:.0f}% дорожки — тишина длиннее 1.5с (мёртвый воздух?)")
        except Exception:
            pass
    print(f"Audio QC: {'; '.join(warns)}" if warns else "Audio QC: клиппинга/аномальной громкости не найдено")


def ahash(photo_path, size=8):
    """Средний хэш (average hash) картинки — 64-битная строка, дешёвая
    замена imagehash-библиотеке (Pillow уже обязательная зависимость, лишний
    пакет не нужен). Похожие по контенту кадры дают близкий хэш даже если
    это разные файлы с разных источников."""
    img = PILImage.open(photo_path).convert("L").resize((size, size), PILImage.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p > avg else "0" for p in pixels)


def hamming(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def qc_report(media_log):
    """Не блокирует сборку — просто честно докладывает, если несколько
    кадров ролика визуально почти одинаковые (Pexels ID разные, а по факту
    похожий кроп/пересъёмка того же сюжета — то, что ловится глазом, но не
    ловится дедупом по ID). Порог 6 бит из 64 — эмпирически "заметно похоже".
    Возвращает список дублей — раньше это тонуло в середине лога, а ГОТОВО
    выглядело как чистый успех; теперь main() отражает находку в итоговой
    строке и коде возврата (не блокирует сборку — только честно не молчит:
    тот же принцип, что уже применён к пропущенным блокам)."""
    if len(media_log) < 2:
        return []
    hashes = []
    for i, path in media_log:
        try:
            hashes.append((i, ahash(path)))
        except Exception:
            pass
    dupes = []
    for a in range(len(hashes)):
        for bb in range(a + 1, len(hashes)):
            i1, h1 = hashes[a]
            i2, h2 = hashes[bb]
            d = hamming(h1, h2)
            if d <= 6:
                dupes.append((i1 + 1, i2 + 1, d))
    if dupes:
        print(f"QC: похожие кадры (проверь глазами) — {dupes}")
    else:
        print("QC: явных повторов по картинке не найдено")
    return dupes


def apply_section_boundary_shift(blocks, durs):
    """Лёгкая версия J/L-cut. Классический приём (звук следующей сцены
    начинается раньше её картинки) тут физически не из чего собрать — вся
    озвучка это один сплошной файл на весь ролик, а не отдельная дорожка
    на клип. Но можно сдвинуть саму ТОЧКУ ВИДЕО-РЕЗА относительно границы
    фразы на 250-400мс раньше или позже вместо ровно момента паузы —
    зритель считывает лёгкую рассинхронность видео/смыслового реза как
    признак ручного монтажа, а не нарезки строго по паузам. Сумма
    длительностей не меняется — время просто перетекает между соседними
    блоками ДО и ПОСЛЕ границы секции (не трогаем границы внутри секции —
    там резы и так короткие/частые, сдвигать нечего и незачем)."""
    d = list(durs)
    for i in range(1, len(blocks)):
        if blocks[i]["section"] == blocks[i - 1]["section"]:
            continue
        h = int(hashlib.md5(blocks[i]["text"][:40].encode()).hexdigest()[:8], 16)
        mag = 0.25 + (h % 150) / 1000.0   # 0.25-0.40с, детерминировано по тексту
        give = mag if (h & 1) else -mag
        floor_prev = hook_floor_for(blocks, i - 1)
        floor_cur = hook_floor_for(blocks, i)
        if d[i - 1] - give >= floor_prev and d[i] + give >= floor_cur:
            d[i - 1] -= give
            d[i] += give
    return d


def apply_human_jitter(blocks, durs, magnitude=0.3):
    """Механическая регулярность длительностей — узнаваемый штамп автослайдшоу:
    даже с real-alignment таймингом (см. load_alignment_weights) сетка
    остаётся слишком "правильной". Живой монтаж имеет микронеровности:
    иногда кадр обрывается на 2.3с раньше расчётного, иногда держится
    дольше. Джиттер — детерминированный (hashlib.md5 текста, не встроенный
    hash() — тот рандомизирован между запусками и сломал бы кэш), сумма
    длительностей сохраняется РОВНО (вычитаем среднее смещение из всех
    дельт вместо пропорционального масштабирования — иначе масштаб просто
    съедает большую часть эффекта на длинном ролике)."""
    deltas = []
    for b in blocks:
        h = int(hashlib.md5((b["section"] + b["text"][:40]).encode()).hexdigest()[:8], 16)
        frac = (h % 1000) / 1000.0
        deltas.append((frac - 0.5) * 2 * magnitude)
    mean_delta = sum(deltas) / len(deltas)
    deltas = [x - mean_delta for x in deltas]
    # Свой пол для хука (hook_floor_for — с учётом ещё более низкого пола на
    # первые резы) — иначе джиттер молча утаскивал бы уже нарочно короткие
    # хук-кадры обратно к общему MIN_CLIP, сводя на нет весь смысл отдельного,
    # более быстрого пола (см. block_durations).
    out = [max(hook_floor_for(blocks, i), d + delta)
           for i, (d, delta) in enumerate(zip(durs, deltas))]
    scale = sum(durs) / sum(out)
    return [x * scale for x in out]


SUBCUT_MIN_SOURCE_DUR = 8.0   # блок короче этого не режем вообще
SUBCUT_MIN_PART_DUR = 3.0     # ни один получившийся под-кадр не короче этого
SUBCUT_CONTRAST_WORDS = {"но", "однако", "зато", "хотя", "просто", "ведь",
                          "потому", "поэтому", "притом", "притом,"}
# Первые HOOK_OPENING_CUTS блоков хука режем куда охотнее обычного — топовые
# ролики держат первые 2-3 реза короче остального хука. Просто пол
# (hook_floor_for/HOOK_OPENING_MIN_CLIP) тут бессилен сам по себе: он
# клэмпит СНИЗУ уже готовую длительность блока, а не даёт больше КУСКОВ на
# ту же фразу — типичная 3-4-секундная фраза открытия просто не проходит
# порог обычного SUBCUT_MIN_SOURCE_DUR=8.0 и остаётся одним клипом
# (проверено вживую на реальном эпизоде: один только floor не изменил ни
# одной цифры длительности — узкое место в гранулярности блоков, не в поле).
SUBCUT_HOOK_MIN_SOURCE_DUR = 2.0
SUBCUT_HOOK_MIN_PART_DUR = 0.9
SUBCUT_HOOK_MIN_WORDS = 4


def split_long_blocks(blocks, real_weights):
    """Один блок = один клип 4-20 сек — механическая сетка "одна мысль = одна
    картинка". Профессиональный монтаж на одну длинную фразу даёт 2-3
    визуальных решения. Режем блок ТОЛЬКО если он длиннее SUBCUT_MIN_SOURCE_DUR
    (по реальному alignment-весу, при его отсутствии — грубая оценка по
    словам), по смысловым триггерам: контраст-союзы ("но"/"однако"/...) или
    первое число за пределами первой четверти блока (новый аргумент). Без
    триггеров, но всё равно длинный — режем ровно посередине по словам.
    Реальный вес блока делится между кусками ПРОПОРЦИОНАЛЬНО их доле слов —
    сумма сохраняется, ничего не выдумываем сверху. Запрос к стоку для
    под-кадров остаётся тем же самым блоком (resolve_queries считается уже
    ПОСЛЕ split, на новом расширенном списке — если у куска нет своего
    ключевого слова, он унаследует запрос соседнего куска той же секции,
    та же логика, что уже работает для обычных блоков), картинка другая —
    засчёт дедупа по used_photo_ids на большем пуле Pexels (per_page=40)."""
    new_blocks, new_weights = [], []
    hook_seen = 0
    for b, w in zip(blocks, real_weights or [None] * len(blocks)):
        is_hook_opening = False
        if b["section"].startswith("HOOK"):
            hook_seen += 1
            is_hook_opening = hook_seen <= HOOK_OPENING_CUTS
        min_source = SUBCUT_HOOK_MIN_SOURCE_DUR if is_hook_opening else SUBCUT_MIN_SOURCE_DUR
        min_part = SUBCUT_HOOK_MIN_PART_DUR if is_hook_opening else SUBCUT_MIN_PART_DUR
        min_words = SUBCUT_HOOK_MIN_WORDS if is_hook_opening else 8
        est = w if w is not None else b["words"] / 2.08
        if est < min_source:
            new_blocks.append(b)
            new_weights.append(w)
            continue
        words = b["text"].split()
        if len(words) < min_words:
            new_blocks.append(b)
            new_weights.append(w)
            continue
        split_at = []
        for i, word in enumerate(words):
            wl = word.strip(",.!?—–:;»«").lower()
            if wl in SUBCUT_CONTRAST_WORDS and 2 <= i <= len(words) - 3:
                split_at.append(i)
        for i, word in enumerate(words):
            if re.search(r'\d', word) and i > len(words) // 4:
                split_at.append(i)
                break
        if not split_at:
            # Без смыслового триггера резать РОВНО посередине по счёту слов
            # означало иногда рвать словосочетание внутри одной фразы ("в
            # тысячу" | "раз" — тайминг не привязан к речи). Ищем ближайшую к
            # середине пунктуационную границу (конец клаузы: , . ! ? — ; :) —
            # рез попадает на естественную паузу произношения. Не нашли ни
            # одной в разумном радиусе — старое поведение (ровно посередине).
            mid = len(words) // 2
            found = None
            for radius in range(0, len(words) // 2 + 1):
                for cand in (mid - radius, mid + radius):
                    if 2 <= cand <= len(words) - 3 and words[cand - 1].rstrip()[-1:] in ",.!?—;:":
                        found = cand
                        break
                if found is not None:
                    break
            split_at = [found if found is not None else mid]
        bounds = sorted(set([0] + split_at + [len(words)]))
        chunks = [(a, c) for a, c in zip(bounds, bounds[1:]) if c > a]
        # Склеиваем куски короче SUBCUT_MIN_PART_DUR (по доле слов от est)
        # с предыдущим — не даём под-кадру уйти ниже комфортной длины.
        total_w = len(words)
        merged = []
        for a, c in chunks:
            frac = (c - a) / total_w
            if merged and est * frac < min_part:
                pa, _ = merged[-1]
                merged[-1] = (pa, c)
            else:
                merged.append((a, c))
        if len(merged) < 2:
            new_blocks.append(b)
            new_weights.append(w)
            continue
        # Плашка раньше ВСЕГДА доставалась первому куску (k==0) независимо от
        # того, где в исходном блоке реально стоял [stat:...] — если рез
        # приходился ДО фразы с числом, цифра на экране показывалась на
        # куске, где она ещё не произнесена, а реальный кусок с числом
        # оставался без плашки вовсе. stat_word_pos (см. parse_blocks) —
        # позиция тега в словах исходного блока; находим, в какой ИМЕННО
        # кусок она попадает, и пересчитываем позицию относительно НАЧАЛА
        # этого куска (nb-локально), чтобы дальше по конвейеру плашку можно
        # было синхронизировать с моментом озвучки, а не с началом клипа.
        stat_pos = b.get("stat_word_pos")
        for k, (a, c) in enumerate(merged):
            nb = dict(b)
            nb["text"] = " ".join(words[a:c])
            nb["words"] = c - a
            has_stat = b["stat"] is not None and stat_pos is not None and (
                (a < stat_pos <= c) or (stat_pos <= 0 and k == 0))
            nb["stat"] = b["stat"] if has_stat else None
            nb["stat_word_pos"] = max(0, stat_pos - a) if has_stat else None
            nb["pause_after"] = b["pause_after"] if k == len(merged) - 1 else 0.0
            nb["is_subcut"] = k > 0   # для choose_motion_mode() (1.4) — деталь, не главный кадр фразы
            new_blocks.append(nb)
            new_weights.append((w * (c - a) / total_w) if w is not None else None)
    return new_blocks, new_weights


def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"Аудио не найдено: {AUDIO_FILE}")
        return 1
    total = get_audio_duration()
    print(f"Аудио: {total:.1f}с ({total/60:.1f} мин)")
    audio_qc(AUDIO_FILE)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    blocks = parse_blocks(SCRIPT_FILE) if os.path.exists(SCRIPT_FILE) else []
    if not blocks:
        print("Сценарий не найден/пуст")
        return 1
    # Реальный тайминг считаем ДО sub-cuts — alignment.csv записан 1:1 на
    # исходные блоки (по [pause]/[short pause]), split_long_blocks() потом
    # честно делит вес пропорционально словам между получившимися кусками.
    real_weights = load_alignment_weights(blocks)
    if real_weights:
        known = sum(1 for w in real_weights if w is not None)
        print(f"Реальный тайминг (alignment.csv): {known}/{len(blocks)} блоков")
    n_before = len(blocks)
    blocks, real_weights = split_long_blocks(blocks, real_weights)
    if len(blocks) != n_before:
        print(f"Sub-cuts: {n_before} -> {len(blocks)} блоков")
    # Кроссфейд между КАЖДОЙ парой кадров суммарно "съедает" какую-то часть
    # длительности — закладываем это в целевую длительность заранее, чтобы
    # после склейки общая длина видео снова совпала с аудио (без этого хвост
    # ролика проигрывался бы без картинки под конец аудиодорожки). Считаем
    # ПОСЛЕ split — sub-cuts добавляют новые склейки, бюджет должен их
    # учитывать. estimate_xfade_budget() — точная оценка по категориям
    # перехода (2.4), не плоская (n-1)*XFADE_DUR.
    xfade_budget = estimate_xfade_budget(blocks)
    target = total + xfade_budget
    durs = block_durations(blocks, target, real_weights=real_weights)
    # Второй проход: ритм по громкости поверх базовой оценки (не вместо неё) —
    # громкие места режутся чаще, тихие держатся дольше. Опционально (нужен numpy).
    # Стартовые точки для сэмплинга энергии считаем от РЕАЛЬНОЙ длины аудио
    # (total), не от раздутой под кроссфейды target — иначе поздние блоки на
    # длинном ролике со множеством склеек сэмплили бы энергию не в том месте.
    curve = audio_energy_curve(AUDIO_FILE)
    energy_lvls = [1.0] * len(blocks)
    if curve:
        baseline = block_durations(blocks, total, real_weights=real_weights)
        starts, acc = [], 0.0
        for d in baseline:
            starts.append(acc)
            acc += d
        mults = energy_pace_multipliers(curve, starts, baseline)
        durs = block_durations(blocks, target, energy_mults=mults, real_weights=real_weights)
        # Та же кривая громкости, что уже двигает тайминг — используем и для
        # лёгкой визуальной пульсации грейда (не только когда резать, но и
        # "с каким нажимом" показать кадр на эмоциональном пике).
        energy_lvls = energy_levels(curve, starts, baseline)
        print("Ритм по громкости: включён")
    durs = apply_section_boundary_shift(blocks, durs)
    durs = apply_human_jitter(blocks, durs)
    print(f"Средний кадр: {sum(durs)/len(durs):.1f}с")

    # Субтитры/главы — бесплатный побочный продукт уже посчитанного тайминга
    # (см. write_subtitles/write_chapters). Нужна РЕАЛЬНАЯ развёртка по аудио
    # (block_durations по total), не по xfade-раздутой durs/target — та же
    # модель реального времени, что уже используется для energy-курса ниже.
    sub_baseline = block_durations(blocks, total, real_weights=real_weights)
    sub_starts, sub_acc = [], 0.0
    for d in sub_baseline:
        sub_starts.append(sub_acc)
        sub_acc += d
    write_subtitles(VIDEO_FOLDER, blocks, sub_starts, sub_baseline)
    write_chapters(VIDEO_FOLDER, blocks, sub_starts)

    # A8: резы хука слегка "прилипают" к пикам голосовой энергии (см.
    # snap_hook_cuts_to_energy) — реальные (не xfade-раздутые) старты sub_starts
    # уже посчитаны как раз выше, переиспользуем ту же временную шкалу.
    durs = snap_hook_cuts_to_energy(blocks, durs, sub_starts, AUDIO_FILE)

    clips, clip_durs, clip_sections, clip_blocks = [], [], [], []
    missing = []   # индексы блоков, для которых не нашлось ни фото, ни видео
    media_log = []   # (индекс, путь_к_фото) — для QC-проверки на похожие кадры в конце
    zoom_hist, pan_hist = [], []
    use_local = os.path.isdir(MEDIA_FOLDER) and bool(local_photo(0))
    use_pexels = bool(PEXELS_API_KEY)
    queries = resolve_queries(blocks)
    write_shot_manifest(VIDEO_FOLDER, blocks, durs, queries)
    used_photo_ids = set()   # общий на весь ролик — не даём одной фотке всплыть дважды
    used_video_ids = set()   # то же самое, отдельно для видео (разные ID-пространства)
    used_photo_hashes = []   # aHash уже отобранных фото — ловит визуальные дубли под РАЗНЫМИ ID (см. pexels_photo)
    recent_shot_sizes = []   # скользящее окно масштаба плана (wide/medium/close/detail) — см. estimate_shot_size
    recent_media_types = []   # скользящее окно фото/видео — content-aware чередование, см. main() ниже
    stat_count = 0   # 2.2: номер плашки по счёту в ролике -> вариант оформления (chередуются по кругу)
    luma_ema = None   # для сглаживания скачков экспозиции между соседними склейками (см. measure_luma())
    typewriter_click_times = []   # D4: абсолютные секунды щелчков клавиатуры на весь ролик
    hook_words = load_hook_word_timings()   # D2: пусто, если alignment.csv недоступен — тихий откат
    for i, (b, d) in enumerate(zip(blocks, durs)):
        # Титр темы — только на ПЕРВОМ кадре новой секции (BLOCK N: Название).
        is_section_start = i == 0 or blocks[i]["section"] != blocks[i - 1]["section"]
        title = section_title(b["section"]) if is_section_start else None
        stat = b.get("stat")
        stat_variant = stat_count
        # D2: кинетические подписи — только ХУК, только если alignment.csv
        # реально дал слова на эту секцию. sub_starts/sub_baseline (реальная,
        # не xfade-раздутая шкала) уже посчитаны выше по тексту — та же
        # шкала, что у 00.csv (хук всегда с t=0 эпизода).
        captions = (hook_captions_for_block(hook_words, sub_starts[i], sub_baseline[i])
                    if (hook_words and b["section"].startswith("HOOK")) else None)
        if stat:
            stat_count += 1
        # Момент внутри клипа, когда число уже произнесено (см. stat_word_pos
        # в parse_blocks/split_long_blocks) — доля слов ДО тега [stat:...] от
        # общего числа слов В ЭТОМ куске, умноженная на его финальную
        # длительность. Та же word-count-пропорция, что уже используется как
        # честный фоллбэк тайминга по всему пайплайну (не выдумываем более
        # точную модель специально под этот случай).
        stat_word_pos = b.get("stat_word_pos")
        stat_delay = (stat_word_pos / max(1, b["words"]) * d) if (stat and stat_word_pos) else 0.0
        # D4: печатная машинка — только на 5-м варианте плашки (см. add_overlays).
        # Таймкоды щелчков считаем на РЕАЛЬНОЙ (не xfade-раздутой) шкале —
        # sub_starts/sub_baseline уже посчитаны выше по тексту, та же шкала,
        # что уже используется для субтитров (см. комментарий у sub_baseline).
        if stat and stat_variant % 5 == 4:
            real_dur = sub_baseline[i]
            real_delay = max(0.0, min(stat_delay, real_dur - 1.2))
            click_text = stat.upper() if FONT_IS_DISPLAY else stat
            _, click_cd = typewriter_reveal_timing(len(click_text), real_delay, real_dur)
            typewriter_click_times.extend(
                sub_starts[i] + real_delay + k * click_cd for k in range(len(click_text)))
        # Хэш параметров рендера в имени — иначе правка script.txt (текст,
        # тайминг, плашка) без ручной чистки temp_smart/ молча оставляла
        # старый клип под новые данные (тот же класс бага, что уже правили
        # для pexels_cache). Запрос (queries[i]) обязателен в хэше отдельно —
        # без него правка channel_themes.json/media_plan/themes.json при
        # совпавшей длительности тихо оставляла старый (уже нерелевантный)
        # клип под новым запросом — поймано QC-сверкой вживую: клип с новым
        # запросом "medieval sword close up" продолжал показывать старую
        # киноплёнку, потому что длительность/заголовок/плашка/секция не
        # изменились, а запрос в хэш не входил.
        # captions в хэше — D2, чтобы правка/пересчёт alignment.csv не
        # оставляла старые (без подписей или с другим текстом/таймингом)
        # закэшированные клипы висеть под тем же именем.
        params_hash = hashlib.md5(
            f"{d:.3f}|{title}|{stat}|{stat_variant}|{b['section']}|{queries[i]}|{stat_delay:.3f}|"
            f"{captions}".encode()).hexdigest()[:8]
        out = os.path.join(TEMP_FOLDER, f"clip_{i:04d}_{params_hash}.mp4")
        if os.path.exists(out):
            clips.append(out)
            clip_durs.append(d)
            clip_sections.append(b["section"])
            clip_blocks.append(b)
            continue
        photo = local_photo(i) if use_local else None
        video = None
        if not photo and use_pexels:
            # Content-aware чередование вместо механического i%2 (ЧАСТЬ 14
            # раньше просто нечётные->фото/чётные->видео) — зритель
            # подсознательно считывает такую периодичность. Видео заказываем,
            # когда текст блока реально описывает действие/движение
            # (ACTION_WORDS — "штурм"/"погоня"/"удар" и т.п., там сток-видео
            # осмысленно показывает движение), плюс детерминированный хэш
            # текста как база для обычного визуального разнообразия (иначе
            # почти весь ролик без экшн-лексики ушёл бы в чистое фото).
            # Никогда видео под цифру-плашку — движущийся фон мешает читать
            # число (см. static_hold в choose_motion_mode — тот же принцип
            # для фото, но video_render своей motion_mode-ветки не имеет).
            # Не встык 2 видео подряд (видео — акцент, не фон) и не больше
            # 3 фото подряд (иначе монотонно) — то же разнообразие, что уже
            # держат zoom_hist/pan_hist через pick_no_repeat, но асимметрично
            # (видео реже, чем фото, по самой природе приёма).
            h_text = int(hashlib.md5(b["text"][:40].encode()).hexdigest()[:8], 16)
            want_video = (has_action_word(b["text"]) or h_text % 2 == 1) and not stat
            if want_video and recent_media_types[-1:] == ["video"]:
                want_video = False
            if (not want_video and not stat and len(recent_media_types) >= 3
                    and all(t == "photo" for t in recent_media_types[-3:])):
                want_video = True
            prefer_video = want_video and d >= MIN_CLIP + 1.0
            if prefer_video:
                video = pexels_video(queries[i], i, used_ids=used_video_ids)
                if not video:
                    photo = pexels_photo(queries[i], i, used_ids=used_photo_ids, used_hashes=used_photo_hashes,
                                      recent_sizes=recent_shot_sizes, target_luma=luma_ema)
            else:
                photo = pexels_photo(queries[i], i, used_ids=used_photo_ids, used_hashes=used_photo_hashes,
                                      recent_sizes=recent_shot_sizes, target_luma=luma_ema)
                if not photo and d >= MIN_CLIP + 1.0:
                    video = pexels_video(queries[i], i, used_ids=used_video_ids)
            # Раньше Pexels отключался навсегда после ЛЮБОГО промаха, включая
            # обычную пустую выдачу по одному неудачному запросу. Гасим источник
            # только если API реально отвалился.
            if not photo and not video and PEXELS_BROKEN:
                use_pexels = False
        if not photo and not video:
            photo = local_photo(i)
        if not photo and not video:
            print(f"  [{i+1}] нет медиа")
            missing.append(i + 1)
            continue
        recent_media_types.append("video" if video else "photo")
        del recent_media_types[:-6]
        # Непрерывность экспозиции: измеряем яркость ИМЕННО этого кадра,
        # сравниваем со скользящим средним последних клипов — если скачок
        # большой (сток из разных источников редко совпадает по свету),
        # подмешиваем небольшую компенсацию в brightness, а не выравниваем
        # целиком (иначе пропала бы естественная вариативность вообще).
        luma = measure_luma(photo, is_video=False) if photo else measure_luma(video, is_video=True)
        if luma is not None:
            brightness_bias = 0.0 if luma_ema is None else max(-0.035, min(0.035, (luma_ema - luma) * 0.35))
            luma_ema = luma if luma_ema is None else luma_ema * 0.6 + luma * 0.4
        else:
            brightness_bias = 0.0
        # Адаптивная ТЕХНИЧЕСКАЯ нормализация экспозиции ЭТОГО кадра (auto-
        # levels по перцентилям, см. measure_levels()/auto_levels_params()) —
        # решает другую задачу, чем brightness_bias выше (тот сглаживает
        # скачок МЕЖДУ соседними кадрами, это подтягивает разброс ВНУТРИ
        # одного кадра к общей базе), применяется в film_look() ПЕРЕД
        # неизменным творческим грейдом, не вместо него.
        levels = measure_levels(photo, is_video=False) if photo else measure_levels(video, is_video=True)
        energy_bias = max(-0.5, min(0.5, energy_lvls[i] - 1.0))
        if video:
            ok = video_render(video, out, d, title=title, stat=stat, section=b["section"],
                               stat_variant=stat_variant, brightness_bias=brightness_bias,
                               energy_bias=energy_bias, stat_delay=stat_delay, levels=levels,
                               handheld=has_action_word(b["text"]), captions=captions)
        else:
            # anti-repetition: хэш сам по себе не мешает 3 зумам подряд случайно
            # совпасть — держим окно последних решений и форсируем смену при повторе.
            _, zi_cand, pd_cand = kb_hash_choices(photo)
            zoom_in = pick_no_repeat(zoom_hist, zi_cand, [True, False], max_repeat=2)
            pan_dir = pick_no_repeat(pan_hist, pd_cand, PAN_DIRECTIONS, max_repeat=2)
            # Параллакс — только на самые заметные точки ролика (хук целиком +
            # первый кадр каждого раздела), не на все фото: покадровый рендер
            # с depth-моделью в разы дороже по времени zoompan-версии, на
            # 40+ кадрах это лишние десятки минут ради эффекта, который
            # большую часть ролика зритель всё равно не разглядывает так
            # пристально, как хук и открывашки разделов.
            is_highlight = b["section"].startswith("HOOK") or is_section_start
            ok = False
            if PARALLAX_ENABLED and is_highlight:
                # Параллакс (2.5D depth remap) — своя, уже отдельно проверенная
                # зум/пан-математика, motion_mode на него не распространяем:
                # это отдельный, более дорогой визуальный приём для самых
                # заметных точек ролика, не нуждается в дополнительной
                # категоризации поверх собственной.
                ok = parallax_kenburns(photo, out, d, title=title, zoom_in=zoom_in,
                                        pan_dir=pan_dir, stat=stat, section=b["section"],
                                        stat_variant=stat_variant, brightness_bias=brightness_bias,
                                        energy_bias=energy_bias, stat_delay=stat_delay, levels=levels,
                                        captions=captions)
            if not ok:
                photo_hash, _, _ = kb_hash_choices(photo)
                cur_shot_size = recent_shot_sizes[-1] if recent_shot_sizes else None
                motion_mode = choose_motion_mode(b, is_section_start, photo_hash, shot_size=cur_shot_size)
                ok = kenburns(photo, out, d, title=title, zoom_in=zoom_in, pan_dir=pan_dir,
                              stat=stat, section=b["section"], motion_mode=motion_mode,
                              brightness_bias=brightness_bias, energy_bias=energy_bias,
                              stat_variant=stat_variant, stat_delay=stat_delay, levels=levels,
                              captions=captions)
        if ok:
            clips.append(out)
            clip_durs.append(d)
            clip_sections.append(b["section"])
            clip_blocks.append(b)
            if not video and photo:
                media_log.append((i, photo))
            if i % 20 == 0 or i < 3:
                print(f"  [{i+1}/{len(blocks)}] {d:.1f}с {b['words']} слов ({'видео' if video else 'фото'})")
        else:
            missing.append(i + 1)
        if use_pexels and not use_local and i % 10 == 9:
            time.sleep(0.4)

    if not clips:
        print("Нет клипов")
        return 1
    merged = os.path.join(TEMP_FOLDER, "merged.mp4")
    ok, xfade_total = xfade_chain_chunked(clips, clip_durs, clip_sections, merged, TEMP_FOLDER,
                                           blocks=clip_blocks)
    if not ok:
        concat = os.path.join(TEMP_FOLDER, "concat.txt")
        # Пути ТОЛЬКО абсолютные: concat-демуксер ffmpeg резолвит относительные
        # пути от папки самого concat.txt, а не от cwd — иначе сборка падает.
        open(concat, "w", encoding="utf-8").write(
            "".join(f"file '{os.path.abspath(c)}'\n" for c in clips))
        r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", concat, "-c", "copy", merged], capture_output=True, text=True)
        if r.returncode != 0:
            print("Склейка:", r.stderr[-300:])
            return 1
    merged, pad_gap = pad_to_length(merged, total, TEMP_FOLDER)

    # -shortest САМ ПО СЕБЕ недостаточен с -c:v copy: копирование пакетов
    # режет только по границам GOP исходного клипа, а не по факту конца
    # аудио — на реальном 17-минутном ролике это давало +6с видео без
    # звука сверху (проверено вживую). Явный -t с точной длительностью
    # аудио режет ровно там, где надо, независимо от размера GOP.
    # A1: атмосферная подложка + сайдчейн-дакинг под голосом (build_music_mix,
    # безопасный откат на чистый голос, если ассет отсутствует/выключен).
    # A7: loudnorm теперь считается на ГОТОВОМ миксе голос+музыка, а не на
    # одном голосе — иначе музыка добавляла бы громкость ПОСЛЕ калибровки
    # -16 LUFS и сводила бы её к нулю (та же ошибка, что "нормализовать
    # трек, потом наложить ещё один трек поверх без пересчёта").
    # A2: границы HOOK/FINAL по реальным длительностям отрендеренных клипов
    # (не по словам сценария — клипы уже учитывают все реальные подрезки/
    # sub-cuts) — секции идут по порядку HOOK -> BLOCK* -> FINAL, так что
    # достаточно суммы длительностей внутри каждой без отдельного поиска границ.
    hook_end = sum(d for sec, d in zip(clip_sections, clip_durs) if sec.startswith("HOOK"))
    final_start = sum(d for sec, d in zip(clip_sections, clip_durs) if not sec.startswith("FINAL"))
    # A6: обработка голоса (highpass/EQ/деэссер/компрессия) ДО подмешивания
    # музыки — тот же порядок, что в реальном пост-продакшене: сначала
    # приводишь дорожку диктора в порядок, потом кладёшь её в микс.
    voice_processed = os.path.join(TEMP_FOLDER, "voice_processed.wav")
    voice_processed = process_voice(AUDIO_FILE, voice_processed)
    premix = os.path.join(TEMP_FOLDER, "premix.wav")
    premix = build_music_mix(voice_processed, total, premix, hook_end=hook_end, final_start=final_start)
    # D4: щелчки клавиатуры поверх готового микса — ПОСЛЕ музыки/дакинга
    # (иначе сайдчейн реагировал бы и на сами щелчки), ДО финального loudnorm
    # (клики тоже участвуют в мастеринге громкости целиком, не бесплатный
    # довесок сверху уже откалиброванного микса).
    if typewriter_click_times:
        premix_clicks = os.path.join(TEMP_FOLDER, "premix_clicks.wav")
        premix = add_typewriter_clicks(premix, typewriter_click_times, total, premix_clicks)
    # loudnorm — стандартная целевая громкость YouTube (-16 LUFS intergated,
    # -1.5dB true peak потолок, LRA 11) вместо "как есть от TTS". Разные
    # заказы озвучки/движки иначе дают заметно разную громкость от эпизода
    # к эпизоду — не проф. уровень, если зритель тянется к громкости между
    # роликами одного канала. Двухпроходный (measure -> linear=true) вместо
    # однопроходного динамического — тот подстраивает усиление на лету и
    # даёт неровную громкость внутри самого ролика. afade — раньше ролик
    # начинался и обрывался всухую (щелчок на монтажный лад, не финал).
    loud_stats = measure_loudnorm_stats(premix)
    fade_out_st = max(0.0, total - 2.0)
    if loud_stats:
        af = (f"loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:"
              f"measured_I={loud_stats['input_i']}:measured_TP={loud_stats['input_tp']}:"
              f"measured_LRA={loud_stats['input_lra']}:measured_thresh={loud_stats['input_thresh']}:"
              f"offset={loud_stats['target_offset']},"
              f"afade=t=in:st=0:d=0.4,afade=t=out:st={fade_out_st:.3f}:d=2")
    else:
        af = (f"loudnorm=I=-16:TP=-1.5:LRA=11,"
              f"afade=t=in:st=0:d=0.4,afade=t=out:st={fade_out_st:.3f}:d=2")
    r = subprocess.run(["ffmpeg", "-y", "-i", merged, "-i", premix,
                        "-t", f"{total:.3f}",
                        "-af", af,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart",
                        "-shortest", OUTPUT_FILE], capture_output=True, text=True)
    if r.returncode != 0:
        print("Аудио:", r.stderr[-300:])
        return 1
    # Честная цифра вместо "надеемся" — измеряем итоговую громкость ГОТОВОГО
    # файла (не входного audio.mp3): если разошлось с целью -16 LUFS больше,
    # чем на децибел, это видно в отчёте, а не тонет молча.
    final_lufs = None
    try:
        fr = subprocess.run(["ffmpeg", "-i", OUTPUT_FILE, "-af",
                              "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                              "-f", "null", "-"], capture_output=True, text=True, timeout=60)
        fs, fe = fr.stderr.rindex("{"), fr.stderr.rindex("}") + 1
        final_lufs = float(json.loads(fr.stderr[fs:fe])["input_i"])
    except Exception:
        pass
    mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    # Пропущенные кадры и раньше не останавливали сборку (стратегия
    # "лучше меньше клипов, чем сорванный рендер") — но раньше это тонуло
    # в середине лога, а ГОТОВО выглядело как чистый успех. Теперь пропуски
    # видны в итоговой строке, и код возврата честно отражает, что не всё
    # доехало (не 0, если что-то пропущено).
    dupes = qc_report(media_log)
    status = f" | ПРОПУЩЕНО {len(missing)} блоков: {missing}" if missing else ""
    if final_lufs is not None:
        status += f" | Громкость: {final_lufs:.1f} LUFS" + (
            "" if abs(final_lufs - (-16.0)) <= 1.0 else " (!!! разошлось с целью -16)")
    # Дубли по фото — та же логика: не блокируем сборку (видео уже готово
    # и в целом смотрибельно), но не даём находке потеряться в середине
    # лога и не даём коду возврата соврать, что всё чисто.
    status += f" | QC: {len(dupes)} похожих пар кадров — проверь глазами" if dupes else ""
    # Freeze-padding (2.4) — estimate_xfade_budget() должен был свести его
    # к нулю почти всегда; если он всё же сработал заметно (>1с), это
    # видно глазом на экране (замёрзший кадр без причины) — та же
    # честная видимость, не молчаливое "ГОТОВО".
    status += f" | ЗАМОРОЗКА {pad_gap:.1f}с в хвосте — расчёт длительностей разошёлся" if pad_gap > 1.0 else ""
    print(f"\nГОТОВО: {OUTPUT_FILE} ({mb:.0f} MB, {total/60:.1f} мин, {len(clips)} кадров){status}")
    return 1 if (missing or dupes or pad_gap > 1.0) else 0


if __name__ == "__main__":
    sys.exit(main())
