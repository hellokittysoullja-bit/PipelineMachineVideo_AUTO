#!/usr/bin/env python3
"""Reference-Guided Look Management — консервативная, объяснимая альтернатива
эвристикам _scene_bias/warm_mult в film_look() (scripts/pipeline_smart.py).
Разбор жалобы "меч стал холоднее" вскрыл, что интуитивная модуляция
творческого грейда ("холодный источник -> усилить пуш") может РАЗДВИГАТЬ
клипы разной цветовой температуры вместо заявленной цели "один визуальный
язык на канал" (CLAUDE.md ЧАСТЬ 15) — см. коммит с фиксом _warm_mult и его
regression-тест. Эта система — не патч той эвристики, а отдельный,
измеримый механизм: явная библиотека одобренных эталонов канала (lookbook),
подбор ближайшего эталона по домену/палитре/экспозиции/температуре и
консервативная, ограниченная по силе коррекция К ЭТОМУ ЭТАЛОНУ — вместо
"чуть больше/чуть меньше" в интуитивно выбранном направлении.

ВАЖНО, честно: lookbook начинался ПУСТЫМ (assets/lookbook/lookbook.json) —
сегодня в нём 13 эталонов, отобранных пользователем вручную с контактного
листа реальных кадров канала (см. коммит) через scripts/lookbook_add.py, не
выдуманных "на глаз" (та же ошибка, которую критиковали в _scene_bias, тут
повторять её нельзя). При пустом lookbook (или LOOK_MANAGEMENT_MODE=off,
дефолт) look_correction_filter() возвращает None для ЛЮБОГО кадра — система
физически не может повлиять на рендер (см.
test_look_correction_filter_noop_on_empty_lookbook в tests/test_look_reference.py,
теперь на синтетическом пустом lookbook, не на реальном файле).
LOOK_MANAGEMENT_MODE остаётся off/shadow по умолчанию даже с непустым
lookbook — assist требует ещё и явного CHANNEL_ID (не "default", см.
load_lookbook() ниже) и отдельного одобрения пользователя.

LOOK_MANAGEMENT_MODE: `off` (дефолт) — лукбук даже не грузится. `shadow` —
ВЕСЬ конвейер отрабатывает по-настоящему (домен/эталон/коррекция/QC) и
пишется в отчёт с decision="shadow_would_apply", но filter_str ВСЕГДА None
— рендер не тронут (тот же принцип, что F0-shadow в Speech Director Stage
B, CLAUDE.md Шаг 6 вариант Б). `assist` — коррекция реально применяется.

ИЗВЕСТНОЕ ОГРАНИЧЕНИЕ shadow-режима: кэш-хит клипы (уже отрендерены в
прошлом прогоне, temp_smart/ содержит готовый файл) НИКОГДА не доходят до
look_correction_filter() — main() уходит на ранний `continue` до
measure_levels(), и у кэш-хита физически нет сохранённого пути к исходному
фото для повторного анализа. На уже отрендеренном эпизоде shadow-прогон
НЕ является честной симуляцией assist — он покажет только то, что реально
проанализировано (см. "cache_hits_skipped_analysis" в media_plan/
look_manifest.json и консольное предупреждение). Для честного превью —
очистить temp_smart/ этого эпизода перед shadow-прогоном.

Архитектура (что переиспользуется из pipeline_smart.py, что новое):
  - measure_levels()/auto_wb_params()-стиль клэмпа силы и gain-диапазона —
    переиспользован ПРИНЦИП (частичная коррекция, gain clamp [0.5,1.8] —
    ffmpeg тихо роняет весь фильтр за пределами colorchannelmixer's [-2,2],
    тот же пойманный вживую баг), не код напрямую (эта система работает в
    Lab, не в RGB gray-world).
  - CLIP (get_clip_model(), CLIP_ENABLED/CLIP_BROKEN) — переиспользован
    НАПРЯМУЮ для классификации домена кадра (classify_domain()), тот же
    margin-gate принцип, что RISKY_QUERY_MARGIN в is_relevant_candidate().
  - detect_face_anchor() — переиспользован напрямую для защиты кожи/лиц:
    ПОЛНЫЙ пропуск коррекции при обнаруженном лице (не по-пиксельная
    маска — маска на движущемся Ken Burns зуме без реальных лиц в кадре для
    проверки была бы недоказанным риском, честнее и безопаснее — полный
    skip).
  - _scene_bias() — переиспользован напрямую для сигнала "температура"
    (warm_bias) кадра при подборе эталона.
  - EMA-сглаживание — тот же ПРИНЦИП, что luma_ema/brightness_bias в main()
    pipeline_smart.py (частичный шаг к цели, клэмп шага, не мгновенный
    скачок) — код свой (сглаживается Lab-дельта, не яркость), тот же дух.
  - QC — АНАЛИТИЧЕСКИ на уже посчитанных levels/wb (clipping/оверсатурация
    ДО рендера, тот же принцип, что клэмп в auto_wb_params уже применяет ДО
    того, как ffmpeg тихо уронит фильтр), НЕ рендер 2-3 вариантов на
    сравнение — на порядок дешевле по времени рендера, тот же результат
    предсказуемости для глобальной (не по-пиксельной) коррекции.

Честно НЕ реализовано (упрощения против исходного технического задания,
см. обсуждение при проектировании):
  - Шум/резкость/DeltaE кожи как отдельные QC-метрики — глобальный gain-
    сдвиг не может внести шум/расфокус по построению (не пространственный
    фильтр), а защита кожи — через full-skip при лице (см. выше), не через
    DeltaE.
  - Коррекция для video-клипов — только фото в этой версии (вызывающий код
    в pipeline_smart.py просто не зовёт look_correction_filter() для видео).
  - Semantic Visual Director (контекстный выбор КАКОЙ кадр использовать по
    смыслу фразы/риторической роли/истории соседних клипов) — отдельная,
    более крупная задача, эта система выбирает только ЦВЕТОВОЙ эталон для
    уже выбранного кадра, не сам кадр.

Не самостоятельный CLI-скрипт (кроме lookbook_add.py) — вызывается из
scripts/pipeline_smart.py."""
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

VIDEO_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
_saved_argv = sys.argv
sys.argv = ["pipeline_smart.py", VIDEO_DIR]
import pipeline_smart  # noqa: E402
sys.argv = _saved_argv

from PIL import Image  # noqa: E402

LOOKBOOK_PATH = os.path.join(REPO_ROOT, "assets", "lookbook", "lookbook.json")

# off — как раньше, ничего не считаем, лукбук даже не грузим (см.
# look_correction_filter). shadow — ВЕСЬ конвейер (домен/эталон/коррекция/QC)
# отрабатывает по-настоящему и пишется в отчёт, но filter_str ВСЕГДА None —
# рендер не тронут. Тот же принцип, что уже применяет Stage B Speech
# Director к F0-перепаду ("измеряется и пишется в отчёт, НЕ влияет на
# решение", CLAUDE.md Шаг 6 вариант Б) — сначала честно посмотреть, что
# система решила бы, прежде чем доверить ей реальный рендер. assist —
# коррекция реально применяется.
_LOOK_MANAGEMENT_MODES = ("off", "shadow", "assist")
LOOK_MANAGEMENT_MODE = os.environ.get("LOOK_MANAGEMENT_MODE", "off").strip().lower()
if LOOK_MANAGEMENT_MODE not in _LOOK_MANAGEMENT_MODES:
    print(f"  ВНИМАНИЕ: LOOK_MANAGEMENT_MODE={LOOK_MANAGEMENT_MODE!r} не входит в "
          f"{_LOOK_MANAGEMENT_MODES} — откатываюсь на 'off'.")
    LOOK_MANAGEMENT_MODE = "off"

# Единственный источник истины для channel-scoping (см. lookbook_add.py и
# load_lookbook() ниже) — "just metadata for audit", не полноценный
# мульти-тенант: один репозиторий пайплайна = один канал (см. CHANNEL.md),
# это только защита от случайного copy-paste lookbook.json между
# репозиториями РАЗНЫХ каналов (git merge/cherry-pick/ручной cp пронесли бы
# его МИМО lookbook_add.py целиком — там же единственная проверка была бы
# бесполезна).
CHANNEL_ID = (os.environ.get("CHANNEL_ID") or "default").strip() or "default"

# Бампить ВРУЧНУЮ при любой смене САМОЙ ЛОГИКИ/формулы classify_domain/
# find_reference/compute_correction — не только чисел констант ниже (те уже
# автоматически попадают в cache_signature() через _cache_relevant_constants()).
# Без этого правка алгоритма (не просто порога) молча пережила бы старый
# кэш temp_smart/ с прошлым поведением — тот же класс бага, что уже
# документирован у params_hash в pipeline_smart.py (там забывали
# queries[i]/captions, поймали вживую).
POLICY_VERSION = "1"

# --- Домены: CLIP-промпты + margin-gate (тот же принцип, что уже применяет
# RISKY_QUERY_MARGIN в pipeline_smart.is_relevant_candidate() — разрыв
# top-1/top-2, не абсолютный скор, решает уверенность). Оба порога ниже —
# разумные стартовые значения, НЕ откалиброванные на реальных кадрах этого
# канала (нет ни одного эпизода, см. докстринг модуля) — та же честная
# маркировка, что уже применяет visual_qc.py к своим SHARPNESS_REJECT/
# NOISE_REJECT/AESTHETIC_BORDERLINE. ---
DOMAIN_PROMPTS = {
    "snow": "snowy winter outdoor scene with snow and ice",
    "night": "dark night scene, low light, moonlight",
    "museum_daylight": "bright museum interior in daylight",
    "portrait": "portrait of a person, face closeup",
    "urban": "urban city street scene",
    "archive_bw": "black and white archival historical photo",
    "ai_illustration": "digital illustration, painted artwork",
    "battle": "battle scene with weapons and armor",
}
DOMAIN_MARGIN = 0.02
# Тай-брейк близкой ничьей между "средой" (определяет освещение/температуру
# цвета кадра) и "предметом" (у оружия/человека нет своей независимой
# температуры цвета — она приходит от среды) — см. classify_domain().
# Реальный случай, найденный по жалобе ("меч всё ещё холодный"): кадр
# меча крупным планом В СНЕГУ CLIP видит одновременно похожим на "battle"
# (оружие в кадре) и "snow" (снежное окружение), margin < DOMAIN_MARGIN —
# оба технически верны, кадр правда и то, и другое. Только environment/
# subject домены участвуют в тай-брейке — неоднозначность с archive_bw/
# ai_illustration (не среда и не предмет) им НЕ разрешается, честный
# отказ (None, 0.0), как и раньше.
ENVIRONMENT_DOMAINS = {"snow", "night", "museum_daylight", "urban"}
SUBJECT_DOMAINS = {"battle", "portrait"}

MAX_MATCH_DISTANCE = 35.0
MAX_STRENGTH = 0.35   # тот же порядок, что AUTO_WB_MAX_STRENGTH — частичная
                        # коррекция, никогда полный снап к эталону
GAIN_CLAMP = (0.5, 1.8)   # тот же диапазон и та же причина, что auto_wb_params
                            # (ffmpeg тихо роняет colorchannelmixer за [-2,2])
DELTA_STEP_CLAMP = (6.0, 4.0, 4.0)   # (dL, da, db) — максимальный шаг EMA за
                                       # один клип, тот же принцип клэмпа
                                       # шага, что max(-0.035,min(0.035,...))
                                       # у brightness_bias
DEFAULT_MAX_CORRECTION_DELTA = (10.0, 8.0, 8.0)   # (dL, da, db), если у
                                                     # эталона поле не задано
OVERSATURATION_FACTOR = 1.5   # рост хромы (sqrt(a^2+b^2)) свыше этого — hard-fail

# Arc-stage-осведомлённая модуляция силы коррекции (см. speech_planner.
# assign_chapter_arcs — "заход-якорь"/"постановка"/"слом"/"доказательство"/
# "вывод"/"перенос-на-зрителя"/"мостик" для BLOCK-секций, "hook"/"final" для
# HOOK/FINAL). Профессиональный колорист держит грейд ПОДЧИНЁННЫМ сюжету —
# полная сила на драматургически важных стадиях (сам момент разоблачения
# мифа), мягче на связках/переходах, а не одинаковая сила по всему ролику
# (это и есть та "шаблонность", от которой отличаются каналы с ручной
# доводкой). НИКОГДА не превышает 1.0 — MAX_STRENGTH остаётся абсолютным
# потолком (см. его комментарий выше: "частичная коррекция, никогда полный
# снап к эталону"), STAGE_BIAS_CLAMP может только УМЕНЬШИТЬ эффективную
# силу на переходных стадиях, никогда не увеличить её сверх уже
# откалиброванного MAX_STRENGTH. Некалиброванные, разумные стартовые
# значения (нет ни одного реального эпизода канала) — та же честная
# маркировка, что у DOMAIN_MARGIN/MAX_MATCH_DISTANCE выше. Отсутствие
# speech_plan.json (эпизод без Speech Director) -> arc_stage=None на каждом
# клипе -> stage_bias=1.0 везде -> байт-в-байт прежнее поведение.
ARC_STAGE_STRENGTH_BIAS = {
    "hook": 1.00, "слом": 1.00, "доказательство": 1.00,
    "заход-якорь": 0.85, "постановка": 0.90, "мостик": 0.80,
    "вывод": 0.95, "перенос-на-зрителя": 0.95, "final": 0.85,
}
STAGE_BIAS_CLAMP = (0.6, 1.0)

# Skin tone corridor — реальный, задокументированный стандарт колористики
# (vectorscope "skin tone line": угол тона кожи держится в узком коридоре
# независимо от тона/освещения, меняется в основном ХРОМА, не угол — см.
# https://caitlinwatson.com/what-is-the-skin-tone-line-in-the-vectorscope-and-how-do-i-use-it/,
# https://pixelvalleystudio.com/pmf-articles/the-skin-tone-line). Классический
# vectorscope использует IQ-плоскость (угол ~116-126°) — другое цветовое
# пространство, не Lab a/b этого модуля; порт готового числа был бы неверен.
# Вместо этого — угол посчитан В ТОМ ЖЕ Lab a/b, что и весь остальной код
# этого файла, на 7 реальных эталонных RGB светлой/средней/смуглой/тёмной
# кожи (через _srgb_to_lab, тот же путь, что кадр -> Lab везде в этом
# модуле): диапазон получился 54.7°-70.8° — тот же структурный вывод, что
# и у стандарта (узкий, стабильный угол на всём диапазоне тонов кожи), с
# запасом на реальные условия съёмки/WB стока. Хрома естественно шире
# (17-49 на тех же эталонах, зависит от освещения/тона) — с запасом.
SKIN_HUE_RANGE_DEG = (40.0, 85.0)
SKIN_CHROMA_RANGE = (8.0, 65.0)
# Экспозиция кожи — отдельный, тоже реальный и задокументированный
# стандарт (не выдуман): классическое broadcast-правило "кожу держать
# ~70 IRE" (светлая кожа ~60-70 IRE, тёмная ~40-55 IRE — см.
# https://wolfcrow.com/a-quick-look-at-understanding-ire-values/). IRE и
# Lab L — РАЗНЫЕ шкалы (гамма-кодированная яркость видео против
# перцептивной CIE-светлоты), порт числа "70" в L было бы неверно, тем
# же способом, что и с углом тона выше. Вместо жёсткого "цель — 70 IRE"
# (это was бы уже НЕ защитой, а принудительной перекоррекцией — конфликт
# с осознанно тёмной/атмосферной сценой, где кожа в кадре законно в
# тени) — более мягкая, чисто защитная граница: РЕАЛЬНЫЙ диапазон L тех
# же 7 эталонных RGB кожи (см. SKIN_HUE_RANGE_DEG выше) — 18.9-91.0, с
# запасом. Отсекает только физически неправдоподобное (кожа выдавлена в
# черноту или засвечена до потери детали), не навязывает конкретную
# яркость.
SKIN_LUMA_RANGE = (12.0, 95.0)


def _lab_hue_chroma(lab):
    return math.degrees(math.atan2(lab[2], lab[1])) % 360, math.hypot(lab[1], lab[2])


def _skin_gains_stay_in_corridor(skin_rgb, gains):
    """True, если применение gains к измеренному цвету кожи В КАДРЕ не
    выталкивает её за пределы естественного коридора (тон + хрома + грубая
    защита экспозиции, см. константы выше). skin_rgb — реальное среднее по
    коже ЭТОГО кадра (см. pipeline_smart.skin_tone_stats), не абстрактный
    эталон — коррекция, подходящая для металла/камня доспеха, может быть
    совершенно неверной для кожи руки в том же кадре, у неё другая физика
    цвета (кровь/меланин, не пигмент/минерал)."""
    projected = tuple(max(0.0, min(1.0, skin_rgb[i] * gains[i])) for i in range(3))
    lab = _srgb_to_lab(projected)
    hue, chroma = _lab_hue_chroma(lab)
    return (SKIN_HUE_RANGE_DEG[0] <= hue <= SKIN_HUE_RANGE_DEG[1] and
            SKIN_CHROMA_RANGE[0] <= chroma <= SKIN_CHROMA_RANGE[1] and
            SKIN_LUMA_RANGE[0] <= lab[0] <= SKIN_LUMA_RANGE[1])

# --- Веса для расстояния "кадр -> эталон" (см. find_reference). Lab — три
# оси в сопоставимом масштабе (L: 0..100, a/b обычно в пределах ±50) — сама
# по себе почти готовая метрика. brightness/contrast (0..1) и temperature
# (warm_bias, -1..1) — вспомогательные оси в другом масштабе, коэффициенты
# ниже приводят их к сопоставимому с Lab порядку величины. Не откалибровано
# (см. DOMAIN_MARGIN выше) — разумная стартовая пропорция. ---
AUX_BRIGHTNESS_WEIGHT = 40.0
AUX_CONTRAST_WEIGHT = 20.0
AUX_TEMP_WEIGHT = 15.0


def _cache_relevant_constants():
    """Все пороги/параметры, которые влияют на итоговый filter_str при
    НЕИЗМЕННОМ lookbook.json — см. cache_signature() ниже. Забыть добавить
    сюда новую константу — тот же класс бага, что уже документирован у
    params_hash в pipeline_smart.py (там забывали queries[i]/captions,
    поймали вживую) — при добавлении новой ручки в compute_correction()/
    find_reference()/classify_domain() дописать её сюда же."""
    return (POLICY_VERSION, DOMAIN_MARGIN, MAX_MATCH_DISTANCE, MAX_STRENGTH,
            GAIN_CLAMP, DELTA_STEP_CLAMP, DEFAULT_MAX_CORRECTION_DELTA,
            OVERSATURATION_FACTOR, AUX_BRIGHTNESS_WEIGHT, AUX_CONTRAST_WEIGHT,
            AUX_TEMP_WEIGHT, tuple(sorted(ARC_STAGE_STRENGTH_BIAS.items())),
            STAGE_BIAS_CLAMP)


def cache_signature():
    """Единственный источник истины для инвалидации кэша temp_smart/
    (см. pipeline_smart.py main()) по состоянию Look Management —
    pipeline_smart.py читает ТОЛЬКО через эту функцию, не лезет во
    внутренние константы этого модуля напрямую (та же дисциплина
    ответственности, что уже описана в ЗАМЕТКЕ НА БУДУЩЕЕ у _warm_mult() в
    pipeline_smart.py).

    "off" И "shadow" дают одну и ту же сигнатуру: обе НИКОГДА не трогают
    filter_str (shadow всегда возвращает None, см. look_correction_filter),
    рендер побитово идентичен — кэш можно свободно переиспользовать при
    экспериментах в shadow, инвалидация имеет смысл только для "assist"."""
    if LOOK_MANAGEMENT_MODE != "assist":
        return "look:off"
    try:
        with open(LOOKBOOK_PATH, "rb") as f:
            lookbook_digest = hashlib.md5(f.read()).hexdigest()[:12]
    except OSError:
        return "look:on:missing"
    payload = repr((lookbook_digest, CHANNEL_ID, _cache_relevant_constants()))
    return f"look:on:{hashlib.md5(payload.encode()).hexdigest()[:12]}"


# ---------- sRGB <-> CIE Lab (D65), скалярные (r,g,b) в [0,1] ----------

def _srgb_to_linear(c):
    c = max(0.0, min(1.0, c))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c):
    """Гамма-кодирование, c ожидается >=0 (отрицательный линейный RGB
    нельзя честно закодировать степенной функцией без комплексных чисел —
    клэмп здесь ТОЛЬКО защита от math domain error, не место для
    обнаружения выхода за гамму, см. _lab_to_linear_rgb/compute_correction,
    где это обнаруживается ДО этой функции, на линейных значениях)."""
    c = max(0.0, c)
    v = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055
    return v


_XN, _YN, _ZN = 0.95047, 1.0, 1.08883


def _srgb_to_lab(rgb):
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    def f(t):
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = f(x / _XN), f(y / _YN), f(z / _ZN)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return (L, a, b_)


def _lab_to_linear_rgb(lab):
    """(r,g,b) ЛИНЕЙНЫЕ (до гамма-кодирования), БЕЗ клэмпа — значения вне
    [0,1] здесь однозначный, чистый сигнал "цель вне гаммы sRGB" (см.
    compute_correction). После гамма-кодирования (_linear_to_srgb)
    отрицательные значения уже неотличимы от лёгкого клэмпа около нуля —
    степенная функция не может честно закодировать отрицательное число, а
    после неё сигнал был бы смазан. Проверять выход за гамму нужно ЗДЕСЬ, в
    линейном пространстве, не постфактум на sRGB."""
    L, a, b_ = lab

    def f_inv(t):
        return t ** 3 if t > 6 / 29 else 3 * (6 / 29) ** 2 * (t - 4 / 29)

    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b_ / 200
    x, y, z = _XN * f_inv(fx), _YN * f_inv(fy), _ZN * f_inv(fz)
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return (r, g, b)


def _lab_to_srgb(lab):
    """(r,g,b) гамма-кодированные (sRGB), клэмпнутые в [0,1] на выходе —
    для реального использования (не для обнаружения выхода за гамму, см.
    _lab_to_linear_rgb/compute_correction)."""
    return tuple(max(0.0, min(1.0, _linear_to_srgb(c))) for c in _lab_to_linear_rgb(lab))


# ---------- Домен ----------

def _domain_scores(image_path):
    """{domain: CLIP-скор}, или None при отключённой/сломанной фиче/сбое —
    единственное место, которое реально вызывает CLIP (вынесено отдельно от
    classify_domain(), чтобы ранжирование/margin-gate тестировались без
    реального torch — тот же принцип разделения, что уже применяет
    test_visual_qc.py к своим scorer-функциям). Кодирует картинку и все
    промпты ОДНИМ вызовом модели (не через pipeline_smart.clip_relevance() в
    цикле — та кодирует картинку заново на каждый текст, здесь текстов
    много (DOMAIN_PROMPTS), лишний повторный проход энкодера того же кадра
    не нужен)."""
    if not pipeline_smart.CLIP_ENABLED or pipeline_smart.CLIP_BROKEN:
        return None
    try:
        import torch
        model, processor = pipeline_smart.get_clip_model()
        img = Image.open(image_path).convert("RGB")
        domains = list(DOMAIN_PROMPTS.keys())
        texts = list(DOMAIN_PROMPTS.values())
        inputs = processor(text=texts, images=[img], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_e = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        scores = (img_e @ txt_e.T)[0].tolist()
    except ImportError:
        pipeline_smart.CLIP_BROKEN = True
        return None
    except Exception:
        return None
    return dict(zip(domains, scores))


def classify_domain(image_path):
    """(domain, margin) — домен с максимальным CLIP-скором среди
    DOMAIN_PROMPTS; margin — разрыв top-1/top-2 (уверенность, не абсолютный
    скор — тот же принцип, что RISKY_QUERY_MARGIN в
    pipeline_smart.is_relevant_candidate()). (None, 0.0), если CLIP
    недоступен или разрыв меньше DOMAIN_MARGIN — С ОДНИМ исключением: если
    top-1/top-2 — ровно один environment- и один subject-домен (см.
    ENVIRONMENT_DOMAINS/SUBJECT_DOMAINS выше), при близкой ничьей
    побеждает environment (для грейда важнее освещение/среда, не предмет
    в кадре) — margin возвращается ЧЕСТНЫЙ (маленький, не подделанный),
    вызывающий код видит реальную неуверенность в отчёте."""
    scores = _domain_scores(image_path)
    if not scores:
        return None, 0.0
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_domain, top_score = ranked[0]
    second_domain = ranked[1][0] if len(ranked) > 1 else None
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = top_score - second_score
    if margin >= DOMAIN_MARGIN:
        return top_domain, margin
    pair = {top_domain, second_domain}
    if len(pair & ENVIRONMENT_DOMAINS) == 1 and len(pair & SUBJECT_DOMAINS) == 1:
        return (pair & ENVIRONMENT_DOMAINS).pop(), margin
    return None, 0.0


def _domain_scores_from_text(text):
    """То же самое, что _domain_scores(), но ТЕКСТ-vs-ТЕКСТ (сам текст
    блока сценария против DOMAIN_PROMPTS), не изображение-vs-текст. Ни в
    этом модуле, ни в pipeline_smart.py такой ветки раньше не было — везде
    до сих пор CLIP вызывался только совместным image+text forward'ом
    (см. clip_relevance()/_domain_scores() выше). model.get_text_features()
    — реальный, документированный метод transformers.CLIPModel для
    текстового-только эмбеддинга, тот же закэшированный get_clip_model(),
    ноль новых зависимостей/загрузок модели — просто раньше не
    использовался. Нужно для Semantic Visual Director (scripts/
    visual_director.py) — домен ОЖИДАНИЯ по тексту фразы, для сравнения с
    доменом КАНДИДАТА (classify_domain() на картинке)."""
    if not pipeline_smart.CLIP_ENABLED or pipeline_smart.CLIP_BROKEN:
        return None
    try:
        import torch
        model, processor = pipeline_smart.get_clip_model()
        domains = list(DOMAIN_PROMPTS.keys())
        texts = [text] + list(DOMAIN_PROMPTS.values())
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            txt_e = model.get_text_features(**inputs)
        txt_e = txt_e / txt_e.norm(dim=-1, keepdim=True)
        scores = (txt_e[0:1] @ txt_e[1:].T)[0].tolist()
    except ImportError:
        pipeline_smart.CLIP_BROKEN = True
        return None
    except Exception:
        return None
    return dict(zip(domains, scores))


def text_domain_hint(text):
    """(domain, margin) — та же margin-gate логика, что classify_domain(),
    только по тексту фразы, не по картинке (см. _domain_scores_from_text()).
    (None, 0.0) при недоступном CLIP или недостаточной уверенности."""
    scores = _domain_scores_from_text(text)
    if not scores:
        return None, 0.0
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_domain, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = top_score - second_score
    if margin < DOMAIN_MARGIN:
        return None, 0.0
    return top_domain, margin


# ---------- Эталон ----------

def _distance(frame_lab, frame_brightness, frame_contrast, frame_temp, ref):
    dl = frame_lab[0] - ref["lab_mean"][0]
    da = frame_lab[1] - ref["lab_mean"][1]
    db = frame_lab[2] - ref["lab_mean"][2]
    lab_dist = math.sqrt(dl * dl + da * da + db * db)
    aux_dist = (abs(frame_brightness - ref.get("brightness", frame_brightness)) * AUX_BRIGHTNESS_WEIGHT
                + abs(frame_contrast - ref.get("contrast", frame_contrast)) * AUX_CONTRAST_WEIGHT
                + abs(frame_temp - ref.get("temperature", frame_temp)) * AUX_TEMP_WEIGHT)
    return lab_dist + aux_dist


def find_reference(domain, frame_lab, frame_brightness, frame_contrast, frame_temp, lookbook):
    """(reference_dict_or_None, confidence 0..1) — ближайший эталон ТОГО ЖЕ
    домена по взвешенному расстоянию (см. _distance). None, если в lookbook
    нет эталонов этого домена, или ближайший всё равно дальше
    MAX_MATCH_DISTANCE (честный fail-safe — лучше не корректировать, чем
    притянуть кадр к чужому по духу эталону)."""
    candidates = [r for r in lookbook.get("references", []) if r.get("domain") == domain]
    if not candidates:
        return None, 0.0
    scored = sorted(
        ((r, _distance(frame_lab, frame_brightness, frame_contrast, frame_temp, r)) for r in candidates),
        key=lambda x: x[1])
    best, best_dist = scored[0]
    if best_dist > MAX_MATCH_DISTANCE:
        return None, 0.0
    confidence = max(0.0, min(1.0, 1.0 - best_dist / MAX_MATCH_DISTANCE))
    return best, confidence


# ---------- Коррекция ----------

# P1-3 форензик-аудита (реальный, подтверждённый чтением кода структурный
# риск): compute_correction() выше считает gains из levels/wb, измеренных
# на СЫРОМ, негрейженном фото — но сам colorchannelmixer(gains) в реальном
# рендере (main() в pipeline_smart.py) приклеивается ПОСЛЕ полного
# творческого грейда film_look() (eq/colorbalance/curves/selectivecolor/
# vignette/halation), не до него. "Ближе к эталону" на входных, сырых
# измерениях НЕ гарантирует "ближе к эталону" на реально показанном
# зрителю кадре — film_look() между измерением и применением коррекции
# нелинейно двигает цвет. compute_correction() сам по себе интерполяция
# к эталону (raw_delta*strength) — она МАТЕМАТИЧЕСКИ не может отдалить
# на измеренном пространстве, проблема именно в разрыве между
# "где измерено" и "где применено".
def _render_graded_preview(image_path, section, levels, wb, domain, extra_filter=None):
    """Крошечный (160x90) превью-кадр через РЕАЛЬНЫЙ film_look()-граф
    (+ опционально доп. фильтр-кандидат коррекции) — то, что реально
    получит зритель ПОСЛЕ творческого грейда, не сырое фото до него.
    photo_hash — детерминированный int по пути файла (не обязан совпадать
    с хэшем финального рендера — обе стороны сравнения ниже используют
    ОДИН и тот же hash/section/bias, так что сравнение честное независимо
    от конкретного значения). Возвращает путь к PNG или None при сбое
    ffmpeg (честный откат — closed-loop проверка тогда просто
    пропускается, см. _closed_loop_improves)."""
    photo_hash = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
    vf = pipeline_smart.film_look(photo_hash, section, 0.0, 0.0, levels, wb, domain)
    if extra_filter:
        vf += f",{extra_filter}"
    vf += ",scale=160:90"
    suffix = hashlib.md5(vf.encode()).hexdigest()[:12]
    out_path = os.path.join(tempfile.gettempdir(), f"lookpreview_{suffix}.png")
    try:
        r = subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", image_path, "-frames:v", "1",
                            "-vf", vf, out_path], capture_output=True, timeout=20)
    except Exception:
        return None
    if r.returncode != 0 or not os.path.exists(out_path):
        return None
    return out_path


GRADE_REFERENCE_SECTIONS = ("HOOK", "BODY", "FINAL")


def _grade_recipe_fingerprint():
    """Хэш исходника film_look() — тот же принцип, что params_hash уже
    использует для инвалидации кэша клипов (см. main() в pipeline_smart.py),
    только здесь применён к lookbook-эталонам. Эталон хранит, ПРИ КАКОМ
    коде film_look() он был измерен (см. graded_reference_lab); расхождение
    с текущим film_look() на чтении — честный сигнал "эталон устарел, надо
    перемерить scripts/lookbook_remeasure.py", а не молчаливое сравнение
    против грейда, который зритель больше не увидит."""
    return hashlib.md5(inspect.getsource(pipeline_smart.film_look).encode()).hexdigest()[:12]


def graded_reference_lab(image_path, domain, levels, wb):
    """{section: [L,a,b] или None} по каждой из GRADE_REFERENCE_SECTIONS —
    реальное измерение эталонного кадра ПОСЛЕ того же film_look()-графа,
    что видит зритель (тот же принцип, что уже применяет
    _render_graded_preview/_closed_loop_improves для candidate-кадра, см.
    докстринг там). ПРОБЛЕМА, которую это закрывает (подтверждено внешним
    разбором, конкретно "pro_color_workflow_sources": lookbook хранит
    СЫРОЙ lab_mean исходника, closed-loop сравнивает с ним УЖЕ ГРЕЙЖЕННЫЙ
    after — сравнение не "в одном пространстве", тот же класс разрыва, что
    P1-3 форензик-аудита уже описал для candidate-стороны, но не для
    reference-стороны). Раздельно по секции, потому что MOOD_GRADE у
    film_look() разный на HOOK/BODY/FINAL, а один и тот же эталон домена
    может быть ближайшим для кадра в любой из трёх секций. Секция с
    неудавшимся рендером/измерением превью получает None (честный отказ,
    вызывающий код обязан считать это "недоступно", не подставлять raw)."""
    out = {}
    for section in GRADE_REFERENCE_SECTIONS:
        preview = _render_graded_preview(image_path, section, levels, wb, domain)
        if preview is None:
            out[section] = None
            continue
        try:
            measured = pipeline_smart.measure_levels(preview, want_wb=True)
            wb_measured = measured[1] if measured else None
        finally:
            try:
                os.remove(preview)
            except OSError:
                pass
        if wb_measured is None or wb_measured[0] is None:
            out[section] = None
            continue
        out[section] = list(_srgb_to_lab(wb_measured))
    return out


def _mood_section_key(section):
    """Тот же выбор MOOD_GRADE-ключа, что film_look() реально использует
    (см. pipeline_smart.py: 'HOOK'/'FINAL' по startswith, иначе 'BODY') —
    нужен, чтобы искать graded_lab_mean эталона по ТОЙ ЖЕ секции, для
    которой он посчитан в graded_reference_lab()."""
    if section.startswith("HOOK"):
        return "HOOK"
    if section.startswith("FINAL"):
        return "FINAL"
    return "BODY"


def _closed_loop_improves(image_path, section, levels, wb, domain, reference, gains):
    """True/False — применение gains (кандидат-коррекция) РЕАЛЬНО
    приближает итоговый (уже прошедший film_look()) кадр к ТАК ЖЕ
    грейженному эталону (см. graded_reference_lab), не только формально
    уменьшает расстояние на сырых, негрейженных измерениях (см. докстринг
    выше про разрыв "где измерено"/"где применено" — раньше этот разрыв
    был закрыт только для candidate-стороны, эталон сравнивался всё ещё в
    сыром виде, см. graded_reference_lab). None — fail-open (та же честная
    деградация, что у остальных опциональных проверок пайплайна): сбой
    рендера превью, эталон без graded_lab_mean этой секции (старый формат,
    не мигрированный scripts/lookbook_remeasure.py) или несовпадение
    graded_recipe_fingerprint (film_look() поменялся после того, как этот
    эталон был измерен — раньше НЕ отслеживалось вообще). Эталон с
    pregraded=True (см. lookbook_add.py --pregraded) пропускает fingerprint-
    гейт: его graded_lab_mean — не результат film_look(), а измерение уже
    готового, утверждённого файла, поэтому смена film_look() его не
    устаревляет — гейт здесь бессмысленен, не просто необязателен."""
    if not reference.get("pregraded") and reference.get("graded_recipe_fingerprint") != _grade_recipe_fingerprint():
        return None
    ref_lab_raw = (reference.get("graded_lab_mean") or {}).get(_mood_section_key(section))
    if ref_lab_raw is None:
        return None
    gains_filter = f"colorchannelmixer=rr={gains[0]:.4f}:gg={gains[1]:.4f}:bb={gains[2]:.4f}"
    base_preview = _render_graded_preview(image_path, section, levels, wb, domain)
    after_preview = _render_graded_preview(image_path, section, levels, wb, domain, extra_filter=gains_filter)
    try:
        if base_preview is None or after_preview is None:
            return None
        base_wb = pipeline_smart.measure_levels(base_preview, want_wb=True)[1]
        after_wb = pipeline_smart.measure_levels(after_preview, want_wb=True)[1]
        if base_wb is None or after_wb is None or base_wb[0] is None or after_wb[0] is None:
            return None
        base_lab = _srgb_to_lab(base_wb)
        after_lab = _srgb_to_lab(after_wb)
        ref_lab = tuple(ref_lab_raw)
        base_dist = math.sqrt(sum((base_lab[i] - ref_lab[i]) ** 2 for i in range(3)))
        after_dist = math.sqrt(sum((after_lab[i] - ref_lab[i]) ** 2 for i in range(3)))
        return after_dist < base_dist
    finally:
        for p in (base_preview, after_preview):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass


def compute_correction(frame_rgb_means, frame_lab, reference, confidence, prev_delta):
    """(gains_or_None, qc_dict, smoothed_delta). gains — (r,g,b) для
    colorchannelmixer, None при hard QC-fail (клиппинг/оверсатурация) —
    вызывающий код в этом случае откатывается к оригиналу БЕЗ коррекции,
    prev_delta НЕ обновляется на отвергнутое значение (см.
    look_correction_filter) — отвергнутая попытка не должна тянуть за собой
    сглаживание будущих клипов."""
    ref_lab = tuple(reference["lab_mean"])
    max_d = reference.get("max_correction_delta", DEFAULT_MAX_CORRECTION_DELTA)
    raw_delta = tuple(ref_lab[i] - frame_lab[i] for i in range(3))
    clamped_delta = tuple(max(-max_d[i], min(max_d[i], raw_delta[i])) for i in range(3))
    strength = confidence * MAX_STRENGTH
    target_delta = tuple(clamped_delta[i] * strength for i in range(3))

    if prev_delta is None:
        smoothed_delta = target_delta
    else:
        step = tuple(target_delta[i] - prev_delta[i] for i in range(3))
        step_clamped = tuple(max(-DELTA_STEP_CLAMP[i], min(DELTA_STEP_CLAMP[i], step[i])) for i in range(3))
        smoothed_delta = tuple(prev_delta[i] + step_clamped[i] for i in range(3))

    target_lab = tuple(frame_lab[i] + smoothed_delta[i] for i in range(3))
    target_rgb_linear = _lab_to_linear_rgb(target_lab)

    qc = {"decision": "ok", "notes": []}
    if any(c < -0.02 or c > 1.02 for c in target_rgb_linear):
        qc["decision"] = "reject_clipping"
        qc["notes"].append(f"target rgb (линейный) вне гаммы: {tuple(round(c, 4) for c in target_rgb_linear)}")
        return None, qc, prev_delta

    chroma_before = math.hypot(frame_lab[1], frame_lab[2])
    chroma_after = math.hypot(target_lab[1], target_lab[2])
    if chroma_before > 1e-3 and chroma_after > chroma_before * OVERSATURATION_FACTOR:
        qc["decision"] = "reject_oversaturation"
        qc["notes"].append(f"хрома выросла x{chroma_after / chroma_before:.2f} (порог x{OVERSATURATION_FACTOR})")
        return None, qc, prev_delta

    target_rgb = tuple(max(0.0, min(1.0, _linear_to_srgb(c))) for c in target_rgb_linear)
    gains = tuple(max(GAIN_CLAMP[0], min(GAIN_CLAMP[1], target_rgb[i] / max(frame_rgb_means[i], 1e-4)))
                  for i in range(3))
    if any(not (GAIN_CLAMP[0] < target_rgb[i] / max(frame_rgb_means[i], 1e-4) < GAIN_CLAMP[1]) for i in range(3)):
        qc["notes"].append("gain клэмпнут на границе диапазона — коррекция слабее запрошенной, не отклонена")
    return gains, qc, smoothed_delta


# ---------- Оркестрация ----------

EMPTY_STATE = {"delta": None, "domain": None, "reference_id": None}


def load_lookbook():
    """{"references": [...], "channel_id": ...} — с fail-closed проверкой
    channel_id ПРИ КАЖДОМ ЧТЕНИИ (не только на записи в lookbook_add.py):
    git merge/cherry-pick/ручной cp между репозиториями РАЗНЫХ каналов
    пронесли бы чужой lookbook.json мимо lookbook_add.py целиком — проверка
    только на запись была бы бесполезна против этого. "just metadata for
    audit", не полноценный мульти-тенант (см. докстринг модуля).

    Два случая MISMATCH (оба возвращают {"references": [], "channel_id":
    stored, "_reject_reason": ...} — эталоны игнорируются ЦЕЛИКОМ, тот же
    безопасный fallback, что и для реально пустого lookbook):
      1. Явное расхождение — записанный channel_id не совпадает с текущим.
      2. "default" не считается настоящей верификацией для ASSIST (режима,
         который реально меняет рендер) — если lookbook непуст, но никто не
         настроил CHANNEL_ID по-настоящему (ни при записи, ни сейчас), это
         тот же риск, что и mismatch, только тише. В shadow/off это
         ограничение НЕ действует — там безопасно смотреть превью даже без
         строгой настройки (см. cache_signature()/LOOK_MANAGEMENT_MODE)."""
    if not os.path.exists(LOOKBOOK_PATH):
        return {"references": []}
    try:
        data = json.load(open(LOOKBOOK_PATH, encoding="utf-8"))
    except Exception:
        return {"references": []}
    references = data.get("references", [])
    if not references:
        return data
    stored_channel = data.get("channel_id")
    if stored_channel is not None and stored_channel != CHANNEL_ID:
        print(f"  ВНИМАНИЕ: {LOOKBOOK_PATH} принадлежит каналу {stored_channel!r}, "
              f"текущий CHANNEL_ID={CHANNEL_ID!r} — игнорирую все эталоны (fail closed).")
        return {"references": [], "channel_id": stored_channel, "_reject_reason": "channel_mismatch"}
    if LOOK_MANAGEMENT_MODE == "assist" and (stored_channel in (None, "default") or CHANNEL_ID == "default"):
        print(f"  ВНИМАНИЕ: LOOK_MANAGEMENT_MODE=assist с непустым lookbook, но channel_id "
              f"не настроен по-настоящему (stored={stored_channel!r}, current={CHANNEL_ID!r}) — "
              f"отказываюсь применять коррекцию, пока не задан явный CHANNEL_ID.")
        return {"references": [], "channel_id": stored_channel, "_reject_reason": "channel_not_configured"}
    return data


def look_correction_filter(image_path, levels, wb, has_face, scene_boundary, section="",
                            prev_state=None, arc_stage=None):
    """Возвращает (filter_str_or_None, report_entry, new_state).
    filter_str — фрагмент ffmpeg-графа (colorchannelmixer), приклеивается
    ПОСЛЕ film_look(...) вызывающим кодом, никогда не заменяет его. None —
    коррекция не применяется/не рендерится (см. report_entry['decision'] за
    причиной) — вызывающий код просто не добавляет ничего.

    arc_stage — риторическая стадия ЭТОГО блока из media_plan/speech_plan.json
    (см. speech_planner.assign_chapter_arcs), None — если эпизод без Speech
    Director. Модулирует ТОЛЬКО эффективную силу коррекции через
    ARC_STAGE_STRENGTH_BIAS/STAGE_BIAS_CLAMP (см. их комментарий выше) — не
    трогает ни один из hard-гейтов ниже (clipping/oversaturation/skin
    corridor/closed-loop) — те применяются к уже посчитанным gains ровно так
    же, независимо от того, как была получена сила коррекции.

    section — секция сценария (HOOK/BLOCK*/FINAL) ЭТОГО клипа, нужна
    ТОЛЬКО для closed-loop проверки (_closed_loop_improves) — та рендерит
    настоящий film_look()-граф, а он выбирает MOOD_GRADE по секции (см.
    P1-3 форензик-аудита в докстринге _render_graded_preview выше).
    Пустая строка — тот же MOOD_GRADE["BODY"], что и раньше для
    неизвестной секции, не регрессия.

    prev_state/new_state — {"delta", "domain", "reference_id"} между
    соседними клипами (см. EMPTY_STATE). scene_boundary=True (граница
    секции сценария) ИЛИ смена domain/reference_id между этим и предыдущим
    СКОРРЕКТИРОВАННЫМ клипом — сбрасывает Lab-дельту сглаживания в None
    (не тянуть "плавный" переход между двумя никак не связанными эталонами
    — тот же принцип, по которому Speech Director никогда не считает паузу
    через границу секции, см. _flat_segment_bounds в speech_validator.py).
    domain/reference_id в состоянии обновляются на значения ЭТОГО клипа
    независимо от исхода QC (честное наблюдение "что видели"); delta же
    берётся НАПРЯМУЮ из возврата compute_correction() — та уже сама
    корректно возвращает либо новую сглаженную дельту, либо (при
    hard-reject) переданный ей prev_delta БЕЗ ИЗМЕНЕНИЙ. Передавать ей
    нужно effective_prev_delta (None после reset), не голый
    prev_state["delta"] — иначе hard-reject на reset-клипе тихо восстановил
    бы дельту старой, несвязанной сцены."""
    prev_state = prev_state or EMPTY_STATE
    if LOOK_MANAGEMENT_MODE == "off":
        return None, {"decision": "skipped_disabled"}, prev_state
    lookbook = load_lookbook()
    if lookbook.get("_reject_reason"):
        return None, {"decision": f"skipped_{lookbook['_reject_reason']}"}, prev_state
    if not lookbook.get("references"):
        return None, {"decision": "skipped_empty_lookbook"}, prev_state
    if has_face:
        return None, {"decision": "skipped_face_detected"}, prev_state
    if wb is None or levels is None or levels[0] is None:
        return None, {"decision": "skipped_no_signal"}, prev_state

    domain, margin = classify_domain(image_path)
    if domain is None:
        return None, {"decision": "skipped_low_domain_confidence"}, prev_state

    frame_lab = _srgb_to_lab(wb)
    warm_bias, _, _ = pipeline_smart._scene_bias(levels, wb)
    frame_brightness = (levels[0] + levels[1]) / 2.0
    frame_contrast = levels[1] - levels[0]

    reference, confidence = find_reference(domain, frame_lab, frame_brightness, frame_contrast, warm_bias, lookbook)
    if reference is None:
        return (None, {"decision": "skipped_no_reference_match", "domain": domain, "domain_margin": margin},
                prev_state)

    reference_id = reference.get("id")
    domain_changed = prev_state["domain"] is not None and prev_state["domain"] != domain
    reference_changed = prev_state["reference_id"] is not None and prev_state["reference_id"] != reference_id
    reset = scene_boundary or domain_changed or reference_changed
    reset_reason = ("scene_boundary" if scene_boundary else
                     "domain_changed" if domain_changed else
                     "reference_changed" if reference_changed else None)
    effective_prev_delta = None if reset else prev_state["delta"]

    # arc_stage=None (эпизод без Speech Director) -> stage_bias=1.0 ->
    # effective_confidence == confidence -> байт-в-байт прежнее поведение.
    stage_bias = (max(STAGE_BIAS_CLAMP[0], min(STAGE_BIAS_CLAMP[1], ARC_STAGE_STRENGTH_BIAS.get(arc_stage, 1.0)))
                  if arc_stage is not None else 1.0)
    effective_confidence = confidence * stage_bias

    gains, qc, new_delta = compute_correction(wb, frame_lab, reference, effective_confidence, effective_prev_delta)
    new_state = {"delta": new_delta, "domain": domain, "reference_id": reference_id}
    report = {
        "domain": domain, "domain_margin": round(margin, 4),
        "reference_id": reference_id, "confidence": round(confidence, 4),
        "arc_stage": arc_stage, "stage_bias": round(stage_bias, 3),
        "applied_delta": [round(x, 3) for x in new_delta] if new_delta else None,
        "ema_reset": reset, "reset_reason": reset_reason,
        "qc": qc,
    }
    if gains is None:
        report["decision"] = qc["decision"]
        return None, report, new_state

    # Skin tone corridor — реальный стандарт колористики (см. константы
    # выше). has_face уже отсёк ПОЛНОЕ лицо целиком (skipped_face_detected
    # выше) — это ДОПОЛНИТЕЛЬНЫЙ, ранее отсутствовавший сигнал для случая
    # "кожа есть (рука/плечо), лица не нашли": раньше такой кадр получал
    # коррекцию без единой проверки кожи вообще, та же физика цвета
    # (кровь/меланин), что и у лица, просто без каскада, который бы её
    # поймал. Дешёвая, локальная проверка — до closed-loop рендера (ниже),
    # чтобы не тратить лишний ffmpeg-вызов на то, что и так отклонится.
    skin_frac, skin_rgb = pipeline_smart.skin_tone_stats(image_path)
    report["skin_fraction"] = round(skin_frac, 4)
    if skin_rgb is not None and not _skin_gains_stay_in_corridor(skin_rgb, gains):
        report["decision"] = "reject_skin_tone_corridor"
        return None, report, new_state

    # P1-3: gains посчитаны на СЫРЫХ измерениях (frame_lab выше) — closed-
    # loop проверяет, что применение gains ПОСЛЕ реального film_look()-грейда
    # действительно приближает кадр к эталону, не только формально на входе
    # (см. _closed_loop_improves). None (сбой рендера превью) — fail-open,
    # решение остаётся как было посчитано (та же честная деградация, что и
    # у остальных опциональных проверок).
    closed_loop_ok = _closed_loop_improves(image_path, section, levels, wb, domain, reference, gains)
    report["closed_loop_improves"] = closed_loop_ok
    if closed_loop_ok is False:
        report["decision"] = "reject_closed_loop_no_improvement"
        return None, report, new_state

    if LOOK_MANAGEMENT_MODE == "shadow":
        report["decision"] = "shadow_would_apply"
        return None, report, new_state
    report["decision"] = "applied"
    filt = f"colorchannelmixer=rr={gains[0]:.4f}:gg={gains[1]:.4f}:bb={gains[2]:.4f}"
    return filt, report, new_state
