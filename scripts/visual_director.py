#!/usr/bin/env python3
"""Semantic Visual Director v1 — БЕСПЛАТНАЯ (без платного LLM в ядре)
надстройка над сегодняшним отбором фото-кандидатов (scripts/
pipeline_smart.py, pexels_photo()). Сегодняшний отбор отвечает на вопрос
«похожа ли картинка на поисковый запрос» (CLIP image-vs-query relevance) —
эта система добавляет несколько ДОПОЛНИТЕЛЬНЫХ, уже существующих в
пайплайне сигналов, чтобы приблизиться к вопросу «правильный ли это кадр
здесь и сейчас»:
  - Полный текст фразы (не производный keyword-запрос) — CLIP image-vs-
    sentence relevance, тот же pipeline_smart.clip_relevance(), который уже
    сегодня умеет принимать произвольный текст, просто раньше никто не
    вызывал его на полном предложении.
  - Функциональная роль кадра — pipeline_smart.classify_shot_function()
    (evidence/detail/context/hook/narrative), уже существует, сегодня
    используется только для QC-манифеста, не для отбора.
  - Домен ожидания по тексту (look_reference.text_domain_hint()) против
    домена кандидата (look_reference.classify_domain()) — совпадение даёт
    небольшой бонус.
  - История соседних клипов — новое скользящее окно (domain, role) уже
    выбранных клипов, штраф за повтор той же пары подряд.
  - Visual QC (резкость/шум) — переиспользует scripts/visual_qc.py
    scorer'ы, та же подготовка кадра, не копия.

ЧЕСТНО, явно: это НЕ понимание смысла фразы. CLIP-эмбеддинги слабы на
композиционных/атрибутивных связках («меч весом 5 кг» — не про «меч»
абстрактно, а про вес/держание) — распознать такую специфику значит
СГЕНЕРИРОВАТЬ уточнённое описание нужного кадра, а CLIP умеет только
СРАВНИВАТЬ уже существующий текст с уже существующими кандидатами. Это
ограничение, не упущение реализации — настоящее понимание смысла остаётся
за LLM-ядром v2, сознательно не строится здесь.

VISUAL_DIRECTOR_MODE: `off` (дефолт) — pexels_photo() не получает
director_score_fn вообще, поведение байт-в-байт как до этой фичи. `shadow`
— extra_score считается и попадает в media_plan/visual_director_report.json
(base_winner/director_winner/diverged), но РЕАЛЬНЫЙ выбор (какой файл
скачивается и используется в рендере) остаётся за сегодняшним 6-элементным
кортежем. `assist` — extra_score реально влияет на выбор.

ВАЖНО, честно про cost-tradeoff: pexels_photo() сегодня останавливает
скачивание кандидатов, как только нашёл 1 (без target_luma) или 2
(с target_luma) прошедших relevance/dedup/size-гейты — то есть Директору
часто буквально нечего ранжировать. При shadow/assist пул кандидатов
расширяется (DIRECTOR_MIN_POOL) — это РЕАЛЬНЫЕ дополнительные запросы к
Pexels на части блоков (не деньги, Pexels бесплатный, но реальная квота,
CLAUDE.md ЧАСТЬ 21: 200/час, 20000/мес) и реальные дополнительные локальные
CLIP-вызовы (sentence relevance + домен кандидата, поверх уже существующего
relevance-гейта) — не бесплатно по времени, только по деньгам. Ещё одна
причина, почему off остаётся дефолтом.

Не самостоятельный CLI-скрипт — вызывается из scripts/pipeline_smart.py."""
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

VIDEO_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
# РЕАЛЬНЫЙ БАГ (двойное исполнение pipeline_smart в одном процессе).
# pipeline_smart.py запускается как СКРИПТ, то есть живёт в sys.modules под
# именем "__main__". Голый `import pipeline_smart` не находит его там и
# исполняет ФАЙЛ ВТОРОЙ РАЗ — как отдельный модуль со своим набором
# глобалов. Последствия не косметические: модуль-дубль лениво грузит СВОЮ
# копию CLIP-модели (get_clip_model кэширует её в глобале _clip_model,
# который у копии свой) — то есть ~+600МБ RSS и второй прогон загрузки
# модели на каждый рендер, при том что _default_render_workers() считает
# число воркеров как раз по свободной памяти (~700МБ на воркера).
# Заодно расходятся флаги CLIP_BROKEN/PARALLAX_BROKEN (сбой, погашенный в
# одной копии, продолжает биться в другой) и дублируются предупреждения
# импорта (find_audio о устаревшем audio_fixed печатается дважды).
# Правильный импорт — переиспользовать уже исполненный __main__, если это и
# есть pipeline_smart. Запуск ЭТОГО модуля отдельным скриптом (__main__ —
# он сам) идёт по обычному пути, поведение не меняется.
_main_mod = sys.modules.get("__main__")
if (getattr(_main_mod, "__file__", None)
        and os.path.basename(_main_mod.__file__) == "pipeline_smart.py"
        and "pipeline_smart" not in sys.modules):
    sys.modules["pipeline_smart"] = _main_mod
_saved_argv = sys.argv
sys.argv = ["pipeline_smart.py", VIDEO_DIR]
import pipeline_smart  # noqa: E402
sys.argv = _saved_argv

import look_reference as lr  # noqa: E402  (text_domain_hint/classify_domain — CLIP-домен, переиспользуется, не дублируется)

# Нормализация/предупреждение/откат на "off" при мусорном значении — в общем
# реестре (scripts/feature_flags.py), поведение то же. Дефолт объявлен ТАМ:
# десятая точка чтения с собственным литералом — ровно тот механизм, которым
# код разошёлся с CLAUDE.md по VLM_ARBITER_MODE (см. докстринг реестра).
import feature_flags  # noqa: E402
_VISUAL_DIRECTOR_MODES = feature_flags.FLAGS["VISUAL_DIRECTOR_MODE"].allowed
VISUAL_DIRECTOR_MODE = feature_flags.mode("VISUAL_DIRECTOR_MODE")

DIRECTOR_MIN_POOL = 8   # ДОЛЖНО совпадать с pipeline_smart.DIRECTOR_MIN_POOL (та
                          # константа реально управляет good_needed в pexels_photo(),
                          # эта — только для cache_signature()/докстрингов ниже; см.
                          # комментарий у pipeline_smart.DIRECTOR_MIN_POOL про апгрейд
                          # 3->8 вместе с so400m+Jina ensemble). Расхождение значений
                          # тихо сломало бы инвалидацию кэша — сигнатура не отразила
                          # бы реальное изменение поведения.

SENTENCE_RELEVANCE_WEIGHT = 1.0   # доминирующий член — тот же порядок величины
                                    # (реалистичный диапазон ~0.15-0.35), что уже
                                    # использует is_relevant_candidate()

# РЕАЛЬНЫЙ, найденный вживую пробел (27 августа, videos/_test20s): compute_
# extra_score() добавляет sentence_relevance() (косинус против РЕАЛЬНОГО
# текста блока) как ранжирующий сигнал, но у него НЕ БЫЛО СВОЕГО порога —
# is_relevant в лексикографическом кортеже _score_and_pick() (pipeline_
# smart.py) гейтит только по ORIGIN QUERY (английская фраза-посредник), а
# extra_score просто переранжирует уже прошедших этот гейт кандидатов.
# Кандидат мог оказаться семантически слабым по смыслу РЕАЛЬНОЙ фразы и всё
# равно победить — просто как лучший из имеющихся, не потому что подходил.
#
# Порог выставляется калибровкой (scripts/calibrate_vision.py) по парам
# good/bad из tests/fixtures/director_calibration/calibration_pairs.json —
# та же методика, что раньше работала при импорте модуля (середина зазора
# между худшей верной и лучшей неверной парой, сани-гейт по запасу), только
# теперь в одном месте со всеми порогами моделей зрения: смена модели делает
# калибровку недействительной целиком (vision_model), а не по частям.
# DIRECTOR_RELEVANCE_FALLBACK — если калибровка не дала надёжного разделения.
DIRECTOR_RELEVANCE_FALLBACK = 0.13
DIRECTOR_RELEVANCE_MIN_MARGIN = 0.02   # ниже этого разделение good/bad на
                                         # калибровочных парах считается
                                         # ненадёжным — новый порог НЕ
                                         # принимается, остаётся прежний
                                         # (кэш или fallback), с громким
                                         # предупреждением в консоль.
CALIBRATION_PAIRS_PATH = os.path.join(
    REPO_ROOT, "tests", "fixtures", "director_calibration", "calibration_pairs.json")

# МОДЕЛЬ ОЦЕНКИ ФРАЗЫ — Qwen3-VL-Embedding-8B (GPU-ветка, 29.09).
#
# Раньше здесь жил ансамбль SigLIP2-so400m + Jina CLIP v2 (история калибровок
# — в git: CLIP -> SigLIP2-base -> so400m -> ансамбль с Jina, 113-позиционный
# бенчмарк). Решение владельца 29.09: SigLIP2 убрана из GPU-ветки целиком, а у
# Jina лицензия CC-BY-NC-4.0, для монетизируемого канала неприемлемая. Теперь
# фраза сценария (русский текст как есть) и кадр сравниваются той же моделью,
# что у гейтов и каскада (pipeline_smart._gate_image_vec/_gate_text_vec — один
# вектор картинки на все роли, одни кэши): Qwen3-VL-Embedding многоязычная и
# понимает составную фразу, а не мешок признаков, текст не обрезается
# (контекст 32k токенов против 64 у SigLIP2, где обрезалась каждая четвёртая
# фраза, см. TEXT_TRUNCATION_REPORT).
#
# ШКАЛА. Веса и бонусы режиссёра ниже (ROLE_SHOT_SIZE_BONUS, DOMAIN_MATCH_BONUS,
# ...) складываются с оценкой фразы и заданы на шкале прежнего ансамбля. Чтобы
# они сохранили смысл, сходство Qwen переводится на ту шкалу z-переносом
# (vision_model.to_legacy_level, «sentence»): среднее и разброс обеих моделей
# на одной матрице «кадры золотого набора × их русские фразы». Шкалу Qwen
# пишет калибровка (scripts/calibrate_vision.py), без неё оценки фразы нет
# (None, бонус не добавляется) — рендер без калибровки и так не начинается.

# Обрезка текста по лимиту токенов модели — видимость сохранена (отчёт эпизода
# media_plan/text_truncation_report.json пишется всегда): у Qwen контекст 32k
# токенов, фраза блока в него помещается целиком, и отчёт честно пуст.
TEXT_TRUNCATION_REPORT = []   # [{"model", "text", "tokens", "limit"}, ...]


def reset_text_truncation_report():
    """Для тестов и для чистого старта каждого прогона main()."""
    TEXT_TRUNCATION_REPORT.clear()


import qwen_vl_embed  # noqa: E402
import vision_model  # noqa: E402

SENTENCE_RELEVANCE_MODEL_VERSION = f"qwen3vl-embed:{qwen_vl_embed.signature()}|legacy-scale-v1"

# Некалиброванные, разумные стартовые бонусы (та же честная маркировка, что
# DOMAIN_MARGIN/MAX_MATCH_DISTANCE в look_reference.py) — нет ни одного
# реального эпизода канала, чтобы подтвердить вживую.
ROLE_SHOT_SIZE_BONUS = {
    ("evidence", "close"): 0.15, ("evidence", "detail"): 0.15,
    ("detail", "detail"): 0.20, ("detail", "close"): 0.10,
    ("hook", "wide"): 0.10, ("hook", "medium"): 0.05,
    ("context", "wide"): 0.15,
    ("narrative", "medium"): 0.05,
}
DOMAIN_MATCH_BONUS = 0.20
REPETITION_WINDOW = 3
REPETITION_PENALTY = 0.15
VISUAL_QC_SHARPNESS_WEIGHT = 0.10
VISUAL_QC_NOISE_WEIGHT = 0.05

# РЕАЛЬНЫЙ найденный вживую баг (deep-audit, videos/_test20s, 29 августа —
# по прямой просьбе пользователя "пересмотри снова получившийся видеоряд").
# extra_queries (см. докстринг pexels_photo()) собирает кандидатов из ВСЕХ
# авторских запросов секции сразу — Директор ранжирует их по compute_extra_
# score() БЕЗ учёта, из какого именно запроса пришёл кандидат. На реальном
# рендере это дало явную перестановку: слот с запросом "weighing scale
# metal object" (для фразы "Пятнадцать килограммов.") получил фото мечей и
# щита из ЧУЖОГО запроса ("knight armor holding sword two hands"), а
# соседний слот с запросом "dark cinema movie theatre screen" (для фразы
# про кино/видеоигры/учебники) получил ВИДЕО ВЕСОВ — которое, судя по
# содержанию, и было честным кандидатом ПЕРВОГО слота, просто выигранным
# у него чужим текстом. Причина: semantic_query_assignment() уже назначает
# запрос блоку ПО СМЫСЛУ его конкретной фразы (не позиционно) — это сильный
# сигнал, который compute_extra_score() полностью игнорировал, оценивая
# только "похож ли кандидат на ПОЛНУЮ (иногда дополненную соседями,
# см. semantic_context_text) фразу", где общая тема эпизода ("меч") может
# перетягивать оценку у короткого/абстрактного блока в пользу чужого,
# более "мечового" кандидата — тот же класс ограничения, что уже честно
# описан в докстринге sentence_relevance() ("композиционные/атрибутивные
# связки вроде «меч весом 5 кг» ему не по силам").
# Небольшой бонус кандидату, пришедшему из СВОЕГО (не чужого) запроса
# слота — не жёсткий приоритет (чужой кандидат всё ещё может победить,
# если он ощутимо лучше по всем остальным осям), просто восстанавливает
# вес уже вложенного смыслового решения semantic_query_assignment(), а не
# отдаёт его целиком на откуп CLIP-скорингу полной фразы. Некалиброванное,
# разумное стартовое значение (та же честная маркировка, что у бонусов
# выше) — нет откалиброванного набора реальных эпизодов, чтобы подтвердить
# точную величину; порядок величины намеренно сопоставим с DOMAIN_MATCH_
# BONUS, а не мельче ROLE_SHOT_SIZE_BONUS — ошибка, которую он чинит,
# оказалась не мелкой (3 из 5 слотов хука на реальном рендере).
SAME_QUERY_BONUS = 0.20

# OPENING_AESTHETIC_WEIGHT — реальный найденный вживую случай (29 августа,
# прямое требование пользователя после того, как VLM-арбитр не смог быть
# живьём проверен из-за исчерпанной квоты Gemini: "реализовать то, что мы
# хотим от Gemini, даже БЕЗ Gemini"). Проблема ("самый первый кадр ролика —
# технически точная, но скучная картинка") решается не только vision-LLM:
# в пайплайне уже есть РЕАЛЬНАЯ, локальная, ничего не стоящая модель
# эстетики — aesthetic_score() (LAION-Aesthetics head поверх CLIP-эмбеддинга,
# см. её докстринг: диапазон ~3-7 на реальных фото, подтверждён вживую на
# 25 кадрах этого же эпизода — топ-3 по оценке визуально заметно сильнее по
# композиции/свету). Баг был не "эстетики не существует", а АРХИТЕКТУРНЫЙ:
# aesthetic_val стоит в лексикографическом кортеже _score_and_pick() ПОСЛЕ
# extra (семантического скора) — то есть решает только когда семантика уже
# идеально равна, а на практике семантика конкретной фразы ("Пятнадцать
# килограммов." -> весы) почти всегда СТРОГО предпочитает буквальный, но
# скучный кандидат раньше, чем эстетика вообще успевает сравниться.
# is_opening внедряет эстетику РАНЬШЕ, прямо в сам extra-скор (не трогая
# порядок кортежа — общий, безопасный способ, уже проверенный на
# SAME_QUERY_BONUS/DOMAIN_MATCH_BONUS), поэтому решает и base-, и director-
# победителя ДО и НЕЗАВИСИМО от VLM-арбитра — работает всегда, при
# VLM_ARBITER_MODE=off, при исчерпанной квоте, без ключа вообще. Арбитр
# (когда доступен) остаётся дополнительной полировкой поверх уже
# качественного base/director выбора, не единственным источником качества.
# Нормализация — честно откалиброванный диапазон ИЗ ДОКСТРИНГА самой
# aesthetic_score() (3-7, не с потолка): (val-3)/4 даёт ~0..1 на типичных
# фото, верхний клэмп 1.5 — не режет реально выдающиеся кадры за пределами
# документированного диапазона. Вес 0.6 — того же порядка, что
# SENTENCE_RELEVANCE_WEIGHT=1.0 (эстетика РЕАЛЬНО конкурирует со смыслом
# для открывающего кадра, не тонет в нём и не полностью его перебивает) —
# как и SAME_QUERY_BONUS/DOMAIN_MATCH_BONUS, разумное стартовое значение,
# не откалиброванное на большом наборе эпизодов; честно так и помечено.
AESTHETIC_NORM_MIN = 3.0
AESTHETIC_NORM_RANGE = 4.0
OPENING_AESTHETIC_WEIGHT = 0.6

# OPENING_DOMAIN_WEIGHT — усиленный DOMAIN_MATCH_BONUS (0.20) специально
# для открывающего кадра (см. блок-комментарий у OPENING_AESTHETIC_WEIGHT
# выше и у её использования в compute_extra_score). Реальный, измеренный
# вживую случай: обычного DOMAIN_MATCH_BONUS не хватило, чтобы жанрово
# уместный ("battle"), но менее конвенционально "красивый" по LAION
# кандидат победил конвенционально более "красивый", но жанрово
# нейтральный — разрыв в эстетике (aesthetic 6.65 vs 5.61 -> ~0.16 разницы
# в OPENING_AESTHETIC_WEIGHT-члене) плюс SAME_QUERY_BONUS (0.20, теперь
# отключённый для opening, см. выше) в сумме перевешивали 0.20. Величина
# сопоставима с OPENING_AESTHETIC_WEIGHT намеренно — жанровое соответствие
# для открывающего кадра ценится наравне с визуальной привлекательностью,
# не подчинено ей. Разумное стартовое значение (та же честная маркировка),
# не откалиброванное на большом наборе эпизодов — тот же принцип.
OPENING_DOMAIN_WEIGHT = 0.5

# Arc-stage-осведомлённый бонус по крупности плана (см. speech_planner.
# assign_chapter_arcs — "заход-якорь"/"слом"/"доказательство"/... для
# BLOCK-секций, "hook"/"final" для HOOK/FINAL) — ДОПОЛНИТЕЛЬНЫЙ, независимый
# сигнал поверх ROLE_SHOT_SIZE_BONUS выше: роль (evidence/detail/context/
# hook/narrative) отвечает "что это за кадр по смыслу", arc_stage — "где мы
# сейчас в разоблачении мифа" (тот же принцип, по которому проф. монтажёр
# держит крупность плана подчинённой сюжету — сближение на слом/доказательство,
# общий план на заходе — а не одинаковой по всему ролику). Оба бонуса
# складываются, не заменяют друг друга. Некалиброванные, разумные стартовые
# значения — та же честная маркировка, что у ROLE_SHOT_SIZE_BONUS. arc_stage=None
# (эпизод без Speech Director) -> бонус 0.0 везде -> байт-в-байт прежнее поведение.
ARC_STAGE_SHOT_SIZE_BONUS = {
    ("слом", "detail"): 0.12, ("слом", "close"): 0.10,
    ("доказательство", "detail"): 0.10, ("доказательство", "close"): 0.08,
    ("заход-якорь", "wide"): 0.10, ("постановка", "wide"): 0.08,
    ("hook", "wide"): 0.05,
}


def cache_signature():
    """Единственный источник истины для инвалидации кэша temp_smart/ по
    состоянию Visual Director — тот же принцип и роль, что
    look_reference.cache_signature() (pipeline_smart.py читает только через
    эту функцию). РЕАЛЬНЫЙ, найденный внешним аудитом баг: раньше её не
    было вообще — params_hash в main() не включал НИЧЕГО про
    VISUAL_DIRECTOR_MODE, поэтому переключение off -> assist на уже
    отрендеренном эпизоде НЕ инвалидировало старые клипы (в отличие от
    Look Management, у которого своя сигнатура уже была правильно
    подключена) — cache-хиты молча пропускали анализ Директора, и
    "assist" на практике ничего не менял, пока не почистишь temp_smart/
    вручную (см. предупреждение "N клипов пропущено анализом" в main()).

    "off" И "shadow" дают одну и ту же сигнатуру (та же логика, что у
    look_reference: shadow никогда не трогает реальный выбор кандидата,
    рендер побитово идентичен off) — инвалидация имеет смысл только для
    "assist"."""
    if VISUAL_DIRECTOR_MODE != "assist":
        return "director:off"
    table_sig = hashlib.md5(repr((
        sorted(ROLE_SHOT_SIZE_BONUS.items()), DOMAIN_MATCH_BONUS,
        REPETITION_WINDOW, REPETITION_PENALTY,
        VISUAL_QC_SHARPNESS_WEIGHT, VISUAL_QC_NOISE_WEIGHT,
        SENTENCE_RELEVANCE_WEIGHT, SENTENCE_RELEVANCE_MODEL_VERSION, DIRECTOR_MIN_POOL,
        sorted(ARC_STAGE_SHOT_SIZE_BONUS.items()),
    )).encode()).hexdigest()[:8]
    return f"director:assist:{table_sig}"


def functional_role(block, is_section_start):
    """Тонкая обёртка над pipeline_smart.classify_shot_function() —
    переиспользование, не копия. Работает на ТЕХ ЖЕ post-split blocks, что
    уже использует главный цикл отбора (см. докстринг модуля про разрыв с
    speech_planner.classify_unit(), которого здесь сознательно избегаем)."""
    return pipeline_smart.classify_shot_function(block, is_section_start)


def sentence_relevance(image_path, block_text):
    """Близость картинки и ПОЛНОГО текста блока (русского, как он есть в
    сценарии) на шкале прежнего ансамбля (см. блок выше). None — нет текста,
    модели или калибровки шкалы: вызывающий код (compute_extra_score) тогда
    не добавляет этот бонус."""
    if not block_text:
        return None
    with pipeline_smart.stage_timer.stage("qwen_sentence"):
        img = pipeline_smart._gate_image_vec(image_path)
        txt = pipeline_smart._gate_text_vec(block_text) if img is not None else None
    if img is None or txt is None:
        return None
    try:
        return vision_model.to_legacy_level(float(img @ txt), "sentence")
    except vision_model.NotCalibrated:
        return None


def text_text_similarity(texts_a, texts_b):
    """Матрица близостей ТЕКСТ-ТЕКСТ (len(a) x len(b)) — русская фраза
    сценария против английского авторского запроса (распределение запросов
    по фразам, pipeline_smart.semantic_query_assignment). Та же модель, что
    у гейтов, с той же инструкцией поиска на обеих сторонах. Распределение
    берёт только ПОРЯДОК близостей, поэтому перенос шкалы здесь не нужен.
    None — модели нет."""
    if not texts_a or not texts_b:
        return None
    va = [pipeline_smart._gate_text_vec(t) for t in texts_a]
    vb = [pipeline_smart._gate_text_vec(t) for t in texts_b]
    if any(v is None for v in va) or any(v is None for v in vb):
        return None
    with pipeline_smart.stage_timer.stage("text_text_qwen", n_a=len(texts_a), n_b=len(texts_b)):
        return [[float(x @ y) for y in vb] for x in va]


def _relevance_model_signature():
    """Отпечаток всего, что влияет на шкалу sentence_relevance(): модель,
    протокол вопросов, шкалы переноса из калибровки. Меняется — порог
    режиссёра из калибровки для другой шкалы не годится."""
    cal = vision_model.calibration() or {}
    parts = "|".join([SENTENCE_RELEVANCE_MODEL_VERSION, str(vision_model.GATE_PROTOCOL_VERSION),
                      json.dumps((cal.get("scale") or {}).get("sentence"), sort_keys=True),
                      json.dumps(vision_model.LEGACY_SENTENCE_SCALE, sort_keys=True)])
    import ml_device
    parts += ml_device.tag()
    return hashlib.md5(parts.encode()).hexdigest()[:16]


def _load_calibration_pairs():
    """Пары image+caption+label из CALIBRATION_PAIRS_PATH. None, если файла
    нет/битый JSON."""
    try:
        with open(CALIBRATION_PAIRS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = data.get("pairs") or []
        out = []
        for p in pairs:
            img = os.path.join(REPO_ROOT, p["image"])
            if os.path.exists(img) and p.get("label") in ("good", "bad") and p.get("caption"):
                out.append((img, p["caption"], p["label"]))
        return out or None
    except Exception:
        return None


def calibrate_relevance_floor(score_fn=None):
    """Порог режиссёра на CALIBRATION_PAIRS_PATH — та же методика (середина
    зазора между худшей верной и лучшей неверной парой), что и раньше.
    Зовёт её scripts/calibrate_vision.py (score_fn — оценка фразы на шкале
    прежнего ансамбля, без калибровки её ещё не на что перевести), результат
    уходит в файл калибровки. None — нет пар или модель не ответила.
    Разделение меньше DIRECTOR_RELEVANCE_MIN_MARGIN возвращается для
    видимости, но порогом не становится (см. _resolve_director_relevance_floor)."""
    pairs = _load_calibration_pairs()
    if not pairs:
        return None
    score_fn = score_fn or sentence_relevance
    good_scores, bad_scores = [], []
    for image_path, caption, label in pairs:
        score = score_fn(image_path, caption)
        if score is None:
            return None
        (good_scores if label == "good" else bad_scores).append(score)
    if not good_scores or not bad_scores:
        return None
    min_good, max_bad = min(good_scores), max(bad_scores)
    margin = min_good - max_bad
    return {"floor": max_bad + margin / 2.0, "margin": margin,
            "min_good": min_good, "max_bad": max_bad,
            "good_scores": good_scores, "bad_scores": bad_scores, "n_pairs": len(pairs)}


def _resolve_director_relevance_floor():
    """Порог режиссёра — из калибровки моделей зрения (поле director_floor,
    на шкале прежнего ансамбля). Нет калибровки или разделение было
    ненадёжным (калибровка пишет null) — DIRECTOR_RELEVANCE_FALLBACK: порог
    только для отчёта (DIRECTOR_RELEVANCE_MISSES), победителя не меняет."""
    try:
        v = vision_model.threshold("director_floor")
    except vision_model.NotCalibrated:
        v = None
    return v if isinstance(v, (int, float)) else DIRECTOR_RELEVANCE_FALLBACK


def role_shot_size_bonus(role, shot_size):
    return ROLE_SHOT_SIZE_BONUS.get((role, shot_size), 0.0)


def arc_stage_shot_bonus(arc_stage, shot_size):
    if arc_stage is None:
        return 0.0
    return ARC_STAGE_SHOT_SIZE_BONUS.get((arc_stage, shot_size), 0.0)


def domain_match_bonus(candidate_domain, text_domain):
    if candidate_domain and text_domain and candidate_domain == text_domain:
        return DOMAIN_MATCH_BONUS
    return 0.0


def repetition_penalty(recent_semantic_tags, candidate_domain, role):
    """Штраф за повтор ТОЙ ЖЕ пары (domain, role) среди последних
    REPETITION_WINDOW уже выбранных клипов — тот же принцип скользящего
    окна anti-repeat, что уже используют recent_shot_sizes/
    recent_media_types в main(), только на семантической, а не
    геометрической/типовой оси."""
    window = recent_semantic_tags[-REPETITION_WINDOW:] if recent_semantic_tags else []
    repeats = sum(1 for d, r in window if d == candidate_domain and r == role)
    return repeats * REPETITION_PENALTY


def visual_qc_bonus(image_path):
    """Небольшой бонус за резкость/чистоту кадра — переиспользует
    scripts/visual_qc.py scorer'ы (sharpness_score/noise_score) и их общую
    подготовку кадра (_load_gray_normalized), не копирует их. Ленивый
    импорт — visual_qc.py тянется только когда Director реально считает
    extra_score (shadow/assist), не в обычном прогоне с off."""
    try:
        import visual_qc
        gray = visual_qc._load_gray_normalized(image_path)
        sharp = visual_qc.sharpness_score(gray)
        noise = visual_qc.noise_score(gray)
    except Exception:
        return 0.0
    bonus = 0.0
    if sharp is not None:
        bonus += min(sharp, 200.0) / 200.0 * VISUAL_QC_SHARPNESS_WEIGHT
    if noise is not None:
        bonus -= min(noise, 40.0) / 40.0 * VISUAL_QC_NOISE_WEIGHT
    return bonus


def _safe_shot_size(image_path):
    try:
        return pipeline_smart.estimate_shot_size(image_path)
    except Exception:
        return None


def candidate_domain_for(image_path):
    """classify_domain() на КАНДИДАТЕ (не на тексте) — вынесено отдельной
    функцией, чтобы pipeline_smart.py могло переиспользовать её же на
    финальном победителе для обновления recent_semantic_tags, не считая
    домен дважды разными путями."""
    domain, _ = lr.classify_domain(image_path)
    return domain


def compute_extra_score(image_path, role, block_text, text_domain, recent_semantic_tags, arc_stage=None,
                         own_query=None, candidate_query=None, is_opening=False, aesthetic_val=None):
    """Один float — оркестрация всех сигналов выше. role/text_domain —
    БЛОК-уровневые значения (посчитаны ОДИН РАЗ на блок вызывающим кодом в
    main(), не на каждого кандидата — functional_role() бесплатна, но
    text_domain_hint() — CLIP-вызов, пересчитывать его на каждого
    кандидата того же блока было бы лишним расходом времени).

    arc_stage — риторическая стадия ЭТОГО блока из media_plan/speech_plan.json
    (см. ARC_STAGE_SHOT_SIZE_BONUS выше), None — если эпизод без Speech
    Director (даёт нулевой бонус, поведение не меняется).

    own_query/candidate_query — см. SAME_QUERY_BONUS выше. own_query —
    запрос, назначенный ЭТОМУ блоку (queries[i] у вызывающего кода);
    candidate_query — из какого запроса пула реально пришёл ЭТОТ кандидат
    (p["_origin_query"], см. pexels_photo()/pexels_video()). Оба None у
    старых вызовов/тестов — ноль изменений поведения.

    is_opening/aesthetic_val — см. OPENING_AESTHETIC_WEIGHT ниже: та же
    проблема, что решает VLM-арбитр (open-shot критерий — эффектность, не
    точность), но БЕЗ Gemini — работает всегда, независимо от квоты/ключа/
    режима VLM_ARBITER_MODE, и именно поэтому определяет САМ base/director
    выбор (не только арбитраж поверх него)."""
    rel = sentence_relevance(image_path, block_text)
    score = (rel or 0.0) * SENTENCE_RELEVANCE_WEIGHT
    # SAME_QUERY_BONUS НЕ применяется для открывающего кадра — реальный
    # найденный вживую конфликт (29 августа, прямая проверка на реальных
    # кандидатах этого эпизода): бонус награждает буквальную точность
    # запроса, а весь смысл is_opening — НЕ награждать буквальную точность
    # ценой эффектности. Оставлять его активным здесь значило бы одной
    # рукой отталкивать эффектного кросс-опылённого кандидата, другой —
    # подталкивать буквальный. Для всех остальных слотов (is_opening=False)
    # поведение не меняется — тот самый фикс query-swap бага остаётся в силе.
    if own_query and candidate_query and candidate_query == own_query and not is_opening:
        score += SAME_QUERY_BONUS
    if is_opening and aesthetic_val is not None:
        normalized = max(0.0, min(1.5, (aesthetic_val - AESTHETIC_NORM_MIN) / AESTHETIC_NORM_RANGE))
        score += normalized * OPENING_AESTHETIC_WEIGHT

    shot_size = _safe_shot_size(image_path)
    if shot_size:
        score += role_shot_size_bonus(role, shot_size)
        score += arc_stage_shot_bonus(arc_stage, shot_size)

    candidate_domain = candidate_domain_for(image_path)
    # OPENING_DOMAIN_WEIGHT — реальный найденный вживую случай (29 августа,
    # прямая проверка на реальных кандидатах _test20s): чистая эстетика
    # (LAION) меряет ОБЩУЮ фотографическую привлекательность, не жанровую
    # уместность — яркие бирюзовые весы при хорошем свете набрали БОЛЬШЕ
    # очков (6.65), чем мрачный, драматичный натюрморт с мечом и шлемом
    # (5.61), хотя именно второй явно сильнее для военно-исторической
    # документалки. classify_domain() уже умеет узнавать жанр кадра
    # ("battle" и т.п., см. DOMAIN_PROMPTS) — обычный DOMAIN_MATCH_BONUS
    # (0.20) уже пытался это компенсировать, но проиграл гонку баллов
    # aesthetic-бонусу (0.55) и SAME_QUERY_BONUS. Для открывающего кадра
    # совпадение жанра — усиленный, самостоятельный сигнал, не смешанный с
    # обычным DOMAIN_MATCH_BONUS (тот остаётся как есть для всех
    # остальных слотов).
    if is_opening and candidate_domain and text_domain and candidate_domain == text_domain:
        score += OPENING_DOMAIN_WEIGHT
    else:
        score += domain_match_bonus(candidate_domain, text_domain)
    score -= repetition_penalty(recent_semantic_tags, candidate_domain, role)
    score += visual_qc_bonus(image_path)
    return score


# Разрешается ЗДЕСЬ, в конце файла (после sentence_relevance() и всего, что
# ей нужно) — см. _resolve_director_relevance_floor() выше. pipeline_smart.py
# обращается к visual_director.DIRECTOR_RELEVANCE_FLOOR как к обычному
# атрибуту модуля (4 места) — этот вызов остаётся плейн-присваиванием,
# ноль изменений на стороне вызывающего кода. Модуль импортируется
# ТОЛЬКО когда VISUAL_DIRECTOR_MODE != "off" (см. pipeline_smart.py) —
# стоимость калибровки/её кэш-чтения не платит никто, кому эта фича не нужна.
DIRECTOR_RELEVANCE_FLOOR = _resolve_director_relevance_floor()
