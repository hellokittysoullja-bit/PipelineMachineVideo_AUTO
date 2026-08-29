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

_VISUAL_DIRECTOR_MODES = ("off", "shadow", "assist")
VISUAL_DIRECTOR_MODE = os.environ.get("VISUAL_DIRECTOR_MODE", "off").strip().lower()
if VISUAL_DIRECTOR_MODE not in _VISUAL_DIRECTOR_MODES:
    print(f"  ВНИМАНИЕ: VISUAL_DIRECTOR_MODE={VISUAL_DIRECTOR_MODE!r} не входит в "
          f"{_VISUAL_DIRECTOR_MODES} — откатываюсь на 'off'.")
    VISUAL_DIRECTOR_MODE = "off"

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
# ВТОРОЙ, ТОЖЕ РЕАЛЬНЫЙ, найденный вживую пробел (deep-audit, 28 августа,
# тот же videos/_test20s): первая калибровка порога (n=8) была на пары ИЗ
# ЭТОГО ЖЕ эпизода (мечи/рыцари) и на ensemble ДО апгрейда so400m — то есть
# сама числовая шкала score с тех пор сдвинулась (см. SIGLIP2_SCORE_MEAN/
# STD выше — они пересчитаны под so400m, а порог, стоявший поверх них,
# пересчитан не был). Живая проверка на этом же коде: EXCALIBUR-фото ПРОТИВ
# СВОЕГО ЖЕ авторского запроса ("medieval european sword blade close up" —
# почти идеальное совпадение) даёт 0.113 — уже НИЖЕ порога 0.13. Хардкод
# числа тут не чинит проблему, а откладывает её же до следующего апгрейда
# модели (это уже второй раз, когда порог тихо устарел). Порог теперь
# ВЫЧИСЛЯЕТСЯ живой калибровкой (calibrate_relevance_floor() ниже) на
# channel-agnostic паре good/bad (tests/fixtures/director_calibration/
# calibration_pairs.json — переиспользует уже лицензированные фикстуры
# tests/fixtures/golden_media/, не мечи из конкретного эпизода), кэшируется
# по подписи модели (см. _model_signature()) и КОММИТИТСЯ в
# scripts/calibration_cache/ — смена модели меняет подпись, следующий
# запуск честно пересчитывает заново, а сам пересчёт виден в git diff (не
# тихий рантайм-эффект). DIRECTOR_RELEVANCE_FALLBACK ниже — последняя
# линия защиты, если фикстур/кэша физически нет (fail-open, не крашим
# рендер из-за advisory-only порога).
DIRECTOR_RELEVANCE_FALLBACK = 0.13
DIRECTOR_RELEVANCE_MIN_MARGIN = 0.02   # ниже этого разделение good/bad на
                                         # калибровочных парах считается
                                         # ненадёжным — новый порог НЕ
                                         # принимается, остаётся прежний
                                         # (кэш или fallback), с громким
                                         # предупреждением в консоль.
CALIBRATION_PAIRS_PATH = os.path.join(
    REPO_ROOT, "tests", "fixtures", "director_calibration", "calibration_pairs.json")
CALIBRATION_CACHE_PATH = os.path.join(
    SCRIPTS_DIR, "calibration_cache", "director_relevance_floor.json")

# РЕАЛЬНЫЙ, эмпирически подтверждённый баг (не гипотеза, история в git-логе):
# sentence_relevance() передавала СЫРОЙ русский block_text прямо в
# pipeline_smart.clip_relevance(), который использует ОДНОЯЗЫЧНУЮ (английскую)
# openai/clip-vit-base-patch32 — та же модель на русском тексте даёт кластер
# 0.17-0.21 независимо от содержимого картинки (доминирующий сигнал Директора
# был фактически пустым для всего этого русскоязычного канала).
#
# Фикс — google/siglip2-base-patch16-256 (SigLIP2 — sigmoid-loss контрастная
# модель от Google, вся, включая картиночную башню, обучена на многоязычных
# данных, 109 языков, WebLI, разрешение 256×256 против 224×224 у обычного
# CLIP): калибровка на 16 реальных парах текст/фото (10 намеренно НЕ
# военно-исторических тем — собака, гора, кофе, велосипед, дождь, гитара,
# книга, мост, кот, ракета — проверка обобщения на ЛЮБОЙ текст, плюс кластер
# визуально похожих мелких предметов — меч/молоток/молоко/скальпель/ноутбук/
# весы, "рука держит металлический предмет крупным планом") дала **top-1
# 16/16 (100%), top-3 16/16 (100%)** — включая весь кластер мелких предметов.
# Нативная поддержка в transformers (без remote-кода — в отличие от
# проверенного, но несовместимого с текущей версией transformers Jina CLIP
# v2), без новых зависимостей сверх уже установленных torch/transformers/PIL.
#
# ДАЛЬНЕЙШИЙ апгрейд (по прямому запросу пользователя "сравни оба варианта,
# бери тот, что даёт лучший эффект на итоговый ролик, невзирая на сложность
# — только если эффект plus-minus одинаковый, тогда бери быстрее"):
# base-256 -> so400m-patch14-384 (крупнее модель — ~400M параметров против
# ~200M, выше разрешение 384×384 против 256×256). На ТОМ ЖЕ 113-позиционном
# бенчмарке: so400m solo top-1=100/113(88%) top-3=109/113(96%) — заметно
# выше base-256 solo (81%/93%) И выше прежнего прод-ensemble base-256+Jina
# (85%/95%). Разница НЕ "плюс-минус" (2 п.п. top-1 = 2 реальные позиции на
# 113, не шум) -> по правилу пользователя эффект решает, не сложность.
#
# ЧЕСТНАЯ цена: so400m на CPU НАМНОГО медленнее base-256 — прямой замер
# одного вызова (1 картинка + 1 текст) ~2.5-3с против долей секунды у
# base-256. Пользователь явно одобрил эту цену ради эффекта (не молчаливое
# решение) — см. коммит.
#
# КРИТИЧНО: это ОТДЕЛЬНАЯ модель/эмбеддинг-пространство от
# pipeline_smart.get_clip_model() — та остаётся ТРОНУТОЙ НЕ БЫЛА и обслуживает
# все уже откалиброванные пороги (CLIP_RELEVANCE_THRESHOLD, RISKY_QUERY_MARGIN,
# PARTICLE_SCORE_THRESHOLD, VISUAL_DOMAIN_GUARDS) — смена модели там задним
# числом обесценила бы ВСЕ эти калибровки (разные модели дают разные шкалы
# скоров). Используется ТОЛЬКО здесь, для sentence_relevance() — единственного
# места, что и раньше получало сырой текст без английского посредника-query.
SIGLIP2_MODEL_NAME = "google/siglip2-so400m-patch14-384"
SIGLIP2_MAX_TEXT_LENGTH = 64   # ровно max_position_embeddings текстовой башни
                                 # этой модели (см. config) — та же величина,
                                 # что и у base-256, не изменилась при апгрейде

# --- Ensemble SigLIP2 + Jina CLIP v2 (по прямому запросу пользователя —
# "объединение сильнейших сторон обоих") ---
#
# 113-позиционный независимый бенчмарк (3 части: A — 95 разнотемных длинных
# фраз, B — 8 идиом/метафор, C — 10 контрастных по настроению фото).
# Первая калибровка (SigLIP2-base-256 + Jina): solo top-1=81%/80%, ensemble
# на сетке весов 0.2..0.8 дал максимум на 0.7/0.3 (85%/95%) — задокументирована
# в git-логе, замена ниже её полностью перекрывает.
#
# ВТОРАЯ калибровка (SigLIP2-so400m-384 + Jina, ДЕЙСТВУЮЩАЯ): so400m solo
# 88%/96% сам по себе уже лучше прежнего прод-ensemble. Сетка весов
# 0.3..0.9 поверх so400m дала максимум на so400m=0.5/Jina=0.5:
# **top-1=102/113(90%) top-3=110/113(97%)** — Part C (тон/настроение)
# доходит до 100/100%. Проверено на ФИКСИРОВАННОМ z-score rescale (та же
# схема, что пойдёт в прод, не per-row minmax бенчмарка) — совпадает с
# результатом на минмаксе, не артефакт нормализации.
#
# ЧЕСТНО, прямо по более раннему требованию пользователя "только плюс, без
# минусов" (применимо и здесь): буквально нулевой регрессии на уровне
# ОТДЕЛЬНЫХ решений НЕ существует ни для какого статистического смешивания
# — это математический факт, не недоработка (см. построчную сверку
# base-256+Jina в предыдущей калибровке для примера метода; для so400m+Jina
# отдельная построчная сверка не переделывалась — агрегат уже настолько
# выше прежнего прод-варианта, что дальнейшая экономия на 1-2 находках
# несущественна). Все существующие гейты (анахронизм/relevance/риск)
# остаются нетронутыми — ensemble только переранжирует УЖЕ прошедших их
# кандидатов.
#
# Нормализация: raw-скоры SigLIP2 (sigmoid-loss) и Jina (softmax-CLIP) НЕ
# сопоставимы по шкале напрямую (в проде sentence_relevance() сравнивает
# ОДНУ картинку с ОДНИМ текстом за вызов — см. _score_and_pick() в
# pipeline_smart.py, там НЕТ пула кандидатов, чтобы нормализовать по строке,
# как в бенчмарке) — вместо per-slot min-max используется ФИКСИРОВАННЫЙ
# z-score перенос шкалы Jina на шкалу SigLIP2, константы посчитаны ОДИН раз
# по статистике всех пар текст/картинка того же 113-позиционного бенчмарка
# (не с потолка, и ПЕРЕСЧИТАНЫ под so400m — шкала сырых скоров у so400m
# отличается от base-256, старые константы были бы неверны).
JINA_MODEL_REPO = "jinaai/jina-clip-v2"
JINA_ONNX_FILENAME = "onnx/model_quantized.onnx"   # публикуемый quantized ONNX,
                                                      # полноточная PyTorch-версия
                                                      # даёт NaN на CPU (см. выше)
JINA_IMG_SIZE = 512
JINA_IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)   # дословно из
JINA_IMG_STD = (0.26862954, 0.26130258, 0.27577711)   # preprocessor_config.json
                                                         # репозитория — ручная
                                                         # реализация, БЕЗ
                                                         # trust_remote_code=True
                                                         # (тот же принцип, что уже
                                                         # применён к AutoModel выше)
JINA_TEXT_MAX_LENGTH = 77

ENSEMBLE_WEIGHT_SIGLIP2 = 0.5   # эмпирический максимум top-1 на 113-позиционном
ENSEMBLE_WEIGHT_JINA = 0.5       # бенчмарке (сетка 0.3..0.9 поверх so400m)

# Константы z-score rescale — посчитаны ОДИН раз на всех парах текст/картинка
# 113-позиционного бенчмарка (95+16+10 картинок x соответствующие тексты,
# полная матрица, не только диагональ), ПЕРЕСЧИТАНЫ под so400m (шкала сырых
# скоров у so400m отличается от base-256). Пересчитывать только вместе с
# новой калибровкой (см. докстринг выше) — НЕ трогать по наитию, это не
# magic number, а измеренное среднее/стд обеих шкал скоров.
SIGLIP2_SCORE_MEAN = -0.0261
SIGLIP2_SCORE_STD = 0.0434
JINA_SCORE_MEAN = 0.1597
JINA_SCORE_STD = 0.0512

SENTENCE_RELEVANCE_MODEL_VERSION = (
    f"siglip2-so400m-patch14-384+jina-clip-v2-onnx-quant"
    f"_w{ENSEMBLE_WEIGHT_SIGLIP2}-{ENSEMBLE_WEIGHT_JINA}"
)   # см. cache_signature() — смена модели/весов меняет шкалу скоров,
    # должна инвалидировать кэш

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


_siglip2_model = None
_siglip2_processor = None
_SIGLIP2_BROKEN = False


def _get_siglip2_model():
    """Ленивая загрузка (см. SENTENCE_RELEVANCE_MODEL_VERSION/SIGLIP2_MODEL_NAME
    выше) — модель+процессор нативно поддержаны transformers (AutoModel/
    AutoProcessor), без remote-кода стороннего репозитория (в отличие от
    Jina CLIP v2 ниже — её PyTorch-путь так же отклонён, используется
    ТОЛЬКО официальный quantized ONNX-экспорт) и без новых зависимостей
    поверх уже установленных torch/transformers."""
    global _siglip2_model, _siglip2_processor
    if _siglip2_model is None:
        from transformers import AutoModel, AutoProcessor
        # trust_remote_code=False — та же защита, что уже стоит у Jina
        # (_get_jina_session() ниже, см. её докстринг про пойманный вживую
        # интерактивный prompt "Do you wish to run the custom code? [y/N]",
        # зависавший на чтении stdin в headless-процессе). SigLIP2 — нативная
        # transformers-модель, remote-код и так не нужен; явный False убирает
        # саму возможность промпта, не только его последствия.
        _siglip2_model = AutoModel.from_pretrained(SIGLIP2_MODEL_NAME, trust_remote_code=False)
        _siglip2_model.eval()
        _siglip2_processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_NAME, trust_remote_code=False)
    return _siglip2_model, _siglip2_processor


def _siglip2_relevance(image_path, block_text):
    """Сырой (не rescale-нутый) косинус SigLIP2 — вынесено из бывшей
    sentence_relevance() без изменения логики, чтобы ensemble ниже мог
    использовать её как один из двух компонентов."""
    global _SIGLIP2_BROKEN
    if _SIGLIP2_BROKEN:
        return None
    try:
        import torch
        from PIL import Image as PILImage
        model, processor = _get_siglip2_model()
        img = PILImage.open(image_path).convert("RGB")
        with torch.no_grad():
            img_inputs = processor(images=[img], return_tensors="pt")
            img_out = model.get_image_features(**img_inputs)
            img_emb = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            txt_inputs = processor(text=[block_text], padding="max_length",
                                    max_length=SIGLIP2_MAX_TEXT_LENGTH, return_tensors="pt")
            txt_out = model.get_text_features(**txt_inputs)
            txt_emb = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

            sim = (txt_emb @ img_emb.T)[0][0]
        return float(sim)
    except ImportError:
        _SIGLIP2_BROKEN = True
        return None
    except Exception:
        return None


_jina_session = None
_jina_tokenizer = None
_JINA_BROKEN = False


def _jina_preprocess_image(pil_img):
    """Ручная реализация preprocessor_config.json репозитория (resize_mode=
    shortest, size=512, bicubic, center-crop 512x512, mean/std выше) — БЕЗ
    trust_remote_code=True. Тот же обоснованный отказ от custom-кода
    репозитория, что уже применён к AutoModel/AutoImageProcessor Jina CLIP
    v2 (см. git-лог) — стандартная OpenCLIP-схема (shortest-side resize +
    center crop), параметры дословно из конфига, не угаданы."""
    import numpy as np
    from PIL import Image as PILImage
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = JINA_IMG_SIZE / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), PILImage.BICUBIC)
    left = (new_w - JINA_IMG_SIZE) // 2
    top = (new_h - JINA_IMG_SIZE) // 2
    img = img.crop((left, top, left + JINA_IMG_SIZE, top + JINA_IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(JINA_IMG_MEAN, dtype=np.float32)) / np.array(JINA_IMG_STD, dtype=np.float32)
    return arr.transpose(2, 0, 1)   # CHW


def _get_jina_session():
    """Ленивая загрузка ТОЛЬКО quantized ONNX (см. докстринг константы
    JINA_ONNX_FILENAME выше — полноточная PyTorch-версия даёт NaN на CPU
    независимо от dtype, воспроизведено и задокументировано в git-логе).
    hf_hub_download (не хардкод локального snapshot-пути с хэшем) — путь к
    файлу переживает обновление кэша/переезд на другую машину.

    РЕАЛЬНЫЙ БАГ, пойманный живым прогоном (не гипотеза): без явного
    `trust_remote_code=False` AutoTokenizer.from_pretrained() у новых версий
    transformers может уйти в интерактивный prompt ("Do wish to run the
    custom code? [y/N]") при разрешении конфига репозитория — в трёх
    прогонах подряд повёл себя по-разному (тихо прошёл дважды, завис на
    чтении stdin на третий). Токенизатор Jina — штатный XLMRobertaTokenizer,
    НИКАКОГО remote-кода реально не требует (проверено — trust_remote_code=
    False загружает его штатно), поэтому явный False убирает и промпт, и
    зависимость поведения от того, что конкретно у процесса на stdin —
    вместо неопределённого "иногда виснет" получаем детерминированный
    успех или чистое исключение (уходит в тот же except в _jina_relevance(),
    fail-open работает как задумано, а не блокируется молча)."""
    global _jina_session, _jina_tokenizer
    if _jina_session is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        onnx_path = hf_hub_download(repo_id=JINA_MODEL_REPO, filename=JINA_ONNX_FILENAME)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4   # эмпирически: заметно быстрее single-thread,
                                        # без OOM при batch=1 (прод — один вызов =
                                        # одна картинка+один текст, не батч 95, см.
                                        # находку про батч-95-OOM в git-логе)
        _jina_session = ort.InferenceSession(onnx_path, sess_options=so,
                                              providers=["CPUExecutionProvider"])
        _jina_tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL_REPO,
                                                         trust_remote_code=False)
    return _jina_session, _jina_tokenizer


def _jina_relevance(image_path, block_text):
    """Сырой (не rescale-нутый) косинус Jina CLIP v2 — batch=1 (прод вызывает
    по одной картинке за раз, см. _score_and_pick() в pipeline_smart.py),
    поэтому найденный на бенчмарке OOM (батч 95, см. git-лог) здесь
    структурно недостижим. fail-open — та же дисциплина, что и SigLIP2
    выше."""
    global _JINA_BROKEN
    if _JINA_BROKEN:
        return None
    try:
        import numpy as np
        from PIL import Image as PILImage
        sess, tokenizer = _get_jina_session()
        img = PILImage.open(image_path).convert("RGB")
        pixel_values = _jina_preprocess_image(img)[None, ...].astype(np.float32)
        enc = tokenizer([block_text], padding=True, truncation=True,
                         max_length=JINA_TEXT_MAX_LENGTH, return_tensors="np")
        input_ids = enc["input_ids"].astype(np.int64)
        img_out = sess.run(["l2norm_image_embeddings"],
                            {"input_ids": np.zeros((1, 1), dtype=np.int64),
                             "pixel_values": pixel_values})[0]
        txt_out = sess.run(["l2norm_text_embeddings"],
                            {"input_ids": input_ids,
                             "pixel_values": np.zeros((1, 3, JINA_IMG_SIZE, JINA_IMG_SIZE),
                                                       dtype=np.float32)})[0]
        sim = float((txt_out @ img_out.T)[0][0])
        return sim
    except ImportError:
        _JINA_BROKEN = True
        return None
    except Exception:
        return None


def text_text_similarity(texts_a, texts_b):
    """Матрица косинусных близостей ТЕКСТ-ТЕКСТ (len(a) x len(b)) через
    мультиязычный текстовый энкодер Jina CLIP v2. None на любой ошибке
    (fail-open, как и весь остальной опциональный слой этого модуля).

    ЗАЧЕМ ОТДЕЛЬНО ОТ sentence_relevance(): та сравнивает КАРТИНКУ с
    текстом (image-text) — принципиально более слабый и шумный сигнал на
    абстрактных фразах (прямое измерение на реальном эпизоде: скоры всех
    кандидатов легли в 0.03-0.11, то есть ранжирование по ним было
    фактически случайным). Здесь обе стороны — ТЕКСТ: русская фраза
    сценария против английского авторского запроса, который буквально
    ОПИСЫВАЕТ желаемый кадр. Прямое измерение на том же реальном эпизоде
    (videos/_test20s, 8 авторских запросов x 6 блоков хука, не гипотеза):
    «Пятнадцать килограммов.» -> weighing scale metal object (0.707),
    «Так говорят кино, видеоигры и школьные учебники» -> dark cinema movie
    theatre screen (0.639), «...сколько весил настоящий боевой меч» ->
    knight armor holding sword two hands (0.674), «Герой на экране заносит
    клинок... вместе с конём» -> warrior on horseback with sword (0.766).
    Каждый визуальный блок получил СВОЙ смысловой запрос с большим отрывом,
    непредметные связки («Готов спорить, что да») закономерно дали низкий
    максимум — то есть сигнал не только точный, но и честно сообщает о
    собственной неуверенности."""
    if not texts_a or not texts_b:
        return None
    global _JINA_BROKEN
    if _JINA_BROKEN:
        return None
    try:
        import numpy as np
        sess, tokenizer = _get_jina_session()

        def _emb(texts):
            enc = tokenizer(list(texts), padding=True, truncation=True,
                             max_length=JINA_TEXT_MAX_LENGTH, return_tensors="np")
            ids = enc["input_ids"].astype(np.int64)
            n = ids.shape[0]
            return sess.run(["l2norm_text_embeddings"],
                             {"input_ids": ids,
                              "pixel_values": np.zeros((n, 3, JINA_IMG_SIZE, JINA_IMG_SIZE),
                                                        dtype=np.float32)})[0]

        return (_emb(texts_a) @ _emb(texts_b).T).tolist()
    except ImportError:
        _JINA_BROKEN = True
        return None
    except Exception:
        return None


def _rescale_jina_to_siglip2_scale(jina_raw):
    """Фиксированный z-score перенос шкалы (константы — см. докстринг
    SIGLIP2_SCORE_MEAN выше, измерены на 113-позиционном бенчмарке, не с
    потолка)."""
    z = (jina_raw - JINA_SCORE_MEAN) / JINA_SCORE_STD
    return z * SIGLIP2_SCORE_STD + SIGLIP2_SCORE_MEAN


def sentence_relevance(image_path, block_text):
    """Косинусная близость картинки и ПОЛНОГО текста блока (русского, как
    он есть в сценарии) — ensemble SigLIP2+Jina CLIP v2 (см. докстринг
    ENSEMBLE_WEIGHT_SIGLIP2/ENSEMBLE_WEIGHT_JINA выше для полной калибровки
    и честного разбора trade-off). None при отсутствии текста (тот же
    fail-open, что и везде в пайплайне) — вызывающий код (compute_extra_score)
    тогда просто не добавляет этот бонус, поведение как до фичи.

    Jina недоступна (нет onnxruntime/huggingface_hub, сеть недоступна,
    ошибка любого рода) -> ПАДАЕТ на SigLIP2 solo (тот же результат, что
    был ДО этой фичи) — ensemble только ДОБАВЛЯЕТ сигнал, никогда не
    убирает базовый. SigLIP2 недоступна -> None целиком (как и раньше)."""
    if not block_text:
        return None
    siglip2_raw = _siglip2_relevance(image_path, block_text)
    if siglip2_raw is None:
        return None
    jina_raw = _jina_relevance(image_path, block_text)
    if jina_raw is None:
        return siglip2_raw   # fail-open: SigLIP2-only, byte-for-byte старое поведение
    jina_rescaled = _rescale_jina_to_siglip2_scale(jina_raw)
    return ENSEMBLE_WEIGHT_SIGLIP2 * siglip2_raw + ENSEMBLE_WEIGHT_JINA * jina_rescaled


def _relevance_model_signature():
    """Отпечаток всего, что влияет на СЫРУЮ шкалу sentence_relevance() —
    имя модели, ensemble-веса, z-rescale константы. Меняется -> старый
    закэшированный DIRECTOR_RELEVANCE_FLOOR больше не доверенный (шкала
    сдвинулась), см. докстринг DIRECTOR_RELEVANCE_FALLBACK выше. Тот же
    принцип, что уже CANDIDATE_GATE_RULES_VERSION в pipeline_smart.py
    использует для инвалидации кэша кандидатов при смене правил гейта."""
    parts = "|".join([
        SIGLIP2_MODEL_NAME, JINA_MODEL_REPO, JINA_ONNX_FILENAME,
        f"{ENSEMBLE_WEIGHT_SIGLIP2}", f"{ENSEMBLE_WEIGHT_JINA}",
        f"{SIGLIP2_SCORE_MEAN}", f"{SIGLIP2_SCORE_STD}",
    ])
    return hashlib.md5(parts.encode()).hexdigest()[:16]


def _load_calibration_pairs():
    """Пары image+caption+label из CALIBRATION_PAIRS_PATH. None, если файла
    нет/битый JSON — fail-open, вызывающий код откатывается на кэш/fallback,
    не крашится (тот же принцип, что и у всего остального в этом файле)."""
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


def calibrate_relevance_floor():
    """Живая калибровка DIRECTOR_RELEVANCE_FLOOR на CALIBRATION_PAIRS_PATH —
    та же ручная методика (min(good) с запасом выше max(bad)), что раньше
    делалась один раз глазами и записывалась числом в комментарий; здесь
    выполняется кодом, поэтому происходит заново при каждой смене модели
    (см. _relevance_model_signature()), а не только когда кто-то вспомнит
    руками пересчитать.

    Возвращает dict {"floor", "margin", "good_scores", "bad_scores",
    "model_signature", ...} или None при любом сбое (нет фикстур, модель
    недоступна, sentence_relevance() падает на всех парах) — ПОЛНОСТЬЮ
    fail-open, вызывающий код откатывается на кэш/DIRECTOR_RELEVANCE_FALLBACK.

    САНИТИ-ГЕЙТ: если разделение good/bad меньше DIRECTOR_RELEVANCE_MIN_
    MARGIN, результат всё равно возвращается (для видимости в логе), но
    caller обязан не принимать его как новый порог — см. _resolve_director_
    relevance_floor(). Это и есть защита от "автоматизация просто быстрее
    ошибается": плохое разделение на новой модели не должно тихо стать
    новым порогом без явного сигнала, что что-то не так."""
    pairs = _load_calibration_pairs()
    if not pairs:
        return None
    good_scores, bad_scores = [], []
    try:
        for image_path, caption, label in pairs:
            score = sentence_relevance(image_path, caption)
            if score is None:
                return None   # модель недоступна на этом прогоне — не считаем частичный результат
            (good_scores if label == "good" else bad_scores).append(score)
    except Exception:
        return None
    if not good_scores or not bad_scores:
        return None
    min_good, max_bad = min(good_scores), max(bad_scores)
    margin = min_good - max_bad
    # Порог — с запасом ниже кластера верных пар, выше кластера промахов же
    # (тот же асимметричный принцип риска, что у RISKY_QUERY_MARGIN в
    # pipeline_smart.py: ложный "miss" в отчёте дешевле, чем ложное
    # молчание о реальной проблеме) — середина зазора, а не его край.
    floor = max_bad + margin / 2.0
    return {
        "floor": floor, "margin": margin,
        "min_good": min_good, "max_bad": max_bad,
        "good_scores": good_scores, "bad_scores": bad_scores,
        "n_pairs": len(pairs),
        "model_signature": _relevance_model_signature(),
    }


def _resolve_director_relevance_floor():
    """Порядок разрешения DIRECTOR_RELEVANCE_FLOOR при импорте модуля:
    1) закэшированное значение для ТЕКУЩЕЙ подписи модели (CALIBRATION_
       CACHE_PATH) — быстро, не гоняет модель на каждый запуск;
    2) живая калибровка, если кэш промахнулся (новая модель/первый запуск)
       — принимается ТОЛЬКО если margin >= DIRECTOR_RELEVANCE_MIN_MARGIN
       (сани-гейт), и тогда атомарно пишется в кэш (см. docstring выше про
       "смена модели видна в git diff, не тихий рантайм-эффект");
    3) DIRECTOR_RELEVANCE_FALLBACK — если live-калибровка недоступна ИЛИ
       не прошла сани-гейт: используем последний известный кэш для ЛЮБОЙ
       подписи (пусть и устаревшей — стоять на месте безопаснее, чем
       принять непроверенный сдвиг), а если кэша вообще ни для чего нет —
       жёсткий хардкод-fallback.

    Полностью fail-open на каждом шаге — advisory-only порог (см.
    DIRECTOR_RELEVANCE_MISSES в pipeline_smart.py, победителя не меняет),
    поэтому худший случай любого сбоя здесь — чуть менее точный отчёт, не
    сорванный рендер."""
    sig = _relevance_model_signature()
    cached = None
    try:
        with open(CALIBRATION_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except Exception:
        cached = None
    if cached and cached.get("model_signature") == sig:
        return cached["floor"]
    result = calibrate_relevance_floor()
    if result is not None and result["margin"] >= DIRECTOR_RELEVANCE_MIN_MARGIN:
        try:
            os.makedirs(os.path.dirname(CALIBRATION_CACHE_PATH), exist_ok=True)
            tmp = CALIBRATION_CACHE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            os.replace(tmp, CALIBRATION_CACHE_PATH)
        except Exception:
            pass   # кэш не записался -> просто пересчитаем в следующий раз, не критично
        return result["floor"]
    if result is not None:
        print(f"  ВНИМАНИЕ: авто-калибровка DIRECTOR_RELEVANCE_FLOOR дала разделение "
              f"good/bad margin={result['margin']:.3f} (< {DIRECTOR_RELEVANCE_MIN_MARGIN}) — "
              f"новый порог НЕ принят, используется прежний кэш/fallback. Модель, похоже, "
              f"плохо разделяет калибровочные пары — проверь calibration_pairs.json глазами.")
    return cached["floor"] if cached else DIRECTOR_RELEVANCE_FALLBACK


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
                         own_query=None, candidate_query=None):
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
    старых вызовов/тестов — ноль изменений поведения."""
    rel = sentence_relevance(image_path, block_text)
    score = (rel or 0.0) * SENTENCE_RELEVANCE_WEIGHT
    if own_query and candidate_query and candidate_query == own_query:
        score += SAME_QUERY_BONUS

    shot_size = _safe_shot_size(image_path)
    if shot_size:
        score += role_shot_size_bonus(role, shot_size)
        score += arc_stage_shot_bonus(arc_stage, shot_size)

    candidate_domain = candidate_domain_for(image_path)
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
