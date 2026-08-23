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
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

VIDEO_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
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

DIRECTOR_MIN_POOL = 3   # расширяет good_needed в pexels_photo() при shadow/assist —
                          # разумное некалиброванное значение (см. докстринг модуля
                          # про cost-tradeoff), без него Директору часто нечего ранжировать

SENTENCE_RELEVANCE_WEIGHT = 1.0   # доминирующий член — тот же порядок величины
                                    # (реалистичный диапазон ~0.15-0.35), что уже
                                    # использует is_relevant_candidate()

# РЕАЛЬНЫЙ, эмпирически подтверждённый баг (не гипотеза): sentence_relevance()
# передавала СЫРОЙ русский block_text прямо в pipeline_smart.clip_relevance(),
# который использует ОДНОЯЗЫЧНУЮ (английскую) openai/clip-vit-base-patch32.
# Прямое измерение на реальных фото канала + реальных фразах сценария (см.
# коммит): та же модель на русском тексте даёт кластер 0.17-0.21 НЕЗАВИСИМО
# от содержимого картинки (меч/пицца/катана — разница в пределах шума) — то
# есть доминирующий сигнал Директора (SENTENCE_RELEVANCE_WEIGHT=1.0, больше
# любого другого бонуса) на практике был ПУСТЫМ для ВСЕГО этого (русскоязычного)
# канала, хотя VISUAL_DIRECTOR_MODE=assist уже включён. Мультиязычная модель
# (sentence-transformers/clip-ViT-B-32-multilingual-v1, та же архитектура CLIP,
# дообученная на параллельных текстах 50+ языков) — калибровка на 16
# реальных парах текст/фото (не только оружейная лексика канала — специально
# добавлены НИКАК не связанные с нишей темы: собака, гора, кофе, велосипед,
# дождь, гитара, книга, мост, кот, ракета — проверка, что сигнал работает
# по смыслу ЛЮБОГО текста, не только знакомой военно-исторической лексики):
# top-1 точность 12/16 (75%), top-3 точность 15/16 (94%). Показательно: ВСЕ
# 10 разнотемных (не оружейных) пар угаданы с первой попытки — 4 промаха
# сосредоточены ИСКЛЮЧИТЕЛЬНО в кластере визуально похожих мелких предметов
# (меч/молоток/молоко/скальпель/ноутбук/весы — фото "рука держит
# металлический предмет крупным планом" объективно похожи друг на друга и
# семантически, и по кадру). 100% top-1 недостижимо для этого класса
# моделей в принципе (открытый словарь, zero-shot) — не цель, честный
# потолок. top-3=94% — реалистичная оценка полезности сигнала, потому что
# в проде это РЕРАНЖИРОВАНИЕ уже отфильтрованного query-based гейтом пула
# кандидатов (см. is_relevant_candidate()/pexels_photo()), а не выбор из
# всех возможных фото — там, где query уже сузил пул до тематически близких
# кандидатов, sentence_relevance() дотягивает точный выбор внутри него.
#
# КРИТИЧНО: это ОТДЕЛЬНАЯ модель/эмбеддинг-пространство от
# pipeline_smart.get_clip_model() — та остаётся ТРОНУТОЙ НЕ БЫЛА и обслуживает
# все уже откалиброванные пороги (CLIP_RELEVANCE_THRESHOLD, RISKY_QUERY_MARGIN,
# PARTICLE_SCORE_THRESHOLD, VISUAL_DOMAIN_GUARDS) — смена модели там задним
# числом обесценила бы ВСЕ эти калибровки (разные модели дают разные шкалы
# скоров). Используется ТОЛЬКО здесь, для sentence_relevance() — единственного
# места, что и раньше получало сырой текст без английского посредника-query.
SENTENCE_RELEVANCE_MODEL_VERSION = "multilingual-clip-v1"   # см. cache_signature() —
                                                               # смена модели меняет
                                                               # шкалу скоров, должна
                                                               # инвалидировать кэш

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


_multilingual_clip_text_model = None
_multilingual_clip_image_model = None
_MULTILINGUAL_CLIP_BROKEN = False


def _get_multilingual_clip_models():
    """Ленивая загрузка (см. SENTENCE_RELEVANCE_MODEL_VERSION выше) — два
    отдельных энкодера (текст/картинка), как того требует sentence-
    transformers для clip-ViT-B-32-multilingual-v1 (текстовая ветка
    дообучена отдельно, картиночная — оригинальный CLIP ViT-B/32,
    совместимое эмбеддинг-пространство, но НЕ тот же объект модели, что
    pipeline_smart.get_clip_model() — см. предупреждение выше про разные
    шкалы скоров)."""
    global _multilingual_clip_text_model, _multilingual_clip_image_model
    if _multilingual_clip_text_model is None:
        from sentence_transformers import SentenceTransformer
        _multilingual_clip_text_model = SentenceTransformer(
            "sentence-transformers/clip-ViT-B-32-multilingual-v1")
        _multilingual_clip_image_model = SentenceTransformer("clip-ViT-B-32")
    return _multilingual_clip_text_model, _multilingual_clip_image_model


def sentence_relevance(image_path, block_text):
    """Косинусная близость картинки и ПОЛНОГО текста блока (русского, как
    он есть в сценарии — см. SENTENCE_RELEVANCE_MODEL_VERSION выше) через
    мультиязычную CLIP-модель. None при отсутствии текста/недоступности
    модели (тот же fail-open, что и везде в пайплайне) — вызывающий код
    (compute_extra_score) тогда просто не добавляет этот бонус, поведение
    как до фичи."""
    global _MULTILINGUAL_CLIP_BROKEN
    if not block_text or _MULTILINGUAL_CLIP_BROKEN:
        return None
    try:
        from PIL import Image as PILImage
        text_model, image_model = _get_multilingual_clip_models()
        img = PILImage.open(image_path).convert("RGB")
        img_emb = image_model.encode([img], convert_to_numpy=True)
        txt_emb = text_model.encode([block_text], convert_to_numpy=True)
        import numpy as np
        img_n = img_emb[0] / np.linalg.norm(img_emb[0])
        txt_n = txt_emb[0] / np.linalg.norm(txt_emb[0])
        return float(np.dot(img_n, txt_n))
    except ImportError:
        _MULTILINGUAL_CLIP_BROKEN = True
        return None
    except Exception:
        return None


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


def compute_extra_score(image_path, role, block_text, text_domain, recent_semantic_tags, arc_stage=None):
    """Один float — оркестрация всех сигналов выше. role/text_domain —
    БЛОК-уровневые значения (посчитаны ОДИН РАЗ на блок вызывающим кодом в
    main(), не на каждого кандидата — functional_role() бесплатна, но
    text_domain_hint() — CLIP-вызов, пересчитывать его на каждого
    кандидата того же блока было бы лишним расходом времени).

    arc_stage — риторическая стадия ЭТОГО блока из media_plan/speech_plan.json
    (см. ARC_STAGE_SHOT_SIZE_BONUS выше), None — если эпизод без Speech
    Director (даёт нулевой бонус, поведение не меняется)."""
    rel = sentence_relevance(image_path, block_text)
    score = (rel or 0.0) * SENTENCE_RELEVANCE_WEIGHT

    shot_size = _safe_shot_size(image_path)
    if shot_size:
        score += role_shot_size_bonus(role, shot_size)
        score += arc_stage_shot_bonus(arc_stage, shot_size)

    candidate_domain = candidate_domain_for(image_path)
    score += domain_match_bonus(candidate_domain, text_domain)
    score -= repetition_penalty(recent_semantic_tags, candidate_domain, role)
    score += visual_qc_bonus(image_path)
    return score
