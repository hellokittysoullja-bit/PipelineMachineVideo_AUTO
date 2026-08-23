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

# РЕАЛЬНЫЙ, эмпирически подтверждённый баг (не гипотеза, история в git-логе):
# sentence_relevance() передавала СЫРОЙ русский block_text прямо в
# pipeline_smart.clip_relevance(), который использует ОДНОЯЗЫЧНУЮ (английскую)
# openai/clip-vit-base-patch32 — та же модель на русском тексте даёт кластер
# 0.17-0.21 независимо от содержимого картинки (доминирующий сигнал Директора
# был фактически пустым для всего этого русскоязычного канала).
#
# Первый фикс — sentence-transformers/clip-ViT-B-32-multilingual-v1 (только
# ТЕКСТОВАЯ башня дообучена на 50+ языков, картиночная — не изменена от
# оригинального английского CLIP): калибровка на 16 реальных парах
# текст/фото (10 намеренно НЕ военно-исторических тем — собака, гора, кофе,
# велосипед, дождь, гитара, книга, мост, кот, ракета — проверка обобщения на
# ЛЮБОЙ текст) дала top-1 75%, top-3 94%; все 4 промаха — в одном кластере
# визуально похожих мелких предметов (меч/молоток/молоко/скальпель/ноутбук/
# весы, "рука держит металлический предмет крупным планом").
#
# По прямому запросу пользователя дожать точность — прогнал ТОТ ЖЕ тест на
# google/siglip2-base-patch16-256 (SigLIP2 — sigmoid-loss контрастная модель
# от Google, В ОТЛИЧИЕ от clip-ViT-B-32-multilingual-v1 вся, включая
# картиночную башню, обучена на многоязычных данных, 109 языков, WebLI, ещё
# и на разрешении 256×256 против 224×224 у обычного CLIP): **top-1 16/16
# (100%), top-3 16/16 (100%)** на ТОЙ ЖЕ выборке — включая ВЕСЬ кластер
# мелких предметов, где предыдущая модель проваливалась (меч/молоток/молоко/
# скальпель/ноутбук/весы — все верно). Нативная поддержка в transformers
# (без remote-кода — в отличие от проверенного, но несовместимого с текущей
# версией transformers Jina CLIP v2), без новых зависимостей сверх уже
# установленных torch/transformers/PIL. 100% на 16 парах — не гарантия 100%
# всегда (открытый словарь, zero-shot, честный потолок остаётся, см. выше),
# но однозначно, воспроизводимо лучше предыдущей модели на РЕАЛЬНЫХ данных —
# заменяю по факту измерения, не по названию бренда.
#
# КРИТИЧНО: это ОТДЕЛЬНАЯ модель/эмбеддинг-пространство от
# pipeline_smart.get_clip_model() — та остаётся ТРОНУТОЙ НЕ БЫЛА и обслуживает
# все уже откалиброванные пороги (CLIP_RELEVANCE_THRESHOLD, RISKY_QUERY_MARGIN,
# PARTICLE_SCORE_THRESHOLD, VISUAL_DOMAIN_GUARDS) — смена модели там задним
# числом обесценила бы ВСЕ эти калибровки (разные модели дают разные шкалы
# скоров). Используется ТОЛЬКО здесь, для sentence_relevance() — единственного
# места, что и раньше получало сырой текст без английского посредника-query.
SIGLIP2_MODEL_NAME = "google/siglip2-base-patch16-256"
SIGLIP2_MAX_TEXT_LENGTH = 64   # ровно max_position_embeddings текстовой башни
                                 # этой модели (см. config) — не произвольное число

# --- Ensemble SigLIP2 + Jina CLIP v2 (по прямому запросу пользователя —
# "объединение сильнейших сторон обоих") ---
#
# 113-позиционный независимый бенчмарк (3 части: A — 95 разнотемных длинных
# фраз, B — 8 идиом/метафор, C — 10 контрастных по настроению фото) дал
# SigLIP2 solo top-1=92/113(81%) top-3=105/113(93%), Jina CLIP v2 (ONNX,
# квантованная — полноточная версия падает в NaN на CPU независимо от
# dtype, см. git-лог) solo top-1=90/113(80%) top-3=103/113(91%). Сетка весов
# 0.2..0.8 (не только цифра внешнего аудита 0.65/0.35 — проверено ЧЕСТНО,
# своим прогоном) дала максимум на SigLIP2=0.7/Jina=0.3:
# top-1=96/113(85%) top-3=107/113(95%).
#
# ЧЕСТНО, прямо по требованию пользователя "только плюс, без минусов":
# буквально нулевой регрессии на уровне ОТДЕЛЬНЫХ решений НЕ существует ни
# для какого статистического смешивания — прямая построчная сверка дала 5
# исправлений SigLIP2-ошибок и 1 РЕГРЕССИЮ (lightning.jpg -> storm.jpg,
# тематически смежная ошибка, не нарушение культуры/эпохи/объекта). Проверил
# гипотезу "включать Jina только когда SigLIP2 сам не уверен" (margin top1-
# top2) — НЕ подтвердилась: у регрессии margin БОЛЬШЕ, чем у 4 из 5
# исправлений, чистой границы нет, порог пришлось бы подгонять под 6 точек
# (не калибровка, а переобучение на шуме) — сознательно НЕ стал городить
# такой gate. Итог, принятый пользователем: 112/113 тот же результат или
# лучше, единственная регрессия низкой значимости, все существующие гейты
# (анахронизм/relevance/риск) остаются нетронутыми — ensemble только
# переранжирует УЖЕ прошедших их кандидатов.
#
# Нормализация: raw-скоры SigLIP2 (sigmoid-loss) и Jina (softmax-CLIP) НЕ
# сопоставимы по шкале напрямую (в проде sentence_relevance() сравнивает
# ОДНУ картинку с ОДНИМ текстом за вызов — см. _score_and_pick() в
# pipeline_smart.py, там НЕТ пула кандидатов, чтобы нормализовать по строке,
# как в бенчмарке) — вместо per-slot min-max используется ФИКСИРОВАННЫЙ
# z-score перенос шкалы Jina на шкалу SigLIP2, константы посчитаны ОДИН раз
# по статистике всех пар текст/картинка того же 113-позиционного бенчмарка
# (не с потолка). Проверено на том же бенчмарке — с фиксированным rescale
# (без per-slot нормализации, ТА ЖЕ схема, что пойдёт в прод) итог почти не
# отличается от per-row minmax: top-1=95-96/113(84-85%) top-3=106-108/113
# (94-96%) в зависимости от точной точки веса — тот же порядок улучшения,
# не артефакт нормализации бенчмарка.
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

ENSEMBLE_WEIGHT_SIGLIP2 = 0.7   # эмпирический максимум top-1 на 113-позиционном
ENSEMBLE_WEIGHT_JINA = 0.3       # бенчмарке (сетка 0.2..0.8), не цифра аудита

# Константы z-score rescale — посчитаны ОДИН раз на всех парах текст/картинка
# 113-позиционного бенчмарка (95+16+10 картинок x соответствующие тексты,
# полная матрица, не только диагональ). Пересчитывать только вместе с новой
# калибровкой (см. докстринг выше) — НЕ трогать по наитию, это не magic
# number, а измеренное среднее/стд обеих шкал скоров.
SIGLIP2_SCORE_MEAN = 0.0000
SIGLIP2_SCORE_STD = 0.0378
JINA_SCORE_MEAN = 0.1597
JINA_SCORE_STD = 0.0512

SENTENCE_RELEVANCE_MODEL_VERSION = (
    f"siglip2-base-patch16-256+jina-clip-v2-onnx-quant"
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
        _siglip2_model = AutoModel.from_pretrained(SIGLIP2_MODEL_NAME)
        _siglip2_model.eval()
        _siglip2_processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_NAME)
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
    файлу переживает обновление кэша/переезд на другую машину."""
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
        _jina_tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL_REPO)
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
