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

ВАЖНО, честно: lookbook начинается ПУСТЫМ (assets/lookbook/lookbook.json —
канал не выпустил ни одного эпизода, реального "одобренного" эталона взять
неоткуда). При пустом lookbook (или LOOK_MANAGEMENT_ENABLED=0, дефолт)
look_correction_filter() возвращает None для ЛЮБОГО кадра — система
физически не может повлиять на рендер, пока лениво (см.
test_look_correction_filter_noop_when_lookbook_empty в tests/test_look_reference.py
— главный тест всего модуля). Пополнять lookbook — только вручную, через
scripts/lookbook_add.py, реально одобренными кадрами канала (не выдуманными
"на глаз" эталонами — та же ошибка, которую критиковали в _scene_bias, тут
повторять её нельзя).

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
import json
import math
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

from PIL import Image  # noqa: E402

LOOKBOOK_PATH = os.path.join(REPO_ROOT, "assets", "lookbook", "lookbook.json")
LOOK_MANAGEMENT_ENABLED = os.environ.get("LOOK_MANAGEMENT_ENABLED", "0") == "1"

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

MAX_MATCH_DISTANCE = 35.0
MAX_STRENGTH = 0.35   # тот же порядок, что AUTO_WB_MAX_STRENGTH — частичная
                        # коррекция, никогда полный снап к эталону
GAIN_CLAMP = (0.5, 1.8)   # тот же диапазон и та же причина, что auto_wb_params
                            # (ffmpeg тихо роняет colorchannelmixer за [-2,2])
DELTA_STEP_CLAMP = (6.0, 4.0, 4.0)   # (dL, da, db) — максимальный шаг EMA за
                                       # один клип, тот же принцип клэмпа
                                       # шага, что max(-0.035,min(0.035,...))
                                       # у brightness_bias
EMA_KEEP = 0.6   # доля предыдущего состояния — тот же коэффициент, что
                  # luma_ema = luma_ema*0.6 + luma*0.4

DEFAULT_MAX_CORRECTION_DELTA = (10.0, 8.0, 8.0)   # (dL, da, db), если у
                                                     # эталона поле не задано
OVERSATURATION_FACTOR = 1.5   # рост хромы (sqrt(a^2+b^2)) свыше этого — hard-fail

# --- Веса для расстояния "кадр -> эталон" (см. find_reference). Lab — три
# оси в сопоставимом масштабе (L: 0..100, a/b обычно в пределах ±50) — сама
# по себе почти готовая метрика. brightness/contrast (0..1) и temperature
# (warm_bias, -1..1) — вспомогательные оси в другом масштабе, коэффициенты
# ниже приводят их к сопоставимому с Lab порядку величины. Не откалибровано
# (см. DOMAIN_MARGIN выше) — разумная стартовая пропорция. ---
AUX_BRIGHTNESS_WEIGHT = 40.0
AUX_CONTRAST_WEIGHT = 20.0
AUX_TEMP_WEIGHT = 15.0


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
    недоступен или разрыв меньше DOMAIN_MARGIN."""
    scores = _domain_scores(image_path)
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

def load_lookbook():
    if not os.path.exists(LOOKBOOK_PATH):
        return {"references": []}
    try:
        return json.load(open(LOOKBOOK_PATH, encoding="utf-8"))
    except Exception:
        return {"references": []}


def look_correction_filter(image_path, levels, wb, has_face, prev_delta=None):
    """Возвращает (filter_str_or_None, report_entry, new_delta_state).
    filter_str — фрагмент ffmpeg-графа (colorchannelmixer), приклеивается
    ПОСЛЕ film_look(...) вызывающим кодом, никогда не заменяет его. None —
    коррекция не применяется (см. report_entry['decision'] за причиной) —
    вызывающий код просто не добавляет ничего, оригинальный рендер film_look
    не затронут ни в одном случае None."""
    if not LOOK_MANAGEMENT_ENABLED:
        return None, {"decision": "skipped_disabled"}, prev_delta
    lookbook = load_lookbook()
    if not lookbook.get("references"):
        return None, {"decision": "skipped_empty_lookbook"}, prev_delta
    if has_face:
        return None, {"decision": "skipped_face_detected"}, prev_delta
    if wb is None or levels is None or levels[0] is None:
        return None, {"decision": "skipped_no_signal"}, prev_delta

    domain, margin = classify_domain(image_path)
    if domain is None:
        return None, {"decision": "skipped_low_domain_confidence"}, prev_delta

    frame_lab = _srgb_to_lab(wb)
    warm_bias, _, _ = pipeline_smart._scene_bias(levels, wb)
    frame_brightness = (levels[0] + levels[1]) / 2.0
    frame_contrast = levels[1] - levels[0]

    reference, confidence = find_reference(domain, frame_lab, frame_brightness, frame_contrast, warm_bias, lookbook)
    if reference is None:
        return None, {"decision": "skipped_no_reference_match", "domain": domain, "domain_margin": margin}, prev_delta

    gains, qc, new_delta = compute_correction(wb, frame_lab, reference, confidence, prev_delta)
    report = {
        "domain": domain, "domain_margin": round(margin, 4),
        "reference_id": reference.get("id"), "confidence": round(confidence, 4),
        "applied_delta": [round(x, 3) for x in new_delta] if new_delta else None,
        "qc": qc,
    }
    if gains is None:
        report["decision"] = qc["decision"]
        return None, report, new_delta
    report["decision"] = "applied"
    filt = f"colorchannelmixer=rr={gains[0]:.4f}:gg={gains[1]:.4f}:bb={gains[2]:.4f}"
    return filt, report, new_delta
