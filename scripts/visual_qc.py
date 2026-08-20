#!/usr/bin/env python3
"""Visual QC — локальный пре-рендер этап контроля качества медиа (ЧАСТЬ 13,
новый шаг между fetch медиа и сборкой). Единый вектор качества на КАЖДЫЙ
файл в media/: резкость, шум, экспозиция, движение (видео), частицы, дубли,
релевантность запросу, эстетика — вместо разрозненных гейтов, каждый из
которых уже существует в pipeline_smart.py, но считается отдельно и только
в момент выбора кандидата ИЗ API-выдачи (pexels_photo/pexels_video). Здесь —
второй, отдельный проход по уже СКАЧАННЫМ файлам media/, ДО сборки:
отклонённый (не AI!) кандидат автоматически заменяется переподбором из тех
же источников, что stock_fetch_multisource.py, а не маскируется фильтром на
этапе рендера.

Локально по построению: CLIP/aesthetic — та же уже загруженная в
pipeline_smart.py локальная модель (torch/transformers, опционально — без
них соответствующие оси просто выключены, откат безопасный, как и везде в
пайплайне); резкость/шум — OpenCV (опционально, numpy-фоллбэк без него) +
FFmpeg-пробники кадров, оба уже обязательные зависимости пайплайна.
Единственный сетевой трафик — тот же самый Pexels/Pixabay/Unsplash, что уже
дёргает stock_fetch_multisource.py при переподборе отклонённого кандидата
(не новый источник, не платный API, не скрытый запрос).

Ни один слот не остаётся ПУСТЫМ из-за QC: если ни один кандидат не прошёл
жёсткие пороги за QC_MAX_REFETCH_TRIES попыток, слот получает ЛУЧШИЙ из
виденных, честно помеченный в отчёте "ниже порога, проверить глазами" —
ровно то, для чего в CLAUDE.md уже есть обязательный ручной Шаг 7.5 (контакт-
лист + просмотр глазами) — QC не отменяет этот шаг, а сужает то, что на нём
реально нужно смотреть.

ПОРОГИ (SHARPNESS_REJECT/NOISE_REJECT/AESTHETIC_BORDERLINE ниже) — разумные
стартовые точки, не откалиброванные на реальных кадрах этого канала (в
отличие от CLIP_RELEVANCE_THRESHOLD/PARTICLE_SCORE_THRESHOLD в
pipeline_smart.py, которые калибровались вживую на конкретных эпизодах) —
после первого реального прогона на живом эпизоде их стоит свериться с
media_plan/visual_qc_report.json и поправить, если QC систематически режет
хорошие кадры или пропускает явный брак.

Usage: python scripts/visual_qc.py <video_dir>
Вход: media/ (уже заполнена stock_fetch_multisource.py и/или AI-картинками)
      + media_plan/slots_master_index.txt (idx|текст) + media_plan/themes.json.
Выход: media_plan/visual_qc_report.json + правки в media/ (замена
       отклонённых СТОКОВЫХ файлов; AI-картинки НИКОГДА не трогаются —
       курируются человеком по ЧАСТИ 14, QC их только оценивает)."""
import glob
import json
import os
import subprocess
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image as PILImage, ImageFilter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

VIDEO_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

# pipeline_smart читает VIDEO_FOLDER из sys.argv[1] НА ИМПОРТЕ (ищет
# audio.mp3, грузит themes.json) — подменяем argv на путь к видео ДО
# импорта (тот же приём, что уже использует tests/test_parse.py), иначе
# find_audio() падает на os.listdir() чужой/несуществующей директории.
_saved_argv = sys.argv
sys.argv = ["pipeline_smart.py", VIDEO_DIR]
import pipeline_smart  # noqa: E402
sys.argv = _saved_argv

import assemble                         # noqa: E402  (resolve_slot — та же слотовая схема имён)
import stock_fetch_multisource as sfm   # noqa: E402


# --- Пороги (см. предупреждение в докстринге модуля выше) ---
SHARPNESS_PROBE_MAX_SIDE = 720   # нормализация размера кадра перед оценкой резкости/шума —
                                  # иначе дисперсия Лапласиана растёт с разрешением исходника,
                                  # не с реальной резкостью
SHARPNESS_REJECT = 25.0          # дисперсия Лапласиана ниже — заметно размыто
NOISE_REJECT = 26.0              # RMS-остаток после гауссова размытия (0..255 шкала) —
                                  # щедрый порог нарочно: ложный reject дороже пропуска
LUMA_BLACK_REJECT = 0.03         # средняя яркость кадра ниже — почти чисто чёрный (частый Pexels-брак)
LUMA_WHITE_REJECT = 0.97         # ...выше — почти чисто белый
AESTHETIC_BORDERLINE = 3.3       # ниже — не reject, но флаг "проверить глазами" (диапазон
                                  # aesthetic_score() на реальных фото ~3-7, см. её докстринг)
QC_MAX_REFETCH_TRIES = 3         # сколько раз реально переподбираем отклонённого кандидата


def _extract_probe_frame(path, is_video, at=0.5):
    """Кадр-пробник: сам файл для фото, один кадр через ffmpeg для видео —
    тот же приём (-ss 0.5, не 0 — реальный сток иногда начинается с чёрного
    лидер-кадра), что уже использует measure_luma()/measure_levels() в
    pipeline_smart.py. Возвращает (путь_к_кадру, нужно_ли_удалить) или
    (None, False) при сбое чтения/декода."""
    if not is_video:
        return (path, False) if os.path.exists(path) else (None, False)
    tmp = path + "._qc_probe.jpg"
    try:
        r = subprocess.run(["ffmpeg", "-y", "-ss", f"{at:.2f}", "-i", path,
                             "-frames:v", "1", "-q:v", "3", tmp],
                            capture_output=True, timeout=20)
        if r.returncode != 0 or not os.path.exists(tmp):
            return None, False
        return tmp, True
    except Exception:
        return None, False


def _load_gray_normalized(image_path):
    """Ч/б кадр, уменьшенный до SHARPNESS_PROBE_MAX_SIDE по большей стороне —
    общая подготовка для sharpness_score()/noise_score(), чтобы не грузить и
    не ресайзить один и тот же файл дважды из qc_verdict()."""
    img = PILImage.open(image_path).convert("L")
    w, h = img.size
    scale = SHARPNESS_PROBE_MAX_SIDE / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), PILImage.LANCZOS)
    return img


def sharpness_score(gray_img):
    """Резкость — дисперсия Лапласиана (стандартный дешёвый blur-детектор):
    резкий кадр даёт много высокочастотной энергии по краям, размытый —
    мало. cv2 — предпочтительный путь; без него — тот же численный приём
    через уже существующий pipeline_smart._convolve2d_same (numpy-only,
    та же функция, что уже использует spectral_saliency_anchor() в
    pipeline_smart.py). None при недоступности numpy — вызывающий код тогда
    просто не гейтит по этой оси (тот же принцип безопасного отката, что у
    остальных опциональных скореров пайплайна)."""
    try:
        if CV2_AVAILABLE:
            arr = np.asarray(gray_img, dtype=np.uint8)
            return float(cv2.Laplacian(arr, cv2.CV_64F).var())
        if np is None:
            return None
        arr = np.asarray(gray_img, dtype=np.float64)
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        lap = pipeline_smart._convolve2d_same(arr, kernel)
        return float(lap.var())
    except Exception:
        return None


NOISE_SMOOTH_PERCENTILE = 40   # нижние N% кадра по градиенту считаются "гладкой" зоной


def noise_score(gray_img):
    """Шум — RMS-остаток между кадром и его гауссовым размытием, ТОЛЬКО в
    ГЛАДКИХ (низкий градиент) областях кадра — иначе резкая, но ЧИСТАЯ мелкая
    текстура (кольчуга, гравировка, дощатый стол — то, чем полон именно этот
    канал) сама по себе даёт большой остаток после блюра просто за счёт
    детализации, и метрика путает "много деталей" с "много шума". Реальный
    сенсорный шум/сжатие-артефакты видны И в гладких зонах (небо, фон,
    расфокус) — там остаток от blur должен быть околонулевым при чистом
    источнике, любой заметный остаток там — почти наверняка шум, не деталь.
    None при недоступности numpy или если в кадре нет достаточно гладкой
    зоны для надёжной оценки (кадр целиком текстура/края — не судим о шуме
    по такому кадру, оставляем это sharpness_score)."""
    if np is None:
        return None
    try:
        arr = np.asarray(gray_img, dtype=np.float64)
        blurred = gray_img.filter(ImageFilter.GaussianBlur(radius=2.5))
        b = np.asarray(blurred, dtype=np.float64)
        residual = arr - b
        gy, gx = np.gradient(arr)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        thresh = np.percentile(grad_mag, NOISE_SMOOTH_PERCENTILE)
        smooth_mask = grad_mag <= thresh
        if smooth_mask.sum() < 50:
            return None
        return float(np.sqrt(np.mean(residual[smooth_mask] ** 2)))
    except Exception:
        return None


def _composite_score(sharp, noise, relevance, aesthetic):
    """Взвешенная сумма для ранжирования ОТКЛОНЁННЫХ кандидатов между собой
    (см. main(): если ни один кандидат слота не прошёл жёсткие пороги,
    остаётся лучший по этой сумме) — НЕ порог accept/reject сам по себе,
    только тай-брейк среди уже проваленных, чтобы "лучший из плохих" был
    реально лучшим, а не первым по счёту."""
    score = 0.0
    if relevance is not None:
        score += relevance * 3.0             # семантическое соответствие теме — самый весомый сигнал
    if aesthetic is not None:
        score += (aesthetic / 10.0) * 1.5
    if sharp is not None:
        score += min(sharp, 200.0) / 200.0
    if noise is not None:
        score -= min(noise, 40.0) / 40.0 * 0.5
    return score


def qc_verdict(path, is_video, query, accepted_hashes, slot_label):
    """Единый вектор качества + вердикт pass/borderline/reject для ОДНОГО
    файла. accepted_hashes — {ahash: slot_label} уже ПРИНЯТЫХ в этом
    эпизоде кадров (мутируется на месте только при pass/borderline — reject
    не занимает место в пуле дублей, иначе один плохой кандидат мог бы
    заблокировать хороший дубль-чек для всех последующих слотов)."""
    info = {"path": path, "is_video": is_video, "query": query, "slot": slot_label}

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        info.update(verdict="reject", reasons=["файл отсутствует/пуст"], score=-1.0)
        return info

    frame, cleanup = _extract_probe_frame(path, is_video)
    if frame is None:
        info.update(verdict="reject", reasons=["не удалось декодировать файл"], score=-1.0)
        return info

    try:
        gray = _load_gray_normalized(frame)
        sharp = sharpness_score(gray)
        noise = noise_score(gray)
        relevance = pipeline_smart.clip_relevance(frame, query) if query else None
        aesthetic = pipeline_smart.aesthetic_score(frame)
        h = pipeline_smart.ahash(frame)
    finally:
        if cleanup and frame and os.path.exists(frame):
            os.remove(frame)

    # Эти три уже умеют работать с видео нативно (multi-sample для видео,
    # не один кадр-пробник) — зовём их на ОРИГИНАЛЬНОМ пути, не на frame.
    luma = pipeline_smart.measure_luma(path, is_video=is_video)
    levels = pipeline_smart.measure_levels(path, is_video=is_video)
    motion = pipeline_smart.measure_motion(path, 0.3, 3.0) if is_video else None
    particle = pipeline_smart.measure_particle_score(path, is_video=is_video)

    info.update(luma=luma, levels=levels, sharpness=sharp, noise=noise,
                motion=motion, particle=particle, relevance=relevance,
                aesthetic=aesthetic, ahash=h)

    reasons = []
    hard_reject = False

    if luma is not None and (luma < LUMA_BLACK_REJECT or luma > LUMA_WHITE_REJECT):
        reasons.append(f"кадр почти чисто чёрный/белый (luma={luma:.3f})")
        hard_reject = True
    if levels is not None and levels[0] is not None:
        black_in, white_in = levels
        if white_in - black_in < pipeline_smart.AUTO_LEVELS_MIN_RANGE:
            reasons.append(f"плоская гистограмма (диапазон {white_in - black_in:.3f})")
            hard_reject = True
    if sharp is not None and sharp < SHARPNESS_REJECT:
        reasons.append(f"размыто (резкость={sharp:.1f})")
        hard_reject = True
    if noise is not None and noise > NOISE_REJECT:
        reasons.append(f"избыточный шум/артефакты (noise={noise:.1f})")
        hard_reject = True
    if relevance is not None:
        if relevance < pipeline_smart.CLIP_RELEVANCE_THRESHOLD:
            reasons.append(f"не по теме запроса (relevance={relevance:.3f})")
            hard_reject = True
        elif pipeline_smart.is_risky_query(query):
            neg = pipeline_smart.clip_relevance(path if not is_video else path, pipeline_smart.NEGATIVE_ANCHOR_PROMPT)
            if neg is not None and (relevance - neg) < pipeline_smart.RISKY_QUERY_MARGIN:
                reasons.append("собирательный запрос без доминирующего предмета в кадре")
                hard_reject = True
    if h is not None and not hard_reject:
        for other_hash, other_slot in accepted_hashes.items():
            if pipeline_smart.hamming(h, other_hash) <= pipeline_smart.PHOTO_DEDUP_HAMMING:
                reasons.append(f"похоже на уже принятый кадр (слот {other_slot})")
                hard_reject = True
                break

    borderline = False
    if not hard_reject:
        if aesthetic is not None and aesthetic < AESTHETIC_BORDERLINE:
            reasons.append(f"низкая эстетическая оценка ({aesthetic:.2f})")
            borderline = True
        if particle is not None and particle >= pipeline_smart.PARTICLE_SCORE_THRESHOLD:
            reasons.append("в кадре собственные частицы (снег/пыль/боке) — учтено в силе зерна при рендере")

    verdict = "reject" if hard_reject else ("borderline" if borderline else "pass")
    info.update(verdict=verdict, reasons=reasons,
                score=_composite_score(sharp, noise, relevance, aesthetic))
    if verdict != "reject" and h is not None:
        accepted_hashes[h] = slot_label
    return info


def load_slot_queries(video_dir):
    """idx -> (сырой_текст, англ.запрос) из media_plan/slots_master_index.txt
    + themes.json — та же логика (build_query с ротацией list-значений
    channel_themes.json по счётчику), что уже использует
    stock_fetch_multisource.py при первичном фетче: слот проверяется на
    релевантность ТОМУ ЖЕ запросу, которым его искали, не пересчитанному
    заново с другим счётчиком ротации."""
    idx_path = os.path.join(video_dir, "media_plan", "slots_master_index.txt")
    if not os.path.exists(idx_path):
        return {}
    themes = sfm.load_themes(video_dir)
    keyword_counts = {}
    out = {}
    for line in open(idx_path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or "|" not in line:
            continue
        i, text = line.split("|", 1)
        try:
            idx = int(i)
        except ValueError:
            continue
        out[idx] = (text, sfm.build_query(text, themes, keyword_counts))
    return out


def _refetch_stock(media_dir, idx, prefer_video, query, attempt, out_video, out_photo):
    """Один переподбор в ВРЕМЕННЫЕ файлы (out_video/out_photo) — ротирует
    ПРОВАЙДЕРА по attempt (Pexels/Pixabay/Unsplash), не просто повторяет тот
    же запрос тому же источнику: для одного текстового запроса поиск
    детерминированно отдаёт один и тот же топ-результат, голый повтор без
    ротации источника заново скачал бы ТОТ ЖЕ уже отклонённый кандидат.
    Фоллбэк видео->фото — та же логика, что уже в stock_fetch_multisource.main().
    Возвращает (kind, path) или (None, None), если ни один источник не дал результата."""
    if prefer_video:
        if sfm.try_sources(sfm.VIDEO_SOURCES, attempt, query, out_video):
            return "video", out_video
        if sfm.try_sources(sfm.PHOTO_SOURCES, attempt, query, out_photo):
            return "photo", out_photo
    else:
        if sfm.try_sources(sfm.PHOTO_SOURCES, attempt, query, out_photo):
            return "photo", out_photo
    return None, None


def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def process_slot(media_dir, idx, query, accepted_hashes):
    """QC одного слота с автопереподбором. Возвращает запись отчёта.
    ГАРАНТИЯ: слот никогда не остаётся без файла из-за QC — при исчерпании
    QC_MAX_REFETCH_TRIES остаётся ЛУЧШИЙ из виденных кандидатов, даже если
    он ниже жёстких порогов (честно помечен verdict="accepted_below_threshold"
    в отчёте — не reject, слот идёт в сборку, но флаг виден для ручного
    Шага 7.5)."""
    slot_label = f"{idx:03d}"
    kind, path, source = assemble.resolve_slot(media_dir, idx)

    if source == "ai":
        # AI-картинки курируются человеком по ЧАСТИ 14 — QC оценивает и
        # докладывает, но НИКОГДА не трогает файл и не переподбирает замену.
        info = qc_verdict(path, is_video=(kind == "video"), query=query,
                           accepted_hashes=accepted_hashes, slot_label=slot_label)
        info["source"] = "ai"
        info["auto_replaced"] = False
        return info

    prefer_video = (idx % 2 == 0) if kind is None else (kind == "video")

    best = None
    if path is not None:
        best = qc_verdict(path, is_video=(kind == "video"), query=query,
                           accepted_hashes={}, slot_label=slot_label)
        best["_kept_path"] = path

    tries = 0
    out_video_tmp = os.path.join(media_dir, f".qc_tmp_{idx:03d}.mp4")
    out_photo_tmp = os.path.join(media_dir, f".qc_tmp_{idx:03d}.jpg")
    while (best is None or best["verdict"] == "reject") and tries < QC_MAX_REFETCH_TRIES:
        _cleanup(out_video_tmp, out_photo_tmp)
        new_kind, new_path = _refetch_stock(media_dir, idx, prefer_video, query, tries,
                                             out_video_tmp, out_photo_tmp)
        tries += 1
        if new_path is None:
            continue
        candidate = qc_verdict(new_path, is_video=(new_kind == "video"), query=query,
                                accepted_hashes={}, slot_label=slot_label)
        candidate["_kept_path"] = new_path
        candidate["_kind"] = new_kind
        if best is None or candidate["score"] > best["score"]:
            if best is not None and best.get("_kept_path") not in (path, None):
                _cleanup(best["_kept_path"])
            best = candidate
        else:
            _cleanup(new_path)

    auto_replaced = False
    if best is not None and best.get("_kept_path") != path:
        # Лучший кандидат — не исходный файл: переносим на постоянное
        # слотовое имя, чистим старый (если был) и оставшиеся temp-файлы.
        final_ext = ".mp4" if best.get("_kind", kind) == "video" else ".jpg"
        final_path = os.path.join(media_dir, f"{idx:03d}_stock{'_video' if final_ext == '.mp4' else ''}{final_ext}")
        if path and os.path.exists(path) and path != final_path:
            _cleanup(path)
        if os.path.exists(final_path) and final_path != best["_kept_path"]:
            _cleanup(final_path)
        if best["_kept_path"] != final_path:
            os.replace(best["_kept_path"], final_path)
        best["path"] = final_path
        auto_replaced = True
    _cleanup(out_video_tmp, out_photo_tmp)

    if best is None:
        return {"slot": slot_label, "verdict": "missing", "reasons": ["нет кандидата ни от одного источника"],
                "source": "stock", "auto_replaced": False, "query": query}

    if best["verdict"] == "reject":
        best["verdict"] = "accepted_below_threshold"
        best["reasons"] = best.get("reasons", []) + [
            f"после {tries} переподбора(ов) — принят лучший из виденных, проверь глазами (Шаг 7.5)"]

    h = best.get("ahash")
    if h is not None:
        accepted_hashes[h] = slot_label
    best["source"] = "stock"
    best["auto_replaced"] = auto_replaced
    best.pop("_kept_path", None)
    best.pop("_kind", None)
    return best


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    media_dir = os.path.join(video_dir, "media")
    if not os.path.isdir(media_dir):
        print(f"media/ не найдена: {media_dir}")
        return 1

    queries = load_slot_queries(video_dir)
    if not queries:
        nums = set()
        import re as _re
        for f in glob.glob(os.path.join(media_dir, "*")):
            m = _re.match(r'(\d+)_', os.path.basename(f))
            if m:
                nums.add(int(m.group(1)))
        queries = {n: ("", sfm.DEFAULT_QUERY) for n in sorted(nums)}
        if queries:
            print(f"media_plan/slots_master_index.txt не найден — QC по {len(queries)} "
                  f"файлам media/ без проверки релевантности запросу (текст неизвестен).")

    if not queries:
        print("Нет слотов для проверки (ни slots_master_index.txt, ни файлов в media/)")
        return 1

    accepted_hashes = {}
    report = []
    print(f"Visual QC: {len(queries)} слотов"
          f"{' (CLIP/aesthetic выключены — нет torch/transformers)' if not pipeline_smart.CLIP_ENABLED else ''}")
    for idx in sorted(queries):
        _, query = queries[idx]
        entry = process_slot(media_dir, idx, query, accepted_hashes)
        report.append(entry)
        tag = {"pass": "OK", "borderline": "ГРАНИЦА", "accepted_below_threshold": "НИЖЕ ПОРОГА",
               "missing": "НЕТ ФАЙЛА", "reject": "REJECT"}.get(entry["verdict"], entry["verdict"].upper())
        extra = f" <- переподобран" if entry.get("auto_replaced") else ""
        print(f"[{idx:03d}] {tag}{extra}" + (f" | {'; '.join(entry['reasons'])}" if entry.get("reasons") else ""))

    counts = {}
    for e in report:
        counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1
    plan_dir = os.path.join(video_dir, "media_plan")
    os.makedirs(plan_dir, exist_ok=True)
    report_path = os.path.join(plan_dir, "visual_qc_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"counts": counts, "slots": report}, f, ensure_ascii=False, indent=2)

    # "reject" в ИТОГОВОМ отчёте бывает только у AI-картинок (см. process_slot —
    # стоковый слот всегда либо резолвится в pass/borderline, либо честно
    # переклассифицируется в accepted_below_threshold/missing, но никогда не
    # остаётся "reject" молча) — то есть каждый reject здесь означает
    # конкретную AI-картинку, которую надо перегенерировать, не проблему,
    # которую что-то уже само починило. Считаем его наравне с остальными
    # "требует внимания" исходами, иначе такие случаи тонут в сводке.
    unresolved = counts.get("missing", 0) + counts.get("accepted_below_threshold", 0) + counts.get("reject", 0)
    print(f"\nVisual QC готово: {report_path}")
    print(f"  OK={counts.get('pass', 0)} ГРАНИЦА={counts.get('borderline', 0)} "
          f"НИЖЕ_ПОРОГА={counts.get('accepted_below_threshold', 0)} "
          f"НЕТ_ФАЙЛА={counts.get('missing', 0)} REJECT(AI)={counts.get('reject', 0)}")
    if unresolved:
        print(f"  {unresolved} слот(ов) требуют проверки глазами (Шаг 7.5) перед сборкой.")
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
