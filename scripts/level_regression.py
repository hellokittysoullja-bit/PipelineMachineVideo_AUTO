# -*- coding: utf-8 -*-
"""Обратный замер уровней по ОТРЕНДЕРЕННОМУ звуку.

Зачем это существует. Проект трижды пострадал от одного класса ошибок:
музыкальная подложка (задумано 16 LU, в публикации 27), акцент кульминации
(+7 дБ против замысла), объектный слой (точка +4 дБ, подзвучник -18 дБ).
Ни один тест не поймал ни одну — все они проверяют, что код делает то, что
в коде написано, а ошибка живёт в ЗВУКЕ.

Отдельно про ложную уверенность: если усиление считается как
`цель - замер`, то повторный замер того же ассета вернёт цель по построению
арифметики. Это проверка вычитания. Настоящая проверка — снять уровень с
файла, прошедшего ВСЮ цепочку (дакинг, loudnorm, лимитер), потому что
уровни разъезжаются именно там: лимитер на -14 LUFS ловит ровно транзиенты.

Метод вычитания. Сцена рендерится ДВАЖДЫ — со слоем и без него. Оба
прогона детерминированы и выровнены по времени, поэтому разность по
сэмплам и есть фактический вклад слоя ПОСЛЕ всей обработки. Иначе из
готового микса отдельную дорожку не достать.
"""
import os
import re
import subprocess
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def decode_mono(path, sr=48000):
    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1",
                        "-ar", str(sr), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32).astype("float64")


def loudness_of_samples(samples, sr=48000, mode="M"):
    """Громкость массива: пишем во временный wav и меряем тем же ebur128,
    что и весь остальной звук проекта — своя формула K-взвешивания
    разошлась бы с ffmpeg на десятые, и сравнивать было бы нечего."""
    import numpy as np
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak <= 0:
            return None
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(sr),
             "-ac", "1", "-i", "-", "-ac", "2", tmp], stdin=subprocess.PIPE)
        p.communicate(samples.astype("float32").tobytes())
        r = _run(["ffmpeg", "-v", "info", "-i", tmp, "-af", "ebur128=peak=true",
                  "-f", "null", "-"])
        if mode == "I":
            m = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", r.stderr or "")
            return float(m[-1]) if m else None
        vals = [float(x) for x in re.findall(r"M:\s*(-?[\d.]+)", r.stderr or "")]
        vals = [v for v in vals if v > -70.0]
        return max(vals) if vals else None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def layer_contribution(with_path, without_path, sr=48000):
    """Фактический вклад слоя = разность двух прогонов по сэмплам."""
    import numpy as np
    a, b = decode_mono(with_path, sr), decode_mono(without_path, sr)
    n = min(a.size, b.size)
    if n == 0:
        return None
    return a[:n] - b[:n]


def window(samples, t0, t1, sr=48000):
    import numpy as np
    i0, i1 = max(0, int(t0 * sr)), min(samples.size, int(t1 * sr))
    return samples[i0:i1] if i1 > i0 else np.array([], dtype="float64")


def limiter_reduction_db(before_path, after_path, t0, t1, sr=48000):
    """Насколько лимитер прижал микс в окне вокруг кюя.

    Транзиент — ровно то, на что срабатывает мастер-лимитер, и прижимает он
    ВЕСЬ микс, включая голос: на слух это «голос дёрнулся на ударе».
    Изолированным замером ассета это не ловится в принципе.
    """
    import numpy as np
    a, b = decode_mono(before_path, sr), decode_mono(after_path, sr)
    n = min(a.size, b.size)
    wa, wb = window(a[:n], t0, t1, sr), window(b[:n], t0, t1, sr)
    if wa.size == 0 or wb.size == 0:
        return None
    ra = float(np.sqrt((wa ** 2).mean()))
    rb = float(np.sqrt((wb ** 2).mean()))
    if ra <= 0 or rb <= 0:
        return None
    return 20.0 * np.log10(ra / rb)


# ------------------------------------------------------------- сцена
FIXTURE_DIR = os.path.join(os.path.dirname(SCRIPTS), "tests", "fixtures", "level_scene")
SCENE_VOICE = os.path.join(FIXTURE_DIR, "voice.flac")
SCENE_BED = os.path.join(FIXTURE_DIR, "obj_bed.flac")
SCENE_POINT = os.path.join(FIXTURE_DIR, "obj_point.flac")
SCENE_VOICE_LUFS = -16.0        # реально измеренная громкость голоса эпизода
SCENE_DUR = 20.0
# Кюи стоят в РАЗНЫХ местах сцены: точка — в паузе (по правилу слоя она
# только туда и ставится), подзвучник — под речью.
# Кюи РАЗВЕДЕНЫ так, чтобы окно замера одного не захватывало другой.
# Первая версия ставила точку на 14.0 внутрь пролёта фона (9.0-15.0), и
# максимальная мгновенная громкость в окне фона бралась от УДАРА — оба
# слоя показывали одинаковые 27.9 LU. Дефект был в замере, не в уровнях,
# и поймал его именно прогон через цепочку, а не рассуждение.
SCENE_POINT_AT = (3.0, 17.0)
SCENE_BED_AT = 8.0
SCENE_BED_MEASURE = (8.6, 13.4)   # чистый участок фона, без соседних кюев


def build_fixture(force=False):
    """Сгенерировать файлы сцены. Запускается ОДИН раз, результат лежит в
    репозитории: генератор со временем поплывёт (версия ffmpeg, параметры),
    а эталон обязан быть неизменным, иначе регрессия сравнивает с движущейся
    целью."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    if not force and all(os.path.exists(p) for p in
                         (SCENE_VOICE, SCENE_BED, SCENE_POINT)):
        return False
    # «речь»: полоса речи с паузами — важны именно паузы, точка ставится в них
    speech = ("anoisesrc=d=%.1f:c=pink:r=48000,highpass=f=120,lowpass=f=6000,"
              "tremolo=f=0.7:d=0.9" % SCENE_DUR)
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", speech,
          "-af", f"loudnorm=I={SCENE_VOICE_LUFS}:TP=-1.5",
          "-ar", "48000", "-ac", "2", "-sample_fmt", "s16", SCENE_VOICE])
    # подзвучник: стационарная текстура, пик как у библиотеки (-12 dBFS)
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
          "-i", "anoisesrc=d=10:c=brown:r=48000",
          "-af", "highpass=f=200,lowpass=f=5000,volume=-12dB",
          "-ar", "48000", "-ac", "2", "-sample_fmt", "s16", SCENE_BED])
    # точка: короткий транзиент, пик как у библиотеки (-10 dBFS)
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
          "-i", "sine=f=220:d=0.6",
          "-af", "afade=t=out:st=0.02:d=0.55,volume=-10dB",
          "-ar", "48000", "-ac", "2", "-sample_fmt", "s16", SCENE_POINT])
    return True


def render_scene(out_path, with_objects=True, tmp_dir=None):
    """Прогнать сцену через НАСТОЯЩУЮ цепочку: объектный слой -> loudnorm
    -> лимитер. Именно там уровни и разъезжаются."""
    import pipeline_smart as ps
    import sfx_plan
    tmp_dir = tmp_dir or FIXTURE_DIR
    cues = []
    if with_objects:
        voice_lufs = ps.measure_integrated_lufs(SCENE_VOICE)
        for t in SCENE_POINT_AT:
            g, _ = ps.object_gain_db(SCENE_POINT, sfx_plan.OBJECT_CLASS_POINT, voice_lufs)
            cues.append({"kind": "object", "cls": sfx_plan.OBJECT_CLASS_POINT,
                         "name": "point", "time": t, "asset": SCENE_POINT, "gain_db": g})
        g, _ = ps.object_gain_db(SCENE_BED, sfx_plan.OBJECT_CLASS_BED, voice_lufs)
        cues.append({"kind": "object", "cls": sfx_plan.OBJECT_CLASS_BED,
                     "name": "bed", "time": SCENE_BED_AT, "asset": SCENE_BED,
                     "gain_db": g, "trim_sec": sfx_plan.OBJECT_BED_SEC,
                     "fade_in_sec": sfx_plan.OBJECT_BED_FADE_IN_SEC,
                     "fade_out_sec": sfx_plan.OBJECT_BED_FADE_OUT_SEC})
    pre = os.path.join(tmp_dir, "_pre_%s.wav" % ("with" if with_objects else "without"))
    mixed = ps.add_planned_sfx(SCENE_VOICE, cues, SCENE_DUR, pre) if cues else SCENE_VOICE
    af = ps.build_master_af(None, max(0.0, SCENE_DUR - 2.0), 0.4)
    r = _run(["ffmpeg", "-y", "-v", "error", "-i", mixed, "-af", af,
              "-t", f"{SCENE_DUR:.3f}", "-ar", "48000", "-ac", "2", out_path])
    return out_path if r.returncode == 0 else None
