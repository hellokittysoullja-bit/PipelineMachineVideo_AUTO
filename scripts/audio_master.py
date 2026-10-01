#!/usr/bin/env python3
"""Звук ролика: обработка голоса, уровень музыки по замеру, двухпроходный
loudnorm -14 LUFS с лимитером и технический QC дорожки.

Функции перенесены из pipeline_smart.py (PipelineMachineVideo_AUTO) без
изменения логики; комментарии с замерами — оттуда же. Папка ролика здесь
не нужна: каждая функция получает пути аргументами."""
import json
import os
import re
import subprocess


def _enabled(name, default="1"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "on", "yes")


# Насколько подложка должна быть тише голоса ДО сайдчейн-дакинга (дакинг —
# страховка на пиках речи, не единственная линия обороны против "музыка
# спорит с закадром"). Это ЗАДУМАННЫЙ разрыв, а не готовое усиление:
# реальное усиление считается под конкретный ассет, см. music_bed_gain_db().
#
# РЕАЛЬНЫЙ найденный баг (07.09, измерено ebur128 на живых файлах эпизода
# 01_ves-mecha, не гипотеза). Здесь стояло глухое MUSIC_BED_GAIN_DB=-13.0, а
# комментарий рядом обосновывал его "собственными -12dBFS ассета". -12 dBFS —
# это ПИК ассета, а не его громкость: интегральная громкость
# assets/music/ambient_bed.flac — -30.0 LUFS, голос эпизода — -16.0 LUFS.
# То есть разрыв составлял 14 LU ЕЩЁ ДО применения -13 дБ, а после —
# 27 LU вместо задуманных 16. Музыку в опубликованном ролике практически
# не слышно, и никакой отчёт этого не показывал: loudnorm выравнивает микс
# ЦЕЛИКОМ, поэтому итоговые -14 LUFS выглядели идеально при неслышимой
# подложке. Число -13 было не "слишком тихо на вкус", а двойным учётом.
MUSIC_BED_GAP_LU = 16.0


# Запасное значение, если измерить громкость не удалось. Посчитано под
# ТЕКУЩИЕ ассеты канала (-30.0 LUFS против -16.0 LUFS голоса): для другого
# набора музыки оно снова будет неверным — ровно поэтому основной путь
# меряет, а не берёт константу.
MUSIC_BED_GAIN_DB = -2.0


# Клэмп на вычисленное усиление: сорванное измерение (пустой/битый ассет
# даёт -inf или -70 LUFS) иначе попросило бы +50 дБ и разнесло бы микс.
# Границы намеренно широкие — они защищают от абсурда, а не подменяют расчёт.
MUSIC_BED_GAIN_MIN_DB, MUSIC_BED_GAIN_MAX_DB = -30.0, 6.0


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
VOICE_PROCESS_ENABLED = _enabled("VOICE_PROCESS")


VOICE_HIGHPASS_HZ = 80


VOICE_EQ_WARMTH_HZ, VOICE_EQ_WARMTH_GAIN = 200, 1.5


VOICE_EQ_PRESENCE_HZ, VOICE_EQ_PRESENCE_GAIN = 3000, 2.0


VOICE_DEESS_INTENSITY = 0.3


VOICE_COMPRESS_THRESHOLD = 0.15


VOICE_COMPRESS_RATIO = 2.5


# Лимитер в самом конце мастер-цепочки — последняя ступень, которую ставит
# любой звукорежиссёр перед выдачей. Раньше её не было вообще: цепочка
# заканчивалась loudnorm + afade, и единственной защитой от пиков был
# TP-режим самого loudnorm. Он честно держит true peak, ПОКА линейного
# усиления хватает; в остальных случаях ffmpeg уходит в динамический режим
# и гарантия становится мягкой. Лимитер стоит копейки и делает потолок
# безусловным — особенно теперь, когда подложка стала на 11 дБ громче
# (см. MUSIC_BED_GAP_LU) и суммарные пики микса выросли.
# level=disabled — критично: включённый auto-level поднял бы громкость
# обратно и обнулил только что выставленный loudnorm.
MASTER_LIMITER_ENABLED = _enabled("MASTER_LIMITER")


MASTER_LIMITER_ATTACK_MS = 5


MASTER_LIMITER_RELEASE_MS = 50

# Единый источник целевой громкости — ЕДИНСТВЕННОЕ место, которое трогать
# при смене цели. Раньше -14 (было -16) была рассыпана литералом по пяти
# местам вручную (дефолт measure_loudnorm_stats, два af= в финальном
# миксе, print_format=json первого прохода, sanity-проверка готового
# файла) — РЕАЛЬНЫЙ найденный баг (02.09): при переходе -16 -> -14 пятое
# место (sanity-проверка) осталось на старом значении, и честный рендер
# на -14.3 (в пределах цели) печатал ложную тревогу "разошлось с целью
# -16". Четыре места нашли и поправили сразу, пятое — нет, потому что
# ничего в коде не требовало держать их согласованными. Константа не
# устраняет риск рассинхрона полностью (использования всё ещё нужно
# найти и подставить руками), но делает следующую смену цели ОДНИМ
# изменением вместо пяти grep'ов по литералу.
LOUDNORM_TARGET_I = -14.0


LOUDNORM_TARGET_TP = -1.5


LOUDNORM_TARGET_LRA = 11


def _audio_len_for_timeout(path):
    try:
        return get_media_duration(path)
    except Exception:
        return 600.0


def measure_loudnorm_stats(audio_path, target_i=LOUDNORM_TARGET_I,
                            target_tp=LOUDNORM_TARGET_TP, target_lra=LOUDNORM_TARGET_LRA):
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
             "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace",
            # Аудит 04.09: измерение — полный декод + ebur128 (~34x realtime на
            # 4 ядрах): при фиксированных 60с любой эпизод длиннее ~34 минут
            # молча уходил в однопроходный loudnorm ("compromise-режим").
            timeout=max(120, int(_audio_len_for_timeout(audio_path) * 0.5)))
        start, end = r.stderr.rindex("{"), r.stderr.rindex("}") + 1
        return json.loads(r.stderr[start:end])
    except Exception as e:
        print(f"  ВНИМАНИЕ: двухпроходный loudnorm не измерился ({type(e).__name__}) — "
              f"финал пойдёт через однопроходный режим (громкость внутри ролика ровнее не будет)")
        return None


def measure_integrated_lufs(audio_path):
    """Интегральная громкость дорожки в LUFS (ebur128), None при сбое.

    Отдельно от measure_loudnorm_stats(): та печатает сообщение про откат
    финального мастеринга в однопроходный режим, которое к измерению уровня
    подложки отношения не имеет и только путало бы лог. Таймаут — по той же
    формуле от длительности, что и там (аудит 04.09: фиксированные 60 с
    молча резали любой эпизод длиннее ~34 минут).
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", audio_path,
             "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=max(120, int(_audio_len_for_timeout(audio_path) * 0.5)))
        m = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", r.stderr)
        return float(m[-1]) if m else None
    except Exception:
        return None


def music_bed_gain_db(voice_path, music_path):
    """Усиление подложки под ЭТОТ голос и ЭТОТ ассет: (дБ, чем обосновано).

    Считается, а не берётся константой, потому что константа уже один раз
    молча разошлась с реальностью (см. MUSIC_BED_GAP_LU выше: задумано 16 LU,
    в опубликованном ролике вышло 27). Замена музыкального ассета или смена
    движка озвучки сдвигает обе громкости независимо — единственный способ
    удержать ЗАДУМАННЫЙ разрыв — измерить обе стороны на месте.

    Разрыв считается по интегральной громкости (LUFS), а не по пикам:
    слышимость подложки под речью определяется именно средней громкостью,
    пик у тихого дрона может быть каким угодно.
    """
    voice_lufs = measure_integrated_lufs(voice_path)
    music_lufs = measure_integrated_lufs(music_path)
    detail = {"voice_lufs": voice_lufs, "music_lufs": music_lufs,
              "target_gap_lu": MUSIC_BED_GAP_LU}
    if voice_lufs is None or music_lufs is None:
        detail.update(gain_db=MUSIC_BED_GAIN_DB, source="fallback_constant")
        print(f"  ВНИМАНИЕ: громкость дорожек не измерилась — подложка идёт по "
              f"запасной константе {MUSIC_BED_GAIN_DB} dB, задуманный разрыв "
              f"{MUSIC_BED_GAP_LU:.0f} LU НЕ гарантирован")
        return MUSIC_BED_GAIN_DB, detail
    raw = voice_lufs - MUSIC_BED_GAP_LU - music_lufs
    gain = max(MUSIC_BED_GAIN_MIN_DB, min(MUSIC_BED_GAIN_MAX_DB, raw))
    detail.update(gain_db=round(gain, 2), raw_gain_db=round(raw, 2),
                  source="measured", clamped=abs(gain - raw) > 1e-6)
    if detail["clamped"]:
        print(f"  ВНИМАНИЕ: расчётное усиление подложки {raw:+.1f} dB вышло за "
              f"[{MUSIC_BED_GAIN_MIN_DB}, {MUSIC_BED_GAIN_MAX_DB}] — обрезано до "
              f"{gain:+.1f} dB, разрыв будет не {MUSIC_BED_GAP_LU:.0f} LU")
    else:
        print(f"  Подложка: голос {voice_lufs:.1f} LUFS, музыка {music_lufs:.1f} LUFS "
              f"-> усиление {gain:+.1f} dB (разрыв {MUSIC_BED_GAP_LU:.0f} LU)")
    return gain, detail


def process_voice(voice_path, out_path):
    """A6: обработка сырой TTS-дорожки перед миксом с музыкой — срез
    суб-баса, лёгкая тональная коррекция, деэссер, мягкая компрессия
    микро-динамики фраз. VOICE_PROCESS_ENABLED=False -> просто копия
    исходника (безопасный откат, тот же принцип, что MUSIC_ENABLED)."""
    if not VOICE_PROCESS_ENABLED:
        r = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-ar", "48000", "-ac", "2", out_path],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
        return out_path if r.returncode == 0 else voice_path
    af = (
        f"highpass=f={VOICE_HIGHPASS_HZ},"
        f"equalizer=f={VOICE_EQ_WARMTH_HZ}:width_type=o:width=1.5:g={VOICE_EQ_WARMTH_GAIN},"
        f"equalizer=f={VOICE_EQ_PRESENCE_HZ}:width_type=o:width=1.2:g={VOICE_EQ_PRESENCE_GAIN},"
        f"deesser=i={VOICE_DEESS_INTENSITY}:m=0.4,"
        f"acompressor=threshold={VOICE_COMPRESS_THRESHOLD}:ratio={VOICE_COMPRESS_RATIO}:"
        f"attack=8:release=120:makeup=1.15"
    )
    r = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-af", af, "-ar", "48000", "-ac", "2", out_path],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"  ВНИМАНИЕ: обработка голоса не собралась, голос идёт как есть: {r.stderr[-200:].strip()}")
        r2 = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-ar", "48000", "-ac", "2", out_path],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        return out_path if r2.returncode == 0 else voice_path
    return out_path


def get_media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def audio_qc(path, label="Audio QC"):
    """Технический QC дорожки — тот же принцип, что qc_report() для
    картинок: не блокирует, только честно докладывает брак, который иначе
    заметили бы только на слух постфактум (клиппинг, слишком тихо/громко,
    длинные участки мёртвой тишины помимо самих [pause] — забытый обрыв
    записи). Бесплатно — штатные ffmpeg-фильтры astats/silencedetect,
    никакого нового API.

    Вызывается ДВАЖДЫ: на входном голосе (что нам дали) и на ГОТОВОМ
    final.mp4 (что услышит зритель) — см. label. Долгое время проверялся
    только вход, и вся мастер-цепочка (обработка голоса, подложка, дакинг,
    loudnorm, лимитер) уходила к зрителю вообще без технической проверки:
    обе реальные поломки музыки в этом проекте нашлись ручными замерами,
    а не отчётом.
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostats", "-i", path, "-af",
             "astats=reset=0,silencedetect=noise=-40dB:d=1.5", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            # Тот же класс, что чинил аудит 04.09 у measure_loudnorm_stats():
            # фиксированные 120 с — это полный декод, на часовом эпизоде он
            # не укладывается, и QC молча не выполнялся вообще.
            timeout=max(120, int(_audio_len_for_timeout(path) * 0.5)))
        err = r.stderr
    except Exception as e:
        print(f"  {label}: не удалось проверить ({e})")
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
    print(f"{label}: {'; '.join(warns)}" if warns
          else f"{label}: клиппинга/аномальной громкости не найдено")


def build_master_af(loud_stats, fade_out_st, fade_in_sec):
    """Мастер-цепочка финального прохода: loudnorm -> лимитер -> фейды.

    Вынесено из main() отдельной функцией, чтобы порядок ступеней можно было
    проверить тестом, а не только прочитать. Порядок здесь — не вкусовщина:

    * лимитер ПОСЛЕ loudnorm — иначе loudnorm поднял бы уровень уже после
      того, как потолок выставлен, и потолок перестал бы быть потолком;
    * лимитер ПЕРЕД afade — afade только уменьшает уровень, так что на
      результат он не влияет, но обратный порядок заставил бы лимитер
      работать на затухающих хвостах, где он бесполезен;
    * порог лимитера — та же константа LOUDNORM_TARGET_TP, что цель true
      peak у loudnorm: две ступени не должны спорить о потолке.

    loud_stats=None — первый проход не измерился, идём однопроходным
    (динамическим) loudnorm; это уже существующий откат, лимитер в нём
    тем более уместен.
    """
    base = f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}"
    if loud_stats:
        base += (f":linear=true:measured_I={loud_stats['input_i']}:"
                 f"measured_TP={loud_stats['input_tp']}:"
                 f"measured_LRA={loud_stats['input_lra']}:"
                 f"measured_thresh={loud_stats['input_thresh']}:"
                 f"offset={loud_stats['target_offset']}")
    stages = [base]
    if MASTER_LIMITER_ENABLED:
        stages.append(f"alimiter=limit={LOUDNORM_TARGET_TP}dB:"
                      f"attack={MASTER_LIMITER_ATTACK_MS}:"
                      f"release={MASTER_LIMITER_RELEASE_MS}:level=disabled")
    stages.append(f"afade=t=in:st=0:d={fade_in_sec}")
    stages.append(f"afade=t=out:st={fade_out_st:.3f}:d=2")
    return ",".join(stages)
