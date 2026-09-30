#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генератор недостающих звуковых эффектов — переход между главами и
появление плашки на экране. Процедурный синтез, тот же принцип и те же
строительные блоки, что у `generate_reveal_sfx.py`, `generate_music_asset.py`
и `generate_grain_asset.py`: звук честно посчитан с нуля и является нашей
собственностью — не сток, не сэмпл, не скачанный файл, никакого риска
Content ID и ни одной строки чужой лицензии.

ЗАЧЕМ. Замер реального эпизода `02_ne-mechom` (27 минут): звуковых событий
за весь ролик — четыре. Девятнадцать границ глав проходят в полной тишине,
девять плашек с цифрами из одиннадцати выскакивают на экран беззвучно.
Ассетов под эти два случая в репозитории не существовало вообще.

УРОВНИ — не на глаз, а от цели. Программа после мастеринга идёт в -14 LUFS
(LOUDNORM_TARGET_I), пики упираются в -1.5 dBTP. Эффект, который обязан
читаться как акцент, но заведомо НЕ спорить с голосом, должен жить в районе
-22..-26 dBFS по пику в готовом миксе. Поэтому здесь каждый ассет
нормируется к ЯВНО ОБЪЯВЛЕННОМУ пику (PEAK_DBFS), а нужное ослабление
добавляет уже сведение (SFX_* _GAIN_DB в pipeline_smart.py) — два числа,
каждое проверяемое, вместо одного «подобранного».

Реальная причина так делать (найдено 13.09 прямо здесь): у
`generate_reveal_sfx.py` версия v2 подняла пики с -14 до -10/-7 dBFS, а
константа `REVEAL_SFX_GAIN_DB = -8.0` в pipeline_smart.py и комментарий
рядом с ней остались от v1 и до сих пор ссылаются на «пик ок. -14 dBFS».
То есть расчётный уровень акцента в миксе разошёлся с реальным на 6-7 дБ,
и заметить это было негде. Здесь пик ассета печатается при генерации и
проверяется тестом.

ХАРАКТЕР ЗВУКА — документальный, не трейлерный (ЧАСТЬ 8 CLAUDE.md: без
пафоса, ничего резкого):

* переход между главами — не «вжух», а короткий низкий ВЫДОХ: полоса шума,
  съезжающая вниз, с мягким входом и выходом. Читается как смена дыхания
  между мыслями, а не как спецэффект;
* появление плашки — короткий приглушённый тик в полосе 1.2-3 кГц. Именно
  эта полоса реально слышна на телефонном динамике (та же причина, по
  которой v2 reveal-ассетов ушла из саб-баса вверх), и именно она не
  конфликтует с низом голоса.

ДВА ВАРИАНТА ДЛИНЫ У ПЕРЕХОДА — не украшение. `fix_pauses.py` подрезает
паузы по кривой в диапазоне 0.42..1.35с, то есть паузы на границах глав
одного и того же эпизода РАЗНЫЕ. Планировщик (`scripts/sfx_plan.py`) берёт
самый длинный вариант, помещающийся в реальную тишину этой конкретной
границы, и отказывается от эффекта, если не помещается ни один.

Запуск (результат коммитится как статический ассет, не часть рантайма):
    .venv/bin/python scripts/generate_sfx_pack.py
    .venv/bin/python scripts/generate_sfx_pack.py --clicks   # см. ниже
"""
import argparse
import os
import subprocess

import numpy as np

SR = 48000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSITION_DIR = os.path.join(ROOT, "assets", "sfx", "transition")
UI_DIR = os.path.join(ROOT, "assets", "sfx", "ui")
CLICKS_DIR = os.path.join(ROOT, "assets", "sfx", "keyboard_clicks")

# Явно объявленный пик каждого сгенерированного ассета. Ослабление до
# рабочего уровня делает сведение — см. докстринг выше.
PEAK_DBFS = -10.0

# Длительности вариантов перехода. Короткий обязан помещаться в САМУЮ
# короткую паузу, которую оставляет fix_pauses.py (KEEP_MIN_SEC = 0.42) с
# запасом на зазор перед первым словом (CHAPTER_HEADROOM_SEC = 0.03).
TRANSITION_SHORT_SEC = 0.38
TRANSITION_LONG_SEC = 0.90
PLATE_TICK_SEC = 0.16


def _write_flac(samples_mono, path):
    """float64 [-1,1] моно -> стерео FLAC через ffmpeg (тот же паттерн, что
    generate_reveal_sfx.py: raw PCM в stdin, без лишней зависимости)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stereo = np.stack([samples_mono, samples_mono], axis=1)
    pcm16 = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    cmd = ["ffmpeg", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-",
           "-c:a", "flac", path]
    r = subprocess.run(cmd, input=pcm16, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg упал на {path}: {r.stderr.decode(errors='ignore')[-800:]}")


def _brown_noise(n, seed):
    """Коричневый шум — интеграл белого, нормированный. Тот же метод, что в
    generate_reveal_sfx.py (одна и та же текстура во всём звуковом пакете,
    а не три разных шума от трёх генераторов)."""
    rng = np.random.default_rng(seed)
    brown = np.cumsum(rng.normal(0.0, 1.0, n))
    brown -= brown.mean()
    return brown / (np.max(np.abs(brown)) or 1.0)


def _bandpass(sig, low_hz, high_hz, sr=SR):
    from scipy.signal import butter, sosfilt
    sos = butter(2, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    return sosfilt(sos, sig)


def _normalize(sig, peak_dbfs=PEAK_DBFS):
    peak = np.max(np.abs(sig)) or 1.0
    return sig / peak * 10 ** (peak_dbfs / 20.0)


def make_transition(duration_sec, seed=20260913):
    """Выдох на границе главы: полоса шума, съезжающая сверху вниз.

    Огибающая — быстрый вход и длинный спад (не симметричная «капля»):
    эффект заканчивается ровно на первом слове новой главы, и симметричная
    форма означала бы максимум энергии в СЕРЕДИНЕ паузы, то есть акцент не
    на переходе, а рядом с ним.

    Полоса едет 1400 -> 260 Гц. Верх взят не ниже: на телефонном динамике
    всё, что ниже ~200 Гц, почти не воспроизводится (та же причина, по
    которой v2 reveal-ассетов ушла вверх из саб-баса), а движение полосы
    вниз — это и есть слышимый «выдох», от которого переход читается как
    завершение мысли, а не как включение эффекта.
    """
    n = int(SR * duration_sec)
    t = np.arange(n) / SR
    x = t / duration_sec

    noise = _brown_noise(n, seed)
    # Съезд полосы вниз — три перекрывающиеся полосы с перекрёстным
    # затуханием дешевле и устойчивее, чем изменяемый во времени фильтр
    # (sosfilt с переменными коэффициентами требует поблочной обработки и
    # даёт слышимые ступеньки на стыках блоков).
    bands = ((900.0, 1900.0), (450.0, 950.0), (200.0, 480.0))
    weights = (np.clip(1.0 - 2.0 * x, 0.0, 1.0),
               np.clip(1.0 - np.abs(2.0 * x - 1.0), 0.0, 1.0),
               np.clip(2.0 * x - 1.0, 0.0, 1.0))
    sig = np.zeros(n)
    for (lo, hi), w in zip(bands, weights):
        sig += w * _bandpass(noise, lo, hi)

    attack_n = max(1, int(0.06 * SR))
    env = np.exp(-2.2 * x)
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n) ** 2
    fade_n = max(1, int(min(0.05, duration_sec * 0.2) * SR))
    env[-fade_n:] *= np.linspace(1.0, 0.0, fade_n) ** 2
    return _normalize(sig * env)


def make_plate_tick(duration_sec=PLATE_TICK_SEC, seed=20260914):
    """Тик появления плашки. Короткий, приглушённый, в полосе 1.2-3 кГц —
    слышен на любом динамике и не лезет в низ голоса.

    Не «клик» с мгновенной атакой: мгновенный фронт на тихом уровне
    воспринимается как щелчок дефекта (тот же артефакт, из-за которого в
    мастер-цепочке стоит fade-in 50мс вместо сухого старта). Атака 4 мс.
    """
    n = int(SR * duration_sec)
    t = np.arange(n) / SR
    noise = _bandpass(_brown_noise(n, seed), 1200.0, 3000.0)
    # лёгкий тональный призвук сверху — иначе чистый шум читается как
    # «пшик», а не как отметка появления элемента
    tone = np.sin(2 * np.pi * 2100.0 * t)
    env = np.exp(-26.0 * t)
    attack_n = max(1, int(0.004 * SR))
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n)
    return _normalize(env * (0.8 * noise + 0.2 * tone))


def make_key_click(duration_sec=0.085, seed=0):
    """Щелчок клавиши — процедурный.

    ЗАЧЕМ (разбор лицензий 13.09): восемь файлов в assets/sfx/keyboard_clicks/
    — единственный звук во всём проекте, происхождение которого не
    записано нигде. Внутри стоит метка «Computer keyboard, text input.» и
    2024 год, ни автора, ни лицензии, ни генератора. Скорее всего взяты из
    бесплатной библиотеки, но ДОКАЗАТЕЛЬСТВА этого не существует.

    Этот вариант закрывает вопрос окончательно, но НЕ применяется
    автоматически: живая запись клавиши почти всегда звучит лучше синтеза,
    и менять рабочий звук на худший ради формальности — плохой обмен.
    Запускается только явным `--clicks`, перезаписывает папку, решение за
    владельцем после прослушивания обоих вариантов.
    """
    n = int(SR * duration_sec)
    t = np.arange(n) / SR
    rng = np.random.default_rng(20260915 + seed)
    # разброс по вариантам — живая печать не повторяет один и тот же звук
    lo = 1800.0 + rng.uniform(-300.0, 300.0)
    hi = 5200.0 + rng.uniform(-700.0, 700.0)
    decay = 55.0 + rng.uniform(-10.0, 10.0)
    click = _bandpass(_brown_noise(n, seed=20260915 + seed), lo, hi)
    thud = _bandpass(_brown_noise(n, seed=20260916 + seed), 120.0, 420.0)
    env = np.exp(-decay * t)
    attack_n = max(1, int(0.002 * SR))
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n)
    return _normalize(env * (0.85 * click + 0.35 * thud))


def _report(name, sig, path):
    peak = 20 * np.log10(np.max(np.abs(sig)) or 1e-9)
    rms = 20 * np.log10(np.sqrt(np.mean(sig ** 2)) or 1e-9)
    print(f"  {name:22s} {len(sig)/SR:5.3f}с  пик {peak:6.1f} dBFS  RMS {rms:6.1f} dB  -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clicks", action="store_true",
                    help="перегенерировать assets/sfx/keyboard_clicks/ (перезапись!)")
    args = ap.parse_args()

    print("Звуковой пакет:")
    short = make_transition(TRANSITION_SHORT_SEC)
    long_ = make_transition(TRANSITION_LONG_SEC)
    tick = make_plate_tick()
    for name, sig, path in (
        ("chapter_turn_short", short, os.path.join(TRANSITION_DIR, "chapter_turn_short.flac")),
        ("chapter_turn_long", long_, os.path.join(TRANSITION_DIR, "chapter_turn_long.flac")),
        ("plate_tick", tick, os.path.join(UI_DIR, "plate_tick.flac")),
    ):
        _write_flac(sig, path)
        _report(name, sig, path)

    if args.clicks:
        print("Щелчки клавиш (перезапись живых записей синтезом):")
        for i in range(8):
            sig = make_key_click(seed=i)
            path = os.path.join(CLICKS_DIR, f"click_{i}.flac")
            _write_flac(sig, path)
            _report(f"click_{i}", sig, path)

    print("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
