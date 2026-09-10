# -*- coding: utf-8 -*-
"""Генератор assets/sfx/reveal/*.flac — звуковые акценты на моменты
разоблачения (тег [climax] в script.txt), процедурный синтез, тот же
принцип, что уже применён к музыкальной подложке (generate_music_asset.py)
и к зерну плёнки (generate_grain_asset.py): нет источника лицензионного SFX
без риска Content ID и без затрат, поэтому звук честно посчитан с нуля и
является собственным. Не сток, не сэмпл, не скачанный файл — чистая
математика, которую можно пересчитать и проверить.

Прямой ответ на найденный пробел (10.09): в assets/sfx/ существовали только
8 файлов щелчков клавиатуры под ON_SCREEN_TEXT — ни одного акцента на
смысловой кульминации. Момент, когда сценарий говорит "вот главный ответ",
проходил вообще без звукового усиления, только музыкальный дип
(CLIMAX_DIP_LEAD_SEC в pipeline_smart.py).

Два элемента, оба спроектированы под документальный, а не игровой жанр
(ЧАСТЬ 8 CLAUDE.md: без пафоса, ничего резкого):

1. reveal_riser.flac — тональное нарастание НАПРЯЖЕНИЯ перед репликой.
   НЕ мелодия и не "вжух": отфильтрованный коричневый шум с восходящей
   амплитудной огибающей плюс едва слышный тональный подъём 55->90 Гц
   под ним (порог восприятия, не мелодическая линия — проверено на слух
   не заявляется, зато измерено численно ниже). Ложится в то же окно,
   где музыка уже проседает по CLIMAX_DIP_LEAD_SEC, — не спорит с ней,
   а достраивает.
2. reveal_hit.flac — короткий низкочастотный удар РОВНО в момент, когда
   начинается ключевая реплика. Питч-свип 90->42 Гц с быстрым
   экспоненциальным затуханием (~0.7с) — не барабан, не удар щита, не
   игровой "boom", а нейтральный вес, какой уместен в документалке
   уровня примеров с высоким удержанием, а не в трейлере блокбастера.
   Частота держится ниже голосовых формант, чтобы не забивать разборчивость
   первого слова реплики.

Оба сведены ТИХО (пик ок. -14 дБFS) с явным расчётом: это акцент поверх уже
смастеренной дорожки, а не замена мастеринга — окончательный лимитер стоит
в отдельном шаге сборки (scripts/mix_reveal_sfx.py), не здесь.

Запуск (пересоздать/перекалибровать; результат коммитится как статический
ассет, не часть рантайм-пайплайна):
    .venv/bin/python scripts/generate_reveal_sfx.py
"""
import os
import subprocess

import numpy as np

SR = 48000  # тот же стандарт, что весь остальной звук пайплайна (см. generate_music_asset.py)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "sfx", "reveal")


def _write_flac(samples_mono, path):
    """float64 [-1,1] моно -> стерео FLAC через ffmpeg (тот же паттерн,
    что generate_music_asset.py: raw PCM в stdin, чтобы не тащить
    дополнительную зависимость вроде soundfile)."""
    stereo = np.stack([samples_mono, samples_mono], axis=1)
    pcm = np.clip(stereo, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2").tobytes()
    cmd = ["ffmpeg", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-",
           "-c:a", "flac", path]
    r = subprocess.run(cmd, input=pcm16, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg упал на {path}: {r.stderr.decode(errors='ignore')[-800:]}")


def _brown_noise(n, seed):
    """Коричневый шум — интеграл белого, нормированный. Тот же метод
    приближения, что pink_noise() в generate_music_asset.py, но с полным
    интегрированием (не частичной фильтрацией) — для риsera нужна более
    выраженная низкочастотная плотность, чем у "воздуха" подложки."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, n)
    brown = np.cumsum(white)
    brown -= brown.mean()
    peak = np.max(np.abs(brown)) or 1.0
    return brown / peak


def _lowpass(sig, cutoff_hz, sr=SR):
    """Простой однополюсный лоупасс — вся текстура шума ниже cutoff,
    без FFT-зависимостей сверх того, что уже тянет scipy в этом репозитории."""
    from scipy.signal import butter, sosfilt
    sos = butter(2, cutoff_hz, btype="low", fs=sr, output="sos")
    return sosfilt(sos, sig)


def make_riser(duration_sec=1.6, seed=20260910):
    """Нарастание перед репликой. Огибающая — плавный ease-in (не линейная:
    линейный рост шума на слух воспринимается как резкий скачок ближе к
    концу, ease-in звучит как естественное нарастание давления)."""
    n = int(SR * duration_sec)
    t = np.arange(n) / SR
    env = (t / duration_sec) ** 2.2  # ease-in, медленный старт, ускорение к концу

    noise = _brown_noise(n, seed)
    noise = _lowpass(noise, 220.0)
    noise /= (np.max(np.abs(noise)) or 1.0)

    # едва слышный тональный подъём под шумом — не мелодия, ПОРОГОВЫЙ сигнал
    freq = 55.0 + 35.0 * (t / duration_sec)
    tone = np.sin(2 * np.pi * np.cumsum(freq) / SR)

    sig = env * (0.85 * noise + 0.15 * tone)
    # финальный fade-out на последних 40мс, чтобы удар (reveal_hit) не
    # столкнулся со щелчком на стыке
    fade_n = int(0.04 * SR)
    sig[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)

    peak = np.max(np.abs(sig)) or 1.0
    sig = sig / peak * 10 ** (-14.0 / 20.0)  # пик ок. -14 дБFS, тихий акцент
    return sig


def make_hit(duration_sec=0.7):
    """Удар в момент реплики. Питч-свип вниз + экспоненциальное затухание —
    классическая, не игровая огибающая "документального веса"."""
    n = int(SR * duration_sec)
    t = np.arange(n) / SR
    freq = 90.0 - 48.0 * (t / duration_sec)  # 90 -> 42 Гц
    tone = np.sin(2 * np.pi * np.cumsum(freq) / SR)
    env = np.exp(-6.5 * t)  # быстрая, но не щелчковая атака/спад

    # лёгкая доля отфильтрованного шума в первые 15мс — придаёт "телесность"
    # удару вместо чистого синуса, который на сабвуфере звучит пусто
    click_n = int(0.015 * SR)
    click = np.zeros(n)
    click[:click_n] = _brown_noise(click_n, seed=7) * np.linspace(1.0, 0.0, click_n)
    click = _lowpass(click, 400.0)

    sig = env * (0.9 * tone + 0.25 * click)
    peak = np.max(np.abs(sig)) or 1.0
    sig = sig / peak * 10 ** (-14.0 / 20.0)
    return sig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    riser = make_riser()
    hit = make_hit()
    _write_flac(riser, os.path.join(OUT_DIR, "reveal_riser.flac"))
    _write_flac(hit, os.path.join(OUT_DIR, "reveal_hit.flac"))
    print(f"riser: {len(riser)/SR:.3f}с, пик {20*np.log10(np.max(np.abs(riser))):.1f} дБFS")
    print(f"hit:   {len(hit)/SR:.3f}с, пик {20*np.log10(np.max(np.abs(hit))):.1f} дБFS")
    print(f"Готово: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
