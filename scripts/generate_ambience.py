#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генератор источников атмосферного слоя — процедурный синтез.

Тот же принцип, что у generate_music_asset.py / generate_grain_asset.py /
generate_reveal_sfx.py: звук считается с нуля и является нашей
собственностью. Ни библиотеки, ни её лимита, ни чужой лицензии, ни риска
Content ID.

ПОЧЕМУ ТРИ ИСТОЧНИКА НА АТМОСФЕРУ, А НЕ ОДИН ФАЙЛ. Один луп слышен как
луп — это ровно то, чего требовалось избежать. Здесь каждая атмосфера
состоит из ТРЁХ слоёв ВЗАИМНО ПРОСТОЙ длины (37/53/71 секунды), которые
при сборке крутятся независимо. Полная комбинация повторяется через их
произведение: 37*53*71 = 139231 секунда = 38.7 часа. Двадцатипятиминутный
эпизод физически не успевает дойти до повторения — это арифметика, а не
обещание. Сверх этого каждая глава стартует слои со своего сдвига (seed от
имени секции), а громкость каждого слоя ведёт своя медленная кривая с
простым периодом 23/31/43с — «дыхание» тоже не совпадает само с собой.

Слои — не три копии одного звука, а РАЗНЫЕ ПОЛОСЫ одной текстуры (низ /
середина / верх). Поэтому по отдельности каждый неполон, а вместе они
дают один связный звук, а не «три ветра одновременно».

ЧЕСТНЫЙ ПРЕДЕЛ МЕТОДА, записан здесь, а не обнаружен потом: синтез хорошо
делает НЕПРЕРЫВНЫЕ текстуры — ветер, дождь, гул зала, огонь, дальний
гомон. Он плохо делает предметные разовые звуки: колокол, лошадь, крик
птицы. Поэтому в слое есть только текстуры, а разовых звуков нет вообще —
не «пока нет», а сознательно: плохо синтезированный колокол слышен как
подделка мгновенно, в отличие от полосы шума.

Каждый файл сведён БЕСШОВНО: хвост длиной CROSSFADE_SEC подмешан в начало
с обратной кривой, поэтому при зацикливании стыка не слышно (тот же приём,
что для шумового слоя в generate_music_asset.py).

Запуск (результат коммитится как статический ассет):
    .venv/bin/python scripts/generate_ambience.py
    .venv/bin/python scripts/generate_ambience.py --preview 20
        собрать 20-секундную демонстрацию каждой атмосферы в
        assets/ambience/_preview/ и ПОСЛУШАТЬ до первого рендера эпизода.
"""
import argparse
import os
import subprocess

import numpy as np

SR = 48000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets", "ambience")

LAYER_SECONDS = (37, 53, 71)     # см. ambience_plan.AMBIENCE_LAYER_SECONDS
LAYER_NAMES = ("low", "mid", "high")
CROSSFADE_SEC = 2.0
PEAK_DBFS = -12.0                # нормировка источника; рабочий уровень ставит сведение


def _write_flac(mono, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stereo = np.stack([mono, mono], axis=1)
    pcm16 = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    r = subprocess.run(["ffmpeg", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "2",
                        "-i", "-", "-c:a", "flac", path],
                       input=pcm16, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg упал на {path}: {r.stderr.decode(errors='ignore')[-600:]}")


def _noise(n, seed):
    return np.random.default_rng(seed).normal(0.0, 1.0, n)


def _bandpass(sig, low_hz, high_hz):
    from scipy.signal import butter, sosfilt
    high_hz = min(high_hz, SR / 2 * 0.98)
    sos = butter(2, [low_hz, high_hz], btype="band", fs=SR, output="sos")
    return sosfilt(sos, sig)


def _slow_env(n, seed, low_hz=0.03, high_hz=0.25, depth=0.6):
    """Медленная огибающая «порывов»: сам шум, отфильтрованный в доли герца.

    Не сумма синусов: сумма синусов даёт узнаваемо периодическое качание,
    а отфильтрованный шум — неповторяющееся движение, которое и слышно как
    живое. Приведён к [1-depth, 1].
    """
    env = _bandpass(_noise(n, seed), max(low_hz, 0.01), high_hz)
    env /= (np.max(np.abs(env)) or 1.0)
    return 1.0 - depth + depth * (env * 0.5 + 0.5)


def _sparse_impulses(n, seed, rate_per_sec, decay, band):
    """Редкие затухающие импульсы (потрескивание огня, капли).

    Моменты — пуассоновский поток, а не сетка: равномерная сетка щелчков
    слышна как метроном, то есть как автомат.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    count = max(1, int(n / SR * rate_per_sec))
    tail = int(SR * 0.35)
    env = np.exp(-decay * np.arange(tail) / SR)
    for pos in rng.integers(0, max(1, n - tail), size=count):
        amp = rng.uniform(0.25, 1.0)
        out[pos:pos + tail] += amp * env * rng.normal(0.0, 1.0, tail)
    return _bandpass(out, *band)


def _seamless(sig, crossfade_sec=CROSSFADE_SEC):
    """Бесшовная петля: хвост подмешивается в начало с обратной кривой."""
    n = len(sig)
    k = min(int(crossfade_sec * SR), n // 4)
    if k <= 0:
        return sig
    ramp = np.linspace(0.0, 1.0, k)
    head, tail = sig[:k].copy(), sig[-k:].copy()
    sig = sig[:n - k]
    sig[:k] = head * ramp + tail * (1.0 - ramp)
    return sig


def _normalize(sig):
    return sig / (np.max(np.abs(sig)) or 1.0) * 10 ** (PEAK_DBFS / 20.0)


# Полосы и характер каждой атмосферы по слоям (низ / середина / верх).
# Ключи совпадают с AMBIENCE_VOCAB в scripts/ambience_plan.py.
BEDS = {
    "wind_open": {
        "low": dict(band=(40, 180), gust=(0.02, 0.12), depth=0.45),
        "mid": dict(band=(180, 1100), gust=(0.04, 0.22), depth=0.65),
        "high": dict(band=(1100, 5000), gust=(0.06, 0.30), depth=0.75),
    },
    "stone_hall": {
        # Зал — почти неподвижный низкий гул и очень тихий «воздух».
        # Быстрое движение тут звучало бы как ветер в помещении.
        "low": dict(band=(35, 140), gust=(0.01, 0.05), depth=0.25),
        "mid": dict(band=(140, 700), gust=(0.01, 0.07), depth=0.35),
        "high": dict(band=(700, 3200), gust=(0.02, 0.10), depth=0.40),
    },
    "forge_fire": {
        "low": dict(band=(45, 200), gust=(0.03, 0.15), depth=0.40),
        "mid": dict(band=(200, 1200), gust=(0.05, 0.25), depth=0.55,
                    impulses=dict(rate=2.5, decay=26.0, band=(400, 2600), mix=0.55)),
        "high": dict(band=(1200, 6000), gust=(0.05, 0.28), depth=0.60,
                     impulses=dict(rate=4.0, decay=40.0, band=(1800, 7000), mix=0.65)),
    },
    "rain_mud": {
        "low": dict(band=(50, 220), gust=(0.02, 0.10), depth=0.30),
        "mid": dict(band=(500, 2500), gust=(0.04, 0.18), depth=0.35),
        "high": dict(band=(2500, 9000), gust=(0.05, 0.22), depth=0.40,
                     impulses=dict(rate=22.0, decay=90.0, band=(2000, 8000), mix=0.45)),
    },
    "crowd_market": {
        # Гомон — узкая речевая полоса с быстрым, но неритмичным качанием.
        # Слов в нём нет и быть не должно: разборчивое слово в фоне
        # перетягивает внимание с закадра мгновенно.
        "low": dict(band=(60, 250), gust=(0.02, 0.10), depth=0.35),
        "mid": dict(band=(250, 1400), gust=(0.08, 0.45), depth=0.70),
        "high": dict(band=(1400, 4500), gust=(0.10, 0.50), depth=0.75),
    },
}


def _seed_for(bed, layer):
    """Детерминированный seed. Не hash(): он солится от запуска к запуску, и
    пересборка ассетов давала бы другой звук при том же коде."""
    return sum(ord(c) * (i + 7) for i, c in enumerate(bed + layer))


def make_layer(bed, layer, seconds, seed):
    spec = BEDS[bed][layer]
    n = int(SR * seconds)
    sig = _bandpass(_noise(n, seed), *spec["band"])
    sig /= (np.max(np.abs(sig)) or 1.0)
    imp = spec.get("impulses")
    if imp:
        pulses = _sparse_impulses(n, seed + 1, imp["rate"], imp["decay"], imp["band"])
        pulses /= (np.max(np.abs(pulses)) or 1.0)
        sig = (1.0 - imp["mix"]) * sig + imp["mix"] * pulses
    lo, hi = spec["gust"]
    sig = sig * _slow_env(n, seed + 2, lo, hi, spec["depth"])
    return _normalize(_seamless(sig))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", type=float, default=0.0,
                    help="собрать демонстрацию такой длины (сек) на каждую атмосферу")
    ap.add_argument("--beds", default="", help="только эти атмосферы, через запятую")
    args = ap.parse_args()

    beds = [b.strip() for b in args.beds.split(",") if b.strip()] or list(BEDS)
    # --preview НЕ пишет рабочие слои намеренно: наличие слоёв на диске и
    # есть выключатель фичи в сборке, и «послушать демонстрацию» не должно
    # незаметно включать атмосферу в следующем же рендере эпизода.
    preview_only = args.preview > 0
    for bed in beds:
        print(f"{bed}:")
        if not preview_only:
            for name, seconds in zip(LAYER_NAMES, LAYER_SECONDS):
                sig = make_layer(bed, name, seconds, _seed_for(bed, name))
                path = os.path.join(OUT_DIR, bed, f"{name}_{seconds}s.flac")
                _write_flac(sig, path)
                peak = 20 * np.log10(np.max(np.abs(sig)) or 1e-9)
                rms = 20 * np.log10(np.sqrt(np.mean(sig ** 2)) or 1e-9)
                print(f"  {name:5s} {seconds:3d}с  пик {peak:6.1f}  RMS {rms:6.1f}  -> {path}")

        if preview_only:
            # Демонстрация: те же три слоя, сложенные так же, как их сложит
            # сборка — чтобы слушать РЕАЛЬНЫЙ результат, а не один слой.
            n = int(SR * args.preview)
            mix = np.zeros(n)
            for name, seconds in zip(LAYER_NAMES, LAYER_SECONDS):
                sig = make_layer(bed, name, seconds, _seed_for(bed, name))
                reps = int(np.ceil(n / len(sig)))
                mix += np.tile(sig, reps)[:n]
            mix = _normalize(mix)
            prev = os.path.join(OUT_DIR, "_preview", f"{bed}.flac")
            _write_flac(mix, prev)
            print(f"  демонстрация -> {prev}")

    if preview_only:
        print("Готово: только демонстрации. Рабочие слои НЕ записаны — атмосфера "
              "в сборке пока выключена. Понравилось — запустить без --preview.")
    else:
        print("Готово: рабочие слои записаны, атмосфера в сборке включится "
              "со следующего рендера.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
