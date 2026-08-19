"""Одноразовый генератор assets/music/ambient_bed.mp3 — процедурная атмосферная
подложка (не сэмпл, честный синтез, см. film_look()/generate_grain_asset.py —
тот же принцип, что и с зерном: у нас нет источника лицензионной музыки без
затрат и без риска Content ID, поэтому подложка честно сгенерирована с нуля
и является нашей собственной). Не часть рантайм-пайплайна — запускается
вручную при необходимости пересоздать/перекалибровать текстуру, результат
коммитится в репозиторий как статический ассет.

Формат ниши (CHANNEL.md) — документальный, спокойный, без пафоса, голос
ровный и низкий. Подложке НЕЛЬЗЯ быть мелодией (спорит с закадром, тянет
внимание) и нельзя быть "стоковым роялти-фри" клише (арпеджио+пэд+легкий бит).
Вместо этого — низкий тональный дрон (несколько расстроенных синусоид на
тонике/квинте/октаве, медленно "дышащих" по громкости) + очень тихая
отфильтрованная шумовая текстура ("воздух") + едва заметный саб-пульс
(медленный тремоло на басовом слое, не ритм, не бит). Ничего резкого,
ничего мелодического — фон, который не потребует пряток за голосом.

Бесшовная петля: несущие синусоиды и их LFO подобраны так, чтобы период
уложился в LOOP_SEC ЦЕЛОЕ число раз (частота*LOOP_SEC — целое) — тогда
сигнал точно периодичен САМ ПО СЕБЕ, без кроссфейда. Единственный
непериодичный слой — фильтрованный шум ("воздух") — для него кроссфейд
хвоста в начало, как в generate_grain_asset.py.
"""
import os
import subprocess

import numpy as np

SR = 44100
LOOP_SEC = 48.0
N = int(SR * LOOP_SEC)
XFADE_SEC = 3.0          # только для непериодичного шумового слоя
XFADE_N = int(SR * XFADE_SEC)
SEED = 20260819

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "music")
OUT_PATH = os.path.join(OUT_DIR, "ambient_bed.mp3")

t = np.arange(N, dtype=np.float64) / SR


def snap_freq(f):
    """Округляет частоту так, чтобы f*LOOP_SEC было целым — несущая
    укладывается ровно в петлю без щелчка на стыке."""
    cycles = round(f * LOOP_SEC)
    return cycles / LOOP_SEC


def drone_layer(freq, amp, lfo_period, lfo_depth, lfo_phase, detune_cents=0.0):
    """Одна несущая: синус на снапнутой частоте + медленная амплитудная
    LFO-волна ("дыхание"). detune_cents — для стереоканала (L/R чуть
    расстроены между собой = естественная ширина без хоруса-эффекта)."""
    f = snap_freq(freq * (2.0 ** (detune_cents / 1200.0)))
    lfo_p = snap_freq(1.0 / lfo_period) if lfo_period > 0 else 0.0
    carrier = np.sin(2 * np.pi * f * t)
    if lfo_p > 0:
        lfo = 1.0 - lfo_depth * 0.5 * (1.0 - np.cos(2 * np.pi * lfo_p * t + lfo_phase))
    else:
        lfo = 1.0
    return amp * carrier * lfo


def pink_noise(rng, n):
    """Розовый шум через кумулятивную фильтрацию белого — грубое, но
    достаточное приближение (1/f спектр примерно, без строгой точности)."""
    white = rng.normal(0, 1, n).astype(np.float64)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    from scipy.signal import lfilter
    pink = lfilter(b, a, white)
    pink /= (np.abs(pink).std() * 3.5 + 1e-9)
    return pink


def air_layer(rng, amp):
    """Тихая "воздушная" текстура: розовый шум, полосовой фильтр (200-3000Hz),
    очень медленная LFO-громкость. Единственный непериодичный слой —
    кроссфейд хвоста в начало отдельно, после сборки основного сигнала."""
    from scipy.signal import butter, sosfilt
    noise = pink_noise(rng, N)
    sos = butter(2, [200, 3000], btype="bandpass", fs=SR, output="sos")
    filtered = sosfilt(sos, noise)
    filtered /= (np.abs(filtered).std() * 4.0 + 1e-9)
    lfo_p = snap_freq(1.0 / 21.0)
    lfo = 0.55 + 0.45 * 0.5 * (1.0 - np.cos(2 * np.pi * lfo_p * t))
    return amp * filtered * lfo


def build_channel(rng, detune_cents):
    sig = np.zeros(N, dtype=np.float64)
    # Тоника (низкий "ля" ~55Hz), квинта, октава — минимальная неопределённая
    # тональность (не мажор/минор впрямую), медленное "дыхание" разной фазы.
    sig += drone_layer(55.0, 0.42, lfo_period=27.0, lfo_depth=0.35, lfo_phase=0.0, detune_cents=detune_cents)
    sig += drone_layer(82.4, 0.24, lfo_period=19.0, lfo_depth=0.40, lfo_phase=1.7, detune_cents=detune_cents)
    sig += drone_layer(110.0, 0.15, lfo_period=33.0, lfo_depth=0.45, lfo_phase=3.1, detune_cents=detune_cents)
    sig += drone_layer(164.8, 0.06, lfo_period=41.0, lfo_depth=0.55, lfo_phase=4.4, detune_cents=detune_cents)
    # Саб-пульс: НЕ бит, а очень медленный тремоло на самой тонике поверх её
    # собственной LFO — едва уловимое "напряжение", период с запасом длиннее
    # любой реальной строки закадра, чтобы не читаться как ритм под речь.
    pulse_p = snap_freq(1.0 / 6.0)
    pulse = 1.0 - 0.12 * 0.5 * (1.0 - np.cos(2 * np.pi * pulse_p * t))
    sig *= (0.85 + 0.15 * pulse)
    air = air_layer(rng, amp=0.05)
    # Кроссфейд хвоста air-слоя в его начало — только этот слой непериодичен.
    head = air[:XFADE_N].copy()
    for k in range(XFADE_N):
        i = N - XFADE_N + k
        w = (k + 1) / (XFADE_N + 1)
        air[i] = air[i] * (1 - w) + head[k] * w
    sig += air
    return sig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng_l = np.random.default_rng(SEED)
    rng_r = np.random.default_rng(SEED + 1)
    left = build_channel(rng_l, detune_cents=-3.0)
    right = build_channel(rng_r, detune_cents=3.0)

    peak = max(np.abs(left).max(), np.abs(right).max())
    target_peak = 10 ** (-12.0 / 20.0)   # -12dBFS запас, подложка тише голоса
    if peak > 1e-9:
        gain = target_peak / peak
        left *= gain
        right *= gain

    stereo = np.stack([left, right], axis=-1)
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")

    cmd = ["ffmpeg", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-",
           "-c:a", "libmp3lame", "-q:a", "2", OUT_PATH]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = proc.communicate(pcm16.tobytes())
    if proc.returncode != 0:
        print("ffmpeg упал:", err.decode(errors="ignore")[-1500:])
        raise SystemExit(1)
    print(f"Готово: {OUT_PATH} ({LOOP_SEC}с петля, {SR}Hz stereo, пик {target_peak:.3f}~-12dBFS)")


if __name__ == "__main__":
    main()
