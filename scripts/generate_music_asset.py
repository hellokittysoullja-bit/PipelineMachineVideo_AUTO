"""Генератор assets/music/ambient_bed*.mp3 — процедурная атмосферная подложка
(не сэмпл, честный синтез, см. film_look()/generate_grain_asset.py — тот же
принцип, что и с зерном: нет источника лицензионной музыки без затрат и без
риска Content ID, поэтому подложка честно сгенерирована с нуля и является
нашей собственной). Не часть рантайм-пайплайна — запускается вручную при
необходимости пересоздать/перекалибровать текстуру, результат коммитится
в репозиторий как статический ассет.

Формат ниши (CHANNEL.md) — документальный, спокойный, без пафоса, голос
ровный и низкий. Подложке НЕЛЬЗЯ быть мелодией (спорит с закадром, тянет
внимание) и нельзя быть "стоковым роялти-фри" клише (арпеджио+пэд+легкий бит).
Вместо этого — низкий тональный дрон (несколько расстроенных синусоид на
тонике/квинте/октаве, медленно "дышащих" по громкости) + очень тихая
отфильтрованная шумовая текстура ("воздух") + едва заметный саб-пульс
(медленный тремоло на басовом слое, не ритм, не бит). Ничего резкого,
ничего мелодического — фон, который не потребует пряток за голосом.

A2: три варианта настроения — та же логика секций, что MOOD_GRADE в
pipeline_smart.py (HOOK холоднее/острее, BODY нейтральная "стальная"
документалка, FINAL теплее/мягче — спад напряжения). Один и тот же
базовый язык синтеза, только пропорции: у HOOK быстрее "дыхание" и
заметнее верхний обертон (тревожнее), у FINAL медленнее и добавлен тихий
суб-октавный слой (теплее/весомее), у BODY — среднее между ними.

Бесшовная петля: несущие синусоиды и их LFO подобраны так, чтобы период
уложился в LOOP_SEC ЦЕЛОЕ число раз (частота*LOOP_SEC — целое) — тогда
сигнал точно периодичен САМ ПО СЕБЕ, без кроссфейда. Единственный
непериодичный слой — фильтрованный шум ("воздух") — для него кроссфейд
хвоста в начало, как в generate_grain_asset.py.

Формат — FLAC, не MP3: пойманная вживую проблема — ffmpeg -stream_loop на
MP3-файле теряет ~26мс (декодерный priming/bit reservoir каждого MP3-кадра)
на КАЖДОМ повторе петли, что на длинном BODY-сегменте (десятки повторов
внутри 18-минутного ролика) накапливается почти в полсекунды рассинхрона
итоговой длины подложки — проверено (893.5с вместо расчётных 894с на
894-секундном отрезке). FLAC lossless — см. проверку в
build_mood_timeline()/pipeline_smart.py на реальных цифрах эпизода.
"""
import json
import os
import re
import subprocess

import numpy as np

SR = 44100
LOOP_SEC = 48.0
N = int(SR * LOOP_SEC)
XFADE_SEC = 3.0          # только для непериодичного шумового слоя
XFADE_N = int(SR * XFADE_SEC)
SEED = 20260819

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "music")

t = np.arange(N, dtype=np.float64) / SR

# mood -> (имя файла, seed-сдвиг, множитель скорости LFO-"дыхания" [меньше —
# быстрее/тревожнее], глубина/период саб-пульса, амплитуда верхнего обертона
# 164.8Hz [ярче/тревожнее], амплитуда доп. суб-октавного слоя 27.5Hz [теплее])
MOODS = {
    "body":  dict(fname="ambient_bed.flac",       seed_off=0,   lfo_mult=1.00,
                  pulse_period=6.0, pulse_depth=0.12, shimmer_amp=0.06, sub_amp=0.0,  air_amp=0.050),
    "hook":  dict(fname="ambient_bed_hook.flac",  seed_off=100, lfo_mult=0.75,
                  pulse_period=4.0, pulse_depth=0.18, shimmer_amp=0.10, sub_amp=0.0,  air_amp=0.065),
    "final": dict(fname="ambient_bed_final.flac", seed_off=200, lfo_mult=1.30,
                  pulse_period=9.0, pulse_depth=0.07, shimmer_amp=0.03, sub_amp=0.05, air_amp=0.035),
}


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


def build_channel(rng, detune_cents, mood):
    lm = mood["lfo_mult"]
    sig = np.zeros(N, dtype=np.float64)
    # Тоника (низкий "ля" ~55Hz), квинта, октава — минимальная неопределённая
    # тональность (не мажор/минор впрямую), медленное "дыхание" разной фазы.
    # lfo_mult короче период (быстрее "дыхание") у HOOK, длиннее (спокойнее)
    # у FINAL — тот же приём, что MOOD_GRADE делает для визуального грейда.
    sig += drone_layer(55.0, 0.42, lfo_period=27.0 * lm, lfo_depth=0.35, lfo_phase=0.0, detune_cents=detune_cents)
    sig += drone_layer(82.4, 0.24, lfo_period=19.0 * lm, lfo_depth=0.40, lfo_phase=1.7, detune_cents=detune_cents)
    sig += drone_layer(110.0, 0.15, lfo_period=33.0 * lm, lfo_depth=0.45, lfo_phase=3.1, detune_cents=detune_cents)
    sig += drone_layer(164.8, mood["shimmer_amp"], lfo_period=41.0 * lm, lfo_depth=0.55, lfo_phase=4.4,
                        detune_cents=detune_cents)
    if mood["sub_amp"] > 0:
        # Тихий суб-октавный слой (27.5Hz) — только FINAL: теплее/весомее,
        # спад напряжения без буквальной "медленной грустной мелодии".
        sig += drone_layer(27.5, mood["sub_amp"], lfo_period=53.0 * lm, lfo_depth=0.30, lfo_phase=2.2,
                            detune_cents=detune_cents)
    # Саб-пульс: НЕ бит, а очень медленный тремоло на самой тонике поверх её
    # собственной LFO — едва уловимое "напряжение", период с запасом длиннее
    # любой реальной строки закадра, чтобы не читаться как ритм под речь.
    # У HOOK короче и глубже (тревожнее), у FINAL длиннее и мельче (спокойнее).
    pulse_p = snap_freq(1.0 / mood["pulse_period"])
    pulse = 1.0 - mood["pulse_depth"] * 0.5 * (1.0 - np.cos(2 * np.pi * pulse_p * t))
    sig *= (1.0 - mood["pulse_depth"] + mood["pulse_depth"] * pulse)
    air = air_layer(rng, amp=mood["air_amp"])
    # Кроссфейд хвоста air-слоя в его начало — только этот слой непериодичен.
    head = air[:XFADE_N].copy()
    for k in range(XFADE_N):
        i = N - XFADE_N + k
        w = (k + 1) / (XFADE_N + 1)
        air[i] = air[i] * (1 - w) + head[k] * w
    sig += air
    return sig


# P0-12 (аудит звукового пайплайна): раньше все 3 mood-варианта нормализо-
# вались к одному ПИКУ (-12dBFS) — но HOOK (яркий shimmer-обертон 164.8Гц)
# и FINAL (добавленный саб-слой 27.5Гц) при равном пике имеют РАЗНУЮ
# субъективную (интегральную) громкость: пик — мгновенное значение, ухо
# слышит энергию за время. На стыке секций HOOK->BODY->FINAL подложка
# "прыгала" по громкости. Нормализация к одному integrated LUFS (тот же
# двухпроходный loudnorm, что уже используется в fix_pauses.py на голосе)
# устраняет это — все 3 файла subjectively одной громкости независимо от
# спектрального состава.
MUSIC_LUFS_TARGET = "I=-30:TP=-3:LRA=3"   # тише голоса (-16 LUFS) на ~14дБ — см. CHANNEL.md house style


def _measure_loudness(path):
    r = subprocess.run(["ffmpeg", "-i", path, "-af",
                        f"loudnorm={MUSIC_LUFS_TARGET}:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', r.stderr, re.S)
    return json.loads(m.group(0)) if m else None


def render_mood(mood_key):
    mood = MOODS[mood_key]
    out_path = os.path.join(OUT_DIR, mood["fname"])
    rng_l = np.random.default_rng(SEED + mood["seed_off"])
    rng_r = np.random.default_rng(SEED + mood["seed_off"] + 1)
    left = build_channel(rng_l, detune_cents=-3.0, mood=mood)
    right = build_channel(rng_r, detune_cents=3.0, mood=mood)

    # Лёгкая пиковая защита ДО loudnorm — не финальная громкость (та ниже,
    # через LUFS), только гарантия, что сигнал не клипует при кодировании
    # в PCM16 перед измерением.
    peak = max(np.abs(left).max(), np.abs(right).max())
    safety_peak = 10 ** (-3.0 / 20.0)
    if peak > 1e-9:
        gain = safety_peak / peak
        left *= gain
        right *= gain

    stereo = np.stack([left, right], axis=-1)
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")

    tmp_path = out_path + ".prenorm.flac"
    cmd = ["ffmpeg", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-",
           "-c:a", "flac", tmp_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = proc.communicate(pcm16.tobytes())
    if proc.returncode != 0:
        print("ffmpeg упал:", err.decode(errors="ignore")[-1500:])
        raise SystemExit(1)

    stats = _measure_loudness(tmp_path)
    if stats:
        af = (f"loudnorm={MUSIC_LUFS_TARGET}:measured_I={stats['input_i']}:"
              f"measured_TP={stats['input_tp']}:measured_LRA={stats['input_lra']}:"
              f"measured_thresh={stats['input_thresh']}:offset={stats.get('target_offset', 0)}:"
              f"linear=true")
    else:
        af = f"loudnorm={MUSIC_LUFS_TARGET}"
    r = subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-af", af, "-c:a", "flac", out_path],
                        capture_output=True, text=True)
    os.remove(tmp_path)
    if r.returncode != 0 or not os.path.exists(out_path):
        print("loudnorm упал:", r.stderr[-1500:])
        raise SystemExit(1)
    print(f"Готово: {out_path} ({LOOP_SEC}с петля, {SR}Hz stereo, {MUSIC_LUFS_TARGET})")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for mood_key in MOODS:
        render_mood(mood_key)


if __name__ == "__main__":
    main()
