"""Одноразовый генератор assets/grain/grain_loop.mp4 — процедурное плёночное
зерно (не настоящий скан, честная математическая имитация, см. film_look()
в pipeline_smart.py). Не часть рантайм-пайплайна — запускается вручную при
необходимости пересоздать/перекалибровать текстуру, результат коммитится
в репозиторий как статический ассет (тот же принцип, что assets/fonts).

Почему не голый noise=alls=3: у настоящего плёночного зерна структура
сгустками (соседние "зёрна" коррелированы по размеру/яркости), лёгкая
цветность (не только яркостный шум) и посекундная смена (не статичная
текстура) — обычный per-pixel независимый шум читается как "цифровой"
именно из-за отсутствия этих свойств.
"""
import os
import subprocess

import numpy as np
from PIL import Image

W, H = 480, 270          # намеренно мельче целевого 1920x1080 — апскейл при
                          # блендинге даёт естественное смягчение/укрупнение
# P2-23 (аудит звукового/видео пайплайна): было FPS=25, а проект с этой же
# сессии рендерит на 24 (см. FPS в pipeline_smart.py) — grain_loop.mp4 через
# -stream_loop блендится (grainmerge/blend, framesync) поверх видео другой
# частоты кадров: ffmpeg сам досэмплирует по PTS, но каденс обновления зерна
# ПОЛЗЁТ относительно кадров видео (иногда дублирует один и тот же кадр
# зерна), а не идёт 1:1 кадр-в-кадр — тот же класс рассинхрона, что уже
# ловили и чинили для звука. Fix — держать оба в одной частоте.
FPS = 24
# P1-21: было 2.4с (60 кадров) — при заметном крупном плане зритель успевает
# "выучить" петлю за пару повторов (зерно с крупными сгустками совершает
# заметно мало уникальных положений). 8с петля перед той же неповторяющейся
# бесшовностью (XFADE_N ниже) делает цикл субъективно незаметным.
DUR = 8.0
N = int(FPS * DUR)        # 192 кадра
SEED = 20260819            # детерминированный сид — воспроизводимый ассет

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "grain")
OUT_PATH = os.path.join(OUT_DIR, "grain_loop.mp4")


# P1-22 (аудит звукового/видео пайплайна): было — N НЕЗАВИСИМЫХ случайных
# кадров + искусственный линейный кроссфейд последних XFADE_N кадров с
# первыми, чтобы шов петли не был виден. Кроссфейд — заплатка: на стыке
# два независимых случайных кадра УСРЕДНЯЮТСЯ, что даёт заметно более
# гладкую (меньше дисперсии) текстуру именно в этих кадрах — то есть
# "плотность" зерна едва заметно проседает раз за петлю, ровно там, где
# должна быть незаметна.
#
# Правильное решение — та же идея, что уже дала точную периодичность
# дрейфу расстройки дрона в generate_music_asset.py (PM с частотами,
# снапнутыми на целое число циклов на петлю), только для шума: случайный
# ФАЗОВЫЙ спектр (амплитуда постоянна по частоте — "белый" по времени,
# та же статистика, что и у прежнего purely-независимого per-frame шума)
# с обратным FFT даёт ДЕЙСТВИТЕЛЬНЫЙ, ТОЧНО периодичный на N кадров
# временной ряд для каждой точки поля — без какого-либо шва и без
# кроссфейда-заплатки в принципе (см. periodic_noise_volume).
def periodic_noise_volume(rng, shape, n_frames):
    """Шум, независимый по каждой пространственной точке `shape`, СЛУЧАЙНЫЙ
    и "белый" по времени (как честный per-frame независимый шум), но ТОЧНО
    периодичный на n_frames кадров — без кроссфейда, по построению (см.
    комментарий выше). Возвращает (n_frames, *shape), std по всему объёму
    приведено к ~1."""
    n_bins = n_frames // 2 + 1
    phases = rng.uniform(0, 2 * np.pi, size=shape + (n_bins,)).astype(np.float32)
    mag = np.ones(shape + (n_bins,), dtype=np.float32)
    mag[..., 0] = 0.0    # DC = 0 — нулевое среднее по петле
    mag[..., -1] = 0.0   # Найквист = 0 — избегаем чисто-действительного частного случая
    spectrum = (mag * np.exp(1j * phases)).astype(np.complex64)
    vol = np.fft.irfft(spectrum, n=n_frames, axis=-1)     # (*shape, n_frames), точно периодично
    vol = np.moveaxis(vol, -1, 0).astype(np.float32)      # (n_frames, *shape)
    vol = (vol - vol.mean()) / (vol.std() + 1e-9)
    return vol


def coarse_volume(rng, scale_div, n_frames):
    """Периодический шум на низком разрешении, апскейленный бикубикой
    покадрово — источник "сгустков" (крупных пятен зерна), не отдельных
    пикселей. Тот же принцип, что был у coarse_field(), но целый объём
    кадров разом, из periodic_noise_volume вместо N независимых вызовов."""
    h2, w2 = max(2, H // scale_div), max(2, W // scale_div)
    small_vol = periodic_noise_volume(rng, (h2, w2), n_frames)
    up = np.empty((n_frames, H, W), dtype=np.float32)
    for i in range(n_frames):
        frame = small_vol[i]
        lo, hi = frame.min(), frame.max()
        frame_u8 = ((frame - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
        img = Image.fromarray(frame_u8).resize((W, H), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        up[i] = (arr - arr.mean()) / (arr.std() + 1e-6)
    return up


def gen_luma_volume(rng, n_frames):
    """Multi-scale: крупные сгустки (делитель 10) + среднее зерно (делитель
    4) + мелкая текстура (полное разрешение), взвешенная сумма по ВСЕМ
    n_frames кадрам разом — органичнее, чем один масштаб шума, точно
    периодично на n_frames без кроссфейда."""
    coarse = coarse_volume(rng, 10, n_frames)
    mid = coarse_volume(rng, 4, n_frames)
    fine = periodic_noise_volume(rng, (H, W), n_frames)
    grain = 0.35 * coarse + 0.35 * mid + 0.30 * fine
    # Нормализация по ВСЕМУ объёму разом (не по каждому кадру отдельно, как
    # раньше в gen_luma_frame) — устраняет источник мелкого кадр-к-кадру
    # дрожания КОНТРАСТА зерна от случайных отличий измеренного std между
    # кадрами; все кадры статистически одинаковы по построению, общая
    # нормализация корректна и строже прежней.
    grain /= (np.abs(grain).std() * 3.2 + 1e-6)
    return np.clip(grain, -1.2, 1.2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    luma_vol = gen_luma_volume(rng, N)   # (N, H, W), точно периодично — без кроссфейда

    # Лёгкая независимая цветность (реальное зерно не строго ахроматично) —
    # отдельный, куда более слабый шумовой канал на Cb/Cr. P1-22: раньше
    # генерировался per-frame независимо БЕЗ какой-либо компенсации шва
    # (даже кроссфейда, который был хотя бы у яркости) — тот же периодический
    # FFT-фазовый метод убирает шов и здесь той же одной правкой.
    chroma_rng = np.random.default_rng(SEED + 1)
    cb_vol = periodic_noise_volume(chroma_rng, (H, W), N) * 3.0
    cr_vol = periodic_noise_volume(chroma_rng, (H, W), N) * 3.0

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "12",
           "-pix_fmt", "yuv420p", OUT_PATH]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for i in range(N):
        base = 128.0 + luma_vol[i] * 22.0   # амплитуда яркостного зерна
        cb_noise, cr_noise = cb_vol[i], cr_vol[i]
        r = base + cr_noise * 0.6
        g = base - cr_noise * 0.2 - cb_noise * 0.2
        b = base + cb_noise * 0.6
        frame = np.stack([r, g, b], axis=-1)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode(errors="ignore")
    proc.wait()
    if proc.returncode != 0:
        print("ffmpeg упал:", err[-1500:])
        raise SystemExit(1)
    print(f"Готово: {OUT_PATH} ({N} кадров, {W}x{H}, {DUR}с петля)")


if __name__ == "__main__":
    main()
