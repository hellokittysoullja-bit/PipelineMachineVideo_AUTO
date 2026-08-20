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
XFADE_N = 10               # хвостовых кадров кроссфейдим с началом — бесшовная петля
SEED = 20260819            # детерминированный сид — воспроизводимый ассет

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "grain")
OUT_PATH = os.path.join(OUT_DIR, "grain_loop.mp4")


def coarse_field(rng, scale_div):
    """Шум на низком разрешении, апскейленный бикубикой — источник "сгустков"
    (крупных пятен зерна), не отдельных пикселей."""
    small = rng.normal(0, 1, (max(2, H // scale_div), max(2, W // scale_div))).astype(np.float32)
    small -= small.min()
    small /= (small.max() + 1e-6)
    img = Image.fromarray((small * 255).astype(np.uint8))
    up = img.resize((W, H), Image.BICUBIC)
    arr = np.asarray(up, dtype=np.float32) / 255.0
    return (arr - arr.mean()) / (arr.std() + 1e-6)


def gen_luma_frame(rng):
    """Multi-scale: крупные сгустки (delen 10) + среднее зерно (делитель 4) +
    мелкая текстура (полное разрешение), взвешенная сумма — органичнее, чем
    один масштаб шума."""
    coarse = coarse_field(rng, 10)
    mid = coarse_field(rng, 4)
    fine = rng.normal(0, 1, (H, W)).astype(np.float32)
    fine = (fine - fine.mean()) / (fine.std() + 1e-6)
    grain = 0.35 * coarse + 0.35 * mid + 0.30 * fine
    grain /= (np.abs(grain).std() * 3.2 + 1e-6)   # нормализация к ~[-1,1] с запасом
    return np.clip(grain, -1.2, 1.2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    luma_frames = [gen_luma_frame(rng) for _ in range(N)]
    # Бесшовная петля: последние XFADE_N кадров линейно кроссфейдим с первыми
    # XFADE_N — иначе в точке склейки петли виден скачок (два независимых
    # случайных кадра встык).
    head = [luma_frames[i].copy() for i in range(XFADE_N)]
    for k in range(XFADE_N):
        i = N - XFADE_N + k
        w = (k + 1) / (XFADE_N + 1)   # 0 -> 1 к концу
        luma_frames[i] = luma_frames[i] * (1 - w) + head[k] * w

    # Лёгкая независимая цветность (реальное зерно не строго ахроматично) —
    # отдельный, куда более слабый шумовой канал на Cb/Cr.
    chroma_rng = np.random.default_rng(SEED + 1)

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "12",
           "-pix_fmt", "yuv420p", OUT_PATH]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for luma in luma_frames:
        base = 128.0 + luma * 22.0   # амплитуда яркостного зерна
        cb_noise = chroma_rng.normal(0, 1, (H, W)).astype(np.float32) * 3.0
        cr_noise = chroma_rng.normal(0, 1, (H, W)).astype(np.float32) * 3.0
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
