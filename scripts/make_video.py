#!/usr/bin/env python3
"""Один ролик одной командой: сценарий -> план кадров -> картинки -> озвучка -> final.mp4.

  python scripts/make_video.py videos/01_tema [--minutes 12] [--confirm-spend] [--tts]

Шаги (каждый можно запускать и отдельно своим скриптом):
  1. wordcount.py      — длина сценария против цели (стоп, если вне коридора).
  2. frame_planner.py  — план кадра на каждую фразу (DeepSeek, копейки).
  3. frame_generator.py — картинки сильной моделью + проверка русских подписей.
                         ПЛАТНО: без --confirm-spend только печатает цену.
  4. lumean_tts.py     — озвучка, только с --tts (платно; иначе положите audio.mp3 сами).
  5. fix_pauses.py     — подрезка длинных пауз TTS (бесплатно, локально).
  6. assemble_frames.py — сборка final.mp4 + final.srt.
Останавливается на первом шаге, который не может продолжить, и говорит, что сделать."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def step(name, args, ok_codes=(0,)):
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, args[0])] + args[1:])
    if r.returncode not in ok_codes:
        sys.exit(f"Шаг «{name}» остановился (код {r.returncode}). Исправьте и перезапустите — "
                 f"готовое берётся из кэша.")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--minutes", type=float, default=None, help="целевая длина T (минут)")
    ap.add_argument("--confirm-spend", action="store_true", help="разрешить платную генерацию картинок")
    ap.add_argument("--tts", action="store_true", help="озвучить через Lumean (платно)")
    ap.add_argument("--allow-text-mismatch", action="store_true",
                    help="собрать, даже если подписи на части кадров не прошли проверку")
    a = ap.parse_args()
    vd = os.path.abspath(a.video_dir)
    script = os.path.join(vd, "script.txt")
    if not os.path.exists(script):
        sys.exit(f"Нет {script}")
    if a.minutes:
        step("Длина сценария", ["wordcount.py", script, str(a.minutes)])
    step("План кадров", ["frame_planner.py", vd])
    gen = ["frame_generator.py", vd] + (["--confirm-spend"] if a.confirm_spend else [])
    step("Картинки", gen, ok_codes=(0, 2) if a.allow_text_mismatch else (0,))
    if not any(os.path.exists(os.path.join(vd, n)) for n in ("audio.mp3", "audio.wav", "audio.flac")):
        if not a.tts:
            sys.exit("Нет audio.mp3. Положите озвучку в папку ролика или запустите с --tts (Lumean, платно).")
        step("Озвучка", ["lumean_tts.py", vd] + ([str(a.minutes)] if a.minutes else []))
    step("Паузы", ["fix_pauses.py", vd])
    step("Сборка", ["assemble_frames.py", vd])


if __name__ == "__main__":
    main()
