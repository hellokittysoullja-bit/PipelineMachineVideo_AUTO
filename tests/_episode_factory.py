#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Крохотный эпизод для сквозных прогонов main() — одна копия на все тесты.

Раньше построитель жил внутри test_absorb_known_bad.py. Второй тест, которому
нужен тот же эпизод (test_select_only.py), получил бы вторую копию — а две
копии одной фикстуры расходятся ровно так же, как две копии продакшн-логики:
правят одну, вторая молча продолжает проверять старое.

Слоты остаются без кадра ЕСТЕСТВЕННО, без тест-хуков: локальных файлов
меньше, чем слотов, а все удалённые источники выключены флагами. Никакой
сети, никаких ключей — прогон детерминирован.
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py")

AUDIO_SEC = 14.0   # короче некуда: при 6 фразах клипы уже по ~2с, а суита не должна удваиваться из-за одного теста
N_BLOCKS = 6          # шесть фраз, разделённых [pause]
MEDIA_FOR = (1, 2, 5)  # 1-based имена файлов: media/001_*, 002_*, 005_*

OFFLINE_ENV = {
    # Ни одного удалённого источника: слот без локального файла обязан
    # остаться без кадра, а не поймать случайного кандидата из сети.
    "OPENVERSE_ENABLED": "0",
    "MUSEUM_SOURCES_ENABLED": "0",
    "PIXABAY_ENABLED": "0",
    "UNSPLASH_ENABLED": "0",
    "SHELF_INDEX": "0",
    "MET_CATALOG": "0",
    "PEXELS_API_KEY": "",
    "PARALLAX": "0",          # без depth-моделей
    "VLM_ARBITER_MODE": "off",
    "AMBIENCE_BED": "0",
    "MUSIC_BED": "0",
    # Эпизод собран из ЗАЛИТЫХ ЦВЕТОМ прямоугольников — они не рассчитаны на
    # настоящую семантическую проверку. С torch/transformers в окружении
    # реальный CLIP и SigLIP2+Jina честно отклоняют такие кадры, и тест
    # путает этот отказ с тем самым «нет медиа», который создаёт нарочно
    # (медиа меньше слотов). Сквозные тесты проверяют МЕХАНИКУ main(), а не
    # качество судей — и не должны быть заложниками того, стоит ли ML-стек
    # в общем окружении сессии.
    "SMART_RELEVANCE_VETO": "0",
    "CLIP_RELEVANCE": "0",
}


def media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def build_episode(d):
    """Шесть фраз в трёх секциях, три локальных кадра на шесть слотов,
    синтетическое аудио AUDIO_SEC секунд. Возвращает путь к эпизоду."""
    media = d / "media"
    media.mkdir(parents=True)
    words = "раз два три четыре пять шесть семь восемь девять десять"
    phrases = [f"{words} фраза номер {n}." for n in range(1, N_BLOCKS + 1)]
    (d / "script.txt").write_text(
        "=== HOOK === " + "[pause]".join(phrases[:2]) + "\n\n"
        "=== BLOCK 1: Тест === " + "[pause]".join(phrases[2:4]) + "\n\n"
        "=== FINAL === " + "[pause]".join(phrases[4:]) + "\n",
        encoding="utf-8")
    # Структурные (не залитые) картинки: ahash на однотонной заливке нулевой
    # у любого цвета, и QC считал бы их дублями (см. test_smoke.py).
    cols = [(200, 40, 40), (40, 60, 200), (40, 180, 70)]
    for k, n in enumerate(MEDIA_FOR):
        img = Image.new("RGB", (1600, 900), cols[k % 3])
        dr = ImageDraw.Draw(img)
        dr.rectangle([60 + k * 40, 60, 60 + k * 40 + 300 + k * 120, 460], fill=cols[(k + 1) % 3])
        dr.ellipse([900, 300, 900 + 200 + k * 90, 300 + 180], fill=cols[(k + 2) % 3])
        img.save(media / f"{n:03d}_stock.jpg", quality=92)
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=320:duration={AUDIO_SEC}",
                    "-c:a", "libmp3lame", str(d / "audio.mp3")],
                   capture_output=True, check=True)
    return d


def run_pipeline(episode, extra_env=None, args=()):
    env = dict(os.environ, **OFFLINE_ENV, **(extra_env or {}))
    return subprocess.run([sys.executable, PIPELINE, str(episode), *args],
                          capture_output=True, text=True, timeout=900, env=env)
