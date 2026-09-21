#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""«Лучший из плохих» удалён как исход — проверено НАСТОЯЩИМ рендером.

Прямая жалоба владельца: короткий тестовый эпизод дал контактный лист, где
явный брак (frame_verifier сам отклонил всех кандидатов) всё равно попадал
на экран — карточка спасала только ОДИН слот из пяти, остальные упирались
в бюджет (`FALLBACK_CARD_MAX_SHARE`) и молча показывали то, что система
сама же назвала негодным.

Почему тест сквозной, а не юнит: решение живёт внутри main() на тысячи
строк и завязано на перенос длительности между клипами. Единственная
честная проверка — собрать ролик и измерить его: если перенос
арифметически неверен, final.mp4 разойдётся с аудио, и это увидит ffprobe,
а не мнение автора.

Слоты остаются без кадра ЕСТЕСТВЕННО, без тест-хуков: локальных файлов
меньше, чем слотов, а все удалённые источники выключены флагами. Никакой
сети, никаких ключей — прогон детерминирован.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py")

AUDIO_SEC = 14.0   # короче некуда: при 6 фразах клипы уже по ~2с, а суита не должна удваиваться из-за одного теста
N_BLOCKS = 6          # шесть фраз, разделённых [pause]
MEDIA_FOR = (1, 2, 5)  # 1-based имена файлов: media/001_*, 002_*, 005_*

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe не найдены в PATH",
)

OFFLINE_ENV = {
    # Ни одного удалённого источника: слот без локального файла обязан
    # остаться без кадра, а не поймать случайного кандидата из сети.
    "OPENVERSE_ENABLED": "0",
    "MUSEUM_SOURCES_ENABLED": "0",
    "PIXABAY_ENABLED": "0",
    "UNSPLASH_ENABLED": "0",
    "SHELF_INDEX": "0",
    "MET_CATALOG": "0",
    "COMMONS_SOURCE": "0",
    "PEXELS_API_KEY": "",
    "PARALLAX": "0",          # без depth-моделей
    "VLM_ARBITER_MODE": "off",
    "AMBIENCE_BED": "0",
    "MUSIC_BED": "0",
    "FRAME_VERIFIER": "0",    # нет ANYMODEL_API_KEY, но выключаем явно
    # Этот тест собирает ролик из ЗАЛИТЫХ ЦВЕТОМ прямоугольников — они не
    # рассчитаны на настоящую семантическую проверку. С torch/transformers,
    # установленными в окружение (для другой работы в этой же сессии),
    # реальный CLIP честно отклоняет такие кадры, и тест путает этот отказ
    # с тем самым «нет медиа», который он же и создаёт нарочно (media
    # меньше слотов) — тест обязан проверять АРИФМЕТИКУ переноса
    # длительности, а не быть заложником того, стоит ли ML-стек в общем
    # окружении сессии.
    "CLIP_RELEVANCE": "0",
}


def _dur(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


@pytest.fixture
def episode(tmp_path):
    d = tmp_path / "absorb_ep"
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


def _run(episode, extra_env):
    env = dict(os.environ, **OFFLINE_ENV, **extra_env)
    return subprocess.run([sys.executable, PIPELINE, str(episode)],
                          capture_output=True, text=True, timeout=900, env=env)


def test_absorbed_slots_keep_the_timeline_and_never_show_bad_media(episode):
    """Слоты без проверенного кадра поглощаются соседом, и ролик остаётся
    ровно той же длины, что аудио: перенос длительности арифметически цел."""
    r = _run(episode, {"NEVER_SHOW_KNOWN_BAD": "1"})
    assert r.returncode in (0, 2), f"сборка упала:\n{r.stdout[-3000:]}\n{r.stderr[-1500:]}"

    final = episode / "final.mp4"
    assert final.exists(), f"final.mp4 не собран\n{r.stdout[-3000:]}"
    got = _dur(str(final))
    assert abs(got - AUDIO_SEC) <= 0.5, (
        f"длительность {got:.2f}с против аудио {AUDIO_SEC}с — перенос "
        f"длительности от поглощённых слотов потерял или добавил время")

    report = episode / "media_plan" / "absorbed_slots_report.json"
    assert report.exists(), "нет absorbed_slots_report.json — слоты исчезли молча"
    data = json.load(open(report, encoding="utf-8"))
    absorbed = {row["index"] for row in data["misses"]}
    assert absorbed, f"ни один слот не поглощён, хотя медиа меньше слотов\n{r.stdout[-2000:]}"

    # Ни одного повтора и ни одной карточки: владелец разрешил только
    # продление соседнего проверенного кадра.
    assert "[карточка]" not in r.stdout, "появилась карточка, хотя разрешено только продление"
    cards = episode / "media_plan" / "fallback_cards_report.json"
    if cards.exists():
        assert not json.load(open(cards, encoding="utf-8"))["misses"], \
            "карточки не разрешены при NEVER_SHOW_KNOWN_BAD=1"

    manifest = json.load(open(episode / "media_plan" / "render_manifest.json",
                              encoding="utf-8"))
    statuses = {row["status"] for row in manifest["clips"]} if isinstance(manifest, dict) \
        else {row["status"] for row in manifest}
    assert "absorbed" in statuses, "манифест не отражает поглощение"


def test_flag_off_restores_the_old_behaviour(episode):
    """Откат должен быть настоящим: со снятым флагом слоты закрываются как
    раньше (повтор из папки/карточка), поглощённых нет вовсе."""
    r = _run(episode, {"NEVER_SHOW_KNOWN_BAD": "0"})
    assert r.returncode in (0, 2), f"сборка упала:\n{r.stdout[-3000:]}"
    assert (episode / "final.mp4").exists()
    report = episode / "media_plan" / "absorbed_slots_report.json"
    if report.exists():
        assert not json.load(open(report, encoding="utf-8"))["misses"], \
            "при снятом флаге поглощений быть не должно"
