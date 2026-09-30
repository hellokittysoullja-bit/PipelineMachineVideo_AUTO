#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""«Лучший из плохих» удалён как исход — проверено НАСТОЯЩИМ рендером.

Замер, ради которого правка сделана (золотой набор эпизода 01, 40 кадров с
вердиктами): **8 браков из 17 гейты забраковали САМИ**, и кадры всё равно
ушли в опубликованный ролик, потому что слот не имел права остаться пустым.
Система знала правильный ответ и не могла им воспользоваться.

Почему тест сквозной, а не юнит: решение живёт внутри main() на 1900 строк
и завязано на перенос длительности между клипами. Единственная честная
проверка — собрать ролик и измерить его: если перенос арифметически неверен,
final.mp4 разойдётся с аудио, и это увидит ffprobe, а не мнение автора.

Слоты остаются без кадра ЕСТЕСТВЕННО, без тест-хуков: локальных файлов
меньше, чем слотов, а все удалённые источники выключены флагами. Никакой
сети, никаких ключей — прогон детерминирован.
"""
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _episode_factory import AUDIO_SEC, build_episode, media_duration, run_pipeline  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe не найдены в PATH",
)


def _dur(path):
    return media_duration(path)


@pytest.fixture
def episode(tmp_path):
    return build_episode(tmp_path / "absorb_ep")


def _run(episode, extra_env):
    return run_pipeline(episode, extra_env)


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
    # Схема общая для всех слотовых отчётов эпизода (merge_slot_report):
    # ключ `misses`, а не `slots`.
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
