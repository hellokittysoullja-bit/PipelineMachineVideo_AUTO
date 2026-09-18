"""Гейты обязаны судить ТОТ отрезок исходника, который увидит зритель.

Найдено арифметикой на реальном кандидате (18.09), а не чтением кода. Все
четыре видео-гейта (relevance, домен-гвард, контрастивное вето, резкость)
брали кадры долями ПОЛНОЙ длительности файла, а video_render() показывает
только [skip, skip+dur]. На кандидате Pexels 855260 (48.0с исходника, слот
6.0с) это давало:

    показано зрителю : 1.6 .. 7.6с
    проверено гейтами: 7.2 / 24.0 / 40.8с

Ноль проверенных точек внутри кадра. При этом ровно тот дефект, ради
которого многокадровая проверка и заведена (толпа современных зрителей на
0.96с — см. докстринг video_negative_anchor_violation), лежал ВНУТРИ
показанного окна и не проверялся никогда; а брак, найденный на 40.9с,
отклонял кандидата за кадр, которого зритель не увидит ни при каких
условиях. Ни порогом, ни списком ловушек это не лечится — гейты смотрели
не туда.

Земля здесь — РЕАЛЬНАЯ команда ffmpeg, которую собирает video_render(), а не
пересказ его формулы в тесте: пересказ проверял бы сам себя (тот же запрет,
что у Шага 7.5, п.1 в CLAUDE.md).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402

FPS = 24
SRC_SEC = 48.0
SLOT_DUR = 6.0

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="нужен настоящий ffmpeg")


@pytest.fixture(scope="module")
def long_source(tmp_path_factory):
    """48-секундный исходник — та же пропорция «файл сильно длиннее слота»,
    на которой дефект и был измерен."""
    p = str(tmp_path_factory.mktemp("win") / "src48.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c=0x203040:s=320x180:d={SRC_SEC}:r={FPS}",
         "-pix_fmt", "yuv420p", p], capture_output=True, check=True)
    return p


def _captured_render_window(src, dur, out_dir):
    """(start, frames) из НАСТОЯЩЕЙ команды рендера.

    Перехватывается subprocess.run, потому что единственный источник истины о
    показанном отрезке — аргументы -ss/-frames:v, реально ушедшие в ffmpeg.
    """
    cmds = []
    real = subprocess.run

    def spy(args, *a, **k):
        if isinstance(args, list) and args and args[0] == "ffmpeg":
            cmds.append(list(args))
        return real(args, *a, **k)

    out = os.path.join(out_dir, "rendered.mp4")
    subprocess.run = spy
    try:
        ps.video_render(src, out, dur)
    finally:
        subprocess.run = real
    # Нужен ИМЕННО кадрирующий вызов (-frames:v = вся длительность клипа),
    # а не однокадровые пробники measure_motion/detect_scene_change_offset.
    for c in cmds:
        if "-frames:v" in c and "-ss" in c:
            frames = int(c[c.index("-frames:v") + 1])
            if frames > 1:
                return float(c[c.index("-ss") + 1]), frames
    raise AssertionError(f"кадрирующий вызов ffmpeg не найден: {cmds}")


def test_window_start_equals_the_real_render_ss(long_source, tmp_path):
    """Расчёт окна обязан совпадать с тем, что рендер РЕАЛЬНО передаёт в -ss.

    Это и есть проверка «одна формула на рендер и на гейты»: разойдись они —
    гейты снова начали бы судить чужой отрезок, причём молча."""
    ss, frames = _captured_render_window(long_source, SLOT_DUR, str(tmp_path))
    start, end = ps.video_display_window(long_source, SLOT_DUR)
    assert abs(start - ss) < 0.01, (
        f"video_display_window() начинается на {start:.2f}, а рендер — на {ss:.2f}")
    shown_end = ss + frames / FPS
    assert end >= shown_end - 0.01, (
        "окно короче того, что реально покажут — часть кадра осталась бы "
        "непроверенной")


def test_every_gate_probe_lands_inside_the_shown_window(long_source, tmp_path):
    ss, frames = _captured_render_window(long_source, SLOT_DUR, str(tmp_path))
    lo, hi = ss, ss + frames / FPS
    for fracs in (ps.VIDEO_DOMAIN_GUARD_SAMPLE_FRACS,
                  ps.VIDEO_SHARPNESS_SAMPLE_FRACS):
        ats = ps.video_sample_times(long_source, fracs, SLOT_DUR)
        assert ats, "гейт остался вообще без точек сэмплирования"
        for at in ats:
            assert lo - 0.05 <= at <= hi + 0.05, (
                f"гейт смотрит на {at:.2f}с, а зритель видит {lo:.2f}..{hi:.2f}с")


def test_old_full_duration_sampling_was_outside_the_frame(long_source, tmp_path):
    """Сам дефект, зафиксированный числом: прежняя формула (доли ПОЛНОЙ
    длительности) не попадала в показанное окно ни одной точкой.

    Тест держит не код, а ПРИЧИНУ: если однажды кто-то вернёт сэмплирование
    по всей длительности, эта проверка объяснит, что именно сломалось."""
    ss, frames = _captured_render_window(long_source, SLOT_DUR, str(tmp_path))
    lo, hi = ss, ss + frames / FPS
    old = [max(0.3, min(SRC_SEC - 0.2, SRC_SEC * f))
           for f in ps.VIDEO_DOMAIN_GUARD_SAMPLE_FRACS]
    assert sum(lo <= at <= hi for at in old) == 0
    new = ps.video_sample_times(long_source, ps.VIDEO_DOMAIN_GUARD_SAMPLE_FRACS,
                                SLOT_DUR)
    assert sum(lo <= at <= hi for at in new) == len(new)


def test_without_slot_dur_behaviour_is_the_old_one(long_source):
    """Правка не имеет права менять то, чего не знает.

    visual_qc, отчёты и старые вызовы длительности слота не передают — для
    них окно обязано остаться «весь файл», то есть прежние доли."""
    ats = ps.video_sample_times(long_source, ps.VIDEO_DOMAIN_GUARD_SAMPLE_FRACS,
                                None)
    old = [max(0.3, min(SRC_SEC - 0.2, SRC_SEC * f))
           for f in ps.VIDEO_DOMAIN_GUARD_SAMPLE_FRACS]
    assert [round(a, 2) for a in ats] == [round(o, 2) for o in old]


def test_source_shorter_than_slot_is_shown_whole(tmp_path):
    """Исходник короче слота растягивается ЦЕЛИКОМ (setpts) — окно = весь файл,
    и гейт обязан проверять его до последнего кадра."""
    p = str(tmp_path / "short.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"color=c=black:s=160x90:d=3:r={FPS}", "-pix_fmt", "yuv420p", p],
                   capture_output=True, check=True)
    start, end = ps.video_display_window(p, 6.0)
    assert start == 0.0 and end >= 2.9


def test_gates_are_called_with_the_slot_duration():
    """Source-level: без проброса slot_dur вся правка — мёртвый слой.

    Ровно тот класс, которым этот репозиторий горел шесть раз: функция есть,
    и её никто не зовёт с нужным аргументом."""
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    for call in ("video_domain_guard_violation(trial, query, slot_dur=slot_dur)",
                 "video_negative_anchor_violation(trial, query, slot_dur=slot_dur)",
                 "video_sharpness_ok(trial, slot_dur=slot_dur)"):
        assert call in src, f"гейт зовётся без длительности слота: {call}"


def test_window_functions_are_in_the_candidate_gate_signature():
    """Правка меняет вердикт по тем же файлам -> обязана инвалидировать кэш,
    иначе на прогретом temp_smart/ не дойдёт до экрана вообще."""
    import inspect
    sig_src = inspect.getsource(ps.candidate_gate_signature)
    for name in ("video_display_window", "video_sample_times", "video_display_skip"):
        assert name in sig_src, f"{name} не входит в подпись гейтов"
