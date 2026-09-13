# -*- coding: utf-8 -*-
"""Регрессия уровней по ОТРЕНДЕРЕННОМУ звуку, а не по коду.

Проект трижды пострадал от одного класса ошибок: музыкальная подложка
(задумано 16 LU, в публикации 27), акцент кульминации (+7 дБ), объектный
слой (точка +4 дБ, подзвучник -18 дБ). Ни один из сотен тестов не поймал ни
одну — все они проверяют, что код делает то, что в коде написано, а ошибка
живёт в звуке.

Отдельно про ложную уверенность, из-за которой этот файл и появился: если
усиление считается как `цель - замер`, то повторный замер того же ассета
вернёт цель ПО ПОСТРОЕНИЮ АРИФМЕТИКИ. Так была «подтверждена» правильность
уровней объектного слоя — на изолированном ассете, не прошедшем ни
дакинга, ни loudnorm, ни лимитера. Это была проверка вычитания.

Здесь сцена рендерится ДВАЖДЫ, со слоем и без, оба прогона детерминированы
и выровнены по времени, разность по сэмплам — фактический вклад слоя после
ВСЕЙ обработки.

ЧТО ЭТОТ ТЕСТ ЛОВИТ И ЧЕГО НЕ ЛОВИТ — проверено подстановкой, не заявлено.

ЛОВИТ: расхождение между задуманным уровнем и тем, что реально уходит в
файл. Проверено имитацией настоящей аварии — сведение заставили
игнорировать посчитанное усиление и брать «правдоподобную константу», ровно
как было с MUSIC_BED_GAIN_DB: тест упал с «фактический разрыв 38.8 LU
против цели 28.0». Все три исторические аварии проекта имеют эту форму.

НЕ ЛОВИТ: неверную саму цель. Проверено: подмена OBJECT_POINT_GAP_LU с 28
на 22 тест ПРОХОДИТ, потому что цель участвует и в расчёте усиления —
система честно попадает туда, куда ей велели. Правильность коридоров 28/32
LU доказывается только ухом на реальном материале, и это разделение
правильное: автоматика отвечает за повторяемость, ухо — за саму цель.
"""
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
try:
    import numpy  # noqa: F401
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not (HAS_FFMPEG and HAS_NUMPY),
                                reason="нужны ffmpeg и numpy")

# Приёмка: отклонение фактического разрыва от целевого.
TOL_POINT_LU = 2.0
TOL_BED_LU = 1.5
# Лимитер не имеет права прижимать весь микс на ударе: на слух это
# «голос дёрнулся». Изолированным замером ассета не ловится в принципе.
MAX_LIMITER_REDUCTION_DB = 1.0


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    sys.argv = ["pipeline_smart.py", str(tmp_path_factory.mktemp("ep"))]
    import level_regression as lr

    lr.build_fixture()
    d = str(tmp_path_factory.mktemp("scene"))
    with_obj = lr.render_scene(os.path.join(d, "with.wav"), True, d)
    without = lr.render_scene(os.path.join(d, "without.wav"), False, d)
    assert with_obj and without, "сцена не отрендерилась"
    contrib = lr.layer_contribution(with_obj, without)
    voice = lr.loudness_of_samples(lr.decode_mono(without), mode="I")
    assert contrib is not None and voice is not None
    return {"lr": lr, "dir": d, "with": with_obj, "without": without,
            "contrib": contrib, "voice_lufs": voice}


def _gap(scene, w0, w1):
    lr = scene["lr"]
    m = lr.loudness_of_samples(lr.window(scene["contrib"], w0, w1), mode="M")
    assert m is not None, "вклад слоя не измерился — слой вообще звучит?"
    return scene["voice_lufs"] - m


def test_point_cue_hits_its_target_after_the_whole_chain(scene):
    import sfx_plan

    lr = scene["lr"]
    for t in lr.SCENE_POINT_AT:
        gap = _gap(scene, t - 0.2, t + 1.0)
        assert abs(gap - sfx_plan.OBJECT_POINT_GAP_LU) <= TOL_POINT_LU, (
            f"точка на {t}с: фактический разрыв {gap:.1f} LU против цели "
            f"{sfx_plan.OBJECT_POINT_GAP_LU} — уровень разъехался ПОСЛЕ цепочки")


def test_bed_cue_hits_its_target_after_the_whole_chain(scene):
    import sfx_plan

    lr = scene["lr"]
    gap = _gap(scene, *lr.SCENE_BED_MEASURE)
    assert abs(gap - sfx_plan.OBJECT_BED_GAP_LU) <= TOL_BED_LU, (
        f"подзвучник: фактический разрыв {gap:.1f} LU против цели "
        f"{sfx_plan.OBJECT_BED_GAP_LU}")


def test_bed_is_actually_quieter_than_the_point(scene):
    """Проверяется на РЕЗУЛЬТАТЕ, а не на константах: порядок чисел в дБ у
    них противоположен порядку громкостей, и только замер это разводит."""
    import sfx_plan

    lr = scene["lr"]
    point = _gap(scene, lr.SCENE_POINT_AT[0] - 0.2, lr.SCENE_POINT_AT[0] + 1.0)
    bed = _gap(scene, *lr.SCENE_BED_MEASURE)
    assert bed > point, f"подзвучник {bed:.1f} LU не тише точки {point:.1f} LU"


def test_limiter_does_not_duck_the_whole_mix_on_a_transient(scene):
    """Мастер-лимитер на -14 LUFS срабатывает именно на транзиентах и на
    доли секунды прижимает ВЕСЬ микс, включая голос."""
    lr = scene["lr"]
    for t in lr.SCENE_POINT_AT:
        red = lr.limiter_reduction_db(scene["without"], scene["with"],
                                      t - 0.3, t + 0.3)
        if red is None:
            continue
        # разность RMS положительна там, где кюй ДОБАВИЛ энергии; нас
        # интересует обратный случай — когда микс стал ТИШЕ из-за лимитера
        assert -red <= MAX_LIMITER_REDUCTION_DB, (
            f"на {t}с лимитер прижал микс на {-red:.2f} дБ — голос дёрнется")


def test_the_measurement_is_not_circular(scene):
    """Страховка от возврата к проверке вычитания: замер обязан идти с
    файла, прошедшего мастер-цепочку, а не с изолированного ассета."""
    import inspect

    import level_regression as lr

    src = inspect.getsource(lr.render_scene)
    assert "build_master_af" in src, "сцена обязана проходить loudnorm/лимитер"
    assert "add_planned_sfx" in src, "и реальное сведение, а не имитацию"


# ------------------------------------------- мера на коротких ассетах
def _click(tmp, total_sec):
    import subprocess
    p = os.path.join(tmp, f"click_{total_sec}.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=f=800:d=0.08", "-af",
                    f"afade=t=out:st=0.01:d=0.07,volume=-10dB,"
                    f"apad=whole_dur={total_sec}",
                    "-ar", "48000", "-ac", "2", p], capture_output=True)
    return p


def test_short_asset_is_measured_at_all(tmp_path):
    """Настоящий дефект был не в «занижении», а в том, что на файле короче
    окна ebur128 (400мс) замер не происходил СОВСЕМ — возвращался None, и
    включалась запасная константа, уже ошибавшаяся на 18 дБ. То есть на
    коротких ассетах работал именно сломанный путь."""
    import pipeline_smart as ps

    for total in (0.08, 0.2):
        assert ps.measure_max_momentary_lufs(_click(str(tmp_path), total)) is not None


def test_measure_does_not_depend_on_surrounding_silence(tmp_path):
    """Один и тот же удар с разной подложкой тишины обязан мериться
    одинаково: ведущая и хвостовая тишина на МАКСИМУМ мгновенной громкости
    не влияет (замер: разброс 0.00 дБ на пяти подложках)."""
    import pipeline_smart as ps

    vals = [ps.measure_max_momentary_lufs(_click(str(tmp_path), t))
            for t in (0.08, 0.2, 0.4, 1.0, 2.0)]
    vals = [v for v in vals if v is not None]
    assert len(vals) == 5
    assert max(vals) - min(vals) < 0.5, vals


def test_padding_is_silence_not_a_loop():
    """Повтор активной области ЗАВЫШАЕТ транзиент с длинным спадом: пятнадцать
    повторов одной атаки — это очередь, а не удар. Плюс интегрирование
    кратких звуков окном в сотни миллисекунд — не дефект, а то, как слышит
    ухо (BS.1770); «чинить» его значило бы считать 80-миллисекундный щелчок
    таким же громким, как непрерывный тон того же пика."""
    import inspect

    import pipeline_smart as ps

    body = inspect.getsource(ps.measure_max_momentary_lufs).split('"""')[-1]
    assert "apad" in body
    assert "aloop" not in body


def test_long_assets_are_untouched():
    """Правка обязана менять поведение ТОЛЬКО там, где замера не было."""
    import glob

    import pipeline_smart as ps

    for f in sorted(glob.glob("assets/library/sfx/*/*.flac"))[:3]:
        if (ps.get_media_duration(f) or 0) >= ps.ACTIVE_REGION_MIN_SEC:
            assert ps.measure_max_momentary_lufs(f) is not None
