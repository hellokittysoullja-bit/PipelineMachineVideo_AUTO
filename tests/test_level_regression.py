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


def test_parallel_measurement_of_the_same_asset_never_fails(tmp_path):
    """Дефект глубокого аудита 13.09, найден замером: временный файл под
    добивку тишиной назывался по хэшу ПУТИ, то есть одинаково на всех
    вызовах. Параллельные замеры одного ассета делили один файл, и кто
    закончил первым — удалял его из-под остальных. Замер: 8 провалов из 16
    одновременных вызовов.

    Провал здесь не безобиден: None включает запасную константу — ровно ту,
    что ошибалась на 18 дБ. То есть на многопоточном рендере уровни
    объектного слоя разъезжались случайным образом от прогона к прогону.
    """
    import concurrent.futures as cf

    import pipeline_smart as ps

    path = _click(str(tmp_path), 0.08)
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        vals = list(ex.map(lambda _: ps.measure_max_momentary_lufs(path), range(16)))
    assert all(v is not None for v in vals), f"провалов {vals.count(None)} из 16"
    assert max(vals) - min(vals) < 0.01, "один файл — один ответ"


def test_clamped_gain_is_never_reported_as_measured(tmp_path):
    """Дефект глубокого аудита 13.09: расчётное усиление обрезается
    границами [-40, 0] dB, но источник всё равно назывался `measured` —
    то есть отчёт эпизода уверял, что задуманный разрыв достигнут, тогда
    как кюй молча стоял на другом уровне. Замер на очень тихом ассете:
    расчёт +17.8 dB, выдано +0.0, источник «measured».

    У музыкальной подложки предупреждение на этот случай есть с самого
    начала — здесь его не было.
    """
    import subprocess

    import pipeline_smart as ps
    import sfx_plan

    quiet = os.path.join(str(tmp_path), "quiet.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=1000:duration=2", "-af", "volume=-40dB",
                    "-ar", "48000", "-ac", "2", quiet], capture_output=True)
    gain, src = ps.object_gain_db(quiet, sfx_plan.OBJECT_CLASS_POINT, -16.0)
    assert src == "measured_clamped", (gain, src)
    assert gain == sfx_plan.OBJECT_GAIN_MAX_DB

    loud = os.path.join(str(tmp_path), "loud.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=1000:duration=2",
                    "-ar", "48000", "-ac", "2", loud], capture_output=True)
    gain2, src2 = ps.object_gain_db(loud, sfx_plan.OBJECT_CLASS_POINT, -16.0)
    assert src2 == "measured", (gain2, src2)
    assert sfx_plan.OBJECT_GAIN_MIN_DB < gain2 < sfx_plan.OBJECT_GAIN_MAX_DB


# ------------------------------------------- п.4 локальный референс голоса
def _voice_two_halves(dirpath, loud_lufs=-14.0, quiet_lufs=-26.0, half=15.0):
    """Голос, громкость которого меняется посреди эпизода — ровно то, что
    ТЗ называет причиной: «Громкость речи гуляет по эпизоду»."""
    import subprocess
    sp = ("anoisesrc=d=%.1f:c=pink:r=48000,highpass=f=120,lowpass=f=6000,"
          "tremolo=f=0.7:d=0.9" % half)
    parts = []
    for tag, target in (("loud", loud_lufs), ("quiet", quiet_lufs)):
        p = os.path.join(dirpath, tag + ".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", sp,
                        "-af", f"loudnorm=I={target}:TP=-1.5",
                        "-ar", "48000", "-ac", "2", p], capture_output=True)
        parts.append(p)
    out = os.path.join(dirpath, "voice_two_halves.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", parts[0], "-i", parts[1],
                    "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1",
                    "-ar", "48000", "-ac", "2", out], capture_output=True)
    return out


def test_level_follows_the_speech_next_to_the_cue_not_the_episode_average(tmp_path):
    """Замер, ради которого п.4 и делался. Голос: 15с на -14 LUFS, затем
    15с на -26. Средняя по эпизоду -16.3 (интеграл энергетически взвешен,
    поэтому она липнет к громкой половине).

    По средней оба кюя получили бы ОДНО усиление -9.3 дБ, то есть во второй
    половине звук шёл бы на 11.4 дБ громче окружающей речи — удар поверх
    тихой реплики. По локальному референсу: -7.2 дБ в громкой половине и
    -18.7 в тихой, разница 11.5 дБ, ровно на величину перепада голоса.
    """
    import level_regression as lr
    import pipeline_smart as ps
    import sfx_plan

    lr.build_fixture()
    v = _voice_two_halves(str(tmp_path))
    ep = ps.measure_integrated_lufs(v)
    assert ep is not None

    loud_ref, loud_kind = ps.local_voice_lufs(v, 5.0, ep)
    quiet_ref, quiet_kind = ps.local_voice_lufs(v, 22.0, ep)
    assert loud_kind == quiet_kind == "local"
    assert loud_ref - quiet_ref > 9.0, (loud_ref, quiet_ref)

    g_loud, _ = ps.object_gain_db(lr.SCENE_POINT, sfx_plan.OBJECT_CLASS_POINT, loud_ref)
    g_quiet, _ = ps.object_gain_db(lr.SCENE_POINT, sfx_plan.OBJECT_CLASS_POINT, quiet_ref)
    g_avg, _ = ps.object_gain_db(lr.SCENE_POINT, sfx_plan.OBJECT_CLASS_POINT, ep)
    assert g_quiet < g_avg < g_loud, (g_quiet, g_avg, g_loud)
    # разрыв с МЕСТНОЙ речью одинаков в обеих половинах — в этом вся суть
    assert abs((loud_ref - g_loud) - (quiet_ref - g_quiet)) < 0.5


def test_local_reference_falls_back_on_the_same_scale_it_replaces(tmp_path):
    """Локальный и запасной референс — ОДНА мера (интегральная громкость).
    Меряй их по-разному, и сам откат на запасной сдвигал бы уровень кюя, то
    есть появлялся бы скачок ровно там, где данных не хватило."""
    import inspect
    import subprocess

    import pipeline_smart as ps

    assert "ebur128" in inspect.getsource(ps._local_voice_lufs_cached)
    assert "I:" in inspect.getsource(ps._local_voice_lufs_cached)

    sil = os.path.join(str(tmp_path), "sil.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "anullsrc=r=48000:cl=stereo", "-t", "30", sil],
                   capture_output=True)
    # окно без речи -> средняя по эпизоду, а не уровень, выведенный из тишины
    assert ps.local_voice_lufs(sil, 15.0, -16.0) == (-16.0, "episode")
    # нет и средней -> честное «нечем мерить», дальше включится константа
    assert ps.local_voice_lufs(sil, 15.0, None) == (None, "none")


def test_unreadable_voice_does_not_take_the_whole_sfx_layer_with_it(tmp_path):
    """Найдено при живом прогоне п.4: get_media_duration() ПАДАЕТ на
    нечитаемом файле (check=True), хотя докстринги соседних функций
    называют её fail-open. Без перехвата исключение уходило в общий
    try/except планировщика, и из ролика исчезал ВЕСЬ звуковой слой — из-за
    одного уровня.
    """
    import pipeline_smart as ps

    missing = os.path.join(str(tmp_path), "нет-такого.wav")
    assert ps.local_voice_lufs(missing, 5.0, -16.0) == (-16.0, "episode")
    assert ps.local_voice_lufs(missing, 5.0, None) == (None, "none")


def test_one_unreadable_asset_costs_one_cue_not_the_whole_layer(tmp_path):
    """Замер 13.09, не гипотеза. Функция сведения объявлена fail-open
    («ошибка ffmpeg -> исходный микс без изменений»), но гранулярность у
    этого обещания была неверная: один нечитаемый файл ронял ВЕСЬ вызов, и
    вместе с ним из ролика исчезали все эффекты эпизода — переходы глав,
    тики плашек, объектные звуки. В логе оставалась одна строка с обрезком
    ошибки ffmpeg, по которой масштаб потери не виден.

    Проверка ДО сведения: файл не просто существует, а читается.
    """
    import subprocess

    import level_regression as lr
    import pipeline_smart as ps

    lr.build_fixture()
    mix = os.path.join(str(tmp_path), "mix.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "anoisesrc=d=20:c=pink:r=48000",
                    "-ar", "48000", "-ac", "2", mix], capture_output=True)
    bad = os.path.join(str(tmp_path), "corrupt.flac")
    with open(bad, "wb") as f:
        f.write(b"\x00" * 4096)

    cues = [{"time": 3.0, "asset": lr.SCENE_POINT, "gain_db": -20.0},
            {"time": 9.0, "asset": lr.SCENE_POINT, "gain_db": -20.0},
            {"time": 15.0, "asset": bad, "gain_db": -20.0}]
    out = ps.add_planned_sfx(mix, cues, 20.0, os.path.join(str(tmp_path), "m.wav"))
    assert out != mix, "годные кюи обязаны уцелеть"
    assert ps.get_media_duration(out) > 19.0


def test_corrupt_asset_does_not_take_the_planner_down(tmp_path):
    """Тот же корень выше по течению: get_media_duration() на битом файле
    бросает KeyError (ffprobe отдаёт пустой JSON), исключение уходило в
    общий try/except планировщика — и план становился пустым целиком.
    Замер: 'ПЛАНИРОВЩИК ЦЕЛИКОМ ПАДАЕТ: KeyError'.
    """
    import pipeline_smart as ps
    import sfx_plan

    bad = os.path.join(str(tmp_path), "corrupt.flac")
    with open(bad, "wb") as f:
        f.write(b"\x00" * 4096)
    assert ps.media_duration_or_none(bad) is None
    gain, src = ps.object_gain_db(bad, sfx_plan.OBJECT_CLASS_POINT, -16.0)
    assert src == "fallback_constant"

    blocks = [{"text": "ф " * 10, "words": 10, "section": "BLOCK %d" % (i // 2 + 1),
               "sfx": [{"name": "x", "word_pos": 3}], "hush": False,
               "stat": None, "is_climax": False, "pause_after": 0.8}
              for i in range(4)]
    resolver = lambda n, at=None: (bad, 0.6, sfx_plan.OBJECT_CLASS_POINT, gain, src)
    acc, _ = sfx_plan.plan_sfx_cues(blocks, [0.0, 10.0, 20.0, 30.0], [8.0] * 4, 60.0,
                                    chapter_variants=(("/x/t.flac", 0.4),),
                                    object_asset_for=resolver)
    # переход главы не имеет к битому ассету никакого отношения и обязан уцелеть
    assert any(c["kind"] == "chapter" for c in acc)
