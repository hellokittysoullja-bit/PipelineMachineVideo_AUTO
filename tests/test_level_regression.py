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


# --------------------------------- точечный кюй обязан стоять в ТИШИНЕ
SILENCE_FLOOR_DB = -40.0      # порог тишины, тот же класс, что у silencedetect
MIN_VERSION_SPREAD_DB = 3.0   # ниже этого версии коридора неразличимы на слух


def _voice_with_real_pauses(dirpath, bursts=6, burst=5.5, pause=1.2, lufs=-16.0):
    """Голос с НАСТОЯЩЕЙ тишиной между фразами.

    Прежняя фикстура (`level_scene/voice.flac`) — розовый шум с tremolo, и
    замер показал, что настоящих пауз в ней НЕТ ВООБЩЕ: `silencedetect` при
    -40 dB и -30 dB находит ноль провалов, первый появляется только на
    -25 dB. Поставить туда точечный кюй «в паузу» физически некуда, и
    именно поэтому собранная на ней демо-лента оказалась бессмысленной.
    """
    import subprocess
    parts = []
    for i in range(bursts):
        p = os.path.join(dirpath, f"b{i}.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"anoisesrc=d={burst}:c=pink:r=48000",
                        "-af", "highpass=f=120,lowpass=f=6000,"
                               f"afade=t=in:d=0.05,afade=t=out:st={burst-0.05}:d=0.05,"
                               f"apad=pad_dur={pause}",
                        "-ar", "48000", "-ac", "2", p], capture_output=True)
        parts.append(p)
    lst = os.path.join(dirpath, "parts.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    out = os.path.join(dirpath, "voice_pauses.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-af", f"loudnorm=I={lufs}:TP=-1.5",
                    "-ar", "48000", "-ac", "2", out], capture_output=True)
    return out


def _silences(path, floor_db=SILENCE_FLOOR_DB, min_sec=0.3):
    import re
    import subprocess
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-af",
                        f"silencedetect=noise={floor_db}dB:d={min_sec}",
                        "-f", "null", "-"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr or "")]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr or "")]
    return list(zip(st, en))


def _level_db_at(path, t, win=0.10):
    """Уровень речи в окне вокруг момента t, dBFS."""
    import numpy as np

    import level_regression as lr
    x = lr.window(lr.decode_mono(path), max(0.0, t - win / 2), t + win / 2)
    if x.size == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12))


def test_point_cue_never_starts_on_top_of_speech(tmp_path):
    """ПРЯМАЯ ПРИЧИНА (замер владельца, подтверждён): собранная демо-лента с
    разрывом 24 / 28 / 32 LU — то есть усилениями, отличающимися на 8 дБ —
    различалась НЕ БОЛЕЕ чем на 0.133 дБ в любом 100мс окне. Все пять кюев
    стартовали поверх речи, и она их полностью маскировала: кюй на 28 LU
    ниже речи добавляет к миксу около 0.04 дБ, то есть менять его уровень
    под речью бессмысленно при любом числе.

    Инвариант: точечный кюй начинается там, где речи НЕТ. Проверяется по
    реальному уровню дорожки голоса в момент старта, а не по намерению
    планировщика — та же дисциплина «источник истины — сэмплы, а не отчёт».
    """
    import sfx_plan

    voice = _voice_with_real_pauses(str(tmp_path))
    sil = _silences(voice)
    assert sil, "фикстура обязана содержать настоящие паузы"

    # блоки строятся ПО РЕАЛЬНЫМ паузам этой дорожки: онсет — конец паузы,
    # длительность речи — до начала следующей
    starts, weights = [], []
    for i, (s0, s1) in enumerate(sil):
        starts.append(s1)
        nxt = sil[i + 1][0] if i + 1 < len(sil) else s1 + 2.0
        weights.append(max(0.1, nxt - s1))
    # Фразы разведены дальше OBJECT_MIN_GAP_SEC намеренно: при более
    # плотной сетке фильтр минимального интервала СЛУЧАЙНО отбирал ровно
    # те кюи, что и так стояли в паузе, и контрольный прогон со снятым
    # правилом проходил. Проверка обязана ловить дефект, а не спасаться
    # побочным эффектом соседнего ограничителя.
    #
    # Половина тегов — В НАЧАЛЕ фразы (для них тишина рядом есть), половина —
    # в СЕРЕДИНЕ. Смесь обязательна: при теге только у начала фразы прежнее,
    # неверное размещение `anchor - pre_lap` СЛУЧАЙНО тоже попадает в паузу,
    # и тест был бы зелёным по построению. Проверено снятием правила: с одними
    # только начальными тегами он проходил и со сломанным кодом.
    blocks = [{"text": "ф " * 10, "words": 10, "section": "BLOCK 1",
               "sfx": [{"name": "hammer", "word_pos": 1 if i % 2 == 0 else 6}],
               "hush": False, "stat": None, "is_climax": False,
               "pause_after": 0.8}
              for i in range(len(starts))]
    asset = lambda n, at=None: ("/x/a.flac", 0.6, sfx_plan.OBJECT_CLASS_POINT,
                                -20.0, "measured")
    acc, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, 60.0,
                                    object_asset_for=asset)
    pts = [c for c in acc if c.get("cls") == sfx_plan.OBJECT_CLASS_POINT]
    assert pts, "ни один точечный кюй не поставлен — проверять нечего"
    loud = [(round(c["time"], 2), round(_level_db_at(voice, c["time"]), 1))
            for c in pts if _level_db_at(voice, c["time"]) > SILENCE_FLOOR_DB]
    assert not loud, (
        f"точечный кюй стартует поверх речи (момент, уровень дБ): {loud}; "
        f"порог тишины {SILENCE_FLOOR_DB} dB")


def test_corridor_versions_are_actually_distinguishable(tmp_path):
    """Лента для прослушивания обязана РАЗЛИЧАТЬСЯ, иначе уши не помогут.

    Замер владельца на первой версии ленты: варианты с разрывом 24 / 28 /
    32 LU — усиления, отличающиеся на 8 дБ — различались не более чем на
    0.133 дБ в любом 100мс окне. Причина не в уровнях, а в размещении: все
    кюи стояли поверх речи, и она маскировала их полностью (кюй на 28 LU
    ниже речи добавляет к миксу около 0.04 дБ). Прослушивание такой ленты не
    могло дать ответа ни при каком старании.

    После правки размещения тот же замер даёт 8.00 дБ — ровно разницу
    усилений, потому что в паузе кюй ничем не маскирован.

    Порог 3.0 дБ — не оптимум, а граница осмысленности: ниже этого разница
    между версиями коридора перестаёт быть предметом слухового выбора.
    """
    import subprocess

    import numpy as np

    import level_regression as lr
    import pipeline_smart as ps
    import sfx_plan

    lr.build_fixture()
    voice = _voice_with_real_pauses(str(tmp_path))
    sil = _silences(voice)
    vl = ps.measure_integrated_lufs(voice)
    dur = ps.get_media_duration(voice)
    assert vl is not None and sil

    starts, weights = [], []
    for i, (s0, s1) in enumerate(sil):
        starts.append(s1)
        nxt = sil[i + 1][0] if i + 1 < len(sil) else s1 + 2.0
        weights.append(max(0.1, nxt - s1))
    blocks = [{"text": "ф " * 10, "words": 10, "section": "BLOCK 1",
               "sfx": [{"name": "point", "word_pos": 1}], "hush": False,
               "stat": None, "is_climax": False, "pause_after": 0.8}
              for _ in starts]

    asset_lufs = ps.measure_max_momentary_lufs(lr.SCENE_POINT)
    assert asset_lufs is not None

    def render(gap):
        g = max(sfx_plan.OBJECT_GAIN_MIN_DB,
                min(sfx_plan.OBJECT_GAIN_MAX_DB, vl - gap - asset_lufs))
        resolver = lambda n, at=None: (lr.SCENE_POINT,
                                       ps.get_media_duration(lr.SCENE_POINT) or 0.6,
                                       sfx_plan.OBJECT_CLASS_POINT, round(g, 2),
                                       "measured")
        acc, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, dur,
                                        object_asset_for=resolver)
        cues = [c for c in acc if c["kind"] == "object"]
        pre = os.path.join(str(tmp_path), f"pre{int(gap)}.wav")
        mixed = ps.add_planned_sfx(voice, cues, dur, pre)
        out = os.path.join(str(tmp_path), f"out{int(gap)}.flac")
        af = ps.build_master_af(None, max(0.0, dur - 2.0), 0.4)
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mixed, "-af", af,
                            "-t", f"{dur:.3f}", "-ar", "48000", "-ac", "2", out],
                           capture_output=True)
        assert r.returncode == 0
        return out, cues

    loud_path, cues = render(24.0)
    quiet_path, _ = render(32.0)
    assert cues, "планировщик не поставил ни одного кюя — сравнивать нечего"

    a, b = lr.decode_mono(loud_path), lr.decode_mono(quiet_path)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    spreads = []
    for c in cues:
        t0 = c["time"]
        t1 = t0 + max(0.25, float(c.get("asset_dur") or 0.25))
        wa, wb = lr.window(a, t0, t1), lr.window(b, t0, t1)
        if wa.size == 0 or wb.size == 0:
            continue
        ra = float(np.sqrt((wa ** 2).mean())) + 1e-12
        rb = float(np.sqrt((wb ** 2).mean())) + 1e-12
        spreads.append(abs(20 * np.log10(ra / rb)))
    assert spreads
    assert min(spreads) >= MIN_VERSION_SPREAD_DB, (
        f"версии коридора неразличимы: минимальный разброс в окне кюя "
        f"{min(spreads):.2f} дБ при пороге {MIN_VERSION_SPREAD_DB} дБ "
        f"(на первой, неверно размещённой ленте было 0.13 дБ)")


# ------------------------------------ ЧЕТВЁРТАЯ авария того же класса (16.09)
# Переход главы и тик плашки — узнано при разборе жалобы «в готовом
# рендере не слышно вообще никакого SFX». Тот же диагноз, что и у трёх
# аварий из шапки файла (пик ассета — не его громкость), плюс отдельный,
# более грубый дефект: переход НЕ ИМЕЛ своего gain_db вовсе и молча занимал
# ЧУЖУЮ константу.


def test_chapter_without_gain_db_no_longer_borrows_the_plate_constant(tmp_path):
    """Реальный найденный баг: sfx_plan.plan_sfx_cues() никогда не пишет
    `gain_db` кюю перехода, а add_planned_sfx() при отсутствии `gain_db`
    подставляла SFX_PLATE_GAIN_DB ДЛЯ ЛЮБОГО вида кюя — то есть переход
    главы годами звучал на -16 дБ вместо задуманных -12 (в media_plan/
    sfx_plan.json готового рендера у kind="chapter" поля gain_db не было
    вовсе). Мера прямая: тон известной амплитуды без gain_db обязан выйти
    на SFX_CHAPTER_GAIN_DB, а не на SFX_PLATE_GAIN_DB — разница 4 дБ,
    видна по пику сэмплов без всякого ebur128.
    """
    import subprocess

    import numpy as np
    import pipeline_smart as ps
    import level_regression as lr

    silence = os.path.join(str(tmp_path), "silence.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=stereo", "-t", "3",
                    silence], capture_output=True)
    tone = os.path.join(str(tmp_path), "tone.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "sine=f=1000:d=1.0", "-ar", "48000", "-ac", "2",
                    tone], capture_output=True)

    # Пик тона у lavfi sine — НЕ 0 dBFS (замер: -18 дБ), поэтому усиление
    # вычисляется относительно СОБСТВЕННОГО пика исходника, замеренного
    # тем же decode_mono, что и результат — иначе сравнение зависит от
    # произвольного значения амплитуды генератора, а не от факта «какой
    # gain_db реально применился».
    tone_peak = float(np.max(np.abs(lr.decode_mono(tone))))
    tone_db = 20 * np.log10(tone_peak + 1e-12)

    cue = {"kind": "chapter", "time": 1.0, "asset": tone}  # НЕТ gain_db
    out = ps.add_planned_sfx(silence, [cue], 3.0,
                             os.path.join(str(tmp_path), "out.wav"))
    assert out != silence, "кюй не наложился вообще"

    samples = lr.window(lr.decode_mono(out), 1.05, 1.5)  # без фронта тона
    peak = float(np.max(np.abs(samples)))
    got_gain = 20 * np.log10(peak + 1e-12) - tone_db
    assert abs(got_gain - ps.SFX_CHAPTER_GAIN_DB) < 1.0, (
        f"чужой кюй без gain_db наложился с усилением {got_gain:.1f} дБ — "
        f"похоже на SFX_PLATE_GAIN_DB={ps.SFX_PLATE_GAIN_DB}, а не на "
        f"свой SFX_CHAPTER_GAIN_DB={ps.SFX_CHAPTER_GAIN_DB}")
    assert abs(got_gain - ps.SFX_PLATE_GAIN_DB) > 2.0


def test_run_sfx_director_measures_chapter_and_plate_gain():
    """Источник обязан звать sfx_cue_gain_db() на ОБА вида — иначе правка
    молча откатится к тому состоянию, где gain_db для перехода не
    проставлялся вовсе."""
    import inspect

    import pipeline_smart as ps

    src = inspect.getsource(ps.run_sfx_director)
    assert "sfx_cue_gain_db(" in src
    assert '"chapter"' in src and '"plate"' in src


def _crest_burst(tmp, dur, vol=-10.0, pad=0.5):
    """Короткий шумовой всплеск на объявленном пике `vol` dBFS — та же
    ФОРМА несоответствия, что у настоящих ассетов библиотеки (пик задан
    нормировкой, мгновенная громкость от него далека), не копия реального
    файла. Дольше всплеск -> выше мгновенная громкость при том же пике —
    ровно то поведение, которое отличает транзиент от устойчивого сигнала.
    """
    import subprocess

    p = os.path.join(tmp, f"burst_{dur}.wav")
    tail_start = max(0.0, dur - dur * 0.3)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"anoisesrc=d={dur}:c=white:r=48000", "-af",
         f"afade=t=out:st={tail_start:.4f}:d={dur * 0.3:.4f},"
         f"volume={vol}dB,apad=whole_dur={pad}",
         "-ar", "48000", "-ac", "2", p], capture_output=True)
    return p


@pytest.mark.parametrize("kind,dur", [("chapter", 0.025), ("plate", 0.008)])
def test_cue_gain_targets_measured_loudness_not_the_old_peak_guess(tmp_path, kind, dur):
    """Реальный найденный баг: SFX_CHAPTER_GAIN_DB/SFX_PLATE_GAIN_DB
    рассчитаны из допущения «пик ассета = его громкость» (верно для
    процедурного генератора, scripts/generate_sfx_pack.py; неверно для
    реальных записей библиотеки, у которых теперь приоритет — замер живых
    ассетов этого канала: chapter_turn пик -9.9 dBFS при мгновенной
    громкости всего -24.3 LUFS, plate_tick пик -10.0 при -29.5 LUFS).
    Применённая по этому допущению константа проваливалась на 14-20 дБ
    мимо творческой цели, которую сама же и объявляла в докстринге.
    """
    import subprocess

    import pipeline_smart as ps

    asset = _crest_burst(str(tmp_path), dur)
    measured_before = ps.measure_max_momentary_lufs(asset)
    assert measured_before is not None

    gain, src = ps.sfx_cue_gain_db(asset, kind)
    assert src == "measured", (gain, src)

    old_flat = ps.SFX_CHAPTER_GAIN_DB if kind == "chapter" else ps.SFX_PLATE_GAIN_DB
    target = ps.SFX_CUE_TARGET_LUFS[kind]

    old_result = measured_before + old_flat
    assert target - old_result > 8.0, (
        f"старая константа {old_flat} dB на этой форме сигнала давала бы "
        f"{old_result:.1f} LUFS против цели {target} — регрессия ожидалась "
        f"большой (в реальном рендере вышло 14-20 дБ), а вышла всего "
        f"{target - old_result:.1f} дБ")

    gained = os.path.join(str(tmp_path), f"gained_{kind}.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", asset, "-af",
                    f"volume={gain}dB", "-ar", "48000", "-ac", "2",
                    gained], capture_output=True)
    after = ps.measure_max_momentary_lufs(gained)
    assert after is not None
    assert abs(after - target) <= 1.5, (
        f"{kind}: усиленный ассет даёт {after:.1f} LUFS против цели {target}")


# ------------------------------------ ПЯТАЯ авария того же класса (17.09)
# Атмосферный слой. Найдена не по жалобе, а живым прогоном при проверке
# двух других фиксов: первый в истории проекта эпизод с ДВУМЯ разными
# видами атмосферы (forge_fire + wind_open) за раз, поэтому промах раньше
# было физически негде заметить — на одном виде за эпизод усиление
# по определению получается «правильным» относительно самого себя.


def _noise_at(tmp, name, lufs_target, dur=6.0):
    """Синтетический шум заданной ИНТЕГРАЛЬНОЙ громкости — приближение
    двух реальных ситуаций (тихая потрескивающая запись / стационарный
    широкополосный шум), не копия конкретного файла библиотеки."""
    import subprocess

    p = os.path.join(tmp, f"{name}.flac")
    # Подбор через loudnorm по объявленной цели — тот же приём, что уже
    # использует build_fixture() в level_regression.py для голоса сцены.
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"anoisesrc=d={dur}:c=pink:r=48000", "-af",
         f"loudnorm=I={lufs_target}:TP=-3.0", "-ar", "48000", "-ac", "2", p],
        capture_output=True)
    return p


def test_ambience_segment_gain_targets_the_common_loudness(tmp_path):
    """Реальный найденный баг: библиотека нормирует файлы к ОДИНАКОВОМУ
    ПИКУ (-12 dBFS), а не к одинаковой ГРОМКОСТИ. Замер живых ассетов
    канала: forge_fire (потрескивание, редкие всплески) -45…-53 LUFS,
    wind_open (стационарный шум) -29…-38 LUFS — разрыв до 24 дБ при одном
    и том же пике. Оба синтетических файла ниже, до фикса, звучали бы с
    этим же разрывом; после — оба должны лечь в целевую точку.
    """
    import pipeline_smart as ps

    quiet = _noise_at(str(tmp_path), "quiet", -50.0)
    loud = _noise_at(str(tmp_path), "loud", -30.0)

    before_gap = abs(ps.measure_integrated_lufs(loud) - ps.measure_integrated_lufs(quiet))
    assert before_gap > 15.0, f"фикстура не воспроизводит реальный разрыв: {before_gap:.1f} дБ"

    for path in (quiet, loud):
        gain, src = ps.ambience_segment_gain_db(path)
        assert src == "measured", (path, gain, src)
        gained = path + ".gained.wav"
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-af",
                        f"volume={gain}dB", "-ar", "48000", "-ac", "2",
                        gained], capture_output=True)
        after = ps.measure_integrated_lufs(gained)
        assert after is not None
        assert abs(after - ps.AMBIENCE_SEGMENT_TARGET_LUFS) <= 1.0, (
            f"{os.path.basename(path)}: {after:.1f} LUFS против цели "
            f"{ps.AMBIENCE_SEGMENT_TARGET_LUFS}")


def test_two_ambience_kinds_no_longer_differ_by_20db(tmp_path, monkeypatch):
    """Сквозная проверка через РЕАЛЬНУЮ _ambience_segment(), а не только
    через голую формулу усиления — тот же принцип, что и у остальных
    тестов этого файла: код, который решает, а не код, который считает.
    """
    import pipeline_smart as ps

    quiet = _noise_at(str(tmp_path), "quiet2", -50.0)
    loud = _noise_at(str(tmp_path), "loud2", -30.0)
    libs = {"kind_a": [quiet], "kind_b": [loud]}
    monkeypatch.setattr(ps, "library_sounds",
                        lambda kind, name: libs.get(name, []))

    out_a = ps._ambience_segment("kind_a", 5.0, 0, os.path.join(str(tmp_path), "seg_a.wav"))
    out_b = ps._ambience_segment("kind_b", 5.0, 0, os.path.join(str(tmp_path), "seg_b.wav"))
    assert out_a and out_b, "оба сегмента обязаны собраться"

    lufs_a = ps.measure_integrated_lufs(out_a)
    lufs_b = ps.measure_integrated_lufs(out_b)
    gap = abs(lufs_a - lufs_b)
    assert gap < 6.0, (
        f"два вида атмосферы разной природной громкости всё ещё расходятся "
        f"на {gap:.1f} дБ после общей нормировки (было бы ~20 дБ без фикса: "
        f"kind_a={lufs_a:.1f}, kind_b={lufs_b:.1f})")


def test_synthesized_three_layer_ambience_is_not_renormalized():
    """Пред-нормализация — ТОЛЬКО для единственной библиотечной записи.
    Три синтезированных слоя (low/mid/high) специально сбалансированы
    друг относительно друга генератором — нормировать их по отдельности
    к одной точке значило бы сломать этот баланс, а не исправить его."""
    import inspect

    import pipeline_smart as ps

    src = inspect.getsource(ps._ambience_segment)
    assert "len(layers) == 1" in src
    assert "normalize_layers" in src
