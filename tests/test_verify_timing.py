"""Измеренная проверка тайминга — по готовому файлу, а не по модели монтажа.

Зачем эта проверка вообще появилась: media_plan/phrase_timeline.json считает
дрейф между `visual_start_sec` и `speech_onset_sec`, но ПЕРВОЕ вычисляет
hook_visual_starts() — модель того, как xfade_chain() сожмёт таймлайн. То есть
дрейф там — разница двух величин, посчитанных одним и тем же кодом из одних и
тех же данных. Протокол канала (CLAUDE.md, п.1) запрещает считать это
проверкой. Здесь источник истины — пиксели готового файла.

Материал теста строится ТЕМ ЖЕ zoompan-выражением, что и рендер канала, с
ИЗВЕСТНЫМИ моментами резов — иначе тест проверял бы детектор на материале,
которого в проде не бывает.

Отдельно заперт реальный промах, найденный при калибровке: и готовый
scene-детектор ffmpeg, и первая версия этой кривой считали разницу по
ЯРКОСТИ — и обе не увидели рез, у которого яркость по обе стороны одинаковая.
"""
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import verify_timing as vt  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="нужен ffmpeg")


def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def _still(path, base, seed):
    from PIL import Image
    import random
    rnd = random.Random(seed)
    im = Image.new("RGB", (320, 180))
    px = im.load()
    for y in range(180):
        for x in range(320):
            n = rnd.randint(-18, 18)
            px[x, y] = tuple(max(0, min(255, c + n + int(20 * ((x // 40 + y // 22 + seed) % 2))))
                             for c in base)
    im.save(path)


def _kenburns_clip(img, out, dur=3.0, fps=24):
    frames = int(dur * fps)
    _run(["ffmpeg", "-v", "error", "-loop", "1", "-framerate", "1", "-i", img,
          "-t", str(dur),
          "-vf", f"zoompan=z='1.0+0.15*on/{frames}':x='iw/2-(iw/zoom/2)':"
                 f"y='ih/2-(ih/zoom/2)':d={frames}:s=320x180:fps={fps}",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", out])


@pytest.fixture(scope="module")
def material(tmp_path_factory):
    """Три клипа Ken Burns, близких по яркости (как тёмный грейд канала),
    склеенных жёстко: резы ровно на 3.0 и 6.0."""
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    d = tmp_path_factory.mktemp("timing")
    bases = [(60, 40, 35), (38, 55, 45), (45, 42, 62)]
    clips = []
    for i, b in enumerate(bases):
        img = str(d / f"img{i}.png")
        _still(img, b, i)
        clip = str(d / f"kb{i}.mp4")
        _kenburns_clip(img, clip)
        clips.append(clip)
    lst = d / "list.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips), encoding="utf-8")
    hard = str(d / "hard.mp4")
    _run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
          "-c", "copy", "-y", hard])
    diss = str(d / "diss.mp4")
    _run(["ffmpeg", "-v", "error", "-i", clips[0], "-i", clips[1], "-i", clips[2],
          "-filter_complex",
          "[0][1]xfade=transition=fade:duration=0.5:offset=2.5[a];"
          "[a][2]xfade=transition=fade:duration=0.5:offset=5.0[v]",
          "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", diss])
    return {"dir": str(d), "hard": hard, "diss": diss}


class TestCutDetection:
    def test_hard_cuts_found_exactly_and_nothing_else(self, material):
        """Резы известны точно (3.0 и 6.0). Ложные пики от зума Ken Burns —
        отдельный отказ теста: тогда детектор «находит» резы там, где их нет,
        и дрейф считался бы по мусору."""
        cuts = vt.detect_cuts(material["hard"])
        assert len(cuts) == 2, f"ожидались ровно два реза, найдено {cuts}"
        assert abs(cuts[0] - 3.0) <= 1.5 / 24
        assert abs(cuts[1] - 6.0) <= 1.5 / 24

    def test_dissolve_material_reports_low_coverage_not_success(self, material, tmp_path):
        """ЗАМЕРЕННЫЙ предел метода, а не недоработка: диссолв 0.5с между
        двумя шумными кадрами близкой яркости даёт отношение к фону ~1.0 —
        он не отличим от дыхания зума ни другим порогом, ни другим
        разрешением (проверено 64x36/128x72, среднее/медиана).

        Важно не то, что детектор его не находит, а то, КАК он об этом
        сообщает: «резов не нашли» обязано читаться как отсутствие измерения,
        иначе ролик с разъехавшимся таймингом получил бы зелёный вердикт
        просто потому, что переходы мягкие."""
        import json
        vd = tmp_path / "ep"
        (vd / "media_plan").mkdir(parents=True)
        blocks = [{"index": i, "speech_onset_sec": t}
                  for i, t in enumerate([2.75, 5.25], start=1)]
        (vd / "media_plan" / "phrase_timeline.json").write_text(
            json.dumps({"locked": True, "blocks": blocks}), encoding="utf-8")
        report, code = vt.verify(str(vd), video_path=material["diss"])
        assert report["verdict"] != "ok"
        assert code != 0
        # И ни одного ЛОЖНОГО реза от дыхания зума на всём материале.
        assert report["detected_cuts"] <= 2, report

    def test_difference_is_measured_in_colour_not_luma(self, tmp_path):
        """РЕАЛЬНЫЙ промах, найденный при калибровке (13.09): у ffmpeg
        «красный» (255,0,0) и «зелёный» (0,128,0) имеют ОДИНАКОВУЮ яркость
        Y=76. И готовый scene-детектор ffmpeg, и первая версия этой кривой
        считали по яркости — и не увидели такой рез даже с порогом 0.01,
        найдя все остальные. На тёмном грейде канала это систематический
        слепой участок, выглядящий как «резов нет»."""
        pytest.importorskip("numpy")
        a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
        for path, colour in ((a, "red"), (b, "green")):
            _run(["ffmpeg", "-v", "error", "-f", "lavfi",
                  "-i", f"color=c={colour}:s=160x90:d=2:r=24",
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", path])
        lst = tmp_path / "l.txt"
        lst.write_text(f"file '{a}'\nfile '{b}'\n", encoding="utf-8")
        merged = str(tmp_path / "m.mp4")
        _run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
              "-i", str(lst), "-c", "copy", "-y", merged])
        cuts = vt.detect_cuts(merged)
        assert cuts and abs(cuts[0] - 2.0) <= 1.5 / 24, (
            "рез между кадрами одинаковой ЯРКОСТИ снова не виден — "
            f"детектор вернулся к яркостной проекции, найдено {cuts}")


class TestMatching:
    def test_each_detected_cut_serves_one_expected_onset(self):
        """Один найденный рез не может закрыть два онсета: иначе пропущенный
        рез маскировался бы соседним и покрытие врало бы в плюс."""
        pairs, unmatched, extra = vt.match_cuts([10.0, 10.2], [10.05])
        assert len(pairs) == 1 and len(unmatched) == 1 and not extra

    def test_far_cut_is_not_matched(self):
        pairs, unmatched, extra = vt.match_cuts([10.0], [12.0])
        assert not pairs and unmatched == [10.0] and extra == [12.0]

    def test_trend_separates_accumulating_drift_from_one_off_error(self):
        """Главный симптом всех трёх исторических поломок тайминга — не
        разовая ошибка, а НАКОПИТЕЛЬНЫЙ уход. Тренд обязан их различать."""
        one_off = [(t, 0.0) for t in range(0, 100, 10)]
        one_off[3] = (30.0, 0.4)
        creeping = [(t, t * 0.002) for t in range(0, 100, 10)]
        assert abs(vt._linear_trend(one_off)) < 0.001
        assert vt._linear_trend(creeping) == pytest.approx(0.002, abs=1e-4)


class TestVerdict:
    def test_missing_video_is_not_a_pass(self, tmp_path):
        report, code = vt.verify(str(tmp_path))
        assert report["verdict"] == "no_video" and code == 1

    def test_low_coverage_never_reads_as_ok(self, material, tmp_path, monkeypatch):
        """«Резов не нашли» — это отсутствие измерения, а не доказательство
        точности. Вердикт обязан отличать одно от другого."""
        import json
        vd = tmp_path / "ep"
        (vd / "media_plan").mkdir(parents=True)
        blocks = [{"index": i, "speech_onset_sec": float(i)} for i in range(1, 12)]
        (vd / "media_plan" / "phrase_timeline.json").write_text(
            json.dumps({"locked": True, "blocks": blocks}), encoding="utf-8")
        report, code = vt.verify(str(vd), video_path=material["hard"])
        assert report["verdict"] == "low_coverage"
        assert code == 2
