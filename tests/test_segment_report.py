# -*- coding: utf-8 -*-
"""segment_report.py — папка-отчёт по готовому ролику.

Проверяется на СИНТЕТИКЕ С ИЗВЕСТНОЙ ИСТИНОЙ (тот же приём, что у
test_verify_timing.py): четыре плана известной длины и известного тона,
склеенные жёстко. Если отчёт находит не те длительности или не те скачки —
он врёт, и смотреть на него нельзя. Это тест ИНСТРУМЕНТА, не подмена
реального прогона: реальный ролик и его числа — только с машины владельца.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import segment_report as sr  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")

# (длительность, базовый цвет) — тона разведены заметно, чтобы известные
# скачки яркости были больше любого шума зума.
PLANS = [(2.0, (40, 40, 40)), (3.0, (120, 120, 120)), (1.0, (60, 90, 60)), (4.0, (200, 200, 200))]
FPS = 24


def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _still(path, base):
    from PIL import Image
    im = Image.new("RGB", (320, 180))
    px = im.load()
    for y in range(180):
        for x in range(320):
            n = 12 * (((x // 40) + (y // 22)) % 2)
            px[x, y] = tuple(max(0, min(255, c + n)) for c in base)
    im.save(path)


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    d = tmp_path_factory.mktemp("seg")
    clips = []
    for i, (dur, base) in enumerate(PLANS):
        img = str(d / f"img{i}.png")
        _still(img, base)
        clip = str(d / f"c{i}.mp4")
        frames = int(dur * FPS)
        _run(["ffmpeg", "-v", "error", "-loop", "1", "-framerate", "1", "-i", img, "-t", str(dur),
              "-vf", f"zoompan=z='1.0+0.15*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                     f"d={frames}:s=320x180:fps={FPS}",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", clip])
        clips.append(clip)
    lst = d / "list.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips), encoding="utf-8")
    final = str(d / "final.mp4")
    _run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-y", final])

    mp = d / "media_plan"
    mp.mkdir()
    shots = [{"index": i, "section": "HOOK" if i < 2 else "BLOCK 1", "text": f"фраза {i}",
              "query": "q", "kind": "photo", "file": f"img{i}.png",
              "source": "pexels" if i != 3 else "local"} for i in range(4)]
    (mp / "shotlist.json").write_text(json.dumps(
        {"version": 1, "gates": {"pexels_api_key": False, "clip_model_loaded": True}, "shots": shots},
        ensure_ascii=False), encoding="utf-8")
    (mp / "fallback_cards_report.json").write_text(json.dumps({"misses": [
        {"index": 2, "reason": "below_relevance_threshold", "text": "", "card_text": ""},
        {"index": 3, "reason": "no_media_at_all", "text": "", "card_text": ""}]}), encoding="utf-8")
    (mp / "relevance_gate_report.json").write_text(json.dumps({"misses": [
        {"index": 1, "kind": "photo", "query": "q", "relevance": 0.1, "threshold": 0.19},
        {"index": 2, "kind": "video", "query": "q", "relevance": 0.12, "threshold": 0.19}]}),
        encoding="utf-8")
    (mp / "render_manifest.json").write_text(json.dumps({"clips": [
        {"index": 0, "status": "ok"}, {"index": 1, "status": "failed", "reason": "x"}]}), encoding="utf-8")
    return str(d)


class TestVideoAxis:
    def test_plan_durations_come_from_pixels_and_match_truth(self, episode):
        v = sr.analyze_video(os.path.join(episode, "final.mp4"), expected_plans=4)
        assert v["status"] == "measured", v
        assert v["plans_found"] == 4, v["plans"]
        got = [p["end"] - p["start"] for p in v["plans"]]
        for g, (want, _) in zip(got, PLANS):
            assert abs(g - want) <= 1.5 / FPS, (got, PLANS)
        assert v["plan_duration_sec"]["min"] == pytest.approx(1.0, abs=0.07)
        assert v["plan_duration_sec"]["max"] == pytest.approx(4.0, abs=0.07)
        assert v["short_plans_lt_1.5s"] == 1
        assert v["long_plans_gt_12s"] == 0

    def test_tone_jumps_match_known_brightness_steps(self, episode):
        """Яркости планов 40→120→~76→200 (Rec.709 от базовых цветов + шахматка
        +12 на половине пикселей): скачки ~80, ~44, ~124. Допуск — кодек и
        зум, не порядок величины."""
        v = sr.analyze_video(os.path.join(episode, "final.mp4"), expected_plans=4)
        lum = [p["luma"] for p in v["plans"]]
        assert lum[0] < lum[2] < lum[1] < lum[3], lum
        jumps = [abs(a - b) for a, b in zip(lum, lum[1:])]
        assert 60 < jumps[0] < 100 and 30 < jumps[1] < 60 and 100 < jumps[2] < 145, jumps
        assert v["luma_jump_0_255"]["n"] == 3
        assert v["luma_jump_0_255"]["max"] == pytest.approx(max(jumps), abs=0.15)   # plans[].luma округлён до 0.1
        # Серые планы — насыщенность около нуля, зелёный — заметно выше.
        sats = [p["sat"] for p in v["plans"]]
        assert sats[2] > max(sats[0], sats[1], sats[3]) + 20, sats

    def test_low_coverage_is_named_not_hidden(self, episode):
        """Если шотлист обещает вдесятеро больше планов, чем найдено резов, —
        это не «длинные планы», а отсутствие измерения."""
        v = sr.analyze_video(os.path.join(episode, "final.mp4"), expected_plans=40)
        assert v["status"] == "low_coverage"
        assert v["cut_coverage"] == 0.1


class TestArtifactsAxis:
    def test_cards_split_by_reason_and_share_over_shots(self, episode):
        a = sr.collect_artifacts(episode)
        assert a["n_shots"] == 4
        assert a["cards"]["n"] == 2 and a["cards"]["share"] == 0.5
        assert a["cards"]["by_reason"] == {"below_relevance_threshold": 1, "no_media_at_all": 1}
        assert a["cards"]["indices"] == [2, 3]

    def test_rejections_aggregated_by_report_and_kind(self, episode):
        a = sr.collect_artifacts(episode)
        assert a["rejections"]["relevance_gate_report:photo"] == 1
        assert a["rejections"]["relevance_gate_report:video"] == 1
        assert a["rejections"]["render_manifest:failed"] == 1
        assert a["known_bad_slots"] == [1, 2]

    def test_sources_kinds_and_gates_are_carried(self, episode):
        a = sr.collect_artifacts(episode)
        assert a["by_source"] == {"pexels": 3, "local": 1}
        assert a["by_kind"] == {"photo": 4}
        assert a["gates"]["pexels_api_key"] is False


class TestBuild:
    def test_build_writes_report_and_contact_pages(self, episode):
        report, path = sr.build(episode)
        assert os.path.exists(path)
        data = json.load(open(path, encoding="utf-8"))
        assert data["video_measured"]["status"] == "measured"
        assert data["artifacts"]["cards"]["n"] == 2
        assert "stack" in data and "determinism" in data
        assert data["contact_pages"] and all(os.path.exists(p) for p in data["contact_pages"])
        text = sr.summarize(report)
        assert "Карточек: 2" in text and "Планов в пикселях: 4" in text

    def test_missing_video_is_reported_not_crashed(self, tmp_path):
        (tmp_path / "media_plan").mkdir()
        report, _ = sr.build(str(tmp_path), with_contact=False)
        assert report["video_measured"]["status"] == "no_video"
        assert report["artifacts"]["n_shots"] == 0


def test_contact_sheets_cover_absorbed_slots_too(tmp_path):
    """segment_report.contact_sheets() — ВТОРОЙ, независимый вызов
    shotlist_contact.render_page() (см. её же комментарий у
    build_absorption_cover_map, найдено 21.09 тем же прогоном, что и
    основной баг в shotlist_contact.main()): без cover_map поглощённые
    слоты здесь по-прежнему рисовались бы красной заливкой «НЕТ ФАЙЛА»,
    хотя в final.mp4 они никогда не пустуют. Проверяем НАПРЯМУЮ, не через
    tяжёлый синтетический рендер эпизода — контактный лист не рендерит
    видео, ему нужен только shotlist.json."""
    from PIL import Image
    vd = tmp_path
    (vd / "media").mkdir()
    (vd / "media" / "001.jpg").parent.mkdir(exist_ok=True)
    Image.new("RGB", (64, 36), (120, 80, 40)).save(vd / "media" / "001.jpg")
    (vd / "media_plan").mkdir()
    shots = [
        {"index": 0, "section": "HOOK", "text": "поглощённая фраза", "query": "q",
         "kind": None, "file": None, "source": "absorbed"},
        {"index": 1, "section": "HOOK", "text": "фраза с кадром", "query": "q",
         "kind": "photo", "file": "media/001.jpg", "source": "local"},
    ]
    (vd / "media_plan" / "shotlist.json").write_text(
        json.dumps({"version": 1, "locked": False, "shots": shots}, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "report_contact"
    out_dir.mkdir()
    pages = sr.contact_sheets(str(vd), str(out_dir), cols=2, per_page=24)
    assert pages and os.path.exists(pages[0])
    import shotlist_contact as sc
    img = Image.open(pages[0])
    px = img.getpixel((sc.PAD + sc.THUMB_W // 2, sc.PAD + sc.THUMB_H // 2))
    assert not all(abs(px[k] - c) <= 25 for k, c in enumerate((70, 20, 20))), \
        f"поглощённый слот остался красной заливкой в отчётном листе ({px})"


def test_render_episode_calls_the_report_last():
    """«Одна команда»: render_episode.py обязан заканчиваться этим отчётом —
    иначе он снова станет слоем, который есть и который никто не зовёт."""
    src = open(os.path.join(SCRIPTS_DIR, "render_episode.py"), encoding="utf-8").read()
    assert "segment_report" in src
    assert src.index("segment_report.build(") > src.index("_timing_verification(")


class TestCoverageVerdict:
    """Покрытие > 1 — не «всё хорошо», а «план не равен слоту».

    Замер 14.09 (videos/_test60s): 10 планов на 9 слотов. Причина не в
    монтаже — у стокового видео своя внутренняя склейка, детектор честно
    видит её как рез. Два «плана» там оказались ОДНИМ клипом на 6.9с, и
    по отчёту это было не прочитать: статус стоял «measured»."""

    def test_over_coverage_is_named(self, episode):
        v = sr.analyze_video(os.path.join(episode, "final.mp4"), expected_plans=2)
        assert v["status"] == "over_coverage", v["cut_coverage"]
        assert v["cut_coverage"] > sr.MAX_CUT_COVERAGE

    def test_exact_match_is_still_measured(self, episode):
        v = sr.analyze_video(os.path.join(episode, "final.mp4"), expected_plans=4)
        assert v["status"] == "measured"

    def test_limit_is_written_into_the_report(self, episode):
        v = sr.analyze_video(os.path.join(episode, "final.mp4"), expected_plans=4)
        assert any("внутренняя склейка" in s for s in v["limits"]), v["limits"]

    def test_over_coverage_still_reports_the_numbers(self, episode):
        """Честность — не отказ от измерения: числа остаются, меняется их
        трактовка."""
        report, _ = sr.build(episode, with_contact=False)
        text = sr.summarize(report)
        assert "Планов в пикселях" in text
