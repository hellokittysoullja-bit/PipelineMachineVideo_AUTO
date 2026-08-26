"""Регрессии на баги, найденные аудитом генератора.

Каждый тест назван по симптому, который был виден в готовом ролике, а не по
имени функции: если тест снова покраснеет, сразу понятно, что именно вернулось.
Запуск: python -m pytest tests/test_regressions.py -v
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# pipeline_smart читает sys.argv[1] на импорте (см. tests/test_parse.py)
import tempfile  # noqa: E402
_STUB = tempfile.mkdtemp(prefix="regr_stub_")
_ARGV = sys.argv
sys.argv = ["pytest", _STUB]
import assemble             # noqa: E402
import pipeline_smart       # noqa: E402
import stock_fetch_multisource  # noqa: E402
sys.argv = _ARGV

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _blocks(spec):
    return [{"text": "x", "words": w, "pause_after": 0.8, "section": s, "stat": None}
            for w, s in spec]


# ---------- Кадры хука перестали быть короткими (block_durations) ----------

def test_hook_frames_stay_within_hook_cap():
    """Симптом: в хуке кадры висели по 9-10с вместо предела 5с — финальный
    scale = total/sum(d) перечёркивал только что применённые капы."""
    blocks = _blocks([(18, "HOOK")] * 12
                     + [(22, f"BLOCK {1 + i // 25}: X") for i in range(120)]
                     + [(14, "FINAL")] * 6)
    for total in (900.0, 1080.0, 1250.0):
        durs = pipeline_smart.block_durations(blocks, total)
        assert abs(sum(durs) - total) < 1e-3, "общая длина обязана совпасть с аудио"
        for d, b in zip(durs, blocks):
            cap = pipeline_smart.HOOK_MAX_CLIP if b["section"].startswith("HOOK") \
                else pipeline_smart.MAX_CLIP
            assert d <= cap + 1e-6, f"{b['section']}: {d:.2f}с > предела {cap}с"
            assert d >= pipeline_smart.MIN_CLIP - 1e-6


def test_durations_keep_total_even_when_caps_unreachable():
    """Если блоков физически мало на такое аудио — синхрон важнее капа,
    но сумма всё равно обязана сойтись (иначе поедет вся дорожка)."""
    blocks = _blocks([(50, "BLOCK 1: X")] * 3)
    durs = pipeline_smart.block_durations(blocks, 600.0)
    assert abs(sum(durs) - 600.0) < 1e-3


def test_fit_to_total_is_stable_at_the_boundaries():
    raw = [10.0, 10.0, 10.0]
    exact = pipeline_smart.fit_to_total(raw, [4.0] * 3, [20.0] * 3, 30.0)
    assert abs(sum(exact) - 30.0) < 1e-6
    assert all(abs(x - 10.0) < 1e-6 for x in exact)


# ---------- Картинка отставала от слов (бюджет кроссфейдов) ----------

def test_xfade_budget_matches_what_the_chain_actually_consumes():
    """Симптом: к концу 18-минутного ролика картинка отставала от слов на ~25с,
    а хвост срезался. Бюджет закладывался как (n-1)*XFADE_DUR, хотя ~2/3
    склеек — hardcut на XFADE_DUR_HARD."""
    for n in (2, 20, 138, 200):
        sections = ["HOOK"] * 12 + [f"BLOCK {1 + i // 25}: X" for i in range(max(0, n - 12))]
        sections = sections[:n]
        plan = pipeline_smart.plan_transitions(sections)
        assert len(plan) == n - 1
        assert abs(pipeline_smart.transitions_budget(sections)
                   - sum(d for _, d in plan)) < 1e-9
        # именно тот бюджет, который потом вычтет цепочка xfade
        durs = [8.0] * n
        cum = durs[0]
        for i in range(1, n):
            cum = cum + durs[i] - plan[i - 1][1]
        assert abs((sum(durs) - cum) - sum(d for _, d in plan)) < 1e-6


def test_transition_plan_is_deterministic_and_independent_of_clip_names():
    """План обязан зависеть только от индекса и секций: если он снова начнёт
    зависеть от имён файлов клипов, бюджет опять станет невычислимым заранее
    (имя клипа содержит хэш длительности, а длительность считается от бюджета)."""
    sections = ["HOOK"] * 5 + ["BLOCK 1: A"] * 5 + ["BLOCK 2: B"] * 5
    assert pipeline_smart.plan_transitions(sections) == pipeline_smart.plan_transitions(sections)


def test_assemble_has_the_same_exact_budget_fix():
    is_hook = [True] * 8 + [False] * 60
    plan = assemble.plan_transitions(is_hook)
    budget = assemble.transitions_budget(is_hook)
    assert len(plan) == len(is_hook) - 1
    assert abs(budget - sum(d for _, d in plan)) < 1e-9
    # и он заметно меньше старой верхней оценки — иначе фикс не применён
    assert budget < (len(is_hook) - 1) * assemble.XFADE_DUR * 0.9


def test_boundary_transitions_are_visible_ones():
    """На стыке секций должен стоять заметный переход, а не hardcut."""
    sections = ["HOOK", "HOOK", "BLOCK 1: A", "BLOCK 1: A", "FINAL"]
    plan = pipeline_smart.plan_transitions(sections)
    assert plan[1][0] in pipeline_smart.BOUNDARY_TRANSITIONS
    assert plan[1][1] == pipeline_smart.XFADE_DUR
    assert plan[3][0] in pipeline_smart.BOUNDARY_TRANSITIONS


# ---------- Плашка с процентом исчезала с экрана (drawtext) ----------

def test_escape_drawtext_does_not_double_percent():
    """Симптом: [stat:на 40% тяжелее] не выводился ВООБЩЕ. %% ffmpeg не спасает
    (проверено), спасает expansion=none — и тогда удвоение даёт "40%%" на экран."""
    assert pipeline_smart.escape_drawtext("40%") == "40%"
    assert "%%" not in pipeline_smart.escape_drawtext("рост 40% за год")


def test_overlays_disable_text_expansion():
    vf = pipeline_smart.add_overlays("null", 6.0, title="Тест", stat="40%")
    if pipeline_smart.FONT_PATH is None:
        pytest.skip("шрифт не найден — drawtext не подставляется")
    assert vf.count("expansion=none") == 2, "оба drawtext обязаны глушить подстановки"
    assert "text='40%'" in vf


@pytest.mark.skipif(not HAS_FFMPEG, reason="нужен ffmpeg")
def test_percent_stat_actually_renders(tmp_path):
    """Главная проверка: реальный ffmpeg должен НАРИСОВАТЬ плашку с процентом.
    Раньше он молча не рисовал ничего и возвращал 0 — откат не срабатывал."""
    if pipeline_smart.FONT_PATH is None:
        pytest.skip("шрифт не найден")
    photo = tmp_path / "p.jpg"
    Image.new("RGB", (1280, 720), (20, 20, 20)).save(photo)

    def frames_with_text(stat):
        out = tmp_path / f"o_{'pct' if stat else 'none'}.mp4"
        assert pipeline_smart.kenburns(str(photo), str(out), 2.0, stat=stat, section="BODY")
        png = tmp_path / f"f_{'pct' if stat else 'none'}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1.0", "-i", str(out),
                        "-frames:v", "1", "-update", "1", str(png)], check=True)
        return Image.open(png).convert("L").point(lambda v: 255 if v > 180 else 0).getbbox()

    assert frames_with_text("40% ПОТЕРЬ") is not None, \
        "плашка с процентом не отрисовалась — вернулся баг Stray %"


# ---------- Медиа и синхрон ----------

def test_local_media_includes_stock_videos(tmp_path, monkeypatch):
    """Симптом: половина скачанного стока (NNN_stock_video.mp4) не попадала в
    ролик вообще, а фотографии прокручивались по кругу с повторами."""
    media = tmp_path / "media"
    media.mkdir()
    for n in (1, 3, 5):
        Image.new("RGB", (64, 36), (1, 2, 3)).save(media / f"{n:03d}_stock.jpg")
    for n in (2, 4):
        (media / f"{n:03d}_stock_video.mp4").write_bytes(b"x")
    monkeypatch.setattr(pipeline_smart, "MEDIA_FOLDER", str(media))
    found = pipeline_smart.scan_local_media()
    assert [k for k, _ in found] == ["photo", "video", "photo", "video", "photo"], \
        "порядок слотов и типы должны совпадать со схемой ЧАСТИ 14"


def test_local_media_ignores_junk(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(media / "001_stock.jpg")
    (media / "notes.txt").write_text("x")
    (media / "_wip.jpg").write_bytes(b"x")
    monkeypatch.setattr(pipeline_smart, "MEDIA_FOLDER", str(media))
    assert len(pipeline_smart.scan_local_media()) == 1


def test_unknown_tags_do_not_glue_words(tmp_path):
    """Симптом: 'грамма.[breath]Так' считалось за одно слово, блок получал
    меньше времени, чем реально звучит (тот же баг чинили в wordcount.py)."""
    f = tmp_path / "script.txt"
    f.write_text("=== HOOK === один два.[breath]три четыре\n", encoding="utf-8")
    blocks = pipeline_smart.parse_blocks(str(f))
    assert sum(b["words"] for b in blocks) == 4


# ---------- Сток ----------

def test_stock_dedup_picks_a_different_result_each_time():
    """Симптом: все слоты с одним тематическим запросом получали ОДНУ картинку —
    каждый источник всегда брал hits[0]."""
    stock_fetch_multisource.used_ids.clear()
    items = [{"id": i} for i in range(5)]
    picked = [stock_fetch_multisource._pick_unused("src", items, lambda x: x["id"])["id"]
              for _ in range(5)]
    assert sorted(picked) == [0, 1, 2, 3, 4]
    # выдача кончилась — повтор лучше пустого слота, но не исключение
    assert stock_fetch_multisource._pick_unused("src", items, lambda x: x["id"])["id"] == 0
    stock_fetch_multisource.used_ids.clear()


def test_download_leaves_no_partial_file_on_failure(tmp_path, monkeypatch):
    """Симптом: оборванная загрузка оставляла обрезанный файл, и следующий
    прогон считал слот готовым — битый кадр доезжал до сборки."""
    dest = tmp_path / "001_stock.jpg"

    class Boom:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            raise OSError("обрыв связи")

    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen",
                        lambda *a, **k: Boom())
    with pytest.raises(OSError):
        stock_fetch_multisource._download("http://x/y.jpg", str(dest))
    assert not dest.exists()
    assert not (tmp_path / "001_stock.jpg.part").exists()


def test_download_rejects_empty_body(tmp_path, monkeypatch):
    dest = tmp_path / "e.jpg"

    class Empty:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen",
                        lambda *a, **k: Empty())
    with pytest.raises(ValueError):
        stock_fetch_multisource._download("http://x/y.jpg", str(dest))
    assert not dest.exists()


# ---------- assemble.py: перенесённые фиксы ----------

def test_assemble_resolves_png_ai_slots(tmp_path):
    """Google Flow отдаёт png — слот с NNN_flow.png раньше считался пустым."""
    media = tmp_path / "media"
    media.mkdir()
    (media / "007_flow.png").write_bytes(b"x")
    kind, path, source = assemble.resolve_slot(str(media), 7)
    assert (kind, source) == ("photo", "ai")
    assert path.endswith("007_flow.png")


def test_assemble_final_mux_trims_to_audio_length():
    """-shortest с -c:v copy режет по границам GOP, а не по концу аудио —
    без явного -t в готовый файл попадали лишние секунды без звука."""
    src = open(os.path.join(SCRIPTS_DIR, "assemble.py"), encoding="utf-8").read()
    tail = src[src.index("merged = pad_to_length"):]
    assert '"-t", f"{audio_dur:.3f}"' in tail


def test_assemble_clip_cache_key_includes_duration():
    """Без хэша параметров правка assemble_config.json переиспользовала клипы
    со старыми длительностями, и весь ролик уезжал по таймингу."""
    src = open(os.path.join(SCRIPTS_DIR, "assemble.py"), encoding="utf-8").read()
    assert 'f"clip_{slot:04d}_{params_hash}.mp4"' in src
    assert 'params_hash = hashlib.md5(' in src


# ---------- Сквозной прогон: видео обязано совпасть с аудио ----------

@pytest.mark.skipif(not HAS_FFMPEG, reason="нужен ffmpeg")
def test_end_to_end_video_length_tracks_audio_on_many_blocks(tmp_path):
    """Тот самый рассинхрон в сборе: много блоков -> много склеек -> раньше
    видео получалось заметно длиннее аудио и хвост срезался. Допуск жёсткий."""
    d = tmp_path / "v"
    media = d / "media"
    media.mkdir(parents=True)
    words = "раз два три четыре пять шесть семь восемь"
    body = "".join(f"=== BLOCK {i+1}: Тема{i+1} === "
                   + f"{words}.[pause]{words}.[pause]{words}.\n" for i in range(6))
    (d / "script.txt").write_text(
        f"=== HOOK === {words}.[pause]{words}.[pause]{words}.\n{body}"
        f"=== FINAL === {words}.[pause]{words}.\n", encoding="utf-8")
    for i in range(12):
        Image.new("RGB", (1280, 720), (10 + i * 15, 40, 200 - i * 10)).save(media / f"{i:03d}.jpg")
    audio_len = 90.0
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"sine=frequency=300:duration={audio_len}",
                    "-c:a", "libmp3lame", str(d / "audio.mp3")], check=True)

    env = dict(os.environ, PARALLAX="0", PEXELS_API_KEY="")
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), str(d)],
                       capture_output=True, text=True, timeout=1800, env=env)
    assert r.returncode == 0, f"пайплайн упал:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"

    probe = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                            "-show_format", "-show_streams", str(d / "final.mp4")],
                           capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    assert abs(float(info["format"]["duration"]) - audio_len) <= 0.5

    # Прямая проверка того самого рассинхрона: склеенное видео ДО финального -t
    # не должно быть длиннее аудио. Раньше бюджет склеек завышался, кадры
    # раздувались, merged.mp4 выходил длиннее — финальный -t срезал хвост, а
    # картинка на всём протяжении отставала от слов.
    merged = d / "temp_smart" / "merged.mp4"
    assert merged.exists()
    merged_len = float(json.loads(subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(merged)],
        capture_output=True, text=True, check=True).stdout)["format"]["duration"])
    assert merged_len <= audio_len + 0.5, (
        f"склейка длиннее аудио на {merged_len - audio_len:.2f}с — вернулся "
        f"завышенный бюджет кроссфейдов, хвост картинки будет срезан")

    streams = {s["codec_type"]: s for s in info["streams"]}
    assert "video" in streams and "audio" in streams
    # Ключевое: видеодорожка НЕ обрезана относительно звука — до фикса бюджета
    # склеек видео было длиннее и финальный -t срезал хвост картинки.
    v_len = float(streams["video"].get("duration") or info["format"]["duration"])
    a_len = float(streams["audio"].get("duration") or info["format"]["duration"])
    assert abs(v_len - a_len) <= 0.6, f"видео {v_len:.2f}с против аудио {a_len:.2f}с"


# ---------- wordcount: реальный хронометраж, а не только слова ----------

def test_wordcount_reports_pause_time(tmp_path):
    """Симптом: отчёт длины показывал только слова/125 и молчал про теги пауз.
    На 18-минутном сценарии их 150+ — это ~2 минуты сверху, то есть разница
    между «в коридоре» и «перебор» перед платной озвучкой (ЧАСТЬ 1)."""
    import wordcount
    f = tmp_path / "s.txt"
    f.write_text("=== HOOK === один два.[pause]три четыре.[short pause]пять\n",
                 encoding="utf-8")
    count, pause_sec = wordcount.count_words(str(f))
    assert count == 5
    assert abs(pause_sec - 1.2) < 1e-9

    r = subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "wordcount.py"),
                        str(f), "1"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "С учётом тегов пауз" in r.stdout


def test_wordcount_pause_values_match_the_assembler():
    """Отчёт длины и сборка обязаны считать паузы одинаково — иначе оператор
    планирует одну длину, а получает другую."""
    import wordcount
    for tag, sec in wordcount.PAUSE_SEC.items():
        assert pipeline_smart.PAUSE_DURATIONS[tag] == sec


# ---------- U1: параллельный рендер кадров ----------

def test_resolve_workers_env_override_and_default():
    assert pipeline_smart.resolve_workers("4", 8) == 4
    assert pipeline_smart.resolve_workers("", 8) == 7
    assert pipeline_smart.resolve_workers("bogus", 4) == 3
    assert pipeline_smart.resolve_workers("0", 4) == 3       # "0" не валидный воркер-каунт
    assert pipeline_smart.resolve_workers("-1", 4) == 3
    assert pipeline_smart.resolve_workers(None, 1) == 1       # даже на одном ядре хотя бы 1


def test_resolve_ffmpeg_threads_keeps_total_within_cpu_budget():
    # WORKERS параллельных процессов, каждый с FFMPEG_THREADS потоков, не должны
    # суммарно перегружать доступные ядра — иначе параллелизм не даёт выигрыша.
    for workers, cpus in [(4, 8), (1, 4), (7, 8), (1, 1)]:
        threads = pipeline_smart.resolve_ffmpeg_threads(workers, cpus)
        assert threads >= 1
        assert threads * workers <= max(cpus, workers)   # разумный клэмп, не оверсабскрипшен


def test_assemble_has_matching_worker_formula():
    """assemble.py не должен расходиться с pipeline_smart.py в этой формуле —
    тот же класс дублирования кода уже дал баги B4/B5 в прошлом аудите."""
    assert assemble.resolve_workers("4", 8) == pipeline_smart.resolve_workers("4", 8)
    assert assemble.resolve_ffmpeg_threads(4, 8) == pipeline_smart.resolve_ffmpeg_threads(4, 8)


def test_render_functions_pass_clip_crf_and_thread_clamp(tmp_path, monkeypatch):
    """CRF/threads должны реально попадать в командную строку ffmpeg, а не
    только существовать как константы — ловит опечатку в имени переменной,
    которая иначе тихо оставила бы старое качество/поток."""
    photo = tmp_path / "p.jpg"
    Image.new("RGB", (64, 36), (10, 20, 30)).save(photo)
    captured = []

    class FakeResult:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return FakeResult()

    monkeypatch.setattr(pipeline_smart.subprocess, "run", fake_run)
    pipeline_smart.kenburns(str(photo), str(tmp_path / "o.mp4"), 3.0)
    cmd = captured[-1]
    assert pipeline_smart.CLIP_CRF in cmd
    assert "-threads" in cmd and str(pipeline_smart.FFMPEG_THREADS) in cmd


def test_xfade_chain_uses_slower_higher_quality_final_pass(monkeypatch):
    """Единственный полноразмерный проход (не N раз, как per-clip) может себе
    позволить -preset slow и более высокий CRF — второе поколение потерь
    поверх уже сжатых клипов иначе съедает большую часть выигрыша от
    поднятого CLIP_CRF."""
    captured = []

    class FakeResult:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return FakeResult()

    monkeypatch.setattr(pipeline_smart.subprocess, "run", fake_run)
    pipeline_smart.xfade_chain(["a.mp4", "b.mp4"], [3.0, 3.0], ["HOOK", "HOOK"], "out.mp4")
    cmd = captured[-1]
    assert pipeline_smart.FINAL_PRESET in cmd
    assert pipeline_smart.FINAL_CRF in cmd


def test_parallel_rendering_produces_correct_length_video(tmp_path):
    """Симптом, если параллелизм ломает синхрон: несколько воркеров, одна и та
    же входная раскладка блоков -> итог должен совпасть с последовательным
    поведением (Фаза A детерминирована независимо от того, сколько потоков
    рендерит Фазу B). Проверяем через два реальных прогона с разным числом
    воркеров и сравниваем длительность/число клипов, а не байты (энкод
    многопоточным x264 не побитово идентичен однопоточному)."""
    d = tmp_path / "v"
    media = d / "media"
    media.mkdir(parents=True)
    words = "раз два три четыре пять шесть семь восемь"
    (d / "script.txt").write_text(
        f"=== HOOK === {words}.[pause]{words}.\n"
        f"=== BLOCK 1: Тема === {words}.[pause]{words}.[pause]{words}.\n"
        f"=== FINAL === {words}.\n", encoding="utf-8")
    for i in range(6):
        Image.new("RGB", (1280, 720), (i * 30, 40, 200 - i * 20)).save(media / f"{i:03d}.jpg")
    audio_len = 20.0
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"sine=frequency=300:duration={audio_len}",
                    "-c:a", "libmp3lame", str(d / "audio.mp3")], check=True)

    durations = []
    for workers in ("1", "4"):
        for p in (d / "temp_smart", d / "final.mp4"):
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
        env = dict(os.environ, PARALLAX="0", PEXELS_API_KEY="", PIPELINE_WORKERS=workers)
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), str(d)],
                           capture_output=True, text=True, timeout=600, env=env)
        assert r.returncode == 0, f"PIPELINE_WORKERS={workers}: {r.stdout[-2000:]}\n{r.stderr[-1000:]}"
        durations.append(pipeline_smart.get_media_duration(str(d / "final.mp4")))
    assert abs(durations[0] - durations[1]) < 0.5, (
        f"1 воркер дал {durations[0]:.2f}с, 4 воркера дали {durations[1]:.2f}с — "
        f"параллелизм изменил результат, а не только время")


# ---------- U2: приоритет === PEXELS QUERIES === над словарём тем ----------

def test_normalize_section_key_variants():
    assert pipeline_smart.normalize_section_key("HOOK") == "HOOK"
    assert pipeline_smart.normalize_section_key("FINAL") == "FINAL"
    assert pipeline_smart.normalize_section_key("BLOCK 1: Название") == "BLOCK 1"
    assert pipeline_smart.normalize_section_key("BLOCK_2") == "BLOCK 2"
    assert pipeline_smart.normalize_section_key("blockquote") is None
    assert pipeline_smart.normalize_section_key("") is None


def test_parse_pexels_queries_multiline_format():
    raw = (
        "=== HOOK === текст хука\n"
        "=== PEXELS QUERIES ===\n"
        "HOOK: templar knight film still, medieval battle painting\n"
        "BLOCK 1: landsknecht mercenary engraving, zweihander sword\n"
        "=== TITLE OPTIONS ===\n"
        "какой-то заголовок\n"
    )
    parsed = pipeline_smart.parse_pexels_queries(raw)
    assert parsed["HOOK"] == ["templar knight film still", "medieval battle painting"]
    assert parsed["BLOCK 1"] == ["landsknecht mercenary engraving", "zweihander sword"]
    assert "TITLE OPTIONS" not in parsed


def test_parse_pexels_queries_slash_separated_single_line():
    raw = "=== PEXELS QUERIES === (HOOK: q1,q2,q3 / BLOCK_1: qa,qb)\n"
    parsed = pipeline_smart.parse_pexels_queries(raw)
    assert parsed["HOOK"] == ["q1", "q2", "q3"]
    assert parsed["BLOCK 1"] == ["qa", "qb"]


def test_parse_pexels_queries_absent_section_returns_empty():
    assert pipeline_smart.parse_pexels_queries("=== HOOK === текст\n") == {}


def test_resolve_queries_prioritizes_script_over_theme_dict(tmp_path, monkeypatch):
    """Раньше эта секция парсилась только чтобы её выбросить: подбор шёл
    исключительно по THEMES. Теперь она приоритетный источник."""
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "generic sword from dict"})
    script = tmp_path / "script.txt"
    script.write_text(
        "=== PEXELS QUERIES ===\nHOOK: script query one, script query two\n",
        encoding="utf-8")
    monkeypatch.setattr(pipeline_smart, "SCRIPT_FILE", str(script))
    blocks = [
        {"text": "Меч был длинным.", "words": 3, "pause_after": 0.0, "section": "HOOK", "stat": None},
        {"text": "Меч сверкал.", "words": 2, "pause_after": 0.0, "section": "HOOK", "stat": None},
        {"text": "Меч убран.", "words": 2, "pause_after": 0.0, "section": "HOOK", "stat": None},
    ]
    resolved = pipeline_smart.resolve_queries(blocks)
    # цикл по двум запросам секции, а не одна и та же строка на все три блока
    assert resolved[0] == "script query one"
    assert resolved[1] == "script query two"
    assert resolved[2] == "script query one"


def test_resolve_queries_falls_back_to_theme_dict_without_script_section(monkeypatch):
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "generic sword from dict"})
    monkeypatch.setattr(pipeline_smart, "SCRIPT_FILE", "/nonexistent/script.txt")
    blocks = [{"text": "Меч был длинным.", "words": 3, "pause_after": 0.0,
              "section": "HOOK", "stat": None}]
    assert pipeline_smart.resolve_queries(blocks) == ["generic sword from dict"]


# ---------- U3: без второго поколения lossy-сжатия звука ----------

def test_fix_pauses_outputs_lossless_flac(tmp_path):
    src = tmp_path / "audio.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=2", "-c:a", "libmp3lame", str(src)],
                   check=True)
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "fix_pauses.py"), str(tmp_path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-1000:]
    out = tmp_path / "audio_fixed.flac"
    assert out.exists()
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
                            "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "flac"


def test_find_audio_prioritizes_lossless_fixed_over_mp3(tmp_path, monkeypatch):
    for name in ("audio.mp3", "audio_fixed.mp3"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "audio_fixed.flac").write_bytes(b"x")
    monkeypatch.setattr(pipeline_smart, "VIDEO_FOLDER", str(tmp_path))
    assert pipeline_smart.find_audio() == str(tmp_path / "audio_fixed.flac")


# ---------- U4: визуально похожий кадр отбрасывается ДО рендера ----------

def _gradient_image(path, vertical=False, lo=0, hi=255):
    w, h = 160, 90
    img = Image.new("RGB", (w, h))
    px = img.load()
    span = max(1, hi - lo)
    for x in range(w):
        for y in range(h):
            t = (y / h) if vertical else (x / w)
            v = lo + int(t * span)
            px[x, y] = (v, v, v)
    img.save(path)


def test_pexels_photo_rejects_near_duplicate_before_render(tmp_path, monkeypatch):
    """Симптом: слоты с одним запросом получали одну и ту же (или визуально
    идентичную) картинку — каждый источник всегда брал hits[0]. Проверяем,
    что похожий по содержимому кандидат (тот же горизонтальный градиент, чуть
    другая экспозиция) отбрасывается, а структурно другой (вертикальный
    градиент) принимается."""
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(pipeline_smart, "PEXELS_API_KEY", "dummy")

    urls = {
        1: "http://x/near-dup-a.jpg",
        2: "http://x/near-dup-b.jpg",
        3: "http://x/different.jpg",
    }
    photos = [{"id": pid, "src": {"large2x": url}} for pid, url in urls.items()]

    class FakeResp:
        def __init__(self, data):
            self._data = json.dumps(data).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._data

    monkeypatch.setattr(pipeline_smart.urllib.request, "urlopen",
                        lambda *a, **k: FakeResp({"photos": photos}))

    def fake_download(url, dest, timeout=20):
        if url == urls[1]:
            _gradient_image(dest, vertical=False, lo=0, hi=255)
        elif url == urls[2]:
            _gradient_image(dest, vertical=False, lo=20, hi=235)   # тот же сюжет, другая экспозиция
        else:
            _gradient_image(dest, vertical=True, lo=0, hi=255)      # структурно другое фото
        return True

    monkeypatch.setattr(pipeline_smart, "download_atomic", fake_download)

    used, avoid = set(), []
    p1 = pipeline_smart.pexels_photo("q", 0, used_ids=used, avoid_hashes=avoid)
    h1 = pipeline_smart.ahash(p1)
    assert len(avoid) == 1

    p2 = pipeline_smart.pexels_photo("q", 1, used_ids=used, avoid_hashes=avoid)
    h2 = pipeline_smart.ahash(p2)
    assert pipeline_smart.hamming(h1, h2) > pipeline_smart.DUPE_HAMMING_THRESHOLD, (
        "визуально похожий кандидат должен был быть отброшен в пользу "
        "структурно другого, а не пройти как есть")


def test_pexels_photo_cache_hit_still_feeds_avoid_hashes(tmp_path, monkeypatch):
    """Резюмированный прогон: кадр уже лежит в кэше pexels_cache/, но его
    ahash всё равно должен попасть в avoid_hashes — иначе дедуп для НОВЫХ
    кадров этого же запуска был бы неполным."""
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path))
    cache_dir = tmp_path / "pexels_cache"
    cache_dir.mkdir()
    qhash = pipeline_smart.hashlib.md5(b"q").hexdigest()[:8]
    cf = cache_dir / f"0000_{qhash}.jpg"
    _gradient_image(str(cf))
    avoid = []
    result = pipeline_smart.pexels_photo("q", 0, avoid_hashes=avoid)
    assert result == str(cf)
    assert len(avoid) == 1


# ---------- U5: Whisper-выравнивание (опционально) ----------

def test_map_index_is_monotonic_and_bounded():
    assert pipeline_smart._map_index(0, 100, 90) == 0
    assert pipeline_smart._map_index(100, 100, 90) == 89
    seq = [pipeline_smart._map_index(k, 100, 90) for k in range(101)]
    assert seq == sorted(seq)
    assert pipeline_smart._map_index(5, 0, 0) == 0


def test_format_srt_timestamp():
    assert pipeline_smart._format_srt_timestamp(0) == "00:00:00,000"
    assert pipeline_smart._format_srt_timestamp(65.25) == "00:01:05,250"
    assert pipeline_smart._format_srt_timestamp(3661.001) == "01:01:01,001"
    assert pipeline_smart._format_srt_timestamp(-1) == "00:00:00,000"


def test_whisper_breakpoints_disabled_by_default_returns_none():
    # По умолчанию WHISPER_ALIGN не выставлен -> пайплайн не должен даже
    # пытаться качать модель/тратить CPU без явного согласия оператора.
    assert pipeline_smart.WHISPER_ENABLED is False
    assert pipeline_smart.whisper_breakpoints([{"words": 3}], "irrelevant.mp3") is None


def test_whisper_breakpoints_maps_recognized_words_to_block_boundaries(tmp_path, monkeypatch):
    """Математику выравнивания проверяем без реальной модели: подсовываем
    свои (start, end, word) и свою длину аудио, смотрим, что границы блоков
    получаются на реальных тайм-кодах, а не на оценке по словам/скорости."""
    monkeypatch.setattr(pipeline_smart, "WHISPER_ENABLED", True)
    fake_words = [(float(i), float(i) + 0.4, f"w{i}") for i in range(10)]
    monkeypatch.setattr(pipeline_smart, "transcribe_words", lambda audio_path: fake_words)
    monkeypatch.setattr(pipeline_smart, "get_media_duration", lambda p: 12.0)
    blocks = [{"words": 4}, {"words": 3}, {"words": 3}]   # сумма = 10, как и распознано
    bp = pipeline_smart.whisper_breakpoints(blocks, "audio.mp3")
    assert bp[0] == 0.0
    assert bp[-1] == 12.0
    assert bp == sorted(bp)
    assert len(bp) == len(blocks) + 1


def test_whisper_breakpoints_none_on_severe_mismatch(monkeypatch):
    """Whisper распознал заметно меньше слов, чем в сценарии — доверять его
    тайм-кодам опаснее, чем прежней оценке по словам."""
    monkeypatch.setattr(pipeline_smart, "WHISPER_ENABLED", True)
    monkeypatch.setattr(pipeline_smart, "transcribe_words", lambda audio_path: [(0.0, 0.1, "w")])
    blocks = [{"words": 50}, {"words": 50}]
    assert pipeline_smart.whisper_breakpoints(blocks, "audio.mp3") is None


def test_whisper_breakpoints_none_on_transcription_failure(monkeypatch):
    monkeypatch.setattr(pipeline_smart, "WHISPER_ENABLED", True)

    def boom(audio_path):
        raise RuntimeError("модель не встала")

    monkeypatch.setattr(pipeline_smart, "transcribe_words", boom)
    assert pipeline_smart.whisper_breakpoints([{"words": 3}], "audio.mp3") is None


def test_write_srt_distributes_time_by_sentence_word_count(tmp_path):
    blocks = [{"text": "Раз два три. Четыре пять шесть семь восемь девять."}]
    breakpoints = [0.0, 9.0]
    out = tmp_path / "subtitles.srt"
    pipeline_smart.write_srt(blocks, breakpoints, str(out))
    content = out.read_text(encoding="utf-8")
    assert "Раз два три." in content
    assert "Четыре пять шесть семь восемь девять." in content
    # первое предложение (3 слова из 9) должно занять примерно треть окна
    first_cue_end = content.split("-->")[1].split("\n")[0].strip()
    assert first_cue_end.startswith("00:00:03")


def test_write_srt_skips_empty_blocks(tmp_path):
    out = tmp_path / "s.srt"
    pipeline_smart.write_srt([{"text": "   "}], [0.0, 1.0], str(out))
    assert not out.exists()


@pytest.mark.skipif(not HAS_FFMPEG, reason="нужен ffmpeg")
def test_whisper_align_end_to_end_uses_real_breakpoints(tmp_path, monkeypatch):
    """Whisper включён (замокан) -> итоговые durs должны идти от РЕАЛЬНЫХ
    границ, а не от word-count оценки, и субтитры должны быть записаны."""
    d = tmp_path / "v"
    media = d / "media"
    media.mkdir(parents=True)
    (d / "script.txt").write_text(
        "=== HOOK === Раз два три четыре.[pause]Пять шесть семь.\n"
        "=== FINAL === Восемь девять десять.\n", encoding="utf-8")
    for i in range(4):
        Image.new("RGB", (640, 360), (i * 40, 10, 200)).save(media / f"{i:03d}.jpg")
    audio_len = 10.0
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"sine=frequency=300:duration={audio_len}",
                    "-c:a", "libmp3lame", str(d / "audio.mp3")], check=True)

    monkeypatch.setattr(pipeline_smart, "WHISPER_ENABLED", True)
    fake_words = [(i * 1.0, i * 1.0 + 0.3, f"w{i}") for i in range(10)]
    monkeypatch.setattr(pipeline_smart, "transcribe_words", lambda audio_path: fake_words)

    env = dict(os.environ, PARALLAX="0", PEXELS_API_KEY="")
    # WHISPER_ENABLED монкипатчится в процессе pytest, а pipeline_smart.py
    # запускается ОТДЕЛЬНЫМ процессом (subprocess) — монкипатч туда не долетит,
    # поэтому здесь вызываем main() прямо в этом процессе, без subprocess.
    monkeypatch.setattr(pipeline_smart, "VIDEO_FOLDER", str(d))
    monkeypatch.setattr(pipeline_smart, "SCRIPT_FILE", str(d / "script.txt"))
    monkeypatch.setattr(pipeline_smart, "MEDIA_FOLDER", str(media))
    monkeypatch.setattr(pipeline_smart, "OUTPUT_FILE", str(d / "final.mp4"))
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(d / "temp_smart"))
    monkeypatch.setattr(pipeline_smart, "AUDIO_FILE", str(d / "audio.mp3"))
    monkeypatch.setattr(pipeline_smart, "LOCAL_MEDIA", pipeline_smart.scan_local_media())
    monkeypatch.setattr(pipeline_smart, "PEXELS_API_KEY", "")
    monkeypatch.setattr(pipeline_smart, "PARALLAX_ENABLED", False)

    rc = pipeline_smart.main()
    assert rc == 0
    assert (d / "final.mp4").exists()
    assert (d / "subtitles.srt").exists()


# =====================================================================
#  Режиссура: решения по смыслу текста и по самой картинке
# =====================================================================

import director  # noqa: E402


def _blk(text, section="BLOCK 1: X", pause=0.8, stat=None):
    return {"text": text, "words": len(text.split()), "pause_after": pause,
            "section": section, "stat": stat}


# ---------- D1: беат = работа реплики, а не тема ----------

def test_beat_classifies_the_work_the_line_does():
    cases = [
        ("На самом деле всё было наоборот.", "BLOCK 1: X", False, director.BEAT_REVEAL),
        ("А ты когда-нибудь держал меч?", "BLOCK 1: X", False, director.BEAT_QUESTION),
        ("Во-первых, вес. Во-вторых, баланс.", "BLOCK 1: X", False, director.BEAT_LIST),
        ("Вот и вся легенда.", "BLOCK 2: X", False, director.BEAT_QUIET),
        ("Он рванул вперёд!", "BLOCK 1: X", False, director.BEAT_ACTION),
        ("Кузнец работал в мастерской.", "BLOCK 1: X", False, director.BEAT_NEUTRAL),
        ("Простая фраза без примет.", "FINAL", False, director.BEAT_QUIET),
    ]
    for text, sec, stat, expected in cases:
        assert director.analyze_beat(text, sec, stat) == expected, text


def test_stat_beat_wins_over_everything_because_the_number_must_be_readable():
    """Плашку зритель обязан успеть прочитать — это сильнее любого другого
    намерения кадра, поэтому цифра перебивает и слом, и вопрос."""
    assert director.analyze_beat("На самом деле меч легче?", "BLOCK 1: X", has_stat=True) \
        == director.BEAT_STAT
    # цифра в самом тексте тоже считается плашкой-обязательством
    assert director.analyze_beat("Он весил 1,3 килограмма.", "BLOCK 1: X") == director.BEAT_STAT
    assert director.analyze_beat("Потери выросли на 40%.", "BLOCK 1: X") == director.BEAT_STAT
    # но одиночная цифра в обычной фразе — не статистика
    assert director.analyze_beat("У него было 5 мечей.", "BLOCK 1: X") != director.BEAT_STAT


def test_reveal_opener_only_counts_at_the_start():
    """«Но» в начале реплики — слом (ЧАСТЬ 8 разрешает так начинать).
    То же слово внутри фразы — обычный союз, маркером быть не должно."""
    assert director.analyze_beat("Но всё вышло иначе.", "BLOCK 1: X") == director.BEAT_REVEAL
    assert director.analyze_beat("Меч был тяжёлым, но удобным.", "BLOCK 1: X") \
        == director.BEAT_NEUTRAL


def test_unrecognised_line_returns_decision_to_the_hash():
    """Ключевое свойство всей режиссуры: хуже случайного хэша быть не может.
    Не распознали — intent None, и наверху срабатывает прежний путь."""
    assert director.zoom_intent(director.BEAT_NEUTRAL, is_section_start=False) is None


def test_zoom_intent_reads_as_editing_grammar():
    assert director.zoom_intent(director.BEAT_STAT) is True        # наезд на цифру
    assert director.zoom_intent(director.BEAT_REVEAL) is True      # наезд на слом
    assert director.zoom_intent(director.BEAT_QUIET) is False      # отъезд на финале
    # нейтральная открывашка раздела — общий план
    assert director.zoom_intent(director.BEAT_NEUTRAL, is_section_start=True) is False
    # но настоящий беат сильнее формальности открывашки
    assert director.zoom_intent(director.BEAT_STAT, is_section_start=True) is True


def test_pace_gives_numbers_and_reveals_more_screen_time_than_lists():
    assert director.pace_multiplier(director.BEAT_STAT) > 1.0
    assert director.pace_multiplier(director.BEAT_REVEAL) > 1.0
    assert director.pace_multiplier(director.BEAT_LIST) < 1.0
    assert director.pace_multiplier(director.BEAT_NEUTRAL) == 1.0
    assert director.pace_multiplier("что-то неизвестное") == 1.0


def test_beat_pace_redistributes_time_but_never_changes_total():
    """Темп по беатам меняет ПРОПОРЦИИ, а не общую длину: синхрон с озвучкой
    держит fit_to_total, и он не должен ломаться от этих множителей."""
    blocks = [_blk("Во-первых, вес. Во-вторых, баланс.", "BLOCK 1: X"),
              _blk("На самом деле всё было наоборот.", "BLOCK 1: X"),
              _blk("Кузнец работал в мастерской.", "BLOCK 1: X")]
    total = 30.0
    plain = pipeline_smart.block_durations(blocks, total, beat_pace=False)
    paced = pipeline_smart.block_durations(blocks, total, beat_pace=True)
    assert abs(sum(plain) - total) < 1e-6
    assert abs(sum(paced) - total) < 1e-6
    assert paced != plain, "темп по беатам вообще не применился"
    # перечисление ужалось относительно слома
    assert paced[0] / plain[0] < paced[1] / plain[1]


def test_beat_pace_never_overrides_real_speech_timing():
    """Когда границы взяты из реальной речи (Whisper), они и есть факт.
    Растянуть их «под беат» значит увести кадр от фразы, ради совпадения с
    которой Whisper и запускали."""
    blocks = [_blk("Во-первых, вес.", "BLOCK 1: X"),
              _blk("На самом деле всё наоборот.", "BLOCK 1: X")]
    raw = [6.0, 9.0]
    a = pipeline_smart.block_durations(blocks, 15.0, raw_override=raw, beat_pace=True)
    b = pipeline_smart.block_durations(blocks, 15.0, raw_override=raw, beat_pace=False)
    assert a == b, "beat_pace просочился в тайминг по факту речи"


# ---------- D2: пан на объект, а не наугад ----------

def _subject_image(path, subj_x, w=480, h=270):
    """Тёмный фон + светлый объект в заданной точке по горизонтали."""
    img = Image.new("RGB", (w, h), (24, 26, 30))
    d = ImageDraw.Draw(img)
    cx = int(w * subj_x)
    d.rectangle([cx - 18, int(h * 0.2), cx + 18, int(h * 0.8)], fill=(230, 225, 210))
    img.save(path, quality=95)


def test_composition_finds_the_subject_and_pans_toward_it(tmp_path):
    """Симптом: на кадре с объектом слева пан с вероятностью 1/2 уезжал
    вправо и выдавливал объект за рамку — направление выбирал md5 имени файла."""
    left, right = tmp_path / "l.jpg", tmp_path / "r.jpg"
    _subject_image(left, 0.22)
    _subject_image(right, 0.78)
    cl = director.frame_composition(str(left))
    cr = director.frame_composition(str(right))
    assert cl["cx"] < -director.COMPOSITION_DEADZONE
    assert cr["cx"] > director.COMPOSITION_DEADZONE
    # dx<0 = кадр идёт влево, к объекту; dx>0 = вправо
    assert director.pan_for_composition(cl, None)[0] == -1
    assert director.pan_for_composition(cr, None)[0] == 1


def test_centred_frame_hands_pan_back_to_the_hash(tmp_path):
    """Отцентрованный кадр — мнения нет, иначе на симметричных кадрах пан
    всегда шёл бы в одну сторону. Возвращаем фоллбэк как есть."""
    p = tmp_path / "c.jpg"
    _subject_image(p, 0.5)
    comp = director.frame_composition(str(p))
    assert abs(comp["cx"]) <= director.COMPOSITION_DEADZONE
    sentinel = ("FALLBACK",)
    assert director.pan_for_composition(comp, sentinel) is sentinel


def test_pan_for_composition_survives_missing_analysis():
    sentinel = (1, -1)
    assert director.pan_for_composition(None, sentinel) is sentinel
    assert director.pan_for_composition({}, sentinel) is sentinel


def test_frame_composition_never_raises_on_a_broken_file(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image at all")
    comp = director.frame_composition(str(bad))
    assert comp == director.NEUTRAL_COMPOSITION
    assert director.frame_composition(str(tmp_path / "missing.jpg")) == director.NEUTRAL_COMPOSITION


# ---------- D3: shot matching ----------

def _flat_image(path, level, w=320, h=180):
    """Почти ровная заливка заданной светлоты — минимальный разброс яркости."""
    img = Image.new("RGB", (w, h), (level, level, level))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 40, 40], fill=(min(255, level + 6),) * 3)   # чуть-чуть структуры
    img.save(path, quality=95)


def test_shot_matching_pulls_dark_and_washed_out_shots_toward_each_other(tmp_path):
    """Симптом: одна и та же кривая eq ложилась и на тёмный кадр, и на
    выбеленный — первый сваливался в кашу, второй оставался блёклым."""
    dark, bright = tmp_path / "d.jpg", tmp_path / "b.jpg"
    _flat_image(dark, 30)
    _flat_image(bright, 200)
    gd = director.grade_offsets(director.frame_composition(str(dark)))
    gb = director.grade_offsets(director.frame_composition(str(bright)))
    assert gd["db"] > 0, "тёмный кадр надо приподнять"
    assert gb["db"] < 0, "пересвеченный кадр надо прибрать"
    assert gd["db"] > gb["db"]


def test_flat_shot_gets_more_contrast_than_a_punchy_one(tmp_path):
    flat = tmp_path / "f.jpg"
    _flat_image(flat, 120)
    punchy = tmp_path / "p.jpg"
    img = Image.new("RGB", (320, 180), (0, 0, 0))
    ImageDraw.Draw(img).rectangle([0, 0, 160, 180], fill=(255, 255, 255))
    img.save(punchy, quality=95)
    gf = director.grade_offsets(director.frame_composition(str(flat)))
    gp = director.grade_offsets(director.frame_composition(str(punchy)))
    assert gf["dc"] > gp["dc"]


def test_grade_offsets_neutral_without_analysis():
    assert director.grade_offsets(None) == {"dc": 0.0, "db": 0.0, "ds": 0.0}


def test_film_look_stays_in_sane_range_for_any_shot(tmp_path):
    """Поправки shot matching не должны выносить eq за разумные пределы —
    иначе «сведение кадров» превратится в перекраску."""
    for level in (5, 30, 120, 200, 250):
        p = tmp_path / f"l{level}.jpg"
        _flat_image(p, level)
        comp = director.frame_composition(str(p))
        for section in ("HOOK", "BLOCK 1: X", "FINAL"):
            for h in (0, 12345, 4294967295):
                vf = pipeline_smart.film_look(h, section, comp)
                m = re.search(r'eq=contrast=([\d.]+):saturation=([\d.]+):brightness=(-?[\d.]+)', vf)
                assert m, vf
                c, s_, b = (float(x) for x in m.groups())
                assert 0.85 <= c <= 1.35, f"contrast {c}"
                assert 0.60 <= s_ <= 1.25, f"saturation {s_}"
                assert -0.06 <= b <= 0.09, f"brightness {b}"


def test_assemble_shares_the_same_shot_matching(tmp_path):
    """assemble.py и pipeline_smart.py не должны разъезжаться по грейду —
    ровно этот класс расхождения дал баги B4/B5."""
    p = tmp_path / "x.jpg"
    _flat_image(p, 28)
    comp = director.frame_composition(str(p))
    plain = assemble.film_look("stock", 999)
    matched = assemble.film_look("stock", 999, comp)
    assert plain != matched, "shot matching в assemble.py не применяется"


# ---------- D4: переход по паузе сценария ----------

def test_transition_follows_the_pause_the_writer_wrote():
    """Симптом: заметный переход мог попасть ровно в середину неразрывной
    мысли — его выбирал хэш номера стыка, а не пауза в сценарии."""
    # длинная пауза (две [pause] подряд) — всегда заметный переход
    assert all(director.transition_kind(1.6, h) == "visible" for h in range(20))
    # пауз нет / [short pause] — мысль не прерывалась, только жёсткий рез
    assert all(director.transition_kind(0.0, h) == "hard" for h in range(20))
    assert all(director.transition_kind(0.4, h) == "hard" for h in range(20))
    # нет данных — решает вызывающий код по-старому
    assert director.transition_kind(None, 0) is None


def test_ordinary_pause_keeps_hard_cuts_the_norm():
    """Дисcольв в реальном монтаже — исключение. На рядовой [pause] заметный
    переход должен быть редкостью, иначе ролик поплывёт."""
    kinds = [director.transition_kind(0.8, h) for h in range(400)]
    visible = kinds.count("visible")
    assert 0.15 < visible / len(kinds) < 0.35, f"заметных переходов {visible}/400"


def test_plan_transitions_uses_pauses_and_keeps_budget_exact():
    sections = ["HOOK"] * 3 + ["BLOCK 1: A"] * 3 + ["BLOCK 2: B"] * 3
    pauses = [0.0, 1.6, 0.0, 0.4, 1.6, 0.0, 0.0, 0.4, 0.0]
    plan = pipeline_smart.plan_transitions(sections, pauses=pauses)
    assert len(plan) == len(sections) - 1
    # бюджет обязан по-прежнему совпадать с тем, что реально съест цепочка
    durs = [8.0] * len(sections)
    cum = durs[0]
    for i in range(1, len(sections)):
        cum = cum + durs[i] - plan[i - 1][1]
    assert abs((sum(durs) - cum) - sum(d for _, d in plan)) < 1e-6
    # plan[k] — стык между блоками k и k+1, то есть пауза pauses[k].
    # Длинная пауза внутри раздела -> заметный переход.
    assert plan[1][1] == pipeline_smart.XFADE_DUR, "длинная пауза не дала заметный переход"
    assert plan[4][1] == pipeline_smart.XFADE_DUR
    # Мысль не прерывалась (pause 0.0) внутри раздела -> жёсткий рез.
    assert plan[0][1] == pipeline_smart.XFADE_DUR_HARD
    assert plan[6][1] == pipeline_smart.XFADE_DUR_HARD
    # [short pause] — тоже не повод рвать мысль.
    assert plan[7][1] == pipeline_smart.XFADE_DUR_HARD
    # Граница раздела перебивает паузу: тут pause 0.0, но переход обязан быть заметным.
    assert plan[2][0] in pipeline_smart.BOUNDARY_TRANSITIONS
    assert plan[2][1] == pipeline_smart.XFADE_DUR


def test_plan_transitions_without_pauses_is_unchanged():
    """Обратная совместимость: без pauses — ровно прежнее поведение по хэшу."""
    sections = ["HOOK"] * 4 + ["BLOCK 1: A"] * 4
    assert pipeline_smart.plan_transitions(sections) == pipeline_smart.plan_transitions(sections)
    with_none = pipeline_smart.plan_transitions(sections, pauses=None)
    assert with_none == pipeline_smart.plan_transitions(sections)


def test_section_boundary_still_wins_over_pause():
    """Смена раздела — всегда заметный переход, даже если паузы там нет."""
    sections = ["HOOK", "BLOCK 1: A"]
    plan = pipeline_smart.plan_transitions(sections, pauses=[0.0])
    assert plan[0][0] in pipeline_smart.BOUNDARY_TRANSITIONS
    assert plan[0][1] == pipeline_smart.XFADE_DUR


def test_clip_cache_key_covers_director_decisions():
    """ЧАСТЬ 21: имя кэшированного клипа обязано содержать хэш ВСЕХ входов.
    Режиссёрские решения (беат, композиция, светлота кадра) — такие же входы
    рендера, как длительность и титр. Без них правка логики режиссуры молча
    переиспользовала бы кадры, снятые по прежним решениям."""
    b = _blk("Кузнец работал.", "BLOCK 1: X")
    base = dict(b=b, d=5.0, title="ТЕМА", stat=None, src="/m/001.jpg", reuse=False,
                comp=dict(director.NEUTRAL_COMPOSITION), beat=director.BEAT_NEUTRAL,
                is_section_start=False, is_photo=True)

    def recipe(**over):
        a = dict(base, **over)
        return pipeline_smart._render_recipe(a["b"], a["d"], a["title"], a["stat"], a["src"],
                                             a["reuse"], a["comp"], a["beat"],
                                             a["is_section_start"], a["is_photo"])

    ref = recipe()
    assert recipe() == ref, "рецепт обязан быть детерминированным"
    # каждый вход обязан менять рецепт
    assert recipe(beat=director.BEAT_STAT) != ref
    assert recipe(is_section_start=True) != ref
    assert recipe(comp=dict(director.NEUTRAL_COMPOSITION, cx=0.8)) != ref
    assert recipe(comp=dict(director.NEUTRAL_COMPOSITION, luma=0.9)) != ref
    assert recipe(comp=dict(director.NEUTRAL_COMPOSITION, spread=0.05)) != ref
    assert recipe(d=6.0) != ref
    assert recipe(title="ДРУГОЕ") != ref
    assert recipe(src="/m/002.jpg") != ref
    assert recipe(reuse=True) != ref
    assert recipe(is_photo=False) != ref
    # ...но шум в третьем знаке замера не должен гонять перерендер зря
    assert recipe(comp=dict(director.NEUTRAL_COMPOSITION, cx=0.0004)) == ref


def test_render_recipe_survives_missing_composition():
    b = _blk("Кузнец работал.", "BLOCK 1: X")
    r = pipeline_smart._render_recipe(b, 5.0, None, None, "/m/x.mp4", False, None,
                                      director.BEAT_NEUTRAL, False, False)
    assert isinstance(r, str) and r
