"""Регрессии на баги, найденные аудитом генератора.

Каждый тест назван по симптому, который был виден в готовом ролике, а не по
имени функции: если тест снова покраснеет, сразу понятно, что именно вернулось.
Запуск: python -m pytest tests/test_regressions.py -v
"""
import json
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image

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
