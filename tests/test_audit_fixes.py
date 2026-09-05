"""Регрессионные тесты на находки глубокого аудита 04.09 (см.
docs/AUDIT_2026-09_DEEP.md). Каждый тест — воспроизведение конкретного
дефекта, который до правки проходил незамеченным."""
import io
import os
import re
import sys
import tempfile
import urllib.error

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402


# ---------- чанкование xfade и PHRASE LOCK ----------

def _blocks(n, per_section):
    blocks = []
    for i in range(n):
        sec = "HOOK" if i < 10 else f"BLOCK {(i - 10) // per_section + 1}"
        blocks.append({"section": sec, "text": f"фраза {i} слова", "words": 3, "is_subcut": False})
    return blocks


def test_effective_plan_zeroes_transition_at_chunk_entry():
    blocks = _blocks(120, 22)
    sections = [b["section"] for b in blocks]
    plan = ps.plan_transitions(sections, blocks)
    eff = ps.effective_transition_plan(plan, sections)
    bounds = ps._chunk_bounds(len(blocks), sections, ps.XFADE_CHUNK_SIZE)
    assert len(bounds) >= 3, "тест должен реально задействовать чанкование"
    dropped = {a for a, _b in bounds if a > 0}
    for j, (t, d) in enumerate(eff):
        if (j + 1) in dropped:
            assert d == 0.0
        else:
            assert d == plan[j][1]
    assert sum(d for _t, d in eff) == pytest.approx(ps.estimate_xfade_budget(blocks))


def _simulate_chunked_timeline(durs, plan, sections):
    """Старты клипов так, как их реально склеит xfade_chain_chunked: внутри
    чанка каждый переход сжимает таймлайн на this_dur, между чанками —
    concat без нахлёста."""
    bounds = ps._chunk_bounds(len(durs), sections, ps.XFADE_CHUNK_SIZE)
    starts, cum = [], 0.0
    for a, b in bounds:
        starts.append(cum)
        chunk_cum = cum + durs[a]
        for i in range(a + 1, b):
            this_dur = plan[i - 1][1]
            starts.append(chunk_cum - this_dur)
            chunk_cum = chunk_cum + durs[i] - this_dur
        cum = chunk_cum
    return starts, cum


def test_phrase_lock_stays_on_onsets_across_chunk_boundaries():
    """До правки после каждой границы чанка кадр отставал от фразы на XFADE_DUR
    накопительно (7 чанков — +2.5с), а phrase_timeline показывал дрейф 0."""
    blocks = _blocks(120, 22)
    sections = [b["section"] for b in blocks]
    onsets = [i * 3.0 for i in range(120)]
    total = 360.0
    plan = ps.plan_transitions(sections, blocks)
    eff = ps.effective_transition_plan(plan, sections)
    durs = ps.phrase_locked_durations(onsets, total, eff)
    assert durs is not None
    starts, video_len = _simulate_chunked_timeline(durs, eff, sections)
    worst = max(abs(starts[i] - onsets[i]) for i in range(120))
    assert worst <= 0.5 / ps.FPS + 1e-6, f"худший дрейф {worst:.3f}с"
    assert abs(video_len - total) <= 0.5 / ps.FPS + 1e-6
    # hook_visual_starts обязан совпадать с той же симуляцией
    vs = ps.hook_visual_starts(blocks, durs)
    assert max(abs(vs[i] - starts[i]) for i in range(120)) < 1e-6


def test_old_naive_plan_would_have_drifted():
    """Контроль, что тест выше вообще чувствителен: с неэффективным планом
    (переход на входе чанка учтён) дрейф после чанков ненулевой."""
    blocks = _blocks(120, 22)
    sections = [b["section"] for b in blocks]
    onsets = [i * 3.0 for i in range(120)]
    plan = ps.plan_transitions(sections, blocks)
    durs = ps.phrase_locked_durations(onsets, 360.0, plan)
    starts, _ = _simulate_chunked_timeline(durs, ps.effective_transition_plan(plan, sections), sections)
    assert max(abs(starts[i] - onsets[i]) for i in range(120)) > 0.3


# ---------- Ken Burns: без остановок внутри клипа ----------

def _eval_expr(expr, t):
    e = expr.replace("\\,", ",").replace("if(", "_if(").replace("lt(", "_lt(")
    return eval(e, {"_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: 1 if a < b else 0,
                    "pow": pow, "t": t})


@pytest.mark.parametrize("points", [
    [(0.0, 0.0), (0.5, 0.72), (0.85, 0.95), (1.0, 1.0)],
    [(0.0, 0.0), (0.55, 0.68), (0.85, 0.92), (1.0, 1.0)],
])
def test_piecewise_ease_is_monotone_and_never_stalls_at_nodes(points):
    expr = ps.piecewise_ease_expr("t", points)
    xs = [i / 1000 for i in range(1001)]
    ys = [_eval_expr(expr, x) for x in xs]
    assert ys[0] == pytest.approx(0.0, abs=1e-9) and ys[-1] == pytest.approx(1.0, abs=1e-9)
    assert all(ys[i + 1] >= ys[i] - 1e-12 for i in range(1000)), "перелёт/немонотонность"
    vel = [(ys[i + 1] - ys[i - 1]) / 0.002 for i in range(1, 1000)]
    for tn, _ in points[1:-1]:
        assert vel[int(tn * 1000) - 1] > 0.3, f"остановка камеры в узле t={tn}"
    # плавный старт и стоп сохранены (как у smoothstep)
    assert vel[0] < 0.05 and vel[-1] < 0.05


def test_piecewise_ease_two_points_is_exact_smoothstep():
    expr = ps.piecewise_ease_expr("t", [(0.0, 0.0), (1.0, 1.0)])
    for i in range(0, 101):
        t = i / 100
        assert _eval_expr(expr, t) == pytest.approx(3 * t ** 2 - 2 * t ** 3, abs=1e-9)


# ---------- local_photo: нумерованная папка ----------

def test_local_photo_numbered_folder_has_no_positional_fallback(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    for name in ("001_flow.jpg", "002_flow.jpg", "007_stock.jpg"):
        Image.new("RGB", (8, 8)).save(media / name)
    monkeypatch.setattr(ps, "MEDIA_FOLDER", str(media))
    monkeypatch.setattr(ps, "_LOCAL_PHOTOS_CACHE", None)
    assert ps.local_photo(0).endswith("001_flow.jpg")
    assert ps.local_photo(6).endswith("007_stock.jpg")
    assert ps.local_photo(4) is None, "слот 5 без своего файла обязан уйти в Pexels, а не взять 007"
    assert ps.local_photo(4, allow_cycle=True) is not None


def test_local_photo_unnumbered_folder_keeps_positional_order(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (8, 8)).save(media / name)
    monkeypatch.setattr(ps, "MEDIA_FOLDER", str(media))
    monkeypatch.setattr(ps, "_LOCAL_PHOTOS_CACHE", None)
    assert ps.local_photo(1).endswith("b.jpg")
    assert ps.local_photo(2) is None


# ---------- профиль доставки ----------

@pytest.mark.parametrize("text,expected", [
    ("", None), ("12", 12000), ("12M", 12000), ("8000k", 8000), ("8000K", 8000),
    ("7.5m", 7500), ("0", 0), ("abc", None), ("12Mbps", None),
])
def test_parse_bitrate_kbps(text, expected):
    assert ps.parse_bitrate_kbps(text) == expected


def test_delivery_maxrate_override_and_garbage(monkeypatch, capsys):
    monkeypatch.setattr(ps, "DELIVERY_PROFILE", "youtube")
    monkeypatch.setattr(ps, "DELIVERY_MAXRATE", "8000k")
    args = ps.final_pass_encode_args()
    assert args[args.index("-maxrate") + 1] == "8000k" and args[args.index("-bufsize") + 1] == "16000k"
    monkeypatch.setattr(ps, "DELIVERY_MAXRATE", "garbage")
    args = ps.final_pass_encode_args()   # не падает, профильный потолок
    assert args[args.index("-maxrate") + 1] == "12M"
    assert "не разобран" in capsys.readouterr().out
    monkeypatch.setattr(ps, "DELIVERY_MAXRATE", "0")
    assert "-maxrate" not in ps.final_pass_encode_args()


# ---------- Windows ----------

def test_ffmpeg_filter_path_escapes_drive_colon_and_backslashes():
    assert ps.ffmpeg_filter_path(r"C:\Users\me\assets\fonts\Benzin.ttf") == "C\\:/Users/me/assets/fonts/Benzin.ttf"
    assert ps.ffmpeg_filter_path("/usr/share/fonts/DejaVuSans.ttf") == "/usr/share/fonts/DejaVuSans.ttf"


def test_all_subprocess_text_calls_declare_utf8():
    """На Windows text=True без encoding декодирует вывод ffmpeg в cp1251 —
    кириллица в пути роняла verify_clip на КАЖДОМ клипе."""
    offenders = []
    for name in os.listdir(SCRIPTS):
        if not name.endswith(".py"):
            continue
        src = io.open(os.path.join(SCRIPTS, name), encoding="utf-8").read()
        for m in re.finditer(r"text=True(?![^\n]*encoding=)", src):
            offenders.append(f"{name}:{src[:m.start()].count(chr(10)) + 1}")
    assert not offenders, offenders


def test_render_workers_env_is_clamped(monkeypatch):
    monkeypatch.setenv("RENDER_WORKERS", "0")
    assert ps._render_workers_from_env() == 1
    monkeypatch.setenv("RENDER_WORKERS", "junk")
    assert ps._render_workers_from_env() >= 1
    monkeypatch.setenv("RENDER_WORKERS", "999")
    assert ps._render_workers_from_env() <= max(1, os.cpu_count() or 4)


# ---------- EXIF ----------

def test_image_size_as_rendered_honours_exif_orientation(tmp_path):
    p = tmp_path / "rot.jpg"
    img = Image.new("RGB", (120, 60), (200, 30, 30))
    exif = img.getexif()
    exif[0x0112] = 6   # Orientation: rotate 90
    img.save(p, exif=exif.tobytes())
    assert Image.open(p).size == (120, 60)
    assert ps.image_size_as_rendered(str(p)) == (60, 120)


# ---------- подбор ----------

def test_has_action_word_ignores_decor_and_false_stems():
    for text in ("Резная рукоять из слоновой кости.", "Заносчивый рыцарь смотрел свысока.",
                 "Штурман корабля.", "Горелка кузнеца.", "Он разительно отличался походкой."):
        assert not ps.has_action_word(text), text
    assert ps.has_action_word("Рыцари штурмом взяли крепость.")


def test_disambiguation_matches_whole_words_only():
    assert "european european" not in ps.disambiguate_search_query("europe castle sword")
    assert ps.disambiguate_search_query("swordfish market") == "swordfish market"


def test_pexels_broken_only_on_auth_error_or_long_streak(monkeypatch, capsys):
    monkeypatch.setattr(ps, "PEXELS_BROKEN", False)
    monkeypatch.setattr(ps, "PEXELS_FAIL_STREAK", 0)
    for _ in range(ps.PEXELS_FAIL_STREAK_LIMIT - 1):
        ps._note_pexels_failure(OSError("connection refused"), "Pexels [x]")
    assert ps.PEXELS_BROKEN is False
    ps._reset_pexels_streak()
    ps._note_pexels_failure(OSError("cdn"), "Pexels [x]")
    assert ps.PEXELS_BROKEN is False and ps.PEXELS_FAIL_STREAK == 1
    for _ in range(ps.PEXELS_FAIL_STREAK_LIMIT):
        ps._note_pexels_failure(OSError("cdn"), "Pexels [x]")
    assert ps.PEXELS_BROKEN is True
    monkeypatch.setattr(ps, "PEXELS_BROKEN", False)
    monkeypatch.setattr(ps, "PEXELS_FAIL_STREAK", 0)
    err = urllib.error.HTTPError("https://api.pexels.com", 403, "Forbidden", {}, None)
    ps._note_pexels_failure(err, "Pexels [x]")
    assert ps.PEXELS_BROKEN is True
    assert "403" in capsys.readouterr().out


# ---------- шотлист ----------

def test_write_shotlist_keeps_user_file_when_lock_not_applied(tmp_path):
    import json
    prev = {"shots": [{"index": 0, "text": "фраза", "file": "media/approved_by_user.jpg", "kind": "photo", "lock": True}]}
    shots = {0: {"index": 0, "section": "HOOK", "text": "фраза", "query": "q", "kind": "photo",
                 "file": "temp_smart/pexels_cache/0000_new.jpg", "source": "pexels", "clip": "c.mp4"}}
    data = json.load(open(ps.write_shotlist(str(tmp_path), shots, {}, prev=prev), encoding="utf-8"))
    shot = data["shots"][0]
    assert shot["lock"] is True and shot["file"] == "media/approved_by_user.jpg"
    assert shot["lock_applied"] is False
    assert shot["rendered_file_this_run"] == "temp_smart/pexels_cache/0000_new.jpg"


# ---------- аудио ----------

@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="нужен ffmpeg")
def test_decoded_audio_duration_ignores_mp3_container_padding(tmp_path):
    import subprocess
    mp3 = tmp_path / "a.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-c:a", "libmp3lame", str(mp3)], check=True)
    dec = ps.decoded_audio_duration(str(mp3))
    assert dec is not None and abs(dec - 3.0) < 0.03
