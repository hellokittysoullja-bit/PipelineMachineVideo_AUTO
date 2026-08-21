"""Speech Direction Engine, Stage B (scripts/speech_generate.py) — тесты.

Живой ElevenLabs НЕ вызывается нигде в этом файле (нет ключа, и тратить
чужие деньги на тест — не та цена, которую стоит платить за уверенность
в коде, см. докстринг speech_generate.py). Сетевой слой
(call_elevenlabs_with_timestamps/_with_retry) — либо подменяется реальной
ffmpeg-сгенерированной аудио + синтетическим alignment (round-trip/
decision-logic тесты — та же общая идея, что moker в test_render_fault_tolerance.py
про 'реальный файл, не фейковый'), либо проверяется напрямую через
urllib.request.urlopen-мок (redaction/формат запроса) — вот тут уже без
сети, чистый контракт.

Требует ffmpeg/ffprobe в PATH (та же зависимость, что test_smoke.py/
test_render_fault_tolerance.py)."""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
SPEECH_GENERATE = os.path.join(SCRIPTS_DIR, "speech_generate.py")
SPEECH_PLANNER = os.path.join(SCRIPTS_DIR, "speech_planner.py")
sys.path.insert(0, SCRIPTS_DIR)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe не найдены в PATH",
)

sys.argv = ["speech_generate.py", tempfile.gettempdir()]
import speech_generate as sg   # noqa: E402
import speech_planner          # noqa: E402


SCRIPT_TEXT = (
    "=== HOOK === Представь: три тысячи лет назад.[pause]"
    "Меч весил больше, чем ты думаешь.\n"
    "=== BLOCK 1: Вес меча === Учёные нашли образец в 1932 году.[short pause]"
    "Но что если это неправда?[pause]На самом деле — всё было иначе.[pause]"
    "Исследование 2019 года показало обратное.[pause]"
    "Средний вес составлял 1.2 килограмма.[stat:1.2 КГ][pause]"
    "Вот почему это важно для тебя.[pause]Переходим дальше.\n"
    "=== FINAL === Вот и всё, что мы знаем на сегодня.\n"
)


def _make_units(tmp_path):
    script_path = tmp_path / "script.txt"
    script_path.write_text(SCRIPT_TEXT, encoding="utf-8")
    saved_argv = sys.argv
    sys.argv = ["pipeline_smart.py", str(tmp_path)]
    import pipeline_smart
    sys.argv = saved_argv
    blocks = pipeline_smart.parse_blocks(str(script_path))
    units = speech_planner.build_units(blocks)
    speech_planner.assign_chapter_arcs(units)
    return units


# ---------- сегментация ----------

def test_segment_fragments_never_crosses_section_boundary(tmp_path):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)
    for frag in fragments:
        sections = {u["section"] for u in frag}
        assert len(sections) == 1


def test_segment_fragments_never_mixes_arc_stages(tmp_path):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)
    for frag in fragments:
        stages = {u["arc_stage"] for u in frag}
        assert len(stages) == 1


def test_segment_fragments_respects_max_units():
    units = [{"section": "BLOCK 1", "arc_stage": "доказательство", "words": 1,
              "unit_id": f"u{i}", "text": "слово", "tag": None}
             for i in range(20)]
    fragments = sg.segment_fragments(units, max_units=4, max_words=999)
    assert all(len(f) <= 4 for f in fragments)
    assert sum(len(f) for f in fragments) == 20


def test_segment_fragments_respects_max_words():
    units = [{"section": "BLOCK 1", "arc_stage": "доказательство", "words": 10,
              "unit_id": f"u{i}", "text": "слово " * 10, "tag": None}
             for i in range(5)]
    fragments = sg.segment_fragments(units, max_units=999, max_words=25)
    for f in fragments:
        assert sum(u["words"] for u in f) <= 25


def test_segment_fragments_preserves_unit_order(tmp_path):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)
    flat = [u["unit_id"] for f in fragments for u in f]
    assert flat == [u["unit_id"] for u in units]


# ---------- реконструкция текста для TTS ----------

def test_fragment_text_includes_pause_tags(tmp_path):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)
    block1_frag = next(f for f in fragments if f[0]["section"].startswith("BLOCK 1")
                        and f[0]["arc_stage"] == "доказательство")
    text = sg.fragment_text_for_tts(block1_frag)
    assert "[pause]" in text


def test_fragment_text_includes_trailing_tag_on_last_unit():
    units = [{"unit_id": "a", "text": "Слово раз", "words": 2, "tag": "[pause]"}]
    text = sg.fragment_text_for_tts(units)
    assert text.strip().endswith("[pause]")


def test_fragment_text_energetic_prefix_only_first_hook_fragment(tmp_path):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)
    for i, frag in enumerate(fragments):
        is_first = sg.is_first_hook_fragment(frag, fragments, i)
        text = sg.fragment_text_for_tts(frag, prefix_energetic=is_first)
        if frag[0]["section"].startswith("HOOK") and i == 0:
            assert text.startswith("[energetic]")
        else:
            assert "[energetic]" not in text


def test_fragment_text_no_energetic_for_block_or_final(tmp_path):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)
    for i, frag in enumerate(fragments):
        if not frag[0]["section"].startswith("HOOK"):
            assert sg.is_first_hook_fragment(frag, fragments, i) is False


# ---------- кэш ----------

def test_cache_round_trip(tmp_path):
    key = sg.cache_key("текст", "voice1", "eleven_v3", 1)
    audio_bytes = b"\x00\x01fake-mp3-bytes"
    alignment = [("а", 0.0, 0.1), ("б", 0.1, 0.2)]
    sg.save_to_cache(str(tmp_path), key, audio_bytes, alignment)
    loaded = sg.load_from_cache(str(tmp_path), key)
    assert loaded is not None
    audio_path, loaded_alignment = loaded
    assert open(audio_path, "rb").read() == audio_bytes
    assert loaded_alignment == [list(a) for a in alignment] or loaded_alignment == alignment


def test_cache_miss_returns_none(tmp_path):
    assert sg.load_from_cache(str(tmp_path), "nonexistent_key") is None


def test_cache_key_differs_by_attempt():
    k1 = sg.cache_key("текст", "v", "m", 1)
    k2 = sg.cache_key("текст", "v", "m", 2)
    assert k1 != k2


def test_cache_key_differs_by_text():
    k1 = sg.cache_key("текст один", "v", "m", 1)
    k2 = sg.cache_key("текст два", "v", "m", 1)
    assert k1 != k2


# ---------- решение: accept / fix_boundary / regenerate ----------

def _eval(tempo_ok=True, pause_ok=True, energy_ok=True, signal=True):
    return {
        "units": [{"unit_id": "u0", "signal": signal, "tempo_ok": tempo_ok, "pause_ok": pause_ok}],
        "energy": {"scored": True, "energy_ok": energy_ok},
    }


def test_decide_accepts_when_everything_ok():
    action, _ = sg.decide_fragment_action(_eval(), attempt=1)
    assert action == "accept"


def test_decide_fix_boundary_when_only_pause_off():
    action, _ = sg.decide_fragment_action(_eval(pause_ok=False), attempt=1)
    assert action == "fix_boundary"


def test_decide_regenerate_when_tempo_off_and_attempts_remain():
    action, _ = sg.decide_fragment_action(_eval(tempo_ok=False), attempt=1)
    assert action == "regenerate"


def test_decide_regenerate_when_energy_off_and_attempts_remain():
    action, _ = sg.decide_fragment_action(_eval(energy_ok=False), attempt=1)
    assert action == "regenerate"


def test_decide_accepts_with_warning_when_attempts_exhausted():
    action, reason = sg.decide_fragment_action(_eval(tempo_ok=False), attempt=sg.SPEECH_GEN_MAX_ATTEMPTS)
    assert action == "accept"
    assert "лимит" in reason


def test_decide_never_exceeds_max_attempts_hard_cap():
    # Даже с attempt искусственно ниже кэпа, кэп сам по себе <= 2 (см.
    # отдельный subprocess-тест на ENV ниже) — здесь просто фиксируем,
    # что дальше MAX_ATTEMPTS_HARD_CAP модуль не пойдёт по определению.
    assert sg.SPEECH_GEN_MAX_ATTEMPTS <= sg.MAX_ATTEMPTS_HARD_CAP == 2


def test_max_attempts_env_cannot_exceed_hard_cap():
    env = dict(os.environ, SPEECH_GEN_MAX_ATTEMPTS="50")
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['speech_generate.py','.']; sys.path.insert(0, %r); "
         "import speech_generate as sg; print(sg.SPEECH_GEN_MAX_ATTEMPTS)" % SCRIPTS_DIR],
        capture_output=True, text=True, env=env, timeout=30)
    assert r.stdout.strip() == "2", f"hard cap bypassed: stdout={r.stdout!r} stderr={r.stderr[-300:]}"


# ---------- API-ключ никогда не логируется ----------

def test_api_key_redacted_from_http_error(monkeypatch):
    secret = "sk_super_secret_key_98765"

    class FakeHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("http://x", 401, "unauthorized", {}, None)

        def read(self):
            return f'{{"error": "bad key {secret}"}}'.encode()

    def raise_it(*a, **kw):
        raise FakeHTTPError()

    monkeypatch.setattr(sg.urllib.request, "urlopen", raise_it)
    with pytest.raises(sg.ElevenLabsAPIError) as exc_info:
        sg.call_elevenlabs_with_timestamps("текст", "voice_x", secret, "eleven_v3")
    assert secret not in str(exc_info.value)
    assert "REDACTED" in str(exc_info.value)


def test_api_key_never_in_request_url_or_query(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        raise urllib.error.URLError("stop before real network")

    monkeypatch.setattr(sg.urllib.request, "urlopen", fake_urlopen)
    secret = "sk_test_key_should_be_in_header_only"
    with pytest.raises(sg.ElevenLabsAPIError):
        sg.call_elevenlabs_with_timestamps("текст", "voice_x", secret, "eleven_v3")
    assert secret not in captured["url"]
    # urllib капитализирует заголовки (Xi-api-key) — ищем без учёта регистра.
    header_values = {k.lower(): v for k, v in captured["headers"].items()}
    assert header_values.get("xi-api-key") == secret


# ---------- жёсткий лимит живых вызовов за прогон ----------

def test_max_calls_per_run_stops_before_exceeding(tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "SPEECH_GEN_MAX_CALLS_PER_RUN", 1)

    def fake_ok(*a, **kw):
        return b"fake-audio-bytes", [("a", 0.0, 0.1)]
    monkeypatch.setattr(sg, "call_elevenlabs_with_retry_on_transient_error", fake_ok)

    units = [
        {"unit_id": "HOOK#0", "section": "HOOK", "text": "Раз два три", "words": 3, "tag": None,
         "chapter_id": 0, "arc_stage": "hook", "target_wpm": 125,
         "rhetorical_kind": "connective", "target_range_sec": [0, 0]},
        {"unit_id": "BLOCK 1#0", "section": "BLOCK 1", "text": "Четыре пять шесть", "words": 3, "tag": None,
         "chapter_id": 1, "arc_stage": "постановка", "target_wpm": 125,
         "rhetorical_kind": "connective", "target_range_sec": [0, 0]},
    ]
    fragments = sg.segment_fragments(units)
    with pytest.raises(sg.SpeechGenerationLimitError):
        sg.generate_all_fragments(str(tmp_path), fragments, "k", "v", "eleven_v3")


# ---------- сборка: round-trip совместимость с pipeline_smart.load_alignment_weights ----------

def _fake_generate(text, dur_hint_wpm=125):
    """Реальный короткий mp3 (ffmpeg) + синтетический равномерный alignment —
    тот же принцип 'настоящий файл, не заглушка', что и остальные тесты
    рендера в этой сессии."""
    words = max(1, len(text.split()))
    dur = max(0.3, words / (dur_hint_wpm / 60))
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur:.2f}",
                    "-c:a", "libmp3lame", tmp.name], capture_output=True, check=True)
    n = len(text)
    alignment = [(c, round(dur * i / n, 4), round(dur * (i + 1) / n, 4)) for i, c in enumerate(text)]
    with open(tmp.name, "rb") as f:
        audio_bytes = f.read()
    os.remove(tmp.name)
    return audio_bytes, alignment


def test_end_to_end_alignment_round_trips_through_pipeline_smart(tmp_path, monkeypatch):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)

    def fake_call(text, voice_id, api_key, model, previous_text=None, next_text=None):
        return _fake_generate(text)
    monkeypatch.setattr(sg, "call_elevenlabs_with_retry_on_transient_error", fake_call)

    results, live_calls, cache_hits = sg.generate_all_fragments(
        str(tmp_path), fragments, "fake_key", "fake_voice", "eleven_v3")
    assert live_calls > 0

    temp_dir = tmp_path / "media_plan" / "speech_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    audio_out = tmp_path / "audio.mp3"
    sg.concat_fragment_audio(results, str(audio_out), str(temp_dir))
    assert audio_out.exists()

    alignment_dir = tmp_path / "media_plan" / "alignment"
    sg.write_section_alignment_csvs(results, str(alignment_dir))
    csv_files = sorted(os.listdir(alignment_dir))
    assert csv_files == ["00.csv", "01.csv", "02.csv"]   # HOOK, BLOCK 1, FINAL

    for fname in csv_files:
        with open(alignment_dir / fname, encoding="utf-8") as f:
            header = next(csv.reader(f))
            assert header == ["char", "start", "end"]

    # pipeline_smart уже импортирован ГДЕ-ТО раньше в этом процессе со
    # своим VIDEO_FOLDER (модуль кэшируется — повторный import ничего не
    # переисполняет) — ALIGNMENT_DIR, вычисленный от НЕГО, указывал бы не
    # в tmp_path этого теста. Подменяем напрямую (тот же приём, что уже
    # использует tests/test_speech_validator.py), а не полагаемся на
    # повторный sys.argv-триггер, который сработал бы только при самом
    # первом импорте за весь процесс pytest.
    saved_argv = sys.argv
    sys.argv = ["pipeline_smart.py", str(tmp_path)]
    import pipeline_smart
    sys.argv = saved_argv
    monkeypatch.setattr(pipeline_smart, "ALIGNMENT_DIR", str(alignment_dir))
    blocks = pipeline_smart.parse_blocks(str(tmp_path / "script.txt"))
    weights = pipeline_smart.load_alignment_weights(blocks)
    assert weights is not None
    assert all(w is not None for w in weights), (
        "load_alignment_weights() откатилась на word-count хотя бы для одного "
        "блока — сгенерированный alignment не совместим с существующим форматом")
    assert all(w > 0 for w in weights)


def test_end_to_end_writes_final_timeline_with_all_units(tmp_path, monkeypatch):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)

    def fake_call(text, voice_id, api_key, model, previous_text=None, next_text=None):
        return _fake_generate(text)
    monkeypatch.setattr(sg, "call_elevenlabs_with_retry_on_transient_error", fake_call)

    results, _, _ = sg.generate_all_fragments(str(tmp_path), fragments, "k", "v", "eleven_v3")
    temp_dir = tmp_path / "media_plan" / "speech_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    audio_out = tmp_path / "audio.mp3"
    sg.concat_fragment_audio(results, str(audio_out), str(temp_dir))
    joins = sg.evaluate_joins([r["audio_path"] for r in results])
    timeline = sg.build_final_timeline(results, joins)
    assert len(timeline["units"]) == len(units)
    assert timeline["total_duration_sec"] > 0
    for entry in timeline["units"]:
        assert entry["arc_stage"] is not None
        assert entry["chapter_id"] is not None


def test_cache_prevents_repeat_live_call_on_rerun(tmp_path, monkeypatch):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)
    calls = {"n": 0}

    def fake_call(text, voice_id, api_key, model, previous_text=None, next_text=None):
        calls["n"] += 1
        return _fake_generate(text)
    monkeypatch.setattr(sg, "call_elevenlabs_with_retry_on_transient_error", fake_call)

    sg.generate_all_fragments(str(tmp_path), fragments, "k", "v", "eleven_v3")
    first_calls = calls["n"]
    assert first_calls > 0

    results2, live_calls2, cache_hits2 = sg.generate_all_fragments(
        str(tmp_path), fragments, "k", "v", "eleven_v3")
    assert live_calls2 == 0, "повторный прогон с тем же текстом обязан обойтись без живых вызовов"
    # >= len(fragments), не ==: фрагмент, которому на первом прогоне
    # понадобилось 2 попытки, на повторе тоже пройдёт обе (детерминированная
    # оценка того же текста) — каждая попытка своя запись в кэше, это
    # ОЖИДАЕМО больше одного хита на такой фрагмент, не баг.
    assert cache_hits2 >= len(fragments)
    assert cache_hits2 == first_calls, (
        "число кэш-хитов на повторе должно совпасть с числом живых вызовов "
        "первого прогона — каждая попытка кэшируется отдельно и должна найтись")


# ---------- main(): защитные проверки ----------

def test_main_refuses_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    r = subprocess.run([sys.executable, SPEECH_GENERATE, str(tmp_path)],
                       capture_output=True, text=True, timeout=30,
                       env={k: v for k, v in os.environ.items()
                            if k not in ("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID")})
    assert r.returncode != 0
    assert "ELEVENLABS_API_KEY" in r.stdout


def test_main_refuses_to_overwrite_existing_audio(tmp_path):
    (tmp_path / "audio.mp3").write_bytes(b"already here")
    env = dict(os.environ, ELEVENLABS_API_KEY="k", ELEVENLABS_VOICE_ID="v")
    r = subprocess.run([sys.executable, SPEECH_GENERATE, str(tmp_path)],
                       capture_output=True, text=True, timeout=30, env=env)
    assert r.returncode != 0
    assert "уже существует" in r.stdout


def test_main_requires_speech_plan_first(tmp_path):
    env = dict(os.environ, ELEVENLABS_API_KEY="k", ELEVENLABS_VOICE_ID="v")
    r = subprocess.run([sys.executable, SPEECH_GENERATE, str(tmp_path)],
                       capture_output=True, text=True, timeout=30, env=env)
    assert r.returncode != 0
    assert "speech_planner.py" in r.stdout
