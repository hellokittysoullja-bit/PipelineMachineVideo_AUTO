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

import numpy as np
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


# ---------- решение: accept / accept_needs_boundary_repair / regenerate ----------

def _eval(tempo_ok=True, pause_ok=True, energy_ok=True, signal=True):
    return {
        "units": [{"unit_id": "u0", "signal": signal, "tempo_ok": tempo_ok, "pause_ok": pause_ok}],
        "energy": {"scored": True, "energy_ok": energy_ok},
    }


def test_decide_accepts_when_everything_ok():
    action, _ = sg.decide_fragment_action(_eval(), attempt=1)
    assert action == "accept"


def test_decide_accept_needs_boundary_repair_when_only_pause_off():
    action, _ = sg.decide_fragment_action(_eval(pause_ok=False), attempt=1)
    assert action == "accept_needs_boundary_repair"


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

    def fake_call(text, voice_id, api_key, model, previous_text=None, next_text=None, voice_settings=None):
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


def test_write_section_alignment_csvs_returns_correct_global_offsets(tmp_path, monkeypatch):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)

    def fake_call(text, voice_id, api_key, model, previous_text=None, next_text=None, voice_settings=None):
        return _fake_generate(text)
    monkeypatch.setattr(sg, "call_elevenlabs_with_retry_on_transient_error", fake_call)

    results, _, _ = sg.generate_all_fragments(str(tmp_path), fragments, "k", "v", "eleven_v3")
    alignment_dir = tmp_path / "media_plan" / "alignment"
    offsets = sg.write_section_alignment_csvs(results, str(alignment_dir))

    assert set(offsets) == {"HOOK", "BLOCK 1: ВЕС МЕЧА", "FINAL"}
    assert offsets["HOOK"] == 0.0, "первая секция всегда начинается на глобальном нуле"
    # HOOK's own audio duration -> BLOCK 1 must start exactly there (не
    # приблизительно — это РЕАЛЬНОЕ измерение через get_media_duration
    # на уже собранных фрагментах, та же величина, что использует
    # concat_fragment_audio для итогового audio.mp3).
    hook_total = sum(sg.pipeline_smart.get_media_duration(r["audio_path"])
                     for r in results if r["units"][0]["section"] == "HOOK")
    assert abs(offsets["BLOCK 1: ВЕС МЕЧА"] - hook_total) < 0.01
    assert offsets["FINAL"] > offsets["BLOCK 1: ВЕС МЕЧА"]


def test_write_section_offsets_round_trips_via_fix_pauses_loader(tmp_path):
    section_global_offsets = {"HOOK": 0.0, "BLOCK 1: ТЕСТ": 6.28}
    sg.write_section_offsets(str(tmp_path), section_global_offsets)

    saved_argv = sys.argv
    sys.argv = ["fix_pauses.py", str(tmp_path)]
    import fix_pauses
    sys.argv = saved_argv
    loaded = fix_pauses.load_section_offsets(str(tmp_path))
    assert loaded == section_global_offsets


def test_end_to_end_second_section_protected_window_matches_real_global_silence(tmp_path, monkeypatch):
    """РЕГРЕССИЯ на реальный найденный баг (не мок изолированной функции —
    полный путь speech_generate.py -> fix_pauses.py на РЕАЛЬНОМ собранном
    audio.mp3): protected_windows для секции ПОСЛЕ первой (BLOCK 1) без
    section_offsets.json физически не совпадали бы с тем, что реально
    находит silencedetect в итоговом audio.mp3 — их локальный ноль не
    там же, где глобальный. С section_offsets.json (пишет Stage B) —
    совпадают, проверено на РЕАЛЬНОМ ffmpeg silencedetect, не на цифрах,
    которые я сам придумал."""
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)

    def fake_call(text, voice_id, api_key, model, previous_text=None, next_text=None, voice_settings=None):
        return _fake_generate(text)
    monkeypatch.setattr(sg, "call_elevenlabs_with_retry_on_transient_error", fake_call)

    results, _, _ = sg.generate_all_fragments(str(tmp_path), fragments, "k", "v", "eleven_v3")
    temp_dir = tmp_path / "media_plan" / "speech_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    audio_out = tmp_path / "audio.mp3"
    sg.concat_fragment_audio(results, str(audio_out), str(temp_dir))
    alignment_dir = tmp_path / "media_plan" / "alignment"
    offsets = sg.write_section_alignment_csvs(results, str(alignment_dir))
    sg.write_section_offsets(str(tmp_path), offsets)

    # Синтетическое protected_window В ЛОКАЛЬНОМ времени BLOCK 1 (как
    # реально пишет speech_validator.py) — где-то в середине секции, не в
    # начале, чтобы гарантированно быть в НЕ-первой секции.
    block1_offset = offsets["BLOCK 1: ВЕС МЕЧА"]
    fake_local_window = [1.0, 1.5, 1.3, "BLOCK 1: ВЕС МЕЧА#0"]
    timeline_path = tmp_path / "media_plan" / "speech_timeline.json"
    timeline_path.write_text(json.dumps({"protected_windows": [fake_local_window]}), encoding="utf-8")

    saved_argv = sys.argv
    sys.argv = ["fix_pauses.py", str(tmp_path)]
    import fix_pauses
    sys.argv = saved_argv
    loaded = fix_pauses.load_protected_windows(str(tmp_path))
    assert len(loaded) == 1
    global_start, global_end, kept, unit_id = loaded[0]
    assert abs(global_start - (1.0 + block1_offset)) < 0.01
    assert abs(global_end - (1.5 + block1_offset)) < 0.01
    # ГЛАВНАЯ проверка: это глобальное время ДЕЙСТВИТЕЛЬНО валидно внутри
    # реального собранного audio.mp3 (не вышло за длительность файла) —
    # если бы offset не применился, global_start остался бы 1.0с, что
    # тоже технически "валидно" (меньше длительности), но указывало бы
    # на середину ХУКА, а не на BLOCK 1 — эту подмену чисто по диапазону
    # не поймать, поэтому проверяем через реальную длительность HOOK'а.
    hook_total = sum(sg.pipeline_smart.get_media_duration(r["audio_path"])
                     for r in results if r["units"][0]["section"] == "HOOK")
    assert global_start >= hook_total, (
        "offset-скорректированное окно обязано лежать ПОСЛЕ хука в глобальном "
        "времени — иначе offset не применился (тот самый баг)")


def test_end_to_end_writes_final_timeline_with_all_units(tmp_path, monkeypatch):
    units = _make_units(tmp_path)
    fragments = sg.segment_fragments(units)

    def fake_call(text, voice_id, api_key, model, previous_text=None, next_text=None, voice_settings=None):
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

    def fake_call(text, voice_id, api_key, model, previous_text=None, next_text=None, voice_settings=None):
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


# ---------- _build_fragment_text_and_spans: регрессия на реальный найденный баг ----------
# (span каждого unit'а после cue-префикса съезжал на len(cue)+1 символов —
# _unit_char_spans раньше не знал про cue вообще, дублировал построение
# текста ОТДЕЛЬНО от fragment_text_for_tts(). Пойман РУЧНЫМ прослеживанием
# арифметики, не тестом — синтетический равномерный alignment в остальных
# e2e-тестах маскировал проблему (любой смежный срез равномерно
# размеченного текста выглядит правдоподобно). Эти тесты — то, чего не
# хватало, чтобы поймать баг автоматически.)

def test_build_fragment_text_and_spans_matches_text_without_cue():
    units = [{"unit_id": "a", "text": "Первая фраза", "tag": "[pause]", "cue": None},
              {"unit_id": "b", "text": "Вторая фраза", "tag": None, "cue": None}]
    text, spans = sg._build_fragment_text_and_spans(units)
    for (s, e), u in zip(spans, units):
        assert text[s:e] == u["text"]


def test_build_fragment_text_and_spans_matches_text_with_cue_before_second_unit():
    units = [{"unit_id": "a", "text": "Первая фраза", "tag": "[pause]", "cue": None},
              {"unit_id": "b", "text": "Вторая фраза", "tag": "[pause]", "cue": "[slowly]"}]
    text, spans = sg._build_fragment_text_and_spans(units)
    assert "[slowly]" in text
    for (s, e), u in zip(spans, units):
        assert text[s:e] == u["text"], f"span для {u['unit_id']} указывает не на его текст"


def test_build_fragment_text_and_spans_matches_text_with_cue_before_first_unit():
    units = [{"unit_id": "a", "text": "Первая фраза", "tag": "[pause]", "cue": "[emphasis]"},
              {"unit_id": "b", "text": "Вторая фраза", "tag": None, "cue": None}]
    text, spans = sg._build_fragment_text_and_spans(units)
    for (s, e), u in zip(spans, units):
        assert text[s:e] == u["text"]


def test_build_fragment_text_and_spans_matches_text_with_energetic_and_cue():
    units = [{"unit_id": "a", "text": "Хук фраза", "tag": "[pause]", "cue": None},
              {"unit_id": "b", "text": "Вторая фраза", "tag": "[pause]", "cue": "[slowly]"}]
    text, spans = sg._build_fragment_text_and_spans(units, prefix_energetic=True)
    assert text.startswith("[energetic]")
    for (s, e), u in zip(spans, units):
        assert text[s:e] == u["text"]


def test_fragment_text_for_tts_delegates_to_shared_builder():
    units = [{"unit_id": "a", "text": "Текст", "tag": None, "cue": "[emphasis]"}]
    assert sg.fragment_text_for_tts(units) == sg._build_fragment_text_and_spans(units)[0]


def test_evaluate_fragment_asserts_on_text_span_mismatch():
    units = [{"unit_id": "a", "text": "Текст один", "tag": None, "cue": None,
              "target_range_sec": [0, 0], "target_wpm": 125, "words": 2}]
    with pytest.raises(AssertionError):
        sg.evaluate_fragment(units, "заведомо другой текст", "/nonexistent.mp3",
                             [("x", 0.0, 0.1)])


# ---------- sparse cue policy ----------

def test_assign_sparse_cues_climax_gets_slowly():
    units = [{"unit_id": "a", "chapter_id": 0, "is_climax": True, "stat": None}]
    sg.assign_sparse_cues(units)
    assert units[0]["cue"] == "[slowly]"


def test_assign_sparse_cues_stat_gets_emphasis():
    units = [{"unit_id": "a", "chapter_id": 0, "is_climax": False, "stat": "1.2 КГ"}]
    sg.assign_sparse_cues(units)
    assert units[0]["cue"] == "[emphasis]"


def test_assign_sparse_cues_neither_signal_gets_no_cue():
    units = [{"unit_id": "a", "chapter_id": 0, "is_climax": False, "stat": None}]
    sg.assign_sparse_cues(units)
    assert units[0]["cue"] is None


def test_assign_sparse_cues_climax_priority_over_stat_same_unit():
    units = [{"unit_id": "a", "chapter_id": 0, "is_climax": True, "stat": "1.2 КГ"}]
    sg.assign_sparse_cues(units)
    assert units[0]["cue"] == "[slowly]"


def test_assign_sparse_cues_slowly_budget_max_one_per_chapter():
    units = [
        {"unit_id": "a", "chapter_id": 0, "is_climax": True, "stat": None},
        {"unit_id": "b", "chapter_id": 0, "is_climax": True, "stat": None},
    ]
    sg.assign_sparse_cues(units)
    assert units[0]["cue"] == "[slowly]"
    assert units[1]["cue"] is None, "второй [slowly] в той же главе не должен пройти бюджет"


def test_assign_sparse_cues_slowly_budget_separate_per_chapter():
    units = [
        {"unit_id": "a", "chapter_id": 0, "is_climax": True, "stat": None},
        {"unit_id": "b", "chapter_id": 1, "is_climax": True, "stat": None},
    ]
    sg.assign_sparse_cues(units)
    assert units[0]["cue"] == "[slowly]"
    assert units[1]["cue"] == "[slowly]", "разные главы — разные бюджеты"


def test_assign_sparse_cues_emphasis_has_no_chapter_budget_limit():
    units = [
        {"unit_id": "a", "chapter_id": 0, "is_climax": False, "stat": "1"},
        {"unit_id": "b", "chapter_id": 0, "is_climax": False, "stat": "2"},
    ]
    sg.assign_sparse_cues(units)
    assert units[0]["cue"] == "[emphasis]"
    assert units[1]["cue"] == "[emphasis]"


def test_fragment_text_for_tts_never_includes_more_than_one_cue():
    units = [
        {"unit_id": "a", "text": "Раз", "tag": "[pause]", "cue": "[slowly]"},
        {"unit_id": "b", "text": "Два", "tag": "[pause]", "cue": "[emphasis]"},
    ]
    text = sg.fragment_text_for_tts(units)
    total_cues = text.count("[slowly]") + text.count("[emphasis]")
    assert total_cues == 1, f"максимум 1 cue на фрагмент, получили: {text!r}"
    assert "[slowly]" in text and "[emphasis]" not in text, (
        "первый по порядку cue должен победить детерминированно")


def test_question_rise_and_closing_hold_never_get_cue_via_classify_and_assign(tmp_path):
    # question_rise/closing_hold сознательно НЕ входят в CUE_MAP (см.
    # докстринг) — is_climax/stat остаются единственными триггерами,
    # рядовой вопрос/финальная фраза без явного климакса/статы cue не
    # получают, даже если их rhetorical_kind заметный.
    script = tmp_path / "script.txt"
    script.write_text(
        "=== HOOK === Ты видел это сто раз?\n=== FINAL === Вот и всё.\n", encoding="utf-8")
    saved_argv = sys.argv
    sys.argv = ["pipeline_smart.py", str(tmp_path)]
    import pipeline_smart
    sys.argv = saved_argv
    blocks = pipeline_smart.parse_blocks(str(script))
    units = speech_planner.build_units(blocks)
    speech_planner.assign_chapter_arcs(units)
    sg.assign_sparse_cues(units)
    assert all(u["cue"] is None for u in units)


# ---------- voice_settings профили ----------

def test_voice_settings_defaults_to_conservative_when_disabled(monkeypatch):
    monkeypatch.setattr(sg, "VOICE_PROFILES_ENABLED", False)
    units = [{"energy_target": "high"}]
    name, settings = sg.voice_settings_for_fragment(units)
    assert name == "conservative"
    assert settings == sg.VOICE_SETTINGS_PROFILES["conservative"]


def test_voice_settings_maps_high_energy_to_energetic_when_enabled(monkeypatch):
    monkeypatch.setattr(sg, "VOICE_PROFILES_ENABLED", True)
    units = [{"energy_target": "high"}]
    name, _ = sg.voice_settings_for_fragment(units)
    assert name == "energetic"


def test_voice_settings_maps_low_energy_to_calm_when_enabled(monkeypatch):
    monkeypatch.setattr(sg, "VOICE_PROFILES_ENABLED", True)
    units = [{"energy_target": "low"}]
    name, _ = sg.voice_settings_for_fragment(units)
    assert name == "calm"


def test_voice_settings_maps_med_energy_to_conservative_when_enabled(monkeypatch):
    monkeypatch.setattr(sg, "VOICE_PROFILES_ENABLED", True)
    units = [{"energy_target": "med"}]
    name, _ = sg.voice_settings_for_fragment(units)
    assert name == "conservative"


def test_voice_settings_only_three_profiles_exist():
    assert set(sg.VOICE_SETTINGS_PROFILES) == {"conservative", "calm", "energetic"}


def test_cache_key_differs_by_voice_profile():
    k1 = sg.cache_key("текст", "v", "m", 1, voice_profile="conservative")
    k2 = sg.cache_key("текст", "v", "m", 1, voice_profile="energetic")
    assert k1 != k2


# ---------- нормализация произношения ----------

def test_normalize_percent_sign():
    text, subs = sg.normalize_pronunciation("Настоящий вес — 5% от ожидаемого.")
    assert "5 процентов" in text
    assert "%" not in text
    assert len(subs) == 1


def test_normalize_kg_after_digit():
    text, subs = sg.normalize_pronunciation("Средний вес составлял 1.2 кг.")
    assert "килограммов" in text
    assert subs[0]["before"] == "2 кг"   # цифра ПЕРЕД сокращением — часть паттерна


def test_normalize_does_not_touch_ambiguous_bare_g():
    # "г" (граммы vs год vs город) НАМЕРЕННО не в таблице (см. докстринг) —
    # "5 г." должно остаться как есть, не превратиться в "5 граммов.".
    text, subs = sg.normalize_pronunciation("Событие произошло в 1932 г.")
    assert text == "Событие произошло в 1932 г."
    assert subs == []


def test_normalize_km_mm_cm():
    text, _ = sg.normalize_pronunciation("3 км, 4 см, 5 мм.")
    assert "километров" in text and "сантиметров" in text and "миллиметров" in text


def test_normalize_disabled_via_env():
    env = dict(os.environ, SPEECH_GEN_PRONUNCIATION_NORMALIZE="0")
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['speech_generate.py','.']; sys.path.insert(0, %r); "
         "import speech_generate as sg; "
         "t, s = sg.normalize_pronunciation('5%% готово'); print(t); print(len(s))" % SCRIPTS_DIR],
        capture_output=True, text=True, env=env, timeout=30)
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "5% готово"
    assert lines[1] == "0"


def test_apply_pronunciation_normalization_recomputes_words():
    units = [{"unit_id": "a", "text": "Вес 5% ровно", "words": 3}]
    sg.apply_pronunciation_normalization(units)
    assert units[0]["text"] == "Вес 5 процентов ровно"
    assert units[0]["words"] == len("Вес 5 процентов ровно".split())


def test_apply_pronunciation_normalization_reports_substitutions_with_unit_id():
    units = [{"unit_id": "u#0", "text": "5%", "words": 1}]
    subs = sg.apply_pronunciation_normalization(units)
    assert subs == [{"unit_id": "u#0", "before": "5%", "after": "5 процентов"}]


def test_apply_pronunciation_normalization_no_subs_returns_empty_list():
    units = [{"unit_id": "a", "text": "Обычная фраза без чисел", "words": 4}]
    subs = sg.apply_pronunciation_normalization(units)
    assert subs == []
    assert units[0]["text"] == "Обычная фраза без чисел"


# ---------- F0 shadow-метрика и реальная тишина на стыке ----------

def test_estimate_f0_hz_detects_known_frequency_within_tolerance():
    sr = 22050
    freq = 220.0
    t = np.arange(int(sr * 0.3)) / sr
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
    f0 = sg.estimate_f0_hz(tone, sr)
    assert f0 is not None
    # Автокорреляция может поймать субгармонику (см. докстринг shadow-
    # метрики) — проверяем ОКТАВНУЮ близость (f0 или f0/2 или f0*2 близко
    # к freq), не точное совпадение.
    ratios = [f0 / freq, (f0 * 2) / freq, f0 / (freq * 2)]
    assert any(abs(r - 1.0) < 0.05 for r in ratios), f"f0={f0} далеко от {freq} и его октав"


def test_estimate_f0_hz_returns_none_on_too_short_input():
    assert sg.estimate_f0_hz(np.zeros(3, dtype=np.float32), 22050) is None


def test_estimate_f0_hz_returns_none_on_none_input():
    assert sg.estimate_f0_hz(None, 22050) is None


def test_trailing_silence_detects_real_silence():
    sr = 22050
    tone = np.sin(2 * np.pi * 220 * np.arange(int(sr * 0.2)) / sr).astype(np.float32) * 0.5
    silence = np.zeros(int(sr * 0.3), dtype=np.float32)
    arr = np.concatenate([tone, silence])
    sec = sg._trailing_silence_sec(arr, sr)
    assert sec > 0.2, f"должен поймать большую часть {0.3}с реальной тишины, получил {sec}"


def test_trailing_silence_zero_when_no_silence():
    sr = 22050
    tone = np.sin(2 * np.pi * 220 * np.arange(int(sr * 0.3)) / sr).astype(np.float32) * 0.5
    assert sg._trailing_silence_sec(tone, sr) == 0.0


def test_evaluate_joins_includes_shadow_f0_and_join_risk_fields(tmp_path):
    sr = 22050
    for name in ("a.mp3", "b.mp3"):
        tone = (np.sin(2 * np.pi * 220 * np.arange(int(sr * 0.3)) / sr) * 0.5).astype(np.float32)
        raw = (tone * 32767).astype(np.int16).tobytes()
        path = tmp_path / name
        subprocess.run(["ffmpeg", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-",
                        "-c:a", "libmp3lame", str(path)], input=raw, capture_output=True, check=True)
    joins = sg.evaluate_joins([str(tmp_path / "a.mp3"), str(tmp_path / "b.mp3")])
    assert len(joins) == 1
    j = joins[0]
    assert "f0_a_hz_shadow" in j and "f0_b_hz_shadow" in j and "f0_jump_hz_shadow" in j
    assert "trailing_silence_sec" in j and "leading_silence_sec" in j
    assert j["join_risk"] in ("low", "high")


# ---------- SPEECH_JOIN_REGEN_BUDGET (зарезервировано, не реализовано) ----------

def test_join_regen_budget_defaults_to_zero():
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['speech_generate.py','.']; sys.path.insert(0, %r); "
         "import speech_generate as sg; print(sg.SPEECH_JOIN_REGEN_BUDGET)" % SCRIPTS_DIR],
        capture_output=True, text=True, timeout=30,
        env={k: v for k, v in os.environ.items() if k != "SPEECH_JOIN_REGEN_BUDGET"})
    assert r.stdout.strip() == "0"


def test_join_regen_budget_reads_env():
    env = dict(os.environ, SPEECH_JOIN_REGEN_BUDGET="3")
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['speech_generate.py','.']; sys.path.insert(0, %r); "
         "import speech_generate as sg; print(sg.SPEECH_JOIN_REGEN_BUDGET)" % SCRIPTS_DIR],
        capture_output=True, text=True, env=env, timeout=30)
    assert r.stdout.strip() == "3"
