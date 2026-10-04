"""Тайминг речи — состояние у экземпляра, а не у модуля.

В исходнике папка ролика читалась из sys.argv при импорте, кэши карты
пауз и итоги (концы речи, причина отказа) были глобальными: второй ролик
в том же процессе получал кэш первого."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import script_parser  # noqa: E402
from speech_timing import SpeechTiming  # noqa: E402

SCRIPT = "=== HOOK ===\nРаз два три. [pause] Четыре пять.\n"


def _episode(root, cuts):
    os.makedirs(os.path.join(root, "media_plan", "alignment"))
    with open(os.path.join(root, "script.txt"), "w", encoding="utf-8") as f:
        f.write(SCRIPT)
    text = SCRIPT.split("\n", 1)[1].strip()
    t, rows = 0.0, []
    for ch in text:
        rows.append({"char": ch, "start": round(t, 3), "end": round(t + 0.1, 3)})
        t += 0.1
    with open(os.path.join(root, "media_plan", "alignment", "00.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["char", "start", "end"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(root, "media_plan", "pause_cuts.json"), "w") as f:
        json.dump({"cuts": cuts}, f)
    return script_parser.parse_blocks(os.path.join(root, "script.txt"))


def test_two_episodes_in_one_process_do_not_share_the_pause_map(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    blocks_a = _episode(a, [])
    blocks_b = _episode(b, [[0.0, 0.5]])          # у второго вырезано полсекунды в начале
    on_a = SpeechTiming(a).onsets(blocks_a)
    on_b = SpeechTiming(b).onsets(blocks_b)
    assert on_a and on_b
    assert on_b[1] == on_a[1] - 0.5


def test_speech_ends_only_when_phrase_lock_holds(tmp_path):
    root = str(tmp_path / "e")
    blocks = _episode(root, [])
    st = SpeechTiming(root)
    assert st.onsets(blocks) and len(st.speech_ends) == len(blocks) and st.failure is None
    blocks[1]["text"] = "Совсем другой текст тут."
    assert st.onsets(blocks) is None
    assert st.speech_ends == [] and st.failure["reason"]


def test_import_reads_nothing_from_argv(tmp_path):
    """Импорт модулей звука и тайминга не смотрит в sys.argv и не трогает диск."""
    code = ("import sys; sys.argv=['x','/nonexistent/folder']; sys.path.insert(0, %r);"
            "import speech_timing, audio_master, subtitles, assemble_frames; print('ok')"
            % os.path.join(ROOT, "scripts"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert out.stdout.strip() == "ok", out.stderr


def test_word_times_follow_alignment_and_pause_cuts(tmp_path):
    root = str(tmp_path / "w")
    blocks = _episode(root, [[0.0, 0.5]])
    st = SpeechTiming(root)
    assert st.onsets(blocks)
    wt = st.word_times
    assert [w["word"] for w in wt[0]] == ["Раз", "два", "три."]
    assert [w["word"] for w in wt[1]] == ["Четыре", "пять."]
    # «два»: символы 4..6 исходного текста -> сырые 0.4..0.7, минус 0.5 вырезанного начала
    assert abs(wt[0][1]["start"] - 0.0) < 1e-6 and abs(wt[0][1]["end"] - 0.2) < 1e-6
    for blk, on in zip(wt, st.onsets(blocks)):
        assert abs(blk[0]["start"] - on) < 1e-6
        assert all(a["end"] <= b["start"] + 1e-9 for a, b in zip(blk, blk[1:]))


def test_word_times_survive_numbers_read_as_words():
    from speech_timing import _word_times
    text = "Это 5 минут."
    want = "Это5минут."
    spoken = "Этопятьминут."
    timed = [(c, i*0.1, i*0.1 + 0.1) for i, c in enumerate(spoken)]
    wt = _word_times(text, want, timed, lambda t: t)
    assert [w["word"] for w in wt] == ["Это", "5", "минут."]
    assert wt[0]["start"] == 0.0 and wt[2]["end"] == timed[-1][2]
    assert wt[0]["end"] <= wt[1]["start"] <= wt[2]["start"]
