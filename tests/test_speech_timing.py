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
