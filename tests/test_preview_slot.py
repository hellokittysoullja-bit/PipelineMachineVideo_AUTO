# -*- coding: utf-8 -*-
"""preview_slot.py — превью одного слота без полной пересборки эпизода.

render_manifest.json несёт "path" ТОЛЬКО для status="ok" — у "absorbed"
(поглощён соседом, NEVER_SHOW_KNOWN_BAD), "failed" и
"skipped_selection_only" своего клипа нет вовсе. Прежде прямой доступ
entry["path"] падал голым KeyError вместо честного сообщения — найдено
тем же аудитом 21.09, что чинил контактный лист на этот же класс
расхождения (консьюмер написан до появления новых статусов).
"""
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "preview_slot.py")


def _episode(tmp_path, status, extra=None):
    mp = tmp_path / "media_plan"
    mp.mkdir()
    entry = {"index": 0, "status": status, "reason": "test", "section": "HOOK", "duration": 2.0}
    if extra:
        entry.update(extra)
    (mp / "render_manifest.json").write_text(json.dumps({"clips": [entry]}), encoding="utf-8")
    (mp / "shot_manifest.json").write_text(json.dumps(
        [{"index": 0, "text_preview": "тест"}]), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("status", ["absorbed", "failed", "skipped_selection_only"])
def test_no_path_status_fails_honestly_not_with_a_raw_keyerror(tmp_path, status):
    d = _episode(tmp_path, status)
    r = subprocess.run([sys.executable, SCRIPT, str(d), "0"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "KeyError" not in r.stderr, r.stderr
    assert "path" not in r.stdout.lower() or "без готового клипа" in r.stdout
    assert status in r.stdout or "без готового клипа" in r.stdout
