#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты не получают ключ шлюза и через .env дочернего процесса."""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_child_process_does_not_get_the_gateway_key_back_from_dotenv(tmp_path):
    (tmp_path / ".env").write_text("LLM_GATEWAY_API_KEY=sk-real-looking-key\n", encoding="utf-8")
    code = ("from dotenv import load_dotenv; import os; load_dotenv(); "
            "print(repr(os.environ.get('LLM_GATEWAY_API_KEY')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         text=True, env=dict(os.environ)).stdout.strip()
    assert out == "''", f"ключ вернулся из .env: {out}"
