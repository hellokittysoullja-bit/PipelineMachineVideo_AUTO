"""Юнит-тесты scripts/render_with_retry.py::parse_args — без реальных
subprocess-вызовов pipeline_smart.py (это только разбор CLI-аргументов)."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import pytest                       # noqa: E402
import render_with_retry as rwr     # noqa: E402


def test_parse_args_max_with_space_form_matches_documented_usage():
    # Регрессия: докстринг модуля и usage-сообщение документируют `--max N`
    # (через пробел) — раньше парсер понимал ТОЛЬКО `--max=N`, и вызов ровно
    # по документации тихо игнорировал флаг.
    video_dir, max_attempts = rwr.parse_args(["videos/01_test", "--max", "3"])
    assert video_dir == "videos/01_test"
    assert max_attempts == 3


def test_parse_args_max_with_equals_form_still_works():
    video_dir, max_attempts = rwr.parse_args(["videos/01_test", "--max=7"])
    assert video_dir == "videos/01_test"
    assert max_attempts == 7


def test_parse_args_no_max_uses_default():
    video_dir, max_attempts = rwr.parse_args(["videos/01_test"])
    assert video_dir == "videos/01_test"
    assert max_attempts == rwr.MAX_ATTEMPTS_DEFAULT


def test_parse_args_max_before_video_dir():
    video_dir, max_attempts = rwr.parse_args(["--max", "2", "videos/01_test"])
    assert video_dir == "videos/01_test"
    assert max_attempts == 2


def test_parse_args_missing_video_dir_returns_none():
    video_dir, max_attempts = rwr.parse_args(["--max", "2"])
    assert video_dir is None


def test_parse_args_max_without_value_raises():
    with pytest.raises(ValueError):
        rwr.parse_args(["videos/01_test", "--max"])
