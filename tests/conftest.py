"""Изоляция тестов (перенесено из PipelineMachineVideo_AUTO/tests/conftest.py).

Тест не должен зависеть от .env рабочей копии: ключи шлюза и платные слои
гасятся, рабочая папка render_core — временная, общая на сессию."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


@pytest.fixture(autouse=True)
def _isolate_from_real_dotenv(monkeypatch):
    # Пустая строка, а не удаление: load_dotenv() в дочернем процессе не
    # перезаписывает существующую пустую переменную (живой случай 24.09 в
    # исходном репозитории — тест платно ходил в шлюз).
    for k in ("LLM_GATEWAY_API_KEY", "GEMINI_API_KEY", "ELEVENLABS_API_KEY", "LUMEAN_API_KEY"):
        monkeypatch.setenv(k, "")
    for k in ("SHOT_JUDGE", "IMAGE_GENERATION", "CAPTION_SCREEN", "RESEARCH_ROUND"):
        monkeypatch.setenv(k, "0")


@pytest.fixture(scope="session")
def _suite_working_root(tmp_path_factory):
    return tmp_path_factory.mktemp("suite_work")


@pytest.fixture(autouse=True)
def _private_working_folders(monkeypatch, _suite_working_root):
    rc = sys.modules.get("render_core")
    if rc is not None:
        root = str(_suite_working_root)
        monkeypatch.setattr(rc, "VIDEO_FOLDER", root, raising=False)
        monkeypatch.setattr(rc, "TEMP_FOLDER", os.path.join(root, "temp_render"), raising=False)
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: живые ML-прогоны, десятки секунд и дольше")
