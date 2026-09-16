# -*- coding: utf-8 -*-
"""Установка движка локального режиссёра.

Блокер найден 16.09 и он не гипотетический: у владельца Windows, а на
PyPI у `llama-cpp-python` лежит ТОЛЬКО sdist — ни одной готовой сборки.
То есть команда «pip install llama-cpp-python», которую скрипт печатал
до этого, на его машине упёрлась бы в требование компилятора C++, и
«одна команда без флагов» кончилась бы на первом шаге.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import setup_local_director as d  # noqa: E402


class TestEngineInstallCommand:
    def test_windows_gets_the_prebuilt_wheel_index(self):
        """На Windows обязателен официальный индекс готовых сборок:
        сборки из исходников там требуют Visual Studio Build Tools."""
        cmd = " ".join(d.engine_install_command("python", "Windows"))
        assert d.ENGINE_WHEEL_INDEX in cmd
        assert f"llama-cpp-python=={d.ENGINE_VERSION}" in cmd

    def test_linux_does_not_get_that_index(self):
        """Поймано ЖИВОЙ установкой, а не вычитано: колесо из того индекса
        собрано под musl, и на обычном дистрибутиве импорт падает с
        `libc.musl-x86_64.so.1: cannot open shared object file`. Одна
        команда на все системы была бы неверна ровно на одной из них."""
        cmd = " ".join(d.engine_install_command("python", "Linux"))
        assert d.ENGINE_WHEEL_INDEX not in cmd, (
            "индекс musl-сборок не должен предлагаться на Linux")
        assert "llama-cpp-python" in cmd

    def test_version_is_pinned_only_where_the_index_is_used(self):
        """Закрепление версии осмысленно только там, где берут готовую
        сборку: индекс отстаёт от PyPI (0.3.19 против 0.3.35), и тащить
        старую версию на систему, где сборка из исходников и так работает,
        значит без причины отставать."""
        assert d.ENGINE_VERSION not in " ".join(
            d.engine_install_command("python", "Linux"))

    @pytest.mark.parametrize("system", ["Windows", "Linux", "Darwin"])
    def test_command_is_a_single_pip_invocation(self, system):
        lines = d.engine_install_command("python", system)
        assert lines and all(isinstance(x, str) and x.strip() for x in lines)
        assert sum("pip install" in x for x in lines) == 1


class TestSmallModelIsTheDefault:
    """Решение владельца 16.09, подкреплённое замером времени: крупная
    впятеро медленнее (180 с на главу против 35) и занимает 11 ГБ ОЗУ, а
    её преимущество по качеству на текущем задании НЕ установлено."""

    @pytest.mark.parametrize("ram_gb", [4, 8, 16, 64, 128, None])
    def test_small_model_regardless_of_ram(self, ram_gb):
        assert d.pick(ram_gb) == "4b"

    def test_big_model_still_reachable_explicitly(self):
        """Не удалена — тот, кто захочет перемерить её сам, должен мочь."""
        assert d.pick(64, forced="30b") == "30b"
        assert "30b" in d.MODELS

    def test_speed_cost_is_recorded_not_remembered(self):
        small, big = d.MODELS["4b"], d.MODELS["30b"]
        assert small["sec_per_chapter"] < big["sec_per_chapter"]
        assert big["sec_per_chapter"] / small["sec_per_chapter"] >= 3


class TestModelChoiceIsBackedByNumbers:
    def test_every_model_carries_its_measured_score(self):
        """Скрипт выбирает модель за пользователя — значит обязан уметь
        назвать, по какому числу. Скор без замера был бы рекламой."""
        for key, m in d.MODELS.items():
            assert isinstance(m.get("score"), int), key
            assert m["score"] > 0, key
            assert m.get("gb", 0) > 0 and m.get("need_ram_gb", 0) > 0, key

    def test_stale_score_is_never_used_to_recommend(self):
        """Скор, снятый СТАРЫМ заданием, нельзя сравнивать с новым — это
        та же ловушка, что уже поймана на квантовке (менялись две
        переменные разом). Модель с таким скором не имеет права стать
        выбором по умолчанию, пока её не перемерят."""
        for key, m in d.MODELS.items():
            if m.get("score_stale"):
                assert not m.get("default"), (
                    f"{key}: скор устарел, но модель предлагается по умолчанию")

    def test_exactly_one_default_model(self):
        defaults = [k for k, m in d.MODELS.items() if m.get("default")]
        assert defaults == ["4b"], defaults

    def test_ram_requirement_exceeds_the_file(self):
        """Веса грузятся в память целиком: требовать памяти меньше, чем
        весит файл, — обещание, которое не выполнится."""
        for key, m in d.MODELS.items():
            assert m["need_ram_gb"] > m["gb"], key
