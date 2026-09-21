"""Машинный сток Шага 4 больше не обходит стек гейтов.

РЕАЛЬНЫЙ пробел, найденный прямой проверкой цепочки. В пайплайне два
независимых механизма подбора, и приоритет у того, у которого нет ни одной
проверки:

    Шаг 4  stock_fetch_multisource.py — round-robin по чётности слота,
           запрос из словаря themes.json, и НИ ОДНОГО упоминания
           relevance/CLIP/домен-гварда/негативного вето во всём файле;
    Шаг 7  pipeline_smart.py — смысловое назначение запроса + полный стек.

`use_local` взводится от наличия media/{index+1:03d}_*, а
`photo = local_photo(i) if use_local else None` проверяется РАНЬШЕ Pexels.
То есть выполнение документированного Шага 4 отключало гейты для каждого
закрытого им слота.

САМОЕ ВАЖНОЕ в этих тестах — не то, что гейт работает, а то, что он НЕ
ТРОГАЕТ курируемые человеком картинки: по ЧАСТИ 14 AI-картинки
оцениваются, но не заменяются никогда.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402


@pytest.mark.parametrize("name", [
    "007_stock.jpg", "001_stock.jpeg", "042_stock.png",
    "/abs/path/media/013_stock.jpg",
])
def test_machine_stock_is_recognised(name):
    assert ps.local_file_is_machine_stock(name)


@pytest.mark.parametrize("name", [
    "001_flow.jpg", "002_grok.jpg", "003_fastgen.jpg", "004_ai.jpg",
])
def test_curated_ai_images_are_never_gated(name):
    """ЧАСТЬ 14: AI-картинки курирует человек, они не заменяются никогда."""
    assert not ps.local_file_is_machine_stock(name)


@pytest.mark.parametrize("name", [
    "my_photo.jpg", "001.jpg", "knight_closeup.png", "hook_frame_2.jpg",
])
def test_unknown_names_are_treated_as_curated(name):
    """Fail-open в сторону человека: выбросить кадр, который кто-то выбрал
    сам, хуже, чем пропустить один машинный."""
    assert not ps.local_file_is_machine_stock(name)


def test_curated_suffix_wins_over_stock_marker():
    """Пограничный случай: если в имени есть и то и другое, побеждает
    курируемость — ошибка в эту сторону дешевле."""
    assert not ps.local_file_is_machine_stock("005_stock_flow.jpg")


def test_gate_is_wired_into_the_selection_loop():
    """Тот самый класс пробела, ради которого модуль и написан: функция
    может существовать и не вызываться ни разу."""
    import inspect
    body = inspect.getsource(ps.main)
    assert "local_file_is_machine_stock(photo)" in body
    assert "LOCAL_STOCK_GATE" in body


def test_gate_falls_through_to_normal_selection_not_to_an_empty_slot():
    """Отклонённый локальный файл обнуляет photo, чтобы слот пошёл обычным
    путём отбора — он не должен остаться пустым."""
    import inspect
    body = inspect.getsource(ps.main)
    idx = body.index("local_file_is_machine_stock(photo)")
    tail = body[idx:idx + 900]
    assert "photo = None" in tail


def test_flag_is_registered_with_default_on():
    import feature_flags as ff
    assert ff.FLAGS["LOCAL_STOCK_GATE"].default == "1"
