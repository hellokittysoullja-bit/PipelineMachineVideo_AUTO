"""Подобранный слот обязан указывать на СУЩЕСТВУЮЩИЙ файл.

Реальный дефект, найденный замером 14.09 (прогон разметки эпизода 02, 42
запроса): `pexels_photo()` возвращала путь к файлу, которого нет на диске.
Механизм — скачивание ПОЛНОРАЗМЕРНОГО файла победителя падало (Wikimedia
отвечает 429 на всплеск запросов, Институт искусств Чикаго — 403 без своего
заголовка), `except` внутри цикла повторного выбора превращал это в
`sharp_ok_full = False`, цикл выходил по исчерпании повторов — и функция
доходила до `return cf`.

Почему это не ловилось ничем: слот числился успешно подобранным (sidecar
писался, used_ids пополнялся, счётчик «выиграл слот» рос), и ни один отчёт
эпизода не задавал вопрос «а файл-то есть». Два слота из 42 (#18 «armoured
knight marching field», #23 «medieval helmet lying dirt») пришли именно
такими.

Это тот же класс, что давно закрыт на стороне рендера `verify_clip()`
(«нулевой код возврата ffmpeg не доказывает, что файл записан») — здесь он
переносится на скачивание.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", os.path.join(REPO_ROOT, "videos", "_none")]
pytest.importorskip("PIL")
import pipeline_smart as ps   # noqa: E402


class TestDownloadedOk:
    def test_missing_file_is_not_ok(self, tmp_path):
        assert ps._downloaded_ok(str(tmp_path / "нет-такого.jpg")) is False

    def test_zero_byte_file_is_not_ok(self, tmp_path):
        """Оборванная закачка оставляет файл нулевого размера — считать его
        готовым кадром значит показать зрителю пустоту."""
        p = tmp_path / "empty.jpg"
        p.write_bytes(b"")
        assert ps._downloaded_ok(str(p)) is False

    def test_real_file_is_ok(self, tmp_path):
        p = tmp_path / "ok.jpg"
        p.write_bytes(b"\xff\xd8\xff")
        assert ps._downloaded_ok(str(p)) is True

    def test_directory_instead_of_file_does_not_raise(self, tmp_path):
        assert ps._downloaded_ok(str(tmp_path)) in (True, False)


def test_photo_path_is_verified_before_being_returned():
    """Проверка обязана стоять МЕЖДУ циклом повторного выбора и возвратом
    пути — если её вынести или потерять, дефект возвращается молча, а
    симптом (выпавший блок или остановка сборки) проявится на часы позже и
    в другом месте."""
    src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    start = src.index("def _select_photo(")
    body = src[start:src.index("\ndef ", start + 10)]
    assert "_downloaded_ok(cf)" in body, (
        "pexels_photo() больше не проверяет, что файл победителя реально "
        "скачан — ровно этот пробел вернул два слота эпизода 02 путём в никуда")
    check = body.index("_downloaded_ok(cf)")
    ret = body.rindex("return cf")
    assert check < ret, "проверка должна стоять ДО возврата пути"


def test_unrecoverable_download_returns_none_not_a_path(monkeypatch, tmp_path):
    """Когда не скачался НИ ОДИН кандидат, честный ответ — «медиа нет»
    (слот уходит на лестницу фолбэков, которая ровно для этого и написана),
    а не путь к несуществующему файлу."""
    src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    start = src.index("def _select_photo(")
    body = src[start:src.index("\ndef ", start + 10)]
    block = body[body.index("_downloaded_ok(cf)"):]
    marker = block.index("слот остаётся без медиа")
    assert "return None" in block[marker:marker + 600], (
        "ветка «ни один кандидат не скачался» обязана возвращать None, "
        "а не проваливаться дальше к `return cf`")
