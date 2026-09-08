"""Контрастивное вето по ловушкам на НЕСКОЛЬКИХ кадрах видео, не на одном.

Реальный, живьём найденный случай (08.09, во время верификационного рендера
videos/_test20s после сегодняшних видео-правок). Хук-слот "Готов спорить,
что да. Герой на экране заносит клинок двумя руками, рычит, враг падает —
вместе с конём, разумеется." по запросу "warrior on horseback with sword"
выиграло реальное Pexels-видео id 855260 (48 секунд). В готовом контактном
листе на этом месте — крупный план рыцаря в латах с занесённым мечом на
фоне ТОЛПЫ СОВРЕМЕННЫХ ЗРИТЕЛЕЙ (флаг, люди в обычной одежде) — ровно тот
класс брака, который CONTENT_NEGATIVE_ANCHORS должен ловить
("crowd of modern spectators in casual clothes watching an event").

Причина, по которой вето не сработало на этапе отбора: is_relevant_
candidate() проверяет РОВНО ОДИН кадр-пробник (extract_video_probe_frame(),
по умолчанию t=0.5с). Прямое измерение на реальном скачанном файле показало:
толпа видна на t=0.96с и t=40.87с, но НЕ видна на t≈24с (та же часть
ролика, что и дефолтный пробник) — единственный проверенный момент оказался
ровно тем, где брака не видно.

video_domain_guard_violation() уже решает ЭТУ ЖЕ архитектурную проблему для
анахронизма формы клинка (см. её докстринг про рукоять без гарды/клинка) —
video_negative_anchor_violation() переносит тот же приём (несколько точек
по длительности, нарушение НА ЛЮБОЙ — нарушение кандидата целиком) на
контрастивное вето.

Фикстуры — РЕАЛЬНЫЕ кадры этого самого видео (id 855260), не синтетика:
синтетический "человек с мечом на фоне толпы" не воспроизвёл бы то, что
реально спрятало брак от одного пробника — сам факт, что видимость толпы
меняется по ходу ролика.
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402

FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures", "video_multiframe")
MANIFEST = os.path.join(FIXTURES, "manifest.json")
QUERY = "warrior on horseback with sword"

HAS_TORCH = True
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except Exception:
    HAS_TORCH = False


def _manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


class TestWiring:
    """Source-level: video_negative_anchor_violation() реально вызывается
    из pexels_video() и участвует в подписи отбора."""

    def test_called_from_pexels_video(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        assert src.count("def pexels_video(") == 1
        block = src[src.index("def pexels_video("):]
        assert "video_negative_anchor_violation(trial, query)" in block

    def test_part_of_candidate_gate_signature(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        start = src.index("def candidate_gate_signature")
        block = src[start:src.index("_CANDIDATE_GATE_SIG = \"gate:\"", start)]
        assert "video_negative_anchor_violation" in block

    def test_reuses_the_same_sample_points_as_domain_guard(self):
        """Не изобретает новый набор точек сэмплирования — переиспользует
        уже откалиброванный VIDEO_DOMAIN_GUARD_SAMPLE_FRACS."""
        import inspect
        src = inspect.getsource(ps.video_negative_anchor_violation)
        assert "VIDEO_DOMAIN_GUARD_SAMPLE_FRACS" in src


@pytest.mark.skipif(not HAS_TORCH, reason="нужны torch/transformers")
@pytest.mark.slow
class TestOnRealVideoFrames:
    """Живая модель на реальных кадрах реального Pexels-видео 855260."""

    def test_single_default_frame_misses_the_crowd(self):
        """Воспроизводит САМ баг: is_relevant_candidate() смотрит только
        сюда и не видит ничего криминального."""
        img = os.path.join(FIXTURES, "clean_mid.jpg")
        vetoed, who = ps.negative_anchor_violation(img, QUERY)
        assert not vetoed, (
            "если этот кадр теперь ловится — тест мог перестать "
            "воспроизводить реальный слепой момент, проверить фикстуру")

    @pytest.mark.parametrize("name", ["crowd_start.jpg", "crowd_end.jpg"])
    def test_other_frames_of_the_same_video_do_show_the_crowd(self, name):
        img = os.path.join(FIXTURES, name)
        vetoed, who = ps.negative_anchor_violation(img, QUERY)
        assert vetoed, f"{name}: толпа больше не ловится на этом кадре"
        assert who

    def test_multi_frame_check_catches_what_the_single_frame_misses(self, monkeypatch, tmp_path):
        """Главная проверка: video_negative_anchor_violation() на ПОЛНОМ
        наборе сэмплов ловит то же видео, которое single-frame-проверка
        пропустила бы (см. test_single_default_frame_misses_the_crowd)."""
        manifest = _manifest()
        fake_video = str(tmp_path / "fake.mp4")
        open(fake_video, "wb").close()
        monkeypatch.setattr(ps, "get_media_duration", lambda p: 48.08)

        def fake_extract(path, base_at=0.5, retry_ats=()):
            # Сопоставляем точку сэмплирования с ближайшей реальной фикстурой
            # по времени, зафиксированному в манифесте — те же доли
            # длительности (0.15/0.5/0.85), что video_negative_anchor_
            # violation() реально запрашивает для 48.08-секундного видео.
            best = min(manifest["images"].items(),
                       key=lambda kv: abs(kv[1]["at_sec"] - base_at))
            return os.path.join(FIXTURES, best[0]), False

        monkeypatch.setattr(ps, "extract_video_probe_frame", fake_extract)
        vetoed, who = ps.video_negative_anchor_violation(fake_video, QUERY)
        assert vetoed, "многокадровая проверка не поймала то, что ловит хотя бы один сэмпл"
        assert who
