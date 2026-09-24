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
    """Source-level: покадровое вето реально стоит в отборе видео и
    участвует в подписи отбора."""

    def test_called_from_the_video_adapter(self):
        import inspect
        src = inspect.getsource(ps.VideoAdapter.choose)
        assert "video_frames_violate(" in src and "is_relevant_candidate(mid, query" in src
        assert "negative_anchor_violation(f, query)" in inspect.getsource(ps.video_frames_violate)

    def test_part_of_candidate_gate_signature(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        start = src.index("def candidate_gate_signature")
        block = src[start:src.index("_CANDIDATE_GATE_SIG = \"gate:\"", start)]
        names = _selection_code_names()
        assert "video_frames_violate" in names and "video_preview_urls" in names

    def test_reuses_the_same_sample_points_as_domain_guard(self):
        """Не изобретает новый набор точек — превью берутся в уже
        откалиброванных VIDEO_DOMAIN_GUARD_SAMPLE_FRACS."""
        assert ps.VIDEO_PREVIEW_FRACS == ps.VIDEO_DOMAIN_GUARD_SAMPLE_FRACS


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
    def test_other_frames_of_the_same_video_no_longer_show_the_crowd_on_the_new_model(self, name):
        """ЧЕСТНЫЙ РЕГРЕСС 18.09 (смена get_clip_model() на SigLIP2-base256,
        см. CLIP_GATE_MODEL_NAME/NEGATIVE_VETO_MARGIN в pipeline_smart.py),
        не скрытый провал.

        На CLIP margin этих кадров был отрицательный (вето срабатывало);
        на новой модели margin положительный на ВСЕХ трёх кадрах этого
        видео (+0.020/+0.034/+0.011) — на калиброванном под ноль ложных
        отказов пороге (-0.06) контрастивное вето эту конкретную сцену
        (толпа современных зрителей на историческом по форме кадре) больше
        не ловит вообще ни на одном сэмпле. Тот же класс регресса, что уже
        задокументирован в test_negative_veto.py для 084.jpg/000.jpg.
        SMART_RELEVANCE_VETO (проверка ПОБЕДИТЕЛЯ более тяжёлым
        so400m+Jina ensemble) остаётся вторым, независимым слоем защиты —
        контрастивное вето не единственная линия."""
        img = os.path.join(FIXTURES, name)
        vetoed, who = ps.negative_anchor_violation(img, QUERY)
        assert not vetoed, (
            f"{name}: margin теперь ловится веткой — если порог перекалибровали "
            f"так, что это стало ловиться, обнови тест на положительное "
            f"утверждение поимки, гэп не потерян молча")

    def test_multi_frame_wiring_still_propagates_a_violation_from_any_sample(self, monkeypatch):
        """Архитектурная проверка (не про ЭТО видео): если ХОТЬ ОДИН кадр
        показывает нарушение, video_frames_violate() обязана его не потерять
        — независимо от того, ловит ли вето сцену crowd_*.jpg на текущей
        модели. Один кадр объявляется нарушением напрямую, остальные идут
        через настоящую модель."""
        frames = [os.path.join(FIXTURES, n) for n in ("crowd_start.jpg", "clean_mid.jpg", "crowd_end.jpg")]
        real = ps.negative_anchor_violation

        def fake(path, query):
            if path.endswith("clean_mid.jpg"):
                return True, "synthetic_violation_for_wiring_test"
            return real(path, query)
        monkeypatch.setattr(ps, "negative_anchor_violation", fake)
        assert ps.video_frames_violate(frames, QUERY) is True


def _selection_code_names():
    """Имена функций, чей код входит в подпись отбора (code_signature)."""
    import code_signature
    import selection_engine
    import pipeline_smart as _ps
    code = code_signature.reachable([_ps.PhotoAdapter, _ps.VideoAdapter, selection_engine.select],
                                    _ps.SELECTION_CODE_MODULES, stop=_ps._judge_code_entries())
    return {k.split(".", 1)[1] for k in code}
