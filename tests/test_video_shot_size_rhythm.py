"""Ритм крупностей плана для ВИДЕО-кандидатов (VIDEO_DIRECTOR_SCORE_VERSION=2).

Фото уже годами избегает повтора той же крупности плана (wide/medium/close/
detail) два раза подряд — pexels_photo() принимает recent_sizes и штрафует
кандидата, чья estimate_shot_size() совпадает с одной из двух последних
(см. size_ok в pexels_photo(), приоритет сразу после дедупа в _score_and_pick).
Видео в этом сигнале не участвовало вообще: recent_sizes не принимался,
крупность кандидата не считалась, а победивший видео-клип не пополнял
историю ни для следующего видео, ни для следующего фото — то есть подряд
идущие клипы одной крупности (два общих плана, два макро) ничем не
отличались от разнообразной последовательности, хотя это ровно то, что
пользователь называет отсутствием "режиссуры/ритма".

Тесты здесь проверяют функциональное поведение видео-отбора (VideoAdapter в
общем ядре, запрос слота с recent_sizes) напрямую (не через main() — она
слишком велика для юнит-теста) и то, что
main() реально прокидывает recent_shot_sizes в оба вызова и обратно
получает от видео-победителя обновление истории (source-level).
"""
import os
import sys
import tempfile


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402
from _media_calls import pick_video  # noqa: E402
from _video_world import QUERY, infra, video  # noqa: E402,F401


def _size_by_candidate(sizes):
    """estimate_shot_size по кадру превью: {id кандидата: крупность}."""
    def est(path):
        name = os.path.basename(path)
        return next((sz for vid, sz in sizes.items() if f"prev_{vid}_" in name), "medium")
    return est


def _two(infra, rel=(0.30, 0.30)):
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["rel"] = {1: rel[0], 2: rel[1]}


class TestShotSizeRhythm:
    def test_repeating_recent_size_loses_to_a_different_one(self, infra, monkeypatch):
        """Кандидат 1 — "wide" (как последние 2 клипа истории), кандидат 2 —
        "detail". При равном прочем побеждает другая крупность — тот же
        size_ok, что у фото, в том же ранжировании."""
        _two(infra)
        monkeypatch.setattr(ps, "estimate_shot_size", _size_by_candidate({1: "wide", 2: "detail"}))
        out = pick_video(ps, QUERY, 9, used_ids=set(), used_hashes=[],
                         recent_sizes=["wide", "wide"])
        assert out is not None and infra["downloads"] == [2]

    def test_all_candidates_repeating_still_fills_the_slot(self, infra, monkeypatch):
        """Слот не пустеет, если ВСЕ кандидаты повторяют крупность."""
        _two(infra)
        monkeypatch.setattr(ps, "estimate_shot_size", lambda p: "wide")
        out = pick_video(ps, QUERY, 9, used_ids=set(), used_hashes=[],
                         recent_sizes=["wide", "wide"])
        assert out is not None

    def test_recent_sizes_none_is_a_pure_no_op(self, infra, monkeypatch):
        """Без истории крупность не оценивается вовсе."""
        _two(infra)
        called = []
        monkeypatch.setattr(ps, "estimate_shot_size", lambda p: called.append(p) or "wide")
        out = pick_video(ps, QUERY, 9, used_ids=set(), used_hashes=[])
        assert out is not None and called == []

    def test_shot_size_ok_outranks_the_director_score(self, infra, monkeypatch):
        """Порядок осей тот же, что был у видео и есть у фото: ритм
        крупностей стоит ВЫШЕ бонуса Режиссёра. Что он стоит и выше тонкой
        разницы релевантности — известный дефект (терм ритма станет
        ограниченным, этап объектива); этот тест держит, что перенос видео в
        ядро порядок НЕ поменял."""
        _two(infra)
        monkeypatch.setattr(ps, "estimate_shot_size", _size_by_candidate({1: "wide", 2: "detail"}))

        def score(path, candidate_query=None, aesthetic_val=None):
            return 0.9 if "prev_1_" in os.path.basename(path) else 0.1
        pick_video(ps, QUERY, 9, used_ids=set(), used_hashes=[], sentence_score_fn=score,
                   director_assist=True, recent_sizes=["wide", "wide"])
        assert infra["downloads"] == [2]


class TestWiring:
    def _main_source(self):
        return open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()

    def test_main_passes_recent_sizes_to_both_video_call_sites(self):
        src = self._main_source()
        # Запрос слота строится один раз и уходит ВСЕМ попыткам слота — и
        # фото, и видео (tests/test_selection_engine.py держит, что попытки
        # получают одну и ту же переменную запроса). Здесь — что ритм в него
        # входит и что видео-путь его читает.
        assert src.count("recent_sizes=recent_shot_sizes") == 1
        import inspect
        vsrc = inspect.getsource(ps.VideoAdapter.choose)
        assert "request.recent_sizes" in vsrc

    def test_video_winner_updates_recent_shot_sizes(self):
        src = self._main_source()
        start = src.index("elif video:\n")
        end = src.index("\n        elif director_entry is None", start)
        block = src[start:end]
        assert 'shown_att.effect("shot_size", recent_shot_sizes, estimate_shot_size(probe))' in block

    def test_version_bumped_for_this_change(self):
        assert ps.VIDEO_DIRECTOR_SCORE_VERSION >= 2, (
            "VIDEO_DIRECTOR_SCORE_VERSION не поднят — смена ранжирования видео "
            "не инвалидирует прогретый temp_smart/pexels_video_cache/")
