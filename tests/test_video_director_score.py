"""Видео-путь ранжирует кандидатов ТЕМ ЖЕ visual_director.compute_extra_score(),

что и фото, а не голым sentence_relevance() без единого её бонуса.

Реальный, найденный вживую пробел (08.09): compute_extra_score() уже
добавляет arc_stage-осознанную крупность плана (ARC_STAGE_SHOT_SIZE_BONUS),
совпадение домена (domain_match_bonus), анти-повтор (repetition_penalty) и
QC-бонус (visual_qc_bonus) поверх голой sentence_relevance() — но main()
раньше строил video_sentence_fn отдельным partial(visual_director.
sentence_relevance, ...), и видео (примерно половина слотов эпизода) не
получало НИ ОДНОГО из этих сигналов, хотя фото получало их все через
director_score_fn.

Тестируется здесь НЕ сам compute_extra_score() (это делает
test_visual_director.py) — а то, что main() реально передаёт его в видео-
путь (source-level, main() слишком велика для end-to-end юнит-теста) и что
pexels_video() остаётся обратно совместимой: её контракт "sentence_score_fn
вызывается одним позиционным аргументом" (см. tests/test_parse.py — там
лямбды принимают ровно один параметр) не должен сломаться оттого, что
теперь туда передают функцию, которая ТАКЖЕ умеет принимать
candidate_query/aesthetic_val как опциональные keyword-параметры.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402


def _main_video_selection_block():
    src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    start = src.index("if not photo and not video and use_pexels:")
    end = src.index("\n        # ЛЕСТНИЦА ФОЛБЭКОВ", start)
    return src[start:end]


class TestWiring:
    """Source-level: main() реально строит video_sentence_fn из
    director_score_fn, не из голого sentence_relevance()."""

    def test_video_sentence_fn_is_the_director_score_fn(self):
        block = _main_video_selection_block()
        assert "video_sentence_fn = director_score_fn" in block, (
            "video_sentence_fn больше не совпадает с director_score_fn — "
            "видео снова осталось бы без arc_stage/domain/repetition/qc-бонусов")

    def test_video_no_longer_uses_bare_sentence_relevance_alone(self):
        """Раньше строка была ровно
        'functools.partial(visual_director.sentence_relevance, block_text=sem_text)'
        привязанной к video_sentence_fn — эта форма не должна вернуться."""
        block = _main_video_selection_block()
        assert "visual_director.sentence_relevance, block_text=sem_text)" not in block

    def test_video_winner_updates_recent_semantic_tags(self):
        """Без этого repetition_penalty был бы слеп к повтору video->video:
        compute_extra_score штрафует по recent_semantic_tags, но если видео-
        победители сами никогда туда не попадают, две видео-вставки одного
        домена/роли подряд не считаются повтором вообще."""
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        start = src.index("elif video:\n")
        end = src.index("\n        elif director_entry is None", start)
        block = src[start:end]
        assert "recent_semantic_tags.append((candidate_domain, director_role))" in block

    def test_version_marker_is_part_of_the_selection_signature(self):
        """Без версии в подписи включение этой фичи не инвалидирует
        уже закэшированного видео-победителя, отобранного до неё —
        тот же класс бага, что уже закрыт для OPENVERSE_ENABLED/FALLBACK_CARD."""
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        start = src.index("def _selection_stack_signature")
        block = src[start:src.index("\ndef candidate_gate_signature", start)]
        assert "VIDEO_DIRECTOR_SCORE_VERSION" in block


class TestBackwardCompatibleCallingConvention:
    """pexels_video() зовёт sentence_score_fn(probe) ОДНИМ позиционным
    аргументом (см. test_parse.py) — director_score_fn обязан оставаться
    вызываемым так же, просто без своих собственных candidate_query/
    aesthetic_val бонусов (те уже добавляются вручную в pexels_video(), см.
    SAME_QUERY_BONUS/OPENING_AESTHETIC_WEIGHT инлайн там)."""

    def test_director_style_fn_is_callable_positionally(self):
        import functools

        def fake_compute_extra_score(image_path, role, block_text, text_domain,
                                     recent_semantic_tags, arc_stage=None, own_query=None,
                                     candidate_query=None, is_opening=False, aesthetic_val=None):
            base = 0.5
            if candidate_query and own_query and candidate_query == own_query:
                base += 10.0   # не должен сработать при вызове без kwargs
            return base

        fn = functools.partial(fake_compute_extra_score, role="narrative",
                               block_text="текст", text_domain=None,
                               recent_semantic_tags=[], own_query="q0")
        # Ровно так, как это делает pexels_video(): один позиционный аргумент.
        assert fn("probe.jpg") == 0.5

    def test_pexels_video_ranks_by_director_style_score(self, tmp_path, monkeypatch):
        """Функциональная проверка: два кандидата различаются ТОЛЬКО
        director-стиля бонусом (переданным через sentence_score_fn) —
        побеждает тот, у кого бонус выше, ровно как раньше побеждал тот, у
        кого выше был голый sentence_relevance()."""
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
        monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
        monkeypatch.setattr(ps, "_pexels_search_videos", lambda q: [
            {"id": 1, "url": "https://www.pexels.com/video/knight-armor-1/",
             "video_files": [{"file_type": "video/mp4", "width": 1920, "link": "http://x/1.mp4"}]},
            {"id": 2, "url": "https://www.pexels.com/video/knight-armor-2/",
             "video_files": [{"file_type": "video/mp4", "width": 1920, "link": "http://x/2.mp4"}]},
        ])

        def fake_download(req, dest, timeout=None):
            with open(dest, "wb") as f:
                f.write(b"1" if "1.mp4" in req.full_url else b"2")

        monkeypatch.setattr(ps, "atomic_url_download", fake_download)
        monkeypatch.setattr(ps, "extract_video_probe_frame",
                            lambda p, **kw: (p + ".jpg", False))
        monkeypatch.setattr(ps, "is_relevant_candidate", lambda *a, **k: True)
        monkeypatch.setattr(ps, "video_domain_guard_violation", lambda *a, **k: (False, None))
        monkeypatch.setattr(ps, "video_sharpness_ok", lambda p: True)
        monkeypatch.setattr(ps, "measure_luma", lambda p: 0.4)

        def director_style(probe, candidate_query=None, aesthetic_val=None):
            # Кандидат 2 явно сильнее по director-сигналу (аналог более
            # выигрышной крупности плана/домена), при равной "смысловой"
            # части — то, что раньше решал только голый sentence_relevance.
            return 0.9 if "2.mp4" in probe else 0.1

        out = ps.pexels_video("medieval sword close up", 9, used_ids=set(),
                              used_hashes=[], sentence_score_fn=director_style)
        assert out is not None
        assert open(out, "rb").read() == b"2", "не победил кандидат с более высоким director-скором"
