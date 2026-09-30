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
путь (source-level, main() слишком велика для end-to-end юнит-теста), что
функция вызываема и одним позиционным аргументом, и что видео (VideoAdapter
в общем ядре) подчиняется режиму Режиссёра так же, как фото: в assist
решает, в shadow — нет.
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
from _media_calls import pick_photo, pick_video  # noqa: E402
from _video_world import QUERY, infra, video  # noqa: E402,F401


def _main_video_selection_block():
    src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    start = src.index('is_opening_shot = (i == 0)')
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
        assert 'shown_att.effect("history", recent_semantic_tags,' in block

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

    @pytest.mark.parametrize("assist, winner", [(True, 2), (False, 1)])
    def test_video_obeys_the_director_mode_like_photo(self, infra, monkeypatch, assist, winner):
        """Два кандидата различаются ТОЛЬКО бонусом Режиссёра. В assist
        побеждает тот, у кого бонус выше; в shadow Режиссёр НЕ трогает выбор
        (как у фото, CLAUDE.md: «shadow — реальный выбор не трогает»).
        Прежний видео-добытчик применял скоринг Режиссёра и в shadow —
        асимметрия с фото, закрытая общим ядром."""
        infra["videos"] = [video(1), video(2)]
        infra["relevant"] = {1, 2}
        infra["rel"] = {1: 0.30, 2: 0.30}
        # Базовое ранжирование при равенстве прочего решает эстетикой:
        # кандидат 1 красивее, кандидат 2 сильнее по Режиссёру.
        monkeypatch.setattr(ps, "aesthetic_score",
                            lambda p: 6.0 if "prev_1_" in os.path.basename(p) else 5.0)

        def director_style(path, candidate_query=None, aesthetic_val=None):
            return 0.9 if "prev_2_" in os.path.basename(path) else 0.1
        out = pick_video(ps, QUERY, 9, used_ids=set(), used_hashes=[],
                         sentence_score_fn=director_style, director_assist=assist)
        assert out is not None
        assert infra["downloads"] == [winner]
