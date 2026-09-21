# -*- coding: utf-8 -*-
"""Длительность кадра: две правки пайплайна дрались, и побеждала длинная.

Найдено владельцем на готовом ролике («для хука фоточки длинноваты»),
подтверждено замером. ЧАСТЬ 14 предписывает хук-слотам 3-6 секунд; замер
по пикселям final.mp4 дал **3 кадра, среднюю 6.94с и максимум 8.42с**,
причём самый длинный кадр — СТАТИЧНОЕ ФОТО.

Механизм, измеренный на реальных блоках:

    split_long_blocks   режет блок 8.10с -> 1.10с + 7.00с
                        (граница предложения после «Я его назову.»)
    merge_short_...     1.10с ниже пола клипа хука 2.2с -> склеивает обратно
    итог                8.10с, как было

Другую точку реза при этом не пробовал никто: fallback-и (контраст-союзы,
число, середина) стоят под `if not split_at`, а список был НЕ пуст — в нём
лежала непригодная точка. Плюс сама середина была недостижима, если
пунктуационная граница нашлась вообще, даже когда та не проходила по полу.
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

# Реальный блок хука эпизода _test60s и его реальный вес по alignment.
HOOK_BLOCK_TEXT = ("Я его назову. Но если сказать прямо сейчас, ты пожмёшь "
                   "плечами и уйдёшь — без подготовки это прозвучит скучно "
                   "и даже глупо.")
HOOK_BLOCK_EST = 8.10


def _block(text, section="HOOK", **kw):
    b = {"text": text, "words": len(text.split()), "section": section,
         "pause_after": 0.8, "stat": None, "stat_word_pos": None,
         "is_climax": False, "sfx": [], "hush": False}
    b.update(kw)
    return b


class TestSplitPointRespectsTheClipFloor:
    def test_the_real_hook_block_is_actually_split(self):
        """Негативный контроль: убрать фильтр _usable_split_points — и блок
        снова уезжает целиком (проверено прогоном на прежнем коде)."""
        blocks, weights = ps.split_long_blocks(
            [_block(HOOK_BLOCK_TEXT)], [HOOK_BLOCK_EST])
        assert len(blocks) == 2, [b["text"] for b in blocks]
        assert all(w >= ps.SUBCUT_MIN_PART_DUR for w in weights), weights

    def test_the_split_survives_the_phrase_lock_merge(self):
        """Главная проверка: рез бесполезен, если следующий же шаг его
        склеивает. Обе правки обязаны работать ВМЕСТЕ."""
        sb, sw = ps.split_long_blocks([_block(HOOK_BLOCK_TEXT)], [HOOK_BLOCK_EST])
        mb, mw = ps.merge_short_phrase_locked_blocks(sb, sw, sum(sw))
        assert len(mb) == 2, [b["text"] for b in mb]
        assert max(mw) < HOOK_BLOCK_EST

    def test_every_part_clears_the_hook_clip_floor(self):
        _, weights = ps.split_long_blocks([_block(HOOK_BLOCK_TEXT)], [HOOK_BLOCK_EST])
        assert all(w >= ps.HOOK_MIN_CLIP for w in weights), weights

    def test_unusable_sentence_boundary_does_not_block_the_fallbacks(self):
        """Корень дефекта в одной строке: непригодная точка оставалась в
        списке и закрывала дорогу остальным триггерам."""
        usable = ps._usable_split_points([3], len(HOOK_BLOCK_TEXT.split()),
                                         HOOK_BLOCK_EST, ps.SUBCUT_MIN_PART_DUR)
        assert usable == [], usable

    def test_block_that_cannot_be_split_is_left_alone(self):
        """Честный отказ вместо молчаливого реза пополам: блок, у которого
        любой рез оставляет кусок ниже пола, остаётся одним кадром."""
        text = "Короткая фраза. И вторая."
        blocks, _ = ps.split_long_blocks([_block(text)], [4.0])
        assert len(blocks) == 1

    def test_a_short_first_chunk_is_never_emitted(self):
        """Дыра в цикле склейки: у ПЕРВОГО куска merged ещё пуст, и короткий
        первый кусок проходил вообще без проверки."""
        for est in (8.1, 9.0, 12.0, 20.0):
            _, weights = ps.split_long_blocks([_block(HOOK_BLOCK_TEXT)], [est])
            assert min(weights) >= ps.SUBCUT_MIN_PART_DUR, (est, weights)


class TestSfxIsNotCopiedIntoEverySubcut:
    """У stat и is_climax сброс по кускам есть, у sfx его не было: dict(b)
    копировал список во ВСЕ под-кадры, а word_pos оставался от исходного
    блока. Замер 14.09: один тег в сценарии дал ДВА кюя."""

    def test_one_tag_stays_one_cue(self):
        b = _block(HOOK_BLOCK_TEXT, sfx=[{"name": "armour_clank", "word_pos": 0}])
        blocks, _ = ps.split_long_blocks([b], [HOOK_BLOCK_EST])
        assert len(blocks) == 2
        assert sum(len(x["sfx"]) for x in blocks) == 1

    def test_the_cue_lands_in_the_chunk_that_owns_its_word(self):
        b = _block(HOOK_BLOCK_TEXT, sfx=[{"name": "armour_clank", "word_pos": 15}])
        blocks, _ = ps.split_long_blocks([b], [HOOK_BLOCK_EST])
        assert blocks[0]["sfx"] == []
        assert [x["name"] for x in blocks[1]["sfx"]] == ["armour_clank"]

    def test_word_pos_is_remapped_to_the_chunk(self):
        """Не пересчитать позицию — значит поставить звук по чужой фразе."""
        b = _block(HOOK_BLOCK_TEXT, sfx=[{"name": "armour_clank", "word_pos": 15}])
        blocks, _ = ps.split_long_blocks([b], [HOOK_BLOCK_EST])
        pos = blocks[1]["sfx"][0]["word_pos"]
        assert 0 <= pos < blocks[1]["words"], (pos, blocks[1]["words"])
        assert pos < 15

    def test_tag_at_the_very_start_stays_in_the_first_chunk(self):
        b = _block(HOOK_BLOCK_TEXT, sfx=[{"name": "armour_clank", "word_pos": 0}])
        blocks, _ = ps.split_long_blocks([b], [HOOK_BLOCK_EST])
        assert [x["name"] for x in blocks[0]["sfx"]] == ["armour_clank"]
        assert blocks[1]["sfx"] == []


def test_measured_hook_average_moves_into_the_documented_corridor():
    """Числа замера целиком: тот же хук, те же веса по alignment.

    Было 3 кадра / средняя 6.94с / макс 8.42с при норме ЧАСТИ 14 «3-6 сек».
    Здесь проверяется не круглая цифра, а направление и коридор."""
    texts = [
        ("При этом он был под ногами у каждого из них. С первого шага на поле.", 4.60),
        (HOOK_BLOCK_TEXT, HOOK_BLOCK_EST),
        ("Поэтому сперва придётся сломать три вещи, которые ты считаешь "
         "правдой про рыцарей. Все три — неправда.", 6.26),
    ]
    blocks = [_block(t) for t, _ in texts]
    weights = [w for _, w in texts]
    sb, sw = ps.split_long_blocks(blocks, weights)
    mb, mw = ps.merge_short_phrase_locked_blocks(sb, sw, sum(sw))
    assert len(mb) == 4, [b["text"][:30] for b in mb]
    avg = sum(mw) / len(mw)
    assert avg < 6.94, avg
    assert max(mw) < 8.42, mw
    assert min(mw) >= ps.HOOK_MIN_CLIP, mw
    # Сумма сохраняется: рез перераспределяет время, а не выдумывает его.
    assert sum(mw) == pytest.approx(sum(weights), abs=0.01)
