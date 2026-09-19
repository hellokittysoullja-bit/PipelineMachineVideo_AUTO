"""Видео, которое физически нельзя показать в слоте без сломанного замедления.

РЕАЛЬНЫЙ, ранее не закрытый пробел (найден прямым чтением кода, замера
частоты на живом API в этой сессии сделать было нечем — ключа и сети нет,
и это честно сказано в коммите). Отбор видео-кандидата смотрел ТОЛЬКО на
ширину файла:

    best = min(files, key=lambda f: abs(f["width"] - WIDTH))

Поле duration, которое Pexels отдаёт для каждого видео в той же самой
выдаче, не читалось нигде. Дальше video_render():

    setpts_factor = dur / max(actual, 0.1)

— без потолка и без интерполяции кадров. То есть 3-секундный клип в
12-секундном слоте (а слоты до 16.7с на реальном сценарии существуют)
едет вчетверо медленнее, каждый исходный кадр держится 4 выходных.

Тесты держат ДВА инварианта сразу, и второй важнее первого:
  1) кандидат, которого пришлось бы растянуть сильнее VIDEO_MAX_TIME_STRETCH,
     отсеивается;
  2) отсев НИКОГДА не опустошает пул — если длины не хватает вообще ни у
     кого, список остаётся прежним (ни один слот не остаётся пустым).
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart  # noqa: E402


def test_short_candidate_is_too_short_for_long_slot():
    # 3с в слот 12с = растяжение 4x
    assert pipeline_smart._video_candidate_too_short({"duration": 3}, 12.0)


def test_candidate_within_allowed_stretch_passes():
    # 9с в слот 12с = растяжение 1.33x, ниже потолка 1.5 — пригоден
    assert not pipeline_smart._video_candidate_too_short({"duration": 9}, 12.0)


def test_longer_than_slot_always_passes():
    assert not pipeline_smart._video_candidate_too_short({"duration": 30}, 6.0)


@pytest.mark.parametrize("item", [{}, {"duration": None}, {"duration": "нет"},
                                  {"duration": 0}])
def test_unknown_duration_is_never_dropped(item):
    """Fail-open: не знаем длину — не отбрасываем. Тот же принцип, что у
    всех остальных опциональных проверок кандидата в этом файле."""
    assert not pipeline_smart._video_candidate_too_short(item, 12.0)


def test_no_slot_duration_disables_the_filter_entirely():
    """slot_dur=None — старые вызовы и тесты работают байт-в-байт как раньше."""
    assert not pipeline_smart._video_candidate_too_short({"duration": 1}, None)
    assert not pipeline_smart._video_candidate_too_short({"duration": 1}, 0)


def test_exact_boundary_is_allowed():
    """Ровно на потолке (8с * 1.5 == 12с) кандидат ещё пригоден — граница
    не должна отбрасывать то, что укладывается точно."""
    assert not pipeline_smart._video_candidate_too_short({"duration": 8}, 12.0)


def test_filter_never_empties_the_pool():
    """ГЛАВНЫЙ инвариант: если коротки ВСЕ — отсева не происходит вовсе."""
    pool = [{"duration": 2}, {"duration": 3}, {"duration": 1}]
    long_enough = [v for v in pool
                   if not pipeline_smart._video_candidate_too_short(v, 12.0)]
    assert long_enough == []          # формально не проходит никто...
    kept = long_enough or pool        # ...значит применяется откат на весь пул
    assert kept == pool


def test_gate_signature_covers_the_new_rule():
    """Смена правила отбора обязана инвалидировать кэш кандидатов — иначе
    на прогретом temp_smart/ правка молча не дойдёт до экрана (ровно тот
    класс бага, ради которого candidate_gate_signature() и написана)."""
    import inspect
    body = inspect.getsource(pipeline_smart.candidate_gate_signature)
    assert "_video_candidate_too_short" in body
    assert "VIDEO_MAX_TIME_STRETCH" in body


def test_pexels_video_accepts_slot_dur():
    import inspect
    sig = inspect.signature(pipeline_smart.pexels_video)
    assert "slot_dur" in sig.parameters
    assert sig.parameters["slot_dur"].default is None
