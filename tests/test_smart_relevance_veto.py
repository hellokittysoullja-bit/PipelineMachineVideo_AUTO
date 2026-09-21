#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Вторая, более точная проверка ПОБЕДИТЕЛЯ слота (SigLIP2+Jina поверх
CLIP) — по прямому требованию владельца «CLIP старая модель, замени».

ИЗМЕРЕНО НА ЗОЛОТОМ НАБОРЕ (17.09), не предположено:
  AUC (ранжирует годное выше брака)   CLIP 0.566   SigLIP2+Jina 0.610
  при нуле ложных отказов годным      CLIP 3/7 терпимых потеряно,
                                       SigLIP2+Jina 2/7
Модель РЕАЛЬНО точнее — первая попытка сравнения (по средним числам, не по
рангу) показала обратное и была ошибкой метода, не находкой.

НО замена ПОЛНОЙ, на каждого кандидата пула — нет, и причина тоже измерена:
живой замер на этой машине, 15 РАЗНЫХ картинок (кэш не участвует):
  CLIP            72 мс/вызов
  SigLIP2+Jina  2358 мс/вызов  (32.8x медленнее)
is_relevant_candidate() вызывается на каждого кандидата (тысячи за эпизод) —
на этой частоте 33x медленнее means часы, а не минуты лишнего времени.
Поэтому CLIP остаётся быстрым фильтром ВСЕГО пула, а новая модель проверяет
только уже выбранного ПОБЕДИТЕЛЯ, один раз на слот.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import pipeline_smart as ps  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_miss_list():
    ps.SMART_VETO_MISSES.clear()
    yield
    ps.SMART_VETO_MISSES.clear()


def test_veto_is_noop_when_flag_disabled(monkeypatch):
    """Флаг выключен — CLIP остаётся единственным судьёй, ни одного
    дополнительного вызова модели."""
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "0")
    called = []

    def fake_sentence_relevance(*a, **kw):
        called.append(1)
        return -0.5  # заведомо ниже порога, если бы вызвалась
    import visual_director
    monkeypatch.setattr(visual_director, "sentence_relevance", fake_sentence_relevance)
    assert ps.smart_relevance_veto("any/path.jpg", "any query") is False
    assert called == [], "модель вызвана при выключенном флаге"


def test_veto_fails_open_when_model_unavailable(monkeypatch):
    """Нет torch/transformers/onnxruntime, любая ошибка — False (не
    отклоняем), тот же принцип, что у CLIP-гейтов на этой же странице."""
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "1")
    import visual_director

    def boom(*a, **kw):
        raise ModuleNotFoundError("no torch")
    monkeypatch.setattr(visual_director, "sentence_relevance", boom)
    assert ps.smart_relevance_veto("any/path.jpg", "any query") is False


def test_veto_rejects_below_threshold_accepts_above(monkeypatch):
    """Порог -0.01 — ровно граница золотого набора (худший годный кадр
    -0.003, буфер вниз, как и у CLIP 0.19 против его худшего годного 0.201)."""
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "1")
    import visual_director
    monkeypatch.setattr(visual_director, "sentence_relevance", lambda *a, **kw: -0.05)
    assert ps.smart_relevance_veto("x.jpg", "q") is True
    monkeypatch.setattr(visual_director, "sentence_relevance", lambda *a, **kw: 0.10)
    assert ps.smart_relevance_veto("x.jpg", "q") is False


def test_veto_score_none_is_not_rejected(monkeypatch):
    """None (модель не смогла оценить конкретную пару) — не отклоняем,
    тот же fail-open, что у clip_relevance()."""
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "1")
    import visual_director
    monkeypatch.setattr(visual_director, "sentence_relevance", lambda *a, **kw: None)
    assert ps.smart_relevance_veto("x.jpg", "q") is False


def test_known_bad_reason_reports_smart_veto_first(monkeypatch):
    """Прямой сигнал (модель посмотрела на РЕАЛЬНЫЙ финальный кадр) обязан
    иметь приоритет: это сильнее, чем отказ арбитра на превью шорт-листа."""
    ps.ARBITER_REJECTED_ALL.append({"index": 5, "kind": "photo"})
    ps.SMART_VETO_MISSES.append({"index": 5, "query": "q", "kind": "photo"})
    try:
        assert ps._slot_known_bad_reason(5) == "smart_relevance_veto"
    finally:
        ps.ARBITER_REJECTED_ALL.clear()


def test_video_helper_extracts_probe_and_cleans_up(monkeypatch, tmp_path):
    """video_smart_relevance_veto() достаёт кадр-пробник, зовёт ту же
    smart_relevance_veto() на нём, и убирает пробник после себя в любом
    случае — второй раз ту же формулу извлечения кадра не заводит."""
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "1")
    probe = tmp_path / "probe.jpg"
    probe.write_bytes(b"fake")
    calls = []

    def fake_extract(path, **kw):
        calls.append(path)
        return str(probe), True

    monkeypatch.setattr(ps, "extract_video_probe_frame", fake_extract)
    monkeypatch.setattr(ps, "smart_relevance_veto", lambda img, q: img == str(probe))
    result = ps.video_smart_relevance_veto("fake_video.mp4", "q")
    assert result is True
    assert calls == ["fake_video.mp4"]
    assert not probe.exists(), "кадр-пробник не удалён после использования"


def test_video_helper_noop_when_no_probe(monkeypatch):
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "1")
    monkeypatch.setattr(ps, "extract_video_probe_frame", lambda path, **kw: (None, False))
    assert ps.video_smart_relevance_veto("fake_video.mp4", "q") is False


def test_flag_and_threshold_enter_selection_signature(monkeypatch):
    """Без этого включение флага на прогретом temp_smart/ не дошло бы до
    уже закэшированных слотов: кэш-хит отдаёт файл раньше, чем эта проверка
    вообще вызывается."""
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "0")
    off = ps._selection_stack_signature()
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "1")
    on = ps._selection_stack_signature()
    assert off != on


def test_registered_in_flag_registry_with_documented_default():
    import feature_flags as ff
    assert "SMART_RELEVANCE_VETO" in ff.FLAGS
    assert ff.FLAGS["SMART_RELEVANCE_VETO"].default == "1"


def test_photo_path_calls_veto_before_accepting_winner():
    """Проверка ИСХОДНИКА места вызова — так же, как для карточки-фолбэка
    (test_fallback_card_motion.py): вызов в pexels_photo() слишком глубоко
    внутри 716-строчной функции, чтобы вызывать её в изоляции без полной
    сетевой фикстуры. Контрольный прогон со снятой правкой роняет тест.

    Ищем ВЫЗОВ, а не конкретную форму `if ...:` — с 21.09 вето итеративное
    (`while True` + ре-пик следующего кандидата), и тест, привязанный к
    старому написанию, падал бы на правке, которая инвариант не нарушает.
    """
    src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    i_veto = src.index("smart_relevance_veto(cf, query)")
    i_sidecar = src.index("write_media_sidecar(\n            cf, pexels_id=pick.get")
    assert i_veto < i_sidecar, "проверка обязана идти ДО принятия победителя, не после"


def test_video_paths_both_call_veto_before_accepting_winner():
    """Обе ветки победителя видео (video_relevance_best И dup/plain
    fallback) — тот класс, что уже трижды находили недоделанным наполовину
    (только фото, видео забыто)."""
    src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    n = src.count("video_smart_relevance_veto(cf, query)")
    # >= 2: с 21.09 у запасного яруса появился второй вызов (первый запасной
    # отклонён -> пробуем второй, вместо того чтобы убить слот). Инвариант
    # теста — «обе ветки проверяются», а не «вызовов ровно два».
    assert n >= 2, f"ожидались обе ветки видео-победителя, нашлось {n}"


def test_veto_is_iterative_not_terminal():
    """Отказ вето обязан приводить к СЛЕДУЮЩЕМУ кандидату, а не к смерти слота.

    Ради чего (замер 21.09, videos/94_dagger_test, слот «Клинок влетает в
    узкую щель»): в пуле 232 кандидата, 19 из 20 просмотренных прошли ВСЕ
    гейты, а на экране не было ничего — вето отклоняло одного победителя и
    делало `return None`. Снаружи это читалось как «сток пуст» и толкало к
    неверному выводу «ослабить гейты».
    """
    src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    assert "VETO_REPICK_MAX" in src, "предел числа ре-пиков не объявлен"
    # демотация ровно тем же приёмом, что у соседнего цикла по резкости
    assert 'winner["is_relevant"] = 0' in src, (
        "отклонённый вето кандидат обязан демотироваться и уступать место "
        "следующему по ранжированию (как winner['sharp_ok'] = 0 у резкости)")
    # у видео — свой ре-пик по already-скачанным кандидатам good
    assert "tried.add(id(nxt))" in src, "видео-путь не берёт следующего кандидата"
    assert ps.VETO_REPICK_MAX >= 1


def test_veto_repick_enters_candidate_gate_signature(monkeypatch):
    """Смена предела ре-пиков меняет победителя -> обязана инвалидировать
    уже закэшированного кандидата (иначе правка не дойдёт до экрана)."""
    src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    i_sig = src.index("def candidate_gate_signature")
    # до конца функции, а не на фиксированное число символов: у этой функции
    # огромные пояснения, и окно «на глаз» уже один раз обрезало проверку
    end = src.index("\ndef ", i_sig + 1)
    body = src[i_sig:end]
    assert "VETO_REPICK_MAX" in body, "VETO_REPICK_MAX не входит в подпись отбора"
    assert "episode_forbidden_anchors()" in body, (
        "ловушки паспорта эпизода не входят в подпись отбора — правка "
        "world_card.json не дойдёт до экрана на прогретом кэше")
