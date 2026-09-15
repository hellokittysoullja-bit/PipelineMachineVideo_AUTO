# -*- coding: utf-8 -*-
"""Язык камеры: инварианты, которые нельзя сломать молча.

Ни один тест здесь не рендерит и не ходит в сеть. Держим ровно то, из-за
чего этот слой вообще опасен: он меняет КАЖДЫЙ кадр ролика, и ошибка в нём
видна зрителю всегда, в отличие от ошибки в подборе одного слота.
"""
import hashlib
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import camera_language as cl  # noqa: E402
import pipeline_smart as ps  # noqa: E402


def _h(name="slot0001.jpg"):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------- fail-open: ноль регресса

def test_without_a_stage_behaviour_is_byte_identical(monkeypatch):
    """Эпизод без speech_plan.json обязан рендериться ровно как раньше.

    Это главная гарантия слоя: он не имеет права трогать ролики, для
    которых драматургия не посчитана. Проверяется перебором, а не на одном
    удобном блоке."""
    monkeypatch.setenv("CAMERA_LANGUAGE", "1")
    b = {"section": "BLOCK 3", "text": "текст"}
    for name in (f"s{i}.jpg" for i in range(40)):
        h = _h(name)
        for shot in (None, "wide", "close", "detail"):
            assert (ps.choose_motion_mode(b, False, h, shot_size=shot, arc_stage=None)
                    == ps.choose_motion_mode(b, False, h, shot_size=shot))


def test_flag_off_behaviour_is_byte_identical(monkeypatch):
    """Откат одной переменной — и камера снова ездит как раньше."""
    b = {"section": "BLOCK 3", "text": "текст"}
    for name in (f"s{i}.jpg" for i in range(40)):
        h = _h(name)
        monkeypatch.setenv("CAMERA_LANGUAGE", "0")
        off = ps.choose_motion_mode(b, False, h, arc_stage="слом")
        monkeypatch.setenv("CAMERA_LANGUAGE", "0")
        plain = ps.choose_motion_mode(b, False, h)
        assert off == plain


def test_an_unknown_stage_name_falls_back_instead_of_crashing(monkeypatch):
    """speech_plan.json от другой версии сценария / чужой канал / опечатка —
    слой обязан молча уступить, а не уронить рендер и не выдумать режим."""
    monkeypatch.setenv("CAMERA_LANGUAGE", "1")
    b = {"section": "BLOCK 3", "text": "текст"}
    h = _h()
    assert (ps.choose_motion_mode(b, False, h, arc_stage="такой-стадии-нет")
            == ps.choose_motion_mode(b, False, h, arc_stage=None))


# ----------------------------------------- структурные правила ВЫШЕ языка

@pytest.mark.parametrize("stage", sorted(cl.STAGE_CAMERA))
def test_a_number_on_screen_always_holds_still(monkeypatch, stage):
    """Цифра обязана читаться. Ни одна стадия рассказа не вправе заставить
    кадр с цифрой ехать — иначе зритель не успевает её прочесть."""
    monkeypatch.setenv("CAMERA_LANGUAGE", "1")
    b = {"section": "BLOCK 3", "text": "текст", "stat": "15 кг"}
    assert ps.choose_motion_mode(b, False, _h(), arc_stage=stage) == "static_hold"


@pytest.mark.parametrize("stage", sorted(cl.STAGE_CAMERA))
def test_a_section_opening_is_still_an_establishing_pull(monkeypatch, stage):
    """Открывашка раздела — это establishing, и это решение старше языка."""
    monkeypatch.setenv("CAMERA_LANGUAGE", "1")
    b = {"section": "BLOCK 3", "text": "текст"}
    assert ps.choose_motion_mode(b, True, _h(), arc_stage=stage) == "slow_pull"


@pytest.mark.parametrize("stage", sorted(cl.STAGE_CAMERA))
def test_a_subcut_never_pulls_attention(monkeypatch, stage):
    """Под-кадр не должен тянуть внимание с главного кадра фразы."""
    monkeypatch.setenv("CAMERA_LANGUAGE", "1")
    b = {"section": "BLOCK 3", "text": "текст", "is_subcut": True}
    assert ps.choose_motion_mode(b, False, _h(), shot_size="detail",
                                 arc_stage=stage) in ("micro_drift", "static_hold")


# ------------------------------------------------------ сама палитра

def test_every_mode_in_every_palette_actually_exists():
    """Опечатка в имени режима дала бы кадр, который kenburns() не узнает, и
    он молча уехал бы в ветку classic_kb — то есть слой выглядел бы
    работающим и не работал."""
    for stage, palette in cl.STAGE_CAMERA.items():
        assert palette, stage
        for mode in palette:
            assert mode in ps.MOTION_MODES, (stage, mode)


def test_shot_size_narrowing_never_empties_the_palette():
    """Пустая палитра означала бы «стадии нет» — то есть молча отключила бы
    слой ровно на самых тесных кадрах, где выбор движения важнее всего."""
    for stage in cl.STAGE_CAMERA:
        for shot in (None, "wide", "close", "detail", "medium", "чепуха"):
            assert cl.camera_palette(stage, shot), (stage, shot)


def test_a_tight_frame_never_gets_a_fast_push():
    """Быстрый наезд на и без того крупном плане съедает сам объект — та же
    причина, что уже записана в choose_motion_mode()."""
    for stage in cl.STAGE_CAMERA:
        for shot in ("close", "detail"):
            assert "snap_push" not in cl.camera_palette(stage, shot), (stage, shot)


def test_the_hash_still_has_something_to_vary():
    """Стадия сужает палитру, но не должна решать ЗА хэш полностью там, где
    у неё есть выбор: палитра из одного элемента превращает стадию в
    жёсткое правило, а это ровно тот «один почерк автослайдшоу», против
    которого слой и написан. Одноэлементные палитры допустимы только как
    осознанное исключение — сейчас их нет ни одной."""
    single = [s for s, p in cl.STAGE_CAMERA.items() if len(p) < 2]
    assert not single, single


def test_hook_rule_in_the_code_agrees_with_the_table():
    """У ХУКА палитра была зашита в choose_motion_mode ЗАДОЛГО до этого
    модуля и остаётся там (структурное правило старше языка). Значит
    таблица и код — две записи одного решения, и разойтись им нельзя:
    вторая копия правила в этом репозитории уже стоила эпизоду PHRASE
    LOCK."""
    b = {"section": "HOOK", "text": "текст"}
    seen = {ps.choose_motion_mode(b, False, _h(f"h{i}.jpg")) for i in range(64)}
    assert seen == set(cl.STAGE_CAMERA["hook"]), (seen, cl.STAGE_CAMERA["hook"])


# ------------------------------------------------------ подпись рецепта

def test_the_palette_table_is_part_of_the_render_recipe():
    """Правка палитры меняет движение УЖЕ отрендеренного клипа, при этом
    исходник choose_motion_mode остаётся побайтово прежним. Без таблицы в
    подписи правка не дошла бы до экрана на прогретом temp_smart/ — ровно
    тот класс, ради которого подпись рецепта и заведена."""
    before = ps.render_recipe_signature()
    original = cl.STAGE_CAMERA["слом"]
    try:
        cl.STAGE_CAMERA["слом"] = ("static_hold", "micro_drift")
        assert ps.render_recipe_signature() != before
    finally:
        cl.STAGE_CAMERA["слом"] = original
    assert ps.render_recipe_signature() == before


def test_direction_table_is_part_of_the_recipe_too():
    before = ps.render_recipe_signature()
    original = cl.STAGE_ZOOM_IN["слом"]
    try:
        cl.STAGE_ZOOM_IN["слом"] = not original
        assert ps.render_recipe_signature() != before
    finally:
        cl.STAGE_ZOOM_IN["слом"] = original
    assert ps.render_recipe_signature() == before


# --------------------------------------- слой РЕАЛЬНО подключён к рендеру
#
# Обе проверки ниже добавлены ПОСЛЕ контрольного прогона, который показал,
# что без них слой можно отключить целиком — и все остальные тесты файла
# останутся зелёными. Это ровно тот класс дефекта, которым этот репозиторий
# отличился пять раз («код есть, ролику не даёт ничего»), только пойманный
# здесь до коммита, а не по готовому ролику.

@pytest.mark.parametrize("stage", sorted(cl.STAGE_CAMERA))
def test_the_stage_actually_decides_the_mode(monkeypatch, stage):
    """С известной стадией выбранный режим ОБЯЗАН быть из её палитры.

    Проверка именно такая, а не «результат отличается от прежнего»: отличие
    можно получить случайно, а принадлежность палитре ломается сразу, как
    только ветка языка перестаёт вызываться, — тогда возвращается прежний
    дефолт (classic_kb/horizontal_pan), которого в палитре может не быть."""
    monkeypatch.setenv("CAMERA_LANGUAGE", "1")
    b = {"section": "BLOCK 3", "text": "текст"}
    for i in range(48):
        mode = ps.choose_motion_mode(b, False, _h(f"x{i}.jpg"), arc_stage=stage)
        assert mode in cl.camera_palette(stage), (stage, mode)


def test_the_stage_changes_what_would_have_been_chosen(monkeypatch):
    """И этот выбор ОТЛИЧАЕТСЯ от того, что дала бы монетка.

    Без этого предыдущий тест можно было бы удовлетворить палитрой, целиком
    совпадающей с дефолтом, то есть слоем, который ничего не меняет."""
    monkeypatch.setenv("CAMERA_LANGUAGE", "1")
    b = {"section": "BLOCK 3", "text": "текст"}
    hashes = [_h(f"y{i}.jpg") for i in range(60)]
    plain = [ps.choose_motion_mode(b, False, h) for h in hashes]
    staged = [ps.choose_motion_mode(b, False, h, arc_stage="доказательство")
              for h in hashes]
    assert plain != staged
    # и это не случайное расхождение на одном кадре
    assert sum(a != b_ for a, b_ in zip(plain, staged)) > len(hashes) // 2


def test_the_narrowing_guard_survives_a_palette_it_would_empty():
    """Гвард «сужение не опустошает палитру» сегодня не срабатывает ни на
    одной реальной стадии — и потому его легко удалить незаметно.
    Проверяем его на палитре, которую крупность СНЯЛА БЫ целиком: без
    гварда слой молча выключался бы ровно на самых тесных кадрах."""
    original = cl.STAGE_CAMERA["слом"]
    try:
        # оба режима этой палитры крупность отбрасывает (см. SHOT_SIZE_DROP)
        cl.STAGE_CAMERA["слом"] = ("snap_push", "horizontal_pan")
        assert cl.camera_palette("слом", "close") == ("snap_push", "horizontal_pan")
        assert cl.pick_from_palette(cl.camera_palette("слом", "detail"), 12345)
    finally:
        cl.STAGE_CAMERA["слом"] = original
