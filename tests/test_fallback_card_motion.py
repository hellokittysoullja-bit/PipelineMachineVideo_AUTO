#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Карточка не едет камерой: она ТЕКСТ, а не фотография.

Найдено глазами на готовом ролике `videos/02_ne-mechom/final.mp4` (17.09):
на 200-й секунде надпись «Лезвие, которое выпустило бы кишки, встречает
нагрудни…» обрезана по обоим краям кадра, а к 203-й секунде влезает
целиком. Вёрстка тут не виновата — fallback_card.py переносит строки ровно
под 1920x1080; виноват Ken Burns, который применяется к карточке как к
обычному фото и на крупной фазе зума съедает края.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import pipeline_smart as ps  # noqa: E402


def test_card_is_recognised_by_its_directory_not_by_a_name_list():
    card = os.path.join("temp_smart", ps.FALLBACK_CARD_DIR_NAME, "card_0007_ab12cd34.png")
    assert ps.is_fallback_card_media(card) is True
    assert ps.is_fallback_card_media(os.path.join("media", "007_stock.jpg")) is False
    assert ps.is_fallback_card_media(None) is False


def test_card_directory_name_is_the_one_the_builder_uses():
    """Две копии имени каталога разошлись бы молча, и проверка начала бы
    отвечать False на настоящую карточку."""
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    assert 'os.path.join(TEMP_FOLDER, FALLBACK_CARD_DIR_NAME)' in src
    assert 'os.path.join(TEMP_FOLDER, "fallback_cards")' not in src


def test_main_forces_static_hold_for_cards():
    """Проверяется ИСХОДНИК места решения, а не поведение через полный
    рендер: motion_mode выбирается внутри main() на 1900 строк, и вызвать
    этот участок в изоляции нельзя без фикстуры целого эпизода. Контрольный
    прогон со снятой правкой роняет этот тест."""
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    i = src.index("motion_mode = choose_motion_mode(")
    tail = src[i:i + 1200]
    assert "is_fallback_card_media(photo)" in tail
    assert 'motion_mode = "static_hold"' in tail
