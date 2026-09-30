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

import render_core as ps  # noqa: E402






