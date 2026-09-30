#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Мир и словарь канала — в профиле, в коде дефолты пустые (М2, 25.09).

ЧАСТЬ 24 CLAUDE.md: клон репозитория под другую нишу не должен молча
тащить творческие характеристики старой. Средневековый словарь жил
литералами в коде (якоря эпохи, правила «european» перед «sword», ловушки
вето, запасные запросы, типы кадра, отделы музея, чужие культуры) — клон
получал его в каждый запрос. Тест проверяет исходники (AST), а не модуль:
модуль загружает профиль этого канала, и значения там законно
средневековые."""
import ast
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMPTY_IN_CODE = {
    "scripts/pipeline_smart.py": ("_CONTENT_ALT_BLOCKLIST_DEFAULT", "QUERY_DISAMBIGUATION_RULES",
                                  "_QUERY_ERA_ANCHORS_DEFAULT", "VISUAL_DOMAIN_GUARDS",
                                  "_CONTENT_NEGATIVE_ANCHORS_DEFAULT", "_OPENVERSE_ERA_ANCHORS_DEFAULT",
                                  "_OPENVERSE_DOMAIN_NOUNS_DEFAULT"),
    "scripts/shot_types.py": ("_TYPE_LEXICON", "_MET_DEPARTMENTS_DEFAULT"),
    "scripts/museum_sources.py": ("DEFAULT_FOREIGN_CULTURE_TERMS",),
}


def _first_literal(path, name):
    tree = ast.parse(open(os.path.join(REPO, path), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path}: {name} не найден")


def test_niche_vocabulary_is_empty_in_code():
    for path, names in EMPTY_IN_CODE.items():
        for name in names:
            assert _first_literal(path, name) in ((), []), f"{path}: {name} снова несёт словарь ниши"


def test_this_channel_declares_its_world_in_the_profile():
    p = json.load(open(os.path.join(REPO, "channel_profile.json"), encoding="utf-8"))
    for key in ("query_disambiguation_rules", "query_era_anchors", "content_negative_anchors",
                "generic_fallbacks", "visual_domain_guards", "openverse_query_modifiers",
                "action_video_qualifiers", "foreign_culture_terms", "shot_type_lexicon",
                "museum_departments", "content_alt_blocklist"):
        assert p.get(key), key


def test_empty_world_leaves_a_neutral_fallback_query(monkeypatch):
    import sys
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    sys.argv = ["pipeline_smart.py", REPO]
    import pipeline_smart as ps
    monkeypatch.setattr(ps, "GENERIC_FALLBACKS", [])
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)
    assert ps.generic_fallback_queries_effective() == [ps.NEUTRAL_FALLBACK_QUERY]
