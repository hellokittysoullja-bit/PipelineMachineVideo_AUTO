#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Внешний аудит 30.09 (код без видеокарты): очистка видеопамяти после главной
модели, карты без bf16, закреплённые ревизии Qwen. Torch подменяется."""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import ml_device  # noqa: E402


def _torch(cap=(8, 9), free_gib=2.0):
    calls = {"empty": 0, "idx": []}

    def mem_get_info(i):
        calls["idx"].append(i)
        return free_gib * 2 ** 30, 48 * 2 ** 30
    cuda = types.SimpleNamespace(current_device=lambda: 0, mem_get_info=mem_get_info,
                                 empty_cache=lambda: calls.__setitem__("empty", calls["empty"] + 1),
                                 get_device_capability=lambda i: cap)
    return types.SimpleNamespace(cuda=cuda), calls


def test_main_model_device_without_number_still_gets_relief():
    """Устройство главной модели записано как «cuda»: int("cuda") падал внутри
    try, и память после неё не возвращалась никогда."""
    torch, calls = _torch(free_gib=2.0)
    ml_device._relieve(torch, "cuda")
    assert calls["idx"] == [0] and calls["empty"] == 1
    torch, calls = _torch(free_gib=2.0)
    ml_device._relieve(torch, "cuda:1")
    assert calls["idx"] == [1]


@pytest.mark.parametrize("cap", [(7, 5), (7, 0), (6, 1)])   # Turing, V100, P40
def test_cards_without_bf16_are_refused_before_work(cap):
    torch, _ = _torch(cap=cap)
    with pytest.raises(RuntimeError, match="bf16"):
        ml_device.require_bf16(torch, "cuda")


@pytest.mark.parametrize("cap", [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)])
def test_ampere_and_newer_pass(cap):
    torch, _ = _torch(cap=cap)
    ml_device.require_bf16(torch, "cuda:0")


def test_qwen_revisions_are_pinned_where_weights_are_fetched():
    import fetch_weights
    import qwen_vl_embed
    import qwen_vl_rerank
    assert qwen_vl_embed.REVISION and qwen_vl_rerank.REVISION
    got = dict(fetch_weights.models(wemm=False))
    assert got[qwen_vl_embed.MODEL_NAME] == qwen_vl_embed.REVISION
    assert got[qwen_vl_rerank.MODEL_NAME] == qwen_vl_rerank.REVISION


def test_loaders_pass_the_pinned_revision():
    """Ревизия должна дойти до загрузки модели И процессора, иначе закрепление
    на диске не совпадёт с тем, что грузится."""
    import inspect
    import qwen_vl_embed
    import qwen_vl_rerank
    for mod in (qwen_vl_embed, qwen_vl_rerank):
        src = inspect.getsource(mod._load)
        assert src.count("revision=REVISION") + src.count('"revision": REVISION') >= 2, mod.__name__
        assert "require_bf16" in src, mod.__name__
