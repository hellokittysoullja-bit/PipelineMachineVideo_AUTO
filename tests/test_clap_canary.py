# -*- coding: utf-8 -*-
"""Канарейка на дрейф ML-стека для отбора звука.

Зачем она есть. Дрейф уже случался вживую, не гипотетически: на torch 2.14 /
transformers 5.17 кадр золотого набора (`ep01_032`, релевантность 0.2006 при
пороге 0.19) поехал через порог без единой правки кода — и отличить
«сломался мой код» от «поменялась модель» было нечем, кроме ручного git diff.
Числа без записанной версии стека — не факт, а воспоминание.

Что здесь заморожено: четыре реальных CC0-отрезка по 8 секунд (по одному на
заметно разный вид — громкая текстура и тихая) и скоры CLAP, которые на них
дал стек, записанный в `canary.json`. Отрезки взяты из ИСХОДНИКА со стока, а
не из готового файла библиотеки: канарейка обязана мерить МОДЕЛЬ, иначе
правка обработки при импорте будет выглядеть как дрейф модели.

Честный предел: канарейка не говорит, какой стек правильный. Она говорит,
что он изменился, и на сколько — дальше решает человек. Провал здесь не
означает «код сломан»; он означает «перемерь пороги, прежде чем доверять
старым числам».

Допуск 0.02 выбран не на глаз: сам прогон на одном стеке побитово
детерминирован (проверено — margin совпадает до 1e-6 между запусками), а
порог принятия записи CLAP_MIN_MARGIN = 0.04, то есть 0.02 — половина
расстояния до решения. Меньший дрейф решений не переворачивает, больший
может.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FIX = os.path.join(ROOT, "tests", "fixtures", "clap_canary")
CANARY = os.path.join(FIX, "canary.json")
TOLERANCE = 0.02


def _has_ml():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _canary():
    with open(CANARY, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------ проверяемо без моделей
def test_canary_records_the_stack_that_produced_the_numbers():
    """Число без версии стека необоснованно: по нему нельзя отличить дрейф
    модели от правки кода — ровно та ситуация, из-за которой канарейка и
    появилась."""
    d = _canary()
    assert d["stack"].get("torch"), d["stack"]
    assert d["stack"].get("transformers"), d["stack"]


def test_canary_fixtures_exist_and_carry_licence():
    """Отрезки — реальные записи со стока, значит у каждой обязано быть
    подтверждённое происхождение: это единственный звук проекта, который
    попадает в репозиторий."""
    d = _canary()
    assert len(d["items"]) >= 3, "одной точки мало: дрейф надо видеть на разных участках шкалы"
    for it in d["items"]:
        assert os.path.exists(os.path.join(FIX, it["file"])), it["file"]
        assert it["license"] == "cc0", it
        assert it["license_url"] and it["landing"], it


def test_canary_margins_are_above_the_gate():
    """Записанные точки обязаны быть принятыми записями — канарейка на
    заведомо пограничном материале мерила бы шум, а не дрейф."""
    import sound_library as sl

    for it in _canary()["items"]:
        assert it["clap_margin"] >= sl.CLAP_MIN_MARGIN, it


# ------------------------------------------------------ живой прогон CLAP
@pytest.mark.skipif(not _has_ml(), reason="нужны torch/transformers")
def test_clap_scores_have_not_drifted():
    import torch
    import transformers

    import sound_library as sl

    d = _canary()
    now = {"torch": torch.__version__, "transformers": transformers.__version__}
    drift = []
    for it in d["items"]:
        w = sl.decode_f32(os.path.join(FIX, it["file"]), 0.0, d["window_sec"], 48000)
        rows = sl.clap_scores([w], [it["prompt"]] + it["negatives"])[0]
        pos = float(rows[0])
        worst = max(float(x) for x in rows[1:])
        margin = pos - worst
        if abs(margin - it["clap_margin"]) > TOLERANCE:
            drift.append(f"{it['name']}: было {it['clap_margin']:+.4f}, стало {margin:+.4f}")
    assert not drift, (
        "Скоры CLAP разошлись с записанными.\n"
        f"Записаны на стеке: {d['stack']}\nСейчас стек: {now}\n"
        + "\n".join(drift)
        + "\nЭто не обязательно поломка кода — но пороги отбора (CLAP_MIN_MARGIN, "
          "конкуренция видов) откалиброваны на старых числах, и доверять им "
          "до перепроверки нельзя."
    )


@pytest.mark.skipif(not _has_ml(), reason="нужны torch/transformers")
def test_ast_probabilities_have_not_drifted():
    """AST важнее сторожить, чем CLAP: у него пороги вето АБСОЛЮТНЫЕ
    (AST_VETO), а не относительные — то есть дрейф самой модели сдвигает
    решение напрямую, без всякой правки кода.

    Допуск здесь свой и жёстче: вето на речь стоит на 0.12, а записанные
    точки лежат около 0.002-0.03, так что расхождение в 0.05 — это уже
    треть расстояния до решения.
    """
    import torch
    import transformers

    import sound_library as sl

    d = _canary()
    labels = d["ast_labels"]
    now = {"torch": torch.__version__, "transformers": transformers.__version__}
    drift = []
    for it in d["items"]:
        w16 = sl.decode_f32(os.path.join(FIX, it["file"]), 0.0, d["window_sec"], 16000)
        probs = sl.ast_probs([w16], labels)
        pr = probs[0] if isinstance(probs, list) else probs
        cur = dict(pr) if isinstance(pr, dict) else dict(zip(labels, pr))
        for lab in labels:
            was = it["ast"][lab]
            got = float(cur.get(lab, 0.0))
            if abs(got - was) > 0.05:
                drift.append(f"{it['name']}/{lab}: было {was:.4f}, стало {got:.4f}")
    assert not drift, (
        f"Вероятности AST разошлись с записанными.\nЗаписаны на стеке: {d['stack']}\n"
        f"Сейчас стек: {now}\n" + "\n".join(drift)
        + "\nПороги AST_VETO абсолютные — их придётся перепроверить."
    )
