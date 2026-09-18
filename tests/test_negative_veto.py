"""Контрастивное вето по ловушкам-негативам: локально, без квот, без арбитра.

Задача, которую оно решает. Ни один гейт не спрашивал «нет ли в историческом
кадре современного спорта, толпы, улицы». Два предыдущих захода на это
провалились, и оба провала полезны:

  1. ОДИН обобщённый негатив («современная съёмка») против «исторической
     сцены» — распределения годных и брака перекрылись полностью.
  2. АБСОЛЮТНЫЕ скоры предметных маркеров — сырой косинус CLIP несопоставим
     между РАЗНЫМИ текстами: максимум маркера «смартфон» у годного музейного
     доспеха оказался выше, чем у половины брака.

Работает третья форма: margin НА ОДНОЙ КАРТИНКЕ между целевым запросом и
конкретной ловушкой. Это ровно тот механизм, что уже работает в
VISUAL_DOMAIN_GUARDS (европейский клинок минус восточноазиатский), только
обобщённый с одной оси на восемь.

Замер на золотом наборе из 40 реальных кадров опубликованного эпизода:
пропуск брака 0.7647 -> 0.5882, анахронизмы 0.6364 -> 0.5455, ложный отказ
годным 0.0 -> 0.0. Улучшение без единого регресса.

Второй заход (07.09, порог -0.02 -> -0.015, по прямой жалобе на кадр #000 —
кухня/хлопья на запрос "milk bottle hand"): его margin оказался -0.0175,
на 0.0025 короче старого порога. Сканирование ВСЕХ margin золотого набора
(не подгонка под один кадр) показало: -0.015 — первый порог строго между
-0.02 и 0.0, который дополнительно ловит именно этот кадр и не теряет НИ
ОДНОГО из 16 годных и 7 терпимых кадров (4 терпимых уже были пойманы и на
старом пороге -0.02 — состав их не меняется, только состав брака растёт).
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402

GOLDEN = os.path.join(REPO_ROOT, "tests", "fixtures", "golden_set", "images")
HAS_TORCH = True
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except Exception:
    HAS_TORCH = False


class TestConfiguration:
    """Без ML: сама конфигурация вето должна быть осмысленной."""

    def test_margin_is_conservative_by_construction(self):
        """Отрицательный запас — не описка.

        Ловушка обязана ощутимо ПЕРЕБИВАТЬ цель, а не просто дотянуться до
        неё. Именно эта консервативность и даёт ноль ложных отказов на
        замере; положительный порог на том же наборе начинал выкашивать
        годные кадры раньше, чем брак.
        """
        assert ps.NEGATIVE_VETO_MARGIN < 0

    def test_anchors_are_overridable_per_channel(self):
        """Для канала про современный спорт эти ловушки — нужный контент.

        Поэтому список живёт в channel_profile.json тем же паттерном, что
        CONTENT_ALT_BLOCKLIST и VISUAL_DOMAIN_GUARDS, а не в коде намертво.
        """
        assert ps.CONTENT_NEGATIVE_ANCHORS
        assert ps.CONTENT_NEGATIVE_ANCHORS == tuple(
            ps.CHANNEL_PROFILE.get("content_negative_anchors",
                                   ps._CONTENT_NEGATIVE_ANCHORS_DEFAULT))

    def test_veto_is_part_of_the_selection_signature(self):
        """Новый гейт обязан инвалидировать уже закэшированных кандидатов.

        Иначе на прогретом temp_smart/ он не дойдёт до экрана вообще: файл
        кандидата возьмётся готовым, отобранным по старым правилам.
        """
        src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"),
                   encoding="utf-8").read()
        start = src.index("def candidate_gate_signature")
        block = src[start:src.index("_CANDIDATE_GATE_SIG = \"gate:\"", start)]
        assert "negative_anchor_violation" in block
        assert "CONTENT_NEGATIVE_ANCHORS" in block
        assert "NEGATIVE_VETO_MARGIN" in block


class TestFailOpen:
    """Недоступная модель не имеет права ни ронять рендер, ни тихо
    отклонять кадры."""

    def test_disabled_mode_never_touches_the_model(self, monkeypatch):
        called = []
        monkeypatch.setattr(ps, "NEGATIVE_VETO_ENABLED", False)
        monkeypatch.setattr(ps, "clip_relevance_multi",
                            lambda *a, **k: called.append(1) or None)
        assert ps.negative_anchor_violation("x.jpg", "q") == (False, None)
        assert called == []

    def test_model_failure_passes_the_candidate_through(self, monkeypatch):
        monkeypatch.setattr(ps, "NEGATIVE_VETO_ENABLED", True)
        monkeypatch.setattr(ps, "clip_relevance_multi", lambda *a, **k: None)
        assert ps.negative_anchor_violation("x.jpg", "q") == (False, None)

    def test_empty_anchor_list_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(ps, "CONTENT_NEGATIVE_ANCHORS", ())
        assert ps.negative_anchor_violation("x.jpg", "q") == (False, None)

    def test_decision_rule_is_the_margin_not_an_absolute_score(self, monkeypatch):
        """Суть третьей формы: решает РАЗНОСТЬ на одной картинке.

        Высокий абсолютный скор ловушки сам по себе ничего не значит —
        именно на этом рассыпался второй заход.
        """
        monkeypatch.setattr(ps, "NEGATIVE_VETO_ENABLED", True)
        monkeypatch.setattr(ps, "CONTENT_NEGATIVE_ANCHORS", ("trap-a", "trap-b"))
        monkeypatch.setattr(ps, "NEGATIVE_VETO_MARGIN", -0.02)
        # Ловушка высокая, но цель выше -> кадр проходит.
        monkeypatch.setattr(ps, "clip_relevance_multi", lambda *a, **k: [0.40, 0.38, 0.10])
        assert ps.negative_anchor_violation("x.jpg", "q") == (False, None)
        # Ловушка НИЖЕ по абсолютной величине, но перебивает цель -> вето.
        monkeypatch.setattr(ps, "clip_relevance_multi", lambda *a, **k: [0.12, 0.10, 0.20])
        vetoed, who = ps.negative_anchor_violation("x.jpg", "q")
        assert vetoed and who == "trap-b"


@pytest.mark.skipif(not HAS_TORCH, reason="нужны torch/transformers")
@pytest.mark.slow
class TestOnRealFrames:
    """Живая модель на реальных кадрах опубликованного эпизода."""

    def test_batched_scoring_actually_returns_numbers(self):
        """Канарейка ровно того бага, который здесь и случился.

        Первая реализация звала get_image_features()/get_text_features() —
        в transformers 5 они возвращают объект выхода модели, а не тензор.
        Широкий except превращал вето в тихий no-op: флаг «включён», гейт не
        работает, метрика не двигается. Молчаливый отказ страшнее падения.
        """
        img = os.path.join(GOLDEN, "001.jpg")
        scores = ps.clip_relevance_multi(img, ["medieval knight sword battle",
                                               "modern city street with cars"])
        assert scores is not None, "батчевый скоринг молча вернул None"
        assert len(scores) == 2
        assert all(-1.0 <= s <= 1.0 for s in scores)

    def test_batched_matches_single_call_closely(self):
        """Батч и одиночный вызов — одна и та же модель, одна и та же пара."""
        img = os.path.join(GOLDEN, "014.jpg")
        q = "knight armor exhibit"
        single = ps.clip_relevance(img, q)
        batched = ps.clip_relevance_multi(img, [q])
        assert single is not None and batched is not None
        assert abs(single - batched[0]) < 0.02

    @pytest.mark.parametrize("name,query,what", [
        ("040.jpg", "spear pike soldiers", "рука в китайском шёлке с цзянем"),
        ("140.jpg", "19th century romantic painting knight",
         "расфокус, содержимое неразличимо"),
        ("020.jpg", "medieval sword museum display",
         "современный фехтовальный зал с вывеской FENCING"),
        ("062.jpg", "landsknecht mercenary engraving",
         "ближневосточная медная утварь"),
    ])
    def test_frames_the_veto_newly_catches(self, name, query, what):
        """Браки, которых до вето не ловил НИ ОДИН гейт.

        Закреплены поимённо, а не только агрегатом: доля может остаться
        прежней при обмене «поймали другое, потеряли это», и тогда конкретный
        кадр тихо вернётся в ролик.

        ПЕРЕСМОТРЕНО 18.09 под SigLIP2-base256 (см. NEGATIVE_VETO_MARGIN в
        pipeline_smart.py): два прежних примера (084.jpg/000.jpg, CLIP-эры)
        на новой модели margin положительный (+0.0514/+0.0934) — этот
        конкретный узкий сигнал их больше не ловит, честно задокументировано
        в test_frames_the_veto_no_longer_catches_on_the_new_model ниже, а
        не молча подогнано. Заменены на 020.jpg/062.jpg — реально пойманные
        на новой модели (см. NEGATIVE_VETO_MARGIN=-0.06)."""
        vetoed, who = ps.negative_anchor_violation(os.path.join(GOLDEN, name), query)
        assert vetoed, f"{what} снова проходит вето"
        assert who

    @pytest.mark.parametrize("name,query,what", [
        ("084.jpg", "ornate sword display", "пастельное декоративное украшение"),
        ("000.jpg", "milk bottle hand", "современная кухня, женщина с хлопьями"),
    ])
    def test_frames_the_veto_no_longer_catches_on_the_new_model(self, name, query, what):
        """ЧЕСТНО зафиксированный РЕГРЕСС узкого сигнала при смене модели
        (18.09, CLIP -> SigLIP2-base256), не скрытый провал.

        084.jpg/000.jpg — раньше ловились контрастивным вето при CLIP
        (margin отрицательный), на новой модели margin положительный
        (+0.0514/+0.0934 соответственно) — на золотом наборе безопасный
        (ноль ложных отказов good/tolerable) порог физически не дотягивается
        до этих двух margin, не подгонка, а измеренный предел (см.
        NEGATIVE_VETO_MARGIN в pipeline_smart.py). SMART_RELEVANCE_VETO
        (проверка ПОБЕДИТЕЛЯ слота более тяжёлым so400m+Jina ensemble)
        остаётся вторым, независимым слоем защиты именно для такого
        случая — контрастивное вето не единственная линия."""
        vetoed, who = ps.negative_anchor_violation(os.path.join(GOLDEN, name), query)
        assert not vetoed, (
            f"{what}: margin теперь ловится веткой — если порог перекалибровали "
            f"так, что это стало ловиться, обнови тест на test_frames_the_veto_"
            f"newly_catches, гэп не потерян молча")

    def test_borderline_miss_is_a_known_limit(self):
        """Честно зафиксированный предел, а не скрытый провал.

        ПЕРЕСМОТРЕНО 18.09 под SigLIP2-base256: #068 (старый пограничный
        случай при CLIP) на новой модели ушёл дальше от порога (margin
        +0.037, не -0.013) и больше не самый близкий промах. #006 (корейский
        дворец, бойцы в ханбоках — non_european) сейчас ближайший к порогу
        НЕПОЙМАННЫЙ брак: margin около +0.019 при пороге -0.06. Опустить
        порог настолько, чтобы поймать и его, значило бы начать терять
        good/tolerable-кадры золотого набора (см. калибровку NEGATIVE_VETO_
        MARGIN). Кадр остаётся задачей верхних ступеней лестницы, а не
        поводом расшатать порог."""
        scores = ps.clip_relevance_multi(
            os.path.join(GOLDEN, "006.jpg"),
            ["greatsword warrior fight"] + list(ps.CONTENT_NEGATIVE_ANCHORS))
        margin = scores[0] - max(scores[1:])
        assert margin > ps.NEGATIVE_VETO_MARGIN, (
            f"margin {margin:+.3f} ушёл ниже порога {ps.NEGATIVE_VETO_MARGIN} — "
            f"вето теперь его ловит, тест можно превратить в утверждение поимки")

    def test_current_threshold_adds_no_false_rejects_on_good_or_tolerable(self):
        """Прямая проверка калибровки NEGATIVE_VETO_MARGIN (18.09,
        SigLIP2-base256): порог обязан ловить брак, ни разу не задевая
        good/tolerable кадры золотого набора — единственный инвариант,
        под который порог реально подбирался (см. комментарий у
        NEGATIVE_VETO_MARGIN в pipeline_smart.py). Сравнение с legacy
        CLIP-значением -0.02 (как было раньше) больше не имеет смысла —
        шкала скоров другая, -0.02 на новой модели не «старый порог», а
        просто число не с той шкалы."""
        import json
        meta = json.load(open(os.path.join(
            REPO_ROOT, "tests", "fixtures", "golden_set", "manifest.json"),
            encoding="utf-8"))
        false_rejects = set()
        for it in meta["items"]:
            if it["verdict"] not in ("good", "tolerable"):
                continue
            img = os.path.join(REPO_ROOT, "tests", "fixtures",
                               "golden_set", it["image"])
            v, _ = ps.negative_anchor_violation(img, it["query"])
            if v:
                false_rejects.add(it["id"])
        assert not false_rejects, (
            f"текущий NEGATIVE_VETO_MARGIN={ps.NEGATIVE_VETO_MARGIN} ложно "
            f"отклоняет good/tolerable кадры: {false_rejects}")

    def test_good_museum_frames_are_not_vetoed(self):
        """Вторая ось: вето не имеет права выкашивать годное.

        Именно эти кадры (макро металла, доспех, ковка) стоят ближе всего к
        ловушкам по сырому косинусу — на них порог и проверяется на прочность.
        """
        cases = [("014.jpg", "knight armor exhibit"),
                 ("015.jpg", "knight armor exhibit"),
                 ("032.jpg", "spear pike soldiers"),
                 ("133.jpg", "blacksmith forging sword"),
                 ("128.jpg", "exhausted soldier armor")]
        wrongly = []
        for name, q in cases:
            vetoed, who = ps.negative_anchor_violation(os.path.join(GOLDEN, name), q)
            if vetoed:
                wrongly.append((name, who))
        assert not wrongly, f"вето отклонило годные кадры: {wrongly}"
