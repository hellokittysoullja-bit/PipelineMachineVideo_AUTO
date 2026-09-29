"""Храповик качества подбора: метрика по золотому набору не должна ухудшаться.

Чем это отличается от tests/test_media_selection_golden.py. Тот тест держит
пороги и отдельные механизмы (CLIP_RELEVANCE_THRESHOLD, risky-margin, ahash)
на специально подобранных учебных фото. Этот — гоняет те же самые функции по
кадрам, которые РЕАЛЬНО ушли в опубликованный ролик и реально опозорились,
и следит за агрегатом: сколько брака гейты пропускают, сколько годного
отклоняют. Разница практическая: пороговый тест может остаться зелёным,
когда качество ролика падает, потому что он не знает, что показали зрителю.

Базовая линия (docs/quality/golden_set_baseline.json) снята 07.09 на
коммите, где эти цифры впервые стали измеримы: пропуск брака 76%,
анахронизмов 64%, ложный отказ годным 0%. Она НЕ является целью — она
является полом, ниже которого падать нельзя. Каждая правка гвардов,
скоринга или корпуса обязана либо улучшить хотя бы одну ось, либо не
трогать ни одной.

Структурная часть (манифест цел, файлы на месте, вердикты из словаря)
работает без ML. Метрическая — требует torch/transformers, как и остальные
живые тесты подбора, и идёт в CI-job `ml`.
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "golden_set")
MANIFEST = os.path.join(GOLDEN_DIR, "manifest.json")
BASELINE = os.path.join(REPO_ROOT, "docs", "quality", "golden_set_baseline.json")

sys.path.insert(0, SCRIPTS_DIR)

VALID_VERDICTS = {"good", "tolerable", "reject"}


def _manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


class TestGoldenSetIntegrity:
    """Без ML: сам набор должен быть цел и самосогласован."""

    def test_manifest_exists_and_is_documented(self):
        meta = _manifest()
        assert meta["items"], "золотой набор пуст"
        # Набор без описания вердиктов через полгода нечитаем: «tolerable»
        # без определения — это мнение, а не разметка.
        for key in ("verdicts", "reject_reasons", "honest_limits", "source"):
            assert meta.get(key), f"в манифесте нет раздела {key}"

    def test_every_item_has_its_image_on_disk(self):
        meta = _manifest()
        missing = [it["id"] for it in meta["items"]
                   if not os.path.exists(os.path.join(GOLDEN_DIR, it["image"]))]
        assert not missing, f"нет файлов кадров: {missing}"

    def test_verdicts_and_reasons_come_from_the_documented_vocabulary(self):
        meta = _manifest()
        reasons = set(meta["reject_reasons"])
        for it in meta["items"]:
            assert it["verdict"] in VALID_VERDICTS, it["id"]
            if it["verdict"] == "reject":
                assert it["reject_reason"] in reasons, (it["id"], it["reject_reason"])
            else:
                assert it.get("reject_reason") is None, it["id"]

    def test_set_covers_both_outcomes(self):
        """Набор только из брака мерил бы одну сторону.

        Ложный отказ годным — вторая ось метрики, и без годных кадров её
        нельзя посчитать вообще: «ужесточить гвард до отказа во всём» дало бы
        идеальный пропуск брака и молча убило бы подбор.
        """
        verdicts = [it["verdict"] for it in _manifest()["items"]]
        assert verdicts.count("good") >= 10
        assert verdicts.count("reject") >= 10

    def test_baseline_is_frozen_next_to_the_set(self):
        with open(BASELINE, encoding="utf-8") as f:
            base = json.load(f)
        for key in ("reject_leak", "good_false_reject", "anachronism_leak"):
            assert key in base["summary"], key

    def test_corpus_filter_does_not_regress(self):
        """Вторая ось метрики — и она НЕ требует ML.

        Гейты по пикселям меряются на уже скачанных кадрах и по построению
        слепы к тому, что происходит раньше — на этапе выдачи поиска. Между
        тем самый дешёвый рычаг лежит именно там: у видео-объектов Pexels
        нет ни alt, ни тегов, но есть человекочитаемый слаг в url. Базовая
        линия зафиксировала состояние ДО правки 07.09 (8.0% отсеянных,
        по видео — ровно 0%, потому что фильтр к видео не применялся).

        Отдельный тест, а не часть ML-класса: если torch недоступен, эта
        ось всё равно обязана сторожиться — она про строки, не про модель.
        """
        sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
        import golden_set_eval as gse

        with open(BASELINE, encoding="utf-8") as f:
            base = json.load(f)["summary"]
        now = gse.evaluate_corpus()
        assert now["blocked_share"] >= base["blocked_share"], (
            f"жанровый фильтр стал отсеивать меньше: "
            f"{base['blocked_share']} -> {now['blocked_share']}")
        assert now["starved_queries"] <= base["starved_queries"], (
            "появились запросы, где фильтр отсеивает ВСЕХ кандидатов — там "
            "срабатывает откат «вернуть как было», и ужесточение бесполезно")
        assert now["blocked_share_video"] > 0, (
            "по видео фильтр снова не работает — это и была главная дыра")


def _stack_now():
    """Версии ML-стека этого прогона."""
    try:
        import torch
        import transformers
        return {"torch": torch.__version__, "transformers": transformers.__version__}
    except Exception:
        return {}


def _stack_note():
    """Приписка к любому падению метрики: тот же стек или другой.

    Без неё красный тест не отличим от дрейфа моделей, и это не теория —
    сверку через `git stash` пришлось делать вручную дважды за одну сессию,
    а четыре красных теста успели стать фоновым шумом. Порог при этом НЕ
    трогается: подгонка под чужую сборку библиотек испортила бы калибровку
    на рабочей машине (CLAUDE.md, замечание к золотому набору 13.09).

    У аудио-канарейки (tests/fixtures/clap_canary/canary.json) поле `stack`
    есть с самого начала — у золотого набора его не было.
    """
    with open(BASELINE, encoding="utf-8") as f:
        base = json.load(f)
    was, now = base.get("stack"), _stack_now()
    if not was:
        return ("\n  СТЕК: базовая линия снята БЕЗ записи версий "
                f"(измерена {base.get('measured_at')}, коммит {base.get('commit')}); "
                f"сейчас {now}. Отличить регрессию кода от дрейфа моделей по этому "
                "красному нельзя — перемерить линию на рабочей машине "
                "(python scripts/golden_set_eval.py) и записать summary И stack.")
    if was != now:
        return (f"\n  СТЕК ДРУГОЙ: линия снята на {was}, сейчас {now}. "
                "Сначала перемерить линию здесь, потом считать это регрессией.")
    return f"\n  СТЕК ТОТ ЖЕ ({now}) — это настоящая регрессия, не дрейф."


# GPU-ветка (29.09): модель гейтов — Qwen3-VL (vision_model.py), пороги — из
# калибровки. Базовая линия SigLIP2 выше (BASELINE) остаётся для оси корпуса
# и как история; ML-замер сравнивается с линией Qwen, которую пишет
# scripts/calibrate_vision.py сразу после калибровки. Поимённые списки
# «этот брак ловится / этот течёт» описывали числа SigLIP2 и сняты вместе с
# моделью: у Qwen другие кадры по разные стороны порогов, и узнать какие —
# можно только замером на видеокарте, а не переносом.
BASELINE_QWEN = os.path.join(REPO_ROOT, "docs", "quality", "golden_set_baseline_qwen3vl.json")


@pytest.fixture(scope="module")
def report():
    """Один прогон реальных гейтов по всему набору на весь модуль. Только
    там, где модели зрения готовы: видеокарта, веса и калибровка."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
    import vision_model
    problems = vision_model.readiness()
    if problems:
        pytest.skip("модели зрения не готовы: " + "; ".join(problems))
    import golden_set_eval as gse

    meta = _manifest()
    rows, canary = gse.evaluate(meta["items"])
    return gse, rows, gse.summarize(rows), canary


@pytest.mark.slow
class TestGoldenSetMetric:
    """С ML: реальные гейты по реальным кадрам, сравнение с базовой линией Qwen."""

    def test_clip_actually_answered(self, report):
        """Канарейка: без неё метрика измеряет тишину, а не качество.

        clip_relevance() возвращает None при недоступной модели, и None во
        всех гейтах читается как «пропустить» — отчёт показал бы 100%
        пропуска брака и выглядел бы как измерение.
        """
        _, _, _, canary = report
        assert canary is not None

    def test_no_regression_against_frozen_baseline(self, report):
        gse, _, summary, _ = report
        if not os.path.exists(BASELINE_QWEN):
            pytest.skip("нет базовой линии Qwen — её пишет scripts/calibrate_vision.py")
        with open(BASELINE_QWEN, encoding="utf-8") as f:
            base = json.load(f)["summary"]
        _, regressed = gse.compare_baseline(summary, base)
        assert not regressed, (
            "метрика подбора ухудшилась по осям: " + ", ".join(regressed) +
            f"\nбыло: { {k: base.get(k) for k in regressed} }"
            f"\nстало: { {k: summary.get(k) for k in regressed} }" + _stack_note()
        )

    def test_good_frames_are_not_falsely_rejected(self, report):
        """Гейт не имеет права выбрасывать кадры, которые человек одобрил —
        калибровка выставляет пороги ровно так (ноль потерь годных и
        терпимых), и здесь это держится на ГОТОВЫХ гейтах рендера, а не на
        арифметике калибровки."""
        _, rows, _, _ = report
        wrongly = [r["id"] for r in rows if r["verdict"] in ("good", "tolerable")
                   and not r["gate_passed"]]
        assert not wrongly, f"гейт отклонил годные/терпимые кадры: {wrongly}" + _stack_note()
