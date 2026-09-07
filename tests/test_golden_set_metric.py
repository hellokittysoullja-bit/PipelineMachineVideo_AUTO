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


@pytest.fixture(scope="module")
def report():
    """Один прогон реальных гейтов по всему набору на весь модуль.

    Модульная область видимости не оптимизация ради оптимизации: каждый тест
    ниже смотрит на РАЗНЫЙ срез одного и того же измерения, и пересчитывать
    40 кадров через CLIP на каждый тест значило бы платить минуты за
    одинаковый результат.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
    import golden_set_eval as gse

    meta = _manifest()
    rows, canary = gse.evaluate(meta["items"])
    return gse, rows, gse.summarize(rows), canary


@pytest.mark.slow
class TestGoldenSetMetric:
    """С ML: реальные гейты по реальным кадрам, сравнение с базовой линией."""

    def test_clip_actually_answered(self, report):
        """Канарейка: без неё метрика измеряет тишину, а не качество.

        clip_relevance() возвращает None при недоступной модели, и None во
        всех гейтах читается как «пропустить» — отчёт показал бы 100%
        пропуска брака и выглядел бы как измерение.
        """
        _, _, _, canary = report
        assert canary is not None and canary > 0

    def test_no_regression_against_frozen_baseline(self, report):
        gse, _, summary, _ = report
        with open(BASELINE, encoding="utf-8") as f:
            base = json.load(f)["summary"]
        _, regressed = gse.compare_baseline(summary, base)
        assert not regressed, (
            "метрика подбора ухудшилась по осям: " + ", ".join(regressed) +
            f"\nбыло: { {k: base.get(k) for k in regressed} }"
            f"\nстало: { {k: summary.get(k) for k in regressed} }"
        )

    def test_good_frames_are_not_falsely_rejected(self, report):
        """Гейт не имеет права выбрасывать кадры, которые человек одобрил.

        На базовой линии эта цифра — ровно 0, и это единственная ось,
        которая сегодня идеальна. Любое ужесточение гвардов проверяется
        в первую очередь здесь.
        """
        _, rows, _, _ = report
        wrongly = [r["id"] for r in rows if r["verdict"] == "good" and not r["gate_passed"]]
        assert not wrongly, f"гейт отклонил годные кадры: {wrongly}"

    @pytest.mark.parametrize("item_id", [
        "ep01_122",   # катана и мотив японского флага на запрос про кузнеца
        "ep01_062",   # ближневосточная медная утварь на запрос про ландскнехтов
        "ep01_020",   # современный фехтовальный зал на запрос про музей мечей
        "ep01_013",   # улица с современным туристом на тот же запрос
    ])
    def test_already_caught_rejects_stay_caught(self, report, item_id):
        """Четыре брака, которые гейты ловят СЕГОДНЯ.

        Зафиксированы поимённо, потому что агрегат может остаться прежним
        при обмене «поймали другое, потеряли это»: доля не изменится, а
        конкретный анахронизм вернётся в ролик.
        """
        _, rows, _, _ = report
        row = next(r for r in rows if r["id"] == item_id)
        assert not row["gate_passed"], (
            f"{item_id} ({row['reject_reason']}) снова проходит гейты: "
            f"relevance={row['relevance']} guard={row['domain_guard']}")

    @pytest.mark.parametrize("item_id,reason", [
        ("ep01_001", "толпа современных зрителей за реконструкторами"),
        ("ep01_005", "младенец с бутылочкой вместо бутылки молока как меры веса"),
        ("ep01_006", "корейский дворец и ханбоки"),
        ("ep01_002", "восточная боевая пластика в тёмном лесу"),
        ("ep01_040", "рука в китайском шёлке с цзянем"),
        ("ep01_068", "спортивная фехтовальная шпага вместо боевого меча"),
        ("ep01_140", "расфокус, содержимое неразличимо"),
    ])
    @pytest.mark.xfail(strict=False, reason=
                       "известная утечка базовой линии: ни один гейт сегодня не "
                       "спрашивает про современные объекты и людей в историческом "
                       "кадре. XPASS здесь — сигнал, что правка сработала.")
    def test_known_leaks_should_eventually_be_caught(self, report, item_id, reason):
        _, rows, _, _ = report
        row = next(r for r in rows if r["id"] == item_id)
        assert not row["gate_passed"], f"{item_id}: {reason}"
