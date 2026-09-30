"""Локальный каталог Мет: точный запрос по полям вместо угадывания словами.

Проверяется не «функция что-то вернула», а три свойства, ради которых
каталог заведён, и одно, которое он обязан НЕ нарушить.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import met_catalog as mc          # noqa: E402
import museum_sources as ms       # noqa: E402


class TestPassportHoleFoundByCatalogue:
    """Каталог впервые показал ВЕСЬ прошедший паспорт корпус — и в нём
    нашлись культуры, которых не было в списке. Дефект действовал и на
    живом API-пути: выдача по «european longsword blade macro» состояла из
    двух предметов, и оба были из этой дыры («Knife blade, Afghan»,
    «Blade (Kudi tranchang), Javanese»).
    """

    @pytest.mark.parametrize("culture", [
        "Afghan", "Afghanistan", "Javanese", "Javanese or Sumatran",
        "Mongolian", "probably Mongolian", "Iraqi",
    ])
    def test_measured_holes_are_closed(self, culture):
        assert ms.culture_is_foreign(culture) is True, culture

    @pytest.mark.parametrize("culture", [
        "French", "Italian, Milan", "Flemish, possibly Antwerp",
        "Spanish, possibly Granada", "European, probably Scandinavia",
        "Western European", "Austrian, Innsbruck", "Swiss", "British",
    ])
    def test_no_european_culture_was_swept_up(self, culture):
        """225 предметов ушло из индекса — все проверены как настоящие чужие.
        Подстрочное совпадение легко задевает лишнее, поэтому нужные
        культуры заперты поимённо."""
        assert ms.culture_is_foreign(culture) is False, culture

    def test_debatable_cultures_are_left_to_the_owner(self):
        """Византия/копты/Кавказ — вопрос вкуса канала, а не анахронизм в том
        же смысле, что яванский крис. Решается в channel_profile.json;
        молча решать это за канал код не должен."""
        for c in ("Byzantine", "Coptic", "Armenian", "Georgian"):
            assert ms.culture_is_foreign(c) is False, c


class TestBudgetInvariant:
    """Главное свойство подключения: каталог занимает НАЧАЛО того же потолка
    карточек, а не добавляет свой. Иначе слот тихо стал бы стоить вдвое
    больше запросов к Мет — ровно та цена, из-за которой уже пришлось
    ограничивать слияние формулировок."""

    def test_catalogue_ids_go_first_and_total_stays_capped(self, monkeypatch):
        calls = {}

        monkeypatch.setattr(ms, "_met_get", lambda url: (
            {"objectIDs": list(range(1000, 1100))} if "/search?" in url else None))

        class FakeCat:
            @staticmethod
            def available():
                return True

            @staticmethod
            def search(q, department_name=None, limit=60):
                calls["limit"] = limit
                return [{"id": str(i)} for i in range(1, 31)]

        monkeypatch.setitem(sys.modules, "met_catalog", FakeCat)
        monkeypatch.setattr(ms.feature_flags, "enabled", lambda *a, **k: True)
        # Карточки качаются параллельно: порядок ЗАПРОСОВ в журнале — гонка
        # потоков. Проверяется порядок ОЧЕРЕДИ, поэтому один поток.
        monkeypatch.setattr(ms, "MET_DETAIL_WORKERS", 1)

        seen = []
        monkeypatch.setattr(ms, "_met_get", lambda url: (
            {"objectIDs": list(range(1000, 1100))} if "/search?" in url
            else seen.append(url) or None))

        ms.search_met("medieval helmet", limit=60, department=4)
        assert len(seen) <= 60, "бюджет карточек вырос — цена слота удвоилась"
        first = [int(u.rsplit("/", 1)[1]) for u in seen[:30]]
        assert first == list(range(1, 31)), (
            "ID каталога обязаны идти первыми: они прошли паспорт, а у выдачи "
            "API выживает около трети")

    def test_empty_catalogue_is_a_byte_for_byte_no_op(self, monkeypatch):
        """«longsword» в словаре Мет нет (там Sword / Two-hand sword), и
        каталог честно отдаёт ноль. Это обязано означать прежний путь, а не
        пустой слот."""
        class EmptyCat:
            @staticmethod
            def available():
                return True

            @staticmethod
            def search(q, department_name=None, limit=60):
                return []

        monkeypatch.setitem(sys.modules, "met_catalog", EmptyCat)
        monkeypatch.setattr(ms.feature_flags, "enabled", lambda *a, **k: True)
        monkeypatch.setattr(ms, "MET_DETAIL_WORKERS", 1)
        seen = []
        monkeypatch.setattr(ms, "_met_get", lambda url: (
            {"objectIDs": [7, 8, 9]} if "/search?" in url
            else seen.append(url) or None))
        ms.search_met("european longsword blade macro", limit=60, department=4)
        assert [int(u.rsplit("/", 1)[1]) for u in seen] == [7, 8, 9]


class TestCatalogueMechanics:
    def test_department_id_maps_to_the_name_used_in_the_dump(self):
        """У API отдел это число, в дампе — строка. Без таблицы локальный
        поиск фильтровал бы не тот отдел, что структурный запрос."""
        assert ms.MET_DEPARTMENT_NAMES[4] == "Arms and Armor"
        assert ms.MET_DEPARTMENT_NAMES[17] == "Medieval Art"

    def test_missing_index_is_not_an_error(self, monkeypatch):
        monkeypatch.setattr(mc, "INDEX_PATH", "/nonexistent/index.json")
        monkeypatch.setattr(mc, "_INDEX", None)
        assert mc.available() is False
        assert mc.search("medieval sword") == []

    def test_query_terms_drop_framing_words_not_the_subject(self):
        """«medieval european helmet visor macro» — предмет это helmet/visor,
        а не medieval/european/macro: по ним фильтрует паспорт и отдел, и в
        поле Object Name их нет."""
        t = mc._terms("medieval european helmet visor macro closeup")
        assert "helmet" in t and "visor" in t
        for junk in ("medieval", "european", "macro", "closeup"):
            assert junk not in t

    def test_index_version_guards_against_a_stale_build(self, monkeypatch, tmp_path):
        """Индекс, собранный старым кодом, не должен молча считаться свежим —
        тот же принцип, что у подписи отбора."""
        import json
        p = tmp_path / "index.json"
        p.write_text(json.dumps({"version": mc.CATALOG_VERSION - 1, "rows": []}))
        monkeypatch.setattr(mc, "INDEX_PATH", str(p))
        monkeypatch.setattr(mc, "_INDEX", None)
        assert mc.available() is False
