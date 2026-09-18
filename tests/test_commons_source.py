"""Викисклад как прямой источник: лицензия fail-closed, форма кандидата, кэш.

Сеть здесь не трогается вообще — _api() подменяется. Живой прогон делался
руками (см. докстринг модуля и CLAUDE.md), а тест обязан быть
воспроизводимым без сети и без лимитов чужого хоста.
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import commons_source as cs   # noqa: E402


def _page(pageid, title, license_name, w=2000, h=1500, restrictions="",
          categories=()):
    em = {}
    if license_name is not None:
        em["LicenseShortName"] = {"value": license_name}
    if restrictions:
        em["Restrictions"] = {"value": restrictions}
    return {"pageid": pageid, "title": title,
            "categories": [{"title": "Category:" + c} for c in categories],
            "imageinfo": [{"width": w, "height": h,
                            "descriptionurl": f"https://commons.wikimedia.org/wiki/{title}",
                            "extmetadata": em}]}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cs, "MIN_REQUEST_INTERVAL", 0.0)
    cs.reset_fetch_stats()


# ---------- лицензия ----------

@pytest.mark.parametrize("name", ["Public domain", "CC0", "PD", "No restrictions"])
def test_free_licenses_pass(name):
    assert cs.license_is_safe({"LicenseShortName": {"value": name}})


@pytest.mark.parametrize("name", [
    "CC BY 4.0", "CC BY-SA 4.0", "CC BY-SA 2.0", "CC BY-NC 3.0", "GFDL", "Fair use",
])
def test_attribution_licenses_are_refused(name):
    """Не потому что лицензия «плохая», а потому что у пайплайна нет
    механизма, который собирает атрибуции в описание ролика. Цена названа
    числом в докстринге модуля: 11 кандидатов из 33 в замере на кофе."""
    assert not cs.license_is_safe({"LicenseShortName": {"value": name}})


def test_substring_match_would_have_let_cc_by_sa_through():
    """Сравнение ТОЧНОЕ, а не «содержит»: «CC BY-SA 4.0» содержит «CC BY»,
    и наивная проверка пропустила бы ровно то, что запрещено."""
    assert "CC BY" in "CC BY-SA 4.0"
    assert not cs.license_is_safe({"LicenseShortName": {"value": "CC BY-SA 4.0"}})


def test_missing_license_is_refused():
    """«Не сказано» — это не «свободно». В живом замере 5 файлов из 172
    пришли вообще без поля лицензии."""
    assert not cs.license_is_safe({})
    assert not cs.license_is_safe({"LicenseShortName": {"value": ""}})
    assert not cs.license_is_safe(None)


def test_extra_restrictions_refuse_even_a_free_license():
    assert not cs.license_is_safe({"LicenseShortName": {"value": "Public domain"},
                                    "Restrictions": {"value": "trademarked"}})


# ---------- форма кандидата и гейты выдачи ----------

def _fake_api(pages):
    def _api(params, timeout=45, _retries=1):
        return {"query": {"pages": pages}}
    return _api


def test_candidate_has_the_pexels_shape(monkeypatch):
    """Форма та же, что у музеев и Openverse — иначе кандидат не сможет
    конкурировать в ОДНОМ пуле под ОДНИМИ гейтами."""
    monkeypatch.setattr(cs, "_api", _fake_api([
        _page(7, "File:Coffee_cherries_on_bush.jpg", "Public domain")]))
    (c,) = cs.search_commons("coffee")
    assert c["id"] == "commons:7"
    assert c["alt"] == "Coffee cherries on bush"      # текст для блоклиста
    assert c["src"]["large2x"].startswith("https://")
    assert c["src"]["medium"] != c["src"]["large2x"]  # превью для гейтов дешевле
    assert "width=640" in c["src"]["medium"]
    assert "width=2000" in c["src"]["large2x"]
    assert c["_download_headers"]["User-Agent"] == cs.USER_AGENT
    assert c["_commons_meta"]["license"] == "Public domain"


def test_small_images_are_refused(monkeypatch):
    """Кадр идёт в 1920x1080 и ещё проходит зум Ken Burns."""
    monkeypatch.setattr(cs, "_api", _fake_api([
        _page(1, "File:a.jpg", "Public domain", w=800, h=600)]))
    assert cs.search_commons("x") == []
    assert cs.FETCH_STATS["rejected_small"] == 1


def test_urls_go_through_the_official_endpoint_and_never_the_original(monkeypatch):
    """Оригинал на upload.wikimedia.org отвечает 429 с прямой просьбой брать
    thumbnail — это уже стоило проекту 49 сорванных скачек.

    И собирать путь к thumbnail подстановкой в URL тоже нельзя: первая
    версия этого модуля так и делала (/2000px- -> /640px-), и живой замер дал
    ОДНУ успешную скачку из десяти, остальные — HTTP 400 (проверено и с
    query-строкой, и без неё). Ширина запрашивается у источника явно.
    """
    monkeypatch.setattr(cs, "_api", _fake_api([
        _page(5, "File:Some name with spaces.jpg", "Public domain")]))
    (c,) = cs.search_commons("x")
    for url in (c["src"]["medium"], c["src"]["large2x"]):
        assert url.startswith("https://commons.wikimedia.org/wiki/Special:FilePath/")
        assert "upload.wikimedia.org" not in url
        assert "px-" not in url          # ни одной собранной подстановкой ширины
        assert "Some_name_with_spaces.jpg" in url


def test_license_rejection_is_counted_not_silent(monkeypatch):
    monkeypatch.setattr(cs, "_api", _fake_api([
        _page(3, "File:c.jpg", "CC BY-SA 4.0"),
        _page(4, "File:d.jpg", "Public domain")]))
    res = cs.search_commons("x")
    assert [r["id"] for r in res] == ["commons:4"]
    assert cs.FETCH_STATS["rejected_license"] == 1


def test_http_failure_is_fail_open_and_counted(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("сеть")
    monkeypatch.setattr(cs, "_api", boom)
    assert cs.search_commons("x") == []
    assert cs.FETCH_STATS["http_errors"] == 1


# ---------- кэш ----------

def test_empty_answer_is_not_cached(monkeypatch):
    """Временный отказ источника не имеет права замёрзнуть на месяц."""
    monkeypatch.setattr(cs, "_api", _fake_api([]))
    cs.search_commons("q")
    assert not os.path.exists(cs._disk_cache_path("q"))


def test_cache_key_contains_the_rules_version(monkeypatch):
    """Кэш хранит ГОТОВЫХ кандидатов: без версии правил ужесточение лицензии
    или порога разрешения молча отдавалось бы из кэша по старым правилам."""
    a = cs._disk_cache_path("q")
    monkeypatch.setattr(cs, "COMMONS_SOURCE_VERSION", cs.COMMONS_SOURCE_VERSION + 1)
    assert cs._disk_cache_path("q") != a
    monkeypatch.setattr(cs, "MIN_SHORT_SIDE", cs.MIN_SHORT_SIDE + 1)
    assert cs._disk_cache_path("q") != a


def test_cache_hit_does_not_call_the_api(monkeypatch):
    monkeypatch.setattr(cs, "_api", _fake_api([
        _page(9, "File:e.jpg", "Public domain")]))
    first = cs.search_commons("q")
    def boom(*a, **k):
        raise AssertionError("кэш-хит обязан обойтись без сети")
    monkeypatch.setattr(cs, "_api", boom)
    assert cs.search_commons("q") == first
    assert cs.FETCH_STATS["cache_hits"] == 1


# ---------- категории: текстовый канал о СОДЕРЖИМОМ ----------

def test_categories_join_the_candidate_text(monkeypatch):
    """Категории Викисклада проставляет человек — это единственное описание
    СОДЕРЖИМОГО в этом источнике, а не набора букв в имени файла.

    Важно не само по себе: filter_alt_blocklist()/pexels_candidate_text()
    судят кандидата ПО ТЕКСТУ, и до этого им доставалось только имя файла.
    Замер 18.09: категории есть у 15 файлов из 16 и приходят ТЕМ ЖЕ
    запросом — канал бесплатный.
    """
    monkeypatch.setattr(cs, "_api", _fake_api([
        _page(11, "File:Freshly_harvested_coffee_cherries.jpg", "Public domain",
              categories=["Coffea (fruit)", "Coffee production in Kenya"])]))
    (c,) = cs.search_commons("coffee")
    assert c["alt"] == ("Freshly harvested coffee cherries, Coffea (fruit), "
                        "Coffee production in Kenya")
    assert c["_commons_meta"]["categories"] == ["Coffea (fruit)",
                                                 "Coffee production in Kenya"]


def test_file_without_categories_still_has_its_name_as_text(monkeypatch):
    """Разреженность канала не имеет права обнулять кандидата: имя файла
    остаётся, поведение прежнее."""
    monkeypatch.setattr(cs, "_api", _fake_api([
        _page(12, "File:Yemeni_Coffee_Natural_Processing.jpg", "CC0")]))
    (c,) = cs.search_commons("coffee")
    assert c["alt"] == "Yemeni Coffee Natural Processing"
    assert c["_commons_meta"]["categories"] == []


def test_categories_come_in_the_same_request(monkeypatch):
    """Ни одного лишнего вызова к чужому API: без этого канал стоил бы по
    запросу на кандидата."""
    seen = {}
    def _api(params, timeout=45, _retries=1):
        seen.update(params)
        return {"query": {"pages": []}}
    monkeypatch.setattr(cs, "_api", _api)
    cs.search_commons("x")
    assert "categories" in seen["prop"]
