# -*- coding: utf-8 -*-
"""Europeana как второй корпус полки: инварианты, которые нельзя ломать молча.

Ни один тест здесь не ходит в сеть. Держим ровно то, из-за чего этот
репозиторий уже горел: fail-closed право, fail-closed источник, лексическую
ловушку года и единый ID предмета.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import europeana_corpus as ec  # noqa: E402
import shelf_index  # noqa: E402


# ------------------------------------------------------------------ права

@pytest.mark.parametrize("rights", [
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/zero/1.0",
    "http://creativecommons.org/publicdomain/mark/1.0/",
])
def test_public_domain_rights_are_accepted(rights):
    assert ec.is_safe_rights([rights]) is True


@pytest.mark.parametrize("rights", [
    "http://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/licenses/by-sa/4.0/",
    "http://rightsstatements.org/vocab/InC/1.0/",
    "", None, "что-то незнакомое",
])
def test_everything_else_is_refused(rights):
    """BY/BY-SA разрешают коммерческое использование, но требуют атрибуции в
    описании ролика, а сборщика атрибуций у пайплайна нет. Цена правила
    названа числом в докстринге модуля: 166 949 записей Portable
    Antiquities Scheme под CC BY в корпус не входят."""
    assert ec.is_safe_rights([rights] if rights is not None else None) is False


def test_rights_check_is_fail_closed_on_garbage():
    assert ec.is_safe_rights([]) is False
    assert ec.is_safe_rights(None) is False


# -------------------------------------------------------------- источники

def test_community_upload_source_is_not_trusted():
    """Ради ЭТОГО списка он и существует. В фасете 94 провайдера, и все,
    кроме одного, — учреждения; IMSLP единственный самотегирующийся."""
    assert ec.is_trusted_provider(["IMSLP/Petrucci Music Library"]) is False


def test_institutions_are_trusted():
    assert ec.is_trusted_provider(["Rijksmuseum"]) is True
    assert ec.is_trusted_provider(["KB, National Library of the Netherlands"]) is True


def test_unknown_provider_is_refused():
    assert ec.is_trusted_provider(["Частный блог о рыцарях"]) is False
    assert ec.is_trusted_provider([]) is False


# ------------------------------------------------------------------- годы

def test_year_clause_never_emits_the_lexical_trap():
    """Замерено на живом API 15.09: `YEAR:[900 TO 1600]` даёт РОВНО НОЛЬ,
    потому что поле строковое и «1372» лексически меньше «900». Тот же
    запрос с дополнением нулями — 38 262 записи. Немая пустая выдача
    оставила бы корпус выключенным, и увидеть это было бы негде."""
    clause = ec.year_clause(900, 1600)
    assert "YEAR:[900 TO 1600]" not in clause
    assert "YEAR:[0900 TO 1600]" in clause


def test_year_clause_keeps_the_unpadded_branch_below_1000():
    """Второй измеренный факт: часть записей хранит трёхзначный год БЕЗ
    дополнения (`[0900 TO 0999]` -> 0, `[900 TO 999]` -> 77). Без второй
    ветки они не нашлись бы никогда."""
    clause = ec.year_clause(900, 1600)
    assert "YEAR:[900 TO 999]" in clause
    assert clause.startswith("(") and " OR " in clause


def test_year_clause_is_a_single_term_when_the_window_starts_after_999():
    assert ec.year_clause(1000, 1600) == "YEAR:[1000 TO 1600]"


def test_record_without_a_year_is_refused_not_guessed():
    """Весь смысл корпуса в том, что дата ИЗВЕСТНА. Додумать её по
    заголовку значило бы вернуть ровно тот анахронизм, против которого
    паспорт и заведён."""
    assert ec.record_years({"year": []}) is None
    assert ec.record_years({"year": ["около 1400"]}) is None
    assert ec.record_years({"year": ["1372", "1400"]}) == (1372, 1400)


# ------------------------------------------------------------ строка корпуса

def _record(**over):
    rec = {
        "id": "/9200122/BibliographicResource_1000056125608",
        "title": ["Alexander the Great encounters the ten hermits"],
        "year": ["1372"],
        "rights": ["http://creativecommons.org/publicdomain/mark/1.0/"],
        "dataProvider": ["KB, National Library of the Netherlands"],
        "country": ["Netherlands"],
        "edmIsShownBy": ["http://resolver.kb.nl/resolve?urn=BYVANCKB:x:280v_min"],
        "dcTypeLangAware": {"def": ["Manuscript", "miniature"]},
    }
    rec.update(over)
    return rec


def test_a_real_shaped_record_passes_the_passport():
    row, why = ec._row(_record())
    assert why is None and row is not None
    assert row["id"] == "euro:/9200122/BibliographicResource_1000056125608"
    assert row["b"] == 1372 and row["e"] == 1372
    assert row["name"] == "Manuscript"
    assert row["source"] == "europeana"
    assert row["page"].startswith("https://www.europeana.eu/item/")
    assert "size=w400" in row["thumb"]


def test_foreign_culture_is_refused_by_text():
    """Поля `Culture` у Europeana нет вообще — это честно текстовая
    эвристика по тем же терминам, что у музейного паспорта, и так она и
    называется в докстринге модуля."""
    _row, why = ec._row(_record(
        title=["Japanese tsuba sword guard"],
        dcTypeLangAware={"def": ["Object"]}))
    assert why == "culture"


@pytest.mark.parametrize("over,why", [
    ({"rights": ["http://creativecommons.org/licenses/by/3.0/"]}, "rights"),
    ({"dataProvider": ["IMSLP/Petrucci Music Library"]}, "provider"),
    ({"year": []}, "no_year"),
    ({"year": ["1850"]}, "era"),
    ({"edmIsShownBy": []}, "no_image"),
])
def test_every_passport_gate_names_its_reason(over, why):
    """Отчёт сборки не имеет права быть немым: по какой оси запись не
    прошла — это ровно то, что нельзя восстановить постфактум."""
    row, got = ec._row(_record(**over))
    assert row is None and got == why


def test_culture_terms_are_not_copied_here():
    """Второй список чужих культур однажды уже разошёлся бы с первым.
    Термины берутся у museum_sources, а не дублируются."""
    src = open(os.path.join(REPO, "scripts", "europeana_corpus.py"),
               encoding="utf-8").read()
    assert "culture_is_foreign" in src
    assert "japanese" not in src.lower()


# ------------------------------------------------------- стык с полкой

def test_shelf_keeps_one_id_per_object():
    """Общий used_ids/дедуп обязан видеть один предмет как один, каким бы
    путём он ни нашёлся."""
    assert shelf_index._row_id({"id": 32684}) == "met:32684"
    assert shelf_index._row_id({"id": "euro:/9200122/X"}) == "euro:/9200122/X"


def test_a_row_with_its_own_image_never_calls_the_met_api(monkeypatch, tmp_path):
    """Строка Europeana принесла ссылки с собой. Лишний запрос к чужому
    API за тем, что уже известно, — это трата чужой квоты, и именно из-за
    неё Мет уже дважды отвечал 403 этому проекту."""
    called = []
    monkeypatch.setattr(shelf_index, "_image_url_for",
                        lambda oid: called.append(oid) or (None, None, None))
    monkeypatch.setattr(shelf_index, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(shelf_index, "ITEMS_PATH", str(tmp_path / "items.jsonl"))
    monkeypatch.setattr(shelf_index, "VECTORS_PATH", str(tmp_path / "vectors.f32"))
    monkeypatch.setattr(shelf_index, "IMAGES_DIR", str(tmp_path / "img"))
    monkeypatch.setattr(shelf_index, "_corpus_rows",
                        lambda *a, **k: [{"id": "euro:/1/x", "image": "http://invalid.test/a.jpg",
                                          "thumb": "http://invalid.test/a_s.jpg",
                                          "b": 1400, "e": 1450, "source": "europeana"}])
    shelf_index.build(corpus=("europeana",))
    assert called == []


def test_unknown_corpus_name_fails_loudly():
    with pytest.raises(SystemExit):
        shelf_index._corpus_rows(("нетакого",), ())


# ------------------------------------------------- контактный лист брифов

def test_brief_contact_sheet_refuses_loudly_without_a_shelf(monkeypatch, capsys, tmp_path):
    """Лист без полки печатал бы пустые плитки и выглядел бы как
    измерение, которым он не является. Тот же принцип, что уже стоит у
    `golden_set_eval.py`: нет ответа от модели — ни одной цифры."""
    import shelf_contact
    monkeypatch.setattr(shelf_contact, "__doc__", shelf_contact.__doc__)
    import shelf_index as si
    monkeypatch.setattr(si, "available", lambda: False)
    rc = shelf_contact.main([str(tmp_path)])
    assert rc == 1
    assert "НЕ собрана" in capsys.readouterr().out


def test_brief_contact_sheet_does_not_copy_the_text_wrapper():
    """Вторая реализация одного переноса рано или поздно разойдётся с
    первой, и один из двух листов станет нечитаемым незаметно."""
    src = open(os.path.join(REPO, "scripts", "shelf_contact.py"),
               encoding="utf-8").read()
    assert "from shotlist_contact import load_font, wrap_text" in src
    assert "def wrap_text" not in src
    assert "def load_font" not in src


# ------------------------------------------------- разрешение снимка

def test_small_images_are_not_harvested_by_default():
    """Замерено на скачанных файлах: `small` у KB это 750x500 альбомных,
    то есть 2.56x растяжения до 1920 ещё ДО зума Ken Burns. `medium` той
    же коллекции — 750x1094 портретных, которые `aspect_fit_backdrop()`
    вписывает по высоте почти один к одному. Разные случаи, и фасет
    Europeana их разделяет точно."""
    assert "small" not in ec.IMAGE_SIZE_PRIORITY
    assert "small" in ec.IMAGE_SIZE_EXCLUDED
    assert ec.IMAGE_SIZE_PRIORITY[0] == "extra_large"


def test_harvest_walks_big_images_first_inside_each_collection(monkeypatch):
    """Приоритет, а не фильтр: сборка резюмируемая и её обрывают на любой
    минуте, поэтому крупные снимки обязаны попасть в индекс раньше
    мелких — и внутри КАЖДОЙ коллекции, а не в среднем по корпусу."""
    asked = []

    def fake_search(params, timeout=60):
        asked.append([q for q in params["qf"]])
        return {"items": [], "nextCursor": None}

    monkeypatch.setattr(ec, "_search", fake_search)
    monkeypatch.setattr(ec.time, "sleep", lambda *_a: None)
    list(ec.harvest(collections=("A", "B"), sizes=("extra_large", "medium")))
    order = [(next(q.split('"')[1] for q in qf if q.startswith("europeana_collectionName")),
              next(q.split(":", 1)[1] for q in qf if q.startswith("IMAGE_SIZE")))
             for qf in asked]
    assert order == [("A", "extra_large"), ("A", "medium"),
                     ("B", "extra_large"), ("B", "medium")]


def test_harvested_row_records_the_resolution_bucket(monkeypatch):
    """Разрешение обязано доехать до строки индекса: по готовому ролику
    иначе не ответить, был ли кадр мягким из-за источника."""
    monkeypatch.setattr(ec, "_search",
                        lambda params, timeout=60: {"items": [_record()],
                                                    "nextCursor": None})
    monkeypatch.setattr(ec.time, "sleep", lambda *_a: None)
    rows = list(ec.harvest(collections=("A",), sizes=("large",)))
    assert rows and rows[0]["image_size"] == "large"


def test_contact_sheet_reports_answer_diversity(monkeypatch, tmp_path, capsys):
    """Молчание полки видно сразу, а повтор — нет: то, что она отвечает на
    РАЗНЫЕ описания ОДНИМ предметом, по плиткам заметно, только если
    разложить девять страниц рядом. Число обязано считаться само."""
    import json as _json
    import shelf_contact
    import shelf_index as si

    monkeypatch.setattr(si, "available", lambda: True)
    monkeypatch.setattr(si, "stats", lambda: {"items": 1462, "model": si.SHELF_MODEL,
                                              "available": True, "dim": 8})
    same = {"id": "met:22292", "score": 0.15, "name": "Breastplate",
            "title": "Breastplate", "dept": "Arms and Armor", "b": 1540, "e": 1540,
            "thumb": None, "image": None}
    cells = [{"index": i, "rank": 0, "section": "HOOK", "text": f"фраза {i}",
              "brief": f"brief {i}", "rec": dict(same)} for i in range(3)]
    cells[2]["rec"] = dict(same, id="met:99999", name="Sword")
    monkeypatch.setattr(shelf_contact, "collect",
                        lambda *a, **k: (cells, si.stats()))
    monkeypatch.setattr(shelf_contact, "render_page",
                        lambda cells, cols, out: out)

    rc = shelf_contact.main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Разных предметов в top-1: 2 на 3" in out
    assert "голод корпуса" in out
    rep = _json.load(open(tmp_path / "media_plan" / "shelf_contact.json",
                          encoding="utf-8"))
    assert rep["distinct_top1"] == 2 and rep["top1_answers"] == 3
    assert rep["most_repeated_top1"][1] == 2


# --------------------------------------- атрибуция источника в отчёте

def test_every_candidate_prefix_is_named_in_the_report():
    """Неизвестный префикс id молча падает в ветку «числовой id» и
    записывается как Pexels.

    Реальная цена, найденная сквозным прогоном 15.09: `euro:` в списке не
    было, и победа кандидата Europeana попала в `source_contribution.json`
    как победа Pexels — на машине, где ключа Pexels вообще нет. То есть
    отчёт называл источником кадра тот, который в прогоне не участвовал."""
    import pipeline_smart as ps

    assert ps.candidate_source({"id": "euro:/9200122/X"}) == "euro"
    assert ps.candidate_source({"id": "met:32684"}) == "met"
    assert ps.candidate_source({"id": "33508363"}) == "pexels"


def test_shelf_candidate_ids_are_covered_by_the_prefix_table():
    """Список префиксов обязан покрывать КАЖДЫЙ префикс, который реально
    выдаёт хоть один сборщик кандидатов. Проверяем не список против списка,
    а против того, что модули действительно строят."""
    import pipeline_smart as ps

    built = {"met:32684", "euro:/9200122/X", "openverse:uuid", "pixabay:12",
             "unsplash:abc", "chicago:1", "cleveland:2"}
    for cid in built:
        prefix = cid.split(":", 1)[0]
        assert ps.candidate_source({"id": cid}) == prefix, cid
        assert prefix in ps.CANDIDATE_ID_PREFIXES


def test_candidate_id_with_slashes_never_becomes_a_path(tmp_path):
    """Дефект, из-за которого КАЖДЫЙ кандидат Europeana терялся до гейтов.

    Имя файла-пробника собиралось из id буквально, а id записи Europeana —
    `euro:/9200122/...` со СЛЭШАМИ: путь уезжал в несуществующий каталог,
    запись падала FileNotFoundError, и relevance-гейт, контрастивное вето,
    домен-гвард, дедуп и ранжирование не отрабатывали по этим кандидатам
    ВООБЩЕ — победитель брался по позиции в списке."""
    import pipeline_smart as ps

    cf = str(tmp_path / "0000_slot.jpg")
    for cid in ("euro:/9200122/BibliographicResource_1000056125434",
                "met:32684", "openverse:a-b-c", "33508363"):
        trial = cf + f".trial_{ps.candidate_path_token({'id': cid})}.jpg"
        assert os.path.dirname(trial) == str(tmp_path), cid
        open(trial, "wb").write(b"x")          # запись обязана состояться
        assert os.path.exists(trial)


def test_path_token_keeps_candidates_distinct():
    """Санитизация не имеет права склеить двух РАЗНЫХ кандидатов в один
    файл — иначе пул молча потерял бы половину выборки."""
    import pipeline_smart as ps

    ids = ["euro:/9200122/A", "euro:/9200122/B", "met:1", "met:2"]
    tokens = [ps.candidate_path_token({"id": i}) for i in ids]
    assert len(set(tokens)) == len(ids)


def test_collection_order_can_be_overridden_at_call_time(monkeypatch):
    """`collections=COLLECTION_PRIORITY` в СИГНАТУРЕ запоминал список на
    момент импорта: подмена `ec.COLLECTION_PRIORITY` снаружи молча не
    действовала, сборка шла по старому порядку и выглядела рабочей.

    Поймано собственным замером 15.09 — опыт «собрать только Альбертину»
    вернул рукописи KB и отчитался «осталось 0». Тот же принцип «читать в
    момент вызова», которым в этом репозитории уже закрыт реестр флагов.
    """
    asked = []
    monkeypatch.setattr(ec, "_search",
                        lambda params, timeout=60: asked.append(params["qf"]) or
                        {"items": [], "nextCursor": None})
    monkeypatch.setattr(ec.time, "sleep", lambda *_a: None)
    monkeypatch.setattr(ec, "COLLECTION_PRIORITY", ("ТОЛЬКО_ЭТА",))
    list(ec.harvest(sizes=("large",)))
    names = [q.split('"')[1] for qf in asked for q in qf
             if q.startswith("europeana_collectionName")]
    assert names == ["ТОЛЬКО_ЭТА"], names


# ---------------------------------------- устойчивость самой выгрузки
#
# Всё, что ниже, найдено ПРОГОНОМ по собственному коду 15.09, а не чтением:
# каждый случай сначала воспроизведён на подставном `_search`, и только
# потом исправлен. Контрольный прогон со снятой правкой роняет
# соответствующий тест — иначе тест не проверяет ничего.

def _fake_pages(n_pages=4, per_page=5, prefix="x"):
    """Подставной поиск: n страниц по per_page годных записей."""
    state = {"n": 0}

    def _search(params, timeout=60):
        state["n"] += 1
        items = [{
            "id": f"/900/{prefix}{state['n']}_{i}",
            "rights": [ec.CC0],
            "dataProvider": ["Rijksmuseum"],
            "year": ["1372"],
            "edmIsShownBy": [f"http://img/{state['n']}_{i}.jpg"],
            "title": ["T"],
            "dcTypeLangAware": {"en": ["Manuscript"]},
        } for i in range(per_page)]
        return {"items": items,
                "nextCursor": f"c{state['n']}" if state["n"] < n_pages else None}
    return _search, state


@pytest.fixture
def no_page_pause(monkeypatch):
    monkeypatch.setattr(ec, "PAGE_PAUSE_SEC", 0)
    monkeypatch.setattr(ec, "SEARCH_RETRY_PAUSE_SEC", 0)


def test_harvest_stats_are_readable_before_the_generator_is_exhausted(
        monkeypatch, no_page_pause):
    """`harvest.stats = stats` в КОНЦЕ генератора выполняется только при
    полном исчерпании. Любой потребитель с ранним break (islice, свой
    лимит, ошибка выше по стеку) читал бы пустой словарь из прошлой жизни:
    проверено прогоном — `harvest.stats['kept']` роняло KeyError."""
    search, _ = _fake_pages()
    monkeypatch.setattr(ec, "_search", search)
    monkeypatch.setattr(ec.harvest, "stats", {})
    gen = ec.harvest(collections=["A"], sizes=["large"])
    [next(gen) for _ in range(3)]
    assert ec.harvest.stats.get("kept") == 3
    assert ec.harvest.stats.get("seen") == 3


def test_three_digit_years_are_not_fetched_and_then_thrown_away():
    """`year_clause` НАМЕРЕННО добавляет ветку без дополнения нулями — там
    77 записей окна, у которых год хранится трёхзначным. Принимать в
    `record_years` только четыре цифры значило запрашивать их страницами и
    выбрасывать каждую как «нет года»: ветка запроса была бы no-op, и
    молча. Оба конца правила держатся ОДНИМ тестом, чтобы их нельзя было
    рассогласовать по отдельности."""
    assert "YEAR:[900 TO 999]" in ec.year_clause(900, 1600)
    assert ec.record_years({"year": ["950"]}) == (950, 950)
    assert ec.record_years({"year": ["1372"]}) == (1372, 1372)
    # Мусор по-прежнему не год: паспорт остаётся fail-closed.
    assert ec.record_years({"year": ["12"]}) is None
    assert ec.record_years({"year": ["14xx"]}) is None
    assert ec.record_years({"year": []}) is None


def test_a_record_in_two_collections_is_yielded_once(monkeypatch, no_page_pause):
    """Одна запись может лежать в ДВУХ коллекциях приоритета. Дубль стоит
    второго вектора (2.28с эмбеддинга), второй строки индекса и двух
    кандидатов ОДНОГО предмета в пуле одного слота."""
    def one(params, timeout=60):
        return {"items": [{"id": "/9/same", "rights": [ec.CC0],
                           "dataProvider": ["Rijksmuseum"], "year": ["1372"],
                           "edmIsShownBy": ["http://i.jpg"], "title": ["T"]}],
                "nextCursor": None}
    monkeypatch.setattr(ec, "_search", one)
    rows = list(ec.harvest(collections=["A", "B"], sizes=["large"]))
    assert [r["id"] for r in rows] == ["euro:/9/same"]
    assert ec.harvest.stats["rejected"].get("duplicate") == 1


def test_one_transient_failure_does_not_cost_the_rest_of_the_collection(
        monkeypatch, no_page_pause):
    """Сборка идёт часами. Разовый сетевой сбой обрывал коллекцию молча и
    на середине — `cursor` после этого восстановить уже нечем."""
    calls = {"n": 0}

    def flaky(params, timeout=60):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return {"items": [{"id": "/9/ok", "rights": [ec.CC0],
                           "dataProvider": ["Rijksmuseum"], "year": ["1372"],
                           "edmIsShownBy": ["http://i.jpg"], "title": ["T"]}],
                "nextCursor": None}
    monkeypatch.setattr(ec, "_search", flaky)
    rows = list(ec.harvest(collections=["A"], sizes=["large"]))
    assert len(rows) == 1 and calls["n"] == 2


def test_a_search_that_keeps_failing_is_still_fail_open(monkeypatch, no_page_pause):
    """Повтор — не бесконечность: упавшее учреждение отдаёт ноль строк и не
    уносит с собой ни остальные коллекции, ни сборку."""
    calls = {"n": 0}

    def dead(params, timeout=60):
        calls["n"] += 1
        raise OSError("down")
    monkeypatch.setattr(ec, "_search", dead)
    assert list(ec.harvest(collections=["A"], sizes=["large"])) == []
    assert calls["n"] == ec.SEARCH_ATTEMPTS
    assert ec.harvest.stats["errors"] == 1


def test_a_cursor_that_repeats_itself_terminates(monkeypatch, no_page_pause):
    """Курсор, равный текущему, — это бесконечный цикл, а не следующая
    страница. Своей выдачей Europeana такого не давала; цикл без потолка
    пишется один раз и живёт годами."""
    calls = {"n": 0}

    def stuck(params, timeout=60):
        calls["n"] += 1
        return {"items": [{"id": f"/9/i{calls['n']}", "rights": [ec.CC0],
                           "dataProvider": ["Rijksmuseum"], "year": ["1372"],
                           "edmIsShownBy": ["http://i.jpg"], "title": ["T"]}],
                "nextCursor": "SAME"}
    monkeypatch.setattr(ec, "_search", stuck)
    rows = list(ec.harvest(collections=["A"], sizes=["large"]))
    assert calls["n"] == 2 and len(rows) == 2


# ------------------------------------------------- лимит и ленивость сборки

def _stub_index_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(shelf_index, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(shelf_index, "ITEMS_PATH", str(tmp_path / "items.jsonl"))
    monkeypatch.setattr(shelf_index, "VECTORS_PATH", str(tmp_path / "v.f32"))
    monkeypatch.setattr(shelf_index, "IMAGES_DIR", str(tmp_path / "img"))


def test_limit_counts_new_items_not_rows_looked_at(monkeypatch, tmp_path):
    """`--limit` — сколько НОВЫХ предметов добавить, а не сколько строк
    посмотреть. При старом счёте повторный запуск той же команды на полке,
    где лимит уже набран, печатал «осталось 0» и останавливался, хотя в
    корпусе оставались десятки тысяч записей: выглядело как «полка
    собрана»."""
    rows = [{"id": f"euro:/1/{i}", "image": f"http://x/{i}.jpg",
             "thumb": f"http://x/{i}_s.jpg", "b": 1400, "e": 1450,
             "source": "europeana"} for i in range(10)]
    monkeypatch.setattr(shelf_index, "_corpus_rows", lambda *a, **k: iter(rows))
    monkeypatch.setattr(shelf_index, "_read_items",
                        lambda: [{"id": "euro:/1/0"}, {"id": "euro:/1/1"}])
    _stub_index_paths(monkeypatch, tmp_path)

    tried = []

    def fake_urlopen(req, timeout=None):
        tried.append(getattr(req, "full_url", str(req)))
        raise OSError("сеть в тестах не нужна")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    shelf_index.build(corpus=("europeana",), limit=3)
    # Именно ТРИ ПЕРВЫЕ НОВЫЕ строки (2, 3, 4), а не «первые три строки
    # корпуса», две из которых уже посчитаны.
    assert tried == ["http://x/2_s.jpg", "http://x/3_s.jpg", "http://x/4_s.jpg"]


def test_the_corpus_is_read_lazily_and_stops_where_the_build_stops(
        monkeypatch, tmp_path):
    """Выдача корпусов ленивая: лимит прогона не имеет права стоить полного
    прохода по чужому API за строками, которые тут же выбрасываются.
    Проверяется не намерением, а тем, сколько строк реально прочитано."""
    read = []

    def endless():
        for i in range(10_000):
            read.append(i)
            yield {"id": f"euro:/1/{i}", "image": f"http://x/{i}.jpg",
                   "thumb": f"http://x/{i}_s.jpg", "b": 1400, "e": 1450,
                   "source": "europeana"}
    monkeypatch.setattr(shelf_index, "_corpus_rows", lambda *a, **k: endless())
    monkeypatch.setattr(shelf_index, "_read_items", lambda: [])
    _stub_index_paths(monkeypatch, tmp_path)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("нет сети")))
    shelf_index.build(corpus=("europeana",), limit=5)
    assert len(read) == 5


def test_an_already_indexed_item_is_never_embedded_twice(monkeypatch, tmp_path):
    """`done` снимается ОДИН раз до цикла, поэтому дубль внутри одной
    выдачи он не видит. Второй вектор одного предмета — это не только
    потраченные 2.28с, но и два кандидата ОДНОГО предмета в пуле слота."""
    rows = [{"id": "euro:/1/a", "thumb": "http://x/a.jpg", "image": "http://x/a.jpg",
             "b": 1400, "e": 1450, "source": "europeana"}] * 3
    monkeypatch.setattr(shelf_index, "_corpus_rows", lambda *a, **k: iter(rows))
    monkeypatch.setattr(shelf_index, "_read_items", lambda: [])
    _stub_index_paths(monkeypatch, tmp_path)
    tried = []
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: tried.append(1) or
                        (_ for _ in ()).throw(OSError("нет сети")))
    shelf_index.build(corpus=("europeana",))
    assert len(tried) == 1


def test_an_unknown_corpus_name_still_fails_at_call_time(monkeypatch):
    """У генератора тело не выполняется до первого `next()`. Неизвестное имя
    корпуса обязано всплыть в момент запуска, а не посреди сборки — и не
    исчезнуть вовсе, если выдачу никто не дочитал."""
    with pytest.raises(SystemExit):
        shelf_index._corpus_rows(("нетакого",), ())


# ------------------------------------------------------ кто принёс кадр

def test_the_shelf_is_told_apart_from_the_museum_api_in_the_report():
    """id кандидата полки обязан оставаться `met:<objectID>` — общий
    `used_ids` должен видеть один предмет как один. Именно поэтому «кто
    принёс кадр» считается по `_shelf_meta`, а не по префиксу id: иначе
    вклад полки неотличим от вклада музейного API, и на единственный
    вопрос, ради которого полка строилась, по отчёту не ответить."""
    import pipeline_smart as ps
    museum = {"id": "met:32684"}
    shelf = {"id": "met:32684", "_shelf_meta": {"score": 0.3, "corpus": "met"}}
    euro = {"id": "euro:/9200122/X", "_shelf_meta": {"corpus": "europeana"}}
    assert ps.candidate_source(shelf) == "met"        # id-пространство не тронуто
    assert ps.candidate_channel(museum) == "met"
    assert ps.candidate_channel(shelf) == "shelf"
    assert ps.candidate_channel(euro) == "shelf"
    assert ps.candidate_channel({"id": "33508363"}) == "pexels"


def test_every_contribution_counter_goes_through_the_channel():
    """Счётчики вклада обязаны считать КАНАЛ, а не пространство id — иначе
    правка выше молча перестала бы действовать на половину счётчиков."""
    import re as _re
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    assert not _re.search(r"_source_bump\(candidate_source\(", src)
    assert _re.search(r"_source_bump\(candidate_channel\(", src)


def test_the_shelf_build_and_the_selection_sanitize_ids_the_same_way():
    """Второй, ослабленной копии правила «id -> имя файла» быть не должно:
    id со слэшами уже стоил проекту КАЖДОГО кандидата Europeana,
    потерянного до гейтов, а двоеточие в имени файла запрещено на Windows.
    Сегодня обе формы совпадают побайтово — уже скачанные превью не
    осиротели; расходиться им запрещено впредь."""
    import pipeline_smart as ps
    for rid in ("euro:/9200122/BibliographicResource_1000056125434",
                "met:32684", "openverse:e8b7760f-aee4", "12345",
                # Формы, на которых наивная замена ":" и "/" расходится с
                # общим правилом: пробел и "?" в имени файла Windows тоже
                # не прощает, а id приходит из чужого API.
                "euro:/9/a b?c", "euro:/9/x*y"):
        assert shelf_index._path_token(rid) == ps.candidate_path_token(rid)
    # И то, ради чего правило вообще существует.
    assert "/" not in shelf_index._path_token("euro:/9/x")
    assert ":" not in shelf_index._path_token("euro:/9/x")


# ------------------------------------------- след происхождения у полки

def test_a_shelf_winner_leaves_a_provenance_record():
    """Победивший кадр ПОЛКИ уходил в ролик без единой строки о
    происхождении: `candidate_provenance()` смотрела только `_museum_meta`
    и `_openverse_meta`, а кандидат полки несёт `_shelf_meta`. Ни записи в
    `source_license_manifest.jsonl`, ни ссылки на предмет в шотлисте —
    притом что у второго корпуса полки основание публикации самое слабое
    из всех (PDM — пометка «ограничений не известно», а не отказ от прав).
    Проверено прогоном до правки: возвращался None."""
    import pipeline_smart as ps
    cand = {"id": "euro:/9200122/X", "alt": "Miniature",
            "url": "https://www.europeana.eu/item/9200122/X",
            "_shelf_meta": {"score": 0.31, "dept": "KB", "begin": 1372, "end": 1372,
                            "culture": None, "license": "public_domain",
                            "license_field": ec.PDM, "corpus": "europeana",
                            "provider": "KB, National Library of the Netherlands"}}
    prov = ps.candidate_provenance(cand)
    assert prov is not None
    assert prov["id"] == "euro:/9200122/X"
    assert prov["page"].endswith("/9200122/X")
    assert prov["license_field"] == ec.PDM
    assert prov["provider"].startswith("KB")
    assert prov["corpus"] == "europeana"
    # Канал, а не пространство id: без него запись не отвечает на вопрос,
    # каким путём кадр найден.
    assert prov["channel"] == "shelf"
    # Скор поиска — не происхождение: в юридическом журнале ему не место.
    assert "score" not in prov


def test_the_museum_path_provenance_did_not_change_shape():
    """Правка обязана только ДОБАВЛЯТЬ: у уже работавшего пути состав полей
    прежний, плюс канал."""
    import pipeline_smart as ps
    prov = ps.candidate_provenance({"id": "met:32684", "alt": "Dagger",
                                    "url": "https://metmuseum.org/x",
                                    "_museum_meta": {"culture": "French",
                                                     "license": "public_domain"}})
    assert prov["culture"] == "French"
    assert prov["license"] == "public_domain"
    assert prov["channel"] == "met"
    assert ps.candidate_provenance({"id": "12345"}) is None


def test_the_contact_sheet_uses_the_same_path_rule():
    """Третья копия «заменить ':' и '/'» жила в контактном листе брифов и
    молча разошлась бы с остальными на первом же id с пробелом."""
    import shelf_contact  # noqa: F401
    src = open(os.path.join(REPO, "scripts", "shelf_contact.py"),
               encoding="utf-8").read()
    assert '.replace(":", "_").replace("/", "_")' not in src
    assert "_path_token(" in src
