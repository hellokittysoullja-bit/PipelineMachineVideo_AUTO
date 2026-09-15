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
        shelf_index._corpus_rows(("нетакого",), (), None)


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
