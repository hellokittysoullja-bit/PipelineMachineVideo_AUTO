"""Текст кандидата переживает кэш-хит — иначе текстовую ось нельзя измерить.

Зачем этот файл. 18.09 измерение показало, что усиление модели-СУДЬИ на оси
сходства не работает (docs/quality/so400m_as_judge.json), а 65% брака — это
анахронизм и чужая культура, то есть вопрос ЗНАНИЯ. Ближайший канал знания,
который у пайплайна уже есть, — ТЕКСТ кандидата: alt, человекочитаемый слаг
url, теги, а у Викисклада имя файла и категории. Именно по нему работает
жанровый фильтр, отсекавший 27.7% живой выдачи.

Проверить эту ось на глазной разметке оказалось НЕЧЕМ: у золотого набора
(40 кадров, единственный размеченный корпус проекта) текста кандидатов нет,
эпизод свой temp_smart не сохранил, а по имени кэш-файла текст не
восстанавливается. То есть ось неаудируема постфактум по построению.

Отсюда правка: текст пишется в sidecar рядом с файлом и живёт ровно столько
же, сколько сам файл, — как провенанс и по той же причине (на кэш-хите
подбор не вызывается вообще, и другого места, где текст ещё известен, нет).
"""
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402


def test_sidecar_keeps_the_text_the_genre_filter_judges(tmp_path):
    media = tmp_path / "0000_a_b.jpg"
    media.write_bytes(b"x")
    cand = {"id": 123, "alt": "roman soldiers reenactment",
            "url": "https://www.pexels.com/video/roman-soldiers-historical-reenactment-event-38103939/"}
    ps.write_media_sidecar(str(media), pexels_id=123, query="medieval knight",
                           kind="photo",
                           candidate_text=ps.pexels_candidate_text(cand))
    data = json.load(open(ps.media_sidecar_path(str(media)), encoding="utf-8"))
    text = data["candidate_text"]
    # Ровно та строка, по которой судит filter_alt_blocklist — не «что-то
    # похожее»: разойдись они, отчёт объяснял бы вердикт не тем текстом.
    assert text == ps.pexels_candidate_text(cand)
    assert "reenactment" in text          # слаг url, а не только alt
    assert "roman soldiers" in text


def test_absent_text_does_not_appear_as_a_null(tmp_path):
    """Кадр, скачанный до этой правки, не должен выглядеть как «текста нет»
    наравне с кадром, у которого текста и правда не было."""
    media = tmp_path / "0001_a_b.jpg"
    media.write_bytes(b"x")
    ps.write_media_sidecar(str(media), pexels_id=1, query="q", kind="photo")
    data = json.load(open(ps.media_sidecar_path(str(media)), encoding="utf-8"))
    assert "candidate_text" not in data


def test_every_sidecar_call_in_the_selection_path_passes_the_text():
    """Source-level: без проброса правка — мёртвый слой.

    Ровно тот класс, которым этот репозиторий горел шесть раз: функция
    принимает параметр, и никто его не передаёт.
    """
    import re
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    # Именно ВЫЗОВЫ, а не упоминания в докстрингах: у этой функции её имя
    # встречается в комментариях чаще, чем в коде, и счёт по подстроке
    # проверял бы не то (тест, зелёный или красный по неверной причине,
    # хуже отсутствующего).
    calls = [m.start() for m in re.finditer(r"^\s+write_media_sidecar\(", src,
                                            re.M)]
    assert len(calls) == 3, f"точек записи sidecar стало {len(calls)}"
    for pos in calls:
        block = src[pos:src.index(")\n", pos) + 1]
        assert "candidate_text=" in block, (
            "точка записи sidecar не передаёт текст кандидата:\n" + block)
