"""Процедурная карточка — уровень видеоряда, который не может быть неверным.

Зачем она есть (разбор 07.09, измерено на опубликованном эпизоде): подбор
устроен как ГЕЙТ, а не фильтр. Провалил весь пул — побеждает «лучший из
плохих», слот всё равно заполняется. Кадр #13 (улица с туристом)
сегодняшний relevance-гейт ОТКЛОНЯЕТ, и он всё равно в ролике; для кадра #5
арбитр явно ответил «ни один не подходит», и он всё равно в ролике. Система
дважды знала правильный ответ и не имела чем его заменить.

Ключевое свойство, которое здесь и проверяется: надпись берётся из фразы
блока ДОСЛОВНО и никогда не сочиняется. Именно поэтому карточка физически не
может изображать чужую эпоху, другую культуру или не тот предмет — зритель
слышит ровно эти слова, а изображения на ней нет вообще.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import fallback_card as fc  # noqa: E402

FONT = os.path.join(REPO_ROOT, "assets", "fonts", "Benzin-ExtraBold.ttf")
HAS_FONT = os.path.exists(FONT)


class TestCardText:
    """Что выносится на карточку. Ошибка здесь — единственный способ сделать
    карточку неверной, поэтому проверяется подробнее самой отрисовки."""

    def test_number_with_unit_wins(self):
        """Вся суть канала в числах: «меч весил полтора кило, а не пятнадцать»."""
        text, kind = fc.choose_card_text(
            "А ближе к концу — меч на четыре килограмма. Настоящий, из музея.")
        assert kind == "number"
        assert text == "четыре килограмма"

    def test_digits_count_as_a_number_too(self):
        text, kind = fc.choose_card_text("Клинок до 170 см длиной, вес 2 кг.")
        assert kind == "number"
        assert text == "170 см"

    def test_inflected_spelled_out_numerals_are_a_known_limit(self):
        """Честно зафиксированный предел, а не скрытый баг.

        Русские числительные склоняются («ста семидесяти», «полутора»), и
        ловить их морфологию словарём форм — переобучение на паре примеров.
        Такая фраза не получает крупной цифры, но карточка всё равно ВЕРНА:
        срабатывает ветка фразы и даёт законченную мысль. Разница
        косметическая (кегль), не смысловая — поэтому и не чинится.
        """
        text, kind = fc.choose_card_text("Клинок до ста семидесяти см длиной.")
        assert kind == "phrase"
        assert text and text in "Клинок до ста семидесяти см длиной."

    def test_short_phrase_keeps_its_punctuation(self):
        """«Не дрались. Несли.» без точки — набор слов вместо фразы,
        ради ритма которой её и писали."""
        text, kind = fc.choose_card_text("Не дрались. Несли.")
        assert kind == "phrase"
        assert text == "Не дрались. Несли."

    def test_long_sentence_gives_a_complete_clause_not_a_truncation(self):
        """Реальный дефект первой версии: обрезка по счётчику слов давала
        «Историки до сих пор спорят, врёт эта» — оборванную мысль."""
        text, _ = fc.choose_card_text(
            "Историки до сих пор спорят, врёт эта табличка или нет.")
        assert text == "Историки до сих пор спорят"
        assert not text.endswith(("врёт эта", "или"))

    def test_multi_sentence_picks_a_short_complete_thought(self):
        text, _ = fc.choose_card_text(
            "Не вставай никуда. Просто вспомни, сколько весит пакет молока "
            "у тебя в руке. Литр. Ерунда.")
        assert text == "Не вставай никуда"

    def test_never_invents_words_outside_the_block_text(self):
        """Гарантия неошибочности целиком держится на этом свойстве."""
        source = ("Смотри, какая штука. Про мечи не знает ничего почти никто, "
                  "но мнение есть у всех.")
        text, _ = fc.choose_card_text(source)
        for word in text.replace(",", " ").replace(".", " ").split():
            assert word in source, f"слово «{word}» отсутствует в исходной фразе"

    def test_empty_input_is_not_a_crash(self):
        assert fc.choose_card_text("") == ("", "empty")
        assert fc.choose_card_text(None) == ("", "empty")

    def test_word_limit_is_respected(self):
        text, _ = fc.choose_card_text("раз два три четыре пять шесть семь восемь девять десять")
        assert len(text.split()) <= fc.CARD_MAX_WORDS


@pytest.mark.skipif(not HAS_FONT, reason="нужен шрифт канала")
class TestCardRender:
    def test_card_is_written_and_has_the_right_size(self, tmp_path):
        from PIL import Image
        out = str(tmp_path / "c.png")
        path, text = fc.build_fallback_card("Пятнадцать килограммов.", out, font_path=FONT)
        assert path == out and os.path.exists(out)
        with Image.open(out) as im:
            assert im.size == (fc.CARD_W, fc.CARD_H)
        assert text == "Пятнадцать килограммов"

    def test_text_is_vertically_centred(self, tmp_path):
        """Численно, а не на глаз (CLAUDE.md): у display-гарнитуры getbbox()
        отдаёт бокс со смещённым верхом, и наивная арифметика ставила блок
        заметно выше середины кадра при формально верном расчёте."""
        import numpy as np
        from PIL import Image
        out = str(tmp_path / "c.png")
        fc.build_fallback_card("Не дрались. Несли.", out, font_path=FONT)
        arr = np.array(Image.open(out).convert("L"), dtype=float)
        rows = np.where(arr.max(axis=1) > 200)[0]
        assert len(rows), "на карточке не нашлось светлого текста"
        centre = (rows.min() + rows.max()) / 2
        assert abs(centre - fc.CARD_H / 2) <= fc.CARD_H * 0.05, (
            f"текст не по центру: {centre:.0f} вместо {fc.CARD_H / 2:.0f}")

    def test_same_text_gives_a_byte_identical_card(self, tmp_path):
        """Детерминизм — не эстетика: кэш клипов ключуется по метаданным, и
        плавающая карточка перерендеривала бы слот каждый прогон."""
        a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
        fc.build_fallback_card("Пятнадцать килограммов.", a, font_path=FONT)
        fc.build_fallback_card("Пятнадцать килограммов.", b, font_path=FONT)
        assert open(a, "rb").read() == open(b, "rb").read()

    def test_different_texts_give_visibly_different_cards(self, tmp_path):
        """Соседние карточки не должны выглядеть одним шаблоном."""
        a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
        fc.build_fallback_card("Пятнадцать килограммов.", a, font_path=FONT)
        fc.build_fallback_card("Не дрались. Несли.", b, font_path=FONT)
        assert open(a, "rb").read() != open(b, "rb").read()

    def test_missing_font_is_an_honest_refusal_not_a_crash(self, tmp_path):
        path, text = fc.build_fallback_card(
            "Текст.", str(tmp_path / "c.png"), font_path="/nope/none.ttf")
        assert path is None
        assert text  # надпись всё равно посчитана — вызывающий код может её залогировать


class TestDensityBudget:
    """Бюджет плотности живёт В КОДЕ, а не в намерении: без него библиотека
    карточек превратилась бы ровно в то, на что была жалоба."""

    @pytest.fixture(autouse=True)
    def _ps(self):
        sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
        import pipeline_smart as ps
        ps.FALLBACK_CARD_SLOTS.clear()
        yield ps
        ps.FALLBACK_CARD_SLOTS.clear()

    def test_two_cards_in_a_row_are_forbidden(self, _ps):
        assert _ps.fallback_card_allowed(10, 165)
        _ps.FALLBACK_CARD_SLOTS.append({"index": 10, "reason": "x",
                                        "text": "", "card_text": ""})
        assert not _ps.fallback_card_allowed(11, 165)
        assert not _ps.fallback_card_allowed(12, 165)
        assert _ps.fallback_card_allowed(13, 165)

    def test_total_share_is_capped(self, _ps):
        cap = int(165 * _ps.FALLBACK_CARD_MAX_SHARE)
        for k in range(cap):
            _ps.FALLBACK_CARD_SLOTS.append({"index": k * 10, "reason": "x",
                                            "text": "", "card_text": ""})
        assert not _ps.fallback_card_allowed(999, 165)

    def test_flag_off_disables_the_whole_ladder(self, _ps, monkeypatch):
        monkeypatch.setattr(_ps, "FALLBACK_CARD_ENABLED", False)
        assert not _ps.fallback_card_allowed(10, 165)

    def test_known_bad_reason_reads_existing_records_only(self, _ps):
        """Функция не запускает новых проверок — читает то, что подбор уже
        записал в этом прогоне, и ранжирует причины по силе сигнала."""
        _ps.ARBITER_REJECTED_ALL.clear()
        _ps.STOCK_EXHAUSTED_MISSES.clear()
        _ps.RELEVANCE_GATE_MISSES.clear()
        try:
            assert _ps._slot_known_bad_reason(4) is None
            _ps.RELEVANCE_GATE_MISSES.append({"index": 4})
            assert _ps._slot_known_bad_reason(4) == "below_relevance_threshold"
            _ps.ARBITER_REJECTED_ALL.append({"index": 4})
            # Отказ арбитра сильнее численного промаха порога.
            assert _ps._slot_known_bad_reason(4) == "arbiter_rejected_all"
        finally:
            _ps.ARBITER_REJECTED_ALL.clear()
            _ps.STOCK_EXHAUSTED_MISSES.clear()
            _ps.RELEVANCE_GATE_MISSES.clear()

    def test_never_on_the_opening_shot(self, _ps):
        """Самый первый кадр ролика — единственное место, где карточка хуже
        даже посредственного живого кадра.

        У зрителя меньше секунды решить, остаться или уйти (CLAUDE.md
        ЧАСТЬ 9). Текстовая заставка на открытии — слабый вход. В остальных
        слотах хука карточка разрешена: там она заменяет уже признанный
        негодным кадр.
        """
        assert not _ps.fallback_card_allowed(0, 165, is_opening=True)
        assert _ps.fallback_card_allowed(0, 165, is_opening=False)


class TestBudgetSpreadOverEpisode:
    """Измеренный симптом (эпизод 02_ne-mechom, 10.09): все 17 карточек ушли
    на слоты 14..85 — первые 39% ролика, — а в оставшихся 61% осталось 43
    слота с уже записанным браком и НОЛЬ карточек. Брак при этом распределён
    почти равномерно (21/16/20/19 по четвертям), то есть дело было не в том,
    где брак, а в том, что бюджет выдавался первым пришедшим и кончался
    задолго до конца эпизода."""

    @pytest.fixture(autouse=True)
    def _ps(self):
        sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
        import pipeline_smart as ps
        ps.FALLBACK_CARD_SLOTS.clear()
        yield ps
        ps.FALLBACK_CARD_SLOTS.clear()

    def test_budget_cannot_be_spent_all_at_the_start(self, _ps):
        """Ранние слоты не могут забрать весь бюджет эпизода."""
        n = 219
        placed = 0
        for i in range(0, 90):
            if _ps.fallback_card_allowed(i, n):
                _ps.FALLBACK_CARD_SLOTS.append({"index": i, "reason": "x",
                                                "text": "", "card_text": ""})
                placed += 1
        total_cap = int(n * _ps.FALLBACK_CARD_MAX_SHARE)
        assert placed < total_cap, (
            "к 90-му слоту из 219 бюджет не должен быть исчерпан — иначе "
            "последние 60% ролика снова остаются без единой карточки")

    def test_late_slots_still_get_budget(self, _ps):
        """Слот во второй половине эпизода получает карточку, даже когда в
        первой половине брака было много."""
        n = 219
        for i in range(0, 110):
            if _ps.fallback_card_allowed(i, n):
                _ps.FALLBACK_CARD_SLOTS.append({"index": i, "reason": "x",
                                                "text": "", "card_text": ""})
        assert _ps.fallback_card_allowed(180, n)

    def test_total_cap_still_holds(self, _ps):
        """Равномерность не поднимает суммарный потолок: карточек по-прежнему
        не больше FALLBACK_CARD_MAX_SHARE от эпизода."""
        n = 219
        for i in range(n):
            if _ps.fallback_card_allowed(i, n):
                _ps.FALLBACK_CARD_SLOTS.append({"index": i, "reason": "x",
                                                "text": "", "card_text": ""})
        assert len(_ps.FALLBACK_CARD_SLOTS) <= int(n * _ps.FALLBACK_CARD_MAX_SHARE)

    def test_budget_version_is_part_of_the_signature(self):
        import os
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts", "pipeline_smart.py"),
            encoding="utf-8").read()
        start = src.index("def _selection_stack_signature")
        block = src[start:src.index("\ndef candidate_gate_signature", start)]
        assert "FALLBACK_CARD_BUDGET_VERSION" in block
