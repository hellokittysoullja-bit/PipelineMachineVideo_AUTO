# -*- coding: utf-8 -*-
"""Ключ кэша называет то, чем РЕАЛЬНО спрашивали, и полка спрашивает фразой.

Три правки одного захода, каждая — ответ на измеренную находку, а не на
рассуждение.

1. НАЙДЕННЫЙ БАГ (15.09). Ключ кэша кандидата считался как
   `brief_to_stock_query(shot_brief)` — то есть ПЕРЕВОД брифа на язык стоков
   (якорь эпохи + пять слов). А в полку уходит бриф ЦЕЛИКОМ, и для сравнения
   с изображениями «close up» и «seen from the front» — разные вопросы.
   Прогон трёх реалистичных пар, отличающихся только ракурсом, дал ТРИ
   СОВПАДЕНИЯ ИЗ ТРЁХ:

       breastplate, close up            -> medieval dented steel breastplate
       breastplate, seen from the front -> medieval dented steel breastplate

   Следствие: на прогретом temp_smart/ слот молча отдаёт кандидата,
   выбранного под ДРУГОЙ вопрос к полке.

2. Полка без брифа получала ЗАПРОС СЕКЦИИ (2-5 английских слов), общий на
   6-10 слотов, — при том что рядом лежит фраза этого блока, а полка
   сравнивает эмбеддинги, а не слова.

   Честно про границу: фраза НЕ заменяет бриф. Бриф переводит «что сказано»
   в «что показать», и на отрицании/абстракции фраза упирается в потолок
   класса моделей (часть B бенчмарка репозитория: 12-38% top-1 у всех трёх).
   Фраза заменяет ЗАПРОС СЕКЦИИ, а не бриф автора.

   Вход выбран замером, а не на глаз: реальный токенизатор so400m по 142
   блокам эпизода 02 — b["text"] превышает лимит 64 токена у 14 блоков
   (10%), semantic_context_text у 20 (14%). Обрезка молчаливая.

3. Гвард полки сверял только ИМЯ модели. Дрейф ML-стека в этом репозитории
   уже задокументирован и уже сдвигал кадр золотого набора через порог без
   правок кода; полка собирается часами и живёт месяцами, то есть она самый
   уязвимый к этому объект проекта.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402
import shelf_index as si  # noqa: E402


# Пары из ЖИВОГО замера: отличаются только ракурсом, стоковый перевод общий.
COLLIDING = [
    ("a dented steel breastplate, close up",
     "a dented steel breastplate, seen from the front"),
    ("a two-handed sword, whole blade",
     "a two-handed sword, blade in the background"),
    ("manuscript illumination of a battle between armoured knights",
     "manuscript illumination of a battle between armoured knights, wide shot"),
]


class TestBriefKeyNamesTheRealQuestion:
    def test_the_collision_is_real_in_the_stock_translation(self):
        """Сначала фиксируем САМ дефект: перевод у пар действительно общий.
        Если это когда-нибудь перестанет быть правдой, тесты ниже потеряют
        смысл и должны об этом сказать, а не молча позеленеть."""
        for a, b in COLLIDING:
            assert (ps.brief_to_stock_query(a, fallback=None)
                    == ps.brief_to_stock_query(b, fallback=None)), (a, b)

    @pytest.mark.parametrize("a,b", COLLIDING)
    def test_different_briefs_get_different_keys_when_shelf_answers(self, monkeypatch, a, b):
        monkeypatch.setattr(ps, "_shelf_question_active", lambda: True)
        assert ps.candidate_brief_key(a) != ps.candidate_brief_key(b)

    @pytest.mark.parametrize("a,b", COLLIDING)
    def test_without_a_shelf_the_key_is_exactly_as_before(self, monkeypatch, a, b):
        """Владелец без собранной полки не платит перерендером за изменение,
        которого у него физически не происходит."""
        monkeypatch.setattr(ps, "_shelf_question_active", lambda: False)
        for brief in (a, b):
            assert ps.candidate_brief_key(brief) == ps.brief_to_stock_query(
                brief, fallback=None)

    def test_stock_query_is_still_in_the_key(self, monkeypatch):
        """Стоковый запрос — не хэш, а сам текст: он и есть реальный запрос,
        и по имени файла кэша должно быть видно, чем спрашивали сток."""
        monkeypatch.setattr(ps, "_shelf_question_active", lambda: True)
        key = ps.candidate_brief_key("a dented steel breastplate, close up")
        assert key.startswith(ps.brief_to_stock_query(
            "a dented steel breastplate, close up", fallback=None))

    def test_no_brief_no_text_is_empty_key(self, monkeypatch):
        monkeypatch.setattr(ps, "_shelf_question_active", lambda: True)
        assert ps.candidate_brief_key(None) == ""
        assert ps.candidate_brief_key("", block_text="") == ""

    def test_phrase_enters_the_key_only_where_the_shelf_is_asked(self, monkeypatch):
        """Видео-путь полку не спрашивает вообще (единственный вызов
        _shelf_search_photos живёт в pexels_photo) — значит смена фразы не
        имеет права перекачивать видео-кандидатов."""
        monkeypatch.setattr(ps, "_shelf_question_active", lambda: True)
        photo = ps.candidate_brief_key(None, block_text="Стрела скользнула по нагруднику.")
        video = ps.candidate_brief_key(None, block_text="Стрела скользнула по нагруднику.",
                                       uses_shelf=False)
        assert photo != ""
        assert video == ""

    def test_different_phrases_give_different_keys(self, monkeypatch):
        monkeypatch.setattr(ps, "_shelf_question_active", lambda: True)
        assert (ps.candidate_brief_key(None, block_text="Стрела скользнула по нагруднику.")
                != ps.candidate_brief_key(None, block_text="Возьми настоящий боевой меч."))

    def test_author_brief_wins_over_the_phrase(self, monkeypatch):
        """Фраза — запасной вопрос, а не замена: там где автор написал бриф,
        спрашивают брифом."""
        monkeypatch.setattr(ps, "_shelf_question_active", lambda: True)
        with_brief = ps.candidate_brief_key("a dented steel breastplate, close up",
                                            block_text="совсем другая фраза")
        brief_only = ps.candidate_brief_key("a dented steel breastplate, close up")
        assert with_brief == brief_only


class TestBothCallSitesUseTheResolver:
    """Негативный контроль: вернуть любой из двух вызовов к старой формуле —
    и тест падает. Две копии одного правила уже стоили этому репозиторию
    PHRASE LOCK на целый эпизод."""

    def _src(self):
        return open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                    encoding="utf-8").read()

    def test_no_raw_stock_translation_left_in_any_key(self):
        """Проверка по AST, а не по тексту файла: старая формула ЦИТИРУЕТСЯ в
        докстринге резолвера как история находки, и текстовый поиск нашёл бы
        именно её. Тот же промах я уже допускал дважды в этой сессии —
        якорь обязан быть кодом, а не строкой, встречающейся где угодно."""
        import ast
        tree = ast.parse(self._src())
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_brief_key" not in names:
                continue
            fn = node.value.func if isinstance(node.value, ast.Call) else None
            calls.append(getattr(fn, "id", None) or getattr(fn, "attr", None))
        assert calls, "присваивания _brief_key не найдены вообще"
        assert set(calls) == {"candidate_brief_key"}, calls

    def test_photo_path_passes_the_phrase(self):
        src = self._src()
        assert "candidate_brief_key(shot_brief, block_text)" in src

    def test_video_path_declares_it_does_not_use_the_shelf(self):
        src = self._src()
        assert "candidate_brief_key(shot_brief, uses_shelf=False)" in src

    def test_main_feeds_the_block_phrase_to_photo_only(self):
        src = self._src()
        assert src.count('block_text=b["text"]') == 2


class TestShelfIsAskedWithThePhrase:
    def test_fallback_order_is_brief_then_phrase_then_nothing(self):
        """Функционально, а не по тексту файла."""
        assert ps.shelf_question("бриф", "фраза") == "бриф"
        assert ps.shelf_question(None, "фраза") == "фраза"
        assert ps.shelf_question("", "фраза") == "фраза"
        assert ps.shelf_question(None, None) == ""
        assert ps.shelf_question("  ", "фраза") == "фраза"

    def test_the_rule_lives_in_exactly_one_place(self):
        """Ключ кэша и вызов полки обязаны спрашивать ОДНУ функцию: разойдись
        они — ключ перестал бы соответствовать заданному вопросу, молча."""
        import ast
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        callers = sum(1 for n in ast.walk(tree)
                      if isinstance(n, ast.Call)
                      and getattr(n.func, "id", None) == "shelf_question")
        assert callers == 2, callers
        assert "shot_brief or block_text" not in src

    def test_without_phrase_the_shelf_still_gets_the_query(self, monkeypatch):
        """Ноль регресса для вызывающих без block_text (тесты, служебные
        прогоны, shelf_contact): brief=None -> полка берёт api_query."""
        seen = {}
        monkeypatch.setattr(ps.feature_flags, "enabled",
                            lambda name, *a, **k: name == "SHELF_INDEX")

        class FakeShelf:
            @staticmethod
            def available():
                return True

            @staticmethod
            def search(text, limit=40):
                seen["text"] = text
                return []

        monkeypatch.setitem(sys.modules, "shelf_index", FakeShelf)
        ps._SHELF_SEARCH_CACHE.clear()
        ps._shelf_search_photos("medieval helmet visor slit", brief=None)
        assert seen["text"] == "medieval helmet visor slit"

    def test_phrase_reaches_the_shelf_when_given(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(ps.feature_flags, "enabled",
                            lambda name, *a, **k: name == "SHELF_INDEX")

        class FakeShelf:
            @staticmethod
            def available():
                return True

            @staticmethod
            def search(text, limit=40):
                seen["text"] = text
                return []

        monkeypatch.setitem(sys.modules, "shelf_index", FakeShelf)
        ps._SHELF_SEARCH_CACHE.clear()
        ps._shelf_search_photos("medieval helmet visor slit",
                                brief="Стрела скользнула по нагруднику.")
        assert seen["text"] == "Стрела скользнула по нагруднику."


class TestShelfStackDrift:
    """Имя модели совпадает, версии библиотек уехали — векторы уже слегка из
    другого пространства, а прежний гвард молчал."""

    def test_nothing_to_compare_is_silence(self):
        assert si.stack_drift([{"stack": "torch2/tf5"}], None) == set()
        assert si.stack_drift([], "torch2/tf5") == set()

    def test_old_index_without_the_field_is_silence(self):
        """Индексы, собранные до этой правки, не должны предупреждать вечно."""
        assert si.stack_drift([{"id": "met:1"}, {"id": "met:2"}], "torch2/tf5") == set()

    def test_same_stack_is_silence(self):
        items = [{"stack": "torch2.14.0+cpu/tf5.17.0"}] * 3
        assert si.stack_drift(items, "torch2.14.0+cpu/tf5.17.0") == set()

    def test_drift_is_named(self):
        items = [{"stack": "torch2.9.0/tf4.44.0"}, {"stack": "torch2.14.0+cpu/tf5.17.0"}]
        assert si.stack_drift(items, "torch2.14.0+cpu/tf5.17.0") == {"torch2.9.0/tf4.44.0"}

    def test_drift_warns_but_does_not_disable_the_shelf(self):
        """Отказ обнулил бы 20 часов сборки из-за обновления библиотеки.
        Чужая МОДЕЛЬ — по-прежнему жёсткий отказ, это разные случаи."""
        src = open(os.path.join(SCRIPTS_DIR, "shelf_index.py"), encoding="utf-8").read()
        block = src[src.index("drifted = stack_drift(items, now)"):]
        block = block[:block.index("dim = items[0]")]
        assert "return (None, None)" not in block
        assert "ВНИМАНИЕ" in block
        # А у чужой модели отказ обязан остаться.
        model_block = src[src.index("stale = {it.get(\"model\")"):]
        model_block = model_block[:model_block.index("now = stack_signature()")]
        assert "return (None, None)" in model_block

    def test_build_records_the_stack(self):
        src = open(os.path.join(SCRIPTS_DIR, "shelf_index.py"), encoding="utf-8").read()
        assert '"stack": _BUILD_STACK,' in src
        # Считается один раз на сборку, а не на каждую запись: вызов тянет
        # импорт torch/transformers.
        assert "_BUILD_STACK = stack_signature()" in src

    def test_signature_is_real_on_this_machine(self):
        sig = si.stack_signature()
        assert sig is None or ("torch" in sig and "tf" in sig)
