"""Тесты scripts/stress_placement.py — коррекция ударений известных
омографов (мука/замок) поверх ruaccent, см. докстринг модуля и CLAUDE.md
("Стресс-плейсмент по-русски"). Реальная модель (~30с загрузка один раз
на процесс, дальше быстро) — тот же паттерн, что
TestSentenceRelevanceSiglip2RealModel в test_visual_director.py."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import stress_placement as sp   # noqa: E402


# ---------- чистая логика, без модели ----------

def test_detect_sense_flour_trigger():
    assert sp._detect_sense("мука", ["хлеба", "стоила"]) == "flour"


def test_detect_sense_suffering_trigger():
    assert sp._detect_sense("мука", ["голода", "страдание"]) == "suffering"


def test_detect_sense_no_signal_returns_none():
    assert sp._detect_sense("мука", ["стоила", "дорого", "очень"]) is None


def test_detect_sense_castle_trigger():
    assert sp._detect_sense("замок", ["рыцари", "крепость"]) == "castle"


def test_detect_sense_lock_trigger():
    assert sp._detect_sense("замок", ["дверь", "ключ"]) == "lock"


def test_match_case_preserves_uppercase_first_letter():
    assert sp._match_case("мук+а", "Мука") == "Мук+а"
    assert sp._match_case("+ука", "Мука") == "+Ука"


def test_match_case_preserves_lowercase():
    assert sp._match_case("мук+а", "мука") == "мук+а"


def test_homograph_forms_cover_common_singular_cases():
    for form in ("мука", "муки", "муке", "муку", "мукой"):
        assert sp.HOMOGRAPH_FORMS[form] == "мука"
    for form in ("замок", "замка", "замку", "замком", "замке"):
        assert sp.HOMOGRAPH_FORMS[form] == "замок"


def test_atlas_deliberately_not_included():
    # "атлас" исследован и СОЗНАТЕЛЬНО не включён — ruaccent путается даже
    # при сильных подсказках (см. докстринг модуля), честнее не притворяться
    # надёжным. Регрессия этого решения была бы тихой и опасной.
    assert "атлас" not in sp.HOMOGRAPH_FORMS
    assert "атлас" not in sp.HOMOGRAPH_SENSES


# ---------- реальная модель (ruaccent, ~30с загрузка один раз) ----------

class TestAccentizeWithHomographCorrectionRealModel:

    def test_fixes_known_bug_bare_nominative_flour(self):
        # Реальный, ранее найденный баг: "Мука для хлеба..." ruaccent сам по
        # себе даёт М+ука (му́ка, страдание) вместо верного мук+а (мука́,
        # продукт). Эта регрессия — единственная причина, по которой пункт
        # был отложен в прошлом заходе; теперь измеримо исправлена.
        out = sp.accentize_with_homograph_correction(
            "Мука для хлеба стоила очень дорого в тот год.")
        assert "мук+а" in out.lower()
        assert "м+ука" not in out.lower()

    def test_does_not_regress_already_correct_suffering_case(self):
        out = sp.accentize_with_homograph_correction(
            "Голод причинял людям страшные муки.")
        assert "м+уки" in out.lower()

    def test_does_not_regress_already_correct_flour_with_adjective(self):
        out = sp.accentize_with_homograph_correction(
            "Пшеничная мука рассыпалась по полу мельницы.")
        assert "мук+а" in out.lower()

    def test_fixes_lock_sense_with_context(self):
        out = sp.accentize_with_homograph_correction(
            "Вор взломал замок за минуту.")
        assert "зам+ок" in out.lower()

    def test_does_not_regress_castle_sense(self):
        out = sp.accentize_with_homograph_correction(
            "Рыцари защищали замок несколько дней.")
        assert "з+амок" in out.lower()

    def test_no_signal_leaves_baseline_untouched(self):
        text = "Крестьяне выходили парить репу в печи."
        baseline = sp._get_accentizer().process_all(text)
        out = sp.accentize_with_homograph_correction(text)
        assert out == baseline

    def test_word_with_no_homograph_entry_untouched(self):
        text = "Солнце светило над полем весь день."
        baseline = sp._get_accentizer().process_all(text)
        out = sp.accentize_with_homograph_correction(text)
        assert out == baseline

    def test_probe_case_insensitivity_bug_regression(self):
        # Реальный найденный баг: заглавная буква ВНУТРИ зонда ("...,
        # Мука.") заставляла модель заподозрить имя собственное и
        # перевернуть предсказание. Зонд должен строиться на строчной
        # форме независимо от регистра исходного слова.
        lower = sp._probe_accent_for_form("мука", "мука", "flour")
        upper_source_result = sp.accentize_with_homograph_correction(
            "Мука для хлеба стоила дорого.")
        assert lower.lower().replace("+", "") == "мука"
        assert "мук+а" in lower.lower()   # верно: стресс на последнем слоге
        assert "мук+а" in upper_source_result.lower()
