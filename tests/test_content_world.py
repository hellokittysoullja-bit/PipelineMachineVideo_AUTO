"""Авто-определение ниши по тексту сценария (scripts/content_world.py).

Дисциплина этого репозитория для любого автоматического слоя: additive и
fail-open — нет файла/битый/неуверенный ответ, эффективный профиль
байт-в-байт равен channel_profile.json, как до модуля. Эти тесты проверяют
именно харнесс (парсинг ответа модели, слияние с channel_profile.json),
а не живой вызов модели — тот проверяется вручную, тем же принципом, что
и у speech_generate.py/shot_director.py ("перед первым использованием —
один живой ручной прогон", CLAUDE.md ЧАСТЬ 13).
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import content_world as cw  # noqa: E402


GOOD_ANSWER = """MOOD_TONE: -1
NICHE: психология / самопомощь
IS_HISTORICAL: no
WORLD: современный город, повседневная жизнь
PEOPLE_IN_FRAME: обычный человек, современная одежда, лицо не обязательно
FORBIDDEN: рыцарские доспехи, мечи, средневековые декорации
ANCHOR_WORDS: empty waiting room, phone face down, cluttered desk, tired eyes, cup of coffee
BLOCKLIST: medieval armor, knight, sword, castle
NEGATIVE_ANCHORS: modern city street with cars, asphalt and printed signage; a medieval manuscript with illumination
USE_MUSEUM_SOURCES: no
MOOD_TENSION: 2
CONFIDENCE: 0.86
"""


class TestParseAnswer:
    def test_extracts_all_fields(self):
        p = cw.parse_answer(GOOD_ANSWER)
        assert p["niche"] == "психология / самопомощь"
        assert p["is_historical"] is False
        assert p["world"].startswith("современный город")
        assert "phone face down" in p["anchor_words"]
        assert "sword" in p["blocklist_additions"]
        assert p["negative_anchor_additions"] == [
            "modern city street with cars, asphalt and printed signage",
            "a medieval manuscript with illumination",
        ]
        assert p["use_museum_sources"] is False
        assert p["mood_tone"] == -1.0
        assert p["mood_tension"] == 2.0
        assert p["confidence"] == 0.86

    def test_missing_lines_degrade_per_field_not_whole_profile(self):
        """Сорванная/непонятая строка теряет ровно одно поле — тот же
        принцип, что у построчного плана shot_brief_director."""
        broken = "NICHE: медицина\nЭТО НЕ СТРОКА ОТВЕТА, просто мусор\nCONFIDENCE: 0.7\n"
        p = cw.parse_answer(broken)
        assert p["niche"] == "медицина"
        assert p["confidence"] == 0.7
        assert "world" not in p

    def test_empty_answer_gives_zero_confidence_not_a_crash(self):
        p = cw.parse_answer("")
        assert p == {"confidence": 0.0}

    def test_garbage_confidence_value_is_ignored(self):
        p = cw.parse_answer("NICHE: тест\nCONFIDENCE: не число\n")
        assert p["confidence"] == 0.0

    def test_confidence_is_clamped_to_0_1(self):
        p = cw.parse_answer("CONFIDENCE: 5\n")
        assert p["confidence"] == 1.0

    def test_era_fields_are_ints_only_when_present(self):
        p = cw.parse_answer("ERA_FROM: 900\nERA_TO: 1600\n")
        assert p["era_from"] == 900 and isinstance(p["era_from"], int)
        assert p["era_to"] == 1600

    def test_yes_no_variants_recognized(self):
        assert cw.parse_answer("IS_HISTORICAL: YES\n")["is_historical"] is True
        assert cw.parse_answer("IS_HISTORICAL: Нет\n")["is_historical"] is False
        assert "is_historical" not in cw.parse_answer("IS_HISTORICAL: непонятно\n")


class TestFullScriptText:
    def test_strips_section_markers_and_pipeline_tags(self, tmp_path):
        script = (
            "=== METADATA ===\n"
            "TITLE: тест\n"
            "=== HOOK ===\n"
            "[energetic]Первая фраза.[pause] Вторая [shot:a lit desk lamp]фраза с брифом.\n"
            "=== PEXELS QUERIES ===\n"
            "HOOK: some query\n"
        )
        (tmp_path / "script.txt").write_text(script, encoding="utf-8")
        text = cw.full_script_text(str(tmp_path))
        assert "===" not in text
        assert "[pause]" not in text and "[shot:" not in text
        assert "Первая фраза." in text and "Вторая" in text and "фраза с брифом." in text
        assert "some query" not in text  # служебная секция запросов — не текст диктора

    def test_missing_script_returns_empty_string_not_error(self, tmp_path):
        assert cw.full_script_text(str(tmp_path)) == ""

    def test_max_chars_truncates(self, tmp_path):
        (tmp_path / "script.txt").write_text("=== HOOK ===\n" + "слово " * 5000, encoding="utf-8")
        text = cw.full_script_text(str(tmp_path), max_chars=100)
        assert len(text) == 100


class TestWriteLoadRoundTrip:
    def test_round_trip(self, tmp_path):
        profile = {"niche": "медицина", "confidence": 0.9, "use_museum_sources": False}
        path = cw.write_content_world(str(tmp_path), profile, source="test")
        assert os.path.exists(path)
        loaded = cw.load_content_world(str(tmp_path))
        assert loaded["niche"] == "медицина"
        assert loaded["source"] == "test"

    def test_low_confidence_file_is_ignored_wholesale(self, tmp_path):
        cw.write_content_world(str(tmp_path), {"niche": "х", "confidence": 0.1}, source="test")
        assert cw.load_content_world(str(tmp_path)) == {}

    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert cw.load_content_world(str(tmp_path)) == {}

    def test_corrupt_file_fails_open(self, tmp_path):
        os.makedirs(tmp_path / "media_plan", exist_ok=True)
        (tmp_path / "media_plan" / "content_world.json").write_text("{не json", encoding="utf-8")
        assert cw.load_content_world(str(tmp_path)) == {}

    def test_non_dict_json_fails_open(self, tmp_path):
        os.makedirs(tmp_path / "media_plan", exist_ok=True)
        (tmp_path / "media_plan" / "content_world.json").write_text("[1, 2, 3]", encoding="utf-8")
        assert cw.load_content_world(str(tmp_path)) == {}


class TestMergeContentWorld:
    """Правила слияния — см. docstring merge_content_world(). Каждый тест
    — один пункт правила, не общее "работает"."""

    def test_no_video_dir_returns_same_object_byte_identical(self, tmp_path):
        base = {"content_alt_blocklist": ["katana"]}
        assert cw.merge_content_world(base, None) is base

    def test_no_content_world_file_returns_same_object(self, tmp_path):
        base = {"content_alt_blocklist": ["katana"]}
        assert cw.merge_content_world(base, str(tmp_path)) is base

    def test_low_confidence_returns_same_object(self, tmp_path):
        cw.write_content_world(str(tmp_path), {"niche": "х", "confidence": 0.2,
                                                "blocklist_additions": ["phone"]}, source="t")
        base = {"content_alt_blocklist": ["katana"]}
        assert cw.merge_content_world(base, str(tmp_path)) is base

    def test_blocklist_additions_land_under_separate_key(self, tmp_path):
        """Списки НЕ вливаются прямо в "content_alt_blocklist" — под
        отдельным "..._additions" (см. docstring merge_content_world:
        реальная найденная дыра — .get(key, DEFAULT) у потребителя теряет
        код-дефолт, если ключ "появился" в профиле хоть с одной добавкой).
        Объединение с код-дефолтом делает merged_list() в точке потребления."""
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.9,
            "blocklist_additions": ["Phone", "sword", "modern car"],
        }, source="t")
        base = {"content_alt_blocklist": ["katana", "PHONE"]}
        merged = cw.merge_content_world(base, str(tmp_path))
        assert merged["content_alt_blocklist_additions"] == ["Phone", "sword", "modern car"]
        # канальный список НЕ тронут правкой (merge не мутирует и не заменяет его).
        assert merged["content_alt_blocklist"] == ["katana", "PHONE"]
        assert base["content_alt_blocklist"] == ["katana", "PHONE"]

    def test_negative_anchor_additions_land_under_separate_key(self, tmp_path):
        """negative_anchor_additions судится ЭМБЕДДИНГОМ (CLIP margin в
        negative_anchor_violation), не подстрокой — фразы содержат запятые
        внутри себя, поэтому разделитель ";", а не ",". Та же защита от
        потери код-дефолта, что у блоклиста выше."""
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.9,
            "negative_anchor_additions": [
                "modern city street with cars, asphalt and signage",
                "a person using a smartphone in a cafe",
            ],
        }, source="t")
        base = {"content_negative_anchors": ["east asian temple, kimono and curved sword"]}
        merged = cw.merge_content_world(base, str(tmp_path))
        assert merged["content_negative_anchors_additions"] == [
            "modern city street with cars, asphalt and signage",
            "a person using a smartphone in a cafe",
        ]
        assert merged["content_negative_anchors"] == ["east asian temple, kimono and curved sword"]


class TestMergedList:
    """merged_list() — финальное объединение "канал-или-код-дефолт" +
    добавки content_world, в ТОЧКЕ ПОТРЕБЛЕНИЯ (pipeline_smart.py и
    аналоги), а не внутри merge_content_world() (см. её docstring)."""

    def test_channel_absent_keeps_code_default_plus_additions(self):
        """РЕГРЕССИОННЫЙ тест на реально найденную дыру 17.09: канал НЕ
        задал ключ явно (как content_negative_anchors у этого репозитория)
        — код-дефолт обязан выжить, а не потеряться под добавками."""
        profile = {"content_negative_anchors_additions": ["a modern city street"]}
        code_default = ("east asian temple, kimono and curved sword",)
        result = cw.merged_list(profile, "content_negative_anchors", code_default)
        assert result == ("east asian temple, kimono and curved sword", "a modern city street")

    def test_channel_explicit_list_wins_over_code_default(self):
        profile = {"content_alt_blocklist": ["katana"],
                   "content_alt_blocklist_additions": ["modern car"]}
        result = cw.merged_list(profile, "content_alt_blocklist", ("some", "code", "default"))
        assert result == ("katana", "modern car")

    def test_no_additions_returns_channel_or_default_unchanged(self):
        assert cw.merged_list({}, "content_alt_blocklist", ("a", "b")) == ("a", "b")
        assert cw.merged_list({"content_alt_blocklist": ["x"]}, "content_alt_blocklist", ("a",)) == ("x",)

    def test_dedup_is_case_insensitive(self):
        profile = {"content_alt_blocklist": ["Katana"],
                   "content_alt_blocklist_additions": ["katana", "sword"]}
        result = cw.merged_list(profile, "content_alt_blocklist")
        assert result == ("Katana", "sword")

    def test_shot_domain_auto_used_when_channel_has_none(self, tmp_path):
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.9, "world": "современный офис",
            "people_in_frame": "обычный человек",
            "anchor_words": ["laptop", "coffee cup"],
        }, source="t")
        merged = cw.merge_content_world({}, str(tmp_path))
        assert merged["shot_domain"]["world"] == "современный офис"
        assert "laptop" in merged["shot_domain"]["anchor_words"]

    def test_is_historical_flows_through_merge_at_min_confidence(self, tmp_path):
        """is_historical не литеральный ключ channel_profile.json (канал
        никогда его не объявляет руками) — достаточно MIN_CONFIDENCE, не
        WORLD_OVERRIDE_MIN_CONFIDENCE: здесь нет явного решения человека,
        которое можно было бы тихо переписать."""
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.6, "is_historical": False,
        }, source="t")
        base = {"shot_domain": {"world": "европейское Средневековье"}}
        merged = cw.merge_content_world(base, str(tmp_path))
        assert merged["is_historical"] is False
        # мир кадра при этом НЕ переписан (confidence ниже WORLD_OVERRIDE_
        # MIN_CONFIDENCE) — is_historical и shot_domain это два разных поля
        # с разными порогами, не связанные одной проверкой.
        assert merged["shot_domain"]["world"] == "европейское Средневековье"

    def test_shot_domain_channel_wins_on_moderate_confidence(self, tmp_path):
        """Устоявшаяся ниша канала — осознанный выбор человека (ЧАСТЬ 24
        CLAUDE.md); неуверенная/пограничная догадка по одному эпизоду не
        имеет права её тихо переписать. confidence между MIN_CONFIDENCE и
        WORLD_OVERRIDE_MIN_CONFIDENCE — авто-профиль ещё эффективен
        (проходит в списки), но мир кадра ещё не трогает."""
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.6, "world": "современный офис",
        }, source="t")
        base = {"shot_domain": {"world": "европейское Средневековье"}}
        merged = cw.merge_content_world(base, str(tmp_path))
        assert merged["shot_domain"]["world"] == "европейское Средневековье"

    def test_shot_domain_high_confidence_overrides_channel(self, tmp_path):
        """Найдено живым прогоном на реальном психологическом сценарии
        (tests/fixtures/other_niche/script_psychology.txt) в ЭТОМ же
        репозитории: у канала уже задан shot_domain (военная история), и
        первая версия правила ("канал всегда побеждает") душила автономию
        именно там, где задача прямо требует её — на теме, которую канал
        никогда не настраивал. При ДОСТАТОЧНО уверенном авто-профиле
        (WORLD_OVERRIDE_MIN_CONFIDENCE) новая тема имеет право переопределить
        даже явно заданный мир — иначе цель "любая тема автономно" не
        выполняется на практике ни в одном реальном репозитории с уже
        настроенным каналом."""
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.92, "world": "современная повседневная жизнь",
        }, source="t")
        base = {"shot_domain": {"world": "европейское Средневековье"}}
        merged = cw.merge_content_world(base, str(tmp_path))
        assert merged["shot_domain"]["world"] == "современная повседневная жизнь"

    def test_use_museum_sources_fills_when_channel_absent(self, tmp_path):
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.9, "use_museum_sources": False,
        }, source="t")
        merged = cw.merge_content_world({}, str(tmp_path))
        assert merged["use_museum_sources"] is False

    def test_use_museum_sources_channel_wins_on_moderate_confidence(self, tmp_path):
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.6, "use_museum_sources": False,
        }, source="t")
        merged = cw.merge_content_world({"use_museum_sources": True}, str(tmp_path))
        assert merged["use_museum_sources"] is True

    def test_use_museum_sources_high_confidence_overrides_channel(self, tmp_path):
        """Один эпизод другой темы в устоявшемся канале — реальный вопрос
        "искать ли музейные предметы для ЭТОГО ролика", а не свойство
        канала навечно; при уверенном авто-профиле имеет право переопределить."""
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.92, "use_museum_sources": False,
        }, source="t")
        merged = cw.merge_content_world({"use_museum_sources": True}, str(tmp_path))
        assert merged["use_museum_sources"] is False

    def test_era_window_fills_when_channel_absent(self, tmp_path):
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.9, "era_from": 1800, "era_to": 1900,
        }, source="t")
        merged = cw.merge_content_world({}, str(tmp_path))
        assert merged["era_from"] == 1800 and merged["era_to"] == 1900

    def test_era_window_channel_wins_on_moderate_confidence(self, tmp_path):
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.6, "era_from": 1800, "era_to": 1900,
        }, source="t")
        merged = cw.merge_content_world({"era_from": 900, "era_to": 1600}, str(tmp_path))
        assert merged["era_from"] == 900 and merged["era_to"] == 1600


class TestHistoricalDefault:
    """historical_default()/resolve_is_historical() — найдено 17.09: список
    ловушек анти-модерн ("толпа зрителей", "современная кухня") САМ ПО
    СЕБЕ кодирует "эта ниша историческая", и применять его как
    "универсальный дефолт" для другой ниши значило бы рисковать отклонить
    ровно тот кадр, который этой теме и нужен."""

    def test_resolve_none_when_no_signal_at_all(self):
        assert cw.resolve_is_historical({}) is None

    def test_bare_shot_domain_is_not_a_history_signal(self):
        """ИСПРАВЛЕНО 17.09 (прошлая версия этого теста требовала True).
        Наличие мира кадра НЕ означает историчность: `shot_domain` объявляет
        и канал про психологию, просто мир там современный. Считать «мир
        задан» за «канал исторический» значило записать в историки любой
        настроенный канал — и тогда подавление чужого мира (см.
        shot_domain_for_prompt) сработало бы против законного современного
        мира. Признак историчности — окно/якоря эпохи."""
        assert cw.resolve_is_historical({"shot_domain": {"world": "x"}}) is None
        assert cw.resolve_is_historical({"shot_domain": {"world": "x"},
                                          "openverse_era_anchors": ["medieval"]}) is True

    def test_resolve_true_from_era_window_presence(self):
        assert cw.resolve_is_historical({"era_from": 900}) is True
        assert cw.resolve_is_historical({"era_to": 1600}) is True

    def test_content_world_is_historical_wins_over_shot_domain_heuristic(self):
        """Явный вывод content_world (даже False) — приоритетнее эвристики
        по shot_domain: канал исторический, но ЭТОТ эпизод — нет."""
        assert cw.resolve_is_historical(
            {"shot_domain": {"world": "x"}, "is_historical": False}) is False
        assert cw.resolve_is_historical({"is_historical": True}) is True

    def test_historical_default_returns_historical_value_when_true_or_unknown(self):
        # Неизвестно -> прежнее поведение (историческое значение), байт-в-байт.
        assert cw.historical_default({}, ("trap",)) == ("trap",)
        assert cw.historical_default({"shot_domain": {"world": "x"}}, ("trap",)) == ("trap",)

    def test_historical_default_returns_other_value_when_confidently_not_historical(self):
        profile = {"shot_domain": {"world": "средневековье"}, "is_historical": False}
        assert cw.historical_default(profile, ("modern kitchen trap",)) == ()
        assert cw.historical_default(profile, ("trap",), other_value=("psych trap",)) == ("psych trap",)


class TestEffectiveProfile:
    def test_uses_env_video_dir_when_no_explicit_arg(self, tmp_path, monkeypatch):
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.9, "use_museum_sources": False,
        }, source="t")
        monkeypatch.setattr(cw, "_base_channel_profile", lambda: {})
        monkeypatch.setenv(cw.ENV_VIDEO_DIR, str(tmp_path))
        assert cw.effective_profile().get("use_museum_sources") is False

    def test_explicit_video_dir_overrides_env(self, tmp_path, monkeypatch):
        other = tmp_path / "other"
        other.mkdir()
        cw.write_content_world(str(other), {"confidence": 0.9, "niche": "тест"}, source="t")
        monkeypatch.setattr(cw, "_base_channel_profile", lambda: {})
        monkeypatch.setenv(cw.ENV_VIDEO_DIR, str(tmp_path))
        merged = cw.effective_profile(str(other))
        # "niche" сам по себе не формирует shot_domain (тот собирается
        # только из world/people_in_frame/forbidden/anchor_words) — здесь
        # важно, что merge не упал и реально читал `other`, а не ENV_VIDEO_DIR.
        assert "shot_domain" not in merged
        assert cw.load_content_world(str(other)).get("niche") == "тест"


class TestValidateProfile:
    """Содержимое профиля проверяется НА ГРАНИЦЕ ЧТЕНИЯ — профиль может
    написать кто угодно (CLI, локальная модель, сессия, человек руками), и
    мусор в ловушках вето означает не «слой не сработал», а СЛУЧАЙНЫЕ
    отказы законным кадрам (см. docstring validate_profile)."""

    def test_cyrillic_anchor_is_rejected(self):
        """Текстовые башни CLIP/SigLIP английские — кириллица в ловушке даёт
        шум, а шум в ВЕТО это отказы годным кандидатам. Диагноз уже записан
        в этом репозитории дословно для CLAP."""
        clean, rejected = cw.validate_profile(
            {"negative_anchor_additions": ["a modern city street", "рыцарь в доспехах"]})
        assert clean["negative_anchor_additions"] == ["a modern city street"]
        assert [r["reason"] for r in rejected] == ["not_latin_or_too_long"]

    def test_overlong_anchor_is_rejected(self):
        """Строка длиннее лимита токенов молча ОБРЕЗАЕТСЯ моделью — ловушка
        сравнивается с кадром не тем текстом, который написан."""
        clean, rejected = cw.validate_profile(
            {"negative_anchor_additions": ["a " + "very " * 100 + "long trap"]})
        assert "negative_anchor_additions" not in clean
        assert rejected and rejected[0]["reason"] == "not_latin_or_too_long"

    def test_non_string_item_is_rejected(self):
        clean, rejected = cw.validate_profile({"blocklist_additions": ["knight", 42, None]})
        assert clean["blocklist_additions"] == ["knight"]
        assert [r["reason"] for r in rejected] == ["not_a_string", "not_a_string"]

    def test_non_list_field_is_dropped(self):
        clean, rejected = cw.validate_profile({"anchor_words": "laptop, desk"})
        assert "anchor_words" not in clean
        assert rejected[0]["reason"] == "not_a_list"

    def test_inverted_era_window_is_dropped_whole(self):
        clean, rejected = cw.validate_profile({"era_from": 1900, "era_to": 1800})
        assert "era_from" not in clean and "era_to" not in clean
        assert rejected[0]["reason"] == "from_after_to"

    def test_valid_profile_passes_untouched(self):
        profile = {"negative_anchor_additions": ["a modern city street with cars"],
                    "blocklist_additions": ["knight"], "era_from": 900, "era_to": 1600}
        clean, rejected = cw.validate_profile(profile)
        assert rejected == []
        assert clean == profile

    def test_load_applies_validation_and_records_rejections(self, tmp_path):
        cw.write_content_world(str(tmp_path), {
            "confidence": 0.9,
            "negative_anchor_additions": ["a modern kitchen", "кириллица"],
        }, source="t")
        loaded = cw.load_content_world(str(tmp_path))
        assert loaded["negative_anchor_additions"] == ["a modern kitchen"]
        assert len(loaded["_rejected"]) == 1


class TestShotDomainForPrompt:
    """Воспроизведено живьём 17.09: психологический эпизод получал «МИР
    КАДРА: европейское Средневековье… в кадре не должно быть современной
    техники» — то есть сценарию про ноутбук запрещался ноутбук."""

    HIST_CHANNEL = {"shot_domain": {"world": "европейское Средневековье"},
                     "openverse_era_anchors": ["medieval"]}

    def test_foreign_historical_world_is_suppressed_for_non_historical_episode(self):
        profile = dict(self.HIST_CHANNEL, is_historical=False)
        assert cw.shot_domain_for_prompt(profile) == {}

    def test_episode_own_world_is_always_used(self):
        profile = dict(self.HIST_CHANNEL, is_historical=False,
                       shot_domain={"world": "современная жизнь"},
                       shot_domain_source="content_world")
        assert cw.shot_domain_for_prompt(profile)["world"] == "современная жизнь"

    def test_historical_episode_keeps_channel_world(self):
        profile = dict(self.HIST_CHANNEL, is_historical=True)
        assert cw.shot_domain_for_prompt(profile)["world"] == "европейское Средневековье"

    def test_no_profile_keeps_channel_world_byte_for_byte(self):
        assert cw.shot_domain_for_prompt(self.HIST_CHANNEL)["world"] == "европейское Средневековье"

    def test_modern_channel_keeps_its_own_modern_world(self):
        """Канал про современную нишу НЕ теряет свой мир: признаком
        историчности канала служит окно/якоря эпохи, а не сам факт наличия
        мира — иначе подавляли бы законный современный мир."""
        modern_channel = {"shot_domain": {"world": "современный офис"}}
        profile = dict(modern_channel, is_historical=False)
        assert cw.shot_domain_for_prompt(profile)["world"] == "современный офис"

    def test_channel_declares_history_ignores_bare_shot_domain(self):
        assert cw.channel_declares_history({"shot_domain": {"world": "x"}}) is False
        assert cw.channel_declares_history({"era_from": 900}) is True
        assert cw.channel_declares_history({"openverse_era_anchors": ["medieval"]}) is True


class TestOrchestratorNicheStage:
    """Шаг ниши в render_episode.py — тесты здесь, а не в
    test_render_episode.py, потому что тот файл целиком помечен skipif по
    отсутствию ffmpeg, а этот шаг к ffmpeg отношения не имеет."""

    def _orchestrator(self):
        import render_episode
        return render_episode

    def test_existing_profile_is_reported_as_present_without_running_anything(self, tmp_path, monkeypatch):
        cw.write_content_world(str(tmp_path), {"confidence": 0.9, "niche": "психология"},
                                source="t")
        re_mod = self._orchestrator()
        calls = []
        monkeypatch.setattr(re_mod, "_run", lambda *a, **k: calls.append(a) or 0)
        status, details = re_mod._content_world_stage(str(tmp_path))
        assert status == "present" and details["niche"] == "психология"
        assert calls == [], "готовый профиль не должен заново запускать определение"

    def test_missing_profile_triggers_generation_attempt(self, tmp_path, monkeypatch):
        re_mod = self._orchestrator()
        calls = []

        def fake_run(script, video_dir, **k):
            calls.append(script)
            cw.write_content_world(video_dir, {"confidence": 0.8, "niche": "медицина"},
                                    source="local:test")
            return 0

        monkeypatch.setattr(re_mod, "_run", fake_run)
        status, details = re_mod._content_world_stage(str(tmp_path))
        assert calls == ["content_world.py"]
        assert status == "generated" and details["niche"] == "медицина"

    def test_unavailable_when_generation_produces_nothing(self, tmp_path, monkeypatch):
        re_mod = self._orchestrator()
        monkeypatch.setattr(re_mod, "_run", lambda *a, **k: 2)
        status, _ = re_mod._content_world_stage(str(tmp_path))
        assert status == "unavailable"

    def test_low_confidence_file_is_distinguished_from_missing(self, tmp_path, monkeypatch):
        """«Профиль есть, но неуверенный» и «мозга нет» — разные диагнозы, и
        в манифесте они обязаны читаться по-разному."""
        cw.write_content_world(str(tmp_path), {"confidence": 0.1, "niche": "х"}, source="t")
        re_mod = self._orchestrator()
        monkeypatch.setattr(re_mod, "_run", lambda *a, **k: 0)
        status, _ = re_mod._content_world_stage(str(tmp_path))
        assert status == "low_confidence"


class TestRunLocalNoModel:
    def test_no_model_returns_none_not_a_crash(self, tmp_path, monkeypatch):
        """Модели/llama_cpp на CI нет — run_local() обязан вернуть честное
        (None, "no_model"), а не ImportError наружу."""
        import shot_brief_director as sbd
        monkeypatch.setattr(sbd, "find_model", lambda explicit=None: None)
        profile, source = cw.run_local(str(tmp_path))
        assert profile is None and source == "no_model"


class TestEmptyNicheListsAreFilled:
    """Авто-ниша заполняет ПУСТЫЕ списки якорей всегда; ОБЪЯВЛЕННЫЕ каналом
    заменяет — не дополняет — только при высокой уверенности
    (WORLD_OVERRIDE_MIN_CONFIDENCE), той же, что у shot_domain.

    ЗАЧЕМ ВООБЩЕ ЗАПОЛНЯТЬ. brief_to_stock_query() ставит якорь эпохи в
    КАЖДЫЙ запрос без своего (CLAUDE.md: «0 запросов из 142 без якоря
    эпохи»), а берёт якоря из openverse_era_anchors. После обнуления
    дефолтов кода (17.09) у канала новой ниши этот список пуст, а парсер
    ответа модели его не заполнял: ANCHOR_WORDS уходили только в
    shot_domain. Для эпизода про каменный век это значит, что «a flint
    hand axe held in a palm» уходит в сток без единого слова об эпохе —
    ровно тот случай, который в этом файле уже записан числом: одиночное
    `plate armour` первым результатом даёт «MkIV-Tank-Plate».

    ЗАЧЕМ ЗАМЕНЯТЬ, А НЕ ТОЛЬКО ЗАПОЛНЯТЬ — ЖИВОЙ ПРОГОН 17.09,
    А НЕ РАССУЖДЕНИЕ. Первая версия этого правила («канал побеждает
    всегда, даже при высокой уверенности») была протестирована ниже
    синтетикой и выглядела безопасной — но на НАСТОЯЩЕМ, уже настроенном
    военно-историческом канале (этот самый репозиторий, где
    openverse_era_anchors уже объявлены: 9 средневековых слов) она дала
    реальный испорченный запрос для тестового эпизода про историю пиццы
    (confidence=0.92, свой мир «Неаполь 18-20 века»):
        'a ripe red tomato on a rustic table' -> 'medieval ripe red tomato rustic'
    Три соседних поля override (shot_domain/use_museum_sources/era) в том
    же эпизоде сработали правильно — только этот список нёс чужую нишу
    дальше, в реальный вызов Pexels/Openverse. Раз мир эпизода при высокой
    уверенности переписывает канальный целиком (shot_domain), список,
    решающий, каким якорем ЭПОХИ подписывать запросы этого же эпизода,
    обязан следовать тому же правилу — иначе три «да» и одно «нет» в
    одном и том же решении о нише самого себя противоречат друг другу.

    ПОЧЕМУ ПРИ УМЕРЕННОЙ УВЕРЕННОСТИ — ПО-ПРЕЖНЕМУ НЕ ДОБАВКОЙ И НЕ
    ЗАМЕНОЙ. Канал, объявивший якоря, на них откалиброван: подмешать
    эпизодные слова В список канала значило бы, что ЧАСТЬ запросов
    сочтётся «уже с якорем» и перестанет получать канальный —
    добавление ОСЛАБИЛО бы гарантию. Ниже уверенности override —
    заменить список угадкой, которой сам механизм не доверяет настолько,
    чтобы переписать даже shot_domain, — тоже неверно. Поэтому при
    умеренной уверенности список остаётся канальным без изменений.
    """

    PROFILE = {
        "niche": "каменный век", "confidence": 0.9, "is_historical": True,
        "era_from": -30000, "era_to": -3000, "use_museum_sources": True,
        "world": "stone age daily life",
        "anchor_words": ["prehistoric", "neolithic", "flint", "stone tool",
                         "cave", "hand axe"],
    }

    def _merged(self, tmp_path, base, profile=None):
        cw.write_content_world(str(tmp_path), dict(profile or self.PROFILE), source="test")
        return cw.merge_content_world(dict(base), str(tmp_path))

    @pytest.mark.parametrize("key", ["openverse_era_anchors",
                                     "openverse_domain_nouns",
                                     "query_era_anchors"])
    def test_empty_list_is_filled_from_anchor_words(self, tmp_path, key):
        merged = self._merged(tmp_path, {})
        assert merged[key] == self.PROFILE["anchor_words"]

    @pytest.mark.parametrize("key", ["openverse_era_anchors",
                                     "openverse_domain_nouns",
                                     "query_era_anchors"])
    def test_moderate_confidence_never_touches_declared_list(self, tmp_path, key):
        """Ниже WORLD_OVERRIDE_MIN_CONFIDENCE (0.75) — ни заполнения, ни
        замены: угадке не доверяют настолько, чтобы переписать даже
        shot_domain, значит и список якорей она переписать не вправе."""
        profile = dict(self.PROFILE, confidence=0.6)
        base = {key: ["medieval", "knight"]}
        merged = self._merged(tmp_path, base, profile=profile)
        assert merged[key] == ["medieval", "knight"]

    @pytest.mark.parametrize("key", ["openverse_era_anchors",
                                     "openverse_domain_nouns",
                                     "query_era_anchors"])
    def test_high_confidence_replaces_declared_list(self, tmp_path, key):
        """>= WORLD_OVERRIDE_MIN_CONFIDENCE — список ЗАМЕНЯЕТСЯ целиком
        (не объединяется: смешение вернуло бы ровно ту порчу запроса
        «medieval ripe red tomato», ради которой список остаётся вне
        _LIST_FIELDS_ADDITIVE)."""
        base = {key: ["medieval", "knight"]}
        merged = self._merged(tmp_path, base)
        assert merged[key] == self.PROFILE["anchor_words"]

    def test_this_repository_unchanged_at_moderate_confidence(self, tmp_path):
        """Канал этого репозитория все три списка объявил — шумная догадка
        (confidence ниже порога override) не имеет права подмешать в них
        ни слова, та же гарантия, что у shot_domain."""
        import json
        base = json.load(open(os.path.join(REPO_ROOT, "channel_profile.json"),
                              encoding="utf-8"))
        profile = dict(self.PROFILE, confidence=0.6)
        merged = self._merged(tmp_path, base, profile=profile)
        for key in ("openverse_era_anchors", "openverse_domain_nouns",
                    "query_era_anchors"):
            assert merged[key] == base[key], key

    def test_this_repository_is_overridden_at_high_confidence(self, tmp_path):
        """ЖИВОЙ СЛУЧАЙ 17.09: настоящий channel_profile.json этого канала
        (медиевализм объявлен явно) + уверенный автопрофиль про историю
        пиццы -> якоря эпохи обязаны стать пиццерийными, а не средневековыми,
        иначе brief_to_stock_query() подпишет пиццу под «medieval»."""
        import json
        base = json.load(open(os.path.join(REPO_ROOT, "channel_profile.json"),
                              encoding="utf-8"))
        merged = self._merged(tmp_path, base)
        for key in ("openverse_era_anchors", "openverse_domain_nouns",
                    "query_era_anchors"):
            assert merged[key] == self.PROFILE["anchor_words"], key
            assert "medieval" not in merged[key], key

    def test_no_anchor_words_changes_nothing(self, tmp_path):
        profile = {k: v for k, v in self.PROFILE.items() if k != "anchor_words"}
        cw.write_content_world(str(tmp_path), profile, source="test")
        merged = cw.merge_content_world({}, str(tmp_path))
        for key in ("openverse_era_anchors", "openverse_domain_nouns",
                    "query_era_anchors"):
            assert key not in merged, key
