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


class TestRunLocalNoModel:
    def test_no_model_returns_none_not_a_crash(self, tmp_path, monkeypatch):
        """Модели/llama_cpp на CI нет — run_local() обязан вернуть честное
        (None, "no_model"), а не ImportError наружу."""
        import shot_brief_director as sbd
        monkeypatch.setattr(sbd, "find_model", lambda explicit=None: None)
        profile, source = cw.run_local(str(tmp_path))
        assert profile is None and source == "no_model"
