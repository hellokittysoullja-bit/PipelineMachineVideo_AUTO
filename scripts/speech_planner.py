#!/usr/bin/env python3
"""Speech Planner — риторическое планирование ритма речи ДО озвучки (новая
ЧАСТЬ 13, шаг между финализацией сценария и озвучкой). Первая половина
Speech Director: script.txt разбирается на уже существующие смысловые
единицы (parse_blocks() — тот же разбор, что использует pipeline_smart.py,
границы по [pause]/[short pause]/секциям), каждой назначается риторическая
роль (reveal_hold/evidence_beat/anticipation/closing_hold/punchy/
question_rise/connective) по разметке, которая и так уже есть в сценарии:
секции, [stat:...], [climax], пунктуация (двоеточие, тире, вопрос), длина
фразы, позиция в хуке/финале — без обращения к платному голосовому API на
этапе планирования.

НЕ вызывает TTS и не трогает script.txt. Результат — media_plan/speech_plan.json
(строго валидируемая структура, см. validate_plan()) + человекочитаемая
аннотация media_plan/speech_plan_annotated.txt с рекомендациями по тегам —
её читает человек ПЕРЕД тем, как озвучивать (тот же ручной шаг, что и
сейчас, просто с режиссёрской подсказкой вместо гадания). Модельно-
осознанный SSML-вариант (media_plan/speech_plan_ssml.txt) генерируется
дополнительно, если TTS_MODEL в .env указывает на модель с поддержкой
<break time="x.xs" /> (Multilingual v2 / Flash v2 / Flash v2.5) — для модели
по умолчанию этого канала (Eleven v3, ЧАСТЬ 10 CLAUDE.md) SSML-breaks не
поддерживаются, поэтому единственный канал управления — уже разрешённый
словарь тегов [pause]/[short pause]/[slowly]/[emphasis]/[energetic]
(НИКОГДА не [long pause] — CLAUDE.md прямо называет его источником
артефактов TTS).

Usage: python scripts/speech_planner.py <video_dir>
Вход: script.txt. Выход: media_plan/speech_plan.json,
media_plan/speech_plan_annotated.txt, (опц.) media_plan/speech_plan_ssml.txt."""
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

VIDEO_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

# pipeline_smart читает VIDEO_FOLDER из sys.argv[1] НА ИМПОРТЕ — та же
# подмена, что уже использует tests/test_parse.py и scripts/visual_qc.py.
_saved_argv = sys.argv
sys.argv = ["pipeline_smart.py", VIDEO_DIR]
import pipeline_smart  # noqa: E402
sys.argv = _saved_argv


# --- Разрешённый словарь тегов (ЧАСТЬ 10 CLAUDE.md) — Speech Planner НИКОГДА
# не рекомендует ничего за пределами этого списка (в частности, никогда
# [long pause] — CLAUDE.md прямо документирует его как источник TTS-
# артефактов на этом голосе). ---
ALLOWED_TAGS = ("[pause]", "[short pause]", "[slowly]", "[emphasis]", "[energetic]")

# target_range_sec для обычной связки ("connective") — НЕ новые глобальные
# константы длительности паузы в pipeline_smart.py (там их как не было, так
# и нет) — это ДИАПАЗОН для планирования/валидации, живёт только здесь и в
# speech_validator.py, не подменяет исполняемый [pause]=0.8/[short pause]=0.4
# pipeline_smart.PAUSE_DURATIONS (та константа по-прежнему единственная
# управляет тем, сколько монтаж резервирует под тег при отсутствии
# speech_timeline.json — safety net, см. fix_pauses.py).
TAG_BASE_RANGE = {
    "[pause]": (0.55, 0.95),
    "[short pause]": (0.25, 0.50),
    None: (0.0, 0.0),
}

# Риторические роли — диапазон И (для reveal_hold/closing_hold/anticipation)
# минимальный РЕКОМЕНДУЕМЫЙ тег, если у блока сейчас тег слабее. Верхняя
# граница reveal_hold была 1.40с (обоснование — "1.5-3с тишины перед
# разоблачением" из продакшн-разбора, см. CLIMAX_DIP_* в pipeline_smart.py);
# понижена до 1.20с по независимому research-обзору пауз в видео-нарративе
# (Campione&Véronis 2002, Krivokapić 2007, Gold et al. 2017 — см. .md-таблицу
# "Рабочие диапазоны пауз", правило 3: "1.40с допустима только как крупная
# граница [смена главы]", для одиночного reveal — 1.05-1.35с, старт 1.20с).
# Тот обзор сам явно называет свои цифры "консервативной стартовой
# калибровкой, не нейробиологическим законом" — 1.20 здесь тоже принят как
# такая калибровка, не как измеренное на этом канале значение. Без выхода за
# словарь допустимых тегов — сам [pause]=0.8 остаётся ЗАЯВЛЕННОЙ длиной тега,
# а 0.9-1.2 здесь — то, чего Cadence Validator позволяет ДОСТИЧЬ монтажной
# обрезкой (fix_pauses.py), не то, что просит от диктора текстовым тегом
# сверх словаря.
RHETORICAL_RANGES = {
    "reveal_hold":    (0.90, 1.20),
    "evidence_beat":  (0.35, 0.55),
    "anticipation":   (0.50, 0.85),
    "closing_hold":   (0.70, 1.10),
    "punchy":         (0.15, 0.35),
    "question_rise":  (0.30, 0.55),
}
RHETORICAL_MIN_TAG = {
    "reveal_hold": "[pause]",
    "closing_hold": "[pause]",
    "anticipation": "[pause]",
    "evidence_beat": "[short pause]",
}
_TAG_STRENGTH = {None: 0, "[short pause]": 1, "[pause]": 2}

DASH_END_RE = re.compile(r'[:—\-]\s*$')   # двоеточие/тире/em-dash у конца фразы


def classify_unit(block, idx, blocks):
    """Риторическая роль ОДНОГО блока (см. parse_blocks() в pipeline_smart.py —
    та же единица, что уже режет монтаж: текст между [pause]/[short pause]/
    границами секций). Приоритет — от самого драматургически весомого сигнала
    к нейтральному: climax и evidence (stat) важнее структурной пунктуации,
    та важнее позиционных эвристик (хук/финал), самый нейтральный случай —
    обычная связка, наследующая диапазон от уже проставленного тега."""
    text = block["text"].strip()
    section = block["section"]

    # "Тишина как акцент ПЕРЕД разоблачением" — уже установленная в этом
    # пайплайне конвенция (см. CLIMAX_DIP_LEAD_SEC в pipeline_smart.py:
    # музыкальный провал начинается ДО t_climax, не после — t_climax там
    # это старт climax-блока). Значит важна ХВОСТОВАЯ пауза блока, который
    # идёт ПЕРЕД climax-блоком (она ведёт в разоблачение) — не хвостовая
    # пауза самой climax-строки (та пауза идёт уже ПОСЛЕ того, как фраза
    # сказана, держать её ради драмы поздно). build_timeline() в
    # speech_validator.py защищает ХВОСТОВУЮ паузу юнита, помеченного
    # reveal_hold — раньше эта роль ошибочно вешалась на сам climax-блок
    # (block.get("is_climax")), из-за чего план защищал паузу ПОСЛЕ
    # разоблачения вместо паузы ПЕРЕД ним. Реальный баг, пойман при
    # эмпирической проверке "действительно ли пауза стала умнее" —
    # см. git log этого коммита.
    if idx + 1 < len(blocks) and blocks[idx + 1].get("is_climax"):
        return "reveal_hold"
    if block.get("stat"):
        return "evidence_beat"
    if section.startswith("FINAL"):
        # Последний блок эпизода — мысль не должна обрываться (буквальная
        # формулировка задачи). "Последний" — по факту следующего блока: либо
        # его нет вообще, либо следующий уже из другой (несуществующей после
        # FINAL) секции.
        is_last = (idx == len(blocks) - 1) or (blocks[idx + 1]["section"] != section)
        if is_last:
            return "closing_hold"
    if text.endswith("?"):
        return "question_rise"
    if DASH_END_RE.search(text):
        return "anticipation"
    if section.startswith("HOOK") and block["words"] <= 6:
        return "punchy"
    return "connective"


def target_range_for(block, kind):
    if kind == "connective":
        tag = _dominant_tag(block)
        return TAG_BASE_RANGE.get(tag, TAG_BASE_RANGE[None])
    return RHETORICAL_RANGES[kind]


def _dominant_tag(block):
    """Какой из двух исполняемых тегов дал эту паузу — по значению
    pause_after by pipeline_smart.PAUSE_DURATIONS (0.8 -> [pause], 0.4 ->
    [short pause], 0.0 -> нет тега/граница секции)."""
    pa = block.get("pause_after", 0.0)
    if abs(pa - pipeline_smart.PAUSE_DURATIONS["[pause]"]) < 1e-6:
        return "[pause]"
    if abs(pa - pipeline_smart.PAUSE_DURATIONS["[short pause]"]) < 1e-6:
        return "[short pause]"
    return None


def tag_suggestion(block, kind):
    """Если риторическая роль требует тег НЕ СЛАБЕЕ определённого
    (RHETORICAL_MIN_TAG), а у блока сейчас тег слабее (или отсутствует) —
    вернуть рекомендованный тег для аннотированного вывода. None, если
    текущий тег уже не слабее рекомендованного (менять нечего)."""
    min_tag = RHETORICAL_MIN_TAG.get(kind)
    if min_tag is None:
        return None
    current = _dominant_tag(block)
    if _TAG_STRENGTH.get(current, 0) < _TAG_STRENGTH[min_tag]:
        return min_tag
    return None


# --- Speech Direction Engine, Stage B-подготовка: план ПОДАЧИ ПО ГЛАВАМ,
# а не только по отдельным юнитам. Один "chapter" = одна секция script.txt
# (HOOK / BLOCK N / FINAL) — та же граница, что уже режет alignment.csv
# (см. load_alignment_weights() в pipeline_smart.py: "по одному файлу на
# КАЖДУЮ секцию"). Внутри BLOCK-секций arc_stage — прямая проекция уже
# документированной структуры блока (ЧАСТЬ 9 CLAUDE.md: "заход-якорь →
# постановка → слом → доказательство → вывод → перенос на зрителя →
# мостик") на позицию юнита внутри секции (0..1 по номеру юнита, не по
# словам — блок из многих коротких punchy-юнитов не должен схлопывать
# все стадии в первые 10% просто потому что слов мало). HOOK/FINAL — не
# 7-стадийная структура блока (в них другая, отдельно описанная логика:
# 5 рычагов хука, "честный финал" без триад) — один stage на всю секцию.
#
# tempo_mult/energy_target — ПРИНЦИПИАЛЬНЫЕ дефолты, выведенные из текста
# CLAUDE.md ("слом" — резче, "доказательство"/"вывод" — спокойные
# объяснения, FINAL — тише), НЕ откалиброваны на реальном сгенерированном
# аудио (в отличие от, например, CLIP_RELEVANCE_THRESHOLD в
# pipeline_smart.py, которая калибрована на измеренных живых парах) —
# честно так и помечено, чтобы не выдавать принцип за измерение.
CHAPTER_ARC_STAGES_HOOK = ("hook",)
CHAPTER_ARC_STAGES_FINAL = ("final",)
CHAPTER_ARC_STAGES_BLOCK = (
    "заход-якорь", "постановка", "слом", "доказательство",
    "вывод", "перенос-на-зрителя", "мостик",
)
# Правая граница накопительной доли юнитов секции для каждой стадии.
_BLOCK_STAGE_BOUNDARIES = [
    ("заход-якорь", 0.10), ("постановка", 0.25), ("слом", 0.40),
    ("доказательство", 0.70), ("вывод", 0.85), ("перенос-на-зрителя", 0.95),
    ("мостик", 1.01),   # >1.0 — подчищает последний юнит от округления
]
ARC_STAGE_PROFILE = {
    "hook":              {"tempo_mult": 1.05, "energy_target": "high"},
    "заход-якорь":       {"tempo_mult": 1.00, "energy_target": "med"},
    "постановка":        {"tempo_mult": 0.95, "energy_target": "med"},
    "слом":              {"tempo_mult": 1.10, "energy_target": "high"},
    "доказательство":    {"tempo_mult": 0.95, "energy_target": "med"},
    "вывод":             {"tempo_mult": 0.95, "energy_target": "med"},
    "перенос-на-зрителя": {"tempo_mult": 1.00, "energy_target": "med-high"},
    "мостик":            {"tempo_mult": 1.00, "energy_target": "med"},
    "final":             {"tempo_mult": 0.90, "energy_target": "low"},
}
BASE_WPM = 125.0   # ЧАСТЬ 9 CLAUDE.md — спокойный закадр, без тегов


def _section_kind(section):
    if section.startswith("HOOK"):
        return "hook"
    if section.startswith("FINAL"):
        return "final"
    return "block"


def _arc_stage_for_position(frac, kind):
    if kind == "hook":
        return "hook"
    if kind == "final":
        return "final"
    for name, bound in _BLOCK_STAGE_BOUNDARIES:
        if frac <= bound:
            return name
    return "мостик"


def assign_chapter_arcs(units):
    """Добавляет к каждому unit: chapter_id (номер секции по порядку
    первого появления — та же нумерация, что media_plan/alignment/NN.csv),
    arc_stage, tempo_mult, energy_target. Мутирует units на месте и
    возвращает их же (для удобства цепочки вызовов)."""
    chapter_order = []
    chapter_units = {}
    for u in units:
        sec = u["section"]
        if sec not in chapter_units:
            chapter_order.append(sec)
            chapter_units[sec] = []
        chapter_units[sec].append(u)

    chapter_id_of = {sec: i for i, sec in enumerate(chapter_order)}
    for sec, sec_units in chapter_units.items():
        kind = _section_kind(sec)
        n = len(sec_units)
        for idx, u in enumerate(sec_units):
            frac = (idx + 1) / n
            stage = _arc_stage_for_position(frac, kind)
            profile = ARC_STAGE_PROFILE[stage]
            u["chapter_id"] = chapter_id_of[sec]
            u["arc_stage"] = stage
            u["tempo_mult"] = profile["tempo_mult"]
            u["energy_target"] = profile["energy_target"]
            u["target_wpm"] = round(BASE_WPM * profile["tempo_mult"], 1)
    return units


def build_units(blocks):
    units = []
    for i, b in enumerate(blocks):
        kind = classify_unit(b, i, blocks)
        rng = target_range_for(b, kind)
        suggestion = tag_suggestion(b, kind)
        # "#", не ":" — секция сама может содержать двоеточие ("BLOCK 1: Вес
        # меча"), unit_id с двумя ":" был бы двусмысленным, если его когда-
        # нибудь понадобится разобрать обратно.
        unit_id = f"{b['section']}#{sum(1 for k in range(i) if blocks[k]['section'] == b['section'])}"
        units.append({
            "unit_id": unit_id,
            "index": i,
            "section": b["section"],
            "text": b["text"],
            "words": b["words"],
            "tag": _dominant_tag(b),
            "rhetorical_kind": kind,
            "target_range_sec": list(rng),
            "protected": kind != "connective",
            "tag_suggestion": suggestion,
            "stat": b.get("stat"),
            "stat_word_pos": b.get("stat_word_pos"),
            "is_climax": bool(b.get("is_climax")),
        })
    return units


def validate_plan(plan):
    """Строгая проверка структуры/диапазонов — рендерер (fix_pauses.py/
    speech_validator.py) НИКОГДА не доверяет speech_plan.json без прогона
    через эту функцию, тот же принцип, что уже применяют auto_levels_params/
    auto_wb_params к любой внешней коррекции: клэмп и проверка типов, не
    слепое доверие. Бросает ValueError с точным описанием на первом же
    нарушении — тихого "почти правильного" плана в рендер не пускаем."""
    if not isinstance(plan, dict) or "units" not in plan:
        raise ValueError("speech_plan: нет ключа 'units'")
    seen_ids = set()
    for u in plan["units"]:
        for key in ("unit_id", "section", "text", "rhetorical_kind", "target_range_sec", "protected"):
            if key not in u:
                raise ValueError(f"speech_plan: у unit нет обязательного поля '{key}': {u}")
        if u["unit_id"] in seen_ids:
            raise ValueError(f"speech_plan: дублирующийся unit_id {u['unit_id']!r}")
        seen_ids.add(u["unit_id"])
        if u["rhetorical_kind"] not in set(RHETORICAL_RANGES) | {"connective"}:
            raise ValueError(f"speech_plan: неизвестный rhetorical_kind {u['rhetorical_kind']!r}")
        rng = u["target_range_sec"]
        if (not isinstance(rng, list) or len(rng) != 2
                or not all(isinstance(x, (int, float)) for x in rng)
                or rng[0] < 0 or rng[1] < rng[0] or rng[1] > 5.0):
            raise ValueError(f"speech_plan: невалидный target_range_sec у {u['unit_id']}: {rng}")
        if not isinstance(u["protected"], bool):
            raise ValueError(f"speech_plan: 'protected' должен быть bool у {u['unit_id']}")
        tag_sug = u.get("tag_suggestion")
        if tag_sug is not None and tag_sug not in ALLOWED_TAGS:
            raise ValueError(f"speech_plan: tag_suggestion {tag_sug!r} вне разрешённого словаря ({ALLOWED_TAGS})")
        # chapter_id/arc_stage — опциональны В ЧТЕНИИ (план, записанный ДО
        # этого поля, не должен внезапно перестать проходить валидацию),
        # но если поле есть — обязано быть валидным, не тихо неправильным.
        if "arc_stage" in u:
            all_stages = (set(CHAPTER_ARC_STAGES_HOOK) | set(CHAPTER_ARC_STAGES_FINAL)
                          | set(CHAPTER_ARC_STAGES_BLOCK))
            if u["arc_stage"] not in all_stages:
                raise ValueError(f"speech_plan: неизвестный arc_stage {u['arc_stage']!r} у {u['unit_id']}")
        if "tempo_mult" in u and not (0.5 <= u["tempo_mult"] <= 2.0):
            raise ValueError(f"speech_plan: невалидный tempo_mult у {u['unit_id']}: {u['tempo_mult']}")
    return True


TTS_MODEL_ENV = "TTS_MODEL"
SSML_BREAK_MODELS = {"eleven_multilingual_v2", "eleven_flash_v2", "eleven_flash_v2_5"}


def render_annotated_text(units):
    """media_plan/speech_plan_annotated.txt — тот же voiced-текст, что уходит
    в TTS, с построчными режиссёрскими пометками ПОСЛЕ каждого юнита, где
    план расходится с текущим тегом (tag_suggestion) или несёт риторическую
    роль, о которой стоит знать диктору/редактору сценария. НЕ переписывает
    script.txt — только человекочитаемая накладка для ручной сверки перед
    отправкой в TTS (тот же ручной шаг 6, что и сейчас, только с
    подсказкой, а не вслепую)."""
    lines = []
    for u in units:
        lines.append(f"[{u['unit_id']}] {u['text']}")
        notes = []
        if u.get("arc_stage"):
            notes.append(f"глава {u['chapter_id']}/{u['arc_stage']} "
                         f"(темп x{u['tempo_mult']:.2f} ~{u['target_wpm']:.0f}сл/мин, "
                         f"энергия {u['energy_target']})")
        if u["rhetorical_kind"] != "connective":
            notes.append(f"роль: {u['rhetorical_kind']} (цель {u['target_range_sec'][0]:.2f}"
                         f"-{u['target_range_sec'][1]:.2f}с)")
        if u["tag_suggestion"]:
            notes.append(f"рекомендован тег {u['tag_suggestion']} "
                         f"(сейчас: {u['tag'] or 'нет тега'})")
        if notes:
            lines.append("    ↳ " + "; ".join(notes))
    return "\n".join(lines) + "\n"


def render_ssml_text(units, model):
    """media_plan/speech_plan_ssml.txt — та же аннотация, но в виде
    <break time="x.xs" /> на границах юнитов, для моделей ElevenLabs с
    документированной поддержкой SSML-breaks (Multilingual v2/Flash v2/
    Flash v2.5 — ПРОВЕРИТЬ актуальную документацию ElevenLabs перед
    использованием в проде, эта функция не обращается к сети и не может
    сама подтвердить, что синтаксис до сих пор поддерживается). Для модели
    по умолчанию этого канала (Eleven v3) эта функция не вызывается —
    v3 SSML-breaks не поддерживает, см. докстринг модуля."""
    parts = []
    for u in units:
        parts.append(u["text"])
        dur = u["target_range_sec"][1]   # верхняя граница диапазона — то, что реально нужно достичь
        if dur > 0.05:
            parts.append(f'<break time="{min(dur, 3.0):.2f}s" />')
    return " ".join(parts) + "\n"


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    script_path = os.path.join(video_dir, "script.txt")
    if not os.path.exists(script_path):
        print(f"script.txt не найден: {script_path}")
        return 1

    blocks = pipeline_smart.parse_blocks(script_path)
    if not blocks:
        print("Сценарий пуст/не распознан")
        return 1

    units = build_units(blocks)
    assign_chapter_arcs(units)
    plan = {"units": units}
    try:
        validate_plan(plan)
    except ValueError as e:
        print(f"Speech Planner: план не прошёл валидацию — {e}")
        return 1

    plan_dir = os.path.join(video_dir, "media_plan")
    os.makedirs(plan_dir, exist_ok=True)
    atomic_write_json(os.path.join(plan_dir, "speech_plan.json"), plan)

    with open(os.path.join(plan_dir, "speech_plan_annotated.txt"), "w", encoding="utf-8") as f:
        f.write(render_annotated_text(units))

    model = os.environ.get(TTS_MODEL_ENV, "").strip()
    if model in SSML_BREAK_MODELS:
        with open(os.path.join(plan_dir, "speech_plan_ssml.txt"), "w", encoding="utf-8") as f:
            f.write(render_ssml_text(units, model))
        print(f"  SSML-вариант ({model}): media_plan/speech_plan_ssml.txt")

    kinds = {}
    suggestions = 0
    for u in units:
        kinds[u["rhetorical_kind"]] = kinds.get(u["rhetorical_kind"], 0) + 1
        if u["tag_suggestion"]:
            suggestions += 1
    print(f"Speech Planner: {len(units)} юнитов — " +
          ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    if suggestions:
        print(f"  {suggestions} рекомендация(й) по тегам — см. media_plan/speech_plan_annotated.txt "
              f"перед озвучкой.")
    print(f"Готово: {os.path.join(plan_dir, 'speech_plan.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
