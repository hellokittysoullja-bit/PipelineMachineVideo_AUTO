#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Предпросмотр сценария: что будет на экране и что молча не сработает.

    python scripts/script_preflight.py <video_dir|script.txt>

Бесплатно, локально, без сети и без аудио. Шаг 2.5 протокола (CLAUDE.md
ЧАСТЬ 13): до озвучки, все предупреждения исправить.

Зачем. Экранные элементы (название после хука, заставки глав, плашки
[stat:], подпись места и года, карточка цитаты) решаются эвристиками по
тексту, и каждая честно молчит, когда не уверена. Снаружи «не уверена» и
«так и задумано» неотличимы, а узнать про пропуск можно было только после
рендера. Живой случай эп.04: карточка цитаты Юниуса не появилась (имя в
начале предложения), подпись места ушла на «ГЕРМАНИЯ» вместо «БАМБЕРГ».

Решает ТЕ ЖЕ функции, что рендер: script_parser.parse_blocks,
chapter_card (название, заставки, вёрстка), place_year + screen_text
(подписи, цитаты), is_climax блоков. Своей копии правил здесь нет: копия
разошлась бы с рендером, и предпросмотр показывал бы не то, что выйдет.

Время фраз — ОЦЕНКА (125 слов/мин + паузы тегов): по ней видно только
расписание («подпись ближе 40 с к прошлой — пропустится»). Что найдено во
фразе — точно то, что найдёт рендер; когда именно встанет — решат реальные
онсеты озвучки.

Уровни: ERROR — экранный элемент не сработает или сработает неверно (код
возврата 1); WARN — подозрительно для TTS/ритма, решает автор.
"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chapter_card  # noqa: E402
import place_year  # noqa: E402
import screen_text  # noqa: E402
from script_parser import PAUSE_DURATIONS, parse_blocks  # noqa: E402

WPM = 125.0                    # ЧАСТЬ 9 CLAUDE.md
LONG_BLOCK_WORDS = 60          # блок длиннее — без [pause] TTS тянет и монтаж стоит
DATES_PER_MIN_WARN = 4.0       # больше дат на минуту секции — зритель теряет нить
STAT_PREVIEW_SIZE = 96         # худший случай плашки — вариант 1, крупно по центру
STAT_MAX_WIDTH = 1920 - 2 * 96
# pipeline_smart.SUBCUT_MIN_SOURCE_DUR: фраза короче не режется на под-кадры,
# то есть заставка или плашка на ней занимает её ЦЕЛИКОМ (тест держит равенство).
SUBCUT_MIN_SOURCE_DUR = 8.0
GAP_NEAR_SEC = 15.0            # подпись в пределах стольких секунд от порога 40 с — «может пропуститься»
PIPELINE_TAG_RE = re.compile(r"^(stat:.*|climax|sfx:.*|hush|shot:.*|amb:.*|music:.*)$", re.S)
QUOTE_FONT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "assets", "fonts", "Montserrat-MediumItalic.ttf")


def _script_path(arg):
    return os.path.join(arg, "script.txt") if os.path.isdir(arg) else arg


def _speech_sections(raw):
    """[(ИМЯ, тело)] озвучиваемых секций — тот же отбор, что parse_blocks."""
    parts = re.split(r"===\s*(.*?)\s*===", raw)
    return [(parts[i].upper(), parts[i + 1] if i + 1 < len(parts) else "")
            for i in range(1, len(parts), 2)
            if parts[i].upper().startswith(("HOOK", "BLOCK", "FINAL"))]


def _est_times(blocks):
    """Оценочные начало/конец каждой фразы (сек)."""
    starts, ends, t = [], [], 0.0
    for b in blocks:
        starts.append(t)
        t += b["words"] / WPM * 60.0
        ends.append(t)
        t += float(b.get("pause_after") or 0.0)
    return starts, ends


def _mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _short(text, n=90):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _check_tags(sections, report):
    known = set(PAUSE_DURATIONS)
    for name, body in sections:
        for m in re.finditer(r"\[([^\]]*)\]", body):
            tag, inner = m.group(0), m.group(1)
            if tag in known or PIPELINE_TAG_RE.match(inner):
                continue
            if tag.lower() in known or PIPELINE_TAG_RE.match(inner.lower()):
                report.error(f"{name[:28]}: тег {tag} в неверном регистре — "
                             + ("TTS-тег узнаётся только строчными, пауза пропадёт"
                                if tag.lower() in known else "пишется строчными"),
                             hint=f"заменить на {tag.lower()}")
            elif inner.strip().lower() == "long pause":
                report.error(f"{name[:28]}: [long pause] запрещён (артефакты TTS, ЧАСТЬ 10)",
                             hint="[pause] или две фразы")
            else:
                report.error(f"{name[:28]}: неизвестный тег {tag} — будет вырезан без паузы",
                             hint="разрешены: " + ", ".join(sorted(known))
                             + ", [stat:], [climax], [sfx:], [hush], [shot:], [amb:], [music:]")


def _check_energetic(sections, report):
    where = [name for name, body in sections for _ in re.finditer(r"\[energetic\]", body)]
    if len(where) != 1 or not where[0].startswith("HOOK"):
        report.warn(f"[energetic] встречается {len(where)} раз"
                    + (f" (в {', '.join(sorted(set(w[:20] for w in where)))})" if where else "")
                    + " — нужен ровно один, в начале хука (ЧАСТЬ 10)")


def _check_dashes(sections, report):
    for name, body in sections:
        spoken = re.sub(r"\[[^\]]*\]", " ", body)
        dashes = [m.start() for m in re.finditer(r"\s[—–]\s", spoken)]
        ell = spoken.count("…") + spoken.count("...")
        if dashes:
            ex = _short(spoken[max(0, dashes[0] - 40):dashes[0] + 40], 80)
            report.warn(f"{name[:28]}: тире « — » {len(dashes)} раз — TTS делает на нём паузу "
                        f"(пример: «{ex}»)", hint="запятая или две фразы")
        if ell:
            report.warn(f"{name[:28]}: многоточие {ell} раз — длинная пауза TTS (ЧАСТЬ 9)")


def _check_dates(blocks, report):
    by_sec = {}
    for b in blocks:
        s = by_sec.setdefault(b["section"], [0, 0])
        s[0] += b["words"]
        s[1] += len(place_year.years_in(b["text"]))
    for sec, (words, dates) in by_sec.items():
        minutes = words / WPM
        if dates >= 3 and minutes > 0 and dates / minutes > DATES_PER_MIN_WARN:
            report.warn(f"{sec[:28]}: {dates} дат на ~{minutes:.1f} мин — "
                        f"{dates / minutes:.1f}/мин, зритель теряет нить")


class Report:
    def __init__(self):
        self.lines, self.errors, self.warnings = [], [], []

    def section(self, title):
        self.lines.append("")
        self.lines.append(f"== {title}")

    def info(self, text):
        self.lines.append(f"   {text}")

    def error(self, text, hint=None):
        self.errors.append(text + (f"  -> {hint}" if hint else ""))

    def warn(self, text, hint=None):
        self.warnings.append(text + (f"  -> {hint}" if hint else ""))

    def render(self):
        out = list(self.lines)
        out.append("")
        if self.errors:
            out.append(f"== ОШИБКИ ({len(self.errors)}) — экранный элемент не сработает или сработает неверно")
            out += [f" ✗ {e}" for e in self.errors]
        if self.warnings:
            out.append(f"== ПРЕДУПРЕЖДЕНИЯ ({len(self.warnings)})")
            out += [f" ! {w}" for w in self.warnings]
        if not self.errors and not self.warnings:
            out.append("== Предупреждений нет.")
        return "\n".join(out)


def preflight(script_path, font_path=None):
    """Report по сценарию. Ничего не пишет на диск и не ходит в сеть."""
    rep = Report()
    font = font_path or chapter_card.default_font()
    raw = open(script_path, encoding="utf-8").read()
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = parse_blocks(script_path)
    sections = _speech_sections(raw)
    if not blocks:
        rep.error("озвучиваемых секций (HOOK/BLOCK/FINAL) нет — сценарий пуст")
        return rep
    for b in blocks:
        b.setdefault("parent_text", b["text"])
    for k, b in enumerate(blocks):
        b["orig_index"] = k
    starts, ends = _est_times(blocks)
    rep.info(f"Сценарий: {script_path}")
    rep.info(f"Фраз: {len(blocks)}, слов: {sum(b['words'] for b in blocks)}, "
             f"оценка длительности {_mmss(ends[-1])} (125 слов/мин + паузы)")

    # --- название и заставки глав -------------------------------------------
    sec_names = [b["section"] for k, b in enumerate(blocks)
                 if k == 0 or b["section"] != blocks[k - 1]["section"]]
    title_sec = chapter_card.hook_title_section(sec_names)
    meta = chapter_card._metadata(script_path)
    film = chapter_card.film_title(script_path, font)
    rep.section("Название ролика после хука")
    if film and title_sec:
        size, lines, _ok = chapter_card.title_layout(film.upper(), font, chapter_card.TITLE)
        rep.info(f"«{film.upper()}» перед «{title_sec}», кегль {size}: " + " / ".join(lines))
    elif not title_sec:
        rep.info("нет: после хука нет главы с названием")
    elif meta.get("TITLE_CARD") or meta.get("TITLE"):
        rep.error(f"название «{meta.get('TITLE_CARD') or meta.get('TITLE')}» не влезает в две строки "
                  f"(и часть до « — » тоже) — заставки названия не будет",
                  hint="METADATA TITLE_CARD: короче")
    else:
        rep.warn("в METADATA нет TITLE/TITLE_CARD — заставки названия не будет")

    rep.section("Заставки глав")
    for name in chapter_card.card_sections(sec_names):
        if film and name == title_sec:
            rep.info(f"{name[:40]}: заменена названием ролика")
            continue
        title = chapter_card.section_title(name)
        size, lines, ok = chapter_card.title_layout(title.upper(), font, chapter_card.CHAPTER)
        rep.info(f"«{title.upper()}» кегль {size}: " + " / ".join(lines) + ("" if ok else "  [НЕ ВЛЕЗАЕТ]"))
        if not ok:
            rep.error(f"заголовок главы «{title}» не влезает в 2 строки даже кеглем {size}",
                      hint="короче название в === BLOCK N: … ===")

    # --- плашки ----------------------------------------------------------------
    rep.section("Плашки [stat:]")
    stats = [(k, b) for k, b in enumerate(blocks) if b.get("stat")]
    if not stats:
        rep.info("нет")
    for k, b in stats:
        text = str(b["stat"]).strip()
        w = chapter_card._text_width(text.upper(), STAT_PREVIEW_SIZE, font)
        rep.info(f"{_mmss(starts[k])} «{text.upper()}» — {_short(b['text'], 70)}")
        if not text:
            rep.error(f"пустая плашка [stat:] во фразе «{_short(b['text'], 60)}»")
        elif w > STAT_MAX_WIDTH:
            rep.error(f"плашка «{text}» шире кадра в варианте «крупно по центру», который выбирается "
                      f"по контрасту кадра ({w:.0f} > {STAT_MAX_WIDTH}px)",
                      hint="2-3 слова, число + единица")

    # --- место и год -------------------------------------------------------------
    confirm = place_year.script_words(raw)
    names = screen_text.script_names(raw)
    known_places = set()
    for b in blocks:
        known_places |= {p for p, _a, _b in place_year.places_in(b["text"], confirm)}
    rep.section("Подписи места и года")
    # Занято целиком (одна фраза = один кадр): первая фраза главы с
    # заставкой и фраза с плашкой, если она короче порога нарезки.
    card_starts = {i for i in chapter_card.card_boundaries(blocks)}
    busy = {k for k, b in enumerate(blocks)
            if (k in card_starts or b.get("stat")) and ends[k] - starts[k] < SUBCUT_MIN_SOURCE_DUR}
    caps = screen_text.plan_place_captions(blocks, starts, ends, busy, confirm)
    found_any, last_cap = False, None
    for k, b in enumerate(blocks):
        got = place_year.place_and_year(b["text"], confirm)
        years = place_year.years_in(b["text"])
        if got:
            found_any = True
            if k in busy:
                sched = "  [не будет: фраза целиком под заставкой главы/плашкой]"
            elif k not in caps:
                sched = "  [пропустится: ближе 40 с к прошлой подписи или фраза короче 2 с — по оценке]"
            else:
                gap = caps[k]["start"] - last_cap if last_cap is not None else None
                sched = (f"  [может пропуститься: до прошлой подписи ~{gap:.0f} с при пороге "
                         f"{screen_text.PLACE_MIN_GAP_SEC:.0f} — решат реальные онсеты]"
                         if gap is not None and gap < screen_text.PLACE_MIN_GAP_SEC + GAP_NEAR_SEC else "")
                last_cap = caps[k]["start"]
            rep.info(f"{_mmss(starts[k])} {got[0]} {got[1]}{sched} — {_short(b['text'], 70)}")
            if place_year.is_region(got[0]):
                other = _named_places(b["text"], known_places, exclude=got[0])
                if other:
                    rep.error(f"подпись «{got[0]}» — страна, а во фразе есть {', '.join(sorted(other))}: "
                              f"«{_short(b['text'], 80)}»",
                              hint="«город X, …» или «в X, …» в той же фразе")
        elif years:
            other = _named_places(b["text"], known_places)
            if other:
                rep.warn(f"год есть, место {', '.join(sorted(other))} есть, подписи нет: "
                          f"«{_short(b['text'], 90)}»",
                          hint="место и год в одной фразе: «в 1626 году в Бамберге» / «город Бамберг»")
    if not found_any:
        rep.info("нет")

    # --- цитаты ------------------------------------------------------------------
    rep.section("Карточки цитат")
    quotes = screen_text.plan_quote_cards(blocks, starts, ends, busy, names)
    seen = set()
    for k in sorted(quotes):
        q = quotes[k]
        if (q["quote"], q["q0"]) in seen:
            continue
        seen.add((q["quote"], q["q0"]))
        rep.info(f"{_mmss(q['q0'])} «{_short(q['quote'], 80)}» — {q['author']}")
        if screen_text.quote_layout(q["quote"], QUOTE_FONT) is None:
            rep.error(f"цитата «{_short(q['quote'], 60)}» не влезает в {screen_text.QUOTE_MAX_LINES} "
                      f"строки — карточки не будет", hint="короче цитата")
    if not seen:
        rep.info("нет")
    for k, b in enumerate(blocks):
        if k in quotes:
            continue
        for m in re.finditer(r"«([^«»]{3,240})»", b["text"]):
            inner = m.group(1)
            if not _looks_like_speech(b["text"], m):
                continue
            got = screen_text.find_quote(b["text"],
                                         blocks[k - 1]["text"] if k else "",
                                         blocks[k + 1]["text"] if k + 1 < len(blocks) else "",
                                         names)
            if got:
                rep.warn(f"цитата «{_short(inner, 50)}» ({got[1]}) короче 2.5 с по оценке — "
                         f"карточки может не быть")
                continue
            rep.error(f"«ёлочки» похожи на речь, но автор не найден — карточки цитаты не будет: "
                      f"«{_short(b['text'], 100)}»",
                      hint="«Имя пишет: «…»» или «…», — писал Имя; пересказ — без «ёлочек»")

    # --- кульминация -------------------------------------------------------------
    rep.section("Кульминация [climax]")
    cl = [k for k, b in enumerate(blocks) if b.get("is_climax")]
    for k in cl:
        rep.info(f"{_mmss(starts[k])} {blocks[k]['section'][:24]} — {_short(blocks[k]['text'], 70)}")
    if not cl:
        rep.info("нет")
        rep.warn("в эпизоде нет [climax] — нет музыкального провала и акцента разоблачения")

    # --- текст для TTS -----------------------------------------------------------
    _check_tags(sections, rep)
    _check_energetic(sections, rep)
    _check_dashes(sections, rep)
    for k, b in enumerate(blocks):
        if b["words"] > LONG_BLOCK_WORDS:
            rep.warn(f"{b['section'][:24]}: фраза {b['words']} слов без [pause] — "
                     f"«{_short(b['text'], 60)}»", hint="[pause] каждые 2-3 предложения")
    _check_dates(blocks, rep)
    return rep


def _looks_like_speech(text, m):
    """«Ёлочки», которые автор, похоже, задумал как цитату: оформлены как
    прямая речь (та же проверка, что у рендера), или во фразе есть глагол
    речи. Названия («Молот ведьм») и кавычки-иронии («ведьм жгли в
    Средневековье») — нет: их короче screen_text.QUOTE_MIN_WORDS или они не
    оформлены речью и глагола речи рядом нет."""
    if len(m.group(1).split()) < screen_text.QUOTE_MIN_WORDS:
        return False
    if screen_text._quote_is_speech(text, m):
        return True
    # глагол речи — в том же предложении, что и «ёлочки»
    head = re.split(r"[.!?…]\s+", text[:m.start()])[-1]
    tail = re.split(r"(?<=[.!?…])\s+", text[m.end():])[0]
    verbs = "|".join(screen_text.SPEECH_VERBS)
    return bool(re.search(rf"\b(?:{verbs})\b", head + " " + tail))


def _named_places(text, known_places, exclude=None):
    """Известные сценарию места (найденные place_year где-либо в нём), чьё
    имя стоит в этой фразе в любой падежной форме — кроме стран."""
    out = set()
    words = set(re.findall(r"[А-ЯЁ][а-яё]+", text or ""))
    for p in known_places:
        if place_year.is_region(p) or (exclude and p.upper() == str(exclude).upper()):
            continue
        stem = p[:-1] if p[-1] in "аяьй" else p
        if len(stem) >= 4 and any(w.startswith(stem) and len(w) - len(stem) <= 3 for w in words):
            out.add(p.upper())
    return out


def print_preflight(script_path, header=True):
    """Печать отчёта; (ошибок, предупреждений). Для рендера — только печать,
    ничего не блокирует; любой сбой самого предпросмотра — строка, не обрыв."""
    try:
        rep = preflight(script_path)
    except Exception as e:  # noqa: BLE001 — предпросмотр не имеет права уронить рендер
        print(f"  script_preflight: не выполнен ({type(e).__name__}: {e})")
        return 0, 0
    if header:
        print("Предпросмотр экрана (script_preflight):")
    print(rep.render())
    return len(rep.errors), len(rep.warnings)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.split("\n\n")[0])
        print("usage: script_preflight.py <video_dir|script.txt>")
        return 2
    path = _script_path(argv[0])
    if not os.path.exists(path):
        print(f"нет файла: {path}")
        return 2
    errors, _warnings = print_preflight(path, header=False)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
