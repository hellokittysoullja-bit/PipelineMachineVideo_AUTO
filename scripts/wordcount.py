#!/usr/bin/env python3
"""Контроль длины сценария перед озвучкой (ЧАСТЬ 9).
Usage: python scripts/wordcount.py <script.txt> [T_минут]
Считает ЧИСТЫЕ слова только в секциях HOOK/BLOCK*/FINAL (без тегов [...],
без === заголовков, без служебных секций METADATA/PEXELS/IMAGE/ANALYSIS/...)."""
import os
import re
import sys

WPM = 125.0
# Теги пауз добавляют время ПОВЕРХ слов (ЧАСТЬ 10). Те же значения, что у
# сборщика в pipeline_smart.PAUSE_DURATIONS — иначе отчёт длины и реальная
# сборка расходятся.
PAUSE_SEC = {"[pause]": 0.8, "[short pause]": 0.4}


def clean_words(s):
    # Теги стоят СПЛОШНЯКОМ с текстом без пробелов (ЧАСТЬ 10 CLAUDE.md,
    # например "грамма.[pause]Так"), поэтому тег заменяем на ПРОБЕЛ, а не на
    # пустоту — иначе соседние слова склеиваются в одно и счётчик занижает
    # реальную длину сценария (тот самый критический класс отказа из
    # ЧАСТИ 1: недосчитанный сценарий -> неверная длина озвучки).
    return len(re.sub(r'=+', ' ', re.sub(r'\[.*?\]', ' ', s)).split())


def count_pause_seconds(text):
    """Сколько секунд добавят теги пауз. На 18-минутном сценарии их набирается
    150+ штук — это около двух минут сверху, то есть разница между «в коридоре»
    и «перебор», которую отчёт раньше не показывал вообще."""
    return sum(text.count(tag) * sec for tag, sec in PAUSE_SEC.items())


def count_words(path):
    """Чистые слова только в озвучиваемых секциях (HOOK/BLOCK*/FINAL).
    Вынесено из main() как отдельная функция — чтобы тестировать логику
    подсчёта без сборки sys.argv (поведение не менялось, тот же код)."""
    section = None
    count = 0
    pause_sec = 0.0
    for line in open(path, encoding="utf-8"):
        m = re.match(r'===\s*(.*?)\s*===\s*(.*)$', line.strip())
        if m:
            section = m.group(1).upper()
            # Текст может стоять на ОДНОЙ строке с заголовком — именно так
            # выглядит формат script.txt в CLAUDE.md (ЧАСТЬ 9). Без этого
            # весь сценарий считался за 0 слов, и проверка длины молчала.
            if section.startswith(("HOOK", "BLOCK", "FINAL")):
                count += clean_words(m.group(2))
                pause_sec += count_pause_seconds(m.group(2))
            continue
        if not section:
            continue
        if section.startswith(("HOOK", "BLOCK", "FINAL")):
            count += clean_words(line)
            pause_sec += count_pause_seconds(line)
    return count, pause_sec


def main():
    if len(sys.argv) < 2:
        print("Usage: wordcount.py <script.txt> [T_минут]")
        return 1
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Файл не найден: {path}")
        return 1
    T = None
    if len(sys.argv) > 2:
        try:
            T = float(sys.argv[2])
        except ValueError:
            print(f"T_минут должно быть числом, получено: {sys.argv[2]!r}")
            return 1
    count, pause_sec = count_words(path)
    mins = count / WPM
    # Коридор проверяется по ЧИСТЫМ словам (ЧАСТЬ 9) — это не меняется. Но
    # рядом показываем и реальный хронометраж с паузами: платная озвучка
    # оплачивается за то, что прозвучит, а не за голые слова.
    with_pauses = mins + pause_sec / 60.0
    print(f"Слов в сценарии: {count}. Расчётная длительность: {mins:.1f} минут (при {WPM:.0f} слов/мин).")
    if pause_sec:
        print(f"С учётом тегов пауз (+{pause_sec:.0f}с): {with_pauses:.1f} минут — "
              f"столько ролик будет идти на самом деле.")
    if T:
        lo, hi = T * WPM * 0.95, T * WPM * 1.07
        if count < lo:
            status = f"МАЛО — дописать до {lo:.0f}+ слов"
        elif count > hi:
            status = f"МНОГО — резать до <{hi:.0f} слов"
        else:
            status = "OK, в коридоре"
        print(f"Цель T={T:g} мин -> коридор {lo:.0f}-{hi:.0f} слов -> {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
