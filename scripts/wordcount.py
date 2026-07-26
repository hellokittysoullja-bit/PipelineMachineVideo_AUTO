#!/usr/bin/env python3
"""Контроль длины сценария перед озвучкой (ЧАСТЬ 9).
Usage: python scripts/wordcount.py <script.txt> [T_минут]
Считает ЧИСТЫЕ слова только в секциях HOOK/BLOCK*/FINAL (без тегов [...],
без === заголовков, без служебных секций METADATA/PEXELS/IMAGE/ANALYSIS/...)."""
import re
import sys

WPM = 125.0


def main():
    if len(sys.argv) < 2:
        print("Usage: wordcount.py <script.txt> [T_минут]")
        return
    path = sys.argv[1]
    T = float(sys.argv[2]) if len(sys.argv) > 2 else None
    section = None
    count = 0
    def clean_words(s):
        return len(re.sub(r'\[.*?\]', '', s).split())

    for line in open(path, encoding="utf-8"):
        m = re.match(r'===\s*(.*?)\s*===\s*(.*)$', line.strip())
        if m:
            section = m.group(1).upper()
            # Текст может стоять на ОДНОЙ строке с заголовком — именно так
            # выглядит формат script.txt в CLAUDE.md (ЧАСТЬ 9). Без этого
            # весь сценарий считался за 0 слов, и проверка длины молчала.
            if section.startswith(("HOOK", "BLOCK", "FINAL")):
                count += clean_words(m.group(2))
            continue
        if not section:
            continue
        if section.startswith(("HOOK", "BLOCK", "FINAL")):
            count += clean_words(line)
    mins = count / WPM
    print(f"Слов в сценарии: {count}. Расчётная длительность: {mins:.1f} минут (при {WPM:.0f} слов/мин).")
    if T:
        lo, hi = T * WPM * 0.95, T * WPM * 1.07
        if count < lo:
            status = f"МАЛО — дописать до {lo:.0f}+ слов"
        elif count > hi:
            status = f"МНОГО — резать до <{hi:.0f} слов"
        else:
            status = "OK, в коридоре"
        print(f"Цель T={T:g} мин -> коридор {lo:.0f}-{hi:.0f} слов -> {status}")


if __name__ == "__main__":
    main()
