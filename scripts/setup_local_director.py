# -*- coding: utf-8 -*-
"""Поставить локального режиссёра кадра: сам выберет модель под машину.

ЗАЧЕМ. Замена прежнему способу (один запрос на главу) для случая, когда
сессии нет: ночная сборка, офлайн, чужая машина. Замер на эпизоде 02,
116 юнитов, «в проде попаданий»:

    Qwen3-30B-A3B   69     12.4 ГБ, ~30 мин на эпизод
    Qwen3-4B-2507   63      2.3 ГБ, ~10 мин на эпизод
    ЗАПРОС СЕКЦИИ   58     как было
    пофразовый 7B   49     выключенный автомат

Обе модели обходят прежнее состояние. Двенадцать гигабайт при этом почти
не окупаются: разница 69 против 63, а весит модель в пять раз больше и
считает втрое дольше. Поэтому выбор по машине, а не «бери что больше».

ЧЕСТНО О ПРЕДЕЛЕ: Claude на том же харнессе даёт 85. Локальный путь —
для случая «без сессии», а не «вместо сессии».
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys

# Движок, которым крутится модель. Версия ЗАКРЕПЛЕНА и индекс указан явно —
# это не осторожность, а найденный блокер (16.09): на PyPI у
# llama-cpp-python лежит ТОЛЬКО sdist, ни одной готовой сборки, поэтому
# обычный `pip install llama-cpp-python` на Windows требует компилятор C++.
# Официальный индекс автора пакета готовые сборки содержит, но отстаёт от
# PyPI: последняя версия там 0.3.19 против 0.3.35 на PyPI.
#
# Что 0.3.19 подходит — ПРОВЕРЕНО, а не предположено: колесо под Windows
# скачано и распаковано, в llama.dll найдены строки архитектур `qwen3`
# (модель 4B) и `qwen3moe` (модель 30B-A3B). Обе модели ниже запустятся.
ENGINE_VERSION = "0.3.19"
ENGINE_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"


def engine_install_command(py=None, system=None):
    """Команда установки движка — РАЗНАЯ ПО СИСТЕМАМ, и это замер.

    Windows: обычный `pip install llama-cpp-python` требует компилятор C++
    (на PyPI лежит только sdist). Официальный индекс автора пакета отдаёт
    готовое колесо, и проверено распаковкой, что в его llama.dll есть
    архитектуры `qwen3` и `qwen3moe` — обе наши модели.

    Linux/macOS: тот же индекс НЕ подходит, и это поймано живой установкой,
    а не прочитано. Колесо оттуда собрано под musl (Alpine), и на обычном
    дистрибутиве импорт падает: `libc.musl-x86_64.so.1: cannot open shared
    object file`. Зато сборка из исходников с PyPI там проходит штатно
    (в этом контейнере так и стоит рабочая 0.3.35), компилятор есть почти
    везде. Поэтому здесь — обычный pip, без индекса и без закрепления.

    Одна команда на все системы была бы неверна ровно на одной из них.
    """
    py = py or os.path.basename(sys.executable)
    system = system or platform.system()
    if system == "Windows":
        return [f"{py} -m pip install llama-cpp-python=={ENGINE_VERSION} ^",
                f"    --extra-index-url {ENGINE_WHEEL_INDEX}"]
    return [f"{py} -m pip install llama-cpp-python"]

MODELS = {
    "30b": {
        "name": "Qwen3-30B-A3B-Instruct-2507-Q3_K_S.gguf",
        "url": ("https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
                "/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q3_K_S.gguf"),
        "gb": 12.4, "need_ram_gb": 15, "score": 69, "minutes": 30,
    },
    "4b": {
        "name": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "url": ("https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF"
                "/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        # Q4_K_M, а не менее сжатая — ЗАМЕРЕНО 16.09, не выбрано по вкусу:
        # та же модель в Q8_0 (4.0 ГБ) даёт РОВНО столько же попаданий
        # (73 против 73 на эпизоде 02, одно задание, packet_version=6),
        # а F16 (7.5 ГБ) отклонена по скорости — 13 минут на главу против
        # 50 секунд, то есть ~3 часа на эпизод. Сырые числа —
        # docs/quality/quant_q4_k_m.json и quant_q8_0.json.
        "gb": 2.3, "need_ram_gb": 6, "score": 63, "minutes": 10,
    },
}


def total_ram_gb():
    """Сколько памяти у машины. Без psutil — он тут лишняя зависимость."""
    try:                                  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    try:                                  # macOS
        # encoding задан явно: без него text=True берёт кодировку локали,
        # и вывод ломается там, где она не UTF-8 (правило аудита 04.09,
        # заперто tests/test_audit_fixes.py).
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=10)
        if out.returncode == 0:
            return int(out.stdout.strip()) / 2**30
    except Exception:
        pass
    try:                                  # Windows
        import ctypes

        class S(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = S(); st.dwLength = ctypes.sizeof(S)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullTotalPhys / 2**30
    except Exception:
        pass
    return None


def pick(ram_gb, forced=None):
    if forced:
        return forced
    if ram_gb is None:
        return "4b"          # не смогли измерить — берём ту, что влезет всюду
    return "30b" if ram_gb >= MODELS["30b"]["need_ram_gb"] else "4b"


def main(argv):
    ap = argparse.ArgumentParser(
        description="Поставить локального режиссёра кадра")
    ap.add_argument("--dir", default="models",
                    help="куда положить модель (по умолчанию ./models)")
    ap.add_argument("--model", choices=sorted(MODELS), default=None,
                    help="выбрать вручную, иначе по объёму памяти")
    ap.add_argument("--yes", action="store_true",
                    help="не спрашивать подтверждения перед скачиванием")
    a = ap.parse_args(argv[1:])

    ram = total_ram_gb()
    key = pick(ram, a.model)
    m = MODELS[key]
    print(f"Память машины: {ram:.1f} ГБ" if ram else "Память определить не удалось")
    print(f"Выбрана модель: {m['name']}")
    print(f"  вес {m['gb']} ГБ · попаданий {m['score']} из 116 · "
          f"эпизод примерно {m['minutes']} мин")
    print(f"  для сравнения: прежний способ 58, выключенный автомат 49, "
          f"Claude 85")

    try:
        import llama_cpp  # noqa: F401
        print("llama-cpp-python: установлен")
    except ImportError:
        print("\nСНАЧАЛА поставить движок. Команда — РОВНО ТАКАЯ:\n")
        for line in engine_install_command():
            print("  " + line)
        if platform.system() == "Windows":
            print("\nПочему не просто «pip install llama-cpp-python»: на PyPI")
            print("у этого пакета ЛЕЖАТ ТОЛЬКО ИСХОДНИКИ, и обычная установка")
            print("на Windows потребует компилятор C++ (Visual Studio Build")
            print("Tools). Индекс выше — официальный, автора того же пакета,")
            print(f"и проверено, что сборка {ENGINE_VERSION} знает архитектуры")
            print("qwen3 и qwen3moe, то есть обе модели ниже.")
        else:
            print("\nЗдесь собирается из исходников и это нормально: нужен")
            print("компилятор C (на Linux/macOS он почти всегда есть).")
            print("Готовые сборки из официального индекса тут НЕ подходят —")
            print("они под musl, и импорт падает на обычном дистрибутиве.")
        return 2

    os.makedirs(a.dir, exist_ok=True)
    dest = os.path.join(a.dir, m["name"])
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000_000:
        print(f"\nМодель уже на месте: {dest}")
    else:
        free_gb = shutil.disk_usage(a.dir).free / 2**30
        if free_gb < m["gb"] + 1:
            print(f"\nНа диске {free_gb:.1f} ГБ, нужно {m['gb'] + 1:.1f} ГБ.")
            return 2
        if not a.yes:
            ans = input(f"\nСкачать {m['gb']} ГБ в {dest}? [д/н] ").strip().lower()
            if ans not in ("д", "да", "y", "yes"):
                print("Отменено.")
                return 1
        print(f"Качаю {m['gb']} ГБ, это надолго...")
        rc = subprocess.call(["curl", "-L", "--progress-bar",
                              "-o", dest, m["url"]])
        if rc != 0 or not os.path.exists(dest):
            print("Скачать не удалось.")
            return 2

    py = os.path.basename(sys.executable)
    print("\n" + "=" * 62)
    print("ГОТОВО. Запускать на каждый эпизод так:\n")
    print(f"  {py} scripts/shot_brief_director.py videos/NN_название")
    print("\nБез единого флага: модель найдётся сама в этой папке, описания")
    print("кадров встанут прямо в script.txt, и дальше сборка пойдёт как")
    print("обычно. Перед записью делается .bak, брифы автора не трогаются.")
    if os.path.abspath(os.path.dirname(dest)) != os.path.join(
            os.path.abspath(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "models"):
        # Модель положили НЕ туда, где её ищет режиссёр: команда выше без
        # флага её не найдёт, и молчать об этом нельзя.
        print(f"\nМодель лежит вне models/ — добавь к команде: --model {dest}")
    print("\nЕсли эпизод НЕ про нишу этого канала — добавить SHOT_BRIEF_WORLD=off,")
    print("иначе объявленный мир канала будет диктовать кадры.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
