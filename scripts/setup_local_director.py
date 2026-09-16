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
import shutil
import subprocess
import sys

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
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=10)
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
        print("\nСНАЧАЛА: pip install llama-cpp-python")
        print("Без него модель запускать нечем.")
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
    print(f"  {py} scripts/shot_brief_director.py videos/NN_название \\")
    print(f"      --brain local --model {dest} --write-inline")
    print("\nОписания кадров встанут прямо в script.txt, и дальше сборка")
    print("пойдёт как обычно — ни плана, ни флагов не нужно.")
    print("\nЕсли эпизод НЕ про нишу этого канала — добавить SHOT_BRIEF_WORLD=off,")
    print("иначе объявленный мир канала будет диктовать кадры.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
