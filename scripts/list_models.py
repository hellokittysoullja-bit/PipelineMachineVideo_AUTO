#!/usr/bin/env python3
"""Модели шлюза с ценой: python scripts/list_models.py [image|chat] [фильтр]

Нужен, чтобы выбрать IMAGE_MODEL / TEXT_CHECK_MODEL по каталогу шлюза, а не
по памяти: каталог меняется, цены тоже."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel  # noqa: E402


def main():
    channel.load_env()
    import llm_gateway
    kind = sys.argv[1] if len(sys.argv) > 1 else "image"
    flt = (sys.argv[2] if len(sys.argv) > 2 else "").lower()
    gw = llm_gateway.Gateway()
    if not gw.configured:
        sys.exit("Нет LLM_GATEWAY_API_KEY в .env")
    cat = gw._request("GET", "/models", timeout=60)
    for m in sorted(cat.get("data", []), key=lambda m: m["id"]):
        b = m.get("billing") or {}
        is_image = b.get("unit") == "image"
        if (kind == "image") != is_image or (flt and flt not in m["id"].lower()):
            continue
        if is_image:
            c = (b.get("coefficient") or {}).get("output")
            price = "?" if c is None or "base_tokens" not in b else round(b["base_tokens"] * float(c))
            print(f"{m['id']:45s} цена картинки (базовая): {price}")
        else:
            c = b.get("coefficient") or {}
            vision = "зрение" if "image" in str((m.get("capabilities") or {})).lower() else ""
            print(f"{m['id']:45s} вход/выход за токен: {c.get('input')}/{c.get('output')} {vision}")


if __name__ == "__main__":
    main()
