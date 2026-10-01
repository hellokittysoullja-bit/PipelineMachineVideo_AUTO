#!/usr/bin/env python3
"""Модели шлюза с ценой: python scripts/list_models.py [image|chat] [фильтр]

Цена картинки — на каждом качестве и размере, по формуле шлюза (каталог
меняется, поэтому из каталога, а не по памяти)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: E402


def main():
    env.load_env()
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
            print(m["id"])
            scales = b.get("scales") or {}
            quals = [q for q in (scales.get("quality") or {}) if q in ("low", "medium", "high")] or [None]
            sizes = [z for z in (scales.get("size") or {}) if z in ("1024x1024", "1536x1024", "1792x1024")] or [None]
            for q in quals:
                row = "  ".join(f"{z or 'любой размер'}: {gw.image_cost(m['id'], z, q):,}" for z in sizes)
                print(f"    {q or 'любое качество':14s} {row}")
        else:
            c = b.get("coefficient") or {}
            vision = "зрение" if "image" in str((m.get("capabilities") or {})).lower() else ""
            print(f"{m['id']:45s} вход/выход за токен: {c.get('input')}/{c.get('output')} {vision}")


if __name__ == "__main__":
    main()
