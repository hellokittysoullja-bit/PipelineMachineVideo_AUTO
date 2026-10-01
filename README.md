# DoodleExplainer_AUTO

Автоматический генератор роликов-объяснялок с рисованными кадрами (человечки, схемы, русские подписи).
Стоит на проверенном коде PipelineMachineVideo_AUTO (планировщик, судья, тайминг, звук) — и только на нём.

```
cp config.example.env .env                     # ключ шлюза и модель картинок
# look/style/ — 1-3 образца стиля, look/hero.png — герой (необязательно)
python scripts/make_video.py videos/01_tema --minutes 15 --confirm-spend   # план, кадры, сборка
python scripts/make_video.py videos/01_tema --minutes 15 --tts    # + озвучка Lumean (платно)
```
Подробно — CLAUDE.md. Цены моделей картинок — python scripts/list_models.py image.
