# DoodleExplainer_AUTO

Автоматический генератор роликов-объяснялок с рисованными кадрами (человечки, схемы, русские подписи).
Стоит на проверенном коде PipelineMachineVideo_AUTO (планировщик, судья, тайминг, звук) — и только на нём.

```
cp config.example.env .env                     # ключи, IMAGE_BACKEND (локальная модель через ComfyUI)
python scripts/make_video.py videos/01_tema --minutes 15          # план, кадры, сборка
python scripts/make_video.py videos/01_tema --minutes 15 --tts    # + озвучка Lumean (платно)
```
Подробно — CLAUDE.md; локальная модель — assets/comfyui/README.md.
