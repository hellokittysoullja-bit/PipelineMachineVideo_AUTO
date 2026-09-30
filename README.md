# DoodleExplainer_AUTO

Автоматический генератор роликов-объяснялок с рисованными кадрами (человечки, схемы, русские подписи).

```
cp config.example.env .env              # ключи
python scripts/list_models.py image     # выбрать IMAGE_MODEL
python scripts/make_video.py videos/01_tema --minutes 15               # план + цена
python scripts/make_video.py videos/01_tema --minutes 15 --confirm-spend  # картинки + сборка
```
Подробно — CLAUDE.md.
