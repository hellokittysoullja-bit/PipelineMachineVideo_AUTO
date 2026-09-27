# Запуск на видеокарте (Lightning AI, T4 и новее)

Ветка `claude/gpu-parallel-ee6bh8`. Код тот же, что у основной ветки. Что именно изменено и почему выбор кадров от этого не меняется — в `CLAUDE.md`, раздел «Ветка под видеокарту».

## 1. Studio

1. Создать Studio с GPU (T4 достаточно; L4 быстрее).
2. В терминале Studio:

```bash
git clone <адрес репозитория> && cd PipelineMachineVideo_AUTO
git checkout claude/gpu-parallel-ee6bh8
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## 2. PyTorch с CUDA

В Studio он обычно уже стоит. Проверка:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Должно напечатать `True` и имя карты. Если `False` — поставить сборку torch с CUDA по команде с pytorch.org для своей версии CUDA.

## 3. Jina на видеокарте (необязательно)

Jina (вторая модель проверки) работает через onnxruntime. Для видеокарты нужен пакет с CUDA, и он заменяет обычный:

```bash
pip uninstall -y onnxruntime && pip install onnxruntime-gpu
```

Без этого шага Jina остаётся на процессоре, остальное работает на видеокарте.

## 4. ffmpeg с NVENC и drawtext

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
ffmpeg -hide_banner -encoders | grep nvenc      # должен быть hevc_nvenc
ffmpeg -hide_banner -filters | grep drawtext    # нужен для титров и плашек
```

Если в сборке ffmpeg нет NVENC, клипы кодируются прежним libx264 (`CLIP_ENCODER=auto` проверит это сам).

## 5. Ключи

Скопировать `config.example.env` в `.env` и вписать свои ключи. Файл `.env` останется на сервере Lightning — учитывайте это.

## 6. Запуск

Как обычно: `python scripts/render_episode.py videos/NN_название`. В начале лога появится строка:

```
Устройство моделей: cuda (Tesla T4); Jina: cuda; кодер клипов: nvenc
```

Если там `cpu`, значит torch видеокарту не увидел (см. шаг 2).

## Честные пределы

- **Прирост на T4 не измерен:** в среде, где писалась ветка, видеокарты нет. Оценка по замеру эпизода 98 — 5 ч → примерно 3–3.5 ч. Главное время уходит на ответы судьи и лимиты источников, а их видеокарта не ускоряет.
- **Кэши.** Числа моделей на видеокарте отличаются от процессорных примерно на 1e-6, поэтому их кэши и подпись отбора помечены отдельно. Первый прогон на видеокарте заново посчитает каскад и заново выберет кадры. Ответы судьи при этом берутся из его кэша.
- **Бесплатный Studio** (по сторонним обзорам, не проверено): перезапуск каждые 4 часа, около 22 часов T4 в месяц. Прерванный прогон продолжается с кэша повторным запуском.
