#!/usr/bin/env bash
# Подготовка пода для отбора кадров: пакеты, ffmpeg и веса моделей ОДНОВРЕМЕННО.
# Запускается из корня кода: `--workdir <папка> --prepare "bash scripts/pod_prepare.sh"`.
# Время оплачивается посекундно, поэтому три части идут параллельно и
# печатают своё время: узкое место видно сразу (обычно — сеть на веса,
# ~38 ГБ: на одном хосте 2.5 минуты, на другом 5+).
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
T0=$(date +%s)
say() { echo "[prepare +$(( $(date +%s) - T0 )) с] $*"; }

(
  # ffmpeg 5+ с NVENC и drawtext. Ubuntu 22.04 отдаёт 4.4: без переходов
  # hlwind/hrwind/zoomin (склейка заменяет их на hblur/fade — вид ролика
  # отличается) и без NVENC-совместимой сборки. Первый кандидат, прошедший
  # проверку (переходы, drawtext, hevc_nvenc), ставится в /usr/local/bin.
  good() {
    "$1" -hide_banner -filters 2>/dev/null | grep -q " drawtext " || return 1
    "$1" -hide_banner -h filter=xfade 2>/dev/null | grep -q "hlwind" || return 1
    "$1" -hide_banner -encoders 2>/dev/null | grep -q "hevc_nvenc" || return 1
  }
  # NVENC в списке кодеков ещё не значит, что он работает: сборка с новыми
  # заголовками nv-codec-headers отказывает на драйвере старше нужного
  # («Driver does not support the required nvenc API version»), а на A40 30.09
  # клипы из-за этого шли процессором. Настоящее пробное кодирование одного
  # кадра; нет видеокарты (nvidia-smi) — проверять нечем, считается годной.
  nvenc_ok() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    "$1" -hide_banner -v error -f lavfi -i testsrc2=s=1280x720:d=0.2 -pix_fmt p010le \
      -c:v hevc_nvenc -f null - 2>/tmp/nvenc_probe.txt
  }
  nvenc_report() {
    say "ffmpeg: NVENC не заработал (драйвер $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)): $(tr '\n' ' ' < /tmp/nvenc_probe.txt | cut -c1-200)"
  }
  if command -v ffmpeg >/dev/null 2>&1 && good ffmpeg && nvenc_ok ffmpeg; then
    say "ffmpeg уже подходит ($(ffmpeg -version | head -1 | cut -c1-40))"
  else
    D=/tmp/ffm; mkdir -p $D; OK=0; KEEP=""
    # BtbN (GPL: NVENC, drawtext, переходы hlwind/hrwind/zoomin). Имя файла на
    # релизе `latest` постоянное; проверено 30.09: 200, 148 МБ, ffmpeg N-1269xx —
    # drawtext есть, три перехода есть, hevc_nvenc есть. Статическая сборка другого автора
    # без NVENC и здесь не подходит; API GitHub с общего адреса пода режется
    # лимитом, поэтому имя не выясняется запросом.
    # Порядок — от проверенной сборки к запасным: у других веток заголовки
    # NVENC другие, и они могут идти на драйвере, где master отказывает.
    for u in "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz" \
             "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz" \
             "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-9.0.tar.xz"; do
      if ! curl -fL --retry 2 --max-time 300 -sS "$u" -o $D/f.tar.xz 2>$D/err.txt; then
        say "ffmpeg: скачивание не удалось ($(tr '\n' ' ' < $D/err.txt | cut -c1-160))"; continue
      fi
      # Контрольная сумма из того же релиза (checksums.sha256). Не совпала — файл
      # не берём (сборка `latest` меняется, файл мог прийти повреждённым или
      # чужим); сумм не получили вовсе — принимаем с предупреждением: проверить
      # нечем, а отказ оставил бы ffmpeg 4.4 без переходов и NVENC.
      if curl -fsL --retry 2 --max-time 60 "${u%/*}/checksums.sha256" -o $D/sums.txt 2>/dev/null; then
        want=$(grep -F " ${u##*/}" $D/sums.txt | head -1 | cut -d' ' -f1)
        got=$(sha256sum $D/f.tar.xz | cut -d' ' -f1)
        if [ -z "$want" ]; then
          say "ffmpeg: в checksums.sha256 нет строки для ${u##*/} — файл принят без проверки"
        elif [ "$want" != "$got" ]; then
          say "ffmpeg: контрольная сумма не совпала (ждали ${want:0:12}, получили ${got:0:12}) — файл отброшен"; continue
        else
          say "ffmpeg: контрольная сумма совпала"
        fi
      else
        say "ffmpeg: checksums.sha256 недоступен — файл принят без проверки"
      fi
      rm -rf $D/w; mkdir -p $D/w
      if ! tar -xf $D/f.tar.xz -C $D/w 2>$D/err.txt; then
        say "ffmpeg: распаковка не удалась ($(tr '\n' ' ' < $D/err.txt | cut -c1-160))"; continue
      fi
      B=$(find $D/w -type f -name ffmpeg -perm -u+x | head -1); P=$(find $D/w -type f -name ffprobe | head -1)
      if [ -n "$B" ] && good "$B"; then
        if nvenc_ok "$B"; then
          cp "$B" /usr/local/bin/ffmpeg; [ -n "$P" ] && cp "$P" /usr/local/bin/ffprobe; OK=1
          say "ffmpeg: $("$B" -version | head -1 | cut -c1-40), NVENC работает"; break
        fi
        nvenc_report
        if [ -z "$KEEP" ]; then   # первая годная по остальному — на случай, если NVENC не пойдёт нигде
          mkdir -p /tmp/ffm_keep; cp "$B" /tmp/ffm_keep/ffmpeg; [ -n "$P" ] && cp "$P" /tmp/ffm_keep/ffprobe; KEEP=1
        fi
      else
        say "ffmpeg из $u не подошёл (нет NVENC, drawtext или переходов)"
      fi
    done
    if [ $OK = 0 ] && [ -n "$KEEP" ]; then
      cp /tmp/ffm_keep/ffmpeg /usr/local/bin/ffmpeg; [ -f /tmp/ffm_keep/ffprobe ] && cp /tmp/ffm_keep/ffprobe /usr/local/bin/ffprobe; OK=1
      say "ВНИМАНИЕ: NVENC не заработал ни в одной сборке — клипы будут кодироваться процессором (x264)"
    fi
    if [ $OK = 0 ]; then
      # apt даёт ffmpeg 4.4 без переходов hlwind/hrwind/zoomin: ролик собрался бы
      # не тем, что у CPU-ветки. Подготовка останавливается, а не деградирует.
      say "ОШИБКА: подходящего ffmpeg (5+, drawtext, переходы) не получено — задача не запускается"
      exit 1
    fi
    hash -r
  fi
  say "ffmpeg готов"
) &
P_FF=$!

(
  # uv собирает окружение в разы быстрее pip; нет uv — обычный pip.
  pip install -q uv >/dev/null 2>&1 && INSTALL="uv pip install --system -q" || INSTALL="pip install -q"
  # `say` в конце подоболочки всегда даёт код 0: без явного выхода сбой
  # установки не доходил до `wait` (аудит 30.09), и задача стартовала на
  # оплаченном поде без пакетов.
  $INSTALL -r requirements.txt -r requirements-gpu.txt \
    || { say "ОШИБКА: установка пакетов не удалась"; exit 1; }
  say "пакеты готовы"
) &
P_PIP=$!

(
  # hf_transfer нужен до закачки; ставим его отдельно и сразу.
  pip install -q huggingface_hub hf_transfer >/dev/null 2>&1
  python scripts/fetch_weights.py || { say "ОШИБКА: веса моделей не скачались"; exit 1; }
  say "веса готовы"
) &
P_W=$!

FAIL=0
for p in $P_FF $P_PIP $P_W; do wait $p || FAIL=1; done
# Версии стека — в лог (аудит: какой именно набор собрал этот под).
python - <<'PY'
import importlib
for m in ("torch", "torchvision", "transformers", "sentence_transformers", "huggingface_hub", "numpy", "cv2", "PIL"):
    try:
        print("  пакет", m, importlib.import_module(m).__version__)
    except Exception as e:  # noqa: BLE001
        print("  пакет", m, "НЕ ИМПОРТИРУЕТСЯ:", type(e).__name__)
PY
ffmpeg -version | head -1
say "подготовка окончена (ошибка: $FAIL)"
exit $FAIL
