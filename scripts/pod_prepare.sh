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
  if command -v ffmpeg >/dev/null 2>&1 && good ffmpeg; then
    say "ffmpeg уже подходит ($(ffmpeg -version | head -1 | cut -c1-40))"
  else
    D=/tmp/ffm; mkdir -p $D; OK=0
    # 1) BtbN (GPL, nvenc, drawtext): имя файла ищем через API — оно меняется.
    URL=$(python3 - <<'PY'
import json, urllib.request
try:
    d = json.load(urllib.request.urlopen("https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/latest", timeout=30))
    for a in d["assets"]:
        n = a["name"]
        if "linux64-gpl" in n and "shared" not in n and n.endswith(".tar.xz") and "n7" in n:
            print(a["browser_download_url"]); break
except Exception:
    pass
PY
)
    # 2) запасной: статическая сборка johnvansickle (ffmpeg 7).
    for u in "$URL" "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"; do
      [ -n "$u" ] || continue
      if curl -sfL --max-time 300 "$u" -o $D/f.tar.xz && tar -xf $D/f.tar.xz -C $D; then
        B=$(find $D -type f -name ffmpeg -perm -u+x | head -1); P=$(find $D -type f -name ffprobe | head -1)
        if [ -n "$B" ] && good "$B"; then
          cp "$B" /usr/local/bin/ffmpeg; [ -n "$P" ] && cp "$P" /usr/local/bin/ffprobe; OK=1
          say "ffmpeg: $u"; break
        else
          say "ffmpeg из $u не подошёл (нет NVENC, drawtext или переходов)"
        fi
      fi
    done
    if [ $OK = 0 ]; then
      apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1
      say "ВНИМАНИЕ: подходящего ffmpeg нет, остался apt ($(ffmpeg -version | head -1 | cut -c1-30)) — переходы будут заменены"
    fi
    hash -r
  fi
  say "ffmpeg готов"
) &
P_FF=$!

(
  # uv собирает окружение в разы быстрее pip; нет uv — обычный pip.
  pip install -q uv >/dev/null 2>&1 && INSTALL="uv pip install --system -q" || INSTALL="pip install -q"
  $INSTALL -r requirements.txt -r requirements-gpu.txt
  say "пакеты готовы"
) &
P_PIP=$!

(
  # hf_transfer нужен до закачки; ставим его отдельно и сразу.
  pip install -q huggingface_hub hf_transfer >/dev/null 2>&1
  python scripts/fetch_weights.py
  say "веса готовы"
) &
P_W=$!

FAIL=0
for p in $P_FF $P_PIP $P_W; do wait $p || FAIL=1; done
say "подготовка окончена (ошибка: $FAIL)"
exit $FAIL
