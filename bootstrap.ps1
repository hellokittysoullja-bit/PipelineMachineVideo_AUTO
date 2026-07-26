# ============================================================
#  FACELESS PIPELINE — Bootstrap (Windows / PowerShell)
#  Разворачивает окружение "с нуля": ffmpeg, Python venv,
#  зависимости, структуру папок, конфиг. Идемпотентен —
#  безопасно запускать повторно.
#
#  Claude Code запускает это САМ при первом старте (см. PART 0
#  в CLAUDE.md). Вручную:  powershell -ExecutionPolicy Bypass -File bootstrap.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Write-Host "==> Bootstrap в $Root" -ForegroundColor Cyan

# 1. FFmpeg -----------------------------------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "==> FFmpeg не найден, ставлю..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        choco install ffmpeg -y
    } else {
        Write-Host "!! Нет ни winget, ни choco. Поставь FFmpeg вручную: https://ffmpeg.org/download.html" -ForegroundColor Red
    }
} else {
    Write-Host "==> FFmpeg на месте" -ForegroundColor Green
}

# 2. Python venv ----------------------------------------------
# Без оператора ?? — он появился только в PowerShell 7, а в Windows 10/11
# из коробки стоит 5.1, где эта строка падала с синтаксической ошибкой.
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) { throw "Python не найден. Поставь Python 3.11+ и повтори." }

$venv = Join-Path $Root ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "==> Создаю venv..." -ForegroundColor Yellow
    & $py.Source -m venv $venv
}
$pip = Join-Path $venv "Scripts\pip.exe"
Write-Host "==> Ставлю Python-зависимости..." -ForegroundColor Yellow
& $pip install --upgrade pip | Out-Null
& $pip install -r (Join-Path $Root "requirements.txt")

# 3. Структура папок ------------------------------------------
foreach ($d in @("videos", "scripts", "secrets")) {
    $p = Join-Path $Root $d
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}
Write-Host "==> Папки готовы (videos, scripts, secrets)" -ForegroundColor Green

# 4. Конфиг ключей --------------------------------------------
$env_file = Join-Path $Root ".env"
if (-not (Test-Path $env_file)) {
    Copy-Item (Join-Path $Root "config.example.env") $env_file
    Write-Host "==> Создан .env — ВПИШИ туда свои ключи (ELEVENLABS, PEXELS и т.д.)" -ForegroundColor Magenta
} else {
    Write-Host "==> .env уже есть" -ForegroundColor Green
}

# 5. Маркер завершения ----------------------------------------
"setup ok $(Get-Date -Format o)" | Out-File (Join-Path $Root ".setup_complete") -Encoding utf8

Write-Host "`n==> Готово. Дальше: 1) заполни .env  2) скажи Claude «Новый ролик: [ТЕМА]» (скрипты уже в scripts\)" -ForegroundColor Cyan
