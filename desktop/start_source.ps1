param([string]$Registry = '', [string]$DataDir = '', [int]$Port = 0)
$ErrorActionPreference = 'Stop'
$reviewRoot = Join-Path (Split-Path $PSScriptRoot -Parent) '课堂复盘'
$python = Join-Path $reviewRoot '.venv-media\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw '缺少 .venv-media Python 环境。请按 backend/docs/DESKTOP.md 安装后端依赖。' }
if (-not $DataDir) { $DataDir = if ($env:REVIEW_DATA_DIR) { $env:REVIEW_DATA_DIR } else { Join-Path $PSScriptRoot 'workspace\source-state' } }
$arguments = @((Join-Path $reviewRoot 'scripts\desktop_entry.py'), '--port', $Port, '--reuse', '--open-browser')
if ($Registry) { $arguments += @('--registry', $Registry) }
$arguments += @('--data-dir', $DataDir)
& $python @arguments
exit $LASTEXITCODE
