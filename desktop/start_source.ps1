param([string]$Registry = '', [string]$DataDir = '', [int]$Port = 0)
$ErrorActionPreference = 'Stop'
$reviewRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'backend'
$python = Join-Path $reviewRoot '.venv-media\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'The .venv-media Python environment is missing. Install backend dependencies as described in backend/docs/DESKTOP.md.' }
if (-not $DataDir) { $DataDir = if ($env:REVIEW_DATA_DIR) { $env:REVIEW_DATA_DIR } else { Join-Path $PSScriptRoot 'workspace\source-state' } }
$arguments = @((Join-Path $reviewRoot 'scripts\desktop_entry.py'), '--port', $Port, '--reuse', '--open-browser')
if ($Registry) { $arguments += @('--registry', $Registry) }
$arguments += @('--data-dir', $DataDir)
& $python @arguments
exit $LASTEXITCODE
