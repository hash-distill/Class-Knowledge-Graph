param([string]$Registry = '', [string]$DataDir = '', [string]$RuntimeRoot = '')
$ErrorActionPreference = 'Stop'
$application = Join-Path $PSScriptRoot 'src-tauri\target\release\four-c-desktop.exe'
if (-not (Test-Path -LiteralPath $application -PathType Leaf)) { throw 'The desktop application has not been built. Run desktop/scripts/build.ps1 first, or use desktop/start_source.ps1.' }
if (-not $Registry) { $Registry = Join-Path (Split-Path $PSScriptRoot -Parent) 'backend\configs\lessons.json' }
if (-not $DataDir) { $DataDir = Join-Path $PSScriptRoot 'workspace\desktop-state' }
if (-not $RuntimeRoot) { $RuntimeRoot = if ($env:REVIEW_RUNTIME_ROOT) { $env:REVIEW_RUNTIME_ROOT } else { Join-Path (Split-Path $PSScriptRoot -Parent) 'backend' } }
$env:REVIEW_DESKTOP_DATA_DIR = [System.IO.Path]::GetFullPath($DataDir)
$env:REVIEW_LESSON_REGISTRY = [System.IO.Path]::GetFullPath($Registry)
$env:REVIEW_RUNTIME_ROOT = [System.IO.Path]::GetFullPath($RuntimeRoot)
& $application
