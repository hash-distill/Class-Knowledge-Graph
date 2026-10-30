param([switch]$SkipBackend, [switch]$NoBundle)
$ErrorActionPreference = 'Stop'
$desktopRoot = Split-Path $PSScriptRoot -Parent
$cargoBin = Join-Path $desktopRoot '.toolchains\cargo\bin'
if (Test-Path -LiteralPath (Join-Path $cargoBin 'cargo.exe')) {
    $env:CARGO_HOME = Join-Path $desktopRoot '.toolchains\cargo'
    $env:RUSTUP_HOME = Join-Path $desktopRoot '.toolchains\rustup'
    $env:Path = $cargoBin + ';' + $env:Path
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw '找不到 Rust/Cargo。请按 backend/docs/DESKTOP.md 安装用户或工作区工具链。' }
if (-not $SkipBackend) { & (Join-Path (Split-Path $desktopRoot -Parent) 'backend\scripts\build_desktop_backend.ps1') }
$buildTemp = Join-Path $desktopRoot 'build\temp'
New-Item -ItemType Directory -Force -Path $buildTemp | Out-Null
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$env:TEMP = $buildTemp
$env:TMP = $buildTemp
Push-Location $desktopRoot
try {
    if (-not (Test-Path -LiteralPath 'node_modules\@tauri-apps\cli')) { throw '请先在 desktop 目录执行 npm ci。' }
    if (-not (Test-Path -LiteralPath 'src-tauri\icons\icon.ico')) {
        & npm.cmd exec tauri -- icon assets/icon.svg --output src-tauri/icons
        if ($LASTEXITCODE -ne 0) { throw '图标生成失败。' }
    }
    if ($NoBundle) { & npm.cmd run build -- --no-bundle } else { & npm.cmd run build }
    if ($LASTEXITCODE -ne 0) { throw "Tauri 构建失败：$LASTEXITCODE" }
} finally {
    Pop-Location
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}
