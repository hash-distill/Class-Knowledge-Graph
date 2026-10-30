param([switch]$SkipBackend, [switch]$NoBundle)
$ErrorActionPreference = 'Stop'
$desktopRoot = Split-Path $PSScriptRoot -Parent
$cargoBin = Join-Path $desktopRoot '.toolchains\cargo\bin'
if (Test-Path -LiteralPath (Join-Path $cargoBin 'cargo.exe')) {
    $env:CARGO_HOME = Join-Path $desktopRoot '.toolchains\cargo'
    $env:RUSTUP_HOME = Join-Path $desktopRoot '.toolchains\rustup'
    $env:Path = $cargoBin + ';' + $env:Path
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw 'Rust/Cargo was not found. Install a user or workspace toolchain as described in backend/docs/DESKTOP.md.' }
if (-not $SkipBackend) { & (Join-Path (Split-Path $desktopRoot -Parent) 'backend\scripts\build_desktop_backend.ps1') }
$buildTemp = Join-Path $desktopRoot 'build\temp'
New-Item -ItemType Directory -Force -Path $buildTemp | Out-Null
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$env:TEMP = $buildTemp
$env:TMP = $buildTemp
Push-Location $desktopRoot
try {
    if (-not (Test-Path -LiteralPath 'node_modules\@tauri-apps\cli')) { throw 'Run npm ci in the desktop directory first.' }
    if (-not (Test-Path -LiteralPath 'src-tauri\icons\icon.ico')) {
        & npm.cmd exec tauri -- icon assets/icon.svg --output src-tauri/icons
        if ($LASTEXITCODE -ne 0) { throw 'Icon generation failed.' }
    }
    if ($NoBundle) { & npm.cmd run build -- --no-bundle } else { & npm.cmd run build }
    if ($LASTEXITCODE -ne 0) { throw "Tauri build failed: $LASTEXITCODE" }
} finally {
    Pop-Location
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}
