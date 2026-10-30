param(
    [string]$Python = '',
    [string]$Target = 'x86_64-pc-windows-msvc'
)
$ErrorActionPreference = 'Stop'
$reviewRoot = Split-Path $PSScriptRoot -Parent
$desktopRoot = Join-Path (Split-Path $reviewRoot -Parent) 'desktop'
if (-not $Python) { $Python = Join-Path $desktopRoot '.venv-build\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'The isolated build environment is missing. Follow docs/DESKTOP.md to install .venv-build dependencies.' }
$frontend = Join-Path (Split-Path $reviewRoot -Parent) 'frontend\dist'
if (-not (Test-Path -LiteralPath (Join-Path $frontend 'index.html'))) { throw 'Frontend dist/index.html is missing. Build the frontend first.' }
$binaryDir = Join-Path $desktopRoot 'src-tauri\binaries'
$buildRoot = Join-Path $desktopRoot 'build'
New-Item -ItemType Directory -Force -Path $binaryDir, $buildRoot | Out-Null
$publicConfigs = Join-Path $buildRoot 'public-configs'
New-Item -ItemType Directory -Force -Path $publicConfigs | Out-Null
foreach ($configName in @('runtime_profiles.json', 'local_llm.json', 'correction_llm.json', 'qwen35_llm.json', 'qwen_asr.json')) {
    Copy-Item -LiteralPath (Join-Path $reviewRoot "configs\$configName") -Destination (Join-Path $publicConfigs $configName) -Force
}
$arguments = @('-m', 'PyInstaller', '--noconfirm', '--clean', '--onefile', '--console', '--name', '4c-review-backend',
    '--distpath', (Join-Path $buildRoot 'dist'), '--workpath', (Join-Path $buildRoot 'work'), '--specpath', $buildRoot,
    '--paths', $reviewRoot, '--collect-all', 'opencc', '--collect-binaries', 'pymupdf', '--collect-all', 'pptx',
    '--collect-submodules', 'uvicorn', '--hidden-import', 'review.runtime', '--hidden-import', 'pypinyin', '--hidden-import', 'jieba',
    '--exclude-module', 'torch', '--exclude-module', 'transformers', '--exclude-module', 'qwen_asr',
    '--exclude-module', 'faster_whisper', '--exclude-module', 'ctranslate2', '--exclude-module', 'onnxruntime',
    '--exclude-module', 'rapidocr_onnxruntime', '--exclude-module', 'av', '--exclude-module', 'huggingface_hub',
    '--exclude-module', 'pytest', '--exclude-module', 'sklearn', '--exclude-module', 'scipy', '--exclude-module', 'matplotlib',
    '--exclude-module', 'pymupdf.layout',
    '--add-data', ($publicConfigs + ';configs'),
    '--add-data', ((Join-Path $PSScriptRoot 'convert_courseware.ps1') + ';scripts'),
    '--add-data', ((Join-Path $reviewRoot 'review\courseware_alignment.py') + ';review'),
    (Join-Path $PSScriptRoot 'desktop_entry.py'))
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Backend packaging failed: $LASTEXITCODE" }
Copy-Item -LiteralPath (Join-Path $buildRoot 'dist\4c-review-backend.exe') -Destination (Join-Path $binaryDir "4c-review-backend-$Target.exe") -Force
# tauri.conf.json maps the current Front/dist directly to the frontend resource.
# There is no mutable staging copy that can retain obsolete hashed UI assets.
Write-Output (Join-Path $binaryDir "4c-review-backend-$Target.exe")
