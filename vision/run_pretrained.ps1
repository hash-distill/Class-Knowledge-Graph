param(
    [Parameter(Mandatory=$true)][string]$Source,
    [string]$Device = 'auto',
    [ValidateSet('unspecified','lecture','exercise','discussion','presentation')][string]$Task = 'unspecified',
    [string]$ContextJson,
    [string]$StudentRoi,
    [int]$MaxFrames = 0,
    [double]$IntervalSec = 0.2,
    [double]$SnapshotIntervalSec = 1.0,
    [switch]$AnnotatedVideo,
    [switch]$PoseOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Join-Path $PSScriptRoot 'Class_Detection'
$pythonExe = Join-Path $PSScriptRoot '.venv-inference/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw 'Missing .venv-inference. See PRETRAINED_USAGE.md for environment setup.'
}
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) { throw 'Source must be a video file.' }
$verifyArgs = @((Join-Path $projectRoot 'scripts/fetch_pretrained.py'), '--verify-only')
if ($PoseOnly) { $verifyArgs += @('--model','pose_coco') }
& $pythonExe @verifyArgs
if ($LASTEXITCODE -ne 0) { throw 'Weight verification failed. Run fetch_pretrained.py first.' }
$inferArgs = @((Join-Path $projectRoot 'scripts/infer_video.py'), '--source', $sourcePath,
    '--device', $Device, '--task', $Task, '--max-frames', "$MaxFrames",
    '--interval-sec', "$IntervalSec", '--snapshot-interval-sec', "$SnapshotIntervalSec", '--save-frames')
if ($ContextJson) { $inferArgs += @('--context-json', (Resolve-Path -LiteralPath $ContextJson).Path) }
if ($StudentRoi) { $inferArgs += @('--student-roi', $StudentRoi) }
if ($AnnotatedVideo) { $inferArgs += '--save' } else { $inferArgs += '--json-only' }
if ($PoseOnly) { $inferArgs += '--pose-only' }
& $pythonExe @inferArgs
if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE" }
