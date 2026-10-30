param(
    [string]$Video=(Join-Path $PSScriptRoot '../语音模型/lesson1.mp4'),
    [string]$LessonId='lesson1-real',
    [string]$Title='课堂复盘',
    [double[]]$Roi=@(0.49,0.055,0.91,0.47),
    [double]$SampleInterval=5,
    [switch]$SkipModelStart,
    [switch]$ChineseReference
)
$ErrorActionPreference='Stop'
if ($LessonId -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$') { throw 'Invalid LessonId.' }
$pythonExe=Join-Path $PSScriptRoot '.venv-media/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Follow README.md to create the inference environment first.' }
$videoPath=(Resolve-Path -LiteralPath $Video).Path
$output=Join-Path $PSScriptRoot "artifacts/$LessonId"
$mediaArgs=@((Join-Path $PSScriptRoot 'scripts/prepare_media.py'),'--video',$videoPath,'--output',$output,'--mode','all','--sample-interval',"$SampleInterval",'--roi')
$mediaArgs+=@($Roi | ForEach-Object { $_.ToString([Globalization.CultureInfo]::InvariantCulture) })
& $pythonExe @mediaArgs
if ($LASTEXITCODE -ne 0) { throw 'Media extraction failed; partial outputs were not published as a lesson.' }
if ($ChineseReference) {
    & $pythonExe (Join-Path $PSScriptRoot 'scripts/generate_chinese_reference.py') --media-dir $output
    if ($LASTEXITCODE -ne 0) { throw 'Chinese reference recognition failed; the original ASR was preserved.' }
}
if (-not $SkipModelStart) {
    $shellExe=(Get-Process -Id $PID).Path
    & $shellExe -NoProfile -File (Join-Path $PSScriptRoot 'scripts/start_local_llm.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Local model startup failed.' }
}
& $pythonExe (Join-Path $PSScriptRoot 'scripts/build_review.py') --lesson-id $LessonId --video $videoPath --media-dir $output --title $Title
if ($LASTEXITCODE -ne 0) { throw 'Lesson build failed; inspect source/configuration/model status.' }
& (Join-Path $PSScriptRoot 'start_review.ps1')
Write-Output "Open http://127.0.0.1:5175/ after starting the frontend. Lesson: $LessonId"
