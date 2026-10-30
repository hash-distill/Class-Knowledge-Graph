param(
    [ValidateSet('cpu','cuda')][string]$LlmBackend='cuda',
    [int]$FrontendPort=5175,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$LessonId='unified-chemistry-001-9',
    [switch]$RestartApi
)
$ErrorActionPreference='Stop'
$reviewDirectory=$PSScriptRoot
$frontendDirectory=[IO.Path]::GetFullPath((Join-Path $reviewDirectory '../frontend'))
& (Join-Path $reviewDirectory 'scripts/start_local_llm.ps1') -Backend $LlmBackend
& (Join-Path $reviewDirectory 'start_review.ps1') -Restart:$RestartApi
$demoAddress="http://127.0.0.1:$FrontendPort/?lesson=$LessonId"
$alreadyReady=$false
try {
    $response=Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort/" -TimeoutSec 2
    if ($response.Content -notmatch '/src/main.ts') { throw 'Port belongs to another application.' }
    $alreadyReady=$true
} catch {
    if (Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $FrontendPort is occupied by an unrecognized service. Select a different FrontendPort."
    }
}
if (-not $alreadyReady) {
    $nodeExecutable=(Get-Command node.exe -ErrorAction Stop).Source
    $viteEntry=Join-Path $frontendDirectory 'node_modules/vite/bin/vite.js'
    if (-not (Test-Path -LiteralPath $viteEntry)) { throw 'Install frontend dependencies with npm ci first.' }
    $runtimeDirectory=Join-Path $reviewDirectory 'artifacts/runtime'
    New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
    $timestamp=Get-Date -Format 'yyyyMMdd-HHmmss'
    $process=Start-Process -FilePath $nodeExecutable -ArgumentList @("`"$viteEntry`"",'--host','127.0.0.1','--port',"$FrontendPort",'--strictPort') -WorkingDirectory $frontendDirectory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeDirectory "frontend-$timestamp.stdout.log") -RedirectStandardError (Join-Path $runtimeDirectory "frontend-$timestamp.stderr.log") -PassThru
    @{pid=$process.Id;executable=$nodeExecutable;started_at=$process.StartTime.ToUniversalTime().ToString('o');port=$FrontendPort} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDirectory 'trained-demo-frontend.json') -Encoding utf8
    for ($attempt=0;$attempt -lt 30;$attempt++) {
        if ($process.HasExited) { throw 'Frontend failed to start. Inspect artifacts/runtime logs.' }
        try {
            $null=Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort/" -TimeoutSec 1
            $alreadyReady=$true
            break
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $alreadyReady) { throw 'Frontend startup timed out.' }
}
$lesson=Invoke-RestMethod -Uri "http://127.0.0.1:$FrontendPort/api/review/lessons/$LessonId" -TimeoutSec 10
if ($lesson.lesson.status -ne 'complete') { throw 'The Chinese classroom demonstration is not prepared. See docs/TRAINED_ACTIVITY.md.' }
Write-Output "Chinese classroom demonstration ready: $demoAddress"
