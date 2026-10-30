param([int]$Port=8000,[switch]$Restart)
$ErrorActionPreference='Stop'
$reviewRoot=$PSScriptRoot
$pythonExe=Join-Path $reviewRoot '.venv-media/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Create .venv-media and install requirements-media.txt plus requirements-review.txt first.' }
& $pythonExe -c 'import fastapi, uvicorn'
if ($LASTEXITCODE -ne 0) { throw 'Install requirements-review.txt into .venv-media before starting the FastAPI service.' }
$runtime=Join-Path $reviewRoot 'artifacts/runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$statePath=Join-Path $runtime 'review-api-state.json'
if ($Restart -and (Test-Path -LiteralPath $statePath)) {
    $state=Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $running=Get-Process -Id $state.pid -ErrorAction SilentlyContinue
    if ($running -and $running.Path -eq $state.executable -and $running.StartTime.ToUniversalTime().Ticks -eq ([datetime]$state.started_at).ToUniversalTime().Ticks) { Stop-Process -Id $running.Id }
    elseif ($running) { throw 'PID identity changed; refusing to stop an unrelated process.' }
}
try { $existing=Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/review/health" -TimeoutSec 2 }
catch { $existing=$null }
if ($existing -and $existing.service -eq 'classroom-review') {
    if ($existing.framework -ne 'fastapi') { throw 'A legacy review API is still running. Run start_review.ps1 -Restart to use FastAPI.' }
    Write-Output "FastAPI review API already running on $Port"; return
}
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$process=Start-Process -FilePath $pythonExe -ArgumentList @('scripts/serve_review.py','--port',"$Port") -WorkingDirectory $reviewRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime "review-$stamp.stdout.log") -RedirectStandardError (Join-Path $runtime "review-$stamp.stderr.log") -PassThru
@{pid=$process.Id;executable=$pythonExe;started_at=$process.StartTime.ToUniversalTime().ToString('o');port=$Port} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
for ($attempt=0;$attempt -lt 30;$attempt++) {
    if ($process.HasExited) { throw 'Review API failed to start; inspect artifacts/runtime logs.' }
    try {
        $health=Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/review/health" -TimeoutSec 1
        if ($health.service -eq 'classroom-review') { Write-Output "Review API ready: http://127.0.0.1:$Port/api/review/health (PID $($process.Id))"; return }
    } catch { }
    Start-Sleep -Milliseconds 500
}
throw 'Review API startup timed out; inspect artifacts/runtime logs.'
