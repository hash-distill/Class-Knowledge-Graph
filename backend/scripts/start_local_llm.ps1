param(
    [ValidateSet('cpu', 'cuda')][string]$Backend = '',
    [int]$WaitSeconds = 120,
    [switch]$Stop,
    [string]$ConfigPath = 'configs/local_llm.json'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$resolvedConfig = if ([IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $projectRoot $ConfigPath }
$config = Get-Content -LiteralPath $resolvedConfig -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $Backend) { $Backend = $config.runtime.default_backend }
$verificationSubdir = if ($config.verification_subdir) { [string]$config.verification_subdir } else { 'llm' }
if ($verificationSubdir -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid verification directory name.' }
$logDir = Join-Path $projectRoot "artifacts/verification/$verificationSubdir"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$statePath = Join-Path $logDir 'server_state.json'
$healthUrl = 'http://{0}:{1}/health' -f $config.server.host, $config.server.port

if ($Stop) {
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw -Encoding utf8 | ConvertFrom-Json
        $running = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
        if ($running -and $running.Path -eq $state.executable -and $running.StartTime.ToUniversalTime().Ticks -eq ([datetime]$state.process_start_utc).ToUniversalTime().Ticks) {
            Stop-Process -Id $running.Id
            Write-Output ('Stopped local llama-server PID {0}' -f $running.Id)
        } elseif ($running) {
            throw 'Recorded PID has been reused; no process was stopped.'
        } else { Write-Output 'Recorded local server is already stopped.' }
    } else { Write-Output 'No recorded local server state exists.' }
    exit 0
}

try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
    if ($health.status -eq 'ok') {
        $models = Invoke-RestMethod -Uri ($config.base_url + '/models') -TimeoutSec 2
        if ($models.data.id -contains $config.model_alias) {
            Write-Output ('Local model already ready: {0}, alias {1}' -f $config.base_url, $config.model_alias)
            exit 0
        }
        throw 'The configured port hosts another model.'
    }
} catch {
    if ($_.Exception.Message -like '*hosts another model*') { throw }
}

$runtimeDir = Join-Path $projectRoot $config.runtime.$Backend.directory
$executables = @(Get-ChildItem -LiteralPath $runtimeDir -Filter 'llama-server.exe' -Recurse -File -ErrorAction SilentlyContinue)
if ($executables.Count -ne 1) { throw "Run python scripts/fetch_local_llm.py --runtime $Backend first; exactly one llama-server.exe is required." }
$executable = $executables[0].FullName
$modelPath = Join-Path $projectRoot $config.model.path
if (-not (Test-Path -LiteralPath $modelPath)) { throw 'Pinned model is missing. Run scripts/fetch_local_llm.py first.' }
if ((Get-Item -LiteralPath $modelPath).Length -ne $config.model.size_bytes) { throw 'Pinned model size mismatch.' }
if ((Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $config.model.sha256) { throw 'Pinned model SHA-256 mismatch.' }
$gpuLayers = if ($Backend -eq 'cuda') { 99 } else { 0 }
$arguments = @(
    '--model', ('"{0}"' -f $modelPath),
    '--host', $config.server.host, '--port', $config.server.port,
    '--alias', $config.model_alias,
    '--ctx-size', $config.server.context_size,
    '--parallel', $config.server.parallel,
    '--threads', $config.server.threads,
    '--threads-batch', $config.server.threads_batch,
    '--n-gpu-layers', $gpuLayers,
    '--jinja', '--reasoning', $(if ($config.server.reasoning) { $config.server.reasoning } else { 'off' })
)
if ($config.server.flash_attention) {
    $arguments += @('--flash-attn', [string]$config.server.flash_attention)
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutPath = Join-Path $logDir "server-$Backend-$stamp.stdout.log"
$stderrPath = Join-Path $logDir "server-$Backend-$stamp.stderr.log"
$process = Start-Process -FilePath $executable -ArgumentList $arguments -WorkingDirectory $executables[0].DirectoryName -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$state = [ordered]@{
    pid = $process.Id
    executable = $executable
    process_start_utc = $process.StartTime.ToUniversalTime().ToString('o')
    backend = $Backend
    base_url = $config.base_url
    alias = $config.model_alias
    model_sha256 = $config.model.sha256
    stdout = $stdoutPath
    stderr = $stderrPath
}
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding utf8
$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline) {
    $process.Refresh()
    if ($process.HasExited) { throw "llama-server exited with code $($process.ExitCode). Inspect $stderrPath" }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($health.status -eq 'ok') {
            Write-Output ('Ready: {0}, model {1}, PID {2}, backend {3}' -f $config.base_url, $config.model_alias, $process.Id, $Backend)
            exit 0
        }
    } catch { }
    Start-Sleep -Seconds 1
}
throw "Server PID $($process.Id) has not become ready within $WaitSeconds seconds. Inspect $stderrPath"
