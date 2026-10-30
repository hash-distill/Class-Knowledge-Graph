param([ValidateSet('cpu','cuda')][string]$Backend = '',[switch]$Stop)
$ErrorActionPreference = 'Stop'
$parameters = @{ ConfigPath = 'configs/correction_llm.json' }
if ($Backend) { $parameters.Backend = $Backend }
if ($Stop) { $parameters.Stop = $true }
& (Join-Path $PSScriptRoot 'start_local_llm.ps1') @parameters
exit $LASTEXITCODE
