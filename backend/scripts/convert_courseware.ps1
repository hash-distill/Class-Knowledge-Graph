param(
    [string]$InputPath,
    [string]$OutputPath,
    [Parameter(Mandatory=$true)][string]$OwnerPath,
    [int]$MaxPages = 200,
    [switch]$CleanupOnly
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

if ($CleanupOnly) {
    if (Test-Path -LiteralPath $OwnerPath) {
        $ownerInfo = Get-Content -LiteralPath $OwnerPath -Raw | ConvertFrom-Json
        $ownedProcess = Get-Process -Id ([int]$ownerInfo.pid) -ErrorAction SilentlyContinue
        if ($ownedProcess -and $ownedProcess.ProcessName -eq 'POWERPNT' -and
            $ownedProcess.StartTime.ToUniversalTime().Ticks.ToString() -eq $ownerInfo.start_ticks) {
            # A user may have opened another presentation since we started.
            # If the running instance cannot be inspected, leave it untouched.
            $cleanupApplication = $null
            try {
                $cleanupApplication = [Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application')
                $safeToStop = $true
                foreach ($openPresentation in $cleanupApplication.Presentations) {
                    if ($openPresentation.FullName -ne $ownerInfo.source_path) { $safeToStop = $false }
                    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($openPresentation)
                }
                if ($safeToStop -and $cleanupApplication.HWND -eq $ownerInfo.hwnd) {
                    Stop-Process -Id $ownedProcess.Id -Force
                }
            } catch {
                Write-Warning 'Timed-out PowerPoint instance could not be inspected; it was left untouched.'
            } finally {
                if ($cleanupApplication) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($cleanupApplication) }
            }
        }
    }
    exit 0
}

$inputResolved = (Resolve-Path -LiteralPath $InputPath).Path
$outputResolved = [IO.Path]::GetFullPath($OutputPath)
$converterTemp = Join-Path ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($OwnerPath))) 'converter-temp'
[void][IO.Directory]::CreateDirectory($converterTemp)
$env:TEMP = $converterTemp
$env:TMP = $converterTemp
if (Test-Path -LiteralPath $outputResolved) { throw 'Output already exists.' }
# PowerPoint COM may reuse an existing desktop instance. Refuse before touching it.
if (@(Get-Process -Name POWERPNT -ErrorAction SilentlyContinue).Count -gt 0) {
    throw 'POWERPOINT_IN_USE: Please close PowerPoint yourself or upload a PDF. Existing presentations were not touched.'
}
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class CoursewareNative {
  [DllImport("user32.dll")]
  public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
}
'@
$application = $null
$presentation = $null
$ownedProcess = $null
$ownershipVerified = $false
$createdAfter = [DateTime]::UtcNow
try {
    $application = New-Object -ComObject PowerPoint.Application
    [uint32]$applicationProcessId = 0
    [void][CoursewareNative]::GetWindowThreadProcessId([IntPtr]$application.HWND, [ref]$applicationProcessId)
    $ownedProcess = Get-Process -Id $applicationProcessId
    if ($ownedProcess.ProcessName -ne 'POWERPNT' -or
        $ownedProcess.StartTime.ToUniversalTime() -lt $createdAfter -or $application.Presentations.Count -ne 0) {
        throw 'POWERPOINT_OWNERSHIP_UNCERTAIN: no document was opened.'
    }
    $ownershipVerified = $true
    @{pid=$ownedProcess.Id;start_ticks=$ownedProcess.StartTime.ToUniversalTime().Ticks.ToString();
      source_path=$inputResolved;hwnd=$application.HWND} |
        ConvertTo-Json | Set-Content -LiteralPath $OwnerPath -Encoding UTF8
    $application.AutomationSecurity = 3 # msoAutomationSecurityForceDisable: never execute macros.
    $application.DisplayAlerts = 1 # ppAlertsNone
    $presentation = $application.Presentations.Open($inputResolved, -1, 0, 0)
    if ($presentation.Slides.Count -lt 1 -or $presentation.Slides.Count -gt $MaxPages) {
        throw 'Slide count exceeds import limits.'
    }
    $presentation.SaveAs($outputResolved, 32) # ppSaveAsPDF
    if (-not (Test-Path -LiteralPath $outputResolved)) { throw 'PDF export produced no file.' }
    Write-Output 'PowerPoint read-only PDF export complete.'
} finally {
    if ($presentation) {
        $presentation.Close()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($presentation)
    }
    if ($application) {
        # Never quit a shared app or an instance that acquired a user presentation.
        if ($ownershipVerified -and $application.Presentations.Count -eq 0) { $application.Quit() }
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($application)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
