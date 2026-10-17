[CmdletBinding()]
param(
    [switch]$Full,
    [switch]$Minimal,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if ($Full -and $Minimal) {
    throw "-Full and -Minimal cannot be used together."
}
$InstallClassical = -not $Minimal

$UvAvailable = $true
python -m uv --version *> $null
if ($LASTEXITCODE -ne 0) {
    $UvAvailable = $false
}
if (-not $UvAvailable) {
    python -m pip install --disable-pip-version-check --user "uv>=0.8"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install uv."
    }
    python -m uv --version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "uv Python module could not be started after installation."
    }
}

$SyncArguments = @("sync")
if ($InstallClassical) {
    $SyncArguments += @("--extra", "classical")
}
$SyncArguments += @("--extra", "energy")
python -m uv @SyncArguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Smoke) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $SmokeArguments = @("scripts\setup_smoke.py")
    if ($InstallClassical) {
        $SmokeArguments += "--classical"
    }
    & $Python @SmokeArguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Local environment ready. Scientific stages were not automatically started."
