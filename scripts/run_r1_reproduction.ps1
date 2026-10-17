[CmdletBinding()]
param(
    [string]$ProjectRoot,
    [string]$Config = "configs\q1_v2.toml",
    [switch]$FromEvaluate
)

# Unattended R1 reproduction: simulate -> train -> gate -> locked evaluation -> figures.
# Progress is written to artifacts_q1_v2\continuation_state.json and per-stage logs.

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$run = Join-Path $ProjectRoot "run.ps1"
$artifactRoot = Join-Path $ProjectRoot "artifacts_q1_v2"
$logRoot = Join-Path $artifactRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$statePath = Join-Path $artifactRoot "continuation_state.json"

function Write-State {
    param([string]$Stage, [string]$Status, [string]$Detail = "")
    $temporary = "$statePath.tmp"
    @{
        stage = $Stage
        status = $Status
        detail = $Detail
        updated_at = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
}

function Invoke-Stage {
    param([string]$Stage)
    Write-State -Stage $Stage -Status "running"
    $log = Join-Path $logRoot "$Stage.log"
    $started = Get-Date
    & $run $Stage -Resume -Config $Config *>> $log
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE (see $log)."
    }
    $minutes = [Math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    Add-Content -LiteralPath (Join-Path $logRoot "stage_durations.csv") -Value "$Stage,$minutes"
}

try {
    if (-not $FromEvaluate) {
        Invoke-Stage "audit"
        Invoke-Stage "pilot"
        Invoke-Stage "simulate"
        Invoke-Stage "train"
    }

    $gatePath = Join-Path $artifactRoot "training\TRAINING_GATE.json"
    $gate = Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json
    if ($gate.decision -ne "GO") {
        Write-State -Stage "training_gate" -Status "no_go" -Detail "Locked evaluation was not opened."
        exit 42
    }

    $tokenBytes = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($tokenBytes)
    $rng.Dispose()
    $env:GRIDIMPAIR_UNLOCK_TOKEN = [BitConverter]::ToString($tokenBytes).Replace("-", "")
    try {
        Invoke-Stage "evaluate"
    }
    finally {
        Remove-Item Env:\GRIDIMPAIR_UNLOCK_TOKEN -ErrorAction SilentlyContinue
        [Array]::Clear($tokenBytes, 0, $tokenBytes.Length)
    }

    Invoke-Stage "figures"
    Write-State -Stage "complete" -Status "completed"
}
catch {
    Write-State -Stage "pipeline" -Status "failed" -Detail $_.Exception.Message
    throw
}
