[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$SimulationProcessId,

    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),

    [string]$Config = "configs\q1_v2.toml"
)

$ErrorActionPreference = "Stop"
$run = Join-Path $ProjectRoot "run.ps1"
$artifactRoot = Join-Path $ProjectRoot "artifacts_q1_v2"
$statePath = Join-Path $artifactRoot "continuation_state.json"

function Write-State {
    param(
        [string]$Stage,
        [string]$Status,
        [string]$Detail = ""
    )
    $temporary = "$statePath.tmp"
    @{
        stage = $Stage
        status = $Status
        detail = $Detail
        updated_at = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
}

try {
    Write-State -Stage "simulate" -Status "waiting"
    Wait-Process -Id $SimulationProcessId -ErrorAction Stop
    $simulationCount = @(
        Get-ChildItem (Join-Path $artifactRoot "simulation") `
            -Recurse -Filter "summary.json" -ErrorAction Stop
    ).Count
    if ($simulationCount -ne 1008) {
        throw "Simulation ended with $simulationCount/1008 summaries."
    }

    Write-State -Stage "train" -Status "running"
    & $run train -Resume -Config $Config
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE."
    }

    $gatePath = Join-Path $artifactRoot "training\TRAINING_GATE.json"
    $gate = Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json
    if ($gate.decision -ne "GO") {
        Write-State -Stage "training_gate" -Status "no_go" `
            -Detail "Locked evaluation was not opened."
        exit 42
    }

    Write-State -Stage "evaluate" -Status "running"
    $tokenBytes = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($tokenBytes)
    $rng.Dispose()
    $env:GRIDIMPAIR_UNLOCK_TOKEN = (
        [BitConverter]::ToString($tokenBytes).Replace("-", "")
    )
    try {
        & $run evaluate -Resume -Config $Config
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Remove-Item Env:\GRIDIMPAIR_UNLOCK_TOKEN -ErrorAction SilentlyContinue
        [Array]::Clear($tokenBytes, 0, $tokenBytes.Length)
    }

    Write-State -Stage "figures" -Status "running"
    & $run figures -Resume -Config $Config
    if ($LASTEXITCODE -ne 0) {
        throw "Figure generation failed with exit code $LASTEXITCODE."
    }
    Write-State -Stage "complete" -Status "completed"
}
catch {
    Write-State -Stage "pipeline" -Status "failed" -Detail $_.Exception.Message
    throw
}
