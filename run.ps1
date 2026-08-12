[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("audit", "pilot", "simulate", "train", "evaluate", "figures", "all")]
    [string]$Stage,

    [switch]$Resume,

    [string]$Config = "configs\base.toml",

    [ValidateRange(0, 2147483647)]
    [Nullable[int]]$Limit
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw ".venv not found. Please run .\setup.ps1 first."
}

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$ConfigPath = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $ProjectRoot $Config
}

$Arguments = @(
    "-m", "gridimpair.runner",
    "--stage", $Stage,
    "--config", $ConfigPath
)
if ($Resume) {
    $Arguments += "--resume"
}
if ($null -ne $Limit) {
    $Arguments += @("--limit", $Limit)
}

& $Python @Arguments
exit $LASTEXITCODE
