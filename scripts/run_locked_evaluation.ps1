[CmdletBinding()]
param(
    [string]$ProjectRoot,
    [string]$Config = "configs\base.toml"
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$tokenBytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($tokenBytes)
$rng.Dispose()
$env:GRIDIMPAIR_UNLOCK_TOKEN = (
    [BitConverter]::ToString($tokenBytes).Replace("-", "")
)
try {
    & (Join-Path $ProjectRoot "run.ps1") evaluate -Resume -Config $Config
    exit $LASTEXITCODE
}
finally {
    Remove-Item Env:\GRIDIMPAIR_UNLOCK_TOKEN -ErrorAction SilentlyContinue
    [Array]::Clear($tokenBytes, 0, $tokenBytes.Length)
}
