param(
    [string]$Config = "configs\config.local.yaml",
    [int]$Workers = 4,
    [int]$PerPrefixLimit = 40,
    [switch]$Retrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $PSScriptRoot
Set-Location $RepoDir
$env:PYTHONPATH = Join-Path $RepoDir "src"

python -m quantify.cli --config $Config fetch-stock-list
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m quantify.cli --config $Config fetch-index --incremental --workers 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m quantify.cli --config $Config fetch-daily --prefixes "00,60" --per-prefix-limit $PerPrefixLimit --incremental --workers $Workers
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Retrain -or -not (Test-Path "models_real\deep_sequence.pt")) {
    python -m quantify.cli --config $Config train
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python -m quantify.cli --config $Config predict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
