param(
    [string]$Config = "configs\config.smoke.yaml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$env:PYTHONPATH = Join-Path (Get-Location) "src"
python -m quantify.cli --config $Config make-sample-data
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m quantify.cli --config $Config train
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m quantify.cli --config $Config predict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
