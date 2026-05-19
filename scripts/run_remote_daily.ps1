param(
    [string]$RepoDir = "D:\Quantify\repo",
    [string]$Config = "configs\config.local.yaml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $RepoDir
git pull
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH = Join-Path $RepoDir "src"
python -m quantify.cli --config $Config extract-my-stock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m quantify.cli --config $Config train
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m quantify.cli --config $Config predict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
