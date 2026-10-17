param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw 'python is required'
}

& $python.Source -m py_compile `
    (Join-Path $Root 'data_loader.py') `
    (Join-Path $Root 'main_train.py') `
    (Join-Path $Root 'engine_train.py') `
    (Join-Path $Root 'model.py') `
    (Join-Path $Root 'eval.py') `
    (Join-Path $Root 'utils.py')

& $python.Source -m pytest (Join-Path $Root 'tests') -q
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host 'NEWSTRIDE verification passed.'
