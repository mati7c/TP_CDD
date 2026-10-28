# Modelo final PM10 y notebook OMS, conservando el NO2 original.
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (Test-Path -LiteralPath $projectPython) {
    $runtimePython = $projectPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $runtimePython = (Get-Command python).Source
} elseif (Test-Path -LiteralPath $bundledPython) {
    $runtimePython = $bundledPython
} else {
    throw 'No se encontró Python. Creá .venv e instalá requirements-modeling.txt.'
}
Push-Location $PSScriptRoot
try {
    & $runtimePython -B -X utf8 -m aire_caba.pm10_cierre @args
    $resultCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $resultCode
