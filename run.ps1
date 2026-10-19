# Ejecutar desde PowerShell: .\run.ps1 plan
# No requiere activar el entorno; ETL y modelado necesitan sus archivos requirements.
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
    throw 'No se encontró Python. Instalá Python 3.10 o superior, o creá .venv.'
}
Push-Location $PSScriptRoot
try {
    & $runtimePython -X utf8 -m aire_caba @args
    $resultCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $resultCode
