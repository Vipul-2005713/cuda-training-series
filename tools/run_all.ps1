param(
    [string]$Suite = 'all',
    [string[]]$Only,
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$pythonCommand = Get-Command python,python3,py -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pythonCommand) {
    $pythonPath = $pythonCommand.Source
} else {
    $pythonPath = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Install Python 3 or put it on PATH.' }
}
$runnerArgs = @((Join-Path $PSScriptRoot 'run_exercises.py'), '--suite', $Suite)
if (-not $NoBuild) { $runnerArgs += '--build' }
if ($Only) { $runnerArgs += '--only'; $runnerArgs += $Only }
Push-Location $root
try { & $pythonPath @runnerArgs; $result = $LASTEXITCODE }
finally { Pop-Location }
exit $result
