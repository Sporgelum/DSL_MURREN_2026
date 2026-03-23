$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPath = Join-Path $Root ".venv"

if (-not (Test-Path $VenvPath)) {
    python -m venv .venv
}

$Python = Join-Path $VenvPath "Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.lock.txt

Write-Host "Done. Virtual environment ready at: $VenvPath"
Write-Host "Run app with: .\\scripts\\run_local.ps1"
