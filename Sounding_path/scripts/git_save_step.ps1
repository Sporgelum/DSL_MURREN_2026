param(
    [Parameter(Mandatory = $false)]
    [string]$Message = "save step"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Stage everything in the current repository (tracked + untracked)
git add -A

# Skip empty commits to avoid noisy history
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "No staged changes. Nothing to commit."
    exit 0
}

git commit -m $Message
Write-Host "Committed with message: $Message"
