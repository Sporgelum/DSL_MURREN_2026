param(
    [Parameter(Mandatory = $false)]
    [string]$Message = "save step",

    [Parameter(Mandatory = $false)]
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Stage and commit all changes
git add -A

$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "No staged changes. Nothing to commit or push."
    exit 0
}

git commit -m $Message

# Push right after every saved step
git push origin $Branch
Write-Host "Committed and pushed to origin/$Branch with message: $Message"
