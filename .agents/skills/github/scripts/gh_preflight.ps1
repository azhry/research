[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[^/\s]+/[^/\s]+$')]
    [string]$Repository
)

$ErrorActionPreference = 'Stop'

$gh = Get-Command gh.exe -All |
    Where-Object { $_.Source -notmatch '(?i)node_modules|\\nodejs\\gh(?:\.cmd|\.ps1)?$' } |
    Select-Object -First 1 -ExpandProperty Source

if (-not $gh) {
    throw 'Official GitHub CLI executable not found.'
}

& $gh --version
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI version check failed with exit code $LASTEXITCODE."
}

& $gh auth status
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI authentication check failed with exit code $LASTEXITCODE."
}

$repositoryJson = & $gh repo view $Repository --json nameWithOwner,viewerPermission
if ($LASTEXITCODE -ne 0) {
    throw "GitHub repository access check failed with exit code $LASTEXITCODE."
}

$repositoryResult = $repositoryJson | ConvertFrom-Json
if ($repositoryResult.nameWithOwner -ne $Repository) {
    throw "GitHub repository check returned '$($repositoryResult.nameWithOwner)' instead of '$Repository'."
}

if ($repositoryResult.viewerPermission -notin @('ADMIN', 'MAINTAIN', 'WRITE')) {
    throw "GitHub repository permission '$($repositoryResult.viewerPermission)' does not allow delivery."
}

$repositoryResult | ConvertTo-Json -Compress
