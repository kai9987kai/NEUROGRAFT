param([ValidateSet('Live','Report','Doctor','Test')][string]$Mode = 'Live')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($Mode -eq 'Live') {
    python -m neurograft live --interval 0.2 --autosave 5
} elseif ($Mode -eq 'Doctor') {
    python -m neurograft doctor
} elseif ($Mode -eq 'Test') {
    python -m pytest -q
} else {
    $reportPath = Join-Path $PSScriptRoot 'reports\index.html'
    if (-not (Test-Path -LiteralPath $reportPath)) {
        throw 'The saved report is missing. Generate it with python -m neurograft report.'
    }
    Start-Process -FilePath $reportPath
}
