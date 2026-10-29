param([int]$Port = 8770, [int]$Seed = 0)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Write-Host "Replay workbench: http://127.0.0.1:$Port/workbench"
Write-Host 'An isolated experiment tool. The saved live session is not opened.'
python -m neurograft workbench --port $Port --seed $Seed
