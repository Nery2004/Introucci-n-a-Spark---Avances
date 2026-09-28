$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$hadoopHome = Join-Path $projectRoot '.tools\hadoop-3.3.6'
$binDirectory = Join-Path $hadoopHome 'bin'
$winutils = Join-Path $binDirectory 'winutils.exe'
$hadoopDll = Join-Path $binDirectory 'hadoop.dll'
$downloadUrl = 'https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin/winutils.exe'
$dllDownloadUrl = 'https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin/hadoop.dll'

New-Item -ItemType Directory -Path $binDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $winutils)) {
    Write-Output "Descargando winutils.exe compatible con Hadoop 3.3.6..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $winutils
}
if (-not (Test-Path -LiteralPath $hadoopDll)) {
    Write-Output "Descargando hadoop.dll compatible con Hadoop 3.3.6..."
    Invoke-WebRequest -Uri $dllDownloadUrl -OutFile $hadoopDll
}

Write-Output "HADOOP_HOME local: $hadoopHome"
Get-FileHash -LiteralPath $winutils, $hadoopDll -Algorithm SHA256 | Format-List
& $winutils systeminfo
