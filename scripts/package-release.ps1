# Builds SubstanceCraft in release mode and packages it for Windows:
#   target/dist/SubstanceCraft-<version>-windows-x64.zip        (portable)
#   target/dist/SubstanceCraft-<version>-windows-x64-setup.exe  (installer)
#
# Needs the Filament build (scripts/build-filament.ps1), Visual Studio 2022
# with the C++ workload (for the VC++ runtime DLLs), and Inno Setup 6 for the
# installer. Without Inno Setup only the zip is produced.
#
#   powershell -ExecutionPolicy Bypass -File scripts/package-release.ps1

param(
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

function Invoke-Checked([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe $($arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

$version = (Select-String -Path Cargo.toml -Pattern '^version\s*=\s*"([^"]+)"' |
    Select-Object -First 1).Matches[0].Groups[1].Value
$name = "SubstanceCraft-$version-windows-x64"
$dist = Join-Path $root 'target\dist'
$stage = Join-Path $dist $name
Write-Host "Packaging $name"

if (-not $SkipBuild) {
    Invoke-Checked cargo @('build', '--release', '-p', 'substancecraft', '-p', 'substancecraft-cli', '-p', 'substancecraft-mcp')
    Invoke-Checked rustup @('target', 'add', 'wasm32-unknown-unknown')
    Push-Location examples\plugins\wavy-bands
    try { Invoke-Checked cargo @('build', '--release', '--target', 'wasm32-unknown-unknown') }
    finally { Pop-Location }
}

# The app and Filament link the VC++ runtime dynamically; ship it app-local so
# the app starts on machines without the VC++ Redistributable.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$crt = Get-ChildItem "$vsRoot\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT" -Directory |
    Sort-Object FullName | Select-Object -Last 1
if (-not $crt) { throw "VC++ runtime redistributables not found under $vsRoot" }

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage, "$stage\docs", "$stage\licenses", "$stage\examples\plugins\wavy-bands" | Out-Null

foreach ($exe in 'substancecraft', 'substancecraft-cli', 'substancecraft-mcp') {
    Copy-Item "target\release\$exe.exe" $stage
}
foreach ($dll in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    Copy-Item (Join-Path $crt.FullName $dll) $stage
}
Copy-Item apps\substancecraft\assets\substancecraft.ico $stage
Copy-Item README.md $stage
Copy-Item docs\*.md "$stage\docs"
Copy-Item third_party\filament\LICENSE "$stage\licenses\Filament-LICENSE.txt"
Copy-Item examples\plugins\wavy-bands\plugin.json "$stage\examples\plugins\wavy-bands"
Copy-Item examples\plugins\wavy-bands\target\wasm32-unknown-unknown\release\wavy_bands.wasm "$stage\examples\plugins\wavy-bands"

$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Write-Host "Wrote $zip"

$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc) {
    Invoke-Checked $iscc @('/Q', "/DAppVersion=$version", "/DStageDir=$stage", "/DOutputDir=$dist", 'installer\substancecraft.iss')
    Write-Host "Wrote $(Join-Path $dist "$name-setup.exe")"
} else {
    Write-Warning 'Inno Setup 6 not found; skipped the installer. Install it from https://jrsoftware.org/isdl.php'
}

Get-ChildItem $dist -File -Filter "$name*" | ForEach-Object {
    '{0}  {1}' -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name
} | Set-Content -Encoding ascii (Join-Path $dist 'SHA256SUMS.txt')
