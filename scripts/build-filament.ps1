# Builds the vendored Filament fork (third_party/filament) and installs it into
# third_party/filament-install, where crates/filament/build.rs looks for it by default.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts/build-filament.ps1 [-Clean]

param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root 'third_party\filament'
$build = Join-Path $root 'third_party\filament-build'
$install = Join-Path $root 'third_party\filament-install'

if (-not (Test-Path (Join-Path $source 'CMakeLists.txt'))) {
    git -C $root submodule update --init --depth 1 third_party/filament
}

if ($Clean) {
    Remove-Item -Recurse -Force $build, $install -ErrorAction SilentlyContinue
}

# Rust's x86_64-pc-windows-msvc target links the dynamic CRT, so Filament must use /MD.
cmake -S $source -B $build `
    -G 'Visual Studio 17 2022' -A x64 `
    -DUSE_STATIC_CRT=OFF `
    -DFILAMENT_SUPPORTS_VULKAN=OFF `
    -DFILAMENT_SKIP_SAMPLES=ON `
    -DFILAMENT_SKIP_SDL2=ON `
    -DFILAMENT_BUILD_TESTING=OFF `
    -DFILAMENT_ENABLE_MATDBG=OFF `
    "-DCMAKE_INSTALL_PREFIX=$install"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $build --config Release --target install -- /m /v:minimal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Filament installed to $install"
