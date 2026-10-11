//! Links the vendored Filament fork (third_party/filament), compiles the viewport material
//! with that build's `matc`, and builds the C ABI shim in `cpp/shim.cpp`.
//!
//! Filament is located through `FILAMENT_DIR`, defaulting to `third_party/filament-install`
//! as produced by `scripts/build-filament.ps1`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const FILAMENT_LIBS: &[&str] = &[
    "filament",
    "backend",
    "bluegl",
    "filabridge",
    "filaflat",
    "geometry",
    "ibl-lite",
    "shaders",
    "smol-v",
    "utils",
    "abseil",
    "zstd",
];

fn main() {
    println!("cargo:rerun-if-env-changed=FILAMENT_DIR");
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=materials/viewport.mat");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let filament_dir = env::var_os("FILAMENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../third_party/filament-install"));
    let include_dir = filament_dir.join("include");
    let lib_dir = filament_dir.join("lib").join("x86_64");
    let matc = filament_dir
        .join("bin")
        .join(if cfg!(windows) { "matc.exe" } else { "matc" });
    if !include_dir.is_dir() || !lib_dir.is_dir() || !matc.is_file() {
        fail(&format!(
            "Filament was not found at {}. Build the vendored fork with \
             `powershell -ExecutionPolicy Bypass -File scripts/build-filament.ps1`, \
             or point FILAMENT_DIR at an existing Filament install.",
            filament_dir.display()
        ));
    }
    println!("cargo:rerun-if-changed={}", lib_dir.join("filament.lib").display());

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap_or_default());
    compile_material(
        &matc,
        &manifest_dir.join("materials/viewport.mat"),
        &out_dir.join("viewport.filamat"),
    );

    cc::Build::new()
        .cpp(true)
        .file("cpp/shim.cpp")
        .include(&include_dir)
        .std("c++20")
        .flag_if_supported("/EHsc")
        .flag_if_supported("/GR-")
        .flag_if_supported("/Zc:__cplusplus")
        .flag_if_supported("/permissive-")
        .flag_if_supported("-fno-rtti")
        .warnings(false)
        .compile("substancecraft_filament_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for lib in FILAMENT_LIBS {
        if lib_dir.join(format!("{lib}.lib")).is_file() || lib_dir.join(format!("lib{lib}.a")).is_file()
        {
            println!("cargo:rustc-link-lib=static={lib}");
        }
    }
    if cfg!(windows) {
        for lib in ["opengl32", "gdi32", "user32", "shlwapi"] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
}

fn compile_material(matc: &Path, source: &Path, output: &Path) {
    let status = Command::new(matc)
        .args(["--platform", "desktop", "--api", "opengl", "--output"])
        .arg(output)
        .arg(source)
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => fail(&format!("matc failed with {status} compiling {}", source.display())),
        Err(error) => fail(&format!("could not run {}: {error}", matc.display())),
    }
}

fn fail(message: &str) -> ! {
    println!("cargo:warning={message}");
    std::process::exit(1);
}
