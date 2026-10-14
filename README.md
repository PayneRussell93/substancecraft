# SubstanceCraft

SubstanceCraft is a clean-room, Rust-native 3D material authoring project. Its target workflow is to import a mesh, paint and composite material channels non-destructively, preview the result with PBR shading, and export textures for real-time and offline renderers.

The project is in early development. It imports FBX (binary or ASCII), glTF, and STEP CAD meshes, with smoothing and subdivision. It paints Base Color, Roughness, Metallic, and ID channels in nondestructive paint and fill layers, masked by painted, procedural (Edge Wear, Dirt, ...), polygon-selection, and ID masks. It bakes geometry-derived mesh maps, renders with Filament, supports undo/redo, and exports TGA textures. See the roadmap for what remains.

## Build

Install Rust with rustup and the stable toolchain. This repository requests the stable toolchain and the `rustfmt` and `clippy` components through `rust-toolchain.toml`.

```powershell
rustup show
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The desktop app links a vendored Filament build. On Windows, install Visual Studio 2022 Build Tools with the C++ workload (MSVC, the Windows SDK, and its bundled CMake) and Python 3, then build Filament once (about an hour):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-filament.ps1
```

The core, API, CLI, and MCP crates build and test without Filament (`cargo test -p substancecraft-core -p substancecraft-api -p substancecraft-cli -p substancecraft-mcp`).

## Direction

The architecture and initial vertical slice are tracked in [the roadmap](ROADMAP.md), [the 50-area feature map](docs/feature-map.md), and [the architecture notes](docs/architecture.md). The API and MCP starter are described in [the API guide](docs/api-and-mcp.md). Quality records are in [Documentation/Quality](Documentation/Quality/mini-QualityPlan.md).

SubstanceCraft is an independent project. PhotoCraft, LightCraft, and VectorCraft inform architectural choices only; no source code, assets, proprietary formats, or product UI assets are copied from them or from Adobe products.