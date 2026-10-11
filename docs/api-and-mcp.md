# API and MCP

## Rust API

`substancecraft-api` is the headless facade for deterministic domain operations. It currently exposes:

- `list_primitive_presets()` for the supported starter-mesh names.
- `inspect_primitive_preset(name)` for indexed topology counts, local-space bounds, and UV bounds.
- `import_binary_fbx_meshes(bytes)` for static mesh geometry from binary FBX 7.4+ files.
- `bake_texture_set_base_color(texture_set)` for a bounded RGBA8 composite respecting layer opacity, grayscale masks, and optional ID-color clipping.
- `bake_texture_set_id_mask(texture_set)` for the discrete color-coded ID map painted into UV layers.
- `bake_texture_set_channel(texture_set, channel, mesh, mesh_maps)` for Base Color, Roughness, Metallic, or ID. Fill layers, mask generators (which need `mesh_maps`), polygon masks (which need `mesh`), painted masks, and ID clipping all compose here.
- `bake_mesh_maps(mesh, smoothing, settings)` for tangent-normal, world-normal, position, curvature, ambient-occlusion, and thickness maps after applying mesh smoothing and subdivision.
- `bake_mesh_maps_high_to_low(low, high, smoothing, settings)` for the same maps, projected from a high-resolution mesh through a cage (`settings.cage_distance`); `MeshMaps::miss_fraction()` reports texels whose rays missed.
- `bake_texture_set_material(texture_set, mesh, mesh_maps)` for the viewport and engine trio: opaque Base Color, packed ORM (R occlusion, G roughness, B metallic), and an OpenGL-convention tangent-space Normal combining baked detail, Normal-channel layers, and Height.
- `encode_project_json(project)` and `decode_project_json(bytes)` for validated, size-bounded project serialization.
- `save_project_file(path, project)` and `load_project_file(path)` for atomic `.sproject` file persistence.

The result types use Serde and plain numeric arrays at the API boundary, so clients do not need to depend on the renderer's math or UI types. Invalid preset names and over-budget texture bakes return typed errors. The desktop exports baked channels and mesh maps as 32-bit TGA. Mesh maps are geometry-derived (no high-to-low projection yet) and import-derived ID maps are not implemented.

The desktop Open/Save controls use the same project file API. Texture sets and saved named poses are serialized and validated on both save and load. Imported FBX geometry is currently referenced by its original file path and reimported on open; it is not embedded in the project package yet.

## MCP server

The `substancecraft-mcp` binary uses the official `rmcp` Rust SDK and stdio transport. It currently exposes these read-only tools:

| Tool | Input | Result |
| --- | --- | --- |
| `list_mesh_presets` | None | JSON list of starter meshes. |
| `inspect_mesh_preset` | `name` | JSON topology, bounds, and UV bounds. |

Run it with `cargo run -p substancecraft-mcp`. Configure an MCP client to launch that command from the SubstanceCraft workspace directory. Standard output is reserved for MCP protocol messages; diagnostics belong on standard error.

The current MCP tools inspect generated presets only; they do not read arbitrary files or mutate a project. The desktop uses the Rust API for local file import after checking the file-size limit. Binary FBX import preserves supported FBX UV mappings, creates a planar UV fallback when a mesh has no UV layer, and preflights declared decoded data against a memory budget. ASCII FBX, materials, model transforms, skins, and animation curves are not imported yet. Future MCP tools should call the same versioned command API as the desktop and CLI, validate all input, and return actionable tool errors rather than panicking.

## Verification

Run `cargo test --workspace` for all tests, `cargo clippy --workspace --all-targets -- -D warnings` for lints, and `cargo fmt --all --check` for formatting.