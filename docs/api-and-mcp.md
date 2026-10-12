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
- `project_summary(project)` and `apply_automation_edit(project, edit)` / `edit_project_file(path, edit)` for the automation API (version `AUTOMATION_API_VERSION`, currently 1). An `AutomationEdit` is JSON tagged by `edit`: `add_fill_layer`, `set_layer`, `remove_layer`, `edit_material`, or `link_material`. Texture sets, layers, and materials are referenced by numeric id or exact name; a name shared by several entries is refused. Each edit becomes one `EditCommand` and the project is validated afterwards, so a refused edit leaves the project unchanged.

The result types use Serde and plain numeric arrays at the API boundary, so clients do not need to depend on the renderer's math or UI types. Invalid preset names and over-budget texture bakes return typed errors. The desktop exports baked channels and mesh maps as 32-bit TGA. Mesh maps are geometry-derived (no high-to-low projection yet) and import-derived ID maps are not implemented.

The desktop Open/Save controls use the same project file API. Texture sets and saved named poses are serialized and validated on both save and load. Imported FBX geometry is currently referenced by its original file path and reimported on open; it is not embedded in the project package yet.

## MCP server

The `substancecraft-mcp` binary uses the official `rmcp` Rust SDK and stdio transport. It exposes these tools:

| Tool | Input | Result |
| --- | --- | --- |
| `list_mesh_presets` | None | JSON list of starter meshes. |
| `inspect_mesh_preset` | `name` | JSON topology, bounds, and UV bounds. |
| `list_export_presets` | None | JSON built-in export presets. |
| `inspect_project` | `project` | JSON project summary. |
| `add_fill_layer` | `project`, `texture_set`, `name`, optional `material`, `color`, `roughness`, `metallic` | Outcome with the new layer id; the project is saved. |
| `set_layer` | `project`, `texture_set`, `layer`, optional `name`, `visible`, `opacity`, `blend_mode` | Outcome; the project is saved. |
| `remove_layer` | `project`, `texture_set`, `layer` | Outcome; the project is saved. |
| `edit_material` | `project`, `material`, optional `name`, `color`, `roughness`, `metallic`, `height`, `opacity` | Outcome naming how many linked layers followed; the project is saved. |
| `link_material` | `project`, `texture_set`, `layer`, `material` | Outcome; the project is saved. |
| `export_textures` | `project`, `output`, optional `preset`, `format` | JSON list of the files written, as `substancecraft-cli export` writes them. |

Run it with `cargo run -p substancecraft-mcp`. Configure an MCP client to launch that command from the SubstanceCraft workspace directory. Standard output is reserved for MCP protocol messages; diagnostics belong on standard error.

Project tools only reach paths inside the server's root directory: `SUBSTANCECRAFT_MCP_ROOT` when set, else the directory the server starts in. Relative paths resolve against it; paths (or, for outputs that do not exist yet, their nearest existing folder) outside it, `..` steps out of it, and files that are not `.sproject` projects are refused. Every edit is saved atomically and only when it changed something.

The command line takes the same edits: `substancecraft-cli inspect <project>` prints the summary, and `substancecraft-cli edit <project> '<edit as JSON>'` applies one edit and prints its outcome (exit status 1 when refused, 2 when the JSON is invalid).

The desktop uses the Rust API for local file import after checking the file-size limit. Binary FBX import preserves supported FBX UV mappings, creates a planar UV fallback when a mesh has no UV layer, and preflights declared decoded data against a memory budget. ASCII FBX, materials, model transforms, skins, and animation curves are not imported yet. Future MCP tools should call the same versioned command API as the desktop and CLI, validate all input, and return actionable tool errors rather than panicking.

## Verification

Run `cargo test --workspace` for all tests, `cargo clippy --workspace --all-targets -- -D warnings` for lints, and `cargo fmt --all --check` for formatting.