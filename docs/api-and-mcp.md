# API and MCP

## Rust API

`substancecraft-api` is the headless facade for deterministic domain operations. It currently exposes:

- `list_primitive_presets()` for the supported starter-mesh names.
- `inspect_primitive_preset(name)` for indexed topology counts, local-space bounds, and UV bounds.
- `import_fbx_meshes(bytes)` for mesh geometry from binary or ASCII FBX 7.4+ files (`import_binary_fbx_meshes(bytes)` accepts binary only), `import_gltf_file(path)` for glTF 2.0, and `import_step_file(path)` / `import_step(bytes, name)` for STEP CAD files (`import_step_with_report(bytes, name, &StepTessellation)` sets the chord tolerance and edge angle and reports bodies, faces, and anything skipped). `load_mesh_file(path)` picks the importer by extension. When a file skins or animates a mesh, its `ImportedMesh::rig` holds the skeleton, up to four skin weights per vertex, and the animation clips; `Rig::pose_positions(positions, scale, sample)` poses the mesh at rest or at a time in a clip (optionally in place).
- `load_environment_file(path)` reads a Radiance `.hdr` or OpenEXR equirectangular panorama into the viewport's `EnvironmentImage` layout (`decode_environment(bytes, extension)` for bytes); `EnvironmentImage::normalized_lux()` gives the lux per unit of radiance that lights it at the presets' average brightness.
- `bake_texture_set_base_color(texture_set)` for a bounded RGBA8 composite respecting layer opacity, grayscale masks, and optional ID-color clipping.
- `bake_texture_set_id_mask(texture_set)` for the discrete color-coded ID map painted into UV layers.
- `bake_texture_set_channel(texture_set, channel, mesh, mesh_maps)` for Base Color, Roughness, Metallic, or ID. Fill layers, mask generators (which need `mesh_maps`), polygon masks (which need `mesh`), painted masks, and ID clipping all compose here.
- `bake_mesh_maps(mesh, smoothing, settings)` for tangent-normal, world-normal, position, curvature, ambient-occlusion, and thickness maps after applying mesh smoothing and subdivision.
- `bake_mesh_maps_high_to_low(low, high, smoothing, settings)` for the same maps, projected from a high-resolution mesh through a cage (`settings.cage_distance`); `MeshMaps::miss_fraction()` reports texels whose rays missed.
- `bake_texture_set_material(texture_set, mesh, mesh_maps)` for the viewport and engine trio: opaque Base Color, packed ORM (R occlusion, G roughness, B metallic), and an OpenGL-convention tangent-space Normal combining baked detail, Normal-channel layers, and Height.
- `read_openpbr_materialx(path)` for the first OpenPBR surface in a MaterialX document: per input its texture file and colour space or its constant value, following nodegraph outputs, direct node bindings, and normal maps. `openpbr_fill(material, resources)` maps it onto fill-layer settings and channels, converting constant colours from the document's working space; `openpbr_texture_color_space` gives the colour space to tag each texture with.
- `encode_project_json(project)` and `decode_project_json(bytes)` for validated, size-bounded project serialization.
- `save_project_file(path, project)` and `load_project_file(path)` for atomic `.sproject` file persistence.
- `project_summary(project)` and `apply_automation_edit(project, edit)` / `edit_project_file(path, edit)` for the automation API (version `AUTOMATION_API_VERSION`, currently 1). An `AutomationEdit` is JSON tagged by `edit`: `add_fill_layer`, `add_layer` (a paint layer or group, optionally inside a group at an index), `set_layer`, `move_layer`, `remove_layer`, `set_mask` (mask generator and final inversion), `paint` and `paint_mask` (one stroke, points in UV and radius in texels), `edit_material`, `link_material`, `set_texture_set` (name, resolution, channels, height strength, user channels as `[{"slot": 0, "name": "Wetness", "gray": true, "default": [0, 0, 0]}]`; channels are named like `Base Color` or `User 0`), `set_mesh` (scale, smoothing angle, subdivision levels; polygon masks follow a subdivision change), `set_filters` (content and mask filter stacks, and the channels content filters apply to), `set_bitmap_mask` (a project bitmap, its inversion, and projection fields such as `{"mode": "Triplanar", "scale": 2}`), `set_polygon_mask` (triangle indices, inversion; `[]` removes it), `set_anchor_mask` (an earlier layer, the channel read, low/high remap, inversion, and the `effect` read at: `layer`, `fill`, `paint`, or `filter N`), `set_graph` (a fill layer's procedural `MaterialGraph` as saved in project files, or `null` to remove it), `insert_smart_material` (a project or built-in smart material with fresh ids, its exposed parameters set by name), or `apply_smart_mask`. A filter is a name such as `"Blur"` or `"Hue / Saturation"`, or an object such as `{"kind": "Levels", "gamma": 2}` whose fields replace its defaults. `library_summary(project)` lists the smart materials (with exposed parameters, current values, and evolution controls), smart masks, and materials a project can use. A `set_mask` generator is a name such as `"Edge Wear"` or `"none"`, or an object such as `{"kind": "EdgeWear", "amount": 0.8}` whose fields replace those of the layer's generator of that kind (or its defaults); unknown fields and out-of-range values are refused. Texture sets, layers, and materials are referenced by numeric id or exact name; a name shared by several entries is refused. Each edit becomes one `EditCommand` and the project is validated afterwards, so a refused edit leaves the project unchanged.

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
| `add_layer` | `project`, `texture_set`, `name`, optional `kind` (`paint` or `group`), `parent`, `index` | Outcome with the new layer id; the project is saved. |
| `move_layer` | `project`, `texture_set`, `layer`, optional `parent`, `index` | Outcome; the project is saved. |
| `set_mask` | `project`, `texture_set`, `layer`, optional `generator`, `invert` | Outcome naming the generator; the project is saved. |
| `paint_stroke` | `project`, `texture_set`, `layer`, `points` (UV), `radius` (texels), optional `color`, `opacity`, `roughness`, `metallic`, `height`, `erase` | Outcome; the project is saved. |
| `paint_mask` | `project`, `texture_set`, `layer`, `points` (UV), `radius` (texels), `value`, optional `opacity` | Outcome; the project is saved. |
| `set_texture_set` | `project`, `texture_set`, optional `name`, `resolution`, `channels`, `height_strength`, `user_channels` (slot, name, optional `gray` and `default`) | Outcome; the project is saved. |
| `set_mesh` | `project`, optional `scale`, `smoothing_angle`, `subdivision_levels` | Outcome, counting remapped polygon masks; the project is saved. |
| `list_library` | `project` | JSON smart materials (with parameters and controls), smart masks, and materials: the project's, then the built-ins. |
| `set_filters` | `project`, `texture_set`, `layer`, optional `filters`, `mask_filters`, `channels` | Outcome counting both stacks; the project is saved. |
| `set_bitmap_mask` | `project`, `texture_set`, `layer`, optional `resource` (`none` removes), `invert`, `projection` | Outcome; the project is saved. |
| `set_polygon_mask` | `project`, `texture_set`, `layer`, optional `triangles` (`[]` removes), `inverted` | Outcome counting the triangles; the project is saved. |
| `set_graph` | `project`, `texture_set`, `layer` (a fill layer), optional `graph` (a material graph as in project files; null removes) | Outcome counting the nodes; the project is saved. |
| `set_anchor_mask` | `project`, `texture_set`, `layer`, optional `anchor` (`none` removes), `channel`, `low`, `high`, `invert`, `effect` (`layer`, `fill`, `paint`, or `filter N`) | Outcome naming the anchor layer and channel; the project is saved. |
| `insert_smart_material` | `project`, `texture_set`, `smart_material`, optional `parent`, `index`, `parameters` | Outcome with the new root layer id; the project is saved. |
| `apply_smart_mask` | `project`, `texture_set`, `layer`, `smart_mask` | Outcome; the project is saved. |
| `export_textures` | `project`, `output`, optional `preset`, `format` | JSON list of the files written, as `substancecraft-cli export` writes them. |

Run it with `cargo run -p substancecraft-mcp`. Configure an MCP client to launch that command from the SubstanceCraft workspace directory. Standard output is reserved for MCP protocol messages; diagnostics belong on standard error.

Project tools only reach paths inside the server's root directory: `SUBSTANCECRAFT_MCP_ROOT` when set, else the directory the server starts in. Relative paths resolve against it; paths (or, for outputs that do not exist yet, their nearest existing folder) outside it, `..` steps out of it, and files that are not `.sproject` projects are refused. Every edit is saved atomically and only when it changed something.

The command line takes the same edits: `substancecraft-cli inspect <project>` prints the summary, and `substancecraft-cli edit <project> '<edit as JSON>'` applies one edit and prints its outcome (exit status 1 when refused, 2 when the JSON is invalid).

The desktop uses the Rust API for local file import after checking the file-size limit. FBX import preserves supported FBX UV mappings, maps per-polygon materials to named material slots, creates a planar UV fallback when a mesh has no UV layer, and preflights declared decoded data against a memory budget. ASCII FBX is parsed, typed as binary FBX stores each value, and re-encoded as binary, so both forms share one importer. Skins, bone hierarchies, and animation stacks become a rig (curve tangents are not read; keys interpolate linearly); static meshes are imported in geometry space without their model transform. Future MCP tools should call the same versioned command API as the desktop and CLI, validate all input, and return actionable tool errors rather than panicking.

## Verification

Run `cargo test --workspace` for all tests, `cargo clippy --workspace --all-targets -- -D warnings` for lints, and `cargo fmt --all --check` for formatting.