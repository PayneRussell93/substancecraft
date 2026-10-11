# SubstanceCraft Feature Map

This is the product capability map derived from the supplied Painter feature inventory. It is a clean-room implementation plan, not a claim of parity. Status is conservative: **Partial** means only a starter foundation exists; **Planned** means not implemented.

## Status key

- **Partial:** a relevant foundation or narrow slice exists.
- **Planned:** no user-ready implementation exists yet.

## Capability areas

| # | Area | Status | Initial acceptance target |
| --- | --- | --- | --- |
| 01 | Project and document management | Partial | Versioned JSON project model, validation, atomic `.sproject` save/load, texture sets, named poses, bitmap resource references, and custom export presets are implemented, and the desktop app opens a `.sproject` passed as its first argument; mesh and bitmap assets remain external references, with recovery, autosave, migration, templates, and mesh replacement still open. |
| 02 | Mesh importing and preparation | Partial | Binary FBX 7.4+ and glTF 2.0 (`.gltf` with external or embedded buffers, and `.glb`) static meshes import through the API and desktop picker, with native UVs or planar fallback. glTF flattens the default scene with node transforms applied and maps each material to a material slot. Auto-smoothed shading normals (angle threshold, welded across unshared FBX corners) and crease-aware Loop subdivision (0–3 levels, UV seams preserved) are implemented; ASCII FBX, FBX material slots, and skeletal animation remain open. |
| 03 | Texture Sets | Partial | Multi-material meshes get one texture set per material slot, each with its own layers and channels; unselected sets bake in the background so every slot shows its own material in the viewport. Per-set resolution and per-set export settings remain open. |
| 04 | Material channels | Partial | Base Color (RGBA); Roughness, Metallic, Height, and Opacity (scalar); Emissive (sRGB); Normal (tangent-space vector); and baked Ambient Occlusion composite per layer with per-layer channel enablement and export. Height converts to normal detail (whiteout blend) with a per-texture-set strength; user-defined channels remain open. |
| 05 | Painting tools | Partial | Multi-channel paint, erase, grayscale-mask, and ID-color freehand brushes work on nearest-triangle mesh hits in the 3D viewport and directly in texture space in the UV viewport; the Select tool builds polygon masks by face, connected element, or UV island. Path, projection, smudge, clone, picker, and quick mask remain open. |
| 06 | Brush and stroke systems | Partial | Persistent interpolated UV strokes with Round, Square, Splatter, and bitmap-alpha tips; hardness, spacing, angle, follow-path rotation, seeded size/angle jitter and scatter, and pen pressure driving radius and/or opacity. Stabilization and reusable brush presets remain open. |
| 07 | Symmetry | Planned | Previewed geometric symmetry applied consistently across enabled channels. |
| 08 | Stroke assistance and precision | Planned | Stabilization, straight lines, surface alignment, cursor preview, and seam-aware painting. |
| 09 | Projection systems | Planned | Camera, planar, UV, triplanar, decal, and geometry-conforming projection. |
| 10 | Fill layers | Partial | Fill layers write uniform values or per-channel bitmaps (Base Color, Roughness, Metallic, Height, Normal, Emissive, Opacity) through UV or triplanar projection with scale, offset, rotation, and blend sharpness, and composite through generator, painted, polygon, bitmap, and ID masks. Physical-scale projection remains open. |
| 11 | Paint layers | Partial | Persistent multi-channel paint, erase, and ID strokes with visibility, opacity, filters, undo/redo, and CPU compositing. |
| 12 | Layer stack | Partial | Ordered paint and fill layers plus nestable groups (children composite first, then blend through the group's opacity, mask, and filters), with add, delete, reorder, move into/out of groups, visibility, opacity, rename, and channel toggles, all undoable. Blend modes remain open. |
| 13 | Layer flattening | Planned | Flatten selected layers/groups while preserving or exporting source structure. |
| 14 | Masks | Partial | Mask = generator base, painted grayscale strokes over it, multiplied by an optional projected bitmap mask (invertible), then polygon (geometry) and ID-colour clipping, with polygon-mask inversion and an ordered mask filter stack. UV-tile masks remain open. |
| 15 | Smart Masks | Partial | Edge Wear and Dirt one-click layers pair a fill with a mesh-map-driven generator; saving reusable custom presets remains open. |
| 16 | Generators | Partial | Edge Wear (convex curvature), Dirt (AO + cavity), Orientation, Gradient, and Noise generators with seeded object-space breakup. ID and thickness generators remain open. |
| 17 | Texture generators and procedural resources | Partial | Deterministic seeded 3D gradient noise and fBm (seamless across UV seams); patterns, grunges, and alphas remain open. |
| 18 | Filters and adjustment effects | Partial | Blur, Sharpen (unsharp mask), Levels, Invert, and Hue/Saturation filters apply as ordered, undoable stacks to a layer's content or its mask, on the deterministic CPU path; blur radii scale with texture resolution. Additional filters and per-channel filter scoping remain open. |
| 19 | Anchor Points | Planned | Stable references to upstream layer and channel results. |
| 20 | Smart Materials | Planned | Serializable reusable layer trees with exposed parameters and map requirements. |
| 21 | Materials and Substance resources | Planned | Editable material resources, parameters, physical scale, and licensed asset ingestion. |
| 22 | Decals, atlases, and surface details | Planned | Reusable projected details with transforms and geometry conformance. |
| 23 | Spline and path-based details | Planned | Editable surface paths that drive brush, erase, blur, and repeated details. |
| 24 | Vector graphics and text | Planned | Licensed SVG and text resources for alphas, fills, and projection. |
| 25 | Baking Mode | Partial | Bake drawer with resolution, AO samples/distance, edge width/angle, cage distance, padding, an optional high-poly source, background baking with progress and cancel, ray-miss diagnostics, per-map preview, and TGA export. Bake logs remain open. |
| 26 | Mesh-map baking | Partial | Tangent-space normal, world normal, position, curvature, BVH-raycast AO, and thickness, baked from the working mesh or high-to-low from a merged high-poly FBX via cage rays (misses fall back to the low surface and are reported). Tangent frames are shared with the renderer. Per-mesh cage editing and name-matched bakes remain open. |
| 27 | Skew correction and baking correction | Planned | Inspect and paint bake correction maps with edge protection. |
| 28 | Automatic rebaking and baking iteration | Partial | Mesh maps are keyed to the mesh revision and rebake automatically in the background when a generator or map preview needs them after mesh, scale, or smoothing changes. |
| 29 | UV Tile and UDIM workflows | Planned | Multi-tile storage, painting, baking, caching, and export. |
| 30 | 2D and 3D viewports | Partial | Filament 3D viewport with full-material and single-channel/mesh-map previews, live background re-baking while painting, and picking for paint and selection; a 2D UV viewport (Model, Split, or UV layouts) shows the selected texture set over its UV wireframe on a checkerboard, with zoom, pan, and texture-space painting. UDIM tile display remains open. |
| 31 | Real-time PBR viewport | Partial | Filament metal/rough shading from baked Base Color, packed ORM (mesh-map AO, roughness, metallic), tangent-space normal, and emissive maps with smoothed tangent frames and alpha-tested opacity; image-based lighting environments and global illumination remain open. |
| 32 | Rendering | Planned | Interactive and export rendering with camera, lighting, and color-managed output. |
| 33 | Shader system | Partial | One built-in metallic/roughness viewport material with material and flat-preview modes; extensible shader definitions remain open. |
| 34 | OpenPBR support | Planned | Open material representation and tested interchange. |
| 35 | Color management | Planned | OCIO-based resource, viewport, processing, and output transforms. |
| 36 | Asset management | Planned | Searchable, typed project and library resources with provenance and licensing. |
| 37 | Supported resource categories | Partial | Imported bitmaps serve as fill-channel textures, layer masks, and brush alphas; materials, filters, generators, decals, and environments as library resources remain open. |
| 38 | Texture importing | Partial | PNG, JPEG, and TGA import as project bitmap resources (straight-alpha RGBA8, bounded to 8192 px edges and 256 MiB files) with bilinear sampling; color-space tagging, sequences, and environments remain open. |
| 39 | Texture exporting | Partial | Base Color, Roughness, Metallic, Height, Normal, Emissive, Opacity, ID, every mesh map, and any export preset's file set export as PNG or TGA, written as 8-bit Gray, RGB, or RGBA to match each output's layout; EXR/16-bit output and UDIMs remain open. |
| 40 | Export presets and channel packing | Partial | Serializable export presets route material channels or mesh maps into each output channel with component selection, inversion, constants, Gray/RGB/RGBA layouts, `$textureSet` file naming, and skip-if-black. Built-in Metal/Rough (ORM), Unreal Engine, Unity (URP Lit), and Separate channels presets ship; custom presets are validated, saved in the project, and edited in the desktop export panel (duplicate, edit, delete). Masks are composited once per export. Per-output file formats and bit depths, and undo for preset edits, remain open. |
| 41 | Pipeline-targeted exporting | Partial | Built-in Unreal Engine (DirectX normal, `T_` naming, packed ORM) and Unity URP Lit (metallic + smoothness in alpha, separate occlusion) texture presets, with tested routing; glTF material files, renderer profiles, and engine-side import documentation remain open. |
| 42 | History and nondestructive editing | Partial | Bounded undo/redo (Ctrl+Z / Ctrl+Shift+Z) covers strokes, layer settings (fill, generator, polygon and bitmap masks, filters, opacity, visibility, channels), and layer and group insert/remove/move; texture-set and mesh edits and persisted history remain open. |
| 43 | Selection and isolation | Partial | Viewport picking selects faces, connected mesh elements, or UV islands into a layer polygon mask, with hover preview and Shift-click removal; viewport isolation remains open. |
| 44 | Interface customization | Partial | Initial dock-like panels exist; saved layouts and complete workspace controls remain open. |
| 45 | Color selection | Planned | Managed color picker and multi-channel material sampling. |
| 46 | Display and diagnostics | Partial | Channel selector and basic status display exist; map views and diagnostic panels remain open. |
| 47 | Physical size | Planned | Physical-unit-aware scale for procedural resources and fills. |
| 48 | Automation and scripting API | Partial | Public Rust API facade (including preset-driven export baking and writing) and read-only MCP tools for mesh and export presets; stable command coverage remains open. |
| 49 | VFX and production-pipeline support | Planned | UDIM, OCIO, automation, deterministic headless jobs, and release compatibility targets. |
| 50 | Substance ecosystem integration | Planned | Interchange based on documented open standards and licensed resources; no proprietary project-format dependency. |

## Delivery order

1. **Foundation and automation:** project schema, mesh/pose data, command API, MCP tools, tests, and clean-room CI.
2. **Mesh-to-preview loop:** robust FBX and glTF import, UV/material sets, actual 3D viewport, camera controls, and channel inspection.
3. **First texturing loop:** UV paint, texture tiles, layer data, deterministic composition, undo/redo, and PNG export.
4. **Material authoring:** multi-channel brushes, masks, fill layers, filters, and export profiles.
5. **Production workflows:** skeletal animation pose selection, projection, baking, generators, UDIMs, and higher-resolution performance.
6. **Extensibility:** versioned public API, MCP resources/prompts/tools, scripting, plugins, and pipeline integrations.

Feature completion is judged by observable behavior and acceptance tests, not by menu presence. Asset libraries, shaders, and reference data must be original or redistributable under an explicit license.