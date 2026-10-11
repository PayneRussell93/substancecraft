# SubstanceCraft Feature Map

This is the product capability map derived from the supplied Painter feature inventory. It is a clean-room implementation plan, not a claim of parity. Status is conservative: **Partial** means only a starter foundation exists; **Planned** means not implemented.

## Status key

- **Implemented:** the acceptance target is met and covered by automated tests (used for backlog items, whose targets are narrow).
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
| 38 | Texture importing | Partial | PNG, JPEG, and TGA import as project bitmap resources (straight-alpha RGBA8, bounded to 8192 px edges and 256 MiB files) with bilinear sampling, and reload automatically when edited outside the app (QoL-01); color-space tagging, sequences, and environments remain open. |
| 39 | Texture exporting | Partial | Base Color, Roughness, Metallic, Height, Normal, Emissive, Opacity, ID, every mesh map, and any export preset's file set export as PNG or TGA, written as 8-bit Gray, RGB, or RGBA to match each output's layout; EXR/16-bit output and UDIMs remain open. |
| 40 | Export presets and channel packing | Partial | Serializable export presets route material channels or mesh maps into each output channel with component selection, inversion, constants, Gray/RGB/RGBA layouts, `$textureSet` file naming, and skip-if-black. Built-in Metal/Rough (ORM), Unreal Engine, Unity (URP Lit), and Separate channels presets ship; custom presets are validated, saved in the project, and edited in the desktop export panel (duplicate, edit, delete). Masks are composited once per export, and every export passes a preflight check first (QoL-16). Per-output file formats and bit depths, and undo for preset edits, remain open. |
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

## Workflow and quality-of-life backlog

These features add little new capability, but they shorten everyday iteration. They come from a technical-art review of the current build. **High** items save hours per week on large projects and are scheduled ahead of the remaining production work (delivery stage 5); **Medium** and **Low** items are taken opportunistically. Each builds on the capability areas listed and follows the same status rules.

| ID | Feature | Builds on | Priority | Status | Acceptance target |
| --- | --- | --- | --- | --- | --- |
| QoL-01 | Resource auto-reload | 36, 38 | High | Implemented | Watch referenced bitmap files; when one changes on disk, reload it and rebake only the texture sets that use it, with an optional confirm-before-reload mode. A missing or undecodable file keeps the last good pixels and reports the error. Files are polled once a second (modification time and size), which also catches save-by-rename; the Assets drawer offers Auto, Ask (Reload/Ignore prompt), and Off. |
| QoL-02 | Surface-to-layer inspector | 12, 43 | High | Planned | Clicking the model lists, top to bottom, the visible layers and groups that contribute at that texel for the active channel, with each one's effective mask value and the generator or mesh map driving it; choosing an entry selects that layer. |
| QoL-03 | Solo and isolation | 12, 43, 46 | High | Planned | One action solos a channel, layer, group, mask, generator, or texture set in the viewport without changing project data; solo state is never saved or exported. |
| QoL-04 | Layer search and filter | 12 | Medium | Planned | A search box above the layer stack filters by name, layer type, enabled channel, and fill resource, keeping the parent groups of matches visible. |
| QoL-05 | Material instances | 20, 21 | High | Planned | A shared master material drives linked fills across layers and texture sets; editing the master updates every instance, and per-instance overrides are explicit and undoable. |
| QoL-06 | Lazy brush stroke stabilization | 06, 08 | Medium | Planned | An adjustable lag distance smooths raw input before stamping, deterministically for a recorded input sequence. |
| QoL-07 | Hold-to-override hotkeys | 05, 44 | Medium | Planned | Holding a key temporarily switches tool (eraser, picker, pan) and releasing it restores the previous tool and settings; bindings are configurable. Only undo/redo shortcuts exist today. |
| QoL-08 | Brush favorites shelf | 06, 37 | Low | Planned | Brush tips and presets, including bitmap alphas, can be pinned to a quick-access shelf saved with user preferences. |
| QoL-09 | Recent colors and materials | 45 | Low | Planned | A history bar of recent brush colours and multi-channel brush materials; one click restores every channel value. |
| QoL-10 | Mask view mode | 14, 46 | Medium | Planned | A toggle shows the selected layer's final mask (generator, bitmap, strokes, filters, and clipping) as grayscale in the 3D and UV viewports. |
| QoL-11 | Mask debug overlay | 14, 46 | Medium | Planned | An overlay tints the shaded model by the selected mask (active, blocked, partial) with adjustable opacity. |
| QoL-12 | Quick mask invert | 14 | Low | Partial | Polygon masks, bitmap masks, and the Invert filter each invert individually; one control that inverts a layer's final mask remains open. |
| QoL-13 | Lighting check presets | 31, 32 | Medium | Planned | One-click studio, overcast, sunset, warehouse, bright outdoor, and night environments from redistributable HDRIs. The viewport currently offers only light rotation and ambient fill. |
| QoL-14 | Pinned inspection cameras | 30, 32 | Low | Planned | Named cameras are saved in the project, recalled in the viewport, and used by preview renders. |
| QoL-15 | UV seam, hard edge, and tangent split display | 30, 46 | Medium | Planned | An overlay draws UV seams, hard (split-normal) edges, and tangent-frame splits on the 3D model to diagnose normal-map issues. |
| QoL-16 | Export preflight | 39, 40 | High | Partial | Before writing, the chosen preset is checked against the texture set: inputs left at defaults (no baked AO, flat normal), per-profile resolution limits, file-name collisions, and overwrites. Warnings can be acknowledged; errors block the export. The export panel lists findings live without baking: errors (invalid preset, missing mesh maps, mesh, or bitmaps, over-budget resolution, colliding file names) block the export, warnings (unpainted channels, flat normal, white AO, empty ID map, non-power-of-two size) are informational, and replacing existing files asks first. Per-profile resolution limits remain open, since presets do not yet carry a maximum size. |
| QoL-17 | One-click engine export | 40, 41 | Medium | Partial | Unreal Engine and Unity (URP Lit) presets exist in the export picker; direct per-engine export actions, and Godot, CryEngine, and glTF material profiles, remain open. |
| QoL-18 | Stale bake detection | 28, 46 | High | Partial | Mesh maps are keyed to the mesh revision and rebake automatically, and the Bake drawer shows Up to date / Not baked / Baking; per-map staleness with its cause (mesh, scale, smoothing, high-poly source, or bake settings) remains open. |
| QoL-19 | Smart import assignment | 38 | High | Planned | When several bitmaps are imported together, a channel is suggested for each from name tokens (basecolor/albedo, normal, rough, metal, ao, height, emissive, opacity) and image traits; the user confirms before fills are created. |
| QoL-20 | Material evolution slider | 20 | Low | Planned | One exposed parameter on a smart material (for example clean to worn) drives several generator and fill parameters through user-defined ranges, as a single undoable edit. |

## Delivery order

1. **Foundation and automation:** project schema, mesh/pose data, command API, MCP tools, tests, and clean-room CI.
2. **Mesh-to-preview loop:** robust FBX and glTF import, UV/material sets, actual 3D viewport, camera controls, and channel inspection.
3. **First texturing loop:** UV paint, texture tiles, layer data, deterministic composition, undo/redo, and PNG export.
4. **Material authoring:** multi-channel brushes, masks, fill layers, filters, and export profiles.
5. **Workflow acceleration:** the High-priority backlog items: resource auto-reload, surface-to-layer inspector, solo and isolation, material instances, export preflight, stale bake detection, and smart import assignment.
6. **Production workflows:** skeletal animation pose selection, projection, baking, generators, UDIMs, and higher-resolution performance.
7. **Extensibility:** versioned public API, MCP resources/prompts/tools, scripting, plugins, and pipeline integrations.

Feature completion is judged by observable behavior and acceptance tests, not by menu presence. Asset libraries, shaders, and reference data must be original or redistributable under an explicit license.