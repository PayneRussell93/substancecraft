# SubstanceCraft Feature Map

This is the product capability map derived from the supplied Painter feature inventory. It is a clean-room implementation plan, not a claim of parity. Status is conservative: **Partial** means only a starter foundation exists; **Planned** means not implemented.

## Status key

- **Partial:** a relevant foundation or narrow slice exists.
- **Planned:** no user-ready implementation exists yet.

## Capability areas

| # | Area | Status | Initial acceptance target |
| --- | --- | --- | --- |
| 01 | Project and document management | Partial | Versioned JSON project model, validation, atomic `.sproject` save/load, texture sets, and named poses are implemented; mesh assets remain external references, with recovery, autosave, migration, templates, and mesh replacement still open. |
| 02 | Mesh importing and preparation | Partial | Binary FBX 7.4+ static mesh geometry imports through the API and desktop picker, with native UVs or planar fallback; ASCII FBX, glTF/GLB, material slots, model transforms, and skeletal animation remain open. |
| 03 | Texture Sets | Partial | Derive independent sets from material slots with per-set channels, resolution, maps, and export. |
| 04 | Material channels | Partial | Channel-aware composition for color, scalar, normal, height, opacity, and emissive data. |
| 05 | Painting tools | Planned | Paint, erase, path, projection, polygon fill, smudge, clone, picker, and quick mask commands. |
| 06 | Brush and stroke systems | Planned | Deterministic stamps, dynamics, alphas, tablet input, smoothing, and reusable presets. |
| 07 | Symmetry | Planned | Previewed geometric symmetry applied consistently across enabled channels. |
| 08 | Stroke assistance and precision | Planned | Stabilization, straight lines, surface alignment, cursor preview, and seam-aware painting. |
| 09 | Projection systems | Planned | Camera, planar, UV, triplanar, decal, and geometry-conforming projection. |
| 10 | Fill layers | Planned | Nondestructive channel fills with texture resources and projection controls. |
| 11 | Paint layers | Partial | Layer stack model exists; editable raster paint content and per-channel data remain open. |
| 12 | Layer stack | Partial | Stable layer model; groups, effects, masks, ordering, and complete commands remain open. |
| 13 | Layer flattening | Planned | Flatten selected layers/groups while preserving or exporting source structure. |
| 14 | Masks | Planned | Painted, bitmap, procedural, geometry, and UV-tile masks with effect stacks. |
| 15 | Smart Masks | Planned | Reusable, parameterized masks driven by mesh maps. |
| 16 | Generators | Planned | Mesh-map generators for wear, dirt, cavities, position, orientation, and IDs. |
| 17 | Texture generators and procedural resources | Planned | Seeded, parameterized noise, pattern, grunge, and alpha generation. |
| 18 | Filters and adjustment effects | Planned | Nondestructive, channel-scoped filters with deterministic CPU references. |
| 19 | Anchor Points | Planned | Stable references to upstream layer and channel results. |
| 20 | Smart Materials | Planned | Serializable reusable layer trees with exposed parameters and map requirements. |
| 21 | Materials and Substance resources | Planned | Editable material resources, parameters, physical scale, and licensed asset ingestion. |
| 22 | Decals, atlases, and surface details | Planned | Reusable projected details with transforms and geometry conformance. |
| 23 | Spline and path-based details | Planned | Editable surface paths that drive brush, erase, blur, and repeated details. |
| 24 | Vector graphics and text | Planned | Licensed SVG and text resources for alphas, fills, and projection. |
| 25 | Baking Mode | Planned | Separate bake workflow with settings, preview, progress, and diagnostics. |
| 26 | Mesh-map baking | Planned | Normal, world normal, AO, curvature, position, thickness, and ID outputs. |
| 27 | Skew correction and baking correction | Planned | Inspect and paint bake correction maps with edge protection. |
| 28 | Automatic rebaking and baking iteration | Planned | Track dependencies and rebake affected maps on relevant mesh/settings edits. |
| 29 | UV Tile and UDIM workflows | Planned | Multi-tile storage, painting, baking, caching, and export. |
| 30 | 2D and 3D viewports | Partial | Starter-mesh preview exists; real 3D/UV viewports and navigation remain open. |
| 31 | Real-time PBR viewport | Planned | Physically based shading for enabled channels, maps, and environment lighting. |
| 32 | Rendering | Planned | Interactive and export rendering with camera, lighting, and color-managed output. |
| 33 | Shader system | Planned | Built-in metallic/roughness and extensible shader definitions. |
| 34 | OpenPBR support | Planned | Open material representation and tested interchange. |
| 35 | Color management | Planned | OCIO-based resource, viewport, processing, and output transforms. |
| 36 | Asset management | Planned | Searchable, typed project and library resources with provenance and licensing. |
| 37 | Supported resource categories | Planned | Materials, masks, brushes, alphas, textures, filters, generators, decals, and environments. |
| 38 | Texture importing | Planned | Bounded, color-aware import for channel textures, masks, sequences, and environments. |
| 39 | Texture exporting | Planned | Export sets, channels, UDIMs, maps, and flattened selections. |
| 40 | Export presets and channel packing | Planned | Serializable output graphs for routing, packing, inversion, formats, and naming. |
| 41 | Pipeline-targeted exporting | Planned | Documented, tested profiles for glTF, game engines, renderers, and custom pipelines. |
| 42 | History and nondestructive editing | Partial | Named pose interpolation exists; project-wide commands, undo/redo, and snapshots remain open. |
| 43 | Selection and isolation | Planned | Texture Set, polygon, element, UV island, layer, and mask selection. |
| 44 | Interface customization | Partial | Initial dock-like panels exist; saved layouts and complete workspace controls remain open. |
| 45 | Color selection | Planned | Managed color picker and multi-channel material sampling. |
| 46 | Display and diagnostics | Partial | Channel selector and basic status display exist; map views and diagnostic panels remain open. |
| 47 | Physical size | Planned | Physical-unit-aware scale for procedural resources and fills. |
| 48 | Automation and scripting API | Partial | Public Rust API facade and MCP starter tools are being established; stable command coverage remains open. |
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