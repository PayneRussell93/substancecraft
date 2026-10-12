# SubstanceCraft Architecture

## Product model

SubstanceCraft treats a project as a mesh plus one or more material texture sets. A texture set owns a resolution, a material-slot identity, enabled PBR channels, and a stack of material layers. Compositing is channel-aware: color, scalar values, normals, and height cannot all use the same ordinary RGBA blend operation.

The intended data flow is:

```text
Mesh + material slots
  -> UVs and texture sets
  -> mesh maps
  -> non-destructive material layers
  -> per-channel compositing
  -> PBR viewport and texture export
```

## Design boundaries

- Keep project data and deterministic domain operations independent of the UI toolkit.
- Route future user-visible edits through explicit commands so undo, automation, and the desktop UI share behavior.
- Use stable typed IDs, schema versions, and validation at project-load boundaries.
- Keep channel compositing specific to channel semantics; normal maps and scalar maps are not color images.
- Add tile-level storage and dirty-region recomputation before production texture sizes make full-surface updates expensive.
- Continue from the initial UV-space paint/mask/bake loop with stroke history and richer channel compositing. Defer projection painting, UDIM, geometric mesh-map baking, and procedural materials until those paths are validated.

## Research use and clean-room boundary

PhotoCraft's public architecture notes inform the separation of core data, commands, history, and UI. LightCraft and VectorCraft provide additional examples of Rust workspace organization and clean-room product development. These repositories are architectural references only; SubstanceCraft does not depend on their crates or copy their source.

The product brief supplied for this project is the initial behavioral scope. Implementations and assets must be original or based on public standards and appropriately licensed research. Do not inspect or copy Adobe application binaries, source, shaders, icons, presets, or proprietary project files.

## Current implementation

`substancecraft-core` provides serializable project, mesh reference, texture set, saved pose, layer, channel, mesh geometry, and resolution types, organised as:

- `lib.rs`: project model, layers, compositing (`TextureSet::bake_channel`), and `EditHistory`.
- `smoothing.rs`: auto-smoothed corner/vertex normals and crease-aware Loop subdivision. Both weld by position, so unshared per-corner FBX vertices smooth correctly.
- `topology.rs`: per-triangle connected-element and UV-island membership for polygon selection.
- `mesh_maps.rs`: multi-threaded, cancellable baking of world normal, position, curvature, AO, and thickness into UV space. A BVH serves the occlusion rays, and a spatial grid of sharp edges gives curvature bands.
- `generators.rs` and `noise.rs`: Edge Wear, Dirt, Orientation, Gradient, Noise, and Thickness masks evaluated from mesh-map samples, with seeded object-space noise so breakup never seams, and the ID generator, which selects the set's ID map by colour.
- `raster.rs`: shared UV triangle rasterisation.
- `compositing.rs`, `export.rs`, and `filters.rs`: the layer compositor (including anchor masks and flattening), export presets and preflight, and layer filters.
- `resources.rs` and `color.rs`: bitmap resources with colour-space tags and provenance, and UV, triplanar, planar, cylindrical, spherical, and decal projections with physical tile sizes.
- `library.rs`: material resources, smart materials, smart masks, and the searchable asset index.
- `templates.rs`: project templates (texture-set settings and starter layers) that instantiate a project with one set per material slot.
- `evolution.rs`: evolution controls (for example clean to worn) on a layer that move recorded property ranges of the layers inside it as one batched edit.
- `instances.rs`: material instances. A fill layer's `material_link` names a project material (its master) and the fields it overrides; the layer keeps the master's values resolved into its own fill, so compositing never needs the project. Commands that change a master are batched with the settings of every instance they re-resolve.
- `painting.rs` and `udim.rs`: symmetry, stabilization, lines, seam detection, spline paths, and UDIM tile shifting.
- `camera_paint.rs`: camera projection painting. A paint or mask stroke with a `camera` holds screen-space points; at bake time each texel's mesh-map position is projected into that view, tested against a depth buffer of the mesh and the surface's facing, and takes the brush coverage there.

`substancecraft-api` adds file I/O (projects, textures, SVG and text resources, smart-material files), FBX and glTF import, OpenPBR MaterialX and glTF export, headless export jobs, and the versioned automation API (`automation.rs`: a project summary and typed edits applied as `EditCommand`s). `substancecraft-cli` runs export jobs and automation edits from the command line for pipelines; `substancecraft-mcp` exposes the same edits, project inspection, and exports as MCP tools confined to a root directory.

A layer is a paint layer or a fill layer (`LayerFill`). Its mask is built in order: a generator base (or white), painted mask strokes blended over it, then polygon-mask and ID-colour clipping. Base Color composites as straight-alpha RGBA; Roughness and Metallic composite as scalars from defaults of 0.5 and 0. Polygon masks index triangles of the *smoothed* mesh. Subdivision children of triangle `t` are `4t..4t+4`, so masks are remapped when the subdivision level changes.

Every user edit goes through `EditCommand`: strokes, layer-settings snapshots, layer insert/remove/move, texture-set and mesh edits, export presets, and project materials (with their instances, as one batch). The desktop keeps a bounded 128-step in-memory history that resets when a project is opened. Inspector edits commit as one step when the pointer is released. `substancecraft-api` exposes channel and mesh-map bakes alongside bounded JSON encode/decode and atomic `.sproject` file I/O. Imported FBX geometry is referenced by source path rather than embedded in the project; scale and smoothing are stored on the mesh asset.

The core generates UV-mapped, outward-wound triangle meshes for cube (3×2 atlas), sphere, cylinder (side plus cap discs), and plane presets, with non-overlapping UV charts. The desktop shell can:

- import binary FBX 7.4+ static meshes, smooth or subdivide them, and paint multi-channel strokes on the nearest picked triangle;
- build polygon masks by face, element, or UV island;
- add fill, Edge Wear, and Dirt layers;
- bake mesh maps in the background, rebaking automatically when the mesh revision changes;
- preview any bakeable channel or mesh map on the mesh, and export them as TGA.

Filament renders with the smoothed shading normals and per-vertex tangents from `smoothing::vertex_tangents`. The normal-map baker uses the same tangents, so baked tangent-space normals decode in exactly the frame they were encoded in. The viewport material takes an sRGB Base Color, a linear ORM map (R mesh-map AO, G roughness, B metallic), and a linear OpenGL-convention normal map, and has a flat preview mode for single channels and mesh maps. `TextureSet::bake_material` builds each layer mask once per bake, shares it across channels, and composites the channels in parallel. The desktop bakes previews on a background thread and keeps the last result on screen until the next arrives, so strokes update live. High-to-low baking casts rays from a cage along the interpolated low-mesh normals onto a BVH of the high mesh and evaluates every map at the hit. Missing pieces: FBX material-derived ID maps, opacity/emissive channels, image-based lighting environments, and global illumination.