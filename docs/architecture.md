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

`substancecraft-core` provides serializable project, mesh reference, texture set, saved pose, layer, channel, mesh geometry, and resolution types. Layers own UV-space Base Color, grayscale-mask, and discrete ID-color strokes; a bounded CPU compositor bakes Base Color with grayscale and optional ID-color clipping, and emits a separate ID map. The desktop app records completed strokes in a bounded 128-command in-memory undo/redo history; it resets when a project is opened and is not serialized. `substancecraft-api` exposes both bake operations alongside bounded JSON encode/decode and atomic `.sproject` file I/O. Imported FBX geometry is referenced by source path rather than embedded in the project.

The core generates UV-mapped indexed triangle meshes for cube, sphere, cylinder, and plane presets. The desktop shell can import binary FBX 7.4+ static meshes, select imported meshes, paint on interpolated face UVs, preview baked Base Color or ID maps, and export either as TGA. The layer panel exposes grayscale masks and ID-color clipping. The projected preview also supports orbit, pan, zoom, a rotating directional key light, and ambient fill. FBX material-derived ID maps, geometric mesh-map baking, a real-time 3D PBR renderer, and global illumination are not implemented.