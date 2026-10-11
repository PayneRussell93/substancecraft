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
- Start with GLB/glTF, UV-space base-color painting, undo, and PNG export. Defer projection painting, UDIM, baking, and procedural materials until that loop is correct.

## Research use and clean-room boundary

PhotoCraft's public architecture notes inform the separation of core data, commands, history, and UI. LightCraft and VectorCraft provide additional examples of Rust workspace organization and clean-room product development. These repositories are architectural references only; SubstanceCraft does not depend on their crates or copy their source.

The product brief supplied for this project is the initial behavioral scope. Implementations and assets must be original or based on public standards and appropriately licensed research. Do not inspect or copy Adobe application binaries, source, shaders, icons, presets, or proprietary project files.

## Current implementation

`substancecraft-core` currently provides serializable project, mesh reference, texture set, layer, channel, and resolution types. `MaterialProject::validate` rejects invalid texture dimensions, duplicate IDs, dangling active texture-set references, and non-finite or out-of-range layer opacity. The model is not yet a persistence format or renderer.

The core also generates UV-mapped indexed triangle meshes for cube, sphere, cylinder, and plane presets. The desktop shell can select these presets and display a shaded isometric preview; this is a starter-mesh preview, not yet the full interactive PBR viewport.