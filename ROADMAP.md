# Roadmap

**Status:** project foundation, JSON project save/load, binary FBX static-mesh import, and projected viewport controls are in place; no painting workflow is implemented.

## Milestones

1. **Project foundation:** Rust workspace, versioned material project model, model validation. Complete.
2. **Mesh inspection:** starter presets, binary FBX 7.4+ static mesh import, UV preservation/fallback, selectable meshes, and projected-preview orbit/pan/zoom are implemented; ASCII FBX, animation, glTF, imported materials, and a true 3D viewport remain open.
3. **First paint loop:** create a texture set and paint base color in UV space; see the result on the mesh.
4. **History and export:** undo/redo a stroke and export a PNG texture.
5. **Multi-channel materials:** roughness, metallic, normal, height, channel-aware compositing, masks.
6. **3D painting and baking:** mesh picking, continuous projection, initial mesh maps and generators.
7. **Automation foundation:** Rust API facade includes preset inspection, FBX mesh import, and project serialization; MCP currently exposes read-only preset tools. In progress.
8. **Full product capability map:** track all 50 capability areas and acceptance targets in [the feature map](docs/feature-map.md).

UDIM, high-to-low baking, smart materials, and plugin support remain later targets. Each milestone must preserve a usable, testable workspace; no Adobe project-format compatibility is planned.

The feature map is intentionally comprehensive, but this repository does not yet implement most Painter-class capabilities. Continue milestone-by-milestone and do not label planned areas complete without behavior tests.