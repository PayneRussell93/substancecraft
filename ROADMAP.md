# Roadmap

**Status:** project-model foundation complete; no end-user painting workflow is implemented.

## Milestones

1. **Project foundation:** Rust workspace, versioned material project model, model validation. Complete.
2. **Mesh inspection:** starter presets (cube, sphere, cylinder, plane) are available with a shaded preview; import GLB/glTF, render an existing PBR material, orbit/pan/zoom, and inspect channels remain open.
3. **First paint loop:** create a texture set and paint base color in UV space; see the result on the mesh.
4. **History and export:** undo/redo a stroke and export a PNG texture.
5. **Multi-channel materials:** roughness, metallic, normal, height, channel-aware compositing, masks.
6. **3D painting and baking:** mesh picking, continuous projection, initial mesh maps and generators.
7. **Automation foundation:** stable Rust API facade and MCP server for mesh preset discovery and inspection. In progress.
8. **Full product capability map:** track all 50 capability areas and acceptance targets in [the feature map](docs/feature-map.md).

UDIM, high-to-low baking, smart materials, and plugin support remain later targets. Each milestone must preserve a usable, testable workspace; no Adobe project-format compatibility is planned.

The feature map is intentionally comprehensive, but this repository does not yet implement most Painter-class capabilities. Continue milestone-by-milestone and do not label planned areas complete without behavior tests.