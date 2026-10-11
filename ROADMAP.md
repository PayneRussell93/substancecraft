# Roadmap

**Status:** Implemented so far:

- project foundation and JSON project save/load;
- binary FBX static-mesh import with auto-smoothed normals and crease-aware subdivision;
- Filament viewport;
- multi-channel paint, fill layers, and masks (painted, generator, polygon, ID);
- face, element, and UV-island polygon selection;
- geometry-derived mesh-map baking with automatic rebaking;
- Edge Wear, Dirt, Orientation, Gradient, and Noise generators;
- TGA export of channels and maps;
- undo/redo for strokes, layer settings, and layer structure.

## Milestones

1. **Project foundation:** Rust workspace, versioned material project model, model validation. Complete.
2. **Mesh inspection:** starter presets, binary FBX 7.4+ static mesh import, UV preservation/fallback, selectable meshes, and projected-preview orbit/pan/zoom are implemented; ASCII FBX, animation, glTF, imported materials, and a true 3D viewport remain open.
3. **First paint loop:** create a texture set, paint Base Color/ID in UV space, apply grayscale masks, and preview the result on the mesh. Implemented; broader channel painting remains open.
4. **History and export:** undo/redo for strokes, layer settings, and layer insert/remove/move; TGA export of Base Color, Roughness, Metallic, ID, and mesh maps. Implemented; PNG export and texture-set/mesh history remain open.
5. **Multi-channel materials:** Base Color, Roughness, Metallic, Height, and Normal compositing with fill layers, generator/painted/polygon/ID masks, height-to-normal detail, and a PBR Filament viewport (ORM + normal maps) are implemented; opacity and emissive channels remain open.
6. **3D painting and baking:** nearest-triangle picking, polygon selection, live background preview while painting, geometry-derived and high-to-low mesh maps (tangent normal, world normal, position, curvature, AO, thickness) with automatic rebake, and Edge Wear/Dirt/Orientation/Gradient/Noise generators are implemented; projection painting remains open.
7. **Automation foundation:** Rust API facade includes preset inspection, FBX mesh import, and project serialization; MCP currently exposes read-only preset tools. In progress.
8. **Full product capability map:** track all 50 capability areas and acceptance targets in [the feature map](docs/feature-map.md).

UDIM, saved smart-material presets, and plugin support remain later targets. Each milestone must preserve a usable, testable workspace; no Adobe project-format compatibility is planned.

The feature map is intentionally comprehensive, but this repository does not yet implement most Painter-class capabilities. Continue milestone-by-milestone and do not label planned areas complete without behavior tests.