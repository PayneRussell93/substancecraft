//! UDIM tiles: UV space beyond the unit square, split into numbered tiles
//! (`1001 + u + 10 v`). A tile is baked by shifting the mesh UVs and every
//! UV-space stroke so the tile lands on the unit square, then baking as
//! usual.

use std::collections::BTreeSet;

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::compositing::{BakeContext, BakeError, BakedTexture, MaterialTextures};
use crate::export::{ExportPreset, ExportedTexture};
use crate::{MaterialLayer, MeshGeometry, TextureSet};

/// Replaced by the tile number in export file names.
pub const UDIM_TOKEN: &str = "$udim";
/// UDIM rows have ten tiles.
const TILES_PER_ROW: u32 = 10;
/// Highest tile accepted (100 rows).
const MAX_TILE: u32 = 1000 + TILES_PER_ROW * 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UdimTile(pub u32);

impl UdimTile {
    pub const FIRST: Self = Self(1001);

    /// The tile holding `uv`, or `None` outside the UDIM range (negative,
    /// or `u` beyond ten tiles).
    pub fn of_uv(uv: Vec2) -> Option<Self> {
        if !uv.is_finite() || uv.x < 0.0 || uv.y < 0.0 || uv.x >= TILES_PER_ROW as f32 {
            return None;
        }
        let tile = 1001 + uv.x.floor() as u32 + TILES_PER_ROW * uv.y.floor() as u32;
        (tile <= MAX_TILE).then_some(Self(tile))
    }

    /// True for tile numbers in the UDIM range (1001 to 2000).
    pub fn is_valid(self) -> bool {
        (1001..=MAX_TILE).contains(&self.0)
    }

    /// Lower-left corner of the tile in UV space.
    pub fn offset(self) -> Vec2 {
        let index = self.0.saturating_sub(1001);
        Vec2::new(
            (index % TILES_PER_ROW) as f32,
            (index / TILES_PER_ROW) as f32,
        )
    }
}

impl MeshGeometry {
    /// Tiles that triangle centroids fall in.
    pub fn udim_tiles(&self) -> BTreeSet<UdimTile> {
        self.triangles
            .iter()
            .filter_map(|triangle| {
                let mut centroid = Vec2::ZERO;
                for corner in triangle {
                    centroid += self.vertices.get(*corner as usize)?.uv / 3.0;
                }
                UdimTile::of_uv(centroid)
            })
            .collect()
    }

    /// A copy with every UV moved by `-offset`.
    pub fn shifted_uvs(&self, offset: Vec2) -> MeshGeometry {
        let mut mesh = self.clone();
        for vertex in &mut mesh.vertices {
            vertex.uv -= offset;
        }
        mesh
    }
}

/// Shifts a layer's UV-space strokes. Projected strokes are in screen space
/// and follow the surface, so they stay put.
fn shift_layer(layer: &mut MaterialLayer, offset: Vec2, tile: UdimTile) {
    // A layer limited to other tiles is hidden in this one.
    if !layer.uv_tiles.is_empty() {
        layer.visible &= layer.uv_tiles.contains(&tile);
        layer.uv_tiles.clear();
    }
    for stroke in &mut layer.paint_strokes {
        if stroke.camera.is_some() {
            continue;
        }
        stroke.points.iter_mut().for_each(|point| *point -= offset);
        if let Some(path) = &mut stroke.path {
            path.controls.iter_mut().for_each(|point| *point -= offset);
        }
    }
    for stroke in &mut layer.mask_strokes {
        if stroke.camera.is_some() {
            continue;
        }
        stroke.points.iter_mut().for_each(|point| *point -= offset);
    }
    for stroke in &mut layer.id_strokes {
        stroke.points.iter_mut().for_each(|point| *point -= offset);
    }
    for child in layer.children.iter_mut().flatten() {
        shift_layer(child, offset, tile);
    }
}

/// True when a UV-space stroke through `points` of `radius_pixels` (in
/// texels of a texture `size` texels across, less one) may touch the unit
/// square. Generous: it allows for size jitter, scatter, and stamp frames
/// that stretch a stamp up to [`crate::brush::MAX_STAMP_STRETCH`] times.
fn reaches_unit_square(
    points: &[Vec2],
    radius_pixels: f32,
    tip: crate::BrushTip,
    size: Vec2,
) -> bool {
    let stretch = crate::brush::MAX_STAMP_STRETCH.sqrt();
    let reach = radius_pixels.max(1.0)
        * (1.0 + tip.size_jitter.max(0.0))
        * (stretch + 2.0 * tip.scatter.max(0.0))
        / size
        + Vec2::splat(1e-3);
    points.iter().any(|point| {
        !point.is_finite() || (point.cmpge(-reach).all() && point.cmple(Vec2::ONE + reach).all())
    })
}

/// How the viewport shows several UDIM tiles at once: each tile's texture
/// in one cell of a single atlas texture, and the mesh's UVs moved into
/// their tile's cell, so one material samples every tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UdimAtlas {
    tiles: Vec<UdimTile>,
    columns: u32,
    rows: u32,
}

impl UdimAtlas {
    /// The atlas for `tiles`, or `None` when they are only tile 1001 (or
    /// none), which needs no atlas.
    pub fn new(tiles: &BTreeSet<UdimTile>) -> Option<Self> {
        if tiles.is_empty() || tiles.iter().eq([&UdimTile::FIRST]) {
            return None;
        }
        let count = tiles.len() as u32;
        let columns = (count as f32).sqrt().ceil() as u32;
        Some(Self {
            tiles: tiles.iter().copied().collect(),
            columns,
            rows: count.div_ceil(columns),
        })
    }

    /// Tiles in cell order: left to right, then up.
    pub fn tiles(&self) -> &[UdimTile] {
        &self.tiles
    }

    /// Columns and rows of cells.
    pub fn grid(&self) -> (u32, u32) {
        (self.columns, self.rows)
    }

    /// Cell of `tile`: column, and row counted up from the bottom (+V).
    fn cell(&self, tile: UdimTile) -> Option<(u32, u32)> {
        let index = self.tiles.binary_search(&tile).ok()? as u32;
        Some((index % self.columns, index / self.columns))
    }

    /// Where `uv`, a point of `tile`, lies in the atlas. Points of tiles
    /// outside the atlas give `None`.
    pub fn atlas_uv(&self, tile: UdimTile, uv: Vec2) -> Option<Vec2> {
        let (column, row) = self.cell(tile)?;
        let local = uv - tile.offset();
        Some(Vec2::new(
            (column as f32 + local.x) / self.columns as f32,
            (row as f32 + local.y) / self.rows as f32,
        ))
    }

    /// A copy of `mesh` with every UV moved into the atlas. Each vertex
    /// follows the tile of the first triangle using it (UDIM islands never
    /// straddle tiles); vertices of no tile keep their UV.
    pub fn mesh(&self, mesh: &MeshGeometry) -> MeshGeometry {
        let mut tiles = vec![None; mesh.vertices.len()];
        for triangle in &mesh.triangles {
            let Some(corners) = triangle
                .iter()
                .map(|corner| mesh.vertices.get(*corner as usize))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let centroid = corners.iter().map(|vertex| vertex.uv).sum::<Vec2>() / 3.0;
            let Some(tile) = UdimTile::of_uv(centroid) else {
                continue;
            };
            for corner in triangle {
                tiles[*corner as usize].get_or_insert(tile);
            }
        }
        let mut atlas = mesh.clone();
        for (vertex, tile) in atlas.vertices.iter_mut().zip(tiles) {
            if let Some(uv) = tile.and_then(|tile| self.atlas_uv(tile, vertex.uv)) {
                vertex.uv = uv;
            }
        }
        atlas
    }

    /// Longest tile edge that keeps the atlas within `max_edge` texels.
    pub fn tile_edge_limit(&self, max_edge: u32) -> u32 {
        (max_edge / self.columns.max(self.rows)).max(1)
    }

    /// Packs one texture per tile (in [`Self::tiles`] order, all one size)
    /// into the atlas. Unused cells are transparent black. `None` when the
    /// count or sizes do not match.
    pub fn assemble(&self, tiles: &[&BakedTexture]) -> Option<BakedTexture> {
        let first = tiles.first()?;
        let (width, height) = (first.width, first.height);
        if tiles.len() != self.tiles.len()
            || tiles
                .iter()
                .any(|tile| tile.width != width || tile.height != height)
        {
            return None;
        }
        let atlas_width = width.checked_mul(self.columns)?;
        let atlas_height = height.checked_mul(self.rows)?;
        let mut rgba8 = vec![0; atlas_width as usize * atlas_height as usize * 4];
        let row_bytes = width as usize * 4;
        for (index, tile) in tiles.iter().enumerate() {
            let (column, row) = (index as u32 % self.columns, index as u32 / self.columns);
            // Image rows run down, so the bottom cell row is the last.
            let top = (self.rows - 1 - row) * height;
            for y in 0..height {
                let source = y as usize * row_bytes;
                let target = (((top + y) * atlas_width + column * width) * 4) as usize;
                rgba8[target..target + row_bytes]
                    .copy_from_slice(tile.rgba8.get(source..source + row_bytes)?);
            }
        }
        Some(BakedTexture {
            width: atlas_width,
            height: atlas_height,
            rgba8,
        })
    }

    /// [`Self::assemble`] for every texture of a material.
    pub fn assemble_material(&self, tiles: &[MaterialTextures]) -> Option<MaterialTextures> {
        let pack = |pick: fn(&MaterialTextures) -> &BakedTexture| {
            self.assemble(&tiles.iter().map(pick).collect::<Vec<_>>())
        };
        Some(MaterialTextures {
            base_color: pack(|textures| &textures.base_color)?,
            orm: pack(|textures| &textures.orm)?,
            normal: pack(|textures| &textures.normal)?,
            emissive: pack(|textures| &textures.emissive)?,
        })
    }
}

impl TextureSet {
    /// A copy whose UV-space strokes are moved so `tile` covers the unit
    /// square, and whose layers limited to other tiles are hidden.
    /// UV-projected bitmaps tile, so they need no change.
    pub fn shifted_to_tile(&self, tile: UdimTile) -> TextureSet {
        let mut set = self.clone();
        for layer in &mut set.layers {
            shift_layer(layer, tile.offset(), tile);
        }
        // Quick masks move with the strokes they limit.
        for stroke in set
            .stencils
            .iter_mut()
            .flat_map(|stencil| &mut stencil.strokes)
            .filter(|stroke| stroke.camera.is_none())
        {
            stroke
                .points
                .iter_mut()
                .for_each(|point| *point -= tile.offset());
        }
        set
    }

    /// [`Self::shifted_to_tile`] without the UV-space strokes that cannot
    /// reach the tile: they write nothing there, so the tile bakes the same
    /// and two edits of other tiles leave this copy unchanged (which keys
    /// per-tile caches).
    pub fn for_tile(&self, tile: UdimTile) -> TextureSet {
        let mut set = self.shifted_to_tile(tile);
        let size = Vec2::new(
            set.resolution.width.saturating_sub(1).max(1) as f32,
            set.resolution.height.saturating_sub(1).max(1) as f32,
        );
        fn cull(layers: &mut [MaterialLayer], size: Vec2) {
            for layer in layers {
                layer.paint_strokes.retain(|stroke| {
                    stroke.camera.is_some()
                        || reaches_unit_square(
                            &stroke.points,
                            stroke.radius_pixels,
                            stroke.tip,
                            size,
                        )
                });
                layer.mask_strokes.retain(|stroke| {
                    stroke.camera.is_some()
                        || reaches_unit_square(
                            &stroke.points,
                            stroke.radius_pixels,
                            stroke.tip,
                            size,
                        )
                });
                layer.id_strokes.retain(|stroke| {
                    reaches_unit_square(&stroke.points, stroke.radius_pixels, stroke.tip, size)
                });
                if let Some(children) = layer.children.as_mut() {
                    cull(children, size);
                }
            }
        }
        cull(&mut set.layers, size);
        set
    }

    /// Bakes `preset` for one tile. `context.mesh` and `context.mesh_maps`
    /// must already be shifted to the tile (see
    /// [`MeshGeometry::shifted_uvs`]). File names replace [`UDIM_TOKEN`] with
    /// the tile number, or end in `.<tile>` when the template has no token.
    pub fn bake_export_tile(
        &self,
        preset: &ExportPreset,
        tile: UdimTile,
        context: BakeContext<'_>,
    ) -> Result<Vec<ExportedTexture>, BakeError> {
        let mut files = self.shifted_to_tile(tile).bake_export(preset, context)?;
        for file in &mut files {
            file.file_stem = if file.file_stem.contains(UDIM_TOKEN) {
                file.file_stem.replace(UDIM_TOKEN, &tile.0.to_string())
            } else {
                format!("{}.{}", file.file_stem, tile.0)
            };
        }
        Ok(files)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{
        ChannelSlot, ExportOutput, LayerId, MaterialChannel, MeshVertex, OutputLayout, PaintStroke,
        SourceComponent, TextureResolution, TextureSetId,
    };
    use glam::Vec3;

    /// Two unit quads side by side in UV tiles 1001 and 1002.
    fn two_tile_mesh() -> MeshGeometry {
        let mut mesh = MeshGeometry::default();
        for tile in 0..2 {
            let base = mesh.vertices.len() as u32;
            for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
                mesh.vertices.push(MeshVertex {
                    position: Vec3::new(x + tile as f32 * 1.1, y, 0.0),
                    uv: Vec2::new(x + tile as f32, y),
                });
            }
            mesh.triangles.push([base, base + 1, base + 2]);
            mesh.triangles.push([base, base + 2, base + 3]);
        }
        mesh
    }

    #[test]
    fn tc_0170_udim_tiles_number_shift_and_export_per_tile() {
        assert_eq!(UdimTile::of_uv(Vec2::new(0.5, 0.5)), Some(UdimTile(1001)));
        assert_eq!(UdimTile::of_uv(Vec2::new(3.2, 1.7)), Some(UdimTile(1014)));
        assert_eq!(UdimTile::of_uv(Vec2::new(-0.1, 0.5)), None);
        assert_eq!(UdimTile::of_uv(Vec2::new(10.5, 0.5)), None);
        assert_eq!(UdimTile(1014).offset(), Vec2::new(3.0, 1.0));

        let mesh = two_tile_mesh();
        assert_eq!(
            mesh.udim_tiles(),
            BTreeSet::from([UdimTile(1001), UdimTile(1002)])
        );
        let shifted = mesh.shifted_uvs(UdimTile(1002).offset());
        assert_eq!(shifted.vertices[4].uv, Vec2::ZERO);

        // A white stroke painted in the middle of tile 1002.
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Paint");
        let mut stroke = PaintStroke::new(
            vec![Vec2::new(1.4, 0.5), Vec2::new(1.6, 0.5)],
            2.0,
            [255, 255, 255, 255],
        );
        stroke.tip.hardness = 1.0;
        layer.paint_strokes.push(stroke);
        let set = TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Body".to_owned(),
            material_slot: "Body".to_owned(),
            resolution: TextureResolution {
                width: 16,
                height: 16,
            },
            channels: [MaterialChannel::BaseColor].into(),
            layers: vec![layer],
            height_strength: crate::DEFAULT_HEIGHT_STRENGTH,
            export: None,
        };
        let preset = ExportPreset {
            max_resolution: None,
            name: "Tiles".to_owned(),
            outputs: vec![
                ExportOutput::new(
                    "$textureSet_BaseColor.$udim",
                    OutputLayout::Rgb,
                    [
                        ChannelSlot::channel(MaterialChannel::BaseColor, SourceComponent::Red),
                        ChannelSlot::channel(MaterialChannel::BaseColor, SourceComponent::Green),
                        ChannelSlot::channel(MaterialChannel::BaseColor, SourceComponent::Blue),
                        ChannelSlot::ONE,
                    ],
                ),
                ExportOutput::new(
                    "$textureSet_Mask",
                    OutputLayout::Gray,
                    [ChannelSlot::ONE; 4],
                ),
            ],
        };
        preset.validate().unwrap();
        let centre =
            |files: &[ExportedTexture]| files[0].texture.sample_uv(Vec2::splat(0.5)).unwrap()[0];
        let first = set
            .bake_export_tile(&preset, UdimTile(1001), BakeContext::default())
            .unwrap();
        let second = set
            .bake_export_tile(&preset, UdimTile(1002), BakeContext::default())
            .unwrap();
        assert_eq!(first[0].file_stem, "Body_BaseColor.1001");
        assert_eq!(second[0].file_stem, "Body_BaseColor.1002");
        assert_eq!(second[1].file_stem, "Body_Mask.1002");
        assert_eq!(centre(&first), 128, "tile 1001 is unpainted grey");
        assert_eq!(centre(&second), 255, "tile 1002 holds the stroke");
    }

    #[test]
    fn tc_0373_udim_atlases_pack_tiles_and_move_uvs_into_their_cells() {
        let one = BTreeSet::from([UdimTile::FIRST]);
        assert_eq!(UdimAtlas::new(&one), None, "tile 1001 alone needs no atlas");
        assert_eq!(UdimAtlas::new(&BTreeSet::new()), None);

        let mesh = two_tile_mesh();
        let atlas = UdimAtlas::new(&mesh.udim_tiles()).unwrap();
        assert_eq!(atlas.grid(), (2, 1));
        assert_eq!(atlas.tiles(), [UdimTile(1001), UdimTile(1002)]);
        // Tile 1002's quad lands in the right half; the shared tile edge
        // u = 1 stays with each vertex's own tile.
        let moved = atlas.mesh(&mesh);
        assert_eq!(moved.vertices[1].uv, Vec2::new(0.5, 0.0));
        assert_eq!(moved.vertices[4].uv, Vec2::new(0.5, 0.0));
        assert_eq!(moved.vertices[6].uv, Vec2::new(1.0, 1.0));
        assert_eq!(atlas.atlas_uv(UdimTile(1003), Vec2::new(2.5, 0.5)), None);
        // Five tiles make a 3 x 2 grid; tile 1011 (u 0, v 1) is the fourth
        // cell, on the upper row.
        let five = UdimAtlas::new(&BTreeSet::from(
            [1001, 1002, 1003, 1011, 1012].map(UdimTile),
        ))
        .unwrap();
        assert_eq!(five.grid(), (3, 2));
        assert_eq!(
            five.atlas_uv(UdimTile(1011), Vec2::new(0.5, 1.5)),
            Some(Vec2::new(0.5 / 3.0, 0.75))
        );
        assert_eq!(five.tile_edge_limit(4096), 1365);

        // Packing puts each tile in its cell, upper rows first in memory.
        let solid = |value: u8| BakedTexture {
            width: 2,
            height: 2,
            rgba8: vec![value; 16],
        };
        let tiles = [solid(1), solid(2), solid(3), solid(4), solid(5)];
        let packed = five.assemble(&tiles.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!((packed.width, packed.height), (6, 4));
        assert_eq!(packed.pixel(0, 3).unwrap()[0], 1, "1001 bottom left");
        assert_eq!(packed.pixel(5, 2).unwrap()[0], 3);
        assert_eq!(packed.pixel(0, 0).unwrap()[0], 4, "1011 top left");
        assert_eq!(packed.pixel(5, 0).unwrap(), [0; 4], "empty cell");
        assert!(
            five.assemble(&tiles[..4].iter().collect::<Vec<_>>())
                .is_none()
        );

        // A tile's copy keeps only strokes that can reach it, so painting
        // another tile leaves it unchanged.
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Paint");
        let dab = |u: f32| PaintStroke::new(vec![Vec2::new(u, 0.5)], 2.0, [255; 4]);
        layer.paint_strokes = vec![dab(0.5), dab(1.5), dab(1.01)];
        let set = TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Body".to_owned(),
            material_slot: "Body".to_owned(),
            resolution: TextureResolution {
                width: 16,
                height: 16,
            },
            channels: [MaterialChannel::BaseColor].into(),
            layers: vec![layer],
            height_strength: crate::DEFAULT_HEIGHT_STRENGTH,
            export: None,
        };
        let first = set.for_tile(UdimTile(1001));
        let second = set.for_tile(UdimTile(1002));
        assert_eq!(
            first.layers[0].paint_strokes.len(),
            2,
            "the edge dab reaches back"
        );
        assert_eq!(second.layers[0].paint_strokes.len(), 2);
        let mut more = set.clone();
        more.layers[0].paint_strokes.push(dab(1.7));
        assert_eq!(more.for_tile(UdimTile(1001)), first);
        assert_ne!(more.for_tile(UdimTile(1002)), second);
    }

    #[test]
    fn tc_0297_uv_tile_masks_limit_layers_to_their_udim_tiles() {
        use std::collections::BTreeSet;

        let fill = |id: u64, color: [u8; 4]| {
            crate::MaterialLayer::new_fill(
                LayerId(id),
                "Fill",
                crate::LayerFill {
                    color,
                    ..crate::LayerFill::default()
                },
            )
        };
        let base = fill(1, [0, 0, 255, 255]);
        // A red fill only in tile 1002, inside a group only in 1001 and
        // 1002.
        let mut red = fill(2, [255, 0, 0, 255]);
        red.uv_tiles = BTreeSet::from([UdimTile(1002)]);
        let mut group = crate::MaterialLayer::new_group(LayerId(3), "Group");
        group.uv_tiles = BTreeSet::from([UdimTile(1001), UdimTile(1002)]);
        group.children = Some(vec![red]);
        let set = TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Body".to_owned(),
            material_slot: "Body".to_owned(),
            resolution: TextureResolution {
                width: 4,
                height: 4,
            },
            channels: BTreeSet::from([MaterialChannel::BaseColor]),
            layers: vec![base, group],
            height_strength: crate::DEFAULT_HEIGHT_STRENGTH,
            export: None,
        };
        let color = |set: &TextureSet| set.bake_base_color().unwrap().pixel(1, 1).unwrap();
        // The plain bake is tile 1001: the red layer stays out.
        assert_eq!(color(&set), [0, 0, 255, 255]);
        assert!(!set.layers[1].children.as_ref().unwrap()[0].shown());
        // Tile 1002 shows it; tile 1003 hides the whole group.
        assert_eq!(
            color(&set.shifted_to_tile(UdimTile(1002))),
            [255, 0, 0, 255]
        );
        let far = set.shifted_to_tile(UdimTile(1003));
        assert!(!far.layers[1].visible);
        assert_eq!(color(&far), [0, 0, 255, 255]);
        // Tile masks are layer settings; tiles outside the range are invalid.
        let mut project = crate::MaterialProject {
            texture_sets: vec![set],
            ..crate::MaterialProject::empty()
        };
        project.validate().unwrap();
        assert_eq!(
            project.texture_sets[0].layers[1].settings().uv_tiles.len(),
            2
        );
        project.texture_sets[0].layers[0].uv_tiles = BTreeSet::from([UdimTile(999)]);
        assert_eq!(
            project.validate(),
            Err(crate::ProjectError::InvalidLayerSettings(LayerId(1)))
        );
        assert!(UdimTile(2000).is_valid() && !UdimTile(2001).is_valid());
    }
}
