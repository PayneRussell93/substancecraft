//! UDIM tiles: UV space beyond the unit square, split into numbered tiles
//! (`1001 + u + 10 v`). A tile is baked by shifting the mesh UVs and every
//! UV-space stroke so the tile lands on the unit square, then baking as
//! usual.

use std::collections::BTreeSet;

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::compositing::{BakeContext, BakeError};
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

fn shift_layer(layer: &mut MaterialLayer, offset: Vec2) {
    for stroke in &mut layer.paint_strokes {
        stroke.points.iter_mut().for_each(|point| *point -= offset);
        if let Some(path) = &mut stroke.path {
            path.controls.iter_mut().for_each(|point| *point -= offset);
        }
    }
    for stroke in &mut layer.mask_strokes {
        stroke.points.iter_mut().for_each(|point| *point -= offset);
    }
    for stroke in &mut layer.id_strokes {
        stroke.points.iter_mut().for_each(|point| *point -= offset);
    }
    for child in layer.children.iter_mut().flatten() {
        shift_layer(child, offset);
    }
}

impl TextureSet {
    /// A copy whose UV-space strokes are moved so `tile` covers the unit
    /// square. UV-projected bitmaps tile, so they need no change.
    pub fn shifted_to_tile(&self, tile: UdimTile) -> TextureSet {
        let mut set = self.clone();
        for layer in &mut set.layers {
            shift_layer(layer, tile.offset());
        }
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
}
