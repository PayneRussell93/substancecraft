//! Incremental preview rebakes (PERF-04): while painting, the selected set's
//! preview is patched over the texels strokes touched instead of being
//! rebaked whole.
//!
//! [`PreviewDirty`] records which texels changed since the revision the
//! shown preview was baked at. It is trusted only while every layer's
//! non-stroke settings still match the snapshot taken then, so an edit that
//! forgot to invalidate the preview can never leave stale texels behind.

use substancecraft_core::{
    DirtyTexels, LayerId, LayerSettings, Stencil, TexelRect, TextureResolution, TextureSet,
    TextureSetId,
};

/// Above this share of dirty tiles a full rebake is cheaper than patching.
const MAX_DIRTY_FRACTION: f32 = 0.5;

/// A texture set apart from its strokes. While it is unchanged, strokes are
/// the only edits, and those are tracked texel by texel.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SetShape {
    resolution: TextureResolution,
    height_strength: f32,
    /// Clone and smudge strokes widen it as they are painted.
    halo: u32,
    stencils: Vec<Stencil>,
    layers: Vec<(LayerId, bool, LayerSettings)>,
}

impl SetShape {
    pub(super) fn of(set: &TextureSet) -> Self {
        Self {
            resolution: set.resolution,
            height_strength: set.height_strength,
            halo: set.region_halo(),
            stencils: set.stencils.clone(),
            layers: set
                .walk_layers()
                .into_iter()
                .map(|layer| (layer.id, layer.is_group(), layer.settings()))
                .collect(),
        }
    }
}

/// Texels of one texture set that strokes changed since `base_revision`.
pub(super) struct PreviewDirty {
    set_id: TextureSetId,
    base_revision: u64,
    shape: SetShape,
    /// How far beyond its own texels an edit changes the bake (blur,
    /// sharpen, clone, and smudge spread it); fixed while the shape is.
    halo: u32,

    texels: DirtyTexels,
}

impl PreviewDirty {
    /// Nothing changed yet since `set` was at `revision`.
    pub(super) fn new(set: &TextureSet, revision: u64) -> Self {
        Self {
            set_id: set.id,
            base_revision: revision,
            shape: SetShape::of(set),
            halo: set.region_halo(),
            texels: DirtyTexels::new(set.resolution.width, set.resolution.height),
        }
    }

    pub(super) fn set_id(&self) -> TextureSetId {
        self.set_id
    }

    /// Records an edit of the texels in `bounds`.
    pub(super) fn mark(&mut self, bounds: TexelRect) {
        let (width, height) = self.texels.resolution();
        self.texels.mark(bounds.grow(self.halo, width, height));
    }

    /// Regions to rebake to bring a preview baked at `base_revision` up to
    /// date with `set`, or `None` when it needs a full rebake. An empty list
    /// means the preview is already current.
    pub(super) fn regions(&self, set: &TextureSet, base_revision: u64) -> Option<Vec<TexelRect>> {
        let resolution = (set.resolution.width, set.resolution.height);
        (set.id == self.set_id
            && base_revision == self.base_revision
            && self.texels.resolution() == resolution
            && self.texels.dirty_fraction() <= MAX_DIRTY_FRACTION
            && SetShape::of(set) == self.shape)
            .then(|| self.texels.rects())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{LayerFilter, MaterialLayer};

    fn set() -> TextureSet {
        TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Set".to_owned(),
            material_slot: "Default".to_owned(),
            resolution: TextureResolution::new(256, 256).unwrap(),
            channels: Default::default(),
            layers: vec![MaterialLayer::new_paint(LayerId(1), "Paint")],
            height_strength: 1.0,
            export: None,
        }
    }

    #[test]
    fn regions_follow_marks_until_the_set_changes_shape() {
        let mut set = set();
        let mut dirty = PreviewDirty::new(&set, 7);
        assert_eq!(dirty.regions(&set, 7), Some(Vec::new()));
        assert_eq!(dirty.regions(&set, 6), None, "different base revision");
        let dab = TexelRect {
            x: 10,
            y: 10,
            width: 4,
            height: 4,
        };
        dirty.mark(dab);
        assert_eq!(
            dirty.regions(&set, 7),
            Some(vec![TexelRect {
                x: 0,
                y: 0,
                width: 64,
                height: 64
            }])
        );
        // Strokes are tracked; anything else needs a full rebake.
        set.layers[0]
            .paint_strokes
            .push(substancecraft_core::PaintStroke::new(
                vec![glam::Vec2::splat(0.05)],
                4.0,
                [255; 4],
            ));
        assert!(dirty.regions(&set, 7).is_some());
        set.layers[0].opacity = 0.5;
        assert_eq!(dirty.regions(&set, 7), None);
        set.layers[0].opacity = 1.0;
        // Blur spreads an edit, so marks grow by the set's halo.
        set.layers[0].filters = vec![LayerFilter::Blur { radius: 8.0 }];
        dirty = PreviewDirty::new(&set, 7);
        dirty.mark(dab);
        let regions = dirty.regions(&set, 7).unwrap();
        assert_eq!(regions.len(), 1);
        assert!(regions[0].width > 64 && regions[0].x == 0);
    }

    #[test]
    fn mostly_dirty_textures_rebake_whole() {
        let set = set();
        let mut dirty = PreviewDirty::new(&set, 1);
        dirty.mark(TexelRect {
            x: 0,
            y: 0,
            width: 256,
            height: 160,
        });
        assert_eq!(dirty.regions(&set, 1), None);
    }
}
