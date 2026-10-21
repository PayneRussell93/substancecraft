//! Incremental preview rebakes (PERF-04): the selected set's preview is
//! patched over the texels and channels edits touched instead of being
//! rebaked whole.
//!
//! [`PreviewDirty`] records which texels strokes changed since the
//! revision the shown preview was baked at, and keeps a snapshot of the
//! set's layer tree and settings from then. Edits other than strokes are
//! not recorded at all: [`PreviewDirty::regions`] diffs the snapshot
//! against the set, so an edit that forgot to report itself can never
//! leave stale texels behind. A changed paint layer reaches only the
//! texels under its strokes; any other changed layer reaches every texel,
//! but only of the channels it composites into.

use std::collections::BTreeSet;

use substancecraft_core::{
    BlendMode, DirtyTexels, LayerId, LayerSettings, MaterialChannel, MaterialLayer, MaterialPlanes,
    Stencil, TexelRect, TextureResolution, TextureSet, TextureSetId,
};

/// Above this share of dirty tiles a full rebake is cheaper than patching.
const MAX_DIRTY_FRACTION: f32 = 0.5;

/// Channels an edit can have changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Channels {
    All,
    Only(BTreeSet<MaterialChannel>),
}

impl Channels {
    /// The material textures that show these channels.
    pub(super) fn planes(&self) -> MaterialPlanes {
        match self {
            Self::All => MaterialPlanes::ALL,
            Self::Only(channels) => MaterialPlanes::of_channels(channels.iter().copied()),
        }
    }

    /// True when a bake of `channel` alone can differ: Normal also shows
    /// the Height channel's slopes.
    pub(super) fn reach(&self, channel: MaterialChannel) -> bool {
        match self {
            Self::All => true,
            Self::Only(channels) => {
                channels.contains(&channel)
                    || (channel == MaterialChannel::Normal
                        && channels.contains(&MaterialChannel::Height))
            }
        }
    }
}

/// One block of texels to rebake, in the channels that may differ there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Patch {
    pub(super) rect: TexelRect,
    pub(super) channels: Channels,
}

/// One layer of a set apart from its strokes.
#[derive(Clone, Debug, PartialEq)]
struct LayerShape {
    id: LayerId,
    parent: Option<LayerId>,
    group: bool,
    settings: LayerSettings,
    /// For a pass-through group, the channels it composites into. Starting
    /// or stopping to composite one remixes (and refilters) the whole stack
    /// beneath it, whichever texels its children cover.
    passes: Option<BTreeSet<MaterialChannel>>,
}

impl LayerShape {
    /// Same layer, same place in the tree, and changes inside it stay
    /// where they happen.
    fn same_slot(&self, other: &Self) -> bool {
        self.id == other.id
            && self.parent == other.parent
            && self.group == other.group
            && self.passes == other.passes
    }
}

/// A texture set apart from its strokes. Strokes are tracked texel by
/// texel; everything here is diffed when the preview is brought up to date.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SetShape {
    resolution: TextureResolution,
    height_strength: f32,
    /// Clone and smudge strokes widen it as they are painted.
    halo: u32,
    stencils: Vec<Stencil>,
    layers: Vec<LayerShape>,
}

impl SetShape {
    pub(super) fn of(set: &TextureSet) -> Self {
        fn walk(layers: &[MaterialLayer], parent: Option<LayerId>, out: &mut Vec<LayerShape>) {
            for layer in layers {
                out.push(LayerShape {
                    id: layer.id,
                    parent,
                    group: layer.is_group(),
                    settings: layer.settings(),
                    passes: (layer.is_group() && layer.blend_mode == BlendMode::PassThrough)
                        .then(|| layer.composited_channels()),
                });
                if let Some(children) = &layer.children {
                    walk(children, Some(layer.id), out);
                }
            }
        }
        let mut layers = Vec::new();
        walk(&set.layers, None, &mut layers);
        Self {
            resolution: set.resolution,
            height_strength: set.height_strength,
            halo: set.region_halo(),
            stencils: set.stencils.clone(),
            layers,
        }
    }
}

/// Texels of one texture set that strokes changed since `base_revision`,
/// and the set's shape then. The selected set's preview keeps one, and so
/// does every cached bake of a texture set (PERF-02).
#[derive(Clone)]
pub(super) struct PreviewDirty {
    set_id: TextureSetId,
    base_revision: u64,
    shape: SetShape,
    texels: DirtyTexels,
}

impl PreviewDirty {
    /// Nothing changed yet since `set` was at `revision`.
    pub(super) fn new(set: &TextureSet, revision: u64) -> Self {
        Self {
            set_id: set.id,
            base_revision: revision,
            shape: SetShape::of(set),
            texels: DirtyTexels::new(set.resolution.width, set.resolution.height),
        }
    }

    pub(super) fn set_id(&self) -> TextureSetId {
        self.set_id
    }

    pub(super) fn base_revision(&self) -> u64 {
        self.base_revision
    }

    /// Records a stroke edit of the texels in `bounds`.
    pub(super) fn mark(&mut self, bounds: TexelRect) {
        let (width, height) = self.texels.resolution();
        self.texels
            .mark(bounds.grow(self.shape.halo, width, height));
    }

    /// Blocks to rebake to bring a preview baked at `base_revision` up to
    /// date with `set`, or `None` when it needs a full rebake. An empty list
    /// means the preview is already current.
    pub(super) fn regions(&self, set: &TextureSet, base_revision: u64) -> Option<Vec<Patch>> {
        let (width, height) = self.texels.resolution();
        if set.id != self.set_id
            || base_revision != self.base_revision
            || (set.resolution.width, set.resolution.height) != (width, height)
        {
            return None;
        }
        let shape = SetShape::of(set);
        if shape == self.shape {
            return self.stroke_patches();
        }
        // Stencils and the halo shape every layer's region math.
        if shape.resolution != self.shape.resolution
            || shape.halo != self.shape.halo
            || shape.stencils != self.shape.stencils
        {
            return None;
        }
        // Layers may arrive or leave, but those that stay must keep their
        // order and parents: a moved layer changes what lies beneath others.
        let kept = |layers: &[LayerShape], others: &[LayerShape]| -> Vec<LayerShape> {
            layers
                .iter()
                .filter(|layer| others.iter().any(|other| other.id == layer.id))
                .cloned()
                .collect()
        };
        let (new_kept, old_kept) = (
            kept(&shape.layers, &self.shape.layers),
            kept(&self.shape.layers, &shape.layers),
        );
        if !new_kept
            .iter()
            .zip(&old_kept)
            .all(|(new, old)| new.same_slot(old))
        {
            return None;
        }
        let full = TexelRect::full(width, height);
        // Every texel of every channel: start from the strokes' tiles.
        let mut everything = self.texels.clone();
        let mut limited: Vec<(BTreeSet<MaterialChannel>, DirtyTexels)> = Vec::new();
        let mut mark = |channels: Option<BTreeSet<MaterialChannel>>, rects: &[TexelRect]| {
            let texels = match channels {
                None => &mut everything,
                Some(channels) => {
                    let index = match limited.iter().position(|(seen, _)| *seen == channels) {
                        Some(index) => index,
                        None => {
                            limited.push((channels, DirtyTexels::new(width, height)));
                            limited.len() - 1
                        }
                    };
                    &mut limited[index].1
                }
            };
            for rect in rects {
                texels.mark(rect.grow(shape.halo, width, height));
            }
        };
        if shape.height_strength != self.shape.height_strength {
            mark(Some(BTreeSet::from([MaterialChannel::Normal])), &[full]);
        }
        // An anchor mask reads one channel of the stack beneath it into
        // every channel of its layer, so channels cannot be told apart.
        let anchored = shape
            .layers
            .iter()
            .chain(&self.shape.layers)
            .any(|layer| layer.settings.mask_anchor.is_some());
        // Each changed layer, with its settings before (`None` when it
        // arrived) and after (`None` when it left).
        let changes = new_kept
            .iter()
            .zip(&old_kept)
            .map(|(new, old)| (Some(old), Some(new)))
            .chain(
                shape
                    .layers
                    .iter()
                    .filter(|layer| !old_kept.iter().any(|old| old.id == layer.id))
                    .map(|new| (None, Some(new))),
            )
            .chain(
                self.shape
                    .layers
                    .iter()
                    .filter(|layer| !new_kept.iter().any(|new| new.id == layer.id))
                    .map(|old| (Some(old), None)),
            );
        for (old, new) in changes {
            let channels = match (old, new) {
                (Some(old), Some(new)) => new.settings.changed_channels(&old.settings),
                (Some(only), None) | (None, Some(only)) => only.settings.channels.clone(),
                (None, None) => continue,
            };
            if channels.is_empty() {
                continue;
            }
            // ID layers clip other layers wherever they paint.
            let channels =
                (!anchored && !channels.contains(&MaterialChannel::Id)).then_some(channels);
            // A layer that left took its strokes along, so where they were
            // is unknown.
            let footprint = match (old, new) {
                (old, Some(new)) if old.is_none_or(|old| old.settings.fill.is_none()) => {
                    set.layer_footprint(new.id)
                }
                _ => None,
            };
            mark(channels, footprint.as_deref().unwrap_or(&[full]));
        }
        if everything.dirty_fraction() > MAX_DIRTY_FRACTION {
            return None;
        }
        let mut patches = rects_as(&everything, &Channels::All);
        for (channels, texels) in &limited {
            patches.extend(rects_as(texels, &Channels::Only(channels.clone())));
        }
        Some(patches)
    }

    /// [`Self::regions`] when strokes are the only edits since; `None` (a
    /// full rebake) after any other.
    pub(super) fn stroke_regions(
        &self,
        set: &TextureSet,
        base_revision: u64,
    ) -> Option<Vec<Patch>> {
        (set.id == self.set_id
            && base_revision == self.base_revision
            && SetShape::of(set) == self.shape)
            .then(|| self.stroke_patches())
            .flatten()
    }

    fn stroke_patches(&self) -> Option<Vec<Patch>> {
        (self.texels.dirty_fraction() <= MAX_DIRTY_FRACTION)
            .then(|| rects_as(&self.texels, &Channels::All))
    }
}

fn rects_as(texels: &DirtyTexels, channels: &Channels) -> Vec<Patch> {
    texels
        .rects()
        .into_iter()
        .map(|rect| Patch {
            rect,
            channels: channels.clone(),
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{LayerFill, LayerFilter, PaintStroke};

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

    fn rects(patches: &[Patch]) -> Vec<TexelRect> {
        patches.iter().map(|patch| patch.rect).collect()
    }

    const DAB: TexelRect = TexelRect {
        x: 10,
        y: 10,
        width: 4,
        height: 4,
    };

    #[test]
    fn regions_follow_marks_until_the_set_changes_shape() {
        let mut set = set();
        let mut dirty = PreviewDirty::new(&set, 7);
        assert_eq!(dirty.regions(&set, 7), Some(Vec::new()));
        assert_eq!(dirty.regions(&set, 6), None, "different base revision");
        dirty.mark(DAB);
        let tile = TexelRect {
            x: 0,
            y: 0,
            width: 64,
            height: 64,
        };
        assert_eq!(
            dirty.regions(&set, 7),
            Some(vec![Patch {
                rect: tile,
                channels: Channels::All
            }])
        );
        // Strokes are tracked by their marks.
        set.layers[0].paint_strokes.push(PaintStroke::new(
            vec![glam::Vec2::splat(0.05)],
            4.0,
            [255; 4],
        ));
        assert_eq!(rects(&dirty.regions(&set, 7).unwrap()), vec![tile]);
        // Renaming changes nothing the bake shows.
        set.layers[0].name = "Renamed".to_owned();
        assert_eq!(rects(&dirty.regions(&set, 7).unwrap()), vec![tile]);
        // A paint layer's settings reach only the texels under its strokes
        // (near the bottom-left: rows run down from V = 1).
        set.layers[0].opacity = 0.5;
        let regions = dirty.regions(&set, 7).unwrap();
        let stroke_tile = TexelRect { y: 192, ..tile };
        assert_eq!(rects(&regions), vec![tile, stroke_tile]);
        assert!(regions.iter().all(|patch| patch.channels == Channels::All));
        set.layers[0].opacity = 1.0;
        // Blur spreads an edit, so marks grow by the set's halo.
        set.layers[0].filters = vec![LayerFilter::Blur { radius: 8.0 }];
        assert_eq!(dirty.regions(&set, 7), None, "the halo changed");
        dirty = PreviewDirty::new(&set, 7);
        dirty.mark(DAB);
        let regions = dirty.regions(&set, 7).unwrap();
        assert_eq!(regions.len(), 1);
        assert!(regions[0].rect.width > 64 && regions[0].rect.x == 0);
    }

    #[test]
    fn fill_edits_rebake_only_their_channels() {
        let mut set = set();
        let mut fill = MaterialLayer::new_fill(LayerId(2), "Fill", LayerFill::default());
        fill.channels = BTreeSet::from([MaterialChannel::Roughness, MaterialChannel::Height]);
        set.layers.push(fill);
        let dirty = PreviewDirty::new(&set, 1);
        set.layers[1].fill.as_mut().unwrap().roughness = 0.2;
        let patches = dirty.regions(&set, 1).unwrap();
        assert_eq!(rects(&patches), vec![TexelRect::full(256, 256)]);
        // Only the value that changed, though the layer enables Height too.
        let channels = &patches[0].channels;
        assert_eq!(
            channels.planes(),
            MaterialPlanes {
                orm: true,
                ..MaterialPlanes::default()
            }
        );
        assert!(!channels.reach(MaterialChannel::Normal));
        set.layers[1].fill.as_mut().unwrap().height = 0.9;
        let patches = dirty.regions(&set, 1).unwrap();
        assert!(patches[0].channels.reach(MaterialChannel::Normal), "slopes");
        assert!(!patches[0].channels.reach(MaterialChannel::BaseColor));
        // A setting every channel shares reaches all the layer enables.
        set.layers[1].opacity = 0.5;
        let patches = dirty.regions(&set, 1).unwrap();
        assert_eq!(
            patches[0].channels,
            Channels::Only(BTreeSet::from([
                MaterialChannel::Roughness,
                MaterialChannel::Height
            ]))
        );
        set.layers[1].opacity = 1.0;
        set.layers[1].fill.as_mut().unwrap().height = LayerFill::default().height;
        // Enabling a channel reaches it too.
        set.layers[1].channels.insert(MaterialChannel::BaseColor);
        let patches = dirty.regions(&set, 1).unwrap();
        assert!(patches[0].channels.planes().base_color);
        // Height force changes the normals alone.
        set.layers[1].channels.remove(&MaterialChannel::BaseColor);
        set.layers[1].fill.as_mut().unwrap().roughness = LayerFill::default().roughness;
        set.height_strength = 2.0;
        let patches = dirty.regions(&set, 1).unwrap();
        assert_eq!(
            patches[0].channels,
            Channels::Only(BTreeSet::from([MaterialChannel::Normal]))
        );
    }

    #[test]
    fn structure_and_id_edits_fall_back_to_safe_rebakes() {
        let mut set = set();
        let mut ids = MaterialLayer::new_fill(LayerId(2), "Ids", LayerFill::default());
        ids.channels = BTreeSet::from([MaterialChannel::Id]);
        set.layers.push(ids);
        let dirty = PreviewDirty::new(&set, 1);
        // ID clips other layers, so the whole texture changes everywhere.
        set.layers[1].opacity = 0.5;
        assert_eq!(dirty.regions(&set, 1), None);
        set.layers[1].opacity = 1.0;
        set.layers.swap(0, 1);
        assert_eq!(dirty.regions(&set, 1), None, "layers moved");
        set.layers.swap(0, 1);
        set.layers.pop();
        assert_eq!(dirty.regions(&set, 1), None, "a layer left");
    }

    #[test]
    fn layers_that_arrive_or_leave_rebake_what_they_covered() {
        let mut set = set();
        let dirty = PreviewDirty::new(&set, 1);
        // An empty paint layer changes nothing.
        set.layers
            .push(MaterialLayer::new_paint(LayerId(2), "Empty"));
        assert_eq!(dirty.regions(&set, 1), Some(Vec::new()));
        // One with a stroke changes the texels under it.
        set.layers[1].paint_strokes.push(PaintStroke::new(
            vec![glam::Vec2::new(0.5, 0.5)],
            4.0,
            [255; 4],
        ));
        let patches = dirty.regions(&set, 1).unwrap();
        assert_eq!(patches.len(), 1);
        assert!(patches[0].rect.contains(127, 127) && patches[0].rect.len() <= 128 * 128);
        // A fill reaches every texel, but only of its channels.
        let mut fill = MaterialLayer::new_fill(LayerId(3), "Fill", LayerFill::default());
        fill.channels = BTreeSet::from([MaterialChannel::Metallic]);
        set.layers.push(fill);
        let patches = dirty.regions(&set, 1).unwrap();
        assert!(
            patches
                .iter()
                .any(|patch| patch.rect == TexelRect::full(256, 256)
                    && patch.channels
                        == Channels::Only(BTreeSet::from([MaterialChannel::Metallic])))
        );
        // A layer that left took its strokes' whereabouts along.
        let dirty = PreviewDirty::new(&set, 2);
        set.layers.remove(1);
        let patches = dirty.regions(&set, 2);
        assert_eq!(patches, None, "every channel of the whole texture");
    }

    #[test]
    fn pass_through_groups_that_start_compositing_rebake_whole() {
        let mut set = set();
        let mut group = MaterialLayer::new_group(LayerId(2), "Pass");
        group.blend_mode = BlendMode::PassThrough;
        group.filters = vec![LayerFilter::Invert];
        let mut child = MaterialLayer::new_paint(LayerId(3), "Child");
        child.channels = group.channels.clone();
        group.children = Some(vec![child]);
        set.layers.push(group);
        let mut dirty = PreviewDirty::new(&set, 1);
        // The first stroke inside makes the group invert the whole stack.
        set.layer_mut(LayerId(3))
            .unwrap()
            .paint_strokes
            .push(PaintStroke::new(
                vec![glam::Vec2::splat(0.5)],
                4.0,
                [255; 4],
            ));
        dirty.mark(DAB);
        assert_eq!(dirty.regions(&set, 1), None);
        // Later strokes stay where they are.
        let mut dirty = PreviewDirty::new(&set, 2);
        dirty.mark(DAB);
        assert!(dirty.regions(&set, 2).is_some());
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
