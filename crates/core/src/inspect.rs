//! Inspection views over a texture set that never change project data:
//! which layers make up one texel (the surface inspector), a set showing
//! one layer alone (solo), and a layer's mask as a grayscale image.

use std::sync::Mutex;

use glam::Vec4;

use crate::compositing::{checked_bake_pixel_count, contributes, unit_to_byte};
use crate::region::TexelRect;
use crate::{
    BakeContext, BakeError, BakedTexture, LayerId, MaskGenerator, MaterialChannel, MaterialLayer,
    TextureSet,
};

/// One layer's part in a texel, as the surface inspector lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct TexelContribution {
    pub layer: LayerId,
    pub name: String,
    pub is_group: bool,
    /// Nesting depth: 0 for top-level layers.
    pub depth: usize,
    /// Coverage of the layer's own content (fill, strokes, or children)
    /// at the texel, before opacity and mask.
    pub coverage: f32,
    /// Final mask value at the texel (1 when the layer has no mask).
    pub mask: f32,
    pub opacity: f32,
    /// How much of the layer reaches the stack: coverage × opacity × mask.
    pub effective: f32,
    /// What drives the mask, e.g. "Edge Wear generator (curvature map)".
    pub drivers: Vec<String>,
}

/// Collects per-layer values at one texel while a channel composites.
pub(crate) struct Probe {
    index: usize,
    entries: Mutex<Vec<(LayerId, f32, f32)>>,
}

impl Probe {
    pub(crate) fn record(&self, layer: &MaterialLayer, local: &[Vec4], mask: Option<&[f32]>) {
        let coverage = local.get(self.index).map_or(0.0, |texel| texel.w);
        let mask = mask
            .and_then(|values| values.get(self.index).copied())
            .unwrap_or(1.0);
        if let Ok(mut entries) = self.entries.lock() {
            entries.push((layer.id, coverage, mask));
        }
    }
}

/// Colours of the mask debug overlay, chosen to stay distinguishable with
/// common colour-vision deficiencies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OverlayPalette {
    /// Green shown, red blocked, yellow partial.
    #[default]
    Standard,
    /// Blue and orange, for protanopia and deuteranopia.
    RedGreenSafe,
    /// Magenta and teal, for tritanopia.
    BlueYellowSafe,
}

impl OverlayPalette {
    pub const ALL: [Self; 3] = [Self::Standard, Self::RedGreenSafe, Self::BlueYellowSafe];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::RedGreenSafe => "Protan / deutan",
            Self::BlueYellowSafe => "Tritan",
        }
    }

    /// sRGB colours for shown, blocked, and partly shown texels.
    pub const fn colors(self) -> [[u8; 3]; 3] {
        match self {
            Self::Standard => [[60, 200, 80], [220, 60, 60], [240, 200, 40]],
            // Okabe–Ito blue and orange; partial is neutral grey.
            Self::RedGreenSafe => [[0, 114, 178], [230, 159, 0], [200, 200, 200]],
            Self::BlueYellowSafe => [[204, 121, 167], [0, 158, 115], [200, 200, 200]],
        }
    }
}

/// Mask values at or above this count as fully shown, at or below
/// `1 - MASK_SHOWN` as blocked.
pub const MASK_SHOWN: f32 = 0.95;

/// Tints `base` (an sRGB Base Color bake) by a grayscale mask bake: shown,
/// blocked, and partial texels each take their palette colour at `opacity`.
/// With `pattern`, blocked texels are also hatched and partial ones dotted,
/// so the overlay does not depend on hue alone.
pub fn mask_overlay(
    base: &BakedTexture,
    mask: &BakedTexture,
    palette: OverlayPalette,
    opacity: f32,
    pattern: bool,
) -> Option<BakedTexture> {
    if base.width != mask.width || base.height != mask.height {
        return None;
    }
    let opacity = if opacity.is_finite() {
        opacity.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let [shown, blocked, partial] = palette.colors();
    let width = base.width.max(1) as usize;
    let mut rgba8 = base.rgba8.clone();
    for (index, (texel, value)) in rgba8
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(mask.rgba8.as_chunks::<4>().0)
        .enumerate()
    {
        let value = f32::from(value[0]) / 255.0;
        let (x, y) = (index % width, index / width);
        let (tint, marked) = if value >= MASK_SHOWN {
            (shown, false)
        } else if value <= 1.0 - MASK_SHOWN {
            (blocked, (x + y) % 6 < 2)
        } else {
            (partial, x % 4 == 0 && y % 4 == 0)
        };
        for (channel, tint) in texel.iter_mut().zip(tint) {
            let mixed = f32::from(*channel) + (f32::from(tint) - f32::from(*channel)) * opacity;
            *channel = if pattern && marked {
                (mixed * 0.35).round() as u8
            } else {
                mixed.round() as u8
            };
        }
    }
    Some(BakedTexture {
        width: base.width,
        height: base.height,
        rgba8,
    })
}

/// What a layer's mask reads, for the inspector.
pub fn mask_drivers(layer: &MaterialLayer, set: &TextureSet) -> Vec<String> {
    let mut drivers = Vec::new();
    if let Some(generator) = &layer.mask_generator {
        let source = match generator {
            MaskGenerator::EdgeWear(_) => "curvature map",
            MaskGenerator::Dirt(_) => "AO and curvature maps",
            MaskGenerator::Orientation(_) => "world normal map",
            MaskGenerator::Gradient(_) => "position map",
            MaskGenerator::Noise(_) => "position map",
            MaskGenerator::Thickness(_) => "thickness map",
            MaskGenerator::IdColor(_) => "ID map",
            MaskGenerator::Pattern(_) => "position map",
        };
        drivers.push(format!("{} generator ({source})", generator.name()));
    }
    if layer.mask_bitmap.is_some() {
        drivers.push("Bitmap mask".to_owned());
    }
    if let Some(anchor) = layer.mask_anchor {
        let name = set.layer(anchor.layer).map_or_else(
            || "a missing layer".to_owned(),
            |target| target.name.clone(),
        );
        drivers.push(format!("Anchor to {name} ({:?})", anchor.channel));
    }
    if !layer.mask_strokes.is_empty() {
        drivers.push("Painted mask".to_owned());
    }
    if layer.polygon_mask.is_some() {
        drivers.push("Polygon selection".to_owned());
    }
    if layer.id_mask_color.is_some() {
        drivers.push("ID colour".to_owned());
    }
    if !layer.mask_filters.is_empty() {
        drivers.push(format!("{} mask filter(s)", layer.mask_filters.len()));
    }
    if layer.invert_final_mask {
        drivers.push("Inverted".to_owned());
    }
    drivers
}

/// The texel at `uv` on a `width` × `height` grid, matching
/// [`BakedTexture::sample_uv`].
fn texel_at(uv: glam::Vec2, width: u32, height: u32) -> Option<TexelRect> {
    if !uv.is_finite() || !(0.0..=1.0).contains(&uv.x) || !(0.0..=1.0).contains(&uv.y) {
        return None;
    }
    let x = (uv.x * width.saturating_sub(1) as f32).round() as usize;
    let y = ((1.0 - uv.y) * height.saturating_sub(1) as f32).round() as usize;
    Some(TexelRect {
        x: x as u32,
        y: y as u32,
        width: 1,
        height: 1,
    })
}

impl TextureSet {
    /// The visible layers and groups with content at `uv` in `channel`,
    /// top of the stack first (a group before its children).
    pub fn inspect_texel(
        &self,
        channel: MaterialChannel,
        uv: glam::Vec2,
        context: BakeContext<'_>,
    ) -> Result<Vec<TexelContribution>, BakeError> {
        let texel = texel_at(uv, self.resolution.width, self.resolution.height)
            .ok_or(BakeError::InvalidLayerSettings)?;
        checked_bake_pixel_count(self.resolution)?;
        // Only the texel and the neighbourhood its filters read.
        let region = self.composited_region(texel);
        let id_map = self.validate_for_region(region)?;
        let masks = self.layer_masks_in(&[channel], context, id_map.as_deref(), region)?;
        let probe = Probe {
            index: region.local_index(texel.x, texel.y),
            entries: Mutex::new(Vec::new()),
        };
        self.composite_region(channel, context, &masks, Some(&probe), region)?;
        let recorded = probe.entries.into_inner().unwrap_or_default();

        let mut order = Vec::new();
        fn top_down<'a>(
            layers: &'a [MaterialLayer],
            depth: usize,
            out: &mut Vec<(&'a MaterialLayer, usize)>,
        ) {
            for layer in layers.iter().rev() {
                out.push((layer, depth));
                if let Some(children) = &layer.children {
                    top_down(children, depth + 1, out);
                }
            }
        }
        top_down(&self.layers, 0, &mut order);
        Ok(order
            .into_iter()
            .filter(|(layer, _)| contributes(layer, &[channel]))
            .filter_map(|(layer, depth)| {
                let &(_, coverage, mask) = recorded.iter().find(|(id, _, _)| *id == layer.id)?;
                (coverage > 0.0).then(|| TexelContribution {
                    layer: layer.id,
                    name: layer.name.clone(),
                    is_group: layer.is_group(),
                    depth,
                    coverage,
                    mask,
                    opacity: layer.opacity,
                    effective: coverage * layer.opacity * mask,
                    drivers: mask_drivers(layer, self),
                })
            })
            .collect())
    }

    /// A copy in which only `layer_id` (with its contents and the groups
    /// containing it) is visible, or `None` when the set lacks the layer.
    pub fn soloed(&self, layer_id: LayerId) -> Option<TextureSet> {
        fn isolate(layers: &mut [MaterialLayer], target: LayerId) -> bool {
            let mut found = false;
            for layer in layers {
                // The target keeps its children as they are; an ancestor of
                // it stays visible around it.
                if layer.id == target
                    || layer
                        .children
                        .as_deref_mut()
                        .is_some_and(|children| isolate(children, target))
                {
                    layer.visible = true;
                    found = true;
                } else {
                    layer.visible = false;
                }
            }
            found
        }
        let mut solo = self.clone();
        isolate(&mut solo.layers, layer_id).then_some(solo)
    }

    /// A layer's final mask as opaque grayscale (white shows the layer),
    /// or with `generator_only`, just its generator.
    pub fn bake_layer_mask(
        &self,
        layer_id: LayerId,
        generator_only: bool,
        context: BakeContext<'_>,
    ) -> Result<BakedTexture, BakeError> {
        let layer = self
            .layer(layer_id)
            .ok_or(BakeError::InvalidLayerSettings)?;
        let alone = if generator_only {
            let generator = layer
                .mask_generator
                .ok_or(BakeError::InvalidLayerSettings)?;
            Some(MaterialLayer {
                mask_generator: Some(generator),
                ..MaterialLayer::new_paint(layer.id, "")
            })
        } else {
            None
        };
        let [rgba8] = self.bake_in_tiles(&self.bake_tiles()?, |region| {
            let id_map = self.validate_for_region(region)?;
            let values = match &alone {
                Some(alone) => self.layer_mask_in(alone, context, id_map.as_deref(), region)?,
                None => self.layer_mask_in(layer, context, id_map.as_deref(), region)?,
            };
            let mut rgba8 = Vec::new();
            rgba8
                .try_reserve_exact(region.len() * 4)
                .map_err(|_| BakeError::AllocationFailed)?;
            match values {
                Some(values) => {
                    for value in values {
                        let gray = unit_to_byte(value);
                        rgba8.extend([gray, gray, gray, 255]);
                    }
                }
                None => rgba8.resize(region.len() * 4, 255),
            }
            Ok([rgba8])
        })?;
        Ok(self.texture(rgba8))
    }

    /// Quick mask `id` as opaque grayscale (white is open to paint).
    pub fn bake_stencil(
        &self,
        id: crate::StencilId,
        context: BakeContext<'_>,
    ) -> Result<BakedTexture, BakeError> {
        let [rgba8] = self.bake_in_tiles(&self.bake_tiles()?, |region| {
            self.validate_for_region(region)?;
            let values = self.stencil_values(id, context, region)?;
            let mut rgba8 = Vec::new();
            rgba8
                .try_reserve_exact(region.len() * 4)
                .map_err(|_| BakeError::AllocationFailed)?;
            for value in values {
                let gray = unit_to_byte(value);
                rgba8.extend([gray, gray, gray, 255]);
            }
            Ok([rgba8])
        })?;
        Ok(self.texture(rgba8))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{
        EdgeWear, LayerFill, MaterialChannel, MeshMapSettings, PaintStroke, PrimitivePreset,
        TextureResolution, TextureSetId,
    };
    use glam::Vec2;

    fn set() -> TextureSet {
        let mut base = MaterialLayer::new_fill(
            LayerId(1),
            "Base",
            LayerFill {
                roughness: 0.8,
                ..LayerFill::default()
            },
        );
        base.channels = [MaterialChannel::Roughness].into();
        let mut gloss = MaterialLayer::new_paint(LayerId(2), "Gloss");
        gloss.channels = [MaterialChannel::Roughness].into();
        gloss.opacity = 0.5;
        let mut stroke = PaintStroke::new(vec![Vec2::new(0.25, 0.5)], 3.0, [255; 4]);
        stroke.roughness = 0.1;
        gloss.paint_strokes.push(stroke);
        let mut group = MaterialLayer::new_group(LayerId(3), "Group");
        group.children = Some(vec![gloss]);
        group.channels = [MaterialChannel::Roughness].into();
        TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Set".to_owned(),
            material_slot: "Default".to_owned(),
            resolution: TextureResolution::new(32, 32).unwrap(),
            channels: [MaterialChannel::Roughness].into(),
            layers: vec![base, group],
            height_strength: 1.0,
            export: None,
        }
    }

    #[test]
    fn tc_0241_inspector_lists_contributing_layers_top_down_with_masks() {
        let set = set();
        let on_stroke = set
            .inspect_texel(
                MaterialChannel::Roughness,
                Vec2::new(0.25, 0.5),
                BakeContext::default(),
            )
            .unwrap();
        let names: Vec<(&str, usize)> = on_stroke
            .iter()
            .map(|entry| (entry.name.as_str(), entry.depth))
            .collect();
        assert_eq!(names, [("Group", 0), ("Gloss", 1), ("Base", 0)]);
        let gloss = &on_stroke[1];
        assert!((gloss.effective - gloss.coverage * 0.5).abs() < 1e-6);
        assert_eq!(gloss.mask, 1.0);

        // Away from the stroke only the base fill contributes.
        let elsewhere = set
            .inspect_texel(
                MaterialChannel::Roughness,
                Vec2::new(0.9, 0.1),
                BakeContext::default(),
            )
            .unwrap();
        assert_eq!(elsewhere.len(), 1);
        assert_eq!(elsewhere[0].layer, LayerId(1));
        assert!(
            set.inspect_texel(
                MaterialChannel::Roughness,
                Vec2::new(2.0, 0.0),
                BakeContext::default()
            )
            .is_err()
        );
    }

    #[test]
    fn tc_0246_mask_overlay_tints_by_class_with_safe_palettes_and_patterns() {
        let base = BakedTexture {
            width: 3,
            height: 1,
            rgba8: [128, 128, 128, 255].repeat(3),
        };
        let mask = BakedTexture {
            width: 3,
            height: 1,
            rgba8: vec![255, 255, 255, 255, 0, 0, 0, 255, 128, 128, 128, 255],
        };
        let tinted = mask_overlay(&base, &mask, OverlayPalette::Standard, 1.0, false).unwrap();
        let [shown, blocked, partial] = OverlayPalette::Standard.colors();
        assert_eq!(tinted.rgba8[0..3], shown);
        assert_eq!(tinted.rgba8[4..7], blocked);
        assert_eq!(tinted.rgba8[8..11], partial);
        assert_eq!(
            mask_overlay(&base, &mask, OverlayPalette::Standard, 0.0, false).unwrap(),
            base
        );
        // Half opacity mixes halfway; alpha is kept.
        let half = mask_overlay(&base, &mask, OverlayPalette::RedGreenSafe, 0.5, false).unwrap();
        assert_eq!(half.rgba8[0], 64);
        assert_eq!(half.rgba8[3], 255);
        // The pattern darkens the blocked texel so it reads without hue.
        let hatched = mask_overlay(&base, &mask, OverlayPalette::Standard, 1.0, true).unwrap();
        assert!(hatched.rgba8[4] < tinted.rgba8[4]);
        assert_eq!(hatched.rgba8[0..3], tinted.rgba8[0..3]);
        // Every palette keeps its three classes apart.
        for palette in OverlayPalette::ALL {
            let [a, b, c] = palette.colors();
            assert!(a != b && b != c && a != c, "{palette:?}");
        }
        let small = BakedTexture {
            width: 1,
            height: 1,
            rgba8: vec![0; 4],
        };
        assert!(mask_overlay(&base, &small, OverlayPalette::Standard, 1.0, false).is_none());
    }

    #[test]
    fn tc_0242_solo_and_mask_views_never_touch_the_set() {
        let mut set = set();
        let original = set.clone();
        let solo = set.soloed(LayerId(2)).unwrap();
        assert!(!solo.layer(LayerId(1)).unwrap().visible);
        assert!(solo.layer(LayerId(2)).unwrap().visible);
        assert!(solo.layer(LayerId(3)).unwrap().visible);
        assert_eq!(set, original);
        assert!(set.soloed(LayerId(99)).is_none());
        // The solo bake shows default roughness where the base was.
        let baked = solo
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap();
        assert_eq!(
            baked.sample_uv(Vec2::new(0.9, 0.1)).unwrap()[0],
            unit_to_byte(0.5)
        );

        // An unmasked layer shows white; an edge-wear mask shows its
        // generator in grey levels.
        let open = set
            .bake_layer_mask(LayerId(1), false, BakeContext::default())
            .unwrap();
        assert!(open.rgba8.iter().all(|byte| *byte == 255));
        assert!(
            set.bake_layer_mask(LayerId(1), true, BakeContext::default())
                .is_err()
        );

        let cube = PrimitivePreset::Cube.build_mesh();
        let maps = crate::bake_mesh_maps(
            &cube,
            &MeshMapSettings {
                resolution: 32,
                ao_samples: 2,
                ..MeshMapSettings::default()
            },
            None,
        )
        .unwrap();
        set.layers[0].mask_generator = Some(MaskGenerator::EdgeWear(EdgeWear::DEFAULT));
        let context = BakeContext {
            mesh: Some(&cube),
            mesh_maps: Some(&maps),
            resources: None,
        };
        let mask = set.bake_layer_mask(LayerId(1), false, context).unwrap();
        let grays: std::collections::BTreeSet<u8> = mask.rgba8.chunks(4).map(|px| px[0]).collect();
        assert!(grays.len() > 2, "{grays:?}");
        assert_eq!(
            set.bake_layer_mask(LayerId(1), true, context).unwrap(),
            mask
        );
        let probe = set
            .inspect_texel(MaterialChannel::Roughness, Vec2::new(0.9, 0.1), context)
            .unwrap();
        assert_eq!(probe[0].drivers, ["Edge Wear generator (curvature map)"]);
    }
}
