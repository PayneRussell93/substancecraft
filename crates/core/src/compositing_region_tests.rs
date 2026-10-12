//! PERF-04 determinism: a region rebake after any sequence of stroke edits
//! must equal a full bake byte for byte.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;
use std::sync::Arc;

use glam::Vec2;

use super::*;
use crate::{
    BrushShape, BrushTip, DEFAULT_HEIGHT_STRENGTH, DirtyTexels, Gradient, IdStroke, LayerFilter,
    MaskGenerator, MaskStroke, MeshMapSettings, NoisePattern, PolygonMask, PrimitivePreset, TextureProjection,
    TextureSetId, bake_mesh_maps, stroke_texel_bounds,
};

/// Deterministic generator so failures replay exactly.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1_u64 << 31) as f32
    }

    fn below(&mut self, count: usize) -> usize {
        self.next() as usize % count
    }

    fn uv(&mut self) -> Vec2 {
        Vec2::new(self.unit(), self.unit())
    }

    fn byte(&mut self) -> u8 {
        (self.next() % 256) as u8
    }
}

const PAINT: LayerId = LayerId(2);
const MULTIPLY: LayerId = LayerId(4);
const MASKED: LayerId = LayerId(5);
const IDS: LayerId = LayerId(8);

fn random_tip(rng: &mut Lcg) -> BrushTip {
    let shape = BrushShape::BUILT_IN[rng.below(3)];
    BrushTip {
        shape,
        hardness: rng.unit(),
        spacing: 0.05 + rng.unit() * 0.5,
        size_jitter: if shape == BrushShape::Round {
            0.0
        } else {
            rng.unit()
        },
        angle_jitter: rng.unit(),
        scatter: rng.unit() * 0.5,
        follow_path: rng.below(2) == 0,
        seed: rng.next(),
        ..BrushTip::ROUND
    }
}

fn checker_image(size: u32) -> TextureImage {
    let rgba8 = (0..size * size)
        .flat_map(|index| {
            let (x, y) = (index % size, index / size);
            let on = (x / 4 + y / 4) % 2 == 0;
            [if on { 230 } else { 40 }, (x * 7) as u8, (y * 5) as u8, 255]
        })
        .collect();
    TextureImage::new(size, size, rgba8).unwrap()
}

fn all_channels() -> BTreeSet<MaterialChannel> {
    BTreeSet::from([
        MaterialChannel::BaseColor,
        MaterialChannel::Roughness,
        MaterialChannel::Metallic,
        MaterialChannel::Height,
        MaterialChannel::Emissive,
        MaterialChannel::Opacity,
        MaterialChannel::Normal,
    ])
}

/// A stack exercising every per-texel stage: generator, bitmap and
/// triplanar fills, anchors, polygon and ID clipping, groups,
/// pass-through, blend modes, per-texel filters, and height-to-normal.
fn stack() -> TextureSet {
    let all = all_channels();
    let mut base = MaterialLayer::new_fill(
        LayerId(1),
        "Base",
        LayerFill {
            color: [90, 110, 140, 255],
            roughness: 0.7,
            ..LayerFill::default()
        },
    );
    let fill = base.fill.as_mut().unwrap();
    fill.textures.base_color = Some(ResourceId(1));
    fill.textures.height = Some(ResourceId(1));
    fill.projection = TextureProjection {
        mode: ProjectionMode::Triplanar,
        ..TextureProjection::default()
    };
    base.channels = all.clone();
    base.mask_generator = Some(MaskGenerator::Noise(NoisePattern::DEFAULT));

    let mut paint = MaterialLayer::new_paint(PAINT, "Paint");
    paint.channels = all.clone();

    let mut multiply = MaterialLayer::new_paint(MULTIPLY, "Multiply");
    multiply.channels = all.clone();
    multiply.blend_mode = BlendMode::Multiply;
    multiply.filters = vec![LayerFilter::Levels {
        in_low: 0.1,
        in_high: 0.9,
        gamma: 1.3,
        out_low: 0.0,
        out_high: 1.0,
    }];
    let mut masked = MaterialLayer::new_fill(
        MASKED,
        "Masked",
        LayerFill {
            color: [240, 200, 40, 255],
            metallic: 1.0,
            height: 0.8,
            emissive: [30, 0, 60],
            ..LayerFill::default()
        },
    );
    masked.channels = all.clone();
    masked.mask_generator = Some(MaskGenerator::Gradient(Gradient::DEFAULT));
    masked.mask_filters = vec![LayerFilter::Invert];
    masked.invert_final_mask = true;
    let mut group = MaterialLayer::new_group(LayerId(3), "Group");
    group.channels = all.clone();
    group.opacity = 0.8;
    group.blend_mode = BlendMode::Overlay;
    group.children = Some(vec![multiply, masked]);

    let mut anchored = MaterialLayer::new_fill(
        LayerId(6),
        "Anchored",
        LayerFill {
            color: [20, 220, 120, 255],
            roughness: 0.1,
            ..LayerFill::default()
        },
    );
    anchored.channels = all.clone();
    anchored.mask_anchor = Some(AnchorMask {
        layer: PAINT,
        channel: MaterialChannel::BaseColor,
        low: 0.1,
        high: 0.6,
        invert: false,
    });
    anchored.polygon_mask = Some(PolygonMask {
        triangles: (0..6).collect(),
        inverted: true,
    });
    let mut pass = MaterialLayer::new_group(LayerId(7), "Pass");
    pass.channels = all.clone();
    pass.blend_mode = BlendMode::PassThrough;
    pass.opacity = 0.6;
    pass.children = Some(vec![anchored]);

    let mut ids = MaterialLayer::new_fill(
        IDS,
        "By ID",
        LayerFill {
            color: [255, 0, 0, 255],
            ..LayerFill::default()
        },
    );
    ids.channels = all.clone();
    ids.channels.insert(MaterialChannel::Id);
    ids.id_mask_color = Some([0, 0, 255]);

    TextureSet {
        id: TextureSetId(1),
        name: "Region".to_owned(),
        material_slot: "Default".to_owned(),
        // Not a multiple of the dirty tile size, so edge tiles clip.
        resolution: TextureResolution {
            width: 150,
            height: 134,
        },
        channels: all,
        layers: vec![base, paint, group, pass, ids],
        height_strength: DEFAULT_HEIGHT_STRENGTH * 4.0,
        export: None,
    }
}

/// Stroke fields the edit needs, whichever kind of stroke it is.
trait Growable {
    fn points_mut(&mut self) -> (&mut Vec<Vec2>, &mut Vec<f32>);
    fn shape(&self) -> (f32, BrushTip);
}

impl Growable for PaintStroke {
    fn points_mut(&mut self) -> (&mut Vec<Vec2>, &mut Vec<f32>) {
        (&mut self.points, &mut self.pressures)
    }
    fn shape(&self) -> (f32, BrushTip) {
        (self.radius_pixels, self.tip)
    }
}

impl Growable for MaskStroke {
    fn points_mut(&mut self) -> (&mut Vec<Vec2>, &mut Vec<f32>) {
        (&mut self.points, &mut self.pressures)
    }
    fn shape(&self) -> (f32, BrushTip) {
        (self.radius_pixels, self.tip)
    }
}

impl Growable for IdStroke {
    fn points_mut(&mut self) -> (&mut Vec<Vec2>, &mut Vec<f32>) {
        (&mut self.points, &mut self.pressures)
    }
    fn shape(&self) -> (f32, BrushTip) {
        (self.radius_pixels, self.tip)
    }
}

/// Appends `point` to the last stroke (as a drag does) or starts `new`,
/// returning the texels that can change.
fn grow_or_start<S: Growable>(
    strokes: &mut Vec<S>,
    new: S,
    grow: bool,
    point: Vec2,
    pressure: f32,
    resolution: TextureResolution,
) -> Option<TexelRect> {
    let (width, height) = (resolution.width, resolution.height);
    if grow
        && let Some(stroke) = strokes.last_mut()
        && stroke.points_mut().0.len() < 64
    {
        let (radius, tip) = stroke.shape();
        let (points, pressures) = stroke.points_mut();
        points.push(point);
        if !pressures.is_empty() {
            pressures.push(pressure);
        }
        return stroke_texel_bounds(&points[points.len() - 2..], radius, tip, width, height);
    }
    let mut new = new;
    let (radius, tip) = new.shape();
    let bounds = stroke_texel_bounds(new.points_mut().0, radius, tip, width, height);
    strokes.push(new);
    bounds
}

/// Applies one random stroke edit and returns the texels it changed.
fn edit(set: &mut TextureSet, rng: &mut Lcg) -> Option<TexelRect> {
    let resolution = set.resolution;
    let layer_id = [PAINT, MULTIPLY, MASKED, IDS][rng.below(4)];
    let mask_stroke = rng.below(3) == 0;
    let radius = 2.0 + rng.unit() * 10.0;
    let tip = random_tip(rng);
    let point = rng.uv();
    let pressure = rng.unit();
    let grow = rng.below(3) != 0;
    let pressures = if rng.below(2) == 0 {
        vec![pressure]
    } else {
        Vec::new()
    };
    let value = rng.unit();
    let opacity = 0.3 + rng.unit() * 0.7;
    let color = [rng.byte(), 200, rng.byte(), 255];
    let emissive = [rng.byte(), 0, 90];
    let erase = rng.below(5) == 0;
    let layer = set.layer_mut(layer_id).unwrap();
    if layer_id == IDS {
        let stroke = IdStroke {
            points: vec![point],
            radius_pixels: radius,
            color: [0, 0, 255],
            opacity: 1.0,
            tip,
            pressures,
        };
        return grow_or_start(
            &mut layer.id_strokes,
            stroke,
            grow,
            point,
            pressure,
            resolution,
        );
    }
    if mask_stroke {
        let stroke = MaskStroke {
            points: vec![point],
            radius_pixels: radius,
            value,
            opacity,
            tip,
            pressures,
        };
        return grow_or_start(
            &mut layer.mask_strokes,
            stroke,
            grow,
            point,
            pressure,
            resolution,
        );
    }
    let stroke = PaintStroke {
        roughness: value,
        metallic: 1.0 - value,
        height: value,
        emissive,
        material_opacity: opacity,
        opacity,
        erase,
        tip,
        pressures,
        ..PaintStroke::new(vec![point], radius, color)
    };
    grow_or_start(
        &mut layer.paint_strokes,
        stroke,
        grow,
        point,
        pressure,
        resolution,
    )
}

#[test]
fn tc_0300_region_rebakes_of_random_edits_match_full_bakes_byte_for_byte() {
    let mesh = PrimitivePreset::Cube.build_mesh();
    let maps = bake_mesh_maps(
        &mesh,
        &MeshMapSettings {
            resolution: 64,
            ao_samples: 4,
            ..MeshMapSettings::default()
        },
        None,
    )
    .unwrap();
    let resources = ResourceImages::from([(ResourceId(1), Arc::new(checker_image(32)))]);
    let context = BakeContext {
        mesh: Some(&mesh),
        mesh_maps: Some(&maps),
        resources: Some(&resources),
    };
    let mut set = stack();
    assert!(set.supports_region_rebake());
    let mut rng = Lcg(0x5eed);
    let mut material = set.bake_material(context).unwrap();
    let channels = [
        MaterialChannel::BaseColor,
        MaterialChannel::Height,
        MaterialChannel::Normal,
        MaterialChannel::Id,
    ];
    let mut singles = channels.map(|channel| set.bake_channel(channel, context).unwrap());
    let mut patched = 0;
    for round in 0..24 {
        let mut dirty = DirtyTexels::new(set.resolution.width, set.resolution.height);
        for _ in 0..1 + rng.below(4) {
            if let Some(bounds) = edit(&mut set, &mut rng) {
                dirty.mark(bounds);
            }
        }
        for rect in dirty.rects() {
            patched += 1;
            set.rebake_material_region(context, &mut material, rect)
                .unwrap();
            for (channel, texture) in channels.iter().zip(&mut singles) {
                set.rebake_channel_region(*channel, context, texture, rect)
                    .unwrap();
            }
        }
        assert!(
            material == set.bake_material(context).unwrap(),
            "material differs after round {round}"
        );
        for (channel, texture) in channels.iter().zip(&singles) {
            assert!(
                *texture == set.bake_channel(*channel, context).unwrap(),
                "{channel:?} differs after round {round}"
            );
        }
    }
    assert!(patched > 24, "edits must exercise several regions");
}

/// Timing, not correctness: `cargo test --release -p substancecraft-core
/// region_rebake_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing report"]
fn region_rebake_timing() {
    let mut set = stack();
    set.resolution = TextureResolution {
        width: 2048,
        height: 2048,
    };
    // A uniform base and a busy paint layer, as while texturing by hand.
    set.layers[0].mask_generator = None;
    let fill = set.layers[0].fill.as_mut().unwrap();
    fill.textures = Default::default();
    fill.projection = TextureProjection::default();
    let mut rng = Lcg(7);
    for _ in 0..300 {
        edit(&mut set, &mut rng);
    }
    set.layers.truncate(3);
    let context = BakeContext::default();
    let started = std::time::Instant::now();
    let mut material = set.bake_material(context).unwrap();
    let full = started.elapsed();
    let dab = stroke_texel_bounds(
        &[Vec2::new(0.4, 0.4), Vec2::new(0.41, 0.4)],
        24.0,
        BrushTip::ROUND,
        2048,
        2048,
    )
    .unwrap();
    let mut dirty = DirtyTexels::new(2048, 2048);
    dirty.mark(dab);
    let started = std::time::Instant::now();
    for rect in dirty.rects() {
        set.rebake_material_region(context, &mut material, rect)
            .unwrap();
    }
    let patch = started.elapsed();
    println!(
        "2048² material, {} layers, {} strokes: full {full:?}, one-dab patch ({} rects) {patch:?}",
        set.walk_layers().len(),
        set.walk_layers()
            .iter()
            .map(|layer| layer.paint_strokes.len() + layer.mask_strokes.len())
            .sum::<usize>(),
        dirty.rects().len(),
    );
}

#[test]
fn tc_0301_region_rebake_refuses_spatial_filters_and_bad_targets() {
    let mut set = stack();
    let context = BakeContext::default();
    set.layers.truncate(2);
    set.layers[0].mask_generator = None;
    let fill = set.layers[0].fill.as_mut().unwrap();
    fill.textures = Default::default();
    fill.projection = TextureProjection::default();
    let mut material = set.bake_material(context).unwrap();
    let region = TexelRect {
        x: 3,
        y: 4,
        width: 10,
        height: 10,
    };
    assert_eq!(
        set.rebake_material_region(context, &mut material, region),
        Ok(())
    );
    let outside = TexelRect {
        x: 140,
        y: 0,
        width: 20,
        height: 4,
    };
    assert_eq!(
        set.rebake_material_region(context, &mut material, outside),
        Err(BakeError::RegionMismatch)
    );
    let mut small = material.clone();
    small.orm.width = 8;
    assert_eq!(
        set.rebake_material_region(context, &mut small, region),
        Err(BakeError::RegionMismatch)
    );
    set.layers[1].filters = vec![LayerFilter::Blur { radius: 2.0 }];
    assert!(!set.supports_region_rebake());
    assert_eq!(
        set.rebake_material_region(context, &mut material, region),
        Err(BakeError::RegionMismatch)
    );
}
