//! PERF-04 determinism: a region rebake after any sequence of stroke edits
//! must equal a full bake byte for byte.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;
use std::sync::Arc;

use glam::Vec2;

use super::*;
use crate::{
    BrushShape, BrushTip, DEFAULT_HEIGHT_STRENGTH, DirtyTexels, Gradient, IdStroke, LayerFilter,
    MaskGenerator, MaskStroke, MeshMapSettings, NoisePattern, PolygonMask, PrimitivePreset,
    StampFrame, Stencil, StencilId, StrokeSource, TextureProjection, TextureSetId, bake_mesh_maps,
    stroke_texel_bounds,
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

/// A valid surface-alignment frame: a turn, a stretch of up to 3, and a
/// shear, scaled to unit determinant.
fn random_frame(rng: &mut Lcg) -> StampFrame {
    let turn = Vec2::from_angle(rng.unit() * std::f32::consts::TAU);
    let stretch = 1.0 + rng.unit() * 2.0;
    let shear = rng.unit() - 0.5;
    // Columns of turn × [[stretch, shear], [0, 1 / stretch]].
    let first = turn * stretch;
    let second = turn * shear + turn.perp() / stretch;
    let frame = StampFrame([first.x, first.y, second.x, second.y]);
    if frame.stretch() <= crate::MAX_STAMP_STRETCH {
        frame
    } else {
        StampFrame::IDENTITY
    }
}

const STENCILS: [StencilId; 2] = [StencilId(1), StencilId(2)];

fn random_stencil(rng: &mut Lcg) -> Option<StencilId> {
    match rng.below(4) {
        0 => Some(STENCILS[0]),
        1 => Some(STENCILS[1]),
        _ => None,
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
    paint.filters = vec![LayerFilter::Invert];

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
    // Procedural content: noise over a triplanar bitmap, through a ramp.
    masked.graph = Some(crate::MaterialGraph {
        nodes: vec![
            crate::GraphNode::Generator(MaskGenerator::Noise(NoisePattern::DEFAULT)),
            crate::GraphNode::Bitmap {
                resource: ResourceId(1),
                projection: TextureProjection {
                    mode: ProjectionMode::Triplanar,
                    ..TextureProjection::default()
                },
            },
            crate::GraphNode::Blend {
                mode: BlendMode::Overlay,
                a: crate::GraphInput::Node(1),
                b: crate::GraphInput::Node(0),
                amount: crate::GraphInput::gray(0.7),
            },
            crate::GraphNode::MeshMap(crate::MeshMapKind::Curvature),
            crate::GraphNode::Ramp {
                input: crate::GraphInput::Node(2),
                stops: vec![
                    crate::RampStop {
                        position: 0.2,
                        color: [20, 40, 90],
                    },
                    crate::RampStop {
                        position: 0.8,
                        color: [250, 200, 60],
                    },
                ],
            },
        ],
        outputs: [
            (MaterialChannel::BaseColor, crate::GraphInput::Node(4)),
            (MaterialChannel::Roughness, crate::GraphInput::Node(3)),
            (MaterialChannel::Height, crate::GraphInput::Node(0)),
        ]
        .into(),
    });
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
        // Before the paint layer's Invert filter.
        effect: Some(crate::AnchorEffect::Paint),
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

    // Two quick masks: soft dabs, and an inverted stripe.
    let dabs = Stencil {
        id: STENCILS[0],
        strokes: vec![
            MaskStroke::new(vec![Vec2::new(0.3, 0.3), Vec2::new(0.7, 0.6)], 18.0, 1.0),
            MaskStroke::new(vec![Vec2::new(0.5, 0.8)], 25.0, 0.6),
        ],
        invert: false,
    };
    let stripe = Stencil {
        id: STENCILS[1],
        strokes: vec![MaskStroke {
            tip: BrushTip {
                shape: BrushShape::Square,
                ..BrushTip::ROUND
            },
            ..MaskStroke::new(vec![Vec2::new(0.0, 0.5), Vec2::new(1.0, 0.5)], 12.0, 1.0)
        }],
        invert: true,
    };
    TextureSet {
        user_channels: Default::default(),
        stencils: vec![dabs, stripe],
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

/// The per-point lists of a stroke: points, pressures, and stamp frames.
type PointLists<'a> = (&'a mut Vec<Vec2>, &'a mut Vec<f32>, &'a mut Vec<StampFrame>);

/// Stroke fields the edit needs, whichever kind of stroke it is.
trait Growable {
    fn points_mut(&mut self) -> PointLists<'_>;
    fn shape(&self) -> (f32, BrushTip);
}

impl Growable for PaintStroke {
    fn points_mut(&mut self) -> PointLists<'_> {
        (&mut self.points, &mut self.pressures, &mut self.frames)
    }
    fn shape(&self) -> (f32, BrushTip) {
        (self.radius_pixels, self.tip)
    }
}

impl Growable for MaskStroke {
    fn points_mut(&mut self) -> PointLists<'_> {
        (&mut self.points, &mut self.pressures, &mut self.frames)
    }
    fn shape(&self) -> (f32, BrushTip) {
        (self.radius_pixels, self.tip)
    }
}

impl Growable for IdStroke {
    fn points_mut(&mut self) -> PointLists<'_> {
        (&mut self.points, &mut self.pressures, &mut self.frames)
    }
    fn shape(&self) -> (f32, BrushTip) {
        (self.radius_pixels, self.tip)
    }
}

/// Appends `point` (with `frame` when the stroke is aligned) to the last
/// stroke, as a drag does, or starts `new`, returning the texels that can
/// change.
#[allow(clippy::too_many_arguments)]
fn grow_or_start<S: Growable>(
    strokes: &mut Vec<S>,
    new: S,
    grow: bool,
    point: Vec2,
    pressure: f32,
    frame: StampFrame,
    resolution: TextureResolution,
) -> Option<TexelRect> {
    let (width, height) = (resolution.width, resolution.height);
    if grow
        && let Some(stroke) = strokes.last_mut()
        && stroke.points_mut().0.len() < 64
    {
        let (radius, tip) = stroke.shape();
        let (points, pressures, frames) = stroke.points_mut();
        points.push(point);
        if !pressures.is_empty() {
            pressures.push(pressure);
        }
        if !frames.is_empty() {
            frames.push(frame);
        }
        let start = points.len() - 2;
        let frames = frames.get(start..).unwrap_or_default();
        return stroke_texel_bounds(&points[start..], radius, tip, frames, width, height);
    }
    let mut new = new;
    let (radius, tip) = new.shape();
    let (points, _, frames) = new.points_mut();
    let bounds = stroke_texel_bounds(points, radius, tip, frames, width, height);
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
    let frame = random_frame(rng);
    let frames = if rng.below(3) == 0 {
        vec![frame]
    } else {
        Vec::new()
    };
    let stencil = random_stencil(rng);
    let source = match rng.below(6) {
        _ if erase => None,
        0 => Some(StrokeSource::Clone {
            offset: (rng.uv() - Vec2::splat(0.5)) * 0.08,
        }),
        1 => Some(StrokeSource::Smudge {
            strength: 0.2 + rng.unit() * 0.8,
        }),
        _ => None,
    };
    let layer = set.layer_mut(layer_id).unwrap();
    if layer_id == IDS {
        let stroke = IdStroke {
            frames,
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
            frame,
            resolution,
        );
    }
    if mask_stroke {
        let stroke = MaskStroke {
            frames,
            stencil,
            points: vec![point],
            radius_pixels: radius,
            value,
            opacity,
            tip,
            pressures,
            camera: None,
        };
        return grow_or_start(
            &mut layer.mask_strokes,
            stroke,
            grow,
            point,
            pressure,
            frame,
            resolution,
        );
    }
    let stroke = PaintStroke {
        frames,
        stencil,
        source,
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
        frame,
        resolution,
    )
}

#[test]
fn tc_0253_region_rebakes_of_random_edits_match_full_bakes_byte_for_byte() {
    let set = stack();
    assert_eq!(set.region_halo(), 0, "no spatial filters");
    check_random_edits(set, 0x5eed, 24);
}

#[test]
fn tc_0257_region_rebakes_stay_exact_through_blur_and_sharpen() {
    let mut set = stack();
    set.resolution = TextureResolution {
        width: 512,
        height: 448,
    };
    // Spatial filters on a layer, a mask, and a group; anchors would
    // multiply the halo past this texture's size.
    set.layer_mut(PAINT).unwrap().filters = vec![LayerFilter::Blur { radius: 8.0 }];
    set.layer_mut(MASKED).unwrap().mask_filters = vec![LayerFilter::Sharpen {
        radius: 6.0,
        amount: 0.8,
    }];
    set.layer_mut(LayerId(3)).unwrap().filters = vec![LayerFilter::Blur { radius: 4.0 }];
    set.layer_mut(LayerId(6)).unwrap().mask_anchor = None;
    let halo = set.region_halo();
    assert!(halo > 0 && halo < 256, "halo {halo}");
    check_random_edits(set, 0xc1, 6);
}

/// A stack with blur and sharpen on a layer, a mask, and a group (and no
/// anchor, which would multiply the halo).
fn blurred_stack() -> TextureSet {
    let mut set = stack();
    set.layer_mut(PAINT).unwrap().filters = vec![LayerFilter::Blur { radius: 30.0 }];
    set.layer_mut(MASKED).unwrap().mask_filters = vec![LayerFilter::Sharpen {
        radius: 20.0,
        amount: 0.8,
    }];
    set.layer_mut(LayerId(3)).unwrap().filters = vec![LayerFilter::Blur { radius: 14.0 }];
    set.layer_mut(LayerId(6)).unwrap().mask_anchor = None;
    set
}

#[test]
fn tc_0258_tiled_bakes_match_single_pass_bakes() {
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
    let presets = crate::ExportPreset::built_in();
    for (name, mut set) in [("plain", stack()), ("blurred", blurred_stack())] {
        let mut rng = Lcg(42);
        for _ in 0..40 {
            edit(&mut set, &mut rng);
        }
        let tiles = set.tiles_of(64);
        assert_eq!(tiles.len(), 3 * 3, "{name}: 150 × 134 in 64-texel tiles");
        assert!(
            set.bake_material_in(context, &tiles).unwrap() == set.bake_material(context).unwrap(),
            "{name}: tiled material differs"
        );
        for preset in &presets {
            assert!(
                set.bake_export_in(preset, context, &tiles).unwrap()
                    == set.bake_export(preset, context).unwrap(),
                "{name}: tiled {} export differs",
                preset.name
            );
        }
    }
}

#[test]
fn tc_0259_sets_beyond_the_working_budget_bake_in_tiles() {
    let mut set = TextureSet {
        resolution: TextureResolution {
            width: 2304,
            height: 2048,
        },
        ..stack()
    };
    let mut rng = Lcg(9);
    while set.layer(PAINT).unwrap().paint_strokes.len() < 6 {
        edit(&mut set, &mut rng);
    }
    // Uniform fills and strokes: no mesh maps or bitmaps needed.
    set.layers.truncate(2);
    set.layers[0].mask_generator = None;
    let fill = set.layers[0].fill.as_mut().unwrap();
    fill.textures = Default::default();
    fill.projection = TextureProjection::default();
    let tiles = set.bake_tiles().unwrap();
    assert!(tiles.len() > 1, "{} tiles", tiles.len());
    assert!(
        tiles
            .iter()
            .all(|tile| set.composited_region(*tile).len() <= crate::MAX_BAKE_PIXELS)
    );
    let context = BakeContext::default();
    let material = set.bake_material(context).unwrap();
    assert_eq!(
        (material.base_color.width, material.base_color.height),
        (2304, 2048)
    );
    // The inspector composites only a texel's neighbourhood, so it works
    // past the working budget too; on either side of a tile seam it finds
    // the base fill.
    let seam = tiles[0].right();
    for x in [seam - 1, seam] {
        let uv = Vec2::new(x as f32 / 2303.0, 0.0);
        let probe = set
            .inspect_texel(MaterialChannel::Roughness, uv, context)
            .unwrap();
        assert!(probe.iter().any(|entry| entry.layer == LayerId(1)));
    }
    let exported = set
        .bake_export(&crate::ExportPreset::built_in()[0], context)
        .unwrap();
    assert!(exported.iter().all(|texture| texture.texture.width == 2304));
    // Past the texture budget a bake is refused before allocating.
    set.resolution = TextureResolution {
        width: 8192,
        height: 8193,
    };
    assert!(matches!(
        set.bake_material(context),
        Err(BakeError::ResolutionExceedsBudget { .. })
    ));
}

/// Applies seeded random stroke edits in rounds, patching each round's
/// dirty tiles (grown by the set's halo) into earlier bakes, and checks
/// the patched textures against full bakes byte for byte.
fn check_random_edits(mut set: TextureSet, seed: u64, rounds: usize) {
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
    let (width, height) = (set.resolution.width, set.resolution.height);
    let mut rng = Lcg(seed);
    let mut material = set.bake_material(context).unwrap();
    let channels = [
        MaterialChannel::BaseColor,
        MaterialChannel::Height,
        MaterialChannel::Normal,
        MaterialChannel::Id,
    ];
    let mut singles = channels.map(|channel| set.bake_channel(channel, context).unwrap());
    let (mut patched, mut partial) = (0, 0);
    for round in 0..rounds {
        let mut dirty = DirtyTexels::new(width, height);
        for _ in 0..1 + rng.below(4) {
            // Clone and smudge strokes widen the halo as they arrive.
            if let Some(bounds) = edit(&mut set, &mut rng) {
                dirty.mark(bounds.grow(set.region_halo(), width, height));
            }
        }
        for rect in dirty.rects() {
            patched += 1;
            if rect != TexelRect::full(width, height) {
                partial += 1;
            }
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
    assert!(
        patched > rounds && partial > rounds / 2,
        "edits must exercise several partial regions ({patched}, {partial})"
    );
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
    // Without mesh maps: no generator or procedural content.
    let masked = set.layer_mut(MASKED).unwrap();
    masked.mask_generator = None;
    masked.graph = None;
    let context = BakeContext::default();
    let started = std::time::Instant::now();
    let mut material = set.bake_material(context).unwrap();
    let full = started.elapsed();
    let dab = stroke_texel_bounds(
        &[Vec2::new(0.4, 0.4), Vec2::new(0.41, 0.4)],
        24.0,
        BrushTip::ROUND,
        &[],
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
fn tc_0254_region_rebake_refuses_bad_targets_and_measures_halos() {
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
    assert_eq!(set.region_halo(), 0);
    // Per-texel filters need no halo; spatial ones do, scaled with the
    // texture, and each anchored layer can repeat them.
    set.layers[1].filters = vec![LayerFilter::Invert];
    assert_eq!(set.region_halo(), 0);
    set.layers[1].filters = vec![LayerFilter::Blur { radius: 20.0 }];
    let blur = set.region_halo();
    assert!(blur > 0);
    set.layers[1].mask_anchor = Some(AnchorMask {
        layer: LayerId(1),
        channel: MaterialChannel::Roughness,
        low: 0.0,
        high: 1.0,
        invert: false,
        effect: None,
    });
    assert_eq!(set.region_halo(), blur * 2);
}
