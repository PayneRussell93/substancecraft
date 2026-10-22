//! Clone, smudge, and quick-mask strokes (feature 05).

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use glam::Vec2;

use super::*;
use crate::{
    BrushTip, DEFAULT_HEIGHT_STRENGTH, LayerId, MaskStroke, MaterialProject, ProjectError,
    StampFrame, Stencil, StencilId, StrokeCamera, StrokeSource, TextureSetId,
};

const SIZE: u32 = 64;

fn set(layers: Vec<MaterialLayer>) -> TextureSet {
    TextureSet {
        user_channels: Default::default(),
        id: TextureSetId(1),
        name: "Strokes".to_owned(),
        material_slot: "Default".to_owned(),
        resolution: TextureResolution {
            width: SIZE,
            height: SIZE,
        },
        channels: BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Roughness]),
        layers,
        height_strength: DEFAULT_HEIGHT_STRENGTH,
        export: None,
        stencils: Vec::new(),
    }
}

fn paint_layer(id: u64, strokes: Vec<PaintStroke>) -> MaterialLayer {
    let mut layer = MaterialLayer::new_paint(LayerId(id), format!("Paint {id}"));
    layer.channels = BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Roughness]);
    layer.paint_strokes = strokes;
    layer
}

/// A hard round stroke through `points`.
fn dab(points: &[(f32, f32)], radius: f32, color: [u8; 4]) -> PaintStroke {
    let mut stroke = PaintStroke::new(
        points.iter().map(|(u, v)| Vec2::new(*u, *v)).collect(),
        radius,
        color,
    );
    stroke.tip.hardness = 1.0;
    stroke.roughness = 0.9;
    stroke
}

fn color_at(texture: &BakedTexture, u: f32, v: f32) -> [u8; 4] {
    texture.sample_uv(Vec2::new(u, v)).unwrap()
}

const RED: [u8; 4] = [230, 20, 20, 255];

#[test]
fn tc_0354_clone_strokes_copy_the_stack_from_an_offset() {
    // A red dab on the layer beneath; the clone layer copies it half the
    // texture to the right.
    let source = paint_layer(1, vec![dab(&[(0.25, 0.5)], 5.0, RED)]);
    let clone = PaintStroke {
        source: Some(StrokeSource::Clone {
            offset: Vec2::new(-0.5, 0.0),
        }),
        ..dab(&[(0.75, 0.5)], 7.0, [0, 0, 255, 255])
    };
    let mut clones = paint_layer(2, vec![clone.clone()]);
    let set = set(vec![source, clones.clone()]);
    let color = set.bake_base_color().unwrap();
    assert_eq!(color_at(&color, 0.25, 0.5), RED);
    assert_eq!(color_at(&color, 0.75, 0.5), RED, "cloned, not blue");
    // Where the source was empty (past the dab's antialiased rim) the clone
    // adds nothing.
    assert_eq!(color_at(&color, 0.75 + 6.5 / 63.0, 0.5)[3], 0);
    // Every enabled channel is copied: roughness too.
    let roughness = set
        .bake_channel(MaterialChannel::Roughness, BakeContext::default())
        .unwrap();
    assert_eq!(color_at(&roughness, 0.75, 0.5)[0], unit_to_byte(0.9));

    // The layer's own earlier strokes are cloned as well.
    clones
        .paint_strokes
        .insert(0, dab(&[(0.25, 0.25)], 5.0, [20, 200, 20, 255]));
    clones.paint_strokes[1] = PaintStroke {
        points: vec![Vec2::new(0.75, 0.25)],
        ..clone
    };
    let set = self::set(vec![clones]);
    let color = set.bake_base_color().unwrap();
    assert_eq!(color_at(&color, 0.75, 0.25), [20, 200, 20, 255]);
    // The clone reads the stack before it writes, so cloning a clone in one
    // stroke never repeats the copy.
    assert_eq!(color_at(&color, 0.25, 0.25), [20, 200, 20, 255]);
}

#[test]
fn tc_0355_smudge_strokes_drag_the_stack_along_their_path() {
    let red = paint_layer(1, vec![dab(&[(0.3, 0.5)], 5.0, RED)]);
    let smudge = PaintStroke {
        source: Some(StrokeSource::Smudge { strength: 1.0 }),
        ..dab(&[(0.3, 0.5), (0.6, 0.5)], 6.0, [0, 0, 255, 255])
    };
    let before = set(vec![red.clone()]).bake_base_color().unwrap();
    let after = set(vec![red, paint_layer(2, vec![smudge])])
        .bake_base_color()
        .unwrap();
    // The dab ends five texels right of its centre (pixel 19, u = 0.3);
    // smudging rightward drags it up to a radius further.
    let at = |texture: &BakedTexture, x: u32| texture.pixel(x, 31).unwrap();
    assert_eq!(at(&before, 27)[3], 0);
    assert_eq!(at(&after, 27), RED, "dragged forward");
    assert_eq!(at(&after, 34)[3], 0, "no further than a radius");
    // Behind the stroke's start the dab is untouched (its antialiased rim
    // included).
    assert_eq!(at(&after, 14), at(&before, 14));
    assert_eq!(at(&after, 16), RED);

    // A single dab has no direction, so it leaves the stack as it is.
    let still = PaintStroke {
        source: Some(StrokeSource::Smudge { strength: 1.0 }),
        ..dab(&[(0.3, 0.5)], 6.0, [0, 0, 255, 255])
    };
    let red = paint_layer(1, vec![dab(&[(0.3, 0.5)], 5.0, RED)]);
    assert_eq!(
        set(vec![red, paint_layer(2, vec![still])])
            .bake_base_color()
            .unwrap(),
        before
    );
}

#[test]
fn sampled_strokes_do_not_thicken_semi_transparent_texels() {
    // A soft dab is semi-transparent over most of its disc.
    let soft = |points: &[(f32, f32)], radius: f32| PaintStroke {
        tip: BrushTip::ROUND,
        ..dab(points, radius, RED)
    };
    let red = || paint_layer(1, vec![soft(&[(0.3, 0.5)], 8.0)]);
    let before = set(vec![red()]).bake_base_color().unwrap();
    let sampled = |source: StrokeSource, points: &[(f32, f32)]| {
        set(vec![
            red(),
            paint_layer(
                2,
                vec![PaintStroke {
                    source: Some(source),
                    ..soft(points, 10.0)
                }],
            ),
        ])
        .bake_base_color()
        .unwrap()
    };
    // Copying the stack onto itself, or smudging without moving, changes
    // nothing.
    let onto_itself = StrokeSource::Clone { offset: Vec2::ZERO };
    assert_eq!(sampled(onto_itself, &[(0.3, 0.5)]), before);
    let still = StrokeSource::Smudge { strength: 1.0 };
    assert_eq!(sampled(still, &[(0.3, 0.5)]), before);
    // A hard, whole-texel copy into empty space carries the soft dab's
    // alpha exactly, edge included.
    let shift = 25.0 / 63.0;
    let cloned = set(vec![
        red(),
        paint_layer(
            2,
            vec![PaintStroke {
                source: Some(StrokeSource::Clone {
                    offset: Vec2::new(-shift, 0.0),
                }),
                ..dab(&[(0.3 + shift, 0.5)], 12.0, RED)
            }],
        ),
    ])
    .bake_base_color()
    .unwrap();
    let alpha = |texture: &BakedTexture, x: u32| i32::from(texture.pixel(x, 31).unwrap()[3]);
    let mut partial = 0;
    for x in 10..28 {
        let source = alpha(&before, x);
        partial += usize::from(source > 0 && source < 255);
        let copied = alpha(&cloned, x + 25);
        assert!(
            (copied - source).abs() <= 1,
            "pixel {x}: {copied} for {source}"
        );
    }
    assert!(partial > 4, "the dab has a soft edge to copy");
}

#[test]
fn tc_0370_blur_strokes_soften_the_stack_under_them() {
    let red = || paint_layer(1, vec![dab(&[(0.3, 0.5)], 5.0, RED)]);
    let blur = PaintStroke {
        source: Some(StrokeSource::Blur { strength: 0.5 }),
        ..dab(&[(0.4, 0.5)], 8.0, [0, 0, 255, 255])
    };
    let before = set(vec![red()]).bake_base_color().unwrap();
    let after = set(vec![red(), paint_layer(2, vec![blur])])
        .bake_base_color()
        .unwrap();
    let at = |texture: &BakedTexture, x: u32| texture.pixel(x, 31).unwrap();
    // The dab's right edge (pixel 24) is hard before and soft after: just
    // outside it some red bleeds in. Like smudge, blur paints over the
    // stack, so it softens edges outward without thinning what is there.
    assert_eq!(at(&before, 26)[3], 0);
    let outside = at(&after, 26);
    assert!(outside[3] > 0 && outside[3] < 255, "{outside:?}");
    assert_eq!(&outside[..3], &RED[..3], "blur adds no colour of its own");
    // Outside the brush the dab is untouched.
    assert_eq!(at(&after, 15), RED);
}

#[test]
fn tc_0356_quick_masks_limit_paint_and_mask_strokes() {
    // A quick mask open around u = 0.25 only; a stroke across the whole
    // texture paints there alone.
    let stencil = Stencil {
        id: StencilId(1),
        strokes: vec![MaskStroke {
            tip: crate::BrushTip {
                hardness: 1.0,
                ..crate::BrushTip::ROUND
            },
            ..MaskStroke::new(vec![Vec2::new(0.25, 0.5)], 10.0, 1.0)
        }],
        invert: false,
    };
    let across = PaintStroke {
        stencil: Some(StencilId(1)),
        ..dab(&[(0.05, 0.5), (0.95, 0.5)], 3.0, RED)
    };
    let mut limited = set(vec![paint_layer(1, vec![across.clone()])]);
    limited.stencils = vec![stencil.clone()];
    let color = limited.bake_base_color().unwrap();
    assert_eq!(color_at(&color, 0.25, 0.5), RED);
    assert_eq!(color_at(&color, 0.75, 0.5)[3], 0);

    // Inverted, it protects that spot instead.
    limited.stencils[0].invert = true;
    let color = limited.bake_base_color().unwrap();
    assert_eq!(color_at(&color, 0.25, 0.5)[3], 0);
    assert_eq!(color_at(&color, 0.75, 0.5), RED);

    // Layer mask strokes honour it too.
    limited.stencils[0].invert = false;
    let mut masked = MaterialLayer::new_fill(LayerId(2), "Fill", LayerFill::default());
    masked.mask_strokes = vec![MaskStroke {
        stencil: Some(StencilId(1)),
        tip: crate::BrushTip {
            hardness: 1.0,
            ..crate::BrushTip::ROUND
        },
        ..MaskStroke::new(vec![Vec2::new(0.05, 0.5), Vec2::new(0.95, 0.5)], 3.0, 0.0)
    }];
    limited.layers = vec![masked];
    let mask = limited
        .layer_mask_in(
            &limited.layers[0],
            BakeContext::default(),
            None,
            limited.texel_rect(),
        )
        .unwrap()
        .unwrap();
    let mask_at = |x: u32| mask[(31 * SIZE + x) as usize];
    assert_eq!(mask_at(16), 0.0, "cut where the quick mask is open");
    assert_eq!(mask_at(47), 1.0, "kept where it is closed");

    // UDIM tiles shift quick masks with their strokes; a quick mask painted
    // through the camera needs mesh maps.
    let shifted = limited.shifted_to_tile(crate::UdimTile(1002));
    assert_eq!(
        shifted.stencils[0].strokes[0].points[0],
        Vec2::new(0.25 - 1.0, 0.5)
    );
    assert!(!limited.requires_mesh_maps());
    limited.stencils[0].strokes[0].camera = Some(StrokeCamera {
        transform: glam::Mat4::IDENTITY,
        viewport: Vec2::splat(64.0),
    });
    assert!(limited.requires_mesh_maps());

    // A stroke naming a missing quick mask does not bake or validate.
    let mut missing = set(vec![paint_layer(1, vec![across])]);
    assert_eq!(missing.bake_base_color(), Err(BakeError::InvalidStrokeData));
    let mut project = MaterialProject::empty();
    project.texture_sets.push(missing.clone());
    assert_eq!(
        project.validate(),
        Err(ProjectError::InvalidTextureSetSettings(TextureSetId(1)))
    );
    // Duplicate ids, and quick masks whose own strokes are limited, are
    // invalid too.
    missing.stencils = vec![stencil.clone(), stencil.clone()];
    assert!(!missing.has_valid_stencils());
    let mut nested = stencil;
    nested.strokes[0].stencil = Some(StencilId(1));
    missing.stencils = vec![nested];
    assert!(!missing.has_valid_stencils());
}

#[test]
fn tc_0357_sampling_strokes_and_frames_reject_projection_and_erasing() {
    let valid = |layer: MaterialLayer| crate::valid_layer_strokes(&layer);
    let clone = PaintStroke {
        source: Some(StrokeSource::Clone {
            offset: Vec2::new(0.1, 0.0),
        }),
        ..dab(&[(0.5, 0.5)], 4.0, RED)
    };
    assert!(valid(paint_layer(1, vec![clone.clone()])));
    assert!(!valid(paint_layer(
        1,
        vec![PaintStroke {
            erase: true,
            ..clone.clone()
        }]
    )));
    let camera = StrokeCamera {
        transform: glam::Mat4::IDENTITY,
        viewport: Vec2::splat(64.0),
    };
    assert!(!valid(paint_layer(
        1,
        vec![PaintStroke {
            camera: Some(camera),
            points: vec![Vec2::splat(32.0)],
            ..clone.clone()
        }]
    )));
    assert!(!valid(paint_layer(
        1,
        vec![PaintStroke {
            source: Some(StrokeSource::Smudge { strength: 0.0 }),
            ..clone.clone()
        }]
    )));
    // Projected strokes take their shape from the screen, not frames.
    let projected = PaintStroke {
        camera: Some(camera),
        points: vec![Vec2::splat(32.0)],
        ..dab(&[(0.5, 0.5)], 4.0, RED)
    };
    assert!(valid(paint_layer(1, vec![projected.clone()])));
    assert!(!valid(paint_layer(
        1,
        vec![PaintStroke {
            frames: vec![StampFrame::IDENTITY],
            ..projected
        }]
    )));

    // Clone reach widens the region halo by the offset in texels.
    let set = set(vec![paint_layer(1, vec![clone])]);
    assert_eq!(set.region_halo(), (0.1 * 63.0_f32).ceil() as u32 + 1);
}
