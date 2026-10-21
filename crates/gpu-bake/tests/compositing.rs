//! GPU layer compositing against the CPU compositor (PERF-06, TC-0412).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{Vec2, Vec3};
use substancecraft_core::{
    Axis, BakeContext, BitmapMask, BlendMode, BrushShape, BrushTip, ColorSpace, DecalPlacement,
    Direction, Dirt, EdgeWear, FillTextures, Gradient, GraphInput, GraphNode, LayerFill,
    LayerFilter, LayerId, MaskGenerator, MaskStroke, MaterialChannel, MaterialGraph, MaterialLayer,
    MaterialTextures, MathOp, MeshGeometry, MeshMapKind, MeshMapSettings, MeshMaps, NoisePattern,
    Orientation, PaintStroke, Pattern, PatternKind, PlanError, PolygonMask, PrimitivePreset,
    ProjectionMode, RampStop, ResourceId, ResourceImages, SourceComponent, StrokeSource,
    TextureImage, TextureProjection, TextureResolution, TextureSet, TextureSetId, Thickness,
    bake_mesh_maps,
};
use substancecraft_gpu_bake::{AdapterChoice, CompositeError, GpuCompositor, texture_differences};

const SIZE: u32 = 128;

fn checker(size: u32, sixteen: bool, color_space: ColorSpace) -> TextureImage {
    let mut rgba8 = Vec::new();
    let mut rgba16 = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let cell = (x / 8 + y / 8) % 2 == 0;
            let r = (x * 255 / size.max(1)) as u8;
            let g = (y * 255 / size.max(1)) as u8;
            let b = if cell { 220 } else { 30 };
            let a = if (x + y) % 23 == 0 { 120 } else { 255 };
            rgba8.extend([r, g, b, a]);
            rgba16.extend(
                [r, g, b, a].map(|value| (u16::from(value) * 257).saturating_add(x as u16 % 7)),
            );
        }
    }
    TextureImage {
        width: size,
        height: size,
        rgba8,
        rgba16: sixteen.then_some(rgba16),
        color_space,
    }
}

fn set(layers: Vec<MaterialLayer>) -> TextureSet {
    TextureSet {
        user_channels: Default::default(),
        stencils: Default::default(),
        id: TextureSetId(1),
        name: "Probe".to_owned(),
        material_slot: "Probe".to_owned(),
        resolution: TextureResolution {
            width: SIZE,
            height: SIZE,
        },
        channels: Default::default(),
        layers,
        height_strength: 1.0,
        export: None,
    }
}

fn fill(id: u64, setup: impl FnOnce(&mut MaterialLayer)) -> MaterialLayer {
    let mut layer = MaterialLayer::new_fill(
        LayerId(id),
        format!("Layer {id}"),
        LayerFill {
            color: [
                (id * 53 % 256) as u8,
                (id * 97 % 256) as u8,
                (id * 31 % 256) as u8,
                255,
            ],
            roughness: (id as f32 * 0.17) % 1.0,
            metallic: (id as f32 * 0.29) % 1.0,
            height: (id as f32 * 0.37) % 1.0,
            emissive: [(id * 13 % 256) as u8, 0, (id * 7 % 256) as u8],
            material_opacity: 1.0 - (id as f32 * 0.05) % 0.5,
            ..LayerFill::default()
        },
    );
    layer.channels = [
        MaterialChannel::BaseColor,
        MaterialChannel::Roughness,
        MaterialChannel::Metallic,
        MaterialChannel::Height,
        MaterialChannel::Emissive,
        MaterialChannel::Opacity,
        MaterialChannel::Normal,
    ]
    .into();
    setup(&mut layer);
    layer
}

struct Scene {
    mesh: MeshGeometry,
    maps: MeshMaps,
    images: ResourceImages,
}

impl Scene {
    fn new(preset: PrimitivePreset) -> Self {
        let mesh = preset.build_mesh();
        let maps = bake_mesh_maps(
            &mesh,
            &MeshMapSettings {
                resolution: SIZE,
                ao_samples: 8,
                ..MeshMapSettings::default()
            },
            None,
        )
        .unwrap();
        let images = ResourceImages::from([
            (
                ResourceId(1),
                Arc::new(checker(SIZE, false, ColorSpace::Srgb)),
            ),
            (
                ResourceId(2),
                Arc::new(checker(48, true, ColorSpace::LinearSrgb)),
            ),
            (
                ResourceId(3),
                Arc::new(checker(32, false, ColorSpace::AcesCg)),
            ),
        ]);
        Self { mesh, maps, images }
    }

    fn context(&self) -> BakeContext<'_> {
        BakeContext {
            mesh: Some(&self.mesh),
            mesh_maps: Some(&self.maps),
            resources: Some(&self.images),
        }
    }
}

/// Every output within tolerance of the CPU bake: a mean difference under
/// 0.5 of an 8-bit level and at most 1% of values more than 2 levels apart
/// (GPU transcendentals and fused arithmetic round differently).
fn assert_close(name: &str, gpu: &MaterialTextures, cpu: &MaterialTextures) {
    for (output, a, b) in [
        ("base colour", &gpu.base_color, &cpu.base_color),
        ("ORM", &gpu.orm, &cpu.orm),
        ("normal", &gpu.normal, &cpu.normal),
        ("emissive", &gpu.emissive, &cpu.emissive),
    ] {
        assert_eq!(a.rgba8.len(), b.rgba8.len(), "{name} {output}");
        let (mean, far) = texture_differences(a, b, 2);
        eprintln!(
            "{name:>24} {output:<12} mean {mean:.4} far {:.3}%",
            far * 100.0
        );
        assert!(mean < 0.5, "{name} {output}: mean difference {mean}");
        assert!(
            far <= 0.01,
            "{name} {output}: {:.2}% far apart",
            far * 100.0
        );
    }
}

fn compare(compositor: &GpuCompositor, scene: &Scene, name: &str, layers: Vec<MaterialLayer>) {
    let set = set(layers);
    let cpu = set.bake_material(scene.context()).unwrap();
    // The stack must change the material, or matching it proves nothing.
    let plain = self::set(Vec::new())
        .bake_material(scene.context())
        .unwrap();
    assert_ne!(cpu.base_color, plain.base_color, "{name} paints nothing");
    let gpu = compositor
        .bake_material(&set, scene.context())
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_close(name, &gpu, &cpu);
}

fn generators() -> Vec<MaskGenerator> {
    let mut generators = vec![
        MaskGenerator::EdgeWear(EdgeWear::DEFAULT),
        MaskGenerator::EdgeWear(EdgeWear {
            feature_size: Some(0.2),
            breakup: 0.0,
            ..EdgeWear::DEFAULT
        }),
        MaskGenerator::Dirt(Dirt::DEFAULT),
        MaskGenerator::Orientation(Orientation {
            direction: Direction::PositiveX,
            ..Orientation::DEFAULT
        }),
        MaskGenerator::Gradient(Gradient {
            axis: Axis::X,
            start: 0.8,
            end: 0.2,
        }),
        MaskGenerator::Noise(NoisePattern::DEFAULT),
        MaskGenerator::Thickness(Thickness::DEFAULT),
        MaskGenerator::Thickness(Thickness {
            thin: false,
            ..Thickness::DEFAULT
        }),
    ];
    for kind in PatternKind::ALL {
        generators.push(MaskGenerator::Pattern(Pattern {
            kind,
            axis: Axis::Z,
            ..Pattern::DEFAULT
        }));
    }
    generators
}

fn projections() -> Vec<TextureProjection> {
    let mut projections = vec![
        TextureProjection::default(),
        TextureProjection {
            scale: 3.0,
            offset: Vec2::new(0.25, -0.1),
            rotation_degrees: 37.0,
            ..TextureProjection::default()
        },
    ];
    for mode in [
        ProjectionMode::Triplanar,
        ProjectionMode::Planar,
        ProjectionMode::Cylindrical,
        ProjectionMode::Spherical,
    ] {
        projections.push(TextureProjection {
            mode,
            axis: Axis::X,
            scale: 2.0,
            ..TextureProjection::default()
        });
    }
    projections.push(TextureProjection {
        mode: ProjectionMode::Triplanar,
        tile_size: Some(0.5),
        blend_sharpness: 9.0,
        ..TextureProjection::default()
    });
    projections.push(TextureProjection {
        mode: ProjectionMode::Decal,
        decal: DecalPlacement {
            center: Vec3::new(0.0, 0.0, 0.5),
            normal: Vec3::Z,
            up: Vec3::Y,
            size: Vec2::new(0.8, 0.6),
            depth: 1.0,
            ..DecalPlacement::default()
        },
        ..TextureProjection::default()
    });
    projections
}

fn per_texel_filters() -> Vec<LayerFilter> {
    LayerFilter::PRESETS
        .into_iter()
        .filter(|filter| filter.is_per_texel())
        .chain([
            LayerFilter::Levels {
                in_low: 0.1,
                in_high: 0.8,
                gamma: 2.2,
                out_low: 0.05,
                out_high: 0.9,
            },
            LayerFilter::HueSaturation {
                hue: 0.2,
                saturation: 1.6,
                lightness: -0.1,
            },
            LayerFilter::BrightnessContrast {
                brightness: 0.1,
                contrast: 0.5,
            },
            LayerFilter::Threshold {
                level: 0.4,
                softness: 0.0,
            },
        ])
        .collect()
}

fn graph() -> MaterialGraph {
    let node = GraphInput::Node;
    MaterialGraph {
        nodes: vec![
            GraphNode::Generator(MaskGenerator::Noise(NoisePattern::DEFAULT)),
            GraphNode::Bitmap {
                resource: ResourceId(2),
                projection: TextureProjection {
                    mode: ProjectionMode::Triplanar,
                    ..TextureProjection::default()
                },
            },
            GraphNode::Blend {
                mode: BlendMode::Overlay,
                a: node(1),
                b: node(0),
                amount: GraphInput::gray(0.7),
            },
            GraphNode::MeshMap(MeshMapKind::Curvature),
            GraphNode::Math {
                op: MathOp::Power,
                a: node(2),
                b: node(3),
            },
            GraphNode::Adjust {
                input: node(4),
                filter: LayerFilter::HueSaturation {
                    hue: 0.3,
                    saturation: 1.2,
                    lightness: 0.0,
                },
            },
            GraphNode::Uv,
            GraphNode::Split {
                input: node(6),
                component: SourceComponent::Green,
            },
            GraphNode::Ramp {
                input: node(7),
                stops: vec![
                    RampStop {
                        position: 0.1,
                        color: [20, 40, 200],
                    },
                    RampStop {
                        position: 0.6,
                        color: [240, 200, 10],
                    },
                    RampStop {
                        position: 0.9,
                        color: [255, 255, 255],
                    },
                ],
            },
            GraphNode::Math {
                op: MathOp::Divide,
                a: node(5),
                b: node(8),
            },
            GraphNode::MeshMap(MeshMapKind::WorldNormal),
        ],
        outputs: BTreeMap::from([
            (MaterialChannel::BaseColor, node(9)),
            (MaterialChannel::Roughness, node(7)),
            (MaterialChannel::Normal, node(10)),
            (MaterialChannel::Metallic, GraphInput::gray(0.3)),
        ]),
    }
}

#[test]
fn tc_0412_gpu_compositing_matches_the_cpu_within_tolerance() {
    let compositor = match GpuCompositor::new(AdapterChoice::Any) {
        Ok(compositor) => compositor,
        // Only a missing adapter skips; a shader the device rejects fails.
        Err(error) if !error.contains("Validation") && !error.contains("hader") => {
            eprintln!("TC-0412 skipped: no GPU adapter ({error})");
            return;
        }
        Err(error) => panic!("the compositor did not start: {error}"),
    };
    eprintln!("compositing on {}", compositor.name());
    for preset in [PrimitivePreset::Sphere, PrimitivePreset::Cube] {
        let scene = Scene::new(preset);

        // Every generator as a mask, under every blend mode.
        let layers = generators()
            .into_iter()
            .zip(BlendMode::ALL.into_iter().cycle())
            .enumerate()
            .map(|(index, (generator, blend))| {
                fill(index as u64 + 1, |layer| {
                    layer.mask_generator = Some(generator);
                    layer.blend_mode = blend;
                    layer.opacity = 0.85;
                })
            })
            .collect();
        compare(&compositor, &scene, "generators and blends", layers);

        // Bitmaps through every projection, 8- and 16-bit, three spaces.
        let layers = projections()
            .into_iter()
            .enumerate()
            .map(|(index, projection)| {
                let resource = ResourceId(index as u64 % 3 + 1);
                fill(index as u64 + 1, |layer| {
                    let fill = layer.fill.as_mut().unwrap();
                    fill.textures = FillTextures {
                        base_color: Some(resource),
                        roughness: Some(resource),
                        height: Some(resource),
                        normal: Some(ResourceId(1)),
                        emissive: Some(resource),
                        ..FillTextures::default()
                    };
                    fill.projection = projection;
                    layer.opacity = 0.7;
                })
            })
            .collect();
        compare(&compositor, &scene, "projections", layers);

        // Per-texel filters on content and masks; bitmap masks; inversion.
        let layers = per_texel_filters()
            .into_iter()
            .enumerate()
            .map(|(index, filter)| {
                fill(index as u64 + 1, |layer| {
                    layer.filters = vec![filter, LayerFilter::Invert];
                    layer.mask_generator = Some(MaskGenerator::Noise(NoisePattern {
                        seed: index as u32,
                        ..NoisePattern::DEFAULT
                    }));
                    layer.mask_filters = vec![filter];
                    layer.mask_bitmap = Some(BitmapMask {
                        resource: ResourceId(2),
                        projection: TextureProjection::default(),
                        invert: index % 2 == 0,
                    });
                    layer.invert_final_mask = index % 3 == 0;
                    layer.blend_mode = BlendMode::ALL[index % BlendMode::ALL.len()];
                    if index % 2 == 1 {
                        layer.filter_channels = [MaterialChannel::BaseColor].into();
                    }
                })
            })
            .collect();
        compare(&compositor, &scene, "filters and masks", layers);

        // A material graph using every kind of node.
        let layers = vec![
            fill(1, |_| {}),
            fill(2, |layer| {
                layer.graph = Some(graph());
                layer.opacity = 0.9;
            }),
        ];
        compare(&compositor, &scene, "graph", layers);

        // Groups: masked and filtered, nested, and pass-through.
        let mut inner = MaterialLayer::new_group(LayerId(10), "Inner");
        inner.children = Some(vec![
            fill(11, |layer| layer.blend_mode = BlendMode::Screen),
            fill(12, |layer| {
                layer.mask_generator = Some(MaskGenerator::EdgeWear(EdgeWear::DEFAULT));
            }),
        ]);
        inner.filters = vec![LayerFilter::Posterize { levels: 6 }];
        inner.mask_generator = Some(MaskGenerator::Gradient(Gradient::DEFAULT));
        let mut outer = MaterialLayer::new_group(LayerId(20), "Outer");
        outer.children = Some(vec![fill(21, |_| {}), inner]);
        outer.opacity = 0.8;
        outer.blend_mode = BlendMode::Multiply;
        let mut through = MaterialLayer::new_group(LayerId(30), "Through");
        through.blend_mode = BlendMode::PassThrough;
        through.children = Some(vec![fill(31, |layer| {
            layer.blend_mode = BlendMode::Difference;
            layer.opacity = 0.6;
        })]);
        through.filters = vec![LayerFilter::Levels {
            in_low: 0.0,
            in_high: 0.9,
            gamma: 1.4,
            out_low: 0.0,
            out_high: 1.0,
        }];
        through.mask_generator = Some(MaskGenerator::Noise(NoisePattern::DEFAULT));
        compare(
            &compositor,
            &scene,
            "groups",
            vec![fill(1, |_| {}), outer, through],
        );

        // Work left to the CPU: strokes, blurs, mask strokes, polygons, and
        // a blurred group, composited with GPU layers around them.
        let mut painted = MaterialLayer::new_paint(LayerId(40), "Painted");
        painted.channels.insert(MaterialChannel::Height);
        painted.paint_strokes.push(PaintStroke {
            tip: BrushTip {
                shape: BrushShape::Splatter,
                ..BrushTip::ROUND
            },
            height: 0.9,
            ..PaintStroke::new(
                vec![Vec2::new(0.2, 0.3), Vec2::new(0.8, 0.6)],
                9.0,
                [250, 20, 20, 255],
            )
        });
        painted.mask_strokes.push(MaskStroke::new(
            vec![Vec2::new(0.3, 0.2), Vec2::new(0.4, 0.9)],
            12.0,
            0.0,
        ));
        let blurred = fill(41, |layer| {
            layer.filters = vec![LayerFilter::Blur { radius: 20.0 }, LayerFilter::Invert];
            layer.polygon_mask = Some(PolygonMask {
                triangles: (0..40).collect(),
                inverted: false,
            });
        });
        let mut soft_group = MaterialLayer::new_group(LayerId(50), "Soft group");
        soft_group.children = Some(vec![fill(51, |layer| {
            layer.mask_generator = Some(MaskGenerator::Dirt(Dirt::DEFAULT));
        })]);
        soft_group.filters = vec![LayerFilter::Sharpen {
            radius: 4.0,
            amount: 1.0,
        }];
        compare(
            &compositor,
            &scene,
            "CPU content and masks",
            vec![
                fill(1, |_| {}),
                painted,
                blurred,
                soft_group,
                fill(2, |layer| {
                    layer.opacity = 0.3;
                }),
            ],
        );
    }

    // Stacks that read the stack beneath a layer stay on the CPU.
    let scene = Scene::new(PrimitivePreset::Sphere);
    let mut smudged = MaterialLayer::new_paint(LayerId(60), "Smudged");
    smudged.paint_strokes.push(PaintStroke {
        source: Some(StrokeSource::Smudge { strength: 0.5 }),
        ..PaintStroke::new(
            vec![Vec2::splat(0.5), Vec2::splat(0.6)],
            8.0,
            [0, 0, 0, 255],
        )
    });
    let refused = compositor.bake_material(&set(vec![fill(1, |_| {}), smudged]), scene.context());
    assert!(
        matches!(
            refused,
            Err(CompositeError::Plan(PlanError::Unsupported(_)))
        ),
        "{refused:?}"
    );
}

/// Timing, not correctness: `cargo test --release -p
/// substancecraft-gpu-bake --test compositing compositing_timing --
/// --ignored --nocapture`.
#[test]
#[ignore = "timing report"]
fn compositing_timing() {
    let compositor = GpuCompositor::new(AdapterChoice::Hardware).unwrap();
    let mesh = PrimitivePreset::Sphere.build_mesh();
    let maps = bake_mesh_maps(
        &mesh,
        &MeshMapSettings {
            resolution: 1024,
            ao_samples: 2,
            ..MeshMapSettings::default()
        },
        None,
    )
    .unwrap();
    let images = ResourceImages::from([
        (
            ResourceId(1),
            Arc::new(checker(1024, false, ColorSpace::Srgb)),
        ),
        (
            ResourceId(2),
            Arc::new(checker(512, true, ColorSpace::LinearSrgb)),
        ),
        (
            ResourceId(3),
            Arc::new(checker(256, false, ColorSpace::AcesCg)),
        ),
    ]);
    let context = BakeContext {
        mesh: Some(&mesh),
        mesh_maps: Some(&maps),
        resources: Some(&images),
    };
    let triplanar = TextureProjection {
        mode: ProjectionMode::Triplanar,
        scale: 3.0,
        ..TextureProjection::default()
    };
    let mut painted = MaterialLayer::new_paint(LayerId(90), "Painted");
    painted.paint_strokes.push(PaintStroke::new(
        (0..40).map(|i| Vec2::new(i as f32 / 40.0, 0.5)).collect(),
        30.0,
        [200, 30, 30, 255],
    ));
    let mut layers = vec![
        fill(1, |layer| {
            let fill = layer.fill.as_mut().unwrap();
            fill.textures.base_color = Some(ResourceId(1));
            fill.textures.height = Some(ResourceId(1));
            fill.projection = triplanar;
            layer.mask_generator = Some(MaskGenerator::Noise(NoisePattern::DEFAULT));
        }),
        fill(2, |layer| layer.graph = Some(graph())),
        fill(3, |layer| {
            layer.mask_generator = Some(MaskGenerator::EdgeWear(EdgeWear::DEFAULT));
            layer.filters = vec![LayerFilter::Levels {
                in_low: 0.1,
                in_high: 0.9,
                gamma: 1.3,
                out_low: 0.0,
                out_high: 1.0,
            }];
        }),
        fill(4, |layer| {
            layer.mask_generator = Some(MaskGenerator::Dirt(Dirt::DEFAULT));
            layer.blend_mode = BlendMode::Multiply;
        }),
        painted,
    ];
    for (index, kind) in PatternKind::ALL.into_iter().enumerate() {
        layers.push(fill(10 + index as u64, |layer| {
            layer.mask_generator = Some(MaskGenerator::Pattern(Pattern {
                kind,
                ..Pattern::DEFAULT
            }));
            layer.opacity = 0.5;
        }));
    }
    let mut set = set(layers);
    set.resolution = TextureResolution {
        width: 2048,
        height: 2048,
    };
    let started = std::time::Instant::now();
    let cpu = set.bake_material(context).unwrap();
    let cpu_time = started.elapsed();
    // The first GPU bake compiles pipelines on some drivers; time a second.
    compositor.bake_material(&set, context).unwrap();
    let started = std::time::Instant::now();
    let gpu = compositor.bake_material(&set, context).unwrap();
    let gpu_time = started.elapsed();
    eprintln!(
        "2048² material, {} layers on {}: CPU {cpu_time:?}, GPU {gpu_time:?}",
        set.layers.len(),
        compositor.name()
    );
    assert_close("timing stack", &gpu, &cpu);
}
