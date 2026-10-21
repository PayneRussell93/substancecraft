//! PROD-08: a headless export of one fixed project writes the same bytes on
//! every OS and CPU a release supports.
//!
//! The digests in `tests/fixtures/determinism-digests.txt` were written by
//! this test. CI runs it on Windows (x86-64), Linux (x86-64), and macOS
//! (ARM64), so a platform that rounds any step differently fails here. The
//! projects lean on everything that used to reach the platform maths
//! library or the path separator: the built-in sphere, an imported glTF
//! corner (so ambient-occlusion rays hit something) stored under a
//! Windows-style relative path, curvature bakes, triplanar, rotated, and
//! spherical projections, a linear bitmap also stored with backslashes,
//! generator masks, gamma filters, a graph power node, and jittered strokes
//! following their path. After an intentional output change, rerun with
//! `SUBSTANCECRAFT_BLESS=1` and commit the new digests.

#![allow(clippy::unwrap_used, clippy::panic)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glam::Vec2;
use substancecraft_api::{
    ExportJob, TextureFormat, file_content_hash, run_export_job, save_project_file,
};
use substancecraft_core::color::ColorSpace;
use substancecraft_core::{
    BrushShape, BrushTip, Dirt, EdgeWear, GraphInput, GraphNode, LayerFill, LayerFilter, LayerId,
    MaskGenerator, MaterialChannel, MaterialGraph, MaterialLayer, MaterialProject, MathOp,
    MeshAsset, MeshId, MeshSmoothing, Orientation, PaintStroke, ProjectResource, ProjectionMode,
    ResourceId, TextureProjection, TextureResolution, TextureSet, TextureSetId,
};

const DIGESTS: &str = "tests/fixtures/determinism-digests.txt";

/// A 16×16 tile with smooth gradients and hard edges, so resampling and
/// colour conversion both show in the output.
fn write_tile(path: &Path) {
    let mut image = image::RgbaImage::new(16, 16);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let checker = if (x / 4 + y / 4) % 2 == 0 { 40 } else { 0 };
        *pixel = image::Rgba([
            (x * 15 + checker) as u8,
            (y * 15) as u8,
            (255 - x * 8 - y * 4) as u8,
            255,
        ]);
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image.save(path).unwrap();
}

/// A floor and a wall meeting at an inside corner, as glTF with an
/// external buffer: each face is its own UV island.
fn write_corner(gltf: &Path) {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    let steps = 4;
    let mut face = |corner: fn(f32, f32) -> [f32; 3], normal: [f32; 3], u0: f32| {
        let first = positions.len() as u16;
        for j in 0..=steps {
            for i in 0..=steps {
                let (a, b) = (i as f32 / steps as f32, j as f32 / steps as f32);
                positions.push(corner(a, b));
                normals.push(normal);
                uvs.push([u0 + a * 0.46, 0.02 + b * 0.96]);
            }
        }
        let row = steps as u16 + 1;
        for j in 0..steps as u16 {
            for i in 0..steps as u16 {
                let at = first + j * row + i;
                // Counter-clockwise seen from the side the normal faces.
                indices.extend([at, at + 1, at + row, at + 1, at + row + 1, at + row]);
            }
        }
    };
    face(
        |a, b| [a * 2.0 - 1.0, 0.0, 1.0 - b * 2.0],
        [0.0, 1.0, 0.0],
        0.02,
    );
    face(|a, b| [a * 2.0 - 1.0, b * 2.0, -1.0], [0.0, 0.0, 1.0], 0.52);

    let mut bytes = Vec::new();
    for value in positions.iter().chain(&normals).flatten() {
        bytes.extend(value.to_le_bytes());
    }
    for value in uvs.iter().flatten() {
        bytes.extend(value.to_le_bytes());
    }
    let index_offset = bytes.len();
    for index in &indices {
        bytes.extend(index.to_le_bytes());
    }
    let count = positions.len();
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],
"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3}}]}}],
"buffers":[{{"uri":"corner.bin","byteLength":{total}}}],
"bufferViews":[
{{"buffer":0,"byteOffset":0,"byteLength":{vec3}}},
{{"buffer":0,"byteOffset":{vec3},"byteLength":{vec3}}},
{{"buffer":0,"byteOffset":{uv_offset},"byteLength":{vec2}}},
{{"buffer":0,"byteOffset":{index_offset},"byteLength":{index_bytes}}}],
"accessors":[
{{"bufferView":0,"componentType":5126,"count":{count},"type":"VEC3","min":[-1,0,-1],"max":[1,2,1]}},
{{"bufferView":1,"componentType":5126,"count":{count},"type":"VEC3"}},
{{"bufferView":2,"componentType":5126,"count":{count},"type":"VEC2"}},
{{"bufferView":3,"componentType":5123,"count":{indices},"type":"SCALAR"}}]}}"#,
        total = bytes.len(),
        vec3 = count * 12,
        uv_offset = count * 24,
        vec2 = count * 8,
        index_bytes = indices.len() * 2,
        indices = indices.len(),
    );
    std::fs::create_dir_all(gltf.parent().unwrap()).unwrap();
    std::fs::write(gltf, json).unwrap();
    std::fs::write(gltf.with_file_name("corner.bin"), bytes).unwrap();
}

fn project(mesh: &str) -> MaterialProject {
    let tile = ResourceId(1);
    let mut base = MaterialLayer::new_fill(
        LayerId(1),
        "Tiled base",
        LayerFill {
            textures: substancecraft_core::FillTextures {
                base_color: Some(tile),
                ..Default::default()
            },
            projection: TextureProjection {
                mode: ProjectionMode::Triplanar,
                scale: 2.0,
                rotation_degrees: 30.0,
                blend_sharpness: 3.0,
                ..TextureProjection::default()
            },
            ..LayerFill::default()
        },
    );
    base.channels.insert(MaterialChannel::Height);
    base.filters = vec![
        LayerFilter::Levels {
            in_low: 0.05,
            in_high: 0.95,
            gamma: 1.7,
            out_low: 0.0,
            out_high: 1.0,
        },
        LayerFilter::HueSaturation {
            hue: 0.1,
            saturation: 1.3,
            lightness: 0.0,
        },
    ];

    let mut worn = MaterialLayer::new_fill(
        LayerId(2),
        "Worn edges",
        LayerFill {
            color: [210, 200, 190, 255],
            roughness: 0.25,
            metallic: 1.0,
            ..LayerFill::default()
        },
    );
    worn.mask_generator = Some(MaskGenerator::EdgeWear(EdgeWear::DEFAULT));

    let mut grime = MaterialLayer::new_fill(
        LayerId(3),
        "Grime",
        LayerFill {
            color: [60, 48, 36, 255],
            roughness: 0.9,
            ..LayerFill::default()
        },
    );
    grime.mask_generator = Some(MaskGenerator::Dirt(Dirt::DEFAULT));
    grime.opacity = 0.8;

    let mut dust = MaterialLayer::new_fill(
        LayerId(4),
        "Dust on top",
        LayerFill {
            color: [190, 180, 160, 255],
            textures: substancecraft_core::FillTextures {
                base_color: Some(tile),
                ..Default::default()
            },
            projection: TextureProjection {
                mode: ProjectionMode::Spherical,
                ..TextureProjection::default()
            },
            ..LayerFill::default()
        },
    );
    dust.mask_generator = Some(MaskGenerator::Orientation(Orientation::DEFAULT));

    let mut graph = MaterialLayer::new_fill(LayerId(5), "Graph roughness", LayerFill::default());
    graph.channels = [MaterialChannel::Roughness].into();
    graph.graph = Some(MaterialGraph {
        nodes: vec![
            GraphNode::Uv,
            GraphNode::Math {
                op: MathOp::Power,
                a: GraphInput::Node(0),
                b: GraphInput::gray(2.2),
            },
        ],
        outputs: BTreeMap::from([(MaterialChannel::Roughness, GraphInput::Node(1))]),
    });
    graph.opacity = 0.5;

    let mut strokes = MaterialLayer::new_paint(LayerId(6), "Scratches");
    strokes.channels.insert(MaterialChannel::Height);
    let points: Vec<Vec2> = (0..24)
        .map(|step| {
            let t = step as f32 / 23.0;
            Vec2::new(0.1 + 0.8 * t, 0.3 + 0.4 * t * t)
        })
        .collect();
    strokes.paint_strokes.push(PaintStroke {
        tip: BrushTip {
            shape: BrushShape::Streak,
            follow_path: true,
            angle_jitter: 0.5,
            size_jitter: 0.3,
            seed: 11,
            ..BrushTip::ROUND
        },
        height: 0.2,
        ..PaintStroke::new(points, 3.0, [30, 25, 20, 255])
    });

    let mut project = MaterialProject {
        mesh: Some(MeshAsset {
            id: MeshId(1),
            source: mesh.to_owned(),
            scale: 1.0,
            smoothing: MeshSmoothing::default(),
            high_poly_source: None,
            content_hash: None,
            high_poly_hash: None,
            high_to_low: Default::default(),
        }),
        texture_sets: vec![TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Probe".to_owned(),
            material_slot: "Default".to_owned(),
            resolution: TextureResolution {
                width: 64,
                height: 64,
            },
            channels: [
                MaterialChannel::BaseColor,
                MaterialChannel::Roughness,
                MaterialChannel::Metallic,
                MaterialChannel::Height,
                MaterialChannel::Normal,
            ]
            .into(),
            layers: vec![base, worn, grime, dust, graph, strokes],
            height_strength: 1.0,
            export: None,
        }],
        active_texture_set: Some(TextureSetId(1)),
        ..MaterialProject::empty()
    };
    // Written on Windows: a relative path with backslashes.
    project.resources.push(ProjectResource {
        color_space: ColorSpace::LinearSrgb,
        ..ProjectResource::new(tile, "Tile", "textures\\tile.png")
    });
    project.validate().unwrap();
    project
}

/// `name digest` per exported file, in export order.
fn export_digests(folder: &Path) -> Vec<String> {
    write_tile(&folder.join("textures").join("tile.png"));
    write_corner(&folder.join("meshes").join("corner.gltf"));

    let mut lines = Vec::new();
    for (mesh, preset, format) in [
        ("builtin:Sphere", None, TextureFormat::Png),
        (r"meshes\corner.gltf", None, TextureFormat::Png),
        (
            r"meshes\corner.gltf",
            Some("Separate channels"),
            TextureFormat::Exr,
        ),
    ] {
        let project_path = folder.join("probe.sproject");
        save_project_file(&project_path, &project(mesh)).unwrap();
        let output = folder.join(format!("{}-{format:?}", lines.len()));
        let mut job = ExportJob::new(&project_path, output);
        job.preset = preset.map(str::to_owned);
        job.format = Some(format);
        job.mesh_maps.ao_samples = 8;
        for file in run_export_job(&job).unwrap() {
            let name = file.file_name().unwrap().to_string_lossy().into_owned();
            let digest = file_content_hash(&file).unwrap().fnv1a;
            let mesh = mesh.rsplit(['\\', ':']).next().unwrap_or(mesh);
            lines.push(format!("{mesh} {name} {digest}"));
        }
    }
    // Exports quantize to 8 or 16 bits, which can hide a last-bit
    // difference; the mesh maps' 32-bit floats show every one.
    let corner = substancecraft_api::load_mesh_file(&folder.join("meshes").join("corner.gltf"))
        .unwrap()
        .remove(0)
        .geometry;
    for (mesh, geometry) in [
        (
            "Sphere",
            substancecraft_core::PrimitivePreset::Sphere.build_mesh(),
        ),
        ("corner.gltf", corner),
    ] {
        let settings = substancecraft_core::MeshMapSettings {
            resolution: 64,
            ao_samples: 8,
            ..Default::default()
        };
        let maps =
            substancecraft_api::bake_mesh_maps(&geometry, &MeshSmoothing::default(), &settings)
                .unwrap();
        let bits = substancecraft_core::encode_mesh_maps(&[maps]);
        let digest = substancecraft_core::ContentHash::of(&bits).fnv1a;
        lines.push(format!("{mesh} mesh-maps.f32 {digest}"));
    }
    lines
}

#[test]
fn tc_0408_exports_are_byte_identical_across_platforms() {
    let folder = tempfile::tempdir().unwrap();
    let digests = export_digests(folder.path());
    assert!(digests.len() >= 14, "{digests:?}");

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DIGESTS);
    if std::env::var_os("SUBSTANCECRAFT_BLESS").is_some() {
        let text = format!(
            "# PROD-08 export digests (FNV-1a 64) of tests/determinism.rs.\n\
             # Regenerate with SUBSTANCECRAFT_BLESS=1 after an intentional change.\n{}\n",
            digests.join("\n")
        );
        std::fs::write(&fixture, text).unwrap();
        return;
    }
    let text = std::fs::read_to_string(&fixture)
        .unwrap_or_else(|error| panic!("{}: {error}", fixture.display()));
    let expected: Vec<&str> = text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .collect();
    assert_eq!(
        digests,
        expected,
        "export bytes differ from {DIGESTS} on {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
}
