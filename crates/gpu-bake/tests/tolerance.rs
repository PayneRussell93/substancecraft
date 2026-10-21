//! GPU bakes against the CPU bake (PERF-01, TC-0404).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use substancecraft_core::{
    MeshGeometry, MeshMapKind, MeshMapSettings, MeshMaps, PrimitivePreset, TextureImage,
    TraceDevice, bake_mesh_maps_traced,
};
use substancecraft_gpu_bake::{AdapterChoice, GpuTracer};

fn scaled(mesh: &MeshGeometry, factor: f32) -> MeshGeometry {
    let mut mesh = mesh.clone();
    for vertex in &mut mesh.vertices {
        vertex.position *= factor;
    }
    mesh
}

/// Name, low mesh, high mesh, skew map, and cage of one bake.
type Case<'a> = (
    &'a str,
    &'a MeshGeometry,
    Option<&'a MeshGeometry>,
    Option<&'a TextureImage>,
    Option<&'a MeshGeometry>,
);

/// Per kind: the mean difference in 8-bit levels and the share of channel
/// values more than `slack` levels apart.
fn differences(gpu: &MeshMaps, cpu: &MeshMaps, kind: MeshMapKind, slack: u8) -> (f64, f64) {
    let (gpu, cpu) = (gpu.to_texture(kind), cpu.to_texture(kind));
    assert_eq!(gpu.rgba8.len(), cpu.rgba8.len());
    let mut total = 0_u64;
    let mut far = 0_usize;
    for (a, b) in gpu.rgba8.iter().zip(&cpu.rgba8) {
        let difference = a.abs_diff(*b);
        total += u64::from(difference);
        far += usize::from(difference > slack);
    }
    let count = gpu.rgba8.len() as f64;
    (total as f64 / count, far as f64 / count)
}

#[test]
fn tc_0404_gpu_bakes_match_the_cpu_within_tolerance() {
    let tracer = match GpuTracer::new(AdapterChoice::Any) {
        Ok(tracer) => tracer,
        Err(error) => {
            eprintln!("TC-0404 skipped: no GPU adapter ({error})");
            return;
        }
    };
    let cube = PrimitivePreset::Cube.build_mesh();
    let sphere = PrimitivePreset::Sphere.build_mesh();
    let cylinder = PrimitivePreset::Cylinder.build_mesh();
    let settings = MeshMapSettings {
        resolution: 128,
        ao_samples: 16,
        skew: 0.5,
        ..MeshMapSettings::default()
    };
    let skew = TextureImage {
        width: 4,
        height: 4,
        rgba8: (0..64).map(|value| (value * 4) as u8).collect(),
        rgba16: None,
        color_space: Default::default(),
    };
    let high = scaled(&sphere, 1.02);
    let inner = scaled(&sphere, 0.9);
    let (raised, cage) = (scaled(&sphere, 1.03), scaled(&sphere, 1.1));
    let cases: [Case<'_>; 5] = [
        ("cube", &cube, None, None, None),
        ("cylinder", &cylinder, None, None, None),
        ("high to low", &sphere, Some(&high), None, None),
        ("skew map", &cube, Some(&inner), Some(&skew), None),
        ("cage", &sphere, Some(&raised), None, Some(&cage)),
    ];
    // One AO sample flips a texel by 255 / samples levels.
    let sample_step = (255 / settings.ao_samples) as u8;
    for (name, low, high, skew_map, cage) in cases {
        let bake = |tracer: Option<&dyn substancecraft_core::RayTracer>| {
            bake_mesh_maps_traced(low, high, &settings, skew_map, cage, tracer, None).unwrap()
        };
        let cpu = bake(None);
        let gpu = bake(Some(&tracer));
        assert!(
            matches!(gpu.trace_device(), TraceDevice::Tracer(_)),
            "{name}: {:?}",
            gpu.trace_device()
        );
        for kind in [
            MeshMapKind::AmbientOcclusion,
            MeshMapKind::Thickness,
            MeshMapKind::Normal,
            MeshMapKind::WorldNormal,
            MeshMapKind::Position,
            MeshMapKind::Curvature,
        ] {
            let slack = if kind == MeshMapKind::AmbientOcclusion {
                sample_step + 1
            } else {
                2
            };
            let (mean, far) = differences(&gpu, &cpu, kind, slack);
            eprintln!(
                "{name} {kind:?}: mean {mean:.4} levels, {:.3}% beyond {slack}",
                far * 100.0
            );
            assert!(
                mean <= 0.5,
                "{name} {kind:?}: mean difference {mean:.3} levels"
            );
            assert!(
                far <= 0.005,
                "{name} {kind:?}: {:.2}% of values differ",
                far * 100.0
            );
        }
        assert!(
            (gpu.miss_fraction() - cpu.miss_fraction()).abs() <= 0.005,
            "{name}: misses {} vs {}",
            gpu.miss_fraction(),
            cpu.miss_fraction()
        );
    }
}

/// Timing of a heavy bake on each device; run with `--ignored --nocapture`
/// in a release build.
#[test]
#[ignore = "benchmark"]
fn gpu_bake_speed() {
    let Ok(tracer) = GpuTracer::new(AdapterChoice::Any) else {
        return;
    };
    let sphere = PrimitivePreset::Sphere.build_mesh();
    let high = dense_sphere(512, 256, 1.02);
    eprintln!("high mesh: {} triangles", high.triangles.len());
    let settings = MeshMapSettings {
        resolution: 1024,
        ao_samples: 64,
        ..MeshMapSettings::default()
    };
    for (label, device) in [
        ("cpu", None),
        ("gpu", Some(&tracer as &dyn substancecraft_core::RayTracer)),
    ] {
        let started = std::time::Instant::now();
        let maps = bake_mesh_maps_traced(&sphere, Some(&high), &settings, None, None, device, None)
            .unwrap();
        eprintln!(
            "{label}: {:.2} s on {:?}",
            started.elapsed().as_secs_f32(),
            maps.trace_device()
        );
    }
}

/// A UV sphere of `columns` by `rows` quads, `radius` units across.
fn dense_sphere(columns: u32, rows: u32, radius: f32) -> MeshGeometry {
    use substancecraft_core::MeshVertex;
    let mut vertices = Vec::new();
    for row in 0..=rows {
        for column in 0..=columns {
            let (u, v) = (column as f32 / columns as f32, row as f32 / rows as f32);
            let (theta, phi) = (u * std::f32::consts::TAU, v * std::f32::consts::PI);
            vertices.push(MeshVertex {
                position: glam::Vec3::new(
                    phi.sin() * theta.cos(),
                    phi.cos(),
                    phi.sin() * theta.sin(),
                ) * radius,
                uv: glam::Vec2::new(u, v),
            });
        }
    }
    let mut triangles = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let a = row * (columns + 1) + column;
            let (b, c, d) = (a + 1, a + columns + 1, a + columns + 2);
            triangles.push([a, c, b]);
            triangles.push([b, c, d]);
        }
    }
    MeshGeometry {
        vertices,
        triangles,
        ..MeshGeometry::default()
    }
}
