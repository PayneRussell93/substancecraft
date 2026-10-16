//! GPU occlusion tracing against the CPU tracer (PERF-01). Tests that need
//! a GPU are ignored by default; run them with `--ignored`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::AtomicBool;

use glam::{Vec2, Vec3};
use substancecraft_core::{
    CpuTracer, MeshGeometry, MeshMapError, MeshMapKind, MeshMapSettings, MeshMaps, MeshVertex,
    PrimitivePreset, bake_mesh_maps_caged, bake_mesh_maps_traced,
};
use substancecraft_gpu::{FallbackTracer, GpuTracer};

/// A sphere resting in the corner of a floor and a wall, each in its own
/// third of UV space, so occlusion and thickness vary everywhere.
fn scene() -> MeshGeometry {
    let mut mesh = MeshGeometry::default();
    let mut add = |part: &MeshGeometry, offset: Vec3, scale: f32, uv_offset: f32| {
        let base = mesh.vertices.len() as u32;
        for vertex in &part.vertices {
            mesh.vertices.push(MeshVertex {
                position: vertex.position * scale + offset,
                uv: Vec2::new(vertex.uv.x / 3.0 + uv_offset, vertex.uv.y),
            });
        }
        mesh.triangles.extend(
            part.triangles
                .iter()
                .map(|triangle| triangle.map(|index| index + base)),
        );
    };
    let sphere = PrimitivePreset::Sphere.build_mesh();
    let floor = PrimitivePreset::Plane.build_mesh();
    let mut wall = floor.clone();
    for vertex in &mut wall.vertices {
        // Stand the floor up along the back edge.
        vertex.position = Vec3::new(vertex.position.x, vertex.position.z + 0.5, -0.5);
    }
    add(&sphere, Vec3::new(0.0, 0.3, -0.2), 0.6, 0.0);
    add(&floor, Vec3::ZERO, 1.0, 1.0 / 3.0);
    add(&wall, Vec3::ZERO, 1.0, 2.0 / 3.0);
    mesh
}

fn settings(resolution: u32, samples: u32) -> MeshMapSettings {
    MeshMapSettings {
        resolution,
        ao_samples: samples,
        ..MeshMapSettings::default()
    }
}

fn cpu_bake(mesh: &MeshGeometry, settings: &MeshMapSettings) -> MeshMaps {
    bake_mesh_maps_caged(mesh, None, settings, None, None, None).unwrap()
}

/// Mean and largest difference of one map, in 0–1, over texels both show.
fn difference(left: &MeshMaps, right: &MeshMaps, kind: MeshMapKind) -> (f32, f32) {
    let (left, right) = (left.to_texture(kind), right.to_texture(kind));
    let mut total = 0.0;
    let mut largest = 0.0_f32;
    let mut count = 0;
    for (a, b) in left.rgba8.chunks(4).zip(right.rgba8.chunks(4)) {
        if a[3] == 0 || b[3] == 0 {
            continue;
        }
        let delta = f32::from(a[0].abs_diff(b[0])) / 255.0;
        total += delta;
        largest = largest.max(delta);
        count += 1;
    }
    (total / count.max(1) as f32, largest)
}

#[test]
fn tc_0403_without_a_gpu_the_fallback_tracer_bakes_exactly_like_the_cpu() {
    let mesh = scene();
    let settings = settings(64, 16);
    let tracer = FallbackTracer::new(None);
    let maps = bake_mesh_maps_traced(&mesh, None, &settings, None, None, &tracer, None).unwrap();
    assert!(maps == cpu_bake(&mesh, &settings));
    assert!(!tracer.traced_on_gpu());
    let explicit = bake_mesh_maps_traced(&mesh, None, &settings, None, None, &CpuTracer, None);
    assert!(explicit.unwrap() == maps);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn tc_0404_gpu_occlusion_and_thickness_match_the_cpu_within_tolerance() {
    let gpu = GpuTracer::new_allowing_software().expect("a GPU adapter");
    eprintln!("tracing on {}", gpu.adapter_name());
    let mesh = scene();
    for (resolution, samples) in [(128, 32), (96, 7)] {
        let settings = settings(resolution, samples);
        let cpu = cpu_bake(&mesh, &settings);
        let tracer = FallbackTracer::new(Some(&gpu));
        let traced =
            bake_mesh_maps_traced(&mesh, None, &settings, None, None, &tracer, None).unwrap();
        assert!(tracer.traced_on_gpu(), "{:?}", tracer.fallback_reason());
        assert_eq!(tracer.fallback_reason(), None);
        // Maps from the surface alone are the CPU's own.
        for kind in [
            MeshMapKind::Normal,
            MeshMapKind::WorldNormal,
            MeshMapKind::Position,
            MeshMapKind::Curvature,
        ] {
            assert_eq!(traced.to_texture(kind), cpu.to_texture(kind), "{kind:?}");
        }
        // Rays follow the same directions, so only rays grazing an edge can
        // differ: a sample's worth here and there, never a trend.
        let one_sample = 1.0 / samples as f32;
        let (mean, largest) = difference(&cpu, &traced, MeshMapKind::AmbientOcclusion);
        eprintln!("AO at {resolution}², {samples} samples: mean {mean:.5}, largest {largest:.3}");
        assert!(mean < 0.004, "AO mean difference {mean}");
        assert!(
            largest <= 2.0 * one_sample + 0.01,
            "AO largest difference {largest}"
        );
        let (mean, largest) = difference(&cpu, &traced, MeshMapKind::Thickness);
        eprintln!("thickness: mean {mean:.5}, largest {largest:.3}");
        assert!(mean < 0.004, "thickness mean difference {mean}");
        assert!(largest < 0.25, "thickness largest difference {largest}");
        // Occlusion is really there: the corner darkens.
        let ao = traced.to_texture(MeshMapKind::AmbientOcclusion);
        assert!(
            ao.rgba8
                .chunks(4)
                .any(|texel| texel[3] > 0 && texel[0] < 150)
        );
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn tc_0404_gpu_bakes_cancel_and_trace_empty_scenes() {
    let gpu = GpuTracer::new_allowing_software().expect("a GPU adapter");
    let mesh = scene();
    let cancelled = AtomicBool::new(true);
    let result = bake_mesh_maps_traced(
        &mesh,
        None,
        &settings(64, 8),
        None,
        None,
        &gpu,
        Some(&cancelled),
    );
    assert_eq!(result.err(), Some(MeshMapError::Cancelled));
    // The same tracer bakes another mesh after (its scene is replaced).
    let settings = settings(48, 8);
    let cube = PrimitivePreset::Cube.build_mesh();
    let traced = bake_mesh_maps_traced(&cube, None, &settings, None, None, &gpu, None).unwrap();
    let (mean, _) = difference(
        &cpu_bake(&cube, &settings),
        &traced,
        MeshMapKind::AmbientOcclusion,
    );
    assert!(mean < 0.004);
}

#[test]
#[ignore = "timing report; needs a GPU adapter"]
fn gpu_bake_timing() {
    let gpu = GpuTracer::new().expect("a hardware GPU");
    // About 15,000 triangles: a production mesh has more, which favours the GPU.
    let mesh = substancecraft_core::subdivide(&scene(), 3, 180.0).unwrap();
    let settings = settings(1024, 64);
    let start = std::time::Instant::now();
    cpu_bake(&mesh, &settings);
    let cpu = start.elapsed();
    let start = std::time::Instant::now();
    bake_mesh_maps_traced(&mesh, None, &settings, None, None, &gpu, None).unwrap();
    let traced = start.elapsed();
    eprintln!(
        "1024², 64 samples, {} triangles: CPU {cpu:.2?}, GPU {traced:.2?} on {}",
        mesh.triangles.len(),
        gpu.adapter_name()
    );
}

struct NoRays;

impl substancecraft_core::OcclusionTracer for NoRays {
    fn trace(
        &self,
        _: &substancecraft_core::occlusion::Bvh,
        _: &substancecraft_core::OcclusionParams,
        queries: &[substancecraft_core::OcclusionQuery],
        _: Option<&AtomicBool>,
    ) -> Result<Vec<[f32; 2]>, MeshMapError> {
        Ok(vec![[1.0, 1.0]; queries.len()])
    }
}

#[test]
#[ignore = "timing report"]
fn surface_stage_timing() {
    let mesh = substancecraft_core::subdivide(&scene(), 3, 180.0).unwrap();
    let settings = settings(1024, 64);
    let start = std::time::Instant::now();
    bake_mesh_maps_traced(&mesh, None, &settings, None, None, &NoRays, None).unwrap();
    eprintln!("surface stage alone: {:.2?}", start.elapsed());
}

#[test]
#[ignore = "timing report; needs a GPU adapter"]
fn gpu_trace_only_timing() {
    use substancecraft_core::occlusion::{Bvh, trace_query};
    use substancecraft_core::{OcclusionParams, OcclusionQuery};
    let gpu = GpuTracer::new().expect("a hardware GPU");
    let mesh = substancecraft_core::subdivide(&scene(), 3, 180.0).unwrap();
    let bvh = Bvh::build(&mesh);
    let params = OcclusionParams::new(&settings(1024, 64), 2.0);
    let queries = (0..200_000_u32)
        .map(|index| {
            let vertex = mesh.vertices[index as usize % mesh.vertices.len()];
            OcclusionQuery {
                point: vertex.position,
                normal: Vec3::Y,
                face_normal: Vec3::Y,
                seed: index,
            }
        })
        .collect::<Vec<_>>();
    gpu.trace_gpu(&bvh, &params, &queries[..1000], None)
        .unwrap();
    let start = std::time::Instant::now();
    gpu.trace_gpu(&bvh, &params, &queries, None).unwrap();
    let elapsed = start.elapsed();
    let rays = queries.len() as f64 * f64::from(params.ao_samples + params.thickness_samples);
    eprintln!(
        "GPU: {elapsed:.2?} for {:.1} M rays ({:.0} M rays/s)",
        rays / 1e6,
        rays / 1e6 / elapsed.as_secs_f64()
    );
    let start = std::time::Instant::now();
    for query in &queries[..20_000] {
        std::hint::black_box(trace_query(&bvh, &params, query));
    }
    let elapsed = start.elapsed();
    eprintln!(
        "CPU, one thread: {:.0} M rays/s",
        rays / 10.0 / 1e6 / elapsed.as_secs_f64()
    );
}
