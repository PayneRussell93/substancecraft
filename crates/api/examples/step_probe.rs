//! Throwaway: imports STEP files and prints diagnostics.
use std::collections::HashMap;

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap_or_default();
        let start = std::time::Instant::now();
        let result = substancecraft_api::import_step_with_report(
            &bytes,
            "probe".to_owned(),
            &substancecraft_api::StepTessellation::default(),
        );
        let elapsed = start.elapsed();
        let name = std::path::Path::new(&path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        match result {
            Err(error) => println!("{name}: ERROR {error} ({elapsed:?})"),
            Ok((mesh, report)) => {
                let g = &mesh.geometry;
                let mut volume = 0.0f64;
                let mut min = glam::Vec3::splat(f32::MAX);
                let mut max = glam::Vec3::splat(f32::MIN);
                for v in &g.vertices {
                    min = min.min(v.position);
                    max = max.max(v.position);
                }
                let diag = (max - min).length() as f64;
                let q = diag * 1e-6;
                let key = |i: u32| {
                    let p = g.vertices[i as usize].position.as_dvec3() / q;
                    (p.x.round() as i64, p.y.round() as i64, p.z.round() as i64)
                };
                let mut edges: HashMap<_, i32> = HashMap::new();
                for t in &g.triangles {
                    let [a, b, c] = t.map(|i| g.vertices[i as usize].position.as_dvec3());
                    volume += a.dot(b.cross(c)) / 6.0;
                    for k in 0..3 {
                        let (a, b) = (key(t[k]), key(t[(k + 1) % 3]));
                        if a != b {
                            *edges.entry((a, b)).or_default() += 1;
                            *edges.entry((b, a)).or_default() -= 1;
                        }
                    }
                }
                let open = edges.values().filter(|c| **c != 0).count() / 2;
                println!(
                    "{name}: {elapsed:?} bodies {} faces {} skipped {} tris {} verts {} slots {:?} unit {} tol {:.2e} size {:.4} vol {:.3e} open-edges {}",
                    report.bodies,
                    report.faces,
                    report.skipped_faces,
                    g.triangles.len(),
                    g.vertices.len(),
                    g.material_slots,
                    report.length_unit,
                    report.tolerance,
                    diag,
                    volume,
                    open
                );
                if !report.unsupported.is_empty() {
                    println!("   unsupported {:?}", report.unsupported);
                }
                if !report.failures.is_empty() {
                    println!("   failures {:?}", report.failures);
                }
            }
        }
    }
}
