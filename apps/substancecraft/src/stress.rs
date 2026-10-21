//! A generated stress-test project (PERF-05): many texture sets with deep
//! layer stacks and long strokes, a bitmap per set, a multi-material mesh,
//! and a dense high-poly source, written to a folder like a real project.

use std::path::{Path, PathBuf};

use substancecraft_api::{save_project_file, write_gltf, write_texture};
use substancecraft_core::{
    BakedTexture, LayerFill, LayerId, MaterialLayer, MaterialProject, MeshAsset, MeshGeometry,
    MeshId, MeshVertex, PaintStroke, ProjectResource, ResourceId, TextureResolution, TextureSet,
    TextureSetId,
};

/// How big a stress project is.
#[derive(Clone, Copy, Debug)]
pub struct StressScale {
    pub sets: u32,
    pub layers_per_set: u32,
    pub strokes_per_paint_layer: u32,
    pub points_per_stroke: u32,
    pub bitmap_edge: u32,
    /// Quads per side of the high-poly sphere (two triangles each).
    pub high_poly_quads: u32,
}

impl StressScale {
    /// Large enough to show where time goes; for benchmarks.
    pub const HEAVY: Self = Self {
        sets: 24,
        layers_per_set: 40,
        strokes_per_paint_layer: 40,
        points_per_stroke: 50,
        bitmap_edge: 2048,
        high_poly_quads: 700,
    };

    /// The same shape at a size tests can afford.
    pub const LIGHT: Self = Self {
        sets: 8,
        layers_per_set: 12,
        strokes_per_paint_layer: 6,
        points_per_stroke: 20,
        bitmap_edge: 64,
        high_poly_quads: 40,
    };
}

/// A grid of `columns` by `rows` quads, `sets` material slots in bands.
fn banded_grid(columns: u32, rows: u32, sets: u32) -> MeshGeometry {
    let mut vertices = Vec::new();
    for row in 0..=rows {
        for column in 0..=columns {
            let (u, v) = (column as f32 / columns as f32, row as f32 / rows as f32);
            vertices.push(MeshVertex {
                position: glam::Vec3::new(u - 0.5, v - 0.5, 0.0),
                uv: glam::Vec2::new(u, v),
            });
        }
    }
    let (mut triangles, mut triangle_materials) = (Vec::new(), Vec::new());
    for row in 0..rows {
        for column in 0..columns {
            let a = row * (columns + 1) + column;
            let (b, c, d) = (a + 1, a + columns + 1, a + columns + 2);
            let slot = column * sets / columns;
            triangles.extend([[a, b, c], [b, d, c]]);
            triangle_materials.extend([slot, slot]);
        }
    }
    MeshGeometry {
        vertices,
        triangles,
        triangle_materials,
        material_slots: (0..sets).map(|set| format!("Set{set}")).collect(),
    }
}

/// A UV sphere of `quads` by `quads / 2` quads.
fn sphere(quads: u32) -> MeshGeometry {
    let (columns, rows) = (quads, (quads / 2).max(2));
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
                ) * 0.5,
                uv: glam::Vec2::new(u, v),
            });
        }
    }
    let mut triangles = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let a = row * (columns + 1) + column;
            let (b, c, d) = (a + 1, a + columns + 1, a + columns + 2);
            triangles.extend([[a, c, b], [b, c, d]]);
        }
    }
    MeshGeometry {
        vertices,
        triangles,
        ..MeshGeometry::default()
    }
}

/// Writes a stress project and its files into `folder`; returns the
/// project file.
pub fn write_stress_project(folder: &Path, scale: StressScale) -> Result<PathBuf, String> {
    let mesh_path = folder.join("model.gltf");
    let low = banded_grid(scale.sets * 4, 32, scale.sets);
    write_gltf(&mesh_path, &low, 45.0, &[]).map_err(|error| error.to_string())?;
    let high_path = folder.join("model_high.gltf");
    write_gltf(&high_path, &sphere(scale.high_poly_quads), 45.0, &[])
        .map_err(|error| error.to_string())?;

    let mut project = MaterialProject {
        mesh: Some(MeshAsset {
            id: MeshId(1),
            source: mesh_path.display().to_string(),
            scale: 1.0,
            smoothing: Default::default(),
            high_poly_source: Some(high_path.display().to_string()),
            content_hash: None,
            high_poly_hash: None,
            high_to_low: Default::default(),
        }),
        texture_sets: Vec::new(),
        ..MaterialProject::default()
    };
    let mut next_layer = 1;
    for set in 0..scale.sets {
        // Noise, so PNG decoding costs what a real bitmap's would.
        let edge = scale.bitmap_edge;
        let mut seed = 0x9e37_79b9_u32.wrapping_mul(set + 1);
        let rgba8 = (0..edge * edge * 4)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as u8
            })
            .collect();
        let bitmap = folder.join(format!("bitmap{set}.png"));
        write_texture(
            &bitmap,
            &BakedTexture {
                width: edge,
                height: edge,
                rgba8,
            },
        )
        .map_err(|error| error.to_string())?;
        let resource = ResourceId(u64::from(set) + 1);
        project.resources.push(ProjectResource::new(
            resource,
            format!("bitmap{set}"),
            bitmap.display().to_string(),
        ));

        let mut layers = Vec::new();
        for layer in 0..scale.layers_per_set {
            let id = LayerId(next_layer);
            next_layer += 1;
            if layer % 4 == 0 {
                let mut fill = LayerFill::default();
                fill.textures.base_color = Some(resource);
                layers.push(MaterialLayer::new_fill(id, format!("Fill {layer}"), fill));
                continue;
            }
            let mut paint = MaterialLayer::new_paint(id, format!("Paint {layer}"));
            for stroke in 0..scale.strokes_per_paint_layer {
                let start = glam::Vec2::new(
                    (stroke * 37 % 97) as f32 / 97.0,
                    (stroke * 53 % 89) as f32 / 89.0,
                );
                let points = (0..scale.points_per_stroke)
                    .map(|point| start + glam::Vec2::new(point as f32 * 0.002, 0.0))
                    .collect();
                paint
                    .paint_strokes
                    .push(PaintStroke::new(points, 6.0, [200, 80, 40, 255]));
            }
            layers.push(paint);
        }
        project.texture_sets.push(TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(u64::from(set) + 1),
            name: format!("Set{set}"),
            material_slot: format!("Set{set}"),
            resolution: TextureResolution {
                width: 256,
                height: 256,
            },
            channels: Default::default(),
            layers,
            height_strength: 1.0,
            export: None,
        });
    }
    project.active_texture_set = Some(TextureSetId(1));
    let path = folder.join("stress.sproject");
    save_project_file(&path, &project).map_err(|error| error.to_string())?;
    Ok(path)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::SubstanceCraftApp;
    use std::time::Instant;

    fn timed<T>(label: &str, run: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let value = run();
        eprintln!(
            "{label:>28}: {:7.1} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
        value
    }

    /// Asserts two projects are equal, naming the first line that differs.
    fn assert_same(left: &MaterialProject, right: &MaterialProject, what: &str) {
        let (a, b) = (
            substancecraft_api::encode_project_json(left).unwrap(),
            substancecraft_api::encode_project_json(right).unwrap(),
        );
        let (a, b) = (String::from_utf8_lossy(&a), String::from_utf8_lossy(&b));
        let first = a.lines().zip(b.lines()).position(|(x, y)| x != y);
        assert!(
            left == right,
            "{what}: first difference at line {first:?}: {:?} vs {:?}",
            first.and_then(|line| a.lines().nth(line)),
            first.and_then(|line| b.lines().nth(line))
        );
    }

    #[test]
    fn tc_0406_stress_projects_open_save_and_undo_in_the_background_safely() {
        let folder = tempfile::tempdir().unwrap();
        let scale = StressScale::LIGHT;
        let path = write_stress_project(folder.path(), scale).unwrap();
        let written = std::fs::read(&path).unwrap();

        // A current file opens through the direct path with a clean report.
        let (project, report) = substancecraft_api::load_project_file_with_report(&path).unwrap();
        assert!(report.is_clean(), "{report:?}");
        // Opening reads and decodes on a background thread; the open
        // project stays until the new one is taken in.
        let context = eframe::egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        let untitled = app.project.clone();
        app.open_project_in_background(path.clone());
        assert_eq!(app.project, untitled);
        assert!(app.project_error.as_deref().unwrap().starts_with("Opening"));
        while app.opening.is_some() {
            app.poll_opening(&context);
            std::thread::yield_now();
        }
        assert_eq!(app.project_error, None);
        assert_eq!(app.project_path.as_deref(), Some(path.as_path()));
        assert_same(&app.project, &project, "opened");
        assert_eq!(
            app.resource_images.len(),
            scale.sets as usize,
            "every bitmap"
        );

        // Saving returns at once; edits made meanwhile are not in the file.
        let copy = folder.path().join("copy.sproject");
        app.start_save(copy.clone());
        assert!(app.saving.busy());
        let saved = app.project.clone();
        app.add_paint_layer();
        app.paint_mode = crate::PaintMode::Paint;
        app.append_stroke_point(glam::Vec2::splat(0.5));
        app.finish_active_stroke();
        while app.saving.busy() {
            app.poll_saves(&context);
            std::thread::yield_now();
        }
        assert_eq!(app.project_error, None);
        assert_eq!(app.project_path.as_deref(), Some(copy.as_path()));
        assert_eq!(std::fs::read(&copy).unwrap(), written, "byte for byte");
        assert_eq!(substancecraft_api::load_project_file(&copy).unwrap(), saved);
        assert_ne!(app.project, saved, "the edit came after the snapshot");

        // Undo and redo work on the large project as on a small one.
        let edited = app.project.clone();
        app.undo();
        app.undo();
        assert_same(&app.project, &saved, "the stroke and its layer undone");
        app.redo();
        app.redo();
        assert_same(&app.project, &edited, "both redone");

        // A save that ends after another project opened leaves it alone.
        app.start_save(folder.path().join("late.sproject"));
        app.open_project_path(path.clone());
        app.finish_saves();
        assert_eq!(app.project_path.as_deref(), Some(path.as_path()));
        assert_same(&app.project, &project, "reopened");
    }

    /// Where opening, saving, and undoing a heavy project spend time; run
    /// with `--ignored --nocapture` in a release build.
    #[test]
    #[ignore = "benchmark"]
    fn stress_project_phases() {
        let folder = tempfile::tempdir().unwrap();
        let path = timed("write fixture", || {
            write_stress_project(folder.path(), StressScale::HEAVY).unwrap()
        });
        eprintln!(
            "{:>28}: {:7.1} MB",
            "project file",
            std::fs::metadata(&path).unwrap().len() as f64 / 1e6
        );
        let (project, _) = timed("parse + migrate", || {
            substancecraft_api::load_project_file_with_report(&path).unwrap()
        });
        let bytes = std::fs::read(&path).unwrap();
        timed("  from_slice -> project", || {
            serde_json::from_slice::<MaterialProject>(&bytes).unwrap()
        });
        let value = timed("  from_slice -> Value", || {
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
        });
        timed("  Value clone", || value.clone());
        timed("  from_value -> project", || {
            serde_json::from_value::<MaterialProject>(value.clone()).unwrap()
        });
        timed("  to_value", || serde_json::to_value(&project).unwrap());
        timed("  to_vec compact", || serde_json::to_vec(&project).unwrap());
        timed("  validate", || project.validate().unwrap());
        timed("  encode (stable)", || {
            substancecraft_api::encode_project_json(&project).unwrap()
        });
        let mut app = SubstanceCraftApp::default();
        timed("restore mesh", || {
            SubstanceCraftApp::restore_project_mesh(&project).unwrap()
        });
        app.project = project.clone();
        timed("restore high poly", || app.restore_high_poly());
        timed("decode bitmaps", || app.reload_resources());

        let mut app = SubstanceCraftApp::default();
        timed("open (total, in place)", || {
            app.open_project_path(path.clone())
        });
        assert_eq!(app.project_error, None);
        // In the app the read runs on a background thread; only taking
        // the result in runs on the UI thread.
        let mut app = SubstanceCraftApp::default();
        let loaded = timed("open (background read)", || crate::load_for_opening(&path));
        timed("open (UI thread)", || {
            app.finish_opening(path.clone(), loaded)
        });
        assert_eq!(app.project_error, None);
        timed("save (encode + write + sync)", || {
            save_project_file(&path, &app.project).unwrap()
        });
        timed("save (UI thread)", || app.start_save(path.clone()));
        timed("save (until on disk)", || app.finish_saves());

        // Undo and redo of a stroke and of a whole texture set.
        app.add_paint_layer();
        app.paint_mode = crate::PaintMode::Paint;
        app.append_stroke_point(glam::Vec2::splat(0.5));
        app.finish_active_stroke();
        timed("undo stroke", || app.undo());
        timed("redo stroke", || app.redo());
        let set = Box::new(app.project.texture_sets[1].clone());
        timed("remove texture set", || {
            let removed = substancecraft_core::EditCommand::RemoveTextureSet { index: 1, set };
            assert!(app.history.apply(&mut app.project, removed));
            app.after_history_change(None);
        });
        timed("undo remove set", || app.undo());
    }
}
