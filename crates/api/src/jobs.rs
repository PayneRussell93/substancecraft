//! Headless, deterministic jobs for pipelines: open a project, restore its
//! mesh, bake mesh maps where the export needs them, and write every texture
//! set through an export preset. The same inputs always produce the same
//! files.

use std::path::{Path, PathBuf};

use substancecraft_core::{
    BakeContext, ExportPreset, MAX_MESH_MAP_RESOLUTION, MeshGeometry, MeshMapSettings, MeshMaps,
    MeshSmoothing, PreflightSeverity, PrimitivePreset, TextureSet,
};
use thiserror::Error;

use crate::{
    ImportedMesh, MAX_FBX_BYTES, STEP_EXTENSIONS, TextureFormat, bake_mesh_maps,
    bake_mesh_maps_high_to_low, import_fbx_meshes, import_gltf_file, import_step_file,
    load_project_file, load_project_resources, write_exported_textures,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum JobError {
    #[error("project: {0}")]
    Project(String),
    #[error("mesh: {0}")]
    Mesh(String),
    #[error("unknown export preset `{0}`")]
    UnknownPreset(String),
    #[error("texture set `{set}`: {message}")]
    Export { set: String, message: String },
}

/// Reads an FBX (every mesh it holds), a glTF/GLB (the flattened scene), or
/// a STEP file (the tessellated, flattened assembly), choosing the importer
/// by extension.
pub fn load_mesh_file(path: &Path) -> Result<Vec<ImportedMesh>, String> {
    let extension = path
        .extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if extension == "gltf" || extension == "glb" {
        return import_gltf_file(path)
            .map(|mesh| vec![mesh])
            .map_err(|error| error.to_string());
    }
    if STEP_EXTENSIONS.contains(&extension.as_str()) {
        return import_step_file(path)
            .map(|mesh| vec![mesh])
            .map_err(|error| error.to_string());
    }
    let size = std::fs::metadata(path)
        .map_err(|error| error.to_string())?
        .len();
    if size > MAX_FBX_BYTES as u64 {
        return Err(format!(
            "FBX file exceeds the {} MiB import limit",
            MAX_FBX_BYTES / (1024 * 1024)
        ));
    }
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    import_fbx_meshes(&bytes).map_err(|error| error.to_string())
}

fn resolve(source: &str, base: Option<&Path>) -> PathBuf {
    let path = PathBuf::from(source);
    match base {
        Some(base) if path.is_relative() => base.join(path),
        _ => path,
    }
}

fn scaled(mut mesh: MeshGeometry, scale: f32) -> MeshGeometry {
    if scale != 1.0 {
        for vertex in &mut mesh.vertices {
            vertex.position *= scale;
        }
    }
    mesh
}

/// The project's mesh as the app shows it: a built-in preset or the first
/// mesh of the referenced file, scaled. Relative paths resolve against
/// `base` (the project's folder).
pub fn load_project_mesh(
    project: &substancecraft_core::MaterialProject,
    base: Option<&Path>,
) -> Result<Option<(MeshGeometry, MeshSmoothing)>, String> {
    let Some(asset) = &project.mesh else {
        return Ok(None);
    };
    let smoothing = if asset.smoothing.is_valid() {
        asset.smoothing
    } else {
        MeshSmoothing::default()
    };
    if let Some(name) = asset.source.strip_prefix("builtin:") {
        let preset = PrimitivePreset::ALL
            .into_iter()
            .find(|preset| preset.name() == name)
            .ok_or_else(|| format!("unknown built-in mesh `{name}`"))?;
        return Ok(Some((preset.build_mesh(), smoothing)));
    }
    let meshes = load_mesh_file(&resolve(&asset.source, base))?;
    let first = meshes
        .into_iter()
        .next()
        .ok_or_else(|| "mesh file contains no geometry".to_owned())?;
    let scale = if asset.scale.is_finite() && asset.scale > 0.0 {
        asset.scale
    } else {
        1.0
    };
    Ok(Some((scaled(first.geometry, scale), smoothing)))
}

/// Preset used when neither the job nor a texture set names one.
pub const DEFAULT_EXPORT_PRESET_NAME: &str = "Metal/Rough (ORM)";

/// What to export.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportJob {
    pub project: PathBuf,
    pub output: PathBuf,
    /// Built-in or project preset name for every set; `None` uses each
    /// set's saved choice, else [`DEFAULT_EXPORT_PRESET_NAME`].
    pub preset: Option<String>,
    /// File format for every set; `None` uses each set's saved format, else
    /// PNG.
    pub format: Option<TextureFormat>,
    /// Export every UDIM tile the mesh uses (`$udim` or `.1001` names) for
    /// every set; sets that saved UDIM export do so regardless.
    pub udim: bool,
    /// Mesh-map bake settings; the resolution follows each texture set.
    pub mesh_maps: MeshMapSettings,
}

impl ExportJob {
    pub fn new(project: impl Into<PathBuf>, output: impl Into<PathBuf>) -> Self {
        Self {
            project: project.into(),
            output: output.into(),
            preset: None,
            format: None,
            udim: false,
            mesh_maps: MeshMapSettings::default(),
        }
    }
}

/// The part of `mesh` a texture set paints: its material slot, or the whole
/// mesh when slots do not name it.
fn set_geometry(mesh: &MeshGeometry, set: &TextureSet) -> MeshGeometry {
    match mesh
        .material_slots
        .iter()
        .position(|slot| *slot == set.material_slot)
    {
        Some(slot) if mesh.material_slots.len() > 1 => mesh.submesh(slot as u32),
        _ => mesh.clone(),
    }
}

/// Runs every job, up to `workers` at once, and returns one result per job
/// in the order given. A job that fails (or panics) does not stop the others.
pub fn run_export_jobs(jobs: &[ExportJob], workers: usize) -> Vec<Result<Vec<PathBuf>, JobError>> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = workers.clamp(1, jobs.len().max(1));
    let mut results: Vec<Option<Result<Vec<PathBuf>, JobError>>> =
        (0..jobs.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(job) = jobs.get(index) else {
                            break done;
                        };
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run_export_job(job)
                        }))
                        .unwrap_or_else(|_| Err(JobError::Project("the job crashed".to_owned())));
                        done.push((index, result));
                    }
                })
            })
            .collect();
        for handle in handles {
            if let Ok(done) = handle.join() {
                for (index, result) in done {
                    results[index] = Some(result);
                }
            }
        }
    });
    results
        .into_iter()
        .map(|result| {
            result.unwrap_or_else(|| Err(JobError::Project("the job did not run".to_owned())))
        })
        .collect()
}

/// Runs an export job and returns the files written, in a stable order.
pub fn run_export_job(job: &ExportJob) -> Result<Vec<PathBuf>, JobError> {
    let project =
        load_project_file(&job.project).map_err(|error| JobError::Project(error.to_string()))?;
    let base = job.project.parent();
    let (images, failures) = load_project_resources_from(&project, base);
    if let Some((id, error)) = failures.first() {
        return Err(JobError::Project(format!("resource {}: {error}", id.0)));
    }
    let mesh = load_project_mesh(&project, base).map_err(JobError::Mesh)?;
    let high = match project
        .mesh
        .as_ref()
        .and_then(|asset| asset.high_poly_source.as_ref())
    {
        Some(source) => {
            let scale = project.mesh.as_ref().map_or(1.0, |asset| asset.scale);
            let mut merged = MeshGeometry::default();
            for part in load_mesh_file(&resolve(source, base)).map_err(JobError::Mesh)? {
                merged
                    .append(&part.geometry)
                    .map_err(|_| JobError::Mesh("high-poly mesh too large".to_owned()))?;
            }
            Some(scaled(merged, scale))
        }
        None => None,
    };
    let presets: Vec<ExportPreset> = ExportPreset::built_in()
        .into_iter()
        .chain(project.export_presets.iter().cloned())
        .collect();
    let find_preset = |name: &str| {
        presets
            .iter()
            .find(|preset| preset.name == name)
            .cloned()
            .ok_or_else(|| JobError::UnknownPreset(name.to_owned()))
    };
    if let Some(name) = &job.preset {
        find_preset(name)?;
    }
    std::fs::create_dir_all(&job.output).map_err(|error| JobError::Project(error.to_string()))?;

    let mut written = Vec::new();
    for set in &project.texture_sets {
        let fail = |message: String| JobError::Export {
            set: set.name.clone(),
            message,
        };
        let saved = set.export.as_ref();
        let preset = find_preset(
            job.preset
                .as_deref()
                .or(saved.map(|export| export.preset.as_str()))
                .unwrap_or(DEFAULT_EXPORT_PRESET_NAME),
        )?;
        let format = job
            .format
            .or_else(|| saved.and_then(|export| TextureFormat::from_extension(&export.format)))
            .unwrap_or(TextureFormat::Png);
        // Outputs without their own format take the job's.
        let preset = preset.resolved(format.extension());
        let udim = job.udim || saved.is_some_and(|export| export.udim);
        let geometry = mesh.as_ref().map(|(mesh, _)| set_geometry(mesh, set));
        let settings = MeshMapSettings {
            resolution: set
                .resolution
                .width
                .max(set.resolution.height)
                .min(MAX_MESH_MAP_RESOLUTION),
            ..job.mesh_maps
        };
        let bake_maps = |geometry: &MeshGeometry| -> Result<MeshMaps, JobError> {
            let smoothing = mesh
                .as_ref()
                .map_or_else(MeshSmoothing::default, |(_, s)| *s);
            match &high {
                Some(high) => bake_mesh_maps_high_to_low(geometry, high, &smoothing, &settings),
                None => bake_mesh_maps(geometry, &smoothing, &settings),
            }
            .map_err(|error| fail(error.to_string()))
        };
        let tiles: Vec<Option<substancecraft_core::UdimTile>> = match (&geometry, udim) {
            (Some(geometry), true) => geometry.udim_tiles().into_iter().map(Some).collect(),
            _ => vec![None],
        };
        for tile in tiles {
            let tile_geometry = match (tile, &geometry) {
                (Some(tile), Some(geometry)) => Some(geometry.shifted_uvs(tile.offset())),
                (_, geometry) => geometry.clone(),
            };
            let maps = tile_geometry.as_ref().map(bake_maps).transpose()?;
            let context = BakeContext {
                mesh: tile_geometry.as_ref(),
                mesh_maps: maps.as_ref(),
                resources: Some(&images),
            };
            let shifted;
            let checked = match tile {
                Some(tile) => {
                    shifted = set.shifted_to_tile(tile);
                    &shifted
                }
                None => set,
            };
            if let Some(error) = checked
                .export_preflight(&preset, context)
                .into_iter()
                .find(|issue| issue.severity == PreflightSeverity::Error)
            {
                return Err(fail(error.message));
            }
            let files = match tile {
                Some(tile) => set.bake_export_tile(&preset, tile, context),
                None => set.bake_export(&preset, context),
            }
            .map_err(|error| fail(error.to_string()))?;
            written.extend(
                write_exported_textures(&job.output, &files, format)
                    .map_err(|error| fail(error.to_string()))?,
            );
        }
    }
    Ok(written)
}

/// [`load_project_resources`] with relative resource paths resolved
/// against `base`.
fn load_project_resources_from(
    project: &substancecraft_core::MaterialProject,
    base: Option<&Path>,
) -> (
    substancecraft_core::ResourceImages,
    Vec<(substancecraft_core::ResourceId, crate::TextureIoError)>,
) {
    let mut resolved = project.clone();
    for resource in &mut resolved.resources {
        resource.source = resolve(&resource.source, base)
            .to_string_lossy()
            .into_owned();
    }
    load_project_resources(&resolved)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::save_project_file;
    use substancecraft_core::{
        LayerFill, LayerId, MaterialChannel, MaterialLayer, MaterialProject, MeshAsset, MeshId,
        TextureResolution, TextureSetId,
    };

    fn project_with_edge_wear() -> MaterialProject {
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Worn",
            LayerFill {
                metallic: 1.0,
                ..LayerFill::default()
            },
        );
        layer.mask_generator = Some(substancecraft_core::MaskGenerator::EdgeWear(
            substancecraft_core::EdgeWear::DEFAULT,
        ));
        MaterialProject {
            mesh: Some(MeshAsset {
                id: MeshId(1),
                source: "builtin:Cube".to_owned(),
                scale: 1.0,
                smoothing: MeshSmoothing::default(),
                high_poly_source: None,
                content_hash: None,
                high_poly_hash: None,
            }),
            texture_sets: vec![TextureSet {
                user_channels: Default::default(),
                stencils: Default::default(),
                id: TextureSetId(1),
                name: "Crate".to_owned(),
                material_slot: "Default".to_owned(),
                resolution: TextureResolution {
                    width: 32,
                    height: 32,
                },
                channels: [MaterialChannel::BaseColor].into(),
                layers: vec![layer],
                height_strength: 1.0,
                export: None,
            }],
            active_texture_set: Some(TextureSetId(1)),
            ..MaterialProject::empty()
        }
    }

    #[test]
    fn tc_0200_headless_export_jobs_are_deterministic() {
        let folder = tempfile::tempdir().unwrap();
        let project_path = folder.path().join("crate.sproject");
        save_project_file(&project_path, &project_with_edge_wear()).unwrap();

        let mut job = ExportJob::new(&project_path, folder.path().join("first"));
        job.mesh_maps.ao_samples = 4;
        let first = run_export_job(&job).unwrap();
        let names: Vec<String> = first
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            ["Crate_BaseColor.png", "Crate_ORM.png", "Crate_Normal.png"]
        );
        // Edge wear needed mesh maps, which the job baked: edges are metal.
        let orm = crate::load_texture_image(&first[1]).unwrap();
        assert!(orm.rgba8.chunks(4).any(|texel| texel[2] > 200));
        assert!(orm.rgba8.chunks(4).any(|texel| texel[2] < 50));

        job.output = folder.path().join("second");
        let second = run_export_job(&job).unwrap();
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(
                std::fs::read(a).unwrap(),
                std::fs::read(b).unwrap(),
                "{a:?}"
            );
        }

        job.preset = Some("Nope".to_owned());
        assert_eq!(
            run_export_job(&job),
            Err(JobError::UnknownPreset("Nope".to_owned()))
        );
        job.project = folder.path().join("missing.sproject");
        assert!(matches!(run_export_job(&job), Err(JobError::Project(_))));
    }

    #[test]
    fn tc_0202_jobs_follow_each_texture_sets_saved_export_settings() {
        let folder = tempfile::tempdir().unwrap();
        let mut project = project_with_edge_wear();
        let mut trim = project.texture_sets[0].clone();
        trim.id = TextureSetId(2);
        trim.name = "Trim".to_owned();
        trim.layers[0].id = LayerId(2);
        trim.export = Some(substancecraft_core::TextureSetExport {
            preset: "Unreal Engine".to_owned(),
            format: "tga".to_owned(),
            udim: false,
        });
        project.texture_sets.push(trim);
        let path = folder.path().join("two.sproject");
        save_project_file(&path, &project).unwrap();

        let mut job = ExportJob::new(&path, folder.path().join("out"));
        job.mesh_maps.ao_samples = 2;
        let names = |files: Vec<PathBuf>| -> Vec<String> {
            files
                .iter()
                .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
                .collect()
        };
        let written = names(run_export_job(&job).unwrap());
        assert_eq!(
            written,
            [
                "Crate_BaseColor.png",
                "Crate_ORM.png",
                "Crate_Normal.png",
                "T_Trim_BC.tga",
                "T_Trim_ORM.tga",
                "T_Trim_N.tga",
            ]
        );

        // A job-wide preset and format override the saved choices.
        job.preset = Some("Separate channels".to_owned());
        job.format = Some(TextureFormat::Png);
        job.output = folder.path().join("override");
        let written = names(run_export_job(&job).unwrap());
        assert!(written.contains(&"Trim_Roughness.png".to_owned()));
        assert!(!written.iter().any(|name| name.ends_with(".tga")));

        // Saved settings must name a known preset and format.
        let mut broken = project.clone();
        if let Some(export) = &mut broken.texture_sets[1].export {
            export.format = "bmp".to_owned();
        }
        assert!(broken.validate().is_err());
    }

    #[test]
    fn tc_0232_job_batches_run_in_parallel_and_continue_past_failures() {
        let folder = tempfile::tempdir().unwrap();
        let project_path = folder.path().join("crate.sproject");
        save_project_file(&project_path, &project_with_edge_wear()).unwrap();
        let job = |out: &str| {
            let mut job = ExportJob::new(&project_path, folder.path().join(out));
            job.mesh_maps.ao_samples = 2;
            job
        };
        let mut missing = job("missing");
        missing.project = folder.path().join("missing.sproject");
        let jobs = [job("a"), missing, job("b")];

        let results = run_export_jobs(&jobs, 3);
        assert_eq!(results.len(), 3);
        assert!(matches!(results[1], Err(JobError::Project(_))));
        let (a, b) = (results[0].as_ref().unwrap(), results[2].as_ref().unwrap());
        assert_eq!(a.len(), 3);
        for (a, b) in a.iter().zip(b) {
            assert_eq!(std::fs::read(a).unwrap(), std::fs::read(b).unwrap());
        }
        // One worker gives the same results in the same order.
        let sequential = run_export_jobs(&jobs, 0);
        assert_eq!(sequential.iter().filter(|result| result.is_ok()).count(), 2);
        assert!(run_export_jobs(&[], 4).is_empty());
    }
}
