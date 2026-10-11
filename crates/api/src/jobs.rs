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
    ImportedMesh, MAX_FBX_BYTES, TextureFormat, bake_mesh_maps, bake_mesh_maps_high_to_low,
    import_binary_fbx_meshes, import_gltf_file, load_project_file, load_project_resources,
    write_exported_textures,
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

/// Reads an FBX (every mesh it holds) or a glTF/GLB (the flattened scene),
/// choosing the importer by extension.
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
    import_binary_fbx_meshes(&bytes).map_err(|error| error.to_string())
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

/// What to export.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportJob {
    pub project: PathBuf,
    pub output: PathBuf,
    /// Built-in or project preset name.
    pub preset: String,
    pub format: TextureFormat,
    /// Export every UDIM tile the mesh uses (`$udim` or `.1001` names).
    pub udim: bool,
    /// Mesh-map bake settings; the resolution follows each texture set.
    pub mesh_maps: MeshMapSettings,
}

impl ExportJob {
    pub fn new(project: impl Into<PathBuf>, output: impl Into<PathBuf>) -> Self {
        Self {
            project: project.into(),
            output: output.into(),
            preset: "Metal/Rough (ORM)".to_owned(),
            format: TextureFormat::Png,
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
    let preset = ExportPreset::built_in()
        .into_iter()
        .chain(project.export_presets.iter().cloned())
        .find(|preset| preset.name == job.preset)
        .ok_or_else(|| JobError::UnknownPreset(job.preset.clone()))?;
    std::fs::create_dir_all(&job.output).map_err(|error| JobError::Project(error.to_string()))?;

    let mut written = Vec::new();
    for set in &project.texture_sets {
        let fail = |message: String| JobError::Export {
            set: set.name.clone(),
            message,
        };
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
        let tiles: Vec<Option<substancecraft_core::UdimTile>> = match (&geometry, job.udim) {
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
                write_exported_textures(&job.output, &files, job.format)
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
            }),
            texture_sets: vec![TextureSet {
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

        job.preset = "Nope".to_owned();
        assert_eq!(
            run_export_job(&job),
            Err(JobError::UnknownPreset("Nope".to_owned()))
        );
        job.project = folder.path().join("missing.sproject");
        assert!(matches!(run_export_job(&job), Err(JobError::Project(_))));
    }
}
