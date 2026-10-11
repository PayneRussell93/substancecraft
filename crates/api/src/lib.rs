#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

mod fbx;
mod gltf_import;
mod interchange;
mod jobs;
mod textures;
mod vector;

pub use fbx::{
    FbxImportError, ImportedFbxMesh, ImportedMesh, MAX_FBX_BYTES, import_binary_fbx_meshes,
};
pub use gltf_import::{GltfImportError, MAX_GLTF_BYTES, import_gltf, import_gltf_file};
pub use interchange::{
    GltfMaterialFiles, InterchangeError, OPENPBR_INPUTS, OpenPbrMaterial, openpbr_preset,
    read_openpbr_materialx, write_gltf, write_openpbr_materialx,
};
pub use jobs::{ExportJob, JobError, load_mesh_file, load_project_mesh, run_export_job};
pub use textures::{
    MAX_TEXTURE_FILE_BYTES, TextureFormat, TextureIoError, decode_texture_image,
    encode_exported_texture, encode_texture, load_texture_image, write_exported_textures,
    write_texture,
};
pub use vector::{
    DEFAULT_SVG_EDGE, TEXT_RESOURCE_EXTENSION, TextSpec, load_text_resource, rasterize_svg,
    render_text, save_text_resource,
};

use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
};
use substancecraft_core::SmartMaterial;
use substancecraft_core::{
    BakeContext, BakeError, BakedTexture, ExportPreset, ExportedTexture, MaterialChannel,
    MaterialProject, MaterialTextures, MeshError, MeshGeometry, MeshMapError, MeshMapSettings,
    MeshMaps, MeshSmoothing, PreflightIssue, PrimitivePreset, ProjectResource, ResourceId,
    ResourceImages, TextureImage, TextureSet,
};
use thiserror::Error;

pub const MAX_PROJECT_JSON_BYTES: usize = 128 * 1024 * 1024;

pub fn encode_project_json(project: &MaterialProject) -> Result<Vec<u8>, ProjectFileError> {
    project
        .validate()
        .map_err(|error| ProjectFileError::InvalidProject(error.to_string()))?;

    let mut writer = BoundedJsonWriter {
        bytes: Vec::new(),
        limit_exceeded: false,
    };
    if let Err(error) = serde_json::to_writer(&mut writer, project) {
        if writer.limit_exceeded {
            return Err(ProjectFileError::TooLarge(MAX_PROJECT_JSON_BYTES));
        }
        return Err(ProjectFileError::Encode(error.to_string()));
    }
    Ok(writer.bytes)
}

pub fn decode_project_json(bytes: &[u8]) -> Result<MaterialProject, ProjectFileError> {
    if bytes.len() > MAX_PROJECT_JSON_BYTES {
        return Err(ProjectFileError::TooLarge(bytes.len()));
    }
    let project: MaterialProject = serde_json::from_slice(bytes)
        .map_err(|error| ProjectFileError::Decode(error.to_string()))?;
    project
        .validate()
        .map_err(|error| ProjectFileError::InvalidProject(error.to_string()))?;
    Ok(project)
}

pub fn save_project_file(
    path: impl AsRef<Path>,
    project: &MaterialProject,
) -> Result<(), ProjectFileError> {
    let path = path.as_ref();
    let bytes = encode_project_json(project)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| ProjectFileError::Io(error.to_string()))?;
    temporary
        .write_all(&bytes)
        .map_err(|error| ProjectFileError::Io(error.to_string()))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| ProjectFileError::Io(error.to_string()))?;
    temporary
        .persist(path)
        .map_err(|error| ProjectFileError::Io(error.error.to_string()))?;
    Ok(())
}

pub fn load_project_file(path: impl AsRef<Path>) -> Result<MaterialProject, ProjectFileError> {
    let file = File::open(path).map_err(|error| ProjectFileError::Io(error.to_string()))?;
    let mut bytes = Vec::new();
    file.take((MAX_PROJECT_JSON_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| ProjectFileError::Io(error.to_string()))?;
    decode_project_json(&bytes)
}

struct BoundedJsonWriter {
    bytes: Vec<u8>,
    limit_exceeded: bool,
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(new_len) = self.bytes.len().checked_add(buffer.len()) else {
            self.limit_exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "project JSON size overflow",
            ));
        };
        if new_len > MAX_PROJECT_JSON_BYTES {
            self.limit_exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "project JSON exceeds size limit",
            ));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveDescriptor {
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveSummary {
    pub name: String,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub uv_bounds_min: [f32; 2],
    pub uv_bounds_max: [f32; 2],
}

pub fn list_primitive_presets() -> Vec<PrimitiveDescriptor> {
    PrimitivePreset::ALL
        .into_iter()
        .map(|preset| PrimitiveDescriptor {
            name: preset.name().to_owned(),
        })
        .collect()
}

pub fn inspect_primitive_preset(name: &str) -> Result<PrimitiveSummary, ApiError> {
    let preset = PrimitivePreset::ALL
        .into_iter()
        .find(|preset| preset.name().eq_ignore_ascii_case(name.trim()))
        .ok_or_else(|| ApiError::UnknownPrimitivePreset(name.to_owned()))?;
    let mesh = preset.build_mesh();
    mesh.validate()?;

    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    let mut uv_min = [f32::INFINITY; 2];
    let mut uv_max = [f32::NEG_INFINITY; 2];
    for vertex in &mesh.vertices {
        let position = vertex.position.to_array();
        let uv = vertex.uv.to_array();
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(position[axis]);
            bounds_max[axis] = bounds_max[axis].max(position[axis]);
        }
        for axis in 0..2 {
            uv_min[axis] = uv_min[axis].min(uv[axis]);
            uv_max[axis] = uv_max[axis].max(uv[axis]);
        }
    }

    Ok(PrimitiveSummary {
        name: preset.name().to_owned(),
        vertex_count: mesh.vertices.len(),
        triangle_count: mesh.triangles.len(),
        bounds_min,
        bounds_max,
        uv_bounds_min: uv_min,
        uv_bounds_max: uv_max,
    })
}

pub fn bake_texture_set_base_color(texture_set: &TextureSet) -> Result<BakedTexture, ApiError> {
    Ok(texture_set.bake_base_color()?)
}

pub fn bake_texture_set_id_mask(texture_set: &TextureSet) -> Result<BakedTexture, ApiError> {
    Ok(texture_set.bake_id_mask()?)
}

/// Bakes one channel of a texture set. `mesh` is required by polygon masks,
/// `mesh_maps` by mask generators and triplanar bitmaps, and `resources` by
/// bitmap fills, masks, and brush alphas; pass `None` when no layer uses them.
pub fn bake_texture_set_channel(
    texture_set: &TextureSet,
    channel: MaterialChannel,
    mesh: Option<&MeshGeometry>,
    mesh_maps: Option<&MeshMaps>,
    resources: Option<&ResourceImages>,
) -> Result<BakedTexture, ApiError> {
    Ok(texture_set.bake_channel(
        channel,
        BakeContext {
            mesh,
            mesh_maps,
            resources,
        },
    )?)
}

/// Applies smoothing/subdivision, then bakes geometry-derived mesh maps.
pub fn bake_mesh_maps(
    mesh: &MeshGeometry,
    smoothing: &MeshSmoothing,
    settings: &MeshMapSettings,
) -> Result<MeshMaps, ApiError> {
    let smoothed = smoothing.apply(mesh)?;
    let settings = MeshMapSettings {
        smoothing_angle_degrees: smoothing.angle_degrees,
        ..*settings
    };
    Ok(substancecraft_core::bake_mesh_maps(
        &smoothed, &settings, None,
    )?)
}

/// High-to-low bake: projects `high`'s surface detail into the smoothed low
/// mesh's UVs, producing a tangent-space normal map plus the other mesh maps
/// evaluated on the high surface.
pub fn bake_mesh_maps_high_to_low(
    low: &MeshGeometry,
    high: &MeshGeometry,
    smoothing: &MeshSmoothing,
    settings: &MeshMapSettings,
) -> Result<MeshMaps, ApiError> {
    let smoothed = smoothing.apply(low)?;
    let settings = MeshMapSettings {
        smoothing_angle_degrees: smoothing.angle_degrees,
        ..*settings
    };
    Ok(substancecraft_core::bake_mesh_maps_from(
        &smoothed,
        Some(high),
        &settings,
        None,
    )?)
}

/// [`bake_mesh_maps_high_to_low`] with a painted skew map scaling
/// [`MeshMapSettings::skew`] per texel.
pub fn bake_mesh_maps_high_to_low_corrected(
    low: &MeshGeometry,
    high: &MeshGeometry,
    smoothing: &MeshSmoothing,
    settings: &MeshMapSettings,
    skew_map: Option<&TextureImage>,
) -> Result<MeshMaps, ApiError> {
    let smoothed = smoothing.apply(low)?;
    let settings = MeshMapSettings {
        smoothing_angle_degrees: smoothing.angle_degrees,
        ..*settings
    };
    Ok(substancecraft_core::bake_mesh_maps_corrected(
        &smoothed,
        Some(high),
        &settings,
        skew_map,
        None,
    )?)
}

/// File format tag of saved smart materials.
pub const SMART_MATERIAL_FORMAT: &str = "substancecraft-smart-material";
const SMART_MATERIAL_VERSION: u32 = 1;
const MAX_SMART_MATERIAL_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct SmartMaterialFile {
    format: String,
    version: u32,
    material: SmartMaterial,
}

/// Writes a smart material as versioned JSON (`.scsmart`).
pub fn save_smart_material(path: &Path, material: &SmartMaterial) -> Result<(), ProjectFileError> {
    material
        .validate()
        .map_err(|error| ProjectFileError::InvalidProject(error.to_string()))?;
    let json = serde_json::to_vec_pretty(&SmartMaterialFile {
        format: SMART_MATERIAL_FORMAT.to_owned(),
        version: SMART_MATERIAL_VERSION,
        material: material.clone(),
    })
    .map_err(|error| ProjectFileError::Encode(error.to_string()))?;
    std::fs::write(path, json).map_err(|error| ProjectFileError::Io(error.to_string()))
}

/// Reads and validates a smart material file.
pub fn load_smart_material(path: &Path) -> Result<SmartMaterial, ProjectFileError> {
    let file = File::open(path).map_err(|error| ProjectFileError::Io(error.to_string()))?;
    let mut bytes = Vec::new();
    file.take(MAX_SMART_MATERIAL_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| ProjectFileError::Io(error.to_string()))?;
    if bytes.len() as u64 > MAX_SMART_MATERIAL_BYTES {
        return Err(ProjectFileError::TooLarge(bytes.len()));
    }
    let file: SmartMaterialFile = serde_json::from_slice(&bytes)
        .map_err(|error| ProjectFileError::Decode(error.to_string()))?;
    if file.format != SMART_MATERIAL_FORMAT || file.version != SMART_MATERIAL_VERSION {
        return Err(ProjectFileError::Decode(format!(
            "not a version {SMART_MATERIAL_VERSION} {SMART_MATERIAL_FORMAT} file"
        )));
    }
    file.material
        .validate()
        .map_err(|error| ProjectFileError::InvalidProject(error.to_string()))?;
    Ok(file.material)
}

/// Bakes the viewport/engine textures: Base Color (Opacity in alpha), packed
/// ORM (occlusion, roughness, metallic), the combined tangent-space normal,
/// and Emissive.
pub fn bake_texture_set_material(
    texture_set: &TextureSet,
    mesh: Option<&MeshGeometry>,
    mesh_maps: Option<&MeshMaps>,
    resources: Option<&ResourceImages>,
) -> Result<MaterialTextures, ApiError> {
    Ok(texture_set.bake_material(BakeContext {
        mesh,
        mesh_maps,
        resources,
    })?)
}

/// The built-in export presets, as serializable output descriptions.
pub fn list_export_presets() -> Vec<ExportPreset> {
    ExportPreset::built_in()
}

/// Bakes the files an export preset describes for one texture set. Takes the
/// same optional inputs as [`bake_texture_set_channel`]; presets that read
/// mesh maps directly also require `mesh_maps`.
pub fn bake_texture_set_export(
    texture_set: &TextureSet,
    preset: &ExportPreset,
    mesh: Option<&MeshGeometry>,
    mesh_maps: Option<&MeshMaps>,
    resources: Option<&ResourceImages>,
) -> Result<Vec<ExportedTexture>, ApiError> {
    Ok(texture_set.bake_export(
        preset,
        BakeContext {
            mesh,
            mesh_maps,
            resources,
        },
    )?)
}

/// Checks an export without baking; see [`TextureSet::export_preflight`].
pub fn export_preflight(
    texture_set: &TextureSet,
    preset: &ExportPreset,
    mesh: Option<&MeshGeometry>,
    mesh_maps: Option<&MeshMaps>,
    resources: Option<&ResourceImages>,
) -> Vec<PreflightIssue> {
    texture_set.export_preflight(
        preset,
        BakeContext {
            mesh,
            mesh_maps,
            resources,
        },
    )
}

/// Files in `folder` that exporting `preset` for `texture_set` as `format`
/// would overwrite.
pub fn existing_export_files(
    folder: &Path,
    texture_set: &TextureSet,
    preset: &ExportPreset,
    format: TextureFormat,
) -> Vec<std::path::PathBuf> {
    preset
        .outputs
        .iter()
        .map(|output| {
            folder.join(format!(
                "{}.{}",
                output.resolved_file_name(&texture_set.name),
                format.extension()
            ))
        })
        .filter(|path| path.exists())
        .collect()
}

/// Loads every bitmap the project lists. Unreadable files are reported per
/// resource rather than failing the whole load, so a project with one
/// missing texture still opens.
/// Loads one resource's pixels, tagged with its colour space.
pub fn load_resource_image(resource: &ProjectResource) -> Result<TextureImage, TextureIoError> {
    let mut image = load_texture_image(&resource.source)?;
    image.color_space = resource.color_space;
    Ok(image)
}

pub fn load_project_resources(
    project: &MaterialProject,
) -> (ResourceImages, Vec<(ResourceId, TextureIoError)>) {
    let mut images = ResourceImages::new();
    let mut failures = Vec::new();
    for resource in &project.resources {
        match load_resource_image(resource) {
            Ok(image) => {
                images.insert(resource.id, std::sync::Arc::new(image));
            }
            Err(error) => failures.push((resource.id, error)),
        }
    }
    (images, failures)
}

#[derive(Debug, Error, PartialEq)]
pub enum ApiError {
    #[error("unknown mesh preset `{0}`")]
    UnknownPrimitivePreset(String),
    #[error(transparent)]
    InvalidGeneratedMesh(#[from] MeshError),
    #[error(transparent)]
    TextureBake(#[from] BakeError),
    #[error(transparent)]
    MeshMapBake(#[from] MeshMapError),
}

#[derive(Debug, Error, PartialEq)]
pub enum ProjectFileError {
    #[error("project JSON is {0} bytes, exceeding the {MAX_PROJECT_JSON_BYTES}-byte limit")]
    TooLarge(usize),
    #[error("project JSON could not be encoded: {0}")]
    Encode(String),
    #[error("project JSON could not be decoded: {0}")]
    Decode(String),
    #[error("project file I/O failed: {0}")]
    Io(String),
    #[error("project data failed validation: {0}")]
    InvalidProject(String),
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tc_0009_preset_listing_and_mesh_inspection_are_consistent() {
        let presets = list_primitive_presets();
        assert_eq!(presets.len(), 4);
        for preset in presets {
            let summary = inspect_primitive_preset(&preset.name).unwrap();
            assert_eq!(summary.name, preset.name);
            assert!(summary.vertex_count > 0);
            assert!(summary.triangle_count > 0);
            let expected_bounds = if preset.name == "Plane" {
                ([-0.5, 0.0, -0.5], [0.5, 0.0, 0.5])
            } else {
                ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
            };
            assert_eq!(summary.bounds_min, expected_bounds.0);
            assert_eq!(summary.bounds_max, expected_bounds.1);
            assert!(summary.uv_bounds_min.iter().all(|value| *value >= 0.0));
            assert!(summary.uv_bounds_max.iter().all(|value| *value <= 1.0));
            assert!(summary.uv_bounds_min[0] < summary.uv_bounds_max[0]);
            assert!(summary.uv_bounds_min[1] < summary.uv_bounds_max[1]);
        }
    }

    #[test]
    fn tc_0010_unknown_preset_is_a_recoverable_api_error() {
        assert_eq!(
            inspect_primitive_preset("icosahedron"),
            Err(ApiError::UnknownPrimitivePreset("icosahedron".to_owned()))
        );
    }

    #[test]
    fn tc_0016_project_json_round_trip_preserves_saved_poses_and_texture_sets() {
        use std::collections::{BTreeMap, BTreeSet};

        use glam::{Quat, Vec2, Vec3};
        use substancecraft_core::{
            BitmapMask, BoneId, BoneTransform, BrushShape, BrushTip, EdgeWear, FillTextures,
            IdStroke, LayerFill, LayerFilter, LayerId, MaskGenerator, MaskStroke, MaterialLayer,
            NamedPose, PaintStroke, PolygonMask, PoseId, ProjectResource, ProjectionMode,
            ResourceId, TextureProjection, TextureResolution, TextureSet, TextureSetId,
        };

        let project = MaterialProject {
            texture_sets: vec![TextureSet {
                id: TextureSetId(2),
                name: "Body".to_owned(),
                material_slot: "BodyMaterial".to_owned(),
                resolution: TextureResolution {
                    width: 2048,
                    height: 2048,
                },
                channels: BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Roughness]),
                layers: vec![
                    MaterialLayer {
                        id: LayerId(3),
                        name: "Paint".to_owned(),
                        visible: true,
                        opacity: 0.75,
                        channels: BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Id]),
                        paint_strokes: vec![PaintStroke {
                            points: vec![Vec2::new(0.25, 0.75), Vec2::new(0.5, 0.5)],
                            radius_pixels: 8.0,
                            color: [32, 64, 128, 255],
                            opacity: 0.8,
                            roughness: 0.3,
                            metallic: 1.0,
                            height: 0.7,
                            emissive: [9, 8, 7],
                            material_opacity: 0.6,
                            erase: true,
                            tip: BrushTip {
                                shape: BrushShape::Bitmap(ResourceId(11)),
                                spacing: 0.5,
                                ..BrushTip::ROUND
                            },
                            pressures: vec![0.2, 0.9],
                            path: Some(substancecraft_core::PathSpec {
                                controls: vec![Vec2::new(0.25, 0.75), Vec2::new(0.5, 0.5)],
                                closed: false,
                            }),
                        }],
                        mask_strokes: vec![MaskStroke {
                            points: vec![Vec2::new(0.5, 0.5)],
                            radius_pixels: 4.0,
                            value: 0.25,
                            opacity: 0.5,
                            tip: BrushTip::ROUND,
                            pressures: Vec::new(),
                        }],
                        id_strokes: vec![IdStroke {
                            points: vec![Vec2::new(0.5, 0.5)],
                            radius_pixels: 4.0,
                            color: [1, 2, 3],
                            opacity: 1.0,
                            tip: BrushTip::ROUND,
                            pressures: Vec::new(),
                        }],
                        id_mask_color: Some([1, 2, 3]),
                        fill: Some(LayerFill {
                            textures: FillTextures {
                                base_color: Some(ResourceId(11)),
                                ..FillTextures::default()
                            },
                            projection: TextureProjection {
                                mode: ProjectionMode::Triplanar,
                                scale: 3.0,
                                ..TextureProjection::default()
                            },
                            ..LayerFill::default()
                        }),
                        mask_generator: Some(MaskGenerator::EdgeWear(EdgeWear::DEFAULT)),
                        polygon_mask: Some(PolygonMask {
                            triangles: BTreeSet::from([0, 7, 9]),
                            inverted: true,
                        }),
                        mask_bitmap: Some(BitmapMask {
                            resource: ResourceId(11),
                            projection: TextureProjection::default(),
                            invert: true,
                        }),
                        filters: vec![LayerFilter::Blur { radius: 3.0 }],
                        mask_filters: vec![LayerFilter::Invert],
                        mask_anchor: None,
                        children: None,
                    },
                    {
                        let mut group = MaterialLayer::new_group(LayerId(6), "Group");
                        group.children = Some(vec![MaterialLayer::new_paint(LayerId(7), "Child")]);
                        group
                    },
                ],
                height_strength: 2.5,
            }],
            saved_poses: vec![NamedPose {
                id: PoseId(4),
                name: "Turned".to_owned(),
                bones: BTreeMap::from([(
                    BoneId(5),
                    BoneTransform {
                        translation: Vec3::new(1.0, 2.0, 3.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    },
                )]),
            }],
            active_texture_set: Some(TextureSetId(2)),
            resources: vec![ProjectResource::new(
                ResourceId(11),
                "Stencil",
                "stencil.png",
            )],
            ..MaterialProject::empty()
        };

        let encoded = encode_project_json(&project).unwrap();
        let decoded = decode_project_json(&encoded).unwrap();

        assert_eq!(decoded, project);
    }

    #[test]
    fn tc_0027_legacy_layers_default_to_empty_paint_and_mask_data() {
        let encoded = serde_json::json!({
            "schema_version": substancecraft_core::PROJECT_SCHEMA_VERSION,
            "mesh": null,
            "texture_sets": [{
                "id": 1,
                "name": "Legacy",
                "material_slot": "Body",
                "resolution": { "width": 16, "height": 16 },
                "channels": ["BaseColor"],
                "layers": [{
                    "id": 2,
                    "name": "Paint",
                    "visible": true,
                    "opacity": 1.0,
                    "channels": ["BaseColor"]
                }]
            }],
            "saved_poses": [],
            "active_texture_set": 1
        })
        .to_string();

        let project = decode_project_json(encoded.as_bytes()).unwrap();
        let layer = &project.texture_sets[0].layers[0];

        assert!(layer.paint_strokes.is_empty());
        assert!(layer.mask_strokes.is_empty());
        assert!(layer.id_strokes.is_empty());
        assert_eq!(layer.id_mask_color, None);
        assert_eq!(layer.fill, None);
        assert_eq!(layer.mask_generator, None);
        assert_eq!(layer.polygon_mask, None);
    }

    #[test]
    fn tc_0017_project_json_load_rejects_invalid_schema() {
        let project = MaterialProject {
            schema_version: substancecraft_core::PROJECT_SCHEMA_VERSION + 1,
            ..MaterialProject::empty()
        };
        let encoded = serde_json::to_vec(&project).unwrap();

        assert!(matches!(
            decode_project_json(&encoded),
            Err(ProjectFileError::InvalidProject(_))
        ));
    }

    #[test]
    fn tc_0018_project_file_save_and_load_round_trip_atomically() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.sproject");
        let project = MaterialProject::empty();
        let updated = MaterialProject {
            mesh: Some(substancecraft_core::MeshAsset {
                id: substancecraft_core::MeshId(9),
                source: "builtin:Sphere".to_owned(),
                scale: 1.0,
                smoothing: MeshSmoothing::default(),
                high_poly_source: Some("sculpt.fbx".to_owned()),
            }),
            ..MaterialProject::empty()
        };

        save_project_file(&path, &project).unwrap();
        save_project_file(&path, &updated).unwrap();
        let loaded = load_project_file(&path).unwrap();

        assert_eq!(loaded, updated);
    }

    #[test]
    fn tc_0155_smart_material_files_round_trip_and_reject_foreign_data() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("worn.scsmart");
        let mut material = SmartMaterial::built_in().remove(0);
        assert!(material.set_parameter(0, substancecraft_core::ParameterValue::Color([1, 2, 3])));
        save_smart_material(&path, &material).unwrap();
        assert_eq!(load_smart_material(&path).unwrap(), material);

        std::fs::write(&path, br#"{"format":"other","version":1,"material":{}}"#).unwrap();
        assert!(matches!(
            load_smart_material(&path),
            Err(ProjectFileError::Decode(_))
        ));
        material.name.clear();
        assert!(matches!(
            save_smart_material(&path, &material),
            Err(ProjectFileError::InvalidProject(_))
        ));
    }
}
