#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

mod fbx;

pub use fbx::{FbxImportError, ImportedFbxMesh, MAX_FBX_BYTES, import_binary_fbx_meshes};

use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
};
use substancecraft_core::{
    BakeContext, BakeError, BakedTexture, MaterialChannel, MaterialProject, MeshError, MeshGeometry,
    MeshMapError, MeshMapSettings, MeshMaps, MeshSmoothing, PrimitivePreset, TextureSet,
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

/// Bakes one channel of a texture set. `mesh` is required by polygon masks and
/// `mesh_maps` by mask generators; pass `None` when no layer uses them.
pub fn bake_texture_set_channel(
    texture_set: &TextureSet,
    channel: MaterialChannel,
    mesh: Option<&MeshGeometry>,
    mesh_maps: Option<&MeshMaps>,
) -> Result<BakedTexture, ApiError> {
    Ok(texture_set.bake_channel(channel, BakeContext { mesh, mesh_maps })?)
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
    Ok(substancecraft_core::bake_mesh_maps(&smoothed, &settings, None)?)
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
            BoneId, BoneTransform, EdgeWear, IdStroke, LayerFill, LayerId, MaskGenerator, MaskStroke,
            MaterialLayer, NamedPose, PaintStroke, PolygonMask, PoseId, TextureResolution, TextureSet,
            TextureSetId,
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
                layers: vec![MaterialLayer {
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
                    }],
                    mask_strokes: vec![MaskStroke {
                        points: vec![Vec2::new(0.5, 0.5)],
                        radius_pixels: 4.0,
                        value: 0.25,
                        opacity: 0.5,
                    }],
                    id_strokes: vec![IdStroke {
                        points: vec![Vec2::new(0.5, 0.5)],
                        radius_pixels: 4.0,
                        color: [1, 2, 3],
                        opacity: 1.0,
                    }],
                    id_mask_color: Some([1, 2, 3]),
                    fill: Some(LayerFill::default()),
                    mask_generator: Some(MaskGenerator::EdgeWear(EdgeWear::DEFAULT)),
                    polygon_mask: Some(PolygonMask {
                        triangles: BTreeSet::from([0, 7, 9]),
                        inverted: true,
                    }),
                }],
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
            }),
            ..MaterialProject::empty()
        };

        save_project_file(&path, &project).unwrap();
        save_project_file(&path, &updated).unwrap();
        let loaded = load_project_file(&path).unwrap();

        assert_eq!(loaded, updated);
    }
}
