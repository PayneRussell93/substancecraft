#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::collections::{BTreeMap, BTreeSet};

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const PROJECT_SCHEMA_VERSION: u32 = 1;
pub const MAX_TEXTURE_DIMENSION: u32 = 32_768;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub u64);
    };
}

id_type!(MeshId);
id_type!(TextureSetId);
id_type!(LayerId);
id_type!(BoneId);
id_type!(PoseId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimitivePreset {
    Cube,
    Sphere,
    Cylinder,
    Plane,
}

impl PrimitivePreset {
    pub const ALL: [Self; 4] = [Self::Cube, Self::Sphere, Self::Cylinder, Self::Plane];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Cube => "Cube",
            Self::Sphere => "Sphere",
            Self::Cylinder => "Cylinder",
            Self::Plane => "Plane",
        }
    }

    pub fn build_mesh(self) -> MeshGeometry {
        match self {
            Self::Cube => cube_mesh(),
            Self::Sphere => sphere_mesh(),
            Self::Cylinder => cylinder_mesh(),
            Self::Plane => plane_mesh(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshVertex {
    pub position: Vec3,
    pub uv: glam::Vec2,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshGeometry {
    pub vertices: Vec<MeshVertex>,
    pub triangles: Vec<[u32; 3]>,
}

impl MeshGeometry {
    pub fn validate(&self) -> Result<(), MeshError> {
        if self.vertices.is_empty() || self.triangles.is_empty() {
            return Err(MeshError::EmptyGeometry);
        }
        for (index, vertex) in self.vertices.iter().enumerate() {
            if !vertex.position.is_finite() || !vertex.uv.is_finite() {
                return Err(MeshError::InvalidVertex(index));
            }
        }
        for (triangle_index, triangle) in self.triangles.iter().enumerate() {
            if triangle
                .iter()
                .any(|index| *index as usize >= self.vertices.len())
            {
                return Err(MeshError::InvalidTriangle(triangle_index));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum MeshError {
    #[error("mesh geometry has no vertices or triangles")]
    EmptyGeometry,
    #[error("mesh vertex {0} contains a non-finite value")]
    InvalidVertex(usize),
    #[error("mesh triangle {0} references an invalid vertex")]
    InvalidTriangle(usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub pan: glam::Vec2,
}

impl Default for ViewportCamera {
    fn default() -> Self {
        Self {
            yaw: 0.72,
            pitch: 0.38,
            zoom: 1.0,
            pan: glam::Vec2::ZERO,
        }
    }
}

impl ViewportCamera {
    pub fn orbit(&mut self, delta: glam::Vec2) {
        if !delta.is_finite() {
            return;
        }
        self.yaw = (self.yaw - delta.x * 0.01).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + delta.y * 0.01).clamp(-1.55, 1.55);
    }

    pub fn pan_by(&mut self, delta: glam::Vec2) {
        if delta.is_finite() {
            self.pan += delta;
        }
    }

    pub fn zoom_by(&mut self, scroll_delta: f32) {
        if scroll_delta.is_finite() {
            self.zoom = (self.zoom * (scroll_delta * 0.002).exp()).clamp(0.1, 20.0);
        }
    }

    pub fn rotation(self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportLighting {
    pub rotation: f32,
    pub ambient_fill: f32,
}

impl Default for ViewportLighting {
    fn default() -> Self {
        Self {
            rotation: 0.65,
            ambient_fill: 0.36,
        }
    }
}

impl ViewportLighting {
    pub fn face_intensity(self, normal: Vec3) -> f32 {
        let rotation = if self.rotation.is_finite() {
            self.rotation
        } else {
            0.0
        };
        let ambient = if self.ambient_fill.is_finite() {
            self.ambient_fill.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let light_direction = Vec3::new(rotation.cos() * 0.8, 0.65, rotation.sin() * 0.8)
            .normalize_or_zero();
        ambient + (1.0 - ambient) * normal.normalize_or_zero().dot(light_direction).max(0.0)
    }
}

fn push_quad(mesh: &mut MeshGeometry, positions: [Vec3; 4]) {
    let base = mesh.vertices.len() as u32;
    let uvs = [
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(1.0, 0.0),
        glam::Vec2::new(1.0, 1.0),
        glam::Vec2::new(0.0, 1.0),
    ];
    mesh.vertices.extend(
        positions
            .into_iter()
            .zip(uvs)
            .map(|(position, uv)| MeshVertex { position, uv }),
    );
    mesh.triangles
        .extend([[base, base + 1, base + 2], [base, base + 2, base + 3]]);
}

fn cube_mesh() -> MeshGeometry {
    let mut mesh = MeshGeometry {
        vertices: Vec::with_capacity(24),
        triangles: Vec::with_capacity(12),
    };
    let n = -0.5;
    let p = 0.5;
    for face in [
        [
            Vec3::new(n, n, p),
            Vec3::new(p, n, p),
            Vec3::new(p, p, p),
            Vec3::new(n, p, p),
        ],
        [
            Vec3::new(p, n, n),
            Vec3::new(n, n, n),
            Vec3::new(n, p, n),
            Vec3::new(p, p, n),
        ],
        [
            Vec3::new(n, n, n),
            Vec3::new(n, n, p),
            Vec3::new(n, p, p),
            Vec3::new(n, p, n),
        ],
        [
            Vec3::new(p, n, p),
            Vec3::new(p, n, n),
            Vec3::new(p, p, n),
            Vec3::new(p, p, p),
        ],
        [
            Vec3::new(n, p, p),
            Vec3::new(p, p, p),
            Vec3::new(p, p, n),
            Vec3::new(n, p, n),
        ],
        [
            Vec3::new(n, n, n),
            Vec3::new(p, n, n),
            Vec3::new(p, n, p),
            Vec3::new(n, n, p),
        ],
    ] {
        push_quad(&mut mesh, face);
    }
    mesh
}

fn plane_mesh() -> MeshGeometry {
    let mut mesh = MeshGeometry {
        vertices: Vec::with_capacity(4),
        triangles: Vec::with_capacity(2),
    };
    push_quad(
        &mut mesh,
        [
            Vec3::new(-0.5, 0.0, -0.5),
            Vec3::new(0.5, 0.0, -0.5),
            Vec3::new(0.5, 0.0, 0.5),
            Vec3::new(-0.5, 0.0, 0.5),
        ],
    );
    mesh
}

fn sphere_mesh() -> MeshGeometry {
    const SEGMENTS: usize = 16;
    const RINGS: usize = 8;
    let mut mesh = MeshGeometry {
        vertices: Vec::with_capacity(2 + (RINGS - 1) * (SEGMENTS + 1)),
        triangles: Vec::with_capacity(SEGMENTS * (RINGS - 1) * 2),
    };
    mesh.vertices.push(MeshVertex {
        position: Vec3::Y * 0.5,
        uv: glam::Vec2::new(0.5, 1.0),
    });
    for ring in 1..RINGS {
        let latitude = std::f32::consts::PI * ring as f32 / RINGS as f32;
        for segment in 0..=SEGMENTS {
            let longitude = std::f32::consts::TAU * segment as f32 / SEGMENTS as f32;
            mesh.vertices.push(MeshVertex {
                position: Vec3::new(
                    latitude.sin() * longitude.cos() * 0.5,
                    latitude.cos() * 0.5,
                    latitude.sin() * longitude.sin() * 0.5,
                ),
                uv: glam::Vec2::new(
                    segment as f32 / SEGMENTS as f32,
                    1.0 - ring as f32 / RINGS as f32,
                ),
            });
        }
    }
    let south = mesh.vertices.len() as u32;
    mesh.vertices.push(MeshVertex {
        position: -Vec3::Y * 0.5,
        uv: glam::Vec2::new(0.5, 0.0),
    });

    for segment in 0..SEGMENTS {
        let first = 1 + segment as u32;
        mesh.triangles.push([0, first + 1, first]);
    }
    for ring in 0..RINGS - 2 {
        let upper = 1 + (ring * (SEGMENTS + 1)) as u32;
        let lower = upper + (SEGMENTS + 1) as u32;
        for segment in 0..SEGMENTS {
            let a = upper + segment as u32;
            let b = a + 1;
            let c = lower + segment as u32;
            let d = c + 1;
            mesh.triangles.extend([[a, b, c], [b, d, c]]);
        }
    }
    let last_ring = 1 + ((RINGS - 2) * (SEGMENTS + 1)) as u32;
    for segment in 0..SEGMENTS {
        let current = last_ring + segment as u32;
        mesh.triangles.push([current, current + 1, south]);
    }
    mesh
}

fn cylinder_mesh() -> MeshGeometry {
    const SEGMENTS: usize = 16;
    let mut mesh = MeshGeometry {
        vertices: Vec::with_capacity((SEGMENTS + 1) * 2 + 2),
        triangles: Vec::with_capacity(SEGMENTS * 4),
    };
    for segment in 0..=SEGMENTS {
        let angle = std::f32::consts::TAU * segment as f32 / SEGMENTS as f32;
        let x = angle.cos() * 0.5;
        let z = angle.sin() * 0.5;
        let u = segment as f32 / SEGMENTS as f32;
        mesh.vertices.push(MeshVertex {
            position: Vec3::new(x, -0.5, z),
            uv: glam::Vec2::new(u, 0.0),
        });
        mesh.vertices.push(MeshVertex {
            position: Vec3::new(x, 0.5, z),
            uv: glam::Vec2::new(u, 1.0),
        });
    }
    let bottom_center = mesh.vertices.len() as u32;
    mesh.vertices.push(MeshVertex {
        position: Vec3::new(0.0, -0.5, 0.0),
        uv: glam::Vec2::splat(0.5),
    });
    let top_center = mesh.vertices.len() as u32;
    mesh.vertices.push(MeshVertex {
        position: Vec3::new(0.0, 0.5, 0.0),
        uv: glam::Vec2::splat(0.5),
    });
    for segment in 0..SEGMENTS {
        let bottom = (segment * 2) as u32;
        let top = bottom + 1;
        let next_bottom = bottom + 2;
        let next_top = top + 2;
        mesh.triangles.extend([
            [bottom, next_bottom, top],
            [top, next_bottom, next_top],
            [bottom_center, next_bottom, bottom],
            [top_center, top, next_top],
        ]);
    }
    mesh
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoneTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl BoneTransform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    fn is_valid(self) -> bool {
        self.translation.is_finite()
            && self.rotation.is_finite()
            && self.rotation.length_squared() > f32::EPSILON
            && self.scale.is_finite()
    }

    fn interpolate(self, target: Self, amount: f32) -> Self {
        Self {
            translation: self.translation.lerp(target.translation, amount),
            rotation: self
                .rotation
                .normalize()
                .slerp(target.rotation.normalize(), amount)
                .normalize(),
            scale: self.scale.lerp(target.scale, amount),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamedPose {
    pub id: PoseId,
    pub name: String,
    pub bones: BTreeMap<BoneId, BoneTransform>,
}

impl NamedPose {
    pub fn validate(&self) -> Result<(), PoseError> {
        if self.name.trim().is_empty() {
            return Err(PoseError::EmptyName);
        }
        for (bone_id, transform) in &self.bones {
            if !transform.is_valid() {
                return Err(PoseError::InvalidTransform(*bone_id));
            }
        }
        Ok(())
    }

    pub fn interpolate(&self, target: &Self, amount: f32) -> Result<Self, PoseError> {
        if !amount.is_finite() || !(0.0..=1.0).contains(&amount) {
            return Err(PoseError::InvalidInterpolationAmount(amount));
        }
        self.validate()?;
        target.validate()?;

        let mut bones = BTreeMap::new();
        for bone_id in self.bones.keys().chain(target.bones.keys()) {
            let from = self
                .bones
                .get(bone_id)
                .copied()
                .unwrap_or(BoneTransform::IDENTITY);
            let to = target
                .bones
                .get(bone_id)
                .copied()
                .unwrap_or(BoneTransform::IDENTITY);
            bones.insert(*bone_id, from.interpolate(to, amount));
        }

        Ok(Self {
            id: if amount < 0.5 { self.id } else { target.id },
            name: if amount < 0.5 {
                self.name.clone()
            } else {
                target.name.clone()
            },
            bones,
        })
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum PoseError {
    #[error("pose name must not be empty")]
    EmptyName,
    #[error("pose interpolation amount {0} is outside 0..=1 or is not finite")]
    InvalidInterpolationAmount(f32),
    #[error("pose contains an invalid transform for bone {0:?}")]
    InvalidTransform(BoneId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaterialChannel {
    BaseColor,
    Roughness,
    Metallic,
    Normal,
    Height,
    AmbientOcclusion,
    Emissive,
    Opacity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextureResolution {
    pub width: u32,
    pub height: u32,
}

impl TextureResolution {
    pub fn new(width: u32, height: u32) -> Result<Self, ProjectError> {
        let resolution = Self { width, height };
        resolution.validate()?;
        Ok(resolution)
    }

    fn validate(self) -> Result<(), ProjectError> {
        if self.width == 0
            || self.height == 0
            || self.width > MAX_TEXTURE_DIMENSION
            || self.height > MAX_TEXTURE_DIMENSION
        {
            return Err(ProjectError::InvalidResolution {
                width: self.width,
                height: self.height,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshAsset {
    pub id: MeshId,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialLayer {
    pub id: LayerId,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub channels: BTreeSet<MaterialChannel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureSet {
    pub id: TextureSetId,
    pub name: String,
    pub material_slot: String,
    pub resolution: TextureResolution,
    pub channels: BTreeSet<MaterialChannel>,
    pub layers: Vec<MaterialLayer>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialProject {
    pub schema_version: u32,
    pub mesh: Option<MeshAsset>,
    pub texture_sets: Vec<TextureSet>,
    #[serde(default)]
    pub saved_poses: Vec<NamedPose>,
    pub active_texture_set: Option<TextureSetId>,
}

impl MaterialProject {
    pub fn empty() -> Self {
        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            mesh: None,
            texture_sets: Vec::new(),
            saved_poses: Vec::new(),
            active_texture_set: None,
        }
    }

    pub fn validate(&self) -> Result<(), ProjectError> {
        if self.schema_version != PROJECT_SCHEMA_VERSION {
            return Err(ProjectError::UnsupportedSchemaVersion(self.schema_version));
        }

        let mut texture_set_ids = BTreeSet::new();
        let mut layer_ids = BTreeSet::new();

        for texture_set in &self.texture_sets {
            texture_set.resolution.validate()?;
            if !texture_set_ids.insert(texture_set.id) {
                return Err(ProjectError::DuplicateTextureSetId(texture_set.id));
            }

            for layer in &texture_set.layers {
                if !layer_ids.insert(layer.id) {
                    return Err(ProjectError::DuplicateLayerId(layer.id));
                }
                if !layer.opacity.is_finite() || !(0.0..=1.0).contains(&layer.opacity) {
                    return Err(ProjectError::InvalidLayerOpacity {
                        layer: layer.id,
                        opacity: layer.opacity,
                    });
                }
            }
        }

        let mut pose_ids = BTreeSet::new();
        let mut pose_names = BTreeSet::new();
        for pose in &self.saved_poses {
            pose.validate()?;
            if !pose_ids.insert(pose.id) {
                return Err(ProjectError::DuplicatePoseId(pose.id));
            }
            if !pose_names.insert(pose.name.trim().to_lowercase()) {
                return Err(ProjectError::DuplicatePoseName(pose.name.clone()));
            }
        }

        if let Some(active_id) = self.active_texture_set
            && !texture_set_ids.contains(&active_id)
        {
            return Err(ProjectError::ActiveTextureSetMissing(active_id));
        }

        Ok(())
    }
}

impl Default for MaterialProject {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum ProjectError {
    #[error("project schema version {0} is not supported")]
    UnsupportedSchemaVersion(u32),
    #[error("texture resolution {width}x{height} is outside the supported range")]
    InvalidResolution { width: u32, height: u32 },
    #[error("texture set id {0:?} is duplicated")]
    DuplicateTextureSetId(TextureSetId),
    #[error("layer id {0:?} is duplicated")]
    DuplicateLayerId(LayerId),
    #[error("active texture set {0:?} does not exist")]
    ActiveTextureSetMissing(TextureSetId),
    #[error("layer {layer:?} has invalid opacity {opacity}")]
    InvalidLayerOpacity { layer: LayerId, opacity: f32 },
    #[error("pose id {0:?} is duplicated")]
    DuplicatePoseId(PoseId),
    #[error("pose name `{0}` is duplicated")]
    DuplicatePoseName(String),
    #[error(transparent)]
    InvalidPose(#[from] PoseError),
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn texture_set(id: u64) -> TextureSet {
        TextureSet {
            id: TextureSetId(id),
            name: format!("Material {id}"),
            material_slot: "DefaultMaterial".to_owned(),
            resolution: TextureResolution {
                width: 2048,
                height: 2048,
            },
            channels: BTreeSet::from([MaterialChannel::BaseColor]),
            layers: Vec::new(),
        }
    }

    #[test]
    fn tc_0001_empty_project_is_valid() {
        assert!(MaterialProject::empty().validate().is_ok());
    }

    #[test]
    fn tc_0002_active_texture_set_must_exist() {
        let project = MaterialProject {
            active_texture_set: Some(TextureSetId(7)),
            ..MaterialProject::empty()
        };

        assert_eq!(
            project.validate(),
            Err(ProjectError::ActiveTextureSetMissing(TextureSetId(7)))
        );
    }

    #[test]
    fn tc_0003_texture_set_ids_must_be_unique() {
        let project = MaterialProject {
            texture_sets: vec![texture_set(1), texture_set(1)],
            ..MaterialProject::empty()
        };

        assert_eq!(
            project.validate(),
            Err(ProjectError::DuplicateTextureSetId(TextureSetId(1)))
        );
    }

    #[test]
    fn tc_0004_resolution_must_be_bounded_and_nonzero() {
        assert!(matches!(
            TextureResolution::new(0, 1024),
            Err(ProjectError::InvalidResolution { .. })
        ));
        assert!(matches!(
            TextureResolution::new(MAX_TEXTURE_DIMENSION + 1, 1024),
            Err(ProjectError::InvalidResolution { .. })
        ));
    }

    #[test]
    fn tc_0005_layer_opacity_must_be_finite_and_in_range() {
        let mut set = texture_set(1);
        set.layers.push(MaterialLayer {
            id: LayerId(4),
            name: "Paint".to_owned(),
            visible: true,
            opacity: f32::NAN,
            channels: BTreeSet::from([MaterialChannel::BaseColor]),
        });
        let project = MaterialProject {
            texture_sets: vec![set],
            ..MaterialProject::empty()
        };

        assert!(matches!(
            project.validate(),
            Err(ProjectError::InvalidLayerOpacity {
                layer: LayerId(4),
                ..
            })
        ));
    }
    #[test]
    fn tc_0006_layer_ids_must_be_unique() {
        let mut set = texture_set(1);
        set.layers = vec![
            MaterialLayer {
                id: LayerId(4),
                name: "Paint A".to_owned(),
                visible: true,
                opacity: 1.0,
                channels: BTreeSet::new(),
            },
            MaterialLayer {
                id: LayerId(4),
                name: "Paint B".to_owned(),
                visible: true,
                opacity: 1.0,
                channels: BTreeSet::new(),
            },
        ];
        let project = MaterialProject {
            texture_sets: vec![set],
            ..MaterialProject::empty()
        };

        assert_eq!(
            project.validate(),
            Err(ProjectError::DuplicateLayerId(LayerId(4)))
        );
    }

    #[test]
    fn tc_0007_unsupported_schema_version_is_rejected() {
        let project = MaterialProject {
            schema_version: PROJECT_SCHEMA_VERSION + 1,
            ..MaterialProject::empty()
        };

        assert_eq!(
            project.validate(),
            Err(ProjectError::UnsupportedSchemaVersion(
                PROJECT_SCHEMA_VERSION + 1
            ))
        );
    }

    #[test]
    fn named_poses_interpolate_translation_scale_and_rotation() {
        let from = NamedPose {
            id: PoseId(1),
            name: "Rest".to_owned(),
            bones: BTreeMap::from([(
                BoneId(1),
                BoneTransform {
                    translation: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
            )]),
        };
        let target = NamedPose {
            id: PoseId(2),
            name: "Turn".to_owned(),
            bones: BTreeMap::from([(
                BoneId(1),
                BoneTransform {
                    translation: Vec3::new(10.0, 0.0, 0.0),
                    rotation: Quat::from_rotation_z(std::f32::consts::PI),
                    scale: Vec3::splat(3.0),
                },
            )]),
        };

        let halfway = from.interpolate(&target, 0.5).unwrap();
        let transform = halfway.bones.get(&BoneId(1)).unwrap();
        assert!((transform.translation.x - 5.0).abs() < 1e-5);
        assert!((transform.scale.x - 2.0).abs() < 1e-5);
        let halfway_angle = transform.rotation.to_axis_angle().1;
        assert!(
            (halfway_angle - std::f32::consts::FRAC_PI_2).abs() < 1e-4
                || (halfway_angle - 3.0 * std::f32::consts::FRAC_PI_2).abs() < 1e-4
        );
    }

    #[test]
    fn named_pose_interpolation_rejects_non_finite_amounts() {
        let pose = NamedPose {
            id: PoseId(1),
            name: "Pose".to_owned(),
            bones: BTreeMap::new(),
        };

        assert!(matches!(
            pose.interpolate(&pose, f32::NAN),
            Err(PoseError::InvalidInterpolationAmount(value)) if value.is_nan()
        ));
    }

    #[test]
    fn sparse_pose_interpolation_fades_missing_bones_to_identity() {
        let from = NamedPose {
            id: PoseId(1),
            name: "Rest".to_owned(),
            bones: BTreeMap::from([(
                BoneId(2),
                BoneTransform {
                    translation: Vec3::new(8.0, 0.0, 0.0),
                    ..BoneTransform::IDENTITY
                },
            )]),
        };
        let target = NamedPose {
            id: PoseId(2),
            name: "Empty".to_owned(),
            bones: BTreeMap::new(),
        };

        let halfway = from.interpolate(&target, 0.5).unwrap();
        let transform = halfway.bones.get(&BoneId(2)).unwrap();
        assert!((transform.translation.x - 4.0).abs() < 1e-5);
    }

    #[test]
    fn tc_0008_every_primitive_preset_builds_valid_uv_mapped_triangles() {
        for preset in PrimitivePreset::ALL {
            let mesh = preset.build_mesh();
            assert!(
                mesh.validate().is_ok(),
                "{} mesh should validate",
                preset.name()
            );
            assert!(!mesh.triangles.is_empty());
            let expected_counts = match preset {
                PrimitivePreset::Cube => (24, 12),
                PrimitivePreset::Sphere => (121, 224),
                PrimitivePreset::Cylinder => (36, 64),
                PrimitivePreset::Plane => (4, 2),
            };
            assert_eq!(
                (mesh.vertices.len(), mesh.triangles.len()),
                expected_counts,
                "{} topology should stay stable",
                preset.name()
            );
            assert!(
                mesh.vertices
                    .iter()
                    .all(|vertex| { vertex.position.is_finite() && vertex.uv.is_finite() })
            );
        }
    }

    #[test]
    fn tc_0019_viewport_camera_orbits_pans_and_clamps_zoom() {
        let mut camera = ViewportCamera::default();
        let original_rotation = camera.rotation();
        camera.orbit(glam::Vec2::new(30.0, 15.0));
        camera.pan_by(glam::Vec2::new(8.0, -4.0));
        camera.zoom_by(30.0);

        assert_ne!(camera.rotation(), original_rotation);
        assert_eq!(camera.pan, glam::Vec2::new(8.0, -4.0));
        assert!(camera.zoom > 1.0);
        camera.zoom_by(f32::INFINITY);
        assert!(camera.zoom <= 20.0);
        camera.orbit(glam::Vec2::splat(f32::NAN));
        assert!(camera.yaw.is_finite() && camera.pitch.is_finite());
    }

    #[test]
    fn tc_0020_viewport_environment_rotation_changes_lambert_shading() {
        let normal = Vec3::X;
        let first = ViewportLighting {
            rotation: 0.0,
            ambient_fill: 0.2,
        }
        .face_intensity(normal);
        let rotated = ViewportLighting {
            rotation: std::f32::consts::PI,
            ambient_fill: 0.2,
        }
        .face_intensity(normal);

        assert!(first > rotated);
        assert!(rotated >= 0.0);
        assert!(first <= 1.0);
    }
}
