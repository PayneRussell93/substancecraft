#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::collections::{BTreeMap, BTreeSet};

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod generators;
mod mesh_maps;
pub mod noise;
mod raster;
mod smoothing;
mod topology;

pub use generators::{
    Axis, Direction, Dirt, EdgeWear, Gradient, MaskGenerator, NoisePattern, Orientation,
};
pub use mesh_maps::{
    MAX_MESH_MAP_PADDING, MAX_MESH_MAP_RESOLUTION, MAX_MESH_MAP_SAMPLES, MIN_MESH_MAP_RESOLUTION,
    MeshMapError, MeshMapKind, MeshMapSample, MeshMapSettings, MeshMaps, bake_mesh_maps,
};
pub use smoothing::{
    DEFAULT_SMOOTHING_ANGLE_DEGREES, MAX_SUBDIVIDED_TRIANGLES, MAX_SUBDIVISION_LEVELS,
    MeshSmoothing, corner_normals, remap_subdivided_triangles, subdivide, vertex_normals,
};
pub use topology::{MeshTopology, SelectionMode};

pub const PROJECT_SCHEMA_VERSION: u32 = 1;
pub const MAX_TEXTURE_DIMENSION: u32 = 32_768;
pub const MAX_BAKE_PIXELS: usize = 4_194_304;
/// Upper bound on triangles referenced by one layer's polygon mask.
pub const MAX_POLYGON_MASK_TRIANGLES: usize = 8_388_608;
const MAX_LAYER_STROKES: usize = 8_192;
const MAX_STROKE_POINTS: usize = 4_096;
const MAX_EDIT_HISTORY: usize = 128;

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
    #[error("subdividing the mesh would exceed the triangle budget")]
    TooManyTriangles,
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
        let light_direction =
            Vec3::new(rotation.cos() * 0.8, 0.65, rotation.sin() * 0.8).normalize_or_zero();
        ambient + (1.0 - ambient) * normal.normalize_or_zero().dot(light_direction).max(0.0)
    }
}

fn push_quad(mesh: &mut MeshGeometry, positions: [Vec3; 4]) {
    push_quad_in_uv_rect(mesh, positions, glam::Vec2::ZERO, glam::Vec2::ONE);
}

/// Gap kept between UV charts of built-in primitives so brush strokes and
/// texture filtering near one face do not bleed into another.
const PRIMITIVE_UV_PADDING: f32 = 0.01;

fn push_quad_in_uv_rect(
    mesh: &mut MeshGeometry,
    positions: [Vec3; 4],
    uv_min: glam::Vec2,
    uv_max: glam::Vec2,
) {
    let base = mesh.vertices.len() as u32;
    let uvs = [
        glam::Vec2::new(uv_min.x, uv_min.y),
        glam::Vec2::new(uv_max.x, uv_min.y),
        glam::Vec2::new(uv_max.x, uv_max.y),
        glam::Vec2::new(uv_min.x, uv_max.y),
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
    for (face_index, face) in [
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
    ]
    .into_iter()
    .enumerate()
    {
        // Lay the six faces out as a 3x2 atlas so every face owns its texels.
        let column = (face_index % 3) as f32;
        let row = (face_index / 3) as f32;
        let cell = glam::Vec2::new(1.0 / 3.0, 0.5);
        let uv_min = glam::Vec2::new(column, row) * cell + PRIMITIVE_UV_PADDING;
        let uv_max = glam::Vec2::new(column + 1.0, row + 1.0) * cell - PRIMITIVE_UV_PADDING;
        push_quad_in_uv_rect(&mut mesh, face, uv_min, uv_max);
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
            Vec3::new(-0.5, 0.0, 0.5),
            Vec3::new(0.5, 0.0, 0.5),
            Vec3::new(0.5, 0.0, -0.5),
            Vec3::new(-0.5, 0.0, -0.5),
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
    // The side wraps the upper half of UV space; each cap is a disc in the
    // lower half so caps and side never share texels.
    const CAP_RADIUS: f32 = 0.25 - PRIMITIVE_UV_PADDING * 2.0;
    let side_v = (0.5 + PRIMITIVE_UV_PADDING, 1.0 - PRIMITIVE_UV_PADDING);
    let mut mesh = MeshGeometry {
        vertices: Vec::with_capacity((SEGMENTS + 1) * 2 + (SEGMENTS + 1) * 2),
        triangles: Vec::with_capacity(SEGMENTS * 4),
    };
    for segment in 0..=SEGMENTS {
        let angle = std::f32::consts::TAU * segment as f32 / SEGMENTS as f32;
        let x = angle.cos() * 0.5;
        let z = angle.sin() * 0.5;
        let u = segment as f32 / SEGMENTS as f32;
        mesh.vertices.push(MeshVertex {
            position: Vec3::new(x, -0.5, z),
            uv: glam::Vec2::new(u, side_v.0),
        });
        mesh.vertices.push(MeshVertex {
            position: Vec3::new(x, 0.5, z),
            uv: glam::Vec2::new(u, side_v.1),
        });
    }
    for segment in 0..SEGMENTS {
        let bottom = (segment * 2) as u32;
        let top = bottom + 1;
        let next_bottom = bottom + 2;
        let next_top = top + 2;
        // Counter-clockwise seen from outside, like the other primitives.
        mesh.triangles
            .extend([[bottom, top, next_bottom], [top, next_top, next_bottom]]);
    }

    for (y, uv_center) in [
        (-0.5_f32, glam::Vec2::new(0.25, 0.25)),
        (0.5_f32, glam::Vec2::new(0.75, 0.25)),
    ] {
        let center = mesh.vertices.len() as u32;
        mesh.vertices.push(MeshVertex {
            position: Vec3::new(0.0, y, 0.0),
            uv: uv_center,
        });
        let ring = mesh.vertices.len() as u32;
        for segment in 0..SEGMENTS {
            let angle = std::f32::consts::TAU * segment as f32 / SEGMENTS as f32;
            let (sin, cos) = angle.sin_cos();
            mesh.vertices.push(MeshVertex {
                position: Vec3::new(cos * 0.5, y, sin * 0.5),
                uv: uv_center + glam::Vec2::new(cos, sin) * CAP_RADIUS,
            });
        }
        for segment in 0..SEGMENTS as u32 {
            let current = ring + segment;
            let next = ring + (segment + 1) % SEGMENTS as u32;
            if y < 0.0 {
                mesh.triangles.push([center, current, next]);
            } else {
                mesh.triangles.push([center, next, current]);
            }
        }
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
    Id,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshAsset {
    pub id: MeshId,
    pub source: String,
    /// Uniform scale applied to imported geometry; built-in primitives ignore it.
    #[serde(default = "default_mesh_scale")]
    pub scale: f32,
    /// Shading-normal smoothing and subdivision applied after import. Polygon
    /// masks index triangles of the subdivided mesh.
    #[serde(default)]
    pub smoothing: MeshSmoothing,
}

fn default_mesh_scale() -> f32 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialLayer {
    pub id: LayerId,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub channels: BTreeSet<MaterialChannel>,
    #[serde(default)]
    pub paint_strokes: Vec<PaintStroke>,
    #[serde(default)]
    pub mask_strokes: Vec<MaskStroke>,
    #[serde(default)]
    pub id_strokes: Vec<IdStroke>,
    #[serde(default)]
    pub id_mask_color: Option<[u8; 3]>,
    /// When set, the layer covers its whole mask with uniform material values
    /// before any paint strokes are applied.
    #[serde(default)]
    pub fill: Option<LayerFill>,
    /// Procedural base for the layer mask; painted mask strokes refine it.
    #[serde(default)]
    pub mask_generator: Option<MaskGenerator>,
    /// Restricts the layer to selected mesh triangles.
    #[serde(default)]
    pub polygon_mask: Option<PolygonMask>,
}

impl MaterialLayer {
    pub fn new_paint(id: LayerId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            visible: true,
            opacity: 1.0,
            channels: BTreeSet::from([
                MaterialChannel::BaseColor,
                MaterialChannel::Roughness,
                MaterialChannel::Metallic,
                MaterialChannel::Id,
            ]),
            paint_strokes: Vec::new(),
            mask_strokes: Vec::new(),
            id_strokes: Vec::new(),
            id_mask_color: None,
            fill: None,
            mask_generator: None,
            polygon_mask: None,
        }
    }

    pub fn new_fill(id: LayerId, name: impl Into<String>, fill: LayerFill) -> Self {
        Self {
            fill: Some(fill),
            channels: BTreeSet::from([
                MaterialChannel::BaseColor,
                MaterialChannel::Roughness,
                MaterialChannel::Metallic,
            ]),
            ..Self::new_paint(id, name)
        }
    }

    /// Non-stroke state that history can snapshot and restore as one edit.
    pub fn settings(&self) -> LayerSettings {
        LayerSettings {
            name: self.name.clone(),
            visible: self.visible,
            opacity: self.opacity,
            channels: self.channels.clone(),
            id_mask_color: self.id_mask_color,
            fill: self.fill,
            mask_generator: self.mask_generator,
            polygon_mask: self.polygon_mask.clone(),
        }
    }

    pub fn apply_settings(&mut self, settings: LayerSettings) {
        self.name = settings.name;
        self.visible = settings.visible;
        self.opacity = settings.opacity;
        self.channels = settings.channels;
        self.id_mask_color = settings.id_mask_color;
        self.fill = settings.fill;
        self.mask_generator = settings.mask_generator;
        self.polygon_mask = settings.polygon_mask;
    }

    fn has_valid_settings(&self) -> bool {
        self.fill.is_none_or(|fill| fill.is_valid())
            && self
                .mask_generator
                .is_none_or(|generator| generator.is_valid())
            && self
                .polygon_mask
                .as_ref()
                .is_none_or(|mask| mask.triangles.len() <= MAX_POLYGON_MASK_TRIANGLES)
    }
}

/// Uniform material values written by a fill layer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerFill {
    pub color: [u8; 4],
    pub roughness: f32,
    pub metallic: f32,
}

impl LayerFill {
    fn is_valid(self) -> bool {
        [self.roughness, self.metallic]
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    }
}

impl Default for LayerFill {
    fn default() -> Self {
        Self {
            color: [128, 128, 128, 255],
            roughness: 0.5,
            metallic: 0.0,
        }
    }
}

/// Mesh triangles a layer is limited to. Indices refer to the project mesh;
/// indices beyond the current mesh are ignored when baking.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolygonMask {
    pub triangles: BTreeSet<u32>,
    /// Limit the layer to every triangle *except* the selected ones.
    #[serde(default)]
    pub inverted: bool,
}

/// Snapshot of a layer's non-stroke state.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerSettings {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub channels: BTreeSet<MaterialChannel>,
    pub id_mask_color: Option<[u8; 3]>,
    pub fill: Option<LayerFill>,
    pub mask_generator: Option<MaskGenerator>,
    pub polygon_mask: Option<PolygonMask>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PaintStroke {
    pub points: Vec<glam::Vec2>,
    pub radius_pixels: f32,
    pub color: [u8; 4],
    pub opacity: f32,
    /// Value written to the Roughness channel when the layer enables it.
    #[serde(default = "default_stroke_roughness")]
    pub roughness: f32,
    /// Value written to the Metallic channel when the layer enables it.
    #[serde(default)]
    pub metallic: f32,
}

fn default_stroke_roughness() -> f32 {
    DEFAULT_ROUGHNESS
}

/// Roughness of texels no layer has painted.
pub const DEFAULT_ROUGHNESS: f32 = 0.5;
/// Metallic value of texels no layer has painted.
pub const DEFAULT_METALLIC: f32 = 0.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskStroke {
    pub points: Vec<glam::Vec2>,
    pub radius_pixels: f32,
    pub value: f32,
    pub opacity: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdStroke {
    pub points: Vec<glam::Vec2>,
    pub radius_pixels: f32,
    pub color: [u8; 3],
    pub opacity: f32,
}

/// One undoable project edit. Commands are recorded after they have been
/// applied; undo reverts them and redo applies them again.
#[derive(Clone, Debug, PartialEq)]
pub enum EditCommand {
    Paint {
        texture_set_id: TextureSetId,
        layer_id: LayerId,
        stroke: PaintStroke,
    },
    Mask {
        texture_set_id: TextureSetId,
        layer_id: LayerId,
        stroke: MaskStroke,
    },
    Id {
        texture_set_id: TextureSetId,
        layer_id: LayerId,
        stroke: IdStroke,
    },
    /// Replaced a layer's non-stroke settings (fill, masks, opacity, ...).
    LayerSettings {
        texture_set_id: TextureSetId,
        layer_id: LayerId,
        before: Box<LayerSettings>,
        after: Box<LayerSettings>,
    },
    InsertLayer {
        texture_set_id: TextureSetId,
        index: usize,
        layer: Box<MaterialLayer>,
    },
    RemoveLayer {
        texture_set_id: TextureSetId,
        index: usize,
        layer: Box<MaterialLayer>,
    },
    MoveLayer {
        texture_set_id: TextureSetId,
        from: usize,
        to: usize,
    },
}

/// Bounded undo/redo stacks of [`EditCommand`]s.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EditHistory {
    undo_stack: Vec<EditCommand>,
    redo_stack: Vec<EditCommand>,
}

impl EditHistory {
    /// Records a command the caller has already applied.
    pub fn record(&mut self, command: EditCommand) {
        if self.undo_stack.len() == MAX_EDIT_HISTORY {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(command);
        self.redo_stack.clear();
    }

    /// Applies `command` to `project` and records it. Returns false, leaving
    /// the project and history untouched, when the command does not apply.
    pub fn apply(&mut self, project: &mut MaterialProject, command: EditCommand) -> bool {
        if !apply_command(project, &command) {
            return false;
        }
        self.record(command);
        true
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo(&mut self, project: &mut MaterialProject) -> bool {
        let Some(command) = self.undo_stack.pop() else {
            return false;
        };
        if !revert_command(project, &command) {
            self.undo_stack.push(command);
            return false;
        }
        self.redo_stack.push(command);
        true
    }

    pub fn redo(&mut self, project: &mut MaterialProject) -> bool {
        let Some(command) = self.redo_stack.pop() else {
            return false;
        };
        if !apply_command(project, &command) {
            self.redo_stack.push(command);
            return false;
        }
        if self.undo_stack.len() == MAX_EDIT_HISTORY {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(command);
        true
    }
}

fn find_texture_set(
    project: &mut MaterialProject,
    texture_set_id: TextureSetId,
) -> Option<&mut TextureSet> {
    project
        .texture_sets
        .iter_mut()
        .find(|set| set.id == texture_set_id)
}

fn find_layer(
    project: &mut MaterialProject,
    texture_set_id: TextureSetId,
    layer_id: LayerId,
) -> Option<&mut MaterialLayer> {
    find_texture_set(project, texture_set_id)?
        .layers
        .iter_mut()
        .find(|layer| layer.id == layer_id)
}

fn layer_id_in_use(project: &MaterialProject, layer_id: LayerId) -> bool {
    project
        .texture_sets
        .iter()
        .flat_map(|set| &set.layers)
        .any(|layer| layer.id == layer_id)
}

fn apply_command(project: &mut MaterialProject, command: &EditCommand) -> bool {
    match command {
        EditCommand::Paint {
            texture_set_id,
            layer_id,
            stroke,
        } => find_layer(project, *texture_set_id, *layer_id)
            .map(|layer| layer.paint_strokes.push(stroke.clone()))
            .is_some(),
        EditCommand::Mask {
            texture_set_id,
            layer_id,
            stroke,
        } => find_layer(project, *texture_set_id, *layer_id)
            .map(|layer| layer.mask_strokes.push(stroke.clone()))
            .is_some(),
        EditCommand::Id {
            texture_set_id,
            layer_id,
            stroke,
        } => find_layer(project, *texture_set_id, *layer_id)
            .map(|layer| layer.id_strokes.push(stroke.clone()))
            .is_some(),
        EditCommand::LayerSettings {
            texture_set_id,
            layer_id,
            before,
            after,
        } => replace_layer_settings(project, *texture_set_id, *layer_id, before, after),
        EditCommand::InsertLayer {
            texture_set_id,
            index,
            layer,
        } => insert_layer(project, *texture_set_id, *index, layer),
        EditCommand::RemoveLayer {
            texture_set_id,
            index,
            layer,
        } => remove_layer(project, *texture_set_id, *index, layer),
        EditCommand::MoveLayer {
            texture_set_id,
            from,
            to,
        } => move_layer(project, *texture_set_id, *from, *to),
    }
}

fn revert_command(project: &mut MaterialProject, command: &EditCommand) -> bool {
    match command {
        EditCommand::Paint {
            texture_set_id,
            layer_id,
            stroke,
        } => find_layer(project, *texture_set_id, *layer_id).is_some_and(|layer| {
            layer.paint_strokes.last() == Some(stroke) && layer.paint_strokes.pop().is_some()
        }),
        EditCommand::Mask {
            texture_set_id,
            layer_id,
            stroke,
        } => find_layer(project, *texture_set_id, *layer_id).is_some_and(|layer| {
            layer.mask_strokes.last() == Some(stroke) && layer.mask_strokes.pop().is_some()
        }),
        EditCommand::Id {
            texture_set_id,
            layer_id,
            stroke,
        } => find_layer(project, *texture_set_id, *layer_id).is_some_and(|layer| {
            layer.id_strokes.last() == Some(stroke) && layer.id_strokes.pop().is_some()
        }),
        EditCommand::LayerSettings {
            texture_set_id,
            layer_id,
            before,
            after,
        } => replace_layer_settings(project, *texture_set_id, *layer_id, after, before),
        EditCommand::InsertLayer {
            texture_set_id,
            index,
            layer,
        } => remove_layer(project, *texture_set_id, *index, layer),
        EditCommand::RemoveLayer {
            texture_set_id,
            index,
            layer,
        } => insert_layer(project, *texture_set_id, *index, layer),
        EditCommand::MoveLayer {
            texture_set_id,
            from,
            to,
        } => move_layer(project, *texture_set_id, *to, *from),
    }
}

fn replace_layer_settings(
    project: &mut MaterialProject,
    texture_set_id: TextureSetId,
    layer_id: LayerId,
    expected: &LayerSettings,
    replacement: &LayerSettings,
) -> bool {
    let Some(layer) = find_layer(project, texture_set_id, layer_id) else {
        return false;
    };
    if layer.settings() != *expected {
        return false;
    }
    layer.apply_settings(replacement.clone());
    true
}

fn insert_layer(
    project: &mut MaterialProject,
    texture_set_id: TextureSetId,
    index: usize,
    layer: &MaterialLayer,
) -> bool {
    if layer_id_in_use(project, layer.id) {
        return false;
    }
    let Some(set) = find_texture_set(project, texture_set_id) else {
        return false;
    };
    if index > set.layers.len() {
        return false;
    }
    set.layers.insert(index, layer.clone());
    true
}

fn remove_layer(
    project: &mut MaterialProject,
    texture_set_id: TextureSetId,
    index: usize,
    layer: &MaterialLayer,
) -> bool {
    let Some(set) = find_texture_set(project, texture_set_id) else {
        return false;
    };
    if set.layers.get(index) != Some(layer) {
        return false;
    }
    set.layers.remove(index);
    true
}

fn move_layer(
    project: &mut MaterialProject,
    texture_set_id: TextureSetId,
    from: usize,
    to: usize,
) -> bool {
    let Some(set) = find_texture_set(project, texture_set_id) else {
        return false;
    };
    if from >= set.layers.len() || to >= set.layers.len() || from == to {
        return false;
    }
    let layer = set.layers.remove(from);
    set.layers.insert(to, layer);
    true
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BakedTexture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl BakedTexture {
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = ((y as usize * self.width as usize) + x as usize) * 4;
        let rgba = self.rgba8.get(index..index + 4)?;
        Some([rgba[0], rgba[1], rgba[2], rgba[3]])
    }

    pub fn sample_uv(&self, uv: glam::Vec2) -> Option<[u8; 4]> {
        if !uv.is_finite() || !(0.0..=1.0).contains(&uv.x) || !(0.0..=1.0).contains(&uv.y) {
            return None;
        }
        let x = (uv.x * self.width.saturating_sub(1) as f32).round() as u32;
        let y = ((1.0 - uv.y) * self.height.saturating_sub(1) as f32).round() as u32;
        self.pixel(x, y)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BakeError {
    #[error("texture bake resolution {width}x{height} exceeds the pixel budget")]
    ResolutionExceedsBudget { width: u32, height: u32 },
    #[error("unable to allocate texture bake buffer")]
    AllocationFailed,
    #[error("texture set contains invalid or excessive paint strokes")]
    InvalidStrokeData,
    #[error("texture set contains invalid layer fill, generator, or polygon mask settings")]
    InvalidLayerSettings,
    #[error("the {0:?} channel cannot be baked yet")]
    UnsupportedChannel(MaterialChannel),
    #[error("a layer uses a mask generator but no mesh maps have been baked")]
    MissingMeshMaps,
    #[error("a layer uses a polygon mask but no mesh geometry was provided")]
    MissingMesh,
}

/// Mesh data a texture bake may need: geometry for polygon masks and baked
/// mesh maps for generators. Layers that need neither bake without it.
#[derive(Clone, Copy, Debug, Default)]
pub struct BakeContext<'a> {
    pub mesh: Option<&'a MeshGeometry>,
    pub mesh_maps: Option<&'a MeshMaps>,
}

/// Composited output of one channel while layers are applied.
enum ChannelBuffer {
    Color(Vec<u8>),
    Scalar(Vec<f32>),
}

impl ChannelBuffer {
    fn blend(&mut self, index: usize, source: ChannelValue, alpha: f32) {
        match (self, source) {
            (Self::Color(rgba8), ChannelValue::Color(color)) => {
                blend_rgba8(&mut rgba8[index * 4..index * 4 + 4], color, alpha);
            }
            (Self::Scalar(values), ChannelValue::Scalar(value)) => {
                let alpha = alpha.clamp(0.0, 1.0);
                values[index] += (value - values[index]) * alpha;
            }
            _ => {}
        }
    }

    fn into_rgba8(self) -> Vec<u8> {
        match self {
            Self::Color(rgba8) => rgba8,
            Self::Scalar(values) => values
                .into_iter()
                .flat_map(|value| {
                    let gray = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    [gray, gray, gray, 255]
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy)]
enum ChannelValue {
    Color([u8; 4]),
    Scalar(f32),
}

impl TextureSet {
    /// True when a visible layer uses a mask generator, so baking needs maps.
    pub fn requires_mesh_maps(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| layer.visible && layer.mask_generator.is_some())
    }

    pub fn bake_base_color(&self) -> Result<BakedTexture, BakeError> {
        self.bake_channel(MaterialChannel::BaseColor, BakeContext::default())
    }

    /// Composites every visible layer that enables `channel`. Base Color
    /// produces straight-alpha RGBA; Roughness and Metallic produce opaque
    /// grayscale starting from [`DEFAULT_ROUGHNESS`] / [`DEFAULT_METALLIC`].
    pub fn bake_channel(
        &self,
        channel: MaterialChannel,
        context: BakeContext<'_>,
    ) -> Result<BakedTexture, BakeError> {
        let default_scalar = match channel {
            MaterialChannel::Id => return self.bake_id_mask(),
            MaterialChannel::BaseColor => None,
            MaterialChannel::Roughness => Some(DEFAULT_ROUGHNESS),
            MaterialChannel::Metallic => Some(DEFAULT_METALLIC),
            other => return Err(BakeError::UnsupportedChannel(other)),
        };
        if self.layers.iter().any(|layer| !valid_layer_strokes(layer)) {
            return Err(BakeError::InvalidStrokeData);
        }
        if self.layers.iter().any(|layer| !layer.has_valid_settings()) {
            return Err(BakeError::InvalidLayerSettings);
        }
        let pixel_count = checked_bake_pixel_count(self.resolution)?;
        let id_map = self
            .layers
            .iter()
            .any(|layer| layer.id_mask_color.is_some())
            .then(|| self.bake_id_mask())
            .transpose()?;
        let mut output = match default_scalar {
            None => ChannelBuffer::Color(try_filled(pixel_count * 4, 0_u8)?),
            Some(value) => ChannelBuffer::Scalar(try_filled(pixel_count, value)?),
        };
        let value_for = |color: [u8; 4], roughness: f32, metallic: f32| match channel {
            MaterialChannel::Roughness => ChannelValue::Scalar(roughness),
            MaterialChannel::Metallic => ChannelValue::Scalar(metallic),
            _ => ChannelValue::Color(color),
        };

        for layer in &self.layers {
            if !layer.visible
                || !layer.channels.contains(&channel)
                || (layer.fill.is_none() && layer.paint_strokes.is_empty())
            {
                continue;
            }
            let mask = self.layer_mask(layer, context, id_map.as_ref())?;
            let mask_at = |index: usize| mask.as_ref().map_or(1.0, |values| values[index]);

            if let Some(fill) = layer.fill {
                let value = value_for(fill.color, fill.roughness, fill.metallic);
                for index in 0..pixel_count {
                    output.blend(index, value, layer.opacity * mask_at(index));
                }
            }
            for stroke in &layer.paint_strokes {
                let value = value_for(stroke.color, stroke.roughness, stroke.metallic);
                rasterize_stroke(
                    &stroke.points,
                    stroke.radius_pixels,
                    self.resolution.width,
                    self.resolution.height,
                    |index, coverage| {
                        let alpha = layer.opacity * stroke.opacity * coverage * mask_at(index);
                        output.blend(index, value, alpha);
                    },
                );
            }
        }

        Ok(BakedTexture {
            width: self.resolution.width,
            height: self.resolution.height,
            rgba8: output.into_rgba8(),
        })
    }

    /// Final per-texel mask of a layer, or `None` when it is fully open. The
    /// generator (or white) is the base, painted mask strokes blend over it,
    /// and polygon and ID-colour clipping multiply the result.
    fn layer_mask(
        &self,
        layer: &MaterialLayer,
        context: BakeContext<'_>,
        id_map: Option<&BakedTexture>,
    ) -> Result<Option<Vec<f32>>, BakeError> {
        if layer.mask_generator.is_none()
            && layer.mask_strokes.is_empty()
            && layer.polygon_mask.is_none()
            && layer.id_mask_color.is_none()
        {
            return Ok(None);
        }
        let width = self.resolution.width;
        let height = self.resolution.height;
        let pixel_count = checked_bake_pixel_count(self.resolution)?;
        let mut values = try_filled(pixel_count, 1.0_f32)?;

        if let Some(generator) = &layer.mask_generator {
            let maps = context.mesh_maps.ok_or(BakeError::MissingMeshMaps)?;
            evaluate_generator(&mut values, generator, maps, width, height);
        }
        for stroke in &layer.mask_strokes {
            rasterize_stroke(
                &stroke.points,
                stroke.radius_pixels,
                width,
                height,
                |index, coverage| {
                    let alpha = (stroke.opacity * coverage).clamp(0.0, 1.0);
                    values[index] += (stroke.value - values[index]) * alpha;
                },
            );
        }
        if let Some(polygon_mask) = &layer.polygon_mask {
            let mesh = context.mesh.ok_or(BakeError::MissingMesh)?;
            let mut selected = try_filled(pixel_count, false)?;
            for triangle in &polygon_mask.triangles {
                let Some(corners) = mesh.triangles.get(*triangle as usize) else {
                    continue;
                };
                let Some(uvs) = corners
                    .iter()
                    .map(|index| mesh.vertices.get(*index as usize).map(|vertex| vertex.uv))
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                raster::rasterize_uv_triangle(
                    [uvs[0], uvs[1], uvs[2]],
                    width,
                    height,
                    0.5,
                    |index, _, _| selected[index] = true,
                );
            }
            for (value, selected) in values.iter_mut().zip(selected) {
                if selected == polygon_mask.inverted {
                    *value = 0.0;
                }
            }
        }
        if let Some(target) = layer.id_mask_color {
            for (index, value) in values.iter_mut().enumerate() {
                let matches = id_map
                    .and_then(|map| map.rgba8.get(index * 4..index * 4 + 4))
                    .is_some_and(|pixel| pixel[0..3] == target && pixel[3] > 0);
                if !matches {
                    *value = 0.0;
                }
            }
        }
        for value in &mut values {
            *value = value.clamp(0.0, 1.0);
        }
        Ok(Some(values))
    }

    pub fn bake_id_mask(&self) -> Result<BakedTexture, BakeError> {
        if self.layers.iter().any(|layer| !valid_layer_strokes(layer)) {
            return Err(BakeError::InvalidStrokeData);
        }
        let pixel_count = checked_bake_pixel_count(self.resolution)?;
        let byte_count = pixel_count
            .checked_mul(4)
            .ok_or(BakeError::AllocationFailed)?;
        let mut rgba8 = try_filled(byte_count, 0_u8)?;

        for layer in &self.layers {
            if !layer.visible || !layer.channels.contains(&MaterialChannel::Id) {
                continue;
            }
            for stroke in &layer.id_strokes {
                rasterize_stroke(
                    &stroke.points,
                    stroke.radius_pixels,
                    self.resolution.width,
                    self.resolution.height,
                    |index, coverage| {
                        if coverage * stroke.opacity * layer.opacity >= 0.5 {
                            let offset = index * 4;
                            rgba8[offset..offset + 4].copy_from_slice(&[
                                stroke.color[0],
                                stroke.color[1],
                                stroke.color[2],
                                255,
                            ]);
                        }
                    },
                );
            }
        }

        Ok(BakedTexture {
            width: self.resolution.width,
            height: self.resolution.height,
            rgba8,
        })
    }
}

/// Fills `values` with generator output, splitting rows across threads.
fn evaluate_generator(
    values: &mut [f32],
    generator: &MaskGenerator,
    maps: &MeshMaps,
    width: u32,
    height: u32,
) {
    let row_len = width as usize;
    if row_len == 0 {
        return;
    }
    let threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, 32);
    let rows_per_chunk = (height as usize).div_ceil(threads).max(1);
    let chunk_len = rows_per_chunk * row_len;
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in values.chunks_mut(chunk_len).enumerate() {
            scope.spawn(move || {
                let start = chunk_index * chunk_len;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    let index = start + offset;
                    let x = (index % row_len) as u32;
                    let y = (index / row_len) as u32;
                    let uv = raster::pixel_to_uv(x, y, width, height);
                    *value = generator.evaluate(&maps.sample(uv));
                }
            });
        }
    });
}

fn try_filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>, BakeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| BakeError::AllocationFailed)?;
    values.resize(count, value);
    Ok(values)
}

fn checked_bake_pixel_count(resolution: TextureResolution) -> Result<usize, BakeError> {
    if resolution.width == 0
        || resolution.height == 0
        || resolution.width > MAX_TEXTURE_DIMENSION
        || resolution.height > MAX_TEXTURE_DIMENSION
    {
        return Err(BakeError::ResolutionExceedsBudget {
            width: resolution.width,
            height: resolution.height,
        });
    }
    (resolution.width as usize)
        .checked_mul(resolution.height as usize)
        .filter(|count| *count <= MAX_BAKE_PIXELS)
        .ok_or(BakeError::ResolutionExceedsBudget {
            width: resolution.width,
            height: resolution.height,
        })
}

fn rasterize_stroke(
    points: &[glam::Vec2],
    radius_pixels: f32,
    width: u32,
    height: u32,
    mut apply: impl FnMut(usize, f32),
) {
    let Some(first_point) = points.first().copied() else {
        return;
    };
    let to_pixel = |point: glam::Vec2| {
        glam::Vec2::new(
            point.x * width.saturating_sub(1) as f32,
            (1.0 - point.y) * height.saturating_sub(1) as f32,
        )
    };
    let mut segments = points
        .windows(2)
        .map(|pair| (to_pixel(pair[0]), to_pixel(pair[1])));
    let first = to_pixel(first_point);
    let Some(first_segment) = segments.next() else {
        rasterize_segment(first, first, radius_pixels, width, height, &mut apply);
        return;
    };
    rasterize_segment(
        first_segment.0,
        first_segment.1,
        radius_pixels,
        width,
        height,
        &mut apply,
    );
    for (start, end) in segments {
        rasterize_segment(start, end, radius_pixels, width, height, &mut apply);
    }
}

fn rasterize_segment(
    start: glam::Vec2,
    end: glam::Vec2,
    radius: f32,
    width: u32,
    height: u32,
    apply: &mut impl FnMut(usize, f32),
) {
    let min_x = ((start.x.min(end.x) - radius).floor() as i32).max(0);
    let max_x = ((start.x.max(end.x) + radius).ceil() as i32).min(width as i32 - 1);
    let min_y = ((start.y.min(end.y) - radius).floor() as i32).max(0);
    let max_y = ((start.y.max(end.y) + radius).ceil() as i32).min(height as i32 - 1);
    let segment = end - start;
    let segment_length_squared = segment.length_squared();

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let pixel = glam::Vec2::new(x as f32, y as f32);
            let amount = if segment_length_squared > f32::EPSILON {
                ((pixel - start).dot(segment) / segment_length_squared).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let distance = pixel.distance(start + segment * amount);
            if distance < radius {
                let coverage = 1.0 - distance / radius;
                let index = y as usize * width as usize + x as usize;
                apply(index, coverage);
            }
        }
    }
}

fn blend_rgba8(destination: &mut [u8], color: [u8; 4], opacity: f32) {
    let source_alpha = (color[3] as f32 / 255.0 * opacity).clamp(0.0, 1.0);
    if source_alpha <= 0.0 {
        return;
    }
    let destination_alpha = destination[3] as f32 / 255.0;
    let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);
    let output_alpha_byte = (output_alpha * 255.0).round() as u8;
    if output_alpha_byte == 0 {
        destination.fill(0);
        return;
    }
    for channel in 0..3 {
        let source = color[channel] as f32 / 255.0;
        let existing = destination[channel] as f32 / 255.0;
        let blended = (source * source_alpha + existing * destination_alpha * (1.0 - source_alpha))
            / output_alpha;
        destination[channel] = (blended * 255.0).round() as u8;
    }
    destination[3] = output_alpha_byte;
}

fn valid_stroke(points: &[glam::Vec2], radius_pixels: f32, opacity: f32) -> bool {
    !points.is_empty()
        && points.len() <= MAX_STROKE_POINTS
        && radius_pixels.is_finite()
        && radius_pixels > 0.0
        && radius_pixels <= MAX_TEXTURE_DIMENSION as f32
        && opacity.is_finite()
        && (0.0..=1.0).contains(&opacity)
        && points.iter().all(|point| {
            point.is_finite() && (0.0..=1.0).contains(&point.x) && (0.0..=1.0).contains(&point.y)
        })
}

fn valid_layer_strokes(layer: &MaterialLayer) -> bool {
    layer
        .paint_strokes
        .len()
        .saturating_add(layer.mask_strokes.len())
        .saturating_add(layer.id_strokes.len())
        <= MAX_LAYER_STROKES
        && layer.paint_strokes.iter().all(|stroke| {
            valid_stroke(&stroke.points, stroke.radius_pixels, stroke.opacity)
                && [stroke.roughness, stroke.metallic]
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        })
        && layer.mask_strokes.iter().all(|stroke| {
            valid_stroke(&stroke.points, stroke.radius_pixels, stroke.opacity)
                && stroke.value.is_finite()
                && (0.0..=1.0).contains(&stroke.value)
        })
        && layer
            .id_strokes
            .iter()
            .all(|stroke| valid_stroke(&stroke.points, stroke.radius_pixels, stroke.opacity))
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
                if !valid_layer_strokes(layer) {
                    return Err(ProjectError::InvalidLayerPaintData(layer.id));
                }
                if !layer.has_valid_settings() {
                    return Err(ProjectError::InvalidLayerSettings(layer.id));
                }
            }
        }

        if let Some(mesh) = &self.mesh
            && (!mesh.smoothing.is_valid() || !mesh.scale.is_finite() || mesh.scale <= 0.0)
        {
            return Err(ProjectError::InvalidMeshSettings);
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
    #[error("layer {0:?} has invalid or excessive paint data")]
    InvalidLayerPaintData(LayerId),
    #[error("layer {0:?} has an invalid fill, mask generator, or polygon mask")]
    InvalidLayerSettings(LayerId),
    #[error("mesh scale or smoothing settings are out of range")]
    InvalidMeshSettings,
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
    fn tc_0022_layer_paint_is_clipped_by_its_mask_during_bake() {
        let mut set = texture_set(1);
        set.resolution = TextureResolution {
            width: 16,
            height: 16,
        };
        set.layers.push(MaterialLayer {
            id: LayerId(1),
            name: "Paint".to_owned(),
            visible: true,
            opacity: 1.0,
            channels: BTreeSet::from([MaterialChannel::BaseColor]),
            paint_strokes: vec![PaintStroke {
                points: vec![glam::Vec2::new(7.0 / 15.0, 8.0 / 15.0)],
                radius_pixels: 4.0,
                color: [255, 24, 16, 255],
                opacity: 1.0,
                roughness: DEFAULT_ROUGHNESS,
                metallic: 0.0,
            }],
            mask_strokes: vec![MaskStroke {
                points: vec![glam::Vec2::new(7.0 / 15.0, 8.0 / 15.0)],
                radius_pixels: 2.0,
                value: 0.0,
                opacity: 1.0,
            }],
            id_strokes: Vec::new(),
            id_mask_color: None,
            fill: None,
            mask_generator: None,
            polygon_mask: None,
        });

        let baked = set.bake_base_color().unwrap();

        assert_eq!(baked.pixel(7, 7), Some([0, 0, 0, 0]));
        assert_eq!(baked.pixel(10, 7), Some([255, 24, 16, 64]));
    }

    #[test]
    fn tc_0023_id_mask_bake_keeps_discrete_colors_and_transparent_background() {
        let mut set = texture_set(1);
        set.resolution = TextureResolution {
            width: 8,
            height: 8,
        };
        set.channels.insert(MaterialChannel::Id);
        set.layers.push(MaterialLayer {
            id: LayerId(1),
            name: "ID Paint".to_owned(),
            visible: true,
            opacity: 1.0,
            channels: BTreeSet::from([MaterialChannel::Id]),
            paint_strokes: Vec::new(),
            mask_strokes: Vec::new(),
            id_strokes: vec![IdStroke {
                points: vec![glam::Vec2::new(3.0 / 7.0, 4.0 / 7.0)],
                radius_pixels: 2.0,
                color: [16, 180, 240],
                opacity: 1.0,
            }],
            id_mask_color: None,
            fill: None,
            mask_generator: None,
            polygon_mask: None,
        });

        let baked = set.bake_id_mask().unwrap();

        assert_eq!(baked.pixel(3, 3), Some([16, 180, 240, 255]));
        assert_eq!(baked.pixel(0, 0), Some([0, 0, 0, 0]));
    }

    #[test]
    fn tc_0024_id_mask_color_clips_base_color_paint() {
        let mut set = texture_set(1);
        set.resolution = TextureResolution {
            width: 8,
            height: 8,
        };
        set.channels.insert(MaterialChannel::Id);
        set.layers.push(MaterialLayer {
            id: LayerId(1),
            name: "Masked Paint".to_owned(),
            visible: true,
            opacity: 1.0,
            channels: BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Id]),
            paint_strokes: vec![PaintStroke {
                points: vec![glam::Vec2::new(3.0 / 7.0, 4.0 / 7.0)],
                radius_pixels: 4.0,
                color: [240, 30, 12, 255],
                opacity: 1.0,
                roughness: DEFAULT_ROUGHNESS,
                metallic: 0.0,
            }],
            mask_strokes: Vec::new(),
            id_strokes: vec![IdStroke {
                points: vec![glam::Vec2::new(3.0 / 7.0, 4.0 / 7.0)],
                radius_pixels: 1.0,
                color: [12, 180, 220],
                opacity: 1.0,
            }],
            id_mask_color: Some([12, 180, 220]),
            fill: None,
            mask_generator: None,
            polygon_mask: None,
        });

        let baked = set.bake_base_color().unwrap();

        assert_eq!(baked.pixel(3, 3), Some([240, 30, 12, 255]));
        assert_eq!(baked.pixel(6, 3), Some([0, 0, 0, 0]));
    }

    #[test]
    fn tc_0028_stroke_history_undoes_redoes_and_branches_all_stroke_types() {
        let paint = PaintStroke {
            points: vec![glam::Vec2::new(0.25, 0.75)],
            radius_pixels: 2.0,
            color: [240, 30, 12, 255],
            opacity: 1.0,
            roughness: DEFAULT_ROUGHNESS,
            metallic: 0.0,
        };
        let mask = MaskStroke {
            points: vec![glam::Vec2::new(0.5, 0.5)],
            radius_pixels: 3.0,
            value: 0.25,
            opacity: 1.0,
        };
        let id = IdStroke {
            points: vec![glam::Vec2::new(0.75, 0.25)],
            radius_pixels: 4.0,
            color: [12, 180, 220],
            opacity: 1.0,
        };
        let mut set = texture_set(1);
        set.layers.push(MaterialLayer {
            id: LayerId(2),
            name: "Paint".to_owned(),
            visible: true,
            opacity: 1.0,
            channels: BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Id]),
            paint_strokes: vec![paint.clone()],
            mask_strokes: vec![mask.clone()],
            id_strokes: vec![id.clone()],
            id_mask_color: None,
            fill: None,
            mask_generator: None,
            polygon_mask: None,
        });
        let mut project = MaterialProject {
            texture_sets: vec![set],
            ..MaterialProject::empty()
        };
        let mut history = EditHistory::default();
        history.record(EditCommand::Paint {
            texture_set_id: TextureSetId(1),
            layer_id: LayerId(2),
            stroke: paint.clone(),
        });
        history.record(EditCommand::Mask {
            texture_set_id: TextureSetId(1),
            layer_id: LayerId(2),
            stroke: mask.clone(),
        });
        history.record(EditCommand::Id {
            texture_set_id: TextureSetId(1),
            layer_id: LayerId(2),
            stroke: id.clone(),
        });

        assert!(history.undo(&mut project));
        assert!(history.undo(&mut project));
        assert!(history.undo(&mut project));
        let layer = &project.texture_sets[0].layers[0];
        assert!(layer.paint_strokes.is_empty());
        assert!(layer.mask_strokes.is_empty());
        assert!(layer.id_strokes.is_empty());
        assert!(history.redo(&mut project));
        assert!(history.redo(&mut project));
        assert!(history.redo(&mut project));
        let layer = &project.texture_sets[0].layers[0];
        assert_eq!(layer.paint_strokes, vec![paint]);
        assert_eq!(layer.mask_strokes, vec![mask]);
        assert_eq!(layer.id_strokes, vec![id]);

        assert!(history.undo(&mut project));
        history.record(EditCommand::Paint {
            texture_set_id: TextureSetId(1),
            layer_id: LayerId(2),
            stroke: PaintStroke {
                points: vec![glam::Vec2::new(0.1, 0.1)],
                radius_pixels: 1.0,
                color: [1, 2, 3, 255],
                opacity: 1.0,
                roughness: DEFAULT_ROUGHNESS,
                metallic: 0.0,
            },
        });
        assert!(!history.can_redo());
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
            paint_strokes: Vec::new(),
            mask_strokes: Vec::new(),
            id_strokes: Vec::new(),
            id_mask_color: None,
            fill: None,
            mask_generator: None,
            polygon_mask: None,
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
                paint_strokes: Vec::new(),
                mask_strokes: Vec::new(),
                id_strokes: Vec::new(),
                id_mask_color: None,
                fill: None,
                mask_generator: None,
                polygon_mask: None,
            },
            MaterialLayer {
                id: LayerId(4),
                name: "Paint B".to_owned(),
                visible: true,
                opacity: 1.0,
                channels: BTreeSet::new(),
                paint_strokes: Vec::new(),
                mask_strokes: Vec::new(),
                id_strokes: Vec::new(),
                id_mask_color: None,
                fill: None,
                mask_generator: None,
                polygon_mask: None,
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
                PrimitivePreset::Cylinder => (68, 64),
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
    fn tc_0032_primitives_face_outward_with_non_overlapping_uv_charts() {
        for preset in PrimitivePreset::ALL {
            let mesh = preset.build_mesh();
            for triangle in &mesh.triangles {
                let [a, b, c] = triangle.map(|index| mesh.vertices[index as usize]);
                let normal = (b.position - a.position).cross(c.position - a.position);
                let centroid = (a.position + b.position + c.position) / 3.0;
                let outward = if preset == PrimitivePreset::Plane {
                    Vec3::Y
                } else {
                    centroid
                };
                assert!(
                    normal.dot(outward) > 0.0,
                    "{} triangle {triangle:?} faces inward",
                    preset.name()
                );
            }
        }

        // Rasterising every cube face into one map must never write a texel
        // twice: each face owns a separate chart.
        let cube = PrimitivePreset::Cube.build_mesh();
        let mut owner = vec![None; 64 * 64];
        for (triangle_index, triangle) in cube.triangles.iter().enumerate() {
            let face = triangle_index / 2;
            let uvs = triangle.map(|index| cube.vertices[index as usize].uv);
            raster::rasterize_uv_triangle(uvs, 64, 64, 0.0, |index, _, _| {
                assert!(owner[index].is_none_or(|existing| existing == face));
                owner[index] = Some(face);
            });
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

    fn small_set(size: u32, layer: MaterialLayer) -> TextureSet {
        let mut set = texture_set(1);
        set.resolution = TextureResolution {
            width: size,
            height: size,
        };
        set.channels = BTreeSet::from([
            MaterialChannel::BaseColor,
            MaterialChannel::Roughness,
            MaterialChannel::Metallic,
        ]);
        set.layers.push(layer);
        set
    }

    /// UV of a point on cube face `face` (0..6), with `local` in -0.5..=0.5.
    fn cube_face_uv(face: usize, local: glam::Vec2) -> glam::Vec2 {
        let cell = glam::Vec2::new(1.0 / 3.0, 0.5);
        let origin = glam::Vec2::new((face % 3) as f32, (face / 3) as f32) * cell;
        let min = origin + PRIMITIVE_UV_PADDING;
        let max = origin + cell - PRIMITIVE_UV_PADDING;
        min + (max - min) * (local + 0.5)
    }

    #[test]
    fn tc_0041_polygon_masked_fill_layer_covers_only_selected_faces() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Red",
            LayerFill {
                color: [220, 20, 20, 255],
                ..LayerFill::default()
            },
        );
        layer.polygon_mask = Some(PolygonMask {
            triangles: BTreeSet::from([0, 1]),
            inverted: false,
        });
        let set = small_set(48, layer.clone());
        let context = BakeContext {
            mesh: Some(&cube),
            mesh_maps: None,
        };

        let baked = set
            .bake_channel(MaterialChannel::BaseColor, context)
            .unwrap();
        let selected = baked.sample_uv(cube_face_uv(0, glam::Vec2::ZERO)).unwrap();
        let other = baked.sample_uv(cube_face_uv(1, glam::Vec2::ZERO)).unwrap();
        assert_eq!(selected, [220, 20, 20, 255]);
        assert_eq!(other[3], 0);
        assert_eq!(
            set.bake_channel(MaterialChannel::BaseColor, BakeContext::default()),
            Err(BakeError::MissingMesh)
        );

        layer.polygon_mask = Some(PolygonMask {
            triangles: BTreeSet::from([0, 1]),
            inverted: true,
        });
        let inverted = small_set(48, layer)
            .bake_channel(MaterialChannel::BaseColor, context)
            .unwrap();
        assert_eq!(
            inverted
                .sample_uv(cube_face_uv(0, glam::Vec2::ZERO))
                .unwrap()[3],
            0
        );
        assert_eq!(
            inverted
                .sample_uv(cube_face_uv(1, glam::Vec2::ZERO))
                .unwrap()[3],
            255
        );
    }

    #[test]
    fn tc_0042_edge_wear_fill_writes_metal_only_along_convex_edges() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let maps = bake_mesh_maps(
            &cube,
            &MeshMapSettings {
                resolution: 64,
                ao_samples: 8,
                edge_width: 0.06,
                ..MeshMapSettings::default()
            },
            None,
        )
        .unwrap();
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Worn metal",
            LayerFill {
                color: [200, 200, 205, 255],
                roughness: 0.2,
                metallic: 1.0,
            },
        );
        layer.mask_generator = Some(MaskGenerator::EdgeWear(EdgeWear {
            breakup: 0.0,
            ..EdgeWear::DEFAULT
        }));
        let set = small_set(64, layer);
        let context = BakeContext {
            mesh: Some(&cube),
            mesh_maps: Some(&maps),
        };

        let metallic = set
            .bake_channel(MaterialChannel::Metallic, context)
            .unwrap();
        let centre = metallic
            .sample_uv(cube_face_uv(0, glam::Vec2::ZERO))
            .unwrap();
        let edge = metallic
            .sample_uv(cube_face_uv(0, glam::Vec2::new(0.48, 0.0)))
            .unwrap();
        assert_eq!(centre, [0, 0, 0, 255]);
        assert!(edge[0] > 200, "edge metallic {}", edge[0]);

        let roughness = set
            .bake_channel(MaterialChannel::Roughness, context)
            .unwrap();
        let centre = roughness
            .sample_uv(cube_face_uv(0, glam::Vec2::ZERO))
            .unwrap();
        assert_eq!(centre[0], 128, "untouched roughness keeps the default");
        assert_eq!(
            set.bake_channel(MaterialChannel::Metallic, BakeContext::default()),
            Err(BakeError::MissingMeshMaps)
        );
        assert!(set.requires_mesh_maps());
        assert_eq!(
            set.bake_channel(MaterialChannel::Height, context),
            Err(BakeError::UnsupportedChannel(MaterialChannel::Height))
        );
    }

    #[test]
    fn tc_0043_paint_strokes_write_roughness_and_metallic_channels() {
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Paint");
        layer.paint_strokes.push(PaintStroke {
            points: vec![glam::Vec2::new(0.5, 0.5)],
            radius_pixels: 3.0,
            color: [10, 20, 30, 255],
            opacity: 1.0,
            roughness: 0.1,
            metallic: 0.8,
        });
        let set = small_set(9, layer);

        let roughness = set
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap();
        let metallic = set
            .bake_channel(MaterialChannel::Metallic, BakeContext::default())
            .unwrap();

        let centre_roughness = roughness.pixel(4, 4).unwrap();
        assert!(centre_roughness[0].abs_diff(26) <= 1 && centre_roughness[3] == 255);
        assert_eq!(roughness.pixel(0, 0), Some([128, 128, 128, 255]));
        assert!(metallic.pixel(4, 4).unwrap()[0].abs_diff(204) <= 1);
    }

    #[test]
    fn tc_0044_edit_history_restores_layer_settings_and_structure() {
        let mut project = MaterialProject {
            texture_sets: vec![small_set(8, MaterialLayer::new_paint(LayerId(1), "Base"))],
            ..MaterialProject::empty()
        };
        let set_id = TextureSetId(1);
        let mut history = EditHistory::default();

        let before = project.texture_sets[0].layers[0].settings();
        let mut after = before.clone();
        after.opacity = 0.25;
        after.polygon_mask = Some(PolygonMask {
            triangles: BTreeSet::from([3]),
            inverted: false,
        });
        project.texture_sets[0].layers[0].apply_settings(after.clone());
        history.record(EditCommand::LayerSettings {
            texture_set_id: set_id,
            layer_id: LayerId(1),
            before: Box::new(before.clone()),
            after: Box::new(after.clone()),
        });
        assert!(history.undo(&mut project));
        assert_eq!(project.texture_sets[0].layers[0].settings(), before);
        assert!(history.redo(&mut project));
        assert_eq!(project.texture_sets[0].layers[0].settings(), after);

        let fill = MaterialLayer::new_fill(LayerId(2), "Fill", LayerFill::default());
        assert!(history.apply(
            &mut project,
            EditCommand::InsertLayer {
                texture_set_id: set_id,
                index: 1,
                layer: Box::new(fill.clone()),
            }
        ));
        assert!(history.apply(
            &mut project,
            EditCommand::MoveLayer {
                texture_set_id: set_id,
                from: 1,
                to: 0,
            }
        ));
        assert_eq!(project.texture_sets[0].layers[0].id, LayerId(2));
        assert!(history.apply(
            &mut project,
            EditCommand::RemoveLayer {
                texture_set_id: set_id,
                index: 0,
                layer: Box::new(fill.clone()),
            }
        ));
        assert_eq!(project.texture_sets[0].layers.len(), 1);
        assert!(history.undo(&mut project));
        assert!(history.undo(&mut project));
        assert_eq!(project.texture_sets[0].layers[1].id, LayerId(2));
        assert!(history.undo(&mut project));
        assert_eq!(project.texture_sets[0].layers.len(), 1);
        // A duplicate layer id cannot be inserted.
        assert!(!history.apply(
            &mut project,
            EditCommand::InsertLayer {
                texture_set_id: set_id,
                index: 0,
                layer: Box::new(MaterialLayer::new_paint(LayerId(1), "Dup")),
            }
        ));
        assert!(project.validate().is_ok());
    }
}
