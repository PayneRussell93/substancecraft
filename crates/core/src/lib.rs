#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::collections::{BTreeMap, BTreeSet};

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod brush;
mod camera_paint;
pub mod color;
mod compositing;
pub mod environment;
mod evolution;
mod export;
mod filters;
mod generators;
mod inspect;
mod instances;
pub mod library;
mod mesh_maps;
pub mod noise;
pub mod painting;
mod raster;
mod region;
mod resources;
pub mod rig;
mod smoothing;
mod templates;
mod thumbnails;
mod topology;
pub mod udim;

pub use brush::{BrushShape, BrushTip, MAX_BRUSH_SPACING, MIN_BRUSH_SPACING, stroke_texel_bounds};
pub use camera_paint::{MAX_STROKE_VIEWPORT, StrokeCamera, projected_stroke_texel_rects};
pub use color::ColorSpace;
pub use compositing::{
    BakeContext, BakeError, BakedTexture, BlendMode, FLATTEN_CHANNELS, FlattenedLayer,
    MaterialSample, MaterialTextures,
};
pub use environment::{EnvironmentImage, LightingPreset, SunLight};
pub use evolution::{Evolution, EvolutionDrive, MAX_EVOLUTION_DRIVES, MAX_EVOLUTIONS};
pub use export::{
    BitDepth, ChannelSlot, ExportOutput, ExportPreset, ExportPresetError, ExportSource,
    ExportedTexture, MAX_EXPORT_FILE_NAME_CHARS, MAX_EXPORT_OUTPUTS, MAX_EXPORT_PRESET_NAME_CHARS,
    OutputLayout, PreflightIssue, PreflightSeverity, SourceComponent, SpecGlossMap,
    TEXTURE_SET_TOKEN,
};
pub use filters::{LayerFilter, MAX_BLUR_RADIUS};
pub use generators::{
    Axis, Direction, Dirt, EdgeWear, Gradient, IdSelection, MaskGenerator, NoisePattern,
    Orientation, Pattern, PatternKind, Thickness,
};
pub use inspect::{MASK_SHOWN, OverlayPalette, TexelContribution, mask_drivers, mask_overlay};
pub use instances::{InstanceField, MaterialLink};
pub use library::{
    AssetEntry, AssetKind, BUILT_IN_LICENSE, ExposedParameter, ExposedProperty, LibraryError,
    MapRequirements, MaterialResource, ParameterValue, SmartMask, SmartMaterial,
};
pub use mesh_maps::{
    MAX_MESH_MAP_PADDING, MAX_MESH_MAP_RESOLUTION, MAX_MESH_MAP_SAMPLES, MIN_MESH_MAP_RESOLUTION,
    MeshMapError, MeshMapKind, MeshMapSample, MeshMapSettings, MeshMaps, bake_mesh_maps,
    bake_mesh_maps_corrected, bake_mesh_maps_from, cage_ray_direction,
};
pub use painting::{
    LazyStabilizer, MAX_RADIAL_COPIES, PathSpec, SurfaceHit, Symmetry, crosses_uv_seam, line_points,
};
pub use region::{DIRTY_TILE_SIZE, DirtyTexels, TexelRect};
pub use resources::{
    BitmapMask, ContentHash, ContentHasher, DecalPlacement, MAX_DECAL_SIZE, MAX_PROJECTION_SCALE,
    MAX_RESOURCE_DIMENSION, MAX_TILE_SIZE, MIN_DECAL_SIZE, MIN_PROJECTION_SCALE, MIN_TILE_SIZE,
    ProjectResource, ProjectionMode, ResourceImages, ResourceProvenance, TextureImage,
    TextureProjection,
};
pub use rig::{
    AnimationClip, ClipSample, Influence, Joint, JointAnimation, MAX_INFLUENCES, Rig, RigError,
    SkinBinding, Track, strongest_influences,
};
pub use smoothing::{
    DEFAULT_SMOOTHING_ANGLE_DEGREES, MAX_SUBDIVIDED_TRIANGLES, MAX_SUBDIVISION_LEVELS,
    MeshSmoothing, corner_normals, remap_subdivided_triangles, subdivide, vertex_normals,
    vertex_tangents,
};
pub use templates::ProjectTemplate;
pub use thumbnails::{MAX_THUMBNAIL_SIZE, THUMBNAIL_BAKE_RESOLUTION, ThumbnailStage};
pub use topology::{DiagnosticEdge, EdgeFlags, MeshTopology, SelectionMode, diagnostic_edges};
pub use udim::{UDIM_TOKEN, UdimTile};

pub const PROJECT_SCHEMA_VERSION: u32 = 3;
pub const MAX_TEXTURE_DIMENSION: u32 = 32_768;
/// Texels one bake works on at once (its f32 working buffers); larger
/// texture sets bake in tiles of at most this many.
pub const MAX_BAKE_PIXELS: usize = 4_194_304;
/// Largest texture set a bake produces (8192 x 8192).
pub const MAX_TEXTURE_SET_PIXELS: usize = 67_108_864;
/// Upper bound on triangles referenced by one layer's polygon mask.
pub const MAX_POLYGON_MASK_TRIANGLES: usize = 8_388_608;
const MAX_LAYER_STROKES: usize = 8_192;
pub const MAX_STROKE_POINTS: usize = 4_096;
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
id_type!(ResourceId);
id_type!(MaterialId);

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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshGeometry {
    pub vertices: Vec<MeshVertex>,
    pub triangles: Vec<[u32; 3]>,
    /// Material slot of each triangle, indexing `material_slots`. Empty when
    /// the whole mesh uses slot 0.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triangle_materials: Vec<u32>,
    /// Material slot names in index order; each slot becomes a texture set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub material_slots: Vec<String>,
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
        if !self.triangle_materials.is_empty()
            && (self.triangle_materials.len() != self.triangles.len()
                || self
                    .triangle_materials
                    .iter()
                    .any(|slot| *slot as usize >= self.material_slots.len().max(1)))
        {
            return Err(MeshError::InvalidMaterialSlots);
        }
        Ok(())
    }

    /// Number of material slots (at least one).
    pub fn material_slot_count(&self) -> usize {
        self.material_slots.len().max(1)
    }

    /// Slot of triangle `triangle`; 0 when the mesh has no per-triangle
    /// assignment.
    pub fn material_of(&self, triangle: usize) -> u32 {
        self.triangle_materials.get(triangle).copied().unwrap_or(0)
    }

    /// Name of slot `slot`, falling back to a generated name.
    pub fn material_slot_name(&self, slot: u32) -> String {
        self.material_slots
            .get(slot as usize)
            .cloned()
            .unwrap_or_else(|| format!("Material {}", slot + 1))
    }

    /// Triangles of slot `slot` with only the vertices they use. Triangle
    /// order is preserved; the result has a single slot.
    pub fn submesh(&self, slot: u32) -> MeshGeometry {
        let mut remap = vec![u32::MAX; self.vertices.len()];
        let mut submesh = MeshGeometry::default();
        for (index, triangle) in self.triangles.iter().enumerate() {
            if self.material_of(index) != slot {
                continue;
            }
            let corners = triangle.map(|corner| {
                let slot = &mut remap[corner as usize];
                if *slot == u32::MAX {
                    *slot = submesh.vertices.len() as u32;
                    submesh.vertices.push(self.vertices[corner as usize]);
                }
                *slot
            });
            submesh.triangles.push(corners);
        }
        submesh
    }

    /// Appends `other`, offsetting its indices and material slots. Slots of
    /// the same name are shared.
    pub fn append(&mut self, other: &MeshGeometry) -> Result<(), MeshError> {
        let offset = u32::try_from(self.vertices.len()).map_err(|_| MeshError::TooManyTriangles)?;
        if self.triangle_materials.is_empty() && !self.triangles.is_empty() {
            self.triangle_materials = vec![0; self.triangles.len()];
        }
        if self.material_slots.is_empty() && !self.triangles.is_empty() {
            self.material_slots.push(self.material_slot_name(0));
        }
        let mut slot_map = Vec::with_capacity(other.material_slot_count());
        for slot in 0..other.material_slot_count() as u32 {
            let name = other.material_slot_name(slot);
            let index = match self
                .material_slots
                .iter()
                .position(|existing| *existing == name)
            {
                Some(index) => index,
                None => {
                    self.material_slots.push(name);
                    self.material_slots.len() - 1
                }
            };
            slot_map.push(index as u32);
        }
        self.vertices.extend_from_slice(&other.vertices);
        for (index, triangle) in other.triangles.iter().enumerate() {
            self.triangles
                .push(triangle.map(|corner| corner.saturating_add(offset)));
            self.triangle_materials
                .push(slot_map[other.material_of(index) as usize]);
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
    #[error("mesh material slot assignments do not match its triangles")]
    InvalidMaterialSlots,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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

    /// Finite, with pitch and zoom in the ranges the controls allow.
    pub fn is_valid(self) -> bool {
        self.yaw.is_finite()
            && (-1.55..=1.55).contains(&self.pitch)
            && (0.1..=20.0).contains(&self.zoom)
            && self.pan.is_finite()
    }
}

/// The most named cameras a project keeps.
pub const MAX_NAMED_CAMERAS: usize = 64;

/// A viewport camera saved in the project under a name, for returning to
/// the same inspection angle and rendering from it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamedCamera {
    pub name: String,
    pub camera: ViewportCamera,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportLighting {
    /// Turns the environment and its sun about the vertical axis (radians).
    pub rotation: f32,
    /// Strength of the environment's light; the default is 1x.
    pub ambient_fill: f32,
    pub preset: LightingPreset,
}

impl Default for ViewportLighting {
    fn default() -> Self {
        Self {
            rotation: 0.65,
            ambient_fill: DEFAULT_AMBIENT_FILL,
            preset: LightingPreset::default(),
        }
    }
}

/// Ambient fill at which an environment shines at its own strength.
pub const DEFAULT_AMBIENT_FILL: f32 = 0.36;

impl ViewportLighting {
    /// Multiplier of the environment's light for the current ambient fill.
    pub fn environment_scale(self) -> f32 {
        if self.ambient_fill.is_finite() {
            self.ambient_fill.clamp(0.0, 1.0) / DEFAULT_AMBIENT_FILL
        } else {
            1.0
        }
    }

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
        ..MeshGeometry::default()
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
        ..MeshGeometry::default()
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
        ..MeshGeometry::default()
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
        ..MeshGeometry::default()
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

/// A moment of the mesh's imported animation, which a saved pose can show.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationFrame {
    /// Clip name; `None` is the rig's rest pose.
    pub clip: Option<String>,
    /// Seconds into the clip.
    pub time: f32,
}

/// Longest clip name a saved pose keeps.
pub const MAX_CLIP_NAME_CHARS: usize = 256;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamedPose {
    pub id: PoseId,
    pub name: String,
    pub bones: BTreeMap<BoneId, BoneTransform>,
    /// The animation frame the mesh shows in this pose; `None` is the mesh
    /// as imported (its bind pose).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation: Option<AnimationFrame>,
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
        if let Some(frame) = &self.animation
            && (!frame.time.is_finite()
                || frame.time < 0.0
                || frame
                    .clip
                    .as_ref()
                    .is_some_and(|clip| clip.chars().count() > MAX_CLIP_NAME_CHARS))
        {
            return Err(PoseError::InvalidAnimationFrame);
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
            // The same clip blends its time; anything else switches halfway.
            animation: match (&self.animation, &target.animation) {
                (Some(from), Some(to)) if from.clip == to.clip => Some(AnimationFrame {
                    clip: to.clip.clone(),
                    time: from.time + (to.time - from.time) * amount,
                }),
                _ if amount < 0.5 => self.animation.clone(),
                _ => target.animation.clone(),
            },
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
    #[error("pose has an invalid animation frame")]
    InvalidAnimationFrame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    /// Optional high-resolution mesh (file path) whose detail mesh-map baking
    /// projects onto this mesh. Loaded with the same scale.
    #[serde(default)]
    pub high_poly_source: Option<String>,
    /// The mesh file's size and hash when it was imported, for relinking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<ContentHash>,
    /// Likewise for the high-poly file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high_poly_hash: Option<ContentHash>,
}

fn default_mesh_scale() -> f32 {
    1.0
}

/// Deepest allowed nesting of layer groups.
pub const MAX_GROUP_DEPTH: usize = 16;

/// Channels a group passes through by default: every composited channel.
pub const GROUP_CHANNELS: [MaterialChannel; 8] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Height,
    MaterialChannel::Normal,
    MaterialChannel::Emissive,
    MaterialChannel::Opacity,
    MaterialChannel::Id,
];

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
    /// When set, the layer covers its whole mask with uniform or textured
    /// material values before any paint strokes are applied.
    #[serde(default)]
    pub fill: Option<LayerFill>,
    /// Procedural base for the layer mask; painted mask strokes refine it.
    #[serde(default)]
    pub mask_generator: Option<MaskGenerator>,
    /// Restricts the layer to selected mesh triangles.
    #[serde(default)]
    pub polygon_mask: Option<PolygonMask>,
    /// Multiplies the mask by a grayscale bitmap.
    #[serde(default)]
    pub mask_bitmap: Option<BitmapMask>,
    /// Inverts the fully evaluated mask after every mask source and filter.
    #[serde(default)]
    pub invert_final_mask: bool,
    /// Filters applied, in order, to the layer's content before masking.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub filters: Vec<LayerFilter>,
    /// Channels `filters` apply to; empty means every channel.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub filter_channels: BTreeSet<MaterialChannel>,
    /// Filters applied, in order, to the layer's mask.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mask_filters: Vec<LayerFilter>,
    /// Multiplies the mask by a channel of the stack beneath, up to an
    /// earlier anchor layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_anchor: Option<AnchorMask>,
    /// How the layer combines with the stack beneath.
    #[serde(default, skip_serializing_if = "is_normal_blend")]
    pub blend_mode: BlendMode,
    /// Per-channel replacements for `blend_mode` (never pass-through).
    /// Normals always blend normally.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub channel_blend_modes: BTreeMap<MaterialChannel, BlendMode>,
    /// `Some` makes this a group: its children composite together first and
    /// the result blends through the group's own opacity, mask, and filters.
    /// Groups ignore fill and paint strokes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<MaterialLayer>>,
    /// Makes this fill layer an instance of a project material: its fill
    /// and channels follow the master except where the link overrides them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material_link: Option<MaterialLink>,
    /// Controls (for example clean to worn) that move properties of this
    /// layer and the layers inside it through recorded ranges.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evolutions: Vec<Evolution>,
    /// Named parameters of a smart material inserted as this layer, each
    /// a property of this layer or one inside it, offered as one panel.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exposed: Vec<ExposedParameter>,
    /// UDIM tiles the layer (with its children) paints in; empty means
    /// every tile. A bake of the unit square is tile 1001.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub uv_tiles: BTreeSet<UdimTile>,
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
            mask_bitmap: None,
            invert_final_mask: false,
            filters: Vec::new(),
            filter_channels: BTreeSet::new(),
            mask_filters: Vec::new(),
            mask_anchor: None,
            blend_mode: BlendMode::Normal,
            channel_blend_modes: BTreeMap::new(),
            children: None,
            material_link: None,
            evolutions: Vec::new(),
            exposed: Vec::new(),
            uv_tiles: BTreeSet::new(),
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

    pub fn new_group(id: LayerId, name: impl Into<String>) -> Self {
        Self {
            channels: GROUP_CHANNELS.into_iter().collect(),
            children: Some(Vec::new()),
            ..Self::new_paint(id, name)
        }
    }

    pub fn is_group(&self) -> bool {
        self.children.is_some()
    }

    /// This layer followed by all its descendants, depth first.
    pub fn walk(&self) -> Vec<&MaterialLayer> {
        let mut layers = vec![self];
        for child in self.children.iter().flatten() {
            layers.extend(child.walk());
        }
        layers
    }

    /// The content filters that apply to `channel`: all of them unless the
    /// layer limits them to other channels.
    pub fn filters_for(&self, channel: MaterialChannel) -> &[LayerFilter] {
        if self.filter_channels.is_empty() || self.filter_channels.contains(&channel) {
            &self.filters
        } else {
            &[]
        }
    }

    /// How this layer blends `channel`: its channel mode, else its mode.
    pub fn blend_mode_for(&self, channel: MaterialChannel) -> BlendMode {
        self.channel_blend_modes
            .get(&channel)
            .copied()
            .unwrap_or(self.blend_mode)
    }

    fn contains_layer(&self, layer_id: LayerId) -> bool {
        self.walk().iter().any(|layer| layer.id == layer_id)
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
            mask_bitmap: self.mask_bitmap,
            invert_final_mask: self.invert_final_mask,
            filters: self.filters.clone(),
            filter_channels: self.filter_channels.clone(),
            mask_filters: self.mask_filters.clone(),
            mask_anchor: self.mask_anchor,
            blend_mode: self.blend_mode,
            channel_blend_modes: self.channel_blend_modes.clone(),
            material_link: self.material_link.clone(),
            evolutions: self.evolutions.clone(),
            uv_tiles: self.uv_tiles.clone(),
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
        self.mask_bitmap = settings.mask_bitmap;
        self.invert_final_mask = settings.invert_final_mask;
        self.filters = settings.filters;
        self.filter_channels = settings.filter_channels;
        self.mask_filters = settings.mask_filters;
        self.mask_anchor = settings.mask_anchor;
        self.blend_mode = settings.blend_mode;
        self.channel_blend_modes = settings.channel_blend_modes;
        self.material_link = settings.material_link;
        self.evolutions = settings.evolutions;
        self.uv_tiles = settings.uv_tiles;
    }

    /// True when the layer is visible in a bake of the unit square (tile
    /// 1001): visible, and not limited to other UDIM tiles.
    pub fn shown(&self) -> bool {
        self.visible && (self.uv_tiles.is_empty() || self.uv_tiles.contains(&UdimTile::FIRST))
    }

    fn has_valid_settings(&self) -> bool {
        if self.blend_mode == BlendMode::PassThrough && !self.is_group() {
            return false;
        }
        // Pass-through groups composite their children directly, so channel
        // modes would be meaningless; channels never pass through on their
        // own.
        if !self.channel_blend_modes.is_empty()
            && (self.blend_mode == BlendMode::PassThrough
                || self
                    .channel_blend_modes
                    .values()
                    .any(|mode| *mode == BlendMode::PassThrough))
        {
            return false;
        }
        // Only fill layers can be material instances.
        if self.material_link.is_some() && (self.is_group() || self.fill.is_none()) {
            return false;
        }
        if !self.uv_tiles.iter().all(|tile| tile.is_valid()) {
            return false;
        }
        self.fill.is_none_or(|fill| fill.is_valid())
            && self.mask_anchor.is_none_or(AnchorMask::is_valid)
            && self
                .mask_generator
                .is_none_or(|generator| generator.is_valid())
            && self
                .polygon_mask
                .as_ref()
                .is_none_or(|mask| mask.triangles.len() <= MAX_POLYGON_MASK_TRIANGLES)
            && self
                .mask_bitmap
                .is_none_or(|bitmap| bitmap.projection.is_valid())
            && self.filters.len() <= MAX_LAYER_FILTERS
            && self.mask_filters.len() <= MAX_LAYER_FILTERS
            && self
                .filters
                .iter()
                .chain(&self.mask_filters)
                .all(|filter| filter.is_valid())
    }

    /// True when this layer's evolution controls fit its tree. Checked when
    /// projects and smart materials are validated, not when baking: controls
    /// never change a bake, and bakes may see a truncated tree.
    pub(crate) fn has_valid_evolutions(&self) -> bool {
        self.evolutions.len() <= MAX_EVOLUTIONS
            && self
                .evolutions
                .iter()
                .all(|evolution| evolution.is_valid(self))
    }

    /// Bitmap resources this layer (not its children) reads.
    fn own_resources(&self) -> impl Iterator<Item = ResourceId> + '_ {
        let strokes = self
            .paint_strokes
            .iter()
            .map(|stroke| stroke.tip)
            .chain(self.mask_strokes.iter().map(|stroke| stroke.tip))
            .chain(self.id_strokes.iter().map(|stroke| stroke.tip))
            .filter_map(BrushTip::resource);
        self.fill
            .iter()
            .flat_map(|fill| fill.textures.all())
            .chain(self.mask_bitmap.map(|bitmap| bitmap.resource))
            .chain(strokes)
    }

    /// True when this layer (not its children) projects bitmaps triplanarly
    /// or uses a generator, so baking needs mesh maps.
    fn own_needs_mesh_maps(&self) -> bool {
        self.mask_generator
            .is_some_and(|generator| generator.needs_mesh_maps())
            || self
                .paint_strokes
                .iter()
                .any(|stroke| stroke.camera.is_some())
            || self
                .mask_strokes
                .iter()
                .any(|stroke| stroke.camera.is_some())
            || self
                .mask_bitmap
                .is_some_and(|bitmap| bitmap.projection.needs_mesh_maps())
            || self.fill.is_some_and(|fill| {
                fill.projection.needs_mesh_maps()
                    && (fill.textures.all().next().is_some()
                        || fill.projection.mode == ProjectionMode::Decal)
            })
    }
}

const MAX_LAYER_FILTERS: usize = 32;

/// Bitmaps a fill layer uses instead of its uniform values, per channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FillTextures {
    pub base_color: Option<ResourceId>,
    pub roughness: Option<ResourceId>,
    pub metallic: Option<ResourceId>,
    pub height: Option<ResourceId>,
    /// Tangent-space normal map (OpenGL convention, +Y = +V).
    pub normal: Option<ResourceId>,
    pub emissive: Option<ResourceId>,
    pub opacity: Option<ResourceId>,
}

impl FillTextures {
    pub fn for_channel(&self, channel: MaterialChannel) -> Option<ResourceId> {
        match channel {
            MaterialChannel::BaseColor => self.base_color,
            MaterialChannel::Roughness => self.roughness,
            MaterialChannel::Metallic => self.metallic,
            MaterialChannel::Height => self.height,
            MaterialChannel::Normal => self.normal,
            MaterialChannel::Emissive => self.emissive,
            MaterialChannel::Opacity => self.opacity,
            MaterialChannel::Id | MaterialChannel::AmbientOcclusion => None,
        }
    }

    pub fn slot_mut(&mut self, channel: MaterialChannel) -> Option<&mut Option<ResourceId>> {
        match channel {
            MaterialChannel::BaseColor => Some(&mut self.base_color),
            MaterialChannel::Roughness => Some(&mut self.roughness),
            MaterialChannel::Metallic => Some(&mut self.metallic),
            MaterialChannel::Height => Some(&mut self.height),
            MaterialChannel::Normal => Some(&mut self.normal),
            MaterialChannel::Emissive => Some(&mut self.emissive),
            MaterialChannel::Opacity => Some(&mut self.opacity),
            MaterialChannel::Id | MaterialChannel::AmbientOcclusion => None,
        }
    }

    pub fn all(&self) -> impl Iterator<Item = ResourceId> {
        [
            self.base_color,
            self.roughness,
            self.metallic,
            self.height,
            self.normal,
            self.emissive,
            self.opacity,
        ]
        .into_iter()
        .flatten()
    }
}

/// Material values written by a fill layer: uniform values, or bitmaps
/// where [`FillTextures`] assigns them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerFill {
    pub color: [u8; 4],
    pub roughness: f32,
    pub metallic: f32,
    /// Height written when the layer enables the Height channel; `0.5` is
    /// neutral, higher raises the surface and lower dents it.
    #[serde(default = "default_height")]
    pub height: f32,
    /// sRGB emitted colour for the Emissive channel.
    #[serde(default)]
    pub emissive: [u8; 3],
    /// Value written to the Opacity channel (`1` is opaque).
    #[serde(default = "default_material_opacity")]
    pub material_opacity: f32,
    #[serde(default)]
    pub textures: FillTextures,
    #[serde(default)]
    pub projection: TextureProjection,
}

impl LayerFill {
    fn is_valid(self) -> bool {
        [
            self.roughness,
            self.metallic,
            self.height,
            self.material_opacity,
        ]
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            && self.projection.is_valid()
    }
}

impl Default for LayerFill {
    fn default() -> Self {
        Self {
            color: [128, 128, 128, 255],
            roughness: 0.5,
            metallic: 0.0,
            height: DEFAULT_HEIGHT,
            emissive: DEFAULT_EMISSIVE,
            material_opacity: DEFAULT_OPACITY,
            textures: FillTextures::default(),
            projection: TextureProjection::default(),
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

fn is_normal_blend(mode: &BlendMode) -> bool {
    *mode == BlendMode::Normal
}

/// An anchor point reference: masks a layer by a channel of the stack as
/// composited up to and including an earlier layer. The layer is referenced
/// by id, so the link survives reordering and renaming.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnchorMask {
    pub layer: LayerId,
    pub channel: MaterialChannel,
    /// The read value is remapped from `low..high` to `0..1`.
    pub low: f32,
    pub high: f32,
    #[serde(default)]
    pub invert: bool,
}

impl AnchorMask {
    /// Channels an anchor can read; colours are read as luminance.
    pub const CHANNELS: [MaterialChannel; 6] = [
        MaterialChannel::BaseColor,
        MaterialChannel::Roughness,
        MaterialChannel::Metallic,
        MaterialChannel::Height,
        MaterialChannel::Emissive,
        MaterialChannel::Opacity,
    ];

    pub fn new(layer: LayerId, channel: MaterialChannel) -> Self {
        Self {
            layer,
            channel,
            low: 0.0,
            high: 1.0,
            invert: false,
        }
    }

    fn is_valid(self) -> bool {
        Self::CHANNELS.contains(&self.channel)
            && self.low.is_finite()
            && self.high.is_finite()
            && (0.0..=1.0).contains(&self.low)
            && (0.0..=1.0).contains(&self.high)
            && self.low < self.high
    }

    pub(crate) fn remap(self, value: f32) -> f32 {
        let mapped = ((value - self.low) / (self.high - self.low)).clamp(0.0, 1.0);
        if self.invert { 1.0 - mapped } else { mapped }
    }
}

/// Snapshot of a layer's non-stroke, non-structural state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerSettings {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub channels: BTreeSet<MaterialChannel>,
    pub id_mask_color: Option<[u8; 3]>,
    pub fill: Option<LayerFill>,
    pub mask_generator: Option<MaskGenerator>,
    pub polygon_mask: Option<PolygonMask>,
    pub mask_bitmap: Option<BitmapMask>,
    pub invert_final_mask: bool,
    pub filters: Vec<LayerFilter>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub filter_channels: BTreeSet<MaterialChannel>,
    pub mask_filters: Vec<LayerFilter>,
    pub mask_anchor: Option<AnchorMask>,
    pub blend_mode: BlendMode,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub channel_blend_modes: BTreeMap<MaterialChannel, BlendMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material_link: Option<MaterialLink>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evolutions: Vec<Evolution>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub uv_tiles: BTreeSet<UdimTile>,
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
    /// Value written to the Height channel when the layer enables it.
    #[serde(default = "default_height")]
    pub height: f32,
    /// Colour written to the Emissive channel when the layer enables it.
    #[serde(default)]
    pub emissive: [u8; 3],
    /// Value written to the Opacity channel when the layer enables it.
    #[serde(default = "default_material_opacity")]
    pub material_opacity: f32,
    /// Removes paint from the layer instead of adding it.
    #[serde(default)]
    pub erase: bool,
    #[serde(default)]
    pub tip: BrushTip,
    /// Pen pressure per point (`0..=1`); empty means full pressure.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pressures: Vec<f32>,
    /// Spline the points were sampled from, kept so the path stays
    /// editable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathSpec>,
    /// Painted through this view (camera projection): `points` and
    /// `radius_pixels` are in the viewport's pixels, and the stroke lands on
    /// whatever surface showed there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<StrokeCamera>,
}

impl PaintStroke {
    /// A stroke following a spline path, with `template`'s material and
    /// tip.
    pub fn along_path(path: PathSpec, template: &PaintStroke) -> Self {
        Self {
            points: path.sample(PATH_SAMPLES_PER_SEGMENT),
            pressures: Vec::new(),
            path: Some(path),
            camera: None,
            ..template.clone()
        }
    }

    /// A stroke with default material values and a soft round tip.
    pub fn new(points: Vec<glam::Vec2>, radius_pixels: f32, color: [u8; 4]) -> Self {
        Self {
            points,
            radius_pixels,
            color,
            opacity: 1.0,
            roughness: DEFAULT_ROUGHNESS,
            metallic: DEFAULT_METALLIC,
            height: DEFAULT_HEIGHT,
            emissive: DEFAULT_EMISSIVE,
            material_opacity: DEFAULT_OPACITY,
            erase: false,
            tip: BrushTip::ROUND,
            pressures: Vec::new(),
            path: None,
            camera: None,
        }
    }
}

/// Points sampled per spline segment of a path stroke.
pub const PATH_SAMPLES_PER_SEGMENT: usize = 12;

fn default_stroke_roughness() -> f32 {
    DEFAULT_ROUGHNESS
}

fn default_height() -> f32 {
    DEFAULT_HEIGHT
}

fn default_material_opacity() -> f32 {
    DEFAULT_OPACITY
}

/// Roughness of texels no layer has painted.
pub const DEFAULT_ROUGHNESS: f32 = 0.5;
/// Metallic value of texels no layer has painted.
pub const DEFAULT_METALLIC: f32 = 0.0;
/// Base Color shown (and exported in material sets) where no layer paints.
pub const DEFAULT_BASE_COLOR: [u8; 3] = [128, 128, 128];
/// Height of texels no layer has painted (neutral: no bump).
pub const DEFAULT_HEIGHT: f32 = 0.5;
/// Emissive colour of texels no layer has painted.
pub const DEFAULT_EMISSIVE: [u8; 3] = [0, 0, 0];
/// Opacity of texels no layer has painted.
pub const DEFAULT_OPACITY: f32 = 1.0;
/// Default strength of Height-to-normal conversion.
pub const DEFAULT_HEIGHT_STRENGTH: f32 = 1.0;
pub const MAX_HEIGHT_STRENGTH: f32 = 16.0;

fn valid_height_strength(value: f32) -> bool {
    value.is_finite() && (0.0..=MAX_HEIGHT_STRENGTH).contains(&value)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskStroke {
    pub points: Vec<glam::Vec2>,
    pub radius_pixels: f32,
    pub value: f32,
    pub opacity: f32,
    #[serde(default)]
    pub tip: BrushTip,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pressures: Vec<f32>,
    /// Painted through this view; see [`PaintStroke::camera`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<StrokeCamera>,
}

impl MaskStroke {
    /// A mask stroke writing `value` at full opacity with a soft round tip.
    pub fn new(points: Vec<glam::Vec2>, radius_pixels: f32, value: f32) -> Self {
        Self {
            points,
            radius_pixels,
            value,
            opacity: 1.0,
            tip: BrushTip::ROUND,
            pressures: Vec::new(),
            camera: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdStroke {
    pub points: Vec<glam::Vec2>,
    pub radius_pixels: f32,
    pub color: [u8; 3],
    pub opacity: f32,
    #[serde(default)]
    pub tip: BrushTip,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pressures: Vec<f32>,
}

/// Where a layer sits: inside the group `parent` (or the texture set's top
/// level when `None`) at `index`, counted from the bottom.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSlot {
    pub parent: Option<LayerId>,
    pub index: usize,
}

/// One undoable project edit. Commands are recorded after they have been
/// applied; undo reverts them and redo applies them again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
        slot: LayerSlot,
        layer: Box<MaterialLayer>,
    },
    RemoveLayer {
        texture_set_id: TextureSetId,
        slot: LayerSlot,
        layer: Box<MaterialLayer>,
    },
    /// Removes the layer at `from`, then inserts it at `to` (an index into
    /// the destination after the removal). Moving into a group is allowed;
    /// moving a group into itself is not.
    MoveLayer {
        texture_set_id: TextureSetId,
        from: LayerSlot,
        to: LayerSlot,
    },
    /// Points a bitmap resource at another file (a moved or renamed asset).
    RelinkResource {
        resource: ResourceId,
        before: String,
        after: String,
    },
    /// Replaced the project's custom export presets.
    ExportPresets {
        before: Vec<ExportPreset>,
        after: Vec<ExportPreset>,
    },
    /// Replaced the project's materials. Edits of a master travel in a
    /// [`EditCommand::Batch`] with the settings of its instances.
    Materials {
        before: Vec<MaterialResource>,
        after: Vec<MaterialResource>,
    },
    /// Replaced the project's smart masks.
    SmartMasks {
        before: Vec<SmartMask>,
        after: Vec<SmartMask>,
    },
    /// Replaced a texture set's own properties (all but its layers).
    TextureSetProperties {
        texture_set_id: TextureSetId,
        before: Box<TextureSetProperties>,
        after: Box<TextureSetProperties>,
    },
    /// Inserted a texture set at `index`.
    InsertTextureSet { index: usize, set: Box<TextureSet> },
    /// Removed the texture set at `index`.
    RemoveTextureSet { index: usize, set: Box<TextureSet> },
    /// Replaced the project's mesh reference (its source, scale, smoothing,
    /// or high-poly source).
    Mesh {
        before: Option<MeshAsset>,
        after: Option<MeshAsset>,
    },
    /// Several commands applied in order and undone in reverse as one step,
    /// e.g. a symmetric stroke or a flatten. All or none apply.
    Batch(Vec<EditCommand>),
}

/// Bounded undo/redo stacks of [`EditCommand`]s.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EditHistory {
    undo_stack: Vec<EditCommand>,
    redo_stack: Vec<EditCommand>,
    /// The last step came from [`Self::apply_merging`] and may still grow.
    #[serde(skip)]
    merging: bool,
}

impl EditHistory {
    /// Records a command the caller has already applied.
    pub fn record(&mut self, command: EditCommand) {
        self.merging = false;
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

    /// Like [`Self::apply`], but successive texture-set property edits of
    /// one set made this way (the frames of a slider drag) join a single
    /// undo step until [`Self::end_merge`] or any other edit.
    pub fn apply_merging(&mut self, project: &mut MaterialProject, command: EditCommand) -> bool {
        if !apply_command(project, &command) {
            return false;
        }
        self.record_merging(command);
        true
    }

    /// Records an applied command like [`Self::record`], joining it to the
    /// last step when both are texture-set property edits of one set, mesh
    /// edits, or material edits, made in one merging run that picks up
    /// where it left off.
    pub fn record_merging(&mut self, command: EditCommand) {
        if self.merging
            && let Some(last) = self.undo_stack.last_mut()
        {
            match (last, &command) {
                (
                    EditCommand::TextureSetProperties {
                        texture_set_id: last_set,
                        after: last_after,
                        ..
                    },
                    EditCommand::TextureSetProperties {
                        texture_set_id,
                        before,
                        after,
                    },
                ) if last_set == texture_set_id && last_after == before => {
                    last_after.clone_from(after);
                    self.redo_stack.clear();
                    return;
                }
                (
                    EditCommand::Mesh {
                        after: last_after, ..
                    },
                    EditCommand::Mesh { before, after },
                ) if last_after == before => {
                    last_after.clone_from(after);
                    self.redo_stack.clear();
                    return;
                }
                (EditCommand::Batch(last), EditCommand::Batch(next)) => {
                    if instances::merge_material_edits(last, next)
                        || instances::merge_settings_batches(last, next)
                    {
                        self.redo_stack.clear();
                        return;
                    }
                }
                _ => {}
            }
        }
        self.record(command);
        self.merging = true;
    }

    /// Ends a run of [`Self::apply_merging`] edits, e.g. when a drag ends.
    pub fn end_merge(&mut self) {
        self.merging = false;
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo(&mut self, project: &mut MaterialProject) -> bool {
        self.merging = false;
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
        self.merging = false;
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
    find_texture_set(project, texture_set_id)?.layer_mut(layer_id)
}

fn relink_resource(project: &mut MaterialProject, id: ResourceId, from: &str, to: &str) -> bool {
    project
        .resources
        .iter_mut()
        .find(|resource| resource.id == id && resource.source == from)
        .map(|resource| resource.source = to.to_owned())
        .is_some()
}

fn layer_id_in_use(project: &MaterialProject, layer: &MaterialLayer) -> bool {
    layer.walk().iter().any(|candidate| {
        project
            .texture_sets
            .iter()
            .any(|set| set.layer(candidate.id).is_some())
    })
}

fn apply_command(project: &mut MaterialProject, command: &EditCommand) -> bool {
    match command {
        EditCommand::Batch(commands) => {
            for (applied, command) in commands.iter().enumerate() {
                if !apply_command(project, command) {
                    for done in commands[..applied].iter().rev() {
                        revert_command(project, done);
                    }
                    return false;
                }
            }
            true
        }
        EditCommand::Paint {
            texture_set_id,
            layer_id,
            stroke,
        } => find_layer(project, *texture_set_id, *layer_id)
            .filter(|layer| !layer.is_group())
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
            .filter(|layer| !layer.is_group())
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
            slot,
            layer,
        } => insert_layer(project, *texture_set_id, *slot, layer),
        EditCommand::RemoveLayer {
            texture_set_id,
            slot,
            layer,
        } => remove_layer(project, *texture_set_id, *slot, layer),
        EditCommand::MoveLayer {
            texture_set_id,
            from,
            to,
        } => move_layer(project, *texture_set_id, *from, *to),
        EditCommand::RelinkResource {
            resource,
            before,
            after,
        } => relink_resource(project, *resource, before, after),
        EditCommand::ExportPresets { before, after } => {
            replace_value(&mut project.export_presets, before, after)
        }
        EditCommand::Materials { before, after } => {
            replace_value(&mut project.materials, before, after)
        }
        EditCommand::SmartMasks { before, after } => {
            replace_value(&mut project.smart_masks, before, after)
        }
        EditCommand::TextureSetProperties {
            texture_set_id,
            before,
            after,
        } => replace_set_properties(project, *texture_set_id, before, after),
        EditCommand::InsertTextureSet { index, set } => insert_texture_set(project, *index, set),
        EditCommand::RemoveTextureSet { index, set } => remove_texture_set(project, *index, set),
        EditCommand::Mesh { before, after } => replace_value(&mut project.mesh, before, after),
    }
}

fn revert_command(project: &mut MaterialProject, command: &EditCommand) -> bool {
    match command {
        EditCommand::Batch(commands) => {
            for (reverted, command) in commands.iter().rev().enumerate() {
                if !revert_command(project, command) {
                    for done in commands[commands.len() - reverted..].iter() {
                        apply_command(project, done);
                    }
                    return false;
                }
            }
            true
        }
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
            slot,
            layer,
        } => remove_layer(project, *texture_set_id, *slot, layer),
        EditCommand::RemoveLayer {
            texture_set_id,
            slot,
            layer,
        } => insert_layer(project, *texture_set_id, *slot, layer),
        EditCommand::MoveLayer {
            texture_set_id,
            from,
            to,
        } => move_layer(project, *texture_set_id, *to, *from),
        EditCommand::RelinkResource {
            resource,
            before,
            after,
        } => relink_resource(project, *resource, after, before),
        EditCommand::ExportPresets { before, after } => {
            replace_value(&mut project.export_presets, after, before)
        }
        EditCommand::Materials { before, after } => {
            replace_value(&mut project.materials, after, before)
        }
        EditCommand::SmartMasks { before, after } => {
            replace_value(&mut project.smart_masks, after, before)
        }
        EditCommand::TextureSetProperties {
            texture_set_id,
            before,
            after,
        } => replace_set_properties(project, *texture_set_id, after, before),
        EditCommand::InsertTextureSet { index, set } => remove_texture_set(project, *index, set),
        EditCommand::RemoveTextureSet { index, set } => insert_texture_set(project, *index, set),
        EditCommand::Mesh { before, after } => replace_value(&mut project.mesh, after, before),
    }
}

/// Sets `value` to `replacement` when it still equals `expected`.
fn replace_value<T: Clone + PartialEq>(value: &mut T, expected: &T, replacement: &T) -> bool {
    if value != expected {
        return false;
    }
    *value = replacement.clone();
    true
}

fn replace_set_properties(
    project: &mut MaterialProject,
    texture_set_id: TextureSetId,
    expected: &TextureSetProperties,
    replacement: &TextureSetProperties,
) -> bool {
    let Some(set) = find_texture_set(project, texture_set_id) else {
        return false;
    };
    if set.properties() != *expected {
        return false;
    }
    set.apply_properties(replacement.clone());
    true
}

fn insert_texture_set(project: &mut MaterialProject, index: usize, set: &TextureSet) -> bool {
    let id_taken = project
        .texture_sets
        .iter()
        .any(|existing| existing.id == set.id);
    let layer_taken = set
        .layers
        .iter()
        .any(|layer| layer_id_in_use(project, layer));
    if id_taken || layer_taken || index > project.texture_sets.len() {
        return false;
    }
    project.texture_sets.insert(index, set.clone());
    true
}

fn remove_texture_set(project: &mut MaterialProject, index: usize, set: &TextureSet) -> bool {
    if project.texture_sets.get(index) != Some(set) {
        return false;
    }
    project.texture_sets.remove(index);
    if project.active_texture_set == Some(set.id) {
        project.active_texture_set = project.texture_sets.first().map(|set| set.id);
    }
    true
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
    slot: LayerSlot,
    layer: &MaterialLayer,
) -> bool {
    if layer_id_in_use(project, layer) {
        return false;
    }
    let Some(set) = find_texture_set(project, texture_set_id) else {
        return false;
    };
    if slot
        .parent
        .is_some_and(|parent| set.depth_of(parent) + 1 + layer.group_depth() > MAX_GROUP_DEPTH)
    {
        return false;
    }
    let Some(container) = set.container_mut(slot.parent) else {
        return false;
    };
    if slot.index > container.len() {
        return false;
    }
    container.insert(slot.index, layer.clone());
    true
}

fn remove_layer(
    project: &mut MaterialProject,
    texture_set_id: TextureSetId,
    slot: LayerSlot,
    layer: &MaterialLayer,
) -> bool {
    let Some(container) =
        find_texture_set(project, texture_set_id).and_then(|set| set.container_mut(slot.parent))
    else {
        return false;
    };
    if container.get(slot.index) != Some(layer) {
        return false;
    }
    container.remove(slot.index);
    true
}

fn move_layer(
    project: &mut MaterialProject,
    texture_set_id: TextureSetId,
    from: LayerSlot,
    to: LayerSlot,
) -> bool {
    let Some(set) = find_texture_set(project, texture_set_id) else {
        return false;
    };
    if from == to {
        return false;
    }
    let Some(moving) = set
        .container(from.parent)
        .and_then(|container| container.get(from.index))
    else {
        return false;
    };
    // A group cannot move into itself or one of its descendants.
    if to
        .parent
        .is_some_and(|parent| moving.contains_layer(parent))
    {
        return false;
    }
    let depth = moving.group_depth();
    let destination_fits = match to.parent {
        None => true,
        Some(parent) => {
            set.layer(parent).is_some_and(MaterialLayer::is_group)
                && set.depth_of(parent) + 1 + depth <= MAX_GROUP_DEPTH
        }
    };
    if !destination_fits {
        return false;
    }
    let Some(source) = set.container_mut(from.parent) else {
        return false;
    };
    let layer = source.remove(from.index);
    match set.container_mut(to.parent) {
        Some(destination) if to.index <= destination.len() => {
            destination.insert(to.index, layer);
            true
        }
        _ => {
            // Put it back where it was; the move does not apply.
            if let Some(source) = set.container_mut(from.parent) {
                source.insert(from.index, layer);
            }
            false
        }
    }
}

impl MaterialLayer {
    /// Nesting levels inside this layer: 0 for a plain layer or empty group.
    fn group_depth(&self) -> usize {
        self.children
            .iter()
            .flatten()
            .filter(|child| child.is_group())
            .map(|child| 1 + child.group_depth())
            .max()
            .unwrap_or(0)
    }
}

pub(crate) fn valid_stroke(
    points: &[glam::Vec2],
    pressures: &[f32],
    radius_pixels: f32,
    opacity: f32,
    tip: BrushTip,
    camera: Option<&StrokeCamera>,
) -> bool {
    let points_valid = match camera {
        // Screen-space points of a projected stroke.
        Some(camera) => camera.is_valid() && camera.valid_points(points),
        // UDIM layouts place strokes beyond the unit square: up to ten tiles
        // across and a hundred rows, either side once shifted to a tile.
        None => points.iter().all(|point| {
            point.is_finite() && point.abs().cmple(glam::Vec2::new(10.0, 100.0)).all()
        }),
    };
    !points.is_empty()
        && points.len() <= MAX_STROKE_POINTS
        && radius_pixels.is_finite()
        && radius_pixels > 0.0
        && radius_pixels <= MAX_TEXTURE_DIMENSION as f32
        && opacity.is_finite()
        && (0.0..=1.0).contains(&opacity)
        && points_valid
        && brush::valid_pressures(points, pressures)
        && tip.is_valid()
}

pub(crate) fn valid_layer_strokes(layer: &MaterialLayer) -> bool {
    layer
        .paint_strokes
        .len()
        .saturating_add(layer.mask_strokes.len())
        .saturating_add(layer.id_strokes.len())
        <= MAX_LAYER_STROKES
        && layer.paint_strokes.iter().all(|stroke| {
            valid_stroke(
                &stroke.points,
                &stroke.pressures,
                stroke.radius_pixels,
                stroke.opacity,
                stroke.tip,
                stroke.camera.as_ref(),
            ) && [
                stroke.roughness,
                stroke.metallic,
                stroke.height,
                stroke.material_opacity,
            ]
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        })
        && layer.mask_strokes.iter().all(|stroke| {
            valid_stroke(
                &stroke.points,
                &stroke.pressures,
                stroke.radius_pixels,
                stroke.opacity,
                stroke.tip,
                stroke.camera.as_ref(),
            ) && stroke.value.is_finite()
                && (0.0..=1.0).contains(&stroke.value)
        })
        && layer.id_strokes.iter().all(|stroke| {
            valid_stroke(
                &stroke.points,
                &stroke.pressures,
                stroke.radius_pixels,
                stroke.opacity,
                stroke.tip,
                None,
            )
        })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureSet {
    pub id: TextureSetId,
    pub name: String,
    pub material_slot: String,
    pub resolution: TextureResolution,
    pub channels: BTreeSet<MaterialChannel>,
    /// Layer stack, bottom first. Groups nest further layers.
    pub layers: Vec<MaterialLayer>,
    /// How strongly the Height channel bends the Normal channel.
    #[serde(default = "default_height_strength")]
    pub height_strength: f32,
    /// Export choices remembered for this set; `None` uses the app-wide
    /// choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export: Option<TextureSetExport>,
}

/// Image file formats texture exports can be written in, by extension.
pub const EXPORT_FILE_FORMATS: [&str; 3] = ["png", "tga", "exr"];

/// How one texture set exports: its preset, file format, and whether it
/// writes one file set per UDIM tile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextureSetExport {
    /// Name of a built-in or project export preset.
    pub preset: String,
    /// File extension, one of [`EXPORT_FILE_FORMATS`].
    #[serde(default = "default_export_format")]
    pub format: String,
    #[serde(default)]
    pub udim: bool,
}

fn default_export_format() -> String {
    "png".to_owned()
}

fn default_height_strength() -> f32 {
    DEFAULT_HEIGHT_STRENGTH
}

/// A texture set's own properties: everything but its id and layers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureSetProperties {
    pub name: String,
    pub material_slot: String,
    pub resolution: TextureResolution,
    pub channels: BTreeSet<MaterialChannel>,
    pub height_strength: f32,
    pub export: Option<TextureSetExport>,
}

impl TextureSet {
    /// Every layer in the tree, depth first, groups before their children.
    pub fn walk_layers(&self) -> Vec<&MaterialLayer> {
        self.layers.iter().flat_map(MaterialLayer::walk).collect()
    }

    pub fn properties(&self) -> TextureSetProperties {
        TextureSetProperties {
            name: self.name.clone(),
            material_slot: self.material_slot.clone(),
            resolution: self.resolution,
            channels: self.channels.clone(),
            height_strength: self.height_strength,
            export: self.export.clone(),
        }
    }

    pub fn apply_properties(&mut self, properties: TextureSetProperties) {
        self.name = properties.name;
        self.material_slot = properties.material_slot;
        self.resolution = properties.resolution;
        self.channels = properties.channels;
        self.height_strength = properties.height_strength;
        self.export = properties.export;
    }

    /// The power-of-two factor that brings the set's longer edge down to at
    /// most `max_edge` (1 when it already fits).
    pub fn preview_factor(&self, max_edge: u32) -> u32 {
        let edge = self.resolution.width.max(self.resolution.height);
        let mut factor = 1_u32;
        while edge.div_ceil(factor) > max_edge.max(1) {
            factor *= 2;
        }
        factor
    }

    /// A copy at most `max_edge` texels on its longer edge for previews,
    /// with stroke radii scaled to match, or `None` when the set already
    /// fits. Positions, filters, and projections are relative to the
    /// texture, so the copy looks like the full set filtered down.
    pub fn downscaled(&self, max_edge: u32) -> Option<TextureSet> {
        let factor = self.preview_factor(max_edge);
        if factor == 1 {
            return None;
        }
        fn scale(layers: &mut [MaterialLayer], factor: f32) {
            // Projected strokes are sized in screen pixels, not texels.
            for layer in layers {
                for stroke in &mut layer.paint_strokes {
                    if stroke.camera.is_none() {
                        stroke.radius_pixels /= factor;
                    }
                }
                for stroke in &mut layer.mask_strokes {
                    if stroke.camera.is_none() {
                        stroke.radius_pixels /= factor;
                    }
                }
                for stroke in &mut layer.id_strokes {
                    stroke.radius_pixels /= factor;
                }
                if let Some(children) = &mut layer.children {
                    scale(children, factor);
                }
            }
        }
        let mut copy = self.clone();
        copy.resolution = TextureResolution {
            width: self.resolution.width.div_ceil(factor),
            height: self.resolution.height.div_ceil(factor),
        };
        scale(&mut copy.layers, factor as f32);
        Some(copy)
    }

    pub fn layer(&self, layer_id: LayerId) -> Option<&MaterialLayer> {
        fn find(layers: &[MaterialLayer], layer_id: LayerId) -> Option<&MaterialLayer> {
            layers.iter().find_map(|layer| {
                if layer.id == layer_id {
                    Some(layer)
                } else {
                    find(layer.children.as_deref()?, layer_id)
                }
            })
        }
        find(&self.layers, layer_id)
    }

    pub fn layer_mut(&mut self, layer_id: LayerId) -> Option<&mut MaterialLayer> {
        fn find(layers: &mut [MaterialLayer], layer_id: LayerId) -> Option<&mut MaterialLayer> {
            for layer in layers {
                if layer.id == layer_id {
                    return Some(layer);
                }
                if let Some(found) = layer
                    .children
                    .as_deref_mut()
                    .and_then(|children| find(children, layer_id))
                {
                    return Some(found);
                }
            }
            None
        }
        find(&mut self.layers, layer_id)
    }

    /// The layer list of group `parent`, or the top level for `None`.
    pub fn container(&self, parent: Option<LayerId>) -> Option<&Vec<MaterialLayer>> {
        match parent {
            None => Some(&self.layers),
            Some(id) => self.layer(id)?.children.as_ref(),
        }
    }

    pub fn container_mut(&mut self, parent: Option<LayerId>) -> Option<&mut Vec<MaterialLayer>> {
        match parent {
            None => Some(&mut self.layers),
            Some(id) => self.layer_mut(id)?.children.as_mut(),
        }
    }

    /// Where `layer_id` sits in the tree.
    pub fn slot_of(&self, layer_id: LayerId) -> Option<LayerSlot> {
        fn find(
            layers: &[MaterialLayer],
            parent: Option<LayerId>,
            layer_id: LayerId,
        ) -> Option<LayerSlot> {
            layers.iter().enumerate().find_map(|(index, layer)| {
                if layer.id == layer_id {
                    Some(LayerSlot { parent, index })
                } else {
                    find(layer.children.as_deref()?, Some(layer.id), layer_id)
                }
            })
        }
        find(&self.layers, None, layer_id)
    }

    /// Groups enclosing `layer_id`: 0 at the top level.
    pub fn depth_of(&self, layer_id: LayerId) -> usize {
        let mut depth = 0;
        let mut current = self.slot_of(layer_id).and_then(|slot| slot.parent);
        while let Some(parent) = current {
            depth += 1;
            current = self.slot_of(parent).and_then(|slot| slot.parent);
        }
        depth
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialProject {
    pub schema_version: u32,
    pub mesh: Option<MeshAsset>,
    pub texture_sets: Vec<TextureSet>,
    #[serde(default)]
    pub saved_poses: Vec<NamedPose>,
    pub active_texture_set: Option<TextureSetId>,
    /// Bitmaps layers may reference (fill textures, bitmap masks, brush
    /// alphas).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<ProjectResource>,
    /// Custom export presets, offered alongside [`ExportPreset::built_in`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub export_presets: Vec<ExportPreset>,
    /// Saved material resources (fill presets).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<MaterialResource>,
    /// Saved smart materials.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub smart_materials: Vec<SmartMaterial>,
    /// Saved smart masks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub smart_masks: Vec<SmartMask>,
    /// Saved inspection cameras.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cameras: Vec<NamedCamera>,
}

impl MaterialProject {
    pub fn empty() -> Self {
        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            mesh: None,
            texture_sets: Vec::new(),
            saved_poses: Vec::new(),
            active_texture_set: None,
            resources: Vec::new(),
            export_presets: Vec::new(),
            materials: Vec::new(),
            smart_materials: Vec::new(),
            smart_masks: Vec::new(),
            cameras: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), ProjectError> {
        if self.schema_version != PROJECT_SCHEMA_VERSION {
            return Err(ProjectError::UnsupportedSchemaVersion(self.schema_version));
        }

        let mut resource_ids = BTreeSet::new();
        for resource in &self.resources {
            if !resource_ids.insert(resource.id) {
                return Err(ProjectError::DuplicateResourceId(resource.id));
            }
        }

        let mut texture_set_ids = BTreeSet::new();
        let mut layer_ids = BTreeSet::new();

        for texture_set in &self.texture_sets {
            texture_set.resolution.validate()?;
            if !valid_height_strength(texture_set.height_strength) {
                return Err(ProjectError::InvalidTextureSetSettings(texture_set.id));
            }
            if let Some(export) = &texture_set.export {
                let preset_known = ExportPreset::built_in()
                    .iter()
                    .chain(&self.export_presets)
                    .any(|preset| preset.name == export.preset);
                if !preset_known || !EXPORT_FILE_FORMATS.contains(&export.format.as_str()) {
                    return Err(ProjectError::InvalidTextureSetSettings(texture_set.id));
                }
            }
            if !texture_set_ids.insert(texture_set.id) {
                return Err(ProjectError::DuplicateTextureSetId(texture_set.id));
            }
            if let Some(layer) = texture_set
                .layers
                .iter()
                .find(|layer| layer.is_group() && layer.group_depth() + 1 > MAX_GROUP_DEPTH)
            {
                return Err(ProjectError::GroupTooDeep(layer.id));
            }

            for layer in texture_set.walk_layers() {
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
                if !layer.has_valid_settings() || !layer.has_valid_evolutions() {
                    return Err(ProjectError::InvalidLayerSettings(layer.id));
                }
                if layer
                    .mask_anchor
                    .is_some_and(|anchor| !texture_set.anchor_is_valid(layer.id, anchor.layer))
                {
                    return Err(ProjectError::InvalidLayerSettings(layer.id));
                }
                if let Some(missing) = layer
                    .own_resources()
                    .find(|resource| !resource_ids.contains(resource))
                {
                    return Err(ProjectError::MissingResource(missing));
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

        let mut material_ids = BTreeSet::new();
        for material in &self.materials {
            material.validate()?;
            if material.id == MaterialId::UNASSIGNED || !material_ids.insert(material.id) {
                return Err(ProjectError::InvalidMaterialId(material.name.clone()));
            }
        }
        for texture_set in &self.texture_sets {
            for layer in texture_set.walk_layers() {
                if let Some(link) = &layer.material_link
                    && !material_ids.contains(&link.material)
                {
                    return Err(ProjectError::MissingMaterial {
                        layer: layer.id,
                        material: link.material,
                    });
                }
            }
        }
        for material in &self.smart_materials {
            material.validate()?;
            if let Some(missing) = material
                .requirements()
                .resources
                .into_iter()
                .find(|resource| !resource_ids.contains(resource))
            {
                return Err(ProjectError::MissingResource(missing));
            }
        }

        for mask in &self.smart_masks {
            mask.validate()?;
            if let Some(missing) = mask
                .requirements()
                .resources
                .into_iter()
                .find(|resource| !resource_ids.contains(resource))
            {
                return Err(ProjectError::MissingResource(missing));
            }
        }

        if self.cameras.len() > MAX_NAMED_CAMERAS {
            return Err(ProjectError::InvalidCamera(format!(
                "more than {MAX_NAMED_CAMERAS} cameras"
            )));
        }
        let mut camera_names = BTreeSet::new();
        for camera in &self.cameras {
            let name = camera.name.trim().to_lowercase();
            if name.is_empty() || !camera.camera.is_valid() || !camera_names.insert(name) {
                return Err(ProjectError::InvalidCamera(camera.name.clone()));
            }
        }

        let built_in_names: BTreeSet<String> = ExportPreset::built_in()
            .iter()
            .map(|preset| preset.name.trim().to_lowercase())
            .collect();
        let mut preset_names = BTreeSet::new();
        for preset in &self.export_presets {
            preset
                .validate()
                .map_err(|error| ProjectError::InvalidExportPreset {
                    name: preset.name.clone(),
                    error,
                })?;
            let name = preset.name.trim().to_lowercase();
            if built_in_names.contains(&name) || !preset_names.insert(name) {
                return Err(ProjectError::DuplicateExportPresetName(preset.name.clone()));
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
    #[error("layer {0:?} has an invalid fill, mask, filter, or projection")]
    InvalidLayerSettings(LayerId),
    #[error("layer group {0:?} is nested more than {MAX_GROUP_DEPTH} levels deep")]
    GroupTooDeep(LayerId),
    #[error("texture set {0:?} has an out-of-range height strength")]
    InvalidTextureSetSettings(TextureSetId),
    #[error("mesh scale or smoothing settings are out of range")]
    InvalidMeshSettings,
    #[error("pose id {0:?} is duplicated")]
    DuplicatePoseId(PoseId),
    #[error("pose name `{0}` is duplicated")]
    DuplicatePoseName(String),
    #[error("resource id {0:?} is duplicated")]
    DuplicateResourceId(ResourceId),
    #[error("a layer references resource {0:?}, which the project does not list")]
    MissingResource(ResourceId),
    #[error("export preset `{name}` is invalid: {error}")]
    InvalidExportPreset {
        name: String,
        error: ExportPresetError,
    },
    #[error("export preset name `{0}` is duplicated or taken by a built-in preset")]
    DuplicateExportPresetName(String),
    #[error("camera `{0}` is unnamed, duplicated, or out of range")]
    InvalidCamera(String),
    #[error("material `{0}` has a missing or duplicated id")]
    InvalidMaterialId(String),
    #[error(
        "layer {layer:?} is an instance of material {material:?}, which the project does not have"
    )]
    MissingMaterial {
        layer: LayerId,
        material: MaterialId,
    },
    #[error(transparent)]
    InvalidLibraryEntry(#[from] LibraryError),
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
            height_strength: DEFAULT_HEIGHT_STRENGTH,
            export: None,
        }
    }

    #[test]
    fn tc_0263_set_presets_and_mesh_edits_undo_redo_and_merge_slider_drags() {
        let mut project = MaterialProject::empty();
        project.texture_sets.push(texture_set(1));
        let mut history = EditHistory::default();

        // Texture-set properties.
        let before = project.texture_sets[0].properties();
        let after = TextureSetProperties {
            resolution: TextureResolution::new(4096, 4096).unwrap(),
            ..before.clone()
        };
        assert!(history.apply(
            &mut project,
            EditCommand::TextureSetProperties {
                texture_set_id: TextureSetId(1),
                before: Box::new(before.clone()),
                after: Box::new(after.clone()),
            },
        ));
        assert_eq!(project.texture_sets[0].resolution.width, 4096);
        // A command whose "before" no longer matches is refused.
        assert!(!history.apply(
            &mut project,
            EditCommand::TextureSetProperties {
                texture_set_id: TextureSetId(1),
                before: Box::new(before.clone()),
                after: Box::new(after.clone()),
            },
        ));
        assert!(history.undo(&mut project));
        assert_eq!(project.texture_sets[0].properties(), before);
        assert!(history.redo(&mut project));
        assert_eq!(project.texture_sets[0].properties(), after);

        // Slider drags merge into one step.
        let mut current = after.clone();
        for strength in [1.5, 2.0, 2.5] {
            let next = TextureSetProperties {
                height_strength: strength,
                ..current.clone()
            };
            assert!(history.apply_merging(
                &mut project,
                EditCommand::TextureSetProperties {
                    texture_set_id: TextureSetId(1),
                    before: Box::new(current.clone()),
                    after: Box::new(next.clone()),
                },
            ));
            current = next;
        }
        assert_eq!(project.texture_sets[0].height_strength, 2.5);
        assert!(history.undo(&mut project));
        assert_eq!(project.texture_sets[0].properties(), after, "one step");

        // Inserting and removing whole sets.
        let mut second = texture_set(2);
        second
            .layers
            .push(MaterialLayer::new_paint(LayerId(9), "Paint"));
        assert!(history.apply(
            &mut project,
            EditCommand::InsertTextureSet {
                index: 1,
                set: Box::new(second.clone()),
            },
        ));
        assert!(
            !history.apply(
                &mut project,
                EditCommand::InsertTextureSet {
                    index: 0,
                    set: Box::new(second.clone()),
                },
            ),
            "ids stay unique"
        );
        project.active_texture_set = Some(TextureSetId(2));
        assert!(history.apply(
            &mut project,
            EditCommand::RemoveTextureSet {
                index: 1,
                set: Box::new(second.clone()),
            },
        ));
        assert_eq!(project.texture_sets.len(), 1);
        assert_eq!(project.active_texture_set, Some(TextureSetId(1)));
        assert!(history.undo(&mut project));
        assert_eq!(project.texture_sets[1], second);
        assert!(history.undo(&mut project));
        assert_eq!(project.texture_sets.len(), 1);

        // Presets and the mesh reference.
        let presets = vec![ExportPreset::built_in().remove(0)];
        assert!(history.apply(
            &mut project,
            EditCommand::ExportPresets {
                before: Vec::new(),
                after: presets.clone(),
            },
        ));
        let mesh = MeshAsset {
            id: MeshId(3),
            source: "scan.fbx".to_owned(),
            scale: 2.0,
            smoothing: MeshSmoothing::default(),
            high_poly_source: None,
            content_hash: None,
            high_poly_hash: None,
        };
        let old_mesh = project.mesh.clone();
        assert!(history.apply(
            &mut project,
            EditCommand::Mesh {
                before: old_mesh.clone(),
                after: Some(mesh.clone()),
            },
        ));
        assert_eq!(project.mesh, Some(mesh));
        assert!(history.undo(&mut project));
        assert_eq!(project.mesh, old_mesh);
        assert!(history.undo(&mut project));
        assert!(project.export_presets.is_empty());

        // Commands serialize, so history can be saved.
        let command = EditCommand::Batch(vec![EditCommand::ExportPresets {
            before: Vec::new(),
            after: presets,
        }]);
        let json = serde_json::to_string(&command).unwrap();
        assert_eq!(serde_json::from_str::<EditCommand>(&json).unwrap(), command);
    }

    #[test]
    fn tc_0260_large_sets_preview_downscaled_with_matching_brush_radii() {
        let mut set = texture_set(1);
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Paint");
        layer.paint_strokes.push(PaintStroke::new(
            vec![glam::Vec2::splat(0.5)],
            64.0,
            [255, 0, 0, 255],
        ));
        let mut group = MaterialLayer::new_group(LayerId(2), "Group");
        let mut masked = MaterialLayer::new_paint(LayerId(3), "Masked");
        masked.mask_strokes.push(MaskStroke {
            points: vec![glam::Vec2::splat(0.25)],
            radius_pixels: 32.0,
            value: 0.0,
            opacity: 1.0,
            tip: BrushTip::ROUND,
            pressures: Vec::new(),
            camera: None,
        });
        group.children = Some(vec![masked]);
        set.layers = vec![layer, group];

        assert_eq!(set.preview_factor(4096), 1);
        assert!(set.downscaled(4096).is_none(), "2K fits as it is");
        set.resolution = TextureResolution::new(8192, 6000).unwrap();
        assert_eq!(set.preview_factor(4096), 2);
        assert_eq!(set.preview_factor(2048), 4);
        let preview = set.downscaled(4096).unwrap();
        assert_eq!(
            preview.resolution,
            TextureResolution::new(4096, 3000).unwrap()
        );
        assert_eq!(preview.layers[0].paint_strokes[0].radius_pixels, 32.0);
        let nested = preview.layer(LayerId(3)).unwrap();
        assert_eq!(nested.mask_strokes[0].radius_pixels, 16.0);
        // Positions are in UV space and stay put.
        assert_eq!(
            preview.layers[0].paint_strokes[0].points,
            set.layers[0].paint_strokes[0].points
        );
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
                height: DEFAULT_HEIGHT,
                ..PaintStroke::new(Vec::new(), 1.0, [0; 4])
            }],
            mask_strokes: vec![MaskStroke {
                points: vec![glam::Vec2::new(7.0 / 15.0, 8.0 / 15.0)],
                radius_pixels: 2.0,
                value: 0.0,
                opacity: 1.0,
                tip: BrushTip::ROUND,
                pressures: Vec::new(),
                camera: None,
            }],
            id_strokes: Vec::new(),
            id_mask_color: None,
            fill: None,
            mask_generator: None,
            polygon_mask: None,
            mask_bitmap: None,
            invert_final_mask: false,
            filters: Vec::new(),
            mask_filters: Vec::new(),
            mask_anchor: None,
            material_link: None,
            blend_mode: BlendMode::Normal,
            children: None,
            evolutions: Vec::new(),
            exposed: Vec::new(),
            channel_blend_modes: BTreeMap::new(),
            uv_tiles: BTreeSet::new(),
            filter_channels: BTreeSet::new(),
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
                tip: BrushTip::ROUND,
                pressures: Vec::new(),
            }],
            id_mask_color: None,
            fill: None,
            mask_generator: None,
            polygon_mask: None,
            mask_bitmap: None,
            invert_final_mask: false,
            filters: Vec::new(),
            mask_filters: Vec::new(),
            mask_anchor: None,
            material_link: None,
            blend_mode: BlendMode::Normal,
            children: None,
            evolutions: Vec::new(),
            exposed: Vec::new(),
            channel_blend_modes: BTreeMap::new(),
            uv_tiles: BTreeSet::new(),
            filter_channels: BTreeSet::new(),
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
                height: DEFAULT_HEIGHT,
                ..PaintStroke::new(Vec::new(), 1.0, [0; 4])
            }],
            mask_strokes: Vec::new(),
            id_strokes: vec![IdStroke {
                points: vec![glam::Vec2::new(3.0 / 7.0, 4.0 / 7.0)],
                radius_pixels: 1.0,
                color: [12, 180, 220],
                opacity: 1.0,
                tip: BrushTip::ROUND,
                pressures: Vec::new(),
            }],
            id_mask_color: Some([12, 180, 220]),
            fill: None,
            mask_generator: None,
            polygon_mask: None,
            mask_bitmap: None,
            invert_final_mask: false,
            filters: Vec::new(),
            mask_filters: Vec::new(),
            mask_anchor: None,
            material_link: None,
            blend_mode: BlendMode::Normal,
            children: None,
            evolutions: Vec::new(),
            exposed: Vec::new(),
            channel_blend_modes: BTreeMap::new(),
            uv_tiles: BTreeSet::new(),
            filter_channels: BTreeSet::new(),
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
            height: DEFAULT_HEIGHT,
            ..PaintStroke::new(Vec::new(), 1.0, [0; 4])
        };
        let mask = MaskStroke {
            points: vec![glam::Vec2::new(0.5, 0.5)],
            radius_pixels: 3.0,
            value: 0.25,
            opacity: 1.0,
            tip: BrushTip::ROUND,
            pressures: Vec::new(),
            camera: None,
        };
        let id = IdStroke {
            points: vec![glam::Vec2::new(0.75, 0.25)],
            radius_pixels: 4.0,
            color: [12, 180, 220],
            opacity: 1.0,
            tip: BrushTip::ROUND,
            pressures: Vec::new(),
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
            mask_bitmap: None,
            invert_final_mask: false,
            filters: Vec::new(),
            mask_filters: Vec::new(),
            mask_anchor: None,
            material_link: None,
            blend_mode: BlendMode::Normal,
            children: None,
            evolutions: Vec::new(),
            exposed: Vec::new(),
            channel_blend_modes: BTreeMap::new(),
            uv_tiles: BTreeSet::new(),
            filter_channels: BTreeSet::new(),
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
                height: DEFAULT_HEIGHT,
                ..PaintStroke::new(Vec::new(), 1.0, [0; 4])
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
            mask_bitmap: None,
            invert_final_mask: false,
            filters: Vec::new(),
            mask_filters: Vec::new(),
            mask_anchor: None,
            material_link: None,
            blend_mode: BlendMode::Normal,
            children: None,
            evolutions: Vec::new(),
            exposed: Vec::new(),
            channel_blend_modes: BTreeMap::new(),
            uv_tiles: BTreeSet::new(),
            filter_channels: BTreeSet::new(),
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
                mask_bitmap: None,
                invert_final_mask: false,
                filters: Vec::new(),
                mask_filters: Vec::new(),
                mask_anchor: None,
                material_link: None,
                blend_mode: BlendMode::Normal,
                children: None,
                evolutions: Vec::new(),
                exposed: Vec::new(),
                channel_blend_modes: BTreeMap::new(),
                uv_tiles: BTreeSet::new(),
                filter_channels: BTreeSet::new(),
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
                mask_bitmap: None,
                invert_final_mask: false,
                filters: Vec::new(),
                mask_filters: Vec::new(),
                mask_anchor: None,
                material_link: None,
                blend_mode: BlendMode::Normal,
                children: None,
                evolutions: Vec::new(),
                exposed: Vec::new(),
                channel_blend_modes: BTreeMap::new(),
                uv_tiles: BTreeSet::new(),
                filter_channels: BTreeSet::new(),
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
    fn tc_0237_named_cameras_validate_names_and_ranges() {
        let camera = |name: &str, zoom: f32| NamedCamera {
            name: name.to_owned(),
            camera: ViewportCamera {
                zoom,
                ..ViewportCamera::default()
            },
        };
        let mut project = MaterialProject {
            cameras: vec![camera("Front", 1.0), camera("Detail", 4.0)],
            ..MaterialProject::empty()
        };
        assert_eq!(project.validate(), Ok(()));
        let json = serde_json::to_string(&project).unwrap();
        assert_eq!(
            serde_json::from_str::<MaterialProject>(&json).unwrap(),
            project
        );

        for bad in [camera(" front ", 1.0), camera("", 1.0), camera("Far", 50.0)] {
            project.cameras.push(bad);
            assert!(matches!(
                project.validate(),
                Err(ProjectError::InvalidCamera(_))
            ));
            project.cameras.pop();
        }
        let mut wild = camera("Wild", 1.0);
        wild.camera.yaw = f32::NAN;
        project.cameras.push(wild);
        assert!(project.validate().is_err());
    }

    #[test]
    fn named_poses_interpolate_translation_scale_and_rotation() {
        let from = NamedPose {
            id: PoseId(1),
            name: "Rest".to_owned(),
            animation: None,
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
            animation: None,
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
            animation: None,
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
            animation: None,
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
            animation: None,
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
            ..ViewportLighting::default()
        }
        .face_intensity(normal);
        let rotated = ViewportLighting {
            rotation: std::f32::consts::PI,
            ambient_fill: 0.2,
            ..ViewportLighting::default()
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
            resources: None,
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
                height: DEFAULT_HEIGHT,
                ..LayerFill::default()
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
            resources: None,
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
            set.bake_channel(MaterialChannel::AmbientOcclusion, context),
            Err(BakeError::UnsupportedChannel(
                MaterialChannel::AmbientOcclusion
            ))
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
            height: DEFAULT_HEIGHT,
            ..PaintStroke::new(Vec::new(), 1.0, [0; 4])
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
                slot: LayerSlot {
                    parent: None,
                    index: 1
                },
                layer: Box::new(fill.clone()),
            }
        ));
        assert!(history.apply(
            &mut project,
            EditCommand::MoveLayer {
                texture_set_id: set_id,
                from: LayerSlot {
                    parent: None,
                    index: 1
                },
                to: LayerSlot {
                    parent: None,
                    index: 0
                },
            }
        ));
        assert_eq!(project.texture_sets[0].layers[0].id, LayerId(2));
        assert!(history.apply(
            &mut project,
            EditCommand::RemoveLayer {
                texture_set_id: set_id,
                slot: LayerSlot {
                    parent: None,
                    index: 0
                },
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
                slot: LayerSlot {
                    parent: None,
                    index: 0
                },
                layer: Box::new(MaterialLayer::new_paint(LayerId(1), "Dup")),
            }
        ));
        assert!(project.validate().is_ok());
    }

    #[test]
    fn tc_0048_vertex_tangents_follow_plus_u_with_plus_v_bitangents() {
        let plane = PrimitivePreset::Plane.build_mesh();
        let normals = vertex_normals(&plane, 60.0);
        let tangents = vertex_tangents(&plane, &normals);

        for (normal, tangent) in normals.iter().zip(&tangents) {
            assert!((*normal - Vec3::Y).length() < 1e-5);
            assert!(
                (tangent.truncate() - Vec3::X).length() < 1e-5,
                "tangent {tangent}"
            );
            // +V runs toward -Z on the built-in plane.
            let bitangent = normal.cross(tangent.truncate()) * tangent.w;
            assert!(
                (bitangent - Vec3::NEG_Z).length() < 1e-5,
                "bitangent {bitangent}"
            );
        }
    }

    #[test]
    fn tc_0049_painted_height_bends_the_normal_channel_outward() {
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Bump");
        layer.channels = BTreeSet::from([MaterialChannel::Height]);
        layer.paint_strokes.push(PaintStroke {
            points: vec![glam::Vec2::new(0.5, 0.5)],
            radius_pixels: 6.0,
            color: [255, 255, 255, 255],
            opacity: 1.0,
            roughness: DEFAULT_ROUGHNESS,
            metallic: 0.0,
            height: 1.0,
            ..PaintStroke::new(Vec::new(), 1.0, [0; 4])
        });
        let mut set = small_set(17, layer);
        // A 17-texel map needs maximum force for a clearly visible tilt.
        set.height_strength = MAX_HEIGHT_STRENGTH;

        let height = set
            .bake_channel(MaterialChannel::Height, BakeContext::default())
            .unwrap();
        assert_eq!(height.pixel(8, 8).unwrap()[0], 255);
        assert_eq!(height.pixel(0, 0).unwrap()[0], 128);

        let normal = set
            .bake_channel(MaterialChannel::Normal, BakeContext::default())
            .unwrap();
        // Flat away from the bump.
        assert_eq!(normal.pixel(0, 0), Some([128, 128, 255, 255]));
        // Right of the bump tilts toward +U, above it (smaller row) toward +V,
        // left toward -U.
        assert!(normal.pixel(11, 8).unwrap()[0] > 150);
        assert!(normal.pixel(8, 5).unwrap()[1] > 150);
        assert!(normal.pixel(5, 8).unwrap()[0] < 106);

        let mut flat = set.clone();
        flat.height_strength = 0.0;
        let unchanged = flat
            .bake_channel(MaterialChannel::Normal, BakeContext::default())
            .unwrap();
        assert!(
            unchanged
                .rgba8
                .chunks(4)
                .all(|pixel| pixel == [128, 128, 255, 255])
        );
    }

    #[test]
    fn tc_0050_material_bake_packs_occlusion_roughness_and_metallic() {
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Metal",
            LayerFill {
                color: [200, 100, 50, 255],
                roughness: 0.2,
                metallic: 1.0,
                height: DEFAULT_HEIGHT,
                ..LayerFill::default()
            },
        );
        layer.opacity = 1.0;
        let set = small_set(8, layer);

        let textures = set.bake_material(BakeContext::default()).unwrap();

        assert_eq!(textures.base_color.pixel(3, 3), Some([200, 100, 50, 255]));
        // No mesh maps: occlusion is open (255).
        assert_eq!(textures.orm.pixel(3, 3), Some([255, 51, 255, 255]));
        assert_eq!(textures.normal.pixel(3, 3), Some([128, 128, 255, 255]));

        // Unpainted texels show (and export) the neutral default, opaque.
        let mut empty = set.clone();
        empty.layers.clear();
        let neutral = empty.bake_material(BakeContext::default()).unwrap();
        assert_eq!(neutral.base_color.pixel(0, 0), Some([128, 128, 128, 255]));
        assert_eq!(neutral.orm.pixel(0, 0), Some([255, 128, 0, 255]));

        let mut invalid = set.clone();
        invalid.height_strength = f32::NAN;
        assert_eq!(
            invalid.bake_material(BakeContext::default()),
            Err(BakeError::InvalidLayerSettings)
        );
    }

    fn uv_of_pixel(x: u32, y: u32, size: u32) -> glam::Vec2 {
        glam::Vec2::new(
            x as f32 / (size - 1) as f32,
            1.0 - y as f32 / (size - 1) as f32,
        )
    }

    #[test]
    fn tc_0060_eraser_removes_paint_from_its_own_layer_only() {
        let size = 16;
        let mut under = MaterialLayer::new_fill(
            LayerId(1),
            "Under",
            LayerFill {
                color: [0, 0, 255, 255],
                ..LayerFill::default()
            },
        );
        under.channels = BTreeSet::from([MaterialChannel::BaseColor]);
        let mut paint = MaterialLayer::new_paint(LayerId(2), "Paint");
        let centre = uv_of_pixel(8, 8, size);
        paint.paint_strokes.push(PaintStroke::new(
            vec![uv_of_pixel(2, 8, size), uv_of_pixel(14, 8, size)],
            3.0,
            [255, 0, 0, 255],
        ));
        paint.paint_strokes.push(PaintStroke {
            erase: true,
            tip: BrushTip {
                hardness: 1.0,
                ..BrushTip::ROUND
            },
            ..PaintStroke::new(vec![centre], 2.0, [0; 4])
        });
        let mut set = small_set(size, under);
        set.layers.push(paint);

        let baked = set.bake_base_color().unwrap();
        assert_eq!(
            baked.pixel(8, 8),
            Some([0, 0, 255, 255]),
            "erased to the fill"
        );
        assert_eq!(
            baked.pixel(3, 8),
            Some([255, 0, 0, 255]),
            "paint elsewhere stays"
        );

        // Painting after erasing shows again.
        set.layers[1]
            .paint_strokes
            .push(PaintStroke::new(vec![centre], 2.0, [0, 255, 0, 255]));
        let baked = set.bake_base_color().unwrap();
        assert_eq!(baked.pixel(8, 8), Some([0, 255, 0, 255]));
    }

    #[test]
    fn tc_0061_groups_composite_children_through_group_opacity_and_mask() {
        let size = 8;
        let red = MaterialLayer::new_fill(
            LayerId(2),
            "Red",
            LayerFill {
                color: [255, 0, 0, 255],
                roughness: 0.2,
                ..LayerFill::default()
            },
        );
        let mut group = MaterialLayer::new_group(LayerId(1), "Group");
        group.children = Some(vec![red]);
        group.opacity = 0.5;
        // Mask out the left half of the group.
        group.mask_strokes.push(MaskStroke {
            points: vec![glam::Vec2::new(0.0, 0.5)],
            radius_pixels: 3.5,
            value: 0.0,
            opacity: 1.0,
            tip: BrushTip {
                shape: BrushShape::Square,
                hardness: 1.0,
                ..BrushTip::ROUND
            },
            pressures: Vec::new(),
            camera: None,
        });
        let set = small_set(size, group);

        let roughness = set
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap();
        // Right half: halfway between the default 0.5 and 0.2.
        assert!(roughness.pixel(7, 4).unwrap()[0].abs_diff(89) <= 1);
        assert_eq!(
            roughness.pixel(0, 4).unwrap()[0],
            128,
            "masked by the group"
        );

        let mut hidden = set.clone();
        hidden.layers[0].visible = false;
        let roughness = hidden
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap();
        assert_eq!(
            roughness.pixel(7, 4).unwrap()[0],
            128,
            "hidden group hides children"
        );
    }

    #[test]
    fn tc_0062_layers_move_into_and_out_of_groups_with_undo() {
        let mut set = texture_set(1);
        set.layers = vec![
            MaterialLayer::new_paint(LayerId(1), "Paint"),
            MaterialLayer::new_group(LayerId(2), "Group"),
        ];
        let mut project = MaterialProject {
            texture_sets: vec![set],
            ..MaterialProject::empty()
        };
        let set_id = TextureSetId(1);
        let mut history = EditHistory::default();
        let top = |index| LayerSlot {
            parent: None,
            index,
        };
        let in_group = |index| LayerSlot {
            parent: Some(LayerId(2)),
            index,
        };

        assert!(history.apply(
            &mut project,
            EditCommand::MoveLayer {
                texture_set_id: set_id,
                from: top(0),
                to: in_group(0),
            }
        ));
        let set = &project.texture_sets[0];
        assert_eq!(set.layers.len(), 1);
        assert_eq!(set.slot_of(LayerId(1)), Some(in_group(0)));
        assert_eq!(set.depth_of(LayerId(1)), 1);
        assert!(project.validate().is_ok());

        // A group cannot move into itself.
        assert!(!history.apply(
            &mut project,
            EditCommand::MoveLayer {
                texture_set_id: set_id,
                from: top(0),
                to: in_group(0),
            }
        ));
        // Inserting into a plain layer fails.
        assert!(!history.apply(
            &mut project,
            EditCommand::InsertLayer {
                texture_set_id: set_id,
                slot: LayerSlot {
                    parent: Some(LayerId(1)),
                    index: 0,
                },
                layer: Box::new(MaterialLayer::new_paint(LayerId(9), "Nope")),
            }
        ));
        assert!(history.undo(&mut project));
        assert_eq!(project.texture_sets[0].slot_of(LayerId(1)), Some(top(0)));
        assert!(
            project.texture_sets[0]
                .layer(LayerId(2))
                .unwrap()
                .children
                .as_ref()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn tc_0065_layer_and_mask_filters_change_the_composite() {
        let size = 8;
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Rough",
            LayerFill {
                roughness: 0.25,
                ..LayerFill::default()
            },
        );
        layer.filters = vec![LayerFilter::Invert];
        let set = small_set(size, layer.clone());
        let roughness = set
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap();
        assert_eq!(roughness.pixel(3, 3).unwrap()[0], 191, "inverted 0.25");

        layer.filters.clear();
        layer.mask_strokes.push(MaskStroke {
            points: vec![glam::Vec2::new(0.5, 0.5)],
            radius_pixels: 100.0,
            value: 0.0,
            opacity: 1.0,
            tip: BrushTip {
                hardness: 1.0,
                ..BrushTip::ROUND
            },
            pressures: Vec::new(),
            camera: None,
        });
        // An inverted all-black mask shows the layer everywhere again.
        layer.mask_filters = vec![LayerFilter::Invert];
        let set = small_set(size, layer);
        let roughness = set
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap();
        assert_eq!(roughness.pixel(3, 3).unwrap()[0], 64);
    }

    #[test]
    fn tc_0067_final_mask_inversion_runs_after_filters_and_is_serialized() {
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Rough",
            LayerFill {
                roughness: 0.25,
                ..LayerFill::default()
            },
        );
        layer.mask_strokes.push(MaskStroke {
            points: vec![glam::Vec2::new(0.5, 0.5)],
            radius_pixels: 100.0,
            value: 0.0,
            opacity: 1.0,
            tip: BrushTip {
                hardness: 1.0,
                ..BrushTip::ROUND
            },
            pressures: Vec::new(),
            camera: None,
        });
        layer.mask_filters = vec![LayerFilter::Invert];
        layer.invert_final_mask = true;
        let set = small_set(8, layer.clone());
        let roughness = set
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap();
        assert_eq!(
            roughness.pixel(3, 3).unwrap()[0],
            128,
            "filter makes the intermediate mask white, final inversion hides the fill"
        );

        let serialized = serde_json::to_string(&layer).unwrap();
        let restored: MaterialLayer = serde_json::from_str(&serialized).unwrap();
        assert!(restored.invert_final_mask);
        let mut changed = restored;
        changed.apply_settings({
            let mut settings = changed.settings();
            settings.invert_final_mask = false;
            settings
        });
        assert!(!changed.invert_final_mask);

        let mut legacy = serde_json::to_value(layer).unwrap();
        legacy.as_object_mut().unwrap().remove("invert_final_mask");
        let restored: MaterialLayer = serde_json::from_value(legacy).unwrap();
        assert!(!restored.invert_final_mask, "older projects default to off");
    }

    #[test]
    fn tc_0066_opacity_and_emissive_channels_bake_into_the_material() {
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Glow",
            LayerFill {
                emissive: [255, 128, 0],
                material_opacity: 0.25,
                ..LayerFill::default()
            },
        );
        layer.channels = BTreeSet::from([MaterialChannel::Emissive, MaterialChannel::Opacity]);
        let set = small_set(4, layer);

        let textures = set.bake_material(BakeContext::default()).unwrap();
        assert_eq!(textures.emissive.pixel(1, 1), Some([255, 128, 0, 255]));
        assert_eq!(textures.base_color.pixel(1, 1).unwrap()[3], 64);
        assert!(textures.has_emission() && textures.has_transparency());
        let opacity = set
            .bake_channel(MaterialChannel::Opacity, BakeContext::default())
            .unwrap();
        assert_eq!(opacity.pixel(0, 0), Some([64, 64, 64, 255]));

        // Nothing painted: black emission, opaque base colour.
        let plain = small_set(4, MaterialLayer::new_paint(LayerId(2), "Empty"));
        let textures = plain.bake_material(BakeContext::default()).unwrap();
        assert!(!textures.has_emission() && !textures.has_transparency());
    }

    #[test]
    fn tc_0072_fill_textures_and_bitmap_masks_read_project_resources() {
        use std::sync::Arc;

        let size = 8;
        // Left half red, right half blue, as an 8x1 bitmap.
        let mut pixels = [255, 0, 0, 255].repeat(4);
        pixels.extend([0, 0, 255, 255].repeat(4));
        let image = TextureImage::new(8, 1, pixels).unwrap();
        let resources = ResourceImages::from([(ResourceId(7), Arc::new(image))]);
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Textured",
            LayerFill {
                textures: FillTextures {
                    base_color: Some(ResourceId(7)),
                    ..FillTextures::default()
                },
                ..LayerFill::default()
            },
        );
        layer.channels = BTreeSet::from([MaterialChannel::BaseColor]);
        let set = small_set(size, layer);
        let context = BakeContext {
            resources: Some(&resources),
            ..BakeContext::default()
        };
        let baked = set
            .bake_channel(MaterialChannel::BaseColor, context)
            .unwrap();
        assert_eq!(baked.pixel(1, 0), Some([255, 0, 0, 255]));
        assert_eq!(baked.pixel(6, 0), Some([0, 0, 255, 255]));
        assert_eq!(
            set.bake_channel(MaterialChannel::BaseColor, BakeContext::default()),
            Err(BakeError::MissingResource(ResourceId(7)))
        );
        assert_eq!(set.referenced_resources(), BTreeSet::from([ResourceId(7)]));

        // As an inverted mask, the red column's luminance (0.21) leaves 0.79
        // and the blue column's (0.07) leaves 0.93.
        let mut masked = MaterialLayer::new_fill(
            LayerId(2),
            "Masked",
            LayerFill {
                color: [255, 255, 255, 255],
                ..LayerFill::default()
            },
        );
        masked.channels = BTreeSet::from([MaterialChannel::BaseColor]);
        masked.mask_bitmap = Some(BitmapMask {
            resource: ResourceId(7),
            projection: TextureProjection::default(),
            invert: true,
        });
        let set = small_set(size, masked);
        let baked = set
            .bake_channel(MaterialChannel::BaseColor, context)
            .unwrap();
        let (left, right) = (baked.pixel(1, 0).unwrap()[3], baked.pixel(6, 0).unwrap()[3]);
        assert!(left.abs_diff(201) <= 1, "inverted red luminance: {left}");
        assert!(right.abs_diff(237) <= 1, "inverted blue luminance: {right}");

        // The project must list every resource its layers reference.
        let mut project = MaterialProject {
            texture_sets: vec![set],
            ..MaterialProject::empty()
        };
        assert_eq!(
            project.validate(),
            Err(ProjectError::MissingResource(ResourceId(7)))
        );
        project.resources.push(ProjectResource::new(
            ResourceId(7),
            "Stripes",
            "stripes.png",
        ));
        assert!(project.validate().is_ok());
    }

    #[test]
    fn tc_0150_anchor_masks_read_the_stack_up_to_an_earlier_layer() {
        use glam::Vec2;

        // Layer 1 paints red in the middle; layer 2 fills roughness 0 only
        // where layer 1's Base Color is bright.
        let mut paint = MaterialLayer::new_paint(LayerId(1), "Scratches");
        paint.channels = BTreeSet::from([MaterialChannel::BaseColor]);
        paint.paint_strokes.push(PaintStroke::new(
            vec![Vec2::new(0.2, 0.5), Vec2::new(0.8, 0.5)],
            3.0,
            [255, 255, 255, 255],
        ));
        paint.paint_strokes[0].tip.hardness = 1.0;
        let mut gloss = MaterialLayer::new_fill(
            LayerId(2),
            "Gloss in scratches",
            LayerFill {
                roughness: 0.0,
                ..LayerFill::default()
            },
        );
        gloss.channels = BTreeSet::from([MaterialChannel::Roughness]);
        gloss.mask_anchor = Some(AnchorMask::new(LayerId(1), MaterialChannel::BaseColor));
        let mut set = small_set(32, paint);
        set.layers.push(gloss);

        let roughness = set
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap();
        assert_eq!(roughness.sample_uv(Vec2::new(0.5, 0.5)).unwrap()[0], 0);
        assert_eq!(roughness.sample_uv(Vec2::new(0.5, 0.1)).unwrap()[0], 128);

        // Inverting flips where the fill lands.
        let mut inverted = set.clone();
        if let Some(anchor) = &mut inverted.layers[1].mask_anchor {
            anchor.invert = true;
        }
        let roughness = inverted
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap();
        assert_eq!(roughness.sample_uv(Vec2::new(0.5, 0.5)).unwrap()[0], 128);
        assert_eq!(roughness.sample_uv(Vec2::new(0.5, 0.1)).unwrap()[0], 0);

        // The truncated stack stops at the anchor.
        let stack = set.truncated_through(LayerId(1)).unwrap();
        assert_eq!(stack.layers.len(), 1);

        // Anchors must point at an earlier layer that is not an ancestor.
        assert!(set.anchor_is_valid(LayerId(2), LayerId(1)));
        assert!(!set.anchor_is_valid(LayerId(1), LayerId(2)), "later layer");
        assert!(!set.anchor_is_valid(LayerId(2), LayerId(2)), "itself");
        assert!(!set.anchor_is_valid(LayerId(2), LayerId(9)), "missing");
        let mut group = MaterialLayer::new_group(LayerId(3), "Group");
        group.children = Some(vec![set.layers[1].clone()]);
        let mut nested = set.clone();
        nested.layers[1] = group;
        if let Some(children) = &mut nested.layers[1].children {
            children[0].mask_anchor = Some(AnchorMask::new(LayerId(3), MaterialChannel::Roughness));
        }
        assert!(!nested.anchor_is_valid(LayerId(2), LayerId(3)), "ancestor");

        // Projects reject a reordered anchor; the bake refuses it too.
        let mut project = MaterialProject {
            texture_sets: vec![set.clone()],
            ..MaterialProject::empty()
        };
        project.validate().unwrap();
        project.texture_sets[0].layers.swap(0, 1);
        assert_eq!(
            project.validate(),
            Err(ProjectError::InvalidLayerSettings(LayerId(2)))
        );
        assert_eq!(
            project.texture_sets[0]
                .bake_channel(MaterialChannel::Roughness, BakeContext::default()),
            Err(BakeError::InvalidLayerSettings)
        );
    }

    #[test]
    fn tc_0151_flattened_groups_composite_like_their_source() {
        use std::sync::Arc;

        // A blue base, then a 60% group holding a metal fill and a stroke.
        let base = MaterialLayer::new_fill(
            LayerId(1),
            "Base",
            LayerFill {
                color: [20, 40, 200, 255],
                roughness: 0.8,
                ..LayerFill::default()
            },
        );
        let metal = MaterialLayer::new_fill(
            LayerId(3),
            "Metal",
            LayerFill {
                color: [200, 60, 30, 255],
                roughness: 0.2,
                metallic: 1.0,
                ..LayerFill::default()
            },
        );
        let mut scratch = MaterialLayer::new_paint(LayerId(4), "Scratch");
        scratch.paint_strokes.push(PaintStroke::new(
            vec![glam::Vec2::new(0.1, 0.3), glam::Vec2::new(0.9, 0.6)],
            2.5,
            [250, 250, 250, 255],
        ));
        let mut group = MaterialLayer::new_group(LayerId(2), "Paint job");
        group.opacity = 0.6;
        group.children = Some(vec![metal, scratch]);
        let mut set = small_set(24, base);
        set.layers.push(group);

        let flattened = set
            .flatten_layer(LayerId(2), BakeContext::default())
            .unwrap();
        let channels: Vec<MaterialChannel> = flattened.channels.iter().map(|(c, _)| *c).collect();
        assert_eq!(
            channels,
            [
                MaterialChannel::BaseColor,
                MaterialChannel::Roughness,
                MaterialChannel::Metallic,
            ],
            "only channels the children write"
        );
        // Coverage is the group opacity.
        assert_eq!(flattened.channels[1].1.rgba8[3], 153);

        let ids: Vec<ResourceId> = (0..channels.len() as u64)
            .map(|i| ResourceId(100 + i))
            .collect();
        let images: ResourceImages = ids
            .iter()
            .zip(&flattened.channels)
            .map(|(id, (_, image))| (*id, Arc::new(image.clone())))
            .collect();
        let mut replaced = set.clone();
        replaced.layers[1] = flattened.to_fill_layer(LayerId(9), "Flattened", &ids);
        let context = BakeContext {
            resources: Some(&images),
            ..BakeContext::default()
        };
        for channel in [
            MaterialChannel::BaseColor,
            MaterialChannel::Roughness,
            MaterialChannel::Metallic,
        ] {
            let original = set.bake_channel(channel, BakeContext::default()).unwrap();
            let rebuilt = replaced.bake_channel(channel, context).unwrap();
            let worst = original
                .rgba8
                .iter()
                .zip(&rebuilt.rgba8)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(worst <= 2, "{channel:?} differs by {worst}");
        }
        assert_eq!(
            set.flatten_layer(LayerId(77), BakeContext::default()),
            Err(BakeError::InvalidLayerSettings)
        );
    }

    #[test]
    fn tc_0244_pass_through_groups_blend_children_onto_the_stack_beneath() {
        let base = MaterialLayer::new_fill(
            LayerId(1),
            "Base",
            LayerFill {
                roughness: 0.8,
                ..LayerFill::default()
            },
        );
        let mut darken = MaterialLayer::new_fill(
            LayerId(2),
            "Darken",
            LayerFill {
                color: [255, 0, 0, 255],
                roughness: 0.5,
                ..LayerFill::default()
            },
        );
        darken.blend_mode = BlendMode::Multiply;
        let mut group = MaterialLayer::new_group(LayerId(3), "Group");
        group.children = Some(vec![darken]);
        let mut set = small_set(4, base);
        set.layers.push(group);
        let roughness = |set: &TextureSet| {
            set.bake_channel(MaterialChannel::Roughness, BakeContext::default())
                .unwrap()
                .rgba8[0]
        };

        // Isolated, the Multiply child has nothing beneath it inside the
        // group, so it shows as itself: 0.5.
        assert_eq!(roughness(&set), compositing::unit_to_byte(0.5));
        // Passed through, it multiplies the base: 0.8 × 0.5.
        set.layers[1].blend_mode = BlendMode::PassThrough;
        assert_eq!(roughness(&set), compositing::unit_to_byte(0.4));
        // Group opacity fades the passed-through result in.
        set.layers[1].opacity = 0.5;
        assert_eq!(roughness(&set), compositing::unit_to_byte(0.6));

        // Over empty Base Color the colour is not darkened by the fade.
        let mut lone = set.clone();
        lone.layers.remove(0);
        if let Some(child) = lone.layers[0].children.as_mut() {
            child[0].blend_mode = BlendMode::Normal;
        }
        let color = lone.bake_base_color().unwrap();
        assert_eq!(color.rgba8[..4], [255, 0, 0, 128]);

        // Only groups may pass through.
        let mut paint = MaterialLayer::new_paint(LayerId(9), "Paint");
        paint.blend_mode = BlendMode::PassThrough;
        assert!(!paint.has_valid_settings());
        assert!(set.layers[1].has_valid_settings());
    }

    #[test]
    fn tc_0157_blend_modes_combine_layers_with_the_stack_beneath() {
        use std::sync::Arc;

        // Separable formulas.
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(close(BlendMode::Multiply.apply(0.5, 0.5), 0.25));
        assert!(close(BlendMode::Screen.apply(0.5, 0.5), 0.75));
        assert!(close(BlendMode::Overlay.apply(0.25, 0.5), 0.25));
        assert!(close(BlendMode::Overlay.apply(0.75, 0.5), 0.75));
        assert!(close(BlendMode::SoftLight.apply(0.3, 0.5), 0.3));
        assert!(close(BlendMode::Difference.apply(0.2, 0.7), 0.5));
        assert!(close(BlendMode::Add.apply(0.7, 0.6), 1.0));
        assert!(close(BlendMode::Subtract.apply(0.2, 0.6), 0.0));
        assert!(close(BlendMode::Darken.apply(0.2, 0.6), 0.2));
        assert!(close(BlendMode::Lighten.apply(0.2, 0.6), 0.6));
        for mode in BlendMode::ALL {
            assert!(close(mode.apply(0.0, 0.0), 0.0) || mode == BlendMode::Screen);
        }

        // A red Multiply fill over mid grey keeps only the red; roughness
        // 0.5 multiplied by 0.5 gives 0.25.
        let base = MaterialLayer::new_fill(
            LayerId(1),
            "Grey",
            LayerFill {
                color: [128, 128, 128, 255],
                roughness: 0.5,
                ..LayerFill::default()
            },
        );
        let mut tint = MaterialLayer::new_fill(
            LayerId(2),
            "Tint",
            LayerFill {
                color: [255, 0, 0, 255],
                roughness: 0.5,
                ..LayerFill::default()
            },
        );
        tint.blend_mode = BlendMode::Multiply;
        let mut set = small_set(8, base);
        set.layers.push(tint.clone());
        let context = BakeContext::default();
        let color = set.bake_base_color().unwrap();
        assert_eq!(&color.pixel(2, 2).unwrap()[..3], &[128, 0, 0]);
        let roughness = set
            .bake_channel(MaterialChannel::Roughness, context)
            .unwrap();
        assert_eq!(roughness.pixel(2, 2).unwrap()[0], 64);

        // Screen lightens instead.
        set.layers[1].blend_mode = BlendMode::Screen;
        let color = set.bake_base_color().unwrap();
        assert_eq!(&color.pixel(2, 2).unwrap()[..3], &[255, 128, 128]);

        // Over nothing, a blended layer shows as itself.
        let alone = small_set(8, tint);
        let color = alone.bake_base_color().unwrap();
        assert_eq!(color.pixel(2, 2).unwrap(), [255, 0, 0, 255]);

        // Flattening keeps the mode, so the rebuilt stack matches exactly.
        set.layers[1].blend_mode = BlendMode::Overlay;
        let flattened = set.flatten_layer(LayerId(2), context).unwrap();
        assert_eq!(flattened.blend_mode, BlendMode::Overlay);
        let ids: Vec<ResourceId> = (0..flattened.channels.len() as u64)
            .map(|index| ResourceId(50 + index))
            .collect();
        let images: ResourceImages = ids
            .iter()
            .zip(&flattened.channels)
            .map(|(id, (_, image))| (*id, Arc::new(image.clone())))
            .collect();
        let mut rebuilt = set.clone();
        rebuilt.layers[1] = flattened.to_fill_layer(LayerId(9), "Flat", &ids);
        let with_images = BakeContext {
            resources: Some(&images),
            ..context
        };
        assert_eq!(
            rebuilt
                .bake_channel(MaterialChannel::BaseColor, with_images)
                .unwrap(),
            set.bake_channel(MaterialChannel::BaseColor, context)
                .unwrap()
        );

        // The mode is a layer setting, so history covers it.
        let mut settings = set.layers[1].settings();
        assert_eq!(settings.blend_mode, BlendMode::Overlay);
        settings.blend_mode = BlendMode::Normal;
        set.layers[1].apply_settings(settings);
        assert_eq!(set.layers[1].blend_mode, BlendMode::Normal);
    }

    #[test]
    fn tc_0300_layer_filters_can_be_limited_to_chosen_channels() {
        let mut fill = MaterialLayer::new_fill(
            LayerId(1),
            "Fill",
            LayerFill {
                color: [200, 100, 50, 255],
                roughness: 0.25,
                ..LayerFill::default()
            },
        );
        fill.filters = vec![LayerFilter::Invert];
        let set = small_set(8, fill.clone());
        let context = BakeContext::default();
        let pixel = |set: &TextureSet, channel| {
            set.bake_channel(channel, context)
                .unwrap()
                .pixel(2, 2)
                .unwrap()
        };
        // Unscoped, the filter inverts every channel.
        assert_eq!(
            &pixel(&set, MaterialChannel::BaseColor)[..3],
            &[55, 155, 205]
        );
        assert_eq!(pixel(&set, MaterialChannel::Roughness)[0], 191);
        // Limited to Roughness, colour is untouched.
        fill.filter_channels = BTreeSet::from([MaterialChannel::Roughness]);
        let scoped = small_set(8, fill.clone());
        assert_eq!(
            &pixel(&scoped, MaterialChannel::BaseColor)[..3],
            &[200, 100, 50]
        );
        assert_eq!(pixel(&scoped, MaterialChannel::Roughness)[0], 191);
        assert!(fill.filters_for(MaterialChannel::BaseColor).is_empty());
        assert_eq!(fill.filters_for(MaterialChannel::Roughness).len(), 1);
        // The scope is a layer setting.
        assert_eq!(fill.settings().filter_channels.len(), 1);
    }

    #[test]
    fn tc_0287_channel_blend_modes_override_the_layer_mode_per_channel() {
        use std::sync::Arc;

        let base = MaterialLayer::new_fill(
            LayerId(1),
            "Grey",
            LayerFill {
                color: [128, 128, 128, 255],
                roughness: 0.5,
                ..LayerFill::default()
            },
        );
        let mut tint = MaterialLayer::new_fill(
            LayerId(2),
            "Tint",
            LayerFill {
                color: [255, 0, 0, 255],
                roughness: 0.5,
                ..LayerFill::default()
            },
        );
        // Screen everywhere, except Multiply for Base Color.
        tint.blend_mode = BlendMode::Screen;
        tint.channel_blend_modes =
            BTreeMap::from([(MaterialChannel::BaseColor, BlendMode::Multiply)]);
        assert_eq!(
            tint.blend_mode_for(MaterialChannel::BaseColor),
            BlendMode::Multiply
        );
        assert_eq!(
            tint.blend_mode_for(MaterialChannel::Roughness),
            BlendMode::Screen
        );
        let mut set = small_set(8, base);
        set.layers.push(tint);
        let context = BakeContext::default();
        let color = set.bake_base_color().unwrap();
        assert_eq!(&color.pixel(2, 2).unwrap()[..3], &[128, 0, 0], "multiplied");
        let roughness = set
            .bake_channel(MaterialChannel::Roughness, context)
            .unwrap();
        assert_eq!(roughness.pixel(2, 2).unwrap()[0], 191, "screened: 0.75");

        // A Normal channel mode turns blending off for that channel only.
        set.layers[1]
            .channel_blend_modes
            .insert(MaterialChannel::Roughness, BlendMode::Normal);
        let roughness = set
            .bake_channel(MaterialChannel::Roughness, context)
            .unwrap();
        assert_eq!(roughness.pixel(2, 2).unwrap()[0], 128);

        // Flattening keeps the channel modes, so the rebuilt stack matches.
        let flattened = set.flatten_layer(LayerId(2), context).unwrap();
        assert_eq!(
            flattened.channel_blend_modes,
            set.layers[1].channel_blend_modes
        );
        let ids: Vec<ResourceId> = (0..flattened.channels.len() as u64)
            .map(|index| ResourceId(50 + index))
            .collect();
        let images: ResourceImages = ids
            .iter()
            .zip(&flattened.channels)
            .map(|(id, (_, image))| (*id, Arc::new(image.clone())))
            .collect();
        let mut rebuilt = set.clone();
        rebuilt.layers[1] = flattened.to_fill_layer(LayerId(9), "Flat", &ids);
        let with_images = BakeContext {
            resources: Some(&images),
            ..context
        };
        for channel in [MaterialChannel::BaseColor, MaterialChannel::Roughness] {
            assert_eq!(
                rebuilt.bake_channel(channel, with_images).unwrap(),
                set.bake_channel(channel, context).unwrap(),
                "{channel:?}"
            );
        }

        // Validation: no pass-through per channel, none on pass-through
        // groups; the modes round-trip and are layer settings.
        let mut project = MaterialProject {
            texture_sets: vec![set.clone()],
            ..MaterialProject::empty()
        };
        project.validate().unwrap();
        let decoded: MaterialProject =
            serde_json::from_str(&serde_json::to_string(&project).unwrap()).unwrap();
        assert_eq!(decoded, project);
        project.texture_sets[0].layers[1]
            .channel_blend_modes
            .insert(MaterialChannel::Metallic, BlendMode::PassThrough);
        assert_eq!(
            project.validate(),
            Err(ProjectError::InvalidLayerSettings(LayerId(2)))
        );
        let mut group = MaterialLayer::new_group(LayerId(5), "Group");
        group.blend_mode = BlendMode::PassThrough;
        group.channel_blend_modes =
            BTreeMap::from([(MaterialChannel::BaseColor, BlendMode::Multiply)]);
        assert!(!group.has_valid_settings());
        let settings = set.layers[1].settings();
        assert_eq!(settings.channel_blend_modes.len(), 2);
    }

    #[test]
    fn tc_0328_several_layers_flatten_into_one_fill() {
        use std::sync::Arc;

        // Blue base; two strokes; a 60% group holding a third; a fill on top.
        let base = MaterialLayer::new_fill(
            LayerId(1),
            "Base",
            LayerFill {
                color: [20, 40, 200, 255],
                roughness: 0.8,
                ..LayerFill::default()
            },
        );
        let stroke = |id: u64, from: (f32, f32), to: (f32, f32), color: [u8; 4]| {
            let mut layer = MaterialLayer::new_paint(LayerId(id), format!("Stroke {id}"));
            layer.paint_strokes.push(PaintStroke::new(
                vec![glam::Vec2::new(from.0, from.1), glam::Vec2::new(to.0, to.1)],
                2.5,
                color,
            ));
            layer
        };
        let mut scratches = stroke(2, (0.1, 0.3), (0.9, 0.6), [250, 250, 250, 255]);
        scratches.opacity = 0.7;
        let rust = stroke(3, (0.2, 0.8), (0.8, 0.2), [150, 60, 20, 255]);
        let mut group = MaterialLayer::new_group(LayerId(4), "Decals");
        group.opacity = 0.6;
        group.children = Some(vec![stroke(5, (0.5, 0.1), (0.5, 0.9), [10, 200, 10, 255])]);
        let top = MaterialLayer::new_fill(
            LayerId(6),
            "Varnish",
            LayerFill {
                roughness: 0.1,
                ..LayerFill::default()
            },
        );
        let mut set = small_set(24, base);
        set.layers.extend([scratches, rust, group]);
        let mut with_top = set.clone();
        with_top.layers.push(top);

        // Each layer once, without those inside a listed group, bottom
        // first; missing ids are refused.
        assert_eq!(
            with_top.flatten_order(&[LayerId(5), LayerId(6), LayerId(2), LayerId(4), LayerId(2)]),
            Some(vec![LayerId(2), LayerId(4), LayerId(6)])
        );
        assert_eq!(with_top.flatten_order(&[LayerId(2), LayerId(99)]), None);
        assert_eq!(
            set.flatten_layers(&[LayerId(99)], BakeContext::default()),
            Err(BakeError::InvalidLayerSettings)
        );
        assert_eq!(
            set.flatten_layers(&[], BakeContext::default()),
            Err(BakeError::InvalidLayerSettings)
        );
        assert_eq!(
            set.flatten_layers(&[LayerId(3)], BakeContext::default()),
            set.flatten_layer(LayerId(3), BakeContext::default()),
            "one layer flattens on its own"
        );

        // Adjacent Normal layers: a fill of the result in their place
        // composites like the originals.
        let flattened = set
            .flatten_layers(&[LayerId(3), LayerId(2)], BakeContext::default())
            .unwrap();
        assert_eq!(flattened.blend_mode, BlendMode::Normal);
        let ids: Vec<ResourceId> = (0..flattened.channels.len() as u64)
            .map(|i| ResourceId(100 + i))
            .collect();
        let images: ResourceImages = ids
            .iter()
            .zip(&flattened.channels)
            .map(|(id, (_, image))| (*id, Arc::new(image.clone())))
            .collect();
        let mut replaced = set.clone();
        replaced
            .layers
            .splice(1..3, [flattened.to_fill_layer(LayerId(9), "Merged", &ids)]);
        let context = BakeContext {
            resources: Some(&images),
            ..BakeContext::default()
        };
        for channel in [MaterialChannel::BaseColor, MaterialChannel::Roughness] {
            let original = set.bake_channel(channel, BakeContext::default()).unwrap();
            let rebuilt = replaced.bake_channel(channel, context).unwrap();
            let worst = original
                .rgba8
                .iter()
                .zip(&rebuilt.rgba8)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(worst <= 2, "{channel:?} differs by {worst}");
        }

        // Unrelated layers: each keeps its own opacity but not its group's.
        let merged = set
            .flatten_layers(&[LayerId(2), LayerId(5)], BakeContext::default())
            .unwrap();
        let alpha = |flattened: &FlattenedLayer, u: f32, v: f32| {
            let x = (u * 23.0).round() as usize;
            let y = ((1.0 - v) * 23.0).round() as usize;
            flattened.channels[0].1.rgba8[(y * 24 + x) * 4 + 3]
        };
        let green = set.layer(LayerId(5)).unwrap().clone();
        let green_alone = small_set(24, green)
            .flatten_layer(LayerId(5), BakeContext::default())
            .unwrap();
        let scratches_alone = set
            .flatten_layer(LayerId(2), BakeContext::default())
            .unwrap();
        assert!(alpha(&green_alone, 0.5, 0.85) > 200);
        assert_eq!(
            alpha(&merged, 0.5, 0.85),
            alpha(&green_alone, 0.5, 0.85),
            "the green stroke is not faded by its group"
        );
        assert_eq!(
            alpha(&merged, 0.15, 0.32),
            alpha(&scratches_alone, 0.15, 0.32),
            "the scratches keep their own opacity"
        );
        assert!(alpha(&scratches_alone, 0.15, 0.32) > 100);
        assert_eq!(alpha(&merged, 0.9, 0.1), 0, "nothing else is baked in");
        assert_eq!(
            merged.channels.len(),
            3,
            "colour, roughness, and metallic from the paint"
        );
    }
}
