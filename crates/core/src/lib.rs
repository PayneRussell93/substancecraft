#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::collections::{BTreeMap, BTreeSet};

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const PROJECT_SCHEMA_VERSION: u32 = 1;
pub const MAX_TEXTURE_DIMENSION: u32 = 32_768;
pub const MAX_BAKE_PIXELS: usize = 4_194_304;
const MAX_LAYER_STROKES: usize = 8_192;
const MAX_STROKE_POINTS: usize = 4_096;
const MAX_STROKE_HISTORY: usize = 128;

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
        let light_direction =
            Vec3::new(rotation.cos() * 0.8, 0.65, rotation.sin() * 0.8).normalize_or_zero();
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
    #[serde(default)]
    pub paint_strokes: Vec<PaintStroke>,
    #[serde(default)]
    pub mask_strokes: Vec<MaskStroke>,
    #[serde(default)]
    pub id_strokes: Vec<IdStroke>,
    #[serde(default)]
    pub id_mask_color: Option<[u8; 3]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PaintStroke {
    pub points: Vec<glam::Vec2>,
    pub radius_pixels: f32,
    pub color: [u8; 4],
    pub opacity: f32,
}

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

#[derive(Clone, Debug, PartialEq)]
pub enum StrokeCommand {
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
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrokeHistory {
    undo_stack: Vec<StrokeCommand>,
    redo_stack: Vec<StrokeCommand>,
}

impl StrokeHistory {
    pub fn record(&mut self, command: StrokeCommand) {
        if self.undo_stack.len() == MAX_STROKE_HISTORY {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(command);
        self.redo_stack.clear();
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
        if !remove_stroke(project, &command) {
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
        if !restore_stroke(project, &command) {
            self.redo_stack.push(command);
            return false;
        }
        if self.undo_stack.len() == MAX_STROKE_HISTORY {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(command);
        true
    }
}

fn remove_stroke(project: &mut MaterialProject, command: &StrokeCommand) -> bool {
    let (texture_set_id, layer_id) = match command {
        StrokeCommand::Paint {
            texture_set_id,
            layer_id,
            ..
        }
        | StrokeCommand::Mask {
            texture_set_id,
            layer_id,
            ..
        }
        | StrokeCommand::Id {
            texture_set_id,
            layer_id,
            ..
        } => (*texture_set_id, *layer_id),
    };
    let Some(layer) = project
        .texture_sets
        .iter_mut()
        .find(|set| set.id == texture_set_id)
        .and_then(|set| set.layers.iter_mut().find(|layer| layer.id == layer_id))
    else {
        return false;
    };

    match command {
        StrokeCommand::Paint { stroke, .. } if layer.paint_strokes.last() == Some(stroke) => {
            layer.paint_strokes.pop();
            true
        }
        StrokeCommand::Mask { stroke, .. } if layer.mask_strokes.last() == Some(stroke) => {
            layer.mask_strokes.pop();
            true
        }
        StrokeCommand::Id { stroke, .. } if layer.id_strokes.last() == Some(stroke) => {
            layer.id_strokes.pop();
            true
        }
        _ => false,
    }
}

fn restore_stroke(project: &mut MaterialProject, command: &StrokeCommand) -> bool {
    let (texture_set_id, layer_id) = match command {
        StrokeCommand::Paint {
            texture_set_id,
            layer_id,
            ..
        }
        | StrokeCommand::Mask {
            texture_set_id,
            layer_id,
            ..
        }
        | StrokeCommand::Id {
            texture_set_id,
            layer_id,
            ..
        } => (*texture_set_id, *layer_id),
    };
    let Some(layer) = project
        .texture_sets
        .iter_mut()
        .find(|set| set.id == texture_set_id)
        .and_then(|set| set.layers.iter_mut().find(|layer| layer.id == layer_id))
    else {
        return false;
    };

    match command {
        StrokeCommand::Paint { stroke, .. } => layer.paint_strokes.push(stroke.clone()),
        StrokeCommand::Mask { stroke, .. } => layer.mask_strokes.push(stroke.clone()),
        StrokeCommand::Id { stroke, .. } => layer.id_strokes.push(stroke.clone()),
    }
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
}

impl TextureSet {
    pub fn bake_base_color(&self) -> Result<BakedTexture, BakeError> {
        if self.layers.iter().any(|layer| !valid_layer_strokes(layer)) {
            return Err(BakeError::InvalidStrokeData);
        }
        let pixel_count = checked_bake_pixel_count(self.resolution)?;
        let id_map = self
            .layers
            .iter()
            .any(|layer| layer.id_mask_color.is_some())
            .then(|| self.bake_id_mask())
            .transpose()?;
        let byte_count = pixel_count
            .checked_mul(4)
            .ok_or(BakeError::AllocationFailed)?;
        let mut rgba8 = Vec::new();
        rgba8
            .try_reserve_exact(byte_count)
            .map_err(|_| BakeError::AllocationFailed)?;
        rgba8.resize(byte_count, 0);

        for layer in &self.layers {
            if !layer.visible || !layer.channels.contains(&MaterialChannel::BaseColor) {
                continue;
            }
            let mut mask = if layer.mask_strokes.is_empty() {
                None
            } else {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(pixel_count)
                    .map_err(|_| BakeError::AllocationFailed)?;
                values.resize(pixel_count, 1.0_f32);
                for stroke in &layer.mask_strokes {
                    rasterize_stroke(
                        &stroke.points,
                        stroke.radius_pixels,
                        self.resolution.width,
                        self.resolution.height,
                        |index, coverage| {
                            let alpha = (stroke.opacity * coverage).clamp(0.0, 1.0);
                            values[index] += (stroke.value - values[index]) * alpha;
                        },
                    );
                }
                Some(values)
            };

            for stroke in &layer.paint_strokes {
                rasterize_stroke(
                    &stroke.points,
                    stroke.radius_pixels,
                    self.resolution.width,
                    self.resolution.height,
                    |index, coverage| {
                        let mask_value = mask.as_mut().map_or(1.0, |values| values[index]);
                        let id_matches = layer.id_mask_color.is_none_or(|target| {
                            id_map
                                .as_ref()
                                .and_then(|map| map.rgba8.get(index * 4..index * 4 + 4))
                                .is_some_and(|pixel| pixel[0..3] == target && pixel[3] > 0)
                        });
                        let opacity = layer.opacity
                            * stroke.opacity
                            * coverage
                            * mask_value.clamp(0.0, 1.0)
                            * f32::from(id_matches);
                        blend_rgba8(&mut rgba8[index * 4..index * 4 + 4], stroke.color, opacity);
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

    pub fn bake_id_mask(&self) -> Result<BakedTexture, BakeError> {
        if self.layers.iter().any(|layer| !valid_layer_strokes(layer)) {
            return Err(BakeError::InvalidStrokeData);
        }
        let pixel_count = checked_bake_pixel_count(self.resolution)?;
        let byte_count = pixel_count
            .checked_mul(4)
            .ok_or(BakeError::AllocationFailed)?;
        let mut rgba8 = Vec::new();
        rgba8
            .try_reserve_exact(byte_count)
            .map_err(|_| BakeError::AllocationFailed)?;
        rgba8.resize(byte_count, 0);

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
        && layer
            .paint_strokes
            .iter()
            .all(|stroke| valid_stroke(&stroke.points, stroke.radius_pixels, stroke.opacity))
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
                if layer
                    .paint_strokes
                    .len()
                    .saturating_add(layer.mask_strokes.len())
                    .saturating_add(layer.id_strokes.len())
                    > MAX_LAYER_STROKES
                    || layer.paint_strokes.iter().any(|stroke| {
                        !valid_stroke(&stroke.points, stroke.radius_pixels, stroke.opacity)
                    })
                    || layer.mask_strokes.iter().any(|stroke| {
                        !valid_stroke(&stroke.points, stroke.radius_pixels, stroke.opacity)
                            || !stroke.value.is_finite()
                            || !(0.0..=1.0).contains(&stroke.value)
                    })
                    || layer.id_strokes.iter().any(|stroke| {
                        !valid_stroke(&stroke.points, stroke.radius_pixels, stroke.opacity)
                    })
                {
                    return Err(ProjectError::InvalidLayerPaintData(layer.id));
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
    #[error("layer {0:?} has invalid or excessive paint data")]
    InvalidLayerPaintData(LayerId),
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
            }],
            mask_strokes: vec![MaskStroke {
                points: vec![glam::Vec2::new(7.0 / 15.0, 8.0 / 15.0)],
                radius_pixels: 2.0,
                value: 0.0,
                opacity: 1.0,
            }],
            id_strokes: Vec::new(),
            id_mask_color: None,
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
            }],
            mask_strokes: Vec::new(),
            id_strokes: vec![IdStroke {
                points: vec![glam::Vec2::new(3.0 / 7.0, 4.0 / 7.0)],
                radius_pixels: 1.0,
                color: [12, 180, 220],
                opacity: 1.0,
            }],
            id_mask_color: Some([12, 180, 220]),
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
        });
        let mut project = MaterialProject {
            texture_sets: vec![set],
            ..MaterialProject::empty()
        };
        let mut history = StrokeHistory::default();
        history.record(StrokeCommand::Paint {
            texture_set_id: TextureSetId(1),
            layer_id: LayerId(2),
            stroke: paint.clone(),
        });
        history.record(StrokeCommand::Mask {
            texture_set_id: TextureSetId(1),
            layer_id: LayerId(2),
            stroke: mask.clone(),
        });
        history.record(StrokeCommand::Id {
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
        history.record(StrokeCommand::Paint {
            texture_set_id: TextureSetId(1),
            layer_id: LayerId(2),
            stroke: PaintStroke {
                points: vec![glam::Vec2::new(0.1, 0.1)],
                radius_pixels: 1.0,
                color: [1, 2, 3, 255],
                opacity: 1.0,
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
