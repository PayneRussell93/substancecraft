//! Geometry-derived mesh maps baked into UV space: tangent-space and
//! world-space normal, position, curvature, ambient occlusion, and thickness.
//!
//! Maps come from the working (low) mesh itself, or — when a high-resolution
//! source is given — from rays cast from a cage around the low mesh onto the
//! high mesh, transferring its detail (high-to-low baking).
//!
//! These maps feed procedural generators (edge wear, dirt, orientation, ...)
//! so effects adapt to each mesh. Baking is deterministic for a given mesh,
//! settings, and seed, runs on all available cores, and can be cancelled.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::noise::{lowbias32, unit_float};
use crate::smoothing::{vertex_normals, vertex_tangents};
use crate::topology::{mesh_bounds, weld_positions};
use crate::{BakedTexture, MAX_BAKE_PIXELS, MeshError, MeshGeometry, raster};

pub const MIN_MESH_MAP_RESOLUTION: u32 = 16;
pub const MAX_MESH_MAP_RESOLUTION: u32 = 2048;
pub const MAX_MESH_MAP_SAMPLES: u32 = 256;
pub const MAX_MESH_MAP_PADDING: u32 = 64;
/// Feature edges reach full strength this many degrees past the threshold.
const EDGE_ANGLE_RAMP_DEGREES: f32 = 40.0;
/// Scales smooth-surface curvature (relative to the mesh diagonal) before it
/// is compressed into the curvature map.
const SMOOTH_CURVATURE_SCALE: f32 = 0.08;
const RASTER_FRINGE_PIXELS: f32 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshMapSettings {
    /// Square map size in texels.
    pub resolution: u32,
    /// Occlusion rays per texel; thickness uses half as many.
    pub ao_samples: u32,
    /// Longest occlusion ray, as a fraction of the mesh bounding-box diagonal.
    pub ao_distance: f32,
    /// Width of the curvature band around sharp edges, as a fraction of the
    /// mesh bounding-box diagonal.
    pub edge_width: f32,
    /// Dihedral angle in degrees at which an edge starts to count as sharp.
    pub edge_angle_degrees: f32,
    /// Texels of edge padding grown around UV islands.
    pub padding: u32,
    pub seed: u32,
    /// Faces meeting at less than this angle share smooth normals.
    #[serde(default = "default_smoothing_angle")]
    pub smoothing_angle_degrees: f32,
    /// High-to-low only: how far outside and inside the low surface rays
    /// search for the high mesh, as a fraction of the bounds diagonal.
    #[serde(default = "default_cage_distance")]
    pub cage_distance: f32,
}

fn default_cage_distance() -> f32 {
    0.03
}

fn default_smoothing_angle() -> f32 {
    crate::smoothing::DEFAULT_SMOOTHING_ANGLE_DEGREES
}

impl Default for MeshMapSettings {
    fn default() -> Self {
        Self {
            resolution: 512,
            ao_samples: 24,
            ao_distance: 0.3,
            edge_width: 0.02,
            edge_angle_degrees: 20.0,
            padding: 4,
            seed: 1,
            smoothing_angle_degrees: default_smoothing_angle(),
            cage_distance: default_cage_distance(),
        }
    }
}

impl MeshMapSettings {
    pub fn validate(&self) -> Result<(), MeshMapError> {
        if !(MIN_MESH_MAP_RESOLUTION..=MAX_MESH_MAP_RESOLUTION).contains(&self.resolution)
            || (self.resolution as usize).pow(2) > MAX_BAKE_PIXELS
        {
            return Err(MeshMapError::InvalidSettings("resolution"));
        }
        if !(1..=MAX_MESH_MAP_SAMPLES).contains(&self.ao_samples) {
            return Err(MeshMapError::InvalidSettings("ao_samples"));
        }
        if !self.ao_distance.is_finite() || !(0.001..=4.0).contains(&self.ao_distance) {
            return Err(MeshMapError::InvalidSettings("ao_distance"));
        }
        if !self.edge_width.is_finite() || !(0.0005..=0.5).contains(&self.edge_width) {
            return Err(MeshMapError::InvalidSettings("edge_width"));
        }
        if !self.edge_angle_degrees.is_finite() || !(1.0..=170.0).contains(&self.edge_angle_degrees)
        {
            return Err(MeshMapError::InvalidSettings("edge_angle_degrees"));
        }
        if !self.smoothing_angle_degrees.is_finite()
            || !(0.0..=180.0).contains(&self.smoothing_angle_degrees)
        {
            return Err(MeshMapError::InvalidSettings("smoothing_angle_degrees"));
        }
        if !self.cage_distance.is_finite() || !(0.0001..=0.5).contains(&self.cage_distance) {
            return Err(MeshMapError::InvalidSettings("cage_distance"));
        }
        if self.padding > MAX_MESH_MAP_PADDING {
            return Err(MeshMapError::InvalidSettings("padding"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MeshMapKind {
    /// Tangent-space normal (OpenGL, +Y = +V) relative to the low mesh.
    Normal,
    WorldNormal,
    Position,
    Curvature,
    AmbientOcclusion,
    Thickness,
}

impl MeshMapKind {
    pub const ALL: [Self; 6] = [
        Self::Normal,
        Self::WorldNormal,
        Self::Position,
        Self::Curvature,
        Self::AmbientOcclusion,
        Self::Thickness,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::WorldNormal => "World Normal",
            Self::Position => "Position",
            Self::Curvature => "Curvature",
            Self::AmbientOcclusion => "Ambient Occlusion",
            Self::Thickness => "Thickness",
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum MeshMapError {
    #[error("mesh map setting `{0}` is out of range")]
    InvalidSettings(&'static str),
    #[error(transparent)]
    InvalidMesh(#[from] MeshError),
    #[error("mesh map bake was cancelled")]
    Cancelled,
    #[error("unable to allocate mesh map buffers")]
    AllocationFailed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Texel {
    position: Vec3,
    normal: Vec3,
    tangent_normal: Vec3,
    curvature: f32,
    ambient_occlusion: f32,
    thickness: f32,
}

/// Interpolated mesh-map values at a UV coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshMapSample {
    /// False when the UV lies outside every UV island (and its padding).
    pub covered: bool,
    /// Object-space position.
    pub position: Vec3,
    /// Position remapped to `0..=1` per axis within the mesh bounds.
    pub unit_position: Vec3,
    /// Position relative to the bounds minimum, divided by the bounds
    /// diagonal; a size-independent space for procedural noise.
    pub relative_position: Vec3,
    /// Unit world-space shading normal.
    pub normal: Vec3,
    /// Unit tangent-space normal relative to the low mesh; `+Z` when flat.
    pub tangent_normal: Vec3,
    /// `0.5` is flat, above is convex, below is concave.
    pub curvature: f32,
    /// `1` is fully open, `0` fully occluded.
    pub ambient_occlusion: f32,
    /// `0` is thin, `1` is thick.
    pub thickness: f32,
}

impl Default for MeshMapSample {
    fn default() -> Self {
        Self {
            covered: false,
            position: Vec3::ZERO,
            unit_position: Vec3::ZERO,
            relative_position: Vec3::ZERO,
            normal: Vec3::Y,
            tangent_normal: Vec3::Z,
            curvature: 0.5,
            ambient_occlusion: 1.0,
            thickness: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshMaps {
    width: u32,
    height: u32,
    settings: MeshMapSettings,
    bounds_min: Vec3,
    bounds_max: Vec3,
    texels: Vec<Texel>,
    /// Texels inside a UV triangle.
    covered: Vec<bool>,
    /// Covered texels plus padding.
    filled: Vec<bool>,
    /// Whether detail was transferred from a high-resolution mesh.
    from_high_poly: bool,
    /// Covered texels whose cage ray found no high-poly surface.
    misses: usize,
}

impl MeshMaps {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn settings(&self) -> &MeshMapSettings {
        &self.settings
    }

    pub fn from_high_poly(&self) -> bool {
        self.from_high_poly
    }

    /// Fraction of covered texels whose cage ray missed the high mesh; these
    /// fall back to the low surface. High values mean the cage is too small
    /// or the meshes do not overlap.
    pub fn miss_fraction(&self) -> f32 {
        let covered = self.covered.iter().filter(|value| **value).count();
        if covered == 0 {
            return 0.0;
        }
        self.misses as f32 / covered as f32
    }

    /// Fraction of texels inside a UV triangle (before padding).
    pub fn coverage(&self) -> f32 {
        if self.covered.is_empty() {
            return 0.0;
        }
        self.covered.iter().filter(|value| **value).count() as f32 / self.covered.len() as f32
    }

    fn diagonal(&self) -> f32 {
        (self.bounds_max - self.bounds_min)
            .length()
            .max(f32::EPSILON)
    }

    /// Bilinearly samples every map at `uv`, ignoring texels outside UV
    /// islands so values never bleed in from the empty background.
    pub fn sample(&self, uv: Vec2) -> MeshMapSample {
        if !uv.is_finite() || self.width == 0 || self.height == 0 {
            return MeshMapSample::default();
        }
        let fx = uv.x.clamp(0.0, 1.0) * self.width.saturating_sub(1) as f32;
        let fy = (1.0 - uv.y.clamp(0.0, 1.0)) * self.height.saturating_sub(1) as f32;
        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let mut accumulated = Texel::default();
        let mut total = 0.0;
        for (x, y, weight) in [
            (x0, y0, (1.0 - tx) * (1.0 - ty)),
            (x1, y0, tx * (1.0 - ty)),
            (x0, y1, (1.0 - tx) * ty),
            (x1, y1, tx * ty),
        ] {
            let index = y as usize * self.width as usize + x as usize;
            if weight <= 0.0 || !self.filled.get(index).copied().unwrap_or(false) {
                continue;
            }
            let Some(texel) = self.texels.get(index) else {
                continue;
            };
            accumulated.position += texel.position * weight;
            accumulated.normal += texel.normal * weight;
            accumulated.tangent_normal += texel.tangent_normal * weight;
            accumulated.curvature += texel.curvature * weight;
            accumulated.ambient_occlusion += texel.ambient_occlusion * weight;
            accumulated.thickness += texel.thickness * weight;
            total += weight;
        }
        if total <= f32::EPSILON {
            return MeshMapSample::default();
        }
        let position = accumulated.position / total;
        let extent = (self.bounds_max - self.bounds_min).max(Vec3::splat(f32::EPSILON));
        MeshMapSample {
            covered: true,
            position,
            unit_position: ((position - self.bounds_min) / extent).clamp(Vec3::ZERO, Vec3::ONE),
            relative_position: (position - self.bounds_min) / self.diagonal(),
            normal: accumulated.normal.normalize_or(Vec3::Y),
            tangent_normal: accumulated.tangent_normal.normalize_or(Vec3::Z),
            curvature: accumulated.curvature / total,
            ambient_occlusion: accumulated.ambient_occlusion / total,
            thickness: accumulated.thickness / total,
        }
    }

    /// Renders one map as an 8-bit preview/export texture. Texels outside UV
    /// islands are transparent black.
    pub fn to_texture(&self, kind: MeshMapKind) -> BakedTexture {
        let extent = (self.bounds_max - self.bounds_min).max(Vec3::splat(f32::EPSILON));
        let to_byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        let mut rgba8 = Vec::with_capacity(self.texels.len() * 4);
        for (texel, filled) in self.texels.iter().zip(&self.filled) {
            if !filled {
                rgba8.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let rgb = match kind {
                MeshMapKind::Normal => texel.tangent_normal * 0.5 + 0.5,
                MeshMapKind::WorldNormal => texel.normal * 0.5 + 0.5,
                MeshMapKind::Position => (texel.position - self.bounds_min) / extent,
                MeshMapKind::Curvature => Vec3::splat(texel.curvature),
                MeshMapKind::AmbientOcclusion => Vec3::splat(texel.ambient_occlusion),
                MeshMapKind::Thickness => Vec3::splat(texel.thickness),
            };
            rgba8.extend_from_slice(&[to_byte(rgb.x), to_byte(rgb.y), to_byte(rgb.z), 255]);
        }
        BakedTexture {
            width: self.width,
            height: self.height,
            rgba8,
        }
    }
}

/// Bakes every mesh map from `mesh` itself. When `cancel` becomes true the
/// bake stops at the next row and returns [`MeshMapError::Cancelled`].
pub fn bake_mesh_maps(
    mesh: &MeshGeometry,
    settings: &MeshMapSettings,
    cancel: Option<&AtomicBool>,
) -> Result<MeshMaps, MeshMapError> {
    bake_mesh_maps_from(mesh, None, settings, cancel)
}

/// Bakes mesh maps into `low`'s UV layout. With `high`, each texel casts a
/// ray from a cage `cage_distance` outside the low surface back along its
/// normal, and every map is evaluated on the high mesh where it hits, so the
/// tangent-space normal map carries the high mesh's detail. Texels whose ray
/// misses fall back to the low surface and are counted in
/// [`MeshMaps::miss_fraction`].
pub fn bake_mesh_maps_from(
    low: &MeshGeometry,
    high: Option<&MeshGeometry>,
    settings: &MeshMapSettings,
    cancel: Option<&AtomicBool>,
) -> Result<MeshMaps, MeshMapError> {
    settings.validate()?;
    low.validate()?;
    if let Some(high) = high {
        high.validate()?;
    }
    let is_cancelled = || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed));

    let width = settings.resolution;
    let height = settings.resolution;
    let pixel_count = (width as usize) * (height as usize);
    let (mut bounds_min, mut bounds_max) = mesh_bounds(low);
    if let Some(high) = high {
        let (high_min, high_max) = mesh_bounds(high);
        bounds_min = bounds_min.min(high_min);
        bounds_max = bounds_max.max(high_max);
    }
    let diagonal = (bounds_max - bounds_min).length().max(1e-6);

    let low_geometry = SurfaceGeometry::build(low, settings, diagonal);
    let low_normals = vertex_normals(low, settings.smoothing_angle_degrees);
    let low_tangents = vertex_tangents(low, &low_normals);
    if is_cancelled() {
        return Err(MeshMapError::Cancelled);
    }
    let high_geometry = high.map(|high| SurfaceGeometry::build(high, settings, diagonal));
    // Occlusion and thickness rays always test the detailed surface.
    let bvh = Bvh::build(high.unwrap_or(low));
    if is_cancelled() {
        return Err(MeshMapError::Cancelled);
    }

    // Rasterise triangle ownership. Interior coverage wins over fringe
    // coverage so neighbouring islands do not steal each other's texels.
    let mut owner = try_filled_vec(pixel_count, u32::MAX)?;
    let mut weights = try_filled_vec(pixel_count, Vec3::ZERO)?;
    let mut owned_by_interior = try_filled_vec(pixel_count, false)?;
    for (triangle_index, triangle) in low.triangles.iter().enumerate() {
        if low_geometry.face_normals[triangle_index] == Vec3::ZERO {
            continue;
        }
        let uvs = triangle.map(|index| low.vertices[index as usize].uv);
        raster::rasterize_uv_triangle(
            uvs,
            width,
            height,
            RASTER_FRINGE_PIXELS,
            |index, barycentric, interior| {
                if owner[index] == u32::MAX || (interior && !owned_by_interior[index]) {
                    owner[index] = triangle_index as u32;
                    weights[index] = barycentric;
                    owned_by_interior[index] = interior;
                }
            },
        );
    }
    let covered = owner
        .iter()
        .map(|owner| *owner != u32::MAX)
        .collect::<Vec<_>>();

    let mut texels = try_filled_vec(pixel_count, Texel::default())?;
    let low_context = TexelContext {
        mesh: low,
        settings,
        geometry: &low_geometry,
        bvh: &bvh,
        diagonal,
    };
    let high_context = high
        .zip(high_geometry.as_ref())
        .map(|(mesh, geometry)| TexelContext {
            mesh,
            settings,
            geometry,
            bvh: &bvh,
            diagonal,
        });
    let cage = settings.cage_distance * diagonal;
    let frame = LowFrame {
        mesh: low,
        normals: &low_normals,
        tangents: &low_tangents,
    };
    let misses = AtomicUsize::new(0);

    let rows_per_chunk = 4_usize;
    let chunk_len = rows_per_chunk * width as usize;
    let work = Mutex::new(texels.chunks_mut(chunk_len).enumerate());
    let threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, 32);
    let cancelled = AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    if is_cancelled() {
                        cancelled.store(true, Ordering::Relaxed);
                        return;
                    }
                    let next = match work.lock() {
                        Ok(mut chunks) => chunks.next(),
                        Err(_) => return,
                    };
                    let Some((chunk_index, chunk)) = next else {
                        return;
                    };
                    let start = chunk_index * chunk_len;
                    for (offset, texel) in chunk.iter_mut().enumerate() {
                        let index = start + offset;
                        let triangle = owner[index];
                        if triangle == u32::MAX {
                            continue;
                        }
                        let basis = frame.at(triangle, weights[index]);
                        let projected = high_context.as_ref().and_then(|context| {
                            let origin = basis.position + basis.normal * cage;
                            context
                                .bvh
                                .closest_hit(origin, -basis.normal, 0.0, cage * 2.0)
                                .map(|hit| {
                                    context.evaluate(hit.triangle, hit.weights, index as u32)
                                })
                        });
                        if high_context.is_some() && projected.is_none() {
                            misses.fetch_add(1, Ordering::Relaxed);
                        }
                        let mut value = projected.unwrap_or_else(|| {
                            low_context.evaluate(triangle, weights[index], index as u32)
                        });
                        value.tangent_normal = basis.to_tangent(value.normal);
                        *texel = value;
                    }
                }
            });
        }
    });
    if cancelled.load(Ordering::Relaxed) || is_cancelled() {
        return Err(MeshMapError::Cancelled);
    }

    let filled = dilate(&mut texels, &covered, width, height, settings.padding);
    Ok(MeshMaps {
        width,
        height,
        settings: *settings,
        bounds_min,
        bounds_max,
        texels,
        covered,
        filled,
        from_high_poly: high.is_some(),
        misses: misses.into_inner(),
    })
}

/// Per-vertex normal/tangent frame of the low mesh, interpolated exactly as
/// the renderer interpolates it, so tangent-space normals decode correctly.
struct LowFrame<'a> {
    mesh: &'a MeshGeometry,
    normals: &'a [Vec3],
    tangents: &'a [glam::Vec4],
}

struct TangentBasis {
    position: Vec3,
    normal: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
}

impl LowFrame<'_> {
    fn at(&self, triangle: u32, weights: Vec3) -> TangentBasis {
        let corners = self.mesh.triangles[triangle as usize];
        let blend3 = |values: [Vec3; 3]| {
            values[0] * weights.x + values[1] * weights.y + values[2] * weights.z
        };
        let position = blend3(corners.map(|index| self.mesh.vertices[index as usize].position));
        let normal =
            blend3(corners.map(|index| self.normals[index as usize])).normalize_or(Vec3::Z);
        let tangent4 = corners.map(|index| self.tangents[index as usize]);
        let tangent = blend3(tangent4.map(|tangent| tangent.truncate()));
        let tangent =
            (tangent - normal * normal.dot(tangent)).normalize_or(normal.any_orthonormal_vector());
        let sign = if tangent4[0].w < 0.0 { -1.0 } else { 1.0 };
        TangentBasis {
            position,
            normal,
            tangent,
            bitangent: normal.cross(tangent) * sign,
        }
    }
}

impl TangentBasis {
    fn to_tangent(&self, world: Vec3) -> Vec3 {
        Vec3::new(
            world.dot(self.tangent),
            world.dot(self.bitangent),
            world.dot(self.normal),
        )
        .normalize_or(Vec3::Z)
    }
}

fn try_filled_vec<T: Clone>(count: usize, value: T) -> Result<Vec<T>, MeshMapError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| MeshMapError::AllocationFailed)?;
    values.resize(count, value);
    Ok(values)
}

/// Grows island texels outward by `padding` texels, copying the nearest
/// island texel, and returns the filled mask.
fn dilate(
    texels: &mut [Texel],
    covered: &[bool],
    width: u32,
    height: u32,
    padding: u32,
) -> Vec<bool> {
    let mut filled = covered.to_vec();
    let width = width as i64;
    let height = height as i64;
    for _ in 0..padding {
        let mut grown = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                if filled[index] {
                    continue;
                }
                let neighbour = [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (-1, 1),
                    (1, -1),
                    (-1, -1),
                ]
                .into_iter()
                .map(|(dx, dy)| (x + dx, y + dy))
                .filter(|(nx, ny)| (0..width).contains(nx) && (0..height).contains(ny))
                .map(|(nx, ny)| (ny * width + nx) as usize)
                .find(|neighbour| filled[*neighbour]);
                if let Some(source) = neighbour {
                    grown.push((index, source));
                }
            }
        }
        if grown.is_empty() {
            break;
        }
        for (index, source) in grown {
            texels[index] = texels[source];
            filled[index] = true;
        }
    }
    filled
}

struct FeatureEdge {
    start: Vec3,
    end: Vec3,
    /// Positive for convex edges, negative for concave, magnitude `0..=1`.
    strength: f32,
    normals: [Vec3; 2],
}

/// Per-triangle and per-corner surface data derived once per bake.
struct SurfaceGeometry {
    face_normals: Vec<Vec3>,
    corner_normals: Vec<[Vec3; 3]>,
    corner_curvature: Vec<[f32; 3]>,
    edges: Vec<FeatureEdge>,
    edge_grid: HashMap<[i32; 3], Vec<u32>>,
    edge_radius: f32,
}

impl SurfaceGeometry {
    fn build(mesh: &MeshGeometry, settings: &MeshMapSettings, diagonal: f32) -> Self {
        let position = |index: u32| mesh.vertices[index as usize].position;
        let face_areas = mesh
            .triangles
            .iter()
            .map(|[a, b, c]| (position(*b) - position(*a)).cross(position(*c) - position(*a)))
            .collect::<Vec<_>>();
        let face_normals = face_areas
            .iter()
            .map(|area| area.normalize_or_zero())
            .collect::<Vec<_>>();

        let welded = weld_positions(mesh);
        let canonical = |index: u32| welded[index as usize];
        let mut incident = HashMap::<u32, Vec<u32>>::new();
        for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
            for vertex in triangle {
                incident
                    .entry(canonical(*vertex))
                    .or_default()
                    .push(triangle_index as u32);
            }
        }

        let smooth_cos = settings.smoothing_angle_degrees.to_radians().cos();
        let corner_normals =
            crate::smoothing::corner_normals(mesh, settings.smoothing_angle_degrees);

        let corner_curvature = mesh
            .triangles
            .iter()
            .enumerate()
            .map(|(triangle_index, triangle)| {
                let own = face_normals[triangle_index];
                let mut corners = [0.0; 3];
                for (corner, vertex) in triangle.iter().enumerate() {
                    let here = position(*vertex);
                    let normal = corner_normals[triangle_index][corner];
                    let key = canonical(*vertex);
                    let mut sum = 0.0;
                    let mut count = 0_u32;
                    for other in incident.get(&key).into_iter().flatten() {
                        let other = *other as usize;
                        if face_normals[other].dot(own) < smooth_cos {
                            continue;
                        }
                        for (other_corner, other_vertex) in mesh.triangles[other].iter().enumerate()
                        {
                            if canonical(*other_vertex) == key {
                                continue;
                            }
                            let offset = position(*other_vertex) - here;
                            let length_squared = offset.length_squared();
                            if length_squared <= f32::EPSILON * diagonal * diagonal {
                                continue;
                            }
                            let normal_delta = corner_normals[other][other_corner] - normal;
                            sum += normal_delta.dot(offset) / length_squared;
                            count += 1;
                        }
                    }
                    if count > 0 {
                        let curvature = sum / count as f32;
                        corners[corner] = (curvature * diagonal * SMOOTH_CURVATURE_SCALE).tanh();
                    }
                }
                corners
            })
            .collect::<Vec<_>>();

        let edges = feature_edges(mesh, &welded, &face_normals, settings);
        let edge_radius = (settings.edge_width * diagonal).max(1e-6);
        let mut edge_grid = HashMap::<[i32; 3], Vec<u32>>::new();
        for (edge_index, edge) in edges.iter().enumerate() {
            let length = edge.start.distance(edge.end);
            let steps = ((length / (edge_radius * 0.5)).ceil() as usize).clamp(1, 1 << 16);
            let mut last = None;
            for step in 0..=steps {
                let point = edge.start.lerp(edge.end, step as f32 / steps as f32);
                let cell = grid_cell(point, edge_radius);
                if last != Some(cell) {
                    let bucket = edge_grid.entry(cell).or_default();
                    if bucket.last() != Some(&(edge_index as u32)) {
                        bucket.push(edge_index as u32);
                    }
                    last = Some(cell);
                }
            }
        }

        Self {
            face_normals,
            corner_normals,
            corner_curvature,
            edges,
            edge_grid,
            edge_radius,
        }
    }

    /// Convex minus concave feature-edge influence at `point`.
    fn edge_curvature(&self, point: Vec3, face_normal: Vec3) -> f32 {
        let cell = grid_cell(point, self.edge_radius);
        let mut convex = 0.0_f32;
        let mut concave = 0.0_f32;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let key = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    let Some(bucket) = self.edge_grid.get(&key) else {
                        continue;
                    };
                    for edge_index in bucket {
                        let edge = &self.edges[*edge_index as usize];
                        // Only edges bounding a surface facing roughly the
                        // same way count, so thin walls do not pick up the
                        // opposite side's edges.
                        if edge
                            .normals
                            .iter()
                            .all(|normal| normal.dot(face_normal) < 0.5)
                        {
                            continue;
                        }
                        let distance = distance_to_segment(point, edge.start, edge.end);
                        if distance >= self.edge_radius {
                            continue;
                        }
                        let falloff = 1.0 - distance / self.edge_radius;
                        let influence = falloff * falloff * (3.0 - 2.0 * falloff);
                        if edge.strength > 0.0 {
                            convex = convex.max(influence * edge.strength);
                        } else {
                            concave = concave.max(influence * -edge.strength);
                        }
                    }
                }
            }
        }
        convex - concave
    }
}

fn feature_edges(
    mesh: &MeshGeometry,
    welded: &[u32],
    face_normals: &[Vec3],
    settings: &MeshMapSettings,
) -> Vec<FeatureEdge> {
    let mut adjacency = HashMap::<(u32, u32), [u32; 2]>::new();
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        if face_normals[triangle_index] == Vec3::ZERO {
            continue;
        }
        for corner in 0..3 {
            let a = welded[triangle[corner] as usize];
            let b = welded[triangle[(corner + 1) % 3] as usize];
            if a == b {
                continue;
            }
            let key = (a.min(b), a.max(b));
            let entry = adjacency.entry(key).or_insert([u32::MAX; 2]);
            if entry[0] == u32::MAX {
                entry[0] = triangle_index as u32;
            } else if entry[1] == u32::MAX {
                entry[1] = triangle_index as u32;
            }
        }
    }

    let start_angle = settings.edge_angle_degrees;
    let full_angle = (start_angle + EDGE_ANGLE_RAMP_DEGREES).min(179.0);
    let mut edges = Vec::new();
    for ((a, b), [first, second]) in adjacency {
        if second == u32::MAX {
            continue;
        }
        let first_normal = face_normals[first as usize];
        let second_normal = face_normals[second as usize];
        let angle = first_normal
            .dot(second_normal)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        let strength = ((angle - start_angle) / (full_angle - start_angle)).clamp(0.0, 1.0);
        if strength <= 0.0 {
            continue;
        }
        let start = mesh.vertices[a as usize].position;
        let end = mesh.vertices[b as usize].position;
        // The second face's far vertex lies below the first face's plane when
        // the edge is convex.
        let far = mesh.triangles[second as usize]
            .iter()
            .map(|index| mesh.vertices[*index as usize].position)
            .max_by(|left, right| {
                distance_to_segment(*left, start, end)
                    .total_cmp(&distance_to_segment(*right, start, end))
            })
            .unwrap_or(start);
        let convex = first_normal.dot(far - start) < 0.0;
        edges.push(FeatureEdge {
            start,
            end,
            strength: if convex { strength } else { -strength },
            normals: [first_normal, second_normal],
        });
    }
    // HashMap iteration order is random; sort so bakes are deterministic.
    edges.sort_by(|left, right| {
        left.start
            .to_array()
            .partial_cmp(&right.start.to_array())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                left.end
                    .to_array()
                    .partial_cmp(&right.end.to_array())
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    edges
}

fn grid_cell(point: Vec3, cell_size: f32) -> [i32; 3] {
    let cell = (point / cell_size).floor();
    [cell.x as i32, cell.y as i32, cell.z as i32]
}

fn distance_to_segment(point: Vec3, start: Vec3, end: Vec3) -> f32 {
    let segment = end - start;
    let length_squared = segment.length_squared();
    let amount = if length_squared > f32::EPSILON {
        ((point - start).dot(segment) / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    point.distance(start + segment * amount)
}

struct TexelContext<'a> {
    mesh: &'a MeshGeometry,
    settings: &'a MeshMapSettings,
    geometry: &'a SurfaceGeometry,
    bvh: &'a Bvh,
    diagonal: f32,
}

impl TexelContext<'_> {
    fn evaluate(&self, triangle: u32, weights: Vec3, texel_index: u32) -> Texel {
        let triangle_index = triangle as usize;
        let corners = self.mesh.triangles[triangle_index];
        let position = |corner: usize| self.mesh.vertices[corners[corner] as usize].position;
        let point = position(0) * weights.x + position(1) * weights.y + position(2) * weights.z;
        let corner_normals = self.geometry.corner_normals[triangle_index];
        let face_normal = self.geometry.face_normals[triangle_index];
        let normal = (corner_normals[0] * weights.x
            + corner_normals[1] * weights.y
            + corner_normals[2] * weights.z)
            .normalize_or(face_normal);
        let corner_curvature = self.geometry.corner_curvature[triangle_index];
        let smooth = corner_curvature[0] * weights.x
            + corner_curvature[1] * weights.y
            + corner_curvature[2] * weights.z;
        let edge = self.geometry.edge_curvature(point, face_normal);
        let curvature = 0.5 + 0.5 * (edge + smooth).clamp(-1.0, 1.0);

        let bias = self.diagonal * 1e-4;
        let seed = lowbias32(texel_index ^ self.settings.seed.wrapping_mul(0x9e37_79b9));
        let ao_distance = self.settings.ao_distance * self.diagonal;
        let ao_samples = self.settings.ao_samples;
        let origin = point + face_normal * bias;
        let mut occluded = 0_u32;
        for sample in 0..ao_samples {
            let direction = cosine_direction(normal, face_normal, sample, ao_samples, seed);
            if self.bvh.any_hit(origin, direction, bias, ao_distance) {
                occluded += 1;
            }
        }
        let ambient_occlusion = 1.0 - occluded as f32 / ao_samples as f32;

        let thickness_samples = (ao_samples / 2).max(4);
        let thickness_distance = self.diagonal * 0.5;
        let inside = point - face_normal * bias;
        let mut total = 0.0;
        for sample in 0..thickness_samples {
            let direction = cosine_direction(
                -normal,
                -face_normal,
                sample,
                thickness_samples,
                lowbias32(seed ^ 0x5bd1_e995),
            );
            total += self
                .bvh
                .closest_hit(inside, direction, bias, thickness_distance)
                .map_or(thickness_distance, |hit| hit.distance);
        }
        let thickness = (total / thickness_samples as f32 / thickness_distance).clamp(0.0, 1.0);

        Texel {
            position: point,
            normal,
            // Set by the caller once the low-mesh tangent frame is known.
            tangent_normal: Vec3::Z,
            curvature,
            ambient_occlusion,
            thickness,
        }
    }
}

/// Cosine-weighted hemisphere direction around `normal` from a rotated
/// Hammersley sequence, reflected above the geometric face plane.
fn cosine_direction(normal: Vec3, face_normal: Vec3, index: u32, count: u32, seed: u32) -> Vec3 {
    let jitter = Vec2::new(unit_float(seed), unit_float(lowbias32(seed)));
    let u = ((index as f32 + 0.5) / count as f32 + jitter.x).fract();
    let v = (radical_inverse(index) + jitter.y).fract();
    let radius = u.sqrt();
    let angle = std::f32::consts::TAU * v;
    let (tangent, bitangent) = normal.any_orthonormal_pair();
    let direction = (tangent * (radius * angle.cos())
        + bitangent * (radius * angle.sin())
        + normal * (1.0 - u).max(0.0).sqrt())
    .normalize_or(normal);
    let below = direction.dot(face_normal);
    if below < 0.0 {
        (direction - face_normal * (2.0 * below)).normalize_or(normal)
    } else {
        direction
    }
}

fn radical_inverse(mut bits: u32) -> f32 {
    bits = bits.reverse_bits();
    bits as f32 * (1.0 / 4_294_967_296.0)
}

#[derive(Clone, Copy)]
struct BvhNode {
    min: Vec3,
    max: Vec3,
    /// Leaf: first index into `order`. Interior: index of the left child; the
    /// right child follows it.
    first: u32,
    /// Triangle count for leaves, zero for interior nodes.
    count: u32,
}

/// Bounding-volume hierarchy over mesh triangles for occlusion rays.
struct Bvh {
    nodes: Vec<BvhNode>,
    order: Vec<u32>,
    triangles: Vec<[Vec3; 3]>,
}

const BVH_LEAF_SIZE: usize = 4;

impl Bvh {
    fn build(mesh: &MeshGeometry) -> Self {
        let triangles = mesh
            .triangles
            .iter()
            .map(|triangle| triangle.map(|index| mesh.vertices[index as usize].position))
            .collect::<Vec<_>>();
        let centroids = triangles
            .iter()
            .map(|[a, b, c]| (*a + *b + *c) / 3.0)
            .collect::<Vec<_>>();
        let mut order = (0..triangles.len() as u32).collect::<Vec<_>>();
        let mut nodes = vec![BvhNode {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            first: 0,
            count: 0,
        }];
        let mut stack = vec![(0_usize, 0_usize, order.len())];
        while let Some((node_index, start, end)) = stack.pop() {
            let mut minimum = Vec3::splat(f32::INFINITY);
            let mut maximum = Vec3::splat(f32::NEG_INFINITY);
            let mut centroid_min = Vec3::splat(f32::INFINITY);
            let mut centroid_max = Vec3::splat(f32::NEG_INFINITY);
            for triangle in &order[start..end] {
                for corner in triangles[*triangle as usize] {
                    minimum = minimum.min(corner);
                    maximum = maximum.max(corner);
                }
                let centroid = centroids[*triangle as usize];
                centroid_min = centroid_min.min(centroid);
                centroid_max = centroid_max.max(centroid);
            }
            let count = end - start;
            let spread = centroid_max - centroid_min;
            if count <= BVH_LEAF_SIZE || spread.max_element() <= f32::EPSILON {
                nodes[node_index] = BvhNode {
                    min: minimum,
                    max: maximum,
                    first: start as u32,
                    count: count as u32,
                };
                continue;
            }
            let axis = if spread.x >= spread.y && spread.x >= spread.z {
                0
            } else if spread.y >= spread.z {
                1
            } else {
                2
            };
            let middle = (centroid_min[axis] + centroid_max[axis]) * 0.5;
            let slice = &mut order[start..end];
            let mut split = 0;
            for index in 0..slice.len() {
                if centroids[slice[index] as usize][axis] < middle {
                    slice.swap(index, split);
                    split += 1;
                }
            }
            if split == 0 || split == slice.len() {
                split = slice.len() / 2;
                slice.select_nth_unstable_by(split, |left, right| {
                    centroids[*left as usize][axis].total_cmp(&centroids[*right as usize][axis])
                });
            }
            let left = nodes.len();
            nodes.push(nodes[node_index]);
            nodes.push(nodes[node_index]);
            nodes[node_index] = BvhNode {
                min: minimum,
                max: maximum,
                first: left as u32,
                count: 0,
            };
            stack.push((left, start, start + split));
            stack.push((left + 1, start + split, end));
        }
        Self {
            nodes,
            order,
            triangles,
        }
    }

    fn any_hit(&self, origin: Vec3, direction: Vec3, t_min: f32, t_max: f32) -> bool {
        self.traverse(origin, direction, t_min, t_max, true)
            .is_some()
    }

    fn closest_hit(&self, origin: Vec3, direction: Vec3, t_min: f32, t_max: f32) -> Option<RayHit> {
        self.traverse(origin, direction, t_min, t_max, false)
    }

    fn traverse(
        &self,
        origin: Vec3,
        direction: Vec3,
        t_min: f32,
        mut t_max: f32,
        any: bool,
    ) -> Option<RayHit> {
        if self.order.is_empty() {
            return None;
        }
        let inverse = direction.recip();
        let mut closest = None;
        let mut stack = [0_u32; 128];
        let mut depth = 1;
        while depth > 0 {
            depth -= 1;
            let node = self.nodes[stack[depth] as usize];
            if !ray_hits_box(origin, inverse, node.min, node.max, t_min, t_max) {
                continue;
            }
            if node.count > 0 {
                let first = node.first as usize;
                for triangle in &self.order[first..first + node.count as usize] {
                    let [a, b, c] = self.triangles[*triangle as usize];
                    if let Some((distance, u, v)) = ray_triangle(origin, direction, a, b, c)
                        && distance > t_min
                        && distance < t_max
                    {
                        let hit = RayHit {
                            distance,
                            triangle: *triangle,
                            weights: Vec3::new(1.0 - u - v, u, v),
                        };
                        if any {
                            return Some(hit);
                        }
                        t_max = distance;
                        closest = Some(hit);
                    }
                }
            } else if depth + 2 <= stack.len() {
                stack[depth] = node.first;
                stack[depth + 1] = node.first + 1;
                depth += 2;
            }
        }
        closest
    }
}

fn ray_hits_box(origin: Vec3, inverse: Vec3, min: Vec3, max: Vec3, t_min: f32, t_max: f32) -> bool {
    let first = (min - origin) * inverse;
    let second = (max - origin) * inverse;
    // `max_element`/`min_element` skip NaN lanes produced by a zero direction
    // component lying exactly on a slab boundary.
    let near = first.min(second).max_element().max(t_min);
    let far = first.max(second).min_element().min(t_max);
    near <= far
}

#[derive(Clone, Copy, Debug)]
struct RayHit {
    distance: f32,
    triangle: u32,
    /// Barycentric weights of the triangle corners at the hit.
    weights: Vec3,
}

/// Two-sided Möller–Trumbore intersection returning the hit distance and
/// the barycentric `u`, `v` of corners `b` and `c`.
fn ray_triangle(
    origin: Vec3,
    direction: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<(f32, f32, f32)> {
    let edge_ab = b - a;
    let edge_ac = c - a;
    let p = direction.cross(edge_ac);
    let determinant = edge_ab.dot(p);
    if determinant.abs() < 1e-12 {
        return None;
    }
    let inverse = 1.0 / determinant;
    let offset = origin - a;
    let u = offset.dot(p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = offset.cross(edge_ab);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge_ac.dot(q) * inverse;
    distance.is_finite().then_some((distance, u, v))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{MeshVertex, PrimitivePreset};

    fn fast_settings(resolution: u32) -> MeshMapSettings {
        MeshMapSettings {
            resolution,
            ao_samples: 16,
            edge_width: 0.06,
            ..MeshMapSettings::default()
        }
    }

    /// UV of a point on the cube's +Z face (face 0 of the 3x2 atlas).
    fn front_face_uv(x: f32, y: f32) -> Vec2 {
        let min = Vec2::splat(0.01);
        let max = Vec2::new(1.0 / 3.0 - 0.01, 0.5 - 0.01);
        min + (max - min) * Vec2::new(x + 0.5, y + 0.5)
    }

    #[test]
    fn tc_0033_cube_curvature_is_convex_at_edges_and_flat_at_face_centres() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let maps = bake_mesh_maps(&cube, &fast_settings(96), None).unwrap();

        let centre = maps.sample(front_face_uv(0.0, 0.0));
        let near_edge = maps.sample(front_face_uv(0.49, 0.0));
        assert!(centre.covered && near_edge.covered);
        assert!(
            (centre.curvature - 0.5).abs() < 0.02,
            "centre {}",
            centre.curvature
        );
        assert!(near_edge.curvature > 0.8, "edge {}", near_edge.curvature);
        assert!((centre.normal - Vec3::Z).length() < 0.01);
        assert!((centre.position - Vec3::new(0.0, 0.0, 0.5)).length() < 0.05);
        assert!(maps.coverage() > 0.5);
        let curvature = maps.to_texture(MeshMapKind::Curvature);
        assert_eq!(curvature.rgba8.len(), 96 * 96 * 4);
    }

    #[test]
    fn tc_0034_ambient_occlusion_darkens_concave_corners_but_not_open_faces() {
        // An L-shaped pair of quads: a floor and a wall meeting at a concave
        // right angle, each with its own UV chart.
        let mut mesh = MeshGeometry {
            vertices: Vec::new(),
            triangles: Vec::new(),
            ..MeshGeometry::default()
        };
        let mut quad = |positions: [Vec3; 4], uv_min: Vec2, uv_max: Vec2| {
            let base = mesh.vertices.len() as u32;
            let uvs = [
                Vec2::new(uv_min.x, uv_min.y),
                Vec2::new(uv_max.x, uv_min.y),
                Vec2::new(uv_max.x, uv_max.y),
                Vec2::new(uv_min.x, uv_max.y),
            ];
            for (position, uv) in positions.into_iter().zip(uvs) {
                mesh.vertices.push(MeshVertex { position, uv });
            }
            mesh.triangles
                .extend([[base, base + 1, base + 2], [base, base + 2, base + 3]]);
        };
        // Floor (y = 0, facing +Y), wall (z = -0.5, facing +Z).
        quad(
            [
                Vec3::new(-0.5, 0.0, 0.5),
                Vec3::new(0.5, 0.0, 0.5),
                Vec3::new(0.5, 0.0, -0.5),
                Vec3::new(-0.5, 0.0, -0.5),
            ],
            Vec2::new(0.0, 0.0),
            Vec2::new(0.49, 1.0),
        );
        quad(
            [
                Vec3::new(-0.5, 0.0, -0.5),
                Vec3::new(0.5, 0.0, -0.5),
                Vec3::new(0.5, 1.0, -0.5),
                Vec3::new(-0.5, 1.0, -0.5),
            ],
            Vec2::new(0.51, 0.0),
            Vec2::new(1.0, 1.0),
        );
        let maps = bake_mesh_maps(&mesh, &fast_settings(64), None).unwrap();

        // Floor UV v=0 is z=+0.5 (open), v=1 is z=-0.5 (at the wall).
        let open = maps.sample(Vec2::new(0.245, 0.1));
        let corner = maps.sample(Vec2::new(0.245, 0.97));
        assert!(
            open.ambient_occlusion > 0.85,
            "open {}",
            open.ambient_occlusion
        );
        assert!(
            corner.ambient_occlusion < 0.7,
            "corner {}",
            corner.ambient_occlusion
        );
        assert!(
            corner.curvature < 0.4,
            "concave corner {}",
            corner.curvature
        );
        assert!(maps.to_texture(MeshMapKind::AmbientOcclusion).rgba8.len() == 64 * 64 * 4);
    }

    #[test]
    fn tc_0035_mesh_map_bake_validates_settings_and_honours_cancellation() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let invalid = MeshMapSettings {
            resolution: 8,
            ..MeshMapSettings::default()
        };
        assert_eq!(
            bake_mesh_maps(&cube, &invalid, None),
            Err(MeshMapError::InvalidSettings("resolution"))
        );
        let cancelled = AtomicBool::new(true);
        assert_eq!(
            bake_mesh_maps(&cube, &fast_settings(32), Some(&cancelled)),
            Err(MeshMapError::Cancelled)
        );
        let first = bake_mesh_maps(&cube, &fast_settings(32), None).unwrap();
        let second = bake_mesh_maps(&cube, &fast_settings(32), None).unwrap();
        assert_eq!(first, second, "bakes must be deterministic");
    }

    #[test]
    fn sphere_thickness_is_high_and_ao_open() {
        let sphere = PrimitivePreset::Sphere.build_mesh();
        let maps = bake_mesh_maps(&sphere, &fast_settings(48), None).unwrap();
        let sample = maps.sample(Vec2::new(0.3, 0.5));
        assert!(sample.covered);
        assert!(sample.ambient_occlusion > 0.9);
        assert!(sample.thickness > 0.5);
        assert!(
            sample.curvature > 0.55,
            "sphere is convex: {}",
            sample.curvature
        );
    }

    /// A 1.2-unit quad tilted about Z so its normal leans `degrees` toward +X,
    /// raised by `lift`.
    fn tilted_quad(degrees: f32, lift: f32) -> MeshGeometry {
        let slope = degrees.to_radians().tan();
        let point = |x: f32, z: f32| MeshVertex {
            position: Vec3::new(x, lift - slope * x, z),
            uv: Vec2::new(x, z),
        };
        MeshGeometry {
            vertices: vec![
                point(-0.6, 0.6),
                point(0.6, 0.6),
                point(0.6, -0.6),
                point(-0.6, -0.6),
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        }
    }

    #[test]
    fn tc_0051_high_to_low_bake_transfers_high_poly_normals_into_tangent_space() {
        let low = PrimitivePreset::Plane.build_mesh();
        let high = tilted_quad(20.0, 0.0);
        let settings = MeshMapSettings {
            cage_distance: 0.2,
            ..fast_settings(32)
        };

        let maps = bake_mesh_maps_from(&low, Some(&high), &settings, None).unwrap();

        assert!(maps.from_high_poly());
        assert_eq!(maps.miss_fraction(), 0.0);
        let expected = 20.0_f32.to_radians().sin();
        for uv in [Vec2::splat(0.5), Vec2::new(0.8, 0.3)] {
            let sample = maps.sample(uv);
            assert!(
                (sample.tangent_normal.x - expected).abs() < 0.02,
                "tangent normal {}",
                sample.tangent_normal
            );
            assert!(sample.tangent_normal.y.abs() < 0.02);
            assert!(
                sample.normal.x > 0.3,
                "world normal comes from the high mesh"
            );
        }
        let normal_map = maps.to_texture(MeshMapKind::Normal);
        let centre = normal_map.sample_uv(Vec2::splat(0.5)).unwrap();
        assert!(centre[0] > 165 && (125..=131).contains(&centre[1]));

        // Without a high mesh the low surface is its own detail: flat normals.
        let plain = bake_mesh_maps(&low, &settings, None).unwrap();
        assert!(!plain.from_high_poly());
        assert!((plain.sample(Vec2::splat(0.5)).tangent_normal - Vec3::Z).length() < 1e-4);
    }

    #[test]
    fn tc_0052_cage_rays_that_miss_the_high_mesh_fall_back_and_are_reported() {
        let low = PrimitivePreset::Plane.build_mesh();
        let far_away = tilted_quad(20.0, 5.0);
        let settings = MeshMapSettings {
            cage_distance: 0.01,
            ..fast_settings(16)
        };

        let maps = bake_mesh_maps_from(&low, Some(&far_away), &settings, None).unwrap();

        assert!(maps.miss_fraction() > 0.99);
        let sample = maps.sample(Vec2::splat(0.5));
        assert!((sample.tangent_normal - Vec3::Z).length() < 1e-4);
        assert!((sample.normal - Vec3::Y).length() < 1e-4);
        assert_eq!(
            bake_mesh_maps_from(
                &low,
                Some(&far_away),
                &MeshMapSettings {
                    cage_distance: 0.0,
                    ..settings
                },
                None
            ),
            Err(MeshMapError::InvalidSettings("cage_distance"))
        );
    }
}
