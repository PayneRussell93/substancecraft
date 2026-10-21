//! Ray queries of a mesh-map bake, behind a trait so they can run on
//! another device (PERF-01). The bake prepares its triangles and BVH once
//! as a [`TraceScene`], then asks for cage-ray hits and per-texel occlusion
//! in batches. [`crate::bake_mesh_maps_traced`] answers every batch a
//! tracer fails on itself, on the CPU, so a tracer can never fail a bake.

use glam::Vec3;

/// One node of a bake's bounding-volume hierarchy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNode {
    pub min: Vec3,
    pub max: Vec3,
    /// Leaf: first index into [`TraceScene::order`]. Interior: index of the
    /// left child; the right child follows it.
    pub first: u32,
    /// Triangle count for leaves, zero for interior nodes.
    pub count: u32,
}

/// The triangles rays are traced against and the BVH over them, exactly as
/// the CPU traverses it, so other tracers visit the same nodes.
#[derive(Clone, Copy, Debug)]
pub struct TraceScene<'a> {
    pub nodes: &'a [BvhNode],
    /// Triangle indices in leaf order.
    pub order: &'a [u32],
    pub triangles: &'a [[Vec3; 3]],
}

/// Longest stack a BVH traversal keeps; deeper nodes are skipped. Tracers
/// should use the same depth so they skip the same nodes.
pub const TRACE_STACK_DEPTH: usize = 128;

/// A ray from a texel's cage back through the low surface. Hits count
/// strictly between 0 and `t_max`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CageRay {
    pub origin: Vec3,
    pub direction: Vec3,
    pub t_max: f32,
}

/// Where a cage ray first meets the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CageHit {
    pub triangle: u32,
    /// Barycentric weights of the triangle's corners.
    pub weights: Vec3,
}

/// A surface point whose ambient occlusion and thickness are wanted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionQuery {
    pub point: Vec3,
    /// Interpolated shading normal.
    pub normal: Vec3,
    pub face_normal: Vec3,
    /// Texel index, which seeds the sample directions.
    pub texel: u32,
}

/// Settings shared by every occlusion query of a bake.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionParams {
    pub seed: u32,
    pub ao_samples: u32,
    pub ao_distance: f32,
    pub thickness_samples: u32,
    pub thickness_distance: f32,
    /// Offset off the surface and minimum hit distance.
    pub bias: f32,
}

/// Ambient occlusion and thickness of one query, both `0..=1`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Occlusion {
    pub ambient_occlusion: f32,
    pub thickness: f32,
}

/// Something that can answer a bake's ray queries, such as a GPU.
pub trait RayTracer: Sync {
    /// A short name for logs, such as the adapter's.
    fn name(&self) -> String;

    /// Uploads `scene` for the queries of one bake.
    fn prepare(&self, scene: &TraceScene<'_>) -> Result<Box<dyn PreparedScene + '_>, String>;
}

/// A scene uploaded by a [`RayTracer`]. Each call answers every query, in
/// order, or fails as a whole.
pub trait PreparedScene {
    fn cage_hits(&mut self, rays: &[CageRay]) -> Result<Vec<Option<CageHit>>, String>;

    fn occlusion(
        &mut self,
        queries: &[OcclusionQuery],
        params: &OcclusionParams,
    ) -> Result<Vec<Occlusion>, String>;
}

/// Which device traced a bake's rays.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum TraceDevice {
    #[default]
    Cpu,
    /// Every batch ran on the named tracer.
    Tracer(String),
    /// The named tracer failed (with this error) and the CPU took over.
    Fallback { tracer: String, error: String },
}
