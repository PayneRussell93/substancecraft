//! Ray casting for mesh-map bakes: a bounding-volume hierarchy over the
//! detailed surface, and the ambient-occlusion and thickness rays traced
//! through it.
//!
//! The bake evaluates each texel's surface on the CPU and hands one
//! [`OcclusionQuery`] per covered texel to an [`OcclusionTracer`]. The
//! [`CpuTracer`] traces them on all cores; other tracers (a GPU compute
//! path) must follow the same sample directions, so their results match it
//! within floating-point tolerance. Everything a tracer needs is public:
//! [`Bvh::nodes`], [`Bvh::order`], [`Bvh::triangles`], and
//! [`sample_direction`].

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use glam::{Vec2, Vec3};

use crate::MeshGeometry;
use crate::mesh_maps::{MeshMapError, MeshMapSettings};
use crate::noise::{lowbias32, unit_float};

/// Seed mixed into a texel's seed for its thickness rays.
pub const THICKNESS_SEED: u32 = 0x5bd1_e995;

/// One covered texel's occlusion and thickness rays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionQuery {
    /// Surface point the rays start from.
    pub point: Vec3,
    /// Unit shading normal the sample hemisphere is centred on.
    pub normal: Vec3,
    /// Unit geometric normal; rays never dip below its plane.
    pub face_normal: Vec3,
    /// Per-texel seed rotating the sample pattern.
    pub seed: u32,
}

/// How far and how many occlusion and thickness rays each query casts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionParams {
    pub ao_samples: u32,
    /// Occlusion ray length in scene units.
    pub ao_distance: f32,
    pub thickness_samples: u32,
    /// Thickness ray length in scene units; a miss counts as this far.
    pub thickness_distance: f32,
    /// Offset of ray origins off the surface, and the nearest hit counted.
    pub bias: f32,
}

impl OcclusionParams {
    /// The rays `settings` asks for on a mesh whose bounds have `diagonal`.
    pub fn new(settings: &MeshMapSettings, diagonal: f32) -> Self {
        Self {
            ao_samples: settings.ao_samples,
            ao_distance: settings.ao_distance * diagonal,
            thickness_samples: (settings.ao_samples / 2).max(4),
            thickness_distance: diagonal * 0.5,
            bias: diagonal * 1e-4,
        }
    }
}

/// Traces occlusion queries against a scene. Results are
/// `[ambient_occlusion, thickness]` per query, in query order: occlusion
/// is `1` fully open and thickness `1` thick, both in `0..=1`.
pub trait OcclusionTracer: Sync {
    fn trace(
        &self,
        scene: &Bvh,
        params: &OcclusionParams,
        queries: &[OcclusionQuery],
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<[f32; 2]>, MeshMapError>;
}

/// Traces queries on every available core.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuTracer;

impl OcclusionTracer for CpuTracer {
    fn trace(
        &self,
        scene: &Bvh,
        params: &OcclusionParams,
        queries: &[OcclusionQuery],
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<[f32; 2]>, MeshMapError> {
        let mut results = vec![[1.0, 1.0]; queries.len()];
        for_each_chunk(&mut results, 1024, cancel, |start, chunk| {
            for (offset, result) in chunk.iter_mut().enumerate() {
                *result = trace_query(scene, params, &queries[start + offset]);
            }
        })?;
        Ok(results)
    }
}

/// Runs `work` over `items` in chunks of `chunk_len` on every core, each
/// call getting the chunk's first index; stops early when `cancel` is set.
pub(crate) fn for_each_chunk<T: Send>(
    items: &mut [T],
    chunk_len: usize,
    cancel: Option<&AtomicBool>,
    work: impl Fn(usize, &mut [T]) + Sync,
) -> Result<(), MeshMapError> {
    let is_cancelled = || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed));
    let chunk_len = chunk_len.max(1);
    let chunks = Mutex::new(items.chunks_mut(chunk_len).enumerate());
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
                    let next = match chunks.lock() {
                        Ok(mut chunks) => chunks.next(),
                        Err(_) => return,
                    };
                    let Some((index, chunk)) = next else {
                        return;
                    };
                    work(index * chunk_len, chunk);
                }
            });
        }
    });
    if cancelled.load(Ordering::Relaxed) || is_cancelled() {
        return Err(MeshMapError::Cancelled);
    }
    Ok(())
}

/// `[ambient_occlusion, thickness]` of one query, traced on this thread.
pub fn trace_query(scene: &Bvh, params: &OcclusionParams, query: &OcclusionQuery) -> [f32; 2] {
    let OcclusionQuery {
        point,
        normal,
        face_normal,
        seed,
    } = *query;
    let bias = params.bias;
    let origin = point + face_normal * bias;
    let mut occluded = 0_u32;
    for sample in 0..params.ao_samples {
        let direction = sample_direction(normal, face_normal, sample, params.ao_samples, seed);
        if scene.any_hit(origin, direction, bias, params.ao_distance) {
            occluded += 1;
        }
    }
    let ambient_occlusion = 1.0 - occluded as f32 / params.ao_samples.max(1) as f32;

    let inside = point - face_normal * bias;
    let distance = params.thickness_distance;
    let mut total = 0.0;
    for sample in 0..params.thickness_samples {
        let direction = sample_direction(
            -normal,
            -face_normal,
            sample,
            params.thickness_samples,
            lowbias32(seed ^ THICKNESS_SEED),
        );
        total += scene
            .closest_hit(inside, direction, bias, distance)
            .map_or(distance, |hit| hit.distance);
    }
    let thickness = (total / params.thickness_samples.max(1) as f32 / distance.max(f32::EPSILON))
        .clamp(0.0, 1.0);
    [ambient_occlusion, thickness]
}

/// Cosine-weighted hemisphere direction around `normal` from a rotated
/// Hammersley sequence, reflected above the geometric face plane.
pub fn sample_direction(
    normal: Vec3,
    face_normal: Vec3,
    index: u32,
    count: u32,
    seed: u32,
) -> Vec3 {
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

/// A node of a [`Bvh`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNode {
    pub min: Vec3,
    pub max: Vec3,
    /// Leaf: first index into [`Bvh::order`]. Interior: index of the left
    /// child; the right child follows it.
    pub first: u32,
    /// Triangle count for leaves, zero for interior nodes.
    pub count: u32,
}

/// Bounding-volume hierarchy over mesh triangles for occlusion rays.
pub struct Bvh {
    /// Unique to this build, so tracers can keep its upload between calls.
    id: u64,
    nodes: Vec<BvhNode>,
    order: Vec<u32>,
    triangles: Vec<[Vec3; 3]>,
}

const BVH_LEAF_SIZE: usize = 4;

/// Nodes a traversal can hold pending; deeper subtrees are skipped. Other
/// tracers must use the same depth to match.
pub const BVH_STACK_DEPTH: usize = 128;

impl Bvh {
    pub fn build(mesh: &MeshGeometry) -> Self {
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
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            nodes,
            order,
            triangles,
        }
    }

    /// Identifies this hierarchy among every one built in this process.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Nodes, the root first.
    pub fn nodes(&self) -> &[BvhNode] {
        &self.nodes
    }

    /// Triangle indices in leaf order.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// Corner positions of every triangle, by mesh triangle index.
    pub fn triangles(&self) -> &[[Vec3; 3]] {
        &self.triangles
    }

    pub(crate) fn any_hit(&self, origin: Vec3, direction: Vec3, t_min: f32, t_max: f32) -> bool {
        self.traverse(origin, direction, t_min, t_max, true)
            .is_some()
    }

    pub(crate) fn closest_hit(
        &self,
        origin: Vec3,
        direction: Vec3,
        t_min: f32,
        t_max: f32,
    ) -> Option<RayHit> {
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
        let mut stack = [0_u32; BVH_STACK_DEPTH];
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
pub(crate) struct RayHit {
    pub(crate) distance: f32,
    pub(crate) triangle: u32,
    /// Barycentric weights of the triangle corners at the hit.
    pub(crate) weights: Vec3,
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
