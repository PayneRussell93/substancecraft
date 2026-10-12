//! Mesh connectivity used for polygon selection: faces, connected mesh
//! elements, and UV islands.

use std::collections::HashMap;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::MeshGeometry;

/// Granularity of a polygon selection pick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SelectionMode {
    /// The single picked triangle.
    Triangle,
    /// Every triangle connected to the pick through shared positions — a
    /// separate object or loose part of the mesh.
    #[default]
    Element,
    /// Every triangle connected to the pick in UV space.
    UvIsland,
}

impl SelectionMode {
    pub const ALL: [Self; 3] = [Self::Triangle, Self::Element, Self::UvIsland];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Triangle => "Face",
            Self::Element => "Element",
            Self::UvIsland => "UV Island",
        }
    }
}

/// Per-triangle element and UV-island membership for one mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshTopology {
    element_of: Vec<u32>,
    island_of: Vec<u32>,
    element_count: u32,
    island_count: u32,
}

impl MeshTopology {
    /// Builds connectivity. Vertices are welded by quantised position (for
    /// elements) and by position plus UV (for islands), so meshes split at
    /// hard edges or seams still connect correctly. Triangles with invalid
    /// indices are treated as isolated.
    pub fn build(mesh: &MeshGeometry) -> Self {
        let welded = weld_positions(mesh);
        let uv_welded = weld_positions_and_uvs(mesh);
        let (element_of, element_count) = connected_triangles(mesh, &welded);
        let (island_of, island_count) = connected_triangles(mesh, &uv_welded);
        Self {
            element_of,
            island_of,
            element_count,
            island_count,
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.element_of.len()
    }

    pub fn element_count(&self) -> u32 {
        self.element_count
    }

    pub fn island_count(&self) -> u32 {
        self.island_count
    }

    pub fn element_of(&self, triangle: u32) -> Option<u32> {
        self.element_of.get(triangle as usize).copied()
    }

    pub fn island_of(&self, triangle: u32) -> Option<u32> {
        self.island_of.get(triangle as usize).copied()
    }

    /// Triangles selected by picking `seed` with `mode`, in ascending order.
    pub fn select(&self, seed: u32, mode: SelectionMode) -> Vec<u32> {
        let groups = match mode {
            SelectionMode::Triangle => {
                return if (seed as usize) < self.element_of.len() {
                    vec![seed]
                } else {
                    Vec::new()
                };
            }
            SelectionMode::Element => &self.element_of,
            SelectionMode::UvIsland => &self.island_of,
        };
        let Some(group) = groups.get(seed as usize).copied() else {
            return Vec::new();
        };
        groups
            .iter()
            .enumerate()
            .filter(|(_, candidate)| **candidate == group)
            .map(|(triangle, _)| triangle as u32)
            .collect()
    }
}

/// Canonical welded-vertex id for every vertex, keyed by quantised position
/// relative to the mesh bounds.
pub(crate) fn weld_positions(mesh: &MeshGeometry) -> Vec<u32> {
    let quantize = position_quantizer(mesh);
    weld_by(mesh, |vertex| {
        let [x, y, z] = quantize(vertex.position);
        (x, y, z, 0, 0)
    })
}

fn weld_positions_and_uvs(mesh: &MeshGeometry) -> Vec<u32> {
    let quantize = position_quantizer(mesh);
    weld_by(mesh, |vertex| {
        let [x, y, z] = quantize(vertex.position);
        let u = (f64::from(vertex.uv.x) * f64::from(1u32 << 20)).round() as i64;
        let v = (f64::from(vertex.uv.y) * f64::from(1u32 << 20)).round() as i64;
        (x, y, z, u, v)
    })
}

fn position_quantizer(mesh: &MeshGeometry) -> impl Fn(Vec3) -> [i64; 3] {
    let (minimum, maximum) = mesh_bounds(mesh);
    let extent = (maximum - minimum).max_element().max(f32::EPSILON);
    let steps = f64::from(1u32 << 20);
    move |position: Vec3| {
        let relative = (position - minimum) / extent;
        [
            (f64::from(relative.x) * steps).round() as i64,
            (f64::from(relative.y) * steps).round() as i64,
            (f64::from(relative.z) * steps).round() as i64,
        ]
    }
}

pub(crate) fn mesh_bounds(mesh: &MeshGeometry) -> (Vec3, Vec3) {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for vertex in &mesh.vertices {
        if vertex.position.is_finite() {
            minimum = minimum.min(vertex.position);
            maximum = maximum.max(vertex.position);
        }
    }
    if minimum.cmpgt(maximum).any() {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    (minimum, maximum)
}

type WeldKey = (i64, i64, i64, i64, i64);

fn weld_by(mesh: &MeshGeometry, key: impl Fn(&crate::MeshVertex) -> WeldKey) -> Vec<u32> {
    let mut canonical = HashMap::<WeldKey, u32>::with_capacity(mesh.vertices.len());
    mesh.vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| *canonical.entry(key(vertex)).or_insert(index as u32))
        .collect()
}

fn connected_triangles(mesh: &MeshGeometry, welded: &[u32]) -> (Vec<u32>, u32) {
    let mut sets = DisjointSet::new(mesh.triangles.len());
    let mut first_triangle_at = HashMap::<u32, usize>::with_capacity(welded.len());
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        for vertex in triangle {
            let Some(canonical) = welded.get(*vertex as usize).copied() else {
                continue;
            };
            let first = *first_triangle_at.entry(canonical).or_insert(triangle_index);
            sets.union(first, triangle_index);
        }
    }
    let mut labels = HashMap::<usize, u32>::new();
    let groups = (0..mesh.triangles.len())
        .map(|triangle| {
            let root = sets.find(triangle);
            let next = labels.len() as u32;
            *labels.entry(root).or_insert(next)
        })
        .collect();
    (groups, labels.len() as u32)
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            rank: vec![0; count],
        }
    }

    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            self.parent[node] = self.parent[self.parent[node]];
            node = self.parent[node];
        }
        node
    }

    fn union(&mut self, left: usize, right: usize) {
        let left = self.find(left);
        let right = self.find(right);
        if left == right {
            return;
        }
        match self.rank[left].cmp(&self.rank[right]) {
            std::cmp::Ordering::Less => self.parent[left] = right,
            std::cmp::Ordering::Greater => self.parent[right] = left,
            std::cmp::Ordering::Equal => {
                self.parent[right] = left;
                self.rank[left] = self.rank[left].saturating_add(1);
            }
        }
    }
}

/// Why the mesh diagnostics overlay draws an edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgeFlags {
    /// The two faces use different UVs along the edge.
    pub uv_seam: bool,
    /// The faces' shading normals do not match (a split-normal edge).
    pub hard: bool,
    /// The tangent frames differ, so tangent-space normal maps can show a
    /// seam here.
    pub tangent_split: bool,
}

impl EdgeFlags {
    pub fn any(self) -> bool {
        self.uv_seam || self.hard || self.tangent_split
    }
}

/// An edge shared by two faces that the diagnostics overlay marks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagnosticEdge {
    pub a: Vec3,
    pub b: Vec3,
    /// The two triangles meeting at the edge.
    pub faces: [u32; 2],
    pub flags: EdgeFlags,
}

/// Edges where shading, UVs, or tangent frames split, found by welding
/// positions and comparing the two faces at each shared edge. Normals and
/// tangents are the ones the renderer uses for `smoothing_angle_degrees`.
/// Open (one-face) and non-manifold edges are skipped.
pub fn diagnostic_edges(mesh: &MeshGeometry, smoothing_angle_degrees: f32) -> Vec<DiagnosticEdge> {
    let welded = weld_positions(mesh);
    let corners = crate::corner_normals(mesh, smoothing_angle_degrees);
    let vertex_normals = crate::vertex_normals(mesh, smoothing_angle_degrees);
    let tangents = crate::vertex_tangents(mesh, &vertex_normals);
    // Per welded edge (low, high): triangle, then for each end the vertex
    // and the corner slot.
    let mut edges = HashMap::<(u32, u32), Vec<(u32, [(u32, usize); 2])>>::new();
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        for corner in 0..3 {
            let next = (corner + 1) % 3;
            let (Some(&a), Some(&b)) = (
                welded.get(triangle[corner] as usize),
                welded.get(triangle[next] as usize),
            ) else {
                continue;
            };
            if a == b {
                continue;
            }
            let ends = if a < b {
                [(triangle[corner], corner), (triangle[next], next)]
            } else {
                [(triangle[next], next), (triangle[corner], corner)]
            };
            edges
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push((triangle_index as u32, ends));
        }
    }
    let normal_limit = 1.0_f32.to_radians().cos();
    let tangent_limit = 5.0_f32.to_radians().cos();
    let mut found: Vec<((u32, u32), DiagnosticEdge)> = edges
        .into_iter()
        .filter_map(|(key, sides)| {
            let [(first, first_ends), (second, second_ends)] = sides.as_slice() else {
                return None;
            };
            let mut flags = EdgeFlags::default();
            for end in 0..2 {
                let (vertex_a, corner_a) = first_ends[end];
                let (vertex_b, corner_b) = second_ends[end];
                let (va, vb) = (
                    mesh.vertices.get(vertex_a as usize)?,
                    mesh.vertices.get(vertex_b as usize)?,
                );
                flags.uv_seam |= (va.uv - vb.uv).length() > 1e-5;
                let (na, nb) = (
                    corners.get(*first as usize)?[corner_a],
                    corners.get(*second as usize)?[corner_b],
                );
                flags.hard |= na.dot(nb) < normal_limit;
                let (ta, tb) = (
                    tangents.get(vertex_a as usize)?,
                    tangents.get(vertex_b as usize)?,
                );
                flags.tangent_split |=
                    ta.truncate().dot(tb.truncate()) < tangent_limit || ta.w != tb.w;
            }
            let (a, b) = (
                mesh.vertices.get(first_ends[0].0 as usize)?.position,
                mesh.vertices.get(first_ends[1].0 as usize)?.position,
            );
            flags.any().then_some((
                key,
                DiagnosticEdge {
                    a,
                    b,
                    faces: [*first, *second],
                    flags,
                },
            ))
        })
        .collect();
    // Map order is random; sort so the overlay is stable.
    found.sort_by_key(|(key, _)| *key);
    found.into_iter().map(|(_, edge)| edge).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{MeshVertex, PrimitivePreset};

    #[test]
    fn tc_0247_diagnostic_edges_mark_seams_hard_edges_and_tangent_splits() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let edges = diagnostic_edges(&cube, 30.0);
        // Every cube edge is both a UV seam and hard; face diagonals are not
        // marked.
        assert_eq!(edges.len(), 12);
        assert!(
            edges
                .iter()
                .all(|edge| edge.flags.uv_seam && edge.flags.hard)
        );
        assert!(edges.iter().all(|edge| (edge.a - edge.b).length() > 0.99));
        // Smoothing everything leaves the seams but no hard edges.
        let smooth = diagnostic_edges(&cube, 180.0);
        assert_eq!(smooth.len(), 12);
        assert!(
            smooth
                .iter()
                .all(|edge| edge.flags.uv_seam && !edge.flags.hard)
        );
        assert!(smooth.iter().any(|edge| edge.flags.tangent_split));
        assert_eq!(diagnostic_edges(&cube, 30.0), edges);
        // A flat plane has nothing to report.
        assert!(diagnostic_edges(&PrimitivePreset::Plane.build_mesh(), 30.0).is_empty());
    }

    #[test]
    fn tc_0030_cube_is_one_element_with_six_uv_islands() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let topology = MeshTopology::build(&cube);

        assert_eq!(topology.triangle_count(), 12);
        assert_eq!(topology.element_count(), 1);
        assert_eq!(topology.island_count(), 6);
        assert_eq!(topology.select(0, SelectionMode::Triangle), vec![0]);
        assert_eq!(topology.select(0, SelectionMode::UvIsland), vec![0, 1]);
        assert_eq!(topology.select(5, SelectionMode::Element).len(), 12);
        assert!(topology.select(99, SelectionMode::Element).is_empty());
    }

    #[test]
    fn tc_0031_disconnected_parts_are_separate_elements() {
        let mut mesh = PrimitivePreset::Plane.build_mesh();
        let offset = mesh.vertices.len() as u32;
        let shifted = mesh
            .vertices
            .iter()
            .map(|vertex| MeshVertex {
                position: vertex.position + Vec3::new(3.0, 0.0, 0.0),
                uv: vertex.uv,
            })
            .collect::<Vec<_>>();
        mesh.vertices.extend(shifted);
        let extra = mesh
            .triangles
            .iter()
            .map(|triangle| triangle.map(|index| index + offset))
            .collect::<Vec<_>>();
        mesh.triangles.extend(extra);

        let topology = MeshTopology::build(&mesh);

        assert_eq!(topology.element_count(), 2);
        assert_eq!(topology.select(3, SelectionMode::Element), vec![2, 3]);
        assert_ne!(topology.element_of(0), topology.element_of(2));
    }
}
