//! Mesh smoothing: auto-smoothed shading normals and crease-aware Loop
//! subdivision.
//!
//! Imported meshes often store unshared vertices per triangle corner, which
//! renders faceted. Both operations weld vertices by position first, so they
//! work the same on welded and unwelded meshes, and both keep edges sharper
//! than the smoothing angle hard.

use std::collections::HashMap;

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::topology::weld_positions;
use crate::{MeshError, MeshGeometry, MeshVertex};

pub const DEFAULT_SMOOTHING_ANGLE_DEGREES: f32 = 60.0;
pub const MAX_SUBDIVISION_LEVELS: u32 = 3;
/// Subdivision refuses to produce more triangles than this.
pub const MAX_SUBDIVIDED_TRIANGLES: usize = 4_194_304;

/// How a mesh is smoothed for display, painting, and baking.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshSmoothing {
    /// Faces meeting at less than this angle share smooth normals; sharper
    /// edges stay hard. `0` gives flat shading, `180` smooths everything.
    pub angle_degrees: f32,
    /// Loop subdivision levels; each level quadruples the triangle count.
    pub subdivision_levels: u32,
}

impl Default for MeshSmoothing {
    fn default() -> Self {
        Self {
            angle_degrees: DEFAULT_SMOOTHING_ANGLE_DEGREES,
            subdivision_levels: 0,
        }
    }
}

impl MeshSmoothing {
    pub fn is_valid(&self) -> bool {
        self.angle_degrees.is_finite()
            && (0.0..=180.0).contains(&self.angle_degrees)
            && self.subdivision_levels <= MAX_SUBDIVISION_LEVELS
    }

    /// Applies subdivision; shading normals are computed separately with
    /// [`vertex_normals`].
    pub fn apply(&self, mesh: &MeshGeometry) -> Result<MeshGeometry, MeshError> {
        subdivide(mesh, self.subdivision_levels, self.angle_degrees)
    }
}

fn smoothing_cosine(angle_degrees: f32) -> f32 {
    if angle_degrees >= 180.0 {
        -1.0 - f32::EPSILON
    } else {
        angle_degrees.clamp(0.0, 180.0).to_radians().cos()
    }
}

/// Area-weighted face normals (unnormalised) and unit face normals.
fn face_normals(mesh: &MeshGeometry) -> (Vec<Vec3>, Vec<Vec3>) {
    let areas = mesh
        .triangles
        .iter()
        .map(|triangle| {
            let [a, b, c] = triangle.map(|index| {
                mesh.vertices
                    .get(index as usize)
                    .map_or(Vec3::ZERO, |vertex| vertex.position)
            });
            (b - a).cross(c - a)
        })
        .collect::<Vec<_>>();
    let units = areas.iter().map(|area| area.normalize_or_zero()).collect();
    (areas, units)
}

/// Shading normal for each triangle corner: the area-weighted average of
/// faces around the welded corner whose normals are within `angle_degrees`.
pub fn corner_normals(mesh: &MeshGeometry, angle_degrees: f32) -> Vec<[Vec3; 3]> {
    let (areas, units) = face_normals(mesh);
    let welded = weld_positions(mesh);
    let mut incident = HashMap::<u32, Vec<u32>>::new();
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        for vertex in triangle {
            if let Some(canonical) = welded.get(*vertex as usize) {
                incident
                    .entry(*canonical)
                    .or_default()
                    .push(triangle_index as u32);
            }
        }
    }
    let threshold = smoothing_cosine(angle_degrees);
    mesh.triangles
        .iter()
        .enumerate()
        .map(|(triangle_index, triangle)| {
            let own = units[triangle_index];
            triangle.map(|vertex| {
                let Some(canonical) = welded.get(vertex as usize) else {
                    return own;
                };
                incident
                    .get(canonical)
                    .into_iter()
                    .flatten()
                    .filter(|other| {
                        **other as usize == triangle_index
                            || units[**other as usize].dot(own) >= threshold
                    })
                    .map(|other| areas[*other as usize])
                    .sum::<Vec3>()
                    .normalize_or(own)
            })
        })
        .collect()
}

/// Per-vertex shading normals. A vertex shared by corners with different
/// smoothing groups receives the average of its corner normals.
pub fn vertex_normals(mesh: &MeshGeometry, angle_degrees: f32) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; mesh.vertices.len()];
    for (triangle, corners) in mesh
        .triangles
        .iter()
        .zip(corner_normals(mesh, angle_degrees))
    {
        for (vertex, normal) in triangle.iter().zip(corners) {
            if let Some(slot) = normals.get_mut(*vertex as usize) {
                *slot += normal;
            }
        }
    }
    normals
        .into_iter()
        .map(|normal| normal.normalize_or(Vec3::Z))
        .collect()
}

/// Per-vertex tangents for tangent-space normal mapping: `xyz` is the unit
/// tangent along +U, orthogonalised against `normals`, and `w` is the
/// bitangent sign so that `bitangent = cross(normal, tangent) * w` points
/// along +V. The renderer and the normal-map baker both use this, so baked
/// normal maps decode in exactly the frame they were encoded in.
pub fn vertex_tangents(mesh: &MeshGeometry, normals: &[Vec3]) -> Vec<glam::Vec4> {
    let count = mesh.vertices.len();
    let mut tangents = vec![Vec3::ZERO; count];
    let mut bitangents = vec![Vec3::ZERO; count];
    for triangle in &mesh.triangles {
        let Some([a, b, c]) = triangle
            .iter()
            .map(|index| mesh.vertices.get(*index as usize).copied())
            .collect::<Option<Vec<_>>>()
            .and_then(|corners| <[MeshVertex; 3]>::try_from(corners).ok())
        else {
            continue;
        };
        let edge_ab = b.position - a.position;
        let edge_ac = c.position - a.position;
        let uv_ab = b.uv - a.uv;
        let uv_ac = c.uv - a.uv;
        let determinant = uv_ab.x * uv_ac.y - uv_ab.y * uv_ac.x;
        if determinant.abs() <= f32::EPSILON {
            continue;
        }
        let inverse = determinant.recip();
        let tangent = (edge_ab * uv_ac.y - edge_ac * uv_ab.y) * inverse;
        let bitangent = (edge_ac * uv_ab.x - edge_ab * uv_ac.x) * inverse;
        for index in triangle {
            tangents[*index as usize] += tangent;
            bitangents[*index as usize] += bitangent;
        }
    }
    (0..count)
        .map(|index| {
            let normal = normals
                .get(index)
                .copied()
                .unwrap_or(Vec3::Z)
                .normalize_or(Vec3::Z);
            let mut tangent = tangents[index] - normal * normal.dot(tangents[index]);
            if tangent.length_squared() <= f32::EPSILON {
                let helper = if normal.x.abs() < 0.9 {
                    Vec3::X
                } else {
                    Vec3::Y
                };
                tangent = normal.cross(helper);
            }
            let tangent = tangent.normalize_or(Vec3::X);
            let sign = if normal.cross(tangent).dot(bitangents[index]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            tangent.extend(sign)
        })
        .collect()
}

/// Crease-aware Loop subdivision. Child triangles of input triangle `t` are
/// `4t..4t + 4`, so per-triangle data can be remapped between levels.
/// UVs are interpolated linearly and UV seams are preserved.
pub fn subdivide(
    mesh: &MeshGeometry,
    levels: u32,
    crease_angle_degrees: f32,
) -> Result<MeshGeometry, MeshError> {
    if levels > MAX_SUBDIVISION_LEVELS {
        return Err(MeshError::TooManyTriangles);
    }
    mesh.validate()?;
    let mut current = mesh.clone();
    for _ in 0..levels {
        if current.triangles.len().saturating_mul(4) > MAX_SUBDIVIDED_TRIANGLES {
            return Err(MeshError::TooManyTriangles);
        }
        current = subdivide_once(&current, crease_angle_degrees);
    }
    Ok(current)
}

#[derive(Default)]
struct WeldedEdge {
    /// Far vertex (welded id) of each adjacent triangle.
    opposite: Vec<u32>,
    triangles: Vec<u32>,
}

fn subdivide_once(mesh: &MeshGeometry, crease_angle_degrees: f32) -> MeshGeometry {
    let welded = weld_positions(mesh);
    let (_, units) = face_normals(mesh);
    let threshold = smoothing_cosine(crease_angle_degrees);
    let position = |welded_id: u32| mesh.vertices[welded_id as usize].position;

    let mut edges = HashMap::<(u32, u32), WeldedEdge>::new();
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        let ids = triangle.map(|vertex| welded[vertex as usize]);
        for corner in 0..3 {
            let a = ids[corner];
            let b = ids[(corner + 1) % 3];
            if a == b {
                continue;
            }
            let edge = edges.entry((a.min(b), a.max(b))).or_default();
            edge.opposite.push(ids[(corner + 2) % 3]);
            edge.triangles.push(triangle_index as u32);
        }
    }

    // An edge is a crease when it is open, non-manifold, or sharper than the
    // smoothing angle.
    let is_crease = |edge: &WeldedEdge| {
        edge.triangles.len() != 2
            || units[edge.triangles[0] as usize].dot(units[edge.triangles[1] as usize]) < threshold
    };

    let mut neighbours = HashMap::<u32, Vec<u32>>::new();
    let mut crease_neighbours = HashMap::<u32, Vec<u32>>::new();
    for ((a, b), edge) in &edges {
        neighbours.entry(*a).or_default().push(*b);
        neighbours.entry(*b).or_default().push(*a);
        if is_crease(edge) {
            crease_neighbours.entry(*a).or_default().push(*b);
            crease_neighbours.entry(*b).or_default().push(*a);
        }
    }

    let mut even_positions = HashMap::<u32, Vec3>::with_capacity(neighbours.len());
    for (vertex, around) in &neighbours {
        let here = position(*vertex);
        let creases = crease_neighbours.get(vertex).map_or(&[][..], Vec::as_slice);
        let moved = match creases.len() {
            0 | 1 => {
                let count = around.len() as f32;
                let beta = if around.len() == 3 {
                    3.0 / 16.0
                } else {
                    3.0 / (8.0 * count)
                };
                let sum = around.iter().map(|other| position(*other)).sum::<Vec3>();
                here * (1.0 - count * beta) + sum * beta
            }
            2 => here * 0.75 + (position(creases[0]) + position(creases[1])) * 0.125,
            _ => here,
        };
        even_positions.insert(*vertex, moved);
    }

    let odd_position = |a: u32, b: u32| {
        let midpoint = (position(a) + position(b)) * 0.5;
        match edges.get(&(a.min(b), a.max(b))) {
            Some(edge) if !is_crease(edge) => {
                (position(a) + position(b)) * 0.375
                    + (position(edge.opposite[0]) + position(edge.opposite[1])) * 0.125
            }
            _ => midpoint,
        }
    };

    let mut vertices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| MeshVertex {
            position: even_positions
                .get(&welded[index])
                .copied()
                .unwrap_or(vertex.position),
            uv: vertex.uv,
        })
        .collect::<Vec<_>>();
    // Midpoint vertices are shared exactly where the input shares both corner
    // vertices, so seams split by the input stay split.
    let mut midpoints = HashMap::<(u32, u32), u32>::new();
    let mut midpoint = |vertices: &mut Vec<MeshVertex>, i: u32, j: u32| {
        *midpoints.entry((i.min(j), i.max(j))).or_insert_with(|| {
            let uv: Vec2 = (mesh.vertices[i as usize].uv + mesh.vertices[j as usize].uv) * 0.5;
            vertices.push(MeshVertex {
                position: odd_position(welded[i as usize], welded[j as usize]),
                uv,
            });
            (vertices.len() - 1) as u32
        })
    };

    let mut triangles = Vec::with_capacity(mesh.triangles.len() * 4);
    for [a, b, c] in &mesh.triangles {
        let ab = midpoint(&mut vertices, *a, *b);
        let bc = midpoint(&mut vertices, *b, *c);
        let ca = midpoint(&mut vertices, *c, *a);
        triangles.extend([[*a, ab, ca], [ab, *b, bc], [ca, bc, *c], [ab, bc, ca]]);
    }
    MeshGeometry {
        vertices,
        triangles,
        // Each triangle splits into four children at 4i..4i+4.
        triangle_materials: mesh
            .triangle_materials
            .iter()
            .flat_map(|slot| [*slot; 4])
            .collect(),
        material_slots: mesh.material_slots.clone(),
    }
}

/// Maps triangle indices selected at subdivision level `from` onto level
/// `to`. Refining selects every child; coarsening keeps a parent when any of
/// its children was selected.
pub fn remap_subdivided_triangles(
    triangles: impl IntoIterator<Item = u32>,
    from: u32,
    to: u32,
) -> Vec<u32> {
    let mut remapped = triangles
        .into_iter()
        .flat_map(|triangle| {
            if to >= from {
                let factor = 4_u64.pow(to - from);
                let first = u64::from(triangle) * factor;
                (first..first + factor)
                    .filter_map(|child| u32::try_from(child).ok())
                    .collect::<Vec<_>>()
            } else {
                vec![triangle / 4_u32.pow(from - to)]
            }
        })
        .collect::<Vec<_>>();
    remapped.sort_unstable();
    remapped.dedup();
    remapped
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::PrimitivePreset;

    /// One quad split into two triangles, unwelded like FBX imports, bent
    /// 30 degrees along the shared diagonal.
    fn unwelded_bent_quad() -> MeshGeometry {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(1.0, 0.0, 1.0);
        let d = Vec3::new(0.0, 0.5, 1.0);
        let vertex = |position: Vec3| MeshVertex {
            position,
            uv: Vec2::new(position.x, position.z),
        };
        MeshGeometry {
            vertices: vec![
                vertex(a),
                vertex(c),
                vertex(b),
                vertex(a),
                vertex(d),
                vertex(c),
            ],
            triangles: vec![[0, 1, 2], [3, 4, 5]],
            ..MeshGeometry::default()
        }
    }

    #[test]
    fn tc_0038_unwelded_corners_share_smooth_normals_below_the_angle() {
        let mesh = unwelded_bent_quad();
        let smooth = vertex_normals(&mesh, 60.0);
        // Vertex 0 and 3 are the same position in different triangles.
        assert!((smooth[0] - smooth[3]).length() < 1e-5);
        let flat = vertex_normals(&mesh, 0.0);
        assert!(
            (flat[0] - flat[3]).length() > 0.1,
            "flat shading keeps facets"
        );

        let cube = PrimitivePreset::Cube.build_mesh();
        let normals = vertex_normals(&cube, 60.0);
        assert!(
            (normals[0] - Vec3::Z).length() < 1e-5,
            "cube edges stay hard"
        );
    }

    #[test]
    fn tc_0039_subdivision_rounds_smooth_meshes_and_keeps_creases() {
        let sphere = PrimitivePreset::Sphere.build_mesh();
        let subdivided = subdivide(&sphere, 1, 60.0).unwrap();
        assert_eq!(subdivided.triangles.len(), sphere.triangles.len() * 4);
        assert!(subdivided.validate().is_ok());
        // Loop smoothing pulls vertices toward a rounder surface: the spread
        // of radii shrinks compared with linear midpoints.
        let radius_spread = |mesh: &MeshGeometry| {
            let radii = mesh
                .vertices
                .iter()
                .map(|vertex| vertex.position.length())
                .collect::<Vec<_>>();
            let max = radii.iter().copied().fold(f32::MIN, f32::max);
            let min = radii.iter().copied().fold(f32::MAX, f32::min);
            max - min
        };
        let linear = subdivide(&sphere, 1, 0.0).unwrap();
        assert!(radius_spread(&subdivided) < radius_spread(&linear));

        // Every cube edge is a 90-degree crease, so the box keeps its shape.
        let cube = PrimitivePreset::Cube.build_mesh();
        let smoothed_cube = subdivide(&cube, 2, 60.0).unwrap();
        assert_eq!(smoothed_cube.triangles.len(), 12 * 16);
        assert!(
            smoothed_cube
                .vertices
                .iter()
                .all(|vertex| vertex.position.abs().max_element() > 0.4999)
        );
        assert_eq!(subdivide(&cube, 4, 60.0), Err(MeshError::TooManyTriangles));
    }

    #[test]
    fn tc_0040_subdivided_triangle_selections_remap_between_levels() {
        assert_eq!(remap_subdivided_triangles([1], 0, 1), vec![4, 5, 6, 7]);
        assert_eq!(remap_subdivided_triangles([5, 6, 17], 1, 0), vec![1, 4]);
        assert_eq!(remap_subdivided_triangles([2], 1, 1), vec![2]);
    }
}
