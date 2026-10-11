//! Stroke assistance shared by the 3D and UV viewports: mirror symmetry,
//! lazy-mouse stabilization, straight lines, seam detection, and spline
//! paths.

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::{Axis, MeshGeometry, MeshTopology};

/// A point on the mesh surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceHit {
    pub triangle: u32,
    pub position: Vec3,
    /// Unit face normal.
    pub normal: Vec3,
    pub uv: Vec2,
}

/// Mirror painting across an object-space plane perpendicular to `axis`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Symmetry {
    pub axis: Axis,
    /// Position of the mirror plane along `axis`.
    pub offset: f32,
}

impl Symmetry {
    pub fn new(axis: Axis) -> Self {
        Self { axis, offset: 0.0 }
    }

    /// `position` reflected across the mirror plane.
    pub fn mirror(self, position: Vec3) -> Vec3 {
        let mut mirrored = position;
        let component = match self.axis {
            Axis::X => &mut mirrored.x,
            Axis::Y => &mut mirrored.y,
            Axis::Z => &mut mirrored.z,
        };
        *component = 2.0 * self.offset - *component;
        mirrored
    }

    /// The surface point mirroring `hit`: the closest surface point to the
    /// reflected position, so asymmetric meshes still get a sensible
    /// mirrored stroke.
    pub fn mirror_hit(self, mesh: &MeshGeometry, hit: &SurfaceHit) -> Option<SurfaceHit> {
        mesh.closest_surface_point(self.mirror(hit.position))
    }

    /// Mirrors a UV-space point across `u = 0.5`, for painting in the UV
    /// viewport of a symmetric layout.
    pub fn mirror_uv(uv: Vec2) -> Vec2 {
        Vec2::new(1.0 - uv.x, uv.y)
    }
}

/// Closest point to `point` on triangle `abc`, as barycentric weights.
fn closest_barycentric(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    // Real-Time Collision Detection, 5.1.5.
    let (ab, ac, ap) = (b - a, c - a, point - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return Vec3::X;
    }
    let bp = point - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return Vec3::Y;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return Vec3::new(1.0 - v, v, 0.0);
    }
    let cp = point - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return Vec3::Z;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return Vec3::new(1.0 - w, 0.0, w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return Vec3::new(0.0, 1.0 - w, w);
    }
    let denominator = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denominator, vc * denominator);
    Vec3::new(1.0 - v - w, v, w)
}

impl MeshGeometry {
    /// The surface point nearest `point`, with its UV and face normal.
    pub fn closest_surface_point(&self, point: Vec3) -> Option<SurfaceHit> {
        let mut best: Option<(f32, SurfaceHit)> = None;
        for (index, triangle) in self.triangles.iter().enumerate() {
            let Some([a, b, c]) = triangle
                .iter()
                .map(|corner| self.vertices.get(*corner as usize).copied())
                .collect::<Option<Vec<_>>>()
                .and_then(|corners| <[_; 3]>::try_from(corners).ok())
            else {
                continue;
            };
            let weights = closest_barycentric(point, a.position, b.position, c.position);
            let position = a.position * weights.x + b.position * weights.y + c.position * weights.z;
            let distance = position.distance_squared(point);
            if best.is_some_and(|(nearest, _)| nearest <= distance) {
                continue;
            }
            let normal = (b.position - a.position)
                .cross(c.position - a.position)
                .normalize_or(Vec3::Z);
            best = Some((
                distance,
                SurfaceHit {
                    triangle: index as u32,
                    position,
                    normal,
                    uv: a.uv * weights.x + b.uv * weights.y + c.uv * weights.z,
                },
            ));
        }
        best.map(|(_, hit)| hit)
    }
}

/// Lazy-mouse stabilization: the brush trails the cursor on a string of
/// length `radius` and only moves when the cursor pulls it, smoothing
/// jitter while keeping deliberate motion exact.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LazyStabilizer {
    pub radius: f32,
    anchor: Option<Vec2>,
}

impl LazyStabilizer {
    pub fn new(radius: f32) -> Self {
        Self {
            radius: radius.max(0.0),
            anchor: None,
        }
    }

    /// Starts a new stroke at `point`.
    pub fn reset(&mut self) {
        self.anchor = None;
    }

    /// Feeds the cursor position; returns the brush position when it moved
    /// (always on the first point of a stroke).
    pub fn update(&mut self, cursor: Vec2) -> Option<Vec2> {
        let Some(anchor) = self.anchor else {
            self.anchor = Some(cursor);
            return Some(cursor);
        };
        let offset = cursor - anchor;
        let distance = offset.length();
        if distance <= self.radius {
            return None;
        }
        let moved = anchor + offset * ((distance - self.radius) / distance);
        self.anchor = Some(moved);
        Some(moved)
    }
}

/// Evenly spaced points from `from` to `to`, both included, at most
/// `spacing` apart: a straight line to project onto the surface.
pub fn line_points(from: Vec2, to: Vec2, spacing: f32) -> Vec<Vec2> {
    let steps = ((from.distance(to) / spacing.max(0.5)).ceil() as usize).clamp(1, 4_096);
    (0..=steps)
        .map(|step| from.lerp(to, step as f32 / steps as f32))
        .collect()
}

/// True when moving from triangle `previous` to `next` crosses a UV seam
/// into another UV island, so the stroke must restart rather than draw a
/// line across UV space.
pub fn crosses_uv_seam(topology: &MeshTopology, previous: u32, next: u32) -> bool {
    match (topology.island_of(previous), topology.island_of(next)) {
        (Some(from), Some(to)) => from != to,
        _ => false,
    }
}

/// Control points of an editable spline path (UV space).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PathSpec {
    pub controls: Vec<Vec2>,
    #[serde(default)]
    pub closed: bool,
}

impl PathSpec {
    /// Catmull-Rom curve through every control point, `samples` points per
    /// segment.
    pub fn sample(&self, samples: usize) -> Vec<Vec2> {
        let controls = &self.controls;
        let count = controls.len();
        if count < 2 {
            return controls.clone();
        }
        let samples = samples.clamp(1, 64);
        let at = |index: isize| -> Vec2 {
            if self.closed {
                controls[index.rem_euclid(count as isize) as usize]
            } else {
                controls[index.clamp(0, count as isize - 1) as usize]
            }
        };
        let segments = if self.closed { count } else { count - 1 };
        let mut points = Vec::with_capacity(segments * samples + 1);
        for segment in 0..segments as isize {
            let (p0, p1, p2, p3) = (
                at(segment - 1),
                at(segment),
                at(segment + 1),
                at(segment + 2),
            );
            for step in 0..samples {
                let t = step as f32 / samples as f32;
                let (t2, t3) = (t * t, t * t * t);
                points.push(
                    0.5 * (2.0 * p1
                        + (p2 - p0) * t
                        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3),
                );
            }
        }
        points.push(if self.closed {
            controls[0]
        } else {
            controls[count - 1]
        });
        points
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::PrimitivePreset;

    #[test]
    fn tc_0160_symmetry_mirrors_surface_hits_onto_the_opposite_side() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let hit = cube
            .closest_surface_point(Vec3::new(0.3, 0.1, 0.9))
            .unwrap();
        assert!((hit.position - Vec3::new(0.3, 0.1, 0.5)).length() < 1e-5);
        assert_eq!(hit.normal, Vec3::Z);

        let symmetry = Symmetry::new(Axis::X);
        let mirrored = symmetry.mirror_hit(&cube, &hit).unwrap();
        assert!((mirrored.position - Vec3::new(-0.3, 0.1, 0.5)).length() < 1e-5);
        // Same face, mirrored horizontally in its UV cell.
        assert_eq!(mirrored.triangle / 2, hit.triangle / 2);
        assert!(mirrored.uv.x < hit.uv.x);

        // Mirroring across Z lands on the back face.
        let back = Symmetry::new(Axis::Z).mirror_hit(&cube, &hit).unwrap();
        assert_eq!(back.normal, Vec3::NEG_Z);
        // An offset plane moves the reflection.
        let offset = Symmetry {
            axis: Axis::X,
            offset: 0.1,
        };
        assert!((offset.mirror(Vec3::new(0.3, 0.0, 0.0)).x - -0.1).abs() < 1e-6);
        assert_eq!(
            Symmetry::mirror_uv(Vec2::new(0.2, 0.7)),
            Vec2::new(0.8, 0.7)
        );
        assert!(
            MeshGeometry::default()
                .closest_surface_point(Vec3::ZERO)
                .is_none()
        );
    }

    #[test]
    fn tc_0161_stabilizer_lines_seams_and_paths_assist_strokes() {
        // The lazy brush ignores jitter inside its radius and trails behind.
        let mut lazy = LazyStabilizer::new(10.0);
        assert_eq!(lazy.update(Vec2::ZERO), Some(Vec2::ZERO));
        assert_eq!(lazy.update(Vec2::new(4.0, 3.0)), None);
        assert_eq!(
            lazy.update(Vec2::new(30.0, 0.0)),
            Some(Vec2::new(20.0, 0.0))
        );
        lazy.reset();
        assert_eq!(lazy.update(Vec2::new(5.0, 5.0)), Some(Vec2::new(5.0, 5.0)));

        // Straight lines are evenly sampled end to end.
        let line = line_points(Vec2::ZERO, Vec2::new(10.0, 0.0), 2.0);
        assert_eq!(line.len(), 6);
        assert_eq!(line[5], Vec2::new(10.0, 0.0));

        // Cube faces are separate UV islands; triangles of a face are not.
        let topology = MeshTopology::build(&PrimitivePreset::Cube.build_mesh());
        assert!(!crosses_uv_seam(&topology, 0, 1));
        assert!(crosses_uv_seam(&topology, 0, 2));

        // Splines pass through every control point.
        let path = PathSpec {
            controls: vec![Vec2::ZERO, Vec2::new(1.0, 1.0), Vec2::new(2.0, 0.0)],
            closed: false,
        };
        let points = path.sample(8);
        assert_eq!(points.len(), 17);
        assert_eq!(points[0], Vec2::ZERO);
        assert!((points[8] - Vec2::new(1.0, 1.0)).length() < 1e-6);
        assert_eq!(points[16], Vec2::new(2.0, 0.0));
        let closed = PathSpec {
            closed: true,
            ..path
        };
        assert_eq!(closed.sample(4).last(), Some(&Vec2::ZERO));
    }
}
