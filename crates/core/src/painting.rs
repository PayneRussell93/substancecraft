//! Stroke assistance shared by the 3D and UV viewports: mirror symmetry,
//! lazy-mouse stabilization, straight lines, seam detection, and spline
//! paths.

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::brush::{MAX_STAMP_STRETCH, StampFrame};
use crate::{Axis, MeshGeometry, MeshTopology, MeshVertex};

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
/// With `radial` of 2 or more it instead repeats strokes `radial` times
/// around the `axis` line through the object origin.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Symmetry {
    pub axis: Axis,
    /// Position of the mirror plane along `axis`.
    pub offset: f32,
    /// Radial copies around `axis` (including the original), at most
    /// [`MAX_RADIAL_COPIES`]; 0 or 1 mirrors instead.
    #[serde(default)]
    pub radial: u32,
}

/// The most radial symmetry copies a stroke can have.
pub const MAX_RADIAL_COPIES: u32 = 32;

impl Symmetry {
    pub fn new(axis: Axis) -> Self {
        Self {
            axis,
            offset: 0.0,
            radial: 0,
        }
    }

    /// `count` copies around `axis` (clamped to [`MAX_RADIAL_COPIES`]).
    pub fn radial(axis: Axis, count: u32) -> Self {
        Self {
            axis,
            offset: 0.0,
            radial: count.min(MAX_RADIAL_COPIES),
        }
    }

    pub fn is_radial(self) -> bool {
        self.radial >= 2
    }

    /// Where the symmetric copies of `position` fall: the mirror image, or
    /// the radial copies after the original, in turn order.
    pub fn copies(self, position: Vec3) -> Vec<Vec3> {
        if !self.is_radial() {
            return vec![self.mirror(position)];
        }
        let count = self.radial.min(MAX_RADIAL_COPIES);
        let axis = match self.axis {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        };
        (1..count)
            .map(|turn| {
                let angle = std::f32::consts::TAU * turn as f32 / count as f32;
                glam::Quat::from_axis_angle(axis, angle) * position
            })
            .collect()
    }

    /// The surface points of every copy of `hit`, in [`Self::copies`] order;
    /// `None` where a copy finds no surface.
    pub fn symmetric_hits(self, mesh: &MeshGeometry, hit: &SurfaceHit) -> Vec<Option<SurfaceHit>> {
        self.copies(hit.position)
            .into_iter()
            .map(|position| mesh.closest_surface_point(position))
            .collect()
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

    /// The object-space outline a UV-space brush of `uv_radius` (per axis)
    /// covers around `center` on triangle `triangle`: `segments` points of
    /// the UV circle mapped through the triangle's UV-to-surface mapping.
    /// `None` when the triangle's UVs are degenerate. This is the brush
    /// footprint as it lands on the surface, stretched where the UVs are.
    pub fn uv_brush_outline(
        &self,
        triangle: u32,
        center: Vec3,
        uv_radius: Vec2,
        segments: usize,
    ) -> Option<Vec<Vec3>> {
        let [a, b, c] = self.triangle_corners(triangle)?;
        let (along_u, along_v) = uv_axes(a, b, c)?;
        Some(
            (0..segments.max(3))
                .map(|index| {
                    let angle = index as f32 / segments.max(3) as f32 * std::f32::consts::TAU;
                    center
                        + along_u * (uv_radius.x * angle.cos())
                        + along_v * (uv_radius.y * angle.sin())
                })
                .collect(),
        )
    }

    /// How a stamp painted on `triangle` must lie in a `width` × `height`
    /// texture to land round and upright on the surface: its x axis along
    /// the surface direction of increasing U, its y axis across it, and
    /// the right way round on mirrored UV islands. `None` when the
    /// triangle or its UVs are degenerate, or the UVs stretch one way more
    /// than [`MAX_STAMP_STRETCH`] times.
    pub fn stamp_frame(&self, triangle: u32, width: u32, height: u32) -> Option<StampFrame> {
        let [a, b, c] = self.triangle_corners(triangle)?;
        let (along_u, along_v) = uv_axes(a, b, c)?;
        // Surface travelled per texel along each texture axis (V up).
        let per_x = along_u / width.saturating_sub(1).max(1) as f32;
        let per_y = along_v / height.saturating_sub(1).max(1) as f32;
        let tangent = per_x.try_normalize()?;
        let normal = (b.position - a.position)
            .cross(c.position - a.position)
            .try_normalize()?;
        let bitangent = normal.cross(tangent);
        let raw = [
            tangent.dot(per_x),
            bitangent.dot(per_x),
            tangent.dot(per_y),
            bitangent.dot(per_y),
        ];
        let determinant = raw[0] * raw[3] - raw[1] * raw[2];
        if !determinant.is_finite() || determinant == 0.0 {
            return None;
        }
        // Keep the texel area: the stamp's radius stays in texels.
        let scale = determinant.abs().sqrt();
        let frame = StampFrame(raw.map(|value| value / scale));
        (frame.0.iter().all(|value| value.is_finite()) && frame.stretch() <= MAX_STAMP_STRETCH)
            .then_some(frame)
    }

    /// UV at `point` projected onto `triangle`'s plane, extrapolated past
    /// its edges. `None` for a degenerate triangle.
    pub fn uv_on_plane(&self, triangle: u32, point: Vec3) -> Option<Vec2> {
        let [a, b, c] = self.triangle_corners(triangle)?;
        let (e1, e2, offset) = (
            b.position - a.position,
            c.position - a.position,
            point - a.position,
        );
        let (d00, d01, d11) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
        let (d20, d21) = (offset.dot(e1), offset.dot(e2));
        let determinant = d00 * d11 - d01 * d01;
        if !determinant.is_finite() || determinant.abs() < 1e-20 {
            return None;
        }
        let v = (d11 * d20 - d01 * d21) / determinant;
        let w = (d00 * d21 - d01 * d20) / determinant;
        Some(a.uv * (1.0 - v - w) + b.uv * v + c.uv * w)
    }

    fn triangle_corners(&self, triangle: u32) -> Option<[MeshVertex; 3]> {
        let corners = self.triangles.get(triangle as usize)?;
        let [a, b, c] = corners.map(|corner| self.vertices.get(corner as usize).copied());
        Some([a?, b?, c?])
    }
}

/// Object-space surface travelled per unit of U and of V across a
/// triangle, or `None` when its UVs are degenerate.
fn uv_axes(a: MeshVertex, b: MeshVertex, c: MeshVertex) -> Option<(Vec3, Vec3)> {
    // Solve dP = du * U + dv * V from the two triangle edges.
    let (e1, e2) = (b.position - a.position, c.position - a.position);
    let (t1, t2) = (b.uv - a.uv, c.uv - a.uv);
    let determinant = t1.x * t2.y - t2.x * t1.y;
    if !determinant.is_finite() || determinant.abs() < 1e-12 {
        return None;
    }
    Some((
        (e1 * t2.y - e2 * t1.y) / determinant,
        (e2 * t1.x - e1 * t2.x) / determinant,
    ))
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

/// Control points of an editable spline path. `controls` are UV points; a
/// path placed on the model also keeps the object-space surface point
/// under each control in `surface`, and then follows the surface across
/// UV seams instead of running straight through UV space.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PathSpec {
    pub controls: Vec<Vec2>,
    #[serde(default)]
    pub closed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub surface: Vec<Vec3>,
}

/// Bisection steps that locate where a surface path crosses a UV seam.
const SEAM_BISECTIONS: usize = 12;

impl PathSpec {
    /// Catmull-Rom curve through every control point, `samples` points per
    /// segment.
    pub fn sample(&self, samples: usize) -> Vec<Vec2> {
        catmull_rom(&self.controls, self.closed, samples)
    }

    /// True when the path keeps a surface point for every control.
    pub fn on_surface(&self) -> bool {
        self.surface.len() >= 2 && self.surface.len() == self.controls.len()
    }

    /// Catmull-Rom curve through the surface points (object space), not
    /// yet snapped onto the mesh.
    pub fn sample_surface(&self, samples: usize) -> Vec<Vec3> {
        catmull_rom(&self.surface, self.closed, samples)
    }

    /// True when the path's numbers are all finite.
    pub fn is_valid(&self) -> bool {
        self.controls.iter().all(|point| point.is_finite())
            && self.surface.iter().all(|point| point.is_finite())
            && (self.surface.is_empty() || self.surface.len() == self.controls.len())
    }

    /// The path traced over `mesh`, as one UV polyline per stretch that
    /// stays in a single UV island. The surface curve is snapped onto the
    /// mesh and cut where it crosses a UV seam; each stretch runs up to the
    /// crossing (its UV carried over the edge of its own last triangle), so
    /// neighbouring stretches meet on the surface. A path without surface
    /// points gives its UV curve as the only stretch.
    pub fn surface_runs(
        &self,
        mesh: &MeshGeometry,
        topology: &MeshTopology,
        samples: usize,
    ) -> Vec<Vec<Vec2>> {
        if !self.on_surface() {
            return vec![self.sample(samples)];
        }
        let mut runs = Vec::new();
        let mut current = Vec::new();
        let mut previous: Option<(Vec3, SurfaceHit)> = None;
        for point in self.sample_surface(samples) {
            let Some(hit) = mesh.closest_surface_point(point) else {
                continue;
            };
            if let Some(from) = previous
                && crosses_uv_seam(topology, from.1.triangle, hit.triangle)
            {
                let (before, after) = seam_crossing(mesh, topology, from, (point, hit));
                current.push(before);
                runs.push(std::mem::take(&mut current));
                current.push(after);
            }
            current.push(hit.uv);
            previous = Some((point, hit));
        }
        runs.push(current);
        runs.retain(|run| !run.is_empty());
        runs
    }
}

/// Where the surface curve from `from` to `to` (curve point and its snap
/// onto the mesh) crosses from `from`'s UV island into another: the
/// crossing's UV on each side of the seam.
fn seam_crossing(
    mesh: &MeshGeometry,
    topology: &MeshTopology,
    (mut from_point, mut from): (Vec3, SurfaceHit),
    (mut to_point, mut to): (Vec3, SurfaceHit),
) -> (Vec2, Vec2) {
    let island = topology.island_of(from.triangle);
    for _ in 0..SEAM_BISECTIONS {
        let middle = from_point.lerp(to_point, 0.5);
        let Some(hit) = mesh.closest_surface_point(middle) else {
            break;
        };
        if topology.island_of(hit.triangle) == island {
            (from_point, from) = (middle, hit);
        } else {
            (to_point, to) = (middle, hit);
        }
    }
    // Snapping jumps over folds (the chord cuts a corner and lands on one
    // face or the other), so the crossing is where the two faces' planes
    // meet, nearest the jump; between near-parallel faces, its middle.
    let middle = from.position.lerp(to.position, 0.5);
    let crossing = planes_meeting(from.position, from.normal, to.position, to.normal)
        .map_or(middle, |(point, direction)| {
            point + direction * direction.dot(middle - point)
        });
    (
        mesh.uv_on_plane(from.triangle, crossing).unwrap_or(from.uv),
        mesh.uv_on_plane(to.triangle, crossing).unwrap_or(to.uv),
    )
}

/// The line where the plane through `a` facing `a_normal` meets the one
/// through `b` facing `b_normal`, as a point and unit direction. `None`
/// for near-parallel planes.
fn planes_meeting(a: Vec3, a_normal: Vec3, b: Vec3, b_normal: Vec3) -> Option<(Vec3, Vec3)> {
    let direction = a_normal.cross(b_normal);
    let length_squared = direction.length_squared();
    if !length_squared.is_finite() || length_squared < 1e-4 {
        return None;
    }
    let point = (b_normal.cross(direction) * a_normal.dot(a)
        + direction.cross(a_normal) * b_normal.dot(b))
        / length_squared;
    Some((point, direction / length_squared.sqrt()))
}

/// Catmull-Rom curve through `controls`, `samples` points per segment,
/// ending on the last control (or back on the first when `closed`).
fn catmull_rom<T>(controls: &[T], closed: bool, samples: usize) -> Vec<T>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<f32, Output = T>,
{
    let count = controls.len();
    if count < 2 {
        return controls.to_vec();
    }
    let samples = samples.clamp(1, 64);
    let at = |index: isize| -> T {
        if closed {
            controls[index.rem_euclid(count as isize) as usize]
        } else {
            controls[index.clamp(0, count as isize - 1) as usize]
        }
    };
    let segments = if closed { count } else { count - 1 };
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
                (p1 * 2.0
                    + (p2 - p0) * t
                    + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
                    + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3)
                    * 0.5,
            );
        }
    }
    points.push(if closed {
        controls[0]
    } else {
        controls[count - 1]
    });
    points
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::PrimitivePreset;

    #[test]
    fn tc_0301_brush_outlines_follow_the_uv_scale_of_the_surface() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let hit = cube
            .closest_surface_point(Vec3::new(0.1, 0.1, 0.9))
            .unwrap();
        // The cube's faces take a third of U and half of V each, minus
        // padding, so a round UV brush lands as an ellipse on a face.
        let outline = cube
            .uv_brush_outline(hit.triangle, hit.position, Vec2::splat(0.05), 16)
            .unwrap();
        assert_eq!(outline.len(), 16);
        let reach = |direction: Vec3| {
            outline
                .iter()
                .map(|point| (*point - hit.position).dot(direction).abs())
                .fold(0.0_f32, f32::max)
        };
        // Every point stays on the face's plane.
        assert!(outline.iter().all(|point| (point.z - 0.5).abs() < 1e-5));
        let (across, up) = (reach(Vec3::X), reach(Vec3::Y));
        assert!(across > 0.0 && up > 0.0);
        assert!(
            (across / up - 1.5).abs() < 0.1,
            "stretched by the UV layout: {across} {up}"
        );
        // Degenerate UVs give no outline.
        let mut flat = cube.clone();
        for vertex in &mut flat.vertices {
            vertex.uv = Vec2::ZERO;
        }
        assert!(
            flat.uv_brush_outline(0, Vec3::ZERO, Vec2::splat(0.1), 8)
                .is_none()
        );
    }

    #[test]
    fn tc_0352_stamp_frames_undo_the_uv_stretch_of_the_hit_triangle() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let hit = cube
            .closest_surface_point(Vec3::new(0.1, 0.1, 0.9))
            .unwrap();
        let frame = cube.stamp_frame(hit.triangle, 512, 512).unwrap();
        let [m00, m10, m01, m11] = frame.0;
        assert!((m00 * m11 - m01 * m10 - 1.0).abs() < 1e-4, "unit area");
        assert!(
            m10.abs() < 1e-4 && m01.abs() < 1e-4,
            "UV axes square to the face"
        );
        // A face spans a third of U and half of V, so a texel covers 1.5
        // times more surface across than down; the frame lengthens texel
        // offsets across by the square root of that and shortens them
        // down by as much.
        assert!((m00 / m11 - 1.5).abs() < 0.05, "{frame:?}");
        // Along +U is along +X on the front face, so stamps stay upright.
        assert!(m00 > 0.0 && m11 > 0.0);
        // A square texture of the same layout as a 2:3 one undoes the
        // aspect: texels then cover equal surface both ways.
        let wide = cube.stamp_frame(hit.triangle, 769, 513).unwrap();
        assert!((wide.0[0] - wide.0[3]).abs() < 0.05, "{wide:?}");

        // Mirrored UVs flip the frame so stamps read the right way round.
        let mut mirrored = cube.clone();
        for vertex in &mut mirrored.vertices {
            vertex.uv.x = 1.0 - vertex.uv.x;
        }
        let flipped = mirrored.stamp_frame(hit.triangle, 512, 512).unwrap();
        let [a, b, c, d] = flipped.0;
        assert!((a * d - b * c + 1.0).abs() < 1e-4, "{flipped:?}");

        // Degenerate UVs give no frame.
        let mut flat = cube.clone();
        for vertex in &mut flat.vertices {
            vertex.uv = Vec2::ZERO;
        }
        assert!(flat.stamp_frame(hit.triangle, 512, 512).is_none());
        assert!(cube.stamp_frame(9_999, 512, 512).is_none());
    }

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
            radial: 0,
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
    fn tc_0236_radial_symmetry_repeats_hits_around_the_axis() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let hit = cube
            .closest_surface_point(Vec3::new(0.3, 0.1, 0.9))
            .unwrap();
        let radial = Symmetry::radial(Axis::Y, 4);
        assert!(radial.is_radial());
        let copies = radial.symmetric_hits(&cube, &hit);
        assert_eq!(copies.len(), 3);
        let normals: Vec<Vec3> = copies.iter().map(|copy| copy.unwrap().normal).collect();
        assert_eq!(normals, [Vec3::X, Vec3::NEG_Z, Vec3::NEG_X]);
        let first = copies[0].unwrap().position;
        assert!(
            (first - Vec3::new(0.5, 0.1, -0.3)).length() < 1e-5,
            "{first}"
        );
        // Every copy keeps its distance from the axis and its height.
        for copy in copies.iter().flatten() {
            assert!((copy.position.y - 0.1).abs() < 1e-5);
            let radius = |p: Vec3| Vec2::new(p.x, p.z).length();
            assert!((radius(copy.position) - radius(hit.position)).abs() < 1e-4);
        }
        // Counts clamp; below two is plain mirroring.
        assert_eq!(Symmetry::radial(Axis::Z, 99).copies(Vec3::X).len(), 31);
        assert!(!Symmetry::radial(Axis::Z, 1).is_radial());
        assert_eq!(Symmetry::radial(Axis::X, 1).copies(Vec3::X), [Vec3::NEG_X]);
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
            surface: Vec::new(),
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

    #[test]
    fn tc_0369_surface_paths_follow_the_model_across_uv_seams() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let topology = MeshTopology::build(&cube);
        let on = |point: Vec3| cube.closest_surface_point(point).unwrap();
        let place = |points: &[Vec3]| PathSpec {
            controls: points.iter().map(|point| on(*point).uv).collect(),
            closed: false,
            surface: points.iter().map(|point| on(*point).position).collect(),
        };

        // Within one face the path is one stretch through its controls.
        let flat = place(&[Vec3::new(-0.3, 0.0, 0.5), Vec3::new(0.3, 0.1, 0.5)]);
        assert!(flat.on_surface() && flat.is_valid());
        let runs = flat.surface_runs(&cube, &topology, 16);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].len(), 17);
        assert!((runs[0][0] - flat.controls[0]).length() < 1e-5);
        assert!((runs[0][16] - flat.controls[1]).length() < 1e-5);

        // From the front face over the top edge onto the top face: two
        // stretches, each running up to the edge in its own island.
        let front = on(Vec3::new(0.0, 0.2, 0.5));
        let top = on(Vec3::new(0.0, 0.5, 0.2));
        let over = place(&[front.position, top.position]);
        let runs = over.surface_runs(&cube, &topology, 16);
        assert_eq!(runs.len(), 2);
        let edge = Vec3::new(0.0, 0.5, 0.5);
        let front_edge = cube.uv_on_plane(front.triangle, edge).unwrap();
        let top_edge = cube.uv_on_plane(top.triangle, edge).unwrap();
        assert!((*runs[0].last().unwrap() - front_edge).length() < 1e-3);
        assert!((runs[1][0] - top_edge).length() < 1e-3);
        assert!((front_edge - top_edge).length() > 0.05, "a real UV seam");
        // Each stretch stays in its own face's UV island.
        assert!((runs[0][0] - front.uv).length() < 1e-5);
        assert!((*runs[1].last().unwrap() - top.uv).length() < 1e-5);

        // Without surface points, or with mismatched ones, the path is the
        // plain UV curve.
        let uv_only = PathSpec {
            surface: Vec::new(),
            ..over.clone()
        };
        assert_eq!(uv_only.surface_runs(&cube, &topology, 16).len(), 1);
        let broken = PathSpec {
            surface: vec![Vec3::NAN; 2],
            ..over
        };
        assert!(!broken.is_valid());
    }
}
