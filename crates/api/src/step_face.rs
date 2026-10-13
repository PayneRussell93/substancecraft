//! Tessellation of one trimmed B-rep face: its boundary loops are mapped
//! into the surface's parameter plane (unwrapped across periodic seams, with
//! poles and apexes opened into segments), closed into one polygon with
//! holes, triangulated there, and refined until every triangle is within the
//! chord tolerance of the surface. Also builds UV charts for triangle soups
//! and packs charts into a UV atlas.

use glam::{DVec2, DVec3, Vec2};

use crate::step_geom::{Surface, distance_to_segment};
use crate::triangulate::{TriangleMesh, earcut, orient};

/// Most points one face may grow to while refining.
const MAX_FACE_POINTS: usize = 60_000;
/// Fewest segments along a closing pole or domain edge.
const DOMAIN_EDGE_SEGMENTS: usize = 8;

/// One boundary of a face in model space, in loop order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Loop3 {
    Points(Vec<DVec3>),
    /// A loop collapsed to a vertex (a cone apex or sphere pole).
    Vertex(DVec3),
}

/// A tessellated face: positions, chart coordinates (parameters scaled to
/// model lengths, for UV layout), and triangles wound by the face normal.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct FacePatch {
    pub positions: Vec<DVec3>,
    pub chart: Vec<DVec2>,
    pub triangles: Vec<[u32; 3]>,
}

impl FacePatch {
    pub(crate) fn flip(&mut self) {
        for triangle in &mut self.triangles {
            triangle.swap(1, 2);
        }
    }

    /// Six times the signed volume the patch encloses with the origin.
    pub(crate) fn signed_volume(&self) -> f64 {
        self.triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|corner| self.positions[corner as usize]);
                a.dot(b.cross(c))
            })
            .sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct UvPoint {
    uv: DVec2,
    position: DVec3,
}

fn ring_area(ring: &[UvPoint]) -> f64 {
    let mut area = 0.0;
    for (index, point) in ring.iter().enumerate() {
        let next = ring[(index + 1) % ring.len()];
        area += point.uv.perp_dot(next.uv);
    }
    area / 2.0
}

fn mean(ring: &[UvPoint], axis: usize) -> f64 {
    ring.iter().map(|point| point.uv[axis]).sum::<f64>() / ring.len().max(1) as f64
}

fn dedup(ring: &mut Vec<UvPoint>, closed: bool) {
    ring.dedup_by(|b, a| (a.uv - b.uv).length_squared() <= 1e-24);
    while closed
        && ring.len() > 1
        && ring
            .first()
            .zip(ring.last())
            .is_some_and(|(a, b)| (a.uv - b.uv).length_squared() <= 1e-24)
    {
        ring.pop();
    }
}

/// Parameter axes along which the surface collapses at `uv`, where the
/// parameter is meaningless.
fn singular_axes(surface: &Surface, uv: DVec2, tolerance: f64) -> [bool; 2] {
    let (_, du, dv) = surface.evaluate(uv.x, uv.y);
    let periods = surface.periods();
    let domain = surface.domain();
    let span = |axis: usize| {
        periods[axis].or_else(|| domain[axis].map(|(low, high)| (high - low).abs()))
    };
    [
        span(0).is_some_and(|span| du.length() * span <= tolerance),
        span(1).is_some_and(|span| dv.length() * span <= tolerance),
    ]
}

/// A boundary mapped into the parameter plane, unwrapped, with the number
/// of times it winds around each periodic direction.
struct MappedLoop {
    points: Vec<UvPoint>,
    winding: [i64; 2],
}

fn map_loop(surface: &Surface, points: &[DVec3], tolerance: f64) -> MappedLoop {
    let periods = surface.periods();
    let count = points.len();
    let mut uv = Vec::with_capacity(count);
    let mut singular = Vec::with_capacity(count);
    let mut seed = None;
    for point in points {
        let parameters = surface.project(*point, seed, tolerance);
        seed = Some(parameters);
        uv.push(parameters);
        singular.push(singular_axes(surface, parameters, tolerance));
    }
    let mut winding = [0_i64; 2];
    for axis in 0..2 {
        let Some(period) = periods[axis] else {
            continue;
        };
        let regular: Vec<usize> = (0..count).filter(|index| !singular[*index][axis]).collect();
        let (Some(&first), Some(&last)) = (regular.first(), regular.last()) else {
            continue;
        };
        let mut previous = uv[first][axis];
        for &index in &regular[1..] {
            let value = &mut uv[index][axis];
            *value += ((previous - *value) / period).round() * period;
            previous = *value;
        }
        let closing = uv[first][axis] - uv[last][axis];
        let closing = closing - (closing / period).round() * period;
        let turns = ((uv[last][axis] + closing - uv[first][axis]) / period).round() as i64;
        winding[axis] = turns;
        // Across a pole the parameter jumps freely; let that jump undo the
        // winding, so a loop through a pole stays closed.
        if turns != 0
            && let Some(pole) = (first..count).find(|index| singular[*index][axis])
        {
            for index in pole..count {
                if !singular[index][axis] {
                    uv[index][axis] -= turns as f64 * period;
                }
            }
            winding[axis] = 0;
        }
    }

    // Open each pole into a segment between its neighbours' parameters.
    let mut mapped = Vec::with_capacity(count + 4);
    for index in 0..count {
        let flags = singular[index];
        let axis = match flags {
            [true, false] => 0,
            [false, true] => 1,
            _ => {
                mapped.push(UvPoint {
                    uv: uv[index],
                    position: points[index],
                });
                continue;
            }
        };
        let neighbour = |step: isize| {
            (1..count).find_map(|offset| {
                let other =
                    (index as isize + step * offset as isize).rem_euclid(count as isize) as usize;
                (!singular[other][axis]).then_some(uv[other][axis])
            })
        };
        let (Some(before), Some(after)) = (neighbour(-1), neighbour(1)) else {
            mapped.push(UvPoint {
                uv: uv[index],
                position: points[index],
            });
            continue;
        };
        let mut start = uv[index];
        start[axis] = before;
        let mut end = uv[index];
        end[axis] = after;
        mapped.push(UvPoint {
            uv: start,
            position: points[index],
        });
        if (after - before).abs() > 1e-12 {
            mapped.push(UvPoint {
                uv: end,
                position: points[index],
            });
        }
    }
    dedup(&mut mapped, true);
    MappedLoop {
        points: mapped,
        winding,
    }
}

/// Re-cuts a loop that winds `winding` times around `axis` (period
/// `period`) so it starts at its point nearest parameter `target` (shifted
/// by whole periods to lie there) and ends at that point one period on.
fn cut_at(
    ring: &[UvPoint],
    axis: usize,
    winding: i64,
    period: f64,
    target: f64,
) -> Option<Vec<UvPoint>> {
    let count = ring.len();
    let shift = winding as f64 * period;
    let extended = |index: usize| {
        let mut point = ring[index % count];
        point.uv[axis] += (index / count) as f64 * shift;
        point
    };
    for index in 0..count {
        let (a, b) = (extended(index), extended(index + 1));
        let (low, high) = (a.uv[axis].min(b.uv[axis]), a.uv[axis].max(b.uv[axis]));
        let level = target + ((low - target) / period).ceil() * period;
        if level > high || (high - low) <= 0.0 && level != a.uv[axis] {
            continue;
        }
        // Cut at the nearer existing point rather than a new one, so the
        // neighbouring face sharing this boundary sees no T-junction.
        let nearer_a = (level - a.uv[axis]).abs() <= (b.uv[axis] - level).abs();
        let first = if nearer_a { index } else { index + 1 };
        let mut cut: Vec<UvPoint> = (first..=first + count).map(extended).collect();
        let offset = target - level;
        for point in &mut cut {
            point.uv[axis] += offset;
        }
        dedup(&mut cut, false);
        return Some(cut);
    }
    None
}

/// Points from `a` to `b` along their straight parameter line, split until
/// each chord is within `tolerance` of the surface; at least `minimum`
/// segments.
fn sample_line(
    surface: &Surface,
    a: UvPoint,
    b: UvPoint,
    tolerance: f64,
    minimum: usize,
) -> Vec<UvPoint> {
    fn split(
        surface: &Surface,
        (a, b): (DVec2, DVec2),
        (t0, t1): (f64, f64),
        tolerance: f64,
        depth: usize,
        params: &mut Vec<f64>,
    ) {
        let at = |t: f64| surface.point(a.lerp(b, t));
        let middle = (t0 + t1) / 2.0;
        if depth < 10 && distance_to_segment(at(middle), at(t0), at(t1)) > tolerance {
            split(surface, (a, b), (t0, middle), tolerance, depth + 1, params);
            split(surface, (a, b), (middle, t1), tolerance, depth + 1, params);
        } else {
            params.push(t1);
        }
    }
    let segments = minimum.max(1);
    let mut params = vec![0.0];
    for step in 0..segments {
        let range = (
            step as f64 / segments as f64,
            (step + 1) as f64 / segments as f64,
        );
        split(surface, (a.uv, b.uv), range, tolerance, 0, &mut params);
    }
    let last = params.len() - 1;
    params
        .iter()
        .enumerate()
        .map(|(index, t)| match index {
            0 => a,
            index if index == last => b,
            _ => {
                let uv = a.uv.lerp(b.uv, *t);
                UvPoint {
                    uv,
                    position: surface.point(uv),
                }
            }
        })
        .collect()
}

/// Joins polylines into one ring, bridging each end to the next start along
/// straight parameter lines sampled to `tolerance`.
fn close_ring(
    surface: &Surface,
    parts: Vec<Vec<UvPoint>>,
    tolerance: f64,
    minimum: usize,
) -> Vec<UvPoint> {
    let mut ring = Vec::new();
    for index in 0..parts.len() {
        let part = &parts[index];
        let next = &parts[(index + 1) % parts.len()];
        ring.extend_from_slice(part);
        if let (Some(end), Some(start)) = (part.last(), next.first()) {
            let bridge = sample_line(surface, *end, *start, tolerance, minimum);
            ring.extend_from_slice(&bridge[1..bridge.len() - 1]);
        }
    }
    ring
}

/// A seam position along `axis` (period `period`) that no hole crosses.
fn seam_avoiding(holes: &[Vec<UvPoint>], axis: usize, period: f64, start: f64) -> f64 {
    let crosses = |seam: f64| {
        holes.iter().any(|hole| {
            let low = hole.iter().map(|p| p.uv[axis]).fold(f64::INFINITY, f64::min);
            let high = hole
                .iter()
                .map(|p| p.uv[axis])
                .fold(f64::NEG_INFINITY, f64::max);
            let shifted = low - ((low - seam) / period).floor() * period;
            shifted + (high - low) > seam + period
        })
    };
    (0..64)
        .map(|step| start + period * step as f64 / 64.0)
        .find(|seam| !crosses(*seam))
        .unwrap_or(start)
}

/// Shifts each hole by whole periods so it sits next to `centre`.
fn place_holes(holes: &mut [Vec<UvPoint>], periods: [Option<f64>; 2], centre: DVec2) {
    for hole in holes {
        for axis in 0..2 {
            if let Some(period) = periods[axis] {
                let shift = ((centre[axis] - mean(hole, axis)) / period).round() * period;
                for point in hole.iter_mut() {
                    point.uv[axis] += shift;
                }
            }
        }
    }
}

fn rectangle(surface: &Surface, low: DVec2, high: DVec2, tolerance: f64) -> Vec<UvPoint> {
    let corners = [
        low,
        DVec2::new(high.x, low.y),
        high,
        DVec2::new(low.x, high.y),
    ];
    let parts = corners
        .iter()
        .map(|uv| {
            vec![UvPoint {
                uv: *uv,
                position: surface.point(*uv),
            }]
        })
        .collect();
    close_ring(surface, parts, tolerance, DOMAIN_EDGE_SEGMENTS)
}

/// Closes the mapped loops into rings: the outer boundary first, then
/// holes.
fn assemble(
    surface: &Surface,
    sense: bool,
    closed: Vec<Vec<UvPoint>>,
    wrapping: Vec<(Vec<UvPoint>, usize, i64)>,
    vertices: &[DVec2],
    tolerance: f64,
) -> Result<Vec<Vec<UvPoint>>, &'static str> {
    let periods = surface.periods();
    let mut holes = closed;
    if wrapping.is_empty() {
        if holes.is_empty() {
            // Only vertex loops, or none: the whole closed surface.
            let domain = surface.domain();
            let (Some(u), Some(v)) = (domain[0], domain[1]) else {
                return Err("face has no boundary");
            };
            let outer = rectangle(surface, DVec2::new(u.0, v.0), DVec2::new(u.1, v.1), tolerance);
            return Ok(vec![outer]);
        }
        let outer_index = (0..holes.len())
            .max_by(|a, b| ring_area(&holes[*a]).abs().total_cmp(&ring_area(&holes[*b]).abs()))
            .unwrap_or(0);
        // On a closed surface (sphere, torus) a loop bounds either side; the
        // face lies to its left, so a clockwise largest loop is a hole in
        // the whole surface.
        let domain = surface.domain();
        if let ([Some(u), Some(v)], Some(period)) = (domain, periods[0]) {
            let oriented = ring_area(&holes[outer_index]) * if sense { 1.0 } else { -1.0 };
            if oriented < 0.0 {
                let seam = seam_avoiding(&holes, 0, period, u.0);
                let low = DVec2::new(seam, v.0);
                let high = DVec2::new(seam + period, v.1);
                place_holes(&mut holes, periods, (low + high) / 2.0);
                let mut rings = vec![rectangle(surface, low, high, tolerance)];
                rings.extend(holes);
                return Ok(rings);
            }
        }
        let outer = holes.swap_remove(outer_index);
        let low = outer
            .iter()
            .fold(DVec2::INFINITY, |low, point| low.min(point.uv));
        let high = outer
            .iter()
            .fold(DVec2::NEG_INFINITY, |high, point| high.max(point.uv));
        let extent = high - low;
        // A loop with no area that only runs up and back down a seam bounds
        // the whole period (a sphere cut along one meridian).
        let degenerate = ring_area(&outer).abs() <= 1e-9 * extent.max_element().powi(2).max(1e-300);
        let outer = match (degenerate, periods) {
            (true, [Some(period), _]) => {
                rectangle(surface, low, DVec2::new(low.x + period, high.y), tolerance)
            }
            (true, [None, Some(period)]) => {
                rectangle(surface, low, DVec2::new(high.x, low.y + period), tolerance)
            }
            (true, _) => return Err("face boundary encloses no area"),
            (false, _) => outer,
        };
        let centre = outer.iter().map(|p| p.uv).sum::<DVec2>() / outer.len() as f64;
        place_holes(&mut holes, periods, centre);
        let mut rings = vec![outer];
        rings.extend(holes);
        return Ok(rings);
    }

    let axis = wrapping[0].1;
    if wrapping.iter().any(|(_, other, _)| *other != axis) || wrapping.len() > 2 {
        return Err("face boundaries wind around the surface in several directions");
    }
    let Some(period) = periods[axis] else {
        return Err("face boundary winds around a non-periodic direction");
    };
    let across = 1 - axis;
    let start = wrapping[0].0.first().map_or(0.0, |p| p.uv[axis]);
    let seam = seam_avoiding(&holes, axis, period, start);
    let cut = |(ring, _, winding): &(Vec<UvPoint>, usize, i64)| {
        // Every cut runs from the seam one period forward.
        let mut cut = cut_at(ring, axis, *winding, period, seam)?;
        if *winding < 0 {
            cut.reverse();
            for point in &mut cut {
                point.uv[axis] += period;
            }
        }
        Some(cut)
    };
    let outer = if wrapping.len() == 2 {
        let mut bands = wrapping;
        bands.sort_by(|a, b| mean(&a.0, across).total_cmp(&mean(&b.0, across)));
        let (Some(low), Some(mut high)) = (cut(&bands[0]), cut(&bands[1])) else {
            return Err("could not cut a face boundary at its seam");
        };
        high.reverse();
        close_ring(surface, vec![low, high], tolerance, 1)
    } else {
        let ring = &wrapping[0];
        let cap = cut(ring).ok_or("could not cut a face boundary at its seam")?;
        let level_now = mean(&ring.0, across);
        let (high, level) = match vertices.first() {
            Some(vertex) => (vertex[across] > level_now, Some(vertex[across])),
            None => {
                // The face lies to the left of its loops (seen along the
                // face normal), so the loop direction picks the cap side.
                let forward = (ring.2 > 0) == sense;
                let high = forward != (axis == 1);
                let level = if axis == 0 {
                    surface.degenerate_v(high, tolerance)
                } else {
                    None
                };
                (high, level)
            }
        };
        let level = level.ok_or("face boundary winds once around a surface with no pole")?;
        if (level > level_now) != high {
            return Err("face cap lies on the wrong side of its boundary");
        }
        let corner = |along: f64| {
            let mut uv = DVec2::ZERO;
            uv[axis] = along;
            uv[across] = level;
            UvPoint {
                uv,
                position: surface.point(uv),
            }
        };
        let edge = sample_line(
            surface,
            corner(cap.last().map_or(seam + period, |point| point.uv[axis])),
            corner(cap.first().map_or(seam, |point| point.uv[axis])),
            tolerance,
            DOMAIN_EDGE_SEGMENTS,
        );
        close_ring(surface, vec![cap, edge], tolerance, 1)
    };
    let centre = DVec2::from_array({
        let mut centre = [0.0; 2];
        centre[axis] = seam + period / 2.0;
        centre[across] = mean(&outer, across);
        centre
    });
    place_holes(&mut holes, periods, centre);
    let mut rings = vec![outer];
    rings.extend(holes);
    Ok(rings)
}

/// Tessellates a face on `surface` bounded by `loops`. `sense` is true when
/// the face normal agrees with the surface normal (`S_u x S_v`).
pub(crate) fn tessellate_face(
    surface: &Surface,
    sense: bool,
    loops: &[Loop3],
    tolerance: f64,
) -> Result<FacePatch, &'static str> {
    let mut closed = Vec::new();
    let mut wrapping = Vec::new();
    let mut vertices = Vec::new();
    for boundary in loops {
        let points = match boundary {
            Loop3::Vertex(point) => {
                vertices.push(surface.project(*point, None, tolerance));
                continue;
            }
            Loop3::Points(points) if points.len() < 3 => {
                if let Some(point) = points.first() {
                    vertices.push(surface.project(*point, None, tolerance));
                }
                continue;
            }
            Loop3::Points(points) => points,
        };
        let mapped = map_loop(surface, points, tolerance);
        if mapped.points.len() < 2 {
            continue;
        }
        match mapped.winding {
            [0, 0] if mapped.points.len() >= 3 => closed.push(mapped.points),
            [0, 0] => {}
            [turns, 0] => wrapping.push((mapped.points, 0, turns.signum())),
            [0, turns] => wrapping.push((mapped.points, 1, turns.signum())),
            _ => return Err("face boundary winds around the surface in both directions"),
        }
    }
    if std::env::var_os("STEP_DEBUG_SENSE").is_some() && wrapping.is_empty() && !closed.is_empty() {
        let largest = closed
            .iter()
            .map(|ring| ring_area(ring))
            .max_by(|a, b| a.abs().total_cmp(&b.abs()))
            .unwrap_or(0.0);
        let kind = format!("{surface:?}").split(['(', ' ', '{']).next().unwrap_or("").to_owned();
        eprintln!(
            "SENSE {kind} {}",
            if (largest > 0.0) == sense { "agree" } else { "disagree" }
        );
    }
    let mut rings = assemble(surface, sense, closed, wrapping, &vertices, tolerance)?;
    for ring in &mut rings {
        dedup(ring, true);
    }
    rings.retain(|ring| ring.len() >= 3);
    if rings.is_empty() {
        return Err("face boundary has fewer than three points");
    }

    // Triangulate in parameters scaled to model lengths, so triangles are
    // well shaped on the surface.
    let mut scale = DVec2::ZERO;
    let mut samples = 0.0;
    for point in rings.iter().flatten() {
        let (_, du, dv) = surface.evaluate(point.uv.x, point.uv.y);
        if du.length() > 0.0 && dv.length() > 0.0 {
            scale += DVec2::new(du.length(), dv.length());
            samples += 1.0;
        }
    }
    let scale = if samples > 0.0 && (scale / samples).min_element() > 1e-300 {
        scale / samples
    } else {
        DVec2::ONE
    };
    let mut starts = Vec::with_capacity(rings.len());
    let mut parameters = Vec::new();
    let mut positions = Vec::new();
    for ring in &rings {
        starts.push(parameters.len());
        for point in ring {
            parameters.push(point.uv);
            positions.push(point.position);
        }
    }
    let scaled: Vec<DVec2> = parameters.iter().map(|uv| *uv * scale).collect();
    if std::env::var_os("STEP_DEBUG_UV").is_some() {
        eprintln!("scale {scale} rings {}", rings.len());
        for (i, p) in parameters.iter().enumerate() {
            eprintln!("  {i} uv {:.5} {:.5}  pos {:.4}", p.x, p.y, positions[i]);
        }
    }
    let triangles = earcut(&scaled, &starts);
    if triangles.is_empty() {
        return Err("face boundary could not be triangulated");
    }
    let mut mesh = TriangleMesh::new(scaled, triangles);
    let ranges: Vec<std::ops::Range<usize>> = (0..starts.len())
        .map(|index| starts[index]..starts.get(index + 1).copied().unwrap_or(parameters.len()))
        .collect();
    mesh.restore_boundary_points(&ranges);
    mesh.make_delaunay();
    refine(&mut mesh, surface, scale, &mut parameters, &mut positions, tolerance);

    let mut patch = FacePatch {
        positions: Vec::new(),
        chart: Vec::new(),
        triangles: Vec::new(),
    };
    let mut remap = vec![u32::MAX; positions.len()];
    for triangle in &mesh.triangles {
        let corners = triangle.map(|corner| positions[corner as usize]);
        if corners[0] == corners[1] || corners[1] == corners[2] || corners[0] == corners[2] {
            continue;
        }
        let mut indices = triangle.map(|corner| {
            let slot = &mut remap[corner as usize];
            if *slot == u32::MAX {
                *slot = patch.positions.len() as u32;
                patch.positions.push(positions[corner as usize]);
                patch.chart.push(mesh.points[corner as usize]);
            }
            *slot
        });
        if !sense {
            indices.swap(1, 2);
        }
        patch.triangles.push(indices);
    }
    if patch.triangles.is_empty() {
        return Err("face tessellates to no triangles");
    }
    if !sense {
        // Mirror the chart too, so it keeps the triangles' winding.
        for point in &mut patch.chart {
            point.x = -point.x;
        }
    }
    Ok(patch)
}

/// Inserts triangle centroids until each triangle's centroid and interior
/// edge midpoints lie within `tolerance` of the surface.
fn refine(
    mesh: &mut TriangleMesh,
    surface: &Surface,
    scale: DVec2,
    parameters: &mut Vec<DVec2>,
    positions: &mut Vec<DVec3>,
    tolerance: f64,
) {
    let mut queue: Vec<u32> = (0..mesh.triangles.len() as u32).collect();
    while let Some(triangle) = queue.pop() {
        if mesh.points.len() >= MAX_FACE_POINTS {
            return;
        }
        let corners = mesh.triangles[triangle as usize];
        let [pa, pb, pc] = corners.map(|corner| positions[corner as usize]);
        let longest = pa.distance(pb).max(pb.distance(pc)).max(pc.distance(pa));
        if longest <= tolerance {
            continue;
        }
        let [qa, qb, qc] = corners.map(|corner| mesh.points[corner as usize]);
        if orient(qa, qb, qc) <= 1e-18 * longest * longest {
            continue;
        }
        let centroid = (qa + qb + qc) / 3.0;
        let centroid_uv = centroid / scale;
        let on_surface = surface.point(centroid_uv);
        let mut deviation = on_surface.distance((pa + pb + pc) / 3.0);
        if deviation <= tolerance {
            for k in 0..3 {
                if !mesh.is_interior(triangle, k) {
                    continue;
                }
                let (a, b) = (corners[k] as usize, corners[(k + 1) % 3] as usize);
                let middle = surface.point((parameters[a] + parameters[b]) / 2.0);
                deviation = deviation.max(middle.distance((positions[a] + positions[b]) / 2.0));
            }
        }
        if deviation <= tolerance {
            continue;
        }
        let touched = mesh.insert_in_triangle(triangle, centroid);
        parameters.push(centroid_uv);
        positions.push(on_surface);
        queue.extend(touched);
    }
}

/// A planar polygon's normal (Newell's method).
pub(crate) fn newell_normal(points: &[DVec3]) -> DVec3 {
    let mut normal = DVec3::ZERO;
    for (index, point) in points.iter().enumerate() {
        normal += point.cross(points[(index + 1) % points.len()]);
    }
    normal
}

/// Splits a triangle soup into charts: connected triangles facing the same
/// principal axis, each projected along it.
pub(crate) fn soup_patches(positions: &[DVec3], triangles: &[[u32; 3]]) -> Vec<FacePatch> {
    let classes: Vec<usize> = triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|corner| positions[corner as usize]);
            let normal = (b - a).cross(c - a);
            let axis = if normal.x.abs() >= normal.y.abs() && normal.x.abs() >= normal.z.abs() {
                0
            } else if normal.y.abs() >= normal.z.abs() {
                1
            } else {
                2
            };
            axis * 2 + usize::from(normal[axis] < 0.0)
        })
        .collect();
    let mut parent: Vec<usize> = (0..triangles.len()).collect();
    fn find(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    let mut edges: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
    for (index, triangle) in triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (triangle[k], triangle[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            match edges.get(&key) {
                Some(&other) if classes[other] == classes[index] => {
                    let (x, y) = (find(&mut parent, other), find(&mut parent, index));
                    parent[x] = y;
                }
                Some(_) => {}
                None => {
                    edges.insert(key, index);
                }
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for index in 0..triangles.len() {
        let root = find(&mut parent, index);
        groups.entry(root).or_default().push(index);
    }
    groups
        .into_values()
        .map(|members| {
            let class = classes[members[0]];
            let project = |p: DVec3| match class {
                0 => DVec2::new(p.y, p.z),
                1 => DVec2::new(p.z, p.y),
                2 => DVec2::new(p.z, p.x),
                3 => DVec2::new(p.x, p.z),
                4 => DVec2::new(p.x, p.y),
                _ => DVec2::new(p.y, p.x),
            };
            let mut patch = FacePatch::default();
            let mut remap = std::collections::HashMap::new();
            for index in members {
                let corners = triangles[index].map(|corner| {
                    *remap.entry(corner).or_insert_with(|| {
                        let position = positions[corner as usize];
                        patch.positions.push(position);
                        patch.chart.push(project(position));
                        patch.positions.len() as u32 - 1
                    })
                });
                patch.triangles.push(corners);
            }
            patch
        })
        .collect()
}

/// Packs charts (in model lengths, so texel density is uniform) into the
/// unit square with shelf packing, rotating tall charts. Returns each
/// chart's UVs.
pub(crate) fn pack_charts(charts: &[&[DVec2]]) -> Vec<Vec<Vec2>> {
    struct Item {
        index: usize,
        min: DVec2,
        size: DVec2,
        rotated: bool,
    }
    let mut items: Vec<Item> = charts
        .iter()
        .enumerate()
        .map(|(index, chart)| {
            let min = chart.iter().fold(DVec2::INFINITY, |a, b| a.min(*b));
            let max = chart.iter().fold(DVec2::NEG_INFINITY, |a, b| a.max(*b));
            let size = if chart.is_empty() { DVec2::ZERO } else { max - min };
            let rotated = size.y > size.x;
            Item {
                index,
                min: if chart.is_empty() { DVec2::ZERO } else { min },
                size: if rotated { DVec2::new(size.y, size.x) } else { size },
                rotated,
            }
        })
        .collect();
    let area: f64 = items.iter().map(|item| item.size.x * item.size.y).sum();
    let largest = items
        .iter()
        .map(|item| item.size.max_element())
        .fold(0.0, f64::max);
    let padding = (area.sqrt() * 0.004).max(largest * 1e-4).max(1e-9);
    let padded: f64 = items
        .iter()
        .map(|item| (item.size.x + padding) * (item.size.y + padding))
        .sum();
    let widest = items.iter().map(|item| item.size.x).fold(0.0, f64::max);
    let width = (padded.sqrt() * 1.05).max(widest + 2.0 * padding);
    items.sort_by(|a, b| b.size.y.total_cmp(&a.size.y).then(a.index.cmp(&b.index)));
    let mut offsets = vec![DVec2::ZERO; charts.len()];
    let (mut x, mut y, mut row) = (padding, padding, 0.0_f64);
    for item in &items {
        if x + item.size.x + padding > width && x > padding {
            y += row + padding;
            x = padding;
            row = 0.0;
        }
        offsets[item.index] = DVec2::new(x, y);
        x += item.size.x + padding;
        row = row.max(item.size.y);
    }
    let extent = width.max(y + row + padding);
    let mut uvs = vec![Vec::new(); charts.len()];
    for item in &items {
        let offset = offsets[item.index];
        uvs[item.index] = charts[item.index]
            .iter()
            .map(|point| {
                let local = *point - item.min;
                let local = if item.rotated {
                    DVec2::new(local.y, item.size.y - local.x)
                } else {
                    local
                };
                ((offset + local) / extent).as_vec2()
            })
            .collect();
    }
    uvs
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::step_geom::Frame;
    use std::f64::consts::{PI, TAU};

    fn circle(radius: f64, z: f64, count: usize, reverse: bool) -> Vec<DVec3> {
        let mut points: Vec<DVec3> = (0..count)
            .map(|index| {
                let angle = index as f64 / count as f64 * TAU;
                DVec3::new(radius * angle.cos(), radius * angle.sin(), z)
            })
            .collect();
        if reverse {
            points.reverse();
        }
        points
    }

    fn area(patch: &FacePatch) -> f64 {
        patch
            .triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|corner| patch.positions[corner as usize]);
                (b - a).cross(c - a).length() / 2.0
            })
            .sum()
    }

    #[test]
    fn cylinder_band_between_two_circles_has_the_lateral_area() {
        let surface = Surface::Cylinder {
            frame: Frame::IDENTITY,
            radius: 1.0,
        };
        let loops = [
            Loop3::Points(circle(1.0, 0.0, 64, false)),
            Loop3::Points(circle(1.0, 2.0, 64, true)),
        ];
        let patch = tessellate_face(&surface, true, &loops, 1e-3).unwrap();
        assert!((area(&patch) - TAU * 2.0).abs() < 0.02, "{}", area(&patch));
        // Outward winding: normals point away from the axis.
        for t in &patch.triangles {
            let [a, b, c] = t.map(|corner| patch.positions[corner as usize]);
            let normal = (b - a).cross(c - a);
            let centre = (a + b + c) / 3.0;
            assert!(normal.dot(DVec3::new(centre.x, centre.y, 0.0)) > 0.0);
        }
    }

    #[test]
    fn sphere_caps_close_at_the_pole_on_the_face_side() {
        let surface = Surface::Sphere {
            frame: Frame::IDENTITY,
            radius: 1.0,
        };
        // The equator, counter-clockwise seen from above: the north cap.
        let loops = [Loop3::Points(circle(1.0, 0.0, 96, false))];
        let patch = tessellate_face(&surface, true, &loops, 1e-3).unwrap();
        assert!((area(&patch) - TAU).abs() < 0.03, "{}", area(&patch));
        assert!(patch.positions.iter().all(|p| p.z >= -1e-9));
        assert!(patch.positions.iter().all(|p| (p.length() - 1.0).abs() < 1e-9));
        // Reversed sense: the south cap.
        let south = tessellate_face(&surface, false, &loops, 1e-3).unwrap();
        assert!(south.positions.iter().all(|p| p.z <= 1e-9));
    }

    #[test]
    fn whole_sphere_from_a_vertex_loop_and_a_cone_with_its_apex() {
        let sphere = Surface::Sphere {
            frame: Frame::IDENTITY,
            radius: 2.0,
        };
        let patch =
            tessellate_face(&sphere, true, &[Loop3::Vertex(DVec3::new(0.0, 0.0, 2.0))], 1e-3)
                .unwrap();
        assert!((area(&patch) - 4.0 * PI * 4.0).abs() < 0.2, "{}", area(&patch));
        assert!((patch.signed_volume() / 6.0 - 4.0 / 3.0 * PI * 8.0).abs() < 0.2);

        // Cone of base radius 1 at z = 0 narrowing to an apex at z = 1.
        let cone = Surface::Cone {
            frame: Frame::IDENTITY,
            radius: 1.0,
            tan: -1.0,
        };
        let loops = [
            Loop3::Points(circle(1.0, 0.0, 64, false)),
            Loop3::Vertex(DVec3::new(0.0, 0.0, 1.0)),
        ];
        let patch = tessellate_face(&cone, true, &loops, 1e-3).unwrap();
        let lateral = PI * 2.0_f64.sqrt();
        assert!((area(&patch) - lateral).abs() < 0.02, "{}", area(&patch));
    }

    #[test]
    fn planar_faces_keep_holes_and_seam_loops_cover_the_period() {
        let plane = Surface::Plane(Frame::IDENTITY);
        let square = vec![
            DVec3::new(-2.0, -2.0, 0.0),
            DVec3::new(2.0, -2.0, 0.0),
            DVec3::new(2.0, 2.0, 0.0),
            DVec3::new(-2.0, 2.0, 0.0),
        ];
        let loops = [
            Loop3::Points(square),
            Loop3::Points(circle(1.0, 0.0, 64, true)),
        ];
        let patch = tessellate_face(&plane, true, &loops, 1e-3).unwrap();
        assert!((area(&patch) - (16.0 - PI)).abs() < 0.01);

        // A cylinder face bounded by a seam loop: bottom, up, top, down.
        let cylinder = Surface::Cylinder {
            frame: Frame::IDENTITY,
            radius: 1.0,
        };
        let mut seam_loop = circle(1.0, 0.0, 64, false);
        seam_loop.push(DVec3::new(1.0, 0.0, 0.0));
        seam_loop.push(DVec3::new(1.0, 0.0, 1.0));
        seam_loop.extend(circle(1.0, 1.0, 64, true));
        seam_loop.push(DVec3::new(1.0, 0.0, 1.0));
        let patch = tessellate_face(&cylinder, true, &[Loop3::Points(seam_loop)], 1e-3).unwrap();
        assert!((area(&patch) - TAU).abs() < 0.01, "{}", area(&patch));
    }

    #[test]
    fn soups_split_by_facing_and_charts_pack_inside_the_unit_square() {
        let positions = vec![
            DVec3::ZERO,
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
        ];
        let triangles = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let patches = soup_patches(&positions, &triangles);
        assert_eq!(patches.iter().map(|p| p.triangles.len()).sum::<usize>(), 4);
        let charts: Vec<&[DVec2]> = patches.iter().map(|p| p.chart.as_slice()).collect();
        let uvs = pack_charts(&charts);
        assert!(uvs.iter().flatten().all(|uv| uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0));
    }
}
