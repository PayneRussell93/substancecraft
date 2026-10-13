//! Curves and surfaces of STEP boundary representations (ISO 10303-42):
//! evaluation with first derivatives, closest-point parameters, and
//! tolerance-driven curve sampling. Parametrisations follow the standard, so
//! a surface's normal is `S_u x S_v` as STEP defines it.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use glam::{DVec2, DVec3, DVec4};

/// Highest B-spline degree accepted.
pub(crate) const MAX_DEGREE: usize = 15;

/// A right-handed orthonormal frame (an `axis2_placement_3d`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Frame {
    pub origin: DVec3,
    pub x: DVec3,
    pub y: DVec3,
    pub z: DVec3,
}

impl Frame {
    pub(crate) const IDENTITY: Self = Self {
        origin: DVec3::ZERO,
        x: DVec3::X,
        y: DVec3::Y,
        z: DVec3::Z,
    };

    /// A frame from an axis and an optional reference direction, as
    /// `axis2_placement_3d` builds its axes.
    pub(crate) fn new(origin: DVec3, axis: Option<DVec3>, reference: Option<DVec3>) -> Self {
        let z = axis
            .and_then(|axis| axis.try_normalize())
            .unwrap_or(DVec3::Z);
        let fallback = if z.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let project = |direction: DVec3| (direction - z * direction.dot(z)).try_normalize();
        let x = reference
            .and_then(project)
            .or_else(|| project(fallback))
            .unwrap_or(DVec3::X);
        Self {
            origin,
            x,
            y: z.cross(x),
            z,
        }
    }

    pub(crate) fn to_local(self, point: DVec3) -> DVec3 {
        let offset = point - self.origin;
        DVec3::new(offset.dot(self.x), offset.dot(self.y), offset.dot(self.z))
    }

    pub(crate) fn affine(&self) -> glam::DAffine3 {
        glam::DAffine3::from_cols(self.x, self.y, self.z, self.origin)
    }
}

/// Knot vector and degree of one B-spline direction.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Knots {
    pub degree: usize,
    /// Expanded (every multiplicity written out); `count + degree + 1` long.
    pub values: Vec<f64>,
    /// Number of control points.
    pub count: usize,
}

impl Knots {
    pub(crate) fn new(degree: usize, values: Vec<f64>, count: usize) -> Option<Self> {
        if degree == 0
            || degree > MAX_DEGREE
            || count <= degree
            || values.len() != count + degree + 1
            || values.iter().any(|value| !value.is_finite())
            || values.windows(2).any(|pair| pair[1] < pair[0])
            || values[degree] >= values[count]
        {
            return None;
        }
        Some(Self {
            degree,
            values,
            count,
        })
    }

    /// Clamped uniform knots (`quasi_uniform` B-splines and Béziers).
    pub(crate) fn quasi_uniform(degree: usize, count: usize) -> Option<Self> {
        if count <= degree {
            return None;
        }
        let interior = count - degree;
        let mut values = vec![0.0; degree + 1];
        values.extend((1..interior).map(|knot| knot as f64));
        values.extend(std::iter::repeat_n(interior as f64, degree + 1));
        Self::new(degree, values, count)
    }

    /// Piecewise Bézier knots: `degree`-fold interior breaks every `degree`
    /// control points; other counts fall back to quasi-uniform knots.
    pub(crate) fn bezier(degree: usize, count: usize) -> Option<Self> {
        if degree == 0 || count <= degree || !(count - 1).is_multiple_of(degree) {
            return Self::quasi_uniform(degree, count);
        }
        let segments = (count - 1) / degree;
        let mut values = vec![0.0; degree + 1];
        for segment in 1..segments {
            values.extend(std::iter::repeat_n(segment as f64, degree));
        }
        values.extend(std::iter::repeat_n(segments as f64, degree + 1));
        Self::new(degree, values, count)
    }

    /// Unclamped uniform knots (`uniform` B-splines).
    pub(crate) fn uniform(degree: usize, count: usize) -> Option<Self> {
        let values = (0..count + degree + 1)
            .map(|knot| knot as f64 - degree as f64)
            .collect();
        Self::new(degree, values, count)
    }

    pub(crate) fn domain(&self) -> (f64, f64) {
        (self.values[self.degree], self.values[self.count])
    }

    fn span(&self, t: f64) -> usize {
        let (start, end) = self.domain();
        if t >= end {
            let mut span = self.count - 1;
            while span > self.degree && self.values[span] >= self.values[span + 1] {
                span -= 1;
            }
            return span;
        }
        if t <= start {
            let mut span = self.degree;
            while span + 1 < self.count && self.values[span + 1] <= self.values[span] {
                span += 1;
            }
            return span;
        }
        let (mut low, mut high) = (self.degree, self.count);
        let mut middle = (low + high) / 2;
        while t < self.values[middle] || t >= self.values[middle + 1] {
            if t < self.values[middle] {
                high = middle;
            } else {
                low = middle;
            }
            let next = (low + high) / 2;
            if next == middle {
                break;
            }
            middle = next;
        }
        middle
    }

    /// Non-zero basis functions of degree `degree` at `t` in `span`.
    fn basis(&self, span: usize, t: f64, degree: usize) -> [f64; MAX_DEGREE + 1] {
        let knots = &self.values;
        let mut basis = [0.0; MAX_DEGREE + 1];
        let mut left = [0.0; MAX_DEGREE + 1];
        let mut right = [0.0; MAX_DEGREE + 1];
        basis[0] = 1.0;
        for j in 1..=degree {
            left[j] = t - knots[span + 1 - j];
            right[j] = knots[span + j] - t;
            let mut saved = 0.0;
            for r in 0..j {
                let denominator = right[r + 1] + left[j - r];
                let temp = if denominator != 0.0 {
                    basis[r] / denominator
                } else {
                    0.0
                };
                basis[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            basis[j] = saved;
        }
        basis
    }

    /// Span, basis values, and first derivatives at `t` (clamped).
    fn evaluate(&self, t: f64) -> (usize, [f64; MAX_DEGREE + 1], [f64; MAX_DEGREE + 1]) {
        let (start, end) = self.domain();
        let t = t.clamp(start, end);
        let p = self.degree;
        let span = self.span(t);
        let values = self.basis(span, t, p);
        let lower = self.basis(span, t, p - 1);
        let knots = &self.values;
        let mut derivatives = [0.0; MAX_DEGREE + 1];
        for (r, derivative) in derivatives.iter_mut().enumerate().take(p + 1) {
            let i = span - p + r;
            let mut value = 0.0;
            if r >= 1 {
                let denominator = knots[i + p] - knots[i];
                if denominator != 0.0 {
                    value += lower[r - 1] / denominator;
                }
            }
            if r < p {
                let denominator = knots[i + p + 1] - knots[i + 1];
                if denominator != 0.0 {
                    value -= lower[r] / denominator;
                }
            }
            *derivative = value * p as f64;
        }
        (span, values, derivatives)
    }
}

fn homogeneous(point: DVec3, weight: f64) -> DVec4 {
    (point * weight).extend(weight)
}

fn dehomogenize(value: DVec4, derivative: DVec4) -> (DVec3, DVec3) {
    let w = if value.w.abs() > 1e-300 { value.w } else { 1.0 };
    let point = value.truncate() / w;
    (point, (derivative.truncate() - point * derivative.w) / w)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BSplineCurve {
    pub knots: Knots,
    pub control: Vec<DVec4>,
}

impl BSplineCurve {
    pub(crate) fn new(knots: Knots, points: Vec<DVec3>, weights: Option<Vec<f64>>) -> Option<Self> {
        if points.len() != knots.count {
            return None;
        }
        let weights = match weights {
            Some(weights) if weights.len() == points.len() => weights,
            Some(_) => return None,
            None => vec![1.0; points.len()],
        };
        if weights
            .iter()
            .any(|weight| !(weight.is_finite() && *weight > 0.0))
        {
            return None;
        }
        Some(Self {
            knots,
            control: points
                .into_iter()
                .zip(weights)
                .map(|(point, weight)| homogeneous(point, weight))
                .collect(),
        })
    }

    fn evaluate(&self, t: f64) -> (DVec3, DVec3) {
        let (span, values, derivatives) = self.knots.evaluate(t);
        let p = self.knots.degree;
        let mut point = DVec4::ZERO;
        let mut derivative = DVec4::ZERO;
        for r in 0..=p {
            let control = self.control[span - p + r];
            point += control * values[r];
            derivative += control * derivatives[r];
        }
        dehomogenize(point, derivative)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Curve {
    Line { origin: DVec3, direction: DVec3 },
    Circle { frame: Frame, radius: f64 },
    Ellipse { frame: Frame, a: f64, b: f64 },
    BSpline(BSplineCurve),
    Polyline(Vec<DVec3>),
}

impl Curve {
    pub(crate) fn evaluate(&self, t: f64) -> (DVec3, DVec3) {
        match self {
            Self::Line { origin, direction } => (*origin + *direction * t, *direction),
            Self::Circle { frame, radius } => {
                let (sin, cos) = t.sin_cos();
                (
                    frame.origin + (frame.x * cos + frame.y * sin) * *radius,
                    (frame.y * cos - frame.x * sin) * *radius,
                )
            }
            Self::Ellipse { frame, a, b } => {
                let (sin, cos) = t.sin_cos();
                (
                    frame.origin + frame.x * (a * cos) + frame.y * (b * sin),
                    frame.y * (b * cos) - frame.x * (a * sin),
                )
            }
            Self::BSpline(curve) => curve.evaluate(t),
            Self::Polyline(points) => {
                let last = points.len().saturating_sub(1);
                if last == 0 {
                    return (points.first().copied().unwrap_or_default(), DVec3::ZERO);
                }
                let t = t.clamp(0.0, last as f64);
                let segment = (t.floor() as usize).min(last - 1);
                let (start, end) = (points[segment], points[segment + 1]);
                (start.lerp(end, t - segment as f64), end - start)
            }
        }
    }

    pub(crate) fn point(&self, t: f64) -> DVec3 {
        self.evaluate(t).0
    }

    pub(crate) fn period(&self) -> Option<f64> {
        matches!(self, Self::Circle { .. } | Self::Ellipse { .. }).then_some(TAU)
    }

    /// Parameter range of a bounded curve.
    pub(crate) fn domain(&self) -> Option<(f64, f64)> {
        match self {
            Self::Line { .. } => None,
            Self::Circle { .. } | Self::Ellipse { .. } => Some((0.0, TAU)),
            Self::BSpline(curve) => Some(curve.knots.domain()),
            Self::Polyline(points) => Some((0.0, points.len().saturating_sub(1) as f64)),
        }
    }

    /// Parameter of the curve point closest to `point`.
    pub(crate) fn parameter_of(&self, point: DVec3) -> f64 {
        match self {
            Self::Line { origin, direction } => {
                let length = direction.length_squared();
                if length > 0.0 {
                    (point - *origin).dot(*direction) / length
                } else {
                    0.0
                }
            }
            Self::Circle { frame, .. } => {
                let local = frame.to_local(point);
                local.y.atan2(local.x).rem_euclid(TAU)
            }
            Self::Ellipse { frame, a, b } => {
                let local = frame.to_local(point);
                (local.y / b.max(1e-300))
                    .atan2(local.x / a.max(1e-300))
                    .rem_euclid(TAU)
            }
            Self::Polyline(points) => {
                let mut best = (f64::INFINITY, 0.0);
                for (index, pair) in points.windows(2).enumerate() {
                    let segment = pair[1] - pair[0];
                    let length = segment.length_squared();
                    let t = if length > 0.0 {
                        ((point - pair[0]).dot(segment) / length).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let distance = (pair[0] + segment * t).distance_squared(point);
                    if distance < best.0 {
                        best = (distance, index as f64 + t);
                    }
                }
                best.1
            }
            Self::BSpline(curve) => {
                let (start, end) = curve.knots.domain();
                let samples = (curve.knots.count * 4).clamp(16, 512);
                let mut best = (f64::INFINITY, start);
                for sample in 0..=samples {
                    let t = start + (end - start) * sample as f64 / samples as f64;
                    let distance = curve.evaluate(t).0.distance_squared(point);
                    if distance < best.0 {
                        best = (distance, t);
                    }
                }
                // Newton on (C - p) . C' = 0, with C'' by central difference.
                let mut t = best.1;
                let h = (end - start).abs().max(1e-12) * 1e-6;
                for _ in 0..32 {
                    let (position, first) = curve.evaluate(t);
                    let (above, below) = ((t + h).min(end), (t - h).max(start));
                    let second = (curve.evaluate(above).1 - curve.evaluate(below).1)
                        / (above - below).max(1e-300);
                    let residual = position - point;
                    let slope = first.length_squared() + residual.dot(second);
                    let step = if slope > 1e-300 {
                        -residual.dot(first) / slope
                    } else if first.length_squared() > 1e-300 {
                        -residual.dot(first) / first.length_squared()
                    } else {
                        break;
                    };
                    let limit = (end - start).abs() * 0.25;
                    let next = (t + step.clamp(-limit, limit)).clamp(start, end);
                    if !next.is_finite() {
                        break;
                    }
                    let done = (next - t).abs() <= 1e-14 * (end - start).abs().max(1.0);
                    t = next;
                    if done {
                        break;
                    }
                }
                if curve.evaluate(t).0.distance_squared(point) > best.0 {
                    best.1
                } else {
                    t
                }
            }
        }
    }

    /// Parameters from `start` to `end` (either order) whose chords stay
    /// within `tolerance` of the curve and turn at most `max_angle` radians.
    pub(crate) fn sample(&self, start: f64, end: f64, tolerance: f64, max_angle: f64) -> Vec<f64> {
        let tolerance = tolerance.max(1e-12);
        let uniform = |count: usize| -> Vec<f64> {
            let count = count.clamp(1, 4096);
            (0..=count)
                .map(|index| start + (end - start) * index as f64 / count as f64)
                .collect()
        };
        match self {
            Self::Line { .. } => vec![start, end],
            Self::Circle { radius, .. } => uniform(arc_segments(
                *radius,
                (end - start).abs(),
                tolerance,
                max_angle,
            )),
            Self::Ellipse { a, b, .. } => uniform(arc_segments(
                a.max(*b),
                (end - start).abs(),
                tolerance,
                max_angle,
            )),
            Self::Polyline(_) => {
                let mut params = vec![start];
                let (low, high) = (start.min(end), start.max(end));
                let mut breaks: Vec<f64> = ((low.floor() as i64 + 1)..=(high.ceil() as i64 - 1))
                    .map(|knot| knot as f64)
                    .filter(|knot| *knot > low && *knot < high)
                    .collect();
                if end < start {
                    breaks.reverse();
                }
                params.extend(breaks);
                params.push(end);
                params
            }
            Self::BSpline(curve) => {
                let (low, high) = (start.min(end), start.max(end));
                let spans = curve
                    .knots
                    .values
                    .windows(2)
                    .filter(|pair| pair[1] > pair[0] && pair[1] > low && pair[0] < high)
                    .count()
                    .max(1);
                let initial = (spans * (curve.knots.degree + 1)).clamp(4, 512);
                let seeds = uniform(initial);
                let mut params = vec![start];
                for pair in seeds.windows(2) {
                    self.refine(pair[0], pair[1], tolerance, max_angle, 0, &mut params);
                }
                params
            }
        }
    }

    fn refine(
        &self,
        start: f64,
        end: f64,
        tolerance: f64,
        max_angle: f64,
        depth: usize,
        params: &mut Vec<f64>,
    ) {
        let (a, ta) = self.evaluate(start);
        let (b, tb) = self.evaluate(end);
        let middle = (start + end) / 2.0;
        let m = self.point(middle);
        let deviation = distance_to_segment(m, a, b);
        let turn = ta.angle_between(tb);
        if depth < 12 && (deviation > tolerance || (turn.is_finite() && turn > max_angle)) {
            self.refine(start, middle, tolerance, max_angle, depth + 1, params);
            self.refine(middle, end, tolerance, max_angle, depth + 1, params);
        } else {
            params.push(end);
        }
    }
}

/// Segments for an arc of `sweep` radians and `radius`.
pub(crate) fn arc_segments(radius: f64, sweep: f64, tolerance: f64, max_angle: f64) -> usize {
    let chord_angle = if radius > tolerance {
        2.0 * (1.0 - tolerance / radius).clamp(-1.0, 1.0).acos()
    } else {
        max_angle
    };
    let step = chord_angle.min(max_angle).max(1e-3);
    let minimum = if sweep > TAU * 0.99 { 6 } else { 1 };
    ((sweep / step).ceil() as usize).max(minimum)
}

pub(crate) fn distance_to_segment(point: DVec3, a: DVec3, b: DVec3) -> f64 {
    let segment = b - a;
    let length = segment.length_squared();
    let t = if length > 0.0 {
        ((point - a).dot(segment) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    point.distance(a + segment * t)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BSplineSurface {
    pub u: Knots,
    pub v: Knots,
    /// `u.count` rows of `v.count` homogeneous points.
    pub control: Vec<DVec4>,
    /// Closed in u or v: first and last control rows coincide.
    pub closed: [bool; 2],
    /// Coarse samples for seeding closest-point searches.
    samples: Vec<(DVec2, DVec3)>,
}

impl BSplineSurface {
    pub(crate) fn new(
        u: Knots,
        v: Knots,
        points: Vec<Vec<DVec3>>,
        weights: Option<Vec<Vec<f64>>>,
    ) -> Option<Self> {
        if points.len() != u.count || points.iter().any(|row| row.len() != v.count) {
            return None;
        }
        let mut control = Vec::with_capacity(u.count * v.count);
        for (i, row) in points.iter().enumerate() {
            for (j, point) in row.iter().enumerate() {
                let weight = match &weights {
                    Some(weights) => *weights.get(i)?.get(j)?,
                    None => 1.0,
                };
                if !(weight.is_finite() && weight > 0.0) || !point.is_finite() {
                    return None;
                }
                control.push(homogeneous(*point, weight));
            }
        }
        let extent = points.iter().flatten().fold(
            (DVec3::INFINITY, DVec3::NEG_INFINITY),
            |(min, max), point| (min.min(*point), max.max(*point)),
        );
        let size = (extent.1 - extent.0).length().max(1e-12);
        let mut surface = Self {
            u,
            v,
            control,
            closed: [false, false],
            samples: Vec::new(),
        };
        // Closed where opposite domain edges evaluate to the same curve;
        // periodic splines close smoothly without their first and last
        // control points meeting.
        let (u0, u1) = surface.u.domain();
        let (v0, v1) = surface.v.domain();
        let meets = |a: &dyn Fn(f64) -> DVec2, b: &dyn Fn(f64) -> DVec2| {
            (0..=8).all(|step| {
                let t = step as f64 / 8.0;
                let (pa, pb) = (a(t), b(t));
                surface
                    .evaluate(pa.x, pa.y)
                    .0
                    .distance(surface.evaluate(pb.x, pb.y).0)
                    <= size * 1e-7
            })
        };
        let closed_u = meets(&|t| DVec2::new(u0, v0 + (v1 - v0) * t), &|t| {
            DVec2::new(u1, v0 + (v1 - v0) * t)
        });
        let closed_v = meets(&|t| DVec2::new(u0 + (u1 - u0) * t, v0), &|t| {
            DVec2::new(u0 + (u1 - u0) * t, v1)
        });
        surface.closed = [closed_u, closed_v];
        let (su, sv) = (
            (surface.u.count * 3).clamp(8, 48),
            (surface.v.count * 3).clamp(8, 48),
        );
        for i in 0..=su {
            for j in 0..=sv {
                let uv = DVec2::new(
                    u0 + (u1 - u0) * i as f64 / su as f64,
                    v0 + (v1 - v0) * j as f64 / sv as f64,
                );
                let point = surface.evaluate(uv.x, uv.y).0;
                surface.samples.push((uv, point));
            }
        }
        Some(surface)
    }

    fn evaluate(&self, u: f64, v: f64) -> (DVec3, DVec3, DVec3) {
        let (span_u, nu, du) = self.u.evaluate(u);
        let (span_v, nv, dv) = self.v.evaluate(v);
        let (p, q) = (self.u.degree, self.v.degree);
        let mut point = DVec4::ZERO;
        let mut along_u = DVec4::ZERO;
        let mut along_v = DVec4::ZERO;
        for r in 0..=p {
            let row = (span_u - p + r) * self.v.count;
            for s in 0..=q {
                let control = self.control[row + span_v - q + s];
                point += control * (nu[r] * nv[s]);
                along_u += control * (du[r] * nv[s]);
                along_v += control * (nu[r] * dv[s]);
            }
        }
        let w = if point.w.abs() > 1e-300 { point.w } else { 1.0 };
        let position = point.truncate() / w;
        (
            position,
            (along_u.truncate() - position * along_u.w) / w,
            (along_v.truncate() - position * along_v.w) / w,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Surface {
    Plane(Frame),
    Cylinder {
        frame: Frame,
        radius: f64,
    },
    /// `tan` is the tangent of the semi-angle.
    Cone {
        frame: Frame,
        radius: f64,
        tan: f64,
    },
    Sphere {
        frame: Frame,
        radius: f64,
    },
    Torus {
        frame: Frame,
        major: f64,
        minor: f64,
    },
    BSpline(Box<BSplineSurface>),
    Extrusion {
        curve: Curve,
        direction: DVec3,
    },
    Revolution {
        curve: Curve,
        origin: DVec3,
        axis: DVec3,
    },
    Offset {
        basis: Box<Surface>,
        distance: f64,
    },
}

fn rotate(vector: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    let (sin, cos) = angle.sin_cos();
    vector * cos + axis.cross(vector) * sin + axis * axis.dot(vector) * (1.0 - cos)
}

impl Surface {
    /// Position and first partial derivatives.
    pub(crate) fn evaluate(&self, u: f64, v: f64) -> (DVec3, DVec3, DVec3) {
        match self {
            Self::Plane(frame) => (frame.origin + frame.x * u + frame.y * v, frame.x, frame.y),
            Self::Cylinder { frame, radius } => {
                let (sin, cos) = u.sin_cos();
                let radial = frame.x * cos + frame.y * sin;
                (
                    frame.origin + radial * *radius + frame.z * v,
                    (frame.y * cos - frame.x * sin) * *radius,
                    frame.z,
                )
            }
            Self::Cone { frame, radius, tan } => {
                let (sin, cos) = u.sin_cos();
                let radial = frame.x * cos + frame.y * sin;
                let r = radius + v * tan;
                (
                    frame.origin + radial * r + frame.z * v,
                    (frame.y * cos - frame.x * sin) * r,
                    radial * *tan + frame.z,
                )
            }
            Self::Sphere { frame, radius } => {
                let (sin_u, cos_u) = u.sin_cos();
                let (sin_v, cos_v) = v.sin_cos();
                let radial = frame.x * cos_u + frame.y * sin_u;
                (
                    frame.origin + (radial * cos_v + frame.z * sin_v) * *radius,
                    (frame.y * cos_u - frame.x * sin_u) * (radius * cos_v),
                    (frame.z * cos_v - radial * sin_v) * *radius,
                )
            }
            Self::Torus {
                frame,
                major,
                minor,
            } => {
                let (sin_u, cos_u) = u.sin_cos();
                let (sin_v, cos_v) = v.sin_cos();
                let radial = frame.x * cos_u + frame.y * sin_u;
                let r = major + minor * cos_v;
                (
                    frame.origin + radial * r + frame.z * (minor * sin_v),
                    (frame.y * cos_u - frame.x * sin_u) * r,
                    (frame.z * cos_v - radial * sin_v) * *minor,
                )
            }
            Self::BSpline(surface) => {
                let u = wrap(u, surface.u.domain(), surface.closed[0]);
                let v = wrap(v, surface.v.domain(), surface.closed[1]);
                surface.evaluate(u, v)
            }
            Self::Extrusion { curve, direction } => {
                let u = match curve.period() {
                    Some(_) => u,
                    None => match curve.domain() {
                        Some(domain) if self.periods()[0].is_some() => wrap(u, domain, true),
                        _ => u,
                    },
                };
                let (point, derivative) = curve.evaluate(u);
                (point + *direction * v, derivative, *direction)
            }
            Self::Revolution {
                curve,
                origin,
                axis,
            } => {
                let (point, derivative) = curve.evaluate(v);
                let offset = point - *origin;
                let rotated = rotate(offset, *axis, u);
                (
                    *origin + rotated,
                    axis.cross(rotated),
                    rotate(derivative, *axis, u),
                )
            }
            Self::Offset { basis, distance } => {
                let (point, along_u, along_v) = basis.evaluate(u, v);
                let normal = along_u.cross(along_v).normalize_or_zero();
                (point + normal * *distance, along_u, along_v)
            }
        }
    }

    pub(crate) fn point(&self, uv: DVec2) -> DVec3 {
        self.evaluate(uv.x, uv.y).0
    }

    /// Period of each parameter, when it wraps.
    pub(crate) fn periods(&self) -> [Option<f64>; 2] {
        match self {
            Self::Plane(_) => [None, None],
            Self::Cylinder { .. } | Self::Cone { .. } | Self::Sphere { .. } => [Some(TAU), None],
            Self::Torus { .. } => [Some(TAU), Some(TAU)],
            Self::Revolution { .. } => [Some(TAU), None],
            Self::BSpline(surface) => {
                let period = |knots: &Knots, closed: bool| {
                    closed.then(|| knots.domain().1 - knots.domain().0)
                };
                [
                    period(&surface.u, surface.closed[0]),
                    period(&surface.v, surface.closed[1]),
                ]
            }
            Self::Extrusion { curve, .. } => match curve {
                Curve::Circle { .. } | Curve::Ellipse { .. } => [Some(TAU), None],
                Curve::BSpline(spline) => {
                    let (start, end) = spline.knots.domain();
                    let closed = spline.evaluate(start).0.distance(spline.evaluate(end).0)
                        <= 1e-9 * spline_size(spline);
                    [closed.then_some(end - start), None]
                }
                _ => [None, None],
            },
            Self::Offset { basis, .. } => basis.periods(),
        }
    }

    /// Bounded parameter ranges, where the surface has them.
    pub(crate) fn domain(&self) -> [Option<(f64, f64)>; 2] {
        match self {
            Self::Sphere { .. } => [Some((0.0, TAU)), Some((-FRAC_PI_2, FRAC_PI_2))],
            Self::Torus { .. } => [Some((0.0, TAU)), Some((0.0, TAU))],
            Self::Cylinder { .. } | Self::Cone { .. } => [Some((0.0, TAU)), None],
            Self::BSpline(surface) => [Some(surface.u.domain()), Some(surface.v.domain())],
            Self::Revolution { curve, .. } => [Some((0.0, TAU)), curve.domain()],
            Self::Extrusion { curve, .. } => [curve.domain(), None],
            Self::Plane(_) => [None, None],
            Self::Offset { basis, .. } => basis.domain(),
        }
    }

    /// The parameter `v` where the surface collapses to a point on the given
    /// side (`true` = high v): sphere poles, the cone apex, or a degenerate
    /// end of a B-spline or revolved curve.
    pub(crate) fn degenerate_v(&self, high: bool, tolerance: f64) -> Option<f64> {
        match self {
            Self::Sphere { .. } => Some(if high { FRAC_PI_2 } else { -FRAC_PI_2 }),
            Self::Cone { radius, tan, .. } => (tan.abs() > 1e-12).then(|| -radius / tan),
            Self::Offset { basis, .. } => basis.degenerate_v(high, tolerance),
            _ => {
                let (start, end) = self.domain()[1]?;
                let v = if high { end } else { start };
                let (u0, u1) = self.domain()[0].unwrap_or((0.0, 1.0));
                let a = self.point(DVec2::new(u0, v));
                let b = self.point(DVec2::new((u0 + u1) / 2.0, v));
                let c = self.point(DVec2::new(u0 + (u1 - u0) * 0.25, v));
                (a.distance(b) <= tolerance && a.distance(c) <= tolerance).then_some(v)
            }
        }
    }

    /// Parameters of the surface point closest to `point`, near `seed` when
    /// the surface has no closed-form inverse.
    pub(crate) fn project(&self, point: DVec3, seed: Option<DVec2>, tolerance: f64) -> DVec2 {
        match self {
            Self::Plane(frame) => {
                let local = frame.to_local(point);
                DVec2::new(local.x, local.y)
            }
            Self::Cylinder { frame, .. } => {
                let local = frame.to_local(point);
                DVec2::new(local.y.atan2(local.x), local.z)
            }
            Self::Cone { frame, radius, tan } => {
                let local = frame.to_local(point);
                let rho = local.x.hypot(local.y);
                let u = local.y.atan2(local.x);
                let along = |rho: f64| ((rho - radius) * tan + local.z) / (1.0 + tan * tan);
                let near = DVec2::new(u, along(rho));
                // Past the apex the cone continues as a second nappe, seen
                // from the far side of the axis.
                let far = DVec2::new(u + PI, along(-rho));
                if self.point(far).distance(point) < self.point(near).distance(point) {
                    far
                } else {
                    near
                }
            }
            Self::Sphere { frame, .. } => {
                let local = frame.to_local(point);
                DVec2::new(
                    local.y.atan2(local.x),
                    local.z.atan2(local.x.hypot(local.y)),
                )
            }
            Self::Torus { frame, major, .. } => {
                let local = frame.to_local(point);
                let rho = local.x.hypot(local.y);
                let u = local.y.atan2(local.x);
                let outer = DVec2::new(u, local.z.atan2(rho - major));
                // A spindle torus (major < minor) also passes through the
                // point from the far side of the axis, on its inner sheet.
                let inner = DVec2::new(u + PI, local.z.atan2(-rho - major));
                if self.point(inner).distance(point) < self.point(outer).distance(point) {
                    inner
                } else {
                    outer
                }
            }
            Self::Offset { basis, .. } => basis.project(point, seed, tolerance),
            Self::BSpline(surface) => {
                let mut best = seed.map(|seed| self.newton(point, seed));
                let good = best.is_some_and(|uv| self.point(uv).distance(point) <= tolerance * 4.0);
                if !good {
                    let nearest = surface
                        .samples
                        .iter()
                        .min_by(|a, b| {
                            a.1.distance_squared(point)
                                .total_cmp(&b.1.distance_squared(point))
                        })
                        .map(|sample| sample.0)
                        .unwrap_or_default();
                    let candidate = self.newton(point, nearest);
                    best = Some(match best {
                        Some(previous)
                            if self.point(previous).distance(point)
                                <= self.point(candidate).distance(point) =>
                        {
                            previous
                        }
                        _ => candidate,
                    });
                }
                best.unwrap_or_default()
            }
            Self::Extrusion { curve, direction } => {
                if let Some(seed) = seed {
                    let uv = self.newton(point, seed);
                    if self.point(uv).distance(point) <= tolerance * 1e-3 {
                        return uv;
                    }
                }
                // The curve point whose extrusion line passes nearest: the
                // closest one seen along the extrusion, which need not be
                // normal to the curve's plane.
                let axis = direction.normalize_or_zero();
                let across = |offset: DVec3| offset - axis * offset.dot(axis);
                let u = match (curve, curve.domain()) {
                    (
                        Curve::Line {
                            origin,
                            direction: along,
                        },
                        _,
                    ) => {
                        let flat = across(*along);
                        across(point - *origin).dot(flat) / flat.length_squared().max(1e-300)
                    }
                    (_, Some((start, end))) => {
                        let samples = 128;
                        let mut best = (f64::INFINITY, start);
                        for sample in 0..=samples {
                            let t = start + (end - start) * sample as f64 / samples as f64;
                            let distance = across(curve.point(t) - point).length_squared();
                            if distance < best.0 {
                                best = (distance, t);
                            }
                        }
                        best.1
                    }
                    (_, None) => curve.parameter_of(point),
                };
                let length = direction.length_squared().max(1e-300);
                let v = (point - curve.point(u)).dot(*direction) / length;
                self.newton(point, DVec2::new(u, v))
            }
            Self::Revolution {
                curve,
                origin,
                axis,
            } => {
                let offset = point - *origin;
                let radial = offset - *axis * offset.dot(*axis);
                let reference_v = curve.domain().map_or(0.0, |(a, b)| (a + b) / 2.0);
                let mut v = reference_v;
                let mut u = 0.0;
                for _ in 0..2 {
                    let reference = curve.point(v) - *origin;
                    let reference_radial = reference - *axis * reference.dot(*axis);
                    u = signed_angle(reference_radial, radial, *axis);
                    let back = *origin + rotate(offset, *axis, -u);
                    v = curve.parameter_of(back);
                }
                self.newton(point, DVec2::new(u, v))
            }
        }
    }

    /// Gauss-Newton closest point from `start`.
    fn newton(&self, point: DVec3, start: DVec2) -> DVec2 {
        let domain = self.domain();
        let periods = self.periods();
        let clamp = |uv: DVec2| {
            let mut uv = uv;
            for axis in 0..2 {
                if periods[axis].is_none()
                    && let Some((low, high)) = domain[axis]
                {
                    uv[axis] = uv[axis].clamp(low, high);
                }
            }
            uv
        };
        let limit = |axis: usize| {
            domain[axis]
                .map(|(low, high)| (high - low).abs() * 0.25)
                .unwrap_or(f64::INFINITY)
        };
        let mut uv = clamp(start);
        for _ in 0..32 {
            let (position, du, dv) = self.evaluate(uv.x, uv.y);
            let residual = position - point;
            let (a, b, c) = (du.dot(du), du.dot(dv), dv.dot(dv));
            let (g1, g2) = (residual.dot(du), residual.dot(dv));
            let determinant = a * c - b * b;
            let mut step = if determinant.abs() > 1e-300 {
                DVec2::new(
                    -(c * g1 - b * g2) / determinant,
                    -(a * g2 - b * g1) / determinant,
                )
            } else if a > 1e-300 {
                DVec2::new(-g1 / a, 0.0)
            } else if c > 1e-300 {
                DVec2::new(0.0, -g2 / c)
            } else {
                break;
            };
            // At a domain bound with the step pointing out, hold that
            // parameter and solve for the other alone (the closest point
            // lies on the boundary curve).
            let pinned = |axis: usize, step: f64| {
                periods[axis].is_none()
                    && domain[axis].is_some_and(|(low, high)| {
                        (uv[axis] <= low && step < 0.0) || (uv[axis] >= high && step > 0.0)
                    })
            };
            match (pinned(0, step.x), pinned(1, step.y)) {
                (true, true) => break,
                (true, false) if c > 1e-300 => step = DVec2::new(0.0, -g2 / c),
                (false, true) if a > 1e-300 => step = DVec2::new(-g1 / a, 0.0),
                _ => {}
            }
            let step = DVec2::new(
                step.x.clamp(-limit(0), limit(0)),
                step.y.clamp(-limit(1), limit(1)),
            );
            if !step.is_finite() {
                break;
            }
            let next = clamp(uv + step);
            let moved = (next - uv).abs();
            uv = next;
            if moved.x <= 1e-13 * (1.0 + uv.x.abs()) && moved.y <= 1e-13 * (1.0 + uv.y.abs()) {
                break;
            }
        }
        uv
    }
}

fn spline_size(spline: &BSplineCurve) -> f64 {
    let (min, max) = spline.control.iter().fold(
        (DVec3::INFINITY, DVec3::NEG_INFINITY),
        |(min, max), control| {
            let point = control.truncate() / control.w;
            (min.min(point), max.max(point))
        },
    );
    (max - min).length().max(1e-12)
}

fn wrap(t: f64, (start, end): (f64, f64), closed: bool) -> f64 {
    let period = end - start;
    if closed && period > 0.0 && (t < start || t > end) {
        start + (t - start).rem_euclid(period)
    } else {
        t
    }
}

/// Angle from `from` to `to` about `axis`, in (-pi, pi].
fn signed_angle(from: DVec3, to: DVec3, axis: DVec3) -> f64 {
    from.cross(to).dot(axis).atan2(from.dot(to))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn quarter_circle_nurbs() -> BSplineCurve {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        BSplineCurve::new(
            Knots::new(2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3).unwrap(),
            vec![DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y],
            Some(vec![1.0, w, 1.0]),
        )
        .unwrap()
    }

    #[test]
    fn rational_b_spline_curves_are_exact_circles() {
        let curve = Curve::BSpline(quarter_circle_nurbs());
        for step in 0..=10 {
            let point = curve.point(step as f64 / 10.0);
            assert!((point.length() - 1.0).abs() < 1e-12);
        }
        let t = curve.parameter_of(DVec3::new(0.8, 0.6, 0.0) * 2.0);
        assert!(curve.point(t).distance(DVec3::new(0.8, 0.6, 0.0)) < 1e-9);
        // The derivative matches a finite difference.
        let (_, derivative) = curve.evaluate(0.3);
        let numeric = (curve.point(0.3 + 1e-6) - curve.point(0.3 - 1e-6)) / 2e-6;
        assert!(derivative.distance(numeric) < 1e-6);
    }

    #[test]
    fn sampling_meets_the_chord_tolerance() {
        let curve = Curve::BSpline(quarter_circle_nurbs());
        let params = curve.sample(0.0, 1.0, 1e-3, 1.0);
        for pair in params.windows(2) {
            let middle = curve.point((pair[0] + pair[1]) / 2.0);
            assert!(
                distance_to_segment(middle, curve.point(pair[0]), curve.point(pair[1])) <= 1e-3
            );
        }
        let circle = Curve::Circle {
            frame: Frame::IDENTITY,
            radius: 1.0,
        };
        // 0.1 allows 0.9 rad chords; the 0.5 rad limit wins: 13 segments.
        assert_eq!(circle.sample(0.0, TAU, 0.1, 0.5).len(), 14);
        assert!(circle.sample(0.0, TAU, 1e-4, 0.5).len() > 200);
    }

    #[test]
    fn analytic_and_numeric_projection_round_trip() {
        let frame = Frame::new(DVec3::new(1.0, 2.0, 3.0), Some(DVec3::Y), Some(DVec3::Z));
        let surfaces = [
            Surface::Cylinder { frame, radius: 2.0 },
            Surface::Cone {
                frame,
                radius: 2.0,
                tan: 0.5,
            },
            Surface::Sphere { frame, radius: 2.0 },
            Surface::Torus {
                frame,
                major: 3.0,
                minor: 1.0,
            },
            Surface::Revolution {
                curve: Curve::Line {
                    origin: DVec3::new(2.0, 0.0, 0.0),
                    direction: DVec3::new(0.5, 0.0, 1.0),
                },
                origin: DVec3::ZERO,
                axis: DVec3::Z,
            },
            Surface::Extrusion {
                curve: Curve::BSpline(quarter_circle_nurbs()),
                direction: DVec3::new(0.0, 0.0, 2.0),
            },
        ];
        for surface in surfaces {
            let uv = DVec2::new(0.4, 0.3);
            let point = surface.point(uv);
            let back = surface.project(point, None, 1e-6);
            assert!(
                surface.point(back).distance(point) < 1e-7,
                "{surface:?} {back}"
            );
            // S_u x S_v is the outward normal of the elementary surfaces.
            let (_, du, dv) = surface.evaluate(uv.x, uv.y);
            if let Surface::Sphere { frame, .. } = &surface {
                assert!(du.cross(dv).dot(point - frame.origin) > 0.0);
            }
        }
    }

    #[test]
    fn projection_finds_inner_sheets_far_nappes_and_oblique_extrusions() {
        let frame = Frame::new(
            DVec3::new(1.0, -2.0, 0.5),
            Some(DVec3::new(0.0, 0.3, -1.0)),
            None,
        );
        // A spindle torus (major < minor): its inner sheet is reached from
        // the far side of the axis. A double cone continues past its apex.
        let spindle = Surface::Torus {
            frame,
            major: 8.25,
            minor: 54.87,
        };
        let cone = Surface::Cone {
            frame,
            radius: 7.5,
            tan: 1.0,
        };
        let oblique = Surface::Extrusion {
            curve: Curve::Ellipse {
                frame: Frame::IDENTITY,
                a: 17.0,
                b: 13.0,
            },
            direction: DVec3::new(0.0, 0.64, 0.77),
        };
        for (surface, uv) in [
            (&spindle, DVec2::new(0.7, 2.0)),
            (&spindle, DVec2::new(-2.5, -1.9)),
            (&cone, DVec2::new(1.0, -15.0)),
            (&oblique, DVec2::new(PI, 20.0)),
            (&oblique, DVec2::new(2.0, -8.0)),
        ] {
            let point = surface.point(uv);
            let back = surface.project(point, None, 1e-6);
            assert!(
                surface.point(back).distance(point) < 1e-7,
                "{surface:?} {uv} -> {back}"
            );
        }
    }

    #[test]
    fn periodic_b_spline_surfaces_close_without_meeting_control_points() {
        // Uniform cubic knots over six control points wrapped by three: the
        // curve closes smoothly though its first and last points differ.
        let ring: Vec<DVec3> = (0..9)
            .map(|i| {
                let angle = (i % 6) as f64 / 6.0 * TAU;
                DVec3::new(angle.cos(), angle.sin(), 0.0)
            })
            .collect();
        let raised: Vec<DVec3> = ring.iter().map(|p| *p + DVec3::Z).collect();
        let surface = BSplineSurface::new(
            Knots::new(1, vec![0.0, 0.0, 1.0, 1.0], 2).unwrap(),
            Knots::uniform(3, 9).unwrap(),
            vec![ring, raised],
            None,
        )
        .unwrap();
        assert_eq!(surface.closed, [false, true]);
        assert_eq!(Surface::BSpline(Box::new(surface)).periods()[1], Some(6.0));
    }

    #[test]
    fn b_spline_surfaces_project_and_detect_closure() {
        let knots = Knots::new(1, vec![0.0, 0.0, 1.0, 1.0], 2).unwrap();
        let surface = Surface::BSpline(Box::new(
            BSplineSurface::new(
                knots.clone(),
                knots,
                vec![
                    vec![DVec3::ZERO, DVec3::new(0.0, 1.0, 0.0)],
                    vec![DVec3::new(1.0, 0.0, 0.0), DVec3::new(1.0, 1.0, 1.0)],
                ],
                None,
            )
            .unwrap(),
        ));
        let point = surface.point(DVec2::new(0.25, 0.75));
        let uv = surface.project(point, None, 1e-9);
        assert!((uv - DVec2::new(0.25, 0.75)).length() < 1e-9);
        assert_eq!(surface.periods(), [None, None]);
        assert_eq!(
            Knots::quasi_uniform(2, 4).unwrap().values,
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
        );
    }
}
