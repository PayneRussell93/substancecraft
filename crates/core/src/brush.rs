//! Brush tips and stroke rasterisation.
//!
//! A stroke's footprint is accumulated with `max`, so overlapping segments
//! and stamps of one stroke never build up beyond the stroke's own opacity.

use crate::math;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::noise::{gradient_noise3, hash3, unit_float};
use crate::region::TexelRect;
use crate::resources::{ResourceImages, luminance};
use crate::{MAX_STROKE_POINTS, ResourceId, raster};

/// Upper bound on stamps placed by one stroke.
const MAX_STAMPS: usize = 65_536;
pub const MIN_BRUSH_SPACING: f32 = 0.02;
pub const MAX_BRUSH_SPACING: f32 = 4.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushShape {
    /// Soft disc; without jitter it is drawn as a continuous capsule.
    #[default]
    Round,
    /// Without jitter or path-following it is swept continuously too, so
    /// diagonal strokes keep straight edges rather than stamp staircases.
    Square,
    /// Clustered speckles, for grime and chipped paint.
    Splatter,
    /// A thin tapered line along the stamp's angle, for scratches.
    Streak,
    /// A hollow circle, for rivets, rings, and water marks.
    Ring,
    /// A soft cloudy blob, for stains and smudges.
    Cloud,
    /// Grayscale bitmap resource: white paints, black does not.
    Bitmap(ResourceId),
}

impl BrushShape {
    pub const BUILT_IN: [Self; 6] = [
        Self::Round,
        Self::Square,
        Self::Splatter,
        Self::Streak,
        Self::Ring,
        Self::Cloud,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Round => "Round",
            Self::Square => "Square",
            Self::Splatter => "Splatter",
            Self::Streak => "Streak",
            Self::Ring => "Ring",
            Self::Cloud => "Cloud",
            Self::Bitmap(_) => "Bitmap",
        }
    }
}

/// How a brush deposits paint along its path.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushTip {
    pub shape: BrushShape,
    /// Fraction of the radius painted at full strength before falloff.
    pub hardness: f32,
    /// Distance between stamps as a fraction of the diameter.
    pub spacing: f32,
    pub angle_degrees: f32,
    /// Rotate stamps to the stroke direction.
    pub follow_path: bool,
    /// Random per-stamp variation, each `0..=1`.
    pub size_jitter: f32,
    pub angle_jitter: f32,
    /// Random offset from the path, in diameters.
    pub scatter: f32,
    pub seed: u32,
    /// Pen pressure scales the radius and/or the opacity.
    pub pressure_size: bool,
    pub pressure_opacity: bool,
}

impl Default for BrushTip {
    fn default() -> Self {
        Self::ROUND
    }
}

impl BrushTip {
    /// Soft round brush with linear falloff: the original stroke look.
    pub const ROUND: Self = Self {
        shape: BrushShape::Round,
        hardness: 0.0,
        spacing: 0.1,
        angle_degrees: 0.0,
        follow_path: false,
        size_jitter: 0.0,
        angle_jitter: 0.0,
        scatter: 0.0,
        seed: 0,
        pressure_size: true,
        pressure_opacity: false,
    };

    pub(crate) fn is_valid(self) -> bool {
        let unit = |value: f32| value.is_finite() && (0.0..=1.0).contains(&value);
        unit(self.hardness)
            && unit(self.size_jitter)
            && unit(self.angle_jitter)
            && self.scatter.is_finite()
            && (0.0..=4.0).contains(&self.scatter)
            && self.spacing.is_finite()
            && (MIN_BRUSH_SPACING..=MAX_BRUSH_SPACING).contains(&self.spacing)
            && self.angle_degrees.is_finite()
    }

    pub fn resource(self) -> Option<ResourceId> {
        match self.shape {
            BrushShape::Bitmap(id) => Some(id),
            _ => None,
        }
    }

    /// Coverage of one stamp (rotated by `angle_degrees`, without jitter)
    /// sampled on a `cells` x `cells` grid, rows top to bottom; for UI
    /// previews.
    pub fn preview(self, cells: usize, resources: Option<&ResourceImages>) -> Vec<f32> {
        let rotation = Vec2::from_angle(-self.angle_degrees.to_radians());
        let size = cells.max(1) as f32;
        (0..cells * cells)
            .map(|index| {
                let offset = Vec2::new(
                    ((index % cells) as f32 + 0.5) / size * 2.0 - 1.0,
                    ((index / cells) as f32 + 0.5) / size * 2.0 - 1.0,
                );
                let local = rotation.rotate(Vec2::new(offset.x, -offset.y));
                self.stamp_coverage(local, 2.0 / size, self.seed, resources)
            })
            .collect()
    }

    /// Continuous sweeps reproduce the classic smooth round stroke, and give
    /// an unrotating square straight edges; any other shape or randomness
    /// needs discrete stamps.
    fn continuous(self) -> bool {
        let swept = match self.shape {
            BrushShape::Round => true,
            BrushShape::Square => !self.follow_path,
            _ => false,
        };
        swept && self.size_jitter == 0.0 && self.angle_jitter == 0.0 && self.scatter == 0.0
    }

    /// Coverage `distance` radii from the centre, where `pixel` is one pixel
    /// in radii.
    fn falloff(self, distance: f32, pixel: f32) -> f32 {
        let hardness = self.hardness.clamp(0.0, 0.999);
        // The ramp runs from `hardness` out to the edge; one narrower than a
        // pixel widens to a pixel about the same middle, so hard edges are
        // antialiased without the brush shrinking.
        let width = (1.0 - hardness).max(pixel);
        let middle = (1.0 + hardness) * 0.5;
        ((middle + width * 0.5 - distance) / width).clamp(0.0, 1.0)
    }

    /// Coverage of one stamp at `local`, the offset from its centre divided
    /// by its radius and rotated into the stamp frame; `pixel` is one pixel
    /// in the same units.
    fn stamp_coverage(
        self,
        local: Vec2,
        pixel: f32,
        stamp_seed: u32,
        resources: Option<&ResourceImages>,
    ) -> f32 {
        match self.shape {
            BrushShape::Round => self.falloff(local.length(), pixel),
            BrushShape::Square => self.falloff(local.x.abs().max(local.y.abs()), pixel),
            BrushShape::Splatter => {
                let disc = self.falloff(local.length(), pixel);
                if disc <= 0.0 {
                    return 0.0;
                }
                let speckle =
                    gradient_noise3(Vec3::new(local.x * 3.5, local.y * 3.5, 0.0), stamp_seed) * 0.5
                        + 0.5;
                disc * ((speckle - 0.52) * 8.0).clamp(0.0, 1.0)
            }
            BrushShape::Streak => {
                // Tapered at both ends, an eighth of the length wide.
                let taper = (1.0 - local.x.abs()).clamp(0.0, 1.0);
                let half_width = 0.125 * taper;
                if half_width <= 0.0 {
                    return 0.0;
                }
                self.falloff(local.y.abs() / half_width, pixel / half_width) * taper.sqrt()
            }
            BrushShape::Ring => {
                // A band around 0.75 of the radius, a quarter radius wide.
                self.falloff((local.length() - 0.75).abs() / 0.25, pixel / 0.25)
            }
            BrushShape::Cloud => {
                let disc = self.falloff(local.length(), pixel);
                if disc <= 0.0 {
                    return 0.0;
                }
                let billow =
                    crate::noise::fbm3(Vec3::new(local.x * 1.6, local.y * 1.6, 0.5), 4, stamp_seed);
                disc * ((billow - 0.35) * 2.5).clamp(0.0, 1.0)
            }
            BrushShape::Bitmap(id) => {
                if local.x.abs() > 1.0 || local.y.abs() > 1.0 {
                    return 0.0;
                }
                resources
                    .and_then(|images| images.get(&id))
                    .map_or(0.0, |image| {
                        let sample = image.sample(local * 0.5 + Vec2::splat(0.5), false);
                        luminance(sample) * sample.w
                    })
            }
        }
    }
}

/// Most a surface-aligned stamp may stretch along one texture axis (its
/// larger singular value); steeper UV distortion stamps unaligned.
pub const MAX_STAMP_STRETCH: f32 = 8.0;

/// How a surface-aligned stamp lies in texture space: a 2 × 2 matrix,
/// columns first (`[m00, m10, m01, m11]`), taking a texel offset (x right,
/// y up the V axis) into the stamp's frame on the surface, in texel-sized
/// units. Its determinant is ±1, so a stamp keeps its texel area while
/// staying round and upright on the surface where the UVs stretch, shear,
/// or turn.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StampFrame(pub [f32; 4]);

impl StampFrame {
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0]);

    pub(crate) fn apply(self, offset: Vec2) -> Vec2 {
        let [m00, m10, m01, m11] = self.0;
        Vec2::new(
            m00 * offset.x + m01 * offset.y,
            m10 * offset.x + m11 * offset.y,
        )
    }

    /// The largest factor by which the frame lengthens a vector. With a
    /// determinant of ±1 its inverse has the same factor, so this is how
    /// far past its radius an aligned stamp reaches in texture space.
    pub fn stretch(self) -> f32 {
        let [a, b, c, d] = self.0;
        let frobenius = a * a + b * b + c * c + d * d;
        let determinant = a * d - b * c;
        let spread = (frobenius * frobenius - 4.0 * determinant * determinant)
            .max(0.0)
            .sqrt();
        ((frobenius + spread) * 0.5).sqrt()
    }

    pub(crate) fn is_valid(self) -> bool {
        let [a, b, c, d] = self.0;
        self.0.iter().all(|value| value.is_finite())
            && ((a * d - b * c).abs() - 1.0).abs() < 1e-3
            && self.stretch() <= MAX_STAMP_STRETCH + 1e-3
    }
}

/// Valid stamp frames: empty, or one valid frame per point.
pub(crate) fn valid_frames(points: &[Vec2], frames: &[StampFrame]) -> bool {
    frames.is_empty()
        || (frames.len() == points.len() && frames.iter().all(|frame| frame.is_valid()))
}

/// The farthest any of `frames` stretches a stamp; 1 without frames.
pub(crate) fn frames_stretch(frames: &[StampFrame]) -> f32 {
    frames
        .iter()
        .map(|frame| frame.stretch())
        .fold(1.0, f32::max)
}

/// Geometry of one stroke in UV space.
pub(crate) struct StrokePath<'a> {
    pub points: &'a [Vec2],
    /// Per-point pen pressure; empty means full pressure everywhere.
    pub pressures: &'a [f32],
    pub radius_pixels: f32,
    pub tip: BrushTip,
    /// Per-point surface alignment; empty stamps in plain texture space.
    pub frames: &'a [StampFrame],
}

impl StrokePath<'_> {
    fn frame(&self, index: usize) -> StampFrame {
        self.frames
            .get(index)
            .copied()
            .unwrap_or(StampFrame::IDENTITY)
    }

    /// Farthest a stamp of this stroke reaches, in radii.
    pub(crate) fn stretch(&self) -> f32 {
        frames_stretch(self.frames)
    }

    fn pressure(&self, index: usize) -> f32 {
        self.pressures
            .get(index)
            .copied()
            .filter(|pressure| pressure.is_finite())
            .map_or(1.0, |pressure| pressure.clamp(0.0, 1.0))
    }

    fn radius_at(&self, pressure: f32) -> f32 {
        if self.tip.pressure_size {
            self.radius_pixels * pressure.max(0.05)
        } else {
            self.radius_pixels
        }
    }

    fn strength_at(&self, pressure: f32) -> f32 {
        if self.tip.pressure_opacity {
            pressure
        } else {
            1.0
        }
    }
}

/// Valid pressure data: empty, or one value in `0..=1` per point.
pub(crate) fn valid_pressures(points: &[Vec2], pressures: &[f32]) -> bool {
    pressures.is_empty()
        || (pressures.len() == points.len()
            && pressures.len() <= MAX_STROKE_POINTS
            && pressures
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)))
}

/// A rectangle of per-pixel stroke coverage.
pub(crate) struct CoverageTile {
    pub(crate) min_x: i64,
    pub(crate) min_y: i64,
    pub(crate) width: i64,
    pub(crate) height: i64,
    values: Vec<f32>,
    /// When tracked, the unit stroke direction (pixel space) of the stamp
    /// or segment that gave each pixel its coverage; zero for a lone dab.
    directions: Option<Vec<Vec2>>,
}

impl CoverageTile {
    /// A zeroed tile over the inclusive pixel bounds, tracking directions
    /// when `directed`, or `None` when the allocation fails.
    fn new((min_x, min_y, max_x, max_y): (i64, i64, i64, i64), directed: bool) -> Option<Self> {
        let mut tile = Self {
            min_x,
            min_y,
            width: max_x - min_x + 1,
            height: max_y - min_y + 1,
            values: Vec::new(),
            directions: None,
        };
        let count = (tile.width * tile.height) as usize;
        tile.values.try_reserve_exact(count).ok()?;
        tile.values.resize(count, 0.0);
        if directed {
            let mut directions = Vec::new();
            directions.try_reserve_exact(count).ok()?;
            directions.resize(count, Vec2::ZERO);
            tile.directions = Some(directions);
        }
        Some(tile)
    }

    fn get(&self, x: i64, y: i64) -> f32 {
        let (local_x, local_y) = (x - self.min_x, y - self.min_y);
        if local_x < 0 || local_y < 0 || local_x >= self.width || local_y >= self.height {
            return 0.0;
        }
        self.values[(local_y * self.width + local_x) as usize]
    }

    /// Bilinear coverage at a fractional pixel position; zero outside.
    pub(crate) fn sample(&self, point: Vec2) -> f32 {
        let (x, y) = (point.x.floor(), point.y.floor());
        let (tx, ty) = (point.x - x, point.y - y);
        let (x, y) = (x as i64, y as i64);
        let top = self.get(x, y) * (1.0 - tx) + self.get(x + 1, y) * tx;
        let bottom = self.get(x, y + 1) * (1.0 - tx) + self.get(x + 1, y + 1) * tx;
        (top * (1.0 - ty) + bottom * ty).min(1.0)
    }

    fn deposit(&mut self, x: i64, y: i64, value: f32, direction: Vec2) {
        let (local_x, local_y) = (x - self.min_x, y - self.min_y);
        if local_x < 0 || local_y < 0 || local_x >= self.width || local_y >= self.height {
            return;
        }
        let index = (local_y * self.width + local_x) as usize;
        if value > self.values[index] {
            self.values[index] = value;
            if let Some(directions) = &mut self.directions {
                directions[index] = direction;
            }
        }
    }
}

/// Farthest a texel can sit from the stroke's points and still be painted;
/// `stretch` is the most a surface-aligned stamp lengthens (1 when none).
fn stroke_reach(radius_pixels: f32, tip: BrushTip, stretch: f32) -> f32 {
    radius_pixels * (1.0 + tip.size_jitter) * std::f32::consts::SQRT_2 * stretch
        + radius_pixels * 2.0 * tip.scatter
        + 1.0
}

/// Inclusive texel bounds `(min_x, min_y, max_x, max_y)` a stroke over
/// `pixels` can touch on a `width` × `height` texture.
pub(crate) fn stroke_pixel_bounds(
    pixels: &[Vec2],
    radius_pixels: f32,
    tip: BrushTip,
    stretch: f32,
    width: u32,
    height: u32,
) -> Option<(i64, i64, i64, i64)> {
    let first = *pixels.first()?;
    let reach = stroke_reach(radius_pixels, tip, stretch);
    let (mut lower, mut upper) = (first, first);
    for pixel in pixels {
        lower = lower.min(*pixel);
        upper = upper.max(*pixel);
    }
    let min_x = ((lower.x - reach).floor() as i64).max(0);
    let min_y = ((lower.y - reach).floor() as i64).max(0);
    let max_x = ((upper.x + reach).ceil() as i64).min(i64::from(width) - 1);
    let max_y = ((upper.y + reach).ceil() as i64).min(i64::from(height) - 1);
    (min_x <= max_x && min_y <= max_y).then_some((min_x, min_y, max_x, max_y))
}

/// Texels a stroke through UV `points` (with this radius, tip, and stamp
/// frames: one per point, or none) can paint on a `width` × `height`
/// texture, or `None` when it misses it. Stamps are placed causally along
/// the path, so the texels that change when a point is appended are the
/// bounds of the last two points (with their frames) alone.
pub fn stroke_texel_bounds(
    points: &[Vec2],
    radius_pixels: f32,
    tip: BrushTip,
    frames: &[StampFrame],
    width: u32,
    height: u32,
) -> Option<TexelRect> {
    if width == 0 || height == 0 {
        return None;
    }
    let pixels = points
        .iter()
        .map(|point| raster::uv_to_pixel(*point, width, height))
        .collect::<Vec<_>>();
    let (min_x, min_y, max_x, max_y) = stroke_pixel_bounds(
        &pixels,
        radius_pixels,
        tip,
        frames_stretch(frames),
        width,
        height,
    )?;
    Some(TexelRect {
        x: min_x as u32,
        y: min_y as u32,
        width: (max_x - min_x + 1) as u32,
        height: (max_y - min_y + 1) as u32,
    })
}

/// Visits every texel of `clip` the stroke touches with its coverage in
/// `0..=1`; the index is row-major within `clip`. Coverage at a texel does
/// not depend on `clip`, so clipped and full rasterisation agree exactly.
pub(crate) fn rasterize_stroke(
    path: &StrokePath<'_>,
    width: u32,
    height: u32,
    resources: Option<&ResourceImages>,
    clip: TexelRect,
    mut apply: impl FnMut(usize, f32),
) {
    rasterize(
        path,
        (width, height),
        resources,
        clip,
        false,
        |index, coverage, _| apply(index, coverage),
    );
}

/// [`rasterize_stroke`], also passing the unit pixel-space direction the
/// stroke travelled where it gave each texel its coverage (zero for a
/// single dab).
pub(crate) fn rasterize_stroke_directed(
    path: &StrokePath<'_>,
    width: u32,
    height: u32,
    resources: Option<&ResourceImages>,
    clip: TexelRect,
    apply: impl FnMut(usize, f32, Vec2),
) {
    rasterize(path, (width, height), resources, clip, true, apply);
}

fn rasterize(
    path: &StrokePath<'_>,
    (width, height): (u32, u32),
    resources: Option<&ResourceImages>,
    clip: TexelRect,
    directed: bool,
    mut apply: impl FnMut(usize, f32, Vec2),
) {
    if path.points.is_empty() || width == 0 || height == 0 || clip.is_empty() {
        return;
    }
    let pixels = path
        .points
        .iter()
        .map(|point| raster::uv_to_pixel(*point, width, height))
        .collect::<Vec<_>>();
    let Some((min_x, min_y, max_x, max_y)) = stroke_pixel_bounds(
        &pixels,
        path.radius_pixels,
        path.tip,
        path.stretch(),
        width,
        height,
    ) else {
        return;
    };
    let min_x = min_x.max(i64::from(clip.x));
    let min_y = min_y.max(i64::from(clip.y));
    let max_x = max_x.min(i64::from(clip.right()) - 1);
    let max_y = max_y.min(i64::from(clip.bottom()) - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }
    let bounds = (min_x, min_y, max_x, max_y);
    let Some(tile) = draw_tile(path, &pixels, bounds, resources, directed) else {
        return;
    };

    for local_y in 0..tile.height {
        for local_x in 0..tile.width {
            let local = (local_y * tile.width + local_x) as usize;
            let coverage = tile.values[local];
            if coverage > 0.0 {
                let index =
                    clip.local_index((tile.min_x + local_x) as u32, (tile.min_y + local_y) as u32);
                let direction = tile
                    .directions
                    .as_ref()
                    .map_or(Vec2::ZERO, |directions| directions[local]);
                apply(index, coverage.min(1.0), direction);
            }
        }
    }
}

/// The stroke through `pixels` drawn into a tile over the inclusive pixel
/// `bounds`. Coverage at a pixel does not depend on the bounds.
pub(crate) fn coverage_tile(
    path: &StrokePath<'_>,
    pixels: &[Vec2],
    bounds: (i64, i64, i64, i64),
    resources: Option<&ResourceImages>,
) -> Option<CoverageTile> {
    draw_tile(path, pixels, bounds, resources, false)
}

fn draw_tile(
    path: &StrokePath<'_>,
    pixels: &[Vec2],
    bounds: (i64, i64, i64, i64),
    resources: Option<&ResourceImages>,
    directed: bool,
) -> Option<CoverageTile> {
    let mut tile = CoverageTile::new(bounds, directed)?;
    // Aligned stamps change shape along the path, so they never merge into
    // capsules.
    if path.tip.continuous() && path.frames.is_empty() {
        draw_capsules(path, pixels, &mut tile);
    } else {
        draw_stamps(path, pixels, &mut tile, resources);
    }
    Some(tile)
}

/// Smooth round stroke: capsules between consecutive points whose radius
/// and strength follow the interpolated pressure. A square tip is swept the
/// same way, keeping its angle.
fn draw_capsules(path: &StrokePath<'_>, pixels: &[Vec2], tile: &mut CoverageTile) {
    let square = path.tip.shape == BrushShape::Square;
    let rotation = Vec2::from_angle(-path.tip.angle_degrees.to_radians());
    // Pixel rows run downward; stamps are upright with V up, then turned.
    let to_stamp = |offset: Vec2| rotation.rotate(Vec2::new(offset.x, -offset.y));
    let corner = if square {
        std::f32::consts::SQRT_2
    } else {
        1.0
    };
    let segment_count = pixels.len().saturating_sub(1).max(1);
    for segment in 0..segment_count {
        let (start_index, end_index) = if pixels.len() == 1 {
            (0, 0)
        } else {
            (segment, segment + 1)
        };
        let (start, end) = (pixels[start_index], pixels[end_index]);
        let (pressure_start, pressure_end) = (path.pressure(start_index), path.pressure(end_index));
        let (radius_start, radius_end) =
            (path.radius_at(pressure_start), path.radius_at(pressure_end));
        // Antialiased edges spill half a pixel past the radius.
        let reach = radius_start.max(radius_end) * corner + 1.0;
        let lower = start.min(end) - Vec2::splat(reach);
        let upper = start.max(end) + Vec2::splat(reach);
        let span = end - start;
        let length_squared = span.length_squared();
        let direction = span.normalize_or_zero();
        for y in (lower.y.floor() as i64).max(tile.min_y)
            ..=(upper.y.ceil() as i64).min(tile.min_y + tile.height - 1)
        {
            for x in (lower.x.floor() as i64).max(tile.min_x)
                ..=(upper.x.ceil() as i64).min(tile.min_x + tile.width - 1)
            {
                let pixel = Vec2::new(x as f32, y as f32);
                let (amount, distance) = if square {
                    swept_square(
                        to_stamp(pixel - start),
                        to_stamp(span),
                        radius_start,
                        radius_end,
                    )
                } else {
                    let amount = if length_squared > f32::EPSILON {
                        ((pixel - start).dot(span) / length_squared).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let radius = radius_start + (radius_end - radius_start) * amount;
                    (amount, pixel.distance(start + span * amount) / radius)
                };
                let radius = radius_start + (radius_end - radius_start) * amount;
                if radius <= 0.0 || !distance.is_finite() {
                    continue;
                }
                let strength =
                    path.strength_at(pressure_start + (pressure_end - pressure_start) * amount);
                let coverage = path.tip.falloff(distance, 1.0 / radius) * strength;
                if coverage > 0.0 {
                    tile.deposit(x, y, coverage, direction);
                }
            }
        }
    }
}

/// The square tip swept from the origin along `span` (both in the stamp
/// frame), its radius running from `radius_start` to `radius_end`: the
/// nearest it comes to `offset` in radii (the square's own distance), and
/// where along the span. Between the breakpoints where an axis term turns
/// or the two terms cross, the ratio is monotonic, so the least sits at a
/// breakpoint or an end.
fn swept_square(offset: Vec2, span: Vec2, radius_start: f32, radius_end: f32) -> (f32, f32) {
    let ratio = |numerator: f32, denominator: f32| {
        (denominator.abs() > f32::EPSILON).then(|| (numerator / denominator).clamp(0.0, 1.0))
    };
    let candidates = [
        Some(0.0),
        Some(1.0),
        ratio(offset.x, span.x),
        ratio(offset.y, span.y),
        ratio(offset.x - offset.y, span.x - span.y),
        ratio(offset.x + offset.y, span.x + span.y),
    ];
    candidates
        .into_iter()
        .flatten()
        .map(|amount| {
            let radius = radius_start + (radius_end - radius_start) * amount;
            let distance = if radius > 0.0 {
                (offset - span * amount).abs().max_element() / radius
            } else {
                f32::INFINITY
            };
            (amount, distance)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0.0, f32::INFINITY))
}

/// Discrete stamps placed every `spacing` diameters along the path.
fn draw_stamps(
    path: &StrokePath<'_>,
    pixels: &[Vec2],
    tile: &mut CoverageTile,
    resources: Option<&ResourceImages>,
) {
    let tip = path.tip;
    let step = (path.radius_pixels * 2.0 * tip.spacing).max(0.5);
    let mut stamps = 0_usize;
    let place = |index: usize,
                 center: Vec2,
                 direction: Vec2,
                 pressure: f32,
                 frame: StampFrame,
                 tile: &mut CoverageTile| {
        let index = index as u32;
        let random = |salt: i32| unit_float(hash3(index as i32, salt, 0, tip.seed));
        let size = 1.0 + (random(1) * 2.0 - 1.0) * tip.size_jitter;
        let radius = (path.radius_at(pressure) * size).max(0.25);
        let scatter_angle = random(2) * std::f32::consts::TAU;
        let scatter =
            Vec2::from_angle(scatter_angle) * random(3) * tip.scatter * path.radius_pixels * 2.0;
        let mut angle = tip.angle_degrees.to_radians()
            + (random(4) * 2.0 - 1.0) * tip.angle_jitter * std::f32::consts::PI;
        if tip.follow_path && direction.length_squared() > f32::EPSILON {
            // The path's heading in the stamp frame (V up), so the stamp's
            // x axis runs along it.
            let heading = frame.apply(Vec2::new(direction.x, -direction.y));
            angle += math::atan2(heading.y, heading.x);
        }
        let rotation = Vec2::from_angle(-angle);
        let center = center + scatter;
        let strength = path.strength_at(pressure);
        let stamp_seed = hash3(index as i32, 7, 0, tip.seed);
        let reach = radius * std::f32::consts::SQRT_2 * frame.stretch().max(1.0) + 1.0;
        for y in ((center.y - reach).floor() as i64).max(tile.min_y)
            ..=((center.y + reach).ceil() as i64).min(tile.min_y + tile.height - 1)
        {
            for x in ((center.x - reach).floor() as i64).max(tile.min_x)
                ..=((center.x + reach).ceil() as i64).min(tile.min_x + tile.width - 1)
            {
                let offset = (Vec2::new(x as f32, y as f32) - center) / radius;
                // Pixel rows run downward; flip so stamps keep their upright
                // orientation in UV space (or on the surface, when aligned).
                let local = rotation.rotate(frame.apply(Vec2::new(offset.x, -offset.y)));
                let coverage =
                    tip.stamp_coverage(local, 1.0 / radius, stamp_seed, resources) * strength;
                if coverage > 0.0 {
                    tile.deposit(x, y, coverage, direction);
                }
            }
        }
    };

    if pixels.len() == 1 {
        place(
            0,
            pixels[0],
            Vec2::ZERO,
            path.pressure(0),
            path.frame(0),
            tile,
        );
        return;
    }
    // Distance travelled since the last stamp; the first stamp sits on the
    // first point.
    let mut carried = step;
    for segment in 0..pixels.len() - 1 {
        let (start, end) = (pixels[segment], pixels[segment + 1]);
        let span = end - start;
        let length = span.length();
        if length <= f32::EPSILON {
            continue;
        }
        let direction = span / length;
        let mut travelled = step - carried;
        while travelled <= length {
            if stamps >= MAX_STAMPS {
                return;
            }
            let amount = travelled / length;
            let pressure = path.pressure(segment)
                + (path.pressure(segment + 1) - path.pressure(segment)) * amount;
            // A stamp takes the frame of the nearer point.
            let frame = path.frame(if amount < 0.5 { segment } else { segment + 1 });
            place(
                stamps,
                start + span * amount,
                direction,
                pressure,
                frame,
                tile,
            );
            stamps += 1;
            travelled += step;
        }
        carried = length - (travelled - step);
    }
    if stamps == 0 {
        place(
            0,
            pixels[0],
            Vec2::ZERO,
            path.pressure(0),
            path.frame(0),
            tile,
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::TextureImage;

    fn coverage(path: &StrokePath<'_>, size: u32, resources: Option<&ResourceImages>) -> Vec<f32> {
        let mut values = vec![0.0_f32; (size * size) as usize];
        let clip = TexelRect::full(size, size);
        rasterize_stroke(path, size, size, resources, clip, |index, value| {
            values[index] = values[index].max(value);
        });
        values
    }

    #[test]
    fn tc_0080_round_tip_keeps_linear_falloff_and_hardness_flattens_the_core() {
        let points = [Vec2::splat(0.5)];
        let soft = StrokePath {
            frames: Default::default(),
            points: &points,
            pressures: &[],
            radius_pixels: 8.0,
            tip: BrushTip::ROUND,
        };
        let values = coverage(&soft, 33, None);
        let at = |values: &[f32], x: usize, y: usize| values[y * 33 + x];
        assert_eq!(at(&values, 16, 16), 1.0);
        assert!((at(&values, 20, 16) - 0.5).abs() < 1e-5);
        assert_eq!(at(&values, 24, 16), 0.0);

        let hard = StrokePath {
            tip: BrushTip {
                hardness: 0.75,
                ..BrushTip::ROUND
            },
            ..soft
        };
        let values = coverage(&hard, 33, None);
        assert_eq!(at(&values, 20, 16), 1.0, "inside the hard core");
        assert!((at(&values, 23, 16) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn tc_0081_overlapping_segments_do_not_accumulate_within_one_stroke() {
        // Back-and-forth over the same texels.
        let points = [
            Vec2::new(0.3, 0.5),
            Vec2::new(0.7, 0.5),
            Vec2::new(0.3, 0.5),
        ];
        let path = StrokePath {
            frames: Default::default(),
            points: &points,
            pressures: &[],
            radius_pixels: 4.0,
            tip: BrushTip::ROUND,
        };
        let mut visits = std::collections::BTreeMap::new();
        rasterize_stroke(&path, 64, 64, None, TexelRect::full(64, 64), |index, _| {
            *visits.entry(index).or_insert(0) += 1;
        });
        assert!(visits.values().all(|count| *count == 1));
    }

    #[test]
    fn tc_0082_pressure_scales_radius_and_square_and_bitmap_tips_stamp_their_shape() {
        let points = [Vec2::new(0.2, 0.5), Vec2::new(0.8, 0.5)];
        let path = StrokePath {
            frames: Default::default(),
            points: &points,
            pressures: &[0.25, 1.0],
            radius_pixels: 8.0,
            tip: BrushTip::ROUND,
        };
        let values = coverage(&path, 65, None);
        let column_extent = |x: usize| (0..65).filter(|y| values[y * 65 + x] > 0.0).count();
        assert!(
            column_extent(14) < column_extent(50),
            "light pressure is thinner"
        );

        let square = StrokePath {
            frames: Default::default(),
            points: &points[..1],
            pressures: &[],
            radius_pixels: 6.0,
            tip: BrushTip {
                shape: BrushShape::Square,
                hardness: 1.0,
                ..BrushTip::ROUND
            },
        };
        let values = coverage(&square, 65, None);
        // Pixel (13, 32) is the stroke point; its square corner is covered.
        assert!(values[(32 + 5) * 65 + 13 + 5] > 0.0);

        // A bitmap whose left half is white and right half black.
        let image = TextureImage::new(2, 1, vec![255, 255, 255, 255, 0, 0, 0, 255]).unwrap();
        let resources = ResourceImages::from([(ResourceId(4), Arc::new(image))]);
        let bitmap = StrokePath {
            tip: BrushTip {
                shape: BrushShape::Bitmap(ResourceId(4)),
                ..BrushTip::ROUND
            },
            ..square
        };
        let values = coverage(&bitmap, 65, Some(&resources));
        assert!(values[32 * 65 + 13 - 4] > 0.9, "left half paints");
        assert!(values[32 * 65 + 13 + 4] < 0.1, "right half does not");
    }

    #[test]
    fn square_tips_sweep_diagonals_with_straight_antialiased_edges() {
        // Wide spacing used to leave a staircase a stamp step tall.
        let points = [Vec2::splat(0.2), Vec2::splat(0.8)];
        let path = StrokePath {
            frames: Default::default(),
            points: &points,
            pressures: &[],
            radius_pixels: 10.0,
            tip: BrushTip {
                shape: BrushShape::Square,
                hardness: 1.0,
                spacing: 0.5,
                ..BrushTip::ROUND
            },
        };
        let values = coverage(&path, 129, None);
        let at = |x: usize, y: usize| values[y * 129 + x];
        // The stroke runs up-right in pixels; across it, every step along it
        // reads the same, which a staircase would not.
        for across in -12_i32..=12 {
            let (x, y) = ((56 + across) as usize, (72 + across) as usize);
            for along in 1..=15 {
                assert!(
                    (at(x + along, y - along) - at(x, y)).abs() < 1e-4,
                    "{across} across, {along} along"
                );
            }
        }
        // The hard edge ramps over a pixel rather than stepping.
        assert!(values.iter().any(|value| (0.05..0.95).contains(value)));
        // The centre line is fully covered.
        assert_eq!(at(64, 64), 1.0);
    }

    #[test]
    fn tc_0291_streak_ring_and_cloud_alphas_have_their_shapes() {
        let tip = |shape| BrushTip {
            shape,
            hardness: 0.5,
            ..BrushTip::ROUND
        };
        let streak = tip(BrushShape::Streak);
        assert!(streak.stamp_coverage(Vec2::new(0.0, 0.0), 0.0, 0, None) > 0.9);
        assert!(
            streak.stamp_coverage(Vec2::new(0.6, 0.0), 0.0, 0, None) > 0.3,
            "long"
        );
        assert_eq!(
            streak.stamp_coverage(Vec2::new(0.0, 0.3), 0.0, 0, None),
            0.0,
            "thin"
        );
        assert_eq!(
            streak.stamp_coverage(Vec2::new(1.0, 0.0), 0.0, 0, None),
            0.0
        );
        let ring = tip(BrushShape::Ring);
        assert_eq!(
            ring.stamp_coverage(Vec2::new(0.0, 0.0), 0.0, 0, None),
            0.0,
            "hollow"
        );
        assert!(ring.stamp_coverage(Vec2::new(0.75, 0.0), 0.0, 0, None) > 0.9);
        assert!(ring.stamp_coverage(Vec2::new(0.0, -0.75), 0.0, 0, None) > 0.9);
        let cloud = tip(BrushShape::Cloud);
        let samples: Vec<f32> = (0..64)
            .map(|index| {
                let point = Vec2::new(
                    (index % 8) as f32 / 8.0 - 0.45,
                    (index / 8) as f32 / 8.0 - 0.45,
                );
                cloud.stamp_coverage(point, 0.0, 3, None)
            })
            .collect();
        assert!(samples.iter().any(|value| *value > 0.5), "dense parts");
        assert!(samples.iter().any(|value| *value < 0.1), "gaps");
        assert_eq!(
            samples,
            (0..64)
                .map(|index| {
                    let point = Vec2::new(
                        (index % 8) as f32 / 8.0 - 0.45,
                        (index / 8) as f32 / 8.0 - 0.45,
                    );
                    cloud.stamp_coverage(point, 0.0, 3, None)
                })
                .collect::<Vec<_>>(),
            "deterministic"
        );
        // The new shapes paint through the stamp path.
        for shape in [BrushShape::Streak, BrushShape::Ring, BrushShape::Cloud] {
            let path = StrokePath {
                frames: Default::default(),
                points: &[Vec2::splat(0.5)],
                pressures: &[],
                radius_pixels: 10.0,
                tip: tip(shape),
            };
            assert!(
                coverage(&path, 32, None).iter().any(|value| *value > 0.5),
                "{shape:?}"
            );
        }
    }

    /// Texels with coverage along row `y` and along column `x`.
    fn extents(values: &[f32], size: usize, x: usize, y: usize) -> (usize, usize) {
        let row = (0..size).filter(|i| values[y * size + i] > 0.0).count();
        let column = (0..size).filter(|i| values[i * size + x] > 0.0).count();
        (row, column)
    }

    #[test]
    fn tc_0352_surface_aligned_stamps_take_the_shape_their_frame_gives() {
        let point = [Vec2::splat(0.5)];
        let plain = StrokePath {
            points: &point,
            pressures: &[],
            radius_pixels: 8.0,
            tip: BrushTip {
                hardness: 1.0,
                ..BrushTip::ROUND
            },
            frames: &[],
        };
        let (row, column) = extents(&coverage(&plain, 65, None), 65, 32, 32);
        assert_eq!(row, column, "round in texture space");
        // A frame that doubles x: the surface is squeezed along U, so the
        // stamp covers half the texels across and twice as many down.
        let squeeze = [StampFrame([2.0, 0.0, 0.0, 0.5])];
        assert!((squeeze[0].stretch() - 2.0).abs() < 1e-6);
        assert!(squeeze[0].is_valid());
        let aligned = StrokePath {
            frames: &squeeze,
            ..plain
        };
        let values = coverage(&aligned, 65, None);
        let (row, column) = extents(&values, 65, 32, 32);
        assert!((7..=9).contains(&row), "across {row}");
        assert!((31..=33).contains(&column), "down {column}");
        // The texel bounds reach as far as the stretched stamp.
        let bounds = stroke_texel_bounds(&point, 8.0, plain.tip, &squeeze, 65, 65).unwrap();
        let painted = values.iter().enumerate().filter(|(_, value)| **value > 0.0);
        for (index, _) in painted {
            let (x, y) = ((index % 65) as u32, (index / 65) as u32);
            assert!(bounds.contains(x, y), "({x}, {y}) outside {bounds:?}");
        }
        // Frames must keep unit area, stay within the stretch limit, and
        // come one per point.
        assert!(!StampFrame([2.0, 0.0, 0.0, 2.0]).is_valid());
        assert!(!StampFrame([10.0, 0.0, 0.0, 0.1]).is_valid());
        assert!(valid_frames(&point, &squeeze));
        assert!(!valid_frames(&[Vec2::ZERO, Vec2::ONE], &squeeze));
    }

    #[test]
    fn tc_0353_follow_path_stamps_run_along_diagonal_strokes() {
        // A thin streak along a 45-degree stroke, spaced wide so stamps
        // stand apart: each must lie along the diagonal.
        let points = [Vec2::new(0.2, 0.8), Vec2::new(0.8, 0.2)];
        let path = StrokePath {
            points: &points,
            pressures: &[],
            radius_pixels: 10.0,
            tip: BrushTip {
                shape: BrushShape::Streak,
                hardness: 1.0,
                spacing: 3.0,
                follow_path: true,
                ..BrushTip::ROUND
            },
            frames: &[],
        };
        let values = coverage(&path, 101, None);
        // The first stamp sits on pixel (20, 20); the diagonal through it
        // is painted, the anti-diagonal is not.
        assert!(values[(20 + 3) * 101 + 20 + 3] > 0.0, "along the path");
        assert_eq!(values[(20 - 3) * 101 + 20 + 3], 0.0, "across the path");
    }

    #[test]
    fn tc_0083_stamps_follow_spacing_and_jitter_is_deterministic() {
        let points = [Vec2::new(0.1, 0.5), Vec2::new(0.9, 0.5)];
        let path = StrokePath {
            frames: Default::default(),
            points: &points,
            pressures: &[],
            radius_pixels: 3.0,
            tip: BrushTip {
                shape: BrushShape::Round,
                hardness: 1.0,
                spacing: 2.0,
                scatter: 0.0,
                size_jitter: 0.5,
                seed: 9,
                ..BrushTip::ROUND
            },
        };
        let first = coverage(&path, 101, None);
        assert_eq!(first, coverage(&path, 101, None));
        // Wide spacing leaves gaps between stamps along the row.
        let row = &first[50 * 101..51 * 101];
        assert!(row[10..90].contains(&0.0));
        assert!(row[10..90].iter().any(|value| *value > 0.0));
    }
}
