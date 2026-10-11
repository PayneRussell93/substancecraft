//! Brush tips and stroke rasterisation.
//!
//! A stroke's footprint is accumulated with `max`, so overlapping segments
//! and stamps of one stroke never build up beyond the stroke's own opacity.

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::noise::{gradient_noise3, hash3, unit_float};
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
    Square,
    /// Clustered speckles, for grime and chipped paint.
    Splatter,
    /// Grayscale bitmap resource: white paints, black does not.
    Bitmap(ResourceId),
}

impl BrushShape {
    pub const BUILT_IN: [Self; 3] = [Self::Round, Self::Square, Self::Splatter];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Round => "Round",
            Self::Square => "Square",
            Self::Splatter => "Splatter",
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

    /// Continuous capsules reproduce the classic smooth round stroke; any
    /// other shape or randomness needs discrete stamps.
    fn continuous(self) -> bool {
        self.shape == BrushShape::Round
            && self.size_jitter == 0.0
            && self.angle_jitter == 0.0
            && self.scatter == 0.0
    }

    fn falloff(self, distance: f32) -> f32 {
        if distance >= 1.0 {
            return 0.0;
        }
        let hardness = self.hardness.clamp(0.0, 0.999);
        if distance <= hardness {
            1.0
        } else {
            (1.0 - distance) / (1.0 - hardness)
        }
    }

    /// Coverage of one stamp at `local`, the offset from its centre divided
    /// by its radius and rotated into the stamp frame.
    fn stamp_coverage(
        self,
        local: Vec2,
        stamp_seed: u32,
        resources: Option<&ResourceImages>,
    ) -> f32 {
        match self.shape {
            BrushShape::Round => self.falloff(local.length()),
            BrushShape::Square => self.falloff(local.x.abs().max(local.y.abs())),
            BrushShape::Splatter => {
                let disc = self.falloff(local.length());
                if disc <= 0.0 {
                    return 0.0;
                }
                let speckle =
                    gradient_noise3(Vec3::new(local.x * 3.5, local.y * 3.5, 0.0), stamp_seed) * 0.5
                        + 0.5;
                disc * ((speckle - 0.52) * 8.0).clamp(0.0, 1.0)
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

/// Geometry of one stroke in UV space.
pub(crate) struct StrokePath<'a> {
    pub points: &'a [Vec2],
    /// Per-point pen pressure; empty means full pressure everywhere.
    pub pressures: &'a [f32],
    pub radius_pixels: f32,
    pub tip: BrushTip,
}

impl StrokePath<'_> {
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
struct CoverageTile {
    min_x: i64,
    min_y: i64,
    width: i64,
    height: i64,
    values: Vec<f32>,
}

impl CoverageTile {
    fn deposit(&mut self, x: i64, y: i64, value: f32) {
        let (local_x, local_y) = (x - self.min_x, y - self.min_y);
        if local_x < 0 || local_y < 0 || local_x >= self.width || local_y >= self.height {
            return;
        }
        let slot = &mut self.values[(local_y * self.width + local_x) as usize];
        *slot = slot.max(value);
    }
}

/// Visits every texel the stroke touches with its coverage in `0..=1`.
pub(crate) fn rasterize_stroke(
    path: &StrokePath<'_>,
    width: u32,
    height: u32,
    resources: Option<&ResourceImages>,
    mut apply: impl FnMut(usize, f32),
) {
    if path.points.is_empty() || width == 0 || height == 0 {
        return;
    }
    let pixels = path
        .points
        .iter()
        .map(|point| raster::uv_to_pixel(*point, width, height))
        .collect::<Vec<_>>();
    let tip = path.tip;
    let reach = path.radius_pixels * (1.0 + tip.size_jitter) * std::f32::consts::SQRT_2
        + path.radius_pixels * 2.0 * tip.scatter
        + 1.0;
    let (mut lower, mut upper) = (pixels[0], pixels[0]);
    for pixel in &pixels {
        lower = lower.min(*pixel);
        upper = upper.max(*pixel);
    }
    let min_x = ((lower.x - reach).floor() as i64).max(0);
    let min_y = ((lower.y - reach).floor() as i64).max(0);
    let max_x = ((upper.x + reach).ceil() as i64).min(i64::from(width) - 1);
    let max_y = ((upper.y + reach).ceil() as i64).min(i64::from(height) - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }
    let mut tile = CoverageTile {
        min_x,
        min_y,
        width: max_x - min_x + 1,
        height: max_y - min_y + 1,
        values: Vec::new(),
    };
    let count = (tile.width * tile.height) as usize;
    if tile.values.try_reserve_exact(count).is_err() {
        return;
    }
    tile.values.resize(count, 0.0);

    if tip.continuous() {
        draw_capsules(path, &pixels, &mut tile);
    } else {
        draw_stamps(path, &pixels, &mut tile, resources);
    }

    let row = width as usize;
    for local_y in 0..tile.height {
        for local_x in 0..tile.width {
            let coverage = tile.values[(local_y * tile.width + local_x) as usize];
            if coverage > 0.0 {
                let index = (tile.min_y + local_y) as usize * row + (tile.min_x + local_x) as usize;
                apply(index, coverage.min(1.0));
            }
        }
    }
}

/// Smooth round stroke: capsules between consecutive points whose radius
/// and strength follow the interpolated pressure.
fn draw_capsules(path: &StrokePath<'_>, pixels: &[Vec2], tile: &mut CoverageTile) {
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
        let reach = radius_start.max(radius_end);
        let lower = start.min(end) - Vec2::splat(reach);
        let upper = start.max(end) + Vec2::splat(reach);
        let span = end - start;
        let length_squared = span.length_squared();
        for y in (lower.y.floor() as i64).max(tile.min_y)
            ..=(upper.y.ceil() as i64).min(tile.min_y + tile.height - 1)
        {
            for x in (lower.x.floor() as i64).max(tile.min_x)
                ..=(upper.x.ceil() as i64).min(tile.min_x + tile.width - 1)
            {
                let pixel = Vec2::new(x as f32, y as f32);
                let amount = if length_squared > f32::EPSILON {
                    ((pixel - start).dot(span) / length_squared).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let radius = radius_start + (radius_end - radius_start) * amount;
                if radius <= 0.0 {
                    continue;
                }
                let distance = pixel.distance(start + span * amount) / radius;
                let strength =
                    path.strength_at(pressure_start + (pressure_end - pressure_start) * amount);
                let coverage = path.tip.falloff(distance) * strength;
                if coverage > 0.0 {
                    tile.deposit(x, y, coverage);
                }
            }
        }
    }
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
            angle += direction.y.atan2(direction.x);
        }
        let rotation = Vec2::from_angle(-angle);
        let center = center + scatter;
        let strength = path.strength_at(pressure);
        let stamp_seed = hash3(index as i32, 7, 0, tip.seed);
        let reach = radius * std::f32::consts::SQRT_2;
        for y in ((center.y - reach).floor() as i64).max(tile.min_y)
            ..=((center.y + reach).ceil() as i64).min(tile.min_y + tile.height - 1)
        {
            for x in ((center.x - reach).floor() as i64).max(tile.min_x)
                ..=((center.x + reach).ceil() as i64).min(tile.min_x + tile.width - 1)
            {
                let offset = (Vec2::new(x as f32, y as f32) - center) / radius;
                // Pixel rows run downward; flip so stamps keep their upright
                // orientation in UV space.
                let local = rotation.rotate(Vec2::new(offset.x, -offset.y));
                let coverage = tip.stamp_coverage(local, stamp_seed, resources) * strength;
                if coverage > 0.0 {
                    tile.deposit(x, y, coverage);
                }
            }
        }
    };

    if pixels.len() == 1 {
        place(0, pixels[0], Vec2::ZERO, path.pressure(0), tile);
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
            place(stamps, start + span * amount, direction, pressure, tile);
            stamps += 1;
            travelled += step;
        }
        carried = length - (travelled - step);
    }
    if stamps == 0 {
        place(0, pixels[0], Vec2::ZERO, path.pressure(0), tile);
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
        rasterize_stroke(path, size, size, resources, |index, value| {
            values[index] = values[index].max(value);
        });
        values
    }

    #[test]
    fn tc_0080_round_tip_keeps_linear_falloff_and_hardness_flattens_the_core() {
        let points = [Vec2::splat(0.5)];
        let soft = StrokePath {
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
            points: &points,
            pressures: &[],
            radius_pixels: 4.0,
            tip: BrushTip::ROUND,
        };
        let mut visits = std::collections::BTreeMap::new();
        rasterize_stroke(&path, 64, 64, None, |index, _| {
            *visits.entry(index).or_insert(0) += 1;
        });
        assert!(visits.values().all(|count| *count == 1));
    }

    #[test]
    fn tc_0082_pressure_scales_radius_and_square_and_bitmap_tips_stamp_their_shape() {
        let points = [Vec2::new(0.2, 0.5), Vec2::new(0.8, 0.5)];
        let path = StrokePath {
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
    fn tc_0083_stamps_follow_spacing_and_jitter_is_deterministic() {
        let points = [Vec2::new(0.1, 0.5), Vec2::new(0.9, 0.5)];
        let path = StrokePath {
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
        assert!(row[10..90].iter().any(|value| *value == 0.0));
        assert!(row[10..90].iter().any(|value| *value > 0.0));
    }
}
