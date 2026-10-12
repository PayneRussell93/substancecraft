//! Nondestructive filters applied to a layer's content or to its mask.

use glam::{Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::region::TexelRect;

/// Blur radii are given in texels of a 1024-wide texture and scale with the
/// texture set's resolution, so a filter keeps its look when resolution
/// changes.
const REFERENCE_WIDTH: f32 = 1024.0;
pub const MAX_BLUR_RADIUS: f32 = 64.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerFilter {
    Blur {
        radius: f32,
    },
    /// Unsharp mask: adds back the difference from a blurred copy.
    Sharpen {
        radius: f32,
        amount: f32,
    },
    /// Remaps `in_low..in_high` to `out_low..out_high` with a gamma curve.
    Levels {
        in_low: f32,
        in_high: f32,
        gamma: f32,
        out_low: f32,
        out_high: f32,
    },
    Invert,
    /// Colour adjustment; grayscale data only uses `lightness`.
    HueSaturation {
        /// Hue rotation in turns, `-0.5..=0.5`.
        hue: f32,
        /// Saturation multiplier, `0..=2`.
        saturation: f32,
        /// Added lightness, `-1..=1`.
        lightness: f32,
    },
    /// Adds `brightness` and scales the distance from mid-grey by
    /// `1 + contrast` (both `-1..=1`).
    BrightnessContrast {
        brightness: f32,
        contrast: f32,
    },
    /// Rounds values to `levels` evenly spaced steps (`2..=64`).
    Posterize {
        levels: u32,
    },
    /// Black below `level`, white above, over a `softness` wide ramp.
    Threshold {
        level: f32,
        softness: f32,
    },
    /// Maps luminance to a ramp from `dark` to `light` (sRGB colours):
    /// tints grayscale detail, or grays colour by the ramp's luminance.
    GradientMap {
        dark: [u8; 3],
        light: [u8; 3],
    },
}

impl LayerFilter {
    pub const PRESETS: [Self; 9] = [
        Self::Blur { radius: 4.0 },
        Self::Sharpen {
            radius: 2.0,
            amount: 1.0,
        },
        Self::Levels {
            in_low: 0.0,
            in_high: 1.0,
            gamma: 1.0,
            out_low: 0.0,
            out_high: 1.0,
        },
        Self::Invert,
        Self::HueSaturation {
            hue: 0.0,
            saturation: 1.0,
            lightness: 0.0,
        },
        Self::BrightnessContrast {
            brightness: 0.0,
            contrast: 0.0,
        },
        Self::Posterize { levels: 4 },
        Self::Threshold {
            level: 0.5,
            softness: 0.05,
        },
        Self::GradientMap {
            dark: [40, 30, 20],
            light: [230, 210, 170],
        },
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Blur { .. } => "Blur",
            Self::Sharpen { .. } => "Sharpen",
            Self::Levels { .. } => "Levels",
            Self::Invert => "Invert",
            Self::HueSaturation { .. } => "Hue / Saturation",
            Self::BrightnessContrast { .. } => "Brightness / Contrast",
            Self::Posterize { .. } => "Posterize",
            Self::Threshold { .. } => "Threshold",
            Self::GradientMap { .. } => "Gradient map",
        }
    }

    pub(crate) fn is_valid(self) -> bool {
        let finite_in =
            |value: f32, low: f32, high: f32| value.is_finite() && (low..=high).contains(&value);
        match self {
            Self::Blur { radius } => finite_in(radius, 0.0, MAX_BLUR_RADIUS),
            Self::Sharpen { radius, amount } => {
                finite_in(radius, 0.0, MAX_BLUR_RADIUS) && finite_in(amount, 0.0, 4.0)
            }
            Self::Levels {
                in_low,
                in_high,
                gamma,
                out_low,
                out_high,
            } => {
                finite_in(in_low, 0.0, 1.0)
                    && finite_in(in_high, 0.0, 1.0)
                    && finite_in(gamma, 0.05, 10.0)
                    && finite_in(out_low, 0.0, 1.0)
                    && finite_in(out_high, 0.0, 1.0)
            }
            Self::Invert => true,
            Self::HueSaturation {
                hue,
                saturation,
                lightness,
            } => {
                finite_in(hue, -0.5, 0.5)
                    && finite_in(saturation, 0.0, 2.0)
                    && finite_in(lightness, -1.0, 1.0)
            }
            Self::BrightnessContrast {
                brightness,
                contrast,
            } => finite_in(brightness, -1.0, 1.0) && finite_in(contrast, -1.0, 1.0),
            Self::Posterize { levels } => (2..=64).contains(&levels),
            Self::Threshold { level, softness } => {
                finite_in(level, 0.0, 1.0) && finite_in(softness, 0.0, 1.0)
            }
            Self::GradientMap { .. } => true,
        }
    }

    fn map_value(self, value: f32) -> f32 {
        match self {
            Self::Levels {
                in_low,
                in_high,
                gamma,
                out_low,
                out_high,
            } => {
                let span = in_high - in_low;
                let normalized = if span.abs() < 1e-4 {
                    if value >= in_high { 1.0 } else { 0.0 }
                } else {
                    ((value - in_low) / span).clamp(0.0, 1.0)
                };
                out_low + (out_high - out_low) * normalized.powf(1.0 / gamma)
            }
            Self::Invert => 1.0 - value,
            Self::HueSaturation { lightness, .. } => (value + lightness).clamp(0.0, 1.0),
            Self::BrightnessContrast {
                brightness,
                contrast,
            } => ((value - 0.5) * (1.0 + contrast) + 0.5 + brightness).clamp(0.0, 1.0),
            Self::Posterize { levels } => {
                let steps = (levels.max(2) - 1) as f32;
                (value.clamp(0.0, 1.0) * steps).round() / steps
            }
            Self::Threshold { level, softness } => {
                if softness <= 0.0 {
                    f32::from(value >= level)
                } else {
                    ((value - (level - softness * 0.5)) / softness).clamp(0.0, 1.0)
                }
            }
            Self::GradientMap { dark, light } => {
                let ramp = gradient_color(dark, light, value);
                luminance(ramp)
            }
            _ => value,
        }
    }

    fn map_color(self, color: Vec3) -> Vec3 {
        match self {
            Self::HueSaturation {
                hue,
                saturation,
                lightness,
            } => {
                let (h, s, l) = rgb_to_hsl(color);
                hsl_to_rgb(
                    (h + hue).rem_euclid(1.0),
                    (s * saturation).clamp(0.0, 1.0),
                    (l + lightness).clamp(0.0, 1.0),
                )
            }
            Self::GradientMap { dark, light } => gradient_color(dark, light, luminance(color)),
            other => Vec3::new(
                other.map_value(color.x),
                other.map_value(color.y),
                other.map_value(color.z),
            ),
        }
    }
}

/// Rec. 709 luminance of a colour.
fn luminance(color: Vec3) -> f32 {
    color.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// The gradient map's colour at `amount` (`0..=1`) between two sRGB
/// colours.
fn gradient_color(dark: [u8; 3], light: [u8; 3], amount: f32) -> Vec3 {
    let unit = |color: [u8; 3]| Vec3::from(color.map(|channel| f32::from(channel) / 255.0));
    unit(dark).lerp(unit(light), amount.clamp(0.0, 1.0))
}

/// What the texels of a plane hold, which decides how filters treat them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlaneData {
    Color,
    Scalar,
    /// Unit vectors; only blurs apply, and results are renormalised.
    Vector,
}

/// Which texels a plane buffer holds: `region` (row-major) of a `width` ×
/// `height` texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneGrid {
    pub region: TexelRect,
    pub width: u32,
    pub height: u32,
}

#[cfg(test)]
impl PlaneGrid {
    const fn full(width: u32, height: u32) -> Self {
        Self {
            region: TexelRect::full(width, height),
            width,
            height,
        }
    }
}

/// Texels a region bake must add around a region so `filters` are exact
/// inside it: spatial filters read neighbours, and the blur's running sums
/// restart only at fixed block boundaries.
pub(crate) fn spatial_halo(filters: &[LayerFilter], texture_width: u32) -> u32 {
    filters
        .iter()
        .map(|filter| match *filter {
            LayerFilter::Blur { radius } | LayerFilter::Sharpen { radius, .. } => {
                let radius = blur_texels(scaled_radius(radius, texture_width));
                if radius == 0 {
                    0
                } else {
                    // Three box passes, each reaching `radius` beyond the
                    // window plus a partial block before it.
                    (3 * (radius + blur_block(radius))) as u32
                }
            }
            _ => 0,
        })
        .sum()
}

/// Applies `filters` in order to straight-alpha texels (`xyz` value, `w`
/// alpha) laid out as `grid`. On a partial region, spatial filters are
/// exact only at least [`spatial_halo`] texels inside its edges (or at the
/// texture's own edges).
pub(crate) fn apply_to_plane(
    texels: &mut [Vec4],
    grid: PlaneGrid,
    data: PlaneData,
    filters: &[LayerFilter],
) {
    for filter in filters {
        match *filter {
            LayerFilter::Blur { radius } => {
                blur_plane(texels, grid, scaled_radius(radius, grid.width));
            }
            LayerFilter::Sharpen { radius, amount } => {
                let mut blurred = texels.to_vec();
                blur_plane(&mut blurred, grid, scaled_radius(radius, grid.width));
                for (texel, soft) in texels.iter_mut().zip(blurred) {
                    let value = texel.truncate() + (texel.truncate() - soft.truncate()) * amount;
                    *texel = match data {
                        PlaneData::Vector => value.normalize_or(Vec3::Z).extend(texel.w),
                        _ => value.clamp(Vec3::ZERO, Vec3::ONE).extend(texel.w),
                    };
                }
            }
            other => {
                for texel in texels.iter_mut() {
                    let value = match data {
                        PlaneData::Color => other.map_color(texel.truncate()),
                        PlaneData::Scalar => Vec3::splat(other.map_value(texel.x)),
                        PlaneData::Vector => continue,
                    };
                    *texel = value.extend(texel.w);
                }
            }
        }
    }
    if data == PlaneData::Vector {
        for texel in texels {
            *texel = texel.truncate().normalize_or(Vec3::Z).extend(texel.w);
        }
    }
}

/// Applies `filters` to grayscale mask values.
pub(crate) fn apply_to_mask(values: &mut [f32], grid: PlaneGrid, filters: &[LayerFilter]) {
    if filters.is_empty() {
        return;
    }
    let mut texels = values
        .iter()
        .map(|value| Vec4::new(*value, *value, *value, 1.0))
        .collect::<Vec<_>>();
    apply_to_plane(&mut texels, grid, PlaneData::Scalar, filters);
    for (value, texel) in values.iter_mut().zip(texels) {
        *value = texel.x.clamp(0.0, 1.0);
    }
}

fn scaled_radius(radius: f32, width: u32) -> f32 {
    radius * width as f32 / REFERENCE_WIDTH
}

/// Box radius in texels for a scaled blur radius.
fn blur_texels(radius: f32) -> usize {
    radius.round() as usize
}

/// Length of the blocks a box pass restarts its running sum at: long
/// enough that restarts cost little, fixed per radius so any region of a
/// texture sums each texel in the same order.
fn blur_block(radius: usize) -> usize {
    (radius * 4).max(16)
}

/// Separable box blur run three times (close to Gaussian) on premultiplied
/// values, so transparent texels do not darken their neighbours.
fn blur_plane(texels: &mut [Vec4], grid: PlaneGrid, radius: f32) {
    let radius = blur_texels(radius);
    let region = grid.region;
    if radius == 0 || region.is_empty() {
        return;
    }
    let (width, height) = (region.width as usize, region.height as usize);
    let mut premultiplied = texels
        .iter()
        .map(|texel| (texel.truncate() * texel.w).extend(texel.w))
        .collect::<Vec<_>>();
    let mut scratch = vec![Vec4::ZERO; width.max(height)];
    let rows = BlurLine {
        start: region.x as usize,
        extent: grid.width as usize,
        radius,
    };
    let columns = BlurLine {
        start: region.y as usize,
        extent: grid.height as usize,
        radius,
    };
    for _ in 0..3 {
        for row in premultiplied.chunks_mut(width) {
            rows.blur(row, &mut scratch);
        }
        let mut column = vec![Vec4::ZERO; height];
        for x in 0..width {
            for (y, slot) in column.iter_mut().enumerate() {
                *slot = premultiplied[y * width + x];
            }
            columns.blur(&mut column, &mut scratch);
            for (y, value) in column.iter().enumerate() {
                premultiplied[y * width + x] = *value;
            }
        }
    }
    for (texel, blurred) in texels.iter_mut().zip(premultiplied) {
        *texel = if blurred.w > 1e-6 {
            (blurred.truncate() / blurred.w).extend(blurred.w)
        } else {
            Vec4::ZERO
        };
    }
}

/// A row or column of a texture being box blurred: the line buffer holds
/// texels `start..` of a line `extent` texels long.
struct BlurLine {
    start: usize,
    extent: usize,
    radius: usize,
}

impl BlurLine {
    /// Running-sum box blur with edges clamped to the texture's. Reads past
    /// the buffer's ends (in a partial region) clamp to them; those texels
    /// lie in the halo a region bake discards.
    fn blur(&self, line: &mut [Vec4], scratch: &mut [Vec4]) {
        let length = line.len();
        if length == 0 || self.extent == 0 {
            return;
        }
        let radius = self.radius as isize;
        let window = (self.radius * 2 + 1) as f32;
        let last = self.extent as isize - 1;
        let at = |texel: isize| {
            let local = texel.clamp(0, last) - self.start as isize;
            line[local.clamp(0, length as isize - 1) as usize]
        };
        let block = blur_block(self.radius);
        let mut index = 0;
        while index < length {
            // The sum restarts at every block boundary of the texture.
            let texel = self.start + index;
            let end = ((texel / block + 1) * block - self.start).min(length);
            let mut sum = Vec4::ZERO;
            for offset in -radius..=radius {
                sum += at(texel as isize + offset);
            }
            for (local, slot) in scratch.iter_mut().enumerate().take(end).skip(index) {
                *slot = sum / window;
                let texel = (self.start + local) as isize;
                sum += at(texel + radius + 1) - at(texel - radius);
            }
            index = end;
        }
        line.copy_from_slice(&scratch[..length]);
    }
}

fn rgb_to_hsl(color: Vec3) -> (f32, f32, f32) {
    let max = color.max_element();
    let min = color.min_element();
    let lightness = (max + min) * 0.5;
    let delta = max - min;
    if delta <= 1e-6 {
        return (0.0, 0.0, lightness);
    }
    let saturation = delta / (1.0 - (2.0 * lightness - 1.0).abs()).max(1e-6);
    let hue = if max == color.x {
        ((color.y - color.z) / delta).rem_euclid(6.0)
    } else if max == color.y {
        (color.z - color.x) / delta + 2.0
    } else {
        (color.x - color.y) / delta + 4.0
    } / 6.0;
    (hue, saturation.clamp(0.0, 1.0), lightness)
}

fn hsl_to_rgb(hue: f32, saturation: f32, lightness: f32) -> Vec3 {
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    let sector = hue * 6.0;
    let secondary = chroma * (1.0 - (sector.rem_euclid(2.0) - 1.0).abs());
    let (r, g, b) = match sector as u32 {
        0 => (chroma, secondary, 0.0),
        1 => (secondary, chroma, 0.0),
        2 => (0.0, chroma, secondary),
        3 => (0.0, secondary, chroma),
        4 => (secondary, 0.0, chroma),
        _ => (chroma, 0.0, secondary),
    };
    let offset = lightness - chroma * 0.5;
    (Vec3::new(r, g, b) + Vec3::splat(offset)).clamp(Vec3::ZERO, Vec3::ONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tc_0063_blur_spreads_values_without_darkening_into_transparency() {
        let (width, height) = (32_u32, 1_u32);
        let mut texels = vec![Vec4::ZERO; 32];
        texels[16] = Vec4::new(1.0, 0.5, 0.25, 1.0);
        apply_to_plane(
            &mut texels,
            PlaneGrid::full(width, height),
            PlaneData::Color,
            // 2 texels at 32 wide is radius 64 at the 1024 reference.
            &[LayerFilter::Blur { radius: 64.0 }],
        );
        assert!(texels[14].w > 0.0 && texels[14].w < 1.0, "alpha spreads");
        // Colour is unpremultiplied, so it keeps its hue at the fringe.
        assert!((texels[14].x - 1.0).abs() < 1e-4);
        assert!((texels[14].y - 0.5).abs() < 1e-4);
    }

    #[test]
    fn region_blurs_match_the_full_blur_inside_their_halo() {
        let (width, height) = (300_u32, 220_u32);
        let texels: Vec<Vec4> = (0..width * height)
            .map(|index| {
                let value = ((index.wrapping_mul(2_654_435_761) >> 8) % 1000) as f32 / 1000.0;
                Vec4::new(value, 1.0 - value, value * 0.5, (index % 7) as f32 / 6.0)
            })
            .collect();
        let filters = [
            LayerFilter::Blur { radius: 9.0 },
            LayerFilter::Sharpen {
                radius: 5.0,
                amount: 0.7,
            },
        ];
        let mut full = texels.clone();
        apply_to_plane(
            &mut full,
            PlaneGrid::full(width, height),
            PlaneData::Color,
            &filters,
        );
        let halo = spatial_halo(&filters, width);
        assert!(halo > 0);
        for inner in [
            TexelRect {
                x: 140,
                y: 100,
                width: 5,
                height: 3,
            },
            TexelRect {
                x: 0,
                y: 0,
                width: 9,
                height: 9,
            },
            TexelRect {
                x: 287,
                y: 211,
                width: 13,
                height: 9,
            },
        ] {
            let region = inner.grow(halo, width, height);
            let mut part: Vec<Vec4> = (region.y..region.bottom())
                .flat_map(|y| {
                    let start = (y * width + region.x) as usize;
                    texels[start..start + region.width as usize].to_vec()
                })
                .collect();
            let grid = PlaneGrid {
                region,
                width,
                height,
            };
            apply_to_plane(&mut part, grid, PlaneData::Color, &filters);
            for y in inner.y..inner.bottom() {
                for x in inner.x..inner.right() {
                    assert_eq!(
                        part[region.local_index(x, y)],
                        full[(y * width + x) as usize],
                        "texel {x},{y} of {inner:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn tc_0064_levels_invert_and_hue_saturation_remap_values() {
        let levels = LayerFilter::Levels {
            in_low: 0.25,
            in_high: 0.75,
            gamma: 1.0,
            out_low: 0.0,
            out_high: 1.0,
        };
        assert!((levels.map_value(0.5) - 0.5).abs() < 1e-6);
        assert_eq!(levels.map_value(0.2), 0.0);
        assert_eq!(levels.map_value(0.9), 1.0);
        assert_eq!(LayerFilter::Invert.map_value(0.25), 0.75);

        let desaturate = LayerFilter::HueSaturation {
            hue: 0.0,
            saturation: 0.0,
            lightness: 0.0,
        };
        let gray = desaturate.map_color(Vec3::new(1.0, 0.0, 0.0));
        assert!((gray.x - gray.y).abs() < 1e-5 && (gray.y - gray.z).abs() < 1e-5);
        let shift = LayerFilter::HueSaturation {
            hue: 1.0 / 3.0,
            saturation: 1.0,
            lightness: 0.0,
        };
        let green = shift.map_color(Vec3::new(1.0, 0.0, 0.0));
        assert!((green - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-4);

        let mut mask = vec![0.2, 0.8];
        apply_to_mask(&mut mask, PlaneGrid::full(2, 1), &[LayerFilter::Invert]);
        assert!((mask[0] - 0.8).abs() < 1e-6 && (mask[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn tc_0299_tone_filters_map_values_and_gradients_tint() {
        let close = |a: f32, b: f32| (a - b).abs() < 1e-5;
        let contrast = LayerFilter::BrightnessContrast {
            brightness: 0.1,
            contrast: 1.0,
        };
        assert!(close(contrast.map_value(0.5), 0.6));
        assert!(close(contrast.map_value(0.75), 1.0));
        assert!(close(contrast.map_value(0.3), 0.2));
        let posterize = LayerFilter::Posterize { levels: 3 };
        assert_eq!(posterize.map_value(0.2), 0.0);
        assert_eq!(posterize.map_value(0.3), 0.5);
        assert_eq!(posterize.map_value(0.9), 1.0);
        let hard = LayerFilter::Threshold {
            level: 0.5,
            softness: 0.0,
        };
        assert_eq!((hard.map_value(0.49), hard.map_value(0.5)), (0.0, 1.0));
        let soft = LayerFilter::Threshold {
            level: 0.5,
            softness: 0.2,
        };
        assert!(close(soft.map_value(0.5), 0.5));
        assert_eq!(soft.map_value(0.35), 0.0);
        let map = LayerFilter::GradientMap {
            dark: [0, 0, 255],
            light: [255, 0, 0],
        };
        // Grey in, the ramp's middle out; black and white hit the ends.
        let middle = map.map_color(Vec3::splat(0.5));
        assert!(close(middle.x, 0.5) && close(middle.z, 0.5));
        assert_eq!(map.map_color(Vec3::ZERO), Vec3::Z);
        assert_eq!(map.map_color(Vec3::ONE), Vec3::X);
        // On scalar data the ramp's luminance is used.
        assert!(close(map.map_value(1.0), 0.2126));
        // Validation and presets.
        assert!(!LayerFilter::Posterize { levels: 1 }.is_valid());
        assert!(
            !LayerFilter::BrightnessContrast {
                brightness: 2.0,
                contrast: 0.0
            }
            .is_valid()
        );
        assert!(LayerFilter::PRESETS.iter().all(|filter| filter.is_valid()));
        assert_eq!(LayerFilter::PRESETS.len(), 9);
        // Per-texel filters need no region halo.
        assert_eq!(spatial_halo(&LayerFilter::PRESETS[3..], 2048), 0);
    }
}
