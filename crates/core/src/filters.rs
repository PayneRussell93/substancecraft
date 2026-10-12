//! Nondestructive filters applied to a layer's content or to its mask.

use glam::{Vec3, Vec4};
use serde::{Deserialize, Serialize};

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
}

impl LayerFilter {
    pub const PRESETS: [Self; 5] = [
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
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Blur { .. } => "Blur",
            Self::Sharpen { .. } => "Sharpen",
            Self::Levels { .. } => "Levels",
            Self::Invert => "Invert",
            Self::HueSaturation { .. } => "Hue / Saturation",
        }
    }

    /// True when each output texel depends only on the same input texel
    /// (blur and sharpen read their neighbours).
    pub const fn is_per_texel(self) -> bool {
        !matches!(self, Self::Blur { .. } | Self::Sharpen { .. })
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
            other => Vec3::new(
                other.map_value(color.x),
                other.map_value(color.y),
                other.map_value(color.z),
            ),
        }
    }
}

/// What the texels of a plane hold, which decides how filters treat them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlaneData {
    Color,
    Scalar,
    /// Unit vectors; only blurs apply, and results are renormalised.
    Vector,
}

/// Applies `filters` in order to straight-alpha texels (`xyz` value, `w`
/// alpha).
pub(crate) fn apply_to_plane(
    texels: &mut [Vec4],
    width: u32,
    height: u32,
    data: PlaneData,
    filters: &[LayerFilter],
) {
    for filter in filters {
        match *filter {
            LayerFilter::Blur { radius } => {
                blur_plane(texels, width, height, scaled_radius(radius, width));
            }
            LayerFilter::Sharpen { radius, amount } => {
                let mut blurred = texels.to_vec();
                blur_plane(&mut blurred, width, height, scaled_radius(radius, width));
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
pub(crate) fn apply_to_mask(values: &mut [f32], width: u32, height: u32, filters: &[LayerFilter]) {
    if filters.is_empty() {
        return;
    }
    let mut texels = values
        .iter()
        .map(|value| Vec4::new(*value, *value, *value, 1.0))
        .collect::<Vec<_>>();
    apply_to_plane(&mut texels, width, height, PlaneData::Scalar, filters);
    for (value, texel) in values.iter_mut().zip(texels) {
        *value = texel.x.clamp(0.0, 1.0);
    }
}

fn scaled_radius(radius: f32, width: u32) -> f32 {
    radius * width as f32 / REFERENCE_WIDTH
}

/// Separable box blur run three times (close to Gaussian) on premultiplied
/// values, so transparent texels do not darken their neighbours.
fn blur_plane(texels: &mut [Vec4], width: u32, height: u32, radius: f32) {
    let radius = radius.round() as usize;
    if radius == 0 || width == 0 || height == 0 {
        return;
    }
    let (width, height) = (width as usize, height as usize);
    let mut premultiplied = texels
        .iter()
        .map(|texel| (texel.truncate() * texel.w).extend(texel.w))
        .collect::<Vec<_>>();
    let mut scratch = vec![Vec4::ZERO; width.max(height)];
    for _ in 0..3 {
        for row in premultiplied.chunks_mut(width) {
            box_blur_line(row, radius, &mut scratch);
        }
        let mut column = vec![Vec4::ZERO; height];
        for x in 0..width {
            for (y, slot) in column.iter_mut().enumerate() {
                *slot = premultiplied[y * width + x];
            }
            box_blur_line(&mut column, radius, &mut scratch);
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

/// Running-sum box blur with clamped edges.
fn box_blur_line(line: &mut [Vec4], radius: usize, scratch: &mut [Vec4]) {
    let length = line.len();
    if length == 0 {
        return;
    }
    let window = (radius * 2 + 1) as f32;
    let at = |index: isize| line[index.clamp(0, length as isize - 1) as usize];
    let mut sum = Vec4::ZERO;
    for offset in -(radius as isize)..=radius as isize {
        sum += at(offset);
    }
    for (index, slot) in scratch.iter_mut().enumerate().take(length) {
        *slot = sum / window;
        sum += at(index as isize + radius as isize + 1) - at(index as isize - radius as isize);
    }
    line.copy_from_slice(&scratch[..length]);
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
            width,
            height,
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
        apply_to_mask(&mut mask, 2, 1, &[LayerFilter::Invert]);
        assert!((mask[0] - 0.8).abs() < 1e-6 && (mask[1] - 0.2).abs() < 1e-6);
    }
}
