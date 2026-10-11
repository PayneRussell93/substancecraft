//! Procedural mask generators driven by baked mesh maps.
//!
//! Each generator turns a [`MeshMapSample`] into a mask value in `0..=1`.
//! Breakup noise is sampled in object space, so masks stay continuous across
//! UV seams and scale with the mesh rather than with its UV layout.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::mesh_maps::MeshMapSample;
use crate::noise::fbm3;

const BREAKUP_OCTAVES: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum MaskGenerator {
    EdgeWear(EdgeWear),
    Dirt(Dirt),
    Orientation(Orientation),
    Gradient(Gradient),
    Noise(NoisePattern),
}

impl MaskGenerator {
    /// One default instance of every generator, in menu order.
    pub const PRESETS: [Self; 5] = [
        Self::EdgeWear(EdgeWear::DEFAULT),
        Self::Dirt(Dirt::DEFAULT),
        Self::Orientation(Orientation::DEFAULT),
        Self::Gradient(Gradient::DEFAULT),
        Self::Noise(NoisePattern::DEFAULT),
    ];

    pub const fn name(&self) -> &'static str {
        match self {
            Self::EdgeWear(_) => "Edge Wear",
            Self::Dirt(_) => "Dirt",
            Self::Orientation(_) => "Orientation",
            Self::Gradient(_) => "Gradient",
            Self::Noise(_) => "Noise",
        }
    }

    pub fn same_kind(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    pub fn is_valid(&self) -> bool {
        match self {
            Self::EdgeWear(settings) => {
                unit(settings.amount)
                    && unit(settings.contrast)
                    && unit(settings.breakup)
                    && noise_scale(settings.breakup_scale)
            }
            Self::Dirt(settings) => {
                unit(settings.amount)
                    && unit(settings.contrast)
                    && unit(settings.cavity_weight)
                    && unit(settings.breakup)
                    && noise_scale(settings.breakup_scale)
            }
            Self::Orientation(settings) => {
                settings.spread_degrees.is_finite()
                    && (0.0..=180.0).contains(&settings.spread_degrees)
                    && unit(settings.softness)
            }
            Self::Gradient(settings) => unit(settings.start) && unit(settings.end),
            Self::Noise(settings) => {
                noise_scale(settings.scale) && unit(settings.amount) && unit(settings.contrast)
            }
        }
    }

    /// Mask value at one mesh-map sample. Uncovered samples are always `0`.
    pub fn evaluate(&self, sample: &MeshMapSample) -> f32 {
        if !sample.covered {
            return 0.0;
        }
        let value = match self {
            Self::EdgeWear(settings) => {
                let convexity = ((sample.curvature - 0.5) * 2.0).clamp(0.0, 1.0);
                let breakup = breakup(
                    sample.relative_position,
                    settings.breakup,
                    settings.breakup_scale,
                    settings.seed,
                );
                level(convexity * breakup, settings.amount, settings.contrast)
            }
            Self::Dirt(settings) => {
                let cavity = ((0.5 - sample.curvature) * 2.0).clamp(0.0, 1.0);
                let occlusion = 1.0 - sample.ambient_occlusion.clamp(0.0, 1.0);
                let grime = occlusion + (cavity - occlusion) * settings.cavity_weight;
                let breakup = breakup(
                    sample.relative_position,
                    settings.breakup,
                    settings.breakup_scale,
                    settings.seed,
                );
                level(grime * breakup, settings.amount, settings.contrast)
            }
            Self::Orientation(settings) => {
                let angle = sample
                    .normal
                    .dot(settings.direction.vector())
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees();
                let half_width = (settings.softness * 45.0).max(0.5);
                1.0 - smoothstep(
                    settings.spread_degrees - half_width,
                    settings.spread_degrees + half_width,
                    angle,
                )
            }
            Self::Gradient(settings) => {
                let coordinate = sample.unit_position[settings.axis.index()];
                let span = settings.end - settings.start;
                if span.abs() <= f32::EPSILON {
                    f32::from(coordinate >= settings.start)
                } else {
                    (coordinate - settings.start) / span
                }
            }
            Self::Noise(settings) => {
                let noise = fbm3(
                    sample.relative_position * settings.scale,
                    BREAKUP_OCTAVES,
                    settings.seed,
                );
                let half_width = 0.5 - 0.48 * settings.contrast;
                let threshold = 1.0 - settings.amount;
                smoothstep(threshold - half_width, threshold + half_width, noise)
            }
        };
        value.clamp(0.0, 1.0)
    }
}

/// Convex-edge mask: chipped paint on corners and rims.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeWear {
    /// How far wear spreads from sharp edges onto flat surfaces.
    pub amount: f32,
    /// Hardness of the wear border.
    pub contrast: f32,
    /// Strength of the noise that chips the wear into irregular patches.
    pub breakup: f32,
    /// Breakup noise frequency relative to the mesh size.
    pub breakup_scale: f32,
    pub seed: u32,
}

impl EdgeWear {
    pub const DEFAULT: Self = Self {
        amount: 0.5,
        contrast: 0.6,
        breakup: 0.6,
        breakup_scale: 14.0,
        seed: 1,
    };
}

impl Default for EdgeWear {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Cavity and occlusion mask: grime collecting in crevices.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dirt {
    pub amount: f32,
    pub contrast: f32,
    /// `0` uses ambient occlusion only, `1` uses concave curvature only.
    pub cavity_weight: f32,
    pub breakup: f32,
    pub breakup_scale: f32,
    pub seed: u32,
}

impl Dirt {
    pub const DEFAULT: Self = Self {
        amount: 0.5,
        contrast: 0.4,
        cavity_weight: 0.4,
        breakup: 0.5,
        breakup_scale: 9.0,
        seed: 2,
    };
}

impl Default for Dirt {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    #[default]
    Up,
    Down,
    PositiveX,
    NegativeX,
    PositiveZ,
    NegativeZ,
}

impl Direction {
    pub const ALL: [Self; 6] = [
        Self::Up,
        Self::Down,
        Self::PositiveX,
        Self::NegativeX,
        Self::PositiveZ,
        Self::NegativeZ,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Up => "+Y",
            Self::Down => "−Y",
            Self::PositiveX => "+X",
            Self::NegativeX => "−X",
            Self::PositiveZ => "+Z",
            Self::NegativeZ => "−Z",
        }
    }

    pub const fn vector(self) -> Vec3 {
        match self {
            Self::Up => Vec3::Y,
            Self::Down => Vec3::NEG_Y,
            Self::PositiveX => Vec3::X,
            Self::NegativeX => Vec3::NEG_X,
            Self::PositiveZ => Vec3::Z,
            Self::NegativeZ => Vec3::NEG_Z,
        }
    }
}

/// Selects surfaces facing a direction, e.g. dust on upward faces.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Orientation {
    pub direction: Direction,
    /// Angle from the direction, in degrees, still fully selected.
    pub spread_degrees: f32,
    pub softness: f32,
}

impl Orientation {
    pub const DEFAULT: Self = Self {
        direction: Direction::Up,
        spread_degrees: 50.0,
        softness: 0.4,
    };
}

impl Default for Orientation {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    X,
    #[default]
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    pub const fn name(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Linear ramp across the mesh bounds; swap start and end to invert.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gradient {
    pub axis: Axis,
    /// Bounds-relative coordinate where the mask is `0`.
    pub start: f32,
    /// Bounds-relative coordinate where the mask is `1`.
    pub end: f32,
}

impl Gradient {
    pub const DEFAULT: Self = Self {
        axis: Axis::Y,
        start: 0.0,
        end: 1.0,
    };
}

impl Default for Gradient {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Seamless object-space fractal noise.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoisePattern {
    pub scale: f32,
    pub amount: f32,
    pub contrast: f32,
    pub seed: u32,
}

impl NoisePattern {
    pub const DEFAULT: Self = Self {
        scale: 6.0,
        amount: 0.5,
        contrast: 0.5,
        seed: 3,
    };
}

impl Default for NoisePattern {
    fn default() -> Self {
        Self::DEFAULT
    }
}

fn unit(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn noise_scale(value: f32) -> bool {
    value.is_finite() && (0.1..=256.0).contains(&value)
}

/// Multiplicative noise factor averaging `1`, so zero-valued regions stay zero.
fn breakup(position: Vec3, strength: f32, scale: f32, seed: u32) -> f32 {
    if strength <= 0.0 {
        return 1.0;
    }
    let noise = fbm3(position * scale, BREAKUP_OCTAVES, seed);
    1.0 + (noise * 2.0 - 1.0) * strength * 1.5
}

/// Threshold `value` with a soft band. `amount` 0 selects nothing and
/// `amount` 1 selects everything above zero; zero input always stays `0`.
fn level(value: f32, amount: f32, contrast: f32) -> f32 {
    let half_width = 0.3 - 0.29 * contrast;
    let threshold = (1.0 + half_width) + (half_width - (1.0 + half_width)) * amount;
    if value <= 0.0 {
        return 0.0;
    }
    smoothstep(threshold - half_width, threshold + half_width, value)
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if edge1 <= edge0 {
        return f32::from(value >= edge0);
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(curvature: f32, ambient_occlusion: f32, normal: Vec3) -> MeshMapSample {
        MeshMapSample {
            covered: true,
            curvature,
            ambient_occlusion,
            normal,
            unit_position: Vec3::new(0.2, 0.75, 0.4),
            relative_position: Vec3::new(0.1, 0.4, 0.2),
            ..MeshMapSample::default()
        }
    }

    #[test]
    fn tc_0036_edge_wear_selects_convex_edges_and_ignores_flat_surfaces() {
        let wear = MaskGenerator::EdgeWear(EdgeWear {
            breakup: 0.0,
            ..EdgeWear::DEFAULT
        });
        assert_eq!(wear.evaluate(&sample(0.5, 1.0, Vec3::Y)), 0.0);
        assert_eq!(wear.evaluate(&sample(0.2, 1.0, Vec3::Y)), 0.0);
        assert!(wear.evaluate(&sample(1.0, 1.0, Vec3::Y)) > 0.95);

        let none = MaskGenerator::EdgeWear(EdgeWear {
            amount: 0.0,
            breakup: 0.0,
            ..EdgeWear::DEFAULT
        });
        assert_eq!(none.evaluate(&sample(1.0, 1.0, Vec3::Y)), 0.0);
        let everything = MaskGenerator::EdgeWear(EdgeWear {
            amount: 1.0,
            breakup: 0.0,
            ..EdgeWear::DEFAULT
        });
        assert!(everything.evaluate(&sample(0.65, 1.0, Vec3::Y)) > 0.9);
        assert_eq!(everything.evaluate(&sample(0.5, 1.0, Vec3::Y)), 0.0);

        let uncovered = MeshMapSample::default();
        assert_eq!(wear.evaluate(&uncovered), 0.0);
    }

    #[test]
    fn tc_0037_dirt_orientation_gradient_and_noise_respond_to_mesh_maps() {
        let dirt = MaskGenerator::Dirt(Dirt {
            breakup: 0.0,
            ..Dirt::DEFAULT
        });
        assert!(dirt.evaluate(&sample(0.0, 0.0, Vec3::Y)) > 0.95);
        assert!(
            dirt.evaluate(&sample(0.3, 0.6, Vec3::Y)) < dirt.evaluate(&sample(0.0, 0.0, Vec3::Y))
        );
        assert_eq!(dirt.evaluate(&sample(0.5, 1.0, Vec3::Y)), 0.0);

        let up = MaskGenerator::Orientation(Orientation::DEFAULT);
        assert_eq!(up.evaluate(&sample(0.5, 1.0, Vec3::Y)), 1.0);
        assert_eq!(up.evaluate(&sample(0.5, 1.0, Vec3::NEG_Y)), 0.0);

        let gradient = MaskGenerator::Gradient(Gradient::DEFAULT);
        assert!((gradient.evaluate(&sample(0.5, 1.0, Vec3::Y)) - 0.75).abs() < 1e-5);

        let noise = MaskGenerator::Noise(NoisePattern::DEFAULT);
        let value = noise.evaluate(&sample(0.5, 1.0, Vec3::Y));
        assert!((0.0..=1.0).contains(&value));

        for preset in MaskGenerator::PRESETS {
            assert!(
                preset.is_valid(),
                "{} defaults must validate",
                preset.name()
            );
        }
        assert!(
            !MaskGenerator::EdgeWear(EdgeWear {
                amount: f32::NAN,
                ..EdgeWear::DEFAULT
            })
            .is_valid()
        );
    }
}
