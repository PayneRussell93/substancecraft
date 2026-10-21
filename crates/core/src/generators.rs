//! Procedural mask generators driven by baked mesh maps (and, for the ID
//! generator, the texture set's ID map).
//!
//! Each mesh-map generator turns a [`MeshMapSample`] into a mask value in
//! `0..=1`; the ID generator selects texels of the ID map by colour.
//! Breakup noise is sampled in object space, so masks stay continuous across
//! UV seams and scale with the mesh rather than with its UV layout.

use crate::math;
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::extensions::PluginRef;
use crate::mesh_maps::MeshMapSample;
use crate::noise::{fbm3, gradient_noise3};
use crate::resources::{MAX_TILE_SIZE, MIN_TILE_SIZE};

const BREAKUP_OCTAVES: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum MaskGenerator {
    EdgeWear(EdgeWear),
    Dirt(Dirt),
    Orientation(Orientation),
    Gradient(Gradient),
    Noise(NoisePattern),
    /// Thin or thick parts from the thickness map: edges for subsurface
    /// glow, solid cores for dust.
    Thickness(Thickness),
    /// Texels of the ID map near one colour.
    IdColor(IdSelection),
    /// A procedural pattern or grunge in object space: stripes, checker,
    /// dots, scratches, or ridged grunge.
    Pattern(Pattern),
    /// A generator from a plugin (PROD-07), run over whole regions by its
    /// registered host.
    Plugin(PluginRef),
}

impl MaskGenerator {
    /// One default instance of every generator, in menu order.
    pub const PRESETS: [Self; 8] = [
        Self::EdgeWear(EdgeWear::DEFAULT),
        Self::Dirt(Dirt::DEFAULT),
        Self::Orientation(Orientation::DEFAULT),
        Self::Gradient(Gradient::DEFAULT),
        Self::Noise(NoisePattern::DEFAULT),
        Self::Thickness(Thickness::DEFAULT),
        Self::IdColor(IdSelection::DEFAULT),
        Self::Pattern(Pattern::DEFAULT),
    ];

    pub const fn name(&self) -> &'static str {
        match self {
            Self::EdgeWear(_) => "Edge Wear",
            Self::Dirt(_) => "Dirt",
            Self::Orientation(_) => "Orientation",
            Self::Gradient(_) => "Gradient",
            Self::Noise(_) => "Noise",
            Self::Thickness(_) => "Thickness",
            Self::IdColor(_) => "ID",
            Self::Pattern(_) => "Pattern",
            Self::Plugin(_) => "Plugin",
        }
    }

    /// True when the generator reads baked mesh maps; the ID generator
    /// reads the texture set's ID map instead.
    pub const fn needs_mesh_maps(&self) -> bool {
        !matches!(self, Self::IdColor(_))
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
                    && feature_size(settings.feature_size)
            }
            Self::Dirt(settings) => {
                unit(settings.amount)
                    && unit(settings.contrast)
                    && unit(settings.cavity_weight)
                    && unit(settings.breakup)
                    && noise_scale(settings.breakup_scale)
                    && feature_size(settings.feature_size)
            }
            Self::Orientation(settings) => {
                settings.spread_degrees.is_finite()
                    && (0.0..=180.0).contains(&settings.spread_degrees)
                    && unit(settings.softness)
            }
            Self::Gradient(settings) => unit(settings.start) && unit(settings.end),
            Self::Noise(settings) => {
                noise_scale(settings.scale)
                    && unit(settings.amount)
                    && unit(settings.contrast)
                    && feature_size(settings.feature_size)
            }
            Self::Thickness(settings) => unit(settings.amount) && unit(settings.contrast),
            Self::IdColor(settings) => unit(settings.tolerance) && unit(settings.softness),
            Self::Pattern(settings) => {
                noise_scale(settings.scale)
                    && unit(settings.width)
                    && unit(settings.softness)
                    && feature_size(settings.feature_size)
            }
            Self::Plugin(plugin) => plugin.is_valid(),
        }
    }

    /// Mask values at many samples: [`Self::evaluate`] per sample, except
    /// that a plugin generator runs once over all of them.
    pub fn evaluate_many(
        &self,
        samples: &[MeshMapSample],
        values: &mut [f32],
    ) -> Result<(), crate::BakeError> {
        if let Self::Plugin(plugin) = self {
            return crate::extensions::generate(plugin, samples, values);
        }
        for (value, sample) in values.iter_mut().zip(samples) {
            *value = self.evaluate(sample);
        }
        Ok(())
    }

    /// Mask value at one mesh-map sample. Uncovered samples are always `0`.
    /// The ID generator reads the ID map ([`IdSelection::select`]) and is
    /// `0` here; a plugin generator that fails, or is not installed, is `0`
    /// too (bakes go through [`Self::evaluate_many`], which reports it).
    pub fn evaluate(&self, sample: &MeshMapSample) -> f32 {
        if !sample.covered {
            return 0.0;
        }
        let value = match self {
            Self::EdgeWear(settings) => {
                let convexity = ((sample.curvature - 0.5) * 2.0).clamp(0.0, 1.0);
                let breakup = breakup(
                    noise_point(sample, settings.breakup_scale, settings.feature_size),
                    settings.breakup,
                    settings.seed,
                );
                level(convexity * breakup, settings.amount, settings.contrast)
            }
            Self::Dirt(settings) => {
                let cavity = ((0.5 - sample.curvature) * 2.0).clamp(0.0, 1.0);
                let occlusion = 1.0 - sample.ambient_occlusion.clamp(0.0, 1.0);
                let grime = occlusion + (cavity - occlusion) * settings.cavity_weight;
                let breakup = breakup(
                    noise_point(sample, settings.breakup_scale, settings.feature_size),
                    settings.breakup,
                    settings.seed,
                );
                level(grime * breakup, settings.amount, settings.contrast)
            }
            Self::Orientation(settings) => {
                let angle = math::acos(
                    sample
                        .normal
                        .dot(settings.direction.vector())
                        .clamp(-1.0, 1.0),
                )
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
                    noise_point(sample, settings.scale, settings.feature_size),
                    BREAKUP_OCTAVES,
                    settings.seed,
                );
                let half_width = 0.5 - 0.48 * settings.contrast;
                let threshold = 1.0 - settings.amount;
                smoothstep(threshold - half_width, threshold + half_width, noise)
            }
            Self::Thickness(settings) => {
                let thickness = sample.thickness.clamp(0.0, 1.0);
                let selected = if settings.thin {
                    1.0 - thickness
                } else {
                    thickness
                };
                level(selected, settings.amount, settings.contrast)
            }
            Self::IdColor(_) => 0.0,
            Self::Pattern(settings) => settings.evaluate(sample),
            Self::Plugin(plugin) => {
                let mut value = [0.0];
                crate::extensions::generate(plugin, std::slice::from_ref(sample), &mut value)
                    .map_or(0.0, |()| value[0])
            }
        };
        value.clamp(0.0, 1.0)
    }
}

/// Shapes of the [`Pattern`] generator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternKind {
    /// Parallel bands across `axis`.
    #[default]
    Stripes,
    /// Alternating cells, solid in 3D so every face shows squares.
    Checker,
    /// Round spots on a grid.
    Dots,
    /// Long thin random lines running along `axis`.
    Scratches,
    /// Crackled ridges, for weathered and dried-out surfaces.
    Grunge,
}

impl PatternKind {
    pub const ALL: [Self; 5] = [
        Self::Stripes,
        Self::Checker,
        Self::Dots,
        Self::Scratches,
        Self::Grunge,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Stripes => "Stripes",
            Self::Checker => "Checker",
            Self::Dots => "Dots",
            Self::Scratches => "Scratches",
            Self::Grunge => "Grunge",
        }
    }
}

/// A procedural pattern evaluated in object space, so it never seams at UV
/// borders. One repetition spans `1 / scale` of the mesh, or `feature_size`
/// scene units when set.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    pub kind: PatternKind,
    pub axis: Axis,
    pub scale: f32,
    /// Fraction of each repetition the pattern covers (stripe width, dot
    /// size, scratch or ridge density).
    pub width: f32,
    /// Edge softness.
    pub softness: f32,
    pub seed: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_size: Option<f32>,
}

impl Pattern {
    pub const DEFAULT: Self = Self {
        kind: PatternKind::Stripes,
        axis: Axis::Y,
        scale: 12.0,
        width: 0.5,
        softness: 0.1,
        seed: 5,
        feature_size: None,
    };

    fn evaluate(&self, sample: &MeshMapSample) -> f32 {
        let point = noise_point(sample, self.scale, self.feature_size);
        // A soft step: 1 inside `edge`, fading over the softness band.
        let band = (self.softness * 0.5).max(1e-3);
        let inside =
            |distance: f32, edge: f32| 1.0 - smoothstep(edge - band, edge + band, distance);
        let along = point[self.axis.index()];
        match self.kind {
            PatternKind::Stripes => {
                // Distance from the stripe's centre line, in periods.
                let phase = (along - along.floor() - 0.5).abs();
                inside(phase, self.width * 0.5)
            }
            PatternKind::Checker => {
                let cells = point.floor();
                let parity = (cells.x + cells.y + cells.z).rem_euclid(2.0);
                // Soften cell borders by the distance to the nearest face.
                let fraction = point - cells;
                let border = fraction.min(Vec3::ONE - fraction).min_element();
                let edge = smoothstep(0.0, band, border);
                if parity < 0.5 {
                    0.5 + 0.5 * edge
                } else {
                    0.5 - 0.5 * edge
                }
            }
            PatternKind::Dots => {
                let offset = point - point.floor() - Vec3::splat(0.5);
                inside(offset.length(), self.width * 0.5)
            }
            PatternKind::Scratches => {
                // Noise that varies slowly along the axis and quickly across
                // it; its zero crossings are long thin lines.
                let mut stretched = point * 3.0;
                stretched[self.axis.index()] *= 0.08;
                let noise = gradient_noise3(stretched, self.seed);
                let ridge = 1.0 - noise.abs().min(1.0);
                let density = gradient_noise3(point * 0.7, self.seed ^ 0x9e37) * 0.5 + 0.5;
                inside(1.0 - ridge, self.width * 0.15) * smoothstep(0.35, 0.6, density)
            }
            PatternKind::Grunge => {
                let noise = fbm3(point * 2.0, BREAKUP_OCTAVES, self.seed) * 2.0 - 1.0;
                let ridge = 1.0 - noise.abs().min(1.0);
                inside(1.0 - ridge, self.width * 0.3)
            }
        }
    }
}

impl Default for Pattern {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Selects the texels of the ID map whose colour lies within `tolerance` of
/// `color`, fading out over `softness` beyond it. Distances are measured
/// in RGB scaled to `0..=1` (1 is black to white).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdSelection {
    pub color: [u8; 3],
    pub tolerance: f32,
    pub softness: f32,
}

impl IdSelection {
    pub const DEFAULT: Self = Self {
        color: [255, 0, 0],
        tolerance: 0.05,
        softness: 0.05,
    };

    /// Mask value for one RGBA texel of the ID map; unpainted (transparent)
    /// texels are never selected.
    pub fn select(&self, texel: [u8; 4]) -> f32 {
        if texel[3] == 0 {
            return 0.0;
        }
        let distance = (0..3)
            .map(|channel| {
                let difference =
                    (f32::from(texel[channel]) - f32::from(self.color[channel])) / 255.0;
                difference * difference
            })
            .sum::<f32>()
            .sqrt()
            / 3.0_f32.sqrt();
        if distance <= self.tolerance {
            1.0
        } else if self.softness <= 0.0 {
            0.0
        } else {
            (1.0 - (distance - self.tolerance) / self.softness).clamp(0.0, 1.0)
        }
    }
}

impl Default for IdSelection {
    fn default() -> Self {
        Self::DEFAULT
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
    /// Size of one breakup feature in scene units (metres after import
    /// scale), replacing `breakup_scale` so the breakup keeps its real size
    /// on any mesh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_size: Option<f32>,
}

impl EdgeWear {
    pub const DEFAULT: Self = Self {
        amount: 0.5,
        contrast: 0.6,
        breakup: 0.6,
        breakup_scale: 14.0,
        seed: 1,
        feature_size: None,
    };
}

impl Default for EdgeWear {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Selects thin (or, with `thin` off, thick) parts of the mesh.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Thickness {
    pub amount: f32,
    pub contrast: f32,
    /// Select thin parts (true) or thick parts (false).
    pub thin: bool,
}

impl Thickness {
    pub const DEFAULT: Self = Self {
        amount: 0.5,
        contrast: 0.5,
        thin: true,
    };
}

impl Default for Thickness {
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
    /// Breakup feature size in scene units; see [`EdgeWear::feature_size`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_size: Option<f32>,
}

impl Dirt {
    pub const DEFAULT: Self = Self {
        amount: 0.5,
        contrast: 0.4,
        cavity_weight: 0.4,
        breakup: 0.5,
        breakup_scale: 9.0,
        seed: 2,
        feature_size: None,
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
    /// Blotch size in scene units; see [`EdgeWear::feature_size`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_size: Option<f32>,
}

impl NoisePattern {
    pub const DEFAULT: Self = Self {
        scale: 6.0,
        amount: 0.5,
        contrast: 0.5,
        seed: 3,
        feature_size: None,
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
/// Where breakup noise is sampled: object-space position divided by a
/// physical feature size, or the mesh-relative position times `scale`.
fn noise_point(sample: &MeshMapSample, scale: f32, feature_size: Option<f32>) -> Vec3 {
    match feature_size {
        Some(size) => sample.position / size,
        None => sample.relative_position * scale,
    }
}

fn feature_size(size: Option<f32>) -> bool {
    size.is_none_or(|size| size.is_finite() && (MIN_TILE_SIZE..=MAX_TILE_SIZE).contains(&size))
}

fn breakup(point: Vec3, strength: f32, seed: u32) -> f32 {
    if strength <= 0.0 {
        return 1.0;
    }
    let noise = fbm3(point, BREAKUP_OCTAVES, seed);
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
    fn tc_0245_thickness_generator_selects_thin_or_thick_parts() {
        let at = |thickness: f32| MeshMapSample {
            covered: true,
            thickness,
            ..MeshMapSample::default()
        };
        let thin = MaskGenerator::Thickness(Thickness::DEFAULT);
        assert!(thin.evaluate(&at(0.05)) > 0.9);
        assert_eq!(thin.evaluate(&at(1.0)), 0.0);
        let thick = MaskGenerator::Thickness(Thickness {
            thin: false,
            ..Thickness::DEFAULT
        });
        assert!(thick.evaluate(&at(0.95)) > 0.9);
        assert_eq!(thick.evaluate(&at(0.0)), 0.0);
        assert_eq!(
            thin.evaluate(&MeshMapSample {
                covered: false,
                ..at(0.0)
            }),
            0.0
        );
        assert!(thin.is_valid());
        assert!(
            !MaskGenerator::Thickness(Thickness {
                amount: 2.0,
                ..Thickness::DEFAULT
            })
            .is_valid()
        );
        assert_eq!(MaskGenerator::PRESETS.len(), 8);
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

    #[test]
    #[allow(clippy::unwrap_used)]
    fn tc_0276_id_generator_selects_id_colours_with_tolerance_and_needs_no_mesh_maps() {
        use std::collections::BTreeSet;

        use crate::{
            BakeContext, IdStroke, LayerFill, LayerId, MaterialChannel, MaterialLayer,
            TextureResolution, TextureSet, TextureSetId, mask_drivers,
        };

        // Colour distance: exact and near colours are in, far ones out, a
        // soft edge in between; unpainted texels are never selected.
        let red = IdSelection::DEFAULT;
        assert_eq!(red.select([255, 0, 0, 255]), 1.0);
        assert_eq!(red.select([250, 8, 4, 255]), 1.0);
        let fringe = red.select([255, 30, 20, 255]);
        assert!(fringe > 0.0 && fringe < 1.0, "{fringe}");
        assert_eq!(red.select([0, 0, 255, 255]), 0.0);
        assert_eq!(red.select([255, 0, 0, 0]), 0.0);
        let hard = IdSelection {
            tolerance: 0.0,
            softness: 0.0,
            ..red
        };
        assert_eq!(hard.select([255, 0, 0, 255]), 1.0);
        assert_eq!(hard.select([254, 0, 0, 255]), 0.0);
        assert!(
            !MaskGenerator::IdColor(IdSelection {
                tolerance: 1.5,
                ..red
            })
            .is_valid()
        );
        assert!(!MaskGenerator::IdColor(red).needs_mesh_maps());
        assert_eq!(MaskGenerator::IdColor(red).name(), "ID");

        // A red ID patch on the left; a blue fill selected by the ID
        // generator shows only there, with no mesh maps.
        let mut ids = MaterialLayer::new_paint(LayerId(1), "IDs");
        ids.channels = BTreeSet::from([MaterialChannel::Id]);
        ids.id_strokes.push(IdStroke {
            frames: Default::default(),
            points: vec![glam::Vec2::new(0.25, 0.5)],
            radius_pixels: 5.0,
            color: [255, 0, 0],
            opacity: 1.0,
            tip: crate::BrushTip {
                hardness: 1.0,
                ..crate::BrushTip::ROUND
            },
            pressures: Vec::new(),
        });
        let mut fill = MaterialLayer::new_fill(
            LayerId(2),
            "Selected",
            LayerFill {
                color: [0, 0, 255, 255],
                ..LayerFill::default()
            },
        );
        fill.channels = BTreeSet::from([MaterialChannel::BaseColor]);
        fill.mask_generator = Some(MaskGenerator::IdColor(red));
        let set = TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Body".to_owned(),
            material_slot: "Body".to_owned(),
            resolution: TextureResolution {
                width: 32,
                height: 32,
            },
            channels: BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Id]),
            layers: vec![ids, fill],
            height_strength: crate::DEFAULT_HEIGHT_STRENGTH,
            export: None,
        };
        assert!(!set.requires_mesh_maps());
        let base = set
            .bake_channel(MaterialChannel::BaseColor, BakeContext::default())
            .unwrap();
        let at = |u: f32| base.sample_uv(glam::Vec2::new(u, 0.5)).unwrap();
        assert_eq!(at(0.25), [0, 0, 255, 255], "inside the red ID patch");
        assert_eq!(at(0.75)[3], 0, "outside it");

        // The generator alone (solo) and the inspector both read the ID map.
        let alone = set
            .bake_layer_mask(LayerId(2), true, BakeContext::default())
            .unwrap();
        assert_eq!(alone.sample_uv(glam::Vec2::new(0.25, 0.5)).unwrap()[0], 255);
        assert_eq!(alone.sample_uv(glam::Vec2::new(0.75, 0.5)).unwrap()[0], 0);
        assert_eq!(
            mask_drivers(set.layer(LayerId(2)).unwrap(), &set),
            ["ID generator (ID map)"]
        );
    }

    #[test]
    fn tc_0289_physical_feature_sizes_keep_noise_the_same_size_on_any_mesh() {
        // One surface point seen on a small mesh and on one twice as large:
        // the same object position, but different mesh-relative positions.
        let on = |relative: Vec3| MeshMapSample {
            covered: true,
            position: Vec3::new(0.31, 0.47, 0.12),
            relative_position: relative,
            curvature: 1.0,
            ambient_occlusion: 0.0,
            ..MeshMapSample::default()
        };
        let small = on(Vec3::new(0.6, 0.7, 0.3));
        let large = on(Vec3::new(0.3, 0.35, 0.15));
        let noise = |feature_size| {
            MaskGenerator::Noise(NoisePattern {
                contrast: 0.0,
                feature_size,
                ..NoisePattern::DEFAULT
            })
        };
        // Relative noise changes with the mesh; physical noise does not.
        let relative = noise(None);
        assert_ne!(relative.evaluate(&small), relative.evaluate(&large));
        let physical = noise(Some(0.2));
        assert_eq!(physical.evaluate(&small), physical.evaluate(&large));
        for generator in [
            MaskGenerator::EdgeWear(EdgeWear {
                feature_size: Some(0.05),
                ..EdgeWear::DEFAULT
            }),
            MaskGenerator::Dirt(Dirt {
                feature_size: Some(0.05),
                ..Dirt::DEFAULT
            }),
        ] {
            assert!(generator.is_valid());
            assert_eq!(generator.evaluate(&small), generator.evaluate(&large));
        }
        // Sizes outside the projection tile-size range are invalid; files
        // without a size still load.
        assert!(!noise(Some(0.0)).is_valid());
        assert!(!noise(Some(f32::INFINITY)).is_valid());
        assert!(!noise(Some(5000.0)).is_valid());
        let old: MaskGenerator = serde_json::from_str(
            r#"{"kind":"Noise","scale":6.0,"amount":0.5,"contrast":0.5,"seed":3}"#,
        )
        .unwrap_or(MaskGenerator::Noise(NoisePattern {
            feature_size: Some(1.0),
            ..NoisePattern::DEFAULT
        }));
        assert_eq!(old, MaskGenerator::Noise(NoisePattern::DEFAULT));
    }

    #[test]
    fn tc_0290_pattern_generators_tile_in_object_space() {
        let at = |position: Vec3| MeshMapSample {
            covered: true,
            position,
            relative_position: position,
            ..MeshMapSample::default()
        };
        let pattern = |kind, width| Pattern {
            kind,
            width,
            softness: 0.0,
            scale: 1.0,
            ..Pattern::DEFAULT
        };
        // Stripes repeat along their axis and are constant across it.
        let stripes = pattern(PatternKind::Stripes, 0.5);
        assert_eq!(stripes.evaluate(&at(Vec3::new(0.0, 0.5, 0.0))), 1.0);
        assert_eq!(stripes.evaluate(&at(Vec3::new(0.0, 0.0, 0.0))), 0.0);
        assert_eq!(stripes.evaluate(&at(Vec3::new(0.0, 1.5, 0.0))), 1.0);
        assert_eq!(stripes.evaluate(&at(Vec3::new(0.7, 0.5, 0.3))), 1.0);
        // Checker cells alternate in every direction.
        let checker = pattern(PatternKind::Checker, 0.5);
        let cell = |x: f32, y: f32, z: f32| checker.evaluate(&at(Vec3::new(x, y, z)));
        assert_eq!(cell(0.5, 0.5, 0.5), 1.0);
        assert_eq!(cell(1.5, 0.5, 0.5), 0.0);
        assert_eq!(cell(1.5, 1.5, 0.5), 1.0);
        assert_eq!(cell(-0.5, 0.5, 0.5), 0.0);
        // Dots fill cell centres, wider dots reach further.
        let dots = pattern(PatternKind::Dots, 0.4);
        assert_eq!(dots.evaluate(&at(Vec3::splat(0.5))), 1.0);
        assert_eq!(dots.evaluate(&at(Vec3::splat(0.05))), 0.0);
        let near = at(Vec3::new(0.5, 0.5, 0.75));
        assert_eq!(dots.evaluate(&near), 0.0);
        assert_eq!(pattern(PatternKind::Dots, 0.8).evaluate(&near), 1.0);
        // Grunges cover part of the surface, deterministically per seed;
        // more width covers more.
        for kind in [PatternKind::Scratches, PatternKind::Grunge] {
            let coverage = |width: f32, seed: u32| {
                let generator = MaskGenerator::Pattern(Pattern {
                    kind,
                    width,
                    seed,
                    softness: 0.1,
                    ..Pattern::DEFAULT
                });
                (0..400)
                    .map(|index| {
                        let position = Vec3::new(
                            (index % 20) as f32 * 0.037,
                            (index / 20) as f32 * 0.041,
                            0.3,
                        );
                        generator.evaluate(&at(position))
                    })
                    .sum::<f32>()
                    / 400.0
            };
            let (thin, wide) = (coverage(0.2, 5), coverage(0.8, 5));
            assert!(
                thin > 0.0 && thin < wide && wide < 1.0,
                "{kind:?}: {thin} {wide}"
            );
            assert_eq!(coverage(0.5, 5), coverage(0.5, 5));
            assert_ne!(coverage(0.5, 5), coverage(0.5, 6));
        }
        assert_eq!(MaskGenerator::Pattern(Pattern::DEFAULT).name(), "Pattern");
        assert!(
            !MaskGenerator::Pattern(Pattern {
                width: 2.0,
                ..Pattern::DEFAULT
            })
            .is_valid()
        );
        assert!(MaskGenerator::Pattern(Pattern::DEFAULT).needs_mesh_maps());
    }
}
