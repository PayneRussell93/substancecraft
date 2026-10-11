//! Colour spaces of resources and the transforms between them.
//!
//! Colour channels (Base Color, Emissive) are stored sRGB-encoded, as
//! painters pick them and as engines expect them. Bitmaps tagged with
//! another colour space are converted to that encoding when a colour
//! channel samples them; data maps (roughness, normals, masks) are read raw.

use glam::{Mat3, Vec3};
use serde::{Deserialize, Serialize};

/// How a bitmap's pixel values are encoded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpace {
    /// sRGB transfer curve with Rec.709 primaries: ordinary colour images.
    #[default]
    Srgb,
    /// Linear light with Rec.709 primaries (scene-linear sRGB).
    LinearSrgb,
    /// Linear light with ACES AP1 primaries, as used in VFX pipelines.
    AcesCg,
    /// Not colour: values are data and are never converted.
    Raw,
}

/// ACEScg (AP1, D60) to linear Rec.709 (D65), Bradford-adapted.
const ACESCG_TO_LINEAR_SRGB: Mat3 = Mat3::from_cols_array(&[
    1.705_051, -0.130_256, -0.024_003, //
    -0.621_792, 1.140_805, -0.128_969, //
    -0.083_258, -0.010_548, 1.152_972,
]);

/// Linear Rec.709 (D65) to ACEScg (AP1, D60), Bradford-adapted.
const LINEAR_SRGB_TO_ACESCG: Mat3 = Mat3::from_cols_array(&[
    0.613_097, 0.070_194, 0.020_616, //
    0.339_523, 0.916_354, 0.109_570, //
    0.047_379, 0.013_452, 0.869_815,
]);

/// sRGB electro-optical transfer: encoded value to linear light.
pub fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse of [`srgb_to_linear`]: linear light to encoded value.
pub fn linear_to_srgb(value: f32) -> f32 {
    let value = value.max(0.0);
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn map(rgb: Vec3, transfer: fn(f32) -> f32) -> Vec3 {
    Vec3::new(transfer(rgb.x), transfer(rgb.y), transfer(rgb.z))
}

impl ColorSpace {
    pub const ALL: [Self; 4] = [Self::Srgb, Self::LinearSrgb, Self::AcesCg, Self::Raw];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Srgb => "sRGB",
            Self::LinearSrgb => "Linear sRGB",
            Self::AcesCg => "ACEScg",
            Self::Raw => "Raw data",
        }
    }

    /// Converts a colour in this space to linear Rec.709 light.
    pub fn to_linear_srgb(self, rgb: Vec3) -> Vec3 {
        match self {
            Self::Srgb => map(rgb, srgb_to_linear),
            Self::LinearSrgb | Self::Raw => rgb,
            Self::AcesCg => ACESCG_TO_LINEAR_SRGB * rgb,
        }
    }

    /// Converts linear Rec.709 light to this space.
    pub fn from_linear_srgb(self, rgb: Vec3) -> Vec3 {
        match self {
            Self::Srgb => map(rgb, linear_to_srgb),
            Self::LinearSrgb | Self::Raw => rgb,
            Self::AcesCg => LINEAR_SRGB_TO_ACESCG * rgb,
        }
    }

    /// Re-encodes a colour from this space as sRGB, the encoding colour
    /// channels store. Raw data passes through untouched.
    pub fn to_srgb(self, rgb: Vec3) -> Vec3 {
        match self {
            Self::Srgb | Self::Raw => rgb,
            other => Self::Srgb.from_linear_srgb(other.to_linear_srgb(rgb)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3, tolerance: f32) -> bool {
        (a - b).abs().max_element() <= tolerance
    }

    #[test]
    fn tc_0140_color_spaces_convert_with_standard_curves_and_matrices() {
        // sRGB transfer reference points.
        assert!((srgb_to_linear(0.5) - 0.214_041).abs() < 1e-5);
        assert!((linear_to_srgb(0.214_041) - 0.5).abs() < 1e-5);
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);

        // Both primaries keep white white, and the matrices invert.
        for space in [ColorSpace::LinearSrgb, ColorSpace::AcesCg] {
            assert!(close(space.to_linear_srgb(Vec3::ONE), Vec3::ONE, 1e-3));
        }
        let colour = Vec3::new(0.8, 0.3, 0.1);
        let round_trip =
            ColorSpace::AcesCg.to_linear_srgb(ColorSpace::AcesCg.from_linear_srgb(colour));
        assert!(close(round_trip, colour, 1e-4));
        // Pure Rec.709 red is inside AP1 with mostly-red weights.
        let red = ColorSpace::AcesCg.from_linear_srgb(Vec3::X);
        assert!(close(red, Vec3::new(0.613, 0.070, 0.021), 1e-3));

        // Linear mid grey encodes to sRGB 0.5; sRGB and raw pass through.
        let grey = Vec3::splat(0.214_041);
        assert!(close(
            ColorSpace::LinearSrgb.to_srgb(grey),
            Vec3::splat(0.5),
            1e-4
        ));
        assert_eq!(ColorSpace::Srgb.to_srgb(colour), colour);
        assert_eq!(ColorSpace::Raw.to_srgb(colour), colour);
    }
}
