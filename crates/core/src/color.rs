//! Colour spaces of resources and the transforms between them.
//!
//! Colour channels (Base Color, Emissive) are stored sRGB-encoded, as
//! painters pick them and as engines expect them. Bitmaps tagged with
//! another colour space are converted to that encoding when a colour
//! channel samples them; data maps (roughness, normals, masks) are read raw.

use crate::math;
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
        math::powf((value + 0.055) / 1.055, 2.4)
    }
}

/// Inverse of [`srgb_to_linear`]: linear light to encoded value.
pub fn linear_to_srgb(value: f32) -> f32 {
    let value = value.max(0.0);
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * math::powf(value, 1.0 / 2.4) - 0.055
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

    /// An sRGB-encoded colour, as brushes and fills store it, expressed in
    /// this space (raw data reads it unconverted).
    pub fn from_srgb8(self, rgb: [u8; 3]) -> Vec3 {
        let srgb = Vec3::from(rgb.map(|component| f32::from(component) / 255.0));
        match self {
            Self::Srgb | Self::Raw => srgb,
            other => other.from_linear_srgb(Self::Srgb.to_linear_srgb(srgb)),
        }
    }

    /// A colour given in this space as the sRGB bytes brushes store, and
    /// whether it lay outside the sRGB gamut or range (negative or above
    /// white in linear Rec.709 light) and was clipped. Non-finite colours
    /// clip to black.
    pub fn to_srgb8(self, rgb: Vec3) -> ([u8; 3], bool) {
        if !rgb.is_finite() {
            return ([0, 0, 0], true);
        }
        let encode = |srgb: Vec3| {
            srgb.to_array()
                .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
        };
        if matches!(self, Self::Srgb | Self::Raw) {
            let clipped = rgb.cmplt(Vec3::ZERO).any() || rgb.cmpgt(Vec3::ONE).any();
            return (encode(rgb), clipped);
        }
        // Half a byte of slack keeps colours read back from bytes unclipped.
        const SLACK: f32 = 0.5 / 255.0;
        let linear = self.to_linear_srgb(rgb);
        let clipped =
            linear.cmplt(Vec3::splat(-SLACK)).any() || linear.cmpgt(Vec3::splat(1.0 + SLACK)).any();
        let srgb = Self::Srgb.from_linear_srgb(linear.clamp(Vec3::ZERO, Vec3::ONE));
        (encode(srgb), clipped)
    }
}

/// How the viewport and renders map scene light to the display: the tone
/// curve applied before the sRGB encoding (and any display LUT).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewTransform {
    /// The renderer's default curve (ACES, legacy fit).
    #[default]
    Standard,
    /// ACES reference rendering transform.
    Aces,
    /// Filmic curve.
    Filmic,
    /// AgX: gentle highlight roll-off that keeps hues.
    Agx,
    /// Khronos PBR Neutral: base colours stay close to their values.
    PbrNeutral,
    /// No tone curve: linear light clipped at white.
    Linear,
}

impl ViewTransform {
    pub const ALL: [Self; 6] = [
        Self::Standard,
        Self::Aces,
        Self::Filmic,
        Self::Agx,
        Self::PbrNeutral,
        Self::Linear,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Aces => "ACES",
            Self::Filmic => "Filmic",
            Self::Agx => "AgX",
            Self::PbrNeutral => "PBR Neutral",
            Self::Linear => "Linear",
        }
    }
}

/// Largest 3D LUT edge accepted (the renderer takes up to 255; files from
/// colour pipelines use 17 to 65).
pub const MAX_CUBE_LUT_SIZE: usize = 129;

/// A 3D colour lookup table from an Adobe/Resolve `.cube` file, as colour
/// pipelines (OpenColorIO's `ociobakelut`, grading tools) export display
/// transforms and looks. Entries run red fastest, then green, then blue.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeLut {
    pub title: String,
    pub size: usize,
    pub data: Vec<Vec3>,
}

impl CubeLut {
    /// Parses `.cube` text. Input domains other than `0..1` are not
    /// supported, nor are 1D-only tables.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut title = String::new();
        let mut size = None;
        let mut data = Vec::new();
        for (number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut words = line.split_whitespace();
            let keyword = words.next().unwrap_or_default();
            let numbers = || -> Result<Vec<f32>, String> {
                line.split_whitespace()
                    .skip(1)
                    .map(|word| {
                        word.parse::<f32>()
                            .ok()
                            .filter(|value| value.is_finite())
                            .ok_or_else(|| format!("line {}: `{word}` is not a number", number + 1))
                    })
                    .collect()
            };
            match keyword {
                "TITLE" => title = line["TITLE".len()..].trim().trim_matches('"').to_owned(),
                "LUT_3D_SIZE" => {
                    let edge = words
                        .next()
                        .and_then(|word| word.parse::<usize>().ok())
                        .filter(|edge| (2..=MAX_CUBE_LUT_SIZE).contains(edge))
                        .ok_or_else(|| {
                            format!(
                                "line {}: LUT_3D_SIZE must be 2 to {MAX_CUBE_LUT_SIZE}",
                                number + 1
                            )
                        })?;
                    size = Some(edge);
                    data.reserve(edge * edge * edge);
                }
                "LUT_1D_SIZE" => return Err("1D LUTs are not supported; use a 3D LUT".to_owned()),
                "DOMAIN_MIN" | "DOMAIN_MAX" => {
                    let expected = if keyword == "DOMAIN_MIN" { 0.0 } else { 1.0 };
                    if numbers()?
                        .iter()
                        .any(|value| (value - expected).abs() > 1e-6)
                    {
                        return Err(format!(
                            "line {}: only the 0..1 input domain is supported",
                            number + 1
                        ));
                    }
                }
                "LUT_3D_INPUT_RANGE" | "LUT_1D_INPUT_RANGE" => {
                    return Err(format!(
                        "line {}: only the 0..1 input domain is supported",
                        number + 1
                    ));
                }
                _ if keyword.starts_with(|first: char| {
                    first.is_ascii_digit() || first == '-' || first == '.'
                }) =>
                {
                    let values: Vec<f32> = std::iter::once(keyword)
                        .chain(words)
                        .map(|word| {
                            word.parse::<f32>()
                                .ok()
                                .filter(|value| value.is_finite())
                                .ok_or_else(|| {
                                    format!("line {}: `{word}` is not a number", number + 1)
                                })
                        })
                        .collect::<Result<_, _>>()?;
                    let [red, green, blue] = values[..] else {
                        return Err(format!("line {}: expected three values", number + 1));
                    };
                    data.push(Vec3::new(red, green, blue));
                }
                // Other keywords (vendor extensions) are ignored.
                _ => {}
            }
        }
        let size = size.ok_or("the file has no LUT_3D_SIZE")?;
        if data.len() != size * size * size {
            return Err(format!(
                "expected {} entries for a {size}³ LUT, found {}",
                size * size * size,
                data.len()
            ));
        }
        Ok(Self { title, size, data })
    }

    /// The table as `.cube` text, which [`Self::parse`] reads back.
    pub fn to_cube_text(&self) -> String {
        let mut text = String::new();
        if !self.title.is_empty() {
            text.push_str(&format!("TITLE \"{}\"\n", self.title.replace('"', "'")));
        }
        text.push_str(&format!("LUT_3D_SIZE {}\n", self.size));
        for entry in &self.data {
            text.push_str(&format!("{:.6} {:.6} {:.6}\n", entry.x, entry.y, entry.z));
        }
        text
    }

    /// The table applied to `color` (each channel clamped to `0..=1`),
    /// interpolated trilinearly.
    pub fn apply(&self, color: Vec3) -> Vec3 {
        let last = (self.size - 1) as f32;
        let at = color.clamp(Vec3::ZERO, Vec3::ONE) * last;
        let low = at.floor().min(Vec3::splat(last - 1.0)).max(Vec3::ZERO);
        let t = at - low;
        let index = |r: usize, g: usize, b: usize| self.data[r + self.size * (g + self.size * b)];
        let (r0, g0, b0) = (low.x as usize, low.y as usize, low.z as usize);
        let mut result = Vec3::ZERO;
        for (db, wb) in [(0, 1.0 - t.z), (1, t.z)] {
            for (dg, wg) in [(0, 1.0 - t.y), (1, t.y)] {
                for (dr, wr) in [(0, 1.0 - t.x), (1, t.x)] {
                    result += index(r0 + dr, g0 + dg, b0 + db) * (wr * wg * wb);
                }
            }
        }
        result
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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

    #[test]
    fn tc_0335_picked_colours_convert_between_spaces_and_report_clipping() {
        // sRGB bytes read back in each space and return unchanged.
        for rgb in [[0_u8, 0, 0], [255, 255, 255], [200, 30, 30], [12, 128, 250]] {
            for space in [ColorSpace::Srgb, ColorSpace::LinearSrgb, ColorSpace::AcesCg] {
                let value = space.from_srgb8(rgb);
                assert_eq!(space.to_srgb8(value), (rgb, false), "{rgb:?} via {space:?}");
            }
        }
        // Byte 128 (0.502 encoded) is about 0.216 in linear light.
        let grey = ColorSpace::LinearSrgb.from_srgb8([128, 128, 128]);
        assert!((grey.x - 0.2158).abs() < 1e-3);
        assert_eq!(
            ColorSpace::LinearSrgb.to_srgb8(Vec3::splat(0.215_860_5)),
            ([128, 128, 128], false)
        );
        // ACEScg white is white; its saturated primaries lie outside sRGB.
        assert_eq!(
            ColorSpace::AcesCg.to_srgb8(Vec3::ONE),
            ([255, 255, 255], false)
        );
        let (green, clipped) = ColorSpace::AcesCg.to_srgb8(Vec3::new(0.0, 1.0, 0.0));
        assert!(clipped, "ACEScg green is wider than sRGB");
        assert_eq!(green[0], 0, "negative red clips to zero");
        // Over-range and non-finite values clip.
        assert_eq!(
            ColorSpace::LinearSrgb.to_srgb8(Vec3::splat(2.0)),
            ([255, 255, 255], true)
        );
        assert!(ColorSpace::Srgb.to_srgb8(Vec3::new(-0.1, 0.5, 0.5)).1);
        assert_eq!(
            ColorSpace::AcesCg.to_srgb8(Vec3::new(f32::NAN, 0.0, 0.0)),
            ([0, 0, 0], true)
        );
    }

    #[test]
    fn tc_0379_cube_luts_parse_and_interpolate_and_reject_what_they_cannot_hold() {
        // A 2³ table that swaps red and blue.
        let mut text =
            String::from("# swap\nTITLE \"Swap R/B\"\nLUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\n");
        for b in 0..2 {
            for g in 0..2 {
                for r in 0..2 {
                    text.push_str(&format!("{b} {g} {r}\n"));
                }
            }
        }
        let lut = CubeLut::parse(&text).unwrap();
        assert_eq!((lut.title.as_str(), lut.size), ("Swap R/B", 2));
        assert!(close(
            lut.apply(Vec3::new(0.8, 0.3, 0.1)),
            Vec3::new(0.1, 0.3, 0.8),
            1e-5
        ));
        assert!(
            close(lut.apply(Vec3::splat(2.0)), Vec3::ONE, 1e-5),
            "clamped"
        );

        // An identity table of edge 17 passes colours through.
        let mut identity = String::from("LUT_3D_SIZE 17\n");
        for b in 0..17 {
            for g in 0..17 {
                for r in 0..17 {
                    identity.push_str(&format!(
                        "{} {} {}\n",
                        r as f32 / 16.0,
                        g as f32 / 16.0,
                        b as f32 / 16.0
                    ));
                }
            }
        }
        let identity = CubeLut::parse(&identity).unwrap();
        let color = Vec3::new(0.123, 0.5, 0.987);
        assert!(close(identity.apply(color), color, 1e-4));

        for (bad, why) in [
            ("LUT_3D_SIZE 2\n0 0 0\n", "too few entries"),
            ("0 0 0\n", "no size"),
            ("LUT_3D_SIZE 1\n0 0 0\n", "size too small"),
            ("LUT_1D_SIZE 4\n", "1D"),
            ("LUT_3D_SIZE 2\nDOMAIN_MAX 2 2 2\n", "domain"),
            ("LUT_3D_SIZE 2\n0 0 x\n", "not a number"),
            ("LUT_3D_SIZE 2\n0 0\n", "two values"),
        ] {
            assert!(CubeLut::parse(bad).is_err(), "{why}");
        }
    }
}
