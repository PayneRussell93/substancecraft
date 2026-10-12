//! Lighting-check environments for the viewport: six original, procedural
//! HDR environments (studio, overcast, sunset, warehouse, bright outdoor,
//! and night), each with an optional sun or moon.
//!
//! Every environment is a function of direction, rendered into an
//! equirectangular image on demand, so nothing is downloaded or licensed:
//! the images are this project's own work and redistributable with it.
//! Radiance is linear and relative; [`LightingPreset::environment_lux`]
//! scales it to the viewport's photometric units. The sun is a directional
//! light rather than a disc in the image, so it is not counted twice.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Width of the equirectangular images the viewport prefilters.
pub const ENVIRONMENT_WIDTH: u32 = 512;
/// Height of those images (half the width).
pub const ENVIRONMENT_HEIGHT: u32 = ENVIRONMENT_WIDTH / 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightingPreset {
    /// Three softboxes in a dark room: key, fill, and rim.
    #[default]
    Studio,
    /// A bright, even overcast sky over grey ground; no sun.
    Overcast,
    /// A low warm sun under a blue-to-orange sky.
    Sunset,
    /// An interior lit by ceiling strip lights and one daylight opening.
    Warehouse,
    /// A clear blue sky with a high, strong sun.
    BrightOutdoor,
    /// A dark sky with cool moonlight and a faint warm city glow.
    Night,
}

/// A directional light: the sun or moon of an environment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SunLight {
    /// Direction the light travels (from the sun towards the scene), +Y up.
    pub direction: Vec3,
    /// Linear colour.
    pub color: Vec3,
    /// Illuminance in lux.
    pub illuminance: f32,
}

/// A linear RGB equirectangular image, rows top (+Y) to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentImage {
    pub width: u32,
    pub height: u32,
    /// `width * height` RGB triples.
    pub rgb: Vec<f32>,
}

impl LightingPreset {
    pub const ALL: [Self; 6] = [
        Self::Studio,
        Self::Overcast,
        Self::Sunset,
        Self::Warehouse,
        Self::BrightOutdoor,
        Self::Night,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Studio => "Studio",
            Self::Overcast => "Overcast",
            Self::Sunset => "Sunset",
            Self::Warehouse => "Warehouse",
            Self::BrightOutdoor => "Bright outdoor",
            Self::Night => "Night",
        }
    }

    /// One line on what the preset is good for checking.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Studio => "Neutral softboxes: clean reads of colour, roughness, and edges",
            Self::Overcast => "Soft, shadowless sky: albedo and ambient occlusion",
            Self::Sunset => "Low warm sun: height detail and colour under tinted light",
            Self::Warehouse => "Interior strip lights: reflections on metals and gloss",
            Self::BrightOutdoor => "Hard noon sun and blue sky: contrast and saturation",
            Self::Night => "Dim cool moonlight: emissive parts and dark values",
        }
    }

    /// Lux that a relative radiance of 1 stands for.
    pub const fn environment_lux(self) -> f32 {
        match self {
            Self::Studio => 22_000.0,
            Self::Overcast => 30_000.0,
            Self::Sunset => 14_000.0,
            Self::Warehouse => 20_000.0,
            Self::BrightOutdoor => 24_000.0,
            Self::Night => 9_000.0,
        }
    }

    /// The preset's sun or moon, before any rotation.
    pub fn sun(self) -> Option<SunLight> {
        let light = |elevation: f32, azimuth: f32, color: Vec3, illuminance: f32| SunLight {
            direction: -direction(elevation.to_radians(), azimuth.to_radians()),
            color,
            illuminance,
        };
        match self {
            Self::Studio | Self::Overcast | Self::Warehouse => None,
            Self::Sunset => Some(light(6.0, 110.0, Vec3::new(1.0, 0.55, 0.3), 45_000.0)),
            Self::BrightOutdoor => Some(light(55.0, -30.0, Vec3::new(1.0, 0.97, 0.9), 100_000.0)),
            Self::Night => Some(light(35.0, 40.0, Vec3::new(0.65, 0.75, 1.0), 12_000.0)),
        }
    }

    /// Relative linear radiance arriving from `direction` (+Y up).
    pub fn radiance(self, toward: Vec3) -> Vec3 {
        let d = toward.normalize_or_zero();
        let elevation = d.y.clamp(-1.0, 1.0).asin();
        match self {
            Self::Studio => {
                let room = mix(
                    Vec3::splat(0.03),
                    Vec3::splat(0.07),
                    smoothstep(-0.2, 0.4, d.y),
                );
                room + softbox(d, -40.0, 35.0, 0.45, 0.6) * 11.0
                    + softbox(d, 65.0, 15.0, 0.35, 0.5) * Vec3::new(3.6, 3.8, 4.2)
                    + softbox(d, 180.0, 55.0, 0.7, 0.25) * 7.0
            }
            Self::Overcast => {
                // CIE overcast sky: three times brighter at the zenith.
                let sky = Vec3::new(0.92, 0.95, 1.0) * ((1.0 + 2.0 * d.y.max(0.0)) / 3.0);
                let ground = Vec3::new(0.9, 0.85, 0.8) * 0.25;
                mix(ground, sky, horizon_blend(d.y))
            }
            Self::Sunset => {
                let height = (elevation / std::f32::consts::FRAC_PI_2).max(0.0);
                let zenith = Vec3::new(0.05, 0.08, 0.2);
                let horizon = Vec3::new(1.2, 0.55, 0.25);
                let sky = mix(horizon, zenith, height.powf(0.45));
                let toward_sun = d.dot(direction(6f32.to_radians(), 110f32.to_radians()));
                let glow = Vec3::new(2.0, 0.9, 0.4) * ((toward_sun - 1.0) * 6.0).exp();
                let ground = Vec3::new(0.05, 0.035, 0.03);
                mix(ground, sky + glow, horizon_blend(d.y))
            }
            Self::Warehouse => {
                let walls = Vec3::new(0.12, 0.11, 0.1);
                let floor = Vec3::new(0.07, 0.07, 0.065);
                let mut light = mix(floor, walls, horizon_blend(d.y));
                if d.y > 0.05 {
                    // Strip lights across a ceiling at height 1, every 1.2
                    // units along Z, 6 units long.
                    let x = d.x / d.y;
                    let z = d.z / d.y;
                    let across = (z / 1.2).rem_euclid(1.0) - 0.5;
                    let strip = (1.0 - smoothstep(0.03, 0.06, across.abs()))
                        * (1.0 - smoothstep(2.8, 3.0, x.abs()))
                        * smoothstep(0.05, 0.2, d.y);
                    light = mix(light, Vec3::splat(0.04), smoothstep(0.1, 0.5, d.y))
                        + Vec3::new(13.0, 13.6, 14.2) * strip;
                }
                // A daylight opening in the +X wall.
                let opening = (1.0 - smoothstep(0.38, 0.45, azimuth_from(d, 90.0).abs()))
                    * smoothstep(-0.02, 0.02, elevation)
                    * (1.0 - smoothstep(0.38, 0.45, elevation));
                light + Vec3::new(5.1, 5.5, 6.0) * opening
            }
            Self::BrightOutdoor => {
                let height = (elevation / std::f32::consts::FRAC_PI_2).max(0.0);
                let sky = mix(
                    Vec3::new(0.75, 0.85, 1.0),
                    Vec3::new(0.25, 0.45, 0.95),
                    height.powf(0.6),
                ) * 1.5;
                let ground = Vec3::new(0.2, 0.18, 0.13);
                mix(ground, sky, horizon_blend(d.y))
            }
            Self::Night => {
                let height = (elevation / std::f32::consts::FRAC_PI_2).max(0.0);
                let sky = mix(
                    Vec3::new(0.03, 0.03, 0.05),
                    Vec3::new(0.008, 0.012, 0.03),
                    height.sqrt(),
                );
                let toward_city = d.dot(direction(0.0, 200f32.to_radians())).max(0.0);
                let glow = Vec3::new(0.09, 0.05, 0.02)
                    * toward_city.powi(6)
                    * (1.0 - smoothstep(0.0, 0.35, elevation));
                let ground = Vec3::splat(0.006);
                mix(ground, sky + glow, horizon_blend(d.y))
            }
        }
    }

    /// The environment as an equirectangular image for the viewport.
    pub fn equirect(self, width: u32) -> EnvironmentImage {
        EnvironmentImage::from_fn(width, |toward| self.radiance(toward))
    }
}

impl EnvironmentImage {
    /// An image `width` wide (and half as tall) of `radiance` in each
    /// direction, laid out as the viewport's prefilter reads one: row `v`
    /// looks at elevation `asin(y) = π (1 − 2v) / 2`, and column `u` along
    /// `atan2(x, z) = π (1 − 2u)`. Filament builds its cubemap mirrored, so
    /// azimuth runs right to left here for directions to land where they
    /// point.
    pub fn from_fn(width: u32, radiance: impl Fn(Vec3) -> Vec3) -> Self {
        let width = width.max(2);
        let height = (width / 2).max(1);
        let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
        for row in 0..height {
            let elevation =
                std::f32::consts::FRAC_PI_2 * (1.0 - 2.0 * (row as f32 + 0.5) / height as f32);
            for column in 0..width {
                let azimuth =
                    std::f32::consts::PI * (1.0 - 2.0 * (column as f32 + 0.5) / width as f32);
                rgb.extend(radiance(direction(elevation, azimuth)).to_array());
            }
        }
        Self { width, height, rgb }
    }
}

/// Unit direction at `elevation` above the horizon and `azimuth` measured
/// from +Z towards +X (radians).
pub fn direction(elevation: f32, azimuth: f32) -> Vec3 {
    Vec3::new(
        elevation.cos() * azimuth.sin(),
        elevation.sin(),
        elevation.cos() * azimuth.cos(),
    )
}

/// `direction` rotated about +Y by `angle` radians, as the viewport's
/// light rotation turns the environment and sun together.
pub fn rotate_about_up(direction: Vec3, angle: f32) -> Vec3 {
    glam::Quat::from_rotation_y(angle) * direction
}

/// Signed azimuth of `d` relative to `degrees`, in radians (`-π..π`).
fn azimuth_from(d: Vec3, degrees: f32) -> f32 {
    let difference = d.x.atan2(d.z) - degrees.to_radians();
    (difference + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

/// Coverage of a soft-edged rectangular light centred at the given azimuth
/// and elevation (degrees), with half extents measured on the plane one
/// unit away from the viewer.
fn softbox(
    d: Vec3,
    azimuth_degrees: f32,
    elevation_degrees: f32,
    half_width: f32,
    half_height: f32,
) -> f32 {
    let center = direction(elevation_degrees.to_radians(), azimuth_degrees.to_radians());
    let facing = d.dot(center);
    if facing <= 0.0 {
        return 0.0;
    }
    let right = Vec3::Y.cross(center).normalize_or_zero();
    let up = center.cross(right);
    let x = d.dot(right) / facing;
    let y = d.dot(up) / facing;
    let edge = |offset: f32, half: f32| 1.0 - smoothstep(half * 0.85, half, offset.abs());
    edge(x, half_width) * edge(y, half_height)
}

/// 0 below the horizon, 1 above, blended over a few degrees.
fn horizon_blend(y: f32) -> f32 {
    smoothstep(-0.03, 0.03, y)
}

fn mix(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luminance(rgb: Vec3) -> f32 {
        rgb.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    fn pixel(image: &EnvironmentImage, column: u32, row: u32) -> Vec3 {
        let index = (row * image.width + column) as usize * 3;
        Vec3::from_slice(&image.rgb[index..index + 3])
    }

    #[test]
    fn tc_0311_every_preset_is_finite_named_and_lit_from_above() {
        let names: std::collections::BTreeSet<&str> =
            LightingPreset::ALL.iter().map(|preset| preset.name()).collect();
        assert_eq!(names.len(), 6);
        for preset in LightingPreset::ALL {
            let image = preset.equirect(64);
            assert_eq!((image.width, image.height), (64, 32));
            assert_eq!(image.rgb.len(), 64 * 32 * 3);
            assert!(
                image.rgb.iter().all(|value| value.is_finite() && *value >= 0.0),
                "{preset:?}"
            );
            // The upper hemisphere carries more light than the ground.
            let half = |rows: std::ops::Range<u32>| {
                rows.flat_map(|row| (0..64).map(move |column| (column, row)))
                    .map(|(column, row)| luminance(pixel(&image, column, row)))
                    .sum::<f32>()
            };
            assert!(half(0..16) > half(16..32), "{preset:?}");
            assert!(!preset.description().is_empty());
            assert!(preset.environment_lux() > 0.0);
            if let Some(sun) = preset.sun() {
                // Suns shine downwards, from above the horizon.
                assert!(sun.direction.y < 0.0, "{preset:?}");
                assert!((sun.direction.length() - 1.0).abs() < 1e-5);
                assert!(sun.illuminance > 0.0);
            }
        }
        assert!(LightingPreset::BrightOutdoor.sun().unwrap().illuminance
            > LightingPreset::Night.sun().unwrap().illuminance * 4.0);
        assert!(LightingPreset::Overcast.sun().is_none());
    }

    #[test]
    fn tc_0312_equirect_layout_matches_directions_and_presets_keep_their_character() {
        // Directions: +Y up, azimuth 0 along +Z and a quarter turn along +X.
        let up = direction(std::f32::consts::FRAC_PI_2, 0.0);
        assert!((up - Vec3::Y).length() < 1e-6);
        assert!((direction(0.0, 0.0) - Vec3::Z).length() < 1e-6);
        assert!((direction(0.0, std::f32::consts::FRAC_PI_2) - Vec3::X).length() < 1e-6);
        let image = LightingPreset::Warehouse.equirect(256);
        // The daylight opening is in the +X wall, a quarter of the way across
        // the image (azimuth runs right to left) just above the horizon row,
        // and not opposite it.
        let row = image.height / 2 - 6;
        let opening = pixel(&image, image.width / 4, row);
        let opposite = pixel(&image, image.width * 3 / 4, row);
        assert!(luminance(opening) > 4.0, "{opening}");
        assert!(luminance(opposite) < 0.5, "{opposite}");

        // Sunset glows warm towards its sun and is blue overhead; night is
        // darker than every daytime sky.
        let sunset = LightingPreset::Sunset;
        let sun_side = sunset.radiance(direction(0.1, 110f32.to_radians()));
        assert!(sun_side.x > sun_side.z * 2.0);
        let zenith = sunset.radiance(Vec3::Y);
        assert!(zenith.z > zenith.x);
        let night = luminance(LightingPreset::Night.radiance(Vec3::Y));
        for preset in [
            LightingPreset::Overcast,
            LightingPreset::Sunset,
            LightingPreset::BrightOutdoor,
        ] {
            assert!(luminance(preset.radiance(Vec3::Y)) > night * 5.0);
        }
        // The studio key light is brighter than its fill.
        let key = LightingPreset::Studio.radiance(direction(35f32.to_radians(), -40f32.to_radians()));
        let fill = LightingPreset::Studio.radiance(direction(15f32.to_radians(), 65f32.to_radians()));
        assert!(luminance(key) > luminance(fill) * 2.0);

        // Rotation turns a sun about the vertical axis only.
        let sun = LightingPreset::BrightOutdoor.sun().unwrap().direction;
        let turned = rotate_about_up(sun, 1.0);
        assert!((turned.y - sun.y).abs() < 1e-6);
        assert!((turned.length() - 1.0).abs() < 1e-5);
        assert!((rotate_about_up(Vec3::Z, std::f32::consts::FRAC_PI_2) - Vec3::X).length() < 1e-6);
    }
}
