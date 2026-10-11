//! Bitmap resources (imported textures) and how layers project them onto
//! the mesh.

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::{MeshMapSample, ResourceId};

/// Largest accepted bitmap edge, in pixels.
pub const MAX_RESOURCE_DIMENSION: u32 = 8_192;
pub const MIN_PROJECTION_SCALE: f32 = 0.01;
pub const MAX_PROJECTION_SCALE: f32 = 256.0;

/// A bitmap referenced by the project. The pixels live in the external
/// `source` file and are loaded into [`ResourceImages`] at runtime, like the
/// project mesh.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectResource {
    pub id: ResourceId,
    pub name: String,
    pub source: String,
}

/// Decoded straight-alpha RGBA8 pixels, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Loaded pixels of project resources, shared cheaply with bake threads.
pub type ResourceImages = BTreeMap<ResourceId, Arc<TextureImage>>;

impl TextureImage {
    /// Returns `None` unless the buffer matches the dimensions and both
    /// edges are within `1..=MAX_RESOURCE_DIMENSION`.
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Option<Self> {
        let valid = (1..=MAX_RESOURCE_DIMENSION).contains(&width)
            && (1..=MAX_RESOURCE_DIMENSION).contains(&height)
            && rgba8.len() == width as usize * height as usize * 4;
        valid.then_some(Self {
            width,
            height,
            rgba8,
        })
    }

    fn texel(&self, x: i64, y: i64, wrap: bool) -> Vec4 {
        let (width, height) = (i64::from(self.width), i64::from(self.height));
        let (x, y) = if wrap {
            (x.rem_euclid(width), y.rem_euclid(height))
        } else {
            (x.clamp(0, width - 1), y.clamp(0, height - 1))
        };
        let index = (y as usize * self.width as usize + x as usize) * 4;
        self.rgba8
            .get(index..index + 4)
            .map_or(Vec4::ZERO, |pixel| {
                Vec4::new(
                    f32::from(pixel[0]),
                    f32::from(pixel[1]),
                    f32::from(pixel[2]),
                    f32::from(pixel[3]),
                ) / 255.0
            })
    }

    /// Bilinear sample with `+V` up (row 0 is `v = 1`); channels in `0..=1`.
    /// `wrap` tiles the image, otherwise edges clamp.
    pub fn sample(&self, uv: Vec2, wrap: bool) -> Vec4 {
        if !uv.is_finite() {
            return Vec4::ZERO;
        }
        let x = uv.x * self.width as f32 - 0.5;
        let y = (1.0 - uv.y) * self.height as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let top = self
            .texel(x0, y0, wrap)
            .lerp(self.texel(x0 + 1, y0, wrap), fx);
        let bottom = self
            .texel(x0, y0 + 1, wrap)
            .lerp(self.texel(x0 + 1, y0 + 1, wrap), fx);
        top.lerp(bottom, fy)
    }
}

/// Grayscale value of a sample: luminance weighted by alpha-free RGB.
pub(crate) fn luminance(sample: Vec4) -> f32 {
    sample.truncate().dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMode {
    /// The mesh's own UVs.
    #[default]
    Uv,
    /// Three planar projections along the object axes, blended by the
    /// surface normal; needs mesh maps and ignores UV seams.
    Triplanar,
}

impl ProjectionMode {
    pub const ALL: [Self; 2] = [Self::Uv, Self::Triplanar];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Uv => "UV",
            Self::Triplanar => "Triplanar",
        }
    }
}

/// How a bitmap is placed on the surface.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureProjection {
    pub mode: ProjectionMode,
    /// Repetitions across the UV square, or across the mesh bounds for
    /// triplanar.
    pub scale: f32,
    pub offset: Vec2,
    pub rotation_degrees: f32,
    /// Higher values sharpen the triplanar transitions.
    pub blend_sharpness: f32,
}

impl Default for TextureProjection {
    fn default() -> Self {
        Self {
            mode: ProjectionMode::Uv,
            scale: 1.0,
            offset: Vec2::ZERO,
            rotation_degrees: 0.0,
            blend_sharpness: 4.0,
        }
    }
}

impl TextureProjection {
    pub(crate) fn is_valid(self) -> bool {
        self.scale.is_finite()
            && (MIN_PROJECTION_SCALE..=MAX_PROJECTION_SCALE).contains(&self.scale)
            && self.offset.is_finite()
            && self.rotation_degrees.is_finite()
            && self.blend_sharpness.is_finite()
            && (1.0..=64.0).contains(&self.blend_sharpness)
    }

    pub fn needs_mesh_maps(self) -> bool {
        self.mode == ProjectionMode::Triplanar
    }

    fn transform(self, coordinate: Vec2) -> Vec2 {
        let (sin, cos) = self.rotation_degrees.to_radians().sin_cos();
        let centered = coordinate - Vec2::splat(0.5);
        let rotated = Vec2::new(
            centered.x * cos - centered.y * sin,
            centered.x * sin + centered.y * cos,
        );
        (rotated + Vec2::splat(0.5)) * self.scale + self.offset
    }

    /// Samples `image` for the texel at `uv`. Triplanar reads position and
    /// normal from `sample`; without it the UV projection is used.
    pub(crate) fn sample(
        self,
        image: &TextureImage,
        uv: Vec2,
        sample: Option<&MeshMapSample>,
    ) -> Vec4 {
        match (self.mode, sample) {
            (ProjectionMode::Triplanar, Some(sample)) => {
                let position = sample.relative_position;
                let weights = sample.normal.abs().powf(self.blend_sharpness);
                let total = weights.x + weights.y + weights.z;
                if total <= f32::EPSILON {
                    return image.sample(self.transform(position.truncate()), true);
                }
                let weights = weights / total;
                let mut result = Vec4::ZERO;
                for (weight, plane) in [
                    (weights.x, Vec2::new(position.z, position.y)),
                    (weights.y, Vec2::new(position.x, position.z)),
                    (weights.z, Vec2::new(position.x, position.y)),
                ] {
                    if weight > 0.001 {
                        result += image.sample(self.transform(plane), true) * weight;
                    }
                }
                result
            }
            _ => image.sample(self.transform(uv), true),
        }
    }
}

/// A grayscale bitmap that multiplies a layer's mask.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BitmapMask {
    pub resource: ResourceId,
    #[serde(default)]
    pub projection: TextureProjection,
    #[serde(default)]
    pub invert: bool,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn checker() -> TextureImage {
        // 2x2: white, black / black, white.
        TextureImage::new(
            2,
            2,
            vec![
                255, 255, 255, 255, 0, 0, 0, 255, //
                0, 0, 0, 255, 255, 255, 255, 255,
            ],
        )
        .unwrap()
    }

    #[test]
    fn tc_0070_bitmap_samples_bilinearly_with_v_up_and_tiles() {
        let image = checker();
        // Texel centres: row 0 is the top (v near 1).
        assert_eq!(image.sample(Vec2::new(0.25, 0.75), false).x, 1.0);
        assert_eq!(image.sample(Vec2::new(0.75, 0.75), false).x, 0.0);
        assert_eq!(image.sample(Vec2::new(0.25, 0.25), false).x, 0.0);
        // Halfway between texels blends.
        assert!((image.sample(Vec2::new(0.5, 0.75), false).x - 0.5).abs() < 1e-5);
        // Wrapping repeats the image.
        assert_eq!(
            image.sample(Vec2::new(1.25, 0.75), true),
            image.sample(Vec2::new(0.25, 0.75), true)
        );
        assert!(TextureImage::new(2, 2, vec![0; 3]).is_none());
    }

    #[test]
    fn tc_0071_projection_scale_tiles_and_triplanar_follows_the_normal() {
        let image = checker();
        let tiled = TextureProjection {
            scale: 2.0,
            ..TextureProjection::default()
        };
        // Scale 2 maps uv 0.125 to 0.25 of the image.
        assert_eq!(
            tiled.sample(&image, Vec2::new(0.125, 0.375), None),
            image.sample(Vec2::new(0.25, 0.75), true)
        );

        let triplanar = TextureProjection {
            mode: ProjectionMode::Triplanar,
            blend_sharpness: 64.0,
            ..TextureProjection::default()
        };
        let facing_up = MeshMapSample {
            covered: true,
            relative_position: Vec3::new(0.25, 0.9, 0.75),
            normal: Vec3::Y,
            ..MeshMapSample::default()
        };
        // An upward normal projects along Y: the XZ plane.
        assert_eq!(
            triplanar.sample(&image, Vec2::ZERO, Some(&facing_up)),
            image.sample(Vec2::new(0.25, 0.75), true)
        );
    }
}
