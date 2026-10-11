//! Bitmap resources (imported textures) and how layers project them onto
//! the mesh.

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::color::ColorSpace;
use crate::{Axis, MeshMapSample, ResourceId};

/// Largest accepted bitmap edge, in pixels.
pub const MAX_RESOURCE_DIMENSION: u32 = 8_192;
pub const MIN_PROJECTION_SCALE: f32 = 0.01;
pub const MAX_PROJECTION_SCALE: f32 = 256.0;
/// Physical tile sizes, in object units (metres after import scale).
pub const MIN_TILE_SIZE: f32 = 0.001;
pub const MAX_TILE_SIZE: f32 = 1000.0;

/// A bitmap referenced by the project. The pixels live in the external
/// `source` file and are loaded into [`ResourceImages`] at runtime, like the
/// project mesh.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectResource {
    pub id: ResourceId,
    pub name: String,
    pub source: String,
    /// Encoding of the pixels; colour channels convert from it.
    #[serde(default)]
    pub color_space: ColorSpace,
    /// Who made the file and under which terms it may be used.
    #[serde(default, skip_serializing_if = "ResourceProvenance::is_empty")]
    pub provenance: ResourceProvenance,
    /// Free-form search keywords.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl ProjectResource {
    /// A resource with sRGB pixels, no provenance, and no tags.
    pub fn new(id: ResourceId, name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            source: source.into(),
            color_space: ColorSpace::default(),
            provenance: ResourceProvenance::default(),
            tags: Vec::new(),
        }
    }
}

/// Origin and licensing of an asset, recorded so a project can show what
/// it may redistribute.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourceProvenance {
    pub author: String,
    /// Licence name or SPDX identifier, e.g. `CC0-1.0`.
    pub license: String,
    /// Where the file came from (URL or description).
    pub origin: String,
}

impl ResourceProvenance {
    pub fn is_empty(&self) -> bool {
        self.author.is_empty() && self.license.is_empty() && self.origin.is_empty()
    }
}

/// Decoded straight-alpha RGBA8 pixels, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    /// Encoding of the pixels, from the resource tag.
    pub color_space: ColorSpace,
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
            color_space: ColorSpace::Srgb,
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
    /// One flat projection along `axis`, like a slide projector.
    Planar,
    /// Wraps around `axis`: U follows the angle, V the height.
    Cylindrical,
    /// Wraps around the bounds centre: U is longitude, V latitude about
    /// `axis`.
    Spherical,
    /// A bounded projection placed on the surface (see [`DecalPlacement`]);
    /// outside its box the layer is clipped.
    Decal,
}

impl ProjectionMode {
    pub const ALL: [Self; 6] = [
        Self::Uv,
        Self::Triplanar,
        Self::Planar,
        Self::Cylindrical,
        Self::Spherical,
        Self::Decal,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Uv => "UV",
            Self::Triplanar => "Triplanar",
            Self::Planar => "Planar",
            Self::Cylindrical => "Cylinder",
            Self::Spherical => "Sphere",
            Self::Decal => "Decal",
        }
    }
}

/// Where a decal sits: a box on the surface in object space. Its image
/// covers `size` across the surface, reaches `depth` in front of and behind
/// the placement plane, and only lands on surfaces facing it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecalPlacement {
    pub center: Vec3,
    /// Direction the decal faces (out of the surface).
    pub normal: Vec3,
    /// Direction of the image's +V on the surface.
    pub up: Vec3,
    /// Width and height in object units.
    pub size: Vec2,
    /// Half-thickness of the projection box in object units.
    pub depth: f32,
    /// Sub-rectangle of the image to use (an atlas cell), in image UVs.
    pub region_min: Vec2,
    pub region_max: Vec2,
}

impl Default for DecalPlacement {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            normal: Vec3::Z,
            up: Vec3::Y,
            size: Vec2::splat(0.25),
            depth: 0.1,
            region_min: Vec2::ZERO,
            region_max: Vec2::ONE,
        }
    }
}

impl DecalPlacement {
    fn is_valid(self) -> bool {
        let unit = |value: Vec2| {
            value.is_finite() && value.cmpge(Vec2::ZERO).all() && value.cmple(Vec2::ONE).all()
        };
        self.center.is_finite()
            && self.normal.is_finite()
            && self.normal.length_squared() > 1e-8
            && self.up.is_finite()
            && self.up.cross(self.normal).length_squared() > 1e-8
            && self.size.is_finite()
            && self.size.cmpgt(Vec2::ZERO).all()
            && self.depth.is_finite()
            && self.depth > 0.0
            && unit(self.region_min)
            && unit(self.region_max)
            && self.region_max.cmpgt(self.region_min).all()
    }

    /// Image UV and coverage (`0` outside the box or on back-facing
    /// surfaces) of a surface point.
    fn locate(self, position: Vec3, surface_normal: Vec3) -> (Vec2, f32) {
        let normal = self.normal.normalize_or(Vec3::Z);
        let right = self.up.cross(normal).normalize_or(Vec3::X);
        let up = normal.cross(right);
        let local = position - self.center;
        let unit = Vec2::new(
            local.dot(right) / self.size.x + 0.5,
            local.dot(up) / self.size.y + 0.5,
        );
        let inside = unit.cmpge(Vec2::ZERO).all()
            && unit.cmple(Vec2::ONE).all()
            && local.dot(normal).abs() <= self.depth;
        // Fade out on surfaces turning away so the decal does not smear
        // along walls parallel to its direction.
        let facing = surface_normal.dot(normal);
        let coverage = if inside {
            ((facing - 0.05) / 0.25).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (
            self.region_min + unit * (self.region_max - self.region_min),
            coverage,
        )
    }
}

/// How a bitmap is placed on the surface.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureProjection {
    pub mode: ProjectionMode,
    /// Repetitions across the UV square, or across the mesh bounds for
    /// object-space projections. Ignored when `tile_size` is set.
    pub scale: f32,
    pub offset: Vec2,
    pub rotation_degrees: f32,
    /// Higher values sharpen the triplanar transitions.
    pub blend_sharpness: f32,
    /// Projection axis of the planar, cylindrical, and spherical modes.
    #[serde(default)]
    pub axis: Axis,
    /// Physical size: one repetition spans this many object units (metres
    /// after import scale) in the object-space modes, so the texture keeps
    /// its real-world size whatever the mesh bounds.
    #[serde(default)]
    pub tile_size: Option<f32>,
    #[serde(default)]
    pub decal: DecalPlacement,
}

impl Default for TextureProjection {
    fn default() -> Self {
        Self {
            mode: ProjectionMode::Uv,
            scale: 1.0,
            offset: Vec2::ZERO,
            rotation_degrees: 0.0,
            blend_sharpness: 4.0,
            axis: Axis::default(),
            tile_size: None,
            decal: DecalPlacement::default(),
        }
    }
}

/// The two coordinates of `position` perpendicular to `axis`.
fn across(position: Vec3, axis: Axis) -> Vec2 {
    match axis {
        Axis::X => Vec2::new(position.z, position.y),
        Axis::Y => Vec2::new(position.x, position.z),
        Axis::Z => Vec2::new(position.x, position.y),
    }
}

fn along(position: Vec3, axis: Axis) -> f32 {
    match axis {
        Axis::X => position.x,
        Axis::Y => position.y,
        Axis::Z => position.z,
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
            && self.tile_size.is_none_or(|size| {
                size.is_finite() && (MIN_TILE_SIZE..=MAX_TILE_SIZE).contains(&size)
            })
            && (self.mode != ProjectionMode::Decal || self.decal.is_valid())
    }

    /// True for the plain UV mapping: no tiling, offset, or rotation.
    pub(crate) fn is_identity_uv(self) -> bool {
        self.mode == ProjectionMode::Uv
            && self.scale == 1.0
            && self.offset == Vec2::ZERO
            && self.rotation_degrees.rem_euclid(360.0) == 0.0
    }

    /// True when the projection reads surface position and normal.
    pub fn needs_mesh_maps(self) -> bool {
        self.mode != ProjectionMode::Uv
    }

    fn transform(self, coordinate: Vec2) -> Vec2 {
        let (sin, cos) = self.rotation_degrees.to_radians().sin_cos();
        let centered = coordinate - Vec2::splat(0.5);
        let rotated = Vec2::new(
            centered.x * cos - centered.y * sin,
            centered.x * sin + centered.y * cos,
        );
        let scale = if self.tile_size.is_some() {
            1.0
        } else {
            self.scale
        };
        (rotated + Vec2::splat(0.5)) * scale + self.offset
    }

    /// Object-space coordinates measured in repetitions: physical units
    /// divided by the tile size, or the bounds-relative position.
    fn object_coordinates(self, sample: &MeshMapSample) -> Vec3 {
        match self.tile_size {
            Some(size) => sample.position / size,
            None => sample.relative_position,
        }
    }

    /// How much of the layer this projection leaves visible at a texel:
    /// decals clip everything outside their box, other modes cover all.
    pub(crate) fn coverage(self, sample: Option<&MeshMapSample>) -> f32 {
        match (self.mode, sample) {
            (ProjectionMode::Decal, Some(sample)) if sample.covered => {
                self.decal.locate(sample.position, sample.normal).1
            }
            (ProjectionMode::Decal, _) => 0.0,
            _ => 1.0,
        }
    }

    /// Samples `image` for the texel at `uv`. Object-space modes read
    /// position and normal from `sample`; without it the UV projection is
    /// used.
    pub(crate) fn sample(
        self,
        image: &TextureImage,
        uv: Vec2,
        sample: Option<&MeshMapSample>,
    ) -> Vec4 {
        let Some(sample) = sample.filter(|_| self.mode != ProjectionMode::Uv) else {
            return image.sample(self.transform(uv), true);
        };
        match self.mode {
            ProjectionMode::Uv => image.sample(self.transform(uv), true),
            ProjectionMode::Triplanar => {
                let position = self.object_coordinates(sample);
                let weights = sample.normal.abs().powf(self.blend_sharpness);
                let total = weights.x + weights.y + weights.z;
                if total <= f32::EPSILON {
                    return image.sample(self.transform(position.truncate()), true);
                }
                let weights = weights / total;
                let mut result = Vec4::ZERO;
                for (weight, axis) in [
                    (weights.x, Axis::X),
                    (weights.y, Axis::Y),
                    (weights.z, Axis::Z),
                ] {
                    if weight > 0.001 {
                        result +=
                            image.sample(self.transform(across(position, axis)), true) * weight;
                    }
                }
                result
            }
            ProjectionMode::Planar => image.sample(
                self.transform(across(self.object_coordinates(sample), self.axis)),
                true,
            ),
            ProjectionMode::Cylindrical => {
                let centered = sample.unit_position - Vec3::splat(0.5);
                let around = across(centered, self.axis);
                let angle = around.y.atan2(around.x) / std::f32::consts::TAU + 0.5;
                let height = along(self.object_coordinates(sample), self.axis);
                image.sample(self.transform(Vec2::new(angle, height)), true)
            }
            ProjectionMode::Spherical => {
                let direction = (sample.unit_position - Vec3::splat(0.5)).normalize_or(Vec3::Y);
                let around = across(direction, self.axis);
                let longitude = around.y.atan2(around.x) / std::f32::consts::TAU + 0.5;
                let latitude = along(direction, self.axis).clamp(-1.0, 1.0).asin()
                    / std::f32::consts::PI
                    + 0.5;
                image.sample(self.transform(Vec2::new(longitude, latitude)), true)
            }
            ProjectionMode::Decal => {
                let (image_uv, coverage) = self.decal.locate(sample.position, sample.normal);
                if coverage <= 0.0 {
                    return Vec4::ZERO;
                }
                let texel = image.sample(image_uv, false);
                texel.truncate().extend(texel.w * coverage)
            }
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

    #[test]
    fn tc_0130_planar_cylindrical_spherical_physical_and_decal_projections() {
        let image = checker();
        let point = |relative: Vec3, unit: Vec3, position: Vec3, normal: Vec3| MeshMapSample {
            covered: true,
            relative_position: relative,
            unit_position: unit,
            position,
            normal,
            ..MeshMapSample::default()
        };
        let surface = point(
            Vec3::new(0.25, 0.4, 0.75),
            Vec3::new(1.0, 0.5, 0.5),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::X,
        );

        // Planar along Z reads XY regardless of the surface normal.
        let planar = TextureProjection {
            mode: ProjectionMode::Planar,
            axis: Axis::Z,
            ..TextureProjection::default()
        };
        assert_eq!(
            planar.sample(&image, Vec2::ZERO, Some(&surface)),
            image.sample(Vec2::new(0.25, 0.4), true)
        );

        // Cylinder around Y: a point on +X sits at angle 0 (U = 0.5); V is
        // the relative height.
        let cylinder = TextureProjection {
            mode: ProjectionMode::Cylindrical,
            ..TextureProjection::default()
        };
        assert_eq!(
            cylinder.sample(&image, Vec2::ZERO, Some(&surface)),
            image.sample(Vec2::new(0.5, 0.4), true)
        );
        // Sphere around Y: the +X equator is longitude 0.5, latitude 0.5.
        let sphere = TextureProjection {
            mode: ProjectionMode::Spherical,
            ..TextureProjection::default()
        };
        assert_eq!(
            sphere.sample(&image, Vec2::ZERO, Some(&surface)),
            image.sample(Vec2::new(0.5, 0.5), true)
        );

        // Physical size: a 0.5-unit tile repeats every half unit, whatever
        // the bounds-relative position says.
        let physical = TextureProjection {
            tile_size: Some(0.5),
            scale: 7.0,
            ..planar
        };
        assert_eq!(
            physical.sample(&image, Vec2::ZERO, Some(&surface)),
            image.sample(Vec2::new(2.0, 4.0), true)
        );
        assert!(
            !TextureProjection {
                tile_size: Some(0.0),
                ..physical
            }
            .is_valid()
        );

        // Decal: inside the box it maps into its atlas region; outside, or
        // on a surface facing away, coverage is zero.
        let decal = TextureProjection {
            mode: ProjectionMode::Decal,
            decal: DecalPlacement {
                center: Vec3::new(0.0, 0.0, 0.5),
                normal: Vec3::Z,
                up: Vec3::Y,
                size: Vec2::splat(0.5),
                depth: 0.1,
                region_min: Vec2::new(0.5, 0.5),
                region_max: Vec2::ONE,
            },
            ..TextureProjection::default()
        };
        assert!(decal.is_valid());
        let front = |position: Vec3, normal: Vec3| point(Vec3::ZERO, Vec3::ZERO, position, normal);
        let center = front(Vec3::new(0.0, 0.0, 0.5), Vec3::Z);
        assert_eq!(decal.coverage(Some(&center)), 1.0);
        // The box centre lands in the middle of the top-right atlas cell.
        assert_eq!(
            decal.sample(&image, Vec2::ZERO, Some(&center)),
            image.sample(Vec2::new(0.75, 0.75), false)
        );
        assert_eq!(
            decal.coverage(Some(&front(Vec3::new(0.4, 0.0, 0.5), Vec3::Z))),
            0.0
        );
        assert_eq!(
            decal.coverage(Some(&front(Vec3::new(0.0, 0.0, 0.8), Vec3::Z))),
            0.0
        );
        assert_eq!(
            decal.coverage(Some(&front(Vec3::new(0.0, 0.0, 0.5), Vec3::NEG_Z))),
            0.0
        );
        assert_eq!(decal.coverage(None), 0.0);
        let degenerate = TextureProjection {
            decal: DecalPlacement {
                up: Vec3::Z,
                ..decal.decal
            },
            ..decal
        };
        assert!(!degenerate.is_valid());
    }

    #[test]
    fn tc_0131_decal_fills_clip_uniform_channels_to_the_decal_box() {
        use crate::{
            BakeContext, LayerFill, LayerId, MaterialChannel, MaterialLayer, MeshMapSettings,
            PrimitivePreset, TextureResolution, TextureSet, TextureSetId, bake_mesh_maps,
        };
        let maps = bake_mesh_maps(
            &PrimitivePreset::Cube.build_mesh(),
            &MeshMapSettings {
                resolution: 64,
                ao_samples: 2,
                ..MeshMapSettings::default()
            },
            None,
        )
        .unwrap();
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Sticker",
            LayerFill {
                roughness: 0.0,
                projection: TextureProjection {
                    mode: ProjectionMode::Decal,
                    decal: DecalPlacement {
                        center: Vec3::new(0.0, 0.0, 0.5),
                        size: Vec2::splat(0.5),
                        ..DecalPlacement::default()
                    },
                    ..TextureProjection::default()
                },
                ..LayerFill::default()
            },
        );
        layer.channels = [MaterialChannel::Roughness].into();
        let set = TextureSet {
            id: TextureSetId(1),
            name: "Cube".to_owned(),
            material_slot: "Cube".to_owned(),
            resolution: TextureResolution {
                width: 64,
                height: 64,
            },
            channels: [MaterialChannel::Roughness].into(),
            layers: vec![layer],
            height_strength: crate::DEFAULT_HEIGHT_STRENGTH,
        };
        assert!(set.requires_mesh_maps());
        let roughness = set
            .bake_channel(
                MaterialChannel::Roughness,
                BakeContext {
                    mesh_maps: Some(&maps),
                    ..BakeContext::default()
                },
            )
            .unwrap();
        let painted = roughness
            .rgba8
            .chunks(4)
            .filter(|texel| texel[0] == 0)
            .count();
        let total = roughness.rgba8.len() / 4;
        // A quarter of one face out of six: about 4% of the UV square.
        assert!(
            painted > total / 60 && painted < total / 12,
            "{painted} of {total}"
        );
    }
}
