//! Layer-stack compositing: turns a texture set's layer tree into baked
//! channel textures.
//!
//! Every layer renders into its own straight-alpha buffer (fill, then paint
//! and erase strokes, or for a group its composited children), runs its
//! filters, and then blends over the stack beneath through its opacity and
//! mask.

use std::collections::BTreeMap;

use glam::{Vec2, Vec3, Vec4};
use thiserror::Error;

use crate::AnchorMask;
use crate::brush::{self, StrokePath};
use crate::filters::{self, PlaneData, PlaneGrid};
use crate::region::{DIRTY_TILE_SIZE, TexelRect};
use crate::resources::{ProjectionMode, ResourceImages, TextureImage, luminance};
use crate::{
    DEFAULT_BASE_COLOR, DEFAULT_HEIGHT, DEFAULT_METALLIC, DEFAULT_OPACITY, DEFAULT_ROUGHNESS,
    LayerFill, LayerId, MAX_BAKE_PIXELS, MAX_TEXTURE_DIMENSION, MAX_TEXTURE_SET_PIXELS,
    MaterialChannel, MaterialLayer, MeshGeometry, MeshMaps, PaintStroke, ResourceId,
    TextureResolution, TextureSet, raster, valid_height_strength, valid_layer_strokes,
};

/// With strength 1, a full 0..1 height step across this fraction of a
/// 100-texel span tilts the normal by 45 degrees; slopes scale with
/// resolution so bumps keep their look in UV space.
const HEIGHT_SLOPE_REFERENCE_TEXELS: f32 = 100.0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BakedTexture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl BakedTexture {
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = ((y as usize * self.width as usize) + x as usize) * 4;
        let rgba = self.rgba8.get(index..index + 4)?;
        Some([rgba[0], rgba[1], rgba[2], rgba[3]])
    }

    pub fn sample_uv(&self, uv: glam::Vec2) -> Option<[u8; 4]> {
        if !uv.is_finite() || !(0.0..=1.0).contains(&uv.x) || !(0.0..=1.0).contains(&uv.y) {
            return None;
        }
        let x = (uv.x * self.width.saturating_sub(1) as f32).round() as u32;
        let y = ((1.0 - uv.y) * self.height.saturating_sub(1) as f32).round() as u32;
        self.pixel(x, y)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BakeError {
    #[error("texture bake resolution {width}x{height} exceeds the pixel budget")]
    ResolutionExceedsBudget { width: u32, height: u32 },
    #[error("unable to allocate texture bake buffer")]
    AllocationFailed,
    #[error("texture set contains invalid or excessive paint strokes")]
    InvalidStrokeData,
    #[error("texture set contains invalid layer fill, generator, filter, or mask settings")]
    InvalidLayerSettings,
    #[error("the {0:?} channel cannot be baked yet")]
    UnsupportedChannel(MaterialChannel),
    #[error(
        "a layer uses a mask generator or triplanar projection but no mesh maps have been baked"
    )]
    MissingMeshMaps,
    #[error("a layer uses a polygon mask but no mesh geometry was provided")]
    MissingMesh,
    #[error("bitmap resource {0:?} is not loaded")]
    MissingResource(ResourceId),
    #[error(transparent)]
    InvalidExportPreset(#[from] crate::export::ExportPresetError),
    #[error("a region rebake needs a target the size of the texture set and a region inside it")]
    RegionMismatch,
}

/// Data a texture bake may need: geometry for polygon masks, baked mesh
/// maps for generators and triplanar projection, and loaded bitmaps. Layers
/// that need none of them bake without.
#[derive(Clone, Copy, Debug, Default)]
pub struct BakeContext<'a> {
    pub mesh: Option<&'a MeshGeometry>,
    pub mesh_maps: Option<&'a MeshMaps>,
    pub resources: Option<&'a ResourceImages>,
}

impl<'a> BakeContext<'a> {
    fn image(&self, id: ResourceId) -> Result<&'a TextureImage, BakeError> {
        self.resources
            .and_then(|images| images.get(&id))
            .map(|image| &**image)
            .ok_or(BakeError::MissingResource(id))
    }
}

/// Channels [`TextureSet::flatten_layer`] bakes.
pub const FLATTEN_CHANNELS: [MaterialChannel; 7] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Height,
    MaterialChannel::Normal,
    MaterialChannel::Emissive,
    MaterialChannel::Opacity,
];

/// A layer flattened into one straight-alpha bitmap per channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlattenedLayer {
    pub channels: Vec<(MaterialChannel, TextureImage)>,
    /// The source layer's blend mode, which the replacement keeps.
    pub blend_mode: BlendMode,
}

impl FlattenedLayer {
    /// A fill layer that reproduces the flattened layer from the given
    /// resources, one per entry of `channels` and in the same order.
    pub fn to_fill_layer(
        &self,
        id: LayerId,
        name: impl Into<String>,
        resources: &[ResourceId],
    ) -> MaterialLayer {
        let mut fill = LayerFill::default();
        let mut channels = std::collections::BTreeSet::new();
        for ((channel, _), resource) in self.channels.iter().zip(resources) {
            if let Some(slot) = fill.textures.slot_mut(*channel) {
                *slot = Some(*resource);
                channels.insert(*channel);
            }
        }
        let mut layer = MaterialLayer::new_fill(id, name, fill);
        layer.channels = channels;
        layer.blend_mode = self.blend_mode;
        layer
    }
}

/// Viewport-ready textures for one texture set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialTextures {
    /// sRGB Base Color composited over [`DEFAULT_BASE_COLOR`], with the
    /// Opacity channel in alpha.
    pub base_color: BakedTexture,
    /// Linear packed R = ambient occlusion (from mesh maps), G = roughness,
    /// B = metallic.
    pub orm: BakedTexture,
    /// Tangent-space normal (OpenGL convention): baked detail, Normal-channel
    /// layers, and height-derived detail combined.
    pub normal: BakedTexture,
    /// Opaque sRGB emitted colour; black where nothing glows.
    pub emissive: BakedTexture,
}

/// Every material channel at one surface point, as the eyedropper picks it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialSample {
    /// sRGB Base Color.
    pub color: [u8; 3],
    pub opacity: f32,
    pub occlusion: f32,
    pub roughness: f32,
    pub metallic: f32,
    /// sRGB emitted colour.
    pub emissive: [u8; 3],
}

impl MaterialTextures {
    /// Reads every channel at `uv`; `None` outside the unit square.
    pub fn sample(&self, uv: glam::Vec2) -> Option<MaterialSample> {
        let base = self.base_color.sample_uv(uv)?;
        let orm = self.orm.sample_uv(uv)?;
        let emissive = self.emissive.sample_uv(uv)?;
        Some(MaterialSample {
            color: [base[0], base[1], base[2]],
            opacity: byte_unit(base[3]),
            occlusion: byte_unit(orm[0]),
            roughness: byte_unit(orm[1]),
            metallic: byte_unit(orm[2]),
            emissive: [emissive[0], emissive[1], emissive[2]],
        })
    }

    /// True when any texel emits light.
    pub fn has_emission(&self) -> bool {
        self.emissive
            .rgba8
            .as_chunks::<4>()
            .0
            .iter()
            .any(|pixel| pixel[..3] != [0, 0, 0])
    }

    /// True when any texel is not fully opaque.
    pub fn has_transparency(&self) -> bool {
        self.base_color
            .rgba8
            .as_chunks::<4>()
            .0
            .iter()
            .any(|pixel| pixel[3] != 255)
    }
}

/// Final per-texel masks of the layers being composited; absent layers are
/// fully open.
pub(crate) struct LayerMasks(BTreeMap<LayerId, Vec<f32>>);

pub(crate) fn plane_data(channel: MaterialChannel) -> PlaneData {
    match channel {
        MaterialChannel::BaseColor | MaterialChannel::Emissive | MaterialChannel::Id => {
            PlaneData::Color
        }
        MaterialChannel::Normal => PlaneData::Vector,
        _ => PlaneData::Scalar,
    }
}

/// Texel of a channel no layer has painted.
pub(crate) fn base_texel(channel: MaterialChannel) -> Result<Vec4, BakeError> {
    let scalar = |value: f32| Vec4::new(value, value, value, 1.0);
    Ok(match channel {
        MaterialChannel::BaseColor => Vec4::ZERO,
        MaterialChannel::Emissive => Vec4::new(0.0, 0.0, 0.0, 1.0),
        MaterialChannel::Roughness => scalar(DEFAULT_ROUGHNESS),
        MaterialChannel::Metallic => scalar(DEFAULT_METALLIC),
        MaterialChannel::Height => scalar(DEFAULT_HEIGHT),
        MaterialChannel::Opacity => scalar(DEFAULT_OPACITY),
        MaterialChannel::Normal => Vec4::new(0.0, 0.0, 1.0, 1.0),
        other => return Err(BakeError::UnsupportedChannel(other)),
    })
}

fn byte_unit(value: u8) -> f32 {
    f32::from(value) / 255.0
}

pub(crate) fn rgb_unit(rgb: [u8; 3]) -> Vec3 {
    Vec3::new(byte_unit(rgb[0]), byte_unit(rgb[1]), byte_unit(rgb[2]))
}

/// How a layer combines with the stack beneath it. Colour and scalar
/// channels blend per component with the standard separable formulas;
/// Normal-channel (vector) data always uses Normal so normals stay unit
/// length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    SoftLight,
    Darken,
    Lighten,
    Difference,
    /// Linear dodge: sums values, clamped to 1.
    Add,
    Subtract,
    /// Groups only: children blend straight onto the stack beneath, as if
    /// ungrouped, and the group's opacity, mask, and filters fade that
    /// result in.
    PassThrough,
}

impl BlendMode {
    pub const ALL: [Self; 10] = [
        Self::Normal,
        Self::Multiply,
        Self::Screen,
        Self::Overlay,
        Self::SoftLight,
        Self::Darken,
        Self::Lighten,
        Self::Difference,
        Self::Add,
        Self::Subtract,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Multiply => "Multiply",
            Self::Screen => "Screen",
            Self::Overlay => "Overlay",
            Self::SoftLight => "Soft light",
            Self::Darken => "Darken",
            Self::Lighten => "Lighten",
            Self::Difference => "Difference",
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::PassThrough => "Pass through",
        }
    }

    /// Blends one component: `backdrop` is the stack beneath, `source` the
    /// layer. Both in `0..=1`.
    pub fn apply(self, backdrop: f32, source: f32) -> f32 {
        let (b, s) = (backdrop, source);
        let result = match self {
            Self::Normal | Self::PassThrough => s,
            Self::Multiply => b * s,
            Self::Screen => b + s - b * s,
            Self::Overlay => {
                if b <= 0.5 {
                    2.0 * b * s
                } else {
                    1.0 - 2.0 * (1.0 - b) * (1.0 - s)
                }
            }
            Self::SoftLight => {
                if s <= 0.5 {
                    b - (1.0 - 2.0 * s) * b * (1.0 - b)
                } else {
                    let d = if b <= 0.25 {
                        ((16.0 * b - 12.0) * b + 4.0) * b
                    } else {
                        b.sqrt()
                    };
                    b + (2.0 * s - 1.0) * (d - b)
                }
            }
            Self::Darken => b.min(s),
            Self::Lighten => b.max(s),
            Self::Difference => (b - s).abs(),
            Self::Add => b + s,
            Self::Subtract => b - s,
        };
        result.clamp(0.0, 1.0)
    }
}

/// Straight-alpha "over": `value` at `alpha` on top of `destination`.
fn over(destination: &mut Vec4, value: Vec3, alpha: f32, data: PlaneData) {
    let alpha = alpha.clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return;
    }
    let out_alpha = alpha + destination.w * (1.0 - alpha);
    if out_alpha <= 1e-6 {
        *destination = Vec4::ZERO;
        return;
    }
    let mut blended =
        (value * alpha + destination.truncate() * destination.w * (1.0 - alpha)) / out_alpha;
    if data == PlaneData::Vector {
        blended = blended.normalize_or(Vec3::Z);
    }
    *destination = blended.extend(out_alpha);
}

/// Uniform value (and alpha) a fill writes to `channel`.
fn fill_texel(fill: &LayerFill, channel: MaterialChannel) -> Vec4 {
    let scalar = |value: f32| Vec4::new(value, value, value, 1.0);
    match channel {
        MaterialChannel::BaseColor => {
            rgb_unit([fill.color[0], fill.color[1], fill.color[2]]).extend(byte_unit(fill.color[3]))
        }
        MaterialChannel::Emissive => rgb_unit(fill.emissive).extend(1.0),
        MaterialChannel::Roughness => scalar(fill.roughness),
        MaterialChannel::Metallic => scalar(fill.metallic),
        MaterialChannel::Height => scalar(fill.height),
        MaterialChannel::Opacity => scalar(fill.material_opacity),
        _ => Vec4::new(0.0, 0.0, 1.0, 1.0),
    }
}

/// Converts a bitmap sample into a channel texel.
fn texture_texel(sample: Vec4, data: PlaneData, color_space: crate::ColorSpace) -> Vec4 {
    match data {
        // Colour channels store sRGB; convert tagged linear or ACEScg images.
        PlaneData::Color => color_space.to_srgb(sample.truncate()).extend(sample.w),
        PlaneData::Scalar => Vec3::splat(luminance(sample)).extend(sample.w),
        PlaneData::Vector => (sample.truncate() * 2.0 - Vec3::ONE)
            .normalize_or(Vec3::Z)
            .extend(sample.w),
    }
}

/// Value a paint stroke writes to `channel`, and its colour alpha.
fn stroke_value(stroke: &PaintStroke, channel: MaterialChannel) -> (Vec3, f32) {
    match channel {
        MaterialChannel::BaseColor => (
            rgb_unit([stroke.color[0], stroke.color[1], stroke.color[2]]),
            byte_unit(stroke.color[3]),
        ),
        MaterialChannel::Emissive => (rgb_unit(stroke.emissive), 1.0),
        MaterialChannel::Roughness => (Vec3::splat(stroke.roughness), 1.0),
        MaterialChannel::Metallic => (Vec3::splat(stroke.metallic), 1.0),
        MaterialChannel::Height => (Vec3::splat(stroke.height), 1.0),
        MaterialChannel::Opacity => (Vec3::splat(stroke.material_opacity), 1.0),
        _ => (Vec3::Z, 1.0),
    }
}

/// True when `layer` adds anything to one of `channels`.
pub(crate) fn contributes(layer: &MaterialLayer, channels: &[MaterialChannel]) -> bool {
    layer.visible
        && channels
            .iter()
            .any(|channel| layer.channels.contains(channel))
        && match &layer.children {
            Some(children) => children.iter().any(|child| contributes(child, channels)),
            None => layer.fill.is_some() || !layer.paint_strokes.is_empty(),
        }
}

pub(crate) fn unit_to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Encodes a unit tangent-space normal as OpenGL-convention RGBA (+Y = +V).
fn encode_normal(normal: Vec3) -> [u8; 4] {
    let encoded = normal * 0.5 + 0.5;
    [
        unit_to_byte(encoded.x),
        unit_to_byte(encoded.y),
        unit_to_byte(encoded.z),
        255,
    ]
}

fn plane_to_rgba8(plane: &[Vec4], data: PlaneData) -> Vec<u8> {
    plane
        .iter()
        .flat_map(|texel| match data {
            PlaneData::Color => [
                unit_to_byte(texel.x),
                unit_to_byte(texel.y),
                unit_to_byte(texel.z),
                unit_to_byte(texel.w),
            ],
            PlaneData::Scalar => {
                let gray = unit_to_byte(texel.x);
                [gray, gray, gray, 255]
            }
            PlaneData::Vector => encode_normal(texel.truncate()),
        })
        .collect()
}

/// Composites one channel of a texture set.
struct Compositor<'a> {
    set: &'a TextureSet,
    channel: MaterialChannel,
    data: PlaneData,
    context: BakeContext<'a>,
    masks: &'a LayerMasks,
    /// Texels composited; every buffer is row-major over this rectangle.
    region: TexelRect,
    /// Records each layer's values at one texel (the surface inspector).
    probe: Option<&'a crate::inspect::Probe>,
}

impl Compositor<'_> {
    fn width(&self) -> u32 {
        self.set.resolution.width
    }

    fn height(&self) -> u32 {
        self.set.resolution.height
    }

    fn grid(&self) -> PlaneGrid {
        PlaneGrid {
            region: self.region,
            width: self.width(),
            height: self.height(),
        }
    }

    /// Blends `layers` (bottom first) over `target`.
    fn composite(&self, layers: &[MaterialLayer], target: &mut [Vec4]) -> Result<(), BakeError> {
        for layer in layers {
            if !contributes(layer, &[self.channel]) {
                continue;
            }
            if layer.blend_mode == BlendMode::PassThrough
                && let Some(children) = &layer.children
            {
                self.pass_through(layer, children, target)?;
                continue;
            }
            let mut local = try_filled(self.region.len(), Vec4::ZERO)?;
            if let Some(children) = &layer.children {
                self.composite(children, &mut local)?;
            } else {
                if let Some(fill) = &layer.fill {
                    self.render_fill(fill, &mut local)?;
                }
                for stroke in &layer.paint_strokes {
                    self.render_stroke(stroke, &mut local);
                }
            }
            filters::apply_to_plane(&mut local, self.grid(), self.data, &layer.filters);
            let mask = self.masks.0.get(&layer.id);
            if let Some(probe) = self.probe {
                probe.record(layer, &local, mask.map(Vec::as_slice));
            }
            for (index, (destination, source)) in target.iter_mut().zip(&local).enumerate() {
                if source.w <= 0.0 {
                    continue;
                }
                let alpha = source.w * layer.opacity * mask.map_or(1.0, |values| values[index]);
                let mut value = source.truncate();
                if layer.blend_mode != BlendMode::Normal && self.data != PlaneData::Vector {
                    // W3C compositing: blend where the backdrop has coverage.
                    let backdrop = destination.truncate();
                    let blended = Vec3::new(
                        layer.blend_mode.apply(backdrop.x, value.x),
                        layer.blend_mode.apply(backdrop.y, value.y),
                        layer.blend_mode.apply(backdrop.z, value.z),
                    );
                    value = value.lerp(blended, destination.w.clamp(0.0, 1.0));
                }
                over(destination, value, alpha, self.data);
            }
        }
        Ok(())
    }

    /// Composites a pass-through group: its children go onto a copy of the
    /// stack, which then fades in over the stack by the group's opacity and
    /// mask.
    fn pass_through(
        &self,
        layer: &MaterialLayer,
        children: &[MaterialLayer],
        target: &mut [Vec4],
    ) -> Result<(), BakeError> {
        let mut through = try_filled(self.region.len(), Vec4::ZERO)?;
        through.copy_from_slice(target);
        self.composite(children, &mut through)?;
        filters::apply_to_plane(&mut through, self.grid(), self.data, &layer.filters);
        let mask = self.masks.0.get(&layer.id);
        if let Some(probe) = self.probe {
            probe.record(layer, &through, mask.map(Vec::as_slice));
        }
        for (index, (destination, passed)) in target.iter_mut().zip(&through).enumerate() {
            let amount = (layer.opacity * mask.map_or(1.0, |values| values[index])).clamp(0.0, 1.0);
            // Straight alpha: mix premultiplied colour, then divide back out.
            let alpha = destination.w + (passed.w - destination.w) * amount;
            let mut color = if alpha > 0.0 {
                (destination.truncate() * destination.w * (1.0 - amount)
                    + passed.truncate() * passed.w * amount)
                    / alpha
            } else {
                destination.truncate()
            };
            if self.data == PlaneData::Vector {
                color = color.normalize_or(Vec3::Z);
            }
            *destination = color.extend(alpha);
        }
        Ok(())
    }

    fn render_fill(&self, fill: &LayerFill, local: &mut [Vec4]) -> Result<(), BakeError> {
        let Some(resource) = fill.textures.for_channel(self.channel) else {
            let texel = fill_texel(fill, self.channel);
            if fill.projection.mode == ProjectionMode::Decal {
                // Uniform channels of a decal fill still stay inside its box.
                let maps = self.context.mesh_maps.ok_or(BakeError::MissingMeshMaps)?;
                let projection = fill.projection;
                parallel_fill_region(local, self.region, self.width(), self.height(), |uv| {
                    let coverage = projection.coverage(Some(&maps.sample(uv)));
                    texel.truncate().extend(texel.w * coverage)
                });
            } else {
                local.fill(texel);
            }
            return Ok(());
        };
        let image = self.context.image(resource)?;
        let maps = if fill.projection.needs_mesh_maps() {
            Some(self.context.mesh_maps.ok_or(BakeError::MissingMeshMaps)?)
        } else {
            None
        };
        let (projection, data) = (fill.projection, self.data);
        if projection.is_identity_uv()
            && image.width == self.width()
            && image.height == self.height()
        {
            // Same grid as the bake: copy texels instead of resampling, so
            // exact-size bitmaps (and flattened layers) stay sharp.
            let pixels = image.rgba8.as_chunks::<4>().0;
            let row = self.width() as usize;
            let span = self.region.width as usize;
            for (local_row, texels) in local.chunks_mut(span).enumerate() {
                let start = (self.region.y as usize + local_row) * row + self.region.x as usize;
                for (texel, pixel) in texels.iter_mut().zip(&pixels[start..start + span]) {
                    let sample = Vec4::new(
                        f32::from(pixel[0]),
                        f32::from(pixel[1]),
                        f32::from(pixel[2]),
                        f32::from(pixel[3]),
                    ) / 255.0;
                    *texel = texture_texel(sample, data, image.color_space);
                }
            }
            return Ok(());
        }
        parallel_fill_region(local, self.region, self.width(), self.height(), |uv| {
            let sample = maps.map(|maps| maps.sample(uv));
            texture_texel(
                projection.sample(image, uv, sample.as_ref()),
                data,
                image.color_space,
            )
        });
        Ok(())
    }

    fn render_stroke(&self, stroke: &PaintStroke, local: &mut [Vec4]) {
        let (value, color_alpha) = stroke_value(stroke, self.channel);
        let path = StrokePath {
            points: &stroke.points,
            pressures: &stroke.pressures,
            radius_pixels: stroke.radius_pixels,
            tip: stroke.tip,
        };
        brush::rasterize_stroke(
            &path,
            self.width(),
            self.height(),
            self.context.resources,
            self.region,
            |index, coverage| {
                let amount = (coverage * stroke.opacity).clamp(0.0, 1.0);
                if stroke.erase {
                    local[index].w *= 1.0 - amount;
                } else {
                    over(&mut local[index], value, amount * color_alpha, self.data);
                }
            },
        );
    }
}

impl TextureSet {
    /// True when a visible layer uses a mask generator or triplanar bitmap,
    /// so baking needs mesh maps.
    pub fn requires_mesh_maps(&self) -> bool {
        fn visible_need(layers: &[MaterialLayer]) -> bool {
            layers.iter().any(|layer| {
                layer.visible
                    && (layer.own_needs_mesh_maps()
                        || layer.children.as_deref().is_some_and(visible_need))
            })
        }
        visible_need(&self.layers)
    }

    /// Every bitmap resource the layer tree references.
    pub fn referenced_resources(&self) -> std::collections::BTreeSet<ResourceId> {
        self.walk_layers()
            .into_iter()
            .flat_map(|layer| layer.own_resources().collect::<Vec<_>>())
            .collect()
    }

    pub fn bake_base_color(&self) -> Result<BakedTexture, BakeError> {
        self.bake_channel(MaterialChannel::BaseColor, BakeContext::default())
    }

    /// Checks the set can bake and returns the ID map over `region` (RGBA,
    /// row-major within it) when a layer clips to an ID colour.
    pub(crate) fn validate_for_region(
        &self,
        region: TexelRect,
    ) -> Result<Option<Vec<u8>>, BakeError> {
        let layers = self.walk_layers();
        if layers.iter().any(|layer| !valid_layer_strokes(layer)) {
            return Err(BakeError::InvalidStrokeData);
        }
        if layers.iter().any(|layer| !layer.has_valid_settings())
            || !valid_height_strength(self.height_strength)
        {
            return Err(BakeError::InvalidLayerSettings);
        }
        checked_bake_pixel_count(self.resolution)?;
        checked_region(region)?;
        if region.intersect(self.texel_rect()) != Some(region) {
            return Err(BakeError::RegionMismatch);
        }
        layers
            .iter()
            .any(|layer| layer.id_mask_color.is_some())
            .then(|| self.id_map_region(region))
            .transpose()
    }

    /// Every texel of the set's textures.
    pub(crate) fn texel_rect(&self) -> TexelRect {
        TexelRect::full(self.resolution.width, self.resolution.height)
    }

    /// Texels a region rebake composites around the region so spatial
    /// filters (blur, sharpen) are exact inside it; 0 when every filter
    /// works per texel. An edit there changes texels this far beyond it.
    ///
    /// The sum over every filter bounds any chain of them. Anchor masks
    /// composite part of the stack again, so each anchored layer can repeat
    /// that chain once more.
    pub fn region_halo(&self) -> u32 {
        let layers = self.walk_layers();
        let filters: u32 = layers
            .iter()
            .map(|layer| {
                filters::spatial_halo(&layer.filters, self.resolution.width)
                    + filters::spatial_halo(&layer.mask_filters, self.resolution.width)
            })
            .sum();
        let anchors = layers
            .iter()
            .filter(|layer| layer.mask_anchor.is_some())
            .count() as u32;
        filters.saturating_mul(anchors + 1)
    }

    /// Rebakes only `region` of `textures`, a [`Self::bake_material`]
    /// result for an earlier state of this set. When the two states differ
    /// only inside `edited`, rebaking `edited` grown by
    /// [`Self::region_halo`] makes the patched textures equal a full bake
    /// byte for byte.
    pub fn rebake_material_region(
        &self,
        context: BakeContext<'_>,
        textures: &mut MaterialTextures,
        region: TexelRect,
    ) -> Result<(), BakeError> {
        let width = self.resolution.width;
        if ![
            &textures.base_color,
            &textures.orm,
            &textures.normal,
            &textures.emissive,
        ]
        .into_iter()
        .all(|texture| self.region_target_fits(texture, region))
        {
            return Err(BakeError::RegionMismatch);
        }
        if region.is_empty() {
            return Ok(());
        }
        let grown = self.composited_region(region);
        let planes = self.material_rgba8(context, grown)?;
        for (texture, rgba8) in [
            &mut textures.base_color,
            &mut textures.orm,
            &mut textures.normal,
            &mut textures.emissive,
        ]
        .into_iter()
        .zip(&planes)
        {
            patch_rgba8(&mut texture.rgba8, width, rgba8, grown, region);
        }
        Ok(())
    }

    /// Texels composited to rebake `region` exactly: the halo spatial
    /// filters need, plus one for the height-to-normal slope.
    pub(crate) fn composited_region(&self, region: TexelRect) -> TexelRect {
        let (width, height) = (self.resolution.width, self.resolution.height);
        region.grow(self.region_halo().saturating_add(1), width, height)
    }

    /// True when the set can rebake `region` of `texture`: the texture is
    /// the set's size and `region` lies inside.
    fn region_target_fits(&self, texture: &BakedTexture, region: TexelRect) -> bool {
        let full = self.texel_rect();
        texture.width == full.width
            && texture.height == full.height
            && texture.rgba8.len() == full.len() * 4
            && (region.is_empty() || region.intersect(full) == Some(region))
    }

    /// Rebakes only `region` of `texture`, a [`Self::bake_channel`] result
    /// for `channel`; see [`Self::rebake_material_region`].
    pub fn rebake_channel_region(
        &self,
        channel: MaterialChannel,
        context: BakeContext<'_>,
        texture: &mut BakedTexture,
        region: TexelRect,
    ) -> Result<(), BakeError> {
        let width = self.resolution.width;
        if !self.region_target_fits(texture, region) {
            return Err(BakeError::RegionMismatch);
        }
        if region.is_empty() {
            return Ok(());
        }
        let grown = self.composited_region(region);
        let rgba8 = self.channel_rgba8(channel, context, grown)?;
        patch_rgba8(&mut texture.rgba8, width, &rgba8, grown, region);
        Ok(())
    }

    /// Composites every visible layer that enables `channel`.
    ///
    /// - Base Color: straight-alpha RGBA.
    /// - Emissive: opaque RGB over black.
    /// - Roughness, Metallic, Height, Opacity: opaque grayscale starting from
    ///   [`DEFAULT_ROUGHNESS`], [`DEFAULT_METALLIC`], [`DEFAULT_HEIGHT`],
    ///   [`DEFAULT_OPACITY`].
    /// - Normal: tangent-space normal combining the baked detail normal from
    ///   the mesh maps, Normal-channel layers (which replace what is
    ///   beneath), and detail derived from the Height channel.
    /// - Id: the discrete ID map.
    pub fn bake_channel(
        &self,
        channel: MaterialChannel,
        context: BakeContext<'_>,
    ) -> Result<BakedTexture, BakeError> {
        if channel == MaterialChannel::Id {
            return self.bake_id_mask();
        }
        let [rgba8] = self.bake_in_tiles(&self.bake_tiles()?, |region| {
            Ok([self.channel_rgba8(channel, context, region)?])
        })?;
        Ok(self.texture(rgba8))
    }

    /// Tiles a bake of the whole set composites one at a time: the whole
    /// set when it fits [`MAX_BAKE_PIXELS`], else squares that still fit
    /// once grown by the halo their exactness needs.
    pub(crate) fn bake_tiles(&self) -> Result<Vec<TexelRect>, BakeError> {
        checked_bake_pixel_count(self.resolution)?;
        let full = self.texel_rect();
        if full.len() <= MAX_BAKE_PIXELS {
            return Ok(vec![full]);
        }
        let side = (MAX_BAKE_PIXELS as f64).sqrt() as u32;
        let margin = self.region_halo().saturating_add(1).saturating_mul(2);
        let inner = side.saturating_sub(margin) / DIRTY_TILE_SIZE * DIRTY_TILE_SIZE;
        if inner == 0 {
            // Filters reach so far that no tile fits the working budget.
            return Err(BakeError::ResolutionExceedsBudget {
                width: full.width,
                height: full.height,
            });
        }
        Ok(self.tiles_of(inner))
    }

    /// The set cut into `side` × `side` squares (clipped at its edges).
    pub(crate) fn tiles_of(&self, side: u32) -> Vec<TexelRect> {
        let full = self.texel_rect();
        let side = side.max(1);
        let mut tiles = Vec::new();
        for y in (0..full.height).step_by(side as usize) {
            for x in (0..full.width).step_by(side as usize) {
                tiles.extend(full.intersect(TexelRect {
                    x,
                    y,
                    width: side,
                    height: side,
                }));
            }
        }
        tiles
    }

    /// Runs `bake` (which returns `N` RGBA buffers over the region it is
    /// given) over each of `tiles` grown by the set's halo, and assembles
    /// the tiles into `N` whole textures. A single tile covering the set
    /// bakes in one pass.
    pub(crate) fn bake_in_tiles<const N: usize>(
        &self,
        tiles: &[TexelRect],
        mut bake: impl FnMut(TexelRect) -> Result<[Vec<u8>; N], BakeError>,
    ) -> Result<[Vec<u8>; N], BakeError> {
        let textures = self.bake_tiles_into(tiles, N, |region| Ok(bake(region)?.to_vec()))?;
        textures.try_into().map_err(|_| BakeError::AllocationFailed)
    }

    /// [`Self::bake_in_tiles`] for `count` textures known at run time.
    pub(crate) fn bake_tiles_into(
        &self,
        tiles: &[TexelRect],
        count: usize,
        mut bake: impl FnMut(TexelRect) -> Result<Vec<Vec<u8>>, BakeError>,
    ) -> Result<Vec<Vec<u8>>, BakeError> {
        let full = self.texel_rect();
        if tiles == [full] {
            return bake(full);
        }
        let byte_count = checked_bake_pixel_count(self.resolution)?
            .checked_mul(4)
            .ok_or(BakeError::AllocationFailed)?;
        let mut textures = Vec::with_capacity(count);
        for _ in 0..count {
            textures.push(try_filled(byte_count, 0_u8)?);
        }
        for tile in tiles {
            let grown = self.composited_region(*tile);
            let parts = bake(grown)?;
            for (texture, part) in textures.iter_mut().zip(&parts) {
                patch_rgba8(texture, full.width, part, grown, *tile);
            }
        }
        Ok(textures)
    }

    /// [`Self::bake_channel`] over `region` only, as RGBA row-major within
    /// it.
    fn channel_rgba8(
        &self,
        channel: MaterialChannel,
        context: BakeContext<'_>,
        region: TexelRect,
    ) -> Result<Vec<u8>, BakeError> {
        if channel == MaterialChannel::Id {
            if self
                .walk_layers()
                .iter()
                .any(|layer| !valid_layer_strokes(layer))
            {
                return Err(BakeError::InvalidStrokeData);
            }
            checked_bake_pixel_count(self.resolution)?;
            return self.id_map_region(region);
        }
        base_texel(channel)?;
        let id_map = self.validate_for_region(region)?;
        let id_map = id_map.as_deref();
        Ok(if channel == MaterialChannel::Normal {
            let masks = self.layer_masks_in(
                &[MaterialChannel::Normal, MaterialChannel::Height],
                context,
                id_map,
                region,
            )?;
            plane_to_rgba8(
                &self.normal_region(context, &masks, region)?,
                PlaneData::Vector,
            )
        } else {
            let masks = self.layer_masks_in(&[channel], context, id_map, region)?;
            plane_to_rgba8(
                &self.composite_region(channel, context, &masks, None, region)?,
                plane_data(channel),
            )
        })
    }

    /// Bakes Base Color (with Opacity in alpha), packed ORM, Normal, and
    /// Emissive for display or engine export.
    pub fn bake_material(&self, context: BakeContext<'_>) -> Result<MaterialTextures, BakeError> {
        self.bake_material_in(context, &self.bake_tiles()?)
    }

    /// [`Self::bake_material`] composited tile by tile.
    pub(crate) fn bake_material_in(
        &self,
        context: BakeContext<'_>,
        tiles: &[TexelRect],
    ) -> Result<MaterialTextures, BakeError> {
        let [base_color, orm, normal, emissive] =
            self.bake_in_tiles(tiles, |region| self.material_rgba8(context, region))?;
        Ok(MaterialTextures {
            base_color: self.texture(base_color),
            orm: self.texture(orm),
            normal: self.texture(normal),
            emissive: self.texture(emissive),
        })
    }

    /// [`Self::bake_material`]'s Base Color, ORM, Normal, and Emissive over
    /// `region` only, each RGBA row-major within it.
    fn material_rgba8(
        &self,
        context: BakeContext<'_>,
        region: TexelRect,
    ) -> Result<[Vec<u8>; 4], BakeError> {
        let id_map = self.validate_for_region(region)?;
        // Masks (generators especially) are the expensive part; build each once
        // and share them across all channels.
        let masks = self.layer_masks_in(
            &[
                MaterialChannel::BaseColor,
                MaterialChannel::Roughness,
                MaterialChannel::Metallic,
                MaterialChannel::Normal,
                MaterialChannel::Height,
                MaterialChannel::Emissive,
                MaterialChannel::Opacity,
            ],
            context,
            id_map.as_deref(),
            region,
        )?;
        let masks = &masks;
        let pixel_count = region.len();
        // Channels nothing paints stay at their defaults without compositing.
        let plane = |channel: MaterialChannel| -> Result<Option<Vec<Vec4>>, BakeError> {
            if self
                .layers
                .iter()
                .any(|layer| contributes(layer, &[channel]))
            {
                self.composite_region(channel, context, masks, None, region)
                    .map(Some)
            } else {
                Ok(None)
            }
        };
        // Three workers keep peak memory bounded at high resolutions.
        let (base, scalars, normal) = std::thread::scope(|scope| {
            let base = scope.spawn(|| plane(MaterialChannel::BaseColor));
            let scalars = scope.spawn(|| -> Result<_, BakeError> {
                Ok((
                    plane(MaterialChannel::Roughness)?,
                    plane(MaterialChannel::Metallic)?,
                    plane(MaterialChannel::Emissive)?,
                    plane(MaterialChannel::Opacity)?,
                ))
            });
            let normal = self.normal_region(context, masks, region);
            let base = base.join().unwrap_or(Err(BakeError::AllocationFailed));
            let scalars = scalars.join().unwrap_or(Err(BakeError::AllocationFailed));
            (base, scalars, normal)
        });
        let (base, normal) = (base?, normal?);
        let (roughness, metallic, emissive, opacity) = scalars?;

        let scalar_at = |plane: &Option<Vec<Vec4>>, index: usize, default: f32| {
            plane.as_ref().map_or(default, |plane| plane[index].x)
        };
        let mut occlusion = try_filled(pixel_count, 1.0_f32)?;
        if let Some(maps) = context.mesh_maps {
            parallel_fill_region(
                &mut occlusion,
                region,
                self.resolution.width,
                self.resolution.height,
                |uv| {
                    let sample = maps.sample(uv);
                    if sample.covered {
                        sample.ambient_occlusion
                    } else {
                        1.0
                    }
                },
            );
        }
        let mut orm = Vec::new();
        let mut base_color = Vec::new();
        let mut emission = Vec::new();
        for buffer in [&mut orm, &mut base_color, &mut emission] {
            buffer
                .try_reserve_exact(pixel_count * 4)
                .map_err(|_| BakeError::AllocationFailed)?;
        }
        let default_base = rgb_unit(DEFAULT_BASE_COLOR);
        for index in 0..pixel_count {
            orm.extend([
                unit_to_byte(occlusion[index]),
                unit_to_byte(scalar_at(&roughness, index, DEFAULT_ROUGHNESS)),
                unit_to_byte(scalar_at(&metallic, index, DEFAULT_METALLIC)),
                255,
            ]);
            let texel = base.as_ref().map_or(Vec4::ZERO, |plane| plane[index]);
            let color = texel.truncate() * texel.w + default_base * (1.0 - texel.w);
            base_color.extend([
                unit_to_byte(color.x),
                unit_to_byte(color.y),
                unit_to_byte(color.z),
                unit_to_byte(scalar_at(&opacity, index, DEFAULT_OPACITY)),
            ]);
            let glow = emissive.as_ref().map_or(Vec4::ZERO, |plane| plane[index]);
            emission.extend([
                unit_to_byte(glow.x),
                unit_to_byte(glow.y),
                unit_to_byte(glow.z),
                255,
            ]);
        }
        Ok([
            base_color,
            orm,
            plane_to_rgba8(&normal, PlaneData::Vector),
            emission,
        ])
    }

    pub(crate) fn texture(&self, rgba8: Vec<u8>) -> BakedTexture {
        BakedTexture {
            width: self.resolution.width,
            height: self.resolution.height,
            rgba8,
        }
    }

    /// Masks over `region` of every layer (and group) that contributes to
    /// one of `channels`; `id_map` covers the same region.
    pub(crate) fn layer_masks_in(
        &self,
        channels: &[MaterialChannel],
        context: BakeContext<'_>,
        id_map: Option<&[u8]>,
        region: TexelRect,
    ) -> Result<LayerMasks, BakeError> {
        let mut masks = BTreeMap::new();
        self.collect_masks(&self.layers, channels, context, id_map, region, &mut masks)?;
        Ok(LayerMasks(masks))
    }

    fn collect_masks(
        &self,
        layers: &[MaterialLayer],
        channels: &[MaterialChannel],
        context: BakeContext<'_>,
        id_map: Option<&[u8]>,
        region: TexelRect,
        masks: &mut BTreeMap<LayerId, Vec<f32>>,
    ) -> Result<(), BakeError> {
        for layer in layers {
            if !contributes(layer, channels) {
                continue;
            }
            if let Some(mask) = self.layer_mask_in(layer, context, id_map, region)? {
                masks.insert(layer.id, mask);
            }
            if let Some(children) = &layer.children {
                self.collect_masks(children, channels, context, id_map, region, masks)?;
            }
        }
        Ok(())
    }

    /// One channel composited over `region` (row-major within it); `masks`
    /// must cover the same region.
    pub(crate) fn composite_region(
        &self,
        channel: MaterialChannel,
        context: BakeContext<'_>,
        masks: &LayerMasks,
        probe: Option<&crate::inspect::Probe>,
        region: TexelRect,
    ) -> Result<Vec<Vec4>, BakeError> {
        checked_bake_pixel_count(self.resolution)?;
        checked_region(region)?;
        let mut plane = try_filled(region.len(), base_texel(channel)?)?;
        if channel == MaterialChannel::Normal
            && let Some(maps) = context.mesh_maps
        {
            parallel_fill_region(
                &mut plane,
                region,
                self.resolution.width,
                self.resolution.height,
                |uv| maps.sample(uv).tangent_normal.extend(1.0),
            );
        }
        Compositor {
            set: self,
            channel,
            data: plane_data(channel),
            context,
            masks,
            region,
            probe,
        }
        .composite(&self.layers, &mut plane)?;
        Ok(plane)
    }

    /// Normal-channel composite over `region` with Height-derived detail
    /// blended on top (whiteout blend), so painted bumps and dents shade in
    /// the viewport. The slope at a texel reads its neighbours, so only
    /// texels whose neighbours lie inside `region` (or off the texture's
    /// edge) are exact.
    pub(crate) fn normal_region(
        &self,
        context: BakeContext<'_>,
        masks: &LayerMasks,
        region: TexelRect,
    ) -> Result<Vec<Vec4>, BakeError> {
        let mut normals =
            self.composite_region(MaterialChannel::Normal, context, masks, None, region)?;
        if self.height_strength <= 0.0
            || !self
                .layers
                .iter()
                .any(|layer| contributes(layer, &[MaterialChannel::Height]))
        {
            return Ok(normals);
        }
        let height =
            self.composite_region(MaterialChannel::Height, context, masks, None, region)?;
        let scale = self.height_strength * self.resolution.width.max(self.resolution.height) as f32
            / HEIGHT_SLOPE_REFERENCE_TEXELS;
        // Texture coordinates, clamped to the texture and then the region.
        let (last_x, last_y) = (self.resolution.width - 1, self.resolution.height - 1);
        let at = |x: u32, y: u32| {
            let x = x.clamp(region.x, region.right() - 1);
            let y = y.clamp(region.y, region.bottom() - 1);
            height[region.local_index(x, y)].x
        };
        for y in region.y..region.bottom() {
            for x in region.x..region.right() {
                let du = (at((x + 1).min(last_x), y) - at(x.saturating_sub(1), y)) * 0.5;
                // Rows run top to bottom, so +V is toward the previous row.
                let dv = (at(x, y.saturating_sub(1)) - at(x, (y + 1).min(last_y))) * 0.5;
                if du == 0.0 && dv == 0.0 {
                    continue;
                }
                let detail = Vec3::new(-du * scale, -dv * scale, 1.0).normalize_or(Vec3::Z);
                let index = region.local_index(x, y);
                let base = normals[index].truncate();
                normals[index] = Vec3::new(base.x + detail.x, base.y + detail.y, base.z * detail.z)
                    .normalize_or(Vec3::Z)
                    .extend(1.0);
            }
        }
        Ok(normals)
    }

    /// Final per-texel mask of a layer, or `None` when it is fully open. The
    /// generator (or white) is the base, a bitmap multiplies it, painted
    /// mask strokes blend over it, mask filters run, and polygon and
    /// ID-colour clipping multiply the result. Covers `region`, as does
    /// `id_map`.
    pub(crate) fn layer_mask_in(
        &self,
        layer: &MaterialLayer,
        context: BakeContext<'_>,
        id_map: Option<&[u8]>,
        region: TexelRect,
    ) -> Result<Option<Vec<f32>>, BakeError> {
        if layer.mask_generator.is_none()
            && layer.mask_strokes.is_empty()
            && layer.polygon_mask.is_none()
            && layer.id_mask_color.is_none()
            && layer.mask_bitmap.is_none()
            && !layer.invert_final_mask
            && layer.mask_filters.is_empty()
            && layer.mask_anchor.is_none()
        {
            return Ok(None);
        }
        let width = self.resolution.width;
        let height = self.resolution.height;
        checked_bake_pixel_count(self.resolution)?;
        let pixel_count = region.len();
        checked_region(region)?;
        let mut values = try_filled(pixel_count, 1.0_f32)?;

        if let Some(generator) = &layer.mask_generator {
            let maps = context.mesh_maps.ok_or(BakeError::MissingMeshMaps)?;
            parallel_fill_region(&mut values, region, width, height, |uv| {
                generator.evaluate(&maps.sample(uv))
            });
        }
        if let Some(bitmap) = layer.mask_bitmap {
            let image = context.image(bitmap.resource)?;
            let maps = if bitmap.projection.needs_mesh_maps() {
                Some(context.mesh_maps.ok_or(BakeError::MissingMeshMaps)?)
            } else {
                None
            };
            let mut bitmap_values = try_filled(pixel_count, 0.0_f32)?;
            parallel_fill_region(&mut bitmap_values, region, width, height, |uv| {
                let sample = maps.map(|maps| maps.sample(uv));
                let texel = bitmap.projection.sample(image, uv, sample.as_ref());
                let value = luminance(texel) * texel.w;
                if bitmap.invert { 1.0 - value } else { value }
            });
            for (value, bitmap_value) in values.iter_mut().zip(bitmap_values) {
                *value *= bitmap_value;
            }
        }
        if let Some(anchor) = layer.mask_anchor {
            let anchored = self.anchor_values(layer.id, anchor, context, region)?;
            for (value, anchored) in values.iter_mut().zip(anchored) {
                *value *= anchored;
            }
        }
        for stroke in &layer.mask_strokes {
            let path = StrokePath {
                points: &stroke.points,
                pressures: &stroke.pressures,
                radius_pixels: stroke.radius_pixels,
                tip: stroke.tip,
            };
            brush::rasterize_stroke(
                &path,
                width,
                height,
                context.resources,
                region,
                |index, coverage| {
                    let alpha = (stroke.opacity * coverage).clamp(0.0, 1.0);
                    values[index] += (stroke.value - values[index]) * alpha;
                },
            );
        }
        let grid = PlaneGrid {
            region,
            width,
            height,
        };
        filters::apply_to_mask(&mut values, grid, &layer.mask_filters);
        if let Some(polygon_mask) = &layer.polygon_mask {
            let mesh = context.mesh.ok_or(BakeError::MissingMesh)?;
            let mut selected = try_filled(pixel_count, false)?;
            for triangle in &polygon_mask.triangles {
                let Some(corners) = mesh.triangles.get(*triangle as usize) else {
                    continue;
                };
                let Some(uvs) = corners
                    .iter()
                    .map(|index| mesh.vertices.get(*index as usize).map(|vertex| vertex.uv))
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                raster::rasterize_uv_triangle_in(
                    [uvs[0], uvs[1], uvs[2]],
                    width,
                    height,
                    0.5,
                    region,
                    |index, _, _| selected[index] = true,
                );
            }
            for (value, selected) in values.iter_mut().zip(selected) {
                if selected == polygon_mask.inverted {
                    *value = 0.0;
                }
            }
        }
        if let Some(target) = layer.id_mask_color {
            for (index, value) in values.iter_mut().enumerate() {
                let matches = id_map
                    .and_then(|map| map.get(index * 4..index * 4 + 4))
                    .is_some_and(|pixel| pixel[0..3] == target && pixel[3] > 0);
                if !matches {
                    *value = 0.0;
                }
            }
        }
        for value in &mut values {
            if layer.invert_final_mask {
                *value = 1.0 - *value;
            }
            *value = value.clamp(0.0, 1.0);
        }
        Ok(Some(values))
    }

    /// True when `layer` may anchor to `anchor`: the anchor exists, is
    /// composited before `layer`, and is neither `layer` nor one of its
    /// ancestors (which would make the mask depend on itself).
    pub fn anchor_is_valid(&self, layer: LayerId, anchor: LayerId) -> bool {
        let order = self.walk_layers();
        let position = |id: LayerId| order.iter().position(|candidate| candidate.id == id);
        let (Some(anchor_at), Some(layer_at)) = (position(anchor), position(layer)) else {
            return false;
        };
        anchor_at < layer_at
            && !self
                .layer(anchor)
                .is_some_and(|candidate| candidate.walk().iter().any(|inner| inner.id == layer))
    }

    /// A copy of the set holding only what is composited up to and
    /// including `anchor`: earlier siblings along its path, and the anchor
    /// with everything inside it.
    pub fn truncated_through(&self, anchor: LayerId) -> Option<TextureSet> {
        fn truncate(layers: &[MaterialLayer], anchor: LayerId) -> Option<Vec<MaterialLayer>> {
            for (index, layer) in layers.iter().enumerate() {
                if layer.id == anchor {
                    return Some(layers[..=index].to_vec());
                }
                if let Some(kept) = layer
                    .children
                    .as_deref()
                    .and_then(|children| truncate(children, anchor))
                {
                    let mut result = layers[..index].to_vec();
                    result.push(MaterialLayer {
                        children: Some(kept),
                        ..layer.clone()
                    });
                    return Some(result);
                }
            }
            None
        }
        Some(TextureSet {
            layers: truncate(&self.layers, anchor)?,
            ..self.clone()
        })
    }

    /// Per-texel anchor mask values for `layer`.
    fn anchor_values(
        &self,
        layer: LayerId,
        anchor: AnchorMask,
        context: BakeContext<'_>,
        region: TexelRect,
    ) -> Result<Vec<f32>, BakeError> {
        if !self.anchor_is_valid(layer, anchor.layer) {
            return Err(BakeError::InvalidLayerSettings);
        }
        let stack = self
            .truncated_through(anchor.layer)
            .ok_or(BakeError::InvalidLayerSettings)?;
        let id_map = stack.validate_for_region(region)?;
        let masks = stack.layer_masks_in(&[anchor.channel], context, id_map.as_deref(), region)?;
        let plane = stack.composite_region(anchor.channel, context, &masks, None, region)?;
        let read = |texel: &Vec4| match plane_data(anchor.channel) {
            // Colour reads luminance of what was painted; unpainted is 0.
            PlaneData::Color => luminance(*texel) * texel.w,
            _ => texel.x,
        };
        Ok(plane
            .iter()
            .map(|texel| anchor.remap(read(texel)))
            .collect())
    }

    /// Flattens one layer or group (with its opacity, mask, and filters)
    /// into a straight-alpha bitmap per channel it writes. Each bitmap's
    /// alpha is the layer's coverage, so a fill layer textured with them,
    /// in the same place in the stack, composites to the same result.
    pub fn flatten_layer(
        &self,
        layer_id: LayerId,
        context: BakeContext<'_>,
    ) -> Result<FlattenedLayer, BakeError> {
        let layer = self
            .layer(layer_id)
            .ok_or(BakeError::InvalidLayerSettings)?;
        let channels: Vec<MaterialChannel> = FLATTEN_CHANNELS
            .into_iter()
            .filter(|channel| contributes(layer, &[*channel]))
            .collect();
        let textures = self.bake_tiles_into(&self.bake_tiles()?, channels.len(), |region| {
            let id_map = self.validate_for_region(region)?;
            let masks = self.layer_masks_in(&channels, context, id_map.as_deref(), region)?;
            let mut textures = Vec::new();
            for channel in &channels {
                let data = plane_data(*channel);
                let mut plane = try_filled(region.len(), Vec4::ZERO)?;
                Compositor {
                    set: self,
                    channel: *channel,
                    data,
                    context,
                    masks: &masks,
                    region,
                    probe: None,
                }
                .composite(std::slice::from_ref(layer), &mut plane)?;
                let mut rgba8 = Vec::new();
                rgba8
                    .try_reserve_exact(region.len() * 4)
                    .map_err(|_| BakeError::AllocationFailed)?;
                for texel in &plane {
                    let rgb = match data {
                        PlaneData::Vector if texel.w > 0.0 => texel.truncate() * 0.5 + 0.5,
                        PlaneData::Vector => Vec3::new(0.5, 0.5, 1.0),
                        _ => texel.truncate(),
                    };
                    rgba8.extend([
                        unit_to_byte(rgb.x),
                        unit_to_byte(rgb.y),
                        unit_to_byte(rgb.z),
                        unit_to_byte(texel.w),
                    ]);
                }
                textures.push(rgba8);
            }
            Ok(textures)
        })?;
        let mut flattened = Vec::new();
        for (channel, rgba8) in channels.into_iter().zip(textures) {
            let image = TextureImage::new(self.resolution.width, self.resolution.height, rgba8)
                .ok_or(BakeError::ResolutionExceedsBudget {
                    width: self.resolution.width,
                    height: self.resolution.height,
                })?;
            flattened.push((channel, image));
        }
        Ok(FlattenedLayer {
            channels: flattened,
            // A flattened group is a plain layer; Normal is the closest match.
            blend_mode: match layer.blend_mode {
                BlendMode::PassThrough => BlendMode::Normal,
                mode => mode,
            },
        })
    }

    pub fn bake_id_mask(&self) -> Result<BakedTexture, BakeError> {
        if self
            .walk_layers()
            .iter()
            .any(|layer| !valid_layer_strokes(layer))
        {
            return Err(BakeError::InvalidStrokeData);
        }
        checked_bake_pixel_count(self.resolution)?;
        Ok(self.texture(self.id_map_region(self.texel_rect())?))
    }

    /// The ID map over `region`, RGBA row-major within it.
    pub(crate) fn id_map_region(&self, region: TexelRect) -> Result<Vec<u8>, BakeError> {
        let byte_count = region
            .len()
            .checked_mul(4)
            .ok_or(BakeError::AllocationFailed)?;
        let mut rgba8 = try_filled(byte_count, 0_u8)?;
        self.paint_id_layers(&self.layers, &mut rgba8, region);
        Ok(rgba8)
    }

    fn paint_id_layers(&self, layers: &[MaterialLayer], rgba8: &mut [u8], region: TexelRect) {
        for layer in layers {
            if !layer.visible || !layer.channels.contains(&MaterialChannel::Id) {
                continue;
            }
            if let Some(children) = &layer.children {
                self.paint_id_layers(children, rgba8, region);
                continue;
            }
            for stroke in &layer.id_strokes {
                let path = StrokePath {
                    points: &stroke.points,
                    pressures: &stroke.pressures,
                    radius_pixels: stroke.radius_pixels,
                    tip: stroke.tip,
                };
                brush::rasterize_stroke(
                    &path,
                    self.resolution.width,
                    self.resolution.height,
                    None,
                    region,
                    |index, coverage| {
                        if coverage * stroke.opacity * layer.opacity >= 0.5 {
                            let offset = index * 4;
                            rgba8[offset..offset + 4].copy_from_slice(&[
                                stroke.color[0],
                                stroke.color[1],
                                stroke.color[2],
                                255,
                            ]);
                        }
                    },
                );
            }
        }
    }
}

/// Below this many texels a fill runs on the calling thread.
const PARALLEL_FILL_MIN_TEXELS: usize = 16_384;

/// Sets every texel of `region` (with `values` row-major over it) of a
/// `width` × `height` texture from `value_at(uv)`, splitting rows across
/// threads.
pub(crate) fn parallel_fill_region<T: Send>(
    values: &mut [T],
    region: TexelRect,
    width: u32,
    height: u32,
    value_at: impl Fn(Vec2) -> T + Sync,
) {
    let row_len = region.width as usize;
    if row_len == 0 {
        return;
    }
    let threads = if values.len() < PARALLEL_FILL_MIN_TEXELS {
        1
    } else {
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .clamp(1, 32)
    };
    let rows_per_chunk = (region.height as usize).div_ceil(threads).max(1);
    let chunk_len = rows_per_chunk * row_len;
    let value_at = &value_at;
    let fill = move |chunk_index: usize, chunk: &mut [T]| {
        let start = chunk_index * chunk_len;
        for (offset, value) in chunk.iter_mut().enumerate() {
            let index = start + offset;
            let x = region.x + (index % row_len) as u32;
            let y = region.y + (index / row_len) as u32;
            *value = value_at(raster::pixel_to_uv(x, y, width, height));
        }
    };
    if threads == 1 {
        fill(0, values);
        return;
    }
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in values.chunks_mut(chunk_len).enumerate() {
            scope.spawn(move || fill(chunk_index, chunk));
        }
    });
}

/// Copies the texels of `region` from `source` (RGBA row-major over
/// `source_region`, which contains `region`) into `target`, a whole
/// RGBA texture `width` texels wide.
fn patch_rgba8(
    target: &mut [u8],
    width: u32,
    source: &[u8],
    source_region: TexelRect,
    region: TexelRect,
) {
    let span = region.width as usize * 4;
    for y in region.y..region.bottom() {
        let from = source_region.local_index(region.x, y) * 4;
        let to = (y as usize * width as usize + region.x as usize) * 4;
        target[to..to + span].copy_from_slice(&source[from..from + span]);
    }
}

pub(crate) fn try_filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>, BakeError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| BakeError::AllocationFailed)?;
    values.resize(count, value);
    Ok(values)
}

pub(crate) fn checked_bake_pixel_count(resolution: TextureResolution) -> Result<usize, BakeError> {
    if resolution.width == 0
        || resolution.height == 0
        || resolution.width > MAX_TEXTURE_DIMENSION
        || resolution.height > MAX_TEXTURE_DIMENSION
    {
        return Err(BakeError::ResolutionExceedsBudget {
            width: resolution.width,
            height: resolution.height,
        });
    }
    (resolution.width as usize)
        .checked_mul(resolution.height as usize)
        .filter(|count| *count <= MAX_TEXTURE_SET_PIXELS)
        .ok_or(BakeError::ResolutionExceedsBudget {
            width: resolution.width,
            height: resolution.height,
        })
}

/// Rejects regions whose working buffers would exceed [`MAX_BAKE_PIXELS`];
/// larger bakes go through [`TextureSet::bake_tiles`].
fn checked_region(region: TexelRect) -> Result<(), BakeError> {
    if region.len() > MAX_BAKE_PIXELS {
        return Err(BakeError::ResolutionExceedsBudget {
            width: region.width,
            height: region.height,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "compositing_region_tests.rs"]
mod region_tests;
