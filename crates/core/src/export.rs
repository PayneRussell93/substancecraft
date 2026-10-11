//! Export presets: serializable descriptions of which files an export
//! writes and how material channels and mesh maps are routed, inverted, and
//! packed into their colour channels.

use std::collections::{BTreeMap, BTreeSet};

use glam::{Vec3, Vec4};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::compositing::{
    BakeContext, BakeError, BakedTexture, LayerMasks, base_texel, checked_bake_pixel_count,
    contributes, parallel_fill, rgb_unit, try_filled, unit_to_byte,
};
use crate::{
    DEFAULT_BASE_COLOR, DEFAULT_HEIGHT, DEFAULT_METALLIC, DEFAULT_ROUGHNESS, MaterialChannel,
    MeshMapKind, MeshMaps, TextureSet,
};

/// Upper bound on files one preset writes.
pub const MAX_EXPORT_OUTPUTS: usize = 32;
/// Upper bound on a file-name template's length, in characters.
pub const MAX_EXPORT_FILE_NAME_CHARS: usize = 120;
/// Upper bound on a preset name's length, in characters.
pub const MAX_EXPORT_PRESET_NAME_CHARS: usize = 64;
/// Replaced by the texture set's name in output file names.
pub const TEXTURE_SET_TOKEN: &str = "$textureSet";

/// Sources baked concurrently; each holds a full-resolution float plane
/// while it composites, so this bounds peak memory.
const SOURCE_WORKERS: usize = 3;

/// Data an output channel can be filled from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExportSource {
    /// A composited material channel, as the material bake produces it:
    /// - Base Color: sRGB over the neutral default (opaque; Opacity is its
    ///   own channel).
    /// - Emissive: sRGB over black.
    /// - Roughness, Metallic, Height, Opacity: linear grayscale.
    /// - Normal: tangent-space normal (OpenGL, +Y = +V) with baked and
    ///   height-derived detail.
    /// - Ambient Occlusion: the baked AO mesh map, white outside UV islands
    ///   or when no mesh maps exist.
    /// - Id: the discrete ID colour map.
    Channel(MaterialChannel),
    /// A baked mesh map resampled to the texture set's resolution; black
    /// outside UV islands. Requires mesh maps.
    MeshMap(MeshMapKind),
}

impl ExportSource {
    /// Every source, in menu order.
    pub const ALL: [Self; 16] = [
        Self::Channel(MaterialChannel::BaseColor),
        Self::Channel(MaterialChannel::Roughness),
        Self::Channel(MaterialChannel::Metallic),
        Self::Channel(MaterialChannel::Normal),
        Self::Channel(MaterialChannel::Height),
        Self::Channel(MaterialChannel::AmbientOcclusion),
        Self::Channel(MaterialChannel::Emissive),
        Self::Channel(MaterialChannel::Opacity),
        Self::Channel(MaterialChannel::Id),
        Self::MeshMap(MeshMapKind::Normal),
        Self::MeshMap(MeshMapKind::WorldNormal),
        Self::MeshMap(MeshMapKind::Position),
        Self::MeshMap(MeshMapKind::Curvature),
        Self::MeshMap(MeshMapKind::AmbientOcclusion),
        Self::MeshMap(MeshMapKind::Thickness),
        Self::MeshMap(MeshMapKind::Skew),
    ];

    /// True when the source carries different data in R, G, and B.
    pub const fn is_color(self) -> bool {
        match self {
            Self::Channel(channel) => matches!(
                channel,
                MaterialChannel::BaseColor
                    | MaterialChannel::Emissive
                    | MaterialChannel::Normal
                    | MaterialChannel::Id
            ),
            Self::MeshMap(kind) => matches!(
                kind,
                MeshMapKind::Normal | MeshMapKind::WorldNormal | MeshMapKind::Position
            ),
        }
    }
}

/// Colour component read from a source. Grayscale sources hold the same
/// value in all three.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceComponent {
    #[default]
    Red,
    Green,
    Blue,
}

impl SourceComponent {
    pub const ALL: [Self; 3] = [Self::Red, Self::Green, Self::Blue];

    const fn index(self) -> usize {
        match self {
            Self::Red => 0,
            Self::Green => 1,
            Self::Blue => 2,
        }
    }
}

/// What one channel of an output file holds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChannelSlot {
    /// A fixed value in `0..=1`.
    Constant(f32),
    Source {
        source: ExportSource,
        component: SourceComponent,
        /// Stores `1 - value`, e.g. smoothness from roughness or a DirectX
        /// normal's green channel.
        invert: bool,
    },
}

impl ChannelSlot {
    pub const ZERO: Self = Self::Constant(0.0);
    pub const ONE: Self = Self::Constant(1.0);

    pub const fn channel(channel: MaterialChannel, component: SourceComponent) -> Self {
        Self::Source {
            source: ExportSource::Channel(channel),
            component,
            invert: false,
        }
    }

    pub const fn inverted(channel: MaterialChannel, component: SourceComponent) -> Self {
        Self::Source {
            source: ExportSource::Channel(channel),
            component,
            invert: true,
        }
    }

    pub const fn source(self) -> Option<ExportSource> {
        match self {
            Self::Constant(_) => None,
            Self::Source { source, .. } => Some(source),
        }
    }
}

/// Channel layout of an output file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputLayout {
    /// One channel, taken from the first slot.
    Gray,
    Rgb,
    Rgba,
}

impl OutputLayout {
    pub const ALL: [Self; 3] = [Self::Gray, Self::Rgb, Self::Rgba];

    pub const fn channel_count(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Gray => "Gray",
            Self::Rgb => "RGB",
            Self::Rgba => "RGBA",
        }
    }
}

/// One file an export writes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportOutput {
    /// File name without extension; [`TEXTURE_SET_TOKEN`] is replaced by the
    /// texture set's name.
    pub file_name: String,
    pub layout: OutputLayout,
    /// R, G, B, A; slots past the layout's channel count are ignored.
    pub slots: [ChannelSlot; 4],
    /// Leave the file out when every colour texel is black, e.g. an
    /// emissive map of a material that does not glow.
    #[serde(default)]
    pub skip_if_black: bool,
}

impl ExportOutput {
    pub fn new(file_name: &str, layout: OutputLayout, slots: [ChannelSlot; 4]) -> Self {
        Self {
            file_name: file_name.to_owned(),
            layout,
            slots,
            skip_if_black: false,
        }
    }

    /// The slots this output actually writes.
    pub fn used_slots(&self) -> &[ChannelSlot] {
        &self.slots[..self.layout.channel_count()]
    }

    /// The file name for `texture_set_name`, with characters that are not
    /// portable in file names replaced by `_`.
    pub fn resolved_file_name(&self, texture_set_name: &str) -> String {
        self.file_name
            .replace(TEXTURE_SET_TOKEN, texture_set_name)
            .chars()
            .map(|character| {
                if character == ' ' || !portable_file_name_char(character) {
                    '_'
                } else {
                    character
                }
            })
            .collect()
    }
}

/// A named list of output files.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportPreset {
    pub name: String,
    pub outputs: Vec<ExportOutput>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExportPresetError {
    #[error("export preset name must be 1 to {MAX_EXPORT_PRESET_NAME_CHARS} characters")]
    InvalidName,
    #[error("export preset must write 1 to {MAX_EXPORT_OUTPUTS} files")]
    InvalidOutputCount,
    #[error(
        "output file name `{0}` must be 1 to {MAX_EXPORT_FILE_NAME_CHARS} characters without \
         path separators or reserved characters"
    )]
    InvalidFileName(String),
    #[error("output file name `{0}` is used more than once")]
    DuplicateFileName(String),
    #[error("output `{0}` has a constant outside 0..=1")]
    InvalidConstant(String),
}

impl ExportPreset {
    /// Presets every project offers.
    pub fn built_in() -> Vec<Self> {
        use MaterialChannel::{
            AmbientOcclusion, BaseColor, Emissive, Height, Metallic, Normal, Opacity, Roughness,
        };
        use OutputLayout::{Gray, Rgb, Rgba};
        use SourceComponent::{Blue, Green, Red};
        let rgb = |channel| {
            [
                ChannelSlot::channel(channel, Red),
                ChannelSlot::channel(channel, Green),
                ChannelSlot::channel(channel, Blue),
                ChannelSlot::ONE,
            ]
        };
        let gray = |channel| {
            [
                ChannelSlot::channel(channel, Red),
                ChannelSlot::ZERO,
                ChannelSlot::ZERO,
                ChannelSlot::ONE,
            ]
        };
        let base_with_opacity = [
            ChannelSlot::channel(BaseColor, Red),
            ChannelSlot::channel(BaseColor, Green),
            ChannelSlot::channel(BaseColor, Blue),
            ChannelSlot::channel(Opacity, Red),
        ];
        let orm = [
            ChannelSlot::channel(AmbientOcclusion, Red),
            ChannelSlot::channel(Roughness, Red),
            ChannelSlot::channel(Metallic, Red),
            ChannelSlot::ONE,
        ];
        let emissive = |name: &str| ExportOutput {
            skip_if_black: true,
            ..ExportOutput::new(name, Rgb, rgb(Emissive))
        };
        vec![
            Self {
                name: "Metal/Rough (ORM)".to_owned(),
                outputs: vec![
                    ExportOutput::new("$textureSet_BaseColor", Rgba, base_with_opacity),
                    ExportOutput::new("$textureSet_ORM", Rgb, orm),
                    ExportOutput::new("$textureSet_Normal", Rgb, rgb(Normal)),
                    emissive("$textureSet_Emissive"),
                ],
            },
            Self {
                name: "Unreal Engine".to_owned(),
                outputs: vec![
                    ExportOutput::new("T_$textureSet_BC", Rgba, base_with_opacity),
                    ExportOutput::new("T_$textureSet_ORM", Rgb, orm),
                    // DirectX convention: +Y points down the texture.
                    ExportOutput::new(
                        "T_$textureSet_N",
                        Rgb,
                        [
                            ChannelSlot::channel(Normal, Red),
                            ChannelSlot::inverted(Normal, Green),
                            ChannelSlot::channel(Normal, Blue),
                            ChannelSlot::ONE,
                        ],
                    ),
                    emissive("T_$textureSet_E"),
                ],
            },
            Self {
                name: "Unity (URP Lit)".to_owned(),
                outputs: vec![
                    ExportOutput::new("$textureSet_BaseMap", Rgba, base_with_opacity),
                    ExportOutput::new(
                        "$textureSet_MetallicSmoothness",
                        Rgba,
                        [
                            ChannelSlot::channel(Metallic, Red),
                            ChannelSlot::ZERO,
                            ChannelSlot::ZERO,
                            ChannelSlot::inverted(Roughness, Red),
                        ],
                    ),
                    ExportOutput::new("$textureSet_Normal", Rgb, rgb(Normal)),
                    ExportOutput::new("$textureSet_Occlusion", Gray, gray(AmbientOcclusion)),
                    emissive("$textureSet_Emission"),
                ],
            },
            Self {
                name: "Separate channels".to_owned(),
                outputs: vec![
                    ExportOutput::new("$textureSet_BaseColor", Rgb, rgb(BaseColor)),
                    ExportOutput::new("$textureSet_Roughness", Gray, gray(Roughness)),
                    ExportOutput::new("$textureSet_Metallic", Gray, gray(Metallic)),
                    ExportOutput::new("$textureSet_Normal", Rgb, rgb(Normal)),
                    ExportOutput::new("$textureSet_Height", Gray, gray(Height)),
                    ExportOutput::new("$textureSet_AmbientOcclusion", Gray, gray(AmbientOcclusion)),
                    ExportOutput::new("$textureSet_Opacity", Gray, gray(Opacity)),
                    emissive("$textureSet_Emissive"),
                ],
            },
        ]
    }

    pub fn validate(&self) -> Result<(), ExportPresetError> {
        let name_chars = self.name.trim().chars().count();
        if name_chars == 0 || name_chars > MAX_EXPORT_PRESET_NAME_CHARS {
            return Err(ExportPresetError::InvalidName);
        }
        if self.outputs.is_empty() || self.outputs.len() > MAX_EXPORT_OUTPUTS {
            return Err(ExportPresetError::InvalidOutputCount);
        }
        let mut file_names = BTreeSet::new();
        for output in &self.outputs {
            if !valid_file_name_template(&output.file_name) {
                return Err(ExportPresetError::InvalidFileName(output.file_name.clone()));
            }
            // Case-insensitive, since common file systems are.
            if !file_names.insert(output.file_name.to_lowercase()) {
                return Err(ExportPresetError::DuplicateFileName(
                    output.file_name.clone(),
                ));
            }
            let constant_ok = |slot: &ChannelSlot| match *slot {
                ChannelSlot::Constant(value) => value.is_finite() && (0.0..=1.0).contains(&value),
                ChannelSlot::Source { .. } => true,
            };
            if !output.slots.iter().all(constant_ok) {
                return Err(ExportPresetError::InvalidConstant(output.file_name.clone()));
            }
        }
        Ok(())
    }

    /// True when any output reads a baked mesh map directly.
    pub fn reads_mesh_maps(&self) -> bool {
        self.sources()
            .iter()
            .any(|source| matches!(source, ExportSource::MeshMap(_)))
    }

    /// Distinct sources the outputs read.
    pub fn sources(&self) -> BTreeSet<ExportSource> {
        self.outputs
            .iter()
            .flat_map(|output| output.used_slots().iter().filter_map(|slot| slot.source()))
            .collect()
    }
}

fn portable_file_name_char(character: char) -> bool {
    !character.is_control()
        && !matches!(
            character,
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
        )
}

fn valid_file_name_template(template: &str) -> bool {
    let chars = template.chars().count();
    chars > 0
        && chars <= MAX_EXPORT_FILE_NAME_CHARS
        && template.trim() == template
        && !template.starts_with('.')
        && !template.ends_with('.')
        && template.chars().all(portable_file_name_char)
}

/// One packed file, ready to encode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportedTexture {
    /// File name without extension.
    pub file_stem: String,
    pub layout: OutputLayout,
    /// RGBA8 rows top to bottom. Gray outputs repeat their value in R, G,
    /// and B; Gray and RGB outputs are opaque.
    pub texture: BakedTexture,
}

impl TextureSet {
    /// Bakes every file `preset` describes. Material channels are composited
    /// once and shared between the outputs that read them.
    pub fn bake_export(
        &self,
        preset: &ExportPreset,
        context: BakeContext<'_>,
    ) -> Result<Vec<ExportedTexture>, BakeError> {
        preset.validate()?;
        let id_map = self.validate_for_bake()?;
        let pixel_count = checked_bake_pixel_count(self.resolution)?;
        let sources: Vec<ExportSource> = preset.sources().into_iter().collect();
        if sources
            .iter()
            .any(|source| matches!(source, ExportSource::MeshMap(_)))
            && context.mesh_maps.is_none()
        {
            return Err(BakeError::MissingMeshMaps);
        }

        let mut channels = BTreeSet::new();
        for source in &sources {
            match source {
                ExportSource::Channel(MaterialChannel::Normal) => {
                    channels.extend([MaterialChannel::Normal, MaterialChannel::Height]);
                }
                ExportSource::Channel(MaterialChannel::BaseColor) => {
                    channels.insert(MaterialChannel::BaseColor);
                }
                ExportSource::Channel(
                    channel @ (MaterialChannel::Roughness
                    | MaterialChannel::Metallic
                    | MaterialChannel::Height
                    | MaterialChannel::Emissive
                    | MaterialChannel::Opacity),
                ) => {
                    channels.insert(*channel);
                }
                _ => {}
            }
        }
        let channels: Vec<MaterialChannel> = channels.into_iter().collect();
        let masks = self.layer_masks(&channels, context, id_map.as_ref())?;

        let mut planes = BTreeMap::new();
        for batch in sources.chunks(SOURCE_WORKERS) {
            let baked: Vec<Result<Vec<u8>, BakeError>> = std::thread::scope(|scope| {
                let workers: Vec<_> = batch
                    .iter()
                    .map(|&source| {
                        let (masks, id_map) = (&masks, id_map.as_ref());
                        scope.spawn(move || {
                            self.export_source(source, context, masks, id_map, pixel_count)
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|worker| worker.join().unwrap_or(Err(BakeError::AllocationFailed)))
                    .collect()
            });
            for (source, rgba8) in batch.iter().zip(baked) {
                planes.insert(*source, rgba8?);
            }
        }

        let mut exported = Vec::new();
        for output in &preset.outputs {
            let mut rgba8 = Vec::new();
            rgba8
                .try_reserve_exact(pixel_count * 4)
                .map_err(|_| BakeError::AllocationFailed)?;
            let slot_byte = |slot: &ChannelSlot, index: usize| -> u8 {
                match *slot {
                    ChannelSlot::Constant(value) => unit_to_byte(value),
                    ChannelSlot::Source {
                        source,
                        component,
                        invert,
                    } => {
                        let value = planes
                            .get(&source)
                            .map_or(0, |plane: &Vec<u8>| plane[index * 4 + component.index()]);
                        if invert { 255 - value } else { value }
                    }
                }
            };
            for index in 0..pixel_count {
                let texel = match output.layout {
                    OutputLayout::Gray => {
                        let gray = slot_byte(&output.slots[0], index);
                        [gray, gray, gray, 255]
                    }
                    OutputLayout::Rgb | OutputLayout::Rgba => [
                        slot_byte(&output.slots[0], index),
                        slot_byte(&output.slots[1], index),
                        slot_byte(&output.slots[2], index),
                        if output.layout == OutputLayout::Rgba {
                            slot_byte(&output.slots[3], index)
                        } else {
                            255
                        },
                    ],
                };
                rgba8.extend(texel);
            }
            if output.skip_if_black
                && rgba8
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|texel| texel[..3] == [0, 0, 0])
            {
                continue;
            }
            exported.push(ExportedTexture {
                file_stem: output.resolved_file_name(&self.name),
                layout: output.layout,
                texture: self.texture(rgba8),
            });
        }
        Ok(exported)
    }

    /// One source as RGBA8 at the texture set's resolution.
    fn export_source(
        &self,
        source: ExportSource,
        context: BakeContext<'_>,
        masks: &LayerMasks,
        id_map: Option<&BakedTexture>,
        pixel_count: usize,
    ) -> Result<Vec<u8>, BakeError> {
        let (width, height) = (self.resolution.width, self.resolution.height);
        let to_rgba8 = |values: &[Vec3]| -> Result<Vec<u8>, BakeError> {
            let mut rgba8 = Vec::new();
            rgba8
                .try_reserve_exact(values.len() * 4)
                .map_err(|_| BakeError::AllocationFailed)?;
            for value in values {
                rgba8.extend([
                    unit_to_byte(value.x),
                    unit_to_byte(value.y),
                    unit_to_byte(value.z),
                    255,
                ]);
            }
            Ok(rgba8)
        };
        let channel = match source {
            ExportSource::MeshMap(kind) => {
                let maps = context.mesh_maps.ok_or(BakeError::MissingMeshMaps)?;
                let mut values = try_filled(pixel_count, Vec3::ZERO)?;
                parallel_fill(&mut values, width, height, |uv| {
                    let sample = maps.sample(uv);
                    if !sample.covered {
                        return Vec3::ZERO;
                    }
                    match kind {
                        MeshMapKind::Normal => sample.tangent_normal * 0.5 + 0.5,
                        MeshMapKind::WorldNormal => sample.normal * 0.5 + 0.5,
                        MeshMapKind::Position => sample.unit_position,
                        MeshMapKind::Curvature => Vec3::splat(sample.curvature),
                        MeshMapKind::AmbientOcclusion => Vec3::splat(sample.ambient_occlusion),
                        MeshMapKind::Thickness => Vec3::splat(sample.thickness),
                        MeshMapKind::Skew => Vec3::splat(sample.skew),
                    }
                });
                return to_rgba8(&values);
            }
            ExportSource::Channel(channel) => channel,
        };
        match channel {
            MaterialChannel::Id => Ok(match id_map {
                Some(id_map) => id_map.rgba8.clone(),
                None => self.bake_id_mask()?.rgba8,
            }),
            MaterialChannel::AmbientOcclusion => {
                let mut values = try_filled(pixel_count, Vec3::ONE)?;
                if let Some(maps) = context.mesh_maps {
                    parallel_fill(&mut values, width, height, |uv| {
                        let sample = maps.sample(uv);
                        Vec3::splat(if sample.covered {
                            sample.ambient_occlusion
                        } else {
                            1.0
                        })
                    });
                }
                to_rgba8(&values)
            }
            MaterialChannel::Normal => {
                let normals = self.normal_plane(context, masks)?;
                let values: Vec<Vec3> = normals
                    .iter()
                    .map(|normal| normal.truncate() * 0.5 + 0.5)
                    .collect();
                to_rgba8(&values)
            }
            MaterialChannel::BaseColor => {
                let default_base = rgb_unit(DEFAULT_BASE_COLOR);
                let plane = self.optional_plane(channel, context, masks)?;
                let values: Vec<Vec3> = match plane {
                    Some(plane) => plane
                        .iter()
                        .map(|texel| texel.truncate() * texel.w + default_base * (1.0 - texel.w))
                        .collect(),
                    None => try_filled(pixel_count, default_base)?,
                };
                to_rgba8(&values)
            }
            channel => {
                let plane = match self.optional_plane(channel, context, masks)? {
                    Some(plane) => plane,
                    None => try_filled(pixel_count, base_texel(channel)?)?,
                };
                let values: Vec<Vec3> = plane.iter().map(|texel| texel.truncate()).collect();
                to_rgba8(&values)
            }
        }
    }

    /// The composited plane, or `None` when no layer paints `channel`.
    fn optional_plane(
        &self,
        channel: MaterialChannel,
        context: BakeContext<'_>,
        masks: &LayerMasks,
    ) -> Result<Option<Vec<Vec4>>, BakeError> {
        if self
            .layers
            .iter()
            .any(|layer| contributes(layer, &[channel]))
        {
            self.composite_plane(channel, context, masks).map(Some)
        } else {
            Ok(None)
        }
    }
}

/// How serious a preflight finding is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PreflightSeverity {
    /// The export would succeed but probably not as intended.
    Warning,
    /// The export cannot run.
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreflightIssue {
    pub severity: PreflightSeverity,
    pub message: String,
}

impl PreflightIssue {
    fn error(message: impl Into<String>) -> Self {
        Self {
            severity: PreflightSeverity::Error,
            message: message.into(),
        }
    }

    fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: PreflightSeverity::Warning,
            message: message.into(),
        }
    }
}

impl TextureSet {
    /// Checks an export without baking: errors are problems that would make
    /// [`TextureSet::bake_export`] fail, warnings are outputs that would be
    /// written but are probably not what the artist wants (a channel left at
    /// its default, a flat normal, white occlusion). Errors come first.
    pub fn export_preflight(
        &self,
        preset: &ExportPreset,
        context: BakeContext<'_>,
    ) -> Vec<PreflightIssue> {
        if let Err(error) = preset.validate() {
            return vec![PreflightIssue::error(format!(
                "The preset is invalid: {error}"
            ))];
        }
        let mut issues = Vec::new();
        let (width, height) = (self.resolution.width, self.resolution.height);
        if checked_bake_pixel_count(self.resolution).is_err() {
            issues.push(PreflightIssue::error(format!(
                "{width} × {height} exceeds the texture bake budget."
            )));
        } else if !width.is_power_of_two() || !height.is_power_of_two() {
            issues.push(PreflightIssue::warning(format!(
                "{width} × {height} is not a power of two; engines may not mip-map or \
                 compress it."
            )));
        }

        let maps = context.mesh_maps;
        if maps.is_none() && self.requires_mesh_maps() {
            issues.push(PreflightIssue::error(
                "Layers use generators or triplanar projection, but mesh maps are not baked.",
            ));
        }
        if context.mesh.is_none()
            && self
                .walk_layers()
                .iter()
                .any(|layer| layer.visible && layer.polygon_mask.is_some())
        {
            issues.push(PreflightIssue::error(
                "A layer uses a polygon mask, but no mesh is loaded.",
            ));
        }
        for id in self.referenced_resources() {
            if context
                .resources
                .is_none_or(|images| !images.contains_key(&id))
            {
                issues.push(PreflightIssue::error(format!(
                    "Bitmap resource {} is used by a layer but is not loaded.",
                    id.0
                )));
            }
        }

        let mut seen = BTreeMap::new();
        for output in &preset.outputs {
            let name = output.resolved_file_name(&self.name);
            if let Some(first) = seen.insert(name.to_lowercase(), output.file_name.clone()) {
                issues.push(PreflightIssue::error(format!(
                    "`{}` and `{}` both resolve to `{name}`.",
                    first, output.file_name
                )));
            }
        }

        // Group the outputs reading each source so a source is reported once.
        let mut readers: BTreeMap<ExportSource, Vec<String>> = BTreeMap::new();
        for output in &preset.outputs {
            for source in output.used_slots().iter().filter_map(|slot| slot.source()) {
                let files = readers.entry(source).or_default();
                let name = output.resolved_file_name(&self.name);
                if !files.contains(&name) {
                    files.push(name);
                }
            }
        }
        let painted = |channels: &[MaterialChannel]| {
            self.layers.iter().any(|layer| contributes(layer, channels))
        };
        for (source, files) in readers {
            let problem = match source {
                ExportSource::MeshMap(kind) if maps.is_none() => {
                    issues.push(PreflightIssue::error(format!(
                        "{} reads the {} mesh map, which is not baked.",
                        files.join(", "),
                        kind.name()
                    )));
                    continue;
                }
                ExportSource::MeshMap(_) => None,
                ExportSource::Channel(channel) => match channel {
                    MaterialChannel::BaseColor if !painted(&[channel]) => {
                        Some("Base Color is the neutral default grey".to_owned())
                    }
                    MaterialChannel::Roughness if !painted(&[channel]) => {
                        Some(format!("Roughness is a uniform {DEFAULT_ROUGHNESS:.2}"))
                    }
                    MaterialChannel::Metallic if !painted(&[channel]) => {
                        Some(format!("Metallic is a uniform {DEFAULT_METALLIC:.2}"))
                    }
                    MaterialChannel::Height if !painted(&[channel]) => {
                        Some(format!("Height is a uniform {DEFAULT_HEIGHT:.2}"))
                    }
                    MaterialChannel::Normal
                        if !painted(&[MaterialChannel::Normal])
                            && !(painted(&[MaterialChannel::Height])
                                && self.height_strength > 0.0)
                            && !maps.is_some_and(MeshMaps::from_high_poly) =>
                    {
                        Some(
                            "Normal is flat: no high-poly bake, normal, or height detail"
                                .to_owned(),
                        )
                    }
                    MaterialChannel::AmbientOcclusion if maps.is_none() => {
                        Some("Occlusion is white because mesh maps are not baked".to_owned())
                    }
                    MaterialChannel::Id
                        if !self
                            .walk_layers()
                            .iter()
                            .any(|layer| !layer.id_strokes.is_empty()) =>
                    {
                        Some("The ID map is empty".to_owned())
                    }
                    _ => None,
                },
            };
            if let Some(problem) = problem {
                issues.push(PreflightIssue::warning(format!(
                    "{problem} in {}.",
                    files.join(", ")
                )));
            }
        }
        issues.sort_by_key(|issue| std::cmp::Reverse(issue.severity));
        issues
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{
        DEFAULT_HEIGHT_STRENGTH, LayerFill, LayerId, MaterialLayer, MaterialProject,
        MeshMapSettings, MeshMaps, PrimitivePreset, ProjectError, TextureResolution, TextureSetId,
        bake_mesh_maps,
    };

    fn set_with(size: u32, layer: MaterialLayer) -> TextureSet {
        TextureSet {
            id: TextureSetId(1),
            name: "Hull Plate".to_owned(),
            material_slot: "Hull".to_owned(),
            resolution: TextureResolution {
                width: size,
                height: size,
            },
            channels: BTreeSet::from([MaterialChannel::BaseColor]),
            layers: vec![layer],
            height_strength: DEFAULT_HEIGHT_STRENGTH,
            export: None,
        }
    }

    fn metal_layer(emissive: [u8; 3]) -> MaterialLayer {
        let mut layer = MaterialLayer::new_fill(
            LayerId(1),
            "Metal",
            LayerFill {
                color: [200, 100, 50, 255],
                roughness: 0.2,
                metallic: 1.0,
                emissive,
                material_opacity: 0.5,
                ..LayerFill::default()
            },
        );
        layer.channels = BTreeSet::from([
            MaterialChannel::BaseColor,
            MaterialChannel::Roughness,
            MaterialChannel::Metallic,
            MaterialChannel::Emissive,
            MaterialChannel::Opacity,
        ]);
        layer
    }

    fn built_in(name: &str) -> ExportPreset {
        ExportPreset::built_in()
            .into_iter()
            .find(|preset| preset.name == name)
            .unwrap()
    }

    fn cube_maps() -> MeshMaps {
        bake_mesh_maps(
            &PrimitivePreset::Cube.build_mesh(),
            &MeshMapSettings {
                resolution: 32,
                ao_samples: 4,
                ..MeshMapSettings::default()
            },
            None,
        )
        .unwrap()
    }

    fn gray_output(name: &str, slot: ChannelSlot) -> ExportOutput {
        ExportOutput::new(name, OutputLayout::Gray, [slot; 4])
    }

    #[test]
    fn tc_0110_orm_preset_matches_the_material_bake_and_skips_black_emission() {
        for preset in ExportPreset::built_in() {
            preset.validate().unwrap();
        }
        let preset = built_in("Metal/Rough (ORM)");
        let maps = cube_maps();
        let context = BakeContext {
            mesh_maps: Some(&maps),
            ..BakeContext::default()
        };

        let glowing = set_with(16, metal_layer([255, 64, 0]));
        let material = glowing.bake_material(context).unwrap();
        let files = glowing.bake_export(&preset, context).unwrap();
        let stems: Vec<&str> = files.iter().map(|file| file.file_stem.as_str()).collect();
        assert_eq!(
            stems,
            [
                "Hull_Plate_BaseColor",
                "Hull_Plate_ORM",
                "Hull_Plate_Normal",
                "Hull_Plate_Emissive"
            ]
        );
        for (file, expected) in files.iter().zip([
            &material.base_color,
            &material.orm,
            &material.normal,
            &material.emissive,
        ]) {
            assert_eq!(&file.texture, expected, "{}", file.file_stem);
        }
        assert_eq!(files[0].layout, OutputLayout::Rgba);
        assert_eq!(files[0].texture.pixel(3, 3).unwrap()[3], 128);

        let dark = set_with(16, metal_layer([0, 0, 0]));
        let files = dark.bake_export(&preset, context).unwrap();
        assert_eq!(files.len(), 3, "no emission, so no emissive file");
    }

    #[test]
    fn tc_0111_presets_route_invert_and_pack_channels_into_each_layout() {
        let set = set_with(8, metal_layer([0, 0, 0]));
        let context = BakeContext::default();
        let roughness = 51; // 0.2
        let opengl = set
            .bake_channel(MaterialChannel::Normal, context)
            .unwrap()
            .pixel(2, 2)
            .unwrap();

        // Unreal: the DirectX normal is the OpenGL normal with green flipped.
        let unreal = set
            .bake_export(&built_in("Unreal Engine"), context)
            .unwrap();
        let normal = unreal
            .iter()
            .find(|file| file.file_stem == "T_Hull_Plate_N")
            .unwrap();
        assert_eq!(
            normal.texture.pixel(2, 2).unwrap(),
            [opengl[0], 255 - opengl[1], opengl[2], 255]
        );

        // Unity: metallic in R, smoothness (inverted roughness) in A, and a
        // white single-channel AO map when no mesh maps exist.
        let unity = set
            .bake_export(&built_in("Unity (URP Lit)"), context)
            .unwrap();
        assert_eq!(unity[1].file_stem, "Hull_Plate_MetallicSmoothness");
        assert_eq!(
            unity[1].texture.pixel(0, 0).unwrap(),
            [255, 0, 0, 255 - roughness]
        );
        assert_eq!(unity[3].layout, OutputLayout::Gray);
        assert_eq!(unity[3].texture.pixel(5, 5).unwrap(), [255, 255, 255, 255]);

        // Gray outputs take their first slot; constants pack as bytes; the
        // texture set name replaces the token and unsafe characters become _.
        let custom = ExportPreset {
            name: "Custom".to_owned(),
            outputs: vec![
                gray_output(
                    "rough",
                    ChannelSlot::channel(MaterialChannel::Roughness, SourceComponent::Red),
                ),
                ExportOutput::new(
                    "$textureSet packed",
                    OutputLayout::Rgb,
                    [
                        ChannelSlot::channel(MaterialChannel::BaseColor, SourceComponent::Blue),
                        ChannelSlot::Constant(0.5),
                        ChannelSlot::inverted(MaterialChannel::Metallic, SourceComponent::Red),
                        ChannelSlot::ZERO,
                    ],
                ),
            ],
        };
        let files = set.bake_export(&custom, context).unwrap();
        assert_eq!(files[0].texture.pixel(1, 1).unwrap(), [51, 51, 51, 255]);
        assert_eq!(files[1].file_stem, "Hull_Plate_packed");
        assert_eq!(files[1].texture.pixel(1, 1).unwrap(), [50, 128, 0, 255]);
        let renamed = TextureSet {
            name: "a/b:c".to_owned(),
            ..set.clone()
        };
        assert_eq!(
            custom.outputs[1].resolved_file_name(&renamed.name),
            "a_b_c_packed"
        );

        // Mesh-map sources need baked maps.
        let curvature = ExportPreset {
            name: "Maps".to_owned(),
            outputs: vec![gray_output(
                "curvature",
                ChannelSlot::Source {
                    source: ExportSource::MeshMap(MeshMapKind::Curvature),
                    component: SourceComponent::Red,
                    invert: false,
                },
            )],
        };
        assert_eq!(
            set.bake_export(&curvature, context),
            Err(BakeError::MissingMeshMaps)
        );
        let maps = cube_maps();
        let with_maps = BakeContext {
            mesh_maps: Some(&maps),
            ..context
        };
        let files = set.bake_export(&curvature, with_maps).unwrap();
        assert!(files[0].texture.rgba8.iter().any(|&byte| byte != 0));
    }

    #[test]
    fn tc_0112_invalid_presets_are_rejected_with_typed_errors() {
        let output = |name: &str| ExportOutput::new(name, OutputLayout::Rgb, [ChannelSlot::ONE; 4]);
        let preset = |name: &str, outputs: Vec<ExportOutput>| ExportPreset {
            name: name.to_owned(),
            outputs,
        };
        assert_eq!(
            preset("  ", vec![output("a")]).validate(),
            Err(ExportPresetError::InvalidName)
        );
        assert_eq!(
            preset("Empty", Vec::new()).validate(),
            Err(ExportPresetError::InvalidOutputCount)
        );
        for bad in ["", "../escape", "a/b", "trailing.", " padded", "what?"] {
            assert_eq!(
                preset("Bad", vec![output(bad)]).validate(),
                Err(ExportPresetError::InvalidFileName(bad.to_owned())),
                "{bad:?}"
            );
        }
        assert_eq!(
            preset(
                "Dup",
                vec![output("$textureSet_N"), output("$TEXTURESET_n")]
            )
            .validate(),
            Err(ExportPresetError::DuplicateFileName(
                "$TEXTURESET_n".to_owned()
            ))
        );
        let mut bright = output("bright");
        bright.slots[1] = ChannelSlot::Constant(2.0);
        assert_eq!(
            preset("Bright", vec![bright]).validate(),
            Err(ExportPresetError::InvalidConstant("bright".to_owned()))
        );

        let set = set_with(4, metal_layer([0, 0, 0]));
        assert_eq!(
            set.bake_export(&preset("Empty", Vec::new()), BakeContext::default()),
            Err(BakeError::InvalidExportPreset(
                ExportPresetError::InvalidOutputCount
            ))
        );

        // Projects keep custom presets, but not under a built-in's name.
        let mut project = MaterialProject::empty();
        project.export_presets = vec![preset("Mine", vec![output("x")])];
        project.validate().unwrap();
        project
            .export_presets
            .push(preset("unreal engine", vec![output("x")]));
        assert_eq!(
            project.validate(),
            Err(ProjectError::DuplicateExportPresetName(
                "unreal engine".to_owned()
            ))
        );
    }

    #[test]
    fn tc_0122_preflight_flags_blocking_errors_and_default_channel_outputs() {
        use crate::{FillTextures, ResourceId};

        let messages = |issues: &[PreflightIssue], severity| -> Vec<String> {
            issues
                .iter()
                .filter(|issue| issue.severity == severity)
                .map(|issue| issue.message.clone())
                .collect()
        };
        let orm = built_in("Metal/Rough (ORM)");
        let set = set_with(16, metal_layer([0, 0, 0]));

        // No mesh maps: the normal is flat and occlusion white, but nothing
        // blocks the export.
        let issues = set.export_preflight(&orm, BakeContext::default());
        assert!(messages(&issues, PreflightSeverity::Error).is_empty());
        assert_eq!(
            messages(&issues, PreflightSeverity::Warning),
            [
                "Normal is flat: no high-poly bake, normal, or height detail in \
                 Hull_Plate_Normal.",
                "Occlusion is white because mesh maps are not baked in Hull_Plate_ORM.",
            ]
        );
        let maps = cube_maps();
        let with_maps = BakeContext {
            mesh_maps: Some(&maps),
            ..BakeContext::default()
        };
        assert_eq!(
            set.export_preflight(&orm, with_maps).len(),
            1,
            "only flat normal"
        );

        // Unpainted channels are reported once per source, listing files.
        let separate = set_with(12, MaterialLayer::new_paint(LayerId(1), "Empty"));
        let issues = separate.export_preflight(&built_in("Separate channels"), with_maps);
        let warnings = messages(&issues, PreflightSeverity::Warning);
        assert!(warnings[0].contains("12 × 12 is not a power of two"));
        assert!(
            warnings
                .iter()
                .any(|message| message == "Roughness is a uniform 0.50 in Hull_Plate_Roughness.")
        );
        assert!(!warnings.iter().any(|message| message.contains("Occlusion")));

        // Blocking problems come first.
        let mut textured = metal_layer([0, 0, 0]);
        if let Some(fill) = &mut textured.fill {
            fill.textures = FillTextures {
                base_color: Some(ResourceId(9)),
                ..FillTextures::default()
            };
        }
        let clashing = ExportPreset {
            name: "Clash".to_owned(),
            outputs: vec![
                gray_output(
                    "$textureSet_A",
                    ChannelSlot::Source {
                        source: ExportSource::MeshMap(MeshMapKind::Thickness),
                        component: SourceComponent::Red,
                        invert: false,
                    },
                ),
                gray_output("Hull Plate_A", ChannelSlot::ONE),
            ],
        };
        let issues = set_with(16, textured).export_preflight(&clashing, BakeContext::default());
        assert_eq!(
            messages(&issues, PreflightSeverity::Error),
            [
                "Bitmap resource 9 is used by a layer but is not loaded.",
                "`$textureSet_A` and `Hull Plate_A` both resolve to `Hull_Plate_A`.",
                "Hull_Plate_A reads the Thickness mesh map, which is not baked.",
            ]
        );
        assert_eq!(issues[0].severity, PreflightSeverity::Error);

        let invalid = ExportPreset {
            name: String::new(),
            outputs: Vec::new(),
        };
        assert_eq!(
            set.export_preflight(&invalid, BakeContext::default()).len(),
            1
        );
    }
}
