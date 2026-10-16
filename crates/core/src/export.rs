//! Export presets: serializable descriptions of which files an export
//! writes and how material channels and mesh maps are routed, inverted, and
//! packed into their colour channels.

use std::collections::{BTreeMap, BTreeSet};

use glam::{Vec3, Vec4};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::compositing::{
    BakeContext, BakeError, BakedTexture, LayerMasks, base_texel, checked_bake_pixel_count,
    contributes, parallel_fill_region, rgb_unit, try_filled, unit_to_byte,
};
use crate::region::TexelRect;
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
    /// A specular/glossiness-workflow map converted from the metallic/
    /// roughness material, for engines that expect one.
    SpecGloss(SpecGlossMap),
}

/// Colour maps of the specular/glossiness workflow. Both are sRGB and are
/// mixed from Base Color and Metallic in linear light; glossiness is
/// inverted Roughness, which any slot can already produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SpecGlossMap {
    /// Base Color without the metal: black where fully metallic.
    Diffuse,
    /// Reflectance at normal incidence: 4 % grey for dielectrics, the Base
    /// Color for metals.
    Specular,
}

impl SpecGlossMap {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Diffuse => "Diffuse",
            Self::Specular => "Specular",
        }
    }

    /// The map's sRGB colour for sRGB `base` and `metallic` (`0..=1`).
    pub fn convert(self, base: Vec3, metallic: f32) -> Vec3 {
        use crate::color::{linear_to_srgb, srgb_to_linear};
        let linear = base.to_array().map(srgb_to_linear);
        let metallic = metallic.clamp(0.0, 1.0);
        linear
            .map(|value| match self {
                Self::Diffuse => linear_to_srgb(value * (1.0 - metallic)),
                Self::Specular => linear_to_srgb(0.04 + (value - 0.04) * metallic),
            })
            .into()
    }
}

impl ExportSource {
    /// Every source, in menu order.
    pub const ALL: [Self; 18] = [
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
        Self::SpecGloss(SpecGlossMap::Diffuse),
        Self::SpecGloss(SpecGlossMap::Specular),
    ];

    /// True when the source holds sRGB-encoded colour (Base Color and
    /// Emissive), which linear file formats such as EXR store decoded.
    /// User channels are written as stored, like other data.
    pub const fn is_srgb(self) -> bool {
        matches!(
            self,
            Self::Channel(MaterialChannel::BaseColor | MaterialChannel::Emissive)
                | Self::SpecGloss(_)
        )
    }

    /// True when the source carries different data in R, G, and B (user
    /// channels may: colour ones do, grey ones hold one value in all three).
    pub const fn is_color(self) -> bool {
        match self {
            Self::Channel(channel) => matches!(
                channel,
                MaterialChannel::BaseColor
                    | MaterialChannel::Emissive
                    | MaterialChannel::Normal
                    | MaterialChannel::Id
                    | MaterialChannel::User(_)
            ),
            Self::MeshMap(kind) => matches!(
                kind,
                MeshMapKind::Normal | MeshMapKind::WorldNormal | MeshMapKind::Position
            ),
            Self::SpecGloss(_) => true,
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

/// Bits per channel of an output file. PNG stores either; TGA holds 8 bits
/// and EXR always stores half floats.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitDepth {
    #[default]
    Eight,
    Sixteen,
}

impl BitDepth {
    pub const ALL: [Self; 2] = [Self::Eight, Self::Sixteen];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Eight => "8-bit",
            Self::Sixteen => "16-bit",
        }
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    const fn is_eight(&self) -> bool {
        matches!(self, Self::Eight)
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
    /// File format (one of [`crate::EXPORT_FILE_FORMATS`]) this output is
    /// always written in; `None` follows the format chosen for the export.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Bits per channel when written as PNG.
    #[serde(default, skip_serializing_if = "BitDepth::is_eight")]
    pub bit_depth: BitDepth,
}

impl ExportOutput {
    pub fn new(file_name: &str, layout: OutputLayout, slots: [ChannelSlot; 4]) -> Self {
        Self {
            file_name: file_name.to_owned(),
            layout,
            slots,
            skip_if_black: false,
            format: None,
            bit_depth: BitDepth::Eight,
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
    /// Largest texture set edge the target accepts, e.g. a mobile profile;
    /// preflight blocks larger sets. `None` accepts any size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_resolution: Option<u32>,
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
    #[error("output `{0}` names an unsupported file format")]
    UnknownFormat(String),
    #[error("output `{0}` is 16-bit, which TGA cannot store; write it as PNG or EXR")]
    DepthNeedsPng(String),
    #[error("the maximum resolution must be 1 to {max} texels", max = crate::MAX_TEXTURE_DIMENSION)]
    InvalidMaxResolution,
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
        let spec_gloss = |map, component| ChannelSlot::Source {
            source: ExportSource::SpecGloss(map),
            component,
            invert: false,
        };
        vec![
            Self {
                name: "Metal/Rough (ORM)".to_owned(),
                max_resolution: None,
                outputs: vec![
                    ExportOutput::new("$textureSet_BaseColor", Rgba, base_with_opacity),
                    ExportOutput::new("$textureSet_ORM", Rgb, orm),
                    ExportOutput::new("$textureSet_Normal", Rgb, rgb(Normal)),
                    emissive("$textureSet_Emissive"),
                ],
            },
            Self {
                name: "Unreal Engine".to_owned(),
                max_resolution: None,
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
                max_resolution: None,
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
            // Godot 4 reads OpenGL normals and packs occlusion, roughness,
            // and metallic like glTF.
            Self {
                name: "Godot 4 (StandardMaterial3D)".to_owned(),
                max_resolution: None,
                outputs: vec![
                    ExportOutput::new("$textureSet_albedo", Rgba, base_with_opacity),
                    ExportOutput::new("$textureSet_orm", Rgb, orm),
                    ExportOutput::new("$textureSet_normal", Rgb, rgb(Normal)),
                    ExportOutput::new("$textureSet_height", Gray, gray(Height)),
                    emissive("$textureSet_emission"),
                ],
            },
            Self {
                name: "glTF 2.0 (metallic-roughness)".to_owned(),
                max_resolution: None,
                outputs: vec![
                    ExportOutput::new("$textureSet_baseColor", Rgba, base_with_opacity),
                    ExportOutput::new("$textureSet_occlusionRoughnessMetallic", Rgb, orm),
                    ExportOutput::new("$textureSet_normal", Rgb, rgb(Normal)),
                    emissive("$textureSet_emissive"),
                ],
            },
            Self {
                name: "Specular/Glossiness".to_owned(),
                max_resolution: None,
                outputs: vec![
                    ExportOutput::new(
                        "$textureSet_Diffuse",
                        Rgba,
                        [
                            spec_gloss(SpecGlossMap::Diffuse, Red),
                            spec_gloss(SpecGlossMap::Diffuse, Green),
                            spec_gloss(SpecGlossMap::Diffuse, Blue),
                            ChannelSlot::channel(Opacity, Red),
                        ],
                    ),
                    ExportOutput::new(
                        "$textureSet_Specular",
                        Rgb,
                        [
                            spec_gloss(SpecGlossMap::Specular, Red),
                            spec_gloss(SpecGlossMap::Specular, Green),
                            spec_gloss(SpecGlossMap::Specular, Blue),
                            ChannelSlot::ONE,
                        ],
                    ),
                    ExportOutput::new(
                        "$textureSet_Glossiness",
                        Gray,
                        [
                            ChannelSlot::inverted(Roughness, Red),
                            ChannelSlot::ZERO,
                            ChannelSlot::ZERO,
                            ChannelSlot::ONE,
                        ],
                    ),
                    ExportOutput::new("$textureSet_Normal", Rgb, rgb(Normal)),
                    ExportOutput::new("$textureSet_AmbientOcclusion", Gray, gray(AmbientOcclusion)),
                    emissive("$textureSet_Emissive"),
                ],
            },
            Self {
                name: "Separate channels".to_owned(),
                max_resolution: None,
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
        if self
            .max_resolution
            .is_some_and(|edge| edge == 0 || edge > crate::MAX_TEXTURE_DIMENSION)
        {
            return Err(ExportPresetError::InvalidMaxResolution);
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
            if let Some(format) = &output.format
                && !crate::EXPORT_FILE_FORMATS.contains(&format.as_str())
            {
                return Err(ExportPresetError::UnknownFormat(output.file_name.clone()));
            }
            if output.bit_depth == BitDepth::Sixteen && output.format.as_deref() == Some("tga") {
                return Err(ExportPresetError::DepthNeedsPng(output.file_name.clone()));
            }
        }
        Ok(())
    }

    /// The preset with every output's file format settled: outputs without
    /// their own format take `default_format`, except that 16-bit outputs
    /// stay PNG when the default (TGA) cannot hold them.
    pub fn resolved(&self, default_format: &str) -> Self {
        let mut resolved = self.clone();
        for output in &mut resolved.outputs {
            if output.format.is_none() {
                let keeps_png = output.bit_depth == BitDepth::Sixteen && default_format == "tga";
                output.format = Some(if keeps_png { "png" } else { default_format }.to_owned());
            }
        }
        resolved
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
    /// File format the output is written in (an extension); `None` uses
    /// the format the writer is given.
    pub format: Option<String>,
    /// Bits per channel when written as PNG.
    pub bit_depth: BitDepth,
    /// RGBA16 samples laid out like `texture`, present when the output is
    /// 16-bit or EXR. They come from the unquantized composite, so they
    /// keep detail the 8-bit texture rounds away.
    pub rgba16: Option<Vec<u16>>,
    /// Channels (R, G, B, A) holding sRGB-encoded colour, which linear
    /// formats such as EXR store decoded.
    pub srgb: [bool; 4],
}

/// True when an output needs more than 8 bits per channel.
fn needs_high_precision(output: &ExportOutput) -> bool {
    output.bit_depth == BitDepth::Sixteen || output.format.as_deref() == Some("exr")
}

fn unit_to_word(value: f32) -> u16 {
    (value.clamp(0.0, 1.0) * 65_535.0).round() as u16
}

/// One texel of an output from its slot samples: Gray repeats slot 0, and
/// Gray and RGB are opaque.
fn pack<T: Copy>(layout: OutputLayout, opaque: T, sample: impl Fn(usize) -> T) -> [T; 4] {
    match layout {
        OutputLayout::Gray => {
            let gray = sample(0);
            [gray, gray, gray, opaque]
        }
        OutputLayout::Rgb => [sample(0), sample(1), sample(2), opaque],
        OutputLayout::Rgba => [sample(0), sample(1), sample(2), sample(3)],
    }
}

/// One export source over a region: RGBA8 (row-major), plus the
/// unquantized values when a high-precision output reads it.
struct SourcePlane {
    rgba8: Vec<u8>,
    values: Option<Vec<Vec3>>,
}

impl TextureSet {
    /// Bakes every file `preset` describes. Material channels are composited
    /// once and shared between the outputs that read them.
    pub fn bake_export(
        &self,
        preset: &ExportPreset,
        context: BakeContext<'_>,
    ) -> Result<Vec<ExportedTexture>, BakeError> {
        self.bake_export_in(preset, context, &self.bake_tiles()?)
    }

    /// [`Self::bake_export`] composited tile by tile.
    pub(crate) fn bake_export_in(
        &self,
        preset: &ExportPreset,
        context: BakeContext<'_>,
        tiles: &[TexelRect],
    ) -> Result<Vec<ExportedTexture>, BakeError> {
        preset.validate()?;
        self.validate_for_region(TexelRect::full(1, 1))?;
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
                ExportSource::SpecGloss(_) => {
                    channels.extend([MaterialChannel::BaseColor, MaterialChannel::Metallic]);
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
        // An RGBA8 texture per output, then an RGBA16 one per high-precision
        // output.
        let high = preset
            .outputs
            .iter()
            .filter(|output| needs_high_precision(output))
            .count();
        let mut texel_bytes = vec![4; preset.outputs.len()];
        texel_bytes.extend(std::iter::repeat_n(8, high));
        let mut textures = self
            .bake_tiles_into(tiles, &texel_bytes, |region| {
                self.export_region(preset, &sources, &channels, context, region)
            })?
            .into_iter();
        let eight: Vec<Vec<u8>> = textures.by_ref().take(preset.outputs.len()).collect();
        let mut sixteen = textures;

        let mut exported = Vec::new();
        for (output, rgba8) in preset.outputs.iter().zip(eight) {
            let rgba16 = needs_high_precision(output).then(|| {
                sixteen
                    .next()
                    .unwrap_or_default()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| u16::from_le_bytes(*bytes))
                    .collect::<Vec<u16>>()
            });
            if output.skip_if_black
                && rgba8
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|texel| texel[..3] == [0, 0, 0])
            {
                continue;
            }
            let srgb_slot = |slot: &ChannelSlot| slot.source().is_some_and(ExportSource::is_srgb);
            let srgb = match output.layout {
                OutputLayout::Gray => {
                    let gray = srgb_slot(&output.slots[0]);
                    [gray, gray, gray, false]
                }
                OutputLayout::Rgb => [
                    srgb_slot(&output.slots[0]),
                    srgb_slot(&output.slots[1]),
                    srgb_slot(&output.slots[2]),
                    false,
                ],
                OutputLayout::Rgba => [
                    srgb_slot(&output.slots[0]),
                    srgb_slot(&output.slots[1]),
                    srgb_slot(&output.slots[2]),
                    srgb_slot(&output.slots[3]),
                ],
            };
            exported.push(ExportedTexture {
                file_stem: output.resolved_file_name(&self.name),
                layout: output.layout,
                texture: self.texture(rgba8),
                format: output.format.clone(),
                bit_depth: output.bit_depth,
                rgba16,
                srgb,
            });
        }
        Ok(exported)
    }

    /// Every output of `preset` over `region`, row-major within it: an
    /// RGBA8 buffer per output, then little-endian RGBA16 buffers for the
    /// high-precision outputs in order. Each of `sources` (reading
    /// `channels`) is baked once and shared.
    fn export_region(
        &self,
        preset: &ExportPreset,
        sources: &[ExportSource],
        channels: &[MaterialChannel],
        context: BakeContext<'_>,
        region: TexelRect,
    ) -> Result<Vec<Vec<u8>>, BakeError> {
        let id_map = self.validate_for_region(region)?;
        let id_map = id_map.as_deref();
        let masks = self.layer_masks_in(channels, context, id_map, region)?;
        let pixel_count = region.len();
        let precise: BTreeSet<ExportSource> = preset
            .outputs
            .iter()
            .filter(|output| needs_high_precision(output))
            .flat_map(|output| output.used_slots().iter().filter_map(|slot| slot.source()))
            .collect();

        let mut planes = BTreeMap::new();
        for batch in sources.chunks(SOURCE_WORKERS) {
            let baked: Vec<Result<SourcePlane, BakeError>> = std::thread::scope(|scope| {
                let workers: Vec<_> = batch
                    .iter()
                    .map(|&source| {
                        let (masks, keep_values) = (&masks, precise.contains(&source));
                        scope.spawn(move || {
                            self.export_source(source, context, masks, id_map, region, keep_values)
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|worker| worker.join().unwrap_or(Err(BakeError::AllocationFailed)))
                    .collect()
            });
            for (source, plane) in batch.iter().zip(baked) {
                planes.insert(*source, plane?);
            }
        }

        let slot_byte = |slot: &ChannelSlot, index: usize| -> u8 {
            match *slot {
                ChannelSlot::Constant(value) => unit_to_byte(value),
                ChannelSlot::Source {
                    source,
                    component,
                    invert,
                } => {
                    let value = planes.get(&source).map_or(0, |plane: &SourcePlane| {
                        plane.rgba8[index * 4 + component.index()]
                    });
                    if invert { 255 - value } else { value }
                }
            }
        };
        let slot_word = |slot: &ChannelSlot, index: usize| -> u16 {
            match *slot {
                ChannelSlot::Constant(value) => unit_to_word(value),
                ChannelSlot::Source {
                    source,
                    component,
                    invert,
                } => {
                    let value = planes.get(&source).map_or(0, |plane: &SourcePlane| {
                        match &plane.values {
                            Some(values) => unit_to_word(values[index][component.index()]),
                            // Discrete data (the ID map) widens exactly.
                            None => u16::from(plane.rgba8[index * 4 + component.index()]) * 257,
                        }
                    });
                    if invert { u16::MAX - value } else { value }
                }
            }
        };
        let mut buffers = Vec::new();
        for output in &preset.outputs {
            let mut rgba8 = Vec::new();
            rgba8
                .try_reserve_exact(pixel_count * 4)
                .map_err(|_| BakeError::AllocationFailed)?;
            for index in 0..pixel_count {
                rgba8.extend(pack(output.layout, 255, |slot| {
                    slot_byte(&output.slots[slot], index)
                }));
            }
            buffers.push(rgba8);
        }
        for output in preset
            .outputs
            .iter()
            .filter(|output| needs_high_precision(output))
        {
            let mut rgba16 = Vec::new();
            rgba16
                .try_reserve_exact(pixel_count * 8)
                .map_err(|_| BakeError::AllocationFailed)?;
            for index in 0..pixel_count {
                let words = pack(output.layout, u16::MAX, |slot| {
                    slot_word(&output.slots[slot], index)
                });
                for word in words {
                    rgba16.extend(word.to_le_bytes());
                }
            }
            buffers.push(rgba16);
        }
        Ok(buffers)
    }

    /// One source over `region`, row-major within it, keeping its
    /// unquantized values when `keep_values`.
    fn export_source(
        &self,
        source: ExportSource,
        context: BakeContext<'_>,
        masks: &LayerMasks,
        id_map: Option<&[u8]>,
        region: TexelRect,
        keep_values: bool,
    ) -> Result<SourcePlane, BakeError> {
        let (width, height) = (self.resolution.width, self.resolution.height);
        let pixel_count = region.len();
        let plane = |values: Vec<Vec3>| -> Result<SourcePlane, BakeError> {
            let mut rgba8 = Vec::new();
            rgba8
                .try_reserve_exact(values.len() * 4)
                .map_err(|_| BakeError::AllocationFailed)?;
            for value in &values {
                rgba8.extend([
                    unit_to_byte(value.x),
                    unit_to_byte(value.y),
                    unit_to_byte(value.z),
                    255,
                ]);
            }
            Ok(SourcePlane {
                rgba8,
                values: keep_values.then_some(values),
            })
        };
        let channel = match source {
            ExportSource::MeshMap(kind) => {
                let maps = context.mesh_maps.ok_or(BakeError::MissingMeshMaps)?;
                let mut values = try_filled(pixel_count, Vec3::ZERO)?;
                parallel_fill_region(&mut values, region, width, height, |uv| {
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
                return plane(values);
            }
            ExportSource::SpecGloss(map) => {
                let base = self.base_color_values(context, masks, region)?;
                let metallic =
                    self.optional_plane(MaterialChannel::Metallic, context, masks, region)?;
                let values = base
                    .iter()
                    .enumerate()
                    .map(|(index, color)| {
                        let metal = metallic
                            .as_ref()
                            .map_or(DEFAULT_METALLIC, |plane| plane[index].x);
                        map.convert(*color, metal)
                    })
                    .collect();
                return plane(values);
            }
            ExportSource::Channel(channel) => channel,
        };
        match channel {
            MaterialChannel::Id => Ok(SourcePlane {
                rgba8: match id_map {
                    Some(id_map) => id_map.to_vec(),
                    None => self.id_map_region(region)?,
                },
                values: None,
            }),
            MaterialChannel::AmbientOcclusion => {
                let mut values = try_filled(pixel_count, Vec3::ONE)?;
                if let Some(maps) = context.mesh_maps {
                    parallel_fill_region(&mut values, region, width, height, |uv| {
                        let sample = maps.sample(uv);
                        Vec3::splat(if sample.covered {
                            sample.ambient_occlusion
                        } else {
                            1.0
                        })
                    });
                }
                plane(values)
            }
            MaterialChannel::Normal => {
                let normals = self.normal_region(context, masks, region)?;
                plane(
                    normals
                        .iter()
                        .map(|normal| normal.truncate() * 0.5 + 0.5)
                        .collect(),
                )
            }
            MaterialChannel::BaseColor | MaterialChannel::User(_) => {
                // Over the neutral grey, or the user channel's default.
                let default = rgb_unit(match self.user_channel(channel) {
                    Some(user) => user.default,
                    None if channel == MaterialChannel::BaseColor => DEFAULT_BASE_COLOR,
                    None => [0, 0, 0],
                });
                plane(
                    match self.optional_plane(channel, context, masks, region)? {
                        Some(composite) => composite
                            .iter()
                            .map(|texel| texel.truncate() * texel.w + default * (1.0 - texel.w))
                            .collect(),
                        None => try_filled(pixel_count, default)?,
                    },
                )
            }
            channel => {
                let composite = match self.optional_plane(channel, context, masks, region)? {
                    Some(composite) => composite,
                    None => try_filled(pixel_count, base_texel(channel)?)?,
                };
                plane(composite.iter().map(|texel| texel.truncate()).collect())
            }
        }
    }

    /// Base Color over the neutral default, as the material bake shows it.
    fn base_color_values(
        &self,
        context: BakeContext<'_>,
        masks: &LayerMasks,
        region: TexelRect,
    ) -> Result<Vec<Vec3>, BakeError> {
        let default_base = rgb_unit(DEFAULT_BASE_COLOR);
        Ok(
            match self.optional_plane(MaterialChannel::BaseColor, context, masks, region)? {
                Some(composite) => composite
                    .iter()
                    .map(|texel| texel.truncate() * texel.w + default_base * (1.0 - texel.w))
                    .collect(),
                None => try_filled(region.len(), default_base)?,
            },
        )
    }

    /// The composited plane, or `None` when no layer paints `channel`.
    fn optional_plane(
        &self,
        channel: MaterialChannel,
        context: BakeContext<'_>,
        masks: &LayerMasks,
        region: TexelRect,
    ) -> Result<Option<Vec<Vec4>>, BakeError> {
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
        if let Some(limit) = preset.max_resolution
            && width.max(height) > limit
        {
            issues.push(PreflightIssue::error(format!(
                "{width} × {height} is larger than {limit} × {limit}, the most `{}` accepts.",
                preset.name
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
                .any(|layer| layer.shown() && layer.polygon_mask.is_some())
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
                ExportSource::SpecGloss(_) => None,
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
                    MaterialChannel::User(slot) => match self.user_channel(channel) {
                        None => Some(format!(
                            "User channel {slot} is not defined on this set, so it is black"
                        )),
                        Some(user) if !painted(&[channel]) => {
                            Some(format!("{} is its uniform default", user.name))
                        }
                        Some(_) => None,
                    },
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
            user_channels: Default::default(),
            stencils: Default::default(),
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
    fn tc_0264_engine_profiles_and_spec_gloss_maps_convert_the_material() {
        // Conversions: a dielectric keeps its colour as diffuse and reflects
        // 4 %; a metal has no diffuse and reflects its colour.
        let base = Vec3::new(0.8, 0.4, 0.2);
        assert!(
            (SpecGlossMap::Diffuse.convert(base, 0.0) - base)
                .abs()
                .max_element()
                < 1e-5
        );
        let four_percent = crate::color::linear_to_srgb(0.04);
        assert!(
            (SpecGlossMap::Specular.convert(base, 0.0) - Vec3::splat(four_percent))
                .abs()
                .max_element()
                < 1e-5
        );
        assert_eq!(SpecGlossMap::Diffuse.convert(base, 1.0), Vec3::ZERO);
        assert!(
            (SpecGlossMap::Specular.convert(base, 1.0) - base)
                .abs()
                .max_element()
                < 1e-5
        );

        let set = set_with(4, metal_layer([0, 0, 0]));
        let context = BakeContext::default();
        let files = set
            .bake_export(&built_in("Specular/Glossiness"), context)
            .unwrap();
        let file = |suffix: &str| {
            files
                .iter()
                .find(|file| file.file_stem.ends_with(suffix))
                .unwrap()
        };
        // Fully metallic: black diffuse with the opacity in alpha, the
        // base colour as specular, glossiness 1 - 0.2.
        assert_eq!(file("_Diffuse").texture.rgba8[..4], [0, 0, 0, 128]);
        assert_eq!(file("_Specular").texture.rgba8[..3], [200, 100, 50]);
        assert_eq!(file("_Glossiness").texture.rgba8[0], 204);
        assert!(file("_Diffuse").srgb[..3].iter().all(|srgb| *srgb));

        for (preset, stems) in [
            (
                "Godot 4 (StandardMaterial3D)",
                &["_albedo", "_orm", "_normal", "_height"][..],
            ),
            (
                "glTF 2.0 (metallic-roughness)",
                &["_baseColor", "_occlusionRoughnessMetallic", "_normal"][..],
            ),
        ] {
            let preset = built_in(preset);
            assert_eq!(preset.validate(), Ok(()));
            let files = set.bake_export(&preset, context).unwrap();
            for stem in stems {
                assert!(
                    files.iter().any(|file| file.file_stem.ends_with(stem)),
                    "{} writes {stem}",
                    preset.name
                );
            }
            // Packed like the ORM preset: occlusion, roughness, metallic.
            let orm = files
                .iter()
                .find(|file| file.file_stem.contains("orm") || file.file_stem.contains("Metallic"))
                .unwrap();
            assert_eq!(orm.texture.rgba8[..3], [255, 51, 255]);
        }
    }

    #[test]
    fn tc_0262_outputs_carry_their_own_format_depth_and_size_limit() {
        // A soft stroke gives a gradient whose values fall between 8-bit
        // steps.
        let mut paint = MaterialLayer::new_paint(LayerId(1), "Paint");
        paint.channels = BTreeSet::from([MaterialChannel::Roughness]);
        paint.paint_strokes.push(crate::PaintStroke {
            roughness: 0.9,
            ..crate::PaintStroke::new(vec![glam::Vec2::splat(0.5)], 20.0, [255, 255, 255, 255])
        });
        let set = set_with(64, paint);
        let gray = |name: &str| ExportOutput {
            ..ExportOutput::new(
                name,
                OutputLayout::Gray,
                [
                    ChannelSlot::channel(MaterialChannel::Roughness, SourceComponent::Red),
                    ChannelSlot::ZERO,
                    ChannelSlot::ZERO,
                    ChannelSlot::ONE,
                ],
            )
        };
        let preset = ExportPreset {
            name: "Deep".to_owned(),
            outputs: vec![
                gray("Plain"),
                ExportOutput {
                    bit_depth: BitDepth::Sixteen,
                    ..gray("Deep")
                },
                ExportOutput {
                    format: Some("exr".to_owned()),
                    ..gray("Float")
                },
            ],
            max_resolution: None,
        };
        let files = set.bake_export(&preset, BakeContext::default()).unwrap();
        assert!(files[0].rgba16.is_none(), "8-bit outputs carry no samples");
        let words = files[1].rgba16.as_ref().unwrap();
        assert_eq!(files[1].texture, files[0].texture, "8-bit view unchanged");
        assert!(
            words.iter().any(|word| word % 257 != 0),
            "16-bit samples keep detail between 8-bit steps"
        );
        // Every 16-bit sample rounds to its 8-bit texel.
        for (word, byte) in words.iter().zip(&files[1].texture.rgba8) {
            assert_eq!(((f32::from(*word) / 257.0).round()) as u8, *byte);
        }
        assert_eq!(files[2].format.as_deref(), Some("exr"));
        assert!(files[2].rgba16.is_some(), "EXR outputs get full precision");

        // Formats settle against the export's choice; 16-bit outputs stay
        // PNG when the choice is TGA.
        let resolved = preset.resolved("tga");
        let formats: Vec<_> = resolved
            .outputs
            .iter()
            .map(|output| output.format.clone().unwrap())
            .collect();
        assert_eq!(formats, ["tga", "png", "exr"]);
        assert_eq!(
            preset.resolved("exr").outputs[0].format.as_deref(),
            Some("exr")
        );

        let invalid = |output: ExportOutput| ExportPreset {
            outputs: vec![output],
            ..preset.clone()
        };
        assert_eq!(
            invalid(ExportOutput {
                format: Some("bmp".to_owned()),
                ..gray("Odd")
            })
            .validate(),
            Err(ExportPresetError::UnknownFormat("Odd".to_owned()))
        );
        assert_eq!(
            invalid(ExportOutput {
                format: Some("tga".to_owned()),
                bit_depth: BitDepth::Sixteen,
                ..gray("Deep")
            })
            .validate(),
            Err(ExportPresetError::DepthNeedsPng("Deep".to_owned()))
        );

        // A preset's size limit blocks larger sets in preflight.
        let mobile = ExportPreset {
            max_resolution: Some(32),
            ..preset.clone()
        };
        let issues = set.export_preflight(&mobile, BakeContext::default());
        assert!(issues.iter().any(|issue| {
            issue.severity == PreflightSeverity::Error && issue.message.contains("32 × 32")
        }));
        assert!(
            set.export_preflight(&preset, BakeContext::default())
                .iter()
                .all(|issue| issue.severity != PreflightSeverity::Error)
        );
        assert_eq!(
            ExportPreset {
                max_resolution: Some(0),
                ..preset.clone()
            }
            .validate(),
            Err(ExportPresetError::InvalidMaxResolution)
        );
        // Saved presets read without the new fields.
        let old: ExportPreset = serde_json::from_str(
            r#"{"name":"Old","outputs":[{"file_name":"X","layout":"Gray","slots":[{"Constant":0.0},{"Constant":0.0},{"Constant":0.0},{"Constant":1.0}]}]}"#,
        )
        .unwrap();
        assert_eq!(old.max_resolution, None);
        assert_eq!(old.outputs[0].bit_depth, BitDepth::Eight);
        assert_eq!(old.outputs[0].format, None);
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
            max_resolution: None,
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
            max_resolution: None,
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
            max_resolution: None,
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
            max_resolution: None,
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
            max_resolution: None,
            name: String::new(),
            outputs: Vec::new(),
        };
        assert_eq!(
            set.export_preflight(&invalid, BakeContext::default()).len(),
            1
        );
    }

    #[test]
    fn tc_0400_sixteen_bit_bitmaps_keep_precision_through_bakes_and_exports() {
        use crate::{FillTextures, ResourceId, ResourceImages, TextureImage};
        use std::sync::Arc;

        // 1000 / 65535 sits between the 8-bit steps 3 and 4.
        let word = 1000_u16;
        let image = TextureImage::new(4, 4, [4, 4, 4, 255].repeat(16))
            .unwrap()
            .with_rgba16([word, word, word, u16::MAX].repeat(16))
            .unwrap();
        assert!(
            TextureImage::new(4, 4, vec![0; 64])
                .unwrap()
                .with_rgba16(vec![0; 63])
                .is_none(),
            "the 16-bit copy must match the dimensions"
        );
        let expected = f32::from(word) / 65535.0;
        assert!((image.pixel(5).x - expected).abs() < 1e-6);
        assert!((image.sample(glam::Vec2::splat(0.3), true).y - expected).abs() < 1e-6);

        let resources = ResourceImages::from([(ResourceId(3), Arc::new(image))]);
        let context = BakeContext {
            resources: Some(&resources),
            ..BakeContext::default()
        };
        let preset = ExportPreset {
            name: "Deep".to_owned(),
            outputs: vec![ExportOutput {
                bit_depth: BitDepth::Sixteen,
                ..ExportOutput::new(
                    "Rough",
                    OutputLayout::Gray,
                    [
                        ChannelSlot::channel(MaterialChannel::Roughness, SourceComponent::Red),
                        ChannelSlot::ZERO,
                        ChannelSlot::ZERO,
                        ChannelSlot::ONE,
                    ],
                )
            }],
            max_resolution: None,
        };
        // Same size as the set (texels copied) and resampled.
        for size in [4, 8] {
            let mut layer = MaterialLayer::new_fill(
                LayerId(1),
                "Rough",
                LayerFill {
                    textures: FillTextures {
                        roughness: Some(ResourceId(3)),
                        ..FillTextures::default()
                    },
                    ..LayerFill::default()
                },
            );
            layer.channels = BTreeSet::from([MaterialChannel::Roughness]);
            let mut set = set_with(size, layer);
            set.channels = BTreeSet::from([MaterialChannel::Roughness]);
            let files = set.bake_export(&preset, context).unwrap();
            let words = files[0].rgba16.as_ref().unwrap();
            assert!(
                words
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|texel| texel[0].abs_diff(word) <= 1),
                "{size}: {:?}",
                &words[..4]
            );
        }
    }
}
