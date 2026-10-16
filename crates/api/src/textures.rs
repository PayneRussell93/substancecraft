//! Bitmap import (PNG, JPEG, TGA, BMP, TIFF, WebP, OpenEXR, Radiance HDR)
//! and baked-texture export (PNG, TGA, and OpenEXR).

use std::io::{Cursor, Read};
use std::path::Path;

use image::{
    DynamicImage, GrayImage, ImageBuffer, ImageFormat, ImageReader, Limits, RgbImage, RgbaImage,
};
use substancecraft_core::color::srgb_to_linear;
use substancecraft_core::{
    BakedTexture, BitDepth, ColorSpace, ExportedTexture, MAX_RESOURCE_DIMENSION, OutputLayout,
    TextureImage,
};
use thiserror::Error;

/// Largest bitmap file accepted for import.
pub const MAX_TEXTURE_FILE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TextureIoError {
    #[error("texture file I/O failed: {0}")]
    Io(String),
    #[error("texture file exceeds the {} MiB limit", MAX_TEXTURE_FILE_BYTES / (1024 * 1024))]
    TooLarge,
    #[error("texture could not be decoded: {0}")]
    Decode(String),
    #[error("texture is {width}x{height}; edges must be 1..={MAX_RESOURCE_DIMENSION} pixels")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("texture could not be encoded: {0}")]
    Encode(String),
    #[error("unsupported texture format `{0}`; use .png, .tga, or .exr")]
    UnsupportedFormat(String),
}

/// Raster file extensions imported as bitmap resources (besides SVG and
/// `.sctext`, which are rasterized).
pub const BITMAP_EXTENSIONS: [&str; 10] = [
    "png", "jpg", "jpeg", "tga", "bmp", "tif", "tiff", "webp", "exr", "hdr",
];

/// The colour space a file's pixels are encoded in when nothing says
/// otherwise: linear for floating-point formats (OpenEXR, Radiance HDR),
/// sRGB for the rest.
pub fn source_color_space(path: impl AsRef<Path>) -> ColorSpace {
    let extension = path
        .as_ref()
        .extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase());
    match extension.as_deref() {
        Some("exr" | "hdr") => ColorSpace::LinearSrgb,
        _ => ColorSpace::Srgb,
    }
}

/// Decodes PNG, JPEG, TGA, BMP, TIFF, WebP, OpenEXR, or Radiance HDR bytes
/// into straight-alpha RGBA8. Sources with more than 8 bits per channel
/// also keep a 16-bit copy ([`TextureImage::rgba16`]); floating-point ones
/// are clamped to `0..=1` and tagged linear. TGA has no
/// signature, so bytes no other format claims are tried as TGA.
pub fn decode_texture_image(bytes: &[u8]) -> Result<TextureImage, TextureIoError> {
    decode_with_hint(bytes, None)
}

fn decode_with_hint(
    bytes: &[u8],
    hint: Option<ImageFormat>,
) -> Result<TextureImage, TextureIoError> {
    if bytes.len() > MAX_TEXTURE_FILE_BYTES {
        return Err(TextureIoError::TooLarge);
    }
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| TextureIoError::Decode(error.to_string()))?;
    if reader.format().is_none() {
        reader.set_format(hint.unwrap_or(ImageFormat::Tga));
    }
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_RESOURCE_DIMENSION);
    limits.max_image_height = Some(MAX_RESOURCE_DIMENSION);
    // Room for the largest image as RGBA 32-bit float.
    limits.max_alloc = Some(u64::from(MAX_RESOURCE_DIMENSION).pow(2) * 16);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|error| TextureIoError::Decode(error.to_string()))?;
    let (width, height) = (image.width(), image.height());
    let invalid = TextureIoError::InvalidDimensions { width, height };
    match image {
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_) => {
            // Alpha is read as straight, as SubstanceCraft writes it, so
            // exported EXRs import back unchanged.
            let mut rgba = image.into_rgba32f().into_raw();
            for value in &mut rgba {
                *value = if value.is_finite() {
                    value.clamp(0.0, 1.0)
                } else {
                    0.0
                };
            }
            let rgba8 = rgba
                .iter()
                .map(|value| (value * 255.0).round() as u8)
                .collect();
            let rgba16 = rgba
                .iter()
                .map(|value| (value * 65535.0).round() as u16)
                .collect();
            let mut texture = TextureImage::new(width, height, rgba8)
                .and_then(|texture| texture.with_rgba16(rgba16))
                .ok_or(invalid)?;
            texture.color_space = ColorSpace::LinearSrgb;
            Ok(texture)
        }
        DynamicImage::ImageLuma16(_)
        | DynamicImage::ImageLumaA16(_)
        | DynamicImage::ImageRgb16(_)
        | DynamicImage::ImageRgba16(_) => {
            let rgba16 = image.to_rgba16().into_raw();
            TextureImage::new(width, height, image.into_rgba8().into_raw())
                .and_then(|texture| texture.with_rgba16(rgba16))
                .ok_or(invalid)
        }
        _ => TextureImage::new(width, height, image.into_rgba8().into_raw()).ok_or(invalid),
    }
}

/// Reads and decodes a bitmap file.
pub fn load_texture_image(path: impl AsRef<Path>) -> Result<TextureImage, TextureIoError> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase());
    match extension.as_deref() {
        Some("svg") => {
            let data =
                std::fs::read(path).map_err(|error| TextureIoError::Io(error.to_string()))?;
            if data.len() > MAX_TEXTURE_FILE_BYTES {
                return Err(TextureIoError::TooLarge);
            }
            return crate::vector::rasterize_svg(&data, crate::vector::DEFAULT_SVG_EDGE);
        }
        Some(crate::vector::TEXT_RESOURCE_EXTENSION) => {
            return crate::vector::load_text_resource(path);
        }
        _ => {}
    }
    let file = std::fs::File::open(path).map_err(|error| TextureIoError::Io(error.to_string()))?;
    let mut bytes = Vec::new();
    file.take(MAX_TEXTURE_FILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| TextureIoError::Io(error.to_string()))?;
    decode_with_hint(&bytes, ImageFormat::from_path(path).ok())
}

/// Output formats for baked textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFormat {
    Png,
    Tga,
    /// OpenEXR with half-float channels and lossless ZIP compression; colour
    /// channels are stored linear.
    Exr,
}

impl TextureFormat {
    pub const ALL: [Self; 3] = [Self::Png, Self::Tga, Self::Exr];

    pub const fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Tga => "tga",
            Self::Exr => "exr",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Tga => "Targa",
            Self::Exr => "OpenEXR",
        }
    }

    /// The format with this extension (case-insensitive), if supported.
    pub fn from_extension(extension: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|format| format.extension().eq_ignore_ascii_case(extension))
    }

    /// Format implied by a file extension (case-insensitive).
    pub fn from_path(path: &Path) -> Result<Self, TextureIoError> {
        let extension = path
            .extension()
            .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        Self::ALL
            .into_iter()
            .find(|format| format.extension() == extension)
            .ok_or(TextureIoError::UnsupportedFormat(extension))
    }

    /// The `image` crate format, or `None` for EXR, which is written here.
    fn image_format(self) -> Option<ImageFormat> {
        match self {
            Self::Png => Some(ImageFormat::Png),
            Self::Tga => Some(ImageFormat::Tga),
            Self::Exr => None,
        }
    }
}

/// Encodes a baked texture (RGBA8, rows top to bottom). EXR stores the
/// values as they are (`0..=1`), without colour decoding.
pub fn encode_texture(
    texture: &BakedTexture,
    format: TextureFormat,
) -> Result<Vec<u8>, TextureIoError> {
    let Some(image_format) = format.image_format() else {
        let unit = |byte: u8| f32::from(byte) / 255.0;
        let channel = |component: usize| {
            texture
                .rgba8
                .as_chunks::<4>()
                .0
                .iter()
                .map(|texel| unit(texel[component]))
                .collect::<Vec<f32>>()
        };
        return encode_exr(
            texture.width,
            texture.height,
            vec![
                ("R", channel(0)),
                ("G", channel(1)),
                ("B", channel(2)),
                ("A", channel(3)),
            ],
        );
    };
    let image = RgbaImage::from_raw(texture.width, texture.height, texture.rgba8.clone())
        .ok_or_else(|| {
            TextureIoError::Encode("RGBA buffer does not match its dimensions".to_owned())
        })?;
    let mut bytes = Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, image_format)
        .map_err(|error| TextureIoError::Encode(error.to_string()))?;
    Ok(bytes.into_inner())
}

/// Writes a baked texture in the format named by the path's extension.
pub fn write_texture(path: impl AsRef<Path>, texture: &BakedTexture) -> Result<(), TextureIoError> {
    let path = path.as_ref();
    let bytes = encode_texture(texture, TextureFormat::from_path(path)?)?;
    std::fs::write(path, bytes).map_err(|error| TextureIoError::Io(error.to_string()))
}

/// The format `file` is written in: its own, or else `default`.
pub fn exported_texture_format(
    file: &ExportedTexture,
    default: TextureFormat,
) -> Result<TextureFormat, TextureIoError> {
    match &file.format {
        Some(extension) => TextureFormat::from_extension(extension)
            .ok_or_else(|| TextureIoError::UnsupportedFormat(extension.clone())),
        None => Ok(default),
    }
}

/// Encodes a packed export output with its own channel count (one channel
/// for Gray, three for RGB, four for RGBA) in its own format, or `default`
/// when it names none. 16-bit outputs write 16-bit PNGs; EXR writes half
/// floats with sRGB colour channels decoded to linear.
pub fn encode_exported_texture(
    file: &ExportedTexture,
    default: TextureFormat,
) -> Result<Vec<u8>, TextureIoError> {
    let format = exported_texture_format(file, default)?;
    let texture = &file.texture;
    let (width, height) = (texture.width, texture.height);
    let mismatch =
        || TextureIoError::Encode("RGBA buffer does not match its dimensions".to_owned());
    let pixel_count = width as usize * height as usize;
    // 16-bit samples: the precise ones when baked, else widened bytes.
    let words = || -> Result<Vec<u16>, TextureIoError> {
        let words = match &file.rgba16 {
            Some(words) => words.clone(),
            None => texture
                .rgba8
                .iter()
                .map(|byte| u16::from(*byte) * 257)
                .collect(),
        };
        if words.len() == pixel_count * 4 {
            Ok(words)
        } else {
            Err(mismatch())
        }
    };
    let used: &[usize] = match file.layout {
        OutputLayout::Gray => &[0],
        OutputLayout::Rgb => &[0, 1, 2],
        OutputLayout::Rgba => &[0, 1, 2, 3],
    };
    let image = match format {
        TextureFormat::Exr => {
            let words = words()?;
            let names: &[&'static str] = match file.layout {
                OutputLayout::Gray => &["Y"],
                OutputLayout::Rgb => &["R", "G", "B"],
                OutputLayout::Rgba => &["R", "G", "B", "A"],
            };
            let channels = names
                .iter()
                .zip(used)
                .map(|(name, component)| {
                    let srgb = file.srgb[*component];
                    let samples = words
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|texel| {
                            let value = f32::from(texel[*component]) / 65_535.0;
                            if srgb { srgb_to_linear(value) } else { value }
                        })
                        .collect();
                    (*name, samples)
                })
                .collect();
            return encode_exr(width, height, channels);
        }
        TextureFormat::Tga if file.bit_depth == BitDepth::Sixteen => {
            return Err(TextureIoError::Encode(format!(
                "`{}` is 16-bit, which TGA cannot store",
                file.file_stem
            )));
        }
        TextureFormat::Png if file.bit_depth == BitDepth::Sixteen => {
            let words = words()?;
            let samples: Vec<u16> = words
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|texel| used.iter().map(|component| texel[*component]))
                .collect();
            match file.layout {
                OutputLayout::Gray => DynamicImage::ImageLuma16(
                    ImageBuffer::from_raw(width, height, samples).ok_or_else(mismatch)?,
                ),
                OutputLayout::Rgb => DynamicImage::ImageRgb16(
                    ImageBuffer::from_raw(width, height, samples).ok_or_else(mismatch)?,
                ),
                OutputLayout::Rgba => DynamicImage::ImageRgba16(
                    ImageBuffer::from_raw(width, height, samples).ok_or_else(mismatch)?,
                ),
            }
        }
        TextureFormat::Png | TextureFormat::Tga => {
            let rgba = texture.rgba8.as_chunks::<4>().0;
            match file.layout {
                OutputLayout::Gray => DynamicImage::ImageLuma8(
                    GrayImage::from_raw(width, height, rgba.iter().map(|texel| texel[0]).collect())
                        .ok_or_else(mismatch)?,
                ),
                OutputLayout::Rgb => DynamicImage::ImageRgb8(
                    RgbImage::from_raw(
                        width,
                        height,
                        rgba.iter()
                            .flat_map(|texel| [texel[0], texel[1], texel[2]])
                            .collect(),
                    )
                    .ok_or_else(mismatch)?,
                ),
                OutputLayout::Rgba => DynamicImage::ImageRgba8(
                    RgbaImage::from_raw(width, height, texture.rgba8.clone())
                        .ok_or_else(mismatch)?,
                ),
            }
        }
    };
    let Some(image_format) = format.image_format() else {
        return Err(TextureIoError::UnsupportedFormat(
            format.extension().to_owned(),
        ));
    };
    let mut bytes = Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, image_format)
        .map_err(|error| TextureIoError::Encode(error.to_string()))?;
    Ok(bytes.into_inner())
}

/// Encodes named channels (row-major, rows top to bottom) as a half-float
/// OpenEXR with lossless ZIP compression.
fn encode_exr(
    width: u32,
    height: u32,
    channels: Vec<(&'static str, Vec<f32>)>,
) -> Result<Vec<u8>, TextureIoError> {
    use exr::prelude::{
        AnyChannel, AnyChannels, Encoding, FlatSamples, Image, Layer, LayerAttributes, SmallVec,
        WritableImage, f16,
    };
    let list: SmallVec<[AnyChannel<FlatSamples>; 4]> = channels
        .into_iter()
        .map(|(name, samples)| {
            AnyChannel::new(
                name,
                FlatSamples::F16(samples.into_iter().map(f16::from_f32).collect()),
            )
        })
        .collect();
    let layer = Layer::new(
        (width as usize, height as usize),
        LayerAttributes::default(),
        Encoding::SMALL_LOSSLESS,
        AnyChannels::sort(list),
    );
    let mut bytes = Cursor::new(Vec::new());
    Image::from_layer(layer)
        .write()
        .to_buffered(&mut bytes)
        .map_err(|error| TextureIoError::Encode(error.to_string()))?;
    Ok(bytes.into_inner())
}

/// Writes each packed output into `folder` as `<file_stem>.<extension>`, in
/// its own format or else `format`; returns the paths written.
pub fn write_exported_textures(
    folder: impl AsRef<Path>,
    files: &[ExportedTexture],
    format: TextureFormat,
) -> Result<Vec<std::path::PathBuf>, TextureIoError> {
    let mut written = Vec::with_capacity(files.len());
    for file in files {
        let extension = exported_texture_format(file, format)?.extension();
        let path = folder
            .as_ref()
            .join(format!("{}.{extension}", file.file_stem));
        let bytes = encode_exported_texture(file, format)?;
        std::fs::write(&path, bytes).map_err(|error| TextureIoError::Io(error.to_string()))?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn texture() -> BakedTexture {
        BakedTexture {
            width: 2,
            height: 1,
            rgba8: vec![10, 20, 30, 40, 200, 150, 100, 255],
        }
    }

    fn exported(stem: &str, layout: OutputLayout) -> ExportedTexture {
        ExportedTexture {
            file_stem: stem.to_owned(),
            layout,
            texture: texture(),
            format: None,
            bit_depth: BitDepth::Eight,
            rgba16: None,
            srgb: [false; 4],
        }
    }

    #[test]
    fn tc_0261_sixteen_bit_png_and_half_float_exr_outputs_keep_precision() {
        let folder = tempfile::tempdir().unwrap();
        // Values between 8-bit steps, which only the 16-bit samples hold.
        let precise = vec![1000, 30_000, 65_535, 65_535, 50_001, 2, 7, 65_535];
        let deep = ExportedTexture {
            bit_depth: BitDepth::Sixteen,
            rgba16: Some(precise.clone()),
            ..exported("Body_ORM", OutputLayout::Rgb)
        };
        let written = write_exported_textures(
            folder.path(),
            std::slice::from_ref(&deep),
            TextureFormat::Png,
        )
        .unwrap();
        let decoded = image::open(&written[0]).unwrap();
        assert_eq!(decoded.color(), image::ColorType::Rgb16);
        assert_eq!(
            decoded.into_rgb16().into_raw(),
            vec![1000, 30_000, 65_535, 50_001, 2, 7]
        );
        // A TGA cannot hold 16 bits.
        assert!(matches!(
            encode_exported_texture(&deep, TextureFormat::Tga),
            Err(TextureIoError::Encode(_))
        ));

        // EXR: a gray output is one Y channel; sRGB colour is stored
        // linear, data channels as they are. The file's own format wins
        // over the writer's default.
        let gray = ExportedTexture {
            format: Some("exr".to_owned()),
            rgba16: Some(precise.clone()),
            ..exported("Body_Height", OutputLayout::Gray)
        };
        let color = ExportedTexture {
            format: Some("exr".to_owned()),
            rgba16: Some(vec![32_768; 8]),
            srgb: [true, true, true, false],
            ..exported("Body_BaseColor", OutputLayout::Rgba)
        };
        let written =
            write_exported_textures(folder.path(), &[gray, color], TextureFormat::Png).unwrap();
        assert!(
            written
                .iter()
                .all(|path| path.extension().unwrap() == "exr")
        );
        let read = |path: &std::path::Path| {
            let image = exr::prelude::read_first_flat_layer_from_file(path).unwrap();
            image
                .layer_data
                .channel_data
                .list
                .iter()
                .map(|channel| {
                    (
                        channel.name.to_string(),
                        channel.sample_data.values_as_f32().collect::<Vec<f32>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let gray = read(&written[0]);
        assert_eq!(gray.len(), 1);
        assert_eq!(gray[0].0, "Y");
        assert!((gray[0].1[0] - 1000.0 / 65_535.0).abs() < 1e-4);
        assert!((gray[0].1[1] - 50_001.0 / 65_535.0).abs() < 1e-3);
        let color = read(&written[1]);
        let names: Vec<&str> = color.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["A", "B", "G", "R"]);
        let linear = substancecraft_core::color::srgb_to_linear(32_768.0 / 65_535.0);
        assert!((color[3].1[0] - linear).abs() < 1e-3, "R is linear");
        assert!((color[0].1[0] - 0.5).abs() < 1e-3, "alpha stays as stored");
    }

    #[test]
    fn tc_0090_png_and_tga_exports_round_trip_through_import() {
        for format in [TextureFormat::Png, TextureFormat::Tga] {
            let bytes = encode_texture(&texture(), format).unwrap();
            let decoded = decode_texture_image(&bytes).unwrap();
            assert_eq!((decoded.width, decoded.height), (2, 1), "{format:?}");
            assert_eq!(decoded.rgba8, texture().rgba8, "{format:?}");
        }
        let png = encode_texture(&texture(), TextureFormat::Png).unwrap();
        assert_eq!(&png[1..4], b"PNG");
    }

    #[test]
    fn tc_0091_texture_files_pick_their_format_by_extension() {
        let folder = tempfile::tempdir().unwrap();
        let png = folder.path().join("Body_BaseColor.PNG");
        write_texture(&png, &texture()).unwrap();
        assert_eq!(load_texture_image(&png).unwrap().rgba8, texture().rgba8);
        assert_eq!(
            write_texture(folder.path().join("map.exr"), &texture()),
            Ok(())
        );
        assert_eq!(
            write_texture(folder.path().join("map.bmp"), &texture()),
            Err(TextureIoError::UnsupportedFormat("bmp".to_owned()))
        );
        assert!(matches!(
            decode_texture_image(b"not an image"),
            Err(TextureIoError::Decode(_))
        ));
    }

    #[test]
    fn tc_0113_packed_outputs_write_gray_rgb_and_rgba_files_into_a_folder() {
        let folder = tempfile::tempdir().unwrap();
        let files: Vec<ExportedTexture> = [
            ("Body_Roughness", OutputLayout::Gray),
            ("Body_ORM", OutputLayout::Rgb),
            ("Body_BaseColor", OutputLayout::Rgba),
        ]
        .into_iter()
        .map(|(stem, layout)| exported(stem, layout))
        .collect();
        for format in [TextureFormat::Png, TextureFormat::Tga] {
            let written = write_exported_textures(folder.path(), &files, format).unwrap();
            assert_eq!(written.len(), 3);
            for (path, expected) in written.iter().zip([
                image::ColorType::L8,
                image::ColorType::Rgb8,
                image::ColorType::Rgba8,
            ]) {
                assert_eq!(path.extension().unwrap().to_str(), Some(format.extension()));
                let decoded = image::open(path).unwrap();
                assert_eq!(decoded.color(), expected, "{path:?}");
            }
            // Gray keeps R; RGB drops alpha; RGBA keeps everything.
            let gray = load_texture_image(&written[0]).unwrap();
            assert_eq!(gray.rgba8[..4], [10, 10, 10, 255]);
            let rgb = load_texture_image(&written[1]).unwrap();
            assert_eq!(rgb.rgba8[..4], [10, 20, 30, 255]);
            let rgba = load_texture_image(&written[2]).unwrap();
            assert_eq!(rgba.rgba8, texture().rgba8);
        }
    }

    #[test]
    fn tc_0399_bmp_tiff_webp_exr_and_hdr_import_with_deep_formats_kept_at_16_bits() {
        let folder = tempfile::tempdir().unwrap();
        let path = |name: &str| folder.path().join(name);
        let pixels = vec![10, 20, 30, 255, 200, 150, 100, 128];
        let rgba = RgbaImage::from_raw(2, 1, pixels.clone()).unwrap();

        // 8-bit formats decode exactly, with no 16-bit copy, tagged sRGB.
        for name in ["plain.bmp", "plain.tif", "plain.tiff", "plain.webp"] {
            rgba.save(path(name)).unwrap();
            let image = load_texture_image(path(name)).unwrap();
            assert_eq!(image.rgba8, pixels, "{name}");
            assert!(image.rgba16.is_none(), "{name}");
            assert_eq!(image.color_space, ColorSpace::Srgb, "{name}");
        }

        // 16-bit PNG and TIFF keep every word; the 8-bit copy rounds them.
        let words = vec![1000_u16, 30_000, 65_535, 65_535, 50_001, 2, 7, 32_768];
        let deep = ImageBuffer::<image::Rgba<u16>, _>::from_raw(2, 1, words.clone()).unwrap();
        for name in ["deep.png", "deep.tif"] {
            deep.save(path(name)).unwrap();
            let image = load_texture_image(path(name)).unwrap();
            assert_eq!(image.rgba16.as_ref(), Some(&words), "{name}");
            assert_eq!(image.rgba8, [4, 117, 255, 255, 195, 0, 0, 128], "{name}");
            assert_eq!(image.color_space, ColorSpace::Srgb, "{name}");
        }

        // OpenEXR: linear floats clamped to 0..=1 (non-finite to 0), alpha
        // straight, tagged linear.
        let floats = vec![0.25_f32, 2.0, -1.0, 0.5, f32::NAN, 0.75, 1.0, 1.0];
        ImageBuffer::<image::Rgba<f32>, _>::from_raw(2, 1, floats)
            .unwrap()
            .save(path("linear.exr"))
            .unwrap();
        let exr = load_texture_image(path("linear.exr")).unwrap();
        assert_eq!(exr.color_space, ColorSpace::LinearSrgb);
        assert_eq!(
            exr.rgba16.as_deref(),
            Some(&[16_384, 65_535, 0, 32_768, 0, 49_151, 65_535, 65_535][..])
        );
        assert_eq!(exr.rgba8, [64, 255, 0, 128, 0, 191, 255, 255]);

        // Radiance HDR (RGBE, no alpha) is opaque and linear.
        ImageBuffer::<image::Rgb<f32>, _>::from_raw(2, 1, vec![0.5, 0.25, 0.125, 4.0, 1.0, 0.0])
            .unwrap()
            .save(path("sky.hdr"))
            .unwrap();
        let hdr = load_texture_image(path("sky.hdr")).unwrap();
        assert_eq!(hdr.color_space, ColorSpace::LinearSrgb);
        let hdr16 = hdr.rgba16.unwrap();
        assert!((f32::from(hdr16[0]) / 65_535.0 - 0.5).abs() < 0.01);
        assert!((f32::from(hdr16[2]) / 65_535.0 - 0.125).abs() < 0.01);
        assert_eq!(hdr16[3], 65_535);
        assert_eq!(hdr16[4], 65_535, "values above 1 clamp");

        // A SubstanceCraft EXR export imports back as written.
        let written = write_exported_textures(
            folder.path(),
            &[ExportedTexture {
                format: Some("exr".to_owned()),
                rgba16: Some(words.clone()),
                ..exported("Body_Mask", OutputLayout::Rgba)
            }],
            TextureFormat::Png,
        )
        .unwrap();
        let round = load_texture_image(&written[0]).unwrap().rgba16.unwrap();
        for (read, wrote) in round.iter().zip(&words) {
            // Half floats hold about 11 significant bits.
            assert!(read.abs_diff(*wrote) <= 32, "{read} vs {wrote}");
        }

        // The tag a file starts with follows its extension.
        assert_eq!(source_color_space("a/b.EXR"), ColorSpace::LinearSrgb);
        assert_eq!(source_color_space("sky.hdr"), ColorSpace::LinearSrgb);
        assert_eq!(source_color_space("wall.tiff"), ColorSpace::Srgb);
    }
}
