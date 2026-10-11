//! Bitmap import (PNG, JPEG, TGA) and baked-texture export (PNG, TGA).

use std::io::{Cursor, Read};
use std::path::Path;

use image::{DynamicImage, GrayImage, ImageFormat, ImageReader, Limits, RgbImage, RgbaImage};
use substancecraft_core::{
    BakedTexture, ExportedTexture, MAX_RESOURCE_DIMENSION, OutputLayout, TextureImage,
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
    #[error("unsupported texture format `{0}`; use .png or .tga")]
    UnsupportedFormat(String),
}

/// Decodes PNG, JPEG, or TGA bytes into straight-alpha RGBA8. TGA has no
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
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|error| TextureIoError::Decode(error.to_string()))?
        .into_rgba8();
    let (width, height) = image.dimensions();
    TextureImage::new(width, height, image.into_raw())
        .ok_or(TextureIoError::InvalidDimensions { width, height })
}

/// Reads and decodes a bitmap file.
pub fn load_texture_image(path: impl AsRef<Path>) -> Result<TextureImage, TextureIoError> {
    let path = path.as_ref();
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
}

impl TextureFormat {
    pub const ALL: [Self; 2] = [Self::Png, Self::Tga];

    pub const fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Tga => "tga",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Tga => "Targa",
        }
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

    fn image_format(self) -> ImageFormat {
        match self {
            Self::Png => ImageFormat::Png,
            Self::Tga => ImageFormat::Tga,
        }
    }
}

/// Encodes a baked texture (RGBA8, rows top to bottom).
pub fn encode_texture(
    texture: &BakedTexture,
    format: TextureFormat,
) -> Result<Vec<u8>, TextureIoError> {
    let image = RgbaImage::from_raw(texture.width, texture.height, texture.rgba8.clone())
        .ok_or_else(|| {
            TextureIoError::Encode("RGBA buffer does not match its dimensions".to_owned())
        })?;
    let mut bytes = Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, format.image_format())
        .map_err(|error| TextureIoError::Encode(error.to_string()))?;
    Ok(bytes.into_inner())
}

/// Writes a baked texture in the format named by the path's extension.
pub fn write_texture(path: impl AsRef<Path>, texture: &BakedTexture) -> Result<(), TextureIoError> {
    let path = path.as_ref();
    let bytes = encode_texture(texture, TextureFormat::from_path(path)?)?;
    std::fs::write(path, bytes).map_err(|error| TextureIoError::Io(error.to_string()))
}

/// Encodes a packed export output with its own channel count: one channel
/// for Gray, three for RGB, four for RGBA.
pub fn encode_exported_texture(
    file: &ExportedTexture,
    format: TextureFormat,
) -> Result<Vec<u8>, TextureIoError> {
    let texture = &file.texture;
    let rgba = texture.rgba8.as_chunks::<4>().0;
    let mismatch =
        || TextureIoError::Encode("RGBA buffer does not match its dimensions".to_owned());
    let image = match file.layout {
        OutputLayout::Gray => DynamicImage::ImageLuma8(
            GrayImage::from_raw(
                texture.width,
                texture.height,
                rgba.iter().map(|texel| texel[0]).collect(),
            )
            .ok_or_else(mismatch)?,
        ),
        OutputLayout::Rgb => DynamicImage::ImageRgb8(
            RgbImage::from_raw(
                texture.width,
                texture.height,
                rgba.iter()
                    .flat_map(|texel| [texel[0], texel[1], texel[2]])
                    .collect(),
            )
            .ok_or_else(mismatch)?,
        ),
        OutputLayout::Rgba => DynamicImage::ImageRgba8(
            RgbaImage::from_raw(texture.width, texture.height, texture.rgba8.clone())
                .ok_or_else(mismatch)?,
        ),
    };
    let mut bytes = Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, format.image_format())
        .map_err(|error| TextureIoError::Encode(error.to_string()))?;
    Ok(bytes.into_inner())
}

/// Writes each packed output into `folder` as `<file_stem>.<extension>`;
/// returns the paths written.
pub fn write_exported_textures(
    folder: impl AsRef<Path>,
    files: &[ExportedTexture],
    format: TextureFormat,
) -> Result<Vec<std::path::PathBuf>, TextureIoError> {
    let mut written = Vec::with_capacity(files.len());
    for file in files {
        let path = folder
            .as_ref()
            .join(format!("{}.{}", file.file_stem, format.extension()));
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

    #[test]
    fn tc_0090_png_and_tga_exports_round_trip_through_import() {
        for format in TextureFormat::ALL {
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
            Err(TextureIoError::UnsupportedFormat("exr".to_owned()))
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
        .map(|(stem, layout)| ExportedTexture {
            file_stem: stem.to_owned(),
            layout,
            texture: texture(),
        })
        .collect();
        for format in TextureFormat::ALL {
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
}
