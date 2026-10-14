//! HDR environment maps for the viewport: Radiance `.hdr` and OpenEXR
//! equirectangular panoramas, converted to the viewport's layout.

use std::io::{Cursor, Read};
use std::path::Path;

use image::{ImageFormat, ImageReader, Limits};
use substancecraft_core::environment::ENVIRONMENT_WIDTH;
use substancecraft_core::{EnvironmentImage, PanoramaError};
use thiserror::Error;

/// File extensions read as environment maps.
pub const ENVIRONMENT_EXTENSIONS: [&str; 2] = ["hdr", "exr"];
/// Largest environment file accepted.
pub const MAX_ENVIRONMENT_FILE_BYTES: usize = 512 * 1024 * 1024;
/// Longest panorama edge accepted, in pixels.
pub const MAX_ENVIRONMENT_EDGE: u32 = 16_384;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EnvironmentIoError {
    #[error("environment file I/O failed: {0}")]
    Io(String),
    #[error(
        "environment file exceeds the {} MiB limit",
        MAX_ENVIRONMENT_FILE_BYTES / (1024 * 1024)
    )]
    TooLarge,
    #[error("environment is {width}x{height}; edges must be 1..={MAX_ENVIRONMENT_EDGE} pixels")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("environment could not be decoded: {0}")]
    Decode(String),
    #[error("unsupported environment format `{0}`; use .hdr or .exr")]
    UnsupportedFormat(String),
    #[error(transparent)]
    Panorama(#[from] PanoramaError),
}

/// Reads a `.hdr` or `.exr` equirectangular panorama and converts it to the
/// viewport's environment layout ([`ENVIRONMENT_WIDTH`] wide).
pub fn load_environment_file(
    path: impl AsRef<Path>,
) -> Result<EnvironmentImage, EnvironmentIoError> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if !ENVIRONMENT_EXTENSIONS.contains(&extension.as_str()) {
        return Err(EnvironmentIoError::UnsupportedFormat(extension));
    }
    let file =
        std::fs::File::open(path).map_err(|error| EnvironmentIoError::Io(error.to_string()))?;
    let mut bytes = Vec::new();
    file.take(MAX_ENVIRONMENT_FILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| EnvironmentIoError::Io(error.to_string()))?;
    decode_environment(&bytes, &extension)
}

/// Decodes panorama bytes of the given format (`hdr` or `exr`).
pub fn decode_environment(
    bytes: &[u8],
    extension: &str,
) -> Result<EnvironmentImage, EnvironmentIoError> {
    if bytes.len() > MAX_ENVIRONMENT_FILE_BYTES {
        return Err(EnvironmentIoError::TooLarge);
    }
    let (width, height, rgb) = match extension {
        "hdr" => decode_radiance(bytes)?,
        "exr" => decode_exr(bytes)?,
        other => return Err(EnvironmentIoError::UnsupportedFormat(other.to_owned())),
    };
    Ok(EnvironmentImage::from_panorama(
        width,
        height,
        &rgb,
        ENVIRONMENT_WIDTH,
    )?)
}

fn check_dimensions(width: u32, height: u32) -> Result<(), EnvironmentIoError> {
    if (1..=MAX_ENVIRONMENT_EDGE).contains(&width) && (1..=MAX_ENVIRONMENT_EDGE).contains(&height) {
        Ok(())
    } else {
        Err(EnvironmentIoError::InvalidDimensions { width, height })
    }
}

fn decode_radiance(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>), EnvironmentIoError> {
    let mut reader = ImageReader::new(Cursor::new(bytes));
    reader.set_format(ImageFormat::Hdr);
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_ENVIRONMENT_EDGE);
    limits.max_image_height = Some(MAX_ENVIRONMENT_EDGE);
    limits.max_alloc = Some(4 * MAX_ENVIRONMENT_FILE_BYTES as u64);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|error| EnvironmentIoError::Decode(error.to_string()))?
        .into_rgb32f();
    let (width, height) = image.dimensions();
    check_dimensions(width, height)?;
    Ok((width, height, image.into_raw()))
}

fn decode_exr(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>), EnvironmentIoError> {
    let decode = |error: exr::error::Error| EnvironmentIoError::Decode(error.to_string());
    // Check the size before allocating pixels for it.
    let meta =
        exr::meta::MetaData::read_from_buffered(Cursor::new(bytes), false).map_err(decode)?;
    let header = meta
        .headers
        .first()
        .ok_or_else(|| EnvironmentIoError::Decode("the file has no layers".to_owned()))?;
    let size = header.layer_size;
    let dimension = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
    check_dimensions(dimension(size.width()), dimension(size.height()))?;

    use exr::prelude::{ReadChannels, ReadLayers};
    let image = exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .rgba_channels(
            |resolution, _| (resolution.width(), vec![0.0f32; resolution.area() * 3]),
            |(width, rgb): &mut (usize, Vec<f32>),
             position: exr::math::Vec2<usize>,
             (r, g, b, _a): (f32, f32, f32, f32)| {
                let index = (position.y() * *width + position.x()) * 3;
                rgb[index..index + 3].copy_from_slice(&[r, g, b]);
            },
        )
        .first_valid_layer()
        .all_attributes()
        .from_buffered(Cursor::new(bytes))
        .map_err(decode)?;
    let size = image.layer_data.size;
    let (width, height) = (dimension(size.width()), dimension(size.height()));
    check_dimensions(width, height)?;
    Ok((width, height, image.layer_data.channel_data.pixels.1))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// An uncompressed Radiance file (flat RGBE scanlines) of `rgb`.
    fn radiance_bytes(width: u32, height: u32, rgb: &[f32]) -> Vec<u8> {
        let mut bytes =
            format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n").into_bytes();
        for texel in rgb.as_chunks::<3>().0 {
            let brightest = texel.iter().copied().fold(0.0f32, f32::max);
            if brightest < 1e-32 {
                bytes.extend([0, 0, 0, 0]);
                continue;
            }
            // brightest = mantissa * 2^exponent with mantissa in [0.5, 1).
            let exponent = brightest.log2().floor() as i32 + 1;
            let scale = 256.0 / 2f32.powi(exponent);
            for value in texel {
                bytes.push((value * scale).clamp(0.0, 255.0) as u8);
            }
            bytes.push((exponent + 128) as u8);
        }
        bytes
    }

    fn exr_bytes(width: usize, height: usize, rgb: &[f32]) -> Vec<u8> {
        use exr::prelude::{
            AnyChannel, AnyChannels, Encoding, FlatSamples, Image, Layer, LayerAttributes,
            SmallVec, WritableImage,
        };
        let channel = |name: &'static str, offset: usize| {
            AnyChannel::new(
                name,
                FlatSamples::F32(rgb.iter().skip(offset).step_by(3).copied().collect()),
            )
        };
        let list: SmallVec<[AnyChannel<FlatSamples>; 4]> =
            SmallVec::from_vec(vec![channel("R", 0), channel("G", 1), channel("B", 2)]);
        let layer = Layer::new(
            (width, height),
            LayerAttributes::default(),
            Encoding::SMALL_LOSSLESS,
            AnyChannels::sort(list),
        );
        let mut bytes = Cursor::new(Vec::new());
        Image::from_layer(layer)
            .write()
            .to_buffered(&mut bytes)
            .unwrap();
        bytes.into_inner()
    }

    /// A 32x16 panorama: grey sky, black ground, and a bright pixel at
    /// u = 0.75 (looking along +X) just above the horizon.
    fn panorama() -> Vec<f32> {
        let mut rgb = Vec::new();
        for row in 0..16 {
            for column in 0..32 {
                let value = if (column, row) == (24, 7) {
                    64.0
                } else if row < 8 {
                    0.5
                } else {
                    0.0
                };
                rgb.extend([value, value * 0.5, value * 0.25]);
            }
        }
        rgb
    }

    #[test]
    fn tc_0346_hdr_and_exr_panoramas_load_as_viewport_environments() {
        let folder = tempfile::tempdir().unwrap();
        let rgb = panorama();
        let hdr = folder.path().join("sky.HDR");
        std::fs::write(&hdr, radiance_bytes(32, 16, &rgb)).unwrap();
        let exr = folder.path().join("sky.exr");
        std::fs::write(&exr, exr_bytes(32, 16, &rgb)).unwrap();

        let from_hdr = load_environment_file(&hdr).unwrap();
        let from_exr = load_environment_file(&exr).unwrap();
        for image in [&from_hdr, &from_exr] {
            assert_eq!(
                (image.width, image.height),
                (ENVIRONMENT_WIDTH, ENVIRONMENT_WIDTH / 2)
            );
            // The bright pixel turns half way round into the viewport
            // layout: the left quarter, which looks along +X.
            let row = 7 * image.height as usize / 16;
            let column = image.width as usize / 4 + 2;
            let index = (row * image.width as usize + column) * 3;
            assert!(image.rgb[index] > 60.0, "{}", image.rgb[index]);
            // Colour ratios survive both formats.
            assert!((image.rgb[index + 1] / image.rgb[index] - 0.5).abs() < 0.02);
        }
        // Both formats agree (RGBE keeps about 1% precision).
        let ratio = from_hdr.mean_luminance() / from_exr.mean_luminance();
        assert!((ratio - 1.0).abs() < 0.02, "{ratio}");

        // Other formats, shapes, broken bytes, and missing files are refused.
        assert_eq!(
            load_environment_file(folder.path().join("sky.png")),
            Err(EnvironmentIoError::UnsupportedFormat("png".to_owned()))
        );
        assert!(matches!(
            load_environment_file(folder.path().join("missing.hdr")),
            Err(EnvironmentIoError::Io(_))
        ));
        assert!(matches!(
            decode_environment(&radiance_bytes(16, 16, &[0.5; 16 * 16 * 3]), "hdr"),
            Err(EnvironmentIoError::Panorama(
                PanoramaError::NotEquirectangular { .. }
            ))
        ));
        assert!(matches!(
            decode_environment(b"#?RADIANCE\nnot really", "hdr"),
            Err(EnvironmentIoError::Decode(_))
        ));
        assert!(matches!(
            decode_environment(b"not an exr", "exr"),
            Err(EnvironmentIoError::Decode(_))
        ));
        assert!(matches!(
            decode_environment(&exr_bytes(4, 2, &[0.0; 24]), "exr"),
            Err(EnvironmentIoError::Panorama(PanoramaError::Black))
        ));
    }
}
