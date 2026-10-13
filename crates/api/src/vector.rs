//! Vector resources: SVG files rasterised on load, and text resources
//! (`.sctext` JSON files) rendered with a font. Both become ordinary
//! straight-alpha bitmaps, so they work as fill textures, masks, brush
//! alphas, and decals, and reload when their source file changes.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};
use resvg::{tiny_skia, usvg};
use serde::{Deserialize, Serialize};
use substancecraft_core::{MAX_RESOURCE_DIMENSION, TextureImage};

use crate::textures::TextureIoError;

/// Longest edge an SVG is rasterised to unless it asks for more.
pub const DEFAULT_SVG_EDGE: u32 = 1_024;
/// Extension of text resource files.
pub const TEXT_RESOURCE_EXTENSION: &str = "sctext";
const MAX_TEXT_CHARS: usize = 1_024;

/// The fonts SVG `<text>` is set in: the bundled Ubuntu Light (Ubuntu Font
/// Licence) for every family, and Hack for `monospace`. System fonts are
/// never read, so an SVG rasterises the same on every machine.
fn svg_fonts() -> &'static (Arc<usvg::fontdb::Database>, String) {
    static FONTS: OnceLock<(Arc<usvg::fontdb::Database>, String)> = OnceLock::new();
    FONTS.get_or_init(|| {
        let mut fonts = usvg::fontdb::Database::new();
        fonts.load_font_data(epaint_default_fonts::UBUNTU_LIGHT.to_vec());
        let family = fonts
            .faces()
            .next()
            .and_then(|face| face.families.first())
            .map_or_else(|| "Ubuntu".to_owned(), |(name, _)| name.clone());
        fonts.load_font_data(epaint_default_fonts::HACK_REGULAR.to_vec());
        let monospace = fonts
            .faces()
            .nth(1)
            .and_then(|face| face.families.first())
            .map_or_else(|| "Hack".to_owned(), |(name, _)| name.clone());
        fonts.set_serif_family(family.clone());
        fonts.set_sans_serif_family(family.clone());
        fonts.set_cursive_family(family.clone());
        fonts.set_fantasy_family(family.clone());
        fonts.set_monospace_family(monospace);
        (Arc::new(fonts), family)
    })
}

/// Rasterises SVG data so its longest edge is `edge` pixels. Shapes keep
/// their fill colours and alpha; `<text>` is set in the bundled fonts (see
/// [`svg_fonts`]), whatever family it names.
pub fn rasterize_svg(data: &[u8], edge: u32) -> Result<TextureImage, TextureIoError> {
    let (fonts, family) = svg_fonts();
    let options = usvg::Options {
        font_family: family.clone(),
        fontdb: Arc::clone(fonts),
        ..usvg::Options::default()
    };
    let tree = usvg::Tree::from_data(data, &options)
        .map_err(|error| TextureIoError::Decode(format!("SVG: {error}")))?;
    let size = tree.size();
    let longest = size.width().max(size.height());
    if longest <= 0.0 || !longest.is_finite() {
        return Err(TextureIoError::Decode("SVG has no size".to_owned()));
    }
    let edge = edge.clamp(1, MAX_RESOURCE_DIMENSION);
    let scale = edge as f32 / longest;
    let width = ((size.width() * scale).round() as u32).clamp(1, MAX_RESOURCE_DIMENSION);
    let height = ((size.height() * scale).round() as u32).clamp(1, MAX_RESOURCE_DIMENSION);
    let mut pixmap = tiny_skia::Pixmap::new(width, height)
        .ok_or(TextureIoError::InvalidDimensions { width, height })?;
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    // tiny-skia stores premultiplied alpha; resources are straight alpha.
    let rgba8 = pixmap
        .pixels()
        .iter()
        .flat_map(|pixel| {
            let color = pixel.demultiply();
            [color.red(), color.green(), color.blue(), color.alpha()]
        })
        .collect();
    TextureImage::new(width, height, rgba8)
        .ok_or(TextureIoError::InvalidDimensions { width, height })
}

/// A text resource: what to write and how big. Stored as JSON in a
/// `.sctext` file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextSpec {
    /// One or more lines.
    pub text: String,
    /// Glyph height in pixels.
    pub size_px: f32,
    /// TrueType/OpenType font file; the bundled Ubuntu Light (Ubuntu Font
    /// Licence) when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font: Option<String>,
}

/// Renders `spec` as white glyphs whose alpha is their coverage, on a
/// transparent background with a small margin.
pub fn render_text(
    spec: &TextSpec,
    font_data: Option<&[u8]>,
) -> Result<TextureImage, TextureIoError> {
    if spec.text.trim().is_empty() || spec.text.chars().count() > MAX_TEXT_CHARS {
        return Err(TextureIoError::Decode(format!(
            "text must be 1 to {MAX_TEXT_CHARS} characters"
        )));
    }
    if !spec.size_px.is_finite() || !(4.0..=1_024.0).contains(&spec.size_px) {
        return Err(TextureIoError::Decode(
            "text size must be 4 to 1024 px".to_owned(),
        ));
    }
    let font = FontRef::try_from_slice(font_data.unwrap_or(epaint_default_fonts::UBUNTU_LIGHT))
        .map_err(|error| TextureIoError::Decode(format!("font: {error}")))?;
    let scaled = font.as_scaled(PxScale::from(spec.size_px));
    let line_height = scaled.height() + scaled.line_gap();
    let margin = (spec.size_px * 0.1).ceil().max(2.0);
    let lines: Vec<&str> = spec.text.lines().collect();
    let line_width = |line: &str| {
        let mut width = 0.0_f32;
        let mut previous = None;
        for character in line.chars() {
            let glyph = font.glyph_id(character);
            if let Some(previous) = previous {
                width += scaled.kern(previous, glyph);
            }
            width += scaled.h_advance(glyph);
            previous = Some(glyph);
        }
        width
    };
    let content_width = lines
        .iter()
        .map(|line| line_width(line))
        .fold(0.0, f32::max);
    let width = ((content_width + margin * 2.0).ceil() as u32).clamp(1, MAX_RESOURCE_DIMENSION);
    let height = ((line_height * lines.len() as f32 + margin * 2.0).ceil() as u32)
        .clamp(1, MAX_RESOURCE_DIMENSION);
    let mut coverage = vec![0.0_f32; width as usize * height as usize];
    for (row, line) in lines.iter().enumerate() {
        let baseline = margin + scaled.ascent() + line_height * row as f32;
        let mut x = margin;
        let mut previous = None;
        for character in line.chars() {
            let id = font.glyph_id(character);
            if let Some(previous) = previous {
                x += scaled.kern(previous, id);
            }
            let glyph = id.with_scale_and_position(spec.size_px, point(x, baseline));
            x += scaled.h_advance(id);
            previous = Some(id);
            let Some(outline) = font.outline_glyph(glyph) else {
                continue;
            };
            let bounds = outline.px_bounds();
            outline.draw(|gx, gy, value| {
                let px = bounds.min.x as i64 + i64::from(gx);
                let py = bounds.min.y as i64 + i64::from(gy);
                if (0..i64::from(width)).contains(&px) && (0..i64::from(height)).contains(&py) {
                    let index = py as usize * width as usize + px as usize;
                    coverage[index] = coverage[index].max(value.clamp(0.0, 1.0));
                }
            });
        }
    }
    let rgba8 = coverage
        .iter()
        .flat_map(|alpha| [255, 255, 255, (alpha * 255.0).round() as u8])
        .collect();
    TextureImage::new(width, height, rgba8)
        .ok_or(TextureIoError::InvalidDimensions { width, height })
}

/// Reads a `.sctext` file and renders it, loading its font if it names one
/// (relative paths resolve against the text file's folder).
pub fn load_text_resource(path: &Path) -> Result<TextureImage, TextureIoError> {
    let json =
        std::fs::read_to_string(path).map_err(|error| TextureIoError::Io(error.to_string()))?;
    let spec: TextSpec = serde_json::from_str(&json)
        .map_err(|error| TextureIoError::Decode(format!("text resource: {error}")))?;
    let font =
        match &spec.font {
            Some(font) => {
                let font_path = path
                    .parent()
                    .map_or_else(|| Path::new(font).to_path_buf(), |folder| folder.join(font));
                Some(std::fs::read(&font_path).map_err(|error| {
                    TextureIoError::Io(format!("{}: {error}", font_path.display()))
                })?)
            }
            None => None,
        };
    render_text(&spec, font.as_deref())
}

/// Reads a `.sctext` file's settings without rendering it.
pub fn read_text_resource(path: &Path) -> Result<TextSpec, TextureIoError> {
    let json =
        std::fs::read_to_string(path).map_err(|error| TextureIoError::Io(error.to_string()))?;
    serde_json::from_str(&json)
        .map_err(|error| TextureIoError::Decode(format!("text resource: {error}")))
}

/// Writes a `.sctext` file.
pub fn save_text_resource(path: &Path, spec: &TextSpec) -> Result<(), TextureIoError> {
    let json = serde_json::to_string_pretty(spec)
        .map_err(|error| TextureIoError::Encode(error.to_string()))?;
    std::fs::write(path, json).map_err(|error| TextureIoError::Io(error.to_string()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::load_texture_image;

    const STAR: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">
        <rect x="0" y="0" width="10" height="10" fill="#ff0000"/>
        <circle cx="15" cy="5" r="4" fill="#0000ff" fill-opacity="0.5"/>
    </svg>"##;

    #[test]
    fn tc_0180_svg_rasterises_to_straight_alpha_at_the_requested_size() {
        let image = rasterize_svg(STAR.as_bytes(), 200).unwrap();
        assert_eq!((image.width, image.height), (200, 100));
        let at = |x: u32, y: u32| {
            let index = (y as usize * image.width as usize + x as usize) * 4;
            image.rgba8[index..index + 4].to_vec()
        };
        assert_eq!(at(50, 50), [255, 0, 0, 255], "opaque red square");
        let blue = at(150, 50);
        assert_eq!(
            &blue[..3],
            &[0, 0, 255],
            "straight (not premultiplied) blue"
        );
        assert!(
            (126..=129).contains(&blue[3]),
            "half alpha, got {}",
            blue[3]
        );
        assert_eq!(at(150, 2)[3], 0, "transparent outside shapes");
        assert!(rasterize_svg(b"<not svg", 64).is_err());

        // `.svg` files load through the normal texture loader.
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("badge.svg");
        std::fs::write(&path, STAR).unwrap();
        let loaded = load_texture_image(&path).unwrap();
        assert_eq!(loaded.width, DEFAULT_SVG_EDGE);
    }

    #[test]
    fn tc_0181_text_resources_render_white_glyph_coverage() {
        let spec = TextSpec {
            text: "AB\nC".to_owned(),
            size_px: 48.0,
            font: None,
        };
        let image = render_text(&spec, None).unwrap();
        assert!(image.width > 48 && image.height > 96, "two lines tall");
        let alphas: Vec<u8> = image.rgba8.chunks(4).map(|texel| texel[3]).collect();
        assert!(alphas.contains(&255), "solid glyph interiors");
        assert!(alphas[0] == 0, "transparent margin");
        assert!(
            image
                .rgba8
                .chunks(4)
                .all(|texel| texel[..3] == [255, 255, 255])
        );
        let wider = render_text(
            &TextSpec {
                text: "ABABAB".to_owned(),
                ..spec.clone()
            },
            None,
        )
        .unwrap();
        assert!(wider.width > image.width);
        assert!(
            render_text(
                &TextSpec {
                    text: " ".to_owned(),
                    ..spec.clone()
                },
                None
            )
            .is_err()
        );
        assert!(
            render_text(
                &TextSpec {
                    size_px: 0.0,
                    ..spec.clone()
                },
                None
            )
            .is_err()
        );

        // Text resource files round-trip and load like bitmaps.
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("label.sctext");
        save_text_resource(&path, &spec).unwrap();
        assert_eq!(load_texture_image(&path).unwrap(), image);
        let missing_font = TextSpec {
            font: Some("missing.ttf".to_owned()),
            ..spec
        };
        save_text_resource(&path, &missing_font).unwrap();
        assert!(matches!(
            load_texture_image(&path),
            Err(TextureIoError::Io(_))
        ));
    }

    /// Columns and rows of `image` holding any ink (alpha above 64), as
    /// `(min_x, max_x, min_y, max_y)`.
    fn ink_bounds(image: &TextureImage) -> Option<(u32, u32, u32, u32)> {
        let mut bounds: Option<(u32, u32, u32, u32)> = None;
        for y in 0..image.height {
            for x in 0..image.width {
                let alpha = image.rgba8[(y as usize * image.width as usize + x as usize) * 4 + 3];
                if alpha > 64 {
                    bounds = Some(match bounds {
                        None => (x, x, y, y),
                        Some((x0, x1, y0, y1)) => (x0.min(x), x1.max(x), y0.min(y), y1.max(y)),
                    });
                }
            }
        }
        bounds
    }

    fn text_svg(attributes: &str, content: &str) -> String {
        format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="50">
                <text x="10" y="35" font-size="30" fill="#ff0000" {attributes}>{content}</text>
            </svg>"##
        )
    }

    #[test]
    fn tc_0327_svg_text_is_set_in_the_bundled_fonts() {
        let sans = rasterize_svg(
            text_svg(r#"font-family="sans-serif""#, "Wear").as_bytes(),
            200,
        )
        .unwrap();
        let (x0, x1, y0, y1) = ink_bounds(&sans).unwrap();
        assert!(x0 >= 9 && x1 < 120, "starts at x = 10: {x0}..{x1}");
        assert!(
            y0 > 5 && y1 <= 36,
            "sits on the baseline at y = 35: {y0}..{y1}"
        );
        assert!(
            sans.rgba8
                .chunks(4)
                .filter(|texel| texel[3] == 255)
                .all(|texel| texel[..3] == [255, 0, 0]),
            "glyphs take the fill colour"
        );

        // Families the bundle lacks fall back to it, so output never
        // depends on the machine's fonts; rendering is repeatable.
        let arial =
            rasterize_svg(text_svg(r#"font-family="Arial""#, "Wear").as_bytes(), 200).unwrap();
        assert_eq!(arial, sans);
        let unnamed = rasterize_svg(text_svg("", "Wear").as_bytes(), 200).unwrap();
        assert_eq!(unnamed, sans);
        let mono = rasterize_svg(
            text_svg(r#"font-family="monospace""#, "Wear").as_bytes(),
            200,
        )
        .unwrap();
        assert_ne!(mono, sans, "monospace uses its own face");

        // Anchoring, spans, and longer runs lay out as SVG specifies.
        let end = rasterize_svg(
            text_svg(r#"text-anchor="end" x="190""#, "Wear")
                .replace(r#"x="10" "#, "")
                .as_bytes(),
            200,
        )
        .unwrap();
        let (end_x0, end_x1, _, _) = ink_bounds(&end).unwrap();
        assert!(end_x1 <= 190 && end_x1 > 180, "ends at x = 190: {end_x1}");
        assert!(
            ((end_x1 - end_x0) as i64 - (x1 - x0) as i64).abs() <= 1,
            "same width when anchored at its end"
        );
        let spans = rasterize_svg(
            text_svg("", r##"We<tspan fill="#0000ff">ar</tspan>"##).as_bytes(),
            200,
        )
        .unwrap();
        assert_eq!(
            ink_bounds(&spans),
            ink_bounds(&sans),
            "spans keep the layout"
        );
        assert!(
            spans.rgba8.chunks(4).any(|texel| texel == [0, 0, 255, 255]),
            "a span takes its own fill"
        );
        let longer = rasterize_svg(text_svg("", "Wear and tear").as_bytes(), 200).unwrap();
        assert!(ink_bounds(&longer).unwrap().1 > x1 + 40);
    }
}
