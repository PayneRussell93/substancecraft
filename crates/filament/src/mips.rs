//! CPU mip chains for viewport maps.
//!
//! Mips are built here rather than with `Texture::generateMipmaps()`, which
//! stalls the frame readback on the OpenGL backend. A chain kept after
//! upload can be updated over a region in place, so a painted dab
//! re-uploads a few small rectangles instead of the whole pyramid.

use std::sync::OnceLock;

use substancecraft_core::TexelRect;

/// One RGBA8 level of a chain.
pub(crate) struct MipLevel {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) rgba8: Vec<u8>,
}

impl MipLevel {
    /// The texels of `rect`, row by row.
    pub(crate) fn extract(&self, rect: TexelRect) -> Vec<u8> {
        let span = rect.width as usize * 4;
        let mut bytes = Vec::with_capacity(rect.len() * 4);
        for y in rect.y..rect.bottom() {
            let start = (y as usize * self.width as usize + rect.x as usize) * 4;
            bytes.extend_from_slice(&self.rgba8[start..start + span]);
        }
        bytes
    }
}

/// A full mip pyramid, level 0 first, down to 1 × 1.
pub(crate) struct MipChain {
    srgb: bool,
    pub(crate) levels: Vec<MipLevel>,
}

/// Number of levels down to 1 × 1.
pub(crate) fn level_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

impl MipChain {
    /// Builds every level from `rgba8`, a `width` × `height` level 0.
    /// sRGB maps are averaged in linear space.
    pub(crate) fn build(width: u32, height: u32, rgba8: &[u8], srgb: bool) -> Self {
        let mut levels = vec![MipLevel {
            width,
            height,
            rgba8: rgba8.to_vec(),
        }];
        for _ in 1..level_count(width, height) {
            let Some(source) = levels.last() else {
                break;
            };
            let mut next = MipLevel {
                width: (source.width / 2).max(1),
                height: (source.height / 2).max(1),
                rgba8: Vec::new(),
            };
            next.rgba8 = vec![0; next.width as usize * next.height as usize * 4];
            let full = TexelRect::full(next.width, next.height);
            downsample(source, &mut next, full, srgb);
            levels.push(next);
        }
        Self { srgb, levels }
    }

    pub(crate) fn matches(&self, width: u32, height: u32, srgb: bool) -> bool {
        self.srgb == srgb
            && self
                .levels
                .first()
                .is_some_and(|level| level.width == width && level.height == height)
    }

    /// Copies `region` of `rgba8` (a whole new level 0) into the chain and
    /// recomputes the mips it affects. Returns the rectangle changed in each
    /// level, level 0 first; the result equals [`Self::build`] on the new
    /// level 0 whenever the old and new level 0 differ only inside `region`.
    pub(crate) fn update(&mut self, rgba8: &[u8], region: TexelRect) -> Vec<TexelRect> {
        let Some(base) = self.levels.first_mut() else {
            return Vec::new();
        };
        let Some(mut rect) = region.intersect(TexelRect::full(base.width, base.height)) else {
            return Vec::new();
        };
        let span = rect.width as usize * 4;
        for y in rect.y..rect.bottom() {
            let start = (y as usize * base.width as usize + rect.x as usize) * 4;
            base.rgba8[start..start + span].copy_from_slice(&rgba8[start..start + span]);
        }
        let mut changed = vec![rect];
        for index in 1..self.levels.len() {
            let (lower, upper) = self.levels.split_at_mut(index);
            let (source, target) = (&lower[index - 1], &mut upper[0]);
            // Texel x reads source texels 2x and 2x + 1 (clamped), so the
            // source columns x0..x1 reach targets x0 / 2 ..= (x1 - 1) / 2.
            let x = rect.x / 2;
            let y = rect.y / 2;
            let right = ((rect.right() - 1) / 2 + 1).min(target.width);
            let bottom = ((rect.bottom() - 1) / 2 + 1).min(target.height);
            if right <= x || bottom <= y {
                break;
            }
            rect = TexelRect {
                x,
                y,
                width: right - x,
                height: bottom - y,
            };
            downsample(source, target, rect, self.srgb);
            changed.push(rect);
        }
        changed
    }
}

fn srgb_to_linear(value: u8) -> f32 {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|index| {
            let v = index as f32 / 255.0;
            if v <= 0.040_45 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        })
    })[usize::from(value)]
}

fn linear_to_srgb(value: f32) -> u8 {
    let v = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// 2 × 2 box filter of `source` into `rect` of `target`. sRGB colour is
/// averaged in linear space; linear data and alpha directly. Odd edges
/// clamp to the last texel.
fn downsample(source: &MipLevel, target: &mut MipLevel, rect: TexelRect, srgb: bool) {
    let (width, height) = (source.width, source.height);
    let texel = |x: u32, y: u32| {
        let index = (y.min(height - 1) as usize * width as usize + x.min(width - 1) as usize) * 4;
        &source.rgba8[index..index + 4]
    };
    for y in rect.y..rect.bottom() {
        for x in rect.x..rect.right() {
            let texels = [
                texel(x * 2, y * 2),
                texel(x * 2 + 1, y * 2),
                texel(x * 2, y * 2 + 1),
                texel(x * 2 + 1, y * 2 + 1),
            ];
            let out = (y as usize * target.width as usize + x as usize) * 4;
            for channel in 0..4 {
                target.rgba8[out + channel] = if srgb && channel < 3 {
                    let sum: f32 = texels.iter().map(|t| srgb_to_linear(t[channel])).sum();
                    linear_to_srgb(sum * 0.25)
                } else {
                    let sum: u32 = texels.iter().map(|t| u32::from(t[channel])).sum();
                    ((sum + 2) / 4) as u8
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, seed: u32) -> Vec<u8> {
        (0..width * height * 4)
            .map(|index| (index.wrapping_mul(2_654_435_761) ^ seed) as u8)
            .collect()
    }

    #[test]
    fn level_counts_reach_one_texel() {
        assert_eq!(level_count(1, 1), 1);
        assert_eq!(level_count(2048, 2048), 12);
        assert_eq!(level_count(150, 7), 8);
        let chain = MipChain::build(150, 7, &image(150, 7, 1), false);
        let last = chain.levels.last().map(|level| (level.width, level.height));
        assert_eq!(last, Some((1, 1)));
    }

    #[test]
    fn region_updates_match_full_rebuilds_on_odd_sizes() {
        for srgb in [false, true] {
            let (width, height) = (151, 97);
            let mut current = image(width, height, 3);
            let mut chain = MipChain::build(width, height, &current, srgb);
            for (round, region) in [
                TexelRect {
                    x: 0,
                    y: 0,
                    width: 5,
                    height: 3,
                },
                TexelRect {
                    x: 140,
                    y: 90,
                    width: 11,
                    height: 7,
                },
                TexelRect {
                    x: 33,
                    y: 41,
                    width: 64,
                    height: 1,
                },
            ]
            .into_iter()
            .enumerate()
            {
                let fresh = image(width, height, 100 + round as u32);
                for y in region.y..region.bottom() {
                    for x in region.x..region.right() {
                        let index = ((y * width + x) * 4) as usize;
                        current[index..index + 4].copy_from_slice(&fresh[index..index + 4]);
                    }
                }
                let changed = chain.update(&current, region);
                assert_eq!(changed.first(), Some(&region));
                let rebuilt = MipChain::build(width, height, &current, srgb);
                for (level, (patched, full)) in chain.levels.iter().zip(&rebuilt.levels).enumerate()
                {
                    assert!(
                        patched.rgba8 == full.rgba8,
                        "srgb {srgb}, round {round}, level {level} differs"
                    );
                }
            }
        }
    }

    #[test]
    fn extract_reads_rows_of_a_rect() {
        let chain = MipChain::build(4, 2, &(0..32).collect::<Vec<u8>>(), false);
        let rect = TexelRect {
            x: 1,
            y: 1,
            width: 2,
            height: 1,
        };
        assert_eq!(chain.levels[0].extract(rect), (20..28).collect::<Vec<u8>>());
    }
}
