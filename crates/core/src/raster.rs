//! UV-space triangle rasterisation shared by mesh-map baking and polygon masks.
//!
//! Pixel `(x, y)` corresponds to UV `(x / (w - 1), 1 - y / (h - 1))`, matching
//! the convention used by stroke rasterisation and `BakedTexture::sample_uv`.

use glam::{Vec2, Vec3};

use crate::region::TexelRect;

pub(crate) fn uv_to_pixel(uv: Vec2, width: u32, height: u32) -> Vec2 {
    Vec2::new(
        uv.x * width.saturating_sub(1) as f32,
        (1.0 - uv.y) * height.saturating_sub(1) as f32,
    )
}

pub(crate) fn pixel_to_uv(x: u32, y: u32, width: u32, height: u32) -> Vec2 {
    Vec2::new(
        x as f32 / width.saturating_sub(1).max(1) as f32,
        1.0 - y as f32 / height.saturating_sub(1).max(1) as f32,
    )
}

/// Visits every pixel covered by a UV triangle. Pixels up to `fringe` pixels
/// outside the triangle are also visited with `interior == false` so shared
/// edges never leave cracks; their barycentric weights are clamped onto the
/// triangle.
pub(crate) fn rasterize_uv_triangle(
    uvs: [Vec2; 3],
    width: u32,
    height: u32,
    fringe: f32,
    apply: impl FnMut(usize, Vec3, bool),
) {
    rasterize_uv_triangle_in(uvs, width, height, fringe, TexelRect::full(width, height), apply);
}

/// [`rasterize_uv_triangle`] limited to the texels of `clip`; the index is
/// row-major within `clip`.
pub(crate) fn rasterize_uv_triangle_in(
    uvs: [Vec2; 3],
    width: u32,
    height: u32,
    fringe: f32,
    clip: TexelRect,
    mut apply: impl FnMut(usize, Vec3, bool),
) {
    if width == 0 || height == 0 || uvs.iter().any(|uv| !uv.is_finite()) {
        return;
    }
    let points = uvs.map(|uv| uv_to_pixel(uv, width, height));
    let area = (points[1] - points[0]).perp_dot(points[2] - points[0]);
    if !area.is_finite() || area.abs() <= 1e-9 {
        return;
    }
    let edge_lengths = [
        points[1].distance(points[2]),
        points[2].distance(points[0]),
        points[0].distance(points[1]),
    ];
    let fringe = fringe.max(0.0);
    let minimum = points[0].min(points[1]).min(points[2]) - Vec2::splat(fringe);
    let maximum = points[0].max(points[1]).max(points[2]) + Vec2::splat(fringe);
    let min_x = (minimum.x.floor() as i64).max(0);
    let min_y = (minimum.y.floor() as i64).max(0);
    let max_x = (maximum.x.ceil() as i64).min(i64::from(width) - 1);
    let max_y = (maximum.y.ceil() as i64).min(i64::from(height) - 1);
    let (min_x, min_y) = (min_x.max(i64::from(clip.x)), min_y.max(i64::from(clip.y)));
    let max_x = max_x.min(i64::from(clip.right()) - 1);
    let max_y = max_y.min(i64::from(clip.bottom()) - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = Vec2::new(x as f32, y as f32);
            let w0 = (points[2] - points[1]).perp_dot(point - points[1]) / area;
            let w1 = (points[0] - points[2]).perp_dot(point - points[2]) / area;
            let w2 = 1.0 - w0 - w1;
            let weights = [w0, w1, w2];
            // Pixels exactly on an edge count as interior despite rounding.
            let interior = weights.iter().all(|weight| *weight >= -1e-5);
            if !interior {
                if fringe <= 0.0 {
                    continue;
                }
                let outside = weights
                    .iter()
                    .zip(edge_lengths)
                    .any(|(weight, length)| length > 0.0 && weight * area.abs() / length < -fringe);
                if outside {
                    continue;
                }
            }
            let clamped = Vec3::new(w0.max(0.0), w1.max(0.0), w2.max(0.0));
            let sum = clamped.x + clamped.y + clamped.z;
            if sum <= f32::EPSILON {
                continue;
            }
            apply(clip.local_index(x as u32, y as u32), clamped / sum, interior);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_square_triangle_covers_its_half_without_fringe() {
        let mut covered = [false; 16];
        rasterize_uv_triangle(
            [
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, 1.0),
            ],
            4,
            4,
            0.0,
            |index, weights, interior| {
                assert!(interior);
                assert!((weights.x + weights.y + weights.z - 1.0).abs() < 1e-5);
                covered[index] = true;
            },
        );
        // UV (0,0) is the bottom-left pixel; UV (1,1) the top-right.
        assert!(covered[12]);
        assert!(!covered[3]);
        assert_eq!(covered.iter().filter(|value| **value).count(), 10);
    }
}
