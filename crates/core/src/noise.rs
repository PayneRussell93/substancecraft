//! Deterministic, seedable 3D gradient noise used by procedural generators.
//!
//! Noise is evaluated in object space (from the baked position map) rather than
//! UV space, so breakup patterns stay continuous across UV seams.

use glam::Vec3;

/// Integer hash with good avalanche behaviour (based on the public-domain
/// "lowbias32" finaliser), combined over lattice coordinates and a seed.
pub fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut value = seed ^ 0x9e37_79b9;
    for component in [x as u32, y as u32, z as u32] {
        value ^= component
            .wrapping_mul(0x85eb_ca6b)
            .wrapping_add(0xc2b2_ae35);
        value = lowbias32(value);
    }
    value
}

pub fn lowbias32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

/// Uniform float in `[0, 1)` from a hash value.
pub fn unit_float(hash: u32) -> f32 {
    (hash >> 8) as f32 / (1u32 << 24) as f32
}

const GRADIENTS: [Vec3; 12] = [
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(-1.0, 1.0, 0.0),
    Vec3::new(1.0, -1.0, 0.0),
    Vec3::new(-1.0, -1.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(-1.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, -1.0),
    Vec3::new(-1.0, 0.0, -1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(0.0, -1.0, 1.0),
    Vec3::new(0.0, 1.0, -1.0),
    Vec3::new(0.0, -1.0, -1.0),
];

fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Gradient noise in roughly `[-1, 1]`. Non-finite input yields `0`.
pub fn gradient_noise3(point: Vec3, seed: u32) -> f32 {
    if !point.is_finite() {
        return 0.0;
    }
    let cell = point.floor();
    let local = point - cell;
    let base = [cell.x as i32, cell.y as i32, cell.z as i32];
    let fade = Vec3::new(quintic(local.x), quintic(local.y), quintic(local.z));

    let corner = |dx: i32, dy: i32, dz: i32| {
        let hash = hash3(
            base[0].wrapping_add(dx),
            base[1].wrapping_add(dy),
            base[2].wrapping_add(dz),
            seed,
        );
        let gradient = GRADIENTS[(hash % GRADIENTS.len() as u32) as usize];
        gradient.dot(local - Vec3::new(dx as f32, dy as f32, dz as f32))
    };

    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), fade.x);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), fade.x);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), fade.x);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), fade.x);
    let y0 = lerp(x00, x10, fade.y);
    let y1 = lerp(x01, x11, fade.y);
    lerp(y0, y1, fade.z).clamp(-1.0, 1.0)
}

/// Fractal sum of gradient noise remapped to `[0, 1]`, centred near `0.5`.
pub fn fbm3(point: Vec3, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 0.5;
    let mut total = 0.0;
    let mut frequency = 1.0;
    for octave in 0..octaves.clamp(1, 8) {
        sum += gradient_noise3(point * frequency, seed.wrapping_add(octave * 0x6d2b_79f5))
            * amplitude;
        total += amplitude;
        amplitude *= 0.5;
        frequency *= 2.03;
    }
    (sum / total * 0.5 + 0.5).clamp(0.0, 1.0)
}

fn lerp(from: f32, to: f32, amount: f32) -> f32 {
    from + (to - from) * amount
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tc_0029_noise_is_deterministic_bounded_and_seed_dependent() {
        let point = Vec3::new(1.37, -4.2, 9.81);
        assert_eq!(fbm3(point, 4, 7), fbm3(point, 4, 7));
        assert_ne!(fbm3(point, 4, 7), fbm3(point, 4, 8));
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        for step in 0..2_000 {
            let t = step as f32 * 0.137;
            let value = fbm3(Vec3::new(t, t * 0.71, -t * 1.3), 5, 3);
            assert!((0.0..=1.0).contains(&value));
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        assert!(maximum - minimum > 0.3, "noise should have usable contrast");
        assert_eq!(gradient_noise3(Vec3::splat(f32::NAN), 1), 0.0);
    }
}
