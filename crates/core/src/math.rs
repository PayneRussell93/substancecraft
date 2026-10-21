//! Transcendental functions that give the same bits on every platform
//! (PROD-08).
//!
//! `f32::sin` and its relatives call the platform's maths library, whose
//! last-bit rounding differs between Windows, Linux, and macOS, so one
//! project would bake and export slightly different textures on each. These
//! use the `libm` crate instead, a pure-Rust port of musl's maths library
//! built only from IEEE operations (add, multiply, divide, square root,
//! fused multiply-add) that round identically on every target. glam is built
//! with its `libm` and `scalar-math` features for the same reason: its own
//! trigonometry then comes from here too, and SIMD dot products (whose
//! summation order differs between SSE2 and NEON) are not used.
//!
//! `clippy.toml` in this crate and in the API crate disallows the platform
//! versions, so new code cannot reach them by accident.

/// The float types the functions here take.
pub trait Float: Copy + private::Sealed {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn powf(self, exponent: Self) -> Self;
    fn powi(self, exponent: i32) -> Self;
    fn tanh(self) -> Self;
    fn hypot(self, other: Self) -> Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

macro_rules! float_impl {
    ($type:ty, $sin:ident, $cos:ident, $tan:ident, $asin:ident, $acos:ident, $atan:ident,
     $atan2:ident, $exp:ident, $ln:ident, $log2:ident, $log10:ident, $pow:ident,
     $tanh:ident, $hypot:ident) => {
        impl Float for $type {
            fn sin(self) -> Self {
                libm::$sin(self)
            }
            fn cos(self) -> Self {
                libm::$cos(self)
            }
            fn tan(self) -> Self {
                libm::$tan(self)
            }
            fn asin(self) -> Self {
                libm::$asin(self)
            }
            fn acos(self) -> Self {
                libm::$acos(self)
            }
            fn atan(self) -> Self {
                libm::$atan(self)
            }
            fn atan2(self, x: Self) -> Self {
                libm::$atan2(self, x)
            }
            fn exp(self) -> Self {
                libm::$exp(self)
            }
            fn ln(self) -> Self {
                libm::$ln(self)
            }
            fn log2(self) -> Self {
                libm::$log2(self)
            }
            fn log10(self) -> Self {
                libm::$log10(self)
            }
            fn powf(self, exponent: Self) -> Self {
                libm::$pow(self, exponent)
            }
            fn powi(self, exponent: i32) -> Self {
                libm::$pow(self, exponent as Self)
            }
            fn tanh(self) -> Self {
                libm::$tanh(self)
            }
            fn hypot(self, other: Self) -> Self {
                libm::$hypot(self, other)
            }
        }
    };
}

float_impl!(
    f32, sinf, cosf, tanf, asinf, acosf, atanf, atan2f, expf, logf, log2f, log10f, powf, tanhf,
    hypotf
);
float_impl!(
    f64, sin, cos, tan, asin, acos, atan, atan2, exp, log, log2, log10, pow, tanh, hypot
);

pub fn sin<T: Float>(x: T) -> T {
    x.sin()
}

pub fn cos<T: Float>(x: T) -> T {
    x.cos()
}

/// `(sin x, cos x)`.
pub fn sin_cos<T: Float>(x: T) -> (T, T) {
    (x.sin(), x.cos())
}

pub fn tan<T: Float>(x: T) -> T {
    x.tan()
}

pub fn asin<T: Float>(x: T) -> T {
    x.asin()
}

pub fn acos<T: Float>(x: T) -> T {
    x.acos()
}

pub fn atan<T: Float>(x: T) -> T {
    x.atan()
}

/// The angle of `(x, y)`, like `y.atan2(x)`.
pub fn atan2<T: Float>(y: T, x: T) -> T {
    y.atan2(x)
}

pub fn exp<T: Float>(x: T) -> T {
    x.exp()
}

/// Natural logarithm.
pub fn ln<T: Float>(x: T) -> T {
    x.ln()
}

pub fn log2<T: Float>(x: T) -> T {
    x.log2()
}

pub fn log10<T: Float>(x: T) -> T {
    x.log10()
}

/// `x` to the power `exponent`.
pub fn powf<T: Float>(x: T, exponent: T) -> T {
    x.powf(exponent)
}

/// `x` to an integer power, through [`powf`] (`f32::powi` has no fixed
/// rounding across platforms).
pub fn powi<T: Float>(x: T, exponent: i32) -> T {
    x.powi(exponent)
}

pub fn tanh<T: Float>(x: T) -> T {
    x.tanh()
}

/// `sqrt(x² + y²)` without overflow.
pub fn hypot<T: Float>(x: T, y: T) -> T {
    x.hypot(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn functions_match_the_platform_to_within_rounding() {
        for step in -40..=40 {
            let x = step as f32 * 0.137;
            let close = |ours: f32, platform: f32| {
                (ours - platform).abs() <= 4.0 * f32::EPSILON * platform.abs().max(1.0)
            };
            #[allow(clippy::disallowed_methods)]
            {
                assert!(close(sin(x), x.sin()), "sin {x}");
                assert!(close(cos(x), x.cos()), "cos {x}");
                assert!(close(atan2(x, 0.7), x.atan2(0.7)), "atan2 {x}");
                assert!(close(exp(x * 0.1), (x * 0.1).exp()), "exp {x}");
                assert!(close(powf(x.abs(), 2.4), x.abs().powf(2.4)), "powf {x}");
                assert!(close(tanh(x), x.tanh()), "tanh {x}");
                assert!(close(powi(x, 3), x.powi(3)), "powi {x}");
            }
        }
        assert_eq!(sin_cos(0.0_f32), (0.0, 1.0));
        assert_eq!(acos(1.0_f64), 0.0);
        assert_eq!(hypot(3.0_f32, 4.0), 5.0);
    }

    /// Bits fixed here were produced by this code; every platform must give
    /// the same ones.
    #[test]
    fn results_are_fixed_bit_for_bit() {
        assert_eq!(sin(1.0_f32).to_bits(), 0x3f57_6aa4);
        assert_eq!(cos(1.0_f32).to_bits(), 0x3f0a_5140);
        assert_eq!(powf(0.5_f32, 2.4).to_bits(), 0x3e42_02ff);
        assert_eq!(atan2(1.0_f32, 2.0).to_bits(), 0x3eed_6338);
        assert_eq!(exp(1.0_f64).to_bits(), 0x4005_bf0a_8b14_576a);
    }
}
