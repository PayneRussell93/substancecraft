//! A SubstanceCraft plugin (plugin ABI 1), built for `wasm32-unknown-unknown`.
//! See `docs/plugins.md` in the SubstanceCraft repository for the contract.
//!
//! - Generator 0, `wavy`: bands across the mesh that wobble along it.
//! - Filter 0, `sepia`: tints colour toward sepia; grey data passes through.
//!
//! The host hands over whole batches through linear memory, so the exports
//! below work on raw pointers into it. Floating-point maths that WebAssembly
//! has no instruction for (`sin` here) comes from Rust's own software
//! implementation, so results are identical on every machine.

/// Floats per mesh-map sample (see `docs/plugins.md`).
const SAMPLE_FLOATS: usize = 16;
const UNIT_POSITION: usize = 6;
const COVERED: usize = 15;

#[unsafe(no_mangle)]
pub extern "C" fn sc_abi_version() -> i32 {
    1
}

/// Hands the host `bytes` of memory, which it fills before each call.
/// Never freed: every call runs in a fresh instance.
#[unsafe(no_mangle)]
pub extern "C" fn sc_alloc(bytes: i32) -> i32 {
    let mut buffer = Vec::<u8>::with_capacity(bytes.max(0) as usize);
    let pointer = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    pointer as i32
}

/// Reads `count` floats the host placed at `pointer`.
fn floats<'a>(pointer: i32, count: usize) -> &'a mut [f32] {
    // SAFETY: the host allocated this memory through `sc_alloc` with room
    // for `count` floats, and nothing else touches it during the call.
    unsafe { std::slice::from_raw_parts_mut(pointer as usize as *mut f32, count) }
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Parameters: frequency (bands across the bounds), wobble, softness.
fn wavy(parameters: &[f32], sample: &[f32]) -> f32 {
    if sample[COVERED] == 0.0 {
        return 0.0;
    }
    let (frequency, wobble, softness) = (parameters[0], parameters[1], parameters[2]);
    let position = &sample[UNIT_POSITION..UNIT_POSITION + 3];
    let across = position[1] * frequency;
    let along = (position[0] * std::f32::consts::TAU * 2.0).sin() * wobble;
    let band = ((across + along) * std::f32::consts::PI).sin() * 0.5 + 0.5;
    let half = softness.max(0.001) * 0.5;
    smoothstep(0.5 - half, 0.5 + half, band)
}

#[unsafe(no_mangle)]
pub extern "C" fn sc_generate(
    generator: i32,
    parameters: i32,
    samples: i32,
    count: i32,
    out: i32,
) -> i32 {
    if generator != 0 {
        return 1;
    }
    let count = count.max(0) as usize;
    let parameters = floats(parameters, 8);
    let samples = floats(samples, count * SAMPLE_FLOATS);
    let out = floats(out, count);
    for (value, sample) in out.iter_mut().zip(samples.as_chunks::<SAMPLE_FLOATS>().0) {
        *value = wavy(parameters, sample);
    }
    0
}

/// Kinds of texel the host passes to `sc_filter`.
const COLOR: i32 = 0;

#[unsafe(no_mangle)]
pub extern "C" fn sc_filter(
    filter: i32,
    parameters: i32,
    texels: i32,
    count: i32,
    kind: i32,
) -> i32 {
    if filter != 0 {
        return 1;
    }
    if kind != COLOR {
        return 0;
    }
    let amount = floats(parameters, 8)[0].clamp(0.0, 1.0);
    for texel in floats(texels, count.max(0) as usize * 4)
        .as_chunks_mut::<4>()
        .0
    {
        let (r, g, b) = (texel[0], texel[1], texel[2]);
        let sepia = [
            r * 0.393 + g * 0.769 + b * 0.189,
            r * 0.349 + g * 0.686 + b * 0.168,
            r * 0.272 + g * 0.534 + b * 0.131,
        ];
        for (channel, toned) in texel.iter_mut().zip(sepia) {
            *channel += (toned.min(1.0) - *channel) * amount;
        }
    }
    0
}
