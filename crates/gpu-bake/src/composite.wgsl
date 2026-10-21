// GPU layer compositing (PERF-06): the CPU compositor's stages, step for
// step, over a whole texture set. `composite.rs` builds the tables from a
// `CompositePlan` and runs one entry point per plan step; every function
// here mirrors the Rust function of the same name in `substancecraft-core`
// (compositing, filters, generators, noise, graph, resources, mesh maps).
//
// Small tables (layer records, projections, filters, generators, graph
// nodes, ramp stops, image headers, and the image pixels) share one buffer
// of u32 vec4s; float fields are stored as their bits and read through
// `row`, so integer data never passes through float registers. Planes of
// straight-alpha texels sit level after level in `planes`, followed by the
// Height plane the Normal channel's detail reads; `outputs` holds the four
// packed RGBA8 textures (Base Color, ORM, Normal, Emissive) per texel.

struct Globals {
    width: u32,
    height: u32,
    count: u32,
    levels: u32,
    maps_width: u32,
    maps_height: u32,
    has_maps: u32,
    _pad0: u32,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    diagonal: f32,
    height_scale: f32,
    _pad1: f32,
    _pad2: f32,
    // Where each table starts in `tables`, in vec4s.
    records: u32,
    projections: u32,
    filters: u32,
    generators: u32,
    nodes: u32,
    stops: u32,
    images: u32,
    pixels: u32,
}

struct Dispatch {
    record: u32,
    level: u32,
    data: u32,
    channel: u32,
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<uniform> dispatch: Dispatch;
@group(0) @binding(2) var<storage, read> tables: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> planes: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> upload: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> maps_a: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> maps_b: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> maps_c: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> maps_d: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> outputs: array<vec4<u32>>;

const EPSILON: f32 = 1.1920929e-7;
const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const NONE: u32 = 0xffffffffu;

const DATA_COLOR: u32 = 0u;
const DATA_SCALAR: u32 = 1u;
const DATA_VECTOR: u32 = 2u;

fn word(at: u32, lane: u32) -> u32 {
    return tables[at][lane];
}

/// A table row of floats.
fn row(at: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(tables[at]);
}

fn finite3(v: vec3<f32>) -> bool {
    return all(v == v) && all(abs(v) <= vec3<f32>(3.4e38));
}

fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let squared = dot(v, v);
    if !(squared > 0.0) || squared > 3.4e38 {
        return fallback;
    }
    let recip = 1.0 / sqrt(squared);
    if !(recip > 0.0) || recip > 3.4e38 {
        return fallback;
    }
    return v * recip;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Rust's `round` (half away from zero) for values ≥ 0.
fn round_half_up(x: f32) -> f32 {
    return floor(x + 0.5);
}

fn unit_to_byte(value: f32) -> u32 {
    return u32(round_half_up(clamp(value, 0.0, 1.0) * 255.0));
}

fn smoothstep_cpu(edge0: f32, edge1: f32, value: f32) -> f32 {
    if edge1 <= edge0 {
        return select(0.0, 1.0, value >= edge0);
    }
    let t = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn rem_euclid(x: f32, m: f32) -> f32 {
    let r = x - m * trunc(x / m);
    return select(r, r + abs(m), r < 0.0);
}

// ---- Mesh maps -------------------------------------------------------

struct Sample {
    covered: bool,
    position: vec3<f32>,
    unit_position: vec3<f32>,
    relative_position: vec3<f32>,
    normal: vec3<f32>,
    tangent_normal: vec3<f32>,
    curvature: f32,
    ambient_occlusion: f32,
    thickness: f32,
    skew: f32,
}

fn default_sample() -> Sample {
    return Sample(false, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0), 0.5, 1.0, 1.0, 0.0);
}

fn pixel_uv(x: u32, y: u32) -> vec2<f32> {
    return vec2<f32>(
        f32(x) / f32(max(g.width - 1u, 1u)),
        1.0 - f32(y) / f32(max(g.height - 1u, 1u)),
    );
}

/// MeshMaps::sample.
fn maps_sample(uv: vec2<f32>) -> Sample {
    if g.has_maps == 0u || g.maps_width == 0u || g.maps_height == 0u {
        return default_sample();
    }
    let fx = clamp(uv.x, 0.0, 1.0) * f32(g.maps_width - 1u);
    let fy = (1.0 - clamp(uv.y, 0.0, 1.0)) * f32(g.maps_height - 1u);
    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, g.maps_width - 1u);
    let y1 = min(y0 + 1u, g.maps_height - 1u);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    var xs = array<u32, 4>(x0, x1, x0, x1);
    var ys = array<u32, 4>(y0, y0, y1, y1);
    var weights = array<f32, 4>((1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty);
    var position = vec3<f32>(0.0);
    var normal = vec3<f32>(0.0);
    var tangent = vec3<f32>(0.0);
    var curvature = 0.0;
    var occlusion = 0.0;
    var thickness = 0.0;
    var skew = 0.0;
    var total = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let weight = weights[k];
        let index = ys[k] * g.maps_width + xs[k];
        let a = maps_a[index];
        if weight <= 0.0 || a.w < 0.5 {
            continue;
        }
        let b = maps_b[index];
        let c = maps_c[index];
        let d = maps_d[index];
        position += a.xyz * weight;
        normal += b.xyz * weight;
        tangent += c.xyz * weight;
        curvature += b.w * weight;
        occlusion += c.w * weight;
        thickness += d.x * weight;
        skew += d.y * weight;
        total += weight;
    }
    if total <= EPSILON {
        return default_sample();
    }
    let p = position / total;
    let extent = max(g.bounds_max.xyz - g.bounds_min.xyz, vec3<f32>(EPSILON));
    return Sample(
        true,
        p,
        clamp((p - g.bounds_min.xyz) / extent, vec3<f32>(0.0), vec3<f32>(1.0)),
        (p - g.bounds_min.xyz) / g.diagonal,
        normalize_or(normal, vec3<f32>(0.0, 1.0, 0.0)),
        normalize_or(tangent, vec3<f32>(0.0, 0.0, 1.0)),
        curvature / total,
        occlusion / total,
        thickness / total,
        skew / total,
    );
}

// ---- Images ------------------------------------------------------------

fn pixel_word(at: u32) -> u32 {
    return tables[g.pixels + at / 4u][at % 4u];
}

/// TextureImage::pixel.
fn image_pixel(image: u32, index: u32) -> vec4<f32> {
    let header = g.images + image * 2u;
    let offset = word(header, 0u);
    if word(header, 3u) != 0u {
        let lo = pixel_word(offset + index * 2u);
        let hi = pixel_word(offset + index * 2u + 1u);
        return vec4<f32>(
            f32(lo & 0xffffu), f32(lo >> 16u), f32(hi & 0xffffu), f32(hi >> 16u),
        ) / 65535.0;
    }
    let rgba = pixel_word(offset + index);
    return vec4<f32>(
        f32(rgba & 0xffu), f32((rgba >> 8u) & 0xffu), f32((rgba >> 16u) & 0xffu), f32(rgba >> 24u),
    ) / 255.0;
}

fn image_texel(image: u32, x: i32, y: i32, wrap: bool) -> vec4<f32> {
    let header = g.images + image * 2u;
    let width = i32(word(header, 1u));
    let height = i32(word(header, 2u));
    var px = x;
    var py = y;
    if wrap {
        px = ((x % width) + width) % width;
        py = ((y % height) + height) % height;
    } else {
        px = clamp(x, 0, width - 1);
        py = clamp(y, 0, height - 1);
    }
    return image_pixel(image, u32(py * width + px));
}

/// TextureImage::sample.
fn image_sample(image: u32, uv: vec2<f32>, wrap: bool) -> vec4<f32> {
    if !finite3(vec3<f32>(uv, 0.0)) {
        return vec4<f32>(0.0);
    }
    let header = g.images + image * 2u;
    let x = uv.x * f32(word(header, 1u)) - 0.5;
    let y = (1.0 - uv.y) * f32(word(header, 2u)) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = x - x0;
    let fy = y - y0;
    let ix = i32(x0);
    let iy = i32(y0);
    let top = mix(image_texel(image, ix, iy, wrap), image_texel(image, ix + 1, iy, wrap), fx);
    let bottom = mix(image_texel(image, ix, iy + 1, wrap), image_texel(image, ix + 1, iy + 1, wrap), fx);
    return mix(top, bottom, fy);
}

fn image_color_space(image: u32) -> u32 {
    return word(g.images + image * 2u + 1u, 0u);
}

// ---- Colour --------------------------------------------------------------

fn linear_to_srgb(value: f32) -> f32 {
    let v = max(value, 0.0);
    if v <= 0.0031308 {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

/// ColorSpace::to_srgb: sRGB 0, linear sRGB 1, ACEScg 2, raw 3.
fn to_srgb(space: u32, rgb: vec3<f32>) -> vec3<f32> {
    if space == 0u || space == 3u {
        return rgb;
    }
    var linear = rgb;
    if space == 2u {
        let aces = mat3x3<f32>(
            vec3<f32>(1.705051, -0.130256, -0.024003),
            vec3<f32>(-0.621792, 1.140805, -0.128969),
            vec3<f32>(-0.083258, -0.010548, 1.152972),
        );
        linear = aces * rgb;
    }
    return vec3<f32>(linear_to_srgb(linear.x), linear_to_srgb(linear.y), linear_to_srgb(linear.z));
}

/// texture_texel.
fn texture_texel(texel: vec4<f32>, data: u32, space: u32) -> vec4<f32> {
    if data == DATA_COLOR {
        return vec4<f32>(to_srgb(space, texel.xyz), texel.w);
    }
    if data == DATA_SCALAR {
        return vec4<f32>(vec3<f32>(luminance(texel.xyz)), texel.w);
    }
    return vec4<f32>(normalize_or(texel.xyz * 2.0 - 1.0, vec3<f32>(0.0, 0.0, 1.0)), texel.w);
}

// ---- Projections -----------------------------------------------------------

const PROJECTION_VEC4S: u32 = 9u;

fn projection_transform(p: u32, c: vec2<f32>) -> vec2<f32> {
    let rotation = row(p + 1u);
    let sin_r = rotation.y;
    let cos_r = rotation.z;
    let centered = c - vec2<f32>(0.5);
    let rotated = vec2<f32>(centered.x * cos_r - centered.y * sin_r, centered.x * sin_r + centered.y * cos_r);
    var scale = rotation.x;
    if word(p, 2u) != 0u {
        scale = 1.0;
    }
    return (rotated + vec2<f32>(0.5)) * scale + row(p + 2u).xy;
}

fn object_coordinates(p: u32, s: Sample) -> vec3<f32> {
    if word(p, 2u) != 0u {
        return s.position / row(p + 2u).z;
    }
    return s.relative_position;
}

fn across(v: vec3<f32>, axis: u32) -> vec2<f32> {
    if axis == 0u {
        return vec2<f32>(v.z, v.y);
    }
    if axis == 1u {
        return vec2<f32>(v.x, v.z);
    }
    return vec2<f32>(v.x, v.y);
}

fn along(v: vec3<f32>, axis: u32) -> f32 {
    if axis == 0u {
        return v.x;
    }
    if axis == 1u {
        return v.y;
    }
    return v.z;
}

/// DecalPlacement::locate: the decal UV and how much of it shows.
fn decal_locate(p: u32, position: vec3<f32>, surface_normal: vec3<f32>) -> vec3<f32> {
    let center = row(p + 3u).xyz;
    let normal = row(p + 4u).xyz;
    let right = row(p + 5u).xyz;
    let up = row(p + 6u).xyz;
    let size = row(p + 7u).xy;
    let region_min = row(p + 7u).zw;
    let region_max = row(p + 8u).xy;
    let depth = row(p + 2u).w;
    let local = position - center;
    let unit = vec2<f32>(dot(local, right) / size.x + 0.5, dot(local, up) / size.y + 0.5);
    let inside = all(unit >= vec2<f32>(0.0)) && all(unit <= vec2<f32>(1.0)) && abs(dot(local, normal)) <= depth;
    let facing = dot(surface_normal, normal);
    var coverage = 0.0;
    if inside {
        coverage = clamp((facing - 0.05) / 0.25, 0.0, 1.0);
    }
    return vec3<f32>(region_min + unit * (region_max - region_min), coverage);
}

/// TextureProjection::coverage.
fn projection_coverage(p: u32, s: Sample) -> f32 {
    if word(p, 0u) != 5u {
        return 1.0;
    }
    if !s.covered {
        return 0.0;
    }
    return decal_locate(p, s.position, s.normal).z;
}

/// TextureProjection::sample; `with_sample` is false where the CPU passes
/// no mesh-map sample.
fn projection_sample(p: u32, image: u32, uv: vec2<f32>, s: Sample, with_sample: bool) -> vec4<f32> {
    let mode = word(p, 0u);
    let axis = word(p, 1u);
    if !with_sample || mode == 0u {
        return image_sample(image, projection_transform(p, uv), true);
    }
    if mode == 1u {
        let position = object_coordinates(p, s);
        let sharpness = row(p + 1u).w;
        let n = abs(s.normal);
        var weights = vec3<f32>(pow(n.x, sharpness), pow(n.y, sharpness), pow(n.z, sharpness));
        let total = weights.x + weights.y + weights.z;
        if total <= EPSILON {
            return image_sample(image, projection_transform(p, position.xy), true);
        }
        weights = weights / total;
        var result = vec4<f32>(0.0);
        if weights.x > 0.001 {
            result += image_sample(image, projection_transform(p, across(position, 0u)), true) * weights.x;
        }
        if weights.y > 0.001 {
            result += image_sample(image, projection_transform(p, across(position, 1u)), true) * weights.y;
        }
        if weights.z > 0.001 {
            result += image_sample(image, projection_transform(p, across(position, 2u)), true) * weights.z;
        }
        return result;
    }
    if mode == 2u {
        return image_sample(image, projection_transform(p, across(object_coordinates(p, s), axis)), true);
    }
    if mode == 3u {
        let around = across(s.unit_position - vec3<f32>(0.5), axis);
        let angle = atan2(around.y, around.x) / TAU + 0.5;
        let height = along(object_coordinates(p, s), axis);
        return image_sample(image, projection_transform(p, vec2<f32>(angle, height)), true);
    }
    if mode == 4u {
        let direction = normalize_or(s.unit_position - vec3<f32>(0.5), vec3<f32>(0.0, 1.0, 0.0));
        let around = across(direction, axis);
        let longitude = atan2(around.y, around.x) / TAU + 0.5;
        let latitude = asin(clamp(along(direction, axis), -1.0, 1.0)) / PI + 0.5;
        return image_sample(image, projection_transform(p, vec2<f32>(longitude, latitude)), true);
    }
    let located = decal_locate(p, s.position, s.normal);
    if located.z <= 0.0 {
        return vec4<f32>(0.0);
    }
    let texel = image_sample(image, located.xy, false);
    return vec4<f32>(texel.xyz, texel.w * located.z);
}

// ---- Noise -----------------------------------------------------------------

fn lowbias32(input: u32) -> u32 {
    var v = input;
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    var value = seed ^ 0x9e3779b9u;
    value ^= bitcast<u32>(x) * 0x85ebca6bu + 0xc2b2ae35u;
    value = lowbias32(value);
    value ^= bitcast<u32>(y) * 0x85ebca6bu + 0xc2b2ae35u;
    value = lowbias32(value);
    value ^= bitcast<u32>(z) * 0x85ebca6bu + 0xc2b2ae35u;
    value = lowbias32(value);
    return value;
}

fn gradient(index: u32) -> vec3<f32> {
    switch index {
        case 0u: { return vec3<f32>(1.0, 1.0, 0.0); }
        case 1u: { return vec3<f32>(-1.0, 1.0, 0.0); }
        case 2u: { return vec3<f32>(1.0, -1.0, 0.0); }
        case 3u: { return vec3<f32>(-1.0, -1.0, 0.0); }
        case 4u: { return vec3<f32>(1.0, 0.0, 1.0); }
        case 5u: { return vec3<f32>(-1.0, 0.0, 1.0); }
        case 6u: { return vec3<f32>(1.0, 0.0, -1.0); }
        case 7u: { return vec3<f32>(-1.0, 0.0, -1.0); }
        case 8u: { return vec3<f32>(0.0, 1.0, 1.0); }
        case 9u: { return vec3<f32>(0.0, -1.0, 1.0); }
        case 10u: { return vec3<f32>(0.0, 1.0, -1.0); }
        default: { return vec3<f32>(0.0, -1.0, -1.0); }
    }
}

fn quintic(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn corner(base: vec3<i32>, local: vec3<f32>, offset: vec3<i32>, seed: u32) -> f32 {
    let hash = hash3(base.x + offset.x, base.y + offset.y, base.z + offset.z, seed);
    return dot(gradient(hash % 12u), local - vec3<f32>(offset));
}

/// gradient_noise3.
fn gradient_noise3(point: vec3<f32>, seed: u32) -> f32 {
    if !finite3(point) {
        return 0.0;
    }
    let cell = floor(point);
    let local = point - cell;
    let base = vec3<i32>(cell);
    let fade = vec3<f32>(quintic(local.x), quintic(local.y), quintic(local.z));
    let x00 = mix(corner(base, local, vec3<i32>(0, 0, 0), seed), corner(base, local, vec3<i32>(1, 0, 0), seed), fade.x);
    let x10 = mix(corner(base, local, vec3<i32>(0, 1, 0), seed), corner(base, local, vec3<i32>(1, 1, 0), seed), fade.x);
    let x01 = mix(corner(base, local, vec3<i32>(0, 0, 1), seed), corner(base, local, vec3<i32>(1, 0, 1), seed), fade.x);
    let x11 = mix(corner(base, local, vec3<i32>(0, 1, 1), seed), corner(base, local, vec3<i32>(1, 1, 1), seed), fade.x);
    let y0 = mix(x00, x10, fade.y);
    let y1 = mix(x01, x11, fade.y);
    return clamp(mix(y0, y1, fade.z), -1.0, 1.0);
}

/// fbm3.
fn fbm3(point: vec3<f32>, octaves: u32, seed: u32) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var total = 0.0;
    var frequency = 1.0;
    for (var octave = 0u; octave < clamp(octaves, 1u, 8u); octave++) {
        sum += gradient_noise3(point * frequency, seed + octave * 0x6d2b79f5u) * amplitude;
        total += amplitude;
        amplitude *= 0.5;
        frequency *= 2.03;
    }
    return clamp(sum / total * 0.5 + 0.5, 0.0, 1.0);
}

// ---- Generators ------------------------------------------------------------

const BREAKUP_OCTAVES: u32 = 4u;
const GENERATOR_VEC4S: u32 = 3u;

fn noise_point(s: Sample, scale: f32, has_feature: bool, feature: f32) -> vec3<f32> {
    if has_feature {
        return s.position / feature;
    }
    return s.relative_position * scale;
}

fn breakup(point: vec3<f32>, strength: f32, seed: u32) -> f32 {
    if strength <= 0.0 {
        return 1.0;
    }
    let noise = fbm3(point, BREAKUP_OCTAVES, seed);
    return 1.0 + (noise * 2.0 - 1.0) * strength * 1.5;
}

fn level(value: f32, amount: f32, contrast: f32) -> f32 {
    let half_width = 0.3 - 0.29 * contrast;
    let threshold = (1.0 + half_width) + (half_width - (1.0 + half_width)) * amount;
    if value <= 0.0 {
        return 0.0;
    }
    return smoothstep_cpu(threshold - half_width, threshold + half_width, value);
}

fn component(v: vec3<f32>, axis: u32) -> f32 {
    return along(v, axis);
}

/// Pattern::evaluate.
fn pattern(at: u32, s: Sample) -> f32 {
    let flags = word(at, 3u);
    let kind = flags >> 8u;
    let seed = word(at, 1u);
    let axis = word(at, 2u);
    let settings = row(at + 1u);
    let scale = settings.x;
    let width = settings.y;
    let softness = settings.z;
    let feature = settings.w;
    let point = noise_point(s, scale, (flags & 1u) != 0u, feature);
    let band = max(softness * 0.5, 1e-3);
    let along_axis = component(point, axis);
    if kind == 0u {
        let phase = abs(along_axis - floor(along_axis) - 0.5);
        return 1.0 - smoothstep_cpu(width * 0.5 - band, width * 0.5 + band, phase);
    }
    if kind == 1u {
        let cells = floor(point);
        let parity = rem_euclid(cells.x + cells.y + cells.z, 2.0);
        let fraction = point - cells;
        let nearest = min(fraction, vec3<f32>(1.0) - fraction);
        let border = min(nearest.x, min(nearest.y, nearest.z));
        let edge = smoothstep_cpu(0.0, band, border);
        if parity < 0.5 {
            return 0.5 + 0.5 * edge;
        }
        return 0.5 - 0.5 * edge;
    }
    if kind == 2u {
        let offset = point - floor(point) - vec3<f32>(0.5);
        return 1.0 - smoothstep_cpu(width * 0.5 - band, width * 0.5 + band, length(offset));
    }
    if kind == 3u {
        var stretched = point * 3.0;
        if axis == 0u {
            stretched.x *= 0.08;
        } else if axis == 1u {
            stretched.y *= 0.08;
        } else {
            stretched.z *= 0.08;
        }
        let noise = gradient_noise3(stretched, seed);
        let ridge = 1.0 - min(abs(noise), 1.0);
        let density = gradient_noise3(point * 0.7, seed ^ 0x9e37u) * 0.5 + 0.5;
        let edge = width * 0.15;
        let inside = 1.0 - smoothstep_cpu(edge - band, edge + band, 1.0 - ridge);
        return inside * smoothstep_cpu(0.35, 0.6, density);
    }
    let noise = fbm3(point * 2.0, BREAKUP_OCTAVES, seed) * 2.0 - 1.0;
    let ridge = 1.0 - min(abs(noise), 1.0);
    let edge = width * 0.3;
    return 1.0 - smoothstep_cpu(edge - band, edge + band, 1.0 - ridge);
}

/// MaskGenerator::evaluate: edge wear 0, dirt 1, orientation 2, gradient 3,
/// noise 4, thickness 5, pattern 6.
fn generate(index: u32, s: Sample) -> f32 {
    if !s.covered {
        return 0.0;
    }
    let at = g.generators + index * GENERATOR_VEC4S;
    let kind = word(at, 0u);
    let seed = word(at, 1u);
    let flags = word(at, 3u);
    let has_feature = (flags & 1u) != 0u;
    let a = row(at + 1u);
    let b = row(at + 2u);
    var value = 0.0;
    if kind == 0u {
        let convexity = clamp((s.curvature - 0.5) * 2.0, 0.0, 1.0);
        let chips = breakup(noise_point(s, a.w, has_feature, b.x), a.z, seed);
        value = level(convexity * chips, a.x, a.y);
    } else if kind == 1u {
        let cavity = clamp((0.5 - s.curvature) * 2.0, 0.0, 1.0);
        let occlusion = 1.0 - clamp(s.ambient_occlusion, 0.0, 1.0);
        let grime = occlusion + (cavity - occlusion) * a.z;
        let chips = breakup(noise_point(s, b.x, has_feature, b.y), a.w, seed);
        value = level(grime * chips, a.x, a.y);
    } else if kind == 2u {
        let angle = acos(clamp(dot(s.normal, a.xyz), -1.0, 1.0)) * (180.0 / PI);
        let half_width = max(b.x * 45.0, 0.5);
        value = 1.0 - smoothstep_cpu(a.w - half_width, a.w + half_width, angle);
    } else if kind == 3u {
        let coordinate = component(s.unit_position, word(at, 2u));
        let span = a.y - a.x;
        if abs(span) <= EPSILON {
            value = select(0.0, 1.0, coordinate >= a.x);
        } else {
            value = (coordinate - a.x) / span;
        }
    } else if kind == 4u {
        let noise = fbm3(noise_point(s, a.x, has_feature, a.w), BREAKUP_OCTAVES, seed);
        let half_width = 0.5 - 0.48 * a.z;
        let threshold = 1.0 - a.y;
        value = smoothstep_cpu(threshold - half_width, threshold + half_width, noise);
    } else if kind == 5u {
        let thickness = clamp(s.thickness, 0.0, 1.0);
        var selected = thickness;
        if (flags & 2u) != 0u {
            selected = 1.0 - thickness;
        }
        value = level(selected, a.x, a.y);
    } else {
        value = pattern(at, s);
    }
    return clamp(value, 0.0, 1.0);
}

// ---- Filters -------------------------------------------------------------

const FILTER_VEC4S: u32 = 3u;

fn gradient_color(dark: vec3<f32>, light: vec3<f32>, amount: f32) -> vec3<f32> {
    return mix(dark, light, clamp(amount, 0.0, 1.0));
}

/// LayerFilter::map_value: levels 0, invert 1, hue/saturation 2,
/// brightness/contrast 3, posterize 4, threshold 5, gradient map 6.
fn map_value(at: u32, value: f32) -> f32 {
    let kind = word(at, 0u);
    let a = row(at + 1u);
    let b = row(at + 2u);
    if kind == 0u {
        let span = a.y - a.x;
        var normalized = 0.0;
        if abs(span) < 1e-4 {
            normalized = select(0.0, 1.0, value >= a.y);
        } else {
            normalized = clamp((value - a.x) / span, 0.0, 1.0);
        }
        var curved = 0.0;
        if normalized > 0.0 {
            curved = pow(normalized, 1.0 / a.z);
        }
        return a.w + (b.x - a.w) * curved;
    }
    if kind == 1u {
        return 1.0 - value;
    }
    if kind == 2u {
        return clamp(value + a.z, 0.0, 1.0);
    }
    if kind == 3u {
        return clamp((value - 0.5) * (1.0 + a.y) + 0.5 + a.x, 0.0, 1.0);
    }
    if kind == 4u {
        let steps = f32(max(word(at, 1u), 2u) - 1u);
        return round_half_up(clamp(value, 0.0, 1.0) * steps) / steps;
    }
    if kind == 5u {
        if a.y <= 0.0 {
            return select(0.0, 1.0, value >= a.x);
        }
        return clamp((value - (a.x - a.y * 0.5)) / a.y, 0.0, 1.0);
    }
    return luminance(gradient_color(a.xyz, b.xyz, value));
}

fn rgb_to_hsl(color: vec3<f32>) -> vec3<f32> {
    let high = max(color.x, max(color.y, color.z));
    let low = min(color.x, min(color.y, color.z));
    let lightness = (high + low) * 0.5;
    let delta = high - low;
    if delta <= 1e-6 {
        return vec3<f32>(0.0, 0.0, lightness);
    }
    let saturation = delta / max(1.0 - abs(2.0 * lightness - 1.0), 1e-6);
    var hue = 0.0;
    if high == color.x {
        hue = rem_euclid((color.y - color.z) / delta, 6.0);
    } else if high == color.y {
        hue = (color.z - color.x) / delta + 2.0;
    } else {
        hue = (color.x - color.y) / delta + 4.0;
    }
    return vec3<f32>(hue / 6.0, clamp(saturation, 0.0, 1.0), lightness);
}

fn hsl_to_rgb(hue: f32, saturation: f32, lightness: f32) -> vec3<f32> {
    let chroma = (1.0 - abs(2.0 * lightness - 1.0)) * saturation;
    let sector = hue * 6.0;
    let secondary = chroma * (1.0 - abs(rem_euclid(sector, 2.0) - 1.0));
    var rgb = vec3<f32>(chroma, 0.0, secondary);
    switch u32(max(sector, 0.0)) {
        case 0u: { rgb = vec3<f32>(chroma, secondary, 0.0); }
        case 1u: { rgb = vec3<f32>(secondary, chroma, 0.0); }
        case 2u: { rgb = vec3<f32>(0.0, chroma, secondary); }
        case 3u: { rgb = vec3<f32>(0.0, secondary, chroma); }
        case 4u: { rgb = vec3<f32>(secondary, 0.0, chroma); }
        default: {}
    }
    return clamp(rgb + vec3<f32>(lightness - chroma * 0.5), vec3<f32>(0.0), vec3<f32>(1.0));
}

/// LayerFilter::map_color.
fn map_color(at: u32, color: vec3<f32>) -> vec3<f32> {
    let kind = word(at, 0u);
    if kind == 2u {
        let a = row(at + 1u);
        let hsl = rgb_to_hsl(color);
        return hsl_to_rgb(
            rem_euclid(hsl.x + a.x, 1.0),
            clamp(hsl.y * a.y, 0.0, 1.0),
            clamp(hsl.z + a.z, 0.0, 1.0),
        );
    }
    if kind == 6u {
        return gradient_color(row(at + 1u).xyz, row(at + 2u).xyz, luminance(color));
    }
    return vec3<f32>(map_value(at, color.x), map_value(at, color.y), map_value(at, color.z));
}

/// filters::apply_to_plane at one texel, for per-texel filters.
fn apply_filters(start: u32, count: u32, texel: vec4<f32>, data: u32) -> vec4<f32> {
    var result = texel;
    if data != DATA_VECTOR {
        for (var k = 0u; k < count; k++) {
            let at = g.filters + (start + k) * FILTER_VEC4S;
            if data == DATA_COLOR {
                result = vec4<f32>(map_color(at, result.xyz), result.w);
            } else {
                result = vec4<f32>(vec3<f32>(map_value(at, result.x)), result.w);
            }
        }
        return result;
    }
    return vec4<f32>(normalize_or(result.xyz, vec3<f32>(0.0, 0.0, 1.0)), result.w);
}

// ---- Blending ----------------------------------------------------------------

/// BlendMode::apply: normal 0, multiply 1, screen 2, overlay 3, soft light
/// 4, darken 5, lighten 6, difference 7, add 8, subtract 9, pass-through 10.
fn blend_apply(mode: u32, b: f32, s: f32) -> f32 {
    var result = s;
    switch mode {
        case 1u: { result = b * s; }
        case 2u: { result = b + s - b * s; }
        case 3u: {
            if b <= 0.5 {
                result = 2.0 * b * s;
            } else {
                result = 1.0 - 2.0 * (1.0 - b) * (1.0 - s);
            }
        }
        case 4u: {
            if s <= 0.5 {
                result = b - (1.0 - 2.0 * s) * b * (1.0 - b);
            } else {
                var d = sqrt(b);
                if b <= 0.25 {
                    d = ((16.0 * b - 12.0) * b + 4.0) * b;
                }
                result = b + (2.0 * s - 1.0) * (d - b);
            }
        }
        case 5u: { result = min(b, s); }
        case 6u: { result = max(b, s); }
        case 7u: { result = abs(b - s); }
        case 8u: { result = b + s; }
        case 9u: { result = b - s; }
        default: {}
    }
    return clamp(result, 0.0, 1.0);
}

/// over.
fn over(destination: vec4<f32>, value: vec3<f32>, alpha_in: f32, data: u32) -> vec4<f32> {
    let alpha = clamp(alpha_in, 0.0, 1.0);
    if alpha <= 0.0 {
        return destination;
    }
    let out_alpha = alpha + destination.w * (1.0 - alpha);
    if out_alpha <= 1e-6 {
        return vec4<f32>(0.0);
    }
    var blended = (value * alpha + destination.xyz * destination.w * (1.0 - alpha)) / out_alpha;
    if data == DATA_VECTOR {
        blended = normalize_or(blended, vec3<f32>(0.0, 0.0, 1.0));
    }
    return vec4<f32>(blended, out_alpha);
}

/// The blend step of Compositor::composite.
fn lay_over(destination: vec4<f32>, source: vec4<f32>, alpha: f32, mode: u32, data: u32) -> vec4<f32> {
    var value = source.xyz;
    if mode != 0u && data != DATA_VECTOR {
        let backdrop = destination.xyz;
        let blended = vec3<f32>(
            blend_apply(mode, backdrop.x, value.x),
            blend_apply(mode, backdrop.y, value.y),
            blend_apply(mode, backdrop.z, value.z),
        );
        value = mix(value, blended, clamp(destination.w, 0.0, 1.0));
    }
    return over(destination, value, alpha, data);
}

// ---- Graphs --------------------------------------------------------------

const NODE_VEC4S: u32 = 4u;

fn graph_read(input: vec4<f32>, values: ptr<function, array<vec3<f32>, 64>>) -> vec3<f32> {
    if input.w < 0.0 {
        return input.xyz;
    }
    return (*values)[u32(input.w)];
}

fn math_apply(op: u32, a: f32, b: f32) -> f32 {
    var value = 0.0;
    switch op {
        case 0u: { value = a + b; }
        case 1u: { value = a - b; }
        case 2u: { value = a * b; }
        case 3u: {
            if b == 0.0 {
                value = 0.0;
            } else {
                value = a / b;
            }
        }
        case 4u: { value = min(a, b); }
        case 5u: { value = max(a, b); }
        default: {
            let base = max(a, 0.0);
            if base == 0.0 {
                value = select(0.0, 1.0, b == 0.0);
                if b < 0.0 {
                    value = 3.5e38;
                }
            } else {
                value = pow(base, b);
            }
        }
    }
    if !finite3(vec3<f32>(value)) || abs(value) > 3.4e38 {
        return 0.0;
    }
    return clamp(value, -1000.0, 1000.0);
}

fn ramp(start: u32, count: u32, amount_in: f32) -> vec3<f32> {
    let amount = clamp(amount_in, 0.0, 1.0);
    if count == 0u {
        return vec3<f32>(amount);
    }
    let first = row(g.stops + start);
    if amount <= first.x {
        return first.yzw;
    }
    for (var k = 0u; k + 1u < count; k++) {
        let low = row(g.stops + start + k);
        let high = row(g.stops + start + k + 1u);
        if amount <= high.x {
            let span = high.x - low.x;
            var t = 1.0;
            if span > EPSILON {
                t = (amount - low.x) / span;
            }
            return mix(low.yzw, high.yzw, t);
        }
    }
    return row(g.stops + start + count - 1u).yzw;
}

fn mesh_map_value(kind: u32, s: Sample) -> vec3<f32> {
    if !s.covered {
        return vec3<f32>(0.0);
    }
    switch kind {
        case 0u: { return s.tangent_normal * 0.5 + 0.5; }
        case 1u: { return s.normal * 0.5 + 0.5; }
        case 2u: { return s.unit_position; }
        case 3u: { return vec3<f32>(s.curvature); }
        case 4u: { return vec3<f32>(s.ambient_occlusion); }
        case 5u: { return vec3<f32>(s.thickness); }
        default: { return vec3<f32>(s.skew); }
    }
}

/// GraphEvaluator::evaluate up to node `last`.
fn evaluate_graph(start: u32, last: u32, uv: vec2<f32>, s: Sample) -> vec3<f32> {
    var values: array<vec3<f32>, 64>;
    for (var index = 0u; index <= last; index++) {
        let at = g.nodes + (start + index) * NODE_VEC4S;
        let kind = word(at, 0u);
        let a = word(at, 1u);
        let b = word(at, 2u);
        var value = vec3<f32>(0.0);
        if kind == 0u {
            value = vec3<f32>(generate(a, s));
        } else if kind == 1u {
            value = mesh_map_value(a, s);
        } else if kind == 2u {
            value = vec3<f32>(uv, 0.0);
        } else if kind == 3u {
            let p = g.projections + b * PROJECTION_VEC4S;
            let texel = projection_sample(p, a, uv, s, g.has_maps != 0u);
            value = to_srgb(image_color_space(a), texel.xyz);
        } else if kind == 4u {
            let x = graph_read(row(at + 1u), &values);
            let y = graph_read(row(at + 2u), &values);
            let amount = graph_read(row(at + 3u), &values);
            let blended = vec3<f32>(blend_apply(a, x.x, y.x), blend_apply(a, x.y, y.y), blend_apply(a, x.z, y.z));
            value = mix(x, blended, clamp(luminance(amount), 0.0, 1.0));
        } else if kind == 5u {
            let x = graph_read(row(at + 1u), &values);
            let y = graph_read(row(at + 2u), &values);
            value = vec3<f32>(math_apply(a, x.x, y.x), math_apply(a, x.y, y.y), math_apply(a, x.z, y.z));
        } else if kind == 6u {
            value = map_color(g.filters + a * FILTER_VEC4S, graph_read(row(at + 1u), &values));
        } else if kind == 7u {
            value = ramp(a, b, luminance(graph_read(row(at + 1u), &values)));
        } else {
            let x = graph_read(row(at + 1u), &values);
            value = vec3<f32>(x.x);
            if a == 1u {
                value = vec3<f32>(x.y);
            } else if a == 2u {
                value = vec3<f32>(x.z);
            }
        }
        if finite3(value) {
            values[index] = value;
        } else {
            values[index] = vec3<f32>(0.0);
        }
    }
    return values[last];
}

// ---- Layer records ---------------------------------------------------------

const RECORD_VEC4S: u32 = 8u;

fn record(index: u32) -> u32 {
    return g.records + index * RECORD_VEC4S;
}

/// What a layer lays over the stack, before its filters.
fn content(r: u32, i: u32, x: u32, y: u32, uv: vec2<f32>, s: Sample, data: u32) -> vec4<f32> {
    let kind = word(r, 0u);
    if kind == 0u {
        let texel = row(r + 2u);
        if word(r + 1u, 3u) != 0u {
            let p = g.projections + word(r, 2u) * PROJECTION_VEC4S;
            return vec4<f32>(texel.xyz, texel.w * projection_coverage(p, s));
        }
        return texel;
    }
    if kind == 1u {
        let image = word(r, 1u);
        let space = image_color_space(image);
        if word(r, 3u) != 0u {
            return texture_texel(image_pixel(image, y * g.width + x), data, space);
        }
        let p = g.projections + word(r, 2u) * PROJECTION_VEC4S;
        let with_sample = word(p, 0u) != 0u;
        return texture_texel(projection_sample(p, image, uv, s, with_sample), data, space);
    }
    if kind == 2u {
        var value = row(r + 2u).xyz;
        let last = word(r + 1u, 2u);
        if last != NONE {
            value = evaluate_graph(word(r + 1u, 0u), last, uv, s);
        }
        if data == DATA_COLOR {
            return vec4<f32>(clamp(value, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
        }
        if data == DATA_SCALAR {
            return vec4<f32>(vec3<f32>(clamp(luminance(value), 0.0, 1.0)), 1.0);
        }
        return vec4<f32>(normalize_or(value * 2.0 - 1.0, vec3<f32>(0.0, 0.0, 1.0)), 1.0);
    }
    return upload[i];
}

/// TextureSet::layer_mask_in for masks a plan hands over.
fn mask_value(r: u32, i: u32, uv: vec2<f32>, s: Sample) -> f32 {
    let kind = word(r + 3u, 2u);
    if kind == 0u {
        return 1.0;
    }
    if kind == 2u {
        return upload[g.count + i].x;
    }
    var value = 1.0;
    let generator = word(r + 3u, 3u);
    if generator != NONE {
        value = generate(generator, s);
    }
    if word(r + 4u, 2u) != 0u {
        let image = word(r + 4u, 0u);
        let p = g.projections + word(r + 4u, 1u) * PROJECTION_VEC4S;
        let with_sample = word(p, 0u) != 0u;
        let texel = projection_sample(p, image, uv, s, with_sample);
        var bitmap = luminance(texel.xyz) * texel.w;
        if word(r + 4u, 3u) != 0u {
            bitmap = 1.0 - bitmap;
        }
        value *= bitmap;
    }
    let filters = word(r + 5u, 1u);
    if filters > 0u {
        let filtered = apply_filters(word(r + 5u, 0u), filters, vec4<f32>(vec3<f32>(value), 1.0), DATA_SCALAR);
        value = clamp(filtered.x, 0.0, 1.0);
    }
    if word(r + 5u, 2u) != 0u {
        value = 1.0 - value;
    }
    return clamp(value, 0.0, 1.0);
}

fn plane_at(level: u32, i: u32) -> u32 {
    return level * g.count + i;
}

// ---- Entry points ------------------------------------------------------------

/// The plane a channel starts from: `dispatch.record` holds the base
/// texel's bits, or NONE to start from the baked tangent normals.
@compute @workgroup_size(256)
fn start_plane(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= g.count {
        return;
    }
    if dispatch.record == NONE {
        let s = maps_sample(pixel_uv(i % g.width, i / g.width));
        planes[plane_at(0u, i)] = vec4<f32>(s.tangent_normal, 1.0);
    } else {
        planes[plane_at(0u, i)] = row(dispatch.record);
    }
}

/// One leaf layer over the plane at `dispatch.level`.
@compute @workgroup_size(256)
fn layer(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= g.count {
        return;
    }
    let r = record(dispatch.record);
    let data = dispatch.data;
    let x = i % g.width;
    let y = i / g.width;
    let uv = pixel_uv(x, y);
    let s = maps_sample(uv);
    let local = apply_filters(word(r + 3u, 0u), word(r + 3u, 1u), content(r, i, x, y, uv, s, data), data);
    if local.w <= 0.0 {
        return;
    }
    let alpha = local.w * row(r + 6u).x * mask_value(r, i, uv, s);
    let at = plane_at(dispatch.level, i);
    planes[at] = lay_over(planes[at], local, alpha, word(r + 5u, 3u), data);
}

/// Starts a group above `dispatch.level`: transparent, or (record 1) a
/// copy of the plane beneath for a pass-through group.
@compute @workgroup_size(256)
fn begin_group(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= g.count {
        return;
    }
    var start = vec4<f32>(0.0);
    if dispatch.record == 1u {
        start = planes[plane_at(dispatch.level, i)];
    }
    planes[plane_at(dispatch.level + 1u, i)] = start;
}

/// Finishes the group above `dispatch.level`.
@compute @workgroup_size(256)
fn end_group(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= g.count {
        return;
    }
    let r = record(dispatch.record);
    let data = dispatch.data;
    let x = i % g.width;
    let y = i / g.width;
    let uv = pixel_uv(x, y);
    let s = maps_sample(uv);
    let local = apply_filters(word(r + 3u, 0u), word(r + 3u, 1u), planes[plane_at(dispatch.level + 1u, i)], data);
    let at = plane_at(dispatch.level, i);
    let destination = planes[at];
    let mask = mask_value(r, i, uv, s);
    let opacity = row(r + 6u).x;
    if word(r + 6u, 1u) != 0u {
        // Compositor::pass_through.
        let amount = clamp(opacity * mask, 0.0, 1.0);
        let alpha = destination.w + (local.w - destination.w) * amount;
        var color = destination.xyz;
        if alpha > 0.0 {
            color = (destination.xyz * destination.w * (1.0 - amount) + local.xyz * local.w * amount) / alpha;
        }
        if data == DATA_VECTOR {
            color = normalize_or(color, vec3<f32>(0.0, 0.0, 1.0));
        }
        planes[at] = vec4<f32>(color, alpha);
        return;
    }
    if local.w <= 0.0 {
        return;
    }
    planes[at] = lay_over(destination, local, local.w * opacity * mask, word(r + 5u, 3u), data);
}

fn set_byte(lane: u32, i: u32, shift: u32, value: u32) {
    var packed = outputs[i];
    packed[lane] = (packed[lane] & ~(0xffu << shift)) | (value << shift);
    outputs[i] = packed;
}

fn rgb_bytes(rgb: vec3<f32>, alpha: u32) -> u32 {
    return unit_to_byte(rgb.x) | (unit_to_byte(rgb.y) << 8u) | (unit_to_byte(rgb.z) << 16u) | (alpha << 24u);
}

/// What texels no layer paints hold: default Base Color and Opacity, ORM
/// with baked occlusion, black emission, and a flat normal.
@compute @workgroup_size(256)
fn defaults(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= g.count {
        return;
    }
    let constants = row(dispatch.record);
    var occlusion = 1.0;
    if g.has_maps != 0u {
        let s = maps_sample(pixel_uv(i % g.width, i / g.width));
        if s.covered {
            occlusion = s.ambient_occlusion;
        }
    }
    let more = row(dispatch.record + 1u);
    let orm = unit_to_byte(occlusion) | (unit_to_byte(constants.w) << 8u)
        | (unit_to_byte(more.x) << 16u) | (255u << 24u);
    let base_color = rgb_bytes(constants.xyz, unit_to_byte(more.y));
    let normal = rgb_bytes(vec3<f32>(0.5, 0.5, 1.0), 255u);
    outputs[i] = vec4<u32>(base_color, orm, normal, 255u << 24u);
}

/// Writes the composited channel `dispatch.channel` (plane 0) into the
/// outputs: Base Color 0, Roughness 1, Metallic 2, Emissive 3, Opacity 4,
/// Height 5 (kept for the Normal channel), Normal 6.
@compute @workgroup_size(256)
fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= g.count {
        return;
    }
    let texel = planes[plane_at(0u, i)];
    switch dispatch.channel {
        case 0u: {
            let default_base = row(dispatch.record).xyz;
            let color = texel.xyz * texel.w + default_base * (1.0 - texel.w);
            set_byte(0u, i, 0u, unit_to_byte(color.x));
            set_byte(0u, i, 8u, unit_to_byte(color.y));
            set_byte(0u, i, 16u, unit_to_byte(color.z));
        }
        case 1u: { set_byte(1u, i, 8u, unit_to_byte(texel.x)); }
        case 2u: { set_byte(1u, i, 16u, unit_to_byte(texel.x)); }
        case 3u: { outputs[i].w = rgb_bytes(texel.xyz, 255u); }
        case 4u: { set_byte(0u, i, 24u, unit_to_byte(texel.x)); }
        case 5u: { planes[g.levels * g.count + i] = texel; }
        default: { outputs[i].z = rgb_bytes(texel.xyz * 0.5 + 0.5, 255u); }
    }
}

fn height_at(x: u32, y: u32) -> f32 {
    return planes[g.levels * g.count + y * g.width + x].x;
}

/// TextureSet::normal_region's height detail, over plane 0.
@compute @workgroup_size(256)
fn normal_detail(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= g.count {
        return;
    }
    let x = i % g.width;
    let y = i / g.width;
    let last_x = g.width - 1u;
    let last_y = g.height - 1u;
    let du = (height_at(min(x + 1u, last_x), y) - height_at(select(x - 1u, 0u, x == 0u), y)) * 0.5;
    let dv = (height_at(x, select(y - 1u, 0u, y == 0u)) - height_at(x, min(y + 1u, last_y))) * 0.5;
    if du == 0.0 && dv == 0.0 {
        return;
    }
    let scale = g.height_scale;
    let detail = normalize_or(vec3<f32>(-du * scale, -dv * scale, 1.0), vec3<f32>(0.0, 0.0, 1.0));
    let base = planes[plane_at(0u, i)].xyz;
    planes[plane_at(0u, i)] = vec4<f32>(
        normalize_or(vec3<f32>(base.x + detail.x, base.y + detail.y, base.z * detail.z), vec3<f32>(0.0, 0.0, 1.0)),
        1.0,
    );
}
