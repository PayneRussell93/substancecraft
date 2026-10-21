// Mesh-map ray queries on the GPU (PERF-01). Mirrors the CPU tracer in
// crates/core/src/mesh_maps.rs: the same BVH (built on the CPU and
// uploaded as is), the same two-sided Moller-Trumbore test, and the same
// cosine-weighted Hammersley sampling, so results differ from the CPU only
// by floating-point precision.

struct Node {
    min: vec3<f32>,
    first: u32,
    max: vec3<f32>,
    count: u32,
};

struct Params {
    seed: u32,
    ao_samples: u32,
    ao_distance: f32,
    thickness_samples: u32,
    thickness_distance: f32,
    bias: f32,
    // First query of this dispatch, and how many it answers.
    base: u32,
    count: u32,
};

struct Query {
    point: vec3<f32>,
    texel: u32,
    normal: vec3<f32>,
    pad0: u32,
    face_normal: vec3<f32>,
    pad1: u32,
};

struct Ray {
    origin: vec3<f32>,
    t_max: f32,
    direction: vec3<f32>,
    pad: u32,
};

struct Hit {
    // 0xffffffff when nothing was hit.
    triangle: u32,
    u: f32,
    v: f32,
    distance: f32,
};

const NO_HIT: u32 = 0xffffffffu;
// STACK_SIZE is defined by the host: deep enough for the scene's BVH.

@group(0) @binding(0) var<storage, read> nodes: array<Node>;
// Three corners per triangle in leaf order; each first corner's w holds
// the triangle's index (as bits).
@group(0) @binding(1) var<storage, read> triangles: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> queries: array<Query>;
@group(0) @binding(4) var<storage, read_write> occlusion_out: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> rays: array<Ray>;
@group(0) @binding(6) var<storage, read_write> hits_out: array<Hit>;

fn safe_inverse(direction: vec3<f32>) -> vec3<f32> {
    // The CPU divides by zero to infinity; a huge value acts the same in
    // the slab test without risking NaN.
    let tiny = abs(direction) < vec3<f32>(1e-30);
    let huge = select(vec3<f32>(1e30), vec3<f32>(-1e30), direction < vec3<f32>(0.0));
    return select(1.0 / direction, huge, tiny);
}

// x: 1 when hit, y: distance, z: u, w: v.
fn ray_triangle(origin: vec3<f32>, direction: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec4<f32> {
    let edge_ab = b - a;
    let edge_ac = c - a;
    let p = cross(direction, edge_ac);
    let determinant = dot(edge_ab, p);
    if abs(determinant) < 1e-12 {
        return vec4<f32>(0.0);
    }
    let inverse = 1.0 / determinant;
    let offset = origin - a;
    let u = dot(offset, p) * inverse;
    if u < 0.0 || u > 1.0 {
        return vec4<f32>(0.0);
    }
    let q = cross(offset, edge_ab);
    let v = dot(direction, q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return vec4<f32>(0.0);
    }
    let distance = dot(edge_ac, q) * inverse;
    if abs(distance) > 3.0e38 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(1.0, distance, u, v);
}

// Entry distance of the ray into a box, or a negative value when it misses
// the box within t_min..t_max.
fn box_entry(origin: vec3<f32>, inverse: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
    let first = (lo - origin) * inverse;
    let second = (hi - origin) * inverse;
    let near_planes = min(first, second);
    let far_planes = max(first, second);
    let near = max(max(max(near_planes.x, near_planes.y), near_planes.z), t_min);
    let far = min(min(min(far_planes.x, far_planes.y), far_planes.z), t_max);
    return select(-1.0, near, near <= far);
}

fn trace(origin: vec3<f32>, direction: vec3<f32>, t_min: f32, t_max_in: f32, any_hit: bool) -> Hit {
    var best = Hit(NO_HIT, 0.0, 0.0, t_max_in);
    let inverse = safe_inverse(direction);
    var t_max = t_max_in;
    let root = nodes[0];
    if box_entry(origin, inverse, root.min, root.max, t_min, t_max) < 0.0 {
        return best;
    }
    // Nodes on the stack have been found to meet the ray; children are
    // tested before they are pushed, the nearer pushed last so it is
    // visited first. Which hit wins does not depend on the order.
    var stack: array<u32, STACK_SIZE>;
    stack[0] = 0u;
    var depth = 1u;
    loop {
        if depth == 0u {
            break;
        }
        depth -= 1u;
        let node = nodes[stack[depth]];
        if node.count > 0u {
            for (var i = 0u; i < node.count; i += 1u) {
                let base = (node.first + i) * 3u;
                let a = triangles[base];
                let found = ray_triangle(origin, direction, a.xyz, triangles[base + 1u].xyz, triangles[base + 2u].xyz);
                if found.x > 0.5 && found.y > t_min && found.y < t_max {
                    best = Hit(bitcast<u32>(a.w), found.z, found.w, found.y);
                    if any_hit {
                        return best;
                    }
                    t_max = found.y;
                }
            }
            continue;
        }
        let left = nodes[node.first];
        let right = nodes[node.first + 1u];
        let left_entry = box_entry(origin, inverse, left.min, left.max, t_min, t_max);
        let right_entry = box_entry(origin, inverse, right.min, right.max, t_min, t_max);
        let left_hit = left_entry >= 0.0;
        let right_hit = right_entry >= 0.0;
        if left_hit && right_hit {
            let near_left = left_entry <= right_entry;
            stack[depth] = select(node.first, node.first + 1u, near_left);
            stack[depth + 1u] = select(node.first + 1u, node.first, near_left);
            depth += 2u;
        } else if left_hit {
            stack[depth] = node.first;
            depth += 1u;
        } else if right_hit {
            stack[depth] = node.first + 1u;
            depth += 1u;
        }
    }
    return best;
}

fn lowbias32(input: u32) -> u32 {
    var value = input;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

fn unit_float(hash: u32) -> f32 {
    return f32(hash >> 8u) / 16777216.0;
}

fn radical_inverse(bits: u32) -> f32 {
    return f32(reverseBits(bits)) * (1.0 / 4294967296.0);
}

// Rust's f32::fract for the non-negative values used here.
fn fraction(value: f32) -> f32 {
    return value - trunc(value);
}

// glam's normalize_or.
fn normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let reciprocal = 1.0 / sqrt(dot(value, value));
    if reciprocal > 0.0 && reciprocal < 3.0e38 {
        return value * reciprocal;
    }
    return fallback;
}

fn cosine_direction(normal: vec3<f32>, face_normal: vec3<f32>, index: u32, count: u32, seed: u32) -> vec3<f32> {
    let jitter = vec2<f32>(unit_float(seed), unit_float(lowbias32(seed)));
    let u = fraction((f32(index) + 0.5) / f32(count) + jitter.x);
    let v = fraction(radical_inverse(index) + jitter.y);
    let radius = sqrt(u);
    let angle = 6.28318530717958647692 * v;
    // glam's any_orthonormal_pair; the sign follows the sign bit, as
    // f32::signum does for -0.0.
    let sign = select(1.0, -1.0, (bitcast<u32>(normal.z) & 0x80000000u) != 0u);
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = vec3<f32>(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    let bitangent = vec3<f32>(b, sign + normal.y * normal.y * a, -normal.y);
    let direction = normalize_or(
        tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle)) + normal * sqrt(max(1.0 - u, 0.0)),
        normal,
    );
    let below = dot(direction, face_normal);
    if below < 0.0 {
        return normalize_or(direction - face_normal * (2.0 * below), normal);
    }
    return direction;
}

@compute @workgroup_size(64)
fn occlusion(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.count {
        return;
    }
    let index = params.base + id.x;
    let query = queries[index];
    let bias = params.bias;
    let seed = lowbias32(query.texel ^ (params.seed * 0x9e3779b9u));
    let origin = query.point + query.face_normal * bias;
    var occluded = 0u;
    for (var sample = 0u; sample < params.ao_samples; sample += 1u) {
        let direction = cosine_direction(query.normal, query.face_normal, sample, params.ao_samples, seed);
        if trace(origin, direction, bias, params.ao_distance, true).triangle != NO_HIT {
            occluded += 1u;
        }
    }
    let ambient_occlusion = 1.0 - f32(occluded) / f32(params.ao_samples);

    let inside = query.point - query.face_normal * bias;
    let thickness_seed = lowbias32(seed ^ 0x5bd1e995u);
    var total = 0.0;
    for (var sample = 0u; sample < params.thickness_samples; sample += 1u) {
        let direction = cosine_direction(-query.normal, -query.face_normal, sample, params.thickness_samples, thickness_seed);
        let hit = trace(inside, direction, bias, params.thickness_distance, false);
        total += select(params.thickness_distance, hit.distance, hit.triangle != NO_HIT);
    }
    let thickness = clamp(total / f32(params.thickness_samples) / params.thickness_distance, 0.0, 1.0);
    occlusion_out[index] = vec2<f32>(ambient_occlusion, thickness);
}

@compute @workgroup_size(64)
fn cage_hits(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.count {
        return;
    }
    let index = params.base + id.x;
    let ray = rays[index];
    hits_out[index] = trace(ray.origin, ray.direction, 0.0, ray.t_max, false);
}
