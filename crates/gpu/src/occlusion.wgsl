// Ambient-occlusion and thickness rays for mesh-map bakes, one invocation
// per query. A port of `substancecraft_core::occlusion` (`trace_query`,
// `sample_direction`, and the BVH traversal): it casts the same rays through
// the same hierarchy, so results match the CPU tracer within floating-point
// tolerance. Only the order nodes are visited in differs (see `traverse`).

struct Node {
    min: vec3<f32>,
    first: u32,
    max: vec3<f32>,
    count: u32,
};

// Triangle corners in leaf order (`Bvh::order` applied).
struct Triangle {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
};

// `point.w` holds the seed's bits.
struct Query {
    point: vec4<f32>,
    normal: vec4<f32>,
    face_normal: vec4<f32>,
};

struct Params {
    ao_samples: u32,
    thickness_samples: u32,
    ao_distance: f32,
    thickness_distance: f32,
    bias: f32,
    // First query of this dispatch, and how many it traces.
    offset: u32,
    count: u32,
    thickness_seed: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

const STACK_DEPTH: u32 = 64u;
const MISS: f32 = -1.0;
const NO_ENTRY: f32 = 3.0e38;
const TAU: f32 = 6.2831855;

fn lowbias32(value: u32) -> u32 {
    var v = value;
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

fn unit_float(hash: u32) -> f32 {
    return f32(hash >> 8u) / 16777216.0;
}

fn radical_inverse(bits: u32) -> f32 {
    return f32(reverseBits(bits)) * (1.0 / 4294967296.0);
}

// glam's `normalize_or`.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(v, v);
    if length_squared > 0.0 {
        return v * (1.0 / sqrt(length_squared));
    }
    return fallback;
}

// `f32::signum`: the sign bit decides, so -0 is negative.
fn signum(value: f32) -> f32 {
    return select(1.0, -1.0, (bitcast<u32>(value) & 0x80000000u) != 0u);
}

fn sample_direction(
    normal: vec3<f32>,
    face_normal: vec3<f32>,
    index: u32,
    count: u32,
    seed: u32,
) -> vec3<f32> {
    let jitter = vec2<f32>(unit_float(seed), unit_float(lowbias32(seed)));
    let u = fract((f32(index) + 0.5) / f32(count) + jitter.x);
    let v = fract(radical_inverse(index) + jitter.y);
    let radius = sqrt(u);
    let angle = TAU * v;
    // glam's `any_orthonormal_pair` (Duff et al.).
    let sign = signum(normal.z);
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = vec3<f32>(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    let bitangent = vec3<f32>(b, sign + normal.y * normal.y * a, -normal.y);
    let direction = normalize_or(
        tangent * (radius * cos(angle))
            + bitangent * (radius * sin(angle))
            + normal * sqrt(max(1.0 - u, 0.0)),
        normal,
    );
    let below = dot(direction, face_normal);
    if below < 0.0 {
        return normalize_or(direction - face_normal * (2.0 * below), normal);
    }
    return direction;
}

// Reciprocal kept finite: WGSL does not promise infinities.
fn safe_recip(direction: vec3<f32>) -> vec3<f32> {
    let tiny = 1e-20;
    let sign = select(vec3<f32>(1.0), vec3<f32>(-1.0), direction < vec3<f32>(0.0));
    let safe = select(direction, sign * tiny, abs(direction) < vec3<f32>(tiny));
    return 1.0 / safe;
}

// Where the ray enters the box within (t_min, t_max), or NO_ENTRY.
fn box_entry(
    origin: vec3<f32>,
    inverse: vec3<f32>,
    lo: vec3<f32>,
    hi: vec3<f32>,
    t_min: f32,
    t_max: f32,
) -> f32 {
    let first = (lo - origin) * inverse;
    let second = (hi - origin) * inverse;
    let near_planes = min(first, second);
    let far_planes = max(first, second);
    let near = max(max(max(near_planes.x, near_planes.y), near_planes.z), t_min);
    let far = min(min(min(far_planes.x, far_planes.y), far_planes.z), t_max);
    return select(NO_ENTRY, near, near <= far);
}

// Two-sided Möller–Trumbore; the hit distance, or MISS.
fn ray_triangle(
    origin: vec3<f32>,
    direction: vec3<f32>,
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
) -> f32 {
    let edge_ab = b - a;
    let edge_ac = c - a;
    let p = cross(direction, edge_ac);
    let determinant = dot(edge_ab, p);
    if abs(determinant) < 1e-12 {
        return MISS;
    }
    let inverse = 1.0 / determinant;
    let offset = origin - a;
    let u = dot(offset, p) * inverse;
    if u < 0.0 || u > 1.0 {
        return MISS;
    }
    let q = cross(offset, edge_ab);
    let v = dot(direction, q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return MISS;
    }
    return dot(edge_ac, q) * inverse;
}

// The nearest hit in (t_min, t_max), or with `first_hit` the first found;
// MISS when there is none. Unlike the CPU traversal, both children are
// tested together and the nearer is entered first, with only the farther
// one stacked: the same hits, found sooner. Whether a ray hits, and the
// nearest hit, do not depend on the order nodes are visited in.
fn traverse(
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_min: f32,
    t_max_in: f32,
    first_hit: bool,
) -> f32 {
    var stack: array<u32, STACK_DEPTH>;
    let inverse = safe_recip(direction);
    var t_max = t_max_in;
    var closest = MISS;
    let root = nodes[0];
    if box_entry(origin, inverse, root.min, root.max, t_min, t_max) == NO_ENTRY {
        return MISS;
    }
    var current = 0u;
    var depth = 0u;
    loop {
        let node = nodes[current];
        if node.count > 0u {
            for (var index = node.first; index < node.first + node.count; index += 1u) {
                let triangle = triangles[index];
                let distance = ray_triangle(
                    origin,
                    direction,
                    triangle.a.xyz,
                    triangle.b.xyz,
                    triangle.c.xyz,
                );
                if distance > t_min && distance < t_max {
                    if first_hit {
                        return distance;
                    }
                    t_max = distance;
                    closest = distance;
                }
            }
        } else {
            let left = node.first;
            let right = node.first + 1u;
            let left_node = nodes[left];
            let right_node = nodes[right];
            let enter_left = box_entry(origin, inverse, left_node.min, left_node.max, t_min, t_max);
            let enter_right =
                box_entry(origin, inverse, right_node.min, right_node.max, t_min, t_max);
            let hit_left = enter_left != NO_ENTRY;
            let hit_right = enter_right != NO_ENTRY;
            if hit_left && hit_right {
                let left_first = enter_left <= enter_right;
                if depth < STACK_DEPTH {
                    stack[depth] = select(left, right, left_first);
                    depth += 1u;
                }
                current = select(right, left, left_first);
                continue;
            }
            if hit_left || hit_right {
                current = select(right, left, hit_left);
                continue;
            }
        }
        // Back to the nearest node still waiting. Stacked nodes may lie
        // beyond a hit found since; their triangles are bounded by t_max.
        if depth == 0u {
            break;
        }
        depth -= 1u;
        current = stack[depth];
    }
    return closest;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.count {
        return;
    }
    let index = params.offset + id.x;
    let query = queries[index];
    let point = query.point.xyz;
    let seed = bitcast<u32>(query.point.w);
    let normal = query.normal.xyz;
    let face_normal = query.face_normal.xyz;
    let bias = params.bias;

    let origin = point + face_normal * bias;
    var occluded = 0u;
    for (var sample = 0u; sample < params.ao_samples; sample += 1u) {
        let direction = sample_direction(normal, face_normal, sample, params.ao_samples, seed);
        if traverse(origin, direction, bias, params.ao_distance, true) != MISS {
            occluded += 1u;
        }
    }
    let ambient_occlusion = 1.0 - f32(occluded) / f32(max(params.ao_samples, 1u));

    let inside = point - face_normal * bias;
    let distance = params.thickness_distance;
    let thickness_seed = lowbias32(seed ^ params.thickness_seed);
    var total = 0.0;
    for (var sample = 0u; sample < params.thickness_samples; sample += 1u) {
        let direction = sample_direction(
            -normal,
            -face_normal,
            sample,
            params.thickness_samples,
            thickness_seed,
        );
        let hit = traverse(inside, direction, bias, distance, false);
        total += select(distance, hit, hit != MISS);
    }
    let thickness = clamp(
        total / f32(max(params.thickness_samples, 1u)) / max(distance, 1.1920929e-7),
        0.0,
        1.0,
    );
    results[index] = vec2<f32>(ambient_occlusion, thickness);
}
