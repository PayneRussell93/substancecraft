//! Reading OpenColorIO configs directly: displays, views, colour spaces,
//! view transforms, and looks resolve into a chain of colour operations
//! that is baked into the display LUT the viewport and renders apply.
//!
//! Supported transforms: Matrix, Exponent, ExponentWithLinear, Log,
//! LogAffine, LogCamera, CDL, Range, Allocation, File (`.cube`, `.spi1d`,
//! `.spi3d`, `.spimtx`), Group, ColorSpace, DisplayView, Look, and the
//! Builtin styles listed in [`builtin`]. Anything else is reported by name,
//! so a view that cannot be read says why (and can still be baked with
//! `ociobakelut` and loaded as a `.cube`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::Vec3;
use substancecraft_core::{CubeLut, MAX_CUBE_LUT_SIZE, color::srgb_to_linear};
use thiserror::Error;

use crate::ocio_yaml::{self, Node};

/// Largest config file read.
pub const MAX_OCIO_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
/// Largest LUT file a config may reference.
pub const MAX_OCIO_LUT_BYTES: u64 = 256 * 1024 * 1024;
/// Stops of scene light above display white the baked view keeps: the
/// renderer exposes this much darker and the LUT brightens it back, so
/// highlights reach the view's tone curve instead of clipping first.
pub const OCIO_HEADROOM_STOPS: f32 = 2.0;
/// Edge of baked view LUTs.
pub const OCIO_BAKE_SIZE: usize = 65;
/// Nesting of groups, colour-space references, and looks followed.
const MAX_DEPTH: usize = 16;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum OcioError {
    #[error("{0}")]
    Io(String),
    #[error("{0}")]
    Parse(String),
    #[error("the config has no colour space or role named `{0}`")]
    UnknownColorSpace(String),
    #[error("the config has no display `{0}`")]
    UnknownDisplay(String),
    #[error("display `{display}` has no view `{view}`")]
    UnknownView { display: String, view: String },
    #[error("the config has no view transform `{0}`")]
    UnknownViewTransform(String),
    #[error("the config has no look `{0}`")]
    UnknownLook(String),
    #[error("{0} is not supported")]
    Unsupported(String),
    #[error("transforms nest too deeply (a colour space may refer to itself)")]
    TooDeep,
}

fn parse_error(message: impl Into<String>) -> OcioError {
    OcioError::Parse(message.into())
}

// ---------------------------------------------------------------------------
// Operations

/// One step of a processor, on RGB in double precision.
#[derive(Clone, Debug)]
pub(crate) enum Op {
    Matrix {
        m: [[f64; 3]; 3],
        offset: [f64; 3],
    },
    /// `sign(x)·|x|^g` when mirrored, else `max(x, 0)^g`.
    Power {
        gamma: [f64; 3],
        mirror: bool,
    },
    /// ExponentWithLinear (monitor curve) forward and inverse.
    MonCurve {
        gamma: [f64; 3],
        offset: [f64; 3],
        inverse: bool,
    },
    /// LogAffine / LogCamera: lin to log (`inverse` = log to lin).
    Log {
        base: f64,
        log_slope: [f64; 3],
        log_offset: [f64; 3],
        lin_slope: [f64; 3],
        lin_offset: [f64; 3],
        /// LogCamera's linear segment below this break.
        lin_break: Option<[f64; 3]>,
        linear_slope: Option<[f64; 3]>,
        inverse: bool,
    },
    Cdl {
        slope: [f64; 3],
        offset: [f64; 3],
        power: [f64; 3],
        saturation: f64,
        clamp: bool,
        inverse: bool,
    },
    Clamp {
        min: Option<f64>,
        max: Option<f64>,
    },
    Lut1d {
        table: Arc<Vec<[f64; 3]>>,
        domain: (f64, f64),
        inverse: bool,
    },
    Lut3d {
        size: usize,
        /// Red fastest, then green, then blue.
        data: Arc<Vec<[f64; 3]>>,
        domain_min: [f64; 3],
        domain_max: [f64; 3],
    },
    /// An ACES 1.0 output transform (RRT and SDR ODT) from ACES2065-1 to
    /// CIE XYZ (D65), normalized to display white.
    AcesOutput(builtin::AcesOutput),
}

fn mat_mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat_inverse(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

pub(crate) fn mat_product(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (row, out_row) in out.iter_mut().enumerate() {
        for (col, value) in out_row.iter_mut().enumerate() {
            *value = (0..3).map(|k| a[row][k] * b[k][col]).sum();
        }
    }
    out
}

fn interpolate_1d(table: &[[f64; 3]], domain: (f64, f64), channel: usize, x: f64) -> f64 {
    let last = (table.len() - 1) as f64;
    let t = ((x - domain.0) / (domain.1 - domain.0)).clamp(0.0, 1.0) * last;
    let low = (t.floor() as usize).min(table.len() - 2);
    let f = t - low as f64;
    table[low][channel] * (1.0 - f) + table[low + 1][channel] * f
}

/// Inverse of a monotonic 1D table by bisection over its samples.
fn invert_1d(table: &[[f64; 3]], domain: (f64, f64), channel: usize, y: f64) -> f64 {
    let first = table[0][channel];
    let last_value = table[table.len() - 1][channel];
    let increasing = last_value >= first;
    let (low_value, high_value) = if increasing {
        (first, last_value)
    } else {
        (last_value, first)
    };
    let y = y.clamp(low_value, high_value);
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    for _ in 0..48 {
        let mid = (low + high) * 0.5;
        let x = domain.0 + mid * (domain.1 - domain.0);
        let value = interpolate_1d(table, domain, channel, x);
        if (value < y) == increasing {
            low = mid;
        } else {
            high = mid;
        }
    }
    domain.0 + (low + high) * 0.5 * (domain.1 - domain.0)
}

impl Op {
    pub(crate) fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        match self {
            Self::Matrix { m, offset } => {
                let v = mat_mul(m, rgb);
                [v[0] + offset[0], v[1] + offset[1], v[2] + offset[2]]
            }
            Self::Power { gamma, mirror } => std::array::from_fn(|i| {
                if *mirror {
                    rgb[i].signum() * rgb[i].abs().powf(gamma[i])
                } else {
                    rgb[i].max(0.0).powf(gamma[i])
                }
            }),
            Self::MonCurve {
                gamma,
                offset,
                inverse,
            } => std::array::from_fn(|i| {
                let (g, o) = (gamma[i], offset[i]);
                let x = rgb[i];
                if !*inverse {
                    // Encoded value to linear (OCIO's "forward" moncurve).
                    let break_point = o / (g - 1.0);
                    let slope = ((g - 1.0) / o) * ((o * g) / ((g - 1.0) * (1.0 + o))).powf(g);
                    if x > break_point {
                        ((x + o) / (1.0 + o)).powf(g)
                    } else {
                        x * slope
                    }
                } else {
                    let break_point = (o * g / ((g - 1.0) * (1.0 + o))).powf(g);
                    let slope = ((g - 1.0) / o) * ((o * g) / ((g - 1.0) * (1.0 + o))).powf(g);
                    if x > break_point {
                        (1.0 + o) * x.powf(1.0 / g) - o
                    } else {
                        x / slope
                    }
                }
            }),
            Self::Log {
                base,
                log_slope,
                log_offset,
                lin_slope,
                lin_offset,
                lin_break,
                linear_slope,
                inverse,
            } => std::array::from_fn(|i| {
                let log_of = |x: f64| x.max(f64::MIN_POSITIVE).ln() / base.ln();
                let linear_segment = lin_break.map(|lin_break| {
                    let lin_break = lin_break[i];
                    let log_break = log_slope[i] * log_of(lin_slope[i] * lin_break + lin_offset[i])
                        + log_offset[i];
                    let slope = linear_slope.map_or_else(
                        || {
                            log_slope[i] * lin_slope[i]
                                / ((lin_slope[i] * lin_break + lin_offset[i]) * base.ln())
                        },
                        |slopes| slopes[i],
                    );
                    let offset = log_break - slope * lin_break;
                    (lin_break, log_break, slope, offset)
                });
                let x = rgb[i];
                if !*inverse {
                    match linear_segment {
                        Some((lin_break, _, slope, offset)) if x <= lin_break => x * slope + offset,
                        _ => log_slope[i] * log_of(lin_slope[i] * x + lin_offset[i]) + log_offset[i],
                    }
                } else {
                    match linear_segment {
                        Some((_, log_break, slope, offset)) if x <= log_break => (x - offset) / slope,
                        _ => {
                            (base.powf((x - log_offset[i]) / log_slope[i]) - lin_offset[i])
                                / lin_slope[i]
                        }
                    }
                }
            }),
            Self::Cdl {
                slope,
                offset,
                power,
                saturation,
                clamp,
                inverse,
            } => {
                const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];
                let saturate = |v: [f64; 3], s: f64| -> [f64; 3] {
                    let luma: f64 = (0..3).map(|i| v[i] * LUMA[i]).sum();
                    std::array::from_fn(|i| luma + s * (v[i] - luma))
                };
                let clamp01 = |x: f64| if *clamp { x.clamp(0.0, 1.0) } else { x };
                if !*inverse {
                    let v: [f64; 3] = std::array::from_fn(|i| {
                        let x = clamp01(rgb[i] * slope[i] + offset[i]);
                        if x > 0.0 { x.powf(power[i]) } else { x }
                    });
                    let v = saturate(v, *saturation);
                    std::array::from_fn(|i| clamp01(v[i]))
                } else {
                    let v: [f64; 3] = std::array::from_fn(|i| clamp01(rgb[i]));
                    let v = saturate(v, 1.0 / saturation.max(1e-6));
                    std::array::from_fn(|i| {
                        let x = clamp01(v[i]);
                        let x = if x > 0.0 { x.powf(1.0 / power[i]) } else { x };
                        clamp01((x - offset[i]) / slope[i])
                    })
                }
            }
            Self::Clamp { min, max } => std::array::from_fn(|i| {
                let mut x = rgb[i];
                if let Some(min) = min {
                    x = x.max(*min);
                }
                if let Some(max) = max {
                    x = x.min(*max);
                }
                x
            }),
            Self::Lut1d {
                table,
                domain,
                inverse,
            } => std::array::from_fn(|i| {
                if *inverse {
                    invert_1d(table, *domain, i, rgb[i])
                } else {
                    interpolate_1d(table, *domain, i, rgb[i])
                }
            }),
            Self::Lut3d {
                size,
                data,
                domain_min,
                domain_max,
            } => {
                let last = (*size - 1) as f64;
                let at: [f64; 3] = std::array::from_fn(|i| {
                    ((rgb[i] - domain_min[i]) / (domain_max[i] - domain_min[i])).clamp(0.0, 1.0)
                        * last
                });
                let low: [usize; 3] =
                    std::array::from_fn(|i| (at[i].floor() as usize).min(*size - 2));
                let t: [f64; 3] = std::array::from_fn(|i| at[i] - low[i] as f64);
                let index = |r: usize, g: usize, b: usize| data[r + size * (g + size * b)];
                let mut out = [0.0; 3];
                for (db, wb) in [(0, 1.0 - t[2]), (1, t[2])] {
                    for (dg, wg) in [(0, 1.0 - t[1]), (1, t[1])] {
                        for (dr, wr) in [(0, 1.0 - t[0]), (1, t[0])] {
                            let entry = index(low[0] + dr, low[1] + dg, low[2] + db);
                            let weight = wr * wg * wb;
                            for c in 0..3 {
                                out[c] += entry[c] * weight;
                            }
                        }
                    }
                }
                out
            }
            Self::AcesOutput(output) => output.apply(rgb),
        }
    }

    fn inverse(&self) -> Result<Self, OcioError> {
        Ok(match self.clone() {
            Self::Matrix { m, offset } => {
                let inv = mat_inverse(&m)
                    .ok_or_else(|| OcioError::Unsupported("inverting a singular matrix".into()))?;
                let shifted = mat_mul(&inv, offset);
                Self::Matrix {
                    m: inv,
                    offset: [-shifted[0], -shifted[1], -shifted[2]],
                }
            }
            Self::Power { gamma, mirror } => Self::Power {
                gamma: gamma.map(|g| 1.0 / g),
                mirror,
            },
            Self::MonCurve {
                gamma,
                offset,
                inverse,
            } => Self::MonCurve {
                gamma,
                offset,
                inverse: !inverse,
            },
            Self::Log {
                base,
                log_slope,
                log_offset,
                lin_slope,
                lin_offset,
                lin_break,
                linear_slope,
                inverse,
            } => Self::Log {
                base,
                log_slope,
                log_offset,
                lin_slope,
                lin_offset,
                lin_break,
                linear_slope,
                inverse: !inverse,
            },
            Self::Cdl {
                slope,
                offset,
                power,
                saturation,
                clamp,
                inverse,
            } => Self::Cdl {
                slope,
                offset,
                power,
                saturation,
                clamp,
                inverse: !inverse,
            },
            Self::Clamp { .. } => self.clone(),
            Self::Lut1d {
                table,
                domain,
                inverse,
            } => Self::Lut1d {
                table,
                domain,
                inverse: !inverse,
            },
            Self::Lut3d { .. } => {
                return Err(OcioError::Unsupported("inverting a 3D LUT".into()));
            }
            Self::AcesOutput(_) => {
                return Err(OcioError::Unsupported(
                    "inverting an ACES output transform".into(),
                ));
            }
        })
    }
}

fn invert_ops(ops: Vec<Op>) -> Result<Vec<Op>, OcioError> {
    ops.iter().rev().map(Op::inverse).collect()
}

/// A compiled chain of operations.
#[derive(Clone, Debug, Default)]
pub struct OcioProcessor {
    ops: Vec<Op>,
}

impl OcioProcessor {
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        self.ops.iter().fold(rgb, |value, op| op.apply(value))
    }

    pub fn is_identity(&self) -> bool {
        self.ops.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Config

#[derive(Clone, Debug, PartialEq)]
pub struct OcioView {
    pub name: String,
    colorspace: Option<String>,
    view_transform: Option<String>,
    display_colorspace: Option<String>,
    looks: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OcioDisplay {
    pub name: String,
    pub views: Vec<OcioView>,
}

#[derive(Clone, Debug)]
struct ColorSpaceDef {
    name: String,
    aliases: Vec<String>,
    is_data: bool,
    display_referred: bool,
    to_reference: Option<Node>,
    from_reference: Option<Node>,
}

#[derive(Clone, Debug)]
struct ViewTransformDef {
    name: String,
    from_scene: Option<Node>,
    to_scene: Option<Node>,
}

#[derive(Clone, Debug)]
struct LookDef {
    name: String,
    process_space: String,
    transform: Option<Node>,
    inverse_transform: Option<Node>,
}

/// A parsed OpenColorIO config.
#[derive(Clone, Debug)]
pub struct OcioConfig {
    path: PathBuf,
    search_paths: Vec<PathBuf>,
    roles: BTreeMap<String, String>,
    displays: Vec<OcioDisplay>,
    active_displays: Vec<String>,
    active_views: Vec<String>,
    colorspaces: Vec<ColorSpaceDef>,
    view_transforms: Vec<ViewTransformDef>,
    default_view_transform: Option<String>,
    looks: Vec<LookDef>,
}

/// A transform node, honouring `direction: inverse`.
fn direction_inverse(node: &Node) -> bool {
    node.get_str("direction")
        .is_some_and(|direction| direction.trim().eq_ignore_ascii_case("inverse"))
}

fn rgb3(values: &[f64]) -> Option<[f64; 3]> {
    match values {
        [v] => Some([*v; 3]),
        [r, g, b] | [r, g, b, _] => Some([*r, *g, *b]),
        _ => None,
    }
}

fn names_match(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

impl OcioConfig {
    /// Reads the config at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, OcioError> {
        let path = path.as_ref();
        let size = std::fs::metadata(path)
            .map_err(|error| OcioError::Io(format!("{}: {error}", path.display())))?
            .len();
        if size > MAX_OCIO_CONFIG_BYTES {
            return Err(OcioError::Io(format!("{}: the file is too large", path.display())));
        }
        let text = std::fs::read_to_string(path)
            .map_err(|error| OcioError::Io(format!("{}: {error}", path.display())))?;
        Self::parse(&text, path)
    }

    /// Parses config text; `path` locates LUT files.
    pub fn parse(text: &str, path: &Path) -> Result<Self, OcioError> {
        let root = ocio_yaml::parse(text).map_err(OcioError::Parse)?;
        if root.get("ocio_profile_version").is_none() {
            return Err(parse_error("not an OCIO config: no ocio_profile_version"));
        }
        let folder = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let working_dir = root
            .get_str("working_dir")
            .map(|dir| folder.join(dir))
            .unwrap_or_else(|| folder.clone());
        let mut search_paths = Vec::new();
        match root.get("search_path") {
            Some(node) if !node.seq().is_empty() => {
                for entry in node.seq().iter().filter_map(Node::str) {
                    search_paths.push(working_dir.join(entry));
                }
            }
            Some(node) => {
                for entry in node.str().unwrap_or_default().split(':') {
                    // Keep Windows drive letters (`C:\luts`) whole.
                    if !entry.trim().is_empty() {
                        search_paths.push(working_dir.join(entry.trim()));
                    }
                }
            }
            None => {}
        }
        if search_paths.is_empty() {
            search_paths.push(working_dir);
        }

        let roles = root
            .get("roles")
            .map(|roles| {
                roles
                    .entries()
                    .iter()
                    .filter_map(|(role, node)| Some((role.to_lowercase(), node.str()?.to_owned())))
                    .collect()
            })
            .unwrap_or_default();

        let view_of = |node: &Node| -> Option<OcioView> {
            Some(OcioView {
                name: node.get_str("name")?.to_owned(),
                colorspace: node.get_str("colorspace").map(str::to_owned),
                view_transform: node.get_str("view_transform").map(str::to_owned),
                display_colorspace: node.get_str("display_colorspace").map(str::to_owned),
                looks: node
                    .get_str("looks")
                    .or_else(|| node.get_str("look"))
                    .map(str::to_owned),
            })
        };
        let shared_views: Vec<OcioView> = root
            .get("shared_views")
            .map(|views| views.seq().iter().filter_map(view_of).collect())
            .unwrap_or_default();
        let mut displays = Vec::new();
        if let Some(node) = root.get("displays") {
            for (name, views_node) in node.entries() {
                let mut views = Vec::new();
                for item in views_node.seq() {
                    if item.tag.as_deref() == Some("Views") {
                        // References to shared views.
                        for reference in item.seq().iter().filter_map(Node::str) {
                            if let Some(view) =
                                shared_views.iter().find(|view| names_match(&view.name, reference))
                            {
                                views.push(view.clone());
                            }
                        }
                    } else if let Some(view) = view_of(item) {
                        views.push(view);
                    }
                }
                displays.push(OcioDisplay {
                    name: name.clone(),
                    views,
                });
            }
        }
        let list = |key: &str| -> Vec<String> {
            match root.get(key) {
                Some(node) if !node.seq().is_empty() => {
                    node.seq().iter().filter_map(Node::str).map(str::to_owned).collect()
                }
                Some(node) => node
                    .str()
                    .unwrap_or_default()
                    .split(',')
                    .map(str::trim)
                    .filter(|entry| !entry.is_empty())
                    .map(str::to_owned)
                    .collect(),
                None => Vec::new(),
            }
        };
        let active_displays = list("active_displays");
        let active_views = list("active_views");

        let mut colorspaces = Vec::new();
        for (key, display_referred) in [("colorspaces", false), ("display_colorspaces", true)] {
            for node in root.get(key).map(Node::seq).unwrap_or_default() {
                let Some(name) = node.get_str("name") else {
                    continue;
                };
                let aliases = match node.get("aliases") {
                    Some(aliases) if !aliases.seq().is_empty() => aliases
                        .seq()
                        .iter()
                        .filter_map(Node::str)
                        .map(str::to_owned)
                        .collect(),
                    Some(aliases) => aliases
                        .str()
                        .map(|alias| vec![alias.to_owned()])
                        .unwrap_or_default(),
                    None => Vec::new(),
                };
                let (to_key, from_key) = if display_referred {
                    ("to_display_reference", "from_display_reference")
                } else {
                    ("to_scene_reference", "from_scene_reference")
                };
                colorspaces.push(ColorSpaceDef {
                    name: name.to_owned(),
                    aliases,
                    is_data: node.get("isdata").and_then(Node::boolean).unwrap_or(false),
                    display_referred,
                    to_reference: node.get(to_key).or_else(|| node.get("to_reference")).cloned(),
                    from_reference: node
                        .get(from_key)
                        .or_else(|| node.get("from_reference"))
                        .cloned(),
                });
            }
        }
        let view_transforms = root
            .get("view_transforms")
            .map(Node::seq)
            .unwrap_or_default()
            .iter()
            .filter_map(|node| {
                Some(ViewTransformDef {
                    name: node.get_str("name")?.to_owned(),
                    from_scene: node.get("from_scene_reference").or_else(|| node.get("from_reference")).cloned(),
                    to_scene: node.get("to_scene_reference").or_else(|| node.get("to_reference")).cloned(),
                })
            })
            .collect();
        let looks = root
            .get("looks")
            .map(Node::seq)
            .unwrap_or_default()
            .iter()
            .filter_map(|node| {
                Some(LookDef {
                    name: node.get_str("name")?.to_owned(),
                    process_space: node.get_str("process_space").unwrap_or_default().to_owned(),
                    transform: node.get("transform").cloned(),
                    inverse_transform: node.get("inverse_transform").cloned(),
                })
            })
            .collect();
        let config = Self {
            path: path.to_owned(),
            search_paths,
            roles,
            displays,
            active_displays,
            active_views,
            colorspaces,
            view_transforms,
            default_view_transform: root.get_str("default_view_transform").map(str::to_owned),
            looks,
        };
        if config.displays().is_empty() {
            return Err(parse_error("the config defines no displays"));
        }
        Ok(config)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The displays and views offered, every colour space, and the space
    /// holding linear Rec.709, as JSON.
    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "config": self.path.display().to_string(),
            "displays": self.displays().iter().map(|display| serde_json::json!({
                "name": display.name,
                "views": self.views(&display.name).iter().map(|view| view.name.clone()).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "colorspaces": self.colorspace_names(),
            "linear_rec709": self.linear_rec709_space(),
        })
    }

    /// Displays offered: the active ones in their order, else all.
    pub fn displays(&self) -> Vec<&OcioDisplay> {
        if self.active_displays.is_empty() {
            return self.displays.iter().collect();
        }
        let active: Vec<&OcioDisplay> = self
            .active_displays
            .iter()
            .filter_map(|name| self.displays.iter().find(|display| names_match(&display.name, name)))
            .collect();
        if active.is_empty() {
            self.displays.iter().collect()
        } else {
            active
        }
    }

    /// Views offered for `display`: the active ones in their order, else all.
    pub fn views(&self, display: &str) -> Vec<&OcioView> {
        let Some(display) = self.display(display) else {
            return Vec::new();
        };
        if self.active_views.is_empty() {
            return display.views.iter().collect();
        }
        let active: Vec<&OcioView> = self
            .active_views
            .iter()
            .filter_map(|name| display.views.iter().find(|view| names_match(&view.name, name)))
            .collect();
        if active.is_empty() {
            display.views.iter().collect()
        } else {
            active
        }
    }

    fn display(&self, name: &str) -> Option<&OcioDisplay> {
        self.displays.iter().find(|display| names_match(&display.name, name))
    }

    /// Names of every colour space (scene- and display-referred).
    pub fn colorspace_names(&self) -> Vec<&str> {
        self.colorspaces.iter().map(|space| space.name.as_str()).collect()
    }

    fn colorspace(&self, name: &str) -> Option<&ColorSpaceDef> {
        let name = self
            .roles
            .get(&name.trim().to_lowercase())
            .map_or(name, String::as_str);
        self.colorspaces.iter().find(|space| {
            names_match(&space.name, name) || space.aliases.iter().any(|alias| names_match(alias, name))
        })
    }

    /// The colour space holding linear Rec.709 (sRGB primaries) light, the
    /// renderer's working space: a space named for it, else the
    /// `scene_linear` role.
    pub fn linear_rec709_space(&self) -> Option<&str> {
        const NAMES: [&str; 9] = [
            "lin_rec709_srgb",
            "linear rec.709 (srgb)",
            "lin_rec709",
            "utility - linear - srgb",
            "utility - linear - rec.709",
            "linear srgb",
            "lin_srgb",
            "linear",
            "scene_linear",
        ];
        NAMES
            .iter()
            .find_map(|name| self.colorspace(name))
            .map(|space| space.name.as_str())
    }

    fn file_path(&self, src: &str) -> Option<PathBuf> {
        let direct = Path::new(src);
        if direct.is_absolute() {
            return direct.exists().then(|| direct.to_owned());
        }
        self.search_paths
            .iter()
            .map(|folder| folder.join(src))
            .find(|path| path.exists())
    }

    // -- compiling ----------------------------------------------------------

    fn to_reference(&self, space: &ColorSpaceDef, depth: usize) -> Result<Vec<Op>, OcioError> {
        if space.is_data {
            return Ok(Vec::new());
        }
        match (&space.to_reference, &space.from_reference) {
            (Some(node), _) => self.compile(node, false, depth + 1),
            (None, Some(node)) => self.compile(node, true, depth + 1),
            (None, None) => Ok(Vec::new()),
        }
    }

    fn from_reference(&self, space: &ColorSpaceDef, depth: usize) -> Result<Vec<Op>, OcioError> {
        if space.is_data {
            return Ok(Vec::new());
        }
        match (&space.from_reference, &space.to_reference) {
            (Some(node), _) => self.compile(node, false, depth + 1),
            (None, Some(node)) => self.compile(node, true, depth + 1),
            (None, None) => Ok(Vec::new()),
        }
    }

    /// Scene reference to display reference, through the default view
    /// transform (OCIO v2's rule for converting between the two).
    fn scene_to_display(&self, depth: usize) -> Result<Vec<Op>, OcioError> {
        let transform = self
            .default_view_transform
            .as_deref()
            .and_then(|name| self.view_transform(name))
            .or_else(|| self.view_transforms.iter().find(|vt| vt.from_scene.is_some() || vt.to_scene.is_some()));
        match transform {
            Some(transform) => self.view_transform_ops(transform, depth),
            None => Ok(Vec::new()),
        }
    }

    fn view_transform(&self, name: &str) -> Option<&ViewTransformDef> {
        self.view_transforms.iter().find(|vt| names_match(&vt.name, name))
    }

    fn view_transform_ops(&self, transform: &ViewTransformDef, depth: usize) -> Result<Vec<Op>, OcioError> {
        match (&transform.from_scene, &transform.to_scene) {
            (Some(node), _) => self.compile(node, false, depth + 1),
            (None, Some(node)) => self.compile(node, true, depth + 1),
            (None, None) => Err(OcioError::Unsupported(format!(
                "view transform `{}` without a scene-referred direction",
                transform.name
            ))),
        }
    }

    /// `src` to `dst`, through the reference space(s).
    fn colorspace_ops(&self, src: &str, dst: &str, depth: usize) -> Result<Vec<Op>, OcioError> {
        if depth > MAX_DEPTH {
            return Err(OcioError::TooDeep);
        }
        let source = self
            .colorspace(src)
            .ok_or_else(|| OcioError::UnknownColorSpace(src.to_owned()))?;
        let target = self
            .colorspace(dst)
            .ok_or_else(|| OcioError::UnknownColorSpace(dst.to_owned()))?;
        if source.is_data || target.is_data || source.name == target.name {
            return Ok(Vec::new());
        }
        let mut ops = self.to_reference(source, depth)?;
        match (source.display_referred, target.display_referred) {
            (false, true) => ops.extend(self.scene_to_display(depth)?),
            (true, false) => ops.extend(invert_ops(self.scene_to_display(depth)?)?),
            _ => {}
        }
        ops.extend(self.from_reference(target, depth)?);
        Ok(ops)
    }

    /// Looks named in `looks` (`+a, -b` or `a, b`), applied in the
    /// reference space they are inserted into.
    fn look_ops(&self, looks: &str, display_referred: bool, depth: usize) -> Result<Vec<Op>, OcioError> {
        let mut ops = Vec::new();
        for entry in looks.split([',', ':']).map(str::trim).filter(|entry| !entry.is_empty()) {
            let (inverse, name) = match entry.strip_prefix('-') {
                Some(name) => (true, name.trim()),
                None => (false, entry.trim_start_matches('+').trim()),
            };
            let look = self
                .looks
                .iter()
                .find(|look| names_match(&look.name, name))
                .ok_or_else(|| OcioError::UnknownLook(name.to_owned()))?;
            let space = self
                .colorspace(&look.process_space)
                .ok_or_else(|| OcioError::UnknownColorSpace(look.process_space.clone()))?;
            let mut into = self.from_reference(space, depth)?;
            let mut back = self.to_reference(space, depth)?;
            if space.display_referred != display_referred {
                return Err(OcioError::Unsupported(format!(
                    "look `{name}` in a {} process space here",
                    if space.display_referred { "display-referred" } else { "scene-referred" }
                )));
            }
            let body = match (inverse, &look.transform, &look.inverse_transform) {
                (false, Some(node), _) => self.compile(node, false, depth + 1)?,
                (false, None, Some(node)) => self.compile(node, true, depth + 1)?,
                (true, _, Some(node)) => self.compile(node, false, depth + 1)?,
                (true, Some(node), None) => self.compile(node, true, depth + 1)?,
                (_, None, None) => Vec::new(),
            };
            ops.append(&mut into);
            ops.extend(body);
            ops.append(&mut back);
        }
        Ok(ops)
    }

    /// The processor converting colour space `src` to `dst`.
    pub fn colorspace_processor(&self, src: &str, dst: &str) -> Result<OcioProcessor, OcioError> {
        Ok(OcioProcessor {
            ops: self.colorspace_ops(src, dst, 0)?,
        })
    }

    /// The processor showing colour space `input` through `display`'s
    /// `view`.
    pub fn view_processor(&self, input: &str, display: &str, view: &str) -> Result<OcioProcessor, OcioError> {
        Ok(OcioProcessor {
            ops: self.view_ops(input, display, view, 0)?,
        })
    }

    fn view_ops(&self, input: &str, display: &str, view: &str, depth: usize) -> Result<Vec<Op>, OcioError> {
        if depth > MAX_DEPTH {
            return Err(OcioError::TooDeep);
        }
        let display_def = self
            .display(display)
            .ok_or_else(|| OcioError::UnknownDisplay(display.to_owned()))?;
        let view_def = display_def
            .views
            .iter()
            .find(|candidate| names_match(&candidate.name, view))
            .ok_or_else(|| OcioError::UnknownView {
                display: display.to_owned(),
                view: view.to_owned(),
            })?;
        let source = self
            .colorspace(input)
            .ok_or_else(|| OcioError::UnknownColorSpace(input.to_owned()))?;
        let resolve_display_name = |name: &str| {
            if name.trim() == "<USE_DISPLAY_NAME>" {
                display_def.name.clone()
            } else {
                name.to_owned()
            }
        };
        if let Some(transform_name) = &view_def.view_transform {
            // OCIO v2: scene reference -> view transform -> display space.
            let transform = self
                .view_transform(transform_name)
                .ok_or_else(|| OcioError::UnknownViewTransform(transform_name.clone()))?;
            let target_name = resolve_display_name(
                view_def
                    .display_colorspace
                    .as_deref()
                    .or(view_def.colorspace.as_deref())
                    .unwrap_or(&display_def.name),
            );
            let target = self
                .colorspace(&target_name)
                .ok_or_else(|| OcioError::UnknownColorSpace(target_name.clone()))?;
            if source.display_referred {
                return Err(OcioError::Unsupported(
                    "viewing a display-referred colour space through a view transform".into(),
                ));
            }
            let mut ops = if source.is_data { Vec::new() } else { self.to_reference(source, depth)? };
            if source.is_data {
                return Ok(ops);
            }
            if let Some(looks) = &view_def.looks {
                ops.extend(self.look_ops(looks, false, depth)?);
            }
            ops.extend(self.view_transform_ops(transform, depth)?);
            ops.extend(self.from_reference(target, depth)?);
            Ok(ops)
        } else {
            let target_name = resolve_display_name(
                view_def
                    .colorspace
                    .as_deref()
                    .ok_or_else(|| parse_error(format!("view `{view}` names no colour space")))?,
            );
            let target = self
                .colorspace(&target_name)
                .ok_or_else(|| OcioError::UnknownColorSpace(target_name.clone()))?;
            if source.is_data || target.is_data {
                return Ok(Vec::new());
            }
            let mut ops = self.to_reference(source, depth)?;
            match (source.display_referred, target.display_referred) {
                (false, true) => {
                    if let Some(looks) = &view_def.looks {
                        ops.extend(self.look_ops(looks, false, depth)?);
                    }
                    ops.extend(self.scene_to_display(depth)?);
                }
                (true, false) => ops.extend(invert_ops(self.scene_to_display(depth)?)?),
                (referred, _) => {
                    if let Some(looks) = &view_def.looks {
                        ops.extend(self.look_ops(looks, referred, depth)?);
                    }
                }
            }
            ops.extend(self.from_reference(target, depth)?);
            Ok(ops)
        }
    }

    /// Compiles one transform node (inverted when `inverse`).
    fn compile(&self, node: &Node, inverse: bool, depth: usize) -> Result<Vec<Op>, OcioError> {
        if depth > MAX_DEPTH {
            return Err(OcioError::TooDeep);
        }
        let inverse = inverse ^ direction_inverse(node);
        let tag = node.tag.as_deref().unwrap_or("");
        let numbers = |key: &str| node.get(key).and_then(Node::numbers);
        let rgb = |key: &str, default: f64| -> Result<[f64; 3], OcioError> {
            match numbers(key) {
                Some(values) => rgb3(&values)
                    .ok_or_else(|| parse_error(format!("{tag}: `{key}` needs 1, 3, or 4 values"))),
                None => Ok([default; 3]),
            }
        };
        let ops = match tag {
            "GroupTransform" => {
                let mut ops = Vec::new();
                for child in node.get("children").map(Node::seq).unwrap_or_default() {
                    ops.extend(self.compile(child, false, depth + 1)?);
                }
                ops
            }
            "MatrixTransform" => {
                let matrix = numbers("matrix").unwrap_or_else(|| {
                    vec![1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
                });
                let offset = numbers("offset").unwrap_or_else(|| vec![0.0; 4]);
                if matrix.len() != 16 || offset.len() < 3 {
                    return Err(parse_error("MatrixTransform needs a 16-value matrix"));
                }
                vec![Op::Matrix {
                    m: [
                        [matrix[0], matrix[1], matrix[2]],
                        [matrix[4], matrix[5], matrix[6]],
                        [matrix[8], matrix[9], matrix[10]],
                    ],
                    offset: [offset[0], offset[1], offset[2]],
                }]
            }
            "ExponentTransform" => {
                let style = node.get_str("style").unwrap_or("basicFwd").to_ascii_lowercase();
                vec![Op::Power {
                    gamma: rgb("value", 1.0)?,
                    mirror: style.contains("mirror") || style.contains("pass"),
                }]
            }
            "ExponentWithLinearTransform" => vec![Op::MonCurve {
                gamma: rgb("gamma", 1.0)?,
                offset: rgb("offset", 0.0)?,
                inverse: false,
            }],
            "LogTransform" => {
                let base = node.get("base").and_then(Node::number).unwrap_or(2.0);
                vec![Op::Log {
                    base,
                    log_slope: [1.0; 3],
                    log_offset: [0.0; 3],
                    lin_slope: [1.0; 3],
                    lin_offset: [0.0; 3],
                    lin_break: None,
                    linear_slope: None,
                    inverse: false,
                }]
            }
            "LogAffineTransform" | "LogCameraTransform" => {
                let base = node.get("base").and_then(Node::number).unwrap_or(2.0);
                let lin_break = if tag == "LogCameraTransform" {
                    Some(rgb("lin_side_break", 0.0).and_then(|value| {
                        numbers("lin_side_break")
                            .map(|_| value)
                            .ok_or_else(|| parse_error("LogCameraTransform needs lin_side_break"))
                    })?)
                } else {
                    None
                };
                let linear_slope = match numbers("linear_slope") {
                    Some(values) => Some(rgb3(&values).ok_or_else(|| parse_error("bad linear_slope"))?),
                    None => None,
                };
                vec![Op::Log {
                    base,
                    log_slope: rgb("log_side_slope", 1.0)?,
                    log_offset: rgb("log_side_offset", 0.0)?,
                    lin_slope: rgb("lin_side_slope", 1.0)?,
                    lin_offset: rgb("lin_side_offset", 0.0)?,
                    lin_break,
                    linear_slope,
                    inverse: false,
                }]
            }
            "CDLTransform" => {
                let style = node.get_str("style").unwrap_or("Fwd").to_ascii_lowercase();
                vec![Op::Cdl {
                    slope: rgb("slope", 1.0)?,
                    offset: rgb("offset", 0.0)?,
                    power: rgb("power", 1.0)?,
                    saturation: node.get("sat").and_then(Node::number).unwrap_or(1.0),
                    clamp: !style.contains("noclamp"),
                    inverse: style.contains("rev"),
                }]
            }
            "RangeTransform" => {
                let get = |key: &str| node.get(key).and_then(Node::number);
                let (min_in, max_in, min_out, max_out) =
                    (get("min_in_value"), get("max_in_value"), get("min_out_value"), get("max_out_value"));
                let clamp = !node
                    .get_str("style")
                    .is_some_and(|style| style.to_ascii_lowercase().contains("noclamp"));
                let mut ops = Vec::new();
                if let (Some(min_in), Some(max_in), Some(min_out), Some(max_out)) =
                    (min_in, max_in, min_out, max_out)
                {
                    let scale = (max_out - min_out) / (max_in - min_in);
                    ops.push(Op::Matrix {
                        m: [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]],
                        offset: [min_out - min_in * scale; 3],
                    });
                    if clamp {
                        ops.push(Op::Clamp {
                            min: Some(min_out.min(max_out)),
                            max: Some(min_out.max(max_out)),
                        });
                    }
                } else if clamp {
                    ops.push(Op::Clamp {
                        min: min_out.or(min_in),
                        max: max_out.or(max_in),
                    });
                }
                ops
            }
            "AllocationTransform" => {
                let allocation = node.get_str("allocation").unwrap_or("uniform").to_ascii_lowercase();
                let vars = numbers("vars").unwrap_or_else(|| vec![0.0, 1.0]);
                let (min, max) = (vars.first().copied().unwrap_or(0.0), vars.get(1).copied().unwrap_or(1.0));
                let mut ops = Vec::new();
                if allocation == "lg2" {
                    let offset = vars.get(2).copied().unwrap_or(0.0);
                    ops.push(Op::Matrix {
                        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        offset: [offset; 3],
                    });
                    ops.push(Op::Log {
                        base: 2.0,
                        log_slope: [1.0; 3],
                        log_offset: [0.0; 3],
                        lin_slope: [1.0; 3],
                        lin_offset: [0.0; 3],
                        lin_break: None,
                        linear_slope: None,
                        inverse: false,
                    });
                }
                let scale = 1.0 / (max - min);
                ops.push(Op::Matrix {
                    m: [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]],
                    offset: [-min * scale; 3],
                });
                ops
            }
            "FileTransform" => {
                let src = node
                    .get_str("src")
                    .ok_or_else(|| parse_error("FileTransform needs src"))?;
                let path = self
                    .file_path(src)
                    .ok_or_else(|| OcioError::Io(format!("LUT file {src} was not found on the search path")))?;
                luts::read(&path)?
            }
            "ColorSpaceTransform" => {
                let src = node.get_str("src").ok_or_else(|| parse_error("ColorSpaceTransform needs src"))?;
                let dst = node.get_str("dst").ok_or_else(|| parse_error("ColorSpaceTransform needs dst"))?;
                self.colorspace_ops(src, dst, depth + 1)?
            }
            "DisplayViewTransform" => {
                let get = |key: &str| {
                    node.get_str(key)
                        .ok_or_else(|| parse_error(format!("DisplayViewTransform needs {key}")))
                };
                self.view_ops(get("src")?, get("display")?, get("view")?, depth + 1)?
            }
            "LookTransform" => {
                let src = node.get_str("src").ok_or_else(|| parse_error("LookTransform needs src"))?;
                let dst = node.get_str("dst").unwrap_or(src);
                let source = self
                    .colorspace(src)
                    .ok_or_else(|| OcioError::UnknownColorSpace(src.to_owned()))?;
                let mut ops = self.to_reference(source, depth)?;
                ops.extend(self.look_ops(node.get_str("looks").unwrap_or_default(), source.display_referred, depth)?);
                let target = self
                    .colorspace(dst)
                    .ok_or_else(|| OcioError::UnknownColorSpace(dst.to_owned()))?;
                ops.extend(self.from_reference(target, depth)?);
                ops
            }
            "BuiltinTransform" => {
                let style = node
                    .get_str("style")
                    .ok_or_else(|| parse_error("BuiltinTransform needs style"))?;
                builtin::ops(style)?
            }
            "" => return Err(parse_error("a transform without a type tag")),
            other => return Err(OcioError::Unsupported(format!("the {other}"))),
        };
        if inverse { invert_ops(ops) } else { Ok(ops) }
    }

    /// Bakes `display`/`view` into a display LUT for the renderer: input
    /// is the display-encoded sRGB image of linear Rec.709 light exposed
    /// [`OCIO_HEADROOM_STOPS`] darker and clipped at white (the renderer's
    /// Linear view), read as colour space `input`.
    pub fn bake_view(&self, input: &str, display: &str, view: &str, size: usize) -> Result<CubeLut, OcioError> {
        let processor = self.view_processor(input, display, view)?;
        let size = size.clamp(2, MAX_CUBE_LUT_SIZE);
        let headroom = f64::from(2.0_f32.powf(OCIO_HEADROOM_STOPS));
        let last = (size - 1) as f32;
        let mut data = Vec::with_capacity(size * size * size);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let encoded = [r, g, b].map(|i| srgb_to_linear(i as f32 / last));
                    let scene = encoded.map(|value| f64::from(value) * headroom);
                    let out = processor.apply(scene);
                    data.push(Vec3::from_array(out.map(|value| {
                        if value.is_finite() { value as f32 } else { 0.0 }
                    })));
                }
            }
        }
        Ok(CubeLut {
            title: format!("{display} / {view}"),
            size,
            data,
        })
    }
}

// ---------------------------------------------------------------------------
// LUT files

mod luts {
    use super::*;

    fn read_text(path: &Path) -> Result<String, OcioError> {
        let size = std::fs::metadata(path)
            .map_err(|error| OcioError::Io(format!("{}: {error}", path.display())))?
            .len();
        if size > MAX_OCIO_LUT_BYTES {
            return Err(OcioError::Io(format!("{}: the LUT is too large", path.display())));
        }
        std::fs::read_to_string(path).map_err(|error| OcioError::Io(format!("{}: {error}", path.display())))
    }

    fn numbers_of(line: &str, path: &Path) -> Result<Vec<f64>, OcioError> {
        line.split_whitespace()
            .map(|word| {
                word.parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| parse_error(format!("{}: `{word}` is not a number", path.display())))
            })
            .collect()
    }

    /// The operations of the LUT file at `path`, by extension.
    pub fn read(path: &Path) -> Result<Vec<Op>, OcioError> {
        let extension = path
            .extension()
            .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let text = read_text(path)?;
        match extension.as_str() {
            "cube" => cube(&text, path),
            "spi1d" => spi1d(&text, path),
            "spi3d" => spi3d(&text, path),
            "spimtx" => spimtx(&text, path),
            other => Err(OcioError::Unsupported(format!("the .{other} LUT format ({})", path.display()))),
        }
    }

    /// Resolve/Iridas `.cube`: a 1D table, a 3D table, or a 1D shaper
    /// followed by a 3D table.
    pub fn cube(text: &str, path: &Path) -> Result<Vec<Op>, OcioError> {
        let mut size_1d = None;
        let mut size_3d = None;
        let mut domain_min = [0.0; 3];
        let mut domain_max = [1.0; 3];
        let mut range_1d = (0.0, 1.0);
        let mut range_3d = None;
        let mut rows = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut words = line.split_whitespace();
            let keyword = words.next().unwrap_or_default();
            let rest = || numbers_of(&line[keyword.len()..], path);
            match keyword {
                "TITLE" => {}
                "LUT_1D_SIZE" => size_1d = rest()?.first().map(|v| *v as usize),
                "LUT_3D_SIZE" => size_3d = rest()?.first().map(|v| *v as usize),
                "DOMAIN_MIN" => domain_min = rgb3(&rest()?).ok_or_else(|| parse_error("bad DOMAIN_MIN"))?,
                "DOMAIN_MAX" => domain_max = rgb3(&rest()?).ok_or_else(|| parse_error("bad DOMAIN_MAX"))?,
                "LUT_1D_INPUT_RANGE" => {
                    let values = rest()?;
                    range_1d = (values.first().copied().unwrap_or(0.0), values.get(1).copied().unwrap_or(1.0));
                }
                "LUT_3D_INPUT_RANGE" => {
                    let values = rest()?;
                    range_3d = Some((values.first().copied().unwrap_or(0.0), values.get(1).copied().unwrap_or(1.0)));
                }
                _ if keyword.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '.') => {
                    let values = numbers_of(line, path)?;
                    let row = rgb3(&values).filter(|_| values.len() == 3).ok_or_else(|| {
                        parse_error(format!("{}: expected three values per row", path.display()))
                    })?;
                    rows.push(row);
                }
                _ => {}
            }
        }
        let mut ops = Vec::new();
        let mut rows = rows.as_slice();
        if let Some(size) = size_1d {
            if size < 2 || rows.len() < size {
                return Err(parse_error(format!("{}: too few 1D entries", path.display())));
            }
            let domain = if size_3d.is_some() {
                range_1d
            } else {
                (domain_min[0], domain_max[0])
            };
            ops.push(Op::Lut1d {
                table: Arc::new(rows[..size].to_vec()),
                domain,
                inverse: false,
            });
            rows = &rows[size..];
        }
        if let Some(size) = size_3d {
            if size < 2 || rows.len() != size * size * size {
                return Err(parse_error(format!("{}: expected {} 3D entries", path.display(), size * size * size)));
            }
            let (min, max) = match (range_3d, size_1d) {
                (Some((low, high)), _) => ([low; 3], [high; 3]),
                (None, Some(_)) => ([0.0; 3], [1.0; 3]),
                (None, None) => (domain_min, domain_max),
            };
            ops.push(Op::Lut3d {
                size,
                data: Arc::new(rows.to_vec()),
                domain_min: min,
                domain_max: max,
            });
        } else if size_1d.is_none() {
            return Err(parse_error(format!("{}: no LUT size", path.display())));
        }
        Ok(ops)
    }

    /// Sony Pictures Imageworks `.spi1d`.
    pub fn spi1d(text: &str, path: &Path) -> Result<Vec<Op>, OcioError> {
        let mut domain = (0.0, 1.0);
        let mut components = 1;
        let mut length = None;
        let mut rows = Vec::new();
        let mut in_values = false;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if in_values {
                if line.starts_with('}') {
                    break;
                }
                let values = numbers_of(line, path)?;
                let row = match (components, values.as_slice()) {
                    (1, [v, ..]) => [*v; 3],
                    (2, [r, g, ..]) => [*r, *g, 0.0],
                    (_, [r, g, b, ..]) => [*r, *g, *b],
                    _ => return Err(parse_error(format!("{}: short row", path.display()))),
                };
                rows.push(row);
                continue;
            }
            let mut words = line.split_whitespace();
            match words.next().unwrap_or_default() {
                "From" => {
                    let values = numbers_of(&line[4..], path)?;
                    domain = (values.first().copied().unwrap_or(0.0), values.get(1).copied().unwrap_or(1.0));
                }
                "Components" => components = words.next().and_then(|word| word.parse().ok()).unwrap_or(1),
                "Length" => length = words.next().and_then(|word| word.parse::<usize>().ok()),
                "{" => in_values = true,
                _ => {}
            }
        }
        if rows.len() < 2 || length.is_some_and(|length| length != rows.len()) {
            return Err(parse_error(format!("{}: the table length does not match", path.display())));
        }
        Ok(vec![Op::Lut1d {
            table: Arc::new(rows),
            domain,
            inverse: false,
        }])
    }

    /// Sony Pictures Imageworks `.spi3d`: each row names its indices.
    pub fn spi3d(text: &str, path: &Path) -> Result<Vec<Op>, OcioError> {
        let mut lines = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'));
        let _header = lines.next();
        let _components = lines.next();
        let sizes = numbers_of(lines.next().unwrap_or_default(), path)?;
        let size = match sizes.as_slice() {
            [a, b, c] if a == b && b == c && *a >= 2.0 => *a as usize,
            _ => return Err(OcioError::Unsupported(format!("non-cubic .spi3d ({})", path.display()))),
        };
        if size > MAX_CUBE_LUT_SIZE {
            return Err(parse_error(format!("{}: the LUT is too large", path.display())));
        }
        let mut data = vec![[0.0; 3]; size * size * size];
        let mut filled = 0;
        for line in lines {
            let values = numbers_of(line, path)?;
            let [r, g, b, red, green, blue] = values[..] else {
                return Err(parse_error(format!("{}: expected six values per row", path.display())));
            };
            let (r, g, b) = (r as usize, g as usize, b as usize);
            if r >= size || g >= size || b >= size {
                return Err(parse_error(format!("{}: an index is out of range", path.display())));
            }
            data[r + size * (g + size * b)] = [red, green, blue];
            filled += 1;
        }
        if filled != size * size * size {
            return Err(parse_error(format!("{}: expected {} entries", path.display(), size * size * size)));
        }
        Ok(vec![Op::Lut3d {
            size,
            data: Arc::new(data),
            domain_min: [0.0; 3],
            domain_max: [1.0; 3],
        }])
    }

    /// `.spimtx`: a 3x4 matrix whose last column is an offset in 16-bit
    /// code values.
    pub fn spimtx(text: &str, path: &Path) -> Result<Vec<Op>, OcioError> {
        let values = numbers_of(&text.split_whitespace().collect::<Vec<_>>().join(" "), path)?;
        if values.len() != 12 {
            return Err(parse_error(format!("{}: expected 12 values", path.display())));
        }
        Ok(vec![Op::Matrix {
            m: [
                [values[0], values[1], values[2]],
                [values[4], values[5], values[6]],
                [values[8], values[9], values[10]],
            ],
            offset: [values[3] / 65535.0, values[7] / 65535.0, values[11] / 65535.0],
        }])
    }
}

// ---------------------------------------------------------------------------
// Built-in transforms

pub(crate) mod builtin {
    //! OCIO v2 BuiltinTransform styles: the ACES colour space conversions,
    //! display encodings, and ACES 1.0 SDR output transforms. HDR outputs
    //! and ACES 2.0 are reported as unsupported.

    use super::*;

    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    /// ACES AP0 to CIE XYZ (D60 white).
    pub const AP0_TO_XYZ: [[f64; 3]; 3] = [
        [0.952_552_395_9, 0.0, 0.000_093_678_6],
        [0.343_966_449_8, 0.728_166_096_6, -0.072_132_546_4],
        [0.0, 0.0, 1.008_825_184_4],
    ];
    /// ACES AP1 to CIE XYZ (D60 white).
    pub const AP1_TO_XYZ: [[f64; 3]; 3] = [
        [0.662_454_181_1, 0.134_004_206_5, 0.156_187_687_0],
        [0.272_228_716_8, 0.674_081_765_8, 0.053_689_517_4],
        [-0.005_574_649_5, 0.004_060_733_5, 1.010_339_100_3],
    ];
    /// Bradford adaptation from the ACES white (D60) to D65.
    pub const D60_TO_D65: [[f64; 3]; 3] = [
        [0.987_224_0, -0.006_113_3, 0.015_953_3],
        [-0.007_598_4, 1.001_860_7, 0.005_330_0],
        [0.003_072_5, -0.005_095_9, 1.081_680_6],
    ];
    /// CIE XYZ (D65) to linear Rec.709.
    pub const XYZ_TO_REC709: [[f64; 3]; 3] = [
        [3.240_969_941_9, -1.537_383_177_6, -0.498_610_760_3],
        [-0.969_243_636_3, 1.875_967_501_5, 0.041_555_057_4],
        [0.055_630_079_7, -0.203_976_958_9, 1.056_971_514_2],
    ];
    /// CIE XYZ (D65) to linear Rec.2020.
    pub const XYZ_TO_REC2020: [[f64; 3]; 3] = [
        [1.716_651_188_0, -0.355_670_783_8, -0.253_366_281_4],
        [-0.666_684_351_8, 1.616_481_236_6, 0.015_768_545_8],
        [0.017_639_857_4, -0.042_770_613_3, 0.942_103_121_2],
    ];
    /// CIE XYZ (D65) to linear Display P3.
    pub const XYZ_TO_P3_D65: [[f64; 3]; 3] = [
        [2.493_496_911_9, -0.931_383_617_9, -0.402_710_784_5],
        [-0.829_488_969_6, 1.762_664_060_3, 0.023_624_685_8],
        [0.035_845_830_2, -0.076_172_389_3, 0.956_884_524_0],
    ];

    fn matrix(m: [[f64; 3]; 3]) -> Op {
        Op::Matrix { m, offset: [0.0; 3] }
    }

    fn ap0_to_ap1() -> [[f64; 3]; 3] {
        mat_product(&super::mat_inverse(&AP1_TO_XYZ).unwrap_or(IDENTITY), &AP0_TO_XYZ)
    }

    /// ACEScct (log) to linear AP1.
    fn acescct_to_lin() -> Op {
        // ACEScct: lin <= 0.0078125 is linear; above, (log2(lin) + 9.72) / 17.52.
        Op::Log {
            base: 2.0,
            log_slope: [1.0 / 17.52; 3],
            log_offset: [9.72 / 17.52; 3],
            lin_slope: [1.0; 3],
            lin_offset: [0.0; 3],
            lin_break: Some([0.007_812_5; 3]),
            linear_slope: Some([10.540_237_741_654_5; 3]),
            inverse: true,
        }
    }

    /// ACEScc (pure log) to linear AP1 (the toe below 2^-15 is ignored).
    fn acescc_to_lin() -> Op {
        Op::Log {
            base: 2.0,
            log_slope: [1.0 / 17.52; 3],
            log_offset: [9.72 / 17.52; 3],
            lin_slope: [1.0; 3],
            lin_offset: [0.0; 3],
            lin_break: None,
            linear_slope: None,
            inverse: true,
        }
    }

    /// Linear light to the sRGB transfer curve.
    fn srgb_encode() -> Op {
        Op::MonCurve {
            gamma: [2.4; 3],
            offset: [0.055; 3],
            inverse: true,
        }
    }

    fn gamma_encode(gamma: f64) -> Op {
        Op::Power {
            gamma: [1.0 / gamma; 3],
            mirror: false,
        }
    }

    /// The operations of BuiltinTransform `style`.
    pub fn ops(style: &str) -> Result<Vec<Op>, OcioError> {
        let key = style.trim().to_ascii_uppercase();
        let key = key.strip_prefix("UTILITY - ").unwrap_or(&key);
        let xyz_d65_from_ap0 = mat_product(&D60_TO_D65, &AP0_TO_XYZ);
        let xyz_d65_from_ap1 = mat_product(&D60_TO_D65, &AP1_TO_XYZ);
        Ok(match key {
            "IDENTITY" => Vec::new(),
            "ACESCG_TO_ACES2065-1" => vec![matrix(super::mat_inverse(&ap0_to_ap1()).unwrap_or(IDENTITY))],
            "ACESCCT_TO_ACES2065-1" => vec![
                acescct_to_lin(),
                matrix(super::mat_inverse(&ap0_to_ap1()).unwrap_or(IDENTITY)),
            ],
            "ACESCC_TO_ACES2065-1" => vec![
                acescc_to_lin(),
                matrix(super::mat_inverse(&ap0_to_ap1()).unwrap_or(IDENTITY)),
            ],
            "ACES-AP0_TO_CIE-XYZ-D65_BFD" => vec![matrix(xyz_d65_from_ap0)],
            "ACES-AP1_TO_CIE-XYZ-D65_BFD" => vec![matrix(xyz_d65_from_ap1)],
            "ACES-AP1_TO_LINEAR-REC709_BFD" => {
                vec![matrix(mat_product(&XYZ_TO_REC709, &xyz_d65_from_ap1))]
            }
            "DISPLAY - CIE-XYZ-D65_TO_SRGB" => vec![matrix(XYZ_TO_REC709), Op::Clamp { min: Some(0.0), max: None }, srgb_encode()],
            "DISPLAY - CIE-XYZ-D65_TO_REC.1886-REC.709" => {
                vec![matrix(XYZ_TO_REC709), gamma_encode(2.4)]
            }
            "DISPLAY - CIE-XYZ-D65_TO_REC.1886-REC.2020" => {
                vec![matrix(XYZ_TO_REC2020), gamma_encode(2.4)]
            }
            "DISPLAY - CIE-XYZ-D65_TO_G2.2-REC.709" => vec![matrix(XYZ_TO_REC709), gamma_encode(2.2)],
            "DISPLAY - CIE-XYZ-D65_TO_DISPLAYP3" => {
                vec![matrix(XYZ_TO_P3_D65), Op::Clamp { min: Some(0.0), max: None }, srgb_encode()]
            }
            "DISPLAY - CIE-XYZ-D65_TO_G2.6-P3-D65" => vec![matrix(XYZ_TO_P3_D65), gamma_encode(2.6)],
            "ACES-OUTPUT - ACES2065-1_TO_CIE-XYZ-D65 - SDR-VIDEO_1.0" => {
                vec![Op::AcesOutput(AcesOutput::Video)]
            }
            "ACES-OUTPUT - ACES2065-1_TO_CIE-XYZ-D65 - SDR-CINEMA_1.0" => {
                vec![Op::AcesOutput(AcesOutput::Cinema)]
            }
            _ => return Err(OcioError::Unsupported(format!("the built-in transform `{style}`"))),
        })
    }

    /// Which ACES 1.0 SDR output transform.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum AcesOutput {
        /// 100 nit video in a dim surround (Rec.709 / sRGB ODTs).
        Video,
        /// 48 nit cinema in a dark surround.
        Cinema,
    }

    // Segmented spline coefficients of the ACES 1.0 reference rendering
    // transform (C5) and 48-nit output tone scale (C9), from the ACES
    // reference implementation.
    const C5_LOW: [f64; 6] = [-4.0, -4.0, -3.157_376_577_3, -0.485_249_995_8, 1.847_732_470_6, 1.847_732_470_6];
    const C5_HIGH: [f64; 6] = [-0.718_548_242_5, 2.081_030_717_2, 3.668_124_123_7, 4.0, 4.0, 4.0];
    const C9_LOW: [f64; 10] = [
        -1.698_970_004_3, -1.698_970_004_3, -1.477_9, -1.229_1, -0.864_8, -0.448, 0.005_18,
        0.451_108_033_4, 0.911_374_441_4, 0.911_374_441_4,
    ];
    const C9_HIGH: [f64; 10] = [
        0.515_438_696_5, 0.847_043_778_3, 1.135_8, 1.380_2, 1.519_7, 1.598_5, 1.646_7,
        1.674_609_135_7, 1.687_873_339, 1.687_873_339,
    ];
    const M: [[f64; 3]; 3] = [[0.5, -1.0, 0.5], [-1.0, 1.0, 0.5], [0.5, 0.0, 0.0]];

    /// A segmented spline of the ACES tone scales: quadratic B-spline
    /// segments in log-log space between the `min`, `mid`, and `max`
    /// points, with linear extensions beyond them.
    struct Spline<'a> {
        low: &'a [f64],
        high: &'a [f64],
        min: (f64, f64),
        mid: (f64, f64),
        max: (f64, f64),
        slope_low: f64,
        slope_high: f64,
        knots: usize,
    }

    fn spline(log_x: f64, spline: &Spline<'_>) -> f64 {
        let Spline {
            low,
            high,
            min,
            mid,
            max,
            slope_low,
            slope_high,
            knots,
        } = *spline;
        let (log_min_x, log_min_y) = (min.0.log10(), min.1.log10());
        let log_mid_x = mid.0.log10();
        let (log_max_x, log_max_y) = (max.0.log10(), max.1.log10());
        let eval = |coefs: &[f64], t: f64| {
            let j = (t.floor() as usize).min(knots - 2);
            let t = t - j as f64;
            let c = [coefs[j], coefs[j + 1], coefs[j + 2]];
            let column = |k: usize| c[0] * M[0][k] + c[1] * M[1][k] + c[2] * M[2][k];
            (t * t) * column(0) + t * column(1) + column(2)
        };
        let log_y = if log_x <= log_min_x {
            log_x * slope_low + (log_min_y - slope_low * log_min_x)
        } else if log_x < log_mid_x {
            let t = (knots - 1) as f64 * (log_x - log_min_x) / (log_mid_x - log_min_x);
            eval(low, t)
        } else if log_x < log_max_x {
            let t = (knots - 1) as f64 * (log_x - log_mid_x) / (log_max_x - log_mid_x);
            eval(high, t)
        } else {
            log_x * slope_high + (log_max_y - slope_high * log_max_x)
        };
        10f64.powf(log_y)
    }

    fn c5(x: f64) -> f64 {
        let log_x = x.max(f64::MIN_POSITIVE).log10();
        spline(
            log_x,
            &Spline {
                low: &C5_LOW,
                high: &C5_HIGH,
                min: (0.18 * 2f64.powf(-15.0), 0.0001),
                mid: (0.18, 4.8),
                max: (0.18 * 2f64.powf(18.0), 10_000.0),
                slope_low: 0.0,
                slope_high: 0.0,
                knots: 4,
            },
        )
    }

    fn c9(x: f64) -> f64 {
        let min = (c5(0.18 * 2f64.powf(-6.5)), 0.02);
        let mid = (c5(0.18), 4.8);
        let max = (c5(0.18 * 2f64.powf(6.5)), 48.0);
        let log_x = x.max(f64::MIN_POSITIVE).log10();
        spline(
            log_x,
            &Spline {
                low: &C9_LOW,
                high: &C9_HIGH,
                min,
                mid,
                max,
                slope_low: 0.0,
                slope_high: 0.04,
                knots: 8,
            },
        )
    }

    fn rgb_to_saturation(rgb: [f64; 3]) -> f64 {
        let max = rgb[0].max(rgb[1]).max(rgb[2]);
        let min = rgb[0].min(rgb[1]).min(rgb[2]);
        (max.max(1e-10) - min.max(1e-10)) / max.max(1e-2)
    }

    fn rgb_to_yc(rgb: [f64; 3]) -> f64 {
        const RADIUS_WEIGHT: f64 = 1.75;
        let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
        let chroma = (b * (b - g) + g * (g - r) + r * (r - b)).max(0.0).sqrt();
        (b + g + r + RADIUS_WEIGHT * chroma) / 3.0
    }

    fn rgb_to_hue(rgb: [f64; 3]) -> f64 {
        let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
        if r == g && g == b {
            return 0.0;
        }
        let hue = (3f64.sqrt() * (g - b)).atan2(2.0 * r - g - b).to_degrees();
        if hue < 0.0 { hue + 360.0 } else { hue }
    }

    fn sigmoid_shaper(x: f64) -> f64 {
        let t = (1.0 - (x / 2.0).abs()).max(0.0);
        (1.0 + x.signum() * (1.0 - t * t)) / 2.0
    }

    fn glow_fwd(yc: f64, glow_gain: f64, glow_mid: f64) -> f64 {
        if yc <= 2.0 / 3.0 * glow_mid {
            glow_gain
        } else if yc >= 2.0 * glow_mid {
            0.0
        } else {
            glow_gain * (glow_mid / yc - 0.5)
        }
    }

    fn center_hue(hue: f64, center: f64) -> f64 {
        let centered = hue - center;
        if centered < -180.0 {
            centered + 360.0
        } else if centered > 180.0 {
            centered - 360.0
        } else {
            centered
        }
    }

    fn cubic_basis_shaper(x: f64, width: f64) -> f64 {
        const M4: [[f64; 4]; 4] = [
            [-1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0],
            [3.0 / 6.0, -6.0 / 6.0, 3.0 / 6.0, 0.0],
            [-3.0 / 6.0, 0.0, 3.0 / 6.0, 0.0],
            [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0],
        ];
        let knots = [-width / 2.0, -width / 4.0, 0.0, width / 4.0, width / 2.0];
        if x <= knots[0] || x >= knots[4] {
            return 0.0;
        }
        let knot = ((x - knots[0]) / (width / 4.0)).floor() as usize;
        let knot = knot.min(3);
        let t = (x - knots[knot]) / (width / 4.0);
        let monomials = [t * t * t, t * t, t, 1.0];
        let column = 3 - knot;
        (0..4).map(|i| monomials[i] * M4[i][column]).sum::<f64>() * 1.5
    }

    fn saturation_matrix(saturation: f64, luminance: [f64; 3]) -> [[f64; 3]; 3] {
        std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                let base = (1.0 - saturation) * luminance[col];
                if row == col { base + saturation } else { base }
            })
        })
    }

    impl AcesOutput {
        /// ACES2065-1 to CIE XYZ (D65), normalized so display white is 1.
        pub fn apply(self, aces: [f64; 3]) -> [f64; 3] {
            // Reference rendering transform.
            let mut rgb = aces;
            let saturation = rgb_to_saturation(rgb);
            let yc = rgb_to_yc(rgb);
            let s = sigmoid_shaper((saturation - 0.4) / 0.2);
            let added_glow = 1.0 + glow_fwd(yc, 0.05 * s, 0.08);
            rgb = rgb.map(|v| v * added_glow);
            let hue = rgb_to_hue(rgb);
            let centered = center_hue(hue, 0.0);
            let weight = cubic_basis_shaper(centered, 135.0);
            rgb[0] += weight * saturation * (0.03 - rgb[0]) * (1.0 - 0.82);
            let ap1 = mat_mul(&ap0_to_ap1(), rgb.map(|v| v.max(0.0))).map(|v| v.max(0.0));
            let luminance = [AP1_TO_XYZ[1][0], AP1_TO_XYZ[1][1], AP1_TO_XYZ[1][2]];
            let ap1 = mat_mul(&saturation_matrix(0.96, luminance), ap1);
            let ocesish = ap1.map(c5);

            // Output tone scale and display normalization.
            let cinema = ocesish.map(c9);
            let (cinema_black, cinema_white) = (0.02, 48.0);
            let mut linear = cinema.map(|y| (y - cinema_black) / (cinema_white - cinema_black));
            if self == Self::Video {
                // Dark to dim surround: a gamma on luminance, kept in xyY.
                let xyz = mat_mul(&AP1_TO_XYZ, linear);
                let sum = (xyz[0] + xyz[1] + xyz[2]).max(1e-10);
                let (x, y) = (xyz[0] / sum, xyz[1] / sum);
                let luma = xyz[1].max(0.0).powf(0.9811);
                let adjusted = if y > 0.0 {
                    [x * luma / y, luma, (1.0 - x - y) * luma / y]
                } else {
                    [0.0, luma, 0.0]
                };
                linear = mat_mul(&super::mat_inverse(&AP1_TO_XYZ).unwrap_or(IDENTITY), adjusted);
                linear = mat_mul(&saturation_matrix(0.93, luminance), linear);
            }
            mat_mul(&mat_product(&D60_TO_D65, &AP1_TO_XYZ), linear)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3], tolerance: f64) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= tolerance)
    }

    const CONFIG: &str = r#"
ocio_profile_version: 2
search_path: luts
roles:
  scene_linear: Linear Rec.709
  default: Linear Rec.709
displays:
  sRGB:
    - !<View> {name: Standard, colorspace: sRGB}
    - !<View> {name: Graded, colorspace: sRGB, looks: +Warm}
    - !<View> {name: Film, colorspace: Film Log}
    - !<View> {name: Raw, colorspace: Raw}
  Rec.1886:
    - !<View> {name: Standard, view_transform: Un-tone-mapped, display_colorspace: Gamma 2.4 Rec.709 - Display}
active_displays: [sRGB, Rec.1886]
looks:
  - !<Look>
    name: Warm
    process_space: Linear Rec.709
    transform: !<CDLTransform> {slope: [1.2, 1, 0.8]}
view_transforms:
  - !<ViewTransform>
    name: Un-tone-mapped
    from_scene_reference: !<BuiltinTransform> {style: "UTILITY - ACES-AP1_to_CIE-XYZ-D65_BFD"}
colorspaces:
  - !<ColorSpace>
    name: Linear Rec.709
    aliases: [lin_rec709]
  - !<ColorSpace>
    name: sRGB
    from_scene_reference: !<ExponentWithLinearTransform> {gamma: 2.4, offset: 0.055, direction: inverse}
  - !<ColorSpace>
    name: Film Log
    from_scene_reference: !<GroupTransform>
      children:
        - !<LogAffineTransform> {base: 2, log_side_slope: 0.1, log_side_offset: 0.6}
        - !<FileTransform> {src: curve.spi1d, interpolation: linear}
  - !<ColorSpace>
    name: ACEScg
    to_scene_reference: !<MatrixTransform> {matrix: [1.705051, -0.621792, -0.083258, 0, -0.130256, 1.140805, -0.010548, 0, -0.024003, -0.128969, 1.152972, 0, 0, 0, 0, 1]}
  - !<ColorSpace>
    name: Raw
    isdata: true
display_colorspaces:
  - !<ColorSpace>
    name: Gamma 2.4 Rec.709 - Display
    from_display_reference: !<GroupTransform>
      children:
        - !<MatrixTransform> {matrix: [3.2409699, -1.5373832, -0.4986108, 0, -0.9692436, 1.8759675, 0.0415551, 0, 0.0556301, -0.2039770, 1.0569715, 0, 0, 0, 0, 1]}
        - !<ExponentTransform> {value: 2.4, direction: inverse}
"#;

    fn write_config(folder: &Path) -> PathBuf {
        std::fs::create_dir_all(folder.join("luts")).unwrap();
        // A 1D LUT that halves its input.
        std::fs::write(
            folder.join("luts/curve.spi1d"),
            "Version 1\nFrom 0.0 1.0\nLength 2\nComponents 1\n{\n0.0\n0.5\n}\n",
        )
        .unwrap();
        let path = folder.join("config.ocio");
        std::fs::write(&path, CONFIG).unwrap();
        path
    }

    #[test]
    fn tc_0390_ocio_configs_resolve_displays_views_and_colour_spaces() {
        let folder = tempfile::tempdir().unwrap();
        let config = OcioConfig::load(write_config(folder.path())).unwrap();
        let displays: Vec<&str> = config.displays().iter().map(|d| d.name.as_str()).collect();
        assert_eq!(displays, ["sRGB", "Rec.1886"]);
        let views: Vec<&str> = config.views("sRGB").iter().map(|v| v.name.as_str()).collect();
        assert_eq!(views, ["Standard", "Graded", "Film", "Raw"]);
        assert_eq!(config.linear_rec709_space(), Some("Linear Rec.709"));

        // sRGB view: linear 0.18 encodes to the sRGB curve's 0.4613.
        let srgb = config.view_processor("lin_rec709", "sRGB", "Standard").unwrap();
        let grey = srgb.apply([0.18; 3]);
        assert!(close(grey, [0.461_356; 3], 1e-4), "{grey:?}");
        assert!(close(srgb.apply([0.002; 3]), [0.002 * 12.92; 3], 1e-4));
        // Roles resolve too.
        assert!(config.view_processor("scene_linear", "sRGB", "Standard").is_ok());

        // A look applies in its process space before the display.
        let graded = config.view_processor("Linear Rec.709", "sRGB", "Graded").unwrap();
        let warm = graded.apply([0.18; 3]);
        assert!(warm[0] > grey[0] && warm[2] < grey[2]);

        // A log curve followed by a LUT file found on the search path.
        let film = config.view_processor("Linear Rec.709", "sRGB", "Film").unwrap();
        let expected = (0.1 * 0.18_f64.log2() + 0.6) * 0.5;
        assert!(close(film.apply([0.18; 3]), [expected; 3], 1e-6));

        // Data passes through untouched.
        let raw = config.view_processor("Linear Rec.709", "sRGB", "Raw").unwrap();
        assert!(raw.is_identity());

        // A v2 view transform with a display colour space, from ACEScg:
        // white stays white.
        let video = config.view_processor("ACEScg", "Rec.1886", "Standard").unwrap();
        let white = video.apply([1.0; 3]);
        assert!(close(white, [1.0; 3], 2e-3), "{white:?}");

        // Unknown names and unsupported transforms are named.
        assert!(matches!(
            config.view_processor("Nope", "sRGB", "Standard"),
            Err(OcioError::UnknownColorSpace(_))
        ));
        assert!(matches!(
            config.view_processor("ACEScg", "sRGB", "Nope"),
            Err(OcioError::UnknownView { .. })
        ));
        let odd = CONFIG.replace("!<CDLTransform> {slope: [1.2, 1, 0.8]}", "!<GradingToneTransform> {}");
        let odd = OcioConfig::parse(&odd, &folder.path().join("config.ocio")).unwrap();
        assert!(matches!(
            odd.view_processor("Linear Rec.709", "sRGB", "Graded"),
            Err(OcioError::Unsupported(message)) if message.contains("GradingToneTransform")
        ));
        assert!(OcioConfig::parse("hello: world", Path::new("x.ocio")).is_err());

        // Baked views follow the renderer's input: encoded sRGB of light
        // exposed down by the headroom.
        let lut = config.bake_view("Linear Rec.709", "sRGB", "Standard", 33).unwrap();
        assert_eq!(lut.size, 33);
        let headroom = 2f32.powf(OCIO_HEADROOM_STOPS);
        let input = substancecraft_core::color::linear_to_srgb(0.18 / headroom);
        let out = lut.apply(Vec3::splat(input));
        assert!((out.x - 0.4614).abs() < 4e-3, "{out:?}");
        // Highlights up to the headroom still reach the view.
        let top = lut.apply(Vec3::ONE);
        assert!(top.x > 1.5, "{top:?}");
    }

    #[test]
    fn tc_0391_ocio_transforms_invert_and_aces_builtins_behave() {
        // Each analytic op round-trips through its inverse.
        let ops = [
            Op::Matrix {
                m: [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.0, 0.1, 0.9]],
                offset: [0.01, 0.02, 0.03],
            },
            Op::Power {
                gamma: [2.2; 3],
                mirror: true,
            },
            Op::MonCurve {
                gamma: [2.4; 3],
                offset: [0.055; 3],
                inverse: false,
            },
            builtin::ops("ACEScct_to_ACES2065-1").unwrap()[0].clone(),
            Op::Cdl {
                slope: [1.1, 0.9, 1.0],
                offset: [0.01; 3],
                power: [1.2; 3],
                saturation: 0.8,
                clamp: true,
                inverse: false,
            },
            Op::Lut1d {
                table: Arc::new(vec![[0.0; 3], [0.2; 3], [1.0; 3]]),
                domain: (0.0, 1.0),
                inverse: false,
            },
        ];
        for op in ops {
            let inverse = op.inverse().unwrap();
            let value = [0.3, 0.45, 0.6];
            let back = inverse.apply(op.apply(value));
            assert!(close(back, value, 1e-5), "{op:?}: {back:?}");
        }
        assert!(
            Op::Lut3d {
                size: 2,
                data: Arc::new(vec![[0.0; 3]; 8]),
                domain_min: [0.0; 3],
                domain_max: [1.0; 3]
            }
            .inverse()
            .is_err()
        );

        // ACEScct 0.4135884 is linear 0.18.
        let cct = builtin::ops("ACEScct_to_ACES2065-1").unwrap();
        let lin = cct[0].apply([0.413_588_4; 3]);
        assert!(close(lin, [0.18; 3], 1e-5), "{lin:?}");

        // ACES 1.0 SDR video: scene grey lands near 10% of display white
        // (4.8 of 48 nits, then the dim-surround gamma); black stays dark,
        // bright light rolls off below white.
        let video = builtin::AcesOutput::Video;
        let grey = video.apply([0.18; 3]);
        assert!((grey[1] - 0.104).abs() < 0.01, "{grey:?}");
        assert!(video.apply([0.0; 3])[1] < 0.001);
        let bright = video.apply([16.0; 3]);
        assert!(bright[1] > 0.8 && bright[1] < 1.01, "{bright:?}");
        let cinema = builtin::AcesOutput::Cinema.apply([0.18; 3]);
        assert!((cinema[1] - 0.0996).abs() < 0.01, "{cinema:?}");
        assert!(builtin::ops("ACES2-OUTPUT - something").is_err());
    }

    #[test]
    fn ocio_lut_files_parse() {
        let path = Path::new("test.cube");
        let shaper = "LUT_1D_SIZE 2\nLUT_1D_INPUT_RANGE 0 2\nLUT_3D_SIZE 2\n0 0 0\n1 1 1\n\
                      0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let ops = luts::cube(shaper, path).unwrap();
        assert_eq!(ops.len(), 2);
        let processor = OcioProcessor { ops };
        assert!(close(processor.apply([1.0, 0.5, 0.0]), [0.5, 0.25, 0.0], 1e-9));
        assert!(luts::cube("LUT_3D_SIZE 2\n0 0 0\n", path).is_err());
        let spi3d = "SPILUT 1.0\n3 3\n2 2 2\n0 0 0 0 0 0\n1 0 0 1 0 0\n0 1 0 0 1 0\n1 1 0 1 1 0\n\
                     0 0 1 0 0 1\n1 0 1 1 0 1\n0 1 1 0 1 1\n1 1 1 1 1 1\n";
        let ops = luts::spi3d(spi3d, Path::new("t.spi3d")).unwrap();
        assert!(close(ops[0].apply([0.25, 0.5, 0.75]), [0.25, 0.5, 0.75], 1e-9));
        let ops = luts::spimtx("2 0 0 0\n0 2 0 0\n0 0 2 65535\n", Path::new("m.spimtx")).unwrap();
        assert!(close(ops[0].apply([0.5; 3]), [1.0, 1.0, 2.0], 1e-9));
    }
}
