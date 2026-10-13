//! STEP (ISO 10303-21; AP203, AP214, and AP242) mesh import. Exact B-rep
//! geometry (manifold and faceted solids, solids with voids, and shell- and
//! face-based surface models) is tessellated to a chord tolerance relative
//! to the model size; AP242 tessellated faces and surface sets are read as
//! they are. Assemblies are flattened with every instance placed (shape
//! representation relationships with item-defined transformations, and
//! mapped items), lengths are converted to metres, and each distinct surface
//! colour becomes a material slot. CAD files carry no UVs, so every face
//! gets its own chart, packed into one atlas per material slot.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Read;
use std::path::Path;

use glam::{DAffine3, DVec2, DVec3, Vec3};
use serde::{Deserialize, Serialize};
use substancecraft_core::{MeshGeometry, MeshVertex};
use thiserror::Error;

use crate::fbx::ImportedMesh;
use crate::step_face::{FacePatch, Loop3, newell_normal, pack_charts, soup_patches, tessellate_face};
use crate::step_geom::{BSplineCurve, BSplineSurface, Curve, Frame, Knots, Surface};
use crate::step_parse::{Instance, Param, Record, StepFile, parse_step};

pub const MAX_STEP_BYTES: usize = 512 * 1024 * 1024;
const MAX_STEP_VERTICES: usize = 3_000_000;
const MAX_PLACED_ITEMS: usize = 100_000;
const MAX_ASSEMBLY_DEPTH: usize = 64;
const MAX_REFERENCE_DEPTH: usize = 16;

/// File extensions read as STEP.
pub const STEP_EXTENSIONS: [&str; 2] = ["step", "stp"];

#[derive(Debug, Error, PartialEq, Eq)]
pub enum StepImportError {
    #[error("STEP file I/O failed: {0}")]
    Io(String),
    #[error("STEP file exceeds the {} MiB import limit", MAX_STEP_BYTES / (1024 * 1024))]
    TooLarge,
    #[error("STEP could not be parsed: {0}")]
    Parse(String),
    #[error("the STEP file contains no solid, shell, or tessellated geometry")]
    NoGeometry,
    #[error("no STEP face could be tessellated ({0})")]
    NoTriangles(String),
    #[error("the STEP tessellation exceeds the {MAX_STEP_VERTICES}-vertex import limit")]
    TooManyVertices,
    #[error("the STEP mesh is invalid: {0}")]
    InvalidMesh(String),
}

/// How finely exact geometry is tessellated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepTessellation {
    /// Largest distance between triangles and the exact surface, as a
    /// fraction of the model's bounding-box diagonal.
    pub chord_tolerance: f64,
    /// Largest angle, in degrees, that a curved edge turns across one
    /// segment.
    pub max_angle_degrees: f64,
}

impl Default for StepTessellation {
    fn default() -> Self {
        Self {
            chord_tolerance: 0.0005,
            max_angle_degrees: 12.0,
        }
    }
}

/// What an import read and what it had to skip.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StepImportReport {
    /// Bodies placed, counting every assembly instance.
    pub bodies: usize,
    /// Faces tessellated, counting every instance.
    pub faces: usize,
    /// Faces that could not be tessellated.
    pub skipped_faces: usize,
    /// Unsupported curve and surface types, with how often they were met.
    pub unsupported: BTreeMap<String, usize>,
    /// Why faces of supported types could not be tessellated, with how
    /// often.
    pub failures: BTreeMap<String, usize>,
    /// The length unit of the file, in metres.
    pub length_unit: f64,
    /// The chord tolerance used, in metres.
    pub tolerance: f64,
}

/// Imports a `.step` or `.stp` file with the default tessellation.
pub fn import_step_file(path: impl AsRef<Path>) -> Result<ImportedMesh, StepImportError> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|error| StepImportError::Io(error.to_string()))?;
    let mut bytes = Vec::new();
    file.take(MAX_STEP_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| StepImportError::Io(error.to_string()))?;
    let name = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "STEP".to_owned());
    import_step(&bytes, name)
}

/// Imports STEP bytes as one mesh with the default tessellation.
pub fn import_step(bytes: &[u8], name: String) -> Result<ImportedMesh, StepImportError> {
    import_step_with_report(bytes, name, &StepTessellation::default()).map(|(mesh, _)| mesh)
}

/// Imports STEP bytes, reporting what was read and skipped.
pub fn import_step_with_report(
    bytes: &[u8],
    name: String,
    tessellation: &StepTessellation,
) -> Result<(ImportedMesh, StepImportReport), StepImportError> {
    if bytes.len() > MAX_STEP_BYTES {
        return Err(StepImportError::TooLarge);
    }
    let file = parse_step(bytes).map_err(|error| StepImportError::Parse(error.to_string()))?;
    let mut reader = Reader::new(&file, tessellation);
    let placements = reader.placements();
    if placements.is_empty() {
        return Err(StepImportError::NoGeometry);
    }
    let unit = placements[0].unit;
    reader.report.length_unit = unit;
    let size = reader.model_size() * unit;
    let tolerance = if size.is_finite() && size > 0.0 {
        size * tessellation.chord_tolerance.clamp(1e-6, 0.1)
    } else {
        1e-4
    };
    reader.report.tolerance = tolerance;

    let mut cache: HashMap<u64, Vec<ColouredPatch>> = HashMap::new();
    let mut slots: Vec<Option<[u8; 3]>> = Vec::new();
    let mut slot_patches: Vec<Vec<FacePatch>> = Vec::new();
    let mut vertex_count = 0_usize;
    for placement in &placements {
        if !cache.contains_key(&placement.item) {
            reader.tolerance = tolerance / placement.unit;
            let patches = reader.item_patches(placement.item);
            cache.insert(placement.item, patches);
        }
        let Some(patches) = cache.get(&placement.item) else {
            continue;
        };
        if patches.is_empty() {
            continue;
        }
        reader.report.bodies += 1;
        let flips = placement.transform.matrix3.determinant() < 0.0;
        for coloured in patches {
            reader.report.faces += 1;
            vertex_count += coloured.patch.positions.len();
            if vertex_count > MAX_STEP_VERTICES {
                return Err(StepImportError::TooManyVertices);
            }
            let colour = coloured.colour.or(placement.colour);
            let slot = match slots.iter().position(|existing| *existing == colour) {
                Some(slot) => slot,
                None => {
                    slots.push(colour);
                    slot_patches.push(Vec::new());
                    slots.len() - 1
                }
            };
            let mut patch = FacePatch {
                positions: coloured
                    .patch
                    .positions
                    .iter()
                    .map(|position| placement.transform.transform_point3(*position))
                    .collect(),
                chart: coloured
                    .patch
                    .chart
                    .iter()
                    .map(|point| *point * placement.unit)
                    .collect(),
                triangles: coloured.patch.triangles.clone(),
            };
            if flips {
                patch.flip();
                for point in &mut patch.chart {
                    point.x = -point.x;
                }
            }
            slot_patches[slot].push(patch);
        }
    }
    if slot_patches.iter().all(Vec::is_empty) {
        return Err(StepImportError::NoTriangles(skipped_summary(&reader.report)));
    }

    let mut geometry = MeshGeometry::default();
    let coloured = slots.iter().any(Option::is_some);
    for (slot, patches) in slot_patches.iter().enumerate() {
        let charts: Vec<&[DVec2]> = patches.iter().map(|patch| patch.chart.as_slice()).collect();
        let uvs = pack_charts(&charts);
        for (patch, uvs) in patches.iter().zip(uvs) {
            let offset = geometry.vertices.len() as u32;
            geometry
                .vertices
                .extend(patch.positions.iter().zip(uvs).map(|(position, uv)| MeshVertex {
                    position: Vec3::new(position.x as f32, position.y as f32, position.z as f32),
                    uv,
                }));
            for triangle in &patch.triangles {
                geometry
                    .triangles
                    .push(triangle.map(|corner| corner + offset));
                if coloured {
                    geometry.triangle_materials.push(slot as u32);
                }
            }
        }
    }
    if coloured {
        geometry.material_slots = slots
            .iter()
            .map(|colour| match colour {
                Some([r, g, b]) => format!("Colour {r:02X}{g:02X}{b:02X}"),
                None => "Default".to_owned(),
            })
            .collect();
    }
    geometry
        .validate()
        .map_err(|error| StepImportError::InvalidMesh(error.to_string()))?;
    Ok((
        ImportedMesh {
            name,
            geometry,
            generated_uvs: true,
            rig: None,
        },
        reader.report,
    ))
}

fn skipped_summary(report: &StepImportReport) -> String {
    let mut summary = format!("{} faces skipped", report.skipped_faces);
    let causes: Vec<&str> = report
        .unsupported
        .keys()
        .chain(report.failures.keys())
        .map(String::as_str)
        .collect();
    if !causes.is_empty() {
        summary.push_str(": ");
        summary.push_str(&causes.join("; "));
    }
    summary
}

struct ColouredPatch {
    patch: FacePatch,
    colour: Option<[u8; 3]>,
}

/// A body to emit: a representation item with its transform from model
/// units to metres in the world, and an inherited colour.
struct Placement {
    item: u64,
    transform: DAffine3,
    unit: f64,
    colour: Option<[u8; 3]>,
}

const BODY_TYPES: [&str; 15] = [
    "MANIFOLD_SOLID_BREP",
    "BREP_WITH_VOIDS",
    "FACETED_BREP",
    "SHELL_BASED_SURFACE_MODEL",
    "FACE_BASED_SURFACE_MODEL",
    "CLOSED_SHELL",
    "OPEN_SHELL",
    "ADVANCED_FACE",
    "FACE_SURFACE",
    "TESSELLATED_SOLID",
    "TESSELLATED_SHELL",
    "TRIANGULATED_FACE",
    "COMPLEX_TRIANGULATED_FACE",
    "TRIANGULATED_SURFACE_SET",
    "COMPLEX_TRIANGULATED_SURFACE_SET",
];

const TESSELLATED_TYPES: [&str; 6] = [
    "TESSELLATED_SOLID",
    "TESSELLATED_SHELL",
    "TRIANGULATED_FACE",
    "COMPLEX_TRIANGULATED_FACE",
    "TRIANGULATED_SURFACE_SET",
    "COMPLEX_TRIANGULATED_SURFACE_SET",
];

fn si_prefix(prefix: Option<&str>) -> f64 {
    match prefix {
        Some("EXA") => 1e18,
        Some("PETA") => 1e15,
        Some("TERA") => 1e12,
        Some("GIGA") => 1e9,
        Some("MEGA") => 1e6,
        Some("KILO") => 1e3,
        Some("HECTO") => 1e2,
        Some("DECA") => 1e1,
        Some("DECI") => 1e-1,
        Some("CENTI") => 1e-2,
        Some("MILLI") => 1e-3,
        Some("MICRO") => 1e-6,
        Some("NANO") => 1e-9,
        Some("PICO") => 1e-12,
        Some("FEMTO") => 1e-15,
        _ => 1.0,
    }
}

struct UnionFind(Vec<usize>);

impl UnionFind {
    fn find(&mut self, mut node: usize) -> usize {
        while self.0[node] != node {
            self.0[node] = self.0[self.0[node]];
            node = self.0[node];
        }
        node
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            // The lower index stays root, so groups are stable.
            let (low, high) = (a.min(b), a.max(b));
            self.0[high] = low;
        }
    }
}

struct Reader<'a> {
    file: &'a StepFile,
    /// Sorted instance ids, for deterministic traversal.
    ids: Vec<u64>,
    angle_unit: Option<f64>,
    /// Chord tolerance in the current body's model units.
    tolerance: f64,
    max_angle: f64,
    edges: HashMap<u64, Vec<DVec3>>,
    colours: HashMap<u64, [u8; 3]>,
    report: StepImportReport,
}

impl<'a> Reader<'a> {
    fn new(file: &'a StepFile, tessellation: &StepTessellation) -> Self {
        let mut ids: Vec<u64> = file.instances.keys().copied().collect();
        ids.sort_unstable();
        let mut reader = Self {
            file,
            ids,
            angle_unit: None,
            tolerance: 1e-3,
            max_angle: tessellation.max_angle_degrees.clamp(1.0, 90.0).to_radians(),
            edges: HashMap::new(),
            colours: HashMap::new(),
            report: StepImportReport::default(),
        };
        reader.collect_colours();
        reader
    }

    fn get(&self, id: u64) -> Option<&'a Instance> {
        self.file.get(id)
    }

    /// The record of a simple instance, with its name.
    fn simple(&self, id: u64) -> Option<(&'a str, &'a Record)> {
        let instance = self.get(id)?;
        let name = instance.name()?;
        Some((name, &instance.records[0]))
    }

    fn unsupported(&mut self, name: &str) {
        *self.report.unsupported.entry(name.to_owned()).or_default() += 1;
    }

    fn failure(&mut self, reason: &str) {
        *self.report.failures.entry(reason.to_owned()).or_default() += 1;
    }

    fn point(&self, id: u64) -> Option<DVec3> {
        let (name, record) = self.simple(id)?;
        if name != "CARTESIAN_POINT" {
            return None;
        }
        let coordinates = record.param(1)?.numbers()?;
        let point = DVec3::new(
            *coordinates.first()?,
            coordinates.get(1).copied().unwrap_or(0.0),
            coordinates.get(2).copied().unwrap_or(0.0),
        );
        point.is_finite().then_some(point)
    }

    fn vertex(&self, id: u64) -> Option<DVec3> {
        let (name, record) = self.simple(id)?;
        match name {
            "VERTEX_POINT" => self.point(record.reference(1)?),
            "CARTESIAN_POINT" => self.point(id),
            _ => None,
        }
    }

    fn direction(&self, id: u64) -> Option<DVec3> {
        let (name, record) = self.simple(id)?;
        match name {
            "DIRECTION" => {
                let ratios = record.param(1)?.numbers()?;
                let direction = DVec3::new(
                    *ratios.first()?,
                    ratios.get(1).copied().unwrap_or(0.0),
                    ratios.get(2).copied().unwrap_or(0.0),
                );
                direction.try_normalize()
            }
            "VECTOR" => self.direction(record.reference(1)?),
            _ => None,
        }
    }

    /// A `vector` with its magnitude, or a unit `direction`.
    fn vector(&self, id: u64) -> Option<DVec3> {
        let (name, record) = self.simple(id)?;
        match name {
            "VECTOR" => Some(self.direction(record.reference(1)?)? * record.number(2)?),
            "DIRECTION" => self.direction(id),
            _ => None,
        }
    }

    fn frame(&self, id: u64) -> Option<Frame> {
        let instance = self.get(id)?;
        if let Some(record) = instance.simple("AXIS2_PLACEMENT_3D") {
            let origin = self.point(record.reference(1)?)?;
            let axis = record.reference(2).and_then(|id| self.direction(id));
            let reference = record.reference(3).and_then(|id| self.direction(id));
            return Some(Frame::new(origin, axis, reference));
        }
        if let Some(record) = instance.simple("AXIS1_PLACEMENT") {
            let origin = self.point(record.reference(1)?)?;
            let axis = record.reference(2).and_then(|id| self.direction(id));
            return Some(Frame::new(origin, axis, None));
        }
        if let Some(record) = instance
            .simple("CARTESIAN_TRANSFORMATION_OPERATOR_3D")
            .or_else(|| instance.record("CARTESIAN_TRANSFORMATION_OPERATOR"))
        {
            // Writers differ in the leading name and description, so count
            // from the end: (..., axis1, axis2, local_origin, scale, axis3).
            let from_end = |back: usize| {
                record
                    .params
                    .len()
                    .checked_sub(back)
                    .and_then(|index| record.reference(index))
            };
            let origin = from_end(3)
                .and_then(|id| self.point(id))
                .unwrap_or(DVec3::ZERO);
            let x = from_end(5).and_then(|id| self.direction(id));
            let z = from_end(1).and_then(|id| self.direction(id));
            return Some(Frame::new(origin, z, x));
        }
        None
    }

    fn knots(&self, degree: usize, count: usize, multiplicities: &Param, values: &Param) -> Option<Knots> {
        let multiplicities = multiplicities.numbers()?;
        let values = values.numbers()?;
        if multiplicities.len() != values.len() {
            return None;
        }
        let mut expanded = Vec::new();
        for (multiplicity, value) in multiplicities.iter().zip(&values) {
            if !(0.0..=64.0).contains(multiplicity) {
                return None;
            }
            expanded.extend(std::iter::repeat_n(*value, *multiplicity as usize));
        }
        Knots::new(degree, expanded, count)
    }

    fn generated_knots(name: &str, degree: usize, count: usize) -> Option<Knots> {
        match name {
            "BEZIER_CURVE" | "BEZIER_SURFACE" => Knots::bezier(degree, count),
            "UNIFORM_CURVE" | "UNIFORM_SURFACE" => Knots::uniform(degree, count),
            _ => Knots::quasi_uniform(degree, count),
        }
    }

    fn points(&self, param: &Param) -> Option<Vec<DVec3>> {
        param
            .references()
            .into_iter()
            .map(|id| self.point(id))
            .collect()
    }

    fn curve(&mut self, id: u64, depth: usize) -> Option<Curve> {
        if depth > MAX_REFERENCE_DEPTH {
            return None;
        }
        let instance = self.get(id)?;
        if instance.complex {
            return self.complex_b_spline_curve(instance);
        }
        let record = &instance.records[0];
        let curve = match record.name.as_str() {
            "LINE" => Curve::Line {
                origin: self.point(record.reference(1)?)?,
                direction: self.vector(record.reference(2)?)?,
            },
            "CIRCLE" => Curve::Circle {
                frame: self.frame(record.reference(1)?)?,
                radius: record.number(2)?,
            },
            "ELLIPSE" => Curve::Ellipse {
                frame: self.frame(record.reference(1)?)?,
                a: record.number(2)?,
                b: record.number(3)?,
            },
            "B_SPLINE_CURVE_WITH_KNOTS" => {
                let degree = record.number(1)? as usize;
                let points = self.points(record.param(2)?)?;
                let knots = self.knots(degree, points.len(), record.param(6)?, record.param(7)?)?;
                Curve::BSpline(BSplineCurve::new(knots, points, None)?)
            }
            "BEZIER_CURVE" | "UNIFORM_CURVE" | "QUASI_UNIFORM_CURVE" => {
                let degree = record.number(1)? as usize;
                let points = self.points(record.param(2)?)?;
                let knots = Self::generated_knots(&record.name, degree, points.len())?;
                Curve::BSpline(BSplineCurve::new(knots, points, None)?)
            }
            "POLYLINE" => Curve::Polyline(self.points(record.param(1)?)?),
            "TRIMMED_CURVE" => return self.curve(record.reference(1)?, depth + 1),
            "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" | "BOUNDED_SURFACE_CURVE" => {
                return self.curve(record.reference(1)?, depth + 1);
            }
            other => {
                let other = other.to_owned();
                self.unsupported(&other);
                return None;
            }
        };
        Some(curve)
    }

    fn complex_b_spline_curve(&mut self, instance: &Instance) -> Option<Curve> {
        let Some(base) = instance.record("B_SPLINE_CURVE") else {
            let names: Vec<&str> = instance.records.iter().map(|r| r.name.as_str()).collect();
            self.unsupported(&format!("({})", names.join(" ")));
            return None;
        };
        let degree = base.number(0)? as usize;
        let points = self.points(base.param(1)?)?;
        let knots = match instance.record("B_SPLINE_CURVE_WITH_KNOTS") {
            Some(record) => self.knots(degree, points.len(), record.param(0)?, record.param(1)?)?,
            None => {
                let kind = ["BEZIER_CURVE", "UNIFORM_CURVE", "QUASI_UNIFORM_CURVE"]
                    .into_iter()
                    .find(|name| instance.has(name))
                    .unwrap_or("QUASI_UNIFORM_CURVE");
                Self::generated_knots(kind, degree, points.len())?
            }
        };
        let weights = match instance.record("RATIONAL_B_SPLINE_CURVE") {
            Some(record) => Some(record.param(0)?.numbers()?),
            None => None,
        };
        Some(Curve::BSpline(BSplineCurve::new(knots, points, weights)?))
    }

    fn control_grid(&self, param: &Param) -> Option<Vec<Vec<DVec3>>> {
        param.list()?.iter().map(|row| self.points(row)).collect()
    }

    fn surface(&mut self, id: u64, depth: usize) -> Option<Surface> {
        if depth > MAX_REFERENCE_DEPTH {
            return None;
        }
        let instance = self.get(id)?;
        if instance.complex {
            return self.complex_b_spline_surface(instance);
        }
        let record = &instance.records[0];
        let surface = match record.name.as_str() {
            "PLANE" => Surface::Plane(self.frame(record.reference(1)?)?),
            "CYLINDRICAL_SURFACE" => Surface::Cylinder {
                frame: self.frame(record.reference(1)?)?,
                radius: record.number(2)?,
            },
            "CONICAL_SURFACE" => {
                let value = record.number(3)?;
                let angle = match self.angle_unit {
                    Some(unit) => value * unit,
                    // Semi-angles are below 90 degrees; larger values can
                    // only be degrees.
                    None if value.abs() >= std::f64::consts::FRAC_PI_2 => value.to_radians(),
                    None => value,
                };
                Surface::Cone {
                    frame: self.frame(record.reference(1)?)?,
                    radius: record.number(2)?,
                    tan: angle.tan(),
                }
            }
            "SPHERICAL_SURFACE" => Surface::Sphere {
                frame: self.frame(record.reference(1)?)?,
                radius: record.number(2)?,
            },
            "TOROIDAL_SURFACE" | "DEGENERATE_TOROIDAL_SURFACE" => Surface::Torus {
                frame: self.frame(record.reference(1)?)?,
                major: record.number(2)?,
                minor: record.number(3)?,
            },
            "B_SPLINE_SURFACE_WITH_KNOTS" => {
                let (du, dv) = (record.number(1)? as usize, record.number(2)? as usize);
                let grid = self.control_grid(record.param(3)?)?;
                let (nu, nv) = (grid.len(), grid.first()?.len());
                let u = self.knots(du, nu, record.param(8)?, record.param(10)?)?;
                let v = self.knots(dv, nv, record.param(9)?, record.param(11)?)?;
                Surface::BSpline(Box::new(BSplineSurface::new(u, v, grid, None)?))
            }
            "BEZIER_SURFACE" | "UNIFORM_SURFACE" | "QUASI_UNIFORM_SURFACE" => {
                let (du, dv) = (record.number(1)? as usize, record.number(2)? as usize);
                let grid = self.control_grid(record.param(3)?)?;
                let (nu, nv) = (grid.len(), grid.first()?.len());
                let u = Self::generated_knots(&record.name, du, nu)?;
                let v = Self::generated_knots(&record.name, dv, nv)?;
                Surface::BSpline(Box::new(BSplineSurface::new(u, v, grid, None)?))
            }
            "SURFACE_OF_LINEAR_EXTRUSION" => Surface::Extrusion {
                curve: self.curve(record.reference(1)?, depth + 1)?,
                direction: self.vector(record.reference(2)?)?,
            },
            "SURFACE_OF_REVOLUTION" => {
                let curve = self.curve(record.reference(1)?, depth + 1)?;
                let axis = self.frame(record.reference(2)?)?;
                Surface::Revolution {
                    curve,
                    origin: axis.origin,
                    axis: axis.z,
                }
            }
            "OFFSET_SURFACE" => Surface::Offset {
                basis: Box::new(self.surface(record.reference(1)?, depth + 1)?),
                distance: record.number(2)?,
            },
            "RECTANGULAR_TRIMMED_SURFACE" | "CURVE_BOUNDED_SURFACE" => {
                return self.surface(record.reference(1)?, depth + 1);
            }
            other => {
                let other = other.to_owned();
                self.unsupported(&other);
                return None;
            }
        };
        Some(surface)
    }

    fn complex_b_spline_surface(&mut self, instance: &Instance) -> Option<Surface> {
        let Some(base) = instance.record("B_SPLINE_SURFACE") else {
            let names: Vec<&str> = instance.records.iter().map(|r| r.name.as_str()).collect();
            self.unsupported(&format!("({})", names.join(" ")));
            return None;
        };
        let (du, dv) = (base.number(0)? as usize, base.number(1)? as usize);
        let grid = self.control_grid(base.param(2)?)?;
        let (nu, nv) = (grid.len(), grid.first()?.len());
        let (u, v) = match instance.record("B_SPLINE_SURFACE_WITH_KNOTS") {
            Some(record) => (
                self.knots(du, nu, record.param(0)?, record.param(2)?)?,
                self.knots(dv, nv, record.param(1)?, record.param(3)?)?,
            ),
            None => {
                let kind = ["BEZIER_SURFACE", "UNIFORM_SURFACE", "QUASI_UNIFORM_SURFACE"]
                    .into_iter()
                    .find(|name| instance.has(name))
                    .unwrap_or("QUASI_UNIFORM_SURFACE");
                (
                    Self::generated_knots(kind, du, nu)?,
                    Self::generated_knots(kind, dv, nv)?,
                )
            }
        };
        let weights = match instance.record("RATIONAL_B_SPLINE_SURFACE") {
            Some(record) => Some(
                record
                    .param(0)?
                    .list()?
                    .iter()
                    .map(Param::numbers)
                    .collect::<Option<Vec<_>>>()?,
            ),
            None => None,
        };
        Some(Surface::BSpline(Box::new(BSplineSurface::new(
            u, v, grid, weights,
        )?)))
    }

    /// Points of an edge from its start vertex to its end vertex, sampled
    /// once and shared by both faces that use it.
    fn edge_points(&mut self, id: u64) -> Option<Vec<DVec3>> {
        if let Some(points) = self.edges.get(&id) {
            return Some(points.clone());
        }
        let (name, record) = self.simple(id)?;
        if name != "EDGE_CURVE" {
            return None;
        }
        let start_id = record.reference(1)?;
        let end_id = record.reference(2)?;
        let start = self.vertex(start_id)?;
        let end = self.vertex(end_id)?;
        let same_sense = record.logical(4).unwrap_or(true);
        let closed = start_id == end_id || start.distance(end) <= self.tolerance * 1e-3;
        let points = match record.reference(3).and_then(|curve| self.curve(curve, 0)) {
            Some(curve) => self.sample_edge(&curve, start, end, same_sense, closed),
            None => vec![start, end],
        };
        self.edges.insert(id, points.clone());
        Some(points)
    }

    fn sample_edge(
        &self,
        curve: &Curve,
        start: DVec3,
        end: DVec3,
        same_sense: bool,
        closed: bool,
    ) -> Vec<DVec3> {
        let mut t0 = curve.parameter_of(start);
        let mut t1 = curve.parameter_of(end);
        if let Some(period) = curve.period() {
            let epsilon = 1e-9;
            if same_sense {
                if closed || t1 <= t0 + epsilon {
                    t1 = if closed { t0 + period } else { t1 + period };
                }
            } else if closed || t1 >= t0 - epsilon {
                t1 = if closed { t0 - period } else { t1 - period };
            }
        } else if closed {
            match curve.domain() {
                Some((low, high))
                    if curve.point(low).distance(curve.point(high)) <= self.tolerance =>
                {
                    (t0, t1) = if same_sense { (low, high) } else { (high, low) };
                }
                // A degenerate edge.
                _ => return vec![start],
            }
        }
        let params = curve.sample(t0, t1, self.tolerance, self.max_angle);
        let mut points: Vec<DVec3> = params.iter().map(|t| curve.point(*t)).collect();
        if let Some(first) = points.first_mut() {
            *first = start;
        }
        if let Some(last) = points.last_mut() {
            *last = end;
        }
        points
    }

    /// One face bound as model-space points in loop order.
    fn bound(&mut self, id: u64) -> Option<Loop3> {
        let (name, record) = self.simple(id)?;
        let (loop_id, orientation) = match name {
            "FACE_OUTER_BOUND" | "FACE_BOUND" => {
                (record.reference(1)?, record.logical(2).unwrap_or(true))
            }
            _ => (id, true),
        };
        let (loop_name, loop_record) = self.simple(loop_id)?;
        let mut points = match loop_name {
            "VERTEX_LOOP" => return Some(Loop3::Vertex(self.vertex(loop_record.reference(1)?)?)),
            "POLY_LOOP" => self.points(loop_record.param(1)?)?,
            "EDGE_LOOP" => {
                let mut points: Vec<DVec3> = Vec::new();
                for edge in loop_record.references(1) {
                    let (edge_id, forward) = self.oriented_edge(edge, 0)?;
                    let mut edge_points = self.edge_points(edge_id)?;
                    if !forward {
                        edge_points.reverse();
                    }
                    let join = points.last().zip(edge_points.first()).is_some_and(|(a, b)| {
                        a.distance(*b) <= self.tolerance * 1e-2
                    });
                    points.extend(edge_points.into_iter().skip(usize::from(join)));
                }
                if points.len() > 1
                    && points
                        .first()
                        .zip(points.last())
                        .is_some_and(|(a, b)| a.distance(*b) <= self.tolerance * 1e-2)
                {
                    points.pop();
                }
                points
            }
            other => {
                let other = other.to_owned();
                self.unsupported(&other);
                return None;
            }
        };
        if !orientation {
            points.reverse();
        }
        Some(Loop3::Points(points))
    }

    /// The edge curve of an `oriented_edge`, with its direction.
    fn oriented_edge(&self, id: u64, depth: usize) -> Option<(u64, bool)> {
        if depth > 4 {
            return None;
        }
        let (name, record) = self.simple(id)?;
        match name {
            "ORIENTED_EDGE" => {
                let (inner, forward) = self.oriented_edge(record.reference(3)?, depth + 1)?;
                Some((inner, forward == record.logical(4).unwrap_or(true)))
            }
            "EDGE_CURVE" => Some((id, true)),
            _ => None,
        }
    }

    fn face(&mut self, id: u64, flip: bool, depth: usize) -> Option<ColouredPatch> {
        if depth > 4 {
            return None;
        }
        let (name, record) = self.simple(id)?;
        if name == "ORIENTED_FACE" {
            let inner = record.reference(3)?;
            let mut patch = self.face(inner, flip != !record.logical(4).unwrap_or(true), depth + 1)?;
            patch.colour = self.colours.get(&id).copied().or(patch.colour);
            return Some(patch);
        }
        if !matches!(name, "ADVANCED_FACE" | "FACE_SURFACE" | "FACE") {
            return None;
        }
        let mut bounds = Vec::new();
        for bound in record.references(1) {
            match self.bound(bound) {
                Some(bound) => bounds.push(bound),
                None => self.failure("unreadable face bound"),
            }
        }
        let (surface, same_sense) = if name == "FACE" {
            (None, true)
        } else {
            (
                record.reference(2).and_then(|surface| self.surface(surface, 0)),
                record.logical(3).unwrap_or(true),
            )
        };
        let surface = match surface {
            Some(surface) => surface,
            None if name == "FACE" || record.reference(2).is_none() => {
                // A polygon face with no surface: its own plane.
                let Some(Loop3::Points(outer)) = bounds.first() else {
                    return None;
                };
                let normal = newell_normal(outer);
                let origin = outer.first().copied().unwrap_or_default();
                Surface::Plane(Frame::new(origin, Some(normal), None))
            }
            None => {
                let surface = record.reference(2).and_then(|surface| self.simple(surface));
                let name = surface.map_or("?", |(name, _)| name);
                self.failure(&format!("unreadable {name}"));
                return None;
            }
        };
        let result = tessellate_face(&surface, same_sense != flip, &bounds, self.tolerance);
        if std::env::var_os("STEP_DEBUG").is_some() {
            let kind = format!("{surface:?}").split(['(', ' ', '{']).next().unwrap_or("").to_owned();
            let loops: Vec<String> = bounds
                .iter()
                .map(|b| match b {
                    Loop3::Points(p) => format!("{}", p.len()),
                    Loop3::Vertex(_) => "v".to_owned(),
                })
                .collect();
            match &result {
                Ok(patch) => eprintln!("#{id} {kind} loops {loops:?} tris {}", patch.triangles.len()),
                Err(reason) => eprintln!("#{id} {kind} loops {loops:?} ERR {reason}"),
            }
        }
        match result {
            Ok(patch) => Some(ColouredPatch {
                patch,
                colour: self.colours.get(&id).copied(),
            }),
            Err(reason) => {
                self.failure(reason);
                None
            }
        }
    }

    fn shell(&mut self, id: u64, flip: bool, depth: usize) -> Vec<ColouredPatch> {
        if depth > 4 {
            return Vec::new();
        }
        let Some((name, record)) = self.simple(id) else {
            return Vec::new();
        };
        if matches!(name, "ORIENTED_CLOSED_SHELL" | "ORIENTED_OPEN_SHELL") {
            let Some(inner) = record.reference(2) else {
                return Vec::new();
            };
            let forward = record.logical(3).unwrap_or(true);
            let mut patches = self.shell(inner, flip == forward, depth + 1);
            if let Some(colour) = self.colours.get(&id) {
                for patch in &mut patches {
                    patch.colour.get_or_insert(*colour);
                }
            }
            return patches;
        }
        if !matches!(name, "CLOSED_SHELL" | "OPEN_SHELL" | "CONNECTED_FACE_SET") {
            return Vec::new();
        }
        let shell_colour = self.colours.get(&id).copied();
        let mut patches = Vec::new();
        for face in record.references(1) {
            match self.face(face, flip, 0) {
                Some(mut patch) => {
                    patch.colour = patch.colour.or(shell_colour);
                    patches.push(patch);
                }
                None => self.report.skipped_faces += 1,
            }
        }
        if name == "CLOSED_SHELL" {
            // A closed shell encloses positive volume when its faces point
            // out; repair files whose orientation flags disagree.
            let volume: f64 = patches.iter().map(|p| p.patch.signed_volume()).sum();
            let expected = if flip { -1.0 } else { 1.0 };
            if volume * expected < 0.0 {
                for patch in &mut patches {
                    patch.patch.flip();
                    for point in &mut patch.patch.chart {
                        point.x = -point.x;
                    }
                }
            }
        }
        patches
    }

    /// Tessellates one body item.
    fn item_patches(&mut self, id: u64) -> Vec<ColouredPatch> {
        let Some((name, record)) = self.simple(id) else {
            return Vec::new();
        };
        let mut patches = match name {
            "MANIFOLD_SOLID_BREP" | "FACETED_BREP" => match record.reference(1) {
                Some(shell) => self.shell(shell, false, 0),
                None => Vec::new(),
            },
            "BREP_WITH_VOIDS" => {
                let mut patches = match record.reference(1) {
                    Some(shell) => self.shell(shell, false, 0),
                    None => Vec::new(),
                };
                for void in record.references(2) {
                    patches.extend(self.shell(void, false, 0));
                }
                patches
            }
            "SHELL_BASED_SURFACE_MODEL" | "FACE_BASED_SURFACE_MODEL" => record
                .references(1)
                .into_iter()
                .flat_map(|shell| self.shell(shell, false, 0))
                .collect(),
            "CLOSED_SHELL" | "OPEN_SHELL" => self.shell(id, false, 0),
            "ADVANCED_FACE" | "FACE_SURFACE" => match self.face(id, false, 0) {
                Some(patch) => vec![patch],
                None => {
                    self.report.skipped_faces += 1;
                    Vec::new()
                }
            },
            "TESSELLATED_SOLID" | "TESSELLATED_SHELL" => record
                .references(1)
                .into_iter()
                .flat_map(|item| self.tessellated(item))
                .collect(),
            _ => self.tessellated(id),
        };
        if let Some(colour) = self.colours.get(&id) {
            for patch in &mut patches {
                patch.colour.get_or_insert(*colour);
            }
        }
        patches
    }

    /// An AP242 triangulated face or surface set.
    fn tessellated(&mut self, id: u64) -> Vec<ColouredPatch> {
        let Some((name, record)) = self.simple(id) else {
            return Vec::new();
        };
        // (index of pnindex, index of triangles or strips, index of fans)
        let (pnindex, triangles, fans) = match name {
            "TRIANGULATED_FACE" => (5, 6, None),
            "COMPLEX_TRIANGULATED_FACE" => (5, 6, Some(7)),
            "TRIANGULATED_SURFACE_SET" => (4, 5, None),
            "COMPLEX_TRIANGULATED_SURFACE_SET" => (4, 5, Some(6)),
            _ => return Vec::new(),
        };
        let Some(coordinates) = record
            .reference(1)
            .and_then(|list| self.coordinates(list))
        else {
            return Vec::new();
        };
        let indices: Vec<usize> = record
            .param(pnindex)
            .and_then(Param::numbers)
            .unwrap_or_default()
            .into_iter()
            .map(|index| index as usize)
            .collect();
        let resolve = |index: f64| -> Option<u32> {
            let index = (index as usize).checked_sub(1)?;
            let coordinate = if indices.is_empty() {
                index
            } else {
                indices.get(index)?.checked_sub(1)?
            };
            (coordinate < coordinates.len()).then_some(coordinate as u32)
        };
        let lists = |param: Option<&Param>| -> Vec<Vec<u32>> {
            param
                .and_then(Param::list)
                .map(|lists| {
                    lists
                        .iter()
                        .filter_map(Param::numbers)
                        .map(|list| list.into_iter().filter_map(resolve).collect())
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut faces = Vec::new();
        let first = lists(record.param(triangles));
        if fans.is_some() {
            for strip in &first {
                for k in 2..strip.len() {
                    faces.push(if k % 2 == 0 {
                        [strip[k - 2], strip[k - 1], strip[k]]
                    } else {
                        [strip[k - 1], strip[k - 2], strip[k]]
                    });
                }
            }
            for fan in lists(fans.and_then(|index| record.param(index))) {
                for k in 2..fan.len() {
                    faces.push([fan[0], fan[k - 1], fan[k]]);
                }
            }
        } else {
            faces.extend(first.iter().filter(|t| t.len() == 3).map(|t| [t[0], t[1], t[2]]));
        }
        faces.retain(|[a, b, c]| a != b && b != c && a != c);
        let colour = self.colours.get(&id).copied();
        soup_patches(&coordinates, &faces)
            .into_iter()
            .map(|patch| ColouredPatch { patch, colour })
            .collect()
    }

    fn coordinates(&self, id: u64) -> Option<Vec<DVec3>> {
        let (name, record) = self.simple(id)?;
        if name != "COORDINATES_LIST" {
            return None;
        }
        record
            .param(2)?
            .list()?
            .iter()
            .map(|point| {
                let values = point.numbers()?;
                Some(DVec3::new(
                    *values.first()?,
                    *values.get(1)?,
                    values.get(2).copied().unwrap_or(0.0),
                ))
            })
            .collect()
    }

    /// Diagonal of the bounding box of every vertex (or tessellation
    /// coordinate), in model units.
    fn model_size(&self) -> f64 {
        let mut low = DVec3::INFINITY;
        let mut high = DVec3::NEG_INFINITY;
        for id in &self.ids {
            let Some((name, _)) = self.simple(*id) else {
                continue;
            };
            let points = match name {
                "VERTEX_POINT" => self.vertex(*id).into_iter().collect(),
                "COORDINATES_LIST" => self.coordinates(*id).unwrap_or_default(),
                _ => Vec::new(),
            };
            for point in points {
                low = low.min(point);
                high = high.max(point);
            }
        }
        if low.x > high.x {
            return 0.0;
        }
        (high - low).length()
    }

    fn collect_colours(&mut self) {
        // Over-riding styles win, so they are applied last.
        for over_riding in [false, true] {
            for id in &self.ids {
                let Some((name, record)) = self.simple(*id) else {
                    continue;
                };
                let wanted = if over_riding {
                    "OVER_RIDING_STYLED_ITEM"
                } else {
                    "STYLED_ITEM"
                };
                if name != wanted {
                    continue;
                }
                let Some(item) = record.reference(2) else {
                    continue;
                };
                let colour = record
                    .references(1)
                    .into_iter()
                    .find_map(|style| self.find_colour(style, 0));
                if let Some(colour) = colour {
                    self.colours.insert(item, colour);
                }
            }
        }
    }

    /// The first surface colour reachable from a presentation style.
    fn find_colour(&self, id: u64, depth: usize) -> Option<[u8; 3]> {
        if depth > 12 {
            return None;
        }
        let instance = self.get(id)?;
        for record in &instance.records {
            match record.name.as_str() {
                "COLOUR_RGB" => {
                    let channel = |index: usize| {
                        record
                            .number(index)
                            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
                    };
                    return Some([channel(1)?, channel(2)?, channel(3)?]);
                }
                "DRAUGHTING_PRE_DEFINED_COLOUR" | "PRE_DEFINED_COLOUR" => {
                    return match record.string(0)?.to_ascii_lowercase().as_str() {
                        "red" => Some([255, 0, 0]),
                        "green" => Some([0, 255, 0]),
                        "blue" => Some([0, 0, 255]),
                        "yellow" => Some([255, 255, 0]),
                        "magenta" => Some([255, 0, 255]),
                        "cyan" => Some([0, 255, 255]),
                        "black" => Some([0, 0, 0]),
                        "white" => Some([255, 255, 255]),
                        _ => None,
                    };
                }
                // Edge and annotation colours are not surface colours.
                "CURVE_STYLE" | "POINT_STYLE" | "TEXT_STYLE" | "SYMBOL_STYLE" => return None,
                _ => {}
            }
            for param in &record.params {
                let references = match param {
                    Param::List(_) => param.references(),
                    other => other.reference().into_iter().collect(),
                };
                for reference in references {
                    if let Some(colour) = self.find_colour(reference, depth + 1) {
                        return Some(colour);
                    }
                }
            }
        }
        None
    }

    /// Length and plane-angle factors of a representation context.
    fn context_units(&self, id: u64) -> (Option<f64>, Option<f64>) {
        let Some(record) = self
            .get(id)
            .and_then(|context| context.record("GLOBAL_UNIT_ASSIGNED_CONTEXT"))
        else {
            return (None, None);
        };
        let mut length = None;
        let mut angle = None;
        for unit in record.references(0) {
            let Some(instance) = self.get(unit) else {
                continue;
            };
            if instance.has("LENGTH_UNIT") {
                length = length.or(self.unit_factor(unit, 0));
            } else if instance.has("PLANE_ANGLE_UNIT") {
                angle = angle.or(self.unit_factor(unit, 0));
            }
        }
        (length, angle)
    }

    /// A unit's size in SI units (metres, radians).
    fn unit_factor(&self, id: u64, depth: usize) -> Option<f64> {
        if depth > 4 {
            return None;
        }
        let instance = self.get(id)?;
        if let Some(si) = instance.record("SI_UNIT") {
            return Some(si_prefix(si.param(0).and_then(Param::enumeration)));
        }
        let conversion = instance.record("CONVERSION_BASED_UNIT")?;
        let measure = self.get(conversion.reference(1)?)?;
        let record = measure
            .records
            .iter()
            .find(|record| record.name.ends_with("MEASURE_WITH_UNIT") && record.params.len() >= 2)?;
        let value = record.number(0)?;
        let factor = value * self.unit_factor(record.reference(1)?, depth + 1)?;
        (factor.is_finite() && factor > 0.0).then_some(factor)
    }

    /// Items and context of a representation.
    fn representation(&self, id: u64) -> Option<(Vec<u64>, Option<u64>)> {
        let instance = self.get(id)?;
        let record = if instance.complex {
            instance.record("REPRESENTATION")?
        } else {
            let record = &instance.records[0];
            if !record.name.ends_with("REPRESENTATION")
                || record.name.ends_with("DEFINITION_REPRESENTATION")
            {
                return None;
            }
            record
        };
        if record.params.len() < 3 {
            return None;
        }
        let items = record.param(1)?.list()?;
        Some((
            items.iter().filter_map(Param::reference).collect(),
            record.reference(2),
        ))
    }

    fn is_body(&self, id: u64) -> bool {
        self.simple(id)
            .is_some_and(|(name, _)| BODY_TYPES.contains(&name))
    }

    /// Every body to tessellate, placed through the assembly structure.
    fn placements(&mut self) -> Vec<Placement> {
        // Representations, grouped where plain relationships share a
        // coordinate system.
        let mut representations: Vec<(u64, Vec<u64>, Option<u64>)> = Vec::new();
        let mut index_of: HashMap<u64, usize> = HashMap::new();
        for id in &self.ids {
            if let Some((items, context)) = self.representation(*id) {
                index_of.insert(*id, representations.len());
                representations.push((*id, items, context));
            }
        }
        let mut groups = UnionFind((0..representations.len()).collect());
        // (child representation, parent representation, child frame item,
        // parent frame item, colour item)
        let mut links: Vec<(u64, u64, Option<u64>, Option<u64>, Option<u64>)> = Vec::new();
        for id in &self.ids {
            let Some(instance) = self.get(*id) else {
                continue;
            };
            let Some(relationship) = instance
                .record("REPRESENTATION_RELATIONSHIP")
                .or_else(|| instance.simple("SHAPE_REPRESENTATION_RELATIONSHIP"))
            else {
                continue;
            };
            let (Some(first), Some(second)) =
                (relationship.reference(2), relationship.reference(3))
            else {
                continue;
            };
            let transformation = instance
                .record("REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
                .and_then(|record| record.reference(0));
            match transformation {
                Some(operator) => {
                    let (from, to) = match self.simple(operator) {
                        Some(("ITEM_DEFINED_TRANSFORMATION", record)) => {
                            (record.reference(2), record.reference(3))
                        }
                        // A transformation operator maps the child into the
                        // parent by itself.
                        _ => (None, Some(operator)),
                    };
                    links.push((first, second, from, to, None));
                }
                None => {
                    if let (Some(a), Some(b)) = (index_of.get(&first), index_of.get(&second)) {
                        groups.union(*a, *b);
                    }
                }
            }
        }
        let mut mapped = Vec::new();
        for (parent, items, _) in &representations {
            for item in items {
                if let Some(("MAPPED_ITEM", record)) = self.simple(*item)
                    && let Some(("REPRESENTATION_MAP", map)) =
                        record.reference(1).and_then(|map| self.simple(map))
                    && let Some(child) = map.reference(1)
                {
                    mapped.push((child, *parent, map.reference(0), record.reference(2), Some(*item)));
                }
            }
        }
        links.extend(mapped);

        let group_count = representations.len();
        let mut group_items: Vec<Vec<u64>> = vec![Vec::new(); group_count];
        let mut group_unit: Vec<Option<f64>> = vec![None; group_count];
        for (index, (_, items, context)) in representations.iter().enumerate() {
            let group = groups.find(index);
            group_items[group].extend(items.iter().copied());
            if let Some(context) = context {
                let (length, angle) = self.context_units(*context);
                if group_unit[group].is_none() {
                    group_unit[group] = length;
                }
                if self.angle_unit.is_none() {
                    self.angle_unit = angle;
                }
            }
        }
        let default_unit = group_unit.iter().flatten().next().copied().unwrap_or(1e-3);
        let unit_of = |group: usize| group_unit[group].unwrap_or(default_unit);

        // Edges parent -> (child, child-to-parent transform in metres,
        // colour).
        let mut children: BTreeMap<usize, Vec<(usize, DAffine3, Option<[u8; 3]>)>> =
            BTreeMap::new();
        let mut has_parent = vec![false; group_count];
        for (child, parent, from, to, colour_item) in links {
            let (Some(&child_index), Some(&parent_index)) =
                (index_of.get(&child), index_of.get(&parent))
            else {
                continue;
            };
            let (child_group, parent_group) =
                (groups.find(child_index), groups.find(parent_index));
            let (mut from, mut to) = (from, to);
            // Writers disagree on which item belongs to which side; the
            // items say.
            let holds = |group: usize, item: Option<u64>| {
                item.is_some_and(|item| group_items[group].contains(&item))
            };
            if holds(parent_group, from)
                && holds(child_group, to)
                && !(holds(child_group, from) && holds(parent_group, to))
            {
                std::mem::swap(&mut from, &mut to);
            }
            if child_group == parent_group {
                continue;
            }
            let metric = |frame: Option<Frame>, unit: f64| {
                let mut frame = frame.unwrap_or(Frame::IDENTITY);
                frame.origin *= unit;
                frame.affine()
            };
            let child_frame = metric(from.and_then(|id| self.frame(id)), unit_of(child_group));
            let parent_frame = metric(to.and_then(|id| self.frame(id)), unit_of(parent_group));
            let transform = parent_frame * child_frame.inverse();
            if !transform.is_finite() {
                continue;
            }
            let colour = colour_item.and_then(|item| self.colours.get(&item).copied());
            children
                .entry(parent_group)
                .or_default()
                .push((child_group, transform, colour));
            has_parent[child_group] = true;
        }

        let tessellated_only = !group_items.iter().flatten().any(|item| {
            self.simple(*item)
                .is_some_and(|(name, _)| BODY_TYPES.contains(&name) && !TESSELLATED_TYPES.contains(&name))
        });
        let mut placements = Vec::new();
        let roots: BTreeSet<usize> = (0..group_count)
            .filter(|group| groups.find(*group) == *group && !has_parent[*group])
            .collect();
        for root in roots {
            let mut stack = vec![root];
            self.place(
                root,
                DAffine3::IDENTITY,
                None,
                &mut stack,
                &group_items,
                &unit_of,
                &children,
                tessellated_only,
                &mut placements,
            );
        }
        if placements.is_empty() {
            // Bodies outside any representation.
            for id in self.ids.clone() {
                if self.is_body(id)
                    && matches!(
                        self.simple(id).map(|(name, _)| name),
                        Some(
                            "MANIFOLD_SOLID_BREP"
                                | "BREP_WITH_VOIDS"
                                | "FACETED_BREP"
                                | "SHELL_BASED_SURFACE_MODEL"
                                | "FACE_BASED_SURFACE_MODEL"
                                | "TESSELLATED_SOLID"
                                | "TESSELLATED_SHELL"
                        )
                    )
                {
                    placements.push(Placement {
                        item: id,
                        transform: DAffine3::from_scale(DVec3::splat(default_unit)),
                        unit: default_unit,
                        colour: None,
                    });
                }
            }
        }
        placements
    }

    #[allow(clippy::too_many_arguments)]
    fn place(
        &self,
        group: usize,
        world: DAffine3,
        colour: Option<[u8; 3]>,
        stack: &mut Vec<usize>,
        group_items: &[Vec<u64>],
        unit_of: &dyn Fn(usize) -> f64,
        children: &BTreeMap<usize, Vec<(usize, DAffine3, Option<[u8; 3]>)>>,
        tessellated_only: bool,
        placements: &mut Vec<Placement>,
    ) {
        let unit = unit_of(group);
        let mut seen = BTreeSet::new();
        for item in &group_items[group] {
            if placements.len() >= MAX_PLACED_ITEMS {
                return;
            }
            let Some((name, _)) = self.simple(*item) else {
                continue;
            };
            // Exact and tessellated forms of one part may both be present;
            // prefer the exact geometry.
            if !BODY_TYPES.contains(&name)
                || (!tessellated_only && TESSELLATED_TYPES.contains(&name))
                || !seen.insert(*item)
            {
                continue;
            }
            placements.push(Placement {
                item: *item,
                transform: world * DAffine3::from_scale(DVec3::splat(unit)),
                unit,
                colour: self.colours.get(item).copied().or(colour),
            });
        }
        if stack.len() > MAX_ASSEMBLY_DEPTH {
            return;
        }
        for (child, transform, child_colour) in children.get(&group).into_iter().flatten() {
            if stack.contains(child) {
                continue;
            }
            stack.push(*child);
            self.place(
                *child,
                world * *transform,
                child_colour.or(colour),
                stack,
                group_items,
                unit_of,
                children,
                tessellated_only,
                placements,
            );
            stack.pop();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Writes STEP entities with sequential ids.
    struct Writer {
        lines: Vec<String>,
        next: u64,
    }

    fn real(value: f64) -> String {
        format!("{value:?}")
    }

    fn triple(values: [f64; 3]) -> String {
        format!("({},{},{})", real(values[0]), real(values[1]), real(values[2]))
    }

    impl Writer {
        fn new() -> Self {
            Self {
                lines: Vec::new(),
                next: 1,
            }
        }

        fn add(&mut self, entity: impl AsRef<str>) -> u64 {
            let id = self.next;
            self.next += 1;
            self.lines.push(format!("#{id}={};", entity.as_ref()));
            id
        }

        fn point(&mut self, p: [f64; 3]) -> u64 {
            self.add(format!("CARTESIAN_POINT('',{})", triple(p)))
        }

        fn direction(&mut self, d: [f64; 3]) -> u64 {
            self.add(format!("DIRECTION('',{})", triple(d)))
        }

        fn axis(&mut self, origin: [f64; 3], z: [f64; 3], x: [f64; 3]) -> u64 {
            let (o, z, x) = (self.point(origin), self.direction(z), self.direction(x));
            self.add(format!("AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})"))
        }

        fn vertex(&mut self, p: [f64; 3]) -> u64 {
            let point = self.point(p);
            self.add(format!("VERTEX_POINT('',#{point})"))
        }

        fn line_edge(&mut self, start: (u64, [f64; 3]), end: (u64, [f64; 3])) -> u64 {
            let origin = self.point(start.1);
            let delta = DVec3::from(end.1) - DVec3::from(start.1);
            let direction = self.direction(delta.normalize().to_array());
            let vector = self.add(format!("VECTOR('',#{direction},{})", real(delta.length())));
            let line = self.add(format!("LINE('',#{origin},#{vector})"));
            self.add(format!("EDGE_CURVE('',#{},#{},#{line},.T.)", start.0, end.0))
        }

        /// A face from bounds of (edge, forward) lists.
        fn face(&mut self, bounds: &[Vec<(u64, bool)>], surface: u64, sense: bool) -> u64 {
            let mut bound_ids = Vec::new();
            for (index, edges) in bounds.iter().enumerate() {
                let oriented: Vec<String> = edges
                    .iter()
                    .map(|(edge, forward)| {
                        let flag = if *forward { "T" } else { "F" };
                        let id = self.add(format!("ORIENTED_EDGE('',*,*,#{edge},.{flag}.)"));
                        format!("#{id}")
                    })
                    .collect();
                let edge_loop = self.add(format!("EDGE_LOOP('',({}))", oriented.join(",")));
                let kind = if index == 0 {
                    "FACE_OUTER_BOUND"
                } else {
                    "FACE_BOUND"
                };
                let bound = self.add(format!("{kind}('',#{edge_loop},.T.)"));
                bound_ids.push(format!("#{bound}"));
            }
            let flag = if sense { "T" } else { "F" };
            self.add(format!(
                "ADVANCED_FACE('',({}),#{surface},.{flag}.)",
                bound_ids.join(",")
            ))
        }

        /// A millimetre context; returns (context, millimetre unit).
        fn millimetre_context(&mut self) -> (u64, u64) {
            let mm = self.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )");
            let rad = self.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )");
            let context = self.add(format!(
                "( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#{mm},#{rad})) REPRESENTATION_CONTEXT('','') )"
            ));
            (context, mm)
        }

        /// A body in its own representation; returns (representation,
        /// origin axis).
        fn representation(&mut self, body: u64, context: u64) -> (u64, u64) {
            let axis = self.axis([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
            let rep = self.add(format!(
                "ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{axis},#{body}),#{context})"
            ));
            (rep, axis)
        }

        fn finish(&self) -> String {
            format!(
                "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\nFILE_NAME('t.stp','2026-10-13',(''),(''),'','','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{}\nENDSEC;\nEND-ISO-10303-21;\n",
                self.lines.join("\n")
            )
        }
    }

    /// A cube of side `size` at the origin as a manifold solid. Every other
    /// face uses a reversed plane with `same_sense` false.
    fn cube(writer: &mut Writer, size: f64) -> u64 {
        let corners: Vec<[f64; 3]> = (0..8)
            .map(|i| {
                [
                    (i & 1) as f64 * size,
                    ((i >> 1) & 1) as f64 * size,
                    ((i >> 2) & 1) as f64 * size,
                ]
            })
            .collect();
        let vertices: Vec<u64> = corners.iter().map(|c| writer.vertex(*c)).collect();
        let faces: [([usize; 4], [f64; 3]); 6] = [
            ([0, 1, 3, 2], [0.0, 0.0, -1.0]),
            ([4, 5, 7, 6], [0.0, 0.0, 1.0]),
            ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
            ([2, 3, 7, 6], [0.0, 1.0, 0.0]),
            ([0, 2, 6, 4], [-1.0, 0.0, 0.0]),
            ([1, 3, 7, 5], [1.0, 0.0, 0.0]),
        ];
        let mut edges: HashMap<(usize, usize), u64> = HashMap::new();
        let mut face_ids = Vec::new();
        for (index, (mut quad, normal)) in faces.into_iter().enumerate() {
            let points: Vec<DVec3> = quad.iter().map(|i| DVec3::from(corners[*i])).collect();
            if newell_normal(&points).dot(DVec3::from(normal)) < 0.0 {
                quad.reverse();
            }
            let mut loop_edges = Vec::new();
            for k in 0..4 {
                let (a, b) = (quad[k], quad[(k + 1) % 4]);
                let key = (a.min(b), a.max(b));
                let edge = *edges.entry(key).or_insert_with(|| {
                    writer.line_edge(
                        (vertices[key.0], corners[key.0]),
                        (vertices[key.1], corners[key.1]),
                    )
                });
                loop_edges.push((edge, a < b));
            }
            let flipped = index % 2 == 1;
            let axis_normal = if flipped { normal.map(|v| -v) } else { normal };
            let reference = if normal[0] != 0.0 {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            let axis = writer.axis(corners[quad[0]], axis_normal, reference);
            let plane = writer.add(format!("PLANE('',#{axis})"));
            face_ids.push(writer.face(&[loop_edges], plane, !flipped));
        }
        let list: Vec<String> = face_ids.iter().map(|id| format!("#{id}")).collect();
        let shell = writer.add(format!("CLOSED_SHELL('',({}))", list.join(",")));
        writer.add(format!("MANIFOLD_SOLID_BREP('cube',#{shell})"))
    }

    /// A cylinder of radius `r` and height `h` on the z axis; with `seam`,
    /// the side face is bounded by one loop through a seam edge, otherwise
    /// by its two circles.
    fn cylinder(writer: &mut Writer, r: f64, h: f64, seam: bool) -> u64 {
        let (p1, p2) = ([r, 0.0, 0.0], [r, 0.0, h]);
        let (v1, v2) = (writer.vertex(p1), writer.vertex(p2));
        let circle = |writer: &mut Writer, z: f64, vertex: u64| {
            let axis = writer.axis([0.0, 0.0, z], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
            let curve = writer.add(format!("CIRCLE('',#{axis},{})", real(r)));
            writer.add(format!("EDGE_CURVE('',#{vertex},#{vertex},#{curve},.T.)"))
        };
        let bottom_edge = circle(writer, 0.0, v1);
        let top_edge = circle(writer, h, v2);
        let bottom_axis = writer.axis([0.0; 3], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]);
        let bottom_plane = writer.add(format!("PLANE('',#{bottom_axis})"));
        let bottom = writer.face(&[vec![(bottom_edge, false)]], bottom_plane, true);
        let top_axis = writer.axis([0.0, 0.0, h], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let top_plane = writer.add(format!("PLANE('',#{top_axis})"));
        let top = writer.face(&[vec![(top_edge, true)]], top_plane, true);
        let side_axis = writer.axis([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let side_surface =
            writer.add(format!("CYLINDRICAL_SURFACE('',#{side_axis},{})", real(r)));
        let side = if seam {
            let seam_edge = writer.line_edge((v1, p1), (v2, p2));
            let bound = vec![
                (bottom_edge, true),
                (seam_edge, true),
                (top_edge, false),
                (seam_edge, false),
            ];
            writer.face(&[bound], side_surface, true)
        } else {
            let bounds = [vec![(bottom_edge, true)], vec![(top_edge, false)]];
            writer.face(&bounds, side_surface, true)
        };
        let shell = writer.add(format!("CLOSED_SHELL('',(#{bottom},#{top},#{side}))"));
        writer.add(format!("MANIFOLD_SOLID_BREP('cylinder',#{shell})"))
    }

    fn volume(geometry: &MeshGeometry) -> f64 {
        geometry
            .triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| geometry.vertices[i as usize].position.as_dvec3());
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    fn bounds(geometry: &MeshGeometry) -> (Vec3, Vec3) {
        geometry.vertices.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(min, max), vertex| (min.min(vertex.position), max.max(vertex.position)),
        )
    }

    /// Every edge, welded by position, is used once in each direction.
    fn assert_watertight(geometry: &MeshGeometry) {
        let key = |i: u32| {
            let p = geometry.vertices[i as usize].position.as_dvec3() * 1e7;
            (p.x.round() as i64, p.y.round() as i64, p.z.round() as i64)
        };
        let mut edges: HashMap<_, i32> = HashMap::new();
        for t in &geometry.triangles {
            for k in 0..3 {
                let (a, b) = (key(t[k]), key(t[(k + 1) % 3]));
                if a == b {
                    continue;
                }
                *edges.entry((a, b)).or_default() += 1;
                *edges.entry((b, a)).or_default() -= 1;
            }
        }
        let open = edges.values().filter(|count| **count != 0).count();
        assert_eq!(open, 0, "{open} unmatched half-edges");
    }

    fn uvs_in_unit_square(geometry: &MeshGeometry) -> bool {
        geometry
            .vertices
            .iter()
            .all(|v| v.uv.min_element() >= 0.0 && v.uv.max_element() <= 1.0)
    }

    fn import(text: &str) -> (ImportedMesh, StepImportReport) {
        import_step_with_report(
            text.as_bytes(),
            "part".to_owned(),
            &StepTessellation::default(),
        )
        .unwrap()
    }

    #[test]
    fn tc_0331_cube_imports_closed_outward_and_in_metres() {
        let mut writer = Writer::new();
        let (context, _) = writer.millimetre_context();
        let body = cube(&mut writer, 10.0);
        writer.representation(body, context);
        let (mesh, report) = import(&writer.finish());
        let geometry = &mesh.geometry;
        assert_eq!(geometry.triangles.len(), 12);
        assert_eq!(geometry.vertices.len(), 24);
        assert!(geometry.material_slots.is_empty() && mesh.generated_uvs);
        assert!((volume(geometry) - 1e-6).abs() < 1e-12, "{}", volume(geometry));
        assert_watertight(geometry);
        let (min, max) = bounds(geometry);
        assert!(min.abs().max_element() < 1e-7);
        assert!((max - Vec3::splat(0.01)).abs().max_element() < 1e-7);
        assert!(uvs_in_unit_square(geometry));
        assert_eq!((report.bodies, report.faces, report.skipped_faces), (1, 6, 0));
        assert!((report.length_unit - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn tc_0332_cylinders_with_and_without_seams_are_round_and_watertight() {
        for seam in [true, false] {
            let mut writer = Writer::new();
            let (context, _) = writer.millimetre_context();
            let body = cylinder(&mut writer, 5.0, 20.0, seam);
            writer.representation(body, context);
            let (mesh, report) = import(&writer.finish());
            let geometry = &mesh.geometry;
            assert_eq!(report.skipped_faces, 0);
            let expected = PI * 0.005 * 0.005 * 0.02;
            assert!(
                (volume(geometry) - expected).abs() < expected * 0.01,
                "seam {seam}: {} vs {expected}",
                volume(geometry)
            );
            assert_watertight(geometry);
            for vertex in &geometry.vertices {
                let p = vertex.position.as_dvec3();
                assert!(p.x.hypot(p.y) <= 0.005 + 1e-7);
            }
        }
    }

    #[test]
    fn tc_0333_sphere_and_nurbs_faces_follow_their_surfaces() {
        let mut writer = Writer::new();
        let (context, _) = writer.millimetre_context();
        let pole = writer.vertex([0.0, 0.0, 30.0]);
        let vertex_loop = writer.add(format!("VERTEX_LOOP('',#{pole})"));
        let bound = writer.add(format!("FACE_BOUND('',#{vertex_loop},.T.)"));
        let axis = writer.axis([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let sphere = writer.add(format!("SPHERICAL_SURFACE('',#{axis},30.)"));
        let face = writer.add(format!("ADVANCED_FACE('',(#{bound}),#{sphere},.T.)"));
        let shell = writer.add(format!("CLOSED_SHELL('',(#{face}))"));
        let body = writer.add(format!("MANIFOLD_SOLID_BREP('ball',#{shell})"));
        writer.representation(body, context);
        let (mesh, _) = import(&writer.finish());
        let expected = 4.0 / 3.0 * PI * 0.03_f64.powi(3);
        assert!((volume(&mesh.geometry) - expected).abs() < expected * 0.01);
        for vertex in &mesh.geometry.vertices {
            assert!((vertex.position.as_dvec3().length() - 0.03).abs() < 1e-6);
        }

        // A biquadratic dome patch bounded by its four boundary curves, one
        // written as a complex rational B-spline.
        let mut writer = Writer::new();
        let (context, _) = writer.millimetre_context();
        let grid: Vec<Vec<[f64; 3]>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        let height = if i == 1 && j == 1 { 20.0 } else { 0.0 };
                        [i as f64 * 50.0, j as f64 * 50.0, height]
                    })
                    .collect()
            })
            .collect();
        let ids: Vec<Vec<u64>> = grid
            .iter()
            .map(|row| row.iter().map(|p| writer.point(*p)).collect())
            .collect();
        let rows: Vec<String> = ids
            .iter()
            .map(|row| {
                let names: Vec<String> = row.iter().map(|id| format!("#{id}")).collect();
                format!("({})", names.join(","))
            })
            .collect();
        let surface = writer.add(format!(
            "B_SPLINE_SURFACE_WITH_KNOTS('',2,2,({}),.UNSPECIFIED.,.F.,.F.,.F.,(3,3),(3,3),(0.,1.),(0.,1.),.UNSPECIFIED.)",
            rows.join(",")
        ));
        let corners = [(0, 0), (2, 0), (2, 2), (0, 2)];
        let corner_vertices: Vec<u64> = corners
            .iter()
            .map(|(i, j)| writer.vertex(grid[*i][*j]))
            .collect();
        let mut loop_edges = Vec::new();
        for k in 0..4 {
            let ((i0, j0), (i1, j1)) = (corners[k], corners[(k + 1) % 4]);
            let path: Vec<u64> = (0..3)
                .map(|step| ids[(i0 * (2 - step) + i1 * step) / 2][(j0 * (2 - step) + j1 * step) / 2])
                .collect();
            let points = format!("(#{},#{},#{})", path[0], path[1], path[2]);
            let curve = if k == 0 {
                writer.add(format!(
                    "( BOUNDED_CURVE() B_SPLINE_CURVE(2,{points},.UNSPECIFIED.,.F.,.F.) B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.),.UNSPECIFIED.) CURVE() GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE((1.,1.,1.)) REPRESENTATION_ITEM('') )"
                ))
            } else {
                writer.add(format!(
                    "B_SPLINE_CURVE_WITH_KNOTS('',2,{points},.UNSPECIFIED.,.F.,.F.,(3,3),(0.,1.),.UNSPECIFIED.)"
                ))
            };
            let edge = writer.add(format!(
                "EDGE_CURVE('',#{},#{},#{curve},.T.)",
                corner_vertices[k],
                corner_vertices[(k + 1) % 4]
            ));
            loop_edges.push((edge, true));
        }
        let face = writer.face(&[loop_edges], surface, true);
        let shell = writer.add(format!("OPEN_SHELL('',(#{face}))"));
        let model = writer.add(format!("SHELL_BASED_SURFACE_MODEL('',(#{shell}))"));
        writer.representation(model, context);
        let (mesh, report) = import(&writer.finish());
        assert_eq!(report.skipped_faces, 0);
        let (min, max) = bounds(&mesh.geometry);
        // The patch peaks at a quarter of the middle control point's height.
        assert!(mesh.geometry.triangles.len() > 20);
        assert!((max.z - 0.005).abs() < 2e-4, "{max}");
        assert!(min.z >= -1e-7);
    }

    #[test]
    fn tc_0334_assemblies_place_instances_with_units_and_colours() {
        let mut writer = Writer::new();
        let (part_context, mm) = writer.millimetre_context();
        let body = cube(&mut writer, 10.0);
        let (part, part_axis) = writer.representation(body, part_context);
        // Red faces through a styled item on the solid.
        let colour = writer.add("COLOUR_RGB('',1.,0.,0.)");
        let fill_colour = writer.add(format!("FILL_AREA_STYLE_COLOUR('',#{colour})"));
        let fill = writer.add(format!("FILL_AREA_STYLE('',(#{fill_colour}))"));
        let area = writer.add(format!("SURFACE_STYLE_FILL_AREA(#{fill})"));
        let side = writer.add(format!("SURFACE_SIDE_STYLE('',(#{area}))"));
        let usage = writer.add(format!("SURFACE_STYLE_USAGE(.BOTH.,#{side})"));
        let assignment = writer.add(format!("PRESENTATION_STYLE_ASSIGNMENT((#{usage}))"));
        writer.add(format!("STYLED_ITEM('',(#{assignment}),#{body})"));
        // A second, uncoloured part placed through a mapped item.
        let body_b = cube(&mut writer, 10.0);
        let (part_b, part_b_axis) = writer.representation(body_b, part_context);

        // The assembly is in inches.
        let measure = writer.add(format!(
            "LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#{mm})"
        ));
        let dimensions = writer.add("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)");
        let inch = writer.add(format!(
            "( CONVERSION_BASED_UNIT('INCH',#{measure}) LENGTH_UNIT() NAMED_UNIT(#{dimensions}) )"
        ));
        let assembly_context = writer.add(format!(
            "( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#{inch})) REPRESENTATION_CONTEXT('','') )"
        ));
        let first = writer.axis([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        // A quarter turn about z, two inches along x.
        let second = writer.axis([2.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let third = writer.axis([0.0, 0.0, 3.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let map = writer.add(format!("REPRESENTATION_MAP(#{part_b_axis},#{part_b})"));
        let mapped = writer.add(format!("MAPPED_ITEM('',#{map},#{third})"));
        let assembly = writer.add(format!(
            "SHAPE_REPRESENTATION('',(#{first},#{second},#{mapped}),#{assembly_context})"
        ));
        for placement in [first, second] {
            let transform = writer.add(format!(
                "ITEM_DEFINED_TRANSFORMATION('','',#{part_axis},#{placement})"
            ));
            writer.add(format!(
                "( REPRESENTATION_RELATIONSHIP('','',#{part},#{assembly}) REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{transform}) SHAPE_REPRESENTATION_RELATIONSHIP() )"
            ));
        }
        let (mesh, report) = import(&writer.finish());
        let geometry = &mesh.geometry;
        assert_eq!(report.bodies, 3);
        assert_eq!(geometry.triangles.len(), 36);
        assert_eq!(geometry.material_slots, vec!["Colour FF0000", "Default"]);
        let red = geometry.triangle_materials.iter().filter(|s| **s == 0).count();
        assert_eq!(red, 24);
        assert!((volume(geometry) - 3e-6).abs() < 1e-11, "{}", volume(geometry));
        // The turned instance spans x from 2 in - 10 mm to 2 in; the mapped
        // one sits 3 in up.
        let (min, max) = bounds(geometry);
        assert!(min.abs().max_element() < 1e-6, "{min}");
        assert!(
            (max - Vec3::new(0.0508, 0.01, 0.0762 + 0.01)).abs().max_element() < 1e-6,
            "{max}"
        );
        assert!(uvs_in_unit_square(geometry));
    }

    #[test]
    fn tc_0335_faceted_and_tessellated_bodies_import() {
        // A faceted tetrahedron with polygon loops and no surfaces.
        let mut writer = Writer::new();
        let (context, _) = writer.millimetre_context();
        let corners = [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ];
        let points: Vec<u64> = corners.iter().map(|c| writer.point(*c)).collect();
        let mut faces = Vec::new();
        for [a, b, c] in [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] {
            let polygon = writer.add(format!(
                "POLY_LOOP('',(#{},#{},#{}))",
                points[a], points[b], points[c]
            ));
            let bound = writer.add(format!("FACE_OUTER_BOUND('',#{polygon},.T.)"));
            let face = writer.add(format!("FACE('',(#{bound}))"));
            faces.push(format!("#{face}"));
        }
        let shell = writer.add(format!("CLOSED_SHELL('',({}))", faces.join(",")));
        let body = writer.add(format!("FACETED_BREP('',#{shell})"));
        writer.representation(body, context);
        let (mesh, _) = import(&writer.finish());
        assert_eq!(mesh.geometry.triangles.len(), 4);
        assert!((volume(&mesh.geometry) - 1e-6 / 6.0).abs() < 1e-12);

        // The same tetrahedron as an AP242 tessellated shell.
        let mut writer = Writer::new();
        let (context, _) = writer.millimetre_context();
        let coordinates = writer.add(
            "COORDINATES_LIST('',4,((0.,0.,0.),(10.,0.,0.),(0.,10.,0.),(0.,0.,10.)))",
        );
        let face = writer.add(format!(
            "TRIANGULATED_FACE('',#{coordinates},0,(),$,(),((1,3,2),(1,2,4),(1,4,3),(2,3,4)))"
        ));
        let shell = writer.add(format!("TESSELLATED_SHELL('',(#{face}),$)"));
        writer.add(format!(
            "TESSELLATED_SHAPE_REPRESENTATION('',(#{shell}),#{context})"
        ));
        let (mesh, _) = import(&writer.finish());
        assert_eq!(mesh.geometry.triangles.len(), 4);
        assert!((volume(&mesh.geometry) - 1e-6 / 6.0).abs() < 1e-12);
        assert!(uvs_in_unit_square(&mesh.geometry));
    }

    #[test]
    fn tc_0336_rejects_files_without_geometry_and_broken_syntax() {
        let mut writer = Writer::new();
        writer.point([0.0; 3]);
        assert_eq!(
            import_step(writer.finish().as_bytes(), "empty".to_owned()),
            Err(StepImportError::NoGeometry)
        );
        assert!(matches!(
            import_step(b"ISO-10303-21;\nDATA;\n#1=PLANE('',;\n", "x".to_owned()),
            Err(StepImportError::Parse(_))
        ));
        assert!(matches!(
            import_step(b"solid ascii stl", "x".to_owned()),
            Err(StepImportError::Parse(_))
        ));
    }
}
