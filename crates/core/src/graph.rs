//! Procedural material graphs: small node graphs evaluated per texel that
//! give a fill layer (or a material resource) its content.
//!
//! Nodes are kept in evaluation order and an input may only read an earlier
//! node, so a graph is acyclic by construction. Every node yields an RGB
//! value; grey values hold the same number in all three. Nodes read only
//! their own texel (no neighbours), so region and tiled bakes of a graph
//! are exact without any halo.

use std::collections::BTreeMap;

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::compositing::{BakeError, BlendMode};
use crate::export::SourceComponent;
use crate::filters::LayerFilter;
use crate::generators::MaskGenerator;
use crate::mesh_maps::{MeshMapKind, MeshMapSample};
use crate::resources::{ResourceImages, TextureImage, TextureProjection};
use crate::{MaterialChannel, ResourceId};

/// Most nodes a graph may hold.
pub const MAX_GRAPH_NODES: usize = 64;
/// Most stops a colour ramp may hold.
pub const MAX_RAMP_STOPS: usize = 8;
/// Largest magnitude a constant input may have.
const MAX_CONSTANT: f32 = 1_000.0;

/// What a node input reads.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphInput {
    /// The output of an earlier node, by index.
    Node(u16),
    /// A constant RGB value (grey when all three match).
    Value([f32; 3]),
}

impl GraphInput {
    /// A constant grey.
    pub const fn gray(value: f32) -> Self {
        Self::Value([value, value, value])
    }

    /// True when the input reads a node before `node` (or a finite
    /// constant).
    fn is_valid(self, node: usize) -> bool {
        match self {
            Self::Node(index) => usize::from(index) < node,
            Self::Value(value) => value
                .iter()
                .all(|component| component.is_finite() && component.abs() <= MAX_CONSTANT),
        }
    }

    fn read(self, values: &[Vec3]) -> Vec3 {
        match self {
            Self::Node(index) => values
                .get(usize::from(index))
                .copied()
                .unwrap_or(Vec3::ZERO),
            Self::Value(value) => Vec3::from(value),
        }
    }

    /// The highest node index this input reads.
    fn node(self) -> Option<usize> {
        match self {
            Self::Node(index) => Some(usize::from(index)),
            Self::Value(_) => None,
        }
    }
}

/// Arithmetic of a [`GraphNode::Math`] node, per component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathOp {
    Add,
    Subtract,
    Multiply,
    /// `a / b`, or 0 where `b` is 0.
    Divide,
    Minimum,
    Maximum,
    /// `a` (clamped at 0) to the power `b`.
    Power,
}

impl MathOp {
    pub const ALL: [Self; 7] = [
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Divide,
        Self::Minimum,
        Self::Maximum,
        Self::Power,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::Multiply => "Multiply",
            Self::Divide => "Divide",
            Self::Minimum => "Minimum",
            Self::Maximum => "Maximum",
            Self::Power => "Power",
        }
    }

    fn apply(self, a: f32, b: f32) -> f32 {
        let value = match self {
            Self::Add => a + b,
            Self::Subtract => a - b,
            Self::Multiply => a * b,
            Self::Divide if b == 0.0 => 0.0,
            Self::Divide => a / b,
            Self::Minimum => a.min(b),
            Self::Maximum => a.max(b),
            Self::Power => a.max(0.0).powf(b),
        };
        if value.is_finite() {
            value.clamp(-MAX_CONSTANT, MAX_CONSTANT)
        } else {
            0.0
        }
    }
}

/// A colour at a position along a [`GraphNode::Ramp`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RampStop {
    /// `0..=1`; stops are kept in increasing order.
    pub position: f32,
    /// sRGB.
    pub color: [u8; 3],
}

/// One step of a [`MaterialGraph`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphNode {
    /// A mask generator's value, as grey. ID-colour selection is not
    /// available here (graphs do not read the ID map).
    Generator(MaskGenerator),
    /// A baked mesh map at the texel, as exports write it: normals and
    /// positions remapped to `0..1`, black outside UV islands.
    MeshMap(MeshMapKind),
    /// The texel's UV as `(u, v, 0)`.
    Uv,
    /// A project bitmap through a projection, as sRGB colour.
    Bitmap {
        resource: ResourceId,
        projection: TextureProjection,
    },
    /// `b` blended over `a` with a layer blend mode, faded in by `amount`
    /// (read as luminance).
    Blend {
        mode: BlendMode,
        a: GraphInput,
        b: GraphInput,
        amount: GraphInput,
    },
    Math {
        op: MathOp,
        a: GraphInput,
        b: GraphInput,
    },
    /// A per-texel layer filter (Levels, Invert, Hue / Saturation,
    /// Brightness / Contrast, Posterize, Threshold, Gradient map), with the
    /// same maths as in a filter stack.
    Adjust {
        input: GraphInput,
        filter: LayerFilter,
    },
    /// Colours mapped from the input's luminance through 2 to 8 stops.
    Ramp {
        input: GraphInput,
        stops: Vec<RampStop>,
    },
    /// One component of the input, as grey.
    Split {
        input: GraphInput,
        component: SourceComponent,
    },
}

impl GraphNode {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Generator(_) => "Generator",
            Self::MeshMap(_) => "Mesh map",
            Self::Uv => "UV",
            Self::Bitmap { .. } => "Bitmap",
            Self::Blend { .. } => "Blend",
            Self::Math { .. } => "Math",
            Self::Adjust { .. } => "Adjust",
            Self::Ramp { .. } => "Ramp",
            Self::Split { .. } => "Split",
        }
    }

    /// The node's inputs, in order.
    pub fn inputs(&self) -> Vec<GraphInput> {
        match self {
            Self::Generator(_) | Self::MeshMap(_) | Self::Uv | Self::Bitmap { .. } => Vec::new(),
            Self::Blend { a, b, amount, .. } => vec![*a, *b, *amount],
            Self::Math { a, b, .. } => vec![*a, *b],
            Self::Adjust { input, .. } | Self::Ramp { input, .. } | Self::Split { input, .. } => {
                vec![*input]
            }
        }
    }

    /// Mutable access to the node's inputs, in the order of
    /// [`Self::inputs`].
    pub fn inputs_mut(&mut self) -> Vec<&mut GraphInput> {
        match self {
            Self::Generator(_) | Self::MeshMap(_) | Self::Uv | Self::Bitmap { .. } => Vec::new(),
            Self::Blend { a, b, amount, .. } => vec![a, b, amount],
            Self::Math { a, b, .. } => vec![a, b],
            Self::Adjust { input, .. } | Self::Ramp { input, .. } | Self::Split { input, .. } => {
                vec![input]
            }
        }
    }

    fn is_valid(&self, index: usize) -> bool {
        let own = match self {
            Self::Generator(generator) => {
                generator.is_valid() && !matches!(generator, MaskGenerator::IdColor(_))
            }
            Self::MeshMap(_) | Self::Uv => true,
            Self::Bitmap { projection, .. } => projection.is_valid(),
            Self::Blend { mode, .. } => *mode != BlendMode::PassThrough,
            Self::Math { .. } | Self::Split { .. } => true,
            Self::Adjust { filter, .. } => filter.is_valid() && filter.is_per_texel(),
            Self::Ramp { stops, .. } => {
                (2..=MAX_RAMP_STOPS).contains(&stops.len())
                    && stops.iter().all(|stop| {
                        stop.position.is_finite() && (0.0..=1.0).contains(&stop.position)
                    })
                    && stops
                        .windows(2)
                        .all(|pair| pair[0].position <= pair[1].position)
            }
        };
        own && self.inputs().iter().all(|input| input.is_valid(index))
    }

    fn needs_mesh_maps(&self) -> bool {
        match self {
            Self::Generator(generator) => generator.needs_mesh_maps(),
            Self::MeshMap(_) => true,
            Self::Bitmap { projection, .. } => projection.needs_mesh_maps(),
            _ => false,
        }
    }
}

/// A procedural material: nodes evaluated per texel, and which node (or
/// constant) feeds each channel. Channels without an output keep the
/// fill's own value or bitmap.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MaterialGraph {
    pub nodes: Vec<GraphNode>,
    /// Colour channels take the value as sRGB, scalar channels its
    /// luminance, and Normal an encoded tangent-space normal.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub outputs: BTreeMap<MaterialChannel, GraphInput>,
}

impl MaterialGraph {
    pub fn is_valid(&self) -> bool {
        self.nodes.len() <= MAX_GRAPH_NODES
            && self
                .nodes
                .iter()
                .enumerate()
                .all(|(index, node)| node.is_valid(index))
            && self.outputs.iter().all(|(channel, input)| {
                !matches!(
                    channel,
                    MaterialChannel::Id | MaterialChannel::AmbientOcclusion
                ) && input.is_valid(self.nodes.len())
            })
    }

    /// True when a node reads mesh maps.
    pub fn needs_mesh_maps(&self) -> bool {
        self.nodes.iter().any(GraphNode::needs_mesh_maps)
    }

    /// Bitmaps the graph reads.
    pub fn resources(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.nodes.iter().filter_map(|node| match node {
            GraphNode::Bitmap { resource, .. } => Some(*resource),
            _ => None,
        })
    }

    /// Removes node `index`: inputs that read it become a constant mid
    /// grey, and later references shift down.
    pub fn remove_node(&mut self, index: usize) {
        if index >= self.nodes.len() {
            return;
        }
        self.nodes.remove(index);
        let rewire = |input: &mut GraphInput| {
            if let GraphInput::Node(node) = input {
                match usize::from(*node).cmp(&index) {
                    std::cmp::Ordering::Equal => *input = GraphInput::gray(0.5),
                    std::cmp::Ordering::Greater => *node -= 1,
                    std::cmp::Ordering::Less => {}
                }
            }
        };
        for node in &mut self.nodes {
            for input in node.inputs_mut() {
                rewire(input);
            }
        }
        for input in self.outputs.values_mut() {
            rewire(input);
        }
    }

    /// The graph's values at one texel, ready to evaluate outputs: bitmaps
    /// are looked up once, up front.
    pub(crate) fn evaluator<'a>(
        &'a self,
        resources: Option<&'a ResourceImages>,
    ) -> Result<GraphEvaluator<'a>, BakeError> {
        let images = self
            .nodes
            .iter()
            .map(|node| match node {
                GraphNode::Bitmap { resource, .. } => resources
                    .and_then(|images| images.get(resource))
                    .map(|image| Some(&**image))
                    .ok_or(BakeError::MissingResource(*resource)),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GraphEvaluator {
            graph: self,
            images,
        })
    }
}

/// A graph with its bitmaps resolved.
pub(crate) struct GraphEvaluator<'a> {
    graph: &'a MaterialGraph,
    images: Vec<Option<&'a TextureImage>>,
}

impl GraphEvaluator<'_> {
    /// The value `output` reads at `uv`, with the mesh maps there (`None`
    /// when the graph needs none).
    pub(crate) fn evaluate(
        &self,
        output: GraphInput,
        uv: Vec2,
        sample: Option<&MeshMapSample>,
    ) -> Vec3 {
        let Some(last) = output.node() else {
            return output.read(&[]);
        };
        let mut values = [Vec3::ZERO; MAX_GRAPH_NODES];
        let fallback = MeshMapSample::default();
        let maps = sample.unwrap_or(&fallback);
        for (index, node) in self.graph.nodes.iter().enumerate().take(last + 1) {
            let earlier = &values[..index];
            let value = match node {
                GraphNode::Generator(generator) => Vec3::splat(generator.evaluate(maps)),
                GraphNode::MeshMap(kind) => mesh_map_value(*kind, maps),
                GraphNode::Uv => uv.extend(0.0),
                GraphNode::Bitmap { projection, .. } => match self.images[index] {
                    Some(image) => {
                        let texel = projection.sample(image, uv, sample);
                        image.color_space.to_srgb(texel.truncate())
                    }
                    None => Vec3::ZERO,
                },
                GraphNode::Blend { mode, a, b, amount } => {
                    let (a, b) = (a.read(earlier), b.read(earlier));
                    let blended = Vec3::new(
                        mode.apply(a.x, b.x),
                        mode.apply(a.y, b.y),
                        mode.apply(a.z, b.z),
                    );
                    a.lerp(blended, luminance(amount.read(earlier)).clamp(0.0, 1.0))
                }
                GraphNode::Math { op, a, b } => {
                    let (a, b) = (a.read(earlier), b.read(earlier));
                    Vec3::new(op.apply(a.x, b.x), op.apply(a.y, b.y), op.apply(a.z, b.z))
                }
                GraphNode::Adjust { input, filter } => filter.map_color(input.read(earlier)),
                GraphNode::Ramp { input, stops } => ramp(stops, luminance(input.read(earlier))),
                GraphNode::Split { input, component } => {
                    let value = input.read(earlier);
                    Vec3::splat(match component {
                        SourceComponent::Red => value.x,
                        SourceComponent::Green => value.y,
                        SourceComponent::Blue => value.z,
                    })
                }
            };
            values[index] = if value.is_finite() { value } else { Vec3::ZERO };
        }
        output.read(&values)
    }
}

/// Rec. 709 luminance.
fn luminance(color: Vec3) -> f32 {
    color.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

fn mesh_map_value(kind: MeshMapKind, sample: &MeshMapSample) -> Vec3 {
    if !sample.covered {
        return Vec3::ZERO;
    }
    match kind {
        MeshMapKind::Normal => sample.tangent_normal * 0.5 + 0.5,
        MeshMapKind::WorldNormal => sample.normal * 0.5 + 0.5,
        MeshMapKind::Position => sample.unit_position,
        MeshMapKind::Curvature => Vec3::splat(sample.curvature),
        MeshMapKind::AmbientOcclusion => Vec3::splat(sample.ambient_occlusion),
        MeshMapKind::Thickness => Vec3::splat(sample.thickness),
        MeshMapKind::Skew => Vec3::splat(sample.skew),
    }
}

/// The ramp's sRGB colour at `amount`.
fn ramp(stops: &[RampStop], amount: f32) -> Vec3 {
    let unit = |color: [u8; 3]| Vec3::from(color.map(|channel| f32::from(channel) / 255.0));
    let amount = amount.clamp(0.0, 1.0);
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else {
        return Vec3::splat(amount);
    };
    if amount <= first.position {
        return unit(first.color);
    }
    for pair in stops.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        if amount <= to.position {
            let span = to.position - from.position;
            let t = if span <= f32::EPSILON {
                1.0
            } else {
                (amount - from.position) / span
            };
            return unit(from.color).lerp(unit(to.color), t);
        }
    }
    unit(last.color)
}
