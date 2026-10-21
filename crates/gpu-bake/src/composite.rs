//! GPU layer compositing for previews (PERF-06): runs a texture set's
//! [`CompositePlan`] as wgpu compute shaders (`composite.wgsl`) and returns
//! the same [`MaterialTextures`] [`TextureSet::bake_material`] makes,
//! within the tolerance TC-0412 checks. Exports keep baking on the CPU.
//!
//! The plan's steps run in order, one dispatch each, over the whole set:
//! a leaf layer's content, filters, mask, and blend run in one pass, and
//! groups composite into their own plane above the one beneath. Content and
//! masks the plan leaves to the CPU (paint strokes, blurs, quick masks,
//! anchors, polygon and ID selections) are computed when their step comes
//! and uploaded, one at a time, so memory stays bounded however many there
//! are.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::sync::mpsc::channel;

use glam::Vec3;
use substancecraft_core::{
    Axis, BakeContext, BakedTexture, BlendMode, ColorSpace, CompositePlan, DEFAULT_BASE_COLOR,
    DEFAULT_METALLIC, DEFAULT_OPACITY, DEFAULT_ROUGHNESS, GraphInput, GraphNode, LayerFilter,
    LayerId, MaskGenerator, MaterialChannel, MaterialTextures, MathOp, MeshMapKind, PatternKind,
    PlanContent, PlanError, PlanMask, PlanStep, PlaneKind, ProjectionMode, SourceComponent,
    TextureProjection, TextureSet,
};

use crate::{AdapterChoice, entry, open_device};

const SHADER: &str = include_str!("composite.wgsl");
const WORKGROUP: u32 = 256;
const NONE: u32 = u32::MAX;
/// vec4 rows per table entry, as `composite.wgsl` reads them.
const RECORD_ROWS: usize = 8;
const PROJECTION_ROWS: usize = 9;

/// Why the GPU did not composite a set; callers bake on the CPU instead.
#[derive(Debug, thiserror::Error)]
pub enum CompositeError {
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error("GPU compositing failed: {0}")]
    Gpu(String),
}

impl From<substancecraft_core::BakeError> for CompositeError {
    fn from(error: substancecraft_core::BakeError) -> Self {
        Self::Plan(PlanError::Bake(error))
    }
}

fn gpu(error: impl std::fmt::Display) -> CompositeError {
    CompositeError::Gpu(error.to_string())
}

struct Pipelines {
    start_plane: wgpu::ComputePipeline,
    layer: wgpu::ComputePipeline,
    begin_group: wgpu::ComputePipeline,
    end_group: wgpu::ComputePipeline,
    defaults: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    normal_detail: wgpu::ComputePipeline,
    make_mask: wgpu::ComputePipeline,
}

/// A GPU device ready to composite layer stacks.
pub struct GpuCompositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    name: String,
    max_binding: u64,
    uniform_stride: u64,
}

/// The tables `composite.wgsl` reads, as rows of four words.
#[derive(Default)]
struct Tables {
    rows: Vec<[u32; 4]>,
}

impl Tables {
    fn push(&mut self, row: [u32; 4]) -> u32 {
        self.rows.push(row);
        (self.rows.len() - 1) as u32
    }

    fn floats(&mut self, values: [f32; 4]) -> u32 {
        self.push(values.map(f32::to_bits))
    }
}

fn floats(values: [f32; 4]) -> [u32; 4] {
    values.map(f32::to_bits)
}

/// What one dispatch runs.
#[derive(Clone, Copy)]
enum Entry {
    StartPlane,
    Layer,
    BeginGroup,
    EndGroup,
    Defaults,
    Pack,
    NormalDetail,
    MakeMask,
}

/// One dispatch: its entry point, its `Dispatch` uniform, and the CPU
/// content or mask to upload first.
struct Dispatch {
    entry: Entry,
    words: [u32; 4],
    content: Option<(LayerId, MaterialChannel)>,
    mask: Option<LayerId>,
}

/// A record's mask fields: kind and generator (row 3, lanes 2–3), the
/// bitmap (row 4), filters and inversion (row 5, lanes 0–2), and the CPU
/// mask to upload.
struct MaskWords {
    head: [u32; 2],
    bitmap: [u32; 4],
    tail: [u32; 3],
    upload: Option<LayerId>,
}

/// Indices of the per-kind tables, in entries, while they are built.
#[derive(Default)]
struct Encoded {
    projections: Vec<[u32; 4]>,
    filters: Vec<[u32; 4]>,
    generators: Vec<[u32; 4]>,
    nodes: Vec<[u32; 4]>,
    stops: Vec<[u32; 4]>,
    records: Vec<[u32; 4]>,
}

impl Encoded {
    fn projection(&mut self, projection: &TextureProjection) -> u32 {
        let index = (self.projections.len() / PROJECTION_ROWS) as u32;
        let mode = match projection.mode {
            ProjectionMode::Uv => 0,
            ProjectionMode::Triplanar => 1,
            ProjectionMode::Planar => 2,
            ProjectionMode::Cylindrical => 3,
            ProjectionMode::Spherical => 4,
            ProjectionMode::Decal => 5,
        };
        let axis = axis_code(projection.axis);
        let (sin, cos) =
            substancecraft_core::math::sin_cos(projection.rotation_degrees.to_radians());
        let decal = projection.decal;
        // DecalPlacement::locate's basis, computed as the CPU does.
        let normal = decal.normal.normalize_or(Vec3::Z);
        let right = decal.up.cross(normal).normalize_or(Vec3::X);
        let up = normal.cross(right);
        let point = |v: Vec3| floats([v.x, v.y, v.z, 0.0]);
        self.projections.extend([
            [mode, axis, u32::from(projection.tile_size.is_some()), 0],
            floats([projection.scale, sin, cos, projection.blend_sharpness]),
            floats([
                projection.offset.x,
                projection.offset.y,
                projection.tile_size.unwrap_or(1.0),
                decal.depth,
            ]),
            point(decal.center),
            point(normal),
            point(right),
            point(up),
            floats([
                decal.size.x,
                decal.size.y,
                decal.region_min.x,
                decal.region_min.y,
            ]),
            floats([decal.region_max.x, decal.region_max.y, 0.0, 0.0]),
        ]);
        index
    }

    /// Per-texel filters, in order: the start and count.
    fn filter_list(&mut self, filters: &[LayerFilter]) -> Result<(u32, u32), CompositeError> {
        let start = (self.filters.len() / 3) as u32;
        for filter in filters {
            self.filter(*filter)?;
        }
        Ok((start, filters.len() as u32))
    }

    fn filter(&mut self, filter: LayerFilter) -> Result<u32, CompositeError> {
        let index = (self.filters.len() / 3) as u32;
        let unit = |color: [u8; 3]| color.map(|channel| f32::from(channel) / 255.0);
        let (kind, levels, a, b) = match filter {
            LayerFilter::Levels {
                in_low,
                in_high,
                gamma,
                out_low,
                out_high,
            } => (
                0,
                0,
                [in_low, in_high, gamma, out_low],
                [out_high, 0.0, 0.0, 0.0],
            ),
            LayerFilter::Invert => (1, 0, [0.0; 4], [0.0; 4]),
            LayerFilter::HueSaturation {
                hue,
                saturation,
                lightness,
            } => (2, 0, [hue, saturation, lightness, 0.0], [0.0; 4]),
            LayerFilter::BrightnessContrast {
                brightness,
                contrast,
            } => (3, 0, [brightness, contrast, 0.0, 0.0], [0.0; 4]),
            LayerFilter::Posterize { levels } => (4, levels, [0.0; 4], [0.0; 4]),
            LayerFilter::Threshold { level, softness } => {
                (5, 0, [level, softness, 0.0, 0.0], [0.0; 4])
            }
            LayerFilter::GradientMap { dark, light } => {
                let (dark, light) = (unit(dark), unit(light));
                (
                    6,
                    0,
                    [dark[0], dark[1], dark[2], 0.0],
                    [light[0], light[1], light[2], 0.0],
                )
            }
            LayerFilter::Blur { .. } | LayerFilter::Sharpen { .. } | LayerFilter::Plugin(_) => {
                return Err(PlanError::Unsupported("a filter the GPU does not run").into());
            }
        };
        self.filters
            .extend([[kind, levels, 0, 0], floats(a), floats(b)]);
        Ok(index)
    }

    fn generator(&mut self, generator: &MaskGenerator) -> Result<u32, CompositeError> {
        let index = (self.generators.len() / 3) as u32;
        let feature = |size: Option<f32>| (u32::from(size.is_some()), size.unwrap_or(1.0));
        let (head, a, b) = match generator {
            MaskGenerator::EdgeWear(settings) => {
                let (flag, size) = feature(settings.feature_size);
                (
                    [0, settings.seed, 0, flag],
                    [
                        settings.amount,
                        settings.contrast,
                        settings.breakup,
                        settings.breakup_scale,
                    ],
                    [size, 0.0, 0.0, 0.0],
                )
            }
            MaskGenerator::Dirt(settings) => {
                let (flag, size) = feature(settings.feature_size);
                (
                    [1, settings.seed, 0, flag],
                    [
                        settings.amount,
                        settings.contrast,
                        settings.cavity_weight,
                        settings.breakup,
                    ],
                    [settings.breakup_scale, size, 0.0, 0.0],
                )
            }
            MaskGenerator::Orientation(settings) => {
                let direction = settings.direction.vector();
                (
                    [2, 0, 0, 0],
                    [
                        direction.x,
                        direction.y,
                        direction.z,
                        settings.spread_degrees,
                    ],
                    [settings.softness, 0.0, 0.0, 0.0],
                )
            }
            MaskGenerator::Gradient(settings) => (
                [3, 0, axis_code(settings.axis), 0],
                [settings.start, settings.end, 0.0, 0.0],
                [0.0; 4],
            ),
            MaskGenerator::Noise(settings) => {
                let (flag, size) = feature(settings.feature_size);
                (
                    [4, settings.seed, 0, flag],
                    [settings.scale, settings.amount, settings.contrast, size],
                    [0.0; 4],
                )
            }
            MaskGenerator::Thickness(settings) => (
                [5, 0, 0, u32::from(settings.thin) << 1],
                [settings.amount, settings.contrast, 0.0, 0.0],
                [0.0; 4],
            ),
            MaskGenerator::Pattern(settings) => {
                let (flag, size) = feature(settings.feature_size);
                let kind = match settings.kind {
                    PatternKind::Stripes => 0,
                    PatternKind::Checker => 1,
                    PatternKind::Dots => 2,
                    PatternKind::Scratches => 3,
                    PatternKind::Grunge => 4,
                };
                (
                    [
                        6,
                        settings.seed,
                        axis_code(settings.axis),
                        flag | (kind << 8),
                    ],
                    [settings.scale, settings.width, settings.softness, size],
                    [0.0; 4],
                )
            }
            MaskGenerator::IdColor(_) | MaskGenerator::Plugin(_) => {
                return Err(PlanError::Unsupported("a generator the GPU does not run").into());
            }
        };
        self.generators.extend([head, floats(a), floats(b)]);
        Ok(index)
    }

    fn input(input: GraphInput) -> [u32; 4] {
        match input {
            GraphInput::Value([x, y, z]) => floats([x, y, z, -1.0]),
            GraphInput::Node(index) => floats([0.0, 0.0, 0.0, f32::from(index)]),
        }
    }

    /// Appends a graph's nodes; returns where they start.
    fn graph(
        &mut self,
        graph: &substancecraft_core::MaterialGraph,
        plan: &CompositePlan<'_>,
    ) -> Result<u32, CompositeError> {
        let start = (self.nodes.len() / 4) as u32;
        let zero = GraphInput::gray(0.0);
        for node in &graph.nodes {
            let (head, inputs) = match node {
                GraphNode::Generator(generator) => {
                    ([0, self.generator(generator)?, 0, 0], [zero; 3])
                }
                GraphNode::MeshMap(kind) => {
                    let code = match kind {
                        MeshMapKind::Normal => 0,
                        MeshMapKind::WorldNormal => 1,
                        MeshMapKind::Position => 2,
                        MeshMapKind::Curvature => 3,
                        MeshMapKind::AmbientOcclusion => 4,
                        MeshMapKind::Thickness => 5,
                        MeshMapKind::Skew => 6,
                    };
                    ([1, code, 0, 0], [zero; 3])
                }
                GraphNode::Uv => ([2, 0, 0, 0], [zero; 3]),
                GraphNode::Bitmap {
                    resource,
                    projection,
                } => {
                    let image = plan
                        .image_index(*resource)
                        .ok_or(substancecraft_core::BakeError::MissingResource(*resource))?
                        as u32;
                    ([3, image, self.projection(projection), 0], [zero; 3])
                }
                GraphNode::Blend { mode, a, b, amount } => {
                    ([4, blend_code(*mode), 0, 0], [*a, *b, *amount])
                }
                GraphNode::Math { op, a, b } => {
                    let code = match op {
                        MathOp::Add => 0,
                        MathOp::Subtract => 1,
                        MathOp::Multiply => 2,
                        MathOp::Divide => 3,
                        MathOp::Minimum => 4,
                        MathOp::Maximum => 5,
                        MathOp::Power => 6,
                    };
                    ([5, code, 0, 0], [*a, *b, zero])
                }
                GraphNode::Adjust { input, filter } => {
                    ([6, self.filter(*filter)?, 0, 0], [*input, zero, zero])
                }
                GraphNode::Ramp { input, stops } => {
                    let first = self.stops.len() as u32;
                    for stop in stops {
                        let [r, g, b] = stop.color.map(|channel| f32::from(channel) / 255.0);
                        self.stops.push(floats([stop.position, r, g, b]));
                    }
                    ([7, first, stops.len() as u32, 0], [*input, zero, zero])
                }
                GraphNode::Split { input, component } => {
                    let code = match component {
                        SourceComponent::Red => 0,
                        SourceComponent::Green => 1,
                        SourceComponent::Blue => 2,
                    };
                    ([8, code, 0, 0], [*input, zero, zero])
                }
            };
            self.nodes.extend([
                head,
                Self::input(inputs[0]),
                Self::input(inputs[1]),
                Self::input(inputs[2]),
            ]);
        }
        Ok(start)
    }

    /// The mask words of a record: rows 3 (lanes 2–3), 4, and 5 (lanes
    /// 0–2), and the CPU mask to upload.
    fn mask(&mut self, mask: &PlanMask) -> Result<MaskWords, CompositeError> {
        let words = |head: [u32; 2], upload: Option<LayerId>| MaskWords {
            head,
            bitmap: [0; 4],
            tail: [0; 3],
            upload,
        };
        Ok(match mask {
            PlanMask::Open => words([0, NONE], None),
            PlanMask::Cpu(layer) => words([2, NONE], Some(*layer)),
            PlanMask::Generated {
                generator,
                bitmap,
                filters,
                invert,
            } => {
                let generator = match generator {
                    Some(generator) => self.generator(generator)?,
                    None => NONE,
                };
                let bitmap_words = match bitmap {
                    Some((image, bitmap)) => [
                        *image as u32,
                        self.projection(&bitmap.projection),
                        1,
                        u32::from(bitmap.invert),
                    ],
                    None => [0; 4],
                };
                let (start, count) = self.filter_list(filters)?;
                MaskWords {
                    head: [1, generator],
                    bitmap: bitmap_words,
                    tail: [start, count, u32::from(*invert)],
                    upload: None,
                }
            }
        })
    }

    /// Encodes a layer or group-end step; returns its record index and the
    /// CPU uploads it needs.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        plan: &CompositePlan<'_>,
        content: Option<&PlanContent>,
        filters: &[LayerFilter],
        mask: &PlanMask,
        opacity: f32,
        blend: BlendMode,
        pass_through: bool,
        mask_slot: Option<u32>,
    ) -> Result<(u32, Option<LayerId>, Option<LayerId>), CompositeError> {
        let index = (self.records.len() / RECORD_ROWS) as u32;
        let mut rows = [[0_u32; 4]; RECORD_ROWS];
        rows[1][2] = NONE;
        let mut upload_content = None;
        match content {
            None => {}
            Some(PlanContent::Uniform { texel, decal }) => {
                rows[0] = [0, 0, 0, 0];
                rows[2] = floats(texel.to_array());
                if let Some(projection) = decal {
                    rows[0][2] = self.projection(projection);
                    rows[1][3] = 1;
                }
            }
            Some(PlanContent::Bitmap {
                image,
                projection,
                copy,
            }) => {
                rows[0] = [
                    1,
                    *image as u32,
                    self.projection(projection),
                    u32::from(*copy),
                ];
            }
            Some(PlanContent::Graph { graph, output }) => {
                let graph = plan
                    .graphs
                    .get(*graph)
                    .ok_or(PlanError::Unsupported("a graph the plan lost"))?;
                let start = self.graph(graph, plan)?;
                rows[0] = [2, 0, 0, 0];
                match output {
                    GraphInput::Value([x, y, z]) => rows[2] = floats([*x, *y, *z, 1.0]),
                    GraphInput::Node(node) => {
                        rows[1] = [start, graph.nodes.len() as u32, u32::from(*node), 0];
                    }
                }
                if let GraphInput::Value(_) = output {
                    rows[1] = [start, 0, NONE, 0];
                }
            }
            Some(PlanContent::Cpu(layer)) => {
                rows[0] = [3, 0, 0, 0];
                upload_content = Some(*layer);
            }
        }
        let (start, count) = self.filter_list(filters)?;
        let mask = match mask_slot {
            Some(slot) => MaskWords {
                head: [3, slot],
                bitmap: [0; 4],
                tail: [0; 3],
                upload: None,
            },
            None => self.mask(mask)?,
        };
        rows[3] = [start, count, mask.head[0], mask.head[1]];
        rows[4] = mask.bitmap;
        rows[5] = [mask.tail[0], mask.tail[1], mask.tail[2], blend_code(blend)];
        let upload_mask = mask.upload;
        rows[6] = [opacity.to_bits(), u32::from(pass_through), 0, 0];
        self.records.extend(rows);
        Ok((index, upload_content, upload_mask))
    }
}

fn axis_code(axis: Axis) -> u32 {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

fn blend_code(mode: BlendMode) -> u32 {
    match mode {
        BlendMode::Normal => 0,
        BlendMode::Multiply => 1,
        BlendMode::Screen => 2,
        BlendMode::Overlay => 3,
        BlendMode::SoftLight => 4,
        BlendMode::Darken => 5,
        BlendMode::Lighten => 6,
        BlendMode::Difference => 7,
        BlendMode::Add => 8,
        BlendMode::Subtract => 9,
        BlendMode::PassThrough => 10,
    }
}

fn data_code(kind: PlaneKind) -> u32 {
    match kind {
        PlaneKind::Color => 0,
        PlaneKind::Scalar => 1,
        PlaneKind::Vector => 2,
    }
}

fn channel_code(channel: MaterialChannel) -> Option<u32> {
    Some(match channel {
        MaterialChannel::BaseColor => 0,
        MaterialChannel::Roughness => 1,
        MaterialChannel::Metallic => 2,
        MaterialChannel::Emissive => 3,
        MaterialChannel::Opacity => 4,
        MaterialChannel::Height => 5,
        MaterialChannel::Normal => 6,
        _ => return None,
    })
}

fn space_code(space: ColorSpace) -> u32 {
    match space {
        ColorSpace::Srgb => 0,
        ColorSpace::LinearSrgb => 1,
        ColorSpace::AcesCg => 2,
        ColorSpace::Raw => 3,
    }
}

fn vec4_bytes(values: impl IntoIterator<Item = [f32; 4]>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|row| row.into_iter().flat_map(f32::to_le_bytes))
        .collect()
}

impl GpuCompositor {
    /// Opens the highest-performance adapter `choice` allows.
    pub fn new(choice: AdapterChoice) -> Result<Self, String> {
        pollster::block_on(Self::open(choice))
    }

    /// One hardware compositor for the whole process, opened on first use;
    /// `None` when there is no usable GPU.
    pub fn shared() -> Option<&'static Self> {
        static SHARED: OnceLock<Option<GpuCompositor>> = OnceLock::new();
        SHARED
            .get_or_init(|| Self::new(AdapterChoice::Hardware).ok())
            .as_ref()
    }

    /// The adapter, for logs.
    pub fn name(&self) -> &str {
        &self.name
    }

    async fn open(choice: AdapterChoice) -> Result<Self, String> {
        let (device, queue, info, limits) = open_device(choice, "layer compositor").await?;
        if limits.max_storage_buffers_per_shader_stage < 8 {
            return Err("the GPU binds too few storage buffers".to_owned());
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32, dynamic: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compositor"),
            entries: &[
                uniform(0, false),
                uniform(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, true),
                storage(5, true),
                storage(6, true),
                storage(7, true),
                storage(8, true),
                storage(9, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compositor"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compositor"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            start_plane: pipeline("start_plane"),
            layer: pipeline("layer"),
            begin_group: pipeline("begin_group"),
            end_group: pipeline("end_group"),
            defaults: pipeline("defaults"),
            pack: pipeline("pack"),
            normal_detail: pipeline("normal_detail"),
            make_mask: pipeline("make_mask"),
        };
        if let Some(error) = scope.pop().await {
            return Err(error.to_string());
        }
        Ok(Self {
            device,
            queue,
            layout,
            pipelines,
            name: format!("{} ({:?})", info.name, info.backend),
            max_binding: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
            uniform_stride: u64::from(limits.min_uniform_buffer_offset_alignment).max(256),
        })
    }

    fn buffer(
        &self,
        label: &str,
        bytes: &[u8],
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, CompositeError> {
        let size = (bytes.len() as u64).max(16).next_multiple_of(16);
        if usage.contains(wgpu::BufferUsages::STORAGE) && size > self.max_binding {
            return Err(gpu(format!(
                "{label} needs {size} bytes, over the GPU's limit"
            )));
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.queue.write_buffer(&buffer, 0, bytes);
        }
        Ok(buffer)
    }

    /// Bakes `set`'s material as [`TextureSet::bake_material`] does, on the
    /// GPU. Fails (and callers bake on the CPU) for stacks the plan refuses
    /// and whenever the GPU does.
    pub fn bake_material(
        &self,
        set: &TextureSet,
        context: BakeContext<'_>,
    ) -> Result<MaterialTextures, CompositeError> {
        let plan = CompositePlan::new(set, context)?;
        let (width, height) = (set.resolution.width, set.resolution.height);
        let count = width as usize * height as usize;
        if count == 0 {
            return Err(PlanError::Unsupported("an empty set").into());
        }

        // Tables: constants first, then everything the steps reference.
        let mut constants = Tables::default();
        let default_base = Vec3::from(DEFAULT_BASE_COLOR.map(|channel| f32::from(channel) / 255.0));
        let defaults_row = constants.floats([
            default_base.x,
            default_base.y,
            default_base.z,
            DEFAULT_ROUGHNESS,
        ]);
        constants.floats([DEFAULT_METALLIC, DEFAULT_OPACITY, 0.0, 0.0]);
        let base_color_row =
            constants.floats([default_base.x, default_base.y, default_base.z, 0.0]);
        let mut encoded = Encoded::default();
        let mut dispatches = vec![Dispatch {
            entry: Entry::Defaults,
            words: [defaults_row, 0, 0, 0],
            content: None,
            mask: None,
        }];
        let height_scale = plan.height_detail_scale();
        let levels = plan.depth as u32 + 1;

        // Each layer's mask is the same in every channel: computed once into
        // a slot when the slots fit in memory, else in every channel.
        let mut masked: Vec<(LayerId, &PlanMask)> = Vec::new();
        for step in plan.channels.iter().flat_map(|channel| &channel.steps) {
            let (PlanStep::Layer { layer, mask, .. } | PlanStep::EndGroup { layer, mask, .. }) =
                step
            else {
                continue;
            };
            if *mask != PlanMask::Open && !masked.iter().any(|(known, _)| known == layer) {
                masked.push((*layer, mask));
            }
        }
        let all_slot_rows = masked.len().div_ceil(4) as u64;
        let planes_size = (u64::from(levels) + 1 + all_slot_rows) * count as u64 * 16;
        let mut slots: BTreeMap<LayerId, u32> = BTreeMap::new();
        if planes_size <= self.max_binding {
            for (slot, (layer, mask)) in masked.iter().enumerate() {
                let (record, _, cpu_mask) =
                    encoded.record(&plan, None, &[], mask, 1.0, BlendMode::Normal, false, None)?;
                dispatches.push(Dispatch {
                    entry: Entry::MakeMask,
                    words: [record, slot as u32, 0, 0],
                    content: None,
                    mask: cpu_mask,
                });
                slots.insert(*layer, slot as u32);
            }
        }
        let slot_rows = slots.len().div_ceil(4) as u64;
        for channel in &plan.channels {
            let code = channel_code(channel.channel)
                .ok_or(PlanError::Unsupported("a channel the GPU does not pack"))?;
            let data = data_code(channel.kind);
            let start = if channel.base_from_maps {
                NONE
            } else {
                constants.floats(channel.base.to_array())
            };
            dispatches.push(Dispatch {
                entry: Entry::StartPlane,
                words: [start, 0, data, code],
                content: None,
                mask: None,
            });
            let mut level = 0_u32;
            for step in &channel.steps {
                match step {
                    PlanStep::Layer {
                        layer,
                        content,
                        filters,
                        mask,
                        opacity,
                        blend,
                    } => {
                        let (record, cpu_content, cpu_mask) = encoded.record(
                            &plan,
                            Some(content),
                            filters,
                            mask,
                            *opacity,
                            *blend,
                            false,
                            slots.get(layer).copied(),
                        )?;
                        dispatches.push(Dispatch {
                            entry: Entry::Layer,
                            words: [record, level, data, code],
                            content: cpu_content.map(|layer| (layer, channel.channel)),
                            mask: cpu_mask,
                        });
                    }
                    PlanStep::BeginGroup { pass_through } => {
                        dispatches.push(Dispatch {
                            entry: Entry::BeginGroup,
                            words: [u32::from(*pass_through), level, data, code],
                            content: None,
                            mask: None,
                        });
                        level += 1;
                    }
                    PlanStep::EndGroup {
                        layer,
                        filters,
                        mask,
                        opacity,
                        blend,
                        pass_through,
                    } => {
                        level = level.saturating_sub(1);
                        let (record, _, cpu_mask) = encoded.record(
                            &plan,
                            None,
                            filters,
                            mask,
                            *opacity,
                            *blend,
                            *pass_through,
                            slots.get(layer).copied(),
                        )?;
                        dispatches.push(Dispatch {
                            entry: Entry::EndGroup,
                            words: [record, level, data, code],
                            content: None,
                            mask: cpu_mask,
                        });
                    }
                }
            }
            if channel.channel == MaterialChannel::Normal && height_scale > 0.0 {
                dispatches.push(Dispatch {
                    entry: Entry::NormalDetail,
                    words: [0, 0, data, code],
                    content: None,
                    mask: None,
                });
            }
            dispatches.push(Dispatch {
                entry: Entry::Pack,
                words: [base_color_row, 0, data, code],
                content: None,
                mask: None,
            });
        }

        // Images: headers and pixels, 8- or 16-bit.
        let mut image_rows = Vec::new();
        let mut pixels: Vec<u32> = Vec::new();
        for (_, image) in &plan.images {
            let offset = pixels.len() as u32;
            let sixteen = image.rgba16.is_some();
            match &image.rgba16 {
                Some(rgba16) => {
                    for pixel in rgba16.as_chunks::<4>().0 {
                        pixels.push(u32::from(pixel[0]) | (u32::from(pixel[1]) << 16));
                        pixels.push(u32::from(pixel[2]) | (u32::from(pixel[3]) << 16));
                    }
                }
                None => pixels.extend(
                    image
                        .rgba8
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|pixel| u32::from_le_bytes(*pixel)),
                ),
            }
            image_rows.push([offset, image.width, image.height, u32::from(sixteen)]);
            image_rows.push([space_code(image.color_space), 0, 0, 0]);
        }
        while !pixels.len().is_multiple_of(4) {
            pixels.push(0);
        }

        let mut rows = constants.rows;
        let mut section = |table: &[[u32; 4]]| {
            let start = rows.len() as u32;
            rows.extend_from_slice(table);
            start
        };
        let records = section(&encoded.records);
        let projections = section(&encoded.projections);
        let filters = section(&encoded.filters);
        let generators = section(&encoded.generators);
        let nodes = section(&encoded.nodes);
        let stops = section(&encoded.stops);
        let images = section(&image_rows);
        let pixel_rows: Vec<[u32; 4]> = pixels.as_chunks::<4>().0.to_vec();
        let pixels_start = section(&pixel_rows);

        let maps = plan.mesh_maps();
        let (maps_width, maps_height, bounds_min, bounds_max, diagonal) = match maps {
            Some(maps) => {
                let (min, max, diagonal) = maps.sample_bounds();
                (maps.width(), maps.height(), min, max, diagonal)
            }
            None => (0, 0, Vec3::ZERO, Vec3::ZERO, 1.0),
        };
        let mut globals = Vec::with_capacity(96);
        for word in [
            width,
            height,
            count as u32,
            levels,
            maps_width,
            maps_height,
            u32::from(maps.is_some()),
            (levels + 1) * count as u32,
        ] {
            globals.extend(word.to_le_bytes());
        }
        globals.extend(vec4_bytes([
            [bounds_min.x, bounds_min.y, bounds_min.z, 0.0],
            [bounds_max.x, bounds_max.y, bounds_max.z, 0.0],
            [diagonal, height_scale, 0.0, 0.0],
        ]));
        for word in [
            records,
            projections,
            filters,
            generators,
            nodes,
            stops,
            images,
            pixels_start,
        ] {
            globals.extend(word.to_le_bytes());
        }

        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let usage = wgpu::BufferUsages::STORAGE;
        let globals = self.buffer("globals", &globals, wgpu::BufferUsages::UNIFORM)?;
        let stride = self.uniform_stride as usize;
        let mut dispatch_bytes = vec![0_u8; dispatches.len() * stride];
        for (index, dispatch) in dispatches.iter().enumerate() {
            for (lane, word) in dispatch.words.iter().enumerate() {
                let at = index * stride + lane * 4;
                dispatch_bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let dispatch_buffer =
            self.buffer("dispatches", &dispatch_bytes, wgpu::BufferUsages::UNIFORM)?;
        let tables = self.buffer("tables", bytemuck::cast_slice(&rows), usage)?;
        let plane_bytes = (u64::from(levels) + 1 + slot_rows) * count as u64 * 16;
        if plane_bytes > self.max_binding {
            return Err(gpu("the planes need more memory than the GPU binds"));
        }
        let planes = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planes"),
            size: plane_bytes,
            usage,
            mapped_at_creation: false,
        });
        let uploads_needed = dispatches
            .iter()
            .any(|dispatch| dispatch.content.is_some() || dispatch.mask.is_some());
        let upload = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upload"),
            size: if uploads_needed {
                (2 * count as u64 * 16).min(self.max_binding)
            } else {
                16
            },
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if uploads_needed && 2 * count as u64 * 16 > self.max_binding {
            return Err(gpu("uploads need more memory than the GPU binds"));
        }
        let map_planes = match maps {
            Some(maps) => maps.texel_planes(),
            None => std::array::from_fn(|_| Vec::new()),
        };
        let map_buffers = map_planes
            .iter()
            .map(|plane| self.buffer("mesh maps", bytemuck::cast_slice(plane), usage))
            .collect::<Result<Vec<_>, _>>()?;
        drop(map_planes);
        let output_size = count as u64 * 16;
        let outputs = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outputs"),
            size: output_size,
            usage: usage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor"),
            layout: &self.layout,
            entries: &[
                entry(0, &globals),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dispatch_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(16),
                    }),
                },
                entry(2, &tables),
                entry(3, &planes),
                entry(4, &upload),
                entry(5, &map_buffers[0]),
                entry(6, &map_buffers[1]),
                entry(7, &map_buffers[2]),
                entry(8, &map_buffers[3]),
                entry(9, &outputs),
            ],
        });

        let groups = (count as u32).div_ceil(WORKGROUP);
        let mut masks: BTreeMap<LayerId, Vec<f32>> = BTreeMap::new();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pending = 0_usize;
        for (index, dispatch) in dispatches.iter().enumerate() {
            if dispatch.content.is_some() || dispatch.mask.is_some() {
                // An upload lands before the next submission, so everything
                // recorded so far goes first.
                self.queue.submit([encoder.finish()]);
                encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                pending = 0;
                if let Some((layer, channel)) = dispatch.content {
                    let texels = plan.content(layer, channel)?;
                    self.queue
                        .write_buffer(&upload, 0, bytemuck::cast_slice(&texels));
                }
                if let Some(layer) = dispatch.mask {
                    if let std::collections::btree_map::Entry::Vacant(slot) = masks.entry(layer) {
                        slot.insert(plan.mask(layer)?);
                    }
                    let values = masks.get(&layer).map(Vec::as_slice).unwrap_or_default();
                    self.queue.write_buffer(
                        &upload,
                        count as u64 * 16,
                        bytemuck::cast_slice(values),
                    );
                }
            }
            let pipeline = match dispatch.entry {
                Entry::StartPlane => &self.pipelines.start_plane,
                Entry::Layer => &self.pipelines.layer,
                Entry::BeginGroup => &self.pipelines.begin_group,
                Entry::EndGroup => &self.pipelines.end_group,
                Entry::Defaults => &self.pipelines.defaults,
                Entry::Pack => &self.pipelines.pack,
                Entry::NormalDetail => &self.pipelines.normal_detail,
                Entry::MakeMask => &self.pipelines.make_mask,
            };
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[(index * stride) as u32]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            pending += 1;
            // Short submissions keep any one from running long enough for
            // the OS to reset the GPU.
            if pending >= 16 {
                self.queue.submit([encoder.finish()]);
                encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                pending = 0;
            }
        }
        encoder.copy_buffer_to_buffer(&outputs, 0, &readback, 0, output_size);
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (sender, receiver) = channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(gpu)?;
        receiver.recv().map_err(gpu)?.map_err(gpu)?;
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(gpu(error));
        }
        let mut textures: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::with_capacity(count * 4));
        {
            let bytes = slice.get_mapped_range().map_err(gpu)?;
            for texel in bytes.as_chunks::<16>().0 {
                for (texture, word) in textures.iter_mut().zip(texel.as_chunks::<4>().0) {
                    texture.extend_from_slice(word);
                }
            }
        }
        readback.unmap();
        let [base_color, orm, normal, emissive] = textures;
        let texture = |rgba8: Vec<u8>| BakedTexture {
            width,
            height,
            rgba8,
        };
        Ok(MaterialTextures {
            base_color: texture(base_color),
            orm: texture(orm),
            normal: texture(normal),
            emissive: texture(emissive),
        })
    }
}

/// Mean difference in 8-bit levels between two bakes of the same size,
/// and the share of channel values more than `slack` levels apart, for
/// tolerance checks.
pub fn texture_differences(a: &BakedTexture, b: &BakedTexture, slack: u8) -> (f64, f64) {
    let mut total = 0_u64;
    let mut far = 0_usize;
    for (x, y) in a.rgba8.iter().zip(&b.rgba8) {
        let difference = x.abs_diff(*y);
        total += u64::from(difference);
        far += usize::from(difference > slack);
    }
    let count = a.rgba8.len().max(1) as f64;
    (total as f64 / count, far as f64 / count)
}
