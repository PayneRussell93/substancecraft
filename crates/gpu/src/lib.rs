//! GPU compute paths for SubstanceCraft bakes (PERF-01), on wgpu.
//!
//! [`GpuTracer`] traces the ambient-occlusion and thickness rays of a
//! mesh-map bake in a compute shader (`occlusion.wgsl`), a port of the CPU
//! tracer in `substancecraft_core::occlusion` that follows the same sample
//! directions and BVH. Every other map is still evaluated on the CPU, where
//! it costs little. [`FallbackTracer`] traces on the GPU while it works and
//! on the CPU otherwise, so a bake never fails for want of a GPU.
//!
//! GPU results match the CPU tracer within floating-point tolerance, not
//! bit for bit, so deterministic exports (the command line) stay on the CPU.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use substancecraft_core::occlusion::{Bvh, THICKNESS_SEED};
use substancecraft_core::{
    CpuTracer, MeshMapError, OcclusionParams, OcclusionQuery, OcclusionTracer,
};
use thiserror::Error;

const SHADER: &str = include_str!("occlusion.wgsl");
const WORKGROUP_SIZE: u32 = 64;
/// Rays one dispatch may cast. Each dispatch is waited for before the next,
/// so none runs long enough to trip the driver's watchdog, and a cancelled
/// bake stops promptly.
const RAYS_PER_DISPATCH: u64 = 1 << 22;
/// Queries uploaded at once.
const QUERIES_PER_UPLOAD: usize = 1 << 18;
const NODE_BYTES: u64 = 32;
const TRIANGLE_BYTES: u64 = 48;
const QUERY_BYTES: u64 = 48;
const RESULT_BYTES: u64 = 8;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum GpuError {
    #[error("no suitable GPU: {0}")]
    NoAdapter(String),
    #[error("could not open the GPU: {0}")]
    Device(String),
    #[error("the scene needs {needed} bytes in one buffer; this GPU allows {allowed}")]
    TooLarge { needed: u64, allowed: u64 },
    #[error("GPU error: {0}")]
    Gpu(String),
}

#[derive(Debug, Error, PartialEq)]
pub enum TraceError {
    #[error("tracing was cancelled")]
    Cancelled,
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// A scene uploaded for tracing, kept while the same BVH is traced.
struct Scene {
    id: u64,
    nodes: wgpu::Buffer,
    triangles: wgpu::Buffer,
}

/// Traces occlusion queries in a wgpu compute shader.
pub struct GpuTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    adapter: String,
    max_binding: u64,
    /// Errors the device reported since they were last read.
    errors: Arc<Mutex<Vec<String>>>,
    lost: Arc<AtomicBool>,
    /// The last scene uploaded; holding it also keeps traces one at a time.
    scene: Mutex<Option<Scene>>,
}

impl GpuTracer {
    /// Opens the best hardware GPU. Software adapters are refused: the CPU
    /// tracer outruns them.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::open(false))
    }

    /// Opens any adapter, a software one included (for tests on machines
    /// without a GPU).
    pub fn new_allowing_software() -> Result<Self, GpuError> {
        pollster::block_on(Self::open(true))
    }

    /// One tracer shared by every bake in the process, opened on first use.
    pub fn shared() -> Result<&'static GpuTracer, GpuError> {
        static SHARED: OnceLock<Result<GpuTracer, GpuError>> = OnceLock::new();
        SHARED
            .get_or_init(GpuTracer::new)
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The adapter's name and backend, for logs.
    pub fn adapter_name(&self) -> &str {
        &self.adapter
    }

    async fn open(allow_software: bool) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|error| GpuError::NoAdapter(error.to_string()))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu && !allow_software {
            return Err(GpuError::NoAdapter(format!(
                "{} is a software adapter",
                info.name
            )));
        }
        if !adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            return Err(GpuError::NoAdapter(format!(
                "{} cannot run compute shaders",
                info.name
            )));
        }
        let available = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: available.max_storage_buffer_binding_size,
            max_buffer_size: available.max_buffer_size,
            ..wgpu::Limits::downlevel_defaults()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("substancecraft-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| GpuError::Device(error.to_string()))?;
        let errors = Arc::new(Mutex::new(Vec::new()));
        let reported = Arc::clone(&errors);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            if let Ok(mut errors) = reported.lock() {
                errors.push(error.to_string());
            }
        }));
        let lost = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&lost);
        device.set_device_lost_callback(move |_, _| flag.store(true, Ordering::Relaxed));

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("occlusion"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("occlusion"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("occlusion"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("occlusion"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let tracer = Self {
            device,
            queue,
            pipeline,
            layout,
            adapter: format!("{} ({:?})", info.name, info.backend),
            max_binding: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
            errors,
            lost,
            scene: Mutex::new(None),
        };
        tracer.check()?;
        Ok(tracer)
    }

    /// The first error the device reported since the last check.
    fn check(&self) -> Result<(), GpuError> {
        if self.lost.load(Ordering::Relaxed) {
            return Err(GpuError::Gpu("the device was lost".to_owned()));
        }
        let mut errors = self
            .errors
            .lock()
            .map_err(|_| GpuError::Gpu("error log poisoned".to_owned()))?;
        match errors.drain(..).next() {
            Some(error) => Err(GpuError::Gpu(error)),
            None => Ok(()),
        }
    }

    fn wait(&self) -> Result<(), GpuError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| GpuError::Gpu(error.to_string()))?;
        self.check()
    }

    fn storage_buffer(&self, label: &str, bytes: &[u8]) -> Result<wgpu::Buffer, GpuError> {
        let size = (bytes.len() as u64).max(16);
        if size > self.max_binding {
            return Err(GpuError::TooLarge {
                needed: size,
                allowed: self.max_binding,
            });
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.next_multiple_of(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        Ok(buffer)
    }

    fn upload_scene(&self, scene: &Bvh) -> Result<Scene, GpuError> {
        let mut nodes = Vec::with_capacity(scene.nodes().len() * NODE_BYTES as usize);
        for node in scene.nodes() {
            push_vec3(&mut nodes, node.min);
            nodes.extend(node.first.to_le_bytes());
            push_vec3(&mut nodes, node.max);
            nodes.extend(node.count.to_le_bytes());
        }
        // Corners in leaf order, so a leaf's triangles are contiguous.
        let mut triangles = Vec::with_capacity(scene.order().len() * TRIANGLE_BYTES as usize);
        for index in scene.order() {
            let corners = scene
                .triangles()
                .get(*index as usize)
                .ok_or_else(|| GpuError::Gpu("BVH order names a missing triangle".to_owned()))?;
            for corner in corners {
                push_vec3(&mut triangles, *corner);
                triangles.extend(0_f32.to_le_bytes());
            }
        }
        Ok(Scene {
            id: scene.id(),
            nodes: self.storage_buffer("bvh nodes", &nodes)?,
            triangles: self.storage_buffer("bvh triangles", &triangles)?,
        })
    }

    /// `[ambient_occlusion, thickness]` per query, as
    /// [`substancecraft_core::occlusion::trace_query`] gives within
    /// floating-point tolerance.
    pub fn trace_gpu(
        &self,
        scene: &Bvh,
        params: &OcclusionParams,
        queries: &[OcclusionQuery],
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<[f32; 2]>, TraceError> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        // Nothing to hit: every ray is open and reaches its full length.
        if scene.order().is_empty() {
            return Ok(vec![[1.0, 1.0]; queries.len()]);
        }
        let mut cached = self
            .scene
            .lock()
            .map_err(|_| GpuError::Gpu("scene cache poisoned".to_owned()))?;
        if cached.as_ref().is_none_or(|cached| cached.id != scene.id()) {
            *cached = None;
            *cached = Some(self.upload_scene(scene)?);
        }
        let Some(uploaded) = cached.as_ref() else {
            return Err(GpuError::Gpu("scene upload failed".to_owned()).into());
        };
        let mut results = Vec::with_capacity(queries.len());
        for part in queries.chunks(QUERIES_PER_UPLOAD) {
            results.extend(self.trace_part(uploaded, params, part, cancel)?);
        }
        Ok(results)
    }

    fn trace_part(
        &self,
        scene: &Scene,
        params: &OcclusionParams,
        queries: &[OcclusionQuery],
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<[f32; 2]>, TraceError> {
        let mut bytes = Vec::with_capacity(queries.len() * QUERY_BYTES as usize);
        for query in queries {
            push_vec3(&mut bytes, query.point);
            bytes.extend(query.seed.to_le_bytes());
            push_vec3(&mut bytes, query.normal);
            bytes.extend(0_f32.to_le_bytes());
            push_vec3(&mut bytes, query.face_normal);
            bytes.extend(0_f32.to_le_bytes());
        }
        let query_buffer = self.storage_buffer("occlusion queries", &bytes)?;
        let result_size = queries.len() as u64 * RESULT_BYTES;
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion results"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion readback"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("occlusion"),
            layout: &self.layout,
            entries: &[
                entry(0, &scene.nodes),
                entry(1, &scene.triangles),
                entry(2, &query_buffer),
                entry(3, &result_buffer),
                entry(4, &uniform),
            ],
        });

        let rays = u64::from(params.ao_samples + params.thickness_samples).max(1);
        let per_dispatch = (RAYS_PER_DISPATCH / rays).clamp(
            u64::from(WORKGROUP_SIZE),
            u64::from(WORKGROUP_SIZE) * 65_535,
        ) as usize;
        let mut offset = 0;
        while offset < queries.len() {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(TraceError::Cancelled);
            }
            let count = per_dispatch.min(queries.len() - offset);
            let mut uniform_bytes = Vec::with_capacity(32);
            uniform_bytes.extend(params.ao_samples.to_le_bytes());
            uniform_bytes.extend(params.thickness_samples.to_le_bytes());
            uniform_bytes.extend(params.ao_distance.to_le_bytes());
            uniform_bytes.extend(params.thickness_distance.to_le_bytes());
            uniform_bytes.extend(params.bias.to_le_bytes());
            uniform_bytes.extend((offset as u32).to_le_bytes());
            uniform_bytes.extend((count as u32).to_le_bytes());
            uniform_bytes.extend(THICKNESS_SEED.to_le_bytes());
            self.queue.write_buffer(&uniform, 0, &uniform_bytes);
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("occlusion"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("occlusion"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
            }
            self.queue.submit([encoder.finish()]);
            self.wait()?;
            offset += count;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("occlusion readback"),
            });
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback, 0, result_size);
        self.queue.submit([encoder.finish()]);
        let mapped = Arc::new(Mutex::new(None));
        let done = Arc::clone(&mapped);
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            if let Ok(mut done) = done.lock() {
                *done = Some(result);
            }
        });
        self.wait()?;
        match mapped.lock().ok().and_then(|mut mapped| mapped.take()) {
            Some(Ok(())) => {}
            Some(Err(error)) => return Err(GpuError::Gpu(error.to_string()).into()),
            None => return Err(GpuError::Gpu("results were not read back".to_owned()).into()),
        }
        let results = {
            let view = readback
                .get_mapped_range(..)
                .map_err(|error| GpuError::Gpu(error.to_string()))?;
            view.as_chunks::<{ RESULT_BYTES as usize }>()
                .0
                .iter()
                .map(|pair| {
                    let (ao, thickness) = pair.split_at(4);
                    [read_f32(ao), read_f32(thickness)]
                })
                .collect::<Vec<_>>()
        };
        readback.unmap();
        Ok(results)
    }
}

impl OcclusionTracer for GpuTracer {
    fn trace(
        &self,
        scene: &Bvh,
        params: &OcclusionParams,
        queries: &[OcclusionQuery],
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<[f32; 2]>, MeshMapError> {
        self.trace_gpu(scene, params, queries, cancel)
            .map_err(|error| match error {
                TraceError::Cancelled => MeshMapError::Cancelled,
                TraceError::Gpu(error) => MeshMapError::Tracer(error.to_string()),
            })
    }
}

/// Traces on the GPU while it works, and on the CPU from the first GPU
/// failure on (or throughout, without a GPU), recording why.
pub struct FallbackTracer<'a> {
    gpu: Option<&'a GpuTracer>,
    traced_on_gpu: AtomicBool,
    fallback: Mutex<Option<String>>,
}

impl<'a> FallbackTracer<'a> {
    pub fn new(gpu: Option<&'a GpuTracer>) -> Self {
        Self {
            gpu,
            traced_on_gpu: AtomicBool::new(false),
            fallback: Mutex::new(None),
        }
    }

    /// Opens the shared GPU tracer, falling back to the CPU (and saying
    /// why) when there is none.
    pub fn shared() -> FallbackTracer<'static> {
        match GpuTracer::shared() {
            Ok(gpu) => FallbackTracer::new(Some(gpu)),
            Err(error) => {
                let tracer = FallbackTracer::new(None);
                tracer.record(error.to_string());
                tracer
            }
        }
    }

    /// The GPU in use, when there is one.
    pub fn gpu(&self) -> Option<&'a GpuTracer> {
        self.gpu
    }

    /// True when any rays were traced on the GPU.
    pub fn traced_on_gpu(&self) -> bool {
        self.traced_on_gpu.load(Ordering::Relaxed)
    }

    /// Why rays were traced on the CPU, if they were.
    pub fn fallback_reason(&self) -> Option<String> {
        self.fallback.lock().ok().and_then(|reason| reason.clone())
    }

    fn record(&self, reason: String) {
        if let Ok(mut fallback) = self.fallback.lock() {
            fallback.get_or_insert(reason);
        }
    }
}

impl OcclusionTracer for FallbackTracer<'_> {
    fn trace(
        &self,
        scene: &Bvh,
        params: &OcclusionParams,
        queries: &[OcclusionQuery],
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<[f32; 2]>, MeshMapError> {
        if let Some(gpu) = self.gpu.filter(|_| self.fallback_reason().is_none()) {
            match gpu.trace_gpu(scene, params, queries, cancel) {
                Ok(results) => {
                    self.traced_on_gpu.store(true, Ordering::Relaxed);
                    return Ok(results);
                }
                Err(TraceError::Cancelled) => return Err(MeshMapError::Cancelled),
                Err(TraceError::Gpu(error)) => self.record(error.to_string()),
            }
        }
        CpuTracer.trace(scene, params, queries, cancel)
    }
}

fn read_f32(bytes: &[u8]) -> f32 {
    let mut word = [0; 4];
    word.copy_from_slice(bytes);
    f32::from_le_bytes(word)
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn push_vec3(bytes: &mut Vec<u8>, value: glam::Vec3) {
    for component in value.to_array() {
        bytes.extend(component.to_le_bytes());
    }
}
