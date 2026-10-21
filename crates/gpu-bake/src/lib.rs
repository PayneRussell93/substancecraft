//! GPU ray tracing for mesh-map bakes (PERF-01): a [`RayTracer`] that runs
//! a bake's cage rays, ambient occlusion, and thickness as wgpu compute
//! shaders over the BVH the CPU bake builds. Bakes fall back to the CPU for
//! any batch this tracer fails on, so it never has to be right about the
//! hardware it finds.

use std::sync::OnceLock;
use std::sync::mpsc::channel;

use substancecraft_core::{
    CageHit, CageRay, Occlusion, OcclusionParams, OcclusionQuery, PreparedScene, RayTracer,
    TraceScene,
};

const SHADER: &str = include_str!("trace.wgsl");
const WORKGROUP: u32 = 64;
/// Queries per dispatch, so no single submission runs long enough for the
/// OS to reset the GPU.
const DISPATCH_QUERIES: usize = 8192;

/// Which adapters [`GpuTracer::new`] may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterChoice {
    /// A real GPU only; software rasterizers are slower than the CPU bake.
    Hardware,
    /// Any adapter, software included (tests and headless machines).
    Any,
}

/// Traversal stack sizes compiled, smallest first. A small stack keeps
/// the shader fast; a scene whose BVH needs more than the largest is traced
/// on the CPU instead, so no node the CPU would visit is ever skipped.
const STACK_SIZES: [u32; 2] = [32, 64];

/// The shader's entry points, compiled for one stack size.
struct Pipelines {
    occlusion: wgpu::ComputePipeline,
    cage_hits: wgpu::ComputePipeline,
}

/// A GPU device ready to trace mesh-map bakes.
pub struct GpuTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<Pipelines>,
    name: String,
    max_binding: u64,
}

impl GpuTracer {
    /// Opens the highest-performance adapter `choice` allows.
    pub fn new(choice: AdapterChoice) -> Result<Self, String> {
        pollster::block_on(Self::open(choice))
    }

    /// One hardware tracer for the whole process, opened on first use;
    /// `None` when there is no usable GPU.
    pub fn shared() -> Option<&'static Self> {
        static SHARED: OnceLock<Option<GpuTracer>> = OnceLock::new();
        SHARED
            .get_or_init(|| Self::new(AdapterChoice::Hardware).ok())
            .as_ref()
    }

    async fn open(choice: AdapterChoice) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let mut adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|error| error.to_string());
        if adapter.is_err() && choice == AdapterChoice::Any {
            adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    force_fallback_adapter: true,
                    ..Default::default()
                })
                .await
                .map_err(|error| error.to_string());
        }
        let adapter = adapter?;
        let info = adapter.get_info();
        if choice == AdapterChoice::Hardware && info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("{} is a software adapter", info.name));
        }
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mesh-map tracer"),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|error| error.to_string())?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelines = STACK_SIZES
            .iter()
            .map(|size| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("mesh-map tracer"),
                    source: wgpu::ShaderSource::Wgsl(
                        format!(
                            "const STACK_SIZE: u32 = {size}u;
{SHADER}"
                        )
                        .into(),
                    ),
                });
                let pipeline = |entry_point: &str| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry_point),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry_point),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                };
                Pipelines {
                    occlusion: pipeline("occlusion"),
                    cage_hits: pipeline("cage_hits"),
                }
            })
            .collect();
        if let Some(error) = scope.pop().await {
            return Err(error.to_string());
        }
        Ok(Self {
            device,
            queue,
            pipelines,
            name: format!("{} ({:?})", info.name, info.backend),
            max_binding: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
        })
    }

    fn storage(&self, label: &str, bytes: &[u8]) -> Result<wgpu::Buffer, String> {
        if bytes.len() as u64 > self.max_binding {
            return Err(format!(
                "{label} needs {} bytes, over the GPU's limit",
                bytes.len()
            ));
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (bytes.len() as u64).max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        Ok(buffer)
    }

    /// Runs `pipeline` over `count` items in dispatches of
    /// [`DISPATCH_QUERIES`], with `input` at `input_binding`, and reads back
    /// `stride` bytes per item from `output_binding`.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        scene: &GpuScene<'_>,
        pipeline: &wgpu::ComputePipeline,
        params: [u32; 6],
        input: &[u8],
        input_binding: u32,
        output_binding: u32,
        count: usize,
        stride: u64,
    ) -> Result<Vec<u8>, String> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let input = self.storage("queries", input)?;
        let size = count as u64 * stride;
        if size > self.max_binding {
            return Err("results are over the GPU's limit".to_owned());
        }
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace"),
            layout: &layout,
            entries: &[
                entry(0, &scene.nodes),
                entry(1, &scene.triangles),
                entry(2, &uniform),
                entry(input_binding, &input),
                entry(output_binding, &output),
            ],
        });
        for base in (0..count).step_by(DISPATCH_QUERIES) {
            let batch = (count - base).min(DISPATCH_QUERIES) as u32;
            let mut words = [0_u32; 8];
            words[..6].copy_from_slice(&params);
            words[6] = base as u32;
            words[7] = batch;
            self.queue.write_buffer(&uniform, 0, &words_bytes(&words));
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(batch.div_ceil(WORKGROUP), 1, 1);
            }
            if base + DISPATCH_QUERIES >= count {
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
            }
            // Each dispatch is its own submission, so none runs long enough
            // for the OS to reset the GPU; they run in order without waits.
            self.queue.submit([encoder.finish()]);
        }
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
            .map_err(|error| error.to_string())?;
        receiver
            .recv()
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(error.to_string());
        }
        let bytes = slice
            .get_mapped_range()
            .map_err(|error| error.to_string())?
            .to_vec();
        readback.unmap();
        Ok(bytes)
    }
}

/// Levels below the root of the deepest leaf; an error for a malformed
/// tree (a child index out of range or a cycle).
fn bvh_depth(nodes: &[substancecraft_core::BvhNode]) -> Result<usize, String> {
    let mut deepest = 0;
    let mut pending = vec![(0_usize, 0_usize)];
    let mut visited = 0_usize;
    while let Some((index, depth)) = pending.pop() {
        visited += 1;
        let node = nodes
            .get(index)
            .filter(|_| visited <= nodes.len())
            .ok_or("malformed BVH")?;
        deepest = deepest.max(depth);
        if node.count == 0 {
            let left = node.first as usize;
            pending.push((left, depth + 1));
            pending.push((left + 1, depth + 1));
        }
    }
    Ok(deepest)
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn words_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn floats(out: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn read_f32(bytes: &[u8], at: usize) -> f32 {
    f32::from_bits(read_u32(bytes, at))
}

impl RayTracer for GpuTracer {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn prepare(&self, scene: &TraceScene<'_>) -> Result<Box<dyn PreparedScene + '_>, String> {
        if scene.nodes.is_empty() || scene.order.is_empty() {
            return Err("nothing to trace".to_owned());
        }
        // A traversal holds at most one node per level plus the root.
        let needed = bvh_depth(scene.nodes)? + 1;
        let pipelines = STACK_SIZES
            .iter()
            .position(|size| *size as usize >= needed)
            .and_then(|index| self.pipelines.get(index))
            .ok_or_else(|| format!("the BVH is {needed} levels deep, too deep for the GPU"))?;
        let mut nodes = Vec::with_capacity(scene.nodes.len() * 32);
        for node in scene.nodes {
            floats(&mut nodes, &node.min.to_array());
            nodes.extend_from_slice(&node.first.to_le_bytes());
            floats(&mut nodes, &node.max.to_array());
            nodes.extend_from_slice(&node.count.to_le_bytes());
        }
        // Leaves index triangles in `order`; storing them in that order
        // spares the GPU the indirection.
        let mut triangles = Vec::with_capacity(scene.order.len() * 48);
        for index in scene.order {
            let corners = scene
                .triangles
                .get(*index as usize)
                .ok_or("BVH names a missing triangle")?;
            for (corner, position) in corners.iter().enumerate() {
                floats(&mut triangles, &position.to_array());
                let w = if corner == 0 { *index } else { 0 };
                triangles.extend_from_slice(&w.to_le_bytes());
            }
        }
        Ok(Box::new(GpuScene {
            tracer: self,
            pipelines,
            nodes: self.storage("bvh nodes", &nodes)?,
            triangles: self.storage("triangles", &triangles)?,
        }))
    }
}

/// A bake's BVH and triangles on the GPU.
struct GpuScene<'a> {
    tracer: &'a GpuTracer,
    pipelines: &'a Pipelines,
    nodes: wgpu::Buffer,
    triangles: wgpu::Buffer,
}

impl PreparedScene for GpuScene<'_> {
    fn cage_hits(&mut self, rays: &[CageRay]) -> Result<Vec<Option<CageHit>>, String> {
        let mut input = Vec::with_capacity(rays.len() * 32);
        for ray in rays {
            floats(&mut input, &ray.origin.to_array());
            floats(&mut input, &[ray.t_max]);
            floats(&mut input, &ray.direction.to_array());
            input.extend_from_slice(&0_u32.to_le_bytes());
        }
        let tracer = self.tracer;
        let bytes = tracer.run(
            self,
            &self.pipelines.cage_hits,
            [0; 6],
            &input,
            5,
            6,
            rays.len(),
            16,
        )?;
        Ok((0..rays.len())
            .map(|index| {
                let at = index * 16;
                let triangle = read_u32(&bytes, at);
                (triangle != u32::MAX).then(|| {
                    let (u, v) = (read_f32(&bytes, at + 4), read_f32(&bytes, at + 8));
                    CageHit {
                        triangle,
                        weights: glam::Vec3::new(1.0 - u - v, u, v),
                    }
                })
            })
            .collect())
    }

    fn occlusion(
        &mut self,
        queries: &[OcclusionQuery],
        params: &OcclusionParams,
    ) -> Result<Vec<Occlusion>, String> {
        let mut input = Vec::with_capacity(queries.len() * 48);
        for query in queries {
            floats(&mut input, &query.point.to_array());
            input.extend_from_slice(&query.texel.to_le_bytes());
            floats(&mut input, &query.normal.to_array());
            input.extend_from_slice(&0_u32.to_le_bytes());
            floats(&mut input, &query.face_normal.to_array());
            input.extend_from_slice(&0_u32.to_le_bytes());
        }
        let words = [
            params.seed,
            params.ao_samples,
            params.ao_distance.to_bits(),
            params.thickness_samples,
            params.thickness_distance.to_bits(),
            params.bias.to_bits(),
        ];
        let tracer = self.tracer;
        let bytes = tracer.run(
            self,
            &self.pipelines.occlusion,
            words,
            &input,
            3,
            4,
            queries.len(),
            8,
        )?;
        Ok((0..queries.len())
            .map(|index| Occlusion {
                ambient_occlusion: read_f32(&bytes, index * 8),
                thickness: read_f32(&bytes, index * 8 + 4),
            })
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use glam::Vec3;
    use substancecraft_core::BvhNode;

    fn node(first: u32, count: u32) -> BvhNode {
        BvhNode {
            min: Vec3::ZERO,
            max: Vec3::ONE,
            first,
            count,
        }
    }

    /// A chain `levels` deep: each interior node's right child is a leaf.
    fn chain(levels: u32) -> Vec<BvhNode> {
        let mut nodes = vec![node(1, 0)];
        for _ in 1..levels {
            let first = nodes.len() as u32 + 2;
            nodes.push(node(first, 0));
            nodes.push(node(0, 1));
        }
        nodes.push(node(0, 1));
        nodes.push(node(0, 1));
        nodes
    }

    #[test]
    fn bvh_depth_picks_a_stack_or_refuses() {
        assert_eq!(bvh_depth(&[node(0, 1)]), Ok(0));
        assert_eq!(bvh_depth(&chain(5)), Ok(5));
        let needed = bvh_depth(&chain(70)).unwrap() + 1;
        assert!(STACK_SIZES.iter().all(|size| (*size as usize) < needed));
        // A child out of range, or a cycle back to the root.
        assert!(bvh_depth(&[node(1, 0)]).is_err());
        assert!(bvh_depth(&[node(0, 0), node(0, 1)]).is_err());
    }
}
