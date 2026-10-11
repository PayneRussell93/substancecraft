#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

//! Offscreen viewport renderer backed by the vendored Filament fork (`third_party/filament`).
//!
//! The C++ engine is reached through the C ABI in `cpp/shim.cpp`; this module owns the only
//! handle to it and keeps all geometry, camera and texture preparation on the Rust side.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr::NonNull;

use glam::{Mat3, Quat, Vec3};
use substancecraft_core::{MeshGeometry, ViewportCamera};
use thiserror::Error;

const MAX_VIEWPORT_DIMENSION: u32 = 2048;
const MAX_VIEWPORT_PIXELS: usize = 4_194_304;
const MAX_MESH_VERTICES: usize = 3_000_000;
const FLOATS_PER_VERTEX: usize = 9;
const DEFAULT_BASE_COLOR: [u8; 4] = [168, 173, 179, 255];

static VIEWPORT_MATERIAL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/viewport.filamat"));

mod ffi {
    use std::ffi::c_void;

    #[repr(C)]
    pub struct ScRenderer {
        _private: [u8; 0],
    }

    pub type ReadbackFn = unsafe extern "C" fn(user: *mut c_void, pixels: *const u8, len: usize);

    unsafe extern "C" {
        pub fn sc_renderer_create(
            material: *const u8,
            material_len: usize,
            width: u32,
            height: u32,
        ) -> *mut ScRenderer;
        pub fn sc_renderer_destroy(renderer: *mut ScRenderer);
        pub fn sc_renderer_resize(renderer: *mut ScRenderer, width: u32, height: u32) -> bool;
        pub fn sc_renderer_set_base_color_map(
            renderer: *mut ScRenderer,
            width: u32,
            height: u32,
            rgba8: *const u8,
        ) -> bool;
        pub fn sc_renderer_set_mesh(
            renderer: *mut ScRenderer,
            vertices: *const f32,
            vertex_count: u32,
            indices: *const u32,
            index_count: u32,
            center: *const f32,
            half_extent: *const f32,
        ) -> bool;
        pub fn sc_renderer_clear_mesh(renderer: *mut ScRenderer);
        pub fn sc_renderer_set_camera(
            renderer: *mut ScRenderer,
            left: f64,
            right: f64,
            bottom: f64,
            top: f64,
            near_plane: f64,
            far_plane: f64,
            eye: *const f64,
            target: *const f64,
            up: *const f64,
        );
        pub fn sc_renderer_render(
            renderer: *mut ScRenderer,
            callback: ReadbackFn,
            user: *mut c_void,
        ) -> bool;
        pub fn sc_renderer_pump(renderer: *mut ScRenderer);
    }
}

/// Pending readback handed to Filament; reclaimed in [`readback_trampoline`].
struct Readback {
    width: u32,
    height: u32,
    callback: Box<dyn FnOnce(Vec<u8>)>,
}

unsafe extern "C" fn readback_trampoline(user: *mut c_void, pixels: *const u8, len: usize) {
    // SAFETY: `user` is the `Box<Readback>` leaked in `render`, and Filament invokes this
    // callback exactly once per successful `sc_renderer_render`.
    let readback = unsafe { Box::from_raw(user.cast::<Readback>()) };
    let expected = (readback.width as usize) * (readback.height as usize) * 4;
    if pixels.is_null() || len < expected {
        return;
    }
    // SAFETY: Filament guarantees `pixels` points at `len` readable bytes during the callback.
    // The swap chain readback already arrives with rows top to bottom.
    let rgba8 = unsafe { std::slice::from_raw_parts(pixels, expected) }.to_vec();
    let Readback { callback, .. } = *readback;
    // A panic must not unwind into Filament's C++ frames.
    let _ = catch_unwind(AssertUnwindSafe(move || callback(rgba8)));
}

pub struct FilamentRuntime {
    raw: NonNull<ffi::ScRenderer>,
    base_color_map_key: Option<usize>,
    mesh_bounds: Option<(Vec3, Vec3)>,
    target_size: [u32; 2],
}

impl FilamentRuntime {
    pub fn new(width: u32, height: u32) -> Result<Self, FilamentError> {
        let size = bounded_size(width, height)?;
        // SAFETY: the material slice outlives the call; the shim copies what it keeps.
        let raw = unsafe {
            ffi::sc_renderer_create(
                VIEWPORT_MATERIAL.as_ptr(),
                VIEWPORT_MATERIAL.len(),
                size[0],
                size[1],
            )
        };
        let raw = NonNull::new(raw).ok_or(FilamentError::EngineUnavailable)?;
        let mut runtime = Self {
            raw,
            base_color_map_key: None,
            mesh_bounds: None,
            target_size: size,
        };
        runtime.set_base_color_map(0, 1, 1, &DEFAULT_BASE_COLOR)?;
        Ok(runtime)
    }

    /// Delivers completed GPU readbacks to their callbacks. Call once per UI frame, including
    /// frames where no new render is submitted, or pending readbacks never complete. While a
    /// readback is outstanding this waits for the GPU to finish the submitted frame.
    pub fn poll(&mut self) {
        // SAFETY: `raw` is a live renderer owned by `self`.
        unsafe { ffi::sc_renderer_pump(self.raw.as_ptr()) };
    }

    pub fn set_base_color_map(
        &mut self,
        key: usize,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<(), FilamentError> {
        if self.base_color_map_key == Some(key) {
            return Ok(());
        }
        let byte_count = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .filter(|bytes| *bytes <= MAX_VIEWPORT_PIXELS * 4)
            .ok_or(FilamentError::InvalidTexture)?;
        if width == 0 || height == 0 || rgba8.len() != byte_count {
            return Err(FilamentError::InvalidTexture);
        }
        // SAFETY: `rgba8` holds exactly width * height * 4 bytes; the shim copies them.
        let uploaded = unsafe {
            ffi::sc_renderer_set_base_color_map(self.raw.as_ptr(), width, height, rgba8.as_ptr())
        };
        if !uploaded {
            return Err(FilamentError::TextureUnavailable);
        }
        self.base_color_map_key = Some(key);
        Ok(())
    }

    /// Renders one frame and schedules an RGBA8 readback (rows top to bottom). Returns
    /// `Ok(false)` when Filament skipped the frame, in which case `readback` is dropped unused.
    /// Otherwise `readback` runs from a later [`FilamentRuntime::poll`] call.
    pub fn render(
        &mut self,
        camera: ViewportCamera,
        width: u32,
        height: u32,
        pixels_per_point: f32,
        readback: impl FnOnce(Vec<u8>) + 'static,
    ) -> Result<bool, FilamentError> {
        let size = bounded_size(width, height)?;
        if size != self.target_size {
            // SAFETY: `raw` is a live renderer owned by `self`.
            if !unsafe { ffi::sc_renderer_resize(self.raw.as_ptr(), size[0], size[1]) } {
                return Err(FilamentError::SwapChainUnavailable);
            }
            self.target_size = size;
        }
        let Some(bounds) = self.mesh_bounds else {
            return Err(FilamentError::InvalidMesh);
        };
        self.configure_camera(bounds, camera, size, pixels_per_point);

        let user = Box::into_raw(Box::new(Readback {
            width: size[0],
            height: size[1],
            callback: Box::new(readback),
        }));
        // SAFETY: ownership of `user` passes to the shim only when it returns true.
        let rendered = unsafe {
            ffi::sc_renderer_render(self.raw.as_ptr(), readback_trampoline, user.cast())
        };
        if !rendered {
            // SAFETY: the shim did not take ownership, so reclaim the box here.
            drop(unsafe { Box::from_raw(user) });
        }
        Ok(rendered)
    }

    fn configure_camera(
        &mut self,
        (center, extent): (Vec3, Vec3),
        camera: ViewportCamera,
        size: [u32; 2],
        pixels_per_point: f32,
    ) {
        // Mirror the CPU viewport projection: world units map to `scale * 0.9` points.
        let pixels_per_point = pixels_per_point.max(f32::EPSILON);
        let width_points = size[0] as f32 / pixels_per_point;
        let height_points = size[1] as f32 / pixels_per_point;
        let scale = width_points.min(height_points) * 0.42 * camera.zoom;
        let pan_scale = (scale * 0.9).max(f32::EPSILON);
        let half_width = width_points / (2.0 * pan_scale);
        let half_height = height_points / (2.0 * pan_scale);

        let to_world = camera.rotation().inverse();
        let radius = extent.length().max(0.01);
        let distance = radius * 4.0;
        let target =
            center + to_world * Vec3::new(-camera.pan.x / pan_scale, camera.pan.y / pan_scale, 0.0);
        let eye = target + to_world * Vec3::new(0.0, 0.0, distance);
        let up = to_world * Vec3::Y;
        let eye = eye.as_dvec3().to_array();
        let target = target.as_dvec3().to_array();
        let up = up.as_dvec3().to_array();
        // SAFETY: `raw` is live and the vectors are three-element arrays read during the call.
        unsafe {
            ffi::sc_renderer_set_camera(
                self.raw.as_ptr(),
                -f64::from(half_width),
                f64::from(half_width),
                -f64::from(half_height),
                f64::from(half_height),
                f64::from(distance - radius * 2.0).max(0.001),
                f64::from(distance + radius * 2.0),
                eye.as_ptr(),
                target.as_ptr(),
                up.as_ptr(),
            );
        }
    }

    pub fn set_mesh(&mut self, geometry: &MeshGeometry) -> Result<(), FilamentError> {
        if geometry.vertices.is_empty()
            || geometry.triangles.is_empty()
            || geometry.vertices.len() > MAX_MESH_VERTICES
        {
            return Err(FilamentError::InvalidMesh);
        }
        self.mesh_bounds = None;
        let MeshBuffers {
            vertex_data,
            indices,
            bounds,
        } = build_mesh_buffers(geometry)?;
        let vertex_count =
            u32::try_from(geometry.vertices.len()).map_err(|_| FilamentError::InvalidMesh)?;
        let index_count = u32::try_from(indices.len()).map_err(|_| FilamentError::InvalidMesh)?;
        let center = bounds.0.to_array();
        let half_extent = bounds.1.to_array();
        // SAFETY: slices match the declared counts; the shim copies all of them.
        let uploaded = unsafe {
            ffi::sc_renderer_set_mesh(
                self.raw.as_ptr(),
                vertex_data.as_ptr(),
                vertex_count,
                indices.as_ptr(),
                index_count,
                center.as_ptr(),
                half_extent.as_ptr(),
            )
        };
        if !uploaded {
            return Err(FilamentError::MeshBufferUnavailable);
        }
        self.mesh_bounds = Some(bounds);
        Ok(())
    }

    pub fn clear_mesh(&mut self) {
        // SAFETY: `raw` is a live renderer owned by `self`.
        unsafe { ffi::sc_renderer_clear_mesh(self.raw.as_ptr()) };
        self.mesh_bounds = None;
    }
}

impl Drop for FilamentRuntime {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `sc_renderer_create` and is destroyed exactly once here.
        unsafe { ffi::sc_renderer_destroy(self.raw.as_ptr()) };
    }
}

fn bounded_size(width: u32, height: u32) -> Result<[u32; 2], FilamentError> {
    if width == 0 || height == 0 || width > MAX_VIEWPORT_DIMENSION || height > MAX_VIEWPORT_DIMENSION
    {
        return Err(FilamentError::InvalidViewport);
    }
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(FilamentError::InvalidViewport)?;
    if pixel_count > MAX_VIEWPORT_PIXELS {
        return Err(FilamentError::InvalidViewport);
    }
    Ok([width, height])
}

fn bounds_center_extent(geometry: &MeshGeometry) -> (Vec3, Vec3) {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for vertex in &geometry.vertices {
        minimum = minimum.min(vertex.position);
        maximum = maximum.max(vertex.position);
    }
    ((minimum + maximum) * 0.5, (maximum - minimum) * 0.5)
}

struct MeshBuffers {
    vertex_data: Vec<f32>,
    indices: Vec<u32>,
    bounds: (Vec3, Vec3),
}

fn build_mesh_buffers(geometry: &MeshGeometry) -> Result<MeshBuffers, FilamentError> {
    let vertex_count = geometry.vertices.len();
    if geometry
        .triangles
        .iter()
        .flatten()
        .any(|index| *index as usize >= vertex_count)
    {
        return Err(FilamentError::InvalidMesh);
    }
    let mut normals = vec![Vec3::ZERO; vertex_count];
    let mut tangents = vec![Vec3::ZERO; vertex_count];
    let mut bitangents = vec![Vec3::ZERO; vertex_count];
    for triangle in &geometry.triangles {
        let [index_a, index_b, index_c] = *triangle;
        let vertex_a = geometry.vertices[index_a as usize];
        let vertex_b = geometry.vertices[index_b as usize];
        let vertex_c = geometry.vertices[index_c as usize];
        let edge_ab = vertex_b.position - vertex_a.position;
        let edge_ac = vertex_c.position - vertex_a.position;
        let normal = edge_ab.cross(edge_ac);
        for index in [index_a, index_b, index_c] {
            normals[index as usize] += normal;
        }
        let uv_ab = vertex_b.uv - vertex_a.uv;
        let uv_ac = vertex_c.uv - vertex_a.uv;
        let determinant = uv_ab.x * uv_ac.y - uv_ab.y * uv_ac.x;
        if determinant.abs() > f32::EPSILON {
            let inverse = determinant.recip();
            let tangent = (edge_ab * uv_ac.y - edge_ac * uv_ab.y) * inverse;
            let bitangent = (edge_ac * uv_ab.x - edge_ab * uv_ac.x) * inverse;
            for index in [index_a, index_b, index_c] {
                tangents[index as usize] += tangent;
                bitangents[index as usize] += bitangent;
            }
        }
    }

    let mut vertex_data = Vec::with_capacity(vertex_count.saturating_mul(FLOATS_PER_VERTEX));
    for (index, vertex) in geometry.vertices.iter().enumerate() {
        let mut normal = normals[index].normalize_or_zero();
        if normal == Vec3::ZERO {
            normal = Vec3::Z;
        }
        let mut tangent = tangents[index] - normal * normal.dot(tangents[index]);
        if tangent.length_squared() <= f32::EPSILON {
            let helper = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            tangent = normal.cross(helper);
        }
        tangent = tangent.normalize_or_zero();
        let orientation = if normal.cross(tangent).dot(bitangents[index]) < 0.0 {
            -1.0
        } else {
            1.0
        };
        let bitangent = normal.cross(tangent);
        let mut tangent_frame =
            Quat::from_mat3(&Mat3::from_cols(tangent, bitangent, normal)).normalize();
        // Filament encodes bitangent handedness in the sign of the quaternion's w component.
        if tangent_frame.w < 0.0 {
            tangent_frame = -tangent_frame;
        }
        tangent_frame.w = tangent_frame.w.max(1e-5);
        if orientation < 0.0 {
            tangent_frame = -tangent_frame;
        }
        vertex_data.extend_from_slice(&[
            vertex.position.x,
            vertex.position.y,
            vertex.position.z,
            vertex.uv.x,
            vertex.uv.y,
            tangent_frame.x,
            tangent_frame.y,
            tangent_frame.z,
            tangent_frame.w,
        ]);
    }
    let indices = geometry
        .triangles
        .iter()
        .flat_map(|triangle| triangle.iter().copied())
        .collect();
    Ok(MeshBuffers {
        vertex_data,
        indices,
        bounds: bounds_center_extent(geometry),
    })
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FilamentError {
    #[error("Filament could not initialize its OpenGL backend")]
    EngineUnavailable,
    #[error("Filament could not create the offscreen swap chain")]
    SwapChainUnavailable,
    #[error("mesh geometry or its buffers are invalid")]
    InvalidMesh,
    #[error("viewport dimensions exceed the Filament render budget")]
    InvalidViewport,
    #[error("Filament could not create mesh buffers")]
    MeshBufferUnavailable,
    #[error("Filament could not create a Base Color texture")]
    TextureUnavailable,
    #[error("Base Color texture data has invalid dimensions or byte length")]
    InvalidTexture,
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::sync_channel;
    use std::time::{Duration, Instant};

    use substancecraft_core::{PrimitivePreset, ViewportCamera};

    use super::{FLOATS_PER_VERTEX, FilamentRuntime, build_mesh_buffers};

    #[test]
    fn mesh_buffer_generation_preserves_topology_and_emits_finite_tangent_frames() {
        let geometry = PrimitivePreset::Cube.build_mesh();
        let buffers = build_mesh_buffers(&geometry).expect("valid cube buffers");

        assert_eq!(buffers.vertex_data.len(), geometry.vertices.len() * FLOATS_PER_VERTEX);
        assert_eq!(buffers.indices.len(), geometry.triangles.len() * 3);
        assert_eq!(buffers.bounds.0.to_array(), [0.0; 3]);
        assert_eq!(buffers.bounds.1.to_array(), [0.5; 3]);
        assert!(buffers.vertex_data.iter().all(|value| value.is_finite()));
        assert_eq!(buffers.indices[0..3], geometry.triangles[0]);
    }

    #[test]
    #[ignore = "requires an OpenGL driver for Filament's headless swap chain"]
    fn headless_filament_render_returns_an_rgba_frame() {
        let mut renderer = FilamentRuntime::new(64, 64).expect("Filament runtime");
        let geometry = PrimitivePreset::Cube.build_mesh();
        renderer.set_mesh(&geometry).expect("cube mesh upload");
        let (sender, receiver) = sync_channel(1);
        let rendered = renderer
            .render(ViewportCamera::default(), 64, 64, 1.0, move |pixels| {
                let _ = sender.send(pixels);
            })
            .expect("Filament frame submission");
        assert!(rendered);

        let deadline = Instant::now() + Duration::from_secs(5);
        let pixels = loop {
            renderer.poll();
            if let Ok(pixels) = receiver.try_recv() {
                break pixels;
            }
            assert!(Instant::now() < deadline, "Filament readback timed out");
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(pixels.len(), 64 * 64 * 4);
        let center = (32 * 64 + 32) * 4;
        let background = &pixels[0..3];
        assert_ne!(&pixels[center..center + 3], background, "cube should cover the center");
    }
}
