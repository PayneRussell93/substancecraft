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
use substancecraft_core::{MeshGeometry, ViewportCamera, vertex_normals, vertex_tangents};
use thiserror::Error;

const MAX_VIEWPORT_DIMENSION: u32 = 2048;
const MAX_VIEWPORT_PIXELS: usize = 4_194_304;
const MAX_MESH_VERTICES: usize = 3_000_000;
const FLOATS_PER_VERTEX: usize = 9;

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
        pub fn sc_renderer_set_map(
            renderer: *mut ScRenderer,
            primitive: u32,
            slot: u32,
            width: u32,
            height: u32,
            rgba8: *const u8,
        ) -> bool;
        pub fn sc_renderer_set_preview_mode(renderer: *mut ScRenderer, preview: bool);
        pub fn sc_renderer_set_mesh(
            renderer: *mut ScRenderer,
            vertices: *const f32,
            vertex_count: u32,
            indices: *const u32,
            index_count: u32,
            primitive_offsets: *const u32,
            primitive_counts: *const u32,
            primitive_count: u32,
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

/// Material texture slots of the viewport material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialMap {
    BaseColor = 0,
    Orm = 1,
    Normal = 2,
    Emissive = 3,
}

const MAP_COUNT: usize = 4;
const MAX_PRIMITIVES: usize = 256;

pub struct FilamentRuntime {
    raw: NonNull<ffi::ScRenderer>,
    /// Last uploaded content key per primitive and map.
    map_keys: Vec<[Option<usize>; MAP_COUNT]>,
    /// Primitive drawing each mesh material slot; `None` for slots without
    /// triangles.
    slot_primitives: Vec<Option<u32>>,
    preview_mode: Option<bool>,
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
        let runtime = Self {
            raw,
            map_keys: vec![[None; MAP_COUNT]],
            slot_primitives: vec![Some(0)],
            preview_mode: None,
            mesh_bounds: None,
            target_size: size,
        };
        Ok(runtime)
    }

    /// Delivers completed GPU readbacks to their callbacks. Call once per UI frame, including
    /// frames where no new render is submitted, or pending readbacks never complete. While a
    /// readback is outstanding this waits for the GPU to finish the submitted frame.
    pub fn poll(&mut self) {
        // SAFETY: `raw` is a live renderer owned by `self`.
        unsafe { ffi::sc_renderer_pump(self.raw.as_ptr()) };
    }

    /// Number of mesh material slots the current mesh draws.
    pub fn material_slot_count(&self) -> usize {
        self.slot_primitives.len()
    }

    /// Uploads one map of mesh material slot `slot`. `key` identifies the
    /// content; re-uploading the same key is skipped, and slots without
    /// triangles ignore uploads. Base Color (Opacity in alpha) and Emissive are
    /// sRGB; ORM (R occlusion, G roughness, B metallic) and tangent-space
    /// normals (+Y = +V) are linear.
    pub fn set_map(
        &mut self,
        slot: usize,
        map: MaterialMap,
        key: usize,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<(), FilamentError> {
        let Some(primitive) = self.slot_primitives.get(slot).copied().flatten() else {
            return Ok(());
        };
        let map_index = map as usize;
        let Some(keys) = self.map_keys.get_mut(primitive as usize) else {
            return Ok(());
        };
        if keys[map_index] == Some(key) {
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
            ffi::sc_renderer_set_map(
                self.raw.as_ptr(),
                primitive,
                map_index as u32,
                width,
                height,
                rgba8.as_ptr(),
            )
        };
        if !uploaded {
            return Err(FilamentError::TextureUnavailable);
        }
        if let Some(keys) = self.map_keys.get_mut(primitive as usize) {
            keys[map_index] = Some(key);
        }
        Ok(())
    }

    pub fn set_base_color_map(
        &mut self,
        key: usize,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<(), FilamentError> {
        self.set_map(0, MaterialMap::BaseColor, key, width, height, rgba8)
    }

    /// Preview mode shows only the base colour map on a neutral dielectric, for
    /// inspecting single channels or mesh maps.
    pub fn set_preview_mode(&mut self, preview: bool) {
        if self.preview_mode == Some(preview) {
            return;
        }
        // SAFETY: `raw` is a live renderer owned by `self`.
        unsafe { ffi::sc_renderer_set_preview_mode(self.raw.as_ptr(), preview) };
        self.preview_mode = Some(preview);
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
        let rendered =
            unsafe { ffi::sc_renderer_render(self.raw.as_ptr(), readback_trampoline, user.cast()) };
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

    /// Uploads `geometry`. Shading normals are smoothed across faces meeting at
    /// less than `smoothing_angle_degrees` (0 = flat, 180 = fully smooth).
    pub fn set_mesh(
        &mut self,
        geometry: &MeshGeometry,
        smoothing_angle_degrees: f32,
    ) -> Result<(), FilamentError> {
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
            primitives,
            slot_primitives,
            bounds,
        } = build_mesh_buffers(geometry, smoothing_angle_degrees)?;
        let offsets = primitives.iter().map(|range| range.0).collect::<Vec<_>>();
        let counts = primitives.iter().map(|range| range.1).collect::<Vec<_>>();
        let primitive_count =
            u32::try_from(primitives.len()).map_err(|_| FilamentError::InvalidMesh)?;
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
                offsets.as_ptr(),
                counts.as_ptr(),
                primitive_count,
                center.as_ptr(),
                half_extent.as_ptr(),
            )
        };
        if !uploaded {
            return Err(FilamentError::MeshBufferUnavailable);
        }
        // Fresh instances start with neutral maps.
        self.map_keys = vec![[None; MAP_COUNT]; primitives.len()];
        self.slot_primitives = slot_primitives;
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
    if width == 0
        || height == 0
        || width > MAX_VIEWPORT_DIMENSION
        || height > MAX_VIEWPORT_DIMENSION
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
    /// Triangle indices grouped by material slot.
    indices: Vec<u32>,
    /// `(first index, index count)` of each non-empty slot, in slot order.
    primitives: Vec<(u32, u32)>,
    /// Primitive drawing each slot, or `None` when the slot has no triangles.
    slot_primitives: Vec<Option<u32>>,
    bounds: (Vec3, Vec3),
}

fn build_mesh_buffers(
    geometry: &MeshGeometry,
    smoothing_angle_degrees: f32,
) -> Result<MeshBuffers, FilamentError> {
    let vertex_count = geometry.vertices.len();
    if geometry
        .triangles
        .iter()
        .flatten()
        .any(|index| *index as usize >= vertex_count)
    {
        return Err(FilamentError::InvalidMesh);
    }
    let normals = vertex_normals(geometry, smoothing_angle_degrees);
    // Shared with the normal-map baker so baked maps decode in the same frame.
    let tangents = vertex_tangents(geometry, &normals);
    let mut vertex_data = Vec::with_capacity(vertex_count.saturating_mul(FLOATS_PER_VERTEX));
    for (index, vertex) in geometry.vertices.iter().enumerate() {
        let normal = normals[index].normalize_or(Vec3::Z);
        let tangent = tangents[index].truncate();
        let orientation = tangents[index].w;
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
    let slot_count = geometry.material_slot_count();
    if slot_count > MAX_PRIMITIVES {
        return Err(FilamentError::InvalidMesh);
    }
    let mut indices = Vec::with_capacity(geometry.triangles.len() * 3);
    let mut primitives = Vec::new();
    let mut slot_primitives = Vec::with_capacity(slot_count);
    for slot in 0..slot_count as u32 {
        let first = indices.len();
        for (triangle_index, triangle) in geometry.triangles.iter().enumerate() {
            if geometry.material_of(triangle_index) == slot {
                indices.extend_from_slice(triangle);
            }
        }
        let count = indices.len() - first;
        if count == 0 {
            slot_primitives.push(None);
            continue;
        }
        let first = u32::try_from(first).map_err(|_| FilamentError::InvalidMesh)?;
        let count = u32::try_from(count).map_err(|_| FilamentError::InvalidMesh)?;
        slot_primitives.push(Some(primitives.len() as u32));
        primitives.push((first, count));
    }
    Ok(MeshBuffers {
        vertex_data,
        indices,
        primitives,
        slot_primitives,
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
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::mpsc::sync_channel;
    use std::time::{Duration, Instant};

    use substancecraft_core::{PrimitivePreset, ViewportCamera};

    use super::{FLOATS_PER_VERTEX, FilamentRuntime, MaterialMap, build_mesh_buffers};

    #[test]
    fn mesh_buffer_generation_preserves_topology_and_emits_finite_tangent_frames() {
        let geometry = PrimitivePreset::Cube.build_mesh();
        let buffers = build_mesh_buffers(&geometry, 60.0).expect("valid cube buffers");

        assert_eq!(
            buffers.vertex_data.len(),
            geometry.vertices.len() * FLOATS_PER_VERTEX
        );
        assert_eq!(buffers.indices.len(), geometry.triangles.len() * 3);
        assert_eq!(buffers.bounds.0.to_array(), [0.0; 3]);
        assert_eq!(buffers.bounds.1.to_array(), [0.5; 3]);
        assert!(buffers.vertex_data.iter().all(|value| value.is_finite()));
        assert_eq!(buffers.indices[0..3], geometry.triangles[0]);
    }

    #[test]
    fn tc_0093_mesh_buffers_group_triangles_into_one_primitive_per_material_slot() {
        let mut geometry = PrimitivePreset::Cube.build_mesh();
        // Alternate the 12 cube triangles between slots 0 and 2; slot 1 is empty.
        geometry.material_slots = vec!["A".to_owned(), "B".to_owned(), "C".to_owned()];
        geometry.triangle_materials = (0..12).map(|index| (index % 2) * 2).collect();
        let buffers = build_mesh_buffers(&geometry, 60.0).expect("valid buffers");

        assert_eq!(buffers.primitives, vec![(0, 18), (18, 18)]);
        assert_eq!(buffers.slot_primitives, vec![Some(0), None, Some(1)]);
        assert_eq!(buffers.indices[0..3], geometry.triangles[0]);
        assert_eq!(buffers.indices[18..21], geometry.triangles[1]);
    }

    #[test]
    #[ignore = "requires an OpenGL driver for Filament's headless swap chain"]
    fn headless_filament_render_returns_an_rgba_frame() {
        let mut renderer = FilamentRuntime::new(64, 64).expect("Filament runtime");
        let geometry = PrimitivePreset::Cube.build_mesh();
        renderer
            .set_mesh(&geometry, 60.0)
            .expect("cube mesh upload");
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
        assert_ne!(
            &pixels[center..center + 3],
            background,
            "cube should cover the center"
        );
    }

    fn render_frame(renderer: &mut FilamentRuntime) -> Vec<u8> {
        let (sender, receiver) = sync_channel(1);
        // The first frames after a texture change may still show old data.
        for _ in 0..3 {
            let sender = sender.clone();
            let rendered = renderer
                .render(ViewportCamera::default(), 64, 64, 1.0, move |pixels| {
                    let _ = sender.try_send(pixels);
                })
                .expect("Filament frame submission");
            assert!(rendered);
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                renderer.poll();
                if receiver.try_recv().is_ok() {
                    break;
                }
                assert!(Instant::now() < deadline, "Filament readback timed out");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        let rendered = renderer
            .render(ViewportCamera::default(), 64, 64, 1.0, move |pixels| {
                let _ = sender.try_send(pixels);
            })
            .expect("Filament frame submission");
        assert!(rendered);
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            renderer.poll();
            if let Ok(pixels) = receiver.try_recv() {
                return pixels;
            }
            assert!(Instant::now() < deadline, "Filament readback timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn mean_difference(left: &[u8], right: &[u8]) -> f32 {
        let total = left
            .iter()
            .zip(right)
            .map(|(a, b)| f32::from(a.abs_diff(*b)))
            .sum::<f32>();
        total / left.len() as f32
    }

    #[test]
    #[ignore = "requires an OpenGL driver for Filament's headless swap chain"]
    fn tc_0053_orm_and_normal_maps_shade_in_material_mode_but_not_in_preview() {
        let mut renderer = FilamentRuntime::new(64, 64).expect("Filament runtime");
        renderer
            .set_mesh(&PrimitivePreset::Cube.build_mesh(), 60.0)
            .expect("cube mesh upload");
        renderer.set_preview_mode(false);
        let baseline = render_frame(&mut renderer);

        // Fully metallic, mirror-smooth: a dielectric grey turns very different.
        renderer
            .set_map(0, MaterialMap::Orm, 1, 1, 1, &[255, 10, 255, 255])
            .expect("ORM upload");
        let metal = render_frame(&mut renderer);
        assert!(
            mean_difference(&baseline, &metal) > 1.0,
            "ORM map must change shading"
        );

        // A strongly tilted tangent-space normal changes lighting.
        renderer
            .set_map(0, MaterialMap::Orm, 2, 1, 1, &[255, 148, 0, 255])
            .expect("ORM reset");
        renderer
            .set_map(0, MaterialMap::Normal, 3, 1, 1, &[240, 128, 140, 255])
            .expect("normal upload");
        let tilted = render_frame(&mut renderer);
        assert!(
            mean_difference(&baseline, &tilted) > 1.0,
            "normal map must change shading"
        );

        // Preview mode ignores ORM and normal maps.
        renderer
            .set_map(0, MaterialMap::Orm, 4, 1, 1, &[255, 10, 255, 255])
            .expect("ORM upload");
        renderer.set_preview_mode(true);
        let preview = render_frame(&mut renderer);
        assert!(
            mean_difference(&baseline, &preview) < 0.5,
            "preview mode shows a neutral dielectric"
        );
    }

    #[test]
    #[ignore = "requires an OpenGL driver for Filament's headless swap chain"]
    fn tc_0094_each_material_slot_shows_its_own_maps_with_emission_and_cutout() {
        let mut renderer = FilamentRuntime::new(64, 64).expect("Filament runtime");
        let mut geometry = PrimitivePreset::Cube.build_mesh();
        // Slot 1 owns the +Z face (triangles 0 and 1), which faces the default camera
        // only partially; slot 0 owns the rest.
        geometry.material_slots = vec!["Body".to_owned(), "Panel".to_owned()];
        geometry.triangle_materials = (0..12).map(|index| u32::from(index < 2)).collect();
        renderer.set_mesh(&geometry, 60.0).expect("two-slot cube");
        assert_eq!(renderer.material_slot_count(), 2);
        renderer.set_preview_mode(false);
        let baseline = render_frame(&mut renderer);

        renderer
            .set_map(1, MaterialMap::Emissive, 10, 1, 1, &[255, 0, 0, 255])
            .expect("emissive upload");
        let glowing = render_frame(&mut renderer);
        let red_gain = glowing
            .chunks(4)
            .zip(baseline.chunks(4))
            .filter(|(glow, base)| glow[0] > base[0].saturating_add(40))
            .count();
        assert!(
            red_gain > 20,
            "the panel slot glows red ({red_gain} pixels)"
        );

        // Opacity 0 on the body cuts it out, revealing background.
        let background = baseline[0..3].to_vec();
        renderer
            .set_map(0, MaterialMap::BaseColor, 11, 1, 1, &[200, 200, 200, 0])
            .expect("transparent base colour");
        let cut = render_frame(&mut renderer);
        let background_pixels = |frame: &[u8]| {
            frame
                .chunks(4)
                .filter(|pixel| {
                    pixel[0..3]
                        .iter()
                        .zip(&background)
                        .all(|(value, expected)| value.abs_diff(*expected) <= 4)
                })
                .count()
        };
        assert!(
            background_pixels(&cut) > background_pixels(&glowing) + 100,
            "cut-out body shows the background"
        );
    }
}
