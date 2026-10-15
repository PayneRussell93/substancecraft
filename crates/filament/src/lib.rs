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
use substancecraft_core::{
    CubeLut, EnvironmentImage, MeshGeometry, SunLight, TexelRect, ViewTransform, ViewportCamera,
    vertex_normals, vertex_tangents,
};
use thiserror::Error;

const MAX_VIEWPORT_DIMENSION: u32 = 2048;
const MAX_VIEWPORT_PIXELS: usize = 4_194_304;
/// Largest material map the viewport takes (4096 x 4096); bigger texture
/// sets preview downscaled.
const MAX_MAP_PIXELS: usize = 16_777_216;
const MAX_MESH_VERTICES: usize = 3_000_000;
const FLOATS_PER_VERTEX: usize = 9;

static VIEWPORT_MATERIAL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/viewport.filamat"));
static BACKGROUND_MATERIAL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/background.filamat"));

mod mips;

use mips::MipChain;

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
            background: *const u8,
            background_len: usize,
            width: u32,
            height: u32,
        ) -> *mut ScRenderer;
        pub fn sc_renderer_destroy(renderer: *mut ScRenderer);
        pub fn sc_renderer_resize(renderer: *mut ScRenderer, width: u32, height: u32) -> bool;
        pub fn sc_renderer_create_map(
            renderer: *mut ScRenderer,
            primitive: u32,
            slot: u32,
            width: u32,
            height: u32,
            levels: u32,
        ) -> bool;
        #[allow(clippy::too_many_arguments)]
        pub fn sc_renderer_set_map_region(
            renderer: *mut ScRenderer,
            primitive: u32,
            slot: u32,
            level: u32,
            x: u32,
            y: u32,
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
        pub fn sc_renderer_set_vertices(
            renderer: *mut ScRenderer,
            vertices: *const f32,
            vertex_count: u32,
        ) -> bool;
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
        #[allow(clippy::too_many_arguments)]
        pub fn sc_renderer_set_environment(
            renderer: *mut ScRenderer,
            rgb: *const f32,
            width: u32,
            height: u32,
            intensity: f32,
            rotation: f32,
            has_sun: bool,
            sun_direction: *const f32,
            sun_color: *const f32,
            sun_lux: f32,
        ) -> bool;
        pub fn sc_renderer_set_environment_transform(
            renderer: *mut ScRenderer,
            intensity: f32,
            rotation: f32,
            has_sun: bool,
            sun_direction: *const f32,
        );
        pub fn sc_renderer_set_background(renderer: *mut ScRenderer, show: bool, blur: f32);
        pub fn sc_renderer_set_rig_lights(
            renderer: *mut ScRenderer,
            count: u32,
            directions: *const f32,
            colors: *const f32,
            lux: *const f32,
        );
        pub fn sc_renderer_set_color_grading(
            renderer: *mut ScRenderer,
            tone_mapper: u32,
            exposure: f32,
            lut: *const f32,
            lut_size: u32,
        ) -> bool;
        pub fn sc_renderer_set_global_illumination(
            renderer: *mut ScRenderer,
            enabled: bool,
            radius: f32,
        );
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

/// Mip chains kept for one primitive's maps.
struct RetainedChains {
    primitive: u32,
    chains: [Option<MipChain>; MAP_COUNT],
}

pub struct FilamentRuntime {
    raw: NonNull<ffi::ScRenderer>,
    /// Last uploaded content key per primitive and map.
    map_keys: Vec<[Option<usize>; MAP_COUNT]>,
    /// CPU mip chains of the slot last patched, so further patches update
    /// them in place.
    retained: Option<RetainedChains>,
    /// Primitive drawing each mesh material slot; `None` for slots without
    /// triangles.
    slot_primitives: Vec<Option<u32>>,
    preview_mode: Option<bool>,
    /// Screen-space global illumination (AO, contact shadows, reflections).
    global_illumination: bool,
    /// Rig lights last sent to the renderer.
    rig_lights: Vec<SunLight>,
    /// Colour grading last applied: view transform, exposure, and LUT identity.
    grading: Option<(ViewTransform, u32, usize)>,
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
                BACKGROUND_MATERIAL.as_ptr(),
                BACKGROUND_MATERIAL.len(),
                size[0],
                size[1],
            )
        };
        let raw = NonNull::new(raw).ok_or(FilamentError::EngineUnavailable)?;
        let runtime = Self {
            raw,
            map_keys: vec![[None; MAP_COUNT]],
            retained: None,
            slot_primitives: vec![Some(0)],
            preview_mode: None,
            global_illumination: false,
            rig_lights: Vec::new(),
            grading: None,
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
        self.upload_map(slot, map, key, None, width, height, rgba8)
    }

    /// Like [`Self::set_map`] for `rgba8` that differs from the content
    /// uploaded under `base_key` only inside `regions`: when that content is
    /// still on the map, only those texels (and the mips above them) are
    /// re-uploaded. Otherwise the whole map is.
    #[allow(clippy::too_many_arguments)]
    pub fn patch_map(
        &mut self,
        slot: usize,
        map: MaterialMap,
        key: usize,
        base_key: usize,
        regions: &[TexelRect],
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<(), FilamentError> {
        self.upload_map(
            slot,
            map,
            key,
            Some((base_key, regions)),
            width,
            height,
            rgba8,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_map(
        &mut self,
        slot: usize,
        map: MaterialMap,
        key: usize,
        patch: Option<(usize, &[TexelRect])>,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<(), FilamentError> {
        let Some(primitive) = self.slot_primitives.get(slot).copied().flatten() else {
            return Ok(());
        };
        let map_index = map as usize;
        let Some(keys) = self.map_keys.get(primitive as usize) else {
            return Ok(());
        };
        let shown = keys[map_index];
        if shown == Some(key) {
            return Ok(());
        }
        let byte_count = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .filter(|bytes| *bytes <= MAX_MAP_PIXELS * 4)
            .ok_or(FilamentError::InvalidTexture)?;
        if width == 0 || height == 0 || rgba8.len() != byte_count {
            return Err(FilamentError::InvalidTexture);
        }
        let srgb = matches!(map, MaterialMap::BaseColor | MaterialMap::Emissive);
        // Texel blocks to send when the map still shows `base_key` and its
        // chain is retained.
        let mut blocks = None;
        if let Some((base_key, regions)) = patch
            && shown == Some(base_key)
            && let Some(retained) = self.retained.as_mut()
            && retained.primitive == primitive
            && let Some(chain) = retained.chains[map_index].as_mut()
            && chain.matches(width, height, srgb)
        {
            let mut changed = Vec::new();
            for region in regions {
                for (level, rect) in chain.update(rgba8, *region).into_iter().enumerate() {
                    changed.push((level, rect, chain.levels[level].extract(rect)));
                }
            }
            blocks = Some(changed);
        }
        if let Some(blocks) = blocks {
            for (level, rect, bytes) in blocks {
                self.upload_region(primitive, map_index, level, rect, &bytes)?;
            }
        } else {
            let chain = MipChain::build(width, height, rgba8, srgb);
            // SAFETY: `raw` is a live renderer owned by `self`.
            let created = unsafe {
                ffi::sc_renderer_create_map(
                    self.raw.as_ptr(),
                    primitive,
                    map_index as u32,
                    width,
                    height,
                    chain.levels.len() as u32,
                )
            };
            if !created {
                return Err(FilamentError::TextureUnavailable);
            }
            for (level, data) in chain.levels.iter().enumerate() {
                let rect = TexelRect::full(data.width, data.height);
                self.upload_region(primitive, map_index, level, rect, &data.rgba8)?;
            }
            // Keep the chain of the slot being painted, so later patches
            // only touch what changed; one slot's chains at a time bounds
            // the memory this costs.
            if patch.is_some() {
                let retained = self
                    .retained
                    .take()
                    .filter(|retained| retained.primitive == primitive)
                    .unwrap_or(RetainedChains {
                        primitive,
                        chains: Default::default(),
                    });
                self.retained = Some(retained);
            }
            if let Some(retained) = self
                .retained
                .as_mut()
                .filter(|retained| retained.primitive == primitive)
            {
                retained.chains[map_index] = patch.is_some().then_some(chain);
            }
        }
        if let Some(keys) = self.map_keys.get_mut(primitive as usize) {
            keys[map_index] = Some(key);
        }
        Ok(())
    }

    fn upload_region(
        &mut self,
        primitive: u32,
        map_index: usize,
        level: usize,
        rect: TexelRect,
        rgba8: &[u8],
    ) -> Result<(), FilamentError> {
        if rgba8.len() != rect.len() * 4 {
            return Err(FilamentError::InvalidTexture);
        }
        // SAFETY: `rgba8` holds exactly rect.width * rect.height * 4 bytes; the shim
        // copies them.
        let uploaded = unsafe {
            ffi::sc_renderer_set_map_region(
                self.raw.as_ptr(),
                primitive,
                map_index as u32,
                level as u32,
                rect.x,
                rect.y,
                rect.width,
                rect.height,
                rgba8.as_ptr(),
            )
        };
        if uploaded {
            Ok(())
        } else {
            Err(FilamentError::TextureUnavailable)
        }
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

    /// Lights the scene with `image` (prefiltered on the GPU) at `lux` per unit of radiance,
    /// turned by `rotation` radians about +Y together with `sun` (given unrotated); without a
    /// sun the directional light is off.
    pub fn set_environment(
        &mut self,
        image: &EnvironmentImage,
        lux: f32,
        rotation: f32,
        sun: Option<SunLight>,
    ) -> Result<(), FilamentError> {
        let texels = image.width as usize * image.height as usize;
        if image.rgb.len() != texels * 3 || !lux.is_finite() || lux < 0.0 {
            return Err(FilamentError::EnvironmentUnavailable);
        }
        let (direction, color, sun_lux) = sun_parameters(sun);
        // SAFETY: `raw` is live; the slices are read during the call and the shim copies the
        // image.
        let applied = unsafe {
            ffi::sc_renderer_set_environment(
                self.raw.as_ptr(),
                image.rgb.as_ptr(),
                image.width,
                image.height,
                lux,
                finite_or_zero(rotation),
                sun.is_some(),
                direction.as_ptr(),
                color.as_ptr(),
                sun_lux,
            )
        };
        if applied {
            Ok(())
        } else {
            Err(FilamentError::EnvironmentUnavailable)
        }
    }

    /// Changes the strength and rotation of the environment last set, without prefiltering it
    /// again.
    pub fn set_environment_transform(&mut self, lux: f32, rotation: f32, sun: Option<SunLight>) {
        let (direction, _, _) = sun_parameters(sun);
        // SAFETY: `raw` is live and `direction` is a three-element array read during the call.
        unsafe {
            ffi::sc_renderer_set_environment_transform(
                self.raw.as_ptr(),
                finite_or_zero(lux).max(0.0),
                finite_or_zero(rotation),
                sun.is_some(),
                direction.as_ptr(),
            );
        }
    }

    /// Shows the environment behind the mesh (`Some(blur)`, blur `0..=1` from sharp to its
    /// roughest prefiltered level) through a virtual perspective lens aimed like the camera, or
    /// hides it (`None`). It appears once an environment is set.
    pub fn set_background(&mut self, background: Option<f32>) {
        let blur = background.map_or(0.0, |blur| finite_or_zero(blur).clamp(0.0, 1.0));
        // SAFETY: `raw` is a live renderer owned by `self`.
        unsafe { ffi::sc_renderer_set_background(self.raw.as_ptr(), background.is_some(), blur) };
    }

    /// Lights the scene with up to four extra directional lights (a light rig); an empty list
    /// removes them.
    pub fn set_rig_lights(&mut self, lights: &[SunLight]) {
        let lights = &lights[..lights.len().min(4)];
        if lights == self.rig_lights.as_slice() {
            return;
        }
        let mut directions = Vec::with_capacity(lights.len() * 3);
        let mut colors = Vec::with_capacity(lights.len() * 3);
        let mut lux = Vec::with_capacity(lights.len());
        for light in lights {
            let (direction, color, illuminance) = sun_parameters(Some(*light));
            directions.extend(direction);
            colors.extend(color);
            lux.push(illuminance);
        }
        // SAFETY: `raw` is live; each array holds `lights.len()` entries, read during the call.
        unsafe {
            ffi::sc_renderer_set_rig_lights(
                self.raw.as_ptr(),
                lights.len() as u32,
                directions.as_ptr(),
                colors.as_ptr(),
                lux.as_ptr(),
            );
        }
        self.rig_lights = lights.to_vec();
    }

    /// Sets the view transform, exposure (stops), and an optional display LUT applied to the
    /// display-referred sRGB image, for the viewport and renders alike.
    pub fn set_color_grading(
        &mut self,
        transform: ViewTransform,
        exposure: f32,
        lut: Option<&CubeLut>,
    ) -> Result<(), FilamentError> {
        let exposure = finite_or_zero(exposure).clamp(-10.0, 10.0);
        let key = (
            transform,
            exposure.to_bits(),
            lut.map_or(0, |lut| lut.data.as_ptr() as usize),
        );
        if self.grading == Some(key) {
            return Ok(());
        }
        let data: Vec<f32> = lut
            .map(|lut| lut.data.iter().flat_map(|entry| entry.to_array()).collect())
            .unwrap_or_default();
        let size = lut.map_or(0, |lut| lut.size as u32);
        let tone_mapper = match transform {
            ViewTransform::Standard => 0,
            ViewTransform::Aces => 1,
            ViewTransform::Filmic => 2,
            ViewTransform::Agx => 3,
            ViewTransform::PbrNeutral => 4,
            ViewTransform::Linear => 5,
        };
        // SAFETY: `raw` is live; `data` holds `size`³ RGB entries, read during the call.
        let applied = unsafe {
            ffi::sc_renderer_set_color_grading(
                self.raw.as_ptr(),
                tone_mapper,
                exposure,
                if data.is_empty() {
                    std::ptr::null()
                } else {
                    data.as_ptr()
                },
                size,
            )
        };
        if !applied {
            return Err(FilamentError::ColorGradingUnavailable);
        }
        self.grading = Some(key);
        Ok(())
    }

    /// Turns screen-space global illumination on or off: ambient occlusion with bent normals
    /// and sun contact shadows, plus screen-space reflections, scaled to the mesh.
    pub fn set_global_illumination(&mut self, enabled: bool) {
        self.global_illumination = enabled;
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
        // Effects reach a fraction of the mesh size, whatever its units; applied each frame so
        // contact shadows follow the sun.
        // SAFETY: `raw` is a live renderer owned by `self`.
        unsafe {
            ffi::sc_renderer_set_global_illumination(
                self.raw.as_ptr(),
                self.global_illumination,
                bounds.1.length().max(0.01),
            );
        }

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
        self.retained = None;
        self.slot_primitives = slot_primitives;
        self.mesh_bounds = Some(bounds);
        Ok(())
    }

    /// Replaces the vertices of the uploaded mesh with `geometry`'s, which must have the same
    /// vertices and triangles in the same order (a posed copy). Textures and framing stay.
    pub fn update_vertices(
        &mut self,
        geometry: &MeshGeometry,
        smoothing_angle_degrees: f32,
    ) -> Result<(), FilamentError> {
        if self.mesh_bounds.is_none() || geometry.vertices.len() > MAX_MESH_VERTICES {
            return Err(FilamentError::InvalidMesh);
        }
        let MeshBuffers { vertex_data, .. } =
            build_mesh_buffers(geometry, smoothing_angle_degrees)?;
        let vertex_count =
            u32::try_from(geometry.vertices.len()).map_err(|_| FilamentError::InvalidMesh)?;
        // SAFETY: `vertex_data` holds `vertex_count` vertices; the shim copies it.
        let updated = unsafe {
            ffi::sc_renderer_set_vertices(self.raw.as_ptr(), vertex_data.as_ptr(), vertex_count)
        };
        if updated {
            Ok(())
        } else {
            Err(FilamentError::InvalidMesh)
        }
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

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

/// Direction, colour, and lux of `sun`, with harmless values when absent.
fn sun_parameters(sun: Option<SunLight>) -> ([f32; 3], [f32; 3], f32) {
    match sun {
        Some(sun) => (
            sun.direction.normalize_or(Vec3::NEG_Y).to_array(),
            sun.color.to_array(),
            finite_or_zero(sun.illuminance).max(0.0),
        ),
        None => ([0.0, -1.0, 0.0], [1.0; 3], 0.0),
    }
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
    #[error("Filament could not prefilter the lighting environment")]
    EnvironmentUnavailable,
    #[error("Filament could not build the colour grading")]
    ColorGradingUnavailable,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::mpsc::sync_channel;
    use std::time::{Duration, Instant};

    use substancecraft_core::{PrimitivePreset, TexelRect, ViewportCamera};

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
    fn tc_0256_patched_maps_render_like_full_uploads() {
        let mut renderer = FilamentRuntime::new(64, 64).expect("Filament runtime");
        renderer
            .set_mesh(&PrimitivePreset::Cube.build_mesh(), 60.0)
            .expect("cube mesh upload");
        renderer.set_preview_mode(true);
        let size = 64_u32;
        let fill = |left: [u8; 4], right: [u8; 4]| -> Vec<u8> {
            (0..size * size)
                .flat_map(|index| if index % size < size / 2 { left } else { right })
                .collect()
        };
        let red = fill([220, 30, 30, 255], [220, 30, 30, 255]);
        let split = fill([220, 30, 30, 255], [30, 40, 220, 255]);
        let right_half = TexelRect {
            x: size / 2,
            y: 0,
            width: size / 2,
            height: size,
        };
        let base = MaterialMap::BaseColor;
        renderer.set_map(0, base, 1, size, size, &red).expect("red");
        let all_red = render_frame(&mut renderer);
        // The first patch has no retained chain, so it uploads whole and
        // keeps the chain; the second updates only the right half.
        let full = [TexelRect::full(size, size)];
        renderer
            .patch_map(0, base, 2, 1, &full, size, size, &red)
            .expect("retaining upload");
        assert!(renderer.retained.as_ref().is_some_and(|retained| {
            retained.primitive == 0 && retained.chains[base as usize].is_some()
        }));
        renderer
            .patch_map(0, base, 3, 2, &[right_half], size, size, &split)
            .expect("patch");
        let patched = render_frame(&mut renderer);
        renderer
            .set_map(0, base, 4, size, size, &split)
            .expect("split");
        let uploaded = render_frame(&mut renderer);
        assert!(
            mean_difference(&all_red, &patched) > 1.0,
            "the patch must show"
        );
        assert!(
            // Identical frames differ by about 0.35 (dithering).
            mean_difference(&patched, &uploaded) < 0.5,
            "a patched map renders like a full upload"
        );
        assert!(
            renderer
                .retained
                .as_ref()
                .is_some_and(|retained| retained.chains[base as usize].is_none()),
            "a plain upload drops the stale chain"
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

    fn render_with_camera(renderer: &mut FilamentRuntime, camera: ViewportCamera) -> Vec<u8> {
        let mut last = Vec::new();
        // The first frames after an environment change may still show old lighting.
        for _ in 0..4 {
            let (sender, receiver) = sync_channel(1);
            let rendered = renderer
                .render(camera, 64, 64, 1.0, move |pixels| {
                    let _ = sender.try_send(pixels);
                })
                .expect("Filament frame submission");
            assert!(rendered);
            let deadline = Instant::now() + Duration::from_secs(5);
            last = loop {
                renderer.poll();
                if let Ok(pixels) = receiver.try_recv() {
                    break pixels;
                }
                assert!(Instant::now() < deadline, "Filament readback timed out");
                std::thread::sleep(Duration::from_millis(5));
            };
        }
        last
    }

    #[test]
    #[ignore = "requires an OpenGL driver for Filament's headless swap chain"]
    fn tc_0378_environment_backgrounds_follow_the_view_and_gi_renders() {
        use glam::Vec3;
        use substancecraft_core::EnvironmentImage;

        let image = EnvironmentImage::from_fn(128, |d| d.max(Vec3::ZERO).powf(4.0) * 3.0);
        let mut renderer = FilamentRuntime::new(64, 64).expect("Filament runtime");
        renderer
            .set_mesh(&PrimitivePreset::Cube.build_mesh(), 0.0)
            .expect("cube mesh upload");
        renderer.set_preview_mode(false);
        renderer
            .set_environment(&image, 30_000.0, 0.0, None)
            .expect("environment");
        // From the -X side looking towards +X, where the environment is red.
        let camera = (0..4)
            .map(|yaw| ViewportCamera {
                yaw: yaw as f32 * std::f32::consts::FRAC_PI_2,
                pitch: 0.0,
                ..ViewportCamera::default()
            })
            .find(|camera| (camera.rotation().inverse() * Vec3::Z).dot(Vec3::NEG_X) > 0.99)
            .expect("a camera on the -X side");
        let corner = |frame: &[u8]| [frame[0], frame[1], frame[2]];

        let plain = render_with_camera(&mut renderer, camera);
        renderer.set_background(Some(0.0));
        let shown = render_with_camera(&mut renderer, camera);
        assert!(
            corner(&shown)[0] > corner(&plain)[0] + 40,
            "the red sky shows behind the mesh: {:?} vs {:?}",
            corner(&shown),
            corner(&plain)
        );
        assert!(corner(&shown)[0] > corner(&shown)[2] + 40);
        // The mesh still covers the middle (up to dithering).
        let center = (32 * 64 + 32) * 4;
        assert!(
            shown[center..center + 3]
                .iter()
                .zip(&plain[center..center + 3])
                .all(|(a, b)| a.abs_diff(*b) <= 3)
        );

        // Turning the environment half round turns the sky with it.
        renderer.set_environment_transform(30_000.0, std::f32::consts::PI, None);
        let turned = render_with_camera(&mut renderer, camera);
        assert!(corner(&turned)[0] + 40 < corner(&shown)[0], "{:?}", corner(&turned));
        renderer.set_environment_transform(30_000.0, 0.0, None);

        // A blurred background is still the environment's colour.
        renderer.set_background(Some(1.0));
        let blurred = render_with_camera(&mut renderer, camera);
        assert!(corner(&blurred)[0] > corner(&plain)[0]);

        // Hidden again, the clear colour returns.
        renderer.set_background(None);
        let hidden = corner(&render_with_camera(&mut renderer, camera));
        assert!(hidden.iter().zip(corner(&plain)).all(|(a, b)| a.abs_diff(b) <= 3));

        // Screen-space GI renders and keeps the frame recognisable.
        renderer.set_global_illumination(true);
        let lit = render_with_camera(&mut renderer, camera);
        assert_eq!(lit.len(), plain.len());
        let difference: i32 = lit[center..center + 3]
            .iter()
            .zip(&plain[center..center + 3])
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .sum();
        assert!(difference < 120, "{:?} vs {:?}", &lit[center..center + 3], &plain[center..center + 3]);

        // In a concave corner (a floor meeting a wall) GI darkens the crease.
        use substancecraft_core::{MeshGeometry, MeshVertex};
        let vertex = |x: f32, y: f32, z: f32, u: f32, v: f32| MeshVertex {
            position: Vec3::new(x, y, z),
            uv: glam::Vec2::new(u, v),
        };
        let corner_mesh = MeshGeometry {
            vertices: vec![
                // Floor (y = 0), facing up.
                vertex(-1.0, 0.0, 1.0, 0.0, 0.0),
                vertex(1.0, 0.0, 1.0, 0.5, 0.0),
                vertex(1.0, 0.0, -1.0, 0.5, 1.0),
                vertex(-1.0, 0.0, -1.0, 0.0, 1.0),
                // Wall (z = -1), facing the camera.
                vertex(-1.0, 0.0, -1.0, 0.5, 0.0),
                vertex(1.0, 0.0, -1.0, 1.0, 0.0),
                vertex(1.0, 2.0, -1.0, 1.0, 1.0),
                vertex(-1.0, 2.0, -1.0, 0.5, 1.0),
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
            ..MeshGeometry::default()
        };
        renderer.set_global_illumination(false);
        renderer.set_mesh(&corner_mesh, 0.0).expect("corner mesh");
        renderer
            .set_environment(&EnvironmentImage::from_fn(64, |_| Vec3::ONE), 30_000.0, 0.0, None)
            .expect("even environment");
        let view = ViewportCamera {
            pitch: 0.5,
            ..ViewportCamera::default()
        };
        let brightness = |frame: &[u8]| frame.iter().map(|v| u64::from(*v)).sum::<u64>();
        let off = render_with_camera(&mut renderer, view);
        let again = render_with_camera(&mut renderer, view);
        renderer.set_global_illumination(true);
        let on = render_with_camera(&mut renderer, view);
        let noise = brightness(&off).abs_diff(brightness(&again));
        assert!(noise < 1_000, "frames repeat: {noise}");
        assert!(
            brightness(&on) + 10_000 < brightness(&off),
            "occlusion darkens the crease: {} vs {}",
            brightness(&on),
            brightness(&off)
        );
    }

    #[test]
    #[ignore = "requires an OpenGL driver for Filament's headless swap chain"]
    fn tc_0381_light_rigs_and_view_transforms_change_the_frame() {
        use glam::Vec3;
        use substancecraft_core::{CubeLut, EnvironmentImage, SunLight, ViewTransform};

        let mut renderer = FilamentRuntime::new(64, 64).expect("Filament runtime");
        renderer
            .set_mesh(&PrimitivePreset::Cube.build_mesh(), 0.0)
            .expect("cube mesh upload");
        renderer.set_preview_mode(false);
        renderer
            .set_environment(&EnvironmentImage::from_fn(32, |_| Vec3::splat(0.2)), 5_000.0, 0.0, None)
            .expect("dim environment");
        let camera = ViewportCamera::default();
        let center = (32 * 64 + 32) * 4;
        let luma = |frame: &[u8]| frame[center..center + 3].iter().map(|v| u32::from(*v)).sum::<u32>();

        // A rig light shining at the camera-facing side brightens it; removing it restores.
        let toward = camera.rotation().inverse() * Vec3::Z;
        let dim = render_with_camera(&mut renderer, camera);
        renderer.set_rig_lights(&[SunLight {
            direction: -toward,
            color: Vec3::ONE,
            illuminance: 60_000.0,
        }]);
        let lit = render_with_camera(&mut renderer, camera);
        assert!(luma(&lit) > luma(&dim) + 60, "{} vs {}", luma(&lit), luma(&dim));
        renderer.set_rig_lights(&[]);
        assert!(luma(&render_with_camera(&mut renderer, camera)).abs_diff(luma(&dim)) <= 6);

        // Exposure and view transforms change the image; Standard with nothing else is the
        // default look.
        renderer
            .set_color_grading(ViewTransform::Standard, 2.0, None)
            .expect("grading");
        assert!(luma(&render_with_camera(&mut renderer, camera)) > luma(&dim) + 30);
        renderer
            .set_color_grading(ViewTransform::Linear, 0.0, None)
            .expect("grading");
        let linear = render_with_camera(&mut renderer, camera);
        assert_ne!(luma(&linear), luma(&dim));

        // A display LUT that inverts colours inverts the frame.
        let mut text = String::from("LUT_3D_SIZE 2\n");
        for b in 0..2 {
            for g in 0..2 {
                for r in 0..2 {
                    text.push_str(&format!("{} {} {}\n", 1 - r, 1 - g, 1 - b));
                }
            }
        }
        let invert = CubeLut::parse(&text).expect("inverting LUT");
        renderer
            .set_color_grading(ViewTransform::Standard, 0.0, Some(&invert))
            .expect("grading with LUT");
        let inverted = render_with_camera(&mut renderer, camera);
        let corner = |frame: &[u8]| u32::from(frame[0]) + u32::from(frame[1]) + u32::from(frame[2]);
        assert!(corner(&inverted) > 600, "the dark background turns light: {}", corner(&inverted));
        renderer
            .set_color_grading(ViewTransform::Standard, 0.0, None)
            .expect("default grading");
        assert!(luma(&render_with_camera(&mut renderer, camera)).abs_diff(luma(&dim)) <= 6);
    }

    #[test]
    #[ignore = "requires an OpenGL driver for Filament's headless swap chain"]
    fn tc_0313_environment_lights_faces_from_their_own_direction() {
        use glam::Vec3;
        use substancecraft_core::{EnvironmentImage, LightingPreset};

        // Red from +X, green from +Y, blue from +Z, black elsewhere, laid
        // out as the presets lay out their images.
        let image = EnvironmentImage::from_fn(128, |d| d.max(Vec3::ZERO).powf(4.0) * 3.0);

        let mut renderer = FilamentRuntime::new(64, 64).expect("Filament runtime");
        renderer
            .set_mesh(&PrimitivePreset::Cube.build_mesh(), 0.0)
            .expect("cube mesh upload");
        renderer.set_preview_mode(false);
        renderer
            .set_environment(&image, 30_000.0, 0.0, None)
            .expect("environment");

        // Cameras looking straight at the +X, +Y, and +Z faces.
        let looking_along = |axis: Vec3| {
            let mut cameras = (0..4).flat_map(|yaw| {
                [0.0, 1.55, -1.55].map(move |pitch| ViewportCamera {
                    yaw: yaw as f32 * std::f32::consts::FRAC_PI_2,
                    pitch,
                    ..ViewportCamera::default()
                })
            });
            cameras
                .find(|camera| (camera.rotation().inverse() * Vec3::Z).dot(axis) > 0.99)
                .expect("a camera along the axis")
        };
        let center = (32 * 64 + 32) * 4;
        for (label, orm) in [
            ("diffuse", [255, 230, 0, 255]),
            ("mirror", [255, 8, 255, 255]),
        ] {
            renderer
                .set_map(0, MaterialMap::Orm, 10 + orm[1] as usize, 1, 1, &orm)
                .expect("ORM upload");
            for (axis, channel) in [(Vec3::X, 0), (Vec3::Y, 1), (Vec3::Z, 2)] {
                let pixels = render_with_camera(&mut renderer, looking_along(axis));
                let texel = &pixels[center..center + 3];
                let others = (0..3).filter(|other| *other != channel);
                for other in others {
                    assert!(
                        texel[channel] > texel[other] + 20,
                        "{label}: the face towards {axis} should take channel {channel}: {texel:?}"
                    );
                }
            }
        }

        // Every preset prefilters and renders; rotating changes the frame.
        renderer
            .set_map(0, MaterialMap::Orm, 99, 1, 1, &[255, 140, 0, 255])
            .expect("ORM upload");
        let camera = ViewportCamera::default();
        let mut frames = Vec::new();
        for preset in LightingPreset::ALL {
            renderer
                .set_environment(
                    &preset.equirect(256),
                    preset.environment_lux(),
                    0.0,
                    preset.sun(),
                )
                .expect("preset environment");
            frames.push(render_with_camera(&mut renderer, camera));
        }
        // Night is darker overall than the evenly lit daylight presets
        // (studio, overcast, bright outdoor).
        let brightness = |frame: &[u8]| -> f32 { frame.iter().map(|v| f32::from(*v)).sum() };
        let night = brightness(&frames[5]);
        for index in [0, 1, 4] {
            assert!(
                brightness(&frames[index]) > night,
                "{:?} should be brighter than night",
                LightingPreset::ALL[index]
            );
        }
        let unrotated = render_with_camera(&mut renderer, camera);
        renderer.set_environment_transform(
            LightingPreset::Night.environment_lux(),
            std::f32::consts::PI,
            LightingPreset::Night.sun(),
        );
        let rotated = render_with_camera(&mut renderer, camera);
        assert!(
            mean_difference(&unrotated, &rotated) > 0.5,
            "rotation turns the lighting"
        );
    }
}
