#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::mem::size_of;

use filament_bindings::{
    backend::{
        self, Backend, BufferDescriptor, ElementType, PixelBufferDescriptor,
        PixelDataFormat, PixelDataType,
    },
    filament::{
        Bounds, Camera, ClearOptions, Engine, IndexBuffer, IndexBufferBuilder, LightBuilder,
        IndexType, LightType, LinearColor, Material, MaterialInstance, RenderableBuilder, Renderer,
        Scene, SwapChain, SwapChainConfig, VertexAttribute, VertexBuffer, VertexBufferBuilder,
        View, Viewport,
    },
    math::{Float3, Float4},
    utils::Entity,
};
use glam::{Mat3, Quat, Vec3};
use substancecraft_core::{MeshGeometry, ViewportCamera};
use thiserror::Error;

const MAX_VIEWPORT_DIMENSION: u32 = 2048;
const MAX_VIEWPORT_PIXELS: usize = 4_194_304;
const MAX_MESH_VERTICES: usize = 3_000_000;

pub struct FilamentRuntime {
    engine: Engine,
    swap_chain: Option<SwapChain>,
    renderer: Option<Renderer>,
    view: Option<View>,
    scene: Option<Scene>,
    camera: Option<Camera>,
    camera_entity: Option<Entity>,
    light_entity: Option<Entity>,
    material: Option<Material>,
    material_instance: Option<MaterialInstance>,
    mesh_entity: Option<Entity>,
    vertex_buffer: Option<VertexBuffer>,
    index_buffer: Option<IndexBuffer>,
    mesh_bounds: Option<(Vec3, Vec3)>,
    target_size: [u32; 2],
}

impl FilamentRuntime {
    pub fn new(width: u32, height: u32) -> Result<Self, FilamentError> {
        let size = bounded_size(width, height)?;
        let engine = unsafe { Engine::create(Backend::VULKAN) }
            .ok_or(FilamentError::EngineUnavailable)?;
        let mut runtime = Self {
            engine,
            swap_chain: None,
            renderer: None,
            view: None,
            scene: None,
            camera: None,
            camera_entity: None,
            light_entity: None,
            material: None,
            material_instance: None,
            mesh_entity: None,
            vertex_buffer: None,
            index_buffer: None,
            mesh_bounds: None,
            target_size: size,
        };
        runtime.initialize()?;
        Ok(runtime)
    }

    fn initialize(&mut self) -> Result<(), FilamentError> {
        self.swap_chain = Some(unsafe {
            self.engine.create_headless_swap_chain(
                self.target_size[0],
                self.target_size[1],
                SwapChainConfig::READABLE,
            )
        }
        .ok_or(FilamentError::SwapChainUnavailable)?);
        self.renderer = Some(
            unsafe { self.engine.create_renderer() }.ok_or(FilamentError::RendererUnavailable)?,
        );
        self.scene = Some(
            unsafe { self.engine.create_scene() }.ok_or(FilamentError::SceneUnavailable)?,
        );
        self.view = Some(unsafe { self.engine.create_view() }.ok_or(FilamentError::ViewUnavailable)?);

        let mut entities = unsafe { self.engine.get_entity_manager() }
            .ok_or(FilamentError::EntityManagerUnavailable)?;
        let camera_entity = unsafe { entities.create() };
        self.camera = Some(
            unsafe { self.engine.create_camera(&camera_entity) }
                .ok_or(FilamentError::CameraUnavailable)?,
        );
        self.camera_entity = Some(camera_entity);

        let light_entity = unsafe { entities.create() };
        let mut light_builder = unsafe { LightBuilder::new(LightType::DIRECTIONAL) }
            .ok_or(FilamentError::LightUnavailable)?;
        let light_color = LinearColor(Float3::new(1.0, 0.96, 0.9));
        let light_direction = Float3::new(-0.45, -0.7, -1.0);
        unsafe {
            light_builder
                .direction(&light_direction)
                .color(&light_color)
                .intensity(80_000.0);
        }
        if unsafe { light_builder.build(&mut self.engine, &light_entity) }.is_none() {
            return Err(FilamentError::LightUnavailable);
        }
        self.light_entity = Some(light_entity);

        let default_material = unsafe { self.engine.get_default_material() };
        self.material = Material::try_from_native(default_material.cast_mut())
            .ok_or(FilamentError::MaterialUnavailable)?
            .into();
        let material_instance = unsafe {
            self.material
                .as_ref()
                .and_then(|material| material.create_instance())
        }
        .ok_or(FilamentError::MaterialUnavailable)?;
        self.material_instance = Some(material_instance);
        if let Some(instance) = self.material_instance.as_mut() {
            unsafe {
                instance
                    .set_rgb_parameter(
                        "baseColor",
                        filament_bindings::filament::RgbType::LINEAR,
                        Float3::new(0.66, 0.68, 0.7),
                    )
                    .map_err(|_| FilamentError::MaterialUnavailable)?;
                instance
                    .set_float_parameter("roughness", &0.58)
                    .map_err(|_| FilamentError::MaterialUnavailable)?;
                instance
                    .set_float_parameter("metallic", &0.0)
                    .map_err(|_| FilamentError::MaterialUnavailable)?;
            }
        }

        self.configure_viewport(self.target_size[0], self.target_size[1])?;
        let clear_options = ClearOptions {
            clear_color: Float4::new(0.025, 0.03, 0.035, 1.0),
            clear: true,
            discard: true,
        };
        if let Some(renderer) = self.renderer.as_mut() {
            unsafe { renderer.set_clear_options(&clear_options) };
        }
        Ok(())
    }

    pub fn render(
        &mut self,
        geometry: &MeshGeometry,
        camera: ViewportCamera,
        width: u32,
        height: u32,
        readback: impl FnOnce(Vec<u8>) + 'static,
    ) -> Result<bool, FilamentError> {
        geometry
            .validate()
            .map_err(|_| FilamentError::InvalidMesh)?;
        let size = bounded_size(width, height)?;
        if size != self.target_size {
            self.resize(size)?;
        }
        if self.mesh_bounds.is_none() {
            self.set_mesh(geometry)?;
        }
        self.configure_camera(camera, size[0], size[1]);

        let (Some(renderer), Some(swap_chain), Some(view)) = (
            self.renderer.as_mut(),
            self.swap_chain.as_mut(),
            self.view.as_ref(),
        ) else {
            return Err(FilamentError::RendererUnavailable);
        };
        if !unsafe { renderer.begin_frame(swap_chain) } {
            return Ok(false);
        }
        unsafe { renderer.render(view) };
        let pixel_bytes = (size[0] as usize)
            .checked_mul(size[1] as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(FilamentError::InvalidViewport)?;
        let pixel_data = unsafe {
            PixelBufferDescriptor::new_callback(
                vec![0_u8; pixel_bytes],
                PixelDataFormat::RGBA,
                PixelDataType::UBYTE,
                move |mut pixels| {
                    flip_rows(&mut pixels, size[0], size[1]);
                    readback(pixels);
                },
            )
        };
        unsafe { renderer.read_pixels(0, 0, size[0], size[1], pixel_data) };
        unsafe { renderer.end_frame() };
        unsafe { self.engine.pump_message_queues() };
        Ok(true)
    }

    fn configure_viewport(&mut self, width: u32, height: u32) -> Result<(), FilamentError> {
        let viewport = Viewport {
            left: 0,
            bottom: 0,
            width,
            height,
        };
        let (Some(view), Some(camera), Some(scene)) = (
            self.view.as_mut(),
            self.camera.as_mut(),
            self.scene.as_mut(),
        ) else {
            return Err(FilamentError::ViewUnavailable);
        };
        unsafe {
            camera.set_projection_fov(45.0, f64::from(width) / f64::from(height), 0.01, 1000.0);
        }
        view.set_post_processing_enabled(true);
        view.set_viewport(&viewport);
        view.set_scene(scene);
        view.set_camera(camera);
        Ok(())
    }

    fn configure_camera(&mut self, camera: ViewportCamera, width: u32, height: u32) {
        let Some((center, extent)) = self.mesh_bounds else {
            return;
        };
        let rotation = camera.rotation();
        let distance = (extent.length() * 3.2 / camera.zoom.max(0.1)).max(1.0);
        let pan_scale = (width.min(height) as f32 * 0.42 * camera.zoom * 0.9).max(1.0);
        let target = center
            + rotation.inverse()
                * Vec3::new(-camera.pan.x / pan_scale, camera.pan.y / pan_scale, 0.0);
        let eye = target + rotation.inverse() * Vec3::new(0.0, 0.0, distance);
        let up = rotation.inverse() * Vec3::Y;
        if let Some(camera_component) = self.camera.as_mut() {
            let eye = Float3::new(eye.x, eye.y, eye.z);
            let target = Float3::new(target.x, target.y, target.z);
            let up = Float3::new(up.x, up.y, up.z);
            unsafe { camera_component.look_at_up(&eye, &target, &up) };
        }
    }

    fn resize(&mut self, size: [u32; 2]) -> Result<(), FilamentError> {
        if let Some(mut old_swap_chain) = self.swap_chain.take() {
            unsafe { self.engine.destroy_swap_chain(&mut old_swap_chain) };
        }
        self.swap_chain = Some(unsafe {
            self.engine
                .create_headless_swap_chain(size[0], size[1], SwapChainConfig::READABLE)
        }
        .ok_or(FilamentError::SwapChainUnavailable)?);
        self.target_size = size;
        self.configure_viewport(size[0], size[1])
    }

    fn set_mesh(&mut self, geometry: &MeshGeometry) -> Result<(), FilamentError> {
        if geometry.vertices.len() > MAX_MESH_VERTICES {
            return Err(FilamentError::InvalidMesh);
        }
        self.destroy_mesh();

        let (vertex_data, indices, bounds) = build_mesh_buffers(geometry)?;
        let mut vertex_builder = unsafe { VertexBufferBuilder::new() }
            .ok_or(FilamentError::MeshBufferUnavailable)?;
        unsafe {
            vertex_builder
                .vertex_count(geometry.vertices.len() as u32)
                .buffer_count(1)
                .attribute(VertexAttribute::POSITION, 0, ElementType::FLOAT3, 0, 36)
                .attribute(VertexAttribute::UV0, 0, ElementType::FLOAT2, 12, 36)
                .attribute(VertexAttribute::TANGENTS, 0, ElementType::FLOAT4, 20, 36);
        }
        let mut vertex_buffer = unsafe { vertex_builder.build(&mut self.engine) }
            .ok_or(FilamentError::MeshBufferUnavailable)?;
        unsafe {
            vertex_buffer.set_buffer_at(
                &mut self.engine,
                0,
                BufferDescriptor::new(vertex_data),
                0,
            );
        }

        let mut index_builder = unsafe { IndexBufferBuilder::new() }
            .ok_or(FilamentError::MeshBufferUnavailable)?;
        unsafe {
            index_builder
                .index_count(indices.len() as u32)
                .buffer_type(IndexType::UINT);
        }
        let mut index_buffer = unsafe { index_builder.build(&mut self.engine) }
            .ok_or(FilamentError::MeshBufferUnavailable)?;
        unsafe {
            index_buffer.set_buffer(&mut self.engine, BufferDescriptor::new(indices), 0);
        }

        let mut entities = unsafe { self.engine.get_entity_manager() }
            .ok_or(FilamentError::EntityManagerUnavailable)?;
        let entity = unsafe { entities.create() };
        let bounds = Bounds {
            center: bounds.0.to_array().into(),
            half_extent: bounds.1.to_array().into(),
        };
        let mut renderable_builder = unsafe { RenderableBuilder::new(1) }
            .ok_or(FilamentError::MeshBufferUnavailable)?;
        let Some(material_instance) = self.material_instance.as_mut() else {
            return Err(FilamentError::MaterialUnavailable);
        };
        let Some(scene) = self.scene.as_mut() else {
            return Err(FilamentError::SceneUnavailable);
        };
        unsafe {
            renderable_builder
                .bounding_box(&bounds)
                .material(0, material_instance)
                .geometry(
                    0,
                    backend::PrimitiveType::TRIANGLES,
                    &mut vertex_buffer,
                    &mut index_buffer,
                )
                .culling(false);
        }
        if unsafe { renderable_builder.build(&mut self.engine, &entity) }.is_none() {
            unsafe {
                self.engine.destroy_vertex_buffer(&mut vertex_buffer);
                self.engine.destroy_index_buffer(&mut index_buffer);
                self.engine.destroy_entity_components(&entity);
                entities.destory(&mut entity.clone());
            }
            return Err(FilamentError::MeshBufferUnavailable);
        }
        unsafe { scene.add_entity(&entity) };
        self.mesh_bounds = Some(bounds_center_extent(geometry));
        self.mesh_entity = Some(entity);
        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        Ok(())
    }

    fn destroy_mesh(&mut self) {
        if let Some(mut entity) = self.mesh_entity.take() {
            if let Some(scene) = self.scene.as_mut() {
                unsafe { scene.remove_entity(&entity) };
            }
            unsafe { self.engine.destroy_entity_components(&entity) };
            if let Some(mut entities) = unsafe { self.engine.get_entity_manager() } {
                unsafe { entities.destory(&mut entity) };
            }
        }
        if let Some(mut buffer) = self.vertex_buffer.take() {
            unsafe { self.engine.destroy_vertex_buffer(&mut buffer) };
        }
        if let Some(mut buffer) = self.index_buffer.take() {
            unsafe { self.engine.destroy_index_buffer(&mut buffer) };
        }
        self.mesh_bounds = None;
    }
}

impl Drop for FilamentRuntime {
    fn drop(&mut self) {
        self.destroy_mesh();
        if let Some(entity) = self.light_entity.take() {
            unsafe { self.engine.destroy_entity_components(&entity) };
            if let Some(mut entities) = unsafe { self.engine.get_entity_manager() } {
                let mut entity = entity;
                unsafe { entities.destory(&mut entity) };
            }
        }
        if let Some(entity) = self.camera_entity.take() {
            unsafe { self.engine.destroy_camera_component(&entity) };
            if let Some(mut entities) = unsafe { self.engine.get_entity_manager() } {
                let mut entity = entity;
                unsafe { entities.destory(&mut entity) };
            }
        }
        unsafe { self.engine.flush_and_wait() };
        unsafe { Engine::destroy(&mut self.engine) };
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

fn build_mesh_buffers(geometry: &MeshGeometry) -> Result<(Vec<f32>, Vec<u32>, (Vec3, Vec3)), FilamentError> {
    let vertex_count = geometry.vertices.len();
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

    let mut vertex_data = Vec::with_capacity(vertex_count.saturating_mul(9));
    for (index, vertex) in geometry.vertices.iter().enumerate() {
        let normal = normals[index].normalize_or_zero();
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
        let bitangent = normal.cross(tangent) * orientation;
        let tangent_frame = Quat::from_mat3(&Mat3::from_cols(tangent, bitangent, normal));
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
    if vertex_data.len() != vertex_count.saturating_mul(size_of::<[f32; 9]>()) / size_of::<f32>() {
        return Err(FilamentError::InvalidMesh);
    }
    let indices = geometry
        .triangles
        .iter()
        .flat_map(|triangle| triangle.iter().copied())
        .collect();
    Ok((vertex_data, indices, bounds_center_extent(geometry)))
}

fn flip_rows(pixels: &mut [u8], width: u32, height: u32) {
    let row_bytes = width as usize * 4;
    for row in 0..height as usize / 2 {
        let opposite = height as usize - row - 1;
        let split = opposite * row_bytes;
        let (top, bottom) = pixels.split_at_mut(split);
        let top_row = &mut top[row * row_bytes..(row + 1) * row_bytes];
        let bottom_row = &mut bottom[..row_bytes];
        top_row.swap_with_slice(bottom_row);
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FilamentError {
    #[error("Filament could not initialize its Vulkan backend")]
    EngineUnavailable,
    #[error("Filament could not create the offscreen swap chain")]
    SwapChainUnavailable,
    #[error("Filament could not create its renderer")]
    RendererUnavailable,
    #[error("Filament could not create its scene")]
    SceneUnavailable,
    #[error("Filament could not create its view")]
    ViewUnavailable,
    #[error("Filament could not create a camera")]
    CameraUnavailable,
    #[error("Filament could not create the entity manager")]
    EntityManagerUnavailable,
    #[error("Filament could not create a directional light")]
    LightUnavailable,
    #[error("Filament default PBR material is unavailable")]
    MaterialUnavailable,
    #[error("mesh geometry or its buffers are invalid")]
    InvalidMesh,
    #[error("viewport dimensions exceed the Filament render budget")]
    InvalidViewport,
    #[error("Filament could not create mesh buffers")]
    MeshBufferUnavailable,
}