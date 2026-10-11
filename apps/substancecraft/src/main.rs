#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
};
use std::time::Duration;

use eframe::egui;
use substancecraft_api::{
    ImportedMesh, MAX_FBX_BYTES, TextureFormat, bake_texture_set_channel,
    bake_texture_set_material, import_binary_fbx_meshes, import_gltf_file, load_project_file,
    load_project_resources, load_texture_image, save_project_file, write_texture,
};
use substancecraft_core::{
    BakedTexture, BoneId, BoneTransform, BrushTip, DEFAULT_EMISSIVE, DEFAULT_HEIGHT,
    DEFAULT_HEIGHT_STRENGTH, DEFAULT_OPACITY, Dirt, EdgeWear, EditCommand, EditHistory, IdStroke,
    LayerFill, LayerId, LayerSettings, LayerSlot, MAX_STROKE_POINTS, MaskGenerator, MaskStroke,
    MaterialChannel, MaterialLayer, MaterialProject, MaterialTextures, MeshAsset, MeshGeometry,
    MeshId, MeshMapError, MeshMapKind, MeshMapSettings, MeshMaps, MeshSmoothing, MeshTopology,
    NamedPose, PaintStroke, PoseId, PrimitivePreset, ProjectResource, ResourceId, ResourceImages,
    SelectionMode, TextureResolution, TextureSet, TextureSetId, ViewportCamera, ViewportLighting,
    bake_mesh_maps_from, remap_subdivided_triangles,
};
use substancecraft_filament::{FilamentRuntime, MaterialMap};

mod icons;
mod panels;
mod theme;
mod uv_view;
mod widgets;

const CHANNELS: [MaterialChannel; 9] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Id,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Normal,
    MaterialChannel::Height,
    MaterialChannel::AmbientOcclusion,
    MaterialChannel::Emissive,
    MaterialChannel::Opacity,
];
/// Channels the compositor can bake, preview, and export.
const BAKEABLE_CHANNELS: [MaterialChannel; 8] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Height,
    MaterialChannel::Normal,
    MaterialChannel::Emissive,
    MaterialChannel::Opacity,
    MaterialChannel::Id,
];
/// Mesh formats the importer reads.
const MESH_EXTENSIONS: [&str; 3] = ["fbx", "gltf", "glb"];
/// Bitmap formats the resource importer reads.
const IMAGE_EXTENSIONS: [&str; 4] = ["png", "jpg", "jpeg", "tga"];
const POSE_TRANSITION_SECONDS: f64 = 0.35;
const MAX_SAVED_POSES: usize = 256;
/// Selection overlays stop drawing beyond this many triangles to keep the
/// viewport responsive on dense meshes.
const MAX_OVERLAY_TRIANGLES: usize = 200_000;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("SubstanceCraft")
            .with_app_id("com.substancecraft.desktop")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "SubstanceCraft",
        options,
        Box::new(|creation_context| {
            theme::apply(&creation_context.egui_ctx);
            let mut app = SubstanceCraftApp::default();
            match FilamentRuntime::new(64, 64) {
                Ok(renderer) => app.filament_renderer = Some(renderer),
                Err(error) => app.filament_error = Some(error.to_string()),
            }
            Ok(Box::new(app))
        }),
    )
}

struct SubstanceCraftApp {
    project: MaterialProject,
    selected_texture_set: Option<TextureSetId>,
    selected_layer: Option<LayerId>,
    selected_primitive: Option<PrimitivePreset>,
    /// Imported or primitive geometry after scaling, before smoothing.
    base_geometry: Option<MeshGeometry>,
    /// Smoothed/subdivided geometry used for display, painting, and baking.
    mesh_geometry: Option<Arc<MeshGeometry>>,
    mesh_topology: Option<MeshTopology>,
    mesh_center: glam::Vec3,
    mesh_smoothing: MeshSmoothing,
    mesh_error: Option<String>,
    mesh_revision: u64,
    filament_mesh_revision: Option<u64>,
    filament_renderer: Option<FilamentRuntime>,
    filament_error: Option<String>,
    filament_frame_texture: Option<egui::TextureHandle>,
    filament_frames_tx: SyncSender<FilamentFrame>,
    filament_frames_rx: Receiver<FilamentFrame>,
    filament_render_pending: Arc<AtomicBool>,
    imported_meshes: Vec<ImportedMesh>,
    selected_imported_mesh: Option<usize>,
    import_error: Option<String>,
    import_scale: f32,
    project_path: Option<std::path::PathBuf>,
    project_error: Option<String>,
    active_channel: MaterialChannel,
    next_texture_set_id: u64,
    next_layer_id: u64,
    current_pose: NamedPose,
    viewport_camera: ViewportCamera,
    viewport_lighting: ViewportLighting,
    paint_mode: PaintMode,
    brush_radius: f32,
    brush_color: egui::Color32,
    brush_roughness: f32,
    brush_metallic: f32,
    brush_height: f32,
    brush_emissive: egui::Color32,
    brush_material_opacity: f32,
    brush_tip: BrushTip,
    /// Latest pen pressure (`0..=1`) while a stylus touches; `None` for mice.
    pen_pressure: Option<f32>,
    /// Loaded pixels of `project.resources`.
    resource_images: ResourceImages,
    /// Per-resource egui thumbnails for the assets drawer.
    resource_thumbnails: BTreeMap<ResourceId, egui::TextureHandle>,
    export_format: TextureFormat,
    view_mode: ViewMode,
    uv_view: uv_view::UvView,
    /// Content revision of each texture set; compared against caches.
    set_revisions: BTreeMap<TextureSetId, u64>,
    /// Last full-material bake of each texture set, for drawing material
    /// slots other than the selected one.
    set_materials: BTreeMap<TextureSetId, CachedMaterial>,
    set_bake_job: Option<SetBakeJob>,
    /// Set revisions whose background bake failed, so they are not retried.
    set_bake_failed: BTreeMap<TextureSetId, u64>,
    /// When true the viewport shows `active_channel` flat instead of the full
    /// PBR material.
    channel_preview: bool,
    mask_value: f32,
    selection_mode: SelectionMode,
    hovered_selection: Vec<u32>,
    active_stroke: Option<(TextureSetId, LayerId, PaintMode)>,
    history: EditHistory,
    /// Last committed settings of the selected layer; edits made through the
    /// inspector are recorded as one history step once the pointer is up.
    layer_settled: Option<(TextureSetId, LayerId, LayerSettings)>,
    baked_preview: Option<BakedPreview>,
    preview_revision: u64,
    preview_job: Option<PreviewJob>,
    preview_failed: bool,
    bake_error: Option<String>,
    mesh_map_settings: MeshMapSettings,
    /// Mesh maps per mesh material slot, baked from each slot's triangles.
    mesh_maps: Option<Arc<Vec<MeshMaps>>>,
    mesh_maps_key: Option<BakeKey>,
    high_poly: Option<HighPolySource>,
    high_poly_revision: u64,
    mesh_map_job: Option<MeshMapJob>,
    mesh_map_error: Option<String>,
    mesh_map_preview: Option<MeshMapKind>,
    selected_pose: Option<PoseId>,
    pose_name: String,
    pose_transition: Option<PoseTransition>,
    drawer: Option<panels::Drawer>,
    last_drawer: panels::Drawer,
    show_inspector: bool,
    layout: panels::OverlayLayout,
}

struct FilamentFrame {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PaintMode {
    View,
    Paint,
    /// Removes paint from the selected paint layer.
    Erase,
    Mask,
    Id,
    Select,
}

impl PaintMode {
    fn paints(self) -> bool {
        matches!(self, Self::Paint | Self::Erase | Self::Mask | Self::Id)
    }
}

/// Which viewports share the central area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewMode {
    Model,
    Split,
    Uv,
}

/// A texture set's last full-material bake.
struct CachedMaterial {
    revision: u64,
    textures: Arc<MaterialTextures>,
}

/// Background bake of a texture set that is not selected, so every material
/// slot of a multi-material mesh shows its own textures.
struct SetBakeJob {
    set_id: TextureSetId,
    revision: u64,
    receiver: Receiver<Result<MaterialTextures, String>>,
}

/// What the viewport currently shows on the mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreviewTarget {
    /// Full PBR material: Base Color, ORM, and Normal.
    Material,
    /// One channel, shown flat for inspection.
    Channel(MaterialChannel),
    MeshMap(MeshMapKind),
}

struct BakedPreview {
    key: PreviewKey,
    textures: PreviewTextures,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreviewKey {
    texture_set: Option<TextureSetId>,
    target: PreviewTarget,
    /// Bumped on every edit that changes baked output.
    revision: u64,
}

struct PreviewJob {
    receiver: Receiver<Result<PreviewTextures, String>>,
    key: PreviewKey,
    /// Revision of the texture set when the bake started.
    set_revision: u64,
}

/// Shared inputs of a background texture-set bake.
struct BakeInputs {
    mesh: Option<Arc<MeshGeometry>>,
    mesh_maps: Option<Arc<Vec<MeshMaps>>>,
    /// Mesh material slot whose maps the set uses.
    slot: usize,
    resources: ResourceImages,
}

impl BakeInputs {
    fn maps(&self) -> Option<&MeshMaps> {
        self.mesh_maps
            .as_deref()
            .and_then(|maps| maps.get(self.slot))
    }

    fn bake_channel(
        &self,
        texture_set: &TextureSet,
        channel: MaterialChannel,
    ) -> Result<BakedTexture, String> {
        bake_texture_set_channel(
            texture_set,
            channel,
            self.mesh.as_deref(),
            self.maps(),
            Some(&self.resources),
        )
        .map_err(|error| error.to_string())
    }

    fn bake_material(&self, texture_set: &TextureSet) -> Result<MaterialTextures, String> {
        bake_texture_set_material(
            texture_set,
            self.mesh.as_deref(),
            self.maps(),
            Some(&self.resources),
        )
        .map_err(|error| error.to_string())
    }
}

enum PreviewTextures {
    Material(Arc<MaterialTextures>),
    Single(Arc<BakedTexture>),
}

/// What one mesh material slot shows in the 3D viewport.
enum SlotSource {
    Material(Arc<MaterialTextures>),
    Single(Arc<BakedTexture>),
    Neutral,
}

impl PreviewTextures {
    /// The texture shown as base colour (and by the software fallback).
    fn base(&self) -> &BakedTexture {
        match self {
            Self::Material(textures) => &textures.base_color,
            Self::Single(texture) => texture,
        }
    }
}

/// Identifies the inputs mesh maps were baked from: the working mesh
/// revision and the high-poly source revision.
type BakeKey = (u64, u64);

/// Optional high-resolution source for high-to-low mesh-map baking.
struct HighPolySource {
    path: std::path::PathBuf,
    geometry: MeshGeometry,
}

struct MeshMapJob {
    receiver: Receiver<Result<Vec<MeshMaps>, MeshMapError>>,
    cancel: Arc<AtomicBool>,
    key: BakeKey,
}

struct PoseTransition {
    from: NamedPose,
    to: NamedPose,
    started_at: f64,
}

struct RestoredMeshScene {
    selected_primitive: Option<PrimitivePreset>,
    imported_meshes: Vec<ImportedMesh>,
    selected_imported_mesh: Option<usize>,
    geometry: Option<MeshGeometry>,
    scale: f32,
    smoothing: MeshSmoothing,
}

impl Default for SubstanceCraftApp {
    fn default() -> Self {
        let (filament_frames_tx, filament_frames_rx) = sync_channel(1);
        let mut app = Self {
            project: MaterialProject {
                mesh: Some(MeshAsset {
                    id: MeshId(1),
                    source: "builtin:Cube".to_owned(),
                    scale: 1.0,
                    smoothing: MeshSmoothing::default(),
                    high_poly_source: None,
                }),
                ..MaterialProject::empty()
            },
            selected_texture_set: None,
            selected_layer: None,
            selected_primitive: Some(PrimitivePreset::Cube),
            base_geometry: None,
            mesh_geometry: None,
            mesh_topology: None,
            mesh_center: glam::Vec3::ZERO,
            mesh_smoothing: MeshSmoothing::default(),
            mesh_error: None,
            mesh_revision: 1,
            filament_mesh_revision: None,
            filament_renderer: None,
            filament_error: None,
            filament_frame_texture: None,
            filament_frames_tx,
            filament_frames_rx,
            filament_render_pending: Arc::new(AtomicBool::new(false)),
            imported_meshes: Vec::new(),
            selected_imported_mesh: None,
            import_error: None,
            import_scale: 1.0,
            project_path: None,
            project_error: None,
            active_channel: MaterialChannel::BaseColor,
            next_texture_set_id: 1,
            next_layer_id: 1,
            current_pose: NamedPose {
                id: PoseId(0),
                name: "Current Pose".to_owned(),
                bones: BTreeMap::from([(BoneId(0), BoneTransform::IDENTITY)]),
            },
            viewport_camera: ViewportCamera::default(),
            viewport_lighting: ViewportLighting::default(),
            paint_mode: PaintMode::View,
            brush_radius: 16.0,
            brush_color: egui::Color32::from_rgb(214, 108, 66),
            brush_roughness: 0.5,
            brush_metallic: 0.0,
            brush_height: DEFAULT_HEIGHT,
            brush_emissive: egui::Color32::from_rgb(
                DEFAULT_EMISSIVE[0],
                DEFAULT_EMISSIVE[1],
                DEFAULT_EMISSIVE[2],
            ),
            brush_material_opacity: DEFAULT_OPACITY,
            brush_tip: BrushTip::ROUND,
            pen_pressure: None,
            resource_images: ResourceImages::new(),
            resource_thumbnails: BTreeMap::new(),
            export_format: TextureFormat::Png,
            view_mode: ViewMode::Model,
            uv_view: uv_view::UvView::default(),
            set_revisions: BTreeMap::new(),
            set_materials: BTreeMap::new(),
            set_bake_job: None,
            set_bake_failed: BTreeMap::new(),
            channel_preview: false,
            mask_value: 0.0,
            selection_mode: SelectionMode::Element,
            hovered_selection: Vec::new(),
            active_stroke: None,
            history: EditHistory::default(),
            layer_settled: None,
            baked_preview: None,
            preview_revision: 0,
            preview_job: None,
            preview_failed: false,
            bake_error: None,
            mesh_map_settings: MeshMapSettings::default(),
            mesh_maps: None,
            mesh_maps_key: None,
            high_poly: None,
            high_poly_revision: 0,
            mesh_map_job: None,
            mesh_map_error: None,
            mesh_map_preview: None,
            selected_pose: None,
            pose_name: "Pose 1".to_owned(),
            pose_transition: None,
            drawer: Some(panels::Drawer::Scene),
            last_drawer: panels::Drawer::Scene,
            show_inspector: true,
            layout: panels::OverlayLayout::default(),
        };
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app
    }
}

impl SubstanceCraftApp {
    fn mark_mesh_changed(&mut self) {
        self.mesh_revision = self.mesh_revision.wrapping_add(1);
        self.filament_frame_texture = None;
        self.cancel_mesh_map_job();
        self.mesh_maps = None;
        self.mesh_maps_key = None;
        self.mesh_map_error = None;
        self.hovered_selection.clear();
        self.invalidate_all_sets();
        // Old textures do not fit the new mesh's UVs.
        self.baked_preview = None;
        self.set_materials.clear();
        self.set_bake_job = None;
    }

    /// Marks the shown preview out of date. It stays visible until the
    /// background rebake replaces it. Edits made through the UI only touch
    /// the selected texture set.
    fn invalidate_preview(&mut self) {
        self.preview_revision = self.preview_revision.wrapping_add(1);
        self.preview_failed = false;
        if let Some(set_id) = self.selected_texture_set {
            self.set_revisions.insert(set_id, self.preview_revision);
        }
    }

    /// Marks every texture set out of date: the mesh, its maps, history, or
    /// shared resources changed.
    fn invalidate_all_sets(&mut self) {
        self.invalidate_preview();
        let revision = self.preview_revision;
        for set in &self.project.texture_sets {
            self.set_revisions.insert(set.id, revision);
        }
    }

    fn set_revision(&self, set_id: TextureSetId) -> u64 {
        self.set_revisions.get(&set_id).copied().unwrap_or(0)
    }

    fn set_base_geometry(&mut self, geometry: Option<MeshGeometry>) {
        self.base_geometry = geometry;
        self.rebuild_mesh();
    }

    /// Re-derives display geometry from the base mesh and smoothing settings.
    fn rebuild_mesh(&mut self) {
        self.mesh_error = None;
        self.mesh_geometry = self.base_geometry.as_ref().map(|base| {
            Arc::new(self.mesh_smoothing.apply(base).unwrap_or_else(|error| {
                self.mesh_error = Some(format!("Smoothing skipped: {error}"));
                base.clone()
            }))
        });
        self.mesh_topology = self.mesh_geometry.as_deref().map(MeshTopology::build);
        self.mesh_center = self
            .mesh_geometry
            .as_deref()
            .map_or(glam::Vec3::ZERO, mesh_bounds_center);
        self.mark_mesh_changed();
    }

    fn set_mesh_smoothing(&mut self, smoothing: MeshSmoothing) {
        if smoothing == self.mesh_smoothing || !smoothing.is_valid() {
            return;
        }
        let from = self.mesh_smoothing.subdivision_levels;
        let to = smoothing.subdivision_levels;
        if from != to {
            // Polygon masks index subdivided triangles; carry them across.
            for set in &mut self.project.texture_sets {
                for_each_layer_mut(&mut set.layers, &mut |layer| {
                    if let Some(mask) = layer.polygon_mask.as_mut() {
                        mask.triangles =
                            remap_subdivided_triangles(mask.triangles.iter().copied(), from, to)
                                .into_iter()
                                .collect();
                    }
                });
            }
            self.layer_settled = None;
        }
        self.mesh_smoothing = smoothing;
        if let Some(mesh) = self.project.mesh.as_mut() {
            mesh.smoothing = smoothing;
        }
        self.rebuild_mesh();
    }

    fn update_filament_frame(&mut self, context: &egui::Context, viewport: egui::Rect) {
        if let Some(renderer) = self.filament_renderer.as_mut() {
            renderer.poll();
        }
        while let Ok(frame) = self.filament_frames_rx.try_recv() {
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [frame.width as usize, frame.height as usize],
                &frame.rgba8,
            );
            if let Some(texture) = self.filament_frame_texture.as_mut() {
                texture.set(image, egui::TextureOptions::LINEAR);
            } else {
                self.filament_frame_texture = Some(context.load_texture(
                    "filament-viewport",
                    image,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }

        let wanted = self.preview_key();
        let target = wanted.target;
        if self.mesh_geometry.is_none() {
            self.filament_frame_texture = None;
            return;
        }
        let plan = self.slot_upload_plan(target);
        let Some(mesh) = self.mesh_geometry.as_deref() else {
            return;
        };
        let Some(renderer) = self.filament_renderer.as_mut() else {
            return;
        };
        if self.filament_mesh_revision != Some(self.mesh_revision) {
            match renderer.set_mesh(mesh, self.mesh_smoothing.angle_degrees) {
                Ok(()) => self.filament_mesh_revision = Some(self.mesh_revision),
                Err(error) => {
                    self.filament_error = Some(error.to_string());
                    return;
                }
            }
        }

        let upload = |renderer: &mut FilamentRuntime,
                      slot: usize,
                      map: MaterialMap,
                      texture: &BakedTexture| {
            let key = (texture.rgba8.as_ptr() as usize)
                ^ (texture.width as usize).rotate_left(7)
                ^ (texture.height as usize).rotate_left(17);
            renderer.set_map(
                slot,
                map,
                key,
                texture.width,
                texture.height,
                &texture.rgba8,
            )
        };
        // Neutral fallbacks: grey dielectric, AO 1 / roughness 0.58, flat
        // normal, no emission.
        let neutral = |renderer: &mut FilamentRuntime, slot: usize| {
            renderer
                .set_map(slot, MaterialMap::BaseColor, 0, 1, 1, &[168, 173, 179, 255])
                .and_then(|()| {
                    renderer.set_map(slot, MaterialMap::Orm, 0, 1, 1, &[255, 148, 0, 255])
                })
                .and_then(|()| {
                    renderer.set_map(slot, MaterialMap::Normal, 0, 1, 1, &[128, 128, 255, 255])
                })
                .and_then(|()| {
                    renderer.set_map(slot, MaterialMap::Emissive, 0, 1, 1, &[0, 0, 0, 255])
                })
        };
        renderer.set_preview_mode(target != PreviewTarget::Material);
        let mut texture_result = Ok(());
        for (slot, source) in plan.into_iter().enumerate() {
            let result = match source {
                SlotSource::Material(textures) => {
                    upload(renderer, slot, MaterialMap::BaseColor, &textures.base_color)
                        .and_then(|()| upload(renderer, slot, MaterialMap::Orm, &textures.orm))
                        .and_then(|()| {
                            upload(renderer, slot, MaterialMap::Normal, &textures.normal)
                        })
                        .and_then(|()| {
                            upload(renderer, slot, MaterialMap::Emissive, &textures.emissive)
                        })
                }
                SlotSource::Single(texture) => {
                    upload(renderer, slot, MaterialMap::BaseColor, &texture)
                }
                SlotSource::Neutral => neutral(renderer, slot),
            };
            if result.is_err() {
                texture_result = result;
                break;
            }
        }
        if let Err(error) = texture_result {
            self.filament_error = Some(error.to_string());
            return;
        }

        if self
            .filament_render_pending
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            context.request_repaint_after(Duration::from_millis(16));
            return;
        }
        let scale = context.pixels_per_point();
        let width = (viewport.width() * scale).ceil().clamp(1.0, 2048.0) as u32;
        let height = (viewport.height() * scale).ceil().clamp(1.0, 2048.0) as u32;
        let sender = self.filament_frames_tx.clone();
        let pending = Arc::clone(&self.filament_render_pending);
        match renderer.render(self.viewport_camera, width, height, scale, move |rgba8| {
            pending.store(false, Ordering::Release);
            let _ = sender.try_send(FilamentFrame {
                width,
                height,
                rgba8,
            });
        }) {
            Ok(true) => {
                self.filament_error = None;
                context.request_repaint_after(Duration::from_millis(16));
            }
            Ok(false) => {
                self.filament_render_pending.store(false, Ordering::Release);
                context.request_repaint_after(Duration::from_millis(16));
            }
            Err(error) => {
                self.filament_render_pending.store(false, Ordering::Release);
                self.filament_error = Some(error.to_string());
            }
        }
    }

    fn save_project(&mut self) {
        let path = self.project_path.clone().or_else(|| {
            rfd::FileDialog::new()
                .add_filter("SubstanceCraft Project", &["sproject"])
                .set_file_name("Untitled.sproject")
                .save_file()
        });
        let Some(path) = path else {
            return;
        };

        match save_project_file(&path, &self.project) {
            Ok(()) => {
                self.project_path = Some(path);
                self.project_error = None;
            }
            Err(error) => self.project_error = Some(error.to_string()),
        }
    }

    fn open_project(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("SubstanceCraft Project", &["sproject"])
            .pick_file()
        else {
            return;
        };
        let project = match load_project_file(&path) {
            Ok(project) => project,
            Err(error) => {
                self.project_error = Some(error.to_string());
                return;
            }
        };
        let scene = match Self::restore_project_mesh(&project) {
            Ok(scene) => scene,
            Err(error) => {
                self.project_error = Some(error);
                return;
            }
        };

        self.project = project;
        self.project_path = Some(path);
        self.selected_primitive = scene.selected_primitive;
        self.imported_meshes = scene.imported_meshes;
        self.selected_imported_mesh = scene.selected_imported_mesh;
        self.import_scale = scene.scale;
        self.mesh_smoothing = scene.smoothing;
        self.set_base_geometry(scene.geometry);
        self.restore_high_poly();
        self.reload_resources();
        self.selected_texture_set = self.project.active_texture_set;
        self.selected_layer = self
            .project
            .texture_sets
            .iter()
            .find(|set| Some(set.id) == self.selected_texture_set)
            .and_then(|set| set.layers.last())
            .map(|layer| layer.id);
        self.next_texture_set_id = self
            .project
            .texture_sets
            .iter()
            .map(|set| set.id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.next_layer_id = self
            .project
            .texture_sets
            .iter()
            .flat_map(|set| set.walk_layers().into_iter().map(|layer| layer.id.0))
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.current_pose = self
            .project
            .saved_poses
            .first()
            .cloned()
            .unwrap_or_else(|| NamedPose {
                id: PoseId(0),
                name: "Current Pose".to_owned(),
                bones: BTreeMap::from([(BoneId(0), BoneTransform::IDENTITY)]),
            });
        self.viewport_camera = ViewportCamera::default();
        self.viewport_lighting = ViewportLighting::default();
        self.selected_pose = self.project.saved_poses.first().map(|pose| pose.id);
        self.pose_name = format!("Pose {}", self.project.saved_poses.len() + 1);
        self.pose_transition = None;
        self.active_stroke = None;
        self.history = EditHistory::default();
        self.layer_settled = None;
        self.invalidate_preview();
        self.bake_error = None;
        self.import_error = None;
        self.project_error = None;
    }

    fn restore_project_mesh(project: &MaterialProject) -> Result<RestoredMeshScene, String> {
        let Some(mesh_asset) = &project.mesh else {
            return Ok(RestoredMeshScene {
                selected_primitive: None,
                imported_meshes: Vec::new(),
                selected_imported_mesh: None,
                geometry: None,
                scale: 1.0,
                smoothing: MeshSmoothing::default(),
            });
        };
        let smoothing = if mesh_asset.smoothing.is_valid() {
            mesh_asset.smoothing
        } else {
            MeshSmoothing::default()
        };
        if let Some(name) = mesh_asset.source.strip_prefix("builtin:") {
            let preset = PrimitivePreset::ALL
                .into_iter()
                .find(|preset| preset.name() == name)
                .ok_or_else(|| format!("unknown built-in mesh `{name}` in project"))?;
            return Ok(RestoredMeshScene {
                selected_primitive: Some(preset),
                imported_meshes: Vec::new(),
                selected_imported_mesh: None,
                geometry: Some(preset.build_mesh()),
                scale: 1.0,
                smoothing,
            });
        }

        let path = std::path::PathBuf::from(&mesh_asset.source);
        let meshes = load_mesh_file(&path)?;
        let Some(first_mesh) = meshes.first() else {
            return Err("referenced mesh file contains no mesh geometry".to_owned());
        };
        let scale = sanitize_mesh_scale(mesh_asset.scale);
        let first_geometry = scaled_geometry(&first_mesh.geometry, scale);
        Ok(RestoredMeshScene {
            selected_primitive: None,
            imported_meshes: meshes,
            selected_imported_mesh: Some(0),
            geometry: Some(first_geometry),
            scale,
            smoothing,
        })
    }

    fn import_mesh(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D mesh (FBX, glTF)", &MESH_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        self.load_imported_mesh(&path);
    }

    /// Replaces the project mesh with the file at `path`. A mesh with several
    /// material slots gets one texture set per slot; otherwise texture sets
    /// are created on first use.
    fn load_imported_mesh(&mut self, path: &std::path::Path) {
        let meshes = match load_mesh_file(path) {
            Ok(meshes) => meshes,
            Err(error) => {
                self.import_error = Some(error);
                return;
            }
        };
        let Some(first_mesh) = meshes.first() else {
            self.import_error = Some("the file contains no mesh geometry".to_owned());
            return;
        };
        let first_geometry = scaled_geometry(&first_mesh.geometry, self.import_scale);
        let display_name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Imported mesh".to_owned());
        self.project.mesh = Some(MeshAsset {
            id: MeshId(1),
            source: path.to_string_lossy().into_owned(),
            scale: self.import_scale,
            smoothing: self.mesh_smoothing,
            high_poly_source: None,
        });
        let slots = if first_geometry.material_slot_count() > 1 {
            first_geometry.material_slots.clone()
        } else {
            Vec::new()
        };
        self.imported_meshes = meshes;
        self.selected_imported_mesh = Some(0);
        self.selected_primitive = None;
        self.set_base_geometry(Some(first_geometry));
        // A new low mesh needs its own high-poly source.
        self.high_poly = None;
        self.high_poly_changed();
        self.import_error = None;
        self.project.texture_sets.clear();
        self.project.active_texture_set = None;
        self.selected_texture_set = None;
        self.selected_layer = None;
        self.active_stroke = None;
        self.history = EditHistory::default();
        self.layer_settled = None;
        self.set_materials.clear();
        self.current_pose.name = display_name;
        for slot in &slots {
            self.add_texture_set_for_slot(slot.clone());
        }
        if let Some(first) = self.project.texture_sets.first().map(|set| set.id) {
            self.project.active_texture_set = Some(first);
            self.selected_texture_set = Some(first);
        }
    }

    fn add_texture_set(&mut self) {
        let name = format!("Material {}", self.next_texture_set_id);
        self.add_texture_set_for_slot(name);
    }

    /// Adds and selects a texture set painting mesh material slot `slot`.
    fn add_texture_set_for_slot(&mut self, slot: String) {
        let Some(next_id) = self.next_texture_set_id.checked_add(1) else {
            return;
        };
        let id = TextureSetId(self.next_texture_set_id);
        self.next_texture_set_id = next_id;
        let name = if slot.starts_with("Material ") {
            format!("Texture Set {}", id.0)
        } else {
            slot.clone()
        };
        self.project.texture_sets.push(TextureSet {
            id,
            name,
            material_slot: slot,
            resolution: TextureResolution {
                width: 2048,
                height: 2048,
            },
            channels: BAKEABLE_CHANNELS.into_iter().collect::<BTreeSet<_>>(),
            layers: Vec::new(),
            height_strength: DEFAULT_HEIGHT_STRENGTH,
        });
        self.project.active_texture_set = Some(id);
        self.selected_texture_set = Some(id);
        self.selected_layer = None;
        self.invalidate_preview();
    }

    fn allocate_layer_id(&mut self) -> Option<LayerId> {
        let next = self.next_layer_id.checked_add(1)?;
        let id = LayerId(self.next_layer_id);
        self.next_layer_id = next;
        Some(id)
    }

    /// Where a new layer goes: inside the selected group (on top), else just
    /// above the selected layer, else on top of the stack.
    fn insertion_slot(&self, set: &TextureSet) -> LayerSlot {
        let top = LayerSlot {
            parent: None,
            index: set.layers.len(),
        };
        let Some(selected) = self.selected_layer else {
            return top;
        };
        if let Some(children) = set
            .layer(selected)
            .and_then(|layer| layer.children.as_ref())
        {
            return LayerSlot {
                parent: Some(selected),
                index: children.len(),
            };
        }
        set.slot_of(selected).map_or(top, |slot| LayerSlot {
            parent: slot.parent,
            index: slot.index + 1,
        })
    }

    /// Inserts `layer` (see [`Self::insertion_slot`]) as one undoable edit,
    /// creating a texture set first when none exists.
    fn insert_layer(&mut self, layer: MaterialLayer) {
        if self.selected_texture_set.is_none() {
            self.add_texture_set();
        }
        let Some(texture_set_id) = self.selected_texture_set else {
            return;
        };
        let Some(slot) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)
            .map(|set| self.insertion_slot(set))
        else {
            return;
        };
        let layer_id = layer.id;
        if self.history.apply(
            &mut self.project,
            EditCommand::InsertLayer {
                texture_set_id,
                slot,
                layer: Box::new(layer),
            },
        ) {
            self.selected_layer = Some(layer_id);
            self.layer_settled = None;
            self.invalidate_preview();
        }
    }

    fn add_paint_layer(&mut self) {
        let Some(id) = self.allocate_layer_id() else {
            return;
        };
        self.insert_layer(MaterialLayer::new_paint(
            id,
            format!("Paint Layer {}", id.0),
        ));
    }

    fn add_fill_layer(&mut self) {
        let Some(id) = self.allocate_layer_id() else {
            return;
        };
        let emissive = self.brush_emissive.to_array();
        let fill = LayerFill {
            color: self.brush_color.to_array(),
            roughness: self.brush_roughness,
            metallic: self.brush_metallic,
            height: self.brush_height,
            emissive: [emissive[0], emissive[1], emissive[2]],
            material_opacity: self.brush_material_opacity,
            ..LayerFill::default()
        };
        self.insert_layer(MaterialLayer::new_fill(
            id,
            format!("Fill Layer {}", id.0),
            fill,
        ));
    }

    fn add_group_layer(&mut self) {
        let Some(id) = self.allocate_layer_id() else {
            return;
        };
        self.insert_layer(MaterialLayer::new_group(id, format!("Group {}", id.0)));
    }

    /// Adds a fill layer pre-wired to a procedural mask generator.
    fn add_smart_layer(&mut self, generator: MaskGenerator) {
        let Some(id) = self.allocate_layer_id() else {
            return;
        };
        let (name, fill) = match generator {
            MaskGenerator::EdgeWear(_) => (
                "Edge Wear",
                LayerFill {
                    color: [196, 198, 204, 255],
                    roughness: 0.28,
                    metallic: 1.0,
                    // Chipped paint sits slightly below the coat around it.
                    height: 0.42,
                    ..LayerFill::default()
                },
            ),
            MaskGenerator::Dirt(_) => (
                "Dirt",
                LayerFill {
                    color: [62, 50, 38, 255],
                    roughness: 0.92,
                    metallic: 0.0,
                    height: DEFAULT_HEIGHT,
                    ..LayerFill::default()
                },
            ),
            _ => ("Generated", LayerFill::default()),
        };
        let mut layer = MaterialLayer::new_fill(id, name, fill);
        layer.mask_generator = Some(generator);
        self.insert_layer(layer);
    }

    fn add_edge_wear_layer(&mut self) {
        self.add_smart_layer(MaskGenerator::EdgeWear(EdgeWear::DEFAULT));
    }

    fn add_dirt_layer(&mut self) {
        self.add_smart_layer(MaskGenerator::Dirt(Dirt::DEFAULT));
    }

    /// The selected layer's texture set, slot, and container length.
    fn selected_layer_slot(&self) -> Option<(TextureSetId, LayerSlot, usize)> {
        let set = self.selected_texture_set_ref()?;
        let slot = set.slot_of(self.selected_layer?)?;
        let count = set.container(slot.parent)?.len();
        Some((set.id, slot, count))
    }

    fn delete_selected_layer(&mut self) {
        self.finish_active_stroke();
        let Some((texture_set_id, slot, _)) = self.selected_layer_slot() else {
            return;
        };
        let Some(layer) = self
            .selected_texture_set_ref()
            .and_then(|set| set.container(slot.parent))
            .and_then(|layers| layers.get(slot.index))
            .cloned()
        else {
            return;
        };
        if self.history.apply(
            &mut self.project,
            EditCommand::RemoveLayer {
                texture_set_id,
                slot,
                layer: Box::new(layer),
            },
        ) {
            self.selected_layer = self
                .selected_texture_set_ref()
                .and_then(|set| set.container(slot.parent))
                .and_then(|layers| layers.get(slot.index.saturating_sub(1)))
                .map(|layer| layer.id)
                .or(slot.parent);
            self.layer_settled = None;
            self.invalidate_preview();
        }
    }

    fn move_selected_layer_to(&mut self, to: LayerSlot) -> bool {
        let Some((texture_set_id, from, _)) = self.selected_layer_slot() else {
            return false;
        };
        let moved = self.history.apply(
            &mut self.project,
            EditCommand::MoveLayer {
                texture_set_id,
                from,
                to,
            },
        );
        if moved {
            self.invalidate_preview();
        }
        moved
    }

    /// Moves the selected layer up (+1, toward the top of its group) or down.
    fn move_selected_layer(&mut self, delta: isize) {
        let Some((_, slot, count)) = self.selected_layer_slot() else {
            return;
        };
        if let Some(index) = slot
            .index
            .checked_add_signed(delta)
            .filter(|index| *index < count)
        {
            self.move_selected_layer_to(LayerSlot {
                parent: slot.parent,
                index,
            });
        }
    }

    /// The group directly below the selected layer, which it can move into.
    fn group_below_selected(&self) -> Option<LayerId> {
        let (_, slot, _) = self.selected_layer_slot()?;
        let below = self
            .selected_texture_set_ref()?
            .container(slot.parent)?
            .get(slot.index.checked_sub(1)?)?;
        below.is_group().then_some(below.id)
    }

    /// Moves the selected layer onto the top of the group below it.
    fn move_selected_into_group(&mut self) {
        let Some(group) = self.group_below_selected() else {
            return;
        };
        let Some(count) = self
            .selected_texture_set_ref()
            .and_then(|set| set.container(Some(group)))
            .map(Vec::len)
        else {
            return;
        };
        self.move_selected_layer_to(LayerSlot {
            parent: Some(group),
            index: count,
        });
    }

    /// Moves the selected layer out of its group, just above the group.
    fn move_selected_out_of_group(&mut self) {
        let Some((_, slot, _)) = self.selected_layer_slot() else {
            return;
        };
        let Some(group_slot) = slot
            .parent
            .and_then(|group| self.selected_texture_set_ref()?.slot_of(group))
        else {
            return;
        };
        self.move_selected_layer_to(LayerSlot {
            parent: group_slot.parent,
            index: group_slot.index + 1,
        });
    }

    fn selected_layer_mut(&mut self) -> Option<&mut MaterialLayer> {
        let texture_set_id = self.selected_texture_set?;
        let layer_id = self.selected_layer?;
        self.project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)?
            .layer_mut(layer_id)
    }

    fn selected_layer(&self) -> Option<&MaterialLayer> {
        self.selected_texture_set_ref()?.layer(self.selected_layer?)
    }

    /// Records inspector edits of the selected layer as single history steps.
    /// Changes accumulate while the pointer is held (slider drags) and are
    /// committed on release.
    fn sync_layer_history(&mut self, context: &egui::Context) {
        let current =
            self.selected_texture_set
                .zip(self.selected_layer)
                .and_then(|(set_id, layer_id)| {
                    self.selected_layer()
                        .map(|layer| (set_id, layer_id, layer.settings()))
                });
        let Some((set_id, layer_id, settings)) = current else {
            self.layer_settled = None;
            return;
        };
        let same_layer =
            self.layer_settled
                .as_ref()
                .is_some_and(|(settled_set, settled_layer, _)| {
                    *settled_set == set_id && *settled_layer == layer_id
                });
        if !same_layer {
            self.layer_settled = Some((set_id, layer_id, settings));
            return;
        }
        if context.input(|input| input.pointer.any_down()) {
            return;
        }
        if let Some((_, _, before)) = self.layer_settled.take() {
            if before != settings {
                self.history.record(EditCommand::LayerSettings {
                    texture_set_id: set_id,
                    layer_id,
                    before: Box::new(before),
                    after: Box::new(settings.clone()),
                });
                self.invalidate_preview();
            }
            self.layer_settled = Some((set_id, layer_id, settings));
        }
    }

    /// Adds (or with `remove`, subtracts) triangles from the selected layer's
    /// polygon mask.
    fn apply_polygon_selection(&mut self, triangles: &[u32], remove: bool) {
        let Some(layer) = self.selected_layer_mut() else {
            return;
        };
        let mask = layer.polygon_mask.get_or_insert_with(Default::default);
        let selecting = remove == mask.inverted;
        for triangle in triangles {
            if selecting {
                mask.triangles.insert(*triangle);
            } else {
                mask.triangles.remove(triangle);
            }
        }
        self.invalidate_preview();
    }

    /// Adds `uv` to the stroke in progress, starting one when needed. Paint
    /// and ID strokes skip groups; mask strokes work on any layer.
    fn append_stroke_point(&mut self, uv: glam::Vec2) {
        let (Some(texture_set_id), Some(layer_id)) =
            (self.selected_texture_set, self.selected_layer)
        else {
            return;
        };
        if !self.paint_mode.paints() {
            return;
        }
        let mode = self.paint_mode;
        let stroke_key = (texture_set_id, layer_id, mode);
        let starts_stroke = self.active_stroke != Some(stroke_key);
        let pressure = self.pen_pressure;
        let color = self.brush_color.to_array();
        let emissive = self.brush_emissive.to_array();
        let tip = self.brush_tip;
        let radius_pixels = self.brush_radius;
        let (roughness, metallic, height, material_opacity) = (
            self.brush_roughness,
            self.brush_metallic,
            self.brush_height,
            self.brush_material_opacity,
        );
        let mask_value = self.mask_value;
        let Some(layer) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)
            .and_then(|set| set.layer_mut(layer_id))
        else {
            return;
        };
        let paints_content = matches!(mode, PaintMode::Paint | PaintMode::Erase | PaintMode::Id);
        if paints_content && layer.is_group() {
            return;
        }
        // Pressure data is kept only for pen strokes.
        let pressures =
            |pressure: Option<f32>| pressure.map(|value| vec![value]).unwrap_or_default();

        if starts_stroke {
            match mode {
                PaintMode::View | PaintMode::Select => return,
                PaintMode::Paint | PaintMode::Erase => layer.paint_strokes.push(PaintStroke {
                    roughness,
                    metallic,
                    height,
                    emissive: [emissive[0], emissive[1], emissive[2]],
                    material_opacity,
                    erase: mode == PaintMode::Erase,
                    tip,
                    pressures: pressures(pressure),
                    ..PaintStroke::new(vec![uv], radius_pixels, color)
                }),
                PaintMode::Mask => layer.mask_strokes.push(MaskStroke {
                    points: vec![uv],
                    radius_pixels,
                    value: mask_value,
                    opacity: 1.0,
                    tip,
                    pressures: pressures(pressure),
                }),
                PaintMode::Id => layer.id_strokes.push(IdStroke {
                    points: vec![uv],
                    radius_pixels,
                    color: [color[0], color[1], color[2]],
                    opacity: 1.0,
                    tip,
                    pressures: pressures(pressure),
                }),
            }
            self.active_stroke = Some(stroke_key);
            self.invalidate_preview();
            return;
        }

        let path = match mode {
            PaintMode::View | PaintMode::Select => return,
            PaintMode::Paint | PaintMode::Erase => layer
                .paint_strokes
                .last_mut()
                .map(|stroke| (&mut stroke.points, &mut stroke.pressures)),
            PaintMode::Mask => layer
                .mask_strokes
                .last_mut()
                .map(|stroke| (&mut stroke.points, &mut stroke.pressures)),
            PaintMode::Id => layer
                .id_strokes
                .last_mut()
                .map(|stroke| (&mut stroke.points, &mut stroke.pressures)),
        };
        if let Some((points, pressures)) = path
            && points.len() < MAX_STROKE_POINTS
            && points
                .last()
                .is_none_or(|last| last.distance_squared(uv) >= 0.000_001)
        {
            points.push(uv);
            // A pen stroke keeps one pressure per point.
            if !pressures.is_empty() {
                pressures.push(pressure.unwrap_or(1.0));
            }
            self.preview_revision = self.preview_revision.wrapping_add(1);
            if let Some(set_id) = self.selected_texture_set {
                self.set_revisions.insert(set_id, self.preview_revision);
            }
        }
    }

    fn finish_active_stroke(&mut self) {
        let Some((texture_set_id, layer_id, mode)) = self.active_stroke.take() else {
            return;
        };
        let Some(layer) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)
            .and_then(|set| set.layer(layer_id))
        else {
            return;
        };
        let command = match mode {
            PaintMode::View | PaintMode::Select => None,
            PaintMode::Paint | PaintMode::Erase => {
                layer
                    .paint_strokes
                    .last()
                    .cloned()
                    .map(|stroke| EditCommand::Paint {
                        texture_set_id,
                        layer_id,
                        stroke,
                    })
            }
            PaintMode::Mask => layer
                .mask_strokes
                .last()
                .cloned()
                .map(|stroke| EditCommand::Mask {
                    texture_set_id,
                    layer_id,
                    stroke,
                }),
            PaintMode::Id => layer
                .id_strokes
                .last()
                .cloned()
                .map(|stroke| EditCommand::Id {
                    texture_set_id,
                    layer_id,
                    stroke,
                }),
        };
        if let Some(command) = command {
            self.history.record(command);
        }
    }

    fn undo(&mut self) {
        self.finish_active_stroke();
        if self.history.undo(&mut self.project) {
            self.after_history_change();
        }
    }

    fn redo(&mut self) {
        self.finish_active_stroke();
        if self.history.redo(&mut self.project) {
            self.after_history_change();
        }
    }

    fn after_history_change(&mut self) {
        // The settled snapshot must not see undo/redo as a new user edit.
        self.layer_settled = None;
        let selected_exists = self.selected_layer().is_some();
        if !selected_exists {
            self.selected_layer = self
                .project
                .texture_sets
                .iter()
                .find(|set| Some(set.id) == self.selected_texture_set)
                .and_then(|set| set.layers.last())
                .map(|layer| layer.id);
        }
        self.invalidate_all_sets();
        self.refresh_baked_preview();
    }

    fn preview_target(&self) -> PreviewTarget {
        if let Some(kind) = self.mesh_map_preview {
            PreviewTarget::MeshMap(kind)
        } else if self.channel_preview {
            PreviewTarget::Channel(self.active_channel)
        } else {
            PreviewTarget::Material
        }
    }

    fn bake_key(&self) -> BakeKey {
        (self.mesh_revision, self.high_poly_revision)
    }

    fn mesh_maps_fresh(&self) -> bool {
        self.mesh_maps.is_some() && self.mesh_maps_key == Some(self.bake_key())
    }

    /// Mesh material slot of the selected texture set.
    fn selected_slot(&self) -> u32 {
        self.selected_texture_set
            .map_or(0, |set_id| self.set_slot(set_id))
    }

    /// Fresh mesh maps of the selected texture set's material slot.
    fn fresh_mesh_maps(&self) -> Option<&MeshMaps> {
        self.fresh_slot_maps(self.selected_slot())
    }

    fn fresh_slot_maps(&self, slot: u32) -> Option<&MeshMaps> {
        self.mesh_maps
            .as_deref()
            .filter(|_| self.mesh_maps_fresh())
            .and_then(|maps| maps.get(slot as usize))
    }

    fn mesh_map_job_running(&self) -> bool {
        self.mesh_map_job
            .as_ref()
            .is_some_and(|job| job.key == self.bake_key())
    }

    fn effective_mesh_map_settings(&self) -> MeshMapSettings {
        MeshMapSettings {
            smoothing_angle_degrees: self.mesh_smoothing.angle_degrees,
            ..self.mesh_map_settings
        }
    }

    fn cancel_mesh_map_job(&mut self) {
        if let Some(job) = self.mesh_map_job.take() {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Starts a background bake of the current mesh's maps (high-to-low when
    /// a high-poly source is loaded), replacing any bake already running.
    fn start_mesh_map_bake(&mut self) {
        self.cancel_mesh_map_job();
        let Some(mesh) = self.mesh_geometry.clone() else {
            return;
        };
        let high = self
            .high_poly
            .as_ref()
            .map(|source| source.geometry.clone());
        let settings = self.effective_mesh_map_settings();
        let cancel = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = sync_channel(1);
        let worker_cancel = Arc::clone(&cancel);
        let spawned = std::thread::Builder::new()
            .name("mesh-map-bake".to_owned())
            .spawn(move || {
                let _ = sender.send(bake_slot_mesh_maps(
                    &mesh,
                    high.as_ref(),
                    &settings,
                    &worker_cancel,
                ));
            });
        match spawned {
            Ok(_) => {
                self.mesh_map_error = None;
                self.mesh_map_job = Some(MeshMapJob {
                    receiver,
                    cancel,
                    key: self.bake_key(),
                });
            }
            Err(error) => self.mesh_map_error = Some(error.to_string()),
        }
    }

    /// Bakes mesh maps automatically when something needs them and they are
    /// missing or stale for the current mesh and high-poly source.
    fn ensure_mesh_maps(&mut self) {
        if self.mesh_maps_fresh() || self.mesh_map_job_running() || self.mesh_map_error.is_some() {
            return;
        }
        self.start_mesh_map_bake();
    }

    fn poll_mesh_map_job(&mut self, context: &egui::Context) {
        let Some(job) = &self.mesh_map_job else {
            return;
        };
        match job.receiver.try_recv() {
            Ok(result) => {
                let key = job.key;
                self.mesh_map_job = None;
                match result {
                    Ok(maps) if key == self.bake_key() => {
                        self.mesh_maps = Some(Arc::new(maps));
                        self.mesh_maps_key = Some(key);
                        self.invalidate_all_sets();
                    }
                    Ok(_) | Err(MeshMapError::Cancelled) => {}
                    Err(error) => self.mesh_map_error = Some(error.to_string()),
                }
            }
            Err(TryRecvError::Empty) => context.request_repaint_after(Duration::from_millis(100)),
            Err(TryRecvError::Disconnected) => {
                self.mesh_map_job = None;
                self.mesh_map_error = Some("mesh map bake stopped unexpectedly".to_owned());
            }
        }
    }

    /// Mesh maps become stale whenever the high-poly source changes.
    fn high_poly_changed(&mut self) {
        self.high_poly_revision = self.high_poly_revision.wrapping_add(1);
        self.cancel_mesh_map_job();
        self.mesh_map_error = None;
        self.invalidate_all_sets();
    }

    fn pick_high_poly(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D mesh (FBX, glTF)", &MESH_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        match load_merged_mesh(&path, self.import_scale) {
            Ok(geometry) => {
                if let Some(mesh) = self.project.mesh.as_mut() {
                    mesh.high_poly_source = Some(path.to_string_lossy().into_owned());
                }
                self.high_poly = Some(HighPolySource { path, geometry });
                self.high_poly_changed();
                self.start_mesh_map_bake();
            }
            Err(error) => self.mesh_map_error = Some(format!("High poly: {error}")),
        }
    }

    fn clear_high_poly(&mut self) {
        self.high_poly = None;
        if let Some(mesh) = self.project.mesh.as_mut() {
            mesh.high_poly_source = None;
        }
        self.high_poly_changed();
    }

    /// Reloads the project's high-poly reference, reporting (not failing on)
    /// a missing or unreadable file.
    fn restore_high_poly(&mut self) {
        self.high_poly = None;
        let source = self
            .project
            .mesh
            .as_ref()
            .and_then(|mesh| mesh.high_poly_source.clone());
        if let Some(source) = source {
            let path = std::path::PathBuf::from(source);
            match load_merged_mesh(&path, self.import_scale) {
                Ok(geometry) => self.high_poly = Some(HighPolySource { path, geometry }),
                Err(error) => {
                    self.mesh_map_error = Some(format!("High poly `{}`: {error}", path.display()));
                }
            }
        }
        self.high_poly_changed();
    }

    /// Texture set painted on mesh material slot `slot`. A single-slot mesh
    /// shows the selected set; otherwise slots match sets by material name.
    fn slot_texture_set(&self, slot: u32) -> Option<TextureSetId> {
        let mesh = self.mesh_geometry.as_deref()?;
        if mesh.material_slot_count() <= 1 {
            return self.selected_texture_set;
        }
        let name = mesh.material_slot_name(slot);
        self.project
            .texture_sets
            .iter()
            .find(|set| set.material_slot == name)
            .map(|set| set.id)
    }

    /// Mesh material slot a texture set paints; 0 when its name matches none.
    fn set_slot(&self, set_id: TextureSetId) -> u32 {
        let Some(mesh) = self.mesh_geometry.as_deref() else {
            return 0;
        };
        self.project
            .texture_sets
            .iter()
            .find(|set| set.id == set_id)
            .and_then(|set| {
                mesh.material_slots
                    .iter()
                    .position(|slot| *slot == set.material_slot)
            })
            .map_or(0, |slot| slot as u32)
    }

    /// True when triangle `triangle` belongs to the selected texture set's
    /// material slot, so strokes there paint that set.
    fn triangle_in_selected_set(&self, triangle: u32) -> bool {
        let (Some(mesh), Some(set_id)) = (self.mesh_geometry.as_deref(), self.selected_texture_set)
        else {
            return false;
        };
        mesh.material_slot_count() <= 1
            || mesh.material_of(triangle as usize) == self.set_slot(set_id)
    }

    /// Textures each mesh material slot should show for `target`.
    fn slot_upload_plan(&self, target: PreviewTarget) -> Vec<SlotSource> {
        let slot_count = self
            .mesh_geometry
            .as_deref()
            .map_or(1, MeshGeometry::material_slot_count);
        (0..slot_count as u32)
            .map(|slot| {
                let set_id = self.slot_texture_set(slot);
                let selected = set_id.is_some() && set_id == self.selected_texture_set;
                let mesh_map_preview = matches!(target, PreviewTarget::MeshMap(_));
                if selected || (mesh_map_preview && self.set_slot_matches_preview(slot)) {
                    return match self.displayed_preview() {
                        Some(PreviewTextures::Material(textures)) => {
                            SlotSource::Material(Arc::clone(textures))
                        }
                        Some(PreviewTextures::Single(texture)) => {
                            SlotSource::Single(Arc::clone(texture))
                        }
                        None => SlotSource::Neutral,
                    };
                }
                match (target, set_id.and_then(|id| self.set_materials.get(&id))) {
                    (PreviewTarget::Material, Some(cached)) => {
                        SlotSource::Material(Arc::clone(&cached.textures))
                    }
                    _ => SlotSource::Neutral,
                }
            })
            .collect()
    }

    /// Mesh-map previews show on the selected set's slot (or the whole mesh
    /// when no set is selected).
    fn set_slot_matches_preview(&self, slot: u32) -> bool {
        self.selected_texture_set
            .map_or(slot == 0, |set_id| self.set_slot(set_id) == slot)
    }

    fn selected_texture_set_ref(&self) -> Option<&TextureSet> {
        self.project
            .texture_sets
            .iter()
            .find(|set| Some(set.id) == self.selected_texture_set)
    }

    /// Identity of the preview the viewport should show right now.
    fn preview_key(&self) -> PreviewKey {
        let target = self.preview_target();
        PreviewKey {
            texture_set: match target {
                PreviewTarget::MeshMap(_) => None,
                _ => self.selected_texture_set,
            },
            target,
            revision: self.preview_revision,
        }
    }

    /// Starts (at most one) background bake of the current preview. The last
    /// finished preview stays on screen until its replacement arrives, so
    /// painting updates live without flicker.
    fn refresh_baked_preview(&mut self) {
        let key = self.preview_key();
        if self.preview_failed
            || self.preview_job.is_some()
            || self
                .baked_preview
                .as_ref()
                .is_some_and(|preview| preview.key == key)
        {
            return;
        }
        if let PreviewTarget::MeshMap(kind) = key.target {
            if let Some(maps) = self.fresh_mesh_maps() {
                // Mesh-map previews are cheap conversions; no job needed.
                let texture = maps.to_texture(kind);
                self.baked_preview = Some(BakedPreview {
                    key,
                    textures: PreviewTextures::Single(Arc::new(texture)),
                });
            } else {
                self.ensure_mesh_maps();
            }
            return;
        }
        if let PreviewTarget::Channel(channel) = key.target
            && !BAKEABLE_CHANNELS.contains(&channel)
        {
            return;
        }
        let Some(texture_set) = self.selected_texture_set_ref().cloned() else {
            self.baked_preview = None;
            return;
        };
        if texture_set.requires_mesh_maps() && !self.mesh_maps_fresh() {
            // Keep the previous preview until fresh maps arrive.
            self.ensure_mesh_maps();
            return;
        }
        let set_revision = self.set_revision(texture_set.id);
        let inputs = self.bake_inputs(texture_set.id);
        let (sender, receiver) = sync_channel(1);
        let spawned = std::thread::Builder::new()
            .name("preview-bake".to_owned())
            .spawn(move || {
                let result = match key.target {
                    PreviewTarget::Channel(channel) => inputs
                        .bake_channel(&texture_set, channel)
                        .map(|texture| PreviewTextures::Single(Arc::new(texture))),
                    _ => inputs
                        .bake_material(&texture_set)
                        .map(|textures| PreviewTextures::Material(Arc::new(textures))),
                };
                let _ = sender.send(result);
            });
        match spawned {
            Ok(_) => {
                self.preview_job = Some(PreviewJob {
                    receiver,
                    key,
                    set_revision,
                });
            }
            Err(error) => {
                self.preview_failed = true;
                self.bake_error = Some(error.to_string());
            }
        }
    }

    /// Everything a background bake of texture set `set_id` reads, shared
    /// cheaply with the worker thread.
    fn bake_inputs(&self, set_id: TextureSetId) -> BakeInputs {
        BakeInputs {
            mesh: self.mesh_geometry.clone(),
            mesh_maps: self.mesh_maps.clone().filter(|_| self.mesh_maps_fresh()),
            slot: self.set_slot(set_id) as usize,
            resources: self.resource_images.clone(),
        }
    }

    fn poll_preview_job(&mut self, context: &egui::Context) {
        let Some(job) = &self.preview_job else {
            return;
        };
        match job.receiver.try_recv() {
            Ok(result) => {
                let (key, set_revision) = (job.key, job.set_revision);
                self.preview_job = None;
                match result {
                    Ok(textures) => {
                        if let (PreviewTextures::Material(material), Some(set_id)) =
                            (&textures, key.texture_set)
                        {
                            self.set_materials.insert(
                                set_id,
                                CachedMaterial {
                                    revision: set_revision,
                                    textures: Arc::clone(material),
                                },
                            );
                        }
                        self.baked_preview = Some(BakedPreview { key, textures });
                        self.bake_error = None;
                    }
                    // Only report failures of the preview still wanted.
                    Err(error) if key == self.preview_key() => {
                        self.preview_failed = true;
                        self.bake_error = Some(error);
                    }
                    Err(_) => {}
                }
                // A stale result is shown meanwhile; queue the current one.
                self.refresh_baked_preview();
                context.request_repaint();
            }
            Err(TryRecvError::Empty) => context.request_repaint_after(Duration::from_millis(16)),
            Err(TryRecvError::Disconnected) => self.preview_job = None,
        }
    }

    /// On multi-material meshes, rebakes (one at a time, after the selected
    /// preview) texture sets whose cached material is out of date, so every
    /// slot shows its own textures.
    fn refresh_set_caches(&mut self) {
        if self.set_bake_job.is_some() || self.preview_job.is_some() {
            return;
        }
        let slot_count = self
            .mesh_geometry
            .as_deref()
            .map_or(1, MeshGeometry::material_slot_count);
        if slot_count <= 1 {
            return;
        }
        let stale = (0..slot_count as u32)
            .filter_map(|slot| self.slot_texture_set(slot))
            .filter(|set_id| Some(*set_id) != self.selected_texture_set)
            .find(|set_id| {
                let revision = self.set_revision(*set_id);
                self.set_materials
                    .get(set_id)
                    .is_none_or(|cached| cached.revision != revision)
                    && self.set_bake_failed.get(set_id) != Some(&revision)
            });
        let Some(set_id) = stale else {
            return;
        };
        let Some(texture_set) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == set_id)
            .cloned()
        else {
            return;
        };
        if texture_set.requires_mesh_maps() && !self.mesh_maps_fresh() {
            self.ensure_mesh_maps();
            return;
        }
        let revision = self.set_revision(set_id);
        let inputs = self.bake_inputs(set_id);
        let (sender, receiver) = sync_channel(1);
        let spawned = std::thread::Builder::new()
            .name("set-bake".to_owned())
            .spawn(move || {
                let _ = sender.send(inputs.bake_material(&texture_set));
            });
        if spawned.is_ok() {
            self.set_bake_job = Some(SetBakeJob {
                set_id,
                revision,
                receiver,
            });
        }
    }

    fn poll_set_bake_job(&mut self, context: &egui::Context) {
        let Some(job) = &self.set_bake_job else {
            return;
        };
        match job.receiver.try_recv() {
            Ok(result) => {
                let (set_id, revision) = (job.set_id, job.revision);
                self.set_bake_job = None;
                match result {
                    Ok(textures) => {
                        self.set_materials.insert(
                            set_id,
                            CachedMaterial {
                                revision,
                                textures: Arc::new(textures),
                            },
                        );
                    }
                    Err(_) => {
                        self.set_bake_failed.insert(set_id, revision);
                    }
                }
                context.request_repaint();
            }
            Err(TryRecvError::Empty) => context.request_repaint_after(Duration::from_millis(32)),
            Err(TryRecvError::Disconnected) => self.set_bake_job = None,
        }
    }

    /// The finished preview matching what the viewport should show, possibly
    /// one revision behind while a newer bake is in flight.
    fn displayed_preview(&self) -> Option<&PreviewTextures> {
        let key = self.preview_key();
        self.baked_preview
            .as_ref()
            .filter(|preview| {
                preview.key.target == key.target && preview.key.texture_set == key.texture_set
            })
            .map(|preview| &preview.textures)
    }

    /// Returns a clone of the selected texture set when it can be baked now,
    /// starting a mesh-map bake (and reporting it) when maps are missing.
    fn exportable_texture_set(&mut self) -> Option<TextureSet> {
        let texture_set = self.selected_texture_set_ref()?.clone();
        if texture_set.requires_mesh_maps() && !self.mesh_maps_fresh() {
            self.ensure_mesh_maps();
            self.bake_error =
                Some("Mesh maps are still baking; export again when they finish.".to_owned());
            return None;
        }
        Some(texture_set)
    }

    /// Asks where to save one texture, offering the chosen export format
    /// first.
    fn texture_save_path(&self, stem: &str) -> Option<std::path::PathBuf> {
        let mut dialog = rfd::FileDialog::new();
        let formats = std::iter::once(self.export_format).chain(
            TextureFormat::ALL
                .into_iter()
                .filter(|format| *format != self.export_format),
        );
        for format in formats {
            dialog = dialog.add_filter(format!("{} texture", format.name()), &[format.extension()]);
        }
        dialog
            .set_file_name(format!("{stem}.{}", self.export_format.extension()).replace(' ', "_"))
            .save_file()
    }

    fn export_baked_map(&mut self, channel: MaterialChannel) {
        let Some(texture_set) = self.exportable_texture_set() else {
            return;
        };
        let stem = format!("{}_{}", texture_set.name, channel_label(channel));
        let Some(path) = self.texture_save_path(&stem) else {
            return;
        };
        let result = self
            .bake_inputs(texture_set.id)
            .bake_channel(&texture_set, channel)
            .and_then(|texture| write_texture(&path, &texture).map_err(|error| error.to_string()));
        match result {
            Ok(()) => self.bake_error = None,
            Err(error) => self.bake_error = Some(error),
        }
    }

    /// Writes `<set>_BaseColor`, `_ORM`, `_Normal`, and (when anything glows)
    /// `_Emissive` into a chosen folder in the export format: the common
    /// metal/rough engine layout. Base Color carries Opacity in alpha.
    fn export_material_set(&mut self) {
        let Some(texture_set) = self.exportable_texture_set() else {
            return;
        };
        let Some(folder) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        match self.write_material_set(&texture_set, &folder) {
            Ok(_) => self.bake_error = None,
            Err(error) => self.bake_error = Some(error),
        }
    }

    /// Bakes and writes the material set; returns the files written.
    fn write_material_set(
        &self,
        texture_set: &TextureSet,
        folder: &std::path::Path,
    ) -> Result<Vec<std::path::PathBuf>, String> {
        let textures = self
            .bake_inputs(texture_set.id)
            .bake_material(texture_set)?;
        let stem = texture_set.name.replace(' ', "_");
        let extension = self.export_format.extension();
        let mut written = Vec::new();
        for (suffix, texture) in [
            ("BaseColor", Some(&textures.base_color)),
            ("ORM", Some(&textures.orm)),
            ("Normal", Some(&textures.normal)),
            (
                "Emissive",
                textures.has_emission().then_some(&textures.emissive),
            ),
        ] {
            let Some(texture) = texture else {
                continue;
            };
            let path = folder.join(format!("{stem}_{suffix}.{extension}"));
            write_texture(&path, texture).map_err(|error| error.to_string())?;
            written.push(path);
        }
        Ok(written)
    }

    fn export_mesh_map(&mut self, kind: MeshMapKind) {
        let Some(texture) = self.fresh_mesh_maps().map(|maps| maps.to_texture(kind)) else {
            self.ensure_mesh_maps();
            return;
        };
        let Some(path) = self.texture_save_path(kind.name()) else {
            return;
        };
        if let Err(error) = write_texture(&path, &texture) {
            self.mesh_map_error = Some(error.to_string());
        }
    }

    /// Loads the pixels of every project resource, reporting unreadable files.
    fn reload_resources(&mut self) {
        let (images, failures) = load_project_resources(&self.project);
        self.resource_images = images;
        self.resource_thumbnails.clear();
        if let Some((id, error)) = failures.first() {
            let name = self
                .project
                .resources
                .iter()
                .find(|resource| resource.id == *id)
                .map_or("bitmap", |resource| resource.name.as_str());
            let more = if failures.len() > 1 {
                format!(" (and {} more)", failures.len() - 1)
            } else {
                String::new()
            };
            self.project_error = Some(format!("Resource `{name}`: {error}{more}"));
        }
        self.invalidate_all_sets();
    }

    fn import_resources(&mut self) {
        let Some(paths) = rfd::FileDialog::new()
            .add_filter("Bitmap (PNG, JPEG, TGA)", &IMAGE_EXTENSIONS)
            .pick_files()
        else {
            return;
        };
        for path in paths {
            if let Err(error) = self.add_resource(&path) {
                self.import_error = Some(error);
            }
        }
    }

    /// Loads a bitmap and lists it as a project resource.
    fn add_resource(&mut self, path: &std::path::Path) -> Result<ResourceId, String> {
        let image =
            load_texture_image(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let id = ResourceId(
            self.project
                .resources
                .iter()
                .map(|resource| resource.id.0)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        );
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("Bitmap {}", id.0));
        self.project.resources.push(ProjectResource {
            id,
            name,
            source: path.to_string_lossy().into_owned(),
        });
        self.resource_images.insert(id, Arc::new(image));
        Ok(id)
    }

    /// True when a layer or the current brush uses `id`.
    fn resource_in_use(&self, id: ResourceId) -> bool {
        self.brush_tip.resource() == Some(id)
            || self
                .project
                .texture_sets
                .iter()
                .any(|set| set.referenced_resources().contains(&id))
    }

    /// Removes an unused resource from the project.
    fn remove_resource(&mut self, id: ResourceId) {
        if self.resource_in_use(id) {
            return;
        }
        self.project.resources.retain(|resource| resource.id != id);
        self.resource_images.remove(&id);
        self.resource_thumbnails.remove(&id);
    }

    /// egui thumbnail of a resource, created on first use.
    fn resource_thumbnail(
        &mut self,
        context: &egui::Context,
        id: ResourceId,
    ) -> Option<egui::TextureHandle> {
        if let Some(texture) = self.resource_thumbnails.get(&id) {
            return Some(texture.clone());
        }
        let image = self.resource_images.get(&id)?;
        let thumbnail = context.load_texture(
            format!("resource-{}", id.0),
            uv_view::thumbnail_image(image, 96),
            egui::TextureOptions::LINEAR,
        );
        self.resource_thumbnails.insert(id, thumbnail.clone());
        Some(thumbnail)
    }

    fn selected_mesh_name(&self) -> Option<&str> {
        if let Some(index) = self.selected_imported_mesh {
            return self
                .imported_meshes
                .get(index)
                .map(|mesh| mesh.name.as_str());
        }
        self.selected_primitive.map(PrimitivePreset::name)
    }

    fn select_imported_mesh(&mut self, index: usize) {
        let Some(mesh) = self.imported_meshes.get(index) else {
            return;
        };
        let geometry = scaled_geometry(&mesh.geometry, self.import_scale);
        self.selected_imported_mesh = Some(index);
        self.selected_primitive = None;
        self.set_base_geometry(Some(geometry));
    }

    fn set_import_scale(&mut self, scale: f32) {
        self.import_scale = sanitize_mesh_scale(scale);
        if let Some(mesh_asset) = self.project.mesh.as_mut()
            && !mesh_asset.source.starts_with("builtin:")
        {
            mesh_asset.scale = self.import_scale;
        }
        if let Some(index) = self.selected_imported_mesh {
            self.select_imported_mesh(index);
        }
    }

    fn update_pose_transition(&mut self, context: &egui::Context) {
        let Some(transition) = &self.pose_transition else {
            return;
        };
        let now = context.input(|input| input.time);
        let amount =
            ((now - transition.started_at) / POSE_TRANSITION_SECONDS).clamp(0.0, 1.0) as f32;
        match transition.from.interpolate(&transition.to, amount) {
            Ok(pose) => self.current_pose = pose,
            Err(_) => {
                self.pose_transition = None;
                return;
            }
        }
        if amount >= 1.0 {
            self.pose_transition = None;
        } else {
            context.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn select_pose(&mut self, pose_id: PoseId, context: &egui::Context) {
        let Some(target) = self
            .project
            .saved_poses
            .iter()
            .find(|pose| pose.id == pose_id)
            .cloned()
        else {
            return;
        };
        self.selected_pose = Some(pose_id);
        self.pose_transition = Some(PoseTransition {
            from: self.current_pose.clone(),
            to: target,
            started_at: context.input(|input| input.time),
        });
    }

    fn save_pose(&mut self) {
        if self.project.saved_poses.len() >= MAX_SAVED_POSES
            || self.current_pose.validate().is_err()
        {
            return;
        }
        let Some(next_id) = self
            .project
            .saved_poses
            .iter()
            .map(|pose| pose.id.0)
            .max()
            .unwrap_or(0)
            .checked_add(1)
        else {
            return;
        };
        let base_name = self.pose_name.trim();
        if base_name.is_empty() {
            return;
        }
        let mut name = base_name.to_owned();
        let mut suffix = 2_u32;
        while self
            .project
            .saved_poses
            .iter()
            .any(|pose| pose.name.eq_ignore_ascii_case(&name))
        {
            name = format!("{base_name} {suffix}");
            let Some(next_suffix) = suffix.checked_add(1) else {
                return;
            };
            suffix = next_suffix;
        }
        let mut pose = self.current_pose.clone();
        pose.id = PoseId(next_id);
        pose.name.clone_from(&name);
        self.project.saved_poses.push(pose);
        self.selected_pose = Some(PoseId(next_id));
        self.pose_name = format!("Pose {}", self.project.saved_poses.len() + 1);
    }

    fn view_projection(&self, viewport: egui::Rect) -> ViewProjection {
        let pose = self
            .current_pose
            .bones
            .get(&BoneId(0))
            .copied()
            .unwrap_or(BoneTransform::IDENTITY);
        let posed_center = pose.rotation.mul_vec3(self.mesh_center * pose.scale) + pose.translation;
        ViewProjection {
            pose,
            camera_rotation: self.viewport_camera.rotation(),
            posed_center,
            center: viewport.center(),
            scale: viewport.width().min(viewport.height()) * 0.42 * self.viewport_camera.zoom * 0.9,
            pan: self.viewport_camera.pan,
        }
    }

    fn update_viewport(&mut self, ui: &mut egui::Ui) {
        self.read_pen_pressure(ui.ctx());
        // Strokes end when every button is up, whichever view drew them.
        if self.active_stroke.is_some() && !ui.input(|input| input.pointer.any_down()) {
            self.finish_active_stroke();
            self.invalidate_preview();
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(theme::VIEWPORT_BG))
            .show(ui, |ui| {
                let area = ui.max_rect();
                let (model, uv) = match self.view_mode {
                    ViewMode::Model => (Some(area), None),
                    ViewMode::Uv => (None, Some(area)),
                    ViewMode::Split => {
                        let middle = area.center().x;
                        (
                            Some(area.with_max_x(middle - 1.0)),
                            Some(area.with_min_x(middle + 1.0)),
                        )
                    }
                };
                // Runs during strokes too: previews bake in the background, so the
                // stroke in progress shows up live.
                self.refresh_baked_preview();
                if let Some(rect) = model {
                    self.model_viewport(ui, rect);
                }
                if let Some(rect) = uv {
                    self.uv_viewport(ui, rect);
                }
                if model.is_some() && uv.is_some() {
                    ui.painter().vline(
                        area.center().x,
                        area.y_range(),
                        egui::Stroke::new(2.0, theme::VIEWPORT_BG.gamma_multiply(0.5)),
                    );
                }
            });
    }

    /// Tracks stylus pressure from touch events; mice report none.
    fn read_pen_pressure(&mut self, context: &egui::Context) {
        context.input(|input| {
            for event in &input.events {
                if let egui::Event::Touch { phase, force, .. } = event {
                    self.pen_pressure = match phase {
                        egui::TouchPhase::Start | egui::TouchPhase::Move => {
                            force.filter(|force| force.is_finite() && *force > 0.0)
                        }
                        egui::TouchPhase::End | egui::TouchPhase::Cancel => None,
                    }
                    .map(|force| force.clamp(0.0, 1.0));
                }
            }
        });
    }

    /// The 3D view: Filament (or the software fallback), picking, painting,
    /// and polygon selection.
    fn model_viewport(&mut self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let response = ui.interact(
            viewport,
            ui.id().with("model_viewport"),
            egui::Sense::click_and_drag(),
        );
        let pointer_position = ui.input(|input| input.pointer.interact_pos());
        let hover_position = ui.input(|input| input.pointer.hover_pos());
        let shift = ui.input(|input| input.modifiers.shift);
        let painting = self.paint_mode.paints();
        if response.dragged() && !painting {
            let delta = ui.input(|input| input.pointer.delta());
            let delta = glam::Vec2::new(delta.x, delta.y);
            if shift {
                self.viewport_camera.pan_by(delta);
            } else {
                self.viewport_camera.orbit(delta);
            }
        }
        if response.hovered() && !painting {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            self.viewport_camera.zoom_by(scroll);
        }
        self.update_filament_frame(ui.ctx(), viewport);
        let painter = ui.painter_at(viewport);
        painter.rect_filled(viewport, 0.0, theme::VIEWPORT_BG);
        if self.mesh_geometry.is_some()
            && let Some(texture) = &self.filament_frame_texture
        {
            painter.image(
                texture.id(),
                viewport,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        let center = viewport.center();
        let software = self.filament_frame_texture.is_none();
        if software {
            let axis = egui::Color32::from_rgb(38, 40, 44);
            painter.line_segment(
                [
                    egui::pos2(viewport.left(), center.y),
                    egui::pos2(viewport.right(), center.y),
                ],
                egui::Stroke::new(1.0, axis),
            );
            painter.line_segment(
                [
                    egui::pos2(center.x, viewport.top()),
                    egui::pos2(center.x, viewport.bottom()),
                ],
                egui::Stroke::new(1.0, axis),
            );
        }

        if self.mesh_geometry.is_none() {
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                "Choose a mesh in the Scene drawer to begin",
                egui::FontId::proportional(16.0),
                theme::TEXT_WEAK,
            );
            return;
        }

        let selecting = self.paint_mode == PaintMode::Select;
        let projection = self.view_projection(viewport);
        let needs_projection =
            software || (painting && response.dragged()) || (selecting && response.hovered());
        let projected = needs_projection
            .then(|| {
                self.mesh_geometry
                    .as_ref()
                    .map(|mesh| projection.project_mesh(mesh))
            })
            .flatten();

        if painting
            && response.dragged()
            && let (Some(points), Some(pointer), Some(mesh)) =
                (&projected, pointer_position, &self.mesh_geometry)
            && let Some(hit) = pick_triangle(mesh, points, pointer)
            && self.triangle_in_selected_set(hit.triangle)
            && let Some(uv) = hit
                .uv
                .filter(|uv| (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y))
        {
            self.append_stroke_point(uv);
        }

        if selecting {
            let hit = match (&projected, hover_position, &self.mesh_geometry) {
                (Some(points), Some(pointer), Some(mesh)) if response.hovered() => {
                    pick_triangle(mesh, points, pointer)
                }
                _ => None,
            };
            self.hovered_selection = match (hit, &self.mesh_topology) {
                (Some(hit), Some(topology)) => topology.select(hit.triangle, self.selection_mode),
                _ => Vec::new(),
            };
            if response.clicked() && !self.hovered_selection.is_empty() {
                let selection = self.hovered_selection.clone();
                self.apply_polygon_selection(&selection, shift);
            }
        } else {
            self.hovered_selection.clear();
        }

        let Some(mesh) = self.mesh_geometry.as_ref() else {
            return;
        };
        if software && let Some(points) = &projected {
            self.paint_software_mesh(&painter, mesh, points);
        }

        if selecting {
            let overlay_points = projected
                .clone()
                .unwrap_or_else(|| projection.project_mesh(mesh));
            if let Some(mask) = self
                .selected_layer()
                .and_then(|layer| layer.polygon_mask.as_ref())
            {
                let triangles: Box<dyn Iterator<Item = u32>> = if mask.inverted {
                    Box::new(
                        (0..mesh.triangles.len() as u32)
                            .filter(|triangle| !mask.triangles.contains(triangle)),
                    )
                } else {
                    Box::new(mask.triangles.iter().copied())
                };
                painter.add(triangle_overlay(
                    mesh,
                    &overlay_points,
                    triangles,
                    theme::ACCENT.gamma_multiply(0.28),
                ));
            }
            let hover_color = if shift {
                theme::DANGER.gamma_multiply(0.35)
            } else {
                egui::Color32::from_white_alpha(46)
            };
            painter.add(triangle_overlay(
                mesh,
                &overlay_points,
                self.hovered_selection.iter().copied(),
                hover_color,
            ));
        }

        if painting
            && response.hovered()
            && let Some(pointer) = pointer_position
        {
            let cursor_color = if self.paint_mode == PaintMode::Mask {
                egui::Color32::WHITE
            } else {
                self.brush_color
            };
            let radius = (self.brush_radius * self.viewport_camera.zoom * 0.35).clamp(3.0, 36.0);
            painter.circle(
                pointer,
                radius,
                cursor_color.gamma_multiply(0.12),
                egui::Stroke::new(1.5, cursor_color),
            );
            painter.circle_filled(pointer, 1.5, cursor_color);
        }
    }

    /// Flat-shaded painter's-algorithm fallback when Filament is unavailable.
    fn paint_software_mesh(
        &self,
        painter: &egui::Painter,
        mesh: &MeshGeometry,
        points: &[ProjectedVertex],
    ) {
        let preview = self.displayed_preview();
        let mut faces = mesh
            .triangles
            .iter()
            .filter_map(|triangle| {
                let [a, b, c] = triangle.map(|index| points.get(index as usize));
                let (a, b, c) = (a?, b?, c?);
                let uvs = triangle.map(|index| mesh.vertices[index as usize].uv);
                let normal = (b.view - a.view).cross(c.view - a.view).normalize_or_zero();
                let intensity = self.viewport_lighting.face_intensity(normal);
                Some((
                    (a.view.z + b.view.z + c.view.z) / 3.0,
                    [a.screen, b.screen, c.screen],
                    uvs,
                    intensity,
                ))
            })
            .collect::<Vec<_>>();
        faces.sort_by(|left, right| left.0.total_cmp(&right.0));
        for (_, points, uvs, intensity) in faces {
            let uv_center = (uvs[0] + uvs[1] + uvs[2]) / 3.0;
            let map_color = preview
                .and_then(|preview| preview.base().sample_uv(uv_center))
                .filter(|color| color[3] > 0)
                .unwrap_or([92, 132, 151, 255]);
            let color = egui::Color32::from_rgb(
                (map_color[0] as f32 * intensity).clamp(0.0, 255.0) as u8,
                (map_color[1] as f32 * intensity).clamp(0.0, 255.0) as u8,
                (map_color[2] as f32 * intensity).clamp(0.0, 255.0) as u8,
            );
            painter.add(egui::Shape::convex_polygon(
                points.to_vec(),
                color,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(182, 195, 202)),
            ));
        }
    }

    fn handle_shortcuts(&mut self, context: &egui::Context) {
        if context.egui_wants_keyboard_input() {
            return;
        }
        let redo = context.input_mut(|input| {
            input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
                egui::Key::Z,
            )) || input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::Y,
            ))
        });
        if redo {
            self.redo();
            return;
        }
        let undo = context.input_mut(|input| {
            input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::Z,
            ))
        });
        if undo {
            self.undo();
        }
    }
}

/// Maps mesh positions through the current pose and camera onto the screen,
/// matching the orientation Filament renders.
struct ViewProjection {
    pose: BoneTransform,
    camera_rotation: glam::Quat,
    posed_center: glam::Vec3,
    center: egui::Pos2,
    scale: f32,
    pan: glam::Vec2,
}

#[derive(Clone, Copy)]
struct ProjectedVertex {
    screen: egui::Pos2,
    /// Camera-space position; larger `z` is closer to the viewer.
    view: glam::Vec3,
}

impl ViewProjection {
    fn project(&self, position: glam::Vec3) -> ProjectedVertex {
        let posed = self.pose.rotation.mul_vec3(position * self.pose.scale) + self.pose.translation;
        let view = self.camera_rotation * (posed - self.posed_center);
        ProjectedVertex {
            screen: egui::pos2(
                self.center.x + view.x * self.scale + self.pan.x,
                self.center.y - view.y * self.scale + self.pan.y,
            ),
            view,
        }
    }

    fn project_mesh(&self, mesh: &MeshGeometry) -> Vec<ProjectedVertex> {
        mesh.vertices
            .iter()
            .map(|vertex| self.project(vertex.position))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TriangleHit {
    triangle: u32,
    /// Interpolated UV, when the triangle's UVs are usable.
    uv: Option<glam::Vec2>,
}

/// Nearest triangle under `pointer`.
fn pick_triangle(
    mesh: &MeshGeometry,
    points: &[ProjectedVertex],
    pointer: egui::Pos2,
) -> Option<TriangleHit> {
    let mut best: Option<(f32, TriangleHit)> = None;
    for (index, triangle) in mesh.triangles.iter().enumerate() {
        let [a, b, c] = triangle.map(|vertex| points.get(vertex as usize));
        let (Some(a), Some(b), Some(c)) = (a, b, c) else {
            continue;
        };
        let min_x = a.screen.x.min(b.screen.x).min(c.screen.x);
        let max_x = a.screen.x.max(b.screen.x).max(c.screen.x);
        let min_y = a.screen.y.min(b.screen.y).min(c.screen.y);
        let max_y = a.screen.y.max(b.screen.y).max(c.screen.y);
        if pointer.x < min_x || pointer.x > max_x || pointer.y < min_y || pointer.y > max_y {
            continue;
        }
        let Some(weights) = barycentric_weights(pointer, [a.screen, b.screen, c.screen]) else {
            continue;
        };
        let depth = a.view.z * weights.x + b.view.z * weights.y + c.view.z * weights.z;
        if best.is_some_and(|(best_depth, _)| best_depth >= depth) {
            continue;
        }
        let uvs = triangle.map(|vertex| mesh.vertices[vertex as usize].uv);
        let uv = uvs[0] * weights.x + uvs[1] * weights.y + uvs[2] * weights.z;
        best = Some((
            depth,
            TriangleHit {
                triangle: index as u32,
                uv: uv.is_finite().then_some(uv),
            },
        ));
    }
    best.map(|(_, hit)| hit)
}

/// Translucent fill over front-facing `triangles`, built as one mesh shape.
fn triangle_overlay(
    mesh: &MeshGeometry,
    points: &[ProjectedVertex],
    triangles: impl Iterator<Item = u32>,
    color: egui::Color32,
) -> egui::Shape {
    let mut overlay = egui::Mesh::default();
    for triangle in triangles.take(MAX_OVERLAY_TRIANGLES) {
        let Some(corners) = mesh.triangles.get(triangle as usize) else {
            continue;
        };
        let [a, b, c] = corners.map(|vertex| points.get(vertex as usize));
        let (Some(a), Some(b), Some(c)) = (a, b, c) else {
            continue;
        };
        if (b.view - a.view).cross(c.view - a.view).z <= 0.0 {
            continue;
        }
        let base = overlay.vertices.len() as u32;
        for point in [a, b, c] {
            overlay.colored_vertex(point.screen, color);
        }
        overlay.add_triangle(base, base + 1, base + 2);
    }
    egui::Shape::mesh(overlay)
}

fn barycentric_weights(point: egui::Pos2, screen: [egui::Pos2; 3]) -> Option<glam::Vec3> {
    let edge_a = glam::Vec2::new(screen[1].x - screen[0].x, screen[1].y - screen[0].y);
    let edge_b = glam::Vec2::new(screen[2].x - screen[0].x, screen[2].y - screen[0].y);
    let relative = glam::Vec2::new(point.x - screen[0].x, point.y - screen[0].y);
    let denominator = edge_a.perp_dot(edge_b);
    if denominator.abs() <= f32::EPSILON {
        return None;
    }
    let weight_b = relative.perp_dot(edge_b) / denominator;
    let weight_c = edge_a.perp_dot(relative) / denominator;
    let weight_a = 1.0 - weight_b - weight_c;
    if weight_a < -0.001 || weight_b < -0.001 || weight_c < -0.001 {
        return None;
    }
    Some(glam::Vec3::new(weight_a, weight_b, weight_c))
}

#[cfg(test)]
fn barycentric_uv(
    point: egui::Pos2,
    screen: [egui::Pos2; 3],
    uvs: [glam::Vec2; 3],
) -> Option<glam::Vec2> {
    let weights = barycentric_weights(point, screen)?;
    let uv = uvs[0] * weights.x + uvs[1] * weights.y + uvs[2] * weights.z;
    (uv.is_finite() && (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y)).then_some(uv)
}

/// Visits every layer of a stack, groups before their children.
fn for_each_layer_mut(layers: &mut [MaterialLayer], visit: &mut impl FnMut(&mut MaterialLayer)) {
    for layer in layers {
        visit(layer);
        if let Some(children) = layer.children.as_mut() {
            for_each_layer_mut(children, visit);
        }
    }
}

fn mesh_bounds_center(mesh: &MeshGeometry) -> glam::Vec3 {
    let mut minimum = glam::Vec3::splat(f32::INFINITY);
    let mut maximum = glam::Vec3::splat(f32::NEG_INFINITY);
    for vertex in &mesh.vertices {
        minimum = minimum.min(vertex.position);
        maximum = maximum.max(vertex.position);
    }
    (minimum + maximum) * 0.5
}

const MIN_MESH_SCALE: f32 = 0.0001;
const MAX_MESH_SCALE: f32 = 1000.0;

fn sanitize_mesh_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(MIN_MESH_SCALE, MAX_MESH_SCALE)
    } else {
        1.0
    }
}

/// Reads an FBX (every mesh it holds) or a glTF/GLB (the flattened scene),
/// choosing the importer by extension.
fn load_mesh_file(path: &std::path::Path) -> Result<Vec<ImportedMesh>, String> {
    let extension = path
        .extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if extension == "gltf" || extension == "glb" {
        return import_gltf_file(path)
            .map(|mesh| vec![mesh])
            .map_err(|error| error.to_string());
    }
    let metadata = std::fs::metadata(path).map_err(|error| error.to_string())?;
    let file_size = usize::try_from(metadata.len())
        .map_err(|_| "FBX file is too large for this platform".to_owned())?;
    if file_size > MAX_FBX_BYTES {
        return Err(format!(
            "FBX file exceeds the {} MiB import limit",
            MAX_FBX_BYTES / (1024 * 1024)
        ));
    }
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    import_binary_fbx_meshes(&bytes).map_err(|error| error.to_string())
}

/// Reads every mesh in a file and merges them into one scaled geometry, as
/// a high-poly source for baking.
fn load_merged_mesh(path: &std::path::Path, scale: f32) -> Result<MeshGeometry, String> {
    let mut merged = MeshGeometry::default();
    for mesh in load_mesh_file(path)? {
        merged
            .append(&mesh.geometry)
            .map_err(|_| "merged mesh has too many vertices".to_owned())?;
    }
    if merged.triangles.is_empty() {
        return Err("the file contains no mesh geometry".to_owned());
    }
    Ok(scaled_geometry(&merged, sanitize_mesh_scale(scale)))
}

/// Bakes mesh maps for each material slot from that slot's triangles, so
/// slots sharing UV space get their own maps. A single-slot mesh bakes
/// whole; an empty slot falls back to the whole mesh.
fn bake_slot_mesh_maps(
    mesh: &MeshGeometry,
    high: Option<&MeshGeometry>,
    settings: &MeshMapSettings,
    cancel: &AtomicBool,
) -> Result<Vec<MeshMaps>, MeshMapError> {
    let slot_count = mesh.material_slot_count();
    if slot_count <= 1 {
        return Ok(vec![bake_mesh_maps_from(
            mesh,
            high,
            settings,
            Some(cancel),
        )?]);
    }
    (0..slot_count as u32)
        .map(|slot| {
            let submesh = mesh.submesh(slot);
            let source = if submesh.triangles.is_empty() {
                mesh
            } else {
                &submesh
            };
            bake_mesh_maps_from(source, high, settings, Some(cancel))
        })
        .collect()
}

fn scaled_geometry(mesh: &MeshGeometry, scale: f32) -> MeshGeometry {
    let mut geometry = mesh.clone();
    if scale != 1.0 {
        for vertex in &mut geometry.vertices {
            vertex.position *= scale;
        }
    }
    geometry
}

impl eframe::App for SubstanceCraftApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.handle_shortcuts(ui.ctx());
        self.poll_mesh_map_job(ui.ctx());
        self.poll_preview_job(ui.ctx());
        self.poll_set_bake_job(ui.ctx());
        self.refresh_set_caches();
        self.update_pose_transition(ui.ctx());
        self.update_viewport(ui);
        self.update_overlay(ui.ctx());
        self.sync_layer_history(ui.ctx());
    }
}

fn channel_label(channel: MaterialChannel) -> &'static str {
    match channel {
        MaterialChannel::BaseColor => "Base Color",
        MaterialChannel::Id => "ID Mask",
        MaterialChannel::Roughness => "Roughness",
        MaterialChannel::Metallic => "Metallic",
        MaterialChannel::Normal => "Normal",
        MaterialChannel::Height => "Height",
        MaterialChannel::AmbientOcclusion => "Ambient Occlusion",
        MaterialChannel::Emissive => "Emissive",
        MaterialChannel::Opacity => "Opacity",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_api::write_texture;

    #[test]
    fn screen_point_maps_to_triangle_uv_and_rejects_outside_points() {
        let screen = [
            egui::pos2(0.0, 0.0),
            egui::pos2(100.0, 0.0),
            egui::pos2(0.0, 100.0),
        ];
        let uvs = [
            glam::Vec2::new(0.0, 0.0),
            glam::Vec2::new(1.0, 0.0),
            glam::Vec2::new(0.0, 1.0),
        ];

        assert_eq!(
            barycentric_uv(egui::pos2(25.0, 25.0), screen, uvs),
            Some(glam::Vec2::new(0.25, 0.25))
        );
        assert_eq!(barycentric_uv(egui::pos2(75.0, 75.0), screen, uvs), None);
    }

    /// Drives background jobs until `done` or a timeout.
    fn pump_until(
        app: &mut SubstanceCraftApp,
        context: &egui::Context,
        done: impl Fn(&SubstanceCraftApp) -> bool,
    ) {
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while !done(app) {
            assert!(
                std::time::Instant::now() < deadline,
                "background jobs timed out"
            );
            app.poll_mesh_map_job(context);
            app.poll_preview_job(context);
            app.poll_set_bake_job(context);
            app.refresh_baked_preview();
            app.refresh_set_caches();
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    #[allow(clippy::unreachable)]
    fn tc_0054_edge_wear_layer_bakes_maps_and_previews_metal_edges_in_background() {
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        app.mesh_map_settings.resolution = 64;
        app.mesh_map_settings.ao_samples = 4;

        app.add_edge_wear_layer();

        let set = app.selected_texture_set_ref().unwrap();
        assert_eq!(set.layers.len(), 1, "Edge Wear adds one layer");
        assert_eq!(app.selected_layer, Some(set.layers[0].id));
        assert!(app.history.can_undo());
        // Small set so the debug-build bake stays quick.
        app.project.texture_sets[0].resolution = TextureResolution {
            width: 96,
            height: 96,
        };
        app.invalidate_preview();

        app.refresh_baked_preview();
        assert!(
            app.mesh_map_job.is_some(),
            "the generator triggers a mesh-map bake"
        );
        pump_until(&mut app, &context, |app| {
            app.displayed_preview()
                .is_some_and(|preview| matches!(preview, PreviewTextures::Material(_)))
                && app.baked_preview.as_ref().map(|preview| preview.key) == Some(app.preview_key())
        });

        let Some(PreviewTextures::Material(textures)) = app.displayed_preview() else {
            unreachable!("checked by pump_until")
        };
        let metal_texels = textures
            .orm
            .rgba8
            .chunks(4)
            .filter(|pixel| pixel[2] > 200)
            .count();
        let total = textures.orm.rgba8.len() / 4;
        assert!(metal_texels > 0, "worn edges are metallic");
        assert!(metal_texels < total / 2, "flat faces stay unworn");
        // Unworn texels show the neutral base, not transparent black.
        assert!(
            textures
                .base_color
                .rgba8
                .chunks(4)
                .all(|pixel| pixel[3] == 255)
        );

        app.undo();
        assert!(app.selected_texture_set_ref().unwrap().layers.is_empty());
    }

    #[test]
    fn tc_0045_viewport_pick_returns_the_nearest_triangle_under_the_pointer() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let projection = ViewProjection {
            pose: BoneTransform::IDENTITY,
            camera_rotation: glam::Quat::IDENTITY,
            posed_center: glam::Vec3::ZERO,
            center: egui::pos2(100.0, 100.0),
            scale: 100.0,
            pan: glam::Vec2::ZERO,
        };
        let points = projection.project_mesh(&cube);

        // Looking down -Z, the screen centre covers the +Z face (triangles 0
        // and 1) in front of the -Z face (triangles 2 and 3).
        let hit = pick_triangle(&cube, &points, egui::pos2(110.0, 90.0)).unwrap();
        assert!(hit.triangle < 2, "picked {}", hit.triangle);
        assert!(hit.uv.is_some());
        assert_eq!(
            pick_triangle(&cube, &points, egui::pos2(400.0, 400.0)),
            None
        );

        let topology = MeshTopology::build(&cube);
        let face = topology.select(hit.triangle, SelectionMode::UvIsland);
        assert_eq!(face, vec![0, 1]);
    }

    /// A unit quad split into two triangles, each with its own glTF material
    /// ("Body" and "Trim"), both covering the full UV square.
    fn write_two_material_gltf(folder: &std::path::Path) -> std::path::PathBuf {
        let positions: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // glTF V runs down; these map to SubstanceCraft UV corners.
        let uvs: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let mut buffer = Vec::new();
        for value in positions.iter().flatten().chain(uvs.iter().flatten()) {
            buffer.extend(value.to_le_bytes());
        }
        for index in [0_u16, 1, 2, 0, 2, 3] {
            buffer.extend(index.to_le_bytes());
        }
        std::fs::write(folder.join("quad.bin"), &buffer).unwrap();
        let json = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "materials": [{{ "name": "Body" }}, {{ "name": "Trim" }}],
  "meshes": [{{ "primitives": [
    {{ "attributes": {{ "POSITION": 0, "TEXCOORD_0": 1 }}, "indices": 2, "material": 0 }},
    {{ "attributes": {{ "POSITION": 0, "TEXCOORD_0": 1 }}, "indices": 3, "material": 1 }}
  ] }}],
  "buffers": [{{ "uri": "quad.bin", "byteLength": {length} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 48 }},
    {{ "buffer": 0, "byteOffset": 48, "byteLength": 32 }},
    {{ "buffer": 0, "byteOffset": 80, "byteLength": 6 }},
    {{ "buffer": 0, "byteOffset": 86, "byteLength": 6 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
       "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] }},
    {{ "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2" }},
    {{ "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }},
    {{ "bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR" }}
  ]
}}"#,
            length = buffer.len()
        );
        let path = folder.join("quad.gltf");
        std::fs::write(&path, json).unwrap();
        path
    }

    fn small_sets(app: &mut SubstanceCraftApp, size: u32) {
        for set in &mut app.project.texture_sets {
            set.resolution = TextureResolution {
                width: size,
                height: size,
            };
        }
        app.invalidate_all_sets();
    }

    #[test]
    fn tc_0101_gltf_material_slots_become_texture_sets_that_each_show_on_their_slot() {
        let context = egui::Context::default();
        let folder = tempfile::tempdir().unwrap();
        let path = write_two_material_gltf(folder.path());
        let mut app = SubstanceCraftApp::default();

        app.load_imported_mesh(&path);
        assert_eq!(app.import_error, None);
        let names = app
            .project
            .texture_sets
            .iter()
            .map(|set| set.material_slot.clone())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["Body", "Trim"]);
        let (body, trim) = (
            app.project.texture_sets[0].id,
            app.project.texture_sets[1].id,
        );
        assert_eq!(app.selected_texture_set, Some(body));
        assert_eq!(app.slot_texture_set(1), Some(trim));
        // Painting hits only count on the selected set's triangles.
        assert!(app.triangle_in_selected_set(0));
        assert!(!app.triangle_in_selected_set(1));
        small_sets(&mut app, 32);

        // Give Trim a red fill; Body stays plain.
        app.select_texture_set(trim);
        app.add_fill_layer();
        if let Some(layer) = app.selected_layer_mut() {
            layer.fill = Some(LayerFill {
                color: [255, 0, 0, 255],
                ..LayerFill::default()
            });
        }
        app.select_texture_set(body);
        app.invalidate_all_sets();
        pump_until(&mut app, &context, |app| {
            app.preview_job.is_none()
                && app.set_bake_job.is_none()
                && app
                    .set_materials
                    .get(&trim)
                    .is_some_and(|cached| cached.revision == app.set_revision(trim))
        });

        let plan = app.slot_upload_plan(PreviewTarget::Material);
        assert_eq!(plan.len(), 2);
        let centre = |source: &SlotSource| match source {
            SlotSource::Material(textures) => textures.base_color.pixel(16, 16),
            _ => None,
        };
        assert_eq!(
            centre(&plan[1]),
            Some([255, 0, 0, 255]),
            "the Trim slot shows its cached material"
        );
        let base_color = centre(&plan[0]);
        assert_eq!(base_color, Some([128, 128, 128, 255]), "Body is untouched");
    }

    #[test]
    fn tc_0102_bitmap_resources_texture_fills_and_material_sets_export_as_png() {
        let folder = tempfile::tempdir().unwrap();
        let bitmap = folder.path().join("stripes.png");
        write_texture(
            &bitmap,
            &BakedTexture {
                width: 2,
                height: 1,
                rgba8: vec![0, 255, 0, 255, 0, 255, 0, 255],
            },
        )
        .unwrap();
        let mut app = SubstanceCraftApp::default();
        let resource = app.add_resource(&bitmap).unwrap();
        assert_eq!(app.project.resources.len(), 1);

        app.add_fill_layer();
        if let Some(layer) = app.selected_layer_mut() {
            layer.channels.insert(MaterialChannel::Emissive);
            let fill = layer.fill.get_or_insert_with(LayerFill::default);
            fill.textures.base_color = Some(resource);
            fill.emissive = [10, 20, 30];
        }
        small_sets(&mut app, 16);
        assert!(app.resource_in_use(resource));
        app.remove_resource(resource);
        assert_eq!(app.project.resources.len(), 1, "used resources stay");

        let set = app.selected_texture_set_ref().unwrap().clone();
        let out = folder.path().join("out");
        std::fs::create_dir(&out).unwrap();
        let written = app.write_material_set(&set, &out).unwrap();
        let names = written
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let stem = set.name.replace(' ', "_");
        assert_eq!(
            names,
            ["BaseColor", "ORM", "Normal", "Emissive"].map(|suffix| format!("{stem}_{suffix}.png"))
        );
        let base = load_texture_image(&written[0]).unwrap();
        assert_eq!(&base.rgba8[0..4], &[0, 255, 0, 255], "textured green fill");
        let emissive = load_texture_image(&written[3]).unwrap();
        assert_eq!(&emissive.rgba8[0..4], &[10, 20, 30, 255]);

        // The project saves the reference and reloads the pixels.
        let project = folder.path().join("scene.sproject");
        save_project_file(&project, &app.project).unwrap();
        app.project = load_project_file(&project).unwrap();
        app.resource_images.clear();
        app.reload_resources();
        assert!(app.resource_images.contains_key(&resource));
        assert_eq!(app.project_error, None);
    }

    #[test]
    fn tc_0103_erase_mode_and_groups_work_through_the_app() {
        let mut app = SubstanceCraftApp::default();
        app.add_paint_layer();
        let paint = app.selected_layer.unwrap();
        app.paint_mode = PaintMode::Paint;
        app.append_stroke_point(glam::Vec2::new(0.4, 0.5));
        app.append_stroke_point(glam::Vec2::new(0.6, 0.5));
        app.finish_active_stroke();
        app.paint_mode = PaintMode::Erase;
        app.pen_pressure = Some(0.5);
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.append_stroke_point(glam::Vec2::new(0.55, 0.5));
        app.finish_active_stroke();
        let strokes = &app.selected_layer().unwrap().paint_strokes;
        assert_eq!(strokes.len(), 2);
        assert!(!strokes[0].erase && strokes[1].erase);
        assert_eq!(
            strokes[1].pressures,
            vec![0.5, 0.5],
            "pen pressure per point"
        );
        assert!(strokes[0].pressures.is_empty(), "mouse strokes carry none");

        // A group added above the paint layer; moving the paint layer up
        // and into the group below needs it to sit above the group first.
        app.add_group_layer();
        let group = app.selected_layer.unwrap();
        app.selected_layer = Some(paint);
        app.move_selected_layer(1);
        assert_eq!(app.group_below_selected(), Some(group));
        app.move_selected_into_group();
        let set = app.selected_texture_set_ref().unwrap();
        assert_eq!(set.slot_of(paint).unwrap().parent, Some(group));
        // Strokes cannot target a group.
        app.selected_layer = Some(group);
        app.paint_mode = PaintMode::Paint;
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        assert!(app.active_stroke.is_none());

        app.selected_layer = Some(paint);
        app.move_selected_out_of_group();
        let set = app.selected_texture_set_ref().unwrap();
        assert_eq!(set.slot_of(paint).unwrap().parent, None);
        app.undo();
        let set = app.selected_texture_set_ref().unwrap();
        assert_eq!(set.slot_of(paint).unwrap().parent, Some(group));
    }
}
