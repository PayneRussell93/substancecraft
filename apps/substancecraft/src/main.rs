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
    ImportedFbxMesh, MAX_FBX_BYTES, bake_texture_set_channel, bake_texture_set_material,
    import_binary_fbx_meshes, load_project_file, save_project_file,
};
use substancecraft_core::{
    BakedTexture, BoneId, BoneTransform, DEFAULT_HEIGHT, DEFAULT_HEIGHT_STRENGTH, Dirt, EdgeWear,
    EditCommand, EditHistory, IdStroke, LayerFill, LayerId, LayerSettings, MaskGenerator,
    MaskStroke, MaterialChannel, MaterialLayer, MaterialProject, MaterialTextures, MeshAsset,
    MeshGeometry, MeshId, MeshMapError, MeshMapKind, MeshMapSettings, MeshMaps, MeshSmoothing,
    MeshTopology, NamedPose, PaintStroke, PoseId, PrimitivePreset, SelectionMode,
    TextureResolution, TextureSet, TextureSetId, ViewportCamera, ViewportLighting,
    bake_mesh_maps_from, remap_subdivided_triangles,
};
use substancecraft_filament::{FilamentRuntime, MaterialMap};

mod icons;
mod panels;
mod theme;
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
const BAKEABLE_CHANNELS: [MaterialChannel; 6] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Height,
    MaterialChannel::Normal,
    MaterialChannel::Id,
];
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
    imported_meshes: Vec<ImportedFbxMesh>,
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
    mesh_maps: Option<Arc<MeshMaps>>,
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
    Mask,
    Id,
    Select,
}

impl PaintMode {
    fn paints(self) -> bool {
        matches!(self, Self::Paint | Self::Mask | Self::Id)
    }
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
}

enum PreviewTextures {
    Material(Box<MaterialTextures>),
    Single(BakedTexture),
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
    receiver: Receiver<Result<MeshMaps, MeshMapError>>,
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
    imported_meshes: Vec<ImportedFbxMesh>,
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
        self.invalidate_preview();
        // Old textures do not fit the new mesh's UVs.
        self.baked_preview = None;
    }

    /// Marks the shown preview out of date. It stays visible until the
    /// background rebake replaces it.
    fn invalidate_preview(&mut self) {
        self.preview_revision = self.preview_revision.wrapping_add(1);
        self.preview_failed = false;
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
            for layer in self
                .project
                .texture_sets
                .iter_mut()
                .flat_map(|set| set.layers.iter_mut())
            {
                if let Some(mask) = layer.polygon_mask.as_mut() {
                    mask.triangles =
                        remap_subdivided_triangles(mask.triangles.iter().copied(), from, to)
                            .into_iter()
                            .collect();
                }
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
        let Some(mesh) = self.mesh_geometry.as_deref() else {
            self.filament_frame_texture = None;
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

        let preview = self
            .baked_preview
            .as_ref()
            .filter(|preview| {
                preview.key.target == wanted.target && preview.key.texture_set == wanted.texture_set
            })
            .map(|preview| &preview.textures);
        let upload = |renderer: &mut FilamentRuntime, map: MaterialMap, texture: &BakedTexture| {
            let key = (texture.rgba8.as_ptr() as usize)
                ^ (texture.width as usize).rotate_left(7)
                ^ (texture.height as usize).rotate_left(17);
            renderer.set_map(map, key, texture.width, texture.height, &texture.rgba8)
        };
        // Neutral fallbacks: grey dielectric, AO 1 / roughness 0.58, flat normal.
        let neutral = |renderer: &mut FilamentRuntime| {
            renderer
                .set_map(MaterialMap::BaseColor, 0, 1, 1, &[168, 173, 179, 255])
                .and_then(|()| renderer.set_map(MaterialMap::Orm, 0, 1, 1, &[255, 148, 0, 255]))
                .and_then(|()| {
                    renderer.set_map(MaterialMap::Normal, 0, 1, 1, &[128, 128, 255, 255])
                })
        };
        let texture_result = match preview {
            Some(PreviewTextures::Material(textures)) => {
                renderer.set_preview_mode(false);
                upload(renderer, MaterialMap::BaseColor, &textures.base_color)
                    .and_then(|()| upload(renderer, MaterialMap::Orm, &textures.orm))
                    .and_then(|()| upload(renderer, MaterialMap::Normal, &textures.normal))
            }
            Some(PreviewTextures::Single(texture)) => {
                renderer.set_preview_mode(true);
                upload(renderer, MaterialMap::BaseColor, texture)
            }
            None => {
                renderer.set_preview_mode(target != PreviewTarget::Material);
                neutral(renderer)
            }
        };
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
            .flat_map(|set| set.layers.iter().map(|layer| layer.id.0))
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
        let metadata = std::fs::metadata(&path).map_err(|error| error.to_string())?;
        let file_size = usize::try_from(metadata.len())
            .map_err(|_| "referenced FBX file is too large for this platform".to_owned())?;
        if file_size > MAX_FBX_BYTES {
            return Err("referenced FBX exceeds the 64 MiB import limit".to_owned());
        }
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        let meshes = import_binary_fbx_meshes(&bytes).map_err(|error| error.to_string())?;
        let Some(first_mesh) = meshes.first() else {
            return Err("referenced FBX contains no mesh geometry".to_owned());
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

    fn import_fbx(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("FBX mesh", &["fbx"])
            .pick_file()
        else {
            return;
        };

        let result = (|| {
            let metadata = std::fs::metadata(&path).map_err(|error| error.to_string())?;
            let file_size = usize::try_from(metadata.len())
                .map_err(|_| "FBX file is too large for this platform".to_owned())?;
            if file_size > MAX_FBX_BYTES {
                return Err(format!(
                    "FBX file exceeds the {} MiB import limit",
                    MAX_FBX_BYTES / (1024 * 1024)
                ));
            }
            let bytes = std::fs::read(&path).map_err(|error| error.to_string())?;
            import_binary_fbx_meshes(&bytes).map_err(|error| error.to_string())
        })();

        match result {
            Ok(meshes) => {
                let Some(first_mesh) = meshes.first() else {
                    self.import_error = Some("FBX contains no mesh geometry".to_owned());
                    return;
                };
                let first_geometry = scaled_geometry(&first_mesh.geometry, self.import_scale);
                let display_name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "Imported FBX".to_owned());
                self.project.mesh = Some(MeshAsset {
                    id: MeshId(1),
                    source: path.to_string_lossy().into_owned(),
                    scale: self.import_scale,
                    smoothing: self.mesh_smoothing,
                    high_poly_source: None,
                });
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
                self.current_pose.name = display_name;
            }
            Err(error) => self.import_error = Some(error),
        }
    }

    fn add_texture_set(&mut self) {
        let Some(next_id) = self.next_texture_set_id.checked_add(1) else {
            return;
        };
        let id = TextureSetId(self.next_texture_set_id);
        self.next_texture_set_id = next_id;
        self.project.texture_sets.push(TextureSet {
            id,
            name: format!("Texture Set {}", id.0),
            material_slot: format!("Material {}", id.0),
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

    /// Inserts `layer` above the selected layer (or on top) as one undoable
    /// edit, creating a texture set first when none exists.
    fn insert_layer(&mut self, layer: MaterialLayer) {
        if self.selected_texture_set.is_none() {
            self.add_texture_set();
        }
        let Some(texture_set_id) = self.selected_texture_set else {
            return;
        };
        let Some(set) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)
        else {
            return;
        };
        let index = self
            .selected_layer
            .and_then(|selected| set.layers.iter().position(|layer| layer.id == selected))
            .map_or(set.layers.len(), |position| position + 1);
        let layer_id = layer.id;
        if self.history.apply(
            &mut self.project,
            EditCommand::InsertLayer {
                texture_set_id,
                index,
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
        let brush = self.brush_color.to_array();
        let fill = LayerFill {
            color: brush,
            roughness: self.brush_roughness,
            metallic: self.brush_metallic,
            height: self.brush_height,
        };
        self.insert_layer(MaterialLayer::new_fill(
            id,
            format!("Fill Layer {}", id.0),
            fill,
        ));
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
                },
            ),
            MaskGenerator::Dirt(_) => (
                "Dirt",
                LayerFill {
                    color: [62, 50, 38, 255],
                    roughness: 0.92,
                    metallic: 0.0,
                    height: DEFAULT_HEIGHT,
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

    fn selected_layer_index(&self) -> Option<(TextureSetId, usize, usize)> {
        let texture_set_id = self.selected_texture_set?;
        let layer_id = self.selected_layer?;
        let set = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)?;
        let index = set.layers.iter().position(|layer| layer.id == layer_id)?;
        Some((texture_set_id, index, set.layers.len()))
    }

    fn delete_selected_layer(&mut self) {
        self.finish_active_stroke();
        let Some((texture_set_id, index, _)) = self.selected_layer_index() else {
            return;
        };
        let Some(layer) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)
            .and_then(|set| set.layers.get(index))
            .cloned()
        else {
            return;
        };
        if self.history.apply(
            &mut self.project,
            EditCommand::RemoveLayer {
                texture_set_id,
                index,
                layer: Box::new(layer),
            },
        ) {
            self.selected_layer = self
                .project
                .texture_sets
                .iter()
                .find(|set| set.id == texture_set_id)
                .and_then(|set| set.layers.get(index.saturating_sub(1)))
                .map(|layer| layer.id);
            self.layer_settled = None;
            self.invalidate_preview();
        }
    }

    /// Moves the selected layer up (+1, toward the top of the stack) or down.
    fn move_selected_layer(&mut self, delta: isize) {
        let Some((texture_set_id, index, count)) = self.selected_layer_index() else {
            return;
        };
        let Some(target) = index
            .checked_add_signed(delta)
            .filter(|target| *target < count)
        else {
            return;
        };
        if self.history.apply(
            &mut self.project,
            EditCommand::MoveLayer {
                texture_set_id,
                from: index,
                to: target,
            },
        ) {
            self.invalidate_preview();
        }
    }

    fn selected_layer_mut(&mut self) -> Option<&mut MaterialLayer> {
        let texture_set_id = self.selected_texture_set?;
        let layer_id = self.selected_layer?;
        self.project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)?
            .layers
            .iter_mut()
            .find(|layer| layer.id == layer_id)
    }

    fn selected_layer(&self) -> Option<&MaterialLayer> {
        let texture_set_id = self.selected_texture_set?;
        let layer_id = self.selected_layer?;
        self.project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)?
            .layers
            .iter()
            .find(|layer| layer.id == layer_id)
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

    fn append_stroke_point(&mut self, uv: glam::Vec2) {
        let (Some(texture_set_id), Some(layer_id)) =
            (self.selected_texture_set, self.selected_layer)
        else {
            return;
        };
        if !self.paint_mode.paints() {
            return;
        }
        let stroke_key = (texture_set_id, layer_id, self.paint_mode);
        let starts_stroke = self.active_stroke != Some(stroke_key);
        let color = self.brush_color.to_array();
        let roughness = self.brush_roughness;
        let metallic = self.brush_metallic;
        let Some(layer) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)
            .and_then(|set| set.layers.iter_mut().find(|layer| layer.id == layer_id))
        else {
            return;
        };

        if starts_stroke {
            match self.paint_mode {
                PaintMode::View | PaintMode::Select => return,
                PaintMode::Paint => layer.paint_strokes.push(PaintStroke {
                    points: vec![uv],
                    radius_pixels: self.brush_radius,
                    color,
                    opacity: 1.0,
                    roughness,
                    metallic,
                    height: self.brush_height,
                }),
                PaintMode::Mask => layer.mask_strokes.push(MaskStroke {
                    points: vec![uv],
                    radius_pixels: self.brush_radius,
                    value: self.mask_value,
                    opacity: 1.0,
                }),
                PaintMode::Id => layer.id_strokes.push(IdStroke {
                    points: vec![uv],
                    radius_pixels: self.brush_radius,
                    color: [color[0], color[1], color[2]],
                    opacity: 1.0,
                }),
            }
            self.active_stroke = Some(stroke_key);
            self.invalidate_preview();
            return;
        }

        let points = match self.paint_mode {
            PaintMode::View | PaintMode::Select => return,
            PaintMode::Paint => layer
                .paint_strokes
                .last_mut()
                .map(|stroke| &mut stroke.points),
            PaintMode::Mask => layer
                .mask_strokes
                .last_mut()
                .map(|stroke| &mut stroke.points),
            PaintMode::Id => layer.id_strokes.last_mut().map(|stroke| &mut stroke.points),
        };
        if let Some(points) = points
            && points
                .last()
                .is_none_or(|last| last.distance_squared(uv) >= 0.000_001)
        {
            points.push(uv);
            self.preview_revision = self.preview_revision.wrapping_add(1);
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
            .and_then(|set| set.layers.iter().find(|layer| layer.id == layer_id))
        else {
            return;
        };
        let command = match mode {
            PaintMode::View | PaintMode::Select => None,
            PaintMode::Paint => {
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
        self.invalidate_preview();
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
        self.fresh_mesh_maps().is_some()
    }

    fn fresh_mesh_maps(&self) -> Option<&MeshMaps> {
        self.mesh_maps
            .as_deref()
            .filter(|_| self.mesh_maps_key == Some(self.bake_key()))
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
                let _ = sender.send(bake_mesh_maps_from(
                    &mesh,
                    high.as_ref(),
                    &settings,
                    Some(&worker_cancel),
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
                        self.invalidate_preview();
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
        self.invalidate_preview();
    }

    fn pick_high_poly(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("FBX mesh", &["fbx"])
            .pick_file()
        else {
            return;
        };
        match load_merged_fbx(&path, self.import_scale) {
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
            match load_merged_fbx(&path, self.import_scale) {
                Ok(geometry) => self.high_poly = Some(HighPolySource { path, geometry }),
                Err(error) => {
                    self.mesh_map_error = Some(format!("High poly `{}`: {error}", path.display()));
                }
            }
        }
        self.high_poly_changed();
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
                    textures: PreviewTextures::Single(texture),
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
        let maps = self
            .mesh_maps
            .clone()
            .filter(|_| self.mesh_maps_key == Some(self.bake_key()));
        let mesh = self.mesh_geometry.clone();
        let (sender, receiver) = sync_channel(1);
        let spawned = std::thread::Builder::new()
            .name("preview-bake".to_owned())
            .spawn(move || {
                let mesh = mesh.as_deref();
                let maps = maps.as_deref();
                let result = match key.target {
                    PreviewTarget::Channel(channel) => {
                        bake_texture_set_channel(&texture_set, channel, mesh, maps)
                            .map(PreviewTextures::Single)
                    }
                    _ => bake_texture_set_material(&texture_set, mesh, maps)
                        .map(|textures| PreviewTextures::Material(Box::new(textures))),
                };
                let _ = sender.send(result.map_err(|error| error.to_string()));
            });
        match spawned {
            Ok(_) => self.preview_job = Some(PreviewJob { receiver, key }),
            Err(error) => {
                self.preview_failed = true;
                self.bake_error = Some(error.to_string());
            }
        }
    }

    fn poll_preview_job(&mut self, context: &egui::Context) {
        let Some(job) = &self.preview_job else {
            return;
        };
        match job.receiver.try_recv() {
            Ok(result) => {
                let key = job.key;
                self.preview_job = None;
                match result {
                    Ok(textures) => {
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

    fn export_baked_map(&mut self, channel: MaterialChannel) {
        let Some(texture_set) = self.exportable_texture_set() else {
            return;
        };
        let map_name = channel_label(channel);
        let path = rfd::FileDialog::new()
            .add_filter("Targa texture", &["tga"])
            .set_file_name(format!("{}_{}.tga", texture_set.name, map_name).replace(' ', "_"))
            .save_file();
        let Some(path) = path else {
            return;
        };
        let maps = self.fresh_mesh_maps();
        let result =
            bake_texture_set_channel(&texture_set, channel, self.mesh_geometry.as_deref(), maps)
                .map_err(|error| error.to_string())
                .and_then(|texture| write_tga(&path, &texture));
        match result {
            Ok(()) => self.bake_error = None,
            Err(error) => self.bake_error = Some(error),
        }
    }

    /// Writes `<set>_BaseColor.tga`, `<set>_ORM.tga`, and `<set>_Normal.tga`
    /// into a chosen folder: the common metal/rough engine layout.
    fn export_material_set(&mut self) {
        let Some(texture_set) = self.exportable_texture_set() else {
            return;
        };
        let Some(folder) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        let maps = self.fresh_mesh_maps();
        let result = bake_texture_set_material(&texture_set, self.mesh_geometry.as_deref(), maps)
            .map_err(|error| error.to_string())
            .and_then(|textures| {
                let stem = texture_set.name.replace(' ', "_");
                for (suffix, texture) in [
                    ("BaseColor", &textures.base_color),
                    ("ORM", &textures.orm),
                    ("Normal", &textures.normal),
                ] {
                    write_tga(&folder.join(format!("{stem}_{suffix}.tga")), texture)?;
                }
                Ok(())
            });
        match result {
            Ok(()) => self.bake_error = None,
            Err(error) => self.bake_error = Some(error),
        }
    }

    fn export_mesh_map(&mut self, kind: MeshMapKind) {
        let Some(maps) = self.mesh_maps.clone().filter(|_| self.mesh_maps_fresh()) else {
            self.ensure_mesh_maps();
            return;
        };
        let path = rfd::FileDialog::new()
            .add_filter("Targa texture", &["tga"])
            .set_file_name(format!("{}.tga", kind.name().replace(' ', "_")))
            .save_file();
        let Some(path) = path else {
            return;
        };
        if let Err(error) = write_tga(&path, &maps.to_texture(kind)) {
            self.mesh_map_error = Some(error);
        }
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
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(theme::VIEWPORT_BG))
            .show(ui, |ui| {
                let viewport = ui.max_rect();
                let response = ui.interact(
                    viewport,
                    ui.id().with("model_viewport"),
                    egui::Sense::click_and_drag(),
                );
                let was_stroke_active = self.active_stroke.is_some();
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
                } else if !response.dragged() {
                    self.finish_active_stroke();
                    if was_stroke_active {
                        self.invalidate_preview();
                    }
                }
                // Runs during strokes too: previews bake in the background, so the
                // stroke in progress shows up live.
                self.refresh_baked_preview();
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
                let needs_projection = software
                    || (painting && response.dragged())
                    || (selecting && response.hovered());
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
                        (Some(hit), Some(topology)) => {
                            topology.select(hit.triangle, self.selection_mode)
                        }
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
                    let radius =
                        (self.brush_radius * self.viewport_camera.zoom * 0.35).clamp(3.0, 36.0);
                    painter.circle(
                        pointer,
                        radius,
                        cursor_color.gamma_multiply(0.12),
                        egui::Stroke::new(1.5, cursor_color),
                    );
                    painter.circle_filled(pointer, 1.5, cursor_color);
                }
            });
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

/// Reads every mesh in a binary FBX and merges them into one scaled
/// geometry, as a high-poly source for baking.
fn load_merged_fbx(path: &std::path::Path, scale: f32) -> Result<MeshGeometry, String> {
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
    let meshes = import_binary_fbx_meshes(&bytes).map_err(|error| error.to_string())?;
    let mut merged = MeshGeometry {
        vertices: Vec::new(),
        triangles: Vec::new(),
        ..MeshGeometry::default()
    };
    for mesh in meshes {
        let offset = u32::try_from(merged.vertices.len())
            .map_err(|_| "merged FBX has too many vertices".to_owned())?;
        merged.vertices.extend(mesh.geometry.vertices);
        merged.triangles.extend(
            mesh.geometry
                .triangles
                .into_iter()
                .map(|triangle| triangle.map(|index| index + offset)),
        );
    }
    if merged.triangles.is_empty() {
        return Err("FBX contains no mesh geometry".to_owned());
    }
    Ok(scaled_geometry(&merged, sanitize_mesh_scale(scale)))
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

fn write_tga(path: &std::path::Path, texture: &BakedTexture) -> Result<(), String> {
    let bytes = encode_tga(texture)?;
    std::fs::write(path, bytes).map_err(|error| error.to_string())
}

fn encode_tga(texture: &BakedTexture) -> Result<Vec<u8>, String> {
    let width = u16::try_from(texture.width)
        .map_err(|_| "TGA width exceeds the format limit".to_owned())?;
    let height = u16::try_from(texture.height)
        .map_err(|_| "TGA height exceeds the format limit".to_owned())?;
    let expected_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "TGA pixel count overflow".to_owned())?;
    if texture.rgba8.len() != expected_bytes {
        return Err("TGA RGBA buffer does not match its dimensions".to_owned());
    }
    let capacity = 18_usize
        .checked_add(texture.rgba8.len())
        .ok_or_else(|| "TGA output size overflow".to_owned())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| error.to_string())?;
    let mut header = [0_u8; 18];
    header[2] = 2;
    header[12..14].copy_from_slice(&width.to_le_bytes());
    header[14..16].copy_from_slice(&height.to_le_bytes());
    header[16] = 32;
    header[17] = 0x28;
    bytes.extend_from_slice(&header);
    let (pixels, remainder) = texture.rgba8.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err("TGA RGBA buffer has an incomplete pixel".to_owned());
    }
    for [red, green, blue, alpha] in pixels {
        bytes.extend_from_slice(&[*blue, *green, *red, *alpha]);
    }
    Ok(bytes)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

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
            app.refresh_baked_preview();
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

    #[test]
    fn tga_export_writes_top_origin_rgba_as_bgra() {
        let texture = BakedTexture {
            width: 1,
            height: 1,
            rgba8: vec![10, 20, 30, 40],
        };

        let encoded = encode_tga(&texture).unwrap();

        assert_eq!(encoded[2], 2);
        assert_eq!(&encoded[12..16], &[1, 0, 1, 0]);
        assert_eq!(&encoded[16..18], &[32, 0x28]);
        assert_eq!(&encoded[18..], &[30, 20, 10, 40]);
    }
}
