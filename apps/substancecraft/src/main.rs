#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, SyncSender, sync_channel},
};
use std::time::Duration;

use eframe::egui;
use substancecraft_api::{
    ImportedFbxMesh, MAX_FBX_BYTES, bake_texture_set_base_color, bake_texture_set_id_mask,
    import_binary_fbx_meshes, load_project_file, save_project_file,
};
use substancecraft_core::{
    BakedTexture, BoneId, BoneTransform, IdStroke, MaskStroke, MaterialChannel, MaterialLayer,
    MaterialProject, MeshAsset, MeshGeometry, MeshId, NamedPose, PaintStroke, PoseId,
    PrimitivePreset, StrokeCommand, StrokeHistory, TextureResolution, TextureSet, TextureSetId,
    ViewportCamera, ViewportLighting,
};
use substancecraft_filament::FilamentRuntime;

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
const POSE_TRANSITION_SECONDS: f64 = 0.35;
const MAX_SAVED_POSES: usize = 256;

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
        Box::new(|_creation_context| {
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
    selected_layer: Option<substancecraft_core::LayerId>,
    selected_primitive: Option<PrimitivePreset>,
    mesh_geometry: Option<MeshGeometry>,
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
    mask_value: f32,
    active_stroke: Option<(TextureSetId, substancecraft_core::LayerId, PaintMode)>,
    stroke_history: StrokeHistory,
    baked_preview: Option<(TextureSetId, MaterialChannel, BakedTexture)>,
    bake_error: Option<String>,
    selected_pose: Option<PoseId>,
    pose_name: String,
    pose_transition: Option<PoseTransition>,
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
}

impl Default for SubstanceCraftApp {
    fn default() -> Self {
        let (filament_frames_tx, filament_frames_rx) = sync_channel(1);
        Self {
            project: MaterialProject {
                mesh: Some(MeshAsset {
                    id: MeshId(1),
                    source: "builtin:Cube".to_owned(),
                }),
                ..MaterialProject::empty()
            },
            selected_texture_set: None,
            selected_layer: None,
            selected_primitive: Some(PrimitivePreset::Cube),
            mesh_geometry: Some(PrimitivePreset::Cube.build_mesh()),
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
            mask_value: 0.0,
            active_stroke: None,
            stroke_history: StrokeHistory::default(),
            baked_preview: None,
            bake_error: None,
            selected_pose: None,
            pose_name: "Pose 1".to_owned(),
            pose_transition: None,
        }
    }
}

impl SubstanceCraftApp {
    fn mark_mesh_changed(&mut self) {
        self.mesh_revision = self.mesh_revision.wrapping_add(1);
        self.filament_frame_texture = None;
    }

    fn update_filament_frame(&mut self, context: &egui::Context, viewport: egui::Rect) {
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

        let Some(mesh) = self.mesh_geometry.as_ref() else {
            self.filament_frame_texture = None;
            return;
        };
        let Some(renderer) = self.filament_renderer.as_mut() else {
            return;
        };
        if self.filament_mesh_revision != Some(self.mesh_revision) {
            match renderer.set_mesh(mesh) {
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
            .filter(|(set_id, channel, _)| {
                Some(*set_id) == self.selected_texture_set
                    && *channel == self.active_channel
                    && matches!(channel, MaterialChannel::BaseColor | MaterialChannel::Id)
            })
            .map(|(_, _, texture)| texture);
        let texture_result = if let Some(texture) = preview {
            let key = (texture.rgba8.as_ptr() as usize)
                ^ (texture.width as usize).rotate_left(7)
                ^ (texture.height as usize).rotate_left(17);
            renderer.set_base_color_map(key, texture.width, texture.height, &texture.rgba8)
        } else {
            renderer.set_base_color_map(0, 1, 1, &[168, 173, 179, 255])
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
        match renderer.render(
            self.viewport_camera,
            width,
            height,
            move |rgba8| {
                pending.store(false, Ordering::Release);
                let _ = sender.try_send(FilamentFrame {
                    width,
                    height,
                    rgba8,
                });
            },
        ) {
            Ok(true) => {
                self.filament_error = None;
                context.request_repaint_after(Duration::from_millis(16));
            }
            Ok(false) => {
                self.filament_render_pending
                    .store(false, Ordering::Release);
                context.request_repaint_after(Duration::from_millis(16));
            }
            Err(error) => {
                self.filament_render_pending
                    .store(false, Ordering::Release);
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
        self.mesh_geometry = scene.geometry;
        self.selected_texture_set = self.project.active_texture_set;
        self.selected_layer = self
            .project
            .texture_sets
            .iter()
            .find(|set| Some(set.id) == self.selected_texture_set)
            .and_then(|set| set.layers.first())
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
        self.stroke_history = StrokeHistory::default();
        self.baked_preview = None;
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
            });
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
        let first_geometry = first_mesh.geometry.clone();
        Ok(RestoredMeshScene {
            selected_primitive: None,
            imported_meshes: meshes,
            selected_imported_mesh: Some(0),
            geometry: Some(first_geometry),
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
                let first_geometry = first_mesh.geometry.clone();
                let display_name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "Imported FBX".to_owned());
                self.project.mesh = Some(MeshAsset {
                    id: MeshId(1),
                    source: path.to_string_lossy().into_owned(),
                });
                self.imported_meshes = meshes;
                self.selected_imported_mesh = Some(0);
                self.selected_primitive = None;
                self.mesh_geometry = Some(first_geometry);
                self.import_error = None;
                self.project.texture_sets.clear();
                self.project.active_texture_set = None;
                self.selected_texture_set = None;
                self.selected_layer = None;
                self.active_stroke = None;
                self.baked_preview = None;
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
            channels: BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Id]),
            layers: Vec::new(),
        });
        self.project.active_texture_set = Some(id);
        self.selected_texture_set = Some(id);
        self.selected_layer = None;
        self.baked_preview = None;
    }

    fn add_paint_layer(&mut self) {
        let Some(texture_set_id) = self.selected_texture_set else {
            return;
        };
        let Some(next_id) = self.next_layer_id.checked_add(1) else {
            return;
        };
        let id = substancecraft_core::LayerId(self.next_layer_id);
        let Some(texture_set) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)
        else {
            return;
        };
        texture_set.layers.push(MaterialLayer {
            id,
            name: format!("Paint Layer {}", id.0),
            visible: true,
            opacity: 1.0,
            channels: BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Id]),
            paint_strokes: Vec::new(),
            mask_strokes: Vec::new(),
            id_strokes: Vec::new(),
            id_mask_color: None,
        });
        self.next_layer_id = next_id;
        self.selected_layer = Some(id);
        self.baked_preview = None;
    }

    fn append_stroke_point(&mut self, uv: glam::Vec2) {
        let (Some(texture_set_id), Some(layer_id)) =
            (self.selected_texture_set, self.selected_layer)
        else {
            return;
        };
        if self.paint_mode == PaintMode::View {
            return;
        }
        let stroke_key = (texture_set_id, layer_id, self.paint_mode);
        let starts_stroke = self.active_stroke != Some(stroke_key);
        let color = self.brush_color.to_array();
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
                PaintMode::View => return,
                PaintMode::Paint => layer.paint_strokes.push(PaintStroke {
                    points: vec![uv],
                    radius_pixels: self.brush_radius,
                    color,
                    opacity: 1.0,
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
            return;
        }

        let points = match self.paint_mode {
            PaintMode::View => return,
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
            PaintMode::View => None,
            PaintMode::Paint => {
                layer
                    .paint_strokes
                    .last()
                    .cloned()
                    .map(|stroke| StrokeCommand::Paint {
                        texture_set_id,
                        layer_id,
                        stroke,
                    })
            }
            PaintMode::Mask => {
                layer
                    .mask_strokes
                    .last()
                    .cloned()
                    .map(|stroke| StrokeCommand::Mask {
                        texture_set_id,
                        layer_id,
                        stroke,
                    })
            }
            PaintMode::Id => layer
                .id_strokes
                .last()
                .cloned()
                .map(|stroke| StrokeCommand::Id {
                    texture_set_id,
                    layer_id,
                    stroke,
                }),
        };
        if let Some(command) = command {
            self.stroke_history.record(command);
        }
    }

    fn undo_stroke(&mut self) {
        self.finish_active_stroke();
        if self.stroke_history.undo(&mut self.project) {
            self.baked_preview = None;
            self.refresh_baked_preview();
        }
    }

    fn redo_stroke(&mut self) {
        self.finish_active_stroke();
        if self.stroke_history.redo(&mut self.project) {
            self.baked_preview = None;
            self.refresh_baked_preview();
        }
    }

    fn refresh_baked_preview(&mut self) {
        let Some(texture_set_id) = self.selected_texture_set else {
            self.baked_preview = None;
            return;
        };
        if !matches!(
            self.active_channel,
            MaterialChannel::BaseColor | MaterialChannel::Id
        ) {
            return;
        }
        if self
            .baked_preview
            .as_ref()
            .is_some_and(|(set_id, channel, _)| {
                *set_id == texture_set_id && *channel == self.active_channel
            })
        {
            return;
        }
        let Some(texture_set) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)
        else {
            self.baked_preview = None;
            return;
        };
        let baked = match self.active_channel {
            MaterialChannel::BaseColor => bake_texture_set_base_color(texture_set),
            MaterialChannel::Id => bake_texture_set_id_mask(texture_set),
            _ => return,
        };
        match baked {
            Ok(texture) => {
                self.baked_preview = Some((texture_set_id, self.active_channel, texture));
                self.bake_error = None;
            }
            Err(error) => {
                self.baked_preview = None;
                self.bake_error = Some(error.to_string());
            }
        }
    }

    fn export_baked_map(&mut self, channel: MaterialChannel) {
        let Some(texture_set) = self
            .project
            .texture_sets
            .iter()
            .find(|set| Some(set.id) == self.selected_texture_set)
            .cloned()
        else {
            return;
        };
        let map_name = if channel == MaterialChannel::Id {
            "ID Mask"
        } else {
            "Base Color"
        };
        let path = rfd::FileDialog::new()
            .add_filter("Targa texture", &["tga"])
            .set_file_name(format!("{}.tga", map_name.replace(' ', "_")))
            .save_file();
        let Some(path) = path else {
            return;
        };
        let result = match channel {
            MaterialChannel::BaseColor => bake_texture_set_base_color(&texture_set),
            MaterialChannel::Id => bake_texture_set_id_mask(&texture_set),
            _ => return,
        }
        .map_err(|error| error.to_string())
        .and_then(|texture| write_tga(&path, &texture));
        match result {
            Ok(()) => {
                self.bake_error = None;
                self.active_channel = channel;
                self.baked_preview = None;
                self.refresh_baked_preview();
            }
            Err(error) => self.bake_error = Some(error),
        }
    }

    fn update_header(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("header").exact_size(42.0).show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                ui.strong("SubstanceCraft");
                ui.separator();
                let project_name = self
                    .project_path
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "Untitled Project".to_owned());
                ui.label(project_name);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(self.stroke_history.can_redo(), egui::Button::new("Redo"))
                        .clicked()
                    {
                        self.redo_stroke();
                    }
                    if ui
                        .add_enabled(self.stroke_history.can_undo(), egui::Button::new("Undo"))
                        .clicked()
                    {
                        self.undo_stroke();
                    }
                    if ui.button("Save").clicked() {
                        self.save_project();
                    }
                    if ui.button("Open").clicked() {
                        self.open_project();
                    }
                    if let Some(error) = &self.project_error {
                        ui.colored_label(egui::Color32::LIGHT_RED, error);
                    }
                    ui.label(self.selected_mesh_name().unwrap_or("No mesh loaded"));
                });
            });
        });
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
        self.mesh_geometry = Some(mesh.geometry.clone());
        self.selected_imported_mesh = Some(index);
        self.selected_primitive = None;
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

    fn update_pose_shelf(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("pose_shelf")
            .exact_size(92.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Saved Poses");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.pose_name)
                            .desired_width(150.0)
                            .hint_text("Pose name"),
                    );
                    let can_save = !self.pose_name.trim().is_empty()
                        && self.project.saved_poses.len() < MAX_SAVED_POSES;
                    if ui
                        .add_enabled(can_save, egui::Button::new("Save Pose"))
                        .clicked()
                    {
                        self.save_pose();
                    }
                });
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    if self.project.saved_poses.is_empty() {
                        ui.weak("Save a pose to add it to this shelf");
                    }
                    for index in 0..self.project.saved_poses.len() {
                        let Some(pose) = self.project.saved_poses.get(index) else {
                            continue;
                        };
                        let pose_id = pose.id;
                        let name = pose.name.clone();
                        let selected = self.selected_pose == Some(pose_id);
                        if ui.selectable_label(selected, name).clicked() {
                            self.select_pose(pose_id, ui.ctx());
                        }
                    }
                });
            });
    }

    fn update_texture_sets(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("texture_sets")
            .exact_size(230.0)
            .show(ui, |ui| {
                ui.heading("Texture Sets");
                ui.add_space(12.0);
                if ui.button("Import FBX...").clicked() {
                    self.import_fbx();
                }
                if let Some(error) = &self.import_error {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
                if !self.imported_meshes.is_empty() {
                    ui.add_space(8.0);
                    ui.heading("Imported Meshes");
                    for index in 0..self.imported_meshes.len() {
                        let Some(mesh) = self.imported_meshes.get(index) else {
                            continue;
                        };
                        let name = mesh.name.clone();
                        let selected = self.selected_imported_mesh == Some(index);
                        if ui.selectable_label(selected, name).clicked() {
                            self.select_imported_mesh(index);
                        }
                    }
                }
                ui.add_space(12.0);
                ui.heading("Mesh Presets");
                for preset in PrimitivePreset::ALL {
                    if ui
                        .selectable_label(self.selected_primitive == Some(preset), preset.name())
                        .clicked()
                    {
                        self.selected_primitive = Some(preset);
                        self.selected_imported_mesh = None;
                        self.mesh_geometry = Some(preset.build_mesh());
                        self.project.mesh = Some(MeshAsset {
                            id: MeshId(1),
                            source: format!("builtin:{}", preset.name()),
                        });
                    }
                }
                ui.add_space(8.0);
                if ui.button("+  Add Texture Set").clicked() {
                    self.add_texture_set();
                }
                ui.add_space(8.0);

                if self.project.texture_sets.is_empty() {
                    ui.label("No texture sets");
                } else {
                    let texture_sets = self
                        .project
                        .texture_sets
                        .iter()
                        .map(|set| (set.id, set.name.clone()))
                        .collect::<Vec<_>>();
                    for (set_id, name) in texture_sets {
                        let selected = self.selected_texture_set == Some(set_id);
                        if ui.selectable_label(selected, name).clicked() {
                            self.selected_texture_set = Some(set_id);
                            self.project.active_texture_set = Some(set_id);
                            self.selected_layer = self
                                .project
                                .texture_sets
                                .iter()
                                .find(|set| set.id == set_id)
                                .and_then(|set| set.layers.first())
                                .map(|layer| layer.id);
                            self.baked_preview = None;
                        }
                    }
                }

                if let Some(texture_set_id) = self.selected_texture_set {
                    ui.add_space(12.0);
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.heading("Layers");
                        if ui.small_button("+").clicked() {
                            self.add_paint_layer();
                        }
                    });
                    let layers = self
                        .project
                        .texture_sets
                        .iter()
                        .find(|set| set.id == texture_set_id)
                        .map(|set| {
                            set.layers
                                .iter()
                                .map(|layer| (layer.id, layer.name.clone(), layer.visible))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    for (layer_id, name, mut visible) in layers {
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut visible, "").changed()
                                && let Some(layer) = self
                                    .project
                                    .texture_sets
                                    .iter_mut()
                                    .find(|set| set.id == texture_set_id)
                                    .and_then(|set| {
                                        set.layers.iter_mut().find(|layer| layer.id == layer_id)
                                    })
                            {
                                layer.visible = visible;
                                self.baked_preview = None;
                            }
                            if ui
                                .selectable_label(self.selected_layer == Some(layer_id), name)
                                .clicked()
                            {
                                self.selected_layer = Some(layer_id);
                            }
                        });
                    }
                }

                ui.add_space(18.0);
                ui.separator();
                ui.add_space(8.0);
                ui.heading("Channels");
                for channel in CHANNELS {
                    ui.selectable_value(&mut self.active_channel, channel, channel_label(channel));
                }
            });
    }

    fn update_properties(&mut self, ui: &mut egui::Ui) {
        egui::Panel::right("properties")
            .exact_size(250.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Properties");
                    ui.add_space(12.0);
                    let mut mode_changed = false;
                    ui.horizontal(|ui| {
                        mode_changed |= ui
                            .selectable_value(&mut self.paint_mode, PaintMode::View, "View")
                            .changed();
                        mode_changed |= ui
                            .selectable_value(&mut self.paint_mode, PaintMode::Paint, "Paint")
                            .changed();
                        mode_changed |= ui
                            .selectable_value(&mut self.paint_mode, PaintMode::Mask, "Mask")
                            .changed();
                        mode_changed |= ui
                            .selectable_value(&mut self.paint_mode, PaintMode::Id, "ID")
                            .changed();
                    });
                    if mode_changed {
                        self.finish_active_stroke();
                        self.baked_preview = None;
                        self.active_channel = match self.paint_mode {
                            PaintMode::Id => MaterialChannel::Id,
                            PaintMode::Paint | PaintMode::Mask => MaterialChannel::BaseColor,
                            PaintMode::View => self.active_channel,
                        };
                    }
                    if self.paint_mode != PaintMode::View {
                        ui.add(
                            egui::Slider::new(&mut self.brush_radius, 1.0..=128.0).text("Brush"),
                        );
                        if matches!(self.paint_mode, PaintMode::Paint | PaintMode::Id) {
                            ui.horizontal(|ui| {
                                ui.label("Color");
                                ui.color_edit_button_srgba(&mut self.brush_color);
                            });
                        }
                        if self.paint_mode == PaintMode::Mask {
                            ui.add(
                                egui::Slider::new(&mut self.mask_value, 0.0..=1.0)
                                    .text("Mask Value"),
                            );
                        }
                    }
                    if let Some(texture_set) = self
                        .project
                        .texture_sets
                        .iter()
                        .find(|set| Some(set.id) == self.selected_texture_set)
                    {
                        ui.label("Texture Set");
                        ui.strong(&texture_set.name);
                        ui.add_space(8.0);
                        ui.label("Material Slot");
                        ui.label(&texture_set.material_slot);
                        ui.add_space(8.0);
                        ui.label("Resolution");
                        ui.label(format!(
                            "{} x {}",
                            texture_set.resolution.width, texture_set.resolution.height
                        ));
                        ui.add_space(8.0);
                        ui.label("Layers");
                        ui.label(texture_set.layers.len().to_string());
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Bake Base Color").clicked() {
                                self.export_baked_map(MaterialChannel::BaseColor);
                            }
                            if ui.button("Bake ID Mask").clicked() {
                                self.export_baked_map(MaterialChannel::Id);
                            }
                        });
                    } else {
                        ui.label("Nothing selected");
                    }
                    if let (Some(texture_set_id), Some(layer_id)) =
                        (self.selected_texture_set, self.selected_layer)
                    {
                        let existing_mask = self
                            .project
                            .texture_sets
                            .iter()
                            .find(|set| set.id == texture_set_id)
                            .and_then(|set| set.layers.iter().find(|layer| layer.id == layer_id))
                            .and_then(|layer| layer.id_mask_color);
                        let mut enabled = existing_mask.is_some();
                        let color = existing_mask.unwrap_or_else(|| {
                            let brush = self.brush_color.to_array();
                            [brush[0], brush[1], brush[2]]
                        });
                        let mut color = egui::Color32::from_rgb(color[0], color[1], color[2]);
                        let mut changed = ui.checkbox(&mut enabled, "Clip by ID").changed();
                        if enabled {
                            changed |= ui.color_edit_button_srgba(&mut color).changed();
                        }
                        if changed
                            && let Some(layer) = self
                                .project
                                .texture_sets
                                .iter_mut()
                                .find(|set| set.id == texture_set_id)
                                .and_then(|set| {
                                    set.layers.iter_mut().find(|layer| layer.id == layer_id)
                                })
                        {
                            let channels = color.to_array();
                            layer.id_mask_color =
                                enabled.then_some([channels[0], channels[1], channels[2]]);
                            self.baked_preview = None;
                        }
                    }
                    if let Some(error) = &self.bake_error {
                        ui.colored_label(egui::Color32::LIGHT_RED, error);
                    }
                    ui.add_space(16.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.heading("Pose Transform");
                    ui.label("Root translation");
                    let transform = self
                        .current_pose
                        .bones
                        .entry(BoneId(0))
                        .or_insert(BoneTransform::IDENTITY);
                    let mut changed = false;
                    ui.horizontal(|ui| {
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut transform.translation.x)
                                    .prefix("X ")
                                    .speed(0.05),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut transform.translation.y)
                                    .prefix("Y ")
                                    .speed(0.05),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut transform.translation.z)
                                    .prefix("Z ")
                                    .speed(0.05),
                            )
                            .changed();
                    });
                    if changed {
                        self.pose_transition = None;
                        self.selected_pose = None;
                    }
                    ui.add_space(16.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.heading("Viewport");
                    if ui.button("Reset View").clicked() {
                        self.viewport_camera = ViewportCamera::default();
                    }
                    ui.add(
                        egui::Slider::new(
                            &mut self.viewport_camera.yaw,
                            0.0..=std::f32::consts::TAU,
                        )
                        .text("Orbit"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.viewport_camera.pitch, -1.55..=1.55)
                            .text("Tilt"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.viewport_camera.zoom, 0.1..=20.0)
                            .logarithmic(true)
                            .text("Zoom"),
                    );
                    ui.add_space(10.0);
                    ui.heading("Environment Light");
                    ui.add(
                        egui::Slider::new(
                            &mut self.viewport_lighting.rotation,
                            0.0..=std::f32::consts::TAU,
                        )
                        .text("Rotation"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.viewport_lighting.ambient_fill, 0.0..=1.0)
                            .text("Ambient Fill"),
                    );
                });
            });
    }

    fn update_viewport(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
            let viewport = ui.max_rect();
            let response = ui.interact(
                viewport,
                ui.id().with("model_viewport"),
                egui::Sense::click_and_drag(),
            );
            let was_stroke_active = self.active_stroke.is_some();
            let pointer_position = ui.input(|input| input.pointer.interact_pos());
            if response.dragged() && self.paint_mode == PaintMode::View {
                let delta = ui.input(|input| input.pointer.delta());
                let delta = glam::Vec2::new(delta.x, delta.y);
                if ui.input(|input| input.modifiers.shift) {
                    self.viewport_camera.pan_by(delta);
                } else {
                    self.viewport_camera.orbit(delta);
                }
            } else if !response.dragged() {
                self.finish_active_stroke();
                if was_stroke_active {
                    self.baked_preview = None;
                }
                self.refresh_baked_preview();
            }
            if response.hovered() && self.paint_mode == PaintMode::View {
                let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                self.viewport_camera.zoom_by(scroll);
            }
            let painter = ui.painter_at(viewport);
            painter.rect_filled(viewport, 0.0, egui::Color32::from_rgb(29, 31, 34));

            let center = viewport.center();
            let axis = egui::Color32::from_rgb(70, 74, 79);
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

            if let Some(mesh) = &self.mesh_geometry {
                let pose_transform = self
                    .current_pose
                    .bones
                    .get(&BoneId(0))
                    .copied()
                    .unwrap_or(BoneTransform::IDENTITY);
                let camera_rotation = self.viewport_camera.rotation();
                let scale =
                    viewport.width().min(viewport.height()) * 0.42 * self.viewport_camera.zoom;
                let project = |vertex_index: u32| {
                    let vertex = mesh.vertices.get(vertex_index as usize)?;
                    let posed = pose_transform
                        .rotation
                        .mul_vec3(vertex.position * pose_transform.scale)
                        + pose_transform.translation;
                    let view_position = camera_rotation * posed;
                    Some((
                        egui::pos2(
                            center.x + view_position.x * scale * 0.9 + self.viewport_camera.pan.x,
                            center.y - view_position.y * scale * 0.9 + self.viewport_camera.pan.y,
                        ),
                        view_position,
                    ))
                };
                let mut faces = mesh
                    .triangles
                    .iter()
                    .filter_map(|triangle| {
                        let vertex_a = mesh.vertices.get(triangle[0] as usize)?;
                        let vertex_b = mesh.vertices.get(triangle[1] as usize)?;
                        let vertex_c = mesh.vertices.get(triangle[2] as usize)?;
                        let a = project(triangle[0])?;
                        let b = project(triangle[1])?;
                        let c = project(triangle[2])?;
                        let normal = (b.1 - a.1).cross(c.1 - a.1).normalize_or_zero();
                        let intensity = self.viewport_lighting.face_intensity(normal);
                        Some((
                            (a.1.z + b.1.z + c.1.z) / 3.0,
                            [a.0, b.0, c.0],
                            [vertex_a.uv, vertex_b.uv, vertex_c.uv],
                            intensity,
                        ))
                    })
                    .collect::<Vec<_>>();
                faces.sort_by(|left, right| left.0.total_cmp(&right.0));

                if response.dragged() && self.paint_mode != PaintMode::View {
                    let hit_uv = pointer_position.and_then(|pointer| {
                        let mut candidates = faces.iter().filter_map(|(_, points, uvs, _)| {
                            barycentric_uv(pointer, *points, *uvs)
                        });
                        candidates.next_back()
                    });
                    if let Some(uv) = hit_uv {
                        self.append_stroke_point(uv);
                    }
                }

                for (_, points, uvs, intensity) in faces {
                    let uv_center = (uvs[0] + uvs[1] + uvs[2]) / 3.0;
                    let map_color = self
                        .baked_preview
                        .as_ref()
                        .filter(|(set_id, channel, _)| {
                            Some(*set_id) == self.selected_texture_set
                                && *channel == self.active_channel
                        })
                        .and_then(|(_, _, texture)| texture.sample_uv(uv_center))
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
                painter.text(
                    egui::pos2(viewport.left() + 16.0, viewport.top() + 16.0),
                    egui::Align2::LEFT_TOP,
                    self.selected_mesh_name().unwrap_or("Mesh"),
                    egui::FontId::proportional(13.0),
                    egui::Color32::from_rgb(190, 194, 199),
                );
                if self.paint_mode != PaintMode::View
                    && response.hovered()
                    && let Some(pointer) = pointer_position
                {
                    let cursor_color = if self.paint_mode == PaintMode::Mask {
                        egui::Color32::WHITE
                    } else {
                        self.brush_color
                    };
                    painter.circle_stroke(
                        pointer,
                        (self.brush_radius * self.viewport_camera.zoom * 0.35).clamp(3.0, 36.0),
                        egui::Stroke::new(1.5, cursor_color),
                    );
                }
            } else {
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    "Choose a mesh preset to begin",
                    egui::FontId::proportional(18.0),
                    egui::Color32::from_rgb(190, 194, 199),
                );
            }
            painter.text(
                egui::pos2(viewport.left() + 16.0, viewport.bottom() - 18.0),
                egui::Align2::LEFT_BOTTOM,
                channel_label(self.active_channel),
                egui::FontId::proportional(12.0),
                egui::Color32::from_rgb(148, 153, 159),
            );
        });
    }
}

fn barycentric_uv(
    point: egui::Pos2,
    screen: [egui::Pos2; 3],
    uvs: [glam::Vec2; 3],
) -> Option<glam::Vec2> {
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
    let uv = uvs[0] * weight_a + uvs[1] * weight_b + uvs[2] * weight_c;
    (uv.is_finite() && (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y)).then_some(uv)
}

impl eframe::App for SubstanceCraftApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.update_pose_transition(ui.ctx());
        self.update_header(ui);
        self.update_texture_sets(ui);
        self.update_properties(ui);
        self.update_pose_shelf(ui);
        self.update_viewport(ui);
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
