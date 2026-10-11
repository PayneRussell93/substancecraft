#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use eframe::egui;
use substancecraft_api::{
    ImportedFbxMesh, MAX_FBX_BYTES, import_binary_fbx_meshes, load_project_file, save_project_file,
};
use substancecraft_core::{
    BoneId, BoneTransform, MaterialChannel, MaterialProject, MeshAsset, MeshGeometry, MeshId,
    NamedPose, PoseId, PrimitivePreset, TextureResolution, TextureSet, TextureSetId,
};

const CHANNELS: [MaterialChannel; 8] = [
    MaterialChannel::BaseColor,
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
        Box::new(|_creation_context| Ok(Box::<SubstanceCraftApp>::default())),
    )
}

struct SubstanceCraftApp {
    project: MaterialProject,
    selected_texture_set: Option<TextureSetId>,
    selected_primitive: Option<PrimitivePreset>,
    mesh_geometry: Option<MeshGeometry>,
    imported_meshes: Vec<ImportedFbxMesh>,
    selected_imported_mesh: Option<usize>,
    import_error: Option<String>,
    project_path: Option<std::path::PathBuf>,
    project_error: Option<String>,
    active_channel: MaterialChannel,
    next_texture_set_id: u64,
    current_pose: NamedPose,
    selected_pose: Option<PoseId>,
    pose_name: String,
    pose_transition: Option<PoseTransition>,
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
        Self {
            project: MaterialProject {
                mesh: Some(MeshAsset {
                    id: MeshId(1),
                    source: "builtin:Cube".to_owned(),
                }),
                ..MaterialProject::empty()
            },
            selected_texture_set: None,
            selected_primitive: Some(PrimitivePreset::Cube),
            mesh_geometry: Some(PrimitivePreset::Cube.build_mesh()),
            imported_meshes: Vec::new(),
            selected_imported_mesh: None,
            import_error: None,
            project_path: None,
            project_error: None,
            active_channel: MaterialChannel::BaseColor,
            next_texture_set_id: 1,
            current_pose: NamedPose {
                id: PoseId(0),
                name: "Current Pose".to_owned(),
                bones: BTreeMap::from([(BoneId(0), BoneTransform::IDENTITY)]),
            },
            selected_pose: None,
            pose_name: "Pose 1".to_owned(),
            pose_transition: None,
        }
    }
}

impl SubstanceCraftApp {
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
        self.next_texture_set_id = self
            .project
            .texture_sets
            .iter()
            .map(|set| set.id.0)
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
        self.selected_pose = self.project.saved_poses.first().map(|pose| pose.id);
        self.pose_name = format!("Pose {}", self.project.saved_poses.len() + 1);
        self.pose_transition = None;
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
            channels: BTreeSet::from([MaterialChannel::BaseColor]),
            layers: Vec::new(),
        });
        self.project.active_texture_set = Some(id);
        self.selected_texture_set = Some(id);
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
                    for texture_set in &self.project.texture_sets {
                        let selected = self.selected_texture_set == Some(texture_set.id);
                        if ui.selectable_label(selected, &texture_set.name).clicked() {
                            self.selected_texture_set = Some(texture_set.id);
                            self.project.active_texture_set = Some(texture_set.id);
                        }
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
                ui.heading("Properties");
                ui.add_space(12.0);
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
                } else {
                    ui.label("Nothing selected");
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
            });
    }

    fn update_viewport(&self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
            let viewport = ui.max_rect();
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
                let scale = viewport.width().min(viewport.height()) * 0.42;
                let project = |vertex_index: u32| {
                    let vertex = mesh.vertices.get(vertex_index as usize)?;
                    let transformed = pose_transform
                        .rotation
                        .mul_vec3(vertex.position * pose_transform.scale)
                        + pose_transform.translation;
                    Some((
                        egui::pos2(
                            center.x + (transformed.x - transformed.z) * scale * 0.72,
                            center.y
                                + (-transformed.y + (transformed.x + transformed.z) * 0.38) * scale,
                        ),
                        transformed.x + transformed.y + transformed.z,
                    ))
                };
                let mut faces = mesh
                    .triangles
                    .iter()
                    .filter_map(|triangle| {
                        let a = project(triangle[0])?;
                        let b = project(triangle[1])?;
                        let c = project(triangle[2])?;
                        Some(((a.1 + b.1 + c.1) / 3.0, [a.0, b.0, c.0]))
                    })
                    .collect::<Vec<_>>();
                faces.sort_by(|left, right| left.0.total_cmp(&right.0));
                for (_, points) in faces {
                    painter.add(egui::Shape::convex_polygon(
                        points.to_vec(),
                        egui::Color32::from_rgb(95, 109, 120),
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
        MaterialChannel::Roughness => "Roughness",
        MaterialChannel::Metallic => "Metallic",
        MaterialChannel::Normal => "Normal",
        MaterialChannel::Height => "Height",
        MaterialChannel::AmbientOcclusion => "Ambient Occlusion",
        MaterialChannel::Emissive => "Emissive",
        MaterialChannel::Opacity => "Opacity",
    }
}
