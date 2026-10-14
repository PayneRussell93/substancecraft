//! Floating pill overlays drawn on top of the full-bleed viewport.
//!
//! Layout: a brand pill and tool pill along the top, a vertical dock on the
//! left that slides out drawers, an inspector column of collapsible cards on
//! the right, and a status pill along the bottom.

use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontId, Id, Layout, Margin, Order, Rect, Response,
    RichText, Sense, Ui, pos2, vec2,
};
use substancecraft_api::TextureFormat;
use substancecraft_core::MIN_TILE_SIZE;
use substancecraft_core::{
    Axis, BitDepth, BitmapMask, BoneId, BoneTransform, BrushShape, ChannelSlot, Direction,
    ExportOutput, ExportSource, InstanceField, LayerFill, LayerFilter, LayerId, MAX_BLUR_RADIUS,
    MAX_BRUSH_SPACING, MAX_EXPORT_OUTPUTS, MAX_PROJECTION_SCALE, MAX_SUBDIVISION_LEVELS,
    MIN_BRUSH_SPACING, MIN_PROJECTION_SCALE, MaskGenerator, MaterialChannel, MaterialId,
    MaterialLayer, MaterialLink, MeshAsset, MeshId, MeshMapKind, MeshSmoothing, OutputLayout,
    PreflightSeverity, PrimitivePreset, ProjectionMode, ResourceId, SelectionMode, SourceComponent,
    TextureProjection, TextureSetId, ViewportCamera,
};

use super::resource_watch::ReloadMode;
use super::{
    BAKEABLE_CHANNELS, CHANNELS, MAX_MESH_SCALE, MAX_SAVED_POSES, MIN_MESH_SCALE, PaintMode,
    SubstanceCraftApp, ViewMode, channel_label,
};
use crate::icons::{self, Icon};
use crate::theme;
use crate::widgets;

const DRAWER_WIDTH: f32 = 284.0;
const INSPECTOR_WIDTH: f32 = 296.0;
/// Vertical space kept free above the bottom edge for the status pill.
const STATUS_RESERVE: f32 = 56.0;
const MODES: [(PaintMode, &str); 6] = [
    (PaintMode::View, "View"),
    (PaintMode::Paint, "Paint"),
    (PaintMode::Erase, "Erase"),
    (PaintMode::Mask, "Mask"),
    (PaintMode::Id, "ID"),
    (PaintMode::Select, "Select"),
];
const VIEW_MODES: [(ViewMode, &str); 3] = [
    (ViewMode::Model, "3D"),
    (ViewMode::Split, "Split"),
    (ViewMode::Uv, "UV"),
];
const MESH_MAP_RESOLUTIONS: [u32; 4] = [256, 512, 1024, 2048];
const WARNING: Color32 = Color32::from_rgb(240, 180, 80);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drawer {
    Scene,
    Brush,
    Assets,
    Bake,
    Poses,
    Environment,
    Preferences,
    Diagnostics,
}

impl Drawer {
    const ALL: [Self; 8] = [
        Self::Scene,
        Self::Brush,
        Self::Assets,
        Self::Bake,
        Self::Poses,
        Self::Environment,
        Self::Preferences,
        Self::Diagnostics,
    ];

    fn icon(self) -> Icon {
        match self {
            Self::Scene => Icon::Cube,
            Self::Brush => Icon::Brush,
            Self::Assets => Icon::Image,
            Self::Bake => Icon::Bake,
            Self::Poses => Icon::Figure,
            Self::Environment => Icon::Sun,
            Self::Preferences => Icon::Sliders,
            Self::Diagnostics => Icon::Alert,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Scene => "Scene",
            Self::Brush => "Tools",
            Self::Assets => "Assets",
            Self::Bake => "Bake",
            Self::Poses => "Poses",
            Self::Environment => "Light",
            Self::Preferences => "Prefs",
            Self::Diagnostics => "Checks",
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Scene => "Scene",
            Self::Brush => "Tools",
            Self::Assets => "Bitmaps",
            Self::Bake => "Mesh Maps",
            Self::Poses => "Poses",
            Self::Environment => "Camera & Light",
            Self::Preferences => "Preferences",
            Self::Diagnostics => "Diagnostics",
        }
    }
}

/// Panel sizes measured on the previous frame, used to place neighbours
/// without overlap.
pub struct OverlayLayout {
    toolbar_width: f32,
    dock_width: f32,
}

impl Default for OverlayLayout {
    fn default() -> Self {
        Self {
            toolbar_width: 520.0,
            dock_width: 66.0,
        }
    }
}

#[derive(Clone, Copy)]
enum ErrorSlot {
    Project,
    Import,
    Mesh,
    Bake,
    MeshMaps,
    Renderer,
}

impl SubstanceCraftApp {
    pub(super) fn update_overlay(&mut self, context: &egui::Context) {
        let screen = context.content_rect();
        let content_top = self.show_top_bars(context, screen);
        self.show_dock(context, screen, content_top);
        self.show_drawer(context, screen, content_top);
        if self.show_inspector {
            self.show_inspector_column(context, screen);
        }
        self.show_status(context, screen);
        self.show_toasts(context, screen);
        self.show_overwrite_prompt(context);
        self.show_recovery_prompt(context);
    }

    /// Asks before an export replaces files already in its folder.
    fn show_overwrite_prompt(&mut self, context: &egui::Context) {
        let Some(prompt) = &self.overwrite_prompt else {
            return;
        };
        let names: Vec<String> = prompt
            .existing
            .iter()
            .filter_map(|path| path.file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .collect();
        let folder = prompt.folder.display().to_string();
        let mut answer = None;
        let modal = egui::Modal::new(Id::new("export_overwrite")).show(context, |ui| {
            ui.set_max_width(420.0);
            ui.label(
                RichText::new(format!("Replace {} existing file(s)?", names.len()))
                    .size(15.0)
                    .color(theme::TEXT),
            );
            ui.add(
                egui::Label::new(RichText::new(folder).size(11.5).color(theme::TEXT_FAINT)).wrap(),
            );
            ui.add_space(4.0);
            for name in &names {
                ui.label(RichText::new(name).size(12.0).color(theme::TEXT_WEAK));
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if widgets::pill_button(ui, None, "Replace", true, false).clicked() {
                    answer = Some(true);
                }
                if widgets::pill_button(ui, None, "Cancel", false, false).clicked() {
                    answer = Some(false);
                }
            });
        });
        if modal.should_close() && answer.is_none() {
            answer = Some(false);
        }
        if let Some(overwrite) = answer {
            self.resolve_overwrite_prompt(overwrite);
        }
    }

    pub(super) fn set_paint_mode(&mut self, mode: PaintMode) {
        if self.paint_mode == mode {
            return;
        }
        self.paint_mode = mode;
        self.finish_active_stroke();
        self.hovered_selection.clear();
        let channel = match mode {
            PaintMode::Id => Some(MaterialChannel::Id),
            _ => None,
        };
        if let Some(channel) = channel {
            self.set_preview_channel(channel);
        } else if matches!(mode, PaintMode::Paint | PaintMode::Erase | PaintMode::Mask)
            && self.channel_preview
            && self.active_channel == MaterialChannel::Id
        {
            self.set_material_preview();
        }
        if mode != PaintMode::View && self.drawer.is_none() {
            self.drawer = Some(Drawer::Brush);
        }
    }

    fn set_preview_channel(&mut self, channel: MaterialChannel) {
        self.active_channel = channel;
        self.channel_preview = true;
        self.mesh_map_preview = None;
        self.invalidate_preview();
    }

    fn set_material_preview(&mut self) {
        self.channel_preview = false;
        self.mesh_map_preview = None;
        self.invalidate_preview();
    }

    fn preview_label(&self) -> &'static str {
        match self.mesh_map_preview {
            Some(kind) => kind.name(),
            None if self.channel_preview => channel_label(self.active_channel),
            None => "Material",
        }
    }

    pub(super) fn select_primitive(&mut self, preset: PrimitivePreset) {
        let before = self.project.mesh.clone();
        self.selected_primitive = Some(preset);
        self.selected_imported_mesh = None;
        self.project.mesh = Some(MeshAsset {
            id: MeshId(1),
            source: format!("builtin:{}", preset.name()),
            scale: 1.0,
            smoothing: self.mesh_smoothing,
            high_poly_source: None,
            content_hash: None,
            high_poly_hash: None,
        });
        self.set_base_geometry(Some(preset.build_mesh()));
        self.high_poly = None;
        self.high_poly_changed();
        self.history.end_merge();
        self.record_mesh_edit(before, Vec::new());
        self.history.end_merge();
    }

    pub(super) fn select_texture_set(&mut self, set_id: TextureSetId) {
        self.finish_active_stroke();
        self.selected_texture_set = Some(set_id);
        self.extra_selected_layers.clear();
        self.project.active_texture_set = Some(set_id);
        self.selected_layer = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == set_id)
            .and_then(|set| set.layers.last())
            .map(|layer| layer.id);
        self.invalidate_preview();
    }

    /// Shows the brand and tool pills and returns the y coordinate where
    /// left-side content may start.
    fn show_top_bars(&mut self, context: &egui::Context, screen: Rect) -> f32 {
        let brand = egui::Area::new(Id::new("overlay_brand"))
            .order(Order::Middle)
            .fixed_pos(screen.min + vec2(theme::EDGE, theme::EDGE))
            .show(context, |ui| self.brand_pill(ui))
            .response
            .rect;

        let left = brand.right() + theme::GAP;
        let inspector = if self.show_inspector {
            INSPECTOR_WIDTH + theme::GAP
        } else {
            0.0
        };
        let right = screen.right() - theme::EDGE - inspector;
        let width = self.layout.toolbar_width;
        let fits = right - left >= width;
        let position = if fits {
            pos2(
                (screen.center().x - width * 0.5).clamp(left, right - width),
                brand.top(),
            )
        } else {
            pos2(brand.left(), brand.bottom() + theme::GAP)
        };
        let toolbar = egui::Area::new(Id::new("overlay_toolbar"))
            .order(Order::Middle)
            .fixed_pos(position)
            .show(context, |ui| self.toolbar_pill(ui))
            .response
            .rect;
        self.layout.toolbar_width = toolbar.width();

        let lowest = if fits {
            brand.bottom()
        } else {
            toolbar.bottom()
        };
        lowest + theme::GAP
    }

    fn brand_pill(&self, ui: &mut Ui) {
        let project_name = self
            .project_path
            .as_ref()
            .and_then(|path| path.file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled Project".to_owned());
        theme::pill_frame()
            .inner_margin(Margin {
                left: 6,
                right: 18,
                top: 6,
                bottom: 6,
            })
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.set_min_height(32.0);
                    let (rect, _) = ui.allocate_exact_size(vec2(32.0, 32.0), Sense::hover());
                    ui.painter()
                        .circle_filled(rect.center(), 16.0, theme::ACCENT);
                    icons::paint(
                        ui.painter(),
                        rect.shrink(8.0),
                        Icon::Cube,
                        theme::ACCENT_TEXT,
                    );
                    ui.add_space(2.0);
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ui.label(
                            RichText::new("SUBSTANCECRAFT")
                                .size(9.5)
                                .extra_letter_spacing(1.0)
                                .color(theme::TEXT_FAINT),
                        );
                        ui.label(RichText::new(project_name).size(13.0).color(theme::TEXT));
                    });
                });
            });
    }

    fn toolbar_pill(&mut self, ui: &mut Ui) {
        theme::pill_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.set_min_height(32.0);
                ui.spacing_mut().item_spacing.x = 2.0;
                if widgets::icon_button(ui, Icon::Folder, true, "Open project…").clicked() {
                    self.open_project();
                }
                if widgets::icon_button(ui, Icon::Save, true, "Save project").clicked() {
                    self.save_project();
                }
                widgets::divider(ui);
                if widgets::icon_button(ui, Icon::Undo, self.history.can_undo(), "Undo (Ctrl+Z)")
                    .clicked()
                {
                    self.undo();
                }
                if widgets::icon_button(
                    ui,
                    Icon::Redo,
                    self.history.can_redo(),
                    "Redo (Ctrl+Shift+Z)",
                )
                .clicked()
                {
                    self.redo();
                }
                widgets::divider(ui);
                let mut mode = self.paint_mode;
                if widgets::segmented(ui, Id::new("toolbar_mode"), &mut mode, &MODES) {
                    self.set_paint_mode(mode);
                }
                widgets::divider(ui);
                ui.add_space(2.0);
                let selected_text = self.preview_label();
                let mut chosen = None;
                let mut material = false;
                let combo = egui::ComboBox::from_id_salt("toolbar_channel")
                    .width(118.0)
                    .selected_text(selected_text)
                    .show_ui(ui, |ui| {
                        let material_selected =
                            self.mesh_map_preview.is_none() && !self.channel_preview;
                        material = ui
                            .add(egui::Button::selectable(material_selected, "Material"))
                            .on_hover_text(
                                "Full PBR material: color, roughness, metallic, normal, AO",
                            )
                            .clicked();
                        ui.separator();
                        for channel in CHANNELS {
                            let selected = self.mesh_map_preview.is_none()
                                && self.channel_preview
                                && self.active_channel == channel;
                            if ui
                                .add_enabled(
                                    BAKEABLE_CHANNELS.contains(&channel),
                                    egui::Button::selectable(selected, channel_label(channel)),
                                )
                                .clicked()
                            {
                                chosen = Some(channel);
                            }
                        }
                    })
                    .response
                    .on_hover_text("Viewport preview");
                widgets::label_from_row(ui, &combo, egui::WidgetType::ComboBox);
                if material {
                    self.set_material_preview();
                } else if let Some(channel) = chosen {
                    self.set_preview_channel(channel);
                }
                ui.add_space(2.0);
                widgets::divider(ui);
                let mut view_mode = self.view_mode;
                if widgets::segmented(ui, Id::new("toolbar_view"), &mut view_mode, &VIEW_MODES) {
                    self.view_mode = view_mode;
                }
                if widgets::icon_button(ui, Icon::Focus, true, "Reset view").clicked() {
                    self.viewport_camera = ViewportCamera::default();
                    self.uv_view.zoom = 1.0;
                    self.uv_view.pan = eframe::egui::Vec2::ZERO;
                }
                let tooltip = if self.show_inspector {
                    "Hide inspector"
                } else {
                    "Show inspector"
                };
                if widgets::icon_toggle(ui, Icon::Sidebar, self.show_inspector, true, tooltip)
                    .clicked()
                {
                    self.show_inspector = !self.show_inspector;
                }
            });
        });
    }

    fn show_dock(&mut self, context: &egui::Context, screen: Rect, top: f32) {
        let dock = egui::Area::new(Id::new("overlay_dock"))
            .order(Order::Middle)
            .fixed_pos(pos2(screen.left() + theme::EDGE, top))
            .show(context, |ui| {
                theme::pill_frame()
                    .inner_margin(Margin::symmetric(5, 10))
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 4.0;
                        for drawer in Drawer::ALL {
                            let selected = self.drawer == Some(drawer);
                            if dock_item(ui, drawer.icon(), drawer.label(), selected).clicked() {
                                self.drawer = (!selected).then_some(drawer);
                            }
                        }
                    });
            })
            .response
            .rect;
        self.layout.dock_width = dock.width();
    }

    fn show_drawer(&mut self, context: &egui::Context, screen: Rect, top: f32) {
        if let Some(drawer) = self.drawer {
            self.last_drawer = drawer;
        }
        let amount =
            context.animate_bool_with_time(Id::new("drawer_open"), self.drawer.is_some(), 0.18);
        if amount <= 0.0 {
            return;
        }
        let drawer = self.last_drawer;
        let left = screen.left() + theme::EDGE + self.layout.dock_width + theme::GAP
            - (1.0 - amount) * 18.0;
        let max_height = (screen.bottom() - theme::EDGE - STATUS_RESERVE - top).max(180.0);

        egui::Area::new(Id::new("overlay_drawer"))
            .order(Order::Middle)
            .fixed_pos(pos2(left, top))
            .show(context, |ui| {
                ui.multiply_opacity(amount);
                theme::card_frame(theme::CARD_RADIUS)
                    .inner_margin(Margin::same(12))
                    .show(ui, |ui| {
                        ui.set_width(DRAWER_WIDTH - 24.0);
                        ui.horizontal(|ui| {
                            let (rect, _) =
                                ui.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
                            icons::paint(ui.painter(), rect, drawer.icon(), theme::ACCENT);
                            ui.label(RichText::new(drawer.title()).size(15.0).color(theme::TEXT));
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if widgets::header_action(ui, Icon::Close, true, "Close").clicked()
                                {
                                    self.drawer = None;
                                }
                            });
                        });
                        ui.add_space(4.0);
                        // The Area hands us last frame's rect as available space, so size
                        // the scroll area from its content rather than from that rect;
                        // otherwise switching to a taller drawer stays clipped.
                        let scroll_height = max_height - 70.0;
                        egui::ScrollArea::vertical()
                            .id_salt(("drawer_scroll", drawer.label()))
                            .max_height(scroll_height)
                            .min_scrolled_height(scroll_height)
                            .auto_shrink([false, true])
                            .show(ui, |ui| match drawer {
                                Drawer::Scene => self.scene_drawer(ui),
                                Drawer::Brush => self.brush_drawer(ui),
                                Drawer::Assets => self.assets_drawer(ui),
                                Drawer::Bake => self.bake_drawer(ui),
                                Drawer::Poses => self.poses_drawer(ui),
                                Drawer::Environment => self.environment_drawer(ui),
                                Drawer::Preferences => self.preferences_drawer(ui),
                                Drawer::Diagnostics => self.diagnostics_drawer(ui),
                            });
                    });
            });
    }

    fn scene_drawer(&mut self, ui: &mut Ui) {
        self.template_controls(ui);
        if self.mesh_file_missing() {
            let name = self
                .project
                .mesh
                .as_ref()
                .map(|mesh| super::path_label(std::path::Path::new(&mesh.source)))
                .unwrap_or_default();
            ui.label(
                RichText::new(format!("The mesh file {name} is missing."))
                    .size(12.0)
                    .color(WARNING),
            );
            if widgets::pill_button(ui, None, "Locate mesh…", true, true)
                .on_hover_text("Point the project at the moved or renamed file (undoable)")
                .clicked()
            {
                self.choose_mesh_location();
            }
            ui.add_space(4.0);
        }
        if widgets::pill_button(ui, Some(Icon::Import), "Import mesh…", true, true)
            .on_hover_text("FBX, glTF, or GLB. glTF materials become texture sets.")
            .clicked()
        {
            self.import_mesh();
        }
        if widgets::pill_button(ui, Some(Icon::Import), "Replace mesh…", false, true)
            .on_hover_text(
                "Swap in another version of the mesh and keep every texture set and layer \n                 (undoable)",
            )
            .clicked()
        {
            self.replace_mesh();
        }

        let mut import_scale = self.import_scale;
        if slider_row(
            ui,
            "Scale",
            &mut import_scale,
            MIN_MESH_SCALE..=MAX_MESH_SCALE,
            true,
            4,
        ) {
            self.set_import_scale(import_scale);
        }
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            for (label, scale) in [("×0.01", 0.01), ("×0.1", 0.1), ("×1", 1.0), ("×100", 100.0)]
            {
                if widgets::chip(ui, label, self.import_scale == scale).clicked() {
                    self.set_import_scale(scale);
                }
            }
        });

        if !self.imported_meshes.is_empty() {
            widgets::caption(ui, "Imported meshes");
            for index in 0..self.imported_meshes.len() {
                let Some(mesh) = self.imported_meshes.get(index) else {
                    continue;
                };
                let name = mesh.name.clone();
                let selected = self.selected_imported_mesh == Some(index);
                let row =
                    widgets::list_row(ui, Id::new(("imported_mesh", index)), selected, |ui| {
                        small_icon(ui, Icon::Cube, theme::TEXT_WEAK);
                        ui.label(name);
                    });
                if row.clicked() {
                    self.select_imported_mesh(index);
                }
            }
        }

        widgets::caption(ui, "Primitives");
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            for preset in PrimitivePreset::ALL {
                let selected = self.selected_primitive == Some(preset);
                if widgets::chip(ui, preset.name(), selected).clicked() {
                    self.select_primitive(preset);
                }
            }
        });

        widgets::caption(ui, "Smoothing");
        let mut smoothing = self.mesh_smoothing;
        let mut changed = slider_row(
            ui,
            "Smooth angle",
            &mut smoothing.angle_degrees,
            0.0..=180.0,
            false,
            0,
        );
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            for (label, angle) in [("Flat", 0.0), ("Auto 60°", 60.0), ("Smooth", 180.0)] {
                if widgets::chip(ui, label, smoothing.angle_degrees == angle).clicked() {
                    smoothing.angle_degrees = angle;
                    changed = true;
                }
            }
        });
        widgets::property_row(ui, "Subdivide", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for level in 0..=MAX_SUBDIVISION_LEVELS {
                let label = if level == 0 {
                    "Off".to_owned()
                } else {
                    format!("{level}")
                };
                if widgets::chip(ui, &label, smoothing.subdivision_levels == level).clicked() {
                    smoothing.subdivision_levels = level;
                    changed = true;
                }
            }
        });
        if changed {
            self.set_mesh_smoothing(MeshSmoothing { ..smoothing });
        }
        hint(
            ui,
            "Faces meeting below the smooth angle shade smoothly; sharper edges stay hard \
             and are kept as creases when subdividing.",
        );

        if let Some(mesh) = &self.mesh_geometry {
            widgets::caption(ui, "Mesh");
            let name = self.selected_mesh_name().unwrap_or("Mesh").to_owned();
            let elements = self
                .mesh_topology
                .as_ref()
                .map(|topology| (topology.element_count(), topology.island_count()));
            widgets::property_row(ui, "Name", |ui| ui.label(name));
            widgets::property_row(ui, "Vertices", |ui| {
                ui.label(mesh.vertices.len().to_string())
            });
            widgets::property_row(ui, "Triangles", |ui| {
                ui.label(mesh.triangles.len().to_string())
            });
            if let Some((elements, islands)) = elements {
                widgets::property_row(ui, "Elements", |ui| ui.label(elements.to_string()));
                widgets::property_row(ui, "UV islands", |ui| ui.label(islands.to_string()));
            }
        }
    }

    fn brush_drawer(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Mode");
        let mut mode = self.paint_mode;
        if widgets::segmented(ui, Id::new("drawer_mode"), &mut mode, &MODES) {
            self.set_paint_mode(mode);
        }
        match self.paint_mode {
            PaintMode::View => {
                ui.add_space(2.0);
                hint(
                    ui,
                    "Pick Paint, Erase, Mask, or ID to paint on the mesh or in the UV view, or \
                     Select to build a polygon mask from faces, elements, or UV islands.",
                );
                self.inspect_section(ui);
                return;
            }
            PaintMode::Select => {
                self.selection_tools(ui);
                return;
            }
            _ => {}
        }

        self.stroke_assist(ui);
        widgets::caption(ui, "Brush");
        self.brush_preview(ui);
        self.brush_shelves(ui);
        slider_row(ui, "Size", &mut self.brush_radius, 1.0..=256.0, true, 0);
        match self.paint_mode {
            PaintMode::Mask => {
                slider_row(ui, "Mask value", &mut self.mask_value, 0.0..=1.0, false, 2);
            }
            PaintMode::Paint => {
                widgets::property_row(ui, "Color", |ui| {
                    ui.color_edit_button_srgba(&mut self.brush_color);
                });
                self.color_space_picker(ui);
                slider_row(
                    ui,
                    "Roughness",
                    &mut self.brush_roughness,
                    0.0..=1.0,
                    false,
                    2,
                );
                slider_row(
                    ui,
                    "Metallic",
                    &mut self.brush_metallic,
                    0.0..=1.0,
                    false,
                    2,
                );
                slider_row(ui, "Height", &mut self.brush_height, 0.0..=1.0, false, 2);
                widgets::property_row(ui, "Emissive", |ui| {
                    ui.color_edit_button_srgba(&mut self.brush_emissive);
                });
                slider_row(
                    ui,
                    "Opacity ch.",
                    &mut self.brush_material_opacity,
                    0.0..=1.0,
                    false,
                    2,
                );
                hint(
                    ui,
                    "One stroke paints every channel the layer enables. Height 0.5 is \
                     neutral; Emissive and Opacity need Emit and Opac on the layer.",
                );
            }
            PaintMode::Erase => {
                hint(
                    ui,
                    "Erases paint from the selected paint layer in every channel; layers \
                     beneath show through.",
                );
            }
            _ => {
                widgets::property_row(ui, "Color", |ui| {
                    ui.color_edit_button_srgba(&mut self.brush_color);
                });
            }
        }
        self.brush_tip_controls(ui);
    }

    /// The brush colour as numbers in a chosen colour space: sRGB-encoded,
    /// linear Rec.709 light, or ACEScg. Entries outside sRGB clip to the
    /// colour brushes can store, and the panel says so.
    fn color_space_picker(&mut self, ui: &mut Ui) {
        use substancecraft_core::ColorSpace;

        let space = self.preferences.values.picker_color_space;
        let mut chosen = None;
        widgets::property_row(ui, "Space", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (option, label, tooltip) in [
                (
                    ColorSpace::Srgb,
                    "sRGB",
                    "Encoded sRGB values, as image editors show them",
                ),
                (
                    ColorSpace::LinearSrgb,
                    "Linear",
                    "Linear Rec.709 light, as renderers and MaterialX use",
                ),
                (
                    ColorSpace::AcesCg,
                    "ACEScg",
                    "Linear light with ACES AP1 primaries (VFX)",
                ),
            ] {
                if widgets::chip(ui, label, option == space)
                    .on_hover_text(tooltip)
                    .clicked()
                {
                    chosen = Some(option);
                }
            }
        });
        if let Some(option) = chosen {
            self.set_picker_color_space(option);
        }
        let space = self.preferences.values.picker_color_space;
        let [r, g, b, _] = self.brush_color.to_array();
        let mut value = space.from_srgb8([r, g, b]);
        let changed = widgets::property_row(ui, "Values", |ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            let mut changed = false;
            for (label, component) in ["R", "G", "B"].into_iter().zip(value.as_mut()) {
                let response = ui.add(
                    egui::DragValue::new(component)
                        .speed(0.002)
                        .range(0.0..=2.0)
                        .fixed_decimals(3)
                        .prefix(format!("{label} ")),
                );
                response.widget_info(|| egui::WidgetInfo::drag_value(true, f64::from(*component)));
                changed |= response.changed();
            }
            changed
        });
        if changed {
            self.set_brush_color_in(space, value);
        }
        if self.picker_clipped == Some(self.brush_color) {
            hint(
                ui,
                "Outside the sRGB gamut: clipped to the nearest colour a brush can store.",
            );
        }
    }

    /// Live preview of the brush tip, drawn with the real rasteriser.
    fn brush_preview(&self, ui: &mut Ui) {
        let tip_color = match self.paint_mode {
            PaintMode::Mask => Color32::from_gray((self.mask_value * 255.0).round() as u8),
            PaintMode::Erase => Color32::from_gray(220),
            _ => self.brush_color,
        };
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 76.0), Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect, CornerRadius::same(16), theme::INPUT_DEEP);
        let radius = (self.brush_radius * 0.3).clamp(3.0, 30.0);
        let samples = brush_shape_samples(&self.brush_tip, &self.resource_images);
        let cell = radius * 2.0 / BRUSH_PREVIEW_CELLS as f32;
        for (index, coverage) in samples.iter().enumerate() {
            if *coverage <= 0.02 {
                continue;
            }
            let x = (index % BRUSH_PREVIEW_CELLS) as f32;
            let y = (index / BRUSH_PREVIEW_CELLS) as f32;
            let min = rect.center() - vec2(radius, radius) + vec2(x * cell, y * cell);
            painter.rect_filled(
                Rect::from_min_size(min, vec2(cell + 0.5, cell + 0.5)),
                0.0,
                tip_color.gamma_multiply(*coverage),
            );
        }
        if let Some(pressure) = self.pen_pressure {
            painter.text(
                rect.right_bottom() - vec2(10.0, 10.0),
                Align2::RIGHT_BOTTOM,
                format!("pen {:.0}%", pressure * 100.0),
                FontId::proportional(10.5),
                theme::TEXT_WEAK,
            );
        }
    }

    fn brush_tip_controls(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Tip");
        let tip = &mut self.brush_tip;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            for shape in BrushShape::BUILT_IN {
                if widgets::chip(ui, shape.name(), tip.shape == shape).clicked() {
                    tip.shape = shape;
                }
            }
            if let BrushShape::Bitmap(id) = tip.shape {
                widgets::chip(ui, "Bitmap", true).on_hover_text(format!("Alpha #{}", id.0));
            }
        });
        if !matches!(tip.shape, BrushShape::Bitmap(_)) {
            slider_row(ui, "Hardness", &mut tip.hardness, 0.0..=1.0, false, 2);
        }
        slider_row(
            ui,
            "Spacing",
            &mut tip.spacing,
            MIN_BRUSH_SPACING..=MAX_BRUSH_SPACING,
            true,
            2,
        );
        slider_row(
            ui,
            "Angle°",
            &mut tip.angle_degrees,
            -180.0..=180.0,
            false,
            0,
        );
        widgets::property_row(ui, "Follow path", |ui| {
            widgets::toggle(ui, &mut tip.follow_path)
        });
        slider_row(ui, "Size jitter", &mut tip.size_jitter, 0.0..=1.0, false, 2);
        slider_row(
            ui,
            "Angle jitter",
            &mut tip.angle_jitter,
            0.0..=1.0,
            false,
            2,
        );
        slider_row(ui, "Scatter", &mut tip.scatter, 0.0..=4.0, false, 2);
        seed_row(ui, &mut tip.seed);
        widgets::caption(ui, "Pen pressure");
        widgets::property_row(ui, "Size", |ui| widgets::toggle(ui, &mut tip.pressure_size));
        widgets::property_row(ui, "Opacity", |ui| {
            widgets::toggle(ui, &mut tip.pressure_opacity)
        });
        hint(
            ui,
            "Stamped tips (any shape but a plain round, or any jitter) place one stamp \
             every Spacing diameters. Bitmap alphas come from the Assets drawer.",
        );
    }

    fn assets_drawer(&mut self, ui: &mut Ui) {
        if widgets::pill_button(ui, Some(Icon::Import), "Import bitmaps…", true, true)
            .on_hover_text(
                "PNG, JPEG, TGA, SVG, or text resources for fills, masks, and brush alphas",
            )
            .clicked()
        {
            self.import_resources();
        }
        if !self.pending_import_assignments.is_empty() {
            widgets::caption(ui, "Assign imported bitmaps");
            hint(
                ui,
                "Review each channel suggestion before fill layers are created. Unassigned files stay in Assets.",
            );
            let resources = self
                .project
                .resources
                .iter()
                .map(|resource| (resource.id, resource.name.clone()))
                .collect::<Vec<_>>();
            let mut remove = Vec::new();
            for assignment in &mut self.pending_import_assignments {
                let name = resources
                    .iter()
                    .find(|(id, _)| *id == assignment.resource)
                    .map_or_else(
                        || format!("Resource {}", assignment.resource.0),
                        |(_, name)| name.clone(),
                    );
                ui.horizontal(|ui| {
                    ui.label(RichText::new(name).size(11.5).color(theme::TEXT));
                    let combo =
                        egui::ComboBox::from_id_salt(("import_channel", assignment.resource.0))
                            .width(ui.available_width().min(132.0))
                            .selected_text(assignment.channel.map_or("Skip", channel_label))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut assignment.channel, None, "Skip");
                                for channel in [
                                    MaterialChannel::BaseColor,
                                    MaterialChannel::Normal,
                                    MaterialChannel::Roughness,
                                    MaterialChannel::Metallic,
                                    MaterialChannel::Height,
                                    MaterialChannel::Emissive,
                                    MaterialChannel::Opacity,
                                ] {
                                    ui.selectable_value(
                                        &mut assignment.channel,
                                        Some(channel),
                                        channel_label(channel),
                                    );
                                }
                            });
                    widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
                    if widgets::icon_button(ui, Icon::Close, true, "Remove from assignment")
                        .clicked()
                    {
                        remove.push(assignment.resource);
                    }
                });
                ui.label(
                    RichText::new(&assignment.reason)
                        .size(10.5)
                        .color(theme::TEXT_FAINT),
                );
            }
            self.pending_import_assignments
                .retain(|assignment| !remove.contains(&assignment.resource));
            ui.horizontal(|ui| {
                if widgets::pill_button(ui, None, "Create fill layers", true, false).clicked() {
                    self.confirm_import_assignments();
                }
                if widgets::pill_button(ui, None, "Dismiss", false, false).clicked() {
                    self.pending_import_assignments.clear();
                }
            });
            ui.add_space(4.0);
        }
        if widgets::pill_button(ui, None, "New text resource…", false, true)
            .on_hover_text(
                "Write a .sctext file (text and size) and use the rendered text like a bitmap",
            )
            .clicked()
        {
            self.new_text_resource();
        }
        self.asset_search(ui);
        self.project_materials(ui);
        self.project_smart_masks(ui);
        hint(
            ui,
            "Bitmaps are referenced by path, like the mesh. Assign them to fill channels \
             and bitmap masks in a layer's properties, or use one as the brush tip here.",
        );
        if self.project.resources.is_empty() {
            empty_state(ui, "Imported bitmaps appear here.");
            return;
        }
        let mode_changed = widgets::property_row(ui, "Auto-reload", |ui| {
            widgets::segmented(
                ui,
                Id::new("resource_reload_mode"),
                &mut self.resource_watcher.mode,
                &ReloadMode::ALL,
            )
        });
        if mode_changed {
            let pending: Vec<ResourceId> = std::mem::take(&mut self.resource_watcher.pending)
                .into_iter()
                .collect();
            if self.resource_watcher.mode == ReloadMode::Auto {
                self.reload_changed_resources(&pending);
            }
        }
        hint(
            ui,
            "When a bitmap is saved by another app, the layers using it update. \
             Ask waits for confirmation.",
        );
        widgets::caption(ui, "Bitmaps");
        self.resource_details(ui);
        let resources = self
            .project
            .resources
            .iter()
            .map(|resource| (resource.id, resource.name.clone()))
            .collect::<Vec<_>>();
        for (id, name) in resources {
            let thumbnail = self.resource_thumbnail(ui.ctx(), id);
            let loaded = self
                .resource_images
                .get(&id)
                .map(|image| (image.width, image.height));
            let in_use = self.resource_in_use(id);
            let is_brush = self.brush_tip.shape == BrushShape::Bitmap(id);
            let mut use_as_brush = false;
            let mut remove = false;
            let mut relink = false;
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(vec2(44.0, 44.0), Sense::hover());
                ui.painter()
                    .rect_filled(rect, CornerRadius::same(8), theme::INPUT_DEEP);
                if let Some(thumbnail) = &thumbnail {
                    let size = thumbnail.size_vec2();
                    let scale = (40.0 / size.x.max(size.y)).min(1.0);
                    ui.painter().image(
                        thumbnail.id(),
                        Rect::from_center_size(rect.center(), size * scale),
                        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                }
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    ui.label(RichText::new(&name).color(theme::TEXT));
                    let detail = match loaded {
                        Some((width, height)) => format!("{width} × {height}"),
                        None => "File missing".to_owned(),
                    };
                    ui.label(RichText::new(detail).size(11.0).color(theme::TEXT_FAINT));
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        use_as_brush = widgets::chip(ui, "Brush alpha", is_brush)
                            .on_hover_text("Stamp this bitmap as the brush tip")
                            .clicked();
                        if loaded.is_none() {
                            relink = widgets::chip(ui, "Relink…", false)
                                .on_hover_text("Point this bitmap at its moved or renamed file")
                                .clicked();
                        }
                        remove = ui
                            .add_enabled_ui(!in_use, |ui| {
                                widgets::chip(ui, "Remove", false).clicked()
                            })
                            .inner;
                    });
                });
            });
            if use_as_brush {
                self.brush_tip.shape = if is_brush {
                    BrushShape::Round
                } else {
                    BrushShape::Bitmap(id)
                };
            }
            if remove {
                self.remove_resource(id);
            }
            if relink {
                self.choose_relink(id);
            }
        }
    }

    fn selection_tools(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Select by");
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            for mode in SelectionMode::ALL {
                if widgets::chip(ui, mode.name(), self.selection_mode == mode).clicked() {
                    self.selection_mode = mode;
                    self.hovered_selection.clear();
                }
            }
        });
        hint(
            ui,
            match self.selection_mode {
                SelectionMode::Triangle => "Click single triangles.",
                SelectionMode::Element => {
                    "Click a connected part of the mesh — a separate object, button, or strap."
                }
                SelectionMode::UvIsland => "Click to take a whole UV island.",
            },
        );
        hint(
            ui,
            "Click adds to the selected layer's polygon mask, Shift-click removes. Drag to \
             orbit.",
        );
        if self.selected_layer.is_none() {
            ui.add_space(4.0);
            ui.label(
                RichText::new("Select or add a layer first.")
                    .size(12.0)
                    .color(WARNING),
            );
            if widgets::pill_button(ui, Some(Icon::Plus), "Add fill layer", true, true).clicked() {
                self.add_fill_layer();
            }
        }
    }

    fn bake_drawer(&mut self, ui: &mut Ui) {
        let running = self.mesh_map_job_running();
        let fresh = self.mesh_maps_fresh();
        widgets::property_row(ui, "Status", |ui| {
            let (text, color) = if running {
                ("Baking…", theme::ACCENT)
            } else if fresh {
                ("Up to date", theme::TEXT)
            } else if self.mesh_map_error.is_some() {
                ("Failed", theme::DANGER)
            } else {
                ("Not baked", theme::TEXT_WEAK)
            };
            if running {
                ui.add(egui::Spinner::new().size(12.0));
            }
            ui.label(RichText::new(text).color(color));
        });
        if let Some(reason) = self.mesh_map_stale_reason() {
            widgets::property_row(ui, "Stale because", |ui| {
                ui.label(RichText::new(reason).color(WARNING))
            });
        }
        if let Some(maps) = self.fresh_mesh_maps() {
            let summary = format!(
                "{}² · {:.0}% UV coverage",
                maps.width(),
                maps.coverage() * 100.0
            );
            widgets::property_row(ui, "Maps", |ui| ui.label(summary));
        }
        if running {
            if widgets::pill_button(ui, Some(Icon::Close), "Cancel bake", false, true).clicked() {
                self.cancel_mesh_map_job();
            }
        } else if widgets::pill_button(ui, Some(Icon::Bake), "Bake mesh maps", true, true).clicked()
        {
            self.mesh_map_error = None;
            self.start_mesh_map_bake();
        }
        hint(
            ui,
            "Generators such as Edge Wear and Dirt read these maps. They rebake \
             automatically when the mesh or its smoothing changes.",
        );

        widgets::caption(ui, "High poly source");
        let high_poly = self.high_poly.as_ref().map(|source| {
            let name = source
                .path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "High poly".to_owned());
            (name, source.geometry.triangles.len())
        });
        match high_poly {
            Some((name, triangles)) => {
                widgets::property_row(ui, "Mesh", |ui| ui.label(name));
                widgets::property_row(ui, "Triangles", |ui| ui.label(triangles.to_string()));
                if let Some(maps) = self.fresh_mesh_maps().filter(|maps| maps.from_high_poly()) {
                    let misses = maps.miss_fraction();
                    let color = if misses > 0.05 { WARNING } else { theme::TEXT };
                    widgets::property_row(ui, "Ray misses", |ui| {
                        ui.label(RichText::new(format!("{:.1}%", misses * 100.0)).color(color))
                    });
                    if misses > 0.05 {
                        hint(
                            ui,
                            "Many cage rays miss the high mesh: raise Cage distance.",
                        );
                    }
                }
                ui.horizontal(|ui| {
                    if widgets::pill_button(ui, Some(Icon::Import), "Replace…", false, false)
                        .clicked()
                    {
                        self.pick_high_poly();
                    }
                    if widgets::pill_button(ui, Some(Icon::Close), "Remove", false, false).clicked()
                    {
                        self.clear_high_poly();
                    }
                });
            }
            None => {
                if widgets::pill_button(ui, Some(Icon::Import), "Load high poly FBX…", false, true)
                    .clicked()
                {
                    self.pick_high_poly();
                }
                hint(
                    ui,
                    "Bake a detailed sculpt onto this mesh: its surface becomes the Normal \
                     map, and AO/curvature come from the high mesh. Same scale as import.",
                );
            }
        }
        slider_row(
            ui,
            "Cage distance",
            &mut self.mesh_map_settings.cage_distance,
            0.001..=0.2,
            true,
            3,
        );
        if self.high_poly.is_some() {
            slider_row(
                ui,
                "Skew fix",
                &mut self.mesh_map_settings.skew,
                0.0..=1.0,
                false,
                2,
            );
            slider_row(
                ui,
                "Edge guard°",
                &mut self.mesh_map_settings.skew_edge_degrees,
                1.0..=90.0,
                false,
                0,
            );
            let resources = self
                .project
                .resources
                .iter()
                .map(|resource| (resource.id, resource.name.clone()))
                .collect::<Vec<_>>();
            let mut skew_map = self.skew_map;
            widgets::property_row(ui, "Skew map", |ui| {
                resource_picker(ui, Id::new("skew_map"), &mut skew_map, &resources);
            });
            self.skew_map = skew_map;
            hint(
                ui,
                "Skew fix sends cage rays straight out of flat faces so details beside \
                 bevels are not skewed; a skew map (white = full) limits it to painted \
                 areas, and the edge guard keeps bevels themselves smooth. Inspect the \
                 result with the Skew map preview, then bake again.",
            );
        }

        widgets::caption(ui, "Settings");
        widgets::property_row(ui, "Resolution", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for resolution in MESH_MAP_RESOLUTIONS {
                let label = if resolution >= 1024 {
                    format!("{}k", resolution / 1024)
                } else {
                    resolution.to_string()
                };
                if widgets::chip(ui, &label, self.mesh_map_settings.resolution == resolution)
                    .clicked()
                {
                    self.mesh_map_settings.resolution = resolution;
                }
            }
        });
        let mut samples = self.mesh_map_settings.ao_samples as f32;
        if slider_row(ui, "AO samples", &mut samples, 4.0..=128.0, false, 0) {
            self.mesh_map_settings.ao_samples = samples.round() as u32;
        }
        slider_row(
            ui,
            "AO distance",
            &mut self.mesh_map_settings.ao_distance,
            0.01..=1.0,
            true,
            2,
        );
        slider_row(
            ui,
            "Edge width",
            &mut self.mesh_map_settings.edge_width,
            0.002..=0.15,
            true,
            3,
        );
        slider_row(
            ui,
            "Edge angle",
            &mut self.mesh_map_settings.edge_angle_degrees,
            1.0..=120.0,
            false,
            0,
        );
        let mut padding = self.mesh_map_settings.padding as f32;
        if slider_row(ui, "Padding", &mut padding, 0.0..=16.0, false, 0) {
            self.mesh_map_settings.padding = padding.round() as u32;
        }
        hint(ui, "Setting changes apply on the next bake.");

        widgets::caption(ui, "Preview on mesh");
        let mut preview = self.mesh_map_preview;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            if widgets::chip(ui, "Material", preview.is_none()).clicked() {
                preview = None;
            }
            for kind in MeshMapKind::ALL {
                if widgets::chip(ui, kind.name(), preview == Some(kind)).clicked() {
                    preview = Some(kind);
                }
            }
        });
        if preview != self.mesh_map_preview {
            self.mesh_map_preview = preview;
            self.invalidate_preview();
        }
        if let Some(kind) = self.mesh_map_preview
            && widgets::pill_button(
                ui,
                Some(Icon::Export),
                &format!("Export {}", kind.name()),
                false,
                true,
            )
            .clicked()
        {
            self.export_mesh_map(kind);
        }
        self.cache_section(ui);
        self.bake_log_section(ui);
    }

    fn poses_drawer(&mut self, ui: &mut Ui) {
        self.animation_section(ui);
        widgets::caption(ui, "Root translation");
        let transform = self
            .current_pose
            .bones
            .entry(BoneId(0))
            .or_insert(BoneTransform::IDENTITY);
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let width = (ui.available_width() - 12.0) / 3.0;
            for (axis, value) in [
                ("X", &mut transform.translation.x),
                ("Y", &mut transform.translation.y),
                ("Z", &mut transform.translation.z),
            ] {
                changed |= ui
                    .add_sized(
                        vec2(width, 26.0),
                        egui::DragValue::new(value)
                            .prefix(format!("{axis}  "))
                            .speed(0.05),
                    )
                    .changed();
            }
        });
        if changed {
            self.pose_transition = None;
            self.selected_pose = None;
        }

        widgets::caption(ui, "Save pose");
        let can_save =
            !self.pose_name.trim().is_empty() && self.project.saved_poses.len() < MAX_SAVED_POSES;
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.pose_name)
                    .desired_width(ui.available_width() - 76.0)
                    .margin(vec2(10.0, 5.0))
                    .hint_text("Pose name"),
            );
            let saved = ui
                .add_enabled_ui(can_save, |ui| {
                    widgets::pill_button(ui, None, "Save", true, false).clicked()
                })
                .inner;
            if saved {
                self.save_pose();
            }
        });

        widgets::caption(ui, "Saved poses");
        if self.project.saved_poses.is_empty() {
            empty_state(ui, "Saved poses appear here.");
        }
        let poses = self
            .project
            .saved_poses
            .iter()
            .map(|pose| (pose.id, pose.name.clone()))
            .collect::<Vec<_>>();
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            for (pose_id, name) in poses {
                let selected = self.selected_pose == Some(pose_id);
                if widgets::chip(ui, &name, selected).clicked() {
                    self.select_pose(pose_id, ui.ctx());
                }
            }
        });
    }

    fn environment_drawer(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Camera");
        slider_row(
            ui,
            "Orbit",
            &mut self.viewport_camera.yaw,
            0.0..=std::f32::consts::TAU,
            false,
            2,
        );
        slider_row(
            ui,
            "Tilt",
            &mut self.viewport_camera.pitch,
            -1.55..=1.55,
            false,
            2,
        );
        slider_row(
            ui,
            "Zoom",
            &mut self.viewport_camera.zoom,
            0.1..=20.0,
            true,
            2,
        );
        ui.add_space(2.0);
        if widgets::pill_button(ui, Some(Icon::Focus), "Reset view", false, true).clicked() {
            self.viewport_camera = ViewportCamera::default();
        }
        self.saved_views_section(ui);
        self.edge_overlay_controls(ui);

        widgets::caption(ui, "Environment light");
        let mut chosen = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
            let custom = self.custom_environment.is_some();
            for preset in substancecraft_core::LightingPreset::ALL {
                let selected = !custom && self.viewport_lighting.preset == preset;
                if widgets::chip(ui, preset.name(), selected)
                    .on_hover_text(preset.description())
                    .clicked()
                {
                    chosen = Some(preset);
                }
            }
            if let Some(environment) = &self.custom_environment {
                widgets::chip(ui, &environment.name(), true)
                    .on_hover_text(environment.path.display().to_string());
            }
        });
        if let Some(preset) = chosen.filter(|preset| {
            self.custom_environment.is_some() || *preset != self.viewport_lighting.preset
        }) {
            self.set_lighting_preset(preset);
        }
        if widgets::pill_button(ui, Some(Icon::Folder), "Load HDRI…", false, true)
            .on_hover_text(
                "Light the viewport and renders with an equirectangular .hdr or .exr panorama",
            )
            .clicked()
        {
            self.pick_custom_environment();
        }
        if let Some(status) = &self.environment_status {
            hint(ui, status);
        }
        slider_row(
            ui,
            "Rotation",
            &mut self.viewport_lighting.rotation,
            0.0..=std::f32::consts::TAU,
            false,
            2,
        );
        slider_row(
            ui,
            "Strength",
            &mut self.viewport_lighting.ambient_fill,
            0.0..=1.0,
            false,
            2,
        );

        widgets::caption(ui, "Render");
        let mut long_edge = None;
        widgets::property_row(ui, "Image", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (edge, label) in [(1024, "1K"), (2048, "2K")] {
                if widgets::chip(ui, label, false)
                    .on_hover_text(format!("Render a {edge}-pixel PNG of this view"))
                    .clicked()
                {
                    long_edge = Some(edge);
                }
            }
        });
        if let Some(edge) = long_edge {
            self.request_render(edge, None);
        }
        widgets::property_row(ui, "Turntable", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for mode in crate::turntable::TurntableMode::ALL {
                if widgets::chip(ui, mode.name(), self.turntable_mode == mode)
                    .on_hover_text(mode.description())
                    .clicked()
                {
                    self.turntable_mode = mode;
                }
            }
        });
        widgets::property_row(ui, "Frames", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for frames in crate::turntable::TURNTABLE_FRAME_COUNTS {
                if widgets::chip(ui, &frames.to_string(), self.turntable_frames == frames)
                    .on_hover_text(format!("One full turn in {frames} frames"))
                    .clicked()
                {
                    self.turntable_frames = frames;
                }
            }
        });
        if self.turntable.is_some() {
            if widgets::pill_button(ui, Some(Icon::Close), "Cancel turntable", false, true)
                .clicked()
            {
                self.cancel_turntable();
            }
        } else if widgets::pill_button(ui, Some(Icon::Image), "Render turntable…", false, true)
            .on_hover_text("Render a looping 1K PNG sequence of one full turn")
            .clicked()
        {
            self.request_turntable(self.turntable_frames, self.turntable_mode, 1024);
        }
        if let Some(status) = &self.render_status {
            hint(ui, status);
        }
        hint(
            ui,
            "Renders the current material with this camera and lighting as an sRGB PNG.",
        );
    }

    fn show_inspector_column(&mut self, context: &egui::Context, screen: Rect) {
        let max_height = (screen.height() - theme::EDGE * 2.0).max(240.0);
        egui::Area::new(Id::new("overlay_inspector"))
            .order(Order::Middle)
            .pivot(Align2::RIGHT_TOP)
            .fixed_pos(pos2(
                screen.right() - theme::EDGE,
                screen.top() + theme::EDGE,
            ))
            .show(context, |ui| {
                ui.set_width(INSPECTOR_WIDTH);
                // An Area offers last frame's rect as available space, so the scroll
                // height must come from the window, not from `ui`; otherwise the column
                // stays clipped at a stale height after its content grows.
                egui::ScrollArea::vertical()
                    .id_salt("inspector_scroll")
                    .max_height(max_height)
                    .min_scrolled_height(max_height)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        ui.set_max_width(INSPECTOR_WIDTH);
                        ui.spacing_mut().item_spacing.y = theme::GAP;
                        self.texture_sets_card(ui);
                        self.layers_card(ui);
                        self.properties_card(ui);
                    });
            });
    }

    fn texture_sets_card(&mut self, ui: &mut Ui) {
        let mut add = false;
        widgets::card(
            ui,
            Id::new("card_texture_sets"),
            Icon::Grid,
            "Texture Sets",
            1,
            |ui| {
                add = widgets::header_action(ui, Icon::Plus, true, "Add texture set").clicked();
            },
            |ui| {
                if self.project.texture_sets.is_empty() {
                    empty_state(ui, "No texture sets yet.");
                    if widgets::pill_button(ui, Some(Icon::Plus), "Add texture set", true, true)
                        .clicked()
                    {
                        self.add_texture_set();
                    }
                    return;
                }
                let sets = self
                    .project
                    .texture_sets
                    .iter()
                    .map(|set| (set.id, set.name.clone(), set.material_slot.clone()))
                    .collect::<Vec<_>>();
                egui::ScrollArea::vertical()
                    .id_salt("texture_sets_scroll")
                    .max_height(120.0)
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 2.0;
                        for (set_id, name, slot) in sets {
                            let selected = self.selected_texture_set == Some(set_id);
                            let row = widgets::list_row(
                                ui,
                                Id::new(("texture_set_row", set_id.0)),
                                selected,
                                |ui| {
                                    ui.label(name);
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        ui.label(
                                            RichText::new(slot).size(11.5).color(theme::TEXT_FAINT),
                                        );
                                    });
                                },
                            );
                            if row.clicked() {
                                self.select_texture_set(set_id);
                            }
                        }
                    });
            },
        );
        if add {
            self.add_texture_set();
        }
    }

    fn layers_card(&mut self, ui: &mut Ui) {
        let mut add = false;
        widgets::card(
            ui,
            Id::new("card_layers"),
            Icon::Layers,
            "Layers",
            1,
            |ui| {
                add = widgets::header_action(ui, Icon::Plus, true, "Add paint layer").clicked();
            },
            |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(5.0, 5.0);
                    if widgets::chip(ui, "+ Paint", false).clicked() {
                        self.add_paint_layer();
                    }
                    if widgets::chip(ui, "+ Fill", false)
                        .on_hover_text("Uniform or textured material covering the layer mask")
                        .clicked()
                    {
                        self.add_fill_layer();
                    }
                    if widgets::chip(ui, "+ Group", false)
                        .on_hover_text(
                            "Children composite together, then blend through the group's \
                             opacity, mask, and filters",
                        )
                        .clicked()
                    {
                        self.add_group_layer();
                    }
                    if widgets::chip(ui, "OpenPBR…", false)
                        .on_hover_text(
                            "Fill layer from a MaterialX document's OpenPBR surface: its \
                             textures become bitmaps, its constant inputs fill values",
                        )
                        .clicked()
                    {
                        self.import_openpbr();
                    }
                    if widgets::chip(ui, "Edge Wear", false)
                        .on_hover_text("Bare metal on convex edges, driven by curvature")
                        .clicked()
                    {
                        self.add_edge_wear_layer();
                    }
                    if widgets::chip(ui, "Dirt", false)
                        .on_hover_text("Grime in cavities, driven by occlusion and curvature")
                        .clicked()
                    {
                        self.add_dirt_layer();
                    }
                });
                ui.add_space(4.0);
                ui.add(
                    egui::TextEdit::singleline(&mut self.layer_query)
                        .desired_width(ui.available_width())
                        .margin(vec2(10.0, 5.0))
                        .hint_text("Search layers: name, type, channel, resource…"),
                );
                let Some(texture_set_id) = self.selected_texture_set else {
                    empty_state(ui, "Add a layer to create a texture set.");
                    return;
                };
                let resources = self
                    .project
                    .resources
                    .iter()
                    .map(|resource| (resource.id, resource.name.clone()))
                    .collect::<Vec<_>>();
                let rows = self
                    .selected_texture_set_ref()
                    .map(|set| layer_rows(&set.layers, 0, &self.layer_query, &resources))
                    .unwrap_or_default();
                if rows.is_empty() {
                    let message = if self.layer_query.trim().is_empty() {
                        "This texture set has no layers."
                    } else {
                        "No layers match this search."
                    };
                    empty_state(ui, message);
                    return;
                }
                ui.add_space(2.0);
                egui::ScrollArea::vertical()
                    .id_salt("layers_scroll")
                    .max_height(240.0)
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 2.0;
                        for row in rows {
                            let selected = self.selected_layer == Some(row.id)
                                || self.extra_selected_layers.contains(&row.id);
                            let mut toggle_visibility = false;
                            let response = widgets::list_row(
                                ui,
                                Id::new(("layer_row", row.id.0)),
                                selected,
                                |ui| {
                                    ui.add_space(row.depth as f32 * 14.0);
                                    let (icon, tooltip) = if row.visible {
                                        (Icon::Eye, "Hide layer")
                                    } else {
                                        (Icon::EyeOff, "Show layer")
                                    };
                                    toggle_visibility = widgets::sized_icon_toggle(
                                        ui, icon, false, true, tooltip, 24.0,
                                    )
                                    .clicked();
                                    let color = if row.visible {
                                        theme::TEXT
                                    } else {
                                        theme::TEXT_FAINT
                                    };
                                    let name = if row.group {
                                        format!("▾ {}", row.name)
                                    } else {
                                        row.name.clone()
                                    };
                                    ui.label(RichText::new(name).color(color));
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        ui.label(
                                            RichText::new(format!("{:.0}%", row.opacity * 100.0))
                                                .size(11.5)
                                                .color(theme::TEXT_FAINT),
                                        );
                                        ui.label(
                                            RichText::new(&row.badge)
                                                .size(10.5)
                                                .color(theme::ACCENT),
                                        );
                                    });
                                },
                            );
                            if toggle_visibility {
                                self.toggle_layer_visibility(texture_set_id, row.id);
                            } else if response.clicked() {
                                self.finish_active_stroke();
                                if ui.input(|input| input.modifiers.command) {
                                    self.toggle_extra_selection(row.id);
                                } else {
                                    self.selected_layer = Some(row.id);
                                    self.extra_selected_layers.clear();
                                }
                            }
                        }
                    });

                if let Some((_, slot, count)) = self.selected_layer_slot() {
                    let into_group = self.group_below_selected().is_some();
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        if widgets::icon_button(
                            ui,
                            Icon::ArrowUp,
                            slot.index + 1 < count,
                            "Move up",
                        )
                        .clicked()
                        {
                            self.move_selected_layer(1);
                        }
                        if widgets::icon_button(ui, Icon::ArrowDown, slot.index > 0, "Move down")
                            .clicked()
                        {
                            self.move_selected_layer(-1);
                        }
                        ui.add_space(4.0);
                        if ui
                            .add_enabled_ui(into_group, |ui| {
                                widgets::chip(ui, "Into group", false)
                                    .on_hover_text("Move onto the top of the group below")
                                    .clicked()
                            })
                            .inner
                        {
                            self.move_selected_into_group();
                        }
                        if ui
                            .add_enabled_ui(slot.parent.is_some(), |ui| {
                                widgets::chip(ui, "Out", false)
                                    .on_hover_text("Move out of the group, just above it")
                                    .clicked()
                            })
                            .inner
                        {
                            self.move_selected_out_of_group();
                        }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if widgets::icon_button(ui, Icon::Trash, true, "Delete layer").clicked()
                            {
                                self.delete_selected_layer();
                            }
                        });
                    });
                }
            },
        );
        if add {
            self.add_paint_layer();
        }
    }

    fn toggle_layer_visibility(&mut self, texture_set_id: TextureSetId, layer_id: LayerId) {
        let Some(layer) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)
            .and_then(|set| set.layer_mut(layer_id))
        else {
            return;
        };
        let before = layer.settings();
        layer.visible = !layer.visible;
        let after = layer.settings();
        self.history
            .record(substancecraft_core::EditCommand::LayerSettings {
                texture_set_id,
                layer_id,
                before: Box::new(before),
                after: Box::new(after.clone()),
            });
        if self.selected_layer == Some(layer_id) {
            self.layer_settled = Some((texture_set_id, layer_id, after));
        }
        self.invalidate_preview();
    }

    fn properties_card(&mut self, ui: &mut Ui) {
        widgets::card(
            ui,
            Id::new("card_properties"),
            Icon::Sliders,
            "Properties",
            0,
            |_| {},
            |ui| {
                let Some(texture_set) = self.selected_texture_set_ref() else {
                    empty_state(ui, "Select a texture set to inspect it.");
                    return;
                };
                let name = texture_set.name.clone();
                let slot = texture_set.material_slot.clone();
                let resolution = texture_set.resolution.width;
                let mut strength = texture_set.height_strength;
                let multi_slot = self
                    .mesh_geometry
                    .as_deref()
                    .is_some_and(|mesh| mesh.material_slot_count() > 1);
                let assigned = !multi_slot
                    || self
                        .mesh_geometry
                        .as_deref()
                        .is_some_and(|mesh| mesh.material_slots.contains(&slot));
                widgets::property_row(ui, "Texture set", |ui| ui.label(name));
                if multi_slot {
                    widgets::property_row(ui, "Material", |ui| {
                        let color = if assigned { theme::TEXT } else { WARNING };
                        ui.label(RichText::new(&slot).color(color))
                    });
                    if !assigned {
                        hint(ui, "No mesh material has this name, so nothing shows it.");
                    }
                }
                let mut chosen_resolution = None;
                widgets::property_row(ui, "Resolution", |ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for size in TEXTURE_SET_RESOLUTIONS {
                        let label = if size >= 1024 {
                            format!("{}k", size / 1024)
                        } else {
                            size.to_string()
                        };
                        if widgets::chip(ui, &label, resolution == size).clicked() {
                            chosen_resolution = Some(size);
                        }
                    }
                });
                let strength_changed =
                    slider_row(ui, "Height force", &mut strength, 0.0..=8.0, false, 2);
                if let Some(id) = self.selected_texture_set {
                    if strength_changed {
                        // A drag is one undo step.
                        self.edit_set_properties_merging(id, |properties| {
                            properties.height_strength = strength;
                        });
                    }
                    if let Some(size) = chosen_resolution {
                        self.edit_set_properties(id, |properties| {
                            properties.resolution.width = size;
                            properties.resolution.height = size;
                        });
                    }
                }

                if self.selected_layer().is_some() {
                    self.layer_properties(ui);
                    self.layer_actions(ui);
                }

                widgets::caption(ui, "Bake & export");
                widgets::property_row(ui, "Format", |ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for format in TextureFormat::ALL {
                        if widgets::chip(ui, format.name(), self.selected_format() == format)
                            .clicked()
                        {
                            self.set_format_choice(format);
                        }
                    }
                });
                self.export_preset_controls(ui);
                ui.add_space(2.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
                    for channel in BAKEABLE_CHANNELS {
                        if widgets::chip(ui, channel_label(channel), false)
                            .on_hover_text(format!(
                                "Export {} as {}",
                                channel_label(channel),
                                self.selected_format().name()
                            ))
                            .clicked()
                        {
                            self.export_baked_map(channel);
                        }
                    }
                });
            },
        );
    }

    fn layer_properties(&mut self, ui: &mut Ui) {
        let mesh_maps_needed = self.selected_layer().is_some_and(|layer| {
            layer
                .mask_generator
                .is_some_and(|generator| generator.needs_mesh_maps())
                || layer
                    .fill
                    .is_some_and(|fill| fill.projection.needs_mesh_maps())
                || layer
                    .mask_bitmap
                    .is_some_and(|mask| mask.projection.needs_mesh_maps())
        });
        let maps_state = if self.mesh_map_job_running() {
            Some(("Baking mesh maps…", theme::ACCENT))
        } else if mesh_maps_needed && !self.mesh_maps_fresh() {
            Some(("Mesh maps needed", WARNING))
        } else {
            None
        };
        let brush = self.brush_color.to_array();
        let triangle_count = self
            .mesh_geometry
            .as_ref()
            .map_or(0, |mesh| mesh.triangles.len());
        let resources = self
            .project
            .resources
            .iter()
            .map(|resource| (resource.id, resource.name.clone()))
            .collect::<Vec<_>>();
        let anchor_candidates: Vec<(LayerId, String)> = self
            .selected_texture_set_ref()
            .zip(self.selected_layer)
            .map(|(set, selected)| {
                set.walk_layers()
                    .into_iter()
                    .filter(|candidate| set.anchor_is_valid(selected, candidate.id))
                    .map(|candidate| (candidate.id, candidate.name.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let mesh_tiles: Vec<substancecraft_core::UdimTile> = self
            .mesh_geometry
            .as_deref()
            .map(|mesh| mesh.udim_tiles().into_iter().collect())
            .unwrap_or_default();
        let smart_masks = self.smart_mask_library();
        let mut save_smart_mask = false;
        let recording = self
            .evolution_start
            .as_ref()
            .is_some_and(|(set, layer, _)| {
                Some(*set) == self.selected_texture_set && Some(*layer) == self.selected_layer
            });
        let mut evolution_name = self.evolution_name.clone();
        let instance = self.selected_layer().and_then(|layer| {
            let link = layer.material_link.clone()?;
            let master = self
                .project
                .material(link.material)
                .map_or_else(|| "Missing material".to_owned(), |m| m.name.clone());
            let shared = self.project.material_instances(link.material).len();
            Some((link, master, shared))
        });
        let overrides = instance.as_ref().map(|(link, ..)| link.overrides.clone());
        let mut instance_action = None;
        let mut decal_gizmo = self.decal_gizmo;
        let Some(layer) = self.selected_layer_mut() else {
            return;
        };

        widgets::caption(ui, if layer.is_group() { "Group" } else { "Layer" });
        widgets::property_row(ui, "Name", |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut layer.name)
                    .desired_width(ui.available_width())
                    .margin(vec2(8.0, 3.0)),
            );
        });
        slider_row(ui, "Opacity", &mut layer.opacity, 0.0..=1.0, false, 2);
        widgets::property_row(ui, "Blend", |ui| {
            let combo = egui::ComboBox::from_id_salt(("blend_mode", layer.id.0))
                .width(ui.available_width() - 8.0)
                .selected_text(layer.blend_mode.name())
                .show_ui(ui, |ui| {
                    for mode in substancecraft_core::BlendMode::ALL {
                        ui.selectable_value(&mut layer.blend_mode, mode, mode.name());
                    }
                    if layer.children.is_some() {
                        let mode = substancecraft_core::BlendMode::PassThrough;
                        ui.selectable_value(&mut layer.blend_mode, mode, mode.name())
                            .on_hover_text(
                                "Children blend straight onto the layers beneath, as if ungrouped",
                            );
                    }
                })
                .response
                .on_hover_text("How this layer combines with the layers beneath; normals always blend normally");
            widgets::label_from_row(ui, &combo, egui::WidgetType::ComboBox);
        });
        if layer.blend_mode != substancecraft_core::BlendMode::PassThrough {
            egui::CollapsingHeader::new(
                RichText::new(if layer.channel_blend_modes.is_empty() {
                    "Blend per channel".to_owned()
                } else {
                    format!("Blend per channel ({})", layer.channel_blend_modes.len())
                })
                .size(12.0)
                .color(theme::TEXT_WEAK),
            )
            .id_salt(("channel_blend", layer.id.0))
            .show(ui, |ui| {
                for channel in [
                    MaterialChannel::BaseColor,
                    MaterialChannel::Roughness,
                    MaterialChannel::Metallic,
                    MaterialChannel::Height,
                    MaterialChannel::Emissive,
                    MaterialChannel::Opacity,
                ] {
                    let current = layer.channel_blend_modes.get(&channel).copied();
                    let mut chosen = current;
                    widgets::property_row(ui, channel_label(channel), |ui| {
                        let combo = egui::ComboBox::from_id_salt((
                            "channel_blend_mode",
                            layer.id.0,
                            channel,
                        ))
                        .width(ui.available_width() - 8.0)
                        .selected_text(current.map_or("Layer mode", |mode| mode.name()))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut chosen, None, "Layer mode");
                            for mode in substancecraft_core::BlendMode::ALL {
                                ui.selectable_value(&mut chosen, Some(mode), mode.name());
                            }
                        });
                        widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
                    });
                    if chosen != current {
                        match chosen {
                            Some(mode) => {
                                layer.channel_blend_modes.insert(channel, mode);
                            }
                            None => {
                                layer.channel_blend_modes.remove(&channel);
                            }
                        }
                    }
                }
            });
        }
        let channels_label =
            override_label(overrides.as_ref(), InstanceField::Channels, "Channels");
        widgets::property_row(ui, &channels_label, |ui| {
            // Wrap so the chips never widen the inspector column.
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                for (channel, label) in [
                    (MaterialChannel::BaseColor, "Color"),
                    (MaterialChannel::Roughness, "Rough"),
                    (MaterialChannel::Metallic, "Metal"),
                    (MaterialChannel::Height, "Hgt"),
                    (MaterialChannel::Normal, "Nrm"),
                    (MaterialChannel::Emissive, "Emit"),
                    (MaterialChannel::Opacity, "Opac"),
                    (MaterialChannel::Id, "ID"),
                ] {
                    let enabled = layer.channels.contains(&channel);
                    if widgets::chip(ui, label, enabled).clicked() {
                        if enabled {
                            layer.channels.remove(&channel);
                        } else {
                            layer.channels.insert(channel);
                        }
                    }
                }
            });
        });
        let evolution_action = evolution_controls(ui, layer, recording, &mut evolution_name);
        let exposed_action = exposed_controls(ui, layer);
        if layer.is_group() {
            hint(
                ui,
                "Children composite first; the group's opacity, mask, and filters then \
                 apply to the result. Only enabled channels pass through.",
            );
        } else if let Some(fill) = layer.fill.as_mut() {
            if let Some((link, master, shared)) = &instance {
                instance_action = instance_controls(ui, link, master, *shared);
            }
            fill_controls(
                ui,
                Id::new(("layer_fill", layer.id.0)),
                fill,
                &resources,
                overrides.as_ref(),
            );
            projection_controls(ui, &mut fill.projection, 0, &mut decal_gizmo);
        } else {
            ui.add_space(2.0);
            if widgets::pill_button(ui, Some(Icon::Plus), "Add fill", false, true)
                .on_hover_text("Cover this layer's mask with uniform or textured material")
                .clicked()
            {
                layer.fill = Some(LayerFill {
                    color: brush,
                    ..LayerFill::default()
                });
            }
        }

        widgets::caption(ui, "Filters");
        filter_stack(
            ui,
            Id::new(("layer_filters", layer.id.0)),
            &mut layer.filters,
        );
        if !layer.filters.is_empty() {
            widgets::property_row(ui, "On channels", |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                    if widgets::chip(ui, "All", layer.filter_channels.is_empty())
                        .on_hover_text("Filter every channel the layer writes")
                        .clicked()
                    {
                        layer.filter_channels.clear();
                    }
                    for (channel, label) in [
                        (MaterialChannel::BaseColor, "Color"),
                        (MaterialChannel::Roughness, "Rough"),
                        (MaterialChannel::Metallic, "Metal"),
                        (MaterialChannel::Height, "Hgt"),
                        (MaterialChannel::Normal, "Nrm"),
                        (MaterialChannel::Emissive, "Emit"),
                        (MaterialChannel::Opacity, "Opac"),
                    ] {
                        let on = layer.filter_channels.contains(&channel);
                        if widgets::chip(ui, label, on)
                            .on_hover_text("Limit the filters to the chosen channels")
                            .clicked()
                        {
                            if on {
                                layer.filter_channels.remove(&channel);
                            } else {
                                layer.filter_channels.insert(channel);
                            }
                        }
                    }
                });
            });
        }

        widgets::caption(ui, "Mask generator");
        let current = layer.mask_generator;
        let mut chosen = current;
        let combo = egui::ComboBox::from_id_salt(("generator_kind", layer.id.0))
            .width(ui.available_width() - 8.0)
            .selected_text(current.map_or("None", |generator| generator.name()))
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "None").clicked() {
                    chosen = None;
                }
                for preset in MaskGenerator::PRESETS {
                    let selected = current.is_some_and(|generator| generator.same_kind(&preset));
                    if ui.selectable_label(selected, preset.name()).clicked() && !selected {
                        chosen = Some(preset);
                    }
                }
            });
        widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
        if chosen != current {
            layer.mask_generator = chosen;
        }
        if let Some((text, color)) = maps_state {
            ui.horizontal(|ui| {
                if color == theme::ACCENT {
                    ui.add(egui::Spinner::new().size(11.0));
                }
                ui.label(RichText::new(text).size(12.0).color(color));
            });
        }
        let pick_id_color = layer
            .mask_generator
            .as_mut()
            .is_some_and(|generator| generator_controls(ui, generator));
        widgets::property_row(ui, "Smart mask", |ui| {
            let mut applied = None;
            let combo = egui::ComboBox::from_id_salt(("smart_mask", layer.id.0))
                .width((ui.available_width() - 60.0).max(60.0))
                .selected_text("Apply…")
                .show_ui(ui, |ui| {
                    for mask in &smart_masks {
                        if ui
                            .selectable_label(false, &mask.name)
                            .on_hover_text(&mask.description)
                            .clicked()
                        {
                            applied = Some(mask);
                        }
                    }
                })
                .response
                .on_hover_text(
                    "Replace the generator, bitmap mask, mask filters, and inversion; \
                     painted mask strokes and selections stay",
                );
            widgets::label_from_row(ui, &combo, egui::WidgetType::ComboBox);
            if let Some(mask) = applied {
                mask.apply_to(layer);
            }
            let reusable = layer.mask_generator.is_some()
                || layer.mask_bitmap.is_some()
                || !layer.mask_filters.is_empty();
            save_smart_mask = ui
                .add_enabled_ui(reusable, |ui| {
                    widgets::chip(ui, "Save", false)
                        .on_hover_text("Keep this mask setup as a smart mask in the project")
                        .clicked()
                })
                .inner;
        });

        widgets::caption(ui, "Bitmap mask");
        let mut bitmap = layer.mask_bitmap.map(|mask| mask.resource);
        if resource_picker(
            ui,
            Id::new(("mask_bitmap", layer.id.0)),
            &mut bitmap,
            &resources,
        ) {
            layer.mask_bitmap = bitmap.map(|resource| BitmapMask {
                resource,
                projection: layer
                    .mask_bitmap
                    .map_or_else(TextureProjection::default, |mask| mask.projection),
                invert: layer.mask_bitmap.is_some_and(|mask| mask.invert),
            });
        }
        if let Some(mask) = layer.mask_bitmap.as_mut() {
            widgets::property_row(ui, "Invert", |ui| widgets::toggle(ui, &mut mask.invert));
            projection_controls(ui, &mut mask.projection, 1, &mut decal_gizmo);
        }

        anchor_controls(ui, layer.id, &mut layer.mask_anchor, &anchor_candidates);

        widgets::property_row(ui, "Invert final mask", |ui| {
            widgets::toggle(ui, &mut layer.invert_final_mask)
        });

        widgets::caption(ui, "Mask filters");
        filter_stack(
            ui,
            Id::new(("mask_filters", layer.id.0)),
            &mut layer.mask_filters,
        );

        widgets::caption(ui, "Polygon mask");
        match layer.polygon_mask.as_mut() {
            None => {
                hint(
                    ui,
                    "Use the Select tool to limit this layer to parts of the mesh.",
                );
            }
            Some(mask) => {
                let count = if mask.inverted {
                    triangle_count.saturating_sub(mask.triangles.len())
                } else {
                    mask.triangles.len()
                };
                widgets::property_row(ui, "Triangles", |ui| {
                    ui.label(format!("{count} of {triangle_count}"))
                });
                widgets::property_row(ui, "Invert", |ui| widgets::toggle(ui, &mut mask.inverted));
                if widgets::pill_button(ui, Some(Icon::Close), "Remove polygon mask", false, true)
                    .clicked()
                {
                    layer.polygon_mask = None;
                }
            }
        }

        if mesh_tiles.len() > 1 || !layer.uv_tiles.is_empty() {
            widgets::caption(ui, "UV tiles");
            widgets::property_row(ui, "Paints in", |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(3.0, 4.0);
                    if widgets::chip(ui, "All", layer.uv_tiles.is_empty())
                        .on_hover_text("Paint in every UDIM tile")
                        .clicked()
                    {
                        layer.uv_tiles.clear();
                    }
                    let tiles: std::collections::BTreeSet<_> =
                        mesh_tiles.iter().chain(&layer.uv_tiles).copied().collect();
                    for tile in tiles {
                        let on = layer.uv_tiles.contains(&tile);
                        if widgets::chip(ui, &tile.0.to_string(), on)
                            .on_hover_text("Limit this layer to the chosen tiles")
                            .clicked()
                        {
                            if on {
                                layer.uv_tiles.remove(&tile);
                            } else {
                                layer.uv_tiles.insert(tile);
                            }
                        }
                    }
                });
            });
        }

        widgets::caption(ui, "ID clip");
        let mut enabled = layer.id_mask_color.is_some();
        let color = layer
            .id_mask_color
            .unwrap_or([brush[0], brush[1], brush[2]]);
        let mut color = Color32::from_rgb(color[0], color[1], color[2]);
        let changed = widgets::property_row(ui, "Clip by ID", |ui| {
            let mut changed = widgets::toggle(ui, &mut enabled).changed();
            if enabled {
                ui.add_space(4.0);
                changed |= widgets::labeled_color_button(ui, &mut color).changed();
            }
            changed
        });
        if changed {
            let channels = color.to_array();
            layer.id_mask_color = enabled.then_some([channels[0], channels[1], channels[2]]);
        }

        if pick_id_color {
            self.picking_id_color = true;
        }
        self.decal_gizmo = decal_gizmo;
        if let Some((index, value)) = exposed_action {
            self.set_exposed_parameter(index, value);
        }
        self.evolution_name = evolution_name;
        match evolution_action {
            Some(EvolutionAction::Move(index, amount)) => self.evolve_selected_layer(index, amount),
            Some(EvolutionAction::Remove(index)) => {
                if let Some(layer) = self.selected_layer_mut()
                    && index < layer.evolutions.len()
                {
                    layer.evolutions.remove(index);
                    self.invalidate_preview();
                }
            }
            Some(EvolutionAction::Start) => self.start_evolution_recording(),
            Some(EvolutionAction::Finish) => self.finish_evolution_recording(),
            Some(EvolutionAction::Cancel) => self.evolution_start = None,
            None => {}
        }
        if save_smart_mask {
            self.save_layer_as_smart_mask();
        }
        match instance_action {
            Some(InstanceAction::Reset(field)) => self.reset_instance_override(field),
            Some(InstanceAction::Unlink) => self.unlink_selected_layer(),
            Some(InstanceAction::EditMaster(id)) => {
                self.editing_material = Some(id);
                self.drawer = Some(Drawer::Assets);
            }
            None => {}
        }
    }

    fn show_status(&self, context: &egui::Context, screen: Rect) {
        egui::Area::new(Id::new("overlay_status"))
            .order(Order::Middle)
            .pivot(Align2::LEFT_BOTTOM)
            .fixed_pos(pos2(
                screen.left() + theme::EDGE,
                screen.bottom() - theme::EDGE,
            ))
            .show(context, |ui| {
                theme::pill_frame()
                    .inner_margin(Margin::symmetric(14, 7))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 8.0;
                            let live =
                                self.filament_renderer.is_some() && self.filament_error.is_none();
                            let (color, renderer) = if live {
                                (theme::ACCENT, "Filament")
                            } else {
                                (WARNING, "Software preview")
                            };
                            let (dot, _) = ui.allocate_exact_size(vec2(8.0, 8.0), Sense::hover());
                            ui.painter().circle_filled(dot.center(), 4.0, color);
                            status_text(ui, renderer, theme::TEXT);
                            status_separator(ui);
                            let mesh = self.selected_mesh_name().unwrap_or("No mesh");
                            status_text(ui, mesh, theme::TEXT);
                            if let Some(geometry) = &self.mesh_geometry {
                                status_text(
                                    ui,
                                    &format!("{} tris", geometry.triangles.len()),
                                    theme::TEXT_FAINT,
                                );
                            }
                            status_separator(ui);
                            let preview = self.preview_label();
                            status_text(ui, preview, theme::TEXT_WEAK);
                            if self.mesh_map_job_running() {
                                status_separator(ui);
                                ui.add(egui::Spinner::new().size(11.0));
                                status_text(ui, "Baking mesh maps", theme::ACCENT);
                            }
                            status_separator(ui);
                            let hint = match self.paint_mode {
                                PaintMode::View => {
                                    "Drag to orbit · Shift-drag to pan · Scroll to zoom"
                                }
                                PaintMode::Paint => "Painting material",
                                PaintMode::Erase => "Erasing layer paint",
                                PaintMode::Mask => "Painting layer mask",
                                PaintMode::Id => "Painting ID mask",
                                PaintMode::Select => {
                                    "Click to select · Shift-click to deselect · Drag to orbit"
                                }
                            };
                            status_text(ui, hint, theme::TEXT_FAINT);
                        });
                    });
            });
    }

    fn show_toasts(&mut self, context: &egui::Context, screen: Rect) {
        let errors = [
            (ErrorSlot::Project, &self.project_error),
            (ErrorSlot::Import, &self.import_error),
            (ErrorSlot::Mesh, &self.mesh_error),
            (ErrorSlot::Bake, &self.bake_error),
            (ErrorSlot::MeshMaps, &self.mesh_map_error),
            (ErrorSlot::Renderer, &self.filament_error),
        ]
        .into_iter()
        .filter_map(|(slot, error)| error.clone().map(|error| (slot, error)))
        .collect::<Vec<_>>();
        let pending_reloads = self.pending_reload_names();
        if errors.is_empty() && pending_reloads.is_empty() {
            return;
        }

        let (mut reload, mut ignore) = (false, false);
        let mut dismissed = Vec::new();
        egui::Area::new(Id::new("overlay_toasts"))
            .order(Order::Foreground)
            .pivot(Align2::CENTER_BOTTOM)
            .fixed_pos(pos2(
                screen.center().x,
                screen.bottom() - theme::EDGE - STATUS_RESERVE,
            ))
            .show(context, |ui| {
                ui.with_layout(Layout::top_down(Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.y = 8.0;
                    if !pending_reloads.is_empty() {
                        egui::Frame::new()
                            .fill(theme::ACCENT_SOFT)
                            .stroke(egui::Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.45)))
                            .corner_radius(theme::PILL)
                            .shadow(theme::panel_shadow())
                            .inner_margin(Margin {
                                left: 14,
                                right: 6,
                                top: 5,
                                bottom: 5,
                            })
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.set_max_width(560.0);
                                    small_icon(ui, Icon::Image, theme::ACCENT);
                                    let text =
                                        format!("{} changed on disk", pending_reloads.join(", "));
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(text).size(12.5).color(theme::TEXT),
                                        )
                                        .wrap(),
                                    );
                                    reload = widgets::chip(ui, "Reload", true).clicked();
                                    ignore = widgets::chip(ui, "Ignore", false).clicked();
                                });
                            });
                    }
                    for (slot, error) in errors {
                        egui::Frame::new()
                            .fill(theme::DANGER_SOFT)
                            .stroke(egui::Stroke::new(1.0, theme::DANGER.gamma_multiply(0.45)))
                            .corner_radius(theme::PILL)
                            .shadow(theme::panel_shadow())
                            .inner_margin(Margin {
                                left: 14,
                                right: 6,
                                top: 5,
                                bottom: 5,
                            })
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.set_max_width(520.0);
                                    small_icon(ui, Icon::Alert, theme::DANGER);
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(error).size(12.5).color(theme::TEXT),
                                        )
                                        .wrap(),
                                    );
                                    if widgets::header_action(ui, Icon::Close, true, "Dismiss")
                                        .clicked()
                                    {
                                        dismissed.push(slot);
                                    }
                                });
                            });
                    }
                });
            });

        if reload {
            let pending: Vec<ResourceId> = self.resource_watcher.pending.iter().copied().collect();
            self.reload_changed_resources(&pending);
        } else if ignore {
            self.resource_watcher.pending.clear();
        }
        for slot in dismissed {
            match slot {
                ErrorSlot::Project => self.project_error = None,
                ErrorSlot::Import => self.import_error = None,
                ErrorSlot::Mesh => self.mesh_error = None,
                ErrorSlot::Bake => {
                    self.bake_error = None;
                    self.invalidate_preview();
                }
                ErrorSlot::MeshMaps => self.mesh_map_error = None,
                ErrorSlot::Renderer => self.filament_error = None,
            }
        }
    }
}

/// Parameter sliders for one mask generator.
/// Settings of a mask generator. Returns true when the ID generator asks
/// to pick its colour from the model.
fn generator_controls(ui: &mut Ui, generator: &mut MaskGenerator) -> bool {
    let mut pick = false;
    match generator {
        MaskGenerator::EdgeWear(settings) => {
            slider_row(ui, "Amount", &mut settings.amount, 0.0..=1.0, false, 2);
            slider_row(ui, "Contrast", &mut settings.contrast, 0.0..=1.0, false, 2);
            slider_row(ui, "Breakup", &mut settings.breakup, 0.0..=1.0, false, 2);
            noise_size_rows(
                ui,
                "Breakup scale",
                &mut settings.breakup_scale,
                &mut settings.feature_size,
            );
            seed_row(ui, &mut settings.seed);
        }
        MaskGenerator::Dirt(settings) => {
            slider_row(ui, "Amount", &mut settings.amount, 0.0..=1.0, false, 2);
            slider_row(ui, "Contrast", &mut settings.contrast, 0.0..=1.0, false, 2);
            slider_row(
                ui,
                "Cavity ↔ AO",
                &mut settings.cavity_weight,
                0.0..=1.0,
                false,
                2,
            );
            slider_row(ui, "Breakup", &mut settings.breakup, 0.0..=1.0, false, 2);
            noise_size_rows(
                ui,
                "Breakup scale",
                &mut settings.breakup_scale,
                &mut settings.feature_size,
            );
            seed_row(ui, &mut settings.seed);
        }
        MaskGenerator::Orientation(settings) => {
            widgets::property_row(ui, "Direction", |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(3.0, 4.0);
                    for direction in Direction::ALL {
                        if widgets::chip(ui, direction.name(), settings.direction == direction)
                            .clicked()
                        {
                            settings.direction = direction;
                        }
                    }
                });
            });
            slider_row(
                ui,
                "Spread°",
                &mut settings.spread_degrees,
                0.0..=180.0,
                false,
                0,
            );
            slider_row(ui, "Softness", &mut settings.softness, 0.0..=1.0, false, 2);
        }
        MaskGenerator::Gradient(settings) => {
            widgets::property_row(ui, "Axis", |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                for axis in Axis::ALL {
                    if widgets::chip(ui, axis.name(), settings.axis == axis).clicked() {
                        settings.axis = axis;
                    }
                }
            });
            slider_row(ui, "Start", &mut settings.start, 0.0..=1.0, false, 2);
            slider_row(ui, "End", &mut settings.end, 0.0..=1.0, false, 2);
        }
        MaskGenerator::Noise(settings) => {
            noise_size_rows(ui, "Scale", &mut settings.scale, &mut settings.feature_size);
            slider_row(ui, "Amount", &mut settings.amount, 0.0..=1.0, false, 2);
            slider_row(ui, "Contrast", &mut settings.contrast, 0.0..=1.0, false, 2);
            seed_row(ui, &mut settings.seed);
        }
        MaskGenerator::Thickness(settings) => {
            widgets::property_row(ui, "Select", |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                if widgets::chip(ui, "Thin", settings.thin).clicked() {
                    settings.thin = true;
                }
                if widgets::chip(ui, "Thick", !settings.thin).clicked() {
                    settings.thin = false;
                }
            });
            slider_row(ui, "Amount", &mut settings.amount, 0.0..=1.0, false, 2);
            slider_row(ui, "Contrast", &mut settings.contrast, 0.0..=1.0, false, 2);
        }
        MaskGenerator::Pattern(settings) => {
            widgets::property_row(ui, "Kind", |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(3.0, 4.0);
                    for kind in substancecraft_core::PatternKind::ALL {
                        if widgets::chip(ui, kind.name(), settings.kind == kind).clicked() {
                            settings.kind = kind;
                        }
                    }
                });
            });
            if matches!(
                settings.kind,
                substancecraft_core::PatternKind::Stripes
                    | substancecraft_core::PatternKind::Scratches
            ) {
                widgets::property_row(ui, "Axis", |ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for axis in Axis::ALL {
                        if widgets::chip(ui, axis.name(), settings.axis == axis).clicked() {
                            settings.axis = axis;
                        }
                    }
                });
            }
            noise_size_rows(ui, "Scale", &mut settings.scale, &mut settings.feature_size);
            slider_row(ui, "Width", &mut settings.width, 0.0..=1.0, false, 2);
            slider_row(ui, "Softness", &mut settings.softness, 0.0..=1.0, false, 2);
            seed_row(ui, &mut settings.seed);
        }
        MaskGenerator::IdColor(settings) => {
            let [r, g, b] = settings.color;
            let mut color = Color32::from_rgb(r, g, b);
            widgets::property_row(ui, "ID colour", |ui| {
                if widgets::labeled_color_button(ui, &mut color).changed() {
                    let [r, g, b, _] = color.to_array();
                    settings.color = [r, g, b];
                }
                ui.add_space(4.0);
                pick = widgets::chip(ui, "Pick", false)
                    .on_hover_text("Click the model to take the ID colour under the cursor")
                    .clicked();
            });
            slider_row(
                ui,
                "Tolerance",
                &mut settings.tolerance,
                0.0..=1.0,
                false,
                2,
            );
            slider_row(ui, "Softness", &mut settings.softness, 0.0..=1.0, false, 2);
            hint(
                ui,
                "Selects texels of this set's ID map near the colour. Paint IDs in ID mode.",
            );
        }
    }
    pick
}

/// Noise size: a frequency relative to the mesh, or with Real size on, a
/// feature size in scene units (metres).
fn noise_size_rows(ui: &mut Ui, label: &str, scale: &mut f32, size: &mut Option<f32>) {
    let mut physical = size.is_some();
    widgets::property_row(ui, "Real size", |ui| {
        widgets::toggle(ui, &mut physical).on_hover_text(
            "Size the noise in scene units (metres) so it keeps its real size on any mesh",
        );
    });
    if physical != size.is_some() {
        *size = physical.then_some(0.1);
    }
    match size {
        Some(size) => {
            slider_row(ui, "Feature m", size, MIN_TILE_SIZE..=10.0, true, 3);
        }
        None => {
            slider_row(ui, label, scale, 0.5..=64.0, true, 1);
        }
    }
}

fn seed_row(ui: &mut Ui, seed: &mut u32) {
    widgets::property_row(ui, "Seed", |ui| {
        ui.add(egui::DragValue::new(seed).range(0..=9_999));
    });
}

/// Dock tab: icon above a short label inside a rounded hover plate.
fn dock_item(ui: &mut Ui, icon: Icon, label: &str, selected: bool) -> Response {
    let (rect, response) = ui.allocate_exact_size(vec2(54.0, 52.0), Sense::click());
    if ui.is_rect_visible(rect) {
        let hover = ui
            .ctx()
            .animate_bool_responsive(response.id, response.hovered());
        let fill = if selected {
            theme::ACCENT_SOFT
        } else {
            theme::HOVER.gamma_multiply(hover)
        };
        let painter = ui.painter();
        painter.rect_filled(rect, CornerRadius::same(20), fill);
        let color = if selected {
            theme::ACCENT
        } else if response.hovered() {
            Color32::WHITE
        } else {
            theme::TEXT
        };
        let icon_rect =
            Rect::from_center_size(rect.center_top() + vec2(0.0, 19.0), vec2(20.0, 20.0));
        icons::paint(painter, icon_rect, icon, color);
        painter.text(
            rect.center_bottom() - vec2(0.0, 10.0),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(10.5),
            if selected {
                theme::ACCENT
            } else {
                theme::TEXT_WEAK
            },
        );
    }
    response.on_hover_text(label)
}

fn small_icon(ui: &mut Ui, icon: Icon, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
    icons::paint(ui.painter(), rect, icon, color);
}

fn empty_state(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).size(12.5).color(theme::TEXT_FAINT));
}

fn hint(ui: &mut Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).size(11.5).color(theme::TEXT_WEAK)).wrap());
}

fn status_text(ui: &mut Ui, text: &str, color: Color32) {
    ui.label(RichText::new(text).size(12.0).color(color));
}

fn status_separator(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(vec2(4.0, 12.0), Sense::hover());
    ui.painter()
        .circle_filled(rect.center(), 1.5, theme::TEXT_FAINT);
}

fn slider_row(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    logarithmic: bool,
    decimals: usize,
) -> bool {
    widgets::property_row(ui, label, |ui| {
        ui.spacing_mut().slider_width = (ui.available_width() - 60.0).max(48.0);
        let response = ui.add(
            egui::Slider::new(value, range)
                .logarithmic(logarithmic)
                .fixed_decimals(decimals),
        );
        widgets::label_from_row(ui, &response, egui::WidgetType::Slider);
        response.changed()
    })
}

const BRUSH_PREVIEW_CELLS: usize = 32;

fn brush_shape_samples(
    tip: &substancecraft_core::BrushTip,
    resources: &substancecraft_core::ResourceImages,
) -> Vec<f32> {
    tip.preview(BRUSH_PREVIEW_CELLS, Some(resources))
}

/// Texture set sizes offered. Sets above 2K bake in tiles; above 4K the
/// viewport shows a downscaled preview while exports keep full size.
const TEXTURE_SET_RESOLUTIONS: [u32; 5] = [512, 1024, 2048, 4096, 8192];

/// One row of the layer list: layers top first, each group followed by its
/// children one level deeper.
struct LayerRow {
    id: LayerId,
    depth: usize,
    name: String,
    visible: bool,
    opacity: f32,
    group: bool,
    badge: String,
}

fn layer_rows(
    layers: &[MaterialLayer],
    depth: usize,
    query: &str,
    resources: &[(ResourceId, String)],
) -> Vec<LayerRow> {
    let query = query
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>();
    filtered_layer_rows(layers, depth, &query, resources)
}

fn filtered_layer_rows(
    layers: &[MaterialLayer],
    depth: usize,
    query: &[String],
    resources: &[(ResourceId, String)],
) -> Vec<LayerRow> {
    let mut rows = Vec::new();
    for layer in layers.iter().rev() {
        let kind = if layer.is_group() {
            "Group"
        } else {
            match (&layer.fill, &layer.mask_generator) {
                (_, Some(generator)) => generator.name(),
                (Some(_), None) => "Fill",
                (None, None) => "Paint",
            }
        };
        let searchable = std::iter::once(layer.name.to_lowercase())
            .chain(std::iter::once(kind.to_lowercase()))
            .chain(
                layer
                    .channels
                    .iter()
                    .map(|channel| channel_label(*channel).to_lowercase()),
            )
            .chain(layer.fill.iter().flat_map(|fill| {
                fill.textures.all().filter_map(|id| {
                    resources
                        .iter()
                        .find(|(resource_id, _)| *resource_id == id)
                        .map(|(_, name)| name.to_lowercase())
                })
            }))
            .collect::<Vec<_>>();
        let matches = query.is_empty()
            || query
                .iter()
                .all(|token| searchable.iter().any(|field| field.contains(token)));
        let mut badge = kind.to_owned();
        if layer.polygon_mask.is_some() {
            badge.push_str(" · Poly");
        }
        if !layer.filters.is_empty() {
            badge.push_str(" · Fx");
        }
        let children = layer
            .children
            .as_deref()
            .map(|children| filtered_layer_rows(children, depth + 1, query, resources))
            .unwrap_or_default();
        if matches || !children.is_empty() {
            rows.push(LayerRow {
                id: layer.id,
                depth,
                name: layer.name.clone(),
                visible: layer.visible,
                opacity: layer.opacity,
                group: layer.is_group(),
                badge,
            });
            rows.extend(children);
        }
    }
    rows
}

/// Uniform values, per-channel bitmaps, and projection of a fill.
/// `label`, marked with `*` when an instance overrides `field`.
fn override_label(
    overrides: Option<&std::collections::BTreeSet<InstanceField>>,
    field: InstanceField,
    label: &str,
) -> String {
    if overrides.is_some_and(|overrides| overrides.contains(&field)) {
        format!("{label}*")
    } else {
        label.to_owned()
    }
}

/// What the evolution controls asked for.
enum EvolutionAction {
    Move(usize, f32),
    Remove(usize),
    Start,
    Finish,
    Cancel,
}

/// A control per exposed parameter of an inserted smart material: a
/// slider for amounts and values, a colour button for colours. Returns the
/// parameter changed this frame.
fn exposed_controls(
    ui: &mut Ui,
    layer: &MaterialLayer,
) -> Option<(usize, substancecraft_core::ParameterValue)> {
    use substancecraft_core::ParameterValue;

    let values = layer.exposed_values();
    if values.is_empty() {
        return None;
    }
    widgets::caption(ui, "Parameters");
    let mut action = None;
    for (index, parameter, value) in values {
        match value {
            ParameterValue::Scalar(mut amount) => {
                if slider_row(ui, &parameter.name, &mut amount, 0.0..=1.0, false, 2) {
                    action = Some((index, ParameterValue::Scalar(amount)));
                }
            }
            ParameterValue::Color([r, g, b]) => {
                let mut color = Color32::from_rgb(r, g, b);
                if widgets::property_row(ui, &parameter.name, |ui| {
                    widgets::labeled_color_button(ui, &mut color).changed()
                }) {
                    let [r, g, b, _] = color.to_array();
                    action = Some((index, ParameterValue::Color([r, g, b])));
                }
            }
        }
    }
    action
}

/// A slider per evolution control of `layer`; on groups, recording a new
/// control from a start and an end look.
fn evolution_controls(
    ui: &mut Ui,
    layer: &MaterialLayer,
    recording: bool,
    name: &mut String,
) -> Option<EvolutionAction> {
    if layer.evolutions.is_empty() && !layer.is_group() {
        return None;
    }
    let mut action = None;
    widgets::caption(ui, "Controls");
    for (index, evolution) in layer.evolutions.iter().enumerate() {
        let mut amount = evolution.amount;
        if slider_row(ui, &evolution.name, &mut amount, 0.0..=1.0, false, 2) {
            action = Some(EvolutionAction::Move(index, amount));
        }
    }
    if !layer.evolutions.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            for (index, evolution) in layer.evolutions.iter().enumerate() {
                if widgets::chip(ui, &format!("{} ×", evolution.name), false)
                    .on_hover_text("Remove this control; the layers keep their current values")
                    .clicked()
                {
                    action = Some(EvolutionAction::Remove(index));
                }
            }
        });
    }
    if !layer.is_group() {
        return action;
    }
    if recording {
        hint(
            ui,
            "Now change the layers inside to the look at 1 (for example worn): fill values, \
             opacity, or generator amounts. Then name the control and finish.",
        );
        widgets::property_row(ui, "Name", |ui| {
            ui.add(
                egui::TextEdit::singleline(name)
                    .desired_width(ui.available_width())
                    .margin(vec2(8.0, 3.0)),
            );
        });
        ui.horizontal(|ui| {
            if widgets::pill_button(ui, None, "Finish control", true, false).clicked() {
                action = Some(EvolutionAction::Finish);
            }
            if widgets::pill_button(ui, None, "Cancel", false, false).clicked() {
                action = Some(EvolutionAction::Cancel);
            }
        });
    } else if layer.evolutions.len() < substancecraft_core::MAX_EVOLUTIONS
        && widgets::pill_button(ui, None, "Record control…", false, true)
            .on_hover_text(
                "Record this group as it is now (the look at 0), change it, and finish: one \
                 slider then moves every changed value between the two looks",
            )
            .clicked()
    {
        action = Some(EvolutionAction::Start);
    }
    action
}

/// What the material-instance controls asked for.
enum InstanceAction {
    Reset(InstanceField),
    Unlink,
    EditMaster(MaterialId),
}

/// The master a linked fill follows, its overrides (each resettable), and
/// unlinking.
fn instance_controls(
    ui: &mut Ui,
    link: &MaterialLink,
    master: &str,
    shared: usize,
) -> Option<InstanceAction> {
    let mut action = None;
    widgets::caption(ui, "Material instance");
    widgets::property_row(ui, "Master", |ui| {
        if ui
            .link(RichText::new(master).color(theme::ACCENT))
            .on_hover_text("Edit the master in the Assets drawer; every instance follows it")
            .clicked()
        {
            action = Some(InstanceAction::EditMaster(link.material));
        }
    });
    let layers = if shared == 1 {
        "this layer".to_owned()
    } else {
        format!("{shared} layers")
    };
    hint(
        ui,
        &format!(
            "Shared by {layers}. Fields marked * are overridden here; editing any other \
             material field overrides it too."
        ),
    );
    if !link.overrides.is_empty() {
        widgets::property_row(ui, "Overrides", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                for field in &link.overrides {
                    if widgets::chip(ui, &format!("{} ×", field.name()), true)
                        .on_hover_text("Reset to the master's value")
                        .clicked()
                    {
                        action = Some(InstanceAction::Reset(*field));
                    }
                }
            });
        });
    }
    if widgets::pill_button(ui, None, "Unlink from master", false, true)
        .on_hover_text("Keep the current values as a plain fill that no longer follows the master")
        .clicked()
    {
        action = Some(InstanceAction::Unlink);
    }
    action
}

/// Uniform values and per-channel bitmaps of a fill (a layer's or a master
/// material's). `overrides` marks the fields an instance overrides.
fn fill_controls(
    ui: &mut Ui,
    id_salt: Id,
    fill: &mut LayerFill,
    resources: &[(ResourceId, String)],
    overrides: Option<&std::collections::BTreeSet<InstanceField>>,
) {
    let marked = |field, label| override_label(overrides, field, label);
    widgets::caption(ui, "Fill");
    let mut color =
        Color32::from_rgba_unmultiplied(fill.color[0], fill.color[1], fill.color[2], fill.color[3]);
    widgets::property_row(ui, &marked(InstanceField::BaseColor, "Color"), |ui| {
        if widgets::labeled_color_button(ui, &mut color).changed() {
            fill.color = color.to_srgba_unmultiplied();
        }
    });
    slider_row(
        ui,
        &marked(InstanceField::Roughness, "Roughness"),
        &mut fill.roughness,
        0.0..=1.0,
        false,
        2,
    );
    slider_row(
        ui,
        &marked(InstanceField::Metallic, "Metallic"),
        &mut fill.metallic,
        0.0..=1.0,
        false,
        2,
    );
    slider_row(
        ui,
        &marked(InstanceField::Height, "Height"),
        &mut fill.height,
        0.0..=1.0,
        false,
        2,
    );
    let mut emissive = Color32::from_rgb(fill.emissive[0], fill.emissive[1], fill.emissive[2]);
    widgets::property_row(ui, &marked(InstanceField::Emissive, "Emissive"), |ui| {
        if widgets::labeled_color_button(ui, &mut emissive).changed() {
            let rgba = emissive.to_array();
            fill.emissive = [rgba[0], rgba[1], rgba[2]];
        }
    });
    slider_row(
        ui,
        &marked(InstanceField::Opacity, "Opacity ch."),
        &mut fill.material_opacity,
        0.0..=1.0,
        false,
        2,
    );

    widgets::caption(ui, "Fill textures");
    if resources.is_empty() {
        hint(
            ui,
            "Import bitmaps in the Assets drawer to texture this fill.",
        );
        return;
    }
    for channel in [
        MaterialChannel::BaseColor,
        MaterialChannel::Roughness,
        MaterialChannel::Metallic,
        MaterialChannel::Height,
        MaterialChannel::Normal,
        MaterialChannel::Emissive,
        MaterialChannel::Opacity,
    ] {
        let Some(slot) = fill.textures.slot_mut(channel) else {
            continue;
        };
        let label = InstanceField::for_channel(channel).map_or_else(
            || channel_label(channel).to_owned(),
            |field| marked(field, channel_label(channel)),
        );
        widgets::property_row(ui, &label, |ui| {
            resource_picker(ui, id_salt.with(("fill_texture", channel)), slot, resources);
        });
    }
}

/// Combo box choosing a bitmap resource or none. Returns true on change.
fn resource_picker(
    ui: &mut Ui,
    id_salt: Id,
    current: &mut Option<ResourceId>,
    resources: &[(ResourceId, String)],
) -> bool {
    let selected_text = current
        .and_then(|id| resources.iter().find(|(resource, _)| *resource == id))
        .map_or("None", |(_, name)| name.as_str());
    let mut chosen = *current;
    let combo = egui::ComboBox::from_id_salt(id_salt)
        .width(ui.available_width() - 8.0)
        .selected_text(selected_text)
        .show_ui(ui, |ui| {
            if ui.selectable_label(current.is_none(), "None").clicked() {
                chosen = None;
            }
            for (id, name) in resources {
                if ui.selectable_label(*current == Some(*id), name).clicked() {
                    chosen = Some(*id);
                }
            }
        });
    widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
    let changed = chosen != *current;
    *current = chosen;
    changed
}

/// egui memory key holding which decal (0 = fill, 1 = bitmap mask) the
/// next viewport click places.
pub(super) const PLACE_DECAL_KEY: &str = "place_decal_target";

fn projection_controls(
    ui: &mut Ui,
    projection: &mut TextureProjection,
    decal_target: u8,
    gizmo: &mut Option<u8>,
) {
    widgets::property_row(ui, "Projection", |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            for mode in ProjectionMode::ALL {
                if widgets::chip(ui, mode.name(), projection.mode == mode).clicked() {
                    projection.mode = mode;
                }
            }
        });
    });
    if matches!(
        projection.mode,
        ProjectionMode::Planar | ProjectionMode::Cylindrical | ProjectionMode::Spherical
    ) {
        widgets::property_row(ui, "Axis", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for axis in Axis::ALL {
                if widgets::chip(ui, axis.name(), projection.axis == axis).clicked() {
                    projection.axis = axis;
                }
            }
        });
    }
    if projection.mode == ProjectionMode::Decal {
        let decal = &mut projection.decal;
        if widgets::pill_button(ui, None, "Place on surface", true, true)
            .on_hover_text("Click the mesh to centre the decal there, facing out of the surface")
            .clicked()
        {
            ui.ctx()
                .data_mut(|data| data.insert_temp(Id::new(PLACE_DECAL_KEY), decal_target));
        }
        let moving = *gizmo == Some(decal_target);
        widgets::property_row(ui, "Gizmo", |ui| {
            if widgets::chip(ui, "Move in viewport", moving)
                .on_hover_text(
                    "Drag on the model to slide the decal along the surface; Shift-drag \
                     spins it, Ctrl-drag resizes it. Esc or a second click stops",
                )
                .clicked()
            {
                *gizmo = if moving { None } else { Some(decal_target) };
            }
        });
        // Keyboard placement: centre, facing, and spin without the mouse.
        widgets::property_row(ui, "Centre", |ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            for (axis, value) in ["X", "Y", "Z"].into_iter().zip(decal.center.as_mut()) {
                let response = ui.add(
                    egui::DragValue::new(value)
                        .speed(0.005)
                        .fixed_decimals(3)
                        .prefix(format!("{axis} ")),
                );
                response.widget_info(|| egui::WidgetInfo::drag_value(true, f64::from(*value)));
            }
        });
        widgets::property_row(ui, "Facing", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(3.0, 4.0);
                for (label, direction) in [
                    ("+X", glam::Vec3::X),
                    ("−X", glam::Vec3::NEG_X),
                    ("+Y", glam::Vec3::Y),
                    ("−Y", glam::Vec3::NEG_Y),
                    ("+Z", glam::Vec3::Z),
                    ("−Z", glam::Vec3::NEG_Z),
                ] {
                    let facing = decal.normal.normalize_or_zero().dot(direction) > 0.999;
                    if widgets::chip(ui, label, facing)
                        .on_hover_text("Face the decal along this axis")
                        .clicked()
                    {
                        decal.face(direction);
                    }
                }
            });
        });
        let mut spin = decal.spin_degrees();
        if slider_row(ui, "Spin°", &mut spin, -180.0..=180.0, false, 0) {
            decal.set_spin_degrees(spin);
        }
        slider_row(ui, "Width", &mut decal.size.x, 0.01..=4.0, true, 3);
        slider_row(ui, "Height", &mut decal.size.y, 0.01..=4.0, true, 3);
        slider_row(ui, "Depth", &mut decal.depth, 0.001..=1.0, true, 3);
        widgets::property_row(ui, "Atlas cell", |ui| {
            // A 2 x 2 atlas: pick a quadrant, or the whole image.
            ui.spacing_mut().item_spacing.x = 4.0;
            for (label, min) in [
                ("All", None),
                ("1", Some(glam::Vec2::new(0.0, 0.5))),
                ("2", Some(glam::Vec2::new(0.5, 0.5))),
                ("3", Some(glam::Vec2::ZERO)),
                ("4", Some(glam::Vec2::new(0.5, 0.0))),
            ] {
                let (region_min, region_max) = match min {
                    Some(min) => (min, min + glam::Vec2::splat(0.5)),
                    None => (glam::Vec2::ZERO, glam::Vec2::ONE),
                };
                let selected = decal.region_min == region_min && decal.region_max == region_max;
                if widgets::chip(ui, label, selected).clicked() {
                    decal.region_min = region_min;
                    decal.region_max = region_max;
                }
            }
        });
        return;
    }
    if projection.mode != ProjectionMode::Uv {
        let mut physical = projection.tile_size.is_some();
        widgets::property_row(ui, "Real size", |ui| {
            widgets::toggle(ui, &mut physical)
                .on_hover_text("Size one repetition in scene units (metres) instead of tiling");
        });
        if physical != projection.tile_size.is_some() {
            projection.tile_size = physical.then_some(0.5);
        }
        if let Some(size) = &mut projection.tile_size {
            slider_row(ui, "Tile size m", size, MIN_TILE_SIZE..=10.0, true, 3);
        }
    }
    if projection.tile_size.is_none() || projection.mode == ProjectionMode::Uv {
        slider_row(
            ui,
            "Tiling",
            &mut projection.scale,
            MIN_PROJECTION_SCALE..=MAX_PROJECTION_SCALE,
            true,
            2,
        );
    }
    slider_row(
        ui,
        "Rotation°",
        &mut projection.rotation_degrees,
        -180.0..=180.0,
        false,
        0,
    );
    widgets::property_row(ui, "Offset", |ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let width = (ui.available_width() - 4.0) / 2.0;
        for (axis, value) in [
            ("U", &mut projection.offset.x),
            ("V", &mut projection.offset.y),
        ] {
            ui.add_sized(
                vec2(width, 22.0),
                egui::DragValue::new(value)
                    .prefix(format!("{axis} "))
                    .speed(0.01)
                    .range(-64.0..=64.0),
            );
        }
    });
    if projection.mode == ProjectionMode::Triplanar {
        slider_row(
            ui,
            "Blend",
            &mut projection.blend_sharpness,
            1.0..=64.0,
            true,
            1,
        );
    }
}

/// Ordered filter list with per-filter controls, reordering, removal, and
/// an "add" picker.
fn filter_stack(ui: &mut Ui, id_salt: Id, filters: &mut Vec<LayerFilter>) {
    let mut remove = None;
    let mut raise = None;
    let count = filters.len();
    for (index, filter) in filters.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(RichText::new(filter.name()).color(theme::TEXT));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                if widgets::header_action(ui, Icon::Close, true, "Remove filter").clicked() {
                    remove = Some(index);
                }
                if widgets::header_action(ui, Icon::ArrowUp, index > 0, "Apply earlier").clicked() {
                    raise = Some(index);
                }
            });
        });
        filter_controls(ui, filter);
    }
    if let Some(index) = remove {
        filters.remove(index);
    } else if let Some(index) = raise.filter(|index| *index > 0 && *index < count) {
        filters.swap(index, index - 1);
    }
    let mut added = None;
    let combo = egui::ComboBox::from_id_salt(id_salt.with("add_filter"))
        .width(ui.available_width() - 8.0)
        .selected_text("+ Add filter")
        .show_ui(ui, |ui| {
            for preset in LayerFilter::PRESETS {
                if ui.selectable_label(false, preset.name()).clicked() {
                    added = Some(preset);
                }
            }
        });
    widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
    if let Some(filter) = added {
        filters.push(filter);
    }
}

fn filter_controls(ui: &mut Ui, filter: &mut LayerFilter) {
    match filter {
        LayerFilter::Blur { radius } => {
            slider_row(ui, "Radius", radius, 0.0..=MAX_BLUR_RADIUS, false, 1);
        }
        LayerFilter::Sharpen { radius, amount } => {
            slider_row(ui, "Radius", radius, 0.0..=MAX_BLUR_RADIUS, false, 1);
            slider_row(ui, "Amount", amount, 0.0..=4.0, false, 2);
        }
        LayerFilter::Levels {
            in_low,
            in_high,
            gamma,
            out_low,
            out_high,
        } => {
            slider_row(ui, "In low", in_low, 0.0..=1.0, false, 2);
            slider_row(ui, "In high", in_high, 0.0..=1.0, false, 2);
            slider_row(ui, "Gamma", gamma, 0.05..=10.0, true, 2);
            slider_row(ui, "Out low", out_low, 0.0..=1.0, false, 2);
            slider_row(ui, "Out high", out_high, 0.0..=1.0, false, 2);
        }
        LayerFilter::Invert => {}
        LayerFilter::HueSaturation {
            hue,
            saturation,
            lightness,
        } => {
            slider_row(ui, "Hue", hue, -0.5..=0.5, false, 2);
            slider_row(ui, "Saturation", saturation, 0.0..=2.0, false, 2);
            slider_row(ui, "Lightness", lightness, -1.0..=1.0, false, 2);
        }
        LayerFilter::BrightnessContrast {
            brightness,
            contrast,
        } => {
            slider_row(ui, "Brightness", brightness, -1.0..=1.0, false, 2);
            slider_row(ui, "Contrast", contrast, -1.0..=1.0, false, 2);
        }
        LayerFilter::Posterize { levels } => {
            widgets::property_row(ui, "Levels", |ui| {
                let response = ui.add(egui::Slider::new(levels, 2..=64));
                widgets::label_from_row(ui, &response, egui::WidgetType::Slider);
            });
        }
        LayerFilter::Threshold { level, softness } => {
            slider_row(ui, "Level", level, 0.0..=1.0, false, 2);
            slider_row(ui, "Softness", softness, 0.0..=1.0, false, 2);
        }
        LayerFilter::GradientMap { dark, light } => {
            for (label, color) in [("Dark", dark), ("Light", light)] {
                let mut swatch = Color32::from_rgb(color[0], color[1], color[2]);
                widgets::property_row(ui, label, |ui| {
                    if widgets::labeled_color_button(ui, &mut swatch).changed() {
                        let [r, g, b, _] = swatch.to_array();
                        *color = [r, g, b];
                    }
                });
            }
        }
    }
}

impl SubstanceCraftApp {
    /// Preset picker, the files the preset writes, the Export button, and
    /// the custom-preset editor.
    fn export_preset_controls(&mut self, ui: &mut Ui) {
        let remembered = self
            .selected_texture_set_ref()
            .is_some_and(|set| set.export.is_some());
        let mut remember = remembered;
        widgets::property_row(ui, "This set", |ui| {
            widgets::toggle(ui, &mut remember).on_hover_text(
                "Remember the preset, format, and UDIM choice for this texture set; \n                 Export all sets and the command line use it",
            );
        });
        if remember != remembered {
            self.remember_export_for_selected(remember);
        }
        let saved_udim = self
            .selected_texture_set_ref()
            .and_then(|set| set.export.as_ref())
            .map(|export| export.udim);
        if let (Some(saved), Some(id)) = (saved_udim, self.selected_texture_set) {
            let mut udim = saved;
            widgets::property_row(ui, "UDIM tiles", |ui| {
                widgets::toggle(ui, &mut udim).on_hover_text("Export one file set per UV tile");
            });
            if udim != saved {
                self.edit_set_properties(id, |properties| {
                    if let Some(export) = &mut properties.export {
                        export.udim = udim;
                    }
                });
            }
        }
        let presets = self.export_presets();
        let selected = self.selected_export_preset();
        widgets::property_row(ui, "Preset", |ui| {
            let combo = egui::ComboBox::from_id_salt("export_preset")
                .width(ui.available_width() - 8.0)
                .selected_text(selected.name.as_str())
                .show_ui(ui, |ui| {
                    for preset in &presets {
                        let label = if self.is_custom_preset(&preset.name) {
                            format!("{} (custom)", preset.name)
                        } else {
                            preset.name.clone()
                        };
                        if ui
                            .selectable_label(preset.name == selected.name, label)
                            .clicked()
                        {
                            self.set_preset_choice(preset.name.clone());
                        }
                    }
                });
            widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
        });
        let set_name = self
            .selected_texture_set_ref()
            .map_or_else(|| "Material".to_owned(), |set| set.name.clone());
        let format = self.selected_format();
        for output in &selected.resolved(format.extension()).outputs {
            let routing = output
                .used_slots()
                .iter()
                .zip(slot_names(output.layout))
                .map(|(slot, channel)| format!("{channel} {}", slot_label(*slot)))
                .collect::<Vec<_>>()
                .join(" · ");
            let optional = if output.skip_if_black {
                " (if not black)"
            } else {
                ""
            };
            let extension = output.format.as_deref().unwrap_or(format.extension());
            let depth = if output.bit_depth == BitDepth::Sixteen && extension == "png" {
                " 16-bit"
            } else {
                ""
            };
            ui.label(
                RichText::new(format!(
                    "{}.{extension}{depth}{optional}",
                    output.resolved_file_name(&set_name),
                ))
                .size(12.0)
                .color(theme::TEXT),
            );
            ui.label(RichText::new(routing).size(11.0).color(theme::TEXT_FAINT));
        }
        if let Some(texture_set) = self.selected_texture_set_ref().cloned() {
            ui.add_space(4.0);
            let issues = self.export_issues(&texture_set, &selected);
            if issues.is_empty() {
                ui.label(
                    RichText::new("Preflight: ready to export")
                        .size(11.5)
                        .color(theme::ACCENT),
                );
            }
            for issue in issues {
                let color = match issue.severity {
                    PreflightSeverity::Error => theme::DANGER,
                    PreflightSeverity::Warning => WARNING,
                };
                ui.horizontal_top(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    small_icon(ui, Icon::Alert, color);
                    ui.add(
                        egui::Label::new(
                            RichText::new(issue.message)
                                .size(11.5)
                                .color(theme::TEXT_WEAK),
                        )
                        .wrap(),
                    );
                });
            }
        }
        ui.add_space(4.0);
        let custom = self.is_custom_preset(&selected.name);
        ui.horizontal(|ui| {
            if widgets::pill_button(ui, None, "Duplicate", false, false)
                .on_hover_text("Copy this preset into an editable custom preset")
                .clicked()
            {
                self.duplicate_export_preset(&selected.name);
            }
            if custom {
                if widgets::pill_button(ui, None, "Edit", false, false).clicked() {
                    self.edit_export_preset(&selected.name);
                }
                if widgets::icon_button(ui, Icon::Trash, true, "Delete this custom preset")
                    .clicked()
                {
                    self.delete_export_preset(&selected.name);
                }
            }
        });
        if self.preset_draft.is_some() {
            self.preset_editor(ui);
        }
        ui.add_space(4.0);
        if widgets::pill_button(ui, Some(Icon::Export), "Export textures…", true, true)
            .on_hover_text(format!(
                "Write the {} files into a folder as {}",
                selected.name,
                self.selected_format().name()
            ))
            .clicked()
        {
            self.export_with_preset();
        }
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            if widgets::chip(ui, "OpenPBR…", false)
                .on_hover_text("MaterialX document with an OpenPBR surface and its textures")
                .clicked()
            {
                self.export_openpbr();
            }
            if widgets::chip(ui, "glTF…", false)
                .on_hover_text("The mesh as glTF 2.0 with every texture set as a material")
                .clicked()
            {
                self.export_gltf();
            }
            if widgets::chip(ui, "Unreal…", false)
                .on_hover_text("Select the Unreal Engine preset and export")
                .clicked()
            {
                self.export_engine_preset("Unreal Engine");
            }
            if widgets::chip(ui, "Unity…", false)
                .on_hover_text("Select the Unity (URP Lit) preset and export")
                .clicked()
            {
                self.export_engine_preset("Unity (URP Lit)");
            }
            if widgets::chip(ui, "Godot…", false)
                .on_hover_text("Select the Godot 4 preset and export")
                .clicked()
            {
                self.export_engine_preset("Godot 4 (StandardMaterial3D)");
            }
            if widgets::chip(ui, "All sets…", false)
                .on_hover_text("Export every texture set with its own preset and format")
                .clicked()
            {
                self.export_all_sets();
            }
            if widgets::chip(ui, "UDIM tiles…", false)
                .on_hover_text("This preset once per UV tile ( in names, else .1001)")
                .clicked()
            {
                self.export_udim_tiles();
            }
        });
    }

    fn preset_editor(&mut self, ui: &mut Ui) {
        let Some(mut draft) = self.preset_draft.take() else {
            return;
        };
        let (mut save, mut cancel) = (false, false);
        egui::Frame::new()
            .fill(theme::INPUT_DEEP)
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(8))
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut draft.preset.name)
                        .desired_width(ui.available_width())
                        .margin(vec2(8.0, 4.0))
                        .hint_text("Preset name"),
                );
                widgets::property_row(ui, "Max size", |ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if widgets::chip(ui, "Any", draft.preset.max_resolution.is_none())
                        .on_hover_text("Accept texture sets of any size")
                        .clicked()
                    {
                        draft.preset.max_resolution = None;
                    }
                    for edge in [1024, 2048, 4096, 8192] {
                        let label = format!("{}k", edge / 1024);
                        if widgets::chip(ui, &label, draft.preset.max_resolution == Some(edge))
                            .on_hover_text("Block exports of texture sets larger than this")
                            .clicked()
                        {
                            draft.preset.max_resolution = Some(edge);
                        }
                    }
                });
                let mut remove = None;
                let removable = draft.preset.outputs.len() > 1;
                for (index, output) in draft.preset.outputs.iter_mut().enumerate() {
                    ui.push_id(("preset_output", index), |ui| {
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut output.file_name)
                                    .desired_width(ui.available_width() - 34.0)
                                    .margin(vec2(8.0, 3.0))
                                    .hint_text("$textureSet_Name"),
                            )
                            .on_hover_text("File name; $textureSet becomes the texture set name");
                            if widgets::icon_button(ui, Icon::Close, removable, "Remove this file")
                                .clicked()
                            {
                                remove = Some(index);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            for layout in OutputLayout::ALL {
                                if widgets::chip(ui, layout.name(), output.layout == layout)
                                    .clicked()
                                {
                                    output.layout = layout;
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            widgets::toggle(ui, &mut output.skip_if_black)
                                .on_hover_text("Leave the file out when it is entirely black");
                            ui.label(
                                RichText::new("Skip when black")
                                    .size(11.5)
                                    .color(theme::TEXT_WEAK),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            // "Export" follows the format chosen when exporting.
                            if widgets::chip(ui, "Export", output.format.is_none())
                                .on_hover_text("Use the format chosen for the export")
                                .clicked()
                            {
                                output.format = None;
                            }
                            for format in TextureFormat::ALL {
                                let selected = output.format.as_deref() == Some(format.extension());
                                if widgets::chip(ui, format.name(), selected)
                                    .on_hover_text("Always write this file in this format")
                                    .clicked()
                                {
                                    output.format = Some(format.extension().to_owned());
                                }
                            }
                        });
                        // Depth matters for PNG; TGA holds 8 bits and EXR
                        // half floats.
                        let fixed = matches!(output.format.as_deref(), Some("tga" | "exr"));
                        ui.add_enabled_ui(!fixed, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 4.0;
                                for depth in BitDepth::ALL {
                                    if widgets::chip(ui, depth.name(), output.bit_depth == depth)
                                        .on_hover_text("Bits per channel when written as PNG")
                                        .clicked()
                                    {
                                        output.bit_depth = depth;
                                    }
                                }
                            });
                        });
                        let count = output.layout.channel_count();
                        for (slot, channel) in output.slots[..count]
                            .iter_mut()
                            .zip(slot_names(output.layout))
                        {
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    vec2(14.0, 22.0),
                                    egui::Label::new(
                                        RichText::new(*channel).size(11.5).color(theme::TEXT_WEAK),
                                    ),
                                );
                                slot_editor(ui, channel, slot);
                            });
                        }
                    });
                }
                if let Some(index) = remove {
                    draft.preset.outputs.remove(index);
                }
                ui.add_space(6.0);
                if draft.preset.outputs.len() < MAX_EXPORT_OUTPUTS
                    && widgets::pill_button(ui, Some(Icon::Plus), "Add file", false, true).clicked()
                {
                    let number = draft.preset.outputs.len() + 1;
                    draft.preset.outputs.push(ExportOutput::new(
                        &format!("$textureSet_Map{number}"),
                        OutputLayout::Gray,
                        [ChannelSlot::channel(MaterialChannel::Roughness, SourceComponent::Red); 4],
                    ));
                }
                let error = draft.preset.validate().err();
                if let Some(error) = &error {
                    ui.label(
                        RichText::new(error.to_string())
                            .size(11.5)
                            .color(theme::DANGER),
                    );
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(error.is_none(), |ui| {
                        save = widgets::pill_button(ui, None, "Save preset", true, false).clicked();
                    });
                    cancel = widgets::pill_button(ui, None, "Cancel", false, false).clicked();
                });
            });
        if cancel {
            return;
        }
        self.preset_draft = Some(draft);
        if save && let Err(error) = self.apply_preset_draft() {
            self.bake_error = Some(error);
        }
    }
}

/// Channel letters of an output layout, in slot order.
fn slot_names(layout: OutputLayout) -> &'static [&'static str] {
    match layout {
        OutputLayout::Gray => &["V"],
        OutputLayout::Rgb => &["R", "G", "B"],
        OutputLayout::Rgba => &["R", "G", "B", "A"],
    }
}

fn source_label(source: ExportSource) -> String {
    match source {
        ExportSource::Channel(MaterialChannel::AmbientOcclusion) => "Occlusion".to_owned(),
        ExportSource::Channel(channel) => channel_label(channel).to_owned(),
        ExportSource::MeshMap(kind) => format!("{} map", kind.name()),
        ExportSource::SpecGloss(map) => format!("{} (spec/gloss)", map.name()),
    }
}

/// Short description of a slot, e.g. `1 − Roughness` or `Normal.G`.
fn slot_label(slot: ChannelSlot) -> String {
    match slot {
        ChannelSlot::Constant(value) => format!("{value:.2}"),
        ChannelSlot::Source {
            source,
            component,
            invert,
        } => {
            let mut label = source_label(source);
            if source.is_color() {
                label.push_str(match component {
                    SourceComponent::Red => ".R",
                    SourceComponent::Green => ".G",
                    SourceComponent::Blue => ".B",
                });
            }
            if invert {
                format!("1 − {label}")
            } else {
                label
            }
        }
    }
}

/// Source picker plus component, invert, or constant controls for one slot.
fn slot_editor(ui: &mut Ui, id_salt: &str, slot: &mut ChannelSlot) {
    let current = slot.source();
    let mut chosen = current;
    let combo = egui::ComboBox::from_id_salt(("slot_source", id_salt))
        .width(104.0)
        .selected_text(current.map_or_else(|| "Constant".to_owned(), source_label))
        .show_ui(ui, |ui| {
            if ui.selectable_label(current.is_none(), "Constant").clicked() {
                chosen = None;
            }
            for source in ExportSource::ALL {
                if ui
                    .selectable_label(current == Some(source), source_label(source))
                    .clicked()
                {
                    chosen = Some(source);
                }
            }
        });
    widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
    if chosen != current {
        let invert = matches!(*slot, ChannelSlot::Source { invert: true, .. });
        *slot = chosen.map_or(ChannelSlot::ONE, |source| ChannelSlot::Source {
            source,
            component: SourceComponent::Red,
            invert,
        });
    }
    match slot {
        ChannelSlot::Constant(value) => {
            ui.add(egui::DragValue::new(value).speed(0.01).range(0.0..=1.0));
        }
        ChannelSlot::Source {
            source,
            component,
            invert,
        } => {
            ui.spacing_mut().item_spacing.x = 4.0;
            if source.is_color() {
                let letter = |component: SourceComponent| match component {
                    SourceComponent::Red => "R",
                    SourceComponent::Green => "G",
                    SourceComponent::Blue => "B",
                };
                let combo = egui::ComboBox::from_id_salt(("slot_component", id_salt))
                    .width(38.0)
                    .selected_text(letter(*component))
                    .show_ui(ui, |ui| {
                        for option in SourceComponent::ALL {
                            ui.selectable_value(component, option, letter(option));
                        }
                    })
                    .response
                    .on_hover_text("Component of the source to read");
                widgets::label_from_row(ui, &combo, egui::WidgetType::ComboBox);
            }
            if widgets::chip(ui, "1−x", *invert)
                .on_hover_text("Invert: store 1 − value")
                .clicked()
            {
                *invert = !*invert;
            }
        }
    }
}

impl SubstanceCraftApp {
    /// Symmetry, stabilization, path placement, and the eyedropper.
    fn stroke_assist(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Assist");
        widgets::property_row(ui, "Projection", |ui| {
            widgets::toggle(ui, &mut self.projection_painting)
        })
        .on_hover_text(
            "Draw paint, erase, and mask strokes on screen and project them onto the \
             visible surface: they cross UV seams and keep their on-screen size",
        );
        if self.projection_painting {
            hint(
                ui,
                "Brush size is in screen pixels. Hidden and back-facing surfaces stay \
                 unpainted. ID strokes, symmetry, and paths stay in UV space.",
            );
        }
        widgets::property_row(ui, "Symmetry", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if widgets::chip(ui, "Off", self.symmetry.is_none()).clicked() {
                self.symmetry = None;
            }
            for axis in Axis::ALL {
                let selected = self.symmetry.is_some_and(|symmetry| symmetry.axis == axis);
                if widgets::chip(ui, axis.name(), selected)
                    .on_hover_text(format!(
                        "Mirror strokes across the {} = 0 plane, or repeat them around the {} axis",
                        axis.name(),
                        axis.name()
                    ))
                    .clicked()
                {
                    let radial = self.symmetry.map_or(0, |symmetry| symmetry.radial);
                    self.symmetry = Some(substancecraft_core::Symmetry::radial(axis, radial));
                }
            }
        });
        if let Some(symmetry) = &mut self.symmetry {
            widgets::property_row(ui, "Copies", |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                if widgets::chip(ui, "Mirror", !symmetry.is_radial())
                    .on_hover_text("Mirror strokes across a plane")
                    .clicked()
                {
                    symmetry.radial = 0;
                }
                for count in [3, 4, 6, 8, 12] {
                    if widgets::chip(ui, &count.to_string(), symmetry.radial == count)
                        .on_hover_text(format!(
                            "Repeat strokes {count} times around the axis through the origin"
                        ))
                        .clicked()
                    {
                        symmetry.radial = count;
                    }
                }
            });
            if !symmetry.is_radial() {
                slider_row(
                    ui,
                    "Mirror offset",
                    &mut symmetry.offset,
                    -2.0..=2.0,
                    false,
                    3,
                );
            }
        }
        slider_row(
            ui,
            "Lazy radius",
            &mut self.stroke_stabilizer.radius,
            0.0..=80.0,
            false,
            0,
        );
        if matches!(self.paint_mode, PaintMode::Paint | PaintMode::Erase) {
            match self.path_controls.as_ref().map(Vec::len) {
                None => {
                    if widgets::pill_button(ui, None, "Draw path", false, true)
                        .on_hover_text(
                            "Click points on the mesh; the stroke follows a smooth curve \
                             through them",
                        )
                        .clicked()
                    {
                        self.path_controls = Some(Vec::new());
                    }
                }
                Some(count) => {
                    hint(
                        ui,
                        &format!(
                            "{count} path points. Click the mesh to add, Enter to finish, Esc to cancel."
                        ),
                    );
                    ui.horizontal(|ui| {
                        if widgets::pill_button(ui, None, "Finish", true, false).clicked() {
                            self.commit_path(false);
                        }
                        if widgets::pill_button(ui, None, "Close loop", false, false).clicked() {
                            self.commit_path(true);
                        }
                        if widgets::pill_button(ui, None, "Cancel", false, false).clicked() {
                            self.path_controls = None;
                        }
                    });
                }
            }
        }
        let picking = self.picking_material;
        if widgets::pill_button(
            ui,
            None,
            if picking {
                "Click the mesh to pick…"
            } else {
                "Pick material"
            },
            picking,
            true,
        )
        .on_hover_text("Copy colour, roughness, metallic, opacity, and emission from the surface")
        .clicked()
        {
            self.picking_material = !picking;
        }
        hint(
            ui,
            "Shift-click draws a straight line from the last dab. Strokes restart at UV seams.",
        );
    }
}

/// Anchor-point mask: pick an earlier layer and the channel to read.
fn anchor_controls(
    ui: &mut Ui,
    layer_id: LayerId,
    anchor: &mut Option<substancecraft_core::AnchorMask>,
    candidates: &[(LayerId, String)],
) {
    widgets::caption(ui, "Anchor mask");
    let current = anchor.map(|anchor| anchor.layer);
    let label = current
        .and_then(|id| candidates.iter().find(|(candidate, _)| *candidate == id))
        .map_or("None", |(_, name)| name.as_str());
    let mut chosen = current;
    widgets::property_row(ui, "Read from", |ui| {
        let combo = egui::ComboBox::from_id_salt(("anchor_layer", layer_id.0))
            .width(ui.available_width() - 8.0)
            .selected_text(label)
            .show_ui(ui, |ui| {
                if ui.selectable_label(current.is_none(), "None").clicked() {
                    chosen = None;
                }
                for (id, name) in candidates {
                    if ui.selectable_label(current == Some(*id), name).clicked() {
                        chosen = Some(*id);
                    }
                }
            });
        widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
    });
    if chosen != current {
        *anchor = chosen.map(|id| match anchor {
            Some(existing) => substancecraft_core::AnchorMask {
                layer: id,
                ..*existing
            },
            None => substancecraft_core::AnchorMask::new(id, MaterialChannel::BaseColor),
        });
    }
    let Some(anchor) = anchor else {
        hint(
            ui,
            "Mask this layer by what the stack looks like at an earlier layer, e.g. gloss \
             only where scratches were painted.",
        );
        return;
    };
    widgets::property_row(ui, "Channel", |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            for channel in substancecraft_core::AnchorMask::CHANNELS {
                if widgets::chip(ui, channel_label(channel), anchor.channel == channel).clicked() {
                    anchor.channel = channel;
                }
            }
        });
    });
    slider_row(ui, "Low", &mut anchor.low, 0.0..=0.99, false, 2);
    slider_row(ui, "High", &mut anchor.high, 0.01..=1.0, false, 2);
    if anchor.high <= anchor.low {
        anchor.high = (anchor.low + 0.01).min(1.0);
    }
    widgets::property_row(ui, "Invert", |ui| widgets::toggle(ui, &mut anchor.invert));
}

impl SubstanceCraftApp {
    /// Flatten, materials, and smart materials for the selected layer.
    fn layer_actions(&mut self, ui: &mut Ui) {
        let Some(layer) = self.selected_layer().cloned() else {
            return;
        };
        widgets::caption(ui, "Layer actions");
        let merging = self.flatten_selection().len();
        let (label, tooltip) = if merging > 1 {
            (
                format!("Flatten {merging} layers to bitmaps"),
                "Bake the Ctrl-clicked layers (each with its mask, opacity, and filters) \
                 together into one fill layer above the topmost; the originals stay, hidden",
            )
        } else {
            (
                "Flatten to bitmaps".to_owned(),
                "Bake this layer (with its mask, opacity, and filters) into one fill layer; \
                 the original stays below, hidden. Ctrl-click other layers to merge them too",
            )
        };
        if widgets::pill_button(ui, None, &label, false, true)
            .on_hover_text(tooltip)
            .clicked()
        {
            self.flatten_selected_layer();
        }

        if !layer.is_group() {
            let materials = self.material_library();
            let mut applied = None;
            widgets::property_row(ui, "Material", |ui| {
                let combo = egui::ComboBox::from_id_salt("apply_material")
                    .width(ui.available_width() - 8.0)
                    .selected_text("Apply…")
                    .show_ui(ui, |ui| {
                        for material in &materials {
                            if ui.selectable_label(false, &material.name).clicked() {
                                applied = Some(material.clone());
                            }
                        }
                    });
                widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
            });
            if let Some(material) = applied {
                self.apply_material(&material);
            }
            let mut linked = None;
            let current = layer.material_link.as_ref().map(|link| link.material);
            widgets::property_row(ui, "Instance", |ui| {
                let combo = egui::ComboBox::from_id_salt("link_material")
                    .width(ui.available_width() - 8.0)
                    .selected_text("Link to…")
                    .show_ui(ui, |ui| {
                        for material in &materials {
                            let in_project = self.project.material(material.id).is_some();
                            let label = if in_project {
                                material.name.clone()
                            } else {
                                format!("{} (new master)", material.name)
                            };
                            let selected = in_project && current == Some(material.id);
                            if ui.selectable_label(selected, label).clicked() {
                                linked = Some(material.clone());
                            }
                        }
                    })
                    .response
                    .on_hover_text(
                        "Follow a shared master material: editing the master updates every \
                         linked layer, in every texture set",
                    );
                widgets::label_from_row(ui, &combo, egui::WidgetType::ComboBox);
            });
            if let Some(material) = linked {
                self.link_selected_layer(&material);
            }
            if layer.fill.is_some() {
                ui.horizontal(|ui| {
                    if widgets::pill_button(ui, None, "Save as material", false, false)
                        .on_hover_text(
                            "Keep a copy of this fill's values in the project's materials",
                        )
                        .clicked()
                    {
                        self.save_layer_as_material();
                    }
                    if current.is_none()
                        && widgets::pill_button(ui, None, "Make master", false, false)
                            .on_hover_text(
                                "Save this fill as a project material and link this layer to it, \
                                 so other layers can share it",
                            )
                            .clicked()
                    {
                        self.make_selected_layer_master();
                    }
                });
            }
        }

        let smart = self.smart_material_library();
        let mut insert = None;
        widgets::caption(ui, "Smart materials");
        // Thumbnails render one per frame; the rest show a placeholder
        // until their turn.
        let mut budget = true;
        let mut pending = false;
        let context = ui.ctx().clone();
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
            let edge = 72.0;
            for material in &smart {
                let texture = self.smart_thumbnail_texture(&context, material, &mut budget);
                pending |= texture.is_none();
                ui.vertical(|ui| {
                    ui.set_width(edge + 8.0);
                    let button = match &texture {
                        Some(texture) => egui::Button::image(
                            egui::Image::new(egui::load::SizedTexture::from_handle(texture))
                                .fit_to_exact_size(vec2(edge, edge)),
                        ),
                        None => egui::Button::new("…").min_size(vec2(edge, edge)),
                    };
                    let hover = if material.description.is_empty() {
                        format!("Insert {}", material.name)
                    } else {
                        format!("Insert {}: {}", material.name, material.description)
                    };
                    let response = ui.add(button).on_hover_text(hover);
                    response.widget_info(|| {
                        egui::WidgetInfo::labeled(
                            egui::WidgetType::Button,
                            true,
                            format!("Insert smart material {}", material.name),
                        )
                    });
                    if response.clicked() {
                        insert = Some(material.clone());
                    }
                    ui.add(
                        egui::Label::new(
                            RichText::new(&material.name)
                                .size(11.0)
                                .color(theme::TEXT_WEAK),
                        )
                        .truncate(),
                    );
                });
            }
        });
        if pending {
            context.request_repaint();
        }
        if let Some(material) = insert {
            self.insert_smart_material(&material);
        }
        ui.horizontal(|ui| {
            if widgets::pill_button(ui, None, "Save smart", false, false)
                .on_hover_text("Save this layer tree as a smart material in the project")
                .clicked()
            {
                self.save_layer_as_smart_material();
            }
            if widgets::pill_button(ui, None, "Export…", false, false)
                .on_hover_text("Write this layer tree to a .scsmart file")
                .clicked()
            {
                let material =
                    substancecraft_core::SmartMaterial::from_layer(&layer, layer.name.clone());
                self.export_smart_material_file(&material);
            }
            if widgets::pill_button(ui, None, "Import…", false, false)
                .on_hover_text("Insert a smart material from a .scsmart file")
                .clicked()
            {
                self.import_smart_material_file();
            }
        });
    }
}

impl SubstanceCraftApp {
    /// New projects from templates (built-in or a file), and saving the
    /// selected texture set as one. Starting from a template asks first.
    fn template_controls(&mut self, ui: &mut Ui) {
        if let Some(template) = self.pending_template.clone() {
            ui.label(
                RichText::new(format!(
                    "Start a new project from \"{}\"? The current project closes; save it first \
                     to keep its changes. The mesh stays.",
                    template.name
                ))
                .size(12.0)
                .color(WARNING),
            );
            ui.horizontal(|ui| {
                if widgets::pill_button(ui, None, "Start new project", true, false).clicked() {
                    self.pending_template = None;
                    self.start_from_template(&template);
                }
                if widgets::pill_button(ui, None, "Cancel", false, false).clicked() {
                    self.pending_template = None;
                }
            });
            ui.add_space(4.0);
            return;
        }
        let mut chosen = None;
        let mut from_file = false;
        widgets::property_row(ui, "New project", |ui| {
            let combo = egui::ComboBox::from_id_salt("project_template")
                .width(ui.available_width() - 8.0)
                .selected_text("From template…")
                .show_ui(ui, |ui| {
                    for template in substancecraft_core::ProjectTemplate::built_in() {
                        if ui
                            .selectable_label(false, &template.name)
                            .on_hover_text(&template.description)
                            .clicked()
                        {
                            chosen = Some(template);
                        }
                    }
                    if ui.selectable_label(false, "From file…").clicked() {
                        from_file = true;
                    }
                });
            widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
        });
        if from_file {
            chosen = self.open_template_file();
        }
        if chosen.is_some() {
            self.pending_template = chosen;
        }
        if self.selected_texture_set.is_some()
            && widgets::pill_button(ui, None, "Save set as template…", false, true)
                .on_hover_text(
                    "Write this texture set's resolution, channels, export choice, and layers \
                     to a .sctemplate file",
                )
                .clicked()
        {
            self.save_template_file();
        }
        ui.add_space(4.0);
    }

    /// Project materials and the editor of the open one. Each is a master:
    /// editing it updates every fill layer linked to it.
    fn project_materials(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Materials");
        if self.project.materials.is_empty() {
            hint(
                ui,
                "Save a fill as a material, or make it a master, from a layer's actions. \
                 Layers linked to a master follow its edits.",
            );
            return;
        }
        let rows: Vec<(MaterialId, String, usize)> = self
            .project
            .materials
            .iter()
            .map(|material| {
                let shared = self.project.material_instances(material.id).len();
                (material.id, material.name.clone(), shared)
            })
            .collect();
        let mut open = self.editing_material;
        for (id, name, shared) in rows {
            ui.horizontal(|ui| {
                if widgets::chip(ui, &name, open == Some(id))
                    .on_hover_text("Edit this master")
                    .clicked()
                {
                    open = (open != Some(id)).then_some(id);
                }
                let usage = match shared {
                    0 => "no instances".to_owned(),
                    1 => "1 instance".to_owned(),
                    count => format!("{count} instances"),
                };
                ui.label(RichText::new(usage).size(11.0).color(theme::TEXT_FAINT));
            });
        }
        self.editing_material = open;
        let Some(original) = open.and_then(|id| self.project.material(id)).cloned() else {
            return;
        };
        let resources = self
            .project
            .resources
            .iter()
            .map(|resource| (resource.id, resource.name.clone()))
            .collect::<Vec<_>>();
        let mut edited = original.clone();
        widgets::property_row(ui, "Name", |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut edited.name)
                    .desired_width(ui.available_width())
                    .margin(vec2(8.0, 3.0)),
            );
        });
        widgets::property_row(ui, "Channels", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                for (channel, label) in [
                    (MaterialChannel::BaseColor, "Color"),
                    (MaterialChannel::Roughness, "Rough"),
                    (MaterialChannel::Metallic, "Metal"),
                    (MaterialChannel::Height, "Hgt"),
                    (MaterialChannel::Normal, "Nrm"),
                    (MaterialChannel::Emissive, "Emit"),
                    (MaterialChannel::Opacity, "Opac"),
                ] {
                    let enabled = edited.channels.contains(&channel);
                    if widgets::chip(ui, label, enabled).clicked() {
                        if enabled {
                            edited.channels.remove(&channel);
                        } else {
                            edited.channels.insert(channel);
                        }
                    }
                }
            });
        });
        fill_controls(
            ui,
            Id::new(("master_fill", original.id.0)),
            &mut edited.fill,
            &resources,
            None,
        );
        ui.add_space(4.0);
        if widgets::pill_button(ui, Some(Icon::Trash), "Delete material", false, true)
            .on_hover_text("Linked layers keep their current values as plain fills")
            .clicked()
        {
            self.delete_project_material(original.id);
        } else if edited != original {
            self.edit_project_material(edited);
        }
    }

    /// The project's smart masks, each removable. They are applied from a
    /// layer's properties.
    fn project_smart_masks(&mut self, ui: &mut Ui) {
        if self.project.smart_masks.is_empty() {
            return;
        }
        widgets::caption(ui, "Smart masks");
        let mut remove = None;
        for (index, mask) in self.project.smart_masks.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&mask.name).color(theme::TEXT));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if widgets::chip(ui, "Delete", false).clicked() {
                        remove = Some(index);
                    }
                });
            });
        }
        if let Some(index) = remove {
            self.delete_smart_mask(index);
        }
    }

    /// Search over every asset (project and built-in) by name, kind, tag,
    /// author, or licence.
    fn asset_search(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);
        ui.add(
            egui::TextEdit::singleline(&mut self.asset_query)
                .desired_width(ui.available_width())
                .margin(vec2(10.0, 5.0))
                .hint_text("Search assets: name, tag, licence…"),
        );
        if self.asset_query.trim().is_empty() {
            return;
        }
        let results = self.project.search_assets(&self.asset_query, None);
        if results.is_empty() {
            hint(ui, "No assets match.");
        }
        for asset in results.iter().take(24) {
            let origin = if asset.built_in {
                "built-in"
            } else {
                "project"
            };
            let license = if asset.provenance.license.is_empty() {
                "no licence recorded".to_owned()
            } else {
                asset.provenance.license.clone()
            };
            ui.label(RichText::new(&asset.name).size(12.5).color(theme::TEXT));
            ui.label(
                RichText::new(format!("{} · {origin} · {license}", asset.kind.name()))
                    .size(11.0)
                    .color(theme::TEXT_FAINT),
            );
        }
        if results.len() > 24 {
            hint(ui, &format!("…and {} more", results.len() - 24));
        }
    }

    /// Colour space, provenance, and (for text resources) content of each
    /// bitmap.
    fn resource_details(&mut self, ui: &mut Ui) {
        let mut reload = Vec::new();
        let mut rewrite_text = None;
        for resource in &mut self.project.resources {
            egui::CollapsingHeader::new(RichText::new(&resource.name).size(12.0))
                .id_salt(("resource_details", resource.id.0))
                .show(ui, |ui| {
                    widgets::property_row(ui, "Colour", |ui| {
                        let combo = egui::ComboBox::from_id_salt(("color_space", resource.id.0))
                            .width(ui.available_width() - 8.0)
                            .selected_text(resource.color_space.name())
                            .show_ui(ui, |ui| {
                                for space in substancecraft_core::ColorSpace::ALL {
                                    if ui
                                        .selectable_label(
                                            resource.color_space == space,
                                            space.name(),
                                        )
                                        .clicked()
                                        && resource.color_space != space
                                    {
                                        resource.color_space = space;
                                        reload.push(resource.id);
                                    }
                                }
                            });
                        widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
                    });
                    for (label, value) in [
                        ("Licence", &mut resource.provenance.license),
                        ("Author", &mut resource.provenance.author),
                        ("Origin", &mut resource.provenance.origin),
                    ] {
                        widgets::property_row(ui, label, |ui| {
                            ui.add(
                                egui::TextEdit::singleline(value)
                                    .desired_width(ui.available_width())
                                    .margin(vec2(6.0, 2.0)),
                            );
                        });
                    }
                    let mut tags = resource.tags.join(", ");
                    widgets::property_row(ui, "Tags", |ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut tags)
                                .desired_width(ui.available_width())
                                .margin(vec2(6.0, 2.0))
                                .hint_text("comma separated"),
                        );
                    });
                    resource.tags = tags
                        .split(',')
                        .map(|tag| tag.trim().to_owned())
                        .filter(|tag| !tag.is_empty())
                        .collect();
                    if resource.source.to_ascii_lowercase().ends_with(".sctext") {
                        let key = Id::new(("text_edit", resource.id.0));
                        let mut text = ui.ctx().data_mut(|data| {
                            data.get_temp::<String>(key).unwrap_or_else(|| {
                                substancecraft_api::read_text_resource(std::path::Path::new(
                                    &resource.source,
                                ))
                                .ok()
                                .map(|spec| spec.text)
                                .unwrap_or_default()
                            })
                        });
                        ui.add(
                            egui::TextEdit::multiline(&mut text)
                                .desired_width(ui.available_width())
                                .desired_rows(2),
                        );
                        if widgets::pill_button(ui, None, "Update text", false, true).clicked() {
                            rewrite_text =
                                Some((resource.id, resource.source.clone(), text.clone()));
                        }
                        ui.ctx().data_mut(|data| data.insert_temp(key, text));
                    }
                });
        }
        if let Some((id, source, text)) = rewrite_text {
            let path = std::path::Path::new(&source);
            let mut spec = substancecraft_api::read_text_resource(path).ok().unwrap_or(
                substancecraft_api::TextSpec {
                    text: String::new(),
                    size_px: 128.0,
                    font: None,
                },
            );
            spec.text = text;
            match substancecraft_api::save_text_resource(path, &spec) {
                Ok(()) => reload.push(id),
                Err(error) => self.import_error = Some(error.to_string()),
            }
        }
        if !reload.is_empty() {
            self.reload_changed_resources(&reload);
        }
    }
}

#[cfg(test)]
mod layer_search_tests {
    use super::*;
    use substancecraft_core::{FillTextures, LayerFill, LayerId};

    #[test]
    fn layer_search_matches_name_type_channel_and_resource_and_keeps_ancestors() {
        let mut fill = MaterialLayer::new_fill(LayerId(3), "Trim Paint", LayerFill::default());
        fill.fill = Some(LayerFill {
            textures: FillTextures {
                base_color: Some(ResourceId(42)),
                ..FillTextures::default()
            },
            ..LayerFill::default()
        });
        let mut group = MaterialLayer::new_group(LayerId(1), "Hardware");
        group.children = Some(vec![
            MaterialLayer::new_paint(LayerId(2), "Brushed metal"),
            fill,
        ]);
        let resources = [(ResourceId(42), "brass_albedo.png".to_owned())];

        for (query, expected) in [
            ("brushed", vec![LayerId(1), LayerId(2)]),
            ("group", vec![LayerId(1)]),
            ("metallic", vec![LayerId(1), LayerId(3), LayerId(2)]),
            ("brass_albedo", vec![LayerId(1), LayerId(3)]),
            ("fill brass", vec![LayerId(1), LayerId(3)]),
            ("missing", Vec::new()),
        ] {
            let rows = layer_rows(&[group.clone()], 0, query, &resources);
            assert_eq!(
                rows.iter().map(|row| row.id).collect::<Vec<_>>(),
                expected,
                "query {query}"
            );
        }
    }
}
