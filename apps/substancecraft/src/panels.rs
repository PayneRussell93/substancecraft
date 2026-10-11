//! Floating pill overlays drawn on top of the full-bleed viewport.
//!
//! Layout: a brand pill and tool pill along the top, a vertical dock on the
//! left that slides out drawers, an inspector column of collapsible cards on
//! the right, and a status pill along the bottom.

use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontId, Id, Layout, Margin, Order, Rect, Response,
    RichText, Sense, Ui, pos2, vec2,
};
use substancecraft_core::{
    Axis, BoneId, BoneTransform, Direction, LayerFill, MAX_SUBDIVISION_LEVELS, MaskGenerator,
    MaterialChannel, MeshAsset, MeshId, MeshMapKind, MeshSmoothing, PrimitivePreset, SelectionMode,
    TextureSetId, ViewportCamera,
};

use super::{
    BAKEABLE_CHANNELS, CHANNELS, MAX_MESH_SCALE, MAX_SAVED_POSES, MIN_MESH_SCALE, PaintMode,
    SubstanceCraftApp, channel_label,
};
use crate::icons::{self, Icon};
use crate::theme;
use crate::widgets;

const DRAWER_WIDTH: f32 = 284.0;
const INSPECTOR_WIDTH: f32 = 296.0;
/// Vertical space kept free above the bottom edge for the status pill.
const STATUS_RESERVE: f32 = 56.0;
const MODES: [(PaintMode, &str); 5] = [
    (PaintMode::View, "View"),
    (PaintMode::Paint, "Paint"),
    (PaintMode::Mask, "Mask"),
    (PaintMode::Id, "ID"),
    (PaintMode::Select, "Select"),
];
const MESH_MAP_RESOLUTIONS: [u32; 4] = [256, 512, 1024, 2048];
const WARNING: Color32 = Color32::from_rgb(240, 180, 80);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drawer {
    Scene,
    Brush,
    Bake,
    Poses,
    Environment,
}

impl Drawer {
    const ALL: [Self; 5] = [
        Self::Scene,
        Self::Brush,
        Self::Bake,
        Self::Poses,
        Self::Environment,
    ];

    fn icon(self) -> Icon {
        match self {
            Self::Scene => Icon::Cube,
            Self::Brush => Icon::Brush,
            Self::Bake => Icon::Bake,
            Self::Poses => Icon::Figure,
            Self::Environment => Icon::Sun,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Scene => "Scene",
            Self::Brush => "Tools",
            Self::Bake => "Bake",
            Self::Poses => "Poses",
            Self::Environment => "Light",
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Scene => "Scene",
            Self::Brush => "Tools",
            Self::Bake => "Mesh Maps",
            Self::Poses => "Poses",
            Self::Environment => "Camera & Light",
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
            PaintMode::Paint | PaintMode::Mask if self.active_channel == MaterialChannel::Id => {
                Some(MaterialChannel::BaseColor)
            }
            _ => None,
        };
        if let Some(channel) = channel {
            self.set_preview_channel(channel);
        }
        if mode != PaintMode::View && self.drawer.is_none() {
            self.drawer = Some(Drawer::Brush);
        }
    }

    fn set_preview_channel(&mut self, channel: MaterialChannel) {
        self.active_channel = channel;
        self.mesh_map_preview = None;
        self.invalidate_preview();
    }

    fn select_primitive(&mut self, preset: PrimitivePreset) {
        self.selected_primitive = Some(preset);
        self.selected_imported_mesh = None;
        self.project.mesh = Some(MeshAsset {
            id: MeshId(1),
            source: format!("builtin:{}", preset.name()),
            scale: 1.0,
            smoothing: self.mesh_smoothing,
        });
        self.set_base_geometry(Some(preset.build_mesh()));
    }

    fn select_texture_set(&mut self, set_id: TextureSetId) {
        self.finish_active_stroke();
        self.selected_texture_set = Some(set_id);
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
                let selected_text = match self.mesh_map_preview {
                    Some(kind) => kind.name(),
                    None => channel_label(self.active_channel),
                };
                let mut chosen = None;
                egui::ComboBox::from_id_salt("toolbar_channel")
                    .width(118.0)
                    .selected_text(selected_text)
                    .show_ui(ui, |ui| {
                        for channel in CHANNELS {
                            let selected =
                                self.mesh_map_preview.is_none() && self.active_channel == channel;
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
                    .on_hover_text("Preview channel");
                if let Some(channel) = chosen {
                    self.set_preview_channel(channel);
                }
                ui.add_space(2.0);
                widgets::divider(ui);
                if widgets::icon_button(ui, Icon::Focus, true, "Reset view").clicked() {
                    self.viewport_camera = ViewportCamera::default();
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
                                Drawer::Bake => self.bake_drawer(ui),
                                Drawer::Poses => self.poses_drawer(ui),
                                Drawer::Environment => self.environment_drawer(ui),
                            });
                    });
            });
    }

    fn scene_drawer(&mut self, ui: &mut Ui) {
        if widgets::pill_button(ui, Some(Icon::Import), "Import FBX…", true, true).clicked() {
            self.import_fbx();
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
                    "Pick Paint, Mask, or ID to paint on the mesh, or Select to build a \
                     polygon mask from faces, elements, or UV islands.",
                );
                return;
            }
            PaintMode::Select => {
                self.selection_tools(ui);
                return;
            }
            _ => {}
        }

        widgets::caption(ui, "Brush");
        let tip_color = if self.paint_mode == PaintMode::Mask {
            Color32::from_gray((self.mask_value * 255.0).round() as u8)
        } else {
            self.brush_color
        };
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 76.0), Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect, CornerRadius::same(16), theme::INPUT_DEEP);
        let radius = (self.brush_radius * 0.3).clamp(2.0, 32.0);
        painter.circle(
            rect.center(),
            radius,
            tip_color,
            egui::Stroke::new(1.0, Color32::from_white_alpha(60)),
        );

        slider_row(ui, "Size", &mut self.brush_radius, 1.0..=128.0, false, 0);
        match self.paint_mode {
            PaintMode::Mask => {
                slider_row(ui, "Mask value", &mut self.mask_value, 0.0..=1.0, false, 2);
            }
            PaintMode::Paint => {
                widgets::property_row(ui, "Color", |ui| {
                    ui.color_edit_button_srgba(&mut self.brush_color);
                });
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
                hint(
                    ui,
                    "One stroke paints every channel the layer enables: color, roughness, \
                     and metallic.",
                );
            }
            _ => {
                widgets::property_row(ui, "Color", |ui| {
                    ui.color_edit_button_srgba(&mut self.brush_color);
                });
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
        if let Some(maps) = self.mesh_maps.as_ref().filter(|_| fresh) {
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
    }

    fn poses_drawer(&mut self, ui: &mut Ui) {
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

        widgets::caption(ui, "Environment light");
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
            "Ambient fill",
            &mut self.viewport_lighting.ambient_fill,
            0.0..=1.0,
            false,
            2,
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
                egui::ScrollArea::vertical()
                    .id_salt("inspector_scroll")
                    .max_height(max_height)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
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
                        .on_hover_text("Uniform material covering the layer mask")
                        .clicked()
                    {
                        self.add_fill_layer();
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
                let Some(texture_set_id) = self.selected_texture_set else {
                    empty_state(ui, "Add a layer to create a texture set.");
                    return;
                };
                let layers = self
                    .project
                    .texture_sets
                    .iter()
                    .find(|set| set.id == texture_set_id)
                    .map(|set| {
                        set.layers
                            .iter()
                            .rev()
                            .map(|layer| {
                                let kind = match (&layer.fill, &layer.mask_generator) {
                                    (_, Some(generator)) => generator.name(),
                                    (Some(_), None) => "Fill",
                                    (None, None) => "Paint",
                                };
                                (
                                    layer.id,
                                    layer.name.clone(),
                                    layer.visible,
                                    layer.opacity,
                                    kind,
                                    layer.polygon_mask.is_some(),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if layers.is_empty() {
                    empty_state(ui, "This texture set has no layers.");
                    return;
                }
                ui.add_space(2.0);
                egui::ScrollArea::vertical()
                    .id_salt("layers_scroll")
                    .max_height(200.0)
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 2.0;
                        for (layer_id, name, visible, opacity, kind, polygon) in layers {
                            let selected = self.selected_layer == Some(layer_id);
                            let mut toggle_visibility = false;
                            let row = widgets::list_row(
                                ui,
                                Id::new(("layer_row", layer_id.0)),
                                selected,
                                |ui| {
                                    let (icon, tooltip) = if visible {
                                        (Icon::Eye, "Hide layer")
                                    } else {
                                        (Icon::EyeOff, "Show layer")
                                    };
                                    toggle_visibility = widgets::sized_icon_toggle(
                                        ui, icon, false, true, tooltip, 24.0,
                                    )
                                    .clicked();
                                    let color = if visible {
                                        theme::TEXT
                                    } else {
                                        theme::TEXT_FAINT
                                    };
                                    ui.label(RichText::new(name).color(color));
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        ui.label(
                                            RichText::new(format!("{:.0}%", opacity * 100.0))
                                                .size(11.5)
                                                .color(theme::TEXT_FAINT),
                                        );
                                        let badge = if polygon {
                                            format!("{kind} · Poly")
                                        } else {
                                            kind.to_owned()
                                        };
                                        ui.label(
                                            RichText::new(badge).size(10.5).color(theme::ACCENT),
                                        );
                                    });
                                },
                            );
                            if toggle_visibility {
                                self.toggle_layer_visibility(texture_set_id, layer_id);
                            } else if row.clicked() {
                                self.finish_active_stroke();
                                self.selected_layer = Some(layer_id);
                            }
                        }
                    });

                if let Some((_, index, count)) = self.selected_layer_index() {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        if widgets::icon_button(ui, Icon::ArrowUp, index + 1 < count, "Move up")
                            .clicked()
                        {
                            self.move_selected_layer(1);
                        }
                        if widgets::icon_button(ui, Icon::ArrowDown, index > 0, "Move down")
                            .clicked()
                        {
                            self.move_selected_layer(-1);
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

    fn toggle_layer_visibility(
        &mut self,
        texture_set_id: TextureSetId,
        layer_id: substancecraft_core::LayerId,
    ) {
        let Some(layer) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)
            .and_then(|set| set.layers.iter_mut().find(|layer| layer.id == layer_id))
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
                let Some(texture_set) = self
                    .project
                    .texture_sets
                    .iter()
                    .find(|set| Some(set.id) == self.selected_texture_set)
                else {
                    empty_state(ui, "Select a texture set to inspect it.");
                    return;
                };
                let name = texture_set.name.clone();
                let resolution = format!(
                    "{} × {}",
                    texture_set.resolution.width, texture_set.resolution.height
                );
                widgets::property_row(ui, "Texture set", |ui| ui.label(name));
                widgets::property_row(ui, "Resolution", |ui| ui.label(resolution));

                if self.selected_layer().is_some() {
                    self.layer_properties(ui);
                }

                widgets::caption(ui, "Bake & export");
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
                    for channel in BAKEABLE_CHANNELS {
                        if widgets::chip(ui, channel_label(channel), false)
                            .on_hover_text(format!("Export {} as TGA", channel_label(channel)))
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
        let mesh_maps_needed = self
            .selected_layer()
            .is_some_and(|layer| layer.mask_generator.is_some());
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
        let Some(layer) = self.selected_layer_mut() else {
            return;
        };

        widgets::caption(ui, "Layer");
        widgets::property_row(ui, "Name", |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut layer.name)
                    .desired_width(ui.available_width())
                    .margin(vec2(8.0, 3.0)),
            );
        });
        slider_row(ui, "Opacity", &mut layer.opacity, 0.0..=1.0, false, 2);
        widgets::property_row(ui, "Channels", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (channel, label) in [
                (MaterialChannel::BaseColor, "Color"),
                (MaterialChannel::Roughness, "Rough"),
                (MaterialChannel::Metallic, "Metal"),
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

        if let Some(fill) = layer.fill.as_mut() {
            widgets::caption(ui, "Fill");
            let mut color = Color32::from_rgba_unmultiplied(
                fill.color[0],
                fill.color[1],
                fill.color[2],
                fill.color[3],
            );
            widgets::property_row(ui, "Color", |ui| {
                if ui.color_edit_button_srgba(&mut color).changed() {
                    fill.color = color.to_srgba_unmultiplied();
                }
            });
            slider_row(ui, "Roughness", &mut fill.roughness, 0.0..=1.0, false, 2);
            slider_row(ui, "Metallic", &mut fill.metallic, 0.0..=1.0, false, 2);
        } else {
            ui.add_space(2.0);
            if widgets::pill_button(ui, Some(Icon::Plus), "Add fill", false, true)
                .on_hover_text("Cover this layer's mask with uniform material values")
                .clicked()
            {
                layer.fill = Some(LayerFill {
                    color: brush,
                    ..LayerFill::default()
                });
            }
        }

        widgets::caption(ui, "Mask generator");
        let current = layer.mask_generator;
        let mut chosen = current;
        egui::ComboBox::from_id_salt(("generator_kind", layer.id.0))
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
        if let Some(generator) = layer.mask_generator.as_mut() {
            generator_controls(ui, generator);
        }

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
                changed |= ui.color_edit_button_srgba(&mut color).changed();
            }
            changed
        });
        if changed {
            let channels = color.to_array();
            layer.id_mask_color = enabled.then_some([channels[0], channels[1], channels[2]]);
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
                            let preview = match self.mesh_map_preview {
                                Some(kind) => kind.name(),
                                None => channel_label(self.active_channel),
                            };
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
        if errors.is_empty() {
            return;
        }

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
fn generator_controls(ui: &mut Ui, generator: &mut MaskGenerator) {
    match generator {
        MaskGenerator::EdgeWear(settings) => {
            slider_row(ui, "Amount", &mut settings.amount, 0.0..=1.0, false, 2);
            slider_row(ui, "Contrast", &mut settings.contrast, 0.0..=1.0, false, 2);
            slider_row(ui, "Breakup", &mut settings.breakup, 0.0..=1.0, false, 2);
            slider_row(
                ui,
                "Breakup scale",
                &mut settings.breakup_scale,
                0.5..=64.0,
                true,
                1,
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
            slider_row(
                ui,
                "Breakup scale",
                &mut settings.breakup_scale,
                0.5..=64.0,
                true,
                1,
            );
            seed_row(ui, &mut settings.seed);
        }
        MaskGenerator::Orientation(settings) => {
            widgets::property_row(ui, "Direction", |ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                for direction in Direction::ALL {
                    if widgets::chip(ui, direction.name(), settings.direction == direction)
                        .clicked()
                    {
                        settings.direction = direction;
                    }
                }
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
            slider_row(ui, "Scale", &mut settings.scale, 0.5..=64.0, true, 1);
            slider_row(ui, "Amount", &mut settings.amount, 0.0..=1.0, false, 2);
            slider_row(ui, "Contrast", &mut settings.contrast, 0.0..=1.0, false, 2);
            seed_row(ui, &mut settings.seed);
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
        ui.add(
            egui::Slider::new(value, range)
                .logarithmic(logarithmic)
                .fixed_decimals(decimals),
        )
        .changed()
    })
}
