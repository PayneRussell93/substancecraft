//! In-app documentation: a floating guide opened from the dock (or F1)
//! that teaches the intended workflow. Topics hold guided steps that tick
//! themselves off from live app state, and "Show me" buttons that open the
//! drawer, tool, or view a step talks about.

use eframe::egui::{
    self, Align, Color32, CornerRadius, FontId, Id, Layout, Margin, Order, Rect, RichText, Sense,
    Stroke, Ui, pos2, text::LayoutJob, vec2,
};

use super::{PaintMode, SubstanceCraftApp, ViewMode};
use crate::icons::{self, Icon};
use crate::panels::Drawer;
use crate::preferences::Action;
use crate::{theme, widgets};

const WINDOW_SIZE: egui::Vec2 = egui::vec2(760.0, 560.0);
const INDEX_WIDTH: f32 = 196.0;

/// Whether the guide is open, and where the reader is in it.
#[derive(Default)]
pub struct DocsState {
    pub open: bool,
    topic: usize,
    query: String,
}

/// Something the guide can do for the reader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DocAction {
    Drawer(Drawer),
    Tool(PaintMode),
    View(ViewMode),
    ShowInspector,
    ImportMesh,
    AddTextureSet,
    AddPaintLayer,
    AddFillLayer,
    /// Jump to another topic by id.
    Topic(&'static str),
}

/// App state a guided step waits for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Check {
    MeshLoaded,
    TextureSet,
    Layer,
    LayerSelected,
    Tool(PaintMode),
    View(ViewMode),
    DrawerOpen(Drawer),
    InspectorShown,
    MeshMapsBaked,
    Bitmaps,
    PoseSaved,
    Hdri,
    Saved,
}

pub struct Step {
    pub text: &'static str,
    pub check: Option<Check>,
    pub action: Option<(&'static str, DocAction)>,
}

pub enum Block {
    Heading(&'static str),
    /// A paragraph; `**text**` marks words that name a control.
    Text(&'static str),
    Tip(&'static str),
    Steps(&'static [Step]),
    /// Term and meaning pairs.
    Terms(&'static [(&'static str, &'static str)]),
    /// The reader's current key bindings.
    Shortcuts,
    Actions(&'static [(&'static str, DocAction)]),
}

pub struct Topic {
    pub id: &'static str,
    pub title: &'static str,
    pub icon: Icon,
    pub summary: &'static str,
    pub blocks: &'static [Block],
}

impl Topic {
    fn steps(&self) -> impl Iterator<Item = &Step> {
        self.blocks.iter().flat_map(|block| match block {
            Block::Steps(steps) => steps.iter(),
            _ => [].iter(),
        })
    }

    fn text(&self) -> impl Iterator<Item = &'static str> + '_ {
        [self.title, self.summary]
            .into_iter()
            .chain(self.blocks.iter().flat_map(|block| -> Vec<&'static str> {
                match block {
                    Block::Heading(text) | Block::Text(text) | Block::Tip(text) => vec![text],
                    Block::Steps(steps) => steps.iter().map(|step| step.text).collect(),
                    Block::Terms(terms) => terms
                        .iter()
                        .flat_map(|(term, meaning)| [*term, *meaning])
                        .collect(),
                    Block::Shortcuts => Action::ALL.iter().map(|action| action.label()).collect(),
                    Block::Actions(actions) => actions.iter().map(|(label, _)| *label).collect(),
                }
            }))
    }

    /// True when every word of `query` appears somewhere in the topic.
    pub fn matches(&self, query: &str) -> bool {
        let haystack = self.text().collect::<Vec<_>>().join(" ").to_lowercase();
        query
            .split_whitespace()
            .all(|word| haystack.contains(&word.to_lowercase()))
    }
}

pub fn topic_index(id: &str) -> Option<usize> {
    TOPICS.iter().position(|topic| topic.id == id)
}

impl SubstanceCraftApp {
    pub(super) fn toggle_docs(&mut self) {
        self.docs.open = !self.docs.open;
    }

    /// Opens the guide at the topic with `id`.
    pub(super) fn open_docs_topic(&mut self, id: &str) {
        if let Some(index) = topic_index(id) {
            self.docs.topic = index;
            self.docs.query.clear();
            self.docs.open = true;
        }
    }

    pub(super) fn docs_check(&self, check: Check) -> bool {
        match check {
            Check::MeshLoaded => self.mesh_geometry.is_some(),
            Check::TextureSet => !self.project.texture_sets.is_empty(),
            Check::Layer => self
                .project
                .texture_sets
                .iter()
                .any(|set| !set.layers.is_empty()),
            Check::LayerSelected => self.selected_layer().is_some(),
            Check::Tool(mode) => self.paint_mode == mode,
            Check::View(mode) => self.view_mode == mode,
            Check::DrawerOpen(drawer) => self.drawer == Some(drawer),
            Check::InspectorShown => self.show_inspector,
            Check::MeshMapsBaked => self.mesh_maps_fresh(),
            Check::Bitmaps => !self.project.resources.is_empty(),
            Check::PoseSaved => !self.project.saved_poses.is_empty(),
            Check::Hdri => self.custom_environment.is_some(),
            Check::Saved => self.project_path.is_some(),
        }
    }

    /// Steps of `topic` already done, and how many it has that can be.
    fn docs_progress(&self, topic: &Topic) -> (usize, usize) {
        let checks = topic.steps().filter_map(|step| step.check);
        checks.fold((0, 0), |(done, total), check| {
            (done + usize::from(self.docs_check(check)), total + 1)
        })
    }

    pub(super) fn run_doc_action(&mut self, action: DocAction) {
        match action {
            DocAction::Drawer(drawer) => self.drawer = Some(drawer),
            DocAction::Tool(mode) => {
                self.set_paint_mode(mode);
                self.drawer = Some(Drawer::Brush);
            }
            DocAction::View(mode) => self.view_mode = mode,
            DocAction::ShowInspector => self.show_inspector = true,
            DocAction::ImportMesh => self.import_mesh(),
            DocAction::AddTextureSet => self.add_texture_set(),
            DocAction::AddPaintLayer => self.add_paint_layer(),
            DocAction::AddFillLayer => self.add_fill_layer(),
            DocAction::Topic(id) => self.open_docs_topic(id),
        }
    }

    /// F1 toggles the guide.
    pub(super) fn handle_docs_key(&mut self, context: &egui::Context) {
        if context.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::F1)) {
            self.toggle_docs();
        }
    }

    /// Draws the guide, first placed at `left` (beside the drawer) when it
    /// fits there, else centred; the reader can drag it anywhere.
    pub(super) fn show_docs(&mut self, context: &egui::Context, screen: Rect, top: f32, left: f32) {
        let amount = context.animate_bool_with_time(Id::new("docs_open"), self.docs.open, 0.18);
        if amount <= 0.0 {
            return;
        }
        let size = vec2(
            WINDOW_SIZE.x.min(screen.width() - 2.0 * theme::EDGE),
            WINDOW_SIZE.y.min(screen.bottom() - top - 70.0).max(240.0),
        );
        let x = if left + size.x <= screen.right() - theme::EDGE {
            left
        } else {
            (screen.center().x - size.x * 0.5).max(screen.left() + theme::EDGE)
        };
        let default_pos = pos2(x, top);
        let mut pending = None;
        egui::Area::new(Id::new("overlay_docs"))
            .order(Order::Foreground)
            .default_pos(default_pos)
            .constrain_to(screen)
            .movable(true)
            .show(context, |ui| {
                ui.multiply_opacity(amount);
                theme::card_frame(theme::CARD_RADIUS)
                    .inner_margin(Margin::same(14))
                    .show(ui, |ui| {
                        ui.set_width(size.x - 28.0);
                        self.docs_header(ui);
                        ui.add_space(8.0);
                        let body_height = size.y - 80.0;
                        ui.horizontal_top(|ui| {
                            ui.vertical(|ui| {
                                ui.set_width(INDEX_WIDTH);
                                self.docs_index(ui, body_height, &mut pending);
                            });
                            let (line, _) =
                                ui.allocate_exact_size(vec2(1.0, body_height), Sense::hover());
                            ui.painter().rect_filled(
                                line,
                                0.0,
                                theme::TEXT_FAINT.gamma_multiply(0.4),
                            );
                            ui.vertical(|ui| {
                                egui::ScrollArea::vertical()
                                    .id_salt(("docs_page", self.docs.topic))
                                    .max_height(body_height)
                                    .min_scrolled_height(body_height)
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| self.docs_page(ui, &mut pending));
                            });
                        });
                    });
            });
        if let Some(action) = pending {
            self.run_doc_action(action);
        }
    }

    fn docs_header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(vec2(20.0, 20.0), Sense::hover());
            icons::paint(ui.painter(), rect, Icon::Book, theme::ACCENT);
            ui.label(RichText::new("Guide").size(16.0).color(theme::TEXT));
            ui.label(
                RichText::new("Drag here to move · F1 toggles")
                    .size(11.0)
                    .color(theme::TEXT_FAINT),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if widgets::header_action(ui, Icon::Close, true, "Close (F1)").clicked() {
                    self.docs.open = false;
                }
                ui.add(
                    egui::TextEdit::singleline(&mut self.docs.query)
                        .hint_text("Search the guide…")
                        .desired_width(220.0),
                );
            });
        });
    }

    fn docs_index(&self, ui: &mut Ui, height: f32, pending: &mut Option<DocAction>) {
        let query = self.docs.query.trim();
        egui::ScrollArea::vertical()
            .id_salt("docs_index")
            .max_height(height)
            .min_scrolled_height(height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                let mut shown = 0;
                for (index, topic) in TOPICS.iter().enumerate() {
                    if !query.is_empty() && !topic.matches(query) {
                        continue;
                    }
                    shown += 1;
                    let (done, total) = self.docs_progress(topic);
                    let selected = index == self.docs.topic;
                    if index_row(ui, topic, selected, (done, total)).clicked() {
                        *pending = Some(DocAction::Topic(topic.id));
                    }
                }
                if shown == 0 {
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("No topic matches this search.")
                            .size(12.0)
                            .color(theme::TEXT_FAINT),
                    );
                }
            });
    }

    fn docs_page(&self, ui: &mut Ui, pending: &mut Option<DocAction>) {
        let Some(topic) = TOPICS.get(self.docs.topic) else {
            return;
        };
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(vec2(22.0, 22.0), Sense::hover());
            icons::paint(ui.painter(), rect, topic.icon, theme::ACCENT);
            ui.label(RichText::new(topic.title).size(19.0).color(theme::TEXT));
        });
        paragraph(ui, topic.summary, 13.0);
        let (done, total) = self.docs_progress(topic);
        if total > 0 {
            ui.add_space(4.0);
            progress_bar(ui, done, total);
        }
        for block in topic.blocks {
            ui.add_space(8.0);
            match block {
                Block::Heading(text) => {
                    ui.add_space(4.0);
                    ui.label(RichText::new(*text).size(14.5).strong().color(theme::TEXT));
                }
                Block::Text(text) => paragraph(ui, text, 12.5),
                Block::Tip(text) => tip(ui, text),
                Block::Steps(steps) => self.docs_steps(ui, steps, pending),
                Block::Terms(terms) => terms_grid(ui, topic.id, terms),
                Block::Shortcuts => self.shortcut_table(ui),
                Block::Actions(actions) => {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
                        for (label, action) in *actions {
                            if action_button(ui, label).clicked() {
                                *pending = Some(*action);
                            }
                        }
                    });
                }
            }
        }
        ui.add_space(12.0);
        self.docs_neighbours(ui, pending);
    }

    fn docs_steps(&self, ui: &mut Ui, steps: &[Step], pending: &mut Option<DocAction>) {
        for (number, step) in steps.iter().enumerate() {
            let done = step.check.is_some_and(|check| self.docs_check(check));
            ui.horizontal_top(|ui| {
                step_badge(ui, number + 1, done);
                ui.vertical(|ui| {
                    ui.set_width(ui.available_width() - 4.0);
                    let size = 12.5;
                    if done {
                        let mut job = rich_job(step.text, size);
                        for section in &mut job.sections {
                            section.format.color = theme::TEXT_WEAK;
                        }
                        ui.add(egui::Label::new(job).wrap());
                    } else {
                        paragraph(ui, step.text, size);
                    }
                    if let Some((label, action)) = step.action
                        && action_button(ui, label).clicked()
                    {
                        *pending = Some(action);
                    }
                });
            });
            ui.add_space(4.0);
        }
    }

    fn shortcut_table(&self, ui: &mut Ui) {
        egui::Grid::new("docs_shortcuts")
            .num_columns(2)
            .spacing(vec2(18.0, 6.0))
            .show(ui, |ui| {
                for action in Action::ALL {
                    ui.label(RichText::new(action.label()).size(12.5).color(theme::TEXT));
                    let key = self
                        .preferences
                        .values
                        .binding_of(action)
                        .map_or_else(|| "Unbound".to_owned(), |binding| binding.label());
                    key_cap(ui, &key);
                    ui.end_row();
                }
                for (label, key) in [
                    ("Open or close this guide", "F1"),
                    ("Orbit the 3D view (View tool)", "Drag"),
                    ("Pan the view", "Shift+Drag"),
                    ("Pan the UV view", "Middle drag"),
                    ("Zoom", "Scroll"),
                ] {
                    ui.label(RichText::new(label).size(12.5).color(theme::TEXT));
                    key_cap(ui, key);
                    ui.end_row();
                }
            });
    }

    fn docs_neighbours(&self, ui: &mut Ui, pending: &mut Option<DocAction>) {
        let index = self.docs.topic;
        ui.horizontal(|ui| {
            if let Some(previous) = index.checked_sub(1).and_then(|i| TOPICS.get(i))
                && action_button(ui, &format!("← {}", previous.title)).clicked()
            {
                *pending = Some(DocAction::Topic(previous.id));
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if let Some(next) = TOPICS.get(index + 1)
                    && action_button(ui, &format!("{} →", next.title)).clicked()
                {
                    *pending = Some(DocAction::Topic(next.id));
                }
            });
        });
    }
}

fn index_row(
    ui: &mut Ui,
    topic: &Topic,
    selected: bool,
    (done, total): (usize, usize),
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 30.0), Sense::click());
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        let fill = if selected {
            theme::ACCENT_SOFT
        } else if response.hovered() {
            theme::HOVER
        } else {
            Color32::TRANSPARENT
        };
        painter.rect_filled(rect, CornerRadius::same(10), fill);
        let color = if selected { theme::ACCENT } else { theme::TEXT };
        let icon_rect =
            Rect::from_center_size(rect.left_center() + vec2(16.0, 0.0), vec2(16.0, 16.0));
        icons::paint(painter, icon_rect, topic.icon, color);
        painter.text(
            rect.left_center() + vec2(32.0, 0.0),
            egui::Align2::LEFT_CENTER,
            topic.title,
            FontId::proportional(12.5),
            color,
        );
        if total > 0 {
            let center = rect.right_center() - vec2(14.0, 0.0);
            if done == total {
                painter.circle_filled(center, 6.0, theme::ACCENT);
                check_mark(painter, center, 6.0, theme::ACCENT_TEXT);
            } else {
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    format!("{done}/{total}"),
                    FontId::proportional(10.0),
                    theme::TEXT_FAINT,
                );
            }
        }
    }
    response.on_hover_text(topic.summary)
}

fn check_mark(painter: &egui::Painter, center: egui::Pos2, radius: f32, color: Color32) {
    let s = radius / 6.0;
    painter.line(
        vec![
            center + vec2(-3.0, 0.0) * s,
            center + vec2(-1.0, 2.2) * s,
            center + vec2(3.2, -2.2) * s,
        ],
        Stroke::new(1.6, color),
    );
}

fn step_badge(ui: &mut Ui, number: usize, done: bool) {
    let (rect, _) = ui.allocate_exact_size(vec2(22.0, 22.0), Sense::hover());
    let painter = ui.painter();
    let center = rect.center();
    if done {
        painter.circle_filled(center, 10.0, theme::ACCENT);
        check_mark(painter, center, 10.0, theme::ACCENT_TEXT);
    } else {
        painter.circle_stroke(center, 9.5, Stroke::new(1.2, theme::TEXT_FAINT));
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            number.to_string(),
            FontId::proportional(11.0),
            theme::TEXT,
        );
    }
}

fn progress_bar(ui: &mut Ui, done: usize, total: usize) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(180.0, 6.0), Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect, CornerRadius::same(3), theme::INPUT);
        let share = done as f32 / total.max(1) as f32;
        let filled = Rect::from_min_size(rect.min, vec2(rect.width() * share, rect.height()));
        painter.rect_filled(filled, CornerRadius::same(3), theme::ACCENT);
        let text = if done == total {
            "All done — nice work.".to_owned()
        } else {
            format!("{done} of {total} steps done")
        };
        ui.label(RichText::new(text).size(11.5).color(theme::TEXT_WEAK));
    });
}

fn action_button(ui: &mut Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).size(12.0).color(theme::ACCENT))
            .fill(theme::ACCENT_SOFT)
            .corner_radius(theme::PILL),
    )
}

fn key_cap(ui: &mut Ui, key: &str) {
    egui::Frame::new()
        .fill(theme::INPUT)
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(7, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(key).size(11.5).monospace().color(theme::TEXT));
        });
}

fn tip(ui: &mut Ui, text: &str) {
    egui::Frame::new()
        .fill(theme::ACCENT_SOFT)
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_top(|ui| {
                ui.label(
                    RichText::new("Tip")
                        .size(12.0)
                        .strong()
                        .color(theme::ACCENT),
                );
                ui.add(egui::Label::new(rich_job(text, 12.0)).wrap());
            });
        });
}

fn terms_grid(ui: &mut Ui, id: &str, terms: &[(&str, &str)]) {
    egui::Grid::new((
        "docs_terms",
        id,
        terms.len(),
        terms.first().map(|term| term.0),
    ))
    .num_columns(2)
    .spacing(vec2(14.0, 7.0))
    .show(ui, |ui| {
        for (term, meaning) in terms {
            ui.label(RichText::new(*term).size(12.5).strong().color(theme::TEXT));
            ui.add(egui::Label::new(rich_job(meaning, 12.5)).wrap());
            ui.end_row();
        }
    });
}

fn paragraph(ui: &mut Ui, text: &str, size: f32) {
    ui.add(egui::Label::new(rich_job(text, size)).wrap());
}

/// Lays out `text`, showing `**marked**` words, which name controls, in
/// the accent colour.
fn rich_job(text: &str, size: f32) -> LayoutJob {
    let mut job = LayoutJob::default();
    for (index, part) in text.split("**").enumerate() {
        if part.is_empty() {
            continue;
        }
        let control = index % 2 == 1;
        job.append(
            part,
            0.0,
            egui::TextFormat {
                font_id: FontId::proportional(size),
                color: if control {
                    theme::ACCENT
                } else {
                    theme::TEXT_WEAK
                },
                ..Default::default()
            },
        );
    }
    job
}

const fn step(text: &'static str) -> Step {
    Step {
        text,
        check: None,
        action: None,
    }
}

const fn checked(text: &'static str, check: Check) -> Step {
    Step {
        text,
        check: Some(check),
        action: None,
    }
}

const fn guided(text: &'static str, check: Check, label: &'static str, action: DocAction) -> Step {
    Step {
        text,
        check: Some(check),
        action: Some((label, action)),
    }
}

pub static TOPICS: &[Topic] = &[
    Topic {
        id: "start",
        title: "Getting started",
        icon: Icon::Book,
        summary: "SubstanceCraft paints PBR materials onto 3D meshes. This walk-through takes \
                  you from an empty scene to your first exported textures. Each step ticks \
                  itself off when you do it, so keep this guide open while you work.",
        blocks: &[
            Block::Heading("Your first material"),
            Block::Steps(&[
                guided(
                    "Bring in a model. Open the **Scene** drawer and choose **Import mesh…** \
                     (FBX, glTF, or GLB), or click one of the **Primitives** to practise on.",
                    Check::MeshLoaded,
                    "Open Scene drawer",
                    DocAction::Drawer(Drawer::Scene),
                ),
                guided(
                    "Make sure the inspector column is visible on the right. It holds \
                     **Texture Sets**, **Layers**, and **Properties**.",
                    Check::InspectorShown,
                    "Show inspector",
                    DocAction::ShowInspector,
                ),
                guided(
                    "Add a layer. A texture set is created for you if the mesh has none.",
                    Check::Layer,
                    "Add paint layer",
                    DocAction::AddPaintLayer,
                ),
                guided(
                    "Pick the **Paint** tool from the toolbar (or press its key) and drag \
                     across the model.",
                    Check::Tool(PaintMode::Paint),
                    "Use Paint tool",
                    DocAction::Tool(PaintMode::Paint),
                ),
                guided(
                    "Bake mesh maps so smart masks, edge wear, and dirt know the shape of \
                     your model.",
                    Check::MeshMapsBaked,
                    "Open Bake drawer",
                    DocAction::Drawer(Drawer::Bake),
                ),
                checked(
                    "Save the project with the toolbar's save button or Ctrl+S.",
                    Check::Saved,
                ),
            ]),
            Block::Tip("Everything you change can be undone with **Ctrl+Z**. Experiment freely."),
            Block::Heading("Where to go next"),
            Block::Actions(&[
                ("Tour the interface", DocAction::Topic("interface")),
                ("Learn layers", DocAction::Topic("layers")),
                ("Painting in depth", DocAction::Topic("painting")),
                ("Exporting", DocAction::Topic("export")),
            ]),
        ],
    },
    Topic {
        id: "interface",
        title: "The interface",
        icon: Icon::Sidebar,
        summary: "The model fills the window. Every panel floats on top of it, so you can \
                  hide what you are not using.",
        blocks: &[
            Block::Terms(&[
                (
                    "Toolbar",
                    "Across the top: open and save, **Undo**/**Redo**, the tool switcher \
                     (**View**, **Paint**, **Erase**, **Mask**, **ID**, **Select**), the \
                     viewport preview picker, and the **3D / Split / UV** view switch.",
                ),
                (
                    "Dock",
                    "The column of buttons on the left. Each opens a drawer; click it again \
                     to close the drawer. **Docs** at the bottom opens this guide.",
                ),
                (
                    "Drawers",
                    "Slide-out panels for the scene, tools, assets, baking, poses, lighting, \
                     preferences, and checks.",
                ),
                (
                    "Inspector",
                    "The column on the right: **Texture Sets**, **Layers**, and the \
                     **Properties** of whatever is selected. Toggle it with the sidebar \
                     button at the end of the toolbar.",
                ),
                (
                    "Status pill",
                    "Along the bottom: what is baking, errors, and quick facts about the \
                     scene. Messages also appear briefly as toasts.",
                ),
            ]),
            Block::Heading("Try the drawers"),
            Block::Actions(&[
                ("Scene", DocAction::Drawer(Drawer::Scene)),
                ("Tools", DocAction::Drawer(Drawer::Brush)),
                ("Assets", DocAction::Drawer(Drawer::Assets)),
                ("Bake", DocAction::Drawer(Drawer::Bake)),
                ("Poses", DocAction::Drawer(Drawer::Poses)),
                ("Light", DocAction::Drawer(Drawer::Environment)),
                ("Prefs", DocAction::Drawer(Drawer::Preferences)),
                ("Checks", DocAction::Drawer(Drawer::Diagnostics)),
            ]),
            Block::Heading("Preview picker"),
            Block::Text(
                "The drop-down in the toolbar chooses what the viewport shows. **Material** \
                 is the full lit result. Picking a single channel, such as Roughness or \
                 Height, shows that channel in grayscale so you can check it in isolation. \
                 The **Bake** drawer can also preview a mesh map.",
            ),
        ],
    },
    Topic {
        id: "navigation",
        title: "Navigating the view",
        icon: Icon::Focus,
        summary: "Moving around the model, and switching between the 3D and UV views.",
        blocks: &[
            Block::Terms(&[
                ("Orbit", "Drag with the **View** tool."),
                ("Pan", "Shift+drag. In the UV view, middle-drag also pans."),
                ("Zoom", "Scroll the mouse wheel over the view."),
                (
                    "Navigate while painting",
                    "Hold **Space** (Hold: navigate): drags move the camera, and releasing \
                     returns to your tool.",
                ),
                (
                    "Reset",
                    "The focus button in the toolbar resets the camera and UV view.",
                ),
            ]),
            Block::Heading("Views"),
            Block::Steps(&[
                guided(
                    "**3D** shows the shaded model.",
                    Check::View(ViewMode::Model),
                    "Switch to 3D",
                    DocAction::View(ViewMode::Model),
                ),
                guided(
                    "**Split** shows the model and its UV layout side by side. Drag the \
                     divider to resize them. Strokes work in both halves.",
                    Check::View(ViewMode::Split),
                    "Switch to Split",
                    DocAction::View(ViewMode::Split),
                ),
                guided(
                    "**UV** shows the flattened texture. It is the best view for painting \
                     precise details across seams.",
                    Check::View(ViewMode::Uv),
                    "Switch to UV",
                    DocAction::View(ViewMode::Uv),
                ),
            ]),
            Block::Tip(
                "The **Light** drawer sets the camera numerically (orbit, tilt, zoom) when \
                 you need repeatable angles for renders.",
            ),
        ],
    },
    Topic {
        id: "scene",
        title: "Meshes & the scene",
        icon: Icon::Cube,
        summary: "Loading the model you will paint, and setting it up for good results.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Open the **Scene** drawer.",
                    Check::DrawerOpen(Drawer::Scene),
                    "Open Scene",
                    DocAction::Drawer(Drawer::Scene),
                ),
                guided(
                    "Choose **Import mesh…** and pick an FBX, glTF, or GLB file. Each glTF \
                     material becomes its own texture set automatically.",
                    Check::MeshLoaded,
                    "Import mesh…",
                    DocAction::ImportMesh,
                ),
                step(
                    "Check the **Mesh** facts: vertices, triangles, elements, and UV \
                     islands. A mesh without UVs gets planar UVs, which stretch, so unwrap \
                     it in your modelling tool first if you can.",
                ),
                step(
                    "Adjust **Smoothing**: **Flat** for hard-surface facets, **Auto 60°** for \
                     most models, **Smooth** for organic shapes. **Subdivide** adds detail \
                     for close-up painting.",
                ),
            ]),
            Block::Terms(&[
                (
                    "Scale",
                    "Import scale. Set it before importing when a file uses centimetres \
                     and the model shows up huge.",
                ),
                (
                    "Replace mesh…",
                    "Swaps in a new version of the model and keeps your layers. Use it when \
                     the modeller sends an update.",
                ),
                (
                    "Locate mesh…",
                    "Shown when the mesh file has moved. It points the project at the new \
                     location.",
                ),
            ]),
            Block::Tip(
                "Templates at the top of the Scene drawer start a project with texture sets \
                 and channels already set up for common targets.",
            ),
        ],
    },
    Topic {
        id: "layers",
        title: "Texture sets & layers",
        icon: Icon::Layers,
        summary: "A texture set is one material slot on the mesh with its own textures. \
                  Inside it, layers stack from bottom to top, like an image editor.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Create a texture set, or select one in **Texture Sets**.",
                    Check::TextureSet,
                    "Add texture set",
                    DocAction::AddTextureSet,
                ),
                guided(
                    "Click **+ Paint** in the **Layers** card to add a paint layer that holds \
                     brush strokes.",
                    Check::Layer,
                    "Add paint layer",
                    DocAction::AddPaintLayer,
                ),
                guided(
                    "Select a layer to edit it in **Properties**.",
                    Check::LayerSelected,
                    "Show inspector",
                    DocAction::ShowInspector,
                ),
            ]),
            Block::Heading("Layer types"),
            Block::Terms(&[
                (
                    "Paint layer",
                    "**+ Paint**. Holds brush strokes. Erase removes them again.",
                ),
                (
                    "Fill layer",
                    "**+ Fill**. Covers its mask with one material, uniform or from bitmaps. Fill \
                     layers are non-destructive: change the material at any time.",
                ),
                (
                    "Edge Wear / Dirt",
                    "Ready-made fills with smart masks: bare metal on convex edges, and grime \
                     in cavities. These need baked mesh maps.",
                ),
                (
                    "OpenPBR…",
                    "A fill set up from an OpenPBR material definition.",
                ),
                (
                    "Group",
                    "**+ Group**. Holds layers so they can share a mask, blend, or visibility. Use \
                     **Into group** and the move buttons to organise them.",
                ),
            ]),
            Block::Heading("Combining layers"),
            Block::Text(
                "**Blend** sets how a layer combines with the layers beneath it, and \
                 **Blend per channel** overrides that for each channel. **Channels** chooses \
                 which material channels the layer writes. Turn off **Rough** to paint \
                 colour without changing roughness, for example.",
            ),
            Block::Tip(
                "The search box above the layer list filters by name, type, channel, or \
                 resource. **Solo** in the Tools drawer (View tool) shows one layer or its \
                 mask on its own.",
            ),
            Block::Actions(&[
                ("Add fill layer", DocAction::AddFillLayer),
                ("Next: painting", DocAction::Topic("painting")),
            ]),
        ],
    },
    Topic {
        id: "painting",
        title: "Painting",
        icon: Icon::Brush,
        summary: "Brush strokes paint every channel the selected layer enables (colour, \
                  roughness, metallic, height, and more) in one pass.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Select a paint layer, then choose the **Paint** tool. The Tools drawer \
                     opens with the brush settings.",
                    Check::Tool(PaintMode::Paint),
                    "Use Paint tool",
                    DocAction::Tool(PaintMode::Paint),
                ),
                step(
                    "Set the **Color**, **Roughness**, **Metallic**, and **Height** the brush \
                     lays down. Height 0.5 is neutral; higher values raise the surface.",
                ),
                step("Set **Size**, then drag on the model or in the UV view."),
                guided(
                    "Switch to **Erase** to remove paint from the layer.",
                    Check::Tool(PaintMode::Erase),
                    "Use Erase tool",
                    DocAction::Tool(PaintMode::Erase),
                ),
            ]),
            Block::Heading("Brush tip"),
            Block::Terms(&[
                ("Hardness", "Soft edges at 0, crisp edges at 1."),
                (
                    "Spacing",
                    "Distance between dabs, as a share of the brush size.",
                ),
                (
                    "Angle° / Follow path",
                    "Rotates the tip, or turns it along the stroke.",
                ),
                (
                    "Jitter & Scatter",
                    "Vary size, angle, and position for organic marks.",
                ),
                (
                    "Pen pressure",
                    "With a tablet, pressure can drive size and opacity.",
                ),
                (
                    "Bitmap",
                    "Use any imported bitmap as the brush alpha. In the **Assets** drawer, \
                     click **Brush alpha** next to a bitmap.",
                ),
            ]),
            Block::Heading("Tools that sample the stack"),
            Block::Terms(&[
                (
                    "Clone",
                    "Click **Set source**, click the model, and then paint to copy the stack \
                     from that point.",
                ),
                ("Smudge", "Drags the layer stack along the stroke."),
                ("Blur", "Softens the layer stack under the stroke."),
            ]),
            Block::Heading("Assists"),
            Block::Terms(&[
                (
                    "Symmetry",
                    "**Mirror** strokes across a plane, or repeat them as **Copies** around \
                     an axis.",
                ),
                (
                    "Lazy radius",
                    "Smooths shaky strokes. The brush trails the cursor by this distance.",
                ),
                (
                    "Draw path",
                    "Click points to lay a stroke along a precise path.",
                ),
                (
                    "Align to surface",
                    "Projects the tip along the surface normal.",
                ),
            ]),
            Block::Tip(
                "Pin brushes you like to the **Shelf**, and use **+ Material** to save a \
                 tip together with its material as a preset. Hold **I** to pick the material \
                 under the cursor, and hold **X** to erase briefly.",
            ),
        ],
    },
    Topic {
        id: "masks",
        title: "Masks & selection",
        icon: Icon::EyeOff,
        summary: "Masks decide where a layer shows. Combine painted, generated, bitmap, and \
                  polygon masks to place materials exactly.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Paint a mask with the **Mask** tool. **Mask value** 1 reveals the layer \
                     and 0 hides it.",
                    Check::Tool(PaintMode::Mask),
                    "Use Mask tool",
                    DocAction::Tool(PaintMode::Mask),
                ),
                guided(
                    "Use **Select** to limit a layer to parts of the mesh. Pick faces, \
                     connected elements, or whole UV islands, then **Add fill layer** or use \
                     them as a polygon mask.",
                    Check::Tool(PaintMode::Select),
                    "Use Select tool",
                    DocAction::Tool(PaintMode::Select),
                ),
                guided(
                    "Paint ID colours with the **ID** tool, then use **Clip by ID** on a \
                     layer to keep it inside one colour.",
                    Check::Tool(PaintMode::Id),
                    "Use ID tool",
                    DocAction::Tool(PaintMode::Id),
                ),
            ]),
            Block::Heading("Procedural masks"),
            Block::Terms(&[
                (
                    "Mask generator",
                    "Builds the mask from the shape of the mesh (curvature, occlusion, \
                     position) or from noise. Generators from plugins appear here too.",
                ),
                (
                    "Smart mask",
                    "A saved generator setup. **Apply…** reuses one, and you can keep the \
                     current setup as a new smart mask.",
                ),
                ("Bitmap mask", "Drives the mask from an imported image."),
                (
                    "Mask filters",
                    "Blur, levels, and other adjustments on the final mask.",
                ),
                (
                    "Invert final mask",
                    "Swaps where the layer shows and hides.",
                ),
                (
                    "UV tiles",
                    "On UDIM meshes, limits the layer to chosen tiles.",
                ),
            ]),
            Block::Tip(
                "Press **T** for a quick mask: paint a temporary mask, and press T again to \
                 finish it. The Select tool's **Hide** and **Isolate** hide faces that get \
                 in the way of your brush.",
            ),
        ],
    },
    Topic {
        id: "fills",
        title: "Fills & procedurals",
        icon: Icon::Grid,
        summary: "Fill layers are the non-destructive backbone of a material. Stack fills \
                  with smart masks instead of painting everything by hand.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Add a fill layer and set its channel values in **Properties**.",
                    Check::Layer,
                    "Add fill layer",
                    DocAction::AddFillLayer,
                ),
                step(
                    "Assign a bitmap to a channel to use a texture, and choose a \
                     **Projection** (UV, triplanar, planar) and **Scale**.",
                ),
                step(
                    "Choose **Make procedural** to drive the fill from a node graph of \
                     generators, adjustments, and gradients.",
                ),
                step("Add a mask generator so the fill only shows where it should."),
            ]),
            Block::Heading("Reusing work"),
            Block::Terms(&[
                (
                    "Smart materials",
                    "Save a layer or group as a smart material and drop it onto another \
                     texture set. It adapts to that mesh's maps.",
                ),
                (
                    "Filters",
                    "Adjust a layer's output. Use **On channels** to limit them to chosen \
                     channels.",
                ),
                (
                    "Flatten to bitmaps",
                    "Merges selected layers into one paint layer when the stack is final.",
                ),
            ]),
            Block::Tip(
                "Generators that read curvature or occlusion need mesh maps. If a layer \
                 shows **Mesh maps needed**, bake them in the Bake drawer.",
            ),
            Block::Actions(&[("Learn baking", DocAction::Topic("bake"))]),
        ],
    },
    Topic {
        id: "bake",
        title: "Baking mesh maps",
        icon: Icon::Bake,
        summary: "Mesh maps (ambient occlusion, curvature, thickness, normals, and more) \
                  describe your model's shape. Smart masks and generators read them.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Open the **Bake** drawer.",
                    Check::DrawerOpen(Drawer::Bake),
                    "Open Bake",
                    DocAction::Drawer(Drawer::Bake),
                ),
                step(
                    "Pick a **Resolution**. Bake at your export size or higher. Raise **AO \
                     samples** for smoother occlusion.",
                ),
                step(
                    "Optional: **Load high poly FBX…** to bake detail from a sculpt. Raise \
                     **Cage distance** if the panel warns that rays miss.",
                ),
                guided(
                    "Click **Bake mesh maps** and wait for the status to read **Up to \
                     date**.",
                    Check::MeshMapsBaked,
                    "Open Bake",
                    DocAction::Drawer(Drawer::Bake),
                ),
            ]),
            Block::Terms(&[
                (
                    "Stale because…",
                    "The mesh or settings changed after the last bake, so bake again.",
                ),
                (
                    "Padding",
                    "Spreads pixels past UV edges so seams do not show at a distance.",
                ),
                (
                    "Skew fix",
                    "Corrects stretched projections on slanted faces. Paint a skew map to \
                     control it by hand.",
                ),
                (
                    "GPU baking",
                    "Traces rays on the graphics card. It is much faster.",
                ),
                (
                    "Preview on mesh",
                    "Shows one baked map on the model to check it.",
                ),
            ]),
        ],
    },
    Topic {
        id: "assets",
        title: "Bitmaps & the library",
        icon: Icon::Image,
        summary: "The Assets drawer manages the images a project uses, and the shared \
                  library that every project can draw from.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Choose **Import bitmaps…** in the **Assets** drawer (PNG, JPEG, TGA, or \
                     SVG).",
                    Check::Bitmaps,
                    "Open Assets",
                    DocAction::Drawer(Drawer::Assets),
                ),
                step(
                    "Review the suggested assignments. Names like \"_rough\" or \"_normal\" \
                     pick a channel. Then choose **Create fill layers** to apply them.",
                ),
            ]),
            Block::Terms(&[
                (
                    "Image sequences",
                    "Animated bitmaps for flipbooks and evolving effects.",
                ),
                (
                    "Text resources",
                    "**New text resource…** renders text you can use like a bitmap, for \
                     labels and decals.",
                ),
                (
                    "Auto-reload",
                    "Picks up edits you save in an image editor without reimporting.",
                ),
                ("Relink…", "Points a missing bitmap at its moved file."),
                (
                    "Shared library",
                    "Filters, generators, and environments kept on disk for every project. \
                     **To library** saves one; **Add to this project** brings one in.",
                ),
            ]),
        ],
    },
    Topic {
        id: "poses",
        title: "Poses",
        icon: Icon::Figure,
        summary: "Skinned meshes can be posed so you can paint areas hidden in the rest pose, \
                  then check how the material moves.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Open the **Poses** drawer on a mesh with a skeleton or animation.",
                    Check::DrawerOpen(Drawer::Poses),
                    "Open Poses",
                    DocAction::Drawer(Drawer::Poses),
                ),
                step("Scrub an animation or adjust bones until the area you need is visible."),
                checked(
                    "Type a **Pose name** and choose **Save** to keep it with the project.",
                    Check::PoseSaved,
                ),
            ]),
            Block::Tip(
                "Painting on a posed mesh lands on the same texels as in the rest pose, so \
                 pose freely.",
            ),
        ],
    },
    Topic {
        id: "light",
        title: "Camera, light & renders",
        icon: Icon::Sun,
        summary: "Light the model the way it will be seen, and render stills or turntables \
                  for review.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Open the **Light** drawer.",
                    Check::DrawerOpen(Drawer::Environment),
                    "Open Light",
                    DocAction::Drawer(Drawer::Environment),
                ),
                checked(
                    "Choose **Load HDRI…** to light the model with an .hdr or .exr \
                     panorama. Set its **Rotation** and **Strength**.",
                    Check::Hdri,
                ),
                step(
                    "Turn on **Background** to see the environment behind the model, and use \
                     **Blur** to soften it.",
                ),
                step(
                    "Under **Render**, save a PNG of the current view, or **Render \
                     turntable…** for a looping sequence of one full turn.",
                ),
            ]),
            Block::Tip(
                "Judge materials under several HDRIs. A material that only looks right \
                 under one light usually has its roughness or colour values wrong.",
            ),
        ],
    },
    Topic {
        id: "export",
        title: "Exporting textures",
        icon: Icon::Export,
        summary: "Export bakes each texture set's layer stack into image files for your \
                  game engine or renderer.",
        blocks: &[
            Block::Steps(&[
                guided(
                    "Select a texture set in the inspector. Its **Properties** show **Bake & \
                     export**.",
                    Check::TextureSet,
                    "Show inspector",
                    DocAction::ShowInspector,
                ),
                step("Pick a **Format** (PNG, TGA, EXR, …)."),
                step(
                    "Choose an export preset for your engine, or click a channel to export \
                     just that map.",
                ),
            ]),
            Block::Terms(&[
                (
                    "Presets",
                    "Pack channels the way an engine expects (for example, occlusion, \
                     roughness, and metallic in one RGB image). Make your own for custom \
                     shaders.",
                ),
                (
                    "Overwrite prompt",
                    "SubstanceCraft asks before replacing files that already exist.",
                ),
            ]),
            Block::Tip(
                "Run the **Checks** drawer before a final export. It flags missing files and \
                 stale mesh maps.",
            ),
            Block::Actions(&[("Open Checks", DocAction::Drawer(Drawer::Diagnostics))]),
        ],
    },
    Topic {
        id: "preferences",
        title: "Preferences & accessibility",
        icon: Icon::Sliders,
        summary: "Make SubstanceCraft fit your screen, your eyes, and your hands.",
        blocks: &[
            Block::Terms(&[
                (
                    "Interface scale",
                    "Enlarges or shrinks every panel and its text.",
                ),
                (
                    "Colour vision",
                    "Changes overlay colours, and can add **Patterns** (hatching, dots, \
                     dashes) so overlays never rely on hue alone.",
                ),
                (
                    "Keys and buttons",
                    "Rebind any action to a key, a mouse side button, or a gamepad button. \
                     Pen side buttons usually send mouse side buttons.",
                ),
                (
                    "Plugins",
                    "Lists installed plugins. **Reload plugins** picks up new ones.",
                ),
            ]),
            Block::Actions(&[
                ("Open Preferences", DocAction::Drawer(Drawer::Preferences)),
                ("Shortcuts", DocAction::Topic("shortcuts")),
            ]),
        ],
    },
    Topic {
        id: "checks",
        title: "Checks & recovery",
        icon: Icon::Alert,
        summary: "When something looks wrong, start here.",
        blocks: &[
            Block::Terms(&[
                (
                    "Checks drawer",
                    "Lists the health of the project, mesh, mesh maps, files, renderer, \
                     preview, and cache. **Copy report** puts everything on the clipboard \
                     for a bug report.",
                ),
                (
                    "Autosave",
                    "If SubstanceCraft did not close properly, it offers to **Restore** \
                     your work when it starts again.",
                ),
                (
                    "Missing files",
                    "Use **Locate mesh…** in the Scene drawer and **Relink…** in the Assets \
                     drawer to point the project at moved files.",
                ),
                (
                    "Mesh maps needed",
                    "A layer's generator or projection needs mesh maps. Bake them in the \
                     Bake drawer.",
                ),
            ]),
            Block::Actions(&[("Open Checks", DocAction::Drawer(Drawer::Diagnostics))]),
        ],
    },
    Topic {
        id: "shortcuts",
        title: "Keyboard shortcuts",
        icon: Icon::Grid,
        summary: "Your current bindings. Change them in Preferences under **Keys and \
                  buttons**. Hold bindings switch tools only while held.",
        blocks: &[
            Block::Shortcuts,
            Block::Actions(&[("Edit bindings", DocAction::Drawer(Drawer::Preferences))]),
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_ids_are_unique_and_links_resolve() {
        for (index, topic) in TOPICS.iter().enumerate() {
            assert_eq!(
                topic_index(topic.id),
                Some(index),
                "duplicate id {}",
                topic.id
            );
            let actions = topic.blocks.iter().flat_map(|block| -> Vec<DocAction> {
                match block {
                    Block::Actions(actions) => actions.iter().map(|(_, a)| *a).collect(),
                    Block::Steps(steps) => steps
                        .iter()
                        .filter_map(|step| step.action.map(|a| a.1))
                        .collect(),
                    _ => Vec::new(),
                }
            });
            for action in actions {
                if let DocAction::Topic(id) = action {
                    assert!(
                        topic_index(id).is_some(),
                        "{} links to missing {id}",
                        topic.id
                    );
                }
            }
        }
    }

    #[test]
    fn control_markup_is_balanced() {
        for topic in TOPICS {
            for text in topic.text() {
                assert_eq!(text.matches("**").count() % 2, 0, "{}: {text}", topic.id);
            }
        }
    }

    #[test]
    fn search_matches_every_word_in_any_block() {
        let bake = &TOPICS[topic_index("bake").unwrap_or_default()];
        assert!(bake.matches("cage DISTANCE"));
        assert!(!bake.matches("cage turntable"));
        let shortcuts = &TOPICS[topic_index("shortcuts").unwrap_or_default()];
        assert!(shortcuts.matches("quick mask"));
    }

    #[test]
    fn steps_tick_off_from_app_state_and_actions_drive_it() {
        let mut app = SubstanceCraftApp::default();
        app.set_paint_mode(PaintMode::View);
        assert!(!app.docs_check(Check::Tool(PaintMode::Paint)));
        app.run_doc_action(DocAction::Tool(PaintMode::Paint));
        assert!(app.docs_check(Check::Tool(PaintMode::Paint)));
        assert_eq!(app.drawer, Some(Drawer::Brush));

        app.run_doc_action(DocAction::Drawer(Drawer::Bake));
        assert!(app.docs_check(Check::DrawerOpen(Drawer::Bake)));

        app.run_doc_action(DocAction::View(ViewMode::Uv));
        assert!(app.docs_check(Check::View(ViewMode::Uv)));

        assert!(app.docs_check(Check::MeshLoaded));
        app.run_doc_action(DocAction::AddPaintLayer);
        assert!(app.docs_check(Check::TextureSet));
        assert!(app.docs_check(Check::Layer));
    }

    #[test]
    fn topic_links_open_the_guide_there() {
        let mut app = SubstanceCraftApp::default();
        app.docs.query = "bake".to_owned();
        app.run_doc_action(DocAction::Topic("export"));
        assert!(app.docs.open);
        assert_eq!(Some(app.docs.topic), topic_index("export"));
        assert!(app.docs.query.is_empty());
        app.toggle_docs();
        assert!(!app.docs.open);
    }
}
