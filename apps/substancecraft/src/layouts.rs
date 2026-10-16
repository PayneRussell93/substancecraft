//! Saved panel layouts: which viewports share the window and where the
//! split sits, whether the inspector shows and how wide, and which drawer
//! is open. Built-in layouts cover common tasks; any arrangement can be
//! saved under a name. Layouts and the arrangement in use are user
//! preferences, so they follow the user across projects and sessions.

use eframe::egui::{RichText, Ui};
use serde::{Deserialize, Serialize};

use crate::panels::{Drawer, hint, slider_row};
use crate::{SubstanceCraftApp, ViewMode, theme, widgets};

pub const DEFAULT_INSPECTOR_WIDTH: f32 = 296.0;
pub const MIN_INSPECTOR_WIDTH: f32 = 240.0;
pub const MAX_INSPECTOR_WIDTH: f32 = 480.0;
/// Smallest share either side of the split view keeps.
const MIN_SPLIT: f32 = 0.2;
/// Saved layouts kept at most.
pub const MAX_SAVED_LAYOUTS: usize = 12;
const MAX_LAYOUT_NAME_CHARS: usize = 32;

pub fn clamp_split(split: f32) -> f32 {
    if split.is_finite() {
        split.clamp(MIN_SPLIT, 1.0 - MIN_SPLIT)
    } else {
        0.5
    }
}

/// One arrangement of the window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PanelLayout {
    pub name: String,
    pub(crate) view_mode: ViewMode,
    pub split: f32,
    pub show_inspector: bool,
    pub inspector_width: f32,
    pub drawer: Option<Drawer>,
}

impl Default for PanelLayout {
    fn default() -> Self {
        Self {
            name: "Default".to_owned(),
            view_mode: ViewMode::Model,
            split: 0.5,
            show_inspector: true,
            inspector_width: DEFAULT_INSPECTOR_WIDTH,
            drawer: None,
        }
    }
}

impl PanelLayout {
    /// Original arrangements for common tasks.
    pub fn built_in() -> Vec<Self> {
        let layout = |name: &str, view_mode, split, show_inspector, drawer| Self {
            name: name.to_owned(),
            view_mode,
            split,
            show_inspector,
            inspector_width: DEFAULT_INSPECTOR_WIDTH,
            drawer,
        };
        vec![
            layout("Default", ViewMode::Model, 0.5, true, None),
            layout("Paint", ViewMode::Model, 0.5, true, Some(Drawer::Brush)),
            layout("Texture", ViewMode::Split, 0.6, true, Some(Drawer::Assets)),
            layout("Bake", ViewMode::Model, 0.5, true, Some(Drawer::Bake)),
            layout("Focus", ViewMode::Model, 0.5, false, None),
        ]
    }

    /// The layout with out-of-range sizes brought into range.
    pub fn sanitized(mut self) -> Self {
        self.split = clamp_split(self.split);
        self.inspector_width = if self.inspector_width.is_finite() {
            self.inspector_width
                .clamp(MIN_INSPECTOR_WIDTH, MAX_INSPECTOR_WIDTH)
        } else {
            DEFAULT_INSPECTOR_WIDTH
        };
        self.name = self
            .name
            .trim()
            .chars()
            .take(MAX_LAYOUT_NAME_CHARS)
            .collect();
        self
    }
}

impl SubstanceCraftApp {
    /// The window's arrangement now, named `name`.
    pub(super) fn current_layout(&self, name: &str) -> PanelLayout {
        PanelLayout {
            name: name.to_owned(),
            view_mode: self.view_mode,
            split: self.layout.split,
            show_inspector: self.show_inspector,
            inspector_width: self.layout.inspector_width,
            drawer: self.drawer,
        }
    }

    /// Arranges the window as `layout` and remembers it for next session.
    pub(super) fn apply_layout(&mut self, layout: &PanelLayout) {
        let layout = layout.clone().sanitized();
        self.view_mode = layout.view_mode;
        self.layout.split = layout.split;
        self.show_inspector = layout.show_inspector;
        self.layout.inspector_width = layout.inspector_width;
        self.drawer = layout.drawer;
        self.remember_layout();
    }

    /// Saves the arrangement in use to preferences when it changed.
    pub(super) fn remember_layout(&mut self) {
        let current = self.current_layout("");
        if self.preferences.values.window_layout.as_ref() != Some(&current) {
            self.preferences.values.window_layout = Some(current);
            self.save_preferences();
        }
    }

    /// Restores the arrangement saved in preferences, if any.
    pub(super) fn restore_layout(&mut self) {
        if let Some(layout) = self.preferences.values.window_layout.clone() {
            let layout = layout.sanitized();
            self.view_mode = layout.view_mode;
            self.layout.split = layout.split;
            self.show_inspector = layout.show_inspector;
            self.layout.inspector_width = layout.inspector_width;
            self.drawer = layout.drawer;
        }
    }

    /// Saves the arrangement in use as layout `name`, replacing a saved
    /// layout of the same name. Returns false for an empty name, a
    /// built-in's name, or a full list.
    pub(super) fn save_layout(&mut self, name: &str) -> bool {
        let layout = self.current_layout(name).sanitized();
        if layout.name.is_empty()
            || PanelLayout::built_in()
                .iter()
                .any(|built_in| built_in.name.eq_ignore_ascii_case(&layout.name))
        {
            return false;
        }
        let saved = &mut self.preferences.values.layouts;
        if let Some(existing) = saved
            .iter_mut()
            .find(|existing| existing.name.eq_ignore_ascii_case(&layout.name))
        {
            *existing = layout;
        } else if saved.len() < MAX_SAVED_LAYOUTS {
            saved.push(layout);
        } else {
            return false;
        }
        self.save_preferences();
        true
    }

    pub(super) fn delete_layout(&mut self, name: &str) {
        let before = self.preferences.values.layouts.len();
        self.preferences
            .values
            .layouts
            .retain(|layout| !layout.name.eq_ignore_ascii_case(name));
        if self.preferences.values.layouts.len() != before {
            self.save_preferences();
        }
    }

    /// The Layouts section of the Preferences drawer.
    pub(super) fn layouts_section(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Layouts");
        let mut apply = None;
        let mut delete = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for layout in PanelLayout::built_in() {
                if widgets::chip(ui, &layout.name, false).clicked() {
                    apply = Some(layout);
                }
            }
        });
        for layout in self.preferences.values.layouts.clone() {
            ui.horizontal(|ui| {
                if widgets::chip(ui, &layout.name, false)
                    .on_hover_text("Arrange the window like this")
                    .clicked()
                {
                    apply = Some(layout.clone());
                }
                if widgets::chip(ui, "Delete", false).clicked() {
                    delete = Some(layout.name.clone());
                }
            });
        }
        let mut name = std::mem::take(&mut self.layout_name);
        let mut save = false;
        widgets::property_row(ui, "Save as", |ui| {
            ui.text_edit_singleline(&mut name);
            save = widgets::pill_button(ui, None, "Save", false, false)
                .on_hover_text("Save the current arrangement under this name")
                .clicked();
        });
        if save {
            if self.save_layout(&name) {
                name.clear();
            } else {
                hint(
                    ui,
                    "Use a new name (not a built-in's); up to 12 layouts are kept.",
                );
            }
        }
        self.layout_name = name;
        let mut width = self.layout.inspector_width;
        slider_row(
            ui,
            "Inspector",
            &mut width,
            MIN_INSPECTOR_WIDTH..=MAX_INSPECTOR_WIDTH,
            false,
            0,
        );
        let mut split = self.layout.split;
        slider_row(
            ui,
            "Split",
            &mut split,
            MIN_SPLIT..=1.0 - MIN_SPLIT,
            false,
            2,
        );
        if (width, split) != (self.layout.inspector_width, self.layout.split) {
            self.layout.inspector_width = width;
            self.layout.split = split;
        }
        ui.label(
            RichText::new("Drag the line between the 3D and UV views to move the split.")
                .size(11.0)
                .color(theme::TEXT_FAINT),
        );
        if let Some(layout) = apply {
            self.apply_layout(&layout);
        }
        if let Some(name) = delete {
            self.delete_layout(&name);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tc_0396_panel_layouts_save_apply_and_come_back_next_session() {
        let folder = tempfile::tempdir().unwrap();
        let preferences = folder.path().join("preferences.json");
        let mut app = SubstanceCraftApp::default();
        app.load_preferences(preferences.clone());

        // A built-in arranges the window.
        let texture = PanelLayout::built_in()
            .into_iter()
            .find(|layout| layout.name == "Texture")
            .unwrap();
        app.apply_layout(&texture);
        assert_eq!(app.view_mode, ViewMode::Split);
        assert_eq!(app.drawer, Some(Drawer::Assets));
        assert_eq!(app.layout.split, 0.6);

        // An adjusted arrangement saves under a name; sizes are clamped.
        app.show_inspector = false;
        app.layout.inspector_width = 9_000.0;
        app.layout.split = 0.01;
        assert!(app.save_layout("Wide UV"));
        let saved = app.preferences.values.layouts[0].clone();
        assert_eq!(saved.inspector_width, MAX_INSPECTOR_WIDTH);
        assert_eq!(saved.split, MIN_SPLIT);
        assert!(!saved.show_inspector);
        assert!(!app.save_layout("Paint"), "built-in names are reserved");
        assert!(!app.save_layout("  "));
        // Saving again under the same name replaces it.
        app.layout.split = 0.7;
        assert!(app.save_layout("wide uv"));
        assert_eq!(app.preferences.values.layouts.len(), 1);
        assert_eq!(app.preferences.values.layouts[0].split, 0.7);

        // Back to Default, then the saved layout returns everything.
        app.apply_layout(&PanelLayout::default());
        assert!(app.show_inspector);
        assert_eq!(app.view_mode, ViewMode::Model);
        let wide = app.preferences.values.layouts[0].clone();
        app.apply_layout(&wide);
        assert!(!app.show_inspector);
        assert_eq!(app.view_mode, ViewMode::Split);

        // Saved layouts and the arrangement in use outlive the session.
        let mut later = SubstanceCraftApp::default();
        later.load_preferences(preferences.clone());
        assert_eq!(later.preferences.values.layouts.len(), 1);
        assert!(!later.show_inspector);
        assert_eq!(later.view_mode, ViewMode::Split);
        assert_eq!(later.layout.split, 0.7);
        later.delete_layout("Wide UV");
        assert!(later.preferences.values.layouts.is_empty());
    }
}
