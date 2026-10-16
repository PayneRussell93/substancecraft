//! OpenColorIO views in the viewport: a config's display and view are
//! baked into the display LUT (see `substancecraft_api::OcioConfig`), and
//! while one is active the renderer runs its Linear view, exposed down by
//! the bake's headroom, so the OCIO view does all of the tone mapping.

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui::{self, RichText, Ui};
use serde::{Deserialize, Serialize};
use substancecraft_api::{OCIO_BAKE_SIZE, OCIO_HEADROOM_STOPS, OcioConfig};
use substancecraft_core::{CubeLut, ViewTransform};

use crate::icons::Icon;
use crate::panels::hint;
use crate::{SubstanceCraftApp, theme, widgets};

/// The OCIO view remembered in preferences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcioChoice {
    pub config: PathBuf,
    pub display: String,
    pub view: String,
}

/// A loaded config and the view baked from it.
pub struct OcioState {
    pub config: Arc<OcioConfig>,
    pub display: String,
    pub view: String,
    pub lut: Arc<CubeLut>,
}

impl SubstanceCraftApp {
    /// Loads the config at `config` and shows `display`/`view` (the
    /// config's first display and view when empty or unknown). A config or
    /// view that cannot be read leaves the current grading and says why.
    pub(super) fn load_ocio_view(
        &mut self,
        config: PathBuf,
        display: &str,
        view: &str,
    ) -> Result<(), String> {
        let loaded = match &self.ocio {
            Some(state) if state.config.path() == config => Arc::clone(&state.config),
            _ => Arc::new(OcioConfig::load(&config).map_err(|error| self.ocio_failed(error))?),
        };
        let displays = loaded.displays();
        let display = displays
            .iter()
            .find(|candidate| candidate.name.eq_ignore_ascii_case(display))
            .or(displays.first())
            .map(|display| display.name.clone())
            .ok_or_else(|| self.ocio_failed("the config has no displays"))?;
        let views = loaded.views(&display);
        let view = views
            .iter()
            .find(|candidate| candidate.name.eq_ignore_ascii_case(view))
            .or(views.first())
            .map(|view| view.name.clone())
            .ok_or_else(|| self.ocio_failed("the display has no views"))?;
        let input = loaded
            .linear_rec709_space()
            .ok_or_else(|| self.ocio_failed("the config has no linear Rec.709 colour space"))?
            .to_owned();
        let lut = loaded
            .bake_view(&input, &display, &view, OCIO_BAKE_SIZE)
            .map_err(|error| self.ocio_failed(format!("{display} / {view}: {error}")))?;
        let choice = OcioChoice {
            config,
            display: display.clone(),
            view: view.clone(),
        };
        self.ocio = Some(OcioState {
            config: loaded,
            display,
            view,
            lut: Arc::new(lut),
        });
        self.grading_status = None;
        if self.preferences.values.ocio.as_ref() != Some(&choice) {
            self.preferences.values.ocio = Some(choice);
            self.save_preferences();
        }
        Ok(())
    }

    fn ocio_failed(&mut self, error: impl std::fmt::Display) -> String {
        let message = format!("OCIO: {error}");
        self.grading_status = Some(message.clone());
        message
    }

    /// Turns the OCIO view off and forgets it.
    pub(super) fn clear_ocio_view(&mut self) {
        self.ocio = None;
        self.grading_status = None;
        if self.preferences.values.ocio.take().is_some() {
            self.save_preferences();
        }
    }

    /// Restores the OCIO view saved in preferences; a missing config stays
    /// remembered and is reported.
    pub(super) fn restore_ocio_view(&mut self) {
        if let Some(choice) = self.preferences.values.ocio.clone() {
            let _ = self.load_ocio_view(choice.config, &choice.display, &choice.view);
        }
    }

    /// What the renderer grades with: the view transform, exposure, and
    /// display LUT, with an OCIO view taking over the tone curve and LUT.
    pub(super) fn grading(&self) -> (ViewTransform, f32, Option<Arc<CubeLut>>) {
        let values = &self.preferences.values;
        match &self.ocio {
            Some(state) => (
                ViewTransform::Linear,
                values.exposure - OCIO_HEADROOM_STOPS,
                Some(Arc::clone(&state.lut)),
            ),
            None => (
                values.view_transform,
                values.exposure,
                self.display_lut.as_ref().map(|lut| Arc::clone(&lut.lut)),
            ),
        }
    }

    /// The OCIO rows of the Light drawer's Colour section.
    pub(super) fn ocio_section(&mut self, ui: &mut Ui) {
        let mut pick = false;
        let mut clear = false;
        let mut choose = None;
        widgets::property_row(ui, "OCIO", |ui| {
            let label = self.ocio.as_ref().map_or("None".to_owned(), |state| {
                crate::path_label(state.config.path())
            });
            ui.label(RichText::new(label).size(11.5).color(theme::TEXT_WEAK));
            pick = widgets::pill_button(ui, Some(Icon::Import), "Config…", false, false)
                .on_hover_text(
                    "Read an OpenColorIO config and view the model through one of its views",
                )
                .clicked();
            clear = self.ocio.is_some()
                && widgets::pill_button(ui, Some(Icon::Close), "Remove", false, false).clicked();
        });
        if let Some(state) = &self.ocio {
            let displays: Vec<String> = state
                .config
                .displays()
                .iter()
                .map(|display| display.name.clone())
                .collect();
            let views: Vec<String> = state
                .config
                .views(&state.display)
                .iter()
                .map(|view| view.name.clone())
                .collect();
            let (mut display, mut view) = (state.display.clone(), state.view.clone());
            widgets::property_row(ui, "Display", |ui| {
                egui::ComboBox::from_id_salt("ocio_display")
                    .width(ui.available_width() - 8.0)
                    .selected_text(&display)
                    .show_ui(ui, |ui| {
                        for name in &displays {
                            ui.selectable_value(&mut display, name.clone(), name);
                        }
                    });
            });
            widgets::property_row(ui, "View", |ui| {
                egui::ComboBox::from_id_salt("ocio_view")
                    .width(ui.available_width() - 8.0)
                    .selected_text(&view)
                    .show_ui(ui, |ui| {
                        for name in &views {
                            ui.selectable_value(&mut view, name.clone(), name);
                        }
                    });
            });
            if display != state.display || view != state.view {
                choose = Some((state.config.path().to_owned(), display, view));
            }
            hint(
                ui,
                "The OCIO view replaces the view transform and display LUT; exposure still applies.",
            );
        }
        if pick {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("OpenColorIO config", &["ocio"])
                .pick_file()
            {
                let _ = self.load_ocio_view(path, "", "");
            }
        } else if clear {
            self.clear_ocio_view();
        } else if let Some((config, display, view)) = choose {
            let _ = self.load_ocio_view(config, &display, &view);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const CONFIG: &str = "ocio_profile_version: 2\n\
roles:\n  scene_linear: Linear Rec.709\n\
displays:\n  sRGB:\n    - !<View> {name: Standard, colorspace: sRGB}\n    - !<View> {name: Raw, colorspace: Raw}\n\
colorspaces:\n  - !<ColorSpace>\n    name: Linear Rec.709\n  - !<ColorSpace>\n    name: Raw\n    isdata: true\n  - !<ColorSpace>\n    name: sRGB\n    from_scene_reference: !<ExponentWithLinearTransform> {gamma: 2.4, offset: 0.055, direction: inverse}\n";

    #[test]
    fn tc_0393_ocio_views_grade_the_viewport_and_are_remembered() {
        let folder = tempfile::tempdir().unwrap();
        let preferences = folder.path().join("preferences.json");
        let config = folder.path().join("config.ocio");
        std::fs::write(&config, CONFIG.replace("\\n", "\n")).unwrap();
        let mut app = SubstanceCraftApp::default();
        app.load_preferences(preferences.clone());
        app.preferences.values.exposure = 1.0;

        // No view: the preference transform and exposure grade.
        let (transform, exposure, lut) = app.grading();
        assert_eq!((transform, exposure), (ViewTransform::Standard, 1.0));
        assert!(lut.is_none());

        // An unknown view falls back to the display's first.
        app.load_ocio_view(config.clone(), "sRGB", "Nope").unwrap();
        assert_eq!(app.ocio.as_ref().unwrap().view, "Standard");
        let (transform, exposure, lut) = app.grading();
        assert_eq!(transform, ViewTransform::Linear);
        assert_eq!(exposure, 1.0 - OCIO_HEADROOM_STOPS);
        assert_eq!(lut.unwrap().size, OCIO_BAKE_SIZE);

        // Switching views rebakes; the choice comes back next session.
        app.load_ocio_view(config.clone(), "sRGB", "Raw").unwrap();
        let mut later = SubstanceCraftApp::default();
        later.load_preferences(preferences.clone());
        assert_eq!(later.ocio.as_ref().unwrap().view, "Raw");

        // A broken config keeps the current view and says why.
        let broken = folder.path().join("broken.ocio");
        std::fs::write(&broken, "nothing: here").unwrap();
        assert!(later.load_ocio_view(broken, "", "").is_err());
        assert_eq!(later.ocio.as_ref().unwrap().view, "Raw");
        assert!(later.grading_status.is_some());
        later.clear_ocio_view();
        assert!(later.ocio.is_none());
        assert!(
            crate::preferences::Preferences::load(&preferences)
                .ocio
                .is_none()
        );
    }
}
