//! Colour management and light rigs of the viewport and renders.
//!
//! Colour: a view transform (tone curve), an exposure adjustment, and an
//! optional display LUT from a `.cube` file, which is how colour pipelines
//! hand over display transforms and looks (OpenColorIO's `ociobakelut`
//! bakes any view to one). The LUT takes the display-referred sRGB image.
//! All three are user preferences, like the lighting preset.
//!
//! Light rigs: up to four directional lights added to the environment,
//! saved with the project so every view and render of it is lit alike. They
//! turn with the environment's rotation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui::{self, Id, RichText, Ui};
use substancecraft_core::{
    CubeLut, LightRig, MAX_RIG_LIGHTS, MAX_RIG_LUX, RigLight, SunLight, ViewTransform,
};

use super::SubstanceCraftApp;
use crate::icons::Icon;
use crate::panels::{hint, slider_row};
use crate::preferences::MAX_EXPOSURE_STOPS;
use crate::{theme, widgets};

/// Largest `.cube` file read.
const MAX_LUT_BYTES: u64 = 64 * 1024 * 1024;

/// A display LUT loaded from disk.
pub struct DisplayLut {
    pub path: PathBuf,
    pub lut: Arc<CubeLut>,
}

/// Reads and parses a `.cube` file.
pub fn load_cube_file(path: &Path) -> Result<CubeLut, String> {
    let size = std::fs::metadata(path)
        .map_err(|error| error.to_string())?
        .len();
    if size > MAX_LUT_BYTES {
        return Err("the file is too large for a LUT".to_owned());
    }
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    CubeLut::parse(&text)
}

impl SubstanceCraftApp {
    /// Applies the display LUT at `path` and remembers it for later
    /// sessions. A file that cannot be read leaves the current one.
    pub(super) fn load_display_lut(&mut self, path: PathBuf) -> Result<(), String> {
        match load_cube_file(&path) {
            Ok(lut) => {
                self.display_lut = Some(DisplayLut {
                    path: path.clone(),
                    lut: Arc::new(lut),
                });
                self.grading_status = None;
                if self.preferences.values.display_lut.as_ref() != Some(&path) {
                    self.preferences.values.display_lut = Some(path);
                    self.save_preferences();
                }
                Ok(())
            }
            Err(error) => {
                let message = format!("Could not load {}: {error}", crate::path_label(&path));
                self.grading_status = Some(message.clone());
                Err(message)
            }
        }
    }

    pub(super) fn clear_display_lut(&mut self) {
        self.display_lut = None;
        self.grading_status = None;
        if self.preferences.values.display_lut.take().is_some() {
            self.save_preferences();
        }
    }

    /// Restores the display LUT saved in preferences, if any; a missing
    /// file stays remembered and is reported.
    pub(super) fn restore_display_lut(&mut self) {
        if let Some(path) = self.preferences.values.display_lut.clone() {
            let _ = self.load_display_lut(path);
        }
    }

    fn pick_display_lut(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("3D LUT (.cube)", &["cube"])
            .pick_file()
        {
            let _ = self.load_display_lut(path);
        }
    }

    /// The project's rig lights as the renderer takes them, turned by the
    /// environment rotation `rotation`.
    pub(super) fn rig_suns(&self, rotation: f32) -> Vec<SunLight> {
        self.project
            .light_rig
            .iter()
            .flat_map(|rig| &rig.lights)
            .map(|light| light.sun(rotation))
            .collect()
    }

    /// Replaces the project's light rig (none removes it). Invalid rigs are
    /// refused.
    pub(super) fn set_light_rig(&mut self, rig: Option<LightRig>) {
        if rig.as_ref().is_some_and(|rig| !rig.is_valid()) || rig == self.project.light_rig {
            return;
        }
        self.project.light_rig = rig;
    }

    /// The Rig section of the Light drawer.
    pub(super) fn light_rig_section(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Light rig");
        let mut rig = self.project.light_rig.clone();
        widgets::property_row(ui, "Rig", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if widgets::chip(ui, "None", rig.is_none()).clicked() {
                rig = None;
            }
            for template in LightRig::built_in() {
                let current = rig.as_ref().is_some_and(|rig| rig.name == template.name);
                if widgets::chip(ui, &template.name, current)
                    .on_hover_text("Start from this arrangement; adjust its lights below")
                    .clicked()
                {
                    rig = Some(template);
                }
            }
        });
        if let Some(current) = rig.as_mut() {
            let mut remove = None;
            for (index, light) in current.lights.iter_mut().enumerate() {
                ui.push_id(Id::new(("rig_light", index)), |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("Light {}", index + 1))
                                .size(12.0)
                                .color(theme::TEXT_WEAK),
                        );
                        let mut color = egui::Color32::from_rgb(
                            (light.color[0] * 255.0).round() as u8,
                            (light.color[1] * 255.0).round() as u8,
                            (light.color[2] * 255.0).round() as u8,
                        );
                        if widgets::labeled_color_button(ui, &mut color).changed() {
                            light.color = [color.r(), color.g(), color.b()]
                                .map(|channel| f32::from(channel) / 255.0);
                        }
                        if widgets::icon_button(ui, Icon::Trash, true, "Remove this light")
                            .clicked()
                        {
                            remove = Some(index);
                        }
                    });
                    slider_row(ui, "Height°", &mut light.elevation_degrees, -90.0..=90.0, false, 0);
                    slider_row(ui, "Around°", &mut light.azimuth_degrees, -180.0..=180.0, false, 0);
                    slider_row(ui, "Lux", &mut light.lux, 0.0..=MAX_RIG_LUX, true, 0);
                });
            }
            if let Some(index) = remove {
                current.lights.remove(index);
            }
            if current.lights.len() < MAX_RIG_LIGHTS
                && widgets::pill_button(ui, Some(Icon::Plus), "Add light", false, false).clicked()
            {
                current.lights.push(RigLight {
                    elevation_degrees: 30.0,
                    azimuth_degrees: 0.0,
                    color: [1.0, 1.0, 1.0],
                    lux: 30_000.0,
                });
            }
            hint(
                ui,
                "Rig lights shine alongside the environment, turn with its rotation, and \
                 are saved with the project.",
            );
        }
        self.set_light_rig(rig);
    }

    /// The Colour section of the Light drawer: view transform, exposure,
    /// and display LUT.
    pub(super) fn color_section(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Colour");
        let before = (
            self.preferences.values.view_transform,
            self.preferences.values.exposure,
        );
        let values = &mut self.preferences.values;
        widgets::property_row(ui, "View", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for transform in ViewTransform::ALL {
                if widgets::chip(ui, transform.name(), values.view_transform == transform).clicked() {
                    values.view_transform = transform;
                }
            }
        });
        slider_row(
            ui,
            "Exposure",
            &mut values.exposure,
            -MAX_EXPOSURE_STOPS..=MAX_EXPOSURE_STOPS,
            false,
            1,
        );
        if (values.view_transform, values.exposure) != before {
            self.save_preferences();
        }
        let (mut pick, mut clear) = (false, false);
        widgets::property_row(ui, "Display LUT", |ui| {
            let label = self
                .display_lut
                .as_ref()
                .map_or("None".to_owned(), |lut| crate::path_label(&lut.path));
            ui.label(RichText::new(label).size(11.5).color(theme::TEXT_WEAK));
            pick = widgets::pill_button(ui, Some(Icon::Import), "Load…", false, false)
                .on_hover_text(
                    "A .cube 3D LUT applied to the display image, e.g. a look or a view \
                     baked from an OpenColorIO config with ociobakelut",
                )
                .clicked();
            clear = self.display_lut.is_some()
                && widgets::pill_button(ui, Some(Icon::Close), "Remove", false, false).clicked();
        });
        if let Some(status) = &self.grading_status {
            hint(ui, status);
        }
        if pick {
            self.pick_display_lut();
        } else if clear {
            self.clear_display_lut();
        }
    }
}
