//! Colour management, light rigs, and shaders of the viewport and renders.
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
    CubeLut, LightRig, MAX_RIG_LIGHTS, MAX_RIG_LUX, MAX_SHADER_FILE_BYTES, RigLight,
    ShaderDefinition, SunLight, ViewTransform,
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
                    slider_row(
                        ui,
                        "Height°",
                        &mut light.elevation_degrees,
                        -90.0..=90.0,
                        false,
                        0,
                    );
                    slider_row(
                        ui,
                        "Around°",
                        &mut light.azimuth_degrees,
                        -180.0..=180.0,
                        false,
                        0,
                    );
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
                if widgets::chip(ui, transform.name(), values.view_transform == transform).clicked()
                {
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
        self.ocio_section(ui);
        if let Some(status) = &self.grading_status {
            hint(ui, status);
        }
        if pick {
            self.pick_display_lut();
        } else if clear {
            self.clear_display_lut();
        }
    }

    /// The folder user shader definitions are read from: `shaders` beside
    /// the preferences file.
    pub(super) fn shader_folder(&self) -> Option<PathBuf> {
        let preferences = self
            .preferences
            .path
            .clone()
            .or_else(crate::preferences::Preferences::default_path)?;
        Some(preferences.parent()?.join("shaders"))
    }

    /// Re-reads the user shader definitions (`*.json` in the shader folder),
    /// skipping, and reporting, files that are not valid definitions.
    pub(super) fn refresh_user_shaders(&mut self) {
        self.user_shaders.clear();
        let Some(folder) = self.shader_folder() else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(&folder) else {
            return;
        };
        let mut paths: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "json")
            })
            .collect();
        paths.sort();
        let mut failures = Vec::new();
        for path in paths {
            match load_shader_file(&path) {
                Ok(shader) => self.user_shaders.push(shader),
                Err(error) => failures.push(format!("{}: {error}", crate::path_label(&path))),
            }
        }
        self.shader_status = failures.first().map(|first| match failures.len() {
            1 => format!("Skipped {first}"),
            count => format!("Skipped {first} (and {} more)", count - 1),
        });
    }

    /// The shader the viewport uses: the project's, else the default.
    pub(super) fn viewport_shader(&self) -> ShaderDefinition {
        self.project.shader.clone().unwrap_or_default()
    }

    /// Sets the project's viewport shader; the default is stored as none.
    pub(super) fn set_viewport_shader(&mut self, shader: ShaderDefinition) {
        if !shader.is_valid() {
            return;
        }
        self.project.shader = (shader != ShaderDefinition::default()).then_some(shader);
    }

    /// The Shader section of the Light drawer.
    pub(super) fn shader_section(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Shader");
        let current = self.viewport_shader();
        let mut chosen = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for shader in ShaderDefinition::built_in()
                .into_iter()
                .chain(self.user_shaders.iter().cloned())
            {
                if widgets::chip(ui, &shader.name, shader == current)
                    .on_hover_text(&shader.description)
                    .clicked()
                {
                    chosen = Some(shader);
                }
            }
        });
        if !current.description.is_empty() {
            hint(ui, &current.description);
        }
        let (mut load, mut refresh) = (false, false);
        ui.horizontal(|ui| {
            load = widgets::pill_button(ui, Some(Icon::Import), "Load…", false, false)
                .on_hover_text("Use a shader definition (.json) for this project")
                .clicked();
            refresh = widgets::pill_button(ui, None, "Rescan", false, false)
                .on_hover_text(
                    self.shader_folder()
                        .map_or("No shader folder".to_owned(), |folder| {
                            format!("List the .json definitions in {}", folder.display())
                        }),
                )
                .clicked();
        });
        if let Some(status) = &self.shader_status {
            hint(ui, status);
        }
        if let Some(shader) = chosen {
            self.set_viewport_shader(shader);
        }
        if refresh {
            self.refresh_user_shaders();
        }
        if load
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Shader definition", &["json"])
                .pick_file()
        {
            match load_shader_file(&path) {
                Ok(shader) => {
                    self.shader_status = None;
                    self.set_viewport_shader(shader);
                }
                Err(error) => {
                    self.shader_status = Some(format!(
                        "Could not load {}: {error}",
                        crate::path_label(&path)
                    ));
                }
            }
        }
    }
}

/// Reads a shader definition file; missing fields take their defaults.
pub fn load_shader_file(path: &Path) -> Result<ShaderDefinition, String> {
    let size = std::fs::metadata(path)
        .map_err(|error| error.to_string())?
        .len();
    if size > MAX_SHADER_FILE_BYTES as u64 {
        return Err("the file is too large for a shader definition".to_owned());
    }
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let shader: ShaderDefinition =
        serde_json::from_str(&text).map_err(|error| error.to_string())?;
    if shader.is_valid() {
        Ok(shader)
    } else {
        Err("a value is out of range or the name is empty".to_owned())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::PrimitivePreset;

    #[test]
    fn tc_0382_display_luts_and_light_rigs_are_kept_and_restored() {
        let folder = tempfile::tempdir().unwrap();
        let preferences = folder.path().join("preferences.json");
        let mut app = SubstanceCraftApp::default();
        app.load_preferences(preferences.clone());

        // A LUT loads, is remembered, and comes back next session.
        let mut text = String::from("LUT_3D_SIZE 2\n");
        for b in 0..2 {
            for g in 0..2 {
                for r in 0..2 {
                    text.push_str(&format!("{r} {g} {b}\n"));
                }
            }
        }
        let lut_path = folder.path().join("look.cube");
        std::fs::write(&lut_path, text).unwrap();
        app.load_display_lut(lut_path.clone()).unwrap();
        assert_eq!(app.display_lut.as_ref().unwrap().lut.size, 2);
        app.preferences.values.view_transform = ViewTransform::Agx;
        app.preferences.values.exposure = 1.5;
        app.save_preferences();
        let mut later = SubstanceCraftApp::default();
        later.load_preferences(preferences.clone());
        assert!(later.display_lut.is_some());
        assert_eq!(later.preferences.values.view_transform, ViewTransform::Agx);
        assert_eq!(later.preferences.values.exposure, 1.5);

        // A broken file keeps the current LUT and says why; clearing forgets it.
        let broken = folder.path().join("broken.cube");
        std::fs::write(&broken, "LUT_3D_SIZE 2\n0 0 0\n").unwrap();
        assert!(later.load_display_lut(broken).is_err());
        assert_eq!(later.display_lut.as_ref().unwrap().path, lut_path);
        assert!(later.grading_status.is_some());
        later.clear_display_lut();
        assert!(later.display_lut.is_none());
        assert_eq!(
            crate::preferences::Preferences::load(&preferences).display_lut,
            None
        );

        // A rig is saved with the project and reopened with it; invalid
        // rigs are refused; rig lights turn with the environment.
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        let rig = LightRig::built_in().remove(0);
        app.set_light_rig(Some(rig.clone()));
        let mut crowded = rig.clone();
        crowded.lights.extend(rig.lights.clone());
        app.set_light_rig(Some(crowded));
        assert_eq!(app.project.light_rig.as_ref(), Some(&rig));
        assert_eq!(app.rig_suns(0.0).len(), 3);
        assert_ne!(
            app.rig_suns(1.0)[0].direction,
            app.rig_suns(0.0)[0].direction
        );
        let project_path = folder.path().join("lit.sproject");
        app.project_path = Some(project_path.clone());
        app.save_project();
        app.finish_saves();
        let mut reopened = SubstanceCraftApp::default();
        reopened.open_project_path(project_path);
        assert_eq!(reopened.project.light_rig, Some(rig));
        reopened.set_light_rig(None);
        assert!(reopened.rig_suns(0.0).is_empty());
    }

    #[test]
    fn tc_0385_shader_definitions_are_listed_swapped_and_saved_with_the_project() {
        let folder = tempfile::tempdir().unwrap();
        let shaders = folder.path().join("shaders");
        std::fs::create_dir(&shaders).unwrap();
        std::fs::write(
            shaders.join("wet.json"),
            r#"{ "name": "Wet", "description": "Rain-soaked", "clear_coat": 0.9 }"#,
        )
        .unwrap();
        std::fs::write(
            shaders.join("broken.json"),
            r#"{ "name": "", "unlit": true }"#,
        )
        .unwrap();
        std::fs::write(shaders.join("notes.txt"), "not a shader").unwrap();

        // The folder beside the preferences lists valid definitions and
        // says which files it skipped.
        let mut app = SubstanceCraftApp::default();
        app.load_preferences(folder.path().join("preferences.json"));
        assert_eq!(app.shader_folder(), Some(shaders.clone()));
        assert_eq!(app.user_shaders.len(), 1);
        assert_eq!(app.user_shaders[0].name, "Wet");
        assert!(
            app.shader_status
                .as_deref()
                .unwrap()
                .contains("broken.json")
        );

        // The default is stored as none; others are saved in the project.
        assert_eq!(app.viewport_shader(), ShaderDefinition::default());
        let wet = app.user_shaders[0].clone();
        app.set_viewport_shader(wet.clone());
        assert_eq!(app.viewport_shader(), wet);
        let mut invalid = wet.clone();
        invalid.emissive_scale = 100.0;
        app.set_viewport_shader(invalid);
        assert_eq!(
            app.viewport_shader(),
            wet,
            "invalid definitions are refused"
        );
        let project = folder.path().join("wet.sproject");
        app.set_base_geometry(Some(
            substancecraft_core::PrimitivePreset::Cube.build_mesh(),
        ));
        app.project_path = Some(project.clone());
        app.save_project();
        app.finish_saves();
        let mut reopened = SubstanceCraftApp::default();
        reopened.open_project_path(project);
        assert_eq!(reopened.viewport_shader(), wet);
        reopened.set_viewport_shader(ShaderDefinition::default());
        assert_eq!(reopened.project.shader, None);

        // Files load on their own too.
        assert_eq!(load_shader_file(&shaders.join("wet.json")).unwrap(), wet);
        assert!(load_shader_file(&shaders.join("broken.json")).is_err());
    }
}
