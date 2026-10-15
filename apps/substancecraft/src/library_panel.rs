//! The Assets drawer's library sections: filter, generator, and
//! environment resources saved in the project, and the shared on-disk
//! library that every project can draw from.

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui::{Align, Layout, RichText, Ui};
use substancecraft_api::{
    SharedLibrary, add_imported_asset, default_shared_library_root, project_asset,
};
use substancecraft_core::{
    AssetEntry, AssetKind, AssetOrigin, DEFAULT_AMBIENT_FILL, EditCommand, EnvironmentResource,
    FilterPreset, GeneratorPreset, MaterialProject,
};

use crate::panels::hint;
use crate::{SubstanceCraftApp, theme, widgets};

/// Library entries listed per kind before "…and N more".
const LISTED_PER_KIND: usize = 12;

/// The shared library as last scanned, and what the last action reported.
#[derive(Default)]
pub struct LibraryState {
    pub shared: Option<SharedLibrary>,
    pub status: Option<String>,
}

/// An action picked in the drawer, run after drawing.
enum LibraryAction {
    ApplyFilter(FilterPreset, bool),
    ApplyGenerator(GeneratorPreset),
    UseEnvironment(EnvironmentResource),
    DeleteFilter(usize),
    DeleteGenerator(usize),
    DeleteEnvironment(usize),
    SaveFilters(bool),
    SaveGenerator,
    SaveEnvironment,
    ToLibrary(AssetKind, String),
    FromLibrary(AssetKind, String),
    RemoveFromLibrary(AssetKind, String),
    ChooseFolder,
    Rescan,
}

impl SubstanceCraftApp {
    /// The folder of the shared library: the preference, else the default.
    pub(super) fn shared_library_root(&self) -> Option<PathBuf> {
        self.preferences
            .values
            .shared_library
            .clone()
            .or_else(default_shared_library_root)
    }

    /// The shared library, scanned on first use.
    pub(super) fn shared_library(&mut self) -> Option<&SharedLibrary> {
        if self.library.shared.is_none() {
            self.library.shared = self.shared_library_root().map(SharedLibrary::open);
        }
        self.library.shared.as_ref()
    }

    /// Points the shared library at `folder` and remembers it.
    pub(super) fn set_shared_library_root(&mut self, folder: PathBuf) {
        self.library.shared = Some(SharedLibrary::open(&folder));
        if self.preferences.values.shared_library.as_ref() != Some(&folder) {
            self.preferences.values.shared_library = Some(folder);
            self.save_preferences();
        }
    }

    fn project_folder(&self) -> Option<PathBuf> {
        self.project_path
            .as_ref()
            .and_then(|path| path.parent())
            .map(PathBuf::from)
    }

    /// Copies the project (or built-in) asset of `kind` named `name` into
    /// the shared library.
    pub(super) fn save_asset_to_library(&mut self, kind: AssetKind, name: &str) -> Result<(), String> {
        let asset = project_asset(&self.project, kind, name)
            .ok_or_else(|| format!("The project has no {} named {name}.", kind.name().to_lowercase()))?;
        let base = self.project_folder();
        self.shared_library();
        let library = self
            .library
            .shared
            .as_mut()
            .ok_or("There is no folder for the shared library.")?;
        library
            .save(asset, &self.project, base.as_deref())
            .map_err(|error| format!("Could not save {name} to the library: {error}"))?;
        self.library.status = Some(format!("Saved {name} to the shared library."));
        Ok(())
    }

    /// Removes the library asset of `kind` named `name`.
    pub(super) fn remove_library_asset(&mut self, kind: AssetKind, name: &str) -> Result<(), String> {
        self.shared_library();
        let library = self
            .library
            .shared
            .as_mut()
            .ok_or("There is no folder for the shared library.")?;
        library
            .remove(kind, name)
            .map_err(|error| format!("Could not remove {name}: {error}"))?;
        self.library.status = Some(format!("Removed {name} from the shared library."));
        Ok(())
    }

    /// Adds the library asset of `kind` named `name` to the project: its
    /// bitmaps become resources and the asset joins the project's list of
    /// its kind (undoably, except for the bitmaps and smart materials,
    /// which the project adds like imports). Returns the name it got.
    pub(super) fn add_library_asset_to_project(
        &mut self,
        kind: AssetKind,
        name: &str,
    ) -> Result<String, String> {
        self.shared_library();
        let library = self
            .library
            .shared
            .as_ref()
            .ok_or("There is no folder for the shared library.")?;
        let import = library
            .import(kind, name, &self.project)
            .map_err(|error| error.to_string())?;
        let mut images = Vec::new();
        for resource in &import.resources {
            let path = PathBuf::from(&resource.source);
            let mut image = substancecraft_api::load_texture_image(&path)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            image.color_space = resource.color_space;
            images.push((resource.id, path, image));
        }
        for (id, path, image) in images {
            self.resource_images.insert(id, Arc::new(image));
            self.resource_watcher.remember(id, &path);
        }
        let first_resource = import.resources.first().map(|resource| resource.name.clone());
        self.project.resources.extend(import.resources);

        let mut after = self.project.clone();
        let added = add_imported_asset(&mut after, import.asset);
        let command = library_list_command(&self.project, &after, kind);
        match command {
            Some(command) => {
                self.history.apply(&mut self.project, command);
            }
            None => self.project = after,
        }
        let added = added.or(first_resource).unwrap_or_else(|| name.to_owned());
        self.library.status = Some(format!("Added {added} to the project."));
        self.invalidate_preview();
        Ok(added)
    }

    /// Appends `preset`'s filters to the selected layer's content filters
    /// or mask filters (recorded as a layer settings edit).
    pub(super) fn apply_filter_preset(&mut self, preset: &FilterPreset, mask: bool) -> bool {
        let Some(layer) = self.selected_layer_mut() else {
            return false;
        };
        let applied = preset.apply_to(layer, mask);
        if applied {
            self.invalidate_preview();
        } else {
            self.library.status = Some("The layer has no room for more filters.".to_owned());
        }
        applied
    }

    /// Makes `preset` the selected layer's mask generator.
    pub(super) fn apply_generator_preset(&mut self, preset: &GeneratorPreset) -> bool {
        let Some(layer) = self.selected_layer_mut() else {
            return false;
        };
        preset.apply_to(layer);
        self.invalidate_preview();
        true
    }

    /// Lights the viewport with `environment`'s panorama at its rotation
    /// and strength.
    pub(super) fn use_environment(&mut self, environment: &EnvironmentResource) -> Result<(), String> {
        self.load_custom_environment(PathBuf::from(&environment.source))?;
        self.viewport_lighting.rotation = environment.rotation_degrees.to_radians();
        self.viewport_lighting.ambient_fill = environment.strength * DEFAULT_AMBIENT_FILL;
        Ok(())
    }

    /// Saves the loaded panorama, with the current rotation and strength,
    /// as a project environment.
    pub(super) fn save_lighting_as_environment(&mut self) -> Option<String> {
        let custom = self.custom_environment.as_ref()?;
        let taken: Vec<String> = self.project.environments.iter().map(|e| e.name.clone()).collect();
        let stem = custom
            .path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Environment".to_owned());
        let name = Self::unique_name(&stem, &taken);
        let mut environment = EnvironmentResource::new(&name, custom.path.to_string_lossy());
        let degrees = self.viewport_lighting.rotation.to_degrees().rem_euclid(360.0);
        environment.rotation_degrees = if degrees > 180.0 { degrees - 360.0 } else { degrees };
        environment.strength = (self.viewport_lighting.ambient_fill / DEFAULT_AMBIENT_FILL)
            .clamp(substancecraft_core::MIN_ENVIRONMENT_STRENGTH, substancecraft_core::MAX_ENVIRONMENT_STRENGTH);
        let before = self.project.environments.clone();
        let mut after = before.clone();
        after.push(environment);
        self.history
            .apply(&mut self.project, EditCommand::Environments { before, after });
        Some(name)
    }

    /// Saves the selected layer's content (or mask) filters as a project
    /// filter preset.
    pub(super) fn save_layer_filters(&mut self, mask: bool) -> Option<String> {
        let layer = self.selected_layer()?;
        let filters = if mask { &layer.mask_filters } else { &layer.filters };
        let taken: Vec<String> = self
            .project
            .filter_presets
            .iter()
            .map(|preset| preset.name.clone())
            .chain(FilterPreset::built_in().into_iter().map(|preset| preset.name))
            .collect();
        let suffix = if mask { "mask filters" } else { "filters" };
        let name = Self::unique_name(&format!("{} {suffix}", layer.name), &taken);
        let preset = FilterPreset::from_filters(filters, &name)?;
        let before = self.project.filter_presets.clone();
        let mut after = before.clone();
        after.push(preset);
        self.history
            .apply(&mut self.project, EditCommand::FilterPresets { before, after });
        Some(name)
    }

    /// Saves the selected layer's generator as a project generator preset.
    pub(super) fn save_layer_generator(&mut self) -> Option<String> {
        let layer = self.selected_layer()?;
        let taken: Vec<String> = self
            .project
            .generator_presets
            .iter()
            .map(|preset| preset.name.clone())
            .chain(GeneratorPreset::built_in().into_iter().map(|preset| preset.name))
            .collect();
        let name = Self::unique_name(&format!("{} generator", layer.name), &taken);
        let preset = GeneratorPreset::from_layer(layer, &name)?;
        let before = self.project.generator_presets.clone();
        let mut after = before.clone();
        after.push(preset);
        self.history
            .apply(&mut self.project, EditCommand::GeneratorPresets { before, after });
        Some(name)
    }

    fn delete_library_item(&mut self, kind: AssetKind, index: usize) {
        let mut after = self.project.clone();
        match kind {
            AssetKind::Filter if index < after.filter_presets.len() => {
                after.filter_presets.remove(index);
            }
            AssetKind::Generator if index < after.generator_presets.len() => {
                after.generator_presets.remove(index);
            }
            AssetKind::Environment if index < after.environments.len() => {
                after.environments.remove(index);
            }
            _ => return,
        }
        if let Some(command) = library_list_command(&self.project, &after, kind) {
            self.history.apply(&mut self.project, command);
        }
    }

    /// Filters, generators, and environments: the project's own and the
    /// built-ins, applied to the selected layer or the viewport.
    pub(super) fn library_resources_section(&mut self, ui: &mut Ui) {
        let mut action = None;
        let has_layer = self.selected_layer().is_some();
        let (has_filters, has_mask_filters, has_generator) = self
            .selected_layer()
            .map(|layer| {
                (
                    !layer.filters.is_empty(),
                    !layer.mask_filters.is_empty(),
                    layer.mask_generator.is_some(),
                )
            })
            .unwrap_or_default();

        widgets::caption(ui, "Filters");
        let project_filters = self.project.filter_presets.clone();
        let rows = project_filters
            .iter()
            .enumerate()
            .map(|(index, preset)| (Some(index), preset.clone()))
            .chain(FilterPreset::built_in().into_iter().map(|preset| (None, preset)));
        for (index, preset) in rows {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&preset.name).size(12.0).color(theme::TEXT))
                    .on_hover_text(&preset.description);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if let Some(index) = index
                        && widgets::chip(ui, "Delete", false).clicked()
                    {
                        action = Some(LibraryAction::DeleteFilter(index));
                    }
                    ui.add_enabled_ui(has_layer, |ui| {
                        if widgets::chip(ui, "Mask", false)
                            .on_hover_text("Append to the selected layer's mask filters")
                            .clicked()
                        {
                            action = Some(LibraryAction::ApplyFilter(preset.clone(), true));
                        }
                        if widgets::chip(ui, "Layer", false)
                            .on_hover_text("Append to the selected layer's filters")
                            .clicked()
                        {
                            action = Some(LibraryAction::ApplyFilter(preset.clone(), false));
                        }
                    });
                });
            });
        }
        ui.horizontal(|ui| {
            ui.add_enabled_ui(has_filters, |ui| {
                if widgets::chip(ui, "Save layer filters", false).clicked() {
                    action = Some(LibraryAction::SaveFilters(false));
                }
            });
            ui.add_enabled_ui(has_mask_filters, |ui| {
                if widgets::chip(ui, "Save mask filters", false).clicked() {
                    action = Some(LibraryAction::SaveFilters(true));
                }
            });
        });

        widgets::caption(ui, "Generators");
        let project_generators = self.project.generator_presets.clone();
        let rows = project_generators
            .iter()
            .enumerate()
            .map(|(index, preset)| (Some(index), preset.clone()))
            .chain(GeneratorPreset::built_in().into_iter().map(|preset| (None, preset)));
        for (index, preset) in rows {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&preset.name).size(12.0).color(theme::TEXT))
                    .on_hover_text(format!("{} ({})", preset.description, preset.generator.name()));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if let Some(index) = index
                        && widgets::chip(ui, "Delete", false).clicked()
                    {
                        action = Some(LibraryAction::DeleteGenerator(index));
                    }
                    ui.add_enabled_ui(has_layer, |ui| {
                        if widgets::chip(ui, "Apply", false)
                            .on_hover_text("Make this the selected layer's mask generator")
                            .clicked()
                        {
                            action = Some(LibraryAction::ApplyGenerator(preset.clone()));
                        }
                    });
                });
            });
        }
        ui.add_enabled_ui(has_generator, |ui| {
            if widgets::chip(ui, "Save layer generator", false).clicked() {
                action = Some(LibraryAction::SaveGenerator);
            }
        });

        widgets::caption(ui, "Environments");
        for (index, environment) in self.project.environments.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&environment.name).size(12.0).color(theme::TEXT))
                    .on_hover_text(format!(
                        "{} · {:.0}° · {:.2}×",
                        environment.source, environment.rotation_degrees, environment.strength
                    ));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if widgets::chip(ui, "Delete", false).clicked() {
                        action = Some(LibraryAction::DeleteEnvironment(index));
                    }
                    if widgets::chip(ui, "Light", false)
                        .on_hover_text("Light the viewport with this environment")
                        .clicked()
                    {
                        action = Some(LibraryAction::UseEnvironment(environment.clone()));
                    }
                });
            });
        }
        ui.add_enabled_ui(self.custom_environment.is_some(), |ui| {
            if widgets::chip(ui, "Save current lighting", false)
                .on_hover_text("Keep the loaded HDRI with its rotation and strength")
                .clicked()
            {
                action = Some(LibraryAction::SaveEnvironment);
            }
        });
        if self.project.environments.is_empty() && self.custom_environment.is_none() {
            hint(ui, "Load an HDRI in the Light drawer to save it as an environment.");
        }
        if let Some(action) = action {
            self.run_library_action(action);
        }
    }

    /// The shared library: its folder, and every asset in it with actions.
    pub(super) fn shared_library_section(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Shared library");
        let mut action = None;
        let root = self.shared_library_root();
        let entries: Vec<AssetEntry> = self
            .shared_library()
            .map(SharedLibrary::asset_entries)
            .unwrap_or_default();
        let problems: Vec<String> = self
            .library
            .shared
            .as_ref()
            .map(|library| library.problems().to_vec())
            .unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(root.as_deref().map_or_else(
                    || "No folder".to_owned(),
                    |root| root.display().to_string(),
                ))
                .size(10.5)
                .color(theme::TEXT_FAINT),
            );
        });
        ui.horizontal(|ui| {
            if widgets::chip(ui, "Folder…", false)
                .on_hover_text("Use another folder, such as one shared with your team")
                .clicked()
            {
                action = Some(LibraryAction::ChooseFolder);
            }
            if widgets::chip(ui, "Rescan", false)
                .on_hover_text("Read the folder again for assets added elsewhere")
                .clicked()
            {
                action = Some(LibraryAction::Rescan);
            }
        });
        if entries.is_empty() {
            hint(
                ui,
                "Assets saved here are available to every project. Use \"To library\" on a \
                 search result to add one.",
            );
        }
        for kind in AssetKind::ALL {
            let of_kind: Vec<&AssetEntry> =
                entries.iter().filter(|entry| entry.kind == kind).collect();
            if of_kind.is_empty() {
                continue;
            }
            ui.label(RichText::new(kind.name()).size(11.0).color(theme::TEXT_WEAK));
            for entry in of_kind.iter().take(LISTED_PER_KIND) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&entry.name).size(12.0).color(theme::TEXT));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if widgets::chip(ui, "Remove", false)
                            .on_hover_text("Delete from the shared library")
                            .clicked()
                        {
                            action = Some(LibraryAction::RemoveFromLibrary(kind, entry.name.clone()));
                        }
                        if widgets::chip(ui, "Add", false)
                            .on_hover_text("Add to this project")
                            .clicked()
                        {
                            action = Some(LibraryAction::FromLibrary(kind, entry.name.clone()));
                        }
                    });
                });
            }
            if of_kind.len() > LISTED_PER_KIND {
                hint(
                    ui,
                    &format!("…and {} more; search to find them", of_kind.len() - LISTED_PER_KIND),
                );
            }
        }
        for problem in problems.iter().take(4) {
            ui.label(RichText::new(problem).size(10.5).color(crate::panels::WARNING));
        }
        if let Some(status) = &self.library.status {
            hint(ui, status);
        }
        if let Some(action) = action {
            self.run_library_action(action);
        }
    }

    /// Search results across the project, the shared library, and the
    /// built-ins, with an action to move each between project and library.
    pub(super) fn search_all_assets(&mut self, query: &str) -> Vec<AssetEntry> {
        let mut results = self.project.search_assets(query, None);
        if let Some(library) = self.shared_library() {
            results.extend(
                library
                    .asset_entries()
                    .into_iter()
                    .filter(|entry| entry.matches(query)),
            );
        }
        results
    }

    /// The action chip of a search result: "To library" for project and
    /// built-in assets, "Add" for library ones.
    pub(super) fn asset_result_action(&mut self, ui: &mut Ui, asset: &AssetEntry) {
        let action = match asset.origin {
            AssetOrigin::Shared => widgets::chip(ui, "Add", false)
                .on_hover_text("Add to this project")
                .clicked()
                .then(|| LibraryAction::FromLibrary(asset.kind, asset.name.clone())),
            AssetOrigin::Project | AssetOrigin::BuiltIn => widgets::chip(ui, "To library", false)
                .on_hover_text("Save to the shared library for other projects")
                .clicked()
                .then(|| LibraryAction::ToLibrary(asset.kind, asset.name.clone())),
        };
        if let Some(action) = action {
            self.run_library_action(action);
        }
    }

    fn run_library_action(&mut self, action: LibraryAction) {
        let result = match action {
            LibraryAction::ApplyFilter(preset, mask) => {
                self.apply_filter_preset(&preset, mask);
                Ok(())
            }
            LibraryAction::ApplyGenerator(preset) => {
                self.apply_generator_preset(&preset);
                Ok(())
            }
            LibraryAction::UseEnvironment(environment) => self.use_environment(&environment),
            LibraryAction::DeleteFilter(index) => {
                self.delete_library_item(AssetKind::Filter, index);
                Ok(())
            }
            LibraryAction::DeleteGenerator(index) => {
                self.delete_library_item(AssetKind::Generator, index);
                Ok(())
            }
            LibraryAction::DeleteEnvironment(index) => {
                self.delete_library_item(AssetKind::Environment, index);
                Ok(())
            }
            LibraryAction::SaveFilters(mask) => {
                self.save_layer_filters(mask);
                Ok(())
            }
            LibraryAction::SaveGenerator => {
                self.save_layer_generator();
                Ok(())
            }
            LibraryAction::SaveEnvironment => {
                self.save_lighting_as_environment();
                Ok(())
            }
            LibraryAction::ToLibrary(kind, name) => self.save_asset_to_library(kind, &name),
            LibraryAction::FromLibrary(kind, name) => {
                self.add_library_asset_to_project(kind, &name).map(|_| ())
            }
            LibraryAction::RemoveFromLibrary(kind, name) => self.remove_library_asset(kind, &name),
            LibraryAction::ChooseFolder => {
                let mut dialog = rfd::FileDialog::new();
                if let Some(root) = self.shared_library_root() {
                    dialog = dialog.set_directory(root);
                }
                if let Some(folder) = dialog.pick_folder() {
                    self.set_shared_library_root(folder);
                }
                Ok(())
            }
            LibraryAction::Rescan => {
                if let Some(library) = self.library.shared.as_mut() {
                    library.rescan();
                } else {
                    self.shared_library();
                }
                Ok(())
            }
        };
        if let Err(error) = result {
            self.library.status = Some(error);
        }
    }
}

/// The undoable edit turning `before`'s list of `kind` into `after`'s, or
/// `None` for kinds the history does not track.
fn library_list_command(
    before: &MaterialProject,
    after: &MaterialProject,
    kind: AssetKind,
) -> Option<EditCommand> {
    match kind {
        AssetKind::Material => Some(EditCommand::Materials {
            before: before.materials.clone(),
            after: after.materials.clone(),
        }),
        AssetKind::SmartMask => Some(EditCommand::SmartMasks {
            before: before.smart_masks.clone(),
            after: after.smart_masks.clone(),
        }),
        AssetKind::Filter => Some(EditCommand::FilterPresets {
            before: before.filter_presets.clone(),
            after: after.filter_presets.clone(),
        }),
        AssetKind::Generator => Some(EditCommand::GeneratorPresets {
            before: before.generator_presets.clone(),
            after: after.generator_presets.clone(),
        }),
        AssetKind::Environment => Some(EditCommand::Environments {
            before: before.environments.clone(),
            after: after.environments.clone(),
        }),
        AssetKind::ExportPreset => Some(EditCommand::ExportPresets {
            before: before.export_presets.clone(),
            after: after.export_presets.clone(),
        }),
        AssetKind::Bitmap | AssetKind::SmartMaterial => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{LayerFilter, MaskGenerator};

    #[test]
    fn tc_0388_library_resources_and_the_shared_library_work_from_the_assets_drawer() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.set_shared_library_root(folder.path().join("library"));
        app.add_paint_layer();

        // A built-in filter and generator apply to the selected layer and
        // undo as layer edits.
        let hard = FilterPreset::built_in()
            .into_iter()
            .find(|preset| preset.name == "Hard edge")
            .unwrap();
        assert!(app.apply_filter_preset(&hard, true));
        let grime = GeneratorPreset::built_in().remove(1);
        assert!(app.apply_generator_preset(&grime));
        let layer = app.selected_layer().unwrap();
        assert_eq!(layer.mask_filters, hard.filters);
        assert_eq!(layer.mask_generator, Some(grime.generator));

        // The layer's setup saves back as project presets.
        let filters = app.save_layer_filters(true).unwrap();
        let generator = app.save_layer_generator().unwrap();
        assert_eq!(app.project.filter_presets[0].filters, hard.filters);
        assert!(matches!(
            app.project.generator_presets[0].generator,
            MaskGenerator::Dirt(_)
        ));

        // Project assets go to the shared library and come back into
        // another project with fresh names.
        app.save_asset_to_library(AssetKind::Filter, &filters).unwrap();
        app.save_asset_to_library(AssetKind::Generator, &generator).unwrap();
        app.save_asset_to_library(AssetKind::Filter, "Soften").unwrap();
        let found = app.search_all_assets("library soften");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].origin, AssetOrigin::Shared);

        let mut other = SubstanceCraftApp::default();
        other.set_shared_library_root(folder.path().join("library"));
        let added = other
            .add_library_asset_to_project(AssetKind::Filter, &filters)
            .unwrap();
        assert_eq!(added, filters);
        assert_eq!(other.project.filter_presets[0].filters, hard.filters);
        // Adding again keeps both under distinct names, and undo removes it.
        let again = other
            .add_library_asset_to_project(AssetKind::Filter, &filters)
            .unwrap();
        assert_ne!(again, filters);
        assert!(other.history.undo(&mut other.project));
        assert_eq!(other.project.filter_presets.len(), 1);

        // Removing from the library hides it from every project's search.
        other.remove_library_asset(AssetKind::Filter, "Soften").unwrap();
        other.library.shared = None;
        assert!(other.search_all_assets("soften library").is_empty());

        // A layer with too many filters refuses more and says so.
        if let Some(layer) = app.selected_layer_mut() {
            layer.filters = vec![LayerFilter::Invert; 32];
        }
        assert!(!app.apply_filter_preset(&hard, false));
        assert!(app.library.status.is_some());
    }
}
