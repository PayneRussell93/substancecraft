//! The Diagnostics drawer: the app's checks of the open project in one
//! place (project data, mesh, mesh maps, files, renderer, preview, caches,
//! and bakes), each with a severity, and a plain-text report to copy.

use eframe::egui::{self, RichText, Ui};
use substancecraft_api::missing_project_files;

use super::SubstanceCraftApp;
use crate::{theme, widgets};

/// How much a check needs attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Ok,
    Info,
    Warning,
    Error,
}

impl Severity {
    fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }

    fn color(self) -> egui::Color32 {
        match self {
            Self::Ok => egui::Color32::from_rgb(110, 190, 130),
            Self::Info => theme::TEXT_WEAK,
            Self::Warning => egui::Color32::from_rgb(240, 180, 80),
            Self::Error => theme::DANGER,
        }
    }
}

/// One finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Check {
    pub area: &'static str,
    pub severity: Severity,
    pub text: String,
}

impl Check {
    fn new(area: &'static str, severity: Severity, text: impl Into<String>) -> Self {
        Self {
            area,
            severity,
            text: text.into(),
        }
    }
}

impl SubstanceCraftApp {
    /// Every check, in drawer order.
    pub(super) fn diagnostics(&self) -> Vec<Check> {
        let mut checks = Vec::new();
        match self.project.validate() {
            Ok(()) => checks.push(Check::new("Project", Severity::Ok, "Project data is valid")),
            Err(error) => checks.push(Check::new("Project", Severity::Error, error.to_string())),
        }
        checks.push(Check::new(
            "Project",
            Severity::Info,
            format!(
                "{} texture set(s), {} layer(s), {} material(s), {} bitmap(s)",
                self.project.texture_sets.len(),
                self.project
                    .texture_sets
                    .iter()
                    .map(|set| set.walk_layers().len())
                    .sum::<usize>(),
                self.project.materials.len(),
                self.project.resources.len()
            ),
        ));

        match self.mesh_geometry.as_deref() {
            None => checks.push(Check::new("Mesh", Severity::Info, "No mesh loaded")),
            Some(mesh) => {
                checks.push(Check::new(
                    "Mesh",
                    Severity::Ok,
                    format!(
                        "{} triangles, {} vertices, {} material slot(s), {} UDIM tile(s)",
                        mesh.triangles.len(),
                        mesh.vertices.len(),
                        mesh.material_slot_count(),
                        mesh.udim_tiles().len().max(1)
                    ),
                ));
                if self
                    .imported_meshes
                    .iter()
                    .any(|imported| imported.generated_uvs)
                {
                    checks.push(Check::new(
                        "Mesh",
                        Severity::Warning,
                        "The mesh had no UVs; planar UVs were generated, so strokes may stretch",
                    ));
                }
                if mesh.material_slot_count() > 1 {
                    for set in &self.project.texture_sets {
                        if !mesh.material_slots.contains(&set.material_slot) {
                            checks.push(Check::new(
                                "Mesh",
                                Severity::Warning,
                                format!(
                                    "Texture set `{}` names material `{}`, which the mesh lacks",
                                    set.name, set.material_slot
                                ),
                            ));
                        }
                    }
                }
            }
        }
        if self.mesh_file_missing() {
            checks.push(Check::new(
                "Mesh",
                Severity::Error,
                "The mesh file is missing: locate it in the Scene drawer",
            ));
        }

        let needs_maps = self
            .project
            .texture_sets
            .iter()
            .any(|set| set.requires_mesh_maps());
        match (&self.mesh_maps, self.mesh_map_stale_reason()) {
            (Some(maps), None) => {
                let slot = maps.first();
                let detail = slot.map_or_else(String::new, |maps| {
                    let mut detail =
                        format!(", {:.0}% of the UV square covered", maps.coverage() * 100.0);
                    if maps.from_high_poly() {
                        detail.push_str(&format!(
                            ", {:.1}% cage-ray misses",
                            maps.miss_fraction() * 100.0
                        ));
                    }
                    detail
                });
                checks.push(Check::new(
                    "Mesh maps",
                    Severity::Ok,
                    format!(
                        "Baked for {} slot(s) at {}²{detail}",
                        maps.len(),
                        self.mesh_map_settings.resolution
                    ),
                ));
            }
            (Some(_), Some(reason)) => checks.push(Check::new(
                "Mesh maps",
                Severity::Warning,
                format!("Out of date: {reason}"),
            )),
            (None, _) if needs_maps => checks.push(Check::new(
                "Mesh maps",
                Severity::Warning,
                "Not baked yet, but layers need them (generators, projections)",
            )),
            (None, _) => checks.push(Check::new("Mesh maps", Severity::Info, "Not baked")),
        }
        if let Some(error) = &self.mesh_map_error {
            checks.push(Check::new("Mesh maps", Severity::Error, error.clone()));
        }

        let missing = missing_project_files(&self.project, None);
        let unloaded = self
            .project
            .resources
            .iter()
            .filter(|resource| !self.resource_images.contains_key(&resource.id))
            .count();
        if missing.resources.is_empty() && unloaded == 0 && missing.high_poly.is_none() {
            checks.push(Check::new(
                "Files",
                Severity::Ok,
                "Every referenced file loads",
            ));
        } else {
            for (_, path) in &missing.resources {
                checks.push(Check::new(
                    "Files",
                    Severity::Error,
                    format!("Missing bitmap {}", path.display()),
                ));
            }
            if unloaded > missing.resources.len() {
                checks.push(Check::new(
                    "Files",
                    Severity::Warning,
                    format!(
                        "{} bitmap(s) could not be decoded",
                        unloaded - missing.resources.len()
                    ),
                ));
            }
            if let Some(path) = &missing.high_poly {
                checks.push(Check::new(
                    "Files",
                    Severity::Error,
                    format!("Missing high-poly mesh {}", path.display()),
                ));
            }
        }

        match (&self.filament_error, &self.filament_frame_texture) {
            (Some(error), _) => checks.push(Check::new(
                "Renderer",
                Severity::Error,
                format!("Filament failed: {error}"),
            )),
            (None, Some(_)) => checks.push(Check::new(
                "Renderer",
                Severity::Ok,
                "Filament renders the viewport",
            )),
            (None, None) => checks.push(Check::new(
                "Renderer",
                Severity::Info,
                "Software preview (Filament has not rendered a frame)",
            )),
        }
        if let Some(error) = &self.bake_error {
            checks.push(Check::new("Preview", Severity::Error, error.clone()));
        } else if self.preview_failed {
            checks.push(Check::new(
                "Preview",
                Severity::Warning,
                "The last preview bake failed",
            ));
        }

        let usage = self.cache_usage();
        let budget = self.preferences.values.cache_budget_mb as usize * 1024 * 1024;
        checks.push(Check::new(
            "Cache",
            if usage.total() > budget {
                Severity::Warning
            } else {
                Severity::Info
            },
            format!(
                "{:.1} MiB in use of a {} MiB budget",
                usage.total() as f32 / (1024.0 * 1024.0),
                self.preferences.values.cache_budget_mb
            ),
        ));
        if let Some(entry) = self.bake_log.entries().next_back() {
            checks.push(Check::new(
                "Bakes",
                Severity::Info,
                format!("Last: {}", entry.line()),
            ));
        }
        checks
    }

    /// The checks as plain text, one per line.
    pub(super) fn diagnostics_report(&self) -> String {
        self.diagnostics()
            .iter()
            .map(|check| {
                format!(
                    "[{}] {}: {}",
                    check.severity.label(),
                    check.area,
                    check.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub(super) fn diagnostics_drawer(&mut self, ui: &mut Ui) {
        let checks = self.diagnostics();
        let worst = checks
            .iter()
            .map(|check| check.severity)
            .max()
            .unwrap_or(Severity::Ok);
        ui.label(
            RichText::new(match worst {
                Severity::Ok | Severity::Info => "Everything checks out.",
                Severity::Warning => "Some things need attention.",
                Severity::Error => "Something is broken.",
            })
            .color(worst.color()),
        );
        let mut area = "";
        for check in &checks {
            if check.area != area {
                area = check.area;
                widgets::caption(ui, area);
            }
            ui.horizontal(|ui| {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 16.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(rect.center(), 4.0, check.severity.color());
                ui.add(
                    egui::Label::new(RichText::new(&check.text).size(12.0).color(theme::TEXT))
                        .wrap(),
                )
                .on_hover_text(check.severity.label());
            });
        }
        ui.add_space(6.0);
        if widgets::pill_button(ui, None, "Copy report", false, true)
            .on_hover_text("Copy every check as text, for a bug report")
            .clicked()
        {
            ui.ctx().copy_text(self.diagnostics_report());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{MaskGenerator, PrimitivePreset, ProjectResource, ResourceId};

    #[test]
    fn tc_0302_diagnostics_collect_project_mesh_map_file_and_cache_checks() {
        let mut app = SubstanceCraftApp::default();
        let find = |checks: &[Check], area: &str| {
            checks
                .iter()
                .filter(|check| check.area == area)
                .cloned()
                .collect::<Vec<_>>()
        };
        app.set_base_geometry(None);
        let checks = app.diagnostics();
        assert_eq!(find(&checks, "Project")[0].severity, Severity::Ok);
        assert_eq!(find(&checks, "Mesh")[0].text, "No mesh loaded");

        // A mesh and a generator layer without mesh maps: a warning.
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.add_fill_layer();
        if let Some(layer) = app.selected_layer_mut() {
            layer.mask_generator = Some(MaskGenerator::PRESETS[0]);
        }
        let checks = app.diagnostics();
        assert!(find(&checks, "Mesh")[0].text.starts_with("12 triangles"));
        let maps = find(&checks, "Mesh maps");
        assert_eq!(maps[0].severity, Severity::Warning, "{maps:?}");

        // A missing bitmap is an error, and the report has one line per
        // check with its severity.
        app.project.resources.push(ProjectResource::new(
            ResourceId(9),
            "Gone",
            "does/not/exist.png",
        ));
        let checks = app.diagnostics();
        let files = find(&checks, "Files");
        assert!(
            files
                .iter()
                .any(|check| check.severity == Severity::Error && check.text.contains("exist.png"))
        );
        let report = app.diagnostics_report();
        assert_eq!(report.lines().count(), checks.len());
        assert!(report.contains("[error] Files: Missing bitmap"));
        assert!(report.contains("[info] Cache:"));
    }
}
