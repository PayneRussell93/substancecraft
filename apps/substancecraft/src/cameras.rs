//! Named inspection cameras: saved in the project, recalled into the
//! viewport, and used by still renders.

use eframe::egui::{self, Id, RichText, Ui};
use substancecraft_core::{MAX_NAMED_CAMERAS, NamedCamera};

use super::SubstanceCraftApp;
use crate::icons::Icon;
use crate::{theme, widgets};

impl SubstanceCraftApp {
    /// Saves the viewport camera under the next free "View N" name.
    pub(super) fn save_named_camera(&mut self) -> Option<usize> {
        if self.project.cameras.len() >= MAX_NAMED_CAMERAS {
            self.project_error = Some(format!("A project keeps at most {MAX_NAMED_CAMERAS} views"));
            return None;
        }
        let taken: Vec<String> = self
            .project
            .cameras
            .iter()
            .map(|camera| camera.name.clone())
            .collect();
        let name = Self::unique_name(&format!("View {}", taken.len() + 1), &taken);
        self.project.cameras.push(NamedCamera {
            name,
            camera: self.viewport_camera,
        });
        Some(self.project.cameras.len() - 1)
    }

    /// Moves the viewport to a saved camera.
    pub(super) fn recall_named_camera(&mut self, index: usize) {
        if let Some(saved) = self.project.cameras.get(index) {
            self.viewport_camera = saved.camera;
        }
    }

    /// Replaces a saved camera with the current viewport camera.
    pub(super) fn update_named_camera(&mut self, index: usize) {
        if let Some(saved) = self.project.cameras.get_mut(index) {
            saved.camera = self.viewport_camera;
        }
    }

    pub(super) fn delete_named_camera(&mut self, index: usize) {
        if index < self.project.cameras.len() {
            self.project.cameras.remove(index);
        }
    }

    /// The "Saved views" list of the Light drawer.
    pub(super) fn saved_views_section(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Saved views");
        enum Action {
            Recall(usize),
            Update(usize),
            Render(usize),
            Delete(usize),
        }
        let mut action = None;
        for (index, saved) in self.project.cameras.iter().enumerate() {
            let current = saved.camera == self.viewport_camera;
            widgets::list_row(ui, Id::new(("saved_view", index)), current, |ui| {
                if ui
                    .add(
                        egui::Label::new(RichText::new(&saved.name).size(12.5).color(theme::TEXT))
                            .sense(egui::Sense::click()),
                    )
                    .on_hover_text("Move the viewport to this view")
                    .clicked()
                {
                    action = Some(Action::Recall(index));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if widgets::icon_button(ui, Icon::Trash, true, "Delete this view").clicked() {
                        action = Some(Action::Delete(index));
                    }
                    if widgets::icon_button(ui, Icon::Image, true, "Render this view at 1K")
                        .clicked()
                    {
                        action = Some(Action::Render(index));
                    }
                    if widgets::icon_button(
                        ui,
                        Icon::Save,
                        !current,
                        "Replace with the current view",
                    )
                    .clicked()
                    {
                        action = Some(Action::Update(index));
                    }
                    if widgets::icon_button(ui, Icon::Focus, true, "Go to this view").clicked() {
                        action = Some(Action::Recall(index));
                    }
                });
            });
        }
        if widgets::pill_button(ui, Some(Icon::Plus), "Save current view", false, true)
            .on_hover_text("Keep this camera in the project to return to and render from")
            .clicked()
        {
            self.save_named_camera();
        }
        match action {
            Some(Action::Recall(index)) => self.recall_named_camera(index),
            Some(Action::Update(index)) => self.update_named_camera(index),
            Some(Action::Render(index)) => {
                let camera = self.project.cameras.get(index).map(|saved| saved.camera);
                self.request_render(1024, camera);
            }
            Some(Action::Delete(index)) => self.delete_named_camera(index),
            None => {}
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::ViewportCamera;

    #[test]
    fn tc_0238_named_cameras_save_recall_and_drive_renders() {
        let mut app = SubstanceCraftApp::default();
        app.viewport_camera.orbit(glam::Vec2::new(40.0, -20.0));
        let front = app.viewport_camera;
        assert_eq!(app.save_named_camera(), Some(0));
        app.viewport_camera.zoom_by(400.0);
        assert_eq!(app.save_named_camera(), Some(1));
        assert_eq!(app.project.cameras[1].name, "View 2");
        assert_eq!(app.project.validate(), Ok(()));

        app.viewport_camera = ViewportCamera::default();
        app.recall_named_camera(0);
        assert_eq!(app.viewport_camera, front);

        // Saved views persist with the project.
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("views.sproject");
        app.project_path = Some(path.clone());
        app.save_project();
        let loaded = substancecraft_api::load_project_file(&path).unwrap();
        assert_eq!(loaded.cameras, app.project.cameras);

        // A still render can use a saved view instead of the viewport.
        let detail = app.project.cameras[1].camera;
        app.queue_render(folder.path().join("detail.png"), 256, Some(detail));
        assert_eq!(app.render_request.as_ref().unwrap().camera, Some(detail));

        app.delete_named_camera(0);
        assert_eq!(app.project.cameras.len(), 1);
        assert_eq!(app.save_named_camera(), Some(1));
        // Names stay unique after deletions.
        assert_ne!(app.project.cameras[0].name, app.project.cameras[1].name);
    }
}
