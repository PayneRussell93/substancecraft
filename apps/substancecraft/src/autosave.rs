//! Periodic autosave and the crash-recovery prompt.
//!
//! Snapshots go to the per-user recovery folder, never over the user's own
//! save. A session that ends without closing leaves its marker behind, and
//! the next launch offers to restore the snapshot.

use std::time::{Duration, Instant};

use eframe::egui::{self, Id, RichText};
use substancecraft_api::{RecoveryCandidate, RecoveryStore};
use substancecraft_core::MaterialProject;

use super::{SubstanceCraftApp, path_label};
use crate::{theme, widgets};

/// How often the project is checked for changes worth snapshotting.
pub const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Default)]
pub struct Autosave {
    store: Option<RecoveryStore>,
    /// A previous session's snapshot, waiting for Restore or Discard.
    pub prompt: Option<RecoveryCandidate>,
    last_check: Option<Instant>,
    /// The project as last autosaved, saved, or opened; unchanged projects
    /// are not written again. Shared with the write that produced it.
    pub(super) snapshot: Option<std::sync::Arc<MaterialProject>>,
}

impl SubstanceCraftApp {
    /// Starts autosaving into `store`, offering a crashed session's snapshot.
    pub(super) fn start_recovery(&mut self, store: RecoveryStore) {
        match store.begin_session() {
            Ok(candidate) => self.autosave.prompt = candidate,
            Err(error) => self.project_error = Some(format!("Autosave unavailable: {error}")),
        }
        self.autosave.store = Some(store);
        self.autosave.snapshot = Some(std::sync::Arc::new(self.project.clone()));
    }

    /// Records that `project` matches what is on disk (opened or saved), so
    /// it is not autosaved until it changes.
    pub(super) fn mark_project_settled(&mut self) {
        self.autosave.snapshot = Some(std::sync::Arc::new(self.project.clone()));
    }

    /// After an explicit save the snapshot is older than the save, so it is
    /// deleted rather than offered after a later crash. A pending recovery
    /// offer keeps its snapshot.
    pub(super) fn discard_autosave(&mut self) {
        if self.autosave.prompt.is_some() {
            return;
        }
        if let Some(store) = &self.autosave.store
            && let Err(error) = store.discard()
        {
            self.project_error = Some(format!("Could not clear the autosave: {error}"));
        }
    }

    /// Writes a snapshot when the interval has passed and the project
    /// changed. Paused while a recovery offer is open, so the old snapshot
    /// is not replaced before the user decides.
    pub(super) fn autosave_tick(&mut self, now: Instant) {
        let Some(store) = self.autosave.store.clone() else {
            return;
        };
        if self.autosave.prompt.is_some() || self.active_stroke.is_some() || self.saving.busy() {
            return;
        }
        let last = *self.autosave.last_check.get_or_insert(now);
        if now.duration_since(last) < AUTOSAVE_INTERVAL {
            return;
        }
        self.autosave.last_check = Some(now);
        if self.autosave.snapshot.as_deref() == Some(&self.project) {
            return;
        }
        self.start_autosave(store);
    }

    /// Answers the recovery offer: `restore` loads the snapshot (keeping the
    /// original save path, which is only written on an explicit save);
    /// otherwise the snapshot is deleted.
    pub(super) fn resolve_recovery(&mut self, restore: bool) {
        let Some(candidate) = self.autosave.prompt.take() else {
            return;
        };
        let Some(store) = self.autosave.store.clone() else {
            return;
        };
        if !restore {
            if let Err(error) = store.discard() {
                self.project_error = Some(format!("Could not discard the autosave: {error}"));
            }
            return;
        }
        self.open_project_path(candidate.autosave.clone());
        if self.project_path.as_deref() != Some(candidate.autosave.as_path()) {
            // Opening failed; the error is already showing.
            return;
        }
        self.project_path = candidate.original.clone();
        // Unsaved until the user saves, so keep autosaving it.
        self.autosave.snapshot = None;
        self.project_error = Some(match &candidate.original {
            Some(original) => format!(
                "Restored autosave of {}. Save to keep it.",
                path_label(original)
            ),
            None => "Restored an autosaved untitled project. Save to keep it.".to_owned(),
        });
    }

    /// Ends the session cleanly so the next launch offers nothing.
    pub(super) fn end_recovery_session(&mut self) {
        if let Some(store) = &self.autosave.store {
            // Nothing useful can be done with an error during shutdown.
            let _ = store.end_session();
        }
    }

    pub(super) fn show_recovery_prompt(&mut self, context: &egui::Context) {
        let Some(candidate) = &self.autosave.prompt else {
            return;
        };
        let source = candidate
            .original
            .as_deref()
            .map_or_else(|| "an untitled project".to_owned(), path_label);
        let mut answer = None;
        let modal = egui::Modal::new(Id::new("crash_recovery")).show(context, |ui| {
            ui.set_max_width(420.0);
            ui.label(
                RichText::new("SubstanceCraft did not close properly")
                    .size(15.0)
                    .color(theme::TEXT),
            );
            ui.add(
                egui::Label::new(
                    RichText::new(format!(
                        "An autosave of {source} is available. Restoring it does not change your last save."
                    ))
                    .size(12.0)
                    .color(theme::TEXT_WEAK),
                )
                .wrap(),
            );
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if widgets::pill_button(ui, None, "Restore", true, false).clicked() {
                    answer = Some(true);
                }
                if widgets::pill_button(ui, None, "Discard", false, false).clicked() {
                    answer = Some(false);
                }
            });
        });
        // Closing the dialog keeps the snapshot for later rather than
        // deleting it.
        if modal.should_close() && answer.is_none() {
            self.autosave.prompt = None;
        }
        if let Some(restore) = answer {
            self.resolve_recovery(restore);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tc_0235_app_autosaves_changes_and_restores_after_a_crash() {
        let folder = tempfile::tempdir().unwrap();
        let saved = folder.path().join("scene.sproject");
        let store = RecoveryStore::new(folder.path().join("recovery"));

        let mut app = SubstanceCraftApp::default();
        app.start_recovery(store.clone());
        assert!(app.autosave.prompt.is_none());
        app.add_texture_set();
        app.project_path = Some(saved.clone());
        app.save_project();
        app.finish_saves();
        let start = Instant::now();
        app.autosave_tick(start);
        app.finish_saves();
        // Unchanged projects are not written.
        app.autosave_tick(start + AUTOSAVE_INTERVAL);
        app.finish_saves();
        assert!(!store.dir().join("autosave.sproject").exists());

        app.add_paint_layer();
        let layers = app.project.texture_sets[0].layers.len();
        app.autosave_tick(start + AUTOSAVE_INTERVAL);
        app.finish_saves();
        assert!(!store.dir().join("autosave.sproject").exists(), "too soon");
        app.autosave_tick(start + AUTOSAVE_INTERVAL * 2);
        app.finish_saves();
        assert!(store.dir().join("autosave.sproject").exists());
        // The app "crashes": no end_recovery_session.
        drop(app);

        let mut next = SubstanceCraftApp::default();
        next.start_recovery(store.clone());
        assert_eq!(
            next.autosave.prompt.as_ref().unwrap().original.as_deref(),
            Some(saved.as_path())
        );
        next.resolve_recovery(true);
        assert_eq!(next.project.texture_sets[0].layers.len(), layers);
        assert_eq!(next.project_path.as_deref(), Some(saved.as_path()));
        // The explicit save still holds the project before the new layer.
        let on_disk = substancecraft_api::load_project_file(&saved).unwrap();
        assert_eq!(on_disk.texture_sets[0].layers.len(), layers - 1);
        next.end_recovery_session();

        let mut clean = SubstanceCraftApp::default();
        clean.start_recovery(store);
        assert!(clean.autosave.prompt.is_none());
    }
}
