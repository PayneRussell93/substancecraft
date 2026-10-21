//! Project saves and autosaves written on a background thread (PERF-05),
//! so encoding and syncing a large project to disk never stalls the UI.
//! Each write works from a snapshot taken when it starts; that snapshot
//! then becomes the autosave baseline, without a second copy.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};
use std::thread::JoinHandle;

use eframe::egui;
use substancecraft_api::{save_project_file, save_project_history};
use substancecraft_core::MaterialProject;

use super::SubstanceCraftApp;

/// What a finished write reports: an error, or a note for a save that
/// succeeded with a sidecar that did not.
type WriteResult = Result<Option<String>, String>;

/// A write in progress.
pub struct ProjectWrite {
    receiver: Receiver<WriteResult>,
    handle: JoinHandle<()>,
    snapshot: Arc<MaterialProject>,
    /// Where an explicit save goes; `None` for an autosave.
    path: Option<PathBuf>,
    /// The open project the write belongs to.
    generation: u64,
}

#[derive(Default)]
pub struct Saving {
    save: Option<ProjectWrite>,
    autosave: Option<ProjectWrite>,
    /// Bumped whenever another project is opened or started, so a write
    /// that finishes later does not touch it.
    pub generation: u64,
}

impl Saving {
    pub fn busy(&self) -> bool {
        self.save.is_some() || self.autosave.is_some()
    }
}

/// Runs `write` on its own thread.
fn spawn(
    write: impl FnOnce() -> WriteResult + Send + 'static,
) -> Result<(Receiver<WriteResult>, JoinHandle<()>), String> {
    let (sender, receiver) = sync_channel(1);
    let handle = std::thread::Builder::new()
        .name("project-save".to_owned())
        .spawn(move || {
            let _ = sender.send(write());
        })
        .map_err(|error| error.to_string())?;
    Ok((receiver, handle))
}

impl SubstanceCraftApp {
    /// Starts saving the project, its undo history, and its bake log to
    /// `path` in the background. A save already running finishes first.
    pub(super) fn start_save(&mut self, path: PathBuf) {
        if let Some(running) = self.saving.save.take() {
            self.finish_write(running, None);
        }
        let snapshot = Arc::new(self.project.clone());
        let (project, history, bake_log, target) = (
            Arc::clone(&snapshot),
            self.history.clone(),
            self.bake_log.clone(),
            path.clone(),
        );
        let spawned = spawn(move || {
            save_project_file(&target, &project).map_err(|error| error.to_string())?;
            let note = save_project_history(&target, &history)
                .err()
                .map(|error| format!("Saved, but the undo history was not: {error}"))
                .or_else(|| {
                    bake_log
                        .save(&target)
                        .err()
                        .map(|error| format!("Saved, but the bake log was not: {error}"))
                });
            Ok(note)
        });
        match spawned {
            Ok((receiver, handle)) => {
                self.saving.save = Some(ProjectWrite {
                    receiver,
                    handle,
                    snapshot,
                    path: Some(path),
                    generation: self.saving.generation,
                });
            }
            Err(error) => self.project_error = Some(format!("Save failed: {error}")),
        }
    }

    /// Starts writing an autosave snapshot of the project in the
    /// background.
    pub(super) fn start_autosave(&mut self, store: substancecraft_api::RecoveryStore) {
        let snapshot = Arc::new(self.project.clone());
        let (project, original) = (Arc::clone(&snapshot), self.project_path.clone());
        let spawned = spawn(move || {
            store
                .write_autosave(&project, original.as_deref())
                .map(|()| None)
                .map_err(|error| format!("Autosave failed: {error}"))
        });
        match spawned {
            Ok((receiver, handle)) => {
                self.saving.autosave = Some(ProjectWrite {
                    receiver,
                    handle,
                    snapshot,
                    path: None,
                    generation: self.saving.generation,
                });
            }
            Err(error) => self.project_error = Some(format!("Autosave failed: {error}")),
        }
    }

    /// Takes in finished writes; call every frame.
    pub(super) fn poll_saves(&mut self, context: &egui::Context) {
        for autosave in [true, false] {
            let slot = if autosave {
                &mut self.saving.autosave
            } else {
                &mut self.saving.save
            };
            let result = match slot.as_ref().map(|write| write.receiver.try_recv()) {
                Some(Ok(result)) => result,
                Some(Err(TryRecvError::Disconnected)) => {
                    Err("the save stopped unexpectedly".to_owned())
                }
                Some(Err(TryRecvError::Empty)) => {
                    context.request_repaint_after(std::time::Duration::from_millis(50));
                    continue;
                }
                None => continue,
            };
            if let Some(write) = slot.take() {
                self.finish_write(write, Some(result));
            }
        }
    }

    /// Waits for every write in progress and takes in its result; for
    /// shutdown, and for callers that need the files on disk now.
    pub(super) fn finish_saves(&mut self) {
        if let Some(write) = self.saving.autosave.take() {
            self.finish_write(write, None);
        }
        if let Some(write) = self.saving.save.take() {
            self.finish_write(write, None);
        }
    }

    /// Takes in a finished write; with no `result` yet, waits for it.
    fn finish_write(&mut self, write: ProjectWrite, result: Option<WriteResult>) {
        let result = result.unwrap_or_else(|| {
            write
                .receiver
                .recv()
                .unwrap_or_else(|_| Err("the save stopped unexpectedly".to_owned()))
        });
        let _ = write.handle.join();
        let current = write.generation == self.saving.generation;
        match (result, write.path) {
            (Err(error), Some(_)) => self.project_error = Some(format!("Save failed: {error}")),
            (Err(error), None) => self.project_error = Some(error),
            (Ok(note), Some(path)) if current => {
                self.project_error = note;
                self.project_path = Some(path);
                self.autosave.snapshot = Some(write.snapshot);
                // An autosave still being written predates this save; let
                // it land before the recovery copy is cleared.
                if let Some(autosave) = self.saving.autosave.take() {
                    let _ = autosave.handle.join();
                }
                self.discard_autosave();
            }
            (Ok(_), None) if current => self.autosave.snapshot = Some(write.snapshot),
            (Ok(_), _) => {}
        }
    }
}
