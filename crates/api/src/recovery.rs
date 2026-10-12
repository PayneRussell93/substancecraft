//! Autosave and crash recovery.
//!
//! A [`RecoveryStore`] owns a folder, separate from wherever the user saves,
//! holding the latest autosave snapshot, a note naming the project it was
//! taken from, and a session marker. The marker is written when a session
//! starts and removed when it ends cleanly, so finding it at startup means
//! the previous session ended without closing: its autosave (if any) is
//! offered for restoring. Autosaves never touch the user's own save.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use substancecraft_core::MaterialProject;

use crate::{ProjectFileError, load_project_file, save_project_file};

const AUTOSAVE_FILE: &str = "autosave.sproject";
const NOTE_FILE: &str = "autosave.json";
const SESSION_FILE: &str = "session.lock";

/// Folder holding autosave snapshots and the session marker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryStore {
    dir: PathBuf,
}

/// An autosave left behind by a session that did not end cleanly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveryCandidate {
    /// The snapshot to restore.
    pub autosave: PathBuf,
    /// Where the user last saved that project, if they had.
    pub original: Option<PathBuf>,
    /// When the snapshot was written, in seconds since the Unix epoch.
    pub saved_at: u64,
}

#[derive(Serialize, Deserialize)]
struct AutosaveNote {
    original: Option<PathBuf>,
    saved_at: u64,
}

fn io_error(error: std::io::Error) -> ProjectFileError {
    ProjectFileError::Io(error.to_string())
}

impl RecoveryStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The per-user recovery folder: `%LOCALAPPDATA%\SubstanceCraft\recovery`
    /// on Windows, else `$XDG_STATE_HOME/substancecraft/recovery` or
    /// `~/.local/state/substancecraft/recovery`.
    pub fn default_location() -> Option<PathBuf> {
        let from = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
        if let Some(local) = from("LOCALAPPDATA") {
            return Some(PathBuf::from(local).join("SubstanceCraft").join("recovery"));
        }
        if let Some(state) = from("XDG_STATE_HOME") {
            return Some(PathBuf::from(state).join("substancecraft").join("recovery"));
        }
        from("HOME").map(|home| {
            PathBuf::from(home)
                .join(".local")
                .join("state")
                .join("substancecraft")
                .join("recovery")
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Starts a session: returns the previous session's autosave when that
    /// session did not end cleanly, then marks this session as running.
    pub fn begin_session(&self) -> Result<Option<RecoveryCandidate>, ProjectFileError> {
        std::fs::create_dir_all(&self.dir).map_err(io_error)?;
        let crashed = self.dir.join(SESSION_FILE).exists();
        let candidate = if crashed { self.candidate() } else { None };
        if !crashed {
            // A clean exit leaves nothing worth restoring.
            self.discard()?;
        }
        std::fs::write(
            self.dir.join(SESSION_FILE),
            format!("pid {}\n", std::process::id()),
        )
        .map_err(io_error)?;
        Ok(candidate)
    }

    fn candidate(&self) -> Option<RecoveryCandidate> {
        let autosave = self.dir.join(AUTOSAVE_FILE);
        if !autosave.is_file() {
            return None;
        }
        let note = std::fs::read(self.dir.join(NOTE_FILE))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<AutosaveNote>(&bytes).ok());
        Some(RecoveryCandidate {
            autosave,
            original: note.as_ref().and_then(|note| note.original.clone()),
            saved_at: note.map_or(0, |note| note.saved_at),
        })
    }

    /// Writes a snapshot of `project`, replacing the previous one. `original`
    /// is where the user last saved it, remembered so a restore can offer to
    /// save back there.
    pub fn write_autosave(
        &self,
        project: &MaterialProject,
        original: Option<&Path>,
    ) -> Result<(), ProjectFileError> {
        std::fs::create_dir_all(&self.dir).map_err(io_error)?;
        save_project_file(self.dir.join(AUTOSAVE_FILE), project)?;
        let note = AutosaveNote {
            original: original.map(Path::to_path_buf),
            saved_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs()),
        };
        let bytes = serde_json::to_vec(&note)
            .map_err(|error| ProjectFileError::Encode(error.to_string()))?;
        std::fs::write(self.dir.join(NOTE_FILE), bytes).map_err(io_error)
    }

    /// Loads the snapshot a candidate points at.
    pub fn restore(
        &self,
        candidate: &RecoveryCandidate,
    ) -> Result<MaterialProject, ProjectFileError> {
        load_project_file(&candidate.autosave)
    }

    /// Deletes the snapshot (after it was restored, declined, or the
    /// project was saved).
    pub fn discard(&self) -> Result<(), ProjectFileError> {
        for name in [AUTOSAVE_FILE, NOTE_FILE] {
            match std::fs::remove_file(self.dir.join(name)) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                    return Err(io_error(error));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Ends the session cleanly: the snapshot and marker are removed, so the
    /// next start offers nothing.
    pub fn end_session(&self) -> Result<(), ProjectFileError> {
        self.discard()?;
        match std::fs::remove_file(self.dir.join(SESSION_FILE)) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(io_error(error)),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{MaterialChannel, TextureResolution, TextureSet, TextureSetId};

    fn project(name: &str) -> MaterialProject {
        MaterialProject {
            texture_sets: vec![TextureSet {
                id: TextureSetId(1),
                name: name.to_owned(),
                material_slot: "Default".to_owned(),
                resolution: TextureResolution::new(16, 16).unwrap(),
                channels: [MaterialChannel::BaseColor].into(),
                layers: Vec::new(),
                height_strength: 1.0,
                export: None,
            }],
            ..MaterialProject::empty()
        }
    }

    #[test]
    fn tc_0234_autosaves_survive_a_crash_but_not_a_clean_exit() {
        let folder = tempfile::tempdir().unwrap();
        let saved = folder.path().join("scene.sproject");
        save_project_file(&saved, &project("Saved")).unwrap();
        let store = RecoveryStore::new(folder.path().join("recovery"));

        // First run: nothing to recover. Autosave, then "crash".
        assert_eq!(store.begin_session().unwrap(), None);
        store
            .write_autosave(&project("Edited"), Some(&saved))
            .unwrap();

        // Next run finds the snapshot; the explicit save is untouched.
        let candidate = store.begin_session().unwrap().unwrap();
        assert_eq!(candidate.original.as_deref(), Some(saved.as_path()));
        assert!(candidate.saved_at > 0);
        assert_eq!(
            store.restore(&candidate).unwrap().texture_sets[0].name,
            "Edited"
        );
        assert_eq!(
            load_project_file(&saved).unwrap().texture_sets[0].name,
            "Saved"
        );

        // A clean exit leaves nothing for the next start.
        store.write_autosave(&project("Later"), None).unwrap();
        store.end_session().unwrap();
        assert_eq!(store.begin_session().unwrap(), None);
        // Declining a recovery discards it.
        store.write_autosave(&project("Again"), None).unwrap();
        let candidate = store.begin_session().unwrap().unwrap();
        assert_eq!(candidate.original, None);
        store.discard().unwrap();
        assert!(store.restore(&candidate).is_err());
        assert_eq!(store.begin_session().unwrap(), None);
    }
}
