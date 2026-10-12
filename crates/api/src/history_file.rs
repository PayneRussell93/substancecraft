//! Undo history kept beside a project as `<project>.history`, so the undo
//! steps of a session survive an explicit save and the next open.
//!
//! The sidecar records a hash of the project file it was written with and
//! is ignored once that file changes without it (edited by hand, saved by
//! another tool, or restored from version control), because its commands
//! only apply to the exact state they were recorded against. Keeping it out
//! of the project file leaves `.sproject` diffs free of history noise
//! (PROD-05).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use substancecraft_core::EditHistory;

use crate::ProjectFileError;

/// Appended to the project file name for the history sidecar.
pub const HISTORY_EXTENSION: &str = "history";
/// Largest history sidecar read back.
pub const MAX_HISTORY_BYTES: usize = 256 * 1024 * 1024;
const HISTORY_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct HistoryFile {
    version: u32,
    /// FNV-1a hash of the project file's bytes, in hex.
    project_hash: String,
    history: EditHistory,
}

/// `scene.sproject` → `scene.sproject.history`.
pub fn history_path(project: &Path) -> PathBuf {
    let mut name = project.as_os_str().to_owned();
    name.push(".");
    name.push(HISTORY_EXTENSION);
    PathBuf::from(name)
}

fn fnv1a(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}

/// Writes `history` beside the project file at `project` (which must be
/// saved already). An empty history removes the sidecar instead.
pub fn save_project_history(project: &Path, history: &EditHistory) -> Result<(), ProjectFileError> {
    let sidecar = history_path(project);
    if !history.can_undo() && !history.can_redo() {
        return match std::fs::remove_file(&sidecar) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                Err(ProjectFileError::Io(error.to_string()))
            }
            _ => Ok(()),
        };
    }
    let project_bytes =
        std::fs::read(project).map_err(|error| ProjectFileError::Io(error.to_string()))?;
    let file = HistoryFile {
        version: HISTORY_VERSION,
        project_hash: fnv1a(&project_bytes),
        history: history.clone(),
    };
    let bytes =
        serde_json::to_vec(&file).map_err(|error| ProjectFileError::Encode(error.to_string()))?;
    let parent = sidecar
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| ProjectFileError::Io(error.to_string()))?;
    std::io::Write::write_all(&mut temporary, &bytes)
        .map_err(|error| ProjectFileError::Io(error.to_string()))?;
    temporary
        .persist(&sidecar)
        .map_err(|error| ProjectFileError::Io(error.to_string()))?;
    Ok(())
}

/// The history saved beside `project`, when there is one and the project
/// file is exactly the one it was saved with.
pub fn load_project_history(project: &Path) -> Option<EditHistory> {
    let sidecar = std::fs::read(history_path(project)).ok()?;
    if sidecar.len() > MAX_HISTORY_BYTES {
        return None;
    }
    let file: HistoryFile = serde_json::from_slice(&sidecar).ok()?;
    let project_bytes = std::fs::read(project).ok()?;
    (file.version == HISTORY_VERSION && file.project_hash == fnv1a(&project_bytes))
        .then_some(file.history)
}
