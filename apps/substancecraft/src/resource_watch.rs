//! Notices when bitmap resources or the high-poly bake source change on
//! disk, so edits made in an image editor or a re-export from a modelling
//! tool show up without re-importing.
//!
//! Files are polled (modification time and size) rather than watched through
//! OS notifications: projects reference few bitmaps, polling needs no extra
//! dependency, and it also catches editors that save by writing a temporary
//! file and renaming it over the original.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use substancecraft_core::{ProjectResource, ResourceId};

/// How often referenced files are checked.
pub(super) const SCAN_INTERVAL: Duration = Duration::from_secs(1);

/// What happens when a bitmap changes on disk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ReloadMode {
    /// Reload immediately.
    #[default]
    Auto,
    /// Collect changes until the user confirms.
    Ask,
    /// Do not check files.
    Off,
}

impl ReloadMode {
    pub(super) const ALL: [(Self, &'static str); 3] =
        [(Self::Auto, "Auto"), (Self::Ask, "Ask"), (Self::Off, "Off")];
}

/// Identity of a file's contents as far as polling can tell; `None` when the
/// file cannot be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStamp {
    modified: Option<SystemTime>,
    len: u64,
}

fn stamp(path: &Path) -> Option<FileStamp> {
    let metadata = std::fs::metadata(path).ok()?;
    Some(FileStamp {
        modified: metadata.modified().ok(),
        len: metadata.len(),
    })
}

#[derive(Debug, Default)]
pub(super) struct ResourceWatcher {
    pub(super) mode: ReloadMode,
    /// Last seen stamp per resource; resources not yet seen get a baseline
    /// on the next scan instead of being reported.
    stamps: BTreeMap<ResourceId, Option<FileStamp>>,
    last_scan: Option<Instant>,
    /// Changes waiting for confirmation in [`ReloadMode::Ask`].
    pub(super) pending: BTreeSet<ResourceId>,
    /// The high-poly bake source and its last seen stamp, so re-exporting
    /// it rebakes the mesh maps.
    high_poly: Option<(PathBuf, Option<FileStamp>)>,
    /// The high-poly source changed and waits for confirmation.
    pub(super) high_poly_pending: bool,
}

impl ResourceWatcher {
    /// Forgets every baseline and pending change, e.g. after a project loads.
    pub(super) fn reset(&mut self) {
        self.stamps.clear();
        self.pending.clear();
    }

    /// Records the current state of a file that was just loaded.
    pub(super) fn remember(&mut self, id: ResourceId, path: &Path) {
        self.stamps.insert(id, stamp(path));
    }

    /// Lets the next poll scan immediately.
    #[cfg(test)]
    pub(super) fn expire(&mut self) {
        self.last_scan = None;
    }

    /// Watches the high-poly source at `path` (or nothing), taking its
    /// current state as the baseline.
    pub(super) fn watch_high_poly(&mut self, path: Option<&Path>) {
        self.high_poly = path.map(|path| (path.to_path_buf(), stamp(path)));
        self.high_poly_pending = false;
    }

    /// True when a high-poly source is watched.
    pub(super) fn watches_high_poly(&self) -> bool {
        self.high_poly.is_some()
    }

    /// True once each time the watched high-poly file changes (including
    /// disappearing or reappearing).
    pub(super) fn scan_high_poly(&mut self) -> bool {
        let Some((path, last)) = &mut self.high_poly else {
            return false;
        };
        let current = stamp(path);
        std::mem::replace(last, current) != current
    }

    pub(super) fn forget(&mut self, id: ResourceId) {
        self.stamps.remove(&id);
        self.pending.remove(&id);
    }

    /// True when the scan interval has passed since the last scan.
    pub(super) fn due(&self, now: Instant) -> bool {
        self.mode != ReloadMode::Off
            && self
                .last_scan
                .is_none_or(|last| now.duration_since(last) >= SCAN_INTERVAL)
    }

    /// Compares every resource's file with its last stamp and returns the
    /// ones that changed, including files that disappeared or reappeared.
    pub(super) fn scan(&mut self, resources: &[ProjectResource], now: Instant) -> Vec<ResourceId> {
        self.last_scan = Some(now);
        self.stamps
            .retain(|id, _| resources.iter().any(|resource| resource.id == *id));
        self.pending.retain(|id| self.stamps.contains_key(id));
        let mut changed = Vec::new();
        for resource in resources {
            let current = stamp(Path::new(&resource.active_source()));
            match self.stamps.insert(resource.id, current) {
                Some(previous) if previous != current => changed.push(resource.id),
                _ => {}
            }
        }
        changed
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn resource(id: u64, path: &Path) -> ProjectResource {
        ProjectResource::new(ResourceId(id), format!("r{id}"), path.to_string_lossy())
    }

    #[test]
    fn tc_0120_watcher_reports_edited_removed_and_restored_files_once() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("grunge.png");
        std::fs::write(&path, b"one").unwrap();
        let resources = [resource(1, &path)];
        let mut watcher = ResourceWatcher::default();
        let now = Instant::now();

        // The first sight of a file is a baseline, not a change.
        assert!(watcher.due(now));
        assert!(watcher.scan(&resources, now).is_empty());
        assert!(!watcher.due(now));
        assert!(watcher.due(now + SCAN_INTERVAL));
        assert!(watcher.scan(&resources, now).is_empty());

        // A different size is a change even within one timestamp tick.
        std::fs::write(&path, b"longer contents").unwrap();
        assert_eq!(watcher.scan(&resources, now), [ResourceId(1)]);
        assert!(watcher.scan(&resources, now).is_empty(), "reported once");

        std::fs::remove_file(&path).unwrap();
        assert_eq!(watcher.scan(&resources, now), [ResourceId(1)]);
        assert!(watcher.scan(&resources, now).is_empty());
        std::fs::write(&path, b"back").unwrap();
        assert_eq!(watcher.scan(&resources, now), [ResourceId(1)]);

        // A remembered load is the new baseline; removed resources are
        // dropped along with their pending changes.
        std::fs::write(&path, b"saved again!").unwrap();
        watcher.remember(ResourceId(1), &path);
        assert!(watcher.scan(&resources, now).is_empty());
        watcher.pending.insert(ResourceId(1));
        assert!(watcher.scan(&[], now).is_empty());
        assert!(watcher.pending.is_empty());

        watcher.mode = ReloadMode::Off;
        assert!(!watcher.due(now + SCAN_INTERVAL * 10));
    }
}
