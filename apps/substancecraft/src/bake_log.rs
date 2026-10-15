//! A log of mesh-map bakes: when each ran, with what settings, how long it
//! took, and how it ended. Shown in the Bake drawer, copyable as text, and
//! saved beside the project (`name.sproject.bakelog`) so it survives
//! reopening.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use eframe::egui::{self, RichText, Ui};
use serde::{Deserialize, Serialize};
use substancecraft_core::{MeshMapSettings, MeshMaps};

use super::SubstanceCraftApp;
use crate::{theme, widgets};

/// Entries kept; older ones drop off.
pub const MAX_BAKE_LOG: usize = 64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BakeOutcome {
    Finished {
        slots: usize,
        /// Share of covered texels whose cage ray missed the high mesh, for
        /// high-to-low bakes.
        miss_percent: Option<f32>,
    },
    /// Finished, but the mesh or settings changed meanwhile; not used.
    Discarded,
    Cancelled,
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BakeLogEntry {
    /// Seconds since the Unix epoch when the bake started.
    pub started: u64,
    pub duration: Duration,
    pub settings: String,
    pub outcome: BakeOutcome,
}

impl BakeLogEntry {
    pub fn line(&self) -> String {
        let outcome = match &self.outcome {
            BakeOutcome::Finished {
                slots,
                miss_percent: Some(misses),
            } => format!("finished, {slots} slot(s), {misses:.1}% ray misses"),
            BakeOutcome::Finished { slots, .. } => format!("finished, {slots} slot(s)"),
            BakeOutcome::Discarded => "finished but outdated, discarded".to_owned(),
            BakeOutcome::Cancelled => "cancelled".to_owned(),
            BakeOutcome::Failed(error) => format!("failed: {error}"),
        };
        format!(
            "{} · {:.1} s · {} · {outcome}",
            clock(self.started),
            self.duration.as_secs_f32(),
            self.settings
        )
    }
}

/// `HH:MM:SS` (UTC) of a Unix time.
fn clock(seconds: u64) -> String {
    let day = seconds % 86_400;
    format!(
        "{:02}:{:02}:{:02} UTC",
        day / 3600,
        day % 3600 / 60,
        day % 60
    )
}

pub fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// A short description of what a bake used.
pub fn settings_summary(settings: &MeshMapSettings, high_poly: bool) -> String {
    format!(
        "{}² · {} AO samples · {}",
        settings.resolution,
        settings.ao_samples,
        if high_poly {
            "high to low"
        } else {
            "from mesh"
        }
    )
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BakeLog {
    entries: VecDeque<BakeLogEntry>,
}

impl BakeLog {
    pub fn record(&mut self, entry: BakeLogEntry) {
        if self.entries.len() == MAX_BAKE_LOG {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    pub fn entries(&self) -> impl DoubleEndedIterator<Item = &BakeLogEntry> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Every entry, oldest first, one per line.
    pub fn to_text(&self) -> String {
        self.entries
            .iter()
            .map(|entry| entry.line() + "\n")
            .collect()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Writes the log beside the project file at `project`; an empty log
    /// removes the sidecar instead.
    pub fn save(&self, project: &Path) -> std::io::Result<()> {
        let sidecar = log_path(project);
        if self.entries.is_empty() {
            return match std::fs::remove_file(&sidecar) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
                _ => Ok(()),
            };
        }
        let file = BakeLogFile {
            version: BAKE_LOG_VERSION,
            log: self.clone(),
        };
        let bytes = serde_json::to_vec(&file).map_err(std::io::Error::other)?;
        let mut temporary = sidecar.clone().into_os_string();
        temporary.push(".tmp");
        std::fs::write(&temporary, bytes)?;
        std::fs::rename(&temporary, &sidecar)
    }

    /// The log saved beside the project file at `project`, or an empty one
    /// when there is none or it cannot be read.
    pub fn load(project: &Path) -> Self {
        std::fs::read(log_path(project))
            .ok()
            .filter(|bytes| bytes.len() <= MAX_BAKE_LOG_BYTES)
            .and_then(|bytes| serde_json::from_slice::<BakeLogFile>(&bytes).ok())
            .filter(|file| file.version == BAKE_LOG_VERSION)
            .map(|file| {
                let mut log = file.log;
                while log.entries.len() > MAX_BAKE_LOG {
                    log.entries.pop_front();
                }
                log
            })
            .unwrap_or_default()
    }
}

const BAKE_LOG_VERSION: u32 = 1;
/// Larger sidecars are not read.
const MAX_BAKE_LOG_BYTES: usize = 1 << 20;

#[derive(Serialize, Deserialize)]
struct BakeLogFile {
    version: u32,
    log: BakeLog,
}

/// Where the bake log of the project at `project` is saved.
pub fn log_path(project: &Path) -> PathBuf {
    let mut name = project.as_os_str().to_owned();
    name.push(".bakelog");
    PathBuf::from(name)
}

/// Outcome of a finished high-to-low or plain bake.
pub fn finished(maps: &[MeshMaps]) -> BakeOutcome {
    let high: Vec<&MeshMaps> = maps.iter().filter(|maps| maps.from_high_poly()).collect();
    BakeOutcome::Finished {
        slots: maps.len(),
        miss_percent: (!high.is_empty()).then(|| {
            high.iter().map(|maps| maps.miss_fraction()).sum::<f32>() / high.len() as f32 * 100.0
        }),
    }
}

impl SubstanceCraftApp {
    /// The Bake log section of the Bake drawer.
    pub(super) fn bake_log_section(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Bake log");
        if self.bake_log.len() == 0 {
            ui.label(
                RichText::new("No bakes logged for this project yet.")
                    .size(11.5)
                    .color(theme::TEXT_FAINT),
            );
            return;
        }
        for entry in self.bake_log.entries().rev().take(8) {
            let color = match entry.outcome {
                BakeOutcome::Failed(_) => theme::DANGER,
                BakeOutcome::Finished { .. } => theme::TEXT_WEAK,
                _ => theme::TEXT_FAINT,
            };
            ui.add(egui::Label::new(RichText::new(entry.line()).size(11.0).color(color)).wrap());
        }
        ui.horizontal(|ui| {
            if widgets::pill_button(ui, None, "Copy log", false, false)
                .on_hover_text("Copy every entry as text")
                .clicked()
            {
                ui.ctx().copy_text(self.bake_log.to_text());
            }
            if widgets::pill_button(ui, None, "Clear", false, false).clicked() {
                self.bake_log.clear();
            }
        });
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::PrimitivePreset;

    #[test]
    fn tc_0251_bake_log_records_finished_cancelled_and_bounded_entries() {
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.mesh_map_settings.resolution = 32;
        app.mesh_map_settings.ao_samples = 2;
        app.start_mesh_map_bake();
        app.cancel_mesh_map_job();
        assert_eq!(app.bake_log.len(), 1);
        assert!(matches!(
            app.bake_log.entries().next().unwrap().outcome,
            BakeOutcome::Cancelled
        ));

        app.start_mesh_map_bake();
        let context = egui::Context::default();
        crate::tests::pump_until(&mut app, &context, |app| app.mesh_map_job.is_none());
        let last = app.bake_log.entries().last().unwrap().clone();
        assert_eq!(
            last.outcome,
            BakeOutcome::Finished {
                slots: 1,
                miss_percent: None
            }
        );
        assert!(last.settings.starts_with("32² · 2 AO samples"));
        assert!(app.bake_log.to_text().lines().count() == 2);

        for _ in 0..MAX_BAKE_LOG + 5 {
            app.bake_log.record(last.clone());
        }
        assert_eq!(app.bake_log.len(), MAX_BAKE_LOG);
        assert_eq!(clock(3_661), "01:01:01 UTC");
    }

    #[test]
    fn tc_0371_bake_log_is_saved_with_the_project_and_restored_on_open() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("scene.sproject");
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        let entry = |outcome| BakeLogEntry {
            started: 3_661,
            duration: Duration::from_millis(1_500),
            settings: "32² · 2 AO samples · from mesh".to_owned(),
            outcome,
        };
        app.bake_log.record(entry(BakeOutcome::Finished {
            slots: 2,
            miss_percent: Some(1.5),
        }));
        app.bake_log
            .record(entry(BakeOutcome::Failed("no mesh".to_owned())));
        app.project_path = Some(path.clone());
        app.save_project();
        assert!(log_path(&path).exists());

        let mut reopened = SubstanceCraftApp::default();
        reopened.open_project_path(path.clone());
        assert_eq!(reopened.bake_log, app.bake_log);
        assert_eq!(reopened.bake_log.to_text(), app.bake_log.to_text());

        // A cleared log removes the sidecar; a damaged one reads as empty.
        reopened.bake_log.clear();
        reopened.save_project();
        assert!(!log_path(&path).exists());
        std::fs::write(log_path(&path), b"{not json").unwrap();
        assert_eq!(BakeLog::load(&path).len(), 0);
    }
}
