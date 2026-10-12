//! A log of mesh-map bakes: when each ran, with what settings, how long it
//! took, and how it ended. Shown in the Bake drawer and copyable as text.

use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use eframe::egui::{self, RichText, Ui};
use substancecraft_core::{MeshMapSettings, MeshMaps};

use super::SubstanceCraftApp;
use crate::{theme, widgets};

/// Entries kept; older ones drop off.
pub const MAX_BAKE_LOG: usize = 64;

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Default)]
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
                RichText::new("No bakes yet this session.")
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
}
