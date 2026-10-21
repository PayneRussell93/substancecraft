//! Bake cache management: how much memory baked previews and mesh maps
//! hold, a size budget for background texture-set bakes, the on-disk
//! mesh-map cache ([`crate::disk_cache`]) and its budget, and a Clear
//! action for both.

use std::collections::BTreeMap;
use std::path::PathBuf;

use eframe::egui::{RichText, Ui};
use substancecraft_core::{MaterialTextures, TextureSetId};

use super::{PreviewTextures, SubstanceCraftApp};
use crate::{theme, widgets};

pub const MIN_CACHE_BUDGET_MB: u32 = 64;
pub const MAX_CACHE_BUDGET_MB: u32 = 16_384;
pub const DEFAULT_CACHE_BUDGET_MB: u32 = 1_024;
/// Rough bytes per mesh-map texel (positions, normals, tangents, scalars).
const MESH_MAP_TEXEL_BYTES: usize = 80;

#[derive(Default)]
pub struct CacheState {
    /// Background bakes dropped to stay in budget, by the set revision they
    /// held; they are not rebaked until the set changes.
    pub evicted: BTreeMap<TextureSetId, u64>,
    /// Bytes the disk cache held when last measured; `None` re-measures.
    pub disk_usage: Option<u64>,
}

pub fn material_bytes(textures: &MaterialTextures) -> usize {
    [
        &textures.base_color,
        &textures.orm,
        &textures.normal,
        &textures.emissive,
    ]
    .iter()
    .map(|texture| texture.rgba8.len())
    .sum()
}

/// Memory each cache holds, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheUsage {
    pub set_bakes: usize,
    pub preview: usize,
    pub mesh_maps: usize,
}

impl CacheUsage {
    pub fn total(self) -> usize {
        self.set_bakes + self.preview + self.mesh_maps
    }
}

fn megabytes(bytes: usize) -> f32 {
    bytes as f32 / (1024.0 * 1024.0)
}

impl SubstanceCraftApp {
    pub(super) fn cache_usage(&self) -> CacheUsage {
        CacheUsage {
            set_bakes: self
                .set_materials
                .values()
                .map(|cached| material_bytes(&cached.textures))
                .sum(),
            preview: self
                .baked_preview
                .as_ref()
                .map_or(0, |preview| match &preview.textures {
                    PreviewTextures::Material(textures) => material_bytes(textures),
                    PreviewTextures::Single(texture) => texture.rgba8.len(),
                }),
            mesh_maps: self.mesh_maps.as_ref().map_or(0, |maps| {
                maps.iter()
                    .map(|maps| maps.width() as usize * maps.height() as usize)
                    .sum::<usize>()
                    * MESH_MAP_TEXEL_BYTES
            }),
        }
    }

    fn cache_budget_bytes(&self) -> usize {
        self.preferences.values.cache_budget_mb as usize * 1024 * 1024
    }

    /// The disk cache's folder when it is on: the preference, else `Cache`
    /// beside the preferences file. Without a preferences file (tests and
    /// headless runs) nothing is cached on disk unless a folder is set.
    pub(super) fn disk_cache_dir(&self) -> Option<PathBuf> {
        let preferences = &self.preferences;
        if !preferences.values.disk_cache {
            return None;
        }
        preferences.values.disk_cache_folder.clone().or_else(|| {
            preferences
                .path
                .as_ref()?
                .parent()
                .map(|folder| folder.join("Cache"))
        })
    }

    pub(super) fn disk_cache_budget_bytes(&self) -> u64 {
        u64::from(self.preferences.values.disk_cache_budget_mb) * 1024 * 1024
    }

    /// Bytes the disk cache holds, measured once until it next changes.
    fn disk_cache_usage(&mut self) -> u64 {
        if let Some(bytes) = self.cache.disk_usage {
            return bytes;
        }
        let bytes = self
            .disk_cache_dir()
            .map_or(0, |folder| crate::disk_cache::usage(&folder));
        self.cache.disk_usage = Some(bytes);
        bytes
    }

    /// Drops the largest background bakes until they fit the budget. The
    /// selected set's preview is never dropped.
    pub(super) fn enforce_cache_budget(&mut self) {
        let budget = self.cache_budget_bytes();
        let mut used = self.cache_usage().set_bakes;
        if used <= budget {
            return;
        }
        let mut by_size: Vec<(usize, TextureSetId, u64)> = self
            .set_materials
            .iter()
            .map(|(id, cached)| (material_bytes(&cached.textures), *id, cached.revision))
            .collect();
        by_size.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for (bytes, id, revision) in by_size {
            if used <= budget {
                break;
            }
            self.set_materials.remove(&id);
            self.cache.evicted.insert(id, revision);
            used -= bytes;
        }
    }

    /// Empties every bake cache, on disk too. The current preview and any
    /// needed mesh maps rebake on demand. Returns the bytes released.
    pub(super) fn clear_caches(&mut self) -> usize {
        let on_disk = self
            .disk_cache_dir()
            .map_or(0, |folder| crate::disk_cache::clear(&folder));
        self.cache.disk_usage = None;
        let freed = self.cache_usage().total() + on_disk as usize;
        self.set_materials.clear();
        self.cache.evicted.clear();
        self.set_bake_failed.clear();
        self.cancel_mesh_map_job();
        self.mesh_maps = None;
        self.mesh_maps_key = None;
        self.resource_thumbnails.clear();
        self.invalidate_preview();
        self.baked_preview = None;
        freed
    }

    /// Cache usage, budget, and the Clear action, in the Bake drawer.
    pub(super) fn cache_section(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Cache");
        let usage = self.cache_usage();
        widgets::property_row(ui, "In memory", |ui| {
            ui.label(format!("{:.1} MB", megabytes(usage.total())));
        });
        ui.label(
            RichText::new(format!(
                "Mesh maps {:.1} MB · preview {:.1} MB · other sets {:.1} MB",
                megabytes(usage.mesh_maps),
                megabytes(usage.preview),
                megabytes(usage.set_bakes)
            ))
            .size(11.5)
            .color(theme::TEXT_FAINT),
        );
        let mut budget = self.preferences.values.cache_budget_mb;
        let response = widgets::property_row(ui, "Budget", |ui| {
            ui.add(
                eframe::egui::DragValue::new(&mut budget)
                    .range(MIN_CACHE_BUDGET_MB..=MAX_CACHE_BUDGET_MB)
                    .suffix(" MB"),
            )
        })
        .on_hover_text("Memory background bakes of other texture sets may use");
        if budget != self.preferences.values.cache_budget_mb {
            self.preferences.values.cache_budget_mb = budget;
            if !response.dragged() {
                self.save_preferences();
            }
        }
        if response.drag_stopped() {
            self.save_preferences();
        }
        self.disk_cache_section(ui);
        if !self.cache.evicted.is_empty() {
            ui.label(
                RichText::new(format!(
                    "{} set(s) over budget show grey until edited or selected.",
                    self.cache.evicted.len()
                ))
                .size(11.5)
                .color(theme::TEXT_WEAK),
            );
        }
        if widgets::pill_button(ui, None, "Clear cache", false, true)
            .on_hover_text(
                "Free baked previews and mesh maps, in memory and on disk; they rebake when needed",
            )
            .clicked()
        {
            let freed = self.clear_caches();
            self.render_status = None;
            self.project_error = Some(format!(
                "Cleared {:.1} MB of cached bakes",
                megabytes(freed)
            ));
        }
    }

    /// The disk cache's switch, folder, usage, and budget.
    fn disk_cache_section(&mut self, ui: &mut Ui) {
        let mut on = self.preferences.values.disk_cache;
        let toggled = widgets::property_row(ui, "Disk cache", |ui| widgets::toggle(ui, &mut on))
            .on_hover_text("Keep mesh-map bakes on disk so later sessions reuse them")
            .changed();
        if toggled {
            self.preferences.values.disk_cache = on;
            self.cache.disk_usage = None;
            self.save_preferences();
        }
        let Some(folder) = self.disk_cache_dir() else {
            return;
        };
        let used = self.disk_cache_usage();
        ui.label(
            RichText::new(format!(
                "{:.1} MB in {}",
                megabytes(used as usize),
                folder.display()
            ))
            .size(11.5)
            .color(theme::TEXT_FAINT),
        );
        let mut budget = self.preferences.values.disk_cache_budget_mb;
        let response = widgets::property_row(ui, "Disk budget", |ui| {
            ui.add(
                eframe::egui::DragValue::new(&mut budget)
                    .range(
                        crate::disk_cache::MIN_DISK_BUDGET_MB
                            ..=crate::disk_cache::MAX_DISK_BUDGET_MB,
                    )
                    .suffix(" MB"),
            )
        })
        .on_hover_text("Disk space cached bakes may use; the least recently used go first");
        if budget != self.preferences.values.disk_cache_budget_mb {
            self.preferences.values.disk_cache_budget_mb = budget;
        }
        if response.drag_stopped() || (response.changed() && !response.dragged()) {
            crate::disk_cache::prune(&folder, self.disk_cache_budget_bytes(), None);
            self.cache.disk_usage = None;
            self.save_preferences();
        }
        if widgets::pill_button(ui, None, "Cache folder…", false, true)
            .on_hover_text("Choose where cached bakes are kept")
            .clicked()
            && let Some(chosen) = rfd::FileDialog::new()
                .set_title("Folder for cached bakes")
                .set_directory(&folder)
                .pick_folder()
        {
            self.preferences.values.disk_cache_folder = Some(chosen);
            self.cache.disk_usage = None;
            self.save_preferences();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::bake_log::BakeOutcome;
    use substancecraft_core::PrimitivePreset;

    #[test]
    fn tc_0403_mesh_map_bakes_persist_on_disk_across_sessions() {
        let folder = tempfile::tempdir().unwrap();
        let context = eframe::egui::Context::default();
        let session = || {
            let mut app = SubstanceCraftApp::default();
            app.preferences.values.disk_cache_folder = Some(folder.path().to_path_buf());
            app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
            app.mesh_map_settings.resolution = 32;
            app.mesh_map_settings.ao_samples = 2;
            app
        };
        let bake = |app: &mut SubstanceCraftApp| {
            app.start_mesh_map_bake();
            crate::tests::pump_until(app, &context, |app| app.mesh_map_job.is_none());
            app.bake_log.entries().last().unwrap().outcome.clone()
        };
        let device =
            |app: &SubstanceCraftApp| app.bake_log.entries().last().unwrap().device.clone();
        let mut first = session();
        assert!(matches!(bake(&mut first), BakeOutcome::Finished { .. }));
        assert!(device(&first).is_some(), "a bake logs its device");
        assert!(first.disk_cache_usage() > 0);

        // A later session with the same mesh and settings reads it back.
        let mut second = session();
        assert_eq!(bake(&mut second), BakeOutcome::Loaded { slots: 1 });
        assert_eq!(device(&second), None, "nothing was traced");
        assert_eq!(second.mesh_maps, first.mesh_maps);
        assert!(second.mesh_maps_fresh());

        // Other settings are another bake; both stay on disk.
        let stored = second.disk_cache_usage();
        second.mesh_map_settings.ao_samples = 3;
        assert!(matches!(bake(&mut second), BakeOutcome::Finished { .. }));
        assert!(second.disk_cache_usage() > stored);

        // Switched off, nothing is read or written.
        let mut off = session();
        off.preferences.values.disk_cache = false;
        off.preferences.values.gpu_baking = false;
        assert!(matches!(bake(&mut off), BakeOutcome::Finished { .. }));
        assert_eq!(device(&off).as_deref(), Some("CPU"));

        // Clear empties the disk cache too.
        let freed = second.clear_caches();
        assert!(freed as u64 >= stored);
        assert_eq!(second.disk_cache_usage(), 0);
        assert!(matches!(bake(&mut session()), BakeOutcome::Finished { .. }));
    }

    #[test]
    fn tc_0252_cache_budget_evicts_background_bakes_and_clear_frees_everything() {
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.mesh_map_settings.resolution = 32;
        app.mesh_map_settings.ao_samples = 2;
        app.start_mesh_map_bake();
        let context = eframe::egui::Context::default();
        crate::tests::pump_until(&mut app, &context, |app| app.mesh_maps.is_some());
        assert_eq!(app.cache_usage().mesh_maps, 32 * 32 * MESH_MAP_TEXEL_BYTES);

        // Fake two background bakes of different sizes.
        let texture = |size: u32| substancecraft_core::BakedTexture {
            width: size,
            height: size,
            rgba8: vec![0; (size * size * 4) as usize],
        };
        let material = |size: u32| super::super::CachedMaterial {
            revision: 3,
            textures: std::sync::Arc::new(MaterialTextures {
                base_color: texture(size),
                orm: texture(size),
                normal: texture(size),
                emissive: texture(size),
            }),
            dirty: None,
        };
        app.set_materials.insert(TextureSetId(7), material(512));
        app.set_materials.insert(TextureSetId(8), material(64));
        app.preferences.values.cache_budget_mb = 1;
        app.enforce_cache_budget();
        // The 4 MiB bake goes; the small one stays.
        assert!(!app.set_materials.contains_key(&TextureSetId(7)));
        assert!(app.set_materials.contains_key(&TextureSetId(8)));
        assert_eq!(app.cache.evicted.get(&TextureSetId(7)), Some(&3));

        let freed = app.clear_caches();
        assert!(freed >= 64 * 64 * 16 + 32 * 32 * MESH_MAP_TEXEL_BYTES);
        assert_eq!(app.cache_usage(), CacheUsage::default());
        assert!(app.cache.evicted.is_empty());
    }
}
