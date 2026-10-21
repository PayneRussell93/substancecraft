//! On-disk cache of mesh-map bakes (PROD-06). Bakes are keyed by a content
//! hash of everything they read (mesh, high-poly source, cage, settings,
//! skew map, and the app version), not by session revisions, so reopening a
//! project or returning to a mesh reuses an earlier bake. The folder keeps
//! to a size budget by dropping the least recently used bakes first.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use substancecraft_core::{
    ContentHasher, MeshGeometry, MeshMapSettings, MeshMaps, TextureImage, decode_mesh_maps,
    encode_mesh_maps,
};

pub const MIN_DISK_BUDGET_MB: u32 = 256;
pub const MAX_DISK_BUDGET_MB: u32 = 65_536;
pub const DEFAULT_DISK_BUDGET_MB: u32 = 4_096;
/// Subfolder of the cache folder holding mesh-map bakes.
const MESH_MAPS_DIR: &str = "mesh-maps";
const EXTENSION: &str = "smaps";

/// Everything a mesh-map bake reads, hashed into its cache key.
pub struct MeshMapInputs<'a> {
    pub mesh: &'a MeshGeometry,
    pub high: Option<&'a MeshGeometry>,
    pub parts: &'a [(String, MeshGeometry)],
    /// Debug form of the high-to-low options.
    pub options: String,
    pub low_name: &'a str,
    pub cage: Option<&'a MeshGeometry>,
    pub settings: &'a MeshMapSettings,
    pub skew_map: Option<&'a TextureImage>,
}

/// The cache key of a bake of `inputs`: 16 hexadecimal digits. A new app
/// version never reuses an older version's bakes.
pub fn mesh_map_key(inputs: &MeshMapInputs<'_>) -> String {
    let mut hasher = ContentHasher::default();
    text(&mut hasher, env!("CARGO_PKG_VERSION"));
    text(&mut hasher, &format!("{:?}", inputs.settings));
    text(&mut hasher, &inputs.options);
    text(&mut hasher, inputs.low_name);
    hash_mesh(&mut hasher, Some(inputs.mesh));
    hash_mesh(&mut hasher, inputs.high);
    hash_mesh(&mut hasher, inputs.cage);
    hasher.update(&(inputs.parts.len() as u64).to_le_bytes());
    for (name, part) in inputs.parts {
        text(&mut hasher, name);
        hash_mesh(&mut hasher, Some(part));
    }
    match inputs.skew_map {
        Some(image) => {
            hasher.update(&[1]);
            hasher.update(&image.width.to_le_bytes());
            hasher.update(&image.height.to_le_bytes());
            hasher.update(&(image.rgba8.len() as u64).to_le_bytes());
            hasher.update(&image.rgba8);
        }
        None => hasher.update(&[0]),
    }
    hasher.finish().fnv1a
}

fn text(hasher: &mut ContentHasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_mesh(hasher: &mut ContentHasher, mesh: Option<&MeshGeometry>) {
    let Some(mesh) = mesh else {
        hasher.update(&[0]);
        return;
    };
    hasher.update(&[1]);
    hasher.update(&(mesh.vertices.len() as u64).to_le_bytes());
    let mut bytes = Vec::with_capacity(mesh.vertices.len() * 20);
    for vertex in &mesh.vertices {
        for value in vertex
            .position
            .to_array()
            .into_iter()
            .chain(vertex.uv.to_array())
        {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    hasher.update(&bytes);
    bytes.clear();
    hasher.update(&(mesh.triangles.len() as u64).to_le_bytes());
    for index in mesh
        .triangles
        .iter()
        .flatten()
        .chain(&mesh.triangle_materials)
    {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    hasher.update(&bytes);
    hasher.update(&(mesh.triangle_materials.len() as u64).to_le_bytes());
    hasher.update(&(mesh.material_slots.len() as u64).to_le_bytes());
    for slot in &mesh.material_slots {
        hasher.update(&(slot.len() as u64).to_le_bytes());
        hasher.update(slot.as_bytes());
    }
}

fn bake_path(folder: &Path, key: &str) -> PathBuf {
    folder
        .join(MESH_MAPS_DIR)
        .join(format!("{key}.{EXTENSION}"))
}

/// The bake stored under `key`, if any. An unreadable or corrupt file is
/// removed so it is baked afresh. A hit counts as a use for the budget.
pub fn load_mesh_maps(folder: &Path, key: &str) -> Option<Vec<MeshMaps>> {
    let path = bake_path(folder, key);
    let bytes = std::fs::read(&path).ok()?;
    match decode_mesh_maps(&bytes) {
        Ok(maps) => {
            if let Ok(file) = std::fs::File::options().write(true).open(&path) {
                let _ = file.set_modified(SystemTime::now());
            }
            Some(maps)
        }
        Err(_) => {
            let _ = std::fs::remove_file(&path);
            None
        }
    }
}

/// Stores `maps` under `key`, then trims the folder to `budget` bytes.
/// The file is written beside its final name and renamed into place, so a
/// reader never sees half of it.
pub fn store_mesh_maps(
    folder: &Path,
    key: &str,
    maps: &[MeshMaps],
    budget: u64,
) -> std::io::Result<()> {
    let path = bake_path(folder, key);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let partial = path.with_extension("partial");
    std::fs::write(&partial, encode_mesh_maps(maps))?;
    std::fs::rename(&partial, &path)?;
    prune(folder, budget, Some(&path));
    Ok(())
}

/// Stored bakes, least recently used first, with their sizes.
fn entries(folder: &Path) -> Vec<(PathBuf, u64, SystemTime)> {
    let Ok(listing) = std::fs::read_dir(folder.join(MESH_MAPS_DIR)) else {
        return Vec::new();
    };
    let mut entries = listing
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == EXTENSION))
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            let used = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            Some((entry.path(), metadata.len(), used))
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
    entries
}

/// Removes the least recently used bakes until the rest fit `budget`
/// bytes, never `keep` (the bake just stored).
pub fn prune(folder: &Path, budget: u64, keep: Option<&Path>) {
    let entries = entries(folder);
    let mut used: u64 = entries.iter().map(|entry| entry.1).sum();
    for (path, size, _) in entries {
        if used <= budget {
            break;
        }
        if Some(path.as_path()) == keep {
            continue;
        }
        if std::fs::remove_file(&path).is_ok() {
            used -= size;
        }
    }
}

/// Bytes the stored bakes take.
pub fn usage(folder: &Path) -> u64 {
    entries(folder).iter().map(|entry| entry.1).sum()
}

/// Removes every stored bake; returns the bytes released.
pub fn clear(folder: &Path) -> u64 {
    entries(folder)
        .into_iter()
        .filter(|(path, ..)| std::fs::remove_file(path).is_ok())
        .map(|entry| entry.1)
        .sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{PrimitivePreset, bake_mesh_maps};

    fn maps(mesh: &MeshGeometry, resolution: u32) -> Vec<MeshMaps> {
        let settings = MeshMapSettings {
            resolution,
            ao_samples: 2,
            ..MeshMapSettings::default()
        };
        vec![bake_mesh_maps(mesh, &settings, None).unwrap()]
    }

    fn inputs<'a>(mesh: &'a MeshGeometry, settings: &'a MeshMapSettings) -> MeshMapInputs<'a> {
        MeshMapInputs {
            mesh,
            high: None,
            parts: &[],
            options: String::new(),
            low_name: "",
            cage: None,
            settings,
            skew_map: None,
        }
    }

    #[test]
    fn keys_follow_content_not_identity() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let settings = MeshMapSettings::default();
        let key = mesh_map_key(&inputs(&cube, &settings));
        assert_eq!(key.len(), 16);
        assert_eq!(key, mesh_map_key(&inputs(&cube.clone(), &settings)));
        let mut moved = cube.clone();
        moved.vertices[0].position.x += 0.001;
        assert_ne!(key, mesh_map_key(&inputs(&moved, &settings)));
        let finer = MeshMapSettings {
            ao_samples: settings.ao_samples + 1,
            ..settings
        };
        assert_ne!(key, mesh_map_key(&inputs(&cube, &finer)));
        let high = MeshMapInputs {
            high: Some(&cube),
            ..inputs(&cube, &settings)
        };
        assert_ne!(key, mesh_map_key(&high));
    }

    #[test]
    fn stores_load_back_and_trim_least_recently_used_first() {
        let folder = tempfile::tempdir().unwrap();
        let cube = PrimitivePreset::Cube.build_mesh();
        let small = maps(&cube, 16);
        store_mesh_maps(folder.path(), "aaaa", &small, u64::MAX).unwrap();
        assert_eq!(load_mesh_maps(folder.path(), "aaaa"), Some(small.clone()));
        assert_eq!(load_mesh_maps(folder.path(), "bbbb"), None);
        let one = usage(folder.path());
        assert!(one > 0);

        // Make "aaaa" the older one, then store a second bake within a
        // budget that only fits one: the older goes, the new one stays.
        let old = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1);
        let file = std::fs::File::options()
            .write(true)
            .open(bake_path(folder.path(), "aaaa"))
            .unwrap();
        file.set_modified(old).unwrap();
        drop(file);
        store_mesh_maps(folder.path(), "cccc", &small, one).unwrap();
        assert_eq!(load_mesh_maps(folder.path(), "aaaa"), None);
        assert!(load_mesh_maps(folder.path(), "cccc").is_some());
        // A bake larger than the whole budget is still kept until the next.
        store_mesh_maps(folder.path(), "dddd", &maps(&cube, 32), one).unwrap();
        assert!(load_mesh_maps(folder.path(), "dddd").is_some());
        assert_eq!(load_mesh_maps(folder.path(), "cccc"), None);

        // A corrupt file is a miss and is removed.
        let path = bake_path(folder.path(), "eeee");
        std::fs::write(&path, b"SCMM-truncated").unwrap();
        assert_eq!(load_mesh_maps(folder.path(), "eeee"), None);
        assert!(!path.exists());

        let stored = usage(folder.path());
        assert_eq!(clear(folder.path()), stored);
        assert_eq!(usage(folder.path()), 0);
    }
}
