//! Finding moved or renamed project files so a project can be relinked
//! instead of failing to load.

use std::io::Read;
use std::path::{Path, PathBuf};

use substancecraft_core::{ContentHash, ContentHasher, MaterialProject, ResourceId};

/// Directory entries scanned at most per search, so a search started from
/// a large folder stays quick.
pub const MAX_RELINK_SCAN: usize = 20_000;
/// Folder levels searched below each root.
pub const MAX_RELINK_DEPTH: usize = 6;

/// Project files that are not where the project says.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MissingFiles {
    pub resources: Vec<(ResourceId, PathBuf)>,
    pub mesh: Option<PathBuf>,
    pub high_poly: Option<PathBuf>,
}

impl MissingFiles {
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty() && self.mesh.is_none() && self.high_poly.is_none()
    }
}

/// Lists the project's files that do not exist. Relative paths resolve
/// against `base`.
pub fn missing_project_files(project: &MaterialProject, base: Option<&Path>) -> MissingFiles {
    let resolve = |source: &str| {
        let path = PathBuf::from(source);
        match base {
            Some(base) if path.is_relative() => base.join(path),
            _ => path,
        }
    };
    let missing = |source: &str| {
        let path = resolve(source);
        (!path.exists()).then_some(path)
    };
    MissingFiles {
        resources: project
            .resources
            .iter()
            .filter_map(|resource| missing(&resource.source).map(|path| (resource.id, path)))
            .collect(),
        mesh: project
            .mesh
            .as_ref()
            .filter(|mesh| !mesh.source.starts_with("builtin:"))
            .and_then(|mesh| missing(&mesh.source)),
        high_poly: project
            .mesh
            .as_ref()
            .and_then(|mesh| mesh.high_poly_source.as_deref())
            .and_then(missing),
    }
}

/// Size and FNV-1a hash of a file's bytes, read in chunks.
pub fn file_content_hash(path: &Path) -> std::io::Result<ContentHash> {
    let mut file = std::fs::File::open(path)?;
    let mut buffer = vec![0_u8; 64 * 1024];
    let mut hasher = ContentHasher::default();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(hasher.finish());
        }
        hasher.update(&buffer[..read]);
    }
}

/// Files under `roots` with the same name as `missing` (ignoring case),
/// shallowest first, then by path. Searches at most [`MAX_RELINK_DEPTH`]
/// levels and [`MAX_RELINK_SCAN`] entries in total.
pub fn find_relink_candidates(missing: &Path, roots: &[PathBuf]) -> Vec<PathBuf> {
    find_relink_candidates_by_content(missing, None, roots)
}

/// Like [`find_relink_candidates`], and when the file's `content` hash is
/// known, also renamed copies with the same bytes. Files whose content
/// matches come first (whatever their name), then files that only share
/// the name; each group shallowest first, then by path.
pub fn find_relink_candidates_by_content(
    missing: &Path,
    content: Option<&ContentHash>,
    roots: &[PathBuf],
) -> Vec<PathBuf> {
    let name = missing
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase());
    if name.is_none() && content.is_none() {
        return Vec::new();
    }
    // (rank, depth, path): rank 0 matches the content, 1 only the name.
    let mut found: Vec<(u8, usize, PathBuf)> = Vec::new();
    let mut scanned = 0;
    let mut queue: std::collections::VecDeque<(PathBuf, usize)> =
        roots.iter().map(|root| (root.clone(), 0)).collect();
    let mut visited = std::collections::BTreeSet::new();
    while let Some((folder, depth)) = queue.pop_front() {
        let Ok(canonical) = folder.canonicalize() else {
            continue;
        };
        if !visited.insert(canonical) {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&folder) else {
            continue;
        };
        for entry in entries.flatten() {
            scanned += 1;
            if scanned > MAX_RELINK_SCAN {
                break;
            }
            let path = entry.path();
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                if depth < MAX_RELINK_DEPTH {
                    queue.push_back((path, depth + 1));
                }
                continue;
            }
            let same_content = content.is_some_and(|content| {
                entry
                    .metadata()
                    .is_ok_and(|metadata| metadata.len() == content.size)
                    && file_content_hash(&path).is_ok_and(|hash| hash == *content)
            });
            let same_name = name
                .as_ref()
                .is_some_and(|name| entry.file_name().to_string_lossy().to_lowercase() == *name);
            if same_content {
                found.push((0, depth, path));
            } else if same_name {
                found.push((1, depth, path));
            }
        }
        if scanned > MAX_RELINK_SCAN {
            break;
        }
    }
    found.sort();
    found.dedup_by(|a, b| a.2 == b.2);
    found.into_iter().map(|(_, _, path)| path).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{MeshAsset, MeshId, MeshSmoothing, ProjectResource};

    #[test]
    fn tc_0249_missing_files_are_found_by_name_under_search_roots() {
        let folder = tempfile::tempdir().unwrap();
        let root = folder.path();
        std::fs::create_dir_all(root.join("textures/old")).unwrap();
        std::fs::create_dir_all(root.join("moved/deeper")).unwrap();
        std::fs::write(root.join("moved/Grunge.PNG"), b"x").unwrap();
        std::fs::write(root.join("moved/deeper/grunge.png"), b"x").unwrap();
        std::fs::write(root.join("present.png"), b"x").unwrap();
        std::fs::write(root.join("crate.fbx"), b"x").unwrap();

        let project = MaterialProject {
            mesh: Some(MeshAsset {
                id: MeshId(1),
                source: "models/crate.fbx".to_owned(),
                scale: 1.0,
                smoothing: MeshSmoothing::default(),
                high_poly_source: Some("sculpt.fbx".to_owned()),
                content_hash: None,
                high_poly_hash: None,
                high_to_low: Default::default(),
            }),
            resources: vec![
                ProjectResource::new(ResourceId(1), "Grunge", "textures/old/grunge.png"),
                ProjectResource::new(ResourceId(2), "Present", "present.png"),
            ],
            ..MaterialProject::empty()
        };
        let missing = missing_project_files(&project, Some(root));
        assert_eq!(
            missing.resources,
            [(ResourceId(1), root.join("textures/old/grunge.png"))]
        );
        assert_eq!(missing.mesh, Some(root.join("models/crate.fbx")));
        assert_eq!(missing.high_poly, Some(root.join("sculpt.fbx")));
        assert!(!missing.is_empty());

        // Shallowest match first, matching names case-insensitively.
        let candidates = find_relink_candidates(&missing.resources[0].1, &[root.to_path_buf()]);
        assert_eq!(
            candidates,
            [
                root.join("moved").join("Grunge.PNG"),
                root.join("moved").join("deeper").join("grunge.png")
            ]
        );
        assert_eq!(
            find_relink_candidates(missing.mesh.as_deref().unwrap(), &[root.to_path_buf()]),
            [root.join("crate.fbx")]
        );
        assert!(
            find_relink_candidates(&root.join("nothing.png"), &[root.to_path_buf()]).is_empty()
        );
        // Searching the same root twice does not duplicate results.
        assert_eq!(
            find_relink_candidates(
                &missing.resources[0].1,
                &[root.to_path_buf(), root.join("moved")]
            )
            .len(),
            2
        );
    }

    #[test]
    fn tc_0292_renamed_copies_are_found_by_content_before_same_named_files() {
        let folder = tempfile::tempdir().unwrap();
        let root = folder.path().to_path_buf();
        // The hash is stable, chunk-independent, and counts the size.
        let bytes = b"rust grunge pixels".repeat(5000);
        let path = root.join("grunge.png");
        std::fs::write(&path, &bytes).unwrap();
        let hash = file_content_hash(&path).unwrap();
        assert_eq!(hash, ContentHash::of(&bytes));
        assert_eq!(hash.size, bytes.len() as u64);
        assert_eq!(hash.fnv1a.len(), 16);
        let mut hasher = ContentHasher::default();
        for chunk in bytes.chunks(777) {
            hasher.update(chunk);
        }
        assert_eq!(hasher.finish(), hash);
        assert_ne!(ContentHash::of(b"a"), ContentHash::of(b"b"));

        // The file is renamed and moved; a different file takes its name
        // and another has the same size but other bytes.
        std::fs::create_dir_all(root.join("textures/old")).unwrap();
        let renamed = root.join("textures").join("rust_v2.png");
        std::fs::rename(&path, &renamed).unwrap();
        std::fs::write(root.join("textures/old/grunge.png"), b"something else").unwrap();
        let mut decoy = bytes.clone();
        decoy[0] ^= 1;
        std::fs::write(root.join("decoy.png"), &decoy).unwrap();

        let roots = [root.clone()];
        let found = find_relink_candidates_by_content(&path, Some(&hash), &roots);
        assert_eq!(
            found,
            [renamed.clone(), root.join("textures/old/grunge.png")],
            "the content match first, then the name match; never the decoy"
        );
        // Without a recorded hash only the name is used.
        assert_eq!(
            find_relink_candidates(&path, &roots),
            [root.join("textures/old/grunge.png")]
        );
    }
}
