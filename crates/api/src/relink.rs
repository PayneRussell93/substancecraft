//! Finding moved or renamed project files so a project can be relinked
//! instead of failing to load.

use std::path::{Path, PathBuf};

use substancecraft_core::{MaterialProject, ResourceId};

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

/// Files under `roots` with the same name as `missing` (ignoring case),
/// shallowest first, then by path. Searches at most [`MAX_RELINK_DEPTH`]
/// levels and [`MAX_RELINK_SCAN`] entries in total.
pub fn find_relink_candidates(missing: &Path, roots: &[PathBuf]) -> Vec<PathBuf> {
    let Some(name) = missing
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
    else {
        return Vec::new();
    };
    let mut found: Vec<(usize, PathBuf)> = Vec::new();
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
            } else if entry.file_name().to_string_lossy().to_lowercase() == name {
                found.push((depth, path));
            }
        }
        if scanned > MAX_RELINK_SCAN {
            break;
        }
    }
    found.sort();
    found.dedup_by(|a, b| a.1 == b.1);
    found.into_iter().map(|(_, path)| path).collect()
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
}
