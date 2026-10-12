//! Relinking moved or renamed mesh and bitmap files. Missing files are
//! looked up by name near the project; bitmap relinks are undoable edits,
//! and every relink is written back on the next save.

use std::path::{Path, PathBuf};

use substancecraft_api::{find_relink_candidates, missing_project_files};
use substancecraft_core::{EditCommand, MaterialProject, ResourceId};

use super::{SubstanceCraftApp, path_label};

/// Folders searched for a missing file: the project's folder and its
/// parent.
pub fn search_roots(project_path: Option<&Path>) -> Vec<PathBuf> {
    let Some(folder) = project_path.and_then(Path::parent) else {
        return Vec::new();
    };
    let mut roots = vec![folder.to_path_buf()];
    if let Some(parent) = folder.parent() {
        roots.push(parent.to_path_buf());
    }
    roots
}

/// Points a missing mesh or high-poly source at the first file with its
/// name under `roots`. Returns one note per relinked file.
pub fn relink_mesh_sources(project: &mut MaterialProject, roots: &[PathBuf]) -> Vec<String> {
    let missing = missing_project_files(project, None);
    let mut notes = Vec::new();
    let Some(mesh) = project.mesh.as_mut() else {
        return notes;
    };
    if let Some(path) = missing.mesh
        && let Some(found) = find_relink_candidates(&path, roots).into_iter().next()
    {
        notes.push(format!("mesh → {}", found.display()));
        mesh.source = found.to_string_lossy().into_owned();
    }
    if let Some(path) = missing.high_poly
        && let Some(found) = find_relink_candidates(&path, roots).into_iter().next()
    {
        notes.push(format!("high-poly mesh → {}", found.display()));
        mesh.high_poly_source = Some(found.to_string_lossy().into_owned());
    }
    notes
}

impl SubstanceCraftApp {
    /// Relinks every missing bitmap that has a same-named file near the
    /// project, as one undoable edit. Returns how many were relinked.
    pub(super) fn relink_missing_resources(&mut self) -> usize {
        let roots = search_roots(self.project_path.as_deref());
        let missing = missing_project_files(&self.project, None).resources;
        let commands: Vec<EditCommand> = missing
            .into_iter()
            .filter_map(|(id, path)| {
                let found = find_relink_candidates(&path, &roots).into_iter().next()?;
                Some(EditCommand::RelinkResource {
                    resource: id,
                    before: self.resource_source(id)?,
                    after: found.to_string_lossy().into_owned(),
                })
            })
            .collect();
        let count = commands.len();
        if count > 0
            && self
                .history
                .apply(&mut self.project, EditCommand::Batch(commands))
        {
            self.reload_resources();
            return count;
        }
        0
    }

    /// Points one bitmap resource at `path`, undoably.
    pub(super) fn relink_resource(&mut self, id: ResourceId, path: &Path) {
        let Some(before) = self.resource_source(id) else {
            return;
        };
        let command = EditCommand::RelinkResource {
            resource: id,
            before,
            after: path.to_string_lossy().into_owned(),
        };
        if self.history.apply(&mut self.project, command) {
            self.reload_resources();
        }
    }

    /// Asks for a replacement file for a missing bitmap.
    pub(super) fn choose_relink(&mut self, id: ResourceId) {
        let name = self
            .resource_source(id)
            .map(|source| path_label(Path::new(&source)))
            .unwrap_or_default();
        if let Some(path) = rfd::FileDialog::new()
            .set_title(format!("Locate {name}"))
            .add_filter("Images", &super::IMAGE_EXTENSIONS)
            .pick_file()
        {
            self.relink_resource(id, &path);
        }
    }

    fn resource_source(&self, id: ResourceId) -> Option<String> {
        self.project
            .resources
            .iter()
            .find(|resource| resource.id == id)
            .map(|resource| resource.source.clone())
    }

    /// Reloads bitmaps after undo or redo changed where any of them live.
    pub(super) fn reload_relinked_resources(&mut self) {
        let sources: Vec<(ResourceId, String)> = self
            .project
            .resources
            .iter()
            .map(|resource| (resource.id, resource.source.clone()))
            .collect();
        if sources != self.loaded_resource_sources {
            self.reload_resources();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tc_0250_moved_meshes_and_bitmaps_relink_by_name_and_undo() {
        let folder = tempfile::tempdir().unwrap();
        let root = folder.path();
        let scene = root.join("scene");
        std::fs::create_dir_all(scene.join("moved")).unwrap();
        let bitmap = scene.join("grunge.png");
        let pixel = substancecraft_core::BakedTexture {
            width: 1,
            height: 1,
            rgba8: vec![200, 200, 200, 255],
        };
        substancecraft_api::write_texture(&bitmap, &pixel).unwrap();

        let mut app = SubstanceCraftApp {
            project_path: Some(scene.join("scene.sproject")),
            ..SubstanceCraftApp::default()
        };
        let id = app.add_resource(&bitmap).unwrap();
        // The bitmap moves into a subfolder.
        let moved = scene.join("moved").join("grunge.png");
        std::fs::rename(&bitmap, &moved).unwrap();
        app.reload_resources();
        assert!(!app.resource_images.contains_key(&id));

        assert_eq!(app.relink_missing_resources(), 1);
        assert!(app.resource_images.contains_key(&id));
        assert_eq!(app.resource_source(id).unwrap(), moved.to_string_lossy());
        // Undo puts the old path back (and the bitmap goes missing again).
        app.undo();
        assert_eq!(app.resource_source(id).unwrap(), bitmap.to_string_lossy());
        assert!(!app.resource_images.contains_key(&id));
        app.redo();
        assert!(app.resource_images.contains_key(&id));
        // Nothing left to relink.
        assert_eq!(app.relink_missing_resources(), 0);

        // A missing mesh source is found by name next to the project.
        std::fs::write(scene.join("crate.fbx"), b"fbx").unwrap();
        let mut project = MaterialProject {
            mesh: Some(substancecraft_core::MeshAsset {
                id: substancecraft_core::MeshId(1),
                source: root
                    .join("old")
                    .join("crate.fbx")
                    .to_string_lossy()
                    .into_owned(),
                scale: 1.0,
                smoothing: substancecraft_core::MeshSmoothing::default(),
                high_poly_source: None,
            }),
            ..MaterialProject::empty()
        };
        let notes = relink_mesh_sources(&mut project, &search_roots(app.project_path.as_deref()));
        assert_eq!(notes.len(), 1);
        assert_eq!(
            project.mesh.unwrap().source,
            root.join("scene").join("crate.fbx").to_string_lossy()
        );
    }
}
