//! Hiding parts of the mesh in the viewport: the Select tool's selection
//! can be hidden, or everything else hidden to isolate it, so painting
//! reaches surfaces behind other parts. Hidden faces are not drawn and
//! clicks and brush strokes pass through them to what lies behind. Hiding
//! is a view setting, like solo: it never changes the project, bakes, or
//! exports, and a new mesh shows in full.

use std::borrow::Cow;
use std::collections::BTreeSet;

use substancecraft_core::MeshGeometry;

use crate::SubstanceCraftApp;

/// Faces hidden from the viewport, for the mesh revision they were picked on.
#[derive(Clone, Debug, Default)]
pub struct HiddenGeometry {
    pub triangles: BTreeSet<u32>,
    /// Mesh revision the triangles index into.
    pub mesh_revision: u64,
    /// Bumped on every change, so the renderer re-uploads.
    pub revision: u64,
}

/// `mesh` with every triangle in `hidden` collapsed to a point, so it
/// draws nothing while triangle numbering and material grouping stay.
pub fn visible_mesh<'a>(mesh: &'a MeshGeometry, hidden: &BTreeSet<u32>) -> Cow<'a, MeshGeometry> {
    if hidden.is_empty() {
        return Cow::Borrowed(mesh);
    }
    let mut visible = mesh.clone();
    for &triangle in hidden {
        if let Some(corners) = visible.triangles.get_mut(triangle as usize) {
            *corners = [corners[0]; 3];
        }
    }
    Cow::Owned(visible)
}

impl SubstanceCraftApp {
    /// Faces hidden on the current mesh (none after the mesh changed).
    pub(super) fn hidden_triangles(&self) -> &BTreeSet<u32> {
        static NONE: BTreeSet<u32> = BTreeSet::new();
        if self.hidden_geometry.mesh_revision == self.mesh_revision {
            &self.hidden_geometry.triangles
        } else {
            &NONE
        }
    }

    fn set_hidden_triangles(&mut self, triangles: BTreeSet<u32>) {
        let triangle_count = self
            .mesh_geometry
            .as_deref()
            .map_or(0, |mesh| mesh.triangles.len() as u32);
        self.hidden_geometry.triangles = triangles
            .into_iter()
            .filter(|triangle| *triangle < triangle_count)
            .collect();
        self.hidden_geometry.mesh_revision = self.mesh_revision;
        self.hidden_geometry.revision = self.hidden_geometry.revision.wrapping_add(1);
    }

    /// The faces the selected layer's polygon mask selects (all others
    /// for an inverted mask).
    fn selected_faces(&self) -> BTreeSet<u32> {
        let triangle_count = self
            .mesh_geometry
            .as_deref()
            .map_or(0, |mesh| mesh.triangles.len() as u32);
        match self
            .selected_layer()
            .and_then(|layer| layer.polygon_mask.as_ref())
        {
            Some(mask) if mask.inverted => (0..triangle_count)
                .filter(|triangle| !mask.triangles.contains(triangle))
                .collect(),
            Some(mask) => mask.triangles.clone(),
            None => BTreeSet::new(),
        }
    }

    /// Hides the selected faces too.
    pub(super) fn hide_selection(&mut self) {
        let mut hidden = self.hidden_triangles().clone();
        hidden.extend(self.selected_faces());
        self.set_hidden_triangles(hidden);
    }

    /// Hides every face but the selected ones.
    pub(super) fn isolate_selection(&mut self) {
        let selected = self.selected_faces();
        if selected.is_empty() {
            return;
        }
        let triangle_count = self
            .mesh_geometry
            .as_deref()
            .map_or(0, |mesh| mesh.triangles.len() as u32);
        self.set_hidden_triangles(
            (0..triangle_count)
                .filter(|triangle| !selected.contains(triangle))
                .collect(),
        );
    }

    /// Shows every face again.
    pub(super) fn show_all_geometry(&mut self) {
        self.set_hidden_triangles(BTreeSet::new());
    }

    /// Swaps hidden and shown faces.
    pub(super) fn invert_hidden_geometry(&mut self) {
        let triangle_count = self
            .mesh_geometry
            .as_deref()
            .map_or(0, |mesh| mesh.triangles.len() as u32);
        let hidden = self.hidden_triangles().clone();
        self.set_hidden_triangles(
            (0..triangle_count)
                .filter(|triangle| !hidden.contains(triangle))
                .collect(),
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{PolygonMask, PrimitivePreset};

    #[test]
    fn tc_0397_selected_faces_hide_and_isolate_in_the_viewport_only() {
        let mut app = SubstanceCraftApp::default();
        app.add_paint_layer();
        let triangles = app.mesh_geometry.as_deref().unwrap().triangles.len() as u32;
        app.selected_layer_mut().unwrap().polygon_mask = Some(PolygonMask {
            triangles: [0, 1].into(),
            inverted: false,
        });
        let project = app.project.clone();

        app.hide_selection();
        assert_eq!(app.hidden_triangles(), &BTreeSet::from([0, 1]));
        app.isolate_selection();
        assert_eq!(app.hidden_triangles().len() as u32, triangles - 2);
        assert!(!app.hidden_triangles().contains(&0));
        app.invert_hidden_geometry();
        assert_eq!(app.hidden_triangles(), &BTreeSet::from([0, 1]));
        assert_eq!(app.project, project, "hiding never changes the project");

        // Hidden faces draw as points and keep the mesh's numbering.
        let mesh = app.mesh_geometry.as_deref().unwrap();
        let visible = visible_mesh(mesh, app.hidden_triangles());
        assert_eq!(visible.triangles.len(), mesh.triangles.len());
        let [a, b, c] = visible.triangles[0];
        assert!(a == b && b == c);
        assert_eq!(visible.triangles[2], mesh.triangles[2]);

        // Clicks pass through hidden faces to what lies behind.
        let cube = PrimitivePreset::Cube.build_mesh();
        let projection = crate::ViewProjection {
            pose: substancecraft_core::BoneTransform::IDENTITY,
            camera_rotation: glam::Quat::IDENTITY,
            posed_center: glam::Vec3::ZERO,
            center: eframe::egui::pos2(100.0, 100.0),
            scale: 100.0,
            pan: glam::Vec2::ZERO,
        };
        let points = projection.project_mesh(&cube);
        let center = eframe::egui::pos2(110.0, 90.0);
        let front = crate::pick_triangle(&cube, &points, center, None).unwrap();
        let hidden = BTreeSet::from([front.triangle]);
        let behind = crate::pick_triangle(&cube, &points, center, Some(&hidden)).unwrap();
        assert_ne!(behind.triangle, front.triangle);

        app.show_all_geometry();
        assert!(app.hidden_triangles().is_empty());
        // A new mesh shows in full.
        app.hide_selection();
        app.mesh_revision += 1;
        assert!(app.hidden_triangles().is_empty());
    }
}
