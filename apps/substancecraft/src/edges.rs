//! Viewport overlay of UV seams, hard (split-normal) edges, and tangent
//! frame splits, for diagnosing normal-map problems.

use std::sync::Arc;

use eframe::egui::{self, Color32, RichText, Ui};
use substancecraft_core::{DiagnosticEdge, OverlayPalette, diagnostic_edges};

use super::{SubstanceCraftApp, ViewProjection};
use crate::{theme, widgets};

/// Edges drawn at most, to keep dense meshes responsive.
const MAX_OVERLAY_EDGES: usize = 100_000;
/// What an overlay edge is drawn as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    Seam = 0,
    Hard = 1,
    Tangent = 2,
}

/// Seam, hard-edge, and tangent-split colours for a colour-vision palette.
pub fn edge_colors(palette: OverlayPalette) -> [Color32; 3] {
    match palette {
        OverlayPalette::Standard => [
            Color32::from_rgb(80, 200, 255),
            Color32::from_rgb(255, 170, 60),
            Color32::from_rgb(230, 90, 200),
        ],
        // Okabe–Ito sky blue, orange, and yellow.
        OverlayPalette::RedGreenSafe => [
            Color32::from_rgb(86, 180, 233),
            Color32::from_rgb(230, 159, 0),
            Color32::from_rgb(240, 228, 66),
        ],
        // Okabe–Ito bluish green, vermillion, and reddish purple.
        OverlayPalette::BlueYellowSafe => [
            Color32::from_rgb(0, 158, 115),
            Color32::from_rgb(213, 94, 0),
            Color32::from_rgb(204, 121, 167),
        ],
    }
}

/// Line width and dash for an edge kind. With `pattern` on, kinds differ
/// by shape as well as colour: hard edges thick, tangent splits dashed.
pub fn edge_style(kind: EdgeKind, pattern: bool) -> (f32, bool) {
    match (kind, pattern) {
        (EdgeKind::Hard, true) => (4.0, false),
        (EdgeKind::Tangent, true) => (2.0, true),
        _ => (2.0, false),
    }
}

#[derive(Default)]
pub struct EdgeOverlay {
    pub seams: bool,
    pub hard: bool,
    pub tangents: bool,
    /// Edges of the current display mesh, keyed by mesh revision and
    /// smoothing angle.
    cache: Option<(u64, u32, Arc<Vec<DiagnosticEdge>>)>,
}

impl EdgeOverlay {
    pub fn any(&self) -> bool {
        self.seams || self.hard || self.tangents
    }

    /// How an edge draws, or `None` when none of its kinds is on. Hard
    /// edges win over seams, seams over tangent splits.
    pub fn kind(&self, edge: &DiagnosticEdge) -> Option<EdgeKind> {
        if self.hard && edge.flags.hard {
            Some(EdgeKind::Hard)
        } else if self.seams && edge.flags.uv_seam {
            Some(EdgeKind::Seam)
        } else if self.tangents && edge.flags.tangent_split {
            Some(EdgeKind::Tangent)
        } else {
            None
        }
    }
}

impl SubstanceCraftApp {
    /// Diagnostic edges of the display mesh, recomputed when it changes.
    pub(super) fn overlay_edges(&mut self) -> Option<Arc<Vec<DiagnosticEdge>>> {
        let mesh = self.mesh_geometry.clone()?;
        let angle = self.mesh_smoothing.angle_degrees;
        let key = (self.mesh_revision, angle.to_bits());
        if let Some((revision, angle_bits, edges)) = &self.edge_overlay.cache
            && (*revision, *angle_bits) == key
        {
            return Some(Arc::clone(edges));
        }
        let edges = Arc::new(diagnostic_edges(&mesh, angle));
        self.edge_overlay.cache = Some((key.0, key.1, Arc::clone(&edges)));
        Some(edges)
    }

    /// Draws the enabled edge kinds over the model, skipping edges whose
    /// faces both point away from the camera.
    pub(super) fn draw_edge_overlay(
        &mut self,
        painter: &egui::Painter,
        projection: &ViewProjection,
    ) {
        if !self.edge_overlay.any() {
            return;
        }
        let (Some(edges), Some(mesh)) = (self.overlay_edges(), self.mesh_geometry.clone()) else {
            return;
        };
        let facing = |triangle: u32| {
            mesh.triangles
                .get(triangle as usize)
                .and_then(|corners| {
                    let [a, b, c] = corners.map(|vertex| {
                        mesh.vertices
                            .get(vertex as usize)
                            .map(|vertex| projection.project(vertex.position).view)
                    });
                    Some((b? - a?).cross(c? - a?).z > 0.0)
                })
                .unwrap_or(false)
        };
        let colors = edge_colors(self.preferences.values.overlay_palette);
        let pattern = self.preferences.values.overlay_pattern;
        for edge in edges.iter().take(MAX_OVERLAY_EDGES) {
            let Some(kind) = self.edge_overlay.kind(edge) else {
                continue;
            };
            if !facing(edge.faces[0]) && !facing(edge.faces[1]) {
                continue;
            }
            let (width, dashed) = edge_style(kind, pattern);
            let stroke = egui::Stroke::new(width, colors[kind as usize]);
            let points = [
                projection.project(edge.a).screen,
                projection.project(edge.b).screen,
            ];
            if dashed {
                painter.extend(egui::Shape::dashed_line(&points, stroke, 6.0, 4.0));
            } else {
                painter.line_segment(points, stroke);
            }
        }
    }

    /// Toggles for the overlay, in the Light drawer.
    pub(super) fn edge_overlay_controls(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Mesh overlays");
        let [seam, hard, tangent] = edge_colors(self.preferences.values.overlay_palette);
        for (label, color, value, tip) in [
            (
                "UV seams",
                seam,
                &mut self.edge_overlay.seams,
                "Edges where the two faces use different UVs",
            ),
            (
                "Hard edges",
                hard,
                &mut self.edge_overlay.hard,
                "Edges sharper than the smoothing angle, with split normals",
            ),
            (
                "Tangent splits",
                tangent,
                &mut self.edge_overlay.tangents,
                "Edges where tangent frames differ, which can show in normal maps",
            ),
        ] {
            widgets::property_row(ui, label, |ui| {
                widgets::toggle(ui, value).on_hover_text(tip);
                let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 4.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 2.0, color);
            });
        }
        if self.edge_overlay.any()
            && let Some(edges) = self.overlay_edges()
        {
            let count =
                |kind: fn(&DiagnosticEdge) -> bool| edges.iter().filter(|edge| kind(edge)).count();
            ui.label(
                RichText::new(format!(
                    "{} seam, {} hard, {} tangent-split edges",
                    count(|edge| edge.flags.uv_seam),
                    count(|edge| edge.flags.hard),
                    count(|edge| edge.flags.tangent_split),
                ))
                .size(11.5)
                .color(theme::TEXT_FAINT),
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::PrimitivePreset;

    #[test]
    fn tc_0248_edge_overlay_caches_per_mesh_and_colors_by_kind() {
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        let first = app.overlay_edges().unwrap();
        assert_eq!(first.len(), 12);
        // Unchanged mesh: the same cached list.
        assert!(Arc::ptr_eq(&first, &app.overlay_edges().unwrap()));
        app.set_base_geometry(Some(PrimitivePreset::Plane.build_mesh()));
        assert!(app.overlay_edges().unwrap().is_empty());

        let edge = first[0];
        let mut overlay = EdgeOverlay::default();
        assert_eq!(overlay.kind(&edge), None);
        overlay.seams = true;
        assert_eq!(overlay.kind(&edge), Some(EdgeKind::Seam));
        overlay.hard = true;
        assert_eq!(overlay.kind(&edge), Some(EdgeKind::Hard));
        // Each palette keeps the three kinds apart; the pattern adds shape.
        for palette in OverlayPalette::ALL {
            let [a, b, c] = edge_colors(palette);
            assert!(a != b && b != c && a != c);
        }
        assert_eq!(edge_style(EdgeKind::Tangent, true), (2.0, true));
        assert_eq!(edge_style(EdgeKind::Hard, true), (4.0, false));
        assert_eq!(edge_style(EdgeKind::Hard, false), (2.0, false));
    }
}
