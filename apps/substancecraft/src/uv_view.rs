//! 2D UV viewport: the selected texture set's preview laid flat over the
//! mesh's UV wireframe. Brushes paint here directly in texture space.

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2, pos2, vec2};
use substancecraft_core::{BakedTexture, TextureImage};

use super::{MAX_OVERLAY_TRIANGLES, PreviewTextures, SubstanceCraftApp};
use crate::theme;

/// Edges drawn at most, so dense meshes keep the 2D view responsive.
const MAX_WIREFRAME_TRIANGLES: usize = 60_000;
const MIN_ZOOM: f32 = 0.2;
const MAX_ZOOM: f32 = 32.0;

/// Pan and zoom of the UV view, and the uploaded preview texture.
pub struct UvView {
    pub zoom: f32,
    pub pan: Vec2,
    texture: Option<(usize, egui::TextureHandle)>,
}

impl Default for UvView {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: Vec2::ZERO,
            texture: None,
        }
    }
}

impl UvView {
    /// Screen rectangle of the 0..1 UV square inside `area`.
    pub fn square(&self, area: Rect) -> Rect {
        let side = area.width().min(area.height()) * 0.9 * self.zoom;
        Rect::from_center_size(area.center() + self.pan, vec2(side, side))
    }

    /// UV coordinate under a screen position (`+V` up).
    pub fn to_uv(&self, area: Rect, position: Pos2) -> glam::Vec2 {
        let square = self.square(area);
        glam::Vec2::new(
            (position.x - square.left()) / square.width(),
            1.0 - (position.y - square.top()) / square.height(),
        )
    }

    fn to_screen(square: Rect, uv: glam::Vec2) -> Pos2 {
        pos2(
            square.left() + uv.x * square.width(),
            square.top() + (1.0 - uv.y) * square.height(),
        )
    }

    /// Zooms by `factor` keeping the UV point under `anchor` fixed.
    fn zoom_about(&mut self, area: Rect, anchor: Pos2, factor: f32) {
        let zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let applied = zoom / self.zoom;
        let offset = anchor - (area.center() + self.pan);
        self.pan -= offset * (applied - 1.0);
        self.zoom = zoom;
    }
}

impl SubstanceCraftApp {
    pub(super) fn uv_viewport(&mut self, ui: &mut egui::Ui, area: Rect) {
        let response = ui.interact(area, ui.id().with("uv_viewport"), Sense::click_and_drag());
        let painting = self.paint_mode.paints();
        let (shift, middle, scroll) = ui.input(|input| {
            (
                input.modifiers.shift,
                input.pointer.middle_down(),
                input.smooth_scroll_delta.y,
            )
        });
        let pointer = ui.input(|input| input.pointer.interact_pos());

        if self.stroke_tools.picking_source {
            if response.clicked()
                && let Some(position) = pointer
            {
                let uv = self.uv_view.to_uv(area, position);
                if self.paintable_uv(uv) {
                    self.stroke_tools.set_clone_source(uv);
                }
            }
        } else if response.dragged() && (!painting || shift || middle) {
            self.uv_view.pan += ui.input(|input| input.pointer.delta());
        } else if painting
            && response.dragged()
            && let Some(position) = pointer
        {
            let uv = self.uv_view.to_uv(area, position);
            if self.paintable_uv(uv) {
                self.append_stroke_point(uv);
                // Radial copies have no UV-space equivalent; only mirroring
                // applies here.
                if self.symmetry.is_some_and(|symmetry| !symmetry.is_radial()) {
                    self.stroke_point(1, substancecraft_core::Symmetry::mirror_uv(uv), None, false);
                }
            }
        }
        if response.hovered()
            && scroll != 0.0
            && let Some(anchor) = ui.input(|input| input.pointer.hover_pos())
        {
            self.uv_view
                .zoom_about(area, anchor, (scroll * 0.002).exp());
        }

        let painter = ui.painter_at(area);
        painter.rect_filled(area, 0.0, theme::VIEWPORT_BG);
        let square = self.uv_view.square(area);
        let texture = self.uv_preview_texture(ui.ctx());
        // Each UDIM tile sits in its own unit square, showing its cell of
        // the atlas; without tiles the preview fills the 0..1 square.
        let tiles: Vec<(Option<substancecraft_core::UdimTile>, Rect, Rect)> =
            match &self.udim_atlas {
                Some(atlas) => {
                    let (columns, rows) = atlas.grid();
                    atlas
                        .tiles()
                        .iter()
                        .enumerate()
                        .map(|(index, tile)| {
                            let offset = tile.offset();
                            let rect = square.translate(vec2(
                                offset.x * square.width(),
                                -offset.y * square.height(),
                            ));
                            let (column, row) = (index as u32 % columns, index as u32 / columns);
                            let cell = Rect::from_min_size(
                                pos2(
                                    column as f32 / columns as f32,
                                    (rows - 1 - row) as f32 / rows as f32,
                                ),
                                vec2(1.0 / columns as f32, 1.0 / rows as f32),
                            );
                            (Some(*tile), rect, cell)
                        })
                        .collect()
                }
                None => vec![(
                    None,
                    square,
                    Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
                )],
            };
        for (tile, rect, cell) in &tiles {
            paint_checker(&painter, *rect);
            if let Some(texture) = &texture {
                painter.image(texture.id(), *rect, *cell, Color32::WHITE);
            }
            painter.rect_stroke(
                *rect,
                0.0,
                Stroke::new(1.0, Color32::from_white_alpha(60)),
                egui::StrokeKind::Outside,
            );
            if let Some(tile) = tile {
                painter.text(
                    rect.left_top() + vec2(6.0, 6.0),
                    egui::Align2::LEFT_TOP,
                    tile.0.to_string(),
                    egui::FontId::proportional(11.0),
                    theme::TEXT_WEAK,
                );
            }
        }
        self.paint_uv_wireframe(&painter, square);

        let label = self.selected_texture_set_ref().map_or_else(
            || "UV — no texture set".to_owned(),
            |set| format!("UV · {}", set.name),
        );
        painter.text(
            area.left_top() + vec2(12.0, area.height() - 22.0),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::proportional(12.0),
            theme::TEXT_WEAK,
        );

        if painting
            && response.hovered()
            && let Some(position) = ui.input(|input| input.pointer.hover_pos())
        {
            let texture_width = self
                .selected_texture_set_ref()
                .map_or(2048.0, |set| set.resolution.width as f32);
            let radius = (self.brush_radius / texture_width * square.width()).max(2.0);
            painter.circle_stroke(position, radius, Stroke::new(1.5, Color32::WHITE));
            painter.circle_filled(position, 1.5, Color32::WHITE);
        }
    }

    /// Uploads the displayed preview to egui when it changes.
    fn uv_preview_texture(&mut self, context: &egui::Context) -> Option<egui::TextureHandle> {
        let baked: &BakedTexture = match self.displayed_preview()? {
            PreviewTextures::Material(textures) => &textures.base_color,
            PreviewTextures::Single(texture) => texture,
        };
        let key = baked.rgba8.as_ptr() as usize ^ (baked.width as usize).rotate_left(11);
        if let Some((current, texture)) = &self.uv_view.texture
            && *current == key
        {
            return Some(texture.clone());
        }
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [baked.width as usize, baked.height as usize],
            &baked.rgba8,
        );
        let texture = match self.uv_view.texture.take() {
            Some((_, mut texture)) => {
                texture.set(image, egui::TextureOptions::LINEAR);
                texture
            }
            None => context.load_texture("uv-preview", image, egui::TextureOptions::LINEAR),
        };
        self.uv_view.texture = Some((key, texture.clone()));
        Some(texture)
    }

    /// UV edges of the selected texture set's triangles.
    fn paint_uv_wireframe(&self, painter: &egui::Painter, square: Rect) {
        let Some(mesh) = self.mesh_geometry.as_deref() else {
            return;
        };
        let stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(120, 220, 255, 90));
        let limit = MAX_WIREFRAME_TRIANGLES.min(MAX_OVERLAY_TRIANGLES);
        let mut drawn = 0;
        for (index, triangle) in mesh.triangles.iter().enumerate() {
            if !self.triangle_in_selected_set(index as u32) {
                continue;
            }
            if drawn >= limit {
                break;
            }
            drawn += 1;
            let corners = triangle.map(|corner| {
                mesh.vertices
                    .get(corner as usize)
                    .map_or(Pos2::ZERO, |vertex| UvView::to_screen(square, vertex.uv))
            });
            painter.add(egui::Shape::closed_line(corners.to_vec(), stroke));
        }
    }
}

/// Light checkerboard behind the texture so transparency reads clearly.
fn paint_checker(painter: &egui::Painter, square: Rect) {
    painter.rect_filled(square, 0.0, Color32::from_gray(52));
    let cells = 16;
    let size = square.width() / cells as f32;
    if size < 2.0 {
        return;
    }
    for row in 0..cells {
        for column in 0..cells {
            if (row + column) % 2 == 0 {
                continue;
            }
            let min = square.min + vec2(column as f32 * size, row as f32 * size);
            painter.rect_filled(
                Rect::from_min_size(min, vec2(size, size)),
                0.0,
                Color32::from_gray(64),
            );
        }
    }
}

/// Downsampled (nearest) copy of a bitmap whose longer edge is at most
/// `max_edge` pixels.
pub fn thumbnail_image(image: &TextureImage, max_edge: u32) -> egui::ColorImage {
    let scale = (max_edge as f32 / image.width.max(image.height) as f32).min(1.0);
    let width = ((image.width as f32 * scale).round() as usize).max(1);
    let height = ((image.height as f32 * scale).round() as usize).max(1);
    let mut pixels = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        let source_y = (y * image.height as usize / height).min(image.height as usize - 1);
        for x in 0..width {
            let source_x = (x * image.width as usize / width).min(image.width as usize - 1);
            let index = (source_y * image.width as usize + source_x) * 4;
            pixels.extend_from_slice(image.rgba8.get(index..index + 4).unwrap_or(&[0; 4]));
        }
    }
    egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tc_0100_uv_view_maps_screen_points_to_uvs_and_zooms_about_the_cursor() {
        let area = Rect::from_min_size(Pos2::ZERO, vec2(200.0, 100.0));
        let mut view = UvView::default();
        let square = view.square(area);
        assert_eq!(
            square,
            Rect::from_center_size(pos2(100.0, 50.0), vec2(90.0, 90.0))
        );
        let uv = view.to_uv(area, square.left_bottom());
        assert!(uv.abs_diff_eq(glam::Vec2::ZERO, 1e-5));
        let uv = view.to_uv(area, square.right_top());
        assert!(uv.abs_diff_eq(glam::Vec2::ONE, 1e-5));

        let anchor = pos2(130.0, 40.0);
        let before = view.to_uv(area, anchor);
        view.zoom_about(area, anchor, 2.0);
        assert!(view.to_uv(area, anchor).abs_diff_eq(before, 1e-4));
        assert_eq!(view.zoom, 2.0);
    }
}
