//! Clone, smudge, and blur brushes, surface-aligned stamps, and the quick
//! mask.
//!
//! A quick mask is painted like a mask, straight into a new
//! [`Stencil`] of the selected texture set, while the viewport shows it over
//! the material. Finishing it records the stencil as one undo step, and from
//! then on every paint and mask stroke on that set is limited to it until it
//! is cleared. Stencils never change once strokes name them.

use substancecraft_core::{
    EditCommand, MaskStroke, StampFrame, Stencil, StencilId, StrokeCamera, StrokeSource,
    TextureSetId, TextureSetProperties,
};

use super::{MAX_STROKE_POINTS, SubstanceCraftApp};

/// What a paint stroke lays down.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum BrushSource {
    /// The brush's own material.
    #[default]
    Paint,
    /// The stack copied from the clone source.
    Clone,
    /// The stack dragged along the stroke.
    Smudge,
    /// The stack softened under the stroke.
    Blur,
}

impl BrushSource {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Paint => "Paint",
            Self::Clone => "Clone",
            Self::Smudge => "Smudge",
            Self::Blur => "Blur",
        }
    }
}

/// A quick mask of one texture set: being painted, or limiting strokes.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct QuickMask {
    pub(super) set: TextureSetId,
    pub(super) id: StencilId,
    pub(super) editing: bool,
    /// The set's properties before the stencil was added, for undo.
    before: Box<TextureSetProperties>,
    /// Paints the mesh's skew map rather than a mask limiting strokes: the
    /// stencil only borrows the set while it is painted.
    pub(super) skew: bool,
}

/// Brush-tool state for clone, smudge, alignment, and the quick mask.
#[derive(Clone, Debug)]
pub(super) struct StrokeTools {
    pub(super) source: BrushSource,
    /// Stamps stay round and upright on the surface instead of in UV space.
    pub(super) align_to_surface: bool,
    /// How far smudging drags, in brush radii.
    pub(super) smudge_strength: f32,
    /// Radius of the disc blurring averages, in brush radii.
    pub(super) blur_strength: f32,
    /// UV point clone strokes copy from.
    pub(super) clone_source: Option<glam::Vec2>,
    /// Offset from the brush to the source, fixed by the first stroke after
    /// the source was set so later strokes stay aligned with it.
    clone_offset: Option<glam::Vec2>,
    /// The next click on the model or UV view sets the clone source.
    pub(super) picking_source: bool,
    pub(super) quick_mask: Option<QuickMask>,
    /// Stencil stroke index of each path of the quick-mask drag in progress.
    quick_paths: Vec<Option<usize>>,
}

impl Default for StrokeTools {
    fn default() -> Self {
        Self {
            source: BrushSource::Paint,
            align_to_surface: false,
            smudge_strength: 0.5,
            blur_strength: 0.3,
            clone_source: None,
            clone_offset: None,
            picking_source: false,
            quick_mask: None,
            quick_paths: Vec::new(),
        }
    }
}

impl StrokeTools {
    /// Sets the point clone strokes copy from; the next stroke fixes the
    /// offset.
    pub(super) fn set_clone_source(&mut self, uv: glam::Vec2) {
        self.clone_source = Some(uv);
        self.clone_offset = None;
        self.picking_source = false;
    }

    /// The source of a paint stroke starting at `uv`, or `None` to paint
    /// the brush material. Clone needs a source first.
    pub(super) fn stroke_source(&mut self, uv: glam::Vec2) -> Option<StrokeSource> {
        match self.source {
            BrushSource::Paint => None,
            BrushSource::Smudge => Some(StrokeSource::Smudge {
                strength: self.smudge_strength.clamp(0.05, 1.0),
            }),
            BrushSource::Blur => Some(StrokeSource::Blur {
                strength: self.blur_strength.clamp(0.05, 1.0),
            }),
            BrushSource::Clone => {
                let source = self.clone_source?;
                let offset = *self.clone_offset.get_or_insert(source - uv);
                Some(StrokeSource::Clone { offset })
            }
        }
    }

    /// True when the brush can paint now: clone waits for a source.
    pub(super) fn ready(&self) -> bool {
        self.source != BrushSource::Clone || self.clone_source.is_some()
    }

    /// Ends the quick-mask drag in progress.
    pub(super) fn end_quick_stroke(&mut self) {
        self.quick_paths.clear();
    }
}

impl SubstanceCraftApp {
    /// The quick mask being painted on `set`, if any.
    pub(super) fn editing_quick_mask(&self, set: TextureSetId) -> Option<StencilId> {
        self.stroke_tools
            .quick_mask
            .as_ref()
            .filter(|mask| mask.editing && mask.set == set)
            .map(|mask| mask.id)
    }

    /// The quick mask new strokes on `set` are limited to: a finished one
    /// that the set still holds (undo may have removed it).
    pub(super) fn active_stencil(&self, set: TextureSetId) -> Option<StencilId> {
        let mask = self.stroke_tools.quick_mask.as_ref()?;
        (!mask.editing && mask.set == set)
            .then_some(mask.id)
            .filter(|id| {
                self.project
                    .texture_sets
                    .iter()
                    .any(|candidate| candidate.id == set && candidate.stencil(*id).is_some())
            })
    }

    /// Starts painting a quick mask on the selected set, or finishes the
    /// one being painted.
    pub(super) fn toggle_quick_mask(&mut self) {
        self.finish_active_stroke();
        match &self.stroke_tools.quick_mask {
            Some(mask) if mask.editing => self.finish_quick_mask(),
            _ => self.start_quick_mask(),
        }
    }

    /// True while the skew map is being painted.
    pub(super) fn painting_skew(&self) -> bool {
        self.stroke_tools
            .quick_mask
            .as_ref()
            .is_some_and(|mask| mask.editing && mask.skew)
    }

    /// Starts painting the mesh's skew map (white applies skew correction)
    /// over the selected set, or finishes painting it.
    pub(super) fn toggle_skew_paint(&mut self) {
        if self.painting_skew() {
            self.finish_active_stroke();
            self.finish_quick_mask();
            return;
        }
        self.clear_quick_mask();
        self.finish_active_stroke();
        let strokes = self.high_to_low_options().skew_strokes;
        self.start_quick_mask();
        let Some(mask) = self.stroke_tools.quick_mask.as_mut() else {
            return;
        };
        mask.skew = true;
        let (set_id, id) = (mask.set, mask.id);
        if let Some(stencil) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == set_id)
            .and_then(|set| set.stencils.iter_mut().find(|stencil| stencil.id == id))
        {
            stencil.strokes = strokes;
        }
        if self.paint_mode == super::PaintMode::View {
            self.set_paint_mode(super::PaintMode::Paint);
        }
        self.invalidate_preview();
    }

    fn start_quick_mask(&mut self) {
        let Some(set_id) = self.selected_texture_set else {
            return;
        };
        let Some(set) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == set_id)
        else {
            return;
        };
        if set.stencils.len() >= substancecraft_core::MAX_STENCILS {
            self.bake_error = Some("This texture set holds too many quick masks.".to_owned());
            return;
        }
        let before = Box::new(set.properties());
        let id = StencilId(
            set.stencils
                .iter()
                .map(|stencil| stencil.id.0)
                .max()
                .map_or(1, |last| last + 1),
        );
        set.stencils.push(Stencil {
            id,
            strokes: Vec::new(),
            invert: false,
        });
        self.stroke_tools.quick_mask = Some(QuickMask {
            set: set_id,
            id,
            editing: true,
            before,
            skew: false,
        });
        self.invalidate_preview();
    }

    /// Stops painting the quick mask: a painted one is recorded and limits
    /// strokes from now on; an empty one is dropped.
    fn finish_quick_mask(&mut self) {
        let Some(mut mask) = self.stroke_tools.quick_mask.take() else {
            return;
        };
        self.stroke_tools.end_quick_stroke();
        let Some(set) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == mask.set)
        else {
            return;
        };
        if mask.skew {
            // The strokes become the mesh's skew map; the set gives the
            // borrowed stencil back unrecorded.
            let strokes = set
                .stencil(mask.id)
                .map(|stencil| stencil.strokes.clone())
                .unwrap_or_default();
            set.stencils.retain(|stencil| stencil.id != mask.id);
            self.set_high_to_low(substancecraft_core::HighToLowOptions {
                skew_strokes: strokes,
                ..self.high_to_low_options()
            });
            self.invalidate_preview();
            return;
        }
        let painted = set
            .stencil(mask.id)
            .is_some_and(|stencil| !stencil.strokes.is_empty());
        if painted {
            self.history.record(EditCommand::TextureSetProperties {
                texture_set_id: mask.set,
                before: mask.before.clone(),
                after: Box::new(set.properties()),
            });
            mask.editing = false;
            self.stroke_tools.quick_mask = Some(mask);
        } else {
            set.stencils.retain(|stencil| stencil.id != mask.id);
        }
        self.invalidate_preview();
    }

    /// Stops limiting strokes to the quick mask (or discards the one being
    /// painted). Strokes already limited by it keep their shape.
    pub(super) fn clear_quick_mask(&mut self) {
        if let Some(mask) = &self.stroke_tools.quick_mask
            && mask.editing
        {
            // An unfinished quick mask was never recorded; take it out.
            let (set_id, id) = (mask.set, mask.id);
            if let Some(set) = self
                .project
                .texture_sets
                .iter_mut()
                .find(|set| set.id == set_id)
            {
                set.stencils.retain(|stencil| stencil.id != id);
            }
        }
        self.stroke_tools.quick_mask = None;
        self.stroke_tools.end_quick_stroke();
        self.invalidate_preview();
    }

    /// Flips the quick mask being painted, so strokes paint where it is
    /// black instead.
    pub(super) fn invert_quick_mask(&mut self) {
        let Some((set_id, id)) = self
            .stroke_tools
            .quick_mask
            .as_ref()
            .filter(|mask| mask.editing)
            .map(|mask| (mask.set, mask.id))
        else {
            return;
        };
        if let Some(stencil) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == set_id)
            .and_then(|set| set.stencils.iter_mut().find(|stencil| stencil.id == id))
        {
            stencil.invert = !stencil.invert;
            self.invalidate_preview();
        }
    }

    /// Adds a point to the quick mask being painted on `set`: Erase clears
    /// it, every other painting mode opens it. `point` is in UV, or in
    /// viewport pixels when painted through `camera`. Returns false when no
    /// quick mask is being painted there.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn quick_mask_point(
        &mut self,
        set_id: TextureSetId,
        path: usize,
        point: glam::Vec2,
        frame: Option<StampFrame>,
        camera: Option<StrokeCamera>,
        restart: bool,
    ) -> bool {
        let Some(id) = self.editing_quick_mask(set_id) else {
            return false;
        };
        let value = if self.paint_mode == super::PaintMode::Erase {
            0.0
        } else {
            1.0
        };
        let (radius, tip, pressure) = (self.brush_radius, self.brush_tip, self.pen_pressure);
        let Some(stencil) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == set_id)
            .and_then(|set| set.stencils.iter_mut().find(|stencil| stencil.id == id))
        else {
            // Undo took the unfinished quick mask away; stop painting it
            // rather than paint the layer by surprise.
            self.stroke_tools.quick_mask = None;
            self.stroke_tools.end_quick_stroke();
            self.invalidate_preview();
            return true;
        };

        let paths = &mut self.stroke_tools.quick_paths;
        let current = paths
            .get(path)
            .copied()
            .flatten()
            .and_then(|index| stencil.strokes.get_mut(index))
            .filter(|stroke| stroke.camera == camera && !restart);
        match current {
            Some(stroke) => {
                if stroke.points.len() >= MAX_STROKE_POINTS
                    || stroke
                        .points
                        .last()
                        .is_some_and(|last| last.distance_squared(point) < 1e-6)
                {
                    return true;
                }
                stroke.points.push(point);
                if !stroke.pressures.is_empty() {
                    stroke.pressures.push(pressure.unwrap_or(1.0));
                }
                if !stroke.frames.is_empty() {
                    stroke.frames.push(frame.unwrap_or(StampFrame::IDENTITY));
                }
            }
            None => {
                if stencil.strokes.len() >= substancecraft_core::MAX_STENCIL_STROKES {
                    return true;
                }
                stencil.strokes.push(MaskStroke {
                    tip,
                    pressures: pressure.map(|value| vec![value]).unwrap_or_default(),
                    camera,
                    frames: frame.map(|frame| vec![frame]).unwrap_or_default(),
                    ..MaskStroke::new(vec![point], radius, value)
                });
                if paths.len() <= path {
                    paths.resize(path + 1, None);
                }
                paths[path] = Some(stencil.strokes.len() - 1);
            }
        }
        // The quick-mask preview rebakes whole; it is a cheap grayscale.
        self.invalidate_preview();
        true
    }

    /// The stamp frame for a 3D hit on `triangle` when stamps align to the
    /// surface. Frames keep unit area, so they suit the downscaled preview
    /// as well as the full texture.
    pub(super) fn hit_frame(&self, triangle: u32) -> Option<StampFrame> {
        if !self.stroke_tools.align_to_surface {
            return None;
        }
        let set = self.selected_texture_set_ref()?;
        self.mesh_geometry.as_deref()?.stamp_frame(
            triangle,
            set.resolution.width,
            set.resolution.height,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{PaintMode, PreviewTarget, TriangleHit};
    use substancecraft_core::{PrimitivePreset, TextureResolution};

    /// An app with a cube, one 64² set, and a paint layer selected.
    fn app() -> SubstanceCraftApp {
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.add_paint_layer();
        for set in &mut app.project.texture_sets {
            set.resolution = TextureResolution::new(64, 64).unwrap();
        }
        app.invalidate_all_sets();
        app.set_paint_mode(PaintMode::Paint);
        app
    }

    /// The cube surface hit nearest `point`.
    fn hit(point: glam::Vec3) -> TriangleHit {
        let surface = PrimitivePreset::Cube
            .build_mesh()
            .closest_surface_point(point)
            .unwrap();
        TriangleHit {
            triangle: surface.triangle,
            uv: Some(surface.uv),
            position: surface.position,
        }
    }

    fn strokes(app: &SubstanceCraftApp) -> &[substancecraft_core::PaintStroke] {
        &app.selected_layer().unwrap().paint_strokes
    }

    #[test]
    fn tc_0352_aligned_3d_strokes_record_a_stamp_frame_per_point() {
        let mut app = app();
        app.paint_hit(hit(glam::Vec3::new(-0.1, 0.0, 0.9)));
        app.paint_hit(hit(glam::Vec3::new(0.1, 0.0, 0.9)));
        app.finish_active_stroke();
        assert!(strokes(&app)[0].frames.is_empty(), "off by default");

        app.stroke_tools.align_to_surface = true;
        app.paint_hit(hit(glam::Vec3::new(-0.1, 0.2, 0.9)));
        app.paint_hit(hit(glam::Vec3::new(0.1, 0.2, 0.9)));
        app.finish_active_stroke();
        let aligned = &strokes(&app)[1];
        assert_eq!(aligned.frames.len(), aligned.points.len());
        // A cube face spans 1.5 times more U than V per unit of surface.
        let [m00, _, _, m11] = aligned.frames[0].0;
        assert!((m00 / m11 - 1.5).abs() < 0.1, "{:?}", aligned.frames[0]);
        // UV-view strokes have no surface to align to.
        app.append_stroke_point(glam::Vec2::new(0.2, 0.2));
        app.finish_active_stroke();
        assert!(strokes(&app)[2].frames.is_empty());
    }

    #[test]
    fn tc_0354_clone_strokes_keep_the_offset_the_first_stroke_fixed() {
        let mut app = app();
        app.stroke_tools.source = BrushSource::Clone;
        // Without a source the brush does nothing.
        app.append_stroke_point(glam::Vec2::new(0.6, 0.5));
        app.finish_active_stroke();
        assert!(strokes(&app).is_empty());

        app.stroke_tools.picking_source = true;
        app.stroke_tools.set_clone_source(glam::Vec2::new(0.2, 0.5));
        assert!(!app.stroke_tools.picking_source);
        app.append_stroke_point(glam::Vec2::new(0.6, 0.5));
        app.append_stroke_point(glam::Vec2::new(0.7, 0.5));
        app.finish_active_stroke();
        app.append_stroke_point(glam::Vec2::new(0.6, 0.3));
        app.finish_active_stroke();
        let offset = |app: &SubstanceCraftApp, index: usize| {
            strokes(app)[index]
                .source
                .and_then(|source| match source {
                    StrokeSource::Clone { offset } => Some(offset),
                    StrokeSource::Smudge { .. } | StrokeSource::Blur { .. } => None,
                })
                .unwrap()
        };

        assert!((offset(&app, 0) - glam::Vec2::new(-0.4, 0.0)).length() < 1e-6);
        assert_eq!(
            offset(&app, 1),
            offset(&app, 0),
            "aligned to the first stroke"
        );
        // A new source starts a new offset.
        app.stroke_tools.set_clone_source(glam::Vec2::new(0.5, 0.9));
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.finish_active_stroke();
        assert!((offset(&app, 2) - glam::Vec2::new(0.0, 0.4)).length() < 1e-6);
        // Erasing never samples.
        app.set_paint_mode(PaintMode::Erase);
        app.append_stroke_point(glam::Vec2::new(0.3, 0.3));
        app.finish_active_stroke();
        assert_eq!(strokes(&app)[3].source, None);
        // The clone strokes bake.
        let set = app.selected_texture_set_ref().unwrap().clone();
        assert!(set.bake_base_color().is_ok());
    }

    #[test]
    fn tc_0355_smudge_strokes_record_their_strength() {
        let mut app = app();
        app.stroke_tools.source = BrushSource::Smudge;
        app.stroke_tools.smudge_strength = 0.7;
        app.append_stroke_point(glam::Vec2::new(0.4, 0.5));
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.finish_active_stroke();
        assert_eq!(
            strokes(&app)[0].source,
            Some(StrokeSource::Smudge { strength: 0.7 })
        );
        // Projection leaves smudge in texture space.
        app.projection_painting = true;
        assert!(!app.projects_strokes());
        app.stroke_tools.source = BrushSource::Paint;
        assert!(app.projects_strokes());
    }

    #[test]
    fn tc_0370_blur_strokes_record_their_strength_and_bake() {
        let mut app = app();
        app.stroke_tools.source = BrushSource::Blur;
        app.stroke_tools.blur_strength = 0.4;
        app.append_stroke_point(glam::Vec2::new(0.4, 0.5));
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.finish_active_stroke();
        assert_eq!(
            strokes(&app)[0].source,
            Some(StrokeSource::Blur { strength: 0.4 })
        );
        app.projection_painting = true;
        assert!(!app.projects_strokes(), "blur stays in texture space");
        let set = app.selected_texture_set_ref().unwrap().clone();
        assert!(set.bake_base_color().is_ok());
    }

    #[test]
    fn tc_0359_brush_paints_user_channels_and_the_viewport_previews_them() {
        let mut app = app();
        let set_id = app.selected_texture_set.unwrap();
        let wetness = substancecraft_core::UserChannel {
            slot: 2,
            name: "Wetness".to_owned(),
            gray: true,
            default: [0, 0, 0],
        };
        assert!(app.edit_set_properties(set_id, |properties| {
            properties.user_channels = vec![wetness.clone()];
        }));
        let channel = wetness.channel();
        assert_eq!(app.channel_name(channel), "Wetness");
        assert_eq!(app.user_channels(), [wetness]);
        app.brush_user.0[2] = [180; 3];
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.finish_active_stroke();
        assert_eq!(strokes(&app)[0].user.get(2), [180; 3]);
        app.set_preview_channel(channel);
        assert_eq!(app.preview_target(), PreviewTarget::Channel(channel));
        // Undo takes the definition away with the set's other properties.
        app.undo();
        app.undo();
        assert!(app.user_channels().is_empty());
        assert_eq!(app.channel_name(channel), "User 2");
    }

    #[test]
    fn tc_0377_skew_maps_are_painted_in_the_app_and_drive_the_bake() {
        let mut app = app();
        app.select_primitive(PrimitivePreset::Sphere);
        app.select_primitive(PrimitivePreset::Cube);
        let set_id = app.selected_texture_set.unwrap();
        let stencils =
            |app: &SubstanceCraftApp| app.selected_texture_set_ref().unwrap().stencils.len();
        app.projection_painting = true;

        // Painting the skew map borrows a quick mask, in texture space.
        app.toggle_skew_paint();
        assert!(app.painting_skew());
        assert!(!app.projects_strokes(), "skew strokes stay in UV space");
        let id = app.editing_quick_mask(set_id).unwrap();
        assert_eq!(app.preview_target(), PreviewTarget::QuickMask(id));
        app.append_stroke_point(glam::Vec2::new(0.2, 0.5));
        app.append_stroke_point(glam::Vec2::new(0.3, 0.5));
        app.finish_active_stroke();
        assert!(strokes(&app).is_empty(), "the layer is untouched");

        // Done: the strokes belong to the mesh, the set gives the stencil
        // back, and nothing limits later strokes.
        app.toggle_skew_paint();
        assert!(!app.painting_skew());
        assert!(app.stroke_tools.quick_mask.is_none());
        assert_eq!(stencils(&app), 0);
        let options = app.high_to_low_options();
        assert_eq!(options.skew_strokes.len(), 1);
        let image = options.skew_image(64).unwrap();
        let gray = |u: f32| (image.sample(glam::Vec2::new(u, 0.5), false).x * 255.0).round();
        assert!(gray(0.25) > 200.0, "painted white");
        assert_eq!(gray(0.8), 0.0, "black elsewhere");

        // Editing again starts from the saved strokes; discarding keeps them.
        app.toggle_skew_paint();
        let id = app.editing_quick_mask(set_id).unwrap();
        assert_eq!(
            app.selected_texture_set_ref()
                .unwrap()
                .stencil(id)
                .unwrap()
                .strokes
                .len(),
            1
        );
        app.clear_quick_mask();
        assert_eq!(stencils(&app), 0);
        assert_eq!(app.high_to_low_options().skew_strokes.len(), 1);

        // One undo step removes it.
        app.undo();
        assert!(app.high_to_low_options().skew_strokes.is_empty());
        app.redo();
        assert_eq!(app.high_to_low_options().skew_strokes.len(), 1);

        // The bake applies skew only where it was painted.
        let low = app.mesh_geometry.as_deref().unwrap().clone();
        let settings = substancecraft_core::MeshMapSettings {
            resolution: 64,
            ao_samples: 2,
            cage_distance: 0.1,
            skew: 1.0,
            ..substancecraft_core::MeshMapSettings::default()
        };
        let skew = options.skew_image(64).unwrap();
        let maps = substancecraft_core::bake_mesh_maps_corrected(
            &low,
            Some(&low),
            &settings,
            Some(&skew),
            None,
        )
        .unwrap();
        let applied = |uv: glam::Vec2| maps.sample(uv).skew;
        assert!(applied(glam::Vec2::new(0.25, 0.5)) > 0.5);
        assert!(applied(glam::Vec2::new(0.8, 0.5)) < 0.05);
    }

    #[test]
    fn tc_0356_quick_masks_are_painted_used_undone_and_cleared() {
        let mut app = app();
        let set_id = app.selected_texture_set.unwrap();
        let can_undo = app.history.can_undo();
        // An empty quick mask is dropped when finished.
        app.toggle_quick_mask();
        app.toggle_quick_mask();
        assert!(app.stroke_tools.quick_mask.is_none());
        assert!(app.selected_texture_set_ref().unwrap().stencils.is_empty());
        assert_eq!(app.history.can_undo(), can_undo, "nothing recorded");

        // Painting it goes into the stencil, not the layer.
        app.toggle_quick_mask();
        let id = app.editing_quick_mask(set_id).unwrap();
        assert_eq!(app.preview_target(), PreviewTarget::QuickMask(id));
        app.append_stroke_point(glam::Vec2::new(0.25, 0.5));
        app.append_stroke_point(glam::Vec2::new(0.3, 0.5));
        app.finish_active_stroke();
        assert!(strokes(&app).is_empty());
        let stencil = app.selected_texture_set_ref().unwrap().stencil(id).unwrap();
        assert_eq!(stencil.strokes.len(), 1);
        assert_eq!(stencil.strokes[0].points.len(), 2);
        assert_eq!(stencil.strokes[0].value, 1.0);
        // Erase closes it.
        app.set_paint_mode(PaintMode::Erase);
        app.append_stroke_point(glam::Vec2::new(0.25, 0.5));
        app.finish_active_stroke();
        let stencil = app.selected_texture_set_ref().unwrap().stencil(id).unwrap();
        assert_eq!(stencil.strokes[1].value, 0.0);
        app.set_paint_mode(PaintMode::Paint);
        app.invert_quick_mask();
        assert!(
            app.selected_texture_set_ref()
                .unwrap()
                .stencil(id)
                .unwrap()
                .invert
        );
        app.invert_quick_mask();

        // Finished, it is one undo step and limits the strokes after it.
        app.toggle_quick_mask();
        assert_eq!(app.active_stencil(set_id), Some(id));
        assert_eq!(app.preview_target(), PreviewTarget::Material);
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.finish_active_stroke();
        assert_eq!(strokes(&app)[0].stencil, Some(id));
        let set = app.selected_texture_set_ref().unwrap().clone();
        let mut project = substancecraft_core::MaterialProject::empty();
        project.texture_sets.push(set.clone());
        assert_eq!(project.validate(), Ok(()));
        assert!(set.bake_base_color().is_ok());

        // Undoing the stroke and the quick mask takes the stencil away, and
        // strokes after that are not limited.
        app.undo();
        app.undo();
        assert!(app.selected_texture_set_ref().unwrap().stencils.is_empty());
        assert_eq!(app.active_stencil(set_id), None);
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.finish_active_stroke();
        assert_eq!(strokes(&app)[0].stencil, None);

        // Clearing stops the limit; discarding drops an unfinished one.
        app.clear_quick_mask();
        app.toggle_quick_mask();
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.clear_quick_mask();
        assert!(app.stroke_tools.quick_mask.is_none());
        assert!(app.selected_texture_set_ref().unwrap().stencils.is_empty());
    }
}
