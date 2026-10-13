//! Posing the viewport mesh with the rig imported with it: the bind pose
//! (the mesh as imported), the rig's rest pose, or a frame of a clip, which
//! can play. Painting, picking, and bakes keep using the bind mesh; only
//! what the viewport shows, and where the pointer lands on it, moves.

use std::sync::Arc;
use std::time::Instant;

use eframe::egui::{self, Ui};
use glam::Vec3;
use substancecraft_core::{AnimationFrame, ClipSample, MeshGeometry, Rig};

use super::{SubstanceCraftApp, widgets};

/// What the viewport mesh is posed by.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) enum PoseSource {
    /// The mesh as imported.
    #[default]
    Bind,
    /// The rig's rest transforms.
    Rest,
    /// A clip, by name, at [`AnimationState::time`].
    Clip(String),
}

#[derive(Clone, Debug, Default)]
pub(super) struct AnimationState {
    pub(super) source: PoseSource,
    /// Seconds into the clip.
    pub(super) time: f32,
    pub(super) playing: bool,
    /// Lets root joints travel with the clip instead of playing in place.
    pub(super) root_motion: bool,
    last_tick: Option<Instant>,
}

/// What a posed mesh was computed for.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PoseKey {
    mesh_revision: u64,
    source: PoseSource,
    time_bits: u32,
    root_motion: bool,
}

pub(super) struct PosedMesh {
    key: PoseKey,
    /// `None` when the mesh could not be posed (it then shows bound).
    geometry: Option<Arc<MeshGeometry>>,
}

impl SubstanceCraftApp {
    /// The rig of the shown imported mesh, when it fits that mesh.
    pub(super) fn current_rig(&self) -> Option<&Rig> {
        let rig = self
            .imported_meshes
            .get(self.selected_imported_mesh?)?
            .rig
            .as_ref()?;
        let base = self.base_geometry.as_ref()?;
        (rig.influences.len() == base.vertices.len()).then_some(rig)
    }

    /// The pose the viewport should show, or `None` for the bind pose.
    pub(super) fn pose_key(&self) -> Option<PoseKey> {
        if self.animation.source == PoseSource::Bind || self.current_rig().is_none() {
            return None;
        }
        Some(PoseKey {
            mesh_revision: self.mesh_revision,
            source: self.animation.source.clone(),
            time_bits: self.animation.time.to_bits(),
            root_motion: self.animation.root_motion,
        })
    }

    /// True while the viewport shows anything but the bind pose.
    pub(super) fn is_posed(&self) -> bool {
        self.pose_key().is_some()
    }

    /// Brings the posed mesh up to date and returns the mesh the viewport
    /// shows: posed, or the bind mesh.
    pub(super) fn display_mesh(&mut self) -> Option<Arc<MeshGeometry>> {
        let Some(key) = self.pose_key() else {
            return self.mesh_geometry.clone();
        };
        if self
            .posed_mesh
            .as_ref()
            .is_none_or(|posed| posed.key != key)
        {
            let geometry = self.pose_mesh(&key.source).map(Arc::new);
            self.posed_mesh = Some(PosedMesh { key, geometry });
        }
        self.shown_mesh()
    }

    /// The mesh the viewport shows, as last brought up to date by
    /// [`Self::display_mesh`].
    pub(super) fn shown_mesh(&self) -> Option<Arc<MeshGeometry>> {
        match (self.pose_key(), &self.posed_mesh) {
            (Some(key), Some(posed)) if posed.key == key => posed
                .geometry
                .clone()
                .or_else(|| self.mesh_geometry.clone()),
            _ => self.mesh_geometry.clone(),
        }
    }

    /// The bind mesh posed by `source`: the scaled imported mesh is skinned,
    /// then smoothed and subdivided like the bind mesh, so both share their
    /// topology.
    fn pose_mesh(&self, source: &PoseSource) -> Option<MeshGeometry> {
        let rig = self.current_rig()?;
        let base = self.base_geometry.as_ref()?;
        let bind = self.mesh_geometry.as_deref()?;
        let sample = match source {
            PoseSource::Bind => return None,
            PoseSource::Rest => None,
            PoseSource::Clip(name) => Some(ClipSample {
                clip: rig.clip(name)?,
                time: self.animation.time,
                in_place: !self.animation.root_motion,
            }),
        };
        let positions: Vec<Vec3> = base.vertices.iter().map(|vertex| vertex.position).collect();
        let posed = rig
            .pose_positions(&positions, self.import_scale, sample)
            .ok()?;
        let mut posed_base = base.clone();
        for (vertex, position) in posed_base.vertices.iter_mut().zip(posed) {
            vertex.position = position;
        }
        let posed = self.mesh_smoothing.apply(&posed_base).ok()?;
        (posed.vertices.len() == bind.vertices.len() && posed.triangles == bind.triangles)
            .then_some(posed)
    }

    /// Duration of the clip being shown.
    fn shown_clip_duration(&self) -> Option<f32> {
        match &self.animation.source {
            PoseSource::Clip(name) => self.current_rig()?.clip(name).map(|clip| clip.duration),
            _ => None,
        }
    }

    /// Moves a playing clip on by the time since the last frame, looping.
    pub(super) fn advance_animation(&mut self, context: &egui::Context) {
        if !self.animation.playing {
            self.animation.last_tick = None;
            return;
        }
        let Some(duration) = self.shown_clip_duration() else {
            self.animation.playing = false;
            self.animation.last_tick = None;
            return;
        };
        let now = Instant::now();
        if let Some(last) = self.animation.last_tick {
            let elapsed = now.duration_since(last).as_secs_f32();
            self.animation.time = if duration > 0.0 {
                (self.animation.time + elapsed).rem_euclid(duration)
            } else {
                0.0
            };
        }
        self.animation.last_tick = Some(now);
        context.request_repaint();
    }

    /// The shown pose as a saved pose stores it.
    pub(super) fn animation_frame(&self) -> Option<AnimationFrame> {
        // Only a mesh with a rig has frames to keep.
        self.current_rig()?;
        match &self.animation.source {
            PoseSource::Bind => None,
            PoseSource::Rest => Some(AnimationFrame {
                clip: None,
                time: 0.0,
            }),
            PoseSource::Clip(name) => Some(AnimationFrame {
                clip: Some(name.clone()),
                time: self.animation.time,
            }),
        }
    }

    /// Shows a saved pose's frame (stopping playback). A clip the mesh does
    /// not have shows the bind pose.
    pub(super) fn show_animation_frame(&mut self, frame: Option<&AnimationFrame>) {
        self.animation.playing = false;
        let source = match frame {
            None => PoseSource::Bind,
            Some(AnimationFrame { clip: None, .. }) => PoseSource::Rest,
            Some(AnimationFrame {
                clip: Some(name), ..
            }) if self
                .current_rig()
                .is_some_and(|rig| rig.clip(name).is_some()) =>
            {
                PoseSource::Clip(name.clone())
            }
            Some(_) => PoseSource::Bind,
        };
        self.animation.time = match (&source, frame) {
            (PoseSource::Clip(_), Some(frame)) => frame.time,
            _ => 0.0,
        };
        self.animation.source = source;
    }

    /// Bind, rest, and clip choices, the timeline, and playback, in the
    /// Poses drawer. Nothing shows for meshes without a rig.
    pub(super) fn animation_section(&mut self, ui: &mut Ui) {
        let Some(rig) = self.current_rig() else {
            return;
        };
        let clips: Vec<(String, f32)> = rig
            .clips
            .iter()
            .map(|clip| (clip.name.clone(), clip.duration))
            .collect();
        let joints = rig.joints.len();
        widgets::caption(ui, "Animation");
        ui.label(
            egui::RichText::new(format!(
                "{joints} joints, {} clip{}",
                clips.len(),
                if clips.len() == 1 { "" } else { "s" }
            ))
            .size(11.5)
            .color(super::theme::TEXT_WEAK),
        );
        let mut chosen = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
            let current = &self.animation.source;
            if widgets::chip(ui, "Bind pose", *current == PoseSource::Bind)
                .on_hover_text("The mesh as imported, as painting and bakes see it")
                .clicked()
            {
                chosen = Some(PoseSource::Bind);
            }
            if widgets::chip(ui, "Rest pose", *current == PoseSource::Rest)
                .on_hover_text("The skeleton's rest transforms")
                .clicked()
            {
                chosen = Some(PoseSource::Rest);
            }
            for (name, _) in &clips {
                let source = PoseSource::Clip(name.clone());
                if widgets::chip(ui, name, *current == source).clicked() {
                    chosen = Some(source);
                }
            }
        });
        if let Some(source) = chosen
            && source != self.animation.source
        {
            self.animation.source = source;
            self.animation.time = 0.0;
            self.animation.playing = false;
            self.pose_transition = None;
            self.selected_pose = None;
        }
        if let PoseSource::Clip(name) = self.animation.source.clone() {
            let duration = clips
                .iter()
                .find(|(clip, _)| *clip == name)
                .map_or(0.0, |(_, duration)| *duration);
            let mut time = self.animation.time;
            widgets::property_row(ui, "Time", |ui| {
                ui.spacing_mut().slider_width = (ui.available_width() - 60.0).max(48.0);
                if ui
                    .add(
                        egui::Slider::new(&mut time, 0.0..=duration.max(0.0))
                            .suffix(" s")
                            .fixed_decimals(2),
                    )
                    .changed()
                {
                    self.animation.playing = false;
                }
            });
            self.animation.time = time;
            let label = if self.animation.playing {
                "Pause"
            } else {
                "Play"
            };
            if widgets::pill_button(ui, None, label, false, true).clicked() {
                self.animation.playing = !self.animation.playing;
                self.pose_transition = None;
                self.selected_pose = None;
            }
            ui.checkbox(&mut self.animation.root_motion, "Root motion")
                .on_hover_text("Let the clip carry the mesh away; off, it plays in place");
        }
        if self.is_posed() {
            ui.label(
                egui::RichText::new(
                    "Painting lands on the posed surface in UV space; projection painting \
                     waits for the bind pose.",
                )
                .size(11.5)
                .color(super::theme::TEXT_WEAK),
            );
        }
    }
}

/// `point` on `triangle` of `bind`, carried to the same barycentric place on
/// that triangle of `posed`. Points off the triangle's plane are projected
/// onto it.
pub(super) fn carry_point(
    bind: &MeshGeometry,
    posed: &MeshGeometry,
    triangle: u32,
    point: Vec3,
) -> Vec3 {
    let (Some(from), Some(to)) = (
        bind.triangles.get(triangle as usize),
        posed.triangles.get(triangle as usize),
    ) else {
        return point;
    };
    let corner = |mesh: &MeshGeometry, index: u32| {
        mesh.vertices
            .get(index as usize)
            .map(|vertex| vertex.position)
    };
    let (Some(a), Some(b), Some(c)) = (
        corner(bind, from[0]),
        corner(bind, from[1]),
        corner(bind, from[2]),
    ) else {
        return point;
    };
    let (Some(pa), Some(pb), Some(pc)) = (
        corner(posed, to[0]),
        corner(posed, to[1]),
        corner(posed, to[2]),
    ) else {
        return point;
    };
    let (edge0, edge1, offset) = (b - a, c - a, point - a);
    let (d00, d01, d11) = (edge0.dot(edge0), edge0.dot(edge1), edge1.dot(edge1));
    let (d20, d21) = (offset.dot(edge0), offset.dot(edge1));
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() <= f32::EPSILON {
        return point;
    }
    let v = (d11 * d20 - d01 * d21) / denominator;
    let w = (d00 * d21 - d01 * d20) / denominator;
    pa * (1.0 - v - w) + pb * v + pc * w
}
