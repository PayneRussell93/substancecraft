//! Turntable renders: a numbered PNG sequence of one full turn, either of
//! the camera around the model or of the environment light around it.
//!
//! Each frame is an ordinary still render, queued once the previous one is
//! saved. The camera and light rotation are taken when the turntable
//! starts, so moving the viewport while it renders changes nothing.

use std::path::{Path, PathBuf};

use substancecraft_core::ViewportCamera;

use crate::{RenderRequest, SubstanceCraftApp};

/// Frame counts offered for a turntable.
pub const TURNTABLE_FRAME_COUNTS: [u32; 3] = [24, 36, 72];
/// Most frames one turntable renders.
pub const MAX_TURNTABLE_FRAMES: u32 = 720;

/// What turns during a turntable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TurntableMode {
    /// The camera orbits the model; the light stays put.
    #[default]
    Camera,
    /// The environment and its sun turn; the camera stays put.
    Lighting,
}

impl TurntableMode {
    pub const ALL: [Self; 2] = [Self::Camera, Self::Lighting];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Camera => "Camera",
            Self::Lighting => "Light",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Self::Camera => "The camera circles the model under fixed light",
            Self::Lighting => "The environment turns around the model, seen from this view",
        }
    }
}

/// A turntable in progress.
pub struct TurntableJob {
    /// The sequence's file name; frame `i` saves as `<stem>_<i>.png`.
    pub base: PathBuf,
    pub frames: u32,
    /// Frames saved so far, which is also the next frame to render.
    pub saved: u32,
    pub mode: TurntableMode,
    pub long_edge: u32,
    /// Camera and light rotation when the turntable started.
    pub camera: ViewportCamera,
    pub rotation: f32,
}

impl TurntableJob {
    /// Camera and light rotation of `frame`: one even step of a full turn
    /// per frame, so the last frame stops one step short of the first and
    /// the sequence loops seamlessly.
    pub fn frame_view(&self, frame: u32) -> (ViewportCamera, f32) {
        let angle = std::f32::consts::TAU * frame as f32 / self.frames.max(1) as f32;
        let turn = |value: f32| (value + angle).rem_euclid(std::f32::consts::TAU);
        match self.mode {
            TurntableMode::Camera => (
                ViewportCamera {
                    yaw: turn(self.camera.yaw),
                    ..self.camera
                },
                self.rotation,
            ),
            TurntableMode::Lighting => (self.camera, turn(self.rotation)),
        }
    }
}

/// `base` with the zero-padded frame number before its extension, always
/// as a PNG: `render.png` frame 7 of 36 is `render_007.png`.
pub fn frame_path(base: &Path, frame: u32, frames: u32) -> PathBuf {
    let stem = base
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "turntable".to_owned());
    let digits = (frames.saturating_sub(1).max(1).ilog10() + 1).max(3) as usize;
    base.with_file_name(format!("{stem}_{frame:0digits$}.png"))
}

impl SubstanceCraftApp {
    /// Asks where to save a turntable, then starts it.
    pub(super) fn request_turntable(&mut self, frames: u32, mode: TurntableMode, long_edge: u32) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG image sequence", &["png"])
            .set_file_name("turntable.png")
            .save_file()
        {
            self.start_turntable(path, frames, mode, long_edge);
        }
    }

    /// Starts rendering `frames` frames of a full turn, named after `base`,
    /// from the current camera and lighting. Replaces any still render or
    /// turntable still waiting.
    pub(super) fn start_turntable(
        &mut self,
        base: PathBuf,
        frames: u32,
        mode: TurntableMode,
        long_edge: u32,
    ) {
        let frames = frames.clamp(1, MAX_TURNTABLE_FRAMES);
        self.turntable = Some(TurntableJob {
            base,
            frames,
            saved: 0,
            mode,
            long_edge,
            camera: self.viewport_camera,
            rotation: self.viewport_lighting.rotation,
        });
        self.render_request = None;
        self.advance_turntable();
    }

    /// Stops a turntable after the frames already saved.
    pub(super) fn cancel_turntable(&mut self) {
        if let Some(job) = self.turntable.take() {
            self.render_request = None;
            self.render_status = Some(format!(
                "Turntable cancelled after {} of {} frames",
                job.saved, job.frames
            ));
        }
    }

    /// Queues the turntable's next frame once the last one is saved, or
    /// reports the finished sequence.
    pub(super) fn advance_turntable(&mut self) {
        if self.render_request.is_some() {
            return;
        }
        let Some(job) = &self.turntable else {
            return;
        };
        if job.saved >= job.frames {
            let folder = job
                .base
                .parent()
                .map(|folder| folder.display().to_string())
                .unwrap_or_default();
            self.render_status = Some(format!("Saved {}-frame turntable to {folder}", job.frames));
            self.turntable = None;
            return;
        }
        let frame = job.saved;
        let (camera, rotation) = job.frame_view(frame);
        self.render_status = Some(format!(
            "Rendering turntable frame {} of {}…",
            frame + 1,
            job.frames
        ));
        self.render_request = Some(RenderRequest {
            path: frame_path(&job.base, frame, job.frames),
            long_edge: job.long_edge,
            camera: Some(camera),
            lighting_rotation: Some(rotation),
            turntable_frame: Some(frame),
            issued: false,
        });
    }

    /// Records a saved turntable frame, or ends the turntable when a frame
    /// fails, keeping the failure message.
    pub(super) fn turntable_frame_done(&mut self, frame: u32, saved: bool) {
        match &mut self.turntable {
            Some(job) if saved && job.saved == frame => job.saved += 1,
            Some(_) => self.turntable = None,
            None => {}
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn job(mode: TurntableMode, frames: u32) -> TurntableJob {
        TurntableJob {
            base: PathBuf::from("out").join("spin.png"),
            frames,
            saved: 0,
            mode,
            long_edge: 1024,
            camera: ViewportCamera {
                yaw: 6.0,
                ..ViewportCamera::default()
            },
            rotation: 1.0,
        }
    }

    #[test]
    fn tc_0349_turntable_frames_step_evenly_through_one_turn() {
        // Camera turntables orbit the camera and keep the light.
        let camera = job(TurntableMode::Camera, 4);
        let quarter = std::f32::consts::FRAC_PI_2;
        for frame in 0..4 {
            let (view, rotation) = camera.frame_view(frame);
            let expected = (6.0 + quarter * frame as f32).rem_euclid(std::f32::consts::TAU);
            assert!((view.yaw - expected).abs() < 1e-5, "{frame}");
            assert!((0.0..std::f32::consts::TAU).contains(&view.yaw));
            assert_eq!(view.pitch, camera.camera.pitch);
            assert_eq!(view.zoom, camera.camera.zoom);
            assert_eq!(rotation, 1.0);
        }
        // The sequence loops: one more step returns to the first frame.
        let (first, _) = camera.frame_view(0);
        let (wrapped, _) = camera.frame_view(4);
        assert!((first.yaw - wrapped.yaw).abs() < 1e-4);

        // Light turntables turn the environment and keep the camera.
        let light = job(TurntableMode::Lighting, 36);
        let (view, rotation) = light.frame_view(9);
        assert_eq!(view, light.camera);
        assert!((rotation - (1.0 + quarter)).abs() < 1e-5);

        // Frame files are numbered with enough digits to sort.
        let base = PathBuf::from("out").join("spin.png");
        assert_eq!(
            frame_path(&base, 7, 36),
            PathBuf::from("out").join("spin_007.png")
        );
        assert_eq!(
            frame_path(&base, 12, 720),
            PathBuf::from("out").join("spin_012.png")
        );
        assert_eq!(
            frame_path(&base, 3, 5000),
            PathBuf::from("out").join("spin_0003.png")
        );
        assert_eq!(
            frame_path(&PathBuf::from("out").join("spin.tga"), 0, 1),
            PathBuf::from("out").join("spin_000.png")
        );
    }
}
