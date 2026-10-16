//! Image sequences: a bitmap resource whose pixels come from one frame of
//! a numbered file series (`wave_0001.png`, `wave_0002.png`, …). The frame
//! is picked in the Assets drawer or follows the Poses drawer's clip time,
//! so an animated texture (a flipbook, a screen, flowing water) plays with
//! the mesh; bakes and exports use the frame shown.

use std::path::Path;
use std::sync::Arc;

use eframe::egui::{self, Ui};
use substancecraft_api::{detect_image_sequence, load_resource_image};
use substancecraft_core::{MAX_SEQUENCE_FPS, MIN_SEQUENCE_FPS, ProjectResource, ResourceId};

use crate::animation::PoseSource;
use crate::panels::hint;
use crate::{SubstanceCraftApp, widgets};

impl SubstanceCraftApp {
    /// Adds the numbered sequence `path` belongs to as one resource showing
    /// `path`'s frame.
    pub(super) fn import_image_sequence(&mut self, path: &Path) -> Result<ResourceId, String> {
        let sequence = detect_image_sequence(path).ok_or_else(|| {
            format!(
                "{} is not part of a numbered sequence (name_0001.png, name_0002.png, …).",
                crate::path_label(path)
            )
        })?;
        let first =
            substancecraft_core::sequence_frame_path(&path.to_string_lossy(), sequence.first)
                .ok_or("The file name has no frame number.")?;
        let mut resource = ProjectResource::new(ResourceId(0), sequence_name(path), first);
        resource.sequence = Some(sequence);
        resource.color_space = substancecraft_api::source_color_space(path);
        let image = load_resource_image(&resource).map_err(|error| error.to_string())?;
        let id = ResourceId(
            self.project
                .resources
                .iter()
                .map(|resource| resource.id.0)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        );
        resource.id = id;
        resource.content_hash =
            substancecraft_api::file_content_hash(Path::new(&resource.source)).ok();
        self.resource_watcher
            .remember(id, Path::new(&resource.active_source()));
        self.project.resources.push(resource);
        self.resource_images.insert(id, Arc::new(image));
        Ok(id)
    }

    /// Asks for one frame of a sequence and imports the whole sequence.
    pub(super) fn pick_image_sequence(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Numbered images (PNG, JPEG, TGA, BMP, TIFF, WebP, EXR, HDR)",
                &substancecraft_api::BITMAP_EXTENSIONS,
            )
            .pick_file()
            && let Err(error) = self.import_image_sequence(&path)
        {
            self.import_error = Some(error);
        }
    }

    /// Shows frame `frame` of sequence resource `id` and rebakes the sets
    /// using it. A frame whose file cannot be read keeps the previous
    /// pixels and is reported. Returns whether the frame changed.
    pub(super) fn set_sequence_frame(&mut self, id: ResourceId, frame: u32) -> bool {
        let Some(resource) = self
            .project
            .resources
            .iter_mut()
            .find(|resource| resource.id == id)
        else {
            return false;
        };
        let Some(sequence) = resource.sequence.as_mut() else {
            return false;
        };
        let frame = frame.clamp(sequence.first, sequence.last);
        if sequence.frame == frame {
            return false;
        }
        sequence.frame = frame;
        let path = resource.active_source();
        self.reload_changed_resources(&[id]);
        self.resource_watcher.remember(id, Path::new(&path));
        if let Some(entry) = self
            .loaded_resource_sources
            .iter_mut()
            .find(|(loaded, _)| *loaded == id)
        {
            entry.1 = path;
        }
        true
    }

    /// Moves sequences that follow animation to the frame of the clip time
    /// shown in the Poses drawer.
    pub(super) fn follow_animation_with_sequences(&mut self) {
        let time = match self.animation.source {
            PoseSource::Clip(_) => self.animation.time,
            _ => return,
        };
        let frames: Vec<(ResourceId, u32)> = self
            .project
            .resources
            .iter()
            .filter_map(|resource| {
                let sequence = resource.sequence?;
                sequence
                    .follow_animation
                    .then(|| (resource.id, sequence.frame_at(time)))
            })
            .collect();
        for (id, frame) in frames {
            self.set_sequence_frame(id, frame);
        }
    }

    /// Frame, rate, and playback controls of a sequence resource, inside
    /// its details in the Assets drawer.
    pub(super) fn sequence_controls(&mut self, ui: &mut Ui, id: ResourceId) {
        let Some(mut sequence) = self
            .project
            .resources
            .iter()
            .find(|resource| resource.id == id)
            .and_then(|resource| resource.sequence)
        else {
            return;
        };
        let mut frame = sequence.frame;
        widgets::property_row(ui, "Frame", |ui| {
            ui.add_enabled(
                !sequence.follow_animation,
                egui::Slider::new(&mut frame, sequence.first..=sequence.last),
            )
            .on_hover_text("The frame layers show, bake, and export");
        });
        widgets::property_row(ui, "FPS", |ui| {
            ui.add(
                egui::DragValue::new(&mut sequence.fps)
                    .range(MIN_SEQUENCE_FPS..=MAX_SEQUENCE_FPS)
                    .speed(0.5),
            );
        });
        ui.checkbox(&mut sequence.follow_animation, "Follow animation")
            .on_hover_text("Take the frame from the clip playing in the Poses drawer, looping");
        hint(
            ui,
            &format!(
                "Frames {} to {} ({} files).",
                sequence.first,
                sequence.last,
                sequence.len()
            ),
        );
        if let Some(resource) = self
            .project
            .resources
            .iter_mut()
            .find(|resource| resource.id == id)
            && let Some(stored) = resource.sequence.as_mut()
        {
            stored.fps = sequence.fps;
            stored.follow_animation = sequence.follow_animation;
        }
        if frame != sequence.frame {
            self.set_sequence_frame(id, frame);
        }
    }
}

/// A resource name for the sequence `path` belongs to: its file name
/// without the frame number and separators before it.
fn sequence_name(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    substancecraft_core::split_frame_number(&name)
        .map(|(prefix, ..)| prefix.trim_end_matches(['_', '-', '.', ' ']).to_owned())
        .filter(|prefix| !prefix.is_empty())
        .unwrap_or_else(|| "Sequence".to_owned())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{
        BakedTexture, ImageSequence, LayerFill, MaterialChannel, MaterialLayer,
    };

    fn frame(path: &Path, value: u8) {
        substancecraft_api::write_texture(
            path,
            &BakedTexture {
                width: 2,
                height: 2,
                rgba8: [value, value, value, 255].repeat(4),
            },
        )
        .unwrap();
    }

    #[test]
    fn tc_0389_image_sequences_import_step_follow_animation_and_bake_their_frame() {
        let folder = tempfile::tempdir().unwrap();
        for (index, value) in [(1, 10), (2, 120), (3, 240)] {
            frame(&folder.path().join(format!("wave_{index:04}.png")), value);
        }
        // A gap ends the run; other names are not frames.
        frame(&folder.path().join("wave_0005.png"), 0);
        frame(&folder.path().join("other_0004.png"), 0);
        frame(&folder.path().join("single.png"), 0);

        let sequence = detect_image_sequence(&folder.path().join("wave_0002.png")).unwrap();
        assert_eq!((sequence.first, sequence.last, sequence.frame), (1, 3, 2));
        assert!(detect_image_sequence(&folder.path().join("single.png")).is_none());
        assert!(detect_image_sequence(&folder.path().join("other_0004.png")).is_none());

        let mut app = SubstanceCraftApp::default();
        let id = app
            .import_image_sequence(&folder.path().join("wave_0002.png"))
            .unwrap();
        let resource = app.project.resources.last().unwrap().clone();
        assert_eq!(resource.name, "wave");
        assert!(
            resource.source.ends_with("wave_0001.png"),
            "source is the first frame"
        );
        assert!(resource.active_source().ends_with("wave_0002.png"));
        assert_eq!(app.resource_images[&id].rgba8[0], 120);
        assert!(
            app.import_image_sequence(&folder.path().join("single.png"))
                .is_err()
        );

        // A fill showing the sequence bakes the current frame.
        app.add_paint_layer();
        let layer_id = app.selected_layer.unwrap();
        let mut fill = LayerFill::default();
        fill.textures.base_color = Some(id);
        let layer = app.selected_layer_mut().unwrap();
        *layer = MaterialLayer {
            channels: [MaterialChannel::BaseColor].into(),
            ..MaterialLayer::new_fill(layer_id, "Screen", fill)
        };
        let bake = |app: &SubstanceCraftApp| {
            let set = app.selected_texture_set_ref().unwrap();
            substancecraft_api::bake_texture_set_channel(
                set,
                MaterialChannel::BaseColor,
                None,
                None,
                Some(&app.resource_images),
            )
            .unwrap()
            .rgba8[0]
        };
        assert_eq!(bake(&app), 120);

        // Stepping the frame reloads the pixels; the range clamps.
        assert!(app.set_sequence_frame(id, 3));
        assert_eq!(app.resource_images[&id].rgba8[0], 240);
        assert_eq!(bake(&app), 240);
        assert!(!app.set_sequence_frame(id, 99), "already on the last frame");

        // Following animation picks the frame from the clip time, looping.
        if let Some(sequence) = app.project.resources.last_mut().unwrap().sequence.as_mut() {
            sequence.follow_animation = true;
            sequence.fps = 2.0;
        }
        app.animation.source = PoseSource::Clip("Wave".to_owned());
        app.animation.time = 0.1;
        app.follow_animation_with_sequences();
        assert_eq!(app.resource_images[&id].rgba8[0], 10);
        app.animation.time = 2.6; // frame index 5, looping to the third frame
        app.follow_animation_with_sequences();
        assert_eq!(app.resource_images[&id].rgba8[0], 240);

        // Sequences save with the project and validate their range.
        app.project.validate().unwrap();
        let saved = folder.path().join("screen.sproject");
        substancecraft_api::save_project_file(&saved, &app.project).unwrap();
        let loaded = substancecraft_api::load_project_file(&saved).unwrap();
        assert_eq!(
            loaded.resources.last().unwrap().sequence,
            app.project.resources.last().unwrap().sequence
        );
        let mut broken = app.project.clone();
        broken.resources.last_mut().unwrap().sequence = Some(ImageSequence {
            frame: 9,
            ..ImageSequence::new(1, 3)
        });
        assert!(broken.validate().is_err());
    }
}
