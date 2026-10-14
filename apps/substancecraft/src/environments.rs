//! Custom HDR environments: a `.hdr` or `.exr` panorama lighting the
//! viewport and still renders in place of a built-in lighting preset. The
//! chosen file is a user preference, like the preset.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use substancecraft_core::{EnvironmentImage, LightingPreset};

use crate::SubstanceCraftApp;

/// A panorama loaded from disk, ready for the renderer.
pub struct CustomEnvironment {
    pub path: PathBuf,
    pub image: Arc<EnvironmentImage>,
    /// Distinguishes loads, so reloading the same file prefilters again.
    pub revision: u64,
}

impl CustomEnvironment {
    /// The file's name, for chips and messages.
    pub fn name(&self) -> String {
        file_name(&self.path)
    }
}

/// What lights the renderer: a built-in preset or a loaded panorama.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvironmentSource {
    Preset(LightingPreset),
    Custom(u64),
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

impl SubstanceCraftApp {
    /// Lights the viewport (and still renders) with the panorama at `path`
    /// and remembers it for later sessions. A file that cannot be read
    /// leaves the current lighting as it is and says why.
    pub(super) fn load_custom_environment(&mut self, path: PathBuf) -> Result<(), String> {
        match substancecraft_api::load_environment_file(&path) {
            Ok(image) => {
                self.environment_revision += 1;
                let custom = CustomEnvironment {
                    path: path.clone(),
                    image: Arc::new(image),
                    revision: self.environment_revision,
                };
                self.environment_status = Some(format!("Lit by {}", custom.name()));
                self.custom_environment = Some(custom);
                if self.preferences.values.environment_file.as_ref() != Some(&path) {
                    self.preferences.values.environment_file = Some(path);
                    self.save_preferences();
                }
                Ok(())
            }
            Err(error) => {
                let message = format!("Could not load {}: {error}", file_name(&path));
                self.environment_status = Some(message.clone());
                Err(message)
            }
        }
    }

    /// Goes back to the built-in preset and forgets the custom file.
    pub(super) fn clear_custom_environment(&mut self) {
        self.custom_environment = None;
        self.environment_status = None;
        if self.preferences.values.environment_file.take().is_some() {
            self.save_preferences();
        }
    }

    /// Restores the panorama saved in preferences, if any. A file gone
    /// missing falls back to the preset but stays remembered, so it comes
    /// back once the file does.
    pub(super) fn restore_custom_environment(&mut self) {
        if let Some(path) = self.preferences.values.environment_file.clone() {
            // The status line already says why it failed.
            let _ = self.load_custom_environment(path);
        }
    }

    /// Asks for a panorama and lights the viewport with it.
    pub(super) fn pick_custom_environment(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "HDR environment (Radiance, OpenEXR)",
                &substancecraft_api::ENVIRONMENT_EXTENSIONS,
            )
            .pick_file()
        {
            let _ = self.load_custom_environment(path);
        }
    }
}
