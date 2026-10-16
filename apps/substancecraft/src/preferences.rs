//! Per-user preferences, kept outside projects: UI scale, keyboard
//! bindings (including hold-to-override tools), pinned brushes, and the
//! recent brush materials.

use std::path::{Path, PathBuf};

use eframe::egui::{self, Key, Modifiers, RichText, Ui};
use serde::{Deserialize, Serialize};
use substancecraft_core::{BrushShape, BrushTip};

use super::{PaintMode, SubstanceCraftApp};
use crate::icons::Icon;
use crate::{theme, widgets};

pub const MIN_UI_SCALE: f32 = 0.75;
pub const MAX_UI_SCALE: f32 = 2.0;
/// Widest exposure adjustment, in stops either way.
pub const MAX_EXPOSURE_STOPS: f32 = 6.0;
/// Recent brush materials kept.
pub const MAX_RECENT: usize = 12;
pub const MAX_FAVORITES: usize = 24;

/// Something a key can do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Undo,
    Redo,
    Save,
    ViewMode,
    PaintMode,
    EraseMode,
    MaskMode,
    /// While held: erase, then go back to the previous tool.
    HoldErase,
    /// While held: the next click picks the material.
    HoldPick,
    /// While held: drag orbits and pans instead of painting.
    HoldView,
    /// Starts painting a quick mask, or finishes the one being painted.
    QuickMask,
}

impl Action {
    pub const ALL: [Self; 11] = [
        Self::Undo,
        Self::Redo,
        Self::Save,
        Self::ViewMode,
        Self::PaintMode,
        Self::EraseMode,
        Self::MaskMode,
        Self::HoldErase,
        Self::HoldPick,
        Self::HoldView,
        Self::QuickMask,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Undo => "Undo",
            Self::Redo => "Redo",
            Self::Save => "Save project",
            Self::ViewMode => "View tool",
            Self::PaintMode => "Paint tool",
            Self::EraseMode => "Erase tool",
            Self::MaskMode => "Mask tool",
            Self::HoldErase => "Hold: erase",
            Self::HoldPick => "Hold: pick material",
            Self::HoldView => "Hold: navigate",
            Self::QuickMask => "Quick mask",
        }
    }

    pub fn is_hold(self) -> bool {
        matches!(self, Self::HoldErase | Self::HoldPick | Self::HoldView)
    }
}

/// A key with modifiers, stored by egui key name so the file stays
/// readable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub action: Action,
    pub key: String,
    #[serde(default)]
    pub command: bool,
    #[serde(default)]
    pub shift: bool,
    #[serde(default)]
    pub alt: bool,
}

impl Binding {
    fn new(action: Action, key: Key, modifiers: Modifiers) -> Self {
        Self {
            action,
            key: key.name().to_owned(),
            command: modifiers.command,
            shift: modifiers.shift,
            alt: modifiers.alt,
        }
    }

    pub fn key(&self) -> Option<Key> {
        Key::from_name(&self.key)
    }

    pub fn modifiers(&self) -> Modifiers {
        let mut modifiers = Modifiers::NONE;
        if self.command {
            modifiers |= Modifiers::COMMAND;
        }
        if self.shift {
            modifiers |= Modifiers::SHIFT;
        }
        if self.alt {
            modifiers |= Modifiers::ALT;
        }
        modifiers
    }

    pub fn label(&self) -> String {
        let mut parts = Vec::new();
        if self.command {
            parts.push("Ctrl");
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.alt {
            parts.push("Alt");
        }
        parts.push(&self.key);
        parts.join("+")
    }
}

pub fn default_bindings() -> Vec<Binding> {
    use Action::*;
    let command = Modifiers::COMMAND;
    vec![
        Binding::new(Undo, Key::Z, command),
        Binding::new(Redo, Key::Z, command | Modifiers::SHIFT),
        Binding::new(Redo, Key::Y, command),
        Binding::new(Save, Key::S, command),
        Binding::new(ViewMode, Key::V, Modifiers::NONE),
        Binding::new(PaintMode, Key::B, Modifiers::NONE),
        Binding::new(EraseMode, Key::E, Modifiers::NONE),
        Binding::new(MaskMode, Key::M, Modifiers::NONE),
        Binding::new(HoldErase, Key::X, Modifiers::NONE),
        Binding::new(HoldPick, Key::I, Modifiers::NONE),
        Binding::new(HoldView, Key::Space, Modifiers::NONE),
        Binding::new(QuickMask, Key::T, Modifiers::NONE),
    ]
}

/// Every channel value a paint brush carries.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushMaterial {
    pub color: [u8; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub height: f32,
    pub emissive: [u8; 3],
    pub opacity: f32,
}

/// A brush pinned to the favourites shelf, or a built-in preset. Bitmap
/// alphas are remembered by file, since resource ids belong to one project.
/// A preset with a material sets every channel of the brush as well.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushFavorite {
    pub name: String,
    pub tip: BrushTip,
    pub radius: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<BrushMaterial>,
}

impl BrushFavorite {
    /// Original presets bundling a tip, size, and material.
    pub fn built_in() -> Vec<Self> {
        let preset =
            |name: &str, tip: BrushTip, radius: f32, color: [u8; 3], roughness, metallic| Self {
                name: name.to_owned(),
                tip,
                radius,
                alpha: None,
                material: Some(BrushMaterial {
                    color: [color[0], color[1], color[2], 255],
                    roughness,
                    metallic,
                    height: substancecraft_core::DEFAULT_HEIGHT,
                    emissive: [0, 0, 0],
                    opacity: 1.0,
                }),
            };
        vec![
            preset(
                "Chipped paint",
                BrushTip {
                    shape: BrushShape::Splatter,
                    hardness: 0.6,
                    size_jitter: 0.4,
                    angle_jitter: 1.0,
                    ..BrushTip::ROUND
                },
                18.0,
                [196, 199, 199],
                0.3,
                1.0,
            ),
            preset(
                "Rust spots",
                BrushTip {
                    shape: BrushShape::Splatter,
                    spacing: 0.4,
                    size_jitter: 0.6,
                    scatter: 0.8,
                    seed: 7,
                    ..BrushTip::ROUND
                },
                30.0,
                [120, 58, 30],
                0.85,
                0.0,
            ),
            preset("Grime", BrushTip::ROUND, 40.0, [60, 50, 40], 0.95, 0.0),
            preset(
                "Gold leaf",
                BrushTip {
                    shape: BrushShape::Square,
                    hardness: 0.9,
                    ..BrushTip::ROUND
                },
                12.0,
                [255, 214, 140],
                0.25,
                1.0,
            ),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Preferences {
    pub ui_scale: f32,
    pub bindings: Vec<Binding>,
    pub favorites: Vec<BrushFavorite>,
    pub recent: Vec<BrushMaterial>,
    /// Colours of the mask debug overlay.
    pub overlay_palette: substancecraft_core::OverlayPalette,
    /// Hatch and dot the overlay so it does not rely on hue.
    pub overlay_pattern: bool,
    /// Memory background texture-set bakes may use, in MiB.
    pub cache_budget_mb: u32,
    /// Traces mesh-map occlusion and thickness rays on the GPU when there
    /// is one (PERF-01).
    pub gpu_baking: bool,
    /// Lighting-check environment of the viewport.
    pub lighting_preset: substancecraft_core::LightingPreset,
    /// HDR panorama lighting the viewport in place of the preset.
    pub environment_file: Option<PathBuf>,
    /// Colour space the brush colour's numeric fields are shown in.
    pub picker_color_space: substancecraft_core::ColorSpace,
    /// Shows the lighting environment behind the model.
    pub environment_background: bool,
    /// Blur of the background, `0..=1`.
    pub background_blur: f32,
    /// Screen-space global illumination in the viewport and renders.
    pub global_illumination: bool,
    /// Tone curve from scene light to the display.
    pub view_transform: substancecraft_core::ViewTransform,
    /// Exposure adjustment in stops.
    pub exposure: f32,
    /// `.cube` display LUT applied after the view transform.
    pub display_lut: Option<PathBuf>,
    /// Folder of the shared asset library; `None` uses the default.
    pub shared_library: Option<PathBuf>,
    /// OpenColorIO config, display, and view grading the viewport.
    pub ocio: Option<crate::ocio_view::OcioChoice>,
    /// Panel layouts saved by name.
    pub layouts: Vec<crate::layouts::PanelLayout>,
    /// The panel arrangement in use, restored next session.
    pub window_layout: Option<crate::layouts::PanelLayout>,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            ui_scale: 1.0,
            bindings: default_bindings(),
            favorites: Vec::new(),
            recent: Vec::new(),
            overlay_palette: substancecraft_core::OverlayPalette::Standard,
            overlay_pattern: false,
            cache_budget_mb: crate::cache::DEFAULT_CACHE_BUDGET_MB,
            gpu_baking: true,
            lighting_preset: substancecraft_core::LightingPreset::default(),
            environment_file: None,
            picker_color_space: substancecraft_core::ColorSpace::Srgb,
            environment_background: false,
            background_blur: 0.0,
            global_illumination: false,
            view_transform: substancecraft_core::ViewTransform::Standard,
            exposure: 0.0,
            display_lut: None,
            shared_library: None,
            ocio: None,
            layouts: Vec::new(),
            window_layout: None,
        }
    }
}

impl Preferences {
    /// `%LOCALAPPDATA%\SubstanceCraft\preferences.json`, else
    /// `$XDG_CONFIG_HOME/substancecraft/preferences.json` or
    /// `~/.config/substancecraft/preferences.json`.
    pub fn default_path() -> Option<PathBuf> {
        let from = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
        if let Some(local) = from("LOCALAPPDATA") {
            return Some(
                PathBuf::from(local)
                    .join("SubstanceCraft")
                    .join("preferences.json"),
            );
        }
        let config = from("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| from("HOME").map(|home| PathBuf::from(home).join(".config")))?;
        Some(config.join("substancecraft").join("preferences.json"))
    }

    /// Reads preferences, falling back to defaults for a missing or broken
    /// file and clamping out-of-range values.
    pub fn load(path: &Path) -> Self {
        let mut preferences: Self = std::fs::read(path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        preferences.sanitize();
        preferences
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let bytes = serde_json::to_vec_pretty(self).map_err(|error| error.to_string())?;
        std::fs::write(path, bytes).map_err(|error| error.to_string())
    }

    fn sanitize(&mut self) {
        self.exposure = if self.exposure.is_finite() {
            self.exposure.clamp(-MAX_EXPOSURE_STOPS, MAX_EXPOSURE_STOPS)
        } else {
            0.0
        };
        self.background_blur = if self.background_blur.is_finite() {
            self.background_blur.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.ui_scale = if self.ui_scale.is_finite() {
            self.ui_scale.clamp(MIN_UI_SCALE, MAX_UI_SCALE)
        } else {
            1.0
        };
        self.bindings.retain(|binding| binding.key().is_some());
        self.favorites.truncate(MAX_FAVORITES);
        // Raw data is not a colour space to pick colours in.
        if self.picker_color_space == substancecraft_core::ColorSpace::Raw {
            self.picker_color_space = substancecraft_core::ColorSpace::Srgb;
        }
        self.cache_budget_mb = self.cache_budget_mb.clamp(
            crate::cache::MIN_CACHE_BUDGET_MB,
            crate::cache::MAX_CACHE_BUDGET_MB,
        );
        self.recent.truncate(MAX_RECENT);
    }

    /// The first binding of `action`, for display.
    pub fn binding_of(&self, action: Action) -> Option<&Binding> {
        self.bindings
            .iter()
            .find(|binding| binding.action == action)
    }

    /// Binds `action` to one key, dropping its other keys and any other
    /// action that key had.
    pub fn rebind(&mut self, action: Action, key: Key, modifiers: Modifiers) {
        let binding = Binding::new(action, key, modifiers);
        self.bindings.retain(|existing| {
            existing.action != action
                && !(existing.key == binding.key && existing.modifiers() == binding.modifiers())
        });
        self.bindings.push(binding);
    }

    /// Records a brush material at the front of the recent list.
    pub fn remember_material(&mut self, material: BrushMaterial) {
        self.recent.retain(|recent| *recent != material);
        self.recent.insert(0, material);
        self.recent.truncate(MAX_RECENT);
    }
}

/// A tool switched on by holding a key, and what to restore on release.
pub struct HeldOverride {
    pub key: Key,
    previous_mode: PaintMode,
    previous_picking: bool,
}

/// Preferences state the app keeps.
#[derive(Default)]
pub struct PreferenceState {
    pub values: Preferences,
    /// Where preferences are saved; `None` keeps them in memory (tests).
    pub path: Option<PathBuf>,
    pub held: Option<HeldOverride>,
    /// The action waiting for its new key in the bindings list.
    pub rebinding: Option<Action>,
}

impl SubstanceCraftApp {
    pub(super) fn load_preferences(&mut self, path: PathBuf) {
        self.preferences.values = Preferences::load(&path);
        self.viewport_lighting.preset = self.preferences.values.lighting_preset;
        self.preferences.path = Some(path);
        self.restore_custom_environment();
        self.restore_display_lut();
        self.restore_ocio_view();
        self.restore_layout();
        self.refresh_user_shaders();
    }

    /// Lights the viewport (and still renders) with `preset`, in place of
    /// any custom environment, and remembers it for later sessions.
    pub(super) fn set_lighting_preset(&mut self, preset: substancecraft_core::LightingPreset) {
        self.clear_custom_environment();
        self.viewport_lighting.preset = preset;
        if self.preferences.values.lighting_preset != preset {
            self.preferences.values.lighting_preset = preset;
            self.save_preferences();
        }
    }

    /// Shows the brush colour's numeric fields in `space` from now on.
    pub(super) fn set_picker_color_space(&mut self, space: substancecraft_core::ColorSpace) {
        if space != substancecraft_core::ColorSpace::Raw
            && self.preferences.values.picker_color_space != space
        {
            self.preferences.values.picker_color_space = space;
            self.save_preferences();
        }
    }

    /// Sets the brush colour from `rgb` given in `space`, keeping its
    /// alpha. Returns whether the colour lay outside sRGB and was clipped;
    /// the clipped result is remembered so the panel can say so while it
    /// stays the brush colour.
    pub(super) fn set_brush_color_in(
        &mut self,
        space: substancecraft_core::ColorSpace,
        rgb: glam::Vec3,
    ) -> bool {
        let ([r, g, b], clipped) = space.to_srgb8(rgb);
        let alpha = self.brush_color.a();
        // The bytes are the stroke colour as stored, so they are set as is.
        self.brush_color = egui::Color32::from_rgba_premultiplied(r, g, b, alpha);
        self.picker_clipped = clipped.then_some(self.brush_color);
        clipped
    }

    pub(super) fn save_preferences(&mut self) {
        if let Some(path) = &self.preferences.path
            && let Err(error) = self.preferences.values.save(path)
        {
            self.project_error = Some(format!("Could not save preferences: {error}"));
        }
    }

    pub(super) fn current_brush_material(&self) -> BrushMaterial {
        let emissive = self.brush_emissive.to_array();
        BrushMaterial {
            color: self.brush_color.to_array(),
            roughness: self.brush_roughness,
            metallic: self.brush_metallic,
            height: self.brush_height,
            emissive: [emissive[0], emissive[1], emissive[2]],
            opacity: self.brush_material_opacity,
        }
    }

    /// Restores every channel of a recent brush material in one click.
    pub(super) fn apply_brush_material(&mut self, material: BrushMaterial) {
        let [r, g, b, a] = material.color;
        self.brush_color = egui::Color32::from_rgba_unmultiplied(r, g, b, a);
        self.brush_roughness = material.roughness;
        self.brush_metallic = material.metallic;
        self.brush_height = material.height;
        let [r, g, b] = material.emissive;
        self.brush_emissive = egui::Color32::from_rgb(r, g, b);
        self.brush_material_opacity = material.opacity;
    }

    /// Called when a paint stroke is committed.
    pub(super) fn remember_brush_material(&mut self) {
        let material = self.current_brush_material();
        if self.preferences.values.recent.first() != Some(&material) {
            self.preferences.values.remember_material(material);
            self.save_preferences();
        }
    }

    /// Pins the current brush tip and size to the favourites shelf.
    pub(super) fn pin_brush(&mut self) {
        self.pin(false);
    }

    /// Pins the current tip, size, and every material channel as a preset.
    pub(super) fn pin_brush_preset(&mut self) {
        self.pin(true);
    }

    fn pin(&mut self, with_material: bool) {
        let material = with_material.then(|| self.current_brush_material());
        let favorites = &self.preferences.values.favorites;
        if favorites.len() >= MAX_FAVORITES {
            self.project_error = Some(format!("The shelf holds at most {MAX_FAVORITES} brushes"));
            return;
        }
        let mut tip = self.brush_tip;
        let alpha = match tip.shape {
            BrushShape::Bitmap(id) => {
                tip.shape = BrushShape::Round;
                self.project
                    .resources
                    .iter()
                    .find(|resource| resource.id == id)
                    .map(|resource| PathBuf::from(&resource.source))
            }
            _ => None,
        };
        let shape = match (&alpha, self.brush_tip.shape) {
            (Some(path), _) => path.file_stem().map_or_else(
                || "Alpha".to_owned(),
                |stem| stem.to_string_lossy().into_owned(),
            ),
            (None, shape) => shape_name(shape).to_owned(),
        };
        let taken: Vec<String> = favorites
            .iter()
            .map(|favorite| favorite.name.clone())
            .collect();
        let base = if material.is_some() {
            format!("{shape} {:.0} preset", self.brush_radius)
        } else {
            format!("{shape} {:.0}", self.brush_radius)
        };
        let name = Self::unique_name(&base, &taken);
        self.preferences.values.favorites.push(BrushFavorite {
            name,
            tip,
            radius: self.brush_radius,
            alpha,
            material,
        });
        self.save_preferences();
    }

    /// Loads a pinned brush; a bitmap alpha is imported into the project
    /// unless it already uses that file.
    pub(super) fn apply_favorite(&mut self, index: usize) {
        if let Some(favorite) = self.preferences.values.favorites.get(index).cloned() {
            self.apply_brush_preset(&favorite);
        }
    }

    /// Sets the brush tip and size from a pinned brush or preset, and every
    /// material channel when it carries a material.
    pub(super) fn apply_brush_preset(&mut self, favorite: &BrushFavorite) {
        let mut tip = favorite.tip;
        if let Some(path) = &favorite.alpha {
            let existing = self
                .project
                .resources
                .iter()
                .find(|resource| Path::new(&resource.source) == path)
                .map(|resource| resource.id);
            let id = match existing {
                Some(id) => id,
                None => match self.add_resource(path) {
                    Ok(id) => id,
                    Err(error) => {
                        self.project_error = Some(format!("Brush `{}`: {error}", favorite.name));
                        return;
                    }
                },
            };
            tip.shape = BrushShape::Bitmap(id);
        }
        self.brush_tip = tip;
        self.brush_radius = favorite.radius;
        if let Some(material) = favorite.material {
            self.apply_brush_material(material);
        }
    }

    pub(super) fn unpin_brush(&mut self, index: usize) {
        if index < self.preferences.values.favorites.len() {
            self.preferences.values.favorites.remove(index);
            self.save_preferences();
        }
    }

    /// Runs bound actions: tap bindings fire once on press; hold bindings
    /// switch tool while the key is down and restore it on release.
    pub(super) fn handle_bindings(&mut self, context: &egui::Context) {
        if self.preferences.rebinding.is_some() {
            self.capture_rebinding(context);
            return;
        }
        if let Some(held) = &self.preferences.held {
            let key = held.key;
            if !context.input(|input| input.key_down(key)) {
                self.release_override();
            }
        }
        // Typing goes to text fields; buttons and the viewport focused from
        // the keyboard still leave shortcuts working.
        if context.text_edit_focused() {
            return;
        }
        let bindings = self.preferences.values.bindings.clone();
        // Most specific first, so Ctrl+Shift+Z is not taken for Ctrl+Z.
        let mut ordered: Vec<&Binding> = bindings.iter().collect();
        ordered.sort_by_key(|binding| {
            std::cmp::Reverse(
                u8::from(binding.command) + u8::from(binding.shift) + u8::from(binding.alt),
            )
        });
        for binding in ordered {
            let Some(key) = binding.key() else {
                continue;
            };
            let shortcut = egui::KeyboardShortcut::new(binding.modifiers(), key);
            if binding.action.is_hold() {
                let pressed = context.input(|input| {
                    input.key_pressed(key)
                        && !input.modifiers.command
                        && !input.modifiers.alt
                        && input.modifiers.shift == binding.shift
                });
                if pressed && self.preferences.held.is_none() {
                    self.begin_override(binding.action, key);
                }
                continue;
            }
            if context.input_mut(|input| input.consume_shortcut(&shortcut)) {
                self.run_action(binding.action);
                return;
            }
        }
    }

    pub(super) fn run_action(&mut self, action: Action) {
        match action {
            Action::Undo => self.undo(),
            Action::Redo => self.redo(),
            Action::Save => self.save_project(),
            Action::ViewMode => self.set_paint_mode(PaintMode::View),
            Action::PaintMode => self.set_paint_mode(PaintMode::Paint),
            Action::EraseMode => self.set_paint_mode(PaintMode::Erase),
            Action::MaskMode => self.set_paint_mode(PaintMode::Mask),
            Action::QuickMask => self.toggle_quick_mask(),
            Action::HoldErase | Action::HoldPick | Action::HoldView => {}
        }
    }

    pub(super) fn begin_override(&mut self, action: Action, key: Key) {
        let held = HeldOverride {
            key,
            previous_mode: self.paint_mode,
            previous_picking: self.picking_material,
        };
        match action {
            Action::HoldErase => self.set_paint_mode(PaintMode::Erase),
            Action::HoldPick => self.picking_material = true,
            Action::HoldView => self.set_paint_mode(PaintMode::View),
            _ => return,
        }
        self.preferences.held = Some(held);
    }

    pub(super) fn release_override(&mut self) {
        let Some(held) = self.preferences.held.take() else {
            return;
        };
        self.set_paint_mode(held.previous_mode);
        self.picking_material = held.previous_picking;
    }

    fn capture_rebinding(&mut self, context: &egui::Context) {
        let Some(action) = self.preferences.rebinding else {
            return;
        };
        let pressed = context.input(|input| {
            input.events.iter().find_map(|event| match event {
                egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } => Some((*key, *modifiers)),
                _ => None,
            })
        });
        let Some((key, modifiers)) = pressed else {
            return;
        };
        self.preferences.rebinding = None;
        if key == Key::Escape {
            return;
        }
        let modifiers = if action.is_hold() {
            // Hold keys work alone or with Shift.
            Modifiers {
                shift: modifiers.shift,
                ..Modifiers::NONE
            }
        } else {
            modifiers
        };
        self.preferences.values.rebind(action, key, modifiers);
        self.save_preferences();
    }

    /// Keeps egui's zoom at the preferred UI scale.
    pub(super) fn apply_ui_scale(&self, context: &egui::Context) {
        let scale = self.preferences.values.ui_scale;
        if (context.zoom_factor() - scale).abs() > f32::EPSILON {
            context.set_zoom_factor(scale);
        }
    }

    /// The Preferences drawer: UI scale and key bindings.
    pub(super) fn preferences_drawer(&mut self, ui: &mut Ui) {
        self.layouts_section(ui);
        widgets::caption(ui, "Interface scale");
        let mut scale = self.preferences.values.ui_scale;
        widgets::property_row(ui, "Scale", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (value, label) in [
                (0.85, "85%"),
                (1.0, "100%"),
                (1.25, "125%"),
                (1.5, "150%"),
                (2.0, "200%"),
            ] {
                if widgets::chip(ui, label, (scale - value).abs() < 0.001).clicked() {
                    scale = value;
                }
            }
        });
        let mut save = (scale - self.preferences.values.ui_scale).abs() > f32::EPSILON;
        let slider = widgets::property_row(ui, "Custom", |ui| {
            ui.add(
                egui::Slider::new(&mut scale, MIN_UI_SCALE..=MAX_UI_SCALE)
                    .step_by(0.05)
                    .custom_formatter(|value, _| format!("{:.0}%", value * 100.0)),
            )
        });
        self.preferences.values.ui_scale = scale.clamp(MIN_UI_SCALE, MAX_UI_SCALE);
        // Written to disk when a drag ends rather than on every frame of it.
        save |= slider.drag_stopped() || (slider.changed() && !slider.dragged());
        if save {
            self.save_preferences();
        }

        widgets::caption(ui, "Colour vision");
        let mut palette = self.preferences.values.overlay_palette;
        let mut pattern = self.preferences.values.overlay_pattern;
        widgets::property_row(ui, "Overlays", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                for option in substancecraft_core::OverlayPalette::ALL {
                    if widgets::chip(ui, option.name(), palette == option)
                        .on_hover_text("Colours of the mask overlay and mesh edge overlays")
                        .clicked()
                    {
                        palette = option;
                    }
                }
            });
        });
        widgets::property_row(ui, "Patterns", |ui| {
            widgets::toggle(ui, &mut pattern).on_hover_text(
                "Tell overlay classes apart by hatching, dots, and dashes, not hue alone",
            );
        });
        if palette != self.preferences.values.overlay_palette
            || pattern != self.preferences.values.overlay_pattern
        {
            self.preferences.values.overlay_palette = palette;
            self.preferences.values.overlay_pattern = pattern;
            self.save_preferences();
            self.invalidate_preview();
        }

        widgets::caption(ui, "Keyboard");
        let waiting = self.preferences.rebinding;
        let mut start_rebind = None;
        for action in Action::ALL {
            let label = if waiting == Some(action) {
                "Press a key… (Esc cancels)".to_owned()
            } else {
                let keys: Vec<String> = self
                    .preferences
                    .values
                    .bindings
                    .iter()
                    .filter(|binding| binding.action == action)
                    .map(Binding::label)
                    .collect();
                if keys.is_empty() {
                    "Unbound".to_owned()
                } else {
                    keys.join(", ")
                }
            };
            widgets::property_row(ui, action.label(), |ui| {
                if widgets::chip(ui, &label, waiting == Some(action))
                    .on_hover_text("Click, then press the new key")
                    .clicked()
                {
                    start_rebind = Some(action);
                }
            });
        }
        if let Some(action) = start_rebind {
            self.preferences.rebinding = Some(action);
        }
        ui.add_space(4.0);
        if widgets::pill_button(ui, Some(Icon::Undo), "Restore default keys", false, true).clicked()
        {
            self.preferences.values.bindings = default_bindings();
            self.save_preferences();
        }
        ui.label(
            RichText::new(
                "Hold keys switch tool only while held. Every control is also reachable \
                 with Tab and Enter.",
            )
            .size(11.5)
            .color(theme::TEXT_FAINT),
        );
    }

    /// Recent brush materials and the pinned-brush shelf, in the Tools
    /// drawer.
    pub(super) fn brush_shelves(&mut self, ui: &mut Ui) {
        let mut apply_recent = None;
        if !self.preferences.values.recent.is_empty() {
            widgets::property_row(ui, "Recent", |ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                for (index, material) in self.preferences.values.recent.iter().enumerate() {
                    let [r, g, b, _] = material.color;
                    let (rect, response) =
                        ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::click());
                    ui.painter()
                        .rect_filled(rect, 4.0, egui::Color32::from_rgb(r, g, b));
                    if material.metallic > 0.5 {
                        ui.painter().rect_stroke(
                            rect,
                            4.0,
                            egui::Stroke::new(1.0, theme::TEXT_WEAK),
                            egui::StrokeKind::Inside,
                        );
                    }
                    if response
                        .on_hover_text(format!(
                            "Roughness {:.2}, metallic {:.2}, height {:.2}",
                            material.roughness, material.metallic, material.height
                        ))
                        .clicked()
                    {
                        apply_recent = Some(index);
                    }
                }
            });
        }
        if let Some(index) = apply_recent
            && let Some(material) = self.preferences.values.recent.get(index).copied()
        {
            self.apply_brush_material(material);
        }

        let mut preset = None;
        widgets::property_row(ui, "Presets", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                for built_in in BrushFavorite::built_in() {
                    if material_chip(ui, &built_in)
                        .on_hover_text("Tip, size, and material together")
                        .clicked()
                    {
                        preset = Some(built_in);
                    }
                }
            });
        });
        if let Some(preset) = preset {
            self.apply_brush_preset(&preset);
        }

        let mut apply = None;
        let mut unpin = None;
        let mut pin_preset = false;
        widgets::property_row(ui, "Shelf", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                for (index, favorite) in self.preferences.values.favorites.iter().enumerate() {
                    let response = material_chip(ui, favorite)
                        .on_hover_text("Click to use; right-click to unpin");
                    if response.clicked() {
                        apply = Some(index);
                    }
                    if response.secondary_clicked() {
                        unpin = Some(index);
                    }
                }
                if widgets::icon_button(ui, Icon::Plus, true, "Pin this brush tip and size")
                    .clicked()
                {
                    self.pin_brush();
                }
                pin_preset = widgets::chip(ui, "+ Material", false)
                    .on_hover_text("Pin this tip and size with every material channel as a preset")
                    .clicked();
            });
        });
        if pin_preset {
            self.pin_brush_preset();
        }
        if let Some(index) = apply {
            self.apply_favorite(index);
        }
        if let Some(index) = unpin {
            self.unpin_brush(index);
        }
    }
}

/// A shelf chip; presets with a material show its colour first.
fn material_chip(ui: &mut Ui, favorite: &BrushFavorite) -> egui::Response {
    match favorite.material {
        Some(material) => {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                let [r, g, b, _] = material.color;
                let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 16.0), egui::Sense::hover());
                ui.painter().rect_filled(
                    rect.shrink2(egui::vec2(0.0, 3.0)),
                    2.0,
                    egui::Color32::from_rgb(r, g, b),
                );
                widgets::chip(ui, &favorite.name, false)
            })
            .inner
        }
        None => widgets::chip(ui, &favorite.name, false),
    }
}

fn shape_name(shape: BrushShape) -> &'static str {
    match shape {
        BrushShape::Round => "Round",
        BrushShape::Square => "Square",
        BrushShape::Splatter => "Splatter",
        BrushShape::Streak => "Streak",
        BrushShape::Ring => "Ring",
        BrushShape::Cloud => "Cloud",
        BrushShape::Bitmap(_) => "Alpha",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tc_0239_preferences_round_trip_clamp_and_rebind() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("prefs").join("preferences.json");
        // Missing and broken files give defaults.
        assert_eq!(Preferences::load(&path), Preferences::default());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{ not json").unwrap();
        assert_eq!(Preferences::load(&path), Preferences::default());

        let mut preferences = Preferences {
            ui_scale: 9.0,
            ..Preferences::default()
        };
        preferences.rebind(Action::HoldErase, Key::E, Modifiers::NONE);
        // E was the Erase tool; it now only holds the eraser.
        assert!(preferences.binding_of(Action::EraseMode).is_none());
        assert_eq!(preferences.binding_of(Action::HoldErase).unwrap().key, "E");
        preferences.save(&path).unwrap();
        let loaded = Preferences::load(&path);
        assert_eq!(loaded.ui_scale, MAX_UI_SCALE);
        assert_eq!(loaded.bindings, preferences.bindings);

        let material = BrushMaterial {
            color: [1, 2, 3, 255],
            roughness: 0.2,
            metallic: 1.0,
            height: 0.5,
            emissive: [0, 0, 0],
            opacity: 1.0,
        };
        for _ in 0..3 {
            preferences.remember_material(material);
        }
        assert_eq!(preferences.recent.len(), 1);
        for shade in 0..20u8 {
            preferences.remember_material(BrushMaterial {
                color: [shade, 0, 0, 255],
                ..material
            });
        }
        assert_eq!(preferences.recent.len(), MAX_RECENT);
        assert_eq!(preferences.recent[0].color[0], 19);
    }

    #[test]
    fn tc_0240_hold_keys_override_tools_and_recent_materials_restore_channels() {
        let mut app = SubstanceCraftApp::default();
        app.add_texture_set();
        app.add_paint_layer();
        app.set_paint_mode(PaintMode::Paint);

        // Holding the eraser key switches tool; releasing restores Paint.
        app.begin_override(Action::HoldErase, Key::X);
        assert_eq!(app.paint_mode, PaintMode::Erase);
        app.release_override();
        assert_eq!(app.paint_mode, PaintMode::Paint);
        app.begin_override(Action::HoldPick, Key::I);
        assert!(app.picking_material);
        app.release_override();
        assert!(!app.picking_material);
        app.begin_override(Action::HoldView, Key::Space);
        assert_eq!(app.paint_mode, PaintMode::View);
        app.release_override();
        assert_eq!(app.paint_mode, PaintMode::Paint);

        // Tap bindings run their action.
        app.run_action(Action::MaskMode);
        assert_eq!(app.paint_mode, PaintMode::Mask);
        app.run_action(Action::PaintMode);

        // A recent material brings back every channel at once.
        app.brush_color = egui::Color32::from_rgb(200, 30, 10);
        app.brush_roughness = 0.15;
        app.brush_metallic = 1.0;
        app.remember_brush_material();
        let remembered = app.current_brush_material();
        app.brush_color = egui::Color32::WHITE;
        app.brush_roughness = 0.9;
        app.brush_metallic = 0.0;
        app.apply_brush_material(app.preferences.values.recent[0]);
        assert_eq!(app.current_brush_material(), remembered);

        // Pinned brushes restore tip and size.
        app.brush_tip.shape = BrushShape::Splatter;
        app.brush_radius = 42.0;
        app.pin_brush();
        assert_eq!(app.preferences.values.favorites[0].name, "Splatter 42");
        app.brush_tip = BrushTip::ROUND;
        app.brush_radius = 5.0;
        app.apply_favorite(0);
        assert_eq!(app.brush_tip.shape, BrushShape::Splatter);
        assert_eq!(app.brush_radius, 42.0);
        app.unpin_brush(0);
        assert!(app.preferences.values.favorites.is_empty());
    }

    #[test]
    fn tc_0288_brush_presets_bundle_tip_size_and_material() {
        use substancecraft_core::{
            LayerId, MaterialLayer, MaterialProject, TextureResolution, TextureSet, TextureSetId,
        };

        let mut app = SubstanceCraftApp::default();
        // A preset keeps tip, size, and every channel.
        app.brush_tip.shape = BrushShape::Square;
        app.brush_radius = 9.0;
        app.brush_color = egui::Color32::from_rgb(10, 200, 30);
        app.brush_roughness = 0.12;
        app.brush_metallic = 1.0;
        app.brush_height = 0.7;
        app.pin_brush_preset();
        app.pin_brush();
        let favorites = &app.preferences.values.favorites;
        assert_eq!(favorites[0].name, "Square 9 preset");
        assert!(favorites[0].material.is_some());
        assert_eq!(favorites[1].material, None, "plain pins keep the material");
        let preset = app.current_brush_material();
        app.brush_tip = BrushTip::ROUND;
        app.brush_radius = 50.0;
        app.brush_color = egui::Color32::WHITE;
        app.brush_roughness = 0.9;
        app.apply_favorite(1);
        assert_eq!(app.brush_tip.shape, BrushShape::Square);
        assert_eq!(app.brush_roughness, 0.9, "a plain pin leaves the material");
        app.apply_favorite(0);
        assert_eq!(app.current_brush_material(), preset);

        // Presets survive a save, and older files without materials load.
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("preferences.json");
        app.preferences.values.save(&path).unwrap();
        assert_eq!(
            Preferences::load(&path).favorites,
            app.preferences.values.favorites
        );
        let old = r#"{"favorites":[{"name":"Old","tip":{"shape":"Round","hardness":0.0,"spacing":0.1,"angle_degrees":0.0,"follow_path":false,"size_jitter":0.0,"angle_jitter":0.0,"scatter":0.0,"seed":0,"pressure_size":true,"pressure_opacity":false},"radius":4.0}]}"#;
        std::fs::write(&path, old).unwrap();
        let loaded = Preferences::load(&path);
        assert_eq!(loaded.favorites[0].name, "Old");
        assert_eq!(loaded.favorites[0].material, None);

        // Built-in presets are valid tips with materials.
        let built_in = BrushFavorite::built_in();
        assert_eq!(built_in.len(), 4);
        for preset in &built_in {
            let stroke = substancecraft_core::PaintStroke {
                tip: preset.tip,
                ..substancecraft_core::PaintStroke::new(
                    vec![glam::Vec2::splat(0.5)],
                    preset.radius,
                    [0, 0, 0, 255],
                )
            };
            let mut layer = MaterialLayer::new_paint(LayerId(1), "Probe");
            layer.paint_strokes.push(stroke);
            let mut project = MaterialProject::empty();
            project.texture_sets.push(TextureSet {
                user_channels: Default::default(),
                stencils: Default::default(),
                id: TextureSetId(1),
                name: "Set".to_owned(),
                material_slot: "Set".to_owned(),
                resolution: TextureResolution {
                    width: 8,
                    height: 8,
                },
                channels: Default::default(),
                layers: vec![layer],
                height_strength: substancecraft_core::DEFAULT_HEIGHT_STRENGTH,
                export: None,
            });
            project.validate().unwrap();
            assert!(preset.material.is_some(), "{}", preset.name);
        }
        app.apply_brush_preset(&built_in[3]);
        assert_eq!(app.brush_tip.shape, BrushShape::Square);
        assert_eq!(app.brush_color, egui::Color32::from_rgb(255, 214, 140));
        assert_eq!(app.brush_metallic, 1.0);
    }
}
