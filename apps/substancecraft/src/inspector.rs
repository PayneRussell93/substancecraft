//! Surface-to-layer inspector, solo views, and mask view. All of it is
//! viewport state: nothing here is saved in the project or reaches exports.

use eframe::egui::{self, Id, RichText, Ui};
use substancecraft_core::{
    BakeContext, BakedTexture, LayerId, MaterialChannel, TexelContribution, TextureSet,
};

use super::{BakeInputs, SubstanceCraftApp, channel_label};
use crate::{theme, widgets};

/// What the viewport isolates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solo {
    /// One layer or group, with everything else hidden.
    Layer(LayerId),
    /// A layer's final mask in grayscale (mask view).
    Mask(LayerId),
    /// A layer's generator alone, in grayscale.
    Generator(LayerId),
    /// The shaded model tinted by a layer's mask.
    Overlay(LayerId),
    /// The selected texture set; other material slots show neutral grey.
    TextureSet,
}

/// The inspector's last result.
pub struct Inspection {
    pub uv: glam::Vec2,
    pub channel: MaterialChannel,
    pub entries: Result<Vec<TexelContribution>, String>,
}

pub struct InspectState {
    pub solo: Option<Solo>,
    /// Tint strength of the mask overlay, `0..=1`.
    pub overlay_opacity: f32,
    /// The next click on the model inspects the texel under it.
    pub armed: bool,
    pub result: Option<Inspection>,
}

impl Default for InspectState {
    fn default() -> Self {
        Self {
            solo: None,
            overlay_opacity: 0.6,
            armed: false,
            result: None,
        }
    }
}

impl BakeInputs {
    fn context(&self) -> BakeContext<'_> {
        BakeContext {
            mesh: self.mesh.as_deref(),
            mesh_maps: self.maps(),
            resources: Some(&self.resources),
        }
    }

    pub(super) fn bake_layer_mask(
        &self,
        texture_set: &TextureSet,
        layer: LayerId,
        generator_only: bool,
    ) -> Result<BakedTexture, String> {
        if let Some(udim) = &self.udim {
            return udim.bake_single(self, texture_set, None, |set, context| {
                set.bake_layer_mask(layer, generator_only, context)
                    .map_err(|error| error.to_string())
            });
        }
        texture_set
            .bake_layer_mask(layer, generator_only, self.context())
            .map_err(|error| error.to_string())
    }

    /// A quick mask of `texture_set` in grayscale.
    pub(super) fn bake_stencil(
        &self,
        texture_set: &TextureSet,
        id: substancecraft_core::StencilId,
    ) -> Result<BakedTexture, String> {
        if let Some(udim) = &self.udim {
            return udim.bake_single(self, texture_set, None, |set, context| {
                set.bake_stencil(id, context)
                    .map_err(|error| error.to_string())
            });
        }
        texture_set
            .bake_stencil(id, self.context())
            .map_err(|error| error.to_string())
    }
}

impl SubstanceCraftApp {
    /// The solo state that applies now: layer solos only while their layer
    /// is in the selected set.
    pub(super) fn active_solo(&self) -> Option<Solo> {
        let solo = self.inspect.solo?;
        let set = self.selected_texture_set_ref()?;
        match solo {
            Solo::Layer(id) | Solo::Mask(id) | Solo::Overlay(id) => set.layer(id).map(|_| solo),
            Solo::Generator(id) => set
                .layer(id)
                .filter(|layer| layer.mask_generator.is_some())
                .map(|_| solo),
            Solo::TextureSet => Some(solo),
        }
    }

    pub(super) fn set_solo(&mut self, solo: Option<Solo>) {
        if self.inspect.solo != solo {
            self.inspect.solo = solo;
            // Solo changes the view only; leaving it shows the set's cached
            // material again without a rebake.
            self.invalidate_view();
        }
    }

    /// Lists the layers making up the texel at `uv` in the active channel.
    pub(super) fn inspect_at(&mut self, uv: glam::Vec2) {
        self.inspect.armed = false;
        let channel = self.active_channel;
        let entries = match self.selected_texture_set_ref().cloned() {
            None => Err("Select a texture set first".to_owned()),
            Some(_) if channel == MaterialChannel::Id => {
                Err("The ID channel has no layer stack to inspect".to_owned())
            }
            Some(set) if set.requires_mesh_maps() && !self.mesh_maps_fresh() => {
                self.ensure_mesh_maps();
                Err("Mesh maps are baking; inspect again when they finish".to_owned())
            }
            Some(set) => {
                let inputs = self.bake_inputs(set.id);
                set.inspect_texel(channel, uv, inputs.context())
                    .map_err(|error| error.to_string())
            }
        };
        self.inspect.result = Some(Inspection {
            uv,
            channel,
            entries,
        });
    }

    /// Opacity, palette, and pattern of the mask overlay. Palette and
    /// pattern are accessibility preferences, kept across projects.
    fn overlay_controls(&mut self, ui: &mut Ui) {
        let mut opacity = self.inspect.overlay_opacity;
        widgets::property_row(ui, "Tint", |ui| {
            ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).fixed_decimals(2));
        });
        if (opacity - self.inspect.overlay_opacity).abs() > f32::EPSILON {
            self.inspect.overlay_opacity = opacity;
            self.invalidate_preview();
        }
        let mut palette = self.preferences.values.overlay_palette;
        let mut pattern = self.preferences.values.overlay_pattern;
        widgets::property_row(ui, "Colours", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                for option in substancecraft_core::OverlayPalette::ALL {
                    if widgets::chip(ui, option.name(), palette == option).clicked() {
                        palette = option;
                    }
                }
            });
        });
        widgets::property_row(ui, "Pattern", |ui| {
            widgets::toggle(ui, &mut pattern)
                .on_hover_text("Hatch blocked areas and dot partial ones, so hue is not needed");
        });
        let [shown, blocked, partial] = palette.colors();
        widgets::property_row(ui, "Key", |ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            for (color, label) in [(shown, "shown"), (blocked, "blocked"), (partial, "partial")] {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().rect_filled(
                    rect,
                    2.0,
                    egui::Color32::from_rgb(color[0], color[1], color[2]),
                );
                ui.label(RichText::new(label).size(11.5).color(theme::TEXT_WEAK));
            }
        });
        if palette != self.preferences.values.overlay_palette
            || pattern != self.preferences.values.overlay_pattern
        {
            self.preferences.values.overlay_palette = palette;
            self.preferences.values.overlay_pattern = pattern;
            self.save_preferences();
            self.invalidate_preview();
        }
    }

    /// The Inspect and Solo controls of the Tools drawer.
    pub(super) fn inspect_section(&mut self, ui: &mut Ui) {
        widgets::caption(ui, "Inspect");
        let armed = self.inspect.armed;
        if widgets::pill_button(
            ui,
            None,
            if armed {
                "Click the model to inspect…"
            } else {
                "Inspect texel"
            },
            armed,
            true,
        )
        .on_hover_text(format!(
            "List the layers that make up the {} channel where you click",
            channel_label(self.active_channel)
        ))
        .clicked()
        {
            self.inspect.armed = !armed;
            self.picking_material = false;
        }
        let mut select = None;
        if let Some(result) = &self.inspect.result {
            ui.label(
                RichText::new(format!(
                    "{} at UV {:.3}, {:.3}",
                    channel_label(result.channel),
                    result.uv.x,
                    result.uv.y
                ))
                .size(11.5)
                .color(theme::TEXT_FAINT),
            );
            match &result.entries {
                Err(error) => {
                    ui.label(RichText::new(error).size(12.0).color(theme::TEXT_WEAK));
                }
                Ok(entries) if entries.is_empty() => {
                    ui.label(
                        RichText::new("No layer paints here; the channel default shows.")
                            .size(12.0)
                            .color(theme::TEXT_WEAK),
                    );
                }
                Ok(entries) => {
                    for (index, entry) in entries.iter().enumerate() {
                        let selected = self.selected_layer == Some(entry.layer);
                        let response = widgets::list_row(
                            ui,
                            Id::new(("inspect_row", index)),
                            selected,
                            |ui| {
                                ui.add_space(entry.depth as f32 * 12.0);
                                ui.label(RichText::new(&entry.name).size(12.5).color(theme::TEXT));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            RichText::new(format!(
                                                "{:.0}%",
                                                entry.effective * 100.0
                                            ))
                                            .size(12.0)
                                            .color(theme::TEXT_WEAK),
                                        );
                                    },
                                );
                            },
                        );
                        let mut details = format!(
                            "Coverage {:.2} · opacity {:.2} · mask {:.2}",
                            entry.coverage, entry.opacity, entry.mask
                        );
                        for driver in &entry.drivers {
                            details.push('\n');
                            details.push_str(driver);
                        }
                        if response.on_hover_text(details).clicked() {
                            select = Some(entry.layer);
                        }
                    }
                }
            }
        }
        if let Some(layer) = select {
            self.selected_layer = Some(layer);
        }

        widgets::caption(ui, "Solo");
        let layer = self.selected_layer;
        let has_generator = self
            .selected_layer()
            .is_some_and(|layer| layer.mask_generator.is_some());
        let current = self.active_solo();
        let mut choice = None;
        widgets::property_row(ui, "Show", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if widgets::chip(ui, "All", current.is_none()).clicked() {
                choice = Some(None);
            }
            if let Some(id) = layer {
                if widgets::chip(ui, "Layer", current == Some(Solo::Layer(id)))
                    .on_hover_text("Show only the selected layer or group")
                    .clicked()
                {
                    choice = Some(Some(Solo::Layer(id)));
                }
                if widgets::chip(ui, "Mask", current == Some(Solo::Mask(id)))
                    .on_hover_text("Show the selected layer's final mask in grayscale")
                    .clicked()
                {
                    choice = Some(Some(Solo::Mask(id)));
                }
                if widgets::chip(ui, "Overlay", current == Some(Solo::Overlay(id)))
                    .on_hover_text("Tint the shaded model where the mask shows, blocks, or partly shows the layer")
                    .clicked()
                {
                    choice = Some(Some(Solo::Overlay(id)));
                }
                if has_generator
                    && widgets::chip(ui, "Generator", current == Some(Solo::Generator(id)))
                        .on_hover_text("Show the selected layer's generator alone")
                        .clicked()
                {
                    choice = Some(Some(Solo::Generator(id)));
                }
            }
            if widgets::chip(ui, "Set", current == Some(Solo::TextureSet))
                .on_hover_text("Grey out every other texture set")
                .clicked()
            {
                choice = Some(Some(Solo::TextureSet));
            }
        });
        if let Some(choice) = choice {
            self.set_solo(choice);
        }
        if matches!(current, Some(Solo::Overlay(_))) {
            self.overlay_controls(ui);
        }
        ui.label(
            RichText::new(
                "Solo only changes the viewport; saving and exporting use the full stack. \
                 Pick a channel in the status bar to solo it.",
            )
            .size(11.5)
            .color(theme::TEXT_FAINT),
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{PreviewTarget, SlotSource};
    use substancecraft_core::LayerFill;

    #[test]
    fn tc_0243_solo_mask_view_and_inspector_work_without_changing_the_project() {
        let mut app = SubstanceCraftApp::default();
        app.add_texture_set();
        for size in &mut app.project.texture_sets {
            size.resolution = substancecraft_core::TextureResolution::new(32, 32).unwrap();
        }
        app.add_fill_layer();
        let fill_id = app.selected_layer.unwrap();
        if let Some(layer) = app.selected_layer_mut() {
            layer.fill = Some(LayerFill {
                roughness: 0.9,
                ..LayerFill::default()
            });
        }
        app.add_paint_layer();
        let saved = app.project.clone();

        // Inspecting a texel lists the fill; choosing it selects it.
        app.active_channel = MaterialChannel::Roughness;
        app.inspect_at(glam::Vec2::new(0.5, 0.5));
        let entries = app
            .inspect
            .result
            .as_ref()
            .unwrap()
            .entries
            .clone()
            .unwrap();
        assert!(entries.iter().any(|entry| entry.layer == fill_id));
        app.active_channel = MaterialChannel::Id;
        app.inspect_at(glam::Vec2::new(0.5, 0.5));
        assert!(app.inspect.result.as_ref().unwrap().entries.is_err());

        // Mask view previews the fill's (open) mask as white.
        app.selected_layer = Some(fill_id);
        app.set_solo(Some(Solo::Mask(fill_id)));
        assert_eq!(app.preview_target(), PreviewTarget::LayerMask(fill_id));
        app.refresh_baked_preview();
        let context = egui::Context::default();
        crate::tests::pump_until(&mut app, &context, |app| {
            app.baked_preview
                .as_ref()
                .is_some_and(|preview| preview.key == app.preview_key())
        });
        let Some(crate::PreviewTextures::Single(mask)) = app.displayed_preview() else {
            panic!("mask view shows one texture");
        };
        assert!(mask.rgba8.iter().all(|byte| *byte == 255));

        // The overlay tints the shaded material: an open mask reads "shown".
        app.set_solo(Some(Solo::Overlay(fill_id)));
        app.inspect.overlay_opacity = 1.0;
        app.preferences.values.overlay_palette = substancecraft_core::OverlayPalette::RedGreenSafe;
        assert!(matches!(
            app.preview_target(),
            PreviewTarget::MaskOverlay { opacity: 100, .. }
        ));
        crate::tests::pump_until(&mut app, &context, |app| {
            app.baked_preview
                .as_ref()
                .is_some_and(|preview| preview.key == app.preview_key())
        });
        let Some(crate::PreviewTextures::Material(material)) = app.displayed_preview() else {
            panic!("the overlay shows the shaded material");
        };
        let shown = substancecraft_core::OverlayPalette::RedGreenSafe.colors()[0];
        assert_eq!(material.base_color.rgba8[0..3], shown);

        // Layer solo bakes the material with only the fill visible.
        app.set_solo(Some(Solo::Layer(fill_id)));
        assert_eq!(app.preview_target(), PreviewTarget::Material);
        assert_eq!(app.preview_key().solo_layer, Some(fill_id));
        // A generator solo needs a generator.
        app.set_solo(Some(Solo::Generator(fill_id)));
        assert_eq!(app.active_solo(), None);

        // Set solo greys other slots only.
        app.set_solo(Some(Solo::TextureSet));
        assert!(matches!(
            app.slot_upload_plan(PreviewTarget::Material)[0],
            SlotSource::Neutral | SlotSource::Material(_)
        ));
        app.set_solo(None);
        // None of this touched the project.
        assert_eq!(app.project, saved);
    }
}
