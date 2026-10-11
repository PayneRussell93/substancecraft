//! Color tokens, metrics, and the global egui style for the pill UI.

use eframe::egui::{
    self, Color32, CornerRadius, Margin, Shadow, Stroke, TextStyle, Theme, ThemePreference,
    epaint::FontId,
};

pub const VIEWPORT_BG: Color32 = Color32::from_rgb(22, 23, 26);
pub const SURFACE: Color32 = Color32::from_rgba_premultiplied(27, 28, 32, 238);
pub const SURFACE_SOLID: Color32 = Color32::from_rgb(29, 30, 34);
pub const INPUT: Color32 = Color32::from_rgb(40, 42, 47);
pub const INPUT_DEEP: Color32 = Color32::from_rgb(18, 19, 22);
pub const HOVER: Color32 = Color32::from_rgb(52, 55, 61);
pub const ACTIVE: Color32 = Color32::from_rgb(62, 66, 73);
pub const HAIRLINE: Color32 = Color32::from_rgba_premultiplied(14, 14, 15, 15);
pub const OUTLINE: Color32 = Color32::from_rgba_premultiplied(24, 24, 26, 26);

pub const TEXT: Color32 = Color32::from_rgb(229, 231, 235);
pub const TEXT_WEAK: Color32 = Color32::from_rgb(139, 144, 153);
pub const TEXT_FAINT: Color32 = Color32::from_rgb(92, 96, 104);

pub const ACCENT: Color32 = Color32::from_rgb(56, 201, 196);
pub const ACCENT_TEXT: Color32 = Color32::from_rgb(10, 30, 30);
pub const ACCENT_SOFT: Color32 = Color32::from_rgba_premultiplied(11, 40, 39, 51);
pub const DANGER: Color32 = Color32::from_rgb(255, 112, 112);
pub const DANGER_SOFT: Color32 = Color32::from_rgba_premultiplied(64, 22, 24, 230);

/// Gap between floating panels and the window edge.
pub const EDGE: f32 = 14.0;
/// Gap between neighbouring floating panels.
pub const GAP: f32 = 10.0;
pub const CARD_RADIUS: f32 = 18.0;
pub const PILL: CornerRadius = CornerRadius::same(255);

pub fn panel_shadow() -> Shadow {
    Shadow {
        offset: [0, 8],
        blur: 28,
        spread: 0,
        color: Color32::from_black_alpha(110),
    }
}

/// Frame for a floating pill (toolbars, dock, status bar).
pub fn pill_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(Stroke::new(1.0, OUTLINE))
        .corner_radius(PILL)
        .shadow(panel_shadow())
        .inner_margin(Margin::symmetric(6, 6))
}

/// Frame for a floating card whose corners morph between card and pill.
pub fn card_frame(radius: f32) -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(Stroke::new(1.0, OUTLINE))
        .corner_radius(radius)
        .shadow(panel_shadow())
        .inner_margin(Margin::symmetric(10, 8))
}

pub fn apply(context: &egui::Context) {
    context.set_theme(ThemePreference::Dark);
    context.style_mut_of(Theme::Dark, |style| {
        style.text_styles = [
            (TextStyle::Small, FontId::proportional(10.5)),
            (TextStyle::Body, FontId::proportional(13.0)),
            (TextStyle::Button, FontId::proportional(13.0)),
            (TextStyle::Heading, FontId::proportional(15.0)),
            (TextStyle::Monospace, FontId::monospace(12.0)),
        ]
        .into();

        let spacing = &mut style.spacing;
        spacing.item_spacing = egui::vec2(8.0, 6.0);
        spacing.button_padding = egui::vec2(12.0, 5.0);
        spacing.interact_size = egui::vec2(36.0, 26.0);
        spacing.slider_width = 150.0;
        spacing.slider_rail_height = 4.0;
        spacing.combo_width = 140.0;
        spacing.menu_margin = Margin::same(6);
        spacing.window_margin = Margin::same(10);

        let visuals = &mut style.visuals;
        visuals.dark_mode = true;
        visuals.panel_fill = VIEWPORT_BG;
        visuals.window_fill = SURFACE_SOLID;
        visuals.window_stroke = Stroke::new(1.0, OUTLINE);
        visuals.window_corner_radius = CornerRadius::same(16);
        visuals.window_shadow = panel_shadow();
        visuals.menu_corner_radius = CornerRadius::same(14);
        visuals.popup_shadow = panel_shadow();
        visuals.extreme_bg_color = INPUT_DEEP;
        visuals.text_edit_bg_color = Some(INPUT);
        visuals.faint_bg_color = INPUT;
        visuals.weak_text_color = Some(TEXT_WEAK);
        visuals.error_fg_color = DANGER;
        visuals.hyperlink_color = ACCENT;
        visuals.slider_trailing_fill = true;
        visuals.handle_shape = egui::style::HandleShape::Circle;
        visuals.selection.bg_fill = ACCENT;
        visuals.selection.stroke = Stroke::new(1.0, ACCENT_TEXT);
        visuals.collapsing_header_frame = false;
        visuals.indent_has_left_vline = false;

        let widgets = &mut visuals.widgets;
        widgets.noninteractive.bg_fill = SURFACE_SOLID;
        widgets.noninteractive.weak_bg_fill = SURFACE_SOLID;
        widgets.noninteractive.bg_stroke = Stroke::new(1.0, HAIRLINE);
        widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        widgets.noninteractive.corner_radius = CornerRadius::same(10);

        for (state, fill, text) in [
            (&mut widgets.inactive, INPUT, TEXT),
            (&mut widgets.hovered, HOVER, Color32::WHITE),
            (&mut widgets.active, ACTIVE, Color32::WHITE),
            (&mut widgets.open, HOVER, TEXT),
        ] {
            state.bg_fill = fill;
            state.weak_bg_fill = fill;
            state.bg_stroke = Stroke::NONE;
            state.fg_stroke = Stroke::new(1.5, text);
            // Half the interact height: a full pill for standard widgets. Avoid PILL here,
            // since sliders extend their trailing fill by this radius.
            state.corner_radius = CornerRadius::same(13);
            state.expansion = 0.0;
        }
        widgets.hovered.bg_stroke = Stroke::new(1.0, OUTLINE);
    });
}
