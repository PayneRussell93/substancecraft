//! Pill-shaped building blocks shared by every floating panel.

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Id, Rect, Response, RichText, Sense, Stroke,
    StrokeKind, Ui, UiBuilder, collapsing_header::CollapsingState, emath::lerp, vec2,
};

use crate::icons::{self, Icon};
use crate::theme;

const ICON_BUTTON: f32 = 30.0;

/// Circular icon button. Disabled buttons never report clicks.
pub fn icon_button(ui: &mut Ui, icon: Icon, enabled: bool, tooltip: &str) -> Response {
    icon_toggle(ui, icon, false, enabled, tooltip)
}

/// Circular icon button that renders an accent ring while `selected`.
pub fn icon_toggle(
    ui: &mut Ui,
    icon: Icon,
    selected: bool,
    enabled: bool,
    tooltip: &str,
) -> Response {
    sized_icon_toggle(ui, icon, selected, enabled, tooltip, ICON_BUTTON)
}

pub fn sized_icon_toggle(
    ui: &mut Ui,
    icon: Icon,
    selected: bool,
    enabled: bool,
    tooltip: &str,
    size: f32,
) -> Response {
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(vec2(size, size), sense);
    if ui.is_rect_visible(rect) {
        let hover = ui
            .ctx()
            .animate_bool_responsive(response.id, enabled && response.hovered());
        let fill = if selected {
            theme::ACCENT_SOFT
        } else {
            theme::HOVER.gamma_multiply(hover)
        };
        let painter = ui.painter();
        painter.circle_filled(rect.center(), size * 0.5, fill);
        if selected {
            painter.circle_stroke(
                rect.center(),
                size * 0.5 - 0.5,
                Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.6)),
            );
        }
        let color = if !enabled {
            theme::TEXT_FAINT
        } else if selected {
            theme::ACCENT
        } else if response.hovered() {
            Color32::WHITE
        } else {
            theme::TEXT
        };
        icons::paint(painter, rect.shrink(size * 0.24), icon, color);
    }
    if tooltip.is_empty() {
        response
    } else {
        response.on_hover_text(tooltip)
    }
}

/// Thin vertical divider for horizontal pill toolbars.
pub fn divider(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(vec2(9.0, 20.0), Sense::hover());
    ui.painter().line_segment(
        [rect.center_top(), rect.center_bottom()],
        Stroke::new(1.0, theme::OUTLINE),
    );
}

/// Pill segmented control with a sliding accent thumb. Returns true on change.
pub fn segmented<T: PartialEq + Copy>(
    ui: &mut Ui,
    id: Id,
    value: &mut T,
    options: &[(T, &str)],
) -> bool {
    let font = FontId::proportional(12.5);
    let padding = 13.0;
    let height = 30.0;
    let inset = 3.0;
    let galleys = options
        .iter()
        .map(|(_, label)| {
            ui.painter()
                .layout_no_wrap((*label).to_owned(), font.clone(), theme::TEXT)
        })
        .collect::<Vec<_>>();
    let widths = galleys
        .iter()
        .map(|galley| galley.size().x + padding * 2.0)
        .collect::<Vec<_>>();
    let total = widths.iter().sum::<f32>() + inset * 2.0;
    let (rect, _) = ui.allocate_exact_size(vec2(total, height), Sense::hover());

    let mut changed = false;
    let mut x = rect.left() + inset;
    let mut thumb = None;
    let mut segments = Vec::with_capacity(options.len());
    for (index, ((option, _), width)) in options.iter().zip(&widths).enumerate() {
        let segment = Rect::from_min_size(
            egui::pos2(x, rect.top() + inset),
            vec2(*width, height - inset * 2.0),
        );
        let response = ui.interact(segment, id.with(index), Sense::click());
        if response.clicked() && *value != *option {
            *value = *option;
            changed = true;
        }
        if *value == *option {
            thumb = Some(segment);
        }
        segments.push((segment, response.hovered()));
        x += width;
    }

    let painter = ui.painter();
    painter.rect_filled(rect, theme::PILL, theme::INPUT_DEEP);
    if let Some(target) = thumb {
        let left = ui
            .ctx()
            .animate_value_with_time(id.with("thumb_x"), target.left(), 0.16);
        let width = ui
            .ctx()
            .animate_value_with_time(id.with("thumb_w"), target.width(), 0.16);
        let thumb =
            Rect::from_min_size(egui::pos2(left, target.top()), vec2(width, target.height()));
        painter.rect_filled(thumb, theme::PILL, theme::ACCENT);
    }
    for (((option, _), galley), (segment, hovered)) in options.iter().zip(galleys).zip(segments) {
        let color = if *value == *option {
            theme::ACCENT_TEXT
        } else if hovered {
            Color32::WHITE
        } else {
            theme::TEXT_WEAK
        };
        let position = segment.center() - galley.size() * 0.5;
        painter.galley_with_override_text_color(position, galley, color);
    }
    changed
}

/// Small selectable pill, used for presets, poses, and other quick picks.
pub fn chip(ui: &mut Ui, label: &str, selected: bool) -> Response {
    let font = FontId::proportional(12.5);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_owned(), font, theme::TEXT);
    let size = vec2(galley.size().x + 22.0, 26.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    if ui.is_rect_visible(rect) {
        let (fill, stroke, text) = if selected {
            (
                theme::ACCENT_SOFT,
                theme::ACCENT.gamma_multiply(0.7),
                theme::ACCENT,
            )
        } else if response.hovered() {
            (theme::HOVER, theme::OUTLINE, Color32::WHITE)
        } else {
            (theme::INPUT, Color32::TRANSPARENT, theme::TEXT)
        };
        let painter = ui.painter();
        painter.rect(
            rect,
            theme::PILL,
            fill,
            Stroke::new(1.0, stroke),
            StrokeKind::Inside,
        );
        painter.galley_with_override_text_color(rect.center() - galley.size() * 0.5, galley, text);
    }
    response
}

/// Pill button with an optional leading icon. `primary` fills it with the accent.
pub fn pill_button(
    ui: &mut Ui,
    icon: Option<Icon>,
    label: &str,
    primary: bool,
    full_width: bool,
) -> Response {
    let font = FontId::proportional(13.0);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_owned(), font, theme::TEXT);
    let icon_space = if icon.is_some() { 22.0 } else { 0.0 };
    let content_width = galley.size().x + icon_space;
    let width = if full_width {
        ui.available_width()
    } else {
        content_width + 28.0
    };
    let (rect, response) = ui.allocate_exact_size(vec2(width, 32.0), Sense::click());
    if ui.is_rect_visible(rect) {
        let hovered = response.hovered();
        let (fill, text) = match (primary, hovered) {
            (true, false) => (theme::ACCENT, theme::ACCENT_TEXT),
            (true, true) => (
                lerp_color(theme::ACCENT, Color32::WHITE, 0.15),
                theme::ACCENT_TEXT,
            ),
            (false, false) => (theme::INPUT, theme::TEXT),
            (false, true) => (theme::HOVER, Color32::WHITE),
        };
        let painter = ui.painter();
        painter.rect_filled(rect, theme::PILL, fill);
        let start = rect.center().x - content_width * 0.5;
        if let Some(icon) = icon {
            let icon_rect =
                Rect::from_center_size(egui::pos2(start + 8.0, rect.center().y), vec2(16.0, 16.0));
            icons::paint(painter, icon_rect, icon, text);
        }
        painter.galley_with_override_text_color(
            egui::pos2(start + icon_space, rect.center().y - galley.size().y * 0.5),
            galley,
            text,
        );
    }
    response
}

/// iOS-style toggle switch.
pub fn toggle(ui: &mut Ui, on: &mut bool) -> Response {
    let (rect, mut response) = ui.allocate_exact_size(vec2(34.0, 20.0), Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    if ui.is_rect_visible(rect) {
        let amount = ui.ctx().animate_bool_responsive(response.id, *on);
        let fill = lerp_color(theme::INPUT, theme::ACCENT, amount);
        let painter = ui.painter();
        painter.rect_filled(rect, theme::PILL, fill);
        let radius = rect.height() * 0.5 - 3.0;
        let x = lerp(
            (rect.left() + radius + 3.0)..=(rect.right() - radius - 3.0),
            amount,
        );
        painter.circle_filled(egui::pos2(x, rect.center().y), radius, Color32::WHITE);
    }
    response
}

/// Uppercase caption above a group of controls.
pub fn caption(ui: &mut Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(
        RichText::new(text.to_uppercase())
            .size(10.5)
            .extra_letter_spacing(0.8)
            .color(theme::TEXT_FAINT),
    );
}

/// A label column followed by right-aligned controls.
pub fn property_row<R>(ui: &mut Ui, label: &str, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(84.0, 26.0), Sense::hover());
        ui.painter().text(
            rect.left_center(),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(12.5),
            theme::TEXT_WEAK,
        );
        add_contents(ui)
    })
    .inner
}

/// Full-width selectable row. Widgets added by `add_contents` sit on top of the
/// row, so their clicks take priority over the row's own click.
pub fn list_row(
    ui: &mut Ui,
    id: Id,
    selected: bool,
    add_contents: impl FnOnce(&mut Ui),
) -> Response {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 34.0), Sense::hover());
    let response = ui.interact(rect, id, Sense::click());
    if ui.is_rect_visible(rect) {
        let fill = if selected {
            theme::ACCENT_SOFT
        } else if response.hovered() {
            theme::INPUT
        } else {
            Color32::TRANSPARENT
        };
        let stroke = if selected {
            Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.45))
        } else {
            Stroke::NONE
        };
        ui.painter().rect(
            rect,
            CornerRadius::same(12),
            fill,
            stroke,
            StrokeKind::Inside,
        );
    }
    ui.scope_builder(
        UiBuilder::new()
            .max_rect(rect.shrink2(vec2(6.0, 0.0)))
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
        add_contents,
    );
    response
}

/// Collapsible floating card. While collapsing, its corners morph from a card
/// radius into a full pill so a closed card reads as a compact header pill.
/// `actions` are drawn right-aligned in the header; reserve their width with
/// `action_count` so the clickable title area never overlaps them.
pub fn card(
    ui: &mut Ui,
    id: Id,
    icon: Icon,
    title: &str,
    action_count: usize,
    actions: impl FnOnce(&mut Ui),
    body: impl FnOnce(&mut Ui),
) {
    let mut state = CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let openness = state.openness(ui.ctx());
    let radius = lerp(theme::CARD_RADIUS..=26.0, 1.0 - openness);
    theme::card_frame(radius).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            let actions_width = action_count as f32 * (ICON_BUTTON - 4.0);
            let title_width = (ui.available_width() - actions_width).max(40.0);
            let (rect, response) = ui.allocate_exact_size(vec2(title_width, 30.0), Sense::click());
            if response.clicked() {
                state.toggle(ui);
            }
            let painter = ui.painter();
            let hovered = response.hovered();
            let chevron = Rect::from_center_size(
                egui::pos2(rect.left() + 10.0, rect.center().y),
                vec2(16.0, 16.0),
            );
            icons::paint_chevron(
                painter,
                chevron,
                lerp(-std::f32::consts::FRAC_PI_2..=0.0, openness),
                if hovered {
                    theme::TEXT
                } else {
                    theme::TEXT_WEAK
                },
            );
            let icon_rect = Rect::from_center_size(
                egui::pos2(rect.left() + 32.0, rect.center().y),
                vec2(16.0, 16.0),
            );
            icons::paint(painter, icon_rect, icon, theme::ACCENT);
            painter.text(
                egui::pos2(rect.left() + 48.0, rect.center().y),
                Align2::LEFT_CENTER,
                title,
                FontId::proportional(13.5),
                if hovered { Color32::WHITE } else { theme::TEXT },
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                actions(ui);
            });
        });
        state.show_body_unindented(ui, |ui| {
            ui.add_space(4.0);
            body(ui);
            ui.add_space(2.0);
        });
    });
}

/// Small action button for card headers.
pub fn header_action(ui: &mut Ui, icon: Icon, enabled: bool, tooltip: &str) -> Response {
    sized_icon_toggle(ui, icon, false, enabled, tooltip, ICON_BUTTON - 4.0)
}

pub fn lerp_color(from: Color32, to: Color32, amount: f32) -> Color32 {
    let blend = |a: u8, b: u8| lerp(f32::from(a)..=f32::from(b), amount).round() as u8;
    Color32::from_rgba_premultiplied(
        blend(from.r(), to.r()),
        blend(from.g(), to.g()),
        blend(from.b(), to.b()),
        blend(from.a(), to.a()),
    )
}
