//! Stroke icons painted on a 24×24 design grid, so they stay crisp at any scale
//! without depending on glyph coverage in the bundled fonts.

use eframe::egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, pos2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    Cube,
    Brush,
    Figure,
    Sun,
    Folder,
    Save,
    Undo,
    Redo,
    Eye,
    EyeOff,
    Plus,
    Close,
    Layers,
    Grid,
    Focus,
    Export,
    Sidebar,
    Sliders,
    Alert,
    Import,
    ArrowUp,
    ArrowDown,
    Trash,
    Bake,
}

pub fn paint(painter: &Painter, rect: Rect, icon: Icon, color: Color32) {
    let scale = rect.width().min(rect.height()) / 24.0;
    let origin = rect.center() - eframe::egui::vec2(12.0, 12.0) * scale;
    let p = |x: f32, y: f32| origin + eframe::egui::vec2(x, y) * scale;
    let stroke = Stroke::new((1.6 * scale).max(1.0), color);
    let line = |points: &[(f32, f32)]| {
        painter.add(Shape::line(
            points.iter().map(|&(x, y)| p(x, y)).collect(),
            stroke,
        ));
    };
    let closed = |points: &[(f32, f32)]| {
        painter.add(Shape::closed_line(
            points.iter().map(|&(x, y)| p(x, y)).collect(),
            stroke,
        ));
    };
    let arc = |center: (f32, f32), radius: f32, from: f32, to: f32| -> Vec<Pos2> {
        let steps = 16;
        (0..=steps)
            .map(|step| {
                let angle = from + (to - from) * step as f32 / steps as f32;
                p(
                    center.0 + radius * angle.cos(),
                    center.1 + radius * angle.sin(),
                )
            })
            .collect()
    };

    match icon {
        Icon::Cube => {
            closed(&[
                (12.0, 3.0),
                (20.0, 7.5),
                (20.0, 16.5),
                (12.0, 21.0),
                (4.0, 16.5),
                (4.0, 7.5),
            ]);
            line(&[(4.0, 7.5), (12.0, 12.0), (20.0, 7.5)]);
            line(&[(12.0, 12.0), (12.0, 21.0)]);
        }
        Icon::Brush => {
            line(&[(19.5, 4.5), (11.5, 12.5)]);
            painter.circle_filled(p(8.5, 15.5), 3.6 * scale, color);
            line(&[(8.5, 15.5), (4.5, 20.0)]);
        }
        Icon::Figure => {
            painter.circle_stroke(p(12.0, 5.5), 2.5 * scale, stroke);
            line(&[(12.0, 8.5), (12.0, 15.0)]);
            line(&[(6.0, 11.0), (18.0, 11.0)]);
            line(&[(8.0, 21.0), (12.0, 15.0), (16.0, 21.0)]);
        }
        Icon::Sun => {
            painter.circle_stroke(p(12.0, 12.0), 4.0 * scale, stroke);
            for ray in 0..8 {
                let angle = ray as f32 * std::f32::consts::FRAC_PI_4;
                let (sin, cos) = angle.sin_cos();
                line(&[
                    (12.0 + cos * 7.0, 12.0 + sin * 7.0),
                    (12.0 + cos * 9.5, 12.0 + sin * 9.5),
                ]);
            }
        }
        Icon::Folder => closed(&[
            (3.0, 6.0),
            (9.0, 6.0),
            (11.0, 8.5),
            (21.0, 8.5),
            (21.0, 19.0),
            (3.0, 19.0),
        ]),
        Icon::Save => {
            closed(&[
                (4.0, 4.0),
                (17.0, 4.0),
                (20.0, 7.0),
                (20.0, 20.0),
                (4.0, 20.0),
            ]);
            line(&[(8.0, 4.0), (8.0, 9.0), (15.0, 9.0), (15.0, 4.0)]);
            closed(&[(7.5, 13.5), (16.5, 13.5), (16.5, 20.0), (7.5, 20.0)]);
        }
        Icon::Undo | Icon::Redo => {
            let flip = |x: f32| if icon == Icon::Redo { 24.0 - x } else { x };
            line(&[(flip(9.0), 5.0), (flip(5.0), 9.0), (flip(9.0), 13.0)]);
            let (from, to) = if icon == Icon::Redo {
                (
                    -std::f32::consts::FRAC_PI_2,
                    -3.0 * std::f32::consts::FRAC_PI_2,
                )
            } else {
                (-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2)
            };
            let mut path = vec![p(flip(5.0), 9.0)];
            path.extend(arc((flip(14.5), 13.5), 4.5, from, to));
            path.push(p(flip(9.0), 18.0));
            painter.add(Shape::line(path, stroke));
        }
        Icon::Eye | Icon::EyeOff => {
            let steps = 14;
            let lens = |sign: f32| -> Vec<Pos2> {
                (0..=steps)
                    .map(|step| {
                        let t = step as f32 / steps as f32;
                        p(
                            3.0 + 18.0 * t,
                            12.0 + sign * 6.5 * (t * std::f32::consts::PI).sin(),
                        )
                    })
                    .collect()
            };
            painter.add(Shape::line(lens(-1.0), stroke));
            painter.add(Shape::line(lens(1.0), stroke));
            painter.circle_stroke(p(12.0, 12.0), 2.6 * scale, stroke);
            if icon == Icon::EyeOff {
                line(&[(4.0, 4.0), (20.0, 20.0)]);
            }
        }
        Icon::Plus => {
            line(&[(12.0, 5.0), (12.0, 19.0)]);
            line(&[(5.0, 12.0), (19.0, 12.0)]);
        }
        Icon::Close => {
            line(&[(6.5, 6.5), (17.5, 17.5)]);
            line(&[(17.5, 6.5), (6.5, 17.5)]);
        }
        Icon::Layers => {
            closed(&[(12.0, 4.0), (20.5, 8.5), (12.0, 13.0), (3.5, 8.5)]);
            line(&[(3.5, 12.5), (12.0, 17.0), (20.5, 12.5)]);
            line(&[(3.5, 16.0), (12.0, 20.5), (20.5, 16.0)]);
        }
        Icon::Grid => {
            for (x, y) in [(4.0, 4.0), (13.5, 4.0), (4.0, 13.5), (13.5, 13.5)] {
                closed(&[(x, y), (x + 6.5, y), (x + 6.5, y + 6.5), (x, y + 6.5)]);
            }
        }
        Icon::Focus => {
            line(&[(4.0, 9.0), (4.0, 4.0), (9.0, 4.0)]);
            line(&[(15.0, 4.0), (20.0, 4.0), (20.0, 9.0)]);
            line(&[(20.0, 15.0), (20.0, 20.0), (15.0, 20.0)]);
            line(&[(9.0, 20.0), (4.0, 20.0), (4.0, 15.0)]);
            painter.circle_filled(p(12.0, 12.0), 2.2 * scale, color);
        }
        Icon::Export | Icon::Import => {
            line(&[(4.0, 15.0), (4.0, 20.0), (20.0, 20.0), (20.0, 15.0)]);
            line(&[(12.0, 4.0), (12.0, 15.0)]);
            if icon == Icon::Export {
                line(&[(8.0, 11.0), (12.0, 15.0), (16.0, 11.0)]);
            } else {
                line(&[(8.0, 8.0), (12.0, 4.0), (16.0, 8.0)]);
            }
        }
        Icon::Sidebar => {
            closed(&[(3.0, 5.0), (21.0, 5.0), (21.0, 19.0), (3.0, 19.0)]);
            line(&[(15.0, 5.0), (15.0, 19.0)]);
        }
        Icon::Sliders => {
            for (y, knob) in [(7.0, 9.0), (12.0, 15.5), (17.0, 7.0)] {
                line(&[(4.0, y), (20.0, y)]);
                painter.circle_filled(p(knob, y), 2.4 * scale, color);
            }
        }
        Icon::Alert => {
            closed(&[(12.0, 3.5), (21.0, 19.5), (3.0, 19.5)]);
            line(&[(12.0, 9.5), (12.0, 13.5)]);
            painter.circle_filled(p(12.0, 16.6), 1.1 * scale, color);
        }
        Icon::ArrowUp | Icon::ArrowDown => {
            let flip = |y: f32| if icon == Icon::ArrowDown { 24.0 - y } else { y };
            line(&[(12.0, flip(19.0)), (12.0, flip(5.0))]);
            line(&[(6.5, flip(10.5)), (12.0, flip(5.0)), (17.5, flip(10.5))]);
        }
        Icon::Trash => {
            line(&[(4.0, 7.0), (20.0, 7.0)]);
            line(&[(9.5, 7.0), (9.5, 4.5), (14.5, 4.5), (14.5, 7.0)]);
            line(&[(6.0, 7.0), (7.0, 20.0), (17.0, 20.0), (18.0, 7.0)]);
            line(&[(10.0, 10.5), (10.0, 16.5)]);
            line(&[(14.0, 10.5), (14.0, 16.5)]);
        }
        Icon::Bake => {
            closed(&[(4.0, 9.0), (20.0, 9.0), (20.0, 20.0), (4.0, 20.0)]);
            closed(&[(7.5, 12.0), (16.5, 12.0), (16.5, 17.0), (7.5, 17.0)]);
            for x in [8.0, 12.0, 16.0] {
                line(&[(x, 6.5), (x - 1.0, 4.5), (x, 2.5)]);
            }
        }
    }
}

/// Rotates a chevron-down icon by `angle` radians around the rect centre.
pub fn paint_chevron(painter: &Painter, rect: Rect, angle: f32, color: Color32) {
    let scale = rect.width().min(rect.height()) / 24.0;
    let center = rect.center();
    let (sin, cos) = angle.sin_cos();
    let points = [(7.0, 9.5), (12.0, 14.5), (17.0, 9.5)]
        .iter()
        .map(|&(x, y)| {
            let (dx, dy) = ((x - 12.0) * scale, (y - 12.0) * scale);
            pos2(
                center.x + dx * cos - dy * sin,
                center.y + dx * sin + dy * cos,
            )
        })
        .collect();
    painter.add(Shape::line(
        points,
        Stroke::new((1.6 * scale).max(1.0), color),
    ));
}
