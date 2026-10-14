//! Editing a procedural fill graph as an ordered list of nodes: each node
//! with its settings and the inputs it reads, then which node feeds each
//! channel. Nodes only read earlier ones, so the list order is the graph.

use eframe::egui::{self, Color32, Id, RichText, Ui};
use substancecraft_core::{
    BlendMode, GraphInput, GraphNode, LayerFilter, MAX_GRAPH_NODES, MAX_PROJECTION_SCALE,
    MAX_RAMP_STOPS, MIN_PROJECTION_SCALE, MaskGenerator, MaterialChannel, MaterialGraph, MathOp,
    MeshMapKind, NoisePattern, ProjectionMode, RampStop, ResourceId, SourceComponent,
    TextureProjection,
};

use crate::icons::Icon;
use crate::panels::{filter_controls, generator_controls, hint, resource_picker, slider_row};
use crate::{theme, widgets};

/// A starting graph: noise through a grey ramp for colour, the noise
/// itself for roughness (where the layer writes them).
fn starter(channels: &[(MaterialChannel, String)]) -> MaterialGraph {
    let writes = |channel| channels.iter().any(|(candidate, _)| *candidate == channel);
    let mut outputs = std::collections::BTreeMap::new();
    if writes(MaterialChannel::BaseColor) {
        outputs.insert(MaterialChannel::BaseColor, GraphInput::Node(1));
    }
    if writes(MaterialChannel::Roughness) {
        outputs.insert(MaterialChannel::Roughness, GraphInput::Node(0));
    }
    MaterialGraph {
        nodes: vec![
            GraphNode::Generator(MaskGenerator::Noise(NoisePattern::DEFAULT)),
            GraphNode::Ramp {
                input: GraphInput::Node(0),
                stops: vec![
                    RampStop {
                        position: 0.0,
                        color: [60, 60, 64],
                    },
                    RampStop {
                        position: 1.0,
                        color: [200, 200, 196],
                    },
                ],
            },
        ],
        outputs,
    }
}

/// Every kind of node, as added: reading the last node where it has an
/// input.
fn templates(last: GraphInput, resource: Option<ResourceId>) -> Vec<GraphNode> {
    let mut nodes = vec![
        GraphNode::Generator(MaskGenerator::Noise(NoisePattern::DEFAULT)),
        GraphNode::MeshMap(MeshMapKind::Curvature),
        GraphNode::Uv,
    ];
    if let Some(resource) = resource {
        nodes.push(GraphNode::Bitmap {
            resource,
            projection: TextureProjection::default(),
        });
    }
    nodes.extend([
        GraphNode::Blend {
            mode: BlendMode::Normal,
            a: last,
            b: GraphInput::gray(0.5),
            amount: GraphInput::gray(0.5),
        },
        GraphNode::Math {
            op: MathOp::Multiply,
            a: last,
            b: GraphInput::gray(1.0),
        },
        GraphNode::Adjust {
            input: last,
            filter: LayerFilter::Invert,
        },
        GraphNode::Ramp {
            input: last,
            stops: vec![
                RampStop {
                    position: 0.0,
                    color: [0, 0, 0],
                },
                RampStop {
                    position: 1.0,
                    color: [255, 255, 255],
                },
            ],
        },
        GraphNode::Split {
            input: last,
            component: SourceComponent::Red,
        },
    ]);
    nodes
}

/// `"3. Ramp"` for node index 2.
fn node_label(nodes: &[GraphNode], index: usize) -> String {
    nodes.get(index).map_or_else(String::new, |node| {
        format!("{}. {}", index + 1, node.name())
    })
}

/// Edits `graph` in place (`None` offers to make the fill procedural).
/// `channels` are the channels the fill writes, labelled; `resources` the
/// project bitmaps. Changes are recorded with the layer's other settings.
pub(super) fn graph_controls(
    ui: &mut Ui,
    id_salt: Id,
    graph: &mut Option<MaterialGraph>,
    channels: &[(MaterialChannel, String)],
    resources: &[(ResourceId, String)],
) {
    widgets::caption(ui, "Procedural");
    let Some(current) = graph else {
        if widgets::pill_button(ui, Some(Icon::Plus), "Make procedural", false, true)
            .on_hover_text("Drive the fill's channels from a graph of generators and adjustments")
            .clicked()
        {
            *graph = Some(starter(channels));
        }
        return;
    };
    let mut remove = None;
    let labels: Vec<String> = (0..current.nodes.len())
        .map(|index| node_label(&current.nodes, index))
        .collect();
    for (index, node) in current.nodes.iter_mut().enumerate() {
        ui.push_id(("graph_node", index), |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&labels[index]).color(theme::TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if widgets::header_action(ui, Icon::Close, true, "Remove node").clicked() {
                        remove = Some(index);
                    }
                });
            });
            node_controls(
                ui,
                id_salt.with(("node", index)),
                node,
                &labels[..index],
                resources,
            );
        });
    }
    if let Some(index) = remove {
        current.remove_node(index);
    }

    if current.nodes.len() < MAX_GRAPH_NODES {
        let last = current
            .nodes
            .len()
            .checked_sub(1)
            .map_or(GraphInput::gray(0.5), |index| {
                GraphInput::Node(index as u16)
            });
        let mut added = None;
        let combo = egui::ComboBox::from_id_salt(id_salt.with("add_node"))
            .width(ui.available_width() - 8.0)
            .selected_text("+ Add node")
            .show_ui(ui, |ui| {
                for template in templates(last, resources.first().map(|(id, _)| *id)) {
                    if ui.selectable_label(false, template.name()).clicked() {
                        added = Some(template);
                    }
                }
            });
        widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
        if let Some(node) = added {
            current.nodes.push(node);
        }
    }

    widgets::caption(ui, "Graph outputs");
    let labels: Vec<String> = (0..current.nodes.len())
        .map(|index| node_label(&current.nodes, index))
        .collect();
    for (channel, label) in channels {
        let selected = current.outputs.get(channel).copied();
        let text = match selected {
            None => "Fill value".to_owned(),
            Some(GraphInput::Node(node)) => {
                labels.get(usize::from(node)).cloned().unwrap_or_default()
            }
            Some(GraphInput::Value(_)) => "Constant".to_owned(),
        };
        let mut chosen = selected;
        widgets::property_row(ui, label, |ui| {
            let combo = egui::ComboBox::from_id_salt(id_salt.with(("output", *channel)))
                .width(ui.available_width() - 8.0)
                .selected_text(text)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(selected.is_none(), "Fill value")
                        .clicked()
                    {
                        chosen = None;
                    }
                    for (index, name) in labels.iter().enumerate() {
                        let node = GraphInput::Node(index as u16);
                        if ui.selectable_label(selected == Some(node), name).clicked() {
                            chosen = Some(node);
                        }
                    }
                });
            widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
        });
        if chosen != selected {
            match chosen {
                Some(input) => current.outputs.insert(*channel, input),
                None => current.outputs.remove(channel),
            };
        }
    }
    hint(
        ui,
        "Each node reads only nodes above it. Channels left on Fill value keep the \
         fill's value or bitmap; layer masks still apply.",
    );
    if widgets::pill_button(ui, None, "Remove procedural", false, false).clicked() {
        *graph = None;
    }
}

/// One node's settings and inputs; `earlier` labels the nodes it may read.
fn node_controls(
    ui: &mut Ui,
    id: Id,
    node: &mut GraphNode,
    earlier: &[String],
    resources: &[(ResourceId, String)],
) {
    match node {
        GraphNode::Generator(generator) => {
            let mut chosen = None;
            let combo = egui::ComboBox::from_id_salt(id.with("generator"))
                .width(ui.available_width() - 8.0)
                .selected_text(generator.name())
                .show_ui(ui, |ui| {
                    for preset in MaskGenerator::PRESETS {
                        // Graphs do not read the ID map.
                        if matches!(preset, MaskGenerator::IdColor(_)) {
                            continue;
                        }
                        let selected = generator.same_kind(&preset);
                        if ui.selectable_label(selected, preset.name()).clicked() && !selected {
                            chosen = Some(preset);
                        }
                    }
                });
            widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
            if let Some(preset) = chosen {
                *generator = preset;
            }
            generator_controls(ui, generator);
        }
        GraphNode::MeshMap(kind) => {
            widgets::property_row(ui, "Map", |ui| {
                let combo = egui::ComboBox::from_id_salt(id.with("map"))
                    .width(ui.available_width() - 8.0)
                    .selected_text(kind.name())
                    .show_ui(ui, |ui| {
                        for option in MeshMapKind::ALL {
                            ui.selectable_value(kind, option, option.name());
                        }
                    });
                widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
            });
        }
        GraphNode::Uv => hint(ui, "U in red, V in green."),
        GraphNode::Bitmap {
            resource,
            projection,
        } => {
            let mut picked = Some(*resource);
            widgets::property_row(ui, "Bitmap", |ui| {
                resource_picker(ui, id.with("bitmap"), &mut picked, resources);
            });
            if let Some(picked) = picked {
                *resource = picked;
            }
            widgets::property_row(ui, "Projection", |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                    for mode in ProjectionMode::ALL {
                        if mode != ProjectionMode::Decal
                            && widgets::chip(ui, mode.name(), projection.mode == mode).clicked()
                        {
                            projection.mode = mode;
                        }
                    }
                });
            });
            slider_row(
                ui,
                "Scale",
                &mut projection.scale,
                MIN_PROJECTION_SCALE..=MAX_PROJECTION_SCALE,
                true,
                2,
            );
        }
        GraphNode::Blend { mode, a, b, amount } => {
            widgets::property_row(ui, "Mode", |ui| {
                let combo = egui::ComboBox::from_id_salt(id.with("mode"))
                    .width(ui.available_width() - 8.0)
                    .selected_text(mode.name())
                    .show_ui(ui, |ui| {
                        for option in BlendMode::ALL {
                            ui.selectable_value(mode, option, option.name());
                        }
                    });
                widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
            });
            input_row(ui, id.with("a"), "Under", a, earlier);
            input_row(ui, id.with("b"), "Over", b, earlier);
            input_row(ui, id.with("amount"), "Amount", amount, earlier);
        }
        GraphNode::Math { op, a, b } => {
            widgets::property_row(ui, "Operation", |ui| {
                let combo = egui::ComboBox::from_id_salt(id.with("op"))
                    .width(ui.available_width() - 8.0)
                    .selected_text(op.name())
                    .show_ui(ui, |ui| {
                        for option in MathOp::ALL {
                            ui.selectable_value(op, option, option.name());
                        }
                    });
                widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
            });
            input_row(ui, id.with("a"), "A", a, earlier);
            input_row(ui, id.with("b"), "B", b, earlier);
        }
        GraphNode::Adjust { input, filter } => {
            input_row(ui, id.with("input"), "Input", input, earlier);
            let mut chosen = None;
            widgets::property_row(ui, "Filter", |ui| {
                let combo = egui::ComboBox::from_id_salt(id.with("filter"))
                    .width(ui.available_width() - 8.0)
                    .selected_text(filter.name())
                    .show_ui(ui, |ui| {
                        for preset in LayerFilter::PRESETS
                            .into_iter()
                            .filter(|preset| preset.is_per_texel())
                        {
                            let selected = preset.name() == filter.name();
                            if ui.selectable_label(selected, preset.name()).clicked() && !selected {
                                chosen = Some(preset);
                            }
                        }
                    });
                widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
            });
            if let Some(preset) = chosen {
                *filter = preset;
            }
            filter_controls(ui, filter);
        }
        GraphNode::Ramp { input, stops } => {
            input_row(ui, id.with("input"), "Input", input, earlier);
            ramp_controls(ui, id.with("stops"), stops);
        }
        GraphNode::Split { input, component } => {
            input_row(ui, id.with("input"), "Input", input, earlier);
            widgets::property_row(ui, "Component", |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                for (option, label) in [
                    (SourceComponent::Red, "R"),
                    (SourceComponent::Green, "G"),
                    (SourceComponent::Blue, "B"),
                ] {
                    if widgets::chip(ui, label, *component == option).clicked() {
                        *component = option;
                    }
                }
            });
        }
    }
}

/// An input: an earlier node, or a constant edited as a number (setting
/// all three components) or a colour.
fn input_row(ui: &mut Ui, id: Id, label: &str, input: &mut GraphInput, earlier: &[String]) {
    let text = match input {
        GraphInput::Node(node) => earlier.get(usize::from(*node)).cloned().unwrap_or_default(),
        GraphInput::Value(_) => "Constant".to_owned(),
    };
    let mut chosen = None;
    widgets::property_row(ui, label, |ui| {
        let combo = egui::ComboBox::from_id_salt(id)
            .width(ui.available_width() - 8.0)
            .selected_text(text)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(matches!(input, GraphInput::Value(_)), "Constant")
                    .clicked()
                {
                    chosen = Some(GraphInput::gray(0.5));
                }
                for (index, name) in earlier.iter().enumerate() {
                    let node = GraphInput::Node(index as u16);
                    if ui.selectable_label(*input == node, name).clicked() {
                        chosen = Some(node);
                    }
                }
            });
        widgets::label_from_row(ui, &combo.response, egui::WidgetType::ComboBox);
    });
    if let Some(chosen) = chosen
        && !matches!(
            (&chosen, &*input),
            (GraphInput::Value(_), GraphInput::Value(_))
        )
    {
        *input = chosen;
    }
    if let GraphInput::Value(value) = input {
        widgets::property_row(ui, "", |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            let mut level = value[0];
            if ui
                .add(
                    egui::DragValue::new(&mut level)
                        .speed(0.01)
                        .range(-100.0..=100.0),
                )
                .on_hover_text("Sets all three components")
                .changed()
            {
                *value = [level; 3];
            }
            let to_byte = |component: f32| (component.clamp(0.0, 1.0) * 255.0).round() as u8;
            let mut color =
                Color32::from_rgb(to_byte(value[0]), to_byte(value[1]), to_byte(value[2]));
            if widgets::labeled_color_button(ui, &mut color).changed() {
                let rgba = color.to_array();
                *value = [rgba[0], rgba[1], rgba[2]].map(|channel| f32::from(channel) / 255.0);
            }
        });
    }
}

/// A ramp's stops: position and colour each, kept in order; 2 to 8.
fn ramp_controls(ui: &mut Ui, id: Id, stops: &mut Vec<RampStop>) {
    let mut remove = None;
    let count = stops.len();
    for (index, stop) in stops.iter_mut().enumerate() {
        ui.push_id(id.with(index), |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                let mut color = Color32::from_rgb(stop.color[0], stop.color[1], stop.color[2]);
                if widgets::labeled_color_button(ui, &mut color).changed() {
                    let rgba = color.to_array();
                    stop.color = [rgba[0], rgba[1], rgba[2]];
                }
                ui.spacing_mut().slider_width = (ui.available_width() - 70.0).max(40.0);
                ui.add(egui::Slider::new(&mut stop.position, 0.0..=1.0).max_decimals(2))
                    .on_hover_text("Stop position");
                if count > 2
                    && widgets::header_action(ui, Icon::Close, true, "Remove stop").clicked()
                {
                    remove = Some(index);
                }
            });
        });
    }
    if let Some(index) = remove {
        stops.remove(index);
    }
    stops.sort_by(|a, b| a.position.total_cmp(&b.position));
    if stops.len() < MAX_RAMP_STOPS
        && widgets::pill_button(ui, Some(Icon::Plus), "Add stop", false, false).clicked()
        && let [.., before, last] = stops.as_slice()
    {
        let stop = RampStop {
            position: (before.position + last.position) * 0.5,
            color: last.color,
        };
        stops.insert(stops.len() - 1, stop);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn channels() -> Vec<(MaterialChannel, String)> {
        vec![
            (MaterialChannel::BaseColor, "Base Color".to_owned()),
            (MaterialChannel::Roughness, "Roughness".to_owned()),
        ]
    }

    #[test]
    fn tc_0368_graph_editor_starts_valid_graphs_and_adds_valid_nodes() {
        let mut graph = starter(&channels());
        assert!(graph.is_valid());
        assert_eq!(graph.outputs.len(), 2);
        // Only channels the fill writes get an output.
        let only_color = starter(&channels()[..1]);
        assert_eq!(
            only_color.outputs.keys().copied().collect::<Vec<_>>(),
            [MaterialChannel::BaseColor]
        );
        // Every template, added after the last node, keeps the graph valid.
        for template in templates(GraphInput::Node(1), Some(ResourceId(3))) {
            graph.nodes.push(template);
            assert!(graph.is_valid(), "{:?}", graph.nodes.last());
        }
        assert!(
            templates(GraphInput::gray(0.5), None)
                .iter()
                .all(|node| !matches!(node, GraphNode::Bitmap { .. }))
        );
        assert_eq!(node_label(&graph.nodes, 1), "2. Ramp");

        // The editor draws a full graph and an empty one headlessly.
        let context = egui::Context::default();
        let resources = vec![(ResourceId(3), "Grunge".to_owned())];
        let mut some = Some(graph);
        let mut none = None;
        let mut output = context.run_ui(egui::RawInput::default(), |ui| {
            graph_controls(ui, Id::new("a"), &mut some, &channels(), &resources);
            graph_controls(ui, Id::new("b"), &mut none, &channels(), &resources);
        });
        // Headless: no renderer takes the texture updates.
        output.textures_delta.clear();
        assert!(some.unwrap().is_valid());
        assert!(none.is_none(), "nothing changes without input");
    }
}
