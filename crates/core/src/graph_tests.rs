//! Procedural material graphs (features 10 and 21).

#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use glam::{Vec2, Vec3};

use crate::graph::GraphEvaluator;
use crate::{
    BakeContext, BakeError, BlendMode, DEFAULT_HEIGHT_STRENGTH, GraphInput, GraphNode, LayerFill,
    LayerFilter, LayerId, MAX_GRAPH_NODES, MaskGenerator, MaterialChannel, MaterialGraph,
    MaterialLayer, MaterialLink, MaterialProject, MaterialResource, MathOp, MeshMapSettings,
    PrimitivePreset, ProjectError, RampStop, ResourceId, SourceComponent, TextureResolution,
    TextureSet, TextureSetId, bake_mesh_maps,
};

const NODE: fn(u16) -> GraphInput = GraphInput::Node;

fn evaluate(graph: &MaterialGraph, output: GraphInput, uv: Vec2) -> Vec3 {
    let evaluator: GraphEvaluator<'_> = graph.evaluator(None).unwrap();
    evaluator.evaluate(output, uv, None)
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).abs().max_element() < 1e-4
}

/// UV, its U as grey, a half-and-half blend with red, U squared, U
/// inverted, and a ramp over U.
fn uv_graph() -> MaterialGraph {
    MaterialGraph {
        nodes: vec![
            GraphNode::Uv,
            GraphNode::Split {
                input: NODE(0),
                component: SourceComponent::Red,
            },
            GraphNode::Blend {
                mode: BlendMode::Normal,
                a: NODE(1),
                b: GraphInput::Value([1.0, 0.0, 0.0]),
                amount: GraphInput::gray(0.5),
            },
            GraphNode::Math {
                op: MathOp::Power,
                a: NODE(1),
                b: GraphInput::gray(2.0),
            },
            GraphNode::Adjust {
                input: NODE(1),
                filter: LayerFilter::Invert,
            },
            GraphNode::Ramp {
                input: NODE(1),
                stops: vec![
                    RampStop {
                        position: 0.0,
                        color: [0, 0, 0],
                    },
                    RampStop {
                        position: 0.5,
                        color: [255, 0, 0],
                    },
                    RampStop {
                        position: 1.0,
                        color: [255, 255, 255],
                    },
                ],
            },
        ],
        outputs: [
            (MaterialChannel::BaseColor, NODE(5)),
            (MaterialChannel::Roughness, NODE(4)),
        ]
        .into(),
    }
}

#[test]
fn tc_0364_graph_nodes_evaluate_per_texel_and_validate_their_wiring() {
    let graph = uv_graph();
    assert!(graph.is_valid());
    let uv = Vec2::new(0.25, 0.75);
    assert!(close(
        evaluate(&graph, NODE(0), uv),
        Vec3::new(0.25, 0.75, 0.0)
    ));
    assert!(close(evaluate(&graph, NODE(1), uv), Vec3::splat(0.25)));
    assert!(close(
        evaluate(&graph, NODE(2), uv),
        Vec3::new(0.625, 0.125, 0.125)
    ));
    assert!(close(evaluate(&graph, NODE(3), uv), Vec3::splat(0.0625)));
    assert!(close(evaluate(&graph, NODE(4), uv), Vec3::splat(0.75)));
    assert!(close(
        evaluate(&graph, NODE(5), uv),
        Vec3::new(0.5, 0.0, 0.0)
    ));
    assert!(close(
        evaluate(&graph, NODE(5), Vec2::new(0.75, 0.0)),
        Vec3::new(1.0, 0.5, 0.5)
    ));
    assert!(close(
        evaluate(&graph, GraphInput::gray(0.3), uv),
        Vec3::splat(0.3)
    ));
    // Division by zero gives 0 rather than infinity.
    let divide = MaterialGraph {
        nodes: vec![GraphNode::Math {
            op: MathOp::Divide,
            a: GraphInput::gray(1.0),
            b: GraphInput::gray(0.0),
        }],
        outputs: Default::default(),
    };
    assert_eq!(evaluate(&divide, NODE(0), uv), Vec3::ZERO);

    // Inputs read earlier nodes only; some nodes and outputs are refused.
    let refused = |edit: &dyn Fn(&mut MaterialGraph)| {
        let mut graph = uv_graph();
        edit(&mut graph);
        !graph.is_valid()
    };
    assert!(refused(&|graph| {
        graph.nodes[1] = GraphNode::Split {
            input: NODE(1),
            component: SourceComponent::Red,
        };
    }));
    assert!(refused(&|graph| {
        graph.outputs.insert(MaterialChannel::Metallic, NODE(6));
    }));
    assert!(refused(&|graph| {
        graph.outputs.insert(MaterialChannel::Id, NODE(0));
    }));
    assert!(refused(&|graph| {
        graph.nodes.push(GraphNode::Adjust {
            input: NODE(0),
            filter: LayerFilter::Blur { radius: 2.0 },
        });
    }));
    assert!(refused(&|graph| {
        graph
            .nodes
            .push(GraphNode::Generator(MaskGenerator::IdColor(
                crate::IdSelection::DEFAULT,
            )));
    }));
    assert!(refused(&|graph| {
        if let GraphNode::Ramp { stops, .. } = &mut graph.nodes[5] {
            stops.swap(0, 2);
        }
    }));
    assert!(refused(&|graph| {
        graph.nodes[2] = GraphNode::Blend {
            mode: BlendMode::PassThrough,
            a: NODE(1),
            b: NODE(1),
            amount: NODE(1),
        };
    }));
    assert!(refused(&|graph| {
        graph.nodes.resize(MAX_GRAPH_NODES + 1, GraphNode::Uv);
    }));
    assert!(refused(&|graph| {
        graph.outputs.insert(
            MaterialChannel::Metallic,
            GraphInput::Value([f32::NAN, 0.0, 0.0]),
        );
    }));

    // Removing a node rewires what read it to mid grey and shifts later
    // references down.
    let mut removed = uv_graph();
    removed.remove_node(1);
    assert!(removed.is_valid());
    assert_eq!(removed.nodes.len(), 5);
    assert_eq!(removed.outputs[&MaterialChannel::BaseColor], NODE(4));
    assert_eq!(
        removed.nodes[3].inputs(),
        [GraphInput::gray(0.5)],
        "the inverter read the removed node"
    );
}

fn set(layers: Vec<MaterialLayer>) -> TextureSet {
    TextureSet {
        id: TextureSetId(1),
        name: "Graph".to_owned(),
        material_slot: "Default".to_owned(),
        resolution: TextureResolution {
            width: 33,
            height: 33,
        },
        channels: BTreeSet::from([MaterialChannel::BaseColor]),
        layers,
        height_strength: DEFAULT_HEIGHT_STRENGTH,
        export: None,
        stencils: Vec::new(),
        user_channels: Vec::new(),
    }
}

#[test]
fn tc_0365_procedural_fills_bake_their_graph_outputs() {
    let mut layer = MaterialLayer::new_fill(
        LayerId(1),
        "Procedural",
        LayerFill {
            metallic: 0.8,
            ..LayerFill::default()
        },
    );
    layer.graph = Some(uv_graph());
    let set = set(vec![layer.clone()]);
    let context = BakeContext::default();
    let color = set.bake_base_color().unwrap();
    // Pixel (24, 16) is u = 0.75: the ramp between red and white.
    assert_eq!(color.pixel(24, 16).unwrap(), [255, 128, 128, 255]);
    assert_eq!(color.pixel(0, 16).unwrap(), [0, 0, 0, 255]);
    // Scalars take the luminance: roughness is 1 - u.
    let roughness = set
        .bake_channel(MaterialChannel::Roughness, context)
        .unwrap();
    assert_eq!(roughness.pixel(8, 16).unwrap()[0], 191);
    // Channels the graph does not feed keep the fill's value.
    let metallic = set
        .bake_channel(MaterialChannel::Metallic, context)
        .unwrap();
    assert_eq!(metallic.pixel(8, 16).unwrap()[0], 204);

    // Graphs belong to fill layers and must be valid.
    let mut paint = MaterialLayer::new_paint(LayerId(2), "Paint");
    paint.graph = Some(uv_graph());
    let mut project = MaterialProject::empty();
    project.texture_sets.push(self::set(vec![paint]));
    assert!(matches!(
        project.validate(),
        Err(ProjectError::InvalidLayerSettings(LayerId(2)))
    ));
    // A bitmap node needs its resource, in the project and when baking.
    let mut bitmap = layer.clone();
    bitmap
        .graph
        .as_mut()
        .unwrap()
        .nodes
        .push(GraphNode::Bitmap {
            resource: ResourceId(7),
            projection: Default::default(),
        });
    project.texture_sets = vec![self::set(vec![bitmap.clone()])];
    assert_eq!(
        project.validate(),
        Err(ProjectError::MissingResource(ResourceId(7)))
    );
    let mut fed = bitmap;
    fed.graph
        .as_mut()
        .unwrap()
        .outputs
        .insert(MaterialChannel::BaseColor, NODE(6));
    assert_eq!(
        self::set(vec![fed]).bake_base_color(),
        Err(BakeError::MissingResource(ResourceId(7)))
    );
    // Mesh-map nodes ask for mesh maps.
    assert!(!set.requires_mesh_maps());
    let mut maps_layer = layer;
    maps_layer
        .graph
        .as_mut()
        .unwrap()
        .nodes
        .push(GraphNode::MeshMap(crate::MeshMapKind::Curvature));
    assert!(self::set(vec![maps_layer]).requires_mesh_maps());
}

#[test]
fn tc_0366_procedural_materials_apply_capture_and_drive_instances() {
    let library = MaterialResource::built_in();
    let procedural: Vec<&MaterialResource> = library
        .iter()
        .filter(|material| material.is_procedural())
        .collect();
    assert_eq!(procedural.len(), 2);
    let rust = procedural
        .iter()
        .find(|material| material.name == "Rust (procedural)")
        .unwrap();
    for material in &procedural {
        material.validate().unwrap();
    }

    // Applied to a layer, the graph comes along and bakes with mesh maps:
    // rust in some places, bare steel in others.
    let mut layer = MaterialLayer::new_paint(LayerId(1), "Rusty");
    rust.apply_to(&mut layer);
    assert_eq!(layer.graph, rust.graph);
    let set = set(vec![layer.clone()]);
    assert!(set.requires_mesh_maps());
    assert_eq!(set.bake_base_color(), Err(BakeError::MissingMeshMaps));
    let mesh = PrimitivePreset::Sphere.build_mesh();
    let maps = bake_mesh_maps(
        &mesh,
        &MeshMapSettings {
            resolution: 33,
            ao_samples: 4,
            ..MeshMapSettings::default()
        },
        None,
    )
    .unwrap();
    let context = BakeContext {
        mesh: Some(&mesh),
        mesh_maps: Some(&maps),
        resources: None,
    };
    let color = set
        .bake_channel(MaterialChannel::BaseColor, context)
        .unwrap();
    let colors: BTreeSet<[u8; 4]> = color
        .rgba8
        .chunks(4)
        .map(|texel| [texel[0], texel[1], texel[2], texel[3]])
        .collect();
    assert!(colors.contains(&[150, 152, 156, 255]), "bare steel");
    assert!(colors.iter().any(|texel| texel[0] > texel[2] + 40), "rust");

    // Saving a procedural layer as a material keeps its graph.
    let captured = MaterialResource::from_layer(&layer, "My rust").unwrap();
    assert_eq!(captured.graph, rust.graph);

    // An instance follows the master's graph, except for channels it
    // overrides.
    let mut master = captured;
    master.id = crate::MaterialId(4);
    let link = MaterialLink::new(master.id);
    let resolved = link.resolve(&master, &layer.settings());
    assert_eq!(resolved.graph, master.graph);
    let mut overriding = link.clone();
    overriding.overrides.insert(crate::InstanceField::BaseColor);
    let resolved = overriding.resolve(&master, &layer.settings());
    let outputs = &resolved.graph.unwrap().outputs;
    assert!(!outputs.contains_key(&MaterialChannel::BaseColor));
    assert!(outputs.contains_key(&MaterialChannel::Roughness));

    // Invalid graphs make an invalid material.
    let mut broken = master;
    broken
        .graph
        .as_mut()
        .unwrap()
        .outputs
        .insert(MaterialChannel::Height, NODE(40));
    assert!(broken.validate().is_err());
}
