//! Material interchange through documented open formats: OpenPBR surfaces in
//! MaterialX documents, and glTF 2.0 metallic-roughness materials.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use quick_xml::events::Event;
use serde_json::json;
use substancecraft_core::{
    ChannelSlot, ColorSpace, ExportOutput, ExportPreset, LayerFill, MaterialChannel, MeshGeometry,
    OutputLayout, ResourceId, SourceComponent, vertex_normals,
};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum InterchangeError {
    #[error("interchange file I/O failed: {0}")]
    Io(String),
    #[error("interchange document is invalid: {0}")]
    Invalid(String),
}

fn io(error: std::io::Error) -> InterchangeError {
    InterchangeError::Io(error.to_string())
}

// ---------------------------------------------------------------------------
// OpenPBR / MaterialX

/// OpenPBR surface inputs SubstanceCraft writes, with the channel each one
/// comes from, the MaterialX type, and the texture colour space.
pub const OPENPBR_INPUTS: [(&str, MaterialChannel, &str, &str); 6] = [
    (
        "base_color",
        MaterialChannel::BaseColor,
        "color3",
        "srgb_texture",
    ),
    (
        "specular_roughness",
        MaterialChannel::Roughness,
        "float",
        "raw",
    ),
    ("base_metalness", MaterialChannel::Metallic, "float", "raw"),
    ("geometry_normal", MaterialChannel::Normal, "vector3", "raw"),
    (
        "emission_color",
        MaterialChannel::Emissive,
        "color3",
        "srgb_texture",
    ),
    ("geometry_opacity", MaterialChannel::Opacity, "float", "raw"),
];

fn openpbr_file_stem(channel: MaterialChannel) -> &'static str {
    match channel {
        MaterialChannel::BaseColor => "$textureSet_base_color",
        MaterialChannel::Roughness => "$textureSet_specular_roughness",
        MaterialChannel::Metallic => "$textureSet_base_metalness",
        MaterialChannel::Normal => "$textureSet_geometry_normal",
        MaterialChannel::Emissive => "$textureSet_emission_color",
        _ => "$textureSet_geometry_opacity",
    }
}

/// The export preset whose files an OpenPBR document references: one
/// texture per OpenPBR input, named after the input.
pub fn openpbr_preset() -> ExportPreset {
    let rgb = |channel| {
        [
            ChannelSlot::channel(channel, SourceComponent::Red),
            ChannelSlot::channel(channel, SourceComponent::Green),
            ChannelSlot::channel(channel, SourceComponent::Blue),
            ChannelSlot::ONE,
        ]
    };
    ExportPreset {
        name: "OpenPBR (MaterialX)".to_owned(),
        max_resolution: None,
        outputs: OPENPBR_INPUTS
            .iter()
            .map(|(_, channel, kind, _)| {
                let layout = if *kind == "float" {
                    OutputLayout::Gray
                } else {
                    OutputLayout::Rgb
                };
                ExportOutput::new(openpbr_file_stem(*channel), layout, rgb(*channel))
            })
            .collect(),
    }
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Writes a MaterialX 1.39 document describing `material_name` as an
/// OpenPBR surface whose inputs read the given texture files (relative to
/// the document). Inputs without a file are left at OpenPBR defaults.
pub fn write_openpbr_materialx(
    path: &Path,
    material_name: &str,
    files: &BTreeMap<String, String>,
) -> Result<(), InterchangeError> {
    let name: String = material_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect();
    let mut graph = String::new();
    let mut inputs = String::new();
    for (input, _, kind, color_space) in OPENPBR_INPUTS {
        let Some(file) = files.get(input) else {
            continue;
        };
        graph.push_str(&format!(
            "    <image name=\"{input}_image\" type=\"{image_type}\">\n      <input name=\"file\" type=\"filename\" value=\"{file}\" colorspace=\"{color_space}\" />\n    </image>\n",
            image_type = if kind == "vector3" { "vector3" } else { kind },
            file = escape(file),
        ));
        let source = if input == "geometry_normal" {
            graph.push_str(&format!(
                "    <normalmap name=\"{input}_map\" type=\"vector3\">\n      <input name=\"in\" type=\"vector3\" nodename=\"{input}_image\" />\n    </normalmap>\n"
            ));
            format!("{input}_map")
        } else {
            format!("{input}_image")
        };
        graph.push_str(&format!(
            "    <output name=\"{input}_out\" type=\"{kind}\" nodename=\"{source}\" />\n"
        ));
        inputs.push_str(&format!(
            "    <input name=\"{input}\" type=\"{kind}\" nodegraph=\"NG_{name}\" output=\"{input}_out\" />\n"
        ));
    }
    let document = format!(
        "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\" colorspace=\"lin_rec709\">\n  <nodegraph name=\"NG_{name}\">\n{graph}  </nodegraph>\n  <open_pbr_surface name=\"SR_{name}\" type=\"surfaceshader\">\n{inputs}  </open_pbr_surface>\n  <surfacematerial name=\"{name}\" type=\"material\">\n    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"SR_{name}\" />\n  </surfacematerial>\n</materialx>\n"
    );
    std::fs::write(path, document).map_err(io)
}

/// An OpenPBR material read back from MaterialX: its name and, per OpenPBR
/// input, the texture file feeding it or its constant value.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenPbrMaterial {
    pub name: String,
    /// Texture file (relative to the document) per input.
    pub textures: BTreeMap<String, String>,
    /// Constant value per unconnected input: three components for colours,
    /// one for floats, in the document's colour space.
    pub values: BTreeMap<String, Vec<f32>>,
    /// The `colorspace` of each texture input's file, where the document
    /// gives one.
    pub texture_color_spaces: BTreeMap<String, String>,
    /// The document's working colour space (`lin_rec709` when unstated).
    pub color_space: String,
}

/// Where a shader input reads from.
enum Binding {
    /// A nodegraph output.
    Output(String),
    /// A node in the document.
    Node(String),
}

/// Reads the first `surfacematerial` with an `open_pbr_surface` shader,
/// following each shader input through its nodegraph output or node (and an
/// optional `normalmap`) to the image file, and keeping the constant value
/// of inputs that have one.
pub fn read_openpbr_materialx(path: &Path) -> Result<OpenPbrMaterial, InterchangeError> {
    let xml = std::fs::read_to_string(path).map_err(io)?;
    let mut reader = quick_xml::Reader::from_str(&xml);
    let invalid = |error: &dyn std::fmt::Display| InterchangeError::Invalid(error.to_string());

    // node name -> file and colour space (images) or upstream node (normal
    // maps); output name -> node; shader input -> binding; material ->
    // shader.
    let mut image_files = BTreeMap::new();
    let mut image_spaces = BTreeMap::new();
    let mut upstream = BTreeMap::new();
    let mut outputs = BTreeMap::new();
    let mut shader_inputs = BTreeMap::new();
    let mut values = BTreeMap::new();
    let mut material = None;
    let mut color_space = "lin_rec709".to_owned();
    let mut stack: Vec<(String, String)> = Vec::new();
    loop {
        let event = reader.read_event().map_err(|error| invalid(&error))?;
        let (element, empty) = match &event {
            Event::Start(element) => (element, false),
            Event::Empty(element) => (element, true),
            Event::End(_) => {
                stack.pop();
                continue;
            }
            Event::Eof => break,
            _ => continue,
        };
        let tag = String::from_utf8_lossy(element.name().as_ref()).into_owned();
        let mut attributes = BTreeMap::new();
        for attribute in element.attributes() {
            let attribute = attribute.map_err(|error| invalid(&error))?;
            let value = attribute
                .normalized_value(quick_xml::XmlVersion::Implicit1_0)
                .map_err(|error| invalid(&error))?
                .into_owned();
            attributes.insert(
                String::from_utf8_lossy(attribute.key.as_ref()).into_owned(),
                value,
            );
        }
        let name = attributes.get("name").cloned().unwrap_or_default();
        let parent = stack.last().cloned();
        match (tag.as_str(), parent) {
            ("materialx", _) => {
                if let Some(space) = attributes.get("colorspace") {
                    color_space.clone_from(space);
                }
            }
            ("output", _) => {
                if let Some(node) = attributes.get("nodename") {
                    outputs.insert(name.clone(), node.clone());
                }
            }
            ("input", Some((parent_tag, parent_name))) => match parent_tag.as_str() {
                "image" if name == "file" => {
                    if let Some(value) = attributes.get("value") {
                        image_files.insert(parent_name.clone(), value.clone());
                    }
                    if let Some(space) = attributes.get("colorspace") {
                        image_spaces.insert(parent_name, space.clone());
                    }
                }
                "normalmap" if name == "in" => {
                    if let Some(node) = attributes.get("nodename") {
                        upstream.insert(parent_name, node.clone());
                    }
                }
                "open_pbr_surface" => {
                    if let Some(output) = attributes.get("output") {
                        shader_inputs.insert(name.clone(), Binding::Output(output.clone()));
                    } else if let Some(node) = attributes.get("nodename") {
                        shader_inputs.insert(name.clone(), Binding::Node(node.clone()));
                    } else if let Some(value) = attributes.get("value") {
                        let components: Vec<f32> = value
                            .split(',')
                            .map(|part| part.trim().parse::<f32>())
                            .collect::<Result<_, _>>()
                            .map_err(|_| {
                                InterchangeError::Invalid(format!(
                                    "input `{name}` has a malformed value `{value}`"
                                ))
                            })?;
                        values.insert(name.clone(), components);
                    }
                }
                "surfacematerial" if name == "surfaceshader" && material.is_none() => {
                    material = Some(parent_name);
                }
                _ => {}
            },
            _ => {}
        }
        if !empty {
            stack.push((tag, name));
        }
    }
    let name =
        material.ok_or_else(|| InterchangeError::Invalid("no surfacematerial".to_owned()))?;
    let mut textures = BTreeMap::new();
    let mut texture_color_spaces = BTreeMap::new();
    for (input, binding) in shader_inputs {
        let node = match binding {
            Binding::Output(output) => match outputs.get(&output) {
                Some(node) => node.clone(),
                None => continue,
            },
            Binding::Node(node) => node,
        };
        let node = upstream.get(&node).cloned().unwrap_or(node);
        if let Some(file) = image_files.get(&node) {
            textures.insert(input.clone(), file.clone());
            if let Some(space) = image_spaces.get(&node) {
                texture_color_spaces.insert(input, space.clone());
            }
        }
    }
    Ok(OpenPbrMaterial {
        name,
        textures,
        values,
        texture_color_spaces,
        color_space,
    })
}

/// The colour space a MaterialX `colorspace` name stands for, or `None`
/// for names SubstanceCraft cannot convert.
pub fn materialx_color_space(name: &str) -> Option<ColorSpace> {
    match name.trim().to_ascii_lowercase().as_str() {
        "srgb_texture" | "srgb" | "srgb_rec709" => Some(ColorSpace::Srgb),
        "lin_rec709" | "lin_srgb" | "linear" => Some(ColorSpace::LinearSrgb),
        "acescg" | "lin_ap1" => Some(ColorSpace::AcesCg),
        "raw" | "none" | "data" => Some(ColorSpace::Raw),
        _ => None,
    }
}

/// The colour space the texture feeding OpenPBR `input` is read in: the
/// one the document names, else sRGB for colours and raw data otherwise.
pub fn openpbr_texture_color_space(material: &OpenPbrMaterial, input: &str) -> ColorSpace {
    material
        .texture_color_spaces
        .get(input)
        .and_then(|name| materialx_color_space(name))
        .unwrap_or(match input {
            "base_color" | "emission_color" => ColorSpace::Srgb,
            _ => ColorSpace::Raw,
        })
}

/// The fill layer settings an OpenPBR material describes, and the channels
/// it writes. Texture inputs bind the resource `resources` maps them to;
/// constant colours convert from the document's colour space to sRGB;
/// unset inputs take OpenPBR's defaults (base colour 0.8, roughness 0.3,
/// metalness 0). Emission is written only when given a texture or a colour
/// other than black; normals and opacity only when given.
pub fn openpbr_fill(
    material: &OpenPbrMaterial,
    resources: &BTreeMap<String, ResourceId>,
) -> (LayerFill, BTreeSet<MaterialChannel>) {
    let space = materialx_color_space(&material.color_space).unwrap_or(ColorSpace::LinearSrgb);
    let scalar = |input: &str, default: f32| {
        material
            .values
            .get(input)
            .and_then(|value| value.first().copied())
            .filter(|value| value.is_finite())
            .map_or(default, |value| value.clamp(0.0, 1.0))
    };
    let color = |input: &str, default: [f32; 3]| {
        let rgb = material
            .values
            .get(input)
            .filter(|value| value.len() == 3 && value.iter().all(|c| c.is_finite()))
            .map_or(glam::Vec3::from(default), |value| {
                glam::Vec3::new(value[0], value[1], value[2])
            });
        space
            .to_srgb(rgb.max(glam::Vec3::ZERO))
            .to_array()
            .map(|component| (component.clamp(0.0, 1.0) * 255.0).round() as u8)
    };
    let mut fill = LayerFill::default();
    let [r, g, b] = color("base_color", [0.8, 0.8, 0.8]);
    fill.color = [r, g, b, 255];
    fill.roughness = scalar("specular_roughness", 0.3);
    fill.metallic = scalar("base_metalness", 0.0);
    fill.emissive = color("emission_color", [0.0, 0.0, 0.0]);
    fill.material_opacity = scalar("geometry_opacity", 1.0);
    let mut channels = BTreeSet::from([
        MaterialChannel::BaseColor,
        MaterialChannel::Roughness,
        MaterialChannel::Metallic,
    ]);
    if resources.contains_key("emission_color") || fill.emissive != [0, 0, 0] {
        channels.insert(MaterialChannel::Emissive);
    }
    if resources.contains_key("geometry_normal") {
        channels.insert(MaterialChannel::Normal);
    }
    if resources.contains_key("geometry_opacity")
        || material.values.contains_key("geometry_opacity")
    {
        channels.insert(MaterialChannel::Opacity);
    }
    for (input, channel, _, _) in OPENPBR_INPUTS {
        if let (Some(resource), Some(slot)) =
            (resources.get(input), fill.textures.slot_mut(channel))
        {
            *slot = Some(*resource);
        }
    }
    (fill, channels)
}

// ---------------------------------------------------------------------------
// glTF 2.0

/// Texture files (relative to the glTF file) of one material slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GltfMaterialFiles {
    pub name: String,
    pub base_color: Option<String>,
    /// Packed occlusion (R), roughness (G), metallic (B).
    pub orm: Option<String>,
    pub normal: Option<String>,
    pub emissive: Option<String>,
    /// Base Color alpha holds opacity that should blend.
    pub blend: bool,
}

/// Writes `mesh` as a glTF 2.0 file (`.gltf` plus a `.bin` buffer beside
/// it) with one primitive and one metallic-roughness material per material
/// slot. `materials[i]` describes slot `i`. Returns the files written.
pub fn write_gltf(
    path: &Path,
    mesh: &MeshGeometry,
    smoothing_angle_degrees: f32,
    materials: &[GltfMaterialFiles],
) -> Result<Vec<PathBuf>, InterchangeError> {
    if mesh.vertices.is_empty() || mesh.triangles.is_empty() {
        return Err(InterchangeError::Invalid("mesh is empty".to_owned()));
    }
    let normals = vertex_normals(mesh, smoothing_angle_degrees);
    let mut buffer: Vec<u8> = Vec::new();
    let push_f32 = |buffer: &mut Vec<u8>, value: f32| buffer.extend(value.to_le_bytes());

    let (mut min, mut max) = (mesh.vertices[0].position, mesh.vertices[0].position);
    for vertex in &mesh.vertices {
        min = min.min(vertex.position);
        max = max.max(vertex.position);
        for value in vertex.position.to_array() {
            push_f32(&mut buffer, value);
        }
    }
    let positions_len = buffer.len();
    for normal in &normals {
        for value in normal.to_array() {
            push_f32(&mut buffer, value);
        }
    }
    let normals_len = buffer.len() - positions_len;
    for vertex in &mesh.vertices {
        // glTF puts the UV origin at the top left.
        push_f32(&mut buffer, vertex.uv.x);
        push_f32(&mut buffer, 1.0 - vertex.uv.y);
    }
    let uvs_len = buffer.len() - positions_len - normals_len;

    let slot_count = mesh.material_slots.len().max(1);
    let mut views = vec![
        json!({"buffer": 0, "byteOffset": 0, "byteLength": positions_len, "target": 34962}),
        json!({"buffer": 0, "byteOffset": positions_len, "byteLength": normals_len, "target": 34962}),
        json!({"buffer": 0, "byteOffset": positions_len + normals_len, "byteLength": uvs_len, "target": 34962}),
    ];
    let count = mesh.vertices.len();
    let mut accessors = vec![
        json!({"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3",
               "min": min.to_array(), "max": max.to_array()}),
        json!({"bufferView": 1, "componentType": 5126, "count": count, "type": "VEC3"}),
        json!({"bufferView": 2, "componentType": 5126, "count": count, "type": "VEC2"}),
    ];
    let mut primitives = Vec::new();
    for slot in 0..slot_count {
        let indices: Vec<u32> = mesh
            .triangles
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                mesh.triangle_materials.get(*index).copied().unwrap_or(0) as usize == slot
            })
            .flat_map(|(_, triangle)| *triangle)
            .collect();
        if indices.is_empty() {
            continue;
        }
        let offset = buffer.len();
        for index in &indices {
            buffer.extend(index.to_le_bytes());
        }
        views.push(json!({"buffer": 0, "byteOffset": offset, "byteLength": indices.len() * 4, "target": 34963}));
        accessors.push(json!({"bufferView": views.len() - 1, "componentType": 5125,
                              "count": indices.len(), "type": "SCALAR"}));
        primitives.push(json!({
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": accessors.len() - 1,
            "material": slot,
        }));
    }

    let mut images: Vec<serde_json::Value> = Vec::new();
    let mut textures: Vec<serde_json::Value> = Vec::new();
    let mut texture_of = |uri: &Option<String>| -> Option<usize> {
        let uri = uri.as_ref()?;
        if let Some(existing) = images.iter().position(|image| image["uri"] == json!(uri)) {
            return textures
                .iter()
                .position(|texture| texture["source"] == json!(existing));
        }
        images.push(json!({"uri": uri}));
        textures.push(json!({"source": images.len() - 1, "sampler": 0}));
        Some(textures.len() - 1)
    };
    let mut gltf_materials = Vec::new();
    for slot in 0..slot_count {
        let files = materials.get(slot).cloned().unwrap_or_default();
        let name = mesh
            .material_slots
            .get(slot)
            .cloned()
            .unwrap_or_else(|| files.name.clone());
        let mut pbr = json!({"metallicFactor": 1.0, "roughnessFactor": 1.0});
        if let Some(index) = texture_of(&files.base_color) {
            pbr["baseColorTexture"] = json!({"index": index});
        }
        let orm = texture_of(&files.orm);
        if let Some(index) = orm {
            pbr["metallicRoughnessTexture"] = json!({"index": index});
        }
        let mut material = json!({"name": name, "pbrMetallicRoughness": pbr});
        if let Some(index) = orm {
            material["occlusionTexture"] = json!({"index": index});
        }
        if let Some(index) = texture_of(&files.normal) {
            material["normalTexture"] = json!({"index": index});
        }
        if let Some(index) = texture_of(&files.emissive) {
            material["emissiveTexture"] = json!({"index": index});
            material["emissiveFactor"] = json!([1.0, 1.0, 1.0]);
        }
        if files.blend {
            material["alphaMode"] = json!("BLEND");
        }
        gltf_materials.push(material);
    }

    let bin_path = path.with_extension("bin");
    let bin_name = bin_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| InterchangeError::Invalid("glTF path has no file name".to_owned()))?;
    let mut document = json!({
        "asset": {"version": "2.0", "generator": "SubstanceCraft"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Mesh"}],
        "meshes": [{"primitives": primitives}],
        "materials": gltf_materials,
        "buffers": [{"uri": bin_name, "byteLength": buffer.len()}],
        "bufferViews": views,
        "accessors": accessors,
    });
    if !images.is_empty() {
        document["images"] = json!(images);
        document["textures"] = json!(textures);
        document["samplers"] =
            json!([{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]);
    }
    let text = serde_json::to_string_pretty(&document)
        .map_err(|error| InterchangeError::Invalid(error.to_string()))?;
    std::fs::write(path, text).map_err(io)?;
    std::fs::write(&bin_path, buffer).map_err(io)?;
    Ok(vec![path.to_path_buf(), bin_path])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::import_gltf_file;
    use substancecraft_core::PrimitivePreset;

    #[test]
    fn tc_0190_openpbr_materialx_round_trips_texture_bindings() {
        let preset = openpbr_preset();
        preset.validate().unwrap();
        assert_eq!(preset.outputs.len(), OPENPBR_INPUTS.len());

        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("hull.mtlx");
        let files: BTreeMap<String, String> = [
            ("base_color", "Hull_base_color.png"),
            ("specular_roughness", "Hull_specular_roughness.png"),
            ("geometry_normal", "Hull_geometry_normal.png"),
        ]
        .into_iter()
        .map(|(input, file)| (input.to_owned(), file.to_owned()))
        .collect();
        write_openpbr_materialx(&path, "Hull Plate", &files).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("<open_pbr_surface name=\"SR_Hull_Plate\""));
        assert!(text.contains("colorspace=\"srgb_texture\""));
        assert!(text.contains("<normalmap"));
        assert!(
            !text.contains("base_metalness"),
            "unbound inputs keep defaults"
        );

        let material = read_openpbr_materialx(&path).unwrap();
        assert_eq!(material.name, "Hull_Plate");
        assert_eq!(material.textures, files);

        std::fs::write(&path, "<materialx></materialx>").unwrap();
        assert!(matches!(
            read_openpbr_materialx(&path),
            Err(InterchangeError::Invalid(_))
        ));
    }

    #[test]
    fn tc_0191_gltf_export_reimports_with_materials_and_textures() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("crate.gltf");
        let mut mesh = PrimitivePreset::Cube.build_mesh();
        // Two slots: the first four triangles use slot 1.
        mesh.material_slots = vec!["Body".to_owned(), "Trim".to_owned()];
        mesh.triangle_materials = (0..mesh.triangles.len() as u32)
            .map(|index| u32::from(index < 4))
            .collect();
        let materials = [
            GltfMaterialFiles {
                name: "Body".to_owned(),
                base_color: Some("Body_BaseColor.png".to_owned()),
                orm: Some("Body_ORM.png".to_owned()),
                normal: Some("Body_Normal.png".to_owned()),
                emissive: None,
                blend: false,
            },
            GltfMaterialFiles {
                name: "Trim".to_owned(),
                base_color: Some("Trim_BaseColor.png".to_owned()),
                emissive: Some("Trim_Emissive.png".to_owned()),
                blend: true,
                ..GltfMaterialFiles::default()
            },
        ];
        let written = write_gltf(&path, &mesh, 60.0, &materials).unwrap();
        assert_eq!(written.len(), 2);

        let imported = import_gltf_file(&path).unwrap();
        assert_eq!(imported.geometry.triangles.len(), mesh.triangles.len());
        assert_eq!(imported.geometry.material_slots, ["Body", "Trim"]);
        let trim = imported
            .geometry
            .triangle_materials
            .iter()
            .filter(|slot| **slot == 1)
            .count();
        assert_eq!(trim, 4);

        let document: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let body = &document["materials"][0];
        let orm = &body["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"];
        assert_eq!(orm, &body["occlusionTexture"]["index"], "ORM is shared");
        assert_eq!(document["materials"][1]["alphaMode"], "BLEND");
        assert_eq!(
            document["materials"][1]["emissiveFactor"],
            json!([1.0, 1.0, 1.0])
        );
        assert_eq!(document["images"].as_array().unwrap().len(), 5);
        assert!(write_gltf(&path, &MeshGeometry::default(), 60.0, &[]).is_err());
    }

    #[test]
    fn tc_0331_openpbr_materials_read_constants_and_map_onto_fill_layers() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("gilded.mtlx");
        let document = |space: &str, base: &str| {
            format!(
                r#"<?xml version="1.0"?>
<materialx version="1.39" colorspace="{space}">
  <image name="rough_img" type="float">
    <input name="file" type="filename" value="rough.png" colorspace="raw" />
  </image>
  <open_pbr_surface name="SR" type="surfaceshader">
    <input name="base_color" type="color3" value="{base}" />
    <input name="base_metalness" type="float" value="1" />
    <input name="specular_roughness" type="float" nodename="rough_img" />
    <input name="emission_color" type="color3" value="0, 0, 0" />
    <input name="geometry_opacity" type="float" value="0.5" />
  </open_pbr_surface>
  <surfacematerial name="Gilded" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR" />
  </surfacematerial>
</materialx>
"#
            )
        };

        // Constants are kept, and inputs bound straight to a node resolve.
        std::fs::write(&path, document("lin_rec709", "0.2158605, 0.0, 1.0")).unwrap();
        let material = read_openpbr_materialx(&path).unwrap();
        assert_eq!(material.name, "Gilded");
        assert_eq!(material.color_space, "lin_rec709");
        assert_eq!(
            material.textures,
            [("specular_roughness".to_owned(), "rough.png".to_owned())].into()
        );
        assert_eq!(material.values["base_metalness"], [1.0]);
        assert_eq!(material.values["base_color"], [0.215_860_5, 0.0, 1.0]);
        assert_eq!(
            openpbr_texture_color_space(&material, "specular_roughness"),
            ColorSpace::Raw
        );
        assert_eq!(
            openpbr_texture_color_space(&material, "base_color"),
            ColorSpace::Srgb,
            "colour textures default to sRGB"
        );

        // The fill: linear colour encoded as sRGB, the texture bound, black
        // emission left out, the given opacity written.
        let resources = [("specular_roughness".to_owned(), ResourceId(7))].into();
        let (fill, channels) = openpbr_fill(&material, &resources);
        assert_eq!(fill.color, [128, 0, 255, 255]);
        assert_eq!(fill.metallic, 1.0);
        assert_eq!(fill.material_opacity, 0.5);
        assert_eq!(fill.textures.roughness, Some(ResourceId(7)));
        assert_eq!(fill.textures.base_color, None);
        assert_eq!(
            channels,
            [
                MaterialChannel::BaseColor,
                MaterialChannel::Roughness,
                MaterialChannel::Metallic,
                MaterialChannel::Opacity,
            ]
            .into()
        );

        // ACEScg documents convert their colours; unknown spaces read as
        // linear Rec.709.
        std::fs::write(&path, document("acescg", "1, 1, 1")).unwrap();
        let (fill, _) = openpbr_fill(&read_openpbr_materialx(&path).unwrap(), &BTreeMap::new());
        assert!(
            fill.color[..3].iter().all(|c| *c >= 254),
            "{:?}",
            fill.color
        );
        assert_eq!(materialx_color_space("lin_ap1"), Some(ColorSpace::AcesCg));
        assert_eq!(materialx_color_space("g22_ap1"), None);

        // Defaults for a bare surface, and every round-tripped texture binds.
        let written: BTreeMap<String, String> = OPENPBR_INPUTS
            .iter()
            .map(|(input, _, _, _)| ((*input).to_owned(), format!("{input}.png")))
            .collect();
        write_openpbr_materialx(&path, "Hull", &written).unwrap();
        let material = read_openpbr_materialx(&path).unwrap();
        assert_eq!(material.textures, written);
        assert!(material.values.is_empty());
        let resources: BTreeMap<String, ResourceId> = OPENPBR_INPUTS
            .iter()
            .enumerate()
            .map(|(index, (input, _, _, _))| ((*input).to_owned(), ResourceId(index as u64 + 1)))
            .collect();
        let (fill, channels) = openpbr_fill(&material, &resources);
        assert_eq!(fill.color, [231, 231, 231, 255], "OpenPBR's 0.8 grey");
        assert_eq!(fill.roughness, 0.3);
        for (index, (_, channel, _, _)) in OPENPBR_INPUTS.iter().enumerate() {
            assert_eq!(
                fill.textures.for_channel(*channel),
                Some(ResourceId(index as u64 + 1))
            );
            assert!(channels.contains(channel), "{channel:?}");
        }

        std::fs::write(&path, document("lin_rec709", "red, 0, 0")).unwrap();
        assert!(matches!(
            read_openpbr_materialx(&path),
            Err(InterchangeError::Invalid(reason)) if reason.contains("base_color")
        ));
    }
}

// ---------------------------------------------------------------------------
// glTF 2.0 material import

/// A glTF 2.0 metallic-roughness material: its factors (colours in linear
/// light) and the external texture files (relative to the glTF) feeding it.
#[derive(Clone, Debug, PartialEq)]
pub struct GltfMaterial {
    pub name: String,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub base_color_texture: Option<String>,
    /// Roughness in G, metallic in B.
    pub metallic_roughness_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub emissive_texture: Option<String>,
    /// `alphaMode` is BLEND or MASK, so base colour alpha is opacity.
    pub uses_alpha: bool,
    /// Textures embedded in the file (data URIs or buffer views), which
    /// are not imported.
    pub embedded_textures: usize,
}

/// The JSON of a `.gltf` file, or of the JSON chunk of a `.glb`.
fn gltf_json(path: &Path) -> Result<serde_json::Value, InterchangeError> {
    let bytes = std::fs::read(path).map_err(io)?;
    let json = if bytes.starts_with(b"glTF") {
        if bytes.len() < 20 {
            return Err(InterchangeError::Invalid("truncated GLB".to_owned()));
        }
        let length = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        if &bytes[16..20] != b"JSON" || bytes.len() < 20 + length {
            return Err(InterchangeError::Invalid(
                "GLB has no JSON chunk".to_owned(),
            ));
        }
        bytes[20..20 + length].to_vec()
    } else {
        bytes
    };
    serde_json::from_slice(&json).map_err(|error| InterchangeError::Invalid(error.to_string()))
}

/// Reads every material of a glTF 2.0 file (`.gltf` or `.glb`).
pub fn read_gltf_materials(path: &Path) -> Result<Vec<GltfMaterial>, InterchangeError> {
    let document = gltf_json(path)?;
    let empty = Vec::new();
    let array = |key: &str| {
        document
            .get(key)
            .and_then(|value| value.as_array())
            .unwrap_or(&empty)
    };
    let (textures, images) = (array("textures"), array("images"));
    let numbers = |value: Option<&serde_json::Value>, default: &[f32]| -> Vec<f32> {
        value
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .map(|item| item.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .filter(|items: &Vec<f32>| {
                items.len() == default.len() && items.iter().all(|value| value.is_finite())
            })
            .unwrap_or_else(|| default.to_vec())
    };
    let scalar = |value: Option<&serde_json::Value>, default: f32| {
        value
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .unwrap_or(default)
    };
    let mut materials = Vec::new();
    for (index, material) in array("materials").iter().enumerate() {
        let mut embedded = 0;
        let mut texture = |info: Option<&serde_json::Value>| -> Option<String> {
            let texture_index = info?.get("index")?.as_u64()? as usize;
            let image_index = textures.get(texture_index)?.get("source")?.as_u64()? as usize;
            let image = images.get(image_index)?;
            match image.get("uri").and_then(|uri| uri.as_str()) {
                Some(uri) if !uri.starts_with("data:") => Some(percent_decode(uri)),
                _ => {
                    embedded += 1;
                    None
                }
            }
        };
        let pbr = material.get("pbrMetallicRoughness");
        let pbr_field = |key: &str| pbr.and_then(|pbr| pbr.get(key));
        let base = numbers(pbr_field("baseColorFactor"), &[1.0, 1.0, 1.0, 1.0]);
        let emissive = numbers(material.get("emissiveFactor"), &[0.0, 0.0, 0.0]);
        let alpha_mode = material
            .get("alphaMode")
            .and_then(|mode| mode.as_str())
            .unwrap_or("OPAQUE");
        let base_color_texture = texture(pbr_field("baseColorTexture"));
        let metallic_roughness_texture = texture(pbr_field("metallicRoughnessTexture"));
        let normal_texture = texture(material.get("normalTexture"));
        let emissive_texture = texture(material.get("emissiveTexture"));
        materials.push(GltfMaterial {
            name: material
                .get("name")
                .and_then(|name| name.as_str())
                .map_or_else(|| format!("Material {}", index + 1), str::to_owned),
            base_color_factor: [base[0], base[1], base[2], base[3]],
            metallic_factor: scalar(pbr_field("metallicFactor"), 1.0).clamp(0.0, 1.0),
            roughness_factor: scalar(pbr_field("roughnessFactor"), 1.0).clamp(0.0, 1.0),
            emissive_factor: [emissive[0], emissive[1], emissive[2]],
            base_color_texture,
            metallic_roughness_texture,
            normal_texture,
            emissive_texture,
            uses_alpha: matches!(alpha_mode, "BLEND" | "MASK"),
            embedded_textures: embedded,
        });
    }
    if materials.is_empty() {
        return Err(InterchangeError::Invalid(
            "the file has no materials".to_owned(),
        ));
    }
    Ok(materials)
}

/// Decodes `%XX` escapes in a glTF URI.
fn percent_decode(uri: &str) -> String {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let Ok(value) = u8::from_str_radix(&uri[index + 1..index + 3], 16)
        {
            out.push(value);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Keys of the resources [`gltf_fill`] binds, with the channel each feeds.
pub const GLTF_FILL_INPUTS: [(&str, MaterialChannel); 5] = [
    ("base_color", MaterialChannel::BaseColor),
    ("roughness", MaterialChannel::Roughness),
    ("metallic", MaterialChannel::Metallic),
    ("normal", MaterialChannel::Normal),
    ("emissive", MaterialChannel::Emissive),
];

/// The fill layer a glTF material describes, and the channels it writes.
/// Factors become fill values (colours converted from linear light to
/// sRGB); a texture replaces its factor. Emission is written when given a
/// texture or a non-black factor, normals when given a texture, and opacity
/// when the material blends or masks.
pub fn gltf_fill(
    material: &GltfMaterial,
    resources: &BTreeMap<String, ResourceId>,
) -> (LayerFill, BTreeSet<MaterialChannel>) {
    let to_srgb8 = |rgb: [f32; 3]| {
        ColorSpace::LinearSrgb
            .to_srgb(glam::Vec3::from(rgb).max(glam::Vec3::ZERO))
            .to_array()
            .map(|component| (component.clamp(0.0, 1.0) * 255.0).round() as u8)
    };
    let [r, g, b, a] = material.base_color_factor;
    let mut fill = LayerFill::default();
    let [cr, cg, cb] = to_srgb8([r, g, b]);
    fill.color = [cr, cg, cb, 255];
    fill.roughness = material.roughness_factor;
    fill.metallic = material.metallic_factor;
    fill.emissive = to_srgb8(material.emissive_factor);
    let mut channels = BTreeSet::from([
        MaterialChannel::BaseColor,
        MaterialChannel::Roughness,
        MaterialChannel::Metallic,
    ]);
    if resources.contains_key("emissive") || fill.emissive != [0, 0, 0] {
        channels.insert(MaterialChannel::Emissive);
    }
    if resources.contains_key("normal") {
        channels.insert(MaterialChannel::Normal);
    }
    if material.uses_alpha {
        fill.material_opacity = a.clamp(0.0, 1.0);
        channels.insert(MaterialChannel::Opacity);
    }
    for (input, channel) in GLTF_FILL_INPUTS {
        if let (Some(resource), Some(slot)) =
            (resources.get(input), fill.textures.slot_mut(channel))
        {
            *slot = Some(*resource);
        }
    }
    (fill, channels)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod gltf_material_tests {
    use super::*;

    #[test]
    fn gltf_materials_read_factors_and_external_textures() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("crate.gltf");
        let document = json!({
            "asset": {"version": "2.0"},
            "images": [{"uri": "crate%20base.png"}, {"uri": "data:image/png;base64,AAAA"}, {"uri": "mr.png"}],
            "textures": [{"source": 0}, {"source": 1}, {"source": 2}],
            "materials": [
                {"name": "Crate", "pbrMetallicRoughness": {
                    "baseColorFactor": [0.2159, 0.2159, 0.2159, 0.5],
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 2},
                    "roughnessFactor": 0.4},
                 "normalTexture": {"index": 1},
                 "emissiveFactor": [1.0, 0.0, 0.0],
                 "alphaMode": "BLEND"},
                {}
            ]
        });
        std::fs::write(&path, document.to_string()).unwrap();
        let materials = read_gltf_materials(&path).unwrap();
        assert_eq!(materials.len(), 2);
        let crate_material = &materials[0];
        assert_eq!(
            crate_material.base_color_texture.as_deref(),
            Some("crate base.png")
        );
        assert_eq!(
            crate_material.metallic_roughness_texture.as_deref(),
            Some("mr.png")
        );
        assert_eq!(crate_material.normal_texture, None);
        assert_eq!(crate_material.embedded_textures, 1);
        assert!(crate_material.uses_alpha);
        assert_eq!(materials[1].name, "Material 2");
        assert_eq!(materials[1].metallic_factor, 1.0);

        let resources = BTreeMap::from([("base_color".to_owned(), ResourceId(4))]);
        let (fill, channels) = gltf_fill(crate_material, &resources);
        assert_eq!(fill.color, [128, 128, 128, 255]);
        assert_eq!(fill.textures.base_color, Some(ResourceId(4)));
        assert!((fill.roughness - 0.4).abs() < 1e-6);
        assert_eq!(fill.emissive, [255, 0, 0]);
        assert_eq!(fill.material_opacity, 0.5);
        assert!(channels.contains(&MaterialChannel::Emissive));
        assert!(channels.contains(&MaterialChannel::Opacity));
        assert!(!channels.contains(&MaterialChannel::Normal));

        // The JSON chunk of a GLB reads the same.
        let json_bytes = document.to_string().into_bytes();
        let mut glb = b"glTF".to_vec();
        glb.extend(2u32.to_le_bytes());
        glb.extend(((20 + json_bytes.len()) as u32).to_le_bytes());
        glb.extend((json_bytes.len() as u32).to_le_bytes());
        glb.extend(b"JSON");
        glb.extend(&json_bytes);
        let glb_path = folder.path().join("crate.glb");
        std::fs::write(&glb_path, glb).unwrap();
        assert_eq!(read_gltf_materials(&glb_path).unwrap(), materials);

        std::fs::write(&path, r#"{"asset": {"version": "2.0"}}"#).unwrap();
        assert!(read_gltf_materials(&path).is_err());
    }
}
