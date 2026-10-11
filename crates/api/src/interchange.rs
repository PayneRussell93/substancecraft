//! Material interchange through documented open formats: OpenPBR surfaces in
//! MaterialX documents, and glTF 2.0 metallic-roughness materials.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use quick_xml::events::Event;
use serde_json::json;
use substancecraft_core::{
    ChannelSlot, ExportOutput, ExportPreset, MaterialChannel, MeshGeometry, OutputLayout,
    SourceComponent, vertex_normals,
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
/// input, the texture file feeding it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenPbrMaterial {
    pub name: String,
    pub textures: BTreeMap<String, String>,
}

/// Reads the first `surfacematerial` with an `open_pbr_surface` shader,
/// following each shader input through its nodegraph output (and an
/// optional `normalmap`) to the image file.
pub fn read_openpbr_materialx(path: &Path) -> Result<OpenPbrMaterial, InterchangeError> {
    let xml = std::fs::read_to_string(path).map_err(io)?;
    let mut reader = quick_xml::Reader::from_str(&xml);
    let invalid = |error: &dyn std::fmt::Display| InterchangeError::Invalid(error.to_string());

    // node name -> file (images) or upstream node (normal maps); output name
    // -> node; shader input -> output; material -> shader.
    let mut image_files = BTreeMap::new();
    let mut upstream = BTreeMap::new();
    let mut outputs = BTreeMap::new();
    let mut shader_inputs = BTreeMap::new();
    let mut material = None;
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
                .unescape_value()
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
            ("output", _) => {
                if let Some(node) = attributes.get("nodename") {
                    outputs.insert(name.clone(), node.clone());
                }
            }
            ("input", Some((parent_tag, parent_name))) => match parent_tag.as_str() {
                "image" if name == "file" => {
                    if let Some(value) = attributes.get("value") {
                        image_files.insert(parent_name, value.clone());
                    }
                }
                "normalmap" if name == "in" => {
                    if let Some(node) = attributes.get("nodename") {
                        upstream.insert(parent_name, node.clone());
                    }
                }
                "open_pbr_surface" => {
                    if let Some(output) = attributes.get("output") {
                        shader_inputs.insert(name.clone(), output.clone());
                    }
                }
                "surfacematerial" if name == "surfaceshader" => {
                    if material.is_none() {
                        material = Some(parent_name);
                    }
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
    for (input, output) in shader_inputs {
        let Some(mut node) = outputs.get(&output).cloned() else {
            continue;
        };
        if let Some(image) = upstream.get(&node) {
            node = image.clone();
        }
        if let Some(file) = image_files.get(&node) {
            textures.insert(input, file.clone());
        }
    }
    Ok(OpenPbrMaterial { name, textures })
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
}
