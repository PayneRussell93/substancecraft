//! glTF 2.0 (`.gltf` with external or embedded buffers, and binary `.glb`)
//! mesh import. The default scene is flattened into one mesh with node
//! transforms applied; each glTF material becomes a material slot. Files with
//! skins or animations also get a [`Rig`]: skinned meshes stay in their bind
//! space (glTF ignores their node transform), other meshes follow their node
//! rigidly, and every animation becomes a clip.

use std::io::Read;
use std::path::Path;

use glam::{Mat4, Quat, Vec2, Vec3};
use substancecraft_core::{
    AnimationClip, Influence, Joint, JointAnimation, MAX_INFLUENCES, MeshGeometry, MeshVertex, Rig,
    SkinBinding, Track, strongest_influences,
};
use thiserror::Error;

use crate::fbx::{ImportedMesh, generate_planar_uvs};

pub const MAX_GLTF_BYTES: usize = 256 * 1024 * 1024;
const MAX_GLTF_VERTICES: usize = 3_000_000;
const MAX_NODE_DEPTH: usize = 128;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GltfImportError {
    #[error("glTF file I/O failed: {0}")]
    Io(String),
    #[error("glTF file exceeds the {} MiB import limit", MAX_GLTF_BYTES / (1024 * 1024))]
    TooLarge,
    #[error("glTF could not be parsed: {0}")]
    Parse(String),
    #[error("glTF mesh data is invalid: {0}")]
    InvalidMesh(String),
    #[error("glTF contains no triangle meshes")]
    NoTriangles,
    #[error("glTF exceeds the {MAX_GLTF_VERTICES}-vertex import limit")]
    TooManyVertices,
}

/// Imports a `.gltf` or `.glb` file. External buffers resolve relative to
/// the file's folder.
pub fn import_gltf_file(path: impl AsRef<Path>) -> Result<ImportedMesh, GltfImportError> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|error| GltfImportError::Io(error.to_string()))?;
    let mut bytes = Vec::new();
    file.take(MAX_GLTF_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| GltfImportError::Io(error.to_string()))?;
    let name = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "glTF".to_owned());
    import_gltf(&bytes, path.parent(), name)
}

/// Imports glTF bytes; `base` resolves external buffer URIs.
pub fn import_gltf(
    bytes: &[u8],
    base: Option<&Path>,
    name: String,
) -> Result<ImportedMesh, GltfImportError> {
    if bytes.len() > MAX_GLTF_BYTES {
        return Err(GltfImportError::TooLarge);
    }
    let gltf::Gltf { document, blob } =
        gltf::Gltf::from_slice(bytes).map_err(|error| GltfImportError::Parse(error.to_string()))?;
    let buffers = gltf::import_buffers(&document, base, blob)
        .map_err(|error| GltfImportError::Parse(error.to_string()))?;

    let rigged = document.skins().next().is_some() || document.animations().next().is_some();
    let mut importer = Importer {
        buffers: &buffers,
        geometry: MeshGeometry::default(),
        has_uvs: true,
        rig: rigged.then(|| Rig {
            joints: node_joints(&document),
            ..Rig::default()
        }),
        skin_bindings: std::collections::BTreeMap::new(),
    };
    match document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        Some(scene) => {
            for node in scene.nodes() {
                importer.visit(&node, Mat4::IDENTITY, 0)?;
            }
        }
        None => {
            for node in document.nodes() {
                if let Some(mesh) = node.mesh() {
                    let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
                    importer.add_mesh(&node, &mesh, transform)?;
                }
            }
        }
    }
    let mut geometry = importer.geometry;
    if geometry.triangles.is_empty() {
        return Err(GltfImportError::NoTriangles);
    }
    let generated_uvs = !importer.has_uvs;
    if generated_uvs {
        generate_planar_uvs(&mut geometry.vertices)
            .map_err(|error| GltfImportError::InvalidMesh(error.to_string()))?;
    }
    geometry
        .validate()
        .map_err(|error| GltfImportError::InvalidMesh(error.to_string()))?;
    let rig = match importer.rig {
        Some(mut rig) => {
            rig.clips = read_clips(&document, &buffers)?;
            rig.validate(geometry.vertices.len())
                .map_err(|error| GltfImportError::InvalidMesh(error.to_string()))?;
            Some(rig)
        }
        None => None,
    };
    Ok(ImportedMesh {
        name,
        geometry,
        generated_uvs,
        rig,
    })
}

/// One joint per node, at its rest transform, with its parent.
fn node_joints(document: &gltf::Document) -> Vec<Joint> {
    let mut parents = vec![None; document.nodes().len()];
    for node in document.nodes() {
        for child in node.children() {
            if let Some(slot) = parents.get_mut(child.index()) {
                *slot = Some(node.index() as u32);
            }
        }
    }
    document
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            Joint::new(
                node.name()
                    .map_or_else(|| format!("Node {}", node.index()), str::to_owned),
                parents[node.index()],
                Vec3::from_array(translation),
                Quat::from_array(rotation),
                Vec3::from_array(scale),
            )
        })
        .collect()
}

/// Every animation as a clip of per-node tracks. Morph-target weights are
/// not read; cubic-spline keys keep their values and drop their tangents.
fn read_clips(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<AnimationClip>, GltfImportError> {
    use gltf::animation::{Interpolation, util::ReadOutputs};
    let invalid = |index: usize| {
        GltfImportError::InvalidMesh(format!("animation {index} has unreadable keys"))
    };
    let mut clips = Vec::new();
    for animation in document.animations() {
        let mut joints: std::collections::BTreeMap<u32, JointAnimation> =
            std::collections::BTreeMap::new();
        for channel in animation.channels() {
            let reader =
                channel.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
            let times: Vec<f32> = reader
                .read_inputs()
                .ok_or_else(|| invalid(animation.index()))?
                .collect();
            let interpolation = channel.sampler().interpolation();
            let cubic = interpolation == Interpolation::CubicSpline;
            let step = interpolation == Interpolation::Step;
            let node = channel.target().node().index() as u32;
            let joint = joints.entry(node).or_insert_with(|| JointAnimation {
                joint: node,
                ..JointAnimation::default()
            });
            match reader.read_outputs() {
                Some(ReadOutputs::Translations(values)) => {
                    joint.translation = Some(Track {
                        times,
                        values: key_values(values.map(Vec3::from_array).collect(), cubic),
                        step,
                    });
                }
                Some(ReadOutputs::Rotations(values)) => {
                    joint.rotation = Some(Track {
                        times,
                        values: key_values(
                            values.into_f32().map(Quat::from_array).collect(),
                            cubic,
                        ),
                        step,
                    });
                }
                Some(ReadOutputs::Scales(values)) => {
                    joint.scale = Some(Track {
                        times,
                        values: key_values(values.map(Vec3::from_array).collect(), cubic),
                        step,
                    });
                }
                Some(ReadOutputs::MorphTargetWeights(_)) => {}
                None => return Err(invalid(animation.index())),
            }
        }
        let mut clip = AnimationClip {
            name: animation
                .name()
                .filter(|name| !name.trim().is_empty())
                .map_or_else(
                    || format!("Animation {}", animation.index() + 1),
                    str::to_owned,
                ),
            duration: 0.0,
            joints: joints.into_values().collect(),
        };
        clip.duration = clip.key_span();
        clips.push(clip);
    }
    Ok(clips)
}

/// Key values of a sampler; cubic-spline outputs are (in-tangent, value,
/// out-tangent) triples, of which the value is kept.
fn key_values<T>(values: Vec<T>, cubic: bool) -> Vec<T> {
    if cubic {
        values.into_iter().skip(1).step_by(3).collect()
    } else {
        values
    }
}

struct Importer<'a> {
    buffers: &'a [gltf::buffer::Data],
    geometry: MeshGeometry,
    /// False once any primitive lacks UVs; planar UVs then replace all.
    has_uvs: bool,
    /// Present when the file has skins or animations; influences grow with
    /// the geometry.
    rig: Option<Rig>,
    /// First binding of each skin added so far, by skin index.
    skin_bindings: std::collections::BTreeMap<usize, u32>,
}

impl Importer<'_> {
    fn visit(
        &mut self,
        node: &gltf::Node<'_>,
        parent: Mat4,
        depth: usize,
    ) -> Result<(), GltfImportError> {
        if depth > MAX_NODE_DEPTH {
            return Err(GltfImportError::InvalidMesh(
                "node hierarchy is too deep".to_owned(),
            ));
        }
        let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            self.add_mesh(node, &mesh, transform)?;
        }
        for child in node.children() {
            self.visit(&child, transform, depth + 1)?;
        }
        Ok(())
    }

    /// Bindings for `node`'s mesh: its skin's joints, or the node itself
    /// for a rigid mesh placed by `transform`.
    fn bindings_for(&mut self, node: &gltf::Node<'_>, transform: Mat4) -> Option<u32> {
        let buffers = self.buffers;
        let rig = self.rig.as_mut()?;
        if let Some(skin) = node.skin() {
            if let Some(first) = self.skin_bindings.get(&skin.index()) {
                return Some(*first);
            }
            let first = rig.bindings.len() as u32;
            let inverse_binds: Vec<Mat4> = skin
                .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()))
                .read_inverse_bind_matrices()
                .map(|matrices| matrices.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                .unwrap_or_default();
            for (index, joint) in skin.joints().enumerate() {
                rig.bindings.push(SkinBinding {
                    joint: joint.index() as u32,
                    before: Mat4::IDENTITY,
                    inverse_bind: inverse_binds.get(index).copied().unwrap_or(Mat4::IDENTITY),
                });
            }
            self.skin_bindings.insert(skin.index(), first);
            return Some(first);
        }
        let binding = rig.bindings.len() as u32;
        rig.bindings.push(SkinBinding {
            joint: node.index() as u32,
            before: Mat4::IDENTITY,
            inverse_bind: transform.inverse(),
        });
        Some(binding)
    }

    fn add_mesh(
        &mut self,
        node: &gltf::Node<'_>,
        mesh: &gltf::Mesh<'_>,
        transform: Mat4,
    ) -> Result<(), GltfImportError> {
        // A skinned mesh is posed by its joints alone; its vertices are
        // already in the space the inverse bind matrices expect.
        let skinned = self.rig.is_some() && node.skin().is_some();
        let transform = if skinned { Mat4::IDENTITY } else { transform };
        let first_binding = self.bindings_for(node, transform);
        let flips_winding = transform.determinant() < 0.0;
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|buffer| {
                self.buffers
                    .get(buffer.index())
                    .map(|data| data.0.as_slice())
            });
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions = positions.collect::<Vec<_>>();
            if self.geometry.vertices.len() + positions.len() > MAX_GLTF_VERTICES {
                return Err(GltfImportError::TooManyVertices);
            }
            let uvs = reader
                .read_tex_coords(0)
                .map(|coordinates| coordinates.into_f32().collect::<Vec<_>>());
            if uvs.as_ref().is_none_or(|uvs| uvs.len() != positions.len()) {
                self.has_uvs = false;
            }
            let indices = match reader.read_indices() {
                Some(indices) => indices.into_u32().collect::<Vec<_>>(),
                None => (0..positions.len() as u32).collect(),
            };
            if indices.len() % 3 != 0
                || indices
                    .iter()
                    .any(|index| *index as usize >= positions.len())
            {
                return Err(GltfImportError::InvalidMesh(format!(
                    "primitive of mesh `{}` has invalid indices",
                    mesh.name().unwrap_or("unnamed")
                )));
            }
            let influences: Vec<[Influence; MAX_INFLUENCES]> = match first_binding {
                Some(first) if skinned => {
                    let joints: Vec<[u16; 4]> = reader
                        .read_joints(0)
                        .map(|joints| joints.into_u16().collect())
                        .unwrap_or_default();
                    let weights: Vec<[f32; 4]> = reader
                        .read_weights(0)
                        .map(|weights| weights.into_f32().collect())
                        .unwrap_or_default();
                    (0..positions.len())
                        .map(|vertex| {
                            let joints = joints.get(vertex).copied().unwrap_or_default();
                            let weights = weights.get(vertex).copied().unwrap_or_default();
                            strongest_influences(
                                joints
                                    .into_iter()
                                    .zip(weights)
                                    .map(|(joint, weight)| (first + u32::from(joint), weight)),
                            )
                        })
                        .collect()
                }
                Some(binding) => vec![strongest_influences([(binding, 1.0)]); positions.len()],
                None => Vec::new(),
            };
            let vertices = positions
                .iter()
                .enumerate()
                .map(|(index, position)| MeshVertex {
                    position: transform.transform_point3(Vec3::from_array(*position)),
                    // glTF's V runs down the image; ours runs up.
                    uv: uvs
                        .as_ref()
                        .and_then(|uvs| uvs.get(index))
                        .map_or(Vec2::ZERO, |uv| Vec2::new(uv[0], 1.0 - uv[1])),
                })
                .collect::<Vec<_>>();
            let triangles = indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|&[a, b, c]| if flips_winding { [a, c, b] } else { [a, b, c] })
                .collect::<Vec<_>>();
            let material = primitive.material();
            let slot = match (material.name(), material.index()) {
                (Some(name), _) if !name.trim().is_empty() => name.to_owned(),
                (_, Some(index)) => format!("Material {}", index + 1),
                _ => "Default".to_owned(),
            };
            let part = MeshGeometry {
                vertices,
                triangle_materials: vec![0; triangles.len()],
                triangles,
                material_slots: vec![slot],
            };
            self.geometry
                .append(&part)
                .map_err(|_| GltfImportError::TooManyVertices)?;
            if let Some(rig) = self.rig.as_mut() {
                rig.influences.extend(influences);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Two triangles, each with its own material, sharing one buffer. The
    /// second node is translated by +2 on X.
    fn write_two_material_gltf(folder: &Path) -> std::path::PathBuf {
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let uvs: [[f32; 2]; 3] = [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]];
        let mut buffer = Vec::new();
        for position in positions {
            for value in position {
                buffer.extend(value.to_le_bytes());
            }
        }
        for uv in uvs {
            for value in uv {
                buffer.extend(value.to_le_bytes());
            }
        }
        for index in [0_u16, 1, 2] {
            buffer.extend(index.to_le_bytes());
        }
        buffer.extend([0, 0]);
        std::fs::write(folder.join("model.bin"), &buffer).unwrap();
        let json = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0, 1] }}],
  "nodes": [
    {{ "mesh": 0 }},
    {{ "mesh": 1, "translation": [2.0, 0.0, 0.0] }}
  ],
  "materials": [{{ "name": "Body" }}, {{}}],
  "meshes": [
    {{ "primitives": [{{ "attributes": {{ "POSITION": 0, "TEXCOORD_0": 1 }}, "indices": 2, "material": 0 }}] }},
    {{ "primitives": [{{ "attributes": {{ "POSITION": 0, "TEXCOORD_0": 1 }}, "indices": 2, "material": 1 }}] }}
  ],
  "buffers": [{{ "uri": "model.bin", "byteLength": {length} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }},
    {{ "buffer": 0, "byteOffset": 36, "byteLength": 24 }},
    {{ "buffer": 0, "byteOffset": 60, "byteLength": 6 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
       "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] }},
    {{ "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2" }},
    {{ "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }}
  ]
}}"#,
            length = buffer.len()
        );
        let path = folder.join("model.gltf");
        std::fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn tc_0092_gltf_import_flattens_nodes_and_keeps_material_slots() {
        let folder = tempfile::tempdir().unwrap();
        let path = write_two_material_gltf(folder.path());

        let imported = import_gltf_file(&path).unwrap();
        let geometry = &imported.geometry;
        assert_eq!(imported.name, "model");
        assert!(!imported.generated_uvs);
        assert_eq!(geometry.triangles.len(), 2);
        assert_eq!(geometry.vertices.len(), 6);
        assert_eq!(geometry.material_slots, vec!["Body", "Material 2"]);
        assert_eq!(geometry.triangle_materials, vec![0, 1]);
        // The second node's translation is applied.
        assert_eq!(geometry.vertices[4].position, Vec3::new(3.0, 0.0, 0.0));
        // V is flipped into SubstanceCraft's up-is-+V convention.
        assert_eq!(geometry.vertices[0].uv, Vec2::new(0.0, 0.0));
        assert_eq!(geometry.vertices[2].uv, Vec2::new(0.0, 1.0));

        let body = geometry.submesh(0);
        assert_eq!(body.triangles, vec![[0, 1, 2]]);
        assert_eq!(body.vertices.len(), 3);

        assert!(matches!(
            import_gltf(b"{ not json", None, "broken".to_owned()),
            Err(GltfImportError::Parse(_))
        ));
    }

    /// A skinned two-joint arm (shoulder at the origin, elbow at +X) whose
    /// elbow turns 90° about Z over two seconds, and a static triangle
    /// parented to the elbow. The skinned mesh's node is translated, which
    /// glTF says to ignore.
    fn write_rigged_gltf(folder: &Path) -> std::path::PathBuf {
        let mut buffer = Vec::new();
        let floats = |buffer: &mut Vec<u8>, values: &[f32]| {
            for value in values {
                buffer.extend(value.to_le_bytes());
            }
        };
        // 0: positions (36 bytes)
        floats(&mut buffer, &[0.5, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.1, 0.0]);
        // 36: joints, u8 × 4 per vertex (12 bytes)
        buffer.extend([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
        // 48: weights (48 bytes)
        floats(
            &mut buffer,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        // 96: indices (6 bytes + 2 padding)
        for index in [0_u16, 1, 2] {
            buffer.extend(index.to_le_bytes());
        }
        buffer.extend([0, 0]);
        // 104: inverse bind matrices (128 bytes), column-major
        floats(&mut buffer, &Mat4::IDENTITY.to_cols_array());
        floats(
            &mut buffer,
            &Mat4::from_translation(-Vec3::X).to_cols_array(),
        );
        // 232: key times (8 bytes)
        floats(&mut buffer, &[0.0, 2.0]);
        // 240: rotations (32 bytes)
        let quarter = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        floats(&mut buffer, &[0.0, 0.0, 0.0, 1.0]);
        floats(&mut buffer, &quarter.to_array());
        std::fs::write(folder.join("arm.bin"), &buffer).unwrap();
        let json = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0, 1] }}],
  "nodes": [
    {{ "name": "Arm", "mesh": 0, "skin": 0, "translation": [5.0, 0.0, 0.0] }},
    {{ "name": "Shoulder", "children": [2] }},
    {{ "name": "Elbow", "translation": [1.0, 0.0, 0.0], "children": [3] }},
    {{ "name": "Gauntlet", "mesh": 1 }}
  ],
  "skins": [{{ "joints": [1, 2], "inverseBindMatrices": 4 }}],
  "animations": [{{
    "name": "Bend",
    "channels": [{{ "sampler": 0, "target": {{ "node": 2, "path": "rotation" }} }}],
    "samplers": [{{ "input": 5, "output": 6 }}]
  }}],
  "meshes": [
    {{ "primitives": [{{ "attributes": {{ "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2 }}, "indices": 3 }}] }},
    {{ "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "indices": 3 }}] }}
  ],
  "buffers": [{{ "uri": "arm.bin", "byteLength": {length} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }},
    {{ "buffer": 0, "byteOffset": 36, "byteLength": 12 }},
    {{ "buffer": 0, "byteOffset": 48, "byteLength": 48 }},
    {{ "buffer": 0, "byteOffset": 96, "byteLength": 6 }},
    {{ "buffer": 0, "byteOffset": 104, "byteLength": 128 }},
    {{ "buffer": 0, "byteOffset": 232, "byteLength": 8 }},
    {{ "buffer": 0, "byteOffset": 240, "byteLength": 32 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
       "min": [0.5, 0.0, 0.0], "max": [2.0, 0.1, 0.0] }},
    {{ "bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4" }},
    {{ "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4" }},
    {{ "bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR" }},
    {{ "bufferView": 4, "componentType": 5126, "count": 2, "type": "MAT4" }},
    {{ "bufferView": 5, "componentType": 5126, "count": 2, "type": "SCALAR",
       "min": [0.0], "max": [2.0] }},
    {{ "bufferView": 6, "componentType": 5126, "count": 2, "type": "VEC4" }}
  ]
}}"#,
            length = buffer.len()
        );
        let path = folder.join("arm.gltf");
        std::fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn tc_0318_gltf_skins_and_animations_become_a_rig() {
        use substancecraft_core::ClipSample;
        let folder = tempfile::tempdir().unwrap();
        let imported = import_gltf_file(write_rigged_gltf(folder.path())).unwrap();
        let geometry = &imported.geometry;
        let rig = imported.rig.as_ref().unwrap();
        // The skinned arm keeps its bind-space positions; the static
        // triangle sits where its node puts it (under the elbow, at +1 X).
        assert_eq!(geometry.vertices.len(), 6);
        assert_eq!(geometry.vertices[1].position, Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(geometry.vertices[4].position, Vec3::new(3.0, 0.0, 0.0));
        assert_eq!(rig.joints.len(), 4);
        assert_eq!(rig.joints[2].name, "Elbow");
        assert_eq!(rig.joints[2].parent, Some(1));
        assert_eq!(rig.influences.len(), 6);
        assert_eq!(rig.clips.len(), 1);
        assert_eq!(rig.clips[0].name, "Bend");
        assert_eq!(rig.clips[0].duration, 2.0);

        let positions: Vec<Vec3> = geometry.vertices.iter().map(|v| v.position).collect();
        let close = |a: Vec3, b: Vec3| (a - b).length() < 1e-5;
        // At rest every vertex is where it was imported.
        let rest = rig.pose_positions(&positions, 1.0, None).unwrap();
        assert!(rest.iter().zip(&positions).all(|(a, b)| close(*a, *b)));
        // At the end the elbow has turned: the skinned tip and the rigid
        // triangle both swing up around it.
        let clip = &rig.clips[0];
        let end = rig
            .pose_positions(
                &positions,
                1.0,
                Some(ClipSample {
                    clip,
                    time: 2.0,
                    in_place: false,
                }),
            )
            .unwrap();
        assert!(close(end[0], positions[0]), "the shoulder vertex stays");
        assert!(close(end[1], Vec3::new(1.0, 1.0, 0.0)), "{}", end[1]);
        assert!(close(end[4], Vec3::new(1.0, 2.0, 0.0)), "{}", end[4]);

        // Files without skins or animations have no rig.
        let plain = import_gltf_file(write_two_material_gltf(folder.path())).unwrap();
        assert!(plain.rig.is_none());
    }
}
