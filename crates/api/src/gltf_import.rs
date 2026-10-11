//! glTF 2.0 (`.gltf` with external or embedded buffers, and binary `.glb`)
//! static mesh import. The default scene is flattened into one mesh with
//! node transforms applied; each glTF material becomes a material slot.

use std::io::Read;
use std::path::Path;

use glam::{Mat4, Vec2, Vec3};
use substancecraft_core::{MeshGeometry, MeshVertex};
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

    let mut importer = Importer {
        buffers: &buffers,
        geometry: MeshGeometry::default(),
        has_uvs: true,
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
                    importer.add_mesh(&mesh, transform)?;
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
    Ok(ImportedMesh {
        name,
        geometry,
        generated_uvs,
    })
}

struct Importer<'a> {
    buffers: &'a [gltf::buffer::Data],
    geometry: MeshGeometry,
    /// False once any primitive lacks UVs; planar UVs then replace all.
    has_uvs: bool,
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
            self.add_mesh(&mesh, transform)?;
        }
        for child in node.children() {
            self.visit(&child, transform, depth + 1)?;
        }
        Ok(())
    }

    fn add_mesh(&mut self, mesh: &gltf::Mesh<'_>, transform: Mat4) -> Result<(), GltfImportError> {
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
}
