use std::io::Cursor;

use fbxcel_dom::{
    any::AnyDocument,
    v7400::{
        data::mesh::layer::TypedLayerElementHandle,
        object::{TypedObjectHandle, geometry::TypedGeometryHandle},
    },
};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use substancecraft_core::{MeshGeometry, MeshVertex, Rig};
use thiserror::Error;

pub const MAX_FBX_BYTES: usize = 64 * 1024 * 1024;
const MAX_IMPORTED_VERTICES: usize = 3_000_000;
pub(crate) const MAX_DECODED_ATTRIBUTES: u64 = 128 * 1024 * 1024;
pub(crate) const MAX_FBX_NODES: usize = 1_000_000;
pub(crate) const MAX_FBX_DEPTH: usize = 128;
const FBX_BINARY_MAGIC: &[u8] = b"Kaydara FBX Binary  \0\x1a\0";

struct BinaryPreflight<'a> {
    bytes: &'a [u8],
    wide_headers: bool,
    decoded_bytes: u64,
    node_count: usize,
}

impl<'a> BinaryPreflight<'a> {
    fn validate(bytes: &'a [u8]) -> Result<(), FbxImportError> {
        if bytes.get(..FBX_BINARY_MAGIC.len()) != Some(FBX_BINARY_MAGIC) {
            return Err(FbxImportError::Parse(
                "only binary FBX input is supported".to_owned(),
            ));
        }
        let version = read_u32(bytes, FBX_BINARY_MAGIC.len())?;
        if version < 7_400 {
            return Err(FbxImportError::UnsupportedVersion);
        }
        let mut preflight = Self {
            bytes,
            wide_headers: version >= 7_500,
            decoded_bytes: 0,
            node_count: 0,
        };
        let first_node = FBX_BINARY_MAGIC
            .len()
            .checked_add(4)
            .ok_or_else(|| FbxImportError::InvalidBinary("header overflow".to_owned()))?;
        preflight.scan_sequence(first_node, bytes.len(), 0)?;
        Ok(())
    }

    fn node_header_size(&self) -> usize {
        if self.wide_headers { 25 } else { 13 }
    }

    fn scan_sequence(
        &mut self,
        mut offset: usize,
        bound: usize,
        depth: usize,
    ) -> Result<usize, FbxImportError> {
        if depth > MAX_FBX_DEPTH {
            return Err(FbxImportError::InvalidBinary(
                "node nesting limit exceeded".to_owned(),
            ));
        }
        loop {
            let header_size = self.node_header_size();
            if offset > bound || bound - offset < header_size {
                return Err(FbxImportError::InvalidBinary(format!(
                    "truncated node header at offset {offset} within boundary {bound}"
                )));
            }
            let Some((end_offset, property_count, property_bytes, name_bytes)) =
                self.read_node_header(offset)?
            else {
                return Ok(offset + header_size);
            };
            self.node_count = self.node_count.saturating_add(1);
            if self.node_count > MAX_FBX_NODES {
                return Err(FbxImportError::InvalidBinary(
                    "node count limit exceeded".to_owned(),
                ));
            }
            let end_offset = usize::try_from(end_offset).map_err(|_| {
                FbxImportError::InvalidBinary("node end offset is too large".to_owned())
            })?;
            let header_end = offset.checked_add(header_size).ok_or_else(|| {
                FbxImportError::InvalidBinary("node header offset overflow".to_owned())
            })?;
            if end_offset > bound || end_offset <= header_end {
                return Err(FbxImportError::InvalidBinary(
                    "node end offset is outside its parent".to_owned(),
                ));
            }
            let name_end = header_end.checked_add(name_bytes).ok_or_else(|| {
                FbxImportError::InvalidBinary("node name length overflow".to_owned())
            })?;
            if name_end > end_offset {
                return Err(FbxImportError::InvalidBinary(
                    "node name exceeds its parent".to_owned(),
                ));
            }
            let property_end = name_end.checked_add(property_bytes).ok_or_else(|| {
                FbxImportError::InvalidBinary("property block overflow".to_owned())
            })?;
            if property_end > end_offset || property_count > property_bytes as u64 {
                return Err(FbxImportError::InvalidBinary(
                    "property block bounds are invalid".to_owned(),
                ));
            }

            let property_count = usize::try_from(property_count).map_err(|_| {
                FbxImportError::InvalidBinary("property count is too large".to_owned())
            })?;
            let mut property_offset = name_end;
            for _ in 0..property_count {
                property_offset = self.scan_attribute(property_offset, property_end)?;
            }
            if property_offset != property_end {
                return Err(FbxImportError::InvalidBinary(
                    "property block length does not match its attributes".to_owned(),
                ));
            }

            if property_end < end_offset {
                let child_end = self.scan_sequence(property_end, end_offset, depth + 1)?;
                if child_end != end_offset {
                    return Err(FbxImportError::InvalidBinary(
                        "child nodes do not end at the declared node boundary".to_owned(),
                    ));
                }
            }
            offset = end_offset;
        }
    }

    fn read_node_header(
        &self,
        offset: usize,
    ) -> Result<Option<(u64, u64, usize, usize)>, FbxImportError> {
        if self.wide_headers {
            let end_offset = read_u64(self.bytes, offset)?;
            let property_count = read_u64(self.bytes, offset + 8)?;
            let property_bytes = read_u64(self.bytes, offset + 16)?;
            let name_bytes = *self
                .bytes
                .get(offset + 24)
                .ok_or_else(|| FbxImportError::InvalidBinary("truncated node header".to_owned()))?;
            if end_offset == 0 && property_count == 0 && property_bytes == 0 && name_bytes == 0 {
                return Ok(None);
            }
            let property_bytes = usize::try_from(property_bytes).map_err(|_| {
                FbxImportError::InvalidBinary("property block is too large".to_owned())
            })?;
            Ok(Some((
                end_offset,
                property_count,
                property_bytes,
                name_bytes as usize,
            )))
        } else {
            let end_offset = read_u32(self.bytes, offset)? as u64;
            let property_count = read_u32(self.bytes, offset + 4)? as u64;
            let property_bytes = read_u32(self.bytes, offset + 8)? as usize;
            let name_bytes = *self
                .bytes
                .get(offset + 12)
                .ok_or_else(|| FbxImportError::InvalidBinary("truncated node header".to_owned()))?;
            if end_offset == 0 && property_count == 0 && property_bytes == 0 && name_bytes == 0 {
                return Ok(None);
            }
            Ok(Some((
                end_offset,
                property_count,
                property_bytes,
                name_bytes as usize,
            )))
        }
    }

    fn scan_attribute(
        &mut self,
        offset: usize,
        property_end: usize,
    ) -> Result<usize, FbxImportError> {
        let kind = *self
            .bytes
            .get(offset)
            .ok_or_else(|| FbxImportError::InvalidBinary("missing property type".to_owned()))?;
        let mut value_offset = offset
            .checked_add(1)
            .ok_or_else(|| FbxImportError::InvalidBinary("property offset overflow".to_owned()))?;
        let payload_bytes = match kind {
            b'Y' => 2,
            b'C' => 1,
            b'I' | b'F' => 4,
            b'D' | b'L' => 8,
            b'S' | b'R' => {
                let length = read_u32(self.bytes, value_offset)? as u64;
                value_offset += 4;
                self.charge_decoded_bytes(length)?;
                usize::try_from(length).map_err(|_| {
                    FbxImportError::InvalidBinary("string property is too large".to_owned())
                })?
            }
            b'b' | b'c' | b'i' | b'l' | b'f' | b'd' => {
                let count = read_u32(self.bytes, value_offset)? as u64;
                let encoding = read_u32(self.bytes, value_offset + 4)?;
                let compressed_bytes = read_u32(self.bytes, value_offset + 8)? as u64;
                value_offset += 12;
                let bytes_per_item = match kind {
                    b'b' | b'c' => 1,
                    b'i' | b'f' => 4,
                    b'l' | b'd' => 8,
                    _ => 0,
                };
                let decoded_bytes = count.checked_mul(bytes_per_item).ok_or_else(|| {
                    FbxImportError::InvalidBinary("array decoded-size overflow".to_owned())
                })?;
                self.charge_decoded_bytes(decoded_bytes)?;
                match encoding {
                    0 => usize::try_from(decoded_bytes).map_err(|_| {
                        FbxImportError::InvalidBinary("array is too large".to_owned())
                    })?,
                    1 => usize::try_from(compressed_bytes).map_err(|_| {
                        FbxImportError::InvalidBinary("compressed array is too large".to_owned())
                    })?,
                    _ => {
                        return Err(FbxImportError::InvalidBinary(
                            "unknown array encoding".to_owned(),
                        ));
                    }
                }
            }
            _ => {
                return Err(FbxImportError::InvalidBinary(
                    "unknown property type".to_owned(),
                ));
            }
        };
        let value_end = value_offset
            .checked_add(payload_bytes)
            .ok_or_else(|| FbxImportError::InvalidBinary("property length overflow".to_owned()))?;
        if value_end > property_end {
            return Err(FbxImportError::InvalidBinary(
                "property payload exceeds its declared block".to_owned(),
            ));
        }
        Ok(value_end)
    }

    fn charge_decoded_bytes(&mut self, bytes: u64) -> Result<(), FbxImportError> {
        self.decoded_bytes = self.decoded_bytes.checked_add(bytes).ok_or_else(|| {
            FbxImportError::InvalidBinary("decoded attribute-size overflow".to_owned())
        })?;
        if self.decoded_bytes > MAX_DECODED_ATTRIBUTES {
            return Err(FbxImportError::ResourceBudgetExceeded(self.decoded_bytes));
        }
        Ok(())
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, FbxImportError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| FbxImportError::InvalidBinary("integer offset overflow".to_owned()))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| FbxImportError::InvalidBinary("truncated integer".to_owned()))?;
    let array = <[u8; 4]>::try_from(slice)
        .map_err(|_| FbxImportError::InvalidBinary("truncated integer".to_owned()))?;
    Ok(u32::from_le_bytes(array))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, FbxImportError> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| FbxImportError::InvalidBinary("integer offset overflow".to_owned()))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| FbxImportError::InvalidBinary("truncated integer".to_owned()))?;
    let array = <[u8; 8]>::try_from(slice)
        .map_err(|_| FbxImportError::InvalidBinary("truncated integer".to_owned()))?;
    Ok(u64::from_le_bytes(array))
}

/// A mesh read from FBX or glTF.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImportedMesh {
    pub name: String,
    pub geometry: MeshGeometry,
    pub generated_uvs: bool,
    /// Skeleton, skin weights (one entry per vertex of `geometry`), and
    /// animation clips, when the file animates this mesh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig: Option<Rig>,
}

/// Former name of [`ImportedMesh`].
pub type ImportedFbxMesh = ImportedMesh;

/// Reads every mesh of a binary or ASCII FBX 7.4+ file.
pub fn import_fbx_meshes(bytes: &[u8]) -> Result<Vec<ImportedMesh>, FbxImportError> {
    check_input_size(bytes)?;
    if crate::fbx_ascii::is_ascii_fbx(bytes) {
        import_binary_document(&crate::fbx_ascii::ascii_to_binary(bytes)?)
    } else {
        import_binary_document(bytes)
    }
}

/// Reads every mesh of a binary FBX 7.4+ file; see [`import_fbx_meshes`]
/// for ASCII files too.
pub fn import_binary_fbx_meshes(bytes: &[u8]) -> Result<Vec<ImportedMesh>, FbxImportError> {
    check_input_size(bytes)?;
    import_binary_document(bytes)
}

fn check_input_size(bytes: &[u8]) -> Result<(), FbxImportError> {
    if bytes.is_empty() {
        return Err(FbxImportError::EmptyInput);
    }
    if bytes.len() > MAX_FBX_BYTES {
        return Err(FbxImportError::InputTooLarge(bytes.len()));
    }
    Ok(())
}

/// Imports binary FBX after its size check: binary converted from ASCII
/// may be larger than the input limit, but its arrays stay in budget.
fn import_binary_document(bytes: &[u8]) -> Result<Vec<ImportedMesh>, FbxImportError> {
    BinaryPreflight::validate(bytes)?;

    let document = AnyDocument::from_seekable_reader(Cursor::new(bytes))
        .map_err(|error| FbxImportError::Parse(error.to_string()))?;
    let document = match document {
        AnyDocument::V7400(_, document) => document,
        _ => return Err(FbxImportError::UnsupportedVersion),
    };

    let mut imported_meshes = Vec::new();
    for object in document.objects() {
        let TypedObjectHandle::Geometry(geometry) = object.get_typed() else {
            continue;
        };
        let TypedGeometryHandle::Mesh(mesh) = geometry else {
            continue;
        };
        let imported = import_mesh(mesh)?;
        imported_meshes.push(imported);
    }

    if imported_meshes.is_empty() {
        return Err(FbxImportError::NoMeshes);
    }
    Ok(imported_meshes)
}

fn import_mesh(
    mesh: fbxcel_dom::v7400::object::geometry::MeshHandle<'_>,
) -> Result<ImportedMesh, FbxImportError> {
    let polygon_vertices = mesh
        .polygon_vertices()
        .map_err(|error| FbxImportError::InvalidMesh(error.to_string()))?;
    let triangulated = polygon_vertices
        .triangulate_each(|_, polygon, output| {
            if polygon.len() >= 3 {
                for index in 1..polygon.len() - 1 {
                    output.push([polygon[0], polygon[index], polygon[index + 1]]);
                }
            }
            Ok(())
        })
        .map_err(|error| FbxImportError::InvalidMesh(error.to_string()))?;

    if triangulated.is_empty() {
        return Err(FbxImportError::NoTriangles);
    }
    if triangulated.len() > MAX_IMPORTED_VERTICES {
        return Err(FbxImportError::TooManyVertices(triangulated.len()));
    }

    let uv_data = mesh
        .layers()
        .flat_map(|layer| layer.layer_element_entries())
        .find_map(|entry| {
            let Ok(TypedLayerElementHandle::Uv(element)) = entry.typed_layer_element() else {
                return None;
            };
            element.uv().ok()
        });

    // Per-polygon material indices, into the materials connected to the
    // model that owns this geometry (in connection order).
    let material_data = mesh
        .layers()
        .flat_map(|layer| layer.layer_element_entries())
        .find_map(|entry| {
            let Ok(TypedLayerElementHandle::Material(element)) = entry.typed_layer_element() else {
                return None;
            };
            element.materials().ok()
        });
    let connected_materials: Vec<String> = mesh
        .models()
        .next()
        .map(|model| {
            model
                .materials()
                .enumerate()
                .map(|(index, material)| {
                    material
                        .name()
                        .filter(|name| !name.is_empty())
                        .map_or_else(|| format!("Material {}", index + 1), str::to_owned)
                })
                .collect()
        })
        .unwrap_or_default();
    let mut triangle_materials = Vec::new();

    let generated_uvs = uv_data.is_none();
    let mut vertices = Vec::with_capacity(triangulated.len());
    let mut triangles = Vec::with_capacity(triangulated.len() / 3);
    let mut control_points = Vec::with_capacity(triangulated.len());
    for (vertex_index, triangle_vertex) in triangulated.triangle_vertex_indices().enumerate() {
        if vertex_index % 3 == 0
            && let Some(materials) = &material_data
        {
            let slot = materials
                .material_index(&triangulated, triangle_vertex)
                .map_err(|error| FbxImportError::InvalidMesh(error.to_string()))?;
            triangle_materials.push(slot.to_u32());
        }
        control_points.push(
            triangulated
                .control_point_index(triangle_vertex)
                .ok_or(FbxImportError::InvalidControlPoint)?
                .to_u32(),
        );
        let point = triangulated
            .control_point(triangle_vertex)
            .ok_or(FbxImportError::InvalidControlPoint)?;
        let position = Vec3::new(point.x as f32, point.y as f32, point.z as f32);
        let uv = if let Some(uv_data) = uv_data {
            let point = uv_data
                .uv(&triangulated, triangle_vertex)
                .map_err(|error| FbxImportError::InvalidUv(error.to_string()))?;
            Vec2::new(point.x as f32, point.y as f32)
        } else {
            Vec2::ZERO
        };
        vertices.push(MeshVertex { position, uv });
        if vertex_index % 3 == 2 {
            let end = u32::try_from(vertex_index)
                .map_err(|_| FbxImportError::TooManyVertices(vertex_index.saturating_add(1)))?;
            let start = end - 2;
            triangles.push([start, start + 1, end]);
        }
    }

    if generated_uvs {
        generate_planar_uvs(&mut vertices)?;
    }

    // One slot per connected material, plus generated names for any index
    // past them, so every triangle's slot exists.
    let slot_count = triangle_materials
        .iter()
        .map(|slot| *slot as usize + 1)
        .max()
        .unwrap_or(0)
        .max(connected_materials.len());
    let material_slots: Vec<String> = (0..slot_count)
        .map(|slot| {
            connected_materials
                .get(slot)
                .cloned()
                .unwrap_or_else(|| format!("Material {}", slot + 1))
        })
        .collect();
    let geometry = MeshGeometry {
        vertices,
        triangles,
        triangle_materials: if material_slots.is_empty() {
            Vec::new()
        } else {
            triangle_materials
        },
        material_slots,
    };
    geometry
        .validate()
        .map_err(|error| FbxImportError::InvalidMesh(error.to_string()))?;

    let rig = crate::fbx_rig::read_rig(&mesh, &control_points)?;
    Ok(ImportedMesh {
        name: mesh.name().unwrap_or("Mesh").to_owned(),
        geometry,
        generated_uvs,
        rig,
    })
}

pub(crate) fn generate_planar_uvs(vertices: &mut [MeshVertex]) -> Result<(), FbxImportError> {
    let Some(first) = vertices.first() else {
        return Err(FbxImportError::NoTriangles);
    };
    let mut minimum = first.position;
    let mut maximum = first.position;
    for vertex in vertices.iter() {
        minimum = minimum.min(vertex.position);
        maximum = maximum.max(vertex.position);
    }

    let extent = maximum - minimum;
    let mut axes = [0_usize, 1, 2];
    axes.sort_by(|left, right| extent[*right].total_cmp(&extent[*left]));
    let u_axis = axes[0];
    let v_axis = axes[1];
    let u_extent = extent[u_axis];
    let v_extent = extent[v_axis];
    for vertex in vertices {
        let offset = vertex.position - minimum;
        vertex.uv = Vec2::new(
            if u_extent > f32::EPSILON {
                offset[u_axis] / u_extent
            } else {
                0.5
            },
            if v_extent > f32::EPSILON {
                offset[v_axis] / v_extent
            } else {
                0.5
            },
        );
    }
    Ok(())
}

#[derive(Debug, Error, PartialEq)]
pub enum FbxImportError {
    #[error("FBX input is empty")]
    EmptyInput,
    #[error("FBX input has {0} bytes; the import limit is 64 MiB")]
    InputTooLarge(usize),
    #[error("FBX could not be parsed: {0}")]
    Parse(String),
    #[error("FBX binary structure is invalid: {0}")]
    InvalidBinary(String),
    #[error("FBX decoded attributes exceed the 128 MiB budget ({0} bytes)")]
    ResourceBudgetExceeded(u64),
    #[error("this FBX version is not supported; FBX 7.4+ (binary or ASCII) is required")]
    UnsupportedVersion,
    #[error("the FBX contains no mesh geometry")]
    NoMeshes,
    #[error("the FBX mesh contains no triangles")]
    NoTriangles,
    #[error("the FBX mesh contains too many triangle vertices: {0}")]
    TooManyVertices(usize),
    #[error("the FBX mesh contains an invalid control-point reference")]
    InvalidControlPoint,
    #[error("the FBX mesh contains invalid UV data: {0}")]
    InvalidUv(String),
    #[error("the FBX mesh is invalid: {0}")]
    InvalidMesh(String),
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use fbxcel::{low::FbxVersion, write_v7400_binary, writer::v7400::binary::Writer};

    #[test]
    fn empty_fbx_input_is_rejected_before_parsing() {
        assert_eq!(
            import_binary_fbx_meshes(&[]),
            Err(FbxImportError::EmptyInput)
        );
    }

    #[test]
    fn malformed_fbx_is_reported_as_a_parse_error() {
        assert!(matches!(
            import_binary_fbx_meshes(b"not an FBX file"),
            Err(FbxImportError::Parse(_))
        ));
    }

    #[test]
    fn planar_uv_fallback_uses_the_two_largest_axes() {
        let mut vertices = vec![
            MeshVertex {
                position: Vec3::new(0.0, 0.0, 0.0),
                uv: Vec2::ZERO,
            },
            MeshVertex {
                position: Vec3::new(2.0, 0.0, 0.0),
                uv: Vec2::ZERO,
            },
            MeshVertex {
                position: Vec3::new(0.0, 0.0, 1.0),
                uv: Vec2::ZERO,
            },
        ];

        assert!(generate_planar_uvs(&mut vertices).is_ok());
        assert_eq!(vertices[0].uv, Vec2::ZERO);
        assert_eq!(vertices[1].uv, Vec2::X);
        assert_eq!(vertices[2].uv, Vec2::Y);
    }

    #[test]
    fn tc_0015_binary_preflight_rejects_excessive_decoded_attribute_bytes() {
        let mut preflight = BinaryPreflight {
            bytes: &[],
            wide_headers: false,
            decoded_bytes: 0,
            node_count: 0,
        };

        assert!(matches!(
            preflight.charge_decoded_bytes(MAX_DECODED_ATTRIBUTES + 1),
            Err(FbxImportError::ResourceBudgetExceeded(_))
        ));
    }

    #[test]
    fn tc_0013_imports_a_binary_fbx_triangle_and_generates_uvs() {
        let mut writer = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
        write_v7400_binary!(
            writer = writer,
            tree = {
                Documents: {
                    Document: [0_i64, "Scene", "Scene"] {
                        RootNode: [0_i64] {}
                    }
                }
                Connections: {}
                Objects: {
                    Geometry: [1_i64, "Triangle\0\u{1}Geometry", "Mesh"] {
                        Vertices: [&[0.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0][..]] {}
                        PolygonVertexIndex: [&[0_i32, 1, -3][..]] {}
                    }
                }
            },
        )
        .unwrap();
        let bytes = writer
            .finalize_and_flush(&Default::default())
            .unwrap()
            .into_inner();

        let meshes = import_binary_fbx_meshes(&bytes).unwrap();

        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].name, "Triangle");
        assert!(meshes[0].generated_uvs);
        assert_eq!(meshes[0].geometry.vertices.len(), 3);
        assert_eq!(meshes[0].geometry.triangles, vec![[0, 1, 2]]);
        assert_eq!(meshes[0].geometry.vertices[1].uv, Vec2::X);
        assert_eq!(meshes[0].geometry.vertices[2].uv, Vec2::Y);
    }

    #[test]
    fn tc_0298_fbx_material_layers_become_named_material_slots() {
        let mut writer = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
        write_v7400_binary!(
            writer = writer,
            tree = {
                Documents: {
                    Document: [0_i64, "Scene\0\u{1}Document", "Scene"] {
                        RootNode: [0_i64] {}
                    }
                }
                Objects: {
                    Geometry: [1_i64, "Crate\0\u{1}Geometry", "Mesh"] {
                        Vertices: [&[
                            0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                        ][..]] {}
                        PolygonVertexIndex: [&[0_i32, 1, -3, 1, 3, -3][..]] {}
                        LayerElementMaterial: [0_i32] {
                            Version: [101_i32] {}
                            Name: [""] {}
                            MappingInformationType: ["ByPolygon"] {}
                            ReferenceInformationType: ["IndexToDirect"] {}
                            Materials: [&[0_i32, 1][..]] {}
                        }
                        Layer: [0_i32] {
                            Version: [100_i32] {}
                            LayerElement: [0_i32] {
                                Type: ["LayerElementMaterial"] {}
                                TypedIndex: [0_i32] {}
                            }
                        }
                    }
                    Model: [2_i64, "Crate\0\u{1}Model", "Mesh"] {}
                    Material: [3_i64, "Paint\0\u{1}Material", ""] {}
                    Material: [4_i64, "Metal\0\u{1}Material", ""] {}
                }
                Connections: {
                    C: ["OO", 1_i64, 2_i64] {}
                    C: ["OO", 3_i64, 2_i64] {}
                    C: ["OO", 4_i64, 2_i64] {}
                    C: ["OO", 2_i64, 0_i64] {}
                }
            },
        )
        .unwrap();
        let bytes = writer
            .finalize_and_flush(&Default::default())
            .unwrap()
            .into_inner();

        let meshes = import_binary_fbx_meshes(&bytes).unwrap();
        let geometry = &meshes[0].geometry;
        assert_eq!(geometry.material_slots, ["Paint", "Metal"]);
        assert_eq!(geometry.triangle_materials, [0, 1]);
        assert_eq!(geometry.material_slot_count(), 2);
        assert_eq!(geometry.submesh(1).triangles.len(), 1);
        geometry.validate().unwrap();
    }

    /// A quad in two triangles with indexed UVs and two materials, as an
    /// Autodesk exporter writes it in ASCII.
    const ASCII_CRATE: &str = r#"; FBX 7.4.0 project file
; Copyright (C) 1997-2015 Autodesk Inc. and/or its licensors.
; All rights reserved.
; ----------------------------------------------------

FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 7400
	Creator: "FBX SDK/FBX Plugins version 2020.3"
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",1
		P: "TimeSpanStop", "KTime", "Time", "",46186158000
	}
}

; Object definitions
;------------------------------------------------------------------

Documents:  {
	Count: 1
	Document: 1890140528, "Scene", "Scene" {
		RootNode: 0
	}
}

; Object properties
;------------------------------------------------------------------

Objects:  {
	Geometry: 1001, "Geometry::Crate", "Mesh" {
		Vertices: *12 {
			a: 0,0,0,1,0,0,0,1,0,1,1,0
		}
		PolygonVertexIndex: *6 {
			a: 0,1,-3,1,3,-3
		}
		GeometryVersion: 124
		LayerElementUV: 0 {
			Version: 101
			Name: "UVChannel_1"
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "IndexToDirect"
			UV: *8 {
				a: 0,0,1,0,0,1,1,1
			}
			UVIndex: *6 {
				a: 0,1,2,1,3,2
			}
		}
		LayerElementMaterial: 0 {
			Version: 101
			Name: ""
			MappingInformationType: "ByPolygon"
			ReferenceInformationType: "IndexToDirect"
			Materials: *2 {
				a: 0,1
			}
		}
		Layer: 0 {
			Version: 100
			LayerElement:  {
				Type: "LayerElementUV"
				TypedIndex: 0
			}
			LayerElement:  {
				Type: "LayerElementMaterial"
				TypedIndex: 0
			}
		}
	}
	Model: 1002, "Model::Crate", "Mesh" {
		Version: 232
		Properties70:  {
			P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1
		}
		Shading: T
		Culling: "CullingOff"
	}
	Material: 1003, "Material::Paint", "" {
		Version: 102
		ShadingModel: "phong"
		Properties70:  {
			P: "DiffuseColor", "Color", "", "A",0.8,0.2,0.1
		}
	}
	Material: 1004, "Material::Metal", "" {
		Version: 102
		ShadingModel: "phong"
	}
}

; Object connections
;------------------------------------------------------------------

Connections:  {

	;Model::Crate, Model::RootNode
	C: "OO",1002,0

	;Geometry::Crate, Model::Crate
	C: "OO",1001,1002

	;Material::Paint, Model::Crate
	C: "OO",1003,1002

	;Material::Metal, Model::Crate
	C: "OO",1004,1002
}
"#;

    #[test]
    fn tc_0305_ascii_fbx_imports_like_the_same_binary_file() {
        let ascii = import_fbx_meshes(ASCII_CRATE.as_bytes()).unwrap();

        let mut writer = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
        write_v7400_binary!(
            writer = writer,
            tree = {
                Documents: {
                    Document: [1_890_140_528_i64, "Scene", "Scene"] {
                        RootNode: [0_i64] {}
                    }
                }
                Objects: {
                    Geometry: [1001_i64, "Crate\0\u{1}Geometry", "Mesh"] {
                        Vertices: [&[
                            0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                        ][..]] {}
                        PolygonVertexIndex: [&[0_i32, 1, -3, 1, 3, -3][..]] {}
                        LayerElementUV: [0_i32] {
                            Version: [101_i32] {}
                            Name: ["UVChannel_1"] {}
                            MappingInformationType: ["ByPolygonVertex"] {}
                            ReferenceInformationType: ["IndexToDirect"] {}
                            UV: [&[0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0][..]] {}
                            UVIndex: [&[0_i32, 1, 2, 1, 3, 2][..]] {}
                        }
                        LayerElementMaterial: [0_i32] {
                            Version: [101_i32] {}
                            Name: [""] {}
                            MappingInformationType: ["ByPolygon"] {}
                            ReferenceInformationType: ["IndexToDirect"] {}
                            Materials: [&[0_i32, 1][..]] {}
                        }
                        Layer: [0_i32] {
                            Version: [100_i32] {}
                            LayerElement: {
                                Type: ["LayerElementUV"] {}
                                TypedIndex: [0_i32] {}
                            }
                            LayerElement: {
                                Type: ["LayerElementMaterial"] {}
                                TypedIndex: [0_i32] {}
                            }
                        }
                    }
                    Model: [1002_i64, "Crate\0\u{1}Model", "Mesh"] {}
                    Material: [1003_i64, "Paint\0\u{1}Material", ""] {}
                    Material: [1004_i64, "Metal\0\u{1}Material", ""] {}
                }
                Connections: {
                    C: ["OO", 1002_i64, 0_i64] {}
                    C: ["OO", 1001_i64, 1002_i64] {}
                    C: ["OO", 1003_i64, 1002_i64] {}
                    C: ["OO", 1004_i64, 1002_i64] {}
                }
            },
        )
        .unwrap();
        let bytes = writer
            .finalize_and_flush(&Default::default())
            .unwrap()
            .into_inner();
        let binary = import_fbx_meshes(&bytes).unwrap();

        assert_eq!(ascii, binary);
        let mesh = &ascii[0];
        assert_eq!(mesh.name, "Crate");
        assert!(!mesh.generated_uvs);
        assert_eq!(mesh.geometry.material_slots, ["Paint", "Metal"]);
        assert_eq!(mesh.geometry.triangle_materials, [0, 1]);
        let uvs: Vec<Vec2> = mesh.geometry.vertices.iter().map(|v| v.uv).collect();
        assert_eq!(
            uvs,
            [Vec2::ZERO, Vec2::X, Vec2::Y, Vec2::X, Vec2::ONE, Vec2::Y]
        );
    }

    #[test]
    fn tc_0306_ascii_fbx_errors_are_typed_and_binary_only_entry_rejects_ascii() {
        // A byte-order mark is accepted.
        let with_bom = [b"\xEF\xBB\xBF".as_slice(), ASCII_CRATE.as_bytes()].concat();
        assert_eq!(import_fbx_meshes(&with_bom).unwrap().len(), 1);
        // Old ASCII files are refused by version, not by a parse failure.
        let old = ASCII_CRATE.replace("FBXVersion: 7400", "FBXVersion: 6100");
        assert_eq!(
            import_fbx_meshes(old.as_bytes()),
            Err(FbxImportError::UnsupportedVersion)
        );
        // A file with no geometry has no meshes.
        let start = ASCII_CRATE.find("\tGeometry:").unwrap();
        let end = ASCII_CRATE.find("\tModel:").unwrap();
        let empty = [&ASCII_CRATE[..start], &ASCII_CRATE[end..]].concat();
        assert_eq!(
            import_fbx_meshes(empty.as_bytes()),
            Err(FbxImportError::NoMeshes)
        );
        // Truncation is a parse error naming the line.
        let truncated = &ASCII_CRATE[..ASCII_CRATE.find("UVIndex").unwrap() + 20];
        assert!(matches!(
            import_fbx_meshes(truncated.as_bytes()),
            Err(FbxImportError::Parse(message)) if message.contains("ASCII FBX line")
        ));
        assert!(matches!(
            import_binary_fbx_meshes(ASCII_CRATE.as_bytes()),
            Err(FbxImportError::Parse(_))
        ));
    }

    #[test]
    fn tc_0014_preserves_polygon_vertex_uvs_from_binary_fbx() {
        let mut writer = Writer::new(Cursor::new(Vec::new()), FbxVersion::V7_4).unwrap();
        write_v7400_binary!(
            writer = writer,
            tree = {
                Documents: {
                    Document: [0_i64, "Scene\0\u{1}Document", "Scene"] {
                        RootNode: [0_i64] {}
                    }
                }
                Connections: {}
                Objects: {
                    Geometry: [1_i64, "TriangleUv\0\u{1}Geometry", "Mesh"] {
                        Vertices: [&[0.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0][..]] {}
                        PolygonVertexIndex: [&[0_i32, 1, -3][..]] {}
                        LayerElementUV: [0_i32] {
                            Version: [101_i32] {}
                            Name: ["UVChannel_1"] {}
                            MappingInformationType: ["ByPolygonVertex"] {}
                            ReferenceInformationType: ["Direct"] {}
                            UV: [&[0.25_f64, 0.5, 0.75, 0.5, 0.25, 1.0][..]] {}
                        }
                        Layer: [0_i32] {
                            Version: [100_i32] {}
                            LayerElement: [0_i32] {
                                Type: ["LayerElementUV"] {}
                                TypedIndex: [0_i32] {}
                            }
                        }
                    }
                }
            },
        )
        .unwrap();
        let bytes = writer
            .finalize_and_flush(&Default::default())
            .unwrap()
            .into_inner();

        let meshes = import_binary_fbx_meshes(&bytes).unwrap();

        assert_eq!(meshes.len(), 1);
        assert!(!meshes[0].generated_uvs);
        assert_eq!(meshes[0].geometry.vertices[0].uv, Vec2::new(0.25, 0.5));
        assert_eq!(meshes[0].geometry.vertices[1].uv, Vec2::new(0.75, 0.5));
        assert_eq!(meshes[0].geometry.vertices[2].uv, Vec2::new(0.25, 1.0));
    }

    /// A triangle skinned to a two-bone arm (shoulder at the origin, elbow
    /// at +X) with a one-second stack turning the elbow 90° about Z.
    const ASCII_ARM: &str = r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {
	FBXVersion: 7400
}
Documents:  {
	Document: 900, "Scene", "Scene" {
		RootNode: 0
	}
}
Objects:  {
	Geometry: 1, "Geometry::Arm", "Mesh" {
		Vertices: *9 {
			a: 0.5,0,0,2,0,0,2,0.1,0
		}
		PolygonVertexIndex: *3 {
			a: 0,1,-3
		}
	}
	Model: 10, "Model::Arm", "Mesh" {
		Version: 232
	}
	Model: 20, "Model::Shoulder", "LimbNode" {
		Version: 232
	}
	Model: 21, "Model::Elbow", "LimbNode" {
		Version: 232
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",1,0,0
		}
	}
	Deformer: 30, "Deformer::Skin", "Skin" {
		Version: 101
	}
	Deformer: 31, "SubDeformer::Shoulder", "Cluster" {
		Version: 100
		Indexes: *1 {
			a: 0
		}
		Weights: *1 {
			a: 1
		}
		Transform: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
		}
		TransformLink: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
		}
	}
	Deformer: 32, "SubDeformer::Elbow", "Cluster" {
		Version: 100
		Indexes: *2 {
			a: 1,2
		}
		Weights: *2 {
			a: 1,1
		}
		Transform: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
		}
		TransformLink: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1
		}
	}
	AnimationStack: 40, "AnimStack::Bend", "" {
		Properties70:  {
			P: "LocalStart", "KTime", "Time", "",0
			P: "LocalStop", "KTime", "Time", "",46186158000
		}
	}
	AnimationLayer: 41, "AnimLayer::BaseLayer", "" {
	}
	AnimationCurveNode: 42, "AnimCurveNode::R", "" {
		Properties70:  {
			P: "d|X", "Number", "", "A",0
			P: "d|Y", "Number", "", "A",0
			P: "d|Z", "Number", "", "A",0
		}
	}
	AnimationCurve: 43, "AnimCurve::", "" {
		Default: 0
		KeyVer: 4009
		KeyTime: *2 {
			a: 0,46186158000
		}
		KeyValueFloat: *2 {
			a: 0,90
		}
	}
}
Connections:  {
	C: "OO",1,10
	C: "OO",10,0
	C: "OO",20,0
	C: "OO",21,20
	C: "OO",30,1
	C: "OO",31,30
	C: "OO",32,30
	C: "OO",20,31
	C: "OO",21,32
	C: "OO",41,40
	C: "OO",42,41
	C: "OP",42,21, "Lcl Rotation"
	C: "OP",43,42, "d|Z"
}
"#;

    #[test]
    fn tc_0319_fbx_skins_bones_and_animation_stacks_become_a_rig() {
        use substancecraft_core::ClipSample;
        let meshes = import_fbx_meshes(ASCII_ARM.as_bytes()).unwrap();
        let mesh = &meshes[0];
        let rig = mesh.rig.as_ref().unwrap();
        let names: Vec<&str> = rig.joints.iter().map(|joint| joint.name.as_str()).collect();
        assert_eq!(names, ["Arm", "Shoulder", "Elbow"]);
        assert_eq!(rig.joints[2].parent, Some(1));
        assert_eq!(rig.bindings.len(), 2);
        assert_eq!(rig.clips.len(), 1);
        assert_eq!(rig.clips[0].name, "Bend");
        assert!((rig.clips[0].duration - 1.0).abs() < 1e-6);

        let positions: Vec<Vec3> = mesh.geometry.vertices.iter().map(|v| v.position).collect();
        let close = |a: Vec3, b: Vec3| (a - b).length() < 1e-4;
        let rest = rig.pose_positions(&positions, 1.0, None).unwrap();
        assert!(rest.iter().zip(&positions).all(|(a, b)| close(*a, *b)));
        let clip = &rig.clips[0];
        let end = rig
            .pose_positions(
                &positions,
                1.0,
                Some(ClipSample {
                    clip,
                    time: 1.0,
                    in_place: false,
                }),
            )
            .unwrap();
        // Vertex order follows the triangle: 0.5 X on the shoulder, then
        // the two elbow vertices.
        assert!(close(end[0], positions[0]));
        assert!(close(end[1], Vec3::new(1.0, 1.0, 0.0)), "{}", end[1]);
        let half = rig
            .pose_positions(
                &positions,
                1.0,
                Some(ClipSample {
                    clip,
                    time: 0.5,
                    in_place: false,
                }),
            )
            .unwrap();
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        assert!(close(half[1], Vec3::new(1.0 + diagonal, diagonal, 0.0)));

        // A static file has no rig.
        assert!(
            import_fbx_meshes(ASCII_CRATE.as_bytes()).unwrap()[0]
                .rig
                .is_none()
        );
    }
}
