#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use serde::{Deserialize, Serialize};
use substancecraft_core::{MeshError, PrimitivePreset};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveDescriptor {
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveSummary {
    pub name: String,
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub uv_bounds_min: [f32; 2],
    pub uv_bounds_max: [f32; 2],
}

pub fn list_primitive_presets() -> Vec<PrimitiveDescriptor> {
    PrimitivePreset::ALL
        .into_iter()
        .map(|preset| PrimitiveDescriptor {
            name: preset.name().to_owned(),
        })
        .collect()
}

pub fn inspect_primitive_preset(name: &str) -> Result<PrimitiveSummary, ApiError> {
    let preset = PrimitivePreset::ALL
        .into_iter()
        .find(|preset| preset.name().eq_ignore_ascii_case(name.trim()))
        .ok_or_else(|| ApiError::UnknownPrimitivePreset(name.to_owned()))?;
    let mesh = preset.build_mesh();
    mesh.validate()?;

    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    let mut uv_min = [f32::INFINITY; 2];
    let mut uv_max = [f32::NEG_INFINITY; 2];
    for vertex in &mesh.vertices {
        let position = vertex.position.to_array();
        let uv = vertex.uv.to_array();
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(position[axis]);
            bounds_max[axis] = bounds_max[axis].max(position[axis]);
        }
        for axis in 0..2 {
            uv_min[axis] = uv_min[axis].min(uv[axis]);
            uv_max[axis] = uv_max[axis].max(uv[axis]);
        }
    }

    Ok(PrimitiveSummary {
        name: preset.name().to_owned(),
        vertex_count: mesh.vertices.len(),
        triangle_count: mesh.triangles.len(),
        bounds_min,
        bounds_max,
        uv_bounds_min: uv_min,
        uv_bounds_max: uv_max,
    })
}

#[derive(Debug, Error, PartialEq)]
pub enum ApiError {
    #[error("unknown mesh preset `{0}`")]
    UnknownPrimitivePreset(String),
    #[error(transparent)]
    InvalidGeneratedMesh(#[from] MeshError),
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tc_0009_preset_listing_and_mesh_inspection_are_consistent() {
        let presets = list_primitive_presets();
        assert_eq!(presets.len(), 4);
        for preset in presets {
            let summary = inspect_primitive_preset(&preset.name).unwrap();
            assert_eq!(summary.name, preset.name);
            assert!(summary.vertex_count > 0);
            assert!(summary.triangle_count > 0);
            let expected_bounds = if preset.name == "Plane" {
                ([-0.5, 0.0, -0.5], [0.5, 0.0, 0.5])
            } else {
                ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
            };
            assert_eq!(summary.bounds_min, expected_bounds.0);
            assert_eq!(summary.bounds_max, expected_bounds.1);
            assert_eq!(summary.uv_bounds_min, [0.0, 0.0]);
            assert_eq!(summary.uv_bounds_max, [1.0, 1.0]);
        }
    }

    #[test]
    fn tc_0010_unknown_preset_is_a_recoverable_api_error() {
        assert_eq!(
            inspect_primitive_preset("icosahedron"),
            Err(ApiError::UnknownPrimitivePreset("icosahedron".to_owned()))
        );
    }
}
