#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::wrapper::Parameters,
    model::{Implementation, ServerCapabilities, ServerConfig},
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::Deserialize;
use substancecraft_api::{inspect_primitive_preset, list_primitive_presets};

#[derive(Debug, Deserialize, JsonSchema)]
struct InspectMeshPresetRequest {
    #[schemars(description = "Preset name: Cube, Sphere, Cylinder, or Plane")]
    name: String,
}

#[derive(Clone)]
struct SubstanceCraftMcp;

#[tool_router]
impl SubstanceCraftMcp {
    #[tool(description = "List the built-in SubstanceCraft starter mesh presets.")]
    fn list_mesh_presets(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&list_primitive_presets()).map_err(|error| error.to_string())
    }

    #[tool(description = "Inspect a starter mesh preset's topology, local bounds, and UV bounds.")]
    fn inspect_mesh_preset(
        &self,
        Parameters(request): Parameters<InspectMeshPresetRequest>,
    ) -> Result<String, String> {
        let summary = inspect_primitive_preset(&request.name).map_err(|error| error.to_string())?;
        serde_json::to_string_pretty(&summary).map_err(|error| error.to_string())
    }
}

#[tool_handler]
impl ServerHandler for SubstanceCraftMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "SubstanceCraft headless tools. Current tools inspect built-in primitive meshes; project editing tools will be added as their command API lands.",
            )
            .with_server_info(Implementation::new(
                "substancecraft-mcp",
                env!("CARGO_PKG_VERSION"),
            ))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    SubstanceCraftMcp.serve(stdio()).await?.waiting().await?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tc_0011_server_advertises_tool_capability() {
        let info = SubstanceCraftMcp.get_info();
        assert!(info.capabilities.tools.is_some());
        assert_eq!(info.server_info.name, "substancecraft-mcp");
    }

    #[test]
    fn tc_0012_mesh_inspection_tool_returns_serialized_summary() {
        let result = SubstanceCraftMcp
            .inspect_mesh_preset(Parameters(InspectMeshPresetRequest {
                name: "sphere".to_owned(),
            }))
            .unwrap();
        assert!(result.contains("\"name\": \"Sphere\""));
        assert!(result.contains("\"triangle_count\": 224"));
    }
}
