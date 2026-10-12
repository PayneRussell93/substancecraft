#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::path::{Component, Path, PathBuf};

use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::wrapper::Parameters,
    model::{Implementation, ServerCapabilities, ServerConfig},
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::Deserialize;
use substancecraft_api::{
    AutomationEdit, ExportJob, TextureFormat, edit_project_file, inspect_primitive_preset,
    list_export_presets, list_primitive_presets, load_project_file_with_report, project_summary,
    run_export_job,
};

/// Environment variable naming the directory project tools may read and
/// write; defaults to the directory the server starts in.
const ROOT_VARIABLE: &str = "SUBSTANCECRAFT_MCP_ROOT";

#[derive(Debug, Deserialize, JsonSchema)]
struct InspectMeshPresetRequest {
    #[schemars(description = "Preset name: Cube, Sphere, Cylinder, or Plane")]
    name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ProjectRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AddFillLayerRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Name of the new layer")]
    name: String,
    #[schemars(
        description = "Optional material: a project material (the layer becomes its linked instance) or a built-in such as Gold or Rubber (copied)"
    )]
    material: Option<String>,
    #[schemars(description = "Optional sRGB base colour [r, g, b], 0-255")]
    color: Option<[u8; 3]>,
    #[schemars(description = "Optional roughness, 0-1")]
    roughness: Option<f32>,
    #[schemars(description = "Optional metallic, 0-1")]
    metallic: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetLayerRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(description = "Optional new name")]
    name: Option<String>,
    #[schemars(description = "Optional visibility")]
    visible: Option<bool>,
    #[schemars(description = "Optional opacity, 0-1")]
    opacity: Option<f32>,
    #[schemars(
        description = "Optional blend mode: Normal, Multiply, Screen, Overlay, Soft light, Darken, Lighten, Difference, Add, Subtract, or (groups only) Pass through"
    )]
    blend_mode: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct LayerRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EditMaterialRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Project material id or exact name")]
    material: String,
    #[schemars(description = "Optional new name")]
    name: Option<String>,
    #[schemars(description = "Optional sRGB base colour [r, g, b], 0-255")]
    color: Option<[u8; 3]>,
    #[schemars(description = "Optional roughness, 0-1")]
    roughness: Option<f32>,
    #[schemars(description = "Optional metallic, 0-1")]
    metallic: Option<f32>,
    #[schemars(description = "Optional height, 0-1 (0.5 is flat)")]
    height: Option<f32>,
    #[schemars(description = "Optional opacity-channel value, 0-1")]
    opacity: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct LinkMaterialRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(description = "Project material id or exact name")]
    material: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ExportTexturesRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Output directory, inside the server's root directory")]
    output: String,
    #[schemars(description = "Optional export preset name; defaults to each set's saved choice")]
    preset: Option<String>,
    #[schemars(description = "Optional file format: png, tga, or exr")]
    format: Option<String>,
}

#[derive(Clone)]
struct SubstanceCraftMcp {
    /// Canonical directory project tools are confined to.
    root: PathBuf,
}

impl SubstanceCraftMcp {
    fn new(root: &Path) -> Result<Self, String> {
        let root = root
            .canonicalize()
            .map_err(|error| format!("root `{}`: {error}", root.display()))?;
        Ok(Self { root })
    }

    /// `requested` resolved against the root. The path (or, for one that
    /// does not exist yet, its nearest existing ancestor) must lie inside
    /// the root, and the rest may not climb out with `..`.
    fn confined(&self, requested: &str) -> Result<PathBuf, String> {
        let path = self.root.join(requested.trim());
        let mut existing = path.as_path();
        let mut rest = Vec::new();
        while !existing.exists() {
            rest.push(existing.file_name().map(PathBuf::from));
            existing = existing
                .parent()
                .ok_or_else(|| format!("`{requested}` is outside the server's root"))?;
        }
        let mut resolved = existing
            .canonicalize()
            .map_err(|error| format!("`{requested}`: {error}"))?;
        for part in rest.into_iter().rev() {
            let part = part.ok_or_else(|| format!("`{requested}` is not a valid path"))?;
            if part
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            {
                return Err(format!("`{requested}` is outside the server's root"));
            }
            resolved.push(part);
        }
        if resolved.starts_with(&self.root) {
            Ok(resolved)
        } else {
            Err(format!("`{requested}` is outside the server's root"))
        }
    }

    /// A confined path to an existing `.sproject` file.
    fn project_path(&self, requested: &str) -> Result<PathBuf, String> {
        let path = self.confined(requested)?;
        let is_project = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("sproject"));
        if !is_project || !path.is_file() {
            return Err(format!("`{requested}` is not an existing .sproject file"));
        }
        Ok(path)
    }

    fn edit(&self, project: &str, edit: AutomationEdit) -> Result<String, String> {
        let path = self.project_path(project)?;
        let outcome = edit_project_file(&path, &edit).map_err(|error| error.to_string())?;
        serde_json::to_string_pretty(&outcome).map_err(|error| error.to_string())
    }
}

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

    #[tool(
        description = "List the built-in texture export presets: each output file's name template, channel layout, and which material channel or mesh map fills each colour channel."
    )]
    fn list_export_presets(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&list_export_presets()).map_err(|error| error.to_string())
    }

    #[tool(
        description = "Summarize a SubstanceCraft project: texture sets with their layer trees (ids, kinds, visibility, blend modes, channels, mask sources, linked materials), project materials with instance counts, smart materials, smart masks, custom export presets, and bitmap resources."
    )]
    fn inspect_project(
        &self,
        Parameters(request): Parameters<ProjectRequest>,
    ) -> Result<String, String> {
        let path = self.project_path(&request.project)?;
        let (project, _) =
            load_project_file_with_report(&path).map_err(|error| error.to_string())?;
        serde_json::to_string_pretty(&project_summary(&project)).map_err(|error| error.to_string())
    }

    #[tool(
        description = "Add a fill layer on top of a texture set and save the project. A project material links the layer as its instance; a built-in material is copied; colour, roughness, and metallic override it."
    )]
    fn add_fill_layer(
        &self,
        Parameters(request): Parameters<AddFillLayerRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::AddFillLayer {
                texture_set: request.texture_set,
                name: request.name,
                material: request.material,
                color: request.color,
                roughness: request.roughness,
                metallic: request.metallic,
            },
        )
    }

    #[tool(
        description = "Change a layer's name, visibility, opacity, or blend mode and save the project."
    )]
    fn set_layer(
        &self,
        Parameters(request): Parameters<SetLayerRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::SetLayer {
                texture_set: request.texture_set,
                layer: request.layer,
                name: request.name,
                visible: request.visible,
                opacity: request.opacity,
                blend_mode: request.blend_mode,
            },
        )
    }

    #[tool(
        description = "Remove a layer (a group with its children) and save the project. Refused when another layer's anchor mask depends on it."
    )]
    fn remove_layer(
        &self,
        Parameters(request): Parameters<LayerRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::RemoveLayer {
                texture_set: request.texture_set,
                layer: request.layer,
            },
        )
    }

    #[tool(
        description = "Change a project material and save the project; every layer linked to it, in every texture set, follows except for the fields it overrides."
    )]
    fn edit_material(
        &self,
        Parameters(request): Parameters<EditMaterialRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::EditMaterial {
                material: request.material,
                name: request.name,
                color: request.color,
                roughness: request.roughness,
                metallic: request.metallic,
                height: request.height,
                opacity: request.opacity,
            },
        )
    }

    #[tool(
        description = "Make a layer an instance of a project material (its fill follows the material) and save the project."
    )]
    fn link_material(
        &self,
        Parameters(request): Parameters<LinkMaterialRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::LinkMaterial {
                texture_set: request.texture_set,
                layer: request.layer,
                material: request.material,
            },
        )
    }

    #[tool(
        description = "Export every texture set of a project headlessly, as the command line does: restore the mesh, bake mesh maps where needed, run preflight, and write the preset's files. Returns the files written."
    )]
    fn export_textures(
        &self,
        Parameters(request): Parameters<ExportTexturesRequest>,
    ) -> Result<String, String> {
        let project = self.project_path(&request.project)?;
        let output = self.confined(&request.output)?;
        let format = match &request.format {
            Some(format) => Some(
                TextureFormat::from_extension(format.trim())
                    .ok_or_else(|| format!("unknown format `{format}`: use png, tga, or exr"))?,
            ),
            None => None,
        };
        let job = ExportJob {
            preset: request.preset,
            format,
            ..ExportJob::new(project, output)
        };
        let written = run_export_job(&job).map_err(|error| error.to_string())?;
        let written: Vec<String> = written
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        serde_json::to_string_pretty(&written).map_err(|error| error.to_string())
    }
}

#[tool_handler]
impl ServerHandler for SubstanceCraftMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "SubstanceCraft headless tools. Inspect built-in meshes and export presets; summarize, edit, and export .sproject files inside the server's root directory (SUBSTANCECRAFT_MCP_ROOT, else the start directory). Edits run through the project's undoable command model and validation, so a refused edit changes nothing; reference texture sets, layers, and materials by id or exact name.",
            )
            .with_server_info(Implementation::new(
                "substancecraft-mcp",
                env!("CARGO_PKG_VERSION"),
            ))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = match std::env::var_os(ROOT_VARIABLE) {
        Some(root) => PathBuf::from(root),
        None => std::env::current_dir()?,
    };
    SubstanceCraftMcp::new(&root)?
        .serve(stdio())
        .await?
        .waiting()
        .await?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{
        LayerId, MaterialChannel, MaterialLayer, MaterialProject, MaterialResource, MeshAsset,
        MeshId, MeshSmoothing, TextureResolution, TextureSet, TextureSetId,
    };

    fn server() -> (tempfile::TempDir, SubstanceCraftMcp) {
        let folder = tempfile::tempdir().unwrap();
        let server = SubstanceCraftMcp::new(folder.path()).unwrap();
        (folder, server)
    }

    #[test]
    fn tc_0011_server_advertises_tool_capability() {
        let (_folder, server) = server();
        let info = server.get_info();
        assert!(info.capabilities.tools.is_some());
        assert_eq!(info.server_info.name, "substancecraft-mcp");
    }

    #[test]
    fn tc_0012_mesh_inspection_tool_returns_serialized_summary() {
        let (_folder, server) = server();
        let result = server
            .inspect_mesh_preset(Parameters(InspectMeshPresetRequest {
                name: "sphere".to_owned(),
            }))
            .unwrap();
        assert!(result.contains("\"name\": \"Sphere\""));
        assert!(result.contains("\"triangle_count\": 224"));
    }

    #[test]
    fn tc_0115_export_preset_tool_lists_built_in_presets_as_json() {
        let (_folder, server) = server();
        let result = server.list_export_presets().unwrap();
        let presets: serde_json::Value = serde_json::from_str(&result).unwrap();
        let names: Vec<&str> = presets
            .as_array()
            .unwrap()
            .iter()
            .map(|preset| preset["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"Unreal Engine"));
        assert!(result.contains("\"file_name\": \"T_$textureSet_N\""));
    }

    #[test]
    fn tc_0282_project_tools_edit_and_export_inside_the_root_only() {
        let (folder, server) = server();
        let mut project = MaterialProject {
            mesh: Some(MeshAsset {
                id: MeshId(1),
                source: "builtin:Cube".to_owned(),
                scale: 1.0,
                smoothing: MeshSmoothing::default(),
                high_poly_source: None,
                content_hash: None,
                high_poly_hash: None,
            }),
            texture_sets: vec![TextureSet {
                id: TextureSetId(1),
                name: "Body".to_owned(),
                material_slot: "Body".to_owned(),
                resolution: TextureResolution::new(16, 16).unwrap(),
                channels: [MaterialChannel::BaseColor, MaterialChannel::Roughness].into(),
                layers: vec![MaterialLayer::new_paint(LayerId(1), "Base")],
                height_strength: 1.0,
                export: None,
            }],
            active_texture_set: Some(TextureSetId(1)),
            ..MaterialProject::empty()
        };
        let gold = MaterialResource::built_in()
            .into_iter()
            .find(|material| material.name == "Gold")
            .unwrap();
        let (_, add) = project.add_material_command(gold).unwrap();
        substancecraft_core::EditHistory::default().apply(&mut project, add);
        let path = folder.path().join("mcp.sproject");
        substancecraft_api::save_project_file(&path, &project).unwrap();

        // Edits by name, through the tools, saved to the file.
        let added = server
            .add_fill_layer(Parameters(AddFillLayerRequest {
                project: "mcp.sproject".to_owned(),
                texture_set: "Body".to_owned(),
                name: "Plating".to_owned(),
                material: Some("Gold".to_owned()),
                color: None,
                roughness: None,
                metallic: None,
            }))
            .unwrap();
        assert!(added.contains("linked to material `Gold`"), "{added}");
        server
            .set_layer(Parameters(SetLayerRequest {
                project: "mcp.sproject".to_owned(),
                texture_set: "1".to_owned(),
                layer: "Base".to_owned(),
                name: None,
                visible: Some(false),
                opacity: None,
                blend_mode: None,
            }))
            .unwrap();
        let edited = server
            .edit_material(Parameters(EditMaterialRequest {
                project: "mcp.sproject".to_owned(),
                material: "Gold".to_owned(),
                name: None,
                color: None,
                roughness: Some(0.5),
                metallic: None,
                height: None,
                opacity: None,
            }))
            .unwrap();
        assert!(edited.contains("1 linked layer(s)"), "{edited}");
        let summary: serde_json::Value = serde_json::from_str(
            &server
                .inspect_project(Parameters(ProjectRequest {
                    project: "mcp.sproject".to_owned(),
                }))
                .unwrap(),
        )
        .unwrap();
        let layers = &summary["texture_sets"][0]["layers"];
        assert_eq!(layers[0]["visible"], false);
        assert_eq!(layers[1]["name"], "Plating");
        assert_eq!(layers[1]["material"], 1);
        assert_eq!(summary["materials"][0]["instances"], 1);

        // A refused edit is an actionable error and leaves the file alone.
        let bytes = std::fs::read(&path).unwrap();
        let error = server
            .remove_layer(Parameters(LayerRequest {
                project: "mcp.sproject".to_owned(),
                texture_set: "Body".to_owned(),
                layer: "Missing".to_owned(),
            }))
            .unwrap_err();
        assert_eq!(error, "no layer `Missing` in that texture set");
        assert_eq!(std::fs::read(&path).unwrap(), bytes);

        // Export writes inside the root.
        let written: Vec<String> = serde_json::from_str(
            &server
                .export_textures(Parameters(ExportTexturesRequest {
                    project: "mcp.sproject".to_owned(),
                    output: "textures/out".to_owned(),
                    preset: None,
                    format: Some("tga".to_owned()),
                }))
                .unwrap(),
        )
        .unwrap();
        assert!(!written.is_empty());
        assert!(written.iter().all(|file| file.ends_with(".tga")));
        assert!(folder.path().join("textures/out").is_dir());

        // Paths outside the root, non-projects, and bad formats are refused.
        let outside = tempfile::tempdir().unwrap();
        let elsewhere = outside.path().join("other.sproject");
        substancecraft_api::save_project_file(&elsewhere, &project).unwrap();
        for requested in [
            elsewhere.display().to_string(),
            "../other.sproject".to_owned(),
            "missing.sproject".to_owned(),
        ] {
            assert!(
                server
                    .inspect_project(Parameters(ProjectRequest {
                        project: requested.clone()
                    }))
                    .is_err(),
                "{requested} must be refused"
            );
        }
        std::fs::write(folder.path().join("notes.txt"), "x").unwrap();
        assert!(
            server
                .inspect_project(Parameters(ProjectRequest {
                    project: "notes.txt".to_owned(),
                }))
                .unwrap_err()
                .contains("not an existing .sproject")
        );
        for output in ["../escape", "textures/../../escape"] {
            assert!(
                server
                    .export_textures(Parameters(ExportTexturesRequest {
                        project: "mcp.sproject".to_owned(),
                        output: output.to_owned(),
                        preset: None,
                        format: None,
                    }))
                    .is_err(),
                "{output} must be refused"
            );
        }
        assert!(
            server
                .export_textures(Parameters(ExportTexturesRequest {
                    project: "mcp.sproject".to_owned(),
                    output: "textures".to_owned(),
                    preset: None,
                    format: Some("bmp".to_owned()),
                }))
                .unwrap_err()
                .contains("unknown format")
        );
    }
}
