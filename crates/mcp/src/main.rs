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
    library_summary, list_export_presets, list_primitive_presets, load_project_file_with_report,
    project_summary, run_export_job,
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
struct AddLayerRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Name of the new layer")]
    name: String,
    #[schemars(description = "Optional kind: paint (default) or group")]
    kind: Option<String>,
    #[schemars(description = "Optional group (id or exact name) to add the layer inside")]
    parent: Option<String>,
    #[schemars(description = "Optional position counted from the bottom; default on top")]
    index: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct MoveLayerRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(
        description = "Optional destination group (id or exact name); none for the top level"
    )]
    parent: Option<String>,
    #[schemars(description = "Optional position counted from the bottom; default on top")]
    index: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetMaskRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(
        description = "Optional mask generator: a name (Edge Wear, Dirt, Orientation, Gradient, Noise, Thickness, ID, Pattern, or none), or an object with a kind and the settings to change, such as {\"kind\": \"EdgeWear\", \"amount\": 0.8}"
    )]
    generator: Option<serde_json::Value>,
    #[schemars(description = "Optional: invert the final mask")]
    invert: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PaintRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name (not a group)")]
    layer: String,
    #[schemars(description = "Stroke points as UV coordinates [u, v]; 0-1 covers tile 1001")]
    points: Vec<[f32; 2]>,
    #[schemars(description = "Brush radius in texels")]
    radius: f32,
    #[schemars(description = "Optional sRGB base colour [r, g, b], 0-255; default white")]
    color: Option<[u8; 3]>,
    #[schemars(description = "Optional stroke opacity, 0-1; default 1")]
    opacity: Option<f32>,
    #[schemars(description = "Optional roughness written where the layer enables it, 0-1")]
    roughness: Option<f32>,
    #[schemars(description = "Optional metallic written where the layer enables it, 0-1")]
    metallic: Option<f32>,
    #[schemars(description = "Optional height written where the layer enables it, 0-1")]
    height: Option<f32>,
    #[schemars(description = "Optional: erase paint along the stroke instead")]
    erase: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PaintMaskRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(description = "Stroke points as UV coordinates [u, v]; 0-1 covers tile 1001")]
    points: Vec<[f32; 2]>,
    #[schemars(description = "Brush radius in texels")]
    radius: f32,
    #[schemars(description = "Mask value painted, 0 (hide) to 1 (show)")]
    value: f32,
    #[schemars(description = "Optional stroke opacity, 0-1; default 1")]
    opacity: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetTextureSetRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Optional new name")]
    name: Option<String>,
    #[schemars(description = "Optional resolution [width, height] in texels, up to 8192")]
    resolution: Option<[u32; 2]>,
    #[schemars(
        description = "Optional enabled channels: Base Color, Id, Roughness, Metallic, Normal, Height, Ambient Occlusion, Emissive, Opacity"
    )]
    channels: Option<Vec<String>>,
    #[schemars(description = "Optional Height-to-normal strength, 0-16")]
    height_strength: Option<f32>,
    #[schemars(
        description = "Optional replacement list of user-defined channels (at most 8); layers enable them as User 0 to User 7"
    )]
    user_channels: Option<Vec<UserChannelRequest>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct UserChannelRequest {
    #[schemars(description = "Slot 0-7; layers and export presets name it User<slot>")]
    slot: u8,
    #[schemars(description = "Unique display name")]
    name: String,
    #[schemars(
        description = "Optional: true for one linear grey value, false (default) for sRGB colour"
    )]
    gray: Option<bool>,
    #[schemars(description = "Optional value where nothing paints, sRGB [r, g, b]; default black")]
    default: Option<[u8; 3]>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetMeshRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Optional uniform import scale, above 0")]
    scale: Option<f32>,
    #[schemars(description = "Optional smoothing angle in degrees, 0-180")]
    smoothing_angle: Option<f32>,
    #[schemars(description = "Optional subdivision levels, 0-3; polygon masks follow")]
    subdivision_levels: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetFiltersRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(
        description = "Optional content filter stack, replacing the current one ([] clears it). Each filter is a name (Blur, Sharpen, Levels, Invert, Hue / Saturation, Brightness / Contrast, Posterize, Threshold, Gradient map) or an object such as {\"kind\": \"Blur\", \"radius\": 8} whose fields replace the defaults"
    )]
    filters: Option<Vec<serde_json::Value>>,
    #[schemars(description = "Optional mask filter stack, in the same form")]
    mask_filters: Option<Vec<serde_json::Value>>,
    #[schemars(
        description = "Optional channels the content filters apply to, such as [\"Roughness\"]; [] means every channel"
    )]
    channels: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetBitmapMaskRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(
        description = "Optional project bitmap resource id or exact name; `none` removes the bitmap mask"
    )]
    resource: Option<String>,
    #[schemars(description = "Optional: invert the bitmap")]
    invert: Option<bool>,
    #[schemars(
        description = "Optional projection settings to change, such as {\"mode\": \"Triplanar\", \"scale\": 2, \"rotation_degrees\": 45}; modes are Uv, Triplanar, Planar, Cylindrical, Spherical, Decal"
    )]
    projection: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetGraphRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Fill layer id or exact name")]
    layer: String,
    #[schemars(
        description = "Material graph as in project files: {\"nodes\": [...], \"outputs\": {\"BaseColor\": {\"Node\": 1}}}; nodes are Generator (a mask generator tagged by kind, e.g. {\"Generator\": {\"kind\": \"Noise\", ...}}), MeshMap, Uv, Bitmap, Blend, Math, Adjust, Ramp, Split, each reading only earlier nodes ({\"Node\": i}) or constants ({\"Value\": [r, g, b]}). Null removes the graph."
    )]
    graph: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetPolygonMaskRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(
        description = "Optional mesh triangle indices the layer is limited to; [] removes the polygon mask"
    )]
    triangles: Option<Vec<u32>>,
    #[schemars(description = "Optional: limit the layer to every other triangle instead")]
    inverted: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetAnchorMaskRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(
        description = "Optional earlier layer (id or exact name) whose composited result the mask reads; `none` removes the anchor"
    )]
    anchor: Option<String>,
    #[schemars(
        description = "Optional channel read: Base Color (as luminance), Roughness, Metallic, Height, Emissive, or Opacity"
    )]
    channel: Option<String>,
    #[schemars(description = "Optional value mapped to 0, 0-1")]
    low: Option<f32>,
    #[schemars(description = "Optional value mapped to 1, 0-1, above low")]
    high: Option<f32>,
    #[schemars(description = "Optional: invert the mask")]
    invert: Option<bool>,
    #[schemars(
        description = "Optional point among the anchor layer's effects: layer (whole layer), fill, paint, or filter N (through its Nth content filter, from 1)"
    )]
    effect: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct InsertSmartMaterialRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(
        description = "Smart material name, from the project or the built-ins (see list_library)"
    )]
    smart_material: String,
    #[schemars(description = "Optional group (id or exact name) to insert into")]
    parent: Option<String>,
    #[schemars(description = "Optional position counted from the bottom; default on top")]
    index: Option<usize>,
    #[schemars(
        description = "Optional exposed parameters by name: a number 0-1 or an sRGB colour [r, g, b]"
    )]
    parameters: Option<std::collections::BTreeMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ApplySmartMaskRequest {
    #[schemars(description = "Path of a .sproject file, inside the server's root directory")]
    project: String,
    #[schemars(description = "Texture set id or exact name")]
    texture_set: String,
    #[schemars(description = "Layer id or exact name")]
    layer: String,
    #[schemars(description = "Smart mask name, from the project or the built-ins")]
    smart_mask: String,
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

    /// Applies an edit given as automation JSON, for edits whose values
    /// (strokes) are built here rather than taken whole from the request.
    fn edit_json(&self, project: &str, edit: serde_json::Value) -> Result<String, String> {
        let edit: AutomationEdit =
            serde_json::from_value(edit).map_err(|error| error.to_string())?;
        self.edit(project, edit)
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
        description = "Add an empty paint layer or group to a texture set, on top or inside a group at a given position, and save the project."
    )]
    fn add_layer(
        &self,
        Parameters(request): Parameters<AddLayerRequest>,
    ) -> Result<String, String> {
        self.edit_json(
            &request.project,
            serde_json::json!({
                "edit": "add_layer",
                "texture_set": request.texture_set,
                "name": request.name,
                "kind": request.kind.as_deref().map_or("paint", str::trim).to_ascii_lowercase(),
                "parent": request.parent,
                "index": request.index,
            }),
        )
    }

    #[tool(
        description = "Move a layer within its texture set, into or out of groups, and save the project. A group cannot move into itself."
    )]
    fn move_layer(
        &self,
        Parameters(request): Parameters<MoveLayerRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::MoveLayer {
                texture_set: request.texture_set,
                layer: request.layer,
                parent: request.parent,
                index: request.index,
            },
        )
    }

    #[tool(
        description = "Set a layer's mask generator (by name, or a kind with the settings to change) and final-mask inversion, and save the project."
    )]
    fn set_mask(&self, Parameters(request): Parameters<SetMaskRequest>) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::SetMask {
                texture_set: request.texture_set,
                layer: request.layer,
                generator: request.generator,
                invert: request.invert,
            },
        )
    }

    #[tool(
        description = "Paint (or erase) a stroke on a layer through UV points with a round brush, writing colour and, where the layer enables them, roughness, metallic, and height; save the project. Undoable in the app like a brush stroke."
    )]
    fn paint_stroke(
        &self,
        Parameters(request): Parameters<PaintRequest>,
    ) -> Result<String, String> {
        let [r, g, b] = request.color.unwrap_or([255, 255, 255]);
        let mut stroke = serde_json::json!({
            "points": request.points,
            "radius_pixels": request.radius,
            "color": [r, g, b, 255],
            "opacity": request.opacity.unwrap_or(1.0),
            "erase": request.erase.unwrap_or(false),
        });
        for (key, value) in [
            ("roughness", request.roughness),
            ("metallic", request.metallic),
            ("height", request.height),
        ] {
            if let Some(value) = value {
                stroke[key] = value.into();
            }
        }
        self.edit_json(
            &request.project,
            serde_json::json!({
                "edit": "paint",
                "texture_set": request.texture_set,
                "layer": request.layer,
                "stroke": stroke,
            }),
        )
    }

    #[tool(
        description = "Paint a grayscale stroke into a layer's mask through UV points (0 hides the layer, 1 shows it) and save the project."
    )]
    fn paint_mask(
        &self,
        Parameters(request): Parameters<PaintMaskRequest>,
    ) -> Result<String, String> {
        self.edit_json(
            &request.project,
            serde_json::json!({
                "edit": "paint_mask",
                "texture_set": request.texture_set,
                "layer": request.layer,
                "stroke": {
                    "points": request.points,
                    "radius_pixels": request.radius,
                    "value": request.value,
                    "opacity": request.opacity.unwrap_or(1.0),
                },
            }),
        )
    }

    #[tool(
        description = "Change a texture set's name, resolution, enabled channels, Height-to-normal strength, or user-defined channels and save the project."
    )]
    fn set_texture_set(
        &self,
        Parameters(request): Parameters<SetTextureSetRequest>,
    ) -> Result<String, String> {
        // Through JSON, so the request needs no core types.
        let user_channels = request
            .user_channels
            .map(|channels| {
                let channels: Vec<_> = channels
                    .into_iter()
                    .map(|user| {
                        serde_json::json!({
                            "slot": user.slot,
                            "name": user.name,
                            "gray": user.gray.unwrap_or(false),
                            "default": user.default.unwrap_or_default(),
                        })
                    })
                    .collect();
                serde_json::from_value(serde_json::Value::Array(channels))
            })
            .transpose()
            .map_err(|error| error.to_string())?;
        self.edit(
            &request.project,
            AutomationEdit::SetTextureSet {
                texture_set: request.texture_set,
                name: request.name,
                resolution: request.resolution,
                channels: request.channels,
                height_strength: request.height_strength,
                user_channels,
            },
        )
    }

    #[tool(
        description = "Change the project mesh's import scale, smoothing angle, or subdivision levels (polygon masks follow a subdivision change) and save the project."
    )]
    fn set_mesh(&self, Parameters(request): Parameters<SetMeshRequest>) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::SetMesh {
                scale: request.scale,
                smoothing_angle: request.smoothing_angle,
                subdivision_levels: request.subdivision_levels,
            },
        )
    }

    #[tool(
        description = "List the smart materials (with their exposed parameters, current values, and evolution controls), smart masks, and materials a project can use: its own first, then the built-ins."
    )]
    fn list_library(
        &self,
        Parameters(request): Parameters<ProjectRequest>,
    ) -> Result<String, String> {
        let path = self.project_path(&request.project)?;
        let (project, _) =
            load_project_file_with_report(&path).map_err(|error| error.to_string())?;
        serde_json::to_string_pretty(&library_summary(&project)).map_err(|error| error.to_string())
    }

    #[tool(
        description = "Replace a layer's content filter stack, mask filter stack, or the channels its content filters apply to, and save the project."
    )]
    fn set_filters(
        &self,
        Parameters(request): Parameters<SetFiltersRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::SetFilters {
                texture_set: request.texture_set,
                layer: request.layer,
                filters: request.filters,
                mask_filters: request.mask_filters,
                channels: request.channels,
            },
        )
    }

    #[tool(
        description = "Set or remove a layer's bitmap mask (a project bitmap through a projection, optionally inverted) and save the project."
    )]
    fn set_bitmap_mask(
        &self,
        Parameters(request): Parameters<SetBitmapMaskRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::SetBitmapMask {
                texture_set: request.texture_set,
                layer: request.layer,
                resource: request.resource,
                invert: request.invert,
                projection: request.projection,
            },
        )
    }

    #[tool(
        description = "Give a fill layer procedural content from a material graph of generators, mesh maps, bitmaps, and adjustments (or remove it) and save the project."
    )]
    fn set_graph(
        &self,
        Parameters(request): Parameters<SetGraphRequest>,
    ) -> Result<String, String> {
        let graph = request
            .graph
            .filter(|graph| !graph.is_null())
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| format!("the graph is not valid: {error}"))?;
        self.edit(
            &request.project,
            AutomationEdit::SetGraph {
                texture_set: request.texture_set,
                layer: request.layer,
                graph,
            },
        )
    }

    #[tool(
        description = "Limit a layer to chosen mesh triangles (or every other triangle), or remove its polygon mask, and save the project."
    )]
    fn set_polygon_mask(
        &self,
        Parameters(request): Parameters<SetPolygonMaskRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::SetPolygonMask {
                texture_set: request.texture_set,
                layer: request.layer,
                triangles: request.triangles,
                inverted: request.inverted,
            },
        )
    }

    #[tool(
        description = "Mask a layer by a channel of the stack as composited through an earlier layer (an anchor point), with a low/high remap and inversion, or remove the anchor; save the project."
    )]
    fn set_anchor_mask(
        &self,
        Parameters(request): Parameters<SetAnchorMaskRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::SetAnchorMask {
                texture_set: request.texture_set,
                layer: request.layer,
                anchor: request.anchor,
                channel: request.channel,
                low: request.low,
                high: request.high,
                invert: request.invert,
                effect: request.effect,
            },
        )
    }

    #[tool(
        description = "Insert a fresh copy of a smart material (a layer tree with fresh ids, keeping its evolution controls) into a texture set, optionally setting its exposed parameters, and save the project."
    )]
    fn insert_smart_material(
        &self,
        Parameters(request): Parameters<InsertSmartMaterialRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::InsertSmartMaterial {
                texture_set: request.texture_set,
                smart_material: request.smart_material,
                parent: request.parent,
                index: request.index,
                parameters: request.parameters.unwrap_or_default(),
            },
        )
    }

    #[tool(
        description = "Apply a smart mask to a layer: replace its generator, bitmap mask, mask filters, and final inversion, keeping painted mask strokes, polygon selections, and ID clipping; save the project."
    )]
    fn apply_smart_mask(
        &self,
        Parameters(request): Parameters<ApplySmartMaskRequest>,
    ) -> Result<String, String> {
        self.edit(
            &request.project,
            AutomationEdit::ApplySmartMask {
                texture_set: request.texture_set,
                layer: request.layer,
                smart_mask: request.smart_mask,
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
    // Plugins (PROD-07) load as for the CLI; stdout carries the protocol, so
    // load errors go to stderr.
    let (plugins, errors) =
        substancecraft_api::load_plugins(&substancecraft_api::default_plugin_folders());
    for error in errors {
        eprintln!("warning: plugin not loaded: {error}");
    }
    substancecraft_api::install_plugins(plugins);
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
                high_to_low: Default::default(),
            }),
            texture_sets: vec![TextureSet {
                user_channels: Default::default(),
                stencils: Default::default(),
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

    #[test]
    fn tc_0310_layer_mask_paint_and_property_tools_edit_project_files() {
        let (folder, server) = server();
        let project = MaterialProject {
            mesh: Some(MeshAsset {
                id: MeshId(1),
                source: "builtin:Cube".to_owned(),
                scale: 1.0,
                smoothing: MeshSmoothing::default(),
                high_poly_source: None,
                content_hash: None,
                high_poly_hash: None,
                high_to_low: Default::default(),
            }),
            texture_sets: vec![TextureSet {
                user_channels: Default::default(),
                stencils: Default::default(),
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
        let path = folder.path().join("tools.sproject");
        substancecraft_api::save_project_file(&path, &project).unwrap();
        let file = || "tools.sproject".to_owned();

        let added = server
            .add_layer(Parameters(AddLayerRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                name: "Wear".to_owned(),
                kind: Some("Group".to_owned()),
                parent: None,
                index: None,
            }))
            .unwrap();
        assert!(added.contains("added group 2 `Wear`"), "{added}");
        server
            .move_layer(Parameters(MoveLayerRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Base".to_owned(),
                parent: Some("Wear".to_owned()),
                index: None,
            }))
            .unwrap();
        let painted = server
            .paint_stroke(Parameters(PaintRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Base".to_owned(),
                points: vec![[0.2, 0.5], [0.8, 0.5]],
                radius: 2.0,
                color: Some([200, 40, 10]),
                opacity: None,
                roughness: Some(0.1),
                metallic: None,
                height: None,
                erase: None,
            }))
            .unwrap();
        assert!(painted.contains("painted 2 point(s)"), "{painted}");
        server
            .paint_mask(Parameters(PaintMaskRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Wear".to_owned(),
                points: vec![[0.5, 0.5]],
                radius: 3.0,
                value: 0.0,
                opacity: None,
            }))
            .unwrap();
        server
            .set_mask(Parameters(SetMaskRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Wear".to_owned(),
                generator: Some(serde_json::json!({"kind": "Dirt", "amount": 0.7})),
                invert: None,
            }))
            .unwrap();
        server
            .set_texture_set(Parameters(SetTextureSetRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                name: None,
                resolution: Some([32, 32]),
                channels: Some(vec!["Base Color".to_owned(), "Height".to_owned()]),
                height_strength: Some(2.0),
                user_channels: None,
            }))
            .unwrap();
        server
            .set_mesh(Parameters(SetMeshRequest {
                project: file(),
                scale: Some(0.5),
                smoothing_angle: None,
                subdivision_levels: Some(1),
            }))
            .unwrap();

        let saved = substancecraft_api::load_project_file(&path).unwrap();
        let set = &saved.texture_sets[0];
        assert_eq!(set.resolution.width, 32);
        assert_eq!(set.height_strength, 2.0);
        let wear = &set.layers[0];
        assert!(wear.is_group());
        assert_eq!(wear.mask_strokes.len(), 1);
        assert!(matches!(
            wear.mask_generator,
            Some(substancecraft_core::MaskGenerator::Dirt(dirt)) if dirt.amount == 0.7
        ));
        let base = &wear.children.as_ref().unwrap()[0];
        assert_eq!(base.name, "Base");
        assert_eq!(base.paint_strokes[0].color, [200, 40, 10, 255]);
        assert_eq!(base.paint_strokes[0].roughness, 0.1);
        let mesh = saved.mesh.as_ref().unwrap();
        assert_eq!((mesh.scale, mesh.smoothing.subdivision_levels), (0.5, 1));

        // Refusals are actionable and leave the file alone.
        let bytes = std::fs::read(&path).unwrap();
        let error = server
            .paint_stroke(Parameters(PaintRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Wear".to_owned(),
                points: vec![[0.5, 0.5]],
                radius: 2.0,
                color: None,
                opacity: None,
                roughness: None,
                metallic: None,
                height: None,
                erase: None,
            }))
            .unwrap_err();
        assert_eq!(error, "groups hold no paint; paint a layer inside");
        let error = server
            .add_layer(Parameters(AddLayerRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                name: "x".to_owned(),
                kind: Some("folder".to_owned()),
                parent: None,
                index: None,
            }))
            .unwrap_err();
        assert!(error.contains("unknown variant `folder`"), "{error}");
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn tc_0326_filter_mask_and_library_tools_edit_project_files() {
        let (folder, server) = server();
        let project = MaterialProject {
            texture_sets: vec![TextureSet {
                user_channels: Default::default(),
                stencils: Default::default(),
                id: TextureSetId(1),
                name: "Body".to_owned(),
                material_slot: "Body".to_owned(),
                resolution: TextureResolution::new(16, 16).unwrap(),
                channels: [MaterialChannel::BaseColor, MaterialChannel::Roughness].into(),
                layers: vec![
                    MaterialLayer::new_paint(LayerId(1), "Base"),
                    MaterialLayer::new_paint(LayerId(2), "Grime"),
                ],
                height_strength: 1.0,
                export: None,
            }],
            resources: vec![substancecraft_core::ProjectResource::new(
                substancecraft_core::ResourceId(1),
                "Stencil",
                "stencil.png",
            )],
            active_texture_set: Some(TextureSetId(1)),
            ..MaterialProject::empty()
        };
        let path = folder.path().join("library.sproject");
        substancecraft_api::save_project_file(&path, &project).unwrap();
        let file = || "library.sproject".to_owned();

        let library: serde_json::Value = serde_json::from_str(
            &server
                .list_library(Parameters(ProjectRequest { project: file() }))
                .unwrap(),
        )
        .unwrap();
        assert!(
            library["smart_materials"]
                .as_array()
                .unwrap()
                .iter()
                .any(|material| material["name"] == "Rusty iron"
                    && material["parameters"]["Rust amount"].is_number())
        );

        server
            .set_filters(Parameters(SetFiltersRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Grime".to_owned(),
                filters: Some(vec![serde_json::json!({"kind": "Blur", "radius": 2.0})]),
                mask_filters: Some(vec![serde_json::json!("Invert")]),
                channels: Some(vec!["Roughness".to_owned()]),
            }))
            .unwrap();
        server
            .set_bitmap_mask(Parameters(SetBitmapMaskRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Base".to_owned(),
                resource: Some("Stencil".to_owned()),
                invert: None,
                projection: Some(serde_json::json!({"mode": "Planar"})),
            }))
            .unwrap();
        server
            .set_polygon_mask(Parameters(SetPolygonMaskRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Base".to_owned(),
                triangles: Some(vec![0, 5]),
                inverted: None,
            }))
            .unwrap();
        server
            .set_anchor_mask(Parameters(SetAnchorMaskRequest {
                effect: Default::default(),
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Grime".to_owned(),
                anchor: Some("Base".to_owned()),
                channel: Some("Base Color".to_owned()),
                low: None,
                high: Some(0.5),
                invert: None,
            }))
            .unwrap();
        let inserted = server
            .insert_smart_material(Parameters(InsertSmartMaterialRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                smart_material: "Rusty iron".to_owned(),
                parent: None,
                index: Some(0),
                parameters: Some([("Rust amount".to_owned(), serde_json::json!(0.9))].into()),
            }))
            .unwrap();
        assert!(inserted.contains("as layer 3"), "{inserted}");
        server
            .apply_smart_mask(Parameters(ApplySmartMaskRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Base".to_owned(),
                smart_mask: "Cavity dirt".to_owned(),
            }))
            .unwrap();

        let saved = substancecraft_api::load_project_file(&path).unwrap();
        let set = &saved.texture_sets[0];
        let grime = set.layer(LayerId(2)).unwrap();
        assert_eq!(
            grime.filters,
            [substancecraft_core::LayerFilter::Blur { radius: 2.0 }]
        );
        assert_eq!(
            grime.mask_filters,
            [substancecraft_core::LayerFilter::Invert]
        );
        assert_eq!(grime.filter_channels, [MaterialChannel::Roughness].into());
        let anchor = grime.mask_anchor.unwrap();
        assert_eq!(
            (anchor.layer, anchor.channel, anchor.high),
            (LayerId(1), MaterialChannel::BaseColor, 0.5)
        );
        let base = set.layer(LayerId(1)).unwrap();
        assert_eq!(base.polygon_mask.as_ref().unwrap().triangles, [0, 5].into());
        assert!(matches!(
            base.mask_generator,
            Some(substancecraft_core::MaskGenerator::Dirt(_))
        ));
        assert_eq!(
            base.mask_bitmap, None,
            "a smart mask without a bitmap clears the layer's"
        );
        let rusty = &set.layers[0];
        assert_eq!(rusty.name, "Rusty iron");
        let rust = set
            .walk_layers()
            .into_iter()
            .find(|layer| layer.name == "Rust")
            .unwrap();
        assert!(matches!(
            rust.mask_generator,
            Some(substancecraft_core::MaskGenerator::Noise(noise)) if noise.amount == 0.9
        ));

        // Refusals are actionable and leave the file alone.
        let bytes = std::fs::read(&path).unwrap();
        let error = server
            .set_anchor_mask(Parameters(SetAnchorMaskRequest {
                effect: Default::default(),
                project: file(),
                texture_set: "Body".to_owned(),
                layer: "Base".to_owned(),
                anchor: Some("Grime".to_owned()),
                channel: None,
                low: None,
                high: None,
                invert: None,
            }))
            .unwrap_err();
        assert!(error.contains("can only anchor"), "{error}");
        let error = server
            .insert_smart_material(Parameters(InsertSmartMaterialRequest {
                project: file(),
                texture_set: "Body".to_owned(),
                smart_material: "Gold leaf".to_owned(),
                parent: None,
                index: None,
                parameters: None,
            }))
            .unwrap_err();
        assert_eq!(
            error,
            "no smart material `Gold leaf` in the project or the built-in library"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
}
