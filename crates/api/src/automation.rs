//! A versioned command API for headless automation (MCP tools and
//! scripts): a summary of a project and a small set of typed edits.
//!
//! Every edit runs through the same [`EditCommand`] history the desktop
//! records, then the whole project is validated. An edit that does not apply,
//! or would leave the project invalid, changes nothing and returns an error
//! that says why.

use std::path::Path;

use serde::{Deserialize, Serialize};
use substancecraft_core::{
    AnchorEffect, AnchorMask, BitmapMask, BlendMode, EditCommand, EditHistory, LayerFill,
    LayerFilter, LayerId, LayerSettings, LayerSlot, MapRequirements, MaskGenerator, MaskStroke,
    MaterialChannel, MaterialId, MaterialLayer, MaterialLink, MaterialProject, MaterialResource,
    PaintStroke, ParameterValue, PolygonMask, SmartMask, SmartMaterial, TextureProjection,
    TextureResolution, TextureSet, TextureSetId, mask_drivers, remap_subdivided_triangles,
};
use thiserror::Error;

use crate::{ProjectFileError, load_project_file_with_report, save_project_file};

/// Version of [`ProjectSummary`] and [`AutomationEdit`]. Additions keep the
/// version; renames and removals bump it.
pub const AUTOMATION_API_VERSION: u32 = 1;

/// The parts of a project automation reads: texture sets with their layer
/// trees, materials, and library entries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectSummary {
    pub api_version: u32,
    pub schema_version: u32,
    /// Source of the mesh (a file path or `builtin:<preset>`).
    pub mesh: Option<String>,
    pub texture_sets: Vec<TextureSetSummary>,
    pub materials: Vec<MaterialSummary>,
    pub smart_materials: Vec<String>,
    pub smart_masks: Vec<String>,
    /// Custom export presets; built-ins are listed by `list_export_presets`.
    pub export_presets: Vec<String>,
    pub resources: Vec<ResourceSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureSetSummary {
    pub id: u64,
    pub name: String,
    pub material_slot: String,
    pub resolution: [u32; 2],
    pub channels: Vec<String>,
    /// Layer tree, bottom first.
    pub layers: Vec<LayerSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerSummary {
    pub id: u64,
    pub name: String,
    /// `paint`, `fill`, or `group`.
    pub kind: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend_mode: String,
    /// Channels that blend differently from `blend_mode`.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub channel_blend_modes: std::collections::BTreeMap<String, String>,
    pub channels: Vec<String>,
    /// The project material this fill is an instance of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<u64>,
    /// What the mask reads (generator, painted mask, polygon selection, ...).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mask: Vec<String>,
    /// Content filters, in the order they apply.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub filters: Vec<String>,
    pub strokes: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<LayerSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialSummary {
    pub id: u64,
    pub name: String,
    pub instances: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceSummary {
    pub id: u64,
    pub name: String,
    pub source: String,
}

fn channel_names(channels: &std::collections::BTreeSet<MaterialChannel>) -> Vec<String> {
    channels
        .iter()
        .map(|channel| format!("{channel:?}"))
        .collect()
}

fn layer_summary(layer: &MaterialLayer, set: &TextureSet) -> LayerSummary {
    LayerSummary {
        id: layer.id.0,
        name: layer.name.clone(),
        kind: if layer.is_group() {
            "group"
        } else if layer.fill.is_some() {
            "fill"
        } else {
            "paint"
        }
        .to_owned(),
        visible: layer.visible,
        opacity: layer.opacity,
        blend_mode: layer.blend_mode.name().to_owned(),
        channel_blend_modes: layer
            .channel_blend_modes
            .iter()
            .map(|(channel, mode)| (format!("{channel:?}"), mode.name().to_owned()))
            .collect(),
        channels: channel_names(&layer.channels),
        material: layer.material_link.as_ref().map(|link| link.material.0),
        mask: mask_drivers(layer, set),
        filters: layer
            .filters
            .iter()
            .map(|filter| filter.name().to_owned())
            .collect(),
        strokes: layer.paint_strokes.len() + layer.mask_strokes.len() + layer.id_strokes.len(),
        children: layer
            .children
            .iter()
            .flatten()
            .map(|child| layer_summary(child, set))
            .collect(),
    }
}

/// A summary of `project` for automation clients.
pub fn project_summary(project: &MaterialProject) -> ProjectSummary {
    ProjectSummary {
        api_version: AUTOMATION_API_VERSION,
        schema_version: project.schema_version,
        mesh: project.mesh.as_ref().map(|mesh| mesh.source.clone()),
        texture_sets: project
            .texture_sets
            .iter()
            .map(|set| TextureSetSummary {
                id: set.id.0,
                name: set.name.clone(),
                material_slot: set.material_slot.clone(),
                resolution: [set.resolution.width, set.resolution.height],
                channels: channel_names(&set.channels),
                layers: set
                    .layers
                    .iter()
                    .map(|layer| layer_summary(layer, set))
                    .collect(),
            })
            .collect(),
        materials: project
            .materials
            .iter()
            .map(|material| MaterialSummary {
                id: material.id.0,
                name: material.name.clone(),
                instances: project.material_instances(material.id).len(),
            })
            .collect(),
        smart_materials: project
            .smart_materials
            .iter()
            .map(|material| material.name.clone())
            .collect(),
        smart_masks: project
            .smart_masks
            .iter()
            .map(|mask| mask.name.clone())
            .collect(),
        export_presets: project
            .export_presets
            .iter()
            .map(|preset| preset.name.clone())
            .collect(),
        resources: project
            .resources
            .iter()
            .map(|resource| ResourceSummary {
                id: resource.id.0,
                name: resource.name.clone(),
                source: resource.source.clone(),
            })
            .collect(),
    }
}

/// Smart materials, smart masks, and materials automation can insert or
/// apply: the project's first, then the built-ins.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LibrarySummary {
    pub api_version: u32,
    pub smart_materials: Vec<SmartMaterialSummary>,
    pub smart_masks: Vec<LibraryEntrySummary>,
    pub materials: Vec<LibraryEntrySummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LibraryEntrySummary {
    pub name: String,
    /// `project` or `built-in`.
    pub origin: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SmartMaterialSummary {
    pub name: String,
    pub origin: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Exposed parameters by name, with their current values (a number or
    /// an sRGB colour).
    pub parameters: std::collections::BTreeMap<String, serde_json::Value>,
    /// Evolution controls (for example `Wear`) carried by its layers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub controls: Vec<String>,
    /// Whether it needs baked mesh maps.
    pub needs_mesh_maps: bool,
}

fn origin(project: bool) -> String {
    if project { "project" } else { "built-in" }.to_owned()
}

/// The library entries automation can use with `project`.
pub fn library_summary(project: &MaterialProject) -> LibrarySummary {
    let smart_materials = project
        .smart_materials
        .iter()
        .cloned()
        .chain(SmartMaterial::built_in())
        .enumerate()
        .map(|(index, material)| SmartMaterialSummary {
            origin: origin(index < project.smart_materials.len()),
            parameters: material
                .parameters
                .iter()
                .enumerate()
                .filter_map(|(index, parameter)| {
                    let value = match material.parameter_value(index)? {
                        ParameterValue::Scalar(value) => serde_json::json!(value),
                        ParameterValue::Color(color) => serde_json::json!(color),
                    };
                    Some((parameter.name.clone(), value))
                })
                .collect(),
            controls: material
                .root
                .walk()
                .iter()
                .flat_map(|layer| layer.evolutions.iter().map(|control| control.name.clone()))
                .collect(),
            needs_mesh_maps: material.requirements().mesh_maps,
            name: material.name,
            description: material.description,
        })
        .collect();
    let smart_masks = project
        .smart_masks
        .iter()
        .cloned()
        .chain(SmartMask::built_in())
        .enumerate()
        .map(|(index, mask)| LibraryEntrySummary {
            name: mask.name,
            origin: origin(index < project.smart_masks.len()),
            description: mask.description,
        })
        .collect();
    let materials = project
        .materials
        .iter()
        .cloned()
        .chain(MaterialResource::built_in())
        .enumerate()
        .map(|(index, material)| LibraryEntrySummary {
            name: material.name,
            origin: origin(index < project.materials.len()),
            description: String::new(),
        })
        .collect();
    LibrarySummary {
        api_version: AUTOMATION_API_VERSION,
        smart_materials,
        smart_masks,
        materials,
    }
}

/// One edit. Texture sets, layers, and materials are referenced by their
/// numeric id or their exact name; a name shared by several is refused.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "edit", rename_all = "snake_case")]
pub enum AutomationEdit {
    /// Adds a fill layer on top of a texture set. With `material`, a project
    /// material links the layer as its instance and a built-in material is
    /// copied in; the values given then override it.
    AddFillLayer {
        texture_set: String,
        name: String,
        #[serde(default)]
        material: Option<String>,
        #[serde(default)]
        color: Option<[u8; 3]>,
        #[serde(default)]
        roughness: Option<f32>,
        #[serde(default)]
        metallic: Option<f32>,
    },
    /// Changes a layer's name, visibility, opacity, or blend mode (by its
    /// display name, e.g. `Multiply` or `Soft light`).
    SetLayer {
        texture_set: String,
        layer: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        visible: Option<bool>,
        #[serde(default)]
        opacity: Option<f32>,
        #[serde(default)]
        blend_mode: Option<String>,
    },
    /// Removes a layer (and, for a group, its children).
    RemoveLayer { texture_set: String, layer: String },
    /// Changes a project material; every linked layer follows.
    EditMaterial {
        material: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        color: Option<[u8; 3]>,
        #[serde(default)]
        roughness: Option<f32>,
        #[serde(default)]
        metallic: Option<f32>,
        #[serde(default)]
        height: Option<f32>,
        #[serde(default)]
        opacity: Option<f32>,
    },
    /// Makes a fill layer an instance of a project material.
    LinkMaterial {
        texture_set: String,
        layer: String,
        material: String,
    },
    /// Adds an empty paint layer or group, inside `parent` (a group) or at
    /// the top level, at `index` counted from the bottom (default: on top).
    AddLayer {
        texture_set: String,
        name: String,
        #[serde(default)]
        kind: NewLayerKind,
        #[serde(default)]
        parent: Option<String>,
        #[serde(default)]
        index: Option<usize>,
    },
    /// Moves a layer into `parent` (a group; none for the top level) at
    /// `index` counted from the bottom (default: on top).
    MoveLayer {
        texture_set: String,
        layer: String,
        #[serde(default)]
        parent: Option<String>,
        #[serde(default)]
        index: Option<usize>,
    },
    /// Sets a layer's mask generator and final inversion. `generator` is a
    /// generator name (`Edge Wear`, `Dirt`, ... or `none` to remove it), or
    /// an object such as `{"kind": "EdgeWear", "amount": 0.8}` whose fields
    /// replace those of the layer's generator of that kind, or its defaults.
    SetMask {
        texture_set: String,
        layer: String,
        #[serde(default)]
        generator: Option<serde_json::Value>,
        #[serde(default)]
        invert: Option<bool>,
    },
    /// Adds a paint (or, with `erase`, erase) stroke to a layer. Points are
    /// UV coordinates; the radius is in texels.
    Paint {
        texture_set: String,
        layer: String,
        stroke: PaintStroke,
    },
    /// Adds a grayscale stroke to a layer's mask.
    PaintMask {
        texture_set: String,
        layer: String,
        stroke: MaskStroke,
    },
    /// Changes a texture set's name, resolution, enabled channels, or
    /// Height-to-normal strength.
    SetTextureSet {
        texture_set: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        resolution: Option<[u32; 2]>,
        #[serde(default)]
        channels: Option<Vec<String>>,
        #[serde(default)]
        height_strength: Option<f32>,
        /// Replaces the set's user channels.
        #[serde(default)]
        user_channels: Option<Vec<substancecraft_core::UserChannel>>,
    },
    /// Changes the mesh's import scale, smoothing angle, or subdivision
    /// levels; polygon masks follow a subdivision change.
    SetMesh {
        #[serde(default)]
        scale: Option<f32>,
        #[serde(default)]
        smoothing_angle: Option<f32>,
        #[serde(default)]
        subdivision_levels: Option<u32>,
    },
    /// Replaces a layer's content filter stack, its mask filter stack, or
    /// the channels its content filters apply to (an empty list: every
    /// channel). Each filter is a name (`Blur`, `Levels`, `Hue / Saturation`,
    /// ...: its defaults) or an object such as `{"kind": "Blur", "radius": 8}`
    /// whose fields replace those defaults. Filters apply in list order.
    SetFilters {
        texture_set: String,
        layer: String,
        #[serde(default)]
        filters: Option<Vec<serde_json::Value>>,
        #[serde(default)]
        mask_filters: Option<Vec<serde_json::Value>>,
        #[serde(default)]
        channels: Option<Vec<String>>,
    },
    /// Sets a layer's bitmap mask to a project bitmap (`resource`, by id or
    /// name; `none` removes the mask), with its inversion and projection.
    /// `projection` is an object whose fields (`mode`, `scale`, `offset`,
    /// `rotation_degrees`, `axis`, `tile_size`, ...) replace those of the
    /// current projection.
    SetBitmapMask {
        texture_set: String,
        layer: String,
        #[serde(default)]
        resource: Option<String>,
        #[serde(default)]
        invert: Option<bool>,
        #[serde(default)]
        projection: Option<serde_json::Value>,
    },
    /// Limits a layer to the given mesh triangles (`triangles`; an empty
    /// list removes the polygon mask), or to every other triangle
    /// (`inverted`).
    SetPolygonMask {
        texture_set: String,
        layer: String,
        #[serde(default)]
        triangles: Option<Vec<u32>>,
        #[serde(default)]
        inverted: Option<bool>,
    },
    /// Gives a fill layer procedural content: a material graph (as saved in
    /// project files), or `null` to remove it.
    SetGraph {
        texture_set: String,
        layer: String,
        graph: Option<substancecraft_core::MaterialGraph>,
    },
    /// Masks a layer by a channel of the stack as composited through an
    /// earlier layer (`anchor`, by id or name; `none` removes the anchor),
    /// remapping `low..high` to `0..1`, optionally inverted.
    SetAnchorMask {
        texture_set: String,
        layer: String,
        #[serde(default)]
        anchor: Option<String>,
        #[serde(default)]
        channel: Option<String>,
        #[serde(default)]
        low: Option<f32>,
        #[serde(default)]
        high: Option<f32>,
        #[serde(default)]
        invert: Option<bool>,
        /// Where among the anchor layer's effects to read: `layer` (the
        /// whole layer), `fill`, `paint`, or `filter N` (through its Nth
        /// content filter, from 1). Choosing another anchor layer resets it.
        #[serde(default)]
        effect: Option<String>,
    },
    /// Inserts a fresh copy of a smart material (project first, then the
    /// built-ins) inside `parent` or at the top level, at `index` from the
    /// bottom (default: on top). `parameters` sets exposed parameters by
    /// name: a number from 0 to 1, or an sRGB colour `[r, g, b]`.
    InsertSmartMaterial {
        texture_set: String,
        smart_material: String,
        #[serde(default)]
        parent: Option<String>,
        #[serde(default)]
        index: Option<usize>,
        #[serde(default)]
        parameters: std::collections::BTreeMap<String, serde_json::Value>,
    },
    /// Replaces a layer's generator, bitmap mask, mask filters, and final
    /// inversion with a smart mask's (project first, then the built-ins).
    /// Painted mask strokes, polygon selections, and ID clipping stay.
    ApplySmartMask {
        texture_set: String,
        layer: String,
        smart_mask: String,
    },
}

/// The kind of layer [`AutomationEdit::AddLayer`] creates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NewLayerKind {
    #[default]
    Paint,
    Group,
}

/// Every channel, for parsing channel names.
const ALL_CHANNELS: [MaterialChannel; 9] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Id,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Normal,
    MaterialChannel::Height,
    MaterialChannel::AmbientOcclusion,
    MaterialChannel::Emissive,
    MaterialChannel::Opacity,
];

/// What an edit did.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditOutcome {
    /// One line describing the change, or that nothing changed.
    pub summary: String,
    /// The layer the edit created or changed, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u64>,
    /// False when the edit asked for what the project already had.
    pub changed: bool,
}

#[derive(Debug, Error, PartialEq)]
pub enum AutomationError {
    #[error("no texture set `{0}`")]
    UnknownTextureSet(String),
    #[error("no layer `{0}` in that texture set")]
    UnknownLayer(String),
    #[error("no material `{0}` in the project or the built-in library")]
    UnknownMaterial(String),
    #[error("no bitmap resource `{0}` in the project")]
    UnknownResource(String),
    #[error("no smart material `{0}` in the project or the built-in library")]
    UnknownSmartMaterial(String),
    #[error("no smart mask `{0}` in the project or the built-in library")]
    UnknownSmartMask(String),
    #[error("`{0}` names several entries; use the numeric id")]
    Ambiguous(String),
    #[error("{0}")]
    InvalidValue(String),
    #[error("the edit was refused: {0}")]
    Rejected(String),
    #[error(transparent)]
    Project(#[from] ProjectFileError),
}

/// The one entry whose id (as text) or exact name is `reference`.
fn find_unique<'a, T>(
    items: impl Iterator<Item = &'a T>,
    reference: &str,
    id_of: impl Fn(&T) -> u64,
    name_of: impl Fn(&T) -> &str,
    missing: impl FnOnce(String) -> AutomationError,
) -> Result<&'a T, AutomationError>
where
    T: 'a,
{
    let items: Vec<&T> = items.collect();
    let reference = reference.trim();
    if let Ok(id) = reference.parse::<u64>()
        && let Some(item) = items.iter().find(|item| id_of(item) == id)
    {
        return Ok(item);
    }
    let mut named = items.into_iter().filter(|item| name_of(item) == reference);
    match (named.next(), named.next()) {
        (Some(item), None) => Ok(item),
        (Some(_), Some(_)) => Err(AutomationError::Ambiguous(reference.to_owned())),
        _ => Err(missing(reference.to_owned())),
    }
}

fn find_set<'a>(
    project: &'a MaterialProject,
    reference: &str,
) -> Result<&'a TextureSet, AutomationError> {
    find_unique(
        project.texture_sets.iter(),
        reference,
        |set| set.id.0,
        |set| set.name.as_str(),
        AutomationError::UnknownTextureSet,
    )
}

fn find_layer<'a>(
    set: &'a TextureSet,
    reference: &str,
) -> Result<&'a MaterialLayer, AutomationError> {
    find_unique(
        set.walk_layers().into_iter(),
        reference,
        |layer| layer.id.0,
        |layer| layer.name.as_str(),
        AutomationError::UnknownLayer,
    )
}

fn find_project_material<'a>(
    project: &'a MaterialProject,
    reference: &str,
) -> Result<&'a MaterialResource, AutomationError> {
    find_unique(
        project.materials.iter(),
        reference,
        |material| material.id.0,
        |material| material.name.as_str(),
        AutomationError::UnknownMaterial,
    )
}

fn unit(name: &str, value: Option<f32>) -> Result<Option<f32>, AutomationError> {
    match value {
        Some(value) if !value.is_finite() || !(0.0..=1.0).contains(&value) => Err(
            AutomationError::InvalidValue(format!("{name} must be between 0 and 1, not {value}")),
        ),
        other => Ok(other),
    }
}

fn blend_mode(name: &str, group: bool) -> Result<BlendMode, AutomationError> {
    BlendMode::ALL
        .into_iter()
        .chain(group.then_some(BlendMode::PassThrough))
        .find(|mode| mode.name().eq_ignore_ascii_case(name.trim()))
        .ok_or_else(|| AutomationError::InvalidValue(format!("unknown blend mode `{name}`")))
}

/// An id no layer of the project uses.
fn next_layer_id(project: &MaterialProject) -> LayerId {
    LayerId(
        project
            .texture_sets
            .iter()
            .flat_map(|set| set.walk_layers())
            .map(|layer| layer.id.0)
            .max()
            .unwrap_or(0)
            + 1,
    )
}

/// Applies `edit` to `project` as one undoable command and validates the
/// result; on any error the project is left as it was.
pub fn apply_automation_edit(
    project: &mut MaterialProject,
    edit: &AutomationEdit,
) -> Result<EditOutcome, AutomationError> {
    let (command, outcome) = automation_command(project, edit)?;
    let Some(command) = command else {
        return Ok(outcome);
    };
    let mut history = EditHistory::default();
    if !history.apply(project, command) {
        return Err(AutomationError::Rejected(
            "it no longer matches the project".to_owned(),
        ));
    }
    if let Err(error) = project.validate() {
        history.undo(project);
        return Err(AutomationError::Rejected(error.to_string()));
    }
    Ok(outcome)
}

/// Loads the project at `path`, applies `edit`, and saves it back atomically
/// when it changed.
pub fn edit_project_file(
    path: &Path,
    edit: &AutomationEdit,
) -> Result<EditOutcome, AutomationError> {
    let (mut project, _) = load_project_file_with_report(path)?;
    let outcome = apply_automation_edit(&mut project, edit)?;
    if outcome.changed {
        save_project_file(path, &project)?;
    }
    Ok(outcome)
}

fn unchanged(layer: Option<LayerId>) -> EditOutcome {
    EditOutcome {
        summary: "nothing to change".to_owned(),
        layer: layer.map(|id| id.0),
        changed: false,
    }
}

/// The command `edit` makes (or `None` when nothing changes) and its
/// outcome.
fn automation_command(
    project: &MaterialProject,
    edit: &AutomationEdit,
) -> Result<(Option<EditCommand>, EditOutcome), AutomationError> {
    match edit {
        AutomationEdit::AddFillLayer {
            texture_set,
            name,
            material,
            color,
            roughness,
            metallic,
        } => {
            let set = find_set(project, texture_set)?;
            if name.trim().is_empty() {
                return Err(AutomationError::InvalidValue(
                    "a layer needs a name".to_owned(),
                ));
            }
            let (roughness, metallic) =
                (unit("roughness", *roughness)?, unit("metallic", *metallic)?);
            let id = next_layer_id(project);
            let mut layer = MaterialLayer::new_fill(id, name.trim(), LayerFill::default());
            let mut source = String::new();
            if let Some(reference) = material {
                match find_project_material(project, reference) {
                    Ok(master) => {
                        layer.apply_settings(
                            MaterialLink::new(master.id).resolve(master, &layer.settings()),
                        );
                        source = format!(" linked to material `{}`", master.name);
                    }
                    Err(AutomationError::UnknownMaterial(_)) => {
                        let built_in = MaterialResource::built_in()
                            .into_iter()
                            .find(|candidate| candidate.name == reference.trim())
                            .ok_or_else(|| AutomationError::UnknownMaterial(reference.clone()))?;
                        built_in.apply_to(&mut layer);
                        source = format!(" from built-in `{}`", built_in.name);
                    }
                    Err(error) => return Err(error),
                }
            }
            if let Some(fill) = &mut layer.fill {
                if let Some([r, g, b]) = color {
                    fill.color = [*r, *g, *b, fill.color[3]];
                }
                if let Some(roughness) = roughness {
                    fill.roughness = roughness;
                }
                if let Some(metallic) = metallic {
                    fill.metallic = metallic;
                }
            }
            // Values given for an instance become explicit overrides.
            layer.apply_settings(project.track_instance_edit(layer.settings()));
            let command = EditCommand::InsertLayer {
                texture_set_id: set.id,
                slot: LayerSlot {
                    parent: None,
                    index: set.layers.len(),
                },
                layer: Box::new(layer),
            };
            Ok((
                Some(command),
                EditOutcome {
                    summary: format!(
                        "added fill layer {} `{}`{source} to `{}`",
                        id.0,
                        name.trim(),
                        set.name
                    ),
                    layer: Some(id.0),
                    changed: true,
                },
            ))
        }
        AutomationEdit::SetLayer {
            texture_set,
            layer,
            name,
            visible,
            opacity,
            blend_mode: mode,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            let before = target.settings();
            let mut after = before.clone();
            if let Some(name) = name {
                if name.trim().is_empty() {
                    return Err(AutomationError::InvalidValue(
                        "a layer needs a name".to_owned(),
                    ));
                }
                after.name = name.trim().to_owned();
            }
            if let Some(visible) = visible {
                after.visible = *visible;
            }
            if let Some(opacity) = unit("opacity", *opacity)? {
                after.opacity = opacity;
            }
            if let Some(mode) = mode {
                after.blend_mode = blend_mode(mode, target.is_group())?;
            }
            if after == before {
                return Ok((None, unchanged(Some(target.id))));
            }
            Ok((
                Some(EditCommand::LayerSettings {
                    texture_set_id: set.id,
                    layer_id: target.id,
                    before: Box::new(before),
                    after: Box::new(after),
                }),
                EditOutcome {
                    summary: format!("updated layer {} `{}`", target.id.0, target.name),
                    layer: Some(target.id.0),
                    changed: true,
                },
            ))
        }
        AutomationEdit::RemoveLayer { texture_set, layer } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            let slot = set
                .slot_of(target.id)
                .ok_or_else(|| AutomationError::UnknownLayer(layer.clone()))?;
            Ok((
                Some(EditCommand::RemoveLayer {
                    texture_set_id: set.id,
                    slot,
                    layer: Box::new(target.clone()),
                }),
                EditOutcome {
                    summary: format!("removed layer {} `{}`", target.id.0, target.name),
                    layer: Some(target.id.0),
                    changed: true,
                },
            ))
        }
        AutomationEdit::EditMaterial {
            material,
            name,
            color,
            roughness,
            metallic,
            height,
            opacity,
        } => {
            let master = find_project_material(project, material)?;
            let mut edited = master.clone();
            if let Some(name) = name {
                edited.name = name.trim().to_owned();
            }
            if let Some([r, g, b]) = color {
                edited.fill.color = [*r, *g, *b, edited.fill.color[3]];
            }
            for (label, value, slot) in [
                ("roughness", *roughness, &mut edited.fill.roughness),
                ("metallic", *metallic, &mut edited.fill.metallic),
                ("height", *height, &mut edited.fill.height),
                ("opacity", *opacity, &mut edited.fill.material_opacity),
            ] {
                if let Some(value) = unit(label, value)? {
                    *slot = value;
                }
            }
            if edited == *master {
                return Ok((None, unchanged(None)));
            }
            edited
                .validate()
                .map_err(|error| AutomationError::InvalidValue(error.to_string()))?;
            let instances = project.material_instances(master.id).len();
            let command = project
                .edit_material_command(edited)
                .ok_or_else(|| AutomationError::Rejected("invalid material".to_owned()))?;
            Ok((
                Some(command),
                EditOutcome {
                    summary: format!(
                        "updated material {} `{}` and {instances} linked layer(s)",
                        master.id.0, master.name
                    ),
                    layer: None,
                    changed: true,
                },
            ))
        }
        AutomationEdit::LinkMaterial {
            texture_set,
            layer,
            material,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            if target.is_group() {
                return Err(AutomationError::InvalidValue(
                    "groups cannot be material instances".to_owned(),
                ));
            }
            let master = find_project_material(project, material)?;
            let master_id: MaterialId = master.id;
            match project.link_material_command(set.id, target.id, master_id) {
                Some(command) => Ok((
                    Some(command),
                    EditOutcome {
                        summary: format!(
                            "linked layer {} `{}` to material `{}`",
                            target.id.0, target.name, master.name
                        ),
                        layer: Some(target.id.0),
                        changed: true,
                    },
                )),
                None => Ok((None, unchanged(Some(target.id)))),
            }
        }
        AutomationEdit::AddLayer {
            texture_set,
            name,
            kind,
            parent,
            index,
        } => {
            let set = find_set(project, texture_set)?;
            if name.trim().is_empty() {
                return Err(AutomationError::InvalidValue(
                    "a layer needs a name".to_owned(),
                ));
            }
            let parent = parent
                .as_deref()
                .map(|parent| find_group(set, parent))
                .transpose()?;
            let index = slot_index(*index, siblings(set, parent).len())?;
            let id = next_layer_id(project);
            let (layer, noun) = match kind {
                NewLayerKind::Paint => (MaterialLayer::new_paint(id, name.trim()), "paint layer"),
                NewLayerKind::Group => (MaterialLayer::new_group(id, name.trim()), "group"),
            };
            Ok((
                Some(EditCommand::InsertLayer {
                    texture_set_id: set.id,
                    slot: LayerSlot {
                        parent: parent.map(|group| group.id),
                        index,
                    },
                    layer: Box::new(layer),
                }),
                EditOutcome {
                    summary: format!(
                        "added {noun} {} `{}` to `{}`{}",
                        id.0,
                        name.trim(),
                        set.name,
                        in_group(parent)
                    ),
                    layer: Some(id.0),
                    changed: true,
                },
            ))
        }
        AutomationEdit::MoveLayer {
            texture_set,
            layer,
            parent,
            index,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            let from = set
                .slot_of(target.id)
                .ok_or_else(|| AutomationError::UnknownLayer(layer.clone()))?;
            let parent = parent
                .as_deref()
                .map(|parent| find_group(set, parent))
                .transpose()?;
            if let Some(group) = parent
                && target.walk().iter().any(|layer| layer.id == group.id)
            {
                return Err(AutomationError::InvalidValue(
                    "a group cannot move into itself".to_owned(),
                ));
            }
            let parent_id = parent.map(|group| group.id);
            // The index counts the destination's layers after the removal.
            let mut count = siblings(set, parent).len();
            if from.parent == parent_id {
                count -= 1;
            }
            let to = LayerSlot {
                parent: parent_id,
                index: slot_index(*index, count)?,
            };
            if to == from {
                return Ok((None, unchanged(Some(target.id))));
            }
            Ok((
                Some(EditCommand::MoveLayer {
                    texture_set_id: set.id,
                    from,
                    to,
                }),
                EditOutcome {
                    summary: format!(
                        "moved layer {} `{}` to position {}{}",
                        target.id.0,
                        target.name,
                        to.index,
                        in_group(parent)
                    ),
                    layer: Some(target.id.0),
                    changed: true,
                },
            ))
        }
        AutomationEdit::SetMask {
            texture_set,
            layer,
            generator,
            invert,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            let before = target.settings();
            let mut after = before.clone();
            if let Some(generator) = generator {
                after.mask_generator = parse_generator(generator, before.mask_generator)?;
            }
            if let Some(invert) = invert {
                after.invert_final_mask = *invert;
            }
            if after == before {
                return Ok((None, unchanged(Some(target.id))));
            }
            let generator = after
                .mask_generator
                .as_ref()
                .map_or("no generator", MaskGenerator::name);
            Ok((
                Some(EditCommand::LayerSettings {
                    texture_set_id: set.id,
                    layer_id: target.id,
                    before: Box::new(before),
                    after: Box::new(after),
                }),
                EditOutcome {
                    summary: format!(
                        "set the mask of layer {} `{}` ({generator})",
                        target.id.0, target.name
                    ),
                    layer: Some(target.id.0),
                    changed: true,
                },
            ))
        }
        AutomationEdit::Paint {
            texture_set,
            layer,
            stroke,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            if target.is_group() {
                return Err(AutomationError::InvalidValue(
                    "groups hold no paint; paint a layer inside".to_owned(),
                ));
            }
            Ok((
                Some(EditCommand::Paint {
                    texture_set_id: set.id,
                    layer_id: target.id,
                    stroke: stroke.clone(),
                }),
                EditOutcome {
                    summary: format!(
                        "{} {} point(s) on layer {} `{}`",
                        if stroke.erase { "erased" } else { "painted" },
                        stroke.points.len(),
                        target.id.0,
                        target.name
                    ),
                    layer: Some(target.id.0),
                    changed: true,
                },
            ))
        }
        AutomationEdit::PaintMask {
            texture_set,
            layer,
            stroke,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            Ok((
                Some(EditCommand::Mask {
                    texture_set_id: set.id,
                    layer_id: target.id,
                    stroke: stroke.clone(),
                }),
                EditOutcome {
                    summary: format!(
                        "painted {} point(s) on the mask of layer {} `{}`",
                        stroke.points.len(),
                        target.id.0,
                        target.name
                    ),
                    layer: Some(target.id.0),
                    changed: true,
                },
            ))
        }
        AutomationEdit::SetTextureSet {
            texture_set,
            name,
            resolution,
            channels,
            height_strength,
            user_channels,
        } => {
            let set = find_set(project, texture_set)?;
            let before = set.properties();
            let mut after = before.clone();
            if let Some(name) = name {
                if name.trim().is_empty() {
                    return Err(AutomationError::InvalidValue(
                        "a texture set needs a name".to_owned(),
                    ));
                }
                after.name = name.trim().to_owned();
            }
            if let Some([width, height]) = resolution {
                after.resolution = TextureResolution::new(*width, *height)
                    .map_err(|error| AutomationError::InvalidValue(error.to_string()))?;
            }
            if let Some(channels) = channels {
                after.channels = channels
                    .iter()
                    .map(|name| parse_channel(name))
                    .collect::<Result<_, _>>()?;
                if after.channels.is_empty() {
                    return Err(AutomationError::InvalidValue(
                        "a texture set needs at least one channel".to_owned(),
                    ));
                }
            }
            if let Some(strength) = height_strength {
                if !strength.is_finite()
                    || !(0.0..=substancecraft_core::MAX_HEIGHT_STRENGTH).contains(strength)
                {
                    return Err(AutomationError::InvalidValue(format!(
                        "height strength must be between 0 and {}, not {strength}",
                        substancecraft_core::MAX_HEIGHT_STRENGTH
                    )));
                }
                after.height_strength = *strength;
            }
            if let Some(user_channels) = user_channels {
                if !substancecraft_core::user_channels_valid(user_channels) {
                    return Err(AutomationError::InvalidValue(format!(
                        "user channels need slots 0 to {} and unique, non-empty names",
                        substancecraft_core::MAX_USER_CHANNELS - 1
                    )));
                }
                after.user_channels.clone_from(user_channels);
                after.user_channels.sort_by_key(|user| user.slot);
            }
            if after == before {
                return Ok((None, unchanged(None)));
            }
            Ok((
                Some(EditCommand::TextureSetProperties {
                    texture_set_id: set.id,
                    before: Box::new(before),
                    after: Box::new(after),
                }),
                EditOutcome {
                    summary: format!("updated texture set {} `{}`", set.id.0, set.name),
                    layer: None,
                    changed: true,
                },
            ))
        }
        AutomationEdit::SetMesh {
            scale,
            smoothing_angle,
            subdivision_levels,
        } => {
            let mesh = project.mesh.as_ref().ok_or_else(|| {
                AutomationError::InvalidValue("the project has no mesh".to_owned())
            })?;
            let mut after = mesh.clone();
            if let Some(scale) = scale {
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err(AutomationError::InvalidValue(format!(
                        "scale must be above 0, not {scale}"
                    )));
                }
                after.scale = *scale;
            }
            if let Some(angle) = smoothing_angle {
                if !angle.is_finite() || !(0.0..=180.0).contains(angle) {
                    return Err(AutomationError::InvalidValue(format!(
                        "smoothing angle must be between 0 and 180 degrees, not {angle}"
                    )));
                }
                after.smoothing.angle_degrees = *angle;
            }
            if let Some(levels) = subdivision_levels {
                if *levels > substancecraft_core::MAX_SUBDIVISION_LEVELS {
                    return Err(AutomationError::InvalidValue(format!(
                        "at most {} subdivision levels, not {levels}",
                        substancecraft_core::MAX_SUBDIVISION_LEVELS
                    )));
                }
                after.smoothing.subdivision_levels = *levels;
            }
            if after == *mesh {
                return Ok((None, unchanged(None)));
            }
            // Polygon masks index triangles of the subdivided mesh, so they
            // follow a change of level, as in the desktop app.
            let (from, to) = (
                mesh.smoothing.subdivision_levels,
                after.smoothing.subdivision_levels,
            );
            let mut commands = vec![EditCommand::Mesh {
                before: Some(mesh.clone()),
                after: Some(after.clone()),
            }];
            if from != to {
                for set in &project.texture_sets {
                    for layer in set.walk_layers() {
                        if layer.polygon_mask.is_none() {
                            continue;
                        }
                        let before = layer.settings();
                        let mut remapped = before.clone();
                        if let Some(mask) = remapped.polygon_mask.as_mut() {
                            mask.triangles = remap_subdivided_triangles(
                                mask.triangles.iter().copied(),
                                from,
                                to,
                            )
                            .into_iter()
                            .collect();
                        }
                        commands.push(EditCommand::LayerSettings {
                            texture_set_id: set.id,
                            layer_id: layer.id,
                            before: Box::new(before),
                            after: Box::new(remapped),
                        });
                    }
                }
            }
            let masks = commands.len() - 1;
            Ok((
                Some(if masks == 0 {
                    commands.remove(0)
                } else {
                    EditCommand::Batch(commands)
                }),
                EditOutcome {
                    summary: format!(
                        "updated the mesh: scale {}, smoothing {}°, {} subdivision level(s){}",
                        after.scale,
                        after.smoothing.angle_degrees,
                        after.smoothing.subdivision_levels,
                        if masks == 0 {
                            String::new()
                        } else {
                            format!("; remapped {masks} polygon mask(s)")
                        }
                    ),
                    layer: None,
                    changed: true,
                },
            ))
        }
        AutomationEdit::SetFilters {
            texture_set,
            layer,
            filters,
            mask_filters,
            channels,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            let before = target.settings();
            let mut after = before.clone();
            if let Some(filters) = filters {
                after.filters = parse_filters(filters)?;
            }
            if let Some(filters) = mask_filters {
                after.mask_filters = parse_filters(filters)?;
            }
            if let Some(channels) = channels {
                after.filter_channels = channels
                    .iter()
                    .map(|name| parse_channel(name))
                    .collect::<Result<_, _>>()?;
            }
            let summary = format!(
                "set the filters of layer {} `{}` ({} on content, {} on mask)",
                target.id.0,
                target.name,
                after.filters.len(),
                after.mask_filters.len()
            );
            Ok(settings_edit(set, target, before, after, summary))
        }
        AutomationEdit::SetBitmapMask {
            texture_set,
            layer,
            resource,
            invert,
            projection,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            let before = target.settings();
            let mut after = before.clone();
            let removing = resource
                .as_deref()
                .is_some_and(|name| name.trim().is_empty() || name.trim() == "none");
            if removing {
                after.mask_bitmap = None;
            } else {
                let mut mask = match (resource, before.mask_bitmap) {
                    (Some(reference), current) => {
                        let resource = find_unique(
                            project.resources.iter(),
                            reference,
                            |resource| resource.id.0,
                            |resource| resource.name.as_str(),
                            AutomationError::UnknownResource,
                        )?;
                        BitmapMask {
                            resource: resource.id,
                            ..current.unwrap_or(BitmapMask {
                                resource: resource.id,
                                projection: TextureProjection::default(),
                                invert: false,
                            })
                        }
                    }
                    (None, Some(current)) => current,
                    (None, None) => {
                        return Err(AutomationError::InvalidValue(
                            "the layer has no bitmap mask; name a `resource`".to_owned(),
                        ));
                    }
                };
                if let Some(invert) = invert {
                    mask.invert = *invert;
                }
                if let Some(fields) = projection {
                    mask.projection = merge_fields("projection", &mask.projection, fields)?;
                }
                after.mask_bitmap = Some(mask);
            }
            let summary = format!(
                "{} the bitmap mask of layer {} `{}`",
                if removing { "removed" } else { "set" },
                target.id.0,
                target.name
            );
            Ok(settings_edit(set, target, before, after, summary))
        }
        AutomationEdit::SetGraph {
            texture_set,
            layer,
            graph,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            if graph.is_some() && (target.is_group() || target.fill.is_none()) {
                return Err(AutomationError::InvalidValue(
                    "only fill layers can be procedural; add a fill first".to_owned(),
                ));
            }
            if graph.as_ref().is_some_and(|graph| !graph.is_valid()) {
                return Err(AutomationError::InvalidValue(format!(
                    "the graph is invalid: inputs must read earlier nodes, at most {} nodes,                      ramps 2 to {} ordered stops, per-texel filters only, and no ID or                      occlusion outputs",
                    substancecraft_core::MAX_GRAPH_NODES,
                    substancecraft_core::MAX_RAMP_STOPS
                )));
            }
            let before = target.settings();
            let after = LayerSettings {
                graph: graph.clone(),
                ..before.clone()
            };
            let summary = match graph {
                Some(graph) => format!(
                    "made layer {} `{}` procedural ({} nodes)",
                    target.id.0,
                    target.name,
                    graph.nodes.len()
                ),
                None => format!(
                    "removed the graph of layer {} `{}`",
                    target.id.0, target.name
                ),
            };
            Ok(settings_edit(set, target, before, after, summary))
        }
        AutomationEdit::SetPolygonMask {
            texture_set,
            layer,
            triangles,
            inverted,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            let before = target.settings();
            let mut after = before.clone();
            match triangles {
                Some(triangles) if triangles.is_empty() => after.polygon_mask = None,
                Some(triangles) => {
                    after.polygon_mask = Some(PolygonMask {
                        triangles: triangles.iter().copied().collect(),
                        inverted: before
                            .polygon_mask
                            .as_ref()
                            .is_some_and(|mask| mask.inverted),
                    });
                }
                None => {}
            }
            if let Some(inverted) = inverted {
                match after.polygon_mask.as_mut() {
                    Some(mask) => mask.inverted = *inverted,
                    None if triangles.is_some() => {}
                    None => {
                        return Err(AutomationError::InvalidValue(
                            "the layer has no polygon mask; give its `triangles`".to_owned(),
                        ));
                    }
                }
            }
            let summary = match &after.polygon_mask {
                Some(mask) => format!(
                    "limited layer {} `{}` to {}{} triangle(s)",
                    target.id.0,
                    target.name,
                    if mask.inverted { "all but " } else { "" },
                    mask.triangles.len()
                ),
                None => format!(
                    "removed the polygon mask of layer {} `{}`",
                    target.id.0, target.name
                ),
            };
            Ok(settings_edit(set, target, before, after, summary))
        }
        AutomationEdit::SetAnchorMask {
            texture_set,
            layer,
            anchor,
            channel,
            low,
            high,
            invert,
            effect,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            let before = target.settings();
            let mut after = before.clone();
            let removing = anchor
                .as_deref()
                .is_some_and(|name| name.trim().is_empty() || name.trim() == "none");
            if removing {
                after.mask_anchor = None;
            } else {
                let mut mask = match (anchor, before.mask_anchor) {
                    (Some(reference), current) => {
                        let source = find_layer(set, reference)?;
                        if !set.anchor_is_valid(target.id, source.id) {
                            return Err(AutomationError::InvalidValue(format!(
                                "layer `{}` can only anchor to a layer composited before it, \
                                 outside it, and not to a group holding it",
                                target.name
                            )));
                        }
                        match current {
                            Some(current) if current.layer == source.id => current,
                            Some(current) => AnchorMask {
                                layer: source.id,
                                effect: None,
                                ..current
                            },
                            None => AnchorMask::new(source.id, MaterialChannel::BaseColor),
                        }
                    }
                    (None, Some(current)) => current,
                    (None, None) => {
                        return Err(AutomationError::InvalidValue(
                            "the layer has no anchor mask; name an `anchor` layer".to_owned(),
                        ));
                    }
                };
                if let Some(channel) = channel {
                    mask.channel = parse_channel(channel)?;
                    if !AnchorMask::reads(mask.channel) {
                        return Err(AutomationError::InvalidValue(format!(
                            "anchors cannot read the {:?} channel",
                            mask.channel
                        )));
                    }
                }
                if let Some(low) = unit("low", *low)? {
                    mask.low = low;
                }
                if let Some(high) = unit("high", *high)? {
                    mask.high = high;
                }
                if mask.low >= mask.high {
                    return Err(AutomationError::InvalidValue(format!(
                        "low ({}) must be below high ({})",
                        mask.low, mask.high
                    )));
                }
                if let Some(invert) = invert {
                    mask.invert = *invert;
                }
                if let Some(effect) = effect {
                    mask.effect = parse_anchor_effect(effect)?;
                }
                after.mask_anchor = Some(mask);
            }
            let summary = match after.mask_anchor {
                Some(mask) => format!(
                    "anchored layer {} `{}` to layer {} ({:?})",
                    target.id.0, target.name, mask.layer.0, mask.channel
                ),
                None => format!(
                    "removed the anchor mask of layer {} `{}`",
                    target.id.0, target.name
                ),
            };
            Ok(settings_edit(set, target, before, after, summary))
        }
        AutomationEdit::InsertSmartMaterial {
            texture_set,
            smart_material,
            parent,
            index,
            parameters,
        } => {
            let set = find_set(project, texture_set)?;
            let wanted = smart_material.trim();
            let mut material = project
                .smart_materials
                .iter()
                .cloned()
                .chain(SmartMaterial::built_in())
                .find(|candidate| candidate.name == wanted)
                .ok_or_else(|| AutomationError::UnknownSmartMaterial(wanted.to_owned()))?;
            require_resources(project, &material.name, &material.requirements())?;
            for (name, value) in parameters {
                let index = material
                    .parameters
                    .iter()
                    .position(|parameter| parameter.name.eq_ignore_ascii_case(name.trim()))
                    .ok_or_else(|| {
                        AutomationError::InvalidValue(format!(
                            "`{}` has no parameter `{name}`; it has {}",
                            material.name,
                            material
                                .parameters
                                .iter()
                                .map(|parameter| format!("`{}`", parameter.name))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ))
                    })?;
                let value = parse_parameter_value(value)?;
                if !material.set_parameter(index, value) {
                    return Err(AutomationError::InvalidValue(format!(
                        "parameter `{name}` does not take {value:?}"
                    )));
                }
            }
            let parent = parent
                .as_deref()
                .map(|parent| find_group(set, parent))
                .transpose()?;
            let index = slot_index(*index, siblings(set, parent).len())?;
            let mut next = next_layer_id(project).0;
            let root = material.instantiate(|| {
                let id = LayerId(next);
                next += 1;
                id
            });
            let id = root.id;
            Ok((
                Some(EditCommand::InsertLayer {
                    texture_set_id: set.id,
                    slot: LayerSlot {
                        parent: parent.map(|group| group.id),
                        index,
                    },
                    layer: Box::new(root),
                }),
                EditOutcome {
                    summary: format!(
                        "inserted smart material `{}` as layer {} in `{}`{}",
                        material.name,
                        id.0,
                        set.name,
                        in_group(parent)
                    ),
                    layer: Some(id.0),
                    changed: true,
                },
            ))
        }
        AutomationEdit::ApplySmartMask {
            texture_set,
            layer,
            smart_mask,
        } => {
            let set = find_set(project, texture_set)?;
            let target = find_layer(set, layer)?;
            let wanted = smart_mask.trim();
            let mask = project
                .smart_masks
                .iter()
                .cloned()
                .chain(SmartMask::built_in())
                .find(|candidate| candidate.name == wanted)
                .ok_or_else(|| AutomationError::UnknownSmartMask(wanted.to_owned()))?;
            require_resources(project, &mask.name, &mask.requirements())?;
            let mut masked = target.clone();
            mask.apply_to(&mut masked);
            let summary = format!(
                "applied smart mask `{}` to layer {} `{}`",
                mask.name, target.id.0, target.name
            );
            Ok(settings_edit(
                set,
                target,
                target.settings(),
                masked.settings(),
                summary,
            ))
        }
    }
}

/// A layer-settings command from `before` to `after`, or no command when
/// they match.
fn settings_edit(
    set: &TextureSet,
    target: &MaterialLayer,
    before: LayerSettings,
    after: LayerSettings,
    summary: String,
) -> (Option<EditCommand>, EditOutcome) {
    if after == before {
        return (None, unchanged(Some(target.id)));
    }
    (
        Some(EditCommand::LayerSettings {
            texture_set_id: set.id,
            layer_id: target.id,
            before: Box::new(before),
            after: Box::new(after),
        }),
        EditOutcome {
            summary,
            layer: Some(target.id.0),
            changed: true,
        },
    )
}

/// Refuses a library entry whose bitmaps the project does not have.
fn require_resources(
    project: &MaterialProject,
    name: &str,
    requirements: &MapRequirements,
) -> Result<(), AutomationError> {
    match requirements
        .resources
        .iter()
        .find(|id| !project.resources.iter().any(|resource| resource.id == **id))
    {
        Some(missing) => Err(AutomationError::InvalidValue(format!(
            "`{name}` uses bitmap resource {}, which this project does not have",
            missing.0
        ))),
        None => Ok(()),
    }
}

/// `current` with the fields of the object `fields` replaced; unknown
/// fields are refused.
fn merge_fields<T: Serialize + serde::de::DeserializeOwned>(
    what: &str,
    current: &T,
    fields: &serde_json::Value,
) -> Result<T, AutomationError> {
    let invalid = |error: serde_json::Error| AutomationError::InvalidValue(error.to_string());
    let serde_json::Value::Object(fields) = fields else {
        return Err(AutomationError::InvalidValue(format!(
            "a {what} is an object of the settings to change"
        )));
    };
    let mut merged = serde_json::to_value(current).map_err(invalid)?;
    if let serde_json::Value::Object(settings) = &mut merged {
        for (key, value) in fields {
            if !settings.contains_key(key) {
                return Err(AutomationError::InvalidValue(format!(
                    "a {what} has no `{key}` setting"
                )));
            }
            settings.insert(key.clone(), value.clone());
        }
    }
    serde_json::from_value(merged).map_err(invalid)
}

/// A smart-material parameter value: a number or an sRGB `[r, g, b]`.
fn parse_parameter_value(value: &serde_json::Value) -> Result<ParameterValue, AutomationError> {
    if let Some(number) = value.as_f64() {
        return Ok(ParameterValue::Scalar(number as f32));
    }
    serde_json::from_value::<[u8; 3]>(value.clone())
        .map(ParameterValue::Color)
        .map_err(|_| {
            AutomationError::InvalidValue(format!(
                "a parameter is a number from 0 to 1 or a colour [r, g, b], not {value}"
            ))
        })
}

/// The serialized variant name of a filter (`Blur`, `HueSaturation`).
fn filter_kind(filter: LayerFilter) -> Option<String> {
    match serde_json::to_value(filter).ok()? {
        serde_json::Value::String(kind) => Some(kind),
        serde_json::Value::Object(map) => map.keys().next().cloned(),
        _ => None,
    }
}

/// The default filter named `name`: its display name (`Hue / Saturation`)
/// or its variant name (`HueSaturation`), ignoring case.
fn filter_preset(name: &str) -> Result<LayerFilter, AutomationError> {
    let name = name.trim();
    LayerFilter::PRESETS
        .into_iter()
        .find(|preset| {
            preset.name().eq_ignore_ascii_case(name)
                || filter_kind(*preset).is_some_and(|kind| kind.eq_ignore_ascii_case(name))
        })
        .ok_or_else(|| AutomationError::InvalidValue(format!("unknown filter `{name}`")))
}

/// A filter stack: each entry a filter name or an object with a `kind` and
/// the settings that replace its defaults.
fn parse_filters(values: &[serde_json::Value]) -> Result<Vec<LayerFilter>, AutomationError> {
    values
        .iter()
        .map(|value| {
            let fields = match value {
                serde_json::Value::String(name) => return filter_preset(name),
                serde_json::Value::Object(fields) => fields,
                _ => {
                    return Err(AutomationError::InvalidValue(
                        "a filter is a name or an object with a `kind`".to_owned(),
                    ));
                }
            };
            let kind = fields
                .get("kind")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    AutomationError::InvalidValue("a filter object needs a `kind`".to_owned())
                })?;
            let preset = filter_preset(kind)?;
            let settings: serde_json::Map<_, _> = fields
                .iter()
                .filter(|(key, _)| key.as_str() != "kind")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            let variant = filter_kind(preset).unwrap_or_default();
            let filter = match serde_json::to_value(preset) {
                Ok(serde_json::Value::Object(mut wrapper)) => {
                    let defaults = wrapper.remove(&variant).unwrap_or_default();
                    let merged = merge_fields(
                        &format!("{} filter", preset.name()),
                        &defaults,
                        &serde_json::Value::Object(settings),
                    )?;
                    serde_json::from_value(serde_json::json!({ variant: merged }))
                        .map_err(|error| AutomationError::InvalidValue(error.to_string()))?
                }
                // Settings-free filters such as Invert.
                _ if settings.is_empty() => preset,
                _ => {
                    return Err(AutomationError::InvalidValue(format!(
                        "the {} filter has no settings",
                        preset.name()
                    )));
                }
            };
            if !filter.is_valid() {
                return Err(AutomationError::InvalidValue(format!(
                    "the {} filter settings are out of range",
                    preset.name()
                )));
            }
            Ok(filter)
        })
        .collect()
}

/// The group whose id or exact name is `reference`.
fn find_group<'a>(
    set: &'a TextureSet,
    reference: &str,
) -> Result<&'a MaterialLayer, AutomationError> {
    let group = find_layer(set, reference)?;
    if group.is_group() {
        Ok(group)
    } else {
        Err(AutomationError::InvalidValue(format!(
            "layer `{}` is not a group",
            group.name
        )))
    }
}

/// The layers directly inside `parent`, or the top level.
fn siblings<'a>(set: &'a TextureSet, parent: Option<&'a MaterialLayer>) -> &'a [MaterialLayer] {
    match parent {
        Some(group) => group.children.as_deref().unwrap_or(&[]),
        None => &set.layers,
    }
}

fn in_group(parent: Option<&MaterialLayer>) -> String {
    parent.map_or_else(String::new, |group| format!(" in group `{}`", group.name))
}

/// `index` (from the bottom) among `count` layers, or the top when absent.
fn slot_index(index: Option<usize>, count: usize) -> Result<usize, AutomationError> {
    match index {
        None => Ok(count),
        Some(index) if index <= count => Ok(index),
        Some(index) => Err(AutomationError::InvalidValue(format!(
            "index {index} is past the top ({count})"
        ))),
    }
}

/// A channel by its name, ignoring case and spaces (`Base Color`,
/// `roughness`, `AmbientOcclusion`, `User 0` to `User 7`).
fn parse_channel(name: &str) -> Result<MaterialChannel, AutomationError> {
    let wanted: String = name
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    ALL_CHANNELS
        .into_iter()
        .chain(MaterialChannel::user_slots())
        .find(|channel| channel.file_name().to_ascii_lowercase() == wanted)
        .ok_or_else(|| AutomationError::InvalidValue(format!("unknown channel `{name}`")))
}

/// An anchor effect by name: `layer` (none), `fill`, `paint`, or
/// `filter N` counted from 1.
fn parse_anchor_effect(name: &str) -> Result<Option<AnchorEffect>, AutomationError> {
    let wanted = name.trim().to_ascii_lowercase();
    match wanted.as_str() {
        "layer" | "whole layer" | "none" => Ok(None),
        "fill" => Ok(Some(AnchorEffect::Fill)),
        "paint" => Ok(Some(AnchorEffect::Paint)),
        _ => wanted
            .strip_prefix("filter")
            .and_then(|number| number.trim().parse::<u32>().ok())
            .filter(|number| (1..=32).contains(number))
            .map(|number| Some(AnchorEffect::Filter(number - 1)))
            .ok_or_else(|| {
                AutomationError::InvalidValue(format!(
                    "unknown anchor effect `{name}`: use layer, fill, paint, or filter 1 to 32"
                ))
            }),
    }
}

/// The serialized `kind` tag of a generator.
fn generator_kind(generator: &MaskGenerator) -> Option<String> {
    serde_json::to_value(generator)
        .ok()?
        .get("kind")?
        .as_str()
        .map(str::to_owned)
}

/// The default generator named `name`: its display name (`Edge Wear`) or
/// its kind tag (`EdgeWear`), ignoring case.
fn generator_preset(name: &str) -> Result<MaskGenerator, AutomationError> {
    let name = name.trim();
    MaskGenerator::PRESETS
        .into_iter()
        .find(|preset| {
            preset.name().eq_ignore_ascii_case(name)
                || generator_kind(preset).is_some_and(|kind| kind.eq_ignore_ascii_case(name))
        })
        .ok_or_else(|| AutomationError::InvalidValue(format!("unknown generator `{name}`")))
}

/// The generator `value` asks for, given the layer's `current` one: a name
/// selects that generator (keeping the current settings when it is already
/// that kind), and an object's fields replace those of that generator.
fn parse_generator(
    value: &serde_json::Value,
    current: Option<MaskGenerator>,
) -> Result<Option<MaskGenerator>, AutomationError> {
    let based_on = |preset: MaskGenerator| match current {
        Some(current) if current.same_kind(&preset) => current,
        _ => preset,
    };
    let fields = match value {
        serde_json::Value::Null => return Ok(None),
        serde_json::Value::String(name)
            if name.trim().is_empty() || name.trim().eq_ignore_ascii_case("none") =>
        {
            return Ok(None);
        }
        serde_json::Value::String(name) => return Ok(Some(based_on(generator_preset(name)?))),
        serde_json::Value::Object(fields) => fields,
        _ => {
            return Err(AutomationError::InvalidValue(
                "a generator is a name or an object with a `kind`".to_owned(),
            ));
        }
    };
    let kind = fields
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            AutomationError::InvalidValue("a generator object needs a `kind`".to_owned())
        })?;
    let base = based_on(generator_preset(kind)?);
    let invalid = |error: serde_json::Error| AutomationError::InvalidValue(error.to_string());
    let mut merged = serde_json::to_value(base).map_err(invalid)?;
    if let serde_json::Value::Object(settings) = &mut merged {
        for (key, value) in fields {
            if key == "kind" {
                continue;
            }
            if !settings.contains_key(key) {
                return Err(AutomationError::InvalidValue(format!(
                    "the {} generator has no `{key}` setting",
                    base.name()
                )));
            }
            settings.insert(key.clone(), value.clone());
        }
    }
    let generator: MaskGenerator = serde_json::from_value(merged).map_err(invalid)?;
    if !generator.is_valid() {
        return Err(AutomationError::InvalidValue(format!(
            "the {} generator settings are out of range",
            generator.name()
        )));
    }
    Ok(Some(generator))
}

/// Texture-set ids, for callers that list them.
pub fn texture_set_ids(project: &MaterialProject) -> Vec<TextureSetId> {
    project.texture_sets.iter().map(|set| set.id).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::{TextureResolution, TextureSetId};

    fn project() -> MaterialProject {
        let set = |id: u64, name: &str, layers: Vec<MaterialLayer>| TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(id),
            name: name.to_owned(),
            material_slot: name.to_owned(),
            resolution: TextureResolution::new(32, 32).unwrap(),
            channels: [MaterialChannel::BaseColor, MaterialChannel::Roughness].into(),
            layers,
            height_strength: 1.0,
            export: None,
        };
        let mut group = MaterialLayer::new_group(LayerId(3), "Details");
        group.children = Some(vec![MaterialLayer::new_paint(LayerId(4), "Scratches")]);
        MaterialProject {
            texture_sets: vec![
                set(
                    1,
                    "Body",
                    vec![MaterialLayer::new_paint(LayerId(1), "Base"), group],
                ),
                set(
                    2,
                    "Trim",
                    vec![MaterialLayer::new_paint(LayerId(2), "Base")],
                ),
            ],
            active_texture_set: Some(TextureSetId(1)),
            ..MaterialProject::empty()
        }
    }

    #[test]
    fn tc_0280_automation_summarizes_projects_and_applies_validated_undoable_edits() {
        let mut project = project();
        let summary = project_summary(&project);
        assert_eq!(summary.api_version, AUTOMATION_API_VERSION);
        assert_eq!(summary.texture_sets[0].layers[1].kind, "group");
        assert_eq!(
            summary.texture_sets[0].layers[1].children[0].name,
            "Scratches"
        );
        assert_eq!(
            texture_set_ids(&project),
            [TextureSetId(1), TextureSetId(2)]
        );

        // A built-in material is copied into a new top layer with an id no
        // other layer uses; values given override it.
        let outcome = apply_automation_edit(
            &mut project,
            &AutomationEdit::AddFillLayer {
                texture_set: "Body".to_owned(),
                name: "Gold trim".to_owned(),
                material: Some("Gold".to_owned()),
                color: None,
                roughness: Some(0.6),
                metallic: None,
            },
        )
        .unwrap();
        assert_eq!(outcome.layer, Some(5));
        let layer = project.texture_sets[0].layer(LayerId(5)).unwrap();
        assert_eq!(layer.fill.unwrap().metallic, 1.0);
        assert_eq!(layer.fill.unwrap().roughness, 0.6);
        assert_eq!(
            project.texture_sets[0].layers.last().unwrap().id,
            LayerId(5)
        );

        // Names resolve uniquely; numeric ids always work.
        let rename = |layer: &str| AutomationEdit::SetLayer {
            texture_set: "1".to_owned(),
            layer: layer.to_owned(),
            name: None,
            visible: Some(false),
            opacity: Some(0.25),
            blend_mode: Some("multiply".to_owned()),
        };
        apply_automation_edit(&mut project, &rename("Scratches")).unwrap();
        let scratches = project.texture_sets[0].layer(LayerId(4)).unwrap();
        assert!(!scratches.visible);
        assert_eq!(scratches.blend_mode, BlendMode::Multiply);
        assert!(
            !apply_automation_edit(&mut project, &rename("4"))
                .unwrap()
                .changed
        );
        project.texture_sets[1].name = "Body".to_owned();
        assert_eq!(
            apply_automation_edit(
                &mut project,
                &AutomationEdit::RemoveLayer {
                    texture_set: "Body".to_owned(),
                    layer: "Base".to_owned(),
                }
            ),
            Err(AutomationError::Ambiguous("Body".to_owned()))
        );
        project.texture_sets[1].name = "Trim".to_owned();

        // Bad references and values are refused without changing anything.
        let before = project.clone();
        for (edit, expected) in [
            (
                AutomationEdit::RemoveLayer {
                    texture_set: "Nope".to_owned(),
                    layer: "Base".to_owned(),
                },
                AutomationError::UnknownTextureSet("Nope".to_owned()),
            ),
            (
                AutomationEdit::RemoveLayer {
                    texture_set: "Trim".to_owned(),
                    layer: "99".to_owned(),
                },
                AutomationError::UnknownLayer("99".to_owned()),
            ),
            (
                AutomationEdit::SetLayer {
                    texture_set: "Body".to_owned(),
                    layer: "Base".to_owned(),
                    name: None,
                    visible: None,
                    opacity: Some(1.5),
                    blend_mode: None,
                },
                AutomationError::InvalidValue(
                    "opacity must be between 0 and 1, not 1.5".to_owned(),
                ),
            ),
            (
                AutomationEdit::SetLayer {
                    texture_set: "Body".to_owned(),
                    layer: "Base".to_owned(),
                    name: None,
                    visible: None,
                    opacity: None,
                    blend_mode: Some("Pass through".to_owned()),
                },
                AutomationError::InvalidValue("unknown blend mode `Pass through`".to_owned()),
            ),
            (
                AutomationEdit::AddFillLayer {
                    texture_set: "Body".to_owned(),
                    name: "x".to_owned(),
                    material: Some("Unobtainium".to_owned()),
                    color: None,
                    roughness: None,
                    metallic: None,
                },
                AutomationError::UnknownMaterial("Unobtainium".to_owned()),
            ),
        ] {
            assert_eq!(apply_automation_edit(&mut project, &edit), Err(expected));
            assert_eq!(project, before);
        }

        // Removing a layer another layer anchors to would break the project,
        // so it is refused and undone.
        let mut anchored = project.clone();
        anchored.texture_sets[0].layers[1].mask_anchor = Some(
            substancecraft_core::AnchorMask::new(LayerId(1), MaterialChannel::Roughness),
        );
        anchored.validate().unwrap();
        let untouched = anchored.clone();
        let refused = apply_automation_edit(
            &mut anchored,
            &AutomationEdit::RemoveLayer {
                texture_set: "Body".to_owned(),
                layer: "1".to_owned(),
            },
        );
        assert!(matches!(refused, Err(AutomationError::Rejected(_))));
        assert_eq!(anchored, untouched);
    }

    #[test]
    fn tc_0281_automation_edits_materials_and_their_instances_in_project_files() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("auto.sproject");
        let mut project = project();
        let steel = MaterialResource::built_in()
            .into_iter()
            .find(|material| material.name == "Steel")
            .unwrap();
        let (_, add) = project.add_material_command(steel).unwrap();
        let mut history = EditHistory::default();
        assert!(history.apply(&mut project, add));
        save_project_file(&path, &project).unwrap();

        // A fill linked to the master in each set, one with an override.
        for (set, roughness) in [("Body", None), ("Trim", Some(0.9))] {
            let outcome = edit_project_file(
                &path,
                &AutomationEdit::AddFillLayer {
                    texture_set: set.to_owned(),
                    name: "Steel".to_owned(),
                    material: Some("Steel".to_owned()),
                    color: None,
                    roughness,
                    metallic: None,
                },
            )
            .unwrap();
            assert!(outcome.summary.contains("linked to material `Steel`"));
        }
        let summary = project_summary(&crate::load_project_file(&path).unwrap());
        assert_eq!(summary.materials[0].instances, 2);
        assert_eq!(summary.texture_sets[1].layers[1].material, Some(1));

        // Editing the master updates both instances, keeping the override.
        let outcome = edit_project_file(
            &path,
            &AutomationEdit::EditMaterial {
                material: "1".to_owned(),
                name: None,
                color: Some([10, 20, 30]),
                roughness: Some(0.2),
                metallic: None,
                height: None,
                opacity: None,
            },
        )
        .unwrap();
        assert_eq!(
            outcome.summary,
            "updated material 1 `Steel` and 2 linked layer(s)"
        );
        let reloaded = crate::load_project_file(&path).unwrap();
        let fill = |set: usize| {
            reloaded.texture_sets[set]
                .layers
                .last()
                .unwrap()
                .fill
                .unwrap()
        };
        assert_eq!(fill(0).color[..3], [10, 20, 30]);
        assert_eq!(fill(0).roughness, 0.2);
        assert_eq!(fill(1).color[..3], [10, 20, 30]);
        assert_eq!(fill(1).roughness, 0.9, "the override is kept");

        // Linking an existing layer; an unchanged edit leaves the file alone.
        edit_project_file(
            &path,
            &AutomationEdit::LinkMaterial {
                texture_set: "Body".to_owned(),
                layer: "Base".to_owned(),
                material: "Steel".to_owned(),
            },
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let again = edit_project_file(
            &path,
            &AutomationEdit::LinkMaterial {
                texture_set: "Body".to_owned(),
                layer: "Base".to_owned(),
                material: "Steel".to_owned(),
            },
        )
        .unwrap();
        assert!(!again.changed);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);

        // Edits are JSON, tagged by kind.
        let edit: AutomationEdit = serde_json::from_str(
            r#"{"edit":"set_layer","texture_set":"Body","layer":"Base","visible":false}"#,
        )
        .unwrap();
        assert!(matches!(
            edit,
            AutomationEdit::SetLayer {
                visible: Some(false),
                ..
            }
        ));
    }

    fn edit(json: &str) -> AutomationEdit {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn tc_0307_automation_adds_and_moves_layers_and_groups() {
        let mut project = project();

        // A group at the bottom, then a paint layer on top inside it.
        let outcome = apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"add_layer","texture_set":"Body","name":"Wear","kind":"group","index":0}"#),
        )
        .unwrap();
        assert_eq!(outcome.layer, Some(5));
        assert_eq!(outcome.summary, "added group 5 `Wear` to `Body`");
        apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"add_layer","texture_set":"Body","name":"Chips","parent":"Wear"}"#),
        )
        .unwrap();
        let body = &project.texture_sets[0];
        assert!(body.layers[0].is_group());
        let chips = &body.layers[0].children.as_ref().unwrap()[0];
        assert_eq!((chips.id, chips.name.as_str()), (LayerId(6), "Chips"));
        assert!(!chips.is_group());

        // Base moves under Chips inside the group; its old place closes up.
        let outcome = apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"move_layer","texture_set":"Body","layer":"Base","parent":"5","index":0}"#),
        )
        .unwrap();
        assert_eq!(
            outcome.summary,
            "moved layer 1 `Base` to position 0 in group `Wear`"
        );
        let body = &project.texture_sets[0];
        let names: Vec<&str> = body.layers[0]
            .children
            .as_ref()
            .unwrap()
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["Base", "Chips"]);
        assert_eq!(body.layers.len(), 2);

        // Without a parent and index, a layer moves to the top level's top;
        // a move to where it already is changes nothing.
        apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"move_layer","texture_set":"Body","layer":"Chips"}"#),
        )
        .unwrap();
        assert_eq!(project.texture_sets[0].layers.last().unwrap().name, "Chips");
        let again = apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"move_layer","texture_set":"Body","layer":"Chips"}"#),
        )
        .unwrap();
        assert!(!again.changed);

        let before = project.clone();
        for (json, expected) in [
            (
                r#"{"edit":"move_layer","texture_set":"Body","layer":"Wear","parent":"Wear"}"#,
                "a group cannot move into itself",
            ),
            (
                r#"{"edit":"add_layer","texture_set":"Body","name":"x","parent":"Chips"}"#,
                "layer `Chips` is not a group",
            ),
            (
                r#"{"edit":"add_layer","texture_set":"Body","name":"x","index":9}"#,
                "index 9 is past the top (3)",
            ),
            (
                r#"{"edit":"add_layer","texture_set":"Body","name":"  "}"#,
                "a layer needs a name",
            ),
        ] {
            assert_eq!(
                apply_automation_edit(&mut project, &edit(json)),
                Err(AutomationError::InvalidValue(expected.to_owned())),
                "{json}"
            );
            assert_eq!(project, before);
        }
        // A group nested inside its own child is refused too.
        apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"add_layer","texture_set":"Body","name":"Inner","kind":"group","parent":"Wear"}"#),
        )
        .unwrap();
        assert_eq!(
            apply_automation_edit(
                &mut project,
                &edit(
                    r#"{"edit":"move_layer","texture_set":"Body","layer":"Wear","parent":"Inner"}"#
                ),
            ),
            Err(AutomationError::InvalidValue(
                "a group cannot move into itself".to_owned()
            ))
        );
    }

    #[test]
    fn tc_0308_automation_sets_mask_generators_and_paints_strokes() {
        let mut project = project();
        let layer = |project: &MaterialProject| project.texture_sets[0].layer(LayerId(1)).cloned();

        // By name: the default Edge Wear generator, inverted.
        apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"set_mask","texture_set":"Body","layer":"Base","generator":"edge wear","invert":true}"#),
        )
        .unwrap();
        let base = layer(&project).unwrap();
        assert_eq!(
            base.mask_generator,
            Some(MaskGenerator::EdgeWear(
                substancecraft_core::EdgeWear::DEFAULT
            ))
        );
        assert!(base.invert_final_mask);

        // An object changes only the fields it names, keeping the others.
        apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"set_mask","texture_set":"Body","layer":"Base","generator":{"kind":"EdgeWear","seed":7}}"#),
        )
        .unwrap();
        let outcome = apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"set_mask","texture_set":"Body","layer":"Base","generator":{"kind":"Edge Wear","amount":0.9}}"#),
        )
        .unwrap();
        assert_eq!(
            outcome.summary,
            "set the mask of layer 1 `Base` (Edge Wear)"
        );
        let wear = match layer(&project).unwrap().mask_generator {
            Some(MaskGenerator::EdgeWear(wear)) => Some(wear),
            _ => None,
        }
        .unwrap();
        assert_eq!((wear.amount, wear.seed), (0.9, 7));
        assert_eq!(
            wear.contrast,
            substancecraft_core::EdgeWear::DEFAULT.contrast
        );
        // Naming the kind it already has keeps its settings.
        let same = apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_mask","texture_set":"Body","layer":"Base","generator":"EdgeWear"}"#,
            ),
        )
        .unwrap();
        assert!(!same.changed);

        let before = project.clone();
        for (generator, expected) in [
            (
                r#"{"kind":"EdgeWear","sharpness":1}"#,
                "the Edge Wear generator has no `sharpness` setting",
            ),
            (
                r#"{"kind":"EdgeWear","amount":3}"#,
                "the Edge Wear generator settings are out of range",
            ),
            (r#""Sparkles""#, "unknown generator `Sparkles`"),
            (r#"{"amount":1}"#, "a generator object needs a `kind`"),
        ] {
            let json = format!(
                r#"{{"edit":"set_mask","texture_set":"Body","layer":"Base","generator":{generator}}}"#
            );
            assert_eq!(
                apply_automation_edit(&mut project, &edit(&json)),
                Err(AutomationError::InvalidValue(expected.to_owned())),
                "{generator}"
            );
            assert_eq!(project, before);
        }
        apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"set_mask","texture_set":"Body","layer":"Base","generator":"none","invert":false}"#),
        )
        .unwrap();
        assert_eq!(layer(&project).unwrap().mask_generator, None);
        assert!(!layer(&project).unwrap().invert_final_mask);

        // Strokes: paint and mask strokes record as written, with defaults
        // for the material values a stroke leaves out.
        let outcome = apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"paint","texture_set":"Body","layer":"Base","stroke":{"points":[[0.25,0.5],[0.75,0.5]],"radius_pixels":3,"color":[255,0,0,255],"opacity":1,"roughness":0.2}}"#),
        )
        .unwrap();
        assert_eq!(outcome.summary, "painted 2 point(s) on layer 1 `Base`");
        apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"paint_mask","texture_set":"Body","layer":"Details","stroke":{"points":[[0.5,0.5]],"radius_pixels":4,"value":0,"opacity":1}}"#),
        )
        .unwrap();
        let base = layer(&project).unwrap();
        assert_eq!(base.paint_strokes.len(), 1);
        assert_eq!(base.paint_strokes[0].roughness, 0.2);
        assert_eq!(
            base.paint_strokes[0].metallic,
            substancecraft_core::DEFAULT_METALLIC
        );
        assert_eq!(
            project.texture_sets[0]
                .layer(LayerId(3))
                .unwrap()
                .mask_strokes
                .len(),
            1
        );
        // The painted texels composite: the stroke's middle is red.
        let red = project.texture_sets[0]
            .bake_channel(MaterialChannel::BaseColor, Default::default())
            .unwrap();
        let texel = (16 * 32 + 16) * 4;
        assert_eq!(red.rgba8[texel..texel + 3], [255, 0, 0]);

        // Groups hold no paint, and invalid strokes are refused unchanged.
        let before = project.clone();
        assert_eq!(
            apply_automation_edit(
                &mut project,
                &edit(
                    r#"{"edit":"paint","texture_set":"Body","layer":"Details","stroke":{"points":[[0.5,0.5]],"radius_pixels":3,"color":[0,0,0,255],"opacity":1}}"#
                ),
            ),
            Err(AutomationError::InvalidValue(
                "groups hold no paint; paint a layer inside".to_owned()
            ))
        );
        assert!(matches!(
            apply_automation_edit(
                &mut project,
                &edit(
                    r#"{"edit":"paint","texture_set":"Body","layer":"Base","stroke":{"points":[[0.5,0.5]],"radius_pixels":0,"color":[0,0,0,255],"opacity":1}}"#
                ),
            ),
            Err(AutomationError::Rejected(_))
        ));
        assert_eq!(project, before);
    }

    #[test]
    fn tc_0367_automation_sets_and_removes_procedural_graphs() {
        let mut project = project();
        let graph = r#"{"nodes":[{"Generator":{"kind":"Noise","scale":6.0,"amount":0.5,"contrast":0.5,"seed":3}},{"Ramp":{"input":{"Node":0},"stops":[{"position":0.0,"color":[0,0,0]},{"position":1.0,"color":[255,128,0]}]}}],"outputs":{"BaseColor":{"Node":1},"Roughness":{"Value":[0.3,0.3,0.3]}}}"#;
        let set_graph = |layer: &str, graph: &str| {
            format!(
                r#"{{"edit":"set_graph","texture_set":"Body","layer":"{layer}","graph":{graph}}}"#
            )
        };
        // Only fill layers take a graph.
        assert!(matches!(
            refused(&mut project, &set_graph("Base", graph)),
            AutomationError::InvalidValue(reason) if reason.contains("only fill layers")
        ));
        project.texture_sets[0].layers[0].fill = Some(LayerFill::default());
        let outcome =
            apply_automation_edit(&mut project, &edit(&set_graph("Base", graph))).unwrap();
        assert_eq!(outcome.summary, "made layer 1 `Base` procedural (2 nodes)");
        let stored = project.texture_sets[0].layers[0].graph.clone().unwrap();
        assert_eq!(stored.nodes.len(), 2);
        assert_eq!(
            stored.outputs[&MaterialChannel::BaseColor],
            substancecraft_core::GraphInput::Node(1)
        );
        // A forward reference is refused and changes nothing.
        let before = project.clone();
        let forward = graph.replace(r#""input":{"Node":0}"#, r#""input":{"Node":1}"#);
        assert!(matches!(
            refused(&mut project, &set_graph("Base", &forward)),
            AutomationError::InvalidValue(reason) if reason.contains("the graph is invalid")
        ));
        assert_eq!(project, before);
        apply_automation_edit(&mut project, &edit(&set_graph("Base", "null"))).unwrap();
        assert_eq!(project.texture_sets[0].layers[0].graph, None);
        project.validate().unwrap();
    }

    #[test]
    fn tc_0362_automation_defines_user_channels_that_layers_paint() {
        let mut project = project();
        apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_texture_set","texture_set":"Body","user_channels":[{"slot":3,"name":"Wetness","gray":true,"default":[10,10,10]}]}"#,
            ),
        )
        .unwrap();
        let body = &project.texture_sets[0];
        assert_eq!(body.user_channels.len(), 1);
        assert_eq!(
            body.user_channel(MaterialChannel::User(3)).unwrap().name,
            "Wetness"
        );
        // Layers and sets name user channels by slot.
        apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"set_texture_set","texture_set":"Body","channels":["Base Color","User 3"]}"#),
        )
        .unwrap();
        assert!(
            project.texture_sets[0]
                .channels
                .contains(&MaterialChannel::User(3))
        );
        apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"paint","texture_set":"Body","layer":"Base","stroke":{"points":[[0.5,0.5]],"radius_pixels":3,"color":[0,0,0,255],"opacity":1,"user":[[0,0,0],[0,0,0],[0,0,0],[255,255,255],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]}}"#,
            ),
        )
        .unwrap();
        let before = project.clone();
        for json in [
            r#"{"edit":"set_texture_set","texture_set":"Body","user_channels":[{"slot":8,"name":"Far"}]}"#,
            r#"{"edit":"set_texture_set","texture_set":"Body","user_channels":[{"slot":0,"name":"A"},{"slot":1,"name":"a"}]}"#,
        ] {
            assert!(
                matches!(
                    apply_automation_edit(&mut project, &edit(json)),
                    Err(AutomationError::InvalidValue(_))
                ),
                "{json}"
            );
        }
        assert_eq!(project, before);
    }

    #[test]
    fn tc_0309_automation_sets_texture_set_and_mesh_properties() {
        let mut project = project();
        apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"set_texture_set","texture_set":"Trim","name":"Trim 2","resolution":[64,128],"channels":["Base Color","height","AmbientOcclusion"],"height_strength":4}"#),
        )
        .unwrap();
        let trim = &project.texture_sets[1];
        assert_eq!(trim.name, "Trim 2");
        assert_eq!((trim.resolution.width, trim.resolution.height), (64, 128));
        assert_eq!(
            trim.channels,
            [
                MaterialChannel::BaseColor,
                MaterialChannel::Height,
                MaterialChannel::AmbientOcclusion
            ]
            .into()
        );
        assert_eq!(trim.height_strength, 4.0);

        let before = project.clone();
        for (json, expected) in [
            (
                r#"{"edit":"set_texture_set","texture_set":"Body","channels":["Shine"]}"#,
                Some("unknown channel `Shine`"),
            ),
            (
                r#"{"edit":"set_texture_set","texture_set":"Body","channels":[]}"#,
                Some("a texture set needs at least one channel"),
            ),
            (
                r#"{"edit":"set_texture_set","texture_set":"Body","height_strength":17}"#,
                Some("height strength must be between 0 and 16, not 17"),
            ),
            (
                r#"{"edit":"set_texture_set","texture_set":"Body","resolution":[0,32]}"#,
                None,
            ),
            (
                r#"{"edit":"set_mesh","scale":2}"#,
                Some("the project has no mesh"),
            ),
        ] {
            let result = apply_automation_edit(&mut project, &edit(json));
            match expected {
                Some(message) => assert_eq!(
                    result,
                    Err(AutomationError::InvalidValue(message.to_owned())),
                    "{json}"
                ),
                None => assert!(
                    matches!(result, Err(AutomationError::InvalidValue(_))),
                    "{json}"
                ),
            }
            assert_eq!(project, before);
        }

        // A subdivision change carries polygon masks to the new level.
        project.mesh = Some(substancecraft_core::MeshAsset {
            id: substancecraft_core::MeshId(1),
            source: "builtin:cube".to_owned(),
            scale: 1.0,
            smoothing: Default::default(),
            high_poly_source: None,
            content_hash: None,
            high_poly_hash: None,
        });
        project.texture_sets[0].layers[0].polygon_mask = Some(substancecraft_core::PolygonMask {
            triangles: [0, 2].into(),
            inverted: false,
        });
        let outcome = apply_automation_edit(
            &mut project,
            &edit(r#"{"edit":"set_mesh","scale":2,"smoothing_angle":30,"subdivision_levels":1}"#),
        )
        .unwrap();
        assert_eq!(
            outcome.summary,
            "updated the mesh: scale 2, smoothing 30°, 1 subdivision level(s); remapped 1 polygon mask(s)"
        );
        let mesh = project.mesh.as_ref().unwrap();
        assert_eq!(mesh.scale, 2.0);
        assert_eq!(mesh.smoothing.angle_degrees, 30.0);
        assert_eq!(mesh.smoothing.subdivision_levels, 1);
        assert_eq!(
            project.texture_sets[0].layers[0]
                .polygon_mask
                .as_ref()
                .unwrap()
                .triangles,
            [0, 1, 2, 3, 8, 9, 10, 11].into()
        );
        assert_eq!(
            apply_automation_edit(
                &mut project,
                &edit(r#"{"edit":"set_mesh","subdivision_levels":4}"#)
            ),
            Err(AutomationError::InvalidValue(
                "at most 3 subdivision levels, not 4".to_owned()
            ))
        );
    }

    /// Applies an edit that must be refused and checks the project did not
    /// change.
    fn refused(project: &mut MaterialProject, json: &str) -> AutomationError {
        let before = project.clone();
        let error = apply_automation_edit(project, &edit(json)).unwrap_err();
        assert_eq!(*project, before, "{json} changed the project");
        error
    }

    #[test]
    fn tc_0323_automation_sets_content_and_mask_filter_stacks() {
        let mut project = project();
        let outcome = apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_filters","texture_set":"Body","layer":"Base",
                    "filters":["Blur",{"kind":"Levels","gamma":2.0},"invert"],
                    "mask_filters":[{"kind":"hue / saturation","saturation":0.5}],
                    "channels":["roughness"]}"#,
            ),
        )
        .unwrap();
        assert_eq!(
            outcome.summary,
            "set the filters of layer 1 `Base` (3 on content, 1 on mask)"
        );
        let base = project.texture_sets[0].layer(LayerId(1)).unwrap().clone();
        assert_eq!(
            base.filters,
            [
                LayerFilter::Blur { radius: 4.0 },
                LayerFilter::Levels {
                    in_low: 0.0,
                    in_high: 1.0,
                    gamma: 2.0,
                    out_low: 0.0,
                    out_high: 1.0,
                },
                LayerFilter::Invert,
            ]
        );
        assert_eq!(
            base.mask_filters,
            [LayerFilter::HueSaturation {
                hue: 0.0,
                saturation: 0.5,
                lightness: 0.0,
            }]
        );
        assert_eq!(base.filter_channels, [MaterialChannel::Roughness].into());
        let summary = project_summary(&project);
        assert_eq!(
            summary.texture_sets[0].layers[0].filters,
            ["Blur", "Levels", "Invert"]
        );
        assert!(
            summary.texture_sets[0].layers[0]
                .mask
                .contains(&"1 mask filter(s)".to_owned())
        );

        // Asking for what the layer has changes nothing.
        let again = apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_filters","texture_set":"Body","layer":"Base","channels":["Roughness"]}"#,
            ),
        )
        .unwrap();
        assert!(!again.changed);

        // Bad filters are refused with their reason and change nothing.
        let base_edit = |filter: &str| {
            format!(
                r#"{{"edit":"set_filters","texture_set":"Body","layer":"Base","filters":[{filter}]}}"#
            )
        };
        for (filter, reason) in [
            (r#""Glow""#, "unknown filter `Glow`"),
            (
                r#"{"kind":"Blur","radius":500}"#,
                "the Blur filter settings are out of range",
            ),
            (
                r#"{"kind":"Blur","strength":1}"#,
                "a Blur filter has no `strength` setting",
            ),
            (
                r#"{"kind":"Invert","amount":1}"#,
                "the Invert filter has no settings",
            ),
            (r#"{"radius":2}"#, "a filter object needs a `kind`"),
            ("3", "a filter is a name or an object with a `kind`"),
        ] {
            assert_eq!(
                refused(&mut project, &base_edit(filter)),
                AutomationError::InvalidValue(reason.to_owned()),
                "{filter}"
            );
        }
        assert_eq!(
            refused(
                &mut project,
                r#"{"edit":"set_filters","texture_set":"Body","layer":"Base","channels":["Shine"]}"#
            ),
            AutomationError::InvalidValue("unknown channel `Shine`".to_owned())
        );

        // Empty lists clear the stacks and the channel scope.
        apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_filters","texture_set":"Body","layer":"Base","filters":[],"mask_filters":[],"channels":[]}"#,
            ),
        )
        .unwrap();
        let base = project.texture_sets[0].layer(LayerId(1)).unwrap();
        assert!(base.filters.is_empty() && base.mask_filters.is_empty());
        assert!(base.filter_channels.is_empty());
    }

    #[test]
    fn tc_0324_automation_sets_bitmap_polygon_and_anchor_masks() {
        let mut project = project();
        project
            .resources
            .push(substancecraft_core::ProjectResource::new(
                substancecraft_core::ResourceId(1),
                "Stencil",
                "stencil.png",
            ));
        let layer = |project: &MaterialProject, id: u64| {
            project.texture_sets[0].layer(LayerId(id)).unwrap().clone()
        };

        // Bitmap masks: a resource by name, inverted, through a projection
        // whose named fields change.
        apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_bitmap_mask","texture_set":"Body","layer":"Base","resource":"Stencil",
                    "invert":true,"projection":{"mode":"Triplanar","scale":2.0}}"#,
            ),
        )
        .unwrap();
        apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_bitmap_mask","texture_set":"Body","layer":"Base","projection":{"rotation_degrees":45.0}}"#,
            ),
        )
        .unwrap();
        let mask = layer(&project, 1).mask_bitmap.unwrap();
        assert_eq!(mask.resource, substancecraft_core::ResourceId(1));
        assert!(mask.invert);
        assert_eq!(
            mask.projection.mode,
            substancecraft_core::ProjectionMode::Triplanar
        );
        assert_eq!(
            (mask.projection.scale, mask.projection.rotation_degrees),
            (2.0, 45.0)
        );
        assert_eq!(
            refused(
                &mut project,
                r#"{"edit":"set_bitmap_mask","texture_set":"Body","layer":"Base","resource":"Missing"}"#
            ),
            AutomationError::UnknownResource("Missing".to_owned())
        );
        assert_eq!(
            refused(
                &mut project,
                r#"{"edit":"set_bitmap_mask","texture_set":"Body","layer":"Base","projection":{"zoom":2}}"#
            ),
            AutomationError::InvalidValue("a projection has no `zoom` setting".to_owned())
        );
        assert!(matches!(
            refused(
                &mut project,
                r#"{"edit":"set_bitmap_mask","texture_set":"Body","layer":"Base","projection":{"scale":-1}}"#
            ),
            AutomationError::Rejected(_)
        ));
        apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_bitmap_mask","texture_set":"Body","layer":"Base","resource":"none"}"#,
            ),
        )
        .unwrap();
        assert_eq!(layer(&project, 1).mask_bitmap, None);
        assert_eq!(
            refused(
                &mut project,
                r#"{"edit":"set_bitmap_mask","texture_set":"Body","layer":"Base","invert":true}"#
            ),
            AutomationError::InvalidValue(
                "the layer has no bitmap mask; name a `resource`".to_owned()
            )
        );

        // Polygon masks: triangles, then inverted, then removed.
        let outcome = apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_polygon_mask","texture_set":"Body","layer":"Base","triangles":[3,1,2,1]}"#,
            ),
        )
        .unwrap();
        assert_eq!(outcome.summary, "limited layer 1 `Base` to 3 triangle(s)");
        let outcome = apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_polygon_mask","texture_set":"Body","layer":"Base","inverted":true}"#,
            ),
        )
        .unwrap();
        assert_eq!(
            outcome.summary,
            "limited layer 1 `Base` to all but 3 triangle(s)"
        );
        let polygon = layer(&project, 1).polygon_mask.unwrap();
        assert_eq!(polygon.triangles, [1, 2, 3].into());
        assert!(polygon.inverted);
        apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_polygon_mask","texture_set":"Body","layer":"Base","triangles":[]}"#,
            ),
        )
        .unwrap();
        assert_eq!(layer(&project, 1).polygon_mask, None);
        assert_eq!(
            refused(
                &mut project,
                r#"{"edit":"set_polygon_mask","texture_set":"Body","layer":"Base","inverted":true}"#
            ),
            AutomationError::InvalidValue(
                "the layer has no polygon mask; give its `triangles`".to_owned()
            )
        );

        // Anchor masks read an earlier layer outside the anchored one.
        let outcome = apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"set_anchor_mask","texture_set":"Body","layer":"Scratches","anchor":"Base",
                    "channel":"Roughness","low":0.2,"high":0.8,"invert":true}"#,
            ),
        )
        .unwrap();
        assert_eq!(
            outcome.summary,
            "anchored layer 4 `Scratches` to layer 1 (Roughness)"
        );
        assert_eq!(
            layer(&project, 4).mask_anchor,
            Some(AnchorMask {
                layer: LayerId(1),
                channel: MaterialChannel::Roughness,
                low: 0.2,
                high: 0.8,
                invert: true,
                effect: None,
            })
        );
        // An anchor can read its layer part-way through its effects.
        let effect_edit = |effect: &str| {
            format!(
                r#"{{"edit":"set_anchor_mask","texture_set":"Body","layer":"Scratches","effect":"{effect}"}}"#
            )
        };
        for (name, expected) in [
            ("paint", Some(substancecraft_core::AnchorEffect::Paint)),
            (
                "Filter 2",
                Some(substancecraft_core::AnchorEffect::Filter(1)),
            ),
            ("fill", Some(substancecraft_core::AnchorEffect::Fill)),
            ("layer", None),
        ] {
            apply_automation_edit(&mut project, &edit(&effect_edit(name))).unwrap();
            assert_eq!(
                layer(&project, 4).mask_anchor.unwrap().effect,
                expected,
                "{name}"
            );
        }
        for name in ["filter 0", "filter 33", "mask"] {
            assert!(
                matches!(
                    refused(&mut project, &effect_edit(name)),
                    AutomationError::InvalidValue(reason) if reason.contains("unknown anchor effect")
                ),
                "{name}"
            );
        }
        assert!(
            project_summary(&project).texture_sets[0].layers[1].children[0]
                .mask
                .contains(&"Anchor to Base (Roughness)".to_owned())
        );
        let anchor_edit = |layer: &str, anchor: &str| {
            format!(
                r#"{{"edit":"set_anchor_mask","texture_set":"Body","layer":"{layer}","anchor":"{anchor}"}}"#
            )
        };
        for (layer, anchor) in [
            ("Base", "Scratches"),
            ("Scratches", "Details"),
            ("Base", "Base"),
        ] {
            assert!(
                matches!(
                    refused(&mut project, &anchor_edit(layer, anchor)),
                    AutomationError::InvalidValue(reason) if reason.contains("can only anchor")
                ),
                "{layer} to {anchor}"
            );
        }
        assert_eq!(
            refused(
                &mut project,
                r#"{"edit":"set_anchor_mask","texture_set":"Body","layer":"Scratches","channel":"Normal"}"#
            ),
            AutomationError::InvalidValue("anchors cannot read the Normal channel".to_owned())
        );
        assert_eq!(
            refused(
                &mut project,
                r#"{"edit":"set_anchor_mask","texture_set":"Body","layer":"Scratches","low":0.9}"#
            ),
            AutomationError::InvalidValue("low (0.9) must be below high (0.8)".to_owned())
        );
        apply_automation_edit(&mut project, &edit(&anchor_edit("Scratches", "none"))).unwrap();
        assert_eq!(layer(&project, 4).mask_anchor, None);
        assert_eq!(
            refused(
                &mut project,
                r#"{"edit":"set_anchor_mask","texture_set":"Body","layer":"Scratches","invert":true}"#
            ),
            AutomationError::InvalidValue(
                "the layer has no anchor mask; name an `anchor` layer".to_owned()
            )
        );
        project.validate().unwrap();
    }

    #[test]
    fn tc_0325_automation_lists_the_library_inserts_smart_materials_and_applies_smart_masks() {
        let mut project = project();
        let library = library_summary(&project);
        let worn = library
            .smart_materials
            .iter()
            .find(|material| material.name == "Worn painted steel")
            .unwrap();
        assert_eq!(worn.origin, "built-in");
        assert_eq!(worn.controls, ["Wear"]);
        assert!(worn.needs_mesh_maps);
        assert_eq!(
            worn.parameters["Paint color"],
            serde_json::json!([40, 70, 120])
        );
        assert!(
            library
                .smart_masks
                .iter()
                .any(|mask| mask.name == "Worn edges")
        );
        assert!(
            library
                .materials
                .iter()
                .any(|material| material.name == "Gold")
        );

        // A smart material inserts as a fresh tree inside a group, with its
        // parameters set and its control following the new ids.
        let outcome = apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"insert_smart_material","texture_set":"Body","smart_material":"Worn painted steel",
                    "parent":"Details","index":0,"parameters":{"paint color":[200,30,30],"Paint roughness":0.2}}"#,
            ),
        )
        .unwrap();
        assert_eq!(
            outcome.summary,
            "inserted smart material `Worn painted steel` as layer 5 in `Body` in group `Details`"
        );
        let details = project.texture_sets[0].layer(LayerId(3)).unwrap();
        let root = &details.children.as_ref().unwrap()[0];
        assert_eq!(root.id, LayerId(5));
        let ids: Vec<u64> = root.walk().iter().map(|layer| layer.id.0).collect();
        assert_eq!(ids, [5, 6, 7, 8]);
        let paint = project.texture_sets[0].layer(LayerId(7)).unwrap();
        assert_eq!(paint.name, "Paint");
        let fill = paint.fill.unwrap();
        assert_eq!(fill.color, [200, 30, 30, 255]);
        assert_eq!(fill.roughness, 0.2);
        let root = project.texture_sets[0].layer(LayerId(5)).unwrap();
        assert!(
            root.evolutions[0]
                .drives
                .iter()
                .all(|drive| (6..=8).contains(&drive.layer.0))
        );
        project.validate().unwrap();

        let insert = |material: &str, parameters: &str| {
            format!(
                r#"{{"edit":"insert_smart_material","texture_set":"Body","smart_material":"{material}","parameters":{parameters}}}"#
            )
        };
        assert_eq!(
            refused(&mut project, &insert("Chrome", "{}")),
            AutomationError::UnknownSmartMaterial("Chrome".to_owned())
        );
        assert!(matches!(
            refused(&mut project, &insert("Rusty iron", r#"{"Shine":1}"#)),
            AutomationError::InvalidValue(reason)
                if reason.starts_with("`Rusty iron` has no parameter `Shine`; it has `Iron color`")
        ));
        for value in ["2.0", r#""red""#, "[1,2,3]"] {
            let parameters = format!(r#"{{"Iron roughness":{value}}}"#);
            assert!(
                matches!(
                    refused(&mut project, &insert("Rusty iron", &parameters)),
                    AutomationError::InvalidValue(_)
                ),
                "{value}"
            );
        }

        // A project smart material wins over a built-in of the same name.
        let mut custom =
            SmartMaterial::from_layer(&MaterialLayer::new_paint(LayerId(1), "Mine"), "Rusty iron");
        custom.parameters.clear();
        project.smart_materials.push(custom);
        apply_automation_edit(&mut project, &edit(&insert("Rusty iron", "{}"))).unwrap();
        assert_eq!(project.texture_sets[0].layers.last().unwrap().name, "Mine");
        assert_eq!(
            library_summary(&project).smart_materials[0].origin,
            "project"
        );

        // A smart mask replaces the generator and keeps painted strokes.
        apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"paint_mask","texture_set":"Body","layer":"Base",
                    "stroke":{"points":[[0.5,0.5]],"radius_pixels":2,"value":0,"opacity":1}}"#,
            ),
        )
        .unwrap();
        let outcome = apply_automation_edit(
            &mut project,
            &edit(
                r#"{"edit":"apply_smart_mask","texture_set":"Body","layer":"Base","smart_mask":"Worn edges"}"#,
            ),
        )
        .unwrap();
        assert_eq!(
            outcome.summary,
            "applied smart mask `Worn edges` to layer 1 `Base`"
        );
        let worn_edges = SmartMask::built_in().remove(0);
        let base = project.texture_sets[0].layer(LayerId(1)).unwrap();
        assert_eq!(base.mask_generator, worn_edges.generator);
        assert_eq!(base.mask_strokes.len(), 1);
        assert_eq!(
            refused(
                &mut project,
                r#"{"edit":"apply_smart_mask","texture_set":"Body","layer":"Base","smart_mask":"Moss"}"#
            ),
            AutomationError::UnknownSmartMask("Moss".to_owned())
        );
    }
}
