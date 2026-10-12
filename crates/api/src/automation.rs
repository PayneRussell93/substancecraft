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
    BlendMode, EditCommand, EditHistory, LayerFill, LayerId, LayerSlot, MaterialChannel,
    MaterialId, MaterialLayer, MaterialLink, MaterialProject, MaterialResource, TextureSet,
    TextureSetId, mask_drivers,
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
}

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
    }
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
}
