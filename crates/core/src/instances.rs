//! Material instances: fill layers linked to a project material, their
//! *master*.
//!
//! A linked layer keeps the master's values resolved into its own fill and
//! channels, so bakes, exports, and background workers never need the
//! project. The link stays authoritative: every command that changes a
//! master re-resolves its instances in the same undo step, and the fields an
//! instance overrides keep the layer's own value.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    EditCommand, LayerFill, LayerId, LayerSettings, MaterialChannel, MaterialId, MaterialLayer,
    MaterialProject, MaterialResource, TextureSetId,
};

impl MaterialId {
    /// The id of materials outside a project (built-ins, unsaved captures).
    /// Project materials never use it.
    pub const UNASSIGNED: Self = Self(0);
}

/// A part of a material an instance can override. Each channel field covers
/// both the uniform value and the bitmap of that channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum InstanceField {
    BaseColor,
    Roughness,
    Metallic,
    Height,
    /// The normal bitmap (normals have no uniform value).
    Normal,
    Emissive,
    Opacity,
    /// Projection mode, transform, and physical tile size.
    Projection,
    /// Which channels the layer writes.
    Channels,
}

impl InstanceField {
    pub const ALL: [Self; 9] = [
        Self::BaseColor,
        Self::Roughness,
        Self::Metallic,
        Self::Height,
        Self::Normal,
        Self::Emissive,
        Self::Opacity,
        Self::Projection,
        Self::Channels,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::BaseColor => "Base Color",
            Self::Roughness => "Roughness",
            Self::Metallic => "Metallic",
            Self::Height => "Height",
            Self::Normal => "Normal",
            Self::Emissive => "Emissive",
            Self::Opacity => "Opacity",
            Self::Projection => "Projection",
            Self::Channels => "Channels",
        }
    }

    /// The field that holds `channel`'s value, if a material has one.
    pub const fn for_channel(channel: MaterialChannel) -> Option<Self> {
        match channel {
            MaterialChannel::BaseColor => Some(Self::BaseColor),
            MaterialChannel::Roughness => Some(Self::Roughness),
            MaterialChannel::Metallic => Some(Self::Metallic),
            MaterialChannel::Height => Some(Self::Height),
            MaterialChannel::Normal => Some(Self::Normal),
            MaterialChannel::Emissive => Some(Self::Emissive),
            MaterialChannel::Opacity => Some(Self::Opacity),
            MaterialChannel::Id | MaterialChannel::AmbientOcclusion => None,
        }
    }

    /// Copies this field from `source` into `target`.
    fn copy(
        self,
        source: (&LayerFill, &BTreeSet<MaterialChannel>),
        target: (&mut LayerFill, &mut BTreeSet<MaterialChannel>),
    ) {
        let (from, from_channels) = source;
        let (to, to_channels) = target;
        match self {
            Self::BaseColor => {
                to.color = from.color;
                to.textures.base_color = from.textures.base_color;
            }
            Self::Roughness => {
                to.roughness = from.roughness;
                to.textures.roughness = from.textures.roughness;
            }
            Self::Metallic => {
                to.metallic = from.metallic;
                to.textures.metallic = from.textures.metallic;
            }
            Self::Height => {
                to.height = from.height;
                to.textures.height = from.textures.height;
            }
            Self::Normal => to.textures.normal = from.textures.normal,
            Self::Emissive => {
                to.emissive = from.emissive;
                to.textures.emissive = from.textures.emissive;
            }
            Self::Opacity => {
                to.material_opacity = from.material_opacity;
                to.textures.opacity = from.textures.opacity;
            }
            Self::Projection => to.projection = from.projection,
            Self::Channels => to_channels.clone_from(from_channels),
        }
    }

    /// True when this field of `a` and `b` holds different values.
    fn differs(self, a: &LayerSettings, b: &LayerSettings) -> bool {
        let (Some(a_fill), Some(b_fill)) = (a.fill, b.fill) else {
            return a.fill != b.fill;
        };
        let mut fill = a_fill;
        let mut channels = a.channels.clone();
        self.copy((&b_fill, &b.channels), (&mut fill, &mut channels));
        fill != a_fill || channels != a.channels
    }
}

/// Links a fill layer to a project material (its master). The layer's fill
/// and channels follow the master, except for the fields in `overrides`,
/// which keep the layer's own values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialLink {
    pub material: MaterialId,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub overrides: BTreeSet<InstanceField>,
}

impl MaterialLink {
    pub fn new(material: MaterialId) -> Self {
        Self {
            material,
            overrides: BTreeSet::new(),
        }
    }

    /// `settings` as an instance of `master` through this link: overridden
    /// fields keep their values, every other field comes from the master.
    pub fn resolve(&self, master: &MaterialResource, settings: &LayerSettings) -> LayerSettings {
        let own = settings.fill.unwrap_or(master.fill);
        let mut fill = master.fill;
        let mut channels = master.channels.clone();
        for field in &self.overrides {
            field.copy((&own, &settings.channels), (&mut fill, &mut channels));
        }
        LayerSettings {
            fill: Some(fill),
            channels,
            material_link: Some(self.clone()),
            ..settings.clone()
        }
    }

    /// This link with an override added for every field in which
    /// `settings` no longer matches what the master gives it.
    pub fn track_overrides(&self, master: &MaterialResource, settings: &LayerSettings) -> Self {
        let resolved = self.resolve(master, settings);
        let mut link = self.clone();
        for field in InstanceField::ALL {
            if field.differs(settings, &resolved) {
                link.overrides.insert(field);
            }
        }
        link
    }
}

impl MaterialProject {
    /// The project material with `id`.
    pub fn material(&self, id: MaterialId) -> Option<&MaterialResource> {
        self.materials.iter().find(|material| material.id == id)
    }

    /// An id no project material uses.
    pub fn next_material_id(&self) -> MaterialId {
        MaterialId(
            self.materials
                .iter()
                .map(|material| material.id.0)
                .max()
                .unwrap_or(0)
                + 1,
        )
    }

    /// Every layer linked to `material`, as (texture set, layer) ids.
    pub fn material_instances(&self, material: MaterialId) -> Vec<(TextureSetId, LayerId)> {
        self.texture_sets
            .iter()
            .flat_map(|set| {
                set.walk_layers()
                    .into_iter()
                    .filter(|layer| {
                        layer
                            .material_link
                            .as_ref()
                            .is_some_and(|link| link.material == material)
                    })
                    .map(|layer| (set.id, layer.id))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn find_layer_ref(&self, set_id: TextureSetId, layer_id: LayerId) -> Option<&MaterialLayer> {
        self.texture_sets
            .iter()
            .find(|set| set.id == set_id)?
            .layer(layer_id)
    }

    /// A settings edit of one layer, or `None` when nothing changes.
    fn settings_command(
        set_id: TextureSetId,
        layer: &MaterialLayer,
        after: LayerSettings,
    ) -> Option<EditCommand> {
        let before = layer.settings();
        (before != after).then(|| EditCommand::LayerSettings {
            texture_set_id: set_id,
            layer_id: layer.id,
            before: Box::new(before),
            after: Box::new(after),
        })
    }

    /// Adds `material` to the project as a new master with a fresh id.
    /// Returns the id and the undoable command, or `None` when the material
    /// is invalid.
    pub fn add_material_command(
        &self,
        material: MaterialResource,
    ) -> Option<(MaterialId, EditCommand)> {
        material.validate().ok()?;
        let id = self.next_material_id();
        let mut after = self.materials.clone();
        after.push(MaterialResource { id, ..material });
        Some((
            id,
            EditCommand::Materials {
                before: self.materials.clone(),
                after,
            },
        ))
    }

    /// Makes a layer an instance of `material`: its fill and channels take
    /// the master's values, with no overrides.
    pub fn link_material_command(
        &self,
        set_id: TextureSetId,
        layer_id: LayerId,
        material: MaterialId,
    ) -> Option<EditCommand> {
        let master = self.material(material)?;
        let layer = self
            .find_layer_ref(set_id, layer_id)
            .filter(|layer| !layer.is_group())?;
        let after = MaterialLink::new(material).resolve(master, &layer.settings());
        Self::settings_command(set_id, layer, after)
    }

    /// Adds `material` as a new master and makes a layer its instance, as
    /// one step. Returns the new id and the command.
    pub fn link_new_material_command(
        &self,
        set_id: TextureSetId,
        layer_id: LayerId,
        material: MaterialResource,
    ) -> Option<(MaterialId, EditCommand)> {
        let layer = self
            .find_layer_ref(set_id, layer_id)
            .filter(|layer| !layer.is_group())?;
        let (id, add) = self.add_material_command(material)?;
        let EditCommand::Materials { after, .. } = &add else {
            return None;
        };
        let master = after.iter().find(|material| material.id == id)?;
        let linked = MaterialLink::new(id).resolve(master, &layer.settings());
        let mut commands = vec![add.clone()];
        commands.extend(Self::settings_command(set_id, layer, linked));
        Some((id, EditCommand::Batch(commands)))
    }

    /// Detaches an instance from its master, keeping its current values.
    pub fn unlink_material_command(
        &self,
        set_id: TextureSetId,
        layer_id: LayerId,
    ) -> Option<EditCommand> {
        let layer = self.find_layer_ref(set_id, layer_id)?;
        layer.material_link.as_ref()?;
        let after = LayerSettings {
            material_link: None,
            ..layer.settings()
        };
        Self::settings_command(set_id, layer, after)
    }

    /// Drops an instance's override of `field`, taking the master's value
    /// again.
    pub fn reset_override_command(
        &self,
        set_id: TextureSetId,
        layer_id: LayerId,
        field: InstanceField,
    ) -> Option<EditCommand> {
        let layer = self.find_layer_ref(set_id, layer_id)?;
        let mut link = layer.material_link.clone()?;
        if !link.overrides.remove(&field) {
            return None;
        }
        let master = self.material(link.material)?;
        Self::settings_command(set_id, layer, link.resolve(master, &layer.settings()))
    }

    /// The settings to record for an inspector edit that produced `edited`:
    /// an instance gains an override for every linked field the edit
    /// changed, and a layer that is no longer a fill (or whose master is
    /// gone) is unlinked. Other layers pass through unchanged.
    pub fn track_instance_edit(&self, edited: LayerSettings) -> LayerSettings {
        let Some(link) = &edited.material_link else {
            return edited;
        };
        match self.material(link.material) {
            Some(master) if edited.fill.is_some() => {
                let link = link.track_overrides(master, &edited);
                LayerSettings {
                    material_link: Some(link),
                    ..edited
                }
            }
            _ => LayerSettings {
                material_link: None,
                ..edited
            },
        }
    }

    /// Replaces the project's materials with `after` as one undoable step:
    /// every instance is re-resolved against its (possibly edited) master,
    /// and instances of removed materials are unlinked, keeping their
    /// values. `None` when nothing changes or `after` is invalid (a bad
    /// material, or a missing or duplicated id).
    pub fn replace_materials_command(&self, after: Vec<MaterialResource>) -> Option<EditCommand> {
        if after == self.materials {
            return None;
        }
        let mut ids = BTreeSet::new();
        for material in &after {
            material.validate().ok()?;
            if material.id == MaterialId::UNASSIGNED || !ids.insert(material.id) {
                return None;
            }
        }
        let mut commands = vec![EditCommand::Materials {
            before: self.materials.clone(),
            after: after.clone(),
        }];
        for set in &self.texture_sets {
            for layer in set.walk_layers() {
                let Some(link) = &layer.material_link else {
                    continue;
                };
                let settings = layer.settings();
                let resolved = match after.iter().find(|material| material.id == link.material) {
                    Some(master) => link.resolve(master, &settings),
                    None => LayerSettings {
                        material_link: None,
                        ..settings
                    },
                };
                commands.extend(Self::settings_command(set.id, layer, resolved));
            }
        }
        Some(EditCommand::Batch(commands))
    }

    /// Replaces one project material (matched by id) as with
    /// [`Self::replace_materials_command`].
    pub fn edit_material_command(&self, material: MaterialResource) -> Option<EditCommand> {
        let mut after = self.materials.clone();
        let id = material.id;
        *after.iter_mut().find(|existing| existing.id == id)? = material;
        self.replace_materials_command(after)
    }

    /// Removes a project material; its instances keep their values as
    /// plain fills.
    pub fn remove_material_command(&self, id: MaterialId) -> Option<EditCommand> {
        let mut after = self.materials.clone();
        let before = after.len();
        after.retain(|material| material.id != id);
        (after.len() < before)
            .then(|| self.replace_materials_command(after))
            .flatten()
    }
}

/// Joins `next`, a batch of layer-settings edits applied right after `last`
/// (the frames of an evolution slider drag), into `last`: every layer's edit
/// keeps its earliest `before` and takes the latest `after`. False, leaving
/// `last` untouched, when either holds other commands or `next` does not
/// pick up where `last` left a layer.
pub(crate) fn merge_settings_batches(last: &mut Vec<EditCommand>, next: &[EditCommand]) -> bool {
    let settings_only = |commands: &[EditCommand]| {
        !commands.is_empty()
            && commands
                .iter()
                .all(|command| matches!(command, EditCommand::LayerSettings { .. }))
    };
    if !settings_only(last) || !settings_only(next) {
        return false;
    }
    let find = |commands: &[EditCommand], set: TextureSetId, layer: LayerId| {
        commands.iter().position(|command| {
            matches!(command, EditCommand::LayerSettings { texture_set_id, layer_id, .. }
                if *texture_set_id == set && *layer_id == layer)
        })
    };
    for command in next {
        if let EditCommand::LayerSettings {
            texture_set_id,
            layer_id,
            before,
            ..
        } = command
            && let Some(index) = find(last, *texture_set_id, *layer_id)
            && let EditCommand::LayerSettings { after, .. } = &last[index]
            && after != before
        {
            return false;
        }
    }
    for command in next {
        let EditCommand::LayerSettings {
            texture_set_id,
            layer_id,
            after,
            ..
        } = command
        else {
            continue;
        };
        match find(last, *texture_set_id, *layer_id) {
            Some(index) => {
                if let EditCommand::LayerSettings {
                    after: last_after, ..
                } = &mut last[index]
                {
                    last_after.clone_from(after);
                }
            }
            None => last.push(command.clone()),
        }
    }
    true
}

/// Joins `next`, a material edit applied right after `last`, into `last`.
/// Both must be batches that start with a materials replacement, `next`
/// starting from where `last` ended. Each layer's settings edit keeps its
/// earliest `before` and takes the latest `after`. False (leaving `last`
/// untouched) when they do not join.
pub(crate) fn merge_material_edits(last: &mut Vec<EditCommand>, next: &[EditCommand]) -> bool {
    let (
        Some(EditCommand::Materials {
            after: last_after, ..
        }),
        Some(EditCommand::Materials { before, after }),
    ) = (last.first_mut(), next.first())
    else {
        return false;
    };
    if last_after != before {
        return false;
    }
    last_after.clone_from(after);
    for command in &next[1..] {
        let EditCommand::LayerSettings {
            texture_set_id,
            layer_id,
            after,
            ..
        } = command
        else {
            last.push(command.clone());
            continue;
        };
        let earlier = last.iter_mut().skip(1).find_map(|existing| match existing {
            EditCommand::LayerSettings {
                texture_set_id: set,
                layer_id: layer,
                after: existing_after,
                ..
            } if set == texture_set_id && layer == layer_id => Some(existing_after),
            _ => None,
        });
        match earlier {
            Some(existing_after) => existing_after.clone_from(after),
            None => last.push(command.clone()),
        }
    }
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{
        BakeContext, EditHistory, MaterialChannel, ProjectError, ResourceId, TextureResolution,
        TextureSet,
    };

    fn fill_layer(id: u64, name: &str) -> MaterialLayer {
        MaterialLayer::new_fill(LayerId(id), name, LayerFill::default())
    }

    fn set(id: u64, layers: Vec<MaterialLayer>) -> TextureSet {
        TextureSet {
            id: TextureSetId(id),
            name: format!("Set {id}"),
            material_slot: format!("Slot {id}"),
            resolution: TextureResolution {
                width: 8,
                height: 8,
            },
            channels: BTreeSet::from([
                MaterialChannel::BaseColor,
                MaterialChannel::Roughness,
                MaterialChannel::Metallic,
            ]),
            layers,
            height_strength: crate::DEFAULT_HEIGHT_STRENGTH,
            export: None,
        }
    }

    /// Two texture sets, three fill layers (one inside a group), and a
    /// steel master in the project.
    fn project_with_master() -> (MaterialProject, MaterialId) {
        let mut group = MaterialLayer::new_group(LayerId(3), "Group");
        group.children = Some(vec![fill_layer(4, "Nested")]);
        let mut project = MaterialProject {
            texture_sets: vec![
                set(1, vec![fill_layer(1, "Body"), fill_layer(2, "Trim"), group]),
                set(2, vec![fill_layer(5, "Other set")]),
            ],
            active_texture_set: Some(TextureSetId(1)),
            ..MaterialProject::empty()
        };
        let steel = MaterialResource::built_in()
            .into_iter()
            .find(|material| material.name == "Steel")
            .unwrap();
        let (id, add) = project.add_material_command(steel).unwrap();
        let mut history = EditHistory::default();
        assert!(history.apply(&mut project, add));
        (project, id)
    }

    fn fill_of(project: &MaterialProject, set: u64, layer: u64) -> LayerFill {
        project.texture_sets[(set - 1) as usize]
            .layer(LayerId(layer))
            .unwrap()
            .fill
            .unwrap()
    }

    fn roughness_of(project: &MaterialProject, set: u64) -> u8 {
        project.texture_sets[(set - 1) as usize]
            .bake_channel(MaterialChannel::Roughness, BakeContext::default())
            .unwrap()
            .pixel(2, 2)
            .unwrap()[0]
    }

    #[test]
    fn tc_0266_master_edits_update_every_instance_across_sets_in_one_undo_step() {
        let (mut project, steel) = project_with_master();
        assert_eq!(steel, MaterialId(1));
        assert_eq!(project.next_material_id(), MaterialId(2));
        let mut history = EditHistory::default();
        for (set, layer) in [(1, 1), (1, 4), (2, 5)] {
            let link = project
                .link_material_command(TextureSetId(set), LayerId(layer), steel)
                .unwrap();
            assert!(history.apply(&mut project, link));
        }
        project.validate().unwrap();
        assert_eq!(
            project.material_instances(steel),
            [
                (TextureSetId(1), LayerId(1)),
                (TextureSetId(1), LayerId(4)),
                (TextureSetId(2), LayerId(5)),
            ]
        );
        assert_eq!(fill_of(&project, 1, 4).metallic, 1.0);
        // Linking an already-linked layer again changes nothing.
        assert!(
            project
                .link_material_command(TextureSetId(1), LayerId(1), steel)
                .is_none()
        );

        // Editing the master updates every instance, in both sets, but not
        // the unlinked layer.
        let mut edited = project.material(steel).unwrap().clone();
        edited.fill.roughness = 0.9;
        edited.fill.color = [200, 20, 20, 255];
        let edit = project.edit_material_command(edited).unwrap();
        assert!(history.apply(&mut project, edit));
        for (set, layer) in [(1, 1), (1, 4), (2, 5)] {
            assert_eq!(fill_of(&project, set, layer).roughness, 0.9);
            assert_eq!(fill_of(&project, set, layer).color, [200, 20, 20, 255]);
        }
        assert_eq!(fill_of(&project, 1, 2), LayerFill::default());
        assert!(roughness_of(&project, 2).abs_diff(230) <= 1);
        project.validate().unwrap();

        // One undo restores the master and every instance.
        assert!(history.undo(&mut project));
        assert_eq!(project.material(steel).unwrap().fill.roughness, 0.35);
        for (set, layer) in [(1, 1), (1, 4), (2, 5)] {
            assert_eq!(fill_of(&project, set, layer).roughness, 0.35);
        }
        assert!(history.redo(&mut project));
        assert_eq!(fill_of(&project, 2, 5).roughness, 0.9);

        // An edit that changes nothing, and an invalid one, make no command.
        let same = project.material(steel).unwrap().clone();
        assert!(project.edit_material_command(same.clone()).is_none());
        let mut broken = same.clone();
        broken.fill.metallic = 2.0;
        assert!(project.edit_material_command(broken).is_none());
        let mut unassigned = same;
        unassigned.id = MaterialId::UNASSIGNED;
        assert!(
            project
                .replace_materials_command(vec![unassigned])
                .is_none()
        );
    }

    #[test]
    fn tc_0267_instance_overrides_are_explicit_undoable_and_survive_master_edits() {
        let (mut project, steel) = project_with_master();
        let mut history = EditHistory::default();
        for layer in [1, 2] {
            let link = project
                .link_material_command(TextureSetId(1), LayerId(layer), steel)
                .unwrap();
            assert!(history.apply(&mut project, link));
        }

        // An inspector edit of a linked field records an override for that
        // field only.
        let layer = project.texture_sets[0].layer(LayerId(1)).unwrap();
        let before = layer.settings();
        let mut edited = before.clone();
        if let Some(fill) = &mut edited.fill {
            fill.roughness = 0.1;
        }
        edited.opacity = 0.5; // not a material field
        let after = project.track_instance_edit(edited);
        let link = after.material_link.clone().unwrap();
        assert_eq!(link.overrides, BTreeSet::from([InstanceField::Roughness]));
        assert!(history.apply(
            &mut project,
            EditCommand::LayerSettings {
                texture_set_id: TextureSetId(1),
                layer_id: LayerId(1),
                before: Box::new(before),
                after: Box::new(after),
            },
        ));

        // A master edit keeps the override but takes the other fields.
        let mut edited = project.material(steel).unwrap().clone();
        edited.fill.roughness = 0.8;
        edited.fill.metallic = 0.25;
        let edit = project.edit_material_command(edited).unwrap();
        assert!(history.apply(&mut project, edit));
        let overridden = fill_of(&project, 1, 1);
        assert_eq!(
            (overridden.roughness, overridden.metallic),
            (0.1, 0.25),
            "override kept, other field follows"
        );
        let plain = fill_of(&project, 1, 2);
        assert_eq!((plain.roughness, plain.metallic), (0.8, 0.25));

        // Undoing the master edit, then the override, steps back in order.
        assert!(history.undo(&mut project));
        assert_eq!(fill_of(&project, 1, 1).metallic, 1.0);
        assert!(history.undo(&mut project));
        let layer = project.texture_sets[0].layer(LayerId(1)).unwrap();
        assert_eq!(layer.fill.unwrap().roughness, 0.35);
        assert!(layer.material_link.as_ref().unwrap().overrides.is_empty());
        assert!(history.redo(&mut project));
        assert!(history.redo(&mut project));

        // Resetting the override takes the master value, undoably.
        let reset = project
            .reset_override_command(TextureSetId(1), LayerId(1), InstanceField::Roughness)
            .unwrap();
        assert!(history.apply(&mut project, reset));
        assert_eq!(fill_of(&project, 1, 1).roughness, 0.8);
        assert!(
            project
                .reset_override_command(TextureSetId(1), LayerId(1), InstanceField::Roughness)
                .is_none(),
            "no override left to reset"
        );
        assert!(history.undo(&mut project));
        assert_eq!(fill_of(&project, 1, 1).roughness, 0.1);

        // Channel toggles and bitmaps are fields too.
        let layer = project.texture_sets[0].layer(LayerId(2)).unwrap();
        let mut edited = layer.settings();
        edited.channels.insert(MaterialChannel::Emissive);
        if let Some(fill) = &mut edited.fill {
            fill.textures.normal = Some(ResourceId(9));
        }
        let link = project.track_instance_edit(edited).material_link.unwrap();
        assert_eq!(
            link.overrides,
            BTreeSet::from([InstanceField::Normal, InstanceField::Channels])
        );

        // A layer turned into a paint layer is unlinked.
        let mut edited = layer.settings();
        edited.fill = None;
        assert_eq!(project.track_instance_edit(edited).material_link, None);
    }

    #[test]
    fn tc_0268_unlinking_removing_masters_and_validation_keep_instances_consistent() {
        let (mut project, steel) = project_with_master();
        let mut history = EditHistory::default();
        for layer in [1, 2] {
            let link = project
                .link_material_command(TextureSetId(1), LayerId(layer), steel)
                .unwrap();
            assert!(history.apply(&mut project, link));
        }

        // Unlinking keeps the values; later master edits pass it by.
        let unlink = project
            .unlink_material_command(TextureSetId(1), LayerId(2))
            .unwrap();
        assert!(history.apply(&mut project, unlink));
        assert!(
            project
                .unlink_material_command(TextureSetId(1), LayerId(2))
                .is_none()
        );
        let mut edited = project.material(steel).unwrap().clone();
        edited.fill.roughness = 0.6;
        let edit = project.edit_material_command(edited).unwrap();
        assert!(history.apply(&mut project, edit));
        assert_eq!(fill_of(&project, 1, 1).roughness, 0.6);
        assert_eq!(fill_of(&project, 1, 2).roughness, 0.35);

        // Removing the master unlinks its instances in the same step.
        let remove = project.remove_material_command(steel).unwrap();
        assert!(history.apply(&mut project, remove));
        assert!(project.materials.is_empty());
        let layer = project.texture_sets[0].layer(LayerId(1)).unwrap();
        assert_eq!(layer.material_link, None);
        assert_eq!(layer.fill.unwrap().roughness, 0.6);
        project.validate().unwrap();
        assert!(history.undo(&mut project));
        assert!(
            project.texture_sets[0]
                .layer(LayerId(1))
                .unwrap()
                .material_link
                .is_some()
        );
        project.validate().unwrap();

        // Validation: dangling links, links on groups or paint layers, and
        // duplicated or unassigned material ids are refused.
        let mut dangling = project.clone();
        dangling.texture_sets[0].layers[0].material_link = Some(MaterialLink::new(MaterialId(7)));
        assert_eq!(
            dangling.validate(),
            Err(ProjectError::MissingMaterial {
                layer: LayerId(1),
                material: MaterialId(7),
            })
        );
        let mut on_group = project.clone();
        on_group.texture_sets[0].layers[2].material_link = Some(MaterialLink::new(steel));
        assert_eq!(
            on_group.validate(),
            Err(ProjectError::InvalidLayerSettings(LayerId(3)))
        );
        let mut on_paint = project.clone();
        on_paint.texture_sets[0].layers[0].fill = None;
        assert_eq!(
            on_paint.validate(),
            Err(ProjectError::InvalidLayerSettings(LayerId(1)))
        );
        let mut duplicated = project.clone();
        let copy = duplicated.materials[0].clone();
        duplicated.materials.push(copy);
        assert!(matches!(
            duplicated.validate(),
            Err(ProjectError::InvalidMaterialId(_))
        ));
        let mut unassigned = project.clone();
        unassigned.materials[0].id = MaterialId::UNASSIGNED;
        assert!(matches!(
            unassigned.validate(),
            Err(ProjectError::InvalidMaterialId(_))
        ));

        // A layer's own fill can become a new master it is linked to, in
        // one step.
        let captured = MaterialResource::from_layer(
            project.texture_sets[0].layer(LayerId(2)).unwrap(),
            "Trim master",
        )
        .unwrap();
        let (trim, command) = project
            .link_new_material_command(TextureSetId(1), LayerId(2), captured)
            .unwrap();
        assert!(history.apply(&mut project, command));
        assert_eq!(trim, MaterialId(2));
        assert_eq!(
            project.material_instances(trim),
            [(TextureSetId(1), LayerId(2))]
        );
        project.validate().unwrap();
        assert!(history.undo(&mut project));
        assert!(project.material(trim).is_none());
        assert_eq!(
            project.texture_sets[0]
                .layer(LayerId(2))
                .unwrap()
                .material_link,
            None
        );

        // Groups cannot be linked; applying a material as a copy, saving a
        // smart material, and capturing a material all drop the link.
        assert!(
            project
                .link_material_command(TextureSetId(1), LayerId(3), steel)
                .is_none()
        );
        let mut layer = project.texture_sets[0].layers[0].clone();
        let smart = crate::SmartMaterial::from_layer(&layer, "Smart");
        assert_eq!(smart.root.material_link, None);
        smart.validate().unwrap();
        let mut linked_smart = smart;
        linked_smart.root.material_link = layer.material_link.clone();
        assert!(linked_smart.validate().is_err());
        let captured = MaterialResource::from_layer(&layer, "Copy").unwrap();
        assert_eq!(captured.id, MaterialId::UNASSIGNED);
        captured.apply_to(&mut layer);
        assert_eq!(layer.material_link, None);
    }

    #[test]
    fn tc_0269_master_slider_drags_merge_into_one_undo_step() {
        let (mut project, steel) = project_with_master();
        let mut history = EditHistory::default();
        for (set, layer) in [(1, 1), (2, 5)] {
            let link = project
                .link_material_command(TextureSetId(set), LayerId(layer), steel)
                .unwrap();
            assert!(history.apply(&mut project, link));
        }
        let undo_depth = |history: &EditHistory, project: &MaterialProject| {
            let mut probe = history.clone();
            let mut scratch = project.clone();
            let mut depth = 0;
            while probe.undo(&mut scratch) {
                depth += 1;
            }
            depth
        };
        let start = undo_depth(&history, &project);

        // Five frames of a roughness drag, then a release.
        for step in 1..=5 {
            let mut edited = project.material(steel).unwrap().clone();
            edited.fill.roughness = 0.35 + 0.1 * step as f32;
            let edit = project.edit_material_command(edited).unwrap();
            assert!(history.apply_merging(&mut project, edit));
        }
        history.end_merge();
        assert_eq!(undo_depth(&history, &project), start + 1);
        assert!((fill_of(&project, 2, 5).roughness - 0.85).abs() < 1e-6);

        // The merged step undoes to the values before the drag.
        assert!(history.undo(&mut project));
        assert_eq!(project.material(steel).unwrap().fill.roughness, 0.35);
        assert_eq!(fill_of(&project, 1, 1).roughness, 0.35);
        assert_eq!(fill_of(&project, 2, 5).roughness, 0.35);
        assert!(history.redo(&mut project));
        assert!((fill_of(&project, 1, 1).roughness - 0.85).abs() < 1e-6);

        // A new drag after the release is its own step.
        let mut edited = project.material(steel).unwrap().clone();
        edited.fill.metallic = 0.5;
        let edit = project.edit_material_command(edited).unwrap();
        assert!(history.apply_merging(&mut project, edit));
        assert_eq!(undo_depth(&history, &project), start + 2);
    }
}
