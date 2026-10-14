//! Material evolution controls (for example clean to worn): one slider on a
//! layer, usually a smart material's group, that moves several properties
//! of the layers inside it, each through its own range, as one undoable
//! edit.

use serde::{Deserialize, Serialize};

use crate::library::{ExposedProperty, ParameterValue, valid_name};
use crate::{EditCommand, LayerId, MaterialLayer, MaterialProject, TextureSetId};

/// Controls one layer may carry.
pub const MAX_EVOLUTIONS: usize = 8;
/// Properties one control may drive.
pub const MAX_EVOLUTION_DRIVES: usize = 64;

/// One property an evolution control moves: from its value at `0` to its
/// value at `1`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvolutionDrive {
    pub layer: LayerId,
    pub property: ExposedProperty,
    pub from: ParameterValue,
    pub to: ParameterValue,
}

impl EvolutionDrive {
    /// The value at `amount` (`0..=1`): scalars and colours mix linearly.
    pub fn value_at(&self, amount: f32) -> Option<ParameterValue> {
        let amount = amount.clamp(0.0, 1.0);
        match (self.from, self.to) {
            (ParameterValue::Scalar(from), ParameterValue::Scalar(to)) => {
                Some(ParameterValue::Scalar(from + (to - from) * amount))
            }
            (ParameterValue::Color(from), ParameterValue::Color(to)) => {
                Some(ParameterValue::Color(std::array::from_fn(|channel| {
                    let (from, to) = (f32::from(from[channel]), f32::from(to[channel]));
                    (from + (to - from) * amount).round().clamp(0.0, 255.0) as u8
                })))
            }
            _ => None,
        }
    }

    fn is_valid(&self) -> bool {
        let scalar = |value: f32| value.is_finite() && (0.0..=1.0).contains(&value);
        match (self.from, self.to) {
            (ParameterValue::Scalar(from), ParameterValue::Scalar(to)) => {
                scalar(from) && scalar(to)
            }
            (ParameterValue::Color(_), ParameterValue::Color(_)) => true,
            _ => false,
        }
    }
}

/// A named control that moves several properties of a layer tree at once.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evolution {
    pub name: String,
    /// Current position, `0..=1`.
    pub amount: f32,
    pub drives: Vec<EvolutionDrive>,
}

fn find_mut(root: &mut MaterialLayer, id: LayerId) -> Option<&mut MaterialLayer> {
    if root.id == id {
        return Some(root);
    }
    root.children
        .iter_mut()
        .flatten()
        .find_map(|child| find_mut(child, id))
}

impl Evolution {
    /// A control at `1` whose drives move every exposable property that
    /// differs between two states of the same layer tree: `start` is the
    /// look at `0`, `end` the look at `1`. Layers are matched by id.
    pub fn between(name: impl Into<String>, start: &MaterialLayer, end: &MaterialLayer) -> Self {
        let mut drives = Vec::new();
        let ends = end.walk();
        for before in start.walk() {
            let Some(after) = ends.iter().find(|layer| layer.id == before.id) else {
                continue;
            };
            for property in ExposedProperty::ALL {
                if let (Some(from), Some(to)) = (property.read(before), property.read(after))
                    && from != to
                    && drives.len() < MAX_EVOLUTION_DRIVES
                {
                    drives.push(EvolutionDrive {
                        layer: before.id,
                        property,
                        from,
                        to,
                    });
                }
            }
        }
        Self {
            name: name.into(),
            amount: 1.0,
            drives,
        }
    }

    /// Moves every drive to `amount` within `root`'s tree. False, leaving
    /// `root` unchanged, when a drive no longer fits its layer.
    pub fn apply(&self, root: &mut MaterialLayer, amount: f32) -> bool {
        let mut evolved = root.clone();
        for drive in &self.drives {
            let applied = find_mut(&mut evolved, drive.layer)
                .zip(drive.value_at(amount))
                .and_then(|(layer, value)| drive.property.access(layer, Some(value)));
            if applied.is_none() {
                return false;
            }
        }
        *root = evolved;
        true
    }

    /// Valid on `root`: a name, an amount in range, and drives of matching
    /// value types on properties the layers of `root`'s tree have.
    pub(crate) fn is_valid(&self, root: &MaterialLayer) -> bool {
        let layers = root.walk();
        valid_name(&self.name)
            && self.amount.is_finite()
            && (0.0..=1.0).contains(&self.amount)
            && self.drives.len() <= MAX_EVOLUTION_DRIVES
            && self.drives.iter().all(|drive| {
                drive.is_valid()
                    && layers
                        .iter()
                        .find(|layer| layer.id == drive.layer)
                        .and_then(|layer| drive.property.read(layer))
                        .is_some_and(|current| {
                            std::mem::discriminant(&current) == std::mem::discriminant(&drive.from)
                        })
            })
    }
}

impl MaterialProject {
    /// Moves evolution control `index` of a layer to `amount`: the control
    /// and every layer it drives change together, as one undoable edit.
    /// Driven fill values of material instances become overrides. `None`
    /// when nothing changes or the control does not apply.
    pub fn evolve_command(
        &self,
        set_id: TextureSetId,
        layer_id: LayerId,
        index: usize,
        amount: f32,
    ) -> Option<EditCommand> {
        if !amount.is_finite() {
            return None;
        }
        let amount = amount.clamp(0.0, 1.0);
        let root = self
            .texture_sets
            .iter()
            .find(|set| set.id == set_id)?
            .layer(layer_id)?;
        let evolution = root.evolutions.get(index)?;
        let mut evolved = root.clone();
        if !evolution.apply(&mut evolved, amount) {
            return None;
        }
        evolved.evolutions[index].amount = amount;
        let commands: Vec<EditCommand> = root
            .walk()
            .into_iter()
            .zip(evolved.walk())
            .filter_map(|(original, changed)| {
                let before = original.settings();
                let after = self.track_instance_edit(changed.settings());
                (before != after).then(|| EditCommand::LayerSettings {
                    texture_set_id: set_id,
                    layer_id: original.id,
                    before: Box::new(before),
                    after: Box::new(after),
                })
            })
            .collect();
        (!commands.is_empty()).then_some(EditCommand::Batch(commands))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::{
        EditHistory, LayerFill, MaskGenerator, MaterialChannel, MaterialResource, ProjectError,
        SmartMaterial, TextureResolution, TextureSet,
    };

    fn project_with(root: MaterialLayer) -> MaterialProject {
        MaterialProject {
            texture_sets: vec![TextureSet {
                user_channels: Default::default(),
                stencils: Default::default(),
                id: TextureSetId(1),
                name: "Body".to_owned(),
                material_slot: "Body".to_owned(),
                resolution: TextureResolution {
                    width: 8,
                    height: 8,
                },
                channels: BTreeSet::from([MaterialChannel::BaseColor]),
                layers: vec![root],
                height_strength: crate::DEFAULT_HEIGHT_STRENGTH,
                export: None,
            }],
            active_texture_set: Some(TextureSetId(1)),
            ..MaterialProject::empty()
        }
    }

    fn generator_amount(layer: &MaterialLayer) -> f32 {
        match layer.mask_generator {
            Some(MaskGenerator::EdgeWear(settings)) => settings.amount,
            Some(MaskGenerator::Dirt(settings)) => settings.amount,
            Some(MaskGenerator::Noise(settings)) => settings.amount,
            _ => f32::NAN,
        }
    }

    #[test]
    fn tc_0284_evolution_controls_record_ranges_and_move_them_as_one_undo_step() {
        let worn = SmartMaterial::built_in().remove(0);
        let mut next = 10;
        let mut root = worn.instantiate(|| {
            next += 1;
            LayerId(next)
        });
        // The built-in carries a wear control whose drives follow the
        // instance's fresh ids.
        assert_eq!(root.evolutions.len(), 1);
        assert_eq!(root.evolutions[0].name, "Wear");
        assert!(
            root.evolutions[0]
                .drives
                .iter()
                .all(|drive| root.walk().iter().any(|layer| layer.id == drive.layer))
        );

        // Record a second control from two states of the tree.
        let start = root.clone();
        let children = root.children.as_mut().unwrap();
        children[0].opacity = 0.4;
        if let Some(fill) = &mut children[2].fill {
            fill.color = [200, 100, 0, 255];
        }
        let grime = Evolution::between("Grime colour", &start, &root);
        assert_eq!(grime.amount, 1.0);
        assert_eq!(grime.drives.len(), 2);
        assert!(
            grime
                .drives
                .iter()
                .any(|drive| drive.property == ExposedProperty::Opacity)
        );
        root.evolutions.push(grime);
        let mut project = project_with(root.clone());
        project.validate().unwrap();

        // Halfway mixes every drive; one command, one undo.
        let mut history = EditHistory::default();
        let command = project
            .evolve_command(TextureSetId(1), root.id, 1, 0.5)
            .unwrap();
        assert!(history.apply(&mut project, command));
        let layers = &project.texture_sets[0].layers[0];
        let children = layers.children.as_ref().unwrap();
        assert!((children[0].opacity - 0.7).abs() < 1e-6);
        let base = start.children.as_ref().unwrap()[2].fill.unwrap().color;
        let mixed = children[2].fill.unwrap().color;
        for channel in 0..3 {
            let expected = (f32::from(base[channel]) + f32::from([200_u8, 100, 0][channel])) / 2.0;
            assert!((f32::from(mixed[channel]) - expected).abs() <= 0.5);
        }
        assert_eq!(layers.evolutions[1].amount, 0.5);
        assert!(history.undo(&mut project));
        assert_eq!(project.texture_sets[0].layers[0], root);

        // The wear control moves generator amounts; a drag merges.
        let wear_layers = |project: &MaterialProject| {
            let root = &project.texture_sets[0].layers[0];
            let children = root.children.as_ref().unwrap();
            (
                generator_amount(&children[1]),
                generator_amount(&children[2]),
            )
        };
        let (paint_start, _) = wear_layers(&project);
        let mut history = EditHistory::default();
        for amount in [0.6, 0.8, 1.0] {
            let command = project
                .evolve_command(TextureSetId(1), root.id, 0, amount)
                .unwrap();
            assert!(history.apply_merging(&mut project, command));
        }
        history.end_merge();
        let (paint_end, grime_end) = wear_layers(&project);
        assert!(paint_end > paint_start, "more wear exposes more metal");
        let drive = &root.evolutions[0].drives[0];
        assert_eq!(drive.value_at(1.0), Some(drive.to));
        assert!((grime_end - 0.75).abs() < 1e-6);
        assert!(history.undo(&mut project));
        assert_eq!(
            project.texture_sets[0].layers[0], root,
            "the drag is one step"
        );
        assert!(!history.can_undo());

        // Unchanged amounts make no command; non-finite amounts are refused.
        assert!(
            project
                .evolve_command(TextureSetId(1), root.id, 0, root.evolutions[0].amount)
                .is_none()
        );
        assert!(
            project
                .evolve_command(TextureSetId(1), root.id, 0, f32::NAN)
                .is_none()
        );
        assert!(
            project
                .evolve_command(TextureSetId(1), root.id, 5, 0.5)
                .is_none()
        );
    }

    #[test]
    fn tc_0285_evolution_validation_instances_and_smart_material_capture() {
        let mut fill = MaterialLayer::new_fill(LayerId(2), "Coat", LayerFill::default());
        fill.channels.insert(MaterialChannel::Height);
        let mut root = MaterialLayer::new_group(LayerId(1), "Material");
        root.children = Some(vec![fill]);
        let start = root.clone();
        if let Some(fill) = &mut root.children.as_mut().unwrap()[0].fill {
            fill.roughness = 0.9;
        }
        root.evolutions
            .push(Evolution::between("Rough", &start, &root));
        let mut project = project_with(root.clone());
        project.validate().unwrap();

        // Broken controls are refused by validation.
        let refused = |edit: &dyn Fn(&mut Evolution)| {
            let mut broken = project.clone();
            edit(&mut broken.texture_sets[0].layers[0].evolutions[0]);
            broken.validate()
        };
        let invalid = Err(ProjectError::InvalidLayerSettings(LayerId(1)));
        assert_eq!(refused(&|evolution| evolution.amount = 2.0), invalid);
        assert_eq!(refused(&|evolution| evolution.name.clear()), invalid);
        assert_eq!(
            refused(&|evolution| evolution.drives[0].layer = LayerId(9)),
            invalid
        );
        assert_eq!(
            refused(&|evolution| evolution.drives[0].to = ParameterValue::Color([1, 2, 3])),
            invalid
        );
        assert_eq!(
            refused(&|evolution| evolution.drives[0].to = ParameterValue::Scalar(4.0)),
            invalid
        );

        // A driven fill value of a material instance becomes an override.
        let rubber = MaterialResource::built_in()
            .into_iter()
            .find(|material| material.name == "Rubber")
            .unwrap();
        let (master, add) = project.add_material_command(rubber).unwrap();
        let mut history = EditHistory::default();
        assert!(history.apply(&mut project, add));
        let link = project
            .link_material_command(TextureSetId(1), LayerId(2), master)
            .unwrap();
        assert!(history.apply(&mut project, link));
        let command = project
            .evolve_command(TextureSetId(1), LayerId(1), 0, 0.0)
            .unwrap();
        assert!(history.apply(&mut project, command));
        let coat = project.texture_sets[0].layer(LayerId(2)).unwrap();
        assert_eq!(coat.fill.unwrap().roughness, 0.5);
        assert!(
            coat.material_link
                .as_ref()
                .unwrap()
                .overrides
                .contains(&crate::InstanceField::Roughness)
        );
        project.validate().unwrap();

        // Smart materials keep controls and remap them with fresh ids;
        // controls in a saved smart material are validated too.
        let smart = SmartMaterial::from_layer(&root, "Coat");
        smart.validate().unwrap();
        let mut next = 40;
        let instance = smart.instantiate(|| {
            next += 1;
            LayerId(next)
        });
        assert_eq!(instance.id, LayerId(41));
        assert_eq!(instance.evolutions[0].drives[0].layer, LayerId(42));
        let mut broken = smart.clone();
        broken.root.evolutions[0].drives[0].layer = LayerId(77);
        assert!(broken.validate().is_err());
        for material in SmartMaterial::built_in() {
            assert_eq!(material.root.evolutions.len(), 1, "{}", material.name);
            assert!(material.root.evolutions[0].drives.len() >= 3);
        }
    }
}
