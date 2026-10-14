//! Project templates: starting points for new projects. A template holds
//! one texture set's settings (resolution, channels, height strength, export
//! choice) and a starter layer stack, without a mesh; a new project gets one
//! texture set per material slot of its mesh.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::library::{LibraryError, SmartMaterial, valid_name};
use crate::resources::ResourceProvenance;
use crate::{
    DEFAULT_HEIGHT_STRENGTH, ExportPreset, LayerFill, LayerId, MaterialChannel, MaterialLayer,
    MaterialProject, TextureResolution, TextureSet, TextureSetExport, TextureSetId,
    valid_height_strength, valid_layer_strokes,
};

/// A starting point for new projects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectTemplate {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub resolution: TextureResolution,
    pub channels: BTreeSet<MaterialChannel>,
    #[serde(default = "default_height_strength")]
    pub height_strength: f32,
    /// A built-in export preset, format, and UDIM choice for every set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export: Option<TextureSetExport>,
    /// Starter layers, bottom first; every set gets a copy with fresh ids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<MaterialLayer>,
    #[serde(default)]
    pub provenance: ResourceProvenance,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

fn default_height_strength() -> f32 {
    DEFAULT_HEIGHT_STRENGTH
}

/// Copies of `layers` with ids counting up from `next`; anchors and
/// evolution drives follow the layers they point at.
fn renumbered(layers: &[MaterialLayer], next: &mut u64) -> Vec<MaterialLayer> {
    let mut mapping = BTreeMap::new();
    for layer in layers.iter().flat_map(MaterialLayer::walk) {
        mapping.insert(layer.id, LayerId(*next));
        *next += 1;
    }
    fn remap(layer: &mut MaterialLayer, mapping: &BTreeMap<LayerId, LayerId>) {
        if let Some(id) = mapping.get(&layer.id) {
            layer.id = *id;
        }
        if let Some(anchor) = &mut layer.mask_anchor
            && let Some(id) = mapping.get(&anchor.layer)
        {
            anchor.layer = *id;
        }
        for drive in layer
            .evolutions
            .iter_mut()
            .flat_map(|evolution| &mut evolution.drives)
        {
            if let Some(id) = mapping.get(&drive.layer) {
                drive.layer = *id;
            }
        }
        for child in layer.children.iter_mut().flatten() {
            remap(child, mapping);
        }
    }
    let mut copies = layers.to_vec();
    for layer in &mut copies {
        remap(layer, &mapping);
    }
    copies
}

impl ProjectTemplate {
    /// A template from a texture set: its settings and layers. Polygon
    /// masks (tied to one mesh) and links to project materials (which the
    /// template does not carry) are dropped, and so is an export choice
    /// naming a project's own preset.
    pub fn from_texture_set(set: &TextureSet, name: impl Into<String>) -> Self {
        fn strip(layer: &mut MaterialLayer) {
            layer.polygon_mask = None;
            layer.material_link = None;
            for child in layer.children.iter_mut().flatten() {
                strip(child);
            }
        }
        let mut layers = set.layers.clone();
        layers.iter_mut().for_each(strip);
        let built_in = |export: &&TextureSetExport| {
            ExportPreset::built_in()
                .iter()
                .any(|preset| preset.name == export.preset)
        };
        Self {
            name: name.into(),
            description: String::new(),
            resolution: set.resolution,
            channels: set.channels.clone(),
            height_strength: set.height_strength,
            export: set.export.as_ref().filter(built_in).cloned(),
            layers,
            provenance: ResourceProvenance::default(),
            tags: Vec::new(),
        }
    }

    /// A new project with one texture set per name in `slots` (one named
    /// "Material" when empty), each with this template's settings and its
    /// own copy of the layers. The project has no mesh.
    pub fn instantiate(&self, slots: &[String]) -> MaterialProject {
        let slots = if slots.is_empty() {
            vec!["Material".to_owned()]
        } else {
            slots.to_vec()
        };
        let mut next_layer = 1;
        let texture_sets: Vec<TextureSet> = slots
            .iter()
            .enumerate()
            .map(|(index, slot)| TextureSet {
                user_channels: Default::default(),
                stencils: Default::default(),
                id: TextureSetId(index as u64 + 1),
                name: slot.clone(),
                material_slot: slot.clone(),
                resolution: self.resolution,
                channels: self.channels.clone(),
                layers: renumbered(&self.layers, &mut next_layer),
                height_strength: self.height_strength,
                export: self.export.clone(),
            })
            .collect();
        MaterialProject {
            active_texture_set: texture_sets.first().map(|set| set.id),
            texture_sets,
            ..MaterialProject::empty()
        }
    }

    pub fn validate(&self) -> Result<(), LibraryError> {
        if !valid_name(&self.name) {
            return Err(LibraryError::InvalidName(self.name.clone()));
        }
        let mut ids = BTreeSet::new();
        let layers_valid = self
            .layers
            .iter()
            .flat_map(MaterialLayer::walk)
            .all(|layer| {
                ids.insert(layer.id)
                    && layer.has_valid_settings()
                    && layer.has_valid_evolutions()
                    && layer.material_link.is_none()
                    && valid_layer_strokes(layer)
            });
        let export_valid = self.export.as_ref().is_none_or(|export| {
            crate::EXPORT_FILE_FORMATS.contains(&export.format.as_str())
                && ExportPreset::built_in()
                    .iter()
                    .any(|preset| preset.name == export.preset)
        });
        if self.resolution.validate().is_err()
            || !valid_height_strength(self.height_strength)
            || !layers_valid
            || !export_valid
        {
            return Err(LibraryError::InvalidLayers(self.name.clone()));
        }
        Ok(())
    }

    /// Original starting points for common pipelines.
    pub fn built_in() -> Vec<Self> {
        let channels = BTreeSet::from([
            MaterialChannel::BaseColor,
            MaterialChannel::Roughness,
            MaterialChannel::Metallic,
            MaterialChannel::Height,
            MaterialChannel::Normal,
            MaterialChannel::Emissive,
            MaterialChannel::Opacity,
            MaterialChannel::Id,
        ]);
        let base = MaterialLayer::new_fill(
            LayerId(1),
            "Base",
            LayerFill {
                color: [128, 128, 128, 255],
                roughness: 0.5,
                ..LayerFill::default()
            },
        );
        let export = |preset: &str, format: &str, udim: bool| {
            Some(TextureSetExport {
                preset: preset.to_owned(),
                format: format.to_owned(),
                udim,
            })
        };
        let template = |name: &str,
                        description: &str,
                        size: u32,
                        export: Option<TextureSetExport>,
                        layers: Vec<MaterialLayer>,
                        tags: &[&str]| Self {
            name: name.to_owned(),
            description: description.to_owned(),
            resolution: TextureResolution {
                width: size,
                height: size,
            },
            channels: channels.clone(),
            height_strength: DEFAULT_HEIGHT_STRENGTH,
            export,
            layers,
            provenance: crate::library::built_in_provenance(),
            tags: tags.iter().map(|tag| (*tag).to_owned()).collect(),
        };
        let mut worn = SmartMaterial::built_in()
            .into_iter()
            .next()
            .map(|material| material.root)
            .into_iter()
            .collect::<Vec<_>>();
        let mut next = 2;
        worn = renumbered(&worn, &mut next);
        vec![
            template(
                "Game asset",
                "2K metal/rough set with a base fill, exported as packed ORM PNGs.",
                2048,
                export("Metal/Rough (ORM)", "png", false),
                vec![base.clone()],
                &["game", "pbr"],
            ),
            template(
                "Unreal Engine asset",
                "2K set exported with Unreal's naming, DirectX normals, and packed ORM.",
                2048,
                export("Unreal Engine", "png", false),
                vec![base.clone()],
                &["game", "unreal"],
            ),
            template(
                "Unity URP asset",
                "2K set exported for URP Lit: metallic with smoothness, separate occlusion.",
                2048,
                export("Unity (URP Lit)", "png", false),
                vec![base.clone()],
                &["game", "unity"],
            ),
            template(
                "Film / VFX",
                "4K channels as separate half-float EXRs, one file set per UDIM tile.",
                4096,
                export("Separate channels", "exr", true),
                vec![base],
                &["film", "vfx", "udim"],
            ),
            template(
                "Worn metal prop",
                "Painted steel chipped at the edges with grime, with a Wear control.",
                2048,
                export("Metal/Rough (ORM)", "png", false),
                worn,
                &["game", "worn", "metal"],
            ),
        ]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tc_0303_templates_start_projects_with_a_set_per_material_slot() {
        let library = ProjectTemplate::built_in();
        assert_eq!(library.len(), 5);
        for template in &library {
            template.validate().unwrap();
            template.instantiate(&[]).validate().unwrap();
        }
        // One set per slot, each with its own layers under fresh ids.
        let worn = &library[4];
        let project = worn.instantiate(&["Body".to_owned(), "Trim".to_owned()]);
        project.validate().unwrap();
        assert_eq!(project.texture_sets.len(), 2);
        assert_eq!(project.texture_sets[1].material_slot, "Trim");
        assert_eq!(project.active_texture_set, Some(TextureSetId(1)));
        let ids: Vec<u64> = project
            .texture_sets
            .iter()
            .flat_map(|set| set.walk_layers())
            .map(|layer| layer.id.0)
            .collect();
        assert_eq!(ids, [1, 2, 3, 4, 5, 6, 7, 8]);
        // Each copy's wear control drives its own layers.
        let control = &project.texture_sets[1].layers[0].evolutions[0];
        assert!(control.drives.iter().all(|drive| drive.layer.0 >= 5));
        let film = &library[3];
        assert_eq!(
            film.instantiate(&[]).texture_sets[0]
                .export
                .as_ref()
                .unwrap()
                .format,
            "exr"
        );

        // Capturing a set drops mesh-specific selections, material links,
        // and project-only export presets.
        let mut set = project.texture_sets[0].clone();
        set.layers[0].polygon_mask = Some(crate::PolygonMask::default());
        set.export = Some(TextureSetExport {
            preset: "My studio preset".to_owned(),
            format: "png".to_owned(),
            udim: false,
        });
        let captured = ProjectTemplate::from_texture_set(&set, "Mine");
        assert_eq!(captured.layers[0].polygon_mask, None);
        assert_eq!(captured.export, None);
        captured.validate().unwrap();
        let mut broken = captured;
        broken.name.clear();
        assert!(broken.validate().is_err());
        broken.name = "Broken".to_owned();
        broken.resolution.width = 0;
        assert!(broken.validate().is_err());
    }
}
