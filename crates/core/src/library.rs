//! Reusable library content: material resources (fill presets), smart
//! materials (layer trees with exposed parameters), smart masks (reusable
//! mask setups), and a searchable index of every asset a project can use.
//!
//! Built-in entries are original SubstanceCraft data, distributed under the
//! project licence.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::evolution::{Evolution, EvolutionDrive};
use crate::export::ExportPreset;
use crate::generators::{Dirt, EdgeWear, MaskGenerator, NoisePattern, Orientation, Thickness};
use crate::resources::{BitmapMask, ResourceProvenance};
use crate::{
    LayerFill, LayerFilter, LayerId, MaterialChannel, MaterialId, MaterialLayer, MaterialProject,
    ResourceId, valid_layer_strokes,
};

/// Licence recorded for built-in library entries.
pub const BUILT_IN_LICENSE: &str = "MIT OR Apache-2.0";
const BUILT_IN_AUTHOR: &str = "SubstanceCraft";
const MAX_LIBRARY_NAME_CHARS: usize = 64;

pub(crate) fn built_in_provenance() -> ResourceProvenance {
    ResourceProvenance {
        author: BUILT_IN_AUTHOR.to_owned(),
        license: BUILT_IN_LICENSE.to_owned(),
        origin: "Built-in".to_owned(),
    }
}

fn unassigned() -> MaterialId {
    MaterialId::UNASSIGNED
}

pub(crate) fn valid_name(name: &str) -> bool {
    let chars = name.trim().chars().count();
    chars > 0 && chars <= MAX_LIBRARY_NAME_CHARS
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LibraryError {
    #[error("`{0}` needs a name of 1 to {MAX_LIBRARY_NAME_CHARS} characters")]
    InvalidName(String),
    #[error("`{0}` has out-of-range material values")]
    InvalidValues(String),
    #[error("smart material `{0}` has an invalid layer tree")]
    InvalidLayers(String),
    #[error("smart material `{name}` exposes `{parameter}`, which no layer provides")]
    DanglingParameter { name: String, parameter: String },
    #[error("smart mask `{0}` has an invalid generator, bitmap projection, or filter")]
    InvalidMask(String),
}

// ---------------------------------------------------------------------------
// Material resources

/// A reusable material: the values (and optional bitmaps, projection, and
/// physical tile size) a fill layer writes, and which channels it enables.
/// A project material is also a master that fill layers can link to as
/// instances (see [`crate::MaterialLink`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialResource {
    /// Unique among the project's materials;
    /// [`MaterialId::UNASSIGNED`] for built-ins and unsaved captures.
    #[serde(default = "unassigned")]
    pub id: MaterialId,
    pub name: String,
    pub fill: LayerFill,
    pub channels: BTreeSet<MaterialChannel>,
    #[serde(default)]
    pub provenance: ResourceProvenance,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl MaterialResource {
    /// Starter physically based materials. Base colours are sRGB; metals use
    /// their reflectance colour as Base Color, per the metal/rough model.
    pub fn built_in() -> Vec<Self> {
        let material = |name: &str,
                        tags: &[&str],
                        color: [u8; 3],
                        metallic: f32,
                        roughness: f32,
                        opacity: f32| Self {
            id: MaterialId::UNASSIGNED,
            name: name.to_owned(),
            fill: LayerFill {
                color: [color[0], color[1], color[2], 255],
                roughness,
                metallic,
                material_opacity: opacity,
                ..LayerFill::default()
            },
            channels: if opacity < 1.0 {
                BTreeSet::from([
                    MaterialChannel::BaseColor,
                    MaterialChannel::Roughness,
                    MaterialChannel::Metallic,
                    MaterialChannel::Opacity,
                ])
            } else {
                BTreeSet::from([
                    MaterialChannel::BaseColor,
                    MaterialChannel::Roughness,
                    MaterialChannel::Metallic,
                ])
            },
            provenance: built_in_provenance(),
            tags: tags.iter().map(|tag| (*tag).to_owned()).collect(),
        };
        vec![
            material("Steel", &["metal"], [196, 199, 199], 1.0, 0.35, 1.0),
            material("Aluminium", &["metal"], [232, 234, 235], 1.0, 0.3, 1.0),
            material("Copper", &["metal"], [242, 186, 159], 1.0, 0.3, 1.0),
            material("Gold", &["metal"], [255, 214, 140], 1.0, 0.25, 1.0),
            material(
                "Chrome",
                &["metal", "polished"],
                [198, 198, 198],
                1.0,
                0.05,
                1.0,
            ),
            material("Rubber", &["dielectric"], [35, 35, 35], 0.0, 0.9, 1.0),
            material(
                "Glossy plastic",
                &["dielectric"],
                [200, 40, 40],
                0.0,
                0.25,
                1.0,
            ),
            material(
                "Matte paint",
                &["dielectric", "paint"],
                [70, 95, 130],
                0.0,
                0.7,
                1.0,
            ),
            material(
                "Concrete",
                &["dielectric", "rough"],
                [150, 148, 140],
                0.0,
                0.85,
                1.0,
            ),
            material(
                "Glass",
                &["dielectric", "transparent"],
                [230, 240, 240],
                0.0,
                0.05,
                0.15,
            ),
        ]
    }

    /// Captures a fill layer's values as a material.
    pub fn from_layer(layer: &MaterialLayer, name: impl Into<String>) -> Option<Self> {
        Some(Self {
            id: MaterialId::UNASSIGNED,
            name: name.into(),
            fill: layer.fill?,
            channels: layer.channels.clone(),
            provenance: ResourceProvenance::default(),
            tags: Vec::new(),
        })
    }

    /// Makes `layer` a fill of this material, keeping its masks and stack
    /// position. This copies the values: any link to a master is dropped.
    pub fn apply_to(&self, layer: &mut MaterialLayer) {
        layer.fill = Some(self.fill);
        layer.channels = self.channels.clone();
        layer.material_link = None;
    }

    pub fn validate(&self) -> Result<(), LibraryError> {
        if !valid_name(&self.name) {
            return Err(LibraryError::InvalidName(self.name.clone()));
        }
        if !self.fill.is_valid() {
            return Err(LibraryError::InvalidValues(self.name.clone()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Smart materials

/// A layer property a smart material exposes as a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExposedProperty {
    Opacity,
    FillColor,
    FillRoughness,
    FillMetallic,
    FillHeight,
    /// The amount of an Edge Wear, Dirt, or Noise mask generator.
    GeneratorAmount,
}

impl ExposedProperty {
    pub const ALL: [Self; 6] = [
        Self::Opacity,
        Self::FillColor,
        Self::FillRoughness,
        Self::FillMetallic,
        Self::FillHeight,
        Self::GeneratorAmount,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Opacity => "opacity",
            Self::FillColor => "color",
            Self::FillRoughness => "roughness",
            Self::FillMetallic => "metallic",
            Self::FillHeight => "height",
            Self::GeneratorAmount => "amount",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    /// `0..=1`.
    Scalar(f32),
    Color([u8; 3]),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExposedParameter {
    pub name: String,
    pub layer: LayerId,
    pub property: ExposedProperty,
}

fn generator_amount(generator: &mut MaskGenerator) -> Option<&mut f32> {
    match generator {
        MaskGenerator::EdgeWear(settings) => Some(&mut settings.amount),
        MaskGenerator::Dirt(settings) => Some(&mut settings.amount),
        MaskGenerator::Noise(settings) => Some(&mut settings.amount),
        MaskGenerator::Thickness(settings) => Some(&mut settings.amount),
        MaskGenerator::Orientation(_)
        | MaskGenerator::Gradient(_)
        | MaskGenerator::IdColor(_)
        | MaskGenerator::Pattern(_) => None,
    }
}

impl ExposedProperty {
    pub(crate) fn read(self, layer: &MaterialLayer) -> Option<ParameterValue> {
        let mut layer = layer.clone();
        self.access(&mut layer, None)
    }

    /// Reads the property, writing `value` first when given. `None` when the
    /// layer lacks the property or the value has the wrong type.
    pub(crate) fn access(
        self,
        layer: &mut MaterialLayer,
        value: Option<ParameterValue>,
    ) -> Option<ParameterValue> {
        let scalar = |slot: &mut f32, value: Option<ParameterValue>| match value {
            Some(ParameterValue::Scalar(new)) if new.is_finite() && (0.0..=1.0).contains(&new) => {
                *slot = new;
                Some(ParameterValue::Scalar(new))
            }
            Some(_) => None,
            None => Some(ParameterValue::Scalar(*slot)),
        };
        match self {
            Self::Opacity => scalar(&mut layer.opacity, value),
            Self::FillColor => {
                let fill = layer.fill.as_mut()?;
                match value {
                    Some(ParameterValue::Color(color)) => {
                        fill.color = [color[0], color[1], color[2], fill.color[3]];
                        Some(ParameterValue::Color(color))
                    }
                    Some(_) => None,
                    None => Some(ParameterValue::Color([
                        fill.color[0],
                        fill.color[1],
                        fill.color[2],
                    ])),
                }
            }
            Self::FillRoughness => scalar(&mut layer.fill.as_mut()?.roughness, value),
            Self::FillMetallic => scalar(&mut layer.fill.as_mut()?.metallic, value),
            Self::FillHeight => scalar(&mut layer.fill.as_mut()?.height, value),
            Self::GeneratorAmount => {
                scalar(generator_amount(layer.mask_generator.as_mut()?)?, value)
            }
        }
    }
}

impl MaterialLayer {
    /// The current value of each exposed parameter whose layer is still
    /// this one or inside it, with its index.
    pub fn exposed_values(&self) -> Vec<(usize, &ExposedParameter, ParameterValue)> {
        let layers = self.walk();
        self.exposed
            .iter()
            .enumerate()
            .filter_map(|(index, parameter)| {
                let layer = layers.iter().find(|layer| layer.id == parameter.layer)?;
                Some((index, parameter, parameter.property.read(layer)?))
            })
            .collect()
    }
}

impl MaterialProject {
    /// The undoable edit setting exposed parameter `index` of layer
    /// `layer_id` to `value`: a batch holding the settings change of the
    /// layer it drives
    /// (an override when that layer is a material instance). `None` when
    /// nothing changes or the index, layer, or value does not fit.
    pub fn exposed_command(
        &self,
        set_id: crate::TextureSetId,
        layer_id: LayerId,
        index: usize,
        value: ParameterValue,
    ) -> Option<crate::EditCommand> {
        let root = self
            .texture_sets
            .iter()
            .find(|set| set.id == set_id)?
            .layer(layer_id)?;
        let parameter = root.exposed.get(index)?;
        let target = root
            .walk()
            .into_iter()
            .find(|layer| layer.id == parameter.layer)?;
        let mut changed = target.clone();
        parameter.property.access(&mut changed, Some(value))?;
        let before = target.settings();
        let after = self.track_instance_edit(changed.settings());
        // A batch, as evolution moves are, so slider drags merge.
        (before != after).then(|| {
            crate::EditCommand::Batch(vec![crate::EditCommand::LayerSettings {
                texture_set_id: set_id,
                layer_id: target.id,
                before: Box::new(before),
                after: Box::new(after),
            }])
        })
    }
}

/// What a smart material needs from the project to bake.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MapRequirements {
    /// Generators or object-space projections read baked mesh maps.
    pub mesh_maps: bool,
    /// Bitmaps the layers use.
    pub resources: BTreeSet<ResourceId>,
}

/// A reusable layer tree (usually a group) with named parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SmartMaterial {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub root: MaterialLayer,
    pub parameters: Vec<ExposedParameter>,
    #[serde(default)]
    pub provenance: ResourceProvenance,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl SmartMaterial {
    /// Captures a layer tree. Polygon masks (tied to one mesh's triangles),
    /// links to project materials (kept as plain fills), and anchors
    /// pointing outside the tree are dropped; every fill's
    /// colour, roughness, and metallic and every generator's amount are
    /// exposed.
    pub fn from_layer(layer: &MaterialLayer, name: impl Into<String>) -> Self {
        let mut root = layer.clone();
        let inside: BTreeSet<LayerId> = root.walk().iter().map(|layer| layer.id).collect();
        fn strip(layer: &mut MaterialLayer, inside: &BTreeSet<LayerId>) {
            layer.polygon_mask = None;
            layer.material_link = None;
            for evolution in &mut layer.evolutions {
                evolution
                    .drives
                    .retain(|drive| inside.contains(&drive.layer));
            }
            if layer
                .mask_anchor
                .is_some_and(|anchor| !inside.contains(&anchor.layer))
            {
                layer.mask_anchor = None;
            }
            layer
                .exposed
                .retain(|parameter| inside.contains(&parameter.layer));
            for child in layer.children.iter_mut().flatten() {
                strip(child, inside);
            }
        }
        strip(&mut root, &inside);
        // An inserted instance keeps the parameters it was given.
        let kept = std::mem::take(&mut root.exposed);
        if !kept.is_empty() {
            return Self {
                name: name.into(),
                description: String::new(),
                root,
                parameters: kept,
                provenance: ResourceProvenance::default(),
                tags: Vec::new(),
            };
        }
        let mut parameters = Vec::new();
        for layer in root.walk() {
            let mut candidates = Vec::new();
            if layer.fill.is_some() {
                candidates.extend([
                    ExposedProperty::FillColor,
                    ExposedProperty::FillRoughness,
                    ExposedProperty::FillMetallic,
                ]);
            }
            if layer
                .mask_generator
                .is_some_and(|mut generator| generator_amount(&mut generator).is_some())
            {
                candidates.push(ExposedProperty::GeneratorAmount);
            }
            parameters.extend(candidates.into_iter().map(|property| ExposedParameter {
                name: format!("{} {}", layer.name, property.name()),
                layer: layer.id,
                property,
            }));
        }
        Self {
            name: name.into(),
            description: String::new(),
            root,
            parameters,
            provenance: ResourceProvenance::default(),
            tags: Vec::new(),
        }
    }

    pub fn requirements(&self) -> MapRequirements {
        let layers = self.root.walk();
        MapRequirements {
            mesh_maps: layers.iter().any(|layer| layer.own_needs_mesh_maps()),
            resources: layers
                .iter()
                .flat_map(|layer| layer.own_resources().collect::<Vec<_>>())
                .collect(),
        }
    }

    fn layer_mut(&mut self, id: LayerId) -> Option<&mut MaterialLayer> {
        fn find(layer: &mut MaterialLayer, id: LayerId) -> Option<&mut MaterialLayer> {
            if layer.id == id {
                return Some(layer);
            }
            layer
                .children
                .iter_mut()
                .flatten()
                .find_map(|child| find(child, id))
        }
        find(&mut self.root, id)
    }

    pub fn parameter_value(&self, index: usize) -> Option<ParameterValue> {
        let parameter = self.parameters.get(index)?;
        let layer = self
            .root
            .walk()
            .into_iter()
            .find(|layer| layer.id == parameter.layer)?;
        parameter.property.read(layer)
    }

    /// Sets a parameter; false when the index, layer, or value type is
    /// wrong or the value is out of range.
    pub fn set_parameter(&mut self, index: usize, value: ParameterValue) -> bool {
        let Some(parameter) = self.parameters.get(index).cloned() else {
            return false;
        };
        self.layer_mut(parameter.layer)
            .and_then(|layer| parameter.property.access(layer, Some(value)))
            .is_some()
    }

    /// A copy of the tree with fresh ids from `next_id`; anchors inside the
    /// tree follow their layers.
    pub fn instantiate(&self, mut next_id: impl FnMut() -> LayerId) -> MaterialLayer {
        let mut mapping = BTreeMap::new();
        for layer in self.root.walk() {
            mapping.insert(layer.id, next_id());
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
            for parameter in &mut layer.exposed {
                if let Some(id) = mapping.get(&parameter.layer) {
                    parameter.layer = *id;
                }
            }
            for child in layer.children.iter_mut().flatten() {
                remap(child, mapping);
            }
        }
        let mut root = self.root.clone();
        // The instance keeps the material's parameters, so they stay
        // adjustable after insertion.
        root.exposed = self
            .parameters
            .iter()
            .filter(|parameter| mapping.contains_key(&parameter.layer))
            .cloned()
            .collect();
        remap(&mut root, &mapping);
        root
    }

    pub fn validate(&self) -> Result<(), LibraryError> {
        if !valid_name(&self.name) {
            return Err(LibraryError::InvalidName(self.name.clone()));
        }
        let layers = self.root.walk();
        let mut ids = BTreeSet::new();
        if layers.iter().any(|layer| {
            !ids.insert(layer.id)
                || !layer.has_valid_settings()
                || !layer.has_valid_evolutions()
                || !valid_layer_strokes(layer)
                || layer.material_link.is_some()
        }) {
            return Err(LibraryError::InvalidLayers(self.name.clone()));
        }
        for (index, parameter) in self.parameters.iter().enumerate() {
            if self.parameter_value(index).is_none() {
                return Err(LibraryError::DanglingParameter {
                    name: self.name.clone(),
                    parameter: parameter.name.clone(),
                });
            }
        }
        Ok(())
    }

    /// Ready-made worn finishes built from the procedural generators.
    pub fn built_in() -> Vec<Self> {
        let fill = |id: u64, name: &str, color: [u8; 3], metallic: f32, roughness: f32| {
            MaterialLayer::new_fill(
                LayerId(id),
                name,
                LayerFill {
                    color: [color[0], color[1], color[2], 255],
                    roughness,
                    metallic,
                    ..LayerFill::default()
                },
            )
        };
        let group = |id: u64, name: &str, children: Vec<MaterialLayer>| MaterialLayer {
            children: Some(children),
            ..MaterialLayer::new_group(LayerId(id), name)
        };

        // Paint over steel, chipped at convex edges, with dirt in crevices.
        let steel = fill(2, "Steel", [196, 199, 199], 1.0, 0.35);
        let mut paint = fill(3, "Paint", [40, 70, 120], 0.0, 0.55);
        paint.mask_generator = Some(MaskGenerator::EdgeWear(EdgeWear::DEFAULT));
        paint.mask_filters = vec![LayerFilter::Invert];
        let mut grime = fill(4, "Grime", [60, 50, 40], 0.0, 0.9);
        grime.mask_generator = Some(MaskGenerator::Dirt(Dirt::DEFAULT));
        let mut worn_paint = group(1, "Worn painted steel", vec![steel, paint, grime]);

        // Iron with blotchy rust and dirt.
        let iron = fill(2, "Iron", [135, 131, 126], 1.0, 0.45);
        let mut rust = fill(3, "Rust", [120, 58, 30], 0.0, 0.85);
        rust.mask_generator = Some(MaskGenerator::Noise(NoisePattern::DEFAULT));
        let mut dirt = fill(4, "Dirt", [70, 60, 48], 0.0, 0.95);
        dirt.mask_generator = Some(MaskGenerator::Dirt(Dirt::DEFAULT));
        let mut rusty_iron = group(1, "Rusty iron", vec![iron, rust, dirt]);

        // Each finish has one control from clean (0) to heavily worn (1),
        // starting halfway.
        let control =
            |root: &mut MaterialLayer, name: &str, drives: &[(u64, ExposedProperty, f32, f32)]| {
                let evolution = Evolution {
                    name: name.to_owned(),
                    amount: 0.5,
                    drives: drives
                        .iter()
                        .map(|(layer, property, from, to)| EvolutionDrive {
                            layer: LayerId(*layer),
                            property: *property,
                            from: ParameterValue::Scalar(*from),
                            to: ParameterValue::Scalar(*to),
                        })
                        .collect(),
                };
                evolution.apply(root, evolution.amount);
                root.evolutions.push(evolution);
            };
        control(
            &mut worn_paint,
            "Wear",
            &[
                (3, ExposedProperty::GeneratorAmount, 0.25, 0.8),
                (4, ExposedProperty::GeneratorAmount, 0.15, 0.75),
                (3, ExposedProperty::FillRoughness, 0.45, 0.7),
            ],
        );
        control(
            &mut rusty_iron,
            "Rust",
            &[
                (3, ExposedProperty::GeneratorAmount, 0.15, 0.85),
                (2, ExposedProperty::FillRoughness, 0.4, 0.6),
                (4, ExposedProperty::GeneratorAmount, 0.2, 0.6),
            ],
        );

        [
            (
                worn_paint,
                "Painted steel chipped at the edges, with grime in cavities.",
            ),
            (
                rusty_iron,
                "Bare iron with procedural rust patches and dirt.",
            ),
        ]
        .into_iter()
        .map(|(root, description)| {
            let mut material = Self::from_layer(&root, root.name.clone());
            material.description = description.to_owned();
            material.provenance = built_in_provenance();
            material.tags = vec!["worn".to_owned(), "metal".to_owned()];
            material
        })
        .collect()
    }
}

// ---------------------------------------------------------------------------
// Smart masks

/// A reusable mask setup: a layer's generator, bitmap mask, mask filters,
/// and final inversion. Painted mask strokes, polygon selections, and ID
/// clipping belong to one layer on one mesh and are not part of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SmartMask {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<MaskGenerator>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitmap: Option<BitmapMask>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub filters: Vec<LayerFilter>,
    #[serde(default)]
    pub invert: bool,
    #[serde(default)]
    pub provenance: ResourceProvenance,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl SmartMask {
    /// Captures a layer's mask setup, or `None` when it has no generator,
    /// bitmap mask, or mask filters to reuse.
    pub fn from_layer(layer: &MaterialLayer, name: impl Into<String>) -> Option<Self> {
        if layer.mask_generator.is_none()
            && layer.mask_bitmap.is_none()
            && layer.mask_filters.is_empty()
        {
            return None;
        }
        Some(Self {
            name: name.into(),
            description: String::new(),
            generator: layer.mask_generator,
            bitmap: layer.mask_bitmap,
            filters: layer.mask_filters.clone(),
            invert: layer.invert_final_mask,
            provenance: ResourceProvenance::default(),
            tags: Vec::new(),
        })
    }

    /// Replaces `layer`'s generator, bitmap mask, mask filters, and final
    /// inversion with this setup. Its content, painted mask strokes, polygon
    /// selection, and ID clipping stay.
    pub fn apply_to(&self, layer: &mut MaterialLayer) {
        layer.mask_generator = self.generator;
        layer.mask_bitmap = self.bitmap;
        layer.mask_filters.clone_from(&self.filters);
        layer.invert_final_mask = self.invert;
    }

    pub fn requirements(&self) -> MapRequirements {
        MapRequirements {
            mesh_maps: self
                .generator
                .is_some_and(|generator| generator.needs_mesh_maps())
                || self
                    .bitmap
                    .is_some_and(|bitmap| bitmap.projection.needs_mesh_maps()),
            resources: self.bitmap.iter().map(|bitmap| bitmap.resource).collect(),
        }
    }

    pub fn validate(&self) -> Result<(), LibraryError> {
        if !valid_name(&self.name) {
            return Err(LibraryError::InvalidName(self.name.clone()));
        }
        let mut layer = MaterialLayer::new_paint(LayerId(0), "");
        self.apply_to(&mut layer);
        if !layer.has_valid_settings() {
            return Err(LibraryError::InvalidMask(self.name.clone()));
        }
        Ok(())
    }

    /// Original procedural masks for common wear and weathering.
    pub fn built_in() -> Vec<Self> {
        let mask = |name: &str, description: &str, generator, tags: &[&str]| Self {
            name: name.to_owned(),
            description: description.to_owned(),
            generator: Some(generator),
            bitmap: None,
            filters: Vec::new(),
            invert: false,
            provenance: built_in_provenance(),
            tags: tags.iter().map(|tag| (*tag).to_owned()).collect(),
        };
        vec![
            mask(
                "Worn edges",
                "Convex edges and rims, chipped by noise.",
                MaskGenerator::EdgeWear(EdgeWear::DEFAULT),
                &["wear", "edges"],
            ),
            mask(
                "Cavity dirt",
                "Occluded crevices and concave corners.",
                MaskGenerator::Dirt(Dirt::DEFAULT),
                &["dirt", "cavity"],
            ),
            mask(
                "Dust from above",
                "Surfaces facing up, where dust settles.",
                MaskGenerator::Orientation(Orientation::DEFAULT),
                &["dust", "orientation"],
            ),
            mask(
                "Thin parts",
                "Thin geometry such as fins and edges, for light passing through.",
                MaskGenerator::Thickness(Thickness::DEFAULT),
                &["thickness"],
            ),
            mask(
                "Blotches",
                "Large irregular patches in object space.",
                MaskGenerator::Noise(NoisePattern::DEFAULT),
                &["noise", "breakup"],
            ),
        ]
    }
}

// ---------------------------------------------------------------------------
// Asset index

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssetKind {
    Bitmap,
    Material,
    SmartMaterial,
    SmartMask,
    ExportPreset,
}

impl AssetKind {
    pub const ALL: [Self; 5] = [
        Self::Bitmap,
        Self::Material,
        Self::SmartMaterial,
        Self::SmartMask,
        Self::ExportPreset,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bitmap => "Bitmap",
            Self::Material => "Material",
            Self::SmartMaterial => "Smart material",
            Self::SmartMask => "Smart mask",
            Self::ExportPreset => "Export preset",
        }
    }
}

/// One searchable asset, from the project or the built-in library.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetEntry {
    pub kind: AssetKind,
    pub name: String,
    pub built_in: bool,
    pub provenance: ResourceProvenance,
    pub tags: Vec<String>,
}

impl AssetEntry {
    /// True when every whitespace-separated term of `query` appears
    /// (case-insensitively) in the name, kind, tags, author, or licence.
    pub fn matches(&self, query: &str) -> bool {
        let haystack = format!(
            "{} {} {} {} {} {}",
            self.name,
            self.kind.name(),
            self.tags.join(" "),
            self.provenance.author,
            self.provenance.license,
            if self.built_in { "built-in" } else { "project" },
        )
        .to_lowercase();
        query
            .split_whitespace()
            .all(|term| haystack.contains(&term.to_lowercase()))
    }
}

impl MaterialProject {
    /// Every asset available to the project: its own resources, materials,
    /// smart materials, smart masks, and export presets, each followed by
    /// the built-in library's.
    pub fn assets(&self) -> Vec<AssetEntry> {
        let entry =
            |kind, name: &str, built_in, provenance: &ResourceProvenance, tags: &[String]| {
                AssetEntry {
                    kind,
                    name: name.to_owned(),
                    built_in,
                    provenance: provenance.clone(),
                    tags: tags.to_vec(),
                }
            };
        let none = ResourceProvenance::default();
        let built_in = built_in_provenance();
        let mut assets = Vec::new();
        for resource in &self.resources {
            assets.push(entry(
                AssetKind::Bitmap,
                &resource.name,
                false,
                &resource.provenance,
                &resource.tags,
            ));
        }
        for (materials, is_built_in) in [
            (self.materials.clone(), false),
            (MaterialResource::built_in(), true),
        ] {
            for material in &materials {
                assets.push(entry(
                    AssetKind::Material,
                    &material.name,
                    is_built_in,
                    &material.provenance,
                    &material.tags,
                ));
            }
        }
        for (materials, is_built_in) in [
            (self.smart_materials.clone(), false),
            (SmartMaterial::built_in(), true),
        ] {
            for material in &materials {
                assets.push(entry(
                    AssetKind::SmartMaterial,
                    &material.name,
                    is_built_in,
                    &material.provenance,
                    &material.tags,
                ));
            }
        }
        for (masks, is_built_in) in [
            (self.smart_masks.clone(), false),
            (SmartMask::built_in(), true),
        ] {
            for mask in &masks {
                assets.push(entry(
                    AssetKind::SmartMask,
                    &mask.name,
                    is_built_in,
                    &mask.provenance,
                    &mask.tags,
                ));
            }
        }
        for preset in &self.export_presets {
            assets.push(entry(
                AssetKind::ExportPreset,
                &preset.name,
                false,
                &none,
                &[],
            ));
        }
        for preset in ExportPreset::built_in() {
            assets.push(entry(
                AssetKind::ExportPreset,
                &preset.name,
                true,
                &built_in,
                &[],
            ));
        }
        assets
    }

    /// Assets matching `query` (see [`AssetEntry::matches`]), optionally of
    /// one kind.
    pub fn search_assets(&self, query: &str, kind: Option<AssetKind>) -> Vec<AssetEntry> {
        self.assets()
            .into_iter()
            .filter(|asset| kind.is_none_or(|kind| asset.kind == kind) && asset.matches(query))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::generators::Dirt;
    use crate::{
        AnchorMask, BakeContext, ColorSpace, MeshMapSettings, PrimitivePreset, ProjectResource,
        ResourceImages, TextureImage, TextureResolution, TextureSet, TextureSetId, bake_mesh_maps,
    };

    fn set_with(layers: Vec<MaterialLayer>) -> TextureSet {
        TextureSet {
            id: TextureSetId(1),
            name: "Hull".to_owned(),
            material_slot: "Hull".to_owned(),
            resolution: TextureResolution {
                width: 16,
                height: 16,
            },
            channels: BTreeSet::from([MaterialChannel::BaseColor]),
            layers,
            height_strength: crate::DEFAULT_HEIGHT_STRENGTH,
            export: None,
        }
    }

    #[test]
    fn tc_0152_material_resources_apply_validate_and_bake() {
        let library = MaterialResource::built_in();
        assert!(library.len() >= 8);
        for material in &library {
            material.validate().unwrap();
        }
        let gold = library
            .iter()
            .find(|material| material.name == "Gold")
            .unwrap();
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Trim");
        gold.apply_to(&mut layer);
        let set = set_with(vec![layer.clone()]);
        let metallic = set
            .bake_channel(MaterialChannel::Metallic, BakeContext::default())
            .unwrap();
        assert_eq!(metallic.pixel(3, 3).unwrap()[0], 255);
        let base = set.bake_base_color().unwrap();
        assert_eq!(&base.pixel(3, 3).unwrap()[..3], &[255, 214, 140]);

        // A captured fill round-trips; invalid values are refused.
        let captured = MaterialResource::from_layer(&layer, "My gold").unwrap();
        assert_eq!(captured.fill, gold.fill);
        assert!(
            MaterialResource::from_layer(&MaterialLayer::new_paint(LayerId(2), "x"), "x").is_none()
        );
        let mut broken = captured.clone();
        broken.fill.roughness = 3.0;
        assert_eq!(
            broken.validate(),
            Err(LibraryError::InvalidValues("My gold".to_owned()))
        );
        let mut project = MaterialProject::empty();
        project.materials.push(broken);
        assert!(project.validate().is_err());

        // Physical tile size travels with the material.
        let mut tiled = captured;
        tiled.fill.projection.tile_size = Some(0.5);
        tiled.validate().unwrap();
    }

    #[test]
    fn tc_0153_smart_materials_expose_parameters_and_instantiate_with_fresh_ids() {
        let library = SmartMaterial::built_in();
        assert_eq!(library.len(), 2);
        for material in &library {
            material.validate().unwrap();
            assert!(
                material.requirements().mesh_maps,
                "generators need mesh maps"
            );
        }
        let mut worn = library[0].clone();
        let names: Vec<String> = worn.parameters.iter().map(|p| p.name.clone()).collect();
        assert!(names.iter().any(|name| name == "Paint color"));
        assert!(names.iter().any(|name| name == "Paint amount"));

        // Parameters read, write, and reject wrong types or ranges.
        let color = names
            .iter()
            .position(|name| *name == "Paint color")
            .unwrap();
        assert_eq!(
            worn.parameter_value(color),
            Some(ParameterValue::Color([40, 70, 120]))
        );
        assert!(worn.set_parameter(color, ParameterValue::Color([200, 30, 30])));
        assert!(!worn.set_parameter(color, ParameterValue::Scalar(0.5)));
        let amount = names
            .iter()
            .position(|name| *name == "Paint amount")
            .unwrap();
        assert!(worn.set_parameter(amount, ParameterValue::Scalar(0.8)));
        assert!(!worn.set_parameter(amount, ParameterValue::Scalar(1.5)));
        assert_eq!(
            worn.parameter_value(amount),
            Some(ParameterValue::Scalar(0.8))
        );

        // Instances get fresh ids, keeping internal anchors linked; the
        // capture drops polygon masks and outside anchors.
        let mut root = worn.root.clone();
        if let Some(children) = &mut root.children {
            children[2].mask_anchor = Some(AnchorMask::new(LayerId(2), MaterialChannel::BaseColor));
            children[1].mask_anchor =
                Some(AnchorMask::new(LayerId(99), MaterialChannel::BaseColor));
            children[1].polygon_mask = Some(crate::PolygonMask::default());
        }
        let captured = SmartMaterial::from_layer(&root, "Captured");
        let mut next = 500;
        let instance = captured.instantiate(|| {
            next += 1;
            LayerId(next)
        });
        let ids: Vec<u64> = instance.walk().iter().map(|layer| layer.id.0).collect();
        assert_eq!(ids, [501, 502, 503, 504]);
        let children = instance.children.as_ref().unwrap();
        assert_eq!(children[2].mask_anchor.unwrap().layer, LayerId(502));
        assert_eq!(children[1].mask_anchor, None);
        assert_eq!(children[1].polygon_mask, None);

        // The instance bakes in a texture set once mesh maps exist.
        let maps = bake_mesh_maps(
            &PrimitivePreset::Cube.build_mesh(),
            &MeshMapSettings {
                resolution: 32,
                ao_samples: 2,
                ..MeshMapSettings::default()
            },
            None,
        )
        .unwrap();
        let set = set_with(vec![instance]);
        let base = set
            .bake_material(BakeContext {
                mesh_maps: Some(&maps),
                ..BakeContext::default()
            })
            .unwrap();
        assert!(base.base_color.rgba8.chunks(4).any(|texel| texel[0] == 200));

        // A parameter pointing at a missing layer is refused.
        let mut dangling = captured;
        dangling.parameters[0].layer = LayerId(404);
        assert!(matches!(
            dangling.validate(),
            Err(LibraryError::DanglingParameter { .. })
        ));
    }

    #[test]
    fn tc_0154_assets_are_typed_and_searchable_with_provenance() {
        let mut project = MaterialProject::empty();
        let mut grunge = ProjectResource::new(ResourceId(1), "Rust grunge", "grunge.png");
        grunge.provenance = ResourceProvenance {
            author: "Ada".to_owned(),
            license: "CC0-1.0".to_owned(),
            origin: "Photographed".to_owned(),
        };
        grunge.tags = vec!["rust".to_owned(), "overlay".to_owned()];
        project.resources.push(grunge);

        let all = project.assets();
        assert!(all.iter().any(|asset| asset.kind == AssetKind::Bitmap));
        assert!(
            all.iter()
                .any(|asset| asset.kind == AssetKind::Material && asset.built_in)
        );
        assert!(
            all.iter()
                .any(|asset| asset.kind == AssetKind::SmartMaterial)
        );
        assert!(
            all.iter()
                .any(|asset| asset.kind == AssetKind::ExportPreset)
        );

        let names = |query: &str, kind| -> Vec<String> {
            project
                .search_assets(query, kind)
                .into_iter()
                .map(|asset| asset.name)
                .collect()
        };
        assert_eq!(names("cc0", None), ["Rust grunge"]);
        assert_eq!(names("RUST overlay", None), ["Rust grunge"]);
        assert_eq!(
            names("rust", Some(AssetKind::SmartMaterial)),
            ["Rusty iron"]
        );
        assert_eq!(
            names("unreal", Some(AssetKind::ExportPreset)),
            ["Unreal Engine"]
        );
        assert!(names("metal", Some(AssetKind::Material)).contains(&"Gold".to_owned()));
        assert!(names("no such thing", None).is_empty());
        assert!(
            project
                .search_assets(BUILT_IN_LICENSE, Some(AssetKind::Material))
                .iter()
                .all(|asset| asset.built_in)
        );
    }

    #[test]
    fn tc_0278_smart_masks_capture_apply_validate_and_are_searchable() {
        let library = SmartMask::built_in();
        assert_eq!(library.len(), 5);
        for mask in &library {
            mask.validate().unwrap();
            assert!(
                mask.requirements().mesh_maps,
                "{} reads mesh maps",
                mask.name
            );
        }

        // Capture: the procedural setup only, never strokes or selections.
        let mut source = MaterialLayer::new_fill(LayerId(1), "Rust", LayerFill::default());
        source.mask_generator = Some(MaskGenerator::Dirt(Dirt::DEFAULT));
        source.mask_filters = vec![LayerFilter::Invert];
        source.invert_final_mask = true;
        source.mask_bitmap = Some(BitmapMask {
            resource: ResourceId(4),
            projection: crate::TextureProjection::default(),
            invert: false,
        });
        source.mask_strokes.push(crate::MaskStroke::new(
            vec![glam::Vec2::splat(0.5)],
            4.0,
            0.0,
        ));
        source.polygon_mask = Some(crate::PolygonMask::default());
        let captured = SmartMask::from_layer(&source, "Inverted dirt").unwrap();
        assert_eq!(
            captured.requirements().resources,
            BTreeSet::from([ResourceId(4)])
        );
        assert!(
            SmartMask::from_layer(&MaterialLayer::new_paint(LayerId(2), "Plain"), "x").is_none(),
            "a layer without a mask setup has nothing to capture"
        );

        // Applying replaces the setup and keeps the target's own strokes,
        // selection, and content.
        let mut target = MaterialLayer::new_paint(LayerId(3), "Paint");
        target.mask_generator = Some(MaskGenerator::EdgeWear(EdgeWear::DEFAULT));
        target.mask_strokes.push(crate::MaskStroke::new(
            vec![glam::Vec2::splat(0.25)],
            2.0,
            1.0,
        ));
        target.id_mask_color = Some([1, 2, 3]);
        captured.apply_to(&mut target);
        assert_eq!(target.mask_generator, source.mask_generator);
        assert_eq!(target.mask_filters, source.mask_filters);
        assert_eq!(target.mask_bitmap, source.mask_bitmap);
        assert!(target.invert_final_mask);
        assert_eq!(target.mask_strokes.len(), 1);
        assert_eq!(target.id_mask_color, Some([1, 2, 3]));
        assert_eq!(target.polygon_mask, None);

        // Validation: names, generator ranges, and the bitmap's resource.
        let mut unnamed = captured.clone();
        unnamed.name = " ".to_owned();
        assert!(unnamed.validate().is_err());
        let mut broken = captured.clone();
        broken.generator = Some(MaskGenerator::Dirt(Dirt {
            amount: 3.0,
            ..Dirt::DEFAULT
        }));
        assert_eq!(
            broken.validate(),
            Err(LibraryError::InvalidMask("Inverted dirt".to_owned()))
        );
        let mut project = MaterialProject::empty();
        project.smart_masks.push(captured.clone());
        assert_eq!(
            project.validate(),
            Err(crate::ProjectError::MissingResource(ResourceId(4)))
        );
        project.resources.push(crate::ProjectResource::new(
            ResourceId(4),
            "Stains",
            "stains.png",
        ));
        project.validate().unwrap();

        // Saving and deleting masks is undoable; masks are searchable assets.
        let mut history = crate::EditHistory::default();
        let before = project.smart_masks.clone();
        let mut after = before.clone();
        after.push(library[2].clone());
        assert!(history.apply(
            &mut project,
            crate::EditCommand::SmartMasks { before, after },
        ));
        assert_eq!(project.smart_masks.len(), 2);
        assert!(history.undo(&mut project));
        assert_eq!(project.smart_masks.len(), 1);
        let names: Vec<String> = project
            .search_assets("dust", Some(AssetKind::SmartMask))
            .into_iter()
            .map(|asset| asset.name)
            .collect();
        assert_eq!(names, ["Dust from above"]);
        assert!(
            project
                .search_assets("smart mask project", None)
                .iter()
                .any(|asset| asset.name == "Inverted dirt")
        );
    }

    #[test]
    fn tc_0141_tagged_linear_and_acescg_bitmaps_convert_for_color_channels_only() {
        // One mid-grey texel in linear light.
        let linear = 55; // 0.214 linear = 0.5 sRGB
        let image = |space| {
            let mut image = TextureImage::new(1, 1, vec![linear, linear, linear, 255]).unwrap();
            image.color_space = space;
            Arc::new(image)
        };
        let mut layer = MaterialLayer::new_fill(LayerId(1), "Textured", LayerFill::default());
        if let Some(fill) = &mut layer.fill {
            fill.textures.base_color = Some(ResourceId(1));
            fill.textures.roughness = Some(ResourceId(1));
        }
        layer.channels = BTreeSet::from([MaterialChannel::BaseColor, MaterialChannel::Roughness]);
        let set = set_with(vec![layer]);
        for (space, color) in [
            (ColorSpace::Srgb, linear),
            (ColorSpace::LinearSrgb, 128),
            (ColorSpace::AcesCg, 128),
            (ColorSpace::Raw, linear),
        ] {
            let images: ResourceImages = [(ResourceId(1), image(space))].into();
            let context = BakeContext {
                resources: Some(&images),
                ..BakeContext::default()
            };
            let base = set
                .bake_channel(MaterialChannel::BaseColor, context)
                .unwrap();
            let value = base.pixel(0, 0).unwrap()[0];
            assert!(value.abs_diff(color) <= 1, "{space:?}: {value}");
            // Data channels read the stored values unconverted.
            let roughness = set
                .bake_channel(MaterialChannel::Roughness, context)
                .unwrap();
            assert_eq!(roughness.pixel(0, 0).unwrap()[0], linear, "{space:?}");
        }
    }

    #[test]
    fn tc_0162_material_sampling_reads_every_channel_at_a_point() {
        let copper = MaterialResource::built_in()
            .into_iter()
            .find(|material| material.name == "Copper")
            .unwrap();
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Copper");
        copper.apply_to(&mut layer);
        if let Some(fill) = &mut layer.fill {
            fill.emissive = [10, 20, 30];
        }
        layer.channels.insert(MaterialChannel::Emissive);
        let textures = set_with(vec![layer])
            .bake_material(BakeContext::default())
            .unwrap();
        let sample = textures.sample(glam::Vec2::new(0.4, 0.6)).unwrap();
        assert_eq!(sample.color, [242, 186, 159]);
        assert_eq!(sample.metallic, 1.0);
        assert!((sample.roughness - 0.3).abs() < 0.003);
        assert_eq!(sample.occlusion, 1.0);
        assert_eq!(sample.opacity, 1.0);
        assert_eq!(sample.emissive, [10, 20, 30]);
        assert!(textures.sample(glam::Vec2::new(1.5, 0.5)).is_none());
    }
}
