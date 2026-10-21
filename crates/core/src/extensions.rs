//! Extension points for plugins (PROD-07): mask generators, layer filters,
//! and export presets defined outside this crate.
//!
//! The core knows a plugin's generators and filters only by id and
//! parameter values ([`PluginRef`]), which projects store like any other
//! stage. Whatever runs them (the API crate's WebAssembly runtime) is
//! registered per plugin with [`register_plugin`]. A bake that meets a
//! plugin stage hands the plugin the whole region at once, never one texel
//! at a time. A bake that needs a plugin nobody registered fails naming it,
//! so a project never exports silently different textures on a machine
//! that lacks one.
//!
//! Plugin filters are per-texel by contract: they read only the texel they
//! write, so region, tiled, and incremental bakes stay exact without a
//! halo.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use glam::Vec4;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ExportPreset;
use crate::mesh_maps::MeshMapSample;

/// Parameters a plugin generator or filter may take.
pub const MAX_PLUGIN_PARAMETERS: usize = 8;
/// Longest [`ExtensionId`], in bytes.
pub const MAX_EXTENSION_ID: usize = 64;

/// Names one generator or filter of one plugin: `plugin/item`, such as
/// `com.example.bands/wavy`. Both parts use lowercase ASCII letters, digits,
/// `.`, `-`, and `_`. Stored inline so the stages that hold it stay `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExtensionId {
    bytes: [u8; MAX_EXTENSION_ID],
    len: u8,
}

impl ExtensionId {
    /// `None` unless `text` is `plugin/item` in the allowed characters.
    pub fn new(text: &str) -> Option<Self> {
        let (plugin, item) = text.split_once('/')?;
        if text.len() > MAX_EXTENSION_ID || !valid_part(plugin) || !valid_part(item) {
            return None;
        }
        let mut bytes = [0; MAX_EXTENSION_ID];
        bytes[..text.len()].copy_from_slice(text.as_bytes());
        Some(Self {
            bytes,
            len: text.len() as u8,
        })
    }

    pub fn as_str(&self) -> &str {
        // Only ASCII is ever stored.
        std::str::from_utf8(&self.bytes[..usize::from(self.len)]).unwrap_or_default()
    }

    /// The plugin part, before the `/`.
    pub fn plugin(&self) -> &str {
        self.as_str()
            .split_once('/')
            .map_or("", |(plugin, _)| plugin)
    }

    /// The generator or filter part, after the `/`.
    pub fn item(&self) -> &str {
        self.as_str().split_once('/').map_or("", |(_, item)| item)
    }
}

/// True for a non-empty id part in the allowed characters.
pub fn valid_part(part: &str) -> bool {
    !part.is_empty()
        && part.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-' | b'_')
        })
}

impl fmt::Debug for ExtensionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ExtensionId({})", self.as_str())
    }
}

impl fmt::Display for ExtensionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for ExtensionId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ExtensionId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::new(&text).ok_or_else(|| {
            serde::de::Error::custom(format!("`{text}` is not a plugin id like `plugin/item`"))
        })
    }
}

/// A plugin generator or filter in a layer stack, with its parameter
/// values. Parameters a plugin does not declare stay `0`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PluginRef {
    pub id: ExtensionId,
    #[serde(default)]
    pub parameters: [f32; MAX_PLUGIN_PARAMETERS],
}

impl PluginRef {
    pub fn new(id: ExtensionId, parameters: [f32; MAX_PLUGIN_PARAMETERS]) -> Self {
        Self { id, parameters }
    }

    pub fn is_valid(&self) -> bool {
        self.parameters.iter().all(|value| value.is_finite())
    }

    /// What the registered plugin calls this generator or filter, else its
    /// id.
    pub fn display_name(&self) -> String {
        plugin_host(self.id.plugin())
            .and_then(|host| host.item_name(self.id))
            .unwrap_or_else(|| self.id.as_str().to_owned())
    }
}

/// What the texels handed to a plugin filter hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexelKind {
    /// sRGB colour, straight alpha.
    Color,
    /// A grey value in all three components.
    Scalar,
}

/// Runs one plugin's generators and filters. Implemented outside the core
/// (see the API crate's plugin runtime).
pub trait PluginHost: Send + Sync {
    /// Mask values in `0..=1` for `samples`; `values` has one slot per
    /// sample. Uncovered samples are passed too and should yield `0`.
    fn generate(
        &self,
        generator: &PluginRef,
        samples: &[MeshMapSample],
        values: &mut [f32],
    ) -> Result<(), String>;

    /// Filters straight-alpha texels (`xyz` value, `w` alpha) in place,
    /// reading only each texel itself.
    fn filter(
        &self,
        filter: &PluginRef,
        kind: TexelKind,
        texels: &mut [Vec4],
    ) -> Result<(), String>;

    /// The display name of a generator or filter this plugin provides.
    fn item_name(&self, id: ExtensionId) -> Option<String>;

    /// Export presets the plugin adds to the built-in ones.
    fn export_presets(&self) -> Vec<ExportPreset> {
        Vec::new()
    }
}

static PLUGINS: RwLock<BTreeMap<String, Arc<dyn PluginHost>>> = RwLock::new(BTreeMap::new());

/// Makes `host` run the plugin named `plugin`, replacing any host it had.
pub fn register_plugin(plugin: &str, host: Arc<dyn PluginHost>) {
    if let Ok(mut plugins) = PLUGINS.write() {
        plugins.insert(plugin.to_owned(), host);
    }
}

/// Forgets the plugin named `plugin`.
pub fn unregister_plugin(plugin: &str) {
    if let Ok(mut plugins) = PLUGINS.write() {
        plugins.remove(plugin);
    }
}

/// The host running `plugin`, if one is registered.
pub fn plugin_host(plugin: &str) -> Option<Arc<dyn PluginHost>> {
    PLUGINS.read().ok()?.get(plugin).cloned()
}

/// Names of the registered plugins, sorted.
pub fn registered_plugins() -> Vec<String> {
    PLUGINS
        .read()
        .map(|plugins| plugins.keys().cloned().collect())
        .unwrap_or_default()
}

/// Every registered plugin's export presets, by plugin name.
pub fn plugin_export_presets() -> Vec<ExportPreset> {
    let hosts: Vec<Arc<dyn PluginHost>> = PLUGINS
        .read()
        .map(|plugins| plugins.values().cloned().collect())
        .unwrap_or_default();
    hosts
        .iter()
        .flat_map(|host| host.export_presets())
        .collect()
}

/// The built-in export presets followed by the registered plugins'.
pub fn available_export_presets() -> Vec<ExportPreset> {
    let mut presets = ExportPreset::built_in();
    for preset in plugin_export_presets() {
        if !presets.iter().any(|known| known.name == preset.name) {
            presets.push(preset);
        }
    }
    presets
}

/// Runs a plugin generator over `samples`.
pub(crate) fn generate(
    generator: &PluginRef,
    samples: &[MeshMapSample],
    values: &mut [f32],
) -> Result<(), crate::BakeError> {
    let host = plugin_host(generator.id.plugin())
        .ok_or_else(|| crate::BakeError::MissingPlugin(generator.id.to_string()))?;
    host.generate(generator, samples, values)
        .map_err(|message| crate::BakeError::Plugin {
            id: generator.id.to_string(),
            message,
        })?;
    for (value, sample) in values.iter_mut().zip(samples) {
        *value = if sample.covered && value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    Ok(())
}

/// Runs a plugin filter over `texels`.
pub(crate) fn filter(
    filter: &PluginRef,
    kind: TexelKind,
    texels: &mut [Vec4],
) -> Result<(), crate::BakeError> {
    let host = plugin_host(filter.id.plugin())
        .ok_or_else(|| crate::BakeError::MissingPlugin(filter.id.to_string()))?;
    host.filter(filter, kind, texels)
        .map_err(|message| crate::BakeError::Plugin {
            id: filter.id.to_string(),
            message,
        })?;
    for texel in texels {
        let clean = |value: f32| {
            if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        *texel = Vec4::new(
            clean(texel.x),
            clean(texel.y),
            clean(texel.z),
            clean(texel.w),
        );
        if kind == TexelKind::Scalar {
            *texel = Vec4::new(texel.x, texel.x, texel.x, texel.w);
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_plugin_and_item_in_safe_characters() {
        let id = ExtensionId::new("com.example.bands/wavy-2").unwrap();
        assert_eq!(id.plugin(), "com.example.bands");
        assert_eq!(id.item(), "wavy-2");
        assert_eq!(id.to_string(), "com.example.bands/wavy-2");
        for bad in [
            "wavy",
            "/wavy",
            "bands/",
            "Bands/wavy",
            "a/b/c",
            "a b/c",
            "a/é",
        ] {
            assert!(ExtensionId::new(bad).is_none(), "{bad}");
        }
        let long = format!("{}/x", "a".repeat(MAX_EXTENSION_ID));
        assert!(ExtensionId::new(&long).is_none());

        let stage = PluginRef::new(id, [0.5; MAX_PLUGIN_PARAMETERS]);
        let json = serde_json::to_string(&stage).unwrap();
        assert!(json.contains("\"com.example.bands/wavy-2\""), "{json}");
        assert_eq!(serde_json::from_str::<PluginRef>(&json).unwrap(), stage);
        assert!(serde_json::from_str::<PluginRef>(r#"{"id":"nope"}"#).is_err());
        assert!(!PluginRef::new(id, [f32::NAN; MAX_PLUGIN_PARAMETERS]).is_valid());
    }

    /// A host in plain Rust: `curvature` scales the curvature map by its
    /// first parameter, `invert` inverts, and anything else fails.
    struct Probe;

    impl PluginHost for Probe {
        fn generate(
            &self,
            generator: &PluginRef,
            samples: &[MeshMapSample],
            values: &mut [f32],
        ) -> Result<(), String> {
            if generator.id.item() != "curvature" {
                return Err("no such generator".to_owned());
            }
            for (value, sample) in values.iter_mut().zip(samples) {
                *value = sample.curvature * generator.parameters[0];
            }
            Ok(())
        }

        fn filter(
            &self,
            filter: &PluginRef,
            _kind: TexelKind,
            texels: &mut [Vec4],
        ) -> Result<(), String> {
            if filter.id.item() != "invert" {
                return Err("no such filter".to_owned());
            }
            for texel in texels {
                *texel = (glam::Vec3::ONE - texel.truncate()).extend(texel.w);
            }
            Ok(())
        }

        fn item_name(&self, id: ExtensionId) -> Option<String> {
            Some(format!("Probe {}", id.item()))
        }

        fn export_presets(&self) -> Vec<ExportPreset> {
            let mut preset = ExportPreset::built_in().remove(0);
            preset.name = "Probe preset".to_owned();
            vec![preset]
        }
    }

    #[test]
    fn tc_0409_plugin_generators_and_filters_bake_like_built_in_stages() {
        use crate::{
            BakeContext, LayerFill, LayerFilter, LayerId, MaskGenerator, MaterialChannel,
            MaterialLayer, MeshMapSettings, PrimitivePreset, TexelRect, TextureResolution,
            TextureSet, TextureSetId, bake_mesh_maps,
        };

        register_plugin("test.probe", Arc::new(Probe));
        let stage = |item: &str, amount: f32| {
            let mut parameters = [0.0; MAX_PLUGIN_PARAMETERS];
            parameters[0] = amount;
            PluginRef::new(
                ExtensionId::new(&format!("test.probe/{item}")).unwrap(),
                parameters,
            )
        };
        let mesh = PrimitivePreset::Sphere.build_mesh();
        let settings = MeshMapSettings {
            resolution: 32,
            ao_samples: 2,
            ..MeshMapSettings::default()
        };
        let maps = bake_mesh_maps(&mesh, &settings, None).unwrap();
        let context = BakeContext {
            mesh: Some(&mesh),
            mesh_maps: Some(&maps),
            resources: None,
        };
        let mut metal = MaterialLayer::new_fill(
            LayerId(1),
            "Metal",
            LayerFill {
                metallic: 1.0,
                roughness: 0.2,
                ..LayerFill::default()
            },
        );
        metal.mask_generator = Some(MaskGenerator::Plugin(stage("curvature", 1.0)));
        metal.filters = vec![LayerFilter::Plugin(stage("invert", 0.0))];
        metal.filter_channels = [MaterialChannel::Roughness].into();
        let set = TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Probe".to_owned(),
            material_slot: "Probe".to_owned(),
            resolution: TextureResolution {
                width: 32,
                height: 32,
            },
            channels: Default::default(),
            layers: vec![metal],
            height_strength: 1.0,
            export: None,
        };

        // The mask is the curvature map, sample for sample.
        let metallic = set
            .bake_channel(MaterialChannel::Metallic, context)
            .unwrap();
        for y in 0..32 {
            for x in 0..32 {
                let sample = maps.sample(crate::raster::pixel_to_uv(x, y, 32, 32));
                let expected = if sample.covered {
                    sample.curvature
                } else {
                    0.0
                };
                let byte = metallic.pixel(x, y).unwrap()[0];
                assert_eq!(byte, crate::compositing::unit_to_byte(expected), "{x},{y}");
            }
        }

        // The filter matches the built-in Invert byte for byte.
        let mut built_in = set.clone();
        built_in.layers[0].filters = vec![LayerFilter::Invert];
        assert_eq!(
            set.bake_channel(MaterialChannel::Roughness, context)
                .unwrap(),
            built_in
                .bake_channel(MaterialChannel::Roughness, context)
                .unwrap()
        );

        // Region rebakes stay exact: plugin filters need no halo.
        assert_eq!(set.region_halo(), 0);
        let full = set
            .bake_channel(MaterialChannel::Roughness, context)
            .unwrap();
        let mut patched = built_in
            .bake_channel(MaterialChannel::Metallic, context)
            .unwrap();
        let region = TexelRect {
            x: 5,
            y: 7,
            width: 11,
            height: 9,
        };
        set.rebake_channel_region(MaterialChannel::Roughness, context, &mut patched, region)
            .unwrap();
        for y in region.y..region.y + region.height {
            for x in region.x..region.x + region.width {
                assert_eq!(patched.pixel(x, y), full.pixel(x, y), "{x},{y}");
            }
        }

        // Names and presets come from the host.
        assert_eq!(stage("invert", 0.0).display_name(), "Probe invert");
        assert!(
            available_export_presets()
                .iter()
                .any(|preset| preset.name == "Probe preset")
        );

        // A failing plugin and a missing one both name themselves.
        let mut failing = set.clone();
        failing.layers[0].mask_generator = Some(MaskGenerator::Plugin(stage("nope", 1.0)));
        assert!(matches!(
            failing.bake_channel(MaterialChannel::Metallic, context),
            Err(crate::BakeError::Plugin { id, .. }) if id == "test.probe/nope"
        ));
        let mut missing = set.clone();
        missing.layers[0].mask_generator = Some(MaskGenerator::Plugin(PluginRef::new(
            ExtensionId::new("test.absent/curvature").unwrap(),
            [0.0; MAX_PLUGIN_PARAMETERS],
        )));
        assert_eq!(
            missing.bake_channel(MaterialChannel::Metallic, context),
            Err(crate::BakeError::MissingPlugin(
                "test.absent/curvature".to_owned()
            ))
        );

        // Graphs evaluate per texel, so they refuse plugin stages.
        let node = crate::GraphNode::Generator(MaskGenerator::Plugin(stage("curvature", 1.0)));
        let graph = crate::MaterialGraph {
            nodes: vec![node],
            outputs: Default::default(),
        };
        assert!(!graph.is_valid());
        unregister_plugin("test.probe");
        assert!(plugin_host("test.probe").is_none());
    }
}
