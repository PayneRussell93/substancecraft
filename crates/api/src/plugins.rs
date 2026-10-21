//! Plugins (PROD-07): mask generators, per-texel layer filters, and export
//! presets loaded from folders at run time, with no rebuild of
//! SubstanceCraft and no change to its crates. `docs/plugins.md` is the
//! author's reference; `examples/plugins/wavy-bands` is a complete plugin
//! built outside this workspace.
//!
//! A plugin is a folder holding `plugin.json` (its manifest) and, when it
//! has generators or filters, a WebAssembly module. Modules run in a
//! sandboxed interpreter (wasmi): they import nothing, so they cannot touch
//! files, the network, or the clock; each call gets a fresh instance, a
//! memory cap, and an instruction budget, so a broken plugin fails its bake
//! instead of hanging or corrupting the app. WebAssembly has no
//! transcendental instructions and wasmi runs in its deterministic mode,
//! so a plugin computes the same bits on every platform (PROD-08).
//!
//! The boundary is versioned by [`PLUGIN_FORMAT`] (the manifest) and
//! [`PLUGIN_ABI`] (the module's exports); a release that changes either
//! raises it and keeps loading the previous one.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use glam::Vec4;
use serde::{Deserialize, Serialize};
use substancecraft_core::extensions::{self, valid_part};
use substancecraft_core::{
    ExportPreset, ExtensionId, MAX_PLUGIN_PARAMETERS, MeshMapSample, PluginHost, PluginRef,
    TexelKind,
};
use thiserror::Error;
use wasmi::{Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc};

/// Version of `plugin.json` this release reads.
pub const PLUGIN_FORMAT: u32 = 1;
/// Version of the module exports this release calls (`sc_abi_version`).
pub const PLUGIN_ABI: i32 = 1;
/// File naming a plugin's manifest inside its folder.
pub const PLUGIN_MANIFEST: &str = "plugin.json";
/// Floats per mesh-map sample handed to `sc_generate`: object-space
/// position (3), unit normal (3), position within the bounds as `0..1`
/// (3), position relative to the bounds diagonal (3), curvature, ambient
/// occlusion, thickness, and `1` when the sample is covered (else `0`).
pub const SAMPLE_FLOATS: usize = 16;
/// Largest manifest or module read.
const MAX_PLUGIN_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Linear memory one plugin call may grow to.
const MAX_PLUGIN_MEMORY: usize = 512 * 1024 * 1024;
/// Instruction budget of a call: a base plus this much per sample or texel.
const FUEL_BASE: u64 = 50_000_000;
const FUEL_PER_ITEM: u64 = 20_000;
/// Items one worker thread takes; smaller batches run on one thread.
const ITEMS_PER_WORKER: usize = 8_192;
/// Stack of the threads plugin calls run on.
const PLUGIN_STACK_BYTES: usize = 16 * 1024 * 1024;

/// One generator or filter a plugin offers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginItem {
    /// Unique within its plugin; with the plugin's id it forms the
    /// [`ExtensionId`] projects store.
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub parameters: Vec<PluginParameter>,
}

/// A slider shown for a plugin generator or filter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginParameter {
    pub name: String,
    pub default: f32,
    pub min: f32,
    pub max: f32,
}

/// The contents of `plugin.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginManifest {
    /// [`PLUGIN_FORMAT`].
    pub format: u32,
    /// Reverse-domain id, such as `com.example.bands`.
    pub id: String,
    pub name: String,
    pub version: String,
    /// The WebAssembly module, relative to the plugin folder. Needed only
    /// for generators and filters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    #[serde(default)]
    pub generators: Vec<PluginItem>,
    #[serde(default)]
    pub filters: Vec<PluginItem>,
    #[serde(default)]
    pub export_presets: Vec<ExportPreset>,
}

impl PluginManifest {
    fn validate(&self) -> Result<(), String> {
        if self.format != PLUGIN_FORMAT {
            return Err(format!(
                "manifest format {} is not supported (this release reads {PLUGIN_FORMAT})",
                self.format
            ));
        }
        if !valid_part(&self.id) {
            return Err(format!(
                "id `{}` must use lowercase letters, digits, `.`, `-`, and `_`",
                self.id
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for (kind, items) in [("generator", &self.generators), ("filter", &self.filters)] {
            for item in items {
                if !valid_part(&item.id) || !seen.insert((kind, item.id.as_str())) {
                    return Err(format!("{kind} id `{}` is invalid or repeated", item.id));
                }
                if ExtensionId::new(&format!("{}/{}", self.id, item.id)).is_none() {
                    return Err(format!(
                        "{kind} `{}`: plugin and item id are too long",
                        item.id
                    ));
                }
                if item.parameters.len() > MAX_PLUGIN_PARAMETERS {
                    return Err(format!(
                        "{kind} `{}` has more than {MAX_PLUGIN_PARAMETERS} parameters",
                        item.id
                    ));
                }
                for parameter in &item.parameters {
                    let ordered = parameter.min.is_finite()
                        && parameter.max.is_finite()
                        && parameter.min <= parameter.default
                        && parameter.default <= parameter.max;
                    if !ordered {
                        return Err(format!(
                            "{kind} `{}`, parameter `{}`: needs finite min ≤ default ≤ max",
                            item.id, parameter.name
                        ));
                    }
                }
            }
        }
        if (!self.generators.is_empty() || !self.filters.is_empty()) && self.module.is_none() {
            return Err("generators and filters need a `module`".to_owned());
        }
        let built_in = ExportPreset::built_in();
        let mut names = std::collections::BTreeSet::new();
        for preset in &self.export_presets {
            preset
                .validate()
                .map_err(|error| format!("export preset `{}`: {error}", preset.name))?;
            let clashes = built_in.iter().any(|known| known.name == preset.name);
            if clashes || !names.insert(preset.name.as_str()) {
                return Err(format!(
                    "export preset name `{}` is taken by a built-in or repeated",
                    preset.name
                ));
            }
        }
        Ok(())
    }

    /// The generator or filter `id` names, with its kind.
    fn item(&self, id: ExtensionId) -> Option<(ItemKind, usize, &PluginItem)> {
        if id.plugin() != self.id {
            return None;
        }
        let find = |items: &[PluginItem]| items.iter().position(|item| item.id == id.item());
        find(&self.generators)
            .map(|index| (ItemKind::Generator, index, &self.generators[index]))
            .or_else(|| {
                find(&self.filters).map(|index| (ItemKind::Filter, index, &self.filters[index]))
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ItemKind {
    Generator,
    Filter,
}

/// Why a plugin could not be loaded.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PluginError {
    #[error("{path}: {message}")]
    Read { path: PathBuf, message: String },
    #[error("{path}: {message}")]
    Manifest { path: PathBuf, message: String },
    #[error("plugin `{plugin}`: {message}")]
    Module { plugin: String, message: String },
    #[error("plugin `{0}` is already loaded from another folder")]
    Duplicate(String),
}

/// A plugin read from its folder, ready to run.
pub struct LoadedPlugin {
    manifest: PluginManifest,
    folder: PathBuf,
    module: Option<(Engine, Module)>,
}

impl std::fmt::Debug for LoadedPlugin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LoadedPlugin")
            .field("id", &self.manifest.id)
            .field("folder", &self.folder)
            .finish_non_exhaustive()
    }
}

impl LoadedPlugin {
    pub fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    pub fn folder(&self) -> &Path {
        &self.folder
    }

    /// The id a project stores for `item`, which must be one of this
    /// plugin's generators or filters.
    pub fn extension_id(&self, item: &PluginItem) -> Option<ExtensionId> {
        ExtensionId::new(&format!("{}/{}", self.manifest.id, item.id))
    }

    /// A stage using `item` at its default parameter values.
    pub fn default_stage(&self, item: &PluginItem) -> Option<PluginRef> {
        let mut parameters = [0.0; MAX_PLUGIN_PARAMETERS];
        for (value, parameter) in parameters.iter_mut().zip(&item.parameters) {
            *value = parameter.default;
        }
        Some(PluginRef::new(self.extension_id(item)?, parameters))
    }

    /// The manifest entry a stage refers to, when it is this plugin's.
    pub fn item_of(&self, id: ExtensionId) -> Option<&PluginItem> {
        self.manifest.item(id).map(|(_, _, item)| item)
    }

    /// Runs `export` of the module over `input` (`stride` floats per item),
    /// split across worker threads, and returns `out_stride` floats per
    /// item: a separate output buffer for generators, the input rewritten
    /// in place for filters.
    fn run(
        &self,
        export: &str,
        index: usize,
        stage: &PluginRef,
        input: &[f32],
        (stride, out_stride): (usize, usize),
        kind: Option<i32>,
    ) -> Result<Vec<f32>, String> {
        let (engine, module) = self
            .module
            .as_ref()
            .ok_or_else(|| "the plugin has no module".to_owned())?;
        let count = input.len() / stride;
        let workers = std::thread::available_parallelism()
            .map_or(1, |threads| threads.get())
            .clamp(1, 16)
            .min(count.div_ceil(ITEMS_PER_WORKER).max(1));
        let per_worker = count.div_ceil(workers).max(1);
        let pieces: Vec<&[f32]> = input.chunks(per_worker * stride).collect();
        // Calls always run on threads of their own, whose stack the
        // interpreter can count on whoever calls in.
        let results: Vec<Result<Vec<f32>, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = pieces
                .iter()
                .map(|piece| {
                    std::thread::Builder::new()
                        .name("plugin".to_owned())
                        .stack_size(PLUGIN_STACK_BYTES)
                        .spawn_scoped(scope, move || {
                            call_module(
                                engine, module, export, index, stage, piece, stride, out_stride,
                                kind,
                            )
                        })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| match handle {
                    Ok(handle) => handle
                        .join()
                        .unwrap_or_else(|_| Err("the plugin call panicked".to_owned())),
                    Err(error) => Err(format!("could not start a plugin thread: {error}")),
                })
                .collect()
        });
        let mut output = Vec::with_capacity(count * out_stride);
        for result in results {
            output.extend(result?);
        }
        Ok(output)
    }
}

/// One call into a fresh instance of `module`.
#[allow(clippy::too_many_arguments)]
fn call_module(
    engine: &Engine,
    module: &Module,
    export: &str,
    index: usize,
    stage: &PluginRef,
    input: &[f32],
    stride: usize,
    out_stride: usize,
    kind: Option<i32>,
) -> Result<Vec<f32>, String> {
    let count = input.len() / stride;
    let mut store = Store::new(
        engine,
        StoreLimitsBuilder::new()
            .memory_size(MAX_PLUGIN_MEMORY)
            .instances(1)
            .build(),
    );
    store.limiter(|limits: &mut StoreLimits| limits);
    let fuel = FUEL_BASE.saturating_add(FUEL_PER_ITEM.saturating_mul(count as u64));
    store.set_fuel(fuel).map_err(|error| error.to_string())?;
    let instance = Linker::<StoreLimits>::new(engine)
        .instantiate_and_start(&mut store, module)
        .map_err(|error| format!("could not start: {error}"))?;
    let memory = instance
        .get_memory(&store, "memory")
        .ok_or("the module exports no `memory`")?;
    let alloc: TypedFunc<i32, i32> = instance
        .get_typed_func(&store, "sc_alloc")
        .map_err(|error| format!("`sc_alloc`: {error}"))?;
    let trap = |error: wasmi::Error| {
        if error.as_trap_code() == Some(wasmi::TrapCode::OutOfFuel) {
            "ran past its instruction budget".to_owned()
        } else {
            error.to_string()
        }
    };
    let bytes = |floats: usize| i32::try_from(floats * 4).map_err(|_| "batch too large".to_owned());
    let place = |store: &mut Store<StoreLimits>, floats: &[f32]| -> Result<i32, String> {
        let pointer = alloc
            .call(&mut *store, bytes(floats.len())?)
            .map_err(trap)?;
        let data: Vec<u8> = floats
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        memory
            .write(&mut *store, pointer as u32 as usize, &data)
            .map_err(|_| "`sc_alloc` returned memory outside the module".to_owned())?;
        Ok(pointer)
    };
    let parameters = place(&mut store, &stage.parameters)?;
    let data = place(&mut store, input)?;
    let item = i32::try_from(index).map_err(|_| "too many items".to_owned())?;
    let count_i32 = i32::try_from(count).map_err(|_| "batch too large".to_owned())?;
    let (status, result) = match kind {
        None => {
            let output = place(&mut store, &vec![0.0; count * out_stride])?;
            let generate: TypedFunc<(i32, i32, i32, i32, i32), i32> = instance
                .get_typed_func(&store, export)
                .map_err(|error| format!("`{export}`: {error}"))?;
            let status = generate
                .call(&mut store, (item, parameters, data, count_i32, output))
                .map_err(trap)?;
            (status, output)
        }
        Some(kind) => {
            let filter: TypedFunc<(i32, i32, i32, i32, i32), i32> = instance
                .get_typed_func(&store, export)
                .map_err(|error| format!("`{export}`: {error}"))?;
            let status = filter
                .call(&mut store, (item, parameters, data, count_i32, kind))
                .map_err(trap)?;
            (status, data)
        }
    };
    if status != 0 {
        return Err(format!("`{export}` returned error {status}"));
    }
    let mut raw = vec![0_u8; count * out_stride * 4];
    memory
        .read(&store, result as u32 as usize, &mut raw)
        .map_err(|_| "the result lies outside the module's memory".to_owned())?;
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

impl PluginHost for LoadedPlugin {
    fn generate(
        &self,
        generator: &PluginRef,
        samples: &[MeshMapSample],
        values: &mut [f32],
    ) -> Result<(), String> {
        let Some((ItemKind::Generator, index, _)) = self.manifest.item(generator.id) else {
            return Err(format!(
                "this version of the plugin has no generator `{}`",
                generator.id.item()
            ));
        };
        let mut input = Vec::with_capacity(samples.len() * SAMPLE_FLOATS);
        for sample in samples {
            input.extend(sample.position.to_array());
            input.extend(sample.normal.to_array());
            input.extend(sample.unit_position.to_array());
            input.extend(sample.relative_position.to_array());
            input.extend([
                sample.curvature,
                sample.ambient_occlusion,
                sample.thickness,
                f32::from(u8::from(sample.covered)),
            ]);
        }
        let output = self.run(
            "sc_generate",
            index,
            generator,
            &input,
            (SAMPLE_FLOATS, 1),
            None,
        )?;
        values.copy_from_slice(output.get(..values.len()).ok_or("short result")?);
        Ok(())
    }

    fn filter(
        &self,
        filter: &PluginRef,
        kind: TexelKind,
        texels: &mut [Vec4],
    ) -> Result<(), String> {
        let Some((ItemKind::Filter, index, _)) = self.manifest.item(filter.id) else {
            return Err(format!(
                "this version of the plugin has no filter `{}`",
                filter.id.item()
            ));
        };
        let input: Vec<f32> = texels.iter().flat_map(|texel| texel.to_array()).collect();
        let kind = match kind {
            TexelKind::Color => 0,
            TexelKind::Scalar => 1,
        };
        let output = self.run("sc_filter", index, filter, &input, (4, 4), Some(kind))?;
        for (texel, values) in texels.iter_mut().zip(output.as_chunks::<4>().0) {
            *texel = Vec4::from_array(*values);
        }
        Ok(())
    }

    fn item_name(&self, id: ExtensionId) -> Option<String> {
        self.item_of(id).map(|item| item.name.clone())
    }

    fn export_presets(&self) -> Vec<ExportPreset> {
        self.manifest.export_presets.clone()
    }
}

fn read_limited(path: &Path) -> Result<Vec<u8>, PluginError> {
    let error = |message: String| PluginError::Read {
        path: path.to_owned(),
        message,
    };
    let size = std::fs::metadata(path)
        .map_err(|io| error(io.to_string()))?
        .len();
    if size > MAX_PLUGIN_FILE_BYTES {
        return Err(error(format!(
            "larger than {} MiB",
            MAX_PLUGIN_FILE_BYTES / (1024 * 1024)
        )));
    }
    std::fs::read(path).map_err(|io| error(io.to_string()))
}

/// Reads and checks the plugin in `folder`: its manifest, and its module's
/// exports and ABI version.
pub fn load_plugin(folder: &Path) -> Result<LoadedPlugin, PluginError> {
    let manifest_path = folder.join(PLUGIN_MANIFEST);
    let manifest: PluginManifest =
        serde_json::from_slice(&read_limited(&manifest_path)?).map_err(|error| {
            PluginError::Manifest {
                path: manifest_path.clone(),
                message: error.to_string(),
            }
        })?;
    manifest
        .validate()
        .map_err(|message| PluginError::Manifest {
            path: manifest_path.clone(),
            message,
        })?;
    let module_error = |message: String| PluginError::Module {
        plugin: manifest.id.clone(),
        message,
    };
    let module = match &manifest.module {
        None => None,
        Some(file) => {
            let bytes = read_limited(&crate::resolve_stored_path(file, Some(folder)))?;
            let mut config = wasmi::Config::default();
            config.consume_fuel(true);
            let engine = Engine::new(&config);
            let module =
                Module::new(&engine, &bytes).map_err(|error| module_error(error.to_string()))?;
            if module.imports().next().is_some() {
                return Err(module_error(
                    "the module imports functions or memory; plugins may not import anything"
                        .to_owned(),
                ));
            }
            check_exports(&engine, &module, &manifest).map_err(module_error)?;
            Some((engine, module))
        }
    };
    Ok(LoadedPlugin {
        manifest,
        folder: folder.to_owned(),
        module,
    })
}

/// Checks the module exports what the manifest needs and speaks
/// [`PLUGIN_ABI`].
fn check_exports(
    engine: &Engine,
    module: &Module,
    manifest: &PluginManifest,
) -> Result<(), String> {
    let mut store = Store::new(
        engine,
        StoreLimitsBuilder::new()
            .memory_size(MAX_PLUGIN_MEMORY)
            .build(),
    );
    store.limiter(|limits: &mut StoreLimits| limits);
    store
        .set_fuel(FUEL_BASE)
        .map_err(|error| error.to_string())?;
    let instance = Linker::<StoreLimits>::new(engine)
        .instantiate_and_start(&mut store, module)
        .map_err(|error| format!("could not start: {error}"))?;
    if instance.get_memory(&store, "memory").is_none() {
        return Err("the module exports no `memory`".to_owned());
    }
    let version: TypedFunc<(), i32> = instance
        .get_typed_func(&store, "sc_abi_version")
        .map_err(|error| format!("`sc_abi_version`: {error}"))?;
    let abi = version
        .call(&mut store, ())
        .map_err(|error| error.to_string())?;
    if abi != PLUGIN_ABI {
        return Err(format!(
            "the module speaks plugin ABI {abi}; this release calls ABI {PLUGIN_ABI}"
        ));
    }
    instance
        .get_typed_func::<i32, i32>(&store, "sc_alloc")
        .map_err(|error| format!("`sc_alloc`: {error}"))?;
    for (export, needed) in [
        ("sc_generate", !manifest.generators.is_empty()),
        ("sc_filter", !manifest.filters.is_empty()),
    ] {
        if needed {
            instance
                .get_typed_func::<(i32, i32, i32, i32, i32), i32>(&store, export)
                .map_err(|error| format!("`{export}`: {error}"))?;
        }
    }
    Ok(())
}

/// Loads every plugin found in `folders`: a folder holding a manifest is
/// one plugin, else each of its subfolders that holds one is. Missing
/// folders are skipped; plugins that fail to load are reported and left
/// out, so one broken plugin does not stop the others.
pub fn load_plugins(folders: &[PathBuf]) -> (Vec<Arc<LoadedPlugin>>, Vec<PluginError>) {
    let mut plugins: Vec<Arc<LoadedPlugin>> = Vec::new();
    let mut errors = Vec::new();
    for folder in folders {
        let candidates = if folder.join(PLUGIN_MANIFEST).is_file() {
            vec![folder.clone()]
        } else {
            let mut found: Vec<PathBuf> = std::fs::read_dir(folder)
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.join(PLUGIN_MANIFEST).is_file())
                .collect();
            found.sort();
            found
        };
        for candidate in candidates {
            match load_plugin(&candidate) {
                Ok(plugin) => {
                    if plugins
                        .iter()
                        .any(|loaded| loaded.manifest.id == plugin.manifest.id)
                    {
                        errors.push(PluginError::Duplicate(plugin.manifest.id));
                    } else {
                        plugins.push(Arc::new(plugin));
                    }
                }
                Err(error) => errors.push(error),
            }
        }
    }
    (plugins, errors)
}

static INSTALLED: RwLock<Vec<Arc<LoadedPlugin>>> = RwLock::new(Vec::new());

/// Makes `plugins` the ones bakes run, replacing those installed before.
pub fn install_plugins(plugins: Vec<Arc<LoadedPlugin>>) {
    let Ok(mut installed) = INSTALLED.write() else {
        return;
    };
    for old in installed.iter() {
        extensions::unregister_plugin(&old.manifest.id);
    }
    for plugin in &plugins {
        let host: Arc<dyn PluginHost> = Arc::clone(plugin) as Arc<dyn PluginHost>;
        extensions::register_plugin(&plugin.manifest.id, host);
    }
    *installed = plugins;
}

/// The plugins [`install_plugins`] installed, in load order.
pub fn installed_plugins() -> Vec<Arc<LoadedPlugin>> {
    INSTALLED
        .read()
        .map(|plugins| plugins.clone())
        .unwrap_or_default()
}

/// Folders plugins load from: those listed in `SUBSTANCECRAFT_PLUGINS`
/// (separated like `PATH`), then `%LOCALAPPDATA%\SubstanceCraft\plugins`,
/// else `$XDG_CONFIG_HOME/substancecraft/plugins` or
/// `~/.config/substancecraft/plugins`.
pub fn default_plugin_folders() -> Vec<PathBuf> {
    let from = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    let mut folders: Vec<PathBuf> = from("SUBSTANCECRAFT_PLUGINS")
        .map(|list| std::env::split_paths(&list).collect())
        .unwrap_or_default();
    if let Some(local) = from("LOCALAPPDATA") {
        folders.push(PathBuf::from(local).join("SubstanceCraft").join("plugins"));
    } else if let Some(config) = from("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| from("HOME").map(|home| PathBuf::from(home).join(".config")))
    {
        folders.push(config.join("substancecraft").join("plugins"));
    }
    folders
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Writes a plugin folder: `manifest` and, when given, a module from
    /// WebAssembly text.
    fn plugin_folder(manifest: &str, module: Option<&str>) -> tempfile::TempDir {
        let folder = tempfile::tempdir().unwrap();
        std::fs::write(folder.path().join(PLUGIN_MANIFEST), manifest).unwrap();
        if let Some(text) = module {
            std::fs::write(
                folder.path().join("plugin.wasm"),
                wat::parse_str(text).unwrap(),
            )
            .unwrap();
        }
        folder
    }

    const MANIFEST: &str = r#"{
        "format": 1, "id": "test.wat", "name": "WAT probe", "version": "0.1.0",
        "module": "plugin.wasm",
        "generators": [{"id": "occlusion", "name": "Occlusion",
            "parameters": [{"name": "Gain", "default": 1, "min": 0, "max": 4}]}],
        "filters": [{"id": "half", "name": "Half"}]
    }"#;

    /// Generator 0 writes AO times the first parameter; filter 0 halves
    /// every component. A bump allocator hands out memory.
    const MODULE: &str = r#"(module
        (memory (export "memory") 1)
        (global $next (mut i32) (i32.const 1024))
        (func (export "sc_abi_version") (result i32) (i32.const 1))
        (func (export "sc_alloc") (param $bytes i32) (result i32)
            (local $at i32)
            (local.set $at (global.get $next))
            (global.set $next (i32.add (global.get $next) (local.get $bytes)))
            (if (i32.gt_u (global.get $next) (i32.mul (memory.size) (i32.const 65536)))
                (then (drop (memory.grow (i32.add (i32.div_u (local.get $bytes) (i32.const 65536)) (i32.const 1))))))
            (local.get $at))
        (func (export "sc_generate") (param $item i32) (param $params i32) (param $samples i32)
                (param $count i32) (param $out i32) (result i32)
            (local $i i32)
            (if (local.get $item) (then (return (i32.const 7))))
            (block $done (loop $next
                (br_if $done (i32.ge_u (local.get $i) (local.get $count)))
                (f32.store
                    (i32.add (local.get $out) (i32.mul (local.get $i) (i32.const 4)))
                    (f32.mul
                        (f32.load (i32.add (local.get $samples)
                            (i32.add (i32.mul (local.get $i) (i32.const 64)) (i32.const 52))))
                        (f32.load (local.get $params))))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $next)))
            (i32.const 0))
        (func (export "sc_filter") (param $item i32) (param $params i32) (param $texels i32)
                (param $count i32) (param $kind i32) (result i32)
            (local $i i32) (local $at i32)
            (block $done (loop $next
                (br_if $done (i32.ge_u (local.get $i) (i32.mul (local.get $count) (i32.const 4))))
                (local.set $at (i32.add (local.get $texels) (i32.mul (local.get $i) (i32.const 4))))
                (f32.store (local.get $at) (f32.mul (f32.load (local.get $at)) (f32.const 0.5)))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $next)))
            (i32.const 0)))"#;

    fn sample(occlusion: f32) -> MeshMapSample {
        MeshMapSample {
            covered: true,
            position: glam::Vec3::ZERO,
            unit_position: glam::Vec3::ZERO,
            relative_position: glam::Vec3::ZERO,
            normal: glam::Vec3::Z,
            tangent_normal: glam::Vec3::Z,
            curvature: 0.5,
            ambient_occlusion: occlusion,
            thickness: 1.0,
            skew: 0.0,
        }
    }

    #[test]
    fn modules_run_generators_and_filters_over_whole_batches() {
        let folder = plugin_folder(MANIFEST, Some(MODULE));
        let plugin = load_plugin(folder.path()).unwrap();
        let generator = plugin
            .default_stage(&plugin.manifest().generators[0])
            .unwrap();
        assert_eq!(generator.id.as_str(), "test.wat/occlusion");
        assert_eq!(generator.parameters[0], 1.0);

        // Large enough to spread over several workers.
        let samples: Vec<MeshMapSample> = (0..40_000)
            .map(|i| sample((i % 100) as f32 / 100.0))
            .collect();
        let mut values = vec![0.0; samples.len()];
        let mut doubled = generator;
        doubled.parameters[0] = 2.0;
        plugin.generate(&doubled, &samples, &mut values).unwrap();
        for (value, sample) in values.iter().zip(&samples) {
            assert_eq!(*value, sample.ambient_occlusion * 2.0);
        }

        let filter = plugin.default_stage(&plugin.manifest().filters[0]).unwrap();
        let mut texels = vec![Vec4::new(1.0, 0.5, 0.25, 1.0); 3];
        plugin
            .filter(&filter, TexelKind::Color, &mut texels)
            .unwrap();
        assert_eq!(texels[2], Vec4::new(0.5, 0.25, 0.125, 0.5));
        assert_eq!(plugin.item_name(filter.id).as_deref(), Some("Half"));

        // A stage the plugin does not have, or a module error, fails.
        let other = PluginRef::new(ExtensionId::new("test.wat/gone").unwrap(), [0.0; 8]);
        assert!(
            plugin
                .generate(&other, &samples[..1], &mut values[..1])
                .is_err()
        );
        assert!(
            plugin
                .filter(&generator, TexelKind::Color, &mut texels)
                .is_err()
        );
    }

    #[test]
    fn runaway_and_misbehaving_modules_fail_their_call() {
        let spin = MODULE.replace(
            "(if (local.get $item) (then (return (i32.const 7))))",
            "(loop $forever (br $forever))",
        );
        let folder = plugin_folder(MANIFEST, Some(&spin));
        let plugin = load_plugin(folder.path()).unwrap();
        let generator = plugin
            .default_stage(&plugin.manifest().generators[0])
            .unwrap();
        let mut values = [0.0; 4];
        let error = plugin
            .generate(&generator, &[sample(0.5); 4], &mut values)
            .unwrap_err();
        assert!(error.contains("budget"), "{error}");

        let failing = MODULE.replace(
            "(if (local.get $item) (then (return (i32.const 7))))",
            "(return (i32.const 7))",
        );
        let folder = plugin_folder(MANIFEST, Some(&failing));
        let plugin = load_plugin(folder.path()).unwrap();
        let error = plugin
            .generate(&generator, &[sample(0.5)], &mut values[..1])
            .unwrap_err();
        assert!(error.contains("error 7"), "{error}");
    }

    #[test]
    fn plugins_are_checked_when_they_load() {
        let load = |manifest: &str, module: Option<&str>| {
            let folder = plugin_folder(manifest, module);
            load_plugin(folder.path())
                .map(|_| ())
                .unwrap_err()
                .to_string()
        };
        assert!(
            load(
                &MANIFEST.replace("\"format\": 1", "\"format\": 2"),
                Some(MODULE)
            )
            .contains("format 2")
        );
        assert!(load(&MANIFEST.replace("test.wat", "Test Wat"), Some(MODULE)).contains("id"));
        assert!(
            load(
                &MANIFEST.replace("\"default\": 1", "\"default\": 9"),
                Some(MODULE)
            )
            .contains("min")
        );
        assert!(
            load(
                MANIFEST,
                Some(&MODULE.replace("(i32.const 1))", "(i32.const 2))"))
            )
            .contains("ABI 2")
        );
        let importing = MODULE.replace(
            "(memory (export \"memory\") 1)",
            "(import \"env\" \"log\" (func $log (param i32)))\n(memory (export \"memory\") 1)",
        );
        assert!(load(MANIFEST, Some(&importing)).contains("may not import"));
        let no_filter = MODULE.replace("(export \"sc_filter\")", "");
        assert!(load(MANIFEST, Some(&no_filter)).contains("sc_filter"));
        assert!(load(MANIFEST, None).contains("plugin.wasm"));
        let unknown = MANIFEST.replace("\"version\"", "\"colour\": 1, \"version\"");
        assert!(load(&unknown, Some(MODULE)).contains("colour"));

        // A preset-only plugin needs no module; its presets must not take
        // a built-in name.
        let mut preset = ExportPreset::built_in().remove(0);
        preset.name = "Studio ORM".to_owned();
        let manifest = serde_json::to_string(&PluginManifest {
            format: PLUGIN_FORMAT,
            id: "test.presets".to_owned(),
            name: "Presets".to_owned(),
            version: "1.0.0".to_owned(),
            module: None,
            generators: Vec::new(),
            filters: Vec::new(),
            export_presets: vec![preset.clone()],
        })
        .unwrap();
        let folder = plugin_folder(&manifest, None);
        let plugin = load_plugin(folder.path()).unwrap();
        assert_eq!(plugin.export_presets(), vec![preset]);
        let clash = manifest.replace("Studio ORM", &ExportPreset::built_in()[0].name);
        assert!(load(&clash, None).contains("taken"));
    }

    #[test]
    fn folders_of_plugins_load_together_and_skip_broken_ones() {
        let root = tempfile::tempdir().unwrap();
        for (name, manifest) in [
            ("good", MANIFEST.to_owned()),
            ("broken", "{ not json".to_owned()),
            ("again", MANIFEST.to_owned()),
        ] {
            let folder = root.path().join(name);
            std::fs::create_dir(&folder).unwrap();
            std::fs::write(folder.join(PLUGIN_MANIFEST), manifest).unwrap();
            std::fs::write(folder.join("plugin.wasm"), wat::parse_str(MODULE).unwrap()).unwrap();
        }
        let (plugins, errors) =
            load_plugins(&[root.path().to_owned(), root.path().join("missing")]);
        assert_eq!(plugins.len(), 1);
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(errors.contains(&PluginError::Duplicate("test.wat".to_owned())));
    }
}
