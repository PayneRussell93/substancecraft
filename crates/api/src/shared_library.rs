//! The shared asset library: a folder outside any project holding
//! materials, smart materials, smart masks, filters, generators,
//! environments, export presets, and bitmaps, so every project can use
//! them.
//!
//! Each asset is one `.scasset` JSON file under `assets/`; the bitmaps and
//! panoramas assets use are copied under `files/` (named by content hash,
//! so a file shared by several assets is stored once). The folder is the
//! index: opening a library scans it, and files added or removed by hand
//! (or synced from another machine) show up on the next scan.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use substancecraft_core::{
    AssetEntry, AssetKind, AssetOrigin, ColorSpace, EnvironmentResource, ExportPreset,
    FilterPreset, GeneratorPreset, MaterialProject, MaterialResource, ProjectResource, ResourceId,
    ResourceProvenance, SmartMask, SmartMaterial,
};
use thiserror::Error;

/// Version of the `.scasset` format this build writes and reads.
pub const SHARED_LIBRARY_FORMAT: u32 = 1;
/// Extension of shared asset files.
pub const SHARED_ASSET_EXTENSION: &str = "scasset";
/// Largest asset file read; larger ones are reported, not loaded.
pub const MAX_SHARED_ASSET_BYTES: u64 = 16 * 1024 * 1024;
/// Asset files read at most per scan.
pub const MAX_SHARED_ASSETS: usize = 4_096;

const ASSETS_DIR: &str = "assets";
const FILES_DIR: &str = "files";

#[derive(Debug, Error)]
pub enum SharedLibraryError {
    #[error("{0}")]
    Io(String),
    #[error("{0}")]
    Invalid(String),
    #[error("the asset uses bitmap {0:?}, which the project does not have")]
    MissingResource(ResourceId),
    #[error("the library has no {kind} named `{name}`")]
    NotFound { kind: &'static str, name: String },
}

fn io(context: &Path, error: impl std::fmt::Display) -> SharedLibraryError {
    SharedLibraryError::Io(format!("{}: {error}", context.display()))
}

/// A bitmap stored in the library. `id` numbers it within its asset file,
/// matching the references of the asset that uses it; `file` is relative
/// to the library folder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SharedBitmap {
    #[serde(default = "first_resource")]
    pub id: ResourceId,
    pub name: String,
    pub file: String,
    #[serde(default)]
    pub color_space: ColorSpace,
    #[serde(default, skip_serializing_if = "ResourceProvenance::is_empty")]
    pub provenance: ResourceProvenance,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

fn first_resource() -> ResourceId {
    ResourceId(1)
}

/// One library asset, of any kind.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "asset", rename_all = "snake_case")]
pub enum LibraryAsset {
    Bitmap(SharedBitmap),
    Material(Box<MaterialResource>),
    SmartMaterial(Box<SmartMaterial>),
    SmartMask(SmartMask),
    Filter(FilterPreset),
    Generator(GeneratorPreset),
    Environment(EnvironmentResource),
    ExportPreset(ExportPreset),
}

impl LibraryAsset {
    pub fn kind(&self) -> AssetKind {
        match self {
            Self::Bitmap(_) => AssetKind::Bitmap,
            Self::Material(_) => AssetKind::Material,
            Self::SmartMaterial(_) => AssetKind::SmartMaterial,
            Self::SmartMask(_) => AssetKind::SmartMask,
            Self::Filter(_) => AssetKind::Filter,
            Self::Generator(_) => AssetKind::Generator,
            Self::Environment(_) => AssetKind::Environment,
            Self::ExportPreset(_) => AssetKind::ExportPreset,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Bitmap(asset) => &asset.name,
            Self::Material(asset) => &asset.name,
            Self::SmartMaterial(asset) => &asset.name,
            Self::SmartMask(asset) => &asset.name,
            Self::Filter(asset) => &asset.name,
            Self::Generator(asset) => &asset.name,
            Self::Environment(asset) => &asset.name,
            Self::ExportPreset(asset) => &asset.name,
        }
    }

    fn provenance_and_tags(&self) -> (ResourceProvenance, Vec<String>) {
        let none = (ResourceProvenance::default(), Vec::new());
        match self {
            Self::Bitmap(asset) => (asset.provenance.clone(), asset.tags.clone()),
            Self::Material(asset) => (asset.provenance.clone(), asset.tags.clone()),
            Self::SmartMaterial(asset) => (asset.provenance.clone(), asset.tags.clone()),
            Self::SmartMask(asset) => (asset.provenance.clone(), asset.tags.clone()),
            Self::Filter(asset) => (asset.provenance.clone(), asset.tags.clone()),
            Self::Generator(asset) => (asset.provenance.clone(), asset.tags.clone()),
            Self::Environment(asset) => (asset.provenance.clone(), asset.tags.clone()),
            Self::ExportPreset(_) => none,
        }
    }

    /// The asset's index entry, as a library entry.
    pub fn entry(&self) -> AssetEntry {
        let (provenance, tags) = self.provenance_and_tags();
        AssetEntry {
            kind: self.kind(),
            name: self.name().to_owned(),
            origin: AssetOrigin::Shared,
            provenance,
            tags,
        }
    }

    /// Project bitmaps the asset references.
    pub fn resources(&self) -> Vec<ResourceId> {
        match self {
            Self::Material(material) => {
                let mut resources: Vec<ResourceId> = material.fill.textures.all().collect();
                if let Some(graph) = &material.graph {
                    resources.extend(graph.resources());
                }
                resources.sort();
                resources.dedup();
                resources
            }
            Self::SmartMaterial(material) => {
                material.requirements().resources.into_iter().collect()
            }
            Self::SmartMask(mask) => mask.requirements().resources.into_iter().collect(),
            _ => Vec::new(),
        }
    }

    fn remap_resources(&mut self, map: &mut impl FnMut(ResourceId) -> ResourceId) {
        match self {
            Self::Material(material) => {
                material.fill.textures.remap(map);
                if let Some(graph) = &mut material.graph {
                    graph.remap_resources(map);
                }
            }
            Self::SmartMaterial(material) => material.root.remap_resources(map),
            Self::SmartMask(mask) => {
                if let Some(bitmap) = &mut mask.bitmap {
                    bitmap.resource = map(bitmap.resource);
                }
            }
            _ => {}
        }
    }

    pub fn validate(&self) -> Result<(), SharedLibraryError> {
        let invalid =
            |error: &dyn std::fmt::Display| SharedLibraryError::Invalid(error.to_string());
        match self {
            Self::Bitmap(bitmap) => {
                if bitmap.name.trim().is_empty() || bitmap.file.trim().is_empty() {
                    return Err(SharedLibraryError::Invalid(
                        "a bitmap needs a name and a file".to_owned(),
                    ));
                }
                Ok(())
            }
            Self::Material(asset) => asset.validate().map_err(|error| invalid(&error)),
            Self::SmartMaterial(asset) => asset.validate().map_err(|error| invalid(&error)),
            Self::SmartMask(asset) => asset.validate().map_err(|error| invalid(&error)),
            Self::Filter(asset) => asset.validate().map_err(|error| invalid(&error)),
            Self::Generator(asset) => asset.validate().map_err(|error| invalid(&error)),
            Self::Environment(asset) => asset.validate().map_err(|error| invalid(&error)),
            Self::ExportPreset(asset) => asset.validate().map_err(|error| invalid(&error)),
        }
    }
}

/// The on-disk form of one asset file.
#[derive(Serialize, Deserialize)]
struct AssetFile {
    format: u32,
    #[serde(flatten)]
    asset: LibraryAsset,
    /// Bitmaps the asset references, numbered as it references them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    bitmaps: Vec<SharedBitmap>,
}

/// One asset found in the library folder.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedEntry {
    /// The `.scasset` file.
    pub path: PathBuf,
    pub asset: LibraryAsset,
    /// Bitmaps the asset uses.
    pub bitmaps: Vec<SharedBitmap>,
}

/// What adding a library asset to a project brings: new bitmap resources
/// (with fresh ids and absolute paths into the library) and the asset with
/// its references renumbered to them.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedImport {
    pub resources: Vec<ProjectResource>,
    pub asset: LibraryAsset,
}

/// A shared library folder and the assets found in it.
#[derive(Clone, Debug, Default)]
pub struct SharedLibrary {
    root: PathBuf,
    entries: Vec<SharedEntry>,
    problems: Vec<String>,
}

/// The default library folder: `%APPDATA%\SubstanceCraft\Library` on
/// Windows, `$XDG_DATA_HOME/substancecraft/library` (or
/// `~/.local/share/substancecraft/library`) elsewhere.
pub fn default_shared_library_root() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("APPDATA").map(|appdata| {
            PathBuf::from(appdata)
                .join("SubstanceCraft")
                .join("Library")
        })
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".local/share")))
            .map(|data| data.join("substancecraft").join("library"))
    }
}

fn slug(name: &str) -> String {
    let slug: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let slug = slug.trim_matches('-').to_owned();
    if slug.is_empty() {
        "asset".to_owned()
    } else {
        slug
    }
}

fn kind_slug(kind: AssetKind) -> &'static str {
    match kind {
        AssetKind::Bitmap => "bitmap",
        AssetKind::Material => "material",
        AssetKind::SmartMaterial => "smart-material",
        AssetKind::SmartMask => "smart-mask",
        AssetKind::Filter => "filter",
        AssetKind::Generator => "generator",
        AssetKind::Environment => "environment",
        AssetKind::ExportPreset => "export-preset",
    }
}

fn same_name(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), SharedLibraryError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|error| io(parent, error))?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).map_err(|error| io(parent, error))?;
    temporary
        .write_all(bytes)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| io(path, error))?;
    temporary
        .persist(path)
        .map_err(|error| io(path, error.error))?;
    Ok(())
}

impl SharedLibrary {
    /// Scans the library at `root`. A missing folder is an empty library;
    /// unreadable or invalid asset files are skipped and listed in
    /// [`Self::problems`].
    pub fn open(root: impl Into<PathBuf>) -> Self {
        let mut library = Self {
            root: root.into(),
            entries: Vec::new(),
            problems: Vec::new(),
        };
        library.rescan();
        library
    }

    /// Reads the folder again.
    pub fn rescan(&mut self) {
        self.entries.clear();
        self.problems.clear();
        let folder = self.root.join(ASSETS_DIR);
        let Ok(listing) = std::fs::read_dir(&folder) else {
            return;
        };
        let mut paths: Vec<PathBuf> = listing
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case(SHARED_ASSET_EXTENSION))
            })
            .collect();
        paths.sort();
        if paths.len() > MAX_SHARED_ASSETS {
            self.problems.push(format!(
                "only the first {MAX_SHARED_ASSETS} of {} assets were read",
                paths.len()
            ));
            paths.truncate(MAX_SHARED_ASSETS);
        }
        for path in paths {
            match Self::read_entry(&path) {
                Ok(entry) => {
                    if self.entries.iter().any(|other| {
                        other.asset.kind() == entry.asset.kind()
                            && same_name(other.asset.name(), entry.asset.name())
                    }) {
                        self.problems.push(format!(
                            "{}: another {} is already named `{}`",
                            path.display(),
                            entry.asset.kind().name().to_lowercase(),
                            entry.asset.name()
                        ));
                    } else {
                        self.entries.push(entry);
                    }
                }
                Err(error) => self.problems.push(error.to_string()),
            }
        }
        self.entries.sort_by(|a, b| {
            (a.asset.kind(), a.asset.name().to_lowercase())
                .cmp(&(b.asset.kind(), b.asset.name().to_lowercase()))
        });
    }

    fn read_entry(path: &Path) -> Result<SharedEntry, SharedLibraryError> {
        let size = std::fs::metadata(path)
            .map_err(|error| io(path, error))?
            .len();
        if size > MAX_SHARED_ASSET_BYTES {
            return Err(io(path, "the file is too large for an asset"));
        }
        let bytes = std::fs::read(path).map_err(|error| io(path, error))?;
        let file: AssetFile = serde_json::from_slice(&bytes).map_err(|error| io(path, error))?;
        if file.format == 0 || file.format > SHARED_LIBRARY_FORMAT {
            return Err(io(
                path,
                format!(
                    "asset format {} is newer than this build reads ({SHARED_LIBRARY_FORMAT})",
                    file.format
                ),
            ));
        }
        file.asset.validate().map_err(|error| io(path, error))?;
        if let Some(missing) = file
            .asset
            .resources()
            .into_iter()
            .find(|id| !file.bitmaps.iter().any(|bitmap| bitmap.id == *id))
        {
            return Err(io(path, SharedLibraryError::MissingResource(missing)));
        }
        Ok(SharedEntry {
            path: path.to_owned(),
            asset: file.asset,
            bitmaps: file.bitmaps,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn entries(&self) -> &[SharedEntry] {
        &self.entries
    }

    /// Asset files that could not be read in the last scan, with why.
    pub fn problems(&self) -> &[String] {
        &self.problems
    }

    /// Index entries of every library asset, for search.
    pub fn asset_entries(&self) -> Vec<AssetEntry> {
        self.entries
            .iter()
            .map(|entry| entry.asset.entry())
            .collect()
    }

    /// The asset of `kind` named `name` (ignoring case).
    pub fn find(&self, kind: AssetKind, name: &str) -> Option<&SharedEntry> {
        self.entries
            .iter()
            .find(|entry| entry.asset.kind() == kind && same_name(entry.asset.name(), name))
    }

    /// Resolves a path stored in the library (relative to its folder).
    pub fn resolve(&self, file: &str) -> PathBuf {
        let path = Path::new(file);
        if path.is_absolute() {
            path.to_owned()
        } else {
            self.root.join(path)
        }
    }

    /// Copies `source` into `files/`, named by its content hash, and
    /// returns the stored path relative to the library folder.
    fn store_file(&self, source: &Path) -> Result<String, SharedLibraryError> {
        let hash = crate::file_content_hash(source).map_err(|error| io(source, error))?;
        let file_name = source
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file".to_owned());
        let stored = format!("{FILES_DIR}/{}-{file_name}", &hash.fnv1a[..12]);
        let target = self.root.join(&stored);
        if !target.exists() {
            let folder = self.root.join(FILES_DIR);
            std::fs::create_dir_all(&folder).map_err(|error| io(&folder, error))?;
            let temporary =
                tempfile::NamedTempFile::new_in(&folder).map_err(|error| io(&folder, error))?;
            std::fs::copy(source, temporary.path()).map_err(|error| io(source, error))?;
            temporary
                .persist(&target)
                .map_err(|error| io(&target, error.error))?;
        }
        Ok(stored)
    }

    /// Saves `asset` to the library, replacing any asset of the same kind
    /// and name. Bitmaps it references are looked up in `project` (relative
    /// sources resolve against `base`) and copied into the library, as is
    /// an environment's panorama. Returns the asset file written.
    pub fn save(
        &mut self,
        asset: LibraryAsset,
        project: &MaterialProject,
        base: Option<&Path>,
    ) -> Result<PathBuf, SharedLibraryError> {
        asset.validate()?;
        let resolve = |source: &str| {
            let path = PathBuf::from(source);
            match base {
                Some(base) if path.is_relative() => base.join(path),
                _ => path,
            }
        };
        let mut asset = asset;
        match &mut asset {
            LibraryAsset::Bitmap(bitmap) => {
                bitmap.file = self.store_file(&resolve(&bitmap.file))?;
                bitmap.id = ResourceId(1);
            }
            LibraryAsset::Environment(environment) => {
                environment.source = self.store_file(&resolve(&environment.source))?;
            }
            _ => {}
        }
        let mut bitmaps = Vec::new();
        for id in asset.resources() {
            let resource = project
                .resources
                .iter()
                .find(|resource| resource.id == id)
                .ok_or(SharedLibraryError::MissingResource(id))?;
            bitmaps.push(SharedBitmap {
                id,
                name: resource.name.clone(),
                file: self.store_file(&resolve(&resource.source))?,
                color_space: resource.color_space,
                provenance: resource.provenance.clone(),
                tags: resource.tags.clone(),
            });
        }

        let kind = asset.kind();
        let name = asset.name().to_owned();
        let path = match self.find(kind, &name) {
            Some(existing) => existing.path.clone(),
            None => {
                let stem = format!("{}-{}", kind_slug(kind), slug(&name));
                let folder = self.root.join(ASSETS_DIR);
                let mut path = folder.join(format!("{stem}.{SHARED_ASSET_EXTENSION}"));
                let mut counter = 2;
                while path.exists() {
                    path = folder.join(format!("{stem}-{counter}.{SHARED_ASSET_EXTENSION}"));
                    counter += 1;
                }
                path
            }
        };
        let file = AssetFile {
            format: SHARED_LIBRARY_FORMAT,
            asset,
            bitmaps,
        };
        let mut bytes = serde_json::to_vec_pretty(&file)
            .map_err(|error| SharedLibraryError::Invalid(error.to_string()))?;
        bytes.push(b'\n');
        write_atomically(&path, &bytes)?;
        self.rescan();
        Ok(path)
    }

    /// Deletes the asset of `kind` named `name`, and any stored file no
    /// other asset uses. Returns false when there is no such asset.
    pub fn remove(&mut self, kind: AssetKind, name: &str) -> Result<bool, SharedLibraryError> {
        let Some(entry) = self.find(kind, name).cloned() else {
            return Ok(false);
        };
        std::fs::remove_file(&entry.path).map_err(|error| io(&entry.path, error))?;
        self.rescan();
        let in_use: Vec<String> = self.entries.iter().flat_map(stored_files).collect();
        for file in stored_files(&entry) {
            if !in_use.contains(&file) && file.starts_with(&format!("{FILES_DIR}/")) {
                // A file that cannot be deleted only wastes space.
                let _ = std::fs::remove_file(self.root.join(&file));
            }
        }
        Ok(true)
    }

    /// Prepares the asset of `kind` named `name` for `project`: its bitmaps
    /// become new resources numbered after the project's, and the asset's
    /// references follow them.
    pub fn import(
        &self,
        kind: AssetKind,
        name: &str,
        project: &MaterialProject,
    ) -> Result<SharedImport, SharedLibraryError> {
        let entry = self
            .find(kind, name)
            .ok_or_else(|| SharedLibraryError::NotFound {
                kind: kind.name(),
                name: name.to_owned(),
            })?;
        let mut next = project
            .resources
            .iter()
            .map(|resource| resource.id.0)
            .max()
            .unwrap_or(0);
        let mut new_resource = |bitmap: &SharedBitmap| {
            next += 1;
            let path = self.resolve(&bitmap.file);
            ProjectResource {
                color_space: bitmap.color_space,
                provenance: bitmap.provenance.clone(),
                tags: bitmap.tags.clone(),
                content_hash: crate::file_content_hash(&path).ok(),
                ..ProjectResource::new(ResourceId(next), &bitmap.name, path.to_string_lossy())
            }
        };
        let mut asset = entry.asset.clone();
        let mut resources = Vec::new();
        match &mut asset {
            LibraryAsset::Bitmap(bitmap) => resources.push(new_resource(bitmap)),
            LibraryAsset::Environment(environment) => {
                environment.source = self
                    .resolve(&environment.source)
                    .to_string_lossy()
                    .into_owned();
            }
            _ => {
                let mut ids = BTreeMap::new();
                for bitmap in &entry.bitmaps {
                    let resource = new_resource(bitmap);
                    ids.insert(bitmap.id, resource.id);
                    resources.push(resource);
                }
                asset.remap_resources(&mut |id| ids.get(&id).copied().unwrap_or(id));
            }
        }
        Ok(SharedImport { resources, asset })
    }
}

fn stored_files(entry: &SharedEntry) -> Vec<String> {
    let mut files: Vec<String> = entry
        .bitmaps
        .iter()
        .map(|bitmap| bitmap.file.clone())
        .collect();
    match &entry.asset {
        LibraryAsset::Bitmap(bitmap) => files.push(bitmap.file.clone()),
        LibraryAsset::Environment(environment) => files.push(environment.source.clone()),
        _ => {}
    }
    files
}

/// Adds an imported asset to `project`'s own list of its kind, renamed
/// with a numeric suffix when the name is taken. Returns the name used, or
/// `None` for bitmaps (their resources are the whole import).
pub fn add_imported_asset(project: &mut MaterialProject, asset: LibraryAsset) -> Option<String> {
    fn unique<T>(items: &[T], name: &str, name_of: impl Fn(&T) -> &str) -> String {
        let taken = |candidate: &str| items.iter().any(|item| same_name(name_of(item), candidate));
        if !taken(name) {
            return name.to_owned();
        }
        (2..)
            .map(|counter| format!("{name} {counter}"))
            .find(|candidate| !taken(candidate))
            .unwrap_or_else(|| name.to_owned())
    }
    match asset {
        LibraryAsset::Bitmap(_) => None,
        LibraryAsset::Material(mut material) => {
            material.name = unique(&project.materials, &material.name, |m| &m.name);
            material.id = substancecraft_core::MaterialId(
                project
                    .materials
                    .iter()
                    .map(|material| material.id.0)
                    .max()
                    .unwrap_or(0)
                    + 1,
            );
            let name = material.name.clone();
            project.materials.push(*material);
            Some(name)
        }
        LibraryAsset::SmartMaterial(mut material) => {
            material.name = unique(&project.smart_materials, &material.name, |m| &m.name);
            let name = material.name.clone();
            project.smart_materials.push(*material);
            Some(name)
        }
        LibraryAsset::SmartMask(mut mask) => {
            mask.name = unique(&project.smart_masks, &mask.name, |m| &m.name);
            let name = mask.name.clone();
            project.smart_masks.push(mask);
            Some(name)
        }
        LibraryAsset::Filter(mut preset) => {
            preset.name = unique(&project.filter_presets, &preset.name, |p| &p.name);
            let name = preset.name.clone();
            project.filter_presets.push(preset);
            Some(name)
        }
        LibraryAsset::Generator(mut preset) => {
            preset.name = unique(&project.generator_presets, &preset.name, |p| &p.name);
            let name = preset.name.clone();
            project.generator_presets.push(preset);
            Some(name)
        }
        LibraryAsset::Environment(mut environment) => {
            environment.name = unique(&project.environments, &environment.name, |e| &e.name);
            let name = environment.name.clone();
            project.environments.push(environment);
            Some(name)
        }
        LibraryAsset::ExportPreset(mut preset) => {
            let built_in = ExportPreset::built_in();
            let taken: Vec<String> = built_in
                .iter()
                .chain(&project.export_presets)
                .map(|preset| preset.name.clone())
                .collect();
            preset.name = unique(&taken, &preset.name, String::as_str);
            let name = preset.name.clone();
            project.export_presets.push(preset);
            Some(name)
        }
    }
}

/// The project asset of `kind` named `name` as a library asset, ready for
/// [`SharedLibrary::save`]. Built-in assets are offered by name too.
pub fn project_asset(
    project: &MaterialProject,
    kind: AssetKind,
    name: &str,
) -> Option<LibraryAsset> {
    let pick = |candidates: Vec<LibraryAsset>| {
        candidates
            .into_iter()
            .find(|asset| same_name(asset.name(), name))
    };
    match kind {
        AssetKind::Bitmap => project
            .resources
            .iter()
            .find(|resource| same_name(&resource.name, name))
            .map(|resource| {
                LibraryAsset::Bitmap(SharedBitmap {
                    id: resource.id,
                    name: resource.name.clone(),
                    file: resource.source.clone(),
                    color_space: resource.color_space,
                    provenance: resource.provenance.clone(),
                    tags: resource.tags.clone(),
                })
            }),
        AssetKind::Material => pick(
            project
                .materials
                .iter()
                .cloned()
                .chain(MaterialResource::built_in())
                .map(|mut material| {
                    material.id = substancecraft_core::MaterialId::UNASSIGNED;
                    LibraryAsset::Material(Box::new(material))
                })
                .collect(),
        ),
        AssetKind::SmartMaterial => pick(
            project
                .smart_materials
                .iter()
                .cloned()
                .chain(SmartMaterial::built_in())
                .map(|material| LibraryAsset::SmartMaterial(Box::new(material)))
                .collect(),
        ),
        AssetKind::SmartMask => pick(
            project
                .smart_masks
                .iter()
                .cloned()
                .chain(SmartMask::built_in())
                .map(LibraryAsset::SmartMask)
                .collect(),
        ),
        AssetKind::Filter => pick(
            project
                .filter_presets
                .iter()
                .cloned()
                .chain(FilterPreset::built_in())
                .map(LibraryAsset::Filter)
                .collect(),
        ),
        AssetKind::Generator => pick(
            project
                .generator_presets
                .iter()
                .cloned()
                .chain(GeneratorPreset::built_in())
                .map(LibraryAsset::Generator)
                .collect(),
        ),
        AssetKind::Environment => pick(
            project
                .environments
                .iter()
                .cloned()
                .map(LibraryAsset::Environment)
                .collect(),
        ),
        AssetKind::ExportPreset => pick(
            project
                .export_presets
                .iter()
                .cloned()
                .chain(ExportPreset::built_in())
                .map(LibraryAsset::ExportPreset)
                .collect(),
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use substancecraft_core::{
        BitmapMask, LayerFill, LayerFilter, LayerId, MaterialLayer, TextureProjection,
    };

    fn png(path: &Path, value: u8) {
        crate::write_texture(
            path,
            &crate::BakedTexture {
                width: 2,
                height: 2,
                rgba8: vec![value; 16],
            },
        )
        .unwrap();
    }

    #[test]
    fn tc_0386_the_shared_library_saves_scans_imports_and_removes_assets_outside_projects() {
        let folder = tempfile::tempdir().unwrap();
        let project_dir = folder.path().join("project");
        std::fs::create_dir_all(&project_dir).unwrap();
        png(&project_dir.join("stains.png"), 90);
        png(&project_dir.join("scratches.png"), 200);

        // A project whose smart mask and material use bitmaps 4 and 7.
        let mut project = MaterialProject::empty();
        project
            .resources
            .push(ProjectResource::new(ResourceId(4), "Stains", "stains.png"));
        let mut scratches = ProjectResource::new(ResourceId(7), "Scratches", "scratches.png");
        scratches.color_space = ColorSpace::Raw;
        scratches.tags = vec!["wear".to_owned()];
        project.resources.push(scratches);
        let mut layer = MaterialLayer::new_fill(LayerId(1), "Rust", LayerFill::default());
        layer.mask_bitmap = Some(BitmapMask {
            resource: ResourceId(4),
            projection: TextureProjection::default(),
            invert: false,
        });
        layer.mask_filters = vec![LayerFilter::Invert];
        let mask = SmartMask::from_layer(&layer, "Stained").unwrap();
        let mut material = MaterialResource::built_in().remove(0);
        material.name = "Scratched".to_owned();
        material.fill.textures.roughness = Some(ResourceId(7));

        let root = folder.path().join("library");
        let mut library = SharedLibrary::open(&root);
        assert!(library.entries().is_empty(), "a missing folder is empty");
        library
            .save(
                LibraryAsset::SmartMask(mask.clone()),
                &project,
                Some(&project_dir),
            )
            .unwrap();
        library
            .save(
                LibraryAsset::Material(Box::new(material)),
                &project,
                Some(&project_dir),
            )
            .unwrap();
        library
            .save(
                LibraryAsset::Filter(FilterPreset::built_in().remove(0)),
                &project,
                None,
            )
            .unwrap();
        let bitmap = project_asset(&project, AssetKind::Bitmap, "stains").unwrap();
        library.save(bitmap, &project, Some(&project_dir)).unwrap();
        let hdr = project_dir.join("sky.hdr");
        std::fs::write(&hdr, b"#?RADIANCE\n").unwrap();
        let mut sky = EnvironmentResource::new("Sky", hdr.to_string_lossy());
        sky.strength = 2.0;
        library
            .save(LibraryAsset::Environment(sky), &project, None)
            .unwrap();
        // The same stains file is stored once for the mask and the bitmap.
        assert_eq!(std::fs::read_dir(root.join("files")).unwrap().count(), 3);

        // Another session (another project) scans the same folder.
        let library = SharedLibrary::open(&root);
        assert!(library.problems().is_empty(), "{:?}", library.problems());
        let kinds: Vec<AssetKind> = library
            .asset_entries()
            .iter()
            .map(|entry| entry.kind)
            .collect();
        assert_eq!(
            kinds,
            [
                AssetKind::Bitmap,
                AssetKind::Material,
                AssetKind::SmartMask,
                AssetKind::Filter,
                AssetKind::Environment
            ]
        );
        assert!(
            library
                .asset_entries()
                .iter()
                .all(|entry| entry.origin == AssetOrigin::Shared && entry.matches("library"))
        );

        // Importing into a project that already has bitmaps 1 and 2
        // renumbers the mask's bitmap and keeps its settings.
        let mut other = MaterialProject::empty();
        other
            .resources
            .push(ProjectResource::new(ResourceId(2), "Mine", "mine.png"));
        let mut plain = mask.clone();
        plain.bitmap = None;
        other.smart_masks.push(plain);
        let import = library
            .import(AssetKind::SmartMask, "STAINED", &other)
            .unwrap();
        assert_eq!(import.resources.len(), 1);
        assert_eq!(import.resources[0].id, ResourceId(3));
        assert!(Path::new(&import.resources[0].source).is_absolute());
        assert!(Path::new(&import.resources[0].source).exists());
        let LibraryAsset::SmartMask(imported) = &import.asset else {
            panic!("expected a smart mask");
        };
        assert_eq!(imported.bitmap.unwrap().resource, ResourceId(3));
        assert_eq!(imported.filters, mask.filters);
        other.resources.extend(import.resources);
        assert_eq!(
            add_imported_asset(&mut other, import.asset).as_deref(),
            Some("Stained 2"),
            "a taken name gets a suffix"
        );
        other.validate().unwrap();

        // Material bitmaps keep their colour space and tags.
        let import = library
            .import(AssetKind::Material, "Scratched", &other)
            .unwrap();
        assert_eq!(import.resources[0].color_space, ColorSpace::Raw);
        assert_eq!(import.resources[0].tags, ["wear"]);
        // Environments point at the stored panorama.
        let import = library
            .import(AssetKind::Environment, "sky", &other)
            .unwrap();
        let LibraryAsset::Environment(sky) = import.asset else {
            panic!("expected an environment");
        };
        assert!(Path::new(&sky.source).starts_with(&root));
        assert_eq!(sky.strength, 2.0);

        // Removing keeps files other assets still use.
        let mut library = library;
        assert!(library.remove(AssetKind::Bitmap, "Stains").unwrap());
        assert!(!library.remove(AssetKind::Bitmap, "Stains").unwrap());
        assert_eq!(std::fs::read_dir(root.join("files")).unwrap().count(), 3);
        assert!(library.remove(AssetKind::SmartMask, "Stained").unwrap());
        assert_eq!(std::fs::read_dir(root.join("files")).unwrap().count(), 2);

        // Broken or future files are reported, not loaded.
        std::fs::write(root.join("assets/broken.scasset"), b"{ nope").unwrap();
        std::fs::write(
            root.join("assets/future.scasset"),
            br#"{"format": 99, "kind": "filter", "asset": {"name": "X", "filters": [{"Invert": null}]}}"#,
        )
        .unwrap();
        library.rescan();
        assert_eq!(library.problems().len(), 2);
        assert_eq!(library.entries().len(), 3);

        // A material referencing a bitmap the project lacks is refused.
        let mut dangling = MaterialResource::built_in().remove(0);
        dangling.fill.textures.base_color = Some(ResourceId(40));
        assert!(matches!(
            library.save(LibraryAsset::Material(Box::new(dangling)), &project, None),
            Err(SharedLibraryError::MissingResource(ResourceId(40)))
        ));
    }
}
