//! UDIM tiles in the viewports. A mesh with UVs beyond tile 1001 is shown
//! through a [`UdimAtlas`]: every tile is baked on its own (as UDIM export
//! does) and packed into one atlas texture, and the renderer gets a copy of
//! the mesh whose UVs point into the atlas. Picking and painting keep the
//! mesh's own UDIM coordinates, so strokes land in whichever tile was hit.
//!
//! Tiles are cached by content: a tile's key hashes the texture set as that
//! tile sees it ([`TextureSet::for_tile`], which leaves out strokes that
//! cannot reach it), so painting one tile rebakes only that tile. Mesh maps
//! of tiles beyond 1001 are baked once per bake key and kept.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use substancecraft_core::{
    BakeContext, BakedTexture, MaterialTextures, MeshGeometry, MeshMapKind, MeshMapSettings,
    MeshMaps, TextureSet, UdimAtlas, UdimTile,
};

use super::BakeInputs;

/// What a cached tile holds.
#[derive(Clone)]
enum TileTextures {
    Material(Arc<MaterialTextures>),
    Single(Arc<BakedTexture>),
}

/// Baked tiles and tile mesh maps, shared with background bakes.
#[derive(Default)]
pub(super) struct TileCache {
    /// Latest bake per (what was baked, tile): its content key and result.
    textures: BTreeMap<(String, UdimTile), (u64, TileTextures)>,
    /// Bake key the tile mesh maps belong to.
    maps_key: Option<u64>,
    maps: BTreeMap<(usize, UdimTile), Arc<MeshMaps>>,
    /// Tile bakes that came from the cache, for tests.
    pub(super) hits: usize,
}

/// Mesh maps of one tile: the regular maps (tile 1001) or a cached bake.
enum TileMaps<'a> {
    Regular(&'a MeshMaps),
    Baked(Arc<MeshMaps>),
}

impl TileMaps<'_> {
    fn get(&self) -> &MeshMaps {
        match self {
            Self::Regular(maps) => maps,
            Self::Baked(maps) => maps,
        }
    }
}

/// What a background bake needs to bake tiles.
#[derive(Clone)]
pub(super) struct UdimInputs {
    pub(super) atlas: UdimAtlas,
    pub(super) cache: Arc<Mutex<TileCache>>,
    /// Hash of the bake key: tile mesh maps follow it.
    pub(super) maps_key: u64,
    pub(super) mesh_revision: u64,
    pub(super) settings: MeshMapSettings,
    /// Trace tile bakes on the GPU when one is available.
    pub(super) gpu: bool,
}

/// Hash of a value's debug form, for keys whose fields do not implement
/// [`Hash`] (floats).
pub(super) fn key_hash(value: &impl std::fmt::Debug) -> u64 {
    hash_of(format!("{value:?}"))
}

fn hash_of(value: impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

impl UdimInputs {
    /// The mesh (the set's material slot, on multi-material meshes) with
    /// `tile` moved onto the unit square.
    fn tile_mesh(&self, inputs: &BakeInputs, tile: UdimTile) -> Option<MeshGeometry> {
        let mesh = inputs.mesh.as_deref()?;
        let mesh = if mesh.material_slot_count() > 1 {
            mesh.submesh(inputs.slot as u32)
        } else {
            mesh.clone()
        };
        Some(mesh.shifted_uvs(tile.offset()))
    }

    /// Mesh maps for `tile`: the regular maps for tile 1001, otherwise
    /// baked from the shifted mesh once per bake key. `None` when the
    /// regular maps are not baked (the bake then runs without maps, as it
    /// would for tile 1001).
    fn tile_maps<'a>(
        &self,
        inputs: &'a BakeInputs,
        tile: UdimTile,
        mesh: Option<&MeshGeometry>,
    ) -> Result<Option<TileMaps<'a>>, String> {
        let Some(regular) = inputs.maps() else {
            return Ok(None);
        };
        if tile == UdimTile::FIRST {
            return Ok(Some(TileMaps::Regular(regular)));
        }
        let key = (inputs.slot, tile);
        {
            let mut cache = self.lock();
            if cache.maps_key != Some(self.maps_key) {
                cache.maps.clear();
                cache.maps_key = Some(self.maps_key);
            }
            if let Some(maps) = cache.maps.get(&key) {
                return Ok(Some(TileMaps::Baked(Arc::clone(maps))));
            }
        }
        let Some(mesh) = mesh else {
            return Ok(None);
        };
        let maps = Arc::new(
            substancecraft_core::bake_mesh_maps_traced(
                mesh,
                None,
                &self.settings,
                None,
                None,
                super::gpu_tracer(self.gpu),
                None,
            )
            .map_err(|error| error.to_string())?,
        );
        let mut cache = self.lock();
        if cache.maps_key == Some(self.maps_key) {
            cache.maps.insert(key, Arc::clone(&maps));
        }
        Ok(Some(TileMaps::Baked(maps)))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TileCache> {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Content key of `tile_set` baked as `kind`: everything the tile's
    /// pixels depend on.
    fn content_key(&self, inputs: &BakeInputs, kind: &str, tile_set: &TextureSet) -> u64 {
        let resources: Vec<(u64, usize)> = tile_set
            .referenced_resources()
            .into_iter()
            .map(|id| {
                (
                    id.0,
                    inputs
                        .resources
                        .get(&id)
                        .map_or(0, |image| Arc::as_ptr(image) as usize),
                )
            })
            .collect();
        hash_of((
            kind,
            serde_json::to_vec(tile_set).unwrap_or_default(),
            self.mesh_revision,
            self.maps_key,
            inputs.slot,
            inputs.maps().is_some(),
            resources,
        ))
    }

    /// Bakes every tile of `texture_set` with `bake` (reusing cached tiles
    /// whose content is unchanged when `kind` is given).
    fn tiles(
        &self,
        inputs: &BakeInputs,
        texture_set: &TextureSet,
        kind: Option<&str>,
        bake: impl Fn(&TextureSet, BakeContext<'_>) -> Result<TileTextures, String>,
    ) -> Result<Vec<TileTextures>, String> {
        let mut baked = Vec::with_capacity(self.atlas.tiles().len());
        for &tile in self.atlas.tiles() {
            let tile_set = texture_set.for_tile(tile);
            let key = kind.map(|kind| {
                (
                    (format!("{}:{kind}", texture_set.id.0), tile),
                    self.content_key(inputs, kind, &tile_set),
                )
            });
            if let Some((slot, content)) = &key {
                let mut cache = self.lock();
                let hit = cache
                    .textures
                    .get(slot)
                    .filter(|(cached, _)| cached == content)
                    .map(|(_, textures)| textures.clone());
                if let Some(textures) = hit {
                    cache.hits += 1;
                    baked.push(textures);
                    continue;
                }
            }
            let mesh = self.tile_mesh(inputs, tile);
            let maps = self.tile_maps(inputs, tile, mesh.as_ref())?;
            let textures = bake(
                &tile_set,
                BakeContext {
                    mesh: mesh.as_ref(),
                    mesh_maps: maps.as_ref().map(TileMaps::get),
                    resources: Some(&inputs.resources),
                },
            )?;
            if let Some((slot, content)) = key {
                self.lock()
                    .textures
                    .insert(slot, (content, textures.clone()));
            }
            baked.push(textures);
        }
        Ok(baked)
    }

    /// The full material of every tile, packed into the atlas.
    pub(super) fn bake_material(
        &self,
        inputs: &BakeInputs,
        texture_set: &TextureSet,
    ) -> Result<MaterialTextures, String> {
        let tiles = self.tiles(inputs, texture_set, Some("material"), |set, context| {
            set.bake_material(context)
                .map(|textures| TileTextures::Material(Arc::new(textures)))
                .map_err(|error| error.to_string())
        })?;
        let tiles: Vec<MaterialTextures> = tiles
            .into_iter()
            .filter_map(|tile| match tile {
                TileTextures::Material(textures) => Some((*textures).clone()),
                TileTextures::Single(_) => None,
            })
            .collect();
        self.atlas
            .assemble_material(&tiles)
            .ok_or_else(|| "UDIM tiles baked at different sizes".to_owned())
    }

    /// One texture per tile from `bake`, packed into the atlas; cached by
    /// content when `kind` names what is baked.
    pub(super) fn bake_single(
        &self,
        inputs: &BakeInputs,
        texture_set: &TextureSet,
        kind: Option<&str>,
        bake: impl Fn(&TextureSet, BakeContext<'_>) -> Result<BakedTexture, String>,
    ) -> Result<BakedTexture, String> {
        let tiles = self.tiles(inputs, texture_set, kind, |set, context| {
            bake(set, context).map(|texture| TileTextures::Single(Arc::new(texture)))
        })?;
        let tiles: Vec<Arc<BakedTexture>> = tiles
            .into_iter()
            .filter_map(|tile| match tile {
                TileTextures::Single(texture) => Some(texture),
                TileTextures::Material(_) => None,
            })
            .collect();
        self.atlas
            .assemble(&tiles.iter().map(Arc::as_ref).collect::<Vec<_>>())
            .ok_or_else(|| "UDIM tiles baked at different sizes".to_owned())
    }

    /// A mesh map of every tile, packed into the atlas.
    pub(super) fn mesh_map(
        &self,
        inputs: &BakeInputs,
        kind: MeshMapKind,
    ) -> Result<BakedTexture, String> {
        let mut tiles = Vec::with_capacity(self.atlas.tiles().len());
        for &tile in self.atlas.tiles() {
            let mesh = self.tile_mesh(inputs, tile);
            let maps = self
                .tile_maps(inputs, tile, mesh.as_ref())?
                .ok_or("mesh maps are not baked")?;
            tiles.push(maps.get().to_texture(kind));
        }
        self.atlas
            .assemble(&tiles.iter().collect::<Vec<_>>())
            .ok_or_else(|| "UDIM tiles baked at different sizes".to_owned())
    }
}
