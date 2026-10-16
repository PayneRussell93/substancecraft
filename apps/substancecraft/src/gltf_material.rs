//! Importing glTF 2.0 materials as fill layers: the open-standard way to
//! bring materials from engines and other tools in, as OpenPBR documents do
//! for MaterialX. Factors become fill values; external textures become
//! bitmap resources. glTF packs roughness (G) and metallic (B) into one
//! texture, which is unpacked into two grey PNGs beside it, since fill
//! channels each read their own bitmap.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use substancecraft_api::{
    GltfMaterial, gltf_fill, load_texture_image, read_gltf_materials, write_texture,
};
use substancecraft_core::{BakedTexture, ColorSpace, LayerId, MaterialLayer, ResourceId};

use crate::SubstanceCraftApp;

/// The grey image of component `component` (0 R … 3 A) of `source`,
/// written to `target` unless it is already there.
fn unpack_component(source: &Path, component: usize, target: &Path) -> Result<(), String> {
    if target.exists() {
        return Ok(());
    }
    let image =
        load_texture_image(source).map_err(|error| format!("{}: {error}", source.display()))?;
    let rgba8 = image
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|texel| [texel[component], texel[component], texel[component], 255])
        .collect();
    write_texture(
        target,
        &BakedTexture {
            width: image.width,
            height: image.height,
            rgba8,
        },
    )
    .map_err(|error| format!("{}: {error}", target.display()))
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{stem}_{suffix}.png"))
}

impl SubstanceCraftApp {
    /// The resource for `path` (an existing one, or a new import) tagged
    /// `space`.
    fn gltf_resource(&mut self, path: &Path, space: ColorSpace) -> Result<ResourceId, String> {
        let id = match self
            .project
            .resources
            .iter()
            .find(|resource| Path::new(&resource.source) == path)
        {
            Some(existing) => existing.id,
            None => self.add_resource(path)?,
        };
        if let Some(resource) = self
            .project
            .resources
            .iter_mut()
            .find(|resource| resource.id == id)
            && resource.color_space != space
        {
            resource.color_space = space;
            self.reload_changed_resources(&[id]);
        }
        Ok(id)
    }

    fn gltf_material_layer(
        &mut self,
        material: &GltfMaterial,
        folder: &Path,
    ) -> Result<MaterialLayer, String> {
        let mut resources = BTreeMap::new();
        if let Some(file) = &material.base_color_texture {
            resources.insert(
                "base_color".to_owned(),
                self.gltf_resource(&folder.join(file), ColorSpace::Srgb)?,
            );
        }
        if let Some(file) = &material.metallic_roughness_texture {
            let packed = folder.join(file);
            let roughness = sibling(&packed, "roughness");
            let metallic = sibling(&packed, "metallic");
            unpack_component(&packed, 1, &roughness)?;
            unpack_component(&packed, 2, &metallic)?;
            resources.insert(
                "roughness".to_owned(),
                self.gltf_resource(&roughness, ColorSpace::Raw)?,
            );
            resources.insert(
                "metallic".to_owned(),
                self.gltf_resource(&metallic, ColorSpace::Raw)?,
            );
        }
        if let Some(file) = &material.normal_texture {
            resources.insert(
                "normal".to_owned(),
                self.gltf_resource(&folder.join(file), ColorSpace::Raw)?,
            );
        }
        if let Some(file) = &material.emissive_texture {
            resources.insert(
                "emissive".to_owned(),
                self.gltf_resource(&folder.join(file), ColorSpace::Srgb)?,
            );
        }
        let (fill, channels) = gltf_fill(material, &resources);
        let id = self.allocate_layer_id().ok_or("Out of layer ids.")?;
        let mut layer = MaterialLayer::new_fill(id, material.name.replace('_', " "), fill);
        layer.channels = channels;
        Ok(layer)
    }

    /// Adds a fill layer per material of the glTF file at `path`: just the
    /// one named like the selected texture set's material slot when there
    /// is one, else all of them. A failed import leaves no layers or
    /// bitmaps behind. Returns the new layers.
    pub(super) fn import_gltf_materials_from(
        &mut self,
        path: &Path,
    ) -> Result<Vec<LayerId>, String> {
        let materials =
            read_gltf_materials(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let slot = self
            .selected_texture_set_ref()
            .map(|set| set.material_slot.clone())
            .unwrap_or_default();
        let matching: Vec<&GltfMaterial> = materials
            .iter()
            .filter(|material| material.name.eq_ignore_ascii_case(&slot))
            .collect();
        let chosen: Vec<&GltfMaterial> = if matching.is_empty() {
            materials.iter().collect()
        } else {
            matching
        };
        let folder = path.parent().unwrap_or(Path::new("")).to_path_buf();
        let resources_before = self.project.resources.len();
        let mut layers = Vec::new();
        for material in &chosen {
            match self.gltf_material_layer(material, &folder) {
                Ok(layer) => layers.push(layer),
                Err(error) => {
                    for added in self.project.resources.split_off(resources_before) {
                        self.resource_images.remove(&added.id);
                        self.resource_watcher.forget(added.id);
                    }
                    return Err(error);
                }
            }
        }
        let mut ids = Vec::new();
        for layer in layers {
            let id = layer.id;
            self.insert_layer(layer);
            if self.selected_layer == Some(id) {
                ids.push(id);
            }
        }
        let embedded: usize = chosen
            .iter()
            .map(|material| material.embedded_textures)
            .sum();
        if embedded > 0 {
            self.import_error = Some(format!(
                "{embedded} embedded texture(s) were skipped; only external image files import."
            ));
        }
        Ok(ids)
    }

    pub(super) fn import_gltf_materials(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("glTF 2.0", &["gltf", "glb"])
            .pick_file()
        else {
            return;
        };
        if let Err(error) = self.import_gltf_materials_from(&path) {
            self.import_error = Some(error);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::MaterialChannel;

    #[test]
    fn tc_0398_gltf_materials_import_as_fill_layers_with_unpacked_roughness_and_metallic() {
        let folder = tempfile::tempdir().unwrap();
        // A packed metallic-roughness texture: roughness 100 (G), metallic 200 (B).
        write_texture(
            folder.path().join("mr.png"),
            &BakedTexture {
                width: 2,
                height: 2,
                rgba8: [0, 100, 200, 255].repeat(4),
            },
        )
        .unwrap();
        write_texture(
            folder.path().join("base.png"),
            &BakedTexture {
                width: 2,
                height: 2,
                rgba8: [10, 20, 30, 255].repeat(4),
            },
        )
        .unwrap();
        let gltf = folder.path().join("prop.gltf");
        std::fs::write(
            &gltf,
            r#"{"asset": {"version": "2.0"},
                "images": [{"uri": "base.png"}, {"uri": "mr.png"}],
                "textures": [{"source": 0}, {"source": 1}],
                "materials": [
                  {"name": "Painted_metal", "pbrMetallicRoughness": {
                     "baseColorTexture": {"index": 0}, "metallicRoughnessTexture": {"index": 1}}},
                  {"name": "Rubber", "pbrMetallicRoughness": {"baseColorFactor": [0.02, 0.02, 0.02, 1], "metallicFactor": 0, "roughnessFactor": 0.9}}
                ]}"#,
        )
        .unwrap();

        let mut app = SubstanceCraftApp::default();
        app.add_paint_layer();
        let ids = app.import_gltf_materials_from(&gltf).unwrap();
        assert_eq!(ids.len(), 2, "no slot matches, so every material imports");
        let set = app.selected_texture_set_ref().unwrap();
        let painted = set.layer(ids[0]).unwrap();
        assert_eq!(painted.name, "Painted metal");
        let fill = painted.fill.unwrap();
        let roughness = fill.textures.roughness.unwrap();
        let metallic = fill.textures.metallic.unwrap();
        assert_eq!(app.resource_images[&roughness].rgba8[0], 100);
        assert_eq!(app.resource_images[&metallic].rgba8[0], 200);
        assert!(folder.path().join("mr_roughness.png").exists());
        let rubber = set.layer(ids[1]).unwrap();
        assert!((rubber.fill.unwrap().roughness - 0.9).abs() < 1e-6);
        assert!(rubber.channels.contains(&MaterialChannel::Metallic));

        // A second import reuses the bitmaps; a missing texture adds nothing.
        let resources = app.project.resources.len();
        app.import_gltf_materials_from(&gltf).unwrap();
        assert_eq!(app.project.resources.len(), resources);
        std::fs::remove_file(folder.path().join("base.png")).unwrap();
        let broken = folder.path().join("broken.gltf");
        std::fs::write(
            &broken,
            r#"{"asset": {"version": "2.0"}, "images": [{"uri": "gone.png"}], "textures": [{"source": 0}],
                "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]}"#,
        )
        .unwrap();
        let layers = app.selected_texture_set_ref().unwrap().layers.len();
        assert!(app.import_gltf_materials_from(&broken).is_err());
        assert_eq!(app.project.resources.len(), resources);
        assert_eq!(app.selected_texture_set_ref().unwrap().layers.len(), layers);
    }
}
