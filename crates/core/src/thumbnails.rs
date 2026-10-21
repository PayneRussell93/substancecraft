//! Preview thumbnails of library entries: a layer tree (a smart material,
//! say) baked onto a sphere and shaded on the CPU, so browsers need no GPU
//! and every machine draws the same picture.

use crate::math;
use std::collections::BTreeSet;

use glam::{Vec2, Vec3};

use crate::color::{linear_to_srgb, srgb_to_linear};
use crate::compositing::{BakeContext, BakeError};
use crate::mesh_maps::{MeshMapError, MeshMapSettings, MeshMaps, bake_mesh_maps};
use crate::resources::{ResourceImages, TextureImage};
use crate::{
    DEFAULT_HEIGHT_STRENGTH, MaterialChannel, MaterialLayer, MeshGeometry, PrimitivePreset,
    TextureResolution, TextureSet, TextureSetId,
};

/// Edge of the square textures a thumbnail bakes.
pub const THUMBNAIL_BAKE_RESOLUTION: u32 = 64;
/// Largest thumbnail edge in pixels.
pub const MAX_THUMBNAIL_SIZE: u32 = 256;

/// Channels a thumbnail bakes; the rest keep their defaults.
const THUMBNAIL_CHANNELS: [MaterialChannel; 5] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Emissive,
    MaterialChannel::Opacity,
];

/// What every thumbnail shares: the sphere, its mesh maps (so generators
/// have curvature, occlusion, and positions to read), and where each pixel
/// of the picture lands on it. Build it once and render many entries.
#[derive(Clone, Debug)]
pub struct ThumbnailStage {
    size: u32,
    mesh: MeshGeometry,
    maps: MeshMaps,
    /// Per pixel, row by row from the top: the sphere's UV there and its
    /// view-space normal, or `None` outside the sphere's disc (with the
    /// disc's coverage of the pixel for antialiased edges).
    pixels: Vec<Option<(Vec2, Vec3, f32)>>,
}

impl ThumbnailStage {
    /// A stage for `size` × `size` thumbnails (clamped to
    /// 8..=[`MAX_THUMBNAIL_SIZE`]).
    pub fn new(size: u32) -> Result<Self, MeshMapError> {
        let size = size.clamp(8, MAX_THUMBNAIL_SIZE);
        let mesh = PrimitivePreset::Sphere.build_mesh();
        let maps = bake_mesh_maps(
            &mesh,
            &MeshMapSettings {
                resolution: THUMBNAIL_BAKE_RESOLUTION,
                ao_samples: 8,
                padding: 2,
                ..MeshMapSettings::default()
            },
            None,
        )?;
        let (low, high) = mesh.vertices.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(low, high), vertex| (low.min(vertex.position), high.max(vertex.position)),
        );
        let (centre, radius) = ((low + high) * 0.5, (high - low).max_element() * 0.5);
        let pixel = 2.0 / size as f32;
        let pixels = (0..size * size)
            .map(|index| {
                let (x, y) = (index % size, index / size);
                let at = Vec2::new(
                    (x as f32 + 0.5) * pixel - 1.0,
                    1.0 - (y as f32 + 0.5) * pixel,
                );
                let distance = at.length();
                // Coverage falls off across the last pixel of the disc.
                let coverage = ((1.0 - distance) / pixel + 0.5).clamp(0.0, 1.0);
                if coverage <= 0.0 {
                    return None;
                }
                let at = at / distance.max(1.0);
                let normal = Vec3::new(at.x, at.y, (1.0 - at.length_squared()).max(0.0).sqrt());
                let hit = mesh.closest_surface_point(centre + normal * radius)?;
                Some((hit.uv, normal, coverage))
            })
            .collect();
        Ok(Self {
            size,
            mesh,
            maps,
            pixels,
        })
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    /// `layer` (with everything inside it) baked onto the sphere and lit
    /// by a key light from the upper left, as straight-alpha RGBA pixels
    /// with a transparent background. Bitmaps it uses come from
    /// `resources`.
    pub fn render(
        &self,
        layer: &MaterialLayer,
        resources: Option<&ResourceImages>,
    ) -> Result<TextureImage, BakeError> {
        let set = TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Thumbnail".to_owned(),
            material_slot: "Thumbnail".to_owned(),
            resolution: TextureResolution {
                width: THUMBNAIL_BAKE_RESOLUTION,
                height: THUMBNAIL_BAKE_RESOLUTION,
            },
            channels: THUMBNAIL_CHANNELS.into_iter().collect::<BTreeSet<_>>(),
            layers: vec![layer.clone()],
            height_strength: DEFAULT_HEIGHT_STRENGTH,
            export: None,
        };
        let textures = set.bake_material(BakeContext {
            mesh: Some(&self.mesh),
            mesh_maps: Some(&self.maps),
            resources,
        })?;
        let light = Vec3::new(-0.45, 0.6, 0.66).normalize();
        let view = Vec3::Z;
        let half = (light + view).normalize();
        let mut rgba8 = Vec::with_capacity(self.pixels.len() * 4);
        for pixel in &self.pixels {
            let Some((uv, normal, coverage)) = *pixel else {
                rgba8.extend([0, 0, 0, 0]);
                continue;
            };
            let sample = |texture: &crate::BakedTexture| {
                texture
                    .sample_uv(uv)
                    .unwrap_or([0, 0, 0, 255])
                    .map(|component| f32::from(component) / 255.0)
            };
            let [r, g, b, _] = sample(&textures.base_color);
            let albedo = Vec3::new(r, g, b).to_array().map(srgb_to_linear);
            let albedo = Vec3::from(albedo);
            let [occlusion, roughness, metallic, _] = sample(&textures.orm);
            let [er, eg, eb, _] = sample(&textures.emissive);
            let emissive = Vec3::from([er, eg, eb].map(srgb_to_linear));

            let lit = normal.dot(light).max(0.0);
            let diffuse = albedo * (1.0 - metallic) * (0.18 + 0.82 * lit) * occlusion;
            let f0 = Vec3::splat(0.04).lerp(albedo, metallic);
            let alpha = (roughness * roughness).max(0.02);
            let shininess = (2.0 / (alpha * alpha) - 2.0).clamp(1.0, 2048.0);
            let highlight = math::powf(normal.dot(half).max(0.0), shininess) * (shininess + 8.0)
                / (8.0 * std::f32::consts::PI)
                * lit;
            // A soft sky reflection so smooth metals do not read as black.
            let reflection = f0 * (0.35 * (1.0 - roughness) + 0.1) * occlusion;
            let color = diffuse + f0 * highlight + reflection + emissive;
            rgba8.extend(color.to_array().map(|component| {
                (linear_to_srgb(component.clamp(0.0, 1.0)) * 255.0).round() as u8
            }));
            rgba8.push((coverage * 255.0).round() as u8);
        }
        TextureImage::new(self.size, self.size, rgba8).ok_or(BakeError::AllocationFailed)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::library::{ParameterValue, SmartMaterial};
    use crate::{EditCommand, EditHistory, LayerFill, LayerId, MaterialProject};

    fn pixel(image: &TextureImage, x: u32, y: u32) -> [u8; 4] {
        let index = (y as usize * image.width as usize + x as usize) * 4;
        [
            image.rgba8[index],
            image.rgba8[index + 1],
            image.rgba8[index + 2],
            image.rgba8[index + 3],
        ]
    }

    #[test]
    fn tc_0337_smart_materials_render_thumbnails_and_keep_their_parameters() {
        // Thumbnails: a lit sphere on a transparent background, repeatable,
        // and different for different materials.
        let stage = ThumbnailStage::new(48).unwrap();
        assert_eq!(stage.size(), 48);
        let built_in = SmartMaterial::built_in();
        let worn = stage.render(&built_in[0].root, None).unwrap();
        let rusty = stage.render(&built_in[1].root, None).unwrap();
        assert_eq!((worn.width, worn.height), (48, 48));
        assert_eq!(pixel(&worn, 0, 0)[3], 0, "transparent corner");
        assert_eq!(pixel(&worn, 24, 24)[3], 255, "opaque centre");
        assert_ne!(worn, rusty);
        assert_eq!(stage.render(&built_in[0].root, None).unwrap(), worn);
        let fill = |color: [u8; 4], roughness: f32, metallic: f32| {
            MaterialLayer::new_fill(
                LayerId(1),
                "Fill",
                LayerFill {
                    color,
                    roughness,
                    metallic,
                    ..LayerFill::default()
                },
            )
        };
        let red = stage
            .render(&fill([200, 20, 20, 255], 0.6, 0.0), None)
            .unwrap();
        let [r, g, b, _] = pixel(&red, 24, 24);
        assert!(r > 3 * g && r > 3 * b, "red reads red: {r} {g} {b}");
        // The lit side is brighter than the far side, and smooth metal
        // shows a brighter highlight than rough metal.
        let brightness = |image: &TextureImage, x: u32, y: u32| {
            pixel(image, x, y)[..3]
                .iter()
                .map(|c| u32::from(*c))
                .sum::<u32>()
        };
        assert!(brightness(&red, 16, 16) > brightness(&red, 34, 34));
        let peak = |image: &TextureImage| {
            (0..48)
                .flat_map(|y| (0..48).map(move |x| (x, y)))
                .map(|(x, y)| brightness(image, x, y))
                .max()
                .unwrap()
        };
        let smooth = stage
            .render(&fill([220, 180, 90, 255], 0.15, 1.0), None)
            .unwrap();
        let rough = stage
            .render(&fill([220, 180, 90, 255], 0.9, 1.0), None)
            .unwrap();
        assert!(peak(&smooth) > peak(&rough));

        // Inserting keeps the parameters on the instance, following its
        // fresh ids.
        let material = &built_in[0];
        let mut next = 100;
        let root = material.instantiate(|| {
            next += 1;
            LayerId(next)
        });
        assert_eq!(root.exposed.len(), material.parameters.len());
        let inside: Vec<LayerId> = root.walk().iter().map(|layer| layer.id).collect();
        assert!(
            root.exposed
                .iter()
                .all(|parameter| inside.contains(&parameter.layer))
        );
        let values = root.exposed_values();
        assert_eq!(values.len(), material.parameters.len());
        let (paint_color, _, value) = values
            .iter()
            .find(|(_, parameter, _)| parameter.name == "Paint color")
            .copied()
            .unwrap();
        assert_eq!(value, ParameterValue::Color([40, 70, 120]));

        // Setting one is an undoable edit of the layer it drives.
        let mut project = MaterialProject::empty();
        let mut set = TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Body".to_owned(),
            material_slot: "Body".to_owned(),
            resolution: TextureResolution {
                width: 16,
                height: 16,
            },
            channels: THUMBNAIL_CHANNELS.into_iter().collect(),
            layers: Vec::new(),
            height_strength: DEFAULT_HEIGHT_STRENGTH,
            export: None,
        };
        set.layers.push(root.clone());
        project.texture_sets.push(set);
        let red = ParameterValue::Color([200, 0, 0]);
        let command = project
            .exposed_command(TextureSetId(1), root.id, paint_color, red)
            .unwrap();
        assert!(matches!(
            &command,
            EditCommand::Batch(edits) if matches!(
                edits.as_slice(),
                [EditCommand::LayerSettings { layer_id, .. }]
                    if inside.contains(layer_id) && *layer_id != root.id
            )
        ));
        let mut history = EditHistory::default();
        assert!(history.apply(&mut project, command));
        let edited = project.texture_sets[0].layer(root.id).unwrap().clone();
        assert!(
            edited
                .exposed_values()
                .iter()
                .any(|(index, _, value)| *index == paint_color && *value == red)
        );
        history.undo(&mut project);
        assert_eq!(project.texture_sets[0].layers[0], root);
        for (index, value) in [
            (paint_color, ParameterValue::Scalar(0.5)),
            (paint_color, ParameterValue::Color([40, 70, 120])),
            (99, red),
        ] {
            assert!(
                project
                    .exposed_command(TextureSetId(1), root.id, index, value)
                    .is_none(),
                "{index} {value:?}"
            );
        }

        // Saving the instance again keeps the same parameters, stored on
        // the material rather than its root.
        let saved = SmartMaterial::from_layer(&edited, "Mine");
        assert!(saved.root.exposed.is_empty());
        let names = |parameters: &[crate::library::ExposedParameter]| {
            parameters
                .iter()
                .map(|parameter| parameter.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&saved.parameters), names(&material.parameters));
        saved.validate().unwrap();
    }
}
