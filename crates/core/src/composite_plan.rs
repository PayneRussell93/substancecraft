//! A texture set's material bake prepared for compositing on another device
//! (PERF-06, GPU layer compositing).
//!
//! [`CompositePlan::new`] walks the layer stack once per material channel
//! and lists, in order, what a compositor has to do: start and finish
//! groups, and lay each layer over the stack through its filters, mask, and
//! opacity. Layer content the plan describes in data (uniform and bitmap
//! fills under any projection, material graphs), masks built from
//! generators, bitmaps, and per-texel filters, and per-texel filters
//! themselves are left to the other device. Everything else stays on the
//! CPU and comes back as a whole plane from [`CompositePlan::content`] or
//! [`CompositePlan::mask`], computed by the same code CPU bakes run. Paint
//! strokes, blurs, quick masks, anchors, and polygon and ID selections all
//! work this way.
//!
//! A stack the plan cannot describe this way (one that reads the stack
//! beneath a layer while compositing it: clone and smudge strokes, or a
//! pass-through group with spatial filters) is refused, and so is a set too
//! large to bake in one tile. Callers then bake on the CPU as before, which
//! also remains the path for exports.

use std::sync::Arc;

use glam::Vec4;

use crate::compositing::{
    BakeContext, BakeError, BlendMode, base_texel, contributes, fill_texel, plane_data,
};
use crate::filters::PlaneData;
use crate::{
    BitmapMask, GraphInput, LayerFilter, LayerId, MAX_BAKE_PIXELS, MaskGenerator, MaterialChannel,
    MaterialGraph, MaterialLayer, MeshMaps, ResourceId, TextureImage, TextureProjection,
    TextureSet,
};

/// The channels a material bake composites, in the order a plan lists
/// them: Height before Normal, whose detail reads it.
pub const PLAN_CHANNELS: [MaterialChannel; 7] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Emissive,
    MaterialChannel::Opacity,
    MaterialChannel::Height,
    MaterialChannel::Normal,
];

/// What a channel's texels hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneKind {
    /// sRGB colour with straight alpha.
    Color,
    /// A grey value in all three components.
    Scalar,
    /// Unit tangent-space normals; only Normal blending applies, and
    /// results are renormalised.
    Vector,
}

impl From<PlaneData> for PlaneKind {
    fn from(data: PlaneData) -> Self {
        match data {
            PlaneData::Color => Self::Color,
            PlaneData::Scalar => Self::Scalar,
            PlaneData::Vector => Self::Vector,
        }
    }
}

/// What a layer lays over the stack, before its filters.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanContent {
    /// The same texel everywhere; a decal projection clips it to its box.
    Uniform {
        texel: Vec4,
        decal: Option<TextureProjection>,
    },
    /// Image `image` of [`CompositePlan::images`] through `projection`.
    /// With `copy`, the image has the set's own size and plain UVs, so each
    /// texel copies the pixel it lies on instead of resampling.
    Bitmap {
        image: usize,
        projection: TextureProjection,
        copy: bool,
    },
    /// Output `output` of graph `graph` of [`CompositePlan::graphs`],
    /// opaque.
    Graph { graph: usize, output: GraphInput },
    /// [`CompositePlan::content`] of the layer: already filtered.
    Cpu(LayerId),
}

/// A layer's mask.
// A plan holds a handful of these, built once per bake; boxing the large
// variant would only add indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum PlanMask {
    Open,
    /// The generator (or white), times the bitmap, through per-texel
    /// filters, then inverted when asked.
    Generated {
        generator: Option<MaskGenerator>,
        /// The image index and how it is placed.
        bitmap: Option<(usize, BitmapMask)>,
        filters: Vec<LayerFilter>,
        invert: bool,
    },
    /// [`CompositePlan::mask`] of the layer.
    Cpu(LayerId),
}

/// One thing to do, in order, for a channel.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanStep {
    /// Lays a layer over the current plane: its content through `filters`
    /// (per-texel), `mask`, and `opacity`, blended by `blend`. A layer's
    /// mask is the same in every channel.
    Layer {
        layer: LayerId,
        content: PlanContent,
        filters: Vec<LayerFilter>,
        mask: PlanMask,
        opacity: f32,
        blend: BlendMode,
    },
    /// Starts a group: a new plane above the current one, transparent, or
    /// for a pass-through group a copy of the current one.
    BeginGroup { pass_through: bool },
    /// Finishes a group: runs `filters` (per-texel) on its plane, then lays
    /// it over the plane beneath like a layer, or for a pass-through group
    /// fades it in by `opacity` and `mask`.
    EndGroup {
        layer: LayerId,
        filters: Vec<LayerFilter>,
        mask: PlanMask,
        opacity: f32,
        blend: BlendMode,
        pass_through: bool,
    },
}

/// The steps of one channel.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanChannel {
    pub channel: MaterialChannel,
    pub kind: PlaneKind,
    /// The texel before any layer. For Normal with mesh maps, each texel
    /// starts at the baked tangent-space normal instead.
    pub base: Vec4,
    pub base_from_maps: bool,
    pub steps: Vec<PlanStep>,
}

/// Why a stack is not planned.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    /// The CPU bake would fail the same way.
    #[error(transparent)]
    Bake(BakeError),
    /// The stack needs the CPU compositor; the reason is for logs.
    #[error("needs the CPU compositor: {0}")]
    Unsupported(&'static str),
}

impl From<BakeError> for PlanError {
    fn from(error: BakeError) -> Self {
        Self::Bake(error)
    }
}

/// A material bake of one texture set, planned for another compositor. See
/// the module documentation.
pub struct CompositePlan<'a> {
    set: &'a TextureSet,
    context: BakeContext<'a>,
    id_map: Option<Vec<u8>>,
    /// Channels some layer paints, each with its steps. Channels left out
    /// keep their defaults, as in CPU bakes.
    pub channels: Vec<PlanChannel>,
    /// Bitmaps plan steps read, by index.
    pub images: Vec<(ResourceId, Arc<TextureImage>)>,
    /// Graphs plan steps evaluate, by index; each graph's bitmap nodes are
    /// among [`Self::images`] (see [`Self::image_index`]).
    pub graphs: Vec<&'a MaterialGraph>,
    /// Most groups open at once in any channel.
    pub depth: usize,
}

impl<'a> CompositePlan<'a> {
    /// Plans `set`'s material bake (Base Color, ORM, Normal, Emissive, as
    /// [`TextureSet::bake_material`] makes it). Fails as the CPU bake would
    /// for invalid stacks, and with [`PlanError::Unsupported`] for stacks
    /// only the CPU compositor handles.
    pub fn new(set: &'a TextureSet, context: BakeContext<'a>) -> Result<Self, PlanError> {
        let region = set.texel_rect();
        if region.len() > MAX_BAKE_PIXELS {
            return Err(PlanError::Unsupported("the set bakes in tiles"));
        }
        let id_map = set.validate_for_region(region)?;
        if set.walk_layers().iter().any(|layer| {
            layer
                .paint_strokes
                .iter()
                .any(|stroke| stroke.source.is_some())
        }) {
            return Err(PlanError::Unsupported(
                "clone and smudge strokes read the stack",
            ));
        }
        let mut plan = Self {
            set,
            context,
            id_map,
            channels: Vec::new(),
            images: Vec::new(),
            graphs: Vec::new(),
            depth: 0,
        };
        for channel in PLAN_CHANNELS {
            // Normal always composites: it starts from the baked normals.
            if channel != MaterialChannel::Normal
                && !set
                    .layers
                    .iter()
                    .any(|layer| contributes(layer, &[channel]))
            {
                continue;
            }
            let mut steps = Vec::new();
            let mut depth = 0;
            plan.walk(&set.layers, channel, &mut steps, &mut depth, 0)?;
            plan.depth = plan.depth.max(depth);
            plan.channels.push(PlanChannel {
                channel,
                kind: plane_data(channel).into(),
                base: base_texel(channel)?,
                base_from_maps: channel == MaterialChannel::Normal && context.mesh_maps.is_some(),
                steps,
            });
        }
        Ok(plan)
    }

    pub fn set(&self) -> &'a TextureSet {
        self.set
    }

    pub fn mesh_maps(&self) -> Option<&'a MeshMaps> {
        self.context.mesh_maps
    }

    /// Height strength scaled to the set's resolution, as the Normal
    /// channel's height detail applies it; `0` when Height adds none.
    pub fn height_detail_scale(&self) -> f32 {
        let height_painted = self
            .channels
            .iter()
            .any(|channel| channel.channel == MaterialChannel::Height);
        if self.set.height_strength <= 0.0 || !height_painted {
            return 0.0;
        }
        let resolution = self.set.resolution;
        self.set.height_strength * resolution.width.max(resolution.height) as f32
            / crate::compositing::HEIGHT_SLOPE_REFERENCE_TEXELS
    }

    /// Where `resource` is in [`Self::images`].
    pub fn image_index(&self, resource: ResourceId) -> Option<usize> {
        self.images.iter().position(|(id, _)| *id == resource)
    }

    /// A [`PlanContent::Cpu`] layer's content for `channel` over the whole
    /// set, after its filters, as straight-alpha texels in row order.
    pub fn content(
        &self,
        layer: LayerId,
        channel: MaterialChannel,
    ) -> Result<Vec<Vec4>, BakeError> {
        let layer = self
            .set
            .layer(layer)
            .ok_or(BakeError::InvalidLayerSettings)?;
        self.set
            .layer_local_plane(layer, channel, self.context, self.id_map.as_deref())
    }

    /// A [`PlanMask::Cpu`] layer's mask over the whole set, in row order.
    pub fn mask(&self, layer: LayerId) -> Result<Vec<f32>, BakeError> {
        let layer = self
            .set
            .layer(layer)
            .ok_or(BakeError::InvalidLayerSettings)?;
        let region = self.set.texel_rect();
        Ok(self
            .set
            .layer_mask_in(layer, self.context, self.id_map.as_deref(), region)?
            .unwrap_or_else(|| vec![1.0; region.len()]))
    }

    fn image(&mut self, resource: ResourceId) -> Result<usize, BakeError> {
        if let Some(index) = self.image_index(resource) {
            return Ok(index);
        }
        let image = self
            .context
            .resources
            .and_then(|images| images.get(&resource))
            .ok_or(BakeError::MissingResource(resource))?;
        self.images.push((resource, Arc::clone(image)));
        Ok(self.images.len() - 1)
    }

    fn walk(
        &mut self,
        layers: &'a [MaterialLayer],
        channel: MaterialChannel,
        steps: &mut Vec<PlanStep>,
        depth: &mut usize,
        level: usize,
    ) -> Result<(), PlanError> {
        for layer in layers {
            if !contributes(layer, &[channel]) {
                continue;
            }
            let filters = layer.filters_for(channel).to_vec();
            let gpu_filters = filters.iter().all(|filter| per_texel(*filter));
            let mask = self.plan_mask(layer)?;
            let opacity = layer.opacity;
            let blend = layer.blend_mode_for(channel);
            match &layer.children {
                Some(children) if layer.blend_mode == BlendMode::PassThrough => {
                    if !gpu_filters {
                        return Err(PlanError::Unsupported(
                            "a pass-through group's spatial filters read the stack",
                        ));
                    }
                    *depth = (*depth).max(level + 1);
                    steps.push(PlanStep::BeginGroup { pass_through: true });
                    self.walk(children, channel, steps, depth, level + 1)?;
                    steps.push(PlanStep::EndGroup {
                        layer: layer.id,
                        filters,
                        mask,
                        opacity,
                        blend,
                        pass_through: true,
                    });
                }
                // Nested groups that need the CPU become CPU content at
                // their own level, so only this group's filters matter.
                Some(children) if gpu_filters => {
                    *depth = (*depth).max(level + 1);
                    steps.push(PlanStep::BeginGroup {
                        pass_through: false,
                    });
                    self.walk(children, channel, steps, depth, level + 1)?;
                    steps.push(PlanStep::EndGroup {
                        layer: layer.id,
                        filters,
                        mask,
                        opacity,
                        blend,
                        pass_through: false,
                    });
                }
                Some(_) => steps.push(PlanStep::Layer {
                    layer: layer.id,
                    content: PlanContent::Cpu(layer.id),
                    filters: Vec::new(),
                    mask,
                    opacity,
                    blend,
                }),
                None => {
                    let content = match self.plan_content(layer, channel)? {
                        Some(content) if gpu_filters => content,
                        _ => {
                            steps.push(PlanStep::Layer {
                                layer: layer.id,
                                content: PlanContent::Cpu(layer.id),
                                filters: Vec::new(),
                                mask,
                                opacity,
                                blend,
                            });
                            continue;
                        }
                    };
                    steps.push(PlanStep::Layer {
                        layer: layer.id,
                        content,
                        filters,
                        mask,
                        opacity,
                        blend,
                    });
                }
            }
        }
        Ok(())
    }

    /// A leaf layer's content in data, or `None` when only the CPU renders
    /// it (paint strokes).
    fn plan_content(
        &mut self,
        layer: &'a MaterialLayer,
        channel: MaterialChannel,
    ) -> Result<Option<PlanContent>, BakeError> {
        let Some(fill) = &layer.fill else {
            return Ok(None);
        };
        if !layer.paint_strokes.is_empty() {
            return Ok(None);
        }
        if let Some(graph) = &layer.graph
            && let Some(output) = graph.outputs.get(&channel).copied()
        {
            for resource in graph.resources() {
                self.image(resource)?;
            }
            if graph.needs_mesh_maps() && self.context.mesh_maps.is_none() {
                return Err(BakeError::MissingMeshMaps);
            }
            let index = match self
                .graphs
                .iter()
                .position(|known| std::ptr::eq(*known, graph))
            {
                Some(index) => index,
                None => {
                    self.graphs.push(graph);
                    self.graphs.len() - 1
                }
            };
            return Ok(Some(PlanContent::Graph {
                graph: index,
                output,
            }));
        }
        let projection = fill.projection;
        let Some(resource) = fill.textures.for_channel(channel) else {
            let decal = (projection.mode == crate::ProjectionMode::Decal).then_some(projection);
            if decal.is_some() && self.context.mesh_maps.is_none() {
                return Err(BakeError::MissingMeshMaps);
            }
            return Ok(Some(PlanContent::Uniform {
                texel: fill_texel(fill, channel),
                decal,
            }));
        };
        let image = self.image(resource)?;
        if projection.needs_mesh_maps() && self.context.mesh_maps.is_none() {
            return Err(BakeError::MissingMeshMaps);
        }
        let (_, pixels) = &self.images[image];
        let copy = projection.is_identity_uv()
            && pixels.width == self.set.resolution.width
            && pixels.height == self.set.resolution.height;
        Ok(Some(PlanContent::Bitmap {
            image,
            projection,
            copy,
        }))
    }

    fn plan_mask(&mut self, layer: &MaterialLayer) -> Result<PlanMask, BakeError> {
        let generated = layer.mask_strokes.is_empty()
            && layer.polygon_mask.is_none()
            && layer.id_mask_color.is_none()
            && layer.mask_anchor.is_none()
            && !matches!(
                layer.mask_generator,
                Some(MaskGenerator::IdColor(_) | MaskGenerator::Plugin(_))
            )
            && layer.mask_filters.iter().all(|filter| per_texel(*filter));
        if !generated {
            return Ok(PlanMask::Cpu(layer.id));
        }
        if layer.mask_generator.is_none()
            && layer.mask_bitmap.is_none()
            && !layer.invert_final_mask
            && layer.mask_filters.is_empty()
        {
            return Ok(PlanMask::Open);
        }
        let needs_maps = layer.mask_generator.is_some()
            || layer
                .mask_bitmap
                .is_some_and(|bitmap| bitmap.projection.needs_mesh_maps());
        if needs_maps && self.context.mesh_maps.is_none() {
            return Err(BakeError::MissingMeshMaps);
        }
        let bitmap = match layer.mask_bitmap {
            Some(bitmap) => Some((self.image(bitmap.resource)?, bitmap)),
            None => None,
        };
        Ok(PlanMask::Generated {
            generator: layer.mask_generator,
            bitmap,
            filters: layer.mask_filters.clone(),
            invert: layer.invert_final_mask,
        })
    }
}

/// Filters a plan hands over: per-texel built-in ones. Blurs read
/// neighbours and plugin filters run on the CPU.
fn per_texel(filter: LayerFilter) -> bool {
    filter.is_per_texel() && !matches!(filter, LayerFilter::Plugin(_))
}
