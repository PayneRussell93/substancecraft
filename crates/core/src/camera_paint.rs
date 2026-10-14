//! Camera projection painting: strokes drawn in screen space through the
//! view they were painted from, then projected onto the surface.
//!
//! A texel takes the brush coverage at the screen point its baked surface
//! position projects to, provided the surface there faced the camera and was
//! not hidden behind other geometry. Strokes therefore keep their on-screen
//! shape and size and cross UV seams without breaks, whatever the UV layout.
//!
//! Coverage at a texel depends only on the stroke, the mesh, and the mesh
//! maps, never on the region being baked, so tiled and incremental bakes
//! match a full bake exactly.

use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::brush::{self, BrushTip, CoverageTile, StrokePath};
use crate::mesh_maps::{MeshMapSample, POSITION_BLOCK};
use crate::raster;
use crate::region::TexelRect;
use crate::resources::ResourceImages;
use crate::{MeshGeometry, MeshMaps};

/// Largest viewport edge a stroke camera may record, in pixels.
pub const MAX_STROKE_VIEWPORT: f32 = 16_384.0;
/// Surfaces fade out as they turn away from the camera, over this range of
/// the cosine between their normal and the direction to the camera.
const FACING_FADE: f32 = 0.2;
/// Depth tolerance of the occlusion test, as a fraction of the mesh's depth
/// extent in the view.
const DEPTH_BIAS: f32 = 0.01;

/// The view a stroke was painted through. Its points and radius are in the
/// viewport's pixels instead of texture space.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrokeCamera {
    /// Object space to the viewport: `x` right and `y` down in pixels, `z`
    /// depth (smaller is nearer). The orthographic viewport makes it affine;
    /// perspective transforms work too.
    pub transform: Mat4,
    /// Viewport width and height in pixels. Only what was on screen paints.
    pub viewport: Vec2,
}

impl StrokeCamera {
    pub(crate) fn is_valid(&self) -> bool {
        self.transform.is_finite()
            && self.transform.inverse().is_finite()
            && self.transform.determinant() != 0.0
            && self.viewport.is_finite()
            && self.viewport.cmpge(Vec2::ONE).all()
            && self.viewport.cmple(Vec2::splat(MAX_STROKE_VIEWPORT)).all()
    }

    /// Valid screen-space stroke points: finite and near the viewport.
    pub(crate) fn valid_points(&self, points: &[Vec2]) -> bool {
        let reach = self.viewport * 4.0;
        points
            .iter()
            .all(|point| point.is_finite() && point.abs().cmple(reach).all())
    }

    /// Screen pixel (`x`, `y`) and depth (`z`) of an object-space point.
    pub fn project(&self, position: Vec3) -> Vec3 {
        self.transform.project_point3(position)
    }

    fn size(&self) -> (u32, u32) {
        (self.viewport.x as u32, self.viewport.y as u32)
    }
}

/// What a projected stroke lands on.
#[derive(Clone, Copy)]
pub(crate) struct ProjectionSurface<'a> {
    pub maps: &'a MeshMaps,
    /// Occluders. Without a mesh only facing limits the paint.
    pub mesh: Option<&'a MeshGeometry>,
}

/// A projected stroke's screen coverage and the depth of the nearest
/// surface under it.
struct Footprint {
    camera: StrokeCamera,
    inverse: Mat4,
    /// Inclusive screen-pixel bounds of the stroke, inside the viewport.
    bounds: (i64, i64, i64, i64),
    coverage: CoverageTile,
    /// Nearest depth per pixel of `bounds`, row-major; infinite where no
    /// triangle covers the pixel.
    depth: Vec<f32>,
    bias: f32,
}

impl Footprint {
    fn new(
        path: &StrokePath<'_>,
        camera: StrokeCamera,
        mesh: Option<&MeshGeometry>,
        resources: Option<&ResourceImages>,
    ) -> Option<Self> {
        let (width, height) = camera.size();
        let bounds = brush::stroke_pixel_bounds(
            path.points,
            path.radius_pixels,
            path.tip,
            path.stretch(),
            width,
            height,
        )?;
        let coverage = brush::coverage_tile(path, path.points, bounds, resources)?;
        let mut depth = Vec::new();
        depth
            .try_reserve_exact((coverage.width * coverage.height) as usize)
            .ok()?;
        depth.resize((coverage.width * coverage.height) as usize, f32::INFINITY);
        let mut footprint = Self {
            camera,
            inverse: camera.transform.inverse(),
            bounds,
            coverage,
            depth,
            bias: 0.0,
        };
        if let Some(mesh) = mesh {
            footprint.rasterize_depth(mesh);
        }
        Some(footprint)
    }

    /// Records the nearest surface depth at every pixel of the bounds.
    fn rasterize_depth(&mut self, mesh: &MeshGeometry) {
        let projected: Vec<Vec3> = mesh
            .vertices
            .iter()
            .map(|vertex| self.camera.project(vertex.position))
            .collect();
        let (near, far) = projected
            .iter()
            .filter(|point| point.z.is_finite())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(near, far), point| {
                (near.min(point.z), far.max(point.z))
            });
        self.bias = if far > near {
            (far - near) * DEPTH_BIAS
        } else {
            0.0
        } + 1e-5;
        let (min_x, min_y, max_x, max_y) = self.bounds;
        let width = max_x - min_x + 1;
        for triangle in &mesh.triangles {
            let Some([a, b, c]) = triangle
                .iter()
                .map(|index| projected.get(*index as usize).copied())
                .collect::<Option<Vec<_>>>()
                .and_then(|corners| <[Vec3; 3]>::try_from(corners).ok())
            else {
                continue;
            };
            if !(a.is_finite() && b.is_finite() && c.is_finite()) {
                continue;
            }
            let area = (b.truncate() - a.truncate()).perp_dot(c.truncate() - a.truncate());
            if area.abs() <= 1e-12 {
                continue;
            }
            let lower = a.truncate().min(b.truncate()).min(c.truncate());
            let upper = a.truncate().max(b.truncate()).max(c.truncate());
            let (x0, x1) = (
                (lower.x.floor() as i64).max(min_x),
                (upper.x.ceil() as i64).min(max_x),
            );
            let (y0, y1) = (
                (lower.y.floor() as i64).max(min_y),
                (upper.y.ceil() as i64).min(max_y),
            );
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let point = Vec2::new(x as f32, y as f32);
                    let weight_a = (b.truncate() - point).perp_dot(c.truncate() - point) / area;
                    let weight_b = (c.truncate() - point).perp_dot(a.truncate() - point) / area;
                    let weight_c = 1.0 - weight_a - weight_b;
                    if weight_a < -1e-4 || weight_b < -1e-4 || weight_c < -1e-4 {
                        continue;
                    }
                    let z = a.z * weight_a + b.z * weight_b + c.z * weight_c;
                    let slot = &mut self.depth[((y - min_y) * width + (x - min_x)) as usize];
                    *slot = slot.min(z);
                }
            }
        }
    }

    /// Nearest depth at a pixel; infinite outside the bounds.
    fn depth_at(&self, x: i64, y: i64) -> f32 {
        let (min_x, min_y, max_x, max_y) = self.bounds;
        if x < min_x || y < min_y || x > max_x || y > max_y {
            return f32::INFINITY;
        }
        self.depth[((y - min_y) * (max_x - min_x + 1) + (x - min_x)) as usize]
    }

    /// Whether a surface point at screen position `screen` (with depth) is
    /// the nearest surface there. The four pixels around the point are
    /// consulted and the farthest wins, so a surface is never hidden by its
    /// own neighbours at silhouettes.
    fn visible(&self, screen: Vec3) -> bool {
        let (x, y) = (screen.x.floor() as i64, screen.y.floor() as i64);
        let nearest_hidden = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
            .into_iter()
            .map(|(x, y)| self.depth_at(x, y))
            .fold(f32::NEG_INFINITY, f32::max);
        screen.z <= nearest_hidden + self.bias
    }

    /// Paint coverage of the surface at one mesh-map sample.
    fn coverage_at(&self, sample: &MeshMapSample) -> f32 {
        if !sample.covered {
            return 0.0;
        }
        let screen = self.camera.project(sample.position);
        if !screen.is_finite() {
            return 0.0;
        }
        let coverage = self.coverage.sample(screen.truncate());
        if coverage <= 0.0 || !self.visible(screen) {
            return 0.0;
        }
        // Direction toward the camera: along the view ray, to smaller depth.
        let toward =
            self.inverse.project_point3(screen - Vec3::Z) - self.inverse.project_point3(screen);
        let facing = sample.normal.dot(toward.normalize_or_zero());
        coverage * (facing / FACING_FADE).clamp(0.0, 1.0)
    }
}

/// Texel rectangles of a `width` × `height` texture whose surface may
/// project into the inclusive screen `bounds`: a superset, found from the
/// mesh maps' position blocks.
fn candidate_rects(
    camera: &StrokeCamera,
    maps: &MeshMaps,
    bounds: (i64, i64, i64, i64),
    width: u32,
    height: u32,
) -> Vec<TexelRect> {
    let (min_x, min_y, max_x, max_y) = bounds;
    let (lower, upper) = (
        Vec2::new(min_x as f32 - 1.0, min_y as f32 - 1.0),
        Vec2::new(max_x as f32 + 1.0, max_y as f32 + 1.0),
    );
    // Texture texel `t` samples mesh-map texel `t * ratio` (see
    // `raster::pixel_to_uv` and `MeshMaps::sample`).
    let ratio = |size: u32, maps: u32| {
        if maps <= 1 || size <= 1 {
            None
        } else {
            Some((size - 1) as f32 / (maps - 1) as f32)
        }
    };
    let (ratio_x, ratio_y) = (ratio(width, maps.width()), ratio(height, maps.height()));
    let span = |start: u32, ratio: Option<f32>, size: u32| match ratio {
        Some(ratio) => {
            let first = ((start as f32 * ratio).floor() as i64 - 1).max(0);
            let last =
                (((start + POSITION_BLOCK) as f32 * ratio).ceil() as i64 + 1).min(i64::from(size));
            (first as u32, last.max(first) as u32)
        }
        None => (0, size),
    };
    let mut rects = Vec::new();
    for block in maps.position_blocks() {
        let mut screen_min = Vec2::splat(f32::INFINITY);
        let mut screen_max = Vec2::splat(f32::NEG_INFINITY);
        let mut finite = true;
        for corner in 0..8 {
            let point = Vec3::new(
                if corner & 1 == 0 {
                    block.min.x
                } else {
                    block.max.x
                },
                if corner & 2 == 0 {
                    block.min.y
                } else {
                    block.max.y
                },
                if corner & 4 == 0 {
                    block.min.z
                } else {
                    block.max.z
                },
            );
            let screen = camera.project(point).truncate();
            finite &= screen.is_finite();
            screen_min = screen_min.min(screen);
            screen_max = screen_max.max(screen);
        }
        if finite && (screen_max.cmplt(lower).any() || screen_min.cmpgt(upper).any()) {
            continue;
        }
        let (x0, x1) = span(block.x, ratio_x, width);
        let (y0, y1) = span(block.y, ratio_y, height);
        if x1 > x0 && y1 > y0 {
            rects.push(TexelRect {
                x: x0,
                y: y0,
                width: x1 - x0,
                height: y1 - y0,
            });
        }
    }
    rects
}

/// Visits every texel of `clip` a camera-projected stroke paints, with its
/// coverage in `0..=1`; the index is row-major within `clip`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rasterize_projected_stroke(
    path: &StrokePath<'_>,
    camera: &StrokeCamera,
    surface: ProjectionSurface<'_>,
    width: u32,
    height: u32,
    resources: Option<&ResourceImages>,
    clip: TexelRect,
    mut apply: impl FnMut(usize, f32),
) {
    if path.points.is_empty() || width == 0 || height == 0 || clip.is_empty() {
        return;
    }
    let Some(footprint) = Footprint::new(path, *camera, surface.mesh, resources) else {
        return;
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(clip.len()).is_err() {
        return;
    }
    // Negative marks texels not evaluated yet; candidate rectangles overlap.
    values.resize(clip.len(), -1.0_f32);
    for rect in candidate_rects(camera, surface.maps, footprint.bounds, width, height) {
        let Some(rect) = rect.intersect(clip) else {
            continue;
        };
        for y in rect.y..rect.bottom() {
            for x in rect.x..rect.right() {
                let index = clip.local_index(x, y);
                if values[index] < 0.0 {
                    let uv = raster::pixel_to_uv(x, y, width, height);
                    values[index] = footprint.coverage_at(&surface.maps.sample(uv));
                }
            }
        }
    }
    for (index, value) in values.into_iter().enumerate() {
        if value > 0.0 {
            apply(index, value.min(1.0));
        }
    }
}

/// Texels a camera-projected stroke through screen `points` can paint on a
/// `width` × `height` texture, as a few rectangles (a superset: occlusion
/// and facing are not considered). Stamps are placed causally, so the
/// texels that change when a point is appended are those of the last two
/// points alone.
pub fn projected_stroke_texel_rects(
    points: &[Vec2],
    radius_pixels: f32,
    tip: BrushTip,
    camera: &StrokeCamera,
    maps: &MeshMaps,
    width: u32,
    height: u32,
) -> Vec<TexelRect> {
    let (screen_width, screen_height) = camera.size();
    // Projected strokes never carry stamp frames.
    match brush::stroke_pixel_bounds(points, radius_pixels, tip, 1.0, screen_width, screen_height) {
        Some(bounds) if width > 0 && height > 0 => {
            candidate_rects(camera, maps, bounds, width, height)
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::BTreeSet;

    use glam::{Quat, Vec4};

    use super::*;
    use crate::{
        BakeContext, DirtyTexels, LayerId, MaskStroke, MaterialChannel, MaterialLayer,
        MeshMapSettings, PaintStroke, PrimitivePreset, TextureResolution, TextureSet, TextureSetId,
        bake_mesh_maps,
    };

    /// An orthographic camera like the viewport's: `scale` pixels per unit,
    /// the mesh centred in a `size` × `size` viewport, looking down `-Z`
    /// after `rotation`.
    fn camera(rotation: Quat, scale: f32, size: f32) -> StrokeCamera {
        let screen = Mat4::from_cols(
            Vec4::new(scale, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -scale, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -1.0, 0.0),
            Vec4::new(size / 2.0, size / 2.0, 0.0, 1.0),
        );
        StrokeCamera {
            transform: screen * Mat4::from_quat(rotation),
            viewport: Vec2::splat(size),
        }
    }

    fn set(size: u32, layers: Vec<MaterialLayer>) -> TextureSet {
        TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Body".to_owned(),
            material_slot: "Body".to_owned(),
            resolution: TextureResolution {
                width: size,
                height: size,
            },
            channels: BTreeSet::from([MaterialChannel::BaseColor]),
            layers,
            height_strength: crate::DEFAULT_HEIGHT_STRENGTH,
            export: None,
        }
    }

    fn cube_maps() -> (MeshGeometry, MeshMaps) {
        let mesh = PrimitivePreset::Cube.build_mesh();
        let maps = bake_mesh_maps(
            &mesh,
            &MeshMapSettings {
                resolution: 64,
                ao_samples: 1,
                ..MeshMapSettings::default()
            },
            None,
        )
        .unwrap();
        (mesh, maps)
    }

    fn red_stroke(points: Vec<Vec2>, radius: f32, camera: StrokeCamera) -> PaintStroke {
        PaintStroke {
            camera: Some(camera),
            ..PaintStroke::new(points, radius, [255, 0, 0, 255])
        }
    }

    /// Fraction of the texels in UV triangles of each cube face that a
    /// texture-space mask marks as painted, keyed by the face's normal.
    fn painted_by_face(
        mesh: &MeshGeometry,
        texture: &crate::BakedTexture,
        painted: impl Fn([u8; 4]) -> bool,
    ) -> Vec<(Vec3, f32)> {
        let mut faces: Vec<(Vec3, usize, usize)> = Vec::new();
        for triangle in &mesh.triangles {
            let [a, b, c] = triangle.map(|index| mesh.vertices[index as usize]);
            let normal = (b.position - a.position)
                .cross(c.position - a.position)
                .normalize();
            let entry = match faces.iter().position(|face| face.0.dot(normal) > 0.99) {
                Some(index) => index,
                None => {
                    faces.push((normal, 0, 0));
                    faces.len() - 1
                }
            };
            raster::rasterize_uv_triangle(
                [a.uv, b.uv, c.uv],
                texture.width,
                texture.height,
                0.0,
                |index, _, interior| {
                    if interior {
                        let pixel = &texture.rgba8[index * 4..index * 4 + 4];
                        faces[entry].1 += 1;
                        if painted([pixel[0], pixel[1], pixel[2], pixel[3]]) {
                            faces[entry].2 += 1;
                        }
                    }
                },
            );
        }
        faces
            .into_iter()
            .map(|(normal, total, painted)| (normal, painted as f32 / total.max(1) as f32))
            .collect()
    }

    fn painted_fraction(faces: &[(Vec3, f32)], normal: Vec3) -> f32 {
        faces
            .iter()
            .find(|face| face.0.dot(normal) > 0.99)
            .map_or(0.0, |face| face.1)
    }

    #[test]
    fn tc_0272_projected_strokes_paint_only_visible_surfaces_facing_the_camera() {
        let (mesh, maps) = cube_maps();
        // Looking straight at +Z: a big dab over the whole cube.
        let front = camera(Quat::IDENTITY, 100.0, 400.0);
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Projected");
        layer.channels = BTreeSet::from([MaterialChannel::BaseColor]);
        layer
            .paint_strokes
            .push(red_stroke(vec![Vec2::splat(200.0)], 150.0, front));
        let front_set = set(128, vec![layer]);
        assert!(front_set.requires_mesh_maps(), "projection reads mesh maps");
        assert_eq!(
            front_set.bake_channel(MaterialChannel::BaseColor, BakeContext::default()),
            Err(crate::BakeError::MissingMeshMaps)
        );
        let context = BakeContext {
            mesh: Some(&mesh),
            mesh_maps: Some(&maps),
            resources: None,
        };
        let texture = front_set
            .bake_channel(MaterialChannel::BaseColor, context)
            .unwrap();
        let faces = painted_by_face(&mesh, &texture, |pixel| pixel[3] > 0);
        assert!(
            painted_fraction(&faces, Vec3::Z) > 0.95,
            "the facing side is painted: {faces:?}"
        );
        for side in [Vec3::NEG_Z, Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y] {
            assert!(
                painted_fraction(&faces, side) < 0.05,
                "{side} faces away or sideways: {faces:?}"
            );
        }

        // Without occluders, a face turned away is still refused by facing;
        // seen at 45°, two faces share the dab across their seam.
        let corner = camera(Quat::from_rotation_y(-45f32.to_radians()), 100.0, 400.0);
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Corner");
        layer.channels = BTreeSet::from([MaterialChannel::BaseColor]);
        layer
            .paint_strokes
            .push(red_stroke(vec![Vec2::splat(200.0)], 40.0, corner));
        let texture = set(128, vec![layer])
            .bake_channel(MaterialChannel::BaseColor, context)
            .unwrap();
        let faces = painted_by_face(&mesh, &texture, |pixel| pixel[3] > 0);
        let (z, x) = (
            painted_fraction(&faces, Vec3::Z),
            painted_fraction(&faces, Vec3::X),
        );
        assert!(z > 0.02 && x > 0.02, "the dab crosses the edge: {faces:?}");
        assert!((z - x).abs() < 0.05, "symmetric about the edge: {faces:?}");
        assert!(painted_fraction(&faces, Vec3::NEG_Z) == 0.0);
    }

    #[test]
    fn tc_0274_projected_strokes_validate_round_trip_and_keep_screen_space_sizes() {
        let view = camera(Quat::IDENTITY, 100.0, 400.0);
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Projected");
        layer
            .paint_strokes
            .push(red_stroke(vec![Vec2::new(150.0, 380.0)], 24.0, view));
        layer.mask_strokes.push(MaskStroke {
            camera: Some(view),
            ..MaskStroke::new(vec![Vec2::new(390.0, 10.0)], 12.0, 0.0)
        });
        layer.paint_strokes.push(PaintStroke::new(
            vec![Vec2::new(1.5, 0.5)],
            8.0,
            [0, 0, 255, 255],
        ));
        let project = crate::MaterialProject {
            texture_sets: vec![set(8192, vec![layer])],
            ..crate::MaterialProject::empty()
        };
        project.validate().unwrap();

        // Screen-space points may run past the unit square that bounds UV
        // strokes; broken cameras and far-off points are refused.
        let broken = |edit: &dyn Fn(&mut PaintStroke)| {
            let mut copy = project.clone();
            edit(&mut copy.texture_sets[0].layers[0].paint_strokes[0]);
            copy.validate()
        };
        let refused = Err(crate::ProjectError::InvalidLayerPaintData(LayerId(1)));
        for edit in [
            &(|stroke: &mut PaintStroke| {
                if let Some(camera) = &mut stroke.camera {
                    camera.viewport = Vec2::ZERO;
                }
            }) as &dyn Fn(&mut PaintStroke),
            &|stroke| {
                if let Some(camera) = &mut stroke.camera {
                    camera.transform.x_axis.x = f32::NAN;
                }
            },
            &|stroke| {
                if let Some(camera) = &mut stroke.camera {
                    camera.transform = Mat4::ZERO;
                }
            },
            &|stroke| stroke.points.push(Vec2::new(1e7, 0.0)),
        ] {
            assert_eq!(broken(edit), refused);
        }

        // Projects round-trip the camera; strokes without one omit it.
        let json = serde_json::to_string(&project).unwrap();
        assert_eq!(json.matches("\"camera\"").count(), 2);
        let decoded: crate::MaterialProject = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, project);

        // A downscaled preview keeps screen-space radii and scales UV ones;
        // a UDIM shift moves only UV strokes.
        let set = &project.texture_sets[0];
        let preview = set.downscaled(2048).unwrap();
        let strokes = &preview.layers[0].paint_strokes;
        assert_eq!(strokes[0].radius_pixels, 24.0);
        assert_eq!(strokes[1].radius_pixels, 2.0);
        assert_eq!(preview.layers[0].mask_strokes[0].radius_pixels, 12.0);
        let shifted = set.shifted_to_tile(crate::UdimTile(1002));
        let strokes = &shifted.layers[0].paint_strokes;
        assert_eq!(strokes[0].points, [Vec2::new(150.0, 380.0)]);
        assert_eq!(strokes[1].points, [Vec2::new(0.5, 0.5)]);
        assert_eq!(
            shifted.layers[0].mask_strokes[0].points,
            [Vec2::new(390.0, 10.0)]
        );
    }

    #[test]
    fn tc_0273_occluded_surfaces_stay_unpainted_and_dirty_rects_cover_every_change() {
        // Two stacked planes, both facing the camera: the front one hides
        // the back one.
        let mut mesh = PrimitivePreset::Plane.build_mesh();
        for vertex in &mut mesh.vertices {
            // The plane lies in XZ facing +Y; stand it up facing +Z.
            vertex.position = Vec3::new(vertex.position.x, -vertex.position.z, 0.5);
            vertex.uv.x *= 0.5;
        }
        let mut back = mesh.clone();
        for vertex in &mut back.vertices {
            vertex.position.z = -0.5;
            vertex.uv.x += 0.5;
        }
        mesh.append(&back).unwrap();
        let maps = bake_mesh_maps(
            &mesh,
            &MeshMapSettings {
                resolution: 64,
                ao_samples: 1,
                ..MeshMapSettings::default()
            },
            None,
        )
        .unwrap();
        let view = camera(Quat::IDENTITY, 100.0, 400.0);
        let mut layer = MaterialLayer::new_paint(LayerId(1), "Through");
        layer.channels = BTreeSet::from([MaterialChannel::BaseColor]);
        layer
            .paint_strokes
            .push(red_stroke(vec![Vec2::splat(200.0)], 30.0, view));
        let mut masked = MaterialLayer::new_fill(
            LayerId(2),
            "Masked",
            crate::LayerFill {
                color: [255, 0, 0, 255],
                ..crate::LayerFill::default()
            },
        );
        masked.channels = BTreeSet::from([MaterialChannel::BaseColor]);
        masked.mask_strokes.push(MaskStroke {
            camera: Some(view),
            tip: BrushTip {
                hardness: 0.8,
                ..BrushTip::ROUND
            },
            ..MaskStroke::new(vec![Vec2::splat(200.0)], 30.0, 0.0)
        });
        let painted = set(64, vec![layer.clone()]);
        let context = BakeContext {
            mesh: Some(&mesh),
            mesh_maps: Some(&maps),
            resources: None,
        };
        let texture = painted
            .bake_channel(MaterialChannel::BaseColor, context)
            .unwrap();
        let alpha = |u: f32, v: f32| texture.sample_uv(Vec2::new(u, v)).unwrap()[3];
        assert!(alpha(0.25, 0.5) > 200, "front plane centre is painted");
        assert_eq!(alpha(0.75, 0.5), 0, "the hidden back plane is not");
        // Without the mesh as an occluder the back plane, which faces the
        // camera too, would be painted: occlusion is what keeps it clean.
        let unoccluded = painted
            .bake_channel(
                MaterialChannel::BaseColor,
                BakeContext {
                    mesh: None,
                    ..context
                },
            )
            .unwrap();
        assert!(unoccluded.sample_uv(Vec2::new(0.75, 0.5)).unwrap()[3] > 200);

        // Mask strokes project too: clearing the mask through the view
        // removes the fill from the front plane only.
        let mask = set(64, vec![masked])
            .bake_channel(MaterialChannel::BaseColor, context)
            .unwrap();
        let alpha = |u: f32, v: f32| mask.sample_uv(Vec2::new(u, v)).unwrap()[3];
        assert_eq!(alpha(0.25, 0.5), 0, "front centre is masked out");
        assert_eq!(
            alpha(0.03, 0.5),
            255,
            "front edge, outside the dab, keeps it"
        );
        assert_eq!(alpha(0.75, 0.5), 255, "the hidden back plane keeps it");

        // Region bakes and dirty rects: a stroke grown by one point changes
        // only texels inside the rects reported for its last two points.
        let mut grown = layer.clone();
        grown.paint_strokes[0].points.push(Vec2::new(260.0, 230.0));
        let before = set(64, vec![layer]);
        let after = set(64, vec![grown]);
        let full_before = before.bake_material(context).unwrap();
        let full_after = after.bake_material(context).unwrap();
        let rects = projected_stroke_texel_rects(
            &after.layers[0].paint_strokes[0].points[..],
            30.0,
            BrushTip::ROUND,
            &view,
            &maps,
            64,
            64,
        );
        let mut dirty = DirtyTexels::new(64, 64);
        for rect in &rects {
            dirty.mark(*rect);
        }
        for index in 0..64 * 64 {
            let (x, y) = ((index % 64) as u32, (index / 64) as u32);
            let changed = full_before.base_color.rgba8[index * 4..index * 4 + 4]
                != full_after.base_color.rgba8[index * 4..index * 4 + 4];
            if changed {
                assert!(
                    rects.iter().any(|rect| rect.contains(x, y)),
                    "texel {x},{y} changed outside the dirty rects"
                );
            }
        }
        let mut patched = full_before.clone();
        for rect in dirty.rects() {
            after
                .rebake_material_region(context, &mut patched, rect)
                .unwrap();
        }
        assert_eq!(patched, full_after, "patched preview equals a full bake");
    }
}
