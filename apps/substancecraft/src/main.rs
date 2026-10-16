#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
};
use std::time::Duration;

use eframe::egui;
#[cfg(test)]
use substancecraft_api::load_project_file;
use substancecraft_api::load_resource_image;
use substancecraft_api::{
    ImportedMesh, MAX_FBX_BYTES, TextureFormat, bake_texture_set_channel, bake_texture_set_export,
    bake_texture_set_material, existing_export_files, export_preflight, import_fbx_meshes,
    import_gltf_file, load_project_file_with_report, load_project_resources, load_texture_image,
    rebake_texture_set_channel_regions, rebake_texture_set_material_plane_regions,
    save_project_file, write_exported_textures, write_texture,
};
use substancecraft_api::{load_project_history, save_project_history};
use substancecraft_core::{
    BakedTexture, BoneId, BoneTransform, BrushTip, DEFAULT_EMISSIVE, DEFAULT_HEIGHT,
    DEFAULT_HEIGHT_STRENGTH, DEFAULT_OPACITY, Dirt, EdgeWear, EditCommand, EditHistory,
    ExportPreset, ExportedTexture, IdStroke, LayerFill, LayerId, LayerSettings, LayerSlot,
    MAX_STROKE_POINTS, MaskGenerator, MaskStroke, MaterialChannel, MaterialLayer, MaterialProject,
    MaterialTextures, MeshAsset, MeshGeometry, MeshId, MeshMapError, MeshMapKind, MeshMapSettings,
    MeshMaps, MeshSmoothing, MeshTopology, NamedPose, PaintStroke, PoseId, PreflightIssue,
    PreflightSeverity, PrimitivePreset, ProjectResource, ResourceId, ResourceImages, SelectionMode,
    ShaderDefinition, TexelRect, TextureImage, TextureResolution, TextureSet, TextureSetId,
    ViewportCamera, ViewportLighting, remap_subdivided_triangles, stroke_texel_bounds,
};
use substancecraft_filament::{FilamentRuntime, MaterialMap};

mod animation;
mod autosave;
mod bake_log;
mod cache;
mod cameras;
mod diagnostics;
mod edges;
mod environments;
mod geometry_visibility;
mod gltf_material;
mod grading;
mod graph_editor;
mod icons;
mod incremental;
mod inspector;
mod layouts;
mod library_panel;
mod ocio_view;
mod panels;
mod preferences;
mod relink;
mod resource_watch;
mod sequences;
mod stroke_tools;
mod theme;
mod turntable;
mod udim_view;
mod uv_view;
mod widgets;

const CHANNELS: [MaterialChannel; 9] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Id,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Normal,
    MaterialChannel::Height,
    MaterialChannel::AmbientOcclusion,
    MaterialChannel::Emissive,
    MaterialChannel::Opacity,
];
/// Channels the compositor can bake, preview, and export.
const BAKEABLE_CHANNELS: [MaterialChannel; 8] = [
    MaterialChannel::BaseColor,
    MaterialChannel::Roughness,
    MaterialChannel::Metallic,
    MaterialChannel::Height,
    MaterialChannel::Normal,
    MaterialChannel::Emissive,
    MaterialChannel::Opacity,
    MaterialChannel::Id,
];
/// Mesh formats the importer reads.
const MESH_EXTENSIONS: [&str; 5] = ["fbx", "gltf", "glb", "step", "stp"];
/// Bitmap formats the resource importer reads.
const IMAGE_EXTENSIONS: [&str; 12] = [
    "png", "jpg", "jpeg", "tga", "bmp", "tif", "tiff", "webp", "exr", "hdr", "svg", "sctext",
];
const POSE_TRANSITION_SECONDS: f64 = 0.35;
const MAX_SAVED_POSES: usize = 256;
/// Edge of smart-material thumbnails in the browser, in pixels.
const SMART_THUMBNAIL_SIZE: u32 = 72;
/// Selection overlays stop drawing beyond this many triangles to keep the
/// viewport responsive on dense meshes.
const MAX_OVERLAY_TRIANGLES: usize = 200_000;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("SubstanceCraft")
            .with_app_id("com.substancecraft.desktop")
            .with_icon(app_icon())
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "SubstanceCraft",
        options,
        Box::new(|creation_context| {
            theme::apply(&creation_context.egui_ctx);
            let mut app = SubstanceCraftApp::default();
            // The UI scale preference owns zoom; egui's Ctrl+/- would fight it.
            creation_context
                .egui_ctx
                .options_mut(|options| options.zoom_with_keyboard = false);
            if let Some(path) = preferences::Preferences::default_path() {
                app.load_preferences(path);
            }
            match FilamentRuntime::new(64, 64) {
                Ok(renderer) => app.filament_renderer = Some(renderer),
                Err(error) => app.filament_error = Some(error.to_string()),
            }
            if let Some(folder) = substancecraft_api::RecoveryStore::default_location() {
                app.start_recovery(substancecraft_api::RecoveryStore::new(folder));
            }
            // `substancecraft path/to/scene.sproject` opens that project.
            if let Some(path) = std::env::args_os().nth(1) {
                app.open_project_path(path.into());
            }
            Ok(Box::new(app))
        }),
    )
}

fn app_icon() -> egui::IconData {
    let size = 64_usize;
    let mut rgba = vec![0_u8; size * size * 4];
    for pixel in rgba.as_chunks_mut::<4>().0 {
        pixel.copy_from_slice(&[18, 27, 31, 255]);
    }

    fill_icon_polygon(
        &mut rgba,
        size,
        &[(32, 8), (54, 20), (32, 32), (10, 20)],
        [103, 220, 190, 255],
    );
    fill_icon_polygon(
        &mut rgba,
        size,
        &[(10, 20), (32, 32), (32, 56), (10, 44)],
        [39, 145, 136, 255],
    );
    fill_icon_polygon(
        &mut rgba,
        size,
        &[(54, 20), (32, 32), (32, 56), (54, 44)],
        [27, 92, 112, 255],
    );

    egui::IconData {
        rgba,
        width: size as u32,
        height: size as u32,
    }
}

fn fill_icon_polygon(rgba: &mut [u8], size: usize, points: &[(i32, i32)], color: [u8; 4]) {
    let min_y = points.iter().map(|point| point.1).min().unwrap_or(0);
    let max_y = points.iter().map(|point| point.1).max().unwrap_or(0);

    for y in min_y..max_y {
        let mut intersections = Vec::new();
        for index in 0..points.len() {
            let (x1, y1) = points[index];
            let (x2, y2) = points[(index + 1) % points.len()];
            if (y1 <= y && y < y2) || (y2 <= y && y < y1) {
                intersections.push(x1 + (y - y1) * (x2 - x1) / (y2 - y1));
            }
        }
        intersections.sort_unstable();

        for pair in intersections.as_chunks::<2>().0 {
            for x in pair[0]..pair[1] {
                let offset = (y as usize * size + x as usize) * 4;
                rgba[offset..offset + 4].copy_from_slice(&color);
            }
        }
    }
}

struct SubstanceCraftApp {
    project: MaterialProject,
    selected_texture_set: Option<TextureSetId>,
    selected_layer: Option<LayerId>,
    /// Layers Ctrl-clicked alongside `selected_layer`, which "Flatten to
    /// bitmaps" merges with it. Never holds `selected_layer` itself.
    extra_selected_layers: BTreeSet<LayerId>,
    /// Which decal (0 = fill, 1 = bitmap mask) of the selected layer the
    /// viewport gizmo drags, while it is on.
    decal_gizmo: Option<u8>,
    /// The brush colour a clipped colour-space entry produced, while it is
    /// still the brush colour.
    picker_clipped: Option<egui::Color32>,
    /// Sphere and mesh maps smart-material thumbnails are shaded on, built
    /// the first time the browser shows.
    thumbnail_stage: Option<Arc<substancecraft_core::ThumbnailStage>>,
    /// Smart-material thumbnails by name, with a hash of the layer tree
    /// they show.
    smart_thumbnails: BTreeMap<String, (u64, Arc<TextureImage>)>,
    /// Uploaded thumbnails, by name and hash.
    smart_thumbnail_textures: BTreeMap<String, (u64, egui::TextureHandle)>,
    selected_primitive: Option<PrimitivePreset>,
    /// Imported or primitive geometry after scaling, before smoothing.
    base_geometry: Option<MeshGeometry>,
    /// Smoothed/subdivided geometry used for display, painting, and baking.
    mesh_geometry: Option<Arc<MeshGeometry>>,
    mesh_topology: Option<MeshTopology>,
    /// How the viewport shows UDIM tiles beyond 1001, when the mesh has any.
    udim_atlas: Option<substancecraft_core::UdimAtlas>,
    /// Baked UDIM tiles and tile mesh maps, shared with background bakes.
    udim_cache: Arc<std::sync::Mutex<udim_view::TileCache>>,
    mesh_center: glam::Vec3,
    mesh_smoothing: MeshSmoothing,
    mesh_error: Option<String>,
    mesh_revision: u64,
    filament_mesh_revision: Option<u64>,
    /// Hidden-face revision the renderer last drew.
    filament_hidden_revision: Option<u64>,
    /// Faces hidden from the viewport (a view setting, never saved).
    hidden_geometry: geometry_visibility::HiddenGeometry,
    /// Environment the renderer was last given, and the lux and rotation it
    /// was applied at.
    filament_environment: Option<(environments::EnvironmentSource, f32, f32)>,
    /// A loaded HDR panorama lighting the viewport instead of the preset.
    custom_environment: Option<environments::CustomEnvironment>,
    /// Display LUT applied after the view transform.
    display_lut: Option<grading::DisplayLut>,
    /// OpenColorIO view grading the viewport, in place of the display LUT.
    ocio: Option<ocio_view::OcioState>,
    /// Why the display LUT did not load, when it did not.
    grading_status: Option<String>,
    /// Shader definitions from the user shader folder.
    user_shaders: Vec<ShaderDefinition>,
    /// Shader files skipped or not loaded, and why.
    shader_status: Option<String>,
    /// Loads of custom environments so far.
    environment_revision: u64,
    /// Result of the last custom environment load.
    environment_status: Option<String>,
    filament_renderer: Option<FilamentRuntime>,
    filament_error: Option<String>,
    filament_frame_texture: Option<egui::TextureHandle>,
    filament_frames_tx: SyncSender<FilamentFrame>,
    filament_frames_rx: Receiver<FilamentFrame>,
    filament_render_pending: Arc<AtomicBool>,
    render_request: Option<RenderRequest>,
    /// Pixels of the finished still render, filled by the renderer.
    render_result: Arc<std::sync::Mutex<Option<FilamentFrame>>>,
    /// Outcome of the last still render, for the Light drawer.
    render_status: Option<String>,
    /// A turntable sequence being rendered, one still at a time.
    turntable: Option<turntable::TurntableJob>,
    turntable_mode: turntable::TurntableMode,
    turntable_frames: u32,
    /// The project mesh reference the loaded geometry reflects; undo and redo
    /// reload the mesh when the project's stops matching it.
    loaded_mesh: Option<MeshAsset>,
    imported_meshes: Vec<ImportedMesh>,
    selected_imported_mesh: Option<usize>,
    /// How the viewport mesh is posed by its imported rig.
    animation: animation::AnimationState,
    /// The posed viewport mesh, kept until the pose or mesh changes.
    posed_mesh: Option<animation::PosedMesh>,
    /// Pose of the vertices Filament has: `Some(None)` is the bind pose.
    filament_pose: Option<Option<animation::PoseKey>>,
    import_error: Option<String>,
    import_scale: f32,
    project_path: Option<std::path::PathBuf>,
    project_error: Option<String>,
    active_channel: MaterialChannel,
    next_texture_set_id: u64,
    next_layer_id: u64,
    current_pose: NamedPose,
    viewport_camera: ViewportCamera,
    viewport_lighting: ViewportLighting,
    paint_mode: PaintMode,
    brush_radius: f32,
    brush_color: egui::Color32,
    brush_roughness: f32,
    brush_metallic: f32,
    brush_height: f32,
    brush_emissive: egui::Color32,
    brush_material_opacity: f32,
    brush_tip: BrushTip,
    /// Latest pen pressure (`0..=1`) while a stylus touches; `None` for mice.
    pen_pressure: Option<f32>,
    /// Loaded pixels of `project.resources`.
    resource_images: ResourceImages,
    /// Per-resource egui thumbnails for the assets drawer.
    resource_thumbnails: BTreeMap<ResourceId, egui::TextureHandle>,
    /// Notices bitmap files edited outside the app.
    resource_watcher: resource_watch::ResourceWatcher,
    export_format: TextureFormat,
    /// Name of the export preset the Export button uses.
    export_preset: String,
    /// Custom preset being edited; applied to the project only when valid.
    preset_draft: Option<PresetDraft>,
    /// Export waiting for permission to replace existing files.
    overwrite_prompt: Option<OverwritePrompt>,
    view_mode: ViewMode,
    uv_view: uv_view::UvView,
    /// Content revision of each texture set; compared against caches.
    set_revisions: BTreeMap<TextureSetId, u64>,
    /// Last full-material bake of each texture set, for drawing material
    /// slots other than the selected one.
    set_materials: BTreeMap<TextureSetId, CachedMaterial>,
    set_bake_job: Option<SetBakeJob>,
    /// What each set's entry in `set_materials` is missing.
    set_dirty: BTreeMap<TextureSetId, SetDirty>,
    /// Set revisions whose background bake failed, so they are not retried.
    set_bake_failed: BTreeMap<TextureSetId, u64>,
    /// When true the viewport shows `active_channel` flat instead of the full
    /// PBR material.
    channel_preview: bool,
    mask_value: f32,
    selection_mode: SelectionMode,
    hovered_selection: Vec<u32>,
    active_stroke: Option<(TextureSetId, LayerId, PaintMode)>,
    /// Index in the layer's stroke list of the first stroke this drag made.
    stroke_first_index: usize,
    /// Stroke index of each path of the current drag (brush, mirror).
    stroke_paths: Vec<Option<usize>>,
    /// Last triangle each path painted, to detect UV seam crossings.
    stroke_last_triangles: Vec<Option<u32>>,
    /// Mirror painting plane, when on.
    symmetry: Option<substancecraft_core::Symmetry>,
    /// Lazy-mouse brush smoothing (radius 0 is off).
    stroke_stabilizer: substancecraft_core::LazyStabilizer,
    /// Clone, smudge, surface alignment, and the quick mask.
    stroke_tools: stroke_tools::StrokeTools,
    /// What the brush paints into each user channel.
    brush_user: substancecraft_core::UserValues,
    /// Screen point where the last brush dab landed, for Shift-click lines.
    last_paint_screen: Option<egui::Pos2>,
    /// Control points of a path being placed: UV, the clicked screen point,
    /// and the object-space surface point under it.
    path_controls: Option<Vec<(glam::Vec2, egui::Pos2, glam::Vec3)>>,
    /// The next click samples the material into the brush.
    picking_material: bool,
    /// Bitmap scaling skew correction per texel in high-to-low bakes.
    skew_map: Option<ResourceId>,
    skew_map_revision: u64,
    /// Search text of the Assets drawer.
    asset_query: String,
    /// Search text for filtering the layer stack.
    layer_query: String,
    /// Imported bitmaps waiting for channel confirmation before fill creation.
    pending_import_assignments: Vec<ImportAssignment>,
    history: EditHistory,
    /// Last committed settings of the selected layer; edits made through the
    /// inspector are recorded as one history step once the pointer is up.
    layer_settled: Option<(TextureSetId, LayerId, LayerSettings)>,
    /// Project material open in the Assets drawer's master editor.
    editing_material: Option<substancecraft_core::MaterialId>,
    /// A template waiting for confirmation before it replaces the project.
    pending_template: Option<substancecraft_core::ProjectTemplate>,
    /// Paint, erase, and mask strokes are drawn on screen and projected
    /// onto the visible surface (camera projection painting).
    projection_painting: bool,
    /// The next model click sets the selected layer's ID generator colour.
    picking_id_color: bool,
    /// The start state of an evolution control being recorded on a layer.
    evolution_start: Option<(TextureSetId, LayerId, MaterialLayer)>,
    /// Name the control being recorded will get.
    evolution_name: String,
    baked_preview: Option<BakedPreview>,
    preview_revision: u64,
    preview_job: Option<PreviewJob>,
    /// Texels strokes changed since the shown preview was baked, so the next
    /// rebake can patch just those (PERF-04).
    preview_dirty: Option<incremental::PreviewDirty>,
    preview_failed: bool,
    bake_error: Option<String>,
    mesh_map_settings: MeshMapSettings,
    /// Mesh maps per mesh material slot, baked from each slot's triangles.
    mesh_maps: Option<Arc<Vec<MeshMaps>>>,
    mesh_maps_key: Option<BakeKey>,
    high_poly: Option<HighPolySource>,
    /// The cage mesh file, scaled but not yet smoothed.
    cage_mesh: Option<(std::path::PathBuf, MeshGeometry)>,
    high_poly_revision: u64,
    mesh_map_job: Option<MeshMapJob>,
    mesh_map_error: Option<String>,
    mesh_map_preview: Option<MeshMapKind>,
    selected_pose: Option<PoseId>,
    pose_name: String,
    pose_transition: Option<PoseTransition>,
    drawer: Option<panels::Drawer>,
    last_drawer: panels::Drawer,
    show_inspector: bool,
    /// Name typed for saving the current panel layout.
    layout_name: String,
    layout: panels::OverlayLayout,
    autosave: autosave::Autosave,
    preferences: preferences::PreferenceState,
    inspect: inspector::InspectState,
    edge_overlay: edges::EdgeOverlay,
    /// Resource paths as of the last reload, to notice relinks by undo.
    loaded_resource_sources: Vec<(ResourceId, String)>,
    bake_log: bake_log::BakeLog,
    cache: cache::CacheState,
    library: library_panel::LibraryState,
}

struct FilamentFrame {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

/// A still render of the viewport waiting to be drawn and saved.
struct RenderRequest {
    path: std::path::PathBuf,
    /// Longest edge of the image in pixels.
    long_edge: u32,
    /// Camera to render from; `None` uses the viewport camera.
    camera: Option<ViewportCamera>,
    /// Environment rotation to render with; `None` uses the viewport's.
    lighting_rotation: Option<f32>,
    /// The turntable frame this render is, if any.
    turntable_frame: Option<u32>,
    issued: bool,
}

/// Pixel size of a still render with the viewport's aspect and the given
/// longest edge, and the factor by which it outscales the viewport (so
/// framing matches what is on screen).
fn render_size(viewport_width: f32, viewport_height: f32, long_edge: u32) -> (u32, u32, f32) {
    let (width, height) = (viewport_width.max(1.0), viewport_height.max(1.0));
    let factor = long_edge.clamp(16, MAX_RENDER_EDGE) as f32 / width.max(height);
    (
        ((width * factor).round() as u32).clamp(1, MAX_RENDER_EDGE),
        ((height * factor).round() as u32).clamp(1, MAX_RENDER_EDGE),
        factor,
    )
}

/// Largest still render edge the renderer accepts.
const MAX_RENDER_EDGE: u32 = 2048;

/// Longest edge the viewport shows a texture set at; larger sets preview a
/// downscaled copy, while exports keep their full size.
const PREVIEW_MAX_EDGE: u32 = 4096;

/// `set` as the viewport bakes it: downscaled past [`PREVIEW_MAX_EDGE`].
fn preview_copy(set: &TextureSet) -> std::borrow::Cow<'_, TextureSet> {
    set.downscaled(PREVIEW_MAX_EDGE)
        .map_or(std::borrow::Cow::Borrowed(set), std::borrow::Cow::Owned)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PaintMode {
    View,
    Paint,
    /// Removes paint from the selected paint layer.
    Erase,
    Mask,
    Id,
    Select,
}

impl PaintMode {
    fn paints(self) -> bool {
        matches!(self, Self::Paint | Self::Erase | Self::Mask | Self::Id)
    }
}

/// Which viewports share the central area.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum ViewMode {
    Model,
    Split,
    Uv,
}

/// A texture set's last full-material bake.
struct CachedMaterial {
    revision: u64,
    textures: Arc<MaterialTextures>,
    /// Set when the textures were patched from the previous cache rather
    /// than baked whole.
    patched: bool,
}

/// What a texture set's cached material is missing (PERF-02): the strokes
/// marked since it was baked, and the set's layer shape then. Kept for
/// every set, so a background rebake, or selecting the set, patches the
/// cache instead of baking the set whole.
struct SetDirty {
    dirty: incremental::PreviewDirty,
    /// Whether the cache was baked with fresh mesh maps; a patch must read
    /// the same maps as the texels around it.
    maps_fresh: bool,
}

/// Background bake of a texture set that is not selected, so every material
/// slot of a multi-material mesh shows its own textures.
struct SetBakeJob {
    set_id: TextureSetId,
    revision: u64,
    /// The cached material, and whether the job patched it.
    receiver: Receiver<(Result<MaterialTextures, String>, bool)>,
}

/// True when a preview for `key` is the selected set's plain material, so
/// it can stand in for the set's cached material (not tinted by a mask or
/// quick mask, nor soloed).
fn caches_material(key: &PreviewKey) -> bool {
    key.target == PreviewTarget::Material && key.solo_layer.is_none()
}

/// What the viewport currently shows on the mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreviewTarget {
    /// Full PBR material: Base Color, ORM, and Normal.
    Material,
    /// One channel, shown flat for inspection.
    Channel(MaterialChannel),
    MeshMap(MeshMapKind),
    /// A layer's final mask in grayscale (mask view).
    LayerMask(LayerId),
    /// A layer's generator alone, in grayscale.
    Generator(LayerId),
    /// The shaded material tinted by a layer's mask (shown, blocked,
    /// partial).
    MaskOverlay {
        layer: LayerId,
        /// Tint opacity in percent.
        opacity: u8,
        palette: substancecraft_core::OverlayPalette,
        pattern: bool,
    },
    /// The shaded material tinted by the quick mask being painted.
    QuickMask(substancecraft_core::StencilId),
}

struct BakedPreview {
    key: PreviewKey,
    textures: PreviewTextures,
    /// Revision of the texture set the textures were baked from.
    set_revision: u64,
    /// Set when the textures were patched from the previous preview.
    patch: Option<PreviewPatch>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreviewKey {
    texture_set: Option<TextureSetId>,
    target: PreviewTarget,
    /// Layer shown alone (solo), if any.
    solo_layer: Option<LayerId>,
    /// Bumped on every edit that changes baked output.
    revision: u64,
}

struct PreviewJob {
    /// The bake result, and whether it patched the shown preview.
    receiver: Receiver<(Result<PreviewTextures, String>, bool)>,
    key: PreviewKey,
    /// Revision of the texture set when the bake started.
    set_revision: u64,
    /// Content keys of the textures being patched and the blocks rebaked,
    /// when the job patches the shown preview.
    patch: Option<(Vec<usize>, Vec<incremental::Patch>)>,
}

/// How a shown preview differs from the one before it: each texture's
/// content key, the key of the texture it was patched from, and the texels
/// rebaked in it. The viewport re-uploads only those texels.
#[derive(Clone, Debug, PartialEq)]
struct PreviewPatch {
    textures: Vec<(usize, usize, Vec<TexelRect>)>,
}

impl PreviewPatch {
    /// `textures` patched from textures with content keys `base` by
    /// `patches`: a material texture takes the blocks of the channels it
    /// shows, a single channel's texture every block.
    fn new(textures: &PreviewTextures, base: Vec<usize>, patches: &[incremental::Patch]) -> Self {
        let regions = |index: usize| {
            patches
                .iter()
                .filter(|patch| match textures {
                    PreviewTextures::Material(_) => patch.channels.planes().flags()[index],
                    PreviewTextures::Single(_) => true,
                })
                .map(|patch| patch.rect)
                .collect()
        };
        Self {
            textures: textures
                .keys()
                .into_iter()
                .zip(base)
                .enumerate()
                .map(|(index, (key, base))| (key, base, regions(index)))
                .collect(),
        }
    }
}

/// Adds what `command` changes to `sets` and the texels of its strokes to
/// `strokes`, by set, in each set's preview (`previews` holds each set's
/// preview `width` × `height` and the `factor` it is downscaled by). False
/// when it changes more than texture sets' layers or a stroke's texels
/// cannot be told (one painted through a camera): then every set needs a
/// full rebake.
fn layer_edit_scope(
    command: &EditCommand,
    previews: &BTreeMap<TextureSetId, (u32, u32, u32)>,
    sets: &mut BTreeSet<TextureSetId>,
    strokes: &mut BTreeMap<TextureSetId, Vec<TexelRect>>,
) -> bool {
    let mut stroke = |set: TextureSetId,
                      points: &[glam::Vec2],
                      radius: f32,
                      tip: substancecraft_core::BrushTip,
                      frames: &[substancecraft_core::StampFrame]| {
        sets.insert(set);
        if let Some(&(width, height, factor)) = previews.get(&set) {
            strokes.entry(set).or_default().extend(stroke_texel_bounds(
                points,
                radius / factor as f32,
                tip,
                frames,
                width,
                height,
            ));
        }
    };
    match command {
        EditCommand::Batch(commands) => commands
            .iter()
            .all(|command| layer_edit_scope(command, previews, sets, strokes)),
        EditCommand::Paint {
            texture_set_id,
            stroke: painted,
            ..
        } => {
            painted.camera.is_none() && {
                stroke(
                    *texture_set_id,
                    &painted.points,
                    painted.radius_pixels,
                    painted.tip,
                    &painted.frames,
                );
                true
            }
        }
        EditCommand::Mask {
            texture_set_id,
            stroke: painted,
            ..
        } => {
            painted.camera.is_none() && {
                stroke(
                    *texture_set_id,
                    &painted.points,
                    painted.radius_pixels,
                    painted.tip,
                    &painted.frames,
                );
                true
            }
        }
        EditCommand::Id {
            texture_set_id,
            stroke: painted,
            ..
        } => {
            stroke(
                *texture_set_id,
                &painted.points,
                painted.radius_pixels,
                painted.tip,
                &painted.frames,
            );
            true
        }
        // The preview's snapshot diff finds what these changed.
        EditCommand::LayerSettings { texture_set_id, .. }
        | EditCommand::InsertLayer { texture_set_id, .. }
        | EditCommand::RemoveLayer { texture_set_id, .. }
        | EditCommand::MoveLayer { texture_set_id, .. } => {
            sets.insert(*texture_set_id);
            true
        }
        _ => false,
    }
}

/// Texture sets whose layers `command` edits.
fn edited_texture_sets(command: &EditCommand) -> BTreeSet<TextureSetId> {
    match command {
        EditCommand::Batch(commands) => commands.iter().flat_map(edited_texture_sets).collect(),
        EditCommand::LayerSettings { texture_set_id, .. } => BTreeSet::from([*texture_set_id]),
        _ => BTreeSet::new(),
    }
}

/// Pixels one arrow-key press orbits or pans the camera by.
const KEYBOARD_CAMERA_STEP: f32 = 12.0;
/// Scroll distance one + or - press zooms by.
const KEYBOARD_ZOOM_STEP: f32 = 40.0;

/// Moves the camera from arrow, plus, and minus key presses (repeats
/// included): arrows orbit, or pan with Shift. Returns true when it moved.
fn keyboard_camera(camera: &mut ViewportCamera, input: &egui::InputState) -> bool {
    let mut delta = glam::Vec2::ZERO;
    for (key, step) in [
        (egui::Key::ArrowLeft, glam::Vec2::new(-1.0, 0.0)),
        (egui::Key::ArrowRight, glam::Vec2::new(1.0, 0.0)),
        (egui::Key::ArrowUp, glam::Vec2::new(0.0, -1.0)),
        (egui::Key::ArrowDown, glam::Vec2::new(0.0, 1.0)),
    ] {
        if input.key_pressed(key) {
            delta += step * KEYBOARD_CAMERA_STEP;
        }
    }
    let mut zoom = 0.0;
    if input.key_pressed(egui::Key::Plus) || input.key_pressed(egui::Key::Equals) {
        zoom += KEYBOARD_ZOOM_STEP;
    }
    if input.key_pressed(egui::Key::Minus) {
        zoom -= KEYBOARD_ZOOM_STEP;
    }
    if delta != glam::Vec2::ZERO {
        if input.modifiers.shift {
            camera.pan_by(delta);
        } else {
            camera.orbit(delta);
        }
    }
    if zoom != 0.0 {
        camera.zoom_by(zoom);
    }
    delta != glam::Vec2::ZERO || zoom != 0.0
}

/// Preset the Export button starts with: the viewport's own metal/rough layout.
const DEFAULT_EXPORT_PRESET: &str = "Metal/Rough (ORM)";

/// An export that would replace files, held until the user confirms.
struct OverwritePrompt {
    texture_set: TextureSet,
    preset: ExportPreset,
    folder: std::path::PathBuf,
    existing: Vec<std::path::PathBuf>,
}

/// The first preflight error (and how many more), if any blocks the export.
fn blocking_export_error(issues: &[PreflightIssue]) -> Option<String> {
    let errors: Vec<&PreflightIssue> = issues
        .iter()
        .filter(|issue| issue.severity == PreflightSeverity::Error)
        .collect();
    let first = errors.first()?;
    let more = match errors.len() {
        1 => String::new(),
        count => format!(" (and {} more)", count - 1),
    };
    Some(format!("Export blocked: {}{more}", first.message))
}

/// A custom export preset being edited.
#[derive(Clone, Debug, PartialEq)]
struct PresetDraft {
    /// Name of the project preset the draft replaces; `None` for a new one.
    original: Option<String>,
    preset: ExportPreset,
}

/// Shared inputs of a background texture-set bake.
struct BakeInputs {
    mesh: Option<Arc<MeshGeometry>>,
    mesh_maps: Option<Arc<Vec<MeshMaps>>>,
    /// Mesh material slot whose maps the set uses.
    slot: usize,
    resources: ResourceImages,
    /// Set for meshes with UDIM tiles beyond 1001: previews bake every
    /// tile into an atlas.
    udim: Option<udim_view::UdimInputs>,
}

impl BakeInputs {
    fn maps(&self) -> Option<&MeshMaps> {
        self.mesh_maps
            .as_deref()
            .and_then(|maps| maps.get(self.slot))
    }

    fn bake_channel(
        &self,
        texture_set: &TextureSet,
        channel: MaterialChannel,
    ) -> Result<BakedTexture, String> {
        if let Some(udim) = &self.udim {
            let kind = format!("{channel:?}");
            return udim.bake_single(self, texture_set, Some(&kind), |set, context| {
                set.bake_channel(channel, context)
                    .map_err(|error| error.to_string())
            });
        }
        bake_texture_set_channel(
            texture_set,
            channel,
            self.mesh.as_deref(),
            self.maps(),
            Some(&self.resources),
        )
        .map_err(|error| error.to_string())
    }

    fn bake_material(&self, texture_set: &TextureSet) -> Result<MaterialTextures, String> {
        if let Some(udim) = &self.udim {
            return udim.bake_material(self, texture_set);
        }
        bake_texture_set_material(
            texture_set,
            self.mesh.as_deref(),
            self.maps(),
            Some(&self.resources),
        )
        .map_err(|error| error.to_string())
    }

    /// A copy of `base`, the material of an earlier state of `texture_set`,
    /// rebaked over `patches` only, in the textures each one reaches.
    fn patch_material(
        &self,
        texture_set: &TextureSet,
        base: &MaterialTextures,
        patches: &[incremental::Patch],
    ) -> Result<MaterialTextures, String> {
        let mut textures = base.clone();
        let regions = patches
            .iter()
            .map(|patch| (patch.rect, patch.channels.planes()))
            .collect::<Vec<_>>();
        rebake_texture_set_material_plane_regions(
            texture_set,
            self.mesh.as_deref(),
            self.maps(),
            Some(&self.resources),
            &mut textures,
            &regions,
        )
        .map_err(|error| error.to_string())?;
        Ok(textures)
    }

    /// A copy of `base`, a preview of `target` for an earlier state of
    /// `texture_set`, rebaked over `regions` only.
    fn patch_preview(
        &self,
        texture_set: &TextureSet,
        target: PreviewTarget,
        base: &PreviewTextures,
        patches: &[incremental::Patch],
    ) -> Result<PreviewTextures, String> {
        let (mesh, maps, resources) = (self.mesh.as_deref(), self.maps(), Some(&self.resources));
        match (base, target) {
            (PreviewTextures::Material(base), PreviewTarget::Material) => self
                .patch_material(texture_set, base, patches)
                .map(|textures| PreviewTextures::Material(Arc::new(textures))),
            (PreviewTextures::Single(base), PreviewTarget::Channel(channel)) => {
                let mut texture = BakedTexture::clone(base);
                let regions = patches
                    .iter()
                    .filter(|patch| patch.channels.reach(channel))
                    .map(|patch| patch.rect)
                    .collect::<Vec<_>>();
                rebake_texture_set_channel_regions(
                    texture_set,
                    channel,
                    mesh,
                    maps,
                    resources,
                    &mut texture,
                    &regions,
                )
                .map_err(|error| error.to_string())?;
                Ok(PreviewTextures::Single(Arc::new(texture)))
            }
            _ => Err("preview kind does not match its target".to_owned()),
        }
    }

    fn bake_export(
        &self,
        texture_set: &TextureSet,
        preset: &ExportPreset,
    ) -> Result<Vec<ExportedTexture>, String> {
        bake_texture_set_export(
            texture_set,
            preset,
            self.mesh.as_deref(),
            self.maps(),
            Some(&self.resources),
        )
        .map_err(|error| error.to_string())
    }
}

#[derive(Clone)]
enum PreviewTextures {
    Material(Arc<MaterialTextures>),
    Single(Arc<BakedTexture>),
}

/// What one mesh material slot shows in the 3D viewport.
enum SlotSource {
    Material(Arc<MaterialTextures>),
    Single(Arc<BakedTexture>),
    Neutral,
}

impl PreviewTextures {
    /// The texture shown as base colour (and by the software fallback).
    fn base(&self) -> &BakedTexture {
        match self {
            Self::Material(textures) => &textures.base_color,
            Self::Single(texture) => texture,
        }
    }

    /// Viewport content keys of every texture, in upload order.
    fn keys(&self) -> Vec<usize> {
        match self {
            Self::Material(textures) => [
                &textures.base_color,
                &textures.orm,
                &textures.normal,
                &textures.emissive,
            ]
            .into_iter()
            .map(texture_key)
            .collect(),
            Self::Single(texture) => vec![texture_key(texture)],
        }
    }
}

/// Identifies a texture's content for the viewport: its buffer, which a
/// bake never reuses for different content while the texture is alive.
fn texture_key(texture: &BakedTexture) -> usize {
    (texture.rgba8.as_ptr() as usize)
        ^ (texture.width as usize).rotate_left(7)
        ^ (texture.height as usize).rotate_left(17)
}

/// Identifies every input that can make a mesh-map bake stale.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BakeKey {
    mesh_revision: u64,
    high_poly_revision: u64,
    import_scale: f32,
    smoothing: MeshSmoothing,
    settings: MeshMapSettings,
    skew_map: Option<ResourceId>,
    skew_map_revision: u64,
    /// Hash of the high-to-low options and the loaded cage.
    high_to_low: u64,
}

/// Optional high-resolution source for high-to-low mesh-map baking.
struct HighPolySource {
    path: std::path::PathBuf,
    geometry: MeshGeometry,
    /// The file's meshes by name, scaled, for name-matched bakes.
    parts: Vec<(String, MeshGeometry)>,
}

struct MeshMapJob {
    /// The maps, and where their occlusion rays were traced.
    receiver: Receiver<(Result<Vec<MeshMaps>, MeshMapError>, String)>,
    cancel: Arc<AtomicBool>,
    key: BakeKey,
    started: std::time::Instant,
    started_at: u64,
    settings: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ImportAssignment {
    resource: ResourceId,
    channel: Option<MaterialChannel>,
    reason: String,
}

struct PoseTransition {
    from: NamedPose,
    to: NamedPose,
    started_at: f64,
}

struct RestoredMeshScene {
    selected_primitive: Option<PrimitivePreset>,
    imported_meshes: Vec<ImportedMesh>,
    selected_imported_mesh: Option<usize>,
    geometry: Option<MeshGeometry>,
    scale: f32,
    smoothing: MeshSmoothing,
}

impl Default for SubstanceCraftApp {
    fn default() -> Self {
        let (filament_frames_tx, filament_frames_rx) = sync_channel(1);
        let mut app = Self {
            project: MaterialProject {
                mesh: Some(MeshAsset {
                    id: MeshId(1),
                    source: "builtin:Cube".to_owned(),
                    scale: 1.0,
                    smoothing: MeshSmoothing::default(),
                    high_poly_source: None,
                    content_hash: None,
                    high_poly_hash: None,
                    high_to_low: Default::default(),
                }),
                ..MaterialProject::empty()
            },
            selected_texture_set: None,
            selected_layer: None,
            extra_selected_layers: BTreeSet::new(),
            decal_gizmo: None,
            picker_clipped: None,
            thumbnail_stage: None,
            smart_thumbnails: BTreeMap::new(),
            smart_thumbnail_textures: BTreeMap::new(),
            selected_primitive: Some(PrimitivePreset::Cube),
            base_geometry: None,
            mesh_geometry: None,
            mesh_topology: None,
            udim_atlas: None,
            udim_cache: Arc::default(),
            mesh_center: glam::Vec3::ZERO,
            mesh_smoothing: MeshSmoothing::default(),
            mesh_error: None,
            mesh_revision: 1,
            filament_mesh_revision: None,
            filament_hidden_revision: None,
            hidden_geometry: geometry_visibility::HiddenGeometry::default(),
            filament_environment: None,
            custom_environment: None,
            display_lut: None,
            ocio: None,
            grading_status: None,
            user_shaders: Vec::new(),
            shader_status: None,
            environment_revision: 0,
            environment_status: None,
            filament_renderer: None,
            filament_error: None,
            filament_frame_texture: None,
            filament_frames_tx,
            filament_frames_rx,
            filament_render_pending: Arc::new(AtomicBool::new(false)),
            render_request: None,
            render_result: Arc::new(std::sync::Mutex::new(None)),
            render_status: None,
            turntable: None,
            turntable_mode: turntable::TurntableMode::default(),
            turntable_frames: turntable::TURNTABLE_FRAME_COUNTS[1],
            loaded_mesh: None,
            imported_meshes: Vec::new(),
            selected_imported_mesh: None,
            animation: animation::AnimationState::default(),
            posed_mesh: None,
            filament_pose: None,
            import_error: None,
            import_scale: 1.0,
            project_path: None,
            project_error: None,
            active_channel: MaterialChannel::BaseColor,
            next_texture_set_id: 1,
            next_layer_id: 1,
            current_pose: NamedPose {
                id: PoseId(0),
                name: "Current Pose".to_owned(),
                animation: None,
                bones: BTreeMap::from([(BoneId(0), BoneTransform::IDENTITY)]),
            },
            viewport_camera: ViewportCamera::default(),
            viewport_lighting: ViewportLighting::default(),
            paint_mode: PaintMode::View,
            brush_radius: 16.0,
            brush_color: egui::Color32::from_rgb(214, 108, 66),
            brush_roughness: 0.5,
            brush_metallic: 0.0,
            brush_height: DEFAULT_HEIGHT,
            brush_emissive: egui::Color32::from_rgb(
                DEFAULT_EMISSIVE[0],
                DEFAULT_EMISSIVE[1],
                DEFAULT_EMISSIVE[2],
            ),
            brush_material_opacity: DEFAULT_OPACITY,
            brush_tip: BrushTip::ROUND,
            pen_pressure: None,
            resource_images: ResourceImages::new(),
            resource_thumbnails: BTreeMap::new(),
            resource_watcher: resource_watch::ResourceWatcher::default(),
            export_format: TextureFormat::Png,
            export_preset: DEFAULT_EXPORT_PRESET.to_owned(),
            preset_draft: None,
            overwrite_prompt: None,
            view_mode: ViewMode::Model,
            uv_view: uv_view::UvView::default(),
            set_revisions: BTreeMap::new(),
            set_materials: BTreeMap::new(),
            set_bake_job: None,
            set_dirty: BTreeMap::new(),
            set_bake_failed: BTreeMap::new(),
            channel_preview: false,
            mask_value: 0.0,
            selection_mode: SelectionMode::Element,
            hovered_selection: Vec::new(),
            active_stroke: None,
            stroke_first_index: 0,
            stroke_paths: Vec::new(),
            stroke_last_triangles: Vec::new(),
            symmetry: None,
            stroke_stabilizer: substancecraft_core::LazyStabilizer::new(0.0),
            stroke_tools: stroke_tools::StrokeTools::default(),
            brush_user: substancecraft_core::UserValues::default(),
            last_paint_screen: None,
            path_controls: None,
            picking_material: false,
            skew_map: None,
            skew_map_revision: 0,
            asset_query: String::new(),
            layer_query: String::new(),
            pending_import_assignments: Vec::new(),
            history: EditHistory::default(),
            layer_settled: None,
            editing_material: None,
            pending_template: None,
            projection_painting: false,
            picking_id_color: false,
            evolution_start: None,
            evolution_name: "Wear".to_owned(),
            baked_preview: None,
            preview_revision: 0,
            preview_job: None,
            preview_dirty: None,
            preview_failed: false,
            bake_error: None,
            mesh_map_settings: MeshMapSettings::default(),
            mesh_maps: None,
            mesh_maps_key: None,
            high_poly: None,
            cage_mesh: None,
            high_poly_revision: 0,
            mesh_map_job: None,
            mesh_map_error: None,
            mesh_map_preview: None,
            selected_pose: None,
            pose_name: "Pose 1".to_owned(),
            pose_transition: None,
            drawer: Some(panels::Drawer::Scene),
            last_drawer: panels::Drawer::Scene,
            show_inspector: true,
            layout_name: String::new(),
            layout: panels::OverlayLayout::default(),
            autosave: autosave::Autosave::default(),
            preferences: preferences::PreferenceState::default(),
            inspect: inspector::InspectState::default(),
            edge_overlay: edges::EdgeOverlay::default(),
            loaded_resource_sources: Vec::new(),
            bake_log: bake_log::BakeLog::default(),
            cache: cache::CacheState::default(),
            library: library_panel::LibraryState::default(),
        };
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.loaded_mesh = app.project.mesh.clone();
        app
    }
}

impl SubstanceCraftApp {
    fn mark_mesh_changed(&mut self) {
        self.mesh_revision = self.mesh_revision.wrapping_add(1);
        self.filament_frame_texture = None;
        self.cancel_mesh_map_job();
        self.mesh_maps = None;
        self.mesh_map_error = None;
        self.hovered_selection.clear();
        self.invalidate_all_sets();
        // Old textures do not fit the new mesh's UVs.
        self.baked_preview = None;
        self.set_materials.clear();
        self.set_bake_job = None;
    }

    /// Marks the shown preview out of date. It stays visible until the
    /// background rebake replaces it. Edits made through the UI only touch
    /// the selected texture set.
    fn invalidate_preview(&mut self) {
        self.invalidate_view();
        if let Some(set_id) = self.selected_texture_set {
            self.set_revisions.insert(set_id, self.preview_revision);
            self.set_dirty.remove(&set_id);
        }
    }

    /// Asks for a new preview without changing any set: the view changed
    /// (another set selected), not what sets bake to.
    fn invalidate_view(&mut self) {
        self.preview_revision = self.preview_revision.wrapping_add(1);
        self.preview_failed = false;
        self.preview_dirty = None;
    }

    /// Like [`Self::invalidate_preview`] for a stroke edit that changed only
    /// the texels in `bounds` (none at all when empty), so the next preview
    /// can rebake just those.
    fn invalidate_preview_texels(&mut self, bounds: impl IntoIterator<Item = TexelRect>) {
        let dirty = self.preview_dirty.take();
        let cached = self
            .selected_texture_set
            .and_then(|set_id| self.set_dirty.remove(&set_id));
        self.invalidate_preview();
        let bounds: Vec<TexelRect> = bounds.into_iter().collect();
        self.preview_dirty = dirty
            .filter(|dirty| Some(dirty.set_id()) == self.selected_texture_set)
            .map(|mut dirty| {
                for bounds in &bounds {
                    dirty.mark(*bounds);
                }
                dirty
            });
        if let Some(mut cached) = cached {
            for bounds in &bounds {
                cached.dirty.mark(*bounds);
            }
            self.set_dirty.insert(cached.dirty.set_id(), cached);
        }
    }

    /// Marks only `set_ids` out of date. When the selected set is not among
    /// them its current preview stays valid and is carried forward to the
    /// new revision instead of being rebaked.
    fn invalidate_sets(&mut self, set_ids: &BTreeSet<TextureSetId>) {
        self.invalidate_set_texels(set_ids, None);
    }

    /// Like [`Self::invalidate_preview_texels`] for an edit of the selected
    /// set's layers or their settings, which the preview finds by diffing
    /// the set against its snapshot.
    fn invalidate_preview_layers(&mut self) {
        self.invalidate_preview_texels(None);
    }

    /// [`Self::invalidate_sets`]; with `strokes`, the edit changed only
    /// layers, and of each set's strokes only those texels (in its preview),
    /// so the selected preview and other sets' cached materials can be
    /// patched (see [`Self::invalidate_preview_layers`]).
    fn invalidate_set_texels(
        &mut self,
        set_ids: &BTreeSet<TextureSetId>,
        mut strokes: Option<BTreeMap<TextureSetId, Vec<TexelRect>>>,
    ) {
        if set_ids.is_empty() {
            return;
        }
        for set_id in set_ids
            .iter()
            .filter(|id| Some(**id) != self.selected_texture_set)
        {
            match &strokes {
                Some(strokes) => {
                    if let Some(cached) = self.set_dirty.get_mut(set_id) {
                        for bounds in strokes.get(set_id).into_iter().flatten() {
                            cached.dirty.mark(*bounds);
                        }
                    }
                }
                None => {
                    self.set_dirty.remove(set_id);
                }
            }
        }
        let selected_affected = self
            .selected_texture_set
            .is_some_and(|id| set_ids.contains(&id));
        let previous = self.preview_key();
        if selected_affected {
            let selected = self
                .selected_texture_set
                .and_then(|id| strokes.as_mut().map(|strokes| strokes.remove(&id)));
            match selected {
                Some(strokes) => self.invalidate_preview_texels(strokes.unwrap_or_default()),
                None => self.invalidate_preview(),
            }
        } else {
            self.preview_revision = self.preview_revision.wrapping_add(1);
            let current = self.preview_key();
            if let Some(preview) = self
                .baked_preview
                .as_mut()
                .filter(|preview| preview.key == previous)
            {
                preview.key = current;
            }
            if let Some(job) = self.preview_job.as_mut().filter(|job| job.key == previous) {
                job.key = current;
            }
        }
        let revision = self.preview_revision;
        for id in set_ids {
            self.set_revisions.insert(*id, revision);
        }
    }

    /// Marks every texture set out of date: the mesh, its maps, history, or
    /// shared resources changed.
    fn invalidate_all_sets(&mut self) {
        self.invalidate_preview();
        self.set_dirty.clear();
        let revision = self.preview_revision;
        for set in &self.project.texture_sets {
            self.set_revisions.insert(set.id, revision);
        }
    }

    fn set_revision(&self, set_id: TextureSetId) -> u64 {
        self.set_revisions.get(&set_id).copied().unwrap_or(0)
    }

    fn set_base_geometry(&mut self, geometry: Option<MeshGeometry>) {
        self.base_geometry = geometry;
        // A new mesh starts in its bind pose.
        self.animation = animation::AnimationState::default();
        self.posed_mesh = None;
        self.rebuild_mesh();
    }

    /// Re-derives display geometry from the base mesh and smoothing settings.
    fn rebuild_mesh(&mut self) {
        self.mesh_error = None;
        self.mesh_geometry = self.base_geometry.as_ref().map(|base| {
            Arc::new(self.mesh_smoothing.apply(base).unwrap_or_else(|error| {
                self.mesh_error = Some(format!("Smoothing skipped: {error}"));
                base.clone()
            }))
        });
        self.mesh_topology = self.mesh_geometry.as_deref().map(MeshTopology::build);
        self.udim_atlas = self
            .mesh_geometry
            .as_deref()
            .and_then(|mesh| substancecraft_core::UdimAtlas::new(&mesh.udim_tiles()));
        self.mesh_center = self
            .mesh_geometry
            .as_deref()
            .map_or(glam::Vec3::ZERO, mesh_bounds_center);
        self.mark_mesh_changed();
    }

    fn set_mesh_smoothing(&mut self, smoothing: MeshSmoothing) {
        if smoothing == self.mesh_smoothing || !smoothing.is_valid() {
            return;
        }
        let before = self.project.mesh.clone();
        let from = self.mesh_smoothing.subdivision_levels;
        let to = smoothing.subdivision_levels;
        let mut remapped = Vec::new();
        if from != to {
            // Polygon masks index subdivided triangles; carry them across,
            // recording each change so undo restores the old indices.
            for set in &mut self.project.texture_sets {
                let set_id = set.id;
                for_each_layer_mut(&mut set.layers, &mut |layer| {
                    if layer.polygon_mask.is_none() {
                        return;
                    }
                    let before = layer.settings();
                    if let Some(mask) = layer.polygon_mask.as_mut() {
                        mask.triangles =
                            remap_subdivided_triangles(mask.triangles.iter().copied(), from, to)
                                .into_iter()
                                .collect();
                    }
                    remapped.push(EditCommand::LayerSettings {
                        texture_set_id: set_id,
                        layer_id: layer.id,
                        before: Box::new(before),
                        after: Box::new(layer.settings()),
                    });
                });
            }
            self.layer_settled = None;
        }
        self.mesh_smoothing = smoothing;
        if let Some(mesh) = self.project.mesh.as_mut() {
            mesh.smoothing = smoothing;
        }
        self.record_mesh_edit(before, remapped);
        self.rebuild_mesh();
    }

    /// Records the change of the project's mesh reference from `before`
    /// (plus any layer edits it caused) as one undo step. Successive edits
    /// in one drag join that step.
    fn record_mesh_edit(&mut self, before: Option<MeshAsset>, mut extra: Vec<EditCommand>) {
        let after = self.project.mesh.clone();
        self.loaded_mesh = after.clone();
        if before == after && extra.is_empty() {
            return;
        }
        let mesh = EditCommand::Mesh { before, after };
        if extra.is_empty() {
            self.history.record_merging(mesh);
        } else {
            extra.insert(0, mesh);
            self.history.record(EditCommand::Batch(extra));
        }
    }

    /// Reloads the mesh the project names (after undo or redo changed it),
    /// keeping texture sets and layers.
    fn reload_project_mesh(&mut self) {
        match Self::restore_project_mesh(&self.project) {
            Ok(scene) => {
                self.selected_primitive = scene.selected_primitive;
                self.imported_meshes = scene.imported_meshes;
                self.selected_imported_mesh = scene.selected_imported_mesh;
                self.import_scale = scene.scale;
                self.mesh_smoothing = scene.smoothing;
                self.set_base_geometry(scene.geometry);
                self.restore_high_poly();
            }
            Err(error) => self.project_error = Some(error),
        }
        self.loaded_mesh = self.project.mesh.clone();
    }

    /// Replaces the mesh with the file at `path` while keeping every texture
    /// set and layer (painting is in UV space), as one undoable edit.
    /// Polygon masks index triangles, so they only survive when the new mesh
    /// keeps the old triangle order.
    fn replace_mesh_with(&mut self, path: &std::path::Path) {
        let meshes = match load_mesh_file(path) {
            Ok(meshes) => meshes,
            Err(error) => {
                self.import_error = Some(error);
                return;
            }
        };
        let Some(first_mesh) = meshes.first() else {
            self.import_error = Some("the file contains no mesh geometry".to_owned());
            return;
        };
        let before = self.project.mesh.clone();
        let geometry = scaled_geometry(&first_mesh.geometry, self.import_scale);
        self.project.mesh = Some(MeshAsset {
            id: MeshId(1),
            source: path.to_string_lossy().into_owned(),
            scale: self.import_scale,
            smoothing: self.mesh_smoothing,
            high_poly_source: None,
            content_hash: substancecraft_api::file_content_hash(path).ok(),
            high_poly_hash: None,
            high_to_low: Default::default(),
        });
        self.imported_meshes = meshes;
        self.selected_imported_mesh = Some(0);
        self.selected_primitive = None;
        self.set_base_geometry(Some(geometry));
        self.high_poly = None;
        self.high_poly_changed();
        self.import_error = None;
        self.history.end_merge();
        self.record_mesh_edit(before, Vec::new());
        self.history.end_merge();
    }

    fn replace_mesh(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D mesh (FBX, glTF, STEP)", &MESH_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        self.replace_mesh_with(&path);
    }

    fn update_filament_frame(&mut self, context: &egui::Context, viewport: egui::Rect) {
        if let Some(renderer) = self.filament_renderer.as_mut() {
            renderer.poll();
        }
        self.save_finished_render();
        self.advance_turntable();
        while let Ok(frame) = self.filament_frames_rx.try_recv() {
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [frame.width as usize, frame.height as usize],
                &frame.rgba8,
            );
            if let Some(texture) = self.filament_frame_texture.as_mut() {
                texture.set(image, egui::TextureOptions::LINEAR);
            } else {
                self.filament_frame_texture = Some(context.load_texture(
                    "filament-viewport",
                    image,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }

        let wanted = self.preview_key();
        let target = wanted.target;
        if self.mesh_geometry.is_none() {
            self.filament_frame_texture = None;
            return;
        }
        let plan = self.slot_upload_plan(target);
        // The posed mesh, when the viewport shows a pose; bound otherwise.
        let pose = self.pose_key();
        let shown = self.display_mesh();
        // Rig lights turn with the environment, including a turntable's turn.
        let rig = self.rig_suns(
            self.render_request
                .as_ref()
                .filter(|request| !request.issued)
                .and_then(|request| request.lighting_rotation)
                .unwrap_or(self.viewport_lighting.rotation),
        );
        let shader = self.viewport_shader();
        let (transform, exposure, lut) = self.grading();
        let Some(mesh) = self.mesh_geometry.as_deref() else {
            return;
        };
        let Some(renderer) = self.filament_renderer.as_mut() else {
            return;
        };
        // UDIM meshes sample one atlas of every tile.
        fn in_atlas<'a>(
            atlas: Option<&substancecraft_core::UdimAtlas>,
            mesh: &'a MeshGeometry,
        ) -> std::borrow::Cow<'a, MeshGeometry> {
            atlas.map_or(std::borrow::Cow::Borrowed(mesh), |atlas| {
                std::borrow::Cow::Owned(atlas.mesh(mesh))
            })
        }
        let atlas = self.udim_atlas.as_ref();
        let atlas_mesh = |mesh| in_atlas(atlas, mesh);
        let hidden_revision = (self.hidden_geometry.mesh_revision == self.mesh_revision)
            .then_some(self.hidden_geometry.revision);
        if self.filament_mesh_revision != Some(self.mesh_revision)
            || self.filament_hidden_revision != hidden_revision
        {
            let hidden = match hidden_revision {
                Some(_) => &self.hidden_geometry.triangles,
                None => &BTreeSet::new(),
            };
            let visible = geometry_visibility::visible_mesh(mesh, hidden);
            match renderer.set_mesh(
                &in_atlas(atlas, &visible),
                self.mesh_smoothing.angle_degrees,
            ) {
                Ok(()) => {
                    self.filament_mesh_revision = Some(self.mesh_revision);
                    self.filament_hidden_revision = hidden_revision;
                    self.filament_pose = Some(None);
                }
                Err(error) => {
                    self.filament_error = Some(error.to_string());
                    return;
                }
            }
        }
        if self.filament_pose.as_ref() != Some(&pose)
            && let Some(shown) = &shown
        {
            match renderer.update_vertices(&atlas_mesh(shown), self.mesh_smoothing.angle_degrees) {
                Ok(()) => self.filament_pose = Some(pose),
                Err(error) => {
                    self.filament_error = Some(error.to_string());
                    return;
                }
            }
        }
        // A still render waiting to be drawn may turn the light (turntables).
        let mut lighting = self.viewport_lighting;
        if let Some(rotation) = self
            .render_request
            .as_ref()
            .filter(|request| !request.issued)
            .and_then(|request| request.lighting_rotation)
        {
            lighting.rotation = rotation;
        }
        if let Err(error) = apply_environment(
            renderer,
            lighting,
            self.custom_environment.as_ref(),
            &mut self.filament_environment,
        ) {
            self.filament_error = Some(error.to_string());
            return;
        }
        let preferences = &self.preferences.values;
        renderer.set_background(
            preferences
                .environment_background
                .then_some(preferences.background_blur),
        );
        renderer.set_global_illumination(preferences.global_illumination);
        if let Err(error) = renderer.set_color_grading(transform, exposure, lut.as_deref()) {
            self.filament_error = Some(error.to_string());
            return;
        }
        renderer.set_rig_lights(&rig);
        renderer.set_shader(&shader);

        let patch = self
            .baked_preview
            .as_ref()
            .and_then(|preview| preview.patch.as_ref());
        let upload = |renderer: &mut FilamentRuntime,
                      slot: usize,
                      map: MaterialMap,
                      texture: &BakedTexture| {
            let key = texture_key(texture);
            // A patched preview re-uploads only the texels it rebaked.
            match patch.and_then(|patch| {
                let (_, base, regions) = patch
                    .textures
                    .iter()
                    .find(|(patched, _, _)| *patched == key)?;
                Some((*base, regions.as_slice()))
            }) {
                Some((base_key, regions)) => renderer.patch_map(
                    slot,
                    map,
                    key,
                    base_key,
                    regions,
                    texture.width,
                    texture.height,
                    &texture.rgba8,
                ),
                None => renderer.set_map(
                    slot,
                    map,
                    key,
                    texture.width,
                    texture.height,
                    &texture.rgba8,
                ),
            }
        };
        // Neutral fallbacks: grey dielectric, AO 1 / roughness 0.58, flat
        // normal, no emission.
        let neutral = |renderer: &mut FilamentRuntime, slot: usize| {
            renderer
                .set_map(slot, MaterialMap::BaseColor, 0, 1, 1, &[168, 173, 179, 255])
                .and_then(|()| {
                    renderer.set_map(slot, MaterialMap::Orm, 0, 1, 1, &[255, 148, 0, 255])
                })
                .and_then(|()| {
                    renderer.set_map(slot, MaterialMap::Normal, 0, 1, 1, &[128, 128, 255, 255])
                })
                .and_then(|()| {
                    renderer.set_map(slot, MaterialMap::Emissive, 0, 1, 1, &[0, 0, 0, 255])
                })
        };
        renderer.set_preview_mode(!matches!(
            target,
            PreviewTarget::Material
                | PreviewTarget::MaskOverlay { .. }
                | PreviewTarget::QuickMask(_)
        ));
        let mut texture_result = Ok(());
        for (slot, source) in plan.into_iter().enumerate() {
            let result = match source {
                SlotSource::Material(textures) => {
                    upload(renderer, slot, MaterialMap::BaseColor, &textures.base_color)
                        .and_then(|()| upload(renderer, slot, MaterialMap::Orm, &textures.orm))
                        .and_then(|()| {
                            upload(renderer, slot, MaterialMap::Normal, &textures.normal)
                        })
                        .and_then(|()| {
                            upload(renderer, slot, MaterialMap::Emissive, &textures.emissive)
                        })
                }
                SlotSource::Single(texture) => {
                    upload(renderer, slot, MaterialMap::BaseColor, &texture)
                }
                SlotSource::Neutral => neutral(renderer, slot),
            };
            if result.is_err() {
                texture_result = result;
                break;
            }
        }
        if let Err(error) = texture_result {
            self.filament_error = Some(error.to_string());
            return;
        }

        if self
            .filament_render_pending
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            context.request_repaint_after(Duration::from_millis(16));
            return;
        }
        let scale = context.pixels_per_point();
        if let Some(request) = self
            .render_request
            .as_mut()
            .filter(|request| !request.issued)
        {
            // A still render takes this frame: same camera and lighting, at
            // the requested size.
            let (width, height, factor) = render_size(
                viewport.width() * scale,
                viewport.height() * scale,
                request.long_edge,
            );
            let result = Arc::clone(&self.render_result);
            let pending = Arc::clone(&self.filament_render_pending);
            let outcome = renderer.render(
                request.camera.unwrap_or(self.viewport_camera),
                width,
                height,
                scale * factor,
                move |rgba8| {
                    pending.store(false, Ordering::Release);
                    if let Ok(mut slot) = result.lock() {
                        *slot = Some(FilamentFrame {
                            width,
                            height,
                            rgba8,
                        });
                    }
                },
            );
            match outcome {
                Ok(true) => request.issued = true,
                Ok(false) => self.filament_render_pending.store(false, Ordering::Release),
                Err(error) => {
                    self.filament_render_pending.store(false, Ordering::Release);
                    self.render_status = Some(format!("Render failed: {error}"));
                    self.render_request = None;
                    self.turntable = None;
                }
            }
            context.request_repaint_after(Duration::from_millis(16));
            return;
        }
        let width = (viewport.width() * scale).ceil().clamp(1.0, 2048.0) as u32;
        let height = (viewport.height() * scale).ceil().clamp(1.0, 2048.0) as u32;
        let sender = self.filament_frames_tx.clone();
        let pending = Arc::clone(&self.filament_render_pending);
        match renderer.render(self.viewport_camera, width, height, scale, move |rgba8| {
            pending.store(false, Ordering::Release);
            let _ = sender.try_send(FilamentFrame {
                width,
                height,
                rgba8,
            });
        }) {
            Ok(true) => {
                self.filament_error = None;
                context.request_repaint_after(Duration::from_millis(16));
            }
            Ok(false) => {
                self.filament_render_pending.store(false, Ordering::Release);
                context.request_repaint_after(Duration::from_millis(16));
            }
            Err(error) => {
                self.filament_render_pending.store(false, Ordering::Release);
                self.filament_error = Some(error.to_string());
            }
        }
    }

    /// Asks where to save a still render of the viewport, then queues it.
    /// `camera` renders from a saved view instead of the viewport.
    fn request_render(&mut self, long_edge: u32, camera: Option<ViewportCamera>) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG image", &["png"])
            .set_file_name("render.png")
            .save_file()
        {
            self.queue_render(path, long_edge, camera);
        }
    }

    fn queue_render(
        &mut self,
        path: std::path::PathBuf,
        long_edge: u32,
        camera: Option<ViewportCamera>,
    ) {
        self.turntable = None;
        self.render_status = Some("Rendering…".to_owned());
        self.render_request = Some(RenderRequest {
            path,
            long_edge,
            camera,
            lighting_rotation: None,
            turntable_frame: None,
            issued: false,
        });
    }

    /// Writes the finished still render as an opaque sRGB PNG (the
    /// renderer outputs display-encoded sRGB).
    fn save_finished_render(&mut self) {
        let Some(frame) = self
            .render_result
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
        else {
            return;
        };
        let Some(request) = self.render_request.take() else {
            return;
        };
        let mut rgba8 = frame.rgba8;
        for texel in rgba8.as_chunks_mut::<4>().0 {
            texel[3] = 255;
        }
        let texture = BakedTexture {
            width: frame.width,
            height: frame.height,
            rgba8,
        };
        let written = write_texture(&request.path, &texture);
        if let Some(frame) = request.turntable_frame {
            self.turntable_frame_done(frame, written.is_ok());
        }
        self.render_status = Some(match written {
            Ok(()) => format!(
                "Saved {} × {} render to {}",
                frame.width,
                frame.height,
                request.path.display()
            ),
            Err(error) => format!("Render failed: {error}"),
        });
    }

    fn save_project(&mut self) {
        let path = self.project_path.clone().or_else(|| {
            rfd::FileDialog::new()
                .add_filter("SubstanceCraft Project", &["sproject"])
                .set_file_name("Untitled.sproject")
                .save_file()
        });
        let Some(path) = path else {
            return;
        };

        match save_project_file(&path, &self.project) {
            Ok(()) => {
                self.project_error = save_project_history(&path, &self.history)
                    .err()
                    .map(|error| format!("Saved, but the undo history was not: {error}"))
                    .or_else(|| {
                        self.bake_log
                            .save(&path)
                            .err()
                            .map(|error| format!("Saved, but the bake log was not: {error}"))
                    });
                self.project_path = Some(path);
                self.mark_project_settled();
                self.discard_autosave();
            }
            Err(error) => self.project_error = Some(error.to_string()),
        }
    }

    fn open_project(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("SubstanceCraft Project", &["sproject"])
            .pick_file()
        {
            self.open_project_path(path);
        }
    }

    /// Loads a project file, reporting failures in the status bar.
    fn open_project_path(&mut self, path: std::path::PathBuf) {
        let (mut project, report) = match load_project_file_with_report(&path) {
            Ok(loaded) => loaded,
            Err(error) => {
                self.project_error = Some(error.to_string());
                return;
            }
        };
        // Moved mesh files are found by name near the project.
        let mut relinked =
            relink::relink_mesh_sources(&mut project, &relink::search_roots(Some(&path)));
        if let Err(error) = self.adopt_project(project, Some(path.clone())) {
            self.project_error = Some(error);
            return;
        }
        let bitmaps = self.relink_missing_resources();
        if bitmaps > 0 {
            relinked.push(format!("{bitmaps} bitmap(s)"));
        }
        let mut notes: Vec<String> = report.summary().into_iter().collect();
        if !relinked.is_empty() {
            notes.push(format!("relinked {}", relinked.join(", ")));
        }
        if !notes.is_empty() {
            self.project_error = Some(format!(
                "Opened {}: {}",
                path_label(&path),
                notes.join("; ")
            ));
        }
    }

    /// Makes `project` the open project: restores its mesh, resets the
    /// selection, camera, pose, and history (restored from beside `path`
    /// when the file is unchanged), and loads its bitmaps.
    fn adopt_project(
        &mut self,
        project: MaterialProject,
        path: Option<std::path::PathBuf>,
    ) -> Result<(), String> {
        let scene = Self::restore_project_mesh(&project)?;
        self.project = project;
        self.project_path = path.clone();
        self.bake_log = path
            .as_deref()
            .map(bake_log::BakeLog::load)
            .unwrap_or_default();
        self.selected_primitive = scene.selected_primitive;
        self.imported_meshes = scene.imported_meshes;
        self.selected_imported_mesh = scene.selected_imported_mesh;
        self.import_scale = scene.scale;
        self.mesh_smoothing = scene.smoothing;
        self.set_base_geometry(scene.geometry);
        self.restore_high_poly();
        self.loaded_mesh = self.project.mesh.clone();
        self.project_error = None;
        self.reload_resources();
        self.selected_texture_set = self.project.active_texture_set;
        self.selected_layer = self
            .project
            .texture_sets
            .iter()
            .find(|set| Some(set.id) == self.selected_texture_set)
            .and_then(|set| set.layers.last())
            .map(|layer| layer.id);
        self.next_texture_set_id = self
            .project
            .texture_sets
            .iter()
            .map(|set| set.id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.next_layer_id = self
            .project
            .texture_sets
            .iter()
            .flat_map(|set| set.walk_layers().into_iter().map(|layer| layer.id.0))
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.current_pose = self
            .project
            .saved_poses
            .first()
            .cloned()
            .unwrap_or_else(|| NamedPose {
                id: PoseId(0),
                name: "Current Pose".to_owned(),
                animation: None,
                bones: BTreeMap::from([(BoneId(0), BoneTransform::IDENTITY)]),
            });
        self.viewport_camera = ViewportCamera::default();
        // The lighting preset is a user preference, not part of the project.
        self.viewport_lighting = ViewportLighting {
            preset: self.viewport_lighting.preset,
            ..ViewportLighting::default()
        };
        self.selected_pose = self.project.saved_poses.first().map(|pose| pose.id);
        self.pose_name = format!("Pose {}", self.project.saved_poses.len() + 1);
        self.pose_transition = None;
        self.active_stroke = None;
        // Undo steps saved with the project come back when the file is
        // unchanged since.
        self.history = path
            .as_deref()
            .and_then(load_project_history)
            .unwrap_or_default();
        self.layer_settled = None;
        self.editing_material = None;
        self.evolution_start = None;
        self.invalidate_all_sets();
        self.invalidate_preview();
        self.bake_error = None;
        self.import_error = None;
        self.mark_project_settled();
        Ok(())
    }

    /// Starts a new, unsaved project from `template`, keeping the current
    /// mesh: one texture set per material slot of it.
    fn start_from_template(&mut self, template: &substancecraft_core::ProjectTemplate) {
        let slots: Vec<String> = self
            .mesh_geometry
            .as_deref()
            .filter(|mesh| mesh.material_slot_count() > 1)
            .map(|mesh| {
                (0..mesh.material_slot_count() as u32)
                    .map(|slot| mesh.material_slot_name(slot))
                    .collect()
            })
            .unwrap_or_default();
        let mut project = template.instantiate(&slots);
        project.mesh = self.project.mesh.clone();
        if let Err(error) = self.adopt_project(project, None) {
            self.project_error = Some(error);
        }
    }

    /// Saves the selected texture set's settings and layers as a template.
    fn save_template_file(&mut self) {
        let Some(set) = self.selected_texture_set_ref() else {
            return;
        };
        let template =
            substancecraft_core::ProjectTemplate::from_texture_set(set, set.name.clone());
        let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "SubstanceCraft project template",
                &[substancecraft_api::PROJECT_TEMPLATE_EXTENSION],
            )
            .set_file_name(format!(
                "{}.{}",
                template.name.replace(' ', "_"),
                substancecraft_api::PROJECT_TEMPLATE_EXTENSION
            ))
            .save_file()
        else {
            return;
        };
        if let Err(error) = substancecraft_api::save_project_template(&path, &template) {
            self.project_error = Some(error.to_string());
        }
    }

    /// Asks for a template file to start a new project from.
    fn open_template_file(&mut self) -> Option<substancecraft_core::ProjectTemplate> {
        let path = rfd::FileDialog::new()
            .add_filter(
                "SubstanceCraft project template",
                &[substancecraft_api::PROJECT_TEMPLATE_EXTENSION],
            )
            .pick_file()?;
        match substancecraft_api::load_project_template(&path) {
            Ok(template) => Some(template),
            Err(error) => {
                self.project_error = Some(error.to_string());
                None
            }
        }
    }

    fn restore_project_mesh(project: &MaterialProject) -> Result<RestoredMeshScene, String> {
        let Some(mesh_asset) = &project.mesh else {
            return Ok(RestoredMeshScene {
                selected_primitive: None,
                imported_meshes: Vec::new(),
                selected_imported_mesh: None,
                geometry: None,
                scale: 1.0,
                smoothing: MeshSmoothing::default(),
            });
        };
        let smoothing = if mesh_asset.smoothing.is_valid() {
            mesh_asset.smoothing
        } else {
            MeshSmoothing::default()
        };
        if let Some(name) = mesh_asset.source.strip_prefix("builtin:") {
            let preset = PrimitivePreset::ALL
                .into_iter()
                .find(|preset| preset.name() == name)
                .ok_or_else(|| format!("unknown built-in mesh `{name}` in project"))?;
            return Ok(RestoredMeshScene {
                selected_primitive: Some(preset),
                imported_meshes: Vec::new(),
                selected_imported_mesh: None,
                geometry: Some(preset.build_mesh()),
                scale: 1.0,
                smoothing,
            });
        }

        let path = std::path::PathBuf::from(&mesh_asset.source);
        let meshes = load_mesh_file(&path)?;
        let Some(first_mesh) = meshes.first() else {
            return Err("referenced mesh file contains no mesh geometry".to_owned());
        };
        let scale = sanitize_mesh_scale(mesh_asset.scale);
        let first_geometry = scaled_geometry(&first_mesh.geometry, scale);
        Ok(RestoredMeshScene {
            selected_primitive: None,
            imported_meshes: meshes,
            selected_imported_mesh: Some(0),
            geometry: Some(first_geometry),
            scale,
            smoothing,
        })
    }

    fn import_mesh(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D mesh (FBX, glTF, STEP)", &MESH_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        self.load_imported_mesh(&path);
    }

    /// Replaces the project mesh with the file at `path`. A mesh with several
    /// material slots gets one texture set per slot; otherwise texture sets
    /// are created on first use.
    fn load_imported_mesh(&mut self, path: &std::path::Path) {
        let meshes = match load_mesh_file(path) {
            Ok(meshes) => meshes,
            Err(error) => {
                self.import_error = Some(error);
                return;
            }
        };
        let Some(first_mesh) = meshes.first() else {
            self.import_error = Some("the file contains no mesh geometry".to_owned());
            return;
        };
        let first_geometry = scaled_geometry(&first_mesh.geometry, self.import_scale);
        let display_name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Imported mesh".to_owned());
        self.project.mesh = Some(MeshAsset {
            id: MeshId(1),
            source: path.to_string_lossy().into_owned(),
            scale: self.import_scale,
            smoothing: self.mesh_smoothing,
            high_poly_source: None,
            content_hash: substancecraft_api::file_content_hash(path).ok(),
            high_poly_hash: None,
            high_to_low: Default::default(),
        });
        let slots = if first_geometry.material_slot_count() > 1 {
            first_geometry.material_slots.clone()
        } else {
            Vec::new()
        };
        self.imported_meshes = meshes;
        self.selected_imported_mesh = Some(0);
        self.selected_primitive = None;
        self.set_base_geometry(Some(first_geometry));
        // A new low mesh needs its own high-poly source.
        self.high_poly = None;
        self.high_poly_changed();
        self.import_error = None;
        self.project.texture_sets.clear();
        self.project.active_texture_set = None;
        self.selected_texture_set = None;
        self.selected_layer = None;
        self.active_stroke = None;
        self.history = EditHistory::default();
        self.layer_settled = None;
        self.set_materials.clear();
        self.loaded_mesh = self.project.mesh.clone();
        self.current_pose.name = display_name;
        for slot in &slots {
            self.add_texture_set_for_slot(slot.clone());
        }
        if let Some(first) = self.project.texture_sets.first().map(|set| set.id) {
            self.project.active_texture_set = Some(first);
            self.selected_texture_set = Some(first);
        }
    }

    fn add_texture_set(&mut self) {
        let name = format!("Material {}", self.next_texture_set_id);
        self.add_texture_set_for_slot(name);
    }

    /// Adds and selects a texture set painting mesh material slot `slot`.
    fn add_texture_set_for_slot(&mut self, slot: String) {
        let Some(next_id) = self.next_texture_set_id.checked_add(1) else {
            return;
        };
        let id = TextureSetId(self.next_texture_set_id);
        self.next_texture_set_id = next_id;
        let name = if slot.starts_with("Material ") {
            format!("Texture Set {}", id.0)
        } else {
            slot.clone()
        };
        self.project.texture_sets.push(TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id,
            name,
            material_slot: slot,
            resolution: TextureResolution {
                width: 2048,
                height: 2048,
            },
            channels: BAKEABLE_CHANNELS.into_iter().collect::<BTreeSet<_>>(),
            layers: Vec::new(),
            height_strength: DEFAULT_HEIGHT_STRENGTH,
            export: None,
        });
        self.project.active_texture_set = Some(id);
        self.selected_texture_set = Some(id);
        self.selected_layer = None;
        self.extra_selected_layers.clear();
        self.invalidate_preview();
    }

    fn allocate_layer_id(&mut self) -> Option<LayerId> {
        let next = self.next_layer_id.checked_add(1)?;
        let id = LayerId(self.next_layer_id);
        self.next_layer_id = next;
        Some(id)
    }

    /// Where a new layer goes: inside the selected group (on top), else just
    /// above the selected layer, else on top of the stack.
    fn insertion_slot(&self, set: &TextureSet) -> LayerSlot {
        let top = LayerSlot {
            parent: None,
            index: set.layers.len(),
        };
        let Some(selected) = self.selected_layer else {
            return top;
        };
        if let Some(children) = set
            .layer(selected)
            .and_then(|layer| layer.children.as_ref())
        {
            return LayerSlot {
                parent: Some(selected),
                index: children.len(),
            };
        }
        set.slot_of(selected).map_or(top, |slot| LayerSlot {
            parent: slot.parent,
            index: slot.index + 1,
        })
    }

    /// Inserts `layer` (see [`Self::insertion_slot`]) as one undoable edit,
    /// creating a texture set first when none exists.
    fn insert_layer(&mut self, layer: MaterialLayer) {
        if self.selected_texture_set.is_none() {
            self.add_texture_set();
        }
        let Some(texture_set_id) = self.selected_texture_set else {
            return;
        };
        let Some(slot) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)
            .map(|set| self.insertion_slot(set))
        else {
            return;
        };
        let layer_id = layer.id;
        if self.history.apply(
            &mut self.project,
            EditCommand::InsertLayer {
                texture_set_id,
                slot,
                layer: Box::new(layer),
            },
        ) {
            self.selected_layer = Some(layer_id);
            self.layer_settled = None;
            self.invalidate_preview_layers();
        }
    }

    /// The selected layer and those Ctrl-clicked with it that still exist
    /// in the selected texture set.
    fn flatten_selection(&self) -> Vec<LayerId> {
        let Some(set) = self.selected_texture_set_ref() else {
            return Vec::new();
        };
        self.selected_layer
            .into_iter()
            .chain(self.extra_selected_layers.iter().copied())
            .filter(|id| set.layer(*id).is_some())
            .collect()
    }

    /// Ctrl-click: adds `id` to the layers flattened together, or takes it
    /// out again. Plain clicks select one layer.
    fn toggle_extra_selection(&mut self, id: LayerId) {
        match self.selected_layer {
            None => self.selected_layer = Some(id),
            Some(selected) if selected == id => {
                // Hand the main selection to another chosen layer.
                self.selected_layer = self.extra_selected_layers.pop_last().or(Some(id));
            }
            Some(_) => {
                if !self.extra_selected_layers.remove(&id) {
                    self.extra_selected_layers.insert(id);
                }
            }
        }
    }

    /// Flattens the selected layer or group, merged with any Ctrl-clicked
    /// with it, into per-channel PNGs in `folder`, inserts a fill layer
    /// using them just above the topmost, and hides the sources, as one
    /// undoable edit. Returns the new layer.
    fn flatten_selected_into(&mut self, folder: &std::path::Path) -> Result<LayerId, String> {
        let (Some(set_id), Some(_)) = (self.selected_texture_set, self.selected_layer) else {
            return Err("Select a layer to flatten.".to_owned());
        };
        let set = self
            .selected_texture_set_ref()
            .cloned()
            .ok_or("No texture set is selected.")?;
        let order = set
            .flatten_order(&self.flatten_selection())
            .ok_or("The layer no longer exists.")?;
        let top_id = *order.last().ok_or("The layer no longer exists.")?;
        let top = set
            .layer(top_id)
            .cloned()
            .ok_or("The layer no longer exists.")?;
        let slot = set.slot_of(top_id).ok_or("The layer no longer exists.")?;
        let flattened = {
            let inputs = self.bake_inputs(set_id);
            set.flatten_layers(
                &order,
                substancecraft_core::BakeContext {
                    mesh: inputs.mesh.as_deref(),
                    mesh_maps: inputs.maps(),
                    resources: Some(&inputs.resources),
                },
            )
            .map_err(|error| error.to_string())?
        };
        if flattened.channels.is_empty() {
            return Err("The layer writes no channels to flatten.".to_owned());
        }
        std::fs::create_dir_all(folder).map_err(|error| error.to_string())?;
        let (name, stem) = if order.len() == 1 {
            (format!("{} (flattened)", top.name), top.name.clone())
        } else {
            (
                format!("{} + {} (flattened)", top.name, order.len() - 1),
                format!("{}_merged", top.name),
            )
        };
        let stem = format!("{}_{stem}", set.name).replace([' ', '/', '\\', ':'], "_");
        let mut resources = Vec::new();
        for (channel, image) in &flattened.channels {
            let path = folder.join(format!(
                "{stem}_{}.png",
                channel_label(*channel).replace(' ', "")
            ));
            write_texture(
                &path,
                &BakedTexture {
                    width: image.width,
                    height: image.height,
                    rgba8: image.rgba8.clone(),
                },
            )
            .map_err(|error| error.to_string())?;
            resources.push(self.add_resource(&path)?);
        }
        let id = self.allocate_layer_id().ok_or("Out of layer ids.")?;
        let layer = flattened.to_fill_layer(id, name, &resources);
        let mut commands: Vec<EditCommand> = order
            .iter()
            .filter_map(|source| set.layer(*source))
            .map(|source| {
                let before = source.settings();
                let after = substancecraft_core::LayerSettings {
                    visible: false,
                    ..before.clone()
                };
                EditCommand::LayerSettings {
                    texture_set_id: set_id,
                    layer_id: source.id,
                    before: Box::new(before),
                    after: Box::new(after),
                }
            })
            .collect();
        commands.push(EditCommand::InsertLayer {
            texture_set_id: set_id,
            slot: LayerSlot {
                parent: slot.parent,
                index: slot.index + 1,
            },
            layer: Box::new(layer),
        });
        if !self
            .history
            .apply(&mut self.project, EditCommand::Batch(commands))
        {
            return Err("The flattened layer could not be inserted.".to_owned());
        }
        self.selected_layer = Some(id);
        self.extra_selected_layers.clear();
        self.layer_settled = None;
        self.invalidate_preview();
        Ok(id)
    }

    /// Flattens into a folder beside the saved project, or one the user
    /// picks when the project has not been saved yet.
    fn flatten_selected_layer(&mut self) {
        let folder = match &self.project_path {
            Some(path) => {
                let stem = path.file_stem().map_or_else(
                    || "project".to_owned(),
                    |stem| stem.to_string_lossy().into_owned(),
                );
                path.with_file_name(format!("{stem}_flattened"))
            }
            None => match rfd::FileDialog::new()
                .set_title("Folder for flattened layer bitmaps")
                .pick_folder()
            {
                Some(folder) => folder,
                None => return,
            },
        };
        if let Err(error) = self.flatten_selected_into(&folder) {
            self.bake_error = Some(error);
        }
    }

    /// Materials offered for fills: the project's, then the built-ins.
    fn material_library(&self) -> Vec<substancecraft_core::MaterialResource> {
        let mut materials = self.project.materials.clone();
        materials.extend(substancecraft_core::MaterialResource::built_in());
        materials
    }

    /// Smart masks offered for layers: the project's, then the built-ins.
    fn smart_mask_library(&self) -> Vec<substancecraft_core::SmartMask> {
        let mut masks = self.project.smart_masks.clone();
        masks.extend(substancecraft_core::SmartMask::built_in());
        masks
    }

    /// Saves the selected layer's mask setup as a project smart mask.
    fn save_layer_as_smart_mask(&mut self) {
        let Some(layer) = self.selected_layer() else {
            return;
        };
        let taken: Vec<String> = self
            .smart_mask_library()
            .into_iter()
            .map(|mask| mask.name)
            .collect();
        let name = Self::unique_name(&format!("{} mask", layer.name), &taken);
        let Some(mask) = substancecraft_core::SmartMask::from_layer(layer, name) else {
            return;
        };
        let before = self.project.smart_masks.clone();
        let mut after = before.clone();
        after.push(mask);
        self.history
            .apply(&mut self.project, EditCommand::SmartMasks { before, after });
    }

    /// Moves evolution control `index` of the selected layer to `amount`;
    /// the frames of a slider drag are one undo step.
    /// The thumbnail of `material` and the hash of the layer tree it
    /// shows: from the cache while that tree is unchanged, else rendered
    /// now if `budget` allows, which spends it (the browser renders at most
    /// one new thumbnail per frame so it never stalls).
    fn smart_thumbnail_image(
        &mut self,
        material: &substancecraft_core::SmartMaterial,
        budget: &mut bool,
    ) -> Option<(u64, Arc<TextureImage>)> {
        use std::hash::{Hash, Hasher};

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        serde_json::to_string(&material.root)
            .unwrap_or_default()
            .hash(&mut hasher);
        let key = hasher.finish();
        if let Some((cached, image)) = self.smart_thumbnails.get(&material.name)
            && *cached == key
        {
            return Some((key, Arc::clone(image)));
        }
        if !*budget {
            return None;
        }
        *budget = false;
        if self.thumbnail_stage.is_none() {
            self.thumbnail_stage = substancecraft_core::ThumbnailStage::new(SMART_THUMBNAIL_SIZE)
                .ok()
                .map(Arc::new);
        }
        let stage = self.thumbnail_stage.clone()?;
        let image = Arc::new(
            stage
                .render(&material.root, Some(&self.resource_images))
                .ok()?,
        );
        self.smart_thumbnails
            .insert(material.name.clone(), (key, Arc::clone(&image)));
        Some((key, image))
    }

    /// [`Self::smart_thumbnail_image`] uploaded as an egui texture, kept
    /// until the thumbnail changes.
    fn smart_thumbnail_texture(
        &mut self,
        context: &egui::Context,
        material: &substancecraft_core::SmartMaterial,
        budget: &mut bool,
    ) -> Option<egui::TextureHandle> {
        let (key, image) = self.smart_thumbnail_image(material, budget)?;
        if let Some((uploaded, texture)) = self.smart_thumbnail_textures.get(&material.name)
            && *uploaded == key
        {
            return Some(texture.clone());
        }
        let texture = context.load_texture(
            format!("smart-thumbnail-{}", material.name),
            egui::ColorImage::from_rgba_unmultiplied(
                [image.width as usize, image.height as usize],
                &image.rgba8,
            ),
            egui::TextureOptions::LINEAR,
        );
        self.smart_thumbnail_textures
            .insert(material.name.clone(), (key, texture.clone()));
        Some(texture)
    }

    /// Sets exposed parameter `index` of the selected (inserted smart
    /// material) layer; slider drags merge into one undo step.
    fn set_exposed_parameter(&mut self, index: usize, value: substancecraft_core::ParameterValue) {
        if let (Some(set_id), Some(layer_id)) = (self.selected_texture_set, self.selected_layer)
            && let Some(command) = self.project.exposed_command(set_id, layer_id, index, value)
        {
            self.apply_material_command(command, true);
        }
    }

    fn evolve_selected_layer(&mut self, index: usize, amount: f32) {
        if let (Some(set_id), Some(layer_id)) = (self.selected_texture_set, self.selected_layer)
            && let Some(command) = self.project.evolve_command(set_id, layer_id, index, amount)
        {
            self.apply_material_command(command, true);
        }
    }

    /// Remembers the selected layer's tree as the start (`0`) of a new
    /// evolution control.
    fn start_evolution_recording(&mut self) {
        if let (Some(set_id), Some(layer)) = (self.selected_texture_set, self.selected_layer()) {
            self.evolution_start = Some((set_id, layer.id, layer.clone()));
        }
    }

    /// Ends a recording: the selected layer gets a control whose drives run
    /// from the recorded start to the layer tree as it is now.
    fn finish_evolution_recording(&mut self) {
        let Some((set_id, layer_id, start)) = self.evolution_start.take() else {
            return;
        };
        if self.selected_texture_set != Some(set_id) || self.selected_layer != Some(layer_id) {
            return;
        }
        let name = self.evolution_name.trim().to_owned();
        let Some(layer) = self.selected_layer_mut() else {
            return;
        };
        let evolution = substancecraft_core::Evolution::between(name, &start, layer);
        if evolution.drives.is_empty() {
            self.project_error = Some(
                "Nothing changed since the start was recorded: change the layers' fill values, \
                 opacity, or generator amounts first."
                    .to_owned(),
            );
            return;
        }
        if layer.evolutions.len() < substancecraft_core::MAX_EVOLUTIONS {
            layer.evolutions.push(evolution);
            self.invalidate_preview();
        }
    }

    /// Removes the project smart mask at `index`.
    fn delete_smart_mask(&mut self, index: usize) {
        let before = self.project.smart_masks.clone();
        if index >= before.len() {
            return;
        }
        let mut after = before.clone();
        after.remove(index);
        self.history
            .apply(&mut self.project, EditCommand::SmartMasks { before, after });
    }

    fn smart_material_library(&self) -> Vec<substancecraft_core::SmartMaterial> {
        let mut materials = self.project.smart_materials.clone();
        materials.extend(substancecraft_core::SmartMaterial::built_in());
        materials
    }

    /// Makes the selected layer a fill of `material` (recorded as a layer
    /// settings edit).
    fn apply_material(&mut self, material: &substancecraft_core::MaterialResource) {
        if let Some(layer) = self.selected_layer_mut()
            && !layer.is_group()
        {
            material.apply_to(layer);
            self.invalidate_preview();
        }
    }

    /// A name not yet used by any entry of `taken`.
    fn unique_name(base: &str, taken: &[String]) -> String {
        (1..)
            .map(|index| match index {
                1 => base.to_owned(),
                index => format!("{base} {index}"),
            })
            .find(|candidate| {
                !taken
                    .iter()
                    .any(|name| name.eq_ignore_ascii_case(candidate))
            })
            .unwrap_or_else(|| base.to_owned())
    }

    fn save_layer_as_material(&mut self) {
        let Some(layer) = self.selected_layer() else {
            return;
        };
        let taken: Vec<String> = self
            .material_library()
            .into_iter()
            .map(|m| m.name)
            .collect();
        let name = Self::unique_name(&layer.name, &taken);
        if let Some(material) = substancecraft_core::MaterialResource::from_layer(layer, name)
            && let Some((_, command)) = self.project.add_material_command(material)
        {
            self.history.apply(&mut self.project, command);
        }
    }

    /// Applies a material or material-instance command, re-snapshots the
    /// selected layer so the inspector does not record the change again,
    /// and marks every texture set it touched out of date. `merging` joins
    /// the frames of a slider drag into one undo step.
    fn apply_material_command(&mut self, command: EditCommand, merging: bool) -> bool {
        let touched = edited_texture_sets(&command);
        let applied = if merging {
            self.history.apply_merging(&mut self.project, command)
        } else {
            self.history.apply(&mut self.project, command)
        };
        if applied {
            self.layer_settled = None;
            // Instances hold the master's values in their own fills, so the
            // sets' snapshot diffs find what changed: no strokes did.
            self.invalidate_set_texels(&touched, Some(BTreeMap::new()));
        }
        applied
    }

    /// Makes the selected layer an instance of `material`. A material not in
    /// the project yet (a built-in) becomes a new project master first, in
    /// the same undo step.
    fn link_selected_layer(&mut self, material: &substancecraft_core::MaterialResource) {
        let (Some(set_id), Some(layer_id)) = (self.selected_texture_set, self.selected_layer)
        else {
            return;
        };
        let command = if self.project.material(material.id).is_some() {
            self.project
                .link_material_command(set_id, layer_id, material.id)
        } else {
            let taken: Vec<String> = self
                .project
                .materials
                .iter()
                .map(|material| material.name.clone())
                .collect();
            let master = substancecraft_core::MaterialResource {
                name: Self::unique_name(&material.name, &taken),
                ..material.clone()
            };
            self.project
                .link_new_material_command(set_id, layer_id, master)
                .map(|(_, command)| command)
        };
        if let Some(command) = command {
            self.apply_material_command(command, false);
        }
    }

    /// Saves the selected fill as a new project material and links the
    /// layer to it, so other layers can share it.
    fn make_selected_layer_master(&mut self) {
        let (Some(set_id), Some(layer)) = (self.selected_texture_set, self.selected_layer()) else {
            return;
        };
        let taken: Vec<String> = self
            .material_library()
            .into_iter()
            .map(|material| material.name)
            .collect();
        let name = Self::unique_name(&layer.name, &taken);
        let layer_id = layer.id;
        let Some(material) = substancecraft_core::MaterialResource::from_layer(layer, name) else {
            return;
        };
        if let Some((id, command)) = self
            .project
            .link_new_material_command(set_id, layer_id, material)
            && self.apply_material_command(command, false)
        {
            self.editing_material = Some(id);
        }
    }

    fn unlink_selected_layer(&mut self) {
        if let (Some(set_id), Some(layer_id)) = (self.selected_texture_set, self.selected_layer)
            && let Some(command) = self.project.unlink_material_command(set_id, layer_id)
        {
            self.apply_material_command(command, false);
        }
    }

    fn reset_instance_override(&mut self, field: substancecraft_core::InstanceField) {
        if let (Some(set_id), Some(layer_id)) = (self.selected_texture_set, self.selected_layer)
            && let Some(command) = self.project.reset_override_command(set_id, layer_id, field)
        {
            self.apply_material_command(command, false);
        }
    }

    /// Replaces a project material, updating every instance. Slider drags
    /// merge into one undo step.
    fn edit_project_material(&mut self, material: substancecraft_core::MaterialResource) {
        if let Some(command) = self.project.edit_material_command(material) {
            self.apply_material_command(command, true);
        }
    }

    /// Deletes a project material; its instances keep their values as
    /// plain fills.
    fn delete_project_material(&mut self, id: substancecraft_core::MaterialId) {
        if let Some(command) = self.project.remove_material_command(id)
            && self.apply_material_command(command, false)
            && self.editing_material == Some(id)
        {
            self.editing_material = None;
        }
    }

    fn save_layer_as_smart_material(&mut self) {
        let Some(layer) = self.selected_layer() else {
            return;
        };
        let taken: Vec<String> = self
            .smart_material_library()
            .into_iter()
            .map(|m| m.name)
            .collect();
        let name = Self::unique_name(&layer.name, &taken);
        let material = substancecraft_core::SmartMaterial::from_layer(layer, name);
        self.project.smart_materials.push(material);
    }

    /// Inserts a fresh instance of `material` above the selection.
    fn insert_smart_material(&mut self, material: &substancecraft_core::SmartMaterial) {
        let count = material.root.walk().len();
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            match self.allocate_layer_id() {
                Some(id) => ids.push(id),
                None => return,
            }
        }
        let mut ids = ids.into_iter();
        let root = material.instantiate(|| ids.next().unwrap_or(LayerId(u64::MAX)));
        self.insert_layer(root);
    }

    fn export_smart_material_file(&mut self, material: &substancecraft_core::SmartMaterial) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("SubstanceCraft smart material", &["scsmart"])
            .set_file_name(format!("{}.scsmart", material.name.replace(' ', "_")))
            .save_file()
        else {
            return;
        };
        if let Err(error) = substancecraft_api::save_smart_material(&path, material) {
            self.project_error = Some(error.to_string());
        }
    }

    fn import_smart_material_file(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("SubstanceCraft smart material", &["scsmart"])
            .pick_file()
        else {
            return;
        };
        match substancecraft_api::load_smart_material(&path) {
            Ok(material) => {
                let requirements = material.requirements();
                if let Some(missing) = requirements.resources.iter().find(|id| {
                    !self
                        .project
                        .resources
                        .iter()
                        .any(|resource| resource.id == **id)
                }) {
                    self.project_error = Some(format!(
                        "`{}` uses bitmap resource {} that this project does not have.",
                        material.name, missing.0
                    ));
                    return;
                }
                self.insert_smart_material(&material);
            }
            Err(error) => self.project_error = Some(error.to_string()),
        }
    }

    fn add_paint_layer(&mut self) {
        let Some(id) = self.allocate_layer_id() else {
            return;
        };
        self.insert_layer(MaterialLayer::new_paint(
            id,
            format!("Paint Layer {}", id.0),
        ));
    }

    fn add_fill_layer(&mut self) {
        let Some(id) = self.allocate_layer_id() else {
            return;
        };
        let emissive = self.brush_emissive.to_array();
        let fill = LayerFill {
            color: self.brush_color.to_array(),
            roughness: self.brush_roughness,
            metallic: self.brush_metallic,
            height: self.brush_height,
            emissive: [emissive[0], emissive[1], emissive[2]],
            material_opacity: self.brush_material_opacity,
            ..LayerFill::default()
        };
        self.insert_layer(MaterialLayer::new_fill(
            id,
            format!("Fill Layer {}", id.0),
            fill,
        ));
    }

    fn add_group_layer(&mut self) {
        let Some(id) = self.allocate_layer_id() else {
            return;
        };
        self.insert_layer(MaterialLayer::new_group(id, format!("Group {}", id.0)));
    }

    /// Adds a fill layer pre-wired to a procedural mask generator.
    fn add_smart_layer(&mut self, generator: MaskGenerator) {
        let Some(id) = self.allocate_layer_id() else {
            return;
        };
        let (name, fill) = match generator {
            MaskGenerator::EdgeWear(_) => (
                "Edge Wear",
                LayerFill {
                    color: [196, 198, 204, 255],
                    roughness: 0.28,
                    metallic: 1.0,
                    // Chipped paint sits slightly below the coat around it.
                    height: 0.42,
                    ..LayerFill::default()
                },
            ),
            MaskGenerator::Dirt(_) => (
                "Dirt",
                LayerFill {
                    color: [62, 50, 38, 255],
                    roughness: 0.92,
                    metallic: 0.0,
                    height: DEFAULT_HEIGHT,
                    ..LayerFill::default()
                },
            ),
            _ => ("Generated", LayerFill::default()),
        };
        let mut layer = MaterialLayer::new_fill(id, name, fill);
        layer.mask_generator = Some(generator);
        self.insert_layer(layer);
    }

    fn add_edge_wear_layer(&mut self) {
        self.add_smart_layer(MaskGenerator::EdgeWear(EdgeWear::DEFAULT));
    }

    fn add_dirt_layer(&mut self) {
        self.add_smart_layer(MaskGenerator::Dirt(Dirt::DEFAULT));
    }

    /// The selected layer's texture set, slot, and container length.
    fn selected_layer_slot(&self) -> Option<(TextureSetId, LayerSlot, usize)> {
        let set = self.selected_texture_set_ref()?;
        let slot = set.slot_of(self.selected_layer?)?;
        let count = set.container(slot.parent)?.len();
        Some((set.id, slot, count))
    }

    fn delete_selected_layer(&mut self) {
        self.finish_active_stroke();
        let Some((texture_set_id, slot, _)) = self.selected_layer_slot() else {
            return;
        };
        let Some(layer) = self
            .selected_texture_set_ref()
            .and_then(|set| set.container(slot.parent))
            .and_then(|layers| layers.get(slot.index))
            .cloned()
        else {
            return;
        };
        if self.history.apply(
            &mut self.project,
            EditCommand::RemoveLayer {
                texture_set_id,
                slot,
                layer: Box::new(layer),
            },
        ) {
            self.selected_layer = self
                .selected_texture_set_ref()
                .and_then(|set| set.container(slot.parent))
                .and_then(|layers| layers.get(slot.index.saturating_sub(1)))
                .map(|layer| layer.id)
                .or(slot.parent);
            self.layer_settled = None;
            self.invalidate_preview_layers();
        }
    }

    fn move_selected_layer_to(&mut self, to: LayerSlot) -> bool {
        let Some((texture_set_id, from, _)) = self.selected_layer_slot() else {
            return false;
        };
        let moved = self.history.apply(
            &mut self.project,
            EditCommand::MoveLayer {
                texture_set_id,
                from,
                to,
            },
        );
        if moved {
            self.invalidate_preview();
        }
        moved
    }

    /// Moves the selected layer up (+1, toward the top of its group) or down.
    fn move_selected_layer(&mut self, delta: isize) {
        let Some((_, slot, count)) = self.selected_layer_slot() else {
            return;
        };
        if let Some(index) = slot
            .index
            .checked_add_signed(delta)
            .filter(|index| *index < count)
        {
            self.move_selected_layer_to(LayerSlot {
                parent: slot.parent,
                index,
            });
        }
    }

    /// The group directly below the selected layer, which it can move into.
    fn group_below_selected(&self) -> Option<LayerId> {
        let (_, slot, _) = self.selected_layer_slot()?;
        let below = self
            .selected_texture_set_ref()?
            .container(slot.parent)?
            .get(slot.index.checked_sub(1)?)?;
        below.is_group().then_some(below.id)
    }

    /// Moves the selected layer onto the top of the group below it.
    fn move_selected_into_group(&mut self) {
        let Some(group) = self.group_below_selected() else {
            return;
        };
        let Some(count) = self
            .selected_texture_set_ref()
            .and_then(|set| set.container(Some(group)))
            .map(Vec::len)
        else {
            return;
        };
        self.move_selected_layer_to(LayerSlot {
            parent: Some(group),
            index: count,
        });
    }

    /// Moves the selected layer out of its group, just above the group.
    fn move_selected_out_of_group(&mut self) {
        let Some((_, slot, _)) = self.selected_layer_slot() else {
            return;
        };
        let Some(group_slot) = slot
            .parent
            .and_then(|group| self.selected_texture_set_ref()?.slot_of(group))
        else {
            return;
        };
        self.move_selected_layer_to(LayerSlot {
            parent: group_slot.parent,
            index: group_slot.index + 1,
        });
    }

    fn selected_layer_mut(&mut self) -> Option<&mut MaterialLayer> {
        let texture_set_id = self.selected_texture_set?;
        let layer_id = self.selected_layer?;
        self.project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)?
            .layer_mut(layer_id)
    }

    fn selected_layer(&self) -> Option<&MaterialLayer> {
        self.selected_texture_set_ref()?.layer(self.selected_layer?)
    }

    /// Records inspector edits of the selected layer as single history steps.
    /// Changes accumulate while the pointer is held (slider drags) and are
    /// committed on release.
    fn sync_layer_history(&mut self, context: &egui::Context) {
        // A slider drag's property edits stop merging once it ends.
        if !context.input(|input| input.pointer.any_down()) {
            self.history.end_merge();
        }
        let current =
            self.selected_texture_set
                .zip(self.selected_layer)
                .and_then(|(set_id, layer_id)| {
                    self.selected_layer()
                        .map(|layer| (set_id, layer_id, layer.settings()))
                });
        let Some((set_id, layer_id, settings)) = current else {
            self.layer_settled = None;
            return;
        };
        let same_layer =
            self.layer_settled
                .as_ref()
                .is_some_and(|(settled_set, settled_layer, _)| {
                    *settled_set == set_id && *settled_layer == layer_id
                });
        if !same_layer {
            self.layer_settled = Some((set_id, layer_id, settings));
            return;
        }
        if context.input(|input| input.pointer.any_down()) {
            return;
        }
        if let Some((_, _, before)) = self.layer_settled.take() {
            let mut settings = settings;
            if before != settings {
                // Editing a field an instance takes from its master makes
                // that field an explicit override.
                let tracked = self.project.track_instance_edit(settings.clone());
                if tracked != settings {
                    if let Some(layer) = self.selected_layer_mut() {
                        layer.apply_settings(tracked.clone());
                    }
                    settings = tracked;
                }
                self.history.record(EditCommand::LayerSettings {
                    texture_set_id: set_id,
                    layer_id,
                    before: Box::new(before),
                    after: Box::new(settings.clone()),
                });
                self.invalidate_preview_layers();
            }
            self.layer_settled = Some((set_id, layer_id, settings));
        }
    }

    /// Adds (or with `remove`, subtracts) triangles from the selected layer's
    /// polygon mask.
    fn apply_polygon_selection(&mut self, triangles: &[u32], remove: bool) {
        let Some(layer) = self.selected_layer_mut() else {
            return;
        };
        let mask = layer.polygon_mask.get_or_insert_with(Default::default);
        let selecting = remove == mask.inverted;
        for triangle in triangles {
            if selecting {
                mask.triangles.insert(*triangle);
            } else {
                mask.triangles.remove(triangle);
            }
        }
        self.invalidate_preview();
    }

    /// Adds `uv` to the stroke in progress, starting one when needed. Paint
    /// and ID strokes skip groups; mask strokes work on any layer.
    fn append_stroke_point(&mut self, uv: glam::Vec2) {
        self.stroke_point(0, uv, None, false);
    }

    /// True when strokes are drawn on screen and projected onto the surface:
    /// projection is on and the mode paints, erases, or masks (ID strokes
    /// stay in UV space).
    fn projects_strokes(&self) -> bool {
        // Projected strokes rasterise the bind mesh, so they wait for it.
        // Clone and smudge sample in texture space. The skew map feeds the
        // mesh-map bake, so it cannot wait for maps of its own.
        self.projection_painting
            && !self.is_posed()
            && !self.painting_skew()
            && matches!(
                self.paint_mode,
                PaintMode::Paint | PaintMode::Erase | PaintMode::Mask
            )
            && (self.stroke_tools.source == stroke_tools::BrushSource::Paint
                || self.paint_mode != PaintMode::Paint
                || self
                    .stroke_tools
                    .quick_mask
                    .as_ref()
                    .is_some_and(|mask| mask.editing))
    }

    /// Adds `point` (viewport pixels) to the projected stroke in progress,
    /// starting one painted through `camera` when there is none or the view
    /// changed. The brush radius is in screen pixels; symmetry does not
    /// apply to screen-space strokes.
    fn camera_stroke_point(
        &mut self,
        point: glam::Vec2,
        camera: substancecraft_core::StrokeCamera,
    ) {
        if !self.projects_strokes() {
            return;
        }
        if let Some(set_id) = self.selected_texture_set
            && self.quick_mask_point(set_id, 0, point, None, Some(camera), false)
        {
            return;
        }
        let (Some(texture_set_id), Some(layer_id)) =
            (self.selected_texture_set, self.selected_layer)
        else {
            return;
        };
        let mode = self.paint_mode;
        let stroke_key = (texture_set_id, layer_id, mode);
        let starts_stroke = self.active_stroke != Some(stroke_key);
        let pressure = self.pen_pressure;
        let pressures = pressure.map(|value| vec![value]).unwrap_or_default();
        let stencil = self.active_stencil(texture_set_id);
        let paint = PaintStroke {
            camera: Some(camera),
            pressures: pressures.clone(),
            stencil,
            ..self.brush_paint_stroke(mode == PaintMode::Erase)
        };
        let (radius, tip) = (paint.radius_pixels, paint.tip);
        let mask = MaskStroke {
            camera: Some(camera),
            tip,
            pressures,
            stencil,
            ..MaskStroke::new(Vec::new(), radius, self.mask_value)
        };
        let maps = self
            .bake_inputs(texture_set_id)
            .mesh_maps
            .zip(Some(self.set_slot(texture_set_id) as usize));
        let Some(texture_set) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)
        else {
            return;
        };
        // Dirty texels are tracked on the preview, which may be downscaled.
        let factor = texture_set.preview_factor(PREVIEW_MAX_EDGE);
        let set_width = texture_set.resolution.width.div_ceil(factor);
        let set_height = texture_set.resolution.height.div_ceil(factor);
        let Some(layer) = texture_set.layer_mut(layer_id) else {
            return;
        };
        if mode != PaintMode::Mask && layer.is_group() {
            return;
        }
        let count = match mode {
            PaintMode::Mask => layer.mask_strokes.len(),
            _ => layer.paint_strokes.len(),
        };
        if starts_stroke {
            self.stroke_first_index = count;
            self.stroke_paths.clear();
        }
        // The stroke in progress, while it was painted through this view.
        let current = self.stroke_paths.first().copied().flatten();
        let target = current.and_then(|index| match mode {
            PaintMode::Mask => layer
                .mask_strokes
                .get_mut(index)
                .filter(|stroke| stroke.camera == Some(camera))
                .map(|stroke| (&mut stroke.points, &mut stroke.pressures)),
            _ => layer
                .paint_strokes
                .get_mut(index)
                .filter(|stroke| stroke.camera == Some(camera))
                .map(|stroke| (&mut stroke.points, &mut stroke.pressures)),
        });
        let changed: Vec<glam::Vec2> = match target {
            Some((points, pressures)) => {
                if points.len() >= MAX_STROKE_POINTS
                    || points
                        .last()
                        .is_some_and(|last| last.distance_squared(point) < 0.25)
                {
                    return;
                }
                points.push(point);
                if !pressures.is_empty() {
                    pressures.push(pressure.unwrap_or(1.0));
                }
                // Stamps are placed causally, so only the new segment changes.
                points[points.len() - 2..].to_vec()
            }
            None => {
                match mode {
                    PaintMode::Mask => layer.mask_strokes.push(MaskStroke {
                        points: vec![point],
                        ..mask
                    }),
                    _ => layer.paint_strokes.push(PaintStroke {
                        points: vec![point],
                        ..paint
                    }),
                }
                self.stroke_paths = vec![Some(count)];
                vec![point]
            }
        };
        self.active_stroke = Some(stroke_key);
        match maps.as_ref().and_then(|(maps, slot)| maps.get(*slot)) {
            Some(maps) => {
                self.invalidate_preview_texels(substancecraft_core::projected_stroke_texel_rects(
                    &changed, radius, tip, &camera, maps, set_width, set_height,
                ))
            }
            // Without mesh maps the preview rebakes whole once they arrive.
            None => self.invalidate_preview(),
        }
    }

    /// Adds `uv` to stroke path `path` of the current drag (0 is the brush,
    /// 1 its mirror), with the stamp `frame` there when stamps align to the
    /// surface. `restart` begins a new stroke for that path, e.g. after
    /// crossing a UV seam. Every stroke a drag creates undoes together.
    /// While a quick mask is being painted, strokes go into it instead.
    fn stroke_point(
        &mut self,
        path: usize,
        uv: glam::Vec2,
        frame: Option<substancecraft_core::StampFrame>,
        restart: bool,
    ) {
        if !self.paint_mode.paints() {
            return;
        }
        if let Some(set_id) = self.selected_texture_set
            && self.quick_mask_point(set_id, path, uv, frame, None, restart)
        {
            return;
        }
        let (Some(texture_set_id), Some(layer_id)) =
            (self.selected_texture_set, self.selected_layer)
        else {
            return;
        };
        if self.paint_mode == PaintMode::Paint && !self.stroke_tools.ready() {
            // Clone waits for its source.
            return;
        }
        let stencil = self.active_stencil(texture_set_id);
        let source = match self.paint_mode {
            PaintMode::Paint => self.stroke_tools.stroke_source(uv),
            _ => None,
        };
        let frames = frame.map(|frame| vec![frame]).unwrap_or_default();
        let mode = self.paint_mode;
        let stroke_key = (texture_set_id, layer_id, mode);
        let starts_stroke = self.active_stroke != Some(stroke_key);
        let pressure = self.pen_pressure;
        let color = self.brush_color.to_array();
        let emissive = self.brush_emissive.to_array();
        let tip = self.brush_tip;
        let radius_pixels = self.brush_radius;
        let (roughness, metallic, height, material_opacity) = (
            self.brush_roughness,
            self.brush_metallic,
            self.brush_height,
            self.brush_material_opacity,
        );
        let mask_value = self.mask_value;
        let user = self.brush_user;
        let Some(texture_set) = self
            .project
            .texture_sets
            .iter_mut()
            .find(|set| set.id == texture_set_id)
        else {
            return;
        };
        // Dirty texels are tracked on the preview, which may be downscaled.
        let factor = texture_set.preview_factor(PREVIEW_MAX_EDGE);
        let set_width = texture_set.resolution.width.div_ceil(factor);
        let set_height = texture_set.resolution.height.div_ceil(factor);
        let preview_radius = |radius: f32| radius / factor as f32;
        let Some(layer) = texture_set.layer_mut(layer_id) else {
            return;
        };
        let paints_content = matches!(mode, PaintMode::Paint | PaintMode::Erase | PaintMode::Id);
        if paints_content && layer.is_group() {
            return;
        }
        // Pressure data is kept only for pen strokes.
        let pressures =
            |pressure: Option<f32>| pressure.map(|value| vec![value]).unwrap_or_default();

        let stroke_count = match mode {
            PaintMode::View | PaintMode::Select => return,
            PaintMode::Paint | PaintMode::Erase => layer.paint_strokes.len(),
            PaintMode::Mask => layer.mask_strokes.len(),
            PaintMode::Id => layer.id_strokes.len(),
        };
        if starts_stroke {
            self.stroke_first_index = stroke_count;
            self.stroke_paths.clear();
        }
        let current = self.stroke_paths.get(path).copied().flatten();
        if restart || current.is_none() {
            match mode {
                PaintMode::View | PaintMode::Select => return,
                PaintMode::Paint | PaintMode::Erase => layer.paint_strokes.push(PaintStroke {
                    roughness,
                    metallic,
                    height,
                    emissive: [emissive[0], emissive[1], emissive[2]],
                    material_opacity,
                    erase: mode == PaintMode::Erase,
                    tip,
                    pressures: pressures(pressure),
                    frames: frames.clone(),
                    source,
                    stencil,
                    user,
                    ..PaintStroke::new(vec![uv], radius_pixels, color)
                }),
                PaintMode::Mask => layer.mask_strokes.push(MaskStroke {
                    points: vec![uv],
                    radius_pixels,
                    value: mask_value,
                    opacity: 1.0,
                    tip,
                    pressures: pressures(pressure),
                    camera: None,
                    frames: frames.clone(),
                    stencil,
                }),
                PaintMode::Id => layer.id_strokes.push(IdStroke {
                    points: vec![uv],
                    radius_pixels,
                    color: [color[0], color[1], color[2]],
                    opacity: 1.0,
                    tip,
                    pressures: pressures(pressure),
                    frames: frames.clone(),
                }),
            }
            if self.stroke_paths.len() <= path {
                self.stroke_paths.resize(path + 1, None);
            }
            self.stroke_paths[path] = Some(stroke_count);
            self.active_stroke = Some(stroke_key);
            let bounds = stroke_texel_bounds(
                &[uv],
                preview_radius(radius_pixels),
                tip,
                &frames,
                set_width,
                set_height,
            );
            self.invalidate_preview_texels(bounds);
            return;
        }

        let Some(index) = current else {
            return;
        };
        let target = match mode {
            PaintMode::View | PaintMode::Select => return,
            PaintMode::Paint | PaintMode::Erase => {
                layer.paint_strokes.get_mut(index).map(|stroke| {
                    let shape = (stroke.radius_pixels, stroke.tip);
                    (
                        &mut stroke.points,
                        &mut stroke.pressures,
                        &mut stroke.frames,
                        shape,
                    )
                })
            }
            PaintMode::Mask => layer.mask_strokes.get_mut(index).map(|stroke| {
                let shape = (stroke.radius_pixels, stroke.tip);
                (
                    &mut stroke.points,
                    &mut stroke.pressures,
                    &mut stroke.frames,
                    shape,
                )
            }),
            PaintMode::Id => layer.id_strokes.get_mut(index).map(|stroke| {
                let shape = (stroke.radius_pixels, stroke.tip);
                (
                    &mut stroke.points,
                    &mut stroke.pressures,
                    &mut stroke.frames,
                    shape,
                )
            }),
        };
        if let Some((points, pressures, stroke_frames, (radius, tip))) = target
            && points.len() < MAX_STROKE_POINTS
            && points
                .last()
                .is_none_or(|last| last.distance_squared(uv) >= 0.000_001)
        {
            points.push(uv);
            // A pen stroke keeps one pressure per point.
            if !pressures.is_empty() {
                pressures.push(pressure.unwrap_or(1.0));
            }
            // An aligned stroke keeps one frame per point; a hit without a
            // usable frame stamps unaligned.
            if !stroke_frames.is_empty() {
                stroke_frames.push(frame.unwrap_or(substancecraft_core::StampFrame::IDENTITY));
            }
            // Stamps are placed causally, so only the new segment changes.
            let start = points.len() - 2;
            let bounds = stroke_texel_bounds(
                &points[start..],
                preview_radius(radius),
                tip,
                stroke_frames.get(start..).unwrap_or_default(),
                set_width,
                set_height,
            );
            self.invalidate_preview_texels(bounds);
        }
    }

    fn finish_active_stroke(&mut self) {
        self.stroke_tools.end_quick_stroke();
        let Some((texture_set_id, layer_id, mode)) = self.active_stroke.take() else {
            return;
        };
        let Some(layer) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)
            .and_then(|set| set.layer(layer_id))
        else {
            return;
        };
        let first = self.stroke_first_index;
        let mut commands: Vec<EditCommand> = match mode {
            PaintMode::View | PaintMode::Select => Vec::new(),
            PaintMode::Paint | PaintMode::Erase => layer
                .paint_strokes
                .get(first..)
                .unwrap_or_default()
                .iter()
                .map(|stroke| EditCommand::Paint {
                    texture_set_id,
                    layer_id,
                    stroke: stroke.clone(),
                })
                .collect(),
            PaintMode::Mask => layer
                .mask_strokes
                .get(first..)
                .unwrap_or_default()
                .iter()
                .map(|stroke| EditCommand::Mask {
                    texture_set_id,
                    layer_id,
                    stroke: stroke.clone(),
                })
                .collect(),
            PaintMode::Id => layer
                .id_strokes
                .get(first..)
                .unwrap_or_default()
                .iter()
                .map(|stroke| EditCommand::Id {
                    texture_set_id,
                    layer_id,
                    stroke: stroke.clone(),
                })
                .collect(),
        };
        self.stroke_paths.clear();
        self.stroke_last_triangles.clear();
        if mode == PaintMode::Paint && !commands.is_empty() {
            self.remember_brush_material();
        }
        match commands.len() {
            0 => {}
            1 => self.history.record(commands.remove(0)),
            _ => self.history.record(EditCommand::Batch(commands)),
        }
    }

    /// Routes a viewport click or drag: eyedropper, path placement,
    /// Shift-click straight lines, or stabilized brush strokes.
    #[allow(clippy::too_many_arguments)]
    fn handle_paint_input(
        &mut self,
        response: &egui::Response,
        points: &[ProjectedVertex],
        mesh: &MeshGeometry,
        camera: (substancecraft_core::StrokeCamera, egui::Pos2),
        pointer: Option<egui::Pos2>,
        shift: bool,
    ) {
        let pick = |app: &Self, at: egui::Pos2| {
            pick_triangle(mesh, points, at, Some(app.hidden_triangles()))
                .filter(|hit| app.triangle_in_selected_set(hit.triangle))
        };
        let Some(pointer) = pointer else {
            return;
        };
        let decal_key = egui::Id::new(panels::PLACE_DECAL_KEY);
        if let Some(target) = response.ctx.data(|data| data.get_temp::<u8>(decal_key)) {
            if response.clicked()
                && let Some(hit) = pick(self, pointer)
            {
                response.ctx.data_mut(|data| data.remove::<u8>(decal_key));
                self.place_decal(target, mesh, hit);
            }
            return;
        }
        if self.active_decal().is_some() {
            if response.dragged() || response.clicked() {
                let command = response.ctx.input(|input| input.modifiers.command);
                let hit = pick(self, pointer);
                self.drag_decal(mesh, hit, response.drag_delta(), shift, command);
            }
            return;
        }
        if self.inspect.armed {
            if response.clicked()
                && let Some(uv) = pick(self, pointer).and_then(|hit| hit.uv)
            {
                self.inspect_at(uv);
            }
            return;
        }
        if self.picking_material {
            if response.clicked()
                && let Some(uv) = pick(self, pointer).and_then(|hit| hit.uv)
            {
                self.pick_material_at(uv);
            }
            return;
        }
        if self.picking_id_color {
            if response.clicked()
                && let Some(uv) = pick(self, pointer).and_then(|hit| hit.uv)
            {
                self.pick_id_color_at(uv);
            }
            return;
        }
        if self.stroke_tools.picking_source {
            if response.clicked()
                && let Some(uv) = pick(self, pointer).and_then(|hit| hit.uv)
            {
                self.stroke_tools.set_clone_source(uv);
            }
            return;
        }
        if !self.paint_mode.paints() {
            return;
        }
        if self.path_controls.is_some() {
            if response.clicked()
                && let Some(hit) = pick(self, pointer)
                && let Some(uv) = hit.uv
                && let Some(controls) = &mut self.path_controls
            {
                controls.push((uv, pointer, hit.position));
            }
            return;
        }
        if self.projects_strokes() {
            // Screen-space strokes, relative to the viewport's corner.
            let (camera, origin) = camera;
            let local = |at: egui::Pos2| glam::Vec2::new(at.x - origin.x, at.y - origin.y);
            if shift
                && response.clicked()
                && let Some(from) = self.last_paint_screen
            {
                self.camera_stroke_point(local(from), camera);
                self.camera_stroke_point(local(pointer), camera);
                self.finish_active_stroke();
                self.invalidate_preview();
                self.last_paint_screen = Some(pointer);
            } else if response.dragged() {
                if response.drag_started() {
                    self.stroke_stabilizer.reset();
                }
                if let Some(at) = self.stabilized_cursor(pointer) {
                    self.camera_stroke_point(local(at), camera);
                    self.last_paint_screen = Some(at);
                }
            }
            return;
        }
        if shift
            && response.clicked()
            && let Some(from) = self.last_paint_screen
        {
            let from = glam::Vec2::new(from.x, from.y);
            let to = glam::Vec2::new(pointer.x, pointer.y);
            for point in substancecraft_core::line_points(from, to, 2.0) {
                if let Some(hit) = pick(self, egui::pos2(point.x, point.y)) {
                    self.paint_hit(hit);
                }
            }
            self.finish_active_stroke();
            self.invalidate_preview();
            self.last_paint_screen = Some(pointer);
            return;
        }
        if response.dragged() {
            if response.drag_started() {
                self.stroke_stabilizer.reset();
            }
            let Some(at) = self.stabilized_cursor(pointer) else {
                return;
            };
            if let Some(hit) = pick(self, at) {
                self.paint_hit(hit);
                self.last_paint_screen = Some(at);
            }
        }
    }

    fn stabilized_cursor(&mut self, pointer: egui::Pos2) -> Option<egui::Pos2> {
        self.stroke_stabilizer
            .update(glam::Vec2::new(pointer.x, pointer.y))
            .map(|point| egui::pos2(point.x, point.y))
    }

    /// The decal the viewport gizmo moves (0 = fill, 1 = bitmap mask) and
    /// its placement, while the gizmo is on and the selected layer still
    /// projects that target as a decal.
    fn active_decal(&self) -> Option<(u8, substancecraft_core::DecalPlacement)> {
        let target = self.decal_gizmo?;
        let layer = self.selected_layer()?;
        let projection = match target {
            0 => layer.fill.map(|fill| fill.projection),
            _ => layer.mask_bitmap.map(|mask| mask.projection),
        }?;
        (projection.mode == substancecraft_core::ProjectionMode::Decal)
            .then_some((target, projection.decal))
    }

    /// Applies one gizmo drag step to the active decal: by default it
    /// slides to the surface point under the pointer, facing out of that
    /// face and keeping its spin; Shift spins it by the horizontal motion
    /// (half a degree per point) and Ctrl resizes it by the vertical motion
    /// (dragging up grows it). Edits merge into one undo step per drag.
    fn drag_decal(
        &mut self,
        mesh: &MeshGeometry,
        hit: Option<TriangleHit>,
        delta: egui::Vec2,
        shift: bool,
        command: bool,
    ) {
        let Some((target, _)) = self.active_decal() else {
            return;
        };
        let surface = hit.and_then(|hit| {
            let corners = mesh.triangles.get(hit.triangle as usize)?;
            let [a, b, c] = corners.map(|corner| mesh.vertices[corner as usize].position);
            Some((hit.position, (b - a).cross(c - a)))
        });
        let Some(layer) = self.selected_layer_mut() else {
            return;
        };
        let projection = match target {
            0 => layer.fill.as_mut().map(|fill| &mut fill.projection),
            _ => layer.mask_bitmap.as_mut().map(|mask| &mut mask.projection),
        };
        let Some(decal) = projection.map(|projection| &mut projection.decal) else {
            return;
        };
        let before = *decal;
        if shift {
            decal.set_spin_degrees(decal.spin_degrees() + delta.x * 0.5);
        } else if command {
            decal.scale_by((-delta.y * 0.01).exp());
        } else if let Some((position, normal)) = surface {
            decal.move_to(position, normal);
        }
        if *decal != before {
            self.invalidate_preview();
        }
    }

    /// Outlines the active decal's image rectangle on the model, with a
    /// tick from its centre towards the image's top edge.
    fn draw_decal_gizmo(&self, painter: &egui::Painter, projection: &ViewProjection) {
        let Some((_, decal)) = self.active_decal() else {
            return;
        };
        let corners: Vec<egui::Pos2> = decal
            .corners()
            .into_iter()
            .map(|corner| projection.project(corner).screen)
            .collect();
        let stroke = egui::Stroke::new(1.5, theme::ACCENT);
        painter.add(egui::Shape::closed_line(corners.clone(), stroke));
        let top = corners[2].lerp(corners[3], 0.5);
        let centre = projection.project(decal.center).screen;
        painter.line_segment([centre, top], stroke);
        painter.circle_filled(centre, 3.0, theme::ACCENT);
    }

    /// Centres the selected layer's decal (0 = fill, 1 = bitmap mask) on a
    /// surface point, facing out of the clicked face.
    fn place_decal(&mut self, target: u8, mesh: &MeshGeometry, hit: TriangleHit) {
        let Some(corners) = mesh.triangles.get(hit.triangle as usize) else {
            return;
        };
        let [a, b, c] = corners.map(|corner| mesh.vertices[corner as usize].position);
        let normal = (b - a).cross(c - a).normalize_or(glam::Vec3::Z);
        let up = if normal.dot(glam::Vec3::Y).abs() > 0.9 {
            glam::Vec3::NEG_Z
        } else {
            glam::Vec3::Y
        };
        let Some(layer) = self.selected_layer_mut() else {
            return;
        };
        let projection = match target {
            0 => layer.fill.as_mut().map(|fill| &mut fill.projection),
            _ => layer.mask_bitmap.as_mut().map(|mask| &mut mask.projection),
        };
        if let Some(projection) = projection {
            projection.mode = substancecraft_core::ProjectionMode::Decal;
            projection.decal.center = hit.position;
            projection.decal.normal = normal;
            projection.decal.up = up;
            self.invalidate_preview();
        }
    }

    /// Turns the placed path into strokes on the selected layer, using the
    /// current brush, as one undo step. The path follows the surface, so
    /// it gives one stroke per UV island it crosses.
    fn commit_path(&mut self, closed: bool) {
        let Some(controls) = self.path_controls.take() else {
            return;
        };
        let (Some(texture_set_id), Some(layer_id)) =
            (self.selected_texture_set, self.selected_layer)
        else {
            return;
        };
        if controls.len() < 2 || !matches!(self.paint_mode, PaintMode::Paint | PaintMode::Erase) {
            return;
        }
        let path = substancecraft_core::PathSpec {
            controls: controls.iter().map(|(uv, ..)| *uv).collect(),
            closed,
            surface: controls.iter().map(|(.., position)| *position).collect(),
        };
        let mut template = self.brush_paint_stroke(self.paint_mode == PaintMode::Erase);
        template.stencil = self.active_stencil(texture_set_id);
        if self.paint_mode == PaintMode::Paint {
            if !self.stroke_tools.ready() {
                return;
            }
            template.source = self.stroke_tools.stroke_source(path.controls[0]);
        }
        let strokes = match (&self.mesh_geometry, &self.mesh_topology) {
            (Some(mesh), Some(topology)) => {
                PaintStroke::along_surface_path(path, &template, mesh, topology)
            }
            _ => vec![PaintStroke::along_path(path, &template)],
        };
        let mut commands: Vec<EditCommand> = strokes
            .into_iter()
            .map(|stroke| EditCommand::Paint {
                texture_set_id,
                layer_id,
                stroke,
            })
            .collect();
        let command = match commands.len() {
            0 => return,
            1 => commands.remove(0),
            _ => EditCommand::Batch(commands),
        };
        if self.history.apply(&mut self.project, command) {
            self.invalidate_preview();
        }
    }

    /// A stroke carrying the current brush material and tip, with no points.
    fn brush_paint_stroke(&self, erase: bool) -> PaintStroke {
        let color = self.brush_color.to_array();
        let emissive = self.brush_emissive.to_array();
        PaintStroke {
            roughness: self.brush_roughness,
            metallic: self.brush_metallic,
            height: self.brush_height,
            emissive: [emissive[0], emissive[1], emissive[2]],
            material_opacity: self.brush_material_opacity,
            erase,
            tip: self.brush_tip,
            user: self.brush_user,
            ..PaintStroke::new(Vec::new(), self.brush_radius, color)
        }
    }

    /// Eyedropper: copies every material channel at `uv` of the selected
    /// texture set into the brush.
    fn pick_material_at(&mut self, uv: glam::Vec2) {
        self.picking_material = false;
        let Some(texture_set) = self.selected_texture_set_ref().cloned() else {
            return;
        };
        let shown = match self.displayed_preview() {
            Some(PreviewTextures::Material(material)) => Some(Arc::clone(material)),
            _ => None,
        };
        let material = match shown {
            Some(material) => material,
            None => match self.bake_inputs(texture_set.id).bake_material(&texture_set) {
                Ok(material) => Arc::new(material),
                Err(error) => {
                    self.bake_error = Some(error);
                    return;
                }
            },
        };
        if let Some(sample) = material.sample(uv) {
            self.apply_material_sample(sample);
        }
    }

    /// Sets the selected layer's ID generator to the ID colour at `uv`;
    /// unpainted texels leave it unchanged.
    fn pick_id_color_at(&mut self, uv: glam::Vec2) {
        self.picking_id_color = false;
        let Some(texture_set) = self.selected_texture_set_ref() else {
            return;
        };
        let texel = match texture_set.bake_id_mask() {
            Ok(ids) => ids.sample_uv(uv),
            Err(error) => {
                self.bake_error = Some(error.to_string());
                return;
            }
        };
        if let Some([r, g, b, alpha]) = texel
            && alpha > 0
            && let Some(MaskGenerator::IdColor(selection)) = self
                .selected_layer_mut()
                .and_then(|layer| layer.mask_generator.as_mut())
        {
            selection.color = [r, g, b];
            self.invalidate_preview();
        }
    }

    fn apply_material_sample(&mut self, sample: substancecraft_core::MaterialSample) {
        let [r, g, b] = sample.color;
        self.brush_color = egui::Color32::from_rgb(r, g, b);
        self.brush_roughness = sample.roughness;
        self.brush_metallic = sample.metallic;
        self.brush_material_opacity = sample.opacity;
        let [r, g, b] = sample.emissive;
        self.brush_emissive = egui::Color32::from_rgb(r, g, b);
    }

    /// The symmetric copies of a brush hit that paint: with symmetry on,
    /// those landing on the selected texture set inside the unit UV square,
    /// each with its stroke path number (1 onwards, in copy order).
    fn symmetry_copies(
        &self,
        hit: &TriangleHit,
        uv: glam::Vec2,
    ) -> Vec<(usize, substancecraft_core::SurfaceHit)> {
        let (Some(symmetry), Some(mesh)) = (self.symmetry, self.mesh_geometry.as_deref()) else {
            return Vec::new();
        };
        symmetry
            .symmetric_hits(
                mesh,
                &substancecraft_core::SurfaceHit {
                    triangle: hit.triangle,
                    position: hit.position,
                    normal: glam::Vec3::Z,
                    uv,
                },
            )
            .into_iter()
            .enumerate()
            .filter_map(|(index, copy)| Some((index + 1, copy?)))
            .filter(|(_, copy)| {
                self.triangle_in_selected_set(copy.triangle) && self.paintable_uv(copy.uv)
            })
            .collect()
    }

    /// True when strokes may land at `uv`: in tile 1001, or in any UDIM
    /// tile the mesh uses.
    fn paintable_uv(&self, uv: glam::Vec2) -> bool {
        if uv.cmpge(glam::Vec2::ZERO).all() && uv.cmple(glam::Vec2::ONE).all() {
            return true;
        }
        self.udim_atlas.as_ref().is_some_and(|atlas| {
            substancecraft_core::UdimTile::of_uv(uv)
                .is_some_and(|tile| atlas.tiles().contains(&tile))
        })
    }

    /// Paints at a picked surface point: continues the brush stroke
    /// (restarting it when the pointer crosses into another UV island) and,
    /// with symmetry on, the mirrored stroke on the other side.
    fn paint_hit(&mut self, hit: TriangleHit) {
        let Some(uv) = hit.uv.filter(|uv| self.paintable_uv(*uv)) else {
            return;
        };
        if self.active_stroke.is_none() {
            self.stroke_last_triangles.clear();
        }
        let mut targets = vec![(0, hit.triangle, uv)];
        // Path numbers follow the copy order, so each copy keeps its own
        // stroke through the drag.
        targets.extend(
            self.symmetry_copies(&hit, uv)
                .into_iter()
                .map(|(path, copy)| (path, copy.triangle, copy.uv)),
        );
        for (path, triangle, uv) in targets {
            let previous = self.stroke_last_triangles.get(path).copied().flatten();
            let restart = match (previous, &self.mesh_topology) {
                (Some(previous), Some(topology)) => {
                    substancecraft_core::crosses_uv_seam(topology, previous, triangle)
                }
                _ => false,
            };
            let frame = self.hit_frame(triangle);
            self.stroke_point(path, uv, frame, restart);
            if self.stroke_last_triangles.len() <= path {
                self.stroke_last_triangles.resize(path + 1, None);
            }
            self.stroke_last_triangles[path] = Some(triangle);
        }
    }

    fn undo(&mut self) {
        self.finish_active_stroke();
        if self.history.undo(&mut self.project) {
            let undone = self.history.last_undone().cloned();
            self.after_history_change(undone.as_ref());
        }
    }

    fn redo(&mut self) {
        self.finish_active_stroke();
        if self.history.redo(&mut self.project) {
            let redone = self.history.last_done().cloned();
            self.after_history_change(redone.as_ref());
        }
    }

    /// Brings caches and previews up to date after `command` was undone or
    /// redone. Edits of layers alone mark only their sets out of date, and
    /// the selected set's preview is patched.
    fn after_history_change(&mut self, command: Option<&EditCommand>) {
        let previews = self
            .project
            .texture_sets
            .iter()
            .map(|set| {
                let factor = set.preview_factor(PREVIEW_MAX_EDGE);
                (
                    set.id,
                    (
                        set.resolution.width.div_ceil(factor),
                        set.resolution.height.div_ceil(factor),
                        factor,
                    ),
                )
            })
            .collect();
        let (mut sets, mut strokes) = (BTreeSet::new(), BTreeMap::new());
        let layers_only = command
            .is_some_and(|command| layer_edit_scope(command, &previews, &mut sets, &mut strokes));
        if self.project.mesh != self.loaded_mesh {
            self.reload_project_mesh();
        }
        // The settled snapshot must not see undo/redo as a new user edit.
        self.layer_settled = None;
        self.reload_relinked_resources();
        let selected_exists = self.selected_layer().is_some();
        if !selected_exists {
            self.selected_layer = self
                .project
                .texture_sets
                .iter()
                .find(|set| Some(set.id) == self.selected_texture_set)
                .and_then(|set| set.layers.last())
                .map(|layer| layer.id);
        }
        if layers_only {
            self.invalidate_set_texels(&sets, Some(strokes));
        } else {
            self.invalidate_all_sets();
        }
        self.refresh_baked_preview();
    }

    fn preview_target(&self) -> PreviewTarget {
        if let Some(id) = self
            .selected_texture_set
            .and_then(|set| self.editing_quick_mask(set))
        {
            return PreviewTarget::QuickMask(id);
        }
        match self.active_solo() {
            Some(inspector::Solo::Mask(id)) => return PreviewTarget::LayerMask(id),
            Some(inspector::Solo::Generator(id)) => return PreviewTarget::Generator(id),
            Some(inspector::Solo::Overlay(layer)) => {
                let preferences = &self.preferences.values;
                return PreviewTarget::MaskOverlay {
                    layer,
                    opacity: (self.inspect.overlay_opacity * 100.0).round() as u8,
                    palette: preferences.overlay_palette,
                    pattern: preferences.overlay_pattern,
                };
            }
            _ => {}
        }
        if let Some(kind) = self.mesh_map_preview {
            PreviewTarget::MeshMap(kind)
        } else if self.channel_preview {
            PreviewTarget::Channel(self.active_channel)
        } else {
            PreviewTarget::Material
        }
    }

    fn bake_key(&self) -> BakeKey {
        BakeKey {
            mesh_revision: self.mesh_revision,
            high_poly_revision: self.high_poly_revision,
            import_scale: self.import_scale,
            smoothing: self.mesh_smoothing,
            settings: self.effective_mesh_map_settings(),
            skew_map: self.skew_map,
            skew_map_revision: self.skew_map_revision,
            high_to_low: self.high_to_low_key(),
        }
    }

    fn mesh_maps_fresh(&self) -> bool {
        self.mesh_maps.is_some() && self.mesh_maps_key == Some(self.bake_key())
    }

    fn mesh_map_stale_reason(&self) -> Option<&'static str> {
        let baked = self.mesh_maps_key?;
        let current = self.bake_key();
        if baked == current {
            return None;
        }
        if baked.high_poly_revision != current.high_poly_revision {
            Some("High-poly source changed")
        } else if baked.high_to_low != current.high_to_low {
            Some("High-to-low options changed")
        } else if baked.import_scale != current.import_scale {
            Some("Scale changed")
        } else if baked.smoothing != current.smoothing {
            Some("Smoothing changed")
        } else if baked.settings != current.settings
            || baked.skew_map != current.skew_map
            || baked.skew_map_revision != current.skew_map_revision
        {
            Some("Bake settings changed")
        } else if baked.mesh_revision != current.mesh_revision {
            Some("Mesh changed")
        } else {
            Some("Bake settings changed")
        }
    }

    /// Mesh material slot of the selected texture set.
    fn selected_slot(&self) -> u32 {
        self.selected_texture_set
            .map_or(0, |set_id| self.set_slot(set_id))
    }

    /// Fresh mesh maps of the selected texture set's material slot.
    fn fresh_mesh_maps(&self) -> Option<&MeshMaps> {
        self.fresh_slot_maps(self.selected_slot())
    }

    fn fresh_slot_maps(&self, slot: u32) -> Option<&MeshMaps> {
        self.mesh_maps
            .as_deref()
            .filter(|_| self.mesh_maps_fresh())
            .and_then(|maps| maps.get(slot as usize))
    }

    fn mesh_map_job_running(&self) -> bool {
        self.mesh_map_job
            .as_ref()
            .is_some_and(|job| job.key == self.bake_key())
    }

    fn effective_mesh_map_settings(&self) -> MeshMapSettings {
        MeshMapSettings {
            smoothing_angle_degrees: self.mesh_smoothing.angle_degrees,
            ..self.mesh_map_settings
        }
    }

    fn cancel_mesh_map_job(&mut self) {
        if let Some(job) = self.mesh_map_job.take() {
            job.cancel.store(true, Ordering::Relaxed);
            self.log_bake(&job, bake_log::BakeOutcome::Cancelled, None);
        }
    }

    fn log_bake(&mut self, job: &MeshMapJob, outcome: bake_log::BakeOutcome, rays: Option<String>) {
        self.bake_log.record(bake_log::BakeLogEntry {
            started: job.started_at,
            duration: job.started.elapsed(),
            settings: job.settings.clone(),
            outcome,
            rays,
        });
    }

    /// Starts a background bake of the current mesh's maps (high-to-low when
    /// a high-poly source is loaded), replacing any bake already running.
    fn start_mesh_map_bake(&mut self) {
        self.cancel_mesh_map_job();
        let Some(mesh) = self.mesh_geometry.clone() else {
            return;
        };
        let high = self
            .high_poly
            .as_ref()
            .map(|source| source.geometry.clone());
        let settings = self.effective_mesh_map_settings();
        let skew_bitmap = self
            .skew_map
            .and_then(|id| self.resource_images.get(&id).cloned());
        let cancel = Arc::new(AtomicBool::new(false));
        let summary = bake_log::settings_summary(&settings, high.is_some());
        let mut plan = HighToLowPlan {
            parts: self
                .high_poly
                .as_ref()
                .map(|source| source.parts.clone())
                .unwrap_or_default(),
            options: self.high_to_low_options(),
            low_name: self.low_mesh_name(),
            cage: None,
        };
        let cage = self.cage_mesh.as_ref().map(|(_, cage)| cage.clone());
        let smoothing = self.mesh_smoothing;
        let gpu_baking = self.preferences.values.gpu_baking;
        let (sender, receiver) = sync_channel(1);
        let worker_cancel = Arc::clone(&cancel);
        let spawned = std::thread::Builder::new()
            .name("mesh-map-bake".to_owned())
            .spawn(move || {
                // The cage is smoothed like the low mesh so they still pair.
                plan.cage = cage.and_then(|cage| smoothing.apply(&cage).ok());
                // A skew map painted in the app wins over a bitmap.
                let skew_map = plan
                    .options
                    .skew_image(settings.resolution)
                    .map(Arc::new)
                    .or(skew_bitmap);
                // Occlusion rays on the GPU when asked for and available.
                let tracer = if gpu_baking {
                    substancecraft_gpu::FallbackTracer::shared()
                } else {
                    substancecraft_gpu::FallbackTracer::new(None)
                };
                let result = bake_slot_mesh_maps(
                    &mesh,
                    high.as_ref(),
                    &settings,
                    skew_map.as_deref(),
                    &plan,
                    &tracer,
                    &worker_cancel,
                );
                let _ = sender.send((result, rays_used(&tracer, gpu_baking)));
            });
        match spawned {
            Ok(_) => {
                self.mesh_map_error = None;
                self.mesh_map_job = Some(MeshMapJob {
                    receiver,
                    cancel,
                    key: self.bake_key(),
                    started: std::time::Instant::now(),
                    started_at: bake_log::now_seconds(),
                    settings: summary,
                });
            }
            Err(error) => self.mesh_map_error = Some(error.to_string()),
        }
    }

    /// Bakes mesh maps automatically when something needs them and they are
    /// missing or stale for the current mesh and high-poly source.
    fn ensure_mesh_maps(&mut self) {
        if self.mesh_maps_fresh() || self.mesh_map_job_running() || self.mesh_map_error.is_some() {
            return;
        }
        self.start_mesh_map_bake();
    }

    fn poll_mesh_map_job(&mut self, context: &egui::Context) {
        let Some(job) = &self.mesh_map_job else {
            return;
        };
        match job.receiver.try_recv() {
            Ok((result, rays)) => {
                let key = job.key;
                let Some(job) = self.mesh_map_job.take() else {
                    return;
                };
                let outcome = match result {
                    Ok(maps) if key == self.bake_key() => {
                        let outcome = bake_log::finished(&maps);
                        self.mesh_maps = Some(Arc::new(maps));
                        self.mesh_maps_key = Some(key);
                        self.invalidate_all_sets();
                        outcome
                    }
                    Ok(_) => bake_log::BakeOutcome::Discarded,
                    Err(MeshMapError::Cancelled) => bake_log::BakeOutcome::Cancelled,
                    Err(error) => {
                        self.mesh_map_error = Some(error.to_string());
                        bake_log::BakeOutcome::Failed(error.to_string())
                    }
                };
                self.log_bake(&job, outcome, Some(rays));
            }
            Err(TryRecvError::Empty) => context.request_repaint_after(Duration::from_millis(100)),
            Err(TryRecvError::Disconnected) => {
                self.mesh_map_job = None;
                self.mesh_map_error = Some("mesh map bake stopped unexpectedly".to_owned());
            }
        }
    }

    /// Mesh maps become stale whenever the high-poly source changes.
    fn high_poly_changed(&mut self) {
        self.resource_watcher
            .watch_high_poly(self.high_poly.as_ref().map(|high| high.path.as_path()));
        self.high_poly_revision = self.high_poly_revision.wrapping_add(1);
        self.cancel_mesh_map_job();
        self.mesh_map_error = None;
        self.invalidate_all_sets();
    }

    fn pick_high_poly(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D mesh (FBX, glTF, STEP)", &MESH_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        match load_high_poly(&path, self.import_scale) {
            Ok(source) => {
                let before = self.project.mesh.clone();
                if let Some(mesh) = self.project.mesh.as_mut() {
                    mesh.high_poly_source = Some(path.to_string_lossy().into_owned());
                    mesh.high_poly_hash = substancecraft_api::file_content_hash(&path).ok();
                }
                self.history.end_merge();
                self.record_mesh_edit(before, Vec::new());
                self.history.end_merge();
                self.high_poly = Some(source);
                self.high_poly_changed();
                self.start_mesh_map_bake();
            }
            Err(error) => self.mesh_map_error = Some(format!("High poly: {error}")),
        }
    }

    /// Re-reads the high-poly source after it changed on disk and rebakes
    /// the mesh maps from it. A file that cannot be read (gone, or caught
    /// mid-export) keeps the previous high poly and is reported.
    fn reload_high_poly(&mut self) {
        self.resource_watcher.high_poly_pending = false;
        let Some(path) = self.high_poly.as_ref().map(|high| high.path.clone()) else {
            return;
        };
        match load_high_poly(&path, self.import_scale) {
            Ok(source) => {
                if let Some(mesh) = self.project.mesh.as_mut() {
                    mesh.high_poly_hash = substancecraft_api::file_content_hash(&path).ok();
                }
                self.high_poly = Some(source);
                self.high_poly_changed();
                self.start_mesh_map_bake();
            }
            Err(error) => {
                self.mesh_map_error = Some(format!(
                    "Could not reload high poly `{}`: {error}; keeping the previous one.",
                    path.display()
                ));
            }
        }
    }

    fn clear_high_poly(&mut self) {
        let before = self.project.mesh.clone();
        self.high_poly = None;
        if let Some(mesh) = self.project.mesh.as_mut() {
            mesh.high_poly_source = None;
            mesh.high_poly_hash = None;
        }
        self.history.end_merge();
        self.record_mesh_edit(before, Vec::new());
        self.history.end_merge();
        self.high_poly_changed();
    }

    /// The mesh's high-to-low options.
    fn high_to_low_options(&self) -> substancecraft_core::HighToLowOptions {
        self.project
            .mesh
            .as_ref()
            .map(|mesh| mesh.high_to_low.clone())
            .unwrap_or_default()
    }

    /// Bake-key part of the high-to-low options and the loaded cage.
    fn high_to_low_key(&self) -> u64 {
        udim_view::key_hash(&(
            self.high_to_low_options(),
            self.cage_mesh
                .as_ref()
                .map(|(_, cage)| (cage.vertices.len(), cage.triangles.len())),
        ))
    }

    /// Changes the high-to-low options as one undoable mesh edit; mesh maps
    /// go stale and rebake when next needed.
    fn set_high_to_low(&mut self, options: substancecraft_core::HighToLowOptions) {
        if !options.is_valid() || options == self.high_to_low_options() {
            return;
        }
        let before = self.project.mesh.clone();
        let Some(mesh) = self.project.mesh.as_mut() else {
            return;
        };
        mesh.high_to_low = options;
        self.history.end_merge();
        self.record_mesh_edit(before, Vec::new());
        self.history.end_merge();
        self.restore_cage();
        self.high_poly_changed();
    }

    /// The low mesh's own name, for name matching when it has one slot.
    fn low_mesh_name(&self) -> String {
        self.selected_imported_mesh
            .and_then(|index| self.imported_meshes.get(index))
            .map(|mesh| mesh.name.clone())
            .filter(|name| !name.is_empty())
            .or_else(|| {
                let source = &self.project.mesh.as_ref()?.source;
                std::path::Path::new(source)
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
            })
            .unwrap_or_default()
    }

    /// Low parts no high-poly part matches by name, when matching.
    fn unmatched_low_parts(&self) -> Vec<String> {
        let (Some(mesh), Some(high)) = (self.mesh_geometry.as_deref(), &self.high_poly) else {
            return Vec::new();
        };
        HighToLowPlan {
            parts: high.parts.clone(),
            options: self.high_to_low_options(),
            low_name: self.low_mesh_name(),
            cage: None,
        }
        .unmatched(mesh)
    }

    fn pick_cage(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D mesh (FBX, glTF, STEP)", &MESH_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        if let Err(error) = load_high_poly(&path, self.import_scale) {
            self.mesh_map_error = Some(format!("Cage: {error}"));
            return;
        }
        self.set_high_to_low(substancecraft_core::HighToLowOptions {
            cage_source: Some(path.to_string_lossy().into_owned()),
            ..self.high_to_low_options()
        });
    }

    /// Loads the cage file the options name, reporting (not failing on) a
    /// missing or unreadable one.
    fn restore_cage(&mut self) {
        self.cage_mesh = None;
        if let Some(source) = self.high_to_low_options().cage_source {
            let path = std::path::PathBuf::from(source);
            match load_high_poly(&path, self.import_scale) {
                Ok(cage) => self.cage_mesh = Some((path, cage.geometry)),
                Err(error) => {
                    self.mesh_map_error = Some(format!("Cage `{}`: {error}", path.display()));
                }
            }
        }
    }

    /// Reloads the project's high-poly reference, reporting (not failing on)
    /// a missing or unreadable file.
    fn restore_high_poly(&mut self) {
        self.restore_cage();
        self.high_poly = None;
        let source = self
            .project
            .mesh
            .as_ref()
            .and_then(|mesh| mesh.high_poly_source.clone());
        if let Some(source) = source {
            let path = std::path::PathBuf::from(source);
            match load_high_poly(&path, self.import_scale) {
                Ok(source) => self.high_poly = Some(source),
                Err(error) => {
                    self.mesh_map_error = Some(format!("High poly `{}`: {error}", path.display()));
                }
            }
        }
        self.high_poly_changed();
    }

    /// Texture set painted on mesh material slot `slot`. A single-slot mesh
    /// shows the selected set; otherwise slots match sets by material name.
    fn slot_texture_set(&self, slot: u32) -> Option<TextureSetId> {
        let mesh = self.mesh_geometry.as_deref()?;
        if mesh.material_slot_count() <= 1 {
            return self.selected_texture_set;
        }
        let name = mesh.material_slot_name(slot);
        self.project
            .texture_sets
            .iter()
            .find(|set| set.material_slot == name)
            .map(|set| set.id)
    }

    /// Mesh material slot a texture set paints; 0 when its name matches none.
    fn set_slot(&self, set_id: TextureSetId) -> u32 {
        let Some(mesh) = self.mesh_geometry.as_deref() else {
            return 0;
        };
        self.project
            .texture_sets
            .iter()
            .find(|set| set.id == set_id)
            .and_then(|set| {
                mesh.material_slots
                    .iter()
                    .position(|slot| *slot == set.material_slot)
            })
            .map_or(0, |slot| slot as u32)
    }

    /// True when triangle `triangle` belongs to the selected texture set's
    /// material slot, so strokes there paint that set.
    fn triangle_in_selected_set(&self, triangle: u32) -> bool {
        let (Some(mesh), Some(set_id)) = (self.mesh_geometry.as_deref(), self.selected_texture_set)
        else {
            return false;
        };
        mesh.material_slot_count() <= 1
            || mesh.material_of(triangle as usize) == self.set_slot(set_id)
    }

    /// Textures each mesh material slot should show for `target`.
    fn slot_upload_plan(&self, target: PreviewTarget) -> Vec<SlotSource> {
        let slot_count = self
            .mesh_geometry
            .as_deref()
            .map_or(1, MeshGeometry::material_slot_count);
        (0..slot_count as u32)
            .map(|slot| {
                let set_id = self.slot_texture_set(slot);
                let selected = set_id.is_some() && set_id == self.selected_texture_set;
                let mesh_map_preview = matches!(target, PreviewTarget::MeshMap(_));
                if selected || (mesh_map_preview && self.set_slot_matches_preview(slot)) {
                    return match self.displayed_preview() {
                        Some(PreviewTextures::Material(textures)) => {
                            SlotSource::Material(Arc::clone(textures))
                        }
                        Some(PreviewTextures::Single(texture)) => {
                            SlotSource::Single(Arc::clone(texture))
                        }
                        None => SlotSource::Neutral,
                    };
                }
                if self.active_solo() == Some(inspector::Solo::TextureSet) {
                    return SlotSource::Neutral;
                }
                match (target, set_id.and_then(|id| self.set_materials.get(&id))) {
                    (PreviewTarget::Material, Some(cached)) => {
                        SlotSource::Material(Arc::clone(&cached.textures))
                    }
                    _ => SlotSource::Neutral,
                }
            })
            .collect()
    }

    /// Mesh-map previews show on the selected set's slot (or the whole mesh
    /// when no set is selected).
    fn set_slot_matches_preview(&self, slot: u32) -> bool {
        self.selected_texture_set
            .map_or(slot == 0, |set_id| self.set_slot(set_id) == slot)
    }

    fn selected_texture_set_ref(&self) -> Option<&TextureSet> {
        self.project
            .texture_sets
            .iter()
            .find(|set| Some(set.id) == self.selected_texture_set)
    }

    /// Identity of the preview the viewport should show right now.
    fn preview_key(&self) -> PreviewKey {
        let target = self.preview_target();
        PreviewKey {
            texture_set: match target {
                PreviewTarget::MeshMap(_) => None,
                _ => self.selected_texture_set,
            },
            target,
            solo_layer: match self.active_solo() {
                Some(inspector::Solo::Layer(id)) => Some(id),
                _ => None,
            },
            revision: self.preview_revision,
        }
    }

    /// The shown preview and the blocks to rebake to bring it up to date
    /// for `key`, when it shows the same view and the edits since it was
    /// baked stay within some texels or channels; `None` means a full
    /// rebake. A soloed layer is patched after strokes only, since solo
    /// rewrites the rest of the stack.
    fn preview_patch(
        &self,
        key: &PreviewKey,
    ) -> Option<(PreviewTextures, Vec<incremental::Patch>)> {
        // UDIM atlases rebake whole tiles, unchanged ones from the cache.
        if self.udim_atlas.is_some() {
            return None;
        }
        let set = preview_copy(self.selected_texture_set_ref()?);
        let shown = self.baked_preview.as_ref().filter(|preview| {
            preview.key.texture_set == key.texture_set
                && preview.key.target == key.target
                && preview.key.solo_layer == key.solo_layer
        });
        let (textures, patches) = match shown {
            Some(preview) => {
                let dirty = self.preview_dirty.as_ref()?;
                let patches = if key.solo_layer.is_some() {
                    dirty.stroke_regions(&set, preview.set_revision)?
                } else {
                    dirty.regions(&set, preview.set_revision)?
                };
                (preview.textures.clone(), patches)
            }
            // Another view was shown (another set was selected): start
            // from this set's cached material.
            None if caches_material(key) => {
                let (textures, patches) = self.cached_material_patch(&set)?;
                (PreviewTextures::Material(textures), patches)
            }
            None => return None,
        };
        let (textures, patches) = match (&textures, key.target) {
            (PreviewTextures::Material(material), PreviewTarget::Material) => (
                PreviewTextures::Material(Arc::clone(material)),
                patches
                    .into_iter()
                    .filter(|patch| !patch.channels.planes().is_empty())
                    .collect(),
            ),
            (PreviewTextures::Single(texture), PreviewTarget::Channel(channel)) => (
                PreviewTextures::Single(Arc::clone(texture)),
                patches
                    .into_iter()
                    .filter(|patch| patch.channels.reach(channel))
                    .collect(),
            ),
            _ => return None,
        };
        Some((textures, patches))
    }

    /// The cached material of `set` (its preview copy) and the blocks to
    /// rebake to bring it up to date, when the edits since it was baked
    /// stay within some texels or channels; `None` means a full rebake.
    fn cached_material_patch(
        &self,
        set: &TextureSet,
    ) -> Option<(Arc<MaterialTextures>, Vec<incremental::Patch>)> {
        // UDIM atlases rebake whole tiles, unchanged ones from the cache.
        if self.udim_atlas.is_some() {
            return None;
        }
        let cached = self.set_materials.get(&set.id)?;
        let record = self
            .set_dirty
            .get(&set.id)
            .filter(|record| record.maps_fresh == self.mesh_maps_fresh())?;
        let patches = record
            .dirty
            .regions(set, cached.revision)?
            .into_iter()
            .filter(|patch| !patch.channels.planes().is_empty())
            .collect();
        Some((Arc::clone(&cached.textures), patches))
    }

    /// Starts tracking what `set` (its preview copy, at `revision`) changes
    /// from here on, for the material about to be cached for it.
    fn track_set_edits(&mut self, set: &TextureSet, revision: u64) {
        if self.udim_atlas.is_some() {
            self.set_dirty.remove(&set.id);
            return;
        }
        let record = SetDirty {
            dirty: incremental::PreviewDirty::new(set, revision),
            maps_fresh: self.mesh_maps_fresh(),
        };
        self.set_dirty.insert(set.id, record);
    }

    /// Starts (at most one) background bake of the current preview. The last
    /// finished preview stays on screen until its replacement arrives, so
    /// painting updates live without flicker.
    fn refresh_baked_preview(&mut self) {
        let key = self.preview_key();
        if self.preview_failed
            || self.preview_job.is_some()
            || self
                .baked_preview
                .as_ref()
                .is_some_and(|preview| preview.key == key)
        {
            return;
        }
        if let PreviewTarget::MeshMap(kind) = key.target {
            if self.udim_atlas.is_some() && self.fresh_mesh_maps().is_some() {
                // Every tile's maps, some baked on first use: a job.
                let inputs = self.bake_inputs(self.selected_texture_set.unwrap_or(TextureSetId(0)));
                let (sender, receiver) = sync_channel(1);
                let spawned = std::thread::Builder::new()
                    .name("preview-bake".to_owned())
                    .spawn(move || {
                        let result = inputs
                            .udim
                            .as_ref()
                            .ok_or_else(|| "no UDIM atlas".to_owned())
                            .and_then(|udim| udim.mesh_map(&inputs, kind))
                            .map(|texture| PreviewTextures::Single(Arc::new(texture)));
                        let _ = sender.send((result, false));
                    });
                if spawned.is_ok() {
                    self.preview_job = Some(PreviewJob {
                        receiver,
                        key,
                        set_revision: 0,
                        patch: None,
                    });
                }
                return;
            }
            if let Some(maps) = self.fresh_mesh_maps() {
                // Mesh-map previews are cheap conversions; no job needed.
                let texture = maps.to_texture(kind);
                self.baked_preview = Some(BakedPreview {
                    key,
                    textures: PreviewTextures::Single(Arc::new(texture)),
                    set_revision: 0,
                    patch: None,
                });
            } else {
                self.ensure_mesh_maps();
            }
            return;
        }
        if let PreviewTarget::Channel(channel) = key.target
            && !BAKEABLE_CHANNELS.contains(&channel)
            && channel.user_slot().is_none()
        {
            return;
        }
        let Some(texture_set) = self.selected_texture_set_ref().cloned() else {
            self.baked_preview = None;
            return;
        };
        // Solo shows one layer alone; the project itself is untouched.
        let texture_set = match key.solo_layer {
            Some(layer) => texture_set.soloed(layer).unwrap_or(texture_set),
            None => texture_set,
        };
        let texture_set = texture_set
            .downscaled(self.preview_edge())
            .unwrap_or(texture_set);
        if texture_set.requires_mesh_maps() && !self.mesh_maps_fresh() {
            // Keep the previous preview until fresh maps arrive.
            self.ensure_mesh_maps();
            return;
        }
        let set_revision = self.set_revision(texture_set.id);
        let inputs = self.bake_inputs(texture_set.id);
        let patch = self.preview_patch(&key);
        // Strokes from here on are tracked against this bake.
        self.preview_dirty = match key.target {
            _ if self.udim_atlas.is_some() => None,
            PreviewTarget::Material | PreviewTarget::Channel(_) => self
                .selected_texture_set_ref()
                .map(|set| incremental::PreviewDirty::new(&preview_copy(set), set_revision)),
            _ => None,
        };
        if caches_material(&key) {
            self.track_set_edits(&texture_set, set_revision);
        }
        if let Some((textures, regions)) = &patch
            && regions.is_empty()
        {
            // Nothing the preview shows changed. The patch of the preview
            // it came from, if that is the one shown, still applies.
            let shown_patch = self
                .baked_preview
                .take()
                .filter(|preview| {
                    preview.key.texture_set == key.texture_set
                        && preview.key.target == key.target
                        && preview.key.solo_layer == key.solo_layer
                })
                .and_then(|preview| preview.patch);
            if let (PreviewTextures::Material(material), Some(set_id), true) =
                (textures, key.texture_set, caches_material(&key))
            {
                self.set_materials.insert(
                    set_id,
                    CachedMaterial {
                        revision: set_revision,
                        textures: Arc::clone(material),
                        patched: false,
                    },
                );
            }
            self.baked_preview = Some(BakedPreview {
                key,
                textures: textures.clone(),
                set_revision,
                patch: shown_patch,
            });
            return;
        }
        let job_patch = patch
            .as_ref()
            .map(|(base, regions)| (base.keys(), regions.clone()));
        // The quick mask tints the last material bake, which painting it
        // leaves unchanged.
        let quick_mask_base = key.texture_set.and_then(|set_id| {
            self.set_materials
                .get(&set_id)
                .map(|cached| Arc::clone(&cached.textures))
        });
        let quick_mask_overlay = (
            self.preferences.values.overlay_palette,
            self.preferences.values.overlay_pattern,
        );
        let (sender, receiver) = sync_channel(1);
        let spawned = std::thread::Builder::new()
            .name("preview-bake".to_owned())
            .spawn(move || {
                let patched = patch.and_then(|(base, regions)| {
                    inputs
                        .patch_preview(&texture_set, key.target, &base, &regions)
                        .ok()
                });
                let was_patched = patched.is_some();
                let result = match (patched, key.target) {
                    // A patch that fails falls back to a full bake below.
                    (Some(textures), _) => Ok(textures),
                    (_, PreviewTarget::Channel(channel)) => inputs
                        .bake_channel(&texture_set, channel)
                        .map(|texture| PreviewTextures::Single(Arc::new(texture))),
                    (_, PreviewTarget::LayerMask(layer)) => inputs
                        .bake_layer_mask(&texture_set, layer, false)
                        .map(|texture| PreviewTextures::Single(Arc::new(texture))),
                    (_, PreviewTarget::Generator(layer)) => inputs
                        .bake_layer_mask(&texture_set, layer, true)
                        .map(|texture| PreviewTextures::Single(Arc::new(texture))),
                    (
                        _,
                        PreviewTarget::MaskOverlay {
                            layer,
                            opacity,
                            palette,
                            pattern,
                        },
                    ) => inputs.bake_material(&texture_set).and_then(|mut textures| {
                        let mask = inputs.bake_layer_mask(&texture_set, layer, false)?;
                        textures.base_color = substancecraft_core::mask_overlay(
                            &textures.base_color,
                            &mask,
                            palette,
                            f32::from(opacity) / 100.0,
                            pattern,
                        )
                        .ok_or("mask and material sizes differ")?;
                        Ok(PreviewTextures::Material(Arc::new(textures)))
                    }),
                    (_, PreviewTarget::QuickMask(id)) => {
                        let base = match quick_mask_base {
                            Some(base) => Ok(base),
                            None => inputs.bake_material(&texture_set).map(Arc::new),
                        };
                        base.and_then(|base| {
                            let mask = inputs.bake_stencil(&texture_set, id)?;

                            let mut textures = (*base).clone();
                            let (palette, pattern) = quick_mask_overlay;
                            textures.base_color = substancecraft_core::mask_overlay(
                                &textures.base_color,
                                &mask,
                                palette,
                                0.55,
                                pattern,
                            )
                            .ok_or("mask and material sizes differ")?;
                            Ok(PreviewTextures::Material(Arc::new(textures)))
                        })
                    }

                    _ => inputs
                        .bake_material(&texture_set)
                        .map(|textures| PreviewTextures::Material(Arc::new(textures))),
                };
                let _ = sender.send((result, was_patched));
            });
        match spawned {
            Ok(_) => {
                self.preview_job = Some(PreviewJob {
                    receiver,
                    key,
                    set_revision,
                    patch: job_patch,
                });
            }
            Err(error) => {
                self.preview_failed = true;
                self.bake_error = Some(error.to_string());
            }
        }
    }

    /// Everything a background bake of texture set `set_id` reads, shared
    /// cheaply with the worker thread.
    fn bake_inputs(&self, set_id: TextureSetId) -> BakeInputs {
        BakeInputs {
            mesh: self.mesh_geometry.clone(),
            mesh_maps: self.mesh_maps.clone().filter(|_| self.mesh_maps_fresh()),
            slot: self.set_slot(set_id) as usize,
            resources: self.resource_images.clone(),
            udim: self.udim_atlas.clone().map(|atlas| udim_view::UdimInputs {
                atlas,
                cache: Arc::clone(&self.udim_cache),
                maps_key: udim_view::key_hash(&self.bake_key()),
                mesh_revision: self.mesh_revision,
                settings: self.effective_mesh_map_settings(),
            }),
        }
    }

    /// Longest texture-set edge the viewport bakes: smaller per tile when a
    /// UDIM atlas packs several tiles into one texture.
    fn preview_edge(&self) -> u32 {
        self.udim_atlas.as_ref().map_or(PREVIEW_MAX_EDGE, |atlas| {
            atlas.tile_edge_limit(PREVIEW_MAX_EDGE)
        })
    }

    fn poll_preview_job(&mut self, context: &egui::Context) {
        let Some(job) = &self.preview_job else {
            return;
        };
        match job.receiver.try_recv() {
            Ok((result, was_patched)) => {
                let (key, set_revision) = (job.key, job.set_revision);
                let job_patch = self.preview_job.take().and_then(|job| job.patch);
                match result {
                    Ok(textures) => {
                        let patch = job_patch
                            .filter(|_| was_patched)
                            .map(|(base, patches)| PreviewPatch::new(&textures, base, &patches));
                        // Tinted, soloed, and single-channel previews are
                        // not the set's material.
                        if let (PreviewTextures::Material(material), Some(set_id), true) =
                            (&textures, key.texture_set, caches_material(&key))
                        {
                            self.set_materials.insert(
                                set_id,
                                CachedMaterial {
                                    revision: set_revision,
                                    textures: Arc::clone(material),
                                    patched: was_patched,
                                },
                            );
                        }
                        self.baked_preview = Some(BakedPreview {
                            key,
                            textures,
                            set_revision,
                            patch,
                        });
                        self.bake_error = None;
                    }
                    // Only report failures of the preview still wanted.
                    Err(error) if key == self.preview_key() => {
                        self.preview_failed = true;
                        self.bake_error = Some(error);
                    }
                    Err(_) => {}
                }
                // A stale result is shown meanwhile; queue the current one.
                self.refresh_baked_preview();
                context.request_repaint();
            }
            Err(TryRecvError::Empty) => context.request_repaint_after(Duration::from_millis(16)),
            Err(TryRecvError::Disconnected) => self.preview_job = None,
        }
    }

    /// On multi-material meshes, rebakes (one at a time, after the selected
    /// preview) texture sets whose cached material is out of date, so every
    /// slot shows its own textures.
    fn refresh_set_caches(&mut self) {
        if self.set_bake_job.is_some() || self.preview_job.is_some() {
            return;
        }
        let slot_count = self
            .mesh_geometry
            .as_deref()
            .map_or(1, MeshGeometry::material_slot_count);
        if slot_count <= 1 {
            return;
        }
        let stale = (0..slot_count as u32)
            .filter_map(|slot| self.slot_texture_set(slot))
            .filter(|set_id| Some(*set_id) != self.selected_texture_set)
            .find(|set_id| {
                let revision = self.set_revision(*set_id);
                self.set_materials
                    .get(set_id)
                    .is_none_or(|cached| cached.revision != revision)
                    && self.set_bake_failed.get(set_id) != Some(&revision)
                    && self.cache.evicted.get(set_id) != Some(&revision)
            });
        let Some(set_id) = stale else {
            return;
        };
        let Some(texture_set) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == set_id)
            .cloned()
        else {
            return;
        };
        if texture_set.requires_mesh_maps() && !self.mesh_maps_fresh() {
            self.ensure_mesh_maps();
            return;
        }
        let texture_set = texture_set
            .downscaled(self.preview_edge())
            .unwrap_or(texture_set);
        let revision = self.set_revision(set_id);
        let inputs = self.bake_inputs(set_id);
        // Only the texels and textures edited since the cached bake.
        let patch = self.cached_material_patch(&texture_set);
        self.track_set_edits(&texture_set, revision);
        if let Some((textures, regions)) = &patch
            && regions.is_empty()
        {
            // Nothing the material shows changed.
            self.set_materials.insert(
                set_id,
                CachedMaterial {
                    revision,
                    textures: Arc::clone(textures),
                    patched: false,
                },
            );
            return;
        }
        let (sender, receiver) = sync_channel(1);
        let spawned = std::thread::Builder::new()
            .name("set-bake".to_owned())
            .spawn(move || {
                let patched = patch.and_then(|(base, regions)| {
                    inputs.patch_material(&texture_set, &base, &regions).ok()
                });
                let was_patched = patched.is_some();
                // A patch that fails falls back to a full bake.
                let result = match patched {
                    Some(textures) => Ok(textures),
                    None => inputs.bake_material(&texture_set),
                };
                let _ = sender.send((result, was_patched));
            });
        if spawned.is_ok() {
            self.set_bake_job = Some(SetBakeJob {
                set_id,
                revision,
                receiver,
            });
        }
    }

    fn poll_set_bake_job(&mut self, context: &egui::Context) {
        let Some(job) = &self.set_bake_job else {
            return;
        };
        match job.receiver.try_recv() {
            Ok((result, patched)) => {
                let (set_id, revision) = (job.set_id, job.revision);
                self.set_bake_job = None;
                match result {
                    Ok(textures) => {
                        self.set_materials.insert(
                            set_id,
                            CachedMaterial {
                                revision,
                                textures: Arc::new(textures),
                                patched,
                            },
                        );
                    }
                    Err(_) => {
                        self.set_bake_failed.insert(set_id, revision);
                    }
                }
                context.request_repaint();
            }
            Err(TryRecvError::Empty) => context.request_repaint_after(Duration::from_millis(32)),
            Err(TryRecvError::Disconnected) => self.set_bake_job = None,
        }
    }

    /// The finished preview matching what the viewport should show, possibly
    /// one revision behind while a newer bake is in flight.
    fn displayed_preview(&self) -> Option<&PreviewTextures> {
        let key = self.preview_key();
        self.baked_preview
            .as_ref()
            .filter(|preview| {
                preview.key.target == key.target && preview.key.texture_set == key.texture_set
            })
            .map(|preview| &preview.textures)
    }

    /// Returns a clone of the selected texture set when it can be baked now,
    /// starting a mesh-map bake (and reporting it) when maps are missing.
    /// `reads_maps` is true when the export itself samples mesh maps.
    fn exportable_texture_set(&mut self, reads_maps: bool) -> Option<TextureSet> {
        let texture_set = self.selected_texture_set_ref()?.clone();
        if (reads_maps || texture_set.requires_mesh_maps()) && !self.mesh_maps_fresh() {
            self.ensure_mesh_maps();
            self.bake_error =
                Some("Mesh maps are still baking; export again when they finish.".to_owned());
            return None;
        }
        Some(texture_set)
    }

    /// Asks where to save one texture, offering the chosen export format
    /// first.
    fn texture_save_path(&self, stem: &str) -> Option<std::path::PathBuf> {
        let mut dialog = rfd::FileDialog::new();
        let chosen = self.selected_format();
        let formats = std::iter::once(chosen).chain(
            TextureFormat::ALL
                .into_iter()
                .filter(|format| *format != chosen),
        );
        for format in formats {
            dialog = dialog.add_filter(format!("{} texture", format.name()), &[format.extension()]);
        }
        dialog
            .set_file_name(format!("{stem}.{}", chosen.extension()).replace(' ', "_"))
            .save_file()
    }

    fn export_baked_map(&mut self, channel: MaterialChannel) {
        let Some(texture_set) = self.exportable_texture_set(false) else {
            return;
        };
        let label = texture_set.user_channel(channel).map_or_else(
            || channel_label(channel).to_owned(),
            |user| user.name.clone(),
        );
        let stem = format!("{}_{label}", texture_set.name);

        let Some(path) = self.texture_save_path(&stem) else {
            return;
        };
        let result = self
            .bake_inputs(texture_set.id)
            .bake_channel(&texture_set, channel)
            .and_then(|texture| write_texture(&path, &texture).map_err(|error| error.to_string()));
        match result {
            Ok(()) => self.bake_error = None,
            Err(error) => self.bake_error = Some(error),
        }
    }

    /// File format a texture set exports in: its saved choice, else the
    /// app-wide one.
    fn format_for(&self, texture_set: &TextureSet) -> TextureFormat {
        texture_set
            .export
            .as_ref()
            .and_then(|export| TextureFormat::from_extension(&export.format))
            .unwrap_or(self.export_format)
    }

    fn preset_name_for(&self, texture_set: &TextureSet) -> String {
        texture_set.export.as_ref().map_or_else(
            || self.export_preset.clone(),
            |export| export.preset.clone(),
        )
    }

    fn selected_format(&self) -> TextureFormat {
        self.selected_texture_set_ref()
            .map_or(self.export_format, |set| self.format_for(set))
    }

    fn selected_preset_name(&self) -> String {
        self.selected_texture_set_ref().map_or_else(
            || self.export_preset.clone(),
            |set| self.preset_name_for(set),
        )
    }

    /// True when the selected set keeps its own export choice.
    fn selected_set_remembers_export(&self) -> bool {
        self.selected_texture_set_ref()
            .is_some_and(|set| set.export.is_some())
    }

    /// Applies `edit` to texture set `id`'s own properties as one undoable
    /// edit; returns false when nothing changed.
    fn edit_set_properties(
        &mut self,
        id: TextureSetId,
        edit: impl FnOnce(&mut substancecraft_core::TextureSetProperties),
    ) -> bool {
        self.edit_set_properties_with(id, edit, false)
    }

    /// [`Self::edit_set_properties`] that joins the previous undo step when
    /// it continues it, as successive frames of a slider drag do.
    fn edit_set_properties_merging(
        &mut self,
        id: TextureSetId,
        edit: impl FnOnce(&mut substancecraft_core::TextureSetProperties),
    ) -> bool {
        self.edit_set_properties_with(id, edit, true)
    }

    fn edit_set_properties_with(
        &mut self,
        id: TextureSetId,
        edit: impl FnOnce(&mut substancecraft_core::TextureSetProperties),
        merge: bool,
    ) -> bool {
        let Some(set) = self.project.texture_sets.iter().find(|set| set.id == id) else {
            return false;
        };
        let before = set.properties();
        let mut after = before.clone();
        edit(&mut after);
        if after == before {
            return false;
        }
        let looks_changed = after.resolution != before.resolution
            || after.height_strength != before.height_strength
            || after.channels != before.channels;
        let command = EditCommand::TextureSetProperties {
            texture_set_id: id,
            before: Box::new(before),
            after: Box::new(after),
        };
        let applied = if merge {
            self.history.apply_merging(&mut self.project, command)
        } else {
            self.history.apply(&mut self.project, command)
        };
        if applied && looks_changed {
            self.invalidate_sets(&BTreeSet::from([id]));
        }
        applied
    }

    /// Changes the selected set's saved format when it has one, else the
    /// app-wide format.
    fn set_format_choice(&mut self, format: TextureFormat) {
        match self
            .selected_texture_set
            .filter(|_| self.selected_set_remembers_export())
        {
            Some(id) => {
                self.edit_set_properties(id, |properties| {
                    if let Some(export) = &mut properties.export {
                        export.format = format.extension().to_owned();
                    }
                });
            }
            None => self.export_format = format,
        }
    }

    fn set_preset_choice(&mut self, name: String) {
        match self
            .selected_texture_set
            .filter(|_| self.selected_set_remembers_export())
        {
            Some(id) => {
                self.edit_set_properties(id, |properties| {
                    if let Some(export) = &mut properties.export {
                        export.preset = name;
                    }
                });
            }
            None => self.export_preset = name,
        }
    }

    /// Whether the selected set keeps its own export choice; turning it on
    /// captures the current one.
    fn remember_export_for_selected(&mut self, remember: bool) {
        let preset = self.selected_preset_name();
        let format = self.selected_format();
        let Some(id) = self.selected_texture_set else {
            return;
        };
        self.edit_set_properties(id, |properties| {
            let udim = properties.export.as_ref().is_some_and(|export| export.udim);
            properties.export = remember.then(|| substancecraft_core::TextureSetExport {
                preset,
                format: format.extension().to_owned(),
                udim,
            });
        });
    }

    /// Exports every texture set into `folder` with its own preset, format,
    /// and UDIM choice. Stops at the first set preflight blocks.
    fn export_all_sets_into(
        &self,
        folder: &std::path::Path,
    ) -> Result<Vec<std::path::PathBuf>, String> {
        let presets = self.export_presets();
        let mut written = Vec::new();
        for texture_set in &self.project.texture_sets {
            let name = self.preset_name_for(texture_set);
            let preset = presets
                .iter()
                .find(|preset| preset.name == name)
                .ok_or_else(|| format!("{}: unknown preset `{name}`", texture_set.name))?;
            if let Some(error) = blocking_export_error(&self.export_issues(texture_set, preset)) {
                return Err(format!("{}: {error}", texture_set.name));
            }
            let udim = texture_set
                .export
                .as_ref()
                .is_some_and(|export| export.udim);
            let files = if udim {
                self.write_udim_tiles(texture_set, preset, folder)
            } else {
                self.write_export(texture_set, preset, folder)
            }
            .map_err(|error| format!("{}: {error}", texture_set.name))?;
            written.extend(files);
        }
        Ok(written)
    }

    fn export_all_sets(&mut self) {
        if self
            .project
            .texture_sets
            .iter()
            .any(TextureSet::requires_mesh_maps)
            && !self.mesh_maps_fresh()
        {
            self.ensure_mesh_maps();
            self.bake_error =
                Some("Mesh maps are still baking; export again when they finish.".to_owned());
            return;
        }
        let Some(folder) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        self.bake_error = self.export_all_sets_into(&folder).err();
    }

    /// Built-in presets followed by the project's custom ones.
    fn export_presets(&self) -> Vec<ExportPreset> {
        let mut presets = ExportPreset::built_in();
        presets.extend(self.project.export_presets.iter().cloned());
        presets
    }

    fn is_custom_preset(&self, name: &str) -> bool {
        self.project
            .export_presets
            .iter()
            .any(|preset| preset.name == name)
    }

    /// The preset the Export button uses, falling back to the default when
    /// the chosen one was deleted or the project changed.
    fn selected_export_preset(&self) -> ExportPreset {
        let presets = self.export_presets();
        presets
            .iter()
            .find(|preset| preset.name == self.selected_preset_name())
            .or_else(|| {
                presets
                    .iter()
                    .find(|preset| preset.name == DEFAULT_EXPORT_PRESET)
            })
            .cloned()
            .unwrap_or_else(|| presets[0].clone())
    }

    /// Writes every file of the selected export preset into a chosen folder
    /// in the export format.
    fn export_with_preset(&mut self) {
        let preset = self.selected_export_preset();
        self.export_with_selected_or_named_preset(preset);
    }

    /// Selects a built-in engine preset before following the regular export
    /// preflight and overwrite-confirmation flow.
    fn export_engine_preset(&mut self, name: &str) {
        let Some(preset) = self.select_engine_preset(name) else {
            return;
        };
        self.export_with_selected_or_named_preset(preset);
    }

    fn select_engine_preset(&mut self, name: &str) -> Option<ExportPreset> {
        let Some(preset) = self
            .export_presets()
            .into_iter()
            .find(|preset| preset.name == name)
        else {
            self.bake_error = Some(format!("Export preset `{name}` is unavailable."));
            return None;
        };
        self.set_preset_choice(preset.name.clone());
        Some(preset)
    }

    fn export_with_selected_or_named_preset(&mut self, preset: ExportPreset) {
        let Some(texture_set) = self.exportable_texture_set(preset.reads_mesh_maps()) else {
            return;
        };
        if let Some(error) = blocking_export_error(&self.export_issues(&texture_set, &preset)) {
            self.bake_error = Some(error);
            return;
        }
        let Some(folder) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        self.begin_export(texture_set, preset, folder);
    }

    /// Preflight findings for exporting `texture_set` with `preset` now.
    fn export_issues(
        &self,
        texture_set: &TextureSet,
        preset: &ExportPreset,
    ) -> Vec<PreflightIssue> {
        let inputs = self.bake_inputs(texture_set.id);
        export_preflight(
            texture_set,
            preset,
            inputs.mesh.as_deref(),
            inputs.maps(),
            Some(&inputs.resources),
        )
    }

    /// Writes the export, first asking for confirmation when it would
    /// replace files already in `folder`.
    fn begin_export(
        &mut self,
        texture_set: TextureSet,
        preset: ExportPreset,
        folder: std::path::PathBuf,
    ) {
        let existing = existing_export_files(
            &folder,
            &texture_set,
            &preset,
            self.format_for(&texture_set),
        );
        if existing.is_empty() {
            self.finish_export(&texture_set, &preset, &folder);
        } else {
            self.overwrite_prompt = Some(OverwritePrompt {
                texture_set,
                preset,
                folder,
                existing,
            });
        }
    }

    /// Answers the overwrite prompt: writes the export when confirmed.
    fn resolve_overwrite_prompt(&mut self, overwrite: bool) {
        if let Some(prompt) = self.overwrite_prompt.take()
            && overwrite
        {
            self.finish_export(&prompt.texture_set, &prompt.preset, &prompt.folder);
        }
    }

    fn finish_export(
        &mut self,
        texture_set: &TextureSet,
        preset: &ExportPreset,
        folder: &std::path::Path,
    ) {
        match self.write_export(texture_set, preset, folder) {
            Ok(_) => self.bake_error = None,
            Err(error) => self.bake_error = Some(error),
        }
    }

    /// Bakes and writes a preset's files; returns the paths written.
    fn write_export(
        &self,
        texture_set: &TextureSet,
        preset: &ExportPreset,
        folder: &std::path::Path,
    ) -> Result<Vec<std::path::PathBuf>, String> {
        let format = self.format_for(texture_set);
        let files = self
            .bake_inputs(texture_set.id)
            .bake_export(texture_set, &preset.resolved(format.extension()))?;
        write_exported_textures(folder, &files, format).map_err(|error| error.to_string())
    }

    /// Writes `texture_set`'s OpenPBR textures and a MaterialX document
    /// binding them into `folder`; returns the document path.
    fn write_openpbr(
        &self,
        texture_set: &TextureSet,
        folder: &std::path::Path,
    ) -> Result<std::path::PathBuf, String> {
        let preset = substancecraft_api::openpbr_preset();
        let written = self.write_export(texture_set, &preset, folder)?;
        let mut files = BTreeMap::new();
        for ((input, _, _, _), path) in substancecraft_api::OPENPBR_INPUTS.iter().zip(&written) {
            if let Some(name) = path.file_name() {
                files.insert((*input).to_owned(), name.to_string_lossy().into_owned());
            }
        }
        let document = folder.join(format!("{}.mtlx", texture_set.name.replace(' ', "_")));
        substancecraft_api::write_openpbr_materialx(&document, &texture_set.name, &files)
            .map_err(|error| error.to_string())?;
        Ok(document)
    }

    /// Writes the mesh as glTF with every texture set's metal/rough
    /// textures beside it; returns the glTF path.
    fn write_gltf_scene(&self, path: &std::path::Path) -> Result<std::path::PathBuf, String> {
        let mesh = self.mesh_geometry.as_deref().ok_or("Load a mesh first.")?;
        let folder = path.parent().ok_or("Choose a file inside a folder.")?;
        let preset = ExportPreset::built_in()
            .into_iter()
            .find(|preset| preset.name == DEFAULT_EXPORT_PRESET)
            .ok_or("The metal/rough preset is missing.")?;
        let slot_count = mesh.material_slot_count().max(1);
        let mut materials = Vec::with_capacity(slot_count);
        for slot in 0..slot_count as u32 {
            let Some(texture_set) = self
                .slot_texture_set(slot)
                .and_then(|id| self.project.texture_sets.iter().find(|set| set.id == id))
            else {
                materials.push(substancecraft_api::GltfMaterialFiles::default());
                continue;
            };
            // glTF images are PNG.
            let files = self
                .bake_inputs(texture_set.id)
                .bake_export(texture_set, &preset.resolved("png"))?;
            let blend = files.first().is_some_and(|file| {
                file.texture
                    .rgba8
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|texel| texel[3] < 255)
            });
            let written = write_exported_textures(folder, &files, TextureFormat::Png)
                .map_err(|error| error.to_string())?;
            let name_of = |suffix: &str| {
                written
                    .iter()
                    .filter_map(|path| path.file_name())
                    .map(|name| name.to_string_lossy().into_owned())
                    .find(|name| name.contains(suffix))
            };
            materials.push(substancecraft_api::GltfMaterialFiles {
                name: texture_set.name.clone(),
                base_color: name_of("_BaseColor"),
                orm: name_of("_ORM"),
                normal: name_of("_Normal"),
                emissive: name_of("_Emissive"),
                blend,
            });
        }
        substancecraft_api::write_gltf(path, mesh, self.mesh_smoothing.angle_degrees, &materials)
            .map_err(|error| error.to_string())?;
        Ok(path.to_path_buf())
    }

    /// Exports `preset` once per UDIM tile the set's mesh slot uses, baking
    /// mesh maps per tile when the export needs them.
    fn write_udim_tiles(
        &self,
        texture_set: &TextureSet,
        preset: &ExportPreset,
        folder: &std::path::Path,
    ) -> Result<Vec<std::path::PathBuf>, String> {
        let mesh = self.mesh_geometry.as_deref().ok_or("Load a mesh first.")?;
        let slot = self.set_slot(texture_set.id);
        let geometry = if mesh.material_slot_count() > 1 {
            mesh.submesh(slot)
        } else {
            mesh.clone()
        };
        let tiles = geometry.udim_tiles();
        if tiles.is_empty() {
            return Err("The mesh has no UVs in the UDIM range.".to_owned());
        }
        let needs_maps = texture_set.requires_mesh_maps() || preset.reads_mesh_maps();
        let resources = self.resource_images.clone();
        let format = self.format_for(texture_set);
        let preset = &preset.resolved(format.extension());
        let mut written = Vec::new();
        for tile in tiles {
            let shifted = geometry.shifted_uvs(tile.offset());
            let maps = needs_maps
                .then(|| {
                    substancecraft_api::bake_mesh_maps(
                        &shifted,
                        &self.mesh_smoothing,
                        &self.effective_mesh_map_settings(),
                    )
                })
                .transpose()
                .map_err(|error| error.to_string())?;
            let files = texture_set
                .bake_export_tile(
                    preset,
                    tile,
                    substancecraft_core::BakeContext {
                        mesh: Some(&shifted),
                        mesh_maps: maps.as_ref(),
                        resources: Some(&resources),
                    },
                )
                .map_err(|error| error.to_string())?;
            written.extend(
                write_exported_textures(folder, &files, format)
                    .map_err(|error| error.to_string())?,
            );
        }
        Ok(written)
    }

    /// Adds a fill layer from the OpenPBR material in MaterialX document
    /// `document`: each texture it reads becomes a bitmap resource tagged
    /// with its colour space, and constant inputs become fill values.
    /// Returns the new layer.
    fn import_openpbr_from(&mut self, document: &std::path::Path) -> Result<LayerId, String> {
        let material = substancecraft_api::read_openpbr_materialx(document)
            .map_err(|error| format!("{}: {error}", document.display()))?;
        let folder = document.parent().unwrap_or(std::path::Path::new(""));
        let mut resources = BTreeMap::new();
        let resources_before = self.project.resources.len();
        for (input, _, _, _) in substancecraft_api::OPENPBR_INPUTS {
            let Some(file) = material.textures.get(input) else {
                continue;
            };
            let path = folder.join(file);
            let id = match self
                .project
                .resources
                .iter()
                .find(|resource| std::path::Path::new(&resource.source) == path)
            {
                Some(existing) => existing.id,
                None => match self.add_resource(&path) {
                    Ok(id) => id,
                    Err(error) => {
                        // Leave no bitmaps behind from a failed import.
                        for added in self.project.resources.split_off(resources_before) {
                            self.resource_images.remove(&added.id);
                            self.resource_watcher.forget(added.id);
                        }
                        return Err(error);
                    }
                },
            };
            let space = substancecraft_api::openpbr_texture_color_space(&material, input);
            if let Some(resource) = self
                .project
                .resources
                .iter_mut()
                .find(|resource| resource.id == id)
            {
                resource.color_space = space;
            }
            resources.insert(input.to_owned(), id);
        }
        let (fill, channels) = substancecraft_api::openpbr_fill(&material, &resources);
        let id = self.allocate_layer_id().ok_or("Out of layer ids.")?;
        let mut layer = MaterialLayer::new_fill(id, material.name.replace('_', " "), fill);
        layer.channels = channels;
        self.insert_layer(layer);
        if self.selected_layer != Some(id) {
            return Err("The OpenPBR layer could not be inserted.".to_owned());
        }
        Ok(id)
    }

    fn import_openpbr(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("MaterialX document", &["mtlx"])
            .pick_file()
        else {
            return;
        };
        if let Err(error) = self.import_openpbr_from(&path) {
            self.import_error = Some(error);
        }
    }

    fn export_openpbr(&mut self) {
        let Some(texture_set) = self.exportable_texture_set(false) else {
            return;
        };
        let Some(folder) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        self.bake_error = self.write_openpbr(&texture_set, &folder).err();
    }

    fn export_gltf(&mut self) {
        if self
            .project
            .texture_sets
            .iter()
            .any(TextureSet::requires_mesh_maps)
            && !self.mesh_maps_fresh()
        {
            self.ensure_mesh_maps();
            self.bake_error =
                Some("Mesh maps are still baking; export again when they finish.".to_owned());
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("glTF 2.0", &["gltf"])
            .set_file_name("scene.gltf")
            .save_file()
        else {
            return;
        };
        self.bake_error = self.write_gltf_scene(&path).err();
    }

    fn export_udim_tiles(&mut self) {
        let preset = self.selected_export_preset();
        let Some(texture_set) = self.selected_texture_set_ref().cloned() else {
            return;
        };
        let Some(folder) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        self.bake_error = self.write_udim_tiles(&texture_set, &preset, &folder).err();
    }

    /// Starts editing a copy of `name` as a new custom preset.
    fn duplicate_export_preset(&mut self, name: &str) {
        let Some(mut preset) = self
            .export_presets()
            .into_iter()
            .find(|preset| preset.name == name)
        else {
            return;
        };
        let taken = self.export_presets();
        preset.name = (1..)
            .map(|index| match index {
                1 => format!("{name} copy"),
                index => format!("{name} copy {index}"),
            })
            .find(|candidate| {
                !taken
                    .iter()
                    .any(|preset| preset.name.eq_ignore_ascii_case(candidate))
            })
            .unwrap_or_default();
        self.preset_draft = Some(PresetDraft {
            original: None,
            preset,
        });
    }

    fn edit_export_preset(&mut self, name: &str) {
        self.preset_draft = self
            .project
            .export_presets
            .iter()
            .find(|preset| preset.name == name)
            .map(|preset| PresetDraft {
                original: Some(preset.name.clone()),
                preset: preset.clone(),
            });
    }

    /// Validates the draft against the project and stores it, selecting it
    /// for export. The draft stays open on error.
    fn apply_preset_draft(&mut self) -> Result<(), String> {
        let Some(draft) = self.preset_draft.clone() else {
            return Ok(());
        };
        let before = self.project.export_presets.clone();
        let mut after = before.clone();
        match draft
            .original
            .as_ref()
            .and_then(|original| after.iter().position(|preset| &preset.name == original))
        {
            Some(index) => after[index] = draft.preset.clone(),
            None => after.push(draft.preset.clone()),
        }
        let mut project = self.project.clone();
        project.export_presets = after.clone();
        project.validate().map_err(|error| error.to_string())?;
        self.history.apply(
            &mut self.project,
            EditCommand::ExportPresets { before, after },
        );
        self.set_preset_choice(draft.preset.name);
        self.preset_draft = None;
        Ok(())
    }

    /// Deletes a custom preset (undoably); texture sets that remembered it
    /// go back to the app-wide choice.
    fn delete_export_preset(&mut self, name: &str) {
        let before = self.project.export_presets.clone();
        let after: Vec<ExportPreset> = before
            .iter()
            .filter(|preset| preset.name != name)
            .cloned()
            .collect();
        let mut commands = vec![EditCommand::ExportPresets { before, after }];
        for set in &self.project.texture_sets {
            if set
                .export
                .as_ref()
                .is_some_and(|export| export.preset == name)
            {
                let before = set.properties();
                let after = substancecraft_core::TextureSetProperties {
                    export: None,
                    ..before.clone()
                };
                commands.push(EditCommand::TextureSetProperties {
                    texture_set_id: set.id,
                    before: Box::new(before),
                    after: Box::new(after),
                });
            }
        }
        self.history
            .apply(&mut self.project, EditCommand::Batch(commands));
        if self
            .preset_draft
            .as_ref()
            .is_some_and(|draft| draft.original.as_deref() == Some(name))
        {
            self.preset_draft = None;
        }
    }

    fn export_mesh_map(&mut self, kind: MeshMapKind) {
        let Some(texture) = self.fresh_mesh_maps().map(|maps| maps.to_texture(kind)) else {
            self.ensure_mesh_maps();
            return;
        };
        let Some(path) = self.texture_save_path(kind.name()) else {
            return;
        };
        if let Err(error) = write_texture(&path, &texture) {
            self.mesh_map_error = Some(error.to_string());
        }
    }

    /// Loads the pixels of every project resource, reporting unreadable files.
    fn reload_resources(&mut self) {
        let (images, failures) = load_project_resources(&self.project);
        self.loaded_resource_sources = self
            .project
            .resources
            .iter()
            .map(|resource| (resource.id, resource.active_source()))
            .collect();
        self.resource_images = images;
        self.resource_thumbnails.clear();
        self.resource_watcher.reset();
        for resource in &self.project.resources {
            self.resource_watcher
                .remember(resource.id, std::path::Path::new(&resource.active_source()));
        }
        if let Some((id, error)) = failures.first() {
            let name = self
                .project
                .resources
                .iter()
                .find(|resource| resource.id == *id)
                .map_or("bitmap", |resource| resource.name.as_str());
            let more = if failures.len() > 1 {
                format!(" (and {} more)", failures.len() - 1)
            } else {
                String::new()
            };
            self.project_error = Some(format!("Resource `{name}`: {error}{more}"));
        }
        self.invalidate_all_sets();
    }

    fn import_resources(&mut self) {
        let Some(paths) = rfd::FileDialog::new()
            .add_filter(
                "Bitmap, SVG, or text (PNG, JPEG, TGA, BMP, TIFF, WebP, EXR, HDR, SVG, SCTEXT)",
                &IMAGE_EXTENSIONS,
            )
            .pick_files()
        else {
            return;
        };
        let mut pending = Vec::new();
        let mut failures = Vec::new();
        for path in paths {
            match self.add_resource(&path) {
                Ok(resource) => {
                    let name = self
                        .project
                        .resources
                        .iter()
                        .find(|candidate| candidate.id == resource)
                        .map_or("", |candidate| candidate.name.as_str());
                    let image = self.resource_images.get(&resource);
                    let (channel, reason) = suggest_import_channel(name, image.map(Arc::as_ref));
                    pending.push(ImportAssignment {
                        resource,
                        channel,
                        reason,
                    });
                }
                Err(error) => failures.push(error),
            }
        }
        if !failures.is_empty() {
            self.import_error = Some(failures.join("\n"));
        }
        if !pending.is_empty() {
            self.pending_import_assignments.extend(pending);
        }
    }

    fn confirm_import_assignments(&mut self) {
        let assignments: Vec<_> = self
            .pending_import_assignments
            .iter()
            .filter_map(|assignment| {
                assignment
                    .channel
                    .map(|channel| (assignment.resource, channel))
            })
            .collect();
        if assignments.is_empty() {
            self.pending_import_assignments.clear();
            return;
        }
        if self.selected_texture_set.is_none() {
            self.add_texture_set();
        }
        let Some(texture_set_id) = self.selected_texture_set else {
            self.import_error = Some("Could not create a texture set for imported bitmaps.".into());
            return;
        };
        let Some(set) = self
            .project
            .texture_sets
            .iter()
            .find(|set| set.id == texture_set_id)
        else {
            self.import_error = Some("The selected texture set no longer exists.".into());
            return;
        };
        let insertion = self.insertion_slot(set);
        let mut commands = Vec::with_capacity(assignments.len());
        let mut inserted = Vec::with_capacity(assignments.len());
        for (offset, (resource, channel)) in assignments.into_iter().enumerate() {
            let Some(resource_name) = self
                .project
                .resources
                .iter()
                .find(|candidate| candidate.id == resource)
                .map(|resource| resource.name.clone())
            else {
                self.import_error = Some(format!(
                    "Imported bitmap resource {} no longer exists.",
                    resource.0
                ));
                return;
            };
            let Some(layer_id) = self.allocate_layer_id() else {
                self.import_error = Some("Out of layer ids.".to_owned());
                return;
            };
            let mut fill = LayerFill::default();
            let Some(slot) = fill.textures.slot_mut(channel) else {
                self.import_error = Some(format!(
                    "{} cannot be assigned to a fill layer.",
                    channel_label(channel)
                ));
                return;
            };
            *slot = Some(resource);
            let mut layer = MaterialLayer::new_fill(
                layer_id,
                format!("{} ({})", resource_name, channel_label(channel)),
                fill,
            );
            layer.channels = BTreeSet::from([channel]);
            commands.push(EditCommand::InsertLayer {
                texture_set_id,
                slot: LayerSlot {
                    parent: insertion.parent,
                    index: insertion.index + offset,
                },
                layer: Box::new(layer),
            });
            inserted.push(layer_id);
        }
        if self
            .history
            .apply(&mut self.project, EditCommand::Batch(commands))
        {
            self.selected_layer = inserted.last().copied();
            self.layer_settled = None;
            self.pending_import_assignments.clear();
            self.import_error = None;
            self.invalidate_preview();
        } else {
            self.import_error =
                Some("Could not assign the imported bitmaps to fill layers.".to_owned());
        }
    }

    /// Writes a starter `.sctext` file where the user chooses and imports
    /// it; its text is edited in the Assets drawer.
    fn new_text_resource(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("SubstanceCraft text", &["sctext"])
            .set_file_name("label.sctext")
            .save_file()
        else {
            return;
        };
        if let Err(error) = self.create_text_resource(&path, "Text", 128.0) {
            self.import_error = Some(error);
        }
    }

    fn create_text_resource(
        &mut self,
        path: &std::path::Path,
        text: &str,
        size_px: f32,
    ) -> Result<ResourceId, String> {
        let spec = substancecraft_api::TextSpec {
            text: text.to_owned(),
            size_px,
            font: None,
        };
        substancecraft_api::save_text_resource(path, &spec).map_err(|error| error.to_string())?;
        self.add_resource(path)
    }

    /// Loads a bitmap and lists it as a project resource.
    fn add_resource(&mut self, path: &std::path::Path) -> Result<ResourceId, String> {
        let image =
            load_texture_image(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let id = ResourceId(
            self.project
                .resources
                .iter()
                .map(|resource| resource.id.0)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        );
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("Bitmap {}", id.0));
        self.project.resources.push(ProjectResource {
            content_hash: substancecraft_api::file_content_hash(path).ok(),
            color_space: image.color_space,
            ..ProjectResource::new(id, name, path.to_string_lossy())
        });
        self.resource_images.insert(id, Arc::new(image));
        self.resource_watcher.remember(id, path);
        Ok(id)
    }

    /// Checks bitmap files for outside edits about once a second and reloads
    /// (or, in Ask mode, queues) the ones that changed.
    fn poll_resource_files(&mut self, context: &egui::Context) {
        use resource_watch::{ReloadMode, SCAN_INTERVAL};

        if (self.project.resources.is_empty() && !self.resource_watcher.watches_high_poly())
            || self.resource_watcher.mode == ReloadMode::Off
        {
            return;
        }
        // Keep polling while the window is idle.
        context.request_repaint_after(SCAN_INTERVAL);
        let now = std::time::Instant::now();
        if !self.resource_watcher.due(now) {
            return;
        }
        let changed = self.resource_watcher.scan(&self.project.resources, now);
        let high_poly = self.resource_watcher.scan_high_poly();
        match self.resource_watcher.mode {
            ReloadMode::Auto => {
                self.reload_changed_resources(&changed);
                if high_poly {
                    self.reload_high_poly();
                }
            }
            ReloadMode::Ask => {
                self.resource_watcher.pending.extend(changed);
                self.resource_watcher.high_poly_pending |= high_poly;
            }
            ReloadMode::Off => {}
        }
    }

    /// Re-reads the given bitmaps and rebakes only the texture sets using
    /// them. A file that cannot be decoded (missing, or caught mid-save)
    /// keeps its previous pixels and is reported.
    fn reload_changed_resources(&mut self, ids: &[ResourceId]) {
        let mut reloaded = BTreeSet::new();
        let mut failures = Vec::new();
        for id in ids {
            self.resource_watcher.pending.remove(id);
            let Some(resource) = self
                .project
                .resources
                .iter()
                .find(|resource| resource.id == *id)
            else {
                continue;
            };
            match load_resource_image(resource) {
                Ok(image) => {
                    self.resource_images.insert(*id, Arc::new(image));
                    self.resource_thumbnails.remove(id);
                    reloaded.insert(*id);
                }
                Err(error) => failures.push(format!("`{}`: {error}", resource.name)),
            }
        }
        if let Some(first) = failures.first() {
            let more = match failures.len() {
                1 => String::new(),
                count => format!(" (and {} more)", count - 1),
            };
            self.project_error = Some(format!(
                "Could not reload {first}{more}; keeping the previous pixels."
            ));
        }
        let affected = self
            .project
            .texture_sets
            .iter()
            .filter(|set| !set.referenced_resources().is_disjoint(&reloaded))
            .map(|set| set.id)
            .collect();
        if self.skew_map.is_some_and(|id| reloaded.contains(&id)) {
            self.skew_map_revision = self.skew_map_revision.wrapping_add(1);
        }
        self.invalidate_sets(&affected);
    }

    /// Names of resources waiting for a reload confirmation.
    fn pending_reload_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .project
            .resources
            .iter()
            .filter(|resource| self.resource_watcher.pending.contains(&resource.id))
            .map(|resource| resource.name.clone())
            .collect();
        if self.resource_watcher.high_poly_pending
            && let Some(high) = &self.high_poly
        {
            names.push(format!("High poly {}", path_label(&high.path)));
        }
        names
    }

    /// Reloads everything waiting for confirmation.
    fn reload_pending(&mut self) {
        let pending: Vec<ResourceId> = self.resource_watcher.pending.iter().copied().collect();
        self.reload_changed_resources(&pending);
        if self.resource_watcher.high_poly_pending {
            self.reload_high_poly();
        }
    }

    /// True when a layer or the current brush uses `id`.
    fn resource_in_use(&self, id: ResourceId) -> bool {
        self.brush_tip.resource() == Some(id)
            || self
                .project
                .texture_sets
                .iter()
                .any(|set| set.referenced_resources().contains(&id))
    }

    /// Removes an unused resource from the project.
    fn remove_resource(&mut self, id: ResourceId) {
        if self.resource_in_use(id) {
            return;
        }
        self.project.resources.retain(|resource| resource.id != id);
        self.resource_images.remove(&id);
        self.resource_thumbnails.remove(&id);
        self.resource_watcher.forget(id);
    }

    /// egui thumbnail of a resource, created on first use.
    fn resource_thumbnail(
        &mut self,
        context: &egui::Context,
        id: ResourceId,
    ) -> Option<egui::TextureHandle> {
        if let Some(texture) = self.resource_thumbnails.get(&id) {
            return Some(texture.clone());
        }
        let image = self.resource_images.get(&id)?;
        let thumbnail = context.load_texture(
            format!("resource-{}", id.0),
            uv_view::thumbnail_image(image, 96),
            egui::TextureOptions::LINEAR,
        );
        self.resource_thumbnails.insert(id, thumbnail.clone());
        Some(thumbnail)
    }

    fn selected_mesh_name(&self) -> Option<&str> {
        if let Some(index) = self.selected_imported_mesh {
            return self
                .imported_meshes
                .get(index)
                .map(|mesh| mesh.name.as_str());
        }
        self.selected_primitive.map(PrimitivePreset::name)
    }

    fn select_imported_mesh(&mut self, index: usize) {
        let Some(mesh) = self.imported_meshes.get(index) else {
            return;
        };
        let geometry = scaled_geometry(&mesh.geometry, self.import_scale);
        self.selected_imported_mesh = Some(index);
        self.selected_primitive = None;
        self.set_base_geometry(Some(geometry));
    }

    fn set_import_scale(&mut self, scale: f32) {
        let before = self.project.mesh.clone();
        self.import_scale = sanitize_mesh_scale(scale);
        if let Some(mesh_asset) = self.project.mesh.as_mut()
            && !mesh_asset.source.starts_with("builtin:")
        {
            mesh_asset.scale = self.import_scale;
        }
        if let Some(index) = self.selected_imported_mesh {
            self.select_imported_mesh(index);
        }
        self.record_mesh_edit(before, Vec::new());
    }

    fn update_pose_transition(&mut self, context: &egui::Context) {
        let Some(transition) = &self.pose_transition else {
            return;
        };
        let now = context.input(|input| input.time);
        let amount =
            ((now - transition.started_at) / POSE_TRANSITION_SECONDS).clamp(0.0, 1.0) as f32;
        match transition.from.interpolate(&transition.to, amount) {
            Ok(pose) => {
                // A saved animation frame shows (blending its time within
                // one clip) as the pose transitions.
                self.show_animation_frame(pose.animation.as_ref());
                self.current_pose = pose;
            }
            Err(_) => {
                self.pose_transition = None;
                return;
            }
        }
        if amount >= 1.0 {
            self.pose_transition = None;
        } else {
            context.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn select_pose(&mut self, pose_id: PoseId, context: &egui::Context) {
        let Some(target) = self
            .project
            .saved_poses
            .iter()
            .find(|pose| pose.id == pose_id)
            .cloned()
        else {
            return;
        };
        self.selected_pose = Some(pose_id);
        self.pose_transition = Some(PoseTransition {
            from: NamedPose {
                animation: self.animation_frame(),
                ..self.current_pose.clone()
            },
            to: target,
            started_at: context.input(|input| input.time),
        });
    }

    fn save_pose(&mut self) {
        if self.project.saved_poses.len() >= MAX_SAVED_POSES
            || self.current_pose.validate().is_err()
        {
            return;
        }
        let Some(next_id) = self
            .project
            .saved_poses
            .iter()
            .map(|pose| pose.id.0)
            .max()
            .unwrap_or(0)
            .checked_add(1)
        else {
            return;
        };
        let base_name = self.pose_name.trim();
        if base_name.is_empty() {
            return;
        }
        let mut name = base_name.to_owned();
        let mut suffix = 2_u32;
        while self
            .project
            .saved_poses
            .iter()
            .any(|pose| pose.name.eq_ignore_ascii_case(&name))
        {
            name = format!("{base_name} {suffix}");
            let Some(next_suffix) = suffix.checked_add(1) else {
                return;
            };
            suffix = next_suffix;
        }
        let mut pose = self.current_pose.clone();
        pose.id = PoseId(next_id);
        pose.animation = self.animation_frame();
        pose.name.clone_from(&name);
        self.project.saved_poses.push(pose);
        self.selected_pose = Some(PoseId(next_id));
        self.pose_name = format!("Pose {}", self.project.saved_poses.len() + 1);
    }

    fn view_projection(&self, viewport: egui::Rect) -> ViewProjection {
        let pose = self
            .current_pose
            .bones
            .get(&BoneId(0))
            .copied()
            .unwrap_or(BoneTransform::IDENTITY);
        let posed_center = pose.rotation.mul_vec3(self.mesh_center * pose.scale) + pose.translation;
        ViewProjection {
            pose,
            camera_rotation: self.viewport_camera.rotation(),
            posed_center,
            center: viewport.center(),
            scale: viewport.width().min(viewport.height()) * 0.42 * self.viewport_camera.zoom * 0.9,
            pan: self.viewport_camera.pan,
        }
    }

    fn update_viewport(&mut self, ui: &mut egui::Ui) {
        self.read_pen_pressure(ui.ctx());
        // Strokes end when every button is up, whichever view drew them.
        if self.active_stroke.is_some() && !ui.input(|input| input.pointer.any_down()) {
            self.finish_active_stroke();
            self.invalidate_preview();
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(theme::VIEWPORT_BG))
            .show(ui, |ui| {
                let area = ui.max_rect();
                let (model, uv) = match self.view_mode {
                    ViewMode::Model => (Some(area), None),
                    ViewMode::Uv => (None, Some(area)),
                    ViewMode::Split => {
                        let middle = area.left() + area.width() * self.layout.split;
                        (
                            Some(area.with_max_x(middle - 1.0)),
                            Some(area.with_min_x(middle + 1.0)),
                        )
                    }
                };
                // The split divider drags; layouts save where it sits.
                let divider = model
                    .filter(|_| uv.is_some())
                    .map(|model| model.right() + 1.0);
                if let Some(x) = divider {
                    let handle = egui::Rect::from_x_y_ranges(x - 4.0..=x + 4.0, area.y_range());
                    let response = ui
                        .interact(handle, egui::Id::new("split_divider"), egui::Sense::drag())
                        .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
                    if response.dragged()
                        && let Some(pointer) = response.interact_pointer_pos()
                    {
                        self.layout.split =
                            layouts::clamp_split((pointer.x - area.left()) / area.width());
                    }
                }
                // Runs during strokes too: previews bake in the background, so the
                // stroke in progress shows up live.
                self.refresh_baked_preview();
                if let Some(rect) = model {
                    self.model_viewport(ui, rect);
                }
                if let Some(rect) = uv {
                    self.uv_viewport(ui, rect);
                }
                if let Some(x) = divider {
                    ui.painter().vline(
                        x,
                        area.y_range(),
                        egui::Stroke::new(2.0, theme::VIEWPORT_BG.gamma_multiply(0.5)),
                    );
                }
            });
    }

    /// Tracks stylus pressure from touch events; mice report none.
    fn read_pen_pressure(&mut self, context: &egui::Context) {
        context.input(|input| {
            for event in &input.events {
                if let egui::Event::Touch { phase, force, .. } = event {
                    self.pen_pressure = match phase {
                        egui::TouchPhase::Start | egui::TouchPhase::Move => {
                            force.filter(|force| force.is_finite() && *force > 0.0)
                        }
                        egui::TouchPhase::End | egui::TouchPhase::Cancel => None,
                    }
                    .map(|force| force.clamp(0.0, 1.0));
                }
            }
        });
    }

    /// The 3D view: Filament (or the software fallback), picking, painting,
    /// and polygon selection.
    fn model_viewport(&mut self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let response = ui.interact(
            viewport,
            ui.id().with("model_viewport"),
            egui::Sense::click_and_drag(),
        );
        let pointer_position = ui.input(|input| input.pointer.interact_pos());
        let hover_position = ui.input(|input| input.pointer.hover_pos());
        let shift = ui.input(|input| input.modifiers.shift);
        let painting = self.paint_mode.paints();
        // Keyboard camera: once the viewport has focus (from Tab; clicks
        // paint and orbit as before), arrow keys orbit, Shift with arrows
        // pans, and + and - zoom.
        response.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Other,
                true,
                "3D viewport. Arrow keys orbit, Shift and arrows pan, plus and minus zoom.",
            )
        });
        if response.has_focus() {
            ui.painter().rect_stroke(
                viewport.shrink(1.0),
                0.0,
                egui::Stroke::new(2.0, theme::ACCENT),
                egui::StrokeKind::Inside,
            );
            ui.memory_mut(|memory| {
                memory.set_focus_lock_filter(
                    response.id,
                    egui::EventFilter {
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        ..egui::EventFilter::default()
                    },
                );
            });
            ui.input(|input| keyboard_camera(&mut self.viewport_camera, input));
        }
        let gizmo = self.active_decal().is_some();
        if gizmo && ui.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.decal_gizmo = None;
        }
        if response.dragged() && !painting && !gizmo {
            let delta = ui.input(|input| input.pointer.delta());
            let delta = glam::Vec2::new(delta.x, delta.y);
            if shift {
                self.viewport_camera.pan_by(delta);
            } else {
                self.viewport_camera.orbit(delta);
            }
        }
        if response.hovered() && !painting {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            self.viewport_camera.zoom_by(scroll);
        }
        self.update_filament_frame(ui.ctx(), viewport);
        let painter = ui.painter_at(viewport);
        painter.rect_filled(viewport, 0.0, theme::VIEWPORT_BG);
        if self.mesh_geometry.is_some()
            && let Some(texture) = &self.filament_frame_texture
        {
            painter.image(
                texture.id(),
                viewport,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        let center = viewport.center();
        let software = self.filament_frame_texture.is_none();
        if software {
            let axis = egui::Color32::from_rgb(38, 40, 44);
            painter.line_segment(
                [
                    egui::pos2(viewport.left(), center.y),
                    egui::pos2(viewport.right(), center.y),
                ],
                egui::Stroke::new(1.0, axis),
            );
            painter.line_segment(
                [
                    egui::pos2(center.x, viewport.top()),
                    egui::pos2(center.x, viewport.bottom()),
                ],
                egui::Stroke::new(1.0, axis),
            );
        }

        if self.mesh_geometry.is_none() {
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                "Choose a mesh in the Scene drawer to begin",
                egui::FontId::proportional(16.0),
                theme::TEXT_WEAK,
            );
            return;
        }

        let selecting = self.paint_mode == PaintMode::Select;
        let projection = self.view_projection(viewport);
        let needs_projection = software
            || (painting && response.hovered())
            || response.clicked()
            || (selecting && response.hovered())
            || (gizmo && response.dragged());
        // Points come from the shown (possibly posed) mesh; picks read UVs
        // and positions from the bind mesh, which shares its topology.
        let shown = self.display_mesh();
        let projected = needs_projection
            .then(|| shown.as_ref().map(|mesh| projection.project_mesh(mesh)))
            .flatten();

        if let (Some(points), Some(mesh)) = (&projected, self.mesh_geometry.clone()) {
            let camera = (projection.stroke_camera(viewport), viewport.min);
            self.handle_paint_input(&response, points, &mesh, camera, pointer_position, shift);
        }
        if self.path_controls.is_some() {
            let (enter, escape) = ui.input(|input| {
                (
                    input.key_pressed(egui::Key::Enter),
                    input.key_pressed(egui::Key::Escape),
                )
            });
            if enter {
                self.commit_path(false);
            } else if escape {
                self.path_controls = None;
            }
        }
        if let Some(controls) = &self.path_controls {
            // The curve as it will run over the surface, from the current
            // view, so it stays on the model while the camera moves.
            let surface = substancecraft_core::PathSpec {
                surface: controls.iter().map(|(.., position)| *position).collect(),
                ..Default::default()
            };
            let curve: Vec<egui::Pos2> = surface
                .sample_surface(substancecraft_core::PATH_SAMPLES_PER_SEGMENT)
                .into_iter()
                .map(|point| projection.project(point).screen)
                .collect();
            painter.add(egui::Shape::line(
                curve,
                egui::Stroke::new(1.5, theme::ACCENT),
            ));
            for (.., position) in controls {
                painter.circle_filled(projection.project(*position).screen, 4.0, theme::ACCENT);
            }
        }

        if selecting {
            let hit = match (&projected, hover_position, &self.mesh_geometry) {
                (Some(points), Some(pointer), Some(mesh)) if response.hovered() => {
                    pick_triangle(mesh, points, pointer, Some(self.hidden_triangles()))
                }
                _ => None,
            };
            self.hovered_selection = match (hit, &self.mesh_topology) {
                (Some(hit), Some(topology)) => topology.select(hit.triangle, self.selection_mode),
                _ => Vec::new(),
            };
            if response.clicked() && !self.hovered_selection.is_empty() {
                let selection = self.hovered_selection.clone();
                self.apply_polygon_selection(&selection, shift);
            }
        } else {
            self.hovered_selection.clear();
        }

        let Some(mesh) = self.mesh_geometry.clone() else {
            return;
        };
        let mesh = mesh.as_ref();
        if software && let Some(points) = &projected {
            self.paint_software_mesh(&painter, mesh, points);
        }
        self.draw_edge_overlay(&painter, &projection);
        self.draw_decal_gizmo(&painter, &projection);

        if selecting {
            let overlay_points = projected
                .clone()
                .unwrap_or_else(|| projection.project_mesh(shown.as_deref().unwrap_or(mesh)));
            if let Some(mask) = self
                .selected_layer()
                .and_then(|layer| layer.polygon_mask.as_ref())
            {
                let triangles: Box<dyn Iterator<Item = u32>> = if mask.inverted {
                    Box::new(
                        (0..mesh.triangles.len() as u32)
                            .filter(|triangle| !mask.triangles.contains(triangle)),
                    )
                } else {
                    Box::new(mask.triangles.iter().copied())
                };
                painter.add(triangle_overlay(
                    mesh,
                    &overlay_points,
                    triangles,
                    theme::ACCENT.gamma_multiply(0.28),
                ));
            }
            let hover_color = if shift {
                theme::DANGER.gamma_multiply(0.35)
            } else {
                egui::Color32::from_white_alpha(46)
            };
            painter.add(triangle_overlay(
                mesh,
                &overlay_points,
                self.hovered_selection.iter().copied(),
                hover_color,
            ));
        }

        if painting
            && response.hovered()
            && let Some(pointer) = pointer_position
        {
            let cursor_color = if self.paint_mode == PaintMode::Mask {
                egui::Color32::WHITE
            } else {
                self.brush_color
            };
            // The footprint the brush paints: screen pixels when projecting,
            // else the texel radius carried onto the surface under the
            // cursor through its UVs.
            // Symmetric copies follow, drawn fainter so the brush under the
            // cursor stays clear.
            let footprints = if self.projects_strokes() {
                Vec::new()
            } else {
                self.brush_footprints(mesh, &projection, projected.as_deref(), pointer)
            };
            match footprints.split_first() {
                Some((outline, copies)) => {
                    painter.add(egui::Shape::closed_line(
                        outline.clone(),
                        egui::Stroke::new(1.5, cursor_color),
                    ));
                    for copy in copies {
                        painter.add(egui::Shape::closed_line(
                            copy.clone(),
                            egui::Stroke::new(1.0, cursor_color.gamma_multiply(0.6)),
                        ));
                    }
                }
                None => {
                    let radius = if self.projects_strokes() {
                        // Projected strokes are sized in screen pixels.
                        self.brush_radius
                    } else {
                        (self.brush_radius * self.viewport_camera.zoom * 0.35).clamp(3.0, 36.0)
                    };
                    painter.circle(
                        pointer,
                        radius,
                        cursor_color.gamma_multiply(0.12),
                        egui::Stroke::new(1.5, cursor_color),
                    );
                }
            }
            painter.circle_filled(pointer, 1.5, cursor_color);
        }
    }

    /// The screen outlines of the brush footprint on the surface under
    /// `pointer`: the brush's texel radius in the selected set, carried
    /// through the hovered triangle's UVs, then the footprint of each
    /// symmetric copy the stroke would paint. Empty off the mesh.
    fn brush_footprints(
        &self,
        mesh: &MeshGeometry,
        projection: &ViewProjection,
        points: Option<&[ProjectedVertex]>,
        pointer: egui::Pos2,
    ) -> Vec<Vec<egui::Pos2>> {
        let (Some(points), Some(set)) = (points, self.selected_texture_set_ref()) else {
            return Vec::new();
        };
        let Some(hit) = pick_triangle(mesh, points, pointer, Some(self.hidden_triangles())) else {
            return Vec::new();
        };
        let uv_radius = glam::Vec2::new(
            self.brush_radius / set.resolution.width.max(1) as f32,
            self.brush_radius / set.resolution.height.max(1) as f32,
        );
        // Outlines are found on the bind mesh; a pose carries them along.
        let posed = self.is_posed().then(|| self.shown_mesh()).flatten();
        let outline = |triangle: u32, position: glam::Vec3| -> Option<Vec<egui::Pos2>> {
            let outline = mesh.uv_brush_outline(triangle, position, uv_radius, 40)?;
            Some(
                outline
                    .into_iter()
                    .map(|point| match &posed {
                        Some(posed) => animation::carry_point(mesh, posed, triangle, point),
                        None => point,
                    })
                    .map(|point| projection.project(point).screen)
                    .collect(),
            )
        };
        let Some(main) = outline(hit.triangle, hit.position) else {
            return Vec::new();
        };
        let mut outlines = vec![main];
        if let Some(uv) = hit.uv {
            outlines.extend(
                self.symmetry_copies(&hit, uv)
                    .into_iter()
                    .filter_map(|(_, copy)| outline(copy.triangle, copy.position)),
            );
        }
        outlines
    }

    /// Flat-shaded painter's-algorithm fallback when Filament is unavailable.
    fn paint_software_mesh(
        &self,
        painter: &egui::Painter,
        mesh: &MeshGeometry,
        points: &[ProjectedVertex],
    ) {
        let preview = self.displayed_preview();
        let mut faces = mesh
            .triangles
            .iter()
            .filter_map(|triangle| {
                let [a, b, c] = triangle.map(|index| points.get(index as usize));
                let (a, b, c) = (a?, b?, c?);
                let uvs = triangle.map(|index| mesh.vertices[index as usize].uv);
                let normal = (b.view - a.view).cross(c.view - a.view).normalize_or_zero();
                let intensity = self.viewport_lighting.face_intensity(normal);
                Some((
                    (a.view.z + b.view.z + c.view.z) / 3.0,
                    [a.screen, b.screen, c.screen],
                    uvs,
                    intensity,
                ))
            })
            .collect::<Vec<_>>();
        faces.sort_by(|left, right| left.0.total_cmp(&right.0));
        for (_, points, uvs, intensity) in faces {
            let uv_center = (uvs[0] + uvs[1] + uvs[2]) / 3.0;
            let map_color = preview
                .and_then(|preview| preview.base().sample_uv(uv_center))
                .filter(|color| color[3] > 0)
                .unwrap_or([92, 132, 151, 255]);
            let color = egui::Color32::from_rgb(
                (map_color[0] as f32 * intensity).clamp(0.0, 255.0) as u8,
                (map_color[1] as f32 * intensity).clamp(0.0, 255.0) as u8,
                (map_color[2] as f32 * intensity).clamp(0.0, 255.0) as u8,
            );
            painter.add(egui::Shape::convex_polygon(
                points.to_vec(),
                color,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(182, 195, 202)),
            ));
        }
    }
}

/// Maps mesh positions through the current pose and camera onto the screen,
/// matching the orientation Filament renders.
struct ViewProjection {
    pose: BoneTransform,
    camera_rotation: glam::Quat,
    posed_center: glam::Vec3,
    center: egui::Pos2,
    scale: f32,
    pan: glam::Vec2,
}

#[derive(Clone, Copy)]
struct ProjectedVertex {
    screen: egui::Pos2,
    /// Camera-space position; larger `z` is closer to the viewer.
    view: glam::Vec3,
}

impl ViewProjection {
    fn project(&self, position: glam::Vec3) -> ProjectedVertex {
        let posed = self.pose.rotation.mul_vec3(position * self.pose.scale) + self.pose.translation;
        let view = self.camera_rotation * (posed - self.posed_center);
        ProjectedVertex {
            screen: egui::pos2(
                self.center.x + view.x * self.scale + self.pan.x,
                self.center.y - view.y * self.scale + self.pan.y,
            ),
            view,
        }
    }

    /// The same mapping as a stroke camera: object space to pixels from the
    /// corner of `viewport`, with depth growing away from the viewer.
    fn stroke_camera(&self, viewport: egui::Rect) -> substancecraft_core::StrokeCamera {
        let pose = glam::Mat4::from_scale_rotation_translation(
            self.pose.scale,
            self.pose.rotation,
            self.pose.translation,
        );
        let view = glam::Mat4::from_quat(self.camera_rotation)
            * glam::Mat4::from_translation(-self.posed_center);
        let origin = self.center - viewport.min;
        let screen = glam::Mat4::from_cols(
            glam::Vec4::new(self.scale, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, -self.scale, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, -1.0, 0.0),
            glam::Vec4::new(origin.x + self.pan.x, origin.y + self.pan.y, 0.0, 1.0),
        );
        substancecraft_core::StrokeCamera {
            transform: screen * view * pose,
            viewport: glam::Vec2::new(viewport.width(), viewport.height()),
        }
    }

    fn project_mesh(&self, mesh: &MeshGeometry) -> Vec<ProjectedVertex> {
        mesh.vertices
            .iter()
            .map(|vertex| self.project(vertex.position))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TriangleHit {
    triangle: u32,
    /// Interpolated UV, when the triangle's UVs are usable.
    uv: Option<glam::Vec2>,
    /// Object-space surface point.
    position: glam::Vec3,
}

/// Nearest triangle under `pointer`.
fn pick_triangle(
    mesh: &MeshGeometry,
    points: &[ProjectedVertex],
    pointer: egui::Pos2,
    hidden: Option<&BTreeSet<u32>>,
) -> Option<TriangleHit> {
    let mut best: Option<(f32, TriangleHit)> = None;
    for (index, triangle) in mesh.triangles.iter().enumerate() {
        if hidden.is_some_and(|hidden| hidden.contains(&(index as u32))) {
            continue;
        }
        let [a, b, c] = triangle.map(|vertex| points.get(vertex as usize));
        let (Some(a), Some(b), Some(c)) = (a, b, c) else {
            continue;
        };
        let min_x = a.screen.x.min(b.screen.x).min(c.screen.x);
        let max_x = a.screen.x.max(b.screen.x).max(c.screen.x);
        let min_y = a.screen.y.min(b.screen.y).min(c.screen.y);
        let max_y = a.screen.y.max(b.screen.y).max(c.screen.y);
        if pointer.x < min_x || pointer.x > max_x || pointer.y < min_y || pointer.y > max_y {
            continue;
        }
        let Some(weights) = barycentric_weights(pointer, [a.screen, b.screen, c.screen]) else {
            continue;
        };
        let depth = a.view.z * weights.x + b.view.z * weights.y + c.view.z * weights.z;
        if best.is_some_and(|(best_depth, _)| best_depth >= depth) {
            continue;
        }
        let uvs = triangle.map(|vertex| mesh.vertices[vertex as usize].uv);
        let positions = triangle.map(|vertex| mesh.vertices[vertex as usize].position);
        let uv = uvs[0] * weights.x + uvs[1] * weights.y + uvs[2] * weights.z;
        best = Some((
            depth,
            TriangleHit {
                triangle: index as u32,
                uv: uv.is_finite().then_some(uv),
                position: positions[0] * weights.x
                    + positions[1] * weights.y
                    + positions[2] * weights.z,
            },
        ));
    }
    best.map(|(_, hit)| hit)
}

/// Translucent fill over front-facing `triangles`, built as one mesh shape.
fn triangle_overlay(
    mesh: &MeshGeometry,
    points: &[ProjectedVertex],
    triangles: impl Iterator<Item = u32>,
    color: egui::Color32,
) -> egui::Shape {
    let mut overlay = egui::Mesh::default();
    for triangle in triangles.take(MAX_OVERLAY_TRIANGLES) {
        let Some(corners) = mesh.triangles.get(triangle as usize) else {
            continue;
        };
        let [a, b, c] = corners.map(|vertex| points.get(vertex as usize));
        let (Some(a), Some(b), Some(c)) = (a, b, c) else {
            continue;
        };
        if (b.view - a.view).cross(c.view - a.view).z <= 0.0 {
            continue;
        }
        let base = overlay.vertices.len() as u32;
        for point in [a, b, c] {
            overlay.colored_vertex(point.screen, color);
        }
        overlay.add_triangle(base, base + 1, base + 2);
    }
    egui::Shape::mesh(overlay)
}

fn barycentric_weights(point: egui::Pos2, screen: [egui::Pos2; 3]) -> Option<glam::Vec3> {
    let edge_a = glam::Vec2::new(screen[1].x - screen[0].x, screen[1].y - screen[0].y);
    let edge_b = glam::Vec2::new(screen[2].x - screen[0].x, screen[2].y - screen[0].y);
    let relative = glam::Vec2::new(point.x - screen[0].x, point.y - screen[0].y);
    let denominator = edge_a.perp_dot(edge_b);
    if denominator.abs() <= f32::EPSILON {
        return None;
    }
    let weight_b = relative.perp_dot(edge_b) / denominator;
    let weight_c = edge_a.perp_dot(relative) / denominator;
    let weight_a = 1.0 - weight_b - weight_c;
    if weight_a < -0.001 || weight_b < -0.001 || weight_c < -0.001 {
        return None;
    }
    Some(glam::Vec3::new(weight_a, weight_b, weight_c))
}

#[cfg(test)]
fn barycentric_uv(
    point: egui::Pos2,
    screen: [egui::Pos2; 3],
    uvs: [glam::Vec2; 3],
) -> Option<glam::Vec2> {
    let weights = barycentric_weights(point, screen)?;
    let uv = uvs[0] * weights.x + uvs[1] * weights.y + uvs[2] * weights.z;
    (uv.is_finite() && (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y)).then_some(uv)
}

/// Visits every layer of a stack, groups before their children.
fn for_each_layer_mut(layers: &mut [MaterialLayer], visit: &mut impl FnMut(&mut MaterialLayer)) {
    for layer in layers {
        visit(layer);
        if let Some(children) = layer.children.as_mut() {
            for_each_layer_mut(children, visit);
        }
    }
}

/// The file name of `path`, for messages.
fn path_label(path: &std::path::Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

fn mesh_bounds_center(mesh: &MeshGeometry) -> glam::Vec3 {
    let mut minimum = glam::Vec3::splat(f32::INFINITY);
    let mut maximum = glam::Vec3::splat(f32::NEG_INFINITY);
    for vertex in &mesh.vertices {
        minimum = minimum.min(vertex.position);
        maximum = maximum.max(vertex.position);
    }
    (minimum + maximum) * 0.5
}

const MIN_MESH_SCALE: f32 = 0.0001;
const MAX_MESH_SCALE: f32 = 1000.0;

fn sanitize_mesh_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(MIN_MESH_SCALE, MAX_MESH_SCALE)
    } else {
        1.0
    }
}

/// Brings the renderer's environment up to `lighting`, or to `custom` when
/// a panorama is loaded: a new preset or panorama is prefiltered again; a
/// new strength or rotation only turns and rescales the one already there.
/// `applied` records what the renderer has.
fn apply_environment(
    renderer: &mut FilamentRuntime,
    lighting: ViewportLighting,
    custom: Option<&environments::CustomEnvironment>,
    applied: &mut Option<(environments::EnvironmentSource, f32, f32)>,
) -> Result<(), substancecraft_filament::FilamentError> {
    use environments::EnvironmentSource;
    let preset = lighting.preset;
    // A panorama's sun is part of its image, so it has no separate light.
    let (source, lux, sun) = match custom {
        Some(custom) => (
            EnvironmentSource::Custom(custom.revision),
            custom.image.normalized_lux(),
            None,
        ),
        None => (
            EnvironmentSource::Preset(preset),
            preset.environment_lux(),
            preset.sun(),
        ),
    };
    let lux = lux * lighting.environment_scale();
    let rotation = if lighting.rotation.is_finite() {
        lighting.rotation
    } else {
        0.0
    };
    match *applied {
        Some(current) if current == (source, lux, rotation) => {}
        Some((current, _, _)) if current == source => {
            renderer.set_environment_transform(lux, rotation, sun);
            *applied = Some((source, lux, rotation));
        }
        _ => {
            match custom {
                Some(custom) => renderer.set_environment(&custom.image, lux, rotation, sun)?,
                None => {
                    let image =
                        preset.equirect(substancecraft_core::environment::ENVIRONMENT_WIDTH);
                    renderer.set_environment(&image, lux, rotation, sun)?;
                }
            }
            *applied = Some((source, lux, rotation));
        }
    }
    Ok(())
}

/// Reads an FBX (every mesh it holds), a glTF/GLB (the flattened scene), or
/// a STEP file (the tessellated, flattened assembly), choosing the importer
/// by extension.
fn load_mesh_file(path: &std::path::Path) -> Result<Vec<ImportedMesh>, String> {
    let extension = path
        .extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if extension == "gltf" || extension == "glb" {
        return import_gltf_file(path)
            .map(|mesh| vec![mesh])
            .map_err(|error| error.to_string());
    }
    if substancecraft_api::STEP_EXTENSIONS.contains(&extension.as_str()) {
        return substancecraft_api::import_step_file(path)
            .map(|mesh| vec![mesh])
            .map_err(|error| error.to_string());
    }
    let metadata = std::fs::metadata(path).map_err(|error| error.to_string())?;
    let file_size = usize::try_from(metadata.len())
        .map_err(|_| "FBX file is too large for this platform".to_owned())?;
    if file_size > MAX_FBX_BYTES {
        return Err(format!(
            "FBX file exceeds the {} MiB import limit",
            MAX_FBX_BYTES / (1024 * 1024)
        ));
    }
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    import_fbx_meshes(&bytes).map_err(|error| error.to_string())
}

/// Reads every mesh in a file as a high-poly source for baking: merged into
/// one scaled geometry, and kept by name for name-matched bakes.
fn load_high_poly(path: &std::path::Path, scale: f32) -> Result<HighPolySource, String> {
    let scale = sanitize_mesh_scale(scale);
    let mut merged = MeshGeometry::default();
    let mut parts = Vec::new();
    for mesh in load_mesh_file(path)? {
        merged
            .append(&mesh.geometry)
            .map_err(|_| "merged mesh has too many vertices".to_owned())?;
        parts.push((mesh.name, scaled_geometry(&mesh.geometry, scale)));
    }
    if merged.triangles.is_empty() {
        return Err("the file contains no mesh geometry".to_owned());
    }
    Ok(HighPolySource {
        path: path.to_path_buf(),
        geometry: scaled_geometry(&merged, scale),
        parts,
    })
}

/// How a high-to-low bake pairs each low part with the high mesh.
#[derive(Clone, Debug, Default)]
struct HighToLowPlan {
    /// The high-poly file's meshes by name, for name matching.
    parts: Vec<(String, MeshGeometry)>,
    options: substancecraft_core::HighToLowOptions,
    /// The low mesh's own name: its part name when it has one slot.
    low_name: String,
    /// Cage mesh, smoothed like the low mesh so the two still match.
    cage: Option<MeshGeometry>,
}

impl HighToLowPlan {
    /// The high-poly parts low part `name` takes detail from, merged:
    /// those of the same name when matching by name, else (or when none
    /// match) `None` for the whole high mesh.
    fn matched_high(&self, name: &str) -> Option<MeshGeometry> {
        if !self.options.match_by_name {
            return None;
        }
        let key = substancecraft_core::bake_name_key(name);
        let mut merged = MeshGeometry::default();
        for (part, geometry) in &self.parts {
            if substancecraft_core::bake_name_key(part) == key {
                merged.append(geometry).ok()?;
            }
        }
        (!merged.triangles.is_empty()).then_some(merged)
    }

    /// Names of low parts no high part matches.
    fn unmatched(&self, mesh: &MeshGeometry) -> Vec<String> {
        if !self.options.match_by_name {
            return Vec::new();
        }
        part_names(mesh, &self.low_name)
            .into_iter()
            .filter(|name| self.matched_high(name).is_none())
            .collect()
    }
}

/// Low part names: material slots, or `low_name` for a single-slot mesh.
fn part_names(mesh: &MeshGeometry, low_name: &str) -> Vec<String> {
    match mesh.material_slot_count() {
        0 | 1 => vec![low_name.to_owned()],
        count => (0..count as u32)
            .map(|slot| mesh.material_slot_name(slot))
            .collect(),
    }
}

/// Where `tracer` traced a bake's occlusion rays, for the bake log: the
/// GPU, the CPU, or the CPU after the GPU failed (and why).
fn rays_used(tracer: &substancecraft_gpu::FallbackTracer<'_>, gpu_wanted: bool) -> String {
    let gpu = tracer.gpu().map(|gpu| gpu.adapter_name().to_owned());
    match (tracer.traced_on_gpu(), tracer.fallback_reason(), gpu) {
        (true, None, Some(gpu)) => gpu,
        (true, Some(reason), Some(gpu)) => format!("{gpu}, then CPU ({reason})"),
        (_, Some(reason), _) if gpu_wanted => format!("CPU ({reason})"),
        _ => "CPU".to_owned(),
    }
}

/// Bakes mesh maps for each material slot from that slot's triangles, so
/// slots sharing UV space get their own maps. A single-slot mesh bakes
/// whole; an empty slot falls back to the whole mesh. With a high mesh,
/// `plan` picks each part's high parts, cage scale, and cage.
fn bake_slot_mesh_maps(
    mesh: &MeshGeometry,
    high: Option<&MeshGeometry>,
    settings: &MeshMapSettings,
    skew_map: Option<&substancecraft_core::TextureImage>,
    plan: &HighToLowPlan,
    tracer: &dyn substancecraft_core::OcclusionTracer,
    cancel: &AtomicBool,
) -> Result<Vec<MeshMaps>, MeshMapError> {
    let names = part_names(mesh, &plan.low_name);
    // The cage takes the low mesh's slots so it splits the same way.
    let cage = plan.cage.as_ref().map(|cage| MeshGeometry {
        triangle_materials: mesh.triangle_materials.clone(),
        material_slots: mesh.material_slots.clone(),
        ..cage.clone()
    });
    let bake_part = |low: &MeshGeometry, cage: Option<&MeshGeometry>, name: &str| {
        let matched = high.and_then(|_| plan.matched_high(name));
        let settings = MeshMapSettings {
            cage_distance: (settings.cage_distance * plan.options.cage_scale(name))
                .clamp(0.0001, 0.5),
            ..*settings
        };
        substancecraft_core::bake_mesh_maps_traced(
            low,
            matched.as_ref().or(high),
            &settings,
            skew_map,
            cage.filter(|_| high.is_some()),
            tracer,
            Some(cancel),
        )
    };
    if names.len() <= 1 {
        return Ok(vec![bake_part(mesh, cage.as_ref(), &names[0])?]);
    }
    names
        .iter()
        .enumerate()
        .map(|(slot, name)| {
            let submesh = mesh.submesh(slot as u32);
            if submesh.triangles.is_empty() {
                bake_part(mesh, cage.as_ref(), name)
            } else {
                let cage = cage.as_ref().map(|cage| cage.submesh(slot as u32));
                bake_part(&submesh, cage.as_ref(), name)
            }
        })
        .collect()
}

fn scaled_geometry(mesh: &MeshGeometry, scale: f32) -> MeshGeometry {
    let mut geometry = mesh.clone();
    if scale != 1.0 {
        for vertex in &mut geometry.vertices {
            vertex.position *= scale;
        }
    }
    geometry
}

impl eframe::App for SubstanceCraftApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.apply_ui_scale(ui.ctx());
        self.handle_bindings(ui.ctx());
        self.poll_mesh_map_job(ui.ctx());
        self.poll_preview_job(ui.ctx());
        self.poll_set_bake_job(ui.ctx());
        self.poll_resource_files(ui.ctx());
        self.refresh_set_caches();
        self.enforce_cache_budget();
        self.update_pose_transition(ui.ctx());
        self.advance_animation(ui.ctx());
        self.follow_animation_with_sequences();
        if !ui.ctx().input(|input| input.pointer.any_down()) {
            self.remember_layout();
        }
        self.update_viewport(ui);
        self.update_overlay(ui.ctx());
        self.sync_layer_history(ui.ctx());
        self.autosave_tick(std::time::Instant::now());
        ui.ctx().request_repaint_after(autosave::AUTOSAVE_INTERVAL);
    }

    fn on_exit(&mut self) {
        self.end_recovery_session();
    }
}

/// A channel's generic label; a set names its user channels itself (see
/// `channel_name`).
fn channel_label(channel: MaterialChannel) -> &'static str {
    const USER: [&str; substancecraft_core::MAX_USER_CHANNELS as usize] = [
        "User 0", "User 1", "User 2", "User 3", "User 4", "User 5", "User 6", "User 7",
    ];
    match channel {
        MaterialChannel::BaseColor => "Base Color",
        MaterialChannel::Id => "ID Mask",
        MaterialChannel::Roughness => "Roughness",
        MaterialChannel::Metallic => "Metallic",
        MaterialChannel::Normal => "Normal",
        MaterialChannel::Height => "Height",
        MaterialChannel::AmbientOcclusion => "Ambient Occlusion",
        MaterialChannel::Emissive => "Emissive",
        MaterialChannel::Opacity => "Opacity",
        MaterialChannel::User(slot) => USER.get(usize::from(slot)).copied().unwrap_or("User"),
    }
}

fn suggest_import_channel(
    name: &str,
    image: Option<&TextureImage>,
) -> (Option<MaterialChannel>, String) {
    let lower = name.to_lowercase();
    let tokens = lower
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<BTreeSet<_>>();
    let has = |aliases: &[&str]| aliases.iter().any(|alias| tokens.contains(alias));
    let pair = |first: &str, second: &str| tokens.contains(first) && tokens.contains(second);
    let named = if lower.contains("metallicroughness")
        || lower.contains("metalrough")
        || pair("metallic", "roughness")
        || pair("metal", "roughness")
    {
        return (
            None,
            "Packed metallic-roughness needs channel extraction".to_owned(),
        );
    } else if lower.contains("basecolor")
        || lower.contains("base_color")
        || has(&["albedo", "diffuse", "diff"])
        || pair("base", "color")
    {
        Some(MaterialChannel::BaseColor)
    } else if has(&["normal", "nrm", "normals"]) {
        Some(MaterialChannel::Normal)
    } else if has(&["rough", "roughness"]) {
        Some(MaterialChannel::Roughness)
    } else if has(&["metal", "metallic"]) {
        Some(MaterialChannel::Metallic)
    } else if has(&["ao", "occlusion"])
        || lower.contains("ambientocclusion")
        || pair("ambient", "occlusion")
    {
        return (
            None,
            "Ambient occlusion detected; this map is baked from mesh data".to_owned(),
        );
    } else if has(&["height", "displacement", "bump"]) {
        Some(MaterialChannel::Height)
    } else if has(&["emissive", "emission", "glow"]) {
        Some(MaterialChannel::Emissive)
    } else if has(&["opacity", "alpha", "transparency", "trans"]) {
        Some(MaterialChannel::Opacity)
    } else {
        None
    };
    if let Some(channel) = named {
        return (
            Some(channel),
            format!("Filename suggests {}", channel_label(channel)),
        );
    }

    let Some(image) = image else {
        return (None, "No channel name or image data detected".to_owned());
    };
    let pixel_count = image.rgba8.len() / 4;
    if pixel_count == 0 {
        return (None, "Image has no pixels to inspect".to_owned());
    }
    let stride = (pixel_count / 4096).max(1);
    let mut sampled = 0;
    let grayscale = image
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .step_by(stride)
        .take(4096)
        .all(|pixel| {
            sampled += 1;
            pixel[0].abs_diff(pixel[1]) <= 2 && pixel[1].abs_diff(pixel[2]) <= 2
        });
    if grayscale && sampled > 0 {
        (
            Some(MaterialChannel::Roughness),
            "Grayscale image; roughness is a candidate".to_owned(),
        )
    } else {
        (
            Some(MaterialChannel::BaseColor),
            "Color image; base color is a candidate".to_owned(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_api::write_texture;
    use substancecraft_core::LightingPreset;

    #[test]
    fn tc_0220_bitmap_import_suggestions_use_names_and_image_traits() {
        let gray = TextureImage::new(2, 1, vec![64, 64, 64, 255, 192, 192, 192, 255]).unwrap();
        let color = TextureImage::new(1, 1, vec![255, 40, 10, 255]).unwrap();
        assert_eq!(
            suggest_import_channel("Hull_BaseColor_4k", Some(&gray)).0,
            Some(MaterialChannel::BaseColor)
        );
        assert_eq!(
            suggest_import_channel("mask_roughness", Some(&color)).0,
            Some(MaterialChannel::Roughness),
            "name tokens take precedence over image traits"
        );
        assert_eq!(
            suggest_import_channel("unknown", Some(&gray)).0,
            Some(MaterialChannel::Roughness)
        );
        assert_eq!(
            suggest_import_channel("unknown", Some(&color)).0,
            Some(MaterialChannel::BaseColor)
        );
        assert_eq!(
            suggest_import_channel("packed_metallic_roughness", Some(&color)).0,
            None,
            "packed data is not silently assigned to one channel"
        );
        let (channel, reason) = suggest_import_channel("chair_ao", Some(&gray));
        assert_eq!(channel, None, "AO comes from mesh-map baking");
        assert!(reason.contains("Ambient occlusion"));
    }

    #[test]
    fn tc_0221_confirmed_bitmap_assignments_create_undoable_channel_fills() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.add_texture_set();
        let base = folder.path().join("BaseColor.png");
        let rough = folder.path().join("Roughness.png");
        for path in [&base, &rough] {
            write_texture(
                path,
                &BakedTexture {
                    width: 1,
                    height: 1,
                    rgba8: vec![128, 128, 128, 255],
                },
            )
            .unwrap();
        }
        let base_id = app.add_resource(&base).unwrap();
        let rough_id = app.add_resource(&rough).unwrap();
        app.pending_import_assignments = vec![
            ImportAssignment {
                resource: base_id,
                channel: Some(MaterialChannel::BaseColor),
                reason: "Filename suggests Base Color".to_owned(),
            },
            ImportAssignment {
                resource: rough_id,
                channel: Some(MaterialChannel::Roughness),
                reason: "Filename suggests Roughness".to_owned(),
            },
        ];

        app.confirm_import_assignments();
        let set = app.selected_texture_set_ref().unwrap();
        assert_eq!(set.layers.len(), 2);
        assert_eq!(
            set.layers[0].fill.unwrap().textures.base_color,
            Some(base_id)
        );
        assert_eq!(
            set.layers[0].channels,
            BTreeSet::from([MaterialChannel::BaseColor])
        );
        assert_eq!(
            set.layers[1].fill.unwrap().textures.roughness,
            Some(rough_id)
        );
        assert!(app.pending_import_assignments.is_empty());
        app.undo();
        assert!(app.selected_texture_set_ref().unwrap().layers.is_empty());
        app.redo();
        assert_eq!(app.selected_texture_set_ref().unwrap().layers.len(), 2);
    }

    #[test]
    fn screen_point_maps_to_triangle_uv_and_rejects_outside_points() {
        let screen = [
            egui::pos2(0.0, 0.0),
            egui::pos2(100.0, 0.0),
            egui::pos2(0.0, 100.0),
        ];
        let uvs = [
            glam::Vec2::new(0.0, 0.0),
            glam::Vec2::new(1.0, 0.0),
            glam::Vec2::new(0.0, 1.0),
        ];

        assert_eq!(
            barycentric_uv(egui::pos2(25.0, 25.0), screen, uvs),
            Some(glam::Vec2::new(0.25, 0.25))
        );
        assert_eq!(barycentric_uv(egui::pos2(75.0, 75.0), screen, uvs), None);
    }

    /// Drives background jobs until `done` or a timeout.
    pub(crate) fn pump_until(
        app: &mut SubstanceCraftApp,
        context: &egui::Context,
        done: impl Fn(&SubstanceCraftApp) -> bool,
    ) {
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while !done(app) {
            assert!(
                std::time::Instant::now() < deadline,
                "background jobs timed out"
            );
            app.poll_mesh_map_job(context);
            app.poll_preview_job(context);
            app.poll_set_bake_job(context);
            app.refresh_baked_preview();
            app.refresh_set_caches();
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    #[allow(clippy::unreachable)]
    fn tc_0054_edge_wear_layer_bakes_maps_and_previews_metal_edges_in_background() {
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        app.mesh_map_settings.resolution = 64;
        app.mesh_map_settings.ao_samples = 4;

        app.add_edge_wear_layer();

        let set = app.selected_texture_set_ref().unwrap();
        assert_eq!(set.layers.len(), 1, "Edge Wear adds one layer");
        assert_eq!(app.selected_layer, Some(set.layers[0].id));
        assert!(app.history.can_undo());
        // Small set so the debug-build bake stays quick.
        app.project.texture_sets[0].resolution = TextureResolution {
            width: 96,
            height: 96,
        };
        app.invalidate_preview();

        app.refresh_baked_preview();
        assert!(
            app.mesh_map_job.is_some(),
            "the generator triggers a mesh-map bake"
        );
        pump_until(&mut app, &context, |app| {
            app.displayed_preview()
                .is_some_and(|preview| matches!(preview, PreviewTextures::Material(_)))
                && app.baked_preview.as_ref().map(|preview| preview.key) == Some(app.preview_key())
        });

        let Some(PreviewTextures::Material(textures)) = app.displayed_preview() else {
            unreachable!("checked by pump_until")
        };
        let metal_texels = textures
            .orm
            .rgba8
            .chunks(4)
            .filter(|pixel| pixel[2] > 200)
            .count();
        let total = textures.orm.rgba8.len() / 4;
        assert!(metal_texels > 0, "worn edges are metallic");
        assert!(metal_texels < total / 2, "flat faces stay unworn");
        // Unworn texels show the neutral base, not transparent black.
        assert!(
            textures
                .base_color
                .rgba8
                .chunks(4)
                .all(|pixel| pixel[3] == 255)
        );

        app.undo();
        assert!(app.selected_texture_set_ref().unwrap().layers.is_empty());
    }

    #[test]
    fn tc_0045_viewport_pick_returns_the_nearest_triangle_under_the_pointer() {
        let cube = PrimitivePreset::Cube.build_mesh();
        let projection = ViewProjection {
            pose: BoneTransform::IDENTITY,
            camera_rotation: glam::Quat::IDENTITY,
            posed_center: glam::Vec3::ZERO,
            center: egui::pos2(100.0, 100.0),
            scale: 100.0,
            pan: glam::Vec2::ZERO,
        };
        let points = projection.project_mesh(&cube);

        // Looking down -Z, the screen centre covers the +Z face (triangles 0
        // and 1) in front of the -Z face (triangles 2 and 3).
        let hit = pick_triangle(&cube, &points, egui::pos2(110.0, 90.0), None).unwrap();
        assert!(hit.triangle < 2, "picked {}", hit.triangle);
        assert!(hit.uv.is_some());
        assert_eq!(
            pick_triangle(&cube, &points, egui::pos2(400.0, 400.0), None),
            None
        );

        let topology = MeshTopology::build(&cube);
        let face = topology.select(hit.triangle, SelectionMode::UvIsland);
        assert_eq!(face, vec![0, 1]);
    }

    /// A unit quad split into two triangles, each with its own glTF material
    /// ("Body" and "Trim"), both covering the full UV square.
    fn write_two_material_gltf(folder: &std::path::Path) -> std::path::PathBuf {
        let positions: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // glTF V runs down; these map to SubstanceCraft UV corners.
        let uvs: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let mut buffer = Vec::new();
        for value in positions.iter().flatten().chain(uvs.iter().flatten()) {
            buffer.extend(value.to_le_bytes());
        }
        for index in [0_u16, 1, 2, 0, 2, 3] {
            buffer.extend(index.to_le_bytes());
        }
        std::fs::write(folder.join("quad.bin"), &buffer).unwrap();
        let json = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "materials": [{{ "name": "Body" }}, {{ "name": "Trim" }}],
  "meshes": [{{ "primitives": [
    {{ "attributes": {{ "POSITION": 0, "TEXCOORD_0": 1 }}, "indices": 2, "material": 0 }},
    {{ "attributes": {{ "POSITION": 0, "TEXCOORD_0": 1 }}, "indices": 3, "material": 1 }}
  ] }}],
  "buffers": [{{ "uri": "quad.bin", "byteLength": {length} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 48 }},
    {{ "buffer": 0, "byteOffset": 48, "byteLength": 32 }},
    {{ "buffer": 0, "byteOffset": 80, "byteLength": 6 }},
    {{ "buffer": 0, "byteOffset": 86, "byteLength": 6 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
       "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] }},
    {{ "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2" }},
    {{ "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }},
    {{ "bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR" }}
  ]
}}"#,
            length = buffer.len()
        );
        let path = folder.join("quad.gltf");
        std::fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn tc_0265_mesh_edits_and_replacement_undo_and_history_survives_saving() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.add_paint_layer();
        let triangles =
            |app: &SubstanceCraftApp| app.mesh_geometry.as_ref().unwrap().triangles.len();
        let cube_triangles = triangles(&app);

        // Switching primitive is undoable and reloads the old mesh.
        app.select_primitive(PrimitivePreset::Sphere);
        assert_ne!(triangles(&app), cube_triangles);
        app.undo();
        assert_eq!(app.project.mesh.as_ref().unwrap().source, "builtin:Cube");
        assert_eq!(triangles(&app), cube_triangles);
        app.redo();
        assert_eq!(app.selected_primitive, Some(PrimitivePreset::Sphere));
        app.undo();

        // Subdividing remaps polygon masks; undo restores both the
        // smoothing and the original triangle indices.
        let layer_id = app.selected_layer.unwrap();
        app.selected_layer_mut().unwrap().polygon_mask = Some(substancecraft_core::PolygonMask {
            triangles: [0, 1].into_iter().collect(),
            inverted: false,
        });
        let smoothing = MeshSmoothing {
            subdivision_levels: 1,
            ..app.mesh_smoothing
        };
        app.set_mesh_smoothing(smoothing);
        let mask = |app: &SubstanceCraftApp| {
            app.selected_texture_set_ref()
                .unwrap()
                .layer(layer_id)
                .unwrap()
                .polygon_mask
                .clone()
                .unwrap()
                .triangles
        };
        assert_eq!(mask(&app).len(), 8, "each triangle splits in four");
        app.undo();
        assert_eq!(app.mesh_smoothing.subdivision_levels, 0);
        assert_eq!(mask(&app), [0, 1].into_iter().collect());
        assert_eq!(triangles(&app), cube_triangles);

        // Replacing the mesh keeps the texture sets and layers.
        let sets = app.project.texture_sets.clone();
        let quad = write_two_material_gltf(folder.path());
        app.replace_mesh_with(&quad);
        assert_eq!(triangles(&app), 2);
        assert_eq!(app.project.texture_sets, sets);
        app.undo();
        assert_eq!(triangles(&app), cube_triangles);
        app.redo();
        assert_eq!(triangles(&app), 2);

        // Saving keeps the undo steps for the next open of the same file.
        let project = folder.path().join("scene.sproject");
        app.project_path = Some(project.clone());
        app.save_project();
        assert_eq!(app.project_error, None);
        assert!(substancecraft_api::history_path(&project).exists());
        let mut reopened = SubstanceCraftApp::default();
        reopened.open_project_path(project.clone());
        assert!(reopened.history.can_undo());
        reopened.undo();
        assert_eq!(
            triangles(&reopened),
            cube_triangles,
            "undo crosses sessions"
        );
        // A project changed behind the history's back drops it.
        let edited = std::fs::read_to_string(&project).unwrap() + "\n";
        std::fs::write(&project, edited).unwrap();
        let mut stale = SubstanceCraftApp::default();
        stale.open_project_path(project);
        assert!(!stale.history.can_undo());
    }

    fn small_sets(app: &mut SubstanceCraftApp, size: u32) {
        for set in &mut app.project.texture_sets {
            set.resolution = TextureResolution {
                width: size,
                height: size,
            };
        }
        app.invalidate_all_sets();
    }

    #[test]
    fn tc_0376_high_to_low_bakes_match_parts_by_name_and_scale_cages_per_part() {
        use substancecraft_core::{HighToLowOptions, MeshVertex};

        // A flat quad over x in `x0..x0 + 1` at height `y`, UVs `u0..u0 + 0.5`.
        let quad = |x0: f32, y: f32, u0: f32| {
            let vertex = |x: f32, z: f32| MeshVertex {
                position: glam::Vec3::new(x0 + x, y, z),
                uv: glam::Vec2::new(u0 + x * 0.5, z),
            };
            MeshGeometry {
                vertices: vec![
                    vertex(0.0, 0.0),
                    vertex(1.0, 0.0),
                    vertex(1.0, 1.0),
                    vertex(0.0, 1.0),
                ],
                triangles: vec![[0, 2, 1], [0, 3, 2]],
                ..MeshGeometry::default()
            }
        };
        // Low: a Lid quad and a Body quad, one material slot each.
        let mut low = quad(0.0, 0.0, 0.0);
        low.append(&quad(2.0, 0.0, 0.5)).unwrap();
        low.triangle_materials = vec![0, 0, 1, 1];
        low.material_slots = vec!["Lid".to_owned(), "Body".to_owned()];
        // High: the lid's detail sits well above it, the body's just above
        // the body, and a stray body piece just above the lid.
        let parts = vec![
            ("Lid_high".to_owned(), quad(0.0, 0.3, 0.0)),
            ("Body_high".to_owned(), quad(2.0, 0.05, 0.5)),
            ("body_high".to_owned(), quad(0.0, 0.05, 0.0)),
        ];
        let mut high = MeshGeometry::default();
        for (_, part) in &parts {
            high.append(part).unwrap();
        }
        let settings = MeshMapSettings {
            resolution: 16,
            ao_samples: 2,
            cage_distance: 0.05,
            ..MeshMapSettings::default()
        };
        let cancel = AtomicBool::new(false);
        let bake = |options: HighToLowOptions| {
            let plan = HighToLowPlan {
                parts: parts.clone(),
                options,
                low_name: "Crate".to_owned(),
                cage: None,
            };
            bake_slot_mesh_maps(
                &low,
                Some(&high),
                &settings,
                None,
                &plan,
                &substancecraft_core::CpuTracer,
                &cancel,
            )
            .unwrap()
            .iter()
            .map(MeshMaps::miss_fraction)
            .collect::<Vec<_>>()
        };

        // Without matching the lid catches the stray body piece.
        let caught = |misses: &[f32]| misses.iter().all(|miss| *miss < 0.3);
        assert!(caught(&bake(HighToLowOptions::default())));
        // Matched by name it only sees its own detail, out of reach...
        let matched = HighToLowOptions {
            match_by_name: true,
            ..HighToLowOptions::default()
        };
        let misses = bake(matched.clone());
        assert!(misses[0] > 0.9 && misses[1] < 0.3, "{misses:?}");
        // ...until its cage alone grows.
        let mut wider = matched.clone();
        wider.cage_scales.insert("Lid".to_owned(), 6.0);
        assert!(caught(&bake(wider.clone())));

        // Through the app: options are mesh edits that undo, stale the maps,
        // and name parts without a match.
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(low.clone()));
        app.project.mesh = Some(MeshAsset {
            id: MeshId(1),
            source: "crate.fbx".to_owned(),
            scale: 1.0,
            smoothing: MeshSmoothing::default(),
            high_poly_source: None,
            content_hash: None,
            high_poly_hash: None,
            high_to_low: Default::default(),
        });
        app.high_poly = Some(HighPolySource {
            path: "sculpt.fbx".into(),
            geometry: high.clone(),
            parts: vec![("Lid_high".to_owned(), quad(0.0, 0.3, 0.0))],
        });
        let key = app.bake_key();
        app.set_high_to_low(wider.clone());
        assert_ne!(app.bake_key().high_to_low, key.high_to_low);
        assert_eq!(app.unmatched_low_parts(), ["Body"]);
        let mut invalid = wider.clone();
        invalid.cage_scales.insert("Body".to_owned(), 50.0);
        app.set_high_to_low(invalid);
        assert_eq!(
            app.high_to_low_options(),
            wider,
            "out-of-range scale refused"
        );
        app.undo();
        assert_eq!(app.high_to_low_options(), HighToLowOptions::default());
        assert_eq!(app.bake_key().high_to_low, key.high_to_low);
    }

    #[test]
    fn tc_0374_udim_tiles_show_and_paint_in_the_viewports_and_cache_per_tile() {
        use substancecraft_core::MeshVertex;

        // Two quads, in tiles 1001 and 1002.
        let mut mesh = MeshGeometry::default();
        for tile in 0..2 {
            let base = mesh.vertices.len() as u32;
            for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
                mesh.vertices.push(MeshVertex {
                    position: glam::Vec3::new(x + tile as f32 * 1.1, y, 0.0),
                    uv: glam::Vec2::new(x + tile as f32, y),
                });
            }
            mesh.triangles.push([base, base + 1, base + 2]);
            mesh.triangles.push([base, base + 2, base + 3]);
        }
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(mesh.clone()));
        app.add_paint_layer();
        small_sets(&mut app, 16);
        let atlas = app.udim_atlas.clone().unwrap();
        assert_eq!(atlas.grid(), (2, 1));
        assert!(app.paintable_uv(glam::Vec2::new(1.5, 0.5)));
        assert!(!app.paintable_uv(glam::Vec2::new(2.5, 0.5)), "no tile 1003");

        // A dab on the second quad lands in tile 1002.
        app.set_paint_mode(PaintMode::Paint);
        app.brush_color = egui::Color32::WHITE;
        app.brush_radius = 2.0;
        let hit = mesh
            .closest_surface_point(glam::Vec3::new(1.6, 0.5, 0.0))
            .unwrap();
        app.paint_hit(TriangleHit {
            triangle: hit.triangle,
            uv: Some(hit.uv),
            position: hit.position,
        });
        app.finish_active_stroke();
        let points = &app.selected_layer().unwrap().paint_strokes[0].points;
        assert!(points[0].x > 1.0);

        // The preview is the atlas: tile 1002 (right half) shows the dab,
        // tile 1001 does not.
        let shown = |app: &SubstanceCraftApp| {
            let preview = app.displayed_preview();
            let material = preview.and_then(|preview| match preview {
                PreviewTextures::Material(material) => Some(material),
                PreviewTextures::Single(_) => None,
            });
            material.unwrap().base_color.clone()
        };
        app.invalidate_preview();
        tests::pump_until(&mut app, &context, |app| {
            app.preview_job.is_none() && app.displayed_preview().is_some()
        });
        let atlas_texture = shown(&app);
        assert_eq!((atlas_texture.width, atlas_texture.height), (32, 16));
        assert!(atlas_texture.pixel(24, 8).unwrap()[0] > 200, "painted");
        assert!(atlas_texture.pixel(8, 8).unwrap()[0] < 200, "unpainted");

        // Painting tile 1002 again rebakes only that tile.
        let hits = app.udim_cache.lock().unwrap().hits;
        let hit = mesh
            .closest_surface_point(glam::Vec3::new(1.8, 0.3, 0.0))
            .unwrap();
        app.paint_hit(TriangleHit {
            triangle: hit.triangle,
            uv: Some(hit.uv),
            position: hit.position,
        });
        app.finish_active_stroke();
        app.invalidate_preview();
        tests::pump_until(&mut app, &context, |app| {
            app.preview_failed
                || (app.preview_job.is_none()
                    && app
                        .baked_preview
                        .as_ref()
                        .is_some_and(|preview| preview.key == app.preview_key()))
        });
        assert_eq!(app.bake_error, None);
        assert_eq!(
            app.udim_cache.lock().unwrap().hits,
            hits + 1,
            "tile 1001 reused"
        );
        assert_eq!(shown(&app).pixel(8, 8), atlas_texture.pixel(8, 8));

        // Mesh-map previews show every tile's maps side by side, baking
        // tile 1002's maps from its shifted UVs.
        app.mesh_map_settings.resolution = 32;
        app.mesh_map_settings.ao_samples = 2;
        app.start_mesh_map_bake();
        tests::pump_until(&mut app, &context, |app| app.mesh_map_job.is_none());
        app.mesh_map_preview = Some(MeshMapKind::WorldNormal);
        tests::pump_until(&mut app, &context, |app| {
            app.preview_job.is_none()
                && app
                    .baked_preview
                    .as_ref()
                    .is_some_and(|preview| preview.key == app.preview_key())
        });
        let maps = app
            .displayed_preview()
            .and_then(|preview| match preview {
                PreviewTextures::Single(texture) => Some(texture.clone()),
                PreviewTextures::Material(_) => None,
            })
            .unwrap();
        assert_eq!((maps.width, maps.height), (64, 32));
        // Both quads face +Z, so both halves carry the same normal.
        assert_eq!(maps.pixel(16, 16), maps.pixel(48, 16));
        assert!(maps.pixel(48, 16).unwrap()[3] > 0, "tile 1002 is covered");
    }

    #[test]
    fn tc_0101_gltf_material_slots_become_texture_sets_that_each_show_on_their_slot() {
        let context = egui::Context::default();
        let folder = tempfile::tempdir().unwrap();
        let path = write_two_material_gltf(folder.path());
        let mut app = SubstanceCraftApp::default();

        app.load_imported_mesh(&path);
        assert_eq!(app.import_error, None);
        let names = app
            .project
            .texture_sets
            .iter()
            .map(|set| set.material_slot.clone())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["Body", "Trim"]);
        let (body, trim) = (
            app.project.texture_sets[0].id,
            app.project.texture_sets[1].id,
        );
        assert_eq!(app.selected_texture_set, Some(body));
        assert_eq!(app.slot_texture_set(1), Some(trim));
        // Painting hits only count on the selected set's triangles.
        assert!(app.triangle_in_selected_set(0));
        assert!(!app.triangle_in_selected_set(1));
        small_sets(&mut app, 32);

        // Give Trim a red fill; Body stays plain.
        app.select_texture_set(trim);
        app.add_fill_layer();
        if let Some(layer) = app.selected_layer_mut() {
            layer.fill = Some(LayerFill {
                color: [255, 0, 0, 255],
                ..LayerFill::default()
            });
        }
        app.select_texture_set(body);
        app.invalidate_all_sets();
        pump_until(&mut app, &context, |app| {
            app.preview_job.is_none()
                && app.set_bake_job.is_none()
                && app
                    .set_materials
                    .get(&trim)
                    .is_some_and(|cached| cached.revision == app.set_revision(trim))
        });

        let plan = app.slot_upload_plan(PreviewTarget::Material);
        assert_eq!(plan.len(), 2);
        let centre = |source: &SlotSource| match source {
            SlotSource::Material(textures) => textures.base_color.pixel(16, 16),
            _ => None,
        };
        assert_eq!(
            centre(&plan[1]),
            Some([255, 0, 0, 255]),
            "the Trim slot shows its cached material"
        );
        let base_color = centre(&plan[0]);
        assert_eq!(base_color, Some([128, 128, 128, 255]), "Body is untouched");
    }

    #[test]
    fn tc_0102_bitmap_resources_texture_fills_and_material_sets_export_as_png() {
        let folder = tempfile::tempdir().unwrap();
        let bitmap = folder.path().join("stripes.png");
        write_texture(
            &bitmap,
            &BakedTexture {
                width: 2,
                height: 1,
                rgba8: vec![0, 255, 0, 255, 0, 255, 0, 255],
            },
        )
        .unwrap();
        let mut app = SubstanceCraftApp::default();
        let resource = app.add_resource(&bitmap).unwrap();
        assert_eq!(app.project.resources.len(), 1);

        app.add_fill_layer();
        if let Some(layer) = app.selected_layer_mut() {
            layer.channels.insert(MaterialChannel::Emissive);
            let fill = layer.fill.get_or_insert_with(LayerFill::default);
            fill.textures.base_color = Some(resource);
            fill.emissive = [10, 20, 30];
        }
        small_sets(&mut app, 16);
        assert!(app.resource_in_use(resource));
        app.remove_resource(resource);
        assert_eq!(app.project.resources.len(), 1, "used resources stay");

        let set = app.selected_texture_set_ref().unwrap().clone();
        let out = folder.path().join("out");
        std::fs::create_dir(&out).unwrap();
        let written = app
            .write_export(&set, &app.selected_export_preset(), &out)
            .unwrap();
        let names = written
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let stem = set.name.replace(' ', "_");
        assert_eq!(
            names,
            ["BaseColor", "ORM", "Normal", "Emissive"].map(|suffix| format!("{stem}_{suffix}.png"))
        );
        let base = load_texture_image(&written[0]).unwrap();
        assert_eq!(&base.rgba8[0..4], &[0, 255, 0, 255], "textured green fill");
        let emissive = load_texture_image(&written[3]).unwrap();
        assert_eq!(&emissive.rgba8[0..4], &[10, 20, 30, 255]);

        // The project saves the reference and reloads the pixels.
        let project = folder.path().join("scene.sproject");
        save_project_file(&project, &app.project).unwrap();
        app.project = load_project_file(&project).unwrap();
        app.resource_images.clear();
        app.reload_resources();
        assert!(app.resource_images.contains_key(&resource));
        assert_eq!(app.project_error, None);
    }

    #[test]
    #[allow(clippy::unreachable)]
    fn tc_0255_painting_patches_the_preview_and_matches_a_full_bake() {
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        small_sets(&mut app, 256);
        app.add_fill_layer();
        app.add_paint_layer();
        app.paint_mode = PaintMode::Paint;
        let settled = |app: &SubstanceCraftApp| {
            app.preview_job.is_none()
                && app
                    .baked_preview
                    .as_ref()
                    .is_some_and(|preview| preview.key == app.preview_key())
        };
        pump_until(&mut app, &context, settled);
        let full_bake = |app: &SubstanceCraftApp| {
            let set = app.selected_texture_set_ref().unwrap().clone();
            app.bake_inputs(set.id).bake_material(&set).unwrap()
        };
        let shown = |app: &SubstanceCraftApp| match &app.baked_preview.as_ref().unwrap().textures {
            PreviewTextures::Material(material) => MaterialTextures::clone(material),
            PreviewTextures::Single(_) => unreachable!("material preview expected"),
        };

        for step in 0..6 {
            app.append_stroke_point(glam::Vec2::new(0.2 + step as f32 * 0.05, 0.4));
            // A stroke alone patches the preview.
            let (_, regions) = app.preview_patch(&app.preview_key()).unwrap();
            assert!(!regions.is_empty() && regions.len() <= 2);
            pump_until(&mut app, &context, settled);
            assert!(shown(&app) == full_bake(&app), "step {step} differs");
        }
        app.finish_active_stroke();

        // A setting changed without invalidating the preview is caught by
        // the shape diff: the next stroke also rebakes the paint layer's
        // strokes, and the result still matches a full bake.
        app.selected_layer_mut().unwrap().opacity = 0.5;
        app.append_stroke_point(glam::Vec2::new(0.7, 0.7));
        let (_, patches) = app.preview_patch(&app.preview_key()).unwrap();
        assert!(patches.len() >= 2, "the new dab and the earlier stroke");
        pump_until(&mut app, &context, settled);
        assert!(shown(&app) == full_bake(&app));
        app.finish_active_stroke();

        // So is a fill layer's: only the textures its channels reach.
        let fill = app.selected_texture_set_ref().unwrap().layers[0].id;
        app.selected_layer = Some(fill);
        app.layer_settled = None;
        app.sync_layer_history(&context);
        let roughness = app.selected_layer().unwrap().fill.unwrap().roughness;
        app.selected_layer_mut()
            .unwrap()
            .fill
            .as_mut()
            .unwrap()
            .roughness = (roughness + 0.3) % 1.0;
        app.sync_layer_history(&context);
        let (_, patches) = app.preview_patch(&app.preview_key()).unwrap();
        assert!(
            patches
                .iter()
                .all(|patch| !patch.channels.planes().base_color),
            "{patches:?}"
        );
        pump_until(&mut app, &context, settled);
        assert!(shown(&app) == full_bake(&app));

        // Undo and redo of layer edits patch too (their bake starts at
        // once): settings, then a stroke.
        let job_patches = |app: &SubstanceCraftApp| {
            app.preview_job
                .as_ref()
                .and_then(|job| job.patch.clone())
                .map(|(_, patches)| patches)
        };
        for step in ["undo settings", "redo settings"] {
            if step.starts_with("undo") {
                app.undo();
            } else {
                app.redo();
            }
            assert!(job_patches(&app).is_some(), "{step}");
            pump_until(&mut app, &context, settled);
            assert!(shown(&app) == full_bake(&app), "{step}");
        }
        app.undo();
        pump_until(&mut app, &context, settled);
        app.undo();
        let patches = job_patches(&app).unwrap();
        assert!(
            patches.iter().all(|patch| patch.rect.len() < 256 * 256 / 2),
            "an undone stroke rebakes its texels: {patches:?}"
        );
        pump_until(&mut app, &context, settled);
        assert!(shown(&app) == full_bake(&app), "undo stroke");
    }

    #[test]
    #[allow(clippy::unreachable)]
    fn tc_0402_background_sets_patch_their_cached_material_and_selection_reuses_it() {
        let context = egui::Context::default();
        let folder = tempfile::tempdir().unwrap();
        let path = write_two_material_gltf(folder.path());
        let mut app = SubstanceCraftApp::default();
        app.load_imported_mesh(&path);
        small_sets(&mut app, 256);
        let (body, trim) = (
            app.project.texture_sets[0].id,
            app.project.texture_sets[1].id,
        );
        // Trim: a fill linked to a project master, and a painted layer.
        app.select_texture_set(trim);
        app.add_fill_layer();
        app.make_selected_layer_master();
        let master = app.editing_material.unwrap();
        app.add_paint_layer();
        app.paint_mode = PaintMode::Paint;
        for step in 0..3 {
            app.append_stroke_point(glam::Vec2::new(0.3 + step as f32 * 0.05, 0.5));
        }
        app.finish_active_stroke();
        app.select_texture_set(body);
        let trim_set = |app: &SubstanceCraftApp| -> TextureSet {
            app.project
                .texture_sets
                .iter()
                .find(|set| set.id == trim)
                .unwrap()
                .clone()
        };
        let cached_current = |app: &SubstanceCraftApp| {
            app.preview_job.is_none()
                && app.set_bake_job.is_none()
                && app
                    .set_materials
                    .get(&trim)
                    .is_some_and(|cached| cached.revision == app.set_revision(trim))
        };
        pump_until(&mut app, &context, cached_current);
        let cached = |app: &SubstanceCraftApp| {
            let cached = &app.set_materials[&trim];
            (cached.patched, MaterialTextures::clone(&cached.textures))
        };
        let full_bake = |app: &SubstanceCraftApp| {
            let set = app
                .project
                .texture_sets
                .iter()
                .find(|set| set.id == trim)
                .unwrap();
            app.bake_inputs(trim).bake_material(set).unwrap()
        };
        assert!(cached(&app).1 == full_bake(&app));

        // Undoing and redoing the Trim stroke while Body is selected patches
        // Trim's cached material in the background.
        for step in ["undo stroke", "redo stroke"] {
            if step.starts_with("undo") {
                app.undo();
            } else {
                app.redo();
            }
            assert_eq!(app.selected_texture_set, Some(body));
            let set = preview_copy(&trim_set(&app)).into_owned();
            let (_, patches) = app.cached_material_patch(&set).unwrap();
            assert!(
                !patches.is_empty() && patches.iter().all(|p| p.rect.len() < 256 * 256 / 2),
                "{step}: only the stroke's texels"
            );
            pump_until(&mut app, &context, cached_current);
            let (patched, textures) = cached(&app);
            assert!(patched, "{step}");
            assert!(
                textures == full_bake(&app),
                "{step} differs from a full bake"
            );
        }

        // Editing the master updates its instance in Trim: only the
        // textures the changed value reaches are rebaked.
        let mut edited = app.project.material(master).unwrap().clone();
        edited.fill.roughness = (edited.fill.roughness + 0.4) % 1.0;
        app.edit_project_material(edited);
        app.history.end_merge();
        let set = preview_copy(&trim_set(&app)).into_owned();
        let (_, patches) = app.cached_material_patch(&set).unwrap();
        assert!(
            patches
                .iter()
                .all(|patch| !patch.channels.planes().base_color),
            "roughness reaches ORM alone: {patches:?}"
        );
        pump_until(&mut app, &context, cached_current);
        let (patched, textures) = cached(&app);
        assert!(patched && textures == full_bake(&app), "master edit");

        // Selecting Trim shows its current cache: no bake at all.
        let before = Arc::clone(&app.set_materials[&trim].textures);
        app.select_texture_set(trim);
        app.refresh_baked_preview();
        assert!(app.preview_job.is_none(), "the cached material is current");
        match &app.baked_preview.as_ref().unwrap().textures {
            PreviewTextures::Material(shown) => assert!(Arc::ptr_eq(shown, &before)),
            PreviewTextures::Single(_) => unreachable!("material preview expected"),
        }
        // ...and its edits from here patch that preview.
        app.append_stroke_point(glam::Vec2::new(0.6, 0.6));
        assert!(app.preview_patch(&app.preview_key()).is_some());
        app.finish_active_stroke();

        // Anything beyond layers (here every set at once) rebakes whole.
        app.select_texture_set(body);
        pump_until(&mut app, &context, cached_current);
        app.invalidate_all_sets();
        pump_until(&mut app, &context, cached_current);
        let (patched, textures) = cached(&app);
        assert!(!patched && textures == full_bake(&app));

        // A mask overlay tints its preview, which never stands in for the
        // set's material; leaving it shows the cached material, unbaked.
        app.select_texture_set(trim);
        app.refresh_baked_preview();
        let material = Arc::clone(&app.set_materials[&trim].textures);
        let layer = app.selected_layer.unwrap();
        app.set_solo(Some(inspector::Solo::Overlay(layer)));
        pump_until(&mut app, &context, |app| {
            app.preview_job.is_none()
                && app
                    .baked_preview
                    .as_ref()
                    .is_some_and(|preview| preview.key == app.preview_key())
        });
        assert!(Arc::ptr_eq(&app.set_materials[&trim].textures, &material));
        app.set_solo(None);
        app.refresh_baked_preview();
        assert!(
            app.preview_job.is_none(),
            "no bake after leaving the overlay"
        );
    }

    #[test]
    fn tc_0103_erase_mode_and_groups_work_through_the_app() {
        let mut app = SubstanceCraftApp::default();
        app.add_paint_layer();
        let paint = app.selected_layer.unwrap();
        app.paint_mode = PaintMode::Paint;
        app.append_stroke_point(glam::Vec2::new(0.4, 0.5));
        app.append_stroke_point(glam::Vec2::new(0.6, 0.5));
        app.finish_active_stroke();
        app.paint_mode = PaintMode::Erase;
        app.pen_pressure = Some(0.5);
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.append_stroke_point(glam::Vec2::new(0.55, 0.5));
        app.finish_active_stroke();
        let strokes = &app.selected_layer().unwrap().paint_strokes;
        assert_eq!(strokes.len(), 2);
        assert!(!strokes[0].erase && strokes[1].erase);
        assert_eq!(
            strokes[1].pressures,
            vec![0.5, 0.5],
            "pen pressure per point"
        );
        assert!(strokes[0].pressures.is_empty(), "mouse strokes carry none");

        // A group added above the paint layer; moving the paint layer up
        // and into the group below needs it to sit above the group first.
        app.add_group_layer();
        let group = app.selected_layer.unwrap();
        app.selected_layer = Some(paint);
        app.move_selected_layer(1);
        assert_eq!(app.group_below_selected(), Some(group));
        app.move_selected_into_group();
        let set = app.selected_texture_set_ref().unwrap();
        assert_eq!(set.slot_of(paint).unwrap().parent, Some(group));
        // Strokes cannot target a group.
        app.selected_layer = Some(group);
        app.paint_mode = PaintMode::Paint;
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        assert!(app.active_stroke.is_none());

        app.selected_layer = Some(paint);
        app.move_selected_out_of_group();
        let set = app.selected_texture_set_ref().unwrap();
        assert_eq!(set.slot_of(paint).unwrap().parent, None);
        app.undo();
        let set = app.selected_texture_set_ref().unwrap();
        assert_eq!(set.slot_of(paint).unwrap().parent, Some(group));
    }

    #[test]
    fn tc_0114_custom_export_presets_are_edited_saved_and_exported_through_the_app() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.add_fill_layer();
        small_sets(&mut app, 8);
        assert_eq!(app.selected_export_preset().name, DEFAULT_EXPORT_PRESET);

        // Duplicate the Unreal preset and keep only its normal map, packed
        // as a single-channel green (inverted) map.
        app.duplicate_export_preset("Unreal Engine");
        let draft = app.preset_draft.as_mut().unwrap();
        assert_eq!(draft.preset.name, "Unreal Engine copy");
        draft
            .preset
            .outputs
            .retain(|output| output.file_name == "T_$textureSet_N");
        draft.preset.outputs[0].layout = substancecraft_core::OutputLayout::Gray;
        draft.preset.outputs[0].slots[0] = draft.preset.outputs[0].slots[1];
        app.apply_preset_draft().unwrap();
        assert!(app.preset_draft.is_none());
        assert_eq!(app.export_preset, "Unreal Engine copy");
        assert!(app.is_custom_preset("Unreal Engine copy"));

        // A second copy gets a distinct name; a draft named like a built-in
        // is refused and stays open without touching the project.
        app.duplicate_export_preset("Unreal Engine");
        assert_eq!(
            app.preset_draft.as_ref().unwrap().preset.name,
            "Unreal Engine copy 2"
        );
        app.preset_draft.as_mut().unwrap().preset.name = "Separate Channels".to_owned();
        assert!(app.apply_preset_draft().is_err());
        assert!(app.preset_draft.is_some());
        assert_eq!(app.project.export_presets.len(), 1);
        app.preset_draft = None;

        let set = app.selected_texture_set_ref().unwrap().clone();
        let written = app
            .write_export(&set, &app.selected_export_preset(), folder.path())
            .unwrap();
        let stem = set.name.replace(' ', "_");
        assert_eq!(written, [folder.path().join(format!("T_{stem}_N.png"))]);
        let flat_green = load_texture_image(&written[0]).unwrap();
        assert_eq!(&flat_green.rgba8[..4], &[127, 127, 127, 255]);

        // Custom presets travel with the project.
        let project = folder.path().join("scene.sproject");
        save_project_file(&project, &app.project).unwrap();
        let loaded = load_project_file(&project).unwrap();
        assert_eq!(loaded.export_presets, app.project.export_presets);

        // An output with its own format and depth writes that way whatever
        // the export format is.
        app.edit_export_preset("Unreal Engine copy");
        let draft = app.preset_draft.as_mut().unwrap();
        draft.preset.outputs[0].bit_depth = substancecraft_core::BitDepth::Sixteen;
        draft.preset.max_resolution = Some(4096);
        app.apply_preset_draft().unwrap();
        let written = app
            .write_export(&set, &app.selected_export_preset(), folder.path())
            .unwrap();
        // PNG header: bit depth at byte 24, colour type (0 = gray) at 25.
        let png = std::fs::read(&written[0]).unwrap();
        assert_eq!((png[24], png[25]), (16, 0));

        // Preset edits undo and redo like any other edit.
        let edited = app.project.export_presets.clone();
        app.undo();
        assert_eq!(
            app.project.export_presets[0].outputs[0].bit_depth,
            substancecraft_core::BitDepth::Eight
        );
        app.redo();
        assert_eq!(app.project.export_presets, edited);

        // Deleting a preset a set remembered sends that set back to the
        // app-wide choice, and undo restores both.
        let set_id = set.id;
        app.remember_export_for_selected(true);
        app.delete_export_preset("Unreal Engine copy");
        assert!(app.project.export_presets.is_empty());
        assert_eq!(app.selected_export_preset().name, DEFAULT_EXPORT_PRESET);
        let remembered = |app: &SubstanceCraftApp| {
            app.project
                .texture_sets
                .iter()
                .find(|candidate| candidate.id == set_id)
                .and_then(|candidate| candidate.export.clone())
        };
        assert_eq!(remembered(&app), None);
        assert!(app.project.validate().is_ok());
        app.undo();
        assert_eq!(app.project.export_presets, edited);
        assert_eq!(
            remembered(&app).map(|export| export.preset),
            Some("Unreal Engine copy".to_owned())
        );
    }

    #[test]
    fn tc_0121_edited_bitmaps_reload_into_only_the_sets_that_use_them() {
        let context = egui::Context::default();
        let folder = tempfile::tempdir().unwrap();
        let bitmap = folder.path().join("grunge.png");
        let save = |width: u32, rgba: [u8; 4]| {
            write_texture(
                &bitmap,
                &BakedTexture {
                    width,
                    height: 1,
                    rgba8: rgba.repeat(width as usize),
                },
            )
            .unwrap();
        };
        let first_pixel =
            |app: &SubstanceCraftApp, id| app.resource_images[&id].rgba8[..4].to_vec();
        save(2, [0, 255, 0, 255]);

        // Set A fills with the bitmap; set B (selected) does not use it.
        let mut app = SubstanceCraftApp::default();
        let resource = app.add_resource(&bitmap).unwrap();
        app.add_fill_layer();
        let set_a = app.selected_texture_set.unwrap();
        if let Some(layer) = app.selected_layer_mut() {
            layer
                .fill
                .get_or_insert_with(LayerFill::default)
                .textures
                .base_color = Some(resource);
        }
        app.add_texture_set();
        app.add_fill_layer();
        let set_b = app.selected_texture_set.unwrap();
        let (revision_a, revision_b) = (app.set_revision(set_a), app.set_revision(set_b));
        app.baked_preview = Some(BakedPreview {
            key: app.preview_key(),
            textures: PreviewTextures::Single(Arc::new(BakedTexture {
                width: 1,
                height: 1,
                rgba8: vec![0; 4],
            })),
            set_revision: 0,
            patch: None,
        });

        // Auto: an outside save reloads the pixels and dirties only set A;
        // set B's shown preview stays current instead of rebaking.
        save(3, [255, 0, 0, 255]);
        app.poll_resource_files(&context);
        assert_eq!(first_pixel(&app, resource), [255, 0, 0, 255]);
        assert_ne!(app.set_revision(set_a), revision_a);
        assert_eq!(app.set_revision(set_b), revision_b);
        assert_eq!(
            app.baked_preview.as_ref().map(|preview| preview.key),
            Some(app.preview_key())
        );
        assert_eq!(app.project_error, None);
        app.invalidate_sets(&BTreeSet::from([set_b]));
        assert_ne!(
            app.baked_preview.as_ref().map(|preview| preview.key),
            Some(app.preview_key()),
            "a change to the selected set does rebake"
        );

        // A file caught mid-save keeps the last good pixels and is reported.
        std::fs::write(&bitmap, b"half-written").unwrap();
        app.resource_watcher.expire();
        app.poll_resource_files(&context);
        assert_eq!(first_pixel(&app, resource), [255, 0, 0, 255]);
        assert!(app.project_error.as_deref().unwrap().contains("grunge"));

        // Ask: changes wait for confirmation, then reload.
        app.resource_watcher.mode = resource_watch::ReloadMode::Ask;
        save(4, [0, 0, 255, 255]);
        app.resource_watcher.expire();
        app.poll_resource_files(&context);
        assert_eq!(first_pixel(&app, resource), [255, 0, 0, 255]);
        assert_eq!(app.pending_reload_names(), ["grunge"]);
        app.reload_changed_resources(&[resource]);
        assert_eq!(first_pixel(&app, resource), [0, 0, 255, 255]);
        assert!(app.pending_reload_names().is_empty());

        // Off: nothing is checked.
        app.resource_watcher.mode = resource_watch::ReloadMode::Off;
        save(5, [9, 9, 9, 255]);
        app.resource_watcher.expire();
        app.poll_resource_files(&context);
        assert_eq!(first_pixel(&app, resource), [0, 0, 255, 255]);
    }

    #[test]
    fn tc_0372_re_exported_high_poly_sources_reload_and_rebake() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("high.gltf");
        let write = |preset: PrimitivePreset| {
            substancecraft_api::write_gltf(&path, &preset.build_mesh(), 30.0, &[]).unwrap();
        };
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.mesh_map_settings.resolution = 32;
        app.mesh_map_settings.ao_samples = 2;
        write(PrimitivePreset::Sphere);
        app.high_poly = Some(load_high_poly(&path, 1.0).unwrap());
        app.high_poly_changed();
        let vertices =
            |app: &SubstanceCraftApp| app.high_poly.as_ref().unwrap().geometry.vertices.len();
        let sphere = vertices(&app);

        // Auto: a re-export reloads the high poly and starts a rebake.
        let revision = app.high_poly_revision;
        write(PrimitivePreset::Cube);
        app.resource_watcher.expire();
        app.poll_resource_files(&context);
        assert_ne!(vertices(&app), sphere);
        assert_ne!(app.high_poly_revision, revision);
        assert!(app.mesh_map_job.is_some(), "rebaking from the new source");
        app.cancel_mesh_map_job();

        // Ask: the change waits in the reload prompt.
        app.resource_watcher.mode = resource_watch::ReloadMode::Ask;
        write(PrimitivePreset::Sphere);
        app.resource_watcher.expire();
        app.poll_resource_files(&context);
        assert_ne!(vertices(&app), sphere);
        assert_eq!(app.pending_reload_names(), ["High poly high.gltf"]);
        app.reload_pending();
        assert_eq!(vertices(&app), sphere);
        assert!(app.pending_reload_names().is_empty());
        app.cancel_mesh_map_job();

        // A file caught mid-export keeps the previous high poly.
        app.resource_watcher.mode = resource_watch::ReloadMode::Auto;
        std::fs::write(&path, b"{ half").unwrap();
        app.resource_watcher.expire();
        app.poll_resource_files(&context);
        assert_eq!(vertices(&app), sphere);
        assert!(
            app.mesh_map_error
                .as_deref()
                .unwrap()
                .contains("keeping the previous")
        );
    }

    #[test]
    fn tc_0123_export_preflight_blocks_errors_and_asks_before_overwriting() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.add_fill_layer();
        small_sets(&mut app, 8);
        let preset = app.selected_export_preset();

        // A missing bitmap blocks the export before any folder is chosen.
        let mut broken = app.selected_texture_set_ref().unwrap().clone();
        if let Some(fill) = broken.layers[0].fill.as_mut() {
            fill.textures.base_color = Some(ResourceId(42));
        }
        let error = blocking_export_error(&app.export_issues(&broken, &preset)).unwrap();
        assert!(
            error.starts_with("Export blocked: Bitmap resource 42"),
            "{error}"
        );

        // A fill with no mesh maps only warns (flat normal, white AO).
        let set = app.selected_texture_set_ref().unwrap().clone();
        let issues = app.export_issues(&set, &preset);
        assert!(!issues.is_empty());
        assert_eq!(blocking_export_error(&issues), None);

        // First export writes straight away; a second asks first.
        app.begin_export(set.clone(), preset.clone(), folder.path().to_path_buf());
        assert!(app.overwrite_prompt.is_none());
        assert_eq!(std::fs::read_dir(folder.path()).unwrap().count(), 3);
        let base = folder
            .path()
            .join(format!("{}_BaseColor.png", set.name.replace(' ', "_")));
        std::fs::write(&base, b"stale").unwrap();

        app.begin_export(set.clone(), preset.clone(), folder.path().to_path_buf());
        assert_eq!(app.overwrite_prompt.as_ref().unwrap().existing.len(), 3);
        app.resolve_overwrite_prompt(false);
        assert!(app.overwrite_prompt.is_none());
        assert_eq!(
            std::fs::read(&base).unwrap(),
            b"stale",
            "cancel leaves files"
        );

        app.begin_export(set, preset, folder.path().to_path_buf());
        app.resolve_overwrite_prompt(true);
        assert!(
            load_texture_image(&base).is_ok(),
            "replaced with a real image"
        );
        assert_eq!(app.bake_error, None);
    }

    #[test]
    fn tc_0163_symmetry_seams_paths_and_eyedropper_work_through_the_app() {
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.add_paint_layer();
        small_sets(&mut app, 16);
        app.set_paint_mode(PaintMode::Paint);
        let cube = PrimitivePreset::Cube.build_mesh();
        let hit_at = |point: glam::Vec3| {
            let hit = cube.closest_surface_point(point).unwrap();
            TriangleHit {
                triangle: hit.triangle,
                uv: Some(hit.uv),
                position: hit.position,
            }
        };
        let strokes = |app: &SubstanceCraftApp| app.selected_layer().unwrap().paint_strokes.len();

        // Mirror across X: one drag makes a stroke on each side, undone
        // together.
        app.symmetry = Some(substancecraft_core::Symmetry::new(
            substancecraft_core::Axis::X,
        ));
        app.paint_hit(hit_at(glam::Vec3::new(0.3, 0.0, 0.5)));
        app.paint_hit(hit_at(glam::Vec3::new(0.35, 0.05, 0.5)));
        app.finish_active_stroke();
        assert_eq!(strokes(&app), 2);
        let layer = app.selected_layer().unwrap();
        assert!(layer.paint_strokes[1].points[0].x < layer.paint_strokes[0].points[0].x);
        app.undo();
        assert_eq!(strokes(&app), 0);

        // Four radial copies around Y paint all four side faces in one step.
        app.symmetry = Some(substancecraft_core::Symmetry::radial(
            substancecraft_core::Axis::Y,
            4,
        ));
        app.paint_hit(hit_at(glam::Vec3::new(0.1, 0.0, 0.5)));
        app.paint_hit(hit_at(glam::Vec3::new(0.15, 0.05, 0.5)));
        app.finish_active_stroke();
        assert_eq!(strokes(&app), 4);
        app.undo();
        assert_eq!(strokes(&app), 0);

        // Crossing from the front face onto the top face restarts the
        // stroke in the new UV island; still one undo step.
        app.symmetry = None;
        app.paint_hit(hit_at(glam::Vec3::new(0.0, 0.4, 0.5)));
        app.paint_hit(hit_at(glam::Vec3::new(0.0, 0.5, 0.4)));
        app.finish_active_stroke();
        assert_eq!(strokes(&app), 2);
        app.undo();
        assert_eq!(strokes(&app), 0);
        app.redo();
        assert_eq!(strokes(&app), 2);

        // A placed path commits as one stroke that keeps its control points.
        let place = |points: &[glam::Vec3]| {
            points
                .iter()
                .map(|point| {
                    let hit = hit_at(*point);
                    (hit.uv.unwrap(), egui::Pos2::ZERO, hit.position)
                })
                .collect::<Vec<_>>()
        };
        app.path_controls = Some(place(&[
            glam::Vec3::new(-0.3, -0.2, 0.5),
            glam::Vec3::new(0.0, 0.1, 0.5),
            glam::Vec3::new(0.3, -0.2, 0.5),
        ]));
        app.commit_path(false);
        assert!(app.path_controls.is_none());
        let path_stroke = app
            .selected_layer()
            .unwrap()
            .paint_strokes
            .last()
            .unwrap()
            .clone();
        let path = path_stroke.path.as_ref().unwrap();
        assert_eq!(path.controls.len(), 3);
        assert_eq!(path.surface.len(), 3);
        assert_eq!(
            path_stroke.points.len(),
            2 * substancecraft_core::SURFACE_PATH_SAMPLES_PER_SEGMENT + 1
        );
        app.undo();
        assert_eq!(strokes(&app), 2);

        // A path running from the front face over the top follows the
        // surface across the seam: one stroke per face, one undo step.
        app.path_controls = Some(place(&[
            glam::Vec3::new(0.0, 0.2, 0.5),
            glam::Vec3::new(0.0, 0.5, 0.2),
        ]));
        app.commit_path(false);
        assert_eq!(strokes(&app), 4);
        app.undo();
        assert_eq!(strokes(&app), 2);

        // The eyedropper copies every channel of a fill into the brush.
        app.add_fill_layer();
        if let Some(fill) = app
            .selected_layer_mut()
            .and_then(|layer| layer.fill.as_mut())
        {
            fill.color = [10, 200, 30, 255];
            fill.roughness = 0.9;
            fill.metallic = 1.0;
        }
        app.picking_material = true;
        app.pick_material_at(glam::Vec2::new(0.5, 0.5));
        assert!(!app.picking_material);
        assert_eq!(app.brush_color, egui::Color32::from_rgb(10, 200, 30));
        assert!((app.brush_roughness - 0.9).abs() < 0.01);
        assert_eq!(app.brush_metallic, 1.0);
    }

    #[test]
    fn tc_0162_lazy_stabilization_records_the_trailing_brush_position() {
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.add_paint_layer();
        small_sets(&mut app, 32);
        app.set_paint_mode(PaintMode::Paint);
        app.stroke_stabilizer = substancecraft_core::LazyStabilizer::new(10.0);

        let mesh = PrimitivePreset::Cube.build_mesh();
        let projection = ViewProjection {
            pose: BoneTransform::IDENTITY,
            camera_rotation: glam::Quat::IDENTITY,
            posed_center: glam::Vec3::ZERO,
            center: egui::pos2(100.0, 100.0),
            scale: 100.0,
            pan: glam::Vec2::ZERO,
        };
        let projected = projection.project_mesh(&mesh);
        let first = app.stabilized_cursor(egui::pos2(100.0, 100.0)).unwrap();
        let first_hit = pick_triangle(&mesh, &projected, first, None).unwrap();
        app.paint_hit(first_hit);

        assert_eq!(
            app.stabilized_cursor(egui::pos2(106.0, 108.0)),
            None,
            "cursor jitter within the lazy radius does not add a dab"
        );
        let trailing = app.stabilized_cursor(egui::pos2(130.0, 100.0)).unwrap();
        assert_eq!(trailing, egui::pos2(120.0, 100.0));
        app.paint_hit(pick_triangle(&mesh, &projected, trailing, None).unwrap());
        app.finish_active_stroke();

        let recorded = &app.selected_layer().unwrap().paint_strokes[0].points;
        let expected = pick_triangle(&mesh, &projected, trailing, None)
            .unwrap()
            .uv
            .unwrap();
        let raw = pick_triangle(&mesh, &projected, egui::pos2(130.0, 100.0), None)
            .unwrap()
            .uv
            .unwrap();
        assert_eq!(recorded.last(), Some(&expected));
        assert_ne!(recorded.last(), Some(&raw));
        assert_eq!(recorded.len(), 2);
    }

    #[test]
    fn tc_0180_mesh_map_staleness_reports_the_changed_input() {
        let mut app = SubstanceCraftApp::default();
        app.mesh_maps_key = Some(app.bake_key());
        assert_eq!(app.mesh_map_stale_reason(), None);

        app.import_scale = 2.0;
        assert_eq!(app.mesh_map_stale_reason(), Some("Scale changed"));

        app.mesh_maps_key = Some(app.bake_key());
        app.mesh_smoothing.angle_degrees += 1.0;
        assert_eq!(app.mesh_map_stale_reason(), Some("Smoothing changed"));

        app.mesh_maps_key = Some(app.bake_key());
        app.high_poly_revision += 1;
        assert_eq!(
            app.mesh_map_stale_reason(),
            Some("High-poly source changed")
        );

        app.mesh_maps_key = Some(app.bake_key());
        app.mesh_map_settings.ao_samples += 1;
        assert_eq!(app.mesh_map_stale_reason(), Some("Bake settings changed"));

        app.mesh_maps_key = Some(app.bake_key());
        app.skew_map_revision += 1;
        assert_eq!(app.mesh_map_stale_reason(), Some("Bake settings changed"));

        app.mesh_maps_key = Some(app.bake_key());
        app.mesh_revision += 1;
        assert_eq!(app.mesh_map_stale_reason(), Some("Mesh changed"));
    }

    #[test]
    fn tc_0156_flatten_materials_and_smart_materials_work_through_the_app() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.add_fill_layer();
        small_sets(&mut app, 8);
        let source = app.selected_layer.unwrap();
        if let Some(layer) = app.selected_layer_mut() {
            layer.opacity = 0.5;
        }
        app.sync_layer_history(&egui::Context::default());
        let set_id = app.selected_texture_set.unwrap();
        let before = app.selected_texture_set_ref().unwrap().clone();
        let original = before
            .bake_channel(
                MaterialChannel::BaseColor,
                substancecraft_core::BakeContext::default(),
            )
            .unwrap();

        // Flatten: bitmaps on disk, a fill above, the source hidden.
        let flattened = app.flatten_selected_into(folder.path()).unwrap();
        let set = app.selected_texture_set_ref().unwrap().clone();
        assert_eq!(set.layers.len(), 2);
        assert_eq!(set.layers[1].id, flattened);
        assert!(!set.layer(source).unwrap().visible);
        assert_eq!(app.project.resources.len(), 3, "color, roughness, metallic");
        let rebuilt = app
            .bake_inputs(set_id)
            .bake_channel(&set, MaterialChannel::BaseColor)
            .unwrap();
        assert_eq!(rebuilt, original, "flattened layer composites identically");
        app.undo();
        assert_eq!(
            app.selected_texture_set_ref().unwrap().layers,
            before.layers
        );

        // Materials apply to the selected layer and can be saved back.
        app.selected_layer = Some(source);
        let gold = substancecraft_core::MaterialResource::built_in()
            .into_iter()
            .find(|material| material.name == "Gold")
            .unwrap();
        app.apply_material(&gold);
        assert_eq!(app.selected_layer().unwrap().fill, Some(gold.fill));
        app.save_layer_as_material();
        app.save_layer_as_material();
        let names: Vec<&str> = app
            .project
            .materials
            .iter()
            .map(|m| m.name.as_str())
            .collect();
        assert_eq!(names.len(), 2);
        assert_ne!(names[0], names[1], "saved names are unique");

        // Smart materials insert with fresh ids and save to the project.
        let worn = substancecraft_core::SmartMaterial::built_in().remove(0);
        app.insert_smart_material(&worn);
        let inserted = app.selected_layer().unwrap().clone();
        assert!(inserted.is_group());
        assert_eq!(inserted.walk().len(), 4);
        app.project.validate().unwrap();
        app.save_layer_as_smart_material();
        assert_eq!(app.project.smart_materials.len(), 1);
        app.project.validate().unwrap();
        let file = folder.path().join("worn.scsmart");
        substancecraft_api::save_smart_material(&file, &app.project.smart_materials[0]).unwrap();
        assert_eq!(
            substancecraft_api::load_smart_material(&file).unwrap(),
            app.project.smart_materials[0]
        );
    }

    #[test]
    fn tc_0271_material_instances_follow_their_master_across_sets_through_the_app() {
        let context = egui::Context::default();
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.load_imported_mesh(&write_two_material_gltf(folder.path()));
        small_sets(&mut app, 8);
        let (body, trim) = (
            app.project.texture_sets[0].id,
            app.project.texture_sets[1].id,
        );
        let fill = |project: &MaterialProject, set: TextureSetId, layer: LayerId| {
            project
                .texture_sets
                .iter()
                .find(|candidate| candidate.id == set)
                .and_then(|candidate| candidate.layer(layer))
                .and_then(|layer| layer.fill)
                .unwrap()
        };

        // A Body fill becomes a master; a Trim fill links to it.
        app.select_texture_set(body);
        app.add_fill_layer();
        let body_layer = app.selected_layer.unwrap();
        app.make_selected_layer_master();
        let master = app.editing_material.unwrap();
        assert_eq!(app.project.materials.len(), 1);
        app.select_texture_set(trim);
        app.add_fill_layer();
        let trim_layer = app.selected_layer.unwrap();
        let material = app.project.material(master).unwrap().clone();
        app.link_selected_layer(&material);
        assert_eq!(app.project.material_instances(master).len(), 2);
        app.project.validate().unwrap();

        // Dragging a master slider updates both sets, marks the other set
        // stale, and is one undo step.
        let body_revision = app.set_revision(body);
        for roughness in [0.6, 0.7, 0.8] {
            let mut edited = app.project.material(master).unwrap().clone();
            edited.fill.roughness = roughness;
            app.edit_project_material(edited);
        }
        app.history.end_merge();
        assert_ne!(app.set_revision(body), body_revision);
        assert_eq!(fill(&app.project, body, body_layer).roughness, 0.8);
        assert_eq!(fill(&app.project, trim, trim_layer).roughness, 0.8);

        // An inspector edit of the instance becomes an override.
        app.sync_layer_history(&context);
        if let Some(fill) = app
            .selected_layer_mut()
            .and_then(|layer| layer.fill.as_mut())
        {
            fill.metallic = 0.7;
        }
        app.sync_layer_history(&context);
        let link = app.selected_layer().unwrap().material_link.clone().unwrap();
        assert_eq!(
            link.overrides,
            BTreeSet::from([substancecraft_core::InstanceField::Metallic])
        );
        // The settled snapshot already includes the override: no extra step.
        app.sync_layer_history(&context);
        let mut probe = app.history.clone();
        let mut scratch = app.project.clone();
        assert!(probe.undo(&mut scratch));
        assert_eq!(
            fill(&scratch, trim, trim_layer).metallic,
            material.fill.metallic
        );

        // Master edits pass the override by; resetting it takes the master.
        let mut edited = app.project.material(master).unwrap().clone();
        edited.fill.metallic = 0.2;
        app.edit_project_material(edited);
        app.history.end_merge();
        assert_eq!(fill(&app.project, trim, trim_layer).metallic, 0.7);
        assert_eq!(fill(&app.project, body, body_layer).metallic, 0.2);
        app.reset_instance_override(substancecraft_core::InstanceField::Metallic);
        assert_eq!(fill(&app.project, trim, trim_layer).metallic, 0.2);
        app.undo();
        assert_eq!(fill(&app.project, trim, trim_layer).metallic, 0.7);
        app.undo();
        assert_eq!(
            fill(&app.project, body, body_layer).metallic,
            material.fill.metallic
        );
        app.project.validate().unwrap();

        // Deleting the master keeps the values as plain fills; undo relinks.
        app.delete_project_material(master);
        assert!(app.project.materials.is_empty());
        assert_eq!(app.editing_material, None);
        assert_eq!(fill(&app.project, body, body_layer).roughness, 0.8);
        assert!(app.project.material_instances(master).is_empty());
        app.project.validate().unwrap();
        app.undo();
        assert_eq!(app.project.material_instances(master).len(), 2);

        // The links survive a save and reload.
        let path = folder.path().join("instances.sproject");
        save_project_file(&path, &app.project).unwrap();
        let reloaded = load_project_file(&path).unwrap();
        assert_eq!(reloaded.material_instances(master).len(), 2);
    }

    #[test]
    fn tc_0275_projected_strokes_paint_through_the_viewport_camera() {
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        app.mesh_map_settings.resolution = 64;
        app.mesh_map_settings.ao_samples = 1;
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.add_paint_layer();
        small_sets(&mut app, 64);
        let viewport = egui::Rect::from_min_size(egui::pos2(30.0, 20.0), egui::vec2(400.0, 300.0));
        app.viewport_camera.orbit(glam::Vec2::new(40.0, -25.0));
        let projection = app.view_projection(viewport);
        let camera = projection.stroke_camera(viewport);

        // The stroke camera reproduces the viewport's own projection.
        for vertex in &app.mesh_geometry.as_ref().unwrap().vertices {
            let expected = projection.project(vertex.position);
            let projected = camera.project(vertex.position);
            assert!((projected.x - (expected.screen.x - viewport.min.x)).abs() < 1e-3);
            assert!((projected.y - (expected.screen.y - viewport.min.y)).abs() < 1e-3);
            assert!((projected.z + expected.view.z).abs() < 1e-4);
        }

        // Off by default, and ID painting stays in UV space.
        assert!(!app.projects_strokes());
        app.projection_painting = true;
        app.paint_mode = PaintMode::Id;
        assert!(!app.projects_strokes());
        app.paint_mode = PaintMode::Paint;
        assert!(app.projects_strokes());

        // A drag across the middle of the view is one projected stroke.
        app.brush_color = egui::Color32::from_rgb(255, 0, 0);
        app.brush_radius = 40.0;
        let middle = glam::Vec2::new(200.0, 150.0);
        app.camera_stroke_point(middle, camera);
        app.camera_stroke_point(middle + glam::Vec2::new(30.0, 0.0), camera);
        app.camera_stroke_point(middle + glam::Vec2::new(30.2, 0.0), camera);
        app.finish_active_stroke();
        let layer = app.selected_layer().unwrap();
        assert_eq!(layer.paint_strokes.len(), 1);
        let stroke = &layer.paint_strokes[0];
        assert_eq!(stroke.camera, Some(camera));
        assert_eq!(stroke.points.len(), 2, "sub-pixel moves add no point");
        assert_eq!(stroke.radius_pixels, 40.0);
        app.project.validate().unwrap();
        assert!(app.selected_texture_set_ref().unwrap().requires_mesh_maps());

        // The mesh maps bake in the background; the preview then shows the
        // paint on the surface under the stroke.
        app.invalidate_preview();
        app.refresh_baked_preview();
        pump_until(&mut app, &context, |app| {
            app.mesh_maps_fresh()
                && app.baked_preview.as_ref().map(|preview| preview.key) == Some(app.preview_key())
        });
        let set = app.selected_texture_set_ref().unwrap().clone();
        let base = app
            .bake_inputs(set.id)
            .bake_channel(&set, MaterialChannel::BaseColor)
            .unwrap();
        let painted = base
            .rgba8
            .chunks(4)
            .filter(|texel| texel[3] > 128 && texel[0] > 200)
            .count();
        assert!(painted > 100, "the stroke landed on the cube: {painted}");

        // A second drag after orbiting starts a new stroke through the new
        // view; undo removes both, one drag at a time.
        app.viewport_camera.orbit(glam::Vec2::new(-80.0, 0.0));
        let turned = app.view_projection(viewport).stroke_camera(viewport);
        app.camera_stroke_point(middle, turned);
        app.finish_active_stroke();
        assert_eq!(app.selected_layer().unwrap().paint_strokes.len(), 2);
        assert_eq!(
            app.selected_layer().unwrap().paint_strokes[1].camera,
            Some(turned)
        );
        app.undo();
        app.undo();
        assert!(app.selected_layer().unwrap().paint_strokes.is_empty());

        // Mask mode projects mask strokes.
        app.paint_mode = PaintMode::Mask;
        app.camera_stroke_point(middle, camera);
        app.finish_active_stroke();
        assert_eq!(
            app.selected_layer().unwrap().mask_strokes[0].camera,
            Some(camera)
        );
    }

    #[test]
    fn tc_0277_id_generator_picks_its_colour_from_the_model_as_one_undo_step() {
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        app.add_paint_layer();
        small_sets(&mut app, 32);
        // Paint a green ID patch.
        app.paint_mode = PaintMode::Id;
        app.brush_color = egui::Color32::from_rgb(0, 200, 0);
        app.brush_radius = 6.0;
        app.append_stroke_point(glam::Vec2::new(0.5, 0.5));
        app.finish_active_stroke();
        // A fill selected by an ID generator, then the colour picked.
        app.add_fill_layer();
        if let Some(layer) = app.selected_layer_mut() {
            layer.mask_generator = Some(MaskGenerator::IdColor(
                substancecraft_core::IdSelection::DEFAULT,
            ));
        }
        app.sync_layer_history(&context);
        app.sync_layer_history(&context);
        app.picking_id_color = true;
        app.pick_id_color_at(glam::Vec2::new(0.1, 0.1));
        assert!(!app.picking_id_color, "a click ends picking");
        let selection = |app: &SubstanceCraftApp| match app.selected_layer().unwrap().mask_generator
        {
            Some(MaskGenerator::IdColor(selection)) => selection.color,
            _ => [0, 0, 0],
        };
        assert_eq!(
            selection(&app),
            [255, 0, 0],
            "unpainted texels change nothing"
        );
        app.pick_id_color_at(glam::Vec2::new(0.5, 0.5));
        assert_eq!(selection(&app), [0, 200, 0]);
        app.sync_layer_history(&context);
        // No mesh maps are needed; the fill shows only on the green patch.
        let set = app.selected_texture_set_ref().unwrap().clone();
        assert!(!set.requires_mesh_maps());
        let base = app
            .bake_inputs(set.id)
            .bake_channel(&set, MaterialChannel::BaseColor)
            .unwrap();
        assert!(base.sample_uv(glam::Vec2::new(0.5, 0.5)).unwrap()[3] > 0);
        assert_eq!(base.sample_uv(glam::Vec2::new(0.1, 0.1)).unwrap()[3], 0);
        // The pick is one undoable settings edit.
        app.undo();
        assert_eq!(selection(&app), [255, 0, 0]);
    }

    #[test]
    fn tc_0279_smart_masks_save_apply_and_delete_through_the_app() {
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        app.add_fill_layer();
        if let Some(layer) = app.selected_layer_mut() {
            layer.name = "Rust".to_owned();
            layer.mask_generator = Some(MaskGenerator::Dirt(Dirt::DEFAULT));
            layer.mask_filters = vec![substancecraft_core::LayerFilter::Invert];
        }
        app.sync_layer_history(&context);
        app.save_layer_as_smart_mask();
        app.save_layer_as_smart_mask();
        let names: Vec<&str> = app
            .project
            .smart_masks
            .iter()
            .map(|mask| mask.name.as_str())
            .collect();
        assert_eq!(names, ["Rust mask", "Rust mask 2"]);
        assert_eq!(
            app.smart_mask_library().len(),
            2 + substancecraft_core::SmartMask::built_in().len()
        );

        // Applying one to another layer takes its setup as one undo step.
        app.add_paint_layer();
        app.sync_layer_history(&context);
        let mask = app.project.smart_masks[0].clone();
        if let Some(layer) = app.selected_layer_mut() {
            mask.apply_to(layer);
        }
        app.sync_layer_history(&context);
        let layer = app.selected_layer().unwrap();
        assert_eq!(
            layer.mask_generator,
            Some(MaskGenerator::Dirt(Dirt::DEFAULT))
        );
        assert_eq!(layer.mask_filters.len(), 1);
        app.undo();
        assert_eq!(app.selected_layer().unwrap().mask_generator, None);

        // Deleting is undoable too, and the masks save with the project.
        app.delete_smart_mask(1);
        assert_eq!(app.project.smart_masks.len(), 1);
        app.undo();
        assert_eq!(app.project.smart_masks.len(), 2);
        app.project.validate().unwrap();
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("masks.sproject");
        save_project_file(&path, &app.project).unwrap();
        assert_eq!(load_project_file(&path).unwrap().smart_masks.len(), 2);
    }

    #[test]
    fn tc_0286_evolution_controls_are_recorded_and_dragged_through_the_app() {
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp::default();
        app.add_group_layer();
        let group = app.selected_layer.unwrap();
        app.add_fill_layer();
        let fill = app.selected_layer.unwrap();
        // Recording on the group: start, change a child, finish.
        app.selected_layer = Some(group);
        app.sync_layer_history(&context);
        app.start_evolution_recording();
        app.finish_evolution_recording();
        assert!(app.project_error.take().is_some(), "nothing changed yet");
        assert!(app.selected_layer().unwrap().evolutions.is_empty());
        app.start_evolution_recording();
        app.selected_layer = Some(fill);
        app.sync_layer_history(&context);
        if let Some(fill) = app
            .selected_layer_mut()
            .and_then(|layer| layer.fill.as_mut())
        {
            fill.roughness = 0.9;
        }
        app.sync_layer_history(&context);
        app.selected_layer = Some(group);
        app.sync_layer_history(&context);
        app.evolution_name = "Rough".to_owned();
        app.finish_evolution_recording();
        app.sync_layer_history(&context);
        let control = app.selected_layer().unwrap().evolutions[0].clone();
        assert_eq!(control.name, "Rough");
        assert_eq!(control.amount, 1.0);
        let roughness = |app: &SubstanceCraftApp| {
            app.selected_texture_set_ref()
                .unwrap()
                .layer(fill)
                .unwrap()
                .fill
                .unwrap()
                .roughness
        };

        // Dragging the slider moves the child; the drag is one undo step.
        for amount in [0.8, 0.4, 0.0] {
            app.evolve_selected_layer(0, amount);
        }
        app.history.end_merge();
        assert_eq!(roughness(&app), 0.5);
        assert_eq!(app.selected_layer().unwrap().evolutions[0].amount, 0.0);
        app.undo();
        assert_eq!(roughness(&app), 0.9);
        assert_eq!(app.selected_layer().unwrap().evolutions[0].amount, 1.0);
        app.undo();
        assert!(
            app.selected_layer().unwrap().evolutions.is_empty(),
            "recording the control was its own step"
        );
        app.project.validate().unwrap();
    }

    #[test]
    fn tc_0294_widgets_are_named_for_assistive_technology_and_work_by_keyboard() {
        let context = egui::Context::default();
        context.enable_accesskit();
        let mut on = false;
        let mut clicks = 0;
        let frame = |context: &egui::Context,
                     events: Vec<egui::Event>,
                     on: &mut bool,
                     clicks: &mut i32| {
            let input = egui::RawInput {
                events,
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                ..egui::RawInput::default()
            };
            let mut output = context.run_ui(input, |ui| {
                if crate::widgets::pill_button(ui, None, "Export textures", false, false).clicked()
                {
                    *clicks += 1;
                }
                crate::widgets::chip(ui, "Gold", true);
                crate::widgets::icon_button(ui, crate::icons::Icon::Trash, true, "Delete layer");
                crate::widgets::property_row(ui, "Invert mask", |ui| {
                    crate::widgets::toggle(ui, on)
                });
                let mut value = 0.5_f32;
                crate::widgets::property_row(ui, "Roughness", |ui| {
                    let response = ui.add(egui::Slider::new(&mut value, 0.0..=1.0));
                    crate::widgets::label_from_row(ui, &response, egui::WidgetType::Slider);
                });
            });
            // Headless: no renderer takes the texture updates.
            output.textures_delta.clear();
            output
        };
        let output = frame(&context, Vec::new(), &mut on, &mut clicks);
        let update = output.platform_output.accesskit_update.unwrap();
        let labels: Vec<String> = update
            .nodes
            .iter()
            .filter_map(|(_, node)| node.label().map(str::to_owned))
            .collect();
        for expected in [
            "Export textures",
            "Gold",
            "Delete layer",
            "Invert mask",
            "Roughness",
        ] {
            assert!(
                labels.iter().any(|label| label == expected),
                "{expected} is named: {labels:?}"
            );
        }

        // Tab moves focus to the first button and Enter presses it; more
        // Tabs reach the toggle, which Space flips.
        let key = |key: egui::Key, pressed: bool| egui::Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let press = |key_code| vec![key(key_code, true), key(key_code, false)];
        frame(&context, press(egui::Key::Tab), &mut on, &mut clicks);
        frame(&context, press(egui::Key::Enter), &mut on, &mut clicks);
        frame(&context, Vec::new(), &mut on, &mut clicks);
        assert_eq!(clicks, 1, "Enter presses the focused button");
        for _ in 0..3 {
            frame(&context, press(egui::Key::Tab), &mut on, &mut clicks);
        }
        frame(&context, press(egui::Key::Space), &mut on, &mut clicks);
        frame(&context, Vec::new(), &mut on, &mut clicks);
        assert!(on, "Space flips the focused toggle");
    }

    #[test]
    fn tc_0296_the_viewport_camera_moves_by_keyboard() {
        let context = egui::Context::default();
        let mut camera = ViewportCamera::default();
        let start = camera;
        let key = |key: egui::Key, modifiers: egui::Modifiers| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        };
        let step = |camera: &mut ViewportCamera,
                    mut events: Vec<egui::Event>,
                    modifiers: egui::Modifiers| {
            events.insert(0, egui::Event::ModifiersChanged(modifiers));
            let mut moved = false;
            let mut output = context.run_ui(
                egui::RawInput {
                    events,
                    ..egui::RawInput::default()
                },
                |ui| {
                    moved = ui.ctx().input(|input| keyboard_camera(camera, input));
                },
            );
            output.textures_delta.clear();
            moved
        };
        assert!(!step(&mut camera, Vec::new(), egui::Modifiers::NONE));
        assert!(step(
            &mut camera,
            vec![key(egui::Key::ArrowRight, egui::Modifiers::NONE)],
            egui::Modifiers::NONE
        ));
        assert_ne!(camera.yaw, start.yaw, "arrows orbit");
        let orbited = camera;
        assert!(step(
            &mut camera,
            vec![key(egui::Key::ArrowUp, egui::Modifiers::SHIFT)],
            egui::Modifiers::SHIFT
        ));
        assert_eq!(camera.yaw, orbited.yaw);
        assert_ne!(camera.pan, orbited.pan, "Shift pans");
        let zoom = camera.zoom;
        assert!(step(
            &mut camera,
            vec![key(egui::Key::Plus, egui::Modifiers::NONE)],
            egui::Modifiers::NONE
        ));
        assert!(camera.zoom > zoom, "plus zooms in");
        assert!(step(
            &mut camera,
            vec![key(egui::Key::Minus, egui::Modifiers::NONE)],
            egui::Modifiers::NONE
        ));
        assert!((camera.zoom - zoom).abs() < 1e-4, "minus zooms back out");
    }

    #[test]
    fn tc_0304_templates_start_new_projects_keeping_the_mesh() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.load_imported_mesh(&write_two_material_gltf(folder.path()));
        app.add_fill_layer();
        assert!(app.history.can_undo());
        let mesh = app.project.mesh.clone();
        let template = substancecraft_core::ProjectTemplate::built_in()
            .into_iter()
            .find(|template| template.name == "Worn metal prop")
            .unwrap();
        app.start_from_template(&template);
        // One set per material slot of the kept mesh, a fresh history, and
        // no file yet.
        let slots: Vec<&str> = app
            .project
            .texture_sets
            .iter()
            .map(|set| set.material_slot.as_str())
            .collect();
        assert_eq!(slots, ["Body", "Trim"]);
        assert_eq!(app.project.mesh, mesh);
        assert_eq!(app.project_path, None);
        assert!(!app.history.can_undo());
        assert_eq!(
            app.selected_texture_set,
            Some(app.project.texture_sets[0].id)
        );
        assert!(app.selected_layer.is_some());
        app.project.validate().unwrap();
        // New layers get ids no template layer uses.
        app.add_paint_layer();
        app.project.validate().unwrap();

        // A set saved as a template file round-trips.
        let set = app.selected_texture_set_ref().unwrap().clone();
        let saved = substancecraft_core::ProjectTemplate::from_texture_set(&set, "Studio");
        let path = folder.path().join("studio.sctemplate");
        substancecraft_api::save_project_template(&path, &saved).unwrap();
        assert_eq!(
            substancecraft_api::load_project_template(&path).unwrap(),
            saved
        );
        std::fs::write(&path, b"{\"format\":\"other\",\"version\":1}").unwrap();
        assert!(substancecraft_api::load_project_template(&path).is_err());
    }

    #[test]
    fn tc_0192_openpbr_gltf_udim_and_text_resources_export_through_the_app() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.add_fill_layer();
        small_sets(&mut app, 8);
        let set = app.selected_texture_set_ref().unwrap().clone();

        // OpenPBR: textures plus a MaterialX document that reads back.
        let document = app.write_openpbr(&set, folder.path()).unwrap();
        let material = substancecraft_api::read_openpbr_materialx(&document).unwrap();
        assert_eq!(
            material.textures.len(),
            substancecraft_api::OPENPBR_INPUTS.len()
        );
        for file in material.textures.values() {
            assert!(folder.path().join(file).exists(), "{file}");
        }

        // glTF: the mesh re-imports with its texture set as the material.
        let gltf = folder.path().join("scene.gltf");
        app.write_gltf_scene(&gltf).unwrap();
        let imported = substancecraft_api::import_gltf_file(&gltf).unwrap();
        assert_eq!(imported.geometry.triangles.len(), 12);

        // UDIM: the cube sits in tile 1001.
        let tiles = folder.path().join("tiles");
        std::fs::create_dir(&tiles).unwrap();
        let written = app
            .write_udim_tiles(&set, &app.selected_export_preset(), &tiles)
            .unwrap();
        assert!(!written.is_empty());
        assert!(
            written
                .iter()
                .all(|path| path.to_string_lossy().ends_with(".1001.png"))
        );

        // Text resources render into ordinary bitmaps.
        let label = folder.path().join("label.sctext");
        let id = app.create_text_resource(&label, "HELLO", 64.0).unwrap();
        let image = &app.resource_images[&id];
        assert!(image.width > image.height);
        assert!(image.rgba8.chunks(4).any(|texel| texel[3] == 255));
    }

    #[test]
    fn tc_0203_texture_sets_remember_their_own_export_choices() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.add_fill_layer();
        let body = app.selected_texture_set.unwrap();
        app.add_texture_set();
        app.add_fill_layer();
        let trim = app.selected_texture_set.unwrap();
        small_sets(&mut app, 8);

        // Trim remembers Unreal as TGA; Body keeps following the app-wide
        // choice.
        app.set_preset_choice("Unreal Engine".to_owned());
        app.set_format_choice(TextureFormat::Tga);
        app.remember_export_for_selected(true);
        app.set_preset_choice(DEFAULT_EXPORT_PRESET.to_owned());
        app.set_format_choice(TextureFormat::Png);
        let saved = app
            .selected_texture_set_ref()
            .unwrap()
            .export
            .clone()
            .unwrap();
        assert_eq!(
            saved.preset, DEFAULT_EXPORT_PRESET,
            "edits go to the remembered set"
        );
        assert_eq!(
            app.export_preset, "Unreal Engine",
            "app-wide choice untouched"
        );
        app.set_preset_choice("Unreal Engine".to_owned());
        app.set_format_choice(TextureFormat::Tga);

        app.selected_texture_set = Some(body);
        app.set_preset_choice(DEFAULT_EXPORT_PRESET.to_owned());
        app.set_format_choice(TextureFormat::Png);
        assert_eq!(app.selected_preset_name(), DEFAULT_EXPORT_PRESET);
        app.selected_texture_set = Some(trim);
        assert_eq!(app.selected_preset_name(), "Unreal Engine");
        assert_eq!(app.selected_format(), TextureFormat::Tga);
        app.project.validate().unwrap();

        let written = app.export_all_sets_into(folder.path()).unwrap();
        let names: Vec<String> = written
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().any(|name| name.ends_with("_ORM.png")));
        assert!(
            names
                .iter()
                .any(|name| name.starts_with("T_") && name.ends_with("_N.tga"))
        );

        // Forgetting falls back to the app-wide choice.
        app.remember_export_for_selected(false);
        assert_eq!(app.selected_format(), TextureFormat::Png);
    }

    #[test]
    fn tc_0204_engine_export_actions_select_presets_and_name_outputs() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.add_fill_layer();
        small_sets(&mut app, 8);
        let set = app.selected_texture_set_ref().unwrap().clone();
        let stem = set.name.replace(' ', "_");

        let unreal = app.select_engine_preset("Unreal Engine").unwrap();
        assert_eq!(app.selected_preset_name(), "Unreal Engine");
        let unreal_folder = folder.path().join("unreal");
        std::fs::create_dir(&unreal_folder).unwrap();
        let unreal_files = app.write_export(&set, &unreal, &unreal_folder).unwrap();
        assert!(unreal_files.iter().any(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with(&format!("T_{stem}_BC")))
        }));

        let unity = app.select_engine_preset("Unity (URP Lit)").unwrap();
        assert_eq!(app.selected_preset_name(), "Unity (URP Lit)");
        let unity_folder = folder.path().join("unity");
        std::fs::create_dir(&unity_folder).unwrap();
        let unity_files = app.write_export(&set, &unity, &unity_folder).unwrap();
        assert!(unity_files.iter().any(|path| {
            path.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .starts_with(&format!("{stem}_BaseMap"))
            })
        }));

        let godot = app
            .select_engine_preset("Godot 4 (StandardMaterial3D)")
            .unwrap();
        assert_eq!(app.selected_preset_name(), "Godot 4 (StandardMaterial3D)");
        let godot_folder = folder.path().join("godot");
        std::fs::create_dir(&godot_folder).unwrap();
        let godot_files = app.write_export(&set, &godot, &godot_folder).unwrap();
        assert!(godot_files.iter().any(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy() == format!("{stem}_orm.png"))
        }));
    }

    #[test]
    fn tc_0210_still_renders_keep_the_viewport_aspect_at_the_requested_size() {
        assert_eq!(render_size(800.0, 600.0, 2048), (2048, 1536, 2.56));
        let (width, height, factor) = render_size(300.0, 900.0, 1024);
        assert_eq!((width, height), (341, 1024));
        assert!((factor - 1024.0 / 900.0).abs() < 1e-6);
        // Requests beyond the renderer's limit are clamped.
        assert_eq!(render_size(1000.0, 1000.0, 9000).0, MAX_RENDER_EDGE);
    }

    /// Needs a GPU: run with `cargo test -p substancecraft -- --ignored`.
    #[test]
    #[ignore]
    fn tc_0211_still_render_saves_an_opaque_png_of_the_material() {
        let folder = tempfile::tempdir().unwrap();
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp {
            filament_renderer: Some(FilamentRuntime::new(64, 64).unwrap()),
            ..SubstanceCraftApp::default()
        };
        app.add_fill_layer();
        small_sets(&mut app, 16);
        let path = folder.path().join("render.png");
        app.queue_render(path.clone(), 256, None);
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0));
        for _ in 0..600 {
            app.update_filament_frame(&context, viewport);
            if app.render_request.is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let status = app.render_status.clone().unwrap_or_default();
        assert!(status.starts_with("Saved 256 × 192"), "{status}");
        let image = load_texture_image(&path).unwrap();
        assert_eq!((image.width, image.height), (256, 192));
        assert!(image.rgba8.chunks(4).all(|texel| texel[3] == 255));
        assert!(
            image
                .rgba8
                .chunks(4)
                .any(|texel| texel[..3] != image.rgba8[..3]),
            "the mesh is visible against the background"
        );
    }

    #[test]
    fn tc_0314_lighting_presets_are_remembered_and_survive_opening_projects() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("preferences.json");
        let mut app = SubstanceCraftApp::default();
        app.load_preferences(path.clone());
        assert_eq!(app.viewport_lighting.preset, LightingPreset::Studio);

        app.set_lighting_preset(LightingPreset::Sunset);
        app.viewport_lighting.rotation = 2.0;
        assert_eq!(app.viewport_lighting.preset, LightingPreset::Sunset);
        assert_eq!(
            preferences::Preferences::load(&path).lighting_preset,
            LightingPreset::Sunset
        );

        // Opening a project resets the camera and rotation but not the
        // user's preset.
        app.adopt_project(MaterialProject::empty(), None).unwrap();
        assert_eq!(app.viewport_lighting.preset, LightingPreset::Sunset);
        assert_eq!(
            app.viewport_lighting.rotation,
            ViewportLighting::default().rotation
        );

        // A later session starts with it.
        let mut later = SubstanceCraftApp::default();
        later.load_preferences(path);
        assert_eq!(later.viewport_lighting.preset, LightingPreset::Sunset);

        // Strength scales the environment around its own level.
        let mut lighting = ViewportLighting::default();
        assert_eq!(lighting.environment_scale(), 1.0);
        lighting.ambient_fill = 0.72;
        assert!((lighting.environment_scale() - 2.0).abs() < 1e-6);
        lighting.ambient_fill = f32::NAN;
        assert_eq!(lighting.environment_scale(), 1.0);
    }

    #[test]
    #[ignore]
    fn tc_0315_still_renders_follow_the_lighting_preset() {
        let folder = tempfile::tempdir().unwrap();
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp {
            filament_renderer: Some(FilamentRuntime::new(64, 64).unwrap()),
            ..SubstanceCraftApp::default()
        };
        app.add_fill_layer();
        small_sets(&mut app, 16);
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 150.0));
        let render = |app: &mut SubstanceCraftApp, name: &str| {
            let path = folder.path().join(name);
            app.queue_render(path.clone(), 128, None);
            for _ in 0..600 {
                app.update_filament_frame(&context, viewport);
                if app.render_request.is_none() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            load_texture_image(&path).unwrap().rgba8
        };
        let mut frames = Vec::new();
        for preset in [LightingPreset::BrightOutdoor, LightingPreset::Night] {
            app.set_lighting_preset(preset);
            // The first render after a change may still use the old light.
            render(&mut app, "warmup.png");
            assert_eq!(
                app.filament_environment.map(|(applied, _, _)| applied),
                Some(environments::EnvironmentSource::Preset(preset))
            );
            frames.push(render(&mut app, &format!("{preset:?}.png")));
        }
        let mean = |frame: &[u8]| -> f64 {
            frame.iter().map(|value| f64::from(*value)).sum::<f64>() / frame.len() as f64
        };
        assert!(
            mean(&frames[0]) > mean(&frames[1]) + 1.0,
            "a bright outdoor render is brighter than a night one"
        );

        // A new rotation keeps the preset and only turns it.
        app.viewport_lighting.rotation += 1.0;
        app.update_filament_frame(&context, viewport);
        let (source, _, rotation) = app.filament_environment.unwrap();
        assert_eq!(
            source,
            environments::EnvironmentSource::Preset(LightingPreset::Night)
        );
        assert_eq!(rotation, app.viewport_lighting.rotation);
    }

    /// An uncompressed Radiance panorama, `width` x `width / 2`, of one
    /// colour (each component below 1).
    fn uniform_radiance_file(path: &std::path::Path, width: u32, rgb: [f32; 3]) {
        let height = width / 2;
        let mut bytes =
            format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n").into_bytes();
        // Mantissas over 2^0: value = byte / 256.
        let texel = rgb.map(|value| (value * 256.0).clamp(0.0, 255.0) as u8);
        for _ in 0..width * height {
            bytes.extend(texel);
            bytes.push(128);
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn tc_0347_custom_hdr_environments_are_remembered_and_presets_replace_them() {
        let folder = tempfile::tempdir().unwrap();
        let preferences_path = folder.path().join("preferences.json");
        let sky = folder.path().join("sky.hdr");
        uniform_radiance_file(&sky, 32, [0.5, 0.5, 0.75]);
        let broken = folder.path().join("broken.hdr");
        std::fs::write(&broken, b"#?RADIANCE\nnot an image").unwrap();

        let mut app = SubstanceCraftApp::default();
        app.load_preferences(preferences_path.clone());
        assert!(app.custom_environment.is_none());

        // Loading lights the viewport with the file and remembers it.
        app.load_custom_environment(sky.clone()).unwrap();
        let custom = app.custom_environment.as_ref().unwrap();
        assert_eq!(custom.name(), "sky.hdr");
        assert_eq!(
            (custom.image.width, custom.image.height),
            (
                substancecraft_core::environment::ENVIRONMENT_WIDTH,
                substancecraft_core::environment::ENVIRONMENT_WIDTH / 2
            )
        );
        let first = custom.revision;
        assert_eq!(app.environment_status.as_deref(), Some("Lit by sky.hdr"));
        assert_eq!(
            preferences::Preferences::load(&preferences_path).environment_file,
            Some(sky.clone())
        );

        // A broken file says why and keeps the current lighting.
        let error = app.load_custom_environment(broken).unwrap_err();
        assert!(error.starts_with("Could not load broken.hdr"), "{error}");
        assert_eq!(app.environment_status.as_deref(), Some(error.as_str()));
        assert_eq!(app.custom_environment.as_ref().unwrap().revision, first);
        assert_eq!(
            preferences::Preferences::load(&preferences_path).environment_file,
            Some(sky.clone())
        );

        // Reloading the same file is a new environment for the renderer.
        app.load_custom_environment(sky.clone()).unwrap();
        assert!(app.custom_environment.as_ref().unwrap().revision > first);

        // Opening a project keeps it; a later session restores it.
        app.adopt_project(MaterialProject::empty(), None).unwrap();
        assert!(app.custom_environment.is_some());
        let mut later = SubstanceCraftApp::default();
        later.load_preferences(preferences_path.clone());
        assert_eq!(
            later.custom_environment.as_ref().map(|custom| &custom.path),
            Some(&sky)
        );

        // Picking a preset replaces it and forgets the file, even the
        // preset that was selected underneath.
        let underneath = later.viewport_lighting.preset;
        later.set_lighting_preset(underneath);
        assert!(later.custom_environment.is_none());
        assert_eq!(later.environment_status, None);
        assert_eq!(
            preferences::Preferences::load(&preferences_path).environment_file,
            None
        );

        // A remembered file gone missing falls back to the preset but stays
        // remembered for when it returns.
        let mut values = preferences::Preferences::load(&preferences_path);
        let missing = folder.path().join("moved.exr");
        values.environment_file = Some(missing.clone());
        values.save(&preferences_path).unwrap();
        let mut session = SubstanceCraftApp::default();
        session.load_preferences(preferences_path.clone());
        assert!(session.custom_environment.is_none());
        assert!(
            session
                .environment_status
                .as_deref()
                .is_some_and(|status| status.starts_with("Could not load moved.exr")),
            "{:?}",
            session.environment_status
        );
        assert_eq!(
            preferences::Preferences::load(&preferences_path).environment_file,
            Some(missing)
        );
    }

    #[test]
    #[ignore]
    fn tc_0348_still_renders_follow_a_custom_environment() {
        use environments::EnvironmentSource;
        let folder = tempfile::tempdir().unwrap();
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp {
            filament_renderer: Some(FilamentRuntime::new(64, 64).unwrap()),
            ..SubstanceCraftApp::default()
        };
        app.add_fill_layer();
        small_sets(&mut app, 16);
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 150.0));
        let render = |app: &mut SubstanceCraftApp, name: &str| {
            let path = folder.path().join(name);
            app.queue_render(path.clone(), 128, None);
            for _ in 0..600 {
                app.update_filament_frame(&context, viewport);
                if app.render_request.is_none() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            load_texture_image(&path).unwrap().rgba8
        };
        // Red and blue skies of the same brightness light the model red and
        // blue.
        let mut tints = Vec::new();
        for (name, rgb) in [
            ("red.hdr", [0.9, 0.05, 0.05]),
            ("blue.hdr", [0.05, 0.05, 0.9]),
        ] {
            let path = folder.path().join(name);
            uniform_radiance_file(&path, 64, rgb);
            app.load_custom_environment(path).unwrap();
            render(&mut app, "warmup.png");
            let revision = app.custom_environment.as_ref().unwrap().revision;
            assert_eq!(
                app.filament_environment.map(|(applied, _, _)| applied),
                Some(EnvironmentSource::Custom(revision))
            );
            let frame = render(&mut app, &format!("{name}.png"));
            let sum = |channel: usize| -> f64 {
                frame
                    .chunks(4)
                    .map(|texel| f64::from(texel[channel]))
                    .sum::<f64>()
            };
            tints.push((sum(0), sum(2)));
        }
        assert!(tints[0].0 > tints[0].1, "the red sky lights red: {tints:?}");
        assert!(
            tints[1].1 > tints[1].0,
            "the blue sky lights blue: {tints:?}"
        );

        // A new rotation keeps the panorama and only turns it; a preset
        // replaces it.
        let revision = app.custom_environment.as_ref().unwrap().revision;
        app.viewport_lighting.rotation += 1.0;
        app.update_filament_frame(&context, viewport);
        let (source, _, rotation) = app.filament_environment.unwrap();
        assert_eq!(source, EnvironmentSource::Custom(revision));
        assert_eq!(rotation, app.viewport_lighting.rotation);
        app.set_lighting_preset(LightingPreset::Overcast);
        app.update_filament_frame(&context, viewport);
        assert_eq!(
            app.filament_environment.map(|(applied, _, _)| applied),
            Some(EnvironmentSource::Preset(LightingPreset::Overcast))
        );
    }

    /// Hands the app a finished 2x2 frame, as the renderer's callback does,
    /// and lets it save and move on.
    fn finish_render(app: &mut SubstanceCraftApp) {
        let request = app.render_request.as_mut().unwrap();
        request.issued = true;
        *app.render_result.lock().unwrap() = Some(FilamentFrame {
            width: 2,
            height: 2,
            rgba8: vec![90; 16],
        });
        app.save_finished_render();
        app.advance_turntable();
    }

    #[test]
    fn tc_0350_turntables_render_numbered_frames_one_after_another() {
        use turntable::TurntableMode;
        let folder = tempfile::tempdir().unwrap();
        let base = folder.path().join("spin.png");
        let mut app = SubstanceCraftApp::default();
        app.viewport_camera.yaw = 1.0;
        app.viewport_lighting.rotation = 0.5;

        // A light turntable keeps the camera and turns the environment.
        app.start_turntable(base.clone(), 4, TurntableMode::Lighting, 256);
        let request = app.render_request.as_ref().unwrap();
        assert_eq!(request.path, folder.path().join("spin_000.png"));
        assert_eq!(request.camera.unwrap().yaw, 1.0);
        assert_eq!(request.lighting_rotation, Some(0.5));
        assert_eq!(request.long_edge, 256);
        assert_eq!(
            app.render_status.as_deref(),
            Some("Rendering turntable frame 1 of 4…")
        );

        // Moving the viewport mid-sequence does not move the turntable.
        app.viewport_camera.yaw = 3.0;
        app.viewport_lighting.rotation = 2.0;
        let mut rotations = Vec::new();
        for frame in 0..4 {
            let request = app.render_request.as_ref().unwrap();
            assert_eq!(request.turntable_frame, Some(frame));
            assert_eq!(request.camera.unwrap().yaw, 1.0);
            rotations.push(request.lighting_rotation.unwrap());
            finish_render(&mut app);
        }
        let step = std::f32::consts::FRAC_PI_2;
        for (frame, rotation) in rotations.iter().enumerate() {
            assert!((rotation - (0.5 + step * frame as f32)).abs() < 1e-5);
        }
        assert!(app.turntable.is_none() && app.render_request.is_none());
        for frame in 0..4 {
            assert!(folder.path().join(format!("spin_{frame:03}.png")).is_file());
        }
        let status = app.render_status.clone().unwrap();
        assert!(status.starts_with("Saved 4-frame turntable to"), "{status}");

        // A camera turntable orbits the camera from where it started.
        app.start_turntable(base.clone(), 3, TurntableMode::Camera, 256);
        finish_render(&mut app);
        let request = app.render_request.as_ref().unwrap();
        let expected = (3.0 + std::f32::consts::TAU / 3.0).rem_euclid(std::f32::consts::TAU);
        assert!((request.camera.unwrap().yaw - expected).abs() < 1e-5);
        assert_eq!(request.lighting_rotation, Some(2.0));

        // Cancelling stops after the frames already saved.
        app.cancel_turntable();
        assert!(app.turntable.is_none() && app.render_request.is_none());
        assert_eq!(
            app.render_status.as_deref(),
            Some("Turntable cancelled after 1 of 3 frames")
        );

        // A still render replaces a running turntable.
        app.start_turntable(base, 3, TurntableMode::Camera, 256);
        app.queue_render(folder.path().join("still.png"), 256, None);
        assert!(app.turntable.is_none());
        assert_eq!(app.render_request.as_ref().unwrap().turntable_frame, None);

        // A frame that cannot be written ends the turntable with the error.
        let blocked = folder.path().join("blocked");
        std::fs::write(&blocked, b"a file, not a folder").unwrap();
        app.render_request = None;
        app.start_turntable(blocked.join("spin.png"), 3, TurntableMode::Camera, 256);
        finish_render(&mut app);
        assert!(app.turntable.is_none() && app.render_request.is_none());
        let status = app.render_status.clone().unwrap();
        assert!(status.starts_with("Render failed"), "{status}");
    }

    #[test]
    #[ignore]
    fn tc_0351_turntable_frames_render_on_the_gpu() {
        use turntable::TurntableMode;
        let folder = tempfile::tempdir().unwrap();
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp {
            filament_renderer: Some(FilamentRuntime::new(64, 64).unwrap()),
            ..SubstanceCraftApp::default()
        };
        app.add_fill_layer();
        small_sets(&mut app, 16);
        app.set_lighting_preset(LightingPreset::Sunset);
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 150.0));
        // Settle the environment before the sequence starts.
        app.queue_render(folder.path().join("warmup.png"), 64, None);
        let run = |app: &mut SubstanceCraftApp| {
            for _ in 0..2000 {
                app.update_filament_frame(&context, viewport);
                if app.render_request.is_none() && app.turntable.is_none() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            assert!(
                app.render_request.is_none() && app.turntable.is_none(),
                "the renders did not finish"
            );
        };
        run(&mut app);
        for (mode, name) in [
            (TurntableMode::Camera, "camera.png"),
            (TurntableMode::Lighting, "light.png"),
        ] {
            app.start_turntable(folder.path().join(name), 4, mode, 96);
            run(&mut app);
            let stem = name.trim_end_matches(".png");
            let frames: Vec<Vec<u8>> = (0..4)
                .map(|frame| {
                    load_texture_image(folder.path().join(format!("{stem}_{frame:03}.png")))
                        .unwrap()
                        .rgba8
                })
                .collect();
            // Every quarter turn looks different: the model (camera) or its
            // shading (light) turns between frames.
            for pair in frames.windows(2) {
                let difference: u64 = pair[0]
                    .iter()
                    .zip(&pair[1])
                    .map(|(a, b)| u64::from(a.abs_diff(*b)))
                    .sum();
                assert!(difference > 1000, "{mode:?}: {difference}");
            }
        }
    }

    /// A triangle skinned to a two-bone arm whose elbow turns 90° about Z
    /// over one second (the same rig as the API's FBX rig test).
    const RIGGED_ARM_FBX: &str = r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {
	FBXVersion: 7400
}
Documents:  {
	Document: 900, "Scene", "Scene" {
		RootNode: 0
	}
}
Objects:  {
	Geometry: 1, "Geometry::Arm", "Mesh" {
		Vertices: *9 {
			a: 0.5,0,0,2,0,0,2,0.1,0
		}
		PolygonVertexIndex: *3 {
			a: 0,1,-3
		}
	}
	Model: 10, "Model::Arm", "Mesh" {
	}
	Model: 20, "Model::Shoulder", "LimbNode" {
	}
	Model: 21, "Model::Elbow", "LimbNode" {
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",1,0,0
		}
	}
	Deformer: 30, "Deformer::Skin", "Skin" {
	}
	Deformer: 31, "SubDeformer::Shoulder", "Cluster" {
		Indexes: *1 {
			a: 0
		}
		Weights: *1 {
			a: 1
		}
		TransformLink: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
		}
	}
	Deformer: 32, "SubDeformer::Elbow", "Cluster" {
		Indexes: *2 {
			a: 1,2
		}
		Weights: *2 {
			a: 1,1
		}
		TransformLink: *16 {
			a: 1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1
		}
	}
	AnimationStack: 40, "AnimStack::Bend", "" {
	}
	AnimationLayer: 41, "AnimLayer::BaseLayer", "" {
	}
	AnimationCurveNode: 42, "AnimCurveNode::R", "" {
	}
	AnimationCurve: 43, "AnimCurve::", "" {
		KeyTime: *2 {
			a: 0,46186158000
		}
		KeyValueFloat: *2 {
			a: 0,90
		}
	}
}
Connections:  {
	C: "OO",1,10
	C: "OO",10,0
	C: "OO",20,0
	C: "OO",21,20
	C: "OO",30,1
	C: "OO",31,30
	C: "OO",32,30
	C: "OO",20,31
	C: "OO",21,32
	C: "OO",41,40
	C: "OO",42,41
	C: "OP",42,21, "Lcl Rotation"
	C: "OP",43,42, "d|Z"
}
"#;

    #[test]
    fn tc_0320_rigged_meshes_pose_in_the_viewport_and_saved_poses_keep_their_frame() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("arm.fbx");
        std::fs::write(&path, RIGGED_ARM_FBX).unwrap();
        let mut app = SubstanceCraftApp::default();
        app.set_mesh_smoothing(MeshSmoothing {
            angle_degrees: 0.0,
            subdivision_levels: 0,
        });
        app.load_imported_mesh(&path);
        assert!(app.import_error.is_none(), "{:?}", app.import_error);
        let rig = app.current_rig().unwrap();
        assert_eq!(rig.clips[0].name, "Bend");

        // Bind pose: the viewport shows the mesh as imported.
        assert!(!app.is_posed());
        let bind = app.display_mesh().unwrap();
        assert_eq!(bind.vertices[1].position, glam::Vec3::new(2.0, 0.0, 0.0));

        // At the end of the clip the elbow vertices have swung up, while the
        // painted (bind) mesh is untouched.
        app.animation.source = animation::PoseSource::Clip("Bend".to_owned());
        app.animation.time = 1.0;
        assert!(app.is_posed());
        let posed = app.display_mesh().unwrap();
        assert!((posed.vertices[1].position - glam::Vec3::new(1.0, 1.0, 0.0)).length() < 1e-4);
        assert_eq!(
            app.mesh_geometry.as_ref().unwrap().vertices[1].position,
            glam::Vec3::new(2.0, 0.0, 0.0)
        );
        assert_eq!(
            posed.triangles,
            app.mesh_geometry.as_ref().unwrap().triangles
        );

        // Picking the posed triangle returns bind-space UVs and positions.
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 400.0));
        app.viewport_camera = ViewportCamera {
            yaw: 0.0,
            pitch: 0.0,
            ..ViewportCamera::default()
        };
        let projection = app.view_projection(viewport);
        let points = projection.project_mesh(&posed);
        let centroid =
            (points[0].screen.to_vec2() + points[1].screen.to_vec2() + points[2].screen.to_vec2())
                / 3.0;
        let bind_mesh = app.mesh_geometry.clone().unwrap();
        let hit = pick_triangle(&bind_mesh, &points, centroid.to_pos2(), None).unwrap();
        let expected = (bind_mesh.vertices[0].position
            + bind_mesh.vertices[1].position
            + bind_mesh.vertices[2].position)
            / 3.0;
        assert!(
            (hit.position - expected).length() < 1e-3,
            "{}",
            hit.position
        );
        // A point on the bind triangle lands on the posed one.
        let carried = animation::carry_point(&bind_mesh, &posed, 0, bind_mesh.vertices[1].position);
        assert!((carried - posed.vertices[1].position).length() < 1e-4);

        // Projection painting waits for the bind pose.
        app.projection_painting = true;
        app.paint_mode = PaintMode::Paint;
        assert!(!app.projects_strokes());

        // A saved pose keeps its frame; selecting it later restores it.
        app.pose_name = "Bent".to_owned();
        app.save_pose();
        let saved = app.project.saved_poses.last().unwrap().clone();
        assert_eq!(
            saved.animation,
            Some(substancecraft_core::AnimationFrame {
                clip: Some("Bend".to_owned()),
                time: 1.0
            })
        );
        app.project.validate().unwrap();
        app.show_animation_frame(None);
        assert!(!app.is_posed());
        assert!(app.projects_strokes());
        let context = egui::Context::default();
        app.select_pose(saved.id, &context);
        // A headless context's clock stands still; start the transition
        // long ago so it completes.
        if let Some(transition) = app.pose_transition.as_mut() {
            transition.started_at = -10.0;
        }
        app.update_pose_transition(&context);
        assert!(app.pose_transition.is_none());
        assert_eq!(
            app.animation.source,
            animation::PoseSource::Clip("Bend".to_owned())
        );
        assert_eq!(app.animation.time, 1.0);

        // Playing moves time on and loops within the clip.
        app.animation.time = 0.9;
        app.animation.playing = true;
        app.advance_animation(&context);
        std::thread::sleep(Duration::from_millis(250));
        app.advance_animation(&context);
        assert!(app.animation.time < 0.9, "looped: {}", app.animation.time);

        // A primitive has no rig and no animation controls.
        app.select_primitive(PrimitivePreset::Cube);
        assert!(app.current_rig().is_none());
        assert!(!app.is_posed());
    }

    #[test]
    #[ignore]
    fn tc_0321_filament_shows_the_posed_mesh() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("arm.fbx");
        std::fs::write(&path, RIGGED_ARM_FBX).unwrap();
        let context = egui::Context::default();
        let mut app = SubstanceCraftApp {
            filament_renderer: Some(FilamentRuntime::new(64, 64).unwrap()),
            ..SubstanceCraftApp::default()
        };
        app.load_imported_mesh(&path);
        app.viewport_camera = ViewportCamera {
            yaw: 0.0,
            pitch: 0.0,
            ..ViewportCamera::default()
        };
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 200.0));
        let render = |app: &mut SubstanceCraftApp, name: &str| {
            let path = folder.path().join(name);
            for _ in 0..2 {
                app.queue_render(path.clone(), 128, None);
                for _ in 0..600 {
                    app.update_filament_frame(&context, viewport);
                    if app.render_request.is_none() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            load_texture_image(&path).unwrap().rgba8
        };
        let bound = render(&mut app, "bind.png");
        app.animation.source = animation::PoseSource::Clip("Bend".to_owned());
        app.animation.time = 1.0;
        let posed = render(&mut app, "posed.png");
        assert!(app.filament_pose.as_ref().is_some_and(Option::is_some));
        let changed = bound
            .chunks(4)
            .zip(posed.chunks(4))
            .filter(|(a, b)| a[..3] != b[..3])
            .count();
        assert!(changed > 50, "posing changes the render ({changed} texels)");
    }

    #[test]
    fn tc_0329_ctrl_clicked_layers_flatten_together_through_the_app() {
        let folder = tempfile::tempdir().unwrap();
        let mut app = SubstanceCraftApp::default();
        app.add_fill_layer();
        let bottom = app.selected_layer.unwrap();
        app.add_paint_layer();
        let middle = app.selected_layer.unwrap();
        app.add_fill_layer();
        let top = app.selected_layer.unwrap();
        small_sets(&mut app, 8);
        if let Some(layer) = app.selected_layer_mut() {
            layer.opacity = 0.5;
            if let Some(fill) = &mut layer.fill {
                fill.color = [200, 20, 20, 255];
            }
        }
        app.sync_layer_history(&egui::Context::default());
        let set_id = app.selected_texture_set.unwrap();
        let before = app.selected_texture_set_ref().unwrap().clone();
        let ids: Vec<LayerId> = before.layers.iter().map(|layer| layer.id).collect();
        assert_eq!(ids, [bottom, middle, top]);
        let original = app
            .bake_inputs(set_id)
            .bake_channel(&before, MaterialChannel::BaseColor)
            .unwrap();

        // Ctrl-clicks add and remove layers; Ctrl-clicking the main one
        // hands the selection on.
        app.selected_layer = Some(bottom);
        app.toggle_extra_selection(top);
        assert_eq!(app.flatten_selection(), [bottom, top]);
        app.toggle_extra_selection(middle);
        app.toggle_extra_selection(middle);
        assert_eq!(app.flatten_selection(), [bottom, top]);
        app.toggle_extra_selection(bottom);
        assert_eq!(app.selected_layer, Some(top));
        assert_eq!(app.flatten_selection(), [top]);
        app.toggle_extra_selection(bottom);
        assert_eq!(app.flatten_selection(), [top, bottom]);

        // One fill above the topmost, both sources hidden, the layer
        // between them untouched; the stack composites as before.
        let merged = app.flatten_selected_into(folder.path()).unwrap();
        let set = app.selected_texture_set_ref().unwrap().clone();
        let ids: Vec<LayerId> = set.layers.iter().map(|layer| layer.id).collect();
        assert_eq!(ids, [bottom, middle, top, merged]);
        assert!(
            set.layers[3].name.ends_with(" + 1 (flattened)"),
            "{}",
            set.layers[3].name
        );
        assert!(!set.layer(bottom).unwrap().visible);
        assert!(set.layer(middle).unwrap().visible);
        assert!(!set.layer(top).unwrap().visible);
        assert!(app.extra_selected_layers.is_empty());
        assert_eq!(app.selected_layer, Some(merged));
        let rebuilt = app
            .bake_inputs(set_id)
            .bake_channel(&set, MaterialChannel::BaseColor)
            .unwrap();
        let worst = original
            .rgba8
            .iter()
            .zip(&rebuilt.rgba8)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst <= 2, "differs by {worst}");

        // One undo step restores the stack.
        app.undo();
        assert_eq!(
            app.selected_texture_set_ref().unwrap().layers,
            before.layers
        );
    }

    #[test]
    fn tc_0330_the_brush_cursor_previews_every_symmetric_copy() {
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.add_paint_layer();
        small_sets(&mut app, 64);
        app.set_paint_mode(PaintMode::Paint);
        app.brush_radius = 2.0;
        let cube = app.mesh_geometry.clone().unwrap();
        // Looking down -Z at the +Z face: screen (130, 90) is (0.3, 0.1).
        let projection = ViewProjection {
            pose: BoneTransform::IDENTITY,
            camera_rotation: glam::Quat::IDENTITY,
            posed_center: glam::Vec3::ZERO,
            center: egui::pos2(100.0, 100.0),
            scale: 100.0,
            pan: glam::Vec2::ZERO,
        };
        let points = projection.project_mesh(&cube);
        let pointer = egui::pos2(130.0, 90.0);
        let footprints = |app: &SubstanceCraftApp| {
            app.brush_footprints(&cube, &projection, Some(&points), pointer)
        };
        let centre = |outline: &[egui::Pos2]| {
            let sum = outline
                .iter()
                .fold(egui::Vec2::ZERO, |sum, point| sum + point.to_vec2());
            (sum / outline.len() as f32).to_pos2()
        };
        let near = |a: egui::Pos2, b: egui::Pos2| (a - b).length() < 1.5;

        // Without symmetry, one outline round the cursor.
        let single = footprints(&app);
        assert_eq!(single.len(), 1);
        assert!(
            near(centre(&single[0]), pointer),
            "{:?}",
            centre(&single[0])
        );

        // Mirrored across X, a second outline on the other side.
        app.symmetry = Some(substancecraft_core::Symmetry::new(
            substancecraft_core::Axis::X,
        ));
        let mirrored = footprints(&app);
        assert_eq!(mirrored.len(), 2);
        assert_eq!(mirrored[0], single[0], "the cursor outline is unchanged");
        assert!(
            near(centre(&mirrored[1]), egui::pos2(70.0, 90.0)),
            "{:?}",
            centre(&mirrored[1])
        );

        // Radial copies: one outline per copy the stroke paints, matching
        // the copies a stroke would paint.
        app.symmetry = Some(substancecraft_core::Symmetry::radial(
            substancecraft_core::Axis::Y,
            4,
        ));
        let radial = footprints(&app);
        assert_eq!(radial.len(), 4);
        let hit = pick_triangle(&cube, &points, pointer, None).unwrap();
        assert_eq!(app.symmetry_copies(&hit, hit.uv.unwrap()).len(), 3);

        // Off the mesh there is no footprint.
        assert!(
            app.brush_footprints(&cube, &projection, Some(&points), egui::pos2(400.0, 400.0))
                .is_empty()
        );
    }

    #[test]
    fn tc_0332_openpbr_documents_import_as_fill_layers_through_the_app() {
        let folder = tempfile::tempdir().unwrap();
        let solid = |rgba: [u8; 4]| BakedTexture {
            width: 4,
            height: 4,
            rgba8: rgba.repeat(16),
        };
        write_texture(
            folder.path().join("crate_color.png"),
            &solid([200, 30, 30, 255]),
        )
        .unwrap();
        write_texture(
            folder.path().join("crate_rough.png"),
            &solid([64, 64, 64, 255]),
        )
        .unwrap();
        let files: BTreeMap<String, String> = [
            ("base_color", "crate_color.png"),
            ("specular_roughness", "crate_rough.png"),
        ]
        .into_iter()
        .map(|(input, file)| (input.to_owned(), file.to_owned()))
        .collect();
        let document = folder.path().join("crate.mtlx");
        substancecraft_api::write_openpbr_materialx(&document, "Painted crate", &files).unwrap();

        let mut app = SubstanceCraftApp::default();
        app.add_paint_layer();
        small_sets(&mut app, 4);
        let before = app.selected_texture_set_ref().unwrap().layers.clone();
        let id = app.import_openpbr_from(&document).unwrap();
        let set = app.selected_texture_set_ref().unwrap().clone();
        let layer = set.layer(id).unwrap();
        assert_eq!(layer.name, "Painted crate");
        let fill = layer.fill.unwrap();
        let color = fill.textures.base_color.unwrap();
        let rough = fill.textures.roughness.unwrap();
        let space = |id: ResourceId| {
            app.project
                .resources
                .iter()
                .find(|resource| resource.id == id)
                .unwrap()
                .color_space
        };
        assert_eq!(space(color), substancecraft_core::ColorSpace::Srgb);
        assert_eq!(space(rough), substancecraft_core::ColorSpace::Raw);
        let set_id = app.selected_texture_set.unwrap();
        let base = app
            .bake_inputs(set_id)
            .bake_channel(&set, MaterialChannel::BaseColor)
            .unwrap();
        assert_eq!(&base.rgba8[..3], &[200, 30, 30]);
        let roughness = app
            .bake_inputs(set_id)
            .bake_channel(&set, MaterialChannel::Roughness)
            .unwrap();
        assert_eq!(roughness.rgba8[0], 64);

        // Importing again reuses the bitmaps; one undo removes each layer.
        let resources = app.project.resources.len();
        app.import_openpbr_from(&document).unwrap();
        assert_eq!(app.project.resources.len(), resources);
        app.undo();
        app.undo();
        assert_eq!(app.selected_texture_set_ref().unwrap().layers, before);

        // A missing texture is reported, and adds no layer and no bitmaps
        // even when an earlier texture loaded.
        let fresh = folder.path().join("fresh");
        std::fs::create_dir(&fresh).unwrap();
        std::fs::copy(&document, fresh.join("crate.mtlx")).unwrap();
        std::fs::copy(
            folder.path().join("crate_color.png"),
            fresh.join("crate_color.png"),
        )
        .unwrap();
        let error = app
            .import_openpbr_from(&fresh.join("crate.mtlx"))
            .unwrap_err();
        assert!(error.contains("crate_rough.png"), "{error}");
        assert_eq!(app.selected_texture_set_ref().unwrap().layers, before);
        assert_eq!(app.project.resources.len(), resources);
    }

    #[test]
    fn tc_0334_the_decal_gizmo_drags_spins_and_resizes_decals_in_one_undo_step() {
        let mut app = SubstanceCraftApp::default();
        app.set_base_geometry(Some(PrimitivePreset::Cube.build_mesh()));
        app.add_fill_layer();
        small_sets(&mut app, 16);
        let cube = app.mesh_geometry.clone().unwrap();
        if let Some(fill) = app
            .selected_layer_mut()
            .and_then(|layer| layer.fill.as_mut())
        {
            fill.projection.mode = substancecraft_core::ProjectionMode::Decal;
            fill.projection.decal.set_spin_degrees(30.0);
        }
        app.sync_layer_history(&egui::Context::default());
        let original = app.selected_layer().unwrap().settings();
        let decal = |app: &SubstanceCraftApp| app.active_decal().unwrap().1;
        let hit_at = |point: glam::Vec3| {
            let hit = cube.closest_surface_point(point).unwrap();
            TriangleHit {
                triangle: hit.triangle,
                uv: Some(hit.uv),
                position: hit.position,
            }
        };

        // Off until switched on; then it follows the selected decal.
        assert!(app.active_decal().is_none());
        app.decal_gizmo = Some(1);
        assert!(app.active_decal().is_none(), "no bitmap-mask decal");
        app.decal_gizmo = Some(0);
        assert!(app.active_decal().is_some());

        // A plain drag slides it onto the +X face, keeping its spin.
        let none = egui::Vec2::ZERO;
        app.drag_decal(
            &cube,
            Some(hit_at(glam::Vec3::new(0.9, 0.1, 0.0))),
            none,
            false,
            false,
        );
        let moved = decal(&app);
        assert!((moved.center - glam::Vec3::new(0.5, 0.1, 0.0)).length() < 1e-4);
        assert!((moved.normal - glam::Vec3::X).length() < 1e-5);
        assert!((moved.spin_degrees() - 30.0).abs() < 1e-3);
        // Off the mesh a plain drag leaves it where it is.
        app.drag_decal(&cube, None, egui::vec2(5.0, 5.0), false, false);
        assert_eq!(decal(&app), moved);

        // Shift spins by half a degree per point; Ctrl resizes, up grows.
        app.drag_decal(&cube, None, egui::vec2(20.0, 0.0), true, false);
        assert!((decal(&app).spin_degrees() - 40.0).abs() < 1e-3);
        let size = decal(&app).size;
        app.drag_decal(&cube, None, egui::vec2(0.0, -100.0), false, true);
        let grown = decal(&app).size;
        assert!((grown.x / size.x - std::f32::consts::E).abs() < 1e-3);
        assert_eq!(
            decal(&app).center,
            moved.center,
            "spin and size keep the centre"
        );

        // The whole drag is one undo step.
        app.sync_layer_history(&egui::Context::default());
        app.undo();
        assert_eq!(app.selected_layer().unwrap().settings(), original);

        // Turning the projection away from Decal switches the gizmo off.
        if let Some(fill) = app
            .selected_layer_mut()
            .and_then(|layer| layer.fill.as_mut())
        {
            fill.projection.mode = substancecraft_core::ProjectionMode::Uv;
        }
        assert!(app.active_decal().is_none());
        app.drag_decal(&cube, Some(hit_at(glam::Vec3::X)), none, false, false);
        assert_eq!(
            app.selected_layer().unwrap().fill.unwrap().projection.decal,
            original.fill.unwrap().projection.decal
        );
    }

    #[test]
    fn tc_0336_brush_colours_are_picked_in_a_remembered_colour_space() {
        use substancecraft_core::ColorSpace;

        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("preferences.json");
        let mut app = SubstanceCraftApp::default();
        app.load_preferences(path.clone());
        assert_eq!(app.preferences.values.picker_color_space, ColorSpace::Srgb);

        // The space is a preference; raw data is not offered.
        app.set_picker_color_space(ColorSpace::AcesCg);
        app.set_picker_color_space(ColorSpace::Raw);
        assert_eq!(
            app.preferences.values.picker_color_space,
            ColorSpace::AcesCg
        );
        let mut later = SubstanceCraftApp::default();
        later.load_preferences(path.clone());
        assert_eq!(
            later.preferences.values.picker_color_space,
            ColorSpace::AcesCg
        );
        let mut raw = preferences::Preferences::load(&path);
        raw.picker_color_space = ColorSpace::Raw;
        raw.save(&path).unwrap();
        assert_eq!(
            preferences::Preferences::load(&path).picker_color_space,
            ColorSpace::Srgb
        );

        // Linear values set the sRGB brush colour, keeping its alpha.
        app.brush_color = egui::Color32::from_rgba_premultiplied(10, 10, 10, 200);
        let clipped =
            app.set_brush_color_in(ColorSpace::LinearSrgb, glam::Vec3::splat(0.215_860_5));
        assert!(!clipped);
        assert_eq!(app.brush_color.to_array(), [128, 128, 128, 200]);
        assert_eq!(app.picker_clipped, None);
        let shown = ColorSpace::LinearSrgb.from_srgb8([128, 128, 128]);
        assert!(
            (shown.x - 0.215_860_5).abs() < 1e-4,
            "fields read the brush back"
        );

        // An ACEScg primary outside sRGB clips, and is flagged only while it
        // stays the brush colour.
        assert!(app.set_brush_color_in(ColorSpace::AcesCg, glam::Vec3::new(0.0, 1.0, 0.0)));
        assert_eq!(app.picker_clipped, Some(app.brush_color));
        assert_eq!(app.brush_color.r(), 0);
        app.brush_color = egui::Color32::from_rgb(1, 2, 3);
        assert_ne!(app.picker_clipped, Some(app.brush_color));
    }

    #[test]
    fn tc_0338_the_smart_material_browser_shows_thumbnails_and_instances_offer_parameters() {
        use substancecraft_core::ParameterValue;

        let mut app = SubstanceCraftApp::default();
        app.add_paint_layer();
        small_sets(&mut app, 8);
        let library = app.smart_material_library();
        let (worn, rusty) = (&library[0], &library[1]);

        // One new thumbnail per frame; cached ones cost nothing.
        let mut budget = true;
        let (_, first) = app.smart_thumbnail_image(worn, &mut budget).unwrap();
        assert!(!budget);
        assert_eq!(first.width, SMART_THUMBNAIL_SIZE);
        assert!(app.smart_thumbnail_image(rusty, &mut budget).is_none());
        let (_, again) = app.smart_thumbnail_image(worn, &mut budget).unwrap();
        assert!(Arc::ptr_eq(&first, &again), "served from the cache");
        let mut budget = true;
        assert!(app.smart_thumbnail_image(rusty, &mut budget).is_some());
        // A changed layer tree renders afresh.
        let mut changed = worn.clone();
        changed.root.opacity = 0.3;
        let mut budget = true;
        let (_, redrawn) = app.smart_thumbnail_image(&changed, &mut budget).unwrap();
        assert!(!budget && redrawn != first);

        // An inserted instance offers its parameters; setting one edits the
        // layer it drives, and a drag is one undo step.
        app.insert_smart_material(worn);
        let root = app.selected_layer().unwrap().clone();
        let values = root.exposed_values();
        assert_eq!(values.len(), worn.parameters.len());
        let index = values
            .iter()
            .find(|(_, parameter, _)| parameter.name == "Paint roughness")
            .map(|(index, _, _)| *index)
            .unwrap();
        app.set_exposed_parameter(index, ParameterValue::Scalar(0.2));
        app.set_exposed_parameter(index, ParameterValue::Scalar(0.9));
        let edited = app.selected_layer().unwrap().clone();
        assert!(
            edited
                .exposed_values()
                .iter()
                .any(|(i, _, value)| *i == index && *value == ParameterValue::Scalar(0.9))
        );
        assert_eq!(edited.id, root.id, "the instance stays selected");
        app.history.end_merge();
        app.undo();
        assert_eq!(app.selected_layer().unwrap(), &root);

        // Saving the instance keeps its parameters.
        app.save_layer_as_smart_material();
        let saved = app.project.smart_materials.last().unwrap();
        assert_eq!(saved.parameters.len(), worn.parameters.len());
        assert!(saved.root.exposed.is_empty());
    }
}
