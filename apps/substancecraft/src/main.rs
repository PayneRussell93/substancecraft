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
    rebake_texture_set_channel_regions, rebake_texture_set_material_regions, save_project_file,
    write_exported_textures, write_texture,
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
    TexelRect, TextureImage, TextureResolution, TextureSet, TextureSetId, ViewportCamera,
    ViewportLighting, remap_subdivided_triangles, stroke_texel_bounds,
};
use substancecraft_filament::{FilamentRuntime, MaterialMap};

mod animation;
mod autosave;
mod bake_log;
mod cache;
mod cameras;
mod diagnostics;
mod edges;
mod icons;
mod incremental;
mod inspector;
mod panels;
mod preferences;
mod relink;
mod resource_watch;
mod theme;
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
const MESH_EXTENSIONS: [&str; 3] = ["fbx", "gltf", "glb"];
/// Bitmap formats the resource importer reads.
const IMAGE_EXTENSIONS: [&str; 6] = ["png", "jpg", "jpeg", "tga", "svg", "sctext"];
const POSE_TRANSITION_SECONDS: f64 = 0.35;
const MAX_SAVED_POSES: usize = 256;
/// Selection overlays stop drawing beyond this many triangles to keep the
/// viewport responsive on dense meshes.
const MAX_OVERLAY_TRIANGLES: usize = 200_000;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("SubstanceCraft")
            .with_app_id("com.substancecraft.desktop")
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

struct SubstanceCraftApp {
    project: MaterialProject,
    selected_texture_set: Option<TextureSetId>,
    selected_layer: Option<LayerId>,
    selected_primitive: Option<PrimitivePreset>,
    /// Imported or primitive geometry after scaling, before smoothing.
    base_geometry: Option<MeshGeometry>,
    /// Smoothed/subdivided geometry used for display, painting, and baking.
    mesh_geometry: Option<Arc<MeshGeometry>>,
    mesh_topology: Option<MeshTopology>,
    mesh_center: glam::Vec3,
    mesh_smoothing: MeshSmoothing,
    mesh_error: Option<String>,
    mesh_revision: u64,
    filament_mesh_revision: Option<u64>,
    /// Lighting preset the renderer was last given, and the lux and rotation
    /// it was applied at.
    filament_environment: Option<(substancecraft_core::LightingPreset, f32, f32)>,
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
    /// Screen point where the last brush dab landed, for Shift-click lines.
    last_paint_screen: Option<egui::Pos2>,
    /// Control points of a path being placed (UV space).
    path_controls: Option<Vec<(glam::Vec2, egui::Pos2)>>,
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
    layout: panels::OverlayLayout,
    autosave: autosave::Autosave,
    preferences: preferences::PreferenceState,
    inspect: inspector::InspectState,
    edge_overlay: edges::EdgeOverlay,
    /// Resource paths as of the last reload, to notice relinks by undo.
    loaded_resource_sources: Vec<(ResourceId, String)>,
    bake_log: bake_log::BakeLog,
    cache: cache::CacheState,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewMode {
    Model,
    Split,
    Uv,
}

/// A texture set's last full-material bake.
struct CachedMaterial {
    revision: u64,
    textures: Arc<MaterialTextures>,
}

/// Background bake of a texture set that is not selected, so every material
/// slot of a multi-material mesh shows its own textures.
struct SetBakeJob {
    set_id: TextureSetId,
    revision: u64,
    receiver: Receiver<Result<MaterialTextures, String>>,
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
    /// Content keys of the textures being patched and the regions rebaked,
    /// when the job patches the shown preview.
    patch: Option<(Vec<usize>, Vec<TexelRect>)>,
}

/// How a shown preview differs from the one before it: each texture's
/// content key paired with the key of the texture it was patched from, and
/// the texels rebaked. The viewport re-uploads only those texels.
#[derive(Clone, Debug, PartialEq)]
struct PreviewPatch {
    keys: Vec<(usize, usize)>,
    regions: Vec<TexelRect>,
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

/// Shared inputs of a background texture-set bake.
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

struct BakeInputs {
    mesh: Option<Arc<MeshGeometry>>,
    mesh_maps: Option<Arc<Vec<MeshMaps>>>,
    /// Mesh material slot whose maps the set uses.
    slot: usize,
    resources: ResourceImages,
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
        bake_texture_set_material(
            texture_set,
            self.mesh.as_deref(),
            self.maps(),
            Some(&self.resources),
        )
        .map_err(|error| error.to_string())
    }

    /// A copy of `base`, a preview of `target` for an earlier state of
    /// `texture_set`, rebaked over `regions` only.
    fn patch_preview(
        &self,
        texture_set: &TextureSet,
        target: PreviewTarget,
        base: &PreviewTextures,
        regions: &[TexelRect],
    ) -> Result<PreviewTextures, String> {
        let (mesh, maps, resources) = (self.mesh.as_deref(), self.maps(), Some(&self.resources));
        match (base, target) {
            (PreviewTextures::Material(base), PreviewTarget::Material) => {
                let mut textures = MaterialTextures::clone(base);
                rebake_texture_set_material_regions(
                    texture_set,
                    mesh,
                    maps,
                    resources,
                    &mut textures,
                    regions,
                )
                .map_err(|error| error.to_string())?;
                Ok(PreviewTextures::Material(Arc::new(textures)))
            }
            (PreviewTextures::Single(base), PreviewTarget::Channel(channel)) => {
                let mut texture = BakedTexture::clone(base);
                rebake_texture_set_channel_regions(
                    texture_set,
                    channel,
                    mesh,
                    maps,
                    resources,
                    &mut texture,
                    regions,
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
}

/// Optional high-resolution source for high-to-low mesh-map baking.
struct HighPolySource {
    path: std::path::PathBuf,
    geometry: MeshGeometry,
}

struct MeshMapJob {
    receiver: Receiver<Result<Vec<MeshMaps>, MeshMapError>>,
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
                }),
                ..MaterialProject::empty()
            },
            selected_texture_set: None,
            selected_layer: None,
            selected_primitive: Some(PrimitivePreset::Cube),
            base_geometry: None,
            mesh_geometry: None,
            mesh_topology: None,
            mesh_center: glam::Vec3::ZERO,
            mesh_smoothing: MeshSmoothing::default(),
            mesh_error: None,
            mesh_revision: 1,
            filament_mesh_revision: None,
            filament_environment: None,
            filament_renderer: None,
            filament_error: None,
            filament_frame_texture: None,
            filament_frames_tx,
            filament_frames_rx,
            filament_render_pending: Arc::new(AtomicBool::new(false)),
            render_request: None,
            render_result: Arc::new(std::sync::Mutex::new(None)),
            render_status: None,
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
            layout: panels::OverlayLayout::default(),
            autosave: autosave::Autosave::default(),
            preferences: preferences::PreferenceState::default(),
            inspect: inspector::InspectState::default(),
            edge_overlay: edges::EdgeOverlay::default(),
            loaded_resource_sources: Vec::new(),
            bake_log: bake_log::BakeLog::default(),
            cache: cache::CacheState::default(),
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
        self.preview_revision = self.preview_revision.wrapping_add(1);
        self.preview_failed = false;
        self.preview_dirty = None;
        if let Some(set_id) = self.selected_texture_set {
            self.set_revisions.insert(set_id, self.preview_revision);
        }
    }

    /// Like [`Self::invalidate_preview`] for a stroke edit that changed only
    /// the texels in `bounds` (none at all when empty), so the next preview
    /// can rebake just those.
    fn invalidate_preview_texels(&mut self, bounds: impl IntoIterator<Item = TexelRect>) {
        let dirty = self.preview_dirty.take();
        self.invalidate_preview();
        self.preview_dirty = dirty
            .filter(|dirty| Some(dirty.set_id()) == self.selected_texture_set)
            .map(|mut dirty| {
                for bounds in bounds {
                    dirty.mark(bounds);
                }
                dirty
            });
    }

    /// Marks only `set_ids` out of date. When the selected set is not among
    /// them its current preview stays valid and is carried forward to the
    /// new revision instead of being rebaked.
    fn invalidate_sets(&mut self, set_ids: &BTreeSet<TextureSetId>) {
        if set_ids.is_empty() {
            return;
        }
        let selected_affected = self
            .selected_texture_set
            .is_some_and(|id| set_ids.contains(&id));
        let previous = self.preview_key();
        if selected_affected {
            self.invalidate_preview();
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
            .add_filter("3D mesh (FBX, glTF)", &MESH_EXTENSIONS)
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
        let Some(mesh) = self.mesh_geometry.as_deref() else {
            return;
        };
        let Some(renderer) = self.filament_renderer.as_mut() else {
            return;
        };
        if self.filament_mesh_revision != Some(self.mesh_revision) {
            match renderer.set_mesh(mesh, self.mesh_smoothing.angle_degrees) {
                Ok(()) => {
                    self.filament_mesh_revision = Some(self.mesh_revision);
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
            match renderer.update_vertices(shown, self.mesh_smoothing.angle_degrees) {
                Ok(()) => self.filament_pose = Some(pose),
                Err(error) => {
                    self.filament_error = Some(error.to_string());
                    return;
                }
            }
        }
        if let Err(error) = apply_environment(
            renderer,
            self.viewport_lighting,
            &mut self.filament_environment,
        ) {
            self.filament_error = Some(error.to_string());
            return;
        }

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
                let (_, base) = patch.keys.iter().find(|(patched, _)| *patched == key)?;
                Some((*base, patch.regions.as_slice()))
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
            PreviewTarget::Material | PreviewTarget::MaskOverlay { .. }
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
        self.render_status = Some("Rendering…".to_owned());
        self.render_request = Some(RenderRequest {
            path,
            long_edge,
            camera,
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
        self.render_status = Some(match write_texture(&request.path, &texture) {
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
                    .map(|error| format!("Saved, but the undo history was not: {error}"));
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
            .add_filter("3D mesh (FBX, glTF)", &MESH_EXTENSIONS)
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
            self.invalidate_preview();
        }
    }

    /// Flattens the selected layer or group into per-channel PNGs in
    /// `folder`, inserts a fill layer using them just above it, and hides
    /// the source, as one undoable edit. Returns the new layer.
    fn flatten_selected_into(&mut self, folder: &std::path::Path) -> Result<LayerId, String> {
        let (Some(set_id), Some(source_id)) = (self.selected_texture_set, self.selected_layer)
        else {
            return Err("Select a layer to flatten.".to_owned());
        };
        let set = self
            .selected_texture_set_ref()
            .cloned()
            .ok_or("No texture set is selected.")?;
        let source = set
            .layer(source_id)
            .cloned()
            .ok_or("The layer no longer exists.")?;
        let slot = set
            .slot_of(source_id)
            .ok_or("The layer no longer exists.")?;
        let flattened = {
            let inputs = self.bake_inputs(set_id);
            set.flatten_layer(
                source_id,
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
        let stem = format!("{}_{}", set.name, source.name).replace([' ', '/', '\\', ':'], "_");
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
        let layer = flattened.to_fill_layer(id, format!("{} (flattened)", source.name), &resources);
        let before = source.settings();
        let after = substancecraft_core::LayerSettings {
            visible: false,
            ..before.clone()
        };
        let applied = self.history.apply(
            &mut self.project,
            EditCommand::Batch(vec![
                EditCommand::LayerSettings {
                    texture_set_id: set_id,
                    layer_id: source_id,
                    before: Box::new(before),
                    after: Box::new(after),
                },
                EditCommand::InsertLayer {
                    texture_set_id: set_id,
                    slot: LayerSlot {
                        parent: slot.parent,
                        index: slot.index + 1,
                    },
                    layer: Box::new(layer),
                },
            ]),
        );
        if !applied {
            return Err("The flattened layer could not be inserted.".to_owned());
        }
        self.selected_layer = Some(id);
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
            self.invalidate_sets(&touched);
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
            self.invalidate_preview();
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
                self.invalidate_preview();
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
        self.stroke_point(0, uv, false);
    }

    /// True when strokes are drawn on screen and projected onto the surface:
    /// projection is on and the mode paints, erases, or masks (ID strokes
    /// stay in UV space).
    fn projects_strokes(&self) -> bool {
        // Projected strokes rasterise the bind mesh, so they wait for it.
        self.projection_painting
            && !self.is_posed()
            && matches!(
                self.paint_mode,
                PaintMode::Paint | PaintMode::Erase | PaintMode::Mask
            )
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
        let (Some(texture_set_id), Some(layer_id)) =
            (self.selected_texture_set, self.selected_layer)
        else {
            return;
        };
        if !self.projects_strokes() {
            return;
        }
        let mode = self.paint_mode;
        let stroke_key = (texture_set_id, layer_id, mode);
        let starts_stroke = self.active_stroke != Some(stroke_key);
        let pressure = self.pen_pressure;
        let pressures = pressure.map(|value| vec![value]).unwrap_or_default();
        let paint = PaintStroke {
            camera: Some(camera),
            pressures: pressures.clone(),
            ..self.brush_paint_stroke(mode == PaintMode::Erase)
        };
        let (radius, tip) = (paint.radius_pixels, paint.tip);
        let mask = MaskStroke {
            camera: Some(camera),
            tip,
            pressures,
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
    /// 1 its mirror). `restart` begins a new stroke for that path, e.g. after
    /// crossing a UV seam. Every stroke a drag creates undoes together.
    fn stroke_point(&mut self, path: usize, uv: glam::Vec2, restart: bool) {
        let (Some(texture_set_id), Some(layer_id)) =
            (self.selected_texture_set, self.selected_layer)
        else {
            return;
        };
        if !self.paint_mode.paints() {
            return;
        }
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
                }),
                PaintMode::Id => layer.id_strokes.push(IdStroke {
                    points: vec![uv],
                    radius_pixels,
                    color: [color[0], color[1], color[2]],
                    opacity: 1.0,
                    tip,
                    pressures: pressures(pressure),
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
                    (&mut stroke.points, &mut stroke.pressures, shape)
                })
            }
            PaintMode::Mask => layer.mask_strokes.get_mut(index).map(|stroke| {
                let shape = (stroke.radius_pixels, stroke.tip);
                (&mut stroke.points, &mut stroke.pressures, shape)
            }),
            PaintMode::Id => layer.id_strokes.get_mut(index).map(|stroke| {
                let shape = (stroke.radius_pixels, stroke.tip);
                (&mut stroke.points, &mut stroke.pressures, shape)
            }),
        };
        if let Some((points, pressures, (radius, tip))) = target
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
            // Stamps are placed causally, so only the new segment changes.
            let bounds = stroke_texel_bounds(
                &points[points.len() - 2..],
                preview_radius(radius),
                tip,
                set_width,
                set_height,
            );
            self.invalidate_preview_texels(bounds);
        }
    }

    fn finish_active_stroke(&mut self) {
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
            pick_triangle(mesh, points, at).filter(|hit| app.triangle_in_selected_set(hit.triangle))
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
        if !self.paint_mode.paints() {
            return;
        }
        if self.path_controls.is_some() {
            if response.clicked()
                && let Some(uv) = pick(self, pointer).and_then(|hit| hit.uv)
                && let Some(controls) = &mut self.path_controls
            {
                controls.push((uv, pointer));
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

    /// Turns the placed path into one undoable stroke on the selected
    /// layer, using the current brush.
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
            controls: controls.into_iter().map(|(uv, _)| uv).collect(),
            closed,
        };
        let template = self.brush_paint_stroke(self.paint_mode == PaintMode::Erase);
        let stroke = PaintStroke::along_path(path, &template);
        if self.history.apply(
            &mut self.project,
            EditCommand::Paint {
                texture_set_id,
                layer_id,
                stroke,
            },
        ) {
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

    /// Paints at a picked surface point: continues the brush stroke
    /// (restarting it when the pointer crosses into another UV island) and,
    /// with symmetry on, the mirrored stroke on the other side.
    fn paint_hit(&mut self, hit: TriangleHit) {
        let Some(uv) = hit
            .uv
            .filter(|uv| uv.cmpge(glam::Vec2::ZERO).all() && uv.cmple(glam::Vec2::ONE).all())
        else {
            return;
        };
        if self.active_stroke.is_none() {
            self.stroke_last_triangles.clear();
        }
        let mut targets = vec![(0, hit.triangle, uv)];
        if let (Some(symmetry), Some(mesh)) = (self.symmetry, self.mesh_geometry.as_deref()) {
            let copies = symmetry.symmetric_hits(
                mesh,
                &substancecraft_core::SurfaceHit {
                    triangle: hit.triangle,
                    position: hit.position,
                    normal: glam::Vec3::Z,
                    uv,
                },
            );
            // Path numbers follow the copy order, so each copy keeps its own
            // stroke through the drag.
            for (index, copy) in copies.into_iter().enumerate() {
                if let Some(copy) = copy
                    && self.triangle_in_selected_set(copy.triangle)
                    && copy.uv.cmpge(glam::Vec2::ZERO).all()
                    && copy.uv.cmple(glam::Vec2::ONE).all()
                {
                    targets.push((index + 1, copy.triangle, copy.uv));
                }
            }
        }
        for (path, triangle, uv) in targets {
            let previous = self.stroke_last_triangles.get(path).copied().flatten();
            let restart = match (previous, &self.mesh_topology) {
                (Some(previous), Some(topology)) => {
                    substancecraft_core::crosses_uv_seam(topology, previous, triangle)
                }
                _ => false,
            };
            self.stroke_point(path, uv, restart);
            if self.stroke_last_triangles.len() <= path {
                self.stroke_last_triangles.resize(path + 1, None);
            }
            self.stroke_last_triangles[path] = Some(triangle);
        }
    }

    fn undo(&mut self) {
        self.finish_active_stroke();
        if self.history.undo(&mut self.project) {
            self.after_history_change();
        }
    }

    fn redo(&mut self) {
        self.finish_active_stroke();
        if self.history.redo(&mut self.project) {
            self.after_history_change();
        }
    }

    fn after_history_change(&mut self) {
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
        self.invalidate_all_sets();
        self.refresh_baked_preview();
    }

    fn preview_target(&self) -> PreviewTarget {
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
            self.log_bake(&job, bake_log::BakeOutcome::Cancelled);
        }
    }

    fn log_bake(&mut self, job: &MeshMapJob, outcome: bake_log::BakeOutcome) {
        self.bake_log.record(bake_log::BakeLogEntry {
            started: job.started_at,
            duration: job.started.elapsed(),
            settings: job.settings.clone(),
            outcome,
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
        let skew_map = self
            .skew_map
            .and_then(|id| self.resource_images.get(&id).cloned());
        let settings = self.effective_mesh_map_settings();
        let cancel = Arc::new(AtomicBool::new(false));
        let summary = bake_log::settings_summary(&settings, high.is_some());
        let (sender, receiver) = sync_channel(1);
        let worker_cancel = Arc::clone(&cancel);
        let spawned = std::thread::Builder::new()
            .name("mesh-map-bake".to_owned())
            .spawn(move || {
                let _ = sender.send(bake_slot_mesh_maps(
                    &mesh,
                    high.as_ref(),
                    &settings,
                    skew_map.as_deref(),
                    &worker_cancel,
                ));
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
            Ok(result) => {
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
                self.log_bake(&job, outcome);
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
        self.high_poly_revision = self.high_poly_revision.wrapping_add(1);
        self.cancel_mesh_map_job();
        self.mesh_map_error = None;
        self.invalidate_all_sets();
    }

    fn pick_high_poly(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D mesh (FBX, glTF)", &MESH_EXTENSIONS)
            .pick_file()
        else {
            return;
        };
        match load_merged_mesh(&path, self.import_scale) {
            Ok(geometry) => {
                let before = self.project.mesh.clone();
                if let Some(mesh) = self.project.mesh.as_mut() {
                    mesh.high_poly_source = Some(path.to_string_lossy().into_owned());
                    mesh.high_poly_hash = substancecraft_api::file_content_hash(&path).ok();
                }
                self.history.end_merge();
                self.record_mesh_edit(before, Vec::new());
                self.history.end_merge();
                self.high_poly = Some(HighPolySource { path, geometry });
                self.high_poly_changed();
                self.start_mesh_map_bake();
            }
            Err(error) => self.mesh_map_error = Some(format!("High poly: {error}")),
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

    /// Reloads the project's high-poly reference, reporting (not failing on)
    /// a missing or unreadable file.
    fn restore_high_poly(&mut self) {
        self.high_poly = None;
        let source = self
            .project
            .mesh
            .as_ref()
            .and_then(|mesh| mesh.high_poly_source.clone());
        if let Some(source) = source {
            let path = std::path::PathBuf::from(source);
            match load_merged_mesh(&path, self.import_scale) {
                Ok(geometry) => self.high_poly = Some(HighPolySource { path, geometry }),
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

    /// The shown preview and the regions to rebake to bring it up to date
    /// for `key`, when it shows the same view and only strokes changed the
    /// set since it was baked; `None` means a full rebake.
    fn preview_patch(&self, key: &PreviewKey) -> Option<(PreviewTextures, Vec<TexelRect>)> {
        let preview = self.baked_preview.as_ref()?;
        if preview.key.texture_set != key.texture_set
            || preview.key.target != key.target
            || preview.key.solo_layer != key.solo_layer
        {
            return None;
        }
        let set = preview_copy(self.selected_texture_set_ref()?);
        let regions = self
            .preview_dirty
            .as_ref()?
            .regions(&set, preview.set_revision)?;
        let textures = match (&preview.textures, key.target) {
            (PreviewTextures::Material(material), PreviewTarget::Material) => {
                PreviewTextures::Material(Arc::clone(material))
            }
            (PreviewTextures::Single(texture), PreviewTarget::Channel(_)) => {
                PreviewTextures::Single(Arc::clone(texture))
            }
            _ => return None,
        };
        Some((textures, regions))
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
            .downscaled(PREVIEW_MAX_EDGE)
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
            PreviewTarget::Material | PreviewTarget::Channel(_) => self
                .selected_texture_set_ref()
                .map(|set| incremental::PreviewDirty::new(&preview_copy(set), set_revision)),
            _ => None,
        };
        if patch
            .as_ref()
            .is_some_and(|(_, regions)| regions.is_empty())
        {
            // Nothing the preview shows changed.
            if let Some(preview) = &mut self.baked_preview {
                preview.key = key;
                preview.set_revision = set_revision;
                if let (PreviewTextures::Material(material), Some(set_id)) =
                    (&preview.textures, key.texture_set)
                {
                    self.set_materials.insert(
                        set_id,
                        CachedMaterial {
                            revision: set_revision,
                            textures: Arc::clone(material),
                        },
                    );
                }
            }
            return;
        }
        let job_patch = patch
            .as_ref()
            .map(|(base, regions)| (base.keys(), regions.clone()));
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
        }
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
                        let patch =
                            job_patch
                                .filter(|_| was_patched)
                                .map(|(base, regions)| PreviewPatch {
                                    keys: textures.keys().into_iter().zip(base).collect(),
                                    regions,
                                });
                        if let (PreviewTextures::Material(material), Some(set_id)) =
                            (&textures, key.texture_set)
                        {
                            self.set_materials.insert(
                                set_id,
                                CachedMaterial {
                                    revision: set_revision,
                                    textures: Arc::clone(material),
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
            .downscaled(PREVIEW_MAX_EDGE)
            .unwrap_or(texture_set);
        let revision = self.set_revision(set_id);
        let inputs = self.bake_inputs(set_id);
        let (sender, receiver) = sync_channel(1);
        let spawned = std::thread::Builder::new()
            .name("set-bake".to_owned())
            .spawn(move || {
                let _ = sender.send(inputs.bake_material(&texture_set));
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
            Ok(result) => {
                let (set_id, revision) = (job.set_id, job.revision);
                self.set_bake_job = None;
                match result {
                    Ok(textures) => {
                        self.set_materials.insert(
                            set_id,
                            CachedMaterial {
                                revision,
                                textures: Arc::new(textures),
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
        let stem = format!("{}_{}", texture_set.name, channel_label(channel));
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
            .map(|resource| (resource.id, resource.source.clone()))
            .collect();
        self.resource_images = images;
        self.resource_thumbnails.clear();
        self.resource_watcher.reset();
        for resource in &self.project.resources {
            self.resource_watcher
                .remember(resource.id, std::path::Path::new(&resource.source));
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
                "Bitmap, SVG, or text (PNG, JPEG, TGA, SVG, SCTEXT)",
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

        if self.project.resources.is_empty() || self.resource_watcher.mode == ReloadMode::Off {
            return;
        }
        // Keep polling while the window is idle.
        context.request_repaint_after(SCAN_INTERVAL);
        let now = std::time::Instant::now();
        if !self.resource_watcher.due(now) {
            return;
        }
        let changed = self.resource_watcher.scan(&self.project.resources, now);
        match self.resource_watcher.mode {
            ReloadMode::Auto => self.reload_changed_resources(&changed),
            ReloadMode::Ask => self.resource_watcher.pending.extend(changed),
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
        self.project
            .resources
            .iter()
            .filter(|resource| self.resource_watcher.pending.contains(&resource.id))
            .map(|resource| resource.name.clone())
            .collect()
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
                        let middle = area.center().x;
                        (
                            Some(area.with_max_x(middle - 1.0)),
                            Some(area.with_min_x(middle + 1.0)),
                        )
                    }
                };
                // Runs during strokes too: previews bake in the background, so the
                // stroke in progress shows up live.
                self.refresh_baked_preview();
                if let Some(rect) = model {
                    self.model_viewport(ui, rect);
                }
                if let Some(rect) = uv {
                    self.uv_viewport(ui, rect);
                }
                if model.is_some() && uv.is_some() {
                    ui.painter().vline(
                        area.center().x,
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
        if response.dragged() && !painting {
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
            || (selecting && response.hovered());
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
            let screen: Vec<egui::Pos2> = controls.iter().map(|(_, screen)| *screen).collect();
            painter.add(egui::Shape::line(
                screen.clone(),
                egui::Stroke::new(1.5, theme::ACCENT),
            ));
            for point in screen {
                painter.circle_filled(point, 4.0, theme::ACCENT);
            }
        }

        if selecting {
            let hit = match (&projected, hover_position, &self.mesh_geometry) {
                (Some(points), Some(pointer), Some(mesh)) if response.hovered() => {
                    pick_triangle(mesh, points, pointer)
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
            let footprint = (!self.projects_strokes())
                .then(|| self.brush_footprint(mesh, &projection, projected.as_deref(), pointer))
                .flatten();
            match footprint {
                Some(outline) => {
                    painter.add(egui::Shape::closed_line(
                        outline,
                        egui::Stroke::new(1.5, cursor_color),
                    ));
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

    /// The screen outline of the brush footprint on the surface under
    /// `pointer`: the brush's texel radius in the selected set, carried
    /// through the hovered triangle's UVs. `None` off the mesh.
    fn brush_footprint(
        &self,
        mesh: &MeshGeometry,
        projection: &ViewProjection,
        points: Option<&[ProjectedVertex]>,
        pointer: egui::Pos2,
    ) -> Option<Vec<egui::Pos2>> {
        let hit = pick_triangle(mesh, points?, pointer)?;
        let set = self.selected_texture_set_ref()?;
        let uv_radius = glam::Vec2::new(
            self.brush_radius / set.resolution.width.max(1) as f32,
            self.brush_radius / set.resolution.height.max(1) as f32,
        );
        let outline = mesh.uv_brush_outline(hit.triangle, hit.position, uv_radius, 40)?;
        // The outline is found on the bind mesh; a pose carries it along.
        let posed = self.is_posed().then(|| self.shown_mesh()).flatten();
        Some(
            outline
                .into_iter()
                .map(|point| match &posed {
                    Some(posed) => animation::carry_point(mesh, posed, hit.triangle, point),
                    None => point,
                })
                .map(|point| projection.project(point).screen)
                .collect(),
        )
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
) -> Option<TriangleHit> {
    let mut best: Option<(f32, TriangleHit)> = None;
    for (index, triangle) in mesh.triangles.iter().enumerate() {
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

/// Brings the renderer's environment up to `lighting`: a new preset is
/// rendered and prefiltered again; a new strength or rotation only turns and
/// rescales the one already there. `applied` records what the renderer has.
fn apply_environment(
    renderer: &mut FilamentRuntime,
    lighting: ViewportLighting,
    applied: &mut Option<(substancecraft_core::LightingPreset, f32, f32)>,
) -> Result<(), substancecraft_filament::FilamentError> {
    let preset = lighting.preset;
    let lux = preset.environment_lux() * lighting.environment_scale();
    let rotation = if lighting.rotation.is_finite() {
        lighting.rotation
    } else {
        0.0
    };
    match *applied {
        Some(current) if current == (preset, lux, rotation) => {}
        Some((current, _, _)) if current == preset => {
            renderer.set_environment_transform(lux, rotation, preset.sun());
            *applied = Some((preset, lux, rotation));
        }
        _ => {
            let image = preset.equirect(substancecraft_core::environment::ENVIRONMENT_WIDTH);
            renderer.set_environment(&image, lux, rotation, preset.sun())?;
            *applied = Some((preset, lux, rotation));
        }
    }
    Ok(())
}

/// Reads an FBX (every mesh it holds) or a glTF/GLB (the flattened scene),
/// choosing the importer by extension.
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

/// Reads every mesh in a file and merges them into one scaled geometry, as
/// a high-poly source for baking.
fn load_merged_mesh(path: &std::path::Path, scale: f32) -> Result<MeshGeometry, String> {
    let mut merged = MeshGeometry::default();
    for mesh in load_mesh_file(path)? {
        merged
            .append(&mesh.geometry)
            .map_err(|_| "merged mesh has too many vertices".to_owned())?;
    }
    if merged.triangles.is_empty() {
        return Err("the file contains no mesh geometry".to_owned());
    }
    Ok(scaled_geometry(&merged, sanitize_mesh_scale(scale)))
}

/// Bakes mesh maps for each material slot from that slot's triangles, so
/// slots sharing UV space get their own maps. A single-slot mesh bakes
/// whole; an empty slot falls back to the whole mesh.
fn bake_slot_mesh_maps(
    mesh: &MeshGeometry,
    high: Option<&MeshGeometry>,
    settings: &MeshMapSettings,
    skew_map: Option<&substancecraft_core::TextureImage>,
    cancel: &AtomicBool,
) -> Result<Vec<MeshMaps>, MeshMapError> {
    let slot_count = mesh.material_slot_count();
    if slot_count <= 1 {
        return Ok(vec![substancecraft_core::bake_mesh_maps_corrected(
            mesh,
            high,
            settings,
            skew_map,
            Some(cancel),
        )?]);
    }
    (0..slot_count as u32)
        .map(|slot| {
            let submesh = mesh.submesh(slot);
            let source = if submesh.triangles.is_empty() {
                mesh
            } else {
                &submesh
            };
            substancecraft_core::bake_mesh_maps_corrected(
                source,
                high,
                settings,
                skew_map,
                Some(cancel),
            )
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

fn channel_label(channel: MaterialChannel) -> &'static str {
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
        let hit = pick_triangle(&cube, &points, egui::pos2(110.0, 90.0)).unwrap();
        assert!(hit.triangle < 2, "picked {}", hit.triangle);
        assert!(hit.uv.is_some());
        assert_eq!(
            pick_triangle(&cube, &points, egui::pos2(400.0, 400.0)),
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
        // the shape check, so the next stroke rebakes the whole set.
        app.selected_layer_mut().unwrap().opacity = 0.5;
        app.append_stroke_point(glam::Vec2::new(0.7, 0.7));
        assert!(app.preview_patch(&app.preview_key()).is_none());
        pump_until(&mut app, &context, settled);
        assert!(shown(&app) == full_bake(&app));
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
        app.path_controls = Some(vec![
            (glam::Vec2::new(0.1, 0.6), egui::Pos2::ZERO),
            (glam::Vec2::new(0.2, 0.8), egui::Pos2::ZERO),
            (glam::Vec2::new(0.3, 0.6), egui::Pos2::ZERO),
        ]);
        app.commit_path(false);
        assert!(app.path_controls.is_none());
        let path_stroke = app
            .selected_layer()
            .unwrap()
            .paint_strokes
            .last()
            .unwrap()
            .clone();
        assert_eq!(path_stroke.path.as_ref().unwrap().controls.len(), 3);
        assert_eq!(
            path_stroke.points.len(),
            2 * substancecraft_core::PATH_SAMPLES_PER_SEGMENT + 1
        );
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
        let first_hit = pick_triangle(&mesh, &projected, first).unwrap();
        app.paint_hit(first_hit);

        assert_eq!(
            app.stabilized_cursor(egui::pos2(106.0, 108.0)),
            None,
            "cursor jitter within the lazy radius does not add a dab"
        );
        let trailing = app.stabilized_cursor(egui::pos2(130.0, 100.0)).unwrap();
        assert_eq!(trailing, egui::pos2(120.0, 100.0));
        app.paint_hit(pick_triangle(&mesh, &projected, trailing).unwrap());
        app.finish_active_stroke();

        let recorded = &app.selected_layer().unwrap().paint_strokes[0].points;
        let expected = pick_triangle(&mesh, &projected, trailing)
            .unwrap()
            .uv
            .unwrap();
        let raw = pick_triangle(&mesh, &projected, egui::pos2(130.0, 100.0))
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
                Some(preset)
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
        let (preset, _, rotation) = app.filament_environment.unwrap();
        assert_eq!(preset, LightingPreset::Night);
        assert_eq!(rotation, app.viewport_lighting.rotation);
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
        let hit = pick_triangle(&bind_mesh, &points, centroid.to_pos2()).unwrap();
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
}
