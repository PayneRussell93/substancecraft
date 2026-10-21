//! PROD-07: the sample plugin in `examples/plugins/wavy-bands`, built
//! outside this workspace, registers a generator, a filter, and an export
//! preset that a headless export uses with no change to any crate here.
//!
//! CI builds the plugin (`cargo build --release --target
//! wasm32-unknown-unknown` in its folder) before running this test. Locally
//! the test is skipped until it has been built; `SUBSTANCECRAFT_SAMPLE_PLUGIN`
//! may point at the module instead.

#![allow(clippy::unwrap_used, clippy::panic)]

use std::path::{Path, PathBuf};

use substancecraft_api::{
    ExportJob, PLUGIN_MANIFEST, TextureFormat, install_plugins, installed_plugins,
    list_export_presets, load_plugins, load_project_file_with_report, load_texture_image,
    run_export_job, save_project_file,
};
use substancecraft_core::{
    LayerFill, LayerFilter, LayerId, MaskGenerator, MaterialChannel, MaterialLayer,
    MaterialProject, MeshAsset, MeshId, MeshSmoothing, TextureResolution, TextureSet, TextureSetId,
};

const PRESET: &str = "Packed RMA (wavy-bands sample)";

fn sample_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/plugins/wavy-bands")
}

/// The sample plugin's folder with its built module beside the manifest,
/// or `None` when it has not been built (never in CI).
fn staged_plugin(into: &Path) -> Option<PathBuf> {
    let module = std::env::var_os("SUBSTANCECRAFT_SAMPLE_PLUGIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            sample_root().join("target/wasm32-unknown-unknown/release/wavy_bands.wasm")
        });
    if !module.is_file() {
        assert!(
            std::env::var_os("CI").is_none(),
            "CI must build the sample plugin first: {}",
            module.display()
        );
        eprintln!("skipped: build examples/plugins/wavy-bands first");
        return None;
    }
    let folder = into.join("wavy-bands");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::copy(
        sample_root().join(PLUGIN_MANIFEST),
        folder.join(PLUGIN_MANIFEST),
    )
    .unwrap();
    std::fs::copy(&module, folder.join("wavy_bands.wasm")).unwrap();
    Some(folder)
}

fn project(wavy: MaskGenerator, sepia: Option<LayerFilter>) -> MaterialProject {
    let mut metal = MaterialLayer::new_fill(
        LayerId(1),
        "Banded metal",
        LayerFill {
            color: [70, 140, 220, 255],
            metallic: 1.0,
            roughness: 0.2,
            ..LayerFill::default()
        },
    );
    metal.mask_generator = Some(wavy);
    if let Some(sepia) = sepia {
        metal.filters = vec![sepia];
        metal.filter_channels = [MaterialChannel::BaseColor].into();
    }
    MaterialProject {
        mesh: Some(MeshAsset {
            id: MeshId(1),
            source: "builtin:Sphere".to_owned(),
            scale: 1.0,
            smoothing: MeshSmoothing::default(),
            high_poly_source: None,
            content_hash: None,
            high_poly_hash: None,
            high_to_low: Default::default(),
        }),
        texture_sets: vec![TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: TextureSetId(1),
            name: "Probe".to_owned(),
            material_slot: "Default".to_owned(),
            resolution: TextureResolution {
                width: 64,
                height: 64,
            },
            channels: [MaterialChannel::BaseColor].into(),
            layers: vec![metal],
            height_strength: 1.0,
            export: None,
        }],
        active_texture_set: Some(TextureSetId(1)),
        ..MaterialProject::empty()
    }
}

fn export(folder: &Path, name: &str, project: &MaterialProject) -> Vec<PathBuf> {
    let path = folder.join(format!("{name}.sproject"));
    save_project_file(&path, project).unwrap();
    let mut job = ExportJob::new(&path, folder.join(name));
    job.preset = Some(PRESET.to_owned());
    job.format = Some(TextureFormat::Png);
    job.mesh_maps.ao_samples = 2;
    run_export_job(&job).unwrap()
}

#[test]
fn tc_0410_an_out_of_tree_plugin_extends_generators_filters_and_presets() {
    let folder = tempfile::tempdir().unwrap();
    let Some(plugin_folder) = staged_plugin(folder.path()) else {
        return;
    };
    let (plugins, errors) = load_plugins(&[folder.path().to_owned()]);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(plugins.len(), 1);
    assert_eq!(plugins[0].folder(), plugin_folder);
    install_plugins(plugins);
    let plugin = installed_plugins().remove(0);
    let manifest = plugin.manifest();
    let wavy = plugin.default_stage(&manifest.generators[0]).unwrap();
    let sepia = plugin.default_stage(&manifest.filters[0]).unwrap();
    assert_eq!(wavy.display_name(), "Wavy bands");
    assert!(
        list_export_presets()
            .iter()
            .any(|preset| preset.name == PRESET)
    );

    // The plugin's preset names the files; its generator bands the metal.
    let plain = project(MaskGenerator::Plugin(wavy), None);
    let files = export(folder.path(), "plain", &plain);
    let names: Vec<String> = files
        .iter()
        .map(|file| file.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["Probe_BaseColor.png", "Probe_RMA.png"]);
    let rma = load_texture_image(&files[1]).unwrap();
    let metallic: Vec<u8> = rma.rgba8.chunks(4).map(|texel| texel[1]).collect();
    assert!(metallic.iter().any(|value| *value > 240), "no bands");
    assert!(metallic.iter().any(|value| *value < 15), "no gaps");

    // Its parameters reach the module: more bands, different pixels.
    let mut busy = wavy;
    busy.parameters[0] = 32.0;
    let busier = export(
        folder.path(),
        "busy",
        &project(MaskGenerator::Plugin(busy), None),
    );
    assert_ne!(
        std::fs::read(&files[1]).unwrap(),
        std::fs::read(&busier[1]).unwrap()
    );

    // The sepia filter tones the base colour: the blue paint warms up.
    let toned = export(
        folder.path(),
        "toned",
        &project(
            MaskGenerator::Plugin(wavy),
            Some(LayerFilter::Plugin(sepia)),
        ),
    );
    let blue = |file: &Path| {
        let image = load_texture_image(file).unwrap();
        image
            .rgba8
            .chunks(4)
            .map(|texel| i64::from(texel[2]) - i64::from(texel[0]))
            .sum::<i64>()
    };
    assert!(blue(&toned[0]) < blue(&files[0]));

    // Projects keep plugin stages through a save, and exports repeat
    // byte for byte.
    let (loaded, _) = load_project_file_with_report(folder.path().join("toned.sproject")).unwrap();
    assert_eq!(
        loaded.texture_sets[0].layers[0].mask_generator,
        Some(MaskGenerator::Plugin(wavy))
    );
    let again = export(
        folder.path(),
        "again",
        &project(
            MaskGenerator::Plugin(wavy),
            Some(LayerFilter::Plugin(sepia)),
        ),
    );
    for (first, second) in toned.iter().zip(&again) {
        assert_eq!(
            std::fs::read(first).unwrap(),
            std::fs::read(second).unwrap()
        );
    }

    // Without the plugin, the bake refuses rather than dropping the stage.
    install_plugins(Vec::new());
    let path = folder.path().join("toned.sproject");
    let mut job = ExportJob::new(&path, folder.path().join("missing"));
    job.mesh_maps.ao_samples = 2;
    let error = run_export_job(&job).unwrap_err().to_string();
    assert!(error.contains("com.example.wavy-bands/wavy"), "{error}");
    assert!(error.contains("not installed"), "{error}");
}
