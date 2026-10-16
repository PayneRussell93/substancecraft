//! Release-compatibility targets (docs/compatibility.md): files written by
//! earlier releases keep opening, and the pipeline-facing contracts keep
//! their shape. The fixtures under `tests/fixtures` are frozen: a release
//! that changes them breaks a published target.

#![allow(clippy::unwrap_used, clippy::panic)]

use std::path::{Path, PathBuf};

use substancecraft_api::{
    AUTOMATION_API_VERSION, AutomationEdit, ExportJob, SHARED_LIBRARY_FORMAT, SharedLibrary,
    edit_project_file, load_project_file_with_report, project_summary, run_export_job,
};
use substancecraft_core::{AssetKind, MaterialChannel, PROJECT_SCHEMA_VERSION};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn tc_0394_projects_from_every_published_schema_open_and_bake_alike() {
    for schema in 1..=PROJECT_SCHEMA_VERSION {
        let path = fixture(&format!("project-schema-{schema}.sproject"));
        let (project, report) = load_project_file_with_report(&path)
            .unwrap_or_else(|error| panic!("schema {schema}: {error}"));
        assert_eq!(report.from_version, schema);
        assert_eq!(project.schema_version, PROJECT_SCHEMA_VERSION);
        assert!(
            report.unknown.is_empty(),
            "schema {schema}: {:?}",
            report.unknown
        );
        // Materials gained ids in schema 3; older files get them on load.
        assert_eq!(project.materials[0].id.0, 1);
        let set = &project.texture_sets[0];
        let names: Vec<&str> = set.layers.iter().map(|layer| layer.name.as_str()).collect();
        assert_eq!(names, ["Paint", "Scratches"]);
        let base = substancecraft_api::bake_texture_set_channel(
            set,
            MaterialChannel::BaseColor,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            &base.pixel(2, 2).unwrap()[..3],
            &[180, 40, 30],
            "schema {schema}"
        );
    }

    // A file from a newer release is refused with its version named.
    let text = std::fs::read_to_string(fixture("project-schema-3.sproject")).unwrap();
    let newer = text.replace(
        "\"schema_version\": 3",
        &format!("\"schema_version\": {}", PROJECT_SCHEMA_VERSION + 1),
    );
    let folder = tempfile::tempdir().unwrap();
    let path = folder.path().join("newer.sproject");
    std::fs::write(&path, newer).unwrap();
    let error = load_project_file_with_report(&path)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(&(PROJECT_SCHEMA_VERSION + 1).to_string()),
        "{error}"
    );
}

#[test]
fn tc_0395_pipeline_contracts_keep_their_published_shape() {
    let folder = tempfile::tempdir().unwrap();

    // Built-in export presets keep their file names.
    let project = folder.path().join("project.sproject");
    std::fs::copy(fixture("project-schema-1.sproject"), &project).unwrap();
    let job = ExportJob::new(&project, folder.path().join("out"));
    let names: Vec<String> = run_export_job(&job)
        .unwrap()
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    for expected in ["Body_BaseColor.png", "Body_ORM.png", "Body_Normal.png"] {
        assert!(names.iter().any(|name| name == expected), "{names:?}");
    }

    // Automation API v1: the version, summary fields, and edit names.
    assert_eq!(AUTOMATION_API_VERSION, 1);
    let summary = serde_json::to_value(project_summary(
        &substancecraft_api::load_project_file(&project).unwrap(),
    ))
    .unwrap();
    for field in ["api_version", "texture_sets"] {
        assert!(summary.get(field).is_some(), "summary lost `{field}`");
    }
    let edit: AutomationEdit = serde_json::from_str(
        r#"{"edit":"set_layer","texture_set":"Body","layer":"Scratches","opacity":0.5}"#,
    )
    .unwrap();
    edit_project_file(&project, &edit).unwrap();

    // Shared library asset format 1 stays readable.
    const { assert!(SHARED_LIBRARY_FORMAT >= 1) };
    let assets = folder.path().join("library/assets");
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::copy(
        fixture("shared-asset-format-1.scasset"),
        assets.join("filter.scasset"),
    )
    .unwrap();
    let library = SharedLibrary::open(folder.path().join("library"));
    assert!(library.problems().is_empty(), "{:?}", library.problems());
    assert!(
        library
            .find(AssetKind::Filter, "Release 1 soften")
            .is_some()
    );
}
