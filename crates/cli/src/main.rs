#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

//! `substancecraft-cli`: headless jobs for pipelines.
//!
//! ```text
//! substancecraft-cli [--plugins <folder>]... <command>
//! substancecraft-cli export <project.sproject> --out <dir>
//!     [--preset <name>] [--format png|tga|exr] [--udim] [--ao-samples <n>]
//!     [--ocio <config.ocio> --colorspace <name>]
//! substancecraft-cli batch <jobs.txt> [--workers <n>]
//! substancecraft-cli presets
//! substancecraft-cli plugins
//! substancecraft-cli inspect <project.sproject>
//! substancecraft-cli edit <project.sproject> '<edit as JSON>'
//! substancecraft-cli ocio <config.ocio>
//! substancecraft-cli ocio-bake <config.ocio> --display <name> --view <name> --out <file.cube>
//!     [--input <colorspace>] [--size <n>]
//! ```
//!
//! A job file lists one export per line, written as the arguments after
//! `export` (double quotes group paths with spaces; `#` starts a comment).
//! Relative paths resolve against the job file. Every job runs even when
//! others fail, and the batch fails if any job did.
//!
//! `inspect` prints a JSON summary of a project; `edit` applies one
//! automation edit (for example
//! `{"edit":"set_layer","texture_set":"Body","layer":"Dirt","opacity":0.5}`)
//! through the project's undoable command model, saves the project, and
//! prints the outcome as JSON.
//!
//! `--ocio` writes colour outputs (Base Color, Emissive) in a colour space of
//! an OpenColorIO config, such as ACEScg for EXR exports. `ocio` lists a
//! config's displays, views, and colour spaces as JSON; `ocio-bake` bakes a
//! view into the `.cube` display LUT the viewport and renders apply.
//!
//! Plugins (generators, filters, and export presets; `docs/plugins.md`)
//! load from each `--plugins` folder, the folders in
//! `SUBSTANCECRAFT_PLUGINS`, and the user's plugin folder. `plugins` lists
//! what loaded, and why anything did not, as JSON.
//!
//! Exit status is 0 on success, 1 when the job fails, and 2 for bad
//! arguments. The same project and options always write the same bytes.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use substancecraft_api::{
    AutomationEdit, ExportJob, OCIO_BAKE_SIZE, OcioConfig, OcioExport, PluginError, TextureFormat,
    default_plugin_folders, edit_project_file, install_plugins, installed_plugins,
    list_export_presets, load_plugins, load_project_file_with_report, project_summary,
    run_export_job, run_export_jobs,
};
use substancecraft_core::MAX_CUBE_LUT_SIZE;

const USAGE: &str = "usage:\n  substancecraft-cli [--plugins <folder>]... <command>\n  substancecraft-cli plugins\n  substancecraft-cli export <project.sproject> --out <dir> [--preset <name>] [--format png|tga|exr] [--udim] [--ao-samples <n>] [--ocio <config.ocio> --colorspace <name>]\n  substancecraft-cli batch <jobs.txt> [--workers <n>]\n  substancecraft-cli presets\n  substancecraft-cli inspect <project.sproject>\n  substancecraft-cli edit <project.sproject> '<edit as JSON>'\n  substancecraft-cli ocio <config.ocio>\n  substancecraft-cli ocio-bake <config.ocio> --display <name> --view <name> --out <file.cube> [--input <colorspace>] [--size <n>]";

/// What the command line asks for.
#[derive(Debug, PartialEq)]
enum Command {
    Export(ExportJob),
    Batch {
        jobs: String,
        workers: usize,
    },
    Presets,
    Plugins,
    Inspect {
        project: String,
    },
    Edit {
        project: String,
        // Boxed: an edit (a whole stroke) dwarfs the other commands.
        edit: Box<AutomationEdit>,
    },
    Ocio {
        config: String,
    },
    OcioBake(OcioBake),
    Help,
}

/// An OCIO view to bake into a `.cube` display LUT.
#[derive(Debug, PartialEq)]
struct OcioBake {
    config: String,
    display: String,
    view: String,
    /// Colour space the renderer's linear Rec.709 light is read as;
    /// `None` finds it in the config.
    input: Option<String>,
    out: String,
    size: usize,
}

impl Default for OcioBake {
    fn default() -> Self {
        Self {
            config: String::new(),
            display: String::new(),
            view: String::new(),
            input: None,
            out: String::new(),
            size: OCIO_BAKE_SIZE,
        }
    }
}

/// The displays, views, and colour spaces of an OCIO config, as JSON.
fn ocio_summary(config: &Path) -> Result<String, String> {
    let config = OcioConfig::load(config).map_err(|error| error.to_string())?;
    serde_json::to_string_pretty(&config.summary()).map_err(|error| error.to_string())
}

/// Bakes an OCIO view for the viewport's display LUT and writes it.
fn ocio_bake(bake: &OcioBake) -> Result<String, String> {
    let config = OcioConfig::load(&bake.config).map_err(|error| error.to_string())?;
    let input = match &bake.input {
        Some(input) => input.clone(),
        None => config
            .linear_rec709_space()
            .ok_or("the config has no linear Rec.709 colour space; pass --input")?
            .to_owned(),
    };
    let lut = config
        .bake_view(&input, &bake.display, &bake.view, bake.size)
        .map_err(|error| error.to_string())?;
    std::fs::write(&bake.out, lut.to_cube_text()).map_err(|error| error.to_string())?;
    Ok(bake.out.clone())
}

fn parse(args: &[String]) -> Result<Command, String> {
    let mut args = args.iter();
    match args.next().map(String::as_str) {
        None | Some("help" | "--help" | "-h") => Ok(Command::Help),
        Some("presets") => Ok(Command::Presets),
        Some("plugins") => Ok(Command::Plugins),
        Some("inspect") => match (args.next(), args.next()) {
            (Some(project), None) => Ok(Command::Inspect {
                project: project.clone(),
            }),
            (None, _) => Err("inspect needs a project file".to_owned()),
            (Some(_), Some(extra)) => Err(format!("unexpected argument `{extra}`")),
        },
        Some("edit") => match (args.next(), args.next(), args.next()) {
            (Some(project), Some(json), None) => {
                let edit = serde_json::from_str(json)
                    .map_err(|error| format!("the edit is not valid JSON: {error}"))?;
                Ok(Command::Edit {
                    project: project.clone(),
                    edit: Box::new(edit),
                })
            }
            (Some(_), Some(_), Some(extra)) => Err(format!("unexpected argument `{extra}`")),
            _ => Err("edit needs a project file and an edit as JSON".to_owned()),
        },
        Some("batch") => {
            let mut jobs = None;
            let mut workers = 1;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--workers" => {
                        let text = args.next().ok_or("--workers needs a value")?;
                        workers = text
                            .parse::<usize>()
                            .ok()
                            .filter(|count| *count > 0)
                            .ok_or_else(|| {
                                format!("--workers: `{text}` is not a positive number")
                            })?;
                    }
                    flag if flag.starts_with("--") => {
                        return Err(format!("unknown option `{flag}`"));
                    }
                    path if jobs.is_none() => jobs = Some(path.to_owned()),
                    extra => return Err(format!("unexpected argument `{extra}`")),
                }
            }
            let jobs = jobs.ok_or("batch needs a job file")?;
            Ok(Command::Batch { jobs, workers })
        }
        Some("export") => {
            let mut project = None;
            let mut output = None;
            let mut preset = None;
            let mut format = None;
            let mut udim = false;
            let mut ao_samples = None;
            let mut ocio_config = None;
            let mut colorspace = None;
            while let Some(arg) = args.next() {
                let mut value = |flag: &str| {
                    args.next()
                        .cloned()
                        .ok_or_else(|| format!("{flag} needs a value"))
                };
                match arg.as_str() {
                    "--out" => output = Some(value("--out")?),
                    "--preset" => preset = Some(value("--preset")?),
                    "--format" => {
                        let name = value("--format")?;
                        format = Some(
                            TextureFormat::from_extension(&name)
                                .ok_or_else(|| format!("unknown format `{name}`"))?,
                        );
                    }
                    "--udim" => udim = true,
                    "--ocio" => ocio_config = Some(value("--ocio")?),
                    "--colorspace" => colorspace = Some(value("--colorspace")?),
                    "--ao-samples" => {
                        let text = value("--ao-samples")?;
                        ao_samples = Some(
                            text.parse::<u32>()
                                .map_err(|_| format!("--ao-samples: `{text}` is not a number"))?,
                        );
                    }
                    flag if flag.starts_with("--") => {
                        return Err(format!("unknown option `{flag}`"));
                    }
                    path if project.is_none() => project = Some(path.to_owned()),
                    extra => return Err(format!("unexpected argument `{extra}`")),
                }
            }
            let project = project.ok_or("export needs a project file")?;
            let output = output.ok_or("export needs --out <dir>")?;
            let mut job = ExportJob::new(project, output);
            job.preset = preset;
            job.format = format;
            job.udim = udim;
            if let Some(samples) = ao_samples {
                job.mesh_maps.ao_samples = samples;
            }
            job.ocio = match (ocio_config, colorspace) {
                (Some(config), Some(colorspace)) => Some(OcioExport {
                    config: config.into(),
                    colorspace,
                }),
                (None, None) => None,
                _ => return Err("--ocio and --colorspace go together".to_owned()),
            };
            Ok(Command::Export(job))
        }
        Some("ocio") => match (args.next(), args.next()) {
            (Some(config), None) => Ok(Command::Ocio {
                config: config.clone(),
            }),
            (None, _) => Err("ocio needs a config file".to_owned()),
            (Some(_), Some(extra)) => Err(format!("unexpected argument `{extra}`")),
        },
        Some("ocio-bake") => {
            let mut config = None;
            let mut bake = OcioBake::default();
            while let Some(arg) = args.next() {
                let mut value = |flag: &str| {
                    args.next()
                        .cloned()
                        .ok_or_else(|| format!("{flag} needs a value"))
                };
                match arg.as_str() {
                    "--display" => bake.display = value("--display")?,
                    "--view" => bake.view = value("--view")?,
                    "--input" => bake.input = Some(value("--input")?),
                    "--out" => bake.out = value("--out")?,
                    "--size" => {
                        let text = value("--size")?;
                        bake.size = text
                            .parse::<usize>()
                            .ok()
                            .filter(|size| (2..=MAX_CUBE_LUT_SIZE).contains(size))
                            .ok_or_else(|| {
                                format!("--size: `{text}` is not 2 to {MAX_CUBE_LUT_SIZE}")
                            })?;
                    }
                    flag if flag.starts_with("--") => {
                        return Err(format!("unknown option `{flag}`"));
                    }
                    path if config.is_none() => config = Some(path.to_owned()),
                    extra => return Err(format!("unexpected argument `{extra}`")),
                }
            }
            bake.config = config.ok_or("ocio-bake needs a config file")?;
            if bake.display.is_empty() || bake.view.is_empty() || bake.out.is_empty() {
                return Err("ocio-bake needs --display, --view, and --out".to_owned());
            }
            Ok(Command::OcioBake(bake))
        }
        Some(other) => Err(format!("unknown command `{other}`")),
    }
}

/// Splits a job-file line into arguments: whitespace separates them, double
/// quotes group, and `#` outside quotes starts a comment.
fn split_line(line: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut word: Option<String> = None;
    let mut quoted = false;
    for character in line.chars() {
        match character {
            '"' => {
                quoted = !quoted;
                word.get_or_insert_with(String::new);
            }
            '#' if !quoted => break,
            space if space.is_whitespace() && !quoted => words.extend(word.take()),
            other => word.get_or_insert_with(String::new).push(other),
        }
    }
    if quoted {
        return Err("unclosed quote".to_owned());
    }
    words.extend(word);
    Ok(words)
}

/// Reads a job file: one `export` argument list per line, with relative
/// project and output paths resolved against `base`.
fn parse_job_file(text: &str, base: &Path) -> Result<Vec<ExportJob>, String> {
    let mut jobs = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let mut words =
            split_line(line).map_err(|error| format!("line {}: {error}", number + 1))?;
        if words.is_empty() {
            continue;
        }
        words.insert(0, "export".to_owned());
        let Command::Export(mut job) =
            parse(&words).map_err(|error| format!("line {}: {error}", number + 1))?
        else {
            return Err(format!("line {}: not an export job", number + 1));
        };
        // Job files are shared between machines, so their relative paths
        // may use either separator.
        let resolve = |path: &Path| {
            substancecraft_api::resolve_stored_path(&path.to_string_lossy(), Some(base))
        };
        job.project = resolve(&job.project);
        job.output = resolve(&job.output);
        if let Some(ocio) = &mut job.ocio {
            ocio.config = resolve(&ocio.config);
        }
        jobs.push(job);
    }
    if jobs.is_empty() {
        return Err("the job file lists no jobs".to_owned());
    }
    Ok(jobs)
}

/// Runs a job file, printing one line per job and a summary. Returns the
/// exit code: 0 when every job succeeded, 1 when any failed, 2 when the job
/// file could not be read.
fn run_batch(path: &Path, workers: usize) -> u8 {
    let jobs = match std::fs::read_to_string(path)
        .map_err(|error| error.to_string())
        .and_then(|text| parse_job_file(&text, path.parent().unwrap_or_else(|| Path::new("."))))
    {
        Ok(jobs) => jobs,
        Err(error) => {
            eprintln!("error: {}: {error}", path.display());
            return 2;
        }
    };
    let results = run_export_jobs(&jobs, workers);
    let mut failed = 0;
    for (job, result) in jobs.iter().zip(&results) {
        match result {
            Ok(files) => println!("ok      {} ({} files)", job.project.display(), files.len()),
            Err(error) => {
                failed += 1;
                println!("FAILED  {}: {error}", job.project.display());
            }
        }
    }
    println!("{} of {} jobs succeeded", jobs.len() - failed, jobs.len());
    u8::from(failed > 0)
}

/// A project summary as JSON, or the error that stopped it.
fn inspect(project: &Path) -> Result<String, String> {
    let (project, _) = load_project_file_with_report(project).map_err(|error| error.to_string())?;
    serde_json::to_string_pretty(&project_summary(&project)).map_err(|error| error.to_string())
}

/// Applies `edit` to the project file and returns the outcome as JSON.
fn edit(project: &Path, edit: &AutomationEdit) -> Result<String, String> {
    let outcome = edit_project_file(project, edit).map_err(|error| error.to_string())?;
    serde_json::to_string_pretty(&outcome).map_err(|error| error.to_string())
}

/// Prints a command's JSON result, or its error with exit status 1.
fn report(result: Result<String, String>) -> ExitCode {
    match result {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

/// Leading `--plugins <folder>` options, which every command shares, and
/// the arguments after them.
fn plugin_options(args: &[String]) -> Result<(Vec<PathBuf>, &[String]), String> {
    let mut folders = Vec::new();
    let mut rest = args;
    while let [flag, tail @ ..] = rest
        && flag == "--plugins"
    {
        let [folder, tail @ ..] = tail else {
            return Err("--plugins needs a folder".to_owned());
        };
        folders.push(PathBuf::from(folder));
        rest = tail;
    }
    Ok((folders, rest))
}

/// Loads plugins from `folders` and the default ones, and installs them.
fn load_all_plugins(mut folders: Vec<PathBuf>) -> Vec<PluginError> {
    folders.extend(default_plugin_folders());
    let (plugins, errors) = load_plugins(&folders);
    install_plugins(plugins);
    errors
}

/// The installed plugins and the load errors, as JSON.
fn plugins_summary(errors: &[PluginError]) -> Result<String, String> {
    let plugins: Vec<serde_json::Value> = installed_plugins()
        .iter()
        .map(|plugin| {
            let manifest = plugin.manifest();
            let ids = |items: &[substancecraft_api::PluginItem]| {
                items
                    .iter()
                    .map(|item| format!("{}/{}", manifest.id, item.id))
                    .collect::<Vec<_>>()
            };
            serde_json::json!({
                "id": manifest.id,
                "name": manifest.name,
                "version": manifest.version,
                "folder": plugin.folder(),
                "generators": ids(&manifest.generators),
                "filters": ids(&manifest.filters),
                "export_presets": manifest
                    .export_presets
                    .iter()
                    .map(|preset| preset.name.clone())
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let errors: Vec<String> = errors.iter().map(ToString::to_string).collect();
    serde_json::to_string_pretty(&serde_json::json!({ "plugins": plugins, "errors": errors }))
        .map_err(|error| error.to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (folders, args) = match plugin_options(&args) {
        Ok(split) => split,
        Err(error) => {
            eprintln!("error: {error}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let plugin_errors = load_all_plugins(folders);
    let command = parse(args);
    if !matches!(command, Ok(Command::Plugins)) {
        for error in &plugin_errors {
            eprintln!("warning: plugin not loaded: {error}");
        }
    }
    match command {
        Ok(Command::Plugins) => report(plugins_summary(&plugin_errors)),
        Ok(Command::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Inspect { project }) => report(inspect(Path::new(&project))),
        Ok(Command::Ocio { config }) => report(ocio_summary(Path::new(&config))),
        Ok(Command::OcioBake(bake)) => report(ocio_bake(&bake)),
        Ok(Command::Edit {
            project,
            edit: change,
        }) => report(edit(Path::new(&project), &change)),
        Ok(Command::Presets) => {
            for preset in list_export_presets() {
                println!("{}", preset.name);
            }
            ExitCode::SUCCESS
        }
        Ok(Command::Batch { jobs, workers }) => {
            ExitCode::from(run_batch(Path::new(&jobs), workers))
        }
        Ok(Command::Export(job)) => match run_export_job(&job) {
            Ok(files) => {
                for file in files {
                    println!("{}", file.display());
                }
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::from(1)
            }
        },
        Err(error) => {
            eprintln!("error: {error}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn args(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn plugin_folders_come_before_the_command() {
        let line = args("--plugins a --plugins b export p.sproject --out o");
        let (folders, rest) = plugin_options(&line).unwrap();
        assert_eq!(folders, [PathBuf::from("a"), PathBuf::from("b")]);
        assert!(matches!(parse(rest), Ok(Command::Export(_))));
        assert_eq!(parse(&args("plugins")), Ok(Command::Plugins));
        assert!(plugin_options(&args("--plugins")).is_err());
        let plain = args("presets");
        let (folders, rest) = plugin_options(&plain).unwrap();
        assert!(folders.is_empty());
        assert_eq!(rest, plain);
    }

    #[test]
    fn tc_0283_cli_inspects_and_edits_projects_with_json_edits() {
        // Parsing: one project for inspect; a project and a JSON edit.
        assert_eq!(
            parse(&args("inspect a.sproject")),
            Ok(Command::Inspect {
                project: "a.sproject".to_owned()
            })
        );
        assert!(parse(&args("inspect")).is_err());
        assert!(parse(&args("inspect a b")).is_err());
        let set_layer = r#"{"edit":"set_layer","texture_set":"Body","layer":"Base","opacity":0.5}"#;
        let parsed = parse(&[
            "edit".to_owned(),
            "a.sproject".to_owned(),
            set_layer.to_owned(),
        ]);
        assert!(matches!(
            parsed,
            Ok(Command::Edit { ref edit, .. }) if matches!(
                **edit,
                AutomationEdit::SetLayer {
                    opacity: Some(0.5),
                    ..
                }
            )
        ));
        let bad = parse(&args("edit a.sproject {not-json}")).unwrap_err();
        assert!(bad.starts_with("the edit is not valid JSON"), "{bad}");
        assert!(parse(&args("edit a.sproject")).is_err());

        // Running: a project on disk is summarized and edited in place.
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("cli.sproject");
        let project = substancecraft_core::MaterialProject {
            texture_sets: vec![substancecraft_core::TextureSet {
                user_channels: Default::default(),
                stencils: Default::default(),
                id: substancecraft_core::TextureSetId(1),
                name: "Body".to_owned(),
                material_slot: "Body".to_owned(),
                resolution: substancecraft_core::TextureResolution::new(8, 8).unwrap(),
                channels: [substancecraft_core::MaterialChannel::BaseColor].into(),
                layers: vec![substancecraft_core::MaterialLayer::new_paint(
                    substancecraft_core::LayerId(1),
                    "Base",
                )],
                height_strength: 1.0,
                export: None,
            }],
            ..substancecraft_core::MaterialProject::empty()
        };
        substancecraft_api::save_project_file(&path, &project).unwrap();
        let summary = inspect(&path).unwrap();
        assert!(summary.contains("\"name\": \"Base\""));
        let change: AutomationEdit = serde_json::from_str(set_layer).unwrap();
        let outcome = edit(&path, &change).unwrap();
        assert!(outcome.contains("\"changed\": true"), "{outcome}");
        let reloaded = substancecraft_api::load_project_file(&path).unwrap();
        assert_eq!(reloaded.texture_sets[0].layers[0].opacity, 0.5);
        assert!(edit(&path, &change).unwrap().contains("\"changed\": false"));
        assert!(inspect(&folder.path().join("missing.sproject")).is_err());
    }

    #[test]
    fn tc_0201_cli_parses_export_jobs_and_rejects_bad_arguments() {
        let job = export_job(
            parse(&args(
            "export scene.sproject --out build --preset Unreal_Engine --format TGA --udim --ao-samples 8",
            ))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(job.project, std::path::PathBuf::from("scene.sproject"));
        assert_eq!(job.output, std::path::PathBuf::from("build"));
        assert_eq!(job.preset.as_deref(), Some("Unreal_Engine"));
        assert_eq!(job.format, Some(TextureFormat::Tga));
        assert!(job.udim);
        assert_eq!(job.mesh_maps.ao_samples, 8);

        assert_eq!(parse(&args("presets")), Ok(Command::Presets));
        assert_eq!(parse(&[]), Ok(Command::Help));
        for bad in [
            "export scene.sproject",
            "export --out build",
            "export a b --out c",
            "export a --out b --format bmp",
            "export a --out b --ao-samples many",
            "export a --out b --nope",
            "render a",
        ] {
            assert!(parse(&args(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn tc_0233_batch_job_files_parse_run_every_job_and_report_failures() {
        assert_eq!(
            parse(&args("batch jobs.txt --workers 4")),
            Ok(Command::Batch {
                jobs: "jobs.txt".to_owned(),
                workers: 4
            })
        );
        for bad in [
            "batch",
            "batch a b",
            "batch a --workers 0",
            "batch a --nope",
        ] {
            assert!(parse(&args(bad)).is_err(), "{bad}");
        }

        let base = Path::new("root");
        let jobs = parse_job_file(
            "# nightly\n\"my scene.sproject\" --out \"out dir\" --format tga\n\n/abs/b.sproject --out b # trailing\n",
            base,
        )
        .unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].project, base.join("my scene.sproject"));
        assert_eq!(jobs[0].output, base.join("out dir"));
        assert_eq!(jobs[0].format, Some(TextureFormat::Tga));
        assert_eq!(jobs[1].output, base.join("b"));
        assert!(
            parse_job_file("a.sproject\n", base)
                .unwrap_err()
                .starts_with("line 1")
        );
        assert!(parse_job_file("\"a --out b\n", base).is_err());
        assert!(parse_job_file("# nothing\n", base).is_err());

        let folder = tempfile::tempdir().unwrap();
        substancecraft_api::save_project_file(
            folder.path().join("empty.sproject"),
            &substancecraft_core::MaterialProject::empty(),
        )
        .unwrap();
        let file = folder.path().join("jobs.txt");
        std::fs::write(&file, "empty.sproject --out a\nmissing.sproject --out b\n").unwrap();
        assert_eq!(run_batch(&file, 2), 1);
        assert!(folder.path().join("a").is_dir());
        std::fs::write(&file, "empty.sproject --out a\nempty.sproject --out c\n").unwrap();
        assert_eq!(run_batch(&file, 2), 0);
        assert_eq!(run_batch(&folder.path().join("none.txt"), 1), 2);
    }

    #[test]
    fn tc_0392_cli_lists_and_bakes_ocio_views_and_exports_into_config_colour_spaces() {
        let folder = tempfile::tempdir().unwrap();
        let config = folder.path().join("config.ocio");
        let text = [
            "ocio_profile_version: 2",
            "roles:",
            "  scene_linear: Linear Rec.709",
            "displays:",
            "  sRGB:",
            "    - !<View> {name: Standard, colorspace: sRGB}",
            "colorspaces:",
            "  - !<ColorSpace>",
            "    name: Linear Rec.709",
            "  - !<ColorSpace>",
            "    name: sRGB",
            "    from_scene_reference: !<ExponentWithLinearTransform> {gamma: 2.4, offset: 0.055, direction: inverse}",
            "  - !<ColorSpace>",
            "    name: ACEScg",
            "    from_scene_reference: !<BuiltinTransform> {style: ACES-AP1_to_LINEAR-REC709_BFD, direction: inverse}",
        ]
        .join("\n");
        std::fs::write(&config, text).unwrap();
        let summary: serde_json::Value =
            serde_json::from_str(&ocio_summary(&config).unwrap()).unwrap();
        assert_eq!(summary["displays"][0]["views"][0], "Standard");
        assert_eq!(summary["linear_rec709"], "Linear Rec.709");

        let out = folder.path().join("view.cube");
        let command = parse(&args(&format!(
            "ocio-bake {} --display sRGB --view Standard --size 9 --out {}",
            config.display(),
            out.display()
        )))
        .unwrap();
        let Command::OcioBake(bake) = command else {
            panic!("expected ocio-bake");
        };
        ocio_bake(&bake).unwrap();
        let lut =
            substancecraft_core::CubeLut::parse(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(lut.size, 9);
        for bad in [
            "ocio",
            "ocio a b",
            "ocio-bake a --display x --view y",
            "ocio-bake a --display x --view y --out z --size 1",
            "export a --out b --ocio c",
        ] {
            assert!(parse(&args(bad)).is_err(), "{bad}");
        }

        // Exports convert colour outputs into the config's ACEScg: pure
        // sRGB red becomes mostly-red AP1 weights, and data stays raw.
        let mut project = substancecraft_core::MaterialProject::empty();
        let mut set = substancecraft_core::TextureSet {
            user_channels: Default::default(),
            stencils: Default::default(),
            id: substancecraft_core::TextureSetId(1),
            name: "Body".to_owned(),
            material_slot: "Body".to_owned(),
            resolution: substancecraft_core::TextureResolution::new(4, 4).unwrap(),
            channels: [
                substancecraft_core::MaterialChannel::BaseColor,
                substancecraft_core::MaterialChannel::Roughness,
            ]
            .into(),
            layers: Vec::new(),
            height_strength: 1.0,
            export: None,
        };
        let fill = substancecraft_core::LayerFill {
            color: [255, 0, 0, 255],
            roughness: 0.5,
            ..Default::default()
        };
        set.layers = vec![substancecraft_core::MaterialLayer {
            channels: [
                substancecraft_core::MaterialChannel::BaseColor,
                substancecraft_core::MaterialChannel::Roughness,
            ]
            .into(),
            ..substancecraft_core::MaterialLayer::new_fill(
                substancecraft_core::LayerId(1),
                "Red",
                fill,
            )
        }];
        project.texture_sets.push(set);
        let scene = folder.path().join("scene.sproject");
        substancecraft_api::save_project_file(&scene, &project).unwrap();
        let command = parse(&args(&format!(
            "export {} --out {} --ocio {} --colorspace ACEScg",
            scene.display(),
            folder.path().join("out").display(),
            config.display()
        )))
        .unwrap();
        let job = export_job(command).unwrap();
        let files = run_export_job(&job).unwrap();
        let base = files
            .iter()
            .find(|file| file.to_string_lossy().contains("BaseColor"))
            .unwrap();
        let image = substancecraft_api::load_texture_image(base).unwrap();
        let [r, g, b, _] = image.rgba8[..4] else {
            panic!("no texels");
        };
        assert!((i32::from(r) - 156).abs() <= 2, "{r}");
        assert!(g > 10 && g < 25 && b < 10, "{g} {b}");
        let mut bad = job.clone();
        bad.ocio = Some(OcioExport {
            config: config.clone(),
            colorspace: "Nope".to_owned(),
        });
        assert!(run_export_job(&bad).is_err());
    }

    fn export_job(command: Command) -> Option<ExportJob> {
        match command {
            Command::Export(job) => Some(job),
            _ => None,
        }
    }
}
