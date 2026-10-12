#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

//! `substancecraft-cli`: headless jobs for pipelines.
//!
//! ```text
//! substancecraft-cli export <project.sproject> --out <dir>
//!     [--preset <name>] [--format png|tga|exr] [--udim] [--ao-samples <n>]
//! substancecraft-cli batch <jobs.txt> [--workers <n>]
//! substancecraft-cli presets
//! substancecraft-cli inspect <project.sproject>
//! substancecraft-cli edit <project.sproject> '<edit as JSON>'
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
//! Exit status is 0 on success, 1 when the job fails, and 2 for bad
//! arguments. The same project and options always write the same bytes.

use std::path::Path;
use std::process::ExitCode;

use substancecraft_api::{
    AutomationEdit, ExportJob, TextureFormat, edit_project_file, list_export_presets,
    load_project_file_with_report, project_summary, run_export_job, run_export_jobs,
};

const USAGE: &str = "usage:\n  substancecraft-cli export <project.sproject> --out <dir> [--preset <name>] [--format png|tga|exr] [--udim] [--ao-samples <n>]\n  substancecraft-cli batch <jobs.txt> [--workers <n>]\n  substancecraft-cli presets\n  substancecraft-cli inspect <project.sproject>\n  substancecraft-cli edit <project.sproject> '<edit as JSON>'";

/// What the command line asks for.
#[derive(Debug, PartialEq)]
enum Command {
    Export(ExportJob),
    Batch {
        jobs: String,
        workers: usize,
    },
    Presets,
    Inspect {
        project: String,
    },
    Edit {
        project: String,
        edit: AutomationEdit,
    },
    Help,
}

fn parse(args: &[String]) -> Result<Command, String> {
    let mut args = args.iter();
    match args.next().map(String::as_str) {
        None | Some("help" | "--help" | "-h") => Ok(Command::Help),
        Some("presets") => Ok(Command::Presets),
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
                    edit,
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
            Ok(Command::Export(job))
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
        if job.project.is_relative() {
            job.project = base.join(&job.project);
        }
        if job.output.is_relative() {
            job.output = base.join(&job.output);
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse(&args) {
        Ok(Command::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Inspect { project }) => report(inspect(Path::new(&project))),
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn args(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
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
            Ok(Command::Edit {
                edit: AutomationEdit::SetLayer {
                    opacity: Some(0.5),
                    ..
                },
                ..
            })
        ));
        let bad = parse(&args("edit a.sproject {not-json}")).unwrap_err();
        assert!(bad.starts_with("the edit is not valid JSON"), "{bad}");
        assert!(parse(&args("edit a.sproject")).is_err());

        // Running: a project on disk is summarized and edited in place.
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("cli.sproject");
        let project = substancecraft_core::MaterialProject {
            texture_sets: vec![substancecraft_core::TextureSet {
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

    fn export_job(command: Command) -> Option<ExportJob> {
        match command {
            Command::Export(job) => Some(job),
            _ => None,
        }
    }
}
