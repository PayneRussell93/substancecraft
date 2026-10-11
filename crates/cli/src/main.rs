#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented, clippy::unreachable)]

//! `substancecraft-cli`: headless jobs for pipelines.
//!
//! ```text
//! substancecraft-cli export <project.sproject> --out <dir>
//!     [--preset <name>] [--format png|tga] [--udim] [--ao-samples <n>]
//! substancecraft-cli presets
//! ```
//!
//! Exit status is 0 on success, 1 when the job fails, and 2 for bad
//! arguments. The same project and options always write the same bytes.

use std::process::ExitCode;

use substancecraft_api::{ExportJob, TextureFormat, list_export_presets, run_export_job};

const USAGE: &str = "usage:\n  substancecraft-cli export <project.sproject> --out <dir> [--preset <name>] [--format png|tga] [--udim] [--ao-samples <n>]\n  substancecraft-cli presets";

/// What the command line asks for.
#[derive(Debug, PartialEq)]
enum Command {
    Export(ExportJob),
    Presets,
    Help,
}

fn parse(args: &[String]) -> Result<Command, String> {
    let mut args = args.iter();
    match args.next().map(String::as_str) {
        None | Some("help" | "--help" | "-h") => Ok(Command::Help),
        Some("presets") => Ok(Command::Presets),
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse(&args) {
        Ok(Command::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Presets) => {
            for preset in list_export_presets() {
                println!("{}", preset.name);
            }
            ExitCode::SUCCESS
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
            "export a --out b --format exr",
            "export a --out b --ao-samples many",
            "export a --out b --nope",
            "render a",
        ] {
            assert!(parse(&args(bad)).is_err(), "{bad}");
        }
    }

    fn export_job(command: Command) -> Option<ExportJob> {
        match command {
            Command::Export(job) => Some(job),
            _ => None,
        }
    }
}
