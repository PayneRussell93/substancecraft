# Release compatibility targets

These are the promises each SubstanceCraft release makes to the pipelines and projects built on earlier ones. Every target is enforced by an automated test; a release that would break one must change this page and say so in its notes.

| Target | Promise | Enforced by |
| --- | --- | --- |
| Project files | A release opens every `.sproject` schema it has ever written (1 to the current, now 3), upgrading in memory and reporting what changed. Saving writes the current schema, which older releases refuse with the version named, never silently dropping data. | Frozen projects for each schema in `crates/api/tests/fixtures/`, opened and baked by TC-0394; migration steps in `crates/api/src/project_format.rs` (TC-0231, TC-0270). |
| Command line | `substancecraft-cli` keeps its commands (`export`, `batch`, `presets`, `inspect`, `edit`, `ocio`, `ocio-bake`), their flags, and exit codes 0 (success), 1 (job failure), and 2 (bad arguments). New flags may be added; existing ones keep their meaning. Exports are byte-identical for the same inputs within a release. | CLI parsing and job tests (TC-0201, TC-0233, TC-0283, TC-0392); the determinism test (49). |
| Export file names | Built-in export presets keep their names and output file names (for example `Body_BaseColor.png`, `Body_ORM.png`, `Body_Normal.png` for Metal/Rough (ORM)). New presets may be added. | TC-0395. |
| Automation API | API version 1: the project summary keeps its fields and every edit keeps its name and arguments. Changes that break this raise `AUTOMATION_API_VERSION`. | TC-0395 and the automation tests (48). |
| Shared library | Asset files of format 1 (`.scasset`) stay readable; newer formats are reported, not misread. | TC-0386 and the frozen asset in TC-0395. |
| Colour | OpenColorIO configs are read with the transforms listed in `crates/api/src/ocio.rs`. A transform that is not supported is named in the error, never approximated silently. | TC-0390 to TC-0393. |

Not covered: the in-memory Rust API of `substancecraft-core` (it changes with features), the preferences file (unknown fields are ignored and missing ones default), and byte-identical output across operating systems or CPUs (PROD-08).
