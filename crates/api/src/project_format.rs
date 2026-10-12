//! On-disk project format: diff-friendly JSON and schema migration.
//!
//! Saves are written with sorted keys, two-space indentation, one array
//! element per line (arrays of plain numbers or strings stay on one line),
//! and a trailing newline, so the same project always produces the same
//! bytes and a one-field edit changes one line.
//!
//! Loads upgrade older schemas one version at a time and report what they
//! changed, which fields were missing and filled with defaults, and which
//! fields the current schema does not know (and so could not keep).

use serde_json::{Map, Value};
use substancecraft_core::{MaterialProject, PROJECT_SCHEMA_VERSION};

use crate::{MAX_PROJECT_JSON_BYTES, ProjectFileError};

/// What a load had to do to bring a project file up to date.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MigrationReport {
    /// The schema version the file was written with.
    pub from_version: u32,
    /// One note per upgrade step that ran (empty for current files).
    pub migrated: Vec<String>,
    /// Paths of fields the file lacked, filled with their defaults.
    pub defaulted: Vec<String>,
    /// Paths of fields the current schema does not know. Their data is not
    /// kept; saving the project drops them.
    pub unknown: Vec<String>,
}

impl MigrationReport {
    /// True when the file was current and nothing was defaulted or lost.
    pub fn is_clean(&self) -> bool {
        self.migrated.is_empty() && self.defaulted.is_empty() && self.unknown.is_empty()
    }

    /// A one-line summary for a status bar, or `None` when clean.
    pub fn summary(&self) -> Option<String> {
        if self.is_clean() {
            return None;
        }
        let mut parts = Vec::new();
        if !self.migrated.is_empty() {
            parts.push(format!(
                "upgraded from schema {} to {PROJECT_SCHEMA_VERSION}",
                self.from_version
            ));
        }
        if !self.defaulted.is_empty() {
            parts.push(format!("{} field(s) defaulted", self.defaulted.len()));
        }
        if !self.unknown.is_empty() {
            parts.push(format!(
                "{} unknown field(s) not kept: {}",
                self.unknown.len(),
                self.unknown
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        Some(parts.join("; "))
    }
}

/// One upgrade step: rewrites a project object written with `from` into
/// the shape of `from + 1`, describing what it did.
type Migration = fn(&mut Map<String, Value>) -> String;

/// `MIGRATIONS[i]` upgrades schema `i + 1` to `i + 2`.
const MIGRATIONS: &[Migration] = &[migrate_1_to_2, migrate_2_to_3];

/// Schema 2 adds optional project-level data (named cameras) and the
/// stable text layout. Version 1 files need no rewriting: the new fields
/// default when absent. The bump makes older builds refuse schema 2 files
/// instead of silently dropping the fields they do not know.
fn migrate_1_to_2(_project: &mut Map<String, Value>) -> String {
    "schema 1 → 2: no field changes".to_owned()
}

/// Schema 3 gives project materials stable ids, so fill layers can link to
/// them as instances. Materials are numbered from 1 in file order.
fn migrate_2_to_3(project: &mut Map<String, Value>) -> String {
    let mut count = 0_u64;
    if let Some(Value::Array(materials)) = project.get_mut("materials") {
        for material in materials.iter_mut().filter_map(Value::as_object_mut) {
            count += 1;
            material.insert("id".to_owned(), Value::from(count));
        }
    }
    format!("schema 2 → 3: numbered {count} material(s)")
}

/// Decodes project JSON of any supported schema, upgrading it in memory.
pub fn decode_project_json_with_report(
    bytes: &[u8],
) -> Result<(MaterialProject, MigrationReport), ProjectFileError> {
    if bytes.len() > MAX_PROJECT_JSON_BYTES {
        return Err(ProjectFileError::TooLarge(bytes.len()));
    }
    let mut value: Value = serde_json::from_slice(bytes)
        .map_err(|error| ProjectFileError::Decode(error.to_string()))?;
    let Value::Object(object) = &mut value else {
        return Err(ProjectFileError::Decode(
            "project JSON is not an object".to_owned(),
        ));
    };
    let from_version = object
        .get("schema_version")
        .and_then(Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| ProjectFileError::Decode("missing or invalid schema_version".to_owned()))?;
    if from_version == 0 || from_version > PROJECT_SCHEMA_VERSION {
        return Err(ProjectFileError::InvalidProject(format!(
            "project schema version {from_version} is not supported"
        )));
    }

    let mut report = MigrationReport {
        from_version,
        ..MigrationReport::default()
    };
    for version in from_version..PROJECT_SCHEMA_VERSION {
        let Some(migration) = MIGRATIONS.get(version as usize - 1) else {
            return Err(ProjectFileError::InvalidProject(format!(
                "no upgrade from project schema version {version}"
            )));
        };
        report.migrated.push(migration(object));
        object.insert(
            "schema_version".to_owned(),
            Value::from(u64::from(version) + 1),
        );
    }

    let project: MaterialProject = serde_json::from_value(value.clone())
        .map_err(|error| ProjectFileError::Decode(error.to_string()))?;
    project
        .validate()
        .map_err(|error| ProjectFileError::InvalidProject(error.to_string()))?;

    let round_trip = serde_json::to_value(&project)
        .map_err(|error| ProjectFileError::Encode(error.to_string()))?;
    compare_fields(&value, &round_trip, String::new(), &mut report);
    Ok((project, report))
}

/// Records fields present on one side only. A field the writer leaves out
/// when it holds its default (empty lists, `null`, Normal blending) is not
/// reported as unknown.
fn compare_fields(input: &Value, output: &Value, path: String, report: &mut MigrationReport) {
    match (input, output) {
        (Value::Object(input), Value::Object(output)) => {
            for (key, value) in input {
                let child = join_path(&path, key);
                match output.get(key) {
                    Some(kept) => compare_fields(value, kept, child, report),
                    None if is_default_like(key, value) => {}
                    None => report.unknown.push(child),
                }
            }
            for key in output.keys() {
                if !input.contains_key(key) {
                    report.defaulted.push(join_path(&path, key));
                }
            }
        }
        (Value::Array(input), Value::Array(output)) => {
            for (index, (value, kept)) in input.iter().zip(output).enumerate() {
                compare_fields(value, kept, format!("{path}[{index}]"), report);
            }
        }
        _ => {}
    }
}

fn join_path(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_owned()
    } else {
        format!("{path}.{key}")
    }
}

fn is_default_like(key: &str, value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => true,
        Value::String(text) => text.is_empty() || (key == "blend_mode" && text == "Normal"),
        Value::Array(items) => items.is_empty(),
        Value::Object(fields) => fields
            .iter()
            .all(|(key, value)| is_default_like(key, value)),
        _ => false,
    }
}

/// Re-lays out compact JSON in the stable, diff-friendly format.
pub(crate) fn stable_json(compact: &[u8]) -> Result<Vec<u8>, ProjectFileError> {
    // `Value` objects are `BTreeMap`s, so keys come out sorted.
    let value: Value = serde_json::from_slice(compact)
        .map_err(|error| ProjectFileError::Encode(error.to_string()))?;
    let mut out = Vec::with_capacity(compact.len() * 2);
    write_value(&value, 0, &mut out)?;
    out.push(b'\n');
    if out.len() > MAX_PROJECT_JSON_BYTES {
        return Err(ProjectFileError::TooLarge(MAX_PROJECT_JSON_BYTES));
    }
    Ok(out)
}

fn write_value(value: &Value, depth: usize, out: &mut Vec<u8>) -> Result<(), ProjectFileError> {
    match value {
        Value::Object(fields) if !fields.is_empty() => {
            out.extend_from_slice(b"{\n");
            for (index, (key, field)) in fields.iter().enumerate() {
                indent(depth + 1, out);
                write_scalar(&Value::String(key.clone()), out)?;
                out.extend_from_slice(b": ");
                write_value(field, depth + 1, out)?;
                if index + 1 < fields.len() {
                    out.push(b',');
                }
                out.push(b'\n');
            }
            indent(depth, out);
            out.push(b'}');
        }
        Value::Array(items) if !items.is_empty() && !items.iter().all(is_scalar) => {
            out.extend_from_slice(b"[\n");
            for (index, item) in items.iter().enumerate() {
                indent(depth + 1, out);
                write_value(item, depth + 1, out)?;
                if index + 1 < items.len() {
                    out.push(b',');
                }
                out.push(b'\n');
            }
            indent(depth, out);
            out.push(b']');
        }
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.extend_from_slice(b", ");
                }
                write_scalar(item, out)?;
            }
            out.push(b']');
        }
        other => write_scalar(other, out)?,
    }
    if out.len() > MAX_PROJECT_JSON_BYTES {
        return Err(ProjectFileError::TooLarge(MAX_PROJECT_JSON_BYTES));
    }
    Ok(())
}

fn is_scalar(value: &Value) -> bool {
    !matches!(value, Value::Array(_) | Value::Object(_))
        || matches!(value, Value::Array(items) if items.is_empty())
        || matches!(value, Value::Object(fields) if fields.is_empty())
}

fn write_scalar(value: &Value, out: &mut Vec<u8>) -> Result<(), ProjectFileError> {
    serde_json::to_writer(out, value).map_err(|error| ProjectFileError::Encode(error.to_string()))
}

fn indent(depth: usize, out: &mut Vec<u8>) {
    out.extend(std::iter::repeat_n(b' ', depth * 2));
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{decode_project_json, encode_project_json};
    use substancecraft_core::{
        LayerId, MaterialChannel, MaterialLayer, TextureResolution, TextureSet, TextureSetId,
    };

    fn sample_project() -> MaterialProject {
        let mut paint = MaterialLayer::new_paint(LayerId(2), "Paint");
        paint
            .paint_strokes
            .push(substancecraft_core::PaintStroke::new(
                vec![glam::Vec2::new(0.25, 0.5), glam::Vec2::new(0.75, 0.5)],
                4.0,
                [200, 10, 10, 255],
            ));
        MaterialProject {
            texture_sets: vec![TextureSet {
                id: TextureSetId(1),
                name: "Body".to_owned(),
                material_slot: "Body".to_owned(),
                resolution: TextureResolution::new(64, 64).unwrap(),
                channels: [MaterialChannel::BaseColor, MaterialChannel::Roughness].into(),
                layers: vec![paint, MaterialLayer::new_paint(LayerId(3), "Top")],
                height_strength: 1.0,
                export: None,
            }],
            active_texture_set: Some(TextureSetId(1)),
            ..MaterialProject::empty()
        }
    }

    #[test]
    fn tc_0230_saves_are_stable_sorted_and_change_one_line_per_edit() {
        let project = sample_project();
        let first = encode_project_json(&project).unwrap();
        let reloaded = decode_project_json(&first).unwrap();
        assert_eq!(reloaded, project);
        // Same project, same bytes, however many times it round-trips.
        assert_eq!(encode_project_json(&reloaded).unwrap(), first);

        let text = String::from_utf8(first.clone()).unwrap();
        assert!(text.ends_with("}\n"));
        assert!(text.contains("\n      \"channels\": [\"BaseColor\", \"Roughness\"],\n"));
        // Keys are sorted at every level.
        let top: Vec<&str> = text
            .lines()
            .filter(|line| line.starts_with("  \""))
            .collect();
        let mut sorted = top.clone();
        sorted.sort();
        assert_eq!(top, sorted);

        // Renaming one layer changes exactly one line.
        let mut renamed = project.clone();
        renamed.texture_sets[0].layers[1].name = "Top coat".to_owned();
        let edited = String::from_utf8(encode_project_json(&renamed).unwrap()).unwrap();
        let before: Vec<&str> = text.lines().collect();
        let after: Vec<&str> = edited.lines().collect();
        assert_eq!(before.len(), after.len());
        let changed = before.iter().zip(&after).filter(|(a, b)| a != b).count();
        assert_eq!(changed, 1);
    }

    #[test]
    fn tc_0231_older_schemas_upgrade_and_report_defaulted_and_unknown_fields() {
        let mut value = serde_json::to_value(sample_project()).unwrap();
        let object = value.as_object_mut().unwrap();
        object.insert("schema_version".to_owned(), Value::from(1));
        object.remove("saved_poses");
        object.insert("plugin_data".to_owned(), serde_json::json!({ "x": 1 }));
        let layer = &mut object["texture_sets"][0]["layers"][1];
        layer
            .as_object_mut()
            .unwrap()
            .insert("glow".to_owned(), Value::from(0.5));
        // Written-out defaults are not mistaken for unknown data.
        layer
            .as_object_mut()
            .unwrap()
            .insert("blend_mode".to_owned(), Value::from("Normal"));
        let bytes = serde_json::to_vec(&value).unwrap();

        let (project, report) = decode_project_json_with_report(&bytes).unwrap();
        assert_eq!(project.schema_version, PROJECT_SCHEMA_VERSION);
        assert_eq!(project.texture_sets[0].layers[1].name, "Top");
        assert_eq!(report.from_version, 1);
        assert_eq!(report.migrated.len(), (PROJECT_SCHEMA_VERSION - 1) as usize);
        assert_eq!(report.defaulted, vec!["saved_poses".to_owned()]);
        assert_eq!(
            report.unknown,
            vec![
                "plugin_data".to_owned(),
                "texture_sets[0].layers[1].glow".to_owned()
            ]
        );
        assert!(report.summary().unwrap().contains("plugin_data"));

        // A current file reports nothing.
        let current = encode_project_json(&project).unwrap();
        let (_, clean) = decode_project_json_with_report(&current).unwrap();
        assert!(clean.is_clean());
        assert_eq!(clean.summary(), None);

        // Future, zero, and missing versions are refused.
        for version in [
            Value::from(PROJECT_SCHEMA_VERSION + 1),
            Value::from(0),
            Value::Null,
        ] {
            object_with_version(&mut value, version);
            assert!(decode_project_json_with_report(&serde_json::to_vec(&value).unwrap()).is_err());
        }
    }

    #[test]
    fn tc_0270_schema_2_materials_get_ids_and_link_after_upgrade() {
        let mut project = sample_project();
        project.materials = substancecraft_core::MaterialResource::built_in()
            .into_iter()
            .take(2)
            .collect();
        let mut value = serde_json::to_value(&project).unwrap();
        let object = value.as_object_mut().unwrap();
        object.insert("schema_version".to_owned(), Value::from(2));
        for material in object["materials"].as_array_mut().unwrap() {
            material.as_object_mut().unwrap().remove("id");
        }
        let bytes = serde_json::to_vec(&value).unwrap();

        let (mut upgraded, report) = decode_project_json_with_report(&bytes).unwrap();
        let ids: Vec<u64> = upgraded.materials.iter().map(|m| m.id.0).collect();
        assert_eq!(ids, [1, 2]);
        assert_eq!(report.from_version, 2);
        assert_eq!(
            report.migrated,
            ["schema 2 → 3: numbered 2 material(s)".to_owned()]
        );
        assert!(report.defaulted.is_empty() && report.unknown.is_empty());

        // The upgraded project links an instance and round-trips it.
        let link = upgraded
            .link_material_command(
                TextureSetId(1),
                LayerId(3),
                substancecraft_core::MaterialId(2),
            )
            .unwrap();
        let mut history = substancecraft_core::EditHistory::default();
        assert!(history.apply(&mut upgraded, link));
        let saved = encode_project_json(&upgraded).unwrap();
        let text = String::from_utf8(saved.clone()).unwrap();
        assert!(text.contains(
            "\"material_link\": {
"
        ));
        assert_eq!(decode_project_json(&saved).unwrap(), upgraded);
    }

    fn object_with_version(value: &mut Value, version: Value) {
        value
            .as_object_mut()
            .unwrap()
            .insert("schema_version".to_owned(), version);
    }
}
