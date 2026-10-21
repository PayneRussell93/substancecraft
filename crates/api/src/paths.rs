//! Paths stored in projects, shared libraries, and job files (PROD-08).
//!
//! A project or job file written on Windows may separate a relative path's
//! parts with `\`, which Linux and macOS read as part of one file name, so
//! the same project would find its bitmaps on one machine and not another.
//! Stored relative paths are read with either separator on every OS.
//! Absolute paths are left as they are: one from another OS cannot resolve
//! anyway, and the relink prompt (PROD-02) handles it.

use std::path::{Path, PathBuf};

/// `stored` as a path on this OS. Relative paths may use `/` or `\`.
///
/// On Linux and macOS a `\` inside a relative stored path is always read as
/// a separator, so a file whose name contains one cannot be referenced
/// relatively.
pub fn stored_path(stored: &str) -> PathBuf {
    let path = Path::new(stored);
    if path.has_root() || looks_like_drive_path(stored) {
        return path.to_owned();
    }
    stored
        .split(['/', '\\'])
        .filter(|part| !part.is_empty() && *part != ".")
        .collect()
}

/// [`stored_path`], joined to `base` when relative.
pub fn resolve_stored_path(stored: &str, base: Option<&Path>) -> PathBuf {
    let path = stored_path(stored);
    match base {
        Some(base) if path.is_relative() => base.join(path),
        _ => path,
    }
}

/// `C:\…` or `C:/…`, which Linux and macOS would otherwise read as
/// relative.
fn looks_like_drive_path(stored: &str) -> bool {
    let bytes = stored.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'/' | b'\\')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_read_with_either_separator() {
        let expected: PathBuf = ["textures", "wood", "albedo.png"].iter().collect();
        assert_eq!(stored_path("textures\\wood\\albedo.png"), expected);
        assert_eq!(stored_path("textures/wood/albedo.png"), expected);
        assert_eq!(stored_path("./textures//wood\\albedo.png"), expected);
        let up: PathBuf = ["..", "shared", "grime.png"].iter().collect();
        assert_eq!(stored_path("..\\shared\\grime.png"), up);

        let base = Path::new("project");
        assert_eq!(
            resolve_stored_path("textures\\wood\\albedo.png", Some(base)),
            base.join(&expected)
        );
        assert_eq!(
            resolve_stored_path("textures/wood/albedo.png", None),
            expected
        );
    }

    #[test]
    fn absolute_paths_stay_as_written() {
        assert_eq!(
            stored_path("/srv/assets/a.png"),
            PathBuf::from("/srv/assets/a.png")
        );
        assert_eq!(
            stored_path("C:\\assets\\a.png"),
            PathBuf::from("C:\\assets\\a.png")
        );
        assert_eq!(
            stored_path("D:/assets/a.png"),
            PathBuf::from("D:/assets/a.png")
        );
        assert_eq!(
            resolve_stored_path("/srv/a.png", Some(Path::new("base"))),
            PathBuf::from("/srv/a.png")
        );
    }
}
