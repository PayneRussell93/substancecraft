# SubstanceCraft Short Test Plan

- **Version:** 0.1.0
- **Status:** Active
- **Scope:** Core material-project model validation (`REQ-0001`).
- **Environment:** Windows x86_64, Rust stable MSVC toolchain.
- **Approach:** Automated unit tests in `crates/core/src/lib.rs`; run with `cargo test -p substancecraft-core`.
- **Entry criteria:** The workspace resolves and the core crate compiles.
- **Exit criteria:** All listed automated cases pass; formatting and strict Clippy checks pass.

| ID | Preconditions and steps | Expected result | Type / automation | Execution record |
| --- | --- | --- | --- | --- |
| TC-0001 | Construct an empty project and validate it. | Validation succeeds. | Functional / automated | Passed 2026-10-10 |
| TC-0002 | Set an active texture-set ID absent from the project and validate. | Validation returns `ActiveTextureSetMissing`. | Functional / automated | Passed 2026-10-10 |
| TC-0003 | Add two texture sets with the same ID and validate. | Validation returns `DuplicateTextureSetId`. | Functional / automated | Passed 2026-10-10 |
| TC-0004 | Construct zero-sized and over-limit texture resolutions. | Both are rejected as invalid. | Boundary / automated | Passed 2026-10-10 |
| TC-0005 | Add a layer with non-finite opacity and validate. | Validation returns `InvalidLayerOpacity`. | Boundary / automated | Passed 2026-10-10 |
| TC-0006 | Add two layers with the same ID and validate. | Validation returns `DuplicateLayerId`. | Functional / automated | Passed 2026-10-10 |
| TC-0007 | Set a schema version newer than the supported version and validate. | Validation returns `UnsupportedSchemaVersion`. | Compatibility / automated | Passed 2026-10-10 |
| TC-0008 | Generate each built-in mesh preset and validate its vertices, UVs, triangle indices, and expected topology counts. | Cube, sphere, cylinder, and plane produce valid geometry with stable counts. | Functional / automated | Passed 2026-10-10 |
| TC-0009 | List presets, inspect each by name, and compare returned topology and bounds. | All four presets return consistent summaries. | API / automated | Passed 2026-10-10 |
| TC-0010 | Inspect an unsupported preset name. | Return a typed recoverable API error. | API boundary / automated | Passed 2026-10-10 |
| TC-0011 | Construct the MCP server metadata. | Advertise tool capability and the SubstanceCraft server identity. | Integration / automated | Passed 2026-10-10 |
| TC-0012 | Invoke the mesh inspection tool for Sphere. | Return a JSON summary with Sphere name and expected triangle count. | MCP tool / automated | Passed 2026-10-10 |
| TC-0013 | Generate a minimal binary FBX 7.4 triangle, import it, and inspect the returned mesh. | One named mesh, one valid triangle, finite positions, and planar UVs are returned. | Import / automated | Passed 2026-10-10 |
| TC-0014 | Generate a binary FBX triangle with ByPolygonVertex/Direct UVs and import it. | Imported corners preserve the source UV values and do not report generated UVs. | Import / automated | Passed 2026-10-10 |
| TC-0015 | Feed the binary preflight a declared decoded-attribute size above its configured budget. | Reject the input before the FBX parser allocates decoded attributes. | Security / automated | Passed 2026-10-10 |

All 21 workspace tests passed on 2026-10-10. `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` also passed. MCP client interoperability has not yet been manually verified; raw command output is not archived separately.