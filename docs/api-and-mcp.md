# API and MCP

## Rust API

`substancecraft-api` is the headless facade for deterministic domain operations. It currently exposes:

- `list_primitive_presets()` for the supported starter-mesh names.
- `inspect_primitive_preset(name)` for indexed topology counts, local-space bounds, and UV bounds.

The result types use Serde and plain numeric arrays at the API boundary, so clients do not need to depend on the renderer's math or UI types. Invalid preset names return a typed error. This facade is intentionally narrow today; the planned stable command API will grow to project, import, layer, paint, bake, and export operations as those commands are implemented.

## MCP server

The `substancecraft-mcp` binary uses the official `rmcp` Rust SDK and stdio transport. It currently exposes these read-only tools:

| Tool | Input | Result |
| --- | --- | --- |
| `list_mesh_presets` | None | JSON list of starter meshes. |
| `inspect_mesh_preset` | `name` | JSON topology, bounds, and UV bounds. |

Run it with `cargo run -p substancecraft-mcp`. Configure an MCP client to launch that command from the SubstanceCraft workspace directory. Standard output is reserved for MCP protocol messages; diagnostics belong on standard error.

The current MCP tools inspect generated presets only; they do not read arbitrary files or mutate a project. Future tools should call the same versioned command API as the desktop and CLI, validate all input, and return actionable tool errors rather than panicking.

## Verification

Run `cargo test -p substancecraft-api -p substancecraft-mcp` for the API and MCP tests, `cargo clippy --workspace --all-targets -- -D warnings` for lints, and `cargo fmt --all --check` for formatting.