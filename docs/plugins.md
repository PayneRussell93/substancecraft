# Writing plugins

A plugin adds mask generators, layer filters, and export presets to SubstanceCraft without changing or rebuilding it. The desktop app, `substancecraft-cli`, and the MCP server all load the same plugins, so a project that uses one bakes and exports the same everywhere the plugin is installed.

A complete plugin lives in [`examples/plugins/wavy-bands`](../examples/plugins/wavy-bands). It is built outside this repository's Cargo workspace, and CI builds and loads it on Windows, Linux, and macOS (`crates/api/tests/plugins.rs`).

## What a plugin is

A plugin is a folder that holds `plugin.json` and, when the plugin has generators or filters, a WebAssembly module:

```text
wavy-bands/
  plugin.json
  wavy_bands.wasm
```

SubstanceCraft loads plugins from these places, in order:

1. each `--plugins <folder>` given to `substancecraft-cli` (before the command);
2. the folders listed in `SUBSTANCECRAFT_PLUGINS`, separated like `PATH`;
3. the user's plugin folder: `%LOCALAPPDATA%\SubstanceCraft\plugins` on Windows, else `$XDG_CONFIG_HOME/substancecraft/plugins` or `~/.config/substancecraft/plugins`.

A folder that holds `plugin.json` is one plugin. Otherwise each of its subfolders that holds one is. A plugin that fails to load is reported and skipped, and the others still load. `substancecraft-cli plugins` prints what loaded, and why anything did not, as JSON. In the app, **Preferences → Plugins** lists the same information and has a **Reload plugins** button.

## `plugin.json`

```json
{
  "format": 1,
  "id": "com.example.wavy-bands",
  "name": "Wavy Bands (sample)",
  "version": "1.0.0",
  "module": "wavy_bands.wasm",
  "generators": [
    {
      "id": "wavy",
      "name": "Wavy bands",
      "parameters": [
        { "name": "Frequency", "default": 8.0, "min": 1.0, "max": 64.0 }
      ]
    }
  ],
  "filters": [
    { "id": "sepia", "name": "Sepia",
      "parameters": [{ "name": "Amount", "default": 1.0, "min": 0.0, "max": 1.0 }] }
  ],
  "export_presets": []
}
```

- `format` is the manifest version, currently `1`. A plugin written for a newer format is refused with its version named.
- `id` is unique among plugins. It uses lowercase letters, digits, `.`, `-`, and `_`; a reverse domain name is a good choice.
- `module` is the WebAssembly file, relative to the plugin folder. It is needed only for generators and filters.
- `generators` and `filters` each need an `id` that is unique within the plugin and uses the same characters. Projects store a stage as `plugin-id/item-id` (64 characters at most) with its parameter values, so **never reuse an id for something that behaves differently**. Change the id instead.
- Each generator or filter takes up to 8 `parameters`. The app shows each one as a slider from `min` to `max` and starts it at `default`. The values reach the module in declaration order. Slots the plugin does not declare are `0`.
- `export_presets` uses the same shape as a project's custom export presets (`substancecraft-cli presets` lists the built-in names; a project file shows the JSON). A preset's name must not be a built-in preset's name. When a texture set remembers a plugin's preset, the app copies that preset into the project, so the project still exports on machines without the plugin.

Unknown fields are refused, so a typo is caught when the plugin loads rather than silently ignored.

## The module (plugin ABI 1)

Any language that compiles to WebAssembly (`wasm32-unknown-unknown`) can write a module. It must **import nothing**: there is no file, network, clock, or console access. It must export:

| Export | Signature | Purpose |
| --- | --- | --- |
| `memory` | linear memory | where the host writes inputs and reads results |
| `sc_abi_version` | `() -> i32` | returns `1` |
| `sc_alloc` | `(bytes: i32) -> i32` | returns a pointer to `bytes` of writable memory |
| `sc_generate` | `(generator, parameters, samples, count, out: i32) -> i32` | only if the plugin has generators |
| `sc_filter` | `(filter, parameters, texels, count, kind: i32) -> i32` | only if the plugin has filters |

`generator` and `filter` are the item's index in the manifest's `generators` or `filters` list. `parameters` points to 8 `f32`. Both functions return `0` on success. Any other value fails the bake with that code in the message.

Each call works on a whole batch, never one texel at a time. Each call also runs in a fresh instance of the module, so memory from `sc_alloc` never needs freeing and no state carries over between calls. All values are little-endian `f32`.

### Generators

`samples` points to `count` mesh-map samples of 16 floats each:

| Floats | Meaning |
| --- | --- |
| 0–2 | object-space position |
| 3–5 | unit world-space shading normal |
| 6–8 | position within the mesh bounds, `0..1` per axis |
| 9–11 | position relative to the bounds minimum, divided by the bounds diagonal |
| 12 | curvature: `0.5` flat, higher convex, lower concave |
| 13 | ambient occlusion: `1` open, `0` occluded |
| 14 | thickness: `0` thin, `1` thick |
| 15 | `1` when the sample lies on the mesh, `0` outside every UV island |

Write one mask value per sample to `out`. Values are clamped to `0..1`. Samples that are not on the mesh, and values that are not finite, become `0`.

### Filters

`texels` points to `count` texels of 4 floats each: red, green, blue, and alpha, with straight (not premultiplied) alpha. `kind` is `0` for colour (sRGB values) and `1` for grey data such as roughness, where all three components hold the same value. Rewrite the texels in place. Results are clamped to `0..1`, and grey results keep their red component.

A filter must read only the texel it writes. That is what lets SubstanceCraft rebake just the region a brush stroke touches. Normal maps never reach plugin filters.

Plugin generators and filters work in layer masks and layer filter stacks. Material graphs evaluate one texel at a time, so they do not accept plugin stages.

## Limits and failures

A call may grow the module's memory to 512 MiB. It also has an instruction budget of 50 million plus 20,000 per sample or texel, which is far more than a reasonable generator needs, and a plugin that loops forever fails instead of hanging. Large batches are split across threads, each with its own instance.

When a bake needs a plugin that is not installed, or a plugin call fails, the bake stops and names the plugin. SubstanceCraft never exports with the stage silently left out. The project keeps the stage and its values either way, so installing the plugin later restores it.

## Determinism

WebAssembly has no instructions for `sin`, `exp`, or `pow`, so a module carries its own implementations (Rust's standard library for `wasm32` uses a software maths library). The interpreter runs in deterministic mode. Together these mean a plugin computes the same bits on every machine, and exports that use it stay byte-identical across platforms (see PROD-08 in the [feature map](feature-map.md)).

## Building the sample

```sh
rustup target add wasm32-unknown-unknown
cd examples/plugins/wavy-bands
cargo build --release --target wasm32-unknown-unknown
mkdir -p ~/.config/substancecraft/plugins/wavy-bands
cp plugin.json target/wasm32-unknown-unknown/release/wavy_bands.wasm ~/.config/substancecraft/plugins/wavy-bands/
substancecraft-cli plugins
```

## Versioning

The manifest `format` and the module's ABI version (`sc_abi_version`) change only together with a release note. A release that raises either keeps loading plugins written for the previous version.
