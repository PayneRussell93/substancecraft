# Filament Material Asset

`aiDefaultMat_ogl.filament` is the precompiled Filament glTF PBR material distributed with the `filament-bindings` 0.2.2 example at `examples/winit/aiDefaultMat_ogl.filament`.

The source repository is [rust-filament](https://github.com/EYHN/rust-filament), licensed Apache-2.0. The material package is compiled for Filament's OpenGL backend and exposes `baseColorMap`, `baseColor`, `roughness`, and `metallic` parameters. Regenerate or replace it when upgrading the Filament runtime or material format.