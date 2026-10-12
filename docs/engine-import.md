# Importing SubstanceCraft exports into engines

Each built-in export preset writes textures laid out for one target. This page lists, per preset, what each file holds and the import settings the engine needs. Two rules apply everywhere:

- **Colour textures are sRGB**: Base Color (albedo, base map, diffuse), Emissive, and Specular. Import them as colour (sRGB on).
- **Everything else is linear data**: packed occlusion/roughness/metallic, normals, height, opacity, glossiness. Import them with sRGB off. Packed maps decoded as sRGB look darker and shade wrongly.

Normals are tangent-space. The OpenGL convention (+Y = +V, "green up") is the default; presets for engines that expect DirectX normals invert green for you.

## Unreal Engine (`Unreal Engine`)

| File | Contents | Import settings |
| --- | --- | --- |
| `T_<set>_BC` | Base Color, Opacity in alpha | Default (sRGB) |
| `T_<set>_ORM` | R occlusion, G roughness, B metallic | Compression *Masks*, sRGB off |
| `T_<set>_N` | Normal, DirectX (green inverted) | Compression *Normalmap* |
| `T_<set>_E` | Emissive (written only when something glows) | Default (sRGB) |

In the material, connect the ORM texture's R, G, and B outputs to Ambient Occlusion, Roughness, and Metallic.

## Unity URP (`Unity (URP Lit)`)

| File | Contents | Import settings |
| --- | --- | --- |
| `<set>_BaseMap` | Base Color, Opacity in alpha | Default (sRGB) |
| `<set>_MetallicSmoothness` | R metallic, A smoothness (1 − roughness) | sRGB off |
| `<set>_Normal` | Normal, OpenGL | Texture Type *Normal map* |
| `<set>_Occlusion` | Occlusion | sRGB off |
| `<set>_Emission` | Emissive (only when something glows) | Default (sRGB) |

On the Lit material set *Smoothness Source* to *Metallic Alpha*.

## Godot 4 (`Godot 4 (StandardMaterial3D)`)

| File | Contents | Import settings |
| --- | --- | --- |
| `<set>_albedo` | Base Color, Opacity in alpha | Default |
| `<set>_orm` | R occlusion, G roughness, B metallic | Default; Godot reads it as data in the ORM slot |
| `<set>_normal` | Normal, OpenGL | Godot detects normal maps assigned to the normal slot |
| `<set>_height` | Height (white is raised) | Default |
| `<set>_emission` | Emissive (only when something glows) | Default |

Use an `ORMMaterial3D` and assign `_orm` to its ORM texture. For parallax, enable *Height* on the material; if the relief looks inverted, toggle the height map's *Flip Texture*.

## glTF 2.0 (`glTF 2.0 (metallic-roughness)` and **Export → glTF…**)

| File | glTF material property |
| --- | --- |
| `<set>_baseColor` | `baseColorTexture` (sRGB, alpha is opacity) |
| `<set>_occlusionRoughnessMetallic` | `metallicRoughnessTexture` (G roughness, B metallic) and `occlusionTexture` (R) |
| `<set>_normal` | `normalTexture` (OpenGL, as glTF requires) |
| `<set>_emissive` | `emissiveTexture` (sRGB; only when something glows) |

**glTF…** in the export panel writes the mesh and these bindings as a complete `.gltf` scene; this preset writes only the textures, for engines that assemble the material themselves.

## Specular/glossiness engines (`Specular/Glossiness`)

| File | Contents | Import settings |
| --- | --- | --- |
| `<set>_Diffuse` | Base Color without the metal (black where fully metallic), Opacity in alpha | sRGB |
| `<set>_Specular` | Reflectance: 4 % grey for dielectrics, Base Color for metals | sRGB |
| `<set>_Glossiness` | 1 − roughness | sRGB off |
| `<set>_Normal` | Normal, OpenGL | Normal map |
| `<set>_AmbientOcclusion` | Occlusion | sRGB off |
| `<set>_Emissive` | Emissive (only when something glows) | sRGB |

Diffuse and specular are mixed from Base Color and Metallic in linear light. For an engine that expects DirectX normals, duplicate the preset and set the normal output's green slot to *inverted*.

## File formats and bit depth

- **PNG** stores 8 bits per channel, or 16 when an output's depth is set to 16-bit in the preset editor. 16-bit outputs carry the unquantized composite, which avoids banding in height maps.
- **TGA** stores 8 bits per channel. Outputs set to 16-bit stay PNG when the export format is TGA.
- **EXR** stores half floats with lossless ZIP compression. Colour channels (Base Color, Emissive, Diffuse, Specular) are decoded to linear light; data channels are stored as they are. A gray output is a single `Y` channel.

An output can name its own format in the preset editor, for example a 16-bit height map inside an otherwise 8-bit preset. A preset can also cap the texture set size it accepts (*Max size*), so that, for example, a mobile profile blocks 4K sets in preflight.
